# ADAPTIVE PROCEDURAL TASK GENERATION FOR HARD-EXPLORATION PROBLEMS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We introduce Adaptive Procedural Task Generation (APT-Gen), an approach to progressively generate a sequence of tasks as curricula to facilitate reinforcement learning in hard-exploration problems. At the heart of our approach, a task generator learns to create tasks from a parameterized task space via a black-box procedural generation module. To enable curriculum learning in the absence of a direct indicator of learning progress, we propose to train the task generator by balancing the agent's performance in the generated tasks and the similarity to the target tasks. Through adversarial training, the task similarity is adaptively estimated by a task discriminator defined on the agent's experiences, allowing the generated tasks to approximate target tasks of unknown parameterization or outside of the predefined task space. Our experiments on grid world and robotic manipulation task domains show that APT-Gen achieves substantially better performance than various existing baselines by generating suitable tasks of rich variations.

# 1 INTRODUCTION

The effectiveness of reinforcement learning (RL) relies on the agent's ability to explore the task environment and collect informative experiences. Given tasks handcrafted with human expertise, RL algorithms have achieved significant progress on solving sequential decision making problems in various domains such as game playing (Badia et al., 2020; Mnih et al., 2015) and robotics (Akkaya et al., 2019; Duan et al., 2016). However, in many hard-exploration problems (Aytar et al., 2018; Paine et al., 2019), such trial-and-error paradigms often suffer from sparse and deceptive rewards, stringent environment constraints, and large state and action spaces.

A plurality of exploration strategies have been developed to encourage the state coverage by an RL agent (Houthooft et al., 2016; Pathak et al., 2017; Burda et al., 2018; Conti et al., 2018). Although successes were achieved in goal-reaching tasks and games of small state spaces, harder tasks often require the agent to complete a series of sub-tasks without any positive feedback until the final mission is accomplished. Naively covering intermediate states can be insufficient for the agent to connect the dots and discover the final solution. In complicated tasks, it could also be difficult to visit diverse states by directly exploring in the given environment (Maillard et al., 2014).

In contrast, recent advances in curriculum learning (Bengio et al., 2009; Graves et al., 2017) aim to utilize similar but easier datasets or tasks to facilitate training. Being applied to RL, these techniques select tasks from a predefined set (Matiisen et al., 2019) or a parameterized space of goals and scenes (Held et al., 2018; Portelas et al., 2019; Racanière et al., 2020) to accelerate the performance improvement on the target task or the entire task space. However, the flexibility of their curricula is often limited to task spaces using low-dimensional parameters, where the search for a suitable task is relatively easy and the similarity between two tasks can be well defined.

In this work, we combat this challenge by generating tasks of rich variations as curricula using procedural content generation (PCG). Developed for automated creation of environments in physics simulations and video games (Summerville et al., 2018; Risi & Togelius, 2019; Cobbe et al., 2019), PCG tools have paved the way for generating diverse tasks of configurable scene layouts, object types, constraints, and objectives. To take advantage of PCG for automated curricula, the key challenge is to measure the learning progress in order to adaptively generate suitable tasks for efficiently learning to solve the target task. In hard-exploration problems, this challenge is intensified since the performance improvement cannot always be directly observed on the target task until it is close to

![](images/0fdf11923329a7ef470d5a373dc7e3aad239032670c62bdc2435a3a9694a12aa.jpg)  
Figure 1: APT-Gen learns to create tasks via a black-box procedural generation module. By jointly training the task generator, the task discriminator, and the policy, suitable tasks are progressively generated to expedite reinforcement learning in hard-exploration problems.

be solved. In addition, the progress in a complex task space is hard to estimate when there does not exist a well-defined measure of task difficulty or similarity. We cannot always expect the agent to thoroughly investigate the task space and learn to solve all tasks therein, especially when the target task has unknown parameterization and the task space has rich variations.

To this end, we introduce Adaptive Procedural Task Generation (APT-Gen), an approach to progressively generate a sequence of tasks to expedite reinforcement learning in hard-exploration problems. As shown in Figure 1, APT-Gen uses a task generator to create tasks via a black-box procedural generation module. Through the interplay between the task generator and the policy, tasks are continuously generated to provide similar but easier scenarios for training the agent. In order to enable curriculum learning in the absence of a direct indicator of learning progress, we propose to train the task generator by balancing the agent's performance in the generated tasks and the task progress score which measures the similarity between the generated tasks and the target task. To encourage the generated tasks to require similar agent's behaviors with the target task, a task discriminator is adversarially trained to estimate the task progress by comparing the agent's experiences collected from both task sources. APT-Gen can thus be trained for target tasks of unknown parameterization or even outside of the task space defined by the procedural generation module, which expands the scope of its application. By jointly training the task generator, the task discriminator, and the policy, APT-Gen is able to adaptively generate suitable tasks from highly configurable task spaces to facilitate the learning process for challenging target tasks.

Our experiments are conducted on various tasks in grid world and robotic manipulation domains. Tasks generated in these domains are parameterized by  $6 \times$  to  $10 \times$  independent variables compared to those in prior work (Wang et al., 2019; 2020; Portelas et al., 2019). In challenging target tasks of sparse rewards and stringent constraints, APT-Gen substantially outperforms existing exploration and curriculum learning baselines by effectively generating new tasks during training. Videos of generated tasks and learned policies are available at https://apt-gen.github.io/.

# 2 RELATED WORK

Hard-Exploration Problems. Many RL algorithms aim to incentivize the agent to visit more diverse and higher-reward states. Methods on intrinsic motivation augment the sparse and deceptive environment rewards with an additional intrinsic reward that encourages curiosity (Pathak et al., 2017; Burda et al., 2018; Raileanu & Rocktäschel, 2020) and state novelty (Conti et al., 2018; Eysenbach et al., 2018). Another family of exploration techniques is derived from an information-theoretical perspective as maximizing information gain of actions (Houthooft et al., 2016; Sun et al., 2011). When human demonstrations are available, they can be used to facilitate an RL agent to visit similar states and transitions as illustrated in the demonstrations (Vecerik et al., 2017; Nair et al., 2018; Zhu et al., 2018). A combination of these techniques have been applied to solve hard-exploration problems in video game domains (Aytar et al., 2018; Ecoffet et al., 2019). However, these methods have focused on learning in relatively simple and fixed environments, and usually

can be ineffective in tasks where explorations are thwarted by stringent environment constraints or naively covering states does not lead to the task success.

Curriculum Learning. Curriculum learning utilizes alternative datasets and tasks to accelerate the learning process of challenging target tasks (Bengio et al., 2009; Graves et al., 2017). To apply curriculum learning to RL, several recent works learn to adaptively select a finite set of easy tasks (Narvekar et al., 2017; Svetlik et al., 2017; Riedmiller et al., 2018; Peng et al., 2018; Czarnecki et al., 2018; Matiisen et al., 2019; Narvekar & Stone, 2019; Lin et al., 2019) or auxiliary rewards (Jaderberg et al., 2017; Shen et al., 2019) hand-designed by human to maximize a progress signal defined on the target task. Parameterized tasks have been used to form a curriculum through the configuration of goals (Forestier et al., 2017; Held et al., 2018; Racanière et al., 2020), initial states (Wöhlke et al., 2020), and reward functions (Gupta et al., 2018; Jabri et al., 2019). OpenAI et al. (2019) and Mehta et al. (2019) propose to actively adjust the hyperparameters in physical simulators to alleviate the domain shift by increasingly adding randomization to physics of the environment. Most of these works are designed for task spaces parameterized by low-dimensional variables, where the task space can be easily explored and the similarity between two tasks is well defined in the parameter space such as goal distance. As a result, these works are only applied to constrained task domains such as reaching a single goal. In contrast, we propose a general framework for highly configurable task spaces with parameters of much higher dimensions. While most of these works focus on parameterizing a single aspect of the task environment, our approach learns to generate new tasks of rich variations with configurable initial state probability, transition probability and reward function. Sukhbaatar et al. (2017) Florensa et al. (2017), and Sukhbaatar et al. (2018) propose to use an adversarial agent to set goals of growing difficulties by reversely traversing the state space from the goal. While this is related to the adversarial training framework in this paper in principle, we apply our framework beyond goal-reaching and reversible task domains.

Procedural Task Generation. An increasing number of task sets have been designed to benchmark and empower reinforcement learning research (Kolve et al., 2017; Xia et al., 2018; Savva et al., 2019; Yu et al., 2019; James et al., 2020). While these handcrafted tasks provide insights and opportunities to learn shareable knowledge through multi-task and meta-learning across different task domains, design and implementation of each task require nontrivial human expertise and heavy engineering. A few works utilize random procedural generation of tasks (Cobbe et al., 2019; Fang et al., 2018; Raileanu & Rocktäschel, 2020; Silver & Chitnis, 2020). However, their generation algorithm is handcrafted with limited configurable features. Evolution strategies (Wang et al., 2019; 2020) and learning-based methods (Gravina et al., 2019; Khalifa et al., 2020; Bontrager & Togelius, 2020) have been proposed to automatically discover diverse games and task environments for training RL agents. Instead of covering the entire task space, our approach aims to generate suitable tasks as curricula in order to solve a set of challenging target tasks of interest.

# 3 ADAPTIVE PROCEDURAL TASK GENERATION

We consider a reinforcement learning problem involving a target task that the policy learns to solve and a parameterized task space that we utilize to generate new tasks. In practice, the parameterized task space can be created by a simulation program or a configurable procedure to set up the environment by a human or a robot in the real world. The target task can be an instance of unknown parameter or a task outside of the task space, as long as there exist shared properties and transferable knowledge between the generated tasks and the target task. This follows the general paradigm of teacher-student curriculum learning (Matiisen et al., 2019; Portelas et al., 2019), while we allow the task space to be parameterized by either continuous or discrete high-dimensional variables and we do not assume the target task has a known parameterization by these variables.

We propose Adaptive Procedural Task Generation (APT-Gen), an approach for progressively generating tasks in highly configurable task spaces as curricula. To enable curriculum learning for hard-exploration problems, our key insight is that the learning progress can be jointly estimated by how well the policy can solve the current generated tasks and how similar the generated tasks are to the target task. Starting with a set of tasks that the policy can easily learn to solve, our approach progressively adapts the generated tasks towards the target task while maintaining their feasibility to the policy. As shown in Figure 1, our approach creates tasks via a black-box procedural generation module by jointly learning the task generator, the task discriminator, and the policy.

# 3.1 PROBLEM FORMULATION

We consider each task as a Markov Decision Process (MDP) denoted by a tuple  $M = (\mathcal{S}, \mathcal{A}, \rho, P, R, \gamma)$  with state space  $\mathcal{S}$ , action space  $\mathcal{A}$ , initial state probability  $\rho$ , transition probability  $P$ , reward function  $R$ , and discount factor  $\gamma$ . The task space  $\mathcal{T}$  defines a finite or infinite number of MDPs of similar designs and properties. We use a multi-dimensional parameter space  $\mathcal{W}$  to represent the inter-task variation of  $\mathcal{T}$ . Given a task parameter  $w \in \mathcal{W}$ , a task  $M(w)$  can be instantiated in the task space by a predefined mapping  $M(\cdot)$ . While a generic task space can be composed of fully configurable MDPs, in this work we assume that all tasks share the same  $\mathcal{S}$ ,  $\mathcal{A}$  and  $\gamma$  such that all policies share the same input and output dimensions. In this case, each  $M(w)$  is defined by a distinct set of  $\rho, P, R$  parameterized by  $w$ . The target task  $\overline{M}$  is either an instance of unknown parameter  $\overline{w} \in \mathcal{W}$  or a task outside of  $\mathcal{T}$  but shares the same  $\mathcal{S}$  and  $\mathcal{A}$ .

Our goal is to learn a policy  $\pi$  to solve the target task  $\overline{M}$ . During training, the curriculum is formed as a sequence of task parameters  $\{w_{i}\}_{i = 1}^{N}$  with index  $i$  for constructing the corresponding sequence of generated tasks  $M(w_{i})$ . The agent collects rollouts by unrolling in both  $\overline{M}$  and  $M(w_{i})$ . Each rollout is denoted as  $\tau$ , which is composed of a sequence of state  $s_t$ , action  $a_{t}$  and reward  $r_t$  at each time step  $t$ . In the generated tasks, the  $w_{i}$  of the source task is recorded alongside with the  $\tau$ . Given a fixed budget of total collected steps in both task domains the objective is to maximize the policy's expected return  $\mathbb{E}[\sum_t\gamma^t r_t]$  in the target task  $\overline{M}$ .

# 3.2 ADAPTIVE GENERATION FOR HARD-EXPLORATION PROBLEMS

The interplay between the policy  $\pi(a|s; \theta_p)$  and the task generator  $G(z; \theta_g)$  is formulated in a teacher-student paradigm (Matiisen et al., 2019), where  $\theta_p$ ,  $\theta_g$  are learnable model parameters and  $z$  is a noise input used in deep generative models (Goodfellow et al., 2014). In contrast to prior work (Held et al., 2018; Matiisen et al., 2019; Portelas et al., 2019; Racanière et al., 2020) which rely on evaluating performance improvements directly on the target task or the entire task space, we propose to define the indicator of learning progress using the expected return  $\mathbb{E}[\sum_t \gamma^t r_t]$  and a task progress  $\eta$  to enable curriculum learning in hard-exploration problems. The expected return measures the policy's performance in the generated tasks sampled by  $G$ . While the task progress  $\eta$  is a continuous score which represents the generated tasks' similarity to the target task. The definition and learning process of  $\eta$  will be detailed in Sec. 3.3. When both the expected return and the task progress reach the maxima, the generated tasks are supposed to be indistinguishable from the target task and  $\pi$  is trained to be the optimal policy for the target task.

The training requires a careful balance between the task progress and the expected return. A highly configurable task space potentially contains a large amount of tasks that are infeasible or of similar difficulties with the target task. If the task distribution of  $G$  moves too fast towards the target task, the policy can quickly be overwhelmed by difficult tasks and lose track of what tasks can be effectively learned. On the contrary, sticking to the tasks that can be solved by the current policy will retard the learning progress and overfit the policy to the easy scenarios. Our approach maximizes the task progress subject to a target minimum expected return  $\delta$  as a chosen hyperparameter. Then the training of the task generator amounts to the optimization problem:

$$
\max  _ {\theta_ {g}} \mathbb {E} _ {w \sim G} [ \eta ], \quad \text {s u b j e c t} \quad \mathbb {E} _ {\tau \sim G, \pi} \left[ \sum_ {t} \gamma^ {t} r _ {t} \right] \geq \delta , \tag {1}
$$

where  $w \sim G$  represents the generation process jointly determined by  $p(z)$  and  $G$  and  $\mathbb{E}_{\tau \sim G,\pi}[\cdot]$  is a shorthand notation for  $\mathbb{E}_{w \sim G}[\mathbb{E}_{\tau \sim M(w),\pi}[\cdot]]$  to represent the expectation over distribution of rollouts. By re-writing Eq. 1 as a Lagrangian under the KKT conditions (Kuhn & Tucker, 2014), we obtain:

$$
\left. \max  _ {\theta_ {g}} \left(\mathbb {E} _ {w \sim G} [ \eta ] + \beta \left(\mathbb {E} _ {\tau \sim G, \pi} \left[ \sum_ {t} \gamma^ {t} r _ {t} \right] - \delta\right)\right), \right. \tag {2}
$$

where  $\beta$  is the KKT multiplier that balances the task feasibility and the task progress.  $\beta$  can have a delayed effect on the optimization problem since  $\pi$  and  $\eta$  in Eq. 2 are learned at the same time. Directly optimizing  $\beta$  can cause the objective to explode. Instead, we adopt an automated procedure (Schulman et al., 2017) to adjust  $\beta$  adaptively when  $\mathbb{E}_{\tau \sim G,\pi}[\sum_t\gamma^t r_t] - \delta$  exceeds a threshold.

Since the gradient in Eq. 2 cannot be directly backpropagated to the task generator, we use two value functions to respectively estimate the two expectation terms similar to that in actor-critic meth

Algorithm 1 Adaptive Procedural Task Generation (APT-Gen)  
Require: target task  $\overline{M}$ , parameterized task space  $M(\cdot)$ , prior probability  $p(z)$ , learning rate  $\alpha$   
1: Initialize parameters  $\theta_p, \theta_g, \theta_d, \theta_1, \theta_2, \beta$   
2: Initialize replay buffers  $\mathcal{D}_g$  and  $\mathcal{D}_{target}$   
3: while not converged do  
4: Sample  $z \sim p(z)$  and create the generated task  $M(w)$  with  $w = G(z; \theta_g)$   
5: Collect a rollout  $\tau_g$  in  $M(w)$  using  $\pi(a|s; \theta_p)$  and store  $w$  and  $\tau_g$  in  $\mathcal{D}_g$   
6: Collect a rollout  $\tau_{target}$  in  $\overline{M}$  using  $\pi(a|s; \theta_p)$  and store  $\tau_{target}$  in  $\mathcal{D}_{target}$   
7: Update  $\theta_d \gets \theta_d - \alpha \nabla_{\theta_d} \mathcal{L}_d(\theta_d, \mathcal{D}_{target}, \mathcal{D}_g)$   
8: Update  $\theta_1 \gets \theta_1 - \alpha \nabla_{\theta_1} \mathbb{E}_{w, \tau_g \sim \mathcal{D}_g}[(V_1(w; \theta_1) - D(\tau_g; \theta_d))^2]$   
9: Update  $\theta_2 \gets \theta_2 - \alpha \nabla_{\theta_2} \mathbb{E}_{w, \tau_g \sim \mathcal{D}_g}[(V_2(w; \theta_2) - \sum_t \gamma^t r_t)^2]$   
10: Update  $\theta_g \gets \theta_g - \alpha \nabla_{\theta_g} \mathcal{L}_g(\theta_g, \theta_1, \theta_2, \beta)$   
11: Update  $\beta$  as described in Sec. 3.2  
12: Update  $\theta_p$  using the RL algorithm with sampled batches from  $\mathcal{D}_g$  and  $\mathcal{D}_{target}$   
13: end while

ods (Konda & Tsitsiklis, 2000). By taking the input task parameter  $w$ , the progress value function  $V_{1}(w;\theta_{1})$  estimates the task progress  $\eta$  and the return value function  $V_{2}(w;\theta_{2})$  estimates the expected return  $\mathbb{E}_{\tau \sim G,\pi}[\sum_{t}\gamma^{t}r_{t}]$ , where  $\theta_{1}$  and  $\theta_{2}$  are learnable model parameters. The two value functions are trained to fit the two empirical values of the two terms with respect to  $\theta_{1}$  and  $\theta_{2}$  using rollouts collected in the generated tasks. The training of the task generator becomes learning  $\theta_{g}$  to maximize the task generator loss:

$$
\mathcal {L} _ {g} \left(\theta_ {g}, \theta_ {1}, \theta_ {2}, \beta\right) = \mathbb {E} _ {z \sim p (z)} \left[ V _ {1} \left(G \left(z; \theta_ {g}\right); \theta_ {1}\right) + \beta \left(V _ {2} \left(G \left(z; \theta_ {g}\right); \theta_ {2}\right) - \delta\right) \right]. \tag {3}
$$

# 3.3 ADVERSARIAL TRAINING OF TASK PROGRESS

The goal of the task progress  $\eta$  is to guide the task generator  $G$  to generate tasks similar to the target task. Since the difficulty level and the task similarity cannot be defined by an objective metric in many complex task domains, we argue that  $\eta$  needs to jointly adapt with  $G$  and  $\pi$  when the task distribution and the policy constantly evolve over the course of training. Ideally,  $\eta$  should satisfy two requirements: First, when the maximum  $\eta$  is achieved at convergence, a generated task  $M(w)$  and the target task  $\overline{M}$  should be indistinguishable from the perspective of the policy  $\pi$ . Second, since a small change in an ill-posed task parameter space can completely alter the required agent's behaviors to solve the task,  $\eta$  needs to provide a smooth signal to adapt  $G$  in the task space.

To this end, we estimate  $\eta$  using a task discriminator  $D(\tau; \theta_d)$  defined on the agent's experiences in the task environment, where  $\theta_d$  is the learnable model parameter. It takes  $\tau$  as input and learns to estimate the probability of the task  $M$  being the target task  $\overline{M}$  conditioned on the rollout  $\tau$  induced by the policy  $\pi$ . The task progress of the task parameter  $w$  can be defined as  $\mathbb{E}_{\tau \sim M(w), \pi}[D(\tau; \theta_d)]$ . In this way,  $D$  forms an adversarial training framework (Goodfellow et al., 2014) against  $G$  and  $\pi$ , which jointly determine the likelihood of  $\tau$ .

The task discriminator is required to comprehensively compare the given task with the target task in APT-Gen. Unlike prior work which aim to discriminate policies (Ho & Ermon, 2016) and physics parameters (Mehta et al., 2019),  $D$  computes the prediction score by taking the overall MDP definition into account. Therefore,  $D$  is designed to separately encode the initial state  $s_1$  and each transition  $(s_t, a_t, r_t, s_{t+1})$  of step  $t$  to discriminate the initial state probability  $\rho$ , the transition probability  $P$  and the reward function  $R$  respectively. The prediction is computed using a pooling function across all encoded features. The implementation details of  $D$  are described in Appendix B.

To train the task generator, we collect rollouts from generated tasks as  $\tau_{g}$  and the target task as  $\tau_{target}$ , stored in two replay buffers  $\mathcal{D}_g$  and  $\mathcal{D}_{target}$  respectively. The training of  $D(\tau; \theta_d)$  is conducted by minimizing a discriminator loss (Goodfellow et al., 2014) to classify the task sources of the collected rollouts:

$$
\mathcal {L} _ {d} \left(\theta_ {d}, \mathcal {D} _ {\text {t a r g e t}}, \mathcal {D} _ {g}\right) = - \mathbb {E} _ {\tau_ {\text {t a r g e t}} \sim \mathcal {D} _ {\text {t a r g e t}}} \left[ \log \left(D \left(\tau_ {\text {t a r g e t}}; \theta_ {d}\right)\right) \right] - \mathbb {E} _ {\tau_ {g} \sim \mathcal {D} _ {g}} \left[ 1 - \log \left(D \left(\tau_ {g}; \theta_ {d}\right)\right) \right]. \tag {4}
$$

In principle, to learn a  $G$  that produces the exact MDP definition of  $\overline{M}$ , we would require  $\overline{M}$  to be an instance of the task space  $\mathcal{T}$  and the training data to be collected by arbitrary  $\pi$  to fully investigate

differences in the two task environments. However, this could be neither computationally practical nor necessary. Given that our goal is to find an optimal policy  $\pi^{*}$  to solve the target task, we only need  $\overline{M}$  and  $M(w)$  to be indistinguishable from the perspective of the policy. In practice, the rollouts are collected using the updated  $\pi(a|s, w; \theta_p)$  with epsilon-greedy exploration (Sutton & Barto, 2011). One could also encourage explorations (Pathak et al., 2017) in the policy learning to efficiently distinguish the two tasks, which we leave out of the scope of this work.

The pseudocode of the algorithm is outlined in Algorithm 1. The training alternates among updates of the policy, the task discriminator, and the task generator. New rollouts are continuously collected from both the target task and the generated tasks using the updated  $\pi$ . In this work, we equally collect experiences from the two sources to train the policy, while a smarter strategy of choosing between task sources can be further investigated in the future work.

# 4 EXPERIMENTS

The goal of our experimental evaluation is to answer the following questions: 1) Can APT-Gen facilitate reinforcement learning in hard-exploration problems? 2) What tasks can be generated by APT-Gen for a target task during training? 3) Can APT-Gen be applied to target tasks outside of the task space predefined by the procedural generation module?

# 4.1 TASKS

The experiments are conducted in two configurable task spaces: Grid-World and Manipulation. The two task spaces are parameterized by 74 and 31 independent variables respectively, which have much higher dimensions than those in prior work (Held et al., 2018; Wang et al., 2019; 2020; Portelas et al., 2019; Racanière et al., 2020). Each task space contains various tasks that share the same state and action spaces but different designs of environments and reward functions. The tasks from these task spaces can have stringent penalties and constraints such as lava regions and pitfalls which make it hard for the agent to succeed or survive. In contrast to many handcrafted hard-exploration tasks in prior work (Salimans & Chen, 2018) which provide sub-task rewards

(e.g. finding a key and opening a door), our tasks only provide a sparse positive reward when the final task is accomplished, which introduces extra challenges for the agent. As shown in Figure 2, we design three target tasks of different complexities in each task space for evaluation. Details of the task design and the task parameterization can be found in Appendix A.

![](images/dfad472bc3a45665f1bc53b007c2a2bb0295f33b654b5088c2f9995c3c410c32.jpg)  
Grid-World-A

![](images/7db5e0b6ba65e321900680f9a91a359c69a99c83ed5130af16277bcebc5c2896.jpg)  
Grid-World-B

![](images/9cf1bd79d4dc466941d7d4bc875ff2766d8370781bcfd3b2875b19be7c825a83.jpg)  
Grid-World-C

![](images/004474bd3da14e2d06695286c151d5a1164a6985dd56c5a4deeb40c56cc45fd2.jpg)  
Manipulation-A  
Figure 2: Target tasks in the two domains.

![](images/26122b0ac337d302ac187139ca73d89e109b05a58e9a8434e7592e8366512dec.jpg)  
Manipulation-B

![](images/9ce0347bb59180771e5ec879222a1c7c99faf364e34288cd54c5eb1d26a691c3.jpg)  
Manipulation-C

# 4.2 QUANTITATIVE RESULTS

We evaluate the performance of the agent in target tasks using different methods. All methods are trained with a fixed budget of total steps collected by the agent. In Appendix B, we provide implementation details, hyperparameters, training and evaluation protocols.

Baselines. We compare with the following baselines. DQN (Mnih et al., 2013; Hessel et al., 2017) directly applies Q-learning in the target task. ICM (Pathak et al., 2017), RND (Burda et al., 2018), and RIDE (Raileanu & Rocktäschel, 2020) adversarially learn intrinsic motivations to encourage exploration in the state space. Random uniformly samples from the task parameter space to create tasks. ALP-GMM (Portelas et al., 2019) and GoalGAN (Held et al., 2018) use Gaussian Mixture Models and GAN to sample tasks as curricula without mechanisms to estimate the task distance and handle complex task spaces. To have a fair comparison, we use the same architecture for the corresponding components and search for the optimal hyperparameters for each method.

![](images/9bcc89a55de72b43cafd801745fa107a8613072fa69e293bf58f17b771311df9.jpg)

![](images/c124ea14bc6512a14c14ab24491b45c1cb2abfc313c04297b338828b313f9efb.jpg)

![](images/1eb059e26727af4805b6c23b4c9e28ffa460232d2417fc1cce945076a28cc88d.jpg)

![](images/91868a39c526959243ed3f68f5f7856a2a46f5c0bbdf5919d179ce9463445454.jpg)  
Figure 3: Quantitative results of the performance of the agent in the target tasks.

![](images/e059c53ae39fa40b7087dd2510f7433b7f3866bf1c4f3def09e703c9dfa96e9c.jpg)

![](images/8a9238669fe203f82808f4ed3bbdb252dccb5bb74974f205381c0b0011f2cced.jpg)

Comparative Analysis. In Figure 3, we present the agent's performance in the target tasks by using different methods. To have fair comparisons, the x-axes of APT-Gen and curriculum learning baselines (Random, GoalGAN, ALP-GMM) indicate the total steps collected from target and generated tasks, while x-axes of other baselines (DQN, ICM, RND, RIDE) indicate the steps collected from only the target task. In all scenarios, our approach achieves superior performance comparing with baseline methods. In Grid-World tasks, APT-Gen successfully trains the agent to find keys in separate locations and access different rooms in the right sequential order. In the Manipulation tasks, APT-Gen enables the agent to solve the puzzle by moving around the obstacles in the correct order without causing collisions. Especially, in Manipulation-C, our agent develops an effective strategy that first moves away the target object away to yield the path for the obstacle to leave and then pushing it back towards the goal to complete the task.

Most baseline methods fail in hard tasks that require sequential problem solving over a longer horizon, although some can achieve comparable results in easier scenarios (i.e. when there is only one room and the environment is mostly empty). ICM, RND, and RIDE demonstrate effective explorations when the environment is relatively simple, but the agent is often thwarted by the penalties caused by environment constraints, before the exploration leads it to the promising states behind those constraints. Without any reward shaping for sub-tasks, naively reaching to the intermediate states (e.g. finding the keys) does not yield any immediate reward unless the goal is reached at the end of the same episode. Without mechanisms to estimate the task similarity and to handle complex task spaces, curriculum learning baselines like ALP-GMM and GoalGAN fail to produce useful tasks that share similar challenges with the target tasks. While randomly sampling tasks can serve as a strong baseline in simpler task spaces as demonstrated by Riedmiller et al. (2018) and Portelas et al. (2019), we found it confuses the policy in high-dimensional task parameter spaces since most random tasks either have misleading goals or are completely infeasible.

Out-of-Space Task. To demonstrate APT-Gen's performance in target tasks that are outside of the predefined task space, we train the model to solve a different robotic manipulation task while still generating tasks from the task space defined in Sec. 4.1. The target task shares the same state and action spaces with the predefined task space, but the table has a different shape and the a variety of static objects are placed on the table as environment constraints. As shown in Figure 4, APT-Gen efficiently learns to solve the out-of-space task while baseline methods take much more steps or completely fail to learn. Qualitative results of out-of-space tasks will be discussed in Sec. 4.4.

# 4.3 ABLATION STUDY

We conduct ablation study on the target task of Manipulation-C and analyze the effect of the indicator of learning progress. As shown in Figure 5, the performance degrades when using only either expected return or task progress as the learning progress. The generation often adapts too fast towards to the target task when only counting on the task progress, although easier tasks can still emerge during the adaptation. When generating tasks only in response to the expected return, the policy is overwhelmed by easy tasks, which retards the learning progress of the target task.

![](images/a95d859d8c112bdad8e36a7310ad195ad658867463b7f7b41881901bd247524a.jpg)  
Figure 4: Results on task that is out of the predefined task space.

![](images/560a577be7c8aec0b74aa0a720cdf5daf0a7dd399c71c8adb7754dfa91cc0911.jpg)

![](images/b720f04123fdeec0e0fe3c78ebf885501cdad0b3745e7ae88fe49a93e4e066bf.jpg)

![](images/afb6314193fca84cc399ef1dd07314aa96f2befc3bc21ad8cec76c6d26f46848.jpg)  
Figure 5: Results of ablation study in Manipulation-C.

![](images/40e5ffcbcd7f3e7eddb1284de2ace19ca1976c40954456db32dfdeab5f401640.jpg)  
Figure 6: Progression of the generated tasks for various target tasks in the two task domains and the out-of-space task.

# 4.4 PROGRESSION OF GENERATED TASKS

We present qualitative results of the generated tasks in Figure 6. Each row shows three generated tasks and the target task (marked by green borderlines) with the number of collected environment steps and the task name shown on the upper right of the images.

When learning for Grid-World-A, the task generator first creates easy tasks in which the goal (green tile) is close to the starting position of the agent (red triangle) with few obstacles in between. Between 15k and 30k steps, the task generator gradually shifts the goal to the bottom right corner as in the target task. At the same time, walls (grey tiles) are created to form rooms enclosing the goal. At around 45k steps, the door is placed on the wall to lock the room and the key is placed in a further location in the labyrinth. The agent learns to grab the key and open the door in the target task after learning to solve the generated task, since the solutions now share a similar routine. In Manipulation-C, generated tasks start with a clear table surface and a small distance between the target object (blue can) and the goal (cyan circle). As the agent learns to tackle such easy scenarios, a green can is placed in between as obstacle while the goal grows larger to make sure the agent can still complete the task. At 60K steps, the environment further morphs towards the target task as more obstacles being added to the scene and the goal shrinking to the correct size.

In the out-of-space task, although the more complicated table and objects cannot be generated by the procedural generation module of limited capabilities, APT-Gen gradually learns to outline the scene of the target task by utilizing the available elements such as cuboids and empty holes. By interacting with the environment and comparing experiences in both task sources, APT-Gen trains the policy to solve the out-of-space task by approximating the challenges in the target task.

# 5 CONCLUSION

To expedite reinforcement learning in hard-exploration problems, we present Adaptive Procedural Task Generation (APT-Gen) to generate suitable tasks via black-box procedural generation modules as curricula. By jointly training the task generator, the task discriminator, and the policy, APT-Gen achieves superior performances to existing exploration and curriculum learning baselines in various target tasks in grid world and robotic manipulation domains. By adversarially training the task discriminator to estimate the similarity between the target task and generated tasks, APT-Gen demonstrates to be effective for target tasks of unknown parameterization and out of the predefined task spaces, which expands its potential use case. We hope this work could encourage more endeavors in utilizing procedural content generation for reinforcement learning.

# REFERENCES

Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, et al. Solving rubik's cube with a robot hand. arXiv preprint arXiv:1910.07113, 2019.  
Yusuf Aytar, Tobias Pfaff, David Budden, Thomas Paine, Ziyu Wang, and Nando de Freitas. Playing hard exploration games by watching youtube. In Advances in Neural Information Processing Systems, 2018.  
Adria Puigdomenech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, and Charles Blundell. Agent57: Outperforming the atari human benchmark. arXiv preprint arXiv:2003.13350, 2020.  
Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In International Conference on Machine Learning, 2009.  
Philip Bontrager and J. Togelius. Fully differentiable procedural content generation through generative playing networks. ArXiv, abs/2002.05259, 2020.  
Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018.  
Maxime Chevalier-Boisvert, Lucas Willems, and Suman Pal. Minimalistic gridworld environment for operai gym. https://github.com/maximecb/gym-minigrid, 2018.  
Karl Cobbe, Christopher Hesse, Jacob Hilton, and John Schulman. Leveraging procedural generation to benchmark reinforcement learning. *ArXiv*, abs/1912.01588, 2019.  
Edoardo Conti, Vashisht Madhavan, Felipe Petroski Such, Joel Lehman, Kenneth Stanley, and Jeff Clune. Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents. In Advances in Neural Information Processing Systems, pp. 5027-5038, 2018.  
Erwin Coumans and Yunfei Bai. Pybullet, a python module for physics simulation for games, robotics and machine learning. http://pybullet.org, 2016-2019.  
Wojciech Czarnecki, Siddhant M. Jayakumar, Max Jaderberg, Leonard Hasenclever, Yee Whye Teh, Nicolas Manfred Otto Heess, Simon Osindero, and Razvan Pascanu. Mix&match - agent curricula for reinforcement learning. In International Conference on Machine Learning, 2018.  
Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning, pp. 1329-1338, 2016.  
Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O Stanley, and Jeff Clune. Go-exlore: a new approach for hard-exploration problems. arXiv preprint arXiv:1901.10995, 2019.  
Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function. arXiv preprint arXiv:1802.06070, 2018.  
Kuan Fang, Yuke Zhu, Animesh Garg, Andrey Kuryenkov, Viraj Mehta, Li Fei-Fei, and Silvio Savarese. Learning task-oriented grasping for tool manipulation from simulated self-supervision. Robotics: Science and Systems (RSS), 2018.  
Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, and Pieter Abbeel. Reverse curriculum generation for reinforcement learning. In Conference on Robot Learning, 2017.  
Sebastien Forestier, Yoan Mollard, and Pierre-Yves Oudeyer. Intrinsically motivated goal exploration processes with automatic curriculum learning. *ArXiv*, abs/1708.02190, 2017.  
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, 2014.

Alex Graves, Marc G. Bellemare, Jacob Menick, Rémi Munos, and Koray Kavukcuoglu. Automated curriculum learning for neural networks. ArXiv, abs/1704.03003, 2017.  
Daniele Gravina, A. Khalifa, Antonios Liapis, J. Togelius, and Georgios N. Yannakakis. Procedural content generation through quality diversity. IEEE Conference on Games (CoG, 2019.  
Abhishek Gupta, Benjamin Eysenbach, Chelsea Finn, and Sergey Levine. Unsupervised meta-learning for reinforcement learning. ArXiv, abs/1806.04640, 2018.  
David Held, Xinyang Geng, Carlos Florensa, and Pieter Abbeel. Automatic goal generation for reinforcement learning agents. In International Conference on Machine Learning, 2018.  
Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. arXiv preprint arXiv:1710.02298, 2017.  
Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. In Advances in Neural Information Processing Systems, 2016.  
Rein Houthooft, Xi Chen, Xi Chen, Yan Duan, John Schulman, Filip De Turck, and Pieter Abbeel. Vime: Variational information maximizing exploration. In Advances in Neural Information Processing Systems, pp. 1109-1117, 2016.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Allan Jabri, Kyle Hsu, Ben Eysenbach, Abhishek Gupta, Alexei A. Efros, Sergey Levine, and Chelsea Finn. Unsupervised curricula for visual meta-reinforcement learning. *ArXiv*, abs/1912.04226, 2019.  
Max Jaderberg, Volodymyr Mnih, Wojciech Czarnecki, Tom Schaul, Joel Z. Leibo, David Silver, and Koray Kavukcuoglu. Reinforcement learning with unsupervised auxiliary tasks. *ArXiv*, abs/1611.05397, 2017.  
Stephen W. James, Z. Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 5:3019-3026, 2020.  
A. Khalifa, Philip Bontrager, Sam Earle, and J. Togelius. Pcgrl: Procedural content generation via reinforcement learning. ArXiv, abs/2001.09212, 2020.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
James N Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 2017.  
Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Daniel Gordon, Yuke Zhu, Abhinav Gupta, and Ali Farhadi. AI2-THOR: An Interactive 3D Environment for Visual AI. arXiv, 2017.  
Vijay R Konda and John N Tsitsiklis. Actor-critic algorithms. In Advances in Neural Information Processing Systems, 2000.  
Harold W Kuhn and Albert W Tucker. Nonlinear programming. In *Traces and emergence of nonlinear programming*, pp. 247-258. Springer, 2014.  
Xingyu Lin, Harjatin Singh Baweja, George Kantor, and David Held. Adaptive auxiliary task weighting for reinforcement learning. In Advances in Neural Information Processing Systems, 2019.  
Odalric-Ambrym Maillard, Timothy A Mann, and Shie Mannor. "How hard is my mdp?" the distribution-norm to the rescue". In Advances in Neural Information Processing Systems, 2014.

Tambet Matiisen, Avital Oliver, Taco Cohen, and John Schulman. Teacher-student curriculum learning. IEEE transactions on neural networks and learning systems, 2019.  
Bhairav Mehta, Manfred Diaz, Florian Golemo, Christopher Joseph Pal, and Liam Paull. Active domain randomization. ArXiv, abs/1904.04762, 2019.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin A. Riedmiller. Playing atari with deep reinforcement learning. *ArXiv*, abs/1312.5602, 2013.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.  
Ashvin Nair, Bob McGrew, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Overcoming exploration in reinforcement learning with demonstrations. In IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018.  
S. Narvekar, J. Sinapov, and P. Stone. Autonomous task sequencing for customized curriculum design in reinforcement learning. In *IJCAI*, 2017.  
Sanmit Narvekar and Peter Stone. Learning curriculum policies for reinforcement learning. In Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems. International Foundation for Autonomous Agents and Multiagent Systems, 2019.  
OpenAI, Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, Jonas Schneider, Nikolas Tezak, Jadwiga Tworek, Peter Welinder, Lilian Weng, Qi-Ming Yuan, Wojciech Zaremba, and Lefei Zhang. Solving rubik's cube with a robot hand. ArXiv, abs/1910.07113, 2019.  
Tom Le Paine, Caglar Gulcehre, Bobak Shahriari, Misha Denil, Matt Hoffman, Hubert Soyer, Richard Tanburn, Steven Kapturowski, Neil Rabinowitz, Duncan Williams, et al. Making efficient use of demonstrations to solve hard exploration problems. arXiv preprint arXiv:1909.01387, 2019.  
Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International Conference on Machine Learning, 2017.  
B. Peng, J. MacGlashan, R. Loftin, M. Littman, D. Roberts, and Matthew E. Taylor. Curriculum design for machine learners in sequential decision tasks. IEEE Transactions on Emerging Topics in Computational Intelligence, 2:268-277, 2018.  
R'emy Portelas, Cédric Colas, Katja Hofmann, and Pierre-Yves Oudeyer. Teacher algorithms for curriculum learning of deep rl in continuously parameterized environments. *ArXiv*, abs/1910.07224, 2019.  
Sebastien Racanière, Andrew Kyle Lampinen, Adam Santoro, David P. Reichert, Vlad Firoiu, and Timothy P. Lillicrap. Automated curriculum generation through setter-solver interactions. In ICLR, 2020.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
Roberta Raileanu and Tim Rocktäschel. Ride: Rewarding impact-driven exploration for procedurally-generated environments. ArXiv, abs/2002.12292, 2020.  
Martin Riedmiller, Roland Hafner, Thomas Lampe, Michael Neunert, Jonas Degrave, Tom Van de Wiele, Volodymyr Mnih, Nicolas Heess, and Jost Tobias Springenberg. Learning by playing-solving sparse reward tasks from scratch. arXiv preprint arXiv:1802.10567, 2018.  
Sebastian Risi and Julian Togelius. Procedural content generation: From automatically generating game levels to increasing generality in machine learning. arXiv preprint arXiv:1911.13071, 2019.

Tim Salimans and Richard Chen. Learning montezuma's revenge from a single demonstration. arXiv preprint arXiv:1812.03381, 2018.  
Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat: A platform for embodied air research. In Proceedings of the IEEE International Conference on Computer Vision, pp. 9339-9347, 2019.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. ArXiv, abs/1707.06347, 2017.  
William B Shen, Danfei Xu, Yuke Zhu, Leonidas J Guibas, Li Fei-Fei, and Silvio Savarese. Situational fusion of visual representation for visual navigation. In Proceedings of the IEEE International Conference on Computer Vision, pp. 2881-2890, 2019.  
Tom Silver and Rohan Chitnis. Pddlgym: Gym environments from pddl problems. ArXiv, abs/2002.06432, 2020.  
Sainbayar Sukhbaatar, Ilya Kostrikov, Arthur Szlam, and Rob Fergus. Intrinsic motivation and automatic curricula via asymmetric self-play. ArXiv, abs/1703.05407, 2017.  
Sainbayar Sukhbaatar, Emily L. Denton, Arthur Szlam, and R. Fergus. Learning goal embeddings via self-play for hierarchical reinforcement learning. *ArXiv*, abs/1811.09083, 2018.  
Adam Summerville, Sam Snodgrass, Matthew Guzdial, Christoffer Holmgård, Amy K Hoover, Aaron Isaksen, Andy Nealen, and Julian Togelius. Procedural content generation via machine learning (pcgml). IEEE Transactions on Games, 10(3):257-270, 2018.  
Yi Sun, Faustino Gomez, and Jürgen Schmidhuber. Planning to be surprised: Optimal bayesian exploration in dynamic environments. In International Conference on Artificial General Intelligence. Springer, 2011.  
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. Cambridge, MA: MIT Press, 2011.  
M. Svetlik, Matteo Leonetti, J. Sinapov, Rishi Shah, Nick Walker, and P. Stone. Automatic curriculum graph generation for reinforcement learning agents. In AAAI, 2017.  
Mel Vecerik, Todd Hester, Jonathan Scholz, Fumin Wang, Olivier Pietquin, Bilal Piot, Nicolas Heess, Thomas Rothörl, Thomas Lampe, and Martin Riedmiller. Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards. arXiv preprint arXiv:1707.08817, 2017.  
Renmin Wang, J. Lehman, A. Rawal, Jiale Zhi, Yulun Li, J. Clune, and K. Stanley. Enhanced poet: Open-ended reinforcement learning through unbounded invention of learning challenges and their solutions. *ArXiv*, abs/2003.08536, 2020.  
Rui Wang, Joel Lehman, Jeff Clune, and Kenneth O. Stanley. Poet: open-ended coevolution of environments and their optimized solutions. Proceedings of the Genetic and Evolutionary Computation Conference, 2019.  
Jan Wöhlke, Felix Schmitt, and Herke van Hoof. A performance-based start state curriculum framework for reinforcement learning. In Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems, pp. 1503-1511, 2020.  
Fei Xia, Amir R Zamir, Zhiyang He, Alexander Sax, Jitendra Malik, and Silvio Savarese. Gibson env: Real-world perception for embodied agents. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9068-9079, 2018.  
Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. arXiv preprint arXiv:1910.10897, 2019.  
Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool, János Kramár, Raia Hadsell, Nando de Freitas, et al. Reinforcement and imitation learning for diverse visuomotor skills. arXiv preprint arXiv:1802.09564, 2018.
