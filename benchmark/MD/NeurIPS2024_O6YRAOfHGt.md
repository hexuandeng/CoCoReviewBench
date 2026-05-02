# Enhancing the Hierarchical Environment Design via Generative Trajectory Modeling

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Unsupervised Environment Design (UED) is a paradigm that automatically generates a curriculum of training environments, enabling agents trained in these environments to develop general capabilities, i.e., achieving good zero-shot transfer performance. However, existing UED approaches focus primarily on the random generation of environments for open-ended agent training. This is impractical in resource-limited scenarios where there is a constraint on the number of environments that can be generated. In this paper, we introduce a hierarchical MDP framework for environment design under resource constraints. It consists of an upper-level RL teacher agent that generates suitable training environments for a lower-level student agent. The RL teacher can leverage previously discovered environment structures and generate environments at the frontier of the student's capabilities by observing the student policy's representation. Additionally, to alleviate the time-consuming process of collecting the experience of the upper-level teacher, we utilize recent advances in generative modeling to synthesize a trajectory dataset for training the teacher agent. Our method significantly reduces the resource-intensive interactions between agents and environments, and empirical experiments across various domains demonstrate the effectiveness of our approach.

# 1 Introduction

The advances of reinforcement learning (RL) [17] have promoted research into the problem of training autonomous agents that are capable of accomplishing complex tasks. One interesting, yet underexplored, area is training agents to perform well in unseen environments, a concept referred to as zero-shot transfer performance. To this end, Unsupervised Environment Design (UED) [3] has emerged as a promising paradigm to address this problem. The objective of UED is to automatically generate environments in a curriculum-based manner, and training agents in these sequentially generated environments can equip agents with general capabilities, enabling agents to learn robust and adaptive behaviors that can be transferred to new scenarios without explicit exposure during training.

Existing approaches in UED primarily focus on building an adaptive curriculum for the environment generation process to train the generally capable agent. Dennis et al. [3] formalize the problem of finding adaptive curricula through a game involving an adversarial environment generator (teacher agent), an antagonist agent (expert agent), and the protagonist agent (student agent). The RL-based teacher is designed to generate environments that maximize regret, defined as the difference between the protagonist and antagonist agent's expected rewards. They show that these agents will reach a Nash Equilibrium where the student agent learns the minimax regret policy. However, since the teacher agent adapts solely based on the regret feedback, it is inherently difficult to adapt to student

policy changes. Meanwhile, training such an RL-based teacher remains a challenge because of the high computational cost of training an expert antagonist agent for each environment.

In contrast, domain randomization [19] based approaches circumvent the overhead of developing an RL teacher by training agents in randomly generated environments, resulting in good empirical performances. Building upon this, Jiang et al. [7] introduce an emergent curriculum by sampling randomly generated environments with high regret value to train the agent. Parker-Holder et al. [10] then propose the adaptive curricula by manually designing a principled, regret-based curriculum, which involves generating random environments with increasing complexity. While these domain randomization-based algorithms have demonstrated good zero-shot transfer performance, they face limitations in efficiently exploring large environment design spaces and exploiting the inherent structure of previously discovered environments. Moreover, existing UED approaches typically rely on open-ended learning, necessitating a long training horizon, which is unrealistic in the real world due to resource constraints. Our goal is to develop a teacher policy capable of generating environments that are perfectly matched to the current skill levels of student agents, thereby allowing students to achieve optimal general capability within a strict budget for the number of environments generated and within a shorter training time horizon.

In this paper, we address these challenges by introducing a novel, adaptive environment design framework. The core idea involves using a hierarchical Markov Decision Process (MDP) to simultaneously formulate the evolution of an upper-level teacher agent, tasked with generating suitable environments to train the lower-level student agent to achieve general capabilities. To accurately guide the generation of environments at the frontier of the student agent's current capabilities, we propose approximating the student agent's policy/capability by its performances across a set of diverse evaluation environments, which acts as the state abstraction for the teacher's decision-making process. The transitions in the teacher's state represent the trajectories of the student agent's capability after training in the generated environment. However, collecting experience for the upper-level teacher agent is slow and resource-intensive, since each upper-level MDP transition evolves a complete training cycle of the student agent on the generated environment. To accelerate the collection of upper-level MDP experiences, we utilize advances in diffusion models that can generate new data points capturing complex distribution properties, such as skewness and multi-modality, exhibited in the collected dataset [11]. Specifically, we employ diffusion probabilistic model [15, 6] to learn the evolution trajectory of student policy/capability and generate synthetic experiences to enhance the training efficiency of the teacher agent. Our method, called Synthetically-enhanced Hierarchical Environment Design (SHED), automatically generates increasingly complex environments suited to the current capabilities of student agents.

In summary, we make the following contributions:

- We develop a novel hierarchical MDP framework for UED that introduces a straightforward method to represent the current capability level of the student agent.  
- We introduce SHED, which utilizes diffusion-based techniques to generate synthetic experiences. This method can accelerate the training of the off-policy teacher agent.  
- We demonstrate that our method outperforms existing UED approaches (i.e., achieving a better general capability under resource constraints) in different task domains.

# 2 Preliminaries

In this section, we provide an overview of two main research areas upon which our work is based.

# 2.1 Unsupervised Environment Design

The objective of UED is to generate a sequence of environments that effectively train the student agent to achieve a general capability. Dennis et al. [3] first model UED with an Underspecified Partially Observable Markov Decision Process (UPOMDP), which is a tuple

$$
\mathcal {M} = <   A, O, \Theta , S ^ {\mathcal {M}}, \mathcal {P} ^ {\mathcal {M}}, \mathcal {I} ^ {\mathcal {M}}, \mathcal {R} ^ {\mathcal {M}}, \gamma >
$$

The UPOMDP has a set  $\Theta$  representing the free parameters of the environments, which are determined by the teacher agent and can be distinct to generate the next new environment. Further, these parameters are incorporated into the environment-dependent transition function  $\mathcal{P}^{\mathcal{M}}: S \times A \times \Theta \to S$ . Here  $A$  represents the set of actions,  $S$  is the set of states. Similarly,  $\mathcal{I}^{\mathcal{M}}: S \to O$  is the environment-dependent observation function,  $\mathcal{R}^{\mathcal{M}}$  is the reward function, and  $\gamma$  is the discount factor. Specifically, given the environment parameters  $\vec{\theta} \in \Theta$ , we denote the corresponding environment instance as  $\mathcal{M}_{\vec{\theta}}$ . The student policy  $\pi$  is trained to maximize the cumulative rewards  $V^{\mathcal{M}_{\vec{\theta}}}(\pi) = \sum_{t=0}^{T} \gamma^{t} r_{t}$  in the given environment  $\mathcal{M}_{\vec{\theta}}$  under a time horizon  $T$ , and  $r_{t}$  are the collected rewards in  $\mathcal{M}_{\vec{\theta}}$ . Existing works on UED consist of two main strands: the RL-based environment generation approach and the domain randomization-based environment generation approach.

The RL-based generation approach was first formalized by Dennis et al. [3] as a self-supervised RL paradigm for generating environments. This approach involves co-evolving an environment generator policy (teacher) with an agent policy  $\pi$  (student), where the teacher's role is to generate environment instances that best support the student agent's continual learning. The teacher is trained to produce challenging yet solvable environments that maximize the regret measure, which is defined as the performance difference between the current student agent and a well-trained expert agent  $\pi^{*}$  within the current environment: Regret $\mathcal{M}_{\vec{\sigma}}(\pi, \pi^{*}) = V^{\mathcal{M}_{\vec{\sigma}}}(\pi^{*}) - V^{\mathcal{M}_{\vec{\sigma}}}(\pi)$ .

The domain randomization-based generation approach, on the other hand, involves randomly generating environments. Jiang et al. [7] propose to collect encountered environments with high learning potentials, which are approximated by the Generalized Advantage Estimation (GAE) [12], and then the student agent can selectively train in these environments, resulting in an emergent curriculum of increasing difficulty. Additionally, Parker-Holder et al. [10] adopt a different strategy by using predetermined starting points for the environment generation process and gradually increasing complexity. They manually divide the environment design space into different difficulty levels and employ human-defined edits to generate similar environments with high learning potentials. Their algorithm, ACCEL, is currently the state-of-the-art (SOTA) in the field, and we use an edited version of ACCEL as a baseline in our experiments.

# 2.2 Diffusion Probabilistic Models

Diffusion models [15] are a specific type of generative model that learns the data distribution. Recent advances in diffusion-based models, including Langevin dynamics and score-based generative models, have shown promising results in various applications, such as time series forecasting [18], robust learning [9], anomaly detection [21] as well as synthesizing high-quality images from text descriptions [8, 11]. These models can be trained using standard optimization techniques, such as stochastic gradient descent, making them highly scalable and easy to implement.

In a diffusion probabilistic model, we assume a  $d$ -dimensional random variable  $x_0 \in \mathbb{R}^d$  with an unknown distribution  $q(x_0)$ . Diffusion Probabilistic model involves two Markov chains: a predefined forward chain  $q(x_k|x_{k-1})$  that perturbs data to noise, and a trainable reverse chain  $p_\phi(x_{k-1}|x_k)$  that converts noise back to data. The forward chain is typically designed to transform any data distribution into a simple prior distribution (e.g., standard Gaussian) by considering perturb data with Gaussian noise of zero mean and a fixed variance schedule  $\{\beta_k\}_{k=1}^K$  for  $K$  steps:

$$
q \left(x _ {k} \mid x _ {k - 1}\right) = \mathcal {N} \left(x _ {k}; \sqrt {1 - \beta_ {k}} x _ {k - 1}, \beta_ {t} \mathbf {I}\right) \quad \text {a n d} \quad q \left(x _ {1: K} \mid x _ {0}\right) = \Pi_ {k = 1} ^ {K} q \left(x _ {k} \mid x _ {k - 1}\right), \tag {1}
$$

where  $k \in \{1, \dots, K\}$ , and  $0 < \beta_{1:K} < 1$  denote the noise scale scheduling. As  $K \to \infty$ ,  $x_{K}$  will converge to isometric Gaussian noise:  $x_{K} \rightarrow \mathcal{N}(0, \mathbf{I})$ . According to the rule of the sum of normally distributed random variables, the choice of Gaussian noise provides a closed-form solution to generate arbitrary time-step  $x_{k}$  through:

$$
x _ {k} = \sqrt {\bar {\alpha} _ {k}} x _ {0} + \sqrt {1 - \bar {\alpha} _ {k}} \epsilon , \quad \text {w h e r e} \quad \epsilon \sim \mathcal {N} (0, \mathbf {I}). \tag {2}
$$

Here  $\alpha_{k} = 1 - \beta_{k}$  and  $\bar{\alpha}_{k} = \prod_{s=1}^{k} \alpha_{s}$ . The reverse chain  $p_{\phi}(x_{k-1}|x_{k})$  reverses the forward process by learning transition kernels parameterized by deep neural networks. Specifically, considering the Markov chain parameterized by  $\phi$ , denoising arbitrary Gaussian noise into clean data samples can be written as:

$$
p _ {\phi} \left(x _ {k - 1} \mid x _ {k}\right) = \mathcal {N} \left(x _ {k - 1}; \mu_ {\phi} \left(x _ {k}, k\right), \Sigma_ {\phi} \left(x _ {k}, k\right)\right) \tag {3}
$$

It uses the Gaussian form  $p_{\phi}(x_{k - 1}|x_k)$  because the reverse process has the identical function form as the forward process when  $\beta_{t}$  is small [15]. Ho et al. [6] consider the following parameterization of

$$
\begin{array}{l} p _ {\phi} (x _ {k - 1} | x _ {k}): \\ \mu_ {\phi} \left(x _ {k}, k\right) = \frac {1}{\alpha_ {k}} \left(x _ {k} - \frac {\beta_ {k}}{\sqrt {1 - \alpha_ {k}}} \epsilon_ {\phi} \left(x _ {k}, k\right)\right) \text {a n d} \Sigma_ {\phi} \left(x _ {k}, k\right) = \tilde {\beta} _ {k} ^ {1 / 2} \text {w h e r e} \tilde {\beta} _ {k} = \left\{ \begin{array}{l l} \frac {1 - \alpha_ {k - 1}}{1 - \alpha_ {k}} \beta_ {k} & k > 1 \\ \beta_ {1} & k = 1 \end{array} \right. \tag {4} \\ \end{array}
$$

$\epsilon_{\phi}$  is a trainable function to predict the noise vector  $\epsilon$  from  $x_{k}$ . Ho et al. [6] show that training the reverse chain to maximize the log-likelihood  $\int q(x_0)\log p_\phi (x_0)dx_0$  is equivalent to minimizing re-weighted evidence lower bound (ELBO) that fits the noise. They derive the final simplified optimization objective:

$$
\mathcal {L} (\phi) = \mathbb {E} _ {x _ {0}, k, \epsilon} \left[ \| \epsilon - \epsilon_ {\phi} \left(\sqrt {\bar {\alpha} _ {k}} x _ {0} + \sqrt {1 - \bar {\alpha} _ {k}} \epsilon , k\right) \| ^ {2} \right]. \tag {5}
$$

Once the model is trained, new data points can be subsequently generated by first sampling a random vector from the prior distribution, followed by ancestral sampling through the reverse Markov chain in Equation 3.

# 3 Approach

In this section, we formally describe our method, Synthetically-enhanced Hierarchical Environment Design (SHED), which is a novel framework for UED under resource constraints. The SHED incorporates two key components that differentiate it from existing UED approaches:

- A hierarchical MDP framework to generate suitable environments,  
- A generative model to generate the synthetic trajectories.

SHED uses a hierarchical MDP framework where an RL teacher leverages the observed student's policy representation to generate environments at the student's capabilities frontier. Such targeted environment generation process enhances the student's general capability by utilizing the underlying structure of previously discovered environments, rather than relying on the open-ended random generation. Besides, SHED leverages advances in generative models to generate synthetic trajectories that can be used to train the off-policy teacher agent, which significantly reduces the costly interactions between the agents and the environments. The overall framework is shown in Figure 1, and the pseudo-code is provided in Algorithm 1.

# 3.1 Hierarchical Environment Design

The objective is to generate a limited number of environments that are designed to enhance the general capability of the student agent. Inspired by the principles of PAIRED [3], we adopt an RL-based approach for the environment generation process. To better generate suitable environments tailored to the current student skill level, SHED uses the hierarchical MDP framework, consisting of an upper-level RL teacher policy  $\Lambda$  and a lower-level student policy  $\pi$ . Specifically, the teacher policy,  $\Lambda : \Pi \rightarrow \Theta$ , maps from the space of all potential student policies  $\Pi$  to the space of environment parameters  $\Theta$ . Existing RL-based methods (e.g., PARIED) rely solely on regret feedback and fail to effectively capture the nuances of the student policy. To address this challenge, SHED enhances understanding by encoding the student policy  $\pi$  into a vector that serves as the state abstraction for teacher  $\Lambda$ . Rather than compressing the knowledge in the student policy network, we approximate the embedding of the student policy  $\pi$  by assessing performance across a set of diverse evaluation environments. This performance vector, denoted as  $p(\pi)$ , gives us a practical estimate of the student's current general capabilities, enabling the teacher to customize the next training environments accordingly. In our hierarchical framework, the environment generation process is governed by discrete-time dynamics. We delve into the specifics below.

Upper-level teacher MDP. The upper-level teacher operates at a coarser layer of student policy abstraction and generates environments to train the lower-level student agent. This process can be formally modeled as an MDP by the tuple  $< S^u, A^u, P^u, R^u, \gamma^u >$ :

-  $S^u$  represents the upper-level state space. Typically,  $s^u = p(\pi) = [p_1, \dots, p_m]$  denotes the student performance vector across  $m$  diverse evaluation environments. This vector serves as the representation of the student policy  $\pi$  and is observed by the teacher.

# Algorithm 1 SHED

Input: real data ratio  $\psi \in [0,1]$ , evaluate environment set  $\theta^{\mathrm{eval}}$ , reward function  $R$ ;

1: Initialize: diffusion model  $D$ , teacher policy  $\Lambda$ , real and synthetic replay buffer  $\mathcal{B}_{\mathrm{real}}$ ,  $\mathcal{B}_{\mathrm{syn}} = \emptyset$ ;  
2: for episode  $ep = 1,\dots ,K$  do  
3: Initialize student policy  $\pi$  
4: Evaluate  $\pi$  on  $\theta^{\mathrm{eval}}$  and get state  $s^u = p(\pi)$  
5: for Budget  $t = 1, \dots, T$  do

6: generate  $\tilde{\theta} \sim \Lambda$ , and create  $\mathcal{M}_{\tilde{\theta}}(\pi)$  
7: train  $\pi$  on  $\mathcal{M}_{\tilde{\theta}}$  to maximize  $V^{\tilde{\theta}}(\pi)$  
8: evaluate  $\pi$  on  $\theta^{\mathrm{eval}}$  and get next state  $s'$  
9: compute teacher's reward  $r_t$  according to  $R$  
0: add experience  $(s_t^u,\vec{\theta},r_t^u,s_t^{u',})$  to  $\mathcal{B}_{real}$  
11: train  $D$  with samples from  $\mathcal{B}_{\mathrm{real}}$  
2: generate synthetic experiences from  $D$  and add them to  $\mathcal{B}_{\mathrm{syn}}$  
3: train  $\Lambda$  on samples from  $\mathcal{B}_{\mathrm{real}} \cup \mathcal{B}_{\mathrm{syn}}$  mixed with ratio  $\psi$  
4: set  $s = s'$  
5: end for  
6: end for

Output:  $\Lambda, \pi, D$

![](images/9b76da2e9b0449339783de0d76ac0d0c4cc172d9c13cf286d8174b24556600c7.jpg)

![](images/0c2c464d64fdcc27400fb5cb667cb10f87fbf241929b1383af9613e1789cef5a.jpg)  
Figure 1: The overall framework of SHED.  
Figure 2: The illustration of the environment generation process.

-  $A^u$  is the upper-level action space. The teacher observes the abstraction of the student policy,  $s^u$  and produces an upper-level action  $a^u$  which is the environment parameters  $\vec{\theta}$ .  $\vec{\theta}(a^u)$  is then used to generate specific environment instances  $\mathcal{M}_{\vec{\theta}}$ . Thus the upper-level action space  $A^u$  is the environment parameter space  $\Theta$ .  
-  $P^u$  denotes the action-dependent transition dynamics of the upper-level state. The general capability of the student policy evolves due to training the student agent on the generated environments.  
-  $R^u$  provides the upper-level reward to the teacher at the end of training the student on the generated environment. The design of  $R^u$  will be discussed in Section 3.3.

As shown in Figure 2, given the student policy  $\pi$ , the teacher  $\Lambda$  first observes the representation of the student policy,  $s^u = [p_1,\dots ,p_m]$ . Then teacher produces an upper-level action  $a^u$  which corresponds to the environment parameters. These environment parameters are subsequently used to generate specific environment instances. The lower-level student policy  $\pi$  will be trained on the generated environments for  $C$  training steps. The upper-level teacher collects and stores the student policy evolution transition  $(s^u,a^u,r^u,s^{u,t})$  every  $C$  times steps for off-policy training. The teacher agent is trained to maximize the cumulative reward giving the budget for the number of generated environments. The choice of the evaluation environments will be discussed in Section 3.3.

Lower-level student MDP. The generated environment is fully specified for the student, characterized by a Partially Observable Markov Decision Process (POMDP), which is defined by a tuple  $\mathcal{M}_{\vec{\theta}} = < A, O, S^{\vec{\theta}}, \mathcal{P}^{\vec{\theta}}, \mathcal{I}^{\vec{\theta}}, \mathcal{R}^{\vec{\theta}}, \gamma >$ , where  $A$  represents the set of actions,  $O$  is the set of observations,  $S^{\vec{\theta}}$  is the set of states determined by the environment parameters  $\vec{\theta}$ , similarly,  $\mathcal{P}^{\vec{\theta}}$  is the environment-dependent transition function, and  $\mathcal{I}^{\vec{\theta}}: \vec{\theta} \to O$  is the environment-dependent observation function,  $\mathcal{R}^{\vec{\theta}}$  is the reward function, and  $\gamma$  is the discount factor. At each time step  $t$ , the environment produces a state observation  $s_t \in S^{\vec{\theta}}$ , the student agent samples the action  $a_t \sim A$  and interacts with environment  $\vec{\theta}$ . The environment yields a reward  $r_t$  according to the reward function  $\mathcal{R}^{\vec{\theta}}$ . The student agent is trained to maximize their cumulative reward  $V^{\vec{\theta}}(\pi) = \sum_{t=0}^{C} \gamma^t r_t$  for the current environment under a finite time horizon  $C$ . The student agent will learn a good general capability from training on a sequence of generated environments.

The hierarchical framework enables the teacher agent to systematically measure and enhance the general capability of the student agent and to adapt the training process accordingly. However, it's worth noting that collecting student policy evolution trajectories  $(s^u, a^u, r^u, s^{u,'})$  to train the teacher agent is notably slow and resource-intensive, since each transition in the upper-level teacher MDP encompasses a training horizon of  $C$  timesteps for the student in the generated environment. Thus, it is essential to reduce the need for costly collection of upper-level teacher experiences.

# 3.2 Generative Trajectory Modeling

In this section, we will formally introduce a generative model designed to ease the collection of upper-level MDP experience. This will allow us to train our teacher policy more efficiently. In particular, we first utilize a diffusion model to learn the conditional data distribution from the collected experiences  $\tau = \{(s_t^u, a_t^u, r_t^u, s_t^{p,t})\}$ . Later we can use the reverse chain in the diffusion model to generate the synthetic trajectories that can be used to help train the teacher agent, thereby alleviating the need for extensive and time-consuming collection of upper-level teacher experiences. We deal with two different types of timesteps in this section: one for the diffusion process and the other for the upper-level teacher agent, respectively. We use subscripts  $k \in 1, \ldots, K$  to represent diffusion timesteps and subscripts  $t \in 1, \ldots, T$  to represent trajectory timesteps in the teacher's experience.

In the image domain, the diffusion process is implemented across all pixel values of the image. In our setting, we diffuse over the next state  $s^{u,t}$  conditioned the given state  $s^u$  and action  $a^u$ . We construct our generative model according to the conditional diffusion process:

$$
q \big (s _ {k} ^ {u, \prime} | s _ {k - 1} ^ {u, \prime} \big), \quad p _ {\phi} \big (s _ {k - 1} ^ {u, \prime} | s _ {k} ^ {u, \prime}, s ^ {u}, a ^ {u} \big)
$$

As usual,  $q(s_{k}^{u, \prime} | s_{k-1}^{u, \prime})$  is the predefined forward noising process while  $p_{\phi}(s_{k-1}^{u, \prime} | s_{k}^{u, \prime}, s^{u}, a^{u})$  is the trainable reverse denoising process. We begin by randomly sampling the collected experiences  $\tau = \{(s_{t}^{u}, a_{t}^{u}, r_{t}^{u}, s_{t}^{u, \prime})\}$  from the real experience buffer  $\mathcal{B}_{real}$ . Giving the observed state  $s^{u}$  and action  $a^{u}$ , we use the reverse process  $p_{\phi}$  to represent the generation of the next state  $s^{u, \prime}$ :

$$
p _ {\phi} (s _ {0: K} ^ {u, \prime} | s ^ {u}, a ^ {u}) = \mathcal {N} (s _ {K} ^ {u, \prime}; 0, \mathbf {I}) \prod_ {k = 1} ^ {K} p _ {\phi} (s _ {k - 1} ^ {u, \prime} | s _ {k} ^ {u, \prime}, s ^ {u}, a ^ {u})
$$

At the end of the reverse chain, the sample  $s_0^{u,'}$ , is the generated next state  $s^{u,'}$ . Similar to Ho et al. [6], we parameterize  $p_{\phi}(s_{k - 1}^{\prime}|s_k^{\prime},s^u,a^u)$  as a noise prediction model with the covariance matrix fixed as  $\Sigma_{\phi}(s_k^{u,'},s^u,a^u,k) = \beta_i\mathbf{I}$ , and the mean is

$$
\mu_ {\phi} (s _ {i} ^ {u, \prime}, s ^ {u}, a ^ {u}, k) = \frac {1}{\sqrt {\alpha_ {k}}} \left(s _ {k} ^ {u, \prime} - \frac {\beta_ {k}}{\sqrt {1 - \bar {\alpha} _ {k}}} \epsilon_ {\phi} (s _ {k} ^ {u, \prime}, s ^ {u}, a ^ {u}, k)\right)
$$

$\epsilon_{\phi}(s_k^{u, \prime}, s^u, a^u, k)$  is the trainable denoising function, which aims to estimate the noise  $\epsilon$  in the noisy input  $s_k^{u, \prime}$  at step  $k$ .

Training objective. We employ a similar simplified objective to train the conditional  $\epsilon$ -model:

$$
\mathcal {L} (\phi) = \mathbb {E} _ {\left(s ^ {u}, a ^ {u}, s ^ {u, \prime}\right) \sim \tau , k \sim \mathcal {U}, \epsilon \sim \mathcal {N} (0, \mathbf {I})} \left[ \| \epsilon - \epsilon_ {\phi} \left(s _ {k} ^ {u, \prime}, s ^ {u}, a ^ {u}, k\right) \| ^ {2} \right] \tag {6}
$$

Where  $s_k^{u,'} = \sqrt{\bar{\alpha}_k} s^{u,'} + \sqrt{1 - \bar{\alpha}_k}\epsilon$ . The intuition for the loss function  $\mathcal{L}(\phi)$  is to predict the noise  $\epsilon \sim \mathcal{N}(0,\mathbf{I})$  at the denoising step  $k$ , and the diffusion model is essentially learning the student policy involution trajectories collected in the real experience buffer  $\mathcal{B}_{\text{reals}}$ . Note that the reverse process necessitates a substantial number of steps  $K$  [15]. Recent research by Xiao et al. [22] has demonstrated that enabling denoising with large steps can reduce the total number of denoising steps  $K$ . To expedite the relatively slow reverse sampling process (as it requires computing  $\epsilon_{\phi}$  networks  $K$  times), we use a small value of  $K$ . Similar to Wang et al. [20], while simultaneously setting  $\beta_{\min} = 0.1$  and  $\beta_{\max} = 10.0$ , we define:

$$
\beta_ {k} = 1 - \exp \left(\beta_ {\mathrm {m i n}} \times \frac {1}{K} - 0. 5 (\beta_ {\mathrm {m a x}} - \beta_ {\mathrm {m i n}}) \frac {2 k - 1}{K ^ {2}}\right)
$$

This noise schedule is derived from the variance-preserving Stochastic Differential Equation by Song et al. [16].

Generate synthetic trajectories. Once the diffusion model has been trained, it can be used to generate synthetic experience data by starting with a draw from the prior  $s_K^{u,\prime} \sim \mathcal{N}(0,\mathbf{I})$  and successively generating denoised next state, conditioned on the given  $s^u$  and  $a^u$  through the reverse chain  $p_{\phi}$ . Note that the giving condition action  $a$  can either be randomly sampled from the action space or use another diffusion model to learn the action distribution giving the initial state  $s^u$ . This new diffusion model is essentially a behavior-cloning model that aims to learn the teacher policy  $\Lambda(a^u | s^u)$ . This process is similar to the work of Wang et al. [20]. We discuss this process in detail in the appendix. In this paper, we randomly sample  $a^u$  as it is straightforward and can also increase the diversity in the generated synthetic experience to help train a more robust teacher agent.

After obtaining the generated next state  $s^{u,'}$  conditioned on  $s^u, a^u$ , we compute reward  $r^u$  using teacher's reward function  $R(s^u, a^u, s^{u,'})$ . The specifics of how the reward function is chosen are explained in the following section.

# 3.3 Rewards and Choice of evaluate environments

Selection of evaluation environments. The upper-level teacher generates environments tailored for the lower-level student to improve its general capability. Thus it is important to select a set of diverse suitable evaluation environments as the performance vector reflects the student agent's general capabilities and serves as an approximation of the policy's embedding. Fontaine and Nikolaidis [5] propose the use of quality diversity (QD) optimization to collect high-quality environments that exhibit diversity for the agent behaviors. Similarly, Bhatt et al. [1] introduce a QD-based algorithm for dynamically designing such evaluation environments based on the current agent's behavior. However, it's worth noting that this QD-based approach can be tedious and time-consuming, and the collected evaluation environments heavily rely on the given agent policy.

Given these considerations, it is natural to take advantage of the domain randomization algorithm, as it has demonstrated compelling results in generating diverse environments and training generally capable agents. In our approach, we first discretize the environment parameters into different ranges, then randomly sample from these ranges, and combine these parameters to generate evaluation environments. This method can generate environments that may induce a diverse performance for the same policy, and it shows promising empirical results in the final experiments.

Reward design. We define the reward function for the upper-level teacher policy as a parameterized function based on the improvement in student performance in the evaluation environments after training in the generated environment:

$$
R \left(s ^ {u}, a ^ {u}, s ^ {u, \prime}\right) = \sum_ {i = 1} ^ {m} \left(p _ {i} ^ {\prime} - p _ {i}\right)
$$

This reward function gives positive rewards to the upper-level teacher for taking action to create the right environment to improve the overall performance of students across diverse environments. However, it may encourage the teacher to obtain higher rewards by sacrificing student performance in one subset of evaluation environments to improve student performance in another subset, which conflicts with our objective to develop a student agent with general capabilities. Therefore, we need to consider fairness in the reward function to ensure that the generated environment can improve student's general capabilities. Similar to [4], we build our fairness metric on top of the change in student's performance in each evaluation environment, denoted as  $\omega_{i} = p_{i}^{\prime} - p_{i}$ , and we have  $\bar{\omega} = \frac{1}{m}\sum_{i=1}^{m}\omega_{i}$ . We then measure the fairness of the teacher's action using the coefficient of variation of student performances:

$$
c v \left(s ^ {u}, a ^ {u}, s ^ {u, \prime}\right) = \sqrt {\frac {1}{m - 1} \sum_ {i} \frac {\left(\omega_ {i} - \bar {\omega}\right) ^ {2}}{\bar {\omega} ^ {2}}} \tag {7}
$$

A teacher is considered to be fair if and only if the  $cv$  is smaller. As a result, our reward function is:

$$
R \left(s ^ {u}, a ^ {u}, s ^ {u, \prime}\right) = \sum_ {i = 1} ^ {m} \left(p _ {i} ^ {\prime} - p _ {i}\right) - \eta \cdot c v \left(s ^ {u}, a ^ {u}, s ^ {u, \prime}\right) \tag {8}
$$

Here  $\eta$  is the coefficient that balances the weight of fairness in the reward function (We set a small value to  $\eta$ ). This reward function motivates the teacher to generate training environments that can improve student's general capability.

![](images/065d8a9b32628e066194c5e0d961ee432e5df575f5a12616851b7ef7bc1163d3.jpg)  
Figure 3: Left: The average zero-shot transfer performances on the test environments in the Lunar lander environment (mean and standard error). Right: The average zero-shot transfer performances on the test environments in the BipedalWalker (mean and standard error).

![](images/b57dd56a83b62a5a69f82174b1e1190d5fd82a1afa0cf1d995ac0bf35844f8d9.jpg)

# 4 Experiments

In this section, we conduct experiments to compare SHED to other leading approaches on three domains: Lunar Lander, maze and a modified BipedalWalker environment. Experimental details and hyperparameters can be found in the Appendix. Specifically, our primary comparisons involve SHED and  $h$ -MDP (our proposed hierarchical approach without diffusion model aiding in training) against four baselines: domain randomization [19], ACCEL, [10], Edited ACCEL(with slight modifications that it does not revisit the previously generated environments), PAIRED [3]. In all cases, we train a student agent via Proximal Policy Optimization (PPO [13], and train the teacher agent via Deterministic policy gradient algorithms(DDPG [14]), because DDPG is an off-policy algorithm and can learn from both real experiences and the synthetic experiences.

Setup. For each domain, we construct a set of evaluation environments and a set of test environments. The vector of student performances in the evaluation environments is used as the approximation of the student policy (as the observation to teacher agent), and the performances in the test environments are used to represent the student's zero-shot transfer performances (general capabilities). Note that in order to obtain a fair comparison of zero-shot transfer performance, the evaluation environments and test environments do not share the same environment and they are not present during training.

Lunar Lander. This is a classic rocket trajectory optimization problem. In this domain, student agents are tasked with controlling a lander's engine to safely land the vehicle. Before the start of each episode, teacher algorithms determine the environment parameters that are used to generate environments in a given play-through, which includes gravity, wind power, and turbulence power. These parameters directly alter the difficulty of landing the vehicle safely. The state is an 8-dimensional vector, which includes the coordinates of the lander, its linear velocities, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

We train the student agent for 1e6 environment time steps and periodically test the agent in test environments. The parameters for the test environments are randomly generated and fixed during training. We report the experiment results on the left side of Figure 3. As we can see, student agents trained under SHED consistently outperform other baselines and have minimal variance in transfer performance. During training, the baselines, except h-MDP, show a performance dip in the middle. This phenomenon could potentially be attributed to the inherent challenge of designing the appropriate environment instance in the large environment parameter space. This further demonstrates the effectiveness of our hierarchical design (SHED and h-MDP), which can successfully create environments that are appropriate to the current skill level of the students.

Bipedalwalker. We also evaluate SHED in the modified BipedalWalker from Parker-Holder et al. [10]. In this domain, the student agent is required to control a bipedal vehicle and navigate across the terrain, and the student receives a 24-dimensional proprioceptive state with respect to its lidar sensors, angles, and contacts. The teacher is tasked to select eight variables (including ground roughness, the

number of stairs steps, min/max range of pit gap width, min/max range of stump height, and min/max range of stair height) to generate the corresponding terrain.

We use similar experiment settings in prior UED works, we train all the algorithms for 1e7 environment time steps, and then evaluate their generalization ability on ten distinct test environments in Bipedal-Walker domain. The parameters for the test environments are randomly generated and fixed during training. As shown in Figure 3, our proposed method SHED surpasses all other baselines and achieves performance levels nearly on par with the SOTA (ACCEL). Meanwhile, SHED maintains a slight edge in terms of stability and overall performance and PAIRED suffers from a considerable degree of variance in its performance.

Partially observable Maze. Here we study navigation tasks, where an agent must explore to find a goal while navigating around obstacles. The environment is partially observable, and the agent's field of view is limited to a  $3 \times 3$  grid area. Unlike the previously mentioned domains, maze environments are non-parametric and cannot be directly represented by compact parameter vectors due to their high complexity. To solve this challenge, we propose a novel method to generate maze by leveraging advances in large language models (e.g., ChatGPT). Specifically, we implement a retrieval-augmented generation (RAG) process to optimize the ChatGPT's output such that it can generate desired maze environments. This process ensures that large language models reference authoritative knowledge bases to generate feasible mazes. To simplify the teacher's action space, we extracted several key factors that constitute the teacher's action space (environmental parameters) for maze generation. Details on maze generation are provided in Appendix D.3, and prompt are included in Appendix D.4.

The average zero-shot transfer performances are reported in Figure 4. Notably, SHED demonstrates the highest performance, consistently improving and achieving the highest cumulative rewards. The performance of h-MDP steadily improves but does not reach the highest levels, which further highlights the advantages of incorporating the generated synthetic datasets to train an effective RL teacher agent. Meanwhile, Accel-Edit and Accel show higher variances in performance, indicating that random teachers are less stable in finding a suitable environment to train student agents.

![](images/2a7e8d06bb6cad242e936905de7bce1530bf7110cae170bde0adfa850e720db8.jpg)  
Figure 4: Average zero-shot transfer performance on the test environments in the maze environments.

# Ablation and additional Experiments In Appendix C, we evaluate

the ability of the diffusion model to generate the synthetic student policy involution trajectories. We further provide ablation studies to assess the impact of different design choices in Appendix E.1. Additionally, in Appendix E.2, we conduct experiments to show how the algorithm performs under different settings, including scenarios with a larger budget constraint on the number of generated environments or a larger weight assigned to CV fairness rewards. Notably, all results consistently demonstrate the effectiveness of our approach.

# 5 Conclusion

In this paper, we introduce an adaptive approach for efficiently training a generally capable agent under resource constraints. Our approach is general, utilizing an upper-level MDP teacher agent that can guide the training of the lower-level MDP student agent. The hierarchical framework can incorporate techniques from existing UED works, such as prioritized level replay (revisiting environments with high learning potential). Furthermore, we have described a method to assist the experience collection for the teacher when it is trained in an off-policy manner. Our experiment demonstrates that our method outperforms existing UED methods, highlighting its effectiveness as a curriculum-based learning approach within the UED framework.

# References

[1] Varun Bhatt, Bryon Tjanaka, Matthew Fontaine, and Stefanos Nikolaidis. Deep surrogate assisted generation of environments. Advances in Neural Information Processing Systems, 35: 37762-37777, 2022.  
[2] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. arXiv preprint arXiv:2402.15391, 2024.  
[3] Michael Dennis, Natasha Jaques, Eugene Vinitsky, Alexandre Bayen, Stuart Russell, Andrew Critch, and Sergey Levine. Emergent complexity and zero-shot transfer via unsupervised environment design. Advances in neural information processing systems, 33:13049-13061, 2020.  
[4] Salma Elmalaki. Fair-iot: Fairness-aware human-in-the-loop reinforcement learning for harnessing human variability in personalized iot. In Proceedings of the International Conference on Internet-of-Things Design and Implementation, pages 119–132, 2021.  
[5] Matthew Fontaine and Stefanos Nikolaidis. Differentiable quality diversity. Advances in Neural Information Processing Systems, 34:10040-10052, 2021.  
[6] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840-6851, 2020.  
[7] Minqi Jiang, Edward Grefenstette, and Tim Rocktäschel. Prioritized level replay. In International Conference on Machine Learning, pages 4940-4950. PMLR, 2021.  
[8] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.  
[9] Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, and Anima Anandkumar. Diffusion models for adversarial purification. arXiv preprint arXiv:2205.07460, 2022.  
[10] Jack Parker-Holder, Minqi Jiang, Michael Dennis, Mikayel Samvelyan, Jakob Foerster, Edward Grefenstette, and Tim Rocktäschel. Evolving curricula with regret-based environment design. arXiv preprint arXiv:2203.01302, 2022.  
[11] Chitwan Sahara, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479-36494, 2022.  
[12] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.  
[13] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
[14] David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic policy gradient algorithms. In International conference on machine learning, pages 387-395. Pmlr, 2014.  
[15] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256-2265. PMLR, 2015.  
[16] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.  
[17] Richard S Sutton, Andrew G Barto, et al. Introduction to reinforcement learning, volume 135. MIT press Cambridge, 1998.

[18] Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. Csdi: Conditional score-based diffusion models for probabilistic time series imputation. Advances in Neural Information Processing Systems, 34:24804-24816, 2021.  
[19] Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS), pages 23-30. IEEE, 2017.  
[20] Zhendong Wang, Jonathan J Hunt, and Mingyuan Zhou. Diffusion policies as an expressive policy class for offline reinforcement learning. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=AHvFDPi-FA.  
[21] Julian Wyatt, Adam Leach, Sebastian M Schmon, and Chris G Willcocks. Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 650-656, 2022.  
[22] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion gans. arXiv preprint arXiv:2112.07804, 2021.
