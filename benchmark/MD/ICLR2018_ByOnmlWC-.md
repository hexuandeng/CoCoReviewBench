# GENETIC POLICY OPTIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Genetic algorithms have been widely used in many practical optimization problems. Inspired by natural selection, operators, including mutation, crossover and selection, provide effective heuristics for search and black-box optimization. However, they have not been shown useful for deep reinforcement learning, possibly due to the catastrophic consequence of parameter crossovers of neural networks. Here, we present Genetic Policy Optimization (GPO), a new genetic algorithm for sample-efficient deep policy optimization. GPO uses imitation learning for policy crossover in the state space and applies policy gradient methods for mutation. Our experiments on Mujoco tasks show that GPO as a genetic algorithm is able to provide superior performance over the state-of-the-art policy gradient methods and achieves comparable or higher sample efficiency.

# 1 INTRODUCTION

Reinforcement learning (RL) has recently demonstrated significant progress and achieves state-of-the-art performance in games (Mnih et al., 2015; Silver et al., 2016), locomotion control (Lillicrap et al., 2015), visual navigation (Zhu et al., 2017), and robotics (Levine et al., 2016). Among these successes, deep neural networks (DNNs) are widely used as powerful functional approximators to enable signal perception, feature extraction and complex decision making. For example, in continuous control tasks, the policy that determines which action to take is often parameterized by a deep neural network that takes the current state observation or sensor measurements as input. In order to optimize such policies, various policy gradient methods (Mnih et al., 2016; Schulman et al., 2015; 2017; Heess et al., 2017) have been proposed to estimate gradients approximately from rollout trajectories. The core idea of these policy gradient methods is to take advantage of the temporal structure in the rollout trajectories to construct a Monte Carlo estimator of the gradient of the expected return.

In addition to the popular policy gradient methods, other alternative solutions, such as those for black-box optimization or stochastic optimization, have been recently studied for policy optimization. Evolution strategies (ES) is a class of stochastic optimization techniques that can search the policy space without relying on the backpropagation of gradients. At each iteration, ES samples a candidate population of parameter vectors ("genotypes") from a probability distribution over the parameter space, evaluates the objective function ("fitness") on these candidates, and constructs a new probability distribution over the parameter space using the candidates with the high fitness. This process is repeated iteratively until the objective is maximized. Covariance matrix adaptation evolution strategy (CMA-ES; Hansen & Ostermeier (2001)) and recent work from Salimans et al. (2017) are examples of this procedure. These ES algorithms have also shown promising results on continuous control tasks and Atari games, but their sample efficiency is often not comparable to the advanced policy gradient methods, because ES is black-box and thus does not fully exploit the policy network architectures or the temporal structure of the RL problems.

Very similar to ES, genetic algorithms (GAs) are a heuristic search technique for search and optimization. Inspired by the process of natural selection, GAs evolve an initial population of genotypes by repeated applications of three genetic operators - mutation, crossover and selection. The major difference between GA and ES is that the crossover operator in GA is able to provide higher diversity of good candidates in the population. However, the crossover operator is often performed on the parameter representations of two parents, thus making it not suitable for nonlinear neural networks. The straightforward crossover of two neural networks by exchanging their parameters can often destroy the hierarchical relationship of the networks and thus cause a catastrophic drop in per

formance. NeuroEvolution of Augmenting Topologies (NEAT; Stanley & Miikkulainen (2002b;a)), which evolves neural networks through evolutionary algorithms such as GA, provides a solution to exchange and augment neurons but has found limited success when used as a method of policy search in deep RL for high-dimensional tasks. A major challenge to making GAs work for policy optimization is to design a good crossover operator which efficiently combines two parent policies represented by neural networks and generates an offspring that takes advantage of both parents. In addition, a good mutation operator is needed as random perturbations are often inefficient for high-dimensional policies.

In this paper, we present Genetic Policy Optimization (GPO), a new genetic algorithm for sample-efficient deep policy optimization. There are two major technical advances in GPO. First, instead of using parameter crossover, GPO applies imitation learning for policy crossovers in the state space. The state-space crossover effectively combines two parent policies into an offspring or child policy that tries to mimic its best parent in generating similar state visitation distributions. Second, GPO applies advanced policy gradient methods for mutation. By randomly rolling out trajectories and performing gradient descent updates, this mutation operator is more efficient than random parameter perturbations and also maintains sufficient genetic diversity. Our experiments on several continuous control tasks show that GPO as a genetic algorithm is able to provide superior performance over the state-of-the-art policy gradient methods and achieves comparable or higher sample efficiency.

# 2 BACKGROUND AND RELATED WORK

# 2.1 REINFORCEMENT LEARNING

In the standard RL setting, an agent interacts with an environment  $\mathcal{E}$  modeled as a Markov Decision Process (MDP). At each discrete time step  $t$ , the agent observes a state  $s_t$  and choose an action  $a_t \in \mathcal{A}$  using a policy  $\pi(a_t | s_t)$ , which is a mapping from states to a distribution over possible actions. Here we consider high-dimensional, continuous state and action spaces. After performing the action  $a_t$ , the agent collects a scalar reward  $r(s_t, a_t) \in \mathcal{R}$  at each time step. The goal in reinforcement learning is to learn a policy which maximizes the expected sum of (discounted) rewards starting from the initial state. Formally, the objective is

$$
J (\pi) = \mathbb {E} _ {\left\{\left(s _ {t}, a _ {t}\right) \right\} \sim \mathcal {E}, \pi} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} r \left(s _ {t}, a _ {t}\right) \right]
$$

where the states  $s_t$  are sampled from the environment  $\mathcal{E}$  using an unknown system dynamics model  $p(s_{t+1}|s_t, a_t)$  and an initial state distribution  $p(s_0)$ , the actions  $a_t$  are sampled from the policy  $\pi(a_t|s_t)$  and  $\gamma \in (0, 1]$  is the discount factor.

# 2.2 POLICY GRADIENT METHODS

Policy-based RL methods search for an optimum policy directly in the policy space. One popular approach is to parameterize the policy  $\pi (a_t|s_t;\theta)$  with  $\theta$ , express the objective  $J(\pi (a_t|s_t;\theta))$  as a function of  $\theta$  and perform gradient descent methods to optimize it. The REINFORCE algorithm (Williams, 1992) calculates an unbiased estimation of the gradient  $\nabla_{\theta}J(\theta)$  using the likelihood ratio trick. Specifically, REINFORCE updates the policy parameters in the direction of the following approximation to policy gradient

$$
\nabla_ {\theta} J (\theta) \approx \sum_ {t = 0} ^ {\infty} \nabla_ {\theta} \log \pi (a _ {t} | s _ {t}; \theta) R _ {t}
$$

based on a single rollout trajectory, where  $R_{t} = \sum_{i=0}^{\infty} \gamma^{i} r(s_{t+i}, a_{t+i})$  is the discounted sum of rewards from time step  $t$ . The advantage actor-critic (A2C) algorithm (Sutton & Barto; Mnih et al., 2016) uses the state value function (or critic) to reduce the variance in the above gradient estimation. The contribution to the gradient at time step  $t$  is  $\nabla_{\theta} \log \pi(a_{t}|s_{t};\theta)(R_{t} - V^{\pi}(s_{t}))$ .  $R_{t} - V^{\pi}(s_{t})$  is an estimate of the advantage function  $A^{\pi}(s_{t},a_{t}) = Q^{\pi}(s_{t},a_{t}) - V^{\pi}(s_{t})$ . In practice, multiple rollouts are performed to get the policy gradient, and  $V^{\pi}(s_{t})$  is learned using a function approximator.

High variance in policy gradient estimates can sometimes lead to large, destructive updates to the policy parameters. Trust-region methods such as TRPO (Schulman et al., 2015) avoid this by restricting the amount by which an update is allowed to change the policy. TRPO is a second order

algorithm that solves an approximation to a constrained optimization problem using conjugate gradient. Proximal policy optimization (PPO) algorithm (Schulman et al., 2017) is an approximation to TRPO that relies only on first order gradients. The PPO objective penalizes the Kullback-Leibler (KL) divergence change between the policy before the update  $(\pi_{\theta_{old}})$  and the policy at the current step  $(\pi_{\theta})$ . The penalty weight  $\beta$  is adaptive and adjusted based on observed change in KL divergence after multiple policy update steps have been performed using the same batch of data.

$$
J ^ {P P O} (\theta) = \hat {\mathbb {E}} _ {t} \bigg [ \frac {\pi_ {\theta} (a _ {t} | s _ {t})}{\pi_ {\theta_ {o l d}} (a _ {t} | s _ {t})} \hat {A} _ {t} - \beta K L \big [ \pi_ {\theta_ {o l d}} (. | s _ {t}), \pi_ {\theta} (. | s _ {t}) \big ] \bigg ]
$$

where  $\hat{\mathbb{E}}_t[\ldots]$  indicates the empirical average over a finite batch of samples, and  $\hat{A}_t$  is the advantage estimation. Schulman et al. (2017) propose another objective based on clipping of the likelihood ratio, but we use the adaptive-KL objective due to its better empirical performance (Heess et al., 2017; Hafner et al., 2017).

# 2.3 EVOLUTIONARY ALGORITHMS

There is growing interest in using evolutionary algorithms as a policy search procedure in RL. We provide a brief summary; a detailed survey is provided by Whiteson (2012). Recently, Salimans et al. (2017) proposed a version of Evolution Strategies (ES) for black-box policy optimization. At each iteration  $k$ , the algorithm samples candidate parameter vectors (policies) using a fixed covariance Gaussian  $\mathcal{N}(0,\sigma^2 I)$  perturbation on the mean vector  $m^{(k)}$ . The mean vector is then updated in the direction of the weighted average of the perturbations, where weight is proportional to the fitness of the candidate. CMA-NeuroES (Heidrich-Meisner & Igel, 2009) uses CMA-ES to learn neural network policies for episodic reinforcement learning. CMA-ES samples candidate parameter vectors using a Gaussian  $\mathcal{N}(0,C^{(k)})$  perturbation on the mean vector  $m^{(k)}$ . The covariance matrix and the mean vector for the next iteration are then calculated using the candidates with high fitness. Cross-Entropy methods use similar ideas and have been found to work reasonably well in simple environments (Szita & Lörincz, 2006).

Among genetic algorithm approaches to policy optimization, NEAT and its extension, HyperNEAT have enjoyed some success (Stanley & Miikkulainen, 2002a). NEAT can evolve connection weights as well as the network topology. The crossover between fixed-topology parents is done by copying the weights of each DNN edge randomly from one of the parents. The weights are mutated using random perturbations. This mode of mutation is also used by Salimans et al. (2017). A more principled take on mutating weights with perturbations can be found in (Hansen & Ostermeier, 2001; Sehnke et al., 2010). In this work, we use policy gradient algorithms for efficient mutation of high-dimensional policies, and also depart from prior work in implementing the crossover operator.

# 3 GENETIC POLICY OPTIMIZATION

# 3.1 OVERALL ALGORITHM

Our procedure for policy optimization proceeds by evolving the policies (genotypes) through a series of selection, crossover and mutation operators (Algorithm 1). We start with an ensemble of policies initialized with random parameters. In line 3, we mutate each of the policies separately by performing a few iterations of updates on the policy parameters. Any standard policy gradient method, such as PPO or A2C, can be used for mutation. In line 4, we create a set of parents using a selection procedure guided by a fitness function. Each element of this set is a policy-pair  $(\pi_x, \pi_y)$  that is used in the reproduction (crossover) step to produce a new child policy  $(\pi_c)$ . This is done in line 7 by mixing the policies of the parents. In line 10, we obtain the population for the next generation by collecting all the newly created children. The algorithm terminates after  $k$  rounds of optimization.

# 3.2 GPO CROSSOVER AND MUTATION

We consider policies that are parameterized using deep neural networks of fixed architectures. If the policy is Gaussian, as is common for many robotics and locomotion tasks (Duan et al., 2016), then

Algorithm 1 Genetic Policy Optimization  
1: population  $\leftarrow \pi_1, \ldots, \pi_m$  ▷ Initial policies with random parameters  
2: repeat  
3: population  $\leftarrow$  MUTATE(population)  
4: parents_set  $\leftarrow$  SELECT(population, FITNESS-FN)  
5: children  $\leftarrow$  empty set  
6: for tuple  $(\pi_x, \pi_y) \in$  parents_set do  
7:  $\pi_c \leftarrow$  CROSSOVER  $(\pi_x, \pi_y)$   
8: add  $\pi_c$  to children  
9: end for  
10: population  $\leftarrow$  children  
11: until  $k$  steps of genetic optimization

![](images/ac699b2b15c52440294117c2dbb0d49614abca50ab88e0791c9855ba2a5bf2dd.jpg)  
Figure 1: Crossover operator for neural network policies.

the network outputs the mean and the standard-deviation of each action in the action-space. Combining two DNN policies such that the final child policy possibly absorbs the best traits of both the parents is non-trivial. Figure 1 illustrates different crossover strategies. The figure includes neural network policies along with the state-visitation distribution plots (in a 2D space) corresponding to some high return rollouts using that policy. The two parent networks are shown in the top half of the figure. The state-visitation distributions are made non-overlapping to indicate that the parents policies have good state-to-action mapping for disparate local regions of the state-space.

A naive approach is to do crossover in the parameter space (bottom-right in figure). In this approach, a DNN child policy is created by copying over certain edge weights from either of the parents. The crossover could be at the granularity of multiple DNN layers, a single layer of edges or even a single edge (e.g. NEAT(Stanley & Miikkulainen, 2002b)). However, this type of crossover is expected to yield a low-performance composition due to the complex non-linear interactions between policy-parameters and the expected policy return. For the same reason, the state-visitation distribution of the child doesn't hold any resemblance to that of either of the parents. The bottom-left part of the figure shows the outcome of an ideal crossover in state-space. The state-visitation distribution of

the child includes regions from both the parents, leading to better performance (in expectation) than either of them. In this work, we propose a new crossover operator that utilizes imitation learning to combine the best traits from both parents and generate a high-performance child or offspring policy. So this crossover is not done directly in the parameter space but in the behavior or the state visitation space. We quantify the effect of these two types of crossovers in Section 4 by mixing several DNN pairs and measuring the policy performance in a simulated environment.

Our second contribution is in utilizing policy gradient algorithms for mutation of neural network weights in lieu of the Gaussian perturbations used in prior work on evolutionary algorithms for policy search. Because of the randomness in rollout samples, the policy-gradient mutation operator also maintains sufficient genetic diversity in the population. This helps our overall genetic algorithm achieve similar or higher sample efficiency compared to the state-of-the-art policy gradient methods.

# 3.3 GENETIC OPERATORS

This section details the three genetic operators. We use different subscripts for different policies. The corresponding parameters of the neural network are sub-scripted with the same letter (e.g.  $\theta_{x}$  for  $\pi_{x}$ ). We also use  $\pi_{x}$  and  $\pi_{\theta_x}$  interchangeably.  $\bigcup_{i = 1}^{m}\pi_{i}$  represents an ensemble of  $m$  policies  $\{\pi_1,\dots ,\pi_m\}$ .

# 3.3.1 CROSSOVER  $(\pi_x,\pi_y)$

This operator mixes two input policies  $\pi_x$  and  $\pi_y$  in state-space and produces a new child policy  $\pi_c$ . The three policies have identical network architecture. The child policy is learned using a two-step procedure. Firstly, we train a two-level policy  $\pi_H(a|s) = \pi_S(h = x|s)\pi_x(a|s) + \pi_S(h = y|s)\pi_y(a|s)$  which, given an observation, first chooses between  $\pi_x$  and  $\pi_y$ , and then outputs the action of the chosen policy.  $\pi_S$  is a binary policy trained with the maximum likelihood objective (cross-entropy loss) on the recent rollout trajectories from both parents. This hierarchical reinforcement learning step acts a medium of knowledge transfer from the parents to the child. We use only high-reward trajectories from  $\pi_x$  and  $\pi_y$  as data samples for training  $\pi_S$  to avoid transfer of negative behavior. It is possible to further refine  $\pi_S$  by running a few iterations of any policy-gradient algorithm, but we find that the maximum likelihood approach works well in practice and can also avoid extra rollout samples. Next, to distill the information from  $\pi_H$  into a policy with the same architecture as the parents, we use imitation learning to train a child policy  $\pi_c$ . We use trajectories from  $\pi_H$  (expert) as supervised data and train  $\pi_c$  to predict the expert action under the state distribution induced by the expert. The surrogate loss for imitation learning is:

$$
L ^ {I M I T} \left(\theta_ {c}\right) = \mathbb {E} _ {s \sim d ^ {*}} \left[ K L \left[ \pi_ {c} (.. | s), \pi_ {H} (.. | s) \right] \right] \tag {1}
$$

where  $d^{*}$  is the state-visitation distribution induced by  $\pi_{H}$ . To avoid compounding errors due to state distribution mismatch between the expert and the student, we adopt the Dataset Aggregation (DAgger) algorithm (Ross et al., 2011). Our training dataset  $\mathcal{D}$  is initialized with trajectories from the expert. After iteration  $i$  of training, we sample some trajectories from the current student  $(\pi_c^{(i)})$ , label the actions in these trajectories using the expert and form a dataset  $\mathcal{D}_i$ . Training for iteration  $i + 1$  then uses  $\{\mathcal{D} \cup \mathcal{D}_1 \dots \cup \mathcal{D}_i\}$  to minimize the loss. This helps to achieve a policy that performs well under its own induced state distribution. The direction of KL-divergence in Equation 1 encourages high entropy in  $\pi_c$ , and empirically, we found this to be marginally better than the reverse direction. For policies with Gaussian actions, the KL has a closed form and therefore the surrogate loss is easily optimized using a first order method. In experiments, we found that this crossover operator is very efficient in terms of sample complexity, which only requires a small size of rollout samples.

# 3.3.2 MUTATE  $\left(\bigcup_{i = 1}^{m}\pi_{i}\right)$

This operator modifies (in parallel) each policy of the input policy ensemble by running some iterations of a policy gradient algorithm. The policies have different initial parameters and are updated with high-variance gradients estimated using rollout trajectories. This leads to sufficient genetic

diversity and good exploration of the state-space, especially in the initial rounds of GPO. For two popular policy gradient algorithms—PPO and A2C—the gradients for policy  $\pi_{i}$  are calculated as

$$
\nabla_ {\theta_ {i}} L ^ {P P O} (\theta_ {i}) = \hat {\mathbb {E}} _ {i, t} \left[ \frac {\nabla_ {\theta_ {i}} \pi_ {\theta_ {i}} (a _ {t} | s _ {t})}{\pi_ {\theta_ {i} ^ {(o l d)}} (a _ {t} | s _ {t})} \hat {A} _ {t} - \nabla_ {\theta_ {i}} \beta K L \big [ \pi_ {\theta_ {i} ^ {(o l d)}} (. | s _ {t}), \pi_ {\theta_ {i}} (. | s _ {t}) \big ] \right] \qquad (2)
$$

$$
\nabla_ {\theta_ {i}} L ^ {A 2 C} \left(\theta_ {i}\right) = \hat {\mathbb {E}} _ {i, t} \left[ \nabla_ {\theta_ {i}} \log \pi_ {\theta_ {i}} \left(a _ {t} \mid s _ {t}\right) \hat {A} _ {t} \right] \tag {3}
$$

where  $\hat{\mathbb{E}}_{i,t}[\ldots]$  indicates the empirical average over a finite batch of samples from  $\pi_i$ , and  $\hat{A}_t$  is the advantage. We use an MLP to model the critic baseline  $V^{\pi}(s_t)$  for advantage estimation. PPO does multiple updates on the policy  $\pi_{\theta_i}$  using the same batch of data collected using  $\pi_{\theta_i^{(old)}}$ , whereas A2C does only a single update.

During mutation, a policy  $\pi_{i}$  can also use data samples from other similar policies in the ensemble for off-policy learning. A larger data-batch (generally) leads to a better estimate of the gradient and stabilizes learning in policy gradient methods. When using data-sharing, the gradients for  $\pi_{i}$  are

$$
\nabla_ {\theta_ {i}} L ^ {P P O} \left(\theta_ {i}\right) = \left(\sum_ {j \in \mathbb {S} _ {i}} \hat {\mathbb {E}} _ {j, t} \left[ \frac {\nabla_ {\theta_ {i}} \pi_ {\theta_ {i}} \left(a _ {t} \mid s _ {t}\right)}{\pi_ {\theta_ {j} ^ {(o l d)}} \left(a _ {t} \mid s _ {t}\right)} \hat {A} _ {t} \right]\right) - \nabla_ {\theta_ {i}} \hat {\mathbb {E}} _ {i, t} \left[ \beta K L \left[ \pi_ {\theta_ {i} ^ {(o l d)}} \left(. | s _ {t}\right), \pi_ {\theta_ {i}} (. | s _ {t}) \right] \right] \tag {4}
$$

$$
\nabla_ {\theta_ {i}} L ^ {A 2 C} \left(\theta_ {i}\right) = \sum_ {j \in \mathbb {S} _ {i}} \hat {\mathbb {E}} _ {j, t} \left[ \frac {\nabla_ {\theta_ {i}} \pi_ {\theta_ {i}} \left(a _ {t} \mid s _ {t}\right)}{\pi_ {\theta_ {j} ^ {(o l d)}} \left(a _ {t} \mid s _ {t}\right)} \hat {A} _ {t} \right] \tag {5}
$$

where  $\mathbb{S}_i\equiv \{j\mid KL[\pi_i,\pi_j] <   \epsilon$  before the start of current round of mutation} contains similar policies to  $\pi_{i}$  (including  $\pi_{i}$ ).

# 3.3.3 SELECT  $\bigcup_{i = 1}^{m}\pi_{i}$  ,FITNESS-FN

Given a set of  $m$  policies and a fitness function, this operator returns a set of policy-couples for use in the crossover step. From all possible  $\binom{m}{2}$  couples, the ones with maximum fitness are selected. The fitness function  $f(\pi_x, \pi_y)$  can be defined according to two criteria, as below.

- Performance fitness as sum of expected returns of both policies, i.e.  $f(\pi_x, \pi_y) \stackrel{\mathrm{def}}{=} E_{\tau \sim \pi_x}[R(\tau)] + E_{\tau \sim \pi_y}[R(\tau)]$  
- Diversity fitness as KL-divergence between policies, i.e.  $f(\pi_x, \pi_y) \stackrel{\text{def}}{=} KL[\pi_x, \pi_y]$

While the first variant favors couples with high cumulative performance, the second variant explicitly encourages crossover between diverse (high KL divergence) parents. A linear combination provides a trade-off of these two measures of fitness that can vary during the genetic optimization process. In the early rounds, a relatively higher weight could be provided to KL-driven fitness to encourage exploration of the state-space. The weight could be annealed with rounds of Algorithm 1 for encouraging high-performance policies.

# 4 EXPERIMENTS

In this section, we conduct experiments to measure the efficacy and robustness of the proposed GPO algorithm on a set of continuous control benchmarks. We begin by describing the experimental setup and our policy representation. We then analyze the effect of our crossover operator. This is followed by learning curves for the simulated environments and comparison with baselines. We conclude the section with discussion on the quality of policies learned by GPO and scalability issues.

![](images/8a257c9cbed5c136230120eab5313fbf15df3a48353980b936135a4cd3480e3b.jpg)  
(a) Average episode reward for the child policies after state-space crossover (left) and parameter-space crossover (right), compared to the performance of the parents. All bars are normalized to the first parent in each crossover. Policies are trained on the HalfCheetah environment.

![](images/39631a6a47e926732180a024eedb88ae4ccd9477a9757224d5a294318b90e9bc.jpg)

Figure 2: Comparison of Crossover operators.  
![](images/b1aca50717e08b6be3f1a1c5a6d96e865305dc90cef7288a6b8ef16301821650.jpg)  
(b) State visitation distribution for high reward rollouts from policies trained on the HalfCheetah environment. From left to right - first parent, second parent, child policy using state-space crossover, child policy using parameter-space crossover. The number above each subplot is the average episode reward for 100 rollouts from the corresponding policy.

![](images/a8798456b62cc6eff90ce7f357b17e818fb347753d53c4c7680a4d6ea7726831.jpg)

![](images/d5bc79d56b505253b2b15c2730310ea32626d928397273c990daf46bbe194871.jpg)

![](images/4da284ae0d436df6b0f649b99d6d6072a21cd366bbbf96fe2163153cd5ffa0ec.jpg)

# 4.1 SETUP

All our experiments are done using the OpenAI rllab framework (Duan et al., 2016). We benchmark 9 continuous-control locomotion tasks based on the MuJoCo physics simulator  $^{1}$ . All our control policies are Gaussian, with the mean parameterized by a neural network of two hidden layers (64 hidden units each), and linear units for the final output layer. The diagonal co-variance matrix is learnt as a parameter, independent of the input observation, similar to (Schulman et al., 2015; 2017). The binary policy  $(\pi_S)$  used for crossover has two hidden layers (32 hidden units each), followed by a softmax. The value-function baseline used for advantage estimation also has two hidden layers (32 hidden units each). All neural networks use tanh as the non-linearity at the hidden units. We show results with PPO and A2C as policy gradient algorithms for mutation. PPO performs 10 steps of full-batch gradient descent on the policy parameters using the same collected batch of simulation data, while A2C does a single descent step. Other hyperparameters are in the Appendix.

# 4.2 CROSSOVER PERFORMANCE

To measure the efficacy of our crossover operator, we run GPO on the HalfCheetah environment, and plot the performance of all the policies involved in 8 different crossovers that occur in the first round of Algorithm 1. Figure 2a shows the average episode reward for the parent policies and their corresponding child. All bars are normalized to the first parent in each crossover. The left subplot depicts state-space crossover. We observe that in many cases, the child either maintains or improves on the better parent. This is in contrast to the right subplot where parameter-space crossover breaks the information structure contained in either of the parents to create a child with very low performance. To visualize the state-space crossover better, in Figure 2b we plot the state-visitation

![](images/7afbaadb6629c57456986361a4d77bb62f18476932ffb19696b881e88cbb9f5c.jpg)

![](images/7bd3c56a14c7117bea3063e756b77d78e491c602b52484e9f9b65fd11146fe3c.jpg)

![](images/7ddac8f09e2aed20a345334bd627543b0529e4e40131864fbd5122df27797743.jpg)

![](images/858d31d99f042404f276f72557ff5211d8f9253d609d7dbfac4051247f032223.jpg)

![](images/babe424c0fb2326351ae3180172752eba93d4fc7c8b77dce7696e86019e36068.jpg)

![](images/016ff55e51c20531c558f32be53aee0af22bfb4a75b8372e1c4ec37a2f4b3418.jpg)

![](images/a3fe3c4f6166d8e3a2898267aa22c0cf863c52897d119e363b0c995345907b69.jpg)  
Figure 3: Performance of GPO and baselines on MuJoCo environments using PPO.

![](images/ea7b7ab41c6d8f870aaca824336035ef7048eefdde00f9ed61732b2bb9c0fbb5.jpg)

![](images/bf143a63e0e2c331c649efa7e00684392d8c1450a18fa4b8dbe4f7147e650a68.jpg)

<table><tr><td></td><td>GPO</td><td>Single</td><td>Joint</td></tr><tr><td>Walker2d</td><td>1464.6 ± 93.42</td><td>540.93 ± 13.54</td><td>809.8 ± 156.53</td></tr><tr><td>HalfCheetah</td><td>2100.54 ± 151.58</td><td>1523.52 ± 45.02</td><td>1766.11 ± 104.37</td></tr><tr><td>HalfCheetah-hilly</td><td>1234.99 ± 38.72</td><td>661.49 ± 86.58</td><td>1033.44 ± 99.15</td></tr><tr><td>Hopper-hilly</td><td>893.69 ± 13.81</td><td>508.62 ± 16.47</td><td>904.87 ± 21.17</td></tr><tr><td>InvertedDoublePendulum</td><td>4647.95 ± 39.69</td><td>4705.72 ± 13.65</td><td>4539.98 ± 37.49</td></tr><tr><td>Ant</td><td>1337.75 ± 120.98</td><td>393.74 ± 18.94</td><td>1215.16 ± 31.18</td></tr><tr><td>Walker2d-hilly</td><td>1140.36 ± 146.69</td><td>467.37 ± 24.9</td><td>1044.77 ± 98.34</td></tr><tr><td>Swimmer</td><td>99.33 ± 0.14</td><td>96.29 ± 0.14</td><td>94.55 ± 3.6</td></tr><tr><td>Hopper</td><td>903.16 ± 99.37</td><td>457.1 ± 16.01</td><td>922.5 ± 61.01</td></tr></table>

Table 1: Mean and standard-error for final performance of GPO and baselines using PPO.

distribution for high reward rollouts from all policies involved in one of the crossovers. All states are projected from a 20 dimensional space (for HalfCheetah) into a 2D space by t-SNE (Maaten & Hinton, 2008). Notwithstanding artifacts due to dimensionality reduction, we observe that high reward rollouts from the child policy obtained with state-space crossover visit regions frequented by both the parents, unlike the parameter-space crossover (rightmost subplot) where the policy mostly meanders in regions for which neither of the parents have strong supervision.

# 4.3 COMPARISON WITH POLICY GRADIENT METHODS

In this subsection, we compare the performance of policies trained using GPO with those trained with standard policy gradient algorithms. GPO is run for 12 rounds (Algorithm 1) with a population size of 8, and simulates 8 million timesteps in total for each environment (1 million steps per candidate policy). We compare with two baselines which use the same amount of data. The first baseline algorithm, Single, trains 8 independent policies with policy gradient using 1 million timesteps each, and selects the policy with the maximum performance at the end of training. Unlike GPO, these policies do not participate in state-space crossover or interact in any way. The second baseline algorithm, Joint, trains a single policy with policy gradient using 8 million timesteps.

![](images/3f745593693c8b6aff007aa43d0c234fb8409dc0ae0187d23684bed899c5d01f.jpg)

![](images/1daf4546a969e9543efef3363e4f516e912690545274c5972fb5a3bd2feb0148.jpg)

![](images/a8181709f4b01c3803cf10e575f002aa04d6566c8f4b1e5a15e12ce86280affb.jpg)

![](images/446656e0891fadbc62da64becf147fd404f4a3dc9a34d74d79a4b463d4f92407.jpg)

![](images/dad2501c665ae8f7de493a8d348b9e7f131abc77e6087260c51821b3abfb2383.jpg)

![](images/0a7eb5c0da15f19136ac0854b9ca78b3ff38525d54717bfe8a1ed4caf93d9e5a.jpg)

![](images/6e27f045d766b2ffa6abd5a6cee01bf566f2c6a24fa9fe756b2d4393779a9b5a.jpg)  
Figure 4: Performance of GPO and baselines on MuJoCo environments using A2C.

![](images/0964d5ca59010ed7df0f55592ebcb6065108fd6a2648a037f4211353e90d3a5e.jpg)

![](images/d7b0aebfb3a4648e7ca073b75c517974e0d005be996449cd877dd771c7072b79.jpg)

<table><tr><td></td><td>GPO</td><td>Single</td><td>Joint</td></tr><tr><td>Walker2d</td><td>444.7 ± 69.39</td><td>233.98 ± 7.9</td><td>340.75 ± 33.59</td></tr><tr><td>HalfCheetah</td><td>1071.49 ± 179.98</td><td>956.84 ± 54.12</td><td>930.17 ± 123.83</td></tr><tr><td>HalfCheetah-hilly</td><td>719.39 ± 63.74</td><td>460.59 ± 43.2</td><td>434.48 ± 75.24</td></tr><tr><td>Hopper-hilly</td><td>279.11 ± 40.28</td><td>216.43 ± 39.78</td><td>240.26 ± 28.67</td></tr><tr><td>InvertedDoublePendulum</td><td>4589.9 ± 45.37</td><td>3545.43 ± 83.63</td><td>3802.8 ± 50.56</td></tr><tr><td>Ant</td><td>308.67 ± 49.63</td><td>182.61 ± 10.39</td><td>503.64 ± 42.01</td></tr><tr><td>Walker2d-hilly</td><td>289.51 ± 56.9</td><td>203.74 ± 12.77</td><td>263.49 ± 27.33</td></tr><tr><td>Swimmer</td><td>95.43 ± 0.11</td><td>93.43 ± 0.1</td><td>95.05 ± 0.05</td></tr><tr><td>Hopper</td><td>441.79 ± 47.21</td><td>421.06 ± 9.1</td><td>321.48 ± 30.92</td></tr></table>

Table 2: Mean and standard-error for final performance of GPO and baselines using A2C.

Both Joint and Single do the same number of gradient update steps on the policy parameters, but each gradient step in Joint uses 8 times the batch-size. For all methods, we replicate 8 runs with different seeds.

Figure 3 plots the moving average of per-episode reward when training with PPO as the policy gradient method for all algorithms. We observe that GPO achieves better performance than Single is almost all environments. Joint is a more challenging baseline since each gradient step uses a larger batch-size, possibly leading to well-informed, low-variance gradient estimates. Nonetheless, GPO reaches a much better score for environments such as Walker2D and HalfCheetah, and also their more difficult (hilly) versions. We think this is due to better exploration and exploitation by the nature of the genetic algorithm. The performance at the end of training is shown in Table 1. Results with A2C as the policy gradient method are in Figure 4 and Table 2. Here, GPO beats the baselines in all but one environments. In summary, these results indicate that, with the new crossover and mutation operators, genetic algorithms could be an alternative policy optimization approach that competes with the state-of-the-arts policy gradient methods.

![](images/9aea909ebde8601f1c308998644741996874234c96c96933596fdd74d7dfcf25.jpg)  
Figure 5: Final performance of the policy ensemble trained with GPO and Single on the HalfCheetah environment.

![](images/0e47df0cf3d1ecbe7f6ae107fefefcfe85ce3855581702f82dfd8f0d538e043f.jpg)  
Figure 6: Scaling by increasing the GPO population size in Walker2d environment.

# 4.4 ROBUSTNESS AND SCALABILITY

The SELECTION operator selects high-performing individuals for crossover in every round of Algorithm 1. Natural selection weeds out poorly-performing policies during the optimization process. In Figure 5, we measure the average episode reward for each of the policies in the ensemble at the final round of GPO. We compare this with the final performance of the 8 policies trained using the Single baseline. We conclude that the GPO policies are more robust. In Figure 6, we experiment with varying the population size for GPO. All the policies in this experiment use the same batch-size for the gradient steps and do the same number of gradient steps. Performance improves by increasing the population size suggesting that GPO is a scalable optimization procedure. Moreover, the MUTATE and CROSSOVER genetic operators lend themselves perfectly to multiprocessor parallelism.

# 5 CONCLUSION

We presented Genetic Policy Optimization (GPO), a new approach to deep policy optimization which combines ideas from evolutionary algorithms and reinforcement learning. First, GPO does efficient policy crossover in state space using imitation learning. Our experiments show the benefits of crossover in state-space over parameter-space for deep neural network policies. Second, GPO mutates the policy weights by using advanced policy gradient algorithms instead of random perturbations. We conjecture that the noisy gradient estimates used by policy gradient methods offer sufficient genetic diversity, while providing a strong learning signal. Our experiments on several MuJoCo locomotion tasks show that GPO has superior performance over the state-of-the-art policy gradient methods and achieves comparable or higher sample efficiency. Future advances in policy gradient methods and imitation learning will also likely improve the performance of GPO for challenging RL tasks.

# REFERENCES

Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning, pp. 1329-1338, 2016.  
Danijar Hafner, James Davidson, and Vincent Vanhoucke. Tensorflow agents: Efficient batched reinforcement learning in tensorflow. arXiv preprint arXiv:1709.02878, 2017.  
Nikolaus Hansen and Andreas Ostermeier. Completely derandomized self-adaptation in evolution strategies. Evolutionary computation, 9(2):159-195, 2001.

Nicolas Heess, Srinivasan Sriram, Jay Lemmon, Josh Merel, Greg Wayne, Yuval Tassa, Tom Erez, Ziyu Wang, Ali Eslami, Martin Riedmiller, et al. Emergence of locomotion behaviours in rich environments. arXiv preprint arXiv:1707.02286, 2017.  
Verena Heidrich-Meisner and Christian Igel. Neuroevolution strategies for episodic reinforcement learning. Journal of Algorithms, 64(4):152-168, 2009.  
Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. Journal of Machine Learning Research, 17(39):1-40, 2016.  
Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(Nov):2579-2605, 2008.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.  
Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, pp. 1928-1937, 2016.  
Stéphane Ross, Geoffrey J Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In International Conference on Artificial Intelligence and Statistics, pp. 627-635, 2011.  
Tim Salimans, Jonathan Ho, Xi Chen, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. arXiv preprint arXiv:1703.03864, 2017.  
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning (ICML-15), pp. 1889-1897, 2015.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
Frank Sehnke, Christian Osendorfer, Thomas Rückstieß, Alex Graves, Jan Peters, and Jürgen Schmidhuber. Parameter-exploring policy gradients. *Neural Networks*, 23(4):551-559, 2010.  
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484-489, 2016.  
Kenneth O Stanley and Risto Miikkulainen. Efficient reinforcement learning through evolving neural network topologies. In Proceedings of the 4th Annual Conference on Genetic and Evolutionary Computation, pp. 569-577. Morgan Kaufmann Publishers Inc., 2002a.  
Kenneth O Stanley and Risto Miikkulainen. Evolving neural networks through augmenting topologies. Evolutionary computation, 10(2):99-127, 2002b.  
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction, volume 1.  
István Szita and András Lörincz. Learning tetris using the noisy cross-entropy method. Learning, 18(12), 2006.  
Shimon Whiteson. Evolutionary computation for reinforcement learning. In Reinforcement learning, pp. 325-355. Springer, 2012.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.

Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi. Target-driven visual navigation in indoor scenes using deep reinforcement learning. In Robotics and Automation (ICRA), 2017 IEEE International Conference on, pp. 3357-3364. IEEE, 2017.
