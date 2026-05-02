# ANALYZING THE ROLE OF TEMPORAL DIFFERENCING IN DEEP REINFORCEMENT LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Wide adoption of deep networks as function approximators in modern reinforcement learning (RL) is changing the research environment, both with regard to best practices and application domains. Yet, our understanding of RL methods has been shaped by theoretical and empirical results with tabular representations and linear function approximators. These results suggest that RL methods using temporal differencing (TD) are superior to direct Monte Carlo (MC) estimation. In this paper, we re-examine the role of TD in modern deep RL, using specially designed environments that each control for a specific factor that affects performance, such as reward sparsity, reward delay or the perceptual complexity of the task. When comparing TD with infinite horizon MC, we are able to reproduce the results from the past in modern settings characterized by perceptual complexity and deep nonlinear models. However, we also find that finite horizon MC methods are not inferior to TD, even in sparse or delayed reward tasks, making MC a viable alternative to TD. We discuss the role of perceptual complexity in reconciling these findings with classic empirical results.

# 1 INTRODUCTION

The use of deep networks as function approximators has significantly expanded the range of problems that can be successfully tackled with reinforcement learning (RL). However, there is little understanding of when and why certain deep reinforcement learning (DRL) algorithms work well. Theoretical results are mainly based on tabular environments or linear function approximators (Sutton & Barto, 2017). Their assumptions do not cover the typical application domains of DRL, which feature extremely high input dimensionality (typically in the tens of thousands) and the use of nonlinear function approximators. Thus, our understanding of DRL is based primarily on empirical results, and these empirical results guide the design of DRL algorithms.

One such design decision common to the vast majority of existing value-based DRL methods is the use of temporal difference (TD) learning – training predictive models by bootstrapping based on their own predictions. This design decision is primarily based on evidence from the pre-DRL era (Sutton, 1988; 1995). The results of those experimental studies are well-known and clearly demonstrate that simple supervised learning, also known as Monte Carlo (MC) prediction, is outperformed by pure TD learning, which, in turn, is outperformed by  $\mathrm{TD}(\lambda)$  – a method that can be seen as a mixture of TD with MC (Sutton, 1988).

However, recent research has shown (Dosovitskiy & Koltun, 2017) that an algorithm based on Monte Carlo prediction can outperform TD-based methods on complex sensorimotor control tasks in three-dimensional environments. These results suggest that the classic understanding of the relative performance of TD and MC may not hold in modern settings. This evidence is not conclusive: the algorithm proposed by Dosovitskiy & Koltun (2017) involves custom components such as parametrized goals and decomposed rewards, and therefore cannot be directly compared to TD-based baselines.

In this paper, we perform a controlled experimental study aiming at better understanding the role of temporal differencing in modern deep reinforcement learning, characterized by essentially infinite-dimensional state spaces, extremely high observation dimensionality, and deep nonlinear models used as function approximators. We focus on environments with visual inputs and discrete action sets, and algorithms that involve prediction of value or action-value functions. This is in contrast to value-free policy optimization algorithms (Schulman et al., 2015; Levine & Koltun, 2013) and

tasks with continuous action spaces and low-dimensional vectorial state representations that have been extensively benchmarked by Duan et al. (2016) and Henderson et al. (2017). We base our study on deep  $Q$ -learning (Mnih et al., 2015), where the  $Q$ -function is learned either via temporal differencing or via a finite-horizon Monte Carlo method. To ensure that our conclusions are not limited to pure value-based methods, we additionally evaluate asynchronous advantage actor-critic (A3C), which combines temporal differencing with a policy gradient method (Mnih et al., 2016).

Our main focus is on performing controlled experiments, both in terms of algorithm configurations and environment properties. This is in contrast to previous works, which typically benchmark a number of existing algorithms on a set of standard environments. While proper benchmarking is crucial for tracking the progress of the field, it is not always sufficient for understanding the reasons behind good or poor performance of the algorithms. In this work, we ensure that the algorithms are comparable by implementing them in a common software framework. By varying the parameters such as the balance between TD and MC in the learning update or the prediction horizon, we are able to clearly isolate the effect of these parameters on learning. Moreover, we designed a series of controlled scenarios that focus on specific characteristics of RL problems: reward sparsity, reward delay, perceptual complexity, and properties of terminal states. Results in these environments shed light on strengths and weaknesses of algorithms under investigation.

Our findings in modern DRL scenarios both support and contradict previous results on merits of TD. On the one hand, value-based infinite-horizon methods perform best in the regime which is a mixture of TD and MC, similar to the  $\mathrm{TD}(\lambda)$  results of Sutton (1988). On the other hand, in sharp contrast with previous belief, we observe that Monte Carlo algorithms can perform very well on challenging RL tasks. This is made possible by simply limiting the prediction to a finite horizon. Surprisingly, finite-horizon Monte Carlo training is successful in dealing with sparse and delayed rewards, which are generally assumed to impair this class of methods. Monte Carlo training is also more stable to noisy rewards and is particularly robust to perceptual complexity and variability.

# 2 PRELIMINARIES

We work in a standard reinforcement learning setting of an agent acting in an environment over discrete time steps. At each time step  $t$ , the agent receives an observation  $\mathbf{o}_t$  and selects an action  $\mathbf{a}_t$ . We assume partial observability: the observation  $\mathbf{o}_t$  need not carry complete information about the environment and can be seen as a function of the environment's "true state". We assume an episodic setup, where an episode starts with time step 0 and concludes at a terminal time step  $T$ . We denote by  $\mathbf{s}_t$  the tuple of all observations collected by the agent from the beginning of the episode:  $\mathbf{s}_t = \langle \mathbf{o}_0, \dots, \mathbf{o}_t \rangle$ . (In practice we will only include a set of recent observations in  $\mathbf{s}$ .) The objective is to find a policy  $\pi(\mathbf{a}_t | \mathbf{s}_t)$  that maximizes the expected return – the sum of all future rewards through the remainder of the episode:

$$
R _ {t} = \sum_ {i = t} ^ {T} r _ {i}. \tag {1}
$$

This sum can become arbitrarily large for long episodes. To avoid divergence, temporally distant rewards can be discounted. This is typically done in one of two ways: by introducing a discount factor  $\gamma$  or by truncating the sum after a fixed number of steps (horizon)  $\tau$ .

$$
R _ {t} ^ {\gamma} = \sum_ {i = t} ^ {T} \gamma^ {i - t} r _ {i} = r _ {t} + \gamma r _ {t + 1} + \gamma^ {2} r _ {t + 2} + \dots ; R _ {t} ^ {\tau} = \sum_ {i = t} ^ {t + \tau} r _ {i}. \tag {2}
$$

The parameters  $\gamma$  and  $\tau$  regulate the contribution of temporally distant rewards to the agent's objective. In what follows  $\hat{R}_t$  stands for  $R_t^\gamma$  or  $R_t^\tau$ .

For a given policy  $\pi$ , the value function and the action-value function are defined as expected returns that are conditioned, respectively, on the observation or the observation-action pair:

$$
V ^ {\pi} \left(\mathbf {s} _ {t}\right) = \mathbb {E} _ {\pi} \left[ \hat {R} _ {t} \mid \mathbf {s} _ {t} \right], \quad Q ^ {\pi} \left(\mathbf {s} _ {t}, \mathbf {a} _ {t}\right) = \mathbb {E} _ {\pi} \left[ \hat {R} _ {t} \mid \mathbf {s} _ {t}, \mathbf {a} _ {t} \right]. \tag {3}
$$

Optimal value and action-value functions are defined as the maxima over all possible policies:

$$
V ^ {\star} \left(\mathbf {s} _ {t}\right) = \max  _ {\pi} V ^ {\pi} \left(\mathbf {s} _ {t}\right), \quad Q ^ {\star} \left(\mathbf {s} _ {t}, \mathbf {a} _ {t}\right) = \max  _ {\pi} Q ^ {\pi} \left(\mathbf {s} _ {t}, \mathbf {a} _ {t}\right). \tag {4}
$$

In value-based, model-free reinforcement learning, the value or action value are estimated by a function approximator  $V$  with parameters  $\theta$ . The function approximator is typically trained by minimizing a loss between the current estimate and a target value:

$$
\mathcal {L} (\theta) = \left(V \left(\mathbf {s} _ {t}; \theta\right) - V _ {\text {t a r g e t}}\right) ^ {2}. \tag {5}
$$

The learning procedure for the action-value function is analogous. Hence, we focus on the value function in the remainder of this section.

Reinforcement learning methods differ in how the target value is obtained. The most straightforward approach is to use the empirical return as target: i.e.,  $V_{\mathrm{target}} = R_t^\gamma$  or  $V_{\mathrm{target}} = R_t^\tau$ . This is referred to as Monte Carlo (MC) training, since the empirical loss becomes a Monte Carlo estimate of the expected loss. Using the empirical return as target requires propagating the environment forward before a training step can take place – by  $\tau$  steps for finite-horizon return  $R_t^\tau$  or until the end of the episode for discounted return  $R_t^\gamma$ . This increases the variance of the target value for long horizons and large discount factors.

An alternative to Monte Carlo training is temporal difference (TD) learning (Sutton, 1988). The idea is to estimate the return by bootstrapping from the function approximator itself, after acting for a fixed number of steps  $n$ :

$$
V _ {\text {t a r g e t}} = \sum_ {i = t} ^ {t + n - 1} \gamma^ {i - t} r _ {i} + \gamma^ {n} V \left(\mathbf {s} _ {t + n}; \theta\right). \tag {6}
$$

TD learning is typically used with infinite-horizon returns. When the rollout length  $n$  approaches infinity (or, in practice, maximal episode duration  $T_{\mathrm{max}}$ ), TD becomes identical to Monte Carlo training. TD learning applied to the action-value function is known as  $Q$ -learning (Watkins, 1989; Watkins & Dayan, 1992; Peng & Williams, 1996; Mnih et al., 2015).

An alternative to value-based methods are policy-based methods, which directly parametrize the policy  $\pi (\mathbf{a}|\mathbf{s};\theta)$ . An approximate gradient of the expected return is computed with respect to the policy parameters, and the return is maximized using gradient ascent. Williams (1992) has shown that an unbiased estimate of the gradient can be computed as  $\nabla_{\theta}\log \pi (\mathbf{a}|\mathbf{s};\theta)\left(R_{t} - b_{t}(\mathbf{s}_{t})\right)$ , where the function  $b_{t}(\mathbf{s}_{t})$  is called a baseline and can be chosen so as to decrease the variance of the estimator. A common choice for the baseline is the value function:  $b_{t}(\mathbf{s}_{t}) = V^{\pi}(\mathbf{s}_{t})$ . A combination of policy gradient with a baseline value function learned via TD is referred to as an actor-critic method, with policy  $\pi$  being the actor and the value function estimator being the critic.

# 3 EXPERIMENTAL SETUP

# 3.1 ALGORITHMS

In our analysis of temporal differencing we focus on three key characteristics of RL algorithms. The first is the balance between TD and MC in the learning update. The second is the prediction horizon, in particular infinite versus finite horizon. The third is the use of pure value-based learning versus an actor-critic approach which includes an explicitly parametrized policy.

To study the first aspect, we use asynchronous n-step Q-learning (n-step  $Q$ ) (Mnih et al., 2016). In this algorithm, an action-value function is learned with n-step TD (Eq. (6)), and actions are selected greedily according to this function. By varying the rollout length  $n$ , we can smoothly interpolate between pure TD and pure MC updates. In order to analyze the second aspect, we implemented a finite-horizon Monte Carlo version of n-step  $Q$ , which we call  $Q_{MC}$ . This algorithm can be seen as a simplified version of Direct Future Prediction (Dosovitskiy & Koltun, 2017). Finally, we select asynchronous advantage actor-critic (A3C) (Mnih et al., 2016) to study the third aspect. In A3C, the value function estimate is learned with n-step TD, and a policy is trained with policy gradient. This allows us to evaluate the interplay of TD learning and policy gradient learning.

To ensure that the comparison is fully controlled and fair, we implemented all algorithms in the asynchronous training framework proposed by Mnih et al. (2016). Multiple actor threads are running in parallel and send the weight updates asynchronously to a parameter server. For A3C and n-step  $Q$ , we use the algorithms as described by Mnih et al. (2016).  $Q_{\mathrm{MC}}$  is the n-step  $Q$  algorithm where the n-step TD targets are replaced by finite-horizon MC targets.

Note that switching to finite horizon necessitates a small additional change in the  $Q_{\mathrm{MC}}$  algorithm. In practice, in n-step  $Q$  each parameter update is not just an  $n$ -step TD update, but a sum of all updates for rollouts from 1 to  $n$ . This improves the stability of training. In  $Q_{MC}$  such accumulation of updates is impossible, since predictions for different horizons are not compatible. We therefore always predict several  $Q$ -values corresponding to different horizons, similar to Dosovitskiy & Koltun (2017). Specifically, for horizon  $\tau = 2^K$ , we additionally predict  $Q$ -values for horizons  $\{2^k\}_{0 \leq k < K}$ . This design choice is further explained and supported with experiments in the supplement. Apart from this, there is no difference between n-step  $Q$  and  $Q_{MC}$ .

# 3.2 ENVIRONMENTS

To calibrate our implementations against results available in the literature, we begin by conducting experiments on several standard benchmark environments: five Atari games from the Arcade Learning Environment (Bellemare et al., 2013) and two environments based on first-person-view 3D simulation in the ViZDoom framework (Kempka et al., 2016). We used a set of Atari games commonly analyzed in the literature: Space Invaders, Pong, Beam Rider, Sea Quest, and Frostbite (Mnih et al., 2015; Schulman et al., 2015; Lake et al., 2016). For the ViZDoom environments, we used the Navigation, Battle and Battle2 scenarios from Dosovitskiy & Koltun (2017).

Our main experiments are on sequences of specialized environments. Each sequence is designed such that a single factor of variation is modified in a controlled fashion. This allows us to study the effect of this factor. Factors of variation include: reward sparsity, reward delay, reward type, and perceptual complexity. These environments form the heart of our analysis and will be publicly released for reproducibility.

For the controlled environments, we used the ViZDoom platform. This platform is compatible with existing map editors with built-in scripting, which allows for flexible and controlled specification of different scenarios. In comparison to Atari games, ViZDoom offers a more realistic setting with a three-dimensional environment and partially observed first-person navigation. We now briefly describe the tasks. Further details are provided in the supplement.

Basic health gathering. The basis for our controlled scenarios is the health gathering task. In this scenario, the agent's aim is to collect health kits while navigating through a maze using visual input. Figure 1(b) shows a typical image observed by the agent. The agent's health level is constantly declining. Health kits add to the health level. The goal is to survive and maintain as much health as possible by collecting health kits. To be precise, the agent loses 6 health units every 8 steps, and obtains 20 health units when collecting a health pack. The agent's total health cannot exceed 100. The reward is  $+1$  when the agent collects a health kit and 0 otherwise. There are 16 health kits in the labyrinth at any given time. When the agent collects one of them, a new one appears at a random location. An episode is terminated after 525 steps, which is equivalent to 1 minute of in-game time.

Terminal states. To test the effect of terminal states on the performance of the algorithms, we modified the health gathering scenario so that each episode terminates after  $m$  health kits are collected. For  $m = 1$ , all useful training signals come from the terminal state. With larger  $m$ , the importance of terminal states diminishes.

Delayed rewards. In this sequence of scenarios we introduce a delay between the act of collecting a health kit and its effect - an increase in health and a reward of 1. We have set up environments with delays of 2, 4, 8, 16, and 32 steps.

Sparse rewards. To examine the effect of reward sparsity, we varied the number of available health kits on the map. We created two variations of the basic health gathering environment with increasingly sparse rewards. In the 'Sparse' setting, there are 4 health kits in the labyrinth – four times fewer than in the basic setting. In the 'Very Sparse' setting, only 2 health kits are in the labyrinth – eight times fewer than in the basic setting. In order to isolate the effect of sparsity and keep the general difficulty of the task fixed, we accordingly adjusted the amount of health the agent loses per time period: 3 in the Sparse configuration and 2 in Very Sparse. In the Very Sparse scenario under random exploration, the agent gathers a health kit on average every 6,440 steps.

Reward type. In this scenario, we compare the standard binary reward with its more natural but more noisy counterpart. In the basic scenario above, the reward is  $+1$  for gathering a health kit and 0 otherwise. A more natural measure of success in the health gathering task is the actual change in

![](images/b75694490ba563b47ddca59362993c82b47971e6782ec0d1212de1139ace22a0.jpg)  
(a)

![](images/a66bea715addedc3912d11c8ec5914f09a82fb2dd1bc5ac5dbf5e395c9070f42.jpg)  
(b)

![](images/b5cefcddcf2bbda71fc6b8e6d4267726b0660423e18ddd9ad007f6be87ca5571.jpg)  
(c)  
Figure 1: Different levels of perceptual complexity in the health gathering task. (a) Map view of a grid world. (b) First-person view of a three-dimensional environment, fixed textures. (c) First-person view of a three-dimensional environment, random textures.

health. With this reward, the agent would directly aim to maximize its health. In this configuration we therefore use a scaled change in health as the reward signal. This reward is more challenging than the basic binary reward due to its noisiness (health is decreased only every eighth step) and the variance in the reward after collecting a health kit due to the total health limit.

Perceptual complexity. To analyze the effect of perceptual complexity, we designed variants of the health gathering task with different input representations. First, to increase the perceptual complexity of the task, we replaced the single maze used in the basic health gathering scenario by 90 randomly textured versions, some of which are shown in Figure 1(c). The labyrinth's texture is changed after each episode during both training and evaluation.

We also created two variants of the health gathering task with reduced visual complexity. These are the only controlled scenarios not using the ViZDoom framework. Both are based on a grid world, where the agent is navigating an  $8 \times 8$  room with 5 available actions: wait, up, down, left, and right. There are 4 randomly placed health kits in the room, and the aim of the agent is to collect these, with reward  $+1$  for collecting a health kit and 0 otherwise. Each time a health kit is collected a new one appears in a random location. The two variants differ in the representation that is fed to the agent. In one, the agent's input is a 10-dimensional vector that concatenates the 2D Cartesian coordinates of the agent itself and the 4 health kits, sorted by their distance to the agent. In the other variant, the agent gets as input an  $8 \times 8$  image of the grid world. This is a raster RGB image of a map of the grid world, as shown in Figure 1(a). This latter representation is compatible with the input representation used in the first-person variants (i.e., an RGB image), but the content of the input image is drastically simplified.

# 3.3 ALGORITHM DETAILS

We used identical network architectures for the three algorithms in all experiments. For experiments in Atari and ViZDoom domains we used deep convolutional networks similar to the one used by Mnih et al. (2015). For gridworld experiments we used fully-connected networks with three hidden layers. For  $Q_{\mathrm{MC}}$  and n-step  $Q$  we usedueling network architectures, similar to Wang et al. (2016). The exact architectures are specified in the supplement.

For experiments in Atari environments we followed a common practice and fed the 4 most recent frames to the networks. In all other environments the input was limited to the observation from the current time step. In ViZDoom scenarios, in addition to the observed image we fed a vector of measurements to all networks. The measurements are the agent's scalar health in the health gathering scenarios and a three-dimensional vector of the agent's health, ammo, and frags in the battle scenario.

We trained all models with 16 asynchronous actor threads, for a total of 60 million steps. We identified optimal hyperparameters for each algorithm via a hyperparameter search on a subset of environments and used these fixed hyperparameters for all environments, unless noted otherwise.

<table><tr><td rowspan="2"></td><td rowspan="2">#steps</td><td rowspan="2">Seaquest</td><td rowspan="2">S. Invaders</td><td rowspan="2">Atari Frostbite</td><td rowspan="2">Pong</td><td rowspan="2">BeamRider</td><td colspan="3">ViZDoom</td></tr><tr><td>Navigat.</td><td>Battle</td><td>Battle 2</td></tr><tr><td>A3C (Mnih et al., 2016)</td><td>80M</td><td>2300</td><td>2215</td><td>180</td><td>11.4</td><td>13236</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DFP (Dosovitskiy &amp; Koltun, 2017)</td><td>50M</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>84.1</td><td>33.5</td><td>16.5</td></tr><tr><td>QMC</td><td>60M</td><td>12708</td><td>1221</td><td>1311</td><td>-4.2</td><td>1839</td><td>84.4</td><td>35.9</td><td>17.5</td></tr><tr><td>20-step Q</td><td>60M</td><td>4276</td><td>1888</td><td>3875</td><td>8.9</td><td>9088</td><td>75.7</td><td>32.4</td><td>16.0</td></tr><tr><td>20-step A3C</td><td>60M</td><td>2021</td><td>1952</td><td>202</td><td>20.6</td><td>7190</td><td>70.8</td><td>22.1</td><td>11.0</td></tr></table>

Table 1: Calibration against published results on standard environments. We report the average score at the end of an episode for Atari games, health for the Navigation scenario, and frags for the Battle scenarios. In all cases, higher is better.

For evaluation, we trained three models on each task, selected the best-performing snapshot for each training run, and averaged the performance of these three best-performing snapshots. Further details are provided in the supplement.

# 4 RESULTS

# 4.1 CALIBRATION

We start by calibrating our implementations of the methods against published results reported in the literature. To this end, we train and test our implementations on standard environments used in prior work. The results are summarized in Table 1. Our implementations perform similarly to corresponding results reported in prior work.

For A3C the results are significantly different only for BeamRider. However, in Mnih et al. (2016) the evaluation used the average over the best 5 out of 50 experiments with different learning rates. We used the average over 3 runs with a fixed learning rate. Since the results for BeamRider have a high variance even for very small learning rate changes, this explains the difference between the results.

On the ViZDoom scenarios, the  $Q_{\mathrm{MC}}$  implementation performs on par with the DFP algorithm. This shows that DFP does not crucially depend on a decomposition of the reward into a vector of measurements, and can perform equally well given a standard RL setup with a scalar reward. Our A3C implementation achieves significantly better results than those reported by Dosovitskiy & Koltun (2017) on the ViZDoom scenarios. We attribute this to (a) using a rollout value of 20 in our experiments instead of 5 as used by Mnih et al. (2016) and Dosovitskiy & Koltun (2017), and (b) providing the measurements as input to the network. Dosovitskiy & Koltun (2017) have not tested DFP on Atari games. We find that in these environments  $Q_{\mathrm{MC}}$  performs worse overall than 20-step  $Q$  and 20-step A3C.

# 4.2 VARYING THE ROLLOUT IN TD-BASED ALGORITHMS

By changing the rollout length  $n$  in n-step  $Q$  and A3C, we can smoothly transition between TD and MC training. 1-step rollouts correspond to pure bootstrapping as used in the standard Bellman equation. Infinite rollouts (until the terminal state), on the other hand, correspond to pure Monte Carlo learning of discounted infinite-horizon returns.

Results on three environments - Basic health gathering, Sparse health gathering, and Battle - are presented in Figure 2. Rollout length of 20 is best on all tasks for n-step  $Q$ . Both very short and very long rollouts lead to decreased performance. These findings are in agreement with prior results of TD( $\lambda$ ) experiments (Sutton, 1988; 1995), considering that longer rollouts increase the MC portion of the value target, converging to a full MC update for infinite rollout. A mixture of TD and MC yields the best performance. The results for A3C are qualitatively similar, and again the 20-step rollout is overall near-optimal.

![](images/40f35cd5a88e183da4e22dbac1e54c9ca0e29eb8f9548adcbc05dc525bbe0dd3.jpg)  
Figure 2: Effect of rollout length on TD learning for n-step  $Q$  and A3C. We report average health at the end of an episode for health gathering and average frags in the Battle scenario. Higher is better.

![](images/c6f0971691767b0228e914d1c657d865f9e7d64767ab5dfc24af984124d61f0b.jpg)

![](images/1004157e18d05689222dfa85acdcd864a54573fc3a10f361ee353df24eb5588c.jpg)

# 4.3 CONTROLLED EXPERIMENTS

We now proceed to a series of controlled experiments on a set of specifically designed environments and compare TD-based methods to  $Q_{MC}$ , a purely Monte Carlo approach. The motivation is as follows. In the previous section we have seen that very long rollouts lead to deteriorated performance of n-step  $Q$  and A3C. This can be attributed to large variance in target values. The variance can be reduced by using a finite horizon, as is the case in  $Q_{\mathrm{MC}}$ . However, the use of a finite horizon means that rewards that are further away than the horizon will not be part of the value target, resulting in a disadvantage in tasks with sparse or delayed rewards. In order to evaluate this we run controlled experiments designed to isolate the reward delay, sparsity, and other factors. We test 20-step  $Q$  and A3C (optimal rollout for TD-based methods), 5-step  $Q$  and A3C (more TD in the update), and  $Q_{\mathrm{MC}}$  (finite horizon Monte Carlo).

Reward type. We contrast the standard binary reward with the more natural reward signal proportional to the change in the health level of the agent. Figure 3 (left) shows that in the scenario with binary reward the performance of  $Q_{\mathrm{MC}}$ , 20-step  $Q$ , and 20-step A3C is nearly identical, within  $4\%$  of each other. However, when trained with the noisier health-based reward,  $Q_{\mathrm{MC}}$  performs within  $1\%$  of the result with binary reward, but the performance of TD-based algorithms decreases significantly, especially for the 5-step rollouts. These results suggest that Monte Carlo training is more robust to noisy rewards than TD-based methods.

Terminal states. Table 2 shows that in environments where terminal states play a crucial role,  $Q_{\mathrm{MC}}$  is outperformed by TD-based methods. This is due to the finite-horizon nature of  $Q_{\mathrm{MC}}$ . A terminal reward only contributes to a single update per episode, while in TD it contributes to every update in the episode. If non-terminal rewards are present ( $m = 2$ ),  $Q_{\mathrm{MC}}$  approaches the TD-based algorithms, but still does not reach the performance of 20-step Q. Difficulties with terminal states can partially explain poor performance of  $Q_{\mathrm{MC}}$  on some Atari games.

<table><tr><td></td><td>m = 1</td><td>m = 2</td></tr><tr><td>QMC</td><td>43.3</td><td>64.0</td></tr><tr><td>20-step Q</td><td>75.9</td><td>75.5</td></tr><tr><td>5-step Q</td><td>74.3</td><td>71.3</td></tr><tr><td>20-step A3C</td><td>64.7</td><td>58.2</td></tr><tr><td>5-step A3C</td><td>61.1</td><td>52.3</td></tr></table>

Table 2: Terminal states.

Delayed rewards. Figure 3 (middle) shows that the performance

of all algorithms declines even with moderate delays in the reward signal. A delay of 2 steps, or approximately 0.2 seconds of in-game time, already leads to a  $8 - 12\%$  relative drop in performance for  $Q_{\mathrm{MC}}$  and 20-step TD algorithms and a  $30 - 40\%$  drop for 5-step TD algorithms. With a delay of 8 steps, or approximately 1 second, the performance of  $Q_{\mathrm{MC}}$  and 20-step TD algorithms drops by  $30 - 70\%$  and 5-step TD agents are essentially unable to survive until the end of an episode. With a delay of 32 steps, all algorithms degrade to a trivial score. Interestingly, the performance of  $Q_{\mathrm{MC}}$  declines less rapidly than the performance of the other algorithms and  $Q_{\mathrm{MC}}$  consistently outperforms the other algorithms in the presence of delayed rewards.

Sparse rewards. TD-based infinite-horizon approaches should theoretically be effective at propagating distal rewards, and are therefore supposed to be advantageous in scenarios with sparse rewards. The results on the Sparse and Very Sparse scenarios however, do not support this expectation (Figure 3 (right)):  $Q_{\mathrm{MC}}$  performs on par with 20-step  $Q$ , and noticeably better than 20-step A3C and 5-step algorithms. We believe the reason for the unexpectedly good performance of

![](images/1356ecf000f01a1041050623cee8b55d0888004a7fd9199f7c4b524e8db389d5.jpg)  
Figure 3: Effect of reward properties. Left to right: reward type, reward delay, reward sparsity. We report the average health at the end of an episode. Higher is better. MC training ( $Q_{MC}$ , green) performs well on all environments.

![](images/f70a436527508307ed9e56c4a392830f23f1c21d04e4ad47e3bacd1b186a7627.jpg)

![](images/15dcb141df2006f1d936e389cd1bd66244f27382a5939bdf16d19b977aaa2da1.jpg)

$Q_{\mathrm{MC}}$  is that Monte Carlo approaches are well suited for training perception systems, as discussed in more detail in Section 4.4. A video of a  $Q_{\mathrm{MC}}$  agent trained on the Very Sparse task is available at https://youtu.be/OJ1eBzW7cJ0.

Perceptual complexity. We test the algorithms on a series of environments of gradually increasing perceptual complexity. The results are summarized in Figure 4. In simple gridworld environments, the locations of the agent and the health kits are given to the agent directly in the form of their coordinates (Grid Vec.) or a map (Grid Map). In these scenarios, TD-based methods perform well. The Grid Vec. task is successfully solved by all methods. 5-step unrolling outperforms the 20-step versions and  $Q_{\mathrm{MC}}$  in both setups.

However, the situation is completely different in the Basic and Multi-texture setups, in which the perceptual input is much more complex. In the Basic setup, all methods perform roughly on par, but 5-step unrolling drops behind the other methods. In the Multi-texture setup,  $Q_{\mathrm{MC}}$  outperforms other algorithms.

To further analyze the effect of perception on DRL, we conduct an additional experiment where we separate the learning of perception and control. We first train two perception systems on the Battle task by predicting  $Q$ -values under a fixed policy with 20-step  $Q$  or  $Q_{MC}$ . We then re-initialize the weights in the top two layers, freeze the weights in the rest of the networks, and re-train the top two layers on the Battle task with 20-step  $Q$  or  $Q_{MC}$ . Further details are provided in the supplement. The results are shown in Table 3. Both  $Q$  and  $Q_{\mathrm{MC}}$  control reach higher score with a perception system trained with  $Q_{\mathrm{MC}}$ . This supports the hypothesis that Monte Carlo training is efficient at training deep perception systems from raw pixels.

<table><tr><td rowspan="2">Perception</td><td colspan="2">Control</td></tr><tr><td>20-step Q</td><td>QMC</td></tr><tr><td>20-step Q</td><td>18.0</td><td>19.9</td></tr><tr><td>QMC</td><td>31.8</td><td>35.2</td></tr></table>

Table 3: Separate training of perception and control on the Battle scenario. Higher is better.

![](images/a41fba6558a6bfb4507810a66a4913a0297435662b31b0194c48644cee871e18.jpg)  
Figure 4: Effect of perceptual complexity. Perceptual complexity increases from left to right. We report average cumulative reward per episode for grid worlds and average health at the end of the episode for ViZDoom-based setups.

![](images/6a0595d8f6dd91c13294bbb33e8541802d27cc5218c67ef2d7aca14f4d98b02a.jpg)

# 4.4 DISCUSSION

Temporal differencing methods are generally considered superior to Monte Carlo methods in reinforcement learning. This opinion is largely based on empirical evidence from domains such as gridworlds (Sutton, 1995), cart pole (Barto et al., 1983), and mountain car (Moore, 1990). Our results agree: in gridworlds and on Atari games we find that n-step  $Q$  learning outperforms  $Q_{\mathrm{MC}}$ . We further find, similar to the TD( $\lambda$ ) experiments from the past (Sutton, 1988), that a mixture of MC and TD achieves best results in n-step  $Q$  and A3C.

However, the situation changes in perceptually complex environments. In our experiments in immersive three-dimensional simulations, a finite-horizon MC method ( $Q_{\mathrm{MC}}$ ) matches or outperforms TD-based methods. Especially interesting are the results of the sparse reward experiments. Sparse problems are supposed to be specifically challenging for finite-horizon Monte Carlo estimation: in our Very Sparse setting, average time between health kits is 44 time steps when a human is controlling the agent. This exceeds  $Q_{\mathrm{MC}}$ 's finite prediction horizon of 32 steps, making it seemingly impossible for the algorithm to achieve nontrivial performance. Yet  $Q_{\mathrm{MC}}$  is able to keep up with the results of the 20-step  $Q$  algorithm and clearly outperforms A3C.

What is the reason for this contrast between classic findings and our results? We believe that the key difference is in the complexity of perception in immersive three-dimensional environments, which was not present in gridworlds and other classic problems, and is only partially present in Atari games. In immersive simulation, the agent's observation is a high-dimensional image that represents a partial view of a large (mostly hidden) three-dimensional environment. The dimensionality of the state space is essentially infinite: the underlying environment is specified by continuous surfaces in three-dimensional space. Memorizing all possible states is easy and routine in gridworlds and is also possible in some Atari games (Blundell et al., 2016), but is not feasible in immersive three-dimensional simulations. Therefore, in order to successfully operate in such simulations, the agent has to learn to extract useful representations from the observations it receives. Encoding a meaningful representation from rich perceptual input is where Monte Carlo methods are at an advantage due to the reliability of their training signals. Monte Carlo methods train on ground-truth targets, not "guess from a guess", as TD methods do (Sutton & Barto, 2017).

These intuitions are supported by our experiments. Figure 4 shows that increasing the perceptual difficulty of the health gathering scenario hurts the performance of  $Q_{\mathrm{MC}}$  less than it does the TD-based approaches. Table 3 shows that  $Q_{\mathrm{MC}}$  is able to learn a better perception network than 20-step  $Q$ . In Figure 3, 20-step TD algorithms perform better than their 5-step counterparts in all tested scenarios. Longer rollouts bring TD closer to MC, in agreement with our hypothesis.

# 5 CONCLUSION

For the past 30 years TD-based methods have dominated the field of reinforcement learning. Our experiments on a range of complex tasks in perceptually challenging environments show that in deep reinforcement learning finite-horizon MC can be a viable alternative to TD. We find that while TD is at advantage in tasks with simple perception, long planning horizons, or terminal rewards, MC training is more robust to noisy rewards, effective for training perception systems from raw sensory inputs, and surprisingly successful at dealing with delayed and sparse rewards. Thus, a key challenge that can be derived from our results is to find ways on how to combine the advantages of noise-free supervised MC learning with those of TD. We hope that our results will contribute to a set of best practices for deep reinforcement learning that are consistent with the empirical reality of modern application domains.

# REFERENCES

Andrew G Barto, Richard S Sutton, and Charles W Anderson. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE Transactions on Systems, Man, and Cybernetics, 13(5), 1983.  
Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. JAIR, 47, 2013.

Charles Blundell, Benigno Uria, Alexander Pritzel, Yazhe Li, Avraham Ruderman, Joel Z. Leibo, Jack Rae, Daan Wierstra, and Demis Hassabis. Model-free episodic control. arXiv:1606.04460, 2016.  
Alexey Dosovitskiy and Vladlen Koltun. Learning to act by predicting the future. In ICLR, 2017.  
Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep reinforcement learning for continuous control. In ICML, 2016.  
Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep reinforcement learning that matters. arXiv preprint arXiv:1709.06560, 2017.  
Michal Kempka, Marek Wydmuch, Grzegorz Runc, Jakub Toczek, and Wojciech Jaskowski. ViZ-Doom: A Doom-based AI research platform for visual reinforcement learning. In IEEE Conference on Computational Intelligence and Games, 2016.  
Brenden M. Lake, Tomer D. Ullman, Joshua B. Tenenbaum, and Samuel J. Gershman. Building machines that learn and think like people. Behavioral and Brain Sciences, 2016.  
Sergey Levine and Vladlen Koltun. Guided policy search. In ICML, 2013.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, et al. Human-level control through deep reinforcement learning. Nature, 518(7540), 2015.  
Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In ICML, 2016.  
Andrew William Moore. Efficient memory-based learning for robot control. Technical Report 209, University of Cambridge, Computer Laboratory, 1990.  
Jing Peng and Ronald J. Williams. Incremental multi-step Q-learning. Machine Learning, 1996.  
John Schulman, Sergey Levine, Pieter Abbeel, Michael I. Jordan, and Philipp Moritz. Trust region policy optimization. In ICML, 2015.  
Richard S. Sutton. Learning to predict by the methods of temporal differences. Machine Learning, 3, 1988.  
Richard S. Sutton. Generalization in reinforcement learning: Successful examples using sparse coarse coding. In NIPS, 1995.  
Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, 2nd edition, 2017.  
Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, and Nando de Freitas. *Dueling network architectures for deep reinforcement learning*. In ICML, 2016.  
Christopher J. C. H. Watkins. Learning from delayed rewards. PhD thesis, University of Cambridge, England, 1989.  
Christopher J. C. H. Watkins and Peter Dayan. Q-learning. Machine Learning, 8, 1992.  
Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. In Machine Learning, 1992.
