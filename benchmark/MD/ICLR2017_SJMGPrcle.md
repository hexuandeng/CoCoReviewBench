# LEARNING TO NAVIGATE IN COMPLEX ENVIRONMENTS

Piotr Mirowski*, Razvan Pascanu*, Fabio Viola, Hubert Soyer, Andy Ballard, Andrea Banino, Misha Denil, Ross Goroshin, Laurent Sifre, Koray Kavukcuoglu, Dharshan Kumaran, Raia Hadsell

DeepMind London, UK

{piotrmirowski, razp, fviola, soyer, aybd, abanino, mdenil, goroshin, sifre, korayk, dkumaran, raia} @google.com

# ABSTRACT

Learning to navigate in complex environments with dynamic elements is an important milestone in developing AI agents. In this work we formulate the navigation question as a reinforcement learning problem and show that data efficiency and task performance can be dramatically improved by relying on additional auxiliary tasks to bootstrap learning. In particular we consider jointly learning the goal-driven reinforcement learning problem with an unsupervised depth prediction task and a self-supervised loop closure classification task. Using this approach we can learn to navigate from raw sensory input in complicated 3D mazes, approaching human-level performance even under conditions where the goal location changes frequently. We provide detailed analysis of the agent behaviour, its ability to localise, and its network activity dynamics. We then show that the agent implicitly learns key navigation abilities, through reinforcement learning with sparse rewards and without direct supervision.

# 1 INTRODUCTION

The ability to navigate efficiently within an environment is fundamental to intelligent animal behavior. Whilst conventional robotics methods, such as Simultaneous Localisation and Mapping (SLAM), tackle navigation through an explicit focus on position inference and mapping (Dissanayake et al., 2001), here we follow recent work in deep reinforcement learning (Mnih et al., 2015; 2016) and propose that navigational abilities will emerge as the by-product of an agent learning a policy that maximizes reward. One advantage of an intrinsic, end-to-end approach is that actions are not divorced from representation, but rather learnt together, thus ensuring that task-relevant features are present in the representation. Learning to navigate from reinforcement learning in partially observable environments, however, poses several challenges.

First, rewards are often sparsely distributed in the environment, where there may be only one goal location. Second, environments often comprise dynamic elements, requiring the agent to use memory at different timescales: rapid one-shot memory for the goal location (Section 4.1), together with short term memory subserving temporal integration of velocity signals and visual observations, and longer term memory for constant aspects of the environment (e.g. boundaries, cues).

We tackle the problem of sparse rewards by augmenting our loss with auxiliary tasks that provide denser training signals that support navigation-relevant representation learning. We consider two additional losses: the first involves reconstruction of a low-dimensional depth map at each time step, and can be seen as inferring depth from monocular imagery (Eigen et al., 2014). This unsupervised task concerns the 3D geometry of the environment, and is aimed to encourage the learning of representations that aid obstacle avoidance and short-term trajectory planning. The second task is

![](images/35578e48913ad7d4d82d603d2433d022e823a8a14d0d3085a1ae5989de1d9b87.jpg)

![](images/94569374804ae2581d3352fd0d2713b3cee022fe8a6170a771010d1a5d666885.jpg)  
Figure 1: Views from a small  $5 \times 10$  maze, a large  $9 \times 15$  maze and an I-maze, with corresponding maze layouts and sample agent trajectories. The mazes, which will be made public, have different textures and visual cues as well as exploration rewards and red goals.

![](images/926a2913e1be60e9f86765e1d094af1c777d67202e7df8bae389045fd0af1371.jpg)

![](images/6e5cb78de6d7d7e6eae4887e2f2ef1e8d4950921585830e0296837d29cd04b5e.jpg)

![](images/158f849669e142a91ed072582c853a5612aee3fe3b1becaeae41dc10bba92c7a.jpg)

![](images/8be0c0208cf742493c1bcfe4ce9a41cf349684049c02f45eece1440e1ac87eca.jpg)

self-supervised, and directly invokes loop closure from SLAM: the agent is trained to predict if the current location has been previously visited within a local trajectory. We show that the addition of these auxiliary tasks bootstraps the learning process and considerably increases data efficiency.

To address the requirement for memory over different timescales, we incorporate a variant of a stacked LSTM architecture (Graves et al., 2013; Pascanu et al., 2013): we believe that this allows one LSTM, which receives the reward signal and the representation constructed by the convolutional encoder as input, to rapidly update and maintain the currently relevant goal location (i.e. in environments where the goal changes frequently) and provide this as contextual input to a separate LSTM that also receives input from the encoder and additional velocity information, which dictates the policy.

To evaluate our approach, we use five 3D maze environments and demonstrate the accelerated learning and increased performance of the proposed agent architecture. These environments feature complex geometry, random start position and orientation, dynamic goal locations, and long episodes that require thousands of agent steps (see Figure 1). We also provide detailed analysis of the trained agent to show that critical navigation skills are acquired. This is important as neither position inference nor mapping are directly part of the loss; therefore, raw performance on the goal finding task is not necessarily a good indication that these skills are acquired. In particular, we show that the proposed agent resolves ambiguous observations and quickly localizes itself in a complex maze, and that this localization capability is correlated with higher task reward.

# 2 APPROACH

Our approach is an online, end-to-end learning framework that incorporates multiple objectives to train a neural network consisting of a convolutional encoder that feeds representations of the visual input to a stacked Long-Short Term Memory (LSTM) recurrent neural network. Specifically, the agent is trained jointly on three losses. Firstly it tries to maximize cumulative reward using reinforcement learning. Secondly it solves an unsupervised loss of inferring the depth map from the RGB observation. Finally, the agent is trained to detect loop closures as a self-supervised task that uses velocity integration.

The reinforcement learning problem is addressed with the Asynchronous Advantage Actor-Critic (A3C) algorithm (Mnih et al., 2016) that relies on learning both a policy  $\pi(a_t | s_t; \theta)$  and value function  $V(s_t; \theta_V)$  given a state observation  $s_t$ . Both the policy and value function share all intermediate representations, both being computed using a separate linear layer from the top most layer of the model. The advantage function  $A$  is used to estimate the policy gradient which updates model parameters  $\theta$  and  $\theta_V$ :

$$
A \left(s _ {t}, a _ {t}; \theta_ {V}\right) = \sum_ {i = 0} ^ {k - 1} \gamma^ {i} r _ {t + i} + \gamma^ {k} V \left(s _ {t + k}; \theta_ {V}\right) - V \left(s _ {t}; \theta_ {V}\right). \tag {1}
$$

The agent setup closely follows the work of (Mnih et al., 2016) and we refer to this work for the details (e.g. the use of a convolutional encoder followed by either a Multi-Layer Perceptron (MLP) or an LSTM, the use of action repetition, entropy regularization to prevent the policy saturation, etc.). These details can be found in the Appendix. We note that the main model used in our work is recurrent, though, for comparison, we also run a feedforward version of the agent.

![](images/2211ebe4497f025627fc15ab2433028c725b485afd011c06635f1add8016b002.jpg)  
Figure 2: Different architectures: (a) is a convolutional encoder followed by a feedforward layer and policy  $(\pi)$  and value function outputs; (b) has an LSTM layer; (c) uses additional inputs (agent-relative velocity, reward, and action), as well as a stacked LSTM; and (d) has additional outputs to predict depth and loop closures.

The baseline that we consider in this work is an A3C agent that receives only RGB input from the environment, using either a recurrent or a purely feed-forward model. To support the navigation capability of our approach, we expand the observations of the agents to include agent-relative velocity measurements. Additionally, the agent is provided with the action sampled from the stochastic policy and the immediate reward, from the previous time step. Thus, the observation  $s_t$  may include an image  $\mathbf{x}_t \in \mathbb{R}^{3 \times W \times H}$  (where  $W$  and  $H$  are the width and height of the image), the agent-relative lateral and rotational velocity  $\mathbf{v}_t \in \mathbb{R}^6$ , the previous action  $\mathbf{a}_{t-1} \in \mathbb{R}^{N_A}$ , and the previous reward  $r_{t-1} \in \mathbb{R}$ .

Figure 2 depicts four agent architectures that we consider. All use a three-layer convolutional encoder. Figure 2a shows a purely feedforward model, while 2b replaces the last linear layer with an LSTM. Figure 2c (Nav A3C) shows the A3C agent with augmented inputs and a stacked LSTM, where reward is input to the first layer, while the velocity and action are input to the second LSTM. Figure 2d (Nav A3C+D+L) shows additional losses, where depth may be predicted from the encoder features and loop closures may be predicted from the LSTM hidden units. The additional losses are computed on the current frame via an MLP from the last hidden state of the model, similarly to the policy and value functions, and from the output of the convolutional encoder. The agent is trained online by applying both the advantage actor-critic gradient update and the gradient updates from the depth and loop prediction, with the predictors scaled by respective coefficients  $\beta_{d}$  and  $\beta_{l}$ . More details of the online learning algorithm are given in Appendix B.

# 2.1 DEPTH PREDICTION

The primary input to the agent is in the form of RGB images. However, depth information, covering the field of view of the agent, might supply valuable information about the 3D structure of the environment. While depth could be directly used as an input, it has been shown that depth can be successfully predicted from single frames using a convolutional neural network (Eigen et al., 2014). Furthermore, the depth prediction loss provides more consistent gradients than those obtained from reward-based updates of the RL loss.

A mean square loss  $\mathcal{L}_d$  is used, scaled by a hyper-parameter  $\beta_{d}$  when combined with the RL update and the loop closure loss term. The predicted depth is a function of the convolutional output  $(\hat{d}_t = g_d(f_t))$ , where  $g_{d}$  is an MLP. To ensure that we converge quickly on the unsupervised loss and hence drive meaningful representation for the RL task, we use a coarse resolution for the depth map  $(8\times 16)$ . The MSE depth loss is expressed as  $\mathcal{L}_d = \frac{1}{2}\sum_t||\hat{d}_t - d_t||_2^2$ .

# 2.2 LOOP CLOSURE PREDICTION

Loop closure, like depth, is valuable for a navigating agent, since it signals that the agent has returned to an already visited location, and can be used for efficient exploration and spatial reasoning. As with depth, we hypothesize that loop closure prediction could be a valuable auxiliary loss to augment the reward-based training signal. Unlike depth prediction, the loop predictor is an MLP that operates on the output of the LSTM, since memory is required to predict the loop closure events.

To produce the training targets, we detect loop closures based on the similarity of local position information during an episode, which is obtained by integrating 2D velocity over time. Specifically, in a trajectory noted  $\{p_0,p_1,\dots ,p_T\}$ , where  $p_t$  is the position of the agent at time  $t$ , we define a loop closure label  $l_{t}$  that is equal to 1 if the position  $p_t$  of the agent is close to the position  $p_{t'}$  at an earlier time  $t'$ . In order to avoid trivial loop closures on consecutive points of the trajectory, we add an extra condition on an intermediary position  $p_{t''}$  being far from  $p_t$ . Thresholds  $\eta_{1}$  and  $\eta_{2}$  provide these two limits. Learning to predict the binary loop label is done by minimizing the Bernoulli loss  $\mathcal{L}_l$  between  $l_{t}$  and the output of a single-layer output from the hidden representation  $h_t$  of the last hidden layer of the model, followed by a sigmoid activation. We note  $g_{l}$  the MLP function applied to  $h_t$ . The scale of this loss is a hyper-parameter of the model, noted  $\beta_{l}$ . The loop closure loss is expressed as  $\mathcal{L}_l = \sum_t l_t g_l(h_t) + (1 - l_t)(1 - g_l(h_t))$ .

# 3 RELATED WORK

There is a rich literature on navigation, primarily in the robotics literature. However, here we focus on related work in deep RL. Deep Q-networks (DQN) have had breakthroughs in extremely challenging domains such as Atari (Mnih et al., 2015). Recent work has developed on-policy RL methods such as advantage actor-critic that use asynchronous training of multiple agents in parallel (Mnih et al., 2016). Recurrent networks have also been successfully incorporated to enable state disambiguation in partially observable environments (Koutnik et al., 2013; Hausknecht & Stone, 2015; Mnih et al., 2016; Narasimhan et al., 2015).

Deep RL has recently been used in the navigation domain. Kulkarni et al. (2016) used a feedforward architecture to learn deep successor representations that enabled behavioral flexibility to reward changes in the MazeBase gridworld, and provided a means to detect bottlenecks in 3D VizDoom. Zhu et al. (2016) used a feedforward siamese actor-critic architecture incorporating a pretrained ResNet to support navigation to a target in a discretised 3D environment. Oh et al. (2016) investigated the performance of a variety of networks with external memory (Weston et al., 2014) on simple navigation tasks in the Minecraft 3D block world environment. Tessler et al. (2016) also used the Minecraft domain to show the benefit of combining feedforward deep-Q networks with the learning of resuable skill modules (cf options: (Sutton et al., 1999)) to transfer between navigation tasks.

Auxiliary tasks have often been used to facilitate representation learning (Suddarth & Kergosien, 1990). Recently, the incorporation of additional objectives, designed to augment representation learning through auxiliary reconstructive decoding pathways (Zhang et al., 2016; Rasmus et al., 2015; Zhao et al., 2015; Mirowski et al., 2010), has yielded benefits in large scale classification tasks. In deep RL settings, however, only one previous paper has examined the benefit of auxiliary tasks. Specifically, Lample & Chaplot (2016) show that the performance of a DQN agent in a first-person shooter game in the VizDoom environment can be substantially enhanced by the addition of a supervised auxiliary task, whereby the convolutional network was trained on an enemy-detection task, with information about the presence of enemies, weapons, etc., provided by the game engine.

In contrast, our contribution addresses fundamental questions of how to learn an intrinsic representation of space, geometry, and movement while simultaneously maximising rewards through reinforcement learning. Our method is validated in challenging maze domains with random start and goal locations.

# 4 EXPERIMENTS

We consider a set of first-person 3D mazes called Labyrinth and based on OpenArena (see Fig. 1) that are visually rich, with additional observations available to the agent such as inertial information

![](images/c80b35364f18555db0ee0fdcae8a8f4257752b3ca41271c148fa17c6c305984c.jpg)  
(a) Static maze (small)

![](images/df7cd0aeef20218ca25fdbfc69222eba3c0bba51cce4e915c42b5ffd0c572919.jpg)  
(b) Static maze (large)

![](images/f3f9f43ce11981534ee7c25e5e334bbe385f44ab8d322609e1bff63bee30d21e.jpg)  
(c) Random Goal I-maze

![](images/7de457392e64cbd0e41739404c2ae6b222b89732b61bb965d76dad322e1619ca.jpg)  
(d) Random Goal maze (small)

![](images/57f679f88651c1831a1fbcc49b7048d11553797318870c9fa5a07e41e244361b.jpg)  
(e) Random Goal maze (large)

![](images/31448ad0e9b605eaaf27a7646867250ee9b57d597f5766fb8beac4910512530c.jpg)  
(f) Static maze: depth input vs target  
Figure 3: Rewards achieved by the agents (6 different architectures) on 5 different tasks: two static mazes (small and large) with fixed goals, two static maze with comparable layout but with dynamic goals and the I-maze. Results are averaged over the top 5 random hyperparameters for each agent-task configuration.

and local depth information. $^{1}$  The action space is discrete, yet allows finegrained control, comprising 8 actions: the agent can rotate in small increments, accelerate forward or backward or sideways, or induce rotational acceleration while moving. Reward is achieved in these environments by reaching a goal from a random start location and orientation. If the goal is reached, the agent is resurrected to a new start location and must return to the goal. The episode terminates when a fixed amount of time expires, typically affording the agent enough time to find the goal several times. There are additional 'fruit' rewards which are very sparse and serve to encourage exploration. Apples are worth 1 point, strawberries 2 points and goals are 10 points. Videos of the agent solving the maze are linked in Appendix A.

In the static variant of the maze, the goal and fruit locations are fixed and only the agent start location changes. In the dynamic (Random Goal) variant, the goal and fruits are randomly placed on every episode, and agent starts randomly, but the maze layout itself is static. Within an episode, the goal and apple locations stay fixed until the episode ends. This allows an explore-exploit strategy, where the agent should initially explore the maze to find the goal, then remember the location and quickly reacquire the goal after each respawn. For both variants (static and random goal) we consider a small and large map. The small mazes are  $5 \times 10$  and episodes last for 3600 timesteps, and the large mazes are  $9 \times 15$  with 10800 steps (see Figure 1). The RGB observation is  $84 \times 84$ .

The I-Maze environment (see Figure 1, right) is inspired by the classic T-maze used to investigate navigation in rodents (Olton et al., 1979): the layout remains fixed throughout, the agent spawns in the central corridor where there are apple rewards and has to locate the goal which is placed in the alcove of one of the four arms. Because the goal is hidden in the alcove, the optimal agent behaviour must rely on memory of the goal location in order to return to the goal using the most direct route. Goal location is constant within an episode but varies randomly across episodes.

The different agent architectures described in Section 2 are evaluated by training on five mazes. Figure 3 shows these learning curves. In each case we ran 64 experiments with randomly sampled hyper-parameters (for ranges and details please see the appendix). The mean over the top 5 runs as well as the top 5 curves are plotted. Expert human scores, established by a professional game player, are compared to these results in Table 1. The Nav A3C+D+L agents reach human-level performance on Static 1 and 2, and attain about  $80\%$  and  $50\%$  of human scores on Random Goal 1 and 2.

![](images/9df138d81f4312d78c8745df963812d104764c8fbae87a42a0072e22fcfbf791.jpg)  
Figure 4: left: Example of depth predictions (pairs of ground truth and predicted depths), sampled every 40 steps. right: Example of loop closure prediction. The agent starts at the gray square and the trajectory is plotted in gray. Blue dots correspond to true positive outputs of the loop closure detector; red cross correspond to false positives and green cross to false negatives. Note the false positives that occur when the agent is actually a few squares away from actual loop closure.

![](images/8035bac8f8c03a63eeefb2abc3475b0ca1ddd50c438cba1e85c1cdbf196a5c7d.jpg)

![](images/c6d09fb19f47c4289086d4a1a66fd0526bbc1f18ccceaf0ab4db1ab52de5b777.jpg)

We note some particular results from these learning curves. In Figure 3 (a and b), consider the feedforward A3C model (red curve) versus the LSTM version (pink curve). Even though navigation seems to intrinsically require memory, as single observations could often be ambiguous, the feedforward model achieves competitive performance on static mazes. This suggests that there might be good strategies that do not involve temporal memory and give good results, namely a reactive policy held by the weights of the encoder, or learning a wall-following strategy. We therefore introduce dynamic environments that encourage the use of memory and more general navigation strategies.

Figure 3 also shows the advantage of adding velocity, reward and action as an input, as well as the impact of using a two layer LSTM (orange curve vs red and pink). Though this agent (Nav A3C) is better than the simple architectures, it is still relatively slow to train on all of the mazes. We believe that this is mainly due to the slower, data inefficient learning that is generally seen in pure RL approaches. Supporting this we see that adding the auxiliary prediction targets of depth and loop closure (Nav A3C+D+L, blue curve) speeds up learning dramatically on most of the mazes (see Table 1: AUC metric). It has the strongest effect on the static mazes because of the accelerated learning, but also gives a substantial and lasting performance increase on the random goal mazes.

Although we place more value on the task performance than on the auxiliary losses, we report the results from the loop closure prediction task. Over 100 test episodes of 2250 steps each, within a large maze (random goal 2), the Nav A3C+D+L agent demonstrated very successful loop detection, reaching an F-1 score of 0.83. A sample trajectory can be seen in Figure 4 (right).

In order to disambiguate the effect of a depth prediction loss vs. simply adding depth as an input to the agent, we compare the performance of the Nav A3C+D agent to a Nav A3C where the visual input is RGBD instead of RGB. The comparison, in Figure 3f, shows that depth is much more useful for self supervision than as input to the agent.

# 5 ANALYSIS

# 5.1 POSITION DECODING

In order to evaluate the internal representation of location within the agent (either in the hidden units  $h_t$  of the last LSTM, or, in the case of the FF A3C agent, in the features  $f_t$  on the last layer of the conv-net), we train a position decoder that takes that representation as input, consisting of a linear classifier with multinomial probability distribution over the discretized maze locations. Small mazes  $(5 \times 10)$  have 50 locations, large mazes  $(9 \times 15)$  have 135 locations, and the I-maze has 77 locations. Note that we do not backpropagate the gradients from the position decoder through the rest of the network. The position decoder can only see the representation exposed by the model, not change it.

An example of position decoding by the Nav A3C+D+L agent is shown in Figure 6, where the initial uncertainty in position is improved to near perfect position prediction as more observations are acquired by the agent. We observe that position entropy spikes after a respawn, then decreases once the agent acquires certainty about its location. Additionally, videos of the agent's position decoding are linked in Appendix A. In these complex mazes, where localization is important for the purpose of reaching the goal, it seems that position accuracy and final score are correlated, as shown in Table 1. In Static 1, the best position decoding is obtained by the plain A3C agent (88.6% accuracy), whereas

<table><tr><td rowspan="2">Maze</td><td rowspan="2">Agent</td><td colspan="3">Mean over top 5 agents</td><td rowspan="2">Goals</td><td colspan="3">Highest reward agent</td></tr><tr><td>AUC</td><td>Score</td><td>% Human</td><td>Position Acc</td><td>Latency 1:&gt;1</td><td>Score</td></tr><tr><td rowspan="3">I-Maze</td><td>FF A3C</td><td>75.5</td><td>98</td><td>-</td><td>94/100</td><td>42.2</td><td>9.3s:9.0s</td><td>102</td></tr><tr><td>LSTM A3C</td><td>112.4</td><td>244</td><td>-</td><td>100/100</td><td>87.8</td><td>15.3s:3.2s</td><td>203</td></tr><tr><td>Nav A3C+D+L</td><td>169.7</td><td>266</td><td>-</td><td>100/100</td><td>68.5</td><td>10.7s:2.7s</td><td>252</td></tr><tr><td rowspan="3">Static 1</td><td>FF A3C</td><td>41.3</td><td>79</td><td>83.2</td><td>100/100</td><td>64.3</td><td>8.8s:8.7s</td><td>84</td></tr><tr><td>LSTM A3C</td><td>44.3</td><td>98</td><td>103.2</td><td>100/100</td><td>88.6</td><td>6.1s:5.9s</td><td>110</td></tr><tr><td>Nav A3C+D+L</td><td>82.9</td><td>101</td><td>106.3</td><td>100/100</td><td>86.0</td><td>7.2s:6.7s</td><td>104</td></tr><tr><td rowspan="3">Static 2</td><td>FF A3C</td><td>35.8</td><td>81</td><td>47.1</td><td>100/100</td><td>55.6</td><td>24.2s:22.9s</td><td>111</td></tr><tr><td>LSTM A3C</td><td>46.0</td><td>153</td><td>91.3</td><td>100/100</td><td>80.4</td><td>15.5s:14.9s</td><td>155</td></tr><tr><td>Nav A3C+D+L</td><td>110.3</td><td>168</td><td>98.8</td><td>100/100</td><td>91.4</td><td>12.3s:12.9s</td><td>182</td></tr><tr><td rowspan="3">Random Goal 1</td><td>FF A3C</td><td>37.5</td><td>61</td><td>57.5</td><td>88/100</td><td>51.8</td><td>11.0:9.9s</td><td>64</td></tr><tr><td>LSTM A3C</td><td>46.6</td><td>65</td><td>61.3</td><td>85/100</td><td>51.1</td><td>11.1s:9.2s</td><td>66</td></tr><tr><td>Nav A3C+D+L</td><td>55.2</td><td>79</td><td>74.5</td><td>100/100</td><td>78.7</td><td>10.0s:8.0s</td><td>87</td></tr><tr><td rowspan="3">Random Goal 2</td><td>FF A3C</td><td>50.0</td><td>69</td><td>40.1</td><td>93/100</td><td>30.0</td><td>27.3s:28.2s</td><td>77</td></tr><tr><td>LSTM A3C</td><td>37.5</td><td>57</td><td>32.6</td><td>74/100</td><td>33.4</td><td>21.5s:29.7s</td><td>51.3</td></tr><tr><td>Nav A3C+D+L</td><td>62.5</td><td>90</td><td>52.3</td><td>90/100</td><td>51.0</td><td>17.9s:18.4s</td><td>106</td></tr></table>

Table 1: Comparison of three agent architectures over five maze configurations, including random and static goals. AUC (Area under learning curve), Score, and % Human are averaged over the best 5 hyperparameters. Evaluation of a single best performing agent is done through analysis on 100 test episodes. Goals gives the number of episodes where the goal was reached one more more times. Position Accuracy is the classification accuracy of the position decoder. Latency  $1: > 1$  is the average time to the first goal acquisition vs. the average time to all subsequent goal acquisitions. Score is the mean score over the 100 test episodes.

![](images/49dacec4307be1a28148fb88b5bba8f4bfb70a5e77aef5d705e76cf049124a7b.jpg)  
Figure 5: Trajectories of the Nav A3C+D+L agent in the I-maze (left) and random goal maze 1 (right) over the course of one episode. At the beginning of the episode (gray curve on the map), the agent explores the environment until it finds the goal at some unknown location (red box). During subsequent respawns (blue path), the agent consistently returns to the goal. The value function, plotted for each episode, rises as the agent approaches the goal. Goals are plotted as vertical red lines.

![](images/aeec23aa87fd405448b4ac8f11c5f260c74d5c865bc490343ebcecccf515370e.jpg)

![](images/4f8eaa104a4c98325b79283fa42cfdaec1c43abb00c3d326fcc2eba780caa998.jpg)

![](images/37d30126d1e65b18aff11a359b4f35c6d5b38fcc211e5ce672ca24b15ef5d94a.jpg)

Nav A3C+D+L follow at  $86.0\%$  accuracy. A pure feed-forward architecture still achieves  $64.3\%$  accuracy in a static maze with static goal, suggesting that the encoder memorizes the position in the weights and that this small maze is solvable by all the agents, with sufficient training time. In Random Goal 1, it is Nav A3C+D+L that achieves the best position decoding performance ( $78.7\%$  accuracy), whereas the FF A3C and the LSTM A3C architectures are at approximately  $50\%$ .

In the I-maze, the opposite branches of the maze are nearly identical, with the exception of very sparse visual cues. We observe that once the goal is first found, the Nav A3C+D+L agent is capable of directly returning to the correct branch in order to achieve the maximal score. However, the linear position decoder for this agent is only  $68.5\%$  accurate, whereas it is  $87.8\%$  in the plain LSTM A3C agent. We hypothesize that the symmetry of the I-maze will induce a symmetric policy that need not be sensitive to the exact position of the agent (see analysis below).

![](images/1cee7b03ccc1b01eb6878418c6e5c6cf973c655eb5f195f5011e637189114a42.jpg)  
Figure 6: Trajectory of the Nav A3C+D+L agent in the random goal maze 1, overlaid with the position probability predictions predicted by a decoder trained on LSTM hidden activations, taken at 4 steps during an episode. Initial uncertainty gives way to accurate position prediction as the agent navigates.

![](images/91eae9193c3ef7bb0f08ab141814c5c7ccc45ae9376cf54699f808855ba33b68.jpg)

![](images/801172883900faea6d4915b865faabec7438eddb9126791a8a6bdb869b512c05.jpg)

![](images/546b5e0c9ca07bb0c40c468077d628a6ee20e7660078f8d6fcf5246503b1f9a6.jpg)

![](images/e7f3a0992d5de1ef849d7c1c8c8bc715ddeb72b35e483188d0dc848e5fbbff5c.jpg)  
(a) Agent trajectories for episodes with different goal locations

![](images/804a653f17d7c42d8719020661c3fc8ec69ce6da71da480ae69b8bf44cb2df7c.jpg)  
(b) LSTM activations from A3C agent

![](images/814d097e0a8cc949296241f8a64f593811817a5be1fdee4eabd0681e668ef194.jpg)  
Figure 7: LSTM cell activations of LSTM A3C and Nav A3C+D+L agents from the I-Maze collected over multiple episodes and reduced to 2 dimensions using tSNE, then coloured to represent the goal location. Policy-dictating LSTM of Nav A3C agent shown.

![](images/2ddabf3ae8eb139e0b6dc6d322db51530f1e4ccaea088a8bed0ab19e9c3549d9.jpg)  
(c) LSTM activations from Nav A3C+D+L agent

A desired property of navigation agents in our Random Goal tasks is to be able to first find the goal, and reliably return to the goal via an efficient route after subsequent re-spawns. The latency column in Table 1 shows that the Nav A3C+D+L agents achieve the lowest latency to goal once the goal has been discovered (the first number shows the time in seconds to find the goal the first time, and the second number is the average time for subsequent finds). Figure 5 shows clearly how the agent finds the goal, and directly returns to that goal for the rest of the episode. For Random Goal 2, none of the agents achieve lower latency after initial goal acquisition; this is presumably due to the larger, more challenging environment.

# 5.2 STACKED LSTM GOAL ANALYSIS

Figure 7(a) shows the trajectories traversed by an agent for each of the four goal locations. After an initial exploratory phase to find the goal, the agent consistently returns to the goal location. We visualize the agent's policy by applying tSNE dimension reduction (Maaten & Hinton, 2008) to the cell activations at each step of the agent for each of the four goal locations. Whilst clusters corresponding to each of the four goal locations are clearly distinct in the LSTM A3C agent, there are 2 main clusters in the Nav A3C agent – with trajectories to diagonally opposite arms of the maze represented similarly. Given that the action sequence to opposite arms is equivalent (e.g. straight, turn left twice for top left and bottom right goal locations), this suggests that the Nav A3C policy-dictating LSTM maintains an efficient representation of 2 sub-policies (i.e. rather than 4 independent policies) – with critical information about the currently relevant goal provided by the additional LSTM.

# 6 CONCLUSION

We proposed a deep RL method, augmented with memory and auxiliary learning targets, for training agents to navigate within large and visually rich environments that include frequently changing start and goal locations. Our results and analysis highlight the utility of un/self-supervised auxiliary objectives, namely depth prediction and loop closure, in providing richer training signals that bootstrap learning and enhance data efficiency. Further, we examine the behavior of trained agents, their ability to localise, and their network activity dynamics, in order to analyse their navigational abilities.

Our approach of augmenting deep RL with auxiliary objectives allows end-end learning and may encourage the development of more general navigation strategies. Notably, our work is related to (Jaderberg et al., 2017) which focuses on data efficiency by exploiting different auxiliary losses that are applicable in many RL settings. Our focus is on the navigation domain and understanding if navigation emerges as a bi-product of solving an RL problem.

Whilst our best performing agents are relatively successful at navigation, their abilities would be stretched if larger demands were placed on rapid memory (e.g. in procedurally generated mazes), due to the limited capacity of the stacked LSTM in this regard. It will be important for future work

to combine visually complex environments with architectures that make use of external memory (Graves et al., 2016; Weston et al., 2014; Olton et al., 1979) to enhance the navigational abilities of agents.

# ACKNOWLEDGEMENTS

We would like to thank Thomas Degris and Joseph Modayil for useful discussions, Charles Beattie, Julian Schrittwieser, Marcus Wainwright, and Stig Petersen for environment design and development, and Amir Sadik and Sarah York for expert human game testing.

# REFERENCES

MWM Gamini Dissanayake, Paul Newman, Steve Clark, Hugh F. Durrant-Whyte, and Michael Csorba. A solution to the simultaneous localization and map building (slam) problem. IEEE Transactions on Robotics and Automation, 17(3):229-241, 2001.  
David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. In Proc. of Neural Information Processing Systems, NIPS, 2014.  
Alex Graves, Mohamed Abdelrahman, and Geoffrey Hinton. Speech recognition with deep recurrent neural networks. In Proceedings of the International Conference on Acoustics, Speech and Signal Processing, ICASSP, 2013.  
Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, et al. Hybrid computing using a neural network with dynamic external memory. Nature, 2016.  
Matthew J. Hausknecht and Peter Stone. Deep recurrent q-learning for partially observable mdps. Proc. of Conf. on Artificial Intelligence, AAAI, 2015.  
Max Jaderberg, Volodymir Mnih, Wojciech Czarnecki, Tom Schaul, Joel Z. Leibo, David Silver, and Koray Kavukcuoglu. Reinforcement learning with unsupervised auxiliary tasks. In Submitted to Int'l Conference on Learning Representations, ICLR, 2017.  
Jan Koutnik, Giuseppe Cuccu, Jürgen Schmidhuber, and Faustino Gomez. Evolving large-scale neural networks for vision-based reinforcement learning. In Proceedings of the 15th annual conference on Genetic and evolutionary computation, GECCO, 2013.  
Tejas D. Kulkarni, Ardavan Saeedi, Simanta Gautam, and Samuel J. Gershman. Deep successor reinforcement learning. CoRR, abs/1606.02396, 2016. URL http://arxiv.org/abs/1606.02396.  
Guillaume Lample and Devendra Singh Chaplot. Playing FPS games with deep reinforcement learning. CoRR, 2016. URL http://arxiv.org/abs/1609.05521.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(Nov):2579-2605, 2008.  
Piotr Mirowski, Marc'Aurelio Ranzato, and Yann LeCun. Dynamic auto-encoders for semantic indexing. In NIPS Deep Learning and Unsupervised Learning Workshop, 2010.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, et al. Human-level control through deep reinforcement learning. Nature, 518:529-533, 2015.  
Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Proc. of Int'l Conf. on Machine Learning, ICML, 2016.  
Arun Nair, Praveen Srinivasan, Sam Blackwell, Cagdas Alcicek, Rory Fearon, et al. Massively parallel methods for deep reinforcement learning. In Proceedings of the International Conference on Machine Learning Deep Learning Workshop, ICML, 2015.  
Karthik Narasimhan, Tejas D. Kulkarni, and Regina Barzilay. Language understanding for text-based games using deep reinforcement learning. In Proc. of Empirical Methods in Natural Language Processing, EMNLP, 2015.

Junhyuk Oh, Valliappa Chockalingam, Satinder P. Singh, and Honglak Lee. Control of memory, active perception, and action in mycraft. In Proc. of International Conference on Machine Learning, ICML, 2016.  
David S Olton, James T Becker, and Gail E Handelmann. Hippocampus, space, and memory. Behavioral and Brain Sciences, 2(03):313-322, 1979.  
Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, and Yoshua Bengio. How to construct deep recurrent neural networks. arXiv preprint arXiv:1312.6026, 2013.  
Antti Rasmus, Mathias Berglund, Mikko Honkala, Harri Valpola, and Tapani Raiko. Semi-supervised learning with ladder networks. In Advances in Neural Information Processing Systems, NIPS, 2015.  
Steven C Suddarth and YL Kergosien. Rule-injection hints as a means of improving network performance and learning time. In Neural Networks, pp. 120-129. Springer, 1990.  
Richard S Sutton, Doina Precup, and Satinder Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. Artificial intelligence, 112(1):181-211, 1999.  
Chen Tessler, Shahar Givony, Tom Zahavy, Daniel J. Mankowitz, and Shie Mannor. A deep hierarchical approach to lifelong learning in apache. CoRR, abs/1604.07255, 2016. URL http://arxiv.org/abs/1604.07255.  
Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5 - rmsprop: Divide the gradient by a running average of its recent magnitude. In Coursera: Neural Networks for Machine Learning, volume 4, 2012.  
Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. arXiv preprint arXiv:1410.3916, 2014.  
Yuting Zhang, Kibok Lee, and Honglak Lee. Augmenting supervised neural networks with unsupervised objectives for large-scale image classification. In Proc. of International Conference on Machine Learning, ICML, 2016.  
Junbo Zhao, Michael Mathieu, Ross Goroshin, and Yann LeCun. Stacked what-where auto-encoders. Int'l Conf. on Learning Representations (Workshop), ICLR, 2015. URL http://arxiv.org/abs/1506.02351.  
Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J. Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi. Target-driven visual navigation in indoor scenes using deep reinforcement learning. CoRR, abs/1609.05143, 2016. URL http://arxiv.org/abs/1609.05143.
