# NEURAL MAP: STRUCTURED MEMORY FOR DEEP REINFORCEMENT LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

A critical component to enabling intelligent reasoning in partially observable environments is memory. Despite this importance, Deep Reinforcement Learning (DRL) agents have so far used relatively simple memory architectures, with the main methods to overcome partial observability being either a temporal convolution over the past  $k$  frames or an LSTM layer. More recent work (Oh et al., 2016) has went beyond these architectures by using memory networks which can allow more sophisticated addressing schemes over the past  $k$  frames. But even these architectures are unsatisfactory due to the reason that they are limited to only remembering information from the last  $k$  frames. In this paper, we develop a memory system with an adaptable write operator that is customized to the sorts of 3D environments that DRL agents typically interact with. This architecture, called the Neural Map, uses a spatially structured 2D memory image to learn to store arbitrary information about the environment over long time lags. We demonstrate empirically that the Neural Map surpasses previous DRL memories on a set of challenging 2D and 3D maze environments and show that it is capable of generalizing to environments that were not seen during training.

# 1 INTRODUCTION

Memory is a crucial aspect of an intelligent agent's ability to plan and reason in partially observable environments. Without memory, agents must act reflexively according only to their immediate percepts and cannot execute plans that occur over an extended time interval. Recently, Deep Reinforcement Learning agents have been capable of solving many challenging tasks such as Atari Arcade Games (Mnih et al., 2015), robot control (Levine et al., 2016) and 3D games such as Doom (Lample & Chaplot, 2016), but successful behaviours in these tasks have often only been based on a relatively short-term temporal context or even just a single frame. On the other hand, many tasks require long-term planning, such as a robot gathering objects or an agent searching a level to find a key in a role-playing game.

Neural networks that utilized external memories have recently had an explosion in variety, which can be distinguished along two main axes: memories with write operators and those without. Writeless external memory systems, often referred to as "Memory Networks" (Sukhbaatar et al., 2015; Oh et al., 2016), typically fix which memories are stored. For example, at each time step, the memory network would store the past M states seen in an environment. What is learnt by the network is therefore how to access or read from this fixed memory pool, rather than what contents to store within it.

The memory network approach has been successful in language modeling, question answering (Sukhbaatar et al., 2015) and was shown to be a successful memory for deep reinforcement learning agents in complex 3D environments (Oh et al., 2016). By side-stepping the difficulty involved in learning what information is salient enough to store in memory, the memory network introduces two main disadvantages. The first disadvantage is that a potentially significant amount of redundant information could be stored. The second disadvantage is that a domain expert must choose what to store in the memory, e.g. for the DRL agent, the expert must set M to a value that is larger than the time horizon of the currently considered task.

On the other hand, external neural memories having write operations are potentially far more efficient, since they can learn to store salient information for unbounded time steps and ignore any other useless information, without explicitly needing any a priori knowledge on what to store. One

prominent research direction within write-based architectures has been neural memories based on the types of memory structures that are found in computers, such as tapes, RAM, and GPUs. In contrast to typical recurrent neural networks, these neural computer emulators have far more structured memories which follow many of the same design paradigms that digital computers have traditionally utilized. One such model, the Differentiable Neural Computer (DNC) (Graves et al., 2016) and its predecessor the Neural Turing Machine (NTM) (Graves et al., 2014), structure the architecture to explicitly separate memory from computation. The DNC has a recurrent neural controller that can access an external memory resource by executing differentiable read and write operations. This allows the DNC to act and memorize in a structured manner resembling a computer processor, where read and write operations are sequential and data is store distinctly from computation. The DNC has been used successfully to solve complicated algorithmic tasks, such as finding shortest paths in a graph or querying a database for entity relations.

Building off these previous external memories, we introduce a new architecture called the Neural Map, a structured memory designed specifically for reinforcement learning agents in 3D environments. The Neural Map architecture overcomes some of the shortcomings of the previously mentioned neural memories. First, it uses an adaptable write operation and so its size and computational cost does not grow with the time horizon of the environment as it does with memory networks. Second, we impose a particular inductive bias on the write operation so that it is 1) well suited to 3D environments where navigation is a core component of successful behaviours, and 2) uses a sparse write operation that prevents frequent overwriting of memory locations that can occur with NTMs and DNCs. To accomplish this, we structure a DNC-style external memory in the form of a 2-dimensional map, where each position in the map is a distinct memory.

To demonstrate the effectiveness of the neural map, we run it on a variety of 2D partially-observable maze-based environments and test it against LSTM and memory network policies. Finally, to establish its scalability, we run a Neural Map agent on a set of challenging 3D maze environments based on the video game Doom.

# 2 BACKGROUND

A Markov Decision Process (MDP) is defined as a tuple  $(\mathcal{S},\mathcal{A},\mathcal{T},\gamma ,\mathcal{R})$  where  $\mathcal{S}$  is a finite set of states,  $\mathcal{A}$  is a finite set of actions,  $\mathcal{T}(s^{\prime}|s,a)$  is the transition probability of arriving in state  $s^\prime$  when executing action  $a$  in initial state  $s,\gamma$  is a discount factor, and  $\mathcal{R}(s,a,s^{\prime})$  is the reward function of executing action  $a$  in state  $s$  and ending up at state  $s^\prime$ . We define a policy  $\pi (\cdot |s)$  as a mapping from a state  $s$  to a distribution over actions, where  $\pi (a_i|s)$  denotes the probability of action  $a_{i}$  given that we are in state  $s$ . The value of a policy  $V^{\pi}(s)$  is the expected discounted cumulative reward when starting from state  $s$  and sampling actions according to  $\pi$ , i.e.:  $V^{\pi}(s) = \mathbb{E}_{\pi}[\sum_{t = 0}^{\infty}\gamma^{t}R_{t}|s_{0} = s]$ .

An optimal value function, denoted  $V^{*}(s)$ , is the maximum value we can get from state  $s$  according to any policy, i.e.  $V^{*}(s) = \max_{\pi} V^{\pi}(s)$ . An optimal policy  $\pi^{*}$  is defined as a policy which achieves optimal value at each state, i.e.  $V^{\pi^{*}}(s) = V^{*}(s)$ . An optimal policy is guaranteed to exist (Sutton & Barto, 1998). The REINFORCE algorithm (Williams, 1992) iteratively updates a given policy  $\pi$  in the direction of the optimal policy. This update direction is defined by  $\nabla_{\pi} \log \pi(a_{t}|s_{t}) G_{t}$  with  $G_{t} = \sum_{k=0}^{\infty} \gamma^{k} R_{t+k}$  being the future cumulated reward for a particular episode rollout. The variance of this update is typically high but can be reduced by using a "baseline"  $b_{t}(s_{t})$ , which is a function of the current state. Therefore the baseline-augmented update equation is  $\nabla_{\pi} \log \pi(a_{t}|s_{t})(G_{t} - b_{t}(s_{t}))$ . The typically used baseline is the value function,  $b_{t}(s_{t}) = V^{\pi}(s_{t})$ . This combination of REINFORCE with value function baseline is commonly termed the "Actor-Critic" algorithm.

In this paper, we utilize Advantage Actor-Critic (A2C) (Mnih et al., 2016) with Generalized Advantage Estimation (Schulman et al., 2015), which can be seen as a specialization of the actor-critic framework when using deep networks to parameterize the policy and value function. The policy is a function of the state, parameterized as a deep neural network:  $\pi(a|s) = f_{\theta}(s,a)$ , where  $f$  is a deep neural network with parameter vector  $\theta$ .

# 3 NEURAL MAP

In this section, we will describe the details of the neural map. We assume we want our agent to act within some 2- or 3-dimensional environment. The neural map is the agent's internal memory

storage that can be read from and written to during interaction with its environment, but where the write operator is selectively limited to affect only the part of the neural map that represents the area where the agent is currently located. For this paper, we assume for simplicity that we are dealing with a 2-dimensional map. This can easily be extended to 3-dimensional or even higher-dimensional maps (i.e. a 4D map with a 3D sub-map for each cardinal direction the agent can face).

Let the agent's position be  $(x,y)$  with  $x\in \mathbb{R}$  and  $y\in \mathbb{R}$  and let the neural map  $M$  be a  $C\times H\times W$  feature block, where  $C$  is the feature dimension,  $H$  is the vertical extent of the map and  $W$  is the horizontal extent. Assume there exists some coordinate normalization function  $\psi (x,y)$  such that every unique  $(x,y)$  can be mapped into  $(x^{\prime},y^{\prime})$ , where  $x^{\prime}\in \{0,\dots ,W - 1\}$  and  $y^\prime \in \{0,\dots ,H - 1\}$ . For ease of notation, suppose in the sequel that all coordinates have been normalized by  $\psi$  into neural map space.

Let  $s_t$  be the current state embedding,  $M_t$  be the current neural map, and  $(x_t, y_t)$  be the current position of the agent. The Neural Map is defined by the following set of equations:

$$
\begin{array}{l} r _ {t} = \operatorname {r e a d} \left(M _ {t}\right), c _ {t} = \operatorname {c o n t e x t} \left(M _ {t}, s _ {t}, r _ {t}\right), \\ w _ {t + 1} ^ {(x _ {t}, y _ {t})} = w r i t e (s _ {t}, r _ {t}, c _ {t}, M _ {t} ^ {(x _ {t}, y _ {t})}), M _ {t + 1} = u p d a t e (M _ {t}, w _ {t + 1} ^ {(x _ {t}, y _ {t})}), \\ o _ {t} = \left[ r _ {t}, c _ {t}, w _ {t + 1} ^ {\left(x _ {t}, y _ {t}\right)} \right], \quad \pi_ {t} (a | s) = \operatorname {S o f t m a x} \left(f \left(o _ {t}\right)\right), \tag {1} \\ \end{array}
$$

where  $w_{t}^{(x_{t},y_{t})}$  represents the feature at position  $(x_{t},y_{t})$  at time  $t$ ,  $[x_1,\ldots ,x_k]$  represents a concatenation operation, and  $o_t$  is the output of the neural map at time  $t$  which is then processed by another deep network  $f$  to get the policy outputs  $\pi_t(a|s)$ . We will now separately describe each of the above operations in more detail:

Global Read Operation: The read operation passes the current neural map  $M_t$  through a deep convolutional network and produces a  $C$ -dimensional feature vector  $r_t$ . The global read vector  $r_t$  summarizes information about the entire map.

Context Read Operation: The context operation performs context-based addressing to check whether certain features are stored in the map. It takes as input the current state embedding  $s_t$  and the current global read vector  $r_t$  and first produces a query vector  $q_t$ . The inner product of the query vector and each feature  $M_t^{(x,y)}$  in the neural map is then taken to get scores  $a_t^{(x,y)}$  at all positions  $(x,y)$ . The scores are then normalized to get a probability distribution  $\alpha_t^{(x,y)}$  over every position in the map, also known as "soft attention" (Bahdanau et al., 2015). This probability distribution is used to compute a weighted average  $c_t$  over all features  $M_t^{(x,y)}$ . To summarize:

$$
q _ {t} = W [ s _ {t}, r _ {t} ], a _ {t} ^ {(x, y)} = q _ {t} \cdot M _ {t} ^ {(x, y)},
$$

$$
\alpha_ {t} ^ {(x, y)} = \frac {e ^ {a _ {t} ^ {(x , y)}}}{\sum_ {(w , z)} e ^ {a _ {t} ^ {(w , z)}}}, c _ {t} = \sum_ {(x, y)} \alpha_ {t} ^ {(x, y)} M _ {t} ^ {(x, y)}, \tag {2}
$$

where  $W$  is a weight matrix. The context read operation allows the neural map to operate as an associative memory: the agent provides some possibly incomplete memory (the query vector  $q_{t}$ ) and the operation will return the completed memory that most closely matches  $q_{t}$ . So, for example, the agent can query whether it has seen something similar to a particular landmark that is currently within its view.

Local Write Operation: Given the agent's current position  $(x_{t},y_{t})$  at time  $t$ , the write operation takes as input the current state embedding  $s_t$ , the global read output  $r_t$ , the context read vector  $c_t$  and the current feature at position  $(x_{t},y_{t})$  in the neural map  $M_t^{(x_t,y_t)}$  and produces, using a deep neural network  $f_w$ , a new C-dimensional vector  $w_{t + 1}^{(x_t,y_t)}$ . This vector functions as the new local write candidate vector at the current position  $(x_{t},y_{t})$ :  $w_{t + 1}^{(x_t,y_t)} = f_w([s_t,r_t,c_t,M_t^{(x_t,y_t)}])$

GRU-based Local Write Operation As previously defined, the write operation simply replaces the vector at the agent's current position with a new feature produced by a deep network. Instead of this hard rewrite of the current position's feature vector, we can use a gated write operation based on the recurrent update equations of the Gated Recurrent Unit (GRU) (Chung et al., 2014). Gated write operations have a long history in unstructured recurrent networks and they have shown a superior

ability to maintain information over long time lags versus ungated networks. The GRU-based write operation is defined as:

$$
r _ {t + 1} ^ {(x _ {t}, y _ {t})} = \sigma \left(W _ {r} \left[ s _ {t}, r _ {t}, c _ {t}, M _ {t} ^ {(x _ {t}, y _ {t})} \right]\right)
$$

$$
\hat {w} _ {t + 1} ^ {(x _ {t}, y _ {t})} = \mathrm {t a n h} (W _ {\hat {h}} [ s _ {t}, r _ {t}, c _ {t} ] + U _ {\hat {h}} (r _ {t + 1} ^ {(x _ {t}, y _ {t})} \odot M _ {t} ^ {(x _ {t}, y _ {t})}))
$$

$$
z _ {t + 1} ^ {(x _ {t}, y _ {t})} = \sigma \left(W _ {z} \left[ s _ {t}, r _ {t}, c _ {t}, M _ {t} ^ {(x _ {t}, y _ {t})} \right]\right)
$$

$$
w _ {t + 1} ^ {(x _ {t}, y _ {t})} = \left(1 - z _ {t + 1} ^ {(x _ {t}, y _ {t})}\right) \odot M _ {t} ^ {(x _ {t}, y _ {t})} + z _ {t + 1} ^ {(x _ {t}, y _ {t})} \odot \hat {w} _ {t + 1} ^ {(x _ {t}, y _ {t})},
$$

where  $x\odot y$  is the Hadamard product between vectors  $x$  and  $y$ ,  $\sigma (\cdot)$  is the sigmoid activation function and  $W_{*}$  and  $U_{*}$  are weight matrices. Using GRU terminology,  $r_{t + 1}^{(x_t,y_t)}$  is the reset gate,  $\hat{w}_{t + 1}^{(x_t,y_t)}$  is the candidate activation and  $z_{t + 1}^{(x_t,y_t)}$  is the update gate. By making use of the reset and update gates, the GRU-based update can modulate how much the new write vector should differ from the currently stored feature.

Map Update Operation: The update operation creates the neural map for the next time step. The new neural map  $M_{t + 1}$  is equal to the old neural map  $M_t$ , except at the current agent position  $(x_{t},y_{t})$  where the current write candidate vector  $w_{t + 1}^{(x_t,y_t)}$  is stored:

$$
M _ {t + 1} ^ {(a, b)} = \left\{ \begin{array}{l l} w _ {t + 1} ^ {\left(x _ {t}, y _ {t}\right)}, & \text {f o r} (a, b) = \left(x _ {t}, y _ {t}\right) \\ M _ {t} ^ {(a, b)}, & \text {f o r} (a, b) \neq \left(x _ {t}, y _ {t}\right) \end{array} \right. \tag {3}
$$

# 4 EGO-CENTRIC NEURAL MAP

A major disadvantage of the neural map as previously described is that it requires some oracle to provide the current  $(x,y)$  position of the agent. This is a difficult problem in and of itself, and, despite being well studied, it is far from solved. The alternative to using absolute positions within the map is to use relative positions. That is, whenever the agent moves between time steps with some velocity  $(u,v)$ , the map is counter-transformed by  $(-u, -v)$ , i.e. each feature in the map is shifted in the  $H$  and  $W$  dimensions. This will mean that the map will be ego-centric, i.e. the agent's position will stay stationary in the center of the neural map while the world as defined by the map moves around them. Therefore in this setup we only need some way of extracting the agent's velocity, which is typically a simpler task in real environments (for example, animals have inner ears and robots have accelerometers). Here we assume that there is some function  $\xi(u', v')$  that discretizes the agent velocities  $(u', v')$  so that they represent valid velocities within the neural map  $(u, v)$ . In the sequel, we assume that all velocities have been properly normalized by  $\xi$  into neural map space.

Let  $(pw,ph)$  be the center position of the neural map. The updated ego-centric neural map operations are shown below:

$$
\bar {M} _ {t} = \text {C o u n t e r T r a n s f o r m} \left(M _ {t}, \left(u _ {t}, v _ {t}\right)\right)
$$

$$
r _ {t} = \operatorname {r e a d} (\bar {M} _ {t}) \quad c _ {t} = \operatorname {c o n t e x t} (\bar {M} _ {t}, s _ {t}, r _ {t})
$$

$$
w _ {t + 1} ^ {(p w, p h)} = \operatorname {w r i t e} (s _ {t}, r _ {t}, c _ {t}, \overline {{M}} _ {t} ^ {(p w, p h)}) \quad M _ {t + 1} = \operatorname {e g o u p d a t e} (\overline {{M}} _ {t}, w _ {t + 1} ^ {(p w, p h)})
$$

$$
o _ {t} = \left[ r _ {t}, c _ {t}, w _ {t + 1} ^ {(p w, p h)} \right] \quad \pi_ {t} = \operatorname {S o f t m a x} (f (o _ {t}))
$$

Where  $\overline{M}_t$  is the current neural map  $M_t$  reverse transformed by the current velocity  $(u_t, v_t)$  so that the agents map position remains in the center  $(pw, ph)$ .

Counter Transform Operation: The CounterTransform operation transforms the current neural map  $M_{t}$  by the inverse of the agent's current velocity  $(u_{t}, v_{t})$ . Written formally:

$$
\overline {{M}} _ {t} ^ {(a, b)} = \left\{ \begin{array}{l l} M _ {t} ^ {(a - u, b - v)}, & \text {f o r} (a - u) \in \{1, \dots , W \} \wedge (b - v) \in \{1, \dots , H \} \\ 0, & \text {e l s e} \end{array} \right. \tag {4}
$$

While here we only deal with reverse translation, it is possible to handle rotations as well if the agent can measure its angular velocity.

![](images/15180ce8355bc8d28d05f9c40c6df1484df8ffbf4ff7cda9e25c79a33aab4b11.jpg)  
(a) 2D Maze

![](images/e74508110fd4b37f79a8c69edd943874d5db93ef55531c3df84ea6ccabc07a99.jpg)

![](images/9c57682a204af75d0ede74f5639205554e2779919edc9b06fbd7a1dc231404d9.jpg)  
(c) Green Torch  $\rightarrow$  Green Tower

![](images/c236539003aba3ce11cee0342dbcc3092fa113e1c0b3b583f43c43cd1de10b2c.jpg)

![](images/b046ac5820eff85fc17a9ed82f1573d94896096afed4535831f056e593cab340.jpg)

![](images/6a330bbf140eb25f03fe02915d4b0ac725248d182f8761ae0dbf1aa16b652250.jpg)  
Figure 1: Left: Images showing the 2D maze environment. The left side (Fig. 1a) represents the fully observable maze while the right side (Fig. 1b) represents the agent observations. The agent is represented by the yellow pixel with its orientation indicated by the black arrow within the yellow block. The starting position is always the topmost position of the maze. The red bounding box represents the area of the maze that is subsampled for the agent observation. In "Goal-Search", the goal of the agent is to find a certain color block (either red or teal), where the correct color is provided by an indicator (either green or blue). This indicator has a fixed position near the start position of the agent. Right: State observations from the "Indicator" Doom maze environment. The agent starts in the middle of a maze looking in the direction of a torch indicator. The torch can be either green (top-left image) or red (bottom-left image) and indicates which of the goals to search for. The goals are two towers which are randomly located within the maze and match the indicator color. The episode ends whenever the agent touches a tower, whereupon it receives a positive reward if it reached the correct tower, while a negative reward otherwise.

![](images/266d954ffdec960fdd44a3378e72f6f6b3f9adb7f1f7d6834db671982aeccba6.jpg)

![](images/df32fc13e2e48c8babe14fd4ae4d01348c41adec9e22c6015cde22aeb5a57e24.jpg)  
(d) Red Torch  $\rightarrow$  Red Tower

Map Egroupdate Operation: The egoupdate operation is functionally equivalent to the update operation except only the center position  $(pw,ph)$  is ever written to:

$$
M _ {t + 1} ^ {(a, b)} = \left\{ \begin{array}{l l} w _ {t + 1} ^ {(p w, p h)}, & \text {f o r} (a, b) = (p w, p h) \\ \overline {{M}} _ {t} ^ {(a, b)}, & \text {f o r} (a, b) \neq (p w, p h) \end{array} \right. \tag {5}
$$

# 5 EXPERIMENTS

To demonstrate the effectiveness of the Neural Map, we run it on 2D and 3D maze-based environments where memory is crucial to optimal behaviour. We compare to previous memory-based DRL agents, namely a simple LSTM-based agent which consists of a single pre-output LSTM layer as well as MemNN (Oh et al., 2016) agents.

# 5.1 2D GOAL-SEARCH ENVIRONMENT

The "Goal-Search" environment is adapted from Oh et al. (2016). Here the agent starts in a fixed starting position within some randomly generated maze with two randomly positioned goal states. It then observes an indicator at a fixed position near the starting state (i.e. the green tile at the top of the maze in Fig. 1a). This indicator will tell the agent which of the two goals it needs to go to (blue indicator  $\rightarrow$  teal goal, green indicator  $\rightarrow$  red goal). If the agent goes to the correct goal, it gains a positive reward while if it goes to the incorrect goal it gains a negative reward. Therefore the agent needs to remember the indicator as it searches for the correct goal state. In depth details of the 2D environment are given in Appendix B. The mazes during training are generated using a random generator. A held-out set of 1000 random mazes is kept for testing. This test set therefore represents maze geometries that have never been seen during training, and measure the agent's ability to generalize to new environments.

The first baseline agent we evaluate is a recurrent network with 128 LSTM units. The other baseline is the MQN, which is a memory-network-based architecture that performs attention over the past K states it has seen (Oh et al., 2016). Both LSTM and MQN models receive a one-hot encoding of the agent's current location, previous velocity, and current orientation at each time step, in order to make the comparison to the fixed-frame Neural Map fair. We test these baselines against several Neural Map architectures, with each architecture having a different design choice.

2D Goal-Search  

<table><tr><td rowspan="2">Agent</td><td colspan="3">Train</td><td colspan="3">Test</td></tr><tr><td>7-11</td><td>13-15</td><td>Total</td><td>7-11</td><td>13-15</td><td>Total</td></tr><tr><td>Random</td><td>41.9%</td><td>25.7%</td><td>38.1%</td><td>46.0%</td><td>29.6%</td><td>38.8%</td></tr><tr><td>LSTM</td><td>84.7%</td><td>74.1%</td><td>87.4%</td><td>96.3%</td><td>83.4%</td><td>91.4%</td></tr><tr><td>MQN-32</td><td>80.2%</td><td>64.4%</td><td>83.3%</td><td>95.9%</td><td>74.6%</td><td>87.4%</td></tr><tr><td>MQN-64</td><td>83.2%</td><td>69.6%</td><td>85.8%</td><td>96.5%</td><td>76.7%</td><td>88.3%</td></tr><tr><td>Neural Map (15x15)</td><td>92.4%</td><td>80.5%</td><td>89.2%</td><td>93.5%</td><td>87.9%</td><td>91.7%</td></tr><tr><td>Neural Map + GRU (15x15)</td><td>97.0%</td><td>89.2%</td><td>94.9%</td><td>97.7%</td><td>94.0%</td><td>96.4%</td></tr><tr><td>Neural Map + GRU (8x8)</td><td>94.9%</td><td>90.7%</td><td>95.6%</td><td>98.0%</td><td>95.8%</td><td>97.3%</td></tr><tr><td>Neural Map + GRU + Pos (8x8)</td><td>95.0%</td><td>91.0%</td><td>95.9%</td><td>98.3%</td><td>94.3%</td><td>96.5%</td></tr><tr><td>Neural Map + GRU + Pos (6x6)</td><td>90.9%</td><td>83.2%</td><td>91.8%</td><td>97.1%</td><td>90.5%</td><td>94.0%</td></tr><tr><td>Ego Neural Map + GRU (15x15)</td><td>94.6%</td><td>91.1%</td><td>95.4%</td><td>97.7%</td><td>92.1%</td><td>95.5%</td></tr><tr><td>Ego Neural Map + GRU + Pos (15x15)</td><td>74.6%</td><td>63.9%</td><td>78.6%</td><td>87.8%</td><td>73.2%</td><td>82.7%</td></tr></table>

Table 1: Results of several different agent architectures on the "Goal-Search" environment. The "train" columns represents the number of mazes solved (in %) when sampling from the same distribution as used during training. The "test" columns represents the number of mazes solved when run on a set of held-out maze samples which are guaranteed not to have been sampled during training.

The results are reported in Table 1. During testing, we extend the maximum episode length from 100 to 500 steps so that the agent is given more time to solve the maze. The brackets next to the model name represent the Neural Map dimensions of that particular model. From the results we can see that the Neural Map architectures solve the most mazes in both the training and test distributions compared to both LSTM and MQN baselines.

The results also demonstrate the effect of certain design decisions. One thing that can be observed is that using GRU updates adds several percentage points to the success rate ("Neural Map (15x15)" v.s. "Neural Map + GRU (15x15)"). We also tried downsampled Neural Maps, such that a pixel in the memory map represents several discrete locations in the environment. The Neural Map seems quite robust to this downsampling, with a downsampling of around 3 (6x6 v.s. 15x15) doing just a few percentage points worse, and still beating all baseline models. The 6x6 model has approximately the same number of memory cells as "MQN-32", but its performance is much better, showing the benefit of having learnable write operations. For the egocentric model, in order to cover the entire map we set the pixels to be  $2\mathrm{x}$  smaller in each direction, so each pixel is only a quarter of a pixel in the fixed-frame map. Even with this coarser representation, the egocentric model did similarly to the fixed frame one. We demonstrate an example of what the Neural Map learned to address using its context operator in Appendix E.

Finally, we tried adding the one-hot position encoding as a state input to the Neural Map, as is done for the baselines. We can see that there is a small improvement, but it is largely marginal, with the Neural Map doing a decent job of learning how to represent its own position without needing to be told explicitly. One interesting thing that we observed is that having the one-hot position encoding as an input to the egocentric map decreased performance, perhaps because it is difficult for the network to learn a mapping between fixed and egocentric frames.

Note that sometimes the percentage results are lower for the training distribution. This is mainly because the training set encompasses almost all random mazes except the fixed 1000 of the test set, thus making it likely that the agent sees each training map only once.

Beyond train/test splits, the results are further separated by maze size. This information reveals that the memory networks are hardest hit by increasing maze size with sometimes a  $20\%$  drop in success on 13-15 v.s. 7-11. This is perhaps unsurprising given the inherent fixed time horizon of memory networks, and further reveals the benefit of using write-based memories.

# 5.2 3D DOOM ENVIRONMENT DESCRIPTION

To demonstrate that our method can work in much more complicated 3D environments with longer time lags, we implemented three 3D maze environments using the ViZDoom (Kempka et al., 2016) API and a random maze generator. Examples of all three environments are given in Figure 2.

![](images/bd684cc1ab679dfa1eeb9f0ace877c6590258e72a438e4a6fd7fda98c715fc14.jpg)  
(a) Indicator

![](images/ce689c07cb927357ba2f65d8f23272afce4316be440fc2838d02875a453c8d08.jpg)  
Figure 2: Top-down views showing successful episodes in each of the 3 Doom maze tasks. The red lines indicate the path traveled by the agent. Indicator is shown in Fig. 2a, where the agent receives positive reward when entering the corresponding tower that matches the torch color it saw at the start of the episode and a negative reward otherwise. The episode terminates once the agent has reached a tower. Repeating, shown in Fig. 2b, has the same underlying mechanics except (1) the episode persists for  $T$  time steps regardless of towers entered and (2) the torch indicator is removed from the maze after the agent has reached a tower once. Therefore the agent needs to find the correct tower and then optimize its path to that tower. Minotaur shown in Fig. 2c requires the agent to reach the red goal and then return to the green goal that is at its starting position. Here the torch does not have any function. This fully-observable top-down view was not made available to the agent and is only used for visualization.

![](images/d71ff78b0620dfdb982f0eb272a4acf1f1117aa32bf083419607084274f668fb.jpg)  
(b) Repeating

![](images/a7f3d7e2f6242f68e5b6d52188114decef7f0941baa57c3146f3c08a827486c7.jpg)  
(c) Minotaur

Indicator Maze: The first environment is a recreation of the 2D indicator maze task, where an indicator is positioned in view of the player's starting state which is either a torch of red or green color. The goals are corresponding red/green towers that are randomly positioned throughout the maze that the player must locate.

Repeating Maze: The second environment is a variant of this indicator maze but whenever the player enters a goal state, it is teleported back to the beginning of the maze without terminating the episode (i.e. it retains its memory of the current maze). It gains a positive reward if it reaches the correct goal and a negative reward if it reaches the incorrect goal. After the first goal is reached, the correct indicator color is no longer displayed within the maze and a red indicator is displayed afterwards instead (regardless if the correct goal is green). An episode ends after a predetermined number of steps which depends on the maze size. The goal is therefore to find a path to the correct goal, and then optimize that path so that it can reach it as many times as possible.

Minotaur Maze: The third environment has the agent start in a fixed starting position next to the green tower, while the red tower is randomly placed somewhere in the maze. The agent receives a small positive reward if it reaches the red tower, and a larger positive reward if after reaching the red tower it returns to the green tower. Therefore the agent must efficiently navigate to the red goal while accurately remember its entire path it so that it can backtrack to the start.

All three environments used a RGB+D image of size 100x60 as input. We generate maze geometries randomly at train time but make sure to exclude a test set of 10 mazes for each size [4, 5, 6, 7, 8] (50 total). For these environments, we tested out four architectures (see Appendix C for more details on both environments and architectures):

Neural Map with Controller LSTM: Standard Neural Map with fixed frame addressing and GRU updates. We combine the neural map design with an LSTM that aggregates past state, read and context vectors and produces the query vector for the next time step's context read operation. See Appendix A for the modified Neural Map equations.

Ego Neural Map with Controller LSTM: Same as previous but with ego-centric addressing. The other difference is that the Ego Neural Map does not receive any positional input unlike the other 3 models, only receiving frame-by-frame ego-motion (quantized to a coarse grid).

LSTM: Single pre-output 256-dimensional LSTM layer.

FRMQN (Oh et al., 2016): Memory network with LSTM feedback. This design uses an LSTM to make recurrent context queries to the memory network database. In addition, for the memory network baselines we did not set a fixed k but instead let it access any state from its entire episode. This means no information is lost to the memory network, it only needs to process its history.

<table><tr><td rowspan="2" colspan="2">Agent Maze Size</td><td colspan="5">Indicator</td><td colspan="5">Repeating</td><td colspan="5">Minotaur</td></tr><tr><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td rowspan="2">LSTM</td><td>Acc</td><td>95.7</td><td>87.5</td><td>81.1</td><td>71.4</td><td>60.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>90.0</td><td>71.5</td><td>48.0</td><td>34.2</td><td>29.4</td></tr><tr><td>Rew</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>7.26</td><td>7.58</td><td>6.06</td><td>5.32</td><td>4.98</td><td>1.35</td><td>1.07</td><td>0.72</td><td>0.51</td><td>0.44</td></tr><tr><td rowspan="2">FRMQN</td><td>Acc</td><td>87.3</td><td>82.9</td><td>78.0</td><td>72.0</td><td>59.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>72.7</td><td>54.5</td><td>38.8</td><td>28.8</td><td>23.7</td></tr><tr><td>Rew</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.45</td><td>1.65</td><td>1.51</td><td>1.37</td><td>1.09</td><td>1.09</td><td>0.82</td><td>0.58</td><td>0.43</td><td>0.36</td></tr><tr><td>Controller</td><td>Acc</td><td>95.8</td><td>90.3</td><td>81.8</td><td>80.4</td><td>70.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>99.7</td><td>92.2</td><td>67.5</td><td>37.9</td><td>30.2</td></tr><tr><td>NMap</td><td>Rew</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>17.4</td><td>17.1</td><td>12.0</td><td>11.4</td><td>12.3</td><td>1.50</td><td>1.38</td><td>1.01</td><td>0.57</td><td>0.45</td></tr><tr><td>Controller</td><td>Acc</td><td>94.6</td><td>91.0</td><td>87.6</td><td>85.8</td><td>72.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>98.6</td><td>90.0</td><td>65.2</td><td>44.7</td><td>33.8</td></tr><tr><td>Ego-NMap</td><td>Rew</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>12.8</td><td>14.1</td><td>11.0</td><td>10.4</td><td>9.72</td><td>1.48</td><td>1.35</td><td>0.98</td><td>0.67</td><td>0.51</td></tr></table>

Table 2: Doom results on mazes not observed during training for the three tasks: Indicator, Repeating and Minotaur. Acc stands for Accuracy and Rew for Reward. Accuracy for Indicator means % of correct goals reached, while for Minotaur it means % of episodes where the agent successfully reached the goal and then backtracked to the beginning. Reward for Repeating is number of times correct goal was visited within the allotted time steps (+1 for correct goal, -1 for incorrect goal). Reward for Minotaur is +0.5 for reaching the goal and then +1.0 for backtracking to start after reaching goal (max episode reward is +1.5). We tested on maze sizes between [4,8] with 10 test mazes for each size. For each of the 50 total test mazes we ran 100 episodes with random goal locations and averaged the result.

The results are shown in Table 2. We can see that the Neural Map architectures work better than the baseline models, even though the memory network has access to its entire episode history at every time step. The ego-centric Neural Map beats the fixed frame map at Indicator, and gets similar performance on both Repeating and Minotaur environments, showing the ability of the Neural Map to function effectively even without global position information. It is possible that having a fixed frame makes path optimization easier, which would explain the larger rewards that the fixed-frame model got in the Repeating task. We also investigated whether the neural map is robust to localization noise, which would be the case in a real world setting where we do not have access to a localization oracle and must instead rely on an error-prone odometry or SLAM-type algorithm to do localization. These results are presented in Appendix D.

For the baselines, we can see that FRMQN has difficulty learning on Repeating, only reaching the goal on average once. This could be because the indicator is only shown before the first goal is reached and so afterwards it needs to remember increasingly longer time horizons. Furthermore, because the red indicator is always shown after the first goal is reached, it might be difficult for the model to learn to do retrieval since the original correct indicator must be indexed by time and not image similarity. The FRMQN also has difficulty on Minotaur, probably due to needing to remember and organize a lot of spatial information (i.e. what actions were taken along the path). For Indicator, the FRMQN does similarly to the LSTM. We can see that the spatial structure of the Neural Map aids in optimizing the path in Repeating, averaging 12 goal reaches even in the largest maze size.

# 6 RELATED WORK

Other than the straightforward architectures of combining an LSTM with Deep Reinforcement Learning (DRL) (Mnih et al., 2016; Hausknecht & Stone, 2015), there has also been work on using more advanced external memory systems with DRL agents to handle partial observability. Oh et al. (2016) used a memory network (MemNN) to solve maze-based environments similar to the ones presented in this paper. MemNN keeps the last  $M$  states in memory and encodes them into (key, value) feature pairs. It then queries this memory using a soft attention mechanism similar to the context operation of the Neural Map, except in the Neural Map the key/value features were written by the agent and aren't just a stored representation of the last  $M$  frames seen. Oh et al. (2016) tested a few variants of this basic model, including ones which combined both LSTM and memory-network style memories.

In contrast to memory networks, another research direction is to design recurrent architectures that mimic computer memory systems. These architectures explicitly separate computation and memory in a way analogous to a modern digital computer, in which some neural controller (akin to a CPU) interacts with an external memory (RAM). One recent model is similar to the Neural Map, called the Differentiable Neural Computer (DNC) (Graves et al., 2016), which combines a recurrent controller with an external memory system that allows several types of read/write access. In addition to defin

ing an unconstrained write operator (in contrast to the neural map's write location being fixed), the DNC has a selective read operation that reads out the memory either by content or in the order that it was written. While the DNC is more specialized to solving algorithmic problems, the Neural Map can be seen as an extension of this Neural Computer framework to 3D environments, with a specific inductive bias on its write operator that allows sparse writes. Recently work has also been done toward sparsifying the read and write operations of the DNC (Rae et al., 2016). This work was not focused on 3D environments and did not make any use of task-specific biases like agent location, but instead used more general biases like "Least-Recently-Used" memory addresses to force sparsity.

Gupta et al. (2017) designed a similar 2D map structured memory, with the aim to do robot navigation in 3D environments. These environments were based off image scans of real office buildings, and they were preprocessed into a grid-world by quantizing the possible positions and orientations the agent could assume. In contrast to our paper, which presents the Neural Map more as a general memory architecture for DRL agents, Gupta et al. (2017) focuses mainly on solving the task of robot navigation. More concretely, the task in these environments was to navigate to a goal state, with the goal position either stated semantically (find a chair) or stated in terms of the position relative to the robot's coordinate frame. Owing to this focus on navigation, they force their internal map representation (e.g.  $M_t$ ) to be a prediction of free space around the robot. Another key difference was that their formulation lacked a context addressing operation. Finally, their method used DAGGER (Ross et al., 2011), an imitation learning algorithm, to train their agent. Since Doom actions affect translational/rotational accelerations, training using imitation learning is more difficult since a search algorithm cannot be used as supervision. An interesting addition they made was the use of a multi-scale map representation and a Value Iteration network (Tamar et al., 2016) to do better path planning.

# 7 CONCLUSION

In this paper we developed a neural memory architecture that organizes the spatial structure of its memory in the form of a 2D map, and allows sparse writes to this memory where the memory address of the write is in a correspondence to the agent's current position in the environment. We showed its ability to learn, using a reinforcement signal, how to behave within challenging 2D and 3D maze tasks that required storing information over long time steps. The results demonstrated that our architecture surpassed baseline memories used in previous work. Additionally, we showed the benefit of certain design decisions made in our architecture: using GRU updates instead of hard writes, demonstrating that the ego-centric viewpoint does not diminish performance and that the Neural Map is robust to downsampling its memory. Finally, to show that our method can scale up to more difficult 3D environments, we implemented several new maze environments in Doom. Using a hybrid Neural Map + LSTM model, we were able to solve most of the scenarios at a performance higher than previous DRL memory-based architectures. Furthermore, we demonstrated the ability of the Neural Map to be robust to a certain level of drift noise in its localization estimate.

# REFERENCES

D. Bahdanau, K. H. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. In Proceedings of the 3rd International Conference on Learning Representations 2015, 2015.  
J. Chung, C. Gulcehre, K. Cho, and Y. Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014. URL https://arxiv.org/abs/1412.3555.  
A. Graves, G. Wayne, and I. Danihelka. Neural tuning machines. CoRR, abs/1410.5401, 2014. URL https://arxiv.org/abs/1410.5401.  
A. Graves, G. Wayne, M. Reynolds, T. Harley, I. Danihelka, A. Grabska-Barwiska, S. G. Colmenarejo, E. Grefenstette, T. Ramalho, J. Agapiou, A. P. Badia, K. M. Hermann, Y. Zwols, G. Ostrovski, A. Cain, H. King, C. Summerfield, P. Blunsom, K. Kavukcuoglu, and D. Hassabis. Hybrid computing using a neural network with dynamic external memory. Nature, 538: 471-476, 2016.

S. Gupta, J. Davidson, S. Levine, R. Sukthankar, and J. Malik. Cognitive mapping and planning for visual navigation. CoRR, abs/1702.03920, 2017. URL https://arxiv.org/abs/1702.03920.  
M. Hausknecht and P. Stone. Deep recurrent q-learning for partially observable mdps. CoRR, abs/1507.06527, 2015. URL https://arxiv.org/abs/1507.06527.  
Michal Kempka, Marek Wydmuch, Grzegorz Runc, Jakub Toczek, and Wojciech Jaskowski. ViZ-Doom: A Doom-based AI research platform for visual reinforcement learning. In IEEE Conference on Computational Intelligence and Games, pp. 341-348, Santorini, Greece, Sep 2016. IEEE.  
G. Lample and D. S. Chaplot. Playing fps games with deep reinforcement learning. CoRR, abs/1609.05521, 2016. URL https://arxiv.org/abs/1609.05521.  
Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. Journal of Machine Learning Research, 17(39):1-40, 2016.  
V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. Human-level control through deep reinforcement learning. Nature, 518:529-533, 2015.  
V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Harley, T. P. Lillicrap, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016.  
J. Oh, V. Chockalingam, S. Singh, and H. Lee. Control of memory, active perception, and action in mycraf. In Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016.  
J. W. Rae, J. J. Hunt, T. Harley, I. Danihelka, A. Senior, G. Wayne, A. Graves, and T. Lillicrap. Scaling memory-augmented neural networks with sparse reads and writes. CoRR, abs/1610.09027, 2016. URL https://arxiv.org/abs/1610.09027.  
Stéphane Ross, Geoffrey J Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In AISTATS, volume 1, pp. 6, 2011.  
John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. In Proceedings of the 4th International Conference on Learning Representations 2016, 2015.  
S. Sukhbaatar, A. Szlam, J. Weston, and R. Fergus. End-to-end memory networks. In Advances in Neural Information Processing Systems, pp. 2440-2448, 2015.  
R. Sutton and A. Barto. Reinforcement Learning: an Introduction. MIT Press, 1998.  
Aviv Tamar, Yi Wu, Garrett Thomas, Sergey Levine, and Pieter Abbeel. Value iteration networks. In Advances in Neural Information Processing Systems, pp. 2146-2154, 2016.  
R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3):229-256, 1992.
