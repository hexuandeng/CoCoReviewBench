# REINFORCEMENT LEARNING THROUGH ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC ON A GPU

Mohammad Babaeizadeh

Department of Computer Science

University of Illinois at Urbana-Champaign, USA

mb2@uiuc.edu

Iuri Frosio, Stephen Tyree, Jason Clemons, Jan Kautz

NVIDIA, USA

{ifrosio,styree,jclemons,jkautz}@nvidia.com

# ABSTRACT

We introduce a hybrid CPU/GPU version of the Asynchronous Advantage Actor-Critic (A3C) algorithm, currently the state-of-the-art method in reinforcement learning for various gaming tasks. We analyze its computational traits and concentrate on the critical aspects to leverage the GPU's computational power. We introduce a system of queues and a dynamic scheduling strategy, potentially helpful for other asynchronous algorithms as well. Our hybrid CPU/GPU version of A3C, based on TensorFlow, achieves a significant speed up compared to a CPU implementation and we will make it publicly available to other researchers.

# 1 INTRODUCTION

In the past, the need for task-specific, or even hand-crafted, features limited the application of Reinforcement Learning (RL) in real world problems (Sutton & Barto, 1998). However, the introduction of Deep Q-Learning Networks (DQN) (Mnih et al., 2015) revived the use of Deep Neural Networks (DNNs) as function approximators for value and policy functions, unleashing a rapid series of advancements. Remarkable results include learning to play video games from raw pixels (Bellemare et al., 2016; Lample & Singh Chaplot, 2016) and demonstrating super-human performance on the ancient board game Go (Silver et al., 2016). Research has yielded a variety of effective training formulations and DNN architectures (van Hasselt et al., 2015; Wang et al., 2015), as well as methods to increase parallelism while decreasing the computational cost and memory footprint (Nair et al., 2015; Mnih et al., 2016). In particular, Mnih et al. (2016) achieve state-of-the-art results on many gaming tasks through a novel lightweight, parallel method called Asynchronous Advantage Actor-Critic (A3C). When the proper learning rate is used, A3C learns to play an Atari game (Brockman et al., 2016) from raw screen inputs on a 16-core CPU, achieving higher scores than previously published methods which ran for the same amount of time on a GPU.

Our study sets aside many of the learning aspects of recent work and instead delves into the computational issues of deep RL. Computational complexities are numerous, largely centering on a common factor: RL has an inherently sequential aspect, since the training data are generated while learning. The DNN model is constantly queried to guide the actions of agents whose gameplay in turn feeds DNN training. Training batches are commonly small and must be efficiently shepherded from the agents and simulator to the DNN trainer. When using a GPU, the mix of small DNN architectures, small training batch sizes, and the competition for using the GPU for both inference and training, can lead to a severe under-utilization of the computational resources.

To systematically investigate these issues, we implement both CPU and GPU versions of A3C in TensorFlow (TF) (Abadi et al., 2015), optimizing each for efficient system utilization and to match published scores in the Atari 2600 environment (Brockman et al., 2016). We analyze a variety of "knobs" in the system and demonstrate effective automatic tuning of those during training. Our hybrid CPU/GPU implementation of A3C, named GA3C, learns substantially faster than its CPU counterpart, up to  $\sim 6\times$  faster for small DNNs and  $\sim 91\times$  for larger DNNs. While we focus on the

A3C architecture, we hope this analysis will be helpful for researchers and framework developers designing the next generation of deep RL methods.

# 2 RELATED WORK

Recent advances in deep RL have derived from both novel algorithmic approaches and related systems optimizations. Investigation of the algorithmic space seems to be the most common approach among researchers. Deep Q-Learning Networks (DQN) demonstrate a general approach to the learning problem (Mnih et al., 2015), relying heavily on the introduction of an experience replay memory to stabilize the learning procedure. This improves reliability but also increases the computational cost and memory footprint of the algorithm. Inspired by DQN, researchers have proposed more effective learning procedures, achieving faster and more stable convergence: Prioritized DQN (Schaul et al., 2015) makes better use of the replay memory by more frequently selecting frames associated with significant experiences. Double-DQN (van Hasselt et al., 2015) separates the estimate of the value function from the choice of actions (policy), thus reducing the tendency in DQN to be overly optimistic when evaluating its choices. Dueling Double DQN (Wang et al., 2015) goes a step further by explicitly splitting the computation of the value and advantage functions within the network. The presence of the replay memory makes the DQN approaches more suitable for a GPU implementation when compared to other LR methods, but state-of-the-art results are achieved by A3C (Mnih et al., 2016), which does not make use of it.

Among systems approaches, AlphaGo (Silver et al., 2016) recently achieved astonishing results through combined algorithmic and hardware specialization. The computational effort is impressive: 40 search threads, 1202 CPUs, and 176 GPUs are used in the distributed version for inference only. Supervised training took around three weeks for the policy network, using 50 GPUs, and another day using the RL approach for refinement. A similar amount of time was required to train the value network. Gorilla DQN (Nair et al., 2015) is a similarly impressive implementation of distributed RL system, achieving a significant improvement over DQN. The system requires 100 concurrent actors on 31 machines, 100 learners and a central parameter server with the network model. This work demonstrates the potential scalability of deep RL algorithms, achieving better results in less time, but with a significantly increased computational load, memory footprint, and cost.

# 3 ASYNCHRONOUS ADVANTAGE ACTOR CRITIC (A3C)

# 3.1 REINFORCEMENT LEARNING BACKGROUND

In standard RL, an agent interacts with an environment over a number of discrete time steps. At each time step  $t$ , the agent observes a state  $s_t$  and, in the discrete case, selects an action  $a_t$  from the set of valid actions. An agent is guided by policy  $\pi$ , a function mapping from states  $s_t$  to actions  $a_t$ . After each action, the agent observes the next state  $s_{t+1}$  and receives feedback in the form of a reward  $r_t$ . This process continues until the agent reaches a terminal state or time limit, after which the environment is reset and a new episode is played.

The goal of learning is to find a policy  $\pi$  that maximizes the expected reward. In policy-based model-free methods, a function approximator such as a neural network computes the policy  $\pi (a_{t}|s_{t};\theta)$  where  $\theta$  is the set of parameters of the function. There are many methods for updating  $\theta$  based on the rewards received from the environment. REINFORCE methods (Williams, 1992) use gradient ascent on  $\mathbb{E}[R_t]$ , where  $R_{t} = \sum_{i = 0}^{\infty}\gamma^{i}r_{t + i}$  is the accumulated reward starting from time step  $t$  and increasingly discounted at each subsequent step by factor  $\gamma \in (0,1]$ .

The standard REINFORCE method updates  $\theta$  using the gradient  $\nabla_{\theta}\log \pi (a_t|s_t;\theta)R_t$ , which is an unbiased estimator of  $\nabla_{\theta}\mathbb{E}[R_t]$ . The variance of the estimator is reduced by subtracting a learned baseline (a function of the state  $b_{t}(s_{t})$ ) and using the gradient  $\nabla_{\theta}\log \pi (a_t|s_t;\theta)\big(R_t - b_t(s_t)\big)$  instead. One common baseline is the value function defined as  $V^{\pi}(s_t) = \mathbb{E}[R_t|s_t]$  which is the expected return for following the policy  $\pi$  in state  $s_t$ . In this approach the policy  $\pi$  and the baseline  $b_{t}$  can be viewed as actor and critic in an actor-critic architecture (Sutton & Barto, 1998).

# 3.2 ASYNCHRONOUS ADVANTAGE ACTOR CRITIC (A3C)

A3C (Mnih et al., 2016), which achieves state-of-the-art results on many gaming tasks including Atari 2600, uses a single DNN to approximate both the policy and value function. The DNN has two convolutional layers with  $16 \times 8 \times 8$  filters with a stride of 4, and  $32 \times 4 \times 4$  filters with a stride of 2, followed by a fully connected layer with 256 units; each hidden layer is followed by a rectifier nonlinearity. The two outputs are a softmax layer which approximates policy function  $\pi(a_t | s_t; \theta)$  and a linear layer to output an estimate of  $V(s_t; \theta)$ . Multiple agents play concurrently and optimize the DNN through asynchronous gradient descent. Similar to other asynchronous methods, the network weights are stored in a central parameter server (Figure 1a). Agents calculate gradients and send updates to the server after every  $t_{max} = 5$  actions, or when a terminal state is reached. After each update, the central server propagates new weights to the agents to guarantee they share a common policy.

Two cost functions are associated with the two DNN outputs. For the policy function, this is:

$$
f _ {\pi} (\theta) = \log \pi \left(a _ {t} \mid s _ {t}; \theta\right) \left(R _ {t} - V \left(s _ {t}; \theta_ {t}\right)\right) + \beta H (\pi (s _ {t}; \theta)), \tag {1}
$$

where  $\theta_{t}$  are the values of the parameters  $\theta$  at time  $t$ ,  $R_{t} = \sum_{i=0}^{k-1} \gamma^{i} r_{t+i} + \gamma^{k} V(s_{t+k}; \theta_{t})$  is the estimated discounted reward in the time interval from  $t$  to  $t+k$  and  $k$  is upper-bounded by  $t_{max}$ , while  $H(\pi(s_{t}; \theta))$  is an entropy term, used to favor exploration during the training process. The factor  $\beta$  controls the strength of the entropy regularization term. The cost function for the estimated value function is:

$$
f _ {v} (\theta) = \left(R _ {t} - V (s _ {t}; \theta)\right) ^ {2}. \tag {2}
$$

Training is performed by collecting the gradients  $\nabla \theta$  from both of the cost functions and using the standard non-centered RMSProp algorithm (Tieleman & Hinton, 2012) as optimization:

$$
\begin{array}{l} g = \alpha g + (1 - \alpha) \Delta \theta^ {2} \\ \theta \leftarrow \theta - \eta \Delta \theta / \sqrt {g + \epsilon}. \end{array} \tag {3}
$$

The gradients  $g$  can be either shared or separated between agent threads but the shared implementation is known to be more robust (Mnih et al., 2016).

The original implementation of A3C (Mnih et al., 2016) uses 16 agents on a 16 core CPU and it takes about four days to learn how to play an Atari game (Brockman et al., 2016). The main reason for using CPU other than GPU, is the inherently sequential nature of RL in general, and A3C in particular. In RL, the training data are generated while learning, which means the training and inference batches are small and GPU is mostly idle during the training, waiting for new data to arrive. Since A3C does not utilize any replay memory, it is completely sequential and therefore a CPU implementation is as fast as a naive GPU implementation.

# 4 HYBRID CPU/GPU A3C (GA3C)

We propose GA3C, an alternative architecture of A3C, with emphasize on an efficient GPU utilization to shorten the training time. We demonstrate that our implementation of GA3C effectively converges significantly faster than our CPU implementation of A3C, achieving the state-of-the-art performance in a much shorter time.

# 4.1 GA3C ARCHITECTURE

The primary components of GA3C (Figure 1b) are a DNN with training and prediction on a GPU, as well as a multi-process, multi-thread CPU architecture with the following components:

- Agent is a process interacting with the simulation environment by actions chosen according to the learned policy and gathering experiences for further training. Similar to A3C, multiple concurrent agents run independent instances of the environment in GA3C. Unlike the original, each agent does not have its own copy of the model. Instead it queues policy requests in a Prediction Queue before each action, and periodically submits a batch of input/reward experiences to a Training Queue.  
- Predictor is a thread which dequeues as many prediction requests as are immediately available and batches them into a single inference query to the DNN model on the GPU. When predictions

![](images/5ab9979115e51b14b1f82412a6647530f7221e163384a1d23498b8e230c67b1c.jpg)  
(a) A3C

![](images/70363790a38dc3854354a02bab8efcde0eaf9740b1f270ad6dfaf112e8cdc384.jpg)  
(b) GA3C  
Figure 1: Comparison of A3C and GA3C architectures. Agents act concurrently both in A3C and GA3C. In A3C, however, each agent has a replica of the model, whereas in GA3C there is only one GPU instance of the model. In GA3C, agents utilize predictors to query the network for policies while trainers gather experiences for network updates.

are completed, the predictor returns the requested policy to each respective waiting agent. To hide latency, one or more predictors can act concurrently.

- Trainer is a thread which dequeues training batches submitted by agents and submits them to the GPU for model updates. While GPU utilization can be increased by grouping training batches among several agents, we found this can lead to slower convergence because fewer updates are applied to the network. Multiple trainers may run in parallel to hide latency.

# 4.2 PERFORMANCE METRICS AND TRADE-OFFS

The GA3C architecture exposes numerous tradeoffs for tuning its computational efficiency. In general, it is most efficient to transfer data to a GPU in large enough blocks to maximize the usage of the bandwidth between the GPU and CPU. Application performance on the GPU is optimized when the application has large amounts of parallel computations that can hide the latency of fetching data from memory. Thus, we want to maximize the parallel computations the GPU is performing, maximize the size of data transfer to the GPU, and minimize the number of transfers to the GPU. Increasing the number of predictors,  $N_P$ , allows faster fetching prediction queries, but leads to smaller prediction batches, resulting in multiple data transfers and overall lower GPU utilization. A larger number of trainers,  $N_T$ , potentially leads to more frequent updates to the model, but an overhead is paid when too many trainers occupy the GPU while predictors cannot access it. Lastly, increasing the number of agents,  $N_A$ , ideally generates more training experiences while hiding prediction latency. However, we would expect diminishing returns from unnecessary context switching overheads after exceeding some threshold depending on the number of CPU cores.

These aspects are well captured by a metric like the Trainings Per Second (TPS), which is the rate at which we remove batches from the training queue. It corresponds to the rate of model updates and it is therefore proportional to the overall learning speed, given a fixed learning rate and agent batch size. Another metric is the Predictions Per Second (PPS), the rate of issuing prediction queries from prediction queue, which maps to the combined rate of gameplay among all agents. Notice that in A3C a model update occurs every time an agent plays  $t_{max} = 5$  actions (Mnih et al., 2016). Hence, in a balanced configuration, PPS ≈ TPS ×  $t_{max}$ . Since each action is repeated four times as in (Mnih et al., 2016), the number of frames per second is  $4 \times \mathrm{PPS}$ .

Computational aspects are not disjointed from considerations regarding the convergence of the learning algorithm. For instance, when a large number of agents tend to fill the training queue, a significant time delay is introduced between the agent experiences  $(a_{t}, s_{t}$  and  $R_{t}$  in Eq. (1)) and the corresponding model updates, possibly threatening the model convergence (see also section 4.4). Another example is batching of the training data, which improves the GPU occupancy by increasing the parallelism, but at the same time may decrease the overall convergence speed as multiple updates are averaged, as we observed experimentally indeed. In short,  $N_{T}, N_{P}$ , and  $N_{A}$  encapsulate many

![](images/ed1165a0fdfabc4157bd76e8d0ceb795b91903c616e5d1c2ce1608dd840ca14d.jpg)  
Figure 2: Automatic dynamic adjustment of  $N_{T}$ ,  $N_{P}$ , and  $N_{A}$ , to maximize TPS for BOXING (left) and PONG (right), starting from a sub-optimal configuration ( $N_{A} = N_{T} = N_{P} = 1$ )

![](images/c3390784f94c50e00acf6facfe386e55b4b6a9256be425d500a9e18781d845c2.jpg)

complex dynamics relating both computational and convergence aspects of the learning procedure. Their effect on the convergence of the learning process has to be measured by analyzing not only TPS, but also the learning curves.

# 4.3 DYNAMIC ADJUSTMENT OF TRADE-OFFS

The setting of  $N_P$ ,  $N_T$  and  $N_A$  that maximizes the TPS depends on many aspects such as the computational load of the simulation environment, the size of the DNN, and the available hardware. As a rule of thumb, we found that the number of agents  $N_A$  should at least match the available CPU cores, with two predictors and two trainers  $N_P = N_T = 2$ . However, this rule hardly generalizes to a large variety of different situations and only occasionally corresponds to the computationally most efficient configuration. Therefore, we propose an annealing process to configure the system dynamically. Every minute, we randomly change  $N_P$ ,  $N_T$ , or  $N_A$  by  $\pm 1$ , monitoring alterations in TPS to accept or reject the new setting. The optimal configuration is then automatically identified in a reasonable time, for different environments or systems. Figure 2 shows the automatic adjustment procedure finding two different optimal settings for two different games, on the same real system.

# 4.4 POLICY LAG IN GA3C

At a first sight, GA3C and A3C are different implementations of the same algorithm, but GA3C has a subtle difference which affects the stability of the algorithm. This problem is caused by the latency between the time  $t - k$  when a training example has been generated and when it is consumed for training,  $t$ , essentially changing the gradients to:

$$
\nabla_ {\theta} \left[ \log \pi \left(a _ {t - k} \mid s _ {t - k}; \theta\right) \left(R _ {t - k} - V \left(s _ {t - k}; \theta_ {t}\right)\right) + \beta H (\pi \left(s _ {t - k}; \theta\right)) \right]. \tag {4}
$$

Since the Training Queue is not blocking, the states it contains can be old. The value of the delay  $k$  is bounded by the maximum size of the queue and influenced by how the system configuration balances training and prediction rates. This delay can lead to notable instabilities because of large values generated by  $\log \pi (a_{t - k}|s_{t - k};\theta_t)$ . While  $\pi (a_{t - k}|s_{t - k};\theta_{t - k})$  is generally large, since it was the probability of the selected action  $a_{t - k}$ , over the course of the lag  $k$  new parameters  $\theta_{t}$  can make  $\pi (a_{t - 1}|s_{t - k};\theta_t)$  very small. In the worst case, where the updated action probability is zero, infinite values are generated by the log, causing the optimization to fail. To overcome this problem we add a small term  $\epsilon$ :

$$
\nabla_ {\theta} \left[ \log \left(\pi \left(a _ {t - k} \mid s _ {t - k}; \theta\right) + \epsilon\right) \left(R _ {t - k} - V \left(s _ {t - k}; \theta_ {t}\right)\right) + \beta H \left(\pi \left(s _ {t - k}; \theta\right)\right) \right] \tag {5}
$$

This fix significantly improves the stability of the algorithm and also removes the necessity of gradient clipping. A term is also added in the entropy computation to avoid a similar explosion.

<table><tr><td></td><td>System I</td><td>System II</td><td>System III</td></tr><tr><td rowspan="3">Processor</td><td>Intel Xeon E5-2640v3</td><td>Intel Core i-3820</td><td>Haswell E5-2698v3</td></tr><tr><td>2.60 GHz</td><td>3.60 GHz</td><td>2.30 GHz</td></tr><tr><td>16 cores, dual socket</td><td>8 cores</td><td>16 cores</td></tr><tr><td>GPU</td><td>NVIDIA GeForce Titan X (Maxwell)</td><td>NVIDIA GeForce 980 (Maxwell)</td><td>NVIDIA Tesla K80 (Kepler)</td></tr><tr><td>Software / Profilers</td><td colspan="3">Python 3.5, CUDA 7.5, CUDAV v5.1, TensorFlow r0.11 nvprof, nvvp</td></tr></table>

![](images/cc7eb0fbe7427a06e27a144a0139c8998390fb791ca877f8fc2d96e437ddc4b7.jpg)  
Figure 3: TPS of the top three configurations of predictors  $N_P$  and trainers  $N_T$  for several settings of agents  $N_A$ . Profiling is performed on System I from Table 1 while learning PONG. TPS is normalized by best performance after 16 minutes. A larger DNN model is also shown, as described in the text.

# 5 ANALYSIS

# 5.1 EFFECT OF RESOURCE UTILIZATION ON TPS

We profile the performance of GA3C and in the process seek to better understand the system dynamics of deep RL training on hybrid CPU/GPU systems. Experiments are conducted on the GPU-enabled systems described in Table 1 and monitored with CUDA profilers and custom profiling code based on performance counter timing within Python. We present profiling and convergence experiments both with and without the automatic adjustment of number of agents  $N_{A}$ , number of trainers  $N_{T}$ , and number of predictors  $N_{P}$ .

Table 1: Systems used for profiling and testing.  

<table><tr><td>System</td><td>DNN</td><td>A3C</td><td>GA3C</td><td>Speed up</td></tr><tr><td rowspan="2">System I</td><td>small</td><td>352</td><td>1080</td><td>3×</td></tr><tr><td>large</td><td>11</td><td>1004</td><td>91×</td></tr><tr><td rowspan="2">System II</td><td>small</td><td>116</td><td>728</td><td>6×</td></tr><tr><td>large</td><td>12</td><td>336</td><td>28×</td></tr><tr><td rowspan="2">System III</td><td>small</td><td>300</td><td>1248</td><td>4×</td></tr><tr><td>large</td><td>38</td><td>256</td><td>6×</td></tr></table>

Table 2: PPS on different systems (Table 1) and DNN sizes.

Maximizing training speed. To begin, consider raw training speed as expressed in model

update frequency, or trainings per second (TPS). Figure 3 shows TPS on System I in Table 1 for the first 16 minutes of training on PONG. We consider numbers of agents  $N_{A} \in \{16, 32, 64, 128\}$  and plot the top 3 combinations of  $N_{P}, N_{T} \in \{1, 2, 4, 8, 16\}$ . On this system, increasing  $N_{A}$  yields a higher TPS up to  $N_{A} = 128$  where diminishing returns are observed likely due to additional process overhead. The highest consistent TPS on this system is observed with  $N_{A} = 128$  and  $N_{P} = N_{T} = 2$  with a speed-up of  $\sim 3 \times$  relative to the CPU-only implementation (see Table 2).

![](images/d2300f3b00dd4a820f5100e3084e9b4e3bbdbdfb08be21396f0d4a39f035b5a3.jpg)  
Figure 4: The average training queue size (left panel) and prediction batch size (right panel) of the top 3 performing configurations of  $N_P$  and  $N_T$ , for each  $N_A$ , for PONG and the System I in Table 1.

![](images/1bcb6c90056d70c1fb9ca8f6eade6a857ac86f945057b9c49e0eb139afc06c68.jpg)

GPU utilization and DNN size. The fastest configuration  $(N_A = 128, N_P = N_T = 2)$  for System I in Table 1 has an average GPU utilization time of only  $63\%$ , with average and peak occupancy of  $76\%$  and  $98\%$ , respectively. $^{1}$  This suggests there is computational capacity for a larger network model. Therefore we profile GA3C on a more shallow and deeper DNN architecture $^{2}$  to evaluate this hypothesis. Figure 3 shows that, even with the larger DNN controller, TPS drops by only  $7\%$ ; at the same time, the average GPU utilization and occupancy increase by approximately only  $5\%$  and  $0.5\%$ , respectively. The  $5\%$  increase in the utilization justifies the  $7\%$  drop of TPS and it is caused by the increased depth of the DNN, forcing the GPU to serialize part of the computational tasks. On the other hand, the negligible  $5\%$  increase in occupancy is explained by an efficient management of the computational resources by cuDNN; there is still room available to run additional parallel tasks without paying any supplementary time cost. In other words, we can use a shallower DNN at no additional cost, whereas we expect a small decrease of TPS for deeper DNNs. In practice, this allows experimenting larger architectures, which may be particularly important when exploring real world problems, like car driving (Lillicrap et al., 2015). On the other hand, our CPU implementation of A3C does not scale well with the size of the DNN controller: with the larger DNN network it achieves TPS  $\approx 11$ , which is approximately  $91\times$  slower than GA3C. This behavior seems to be consistent across different systems, as showed in Table 2, where the CPU implementation of A3C using the large DNN is  $7\times$  to  $32\times$  slower than when using the small network. Scaling with the DNN size is more favorable on a GPU, with a slow down factor of  $4.9\times$  in the worst case and a negligible  $1.1\times$  in the best case. Quite interestingly, the GPUs of last generation (based on the Maxwell architecture) seems to scale better ( $2.2\times$  and  $1.1\times$  slow down) than older GPUs ( $4.8\times$  slow down for the Kepler architecture).

Significant latency. Profiling on System I in Table 1 reveals that the average time spent by an agent waiting for a prediction call to be completed is  $108\mathrm{ms}$ , only  $10\%$  of which is taken by the GPU inference. The remaining  $90\%$  is overhead spent accumulating the batch and calling the prediction function in Python. Similarly, for training we find that of the average  $11.1\mathrm{ms}$  spent to perform a DNN update,  $59\%$  is overhead. This seems to suggest that a more optimized implementation (possibly based on a low level language like  $\mathrm{C++}$ ) may reduce these overheads, but this investigation remains for future work.

Manually balancing components. Agents, predictors, and trainers all share the GPU as a resource and thus balance is important. Figure 3 shows the top three (for different number of agents,  $N_A$ ) performing configurations of  $N_P$  and  $N_T$ , for System I in Table 1, showing a significant  $14\%$  drop in TPS between the best and worst configuration. Notice that other configurations (not represented in this figure because not in the top performers list) achieve an even lower TPS. The best results have 4 or fewer predictor threads, seemingly preventing batches from becoming too small. The  $N_P: N_T$

![](images/1a1f24fc96752a5c8e9e4f0b849c4e5f1c5779ccf47ea75921485388e586f2f0.jpg)  
Figure 5: Effect of TPS on convergence speed. For each game, four different settings of GA3C are shown, all starting from the same DNN initialization. Numbers on the right show the cumulative number of frames played among all agents for each setting. Configurations playing more frames converge faster. The dynamic configuration method is capable of catching up with the optimal configuration despite starting with a sub-optimal setting,  $N_{T} = N_{P} = N_{A} = 1$ . Experiments were conducted on System III in Table 1.

![](images/bfeecaf0ea1f0af20bdefef460be660bbb40a40a160fe5cb8fc3ec00a08211b1.jpg)

ratios for top performers tend to be  $1:2, 1:1$ , or  $2:1$ , whereas higher ratios such as  $1:8$  and  $1:4$  are rarely successful, likely due to the implicit dependence of training on prediction speed. At the same time if the training queue is too full, training calls take up more time on the GPU thereby throttling the prediction speed. This is further confirmed by our experimental finding that the TPS and PPS plots track closely. Figure 4 shows the training queue size and prediction batch size for the top three configurations. In all cases, the training queue stabilizes well below its maximum capacity. Additionally, the fastest configuration has one of the largest average prediction batch size, yielding higher utilization of the GPU.

# 5.2 EFFECT OF TPS ON LEARNING SPEED

The beneficial effect of finding an efficient configuration on the training speed is shown in Figure 5. Training with a suboptimal configuration (e.g.  $N_P = N_T = N_A = 1$  or  $N_P = N_T = 1$ ,  $N_A = 16$ ) leads to a severe underutilization of the GPU, a low TPS, and a slow training process. Using the optimal configuration achieves a much higher score in a shorter period of time, mainly driven by playing more frames, i.e. collecting more experiences, in the same amount of time.

(Mnih et al., 2016) also notice that asynchronous methods generally achieve significant speedups from using a greater number of agents, and they even report superlinear speedups for asynchronous one-step Q-learning. It is worth noting that optimal configurations for GA3C generally employ a much higher number of agents compared to the CPU counterpart—e.g. the optimal configuration for System I in Table 1 uses 128 agents. This suggests that the GPU implementation of asynchronous learning methods may benefit from the higher TPS and from the advantage of collecting experience from a wide number of agents at the same time.

The learning curve for GA3C with dynamic configuration enabled (see Figure 5) tracks very closely with the learning curve of the optimal configuration. The total number of frames played is slightly lower over the same time period as a result of the search procedure overhead: the configuration is changed once every minute, tending to oscillate around the optimal configuration. But the scoring performance is nearly identical, indicating that the dynamic method may significantly ease the burden of configuring GA3C on a new system.

Table 3 compares scores achieved by A3C on the CPU (as reported in (Mnih et al., 2016)) with our TensorFlow implementation GA3C. Unfortunately, a direct speed comparison is infeasible without either the original source code or the average number of frames or training updates per second. However, results in this table do show that after one day of training our open-source implementation achieves similar scores to A3C after four days of training.

<table><tr><td rowspan="2"></td><td colspan="9">Atari Game Scores</td><td colspan="2">Attributes</td></tr><tr><td>AMIDAR</td><td>BOXING</td><td>CENTIPEDE</td><td>NAME THIS GAME</td><td>PACMAN</td><td>PONG</td><td>QBERT</td><td>SEAQUEST</td><td>UP-DOWN</td><td>Time</td><td>System</td></tr><tr><td>Human</td><td>1676</td><td>10</td><td>10322</td><td>6796</td><td>15375</td><td>16</td><td>12085</td><td>40426</td><td>9896</td><td>-</td><td>-</td></tr><tr><td>Random</td><td>6</td><td>-2</td><td>1926</td><td>198</td><td>1748</td><td>-18</td><td>272</td><td>216</td><td>533</td><td>-</td><td>-</td></tr><tr><td>A3C</td><td>264</td><td>60</td><td>3756</td><td>10476</td><td>654</td><td>6</td><td>15149</td><td>2355</td><td>74706</td><td>4 days</td><td>CPU</td></tr><tr><td>GA3C</td><td>218</td><td>92</td><td>7386</td><td>5643</td><td>1978</td><td>18</td><td>14966</td><td>1706</td><td>8623</td><td>1 day</td><td>GPU</td></tr></table>

Table 3: Average scores on a subset of Atari games achieved by: a random player (Mnih et al., 2015); a human player (Mnih et al., 2015); A3C after four days of training on a CPU (Mnih et al., 2016); and GA3C after one day of training.

# 6 CONCLUSION

By investigating the computational aspects of our hybrid CPU/GPU implementation of GA3C, we achieve a significant speed up with respect to its CPU counter part. This comes as a result of a flexible system capable of finding a reasonable allocation of the available computational resources. Our approach allows producing and consuming training data at the maximum pace on different systems, or to adapt to temporal changes of the computational load on one system. Despite the fact that we analyze A3C only, most of our findings can be applied to similar RL asynchronous algorithms.

We believe that the analysis of the computational aspects of RL algorithms may be a consistent theme in RL in the future, motivating further studies such as this one. The potential benefits of such investigation goes well beyond the computational aspects. For instance, we demonstrate that GA3C scales with the size of the DNN much more efficiently than our CPU implementation of A3C, thus opening the possibility to explore the use of large DNN controllers to solve real world RL problems.

By open sourcing GA3C, we allow other researchers to further explore this space, investigate in detail the computational aspects of deep RL algorithms, and test new algorithmic solutions.

# ACKNOWLEDGMENTS

We thank Prof. Roy H. Campbell for partially supporting this work.

# REFERENCES

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. URL http://tensorflow.org/. Software available from tensorflow.org.  
M. G. Bellemare, S. Srinivasan, G. Ostrovski, T. Schaul, D. Saxton, and R. Munos. Unifying Count-Based Exploration and Intrinsic Motivation. ArXiv e-prints, June 2016.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym, 2016.  
G. Lample and D. Singh Chaplot. Playing FPS Games with Deep Reinforcement Learning. *ArXiv e-prints*, September 2016.  
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. CoRR, abs/1509.02971, 2015. URL http://arxiv.org/abs/1509.02971.

V. Mnih, A. Puigdomenech Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous Methods for Deep Reinforcement Learning. *ArXiv preprint arXiv:1602.01783*, 2016.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 02 2015. URL http://dx.doi.org/10.1038/nature14236.  
Arun Nair, Praveen Srinivasan, Sam Blackwell, Cagdas Alcicek, Rory Fearon, Alessandro De Maria, Vedavyas Panneershelvam, Mustafa Suleyman, Charles Beattie, Stig Petersen, Shane Legg, Volodymyr Mnih, Koray Kavukcuoglu, and David Silver. Massively parallel methods for deep reinforcement learning. CoRR, abs/1507.04296, 2015. URL http://arxiv.org/abs/1507.04296.  
Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. CoRR, abs/1511.05952, 2015. URL http://arxiv.org/abs/1511.05952.  
David Silver, Aja Huang, Christopher J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of go with deep neural networks and tree search. Nature, 529:484-503, 2016. URL http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html.  
Richard S. Sutton and Andrew G. Barto. Introduction to Reinforcement Learning. MIT Press, Cambridge, MA, USA, 1st edition, 1998. ISBN 0262193981.  
Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 4(2), 2012.  
Hado van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. CoRR, abs/1509.06461, 2015. URL http://arxiv.org/abs/1509.06461.  
Ziyu Wang, Nando de Freitas, and Marc Lanctot. *Dueling network architectures for deep reinforcement learning*. CoRR, abs/1511.06581, 2015. URL http://arxiv.org/abs/1511.06581.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.