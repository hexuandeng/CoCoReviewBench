# RLX2: TRAINING A SPARSE DEEP REINFORCEMENT LEARNING MODEL FROM SCRATCH

Anonymous authors

Paper under double-blind review

# ABSTRACT

Training deep reinforcement learning (DRL) models usually require high computation costs. Therefore, compressing DRL models possesses immense potential for training acceleration and model deployment. However, existing methods that generate small models mainly adopt the knowledge distillation-based approach by iteratively training a dense network. As a result, the training process still demands massive computing resources. Indeed, sparse training from scratch in DRL has not been well explored and is particularly challenging due to non-stationarity in bootstrap training. In this work, we propose a novel sparse DRL training framework, "the Rigged Reinforcement Learning Lottery" (RLx2), which builds upon gradient-based topology evolution and is capable of training a sparse DRL model based entirely on a sparse network. Specifically, RLx2 introduces a novel multi-step TD target mechanism with a dynamic-capacity replay buffer to achieve robust value learning and efficient topology exploration in sparse models. It also reaches state-of-the-art sparse training performance in several tasks, showing  $7.5 \times -20 \times$  model compression with less than  $3\%$  performance degradation and up to  $20 \times$  and  $50 \times$  FLOPs reduction for training and inference, respectively.

# 1 INTRODUCTION

Deep reinforcement learning (DRL) has found successful applications in many important areas, e.g., games (Silver et al., 2017), robotics(Gu et al., 2017) and nuclear fusion (Degrave et al., 2022). However, training a DRL model demands heavy computational resources. For instance, AlphaGo-Zero for Go games (Silver et al., 2017), which defeats all Go-AIs and human experts, requires more than 40 days of training time on four tensor processing units (TPUs). The heavy resource requirement results in expensive consumption and hinders the application of DRL on resource-limited devices.

Sparse networks, initially proposed in deep supervised learning, have demonstrated great potential for model compression and training acceleration of deep reinforcement learning. Specifically, in deep supervised learning, the state-of-the-art sparse training frameworks, e.g., SET (Mocanu et al., 2018) and RigL (Evci et al., 2020), can train a  $90\%$  -sparse network (i.e., the resulting network size is  $10\%$  of the original network) from scratch without performance degradation. On the DRL side, existing works including Rusu et al. (2016); Schmitt et al. (2018); Zhang et al. (2019) succeeded in generating ultimately sparse DRL networks. Yet, their approaches still require iteratively training dense networks, e.g., pre-trained dense teachers may be needed. As a result, the training cost for DRL remains prohibitively high, and existing methods cannot be directly implemented on resource-limited devices, leading to low flexibility in adapting the compressed DRL models to new environments, i.e., on-device models have to be retrained at large servers and re-deployed.

Training a sparse DRL model from scratch, if done perfectly, has the potential to significantly reduce computation expenditure and enable efficient deployment on resource-limited devices, and achieves excellent flexibility in model adaptation. However, training an ultra sparse network (e.g.,  $90\%$  sparsity) from scratch in DRL is challenging due to the non-stationarity in bootstrap training. Specifically, in DRL, the learning target is not fixed but evolves in a bootstrap way (Tesauro et al., 1995), and the distribution of the training data can also be non-stationary (Desai et al., 2019). Moreover, using a sparse network structure means searching in a smaller hypothesis (topology) space, which further reduces the learning target's confidence. As a result, improper sparsification can cause irreversible damage to the learning path (Igl et al., 2021), resulting in poor performance. Indeed, recent

works (Sokar et al., 2021; Graesser et al., 2022) show that a direct adoption of a dynamic sparse training (DST) framework in DRL still fails to achieve good compression of the model for different environments uniformly. Therefore, the following interesting question remains open:

Can an efficient DRL agent be trained from scratch with an ultra-sparse network throughout?

In this paper, we give an affirmative answer to the problem and propose a novel sparse training framework, "the Rigged Reinforcement Learning Lottery" (RLx2), for off-policy RL, which is the first algorithm to achieve sparse training throughout using sparsity of more than  $90\%$  with only minimal performance loss. RLx2 is inspired by the gradient-based topology evolution criteria in RigL (Evci et al., 2020) for supervised learning. However, a direct application of RigL does not achieve high sparsity, because sparse DRL models suffer from unreliable value estimation due to limited hypothesis space, which further disturbs topology evolution. Thus, RLx2 is equipped with a multi-step Temporal Difference (TD) target mechanism and a novel dynamic-capacity replay buffer to achieve robust value learning and efficient topology exploration. These two new components address the value estimation problem under sparse topology, and together with RigL, achieve superior sparse-training performance.

The main contributions of the paper are summarized as follows.

- We investigate the fundamental obstacles in training a sparse DRL agent from scratch, and discover two key factors for achieving good performance under sparse networks, namely robust value estimation and efficient topology exploration.  
- Motivated by our findings, we propose RLx2, the first framework that enables DRL training based entirely on sparse networks. RLx2 possesses two key functions, i.e., a gradient-based search scheme for efficient topology exploration, and a multi-step TD target mechanism with a dynamic-capacity replay buffer for robust value learning.  
- Through extensive experiments, we demonstrate the state-of-the-art sparse training performance of RLx2 with two popular DRL algorithms, TD3 (Fujimoto et al., 2018) and SAC (Haarnoja et al., 2018), on several MuJoCo (Todorov et al., 2012) continuous control tasks. Our results show up to  $20 \times$  model compression. RLx2 also achieves  $20 \times$  acceleration in training and  $50 \times$  in inference in terms of floating-point operations (FLOPs).

# 2 RELATED WORKS

We discuss the related works on training sparse models in deep supervised learning and reinforcement learning below. We also provide a comprehensive performance comparison in Table 1.

Sparse Models in Deep Supervised Learning Han et al. (2015; 2016); Srinivas et al. (2017); Zhu & Gupta (2018) focus on finding a sparse network by pruning pre-trained dense networks. Iterative Magnitude Pruning (IMP) in Han et al. (2016) achieves a sparsity of more than  $90\%$ . Techniques including neuron characteristic (Hu et al., 2016), dynamic network surgery (Guo et al., 2016), derivatives (Dong et al., 2017; Molchanov et al., 2019b), regularization (Louizos et al., 2018; Tartaglione et al., 2018), and dropout (Molchanov et al., 2017) have also been applied in network pruning. Another line of work focuses on the Lottery Ticket Hypothesis (LTH), first proposed in Frankle & Carbin (2019), which shows that training from a sparse network from scratch is possible if one finds a sparse "winning ticket" initialization in deep supervised learning. The LTH is also validated in other deep learning models (Chen et al., 2020; Brix et al., 2020; Chen et al., 2021).

Many works (Bellec et al., 2017; Mocanu et al., 2018; Mostafa & Wang, 2019; Dettmers & Zettlemoyer, 2019; Evci et al., 2020) also try to train a sparse neural network from scratch without having to pre-trained dense models. These works adjust structures of sparse networks during training, including Deep Rewiring (DeepR) (Bellec et al., 2017), Sparse Evolutionary Training (SET) (Mocanu et al., 2018), Dynamic Sparse Reparameterization (DSR) (Mostafa & Wang, 2019), Sparse Networks from Scratch (SNFS) (Dettmers & Zettlemoyer, 2019), and Rigged Lottery (RigL) (Evci et al., 2020). Works Single-Shot Network Pruning (SNIP) (Lee et al., 2019) and Gradient Signal Preservation (GraSP) (Wang et al., 2020) focus on finding static sparse networks before training.

Table 1: Comparison of different sparse training techniques in DRL. Here ST and TA stand for "sparse throughout training" and "training acceleration", respectively. The shown sparsity is the maximum sparsity level without performance degradation under the algorithms.  

<table><tr><td>Name</td><td>Paradigm</td><td>Scenario</td><td>ST</td><td>TA</td><td>Sparsity</td></tr><tr><td>PoPS (Livne &amp; Cohen, 2020)</td><td>IMP</td><td>Online</td><td>No</td><td>No</td><td>~ 99%</td></tr><tr><td>LTH-RL (Yu et al., 2020)</td><td>IMP</td><td>Online</td><td>Yes</td><td>No</td><td>~ 99%</td></tr><tr><td>LTH-IL (Vischer et al., 2022)</td><td>IMP</td><td>Online</td><td>Yes</td><td>No</td><td>~ 95%</td></tr><tr><td>SSP (Arnob et al., 2021)</td><td>Single-shot pruning</td><td>Offline</td><td>Yes</td><td>Yes</td><td>~ 95%</td></tr><tr><td>GST (Lee et al., 2021)</td><td>Gradual pruning</td><td>Online</td><td>No</td><td>No</td><td>~ 70%</td></tr><tr><td>DST (Sokar et al., 2021)</td><td>Topology Evolution</td><td>Online</td><td>Yes</td><td>Yes</td><td>~ 50%</td></tr><tr><td>RLx2 (Ours)</td><td>Topology Evolution</td><td>Online</td><td>Yes</td><td>Yes</td><td>~ 95%</td></tr></table>

Sparse Models in DRL Evci et al. (2020); Sokar et al. (2021) show that finding a sparse model in DRL is difficult due to training instability. Existing works (Rusu et al., 2016; Schmitt et al., 2018; Zhang et al., 2019) leverage knowledge distillation with static data to avoid unstable training and obtain small dense agents. Policy Pruning and Shrinking (PoPs) in Livne & Cohen (2020) obtains a sparse DRL agent with iterative policy pruning (similar to IMP). LTH in DRL is firstly investigated in Yu et al. (2020), and then Vischer et al. (2022) shows that a sparse winning ticket can also be found by behavior cloning (BC). Another line of works (Lee et al., 2021; Sokar et al., 2021; Arnob et al., 2021) attempts to train a sparse neural network from scratch without pre-training a dense teacher. Group Sparse Training (GST) in Lee et al. (2021) utilizes block-circuit compression and pruning. Sokar et al. (2021) proposes using SET in topology evolution in DRL and achieves  $50\%$  sparsity. Arnob et al. (2021) proposes single-shot pruning (SSP) for offline RL. Graesser et al. (2022) finds that pruning often obtains the best results and plain dynamic sparse training methods, including SET and RigL, improves over static sparse training significantly. However, existing works either demands massive computing resources, e.g. pruning-based methods (Rusu et al., 2016; Schmitt et al., 2018; Zhang et al., 2019; Livne & Cohen, 2020), or fail in ultra sparse models, e.g. DST-based methods (Sokar et al., 2021; Graesser et al., 2022). In this paper, we further improve the performance of DST by introducing a multi-step TD target mechanism with a dynamic-capacity replay buffer, which effectively addresses the unreliability of fixed-topology models during sparse training.

# 3 DEEP REINFORCEMENT LEARNING PRELIMINARIES

In reinforcement learning, an agent interacts with an unknown environment to learn an optimal policy. The learning process is formulated as a Markov decision process (MDP)  $\mathcal{M} = \langle S, \mathcal{A}, r, \mathbb{P}, \gamma \rangle$ , where  $S$  is the state space,  $\mathcal{A}$  is the action space,  $r$  is the reward function,  $\mathbb{P}$  denotes the transition matrix, and  $\gamma$  stands for the discount factor. Specifically, at time slot  $t$ , given the current state  $s_t \in S$ , the agent selects an action  $a_t \in \mathcal{A}$  by policy  $\pi : S \to \mathcal{A}$ , which then incurs a reward  $r_t(s, a)$ .

Denote the Q function associated with the policy  $\pi$  for state-action pair  $(s,a)$  as

$$
Q _ {\pi} (s, a) = \mathbb {E} _ {\pi} \left[ \sum_ {i = t} ^ {T} \gamma^ {i - t} r \left(s _ {i}, a _ {i}\right) | s _ {t} = s, a _ {t} = a \right]. \tag {1}
$$

In actor-critic methods (Silver et al., 2014), the policy  $\pi(s; \phi)$  is parameterized by a policy (actor) network with weight parameter  $\phi$ , and the Q function  $Q(s, a; \theta)$  is parameterized by a value (critic) network with parameter  $\theta$ . The goal of the agent is to find an optimal policy  $\pi^*(s; \phi^*)$  which maximizes long-term cumulative reward, i.e.,  $J^* = \max_{\phi} \mathbb{E}_{\pi(\phi)} \left[ \sum_{i=0}^{T} \gamma^{i-t} r(s_i, a_i) | s_0, a_0 \right]$ .

There are various DRL methods for learning an efficient policy. In this paper, we focus on off-policy TD learning methods, including a broad range of state-of-the-art algorithms, e.g., TD3 (Fujimoto et al., 2018) and SAC (Haarnoja et al., 2018). Specifically, the critic network is updated by gradient descent to fit the TD targets  $\mathcal{T}_1(s,a)$  generated by a target network  $Q(s,a;\theta^{\prime})$ , i.e.,

$$
\mathcal {T} (s, a) = r (s, a) + \gamma Q (s, a; \theta^ {\prime}) \tag {2}
$$

for each state-action pair  $(s, a)$ , where the action  $a = \pi(s; \phi)$ . The loss function of the value network is defined as the expected squared loss between the current value network and TD targets:

$$
\mathcal {L} (\theta) = \mathbb {E} _ {\pi (\phi)} \left[ Q \left(s _ {i}, a _ {i}; \theta\right) - \mathcal {T} \right] ^ {2}. \tag {3}
$$

The policy  $\pi(s; \phi)$  is updated by the deterministic policy gradient algorithm in Silver et al. (2014):

$$
\nabla_ {\phi} J (\phi) = \mathbb {E} _ {\pi (\phi)} \left[ \left. \nabla_ {a} Q _ {\pi} (s, a; \theta) \right| _ {a = \pi (s)} \nabla_ {\phi} \pi_ {\phi} (s) \right].
$$

# 4 RLX2: RIGGING THE LOTTERY IN DRL

In this section, we present the RLx2 algorithm, which is capable of training a sparse DRL model from scratch. An overview of the RLx2 framework on an actor-critic architecture is shown in Figure 1. To motivate the design of RLx2, we present a comparison of four sparse DRL training methods using TD3 with different topology update schemes on InvertedPendulum-v2, a simple control task from MuJoCo, in Figure 2.

![](images/6c2ef38e56a32a94ce6cfd35689231fe2179053ae5ce6065acf4c9381e7d5afb.jpg)  
Figure 1: The RLx2 framework contains three key components, i.e., multi-step TD target mechanism, dynamic-capacity replay buffer and gradient-based topology evolution.

![](images/1ddf0e6a426888303e20f4bcf1e4f1a146df21ebea4266dedf1863ccd78704b8.jpg)  
Figure 2: Performance comparison for four sparse training methods, i.e., SS, RigL, RigL+Q* and RLx2. The results show that both efficient topology evolution and robust value estimation are critical.

From the results, we make the following important observations. (i) Topology evolution is crucial. It can be seen that a random static sparse network (SS) leads to much worse performance than RigL. (ii) Robust value estimation is essential. This is validated by the comparison between RigL and RigL+Q*, both using the same topology adjustment scheme but with different Q-values.

Motivated by the above findings, RLx2 utilizes gradient-based topology adjustment, i.e., RigL (for topology evolution), and introduces a multi-step TD target mechanism with a dynamic-capacity replay buffer (for robust value estimation). Below, we explain the key components of RLx2 in detail, to illustrate why RLx2 is capable of achieving robust value learning and efficient topology exploration simultaneously.

# 4.1 GRADIENT-BASED TOPOLOGY EVOLUTION

The topology evolution in RLx2 is conducted by adopting the RigL method (Evci et al., 2020). Specifically, we compute the gradient values of the loss function with respect to link weights. Then, we dynamically grow connections (connecting neurons) with large gradients and delete existing links with the smallest weights. In this way, we obtain a sparse mask that evolves by self-adjustment.

The pseudo-code of our scheme is given in Algorithm 1, where  $\odot$  is an element-wise multiplication operator and  $M_{\phi}$  is a binary mask to represent the sparse architecture of the network  $\phi$ . The update fraction anneals during the training process according to  $\zeta_t = \frac{\zeta_0}{2} (1 + \cos (\frac{\pi t}{T_{\mathrm{end}}}))$ , where  $\zeta_0$  is the initial update fraction and  $T_{\mathrm{end}}$  is the total number of iterations. Note that this topology adjustment happens very infrequently, i.e., every 10000 step, such that consumption of this step is negligible (detailed analysis in Appendix C.3). Also, finding top- $k$  links with maximum gradients in Line 10 can be efficiently implemented (detailed in Appendix A.1). Thus, the topology evolution can be implemented efficiently on resource-limited devices.

# Algorithm 1 Topology Evolution (Evci et al., 2020)

1:  $N_{l}$ : Number of parameters in layer  $l$  
2:  $\theta_{l}$ : Parameters in layer  $l$  
3:  $M_{\theta_l}$ : Sparse mask of layer  $l$  
4:  $s_l$ : Sparsity of layer  $l$  
5:  $L$ : Loss function  
6:  $\zeta_t$ : Update fraction in training step  $t$  
7: for each layer  $l$  do  
8:  $k = \zeta_t(1 - s_l)N_l$  
9:  $\mathbb{I}_{\mathrm{drop}} = \mathrm{ArgTopK}(-|\theta_l\odot M_{\theta_l}|,k)$  
10:  $\mathbb{I}_{\mathrm{grow}} = \mathrm{ArgTopK}_{i\notin \theta_l\odot M_{\theta_l}\setminus \mathbb{I}_{\mathrm{drop}}}(|\nabla_{\theta_l}L,k|)$  
11: Update  $M_{\theta_l}$  according to  $\mathbb{I}_{\mathrm{drop}}$  and  $\mathbb{I}_{\mathrm{grow}}$  
12:  $\theta_{l}\gets \theta_{l}\odot M_{\theta_{l}}$  
13: end for

# 4.2 ROBUST VALUE LEARNING

As discussed above, value function learning is crucial in sparse training. Specifically, we find that under sparse models, robust value learning not only serves to guarantee the efficiency of bootstrap training as in dense DRL training, but also guides the gradient-based topology exploration of the sparse network during training.

Figure 3 compares the performance of the masks (sparse networks) obtained under RigL and RLx2 (which is RigL + robust value learning) on Ant-v3. Here we adopt the methods in (Frankle & Carbin, 2019) for evaluating the obtained sparse model: 1) first initialize a random sparse topology; 2) keep adjusting the topology during the training and obtain the final mask; 3) train a sparse agent

with the obtained mask (the mask is fixed throughout this training phase, only the weights are restored to their initial values as in the first step at the beginning). It can be clearly observed that the mask by RLx2 significantly outperforms that by solely using RigL (Please see Appendix C.4 for details and experiments in other environments, where similar results are observed).

To achieve robust value estimation and properly guide the topology search, RLx2 utilizes two major components: i) multi-step TD targets to bootstrap value estimation; ii) a novel dynamic-capacity replay buffer to eliminate the potential data inconsistency due to policy change during training.

![](images/28498661c1b9271e4a2cc8d29d075b2e3c41f3f9ef746cd37486cbb68d1652bd.jpg)  
Figure 3: Sparse model comparison in Ant-v3.

# 4.2.1 MULTI-STEP TD TARGET

In TD learning, a TD target is generated, and the value network will be iteratively updated by minimizing a squared loss induced by the TD target. Single-step methods generate the TD target by combining one-step reward and discounted target network output, i.e.,  $\mathcal{T}_1 = r_t + \gamma Q(s_{t+1}, \pi(s_{t+1}); \theta)$ . However, a sparse network parameter  $\widehat{\theta} = \theta \odot M_\theta$ , obtained from its dense counterpart  $\theta$ , will inevitably reside in a smaller hypothesis space due to using fewer parameters. This means that the output of the sparse value network  $\widehat{\theta}$  can be unreliable and may lead to inaccurate value estimation. Denote the fitting error of the value network as  $\epsilon(s, a) = Q(s, a; \theta) - Q_\pi(s, a)$ . One sees that this error may be larger under a sparse model compared to that under a dense network.

To overcome this issue, we adopt a multi-step target, i.e.,  $\mathcal{T}_n = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n Q(s_{t+n}, \pi(s_{t+n}); \theta)$ , where the target combines an  $N$ -step sample and the output of the sparse value network after  $N$ -step, both appropriately discounted. By doing so, we reduce the expected error between the TD target and the true target. Specifically, Eq.(4) shows the expected TD error between multi-step TD target  $\mathcal{T}_n$  and the true Q-value  $Q_\pi$  associated with the target policy  $\pi$ , conditioned on transitions from behavior policy  $b$  (see detailed derivation in Appendix A.2).

$$
\mathbb {E} _ {b} \left[ \mathcal {T} _ {n} (s, a) \right] - Q _ {\pi} (s, a) = \underbrace {\left(\mathbb {E} _ {b} \left[ \mathcal {T} _ {n} (s , a) \right] - \mathbb {E} _ {\pi} \left[ \mathcal {T} _ {n} (s , a) \right]\right)} _ {\text {P o l i c y i n c o s t e n c y e r r o r}} + \gamma^ {n} \underbrace {\mathbb {E} _ {\pi} \left[ \epsilon \left(s _ {n} , \pi \left(s _ {n}\right)\right) \right]} _ {\text {N e t w o r k f i t t i n g e r r o r}} \tag {4}
$$

The multi-step target has been studied in existing works (Bertsekas & Ioffe, 1996; Precup, 2000; Munos et al., 2016) for improving TD learning. In our case, we also find that introducing a multi-

step target reduces the network fitting error by a multiplicative factor  $\gamma^n$ , as shown in Eq. (4). On the other hand, it has been observed, e.g., in Fedus et al. (2020), that an immediate adoption of multi-step TD targets may cause a larger policy inconsistency error (the first term in Eq. (4)). Thus, we adopt a hybrid scheme to suppress policy inconsistency and further improve value learning. Specifically, at the early stage of training, we use one-step TD targets to better handle the quickly changing policy during this period, where a multi-step target may not be meaningful. Then, after several training epochs, when the policy change becomes less abrupt, We permanently switch to multi-step TD targets, to exploit its better approximation of the value function.

# 4.2.2 DYNAMIC-CAPACITY BUFFER

The second component of RLx2 for robust value learning is a novel dynamic buffer scheme for controlling data inconsistency. Off-policy algorithms use a replay buffer to store collected data and train networks with sampled batches from the buffer. Their performances generally improve when larger replay capacities are used (Fedus et al., 2020). However, off-policy algorithms with unlimited-size replay buffers can suffer from policy inconsistency due to the following two aspects.

(i) Inconsistent multi-step targets: In off-policy algorithms with multi-step TD targets, the value function is updated to minimize the squared loss in Eq. (3) on transitions sampled from the replay buffer, i.e., the reward sequence  $r_t, r_{t+1}, \dots, r_{t+n}$  collected during training. However, the fact that the policy can evolve during training means that the data in the replay buffer, used for Monte-Carlo approximation of the current policy  $\pi$ , may be collected under a different behavior policy  $b$  (Hernandez-Garcia & Sutton, 2019; Fedus et al., 2020). As a result, it may lead to a large policy inconsistency error in Eq. (4), causing inaccurate estimation.

(ii) Mismatched training data: In practice, the agent minimizes the value loss  $\widehat{\mathcal{L}} (\theta)$  with respect to the sampled value in mini-batch  $\mathcal{B}_t$ , given by

![](images/460d4d4092cacbc69e4681dd540393fecd157fbc70bce33fcf64478a3f41da96.jpg)  
Figure 4: Dynamic buffer capacity & policy inconsistency

$$
\widehat {\mathcal {L}} (\theta) = \frac {1}{| \mathcal {B} _ {t} |} \sum_ {(s _ {i}, a _ {i}) \sim \mathcal {B} _ {t}} (Q (s _ {i}, a _ {i}; \theta) - \mathcal {T}) ^ {2} \tag {5}
$$

Compared to Eq. (3), the difference between the distribution of transitions in the mini-batch  $\mathcal{B}_t$  and the true transition distribution induced by the current policy also leads to a mismatch in the training objective (Fujimoto et al., 2019). Indeed, our analysis in Appendix A.4 shows that training performance is closely connected to policy consistency.

Motivated by our analysis, we introduce a dynamically-sized buffer to reduce the policy gap based on the policy distance of the collected data. The formal scheme is given in Algorithm 3. Specifically, we introduce the following policy distance measure to evaluate the inconsistency of data in the buffer, i.e.,

$$
\mathcal {D} (\mathcal {B}, \phi) = \frac {1}{K} \sum_ {\left(s _ {i}, a _ {i}\right) \in \operatorname {O l d K} (\mathcal {B})} \| \pi \left(s _ {i}; \phi\right) - a _ {i} \| _ {2}, \tag {6}
$$

where  $\mathcal{B}$  denotes the current replay buffer, OldK( $\mathcal{B}$ ) denotes the oldest  $K$  transitions in  $\mathcal{B}$ , and  $\pi(\cdot; \phi)$  is the current policy. Here  $K$  is a hyperparameter. Upon any update to  $\phi$  and appending new transitions to  $\mathcal{B}$ , we calculate the latest  $\mathcal{D}(\mathcal{B}, \phi)$  value. If  $\mathcal{D}(\mathcal{B}, \phi)$  gets above a certain prespecified threshold, we start to pop items from  $\mathcal{B}$  in a First-In-First-Out (FIFO) order until this distance measure  $\mathcal{D}$  becomes below the threshold.

A visualization of the number of stored samples (dynamic capacity) and the proposed policy distance metric during training is shown in Figure 4. We see that the policy distance oscillates in the early stage as the policy evolves, but it is tightly controlled and does not violate the threshold condition to effectively address the off-policyness issue. As the policy converges, the policy distance tends to decrease and converge (We also show in Appendix C.2 that the performance of RLx2 is insensitive to the policy threshold).

# 5 EXPERIMENTS

In this section, we investigate the performance improvement of RLx2 in Section 5.1, and the importance of each component in RLx2 in Section 5.2. In particular, we pay extra attention to the role topology evolution plays in sparse training in Section 5.3. Our experiments are conducted in four popular MuJoCo environments: HalfCheetah-v3 (Hal.), Hopper-v3 (Hop.), Walker2d-v3 (Wal.), and Ant-v3 (Ant.), for RLx2 with two off-policy algorithms, TD3 and SAC. Instantiations of RLx2 on TD3 and SAC are provided in Appendix B. Each result is averaged over eight random seeds.

# 5.1 COMPARATIVE EVALUATION

Table 2: Comparisons of RLx2 with sparse training baselines. Here "Sp." refers to the sparsity level (percentage of model size reduced), "Total Size" refers to the total parameters of both critic and actor networks (detailed calculation of training and inference FLOPs are given in Appendix C.3). The right five columns show the final performance of different methods. The "Total size," "FLOPs", and "Performance" are all normalized w.r.t. the original large dense model (detailed in Appendix C.2).  

<table><tr><td>Alg.</td><td>Env.</td><td>Actor Sp.</td><td>Critic Sp.</td><td>Total Size</td><td>FLOPs (Train)</td><td>FLOPs (Test)</td><td>Tiny (%)</td><td>SS (%)</td><td>SET (%)</td><td>RigL (%)</td><td>RLx2 (%)</td></tr><tr><td rowspan="4">TD3</td><td>Hal.</td><td>90%</td><td>85%</td><td>0.133x</td><td>0.138x</td><td>0.100x</td><td>86.3</td><td>77.1</td><td>92.6</td><td>90.8</td><td>99.8</td></tr><tr><td>Hop.</td><td>98%</td><td>95%</td><td>0.040x</td><td>0.043x</td><td>0.020x</td><td>64.5</td><td>67.7</td><td>66.5</td><td>90.6</td><td>97.0</td></tr><tr><td>Wal.</td><td>97%</td><td>95%</td><td>0.043x</td><td>0.045x</td><td>0.030x</td><td>60.8</td><td>42.9</td><td>39.3</td><td>35.7</td><td>98.1</td></tr><tr><td>Ant.</td><td>96%</td><td>88%</td><td>0.093x</td><td>0.100x</td><td>0.040x</td><td>16.5</td><td>49.6</td><td>62.5</td><td>68.5</td><td>103.9</td></tr><tr><td></td><td>Avg.</td><td>95%</td><td>91%</td><td>0.077x</td><td>0.081x</td><td>0.048x</td><td>57.0</td><td>59.3</td><td>65.2</td><td>71.4</td><td>99.7</td></tr><tr><td rowspan="4">SAC</td><td>Hal.</td><td>90%</td><td>80%</td><td>0.180x</td><td>0.197x</td><td>0.100x</td><td>95.0</td><td>75.4</td><td>94.8</td><td>89.8</td><td>102.2</td></tr><tr><td>Hop.</td><td>98%</td><td>95%</td><td>0.044x</td><td>0.048x</td><td>0.020x</td><td>89.1</td><td>81.6</td><td>103.9</td><td>110.0</td><td>109.7</td></tr><tr><td>Wal.</td><td>90%</td><td>90%</td><td>0.100x</td><td>0.113x</td><td>0.100x</td><td>73.8</td><td>83.4</td><td>95.8</td><td>81.9</td><td>103.2</td></tr><tr><td>Ant</td><td>90%</td><td>75%</td><td>0.220x</td><td>0.239x</td><td>0.100x</td><td>49.6</td><td>49.3</td><td>79.8</td><td>90.9</td><td>105.6</td></tr><tr><td></td><td>Avg.</td><td>92%</td><td>85%</td><td>0.136x</td><td>0.149x</td><td>0.080x</td><td>76.9</td><td>72.4</td><td>93.6</td><td>93.2</td><td>105.2</td></tr><tr><td colspan="2">Avg.</td><td>94%</td><td>88%</td><td>0.107x</td><td>0.115x</td><td>0.064x</td><td>67.0</td><td>65.9</td><td>79.4</td><td>82.3</td><td>102.4</td></tr></table>

Table 2 summarizes the comparison results. In our experiments, we compare RLx2 with the following baselines: (i) Tiny, which uses tiny dense networks with the same number of parameters as the sparse model in training. (ii) SS: using static sparse networks with random initialization. (iii) SET (Bellec et al., 2017), which uses dynamic sparse training by dropping connections according to the magnitude and growing connections randomly. Please notice that the previous work (Sokar et al., 2021) also adopts the SET algorithm for topology evolution in reinforcement learning. Our implementations reach better performance due to different hyperparameters. (iv) RigL (Evci et al., 2020), which uses dynamic sparse training by dropping and growing connections with magnitude and gradient criteria, respectively, the same as RLx2's topology evolution procedure.

In our experiments, we allow the actor and critic networks to take different sparsities. We define an ultimate compression ratio, i.e., the largest sparsity level under which the performance degradation under RLx2 is within  $\pm \% 3$  of that under the original dense models. This can also be understood as the minimum size of the sparse model with the full performance of the original dense model. We present performance comparison results in Table 2 based on the ultimate compression ratio. The performance of each algorithm is evaluated with the average reward per episode over the last 30 policy evaluations of the training (policy evaluation is conducted every 5000 steps). Hyperparameters are fixed in all four environments for TD3 and SAC, respectively, which are presented in Appendix C.2.

Performance Table 2 shows RLx2 performs best among all baselines in all four environments by a large margin (except for a close performance with RigL with SAC in Hopper). In addition, tiny dense (Tiny) and random static sparse networks (SS) performance are worst on average. SET and RigL are better yet fail to maintain the performance in Walker2d-v3 and Ant-v3, which means robust value learning is necessary under sparse training. To further validate the performance of RLx2, we compare the performance of different methods under different sparsity levels in Hopper-v3 and Ant-v3 in Figure 5, showing RLx2 has a significant performance gain over other baselines.

Model Compression RLx2 achieves superior compression ratios (the reciprocal of the total size), with minor performance degradation (less than  $3\%$ ). Specifically, RLx2 with TD3 achieves  $7.5 \times$

$25 \times$  model compression, with the best compression ratios of  $25 \times$  on Hopper-v3. The actor can be compressed for each environment by more than  $96\%$ , and the critic is compressed by  $85\% - 95\%$ . The results for SAC are similar. RLx2 with SAC achieves a  $5 \times -20 \times$  model compression.

![](images/b9e335b3ca97181208c754a5a83b4e0fd6451da5846c02e73ecd8ced2859e12e.jpg)  
Figure 5: Performance comparison under different model sparsity.

Acceleration in FLOPs Different from knowledge-distillation/BC based methods, e.g., Livne & Cohen (2020); Vischer et al. (2022); Lee et al. (2021), RLx2 uses a sparse network throughout training. Thus, it has an additional advantage of immensely accelerating training and saving computation, i.e.,  $12 \times$  training acceleration and  $20 \times$  inference acceleration for RLx2-TD3, and  $7 \times$  training acceleration and  $12 \times$  inference acceleration for RLx2-SAC.

# 5.2 ABLATION STUDY

We conduct a comprehensive ablation study on the three critical components of RLx2 on TD3, i.e., topology evolution, multi-step TD target, and dynamic-capacity buffer, to examine the effect of each component in RLx2 and their robustness in hyperparameters.

Topology evolution RLx2 drops and grows connections with magnitude and gradient criteria, respectively, which has been adopted in RigL (Evci et al., 2020) for deep supervised learning. To validate the necessity of our topology evolution criteria, we compare RLx2 with three baselines, which replace the topology evolution scheme in RLx2 with Tiny, SS and SET, while keeping other components in RLx2 unchanged, i.e., they are also equipped with multi-step targets and dynamic-capacity buffer. The left part of Table 3 shows that RigL as a topology adjustment scheme (the resulting scheme is RLx2 when using RigL) performs best among the four baselines. We also observe that Tiny performs worst, which is consistent with the conclusion in existing works (Zhu & Gupta, 2018) that a sparse network may contain a smaller hypothesis space and leads to performance loss, which necessitates a topology evolution scheme.

Table 3: Ablation study on topology evolution and multi-step target, where the performance (%) is normalized with respect to the performance of dense models.  

<table><tr><td rowspan="2">Env.</td><td colspan="4">Topoloy Evolution</td><td colspan="5">Multi-step Target</td></tr><tr><td>Tiny</td><td>SS</td><td>SET</td><td>RLx2</td><td>1-step</td><td>2-step</td><td>3-step</td><td>4-step</td><td>5-step</td></tr><tr><td>Hal.</td><td>93.3</td><td>86.1</td><td>100.1</td><td>99.8</td><td>96.5</td><td>101.7</td><td>99.8</td><td>98.8</td><td>97.0</td></tr><tr><td>Hop.</td><td>74.4</td><td>84.2</td><td>88.8</td><td>97.0</td><td>77.9</td><td>91.7</td><td>97.0</td><td>84.0</td><td>87.5</td></tr><tr><td>Wal.</td><td>84.1</td><td>83.8</td><td>89.4</td><td>98.1</td><td>73.9</td><td>93.7</td><td>98.1</td><td>99.1</td><td>99.3</td></tr><tr><td>Ant.</td><td>28.7</td><td>80.2</td><td>83.5</td><td>103.9</td><td>103.9</td><td>105.1</td><td>103.9</td><td>96.7</td><td>94.5</td></tr><tr><td>Avg.</td><td>70.1</td><td>83.6</td><td>90.4</td><td>99.7</td><td>88.1</td><td>98.1</td><td>99.7</td><td>94.6</td><td>94.6</td></tr></table>

Multi-step TD targets We also compare different step lengths in multi-step TD targets for RLx2 in the right part of Table 3. We find that multi-step TD targets with a step length of 3 obtain the maximum performance. In particular, multi-step TD targets improve the performance dramatically in Hopper-v3 and Walker2d-v3, while the improvement in HalfCheetach-v3 and Ant-v3 is minor.

Dynamic-capacity Buffer We compare different replay buffer sizing schemes, including our dynamic scheme and using different fixed buffer capacities or an unlimited buffer. Figure 6 shows that our dynamic-capacity buffer performs best among all settings

![](images/d2a14c7d4c63b51197e8115df14ecc8fe5679a5594492032502d5b040fce9f0b.jpg)  
Figure 6: Performance with different buffer schemes.

of the buffer. Smaller buffer capacity benefits the performance in

the early stage but may reduce the final performance. This is because using a smaller buffer results in higher sample efficiency in the early stage of training but fails in reaching high performance in the long term, whereas a large or even unlimited one may perform poorly in all stages.

# 5.3 WHY EVOLVE TOPOLOGY IN DRL?

Compared to dense networks, sparse networks have smaller hypothesis spaces. Even under the same sparsity, different sparse architectures correspond to different hypothesis spaces. As Frankle & Carbin (2019) has shown, some sparse architecture (e.g., the "winning ticket") performs better than a random one. To emphasize the necessity of topology evolution in sparse training, we compare different sparse network architectures in Figure 7, including the random ticket (topology sampled at random and fixed throughout training), the winning ticket (topology from an RLx2 run and fixed throughout training), and a dynamic ticket (i.e., training using RLx2) under both reinforcement learning (RL) and behavior cloning (BC).<sup>2</sup>

![](images/96446c8c803b1a118af9d33c9fc67bd8ee03f69ac5bb4f3764bce2dadd697bbf.jpg)  
(a) Rigging the Lottery Ticket in RL training

![](images/f15275a383794be33feecec8526136dd6cf7d3387799dccf328de2cf64de7aaf.jpg)  
Figure 7: Comparison of different sparse network architecture for training a sparse DRL agent in Ant-v3, where the sparsity is the same as that in Table 2.  
(b) Rigging the Lottery Ticket in behavior cloning

From Figure 7(a), we see that RLx2 achieves the best performance, which is comparable with that under the original dense model. Due to the potential data inconsistency problem in value learning and the smaller hypothesis search space under sparse networks, training with a single fixed topology does not fully reap the benefit of high sparsity and can cause significantly degraded performance. That is why the winning ticket and random ticket both lead to significant performance loss compared to RLx2. On the other hand, Figure 7(b) shows that in BC tasks, the winning ticket and RLx2 perform almost the same as the dense model, while the random ticket performs worst. This indicates that an appropriate fixed topology can indeed be sufficient to reach satisfactory performance in BC, which is intuitive since BC adopts a supervised learning approach and eliminates non-stationarity due to bootstrapping training. In conclusion, we find that a fixed winning ticket can perform as well as a dynamic topology that evolves during the training in behavior cloning, while RLx2 outperforms the winning ticket in RL training. This observation indicates that topology evolution not only helps find the winning ticket in sparse DRL training but is also a necessary component of training a sparse DRL agent due to the extra non-stationary in bootstrapping training, compared to deep supervised learning.

# 6 CONCLUSION

This paper proposes a sparse training framework, RLx2, for off-policy reinforcement learning (RL). RLx2 utilizes gradient-based evolution to enable efficient topology exploration and establishes robust value learning using a multi-step TD target mechanism with a dynamic-capacity replay buffer. RLx2 enables training an efficient DRL agent with minimal performance loss using an ultra-sparse network throughout training and removes the need for pre-training dense networks. Our extensive experiments on RLx2 with TD3 and SAC demonstrate state-of-the-art sparse training performance, showing a  $7.5 \times -20 \times$  model compression with less than  $3\%$  performance degradation and up to  $20 \times$  and  $50 \times$  FLOPs reduction in training and inference, respectively.

# REPRODUCIBILITY STATEMENT

Experiment details (including an efficient implementation for RLx2, implementation details of the dynamic buffer, hyperparameters, and network architectures) are included in Appendix C for reproduction. The proof for our analysis of the dynamic buffer can be found in Appendix A.4. The code will be open-sourced upon publication of the paper.

# REFERENCES

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International conference on machine learning, pp. 22-31. PMLR, 2017.  
Samin Yeasar Arnob, Riyasat Ohib, Sergey Plis, and Doina Precup. Single-shot pruning for offline reinforcement learning. arXiv preprint arXiv:2112.15579, 2021.  
Guillaume Bellec, David Kappel, Wolfgang Maass, and Robert Legenstein. Deep rewiring: Training very sparse deep networks. arXiv preprint arXiv:1711.05136, 2017.  
Dimitri P Bertsekas and Sergey Ioffe. Temporal differences-based policy iteration and applications in neuro-dynamic programming. Lab. for Info. and Decision Systems Report LIDS-P-2349, MIT, Cambridge, MA, 14, 1996.  
Christopher Brix, Parnia Bahar, and Hermann Ney. Successfully applying the stabilized lottery ticket hypothesis to the transformer architecture. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 3909-3915, 2020.  
Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Zhangyang Wang, and Michael Carbin. The lottery ticket hypothesis for pre-trained bert networks. Advances in neural information processing systems, 33:15834-15846, 2020.  
Tianlong Chen, Yongduo Sui, Xuxi Chen, Aston Zhang, and Zhangyang Wang. A unified lottery ticket hypothesis for graph neural networks. In International Conference on Machine Learning, pp. 1695-1706. PMLR, 2021.  
Jonas Degrave, Federico Felici, Jonas Buchli, Michael Neunert, Brendan Tracey, Francesco Carpanese, Timo Ewalds, Roland Hafner, Abbas Abdolmaleki, Diego de Las Casas, et al. Magnetic control of tokamak plasmas through deep reinforcement learning. Nature, 602(7897):414-419, 2022.  
Shrey Desai, Hongyuan Zhan, and Ahmed Aly. Evaluating lottery tickets under distributional shifts. arXiv preprint arXiv:1910.12708, 2019.  
Tim Dettmers and Luke Zettlemoyer. Sparse networks from scratch: Faster training without losing performance. arXiv preprint arXiv:1907.04840, 2019.  
Xin Dong, Shangyu Chen, and Sinno Pan. Learning to prune deep neural networks via layer-wise optimal brain surgeon. Advances in Neural Information Processing Systems, 30, 2017.  
Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro, and Erich Elsen. Rigging the lottery: Making all tickets winners. In International Conference on Machine Learning, pp. 2943-2952. PMLR, 2020.  
William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo Larochelle, Mark Rowland, and Will Dabney. Revisiting fundamentals of experience replay. In International Conference on Machine Learning, pp. 3061-3071. PMLR, 2020.  
Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International conference on learning representations, 2019.  
Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International conference on machine learning, pp. 1587-1596. PMLR, 2018.

Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In International Conference on Machine Learning, pp. 2052-2062. PMLR, 2019.  
Laura Graesser, Utku Evci, Erich Elsen, and Pablo Samuel Castro. The state of sparse training in deep reinforcement learning. In International Conference on Machine Learning, pp. 7766-7792. PMLR, 2022.  
Shixiang Gu, Ethan Holly, Timothy Lillicrap, and Sergey Levine. Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates. In 2017 IEEE international conference on robotics and automation (ICRA), pp. 3389-3396. IEEE, 2017.  
Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. Advances in neural information processing systems, 29, 2016.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861-1870. PMLR, 2018.  
Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. Advances in neural information processing systems, 28, 2015.  
Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In International conference on learning representations, 2016.  
J. Fernando Hernandez-Garcia and Richard S. Sutton. Understanding multi-step deep reinforcement learning: A systematic study of the dqn target. CoRR, abs/1901.07510, 2019. URL http://arxiv.org/abs/1901.07510.  
Hengyuan Hu, Rui Peng, Yu-Wing Tai, and Chi-Keung Tang. Network trimming: A data-driven neuron pruning approach towards efficient deep architectures. arXiv preprint arXiv:1607.03250, 2016.  
Maximilian Igl, Gregory Farquhar, Jelena Luketina, Wendelin Boehmer, and Shimon Whiteson. Transient non-stationarity and generalisation in deep reinforcement learning. In International conference on learning representations, 2021.  
Juhyoung Lee, Sangyeob Kim, Sangjin Kim, Wooyoung Jo, and Hoi-Jun Yoo. GST: Group-sparse training for accelerating deep reinforcement learning. arXiv preprint arXiv:2101.09650, 2021.  
Namhoon Lee, Thalaiyasingam Ajanthan, and Philip HS Torr. Snip: Single-shot network pruning based on connection sensitivity. In International conference on learning representations, 2019.  
Dor Livne and Kobi Cohen. Pops: Policy pruning and shrinking for deep reinforcement learning. IEEE Journal of Selected Topics in Signal Processing, 14(4):789-801, 2020.  
Christos Louizos, Max Welling, and Diederik P Kingma. Learning sparse neural networks through  $l_{-}0$  regularization. In International conference on learning representations, 2018.  
Decebal Constantin Mocanu, Elena Mocanu, Peter Stone, Phuong H Nguyen, Madeleine Gibescu, and Antonio Liotta. Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science. Nature communications, 9(1):1-12, 2018.  
Dmitry Molchanov, Arsenii Ashukha, and Dmitry Vetrov. Variational dropout sparsifies deep neural networks. In International Conference on Machine Learning, pp. 2498-2507. PMLR, 2017.  
P Molchanov, S Tyree, T Karras, T Aila, and J Kautz. Pruning convolutional neural networks for resource efficient inference. In 5th International Conference on Learning Representations, ICLR 2017-Conference Track Proceedings, 2019a.  
Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, and Jan Kautz. Importance estimation for neural network pruning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11264-11272, 2019b.

Hesham Mostafa and Xin Wang. Parameter efficient training of deep convolutional neural networks by dynamic sparse reparameterization. In International Conference on Machine Learning, pp. 4646-4655. PMLR, 2019.  
Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare. Safe and efficient off-policy reinforcement learning. Advances in neural information processing systems, 29, 2016.  
Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.  
Doina Precup. Eligibility traces for off-policy policy evaluation. Computer Science Department Faculty Publication Series, pp. 80, 2000.  
Andrei A Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, and Raia Hadsell. Policy distillation. In International conference on learning representations, 2016.  
Simon Schmitt, Jonathan J Hudson, Augustin Zidek, Simon Osindero, Carl Doersch, Wojciech M Czarnecki, Joel Z Leibo, Heinrich Kuttler, Andrew Zisserman, Karen Simonyan, et al. Kickstarting deep reinforcement learning. arXiv preprint arXiv:1803.03835, 2018.  
David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic policy gradient algorithms. In International conference on machine learning, pp. 387-395. PMLR, 2014.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676):354-359, 2017.  
Ghada Sokar, Elena Mocanu, Decebal Constantin Mocanu, Mykola Pechenizkiy, and Peter Stone. Dynamic sparse training for deep reinforcement learning. arXiv preprint arXiv:2106.04217, 2021.  
Suraj Srinivas, Akshayvarun Subramanya, and R Venkatesh Babu. Training sparse neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp. 138-145, 2017.  
Enzo Tartaglione, Skjalg Lepsøy, Attilio Fiandrotti, and Gianluca Francini. Learning sparse neural networks via sensitivity-driven regularization. Advances in neural information processing systems, 31, 2018.  
Gerald Tesauro et al. Temporal difference learning and td-gammon. Communications of the ACM, 38(3):58-68, 1995.  
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pp. 5026-5033. IEEE, 2012.  
Marc Aurel Vischer, Robert Tjarko Lange, and Henning Sprekeler. On lottery tickets and minimal task representations in deep reinforcement learning. In International conference on learning representations, 2022.  
Chaoqi Wang, Guodong Zhang, and Roger Grosse. Picking winning tickets before training by preserving gradient flow. arXiv preprint arXiv:2002.07376, 2020.  
Haonan Yu, Sergey Edunov, Yuandong Tian, and Ari S Morcos. Playing the lottery with rewards and multiple languages: lottery tickets in rl and nlp. In International conference on learning representations, 2020.  
Hongjie Zhang, Zhuocheng He, and Jing Li. Accelerating the deep reinforcement learning with neural network compression. In 2019 International Joint Conference on Neural Networks (IJCNN), pp. 1-8. IEEE, 2019.  
Michael Zhu and Suyog Gupta. To prune, or not to prune: exploring the efficacy of pruning for model compression. In Workshop at international conference on learning representations, 2018.
