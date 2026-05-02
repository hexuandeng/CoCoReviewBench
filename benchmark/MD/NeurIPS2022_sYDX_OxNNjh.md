# Unsupervised Skill Discovery via Recurrent Skill Training

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Being able to discover diverse useful skills without external reward functions is beneficial in reinforcement learning research. Previous unsupervised skill discovery approaches mainly train different skills in parallel. Although impressive results have been provided, we found that such parallel training procedure inherently discourages exploration, which leads to poor state coverage and restricts the diversity of learned skills. In this paper, we take a deeper look into this phenomenon and propose a novel framework to address this issue, which we call Recurrent Skill Training (ReST). Instead of training all the skills in parallel, ReST trains different skills one after another recurrently, along with a state coverage based intrinsic reward. We conduct experiments on a number of challenging 2D navigation environments and robotic locomotion environments. Evaluation results show that our proposed approach outperforms previous parallel training approaches in terms of state coverage and skill diversity. Videos of the discovered skills are available at https://sites.google.com/view/neurips22-rest.

# 1 Introduction

Recent advances in deep reinforcement learning have shown its promising performance in domains ranging from game playing [2, 3], robotics [4, 5] and recommender systems [6]. These applications of reinforcement learning rely on task-specific reward functions for the agents to successfully accomplish the tasks. However, intelligent creatures can automatically explore the environments and learn diverse useful skills in the absence of external supervision. Such ability is beneficial in a variety of situations. For tasks where rewards are non-trivial to design or where the reward signal is sparse, unsupervised skill discovery approaches can provide intrinsic rewards to help accomplish tasks. Moreover, in hierarchical control problems, unsupervised skill discovery can serve as low-level policies for downstream tasks [7, 1].

![](images/b31581d28b7213d1e0ecf7818f1d4589830eed8d2db3e65a0bd5c503d9179948.jpg)  
(a) Explored

![](images/b3cf4a43274cb5e379224f4873741eb1e519068e21a7325edd88bdfe47cb60f4.jpg)  
(b) Converge

![](images/85bf8999c64bffa8d239b8ee2189e1eeb1122e39eecaef76e2ff44bf1f9a04d1.jpg)  
Figure 1: Skills discovered in a 2D navigation environment. (a) shows the explored states of skills during the training phase of a baseline method [1], which indicates that it has successfully passed through the bottleneck and explored the states on the right room. (b) shows the states covered by the converged skills of baseline, which does not reach states on the right. Our proposed method, as shown in (c), successfully reached the states on the right after convergence.  
(c) ReST (ours)

Although existing works have shown great potential in discovering diverse useful skills in an unsupervised manner [1, 7, 8], one of the key problems of such methods, as observed in our preliminary experiments and some recent works [9, 10], is that they might suffer from poor state coverage,

which may lead to failures in learning desirable useful skills. For instance, in robotic locomotion environments, previous approaches tend to learn 'posing' skills instead of dynamic, far-reaching skills [1]. A straightforward explanation for this phenomenon would be the lack of exploration [9]. However, we argue that this is not the only reason for the poor state coverage. Counterintuitively, we observed that the skills after convergence may avoid visiting certain states even if they are explored during training. For instance, as shown in our preliminary 2D navigation experiments, the discovered skills fail to cover the states passing through the bottleneck to the right (Figure 1b), even if they have been explored during training (Figure 1a). We call this phenomenon Exploration degradation.

In this paper, we take a deeper look into the above phenomenon and show that it is mainly caused by the parallel training paradigm, which is a common choice for most existing works [1, 7, 8, 10]. When multiple skills trained in parallel have visited the same state, such state will be prevented from being visited again. Detailed analysis is provided in Section 3.1. Based on the analysis, we propose Recurrent Skill Training (ReST), an unsupervised skill discovery algorithm that addresses the exploration degradation issue. Instead of training all the skills in parallel, ReST trains the skills one after another in a recurrent fashion, along with an intrinsic reward that discourages covering frequently visited states of other skills. A preliminary result of using ReST is shown in Figure 1c, where the exploration degradation problem is eliminated and the converged skills are able to visit the space in the right room. Evaluation results on complex 2D navigation and robot locomotion tasks show that our approach can achieve better state coverage and skill divergence compared to baselines.

Our contributions are summarized as follows:

- We discover a new phenomenon that reduces state coverage called exploration degradation, which indicates that some certain states are discouraged from being visited by the learned skills, even if they have been explored during training.  
- We show that the main reason causing exploration degradation is that multiple skills visiting the same states can reduce the MI reward in the parallel training paradigm. We then propose Recurrent Skill Training (ReST), a recurrent training paradigm along with a state coverage based intrinsic reward, which prevents multiple skills from visiting the same states and alleviates the exploration degradation issue.  
- We conduct experiments on various 2D navigation tasks and robot locomotion tasks. Evaluation results show that our method achieves better state coverage and divergence compared to baseline methods. Moreover, ReST learns diverse meaningful robot locomotion skills that have not been shown in previous works.

# 2 Preliminaries

# 2.1 Markov Decision Process

Markov decision process (MDP) can be used to find a reward maximizing policy. It is represented by a tuple  $(\mathcal{S},\mathcal{A},\mathcal{P},R,\gamma ,\mu)$ , where  $\mathcal{S}$ ,  $\mathcal{A}$  are the state and action spaces.  $\mathcal{P}:\mathcal{S}\times \mathcal{A}\times \mathcal{S}\rightarrow [0,1]$  is the transition dynamic that maps the state and action into a probability distribution over the next state.  $R:S\times \mathcal{A}\to \mathbb{R}$  is the reward function.  $\gamma$  is the discount factor for the reward function while  $\mu :S\rightarrow [0,1]$  is the initial state distribution. The expected discounted cumulative reward can be formulated as  $J_{R}(\pi) = \mathbb{E}_{\tau \sim \pi}[\sum_{t = 0}^{\infty}\gamma^{t}R(s_{t},a_{t})]$ . Thus the overall optimization problem can be written as

$$
\pi^ {*} = \underset {\pi} {\arg \max } J _ {R} (\pi) \tag {1}
$$

# 2.2 Unsupervised Skill Discovery

Generally speaking, unsupervised skill discovery aims to find a family of skills conditioned on latent  $z$ , which results in a latent-conditioned policy  $\pi(a|s,z)$  that maximizes the mutual information between state and latent:

$$
\begin{array}{l} I (S; Z) = H (Z) - H (Z \mid S) (2) \\ = H (S) - H (S \mid Z) (3) \\ \end{array}
$$

where  $s \in S$ ,  $a \in \mathcal{A}$ ,  $z \in \mathcal{Z}$  are the state, action and latent respectively. Let  $(S,Z) \sim p(s,z)$  as the random variables of the state distribution and the latent distribution. As suggested by [11], a variational lower bound for Equation (2) can be derived as:

$$
\begin{array}{l} I (S; Z) = \mathbb {E} _ {(z, s) \sim p (z, s)} [ \log p (z | s) - \log p (z) ] (4) \\ \geq \mathbb {E} _ {(z, s) \sim p (z, s)} [ \log q _ {\phi} (z | s) - \log p (z) ] (5) \\ \end{array}
$$

where  $q_{\phi}(z|s)$  is a learned discriminator approximating  $p(z|s)$ . Such lower bound also exists for Equation (3):

$$
\begin{array}{l} I (S; Z) = \mathbb {E} _ {(z, s) \sim p (z, s)} [ \log p (s | z) - \log p (s) ] (6) \\ \geq \mathbb {E} _ {(z, s) \sim p (z, s)} \left[ \log q _ {\phi} (s | z) - \log p (s) \right] (7) \\ \end{array}
$$

where  $q_{\phi}(s|z)$  is a learned function approximator of  $p(s|z)$ .

# 3 Recurrent Skill Training

![](images/c44649c596a2dacd7145b6248dd778ddc71f7a714ead3f1b4cec5fe6c63a3058.jpg)  
(a) Explored  
Figure 2: Grid world example. The orange grids are states explored by both the two skills and the green and yellow grids denotes states explored by skill 1 and 2 respectively. Blue grids are unvisited states. (a) shows the state visitation map during exploration while (b) and (c) shows the state visitation of the final converged policies of parallel and recurrent training paradigms respectively.  
(b) Parallel  
(c) Recurrent

In this section, we first introduce the exploration degradation phenomenon of previous skill discovery approaches with parallel training paradigm. Then we present our proposed recurrent skill training method that addresses this issue.

![](images/191c9d4957f832e281d22a9521975addc49a956c306a9958c41e2946e411d755.jpg)  
(a) Illustration of recurrent training paradigm  
Figure 3: ReST Algorithm. Instead of training different skills in parallel, ReST trains skills one after another recurrently to optimize a state coverage based intrinsic reward.

![](images/557abc7239a8da656e2069584e2a94624e26fb88c90c51c66506ea6edf195245.jpg)  
(b) Illustration of ReST

Algorithm 1 Recurrent Skill Training  
Initialize: random initialized skill  $\pi$ , RND networks  $\hat{f}$  and  $f$   
for  $i = 1$  to  $N$  do  
Set skill  $\pi_i = \pi$   
Set RND network  $\hat{f}_i = \hat{f}, f_i = f$   
Collect on-policy samples with  $\pi_i$   
Update RND network  $\hat{f}_i$  by minimizing loss in Equation (8)  
end for  
repeat  
for  $i = 1$  to  $N$  do  
for  $j = 1$  to  $M$  do  
    Collect on-policy samples with  $\pi_i$ ;  
    Calculate reward using Equation (10);  
    Update skill policy  $\pi_i$  using any RL algorithms;  
    Update RND network  $\hat{f}_i$  by minimizing loss in Equation (8) using the latest on-policy samples;  
    end for  
end for  
until convergence

# 3.1 Exploration Degradation

As we observed in our preliminary experiment shown in Figure 1, previous unsupervised skill discovery approaches might suffer from the exploration degradation problem, such that some explored states (e.g., states near the bottleneck and in the right room in Figure 1a) are prevented from being visited by the learned skills. We now provide a simplified analysis to show that this phenomenon is mainly caused by the parallel training paradigm commonly used in previous works.

Let us focus on discrete latent case with  $N$  different skills. Consider at state  $s$ , skill  $k$  visited state  $s$  with  $p(z_k|s)$  probability, which is modeled by the discriminator  $q_{\phi}(z_k|s)$ . Assume state  $s_0$  is only visited by skill  $k$  whereas  $s_1$  is visited by multiple skills, which means  $p(z_k|s_0) = 1$  and  $p(z_k|s_1) < 1$ . As long as the discriminator  $q_{\phi}$  can model the difference and output  $q_{\phi}(z_k|s_0) > q_{\phi}(z_k|s_1)$ , which is not a strong assumption, the intrinsic reward used by MI approaches would encourage the visitation of state  $s_0$  by skill  $k$  and discourage visiting  $s_1$ . This means MI based approaches end up with skills that prefer exploiting states only visited by themselves in the previous training epochs, causing exploration degradation.

For clarity, we further explain this issue with a toy example. Consider a  $2 \times 2$  grid world as shown in Figure 2. There are four states  $0,1,2,3$  and at each state, there are three actions to take: go to the two adjacent states or stay where it is. For simplicity, we use the number of the next state to denote the action, for instance, if the agent is in state 0 and chooses to go to state 1, then the action will be denoted as 1. We use the mutual information described in Equation (2) and assume that the discriminator is perfect:  $q_{\phi}(z|s) = p_{\pi}(z|s)$ . Consider a case when the number of skills is  $N = 2$  and during the first collection of samples, skill 1 visited  $\{0,2,3\}$  while skill 2 visited  $\{0,1,3\}$ , as shown in Figure 2a. Since we have a perfect discriminator, for state 3 the discriminator would output  $q_{\phi}(z_1|3) = q_{\phi}(z_2|3) = 0.5$ , which results in 0 reward for both skills. Therefore, the optimal converged policy for skill 1 would generate trajectory  $\{0,2,2\}$  while skill 2 would generate trajectory  $\{0,1,1\}$ , as shown in Figure 2b. State 3 is not covered, which is undesirable. This example indicates that even in such an extremely simple case, the exploration degradation phenomenon still exists.

# 3.2 Recurrent Skill Training

Instead of training all skills in parallel, we propose a recurrent training paradigm, along with a state coverage intrinsic reward. We now introduce the details of our proposed method.

Recurrent Training Paradigm. As analyzed in Section 3.1, the main reason causing the exploration degradation issue is that the same states are visited by multiple skills in the parallel training paradigm. A natural way to alleviate this issue is to train the skills one after another recurrently. In this way, we

can encourage the latter trained skills to avoid entering the same states covered by the previous skills. Figure 3a illustrates the recurrent training paradigm compared with parallel training. Starting with  $N$  randomly initialized skills ( $N = 3$  in this case), the recurrent training paradigm trains one skill at each epoch whereas the parallel training paradigm trains all the skills together. Furthermore, in order to improve convergence, the recurrent training paradigm updates each skill for  $M$  epochs ( $M = 2$  in this case) before switching to another skill. In this work, we use  $N$  independent neural networks to parametrize the  $N$  skills, which can be considered as discrete latent conditioned policies.

State Coverage Intrinsic Reward. When implementing the above recurrent training paradigm, the latter trained skill needs to avoid visiting the states frequently visited by other skills. In order to accomplish this objective, we need to identify how frequently each state is visited by each skill, or equally, how novel a given state is to a skill. In this paper, we adopt random network distillation (RND) [12], a simple yet scalable novelty detection approach, to estimate the novelty of a state to a specific skill.

For each skill, we have two neural networks for state novelty detection: a randomly initialized fixed target network and a prediction network. During training, the fixed target network is set as the ground truth while the prediction network is trained to fit the target network's output. The input data is composed of the states visited by this skill, which follows the state distribution  $s \sim p(s|z_i)$ . We denote the target network for skill with latent  $z_i \in \mathcal{Z}$  as  $f_i : S \to \mathbb{R}^k$  and the prediction network as  $\hat{f}_i : S \to \mathbb{R}^k$ , where  $k$  is the dimension of the two networks' output and  $i \in \{1, 2, \dots, N\}$  is the index of the skill. The prediction network  $\hat{f}_i$  is optimized by gradient descent to minimize the expected mean square error loss  $\mathcal{L}_i$ :

$$
\mathcal {L} _ {i} = \mathbb {E} _ {s \sim p (s | z _ {i})} \left[ \left| \left| \hat {f} _ {i} (s) - f _ {i} (s) \right| \right| ^ {2} \right] \tag {8}
$$

As suggested by [12], with the learned prediction network, we can estimate the frequency of state  $s$  visited by the  $i$ th skill using the prediction error  $||\hat{f}_i(s) - f_i(s)||^2$ . Intuitively, a higher prediction error indicates higher uncertainty of  $\hat{f}_i$  on this state, which further implies its higher novelty. Since we need to avoid visiting states visited by other skills when training a certain skill, a straightforward solution is to define a reward function for skill  $i$  with state  $s_t$  and action  $a_t$  as:

$$
r _ {i} \left(s _ {t}, a _ {t}\right) = \min  _ {j \in \{1, 2, \dots , N \}, j \neq i} \left\| \hat {f} _ {j} \left(s _ {t + 1}\right) - f _ {j} \left(s _ {t + 1}\right) \right\| ^ {2} \tag {9}
$$

such that states frequently visited by any other skills are less desired. However, this minimum operator would make the reward landscape rugged, which might lead to poor convergence property. To stabilize the training process, we introduce a soft version of the minimum operator in (9):

$$
r _ {i} \left(s _ {t}, a _ {t}\right) = - \log \left[ \frac {\sum_ {j \in \{1 , 2 , \dots , N \} , j \neq i} e ^ {\left(- \alpha \cdot \left\| \hat {f} _ {j} \left(s _ {t + 1}\right) - f _ {j} \left(s _ {t + 1}\right) \right\| ^ {2}\right)}}{N - 1} \right] \tag {10}
$$

where  $\alpha$  is a task-specific temperature parameter.

Practical Algorithm. We summarize our algorithm in Figure 3b and Algorithm 1. Firstly, we randomly initialize a policy network  $\pi_{i}$  and a pair of RND networks  $f_{i}$  and  $\hat{f}_{i}$  for each skill  $i\in$ $\{1,2,\dots,N\}$ . Before training skills, ReST first collects on-policy samples for each initialized skill and train their RND networks. Then we train each skill's policy network and the corresponding RND networks for  $M$  times during each training epoch. The skill networks are trained recurrently until convergence. For maximizing the intrinsic reward, the ReST algorithm can be combined with an arbitrary reinforcement learning algorithm. In this paper we choose Proximal Policy Optimization (PPO) [13] with generalized advantage estimation (GAE) [14].

Grid World Example. We further illustrate the effectiveness of the proposed algorithm using the  $2 \times 2$  grid world example with the same setting as in Section 3.1. As suggested by our previous analysis, the parallel training paradigm will end up with two skills that neither of them visits state 3. As mentioned in Section 3.1, the initial two skills visits  $\{0,2,3\}$  and  $\{0,1,3\}$  respectively. Here we assume that the RND networks  $f_{i}$  and  $\hat{f}_i$  perfectly obtain the state visitation frequency, which means:

$$
\left| \left| \hat {f} _ {i} (s) - f _ {i} (s) \right| \right| ^ {2} = \left\{ \begin{array}{l l} 0 & \text {i f s t a t e s i s v i s i t e d b y s k i l l i} \\ r & \text {o t h e r w i s e} \end{array} \right. \tag {11}
$$

where  $r > 0$  is the prediction error. ReST starts by training the corresponding RND networks of each skill and then recurrently train different skills to maximize the intrinsic reward. For skill 1, visiting state 2 would gain reward  $r$  while visiting state 1 and 3 would get zero rewards. Therefore, the optimal policy for skill 1 would be visiting  $\{0,2,2\}$  in sequence. After updating the RND networks of skill 1 correspondingly, for skill 2, visiting state 1 and 3 would gain reward  $r$  while visiting 2 would get zero rewards. Therefore, the initial skill 2 is already the optimal policy and the training converges. This way, the optimal trajectory for skill 1 is  $\{0,2,2\}$  while for skill 2 it is  $\{0,1,3\}$ , as shown in Figure 2c. Therefore, the state space of the  $2 \times 2$  grid world is fully covered by the two skills.

# 4 Experiments

![](images/c1c95b0b5a27a282a4fd98b0e2eda73bfd6bb31c008390e06b170cc35054e066.jpg)  
(a) DoorMaze

![](images/50874a173dcfbfd41ad59144f3c37c1605767b83124f8ae599f81d2b7da88f91.jpg)  
(b) CenterMaze

![](images/7eee5161feb106a489908c533a4874fc0d863bf67e803bf605aec7e2020584a3.jpg)

![](images/2bf0e660d59c49f743fe3bde7e430d1eb5445b834ec99d32116fbda69b700bbd.jpg)  
(c) 4RoomMaze

![](images/65036f1beebcaabdc7fc31d4c07db471e7b705609178b8851c7db04ddff82ce5.jpg)  
Figure 4: Results for 2D navigation experiments. The left five columns of the figure show qualitative results of skills discovered by different algorithms in the corresponding navigation environments. We trained 10 skills for each algorithm where 20 trajectories are rendered for each discovered skill. The right two columns of the figure show the quantitative results of different algorithms. SC stands for state coverage while MI stands for mutual information. SC and MI are calculated based on converged models of each algorithm with three different random seeds. We calculate the SC and MI metrics three times for each converged skill to draw the histogram. Detailed analysis of the evaluation metrics can be found in the Appendix  
(d) 9RoomMaze  
We conduct experiments on several challenging 2D navigation environments and several robotic locomotion tasks. We compare our proposed approach with two of the most popular unsupervised skill discovery approaches, DIAYN [1] and DADS [7]. Both of these approaches use the parallel training paradigm to optimize different skills. Since our proposed approach parameterizes each skill with an independent neural network, we also compare our proposed approach with DIAYN and DADS using independent neural networks, which we call DIAYN-i and DADS-i respectively, for a fair comparison. We present our empirical results both qualitatively and quantitatively. The quantitative results introduce a state coverage metric and a mutual information metric. Our proposed approach

![](images/62600b5143b7d2e3b04a40620ce2d36f2c94a5cb3455495172f7c3e11d9da7f5.jpg)  
(a) HalfCheetah flipped running forward

![](images/3cd7ae57fad881ac3f2388a9f235bf7c6e784054df9deff81949bfd94669e99b.jpg)  
(b) HalfCheetah rolling forward

![](images/153946f1a7120cfd44ad834092a77a2e3b04e7d80ed2c72a894dae069973fa34.jpg)  
(d) HalfCheetah running backward

![](images/3b4752d42cb78fe27c5915fd15316d820a3d5130a5c37aab4e11c20d5b0f059a.jpg)  
(c) HalfCheetah rolling backward

![](images/c95281a338c70f97321385aa78dd96c743844ddb90b88e7e27e998792e4cd4f1.jpg)  
(e) Hopper hopping forward

![](images/63a1b94c46d794fae0a19ef937097e16bd1d144ef90b3924e9ae97ea1ad227fc.jpg)  
(f) Hopper crawling forward

![](images/8bc16a2ff0f34bfcdb306f51e0b8b5d99868b8f87c301dd2ee4cc0237eececed.jpg)  
(g) Hopper hopping backward

![](images/c7d27e48d1a1856e14bf089d8578ea842ca4b8a02a80d58e142f602b904b4208.jpg)  
(h) Hopper backflip

![](images/cd89426b2b0026c35a14c3aa726e580b4963d31afb973ad1f3b171588740b569.jpg)  
(i) Walker2d trotting forward

![](images/5a317dbdfd8e5526c5b53002e59329d1ce92c4b351eeb022a1630f9b4f8834fa.jpg)  
(j) Walker2d dashing forward

![](images/a71f3b36f9cd3ed7209c88a500de6c83f5cf99a621cdbe74f69180c3c1060de7.jpg)  
(k) Walker2d backward  
digging (1) Walker2d backward

![](images/dfb74393f93e858bba3461df263a1ec29691ffee4080c96e4e76576914a49430.jpg)  
walking

![](images/0c5a92b99f955a80e5364641f3ff93c3c408011621bbbba29d3a653dc5af9e0f.jpg)  
Figure 5: Visualization of skills discovered using ReST. The proposed ReST algorithm discovers several high-quality, diverse skills for each robot, including running, flipped running, backflip, dashing, etc. The arrows on the bottom of each sub-figure show the moving direction of each skill.

![](images/df192e6b12e89567203956327311edea02ea60edabe8601c1cb6b8b3c9dd88a4.jpg)  
ReST (ours)  
DIAYN

![](images/7635883a2d26602bfb768856229076edc0fc655d100eb95d108b521575aa9fd9.jpg)  
DADS  
(a) HalfCheetah

![](images/20ad90aedbfd0f81e1004163f4675161ee7c3f4cecbf9c7691ebac441007e5cb.jpg)  
DIAYN-i

![](images/eaeed67d702da3dd43f7907cb8317ac13562709da9b96259fba349ea5d2676f3.jpg)  
DADS-i

![](images/fd92b54412364e7abbd8a32a573f7d1c8aa07f4e54e7f13314c15540bee2cca1.jpg)

![](images/88231abb6aee7e843d0bd26604df1ff425f5e3eb4e42361a3425815382bdac28.jpg)

![](images/39e6d505cdd9c02f7cc4c62a818d125581f56d624a7bbc9e5c20e5ae87d2b650.jpg)  
(b) Hopper

![](images/b1e691ccdb567b73724b5c3c5a7efbf556706dae0e38d146ee6952e1d9462303.jpg)

![](images/29d40432e7b186c6927f6e26029a6c2b39b9ba51f0376c582692c432e4079c98.jpg)

![](images/74308adeafe8185fe501731cd8e70d75feb2fda53428ac48c893c42347b9ac7d.jpg)  
Figure 6: Semi-quantitative results for robotic locomotion tasks. The horizontal axis represents the time-step during evaluation while the vertical axis represents the  $x$  position of the robot at each time step. Each color represents a discovered skill and we rollout three trajectories for each skill. Generally speaking, baseline methods discover 'posing' skills instead of dynamic locomotion skills whereas ReST discovers dynamic, far-reaching locomotion skills.

![](images/cdfdda221affd1275665977e5dbedfd852efa73d8c3986656c6bf6f469ab0e4c.jpg)

![](images/d99482576f458e47bfa4315bb9651d32225b1fce128935ddce340fc091bf4061.jpg)  
(c) Walker2d

![](images/fa5b29f53b6200f148e299f218c1c990adea7c2d929546dce08bdff170b451f2.jpg)

![](images/6a6497385265ceaa46e3d730882f8deb4ea6ba7768dbad7f1329ea666d17294a.jpg)

significantly outperforms previous skill discovery approaches in terms of state coverage while staying comparable with previous approaches in terms of mutual information. Qualitative results include visualizations of the converged navigation experiments, the state visitation of robotic locomotion tasks, and rendered videos of novel skills discovered using our proposed approach, which can be found in our project website https://sites.google.com/view/neurips22-rest. Details of our implementation can be found in the Appendix.

# 4.1 2D Navigation Tasks

192 Environments. We conducted experiments on several challenging 2D navigation environments. The agent is a point mass navigating in a 2D plane with boundaries  $[0,1]\times [0,1]$ . The observation space of the environment has 4 dimensions, including the  $x\in [0,1]$  and  $y\in [0,1]$  position on the plane

and the corresponding velocity  $v_{x}$  and  $v_{y}$  belonging to the velocity space  $\{(v_{x}, v_{y}) | v_{x}^{2} + v_{y}^{2} \leq 0.01\}$ . The action space has 2 dimensions, including the acceleration  $a_{x} \in [0, 0.1]$  and  $a_{y} \in [0, 0.1]$  of the point mass agent. There are walls inside the plane and the agent cannot go through the walls. We designed diverse placements of the walls with increasing difficulties for exploring the environments, which could be used to test the effectiveness of our proposed approach.

Evaluation Metrics. We introduce two metrics to compare our proposed approach with our comparison baselines. The first one is the state coverage metric. State coverage matters since not covering enough state space might result in failures in learning desirable useful skills. We evaluate the state coverage on the  $X - Y$  plane by first decomposing the environment into cells and then testing whether each cell is successfully visited. We roll out 1 trajectory for each skill and use the visited states to calculate the state coverage metric. The percentage of cells visited by at least one of the skills is the state coverage. Moreover, merely covering the state space is not enough. The skills need to be informative about which states they are going to visit in the environment so that the skills are meaningful. We use mutual information between skill latent  $z$  and the corresponding covered states  $s$  to quantify how informative the skill is. Similarly, we decompose the state space into multiple cells and roll out 20 trajectories for each skill to record the probability distribution of states  $p(s)$  for all the states visited by the skills. Then we record the probability distribution of states visited by each skill  $p(s|z)$  and calculate the entropy  $H(S)$  and  $H(S|Z_i)$ . Since the skill distribution  $p(z)$  is a uniform distribution for all skills, we can calculate the mutual information as:

$$
I (S; Z) = H (S) - \frac {1}{N} \left(\sum_ {i \in \{1, 2, \dots , N \}} H (S | Z _ {i})\right) \tag {12}
$$

Qualitative Results. We train  $N = 10$  skills for each of the algorithms and rollout 20 trajectories to obtain the qualitative results. We conduct experiments on 4 challenging 2D navigation mazes, which we call the DoorMaze, CenterMaze, 4RoomMaze and 9RoomMaze, which are shown in Figure 4. The left 5 columns show the qualitative results. Each rolled-out trajectory is rendered as a curved line and different colors represent different skills. The results indicate that our proposed ReST algorithm can reach more diverse states in the environments whereas baseline approaches like DIAYN or DADS can only reach a small portion of the environments. Usually, the baseline approaches cannot pass through the 'bottlenecks' in the environment. The results support our insight that the parallel training paradigm is one of the causes of previous skill discovery approaches' state coverage issues. Moreover, when using independent neural networks to parameterize different skills, we observe that the performance is not so different from the latent-conditioned parameterization version.

Quantitative Results. Beyond qualitative results, we quantify the state coverage and the informativeness of different algorithms using the aforementioned two metrics. As shown in the right two columns of Figure 4, our proposed ReST algorithm significantly outperforms baseline approaches with a parallel training paradigm in terms of state coverage and is comparable with baseline approaches in terms of mutual information  $I(S;Z)$ . The above results indicate that our proposed approach can cover diverse sets of states without too much sacrifice on informativeness. Quantitative results also evidenced that the difference in performance is not because of the different parameterization since DIAYN and DADS using independent neural networks are not so different from the original ones.

# 4.2 Robotic Locomotion Tasks

We present qualitative results of robotic locomotion skills discovered using our proposed ReST algorithm in MuJoCo [15]. Generally speaking, our proposed ReST algorithm can discover dynamic, far-reaching robotic locomotion skills whereas DIAYN and DADS tend to discover 'posing' skills. Figure 5 shows visualizations of parts of skills discovered using ReST. There are several novel skills discovered, such as Hopper backflip, that have not been presented in previous works to the best of our knowledge. More rendered results can be found in the Appendix and our project website https://sites.google.com/view/neurips22-rest.

Moreover, we provide semi-quantitative results of the proposed approach. As shown in Figure 6, we draw the agent's  $x$  position over timestep, using the skills discovered by ReST and the comparison baselines. The results are evaluated on HalfCheetah, Hopper and Walker2d tasks. We use the OpenAI Gym [16] settings of the three tasks, where HalfCheetah is trained with fixed episode

length whereas Hopper and Walker2d terminate when the agents fall during training. The resulting timestep-  $x$  curve indicates that our proposed approach learns skills that are more far-reaching, diverse, and dynamic than baseline methods. This also evidenced that our recurrent training paradigm outperforms the parallel training-based baselines by alleviating their state coverage issues.

# 5 Related Work

Unsupervised Skill Discovery. Previous unsupervised skill discovery approaches mainly focus on maximizing the mutual information  $I(S;Z)$  to obtain meaningful skills. As discussed in Equation (2) and Equation (3), the mutual information has two forms. Several works follow the Equation (2). VIC [17] uses the mutual information between the final states and the skill latent as the intrinsic reward and optimizes it via reinforcement learning. DIAYN [1] fixes the prior  $p(z)$  and uses the mutual information between the skill and its visited states as its intrinsic reward to learn meaningful skills, which has superior performance compared to VIC. VALOR [8] further improves DIAYN by replacing the state-based objective with a trajectory-based objective. DADS [7], on the other hand, uses the Equation (3) form of mutual information and learns a transition model  $q_{\phi}(s'|s,z)$  and uses model predictive control to solve downstream tasks. EDL [9] provides insight into why previous approaches suffer from lack of exploration and proposes an algorithm using a fixed state prior  $p(s)$  that to alleviate the issues. IBOL [10] tries to relieve the difficulty of reaching diverse states by introducing a low-level controller. Besides learning a set of skills, SMM [18] formulates skill discovery as a state marginal matching problem and optimizes the KL divergence between the expected state distribution and the current policy's state distribution. MUSIC [19] improves previous unsupervised skill discovery algorithms by adding the mutual information between the surrounding state and the agent state. DDL [20] learns one skill that maximizes the dynamical distance functions of the previous skill. Besides the above approaches, there are also other unsupervised skill discovery methods [21, 22, 23, 24].

Intrinsic Reward. Another stream of works related to this work is intrinsic reward. In our proposed approach, we use intrinsic reward as an objective to help learn a set of meaningful skills. Intrinsic reward can also help with cases where rewards are sparse by augmenting them to the original reward function. Count-based exploration [25, 26, 27, 28] uses pseudo count to identify the frequently visited states and the less frequently visited ones and adds the count-based bonus as intrinsic rewards to accelerate exploration. Prediction error exploration methods [29, 30, 12, 31, 32] make use of prediction errors as intrinsic rewards based on an insight that states with high prediction error should have higher novelty. Other works augment an information-theoretic intrinsic reward with extrinsic rewards that encourage information gain about the environments [33, 34, 35].

# 6 Discussion

In this paper, we proposed a novel, effective yet simple algorithm called Recurrent Skill Training (ReST). We began by introducing a new phenomenon called exploration degradation which reduces state coverage of the learned skills. We found the key reason for this phenomenon is the parallel training paradigm commonly used in previous skill discovery approaches, such that the same states visited by multiple skills are discouraged from being visited again. Instead of training all skills in parallel at each epoch, ReST trains different skills one after another recurrently. This recurrent training paradigm is supported by an effective prediction error-based intrinsic reward inspired by novelty detection methods. We then conducted experiments on 2D maze navigation to continuous robotic control tasks. Both qualitative and quantitative results show that ReST is able to discover more diverse skills with better state coverage compared to baseline algorithms. Moreover, we demonstrated several novel and dynamic robot locomotion skills that have not been presented in previous works.

There are also some limitations of the proposed algorithm. First of all, compared with previous approaches, our proposed approach has worse sample complexity during skill discovery since only one skill is trained at each epoch, as shown in Figure3a. Moreover, the computational complexity is higher than approaches like [1] or [10] since ReST needs to compute intrinsic reward based on all other skills' prediction errors. Finally, due to the recurrent training paradigm, ReST is currently not scalable to continuous latent, which is in general a better choice as a low-level controller for hierarchical control in downstream tasks. Future works include addressing the above limitations and apply ReST to downstream/hierarchical tasks.

# References

[1] Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function. arXiv preprint arXiv:1802.06070, 2018.  
[2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529-533, 2015.  
[3] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484-489, 2016.  
[4] OpenAI: Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, et al. Learning dexterous in-hand manipulation. The International Journal of Robotics Research, 39(1):3-20, 2020.  
[5] Alex Kendall, Jeffrey Hawke, David Janz, Przemyslaw Mazur, Daniele Reda, John-Mark Allen, Vinh-Dieu Lam, Alex Bewley, and Amar Shah. Learning to drive in a day. In 2019 International Conference on Robotics and Automation (ICRA), pages 8248-8254. IEEE, 2019.  
[6] Guy Shani, David Heckerman, Ronen I Brafman, and Craig Boutilier. An mdp-based recommender system. Journal of Machine Learning Research, 6(9), 2005.  
[7] Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, and Karol Hausman. Dynamics-aware unsupervised discovery of skills. arXiv preprint arXiv:1907.01657, 2019.  
[8] Joshua Achiam, Harrison Edwards, Dario Amodei, and Pieter Abbeel. Variational option discovery algorithms. arXiv preprint arXiv:1807.10299, 2018.  
[9] Víctor Campos, Alexander Trott, Caiming Xiong, Richard Socher, Xavier Giró-i Nieto, and Jordi Torres. Explore, discover and learn: Unsupervised discovery of state-covering skills. In International Conference on Machine Learning, pages 1317–1327. PMLR, 2020.  
[10] Jaekyeom Kim, Seohong Park, and Gunhee Kim. Unsupervised skill discovery with bottleneck option learning. arXiv preprint arXiv:2106.14305, 2021.  
[11] David Barber Felix Agakov. The im algorithm: a variational approach to information maximization. Advances in neural information processing systems, 16(320):201, 2004.  
[12] Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018.  
[13] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
[14] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation, 2018.  
[15] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026-5033. IEEE, 2012.  
[16] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.  
[17] Karol Gregor, Danilo Jimenez Rezende, and Daan Wierstra. Variational intrinsic control. arXiv preprint arXiv:1611.07507, 2016.  
[18] Lisa Lee, Benjamin Eysenbach, Emilio Parisotto, Eric Xing, Sergey Levine, and Ruslan Salakhutdinov. Efficient exploration via state marginal matching. arXiv preprint arXiv:1906.05274, 2019.

[19] Rui Zhao, Yang Gao, Pieter Abbeel, Volker Tresp, and Wei Xu. Mutual information state intrinsic control. arXiv preprint arXiv:2103.08107, 2021.  
[20] Kristian Hartikainen, Xinyang Geng, Tuomas Haarnoja, and Sergey Levine. Dynamical distance learning for semi-supervised and unsupervised skill discovery. arXiv preprint arXiv:1907.08225, 2019.  
[21] Hao Liu and Pieter Abbeel. Aps: Active pretraining with successor features. In International Conference on Machine Learning, pages 6736-6747. PMLR, 2021.  
[22] Hao Liu and Pieter Abbeel. Behavior from the void: Unsupervised active pre-training. arXiv preprint arXiv:2103.04551, 2021.  
[23] David Warde-Farley, Tom Van de Wiele, Tejas Kulkarni, Catalin Ionescu, Steven Hansen, and Volodymyr Mnih. Unsupervised control through non-parametric discriminative rewards. arXiv preprint arXiv:1811.11359, 2018.  
[24] Pierre-Alexandre Kamienny, Jean Tarbouriech, Alessandro Lazaric, and Ludovic Denoyer. Direct then diffuse: Incremental unsupervised skill discovery for state covering and goal reaching. arXiv preprint arXiv:2110.14457, 2021.  
[25] Georg Ostrovski, Marc G Bellemare, Aäron Oord, and Rémi Munos. Count-based exploration with neural density models. In International conference on machine learning, pages 2721-2730. PMLR, 2017.  
[26] Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, Xi Chen, Yan Duan, John Schulman, Filip De Turck, and Pieter Abbeel. # exploration: A study of count-based exploration for deep reinforcement learning. In 31st Conference on Neural Information Processing Systems (NIPS), volume 30, pages 1-18, 2017.  
[27] Justin Fu, John D Co-Reyes, and Sergey Levine. Ex2: Exploration with exemplar models for deep reinforcement learning. arXiv preprint arXiv:1703.01260, 2017.  
[28] Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. Advances in neural information processing systems, 29:1471-1479, 2016.  
[29] Jürgen Schmidhuber. A possibility for implementing curiosity and boredom in model-building neural controllers. In Proc. of the international conference on simulation of adaptive behavior: From animals to animals, pages 222-227, 1991.  
[30] Bradley C Stadie, Sergey Levine, and Pieter Abbeel. Incentivizing exploration in reinforcement learning with deep predictive models. arXiv preprint arXiv:1507.00814, 2015.  
[31] Joshua Achiam and Shankar Sastry. Surprise-based intrinsic motivation for deep reinforcement learning. arXiv preprint arXiv:1703.01732, 2017.  
[32] Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning, pages 2778-2787. PMLR, 2017.  
[33] Rein Houthooft, Xi Chen, Yan Duan, John Schulman, Filip De Turck, and Pieter Abbeel. Vime: Variational information maximizing exploration. arXiv preprint arXiv:1605.09674, 2016.  
[34] Navneet Madhu Kumar. Empowerment-driven exploration using mutual information estimation. arXiv preprint arXiv:1810.05533, 2018.  
[35] Shakir Mohamed and Danilo Jimenez Rezende. Variational information maximisation for intrinsically motivated reinforcement learning. arXiv preprint arXiv:1509.08731, 2015.
