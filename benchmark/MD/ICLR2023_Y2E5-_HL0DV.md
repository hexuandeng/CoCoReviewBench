# ONE CANNOT STAND FOR EVERYONE!  
LEVERAGING MULTIPLE USER SIMULATORS TO TRAIN TASK-ORIENTED DIALOGUE SYSTEMS

Anonymous authors

Paper under double-blind review

# ABSTRACT

User simulators are agents designed to imitate human users; recent advances have found that Task-oriented Dialogue (ToD) systems optimized toward a user simulator could better satisfy the need of human users. However, this might result in a sub-optimal ToD system if it is tailored to only one ad hoc user simulator, since human users can behave differently. In this paper, we propose a framework called MUST to optimize ToD systems via leveraging multiple user simulators.

The main challenges of MUST fall in 1) how to adaptively specify which user simulator to interact with the ToD system at each optimization step, since the ToD system might be over-fitted to some specific user simulators, and simultaneously under-fitted to some others; 2) how to avoid catastrophic forgetting of the adaption for a simulator that is not selected for several consecutive optimization steps. To tackle these challenges, we formulate MUST as a Multi-armed bandits (MAB) problem and provide a method called  $\mathrm{MUST}_{\mathrm{adaptive}}$  that balances  $i)$  the boosting adaption for adaptive interactions between different user simulators and the ToD system and  $ii)$  the uniform adaption to avoid the catastrophic forgetting issue. With both automatic evaluations and human evaluations, our extensive experimental results on the restaurant search task from MultiWOZ show that the dialogue system trained by our proposed MUST achieves a better performance than those trained by any single user simulator. It also has a better generalization ability when testing with unseen user simulators. Moreover, our method  $\mathrm{MUST}_{\mathrm{adaptive}}$  is indeed more efficient and effective to leverage multiple user simulators by our visualization analysis.

# 1 INTRODUCTION

Task-oriented dialogue systems aim to help users accomplish their various tasks (e.g., restaurant reservations) through natural language conversations. Training task-oriented dialogue systems in supervised learning (SL) approaches often requires a large amount of expert-labeled dialogues, however collecting these dialogues is usually expensive and time-consuming. Moreover, even with a large amount of dialogue data, some dialogue states may not be explored sufficiently for dialogue systems  $^{1}$  (Li et al., 2016b). To this end, many researchers try to build user simulators to mimic human users for generating reasonable and natural conversations. By using a user simulator and sampled user goals, we can train the dialogue system from scratch by reinforcement learning (RL) algorithms. Previous works tend to design better user simulator models (Schatzmann et al., 2007; Asri et al., 2016; Gur et al., 2018; Kreyssig et al., 2018; Lin et al., 2021). Especially, Shi et al. (2019) builds various user simulators and analyzes the behavior of each user simulator in the popular restaurant search task from MultiWOZ (Budzianowski et al., 2018).

In real application scenarios, the deployed dialogue system needs to face various types of human users. A single ad hoc user simulator can only represent one or a group of users, while other users might be under-represented. Instead of choosing the best-performed one from many dialogue systems trained by different single user simulators, we believe that it is worth trying to train a dialogue system by leveraging all user simulators simultaneously.

In this paper, we propose a framework called MUST to utilize Multiple User SimulaTors simultaneously to obtain a better system agent. There exist several simple ways to implement the MUST framework, including a merging strategy, a continual reinforcement learning strategy, and a uniform adaption strategy, denoted as  $\mathrm{MUST}_{\mathrm{merging}}$ ,  $\mathrm{MUST}_{\mathrm{CRL}}$ , and  $\mathrm{MUST}_{\mathrm{uniform}}$  respectively (See Sec. 3.2). However, none of them could effectively tackle the challenges of MUST: 1) how to efficiently leverage multiple user simulators when training the dialogue system since the system might be easily over-fitted to some specific user simulators and simultaneously under-fitted to some others, and 2) it should avoid a catastrophic forgetting issue. Therefore, we first formulate the problem as a Multi-armed bandits (MAB) problem (Auer et al., 2002); similar to the exploration vs exploitation trade-off, specifying multiple user simulators should trade off a boosting adaption (tackling challenge  $i$ ) and uniform adaption (tackling challenge  $ii$ ), see Sec. 4.1 for more details. Then we implement a method called  $\mathrm{MUST}_{\mathrm{adaptive}}$  which utilizes an adaptively-updated distribution among all user simulators to sample them in a real-time manner when training the dialogue system.

Extensive experimental results on the restaurant search task from MultiWOZ with both automatic evaluations and human evaluations show that the dialogue system trained by our proposed MUST achieves a better performance than those trained by any single user simulator. It also has a better generalization ability when testing with unseen user simulators and is more robust to the diversity of user simulators. Moreover, our method  $\mathrm{MUST}_{\mathrm{adaptive}}$  is indeed more efficient for leveraging multiple user simulators by our visualization analysis.

Our contributions are three-fold: (1) To the best of our knowledge, our proposed MUST is the first developed work to improve the dialogue system by using multiple user simulators simultaneously; (2) We design several ways to implement the MUST. Especially, we formulate MUST as a Multi-armed bandits (MAB) problem, based on which we provide a novel method  $\mathrm{MUST}_{\mathrm{adaptive}}$ ; and (3) The results show that dialogue systems trained with MUST consistently outperform those trained with a single user simulator through automatic and human evaluations. Especially, it largely improves the performance of the dialogue system tested on out-of-domain evaluation. Furthermore, The proposed method  $\mathrm{MUST}_{\mathrm{adaptive}}$  for MUST converges faster than  $\mathrm{MUST}_{\mathrm{uniform}}$ .

# 2 BACKGROUND

Dialogue System. Task-oriented dialogue systems aim to help users accomplish various tasks such as restaurant reservations through natural language conversations. Researchers usually divide the task-oriented dialogue systems into four modules (Wen et al., 2017; Ham et al., 2020; Peng et al., 2021): Natural Language Understanding (NLU) (Liu & Lane, 2016) that first comprehends user's intents and extracts the slots-values pairs, Dialog State Tracker (DST) (Williams et al., 2013) that tracks the values of slots, Dialog Policy Learning (POL) (Peng et al., 2017; 2018) that decides the dialog actions, and Natural Language Generation (NLG) (Wen et al., 2015; Peng et al., 2020) that translates the dialog actions into a natural-language form. The DST module and the POL module usually are collectively referred to as the dialogue manager (DM) (Chen et al., 2017). These different modules can be trained independently or jointly in an end-to-end manner (Wen et al., 2017; Liu & Lane, 2018; Ham et al., 2020; Peng et al., 2021; Hosseini-Asl et al., 2020).

User Simulator. The user simulator is an agent but also plays a user role. Different from dialogue systems, the user agent has a goal describing a target entity (e.g., a restaurant at a specific location) and should express its goal completely in an organized way by interacting with the system agent (Takanobu et al., 2020). Therefore, besides the modules of NLU, DM, and NLG like dialogue systems, the user agent should have another module called Goal Generator (Kreyssig et al., 2018), which is responsible for generating the user's goal. Building a user simulator usually could use an agenda-based approach (Schatzmann et al., 2007; Schatzmann & Young, 2009) designing handcrafted rules to mimic user behaviors or a model-based approach such as neural networks (Asri et al., 2016; Kreyssig et al., 2018; Gur et al., 2018) learned on a corpus of dialogues.

Training Dialogue Systems with a User Simulator. At the beginning of a dialogue, the user agent obtains its initial goal from the Goal Generator and then expresses its goal in natural languages. For the system agent, it does not know the user's goal and it should gradually understand the user's utterances, query the database to find entities, and provide useful information to see if accomplishing the user's task. Since only the system can access the database, the user does not know if its goal

![](images/095b5b561de80b8b4a6f4e35caf2f76e03de10ae1885d21aa79c451245758266.jpg)  
(a) Success rates of different systems.

![](images/189e29bbefce4ef6ddc6c723d779f7cc48eb6ede279a8b00920d51b7cf9333ff.jpg)  
Figure 1: (a) is the heat map on the success rates of system agents tested by different user simulators on 200 dialogues. (b) shows the dialog act distributions of Agenda-based User Simulators (ABUS) and Neural networks-based User Simulators (NUS) provided by Shi et al. (2019). There exist seven user dialog acts annotated in the restaurant search task from MultiWOZ, as shown on the Y-axis.  
(b) Dialog act distributions of different user simulators.

can be satisfied. Once the database result returned by the system agent is empty, the user agent should learn to compromise and change its goal with the help of Goal Generator (Kreyssig et al., 2018). When the dialogue ends, the user simulator will reward the system agent according to if the system agent accomplished the task. Then we could use the reward to update the system agent by reinforcement learning algorithms (Tseng et al., 2021).

# 3 MUST: LEVERAGE MULTIPLE USER SIMULATORS

# 3.1 MOTIVATIONS TO LEVERAGE MULTIPLE USER SIMULATORS

User simulators behave differently. Shi et al. (2019) implement six user simulators (AgenT, AgenR, AgenG, RNNT, RNNR, RNN $^2$ ) with both agenda-based methods and neural networks-based methods on the popular restaurant search task from MultiWOZ (Budzianowski et al., 2018). From their experiments, we observed that the dialogue systems trained by different user simulators vary in their performances (i.e., the success rates tested by the same user simulators). For example, when interacting with the user simulator of AgenT, the success rates of the system agents trained by Agenda-based user simulators (i.e., AgenT, AgenR, AgenG) are much higher than the system agents trained by RNN-based user simulators (i.e., RNNT, RNNR, RNN), see Fig. 1(a). The reason might be that these user simulators (i.e., with either handcrafted rules or data-driven learning in their DM modules) have different user dialog act distributions $^3$  (see Fig. 1(b)) which determines the dialogue state space explored by the dialogue system.

One cannot stand for everyone. Users might behave differently, one could design different user simulators with specific user dialog act distributions, see Shi et al. (2019). A single user simulator learned on a task-oriented dialogue corpus can just represent one or a group of users, while the dialogue system needs to accomplish tasks from various human users in real scenarios. We argue that it is beneficial to utilize all different user simulators to train the dialogue system. By leveraging multiple user simulators that have different user dialog act distributions, the dialogue systems can explore a larger dialogue state space, which might improve the ability of learned dialogue system.

# 3.2 SOME PRELIMINARY PROPOSALS FOR MUST

We propose a framework to leverage Multiple User SimulaTors (called 'MUST'), the core idea of which is to train a better dialogue system by using multiple user simulators. There are several simple ways to implement the MUST framework, including a merging strategy denoted as  $\mathrm{MUST}_{\mathrm{merging}}$ , a

Table 1: The comparison of different strategies for leveraging multiple user simulators.  

<table><tr><td></td><td>dynamic adaption</td><td>avoid catastrophic forgetting</td><td>efficiency</td></tr><tr><td>MUSTmerging</td><td>×</td><td>×</td><td>×</td></tr><tr><td>MUSTCRL</td><td>×</td><td>×</td><td>×</td></tr><tr><td>MUSTuniform</td><td>×</td><td>✓</td><td>×</td></tr><tr><td>MUSTadaptive</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

Continual Reinforcement Learning strategy denoted as  $\mathrm{MUST}_{\mathrm{CRL}}$ , and a uniform adaption strategy denoted as  $\mathrm{MUST}_{\mathrm{uniform}}$ .

(I)  $\mathbf{MUST}_{\mathrm{merging}}$  first sample some dialogues from each user simulator and the corresponding dialogue system trained by this simulator. Then it combines the collected dialogues to train a new user simulator for ensembling different user dialog act distributions. Finally, it uses this new user simulator to train the dialogue system by RL algorithms.  
$(\mathbf{II}) \mathbf{MUST}_{\mathrm{CRL}}$  treats each user simulator as an independent RL environment. It moves the trained system agent to another environment (i.e., let the system agent interact with another user simulator) if the system converges in the current RL environment.  
(III)  $\mathrm{MUST}_{\mathrm{uniform}}$  allows the system agent have chances to interact with all user simulators simultaneously. Different with  $\mathrm{MUST}_{\mathrm{CRL}}$ ,  $\mathrm{MUST}_{\mathrm{uniform}}$  puts all user simulators in a single RL environment and adopts the simplest way to specify different user simulators to train the dialogue system, which is to pick a user simulator among all user simulators with a uniform distribution for each iteration in the RL training.

Challenges to leverage multiple user simulators. The problem with  $\mathrm{MUST}_{\mathrm{merging}}$  is that it becomes difficult to adjust the weights of each user simulator adaptively in the training process. In  $\mathrm{MUST}_{\mathrm{merging}}$ , the proportions of dialogues from each user simulator are fixed; however, with the fixed proportions of user simulators, some user simulators might be not well-adapted and some might be overfitted. Because  $\mathrm{MUST}_{\mathrm{CRL}}$  has a problem of catastrophic forgetting (Khetarpal et al., 2020), it would be sensitive to the order of different user agents interacting with the dialogue system. Like Shi et al. (2019) shows, the system agents trained by different user simulators have different convergence speeds and converged performances. Namely, the system agent might be easily adapted to some user simulators but might be much harder to be adapted to others. A static distribution for user simulator selection (like in  $\mathrm{MUST}_{\mathrm{uniform}}$ ) under our proposed MUST will result in inefficient training since some unnecessary efforts will be costed for easily-adapted user simulators. Overall, the challenging problems under the MUST framework are i) how to efficiently leverage multiple user simulators when training the system agent, and ii) avoiding the catastrophic forgetting issue.

# 4 MUST AS A MULTI-ARMED BANDIT PROBLEM

Under the setting of  $\mathrm{MUST}_{\mathrm{uniform}}$ , we put multiple user simulators into the same RL environment and allow the dialogue system to learn from all of them. To make its training process more efficient, we first formulate the proposed MUST as a Multi-armed bandit (MAB) problem, see Sec. 4.1. In Sec. 4.2, we propose an adaptively-updated distribution used to replace the uniform distribution for accelerating the MUST training, which is denoted as  $\mathrm{MUST}_{\mathrm{adaptive}}$ . We briefly compare different implementations of MUST in Tab. 1.

# 4.1 FORMULATE MUST AS A MULTI-ARMED BANDIT PROBLEM

Adaptively specifying different user simulators recalls us a similar thought in machine learning, i.e., the boosting strategy (Zhou, 2012). From a boosting point of view, one should increase the weights of weakly-performed data examples and decrease the weights for well-performed ones. In MUST, we accordingly assume that it should reduce the interactions with user simulators that dialogue system has performed well and allocate more interactions with those user simulators that dialogue system has not performed well yet. We refer to this strategy as boosting adaption.

Meanwhile, we should also give some chances to all user simulators to relieve the catastrophic forgetting issue. We refer to this as uniform adaption. Such a trade-off between boosting adaption and uniform adaption is similar to the Multi-armed bandit (MAB) problem (Auer et al., 2002) that mainly studies the exploration vs exploitation trade-off in Reinforcement Learning policies.

Here, we interpret MUST as a MAB problem. We treat each user simulator as an arm. Suppose there are  $K$  arms (simulators), and each arm  $i$  has a fixed but unknown reward distribution  $R_{i}$  with an expectation  $\mu_{i}$ . At each time step  $t = 1,2,\dots,T$ , one must choose one of these  $K$  arms. We denote the arm pulled at time step  $t$  as  $i_t \in \{1,\dots,K\}$ . After pulling an arm, it will receive a reward  $x_{i_t}$  drawn from the arm's underlying reward distribution. The decision masker's objective is to maximize the cumulative expected reward over the time horizon

$$
\sum_ {t = 1} ^ {T} \mathbb {E} \left[ x _ {i _ {t}} \right] = \sum_ {t = 1} ^ {T} \mu_ {i _ {t}}. \tag {1}
$$

In MUST, the reward received in each arm-pulling step refers to the possible performance gain of the dialogue system after it interacts with a selected user simulator. A significant difference between the standard MAB problem and MUST is that the reward expectation of a user simulator (arm) in MUST is not static; it changes over time. For example, by consecutively interacting with the same user simulator, the performance gain (reward) of the system will decay since it might be in saturation or overfitting to this simulator. Moreover, the performance gain of the system after interacting with a simulator might increase if the simulator has not been selected for a period. To deal with this difference, we should tailor the solution of MAB to the MUST framework, e.g., considering boosting adaption and uniform adaption.

# 4.2 TRAINING THE DIALOGUE SYSTEM WITH MUST<sub>adaptive</sub>

To solve this MAB problem in MUST, we implement a method (called 'MUST_adaptive') with a two-phase procedure, as presented in Algorithm 1. Similar to the UCB1 $^5$  algorithm,  $\mathrm{MUST}_{\mathrm{adaptive}}$  specifies user simulators in a uniform distribution to interact with the system  $S$  in the first  $T_{\mathrm{warmup}}$  steps (i.e., in the warm-up phase). After that, the adaptive phase will balance the boosting adaption and uniform adaption by introducing an adaptively-updated distribution  $\pmb{p}$ , which is used to specify different user simulators to train the system agent  $S$  in later RL training. To accelerate the RL training, intuitively,  $\pmb{p}$  is expected to assign lower weights to user simulators that the system agent  $S$  already performs well and higher weights to those user simulators that  $S$  performs not well.

(1) Warm-up phase: in the first  $T_{\text{warmup}}$  dialogues, we use a uniform distribution to sample all user simulators to train the system agent  $S$  (lines 4-7). This phase is mainly used to warm up the dialogue system  $S$  and make it have little ability to converse with all user simulators.  
(2) Adaptive phase: in this phase, the distribution  $\pmb{p}$  used to sample all user simulators will be adaptively updated, which is why we call this phase adaptive phase. When this phase begins, that is  $t = 0$ , we will first evaluate the performance (i.e., the success rate  $\bar{x}_j, j \in \{1, \dots, K\}$ ) of the dialogue system  $S$  trained after the warm-up phase. The success rate  $\bar{x}_j$  is obtained by letting  $S$  interact  $d$  times with the simulator  $U_j$  (e.g.,  $j \in \{1, \dots, K\}$ ) and calculating the success rates.

Inspired by UCB1 Auer et al. (2002), we design a calibrated performance expectation  $\hat{x}_j$  of the system agent  $S$  interacting with each user simulator  $U_j$  taking exploration (the latter term) into consideration beyond purely exploitation (the former term):

$$
\hat {x} _ {j} = \bar {x} _ {j} + \sqrt {\frac {2 \ln t}{T _ {j , t}}}, j \in \{1, \dots , K \}; \tag {2}
$$

where  $\bar{x}_j$  is the success rate tested with user simulator  $U_j$ ,  $T_{j,t}$  is the number of times user simulator  $U_j$  has been interacted so far. Then we normalize  $\hat{x}_j$  into

$$
z _ {j} = 1 / \left(\hat {x} _ {j} - \tau \min  \left(\left\{x _ {1}, \dots , x _ {K} \right\}\right)\right), \tag {3}
$$

Algorithm 1: Implementing MUST-adaptive with the modified UCB1 algorithm  
Output: The learned dialogue system  $S$  
Input: K fixed User simulators  $\mathbf{U} = \{U_1,U_2,\dots U_K\}$  and the values of hyperparameters  $T_{\mathrm{warmup}},T,e,d,s;$    
1 Initialization: randomly-initializing System agent  $S$    
2 Initialization: initializing the simulator sampling distribution  $\pmb{p}$  as a uniform distribution.   
3 (1) Warm-up phase:   
4 for  $t = 0,\ldots ,T_{\mathrm{warmup}} - 1$  do sampling a simulator  $U_{i}$  in U w.r.t. the distribution  $\pmb{p}$  .. synthesizing a new dialogue using the system agent  $S$  and the sampled  $U_{i}$  .. using the reward obtained for the dialogue to update  $S$  with a RL algorithm;   
8 (2) Adaptive phase:   
9 for  $t = 0,\ldots ,T - 1$  do if  $t \% e == 0$  then for  $j = 1,\dots ,K$  do evaluating the performance i.e. the success rate  $\bar{x}_j$  of the agent  $S$  by letting it interact d times with the simulator  $U_{j}$  .. updating  $\pmb{p}$  based on these success rates  $\{\bar{x}_1,\dots,\bar{x}_K\}$  (see Eq. 2, Eq. 3, and Eq. 4); else sampling a simulator  $U_{i}$  in U w.r.t. the distribution  $\pmb{p}$  . synthesizing a new dialogue using the system agent  $S$  and the sampled  $U_{i}$  .. using the reward obtained for the dialogue to update  $S$  with a RL algorithm;

Eq. 3 penalizes the user simulators that the dialogue system already performs well in the expectation term. Where the hyperparameter  $\tau$  is the smooth factor for distribution  $\pmb{p} = \{\pmb{p}_1, \dots, \pmb{p}_K\} - \text{the larger } s$  is, the sharper  $\pmb{p}$  is. Each probability  $\pmb{p}_j$  in  $\pmb{p}$  is calculated as

$$
\boldsymbol {p} _ {i} = \frac {z _ {j}}{\sum_ {j = 1} ^ {K} z _ {j}}. \tag {4}
$$

In the following  $T - 1$  dialogues, we will specify all user simulators to train the system agent  $S$  with this distribution  $p$  (lines 15-18). And we will also evaluate the RL model  $S$  for every  $e$  episodes (line 10-12) and update the distribution  $p$  with the new  $K$  success rates (line 13).

Difference with the original UCB1. The main differences between our modified UCB1 algorithm and the original UCB1 algorithm are twofold. First, we tailor the original UCB1 in our scenario using Eq. 3, to penalize the user simulators that the dialogue system has performed well. Secondly, we adopt a sampling schema based on a well-designed distribution (see Eq. 4), instead of taking the arm with the highest expectation. This is to increase the diversity and flexibility of arm selection.

# 5 EXPERIMENTS

To verify the effectiveness of MUST, we benchmark the system agents trained either with a single user simulator or multiple user simulators (including  $\mathrm{MUST}_{\mathrm{merging}}$ ,  $\mathrm{MUST}_{\mathrm{uniform}}$ ,  $\mathrm{MUST}_{\mathrm{adaptive}}$ ).

# 5.1 EXPERIMENTAL SETUP

Available user simulators. There are six user simulators provided by Shi et al. (2019), which are Agenda-Template (AgenT), Agenda-Retrieval (AgenR), Agenda-Generation (AgenG), RNN-Template (RNNT), RNN-Retrieval (RNNR), RNN-End2End (RNN) trained with different dialog planning and generation methods. The NLU modules of all six user simulators are using the RNN model. The DM modules of AgenT, AgenR, and AgenG are rule-based methods. For the NLG module, these three simulators are using the template, retrieval, and generation methods respectively. The DM modules of RNNT, RNNR are using Sequicity (Lei et al., 2018) as their backbones which is an RNN-based seq2seq model with copy mechanism. The NLG modules of these two simulators

are using the template and retrieval methods respectively. The user simulator of RNN uses Sequicity as its backbone in an end-to-end manner.

Baselines. The baselines are the dialogue systems trained by each user simulator, including Sys-AgenT, Sys-AgenR, Sys-AgenG, Sys-RNNT, Sys-RNNR, and Sys-RNN. For a fair comparison, all system agents (including the systems trained by our MUST) have the same architecture described in Shi et al. (2019). See basic modules of user simulators and dialogue systems in App. B.1.

MultiWOZ Restaurant Domain Dataset. The original task in MultiWOZ (Budzianowski et al., 2018) is to model the system response. Shi et al. (2019) annotate the user intents and the user-side dialog acts in the restaurant domain of MultiWOZ to build user simulators, which has a total of 1,310 dialogues. Moreover, we randomly simulated 2,000 dialogues from each rule-based simulator AgenT, AgenR, AgenG, and their corresponding system agents respectively, and processed these dialogues to have the same annotation format as the MultiWOZ restaurant domain dataset. We denoted this dataset as Simulated Agenda Dataset.

Evaluation Measures. The direct automatic metric to evaluate the dialogue system is the success rate tested by each user simulator. We calculate the success rate between a user simulator and a system agent by sampling 200 dialogues. We exclude some user simulators in MUST training and test the system by them as out-of-domain evaluation. According to the previous study Gunasekara et al. (2020), there usually has a gap between automatic evaluations and human evaluations of dialogue systems. Therefore, we ask human users to converse with dialogue systems. Each dialogue system has conversed with 5 different users; each user has 10 dialogues. In total, we collect 50 dialogues for each dialogue system to calculate its success rate. See more details in App. B.2.

# 5.2 IMPLEMENTATIONS

# 5.2.1 TWO NEW USER SIMULATORS

We believe Pre-trained Language Models (PLMs) might improve the capacity of user simulators since they have recently shown remarkable success in building task-oriented dialogue systems (Ham et al., 2020; Peng et al., 2021; Hosseini-Asl et al., 2020). Here we implement another two user simulators using GPT (Radford et al., 2018; 2019). Building a user simulator using GPT is similar to building a ToD system with GPT. See more details in App. C.

GPT Simulator. It is first fine-tuned on the simulated agenda dataset and then fine-tuned on the MultiWOZ restaurant domain dataset by leveraging GPT. This user simulator will be used to help implementing MUST.

$\mathrm{GPT}_{\mathrm{IL}}$  Simulator. Similar to Imitation Learning (IL), we first train a new user simulator with dialogue sessions collected from different user simulators and their corresponding dialogue systems. Like the GPT simulator, we learn this new user simulator based on GPT model and denote it as  $\mathrm{GPT}_{\mathrm{IL}}$ .  $\mathrm{GPT}_{\mathrm{IL}}$  is first fine-tuned on the simulated agenda dataset which has a total of 6,000 dialogues. Then we sample 1,400 dialogues from the simulated agenda dataset and merge them with 1,310 MultiWOZ restaurant domain dialogues to continue fine-tuning  $\mathrm{GPT}_{\mathrm{IL}}$ .

# 5.2.2 DIALOGUE SYSTEMS

Sys-GPT is trained with the single user simulator GPT. Sys-MUST $_{\text{merging}}$  is trained by  $\mathbf{GPT}_{\text{IL}}$  for implementing  $\mathbf{MUST}_{\text{merging}}$  strategy. Sys-MUST $_{\text{uniform}}$  is trained by user simulators AgenT, AgenR, RNNT, and GPT $^6$  with uniform sampling distribution. In Sys-MUST $_{\text{adaptive}}$ , the distribution  $p$  is adaptively updated using our modified UCB1 algorithm.

Table 2: The success rates of the system agents were tested against various user simulators. Each column represents a user simulator, each row represents a dialogue system trained with a specific simulator, e.g., Sys-AgenT means the system trained with AgenT. Each entry shows the success rate on 200 dialogues collected from a user simulator and a system agent. We use four user simulators: AgenT, AgenR, RNNT, and GPT simulator to implement  $\mathrm{MUST}_{\mathrm{uniform}}$  and  $\mathrm{MUST}_{\mathrm{adaptive}}$ .  

<table><tr><td rowspan="2" colspan="2">Dialogue Systems</td><td colspan="4">In-domain evaluation</td><td colspan="5">Out-of-domain evaluation</td><td colspan="2">All</td></tr><tr><td>AgenT</td><td>AgenR</td><td>RNNT</td><td>GPT</td><td>AgenG</td><td>RNNR</td><td>RNN</td><td>Avg.↑</td><td>Std.↓</td><td>Avg.↑</td><td>Std.↓</td></tr><tr><td rowspan="4">single</td><td>Sys-AgenT</td><td>97.5</td><td>54.0↓36.9%</td><td>98.5↓0.5%</td><td>78.0↓4.1%</td><td>72.5</td><td>92.5</td><td>77.0</td><td>80.7</td><td>8.6</td><td>81.4</td><td>14.8</td></tr><tr><td>Sys-AgenR</td><td>96.0↓1.5%</td><td>90.0</td><td>98.5↓0.5%</td><td>80.5↓1.5%</td><td>97.5</td><td>97.5</td><td>82.0</td><td>92.3</td><td>7.3</td><td>91.7</td><td>7.1</td></tr><tr><td>Sys-RNNT</td><td>30.5↓68.7%</td><td>23.0↓68.7%</td><td>99.0</td><td>75.5↓6.7%</td><td>35.5</td><td>97.5</td><td>84.0</td><td>72.3</td><td>26.6</td><td>63.6</td><td>30.5</td></tr><tr><td>Sys-GPT</td><td>60.5↓37.9%</td><td>51.5↓39.5%</td><td>97.0↓2.0%</td><td>82.0</td><td>59.5</td><td>94.0</td><td>92.0</td><td>81.8</td><td>15.8</td><td>76.6</td><td>17.6</td></tr><tr><td rowspan="3">MUST</td><td>Sys-MUSTmerging</td><td>97.5↑0.0%</td><td>83.5↓6.7%</td><td>94.5↓4.6%</td><td>80.5↓1.5%</td><td>97.5</td><td>94.0</td><td>82.5</td><td>91.3</td><td>6.4</td><td>90.0</td><td>6.9</td></tr><tr><td>Sys-MUSTuniform</td><td>97.5↑0.0%</td><td>89.0↓1.0%</td><td>97.5↓1.5%</td><td>82.5↑0.5%</td><td>96.5</td><td>96.0</td><td>87.5</td><td>93.4</td><td>4.2</td><td>92.4</td><td>5.6</td></tr><tr><td>Sys-MUSTadaptive</td><td>97.5↑0.0%</td><td>89.5↓0.5%</td><td>97.0↓2.0%</td><td>82.5↑0.5%</td><td>96.5</td><td>97.5</td><td>90.0</td><td>94.7</td><td>3.3</td><td>92.9</td><td>5.3</td></tr></table>

[1] The underlined number represents the success rate between a user simulator and its corresponding dialogue system trained by this user simulator. The increasing and decreasing percentages (in red and green colors) use the underlined numbers as the base success rates.

# 5.3 EXPERIMENTAL RESULTS

Automatic Evaluation. As seen in Tab. 2, Sys-MUST $_{\text{uniform}}$  and Sys-MUST $_{\text{adaptive}}$  outperform the dialogue systems (Sys-AgenT, Sys-AgenR, Sys-RNNT, and Sys-GPT) trained by a single user simulator in the overall performance, demonstrating the superior of leveraging multiple user simulators. Especially, Sys-MUST $_{\text{adaptive}}$  has a 1.2 absolute value improvement (92.9 vs. 91.7) over the previous SOTA system Sys-AgenR. Observing that Sys-MUST $_{\text{merging}}$  is not as competitive as Sys-MUST $_{\text{uniform}}$  and Sys-MUST $_{\text{adaptive}}$ , this might be because the merging strategy cannot effectively leverage multiple user simulators.

Table 3: Human evaluation results on dialogue systems.  

<table><tr><td colspan="2">Dialogue Systems</td><td>human evaluation</td></tr><tr><td rowspan="4">single</td><td>Sys-AgenT</td><td>76.0</td></tr><tr><td>Sys-AgenR</td><td>84.0</td></tr><tr><td>Sys-RNNT</td><td>34.0</td></tr><tr><td>Sys-GPT</td><td>58.0</td></tr><tr><td rowspan="3">MUST</td><td>Sys-MUSTmerging</td><td>90.0</td></tr><tr><td>Sys-MUSTuniform</td><td>92.0</td></tr><tr><td>Sys-MUSTadaptive</td><td>92.0</td></tr></table>

[2]  $\downarrow$  (↑) indicates by what percentages the success rate has decreased (increased) compared with the base success rate by interacting with the same user simulator.

In in-domain evaluation, the performances of systems (Sys-AgenT, Sys-AgenR, Sys-RNNT, and Sys-GPT) trained by a single user simulator drop a lot when testing with a different user simulator. It requires us to delicately select a suitable user simulator for obtaining a good dialogue system. However, human users might be multi-facet or even unknown, which makes the selection difficult. Therefore, it is essential to leverage multiple user simulators when training dialogue systems. At least, the performance gap of dialogue systems trained by MUST becomes smaller than that without MUST, see the percentages labeled in green and red colors.

In out-of-domain evaluation where the user simulators used for testing are unseen by our MUST, Sys-MUST $_{\text{uniform}}$  and Sys-MUST $_{\text{adaptive}}$  achieve 2.4 absolute value improvements. This evidences that MUST has a better generalization ability for interacting with unseen user simulators. Moreover, MUST approaches (Sys-MUST $_{\text{merging}}$ , Sys-MUST $_{\text{uniform}}$ , and Sys-MUST $_{\text{adaptive}}$ ) have a lower standard deviation, which indicates that they are more robust to the diversity of user simulators.

Human Evaluation. The human evaluation results in Tab. 3 show that our Sys-MUST<sub>uniform</sub> and Sys-MUST<sub>adaptive</sub> largely outperforms the other dialogue systems when interacting with real users. The consistency between automatic evaluations and human evaluations evidences the effectiveness of our proposed MUST.

# 5.4 ANALYSIS AND DISCUSSIONS

Convergences between  $\mathbf{MUST}_{\mathrm{uniform}}$  and  $\mathbf{MUST}_{\mathrm{adaptive}}$ . In Fig. 2, we show the learning curves of Sys-MUST $_{\mathrm{uniform}}$  and Sys-MUST $_{\mathrm{adaptive}}$  in 100,000 steps; the first 40,000 steps are in the warm-up phase for Sys-MUST $_{\mathrm{adaptive}}$ . From Fig. 2(a), we see that training the dialogue system with AgenT, AgenR, RNNT, and GPT by  $\mathbf{MUST}_{\mathrm{adaptive}}$  converges faster than  $\mathbf{MUST}_{\mathrm{uniform}}$ . We further plot the performances of the dialogue system tested by each user simulator in the RL training, which are shown in Fig. 2(b)-2(e).

![](images/9e77111de9d3b7cedea1dac5e3fa17e735c64e970c5a4ac4111cf25fb7f8928c.jpg)  
(a) The performance(b)

![](images/4335012107ca0ae21c3f6933a7a47da12523601ba9bd7435930204df3540f99e.jpg)  
Tested by AgenR(c)

![](images/7ffc84c466764eeb5ad0b53f3adb48270638f0f272a4f2cc958527310bfee23a.jpg)  
Tested by AgenT

![](images/f7cc4f434f48a2d30a83576c194824adabaa737b483ac5c8eca4fe7da9ce8bab.jpg)  
(d) Tested by GPT

![](images/79de204dfad349c5f9cc8b8ab3cd5ad8af92a23c8b63a28c70b1079e28b1e591.jpg)  
(e) Tested by RNNT

![](images/3d58bf8727ef9d1e1dbbec4cbda1bcc5a92110380792ceae74c07f17a36c04f9.jpg)  
Figure 2: The learning curves of Sys-MUST $_{\text{uniform}}$  and Sys-MUST $_{\text{adaptive}}$ . (a) shows their average success rates tested by all user simulators (AgenT, AgenR, RNNT, and GPT). Success rates tested by each user simulator are in (b)-(e).  
(a) The sampling proportion of simulators.

![](images/b2748d107745cecf08b5271d44ff130bcd489fd1c597c064012b5bf10c493dd0.jpg)  
(b) Variations of the sampling proportion (in every 2000 steps) of simulators.  
Figure 3: The sampling proportion of user simulators in average (a) and in time horizon (b).

Visualization of the patterns learned by  $\mathbf{MUST}_{\mathrm{adaptive}}$ . Let us define the adaptation difficulty of a user simulator as the steps that the dialogue system should take to converge when trained by this user simulator. The adaptation difficulty of all user simulators could be ranked like AgenR > AgenT > GPT > RNNT according to Fig. 2(b)- 2(e). To check whether  $\mathbf{MUST}_{\mathrm{adaptive}}$  tend to sample harder-to-adapt user simulators in the adaptive phase, as assumed in Sec. 4.2, Fig. 3(a) visualizes the sampling proportions of all user simulators. We could observe that AgenR was sampled with  $45.1\%$  (the biggest proportion) and it is indeed the hardest user simulator to be adapted; GPT has the smallest sampling proportion and it is the easiest to be adapted. The consistency between the adaptation difficulty and sampling proportions for these four user simulators evidences our assumption in Sec. 4.2. Interestingly, Fig. 3(b) visualizes the variations of the sampling distribution of user simulators, which shows that AgenR and AgenT are competitive with GPT simulator; while RNNT and GPT are cooperative with each other. This might be because RNNT and GPT simulator are learned from the dialogue corpus and share some similar behaviors.

# 6 CONCLUSION

In this paper, we propose a framework named MUST to improve the system agent by using multiple user simulators simultaneously. We discuss several simple methods to implement MUST, which are either inflexible or inefficient. Therefore, we formulate MUST as a Multi-armed bandits (MAB) problem, based on which we propose a novel implementation for MUST. The experimental results on the restaurant search task from MultiWOZ demonstrate that our proposed MUST can largely improve the system agent upon the baseline methods, especially when the tested user simulators are unseen. Moreover,  $\mathrm{MUST}_{\mathrm{adaptive}}$  is robust to the diversity of user simulators and its training is more efficient. The main limitation of this work is mainly that we only conduct our experiments on the restaurant domain of the MultiWOZ since we can only find multiple user simulators from Shi et al. (2019) and they build these simulators only on the restaurant search task. In future work, we plan to apply our proposed methods to multi-domain scenarios.

# REFERENCES

Layla El Asri, Jing He, and Kaheer Suleman. A sequence-to-sequence model for user simulation in spoken dialogue systems, 2016.  
Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2-3):235-256, 2002. URL http://homes.dsi.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf.  
Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Iñigo Casanueva, Stefan Ultes, Osman Ramadan, and Milica Gašić. MultiWOZ - a large-scale multi-domain Wizard-of-Oz dataset for task-oriented dialogue modelling. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 5016-5026, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1547. URL https://aclanthology.org/D18-1547.  
Yun-Nung Chen, Asli Celikyilmaz, and Dilek Hakkani-Tür. Deep learning for dialogue systems. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics: Tutorial Abstracts, pp. 8-14, Vancouver, Canada, July 2017. Association for Computational Linguistics. URL https://aclanthology.org/P17-5004.  
R. Chulaka Gunasekara, Seokhwan Kim, Luis Fernando D'Haro, Abhinav Rastogi, Yun-Nung Chen, Mihail Eric, Behnam Hedayatnia, Karthik Gopalakrishnan, Yang Liu, Chao-Wei Huang, Dilek Hakkani-Tur, Jinchao Li, Qi Zhu, Lingxiao Luo, Lars Liden, Kaili Huang, Shahin Shayandeh, Runze Liang, Baolin Peng, Zheng Zhang, Swadheen Shukla, Minlie Huang, Jianfeng Gao, Shikib Mehri, Yulan Feng, Carla Gordon, Seyed Hossein Alavi, David R. Traum, Maxine Eskenazi, Ahmad Beirami, Eunjoon Cho, Paul A. Crook, Ankita De, Alborz Geramifard, Satwik Kottur, Seungwhan Moon, Shivani Poddar, and Rajen Subba. Overview of the ninth dialog system technology challenge: DSTC9. CoRR, abs/2011.06486, 2020. URL https://arxiv.org/abs/2011.06486.  
Izzeddin Gur, Dilek Hakkani-Tur, Gokhan Tur, and Pararth Shah. User modeling for task oriented dialogues, 2018.  
Donghoon Ham, Jeong-Gwan Lee, Youngsoo Jang, and Kee-Eung Kim. End-to-end neural pipeline for goal-oriented dialogue systems using GPT-2. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 583-592, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.54. URL https://aclanthology.org/2020.acl-main.54.  
Ehsan Hosseini-Asl, Bryan McCann, Chien-Sheng Wu, Semih Yavuz, and Richard Socher. A simple language model for task-oriented dialogue. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 20179-20191. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/e946209592563be0f01c844ab2170f0c-Paper.pdf.  
Khimya Khetarpal, Matthew Riemer, Irina Rish, and Doina Precup. Towards continual reinforcement learning: A review and perspectives. CoRR, abs/2012.13490, 2020. URL https://arxiv.org/abs/2012.13490.  
Florian Kreyssig, Inigo Casanueva, Paweł Budzianowski, and Milica Gašić. Neural user simulation for corpus-based policy optimisation of spoken dialogue systems. In Proceedings of the 19th Annual SIGdial Meeting on Discourse and Dialogue, pp. 60-69, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/W18-5007. URL https://aclanthology.org/W18-5007.  
Wenqiang Lei, Xisen Jin, Min-Yen Kan, Zhaochun Ren, Xiangnan He, and Dawei Yin. Sequicity: Simplifying task-oriented dialogue systems with single sequence-to-sequence architectures. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1437-1447, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1133. URL https://aclanthology.org/P18-1133.

Jiwei Li, Will Monroe, Alan Ritter, Dan Jurafsky, Michel Galley, and Jianfeng Gao. Deep reinforcement learning for dialogue generation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1192-1202, Austin, Texas, November 2016a. Association for Computational Linguistics. doi: 10.18653/v1/D16-1127. URL https://aclanthology.org/D16-1127.  
Xiujun Li, Zachary C Lipton, Bhuwan Dhingra, Lihong Li, Jianfeng Gao, and Yun-Nung Chen. A user simulator for task-completion dialogues. arXiv preprint arXiv:1612.05688, 2016b.  
Hsien-chin Lin, Nurul Lubis, Songbo Hu, Carel van Niekerk, Christian Geishauser, Michael Heck, Shutong Feng, and Milica Gasic. Domain-independent user simulation with transformers for task-oriented dialogue systems. In Proceedings of the 22nd Annual Meeting of the Special Interest Group on Discourse and Dialogue, pp. 445-456, Singapore and Online, July 2021. Association for Computational Linguistics. URL https://aclanthology.org/2021.sigdial-1.47.  
Bing Liu and Ian Lane. End-to-end learning of task-oriented dialogs. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Student Research Workshop, pp. 67-73, New Orleans, Louisiana, USA, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-4010. URL https://aclanthology.org/N18-4010.  
Bing Liu and Ian R. Lane. Attention-based recurrent neural network models for joint intent detection and slot filling. CoRR, abs/1609.01454, 2016. URL http://arxiv.org/abs/1609.01454.  
Baolin Peng, Xiujun Li, Lihong Li, Jianfeng Gao, Asli Celikyilmaz, Sungjin Lee, and Kam-Fai Wong. Composite task-completion dialogue policy learning via hierarchical deep reinforcement learning. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 2231-2240, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1237. URL https://aclanthology.org/D17-1237.  
Baolin Peng, Xiujun Li, Jianfeng Gao, Jingjing Liu, and Kam-Fai Wong. Deep Dyna-Q: Integrating planning for task-completion dialogue policy learning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2182-2192, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1203. URL https://aclanthology.org/P18-1203.  
Baolin Peng, Chenguang Zhu, Chunyuan Li, Xiujun Li, Jinchao Li, Michael Zeng, and Jianfeng Gao. Few-shot natural language generation for task-oriented dialog. CoRR, abs/2002.12328, 2020. URL https://arxiv.org/abs/2002.12328.  
Baolin Peng, Chunyuan Li, Jinchao Li, Shahin Shayandeh, Lars Liden, and Jianfeng Gao. Soloist: Building task bots at scale with transfer learning and machine teaching, 2021.  
Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.  
Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.  
Victor Sanh, Lysandre Debut, Julien Chaumont, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter, 2020.  
Jost Schatzmann and Steve Young. The hidden agenda user simulation model. IEEE Transactions on Audio, Speech, and Language Processing, 17(4):733-747, 2009. doi: 10.1109/TASL.2008.2012071.  
Jost Schatzmann, Blaise Thomson, Karl Weilhammer, Hui Ye, and Steve Young. Agenda-based user simulation for bootstrapping a POMDP dialogue system. In Human Language Technologies 2007: The Conference of the North American Chapter of the Association for Computational Linguistics; Companion Volume, Short Papers, pp. 149-152, Rochester, New York, April 2007. Association for Computational Linguistics. URL https://aclanthology.org/N07-2038.

Weiyan Shi, Kun Qian, Xuewei Wang, and Zhou Yu. How to build user simulators to train RL-based dialog systems. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 1990-2000, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1206. URL https://aclanthology.org/D19-1206.  
Ryuichi Takanobu, Runze Liang, and Minlie Huang. Multi-agent task-oriented dialog policy learning with role-aware reward decomposition. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 625-638, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.59. URL https://aclanthology.org/2020.acl-main.59.  
Bo-Hsiang Tseng, Yinpei Dai, Florian Kreyssig, and Bill Byrne. Transferable dialogue systems and user simulators. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 152-166, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.13. URL https://aclanthology.org/2021.acl-long.13.  
Tsung-Hsien Wen, Milica Gašić, Nikola Mrkšić, Pei-Hao Su, David Vandyke, and Steve Young. Semantically conditioned LSTM-based natural language generation for spoken dialogue systems. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 1711-1721, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1199. URL https://aclanthology.org/D15-1199.  
Tsung-Hsien Wen, David Vandyke, Nikola Mrkšić, Milica Gašić, Lina M. Rojas-Barahona, PeiHao Su, Stefan Ultes, and Steve Young. A network-based end-to-end trainable task-oriented dialogue system. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pp. 438-449, Valencia, Spain, April 2017. Association for Computational Linguistics. URL https://aclanthology.org/E17-1042.  
Jason Williams, Antoine Raux, Deepak Ramachandran, and Alan Black. The dialog state tracking challenge. In Proceedings of the SIGDIAL 2013 Conference, pp. 404-413, Metz, France, August 2013. Association for Computational Linguistics. URL https://aclanthology.org/W13-4065.  
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumont, Clement Delangue, Anthony Moi, Pierrick Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38-45, Online, October 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-demos.6. URL https://aclanthology.org/2020.emnlp-demos.6.  
Yizhe Zhang, Michel Galley, Jianfeng Gao, Zhe Gan, Xiujun Li, Chris Brockett, and Bill Dolan. Generating informative and diverse conversational responses via adversarial information maximization. In NeurIPS, 2018.  
Zhi-Hua Zhou. Ensemble methods: foundations and algorithms. CRC press, 2012.
