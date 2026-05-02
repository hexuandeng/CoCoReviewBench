# EVOLVING REINFORCEMENT LEARNING ALGORITHMS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose a method for meta-learning reinforcement learning algorithms by searching over the space of computational graphs which compute the loss function for a value-based model-free RL agent to optimize. The learned algorithms are domain-agnostic and can generalize to new environments not seen during training. Our method can both learn from scratch and bootstrap off known existing algorithms, enabling interpretable modifications which improve performance. We highlight two learned algorithms which obtain good generalization performance over a set of classical control tasks and gridworld type tasks. Our analysis of the learned algorithm behavior shows resemblance to recently proposed RL algorithms that have been designed manually.

# 1 INTRODUCTION

Designing new deep reinforcement learning algorithms that can efficiently solve across a wide variety of problems generally requires a tremendous amount of manual effort. Learning to design reinforcement learning algorithms or even small sub-components of algorithms would help ease this burden and could result in better algorithms than researchers could design manually. Our work might then shift from designing these algorithms manually into designing the language and optimization methods for developing these algorithms automatically.

Reinforcement learning algorithms can be viewed as a procedure that maps an agent's experience to a policy that obtains high cumulative reward over the course of training. We formulate the problem of training an agent as one of meta-learning: an outer loop searches over the space of computational graphs or programs that compute the objective function for the agent to minimize and an inner loop performs the updates using the learned loss function. The objective of the outer loop is to maximize the training return of the inner loop algorithm.

Our learned loss function should generalize across many different environments, instead of being specific to a particular domain. Thus, we design a search language that can express general symbolic loss functions which can be applied to any environment. Data typing and a generic interface to variables in the MDP allow the learned program to be domain agnostic. This language also supports the use of neural network modules as subcomponents of the program, so that more complex neural network architectures can be realized. Efficiently searching over the space of useful programs is generally difficult. For the outer loop optimization, we use a population based method, regularized evolution from Real et al. (2019), which can scale with the number of compute nodes and has been shown to work for designing algorithms for supervised learning (Real et al., 2020). We adapt this method to design algorithms for reinforcement learning.

While learning from scratch is generally less biased, encoding existing human knowledge into the learning process can speed up the optimization and also make the learned algorithm more interpretable. Because our search language expresses algorithms as a generalized computation graph, we can embed known RL algorithms in the graphs of the starting population of programs. We compare starting from scratch with bootstrapping off existing algorithms and find that while starting from scratch can learn existing algorithms, starting from existing knowledge leads to new RL algorithms which can outperform the initial programs.

The contribution of this paper is a method for searching over the space of RL algorithms, which we instantiate by developing a formal language that describes a broad class of value-based model-free

![](images/decb02b8a4fb2b5f5cfbaa88ca448222df4bd41f853ca0ae4478ca5822cb7e3c.jpg)  
Figure 1: Method overview. We use regularized evolution to evolve a population of RL algorithms. A mutator alters top performing algorithms to produce a new algorithm. The performance of the algorithm is evaluated over a set of training environments and the population is updated. Our method can incorporating existing knowledge by starting the population from known RL algorithms instead of purely from scratch.

reinforcement learning methods. The learned algorithms are domain agnostic and can generalize to new environments. Our search language enables us to embed existing algorithms into the starting graphs which leads to faster learning and interpretable algorithms. We learn two new RL algorithms which outperform existing algorithms in both sample efficiency and final performance on both the training and test environments. The environments consist of a suite of discrete action classical control and gridworld style environments. Our analysis of the meta-learned programs shows that our method automatically discovers algorithms that are structured in a way that resembles recently proposed innovations in RL, and empirically attain better performance than deep Q-learning methods.

# 2 RELATED WORK

Learning to learn is an established idea in in supervised learning, including meta-learning with genetic programming (Schmidhuber, 1987; Holland, 1975; Koza, 1993), learning a neural network update rule (Bengio et al., 1991), and self modifying RNNs (Schmidhuber, 1993). More recently, AutoML (Hutter et al., 2018) aims to automate the machine learning training process. Automated neural network architecture search (Real et al., 2017; 2019; Liu et al., 2017; Stanley & Miikkulainen, 2002; Zoph & Le, 2016; Elsken et al., 2018; Pham et al., 2018) has made large improvements in image classification. Instead of learning the architecture, AutoML-Zero (Real et al., 2020) learns the algorithm from scratch using basic mathematical operations. Our work shares similar ideas, but is applied to the RL setting and assumes additional primitives such as neural network modules. We learn computational graphs with the goal of automating RL algorithm design. In contrast to AutoML-Zero, our learned RL algorithms generalize to new problems, not encountered in training.

Automating RL. While RL is used for AutoML (Zoph & Le, 2016; Zoph et al., 2018; Cai et al., 2018; Bello et al., 2017), automating RL itself has been somewhat limited. RL requires different design choices compared to supervised learning, including the formulation of reward and policy update rules. All of which affect learning and performance, and are usually chosen through trial and error. AutoRL addresses the gap by applying the AutoML framework from supervised learning to the MDP setting in RL. For example, evolutionary algorithms are used to mutate the actor network weights (Khadka & Tumer, 2018), learn the rewards for a task (Faust et al., 2019), tune hyperparameters (Tang & Choromanski, 2020; Franke et al., 2020), or search for a neural network architecture (Song et al., 2020; Franke et al., 2020). This paper focuses on task-agnostic RL update rules in the value-based RL setting which are both interpretable and generalizable.

Meta-learning in RL. Recent work has focused on few-shot task adaptation. Finn et al. (2017); Finn & Levine (2018) meta-learns initial parameters which can quickly adapt to new tasks, while  $\mathbf{R}\mathbf{L}^2$  (Duan et al., 2016) and concurrent work (Wang et al., 2017), formulates RL itself as a learning problem that is learned with an RNN. The meta-learned component of these works is tuned to a particular domain or environment, in the form of NN weights which cannot be used for completely new domains with potentially different sized inputs. Neural Programmer-Interpreters (Reed & De Freitas, 2015; Pierrot et al., 2019) overcome the environment generalization challenge by learning hierarchical neural programs with domain-specific encoders for different environments. Here, the computational graph has a flexible architecture and generalizes across different environments.

Learning RL algorithms or their components, such as a reward bonus or value update function, has been studied previously with meta-gradients (Kirsch et al., 2020; Chebotar et al., 2019; Oh et al., 2020), evolutionary strategies (Houthooft et al., 2018), and RNNs (Duan et al., 2016). Although

our work also learns RL algorithms, the update rule is represented as a computation graph which includes both neural network modules and symbolic operators. One key benefit is that the resulting graph can be interpreted analytically and can optionally be initialized from known existing algorithms. Prior work that focuses on learning RL losses, generalizes to different goals and initial conditions within a single environment (Houthooft et al., 2018), or learns a domain invariant policy update rule that can generalize to new environments (Kirsch et al., 2020). Another approach searches over the space of curiosity programs using a similar language of DAGs with neural network modules (Alet et al., 2020) and performs the meta-training on a single environment. In contrast, our method is applied to learn general RL update rules and meta-trained over a diverse set of environments.

# 3 LEARNING REINFORCEMENT LEARNING ALGORITHMS

In this section, we first describe the problem setup. An inner loop method  $\operatorname{Eval}(L, \mathcal{E})$  evaluates a learned RL algorithm  $L$  on a given environment  $\mathcal{E}$ . Given access to this procedure, the goal for the outer loop optimization is to learn a RL algorithm with high training return over a set of training environments. We then describe the search language which enables the learning of general loss functions and the outer loop method which can efficiently search over this space.

# 3.1 PROBLEM SETUP

We assume that the agent parameterized with policy  $\pi_{\theta}(a_t|s_t)$  outputs actions  $a_{t}$  at each time step to an environment  $\mathcal{E}$  and receives reward  $r_t$  and next state  $s_{t + 1}$ . Since we are focusing on discrete action value-based RL methods,  $\theta$  will be the parameters for a Q-value function and the policy is obtained from the Q-value function using an  $\epsilon$ -greedy strategy. The agent saves this stream of transitions  $(s_t,s_{t + 1},a_t,r_t)$  to a replay buffer and continually updates the policy by minimizing a loss function  $L(s_{t},a_{t},r_{t},s_{t + 1},\theta ,\gamma)$  over these transitions with gradient descent. Training will occur for a fixed number of  $M$  training episodes where in each episode  $m$ , the agent earns episode return  $R_{m} = \sum_{t = 0}^{T}r_{t}$ . The performance of an algorithm for a given environment is summarized by the normalized average training return,  $\frac{1}{M}\sum_{m = 1}^{M}\frac{R_i - R_{min}}{R_{max} - R_{min}}$ , where  $R_{min}$  and  $R_{max}$  are the minimum and maximum re

turn for that environment. We assume these are known ahead of time. This inner loop evaluation procedure  $\operatorname{Eval}(L, \mathcal{E})$  is outlined in Algorithm 1.

The goal of the meta-learner is to find the optimal loss function  $L(s_{t}, a_{t}, r_{t}, s_{t+1}, \theta, \gamma)$  to optimize  $\pi_{\theta}$  with maximal normalized average training return over the set of training environments. The full objective for the meta-learner is:

$$
L ^ {*} = \arg \max _ {L} \left[ \sum_ {\mathcal {E}} \operatorname {E v a l} (L, \mathcal {E}) \right]
$$

$L$  is represented as a computational graph which we describe in the next section.

# 3.2 SEARCH LANGUAGE

Our search language for the algorithm  $L$  should be expressive enough to represent existing algorithms while enabling the learning of new algorithms which can obtain good generalization performance across a wide range of environments. Similar to Alet et al. (2020), we describe the RL algorithm as general programs with a domain specific language, but we target updates to the policy

![](images/7f51f9da60e3a6b28cca56cddf39cc86a9fc971d5791d2218016f377405aac72.jpg)  
Figure 2: Visualization of a RL algorithm, DQN, as a computational graph. Input nodes are outlined in blue, parameter nodes in gray, operation nodes in orange, and output in green.

rather than reward bonuses for exploration. Algorithms will map transitions  $(s_t, a_t, s_{t+1}, r_t)$ , policy parameters  $\theta$ , and discount factor  $\gamma$  into a scalar loss to be optimized with gradient descent. We express  $L$  as a computational graph or directed acyclic graph (DAG) of nodes with typed inputs and outputs. See Figure 2 for a visualization of DQN expressed in this form. Nodes can be of several types:

Input nodes represent inputs to the program, and include elements from transitions  $(s_t, a_t, s_{t+1}, r_t)$  and constants, such as the discount factor  $\gamma$ .

Parameter nodes are neural network weights, which can map between various data types. For example, the weights for the Q-value network will map an input node with state data type to a list of real numbers for each action.

Operation nodes compute outputs given inputs from parent nodes. This includes applying parameter nodes, as well as basic math operators from linear algebra, probability, and statistics. A full list of operation nodes is provided in Appendix A. By default, we set the last node in the graph to compute the output of the program which is the scalar loss function to be optimized. Importantly, the inputs and outputs of nodes are typed among (state, action, vector, float, list, probability). This typing allows for programs to be applied to any domain. It also restricts the space of programs to ones with valid typing which reduces the search space.

Algorithm 1 Algorithm Evaluation, Eval  $(L,\mathcal{E})$  1: Input: RL Algorithm  $L$ , Environment  $\mathcal{E}$ , training episodes  $M$  2: Initialize: Q-value parameters  $\theta$ , target parameters  $\theta^{\prime}$  empty replay buffer  $\mathcal{D}$  3: for  $i = 1$  to  $M$  do 4: for  $t = 0$  to  $T$  do 5: With probability  $\epsilon$ , select a random action  $a_{t}$  6: otherwise select  $a_{t} = \arg \max_{a}Q(s_{t},a)$  7: Step environment  $s_{t + 1},r_t\sim \mathcal{E}(a_t,s_t)$  8:  $\mathcal{D}\gets \mathcal{D}\cup \{s_t,a_t,r_t,s_{t + 1}\}$  9: Update parameters  $\theta \leftarrow \theta -\nabla_{\theta}L(s_{t},a_{t},r_{t},s_{t + 1},\theta ,\gamma)$  10: Update target  $\theta^{\prime}\gets \theta$  11: end for 12: Compute episode return  $R_{m} = \sum_{t = 0}^{T}r_{t}$  13: end for 14: Output: Normalized training performance  $\frac{1}{M}\sum_{m = 1}^{M}\frac{R_m - R_{min}}{R_{max} - R_{min}}$  15: Output: Algorithm L with highest score

# 3.3 EVOLUTIONARY SEARCH METHOD

Evaluating thousands of programs over a range of complex environments is prohibitively expensive, especially if done serially. For the search method, we adapt regularized evolution, RE, (Real et al., 2019) which has been shown to work for learning supervised learning algorithms (Real et al., 2020) and can be parallelized across compute nodes. RE keeps a population of  $P$  algorithms and improves the population through cycles. Each cycle picks a tournament of  $T < P$  algorithms at random and selects the best algorithm in the tournament as a parent. The parent is mutated into a child algorithm which gets added to the population while the oldest algorithm in the population is removed. We use a single type of mutation which first chooses which node in the graph to mutate and then replaces it with a random operation node with inputs drawn uniformly from all possible inputs.

There exists a combinatorially large number of graph configurations. Furthermore, in RL, evaluating a single graph, which means training the full inner loop RL algorithm, can take up a large amount of time compared to the supervised learning setting. Speeding up the search and avoiding needless computation are needed to make the problem more tractable. We extend Real et al. (2019) with several techniques, detailed below, to make the optimization more efficient. The full training procedure is outlined in Algorithm 2.

Functional equivalence check. Before evaluating a program, we check if it is functionally equivalent to any previously evaluated program. This check is done by hashing the concatenated output of the program for 10 values of randomized inputs. This technique is similar to the ones used by Alet et al. (2020) and Real et al. (2020). For regularized evolution, if a mutated program is functionally equivalent to an older program, we still add it to the population, but use the saved score of the older program. Since some nodes of the graph do not always contribute to the output, parts of the mutated program may eventually contribute to a functionally different program.

Early hurdles. We want poor performing programs to terminate early so that we can avoid unneeded computation. We use the CartPole environment as an early hurdle environment  $\mathcal{E}_h$  by training a program for a fixed number of episodes. If an algorithm performs poorly, then episodes will terminate in a short number of steps (as the pole falls rapidly) which quickly exhausts the number of training episodes. We use  $\operatorname{Eval}(L, \mathcal{E}_h) < \alpha$  as the threshold for poor performance with  $\alpha$  chosen empirically.

Program checks. We perform basic checks to rule out and skip training invalid programs. The loss function needs to be a scalar value so we check if the program output type is a float  $(\text{type}(L) = \mathbb{R})$ . Additionally, we check if each program is differentiable with respect to the policy parameters by checking if a path exists in the graph between the output and the policy parameter node.

Learning from Scratch and Bootstrapping. Our method enables both learning from scratch and learning from existing knowledge by bootstrapping the initial algorithm population with existing algorithms. We learn algorithms from scratch by initializing the population of algorithms randomly. An algorithm is sampled by sampling each operation node sequentially in the DAG. For each node, an operation and valid inputs to that operation are sampled uniformly over all possible options.

While learning from scratch might uncover completely new algorithms that differ substantially from the existing methods, this method can take longer to converge to a reasonable algorithm. We would like to incorporate the knowledge we do have of good algorithms to bootstrap our search from a better starting point. We initialize our graph with the loss function of DQN (Mnih et al., 2013) so that the first 7 nodes represent the standard DQN loss, while the remaining nodes are initialized randomly. During regularized evolution, the nodes are not frozen, such that it is possible for the existing sub-graph to be completely replaced if a better solution is found.

# 4 LEARNED RL ALGORITHM RESULTS AND ANALYSIS

We discuss the training setup and results of our experiments. We highlight two learned algorithms with good generalization performance, DQNClipped and DQNReg, and analyze their structure.

# 4.1 TRAINING SETUP

Meta-Training details: We search over programs with maximum 20 nodes, not including inputs or parameter nodes. A full list of node types is provided in Appendix A. We use a population size of 300, tournament size of 25, and choose these parameters based on the ones used in (Real et al., 2019). Mutations occur with probability 0.95. Otherwise a new random program is sampled. The search is done over 300 CPUs and run for roughly 72 hours, at which point around 20,000 programs have been evaluated. Further meta-training details are in Appendix B.

Training environments: The choice of training environments greatly affects the learned algorithms and their generalization performance. At the same time, our training environments should be not too computationally expensive to run as we will be evaluating thousands of RL algorithms. We use a range of 4 classical control tasks (CartPole, Acrobot, MountainCar, LunarLander) and a set of 12 multitask gridworld style environments from MiniGrid (Chevalier-Boisvert et al., 2018). These environments are computationally cheap to run but also chosen to cover a diverse set of situations. This includes dense and sparse reward, long time horizon, and tasks requiring solving a sequence of subgoals such as picking up a key and unlocking a door. More details are in Appendix C.

The training environments always include CartPole as an initial hurdle. If an algorithm succeeds on CartPole (normalized training performance greater than 0.6), it then proceeds to a harder set of training environments. For our experiments, we choose these training environments by sampling a set of 3 environments and leave the rest as test environments. For learning from scratch we also compare the effect of number of training environments on the learned algorithm by comparing training on just CartPole versus training on CartPole and LunarLander.

RL Training details: For training the RL agent, we use the same hyperparameters across all training and test environments except as noted. All neural networks are MLPs of size (256, 256) with ReLU activations. We use the Adam optimizer with a learning rate of 0.0001.  $\epsilon$  is decayed linearly from 1 to 0.05 over 1e3 steps for the classical control tasks and over 1e5 steps for the MiniGrid tasks.

![](images/0b351cb0ac4ab42393022b2e135accb03c8726b179da0f590b8d471899656473.jpg)  
(a) Learning curve

![](images/2a8050c5732bf9528e98dca1acce50c82a614fa1a7b75eb563b6c9d788cbd5f4.jpg)  
Figure 3: Left: Meta-training performance over different number of environments from scratch, and bootstrapping. Plotted as RL evaluation performance (sum of normalized training return across the training environments) over the number of candidate algorithms. More training environments leads to better algorithms. Bootstrapping from DQN speeds up convergence and higher final performance. Right: Meta-training performance histogram for bootstrapped training. Many of the top programs have similar structure (Appendix D).  
(b) Performance histogram

# 4.2 LEARNING CONVERGENCE

Figure 3a shows convergence over several training configurations. We find that at the end of training roughly  $70\%$  of proposed algorithms are functionally equivalent to a previously evaluated program, while early hurdles cut roughly another  $40\%$  of proposed non-duplicate programs.

Varying number of training environments: We compare learning from scratch with a single training environment (CartPole) versus with two training environments (CartPole and LunarLander). While both experiments reach the maximum performance on these environments (Figure 3a), the learned algorithms are different. The two-environment training setup is learns the known TD loss

$$
L = (Q (s _ {t}, a _ {t}) - (r _ {t} + \gamma * \max _ {a} Q _ {t a r g} (s _ {t}, a))) ^ {2}
$$

while the single-environment training setup learns a slight variation  $L = (Q(s_{t},a_{t}) - (r_{t} + \max_{a}Q_{\text{target}}(s_{t},a)))^{2}$  that does not use the discount, indicating that the range of difficulty on the training environments is important for learning algorithms which can generalize.

Learning from scratch versus bootstrapping: In Figure 3a, we compare training from scratch versus training from bootstrapping on four training environments (CartPole, KeyCorridorS3R1, DynamicObstacle-6x6, DoorKey-5x5). The training performance does not saturate, leaving room for improvement. Bootstrapping from DQN significantly improves both the convergence and performance of the meta-training, resulting in a  $40\%$  increase in final training performance.

# 4.3 LEARNED RL ALGORITHMS: DQNCLIPPED AND DQNREG

In this section, we discuss two particularly interesting loss functions that were learned by our method, and that have good generalization performance on the test environments. Let

$$
Y _ {t} = r _ {t} + \gamma * \max  _ {a} Q _ {t a r g} (s _ {t}, a), \text {a n d} \delta = Q (s _ {t}, a _ {t}) - Y _ {t}.
$$

The first loss function DQNClipped is

$$
L _ {\mathrm {D Q N C l i p p e d}} = \max \left[ Q (s _ {t}, a _ {t}), \delta^ {2} + Y _ {t} \right] + \max \left[ Q (s _ {t}, a _ {t}) - Y _ {t}, \gamma (\max _ {a} Q _ {t a r g} (s _ {t}, a)) ^ {2} \right].
$$

$L_{\mathrm{DQNClipped}}$  was trained from bootstrapping off DQN using three training environments (LunarLander, MiniGrid-Dynamic-Obstacles-5x5, MiniGrid-LavaGapS5). It outperforms DQN and double-DQN, DDQN, (van Hasselt et al., 2015) on both the training and unseen environments (Figure 4). The intuition behind this loss function is that, if the Q-values become too large (when  $Q(s_{t},a_{t}) > \delta^{2} + Y_{t}$ ), the loss will act to minimize  $Q(s_{t},a_{t})$  instead of the normal  $\delta^2$  loss. Alternatively, we can view this condition as  $\delta = Q(s_{t},a_{t}) - Y_{t} > \delta^{2}$ . This means when  $\delta$  is small enough then  $Q(s_{t},a_{t})$  are relatively close and the loss is just to minimize  $Q(s_{t},a_{t})$ .

The second learned loss function, which we call DQNReg, is given by

$$
L _ {\mathrm {D Q N R e g}} = 0. 1 * Q \left(s _ {t}, a _ {t}\right) + \delta^ {2}.
$$

![](images/b91e2c091bf005a5f67853011f0d989aa3e19cf907f61006081267c6deb1403d.jpg)

![](images/0676b7672319c2aa450623d75edb1e7302cf735663fee471a50cfff8cefffc43.jpg)

![](images/5bf8096d769bdd17c494d3c70613cd2bb67a681f5f7d62ff3a39a7d35134abdc.jpg)

![](images/cf80d2b725658cbc8e59a3b30fb8de67efc63385f5dd2ae073be80f840f915e9.jpg)

![](images/2a1beb4f432b76ad15b3de49dad4a2885b15bec3202644a5c0fbed9368da9257.jpg)

![](images/ab8f6f026abea8e58b17f67323a874d6775d30338e21891c1e3abf9e87b5b0bb.jpg)

![](images/adeffbc76d88b03d5a43399c7fc2b27a567836b55cb1a3d0bc28946492dd43b8.jpg)

![](images/8ea56a277db7032d1d97862b95cc340aa525666af2b217c88fce2bfd49d914ea.jpg)

![](images/53d1caa73c230761b56aa09440541830dfca934eb181e7f60aaf62baf7d0792c.jpg)

![](images/7071b666898a8944e95789d71602bae48b8c83dee592819013b723a5d8d0212c.jpg)

![](images/fabdee83d31e76aa44b4a53090c52a8f1cbd3e8a1ea907f7a4aabd650d821e9b.jpg)

![](images/ebb560d1c488aa6998df6c82036aa066b2dcdb67430973a7aae213b5778437cc.jpg)

![](images/7ac678b51df215d71534b3aa3db5ea5a7f545ddf44ef1e92762185827ebfb8eb.jpg)  
Figure 4: Performance of learned algorithms (DQNClipped and DQNReg) versus baselines (DQN and DDQN) on training and test environments as measured by episode return over 10 training seeds. A dashed line indicates that the algorithm was meta-trained on that environment while a solid line indicates a test environment. DQNReg can match or outperform the baselines on almost all the training and test environments. Shaded regions correspond to 1 standard deviation.

![](images/8999d32a85968081bbba0c5befbcb6f5d7536e31818cbb18329d7982964294ec.jpg)

![](images/fd93c9ac480879359ed7265a56c9e4357a1a462927ea9414559e3c2cadcd4ecd.jpg)

![](images/e9c6d395296798588d85b29f91256739017fe8dff94d2aca1939b16891674f6a.jpg)

DQNReg was trained from bootstrapping off DQN using three training environments (KeyCorridorS3R1, Dynamic-Obstacles-6x6, DoorKey-5x5). In comparison to DQNClipped, DQNReg directly regularizes the Q values with a weighted term that is always active. We note that both of these loss functions modify the original DQN loss function to regularize the Q-values to be lower in value. While DQNReg is quite simple, it matches or outperforms the baselines on all training and test environments including from classical control and Minigrid. It does particularly well on a few test environments (SimpleCrossingS9N1, DoorKey-6x6, and Unlock) and solves the tasks when other methods fail to attain any reward. It is also much more stable with lower variance between seeds, and more sample efficient on test environments (LavaGapS5, Empty-6x6, Empty-Random-5x5).

These algorithms are related to recently proposed RL algorithms, conservative Q-learning (CQL) (Kumar et al., 2020) and M-DQN (Vieillard et al., 2020). CQL learns a conservative Q-function by augmenting the standard Bellman error objective with a simple Q-value regularizer:  $\log \sum_{a}\exp \left(Q(s_{t},a)\right) - Q(s_{t},a_{t})$  which encourages the agent to stay close to the data distribution while maintaining a maximum entropy policy. DQNReg similarly augments the standard objective with a Q-value regularizer although does so in a different direction by preventing overestimation. M-DQN modifies DQN by adding the scaled log-policy (using the softmax Q-values) to the immediate reward. Both of these methods can be seen as ways to regularize a value-based policy. This resemblance indicates that our method can find useful structures automatically that are currently being explored manually, and could be used to propose new areas for researchers to explore.

We discover that the best performing algorithms from the experiment which learned DQNReg are consistent, and in the form  $L = \delta^2 + k * Q(s_t, a_t)$ . This loss could use further analysis and investigation, possibly environment-specific tuning of the parameter  $k$ . See Appendix 2 for details.

# 4.4 ANALYSIS OF LEARNED ALGORITHMS

We analyze the learned algorithms further to attempt to understand their beneficial effect on performance. In Figure 5, we compare the estimated Q-values for each algorithm during training. We

![](images/2edf37f645cf3ee6fb8a8cb05f45ed59190a5b0e9fb31e8d5272464680d68d9c.jpg)  
Figure 5: Overestimated value estimates is generally problematic in value-based RL. Our method learns algorithms which regularize the Q-values helping with overestimation. We compare the estimated Q-values for our learned algorithms and baselines with the optimal ground truth Q-values across several environments during training. Estimate is for taking action zero from the initial state of the environment. While DQN overestimates the Q-values, our learned algorithms DQNClipped and DQNReg underestimate the Q-values.

![](images/531195e8c73734382a0167a59de9e0d299c47826b9f070195de58ec8001de279.jpg)

![](images/38b75ccfa41765e7a7b99866df113fb46ad3716ee38b7162d9b94ee922271fe0.jpg)

see that DQN frequently overestimates the Q values while DDQN consistently underestimates the Q values before converging to the ground truth Q value which are computed with a manually designed optimal policy. DQNClipped has similar performance to DDQN, in that it also consistently underestimates the Q values and does so slightly more aggressively than DDQN. DQNReg significantly undershoots the Q values and does not converge to the ground truth. Various works (van Hasselt et al., 2015; Haarnoja et al., 2018; Fujimoto et al., 2018) have shown that overestimated value estimates is problematic and restricting the overestimation improves performance.

The loss function in DQNClipped is composed of the sum of two max operations, and so we can analyze when each update rule is active. We interpret DQNClipped as  $\max(v_1, v_2) + \max(v_2, v_3)$  with four cases: 1)  $v_1 > v_2$  and  $v_3 > v_4$ , 2)  $v_1 > v_2$  and  $v_3 < v_4$ , 3)  $v_1 < v_2$  and  $v_3 < v_4$ , 4)  $v_1 < v_2$  and  $v_3 > v_4$ . Case 2 corresponds to minimizing the Q values. Case 3 would correspond to the normal DQN loss of  $\delta^2$  since the parameters of  $Q_{targ}$  are not updated during gradient descent. In Figure 6, we plot the proportion of when each case is active during training. We see that usually case 3 is generally the most active with a small dip in the beginning but then stays around  $95\%$ . Meanwhile, case 2, which regularizes the Q-values, has a small increase in the beginning and then decreases later, matching with our analysis in Figure 6, which shows that DQNClipped strongly underestimates the Q-values in the beginning of training. This can be seen as a constrained optimization where the amount of Q-value regularization is tuned accordingly. The regularization is stronger in the beginning of training when overestimation are problematic ( $Q(s_t, a_t) > \delta^2 + Y_t$ ) and gets weaker as  $\delta^2$  gets smaller.

![](images/9f196eca9275b7765607b786f286ba061376c9c17dcd3dfb5f0532f76cdc4602.jpg)  
Figure 6: Our learned algorithm, DQNClipped, can be broken down into four update rules where each rule is active under certain conditions. Case 3 corresponds to normal TD learning while case 2 corresponds to minimizing the Q-values. Case 2 is more active in the beginning when value overestimation is a problem and then becomes less active as it is no longer needed.

# 5 CONCLUSION

In this work, we have presented a method for learning reinforcement learning algorithms. We design a general language for representing algorithms which compute the loss function for value-based model-free RL agents to optimize. We highlight two learned algorithms which although relatively simple, can obtain good generalization performance over a wide range of environments. Our analysis of the learned algorithms sheds insight on their benefit as regularization terms which are similar to recently proposed algorithms.

Our work is limited to discrete action and value-based RL algorithms that are close to DQN, but could easily be expanded to express more general RL algorithms such as actor-critic or policy gradient methods. How actions are sampled from the policy could also be part of the search space. The set of environments we use for both training and testing could also be expanded to include a more diverse set of problem types. We leave these problems for future work.

# REFERENCES

Ferran Alet, M. F. Schneider, Tomas Lozano-Perez, and L. Kaelbling. Meta-learning curiosity algorithms. ArXiv, abs/2003.05325, 2020.  
Irwan Bello, Barret Zoph, Vijay Vasudevan, and Quoc V Le. Neural optimizer search with reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 459-468. JMLR.org, 2017.  
Yoshua Bengio, S. Bengio, and J. Cloutier. Learning a synaptic learning rule. *IJCNN-91-Seattle International Joint Conference on Neural Networks*, ii:969 vol.2-, 1991.  
Han Cai, Tianyao Chen, Weinan Zhang, Yong Yu, and Jun Wang. Efficient architecture search by network transformation. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pp. 2787-2794, 2018.  
Yevgen Chebotar, Artem Molchanov, Sarah Bechtle, Ludovic Righetti, F. Meier, and Gaurav S. Sukhatme. Meta-learning via learned loss. *ArXiv*, abs/1906.05374, 2019.  
Maxime Chevalier-Boisvert, Lucas Willems, and Suman Pal. Minimalistic gridworld environment for operai gym. https://github.com/maximecb/gym-minigrid, 2018.  
Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and P. Abbeel. RL2: Fast reinforcement learning via slow reinforcement learning. *ArXiv*, abs/1611.02779, 2016.  
Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Neural architecture search: A survey. arXiv preprint arXiv:1808.05377, 2018.  
Aleksandra Faust, Anthony Francis, and Dar Mehta. Evolving rewards to automate reinforcement learning. arXiv preprint arXiv:1905.07628, 2019.  
Chelsea Finn and S. Levine. Meta-learning and universality: Deep representations and gradient descent can approximate any learning algorithm. ArXiv, abs/1710.11622, 2018.  
Chelsea Finn, P. Abbeel, and S. Levine. Model-agnostic meta-learning for fast adaptation of deep networks. *ArXiv*, abs/1703.03400, 2017.  
Jörg K. H. Franke, Gregor Köhler, André Biedenkapp, and Frank Hutter. Sample-efficient automated deep reinforcement learning. 2020.  
Scott Fujimoto, H. V. Hoof, and David Meger. Addressing function approximation error in actor-critic methods. *ArXiv*, abs/1802.09477, 2018.  
Toumas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In ICML, 2018.  
John H. Holland. Adaptation in Natural and Artificial Systems. University of Michigan Press, Ann Arbor, MI, 1975. second edition, 1992.  
Rein Houthooft, Richard Y. Chen, Phillip Isola, Bradly C. Stadie, F. Wolski, Jonathan Ho, and P. Abbeel. Evolved policy gradients. ArXiv, abs/1802.04821, 2018.  
Frank Hutter, Lars Kotthoff, and Joaquin Vanschoren (eds.). Automated Machine Learning: Methods, Systems, Challenges. Springer, 2018. In press, available at http://automl.org/book.  
Shauharda Khadka and Kagan Tumer. Evolution-guided policy gradient in reinforcement learning. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems 31, pp. 1188-1200. Curran Associates, Inc., 2018.  
Louis Kirsch, Sjoerd van Steenkiste, and J. Schmidhuber. Improving generalization in meta reinforcement learning using learned objectives. ArXiv, abs/1910.04098, 2020.

John Koza. Genetic programming - on the programming of computers by means of natural selection. In Complex adaptive systems, 1993.  
Aviral Kumar, Aurick Zhou, G. Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. *ArXiv*, abs/2006.04779, 2020.  
Chenxi Liu, Barret Zoph, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan L. Yuille, Jonathan Huang, and Kevin Murphy. Progressive neural architecture search. CoRR, abs/1712.00559, 2017. URL http://arxiv.org/abs/1712.00559.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin A. Riedmiller. Playing atari with deep reinforcement learning. *ArXiv*, abs/1312.5602, 2013.  
Junhyuk Oh, Matteo Hessel, Wojciech Marian Czarnecki, Zhongwen Xu, Hado van Hasselt, Satinder Singh, and David Silver. Discovering reinforcement learning algorithms. ArXiv, abs/2007.08794, 2020.  
Hieu Pham, Melody Y Guan, Barret Zoph, Quoc V Le, and Jeff Dean. Efficient neural architecture search via parameter sharing. In ICML, 2018.  
Thomas Pierrot, Guillaume Ligner, Scott Reed, Olivier Sigaud, Nicolas Perrin, Alexandre Laterre, David Kas, Karim Beguir, and Nando de Freitas. Learning compositional neural programs with recursive tree search and planning. NeurIPS, 2019.  
Esteban Real, Sherry Moore, Andrew Selle, Saurabh Saxena, Yutaka Leon Suematsu, Jie Tan, Quoc V. Le, and Alexey Kurakin. Large-scale evolution of image classifiers. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML'17, pp. 2902-2911. JMLR.org, 2017.  
Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V. Le. Regularized evolution for image classifier architecture search. In AAAI, volume abs/1802.01548, 2019.  
Esteban Real, Chen Liang, David So, and Quoc V. Le. Automl-zero: Evolving machine learning algorithms from scratch. In ICML, volume abs/2003.03384, 2020.  
Scott Reed and Nando De Freitas. Neural programmer-interpreters. arXiv preprint arXiv:1511.06279, 2015.  
Juergen Schmidhuber. Evolutionary principles in self-referential learning. 1987.  
Juergen Schmidhuber. A self-referential weight matrix. 1993.  
Xingyou Song, Krzysztof Choromanski, Jack Parker-Holder, Yunhao Tang, Wenbo Gao, Aldo Pacchiano, Tamas Sarlos, Deepali Jain, and Yuxiang Yang. Reinforcement learning with chromatic networks for compact architecture search. 2020.  
Kenneth O Stanley and Risto Miikkulainen. Evolving neural networks through augmenting topologies. Evolutionary computation, 10(2):99-127, 2002.  
Yunhao Tang and Krzysztof Choromanski. Online hyper-parameter tuning in off-policy learning via evolutionary strategies, 2020.  
Hado van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. In AAAI, 2015.  
Nino Vieillard, Olivier Pietquin, and M. Geist. *Munchhausen reinforcement learning.* ArXiv, abs/2007.14430, 2020.  
Jane X. Wang, Zeb Kurth-Nelson, Hubert Soyer, Joel Z. Leibo, Dhruva Tirumala, Rémi Munos, Charles Blundell, D. Kumaran, and Matt M. Botvinick. Learning to reinforcement learn. ArXiv, abs/1611.05763, 2017.  
Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. In ICLR, 2016.  
Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V Le. Learning transferable architectures for scalable image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8697-8710, 2018.
