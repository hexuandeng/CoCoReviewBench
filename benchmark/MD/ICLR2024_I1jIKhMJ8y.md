# LEARNING EMBEDDINGS FORSEQUENTIAL TASKS USING POPULATION OF AGENTS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present an information-theoretic framework to learn fixed-dimensional embeddings for tasks in reinforcement learning. We leverage the idea that two tasks are similar if observing an agent's performance on one task reduces our uncertainty about its performance on the other. This intuition is captured by our information-theoretic criterion which uses a diverse agent population as an approximation for the space of agents to measure similarity between tasks in sequential decision-making settings. In addition to qualitative assessment, we empirically demonstrate the effectiveness of our techniques based on task embeddings by quantitative comparisons against strong baselines on two application scenarios: predicting an agent's performance on a new task by observing its performance on a small quiz of tasks, and selecting tasks with desired characteristics from a given set of options.

# 1 INTRODUCTION

Embeddings are widely used to represent data points as vectors in a space that captures meaningful relations between them (Sun et al., 2014; Sung et al., 2018; Athar et al., 2020; Mikolov et al., 2013; Pennington et al., 2014; Cer et al., 2018; Zhang et al., 2021). They could also be utilized as representations for tasks, as studied in various areas such as multi-task learning (Zhang et al., 2018), meta-learning (Achille et al., 2019), and domain-adaptation (Peng et al., 2020).

In reinforcement learning (RL), task embeddings could be used to understand the shared structure in sequential decision-making problems if similar tasks are embedded in close proximity. Such embeddings could enable efficient, one-shot computation of task similarity, eliminating the need for time-consuming policy rollouts. Essentially, there is an underlying notion of skills required to solve sequential tasks, and several of these tasks require some skills in common. For instance, consider the tasks shown in Fig. 1. Each requires the agent to pick-up certain keys to unlock the door. The door in task  $s_1$  requires the green key and the blue key, while the door in task  $s_2$  requires the yellow key and the blue key. Thus, these tasks require the common skills of navigation, and picking the blue key.

Despite the potential benefits, prior work on learning task embeddings in RL (Qin et al., 2022; Schäfer et al., 2022; Arnekvist et al., 2018; Yoo et al., 2022; Rakelly et al., 2019; Bing et al., 2023; Gupta et al., 2018; Fu et al., 2020; Li et al., 2021; Lan et al., 2019; Walke et al., 2022; Sodhani et al., 2021b; Vuorio et al., 2019) does not explicitly optimize for task similarity. This could primarily be attributed to the lack of a general framework to measure (and reason about) similarities among sequential tasks.

To this end, we introduce an information-theoretic framework to learn fixed-dimensional embeddings for tasks in RL; the inner product in the embedding space captures similarity between tasks, and the norm of the embedding induces an ordering on the tasks based on their difficulties (see Fig. 1). A critical component of the framework is a population of agents exhibiting a diverse set of behaviors, which serves as an approximation for the space of agents. Our framework leverages the idea that two sequential tasks are similar to each other if observing the performance of an agent from this population on one task significantly decreases our uncertainty about its performance on the other. Concretely, we introduce an information-theoretic criterion to measure task similarity (Section 4.1), and an algorithm to empirically estimate it (Section 4.2). Through this, we construct a set of ordinal constraints on the embeddings (with each such constraint asserting the relative similarity between a triplet of tasks), and propose a training scheme for an embedding network to learn them (Section 4.3).

![](images/5fddd3517e6012781e27013554a192ed716e60afdbea775c9f298aaa3d46224d.jpg)  
Figure 1: Schematics of our approach. We learn a task embedding function  $f_{\phi}(.)$  that maps a task  $s$  to its fixed-dimensional representation  $E$ . In this illustration, we show the properties of the learned embeddings using the MULTIKEYNAV environment in which tasks require the agent (shown as a black circle) to pick-up certain keys (from the gray segments) to unlock the door (the right-most segment) that has certain requirements (shown in color in the form of gates). A possible solution trajectory is depicted using dotted lines. Keys on this trajectory correspond to the ones that the agent possesses at that point in time. For instance, in task  $s_2$ , the agent starts off with the yellow key in possession already.  $\langle \mathrm{E}_1, \mathrm{E}_2 \rangle$  is greater than  $\langle \mathrm{E}_1, \mathrm{E}_3 \rangle$ , since tasks  $s_1$  and  $s_2$  have a common requirement of picking the blue key, and thus, are similar. Additionally,  $\| \mathrm{E}_2 \|_2$  is less than both  $\| \mathrm{E}_1 \|_2$  and  $\| \mathrm{E}_3 \|_2$ , since task  $s_2$  requires picking a single key, while tasks  $s_1$  and  $s_3$  require picking two keys, which makes them harder than  $s_2$ .

Besides assessing the learned embedding spaces through visualizations (Section 5), we ground our framework in two downstream scenarios that are inspired by real-world applications (Section 6). Firstly, we show the utility of our framework in predicting an agent's performance on a new task given its performance on a small quiz of tasks, which is similar to assessing a student's proficiency in adaptive learning platforms via a compact quiz (He-Yueya & Singla, 2021). Secondly, we demonstrate the application of our framework in selecting tasks with desired characteristics from a given set of options, such as choosing tasks that are slightly harder than a reference task. This is analogous to selecting desired questions from a pool for a personalized learning experience in online education systems (Ghosh et al., 2022). Through comparisons with strong baselines on a diverse set of environments, we show the efficacy of our techniques based on task embeddings.

To summarize, our work makes the following key contributions:

I. We introduce an information-theoretic framework to learn task embeddings in RL. As part of the framework, we propose a task similarity criterion which uses a diverse population of agents to measure similarity among sequential tasks (Sections 4.1 and 4.2).  
II. We propose a scheme to learn task embeddings by leveraging the ordinal constraints imposed by our similarity criterion (Section 4.3).  
III. To assess our framework, we perform visual assessments of the learned embedding spaces, and introduce two quantitative benchmarks: (a) agent's performance prediction, and (b) task selection with desired characteristics (Sections 5 and 6).

# 2 RELATED WORK

Task embeddings in RL. Several works in the meta-learning and multi-task learning literature have explored the use of task embeddings to model relationships between sequential tasks, where embeddings are either learned explicitly through objectives such as reconstruction (Arnekvist et al., 2018; Yoo et al., 2022; Bing et al., 2023) and trajectory-based contrastive learning (Fu et al., 2020; Li et al., 2021), or implicitly to aid generalization to new tasks (Lan et al., 2019; Walke et al., 2022; Sodhani et al., 2021b; Vuorio et al., 2019). While these methods integrate task embeddings with policies solely to improve performance, we propose a framework to learn general-purpose embeddings that can be used to quantify and analyze task similarities. Furthermore, in our framework, embedding computation is a one-shot operation, unlike prior work that relies on experience data from the policy for the task. These distinctions position our work as complementary to existing methods.

Population-based techniques. Our framework requires a diverse agent population. This is inline with (Furuta et al., 2021; Tylkin et al., 2021; Vinyals & et al., 2019; Jaderberg & et al., 2019; Parker-Holder et al., 2020), which use agent populations in the RL setting. For instance, Furuta et al. (2021) use a randomly generated agent population to empirically estimate policy information capacity, an information-theoretic measure of task difficulty in RL.

# 3 PROBLEM SETUP

MDP and Tasks. We use the Markov Decision Process (MDP) framework to define an environment. An MDP  $\mathcal{M}$  is defined as a 6-tuple  $(S, \mathcal{A}, \mathcal{R}, \mathcal{T}, S_{\mathrm{init}}, \gamma)$ , where  $\mathcal{S}$  is the state space,  $\mathcal{A}$  is the action space,  $\mathcal{R}: S \times \mathcal{A} \to \mathbb{R}$  is the reward function,  $\mathcal{T}: S \times S \times \mathcal{A} \to [0,1]$  is the transition dynamics, and  $S_{\mathrm{init}} \subseteq S$  is the set of initial states. Each state  $s \in S_{\mathrm{init}}$  corresponds to a goal-based task (for example, the goal could be to reach a specific destination in a navigation task) where the reward is 0 on all transitions but those on which a task gets completed. On task completion, the reward is 1. As an alternative to discounting, at each time step, there is a failure probability of  $1 - \gamma$ , which incentivizes the agent to solve the task quickly. This ensures that the cumulative reward is binary.

Population of agents and task solvability.  $p(\theta)$  represents a distribution over the population of agents. Concretely, it is a distribution over the agents' policy parameters. We use  $\Theta$  to represent the random variable that takes on the value  $\theta$ .  $\mathcal{O}_{s,\Theta} \in \{0,1\}$  is a Bernoulli random variable that takes on the value 1 if, on a rollout, an agent sampled from  $p(\theta)$  could successfully solve the task  $s \in S_{\mathrm{init}}$  (i.e., the cumulative reward is 1), and 0 otherwise. We call  $\mathcal{O}_{s,\Theta}$  the optimality variable for task  $s$ .  $\mathrm{POS}(s) := \mathbb{E}[\mathcal{O}_{s,\Theta}]$  denotes the probability of success on  $s$ , and is the complement of task difficulty.

Task embedding space. Formally, we wish to learn a task embedding function (parameterized by  $\phi$ )  $f_{\phi}: S_{\mathrm{init}} \to \mathbb{R}^n$ , for an MDP  $\mathcal{M}$  and a prior over the population of agents  $p(\theta)$ , that maps tasks to  $n$ -dimensional representations. The range of  $f_{\phi}(.)$  is the task embedding space.

Objective. Our objective is to learn embeddings for sequential tasks with the following properties: (a) the inner product in the embedding space captures task similarity, where the realizations of optimality variables are similar for tasks that are embedded in close proximity, and (b) the norm of the embedding induces an ordering on the tasks based on their difficulties. We formalize these objectives in Section 4.

# 4 LEARNING FRAMEWORK

In Sections 4.1 and 4.2, we formally define our information-theoretic criterion to measure task similarity in RL and describe an algorithm to empirically estimate it. In Section 4.3, we view the problem of learning task embeddings through the lens of ordinal constraint satisfaction.

# 4.1 INFORMATION-THEORETIC MEASURE OF TASK SIMILARITY

Our goal is to measure similarity between sequential tasks. To this end, we propose the mutual information between task optimality variables as a measure of task similarity. This metric captures the intuition that two tasks are similar to each other if observing an agent's performance on one task reduces our uncertainty about its performance on the other. We begin by formally defining performance uncertainty. Thereafter, we provide a formal definition of our task similarity criterion.

Definition 1 (Performance Uncertainty). The entropy of the population with prior  $p(\theta)$  solving a task  $s$  is defined as:

$$
\mathcal {H} (\mathcal {O} _ {s, \Theta}) = - \sum_ {o \in \{0, 1 \}} \mathrm {P} (\mathcal {O} _ {s, \Theta} = o) \log \mathrm {P} (\mathcal {O} _ {s, \Theta} = o),
$$

where  $\mathcal{O}_{s,\Theta}$  is the optimality variable for  $s$ .

Thus, we could measure the similarity between two tasks  $s_i, s_j \in S_{\mathrm{init}}$  as the reduction in  $\mathcal{H}(\mathcal{O}_{s_i,\Theta})$  by observing  $\mathcal{O}_{s_j,\Theta}$ .

Definition 2 (Task Similarity). Given a prior over the population of agents  $p(\theta)$ , we measure the similarity between two tasks  $s_i, s_j \in S_{\mathrm{init}}$  as the mutual information  $\mathcal{I}(:,.)$  between their optimality

variables  $\mathcal{O}_{s_i,\Theta}$ $\mathcal{O}_{s_j,\Theta}$  ..

$$
\mathcal {I} \left(\mathcal {O} _ {s _ {i}, \Theta}; \mathcal {O} _ {s _ {j}, \Theta}\right) = \mathcal {H} \left(\mathcal {O} _ {s _ {i}, \Theta}\right) - \mathcal {H} \left(\mathcal {O} _ {s _ {i}, \Theta} \mid \mathcal {O} _ {s _ {j}, \Theta}\right).
$$

It quantifies the information obtained about  $\mathcal{O}_{s_i,\Theta}$  by observing  $\mathcal{O}_{s_j,\Theta}$ .

# 4.2 EMPIRICAL ESTIMATION OF  $\mathcal{I}$

We now outline an algorithm to empirically estimate  $\mathcal{I}$ . A comprehensive pseudocode detailing the computation of the criterion is provided in Appendix B. Given an MDP  $\mathcal{M}$  and a prior distribution of the agent parameters  $p(\theta)$ , our algorithm uses  $\mathbb{N}$  samples to estimate  $\mathcal{I}(\mathcal{O}_{s_i,\Theta};\mathcal{O}_{s_j,\Theta})$ . For each sample, the algorithm randomly samples  $\theta_l\sim p(\theta)$ , and performs rollouts of  $\pi_{\theta_l}$  from  $s_i$  and  $s_j$  to obtain estimates of the probability mass functions required for the computation of  $\mathcal{I}$ . The estimation procedure can be invoked with the signature  $\mathrm{ESTIMATE}(s_i,s_j,\mathcal{M},\pi ,p(\theta),\mathbb{N})$ .

# 4.3 LEARNING

TASK EMBEDDINGS

# Algorithm 1 Learn the Task Embedding Function  $(f_{\phi})$

1: procedure TRAIN(Set of tasks  $\mathcal{S}_{\mathrm{init}}$  , MDP  $\mathcal{M}$  , Policy  $\pi$  , Prior distribution of the agent parameters  $p(\theta)$  , Number of samples N, Hyperparameter  $\lambda$  , Number of iterations M)   
2: Initialize  $\phi$    
3: for  $i\in \{1,\dots ,\mathbb{M}\}$  do   
4: Sample task  $s_1,s_2,s_3\sim S_{\mathrm{init}}$    
5:  $\mathrm{E}_1,\mathrm{E}_2,\mathrm{E}_3\gets f_\phi (s_1),f_\phi (s_2),f_\phi (s_3)$    
6:  $\hat{\mathcal{I}}_{12}\leftarrow$  ESTIMATE(s1,s2,M,  $\pi ,p(\theta),\mathbb{N})$    
7:  $\hat{\mathcal{I}}_{13}\leftarrow$  ESTIMATE(s1,s3,M,  $\pi ,p(\theta),\mathbb{N})$    
8: if  $\hat{\mathcal{I}}_{12} > \hat{\mathcal{I}}_{13}$  then   
9: loss  $\leftarrow \log (1 + \exp (\langle \mathrm{E}_1,\mathrm{E}_3\rangle -\langle \mathrm{E}_1,\mathrm{E}_2\rangle))$    
10: else   
11: loss  $\leftarrow \log (1 + \exp (\langle \mathrm{E}_1,\mathrm{E}_2\rangle -\langle \mathrm{E}_1,\mathrm{E}_3\rangle))$    
12: Sample task  $s_4,s_5\sim S_{\mathrm{init}}$    
13:  $\mathrm{E}_4,\mathrm{E}_5\gets f_\phi (s_4),f_\phi (s_5)$    
14: if POS(s4) > POS(s5) then   
15: loss  $\leftarrow$  loss +  $\lambda \log (1 + \exp (\| \mathrm{E}_4\| _2 - \| \mathrm{E}_5\| _2))$    
16: else   
17: loss  $\leftarrow$  loss +  $\lambda \log (1 + \exp (\| \mathrm{E}_5\| _2 - \| \mathrm{E}_4\| _2))$    
18: Update  $\phi$  to minimize loss.   
19: return  $\phi$

With the criterion to measure task sim

ality defined, we are interested in learning a task embedding function  $f_{\phi}: S_{\mathrm{init}} \to \mathbb{R}^n$  (consequently, an embedding space) that satisfies the desiderata introduced in Section 3. To this end, we pose the problem of learning  $f_{\phi}(.)$  as an ordinal constraint satisfaction problem. Essentially, the task similarity criterion  $\mathcal{I}$  imposes a set  $C_{\mathrm{MI}}$  of triplet ordinal constraints on the task embeddings.  $\mathrm{POS}(.)$  imposes another set  $C_{\mathrm{NORM}}$  of pairwise ordinal constraints.

Concretely,  $\mathcal{C}_{\mathrm{MI}}$  is a collection of ordered triplets of tasks s.t. for each  $(s_1,s_2,s_3)\in \mathcal{C}_{\mathrm{MI}}$ ,  $\mathcal{I}(\mathcal{O}_{s_1,\Theta};\mathcal{O}_{s_2,\Theta}) > \mathcal{I}(\mathcal{O}_{s_1,\Theta};\mathcal{O}_{s_3,\Theta})$ . Consequently, we would like to satisfy the constraint  $\langle f_{\phi}(s_1),f_{\phi}(s_2)\rangle >\langle f_{\phi}(s_1),f_{\phi}(s_3)\rangle$ . Likewise,  $\mathcal{C}_{\mathrm{NORM}}$  is a collection of ordered tuples of tasks s.t. for each  $(s_1,s_2)\in \mathcal{C}_{\mathrm{NORM}}$ ,  $\mathrm{POS}(s_1) > \mathrm{POS}(s_2)$ . Consequently, we would like to satisfy the constraint  $\| f_{\phi}(s_2)\| _2 > \| f_{\phi}(s_1)\| _2$  (embeddings for easier tasks have smaller norm).

We learn the task embedding function  $f_{\phi}(.)$ , for an MDP  $\mathcal{M}$  and a prior over the agent population  $p(\theta)$ , by optimizing the parameters  $\phi$  to maximize the log-likelihood of the ordinal constraints under the Bradley-Terry-Luce (BTL) model (Luce, 1959). Concretely, given a triplet of tasks  $(s_1, s_2, s_3)$ , we define:

$$
\mathrm {P} \big ((s _ {1}, s _ {2}, s _ {3}) \in \mathcal {C} _ {\mathrm {M I}} \big) := \frac {\exp \big (\langle f _ {\phi} (s _ {1}) , f _ {\phi} (s _ {2}) \rangle \big)}{\exp \big (\langle f _ {\phi} (s _ {1}) , f _ {\phi} (s _ {2}) \rangle \big) + \exp \big (\langle f _ {\phi} (s _ {1}) , f _ {\phi} (s _ {3}) \rangle \big)}.
$$

Similarly, given a tuple of tasks  $(s_1,s_2)$ , we define:

$$
\mathrm {P} \big ((s _ {1}, s _ {2}) \in \mathcal {C} _ {\mathrm {N O R M}} \big) := \frac {\exp \big (\| f _ {\phi} (s _ {2}) \| _ {2} \big)}{\exp \big (\| f _ {\phi} (s _ {1}) \| _ {2} \big) + \exp \big (\| f _ {\phi} (s _ {2}) \| _ {2} \big)}.
$$

Hence, the task embedding function  $f_{\phi}(.)$  is learned by solving the following optimization problem:

$$
\min_{\phi}\left[\mathbb{E}_{(s_{1},s_{2},s_{3})\sim \mathcal{C}_{\mathrm{MI}}}\log \Bigg(1 + \exp \big(\langle \mathrm{E}_{1},\mathrm{E}_{3}\rangle -\langle \mathrm{E}_{1},\mathrm{E}_{2}\rangle \big)\Big) + \lambda_{(s_{4},s_{5})\sim \mathcal{C}_{\mathrm{NORM}}}\mathbb{E}\log \Bigg(1 + \exp \big(\| \mathrm{E}_{4}\|_{2} - \| \mathrm{E}_{5}\|_{2}\big)\Big)\right],
$$

where  $\mathrm{E}_i$  denotes  $f_{\phi}(s_i)$ , and  $\lambda$  is a hyperparameter. The pseudocode for the proposed algorithm to learn the task embedding function  $f_{\phi}(.)$  is given in Algorithm 1.

(a) Comparison of environments' complexity  

<table><tr><td>Environment</td><td>Task Variability</td><td>Action</td><td>State</td><td>Number of Tasks</td></tr><tr><td>MULTIKEYNAV</td><td>Reward Function</td><td>7</td><td>R × {0,1}6</td><td>Infinite</td></tr><tr><td>CARTPOLEVAR</td><td>Dynamics</td><td>2</td><td>R5 × {0,1} × [200]</td><td>Infinite</td></tr><tr><td>POINTMASS</td><td>Dynamics</td><td>R2</td><td>R7</td><td>Infinite</td></tr><tr><td>KAREL</td><td>Reward Function + Dynamics</td><td>52</td><td>{0,1}51840</td><td>73688</td></tr><tr><td>BASICAREL</td><td>Reward Function + Dynamics</td><td>6</td><td>{0,1}88</td><td>24000</td></tr></table>

![](images/a6f92f9c57147039b3797e332cc3fb75e3a0ae2494ac265a85698a553f695a67.jpg)  
(b) Illustrations  
Figure 2: We evaluate our framework on a diverse set of environments. (a) compares the characteristics of these environments. (b) illustrates these environments for a better understanding of the tasks.

# 5 EXPERIMENTS: VISUALIZATION OF EMBEDDING SPACES

In this section, we visualize the embedding spaces to gather qualitative insights, addressing the following research questions: (i) Can distinct clusters of tasks be identified by visualizing the embedding space? (ii) How does regularization through  $\mathcal{C}_{\mathrm{NORM}}$  affect the embedding space? (iii) What influence do agent population and environment specification have on the embedding space? We begin by discussing the rationale for environment selection, describing these environments. Subsequently, we provide an overview of the embedding networks' training process, followed by the qualitative results.

# 5.1 ENVIRONMENTS

We evaluate our framework on environments with diverse characteristics to demonstrate its generality and scalability to different sequential decision-making problems (see Fig. 2). As the running example, we use MULTIKEYNAV (based on (Devidze et al., 2021)) because of its compositional nature in which the agent needs to compose different actions for picking keys (with four distinct key types, each requiring a specific action to be picked) in a task-specific manner to unlock the door. This also makes it suitable for ablation experiments. Task variability comes from the agent's initial position, the keys that it possesses initially, and the door type (with each type requiring a unique set of keys).

Given that task variability in MULTIKEYNAV comes from the reward function, we use CARTPOLEVAR to highlight our framework's applicability to environments where it comes from the dynamics instead. This environment is a variation of the classic control task from OpenAI gym (Brockman et al., 2016), and also takes inspiration from (Sodhani et al., 2021a) in which the forces applied by each action could be negative as well. Tasks in this environment require keeping a pole attached by an unactuated joint to a cart upright for 200 timesteps by applying forces to the left (action 0) or to the right (action 1) of the cart. Task variability comes from the force F applied on the cart by each action, and the TaskType  $\in$  {0, 1}. Tasks of Type 0 involve "Pulling" with action 0 pulling the cart from the left and action 1 pulling the cart from the right, while tasks of Type 1 involve "Pushing".

We select POINTMASS (introduced in (Klink et al., 2020)) to test if our framework can handle continuous action spaces. In this environment, the agent applies forces to control a point mass inside a walled square. Tasks require reaching a fixed goal position through a gate, with task variability arising from the gate width and position, along with the coefficient of kinetic friction of the space.

Finally, to investigate our framework's scalability, we use the real-world environment KAREL from (Bunel et al., 2018), which is a challenging environment with applications in programming education. Tasks in this environment require the agent to synthesize a program, potentially containing control flow constructs such as loops and conditionals, satisfying a given specification comprising input-output examples. This program serves as a controller for an avatar navigating a grid, where each cell could

![](images/43f48d972735d2ffea951df1db2b8c4a0c89c476c500c361f1003898da05724d.jpg)  
(a) MULTIKEYNAV

![](images/1a897c4e0dd96ade8f26dfb03b8bf4b71587d95157e438758de297fa57427809.jpg)

![](images/565e78ef66bf79771784879536381736887850b5e5121103fc9bb97152e8eab4.jpg)

![](images/6fc4c1da2582caad2253177a619b0a773b796f9434e5c513e2c733e8bc734809.jpg)  
(c) POINTMASS

![](images/eea74ec5f66d7c92e65940de020d0a730e954c09b1c8aaac5fb3fcdda4b8a745.jpg)  
(b) CARTPOLEVAR  
(d) KAREL

![](images/909e9f034a9d4b22f3e17ff5870061c126790ea6b19c9e091a5d5395ff3f7876.jpg)  
Figure 3: Visualization of the task embedding spaces learnt through our framework. Each point represents a task, and the size of the points is proportional to the norm of the embeddings.  
(e) BASICKAREL

contain marker(s), correspond to a wall, or be empty. The avatar can traverse the grid and manipulate it by picking or placing markers. Thus, an example in the specification comprises the Pre-Grid and the corresponding Post-Grid. In addition, we evaluate our framework on BASICKAREL (Tzannes et al., 2023), which is a simpler variant of KAREL that excludes control flow constructs.

# 5.2 TRAINING PROCESS

To learn the task embedding function, we first obtain the agent population by taking snapshots while training a neural network policy using either behavioral cloning (Bain & Sammut, 1995) or policy gradient methods (Sutton et al., 1999). Concretely, a snapshot is recorded if the average performance on a validation set of tasks (denoted as  $S_{\mathrm{snap}}$ ) improves by  $\delta_{\mathrm{snap}}$  compared to the previously recorded snapshot. A snapshot of the untrained policy is recorded by default. Different subpopulations, obtained by either masking actions or by using biased task distributions during training, are combined to form the final population. Here, masking a certain action corresponds to setting its logit to a large negative number. Using biased task distribution during training is another way to inject diversity into the population. In MULTIKEYNAV, for instance, using a biased task distribution could correspond to assigning low probability mass to tasks with certain types of doors in the initial state distribution during training. Finally, we parameterize the task embedding function  $f_{\phi}(.)$  with a neural network, optimizing its parameters as described in Algorithm 1. We provide additional details in Appendix E.

# 5.3 VISUALIZATIONS AND QUALITATIVE RESULTS

We visualize the embedding spaces on a 2-dimensional map using t-SNE (van der Maaten & Hinton, 2008) to identify distinct clusters of tasks. Although t-SNE preserves the local structure, it does not necessarily preserve the embeddings' norm. For this reason, we scale the points in proportion to the norm of the embeddings. Additionally, we provide PCA plots in Appendix G.

Visualizations. For MULTIKEYNAV (Fig. 3a), our framework discovers distinct clusters of tasks, with each cluster corresponding to a unique set of keys that need to be picked. The norm of the embeddings is in accordance with the number of keys that need to be picked (with tasks requiring navigation only having the smallest norm). Additionally, tasks in clusters adjacent to each other share a common key requirement. For CARTPOLEVAR (Fig. 3b), our framework discovers that each task exhibits one of two types of underlying dynamics. In one (+ve F and Type 0, or -ve F and Type 1), action 0 moves the cart to the left, while in the other (-ve F and Type 0, or +ve F and Type 1), action 0 moves the cart to the right. For POINTMASS (Fig. 3c), our framework discovers three clusters of tasks based on the behavior that the agent needs to exhibit near the gate. The first cluster includes tasks in which the agent need not steer to cross the gate, while the second and third clusters contain tasks in which the agent must steer left or right to cross the gate, respectively. For KAREL and BASICKAREL (Fig. 3d and 3e), our framework discovers different clusters of tasks based on

![](images/de592a4f91b75aeadd8b2cf835936262336d2cd4c1ae66ded8e184f3ab79ffc5.jpg)  
(a)

![](images/3e5020ce53ee1cbfb302a59258375ecdd28d9b94ae02dd3eb36aea896a2bb900.jpg)  
Figure 4: Task embedding spaces for the MULTIKEYNAV environment: (a) without  $\mathcal{C}_{\mathrm{NORM}}$ , (b) pickKey actions masked, (c) all doors require KeyA, KeyB, and (d) all doors require KeyA.  
(b)

![](images/5c013b5d08e97671de1a67993e111fa73baebc8dfa8679fab096d9518cbc1d74.jpg)  
(c)

![](images/77ce56ce54417570728b2259d69c94e5709badbff7c3fc10b2bd06dace0eedf8.jpg)  
(d)

whether the solution code requires loops or conditionals, and whether the agent needs to pick or put markers in the grid, respectively.

Ablation w.r.t.  $\mathbf{C}_{\mathrm{NORM}}$ . Fig. 4a shows the task embedding space learned without the norm ordinal constraints  $\mathcal{C}_{\mathrm{NORM}}$  (i.e.,  $\lambda$  is set to 0). As expected, the norm of the embeddings is not proportional to the number of keys that need to be picked. Instead, the points are nearly uniform in size.

Ablation w.r.t. population specification. To understand the effect of population on the task embedding space, we learn the embedding function  $f_{\phi}(\cdot)$  for MULTIKEYNAV using an agent population in which pickKey actions are masked (Fig. 4b). In this case, we obtain two distinct clusters of tasks – one of the clusters contains tasks that cannot be solved (these tasks require picking key(s)), and the other contains tasks that require navigation only. These results emphasize the importance of the population's quality in learning a good task embedding space.

Ablation w.r.t. environment specification. In this ablation experiment, we change the environment specification and check its impact on the task embedding space. Concretely, we learn the embedding space for the following variants of MULTIKEYNAV: (a) each door requires KeyA and KeyB (Fig. 4c), i.e., all the doors have identical key requirements, and (b) each door requires KeyA only (Fig. 4d). Modifying the environment specification changes the task semantics, thereby impacting the task embedding space. Thus, these results are inline with our intuition.

# 5.4 COMPARISON WITH EXISTING WORK

To compare our framework with existing methods, we introduce PredModel baseline (inspired by prior work) and use silhouette scores based on the intuitively identified clusters of tasks to measure clustering quality in the learned embedding spaces. We also compare our method against embedding networks with random weights (RandomModel).

Figure 5: Comparison of silhouette scores (higher is better) based on intuitively identified clusters of tasks in the learned embedding spaces. The scores for our models are consistently better.  

<table><tr><td>Environment</td><td>RandomModel</td><td>PredModel</td><td>Ours</td></tr><tr><td>MULTIKEYNAV</td><td>0.036 ± 0.048</td><td>-0.037 ± 0.003</td><td>0.753 ± 0.001</td></tr><tr><td>CARTPOLEVAR</td><td>0.015 ± 0.016</td><td>0.242 ± 0.007</td><td>0.325 ± 0.009</td></tr><tr><td>POINTMASS</td><td>0.104 ± 0.026</td><td>-0.010 ± 0.004</td><td>0.380 ± 0.019</td></tr><tr><td>BASICAREL</td><td>-0.058 ± 0.007</td><td>-0.002 ± 0.003</td><td>0.811 ± 0.019</td></tr></table>

Most existing methods (e.g., PEARL (Rakelly et al., 2019)) utilize variational inference to learn latent context from task-specific experience data, where the inference network could be trained to reconstruct the MDP for the task through predictive models of reward and dynamics. To adapt this approach to our setting, we connect our formalism of tasks as initial states to the contextual MDP setting (Hallak et al., 2015), where each context (e.g., MULTIKEYNAV's context: agent's initial position, possessed keys initially, door type) corresponds to a distinct task represented by a separate MDP with context-dependent transitions and rewards. This set of MDPs can be converted into an equivalent MDP by including context variables as part of the state. In this converted MDP, each initial state represents a task, as it determines the context for the entire episode. The context is observable.

The modifications needed for the PredModel baseline are as follows: Firstly, since context is observable in our setup, we condition the approximate posterior over the embeddings on the initial state, eliminating the need for experience data. Secondly, we train the predictive models on states with context variables removed, ensuring the utilization of the task embedding that the model is conditioned on. We provide additional technical details in Appendix D.

Results. Fig. 5 reports the silhouette scores, averaged across 3 random seeds, with 1000 tasks per seed (5000 for BASICKAREL). The scores for the models learned through our framework are consistently better. While the PredModel baseline clusters similar tasks together in the embedding space for CARTPOLEVAR, it fails to do so in rest of the environments. In contrast to CARTPOLEVAR, where task variability comes from dense differences in the dynamics, task variability in other environments comes from sparse differences in the reward function and/or dynamics. Therefore, we hypothesize that the PredModel baseline fails on environments with sparse variability across tasks.

# 6 EXPERIMENTS: APPLICATION SCENARIOS

In this section, we evaluate our framework on two application scenarios: performance prediction, and task selection. We conduct this evaluation on MULTIKEYNAV and CARTPOLEVAR, as they cover two distinct sources of task variability, namely reward function and dynamics.

# 6.1 PERFORMANCE PREDICTION

First, we assess the learned task embeddings by using them to predict an agent's performance on a task  $s_{\mathrm{test}} \in S_{\mathrm{init}}$  after observing its performance on a quiz  $S_{\mathrm{quiz}} \subseteq S_{\mathrm{init}}$ . Specifically, we seek to answer the following research question: Would an agent show similar performance on tasks that are close to each other in the learned task embedding space? We begin by creating a benchmark for this application scenario, and then compare our technique against various baselines.

Benchmark. Formally, given the realizations of the task optimality variables of a set of tasks for an agent  $\theta$ , we are interested in predicting the most probable realization of the task optimality variable of a new task for the same agent without observing  $\theta$ . To create benchmarks for this scenario, we generate datasets for quiz sizes ranging from 1 to 20, with 5000 examples for both training and testing. Each example is generated by randomly sampling a quiz  $S_{\mathrm{quiz}}$  of desired size, along with a task  $s_{\mathrm{test}}$  from  $S_{\mathrm{init}}$ , and then recording the performance of an agent  $\theta$ , sampled from the population, on these tasks. Performance prediction techniques are evaluated on this benchmark by measuring prediction ac

![](images/e8b37d490e115cafceefadd492267a5e1b2255ba2935bbba4d5179a595fbdbcb.jpg)  
Figure 6: Results for performance prediction using task embeddings. Our technique (listed as Ours) is competitive with the OPT baseline, which is the best one could do on this benchmark.

curacy on the test examples. The techniques are evaluated on each dataset by partitioning it into 10 folds and reporting the mean prediction accuracy across the folds along with the standard error.

Our approach. Our prediction technique performs soft-nearest neighbor matching of  $s_{\mathrm{test}}$  with  $S_{\mathrm{quiz}}$  in the task embedding space to predict performance on  $s_{\mathrm{test}}$ . Concretely, given the embedding function  $f_{\phi}(.)$ , the prediction is  $\mathbb{1}_{c > 0.5}$ , where  $c$  equals  $\frac{\sum_{s \in S_{\mathrm{quiz}}} o_s \exp(-\beta \|f_{\phi}(s) - f_{\phi}(s_{\mathrm{test}})\|_2^2)}{\sum_{s \in S_{\mathrm{quiz}}} \exp(-\beta \|f_{\phi}(s) - f_{\phi}(s_{\mathrm{test}})\|_2^2)}$ ,  $o_s$  is the realization of the task optimality variable for task  $s$ , and  $\beta$  is a hyperparameter.

Baselines. Besides PredModel, we compare against different levels of oracle knowledge: (i) Random: Randomly predicts the agent's performance. (ii) IgnoreTask: Predicts the agent to succeed on  $s_{\text{test}}$  iff the probability that it succeeds on a random task exceeds 0.5. (iii) IgnoreAgent: Predicts the agent to succeed on  $s_{\text{test}}$  iff the probability that a random agent succeeds on it exceeds 0.5. (iv) OPT: Predicts the agent to succeed on  $s_{\text{test}}$  iff the probability that it succeeds on  $s_{\text{test}}$  exceeds 0.5.

Results. Fig. 6 shows the prediction accuracies of various techniques. Our method is competitive with the  $OPT$  baseline, which provides an upper-bound on the prediction accuracy but relies on the unrealistic assumption of full observability of both the agent and task.

# 6.2 TASK SELECTION

Next, we assess the learned embeddings by using them to select tasks with desired characteristics. Specifically, we seek to answer the following research questions: (i) Does the inner product in the learned task embedding space capture task similarity according to our information-theoretic criterion? (ii) Does the norm of the embedding learned by our framework induce an ordering on the tasks

![](images/d9c7bc18cbc2ac5b30d47d8792bb9e3de126ad9431a1c151528c4367dcfeb306.jpg)  
Figure 7: Results for task selection using task embeddings (dark bars represent Top-3 accuracy and light bars represent Top-1). Our technique (listed as Ours) is competitive with  $\widehat{OPT}_{50}$ . Further, it outperforms  $Ours_{woNorm}$  on Type-2 queries, highlighting the significance of  $\mathcal{C}_{\mathrm{NORM}}$  in our framework.

based on their difficulties? We begin by creating a benchmark for this application scenario, and then compare our technique for task selection using task embeddings against various baselines.

Benchmark. Amongst several options of tasks  $S_{\text{options}}$ , we are interested in choosing the task that best matches the desired characteristics, which we categorize into two query types: Type-1: Select the task that is the most similar to a given reference task  $s_{\text{ref}}$ . The ground-truth answer to this query is arg  $\max_{s \in S_{\text{options}}} \mathcal{I}(\mathcal{O}_{s_{\text{ref}}, \Theta}; \mathcal{O}_{s, \Theta})$ . Type-2: Select the task that is the most similar to (but harder than) a given reference task  $s_{\text{ref}}$ . Out of all the tasks in  $S_{\text{options}}$  that are harder than  $s_{\text{ref}}$ , the ground-truth answer to this query is the task most similar to it. To create benchmarks for this scenario, we generate a dataset of 50 examples. Each example consists of a randomly sampled  $s_{\text{ref}}$  and 10 tasks that form  $S_{\text{options}}$ . Additionally, each benchmark includes 5 easy tasks for reference (determined by ranking a randomly sampled pool of 500 tasks). We evaluate task selection techniques by reporting mean selection accuracy across 4 randomly sampled datasets, along with the standard error.

Our approach. We use task embeddings to rank the options according to similarity and/or difficulty, based on which the selection is made. We additionally compare our technique based on task embeddings learned without  $\mathcal{C}_{\mathrm{NORM}}$  (listed as  $Ours_{woNorm}$ ).

Baselines. Besides PredModel, we compare against the following baselines: (i) Random: Randomly selects answers from  $S_{\text{options}}$ . (ii) StateSim: Measures task similarity based on state representation distances. For queries of type 2, it considers a task  $s_1$  to be harder than  $s_2$  iff the similarity between  $s_1$  and the task most similar to it in the set of easy reference tasks, is less than that for  $s_2$ . (iii) TrajectorySim: Measures task similarity using the edit distance between expert trajectories. (iv) OPT: Estimates task similarity and difficulty using the entire agent population. Given the variance in the estimation process, this is the best one could do on this benchmark. (v)  $\widehat{OPT}_{50}$ : Estimates task similarity and difficulty using a randomly sampled 50% of the population.

Results. Fig. 7 compares different techniques' selection accuracies on the task selection benchmark. Our technique outperforms Random, StateSim, TrajectorySim, and PredModel, and is competitive with  $\widehat{OPT}_{50}$ . This suggests that the inner product in the learned task embedding space successfully captures similarity between tasks. Notably, our technique significantly outperforms  $Ours_{woNorm}$  on Type-2 queries, indicating that the norm of the embedding effectively orders tasks by difficulty.

# 7 CONCLUSION

In this work, we introduced an information-theoretic framework for learning task embeddings in sequential decision-making settings. Through experiments on diverse environments, we empirically demonstrated that the inner product in the embedding space captures task similarity, and the norm of the embedding induces an ordering on the tasks based on their difficulties. A limitation of our current framework is the requirement for tasks to be goal-based, which we plan to address in future work. This could involve using the difference between the cumulative reward obtained during the rollout and the maximum achievable cumulative reward for the given task to parameterize the Bernoulli optimality variable. Additionally, the agent population plays a crucial role in our framework, and it would be interesting to explore more principled methods for construction that explicitly optimize for diversity. Further, empirically estimating the proposed similarity criterion by directly estimating the underlying mass functions could be sample-inefficient for some environments. Therefore, a promising direction is to construct sample-efficient estimators for it. Moreover, evaluation in multi-agent settings, where the task embedding could encode the behavior of non-ego agents, is another interesting direction.

# REFERENCES

Alessandro Achille, Michael Lam, Rahul Tewari, Avinash Ravichandran, Subhransu Maji, Charless C. Fowlkes, Stefano Soatto, and Pietro Perona. Task2Vec: Task Embedding for Meta-Learning. In ICCV, 2019.  
Isac Arnekvist, Danica Kragic, and Johannes Andreas Stork. VPE: Variational Policy Embedding for Transfer Reinforcement Learning. In ICRA, 2018.  
Ali Athar, Sabarinath Mahadevan, Aljosa Osep, Laura Leal-Taixe, and Bastian Leibe. STEm-Seg: Spatio-temporal Embeddings for Instance Segmentation in Videos. In ECCV, 2020.  
Michael Bain and Claude Sammut. A Framework for Behavioural Cloning. In Machine Intelligence, Intelligent Agents, 1995.  
Z. Bing, D. Lerch, K. Huang, and A. Knoll. Meta-Reinforcement Learning in Non-Stationary and Dynamic Environments. In IEEE TPAMI, 2023.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. OpenAI Gym. ArXiv, 2016.  
Rudy Bunel, Matthew J. Hausknecht, Jacob Devlin, Rishabh Singh, and Pushmeet Kohli. Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis. In ICLR, 2018.  
Daniel Matthew Cer, Yinfei Yang, Sheng yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Brian Strope, and Ray Kurzweil. Universal Sentence Encoder for English. In EMNLP, 2018.  
Rati Devidze, Goran Radanovic, Parameswaran Kamalaruban, and Adish Singla. Explicable Reward Design for Reinforcement Learning Agents. In NeurIPS, 2021.  
Haotian Fu, Hongyao Tang, Jianye Hao, Chen Chen, Xidong Feng, Dong Li, and Wulong Liu. Towards Effective Context for Meta-Reinforcement Learning: an Approach based on Contrastive Learning. In AAAI, 2020.  
Hiroki Furuta, Tatsuya Matsushima, Tadashi Kozuno, Yutaka Matsuo, Sergey Levine, Ofir Nachum, and Shixiang Shane Gu. Policy Information Capacity: Information-Theoretic Measure for Task Complexity in Deep Reinforcement Learning. In ICML, 2021.  
Ahana Ghosh, Sebastian Tschiatschek, Sam Devlin, and Adish Singla. Adaptive Scaffolding in Block-based programming via Synthesizing New Tasks as Pop quizzes. In AIED, 2022.  
Abhishek Gupta, Russell Mendonca, YuXuan Liu, Pieter Abbeel, and Sergey Levine. Meta-Reinforcement Learning of Structured Exploration Strategies. In NIPS, 2018.  
Assaf Hallak, Dotan Di Castro, and Shie Mannor. Contextual Markov Decision Processes. ArXiv, 2015.  
Joy He-Yueya and Adish Kumar Singla. Quizzing Policy Using Reinforcement Learning for Inferring the Student Knowledge State. In *EDM*, 2021.  
Max Jaderberg and et al. Human-level Performance in 3D Multiplayer Games with Population-based Reinforcement Learning. In Science, 2019.  
P. Klink, C. D'Eramo, J. Peters, and J. Pajarinen. Self-Paced Deep Reinforcement Learning. In NeurIPS, 2020.  
Lin Lan, Zhenguo Li, Xiaohong Guan, and Pinghui Wang. Meta Reinforcement Learning with Task Embedding and Shared Policy. In *IJCAI*, 2019.  
Lanqing Li, Rui Yang, and Dijun Luo. FOCAL: Efficient Fully-Offline Meta-Reinforcement Learning via Distance Metric Learning and Behavior Regularization. In ICLR, 2021.  
R. Duncan Luce. Individual Choice Behavior: A Theoretical analysis. Wiley, 1959.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed Representations of Words and Phrases and their Compositionality. In NIPS, 2013.  
Jack Parker-Holder, Aldo Pacchiano, Krzysztof M Choromanski, and Stephen J Roberts. Effective Diversity in Population Based Reinforcement Learning. In NeurIPS, 2020.  
Xingchao Peng, Yichen Li, and Kate Saenko. Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation. In ECCV, 2020.  
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. GloVe: Global Vectors for Word Representation. In EMNLP, 2014.  
Rongjun Qin, F. Chen, Tonghan Wang, Lei Yuan, Xiaoran Wu, Zongzhang Zhang, Chongjie Zhang, and Yang Yu. Multi-Agent Policy Transfer via Task Relationship Modeling. ArXiv, 2022.  
Kate Rakelly, Aurick Zhou, Chelsea Finn, Sergey Levine, and Deirdre Quillen. Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables. In ICML, 2019.  
Lukas Schäfer, Filippos Christianos, Amos J. Storkey, and Stefano V. Albrecht. Learning Task Embeddings for Teamwork Adaptation in Multi-Agent Reinforcement Learning. *ArXiv*, 2022.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms. ArXiv, 2017.  
Shagun Sodhani, Ludovic Denoyer, Pierre-Alexandre Kamienny, and Olivier Delalleau. MTEnv - Environment interface for multit-task reinforcement learning. GitHub, 2021a. URL https://github.com/facebookresearch/mtenv.  
Shagun Sodhani, Amy Zhang, and Joelle Pineau. Multi-Task Reinforcement Learning with Context-based Representations. In ICML, 2021b.  
Yi Sun, Xiaogang Wang, and Xiaou Tang. Deep Learning Face Representation from Predicting 10,000 Classes. In CVPR, 2014.  
Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip H. S. Torr, and Timothy M. Hospedales. Learning to Compare: Relation Network for Few-Shot Learning. In CVPR, 2018.  
Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy Gradient Methods for Reinforcement Learning with Function Approximation. In NIPS, 1999.  
Paul Tylkin, Goran Radanovic, and David C. Parkes. Learning Robust Helpful Behaviors in Two-Player Cooperative Atari Environments. In AAMAS, 2021.  
Georgios Tzannetos, Barbara Gomes Ribeiro, Parameswaran Kamalaruban, and Adish Singla. Proximal Curriculum for Reinforcement Learning Agents. In TMLR, 2023.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing Data using t-SNE. In JMLR, 2008.  
Oriol Vinyals and et al. Grandmaster Level in StarCraft II Using Multi-agent Reinforcement Learning. In Nature, 2019.  
Risto Vuorio, Shao-Hua Sun, Hexiang Hu, and Joseph J. Lim. Multimodal Model-Agnostic Meta-Learning via Task-Aware Modulation. In NeurIPS, 2019.  
Homer Walke, Jonathan Yang, Albert Yu, Aviral Kumar, Jedrzej Orbik, Avi Singh, and Sergey Levine. Don't Start From Scratch: Leveraging Prior Data to Automate Robotic Reinforcement Learning. In CoRL, 2022.  
Minjong Yoo, Sangwoo Cho, and Honguk Woo. Skills Regularized Task Decomposition for Multi-task Offline Reinforcement Learning. In NeurIPS, 2022.  
Amy Zhang, Rowan Thomas McAllister, Roberto Calandra, Yarin Gal, and Sergey Levine. Learning Invariant Representations for Reinforcement Learning without Reconstruction. In ICLR, 2021.  
Yu Zhang, Ying Wei, and Qiang Yang. Learning to Multitask. In NeurIPS, 2018.
