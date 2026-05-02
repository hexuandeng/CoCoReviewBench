# EduQate: Generating Adaptive Curricula through RMABs in Education Settings

Anonymous Author(s)

Affiliation

Address

email

# Abstract

There has been significant interest in the development of personalized and adaptive educational tools that cater to a student's individual learning progress. A crucial aspect in developing such tools is in exploring how mastery can be achieved across a diverse yet related range of content in an efficient manner. While Reinforcement Learning and Multi-armed Bandits have shown promise in educational settings, existing works often assume the independence of learning content, neglecting the prevalent interdependencies between such content. In response, we introduce Education Network Restless Multi-armed Bandits (EdNetRMABs), utilizing a network to represent the relationships between interdependent arms. Subsequently, we propose EduQate, a method employing interdependency-aware Q-learning to make informed decisions on arm selection at each time step. We establish the optimality guarantee of EduQate and demonstrate its efficacy compared to baseline policies, using students modeled from both synthetic and real-world data.

# 1 Introduction

The COVID-19 pandemic has accelerated the adoption of educational technologies, especially on eLearning platforms. Despite abundant data and advancements in modeling student learning, effectively capturing the learning process with interdependent content remains a significant challenge [9]. The conventional rules-based approach to creating personalized learning curricula is impractical due to its labor-intensive nature and need for expert knowledge. Machine learning-based systems offer a scalable alternative, automatically generating personalized content to optimize learning [22, 24].

One possible approach to model the learning process is the Restless Multi-Armed Bandits (RMAB, [26]), where a teacher agent selects a subset of arms (concepts) to teach each round. However, RMAB's assumption that arms are independent is unrealistic in educational settings. For example, solving a math question on the area of a triangle requires knowledge of algebra, arithmetic, and geometry. Practicing this question should enhance proficiency in all three areas. Models that ignore such interdependencies may inaccurately predict knowledge levels by assuming each exercise impacts only a single area.

In response to this challenge, we introduce an interdependency-aware RMAB model to the education setting. We posit that by acknowledging and modeling the learning dynamics of interdependent content, both teachers and algorithms can strategically leverage overlapping utility to foster mastery over a broader range of topics within a curriculum. We advocate for RMABs as a fitting model for this context, as the inherent dynamics of such a model align closely with the learning process.

In this study, our objective is to derive a teacher policy that effectively recommends educational content to students, accounting for interdependencies among the content to enhance overall utility (that characterizes understanding and retention of content). Our contributions are as follows:

1. We introduce Restless Multi-armed Bandits for Education (EdNetRMABs), enabling the modeling of learning processes with interdependent educational content.  
2. We propose EduQate, a Whittle index-based heuristic algorithm that uses Q-learning to compute an inter-dependency-aware teacher policy. Unlike previous methods, EduQate does not require knowledge of the transition matrix to compute an optimal policy.  
3. We provide a theoretical analysis of EduQate, demonstrating guarantees of optimality.  
4. We present empirical results on simulated students and real-world datasets, showing the effectiveness of EduQate over other teacher policies.

# 2 Related Work and Preliminaries

# 2.1 Restless Multi-Armed Bandits

The selection of the right time and manner for limited interventions is a problem of great practical importance across various domains, including health intervention [17, 5], anti-poaching operations [20], education [13, 6, 2], etc. These problems share a common characteristic of having multiple arms in a Multi-armed Bandit (MAB) problem, representing entities such as patients, regions of a forest, or students' mastery of concepts. These arms evolve in an uncertain manner, and interventions are required to guide them from "bad" states to "good" states. The inherent challenge lies in the limited number of interventions, dictated by the limited resources (e.g., public health workers, the number of student interactions). RMAB, a generalization of MAB, offers an ideal model for representing the aforementioned problems of interest. RMAB allows non-active bandits to also undergo the Markovian state transition, effectively capturing uncertainty in arm state transitions (reflecting uncertain state evolution), actions (representing interventions), and budget constraints (illustrating limited resources).

RMABs and the associated Markov Decision Processes (MDP) for each arm offer a valuable model for representing the learning process. Firstly, leveraging the MDPs associated with each arm provides the flexibility to adopt nuanced modeling of learning content, accommodating different learning curves for various content based on students' strengths and weaknesses. Secondly, the transition probabilities serve as a useful mechanism to model forgetting (through state decay due to passivity or negligence) and learning (through state transitions to the positive state from repeated practice). Considering these aspects, RMABs prove to be a beneficial framework for personalizing and generating adaptive curricula across a diverse range of students.

In general, computing the optimal policy for a given set of restless arms in RMABs is recognized as a PSPACE-hard problem [18]. The Whittle index [26] provides an approach with a tractable solution that is provably optimal, especially when each arm is indexable. However, proving indexability can be challenging and often requires specification of the problem's structure, such as the optimality of threshold policies [17, 16]. Moreover, much of the research on Whittle Index policies has focused on two-action settings or requires prior knowledge of the transition matrix of the RMABs. Meeting these conditions proves challenging in the educational context, where diverse students interact with educational systems, each possessing different prior knowledge and distinct learning curves for various topics.

WIQL [5], on the other hand, employs a Q-learning-based method to estimate the Whittle Index and has demonstrated provable optimality without requiring prior knowledge of the transition matrix. We utilize WIQL as a baseline method in our subsequent experiments.

In a recent investigation by [12], RMABs were explored within a network framework, requiring the agent to manage a budget while allocating a high-cost, high-benefit resource to one arm to "unlock" potential lower-cost, intermediate-benefit resources for the arm's neighbors. The network effects emphasized in their work are triggered by an intentional, active action, enabling the agent to choose to propagate positive externalities to a selected arm's neighbors within budget constraints. In contrast, our study delves into scenarios where network effects are indirect results of an active action, and the agent lacks direct control over such effects. Thus, the challenge lies in accurately modeling these network effects and leveraging them when beneficial.

# 2.2 Reinforcement Learning in Education

In the realm of education, numerous researchers have explored optimizing the sequencing of instructional activities and content, assuming that optimal sequencing can significantly impact student learning. RL is a natural approach for making sequential decisions under uncertainty [1]. While RL has seen success in various educational applications, effectively sequencing interdependent content in a personalized and adaptive manner has yielded mixed or insignificant results compared to baseline teacher policies [11, 21, 8]. In general, these RL works focus on data-driven methods using student activity logs to estimate students' knowledge states and progress, assuming that the interdependencies between learning content are encapsulated in students' learning histories [9, 3, 19]. In contrast, our work focuses on modelling these interdependencies directly.

Of particular relevance are factored MDPs applied to skill acquisition introduced by [11]. While factored MDPs account for interdependencies amongst skills, decentralized policy learning is infeasible as policies must consider the joint state space. Our work leverages the advantage of decentralized policy learning provided by RMABs and introduces a novel decentralized learning approach that exploits interdependencies between arms.

Complementary to RL methods in education is the utilization of knowledge graphs to uncover relationships between learning content [9]. Existing research primarily focuses on establishing these relationships through data-driven methods (e.g. [7, 23]) often leveraging student-activity logs. In this work, we complement such research by presenting an approach where bandit methods can effectively operate with knowledge graphs derived by such methods.

# 3 Model

In this section, we introduce the Restless Multi-Armed Bandits for Education (EdNetRMABs). It is important to note that while we specifically apply EdNetRMABs to the education setting, the framework can be seamlessly translated to other scenarios where modeling the effects of active actions within a network is critical. For ease of access, a table of notations is provided in Table 2.

In education, a teacher recommends learning content, or items, to maximize student education, often with content from online platforms. Items are grouped by topics, such as "Geometry," where exposure to one piece of content can enhance knowledge across others in the same group. This cumulative learning effect which we refer to as "network effects", implies that exposure to an item is likely to positively impact the student's success on items within the same group. A successful teacher accurately estimates a student's knowledge state over repeated interactions, leveraging these network effects to promote both breadth and depth of understanding through recommendations.

# 3.1 EdNetRMABs

The RMAB model tasks an agent with selecting  $k$  arms from  $N$  arms, constrained by a limit on the number of arms that can be pulled at each time step. The objective is to find a policy that maximizes the total expected discounted reward, assuming that the state of each arm evolves independently according to an underlying MDP.

The EdNetRMABs model extends RMABs by allowing for active actions to propagate to other arms dependent on the current arm when it is being pulled, thus relaxing the assumption of independent arms. This is operationalized by organising the arms in a network, and pulling of an arm results in changes for its neighbors, or members in the same group.

When applied to education setting, the EdNetRMABs is formalized as follows:

Arms Each arm, denoted as  $i \in 1, \dots, N$ , signifies an item. In the context of this networked environment, each arm belongs to a group  $\phi \in \{1, \dots, L\}$  representing the overarching topic that encompasses related items. It's important to note that arm membership is not mutually exclusive, allowing arms to be part of multiple groups. This flexibility enables a more nuanced modeling of interdependencies among educational content. For instance, a question involving the calculation of the area of a triangle may span both arithmetic and geometry groups.

State space In this framework, each arm possesses a binary latent state, denoted as  $s_i \in \{0,1\}$ , where "0" represents an "unlearned" state, and "1" indicates a "learned" state. Considering all arms collectively, these states serve as a representation of the student's overall knowledge state. In the current work, it is assumed that the states of all arms are fully observable, providing a comprehensive model of the student's understanding of the various educational concepts.

Action space To capture the network effects associated with arm pulls, we depart from the conventional RMAB framework with a binary action space  $A = \{0,1\}$  by introducing a pseudo-action. In this modified setup, the action space is extended to  $A = \{0,1,2\}$ , where actions 0 and 2 represent "no-pull" and "pull", as commonly used in bandit literature. Notably, in EdNetRMABs, a third action 1 is introduced to simulate the network effects resulting from pulling another arm within the same group. It is important to clarify that agents do not directly engage with action 1 but we employ it solely for modeling network effects, hence the term "pseudo-action".

Transition function For a given arm  $i$ , let  $P_{s,s'}^{a,i}$  represent the probability of the arm transitioning from state  $s$  to  $s'$  under action  $a$ . It's noteworthy that, in typical real-world educational settings, the actual transition functions governing the states of the arms are often unknown and, even for the same concept, may vary among students due to differences in prior knowledge. To address this challenge, we adopt model-free approaches in this study, devising methods to compute the teacher policy without relying on explicit knowledge of these transition functions. In the following experiments, we maintain the assumption of non-zero transition probabilities, and enforce constraints that are aligned with the current domain [17]: (i) The arms are more likely to stay in the positive state than change to the negative state:  $P_{0,1}^{0} < P_{1,1}^{0}$ ,  $P_{0,1}^{1} < P_{1,1}^{1}$  and  $P_{0,1}^{2} < P_{1,1}^{2}$ ; (ii) The arm tends to improve the latent state if more efforts is spent on that arm, i.e., it is active or semi-active:  $P_{0,1}^{0} < P_{0,1}^{1} < P_{0,1}^{2}$  and  $P_{1,1}^{0} < P_{1,1}^{1} < P_{1,1}^{2}$ .

With the formalization of the EdNetRMABs model provided, we now apply it to an educational context. In this scenario, the agent assumes the role of a teacher and takes actions during each time step  $t \in \{1, \dots, T\}$ . Specifically, at each time step, the teacher recommends an item for the student to study. We represent the vector of actions taken by the teacher at time step  $t$  as  $\mathbf{a}^t \in \{0, 1, 2\}^N$ . Here, arm  $i$  is considered to be active at time  $t$  if  $a^t(i) = 2$  and passive when  $a^t(i) = 0$ . When arm  $i$  is pulled, the set of arms that share the same group membership as arm  $i$ , denoted as  $\phi_i^-$  under goes the pseudo-action, represented as  $a^t(j) = 1$  for all  $j \in \phi^-$ . In our framework, the teacher agent acts on exactly one arm per time step to simulate the real-world constraint that the teacher can only recommend one concept to students ( $\sum_{i} I_{a^t(i) = 2} = 1, \forall t$ ). Subsequent to taking action, the teacher receives  $\mathbf{s}^t \in \{0, 1\}^N$ , a vector reflecting the state of all arms, and reward  $r_t = \sum_{i=1}^N s^t(i)$ . The vector  $\mathbf{s}^t$  represents the overall knowledge state of the student. The teacher agent's goal, therefore, is to maximize the long term rewards, either discounted or averaged.

# 4 EduQate

Q-learning [25] is a popular reinforcement learning method that enables an agent to learn optimal actions in an environment by iteratively updating its estimate of state-action value,  $Q(s, a)$ , based on the rewards it receives. At each time step  $t$ , the agent takes an action  $a$  using its current estimate of  $Q$  values and current state  $s$ , thus received a reward of  $r(s)$  and new state  $s'$ . We provide an abridged introduction to Q-learning in the Appendix F.

Expanding upon Q-learning, we introduce EduQate, a tailored Q-learning approach designed for learning Whittle-index policies in EdNetRMABs. In the interaction with the environment, the agent chooses a single item, represented by arm  $i$ , to recommend to the student. In this context, the agent possesses knowledge of the group membership  $\phi_i$  of the selected arm and observes the rewards generated by activating arm  $i$  and semi-activating arms in  $\phi_i^-$ . EduQate utilizes this interaction to learn the Q-values for all arms and actions.

To adapt Q-learning to EdNetRMABs, we propose leveraging the learned Q-values to select the arm with the highest estimate of the Whittle index, defined as:

Algorithm 1 Q-Learning for EdNetRMABs (EduQate)  
Input: Number of arms  $N$    
Initialize  $Q_{i}(s,a)\gets 0$  and  $\lambda_i(s)\gets 0$  for each state  $s\in S$  and each action  $a\in \{0,1,2\}$  , for each arm  $i\in 1,\dots,N$    
Initialize replay buffer  $D$  with capacity  $C$    
for  $t$  in  $1,\ldots ,T$  do  $\epsilon \leftarrow \frac{N}{N + t}$  With probability  $\epsilon$  , select one arm uniformly at random. Otherwise, select arm with highest Whittle Index,  $i = \arg \max_{i}\lambda_{i}$  .   
for arm  $n$  in  $1,\ldots ,N$  do if  $n\neq i$  then Set arm  $n$  to passive,  $a_{n}^{t} = 0$  else Set arm  $n$  to active,  $a_{n}^{t} = 2$  for  $j\in \phi_i^-$  do Set arms in same group as  $i$  to semi-active,  $a_j^t = 1$  end for end if   
end for   
Execute actions  $\mathbf{a}^{\mathrm{t}}$  and observe reward  $r^t$  and next state  $s^{t + 1}$  for all arms Store experience  $(s^t,\mathbf{a}^{\mathrm{t}},\mathbf{r}^{\mathrm{t}},\mathbf{s}^{\mathrm{t + 1}})$  in replay buffer  $D$  Sample minibatch  $B$  of Experience from replay buffer  $D$  .   
for Experience in minibatch  $B$  do Update  $Q_{n}(s,a)$  using Q-learning update in Equation 11. Compute  $\lambda_{n}$  using Equation 1   
end for   
end for

$$
\lambda_ {i} = Q \left(s _ {i}, a _ {i} = 2\right) - Q \left(s _ {i}, a _ {i} = 0\right) + \sum_ {j \in \phi_ {i} ^ {-}} \left(Q \left(s _ {j}, a _ {j} = 1\right) - Q \left(s _ {j}, a _ {j} = 0\right)\right) \tag {1}
$$

Here,  $\lambda_{i}$  is the Whittle Index estimate for arm  $i$ . In essence, the Whittle Index of arm  $i$  is computed as the linear combination of the value associated with taking action on arm  $i$  over passivity and the value of associated with semi-actively engaging with members from same group, compared to passivity.

To improve the convergence of Q-learning, we incorporate Experience Replay [15]. This involves saving the teacher algorithm's previous experiences in a replay buffer and drawing mini-batches of samples from this buffer during updates to enhance convergence. In Section 4.1, we prove that EduQate will converge to the optimal policy. However, in practice, we may not have enough episodes to fully train EduQate. Therefore, we propose Experience Replay to mitigate the cold-start problem common in RL applications, a common problem where initial student interactions with sub-optimal teachers can lead to poor learning experiences [3].

The pseudo-code is provided in Algorithm 1. Similar to WIQL [5], we employ a  $\epsilon$ -decay policy that facilitates exploration and learning in the early steps, and proceeds to exploit the learned Q-values in later stages.

# 4.1 Analysis of EduQate

In this section, we analyze EduQate closely, and show that EduQate does not alter the optimality guarantees of Q-learning under the constraint that  $k = 1$  (Theorem 1). Our method relies on the assumption that teachers are limited to assign 1 item to the student at each time step. Theorem 2 analyzes EduQate under the conditions that  $k > 1$ . Since our setting involves the semi-active actions, we should compute Equation 1. To reiterate,  $\phi_{i}$  here refers to the group that arm  $i$  belongs to, and  $\phi_{i}^{-}$  is the same group but does not include arm  $i$ . If arm  $i$  is selected, then all the remaining arms in group  $\phi_{i}^{-}$  should be semi-active.

Theorem 1 Choosing the top arm with the largest  $\lambda$  value in Equation 1 is equivalent to maximizing the cumulative long-term reward.

Proof. According to the approach, we select the arm according to the  $\lambda$  value. Assume arm  $i$  has the highest  $\lambda$  value, then for any arm  $j$  where  $j \neq i$ , we have

$$
\lambda_ {i} \geq \lambda_ {j} \tag {2}
$$

According to the definition of  $\lambda$  in Equation 1, we move the negative part to the other side, and the left side becomes:

$$
Q \left(s _ {i}, a _ {i} = 1\right) + \sum_ {i \in \phi_ {i} ^ {-}} \left(Q \left(s _ {i}, a _ {i} = 1\right)\right) + Q \left(s _ {j}, a _ {j} = 0\right) + \sum_ {j \in \phi_ {j} ^ {-}} \left(Q \left(s _ {j}, a _ {j} = 0\right)\right)
$$

and the right side is similar. There are three cases:

- arm  $i$  and arm  $j$  are not connected, and group  $\phi_i$  and  $\phi_j$  has no overlap, i.e.,  $\phi_i \cap \phi_j = \emptyset$ . We add  $\sum_{z \notin \phi_i \wedge z \notin \phi_j} Q(s_z, a_z = 0)$  on both sides. This denotes the addition of  $Q(s_z, a_z = 0)$  for all arm  $z$  that are not included in the set of  $\phi_i$  or  $\phi_j$ . We have the left side:

$$
\begin{array}{l} Q \left(s _ {i}, a _ {i} = 1\right) + \sum_ {i \in \phi_ {i} ^ {-}} \left(Q \left(s _ {i}, a _ {i} = 1\right)\right) + Q \left(s _ {j}, a _ {j} = 0\right) + \sum_ {j \in \phi_ {j} ^ {-}} \left(Q \left(s _ {j}, a _ {j} = 0\right)\right) + \sum_ {z \notin \phi_ {i} \wedge z \notin \phi_ {j}} Q \left(s _ {z}, a _ {z} = 0\right) \\ = Q \left(s _ {i}, a _ {i} = 1\right) + \sum_ {i \in \phi_ {i} ^ {-}} \left(Q \left(s _ {i}, a _ {i} = 1\right)\right) + \sum_ {j \notin \phi_ {i}} \left(Q \left(s _ {j}, a _ {j} = 0\right)\right) \\ = Q (\mathbf {s}, \mathbf {a} = \mathbb {I} _ {i}) \tag {3} \\ \end{array}
$$

Similarly, we do the same for the right side and thus, the equation 2 becomes

$$
Q (\mathbf {s}, \mathbf {a} = \mathbb {I} _ {i}) \geq Q (\mathbf {s}, \mathbf {a} = \mathbb {I} _ {j})
$$

- arm  $i$  and arm  $j$  are not connected, but group  $\phi_i$  and  $\phi_j$  has overlap, i.e.,  $\phi_i \cap \phi_j \neq \emptyset$ . In this case, we add  $\sum_{z \notin \phi_i \wedge z \notin \phi_j} Q(s_z, a_z = 0) - \sum_{z \in \phi_i \cap \phi_j} Q(s_z, a_z = 0)$  on both sides.  
- arm  $i$  and arm  $j$  are connected, and group  $\phi_i$  and  $\phi_j$  has overlap, i.e.,  $\phi_i \cap \phi_j \neq \emptyset$ , and  $\{i,j\} \subset \phi_i \cap \phi_j$ . This case is similar to the previous one, we add  $\sum_{z \notin \phi_i \wedge z \notin \phi_j} Q(s_z, a_z = 0) - \sum_{z \in \phi_i \cap \phi_j} Q(s_z, a_z = 0)$  on both sides.

The detailed proof is provided in Appendix B.

Thus when  $k = 1$ , selecting the top arm according to the  $\lambda$  value is equivalent to maximizing the cumulative long-term reward, and is guaranteed to be optimal.

Theorem 2 When  $k > 1$ , selecting the  $k$  arms is a NP-hard problem. The non-asymptotic tight upper bound and non-asymptotic tight lower bound for getting the optimal solution are  $o(C(n, k))$  and  $\omega(N)$ , respectively.

Proof Sketch. This problem can be considered as a variant of the knapsack problem. If we disregard the influence of the shared neighbor nodes for two selected arms, then selecting arm  $i$  will not influence the future selection of arm  $j$ . In such instances, the problem of selecting the  $k$  arms is simplified to the traditional 0/1 knapsack problem, a classic NP-hard problem. Therefore, when considering the effect of shared neighbor nodes for two selected arms, this problem is at least as challenging as the 0/1 knapsack problem.

When  $k > 1$ , it is difficult to compute the optimal solution, we provide a heuristic greedy algorithm with the complexity of  $O\left(\frac{(2N - k)*k}{2}\right)$  in Section C in the appendix.

# 5 Experiment

In this section, we demonstrate the effectiveness of EduQate against benchmark algorithms on synthetic students and students derived from a real-world dataset, the Junyi Dataset and the OLI Statics dataset. All experiments are run on CPU only. In our experiments, we compare EduQate with the following policies:

![](images/5d7c8aec38cc109ed0a0f41124bc5b1a9eb6ea1b97efd6ff62a8711f5e3e012d.jpg)

![](images/0a6844dc7771565551cbd3fbe3d90180e658e1104e254ced3845e817e13507a3.jpg)  
Figure 1: Average rewards for the respective algorithms on 3 datasets, averaged across 30 runs. Shaded regions represent standard error.

![](images/7033f2e94685e66c7ede7dd4073a23a04925f65d2883a00c5ab5f2857aebedbf.jpg)

![](images/b8667bb17aff42b1fbc606db38b3e7bc35c171fc74da70ed252b0e5edf8c430f.jpg)

- Threshold Whittle (TW): This algorithm, proposed by [17], utilizes an efficient closed-form approach to compute the Whittle index, considering only the pull action as active. It operates under the assumption that transition probabilities are known and stands as the state-of-the-art in RMABs.  
- WIQL: This algorithm employs a Q-learning-based Whittle Index approach [5]. It learns Q-values using the pull action as the only active strategy and calculates the Whittle Index based on the acquired Q-values.  
- Myopic: This strategy disregards the impact of the current action on future rewards, concentrating solely on predicted immediate rewards. It selects the arm that maximizes the expected reward at the immediate time step.  
- Random: This strategy randomly selects arms with uniform probability, irrespective of the underlying state.

Inspired by work in healthcare settings [12, 14], we compare the policies by the Intervention Benefit (IB), as shown in the following equation:

$$
I B _ {R a n d o m, E Q} (\pi) = \frac {\mathbb {E} _ {\pi} (R (.) - \mathbb {E} _ {R a n d o m} (R (.))}{\mathbb {E} _ {E Q} (R (.) - \mathbb {E} _ {R a n d o m} (R (.)}} \tag {4}
$$

where  $EQ$  represents EduQate, and  $Random$  represents a policy where the arms are selected at random. Prior work in educational settings has demonstrated that random policies can yield robust learning outcomes through spaced repetition [9, 10]. Therefore, to establish efficacy, successful algorithms must demonstrate superiority over random policies. Our chosen metric,  $IB$ , effectively compares the extent to which a challenger algorithm  $\pi$  outperforms a random policy in comparison to our algorithm.

# 5.1 Experiment setup

In all experiments, we commence by initializing all arms in state 0 and permit the teacher algorithms to engage with the student for a total of 50 actions, pulling exactly 1 arm (i.e.  $k = 1$ ) at each time step. Following the completion of these actions, the episode concludes, and the student state is reset. This process is iterated across 800 episodes, for a total of 30 seeds. The datasets used in our experiment are described below:

Synthetic dataset. Given the domain-motivated constraints on the transition functions highlighted in Section 3.1, we create a simulator based on  $N = 50$ ,  $S \in \{0,1\}$ ,  $N_{\mathrm{topics}} = 20$ . We randomly assign arms to topic groups, and allow arms to be assigned to be more than one topic. Under this method, number of arms under each group may not be equal. For each trial, a new transition matrix is generated to simulate distinct student scenarios.

Junyi dataset. The Junyi dataset [7] is an extensive dataset collected from the Junyi Academy  $^{1}$ , an eLearning platform established in 2012 on the basis of the open-source code released by Khan

Table 1: Comparison of policies on synthetic, Junyi, and OLI datasets.  $\mathbb{E}[R]$  represents the average reward obtained in the final episode of training. Statistic after  $\pm$  represents standard error across 30 trials.  

<table><tr><td rowspan="2">Policy</td><td colspan="2">Synthetic</td><td colspan="2">Junyi</td><td colspan="2">OLI</td></tr><tr><td>E[IB](%)±</td><td>E[R]±</td><td>E[IB](%)±</td><td>E[R]±</td><td>E[IB](%)±</td><td>E[R]±</td></tr><tr><td>Random</td><td>-</td><td>26.84 ± 0.46</td><td>-</td><td>15.82 ± 0.34</td><td>-</td><td>18.46 ± 0.35</td></tr><tr><td>WIQL</td><td>-49.03 ± 15.07</td><td>24.60 ± 0.43</td><td>-26.77 ± 7.39</td><td>14.01 ± 0.97</td><td>-60.20 ± 19.38</td><td>14.33 ± 0.42</td></tr><tr><td>Myopic</td><td>-3.44 ± 5.81</td><td>27.07 ± 0.52</td><td>10.74 ± 3.13</td><td>16.86 ± 0.356</td><td>39.92 ± 12.00</td><td>20.51 ± 0.48</td></tr><tr><td>TW</td><td>37.21 ± 17.02</td><td>28.50 ± 0.47</td><td>31.284 ± 2.65</td><td>15.819 ± 0.34</td><td>0.20 ± 9.27</td><td>18.07 ± 0.21</td></tr><tr><td>EduQate</td><td>100.0</td><td>34.33 ± 0.49</td><td>100.0</td><td>24.53 ± 0.31</td><td>100.0</td><td>25.47 ± 0.47</td></tr></table>

Academy. In this dataset, there are nearly 26 million student-exercise interactions across 250 000 students in its mathematics curriculum. For this experiment, we selected the top 100 exercises with the most student interactions to create our student models. Using our method to generate groups, the resultant EdNetRMAB has  $N = 100$  and  $N_{\text{topics}} = 21$ .

OLI Statics dataset. The OLI Statics dataset [4] comprises student interactions with an online Engineering Statics course<sup>2</sup>. In this dataset, each item is assigned one or more Knowledge Components (KCs) based on the related topics. After filtering for the top 100 items with the most student interactions, the resultant EdNetRMAB includes  $N = 100$  items and  $N_{\text{topics}} = 76$  distinct topics.

# 5.2 Creating student models

In this section, we outline the procedure for generating student models aimed at simulating the learning process. To clarify, a student model in this context is defined as a set of transition matrices for all items. These matrices are employed with EdNetRMABs to simulate the learning dynamics.

We employ various strategies to model transitions within the RMAB framework. Active transitions are determined by assessing the average success rate on a question before and after a learning intervention. Passive transitions are influenced by difficulty ratings, with more challenging questions more prone to rapid forgetting. Semi-active transitions, on the other hand, are computed as proportion of active transition, guided by similarity scores. Here, we provide an outline and the full details can be found in Appendix D.

Active Transitions. We use data on students' correct response rate after interacting with an item to create the transition matrix for action 2, based on the change in correctness rates before and after a learning intervention.

Passive Transitions. To construct passive transitions for items, we use relative difficulty scores to determine transitions based on difficulty levels. We assume that higher difficulty correlates with a greater likelihood of forgetting, resulting in higher failure rates. Specifically, higher difficulty values correspond to higher  $P_{1,0}^{0}$  values, indicating a greater likelihood of forgetting. The transition matrix for the passive action  $a = 0$  is then randomly generated, with values influenced by difficulty levels.

Semi-active Transitions. To derive semi-active transitions, we use similarity scores between exercises from the Junyi dataset. We first normalize these scores to the range  $[0, 1]$ . Then, for any chosen arm, we compute its transition matrix under the semi-active action  $a = 1$  as a proportion of its active action transitions,  $P_{0,1}^{1} = \sigma(P_{0,1}^{2})$ , where  $\sigma$  signifies the similarity proportion.

The arm's transition matrix for the semi-active action varies due to different similarity scores between pairs in the same group. To address this, we use the average similarity score to determine the proportion. Since the OLI dataset does not contain similarity ratings, we assume a constant similarity rating of  $\sigma = 0.8$  for all pairs.

# 6 Results

The experimental results for the synthetic, Junyi, and OLI datasets are shown in Table 1. We report the average intervention benefit  $IB$  and final episode rewards from thirty independent runs for five algorithms: EduQate, TW, WIQL, Myopic, and Random. EduQate consistently outperforms the other policies across all datasets, demonstrating higher intervention benefits and average rewards.

![](images/70fc0fec73d6efa632b5af7f5f388fe070c4add769a5a854657c7cc1f8a583c9.jpg)  
Synthetic Network  $N = 100$ $N_{\text{topics}} = 20$

![](images/0946e71f8ac7378ec1b222ad2986c1b7045dd8c701742b72b74430c5d2f2499e.jpg)  
Junyi network, abridged to  $N_{topics} = 7$  for brevity.

![](images/694ce0eb5f7311b143a4eb821644f54e45839b8444234ee72e76cd10a1650af4.jpg)  
Figure 2: This visualization compares network complexities from our experiments. The synthetic dataset (left) shows simpler, isolated groups, while the real-world datasets (Junyi, middle; OLI,right) displays more intricate and interconnected relationships amongst items.  
OLI network,  $N = 100$ ,  $N_{\text{topics}} = 76$ .

In terms of  $IB$ , we note that all challenger policies do not exceed  $50\%$ , indicating two key points. First, as noted in prior works [9], our results confirm that random policies in educational settings are robust and difficult to surpass, even when algorithms are equipped with knowledge of the learning dynamics. Second, our interdependency-aware EduQate performs well over random policies and other algorithms, highlighting the importance of considering network effects and interdependencies in EdNetRMABs.

Notably, WIQL, which relies solely on Q-learning for active and passive actions, performs worse than a random policy, likely due to misattributing positive network effects to passive actions. Despite having access to the transition matrix, TW does not perform as well as the interdependency-aware EduQate. While it has demonstrated effectiveness in traditional RMABs, TW weaknesses become evident in the current setting, where pulling an arm has wider implications to other arms. Overall, EduQate has demonstrated robust and effective performance in maximizing rewards across different datasets. Figure 1 shows the average rewards obtained in the final episode for each algorithm.

Figure 2 provides visualizations of the networks generated from synthetic students and mined from real-world datasets. The synthetic dataset produces networks with distinct isolated groups, contrasting with the more intricate and interconnected networks from the Junyi and OLI datasets, reflecting real-world complexities. Despite these differing topologies and levels of interdependency, EduQate performs well under all network setups. In Appendix E.1, we explore the effects of different network topologies by varying the number of topics while limiting the membership of each item. We find that as network interdependencies are reduced, the network effects diminish, and such EdNetRMABs can be approximated to traditional RMABs with independent arms. Under these conditions, our algorithm does not perform as well as other baseline policies.

Finally, an ablation study detailed in Appendix E.2 examines the effectiveness of the replay buffer in EduQate. The study shows that the replay buffer helps overcome the cold-start problem, where initial learning episodes provide sub-optimal experiences for students [3].

# 7 Conclusion and Limitations

In this paper, we introduced EdNetRMABs to the education setting, a variant of MAB designed to model interdependencies in educational content. We also proposed EduQate, a novel Whittle-based learning algorithm tailored for EdNetRMABs. Unlike other Whittle-based algorithms, EduQate computes an optimal policy without requiring knowledge of the transition matrix, while still accounting for the network effects of pulling an arm. We demonstrated the guaranteed optimality of a policy trained under EduQate and showcased its effectiveness on synthetic and real-world datasets, each with its own characteristic.

Our work assumes that student knowledge states are fully observable and available at all times, which is a limitation. Despite this, we believe our work is significant and can inspire further research to improve efficiencies in education. For future work, we aim to extend EduQate to handle partially observable states and address the cold-start problem in education systems by minimizing the initial exploratory phase.

# References

[1] Richard C Atkinson. Ingredients for a theory of instruction. American Psychologist, 27(10): 921, 1972.  
[2] Aqil Zainal Azhar, Avi Segal, and Kobi Gal. Optimizing representations and policies for question sequencing using reinforcement learning. International Educational Data Mining Society, 2022.  
[3] Jonathan Bassen, Bharathan Balaji, Michael Schaarschmidt, Candace Thille, Jay Painter, Dawn Zimmaro, Alex Games, Ethan Fast, and John C Mitchell. Reinforcement learning for the adaptive scheduling of educational activities. In Proceedings of the 2020 CHI conference on human factors in computing systems, pages 1-12, 2020.  
[4] Norman Bier. Oli engineering statics - fall 2011 (114 students), 2011. URL https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=590.  
[5] Arpita Biswas, Gaurav Aggarwal, Pradeep Varakantham, and Milind Tambe. Learn to intervene: An adaptive learning policy for restless bandits in application to preventive healthcare. arXiv preprint arXiv:2105.07965, 2021.  
[6] Colton Botta, Avi Segal, and Kobi Gal. Sequencing educational content using diversity aware bandits. 2023.  
[7] Haw-Shiuan Chang, Hwai-Jung Hsu, and Kuan-Ta Chen. Modeling exercise relationships in e-learning: A unified approach. In *EDM*, pages 532-535, 2015.  
[8] Shayan Doroudi, Vincent Aleven, and Emma Brunskill. Robust evaluation matrix: Towards a more principled offline exploration of instructional policies. In Proceedings of the fourth (2017) ACM conference on learning@ scale, pages 3-12, 2017.  
[9] Shayan Doroudi, Vincent Aleven, and Emma Brunskill. Where's the reward? a review of reinforcement learning for instructional sequencing. International Journal of Artificial Intelligence in Education, 29:568-620, 2019.  
[10] Hermann Ebbinghaus. Über das gedächtnis: untersuchungen zur experimentellen psychologie. Duncker & Humblot, 1885.  
[11] Derek Green, Thomas Walsh, Paul Cohen, and Yu-Han Chang. Learning a skill-teaching curriculum with dynamic bayes nets. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 25, pages 1648-1654, 2011.  
[12] Christine Herlihy and John P. Dickerson. Networked restless bandits with positive externalities, 2022.  
[13] Andrew S Lan and Richard G Baraniuk. A contextual bandits framework for personalized learning action selection. In EDM, pages 424-429, 2016.  
[14] Dexun Li and Pradeep Varakantham. Avoiding starvation of arms in restless multi-armed bandits. In Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems, pages 1303–1311, 2023.  
[15] Long-Ji Lin. Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine learning, 8:293-321, 1992.  
[16] Keqin Liu and Qing Zhao. Indexability of restless bandit problems and optimality of whittle index for dynamic multichannel access. IEEE Transactions on Information Theory, 56(11): 5547-5567, 2010.  
[17] Aditya Mate, Jackson A Killian, Haifeng Xu, Andrew Perrault, and Milind Tambe. Collapsing bandits and their application to public health interventions. arXiv preprint arXiv:2007.04432, 2020.

[18] Christos H Papadimitriou and John N Tsitsiklis. The complexity of optimal queueing network control. In Proceedings of IEEE 9th Annual Conference on Structure in Complexity Theory, pages 318-322. IEEE, 1994.  
[19] Chris Piech, Jonathan Bassen, Jonathan Huang, Surya Ganguli, Mehran Sahami, Leonidas J Guibas, and Jascha Sohl-Dickstein. Deep knowledge tracing. Advances in neural information processing systems, 28, 2015.  
[20] Yundi Qian, Chao Zhang, Bhaskar Krishnamachari, and Milind Tambe. Restless poachers: Handling exploration-exploitation tradeoffs in security domains. In Proceedings of the 2016 International Conference on Autonomous Agents & Multiagent Systems, pages 123-131, 2016.  
[21] Avi Segal, Yossi Ben David, Joseph Jay Williams, Kobi Gal, and Yaar Shalom. Combining difficulty ranking with multi-armed bandits to sequence educational content. In Artificial Intelligence in Education: 19th International Conference, AIED 2018, London, UK, June 27–30, 2018, Proceedings, Part II 19, pages 317–321. Springer, 2018.  
[22] Shitian Shen, Markel Sanz Ausin, Behrooz Mostafavi, and Min Chi. Improving learning & reducing time: A constrained action-based reinforcement learning approach. In Proceedings of the 26th conference on user modeling, adaptation and personalization, pages 43-51, 2018.  
[23] Anni Siren and Vassilios Tzerpos. Automatic learning path creation using oer: a systematic literature mapping. IEEE Transactions on Learning Technologies, 2022.  
[24] Utkarsh Upadhyay, Abir De, and Manuel Gomez Rodriguez. Deep reinforcement learning of marked temporal point processes. Advances in Neural Information Processing Systems, 31, 2018.  
[25] Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3):279-292, 1992.  
[26] Peter Whittle. Restless bandits: Activity allocation in a changing world. Journal of applied probability, pages 287-298, 1988.
