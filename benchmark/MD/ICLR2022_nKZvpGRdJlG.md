# MIND YOUR SOLVER! ON ADVERSARIAL ATTACK AND DEFENSE FOR COMBINATORIAL OPTIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Combinatorial optimization (CO) is a long-standing challenging task not only in its inherent complexity (e.g. NP-hard) but also the possible sensitivity to input conditions. In this paper, we take an initiative on developing the mechanisms for adversarial attack and defense towards combinatorial optimization solvers, whereby the solver is treated as a black-box function and the original problem's underlying graph structure (which is often available and associated with the problem instance, e.g. DAG, TSP) is attacked under a given budget. Experimental results on three real-world combinatorial optimization problems reveal the vulnerability of existing solvers to adversarial attack, including the commercial solvers like Gurobi. In particular, we present a simple yet effective defense strategy to modify the graph structure to increase the robustness of solvers, which shows its universal effectiveness across tasks and solvers.

# 1 INTRODUCTION

The combinatorial optimization (CO) problems are widely studied due to their importance in practice (e.g. job scheduling, routing, matching, etc). In the last century, a variety of heuristic methods (Van Laarhoven & Aarts, 1987; Whitley, 1994) are proposed to tackle these standing and often NP-hard problems. Driven by the recent development of deep learning and reinforcement learning, many learning-based methods (Khalil et al., 2017; Mao et al., 2019; Kwon et al., 2021) are also developed in this area, which show promising potential often for their cost-efficiency.

Despite the success of solvers in various combinatorial optimization tasks, few attention has been paid to the vulnerability and robustness of combinatorial solvers, regardless of whether they are learning based or not. A line of relevant works aims at handling combinatorial optimization under uncertainty (Buchheim & Kurtz, 2018). However, to our best knowledge, ensuring the robustness of combinatorial solvers with slightly modified problem instances remains relatively unexplored. It is worth noting that many CO problems can be essentially formulated as a graph problem (Khalil et al., 2017; Bengio et al., 2020), hence it is attractive and natural to modify the problem instance by modifying the graph structure, to generate more test cases for solvers. In fact, vulnerability can often be an inherent challenge for CO solvers since the problem is often strong nonlinear and NP-hard. From this perspective, we consider attack and defense CO solvers in the following aspects.

From the attack side, developing attack models can be useful for thoroughly evaluating a solver's robustness. The solvers may be more fragile than the general impression: for traditional learning-free solvers, in some cases, their heuristics and hyperparameters may not be universal and stable enough such that a small change on problem condition or graph structure may deteriorate the performance notably. This also holds for recent machine learning based solvers as the model may be overfit and the objective landscape can be complex due to the inherent difficulty of discrete CO problems.

As a result, it is imperative to develop defense mechanisms and techniques to improve the robustness of CO solvers, either for learning-based models or traditional ones, especially if the approach can be in black-box mode without knowing the details of the solvers. In particular, it is even desirable to develop out-of-box defense mechanism. Our hope is that this may be realized when the problem instance change<sup>1</sup> involves only graph structure variation – which is often the case.

Table 1: Comparing our framework (ROCO) with FGSM (Goodfellow et al., 2015) and RL-S2V (Dai et al., 2018).  $\epsilon$ -perturb. means the change of one pixel should be bounded in  $\epsilon$ . B-hop neighbourhood means the new attack edges can only connect two nodes with distance less than  $B$ .  

<table><tr><td>Method</td><td>Data</td><td>Task</td><td>Attack target</td><td>Attack cost</td><td>Attack principle</td><td>Defense tech.</td></tr><tr><td>FGSM</td><td>image</td><td>classification</td><td>pixels</td><td>ε-Perturb.</td><td>invisible change</td><td>adversarial Training</td></tr><tr><td>RL-S2V</td><td>graph</td><td>classification</td><td>edges (connectivity)</td><td>edge #</td><td>B-hop neighbour</td><td>random drop</td></tr><tr><td>ROCO</td><td>CO instance</td><td>CO solution</td><td>edges (constraints)</td><td>edge #</td><td>no worse optimum</td><td>symmetric RL</td></tr></table>

To this end, we present Robust Combinaotorial Optimization (ROCO), a framework for testing and improving the robustness of a given combinatorial optimization solver. Table 1 compares our framework to classical works in images and graphs. Our attacker limits the number of attacked edges in the graph and guarantees that the optimal solution must not become worse. Our defender ensures that the new solution is also feasible for the pre-defended problem. The overview of ROCO framework is summarized in Fig. 1. In summary, this paper makes the following contributions:

1) Given the fact that combinatorial problems can often be represented by underlying graphs, we propose to perform adversarial attacks toward CO solvers to deteriorate their solution quality. To our best knowledge, this is the pioneering work that formally studies adversarial attacks on combinatorial solvers, though their vulnerability has been occasionally recognized by the community.  
2) We propose ROCO, an adversarial framework that consists of both attack and defense models on top of CO solvers. We design our attack models with both learning-based and traditional simulated annealing methods by slightly modifying the graph structures (e.g. add, delete or modify edges). To increase the robustness of the combinatorial solvers, we further propose defense mechanism against attacks. Our attack and defense models are applicable to solvers regardless of learning-based or not.  
3) We implement and apply our adversarial attack and defense models to three common combinatorial optimization tasks: Directed Acyclic Graph Scheduling, Asymmetric Traveling Salesman Problem and Fraud Coverage. The experimental results on black-box attack/defense show the effectiveness and generality of our approach. The source code will be made public available.

# 2 RELATED WORK

Combinatorial optimization. As a widely studied problem, there exist many traditional algorithms for CO, including but not limited to greedy algorithms, heuristic algorithms like simulated annealing (SA) (Van Laarhoven & Aarts, 1987) or Lin-Kernighan-Helsgaun (LKH3) (Helsgaun, 2017), as well as commercial solvers like Gurobi (Gurobi Optimization, 2020). Besides, driven by the recent development of deep learning and reinforce learning, many learning-based methods have also been proposed to tackle these problems. A mainstream approach using deep learning is to predict the solution end-to-end, such as the supervised model Pointer Networks (Vinyals et al., 2015), reinforcement learning models S2V-DQN (Khalil et al., 2017) and MatNet (Kwon et al., 2021). Though these methods did perform well on different types of COPs, they are not that robust and universal, as discussed in (Bengio et al., 2020), the solvers may get stuck around poor solutions in many cases. Different from works (Moon et al., 2019; Zang et al., 2020) which apply CO for attack against neural networks, we take an initiative on the adversarial attack and defense on CO.

Adversarial attack and defense. Since the seminal study (Szegedy et al., 2014) showed that small input perturbations can change model predictions, many adversarial attack methods have been devised to construct such attacks. In general, adversarial attacks can be roughly divided into two categories: white-box attacks with access to the model gradients, e.g. (Goodfellow et al., 2015; Madry et al., 2018; Carlini & Wagner, 2017), and black-box attacks, with only access to the model predictions, e.g. (Ilyas et al., 2018; Narodytska & Kasiviswanathan, 2016). Besides image and text adversarial attacks (Jia & Liang, 2017), given the importance of graph-related applications and the successful applications of graph neural networks (GNN) (Scarselli et al., 2008), more attentions are recently paid to the robustness of GNNs. In the mean time, many defense strategies like adversarial training (Ganin et al., 2016; Tramér et al., 2020) have also been proposed to counter this series of attack methods. Since CO problems can usually be encoded by a graph structure and inspired by (Dai et al., 2018), which develops an RL based attack policy towards GNNs, we propose a novel and flexible attack and defense framework for CO solvers using both heuristic and RL methods.

Note that the recent adversarial graph matching (GM) network show how to fulfill attack or defense via perturbing or regularizing geometry property on the GM solver. (Zhang et al., 2020) degrades the

![](images/131ef6ba1f940520f358a19f5956fe4f8810202d46d0b79e13c453acaec08f0c.jpg)  
Figure 1: Overview of our attack and defense framework ROCO for CO solvers. ROCO targets on the CO problems which can be encoded by graph (often holds in practice). Here delete/add the edges in the encoded graph represents delete/add constraints in CO. Symmetric RL denotes that the defender and attacker share the same structure with symmetric reward and action space.

quality of GM by perturbing nodes to more dense regions while (Ren et al., 2021) improves robustness by separating nodes to be distributed more broadly. However, the techniques are deliberately tailored to the specific problem and can hardly generalize to the general CO problems. Meanwhile they work in a white box mode while we aim to develop more flexible black box models.

# 3 COMBINATORIAL OPTIMIZATION WITH ATTACK AND DEFENSE

# 3.1 PROBLEM FORMULATION

In general, a traditional CO problem  $Q$  defined on graph  $\mathcal{G} = (V,E)$  can be formulated as:

$$
Q: \min  _ {\mathbf {x}} f (\mathbf {x} | \mathcal {G}) \quad s. t. \quad h _ {i} (\mathbf {x}, \mathcal {G}) \leq 0, i = 1, \dots , I \tag {1}
$$

where  $\mathbf{x}$  denotes the decision variable,  $f(\mathbf{x}|\mathcal{G})$  represents the target function w.r.t. the specific CO problem and  $h_i(\mathbf{x},\mathcal{G})$  denotes the set of constraints (usually encoded in graphs). However, due to the NP-hard nature (which is often the case in CO), it can be infeasible to find the optimal solution within polynomial time. Therefore, we denote a different solver  $\mathcal{S}$  (which gives the feasible solution  $f(\mathcal{S}(Q)|\mathcal{G}))$  to approach the global optimum  $f^{*}(Q)$ .

It is worth noting that the optimum  $f^{*}(Q)$  of Eq. 1 will become no worse if we loosen part of the constraints  $h_{i}$  since the previous decision variable  $\mathbf{x}$  is still feasible under the new setting. Intuitively, we may expect the solver to give a better (at least the same) solution on the new problem  $Q^{\prime}$ . However, we will show in this paper that many solvers are vulnerable to such perturbations and their solutions can become worse under our attacks, despite the loose bound  $f^{*}(Q^{\prime}) \leq f^{*}(Q)$ .

Given a solver  $S$  and an original problem  $Q$  represented by a graph  $\mathcal{G}$ , the adversarial attacker  $g$  is asked to modify the graph  $\mathcal{G}$  into  $\mathcal{G}'$  to attack the solver  $S$ , such that:

$$
\max  _ {\mathcal {G} ^ {\prime}} \quad f (\mathcal {S} (Q ^ {\prime}) | \mathcal {G} ^ {\prime}) - f (\mathcal {S} (Q) | \mathcal {G}) \tag {2}
$$

$$
s. t. \quad \mathcal {G} ^ {\prime} = g (\mathcal {S}, \mathcal {G}), \text {h e n c e} Q \rightarrow Q ^ {\prime}, \quad f ^ {*} (Q ^ {\prime}) \leq f ^ {*} (Q), \quad \mathcal {T} (\mathcal {G}, \mathcal {G} ^ {\prime}) = 1
$$

Here  $\mathcal{T}(\cdot, \cdot) \to \{0,1\}$  is an equivalency indicator (Dai et al., 2018) that tells whether two graphs  $\mathcal{G}$  and  $\mathcal{G}'$  satisfy a specified constraint. In short, the above equation tells that the attacker is aiming at making small modifications to the original graph, loosening the constraints while making the solver solution as bad as possible.

In this paper, concretely our attacker  $g$  is allowed to modify edges (e.g. adding or removing edges) from  $\mathcal{G}$  to construct the new graph. Accordingly, we define the equivalency indicator as:

$$
\mathcal {T} \left(\mathcal {G}, \mathcal {G} ^ {\prime}\right) = \mathbb {I} \left(\left| \left(E - E ^ {\prime}\right) \cup \left(E ^ {\prime} - E\right) \right| \leq K\right) \tag {3}
$$

which ensures that the attacker can modify no more than  $K$  edges of the original graph.

On the other hand, it is imperative to develop defense mechanism for against the above attacks. Notice that the attack methods we mentioned before have some degree of symmetry (adding/deleting edges), we can simply do reverse operations for defense. For example, if we can relax the constraints by removing edges while worsening the solver's solution, then we can add some edges (constraints) and get a better solution (that is, the symmetry). Besides, the new solution under stronger constraints

![](images/a732cf523d1cc48e18ff8d222702c9d2c672c29ddfaa0f693f723726c4970e2a.jpg)  
Figure 2: Attack and defense on applying Shortest Job First algorithm for solving DAG. The edges show the dependencies.  $(x,y)$  of each node means run time  $(x)$  and resource occupancy rate  $(y)$ .

is surely feasible for the original graph (then we can use it in the original graph to get  $f(\mathcal{S}(Q')|\mathcal{G}))$ ). Hence, the new problem can be formulated as:

$$
\min  _ {\mathcal {G} ^ {\prime}} \quad f (\mathcal {S} (Q ^ {\prime}) | \mathcal {G}) - f (\mathcal {S} (Q) | \mathcal {G}) \tag {4}
$$

$$
s. t. \quad \mathcal {G} ^ {\prime} = d (\mathcal {S}, \mathcal {G}), \text {h e n c e} Q \rightarrow Q ^ {\prime}, \quad H _ {j} \left(\mathcal {G} ^ {\prime}, \mathcal {G}\right) \leq 0, \text {f o r} j = 1 \dots J, \quad \mathcal {T} \left(\mathcal {G}, \mathcal {G} ^ {\prime}\right) = 1
$$

here the constraints  $H_{j}(\mathcal{G}^{\prime},\mathcal{G})\leq 0$  ensure that the feasible space of  $\mathcal{G}'$  is a subset of  $\mathcal{G}$

Concrete Examples. Fig. 2 shows the attack and defense of the Shortest Job First algorithm on DAG (TSP and FC examples are in Appendix A). We remove an edge but get a worse finish time (objective – the smaller the better). Then we add an edge for defense, which leads to a better solution.

In this paper, we focus on black-box attack and defense, which means we have no idea on the solver. This setting is practical especially considering there are plenty of commercial solvers e.g. Gorubi and CPLEX etc. We leave white box attack and defense for future work.

# 3.2 ATTACK VIA GRAPH MODIFICATION

We devise both reinforce learning (RL) and heuristic based attackers. For RL, the popular Proximal Policy Optimization (PPO) (Schulman et al., 2017) framework is adopted. We also design three traditional heuristic attackers: random sampling, optimum-guided search and simulated annealing.

# 3.2.1 REINFORCE LEARNING BASED ATTACK

Eq. 2 is treated as the learning objective and we resort to reinforcement learning (RL) to optimize  $\mathcal{G}'$  in a data-driven manner. In general, we modify the graph structure and compute  $f(S(Q')|\mathcal{G}')$  alternatively, getting rewards that will be fed into the PPO framework and train the agent iteratively.

MDP Formulation. Given an instance  $(S, \mathcal{G})$ , with a total modification budget, we model the attack via sequential edge modification as a Finite Horizon Markov Decision Process (MDP).

- State. The current graph  $\mathcal{G}^k$  (i.e. the graph  $\mathcal{G}'$  after  $k$  actions) is treated as the state, whose nodes and edges encode both current input and constraints. The original graph  $\mathcal{G}^0$  is the starting state.  
- Action. As mentioned in Sec. 3.1, the attacker is allowed to add/delete edges in the graph. So a single action at time step  $k$  is  $a^k \in \mathcal{A}^k \subseteq E^k$ . Here our action space  $\mathcal{A}^k$  is usually a subset of all the edges  $E^k$  because we restrict the action space (i.e. abandon some useless edge candidates) according to the previous solution  $S(Q^k)$  to speed up our algorithm. Furthermore, we decompose the action space  $(O(|V|^2) \to O(|V|))$  by transforming the edge selection into two node selections: first selecting the starting node, then the ending node.  
- Reward. The new graph  $\mathcal{G}^{k + 1}$  results in a new CO problem  $Q^{k + 1}$  whose objective becomes  $f(S(Q^{k + 1})|\mathcal{G}^{k + 1})$ . The reward is the increase of the objective:

$$
r = f \left(\mathcal {S} \left(Q ^ {k + 1}\right) \mid \mathcal {G} ^ {k + 1}\right) - f \left(\mathcal {S} \left(Q ^ {k}\right) \mid \mathcal {G} ^ {k}\right) \tag {5}
$$

- Terminal. Once the agent modifies  $K$  edges or edge candidates become empty, the process stops.

PPO Design. The input and constraints of a CO problem are usually tightly encoded in the graph structure. Thus, our PPO agent (i.e. the actor and the critic) should behave according to the graph features. Specifically, We resort to the Graph Neural Networks (GNN) for graph embedding:

$$
\mathbf {n} = \operatorname {G N N} \left(\mathcal {G} ^ {k}\right), \mathbf {g} = \operatorname {A t t P o o l} (\mathbf {n}) \tag {6}
$$

where the matrix  $\mathbf{n}$  (with the size of node number  $\times$  embedding dim) is the node embedding, and an attention pooling layer is used to extract a graph level embedding  $\mathbf{g}$ . The GNN model can differ by the CO problem. After graph feature extraction, we design the corresponding actor and critic net:

- Critic. The critic predicts the value of each state  $\mathcal{G}^k$ . Since it aims reward maximization, a max pooling layer is adopted over all node features which are concatenated (denoted by  $[\cdot ||\cdot ]$  ) with the graph embedding  $\mathbf{g}$ , fed into a network (e.g. ResNet block (He et al., 2016)) for value prediction:

$$
\mathcal {V} \left(\mathcal {G} ^ {k}\right) = \operatorname {R e s N e t} _ {1} \left( \right.\left[ \right. \operatorname {M a x P o o l} (\mathbf {n}) \left. \right\rVert \mathbf {g} \left. \right]\left. \right) \tag {7}
$$

- Actor. As mentioned in Sec. 3.2.1, the edge selection is implemented by selecting the start and end node. The action scores are computed using two independent ResNet blocks, and a Softmax layer is added to regularize the scores into probabilities within  $[0,1]$  as follows:

$$
P \left(a _ {1}\right) = \operatorname {s o f t m a x} \left(\operatorname {R e s N e t} _ {2} ([ \mathbf {n} | | \mathbf {g} ])\right), P \left(a _ {2} \mid a _ {1}\right) = \operatorname {s o f t m a x} \left(\operatorname {R e s N e t} _ {3} ([ \mathbf {n} | | \mathbf {n} [ a _ {1} ] | | \mathbf {g})\right) \tag {8}
$$

where  $\mathbf{n}[a_1]$  denotes the embedding for node  $a_1$ . We add the feature vector of the selected start node for the end node selection. For training, actions are sampled w.r.t. their probabilities. For testing, beam search is adopted to find the optimal solution: actions with top- $B$  probabilities are chosen for each graph in the last time step, and only those actions with top- $B$  rewards will be reserved for the next search step (see Alg. 1 for details).

# 3.2.2 HEURISTIC ALGORITHM ATTACKING

Traditional heuristic algorithms are also studied, with three attack algorithms as follows.

Random sampling. In each iteration, an edge is randomly chosen to be modified in the graph and it repeats for  $K$  iterations. We run  $N$  attack trials and choose the best solution. It can reflect the robustness of solvers with the cost of time complexity  $O(NK)$ .

Optimum-guided search (OG-Search). It focuses on finding the optimum solution during each iteration. We use beam search to maintain the best  $B$  current states and randomly sample  $M$  different actions from the candidates to generate next states. The number of iterations is set to be no more than  $K$ . Its time complexity is  $O(BMK)$ .

Simulated Annealing (SA). Simulated annealing (Van Laarhoven & Aarts, 1987) comes from the idea of annealing and cool-

# Algorithm 1: Attack framework by iterative edge manipulation (RL version)

Input: Input graph  $\mathcal{G}$  solver  $S$  max number of actions  $K$  beam size  $B$ $\mathcal{G}_{1..B}^0\gets \mathcal{G};\mathcal{G}^*\gets \mathcal{G};$  # set initial state for  $k\gets 1..K$  do

for  $b\gets 1..B$  do #do beam search for graphs in last step Predict  $P(a_{1}),P(a_{2}|a_{1})$  on  $\mathcal{G}_b^{k - 1}$  Select  $(a_1,a_2)$  with top-  $B$  probabilities;

for each  $(b,a_{1},a_{2})$  pair do  $\begin{array}{rl} & {\mathcal{G}^{\prime}(b,a_1,a_2)\gets \mathrm{mod}\mathrm{f}\mathrm{y}\mathrm{e}\mathrm{d}\mathrm{e}\mathrm{g}\mathrm{e}\left(a_1,a_2\right)\mathrm{in}}\\ & {\mathcal{G}_b^{k - 1};\# \mathrm{new}\mathrm{state}\mathrm{by}\mathrm{tentative}\mathrm{action}}\\ & {\mathsf{if}f(\mathcal{S}|\mathcal{G}^\prime (b,a_1,a_2)) > f(\mathcal{S}|G^*)\mathrm{then}}\\ & {\mathcal{G}^*\gets \mathcal{G}^\prime (b,a_1,a_2)\# \mathrm{update}\mathrm{the}}\\ & {\mathsf{optimal}\mathrm{attacked}\mathrm{graph}} \end{array}$

Sort  $\mathcal{G}'(\cdot, \cdot, \cdot)$  w.r.t. their solutions by decreasing order; # select top- $B$  graphs for next step  $\mathcal{G}_{1..B}^k \gets \mathcal{G}_{1..B}^l$ ;

Output: Optimal Attacked Graph  $\mathcal{G}^*$

ing used in physics for particle crystallization. In our scenario, a higher temperature indicates a higher probability of accepting a worse solution, allowing to jump out of the local optimum. As the action number increases and the temperature decreases, we will be more conservative and tend to reject the bad solution. The detailed process is shown in Appendix B and we will repeat the algorithm for  $N$  times. SA is a fine-tuned algorithm and we can use grid search to find the best parameter to

fit the training set. Its time complexity is  $O(NMK)$ .

Table 2 concludes the attacking methods property and time complexity. Since the former three algorithms are inherently stochastic, we will run them multiple times to calculate the mean and standard deviation for fair comparison.

Table 2: Comparison of attack models. Random means it will produce different results in different trials. Finetune means the algorithm can be tuned by training set.

<table><tr><td>Technique</td><td>Random</td><td>Finetune</td><td>Time</td></tr><tr><td>Random</td><td>✓</td><td></td><td>O(NK)</td></tr><tr><td>OG-Search</td><td>✓</td><td></td><td>O(BMK)</td></tr><tr><td>SA</td><td>✓</td><td>✓</td><td>O(NMK)</td></tr><tr><td>RL</td><td></td><td>✓</td><td>O(BMK)</td></tr></table>

# 3.3 DEFENSE VIA GRAPH MODIFICATION

We adopt RL as the defender and treat Eq. 4 as the

learning objective. The defense MDP formulation is just the same as Sec. 3.2.1 except that we set  $r = f(\mathcal{S}(Q^k)|\mathcal{G}) - f(\mathcal{S}(Q^{k + 1})|\mathcal{G})$  and use the symmetric action of the attacker. It is worth noting

Table 3: DAG attack results of Ratio  $(\%)\uparrow \pm$  Std. Baseline denotes mean finish time (real time should  $\times 5000$  ) on test set. Ratio represents time improvement after attack w.r.t. baselines. The larger the ratio, the better attack performance the adversarial attack method achieve. Random, OG-search, SA are tested for 10 trials to calculate the mean and std.  

<table><tr><td rowspan="2">Solver</td><td rowspan="2">TPC-H job#</td><td rowspan="2">Baseline</td><td colspan="4">Attack Method (Ratio ± Std)</td></tr><tr><td>Random</td><td>OG-Search</td><td>SA</td><td>RL</td></tr><tr><td>Shortest Job First</td><td>50</td><td>20.9228</td><td>1.08 ± 0.12</td><td>1.33 ± 0.18</td><td>1.54 ± 0.07</td><td>1.41</td></tr><tr><td>Critical Path</td><td>50</td><td>17.3900</td><td>8.13 ± 0.44</td><td>9.03 ± 0.25</td><td>9.58 ± 0.15</td><td>9.26</td></tr><tr><td>Tetris (Grandl et al., 2014)</td><td>50</td><td>16.4538</td><td>11.57 ± 0.60</td><td>12.05 ± 0.80</td><td>14.02 ± 0.52</td><td>14.22</td></tr><tr><td>Shortest Job First</td><td>100</td><td>38.3202</td><td>0.26 ± 0.03</td><td>0.41 ± 0.04</td><td>0.48 ± 0.02</td><td>0.54</td></tr><tr><td>Critical Path</td><td>100</td><td>32.0355</td><td>8.57 ± 0.28</td><td>8.98 ± 0.27</td><td>9.13 ± 0.02</td><td>9.24</td></tr><tr><td>Tetris (Grandl et al., 2014)</td><td>100</td><td>30.3722</td><td>13.27 ± 0.36</td><td>12.60 ± 0.73</td><td>14.70 ± 0.49</td><td>15.41</td></tr><tr><td>Shortest Job First</td><td>150</td><td>57.1554</td><td>0.84 ± 0.07</td><td>1.12 ± 0.08</td><td>1.30 ± 0.05</td><td>1.35</td></tr><tr><td>Critical Path</td><td>150</td><td>48.7963</td><td>5.33 ± 0.37</td><td>6.27 ± 0.37</td><td>6.65 ± 0.12</td><td>6.85</td></tr><tr><td>Tetris (Grandl et al., 2014)</td><td>150</td><td>44.9376</td><td>11.21 ± 0.85</td><td>11.44 ± 0.90</td><td>13.04 ± 0.26</td><td>12.73</td></tr></table>

Table 4: DAG attack and defense results of Time ↓ and Ratio (%) ↓. The solvers' solutions are recorded and the all the ratio is computed by the solved finish time w.r.t. Normal solution.  

<table><tr><td rowspan="2">Solver</td><td rowspan="2">Mode</td><td colspan="2">job#=50</td><td colspan="2">job#=100</td><td colspan="2">job#=150</td></tr><tr><td>Time↓</td><td>Ratio↓</td><td>Time↓</td><td>Ratio↓</td><td>Time↓</td><td>Ratio↓</td></tr><tr><td>Shortest Job First</td><td>Normal</td><td>20.9228</td><td>0.00</td><td>38.3202</td><td>0.00</td><td>57.1554</td><td>0.00</td></tr><tr><td>Shortest Job First</td><td>Attack</td><td>21.2093</td><td>1.37</td><td>38.5335</td><td>0.55</td><td>57.9326</td><td>1.36</td></tr><tr><td>Shortest Job First</td><td>Defense</td><td>20.9151</td><td>-0.04</td><td>38.0470</td><td>-0.71</td><td>57.4370</td><td>0.49</td></tr><tr><td>Critical Path</td><td>Normal</td><td>17.3900</td><td>0.00</td><td>32.0355</td><td>0.00</td><td>48.7963</td><td>0.00</td></tr><tr><td>Critical Path</td><td>Attack</td><td>18.9782</td><td>9.13</td><td>34.9976</td><td>9.25</td><td>52.1519</td><td>6.88</td></tr><tr><td>Critical Path</td><td>Defense</td><td>18.4335</td><td>6.00</td><td>33.4258</td><td>4.34</td><td>49.9011</td><td>2.26</td></tr><tr><td>Tetris (Grandl et al., 2014)</td><td>Normal</td><td>16.4538</td><td>0.00</td><td>30.3722</td><td>0.00</td><td>44.9376</td><td>0.00</td></tr><tr><td>Tetris (Grandl et al., 2014)</td><td>Attack</td><td>18.7944</td><td>14.22</td><td>35.0321</td><td>15.34</td><td>50.6415</td><td>12.69</td></tr><tr><td>Tetris (Grandl et al., 2014)</td><td>Defense</td><td>17.7033</td><td>7.59</td><td>34.2604</td><td>12.80</td><td>49.2008</td><td>9.49</td></tr></table>

that the defense RL agent can not only play a defensive role against the attacked problem instance, but can also help further improve the solution of normal instances, as will be shown in some of our experiments. We leave more in-depth analysis and corresponding approach design for future work.

# 4 EXPERIMENTS AND RESULTS

We conduct experiments on three representative tasks: Directed Acyclic Graph Scheduling, Asymmetric Traveling Salesman Problem and Fraud Coverage. The former two problems are popular problems in CO. The third problem is originated from a real-world transaction dataset. The detailed graph embedding for the three tasks is shown in Appendix C. In Appendix G, we provide the training and evaluation parameters of different solvers for fair time comparison and reproducibility. All experiments are run on RTX 2080Ti and RTX 3090 (see Appendix H for the detailed testbed).

# 4.1 TASK I: DIRECTED ACYCLIC GRAPH SCHEDULING

Task scheduling for heterogeneous systems and various jobs is a popular problem due to its practical importance. Many systems formulate the job stages and their dependencies as a Directed Acyclic Graph (DAG) (Saha et al., 2015; Chambers et al., 2010; Zaharia et al., 2012). The data center has limited computing resources to allocate the jobs with different resource requirements. These jobs can run in parallel if all their parent jobs have finished and the required resources are available. Our goal is to minimize the finish time of the jobs i.e. we should finish all jobs as soon as possible.

Solvers. We choose three popular heuristic solvers as our attack targets. First, the Shortest Job First algorithm chooses the jobs greedily with minimum completion time. Second, the Critical Path algorithm analyzes the bottleneck and finishes the jobs in the critical path sequence. Third, the Tetris (Grandl et al., 2014) scheduling algorithm models the jobs as 2-dimension blocks in the Tetris games according to their finish time and resource requirement.

Attack model. The edges in a DAG represent job dependencies, and removing edges will relax the constraints. After removing existing edges in a DAG, it is obvious that the new solution will be equal or better than the original one since there are less restrictions. As a result, in the DAG scheduling tasks, the attack model is to selectively remove existing edges.

Defense model. We propose to add non-existing edges on the input graph associated with the CO problem, and obviously the new solution under more constraints is still feasible for the original CO problem. The motivation is to help tune the graph structure to be more suitable for heuristic

algorithms. To reduce the action space, we propose to pre-process the node pairs that already have dependencies and remove the corresponding edges in the candidate set.

Dataset. We use the TPC-H dataset (http://tpc.org/tpch/default5.asp), which is composed of business-oriented queries and concurrent data modification. Many DAGs have tens or even hundreds of stages with different duration and numbers of parallel tasks. As each DAG in TPC-H dataset represents a computation job, we gather the DAGs randomly and generate three different datasets, TPC-H-50, TPC-H-100, TPC-H-150, of each containing 50 training and 10 testing samples. Each DAG node has two properties: execution time and resource requirement.

Results for attack. Table 3 reports the results of our four attack methods, where RL outperforms other learning-free methods in most cases, illustrating the correctness of our feature extraction techniques and training framework. It is worth noting that even the simplest random attack can cause a significant performance degradation to the CO solvers, showing their vulnerability and the effectiveness of the attack framework.

Results for attack and defense. Table 4 and Fig. 3 show the results of attack and defense experiments on DAG. In general, the defense model can compensate for the damage of the attack and can even obtain better solutions than the baseline in some cases. It's also worth noting that for some instances, the edges removed in the attack stage will be added back in the defense.

![](images/90ba519504fbb39f2d05f418adcb20b0824a7b049198e87d072d1cd11072443b.jpg)  
Figure 3: Finish time  $\downarrow$  as DAG objective score (mean and std by 10 trials) among three modes: attack, defense and normal: schedule 100 jobs from TPC-H. Attack will incur worse score than in normal mode, which can be remedied by defense.

# 4.2 TASK II: ASYMMETRIC TRAVELING SALESMAN PROBLEM

The classic traveling salesman problem (TSP) is to find the shortest cycle to travel across all the cities. Here we tackle the even challenging asymmetric TSP (ATSP) for its generality.

Solvers. Four algorithms are treated as our attack targets: i) Nearest Neighbour greedily adds the nearest city to the tour. ii) Furthest Insertion finds the city with the furthest distance to the existing cities in the tour and inserts it. iii) Lin-Kernighan Heuristic (LKH3) (Helsgaun, 2017) is the traditional SOTA TSP solver. iv) Matrix Encoding Networks (MatNet) (Kwon et al., 2021) claims as a SOTA learning-based solver for ATSP and flexible flow shop (FFSP).

Attack model. The attack is to choose an edge and half its value, after which we will get a better theoretical optimum. To reduce the action space, we will not select the edges in the current path predicted by the solver at the last time step.

Defense model. First we calculate the optimal path by the solver and add these edges to the candidate set. The action is to modify an edge's weight by doubling the distance of that edge in order to encourage the solver to explore other paths.

Dataset. It comes from (Kwon et al., 2021) consisting of 'tmat' class ATSP instances which have the triangle inequality and are widely studied by the operation research community (Cirasella et al., 2001). We solve the ATSP of three sizes, 20, 50 and 100 cities. The distance matrix is fully connected and asymmetric, and each dataset consists of 50 training samples and 20 testing samples.

Results for attack. Table 5 reports the attack results of four target solvers. In general, the learning-based solvers (e.g. MatNet) or those with intrinsic randomness (e.g. LKH3) show stronger robustness to the attacks. Furthermore, it is notable that the RL based attack outperforms in most cases.

Results for attack and defense. Table 6 shows that the defense model works well on ATSP. In addition to making up the degeneration by attack, in some cases it even obtains shorter total distance.

# 4.3 TASK III: FRAUD COVERAGE

Our last problem instance refers to Fraud Coverage (FC), which is an emerging NP-Complete (details in Appendix D.1) problem abstracted from real life: the growing online transactions have also spawned criminals and scams. The transactions  $\mathcal{E}$  can be classified into black (fraudulent) events  $\mathcal{B}$  and white (normal) events  $\mathcal{W}$ . To block fraud events, the bank system designs a series of rules  $\mathcal{R}$  to

Table 5: ATSP attack results of Ratio  $(\%)\uparrow \pm$  Std. Baseline denotes mean tour length on test set. Result is the mean ratio on all test instances computed by the solved tour length w.r.t. baselines.  

<table><tr><td rowspan="2">Solver</td><td rowspan="2">City#</td><td rowspan="2">Baseline ×106</td><td colspan="4">Attack Method</td></tr><tr><td>Random</td><td>OG-Search</td><td>SA</td><td>RL</td></tr><tr><td>Nearest Neighbour</td><td>20</td><td>1.9354</td><td>10.09 ± 0.79</td><td>9.34 ± 1.67</td><td>10.28 ± 0.82</td><td>12.94</td></tr><tr><td>Furthest Insertion</td><td>20</td><td>1.6092</td><td>5.35 ± 0.65</td><td>5.18 ± 0.73</td><td>6.78 ± 0.71</td><td>8.56</td></tr><tr><td>LKH3 (Helsgaun, 2017)</td><td>20</td><td>1.4595</td><td>0.03 ± 0.02</td><td>0.03 ± 0.03</td><td>0.10 ± 0.07</td><td>0.11</td></tr><tr><td>MatNet (Kwon et al., 2021)</td><td>20</td><td>1.4616</td><td>0.40 ± 0.08</td><td>0.46 ± 0.04</td><td>0.46 ± 0.06</td><td>0.65</td></tr><tr><td>Nearest Neighbour</td><td>50</td><td>2.2247</td><td>6.24 ± 0.37</td><td>7.02 ± 0.43</td><td>8.14 ± 0.68</td><td>10.26</td></tr><tr><td>Furthest Insertion</td><td>50</td><td>1.9772</td><td>4.15 ± 0.36</td><td>3.51 ± 0.63</td><td>4.35 ± 0.45</td><td>6.97</td></tr><tr><td>LKH3 (Helsgaun, 2017)</td><td>50</td><td>1.6621</td><td>0.19 ± 0.04</td><td>0.21 ± 0.04</td><td>0.37 ± 0.06</td><td>0.35</td></tr><tr><td>MatNet (Kwon et al., 2021)</td><td>50</td><td>1.6915</td><td>1.39 ± 0.07</td><td>1.71 ± 0.06</td><td>2.01 ± 0.07</td><td>2.15</td></tr><tr><td>Nearest Neighbour</td><td>100</td><td>2.1456</td><td>4.02 ± 0.46</td><td>3.53 ± 0.71</td><td>3.81 ± 0.52</td><td>5.02</td></tr><tr><td>Furthest Insertion</td><td>100</td><td>1.9209</td><td>2.88 ± 0.46</td><td>2.97 ± 0.58</td><td>3.35 ± 0.33</td><td>4.87</td></tr><tr><td>LKH3 (Helsgaun, 2017)</td><td>100</td><td>1.5763</td><td>0.40 ± 0.04</td><td>0.54 ± 0.03</td><td>0.59 ± 0.02</td><td>0.63</td></tr><tr><td>MatNet (Kwon et al., 2021)</td><td>100</td><td>1.6545</td><td>1.37 ± 0.06</td><td>1.63 ± 0.03</td><td>1.79 ± 0.04</td><td>1.98</td></tr></table>

Table 6: ATSP attack and defense results of Distance  $\downarrow$  and Ratio  $\left( \% \right)  \downarrow$  . The solutions are recorded and the ratio is computed by the solved tour length w.r.t. normal solution.  

<table><tr><td rowspan="2">Solver</td><td rowspan="2">Mode</td><td colspan="2">ATSP-20</td><td colspan="2">ATSP-50</td><td colspan="2">ATSP-100</td></tr><tr><td>Distance↓</td><td>Ratio↓</td><td>Distance↓</td><td>Ratio↓</td><td>Distance↓</td><td>Ratio↓</td></tr><tr><td>Nearest Neighbour</td><td>Normal</td><td>1.9354</td><td>0.00</td><td>2.2247</td><td>0.00</td><td>2.1456</td><td>0.00</td></tr><tr><td>Nearest Neighbour</td><td>Attack</td><td>2.1366</td><td>10.40</td><td>2.4264</td><td>9.07</td><td>2.2439</td><td>4.58</td></tr><tr><td>Nearest Neighbour</td><td>Defense</td><td>1.7564</td><td>-9.25</td><td>2.2069</td><td>-0.80</td><td>2.0319</td><td>-5.30</td></tr><tr><td>Furthest Insertion</td><td>Normal</td><td>1.6092</td><td>0.00</td><td>1.9772</td><td>0.00</td><td>1.9272</td><td>0.00</td></tr><tr><td>Furthest Insertion</td><td>Attack</td><td>1.7088</td><td>6.19</td><td>2.0957</td><td>5.99</td><td>1.9963</td><td>3.58</td></tr><tr><td>Furthest Insertion</td><td>Defense</td><td>1.5210</td><td>-5.48</td><td>1.9558</td><td>-1.08</td><td>1.8990</td><td>-1.46</td></tr><tr><td>LKH3 (Helsgaun, 2017)</td><td>Normal</td><td>1.4595</td><td>0.00</td><td>1.6621</td><td>0.00</td><td>1.5763</td><td>0.00</td></tr><tr><td>LKH3 (Helsgaun, 2017)</td><td>Attack</td><td>1.4598</td><td>0.02</td><td>1.6671</td><td>0.30</td><td>1.5867</td><td>0.66</td></tr><tr><td>LKH3 (Helsgaun, 2017)</td><td>Defense</td><td>1.4595</td><td>0.00</td><td>1.6610</td><td>-0.07</td><td>1.5744</td><td>-0.12</td></tr><tr><td>MatNet (Kwon et al., 2021)</td><td>Normal</td><td>1.4617</td><td>0.00</td><td>1.6915</td><td>0.00</td><td>1.6545</td><td>0.00</td></tr><tr><td>MatNet (Kwon et al., 2021)</td><td>Attack</td><td>1.4708</td><td>0.62</td><td>1.7261</td><td>2.04</td><td>1.6841</td><td>1.79</td></tr><tr><td>MatNet (Kwon et al., 2021)</td><td>Defense</td><td>1.4591</td><td>-0.18</td><td>1.6696</td><td>-1.29</td><td>1.6185</td><td>-2.18</td></tr></table>

identify transactions as either black or white events. The goal is to select a subset of rules  $R \subseteq \mathcal{R}$  to maximize the coverage of fraudulent monetary values while affecting no more than  $K$  white events. The problem can be represented by a bipartite graph, where any edge exists between a rule node and an event node only when the event is deemed as black by the rule. Formally:

$$
\max  _ {R} \sum_ {b \in \mathcal {B}} w (b) \times \mathbb {I} \left(b \in \bigcup_ {r _ {i} \in R} C ^ {+} \left(r _ {i}\right)\right) \quad s. t. \quad | \bigcup_ {r _ {i} \in R} C ^ {-} \left(r _ {i}\right) | \leq K \tag {9}
$$

where  $w(\cdot)$  denotes the monetary value of a certain transaction event,  $C(\cdot)$  denotes the set of events covered by a rule that are deemed as black events, and  $C^{+}(\cdot)$  and  $C^{-}(\cdot)$  denotes the subset of events in  $C(\cdot)$  with true labels being black and white, respectively.

Solvers. As an emerging real-world CO task, the FC problem is very challenging and here we propose three different solvers as the target for attacking. First, the trivial Local algorithm which iterates over the rules sequentially, adding any rules that will not exceed the threshold. Second, a more intelligent Greedy Average algorithm that always chooses the most cost-effective (the ratio of the increase of black event money values to the increase of number of covered white events) rule at each step until the constraint isn't satisfied. Third, we formulate the problem into standard ILP form (details in Appendix D.2) and solve it by Gurobi.

Attack model. Intuitively, when a white event is mislabeled as a black event, the FC problem will achieve an equal or better optimum  $f^{*}(Q')$ , since we can possibly cover more white events while not exceeding the threshold. In our attack model, we focus on the attack toward the edges rather than the nodes. We choose to add non-existing black edges that connect rules to black events, which leads to a theoretically better optimum and can potentially mislead the solvers. Further, in order to reduce action space, we only select the unchosen rules, otherwise it will be useless since adding edges for selected rules would not affect a solver's output solution. Here we report the attack method on adding black edges and present the results for attacking black nodes in the Appendix F.1.

Defense model. Similar to the attack method, as defense we remove the existing black edges that connect rules to black events. To reduce action space, we select the rules chosen in the prior solution, since deleting black edges for the unchosen rules will not change the solution.

Dataset. We analyze the distribution of transaction amounts and rule coverage of the real dataset, then generate a series of simulated data for experiments. The distribution of events amount and the

Table 7: FC attack results of Ratio (\%)  $\uparrow \pm$  Std. The Gurobi time limit is shown in brackets w.r.t. different data sizes (it should be long enough to give a feasible solution but not too long for attack). Baseline is the average original solution of the solvers on test set. The ratio here is the mean of ratios on all test instances computed by the solved FC monetary value w.r.t. baselines.  

<table><tr><td rowspan="2">Solver</td><td rowspan="2">problem size: rule#-event#</td><td rowspan="2">Baseline</td><td colspan="4">Attack Method (Ratio ± Std)</td></tr><tr><td>Random</td><td>OG-Search</td><td>SA</td><td>RL</td></tr><tr><td>Local Search</td><td>30-3K</td><td>9.5713</td><td>0.78 ± 0.06</td><td>0.77 ± 0.11</td><td>0.85 ± 0.03</td><td>0.89</td></tr><tr><td>Greedy Average</td><td>30-3K</td><td>18.0038</td><td>2.72 ± 0.16</td><td>3.17 ± 0.23</td><td>2.70 ± 0.22</td><td>4.79</td></tr><tr><td>Gurobi(1s)</td><td>30-3K</td><td>18.8934</td><td>10.41 ± 1.13</td><td>18.42 ± 1.88</td><td>18.99 ± 1.95</td><td>50.68</td></tr><tr><td>Local Search</td><td>60-6K</td><td>24.9913</td><td>0.47 ± 0.04</td><td>0.80 ± 0.14</td><td>0.69 ± 0.15</td><td>0.76</td></tr><tr><td>Greedy Average</td><td>60-6K</td><td>43.1625</td><td>0.91 ± 0.09</td><td>0.93 ± 0.11</td><td>1.02 ± 0.09</td><td>2.29</td></tr><tr><td>Gurobi(2s)</td><td>60-6K</td><td>41.1828</td><td>7.15 ± 0.84</td><td>9.35 ± 1.02</td><td>7.02 ± 0.89</td><td>100.00</td></tr><tr><td>Local Search</td><td>100-10K</td><td>22.9359</td><td>0.76 ± 0.09</td><td>1.23 ± 0.08</td><td>0.83 ± 0.09</td><td>1.55</td></tr><tr><td>Greedy Average</td><td>100-10K</td><td>51.3905</td><td>1.25 ± 0.14</td><td>1.70 ± 0.34</td><td>1.37 ± 0.08</td><td>1.61</td></tr><tr><td>Gurobi(5s)</td><td>100-10K</td><td>49.3296</td><td>6.33 ± 0.70</td><td>7.69 ± 0.96</td><td>4.26 ± 0.48</td><td>92.01</td></tr></table>

Table 8: FC attack and defense results of Fraud$ ↑ and Ratio (%) ↑. The solvers' solutions are recorded and the ratio is computed by the solved FC monetary value (Fraud$) w.r.t. Normal solution.  

<table><tr><td rowspan="2">Solver</td><td rowspan="2">Mode</td><td colspan="2">rule#=30, event#=3K</td><td colspan="2">rule#=60, event#=6K</td><td colspan="2">rule#=100, event#=10K</td></tr><tr><td>Fraud$↑</td><td>Ratio↑</td><td>Fraud$↑</td><td>Ratio↑</td><td>Fraud$↑</td><td>Ratio↑</td></tr><tr><td>Local Search</td><td>Normal</td><td>9.5713</td><td>0.00</td><td>24.9913</td><td>0.00</td><td>22.9359</td><td>0.00</td></tr><tr><td>Local Search</td><td>Attack</td><td>9.4638</td><td>-1.12</td><td>24.8038</td><td>-0.75</td><td>22.6930</td><td>-1.06</td></tr><tr><td>Local Search</td><td>Defense</td><td>10.0680</td><td>5.19</td><td>25.8300</td><td>3.36</td><td>23.7252</td><td>3.44</td></tr><tr><td>Greedy Average</td><td>Normal</td><td>18.0038</td><td>0.00</td><td>43.1625</td><td>0.00</td><td>51.3905</td><td>0.00</td></tr><tr><td>Greedy Average</td><td>Attack</td><td>17.1256</td><td>-4.88</td><td>42.3911</td><td>-1.79</td><td>50.5651</td><td>-1.61</td></tr><tr><td>Greedy Average</td><td>Defense</td><td>17.6850</td><td>-1.77</td><td>42.8371</td><td>-0.75</td><td>51.0684</td><td>-0.63</td></tr><tr><td>Gurobi</td><td>Normal</td><td>18.8934</td><td>0.00</td><td>41.1828</td><td>0.00</td><td>49.3296</td><td>0.00</td></tr><tr><td>Gurobi</td><td>Attack</td><td>2.7194</td><td>-85.61</td><td>2.2218</td><td>-94.60</td><td>4.6731</td><td>-90.53</td></tr><tr><td>Gurobi</td><td>Defense</td><td>17.2712</td><td>-8.59</td><td>42.1617</td><td>2.38</td><td>51.2941</td><td>3.98</td></tr></table>

rule coverage is shown in Appendix E. The dataset consists of three rule-event pairs 30-3K, 60-6K and 100-10K, each with 50 training samples and 20 testing samples.

Results for attack. Table 7 shows the attack results of our simulated dataset. We can observe that both heuristic and RL approaches have yielded significant attack effects, while RL outperforms the others in most cases (especially for Gurobi, in many cases it is not even possible to give a feasible solution within time after employing RL attacks).

Results for attack and defense. Table 8 records the results of attack and defense experiments on FC problems. Experiments are conducted on the same test set. In general, the defender can compensate for the damage of attack effectively and obtain an even better solution than the baseline in some cases. Besides, as a commercial solver, Gurobi should be able to obtain optimal solutions if in sufficient time (assuming we have unlimited computational resources). So we record the time for Gurobi to find the optimal solution under attack and defense. The result is shown in Fig. 4, where Gurobi's solution time after attack (defense) significantly increases (decreases). This inspires us to attack toward the solvers' solution time in future work.

![](images/5324cbe9a6715e4eb7ec7b6e5cb9a5fa44388a7b37447a1b57ef19a016defd74.jpg)  
Figure 4: Gurobi's mean time cost in solving FC problems. Run experiments on 3 datasets (20 instances) of different sizes.

# 5 CONCLUSION AND OUTLOOK

We have presented a general adversarial attack and defense framework called ROCO on top of combinatorial solvers. For attack, we devise both RL and traditional heuristic attackers to modify the underlying graph structure of combinatorial problems. Meanwhile, we propose a simple yet effective defense mechanism to modify the ill-posed problem in a reversed way to increase the robustness of combinatorial solvers. Experiments show the effectiveness of our paradigm and techniques.

The proposed paradigm opens up large space for further research, at least in the following aspects: 1) new attack/defense techniques beyond graph structure but also node/edge attribute; 2) iterative adversarial training for defense model, especially for learning-based solvers (at least in the sense of tailored data augmentation); 3) white-box attack/defense when the solver information is known.

# REFERENCES

Yoshua Bengio, Andrea Lodi, and Antoine Prouvost. Machine learning for combinatorial optimization: a methodological tour d'horizon, 2020.  
Christoph Buchheim and Jannis Kurtz. Robust combinatorial optimization under convex and discrete cost uncertainty. EURO Journal on Computational Optimization, 6(3):211-238, 2018.  
Nicholas Carlini and David A. Wagner. Towards evaluating the robustness of neural networks. 2017 IEEE Symposium on Security and Privacy (SP), pp. 39-57, 2017.  
Craig Chambers, Ashish Raniwala, Frances Perry, Stephen Adams, Robert R Henry, Robert Bradshaw, and Nathan Weizenbaum. Flumejava: easy, efficient data-parallel pipelines. ACM Sigplan Notices, 45(6):363-375, 2010.  
Jill Cirasella, David S Johnson, Lyle A McGeoch, and Weixiong Zhang. The asymmetric traveling salesman problem: Algorithms, instance generators, and tests. In Workshop on Algorithm Engineering and Experimentation, pp. 32-59. Springer, 2001.  
Hanjun Dai, Hui Li, Tian Tian, Xin Huang, Lin Wang, Jun Zhu, and Le Song. Adversarial attack on graph structured data. In International conference on machine learning(ICML), pp. 1115-1124. PMLR, 2018.  
Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks, 2016.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. International Conference on Learning Representations (ICLR), 2015.  
Robert Grandl, Ganesh Ananthanarayanan, Srikanth Kandula, Sriram Rao, and Aditya Akella. Multi-resource packing for cluster schedulers. ACM SIGCOMM Computer Communication Review, 44(4):455-466, 2014.  
Gurobi Optimization. Gurobi optimizer reference manual. http://www.gurobi.com, 2020.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pp. 770-778, 2016.  
Keld Helsgaun. An extension of the lin-kernighan-helsgaun tsp solver for constrained traveling salesman and vehicle routing problems. Roskilde: Roskilde University, 2017.  
Andrew Ilyas, Logan Engstrom, Anish Athalye, and Jessy Lin. Black-box adversarial attacks with limited queries and information. In International Conference on Machine Learning(ICML), pp. 2137-2146. PMLR, 2018.  
Robin Jia and Percy Liang. Adversarial examples for evaluating reading comprehension systems. In Empirical Methods in Natural Language Processing(EMNLP), 2017.  
Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial optimization algorithms over graphs. Advances in Neural Information Processing Systems(NIPS), 30: 6348-6358, 2017.  
Yeong-Dae Kwon, Jinho Choo, Iljoo Yoon, Minah Park, Duwon Park, and Youngjune Gwon. Matrix encoding networks for neural combinatorial optimization. arXiv preprint arXiv:2106.11113, 2021.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations(ICLR), 2018.  
Hongzi Mao, Malte Schwarzkopf, Shaileshh Bojja Venkatakrishnan, Zili Meng, and Mohammad Alizadeh. Learning scheduling algorithms for data processing clusters. In Proceedings of the ACM Special Interest Group on Data Communication, pp. 270-288. 2019.

Seungyong Moon, Gaon An, and Hyun Oh Song. Parsimonious black-box adversarial attacks via efficient combinatorial optimization, 2019.  
Nina Narodytska and Shiva Prasad Kasiviswanathan. Simple black-box adversarial perturbations for deep networks, 2016.  
Jiaxiang Ren, Zijie Zhang, Jiayin Jin, Xin Zhao, Sixing Wu, Yang Zhou, Yelong Shen, Tianshi Che, Ruoming Jin, and Dejing Dou. Integrated defense for resilient graph matching. In ICML, pp. 8982-8997, 2021.  
Bikas Saha, Hitesh Shah, Siddharth Seth, Gopal Vijayaraghavan, Arun Murthy, and Carlo Curino. Apache tez: A unifying framework for modeling and building data processing applications. In Proceedings of the 2015 ACM SIGMOD international conference on Management of Data, pp. 1357-1369, 2015.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE transactions on neural networks, 20(1):61-80, 2008.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017.  
Christian Szegedy, W. Zaremba, Ilya Sutskever, Joan Bruna, D. Erhan, I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. CoRR, abs/1312.6199, 2014.  
Florian Tramèr, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel. Ensemble adversarial training: Attacks and defenses, 2020.  
Peter JM Van Laarhoven and Emile HL Aarts. Simulated annealing. In Simulated annealing: Theory and applications, pp. 7-15. Springer, 1987.  
Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. Advances in Neural Information Processing Systems(NIPS), 28:2692-2700, 2015.  
Darrell Whitley. A genetic algorithm tutorial. Statistics and computing, 4(2):65-85, 1994.  
Matei Zaharia, Mosharaf Chowdhury, Tathagata Das, Ankur Dave, Justin Ma, Murphy McCauly, Michael J Franklin, Scott Shenker, and Ion Stoica. Resilient distributed datasets: A fault-tolerant abstraction for in-memory cluster computing. In 9th {USENIX} Symposium on Networked Systems Design and Implementation ({NSDI} 12), pp. 15-28, 2012.  
Yuan Zang, Fanchao Qi, Chenghao Yang, Zhiyuan Liu, Meng Zhang, Qun Liu, and Maosong Sun. Word-level textual adversarial attacking as combinatorial optimization. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020. acl-main.540. URL http://dx.doi.org/10.18653/v1/2020.acl-main.540.  
Zijie Zhang, Zeru Zhang, Yang Zhou, Yelong Shen, Ruoming Jin, and Dejing Dou. Adversarial attacks on deep graph matching. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), NeurIPS, pp. 20834-20851. Curran Associates, Inc., 2020.
