# GRAPH TRANSFORMATION POLICY NETWORK FOR CHEMICAL REACTION PREDICTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We address a fundamental problem in chemistry known as chemical reaction product prediction. Our main insight is that the input reactant and reagent molecules can be jointly represented as a graph and the process of generating product molecules from reactant molecules (reaction mechanism) can be formulated as a sequence of graph transformations. To this end, we propose Graph Transformation Policy Network (GTPN) – a novel generic method that combines the strengths of graph neural networks and reinforcement learning to learn the reaction mechanisms directly from known reactions with minimal chemical knowledge. Compared to previous methods, GTPN has some appealing properties such as: end-to-end learning, and making no assumption about the length or the order of graph transformations. In order to guide model search through the complex discrete space of sets of bond changes effectively, we extend the standard policy gradient loss by adding useful constraints. Evaluation results show that GTPN improves the top-1 accuracy over the current state-of-the-art method by about  $3\%$  on the large USPTO dataset, setting a new record of  $83.2\%$ . Our model's performances and prediction errors are also analyzed carefully in the paper.

# 1 INTRODUCTION

Chemical reaction product prediction is a fundamental problem in organic chemistry. It paves the way for planning syntheses of new substances (Chen & Baldi, 2009). For decades, huge effort has been spent to solve this problem. However, most methods still depend on the handcrafted reaction rules (Chen & Baldi, 2009; Kayala & Baldi, 2011; Wei et al., 2016) or heuristically extracted reaction templates (Segler & Waller, 2017; Coley et al., 2017), thus are not well generalizable to unseen reactions.

A reaction can be regarded as a set or sequence of graph transformations in which reactants represented as molecular graphs are transformed into products by modifying the bonds between some atom pairs. See Fig. 1 for an illustration. We call an atom pair  $(u, v)$  that changes its connectivity during reaction and its new bond  $b$  a reaction triple  $(u, v, b)$ . The reaction product prediction problem now becomes predicting a set of reaction triples given the input reactants and reagents. We argue that in order to solve this problem well, an intelligent system should have two key capabilities: (a) Understanding the molecular graph structure of the input reactants and reagents so that it can identify possible reactivity patterns (i.e., atom pairs with changing connectivity). (b) Learning the reaction mechanism so that it can find the correct set of reaction triples to generate the desired products.

Recent state-of-the-art methods (Jin et al., 2017; Bradshaw et al., 2018) have built the first capability by leveraging graph neural networks (Duvenaud et al., 2015; Hamilton et al., 2017; Pham et al., 2017; Gilmer et al., 2017). However, these methods are either unaware of the reaction mechanisms (Jin et al., 2017) or limited to reactions with simple reaction mechanisms (Bradshaw et al., 2018). The main challenge is that the space of all possible configurations of reaction triples is extremely large and non-differentiable. Moreover, a small change in the predicted set of reaction triples can lead to very different reaction products and a little mistake can produce invalid prediction.

In this paper, we propose a novel method called Graph Transformation Policy Network (GTPN) that addresses the aforementioned challenges. Our model consists of three main components: a graph neural network (GNN), a node pair prediction network (NPPN) and a policy network (PN). Starting from the initial graph of reactant and reagent molecules, our model iteratively alternates between

![](images/d12e67e7f3c4fffdaaa2736883f90cd0635b94c2b4842ac9e0ee5bb0b7fc0895.jpg)  
Figure 1: A sample reaction represented as a sequence of graph transformations from reactants (leftmost) to products (rightmost). Atoms are labeled with their type (Carbon, Oxygen,...) and their index (1, 2,...) in the molecular graph. The atom pairs that change connectivity and their new bonds (if existed) are highlighted in green. There are two bond changes in this case: 1) The double bond between O:1 and C:2 becomes single. 2) A new single bond between C:2 and C:10 is added.

![](images/5929eb06f124bff15675b56ea25e82248a10ae00cab4d1092f32056039c90db3.jpg)

![](images/7e103165cc14195a10011e06cd5f2b2052141a6075054762c90239f7d13bd697.jpg)

modeling an input graph using GNN and predicting a reaction triple using NPPN and PN to generate a new intermediate graph as input for the next step until it decides to stop. The final generated graph is considered as the predicted products of the reaction. Importantly, GTPN does not assume any fixed number or any order of bond changes but learn these properties itself. Therefore, one can view GTPN as a reinforcement learning (RL) agent that operates on a complex and non-differentiable space of sets of reaction triples. To guide our model towards learning a diverse yet robust-to-small-changes policy, we customize our loss function by adding some useful constraints to the standard RL loss.

To the best of our knowledge, GTPN is the most generic approach for the reaction product prediction problem so far in the sense that: i) It combines graph neural networks and reinforcement learning into a unified framework and trains everything end-to-end; ii) It does not use any handcrafted or heuristically extracted reaction rules/template to predict the products. Instead, it automatically learns various reaction mechanisms from the training data and can generalize to unseen reactions; iii) It can interpret how the products are formed via the sequence of reaction triples it generates.

We evaluate GTPN on two large public datasets named USPTO-15k and USPTO. Our method significantly outperforms all baselines in the top-1 accuracy, achieving new state-of-the-art results of  $82.39\%$  and  $83.20\%$  on USPTO-15k and USPTO, respectively. In addition, we also provide comprehensive analyses about the performance of GTPN and about different types of errors our model could make.

# 2 METHOD

# 2.1 CHEMICAL REACTION AS MARKOV DECISION PROCESS OF GRAPH TRANSFORMATIONS

A reaction occurs when reactant molecules interact with each other in the presence (or absence) of reagent molecules to form new product molecules by breaking or adding some of their bonds. Our main insight is that reaction product prediction can be formulated as predicting a sequence of such bond changes (also known as reaction mechanism  ${}^{1}$  ) given the reactant and reagent molecules as input. A bond change is characterized by the atom pair (where the change happens) and the new bond type (what is the change). We call this atom pair a reaction atom pair and call this atom pair with the new bond type a reaction triple.

More formally, we represent the entire system of input reactant and reagent molecules as a labeled graph  $\mathcal{G} = (\mathcal{V},\mathcal{E})$  with multiple connected components, each of which corresponds to a molecule. Nodes in  $\mathcal{V}$  are atoms labeled with their atomic numbers and edges in  $\mathcal{E}$  are bonds labeled with their bond types. Given  $\mathcal{G}$  as input, we predict a sequence of reaction triples that transforms  $\mathcal{G}$  into a graph of product molecules  $\mathcal{G}'$ .

As reactions vary in number of transformation steps, we represent the sequence of reaction triples as  $(\xi ,u,v,b)^{0},(\xi ,u,v,b)^{1},\ldots ,(\xi ,u,v,b)^{T - 1}$  or  $(\xi ,u,v,b)^{0:T}$  for short. Here  $T$  is the maximum number of steps,  $(u,v)$  is a pair of nodes,  $b$  is the new edge type of  $(u,v)$ , and  $\xi$  is a binary signal that indicates the end of the sequence. If the sequence ends at  $T_{\mathrm{end}} < T$ ,  $\xi^0,\dots \xi^{T_{\mathrm{end}} - 1}$  will be 1 and

![](images/497e04ab7b8819dfac31b05981264665c0db25745bee93ddcfc90df23e50f607.jpg)  
Figure 2: Workflow of a Graph Transformation Policy Network (GTPN). At every step of the forward pass, our model performs 7 major functions: 1) Computing the atom representation vectors, 2) Computing the most possible  $K$  reaction atom pairs, 3) Predicting the continuation signal  $\xi$ , 4) Predicting the reaction atom pair  $(u, v)$ , 5) Predicting a new bond  $b$  of this atom pair, 6) Updating the atom representation vectors, and 7) Updating the recurrent state.

$\xi^{T_{\mathrm{end}}}, \ldots, \xi^{T-1}$  will be 0. At every step  $\tau$ , if  $\xi^{\tau} = 1$ , we apply the predicted edge change  $(u, v, b)^{\tau}$  on the current graph  $\mathcal{G}^{\tau}$  to create a new intermediate graph  $\mathcal{G}^{\tau+1}$  as input for the next step  $\tau + 1$ . This iterative process of graph transformation can be formulated as a Markov Decision Process (MDP) characterized by a tuple  $(S, A, P, R, \gamma)$ , in which  $S$  is a set of states,  $A$  is a set of actions,  $P$  is a state transition function,  $R$  is a reward function, and  $\gamma$  is a discount factor. Since the process is finite and contains no loop, we set the discount factor  $\gamma$  to be 1. The rest of the MDP tuple are defined as follows:

- State: A state  $s^{\tau} \in S$  is an intermediate graph  $\mathcal{G}^{\tau}$  generated at step  $\tau$  ( $0 \leq \tau < T$ ). When  $\tau = 0$ , we denote  $s^0 = \mathcal{G}^0 = \mathcal{G}$ .  
- Action: An action  $a^{\tau} \in \mathcal{A}$  performed at step  $\tau$  is the tuple  $(\xi, u, v, b)^{\tau}$ . The action is composed of three consecutive sub-actions:  $\xi^{\tau}, (u, v)^{\tau}$ , and  $b^{\tau}$ . If  $\xi^{\tau} = 0$ , our model will ignore the next sub-actions  $(u, v)^{\tau}$  and  $b^{\tau}$ , and all the future actions  $(\xi, u, v, b)^{\tau + 1:T}$ . Note that setting  $\xi^{\tau}$  to be the first sub-action is useful in case a reaction does not happen, i.e.,  $\xi^0 = 0$  
- State Transition: If  $\xi^{\tau} = 1$ , the current graph  $\mathcal{G}^{\tau}$  is modified based on the reaction triple  $(u, v, b)^{\tau}$  to generate a new intermediate graph  $\mathcal{G}^{\tau + 1}$ . We do not incorporate chemical rules such as valency check during state transition because the current bond change may result in invalid intermediate molecules  $\mathcal{G}^{\tau}$ , but later, other bond changes may compensate it to create the valid final products  $\mathcal{G}^{T_{\mathrm{end}}}$ .  
- Reward: We use both immediate rewards and delayed rewards to encourage our model to learn the optimal policy faster. At every step  $\tau$ , if the model predicts  $\xi^{\tau}$ ,  $(u, v)^{\tau}$  or  $b^{\tau}$  correctly, it will receive a positive reward for each correct sub-action. Otherwise, a negative reward is given. After the prediction process has terminated, if the generated products are exactly the same as the groundtruth products, we give the model a positive reward, otherwise a negative reward. The concrete reward values are provided in Appendix A.3.

# 2.2 GRAPH TRANSFORMATION POLICY NETWORK

In this section, we describe the architecture of our model - a Graph Transformation Policy Network (GTPN). GTPN has three main components namely a Graph Neural Network (GNN), a Node Pair Prediciton Network (NPPN), and a Policy Network (PN). Each component is responsible for one or several key functions shown in Fig. 2: GNN performs functions 1 and 6; NPPN performs function 2; and PN performs functions 3, 4 and 5. Apart from these components, GTPN also has a Recurrent

Neural Network (RNN) to keep track of the past transformations. The hidden state  $h$  of this RNN is used by NPPN and PN to make accurate prediction.

# 2.2.1 GRAPH NEURAL NETWORK

To model the intermediate graph  $\mathcal{G}^{\tau}$  at step  $\tau$ , we compute the node state vector  $\boldsymbol{x}_i^\tau$  of every node  $i$  in  $\mathcal{G}^{\tau}$  by using a variant of the Message Passing Neural Networks (Gilmer et al., 2017):

$$
\boldsymbol {x} _ {i} ^ {\tau} = \text {M e s s a g e P a s s i n g} ^ {m} \left(\boldsymbol {x} _ {i} ^ {\tau - 1}, \boldsymbol {v} _ {i}, \mathcal {N} ^ {\tau} (i)\right) \tag {1}
$$

where  $m$  is the number of message passing steps;  $\pmb{v}_i$  is the feature vector of node  $i$ ;  $\mathcal{N}^{\tau}(i)$  is the set of all neighbor nodes of node  $i$ ; and  $\pmb{x}_i^{\tau - 1}$  is the state vector of node  $i$  at the previous step. When  $\tau = 0$ ,  $\pmb{x}_i^{\tau - 1}$  is initialized from  $\pmb{v}_i$  using a neural network. Details about the MessagePassing(.) function are provided in Appendix A.1.

# 2.2.2 NODE PAIR PREDICTION NETWORK

In order to predict how likely an atom pair  $(i,j)$  of the intermediate graph  $\mathcal{G}^{\tau}$  will change its bond, we assign  $(i,j)$  with a score  $s_{ij}^{\tau} \in \mathbb{R}$ . If  $s_{ij}^{\tau}$  is high,  $(i,j)$  is more probably a reaction atom pair, otherwise, less probably. Similar to (Jin et al., 2017), we use two different networks called "local" network and "global" network for this task. In case of the "local" network,  $s_{ij}^{\tau}$  is computed as:

$$
\boldsymbol {z} _ {i j} ^ {\tau} = \sigma \left(W _ {1} \left[ \boldsymbol {h} ^ {\tau - 1}, \left(\boldsymbol {x} _ {i} ^ {\tau} + \boldsymbol {x} _ {j} ^ {\tau}\right), \boldsymbol {e} _ {i j} \right] + b _ {1}\right) \tag {2}
$$

$$
s _ {i j} ^ {\tau} = f ^ {\text {a t o m p a i r}} \left(\boldsymbol {z} _ {i j} ^ {\tau}\right) \tag {3}
$$

where  $f^{\mathrm{atom~pair}}$  is a neural network;  $\sigma$  is a nonlinear activation function (e.g., ReLU); [.] denotes vector concatenation;  $W_{1}$  and  $b_{1}$  are parameters;  $h^{\tau - 1}$  is the hidden state of the RNN at the previous step; and  $e_{ij}$  is the representation vector of the bond between  $(i, j)$ . If there is no bond between  $(i, j)$  we assume that its bond type is "NULL". We consider  $z_{ij}$  as the representation vector for the atom pair  $(i, j)$ .

The "global" network leverages self-attention (Vaswani et al., 2017; Wang et al., 2018) to detect compatibility between atom  $i$  and all other atoms before computing the scores:

$$
\boldsymbol {r} _ {i j} ^ {\tau} = \sigma \left(V _ {1} \left[ \left(\boldsymbol {x} _ {i} ^ {\tau} + \boldsymbol {x} _ {j} ^ {\tau}\right), \boldsymbol {e} _ {i j} \right] + c _ {1}\right)
$$

$$
a _ {i j} ^ {\tau} = \operatorname {s o f t m a x} \left(V _ {2} r _ {i j} ^ {\tau} + c _ {2}\right)
$$

$$
\boldsymbol {c} _ {i} ^ {\tau} = \sum_ {j \in \mathcal {V}} a _ {i j} \boldsymbol {x} _ {j} ^ {\tau}
$$

$$
\boldsymbol {z} _ {i j} ^ {\tau} = \sigma \left(W _ {1} \left[ \boldsymbol {h} ^ {\tau - 1}, \left(\boldsymbol {x} _ {i} ^ {\tau} + \boldsymbol {x} _ {j} ^ {\tau}\right), \left(\boldsymbol {c} _ {i} ^ {\tau} + \boldsymbol {c} _ {j} ^ {\tau}\right), \boldsymbol {e} _ {i j} \right] + b _ {1}\right) \tag {4}
$$

$$
s _ {i j} ^ {\tau} = f ^ {\text {a t o m p a i r}} \left(z _ {i j} ^ {\tau}\right) \tag {5}
$$

where  $a_{ij}$  is the attention score from node  $i$  to every other node  $j$ ;  $c_i$  is the context vector of atom  $i$  that summarizes the information from all other atoms.

During experiments, we tried both options mentioned above and saw that the "global" network clearly outperforms the "local" network so we set the "global" network as a default module in our model. In addition, since reagents never change their form during a reaction, we explicitly exclude all atom pairs that have either atoms belong to the reagents. This leads to better results than not using reagent information. Detailed analyses are provided in Appendix A.5.

Top-  $K$  atom pairs Because the number of atom pairs that actually participate in a reaction is very small (usually smaller than 10) compared to the total number of atom pairs of the input molecules (usually hundreds or thousands), it is much more efficient to identify reaction triples from a small subset of highly probable reaction atom pairs. For that reason, we extract  $K$  ( $K \ll |\mathcal{V}|^2$ ) atom pairs with the highest scores. Later, we will predict reaction triples taken from these  $K$  atom pairs only. We denote the set of top- $K$  atom pairs, their corresponding scores, and representation vectors as  $\{(u_k, v_k) | k = \overline{1, K}\}$ ,  $\{s_{u_k v_k} | k = \overline{1, K}\}$  and  $Z_K = \{z_{u_k v_k} | k = \overline{1, K}\}$ , respectively.

# 2.2.3 POLICY NETWORK

Predicting continuation signal To account for varying number of transformation steps, PN generates a continuation signal  $\xi^{\tau} \in \{0,1\}$  to indicate whether prediction should continue or terminate.  $\xi^{\tau}$  is drawn from a Bernoulli distribution:

$$
p \left(\xi^ {\tau} = 1\right) = \text {s i g m o i d} \left(f ^ {\text {s i g n a l}} \left(\left[ h ^ {\tau - 1}, g \left(Z _ {K} ^ {\tau}\right) \right]\right)\right) \tag {6}
$$

where  $h^{\tau - 1}$  is the previous RNN state;  $Z_K^\tau$  is the set of representation vectors of the top  $K$  atom pairs at the current step;  $f^{\mathrm{signal}}$  is a neural network;  $g$  is a function that maps an unordered set of inputs to an output vector. For simplicity, we use a mean function:

$$
\boldsymbol {z} _ {K} ^ {\tau - 1} = g \left(Z _ {K} ^ {\tau}\right) = \frac {1}{K} \sum_ {k = 1} ^ {K} W \boldsymbol {z} _ {u _ {k} v _ {k}} ^ {\tau - 1}
$$

Predicting atom pair At the next sub-step, PN predicts which atom pair changes its bond during the reaction by sampling from the top-  $K$  atom pairs with probability:

$$
p \left(\left(u _ {k}, v _ {k}\right) ^ {\tau}\right) = \operatorname {s o f t m a x} _ {K} \left(s _ {u _ {k} v _ {k}} ^ {\tau}\right) \tag {7}
$$

where  $s_{u_k v_k}^\tau$  is the score of the atom pair  $(u_k, v_k)^\tau$  computed in Eq. (5).

Predicting bond type Given an atom pair  $(u,v)^{\tau}$  sampled from the previous sub-step, we predict a new bond type  $b^{\tau}$  between  $u$  and  $v$  to get a complete reaction triple  $(u,v,b)^{\tau}$  using the probability:

$$
p \left(b ^ {\tau} | (u, v) ^ {\tau}\right) = \operatorname {s o f t m a x} _ {B} \left(f ^ {\text {b o n d}} \left(\left[ h ^ {\tau - 1}, z _ {u v} ^ {\tau}, \left(e _ {b} - e _ {b ^ {\text {o l d}}}\right) \right]\right)\right) \tag {8}
$$

where  $B$  is the total number of bond types;  $\pmb{z}_{uv}^{\tau}$  is the representation vector of  $(u,v)^{\tau}$  computed in Eq. (4);  $b^{\mathrm{old}}$  is the old bond of  $(u,v)$ ;  $\pmb{e}_{b^{\mathrm{old}}}$  and  $\pmb{e}_b$  are the embedding vectors corresponding to the bond type  $b^{\mathrm{old}}$  and  $b$ , respectively; and  $f^{\mathrm{bond}}$  is a neural network.

# 2.3 UPDATING STATES

After predicting a complete reaction triple  $(u, v, b)^{\tau}$ , our model updates: i) the new recurrent hidden state  $h^{\tau}$ , and ii) the new node representation vectors  $\boldsymbol{x}_i^{\tau + 1}$  of the new intermediate graph  $\mathcal{G}^{\tau + 1}$  for  $i \in \mathcal{V}$ . These updates are presented in Appendix A.2.

# 2.4 TRAINING

Loss function plays a central role in achieving fast training and high performance. We design the following loss:

$$
\mathcal {L} = \lambda_ {1} \mathcal {L} ^ {\text {a t o m p a i r}} + \lambda_ {2} \mathcal {L} ^ {\mathrm {A 2 C}} + \lambda_ {3} \mathcal {L} ^ {\text {v a l u e}} + \lambda_ {4} \mathcal {L} ^ {\text {o v e r l e n g t h}} + \lambda_ {5} \mathcal {L} ^ {\text {i n t o p K}}
$$

where  $\mathcal{L}^{\mathrm{atom~pair}}$  accounts for binary change in the bond of an atom pair;  $\mathcal{L}^{\mathrm{A2C}}$  is the Advantage Actor-Critic (A2C) loss (Mnih et al., 2016) to account for the correct sequence of reaction triples;  $\mathcal{L}^{\mathrm{value}}$  is the loss for estimating the value function used in A2C;  $\mathcal{L}^{\mathrm{over~length}}$  penalizes long predicted sequences; and  $\mathcal{L}^{\mathrm{in~top~}K}$  is the rank loss to force a groundtruth reaction atom pair to appear in the top- $K$ ; and  $\lambda_1, \dots, \lambda_5 > 0$  are tunable coefficients. The component losses are explained in the following.

# 2.4.1 REACTION ATOM PAIR LOSS

This loss function is a cross-entropy loss:

$$
\mathcal {L} ^ {\text {a t o m p a i r}} = - \sum_ {i \in \mathcal {V}} \sum_ {j \in \mathcal {V}, j \neq i} \left(y _ {i j} \log p _ {i j} + \left(1 - y _ {i j}\right) \log \left(1 - p _ {i j}\right)\right) \tag {9}
$$

where  $y_{ij} \in \{0,1\}$  is the label indicating whether the atom pair  $(i,j)$  is a reaction atom pair or not;  $p_{ij} = \mathrm{sigmoid}(s_{ij})$  (see Eq. (5)).

# 2.4.2 REACTION TRIPLE LOSS

The loss follows a policy gradient method known as Advantage Actor-Critic (A2C):

$$
\begin{array}{l} \mathcal {L} ^ {\mathrm {A 2 C}} = - \sum_ {\tau = 0} ^ {T _ {\text {e n d}} - 1} \left(A _ {\text {s i g n a l}} ^ {\tau} \log p (\xi^ {\tau}) + A _ {\text {a t o m p a i r}} ^ {\tau} \log p ((u, v) ^ {\tau}) + A _ {\text {b o n d}} ^ {\tau} \log p (b ^ {\tau})\right) \\ - A _ {\text {s i g n a l}} ^ {T _ {\text {e n d}}} \log \pi (\xi^ {T _ {\text {e n d}}}) \tag {10} \\ \end{array}
$$

where  $T_{\mathrm{end}}$  is the first step that  $\xi = 0$ ;  $A_{\mathrm{signal}}$ ,  $A_{\mathrm{atom pair}}$  and  $A_{\mathrm{bond}}$  are called advantages. To compute these advantages, we use the unbiased estimations called Temporal Different errors, defined as:

$$
A _ {\text {s i g n a l}} ^ {\tau} = r _ {\text {s i g n a l}} ^ {\tau} + \gamma V _ {\phi} \left(Z _ {K} ^ {\tau + 1}\right) - V _ {\phi} \left(Z _ {K} ^ {\tau}\right) \tag {11}
$$

$$
A _ {\text {a t o m p a i r}} ^ {\tau} = r _ {\text {a t o m p a i r}} ^ {\tau} + \gamma V _ {\phi} \left(Z _ {K} ^ {\tau + 1}\right) - V _ {\phi} \left(Z _ {K} ^ {\tau}\right) \tag {12}
$$

$$
A _ {\text {b o n d}} ^ {\tau} = r _ {\text {b o n d}} ^ {\tau} + \gamma V _ {\phi} \left(Z _ {K} ^ {\tau + 1}\right) - V _ {\phi} \left(Z _ {K} ^ {\tau}\right) \tag {13}
$$

where  $r_{\mathrm{signal}}^{\tau}$ ,  $r_{\mathrm{atom\_pair}}^{\tau}$ ,  $r_{\mathrm{bond}}^{\tau}$  are immediate rewards at step  $\tau$ ; at the final step  $\tau = T_{\mathrm{end}}$ , the model receives additional delayed rewards;  $\gamma$  is the discount factor; and  $V_{\phi}$  is the parametric value function. We train  $V_{\phi}$  using the following mean square error loss:

$$
\mathcal {L} ^ {\text {v a l u e}} = \sum_ {\tau = 0} ^ {T _ {\text {e n d}}} \| V _ {\phi} \left(Z _ {K} ^ {\tau}\right) - R ^ {\tau} \| ^ {2} \tag {14}
$$

where  $R^{\tau}$  is the return at step  $\tau$

Episode termination during training Although the loss defined in Eq. (10) is correct, it is not good to use in practice because: i) If our model selects a wrong sub-action at any sub-step of the step  $T_{\mathrm{wrong}}$  ( $T_{\mathrm{wrong}} < T_{\mathrm{end}}$ ), the whole predicted sequence will be incorrect regardless of what will be predicted from  $T_{\mathrm{wrong}} + 1$  to  $T_{\mathrm{end}}$ . Therefore, computing the loss for actions from  $T_{\mathrm{wrong}} + 1$  to  $T_{\mathrm{end}}$  is redundant. ii) More importantly, the incorrect updates of the graph structure at subsequent steps from  $T_{\mathrm{wrong}} + 1$  to  $T_{\mathrm{end}}$  will lead to cumulative prediction errors which make the training of our model much more difficult.

To resolve this issue, during training, we use a binary vector  $\zeta \in \{0,1\}^{3T}$  to keep track of the first wrong sub-action:  $\zeta^t = \left\{ \begin{array}{ll} 1 & \text{if } t \leq t_{\text{first wrong}} \\ 0 & \text{if } t > t_{\text{first wrong}} \end{array} \right.$  where  $t_{\text{first wrong}}$  denotes the sub-step at which our model chooses a wrong sub-action the first time. The actor-critic loss in Eq. (10) now becomes:

$$
\mathcal {L} ^ {\mathrm {A 2 C}} = - \sum_ {\tau = 0} ^ {T} \left(\zeta^ {\tau} A _ {\text {s i g n a l}} ^ {\tau} \log p \left(\xi^ {\tau}\right) + \zeta^ {(\tau + 1)} A _ {\text {a t o m p a i r}} ^ {\tau} \log p \left((u, v) ^ {\tau}\right) + \zeta^ {(\tau + 2)} A _ {\text {b o n d}} ^ {\tau} \log p \left(b ^ {\tau}\right)\right) \tag {15}
$$

where  $T$  is the maximum number of steps. Similarly, we change the value loss into:

$$
\mathcal {L} ^ {\text {v a l u e}} = \sum_ {\tau = 0} ^ {T} \zeta^ {\tau} \left\| V _ {\phi} \left(Z _ {K} ^ {\tau}\right) - R ^ {\tau} \right\| ^ {2}
$$

# 2.4.3 CONSTRAINT ON THE SEQUENCE LENGTH

One major difficulty of the chemical reaction prediction problem is to know exactly when to stop prediction so we can make accurate inference. By forcing the model to stop immediately when making wrong prediction, we can prevent cumulative error and significantly reduce variance during training. But it also comes with a cost: The model cannot learn (because it does not have to learn) when to stop. This phenomenon can be visualized easily as the model predicts 1 for the signal at every step  $\tau$  during inference. In order to make the model aware of the correct sequence length during training, we define a loss that punishes the model if it produces a longer sequence than the ground truth sequence:

<table><tr><td colspan="2">Dataset</td><td>#reactions</td><td>#changes</td><td>#molecules</td><td>#atoms</td><td>#bonds</td></tr><tr><td rowspan="3">USPTO-15k</td><td>train</td><td>10,500</td><td>1 | 11 | 2.3</td><td>1 | 20 | 3.6</td><td>4 | 100 | 34.9</td><td>3 | 110 | 34.7</td></tr><tr><td>valid</td><td>1,500</td><td>1 | 11 | 2.3</td><td>1 | 20 | 3.6</td><td>7 | 94 | 34.5</td><td>5 | 99 | 34.2</td></tr><tr><td>test</td><td>3,000</td><td>1 | 11 | 2.3</td><td>1 | 16 | 3.6</td><td>7 | 98 | 34.9</td><td>5 | 102 | 34.7</td></tr><tr><td rowspan="3">USPTO</td><td>train</td><td>409,035</td><td>1 | 6 | 2.2</td><td>2 | 29 | 4.8</td><td>9 | 150 | 39.7</td><td>6 | 165 | 38.6</td></tr><tr><td>valid</td><td>30,000</td><td>1 | 6 | 2.2</td><td>2 | 25 | 4.8</td><td>9 | 150 | 39.6</td><td>7 | 158 | 38.5</td></tr><tr><td>test</td><td>40,000</td><td>1 | 6 | 2.2</td><td>2 | 22 | 4.8</td><td>9 | 150 | 39.8</td><td>7 | 162 | 38.7</td></tr></table>

Table 1: Statistics of USPTO-15k and USPTO datasets. "changes" means bond changes, "molecules" means reactants and reagents in a reaction; "atoms" and "bonds" are defined for a molecule. Apart from "#reactions", other columns are presented in the format "min | max | mean".

$$
\mathcal {L} ^ {\text {o v e r l e n g t h}} = - \sum_ {T _ {\text {e n d}} ^ {\mathrm {g t}} \leq \tau <   T _ {\text {e n d}}} \log p (\xi^ {\tau} = 0) \tag {16}
$$

where  $T_{\mathrm{end}}^{\mathrm{gt}}$  is the end step of the groundtruth sequence. Note that the loss in Eq. (16) is not applied when  $T_{\mathrm{end}} \leq T_{\mathrm{end}}^{\mathrm{gt}}$ . The reason is that forcing  $\xi^{\tau} = 1$  with  $T_{\mathrm{end}} \leq \tau < T_{\mathrm{end}}^{\mathrm{gt}}$  is not theoretically correct because all the signals after  $T_{\mathrm{end}}$  are assumed to be 0. The incentive to force  $T_{\mathrm{end}}$  close to  $T_{\mathrm{end}}^{\mathrm{gt}}$  when it is smaller than  $T_{\mathrm{end}}^{\mathrm{gt}}$  has already been included in the advantages in Eq. (15).

# 2.4.4 CONSTRAINT ON THE TOP-  $K$  ATOM PAIRS

Ideally, the loss from Eq. (9) pushes a reaction atom pair  $(\tilde{u},\tilde{v})^{\tau}$  into the top-  $K$  atom pairs at each step  $\tau < T_{\mathrm{end}}^{\mathrm{gt}}$ . However, this is not guaranteed, especially when  $\tau$  comes close to  $T_{\mathrm{end}}^{\mathrm{gt}}$ . To encourage the groundtruth reaction atom pair  $(\tilde{u},\tilde{v})^{\tau}$  with the highest score to appear in the top  $K$ , we introduce an additional rank-based loss:

$$
\mathcal {L} ^ {\text {i n} \operatorname {t o p} K} = - \sum_ {\tau = 0} ^ {T _ {\text {f i r s t w r o n g}}} \log p \left(\left(\tilde {u}, \tilde {v}\right) ^ {\tau} \text {i n} \operatorname {t o p} K\right)
$$

where  $T_{\mathrm{first~wrong}} = \left\lfloor \frac{t_{\mathrm{first~wrong}}}{3} \right\rfloor$ ; and  $p((\tilde{u},\tilde{v})^{\tau}$  in top  $K$ ) is computed as:

$$
p \left((\tilde {u}, \tilde {v}) ^ {\tau} \text {i n} \operatorname {t o p} K\right) = \frac {\exp \left(s _ {\tilde {u} \tilde {v}} ^ {\tau}\right)}{\exp \left(s _ {\tilde {u} \tilde {v}} ^ {\tau}\right) + \sum_ {k = 1} ^ {K} \exp \left(s _ {u _ {k} v _ {k}} ^ {\tau}\right)} \tag {17}
$$

# 3 EXPERIMENTS

# 3.1 DATASET

We evaluate our model on two standard datasets USPTO-15k (15K reactions) and USPTO (480K reactions) which have been used in previous works (Jin et al., 2017; Schwaller et al., 2018; Bradshaw et al., 2018). Details about these datasets are given in Table 1.

# 3.2 REACTION ATOM PAIR PREDICTION

In this section, we test our model's ability to identify reaction atom pairs by formulating it as a ranking problem with the scores computed in Eq. (5). Similar to (Jin et al., 2017), we use Coverage@  $k$  as the evaluation metric, which is the proportion of reactions that have all groundtruth reaction atom pairs appear in the top  $k$  predicted atom pairs.

We compare our proposed graph neural network (GNN) with Weisfeiler-Lehman Network (WLN) (Jin et al., 2017) and Column Network (CLN) (Pham et al., 2017). Since our GNN explicitly uses reagent information to compute the scores of atom pairs, we modify the implementation of WLN and CLN accordingly for fair comparison. From Table 2, we observe that our GNN clearly outperforms WLN and CLN in all cases. We attribute this improvement to the use of a separate node state vector

<table><tr><td rowspan="2">Model</td><td colspan="3">USPTO-15k</td><td colspan="3">USPTO</td></tr><tr><td>C@6</td><td>C@8</td><td>C@10</td><td>C@6</td><td>C@8</td><td>C@10</td></tr><tr><td>\( WLN^* \) (Jin et al., 2017)</td><td>81.6</td><td>86.1</td><td>89.1</td><td>89.8</td><td>92.0</td><td>93.3</td></tr><tr><td>WLN (Jin et al., 2017)</td><td>88.45</td><td>91.65</td><td>93.34</td><td>90.97</td><td>93.98</td><td>95.26</td></tr><tr><td>CLN (Pham et al., 2017)</td><td>88.68</td><td>91.63</td><td>93.07</td><td>90.72</td><td>93.57</td><td>94.80</td></tr><tr><td>Our GNN</td><td>88.92</td><td>92.00</td><td>93.57</td><td>91.24</td><td>94.17</td><td>95.33</td></tr></table>

Table 2: Results for reaction atom pair prediction.  $C@k$  is coverage at  $k$ . Best results are highlighted in bold. WLN* is the original model from (Jin et al., 2017) while WLN is our re-implemented version. Except for WLN*, other models explicitly use reagent information.

![](images/c5d92f5385c626f8dbc9d35699ca67b9524319417c27ed8025de5db11f6b7095.jpg)  
Figure 3: Coverage@k and Recall@k with respect to  $k$  for the USPTO dataset.

$\pmb{x}_i^t$  (different from the node feature vector  $\pmb{v}_i$ ) for updating the structural information of a node (see Eq. (21)). The other two models, on the other hand, only use a single vector to store both the node features and structure, hence, some information may be lost. In addition, using explicit reagent information boosts the prediction accuracy, which improves the WLN by  $1 - 7\%$  depending on the metrics. The presence of reagent information reduces the number of atom pairs to be searched on and contributes to the likelihood of reaction atom pairs. Further results are presented in Appendix A.5.

# 3.3 TOP-K ATOM PAIR EXTRACTION

The performance of our model depends on the number of selected top atom pairs  $K$ . The value of  $K$  presents a trade-off between coverage and efficiency. In addition to the metric Coverage@k in Sec. 3.2, we use Recall@k which is the proportion of correct atom pairs that appear in top  $k$  to find the good  $K$ . Fig. 3 shows Coverage@k and Recall@k for the USPTO dataset with respect to  $k$ . We see that both curves increase rapidly when  $k < 10$  and stabilize when  $k > 10$ . We also ran experiments with  $k = 10, 15, 20$  and observed that their prediction results are quite similar. Hence, in what follows we select  $K = 10$  for efficiency.

# 3.4 REACTION PRODUCT PREDICTION

This experiment validates GTPN on full reaction product prediction against the recent state-of-the-art methods (Jin et al., 2017; Schwaller et al., 2018) using the accuracy metric. The recent method ELECTRO (Bradshaw et al., 2018) is not compatible here because it was only evaluated on a subset of USPTO limited to linear chain topology. Comparison against ELECTRO is reported separately in Appendix A.6. Table 3 shows the prediction results. We produce multiple reaction product candidates by using beam search decoding with beam width  $N = 20$ . Details about beam search and its behaviors are presented in Appendix A.4.

In brief, we compute the normalized-over-length log probabilities of  $N$  predicted sequences of reaction triples and sort these values in descending order to get a rank list of  $N$  possible reaction

<table><tr><td rowspan="2">Model</td><td colspan="3">USPTO-15k</td><td colspan="3">USPTO</td></tr><tr><td>P@1</td><td>P@3</td><td>P@5</td><td>P@1</td><td>P@3</td><td>P@5</td></tr><tr><td>WLDN (Jin et al., 2017)</td><td>76.7</td><td>85.6</td><td>86.8</td><td>79.6</td><td>87.7</td><td>89.2</td></tr><tr><td>Seq2Seq (Schwaller et al., 2018)</td><td>-</td><td>-</td><td>-</td><td>80.3*</td><td>86.2*</td><td>87.5*</td></tr><tr><td>GTPN</td><td>72.31</td><td>-</td><td>-</td><td>71.26</td><td>-</td><td>-</td></tr><tr><td>GTPN◇</td><td>74.56</td><td>82.62</td><td>84.23</td><td>73.25</td><td>80.56</td><td>83.53</td></tr><tr><td>GTPN◇♣</td><td>74.56</td><td>83.19</td><td>84.97</td><td>73.25</td><td>84.31</td><td>85.76</td></tr><tr><td>GTPN◇♠</td><td>82.39</td><td>85.60</td><td>86.68</td><td>83.20</td><td>84.97</td><td>85.90</td></tr><tr><td>GTPN◇♠♣</td><td>82.39</td><td>85.73</td><td>86.78</td><td>83.20</td><td>86.03</td><td>86.48</td></tr></table>

Table 3: Results for reaction prediction.  $P@k$  is precision at  $k$ . State-of-the-art results from (Jin et al., 2017) are written in italic. Results from (Schwaller et al., 2018) are marked with  $^*$  and they are computed on a slightly different version of USPTO that contains only single-product reactions. Best results are highlighted in bold.  $\diamond$ : With beam search (beam width = 20),  $\spadesuit$ : Invalid product removal,  $\clubsuit$ : Duplicated product removal.

outcomes. Given a predicted sequence of reaction triples  $(u, v, b)^{0:T}$ , we can generate reaction products from input reactants simply by replacing the old bond of  $(u, v)^{\tau}$  with  $b^{\tau}$ . However, these products are not guaranteed to be valid (e.g., maximum valence constraint violation or aromatic molecules cannot be kekulized) so we post-process the outputs by removing all invalid products. The removal increases the top-1 accuracy by about  $8\%$  and  $10\%$  on USPTO-15k and USPTO, respectively. Due to the permutation invariance of the predicted sequence of reaction triples, some product candidates are duplicate and will also be removed. This does not lead to any change in  $P@1$  but slightly improves  $P@3$  and  $P@5$  by about  $0.5 - 1\%$  on the two datasets.

Overall, GTPN with beam search and post-processing convincingly beats both WLDN (Jin et al., 2017) and Seq2Seq (Schwaller et al., 2018) in the top-1 accuracy. For the top-3 and top-5, our model's performance is comparable to WLDN's on USPTO-15k and is worse than WLDN's on USPTO. It is not surprising since our model is trained to accurately predict the top-1 outcomes instead of ranking the candidates directly like WLDN. It is important to emphasize that we did not tune the model hyper-parameters when training on USPTO but reused the optimal settings from USPTO-15k (which is 25 times smaller than USPTO) so the results may not be optimal (see Appendix A.3 for more training detail).

# 4 RELATED WORK

# 4.1 LEARNING TO PREDICT CHEMICAL REACTION

In chemical reaction prediction, machine learning has replaced rule-based methods (Chen & Baldi, 2009) for better generalizability and scalability. Existing machine learning-based techniques are either template-free (Kayala & Baldi, 2011; Jin et al., 2017; Fooshee et al., 2018) and template-based (Wei et al., 2016; Segler & Waller, 2017; Coley et al., 2017). Both groups share the same mechanism: running multiple stages with the aid of reaction templates or rules. For example, in (Wei et al., 2016) the authors proposed a two-stage model that first classifies reactions into different types based on the neural fingerprint vectors (Duvenaud et al., 2015) of reactant and reagent molecules. Then, it applies pre-designed SMARTS transformation on the reactants with respect to the most suitable predicted reaction type to generate the reaction products.

The work of (Jin et al., 2017) treats a reaction as a set of bond changes so in the first step, they predict which atom pairs are likely to be reactive using a variant of graph neural networks called Weisfeiler-Lehman Networks (WLN). In the next step, they do almost the same as (Coley et al., 2017) by modifying the bond type between the selected atom pairs (with chemical rules satisfied) to create product candidates and rank them (with reactant molecules as addition input) using another kind of WLN called Weifeiler-Lehman Different Networks (WLDN).

To the best of our knowledge, (Jin et al., 2017) is the first work that achieves remarkable results (with the Precision@1 is about  $79.6\%$ ) on the large USPTO dataset containing more than 480 thousands reactions. Works in (Nam & Kim, 2016) and (Schwaller et al., 2018) avoid multi-stage prediction by

building a seq2seq model that generates the (canonical) SMILES string of the single product from the concatenated SMILES strings of the reactants and reagents in an end-to-end manner. However, their methods cannot deal with sets of reactants/reagents/products properly as well as cannot provide concrete reaction mechanism for every reaction.

The most recent work on this topic is (Bradshaw et al., 2018) which solves the reaction prediction problem by predicting a sequence of bond changes given input reactants and reagents represented as graphs. To handle ordering, they only select reactions with predefined topology. Our method, by contrast, is order-free and can be applied to almost any kind of reactions.

# 4.2 GRAPH NEURAL NETWORKS FOR MODELING MOLECULES

In recent years, there has been a fast development of graph neural networks (GNNs) for modeling molecules. These models are proposed to solve different problems in chemistry including toxicity prediction (Duvenaud et al., 2015), drug activity classification (Shervashidze et al., 2011; Dai et al., 2016; Pham et al., 2018), protein interface prediction (Fout et al., 2017) and drug generation (Simonovsky & Komodakis, 2018; Jin et al., 2018). Most of them can be regarded as variants of message-passing graph neural networks (MPGNNs) (Gilmer et al., 2017).

# 4.3 REINFORCEMENT LEARNING FOR STRUCTURAL REASONING

Reinforcement learning (RL) has become a standard approach to many structural reasoning problems because it allows agents to perform discrete actions. A typical example of using RL for structural reasoning is drug generation (Li et al., 2018; You et al., 2018). Both (Li et al., 2018) and (You et al., 2018) learn the same generation policy whose action set including: i) adding a new atom or a molecular scaffold to the intermediate graph, ii) connecting existing pair of atoms with bonds, and iii) terminating generation. However, (You et al., 2018) uses an adversarial loss to enforce global chemical constraints on the generated molecules as a whole instead of using the common reconstruction loss as in (Li et al., 2018). Other examples are path-based relational reasoning in knowledge graphs (Das et al., 2018; Xiong et al., 2017) and learning combinatorial optimization over graphs (Khalil et al., 2017).

# 5 DISCUSSION

We have introduced a novel method named Graph Transformation Policy Network (GTPN) for predicting products of a chemical reaction. GTPN uses graph neural networks to represent input reactant and reagent molecules, and uses reinforcement learning to find an optimal sequence of bond changes that transforms the reactants into products. We train GTPN using the Advantage Actor-Critic (A2C) method with appropriate constraints to account for notable aspects of chemical reaction. Experiments on real datasets have demonstrated the competitiveness of our model.

Although the GTPN was proposed to solve the chemical reaction problem, it is indeed generic to solve the graph transformation problem, which can be useful in reasoning about relations (e.g., see (Zambaldi et al., 2018)) and changes in relation. Open rooms include addressing dynamic graphs over time, extending toward full chemical planning and structural reasoning using RL.

# REFERENCES

Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Jimenez Rezende, et al. Interaction networks for learning about objects, relations and physics. In Advances in neural information processing systems, pp. 4502-4510, 2016.  
John Bradshaw, Matt J Kusner, Brooks Paige, Marwin HS Segler, and José Miguel Hernández-Lobato. Predicting electron paths. arXiv preprint arXiv:1805.10970, 2018.  
Jonathan H Chen and Pierre Baldi. No electron left behind: a rule-based expert system to predict chemical reactions and reaction mechanisms. Journal of chemical information and modeling, 49 (9):2034-2043, 2009.

Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP, 2014.  
Connor W Coley, Regina Barzilay, Tommi S Jaakkola, William H Green, and Klavs F Jensen. Prediction of organic reaction outcomes using machine learning. ACS central science, 3(5): 434-443, 2017.  
Hanjun Dai, Bo Dai, and Le Song. Discriminative embeddings of latent variable models for structured data. In International Conference on Machine Learning, pp. 2702-2711, 2016.  
Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, Luke Vilnis, Ishan Durugkar, Akshay Krishnamurthy, Alex Smola, and Andrew McCallum. Go for a walk and arrive at the answer: Reasoning over paths in knowledge bases using reinforcement learning. *ICLR*, 2018.  
David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. In Advances in Neural Information Processing Systems, pp. 2224-2232, 2015.  
David Fooshee, Aaron Mood, Eugene Gutman, Mohammadamin Tavakoli, Gregor Urban, Frances Liu, Nancy Huynh, David Van Vranken, and Pierre Baldi. Deep learning for chemical reaction prediction. Molecular Systems Design & Engineering, 2018.  
Alex Fout, Jonathon Byrd, Basir Shariat, and Asa Ben-Hur. Protein interface prediction using graph convolutional networks. In Advances in Neural Information Processing Systems, pp. 6530-6539, 2017.  
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In Proceedings of the International Conference on Machine Learning, 2017.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Proceedings of Advances in Neural Information Processing Systems, pp. 1025-1035, 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Wengong Jin, Connor Coley, Regina Barzilay, and Tommi Jaakkola. Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network. In Advances in Neural Information Processing Systems, pp. 2604-2613, 2017.  
Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. International Conference on Machine Learning (ICML), 2018.  
Matthew A Kayala and Pierre F Baldi. A machine learning approach to predict chemical reactions. In Advances in Neural Information Processing Systems, pp. 747-755, 2011.  
Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial optimization algorithms over graphs. In Advances in Neural Information Processing Systems, pp. 6348-6358, 2017.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR), 2015.  
Yibo Li, Liangren Zhang, and Zhenming Liu. Multi-objective de novo drug design with conditional graph generative model. Journal of Cheminformatics, 10, 2018.  
Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pp. 1928-1937, 2016.  
Juno Nam and Jurae Kim. Linking the neural machine translation and the prediction of organic chemistry reactions. arXiv preprint arXiv:1612.09529, 2016.

Trang Pham, Truyen Tran, Dinh Phung, and Svetha Venkatesh. Column networks for collective classification. In Proceedings of AAAI Conference on Artificial Intelligence, 2017.  
Trang Pham, Truyen Tran, and Svetha Venkatesh. Graph memory networks for molecular activity prediction. ICPR, 2018.  
Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. Modeling relational data with graph convolutional networks. 15th European Semantic Web Conference (ESWC-18), 2018.  
Philippe Schwaller, Theophile Gaudin, David Lanyi, Costas Bekas, and Teodoro Laino. "found in translation": Predicting outcome of complex organic chemistry reactions using neural sequence-to-sequence models. Chemical Science, 9:6091-6098, 2018.  
Marwin HS Segler and Mark P Waller. Neural-symbolic machine learning for retrosynthesis and reaction prediction. Chemistry-A European Journal, 23(25):5966-5971, 2017.  
Nino Shervashidze, Pascal Schweitzer, Erik Jan van Leeuwen, Kurt Mehlhorn, and Karsten M Borgwardt. Weisfeiler-Lehman graph kernels. Journal of Machine Learning Research, 12(Sep): 2539-2561, 2011.  
Martin Simonovsky and Nikos Komodakis. GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders. arXiv preprint arXiv:1802.03480, 2018.  
Rupesh K Srivastava, Klaus Greff, and Jürgen Schmidhuber. Training very deep networks. In Advances in neural information processing systems, pp. 2377-2385, 2015.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pp. 5998-6008, 2017.  
Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.  
Jennifer N Wei, David Duvenaud, and Alán Aspuru-Guzik. Neural networks for the prediction of organic chemistry reactions. ACS Central Science, 2(10):725-732, 2016.  
Wenhan Xiong, Thien Hoang, and William Yang Wang. DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 564-573, 2017.  
Jiaxuan You, Bowen Liu, Rex Ying, Vijay Pande, and Jure Leskovec. Graph convolutional policy network for goal-directed molecular graph generation. NIPS, 2018.  
Vinicius Zambaldi, David Raposo, Adam Santoro, Victor Bapst, Yujia Li, Igor Babuschkin, Karl Tuyls, David Reichert, Timothy Lillicrap, Edward Lockhart, et al. Relational deep reinforcement learning. arXiv preprint arXiv:1806.01830, 2018.
