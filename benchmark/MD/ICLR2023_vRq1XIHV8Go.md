# GRAPH NEURAL BANDITS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Contextual bandits aim to choose the optimal arm with the highest reward out of a set of candidates based on their contextual information, and various bandit algorithms have been applied to personalized recommendation due to their ability of solving the exploitation-exploration dilemma. Motivated by online recommendation scenarios, in this paper, we propose a framework named Graph Neural Bandits (GNB) to leverage the collaborative nature among users empowered by graph neural networks (GNNs). Instead of estimating rigid user clusters, we model the "fine-grained" collaborative effects through estimated user graphs in terms of exploitation and exploration individually. Then, to refine the recommendation strategy, we utilize separate GNN-based models on estimated user graphs for exploitation and adaptive exploration. Theoretical analysis and experimental results on multiple real data sets in comparison with state-of-the-art baselines are provided to demonstrate the effectiveness of our proposed framework.

# 1 INTRODUCTION

Contextual bandits are a specific type of multi-armed bandit problem where the additional contextual information (contexts) related to arms are available at each round, and the learner intends to refine its selection strategy based on the received arm contexts and rewards. Various contextual bandit algorithms have been applied in real-world recommendation tasks, such as online content recommendation and advertising (Li et al., 2010; Wu et al., 2016), and clinical trials (Durand et al., 2018; Villar et al., 2015). Meanwhile, collaborative effects among users provide us the opportunity to design better recommender strategies, since the target user's preference can be inferred based on other similar users. Such effects have been studied by many bandit works (Gentile et al., 2014; Li et al., 2019; Gentile et al., 2017; Li et al., 2016; Ban & He, 2021). Different from the conventional collaborative filtering methods (He et al., 2017; Wang et al., 2019), bandit-based approaches focus on more dynamic environments (such as news, short-video platform) and the exploitation-exploration dilemma inherently existed in the decisions of recommendation.

Existing works for clustering of bandits (Gentile et al., 2014; Li et al., 2019; Gentile et al., 2017; Li et al., 2016; Ban & He, 2021; Ban et al., 2022a) have been proposed to model the user correlations (collaborative effects) by clustering users into rigid groups, and assigning each formed group with an estimator to learn the assumed reward functions combined with an Upper Confidence Bound (UCB) strategy for exploration. However, these works only consider the "coarse-grained" user correlations. To be specific, they assume that users from the same group would share identical preferences, i.e., the users from the same group are compelled to make equal contributions to the final decision (arm selection) with regard to the target user. Such formulation of user correlations ("coarse-grained" collaborative effects), evidently fails to comply with real-world application scenarios, since users within the same group tend to have similar but subtly different preferences instead of sharing completely identical tastes. Therefore, given a target user, it is more practical to assume that the rest of the users would impose different levels of (collaborative) effects on this user.

Motivated by aforementioned limitations of existing works, in this paper, we propose a novel framework, named Graph Neural Bandits (GNB), to formulate the "fine-grained" collaborative effects, where the correlation of each user pair is preserved by user graphs. Given a target user, other users are allowed to make different contributions to the final decision based on the strength of their correlation to the target user, which therefore corresponds to the "fine-grained" collaborative effects. In particular, in GNB, we propose a novel approach to construct two kinds of user graphs with distinct purposes, called "user exploitation graphs" and "user exploration graphs". Then, we apply

two separate graph neural network (GNN) models on these two kinds of user graphs, to incorporate the collaborative effects for both exploitation and exploration purposes in the final decision-making process. Our main contributions can be summarized as follows:

1. Different from existing works that only formulate the "coarse-grained" collaborative effects by neglecting the divergence within user groups, we introduce a new problem setting to model the "fine-grained" user collaborative effects via user graphs. In our setting, the pair-wise user correlations are preserved to contribute differently to the decision-making.  
2. We propose a framework named GNB, which has the novel ways to build two kinds of user graphs with two different purposes, i.e., exploitation and adaptive exploration, respectively. Then, GNB utilizes GNN-based models for a refined arm selection strategy by leveraging the user correlations encoded in these two kinds of user graphs.  
3. With standard assumptions, we provide the theoretical analysis showing that GNB can achieve the regret upper bound of complexity  $\mathcal{O}(\sqrt{T\log(Tn)})$ , where  $T$  is the number of rounds and  $n$  is the number of users. This bound is sharper than the existing related works.  
4. Extensive experiments comparing GNB with nine state-of-the-art algorithms are conducted on various real data sets, which demonstrate the effectiveness of our proposed method.

After introducing the problem definition in Section 2, we provide the details of our proposed framework in Section 3. Then, we present the theoretical analysis in Section 4, and the experiments in Section 5. Finally, we conclude the paper in Section 6. Due to page limit, we will leave the review of related works to the Section A in the Appendix.

# 2 GRAPH NEURAL BANDITS: PROBLEM DEFINITION AND NOTATION

Suppose there are a total of  $n$  users with the user set  $\mathcal{U} = \{1, \dots, n\}$ . At each time step  $t \in [T]$ , the learner will receive a user  $u_t \in \mathcal{U}$  to serve. Then, as the arm pool is not fixed, we use  $\mathcal{X}_t = \{\pmb{x}_{i,t}\}_{i \in [a]}$  to denote the set of candidate arms for recommendation in round  $t$ . The volume of this arm set is  $|\mathcal{X}_t| = a$ , and each arm is described by a  $d$ -dimensional context vector  $\pmb{x}_{i,t} \in \mathbb{R}^d$  with  $\| \pmb{x}_{i,t} \|_2 = 1$ . Meanwhile, each arm  $\pmb{x}_{i,t}$  is associated with a reward  $r_{i,t}$ . As the user correlation is one important factor in determining the reward, we define the following reward function:

$$
r _ {i, t} = h \left(\boldsymbol {x} _ {i, t}, u _ {t}, \boldsymbol {\Lambda} _ {i, t} ^ {*}\right) + \epsilon_ {i, t} \tag {1}
$$

where  $h(\cdot)$  is the unknown reward mapping function, and  $\epsilon_{i,t}$  stands for some zero-mean noise such that  $\mathbb{E}[r_{i,t}] = h(\pmb{x}_{i,t}, u_t, \pmb{\Lambda}_{i,t}^*)$ . Motivated by various real applications (e.g., online recommendation with normalized ratings), we consider  $r_{i,t}$  to be bounded  $r_{i,t} \in [0,1]$  in this paper, which is standard in existing works (e.g., Gentile et al. (2014; 2017); Ban & He (2021); Ban et al. (2022a)). Note that as long as  $r_{i,t} \in [0,1]$ , we do not need any distribution assumption (e.g., sub-Gaussian) on noise  $\epsilon_{i,t}$ .

Here, the unknown user affinity matrix  $\Lambda_{i,t}^{*}\in \mathbb{R}^{n\times n}$  encodes the user correlations w.r.t. the arm  $x_{i,t}$ . Under real-world application scenarios, the users sharing the same preference for specific arms (e.g., sports news) may have different tastes over other arms (e.g., political news). Therefore, inspired by this phenomenon, we allow each arm  $x_{i,t}\in \mathcal{X}_t$  to induce different user collaborations  $\Lambda_{i,t}^{*}$ .

Comparison with Existing Problem Definitions. The problem definition of existing user clustering works (e.g., Gentile et al. (2014); Li et al. (2019); Gentile et al. (2017); Ban & He (2021); Ban et al. (2022a)) only can formulate "coarse-grained" user correlations. In their settings, given a user group  $\mathcal{N} \subseteq \mathcal{U}$ , all the users in  $\mathcal{N}$  are forced to share the same reward function given an arm  $x_{i,t}$ , i.e.,  $\mathbb{E}[r_{i,t} \mid u, x_{i,t}] = h_{\mathcal{N}}(x_{i,t}), \forall u \in \mathcal{N}$ . In contrast, our definition of the reward function enables us to model the pair-wise fine-grained user correlations by introducing another two important factors  $u$  and  $\Lambda_{i,t}^{*}$ . With our formulation, each user here is allowed to produce different rewards facing the same arm, i.e.,  $\mathbb{E}[r_{i,t} \mid u, x_{i,t}] = h(x_{i,t}, u, \Lambda_{i,t}^{*}), \forall u \in \mathcal{N}$ . Here, with different users  $u$ , the corresponding expected reward  $h(x_{i,t}, u, \Lambda_{i,t}^{*})$  can be different. Therefore, our definition of the reward function is more generic, and it can also readily generalize to above user clustering algorithms (with "coarse-grained" user correlations), by allowing the affinity matrix  $\Lambda_{i,t}^{*}$  to be a block matrix where each block corresponds to a single user group.

To bridge user collaborative effects with user preferences (rewards), we consider the following constrain for the reward function in Eq. 1. The intuition is that for any two users with comparable user correlations, they would share similar tastes over the items with a high probability. For arm  $x_{i,t}$

we consider the difference of expected rewards between any two users  $u, u' \in \mathcal{U}$  to be governed by

$$
\left| h \left(\boldsymbol {x} _ {i, t}, u, \boldsymbol {\Lambda} _ {i, t} ^ {*}\right) - h \left(\boldsymbol {x} _ {i, t}, u ^ {\prime}, \boldsymbol {\Lambda} _ {i, t} ^ {*}\right) \right| \leq \Psi \left(\boldsymbol {\Lambda} _ {i, t} ^ {*} [ u,: ], \boldsymbol {\Lambda} _ {i, t} ^ {*} [ u ^ {\prime}, : ]\right) \tag {2}
$$

where  $\Lambda_{i,t}^{*}[u,:]$  is the user correlation vector (i.e., the corresponding row in  $\Lambda_{i,t}^{*}$ ) of user  $u$ , and  $\Psi: \mathbb{R}^n \times \mathbb{R}^n \mapsto \mathbb{R}$  denotes an unknown mapping function. The reward function definition and the constraint (Eq. 1-2) motivate us to design the GNB framework, to be introduced in Section 3.

Modeling User Correlations with User Graphs. In order to model the unknown user correlations  $(\Lambda_{i,t}^{*}$  from Eq. 1) and deal with the exploration-exploitation dilemma, for each candidate arm  $x_{i,t}\in \mathcal{X}_t$  , we propose to formulate two user correlation graphs: a user exploitation graph  $\mathcal{G}_{i,t}^{(1),*} =$ $(V,E,W_{i,t}^{(1),*})$  and a user exploration graph  $\mathcal{G}_{i,t}^{(2),*} = (V,E,W_{i,t}^{(2),*})$  . The defined arm-specific user graphs correspond to our formulation in Eq. 1 where each arm can induce different user collaboration effects. Here, the user exploitation graph  $\mathcal{G}_{i,t}^{(1),*}$  encodes the collaborative effects in terms of user preferences towards arm  $x_{i,t}$  , which makes effective use of the information in  $\Lambda_{i,t}^{*}$  (exploitation). Then, we formulate the user exploration graph  $\mathcal{G}_{i,t}^{(2),*}$  to model the user correlation regarding the uncertainty of reward estimation (exploration) from the reward prediction model.

For both kinds of user graphs, each user from  $\mathcal{U}$  is mapped to a corresponding node in node set  $V$ . With  $E = \{e(c_i, c_j)\}_{\forall c_i, c_j \in \mathcal{X}}$  being the set of edges, we have  $W_{i,t}^{(1),*}, W_{i,t}^{(2),*}$  to respectively represent the set of edge weights for  $\mathcal{G}_{i,t}^{(1),*}, \mathcal{G}_{i,t}^{(2),*}$ . Here, the estimated user (exploitation/exploration) correlations are modeled by the edge weights of node (user) pairs. Next, we proceed to give the definitions of two arm-specific user correlations, which are encoded by  $\mathcal{G}_{i,t}^{(1),*}, \mathcal{G}_{i,t}^{(2),*}$  respectively.

Definition 1 (User Correlation for Exploitation). In round  $t$ , for any two users  $u, u' \in \mathcal{U}$ , their exploitation correlation score  $w_{i,t}^{(1),*}(u, u')$  w.r.t. a candidate arm  $x_{i,t} \in \mathcal{X}_t$  is defined as

$$
w _ {i, t} ^ {(1), \ast} (u, u ^ {\prime}) = \Psi^ {(1)} \left(\mathbb {E} [ r _ {i, t} | u, \boldsymbol {x} _ {i, t} ], \mathbb {E} [ r _ {i, t} | u ^ {\prime}, \boldsymbol {x} _ {i, t} ]\right)
$$

where  $\mathbb{E}[r_{i,t}|u,\pmb{x}_{i,t}],i\in [a]$  is the expected reward in terms of the user-arm pair  $(u,\pmb{x}_{i,t})$ . Given two users  $u,u^{\prime}\in \mathcal{U}$ , the function  $\Psi^{(1)}:\mathbb{R}\times \mathbb{R}\mapsto \mathbb{R}$  maps from their expected rewards  $\mathbb{E}[r_{i,t}|u,\pmb{x}_{i,t}]$  to their user exploitation score  $w_{i,t}^{(1),*}(u,u')$ .

Given an arm  $\boldsymbol{x}_{i,t} \in \mathcal{X}_t$ , the user correlation for exploitation measures the user preference (i.e., expected reward) correlation between two users  $u, u' \in \mathcal{U}$ , and the corresponding exploitation score  $w_{i,t}^{(1),*}(u, u')$  refers to the edge weight between these two users (nodes)  $u, u'$  in exploitation graph  $\mathcal{G}_{i,t}^{(1),*}$ . Inspired by Ban et al. (2022b), before defining the second kind of user correlation (i.e., user exploration correlation), we first introduce the definition of expected potential gain for reward estimation, which measures the prediction uncertainty of reward estimators.

Definition 2 (Expected Potential Gain). Given user  $u \in \mathcal{U}$  at time step  $t$ , given a candidate arm  $\pmb{x}_{i,t} \in \mathcal{X}_t$ ,  $i \in [a]$  and a reward estimation function  $f_u(\cdot)$  corresponding to user  $u$ , the expected potential gain for the reward estimation  $f_u(\pmb{x}_{i,t})$  is defined as  $\mathbb{E}[r_{i,t}|u, \pmb{x}_{i,t}] - f_u(\pmb{x}_{i,t})$ .

Here, the potential gain for reward estimation essentially formulates the uncertainty of model  $f_{u}(\cdot)$  by measuring the difference between the expected reward  $\mathbb{E}[r_{i,t}|u,\pmb{x}_{i,t}]$  and the prediction  $f_{u}(\pmb{x}_{i,t})$ . Next, we proceed to introduce the second kind of user correlation, i.e., user exploration correlation.

Definition 3 (User Correlation for Exploration). In round  $t$ , given two users  $u, u' \in \mathcal{U}$  and an arm  $x_{i,t} \in \mathcal{X}_t$ , their underlying exploration correlation score  $w_{i,t}^{(2),*}(u, u')$  is

$$
w _ {i, t} ^ {(2), *} (u, u ^ {\prime}) = \Psi^ {(2)} \big (\mathbb {E} [ r _ {i, t} | u, \boldsymbol {x} _ {i, t} ] - f _ {u} (\boldsymbol {x} _ {i, t}), \mathbb {E} [ r _ {i, t} | u ^ {\prime}, \boldsymbol {x} _ {i, t} ] - f _ {u ^ {\prime}} (\boldsymbol {x} _ {i, t}) \big)
$$

with  $\mathbb{E}[r_{i,t}|u,\pmb{x}_{i,t}] - f_u(\pmb{x}_{i,t}),i\in [a]$  being the potential gain for the user-arm pair  $(u,\pmb{x}_{i,t})$ . Here,  $f_{u}(\cdot)$  is the reward estimation function specified to user  $u$ , and  $\Psi^{(2)}:\mathbb{R}\times \mathbb{R}\mapsto \mathbb{R}$  is the mapping from user potential gains  $\mathbb{E}[r_{i,t}|u,\pmb{x}_{i,t}] - f_u(\pmb{x}_{i,t})$  to their exploration correlation score.

For the arm  $\boldsymbol{x}_{i,t}$  and two users  $u, u' \in \mathcal{U}$ , the user exploration correlation score  $w_{i,t}^{(2),*}(u, u')$  refers to the correlation of prediction uncertainty between two user-specific functions  $f_u(\cdot)$  and  $f_{u'}(\cdot)$ .

![](images/4585915245882587c95ec38af2cc9b58107069b89803333740008f9cdee16d42.jpg)  
Figure 1: Workflow of the proposed Graph Neural Bandits (GNB) framework.

Then, the exploration score  $w_{i,t}^{(2),*}(u,u')$  will be considered as the edge weight between these two nodes (users)  $u, u'$  in the true user exploration graph  $\mathcal{G}_{i,t}^{(2),*}$ . Intuitively, when the exploration score  $w_{i,t}^{(2),*}(u,u')$  is high, we can apply similar exploration strategies for both users  $u, u'$ . For example, given arm  $x_{i,t}$ , if the reward estimation error (i.e., prediction uncertainty) is large for both  $u$  and  $u'$ , we may want to explore these two user-arm pairs  $(u,x_{i,t})$ ,  $(u',x_{i,t})$  more for additional knowledge. In this paper, we consider the mapping functions  $\Psi^{(1)}, \Psi^{(2)}$  as the prior knowledge, which can be functions such as the radial basis function (RBF) kernel or normalized absolute difference in practice.

Learning Objective. For the received user  $u_{t}$  in round  $t$ , the learner is expected to recommend an arm  $x_{t} \in \mathcal{X}_{t}$  (with reward  $r_{t}$ ) in order to minimize the cumulative pseudo-regret  $R(T) = \mathbb{E}\left[\sum_{t=1}^{T}(r_{t}^{*} - r_{t})\right]$  where  $r_{t}^{*}$  is the reward for the optimal arm  $\mathbb{E}[r_{t}^{*}|u_{t},\mathcal{X}_{t}] = \max_{\boldsymbol{x}_{i,t} \in \mathcal{X}_{t}} h(\boldsymbol{x}_{i,t},u_{t},\boldsymbol{\Lambda}_{i,t}^{*})$ .

Notation. Denoting  $\mathcal{T}_{u,t} \subseteq [t]$  as the collection of time steps that user  $u \in \mathcal{U}$  is served up to round  $t$ , we use  $\mathcal{P}_{u,t} = \{(x_{\tau}, r_{\tau})\}_{\tau \in \mathcal{T}_{u,t}}$  to represent the collection of received arm-reward pairs associated with user  $u$ , and  $T_{u,t} = |\mathcal{T}_{u,t}|$  refers to the number of rounds that user  $u$  has been served. Here,  $\pmb{x}_{\tau} \in \mathcal{A}_{\tau}$ ,  $r_{\tau} \in \mathbb{R}$  separately refer to the chosen arm and actual received reward in round  $\tau \in \mathcal{T}_{u,t}$ . Similarly, we use  $\mathcal{P}_t = \{(x_{\tau}, r_{\tau})\}_{\tau \in [t]}$  to denote all the past records (i.e., arm-reward pairs), up to round  $t$ . For any graph  $\mathcal{G}$ , we denote  $\pmb{A} \in \mathbb{R}^{n \times n}$  as its adjacency matrix (with added self-loops), and  $\pmb{D} \in \mathbb{R}^{n \times n}$  as its degree matrix. Then, we will introduce our proposed solution, the GNB framework.

# 3 GRAPH NEURAL BANDITS: PROPOSED FRAMEWORK

The workflow of our proposed GNB framework is illustrated by Figure 1, and it consists of four major components: (1) estimating the user exploitation graph  $\mathcal{G}^{(1),*}$ , denoted by  $\mathcal{G}^{(1)}$ , and user exploration graph  $\mathcal{G}^{(2),*}$ , denoted by  $\mathcal{G}^{(2)}$  to model the user correlations in terms of exploitation and exploration respectively; (2) applying GNN models  $f_{gnn}^{(1)}(\cdot)$ ,  $f_{gnn}^{(2)}(\cdot)$  on the estimated user graphs  $\mathcal{G}^{(1)}$  and  $\mathcal{G}^{(2)}$ , to collaboratively derive the estimated reward for exploitation, and potential gain for exploration; (3) selecting the arm  $x_{t}$  based on estimated reward and potential gain; and (4) training parameters for GNN models and user neural networks with gradient descent (GD). The pseudo-code is presented in Alg. 1-3, and we move Alg. 2 and 3 to the Appendix Section D due to page limit.

# 3.1 USER GRAPH ESTIMATION WITH USER NETWORKS

Based on the definition of unknown true user graphs  $\mathcal{G}_{i,t}^{(1),*}$ ,  $\mathcal{G}_{i,t}^{(2),*}$  w.r.t. arm  $x_{i,t} \in \mathcal{X}_t$  (Definitions 1, 3), we proceed to derive their estimations  $\mathcal{G}_{i,t}^{(1)}$ ,  $\mathcal{G}_{i,t}^{(2)}$ ,  $i \in [a]$  with individual user networks  $f_u^{(1)}$ ,  $f_u^{(2)}$ ,  $u \in \mathcal{U}$ . With these two kinds of estimated user graphs  $\mathcal{G}_{i,t}^{(1)}$  and  $\mathcal{G}_{i,t}^{(2)}$ , we can thus model the user behaviors under the exploitation setting and the exploration setting separately. Due to page limit, pseudo-code summarizing the workflow is presented in Alg. 2 in Section D of the Appendix.

User Exploitation Network  $f_{u}^{(1)}$ . For each user  $u \in \mathcal{U}$ , we propose to apply an exploitation network  $f_{u}^{(1)}(\cdot)$  to learn user  $u$ 's preference for  $\boldsymbol{x}_{i,t}$ , i.e.,  $\mathbb{E}[r_{i,t}|u,\boldsymbol{x}_{i,t}]$ . This aims to construct the exploitation graph  $\mathcal{G}_{i,t}^{(1)}$  by estimating the user exploitation correlation with user preferences. Here,

$f_{u}^{(1)}(\cdot)$  will be trained on the past records (arm contexts and rewards)  $\mathcal{P}_{u,t}$  from user  $u$ , and the loss function will be the quadratic loss between the predicted reward and the actual reward. In the estimated user exploitation graph  $\mathcal{G}_{i,t}^{(1)}$ , we consider the edge weight between two user nodes  $u, u'$  to be  $w_{i,t}^{(1)}(u, u') = \Psi^{(1)}\big(f_{u}^{(1)}(\boldsymbol{x}_{i,t}), f_{u'}^{(1)}(\boldsymbol{x}_{i,t})\big)$ , where  $\Psi^{(1)}(\cdot, \cdot)$  is the mapping function mentioned in Definition 1 (line 11, Alg. 2).

User Exploration Network  $f_{u}^{(2)}$ . To estimate the potential gain (i.e., the uncertainty for the reward estimation)  $\mathbb{E}[r|u, \boldsymbol{x}_{i,t}] - f_{u}^{(1)}(\boldsymbol{x}_{i,t})$ , we adopt an additional user exploration network  $f_{u}^{(2)}(\cdot)$  inspired by Ban et al. (2022b). Here, the input of  $f_{u}^{(2)}(\cdot)$  is the network gradient of  $f_{u}^{(1)}(\cdot)$  given arm  $\boldsymbol{x}_{i,t}$  as the input, denoted as  $\nabla f_{u}^{(1)}(\boldsymbol{x}_{i,t})$ . Then,  $f_{u}^{(2)}(\cdot)$  will be trained with the input as past gradients of  $f_{u}^{(1)}$ , i.e.,  $\{\nabla f_{u}^{(1)}(\boldsymbol{x}_{\tau})\}_{\tau \in \mathcal{T}_{u,t}}$ ; and the residual of reward prediction  $\{r_{\tau} - f_{u}^{(1)}(\boldsymbol{x}_{\tau})\}_{\tau \in \mathcal{T}_{u,t}}$  will be the output. As it is proved that the confidence interval of reward estimation can be expressed as a function of network gradients (Zhou et al., 2020; Qi et al., 2022), we thus apply  $f_{u}^{(2)}(\cdot)$  to directly learn the prediction uncertainty with the gradient of  $f_{u}^{(1)}(\cdot)$ . Analogously, for the estimated user exploration graph  $\mathcal{G}_{i,t}^{(2)}$  and given two user nodes  $u, u'$ , we let their edge weight be  $w_{i,t}^{(2)}(u, u') = \Psi^{(2)}\left(f_{u}^{(2)}\left(\nabla f_{u}^{(1)}(\boldsymbol{x}_{i,t})\right), f_{u'}^{(2)}\left(\nabla f_{u'}^{(1)}(\boldsymbol{x}_{i,t})\right)\right)$ , where  $\nabla f_{u}^{(1)}(\boldsymbol{x}_{i,t})$  stands for the gradient of  $f_{u}^{(1)}(\cdot)$  given arm  $\boldsymbol{x}_{i,t}$  as the input (line 12, Alg. 2), and  $\Psi^{(2)}(\cdot,\cdot)$  is the mapping function as in Definition 3.

Network Architecture. In this paper, for the theoretical analysis and experiments, we apply separate  $L$ -layer ( $L \geq 2$ ) fully-connected (FC) networks for user exploitation models as well as user exploration models, and their trainable weight matrices are initialized as Gaussian matrices. Details are presented in Section C in the Appendix.

# 3.2 EXPLOITATION AND EXPLORATION WITH USER GRAPHS

With two kinds of estimated user graphs encoding user correlations, we apply two GNN models to separately estimate arm rewards and potential gains for a refined arm selection strategy, which enables us to utilize the past records from all the users compared with single-bandit algorithms (i.e., methods with no user collaboration).

# 3.2.1 ARCHITECTURE OF GNN MODELS

In round  $t$ , with user exploitation graph  $\mathcal{G}_{i,t}^{(1)}$  for each arm  $\pmb{x}_{i,t}\in \mathcal{X}_t$ , we apply the exploitation GNN model  $f_{gnn}^{(1)}(\pmb{x}_{i,t},\mathcal{G}_{i,t}^{(1)};\Theta_{gnn}^{(1)})$  to collaboratively estimate the arm reward  $\hat{r}_{i,t}$  for the received user  $u_{t}\in \mathcal{U}$ . We start from learning the aggregated hidden representation over the user graph, denoted as

$$
\boldsymbol {H} _ {a g g} = \sigma \left(\left(\boldsymbol {S} _ {i, t} ^ {(1)}\right) ^ {k} \cdot \left(\boldsymbol {X} _ {i, t} \boldsymbol {\Theta} _ {a g g} ^ {(1)}\right)\right) \in \mathbb {R} ^ {n \times m} \tag {3}
$$

where  $S_{i,t}^{(1)} = (D_{i,t}^{(1)})^{-\frac{1}{2}}A_{i,t}(D_{i,t}^{(1)})^{-\frac{1}{2}}$  is the symmetrically normalized adjacency matrix of  $\mathcal{G}_{i,t}^{(1)}$ , and  $\sigma$  represents the ReLU activation function. With  $m$  being the network width, we have  $\Theta_{agg}^{(1)} \in \mathbb{R}^{nd \times m}$  as the trainable weight matrix. After propagating the information for  $k$  hops over the user graph, each row of  $H_{agg}$  corresponds to the aggregated  $m$ -dimensional hidden representation for one specific user-arm pair  $(u, x_{i,t})$ ,  $u \in \mathcal{U}$ . In this way, the propagation of multi-hop information can provide a global perspective over the users, since it also involves the neighborhood information of users' neighbors (Zhou et al., 2004). Here in Eq. 3, the embedding matrix  $X_{i,t}$  for the arm  $x_{i,t} \in \mathcal{X}_t$ ,  $i \in [a]$  is defined as

$$
\boldsymbol {X} _ {i, t} = \left( \begin{array}{c c c c} \boldsymbol {x} _ {i, t} ^ {\intercal} & \boldsymbol {0} & \dots & \boldsymbol {0} \\ \boldsymbol {0} & \boldsymbol {x} _ {i, t} ^ {\intercal} & \dots & \boldsymbol {0} \\ \vdots & & \ddots & \vdots \\ \boldsymbol {0} & \boldsymbol {0} & \dots & \boldsymbol {x} _ {i, t} ^ {\intercal} \end{array} \right) \in \mathbb {R} ^ {n \times n d} \tag {4}
$$

to partition the weight matrix  $\Theta_{gnn}^{(1)}$  for different users. In this way, it is designed to generate individual  $m$ -dimensional representations w.r.t. each user-arm pair  $(u, x_{i,t})$ ,  $u \in \mathcal{U}$ , which correspond to the rows of the matrix multiplication  $(X_{i,t} \Theta_{agg}^{(1)}) \in \mathbb{R}^{n \times m}$ .

Then, with  $\pmb{H}_0 = \pmb{H}_{agg}$ , we feed aggregated representations to the  $L$ -layer ( $L \geq 2$ ) FC network as

$$
\boldsymbol {H} _ {l} = \sigma (\boldsymbol {H} _ {l - 1} \cdot \boldsymbol {\Theta} _ {l} ^ {(1)}) \in \mathbb {R} ^ {n \times m}, l \in [ L - 1 ], \quad \widehat {\boldsymbol {r}} _ {a l l} (\boldsymbol {x} _ {i, t}) = \boldsymbol {H} _ {L - 1} \cdot \boldsymbol {\Theta} _ {L} ^ {(1)} \in \mathbb {R} ^ {n} \tag {5}
$$

where  $\widehat{\pmb{r}}_{all}(\pmb{x}_{i,t})\in \mathbb{R}^n$  represents the reward estimation for all the users in  $\mathcal{U}$ , given the arm  $\pmb{x}_{i,t}$ . Received user  $u_{t}$  in round  $t$ , the reward estimation for the user-arm pair  $(u_{t},\pmb{x}_{i,t})$  would be the corresponding element in  $\widehat{\pmb{r}}_{all}$  (line 8, Alg. 1), represented by:

$$
\hat {r} _ {i, t} = f _ {g n n} ^ {(1)} \left(\boldsymbol {x} _ {i, t}, \mathcal {G} _ {i, t} ^ {(1)}; \boldsymbol {\Theta} _ {g n n} ^ {(1)}\right) = \left[ \hat {\boldsymbol {r}} _ {a l l} \left(\boldsymbol {x} _ {i, t}\right) \right] _ {u _ {t}}. \tag {6}
$$

For the FC network, the weight matrices for the first  $L - 1$  layers are  $\Theta_{l} \in \mathbb{R}^{m \times m}, l \in [1, \dots, L - 1]$ , and for the  $L$ -th layer, we have  $\Theta_{L} \in \mathbb{R}^{m}$ . Here, we use  $\Theta_{gnn}^{(1)} = [\mathrm{vec}(\Theta_{agg}^{(1)})^{\intercal}, \mathrm{vec}(\Theta_{1}^{(1)})^{\intercal}, \dots, \mathrm{vec}(\Theta_{L}^{(1)})^{\intercal}]^{\intercal} \in \mathbb{R}^{p}$  to represent the trainable parameters of the GNN exploitation model. The exploitation GNN model  $f_{gnn}^{(1)}(\cdot)$  will be trained with GD based on all the received records  $\mathcal{P}_t$ . Then we apply the quadratic loss function between the reward prediction  $\{f_{gnn}^{(1)}(\boldsymbol{x}_{\tau}, \mathcal{G}_{\tau}^{(1)}; \Theta_{gnn}^{(1)})\}_{\tau \in [t]}$  of chosen arms  $\boldsymbol{x}_{\tau}$ , and the actual received rewards  $\{r_{\tau}\}_{\tau \in [t]}$ .

Connection with Reward Function Definition (Eq. 1) and Constraint (Eq. 2). It is known that when width  $m$  is large enough, the FC network is naturally Lipschitz continuous with respect to the input (Allen-Zhu et al., 2019). In our case, with aggregated hidden representations  $H_{agg}$  being the input to the FC network (Eq. 5), we will have the difference of reward estimations  $\hat{r}_{i,t}$  bounded by the distance of rows in matrix  $H_{agg}$  (i.e., aggregated hidden representations). Therefore, given arm  $x_{i,t} \in \mathcal{X}_t$  and two users  $u_i, u_j \in \mathcal{U}$ , the difference of their estimated rewards  $|[\widehat{\boldsymbol{r}}_{all}(\boldsymbol{x}_{i,t})]_{u_i} - [\widehat{\boldsymbol{r}}_{all}(\boldsymbol{x}_{i,t})]_{u_j}|$  can be bounded by the distance of their estimated correlation vectors (i.e., the corresponding rows in  $S_{i,t}$ ). This matches the reward function definition and the constraint presented in Eq. 1-2.

Exploration GNN Model. To achieve adaptive exploration with user collaborations, we apply a second GNN model  $f_{gnn}^{(2)}(\nabla [f_{gnn}^{(1)}]_{i,t}, \mathcal{G}_{i,t}^{(2)}; \Theta_{gnn}^{(2)})$  to evaluate the potential gain  $\hat{b}_{i,t}$  of the reward estimation  $f_{gnn}^{(1)}(\boldsymbol{x}_{i,t}, \mathcal{G}_{i,t}^{(1)}; \Theta_{gnn}^{(1)})$  (line 8, Alg. 1). Here, the input is the user exploration graph  $\mathcal{G}_{i,t}^{(2)}$ , and the corresponding input graph signal is the gradient of the exploitation GNN model  $\nabla [f_{gnn}^{(1)}]_{i,t} = \nabla_{\Theta_{gnn}^{(1)}} f_{gnn}^{(1)}(\boldsymbol{x}_{i,t}, \mathcal{G}_{i,t}^{(1)}; \Theta_{gnn}^{(1)})$ . Analogous to  $f_{gnn}^{(1)}(\cdot)$ , the architecture of  $f_{gnn}^{(2)}(\cdot)$  can also be represented by Eq. 3-Eq. 6. Note that while  $f_{gnn}^{(1)}(\cdot), f_{gnn}^{(2)}(\cdot)$  have the same network width and number of layers, the dimensionality of weight matrices  $\Theta_{agg}^{(1)} \in \mathbb{R}^{nd \times m}$ ,  $\Theta_{agg}^{(2)} \in \mathbb{R}^{np \times m}$  is different. Similarly, the exploration GNN model will be trained with GD. With the quadratic loss function, we aim to minimize the difference between predicted potential gains  $\{f_{gnn}^{(2)}(\nabla [f_{gnn}^{(1)}]_{\tau}, \mathcal{G}_{\tau}^{(2)}; \Theta_{gnn}^{(2)})\}_{\tau \in [t]}$  and the actual ones  $\{r_{\tau} - f_{gnn}^{(1)}(\boldsymbol{x}_{\tau}, \mathcal{G}_{\tau}^{(1)}; \Theta_{gnn}^{(1)})\}_{\tau \in [t]}$ .

Instead of calculating non-negative UCB intervals (upward exploration only) as in existing works (e.g., Gentile et al. (2014); Ban et al. (2022a)), the exploration GNN model  $f_{gnn}^{(2)}(\cdot)$  leverages both gradient information from the exploitation GNN model  $f_{gnn}^{(1)}(\cdot)$  and the user exploration correlations (i.e.,  $\mathcal{G}_{i,t}^{(2)}$ ) to achieve adaptive exploration (downward and upward exploration).

Remark 3.1 (Reducing Input Complexity). The input of  $f_{gnn}^{(2)}(\cdot)$  is the gradient  $\nabla_{\Theta}f_{gnn}^{(1)}(\pmb{x})$  given the arm  $\pmb{x}$ , and its dimensionality is naturally  $p = (nd\times m) + (L - 1)\times m^2 +m$ , which can be large when increasing the network width  $m$  and depth  $L$ . Inspired by Convolutional Neural Networks (CNNs), e.g., Radenovic et al. (2018), we apply the average pooling to calculate the approximation for the original gradient vector in practice. In this way, we can save the running time for large matrix multiplications, and reduce the space complexity at the same time. Note this approach is also compatible with user networks in Subsection 3.1. To prove its effectiveness, we will apply this method on GNB for all the experiments in Section 5.

Remark 3.2 (Working with Large Systems). When facing a large number of users, to deal with potentially high computational cost, we can apply approximated user neighborhoods to reduce the running time of GNB. Given user graphs  $\mathcal{G}_{i,t}^{(1)},\mathcal{G}_{i,t}^{(2)}$  in terms of arm  $x_{i,t}$ , we derive approximated user neighborhoods  $\tilde{\mathcal{N}}^{(1)}(u_t)$ ,  $\tilde{\mathcal{N}}^{(2)}(u_t)\subset \mathcal{U}$  for target user  $u_{t}$ , with the size  $|\tilde{\mathcal{N}}^{(1)}(u_t)| = |\tilde{\mathcal{N}}^{(2)}(u_t)| = \tilde{n}$ , where  $\tilde{n} < n$ . For instance, we can choose a subset of  $\tilde{n}$  representative users (e.g., users who

ALGORITHM 1: Graph Neural Bandits (GNB)  
1 Input: Number of rounds  $T$ , network width  $m$ , information propagation hops  $k$ . Functions for edge weight estimation  $\Psi^{(1)}(\cdot, \cdot)$ ,  $\Psi^{(2)}(\cdot, \cdot): \mathbb{R} \times \mathbb{R} \mapsto \mathbb{R}$ .  
2 Output: Arm recommendation  $\pmb{x}_t$  for each time step  $t$ .  
3 Initialization: Initialize parameter  $\Theta_0$  for all models.  
4 for  $t = 1, 2, \ldots, T$  do  
5 Receive a user  $u_t$  and a set of arm contexts  $\mathcal{X}_t = \{\pmb{x}_{i,t}\}_{i \in [a]}$ .  
6 Construct two kinds of user graphs  $\{\mathcal{G}_{i,t}^{(1)}\}_{i \in [a]}$ ,  $\{\mathcal{G}_{i,t}^{(2)}\}_{i \in [a]}$  for arm set  $\mathcal{X}_t$  with Algorithm 2.  
7 for each arm  $\pmb{x}_{i,t} \in \mathcal{X}_t$  do  
8 Compute reward estimation  $\hat{r}_{i,t} = f_{gnn}^{(1)}(\pmb{x}_{i,t}, \mathcal{G}_{i,t}^{(1)}; [\Theta_{gnn}^{(1)}]_{t-1})$ , and the potential gain  $\hat{b}_{i,t} = f_{gnn}^{(2)}(\nabla_{\Theta_{gnn}^{(1)}}f_{gnn}^{(1)}(\pmb{x}_{i,t}, \mathcal{G}_{i,t}^{(1)}; [\Theta_{gnn}^{(1)}]_{t-1}), \mathcal{G}_{i,t}^{(2)}; [\Theta_{gnn}^{(2)}]_{t-1})$ .  
9 end  
10 Play arm  $\pmb{x}_t = \arg \max_{\pmb{x}_{i,t} \in \mathcal{X}_t} (\hat{r}_{i,t} + \hat{b}_{i,t})$ , and observe its true reward  $r_t$ .  
11 Train the user networks  $f_u^{(1)}(\cdot; \Theta_u^{(1)})$ ,  $f_u^{(2)}(\cdot; \Theta_u^{(2)})$  and GNN models  $f_{gnn}^{(1)}(\cdot; \Theta_{gnn}^{(1)})$ ,  $f_{gnn}^{(2)}(\cdot; \Theta_{gnn}^{(2)})$  with gradient descent, according to Algorithm 3.  
12 end

always post high quality reviews in e-commerce platforms) to form  $\tilde{\mathcal{N}}^{(1)}(u_t),\tilde{\mathcal{N}}^{(2)}(u_t)$  for the downstream GNN models, which can significantly reduce the computation cost. Related experiments are provided in Subsection 5.3 and Appendix Section B.

Weight Matrices Initialization. For both GNN models  $\Theta_{gnn}^{(1)}$  and  $\Theta_{gnn}^{(2)}$ , the matrix entries of the aggregation weight matrix  $\Theta_{agg}$  and the first  $L - 1$  FC layers  $\{\Theta_1,\dots \Theta_{L - 1}\}$  are drawn from the Gaussian distribution  $N(0,2 / m)$ . Then, for the last layer  $\Theta_L$ , we draw its entries from  $N(0,1 / m)$ .

# 3.2.2 ARM SELECTION MECHANISM AND MODEL TRAINING

In round  $t$ , with the current parameters  $[\Theta_{gnn}^{(1)}]_{t-1}, [\Theta_{gnn}^{(2)}]_{t-1}$  for GNN models before training, the selected arm is chosen as  $\boldsymbol{x}_t = \arg \max_{\boldsymbol{x}_{i,t} \in \mathcal{X}_t} (f_{gnn}^{(1)}(\boldsymbol{x}_{i,t}, \mathcal{G}_{i,t}^{(1)}; [\Theta_{gnn}^{(1)}]_{t-1}) + f_{gnn}^{(2)}(\nabla_{\Theta_{gnn}^{(1)}} f_{gnn}^{(1)}(\boldsymbol{x}_{i,t}, \mathcal{G}_{i,t}^{(1)}; [\Theta_{gnn}^{(1)}]_{t-1}), \mathcal{G}_{i,t}^{(2)}; [\Theta_{gnn}^{(2)}]_{t-1}))$  based on the estimated reward and potential gain (line 10, Alg. 1). After receiving the true reward  $r_t$ , we proceed to update the user networks and GNN models based on GD and quadratic loss function (line 11, Alg. 1). Pseudo-code of detailed training procedure is shown in Alg. 3 from Appendix (Section D), due to page limit.

# 4 THEORETICAL ANALYSIS

In this section, we present the theoretical analysis for the proposed GNB. Here, we consider each user  $u \in \mathcal{U}$  to be evenly served  $T / n$  rounds up to time step  $T$ , i.e.,  $|\mathcal{T}_{u,t}| = T_{u,t} = T / n$ , which is standard in closely related works (e.g., Gentile et al. (2014); Ban & He (2021)). To ensure the neural models are able to efficiently learn the underlying reward mapping, we have the following mild assumption on arm separateness.

Assumption 4.1 ( $\rho$ -Separateness of Arms). After a total of  $T$  rounds, for every pair  $\pmb{x}_{i,t}, \pmb{x}_{i',t'}$  with  $t, t' \in [T]$  and  $i, i' \in [a]$ , if  $(t,i) \neq (t',i')$ , we have  $\| \pmb{x}_{i,t} - \pmb{x}_{i',t'} \|_2 \geq \rho$  where  $0 < \rho \leq \mathcal{O}\left(\frac{1}{L}\right)$ .

Note that the above assumption is mild, and it has been repeatedly applied in previous works on neural bandits (Ban et al., 2022b) and over-parameterized neural networks (Allen-Zhu et al., 2019). Meanwhile, Assumption 4.2 in Zhou et al. (2020) and Assumption 3.4 from Zhang et al. (2021) also imply that no two arms are the same, and they measure the arm separateness in terms of the minimum eigenvalue  $\lambda_0$  (with  $\lambda_0 > 0$ ) of the Neural Tangent Kernel (NTK) (Jacot et al., 2018) matrix, which is comparable with our Euclidean separateness  $\rho$ . Note that since  $L$  can be manually set (e.g.,  $L = 2$ ), we can easily satisfy the condition  $0 < \rho \leq \mathcal{O}\left(\frac{1}{L}\right)$  as long as no two arms are identical.

Based on Definition 1 and Definition 3, given an arm  $\pmb{x}_{i,t} \in \mathcal{X}_t$ , we have the adjacency matrices  $\pmb{A}_{i,t}^{(1),*}$  and  $\pmb{A}_{i,t}^{(2),*}$  for the true arm graphs  $\mathcal{G}_{i,t}^{(1),*}$ ,  $\mathcal{G}_{i,t}^{(2),*}$ . For the sake of analysis, given any

adjacency matrix  $\mathbf{A}$ , we derive the normalized adjacency matrix  $S$  by scaling the elements of  $\mathbf{A}$  with  $1/n$ . We also set the neighborhood parameter  $k = 1$ , and define the mapping functions  $\Psi^{(1)}(a,b), \Psi^{(2)}(a,b) \coloneqq \exp(-|a - b|)$  given the inputs  $a, b \in \mathbb{R}$ . Note that our results can be readily generalized to other mapping functions with the Lipschitz-continuity properties.

We proceed to derive the regret bound for  $T$  time steps, denoted as  $R(T)$ . Here, the following Theorem 4.2 offers the cumulative regret bound covering both types of error: (1) the estimation error of user graphs; and (2) the approximation error of neural models. Let  $\eta_1, J_1$  be the learning rate and iterations for user networks, and  $\eta_2, J_2$  denote the learning rate and iterations for GNN models.

Theorem 4.2. Define  $\delta \in (0,1)$ ,  $0 < \xi_1, \xi_2 \leq \mathcal{O}(1 / T)$  and  $0 < \rho \leq \mathcal{O}(1 / L)$ . With the user networks defined in Eq. 7 and the GNN models defined in Eq. 3-5 with  $L$  FC-layers, let their width  $m \geq \Omega\left(Poly(T,L,a,\frac{1}{\rho}) \cdot \log(1 / \delta)\right)$ . With training process in Algorithm 3, set parameters

$$
\begin{array}{l} \eta_ {1} = \Theta \left(\frac {\rho}{m \cdot \operatorname {P o l y} (T , n , a , L)}\right), \quad \eta_ {2} = \Theta \left(\frac {\rho}{m \cdot \operatorname {P o l y} (T , a , L)}\right) \\ J _ {1} = \Theta \bigg (\frac {P o l y (T , n , a , L)}{\rho \cdot \delta^ {2}} \cdot \log (\frac {1}{\xi_ {1}}) \bigg), \quad J _ {2} = \Theta \bigg (\frac {P o l y (T , a , L)}{\rho \cdot \delta^ {2}} \cdot \log (\frac {1}{\xi_ {2}}) \bigg). \\ \end{array}
$$

Then, following Algorithm 1, Algorithm 2 for arm pulling and user group update, with probability at least  $1 - \delta$ , the  $T$ -round pseudo-regret  $R(T)$  of GNB could be bounded by

$$
R (T) \leq (\sqrt {4 T} - 1) \cdot \mathcal {O} (L ^ {3}) + (\sqrt {4 T} - 1) \cdot \mathcal {O} (L ^ {2}) \cdot \sqrt {\log (\frac {T n \cdot a}{\delta})} + \mathcal {O} (L ^ {2}) + \mathcal {O} (1).
$$

Due to page limit, detailed regret bound and the proof of Theorem 4.2 are presented in the Appendix.

Remark 4.3 (Removing  $d$ ,  $\tilde{d}$  Terms). Existing neural single-bandit (i.e., with no user collaboration) algorithms (Zhou et al., 2020; Zhang et al., 2021) derive the bound  $\mathcal{O}(\tilde{d}\sqrt{T}\log(T))$  based on neural gradient mappings and ridge regression, and they involve the effective dimension term  $\tilde{d}$  of the NTK matrix, which can grow along with the scale of network parameters and number of rounds  $T$ . The linear user clustering algorithms (e.g., Li et al. (2019); Ban & He (2021); Gentile et al. (2017)) have the bound  $\mathcal{O}(d\sqrt{T}\log(T))$  with the term of arm dimension  $d$ , which can be large given arm contexts in the high-dimensional space. Here, we improve their bounds by a multiplicative factor of  $\sqrt{\log(T)}$  and remove the dimension terms  $d$ ,  $\tilde{d}$ . We apply the generalization bound for overparameterized neural networks (Allen-Zhu et al., 2019; Cao & Gu, 2019) instead of regression-based analysis to remove the  $\sqrt{\log(T)}$  term, and the generalization error is also unrelated to  $d$  or  $\tilde{d}$  for over-parameterized neural networks.

Remark 4.4 (Reducing  $\sqrt{n}$  to  $\sqrt{\log(n)}$ ). While our  $\mathcal{O}(\sqrt{T\log(T)})$  bound matches theoretical bound of state-of-the-art EE-Net (Ban et al., 2022b), EE-Net only considers the single-bandit setting with no collaboration among users. Compared with Meta-Ban (Ban et al., 2022a), we provide the theoretical analysis from a new perspective regarding the fine-grained user collaborative effect and GNNs. In particular, compared with existing user clustering works (e.g., Ban et al. (2022a); Gentile et al. (2014); Li et al. (2019); Ban & He (2021)) imposing the additional  $\sqrt{n}$  (where  $n$  is the number of users) factor to incorporate user collaborative effects, our GNB only end up with the  $\sqrt{\log(n)}$  term by adopting GNN models for user collaboration, which is sharper than existing works.

Remark 4.5 (Removing i.i.d. Assumption). Compared with existing clustering of bandits algorithms (e.g., Gentile et al. (2014); Li et al. (2019); Gentile et al. (2017); Ban et al. (2022a)) and the single-bandit algorithm EE-Net (Ban et al., 2022b), we avoid making the i.i.d. assumption for the arms by applying the martingale-based analysis. For real-world applications, their i.i.d. assumption can be strong since the candidate arm pool is always conditioned on the received records, and candidate arms for a specific round can also come from different distributions.

# 5 EXPERIMENTS

In this section, we evaluate the proposed GNB framework on multiple real data sets against nine state-of-the-art algorithms, including: CLUB (Gentile et al., 2014), SCLUB (Li et al., 2019), LOCB (Ban & He, 2021), DynUCB (Nguyen & Lauw, 2014), COFIBA (Li et al., 2016), Neural-UCB-Pool (Neural-Pool) (Zhou et al., 2020), Neural-UCB-Ind (Neural-Ind) (Zhou et al., 2020), EE-Net (Ban et al., 2022b), and Meta-Ban (Ban et al., 2022a). Due to the page limit, we will include the descriptions for the baselines and experiment settings in the Appendix Section B.

![](images/7919acebe1a62c2440700290721b25eea964b0585379ae424acf66aeb825b8a2.jpg)  
Figure 2: Cumulative regrets on the recommendation and classification data sets.

![](images/e0804a06c52ac88285d7f0d9dcf083516e55aeb68151a8182d8fcbd7ea0b7ffc.jpg)

![](images/46341bff36ebc7b50a02827f9c73845fa608ab9efa52450c158dbaa87c10b986.jpg)

![](images/5fb5f4a76781c0b925203679f7f2b4d95d621ca43857e9b31aaf8d5aa4336289.jpg)

# 5.1 REAL DATA SETS

Recommendation Data Sets. First, we conduct the experiments for two recommendation data sets with different specifications, which are the "MovieLens" data set and the "Yelp" data set. Given one user  $u_{t}$  to serve in each round  $t$ , our goal is to recommend the optimal arm (movie / restaurant) from the candidate pool  $\mathcal{X}_{t}$  to the user.

Classification Data Sets. In addition to the two recommendation data sets above, we also perform experiments on two real classification data sets under the recommendation settings, which are: (1) the "MNIST" data set with  $\mathcal{C} = 10$  different classes, and (2) the "Shuttle" data set with  $\mathcal{C} = 7$  classes. Here, each class from  $\mathcal{C}$  to be a user, and we need to recommend the arm that correctly matches the received user for each round. Due to the page limit, details for these four real recommendation data sets, including settings and URLs, are shown in the Appendix, Subsection B.2.

# 5.2 EXPERIMENT RESULTS

Figure 2 illustrates the cumulative regret results on the four data sets, our proposed GNB manages to achieve the best performance against all these strong benchmarks. First, since the MovieLens data set involves real arm features unlike the Yelp data set that includes high inherent noise, the performance of different algorithms on the MovieLens data set tends to have larger divergence. Among those regret results, the algorithms with neural architectures (Neural-Pool, EE-Net, Meta-Ban) generally perform better than linear algorithms due to the approximation power of neural networks. However, as Neural-Ind considers no collaboration among users, it performs the worst among all baselines on these two data sets. EE-Net outperforms Neural-Pool thanks to its adaptive exploration strategy.

For classification data sets, Meta-Ban outperforms the other baselines by considering the user collaborative effects, and EE-Net also performs better than Neural-Pool thanks to its adaptive exploration strategy. Different from recommendation data sets, the classification data sets involve more complicated reward mapping functions, and this might lead to the poor performances of linear algorithms. Our proposed GNB consistently outperforms all baselines by modeling the fine-grained user (class) correlations and utilizing the adaptive exploration strategy simultaneously. In addition, we note that GNB only takes approximately  $75\%$  of Meta-Ban's running time to finish the experiments, since it does not require to train the framework individually for each arm before making predictions.

# 5.3 SUPPLEMENTARY EXPERIMENTS

Due to the page limit, we present additional supplementary experiments in the Appendix Section B, including: (1) experiments on additional data sets; (2) with increasing number of users, experiments demonstrating the effectiveness of applying approximated user neighborhoods (Remark 3.2); (3) experiments showing the potential performance impact on GNB when there exist underlying user clusters; (4) the parameter sensitivity study showing that our adaptive exploration strategy can indeed improve the performance of GNB, and the effects of different hops  $k$  for information propagation.

# 6 CONCLUSION

In this paper, we propose a novel framework named GNB to model the fine-grained user collaborative effects. Instead of modeling user correlations through the estimation of rigid user groups, we estimate the user graphs to preserve the pair-wise user correlations for exploitation and exploration separately, and utilize individual GNN-based models to achieve the adaptive exploration. Moreover, under standard assumptions, we also demonstrate the improvement of regret bounds over existing methods from a new perspective of "fine-grained" user collaborative effects and GNNs. Extensive experiments are conducted to show the effectiveness of our proposed framework against strong baselines.

# REFERENCES

Yasin Abbasi-Yadkori, David Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems, 24:2312-2320, 2011.  
Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via overparameterization. In International Conference on Machine Learning, pp. 242-252. PMLR, 2019.  
Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2-3):235-256, 2002.  
Yikun Ban and Jingrui He. Local clustering in contextual multi-armed bandits. In Proceedings of the Web Conference 2021, pp. 2335-2346, 2021.  
Yikun Ban, Jingrui He, and Curtiss B Cook. Multi-facet contextual bandits: A neural network perspective. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 35-45, 2021.  
Yikun Ban, Yunzhe Qi, Tianxin Wei, and Jingrui He. Neural collaborative filtering bandits via meta learning. arXiv preprint arXiv:2201.13395, 2022a.  
Yikun Ban, Yuchen Yan, Arindam Banerjee, and Jingrui He. Ee-net: Exploitation-exploration neural networks in contextual bandits. In International Conference on Learning Representations, 2022b.  
Yuan Cao and Quanquan Gu. Generalization bounds of stochastic gradient descent for wide and deep neural networks. Advances in Neural Information Processing Systems, 32:10836-10846, 2019.  
Nicolo Cesa-Bianchi, Alex Conconi, and Claudio Gentile. On the generalization ability of on-line learning algorithms. IEEE Transactions on Information Theory, 50(9):2050-2057, 2004.  
Nicolo Cesa-Bianchi, Claudio Gentile, and Giovanni Zappella. A gang of bandits. In NeurIPS, pp. 737-745, 2013.  
Jie Chen, Tengfei Ma, and Cao Xiao. Fastgcn: fast learning with graph convolutional networks via importance sampling. arXiv preprint arXiv:1801.10247, 2018.  
Wei Chu, Lihong Li, Lev Reyzin, and Robert Schapire. Contextual bandits with linear payoff functions. In AISTATS, pp. 208-214, 2011.  
Aniket Anand Deshmukh, Urun Dogan, and Clay Scott. Multi-task learning for contextual bandits. In NeurIPS, pp. 4848-4856, 2017.  
Audrey Durand, Charis Achilleos, Demetris Iacovides, Katerina Strati, Georgios D Mitsis, and Joelle Pineau. Contextual bandits for adapting treatment in a mouse model of de novo carcinogenesis. In Machine learning for healthcare conference, pp. 67-82. PMLR, 2018.  
Johannes Gasteiger, Aleksandar Bojchevski, and Stephan Gunnemann. Predict then propagate: Graph neural networks meet personalized pagerank. In International Conference on Learning Representations, 2019.  
Claudio Gentile, Shuai Li, and Giovanni Zappella. Online clustering of bandits. In ICML, pp. 757-765, 2014.  
Claudio Gentile, Shuai Li, Purushottam Kar, Alexandros Karatzoglou, Giovanni Zappella, and Evans Etrue. On context-dependent clustering of bandits. In ICML, pp. 1253-1262, 2017.  
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In WWW, pp. 173-182, 2017.  
Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. Lightgen: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pp. 639-648, 2020.

Joey Hong, Branislav Kveton, Manzil Zaheer, Yinlam Chow, Amr Ahmed, and Craig Boutelier. Latent bandits revisited. Advances in Neural Information Processing Systems, 33:13423-13433, 2020.  
Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31, 2018.  
Parnian Kassraie, Andreas Krause, and Ilija Bogunovic. Graph neural network bandits. arXiv preprint arXiv:2207.06456, 2022.  
Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextual-bandit approach to personalized news article recommendation. In WWW, pp. 661-670, 2010.  
Shuai Li, Alexandros Karatzoglou, and Claudio Gentile. Collaborative filtering bandits. In SIGIR, pp. 539-548, 2016.  
Shuai Li, Wei Chen, Shuai Li, and Kwong-Sak Leung. Improved algorithm on online clustering of bandits. In IJCAI, pp. 2923-2929, 2019.  
Odalric-Ambrym Maillard and Shie Mannor. Latent bandits. In International Conference on Machine Learning, pp. 136-144. PMLR, 2014.  
Trong T Nguyen and Hady W Lauw. Dynamic clustering of contextual multi-armed bandits. In Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management, pp. 1959–1962, 2014.  
Yunzhe Qi, Yikun Ban, and Jingrui He. Neural bandit with arm group graph. arXiv preprint arXiv:2206.03644, 2022.  
Filip Radenović, Giorgos Tolias, and Ondřej Chum. Fine-tuning cnn image retrieval with no human annotation. IEEE transactions on pattern analysis and machine intelligence, 41(7):1655-1668, 2018.  
Victor Garcia Satorras and Joan Bruna Estrach. Few-shot learning with graph neural networks. In International Conference on Learning Representations, 2018.  
Sohini Upadhyay, Mikhail Yurochkin, Mayank Agarwal, Yasaman Khazaeni, and Djallel Bouneffouf. Graph convolutional network upper confident bound. 2020.  
Michal Valko, Nathan Korda, Rémi Munos, Ilias Flaounas, and Nello Cristianini. Finite-time analysis of kernelised contextual bandits. In Uncertainty in Artificial Intelligence, 2013.  
Sofía S Villar, Jack Bowden, and James Wason. Multi-armed bandit models for the optimal design of clinical trials: benefits and challenges. Statistical science: a review journal of the Institute of Mathematical Statistics, 30(2):199, 2015.  
Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. Neural graph collaborative filtering. In Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval, pp. 165-174, 2019.  
Max Welling and Thomas N Kipf. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.  
Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. Simplifying graph convolutional networks. In International conference on machine learning, pp. 6861-6871. PMLR, 2019.  
Qingyun Wu, Huazheng Wang, Quanquan Gu, and Hongning Wang. Contextual bandits in a collaborative environment. In SIGIR, pp. 529-538, 2016.  
Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. In International Conference on Machine Learning, pp. 5453-5462. PMLR, 2018.

Keyulu Xu, Mozhi Zhang, Stefanie Jegelka, and Kenji Kawaguchi. Optimization of graph neural networks: Implicit acceleration by skip connections and more depth. In International Conference on Machine Learning, pp. 11592-11602. PMLR, 2021.  
Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L Hamilton, and Jure Leskovec. Graph convolutional neural networks for web-scale recommender systems. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 974-983, 2018.  
Jiaxuan You, Rex Ying, and Jure Leskovec. Position-aware graph neural networks. In International Conference on Machine Learning, pp. 7134-7143. PMLR, 2019.  
Weitong Zhang, Dongruo Zhou, Lihong Li, and Quanquan Gu. Neural thompson sampling. In International Conference on Learning Representations, 2021.  
Dengyong Zhou, Olivier Bousquet, Thomas N Lal, Jason Weston, and Bernhard Scholkopf. Learning with local and global consistency. In NeurIPS, pp. 321-328, 2004.  
Dongruo Zhou, Lihong Li, and Quanquan Gu. Neural contextual bandits with ucb-based exploration. In International Conference on Machine Learning, pp. 11492-11502. PMLR, 2020.
