# DEEP INTERACTION PROCESSES FOR TIME-EVOLVING GRAPHS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Time-evolving graphs are ubiquitous such as online transactions on an e-commerce platform and user interactions on social networks. While neural approaches have been proposed for graph modeling, most of them focus on static graphs. In this paper we present a principled deep neural approach that models continuous time-evolving graphs at multiple time resolutions based on a temporal point process framework. To model the dependency between latent dynamic representations of each node, we define a mixture of temporal cascades in which a node's neural representation depends on not only this node's previous representations but also the previous representations of related nodes that have interacted with this node. We generalize LSTM on this temporal cascade mixture and introduce novel time gates to model time intervals between interactions. Furthermore, we introduce a selection mechanism that gives important nodes large influence in both  $k$ -hop subgraphs of nodes in an interaction. To capture temporal dependency at multiple time-resolutions, we stack our neural representations in several layers and fuse them based on attention. Based on the temporal point process framework, our approach can naturally handle growth (and shrinkage) of graph nodes and interactions, making it inductive. Experimental results on interaction prediction and classification tasks – including a real-world financial application – illustrate the effectiveness of the time gate, the selection and attention mechanisms of our approach, as well as its superior performance over the alternative approaches.

# 1 INTRODUCTION

Representation learning over graph data has become a core machine learning task with a wide range of applications including e-commerce, finance, social networks, and bioinformatics. Various neural graph representations such as (Perozzi et al., 2014; Grover & Leskovec, 2016; Wang et al., 2016; Kipf & Welling, 2017; Defferrard et al., 2016; Scarselli et al., 2009; Ying et al., 2018; Hamilton et al., 2017b; Monti et al., 2017; Den Berg et al., 2017) have been proposed to learn from static graph data and successfully used for downstream tasks (e.g., classification). Graph data, however, are often dynamic in practice; nodes and interactions between them can grow and shrink. A straightforward approach to handle dynamic graphs is to compress them into one or several static graphs. The drawbacks of this approach are multifold; we not only blur temporal structural information but also miss time information that can be critical for real-world applications. An illustrative example is given in figure 1.

To handle continuous time-evolving graph, we can approximate a by a sequence of snapshot graphs, each of which includes all interactions that occur during a user-specified discrete-time interval, as shown in (Leskovec et al., 2007; Hamilton et al., 2016; Kulkarni et al., 2015; Goyal et al., 2018). This treatment reduces time resolution and it is tricky to specify the appropriate aggregation granularity. To avoid these problems, Nguyen et al. (2018) proposed continuous-time dynamic networks (CTDNE) that generalize deep walk methods to learn time-dependent network embedding. As a transductive method, CTDNE cannot handle the growth of new nodes. Dai et al. (2016) applied temporal point processes to model time-evolving graphs and, as a nonparametric Bayesian approach, their approach can naturally cope with the growth of new nodes and interactions. They used recurrent neural networks (RNNs) to define an intensity function in temporal point processes. These RNN models are shallow and one-step unrolled, making it easy to compute but relatively limited in modeling power. Trivedi

![](images/0d4b872a900611537f1a947af889f9b83f4514e7173293874bfe2b392d90a6a2.jpg)  
(a) An illegal cash-out event

![](images/fffcf66691460a75a82d0b03a5f4f05f9810194401df21290bd05d82c6bedaef.jpg)  
(b) Legal transactions  
Figure 1: An illustrative example. Figure (a) shows an illegal cash-out event. It can be revealed by high-frequency transactions with multiple merchants. However, if we merge the transaction data into a static graph, we cannot distinguish it from the static graph generated from normal online shopping activities. Thus, learning from such a static graph will fail to detect the cash-out event.

et al. (2019) extended this approach by modeling two-time scale and adopting temporal-attention mechanism.

In this paper we present a powerful deep neural approach that models continuous time-evolving graphs at multiple time resolutions based on a temporal point process framework. We name the new approach deep interaction processes (DIPs). To model the dependency between latent dynamic representations of each node, we define a mixture of temporal cascades in which a node's neural representation depends on not only this node's previous representations but also the previous representations of related nodes that have interacted with this node. We generalize LSTM on this temporal cascade mixture and introduce novel time gates to model time intervals between interactions. Furthermore, We introduce a selection mechanism that gives important nodes large influence in both  $k$ -hop subgraphs of nodes in an interaction. To obtain representations from fine-to-coarse time-resolutions, we stack our neural representations in several layers and fuse them based on attention. Based on the temporal point process framework, our approach can naturally handle growth of graph nodes and interactions, making it inductive.

The rest of the paper is organized as follows. In Section 2 we give background on temporal point processes and in Section 3 we present the new DIP approach. In Section 4 we discuss related works. In Section 5 we report experimental results on multiple interaction prediction and classification tasks including an important real-world anti-fraud financial application, demonstrating superior performance of the new approach over the alternatives.

# 2 TEMPORAL POINT PROCESSES

We first describe temporal point processes (a class of nonparametric Bayesian models) that our approach is based on. Specifically, a temporal point process is a stochastic process that generates a sequence of discrete events localized at times  $\{t_i\}_{i=1}^N$  in any given observed time window  $[0,T]$ . An important way to characterize temporal point processes is via the conditional intensity function  $\lambda(t|H_t)$  -the stochastic model for the next event time  $t$  given all historical events before time  $t$ , denoted as  $H_t = \{t_i | t_i < t\}$ . Formally, within a small time window  $[t,t + dt)$ ,  $\lambda(t|H_t)dt$  is the probability for the occurrence for a new event given the  $H_t$ :  $\lambda(t|H_t)dt = P\{event in [t,t + dt)|H_t\}$ . From the survival analysis theory(Aalen et al., 2008), given the times of the past events  $\{t_1,t_2,\ldots,t_i\}$ , the conditional density that an event occurs at  $t_{i+1}$  is given as follows:  $p\left(t_{i+1}|H_{t_{i+1}}\right) = \lambda\left(t_{i+1}|H_{t_{i+1}}\right)\exp\left\{-\int_{t_i}^{t_{i+1}}\lambda(t|H_t)dt\right\}$ , where the exponential part in the above equation means the conditional probability that no event happens during  $[t_i,t_{i+1})$ . The functional forms of the conditional intensity function  $\lambda(t|H_t)$  can represent certain forms of dependencies of the historical events. For instance, for Poisson processes(Kingman, 2005) we set  $\lambda$  to be constant - making the assumption that the process is stationary and the temporal events in history are independent of each other. For classical Hawkes processes(Hawkes, 1971), the intensity function  $\lambda$  is often set to be a sum of multiple exponential functions, assuming that the mutual excitation among events is positive, additive over the past events, and exponentially decaying with time. Mei & Eisner (2017a) removed these limiting assumptions using LSTM to learn  $\lambda$  from data.

# 3 DEEP INTERACTION PROCESSES

In this section, we present the new neural nonparametric Bayesian approach over continuous-time evolving graphs. First, we present a temporal dependency graph that is a mixture of the temporal cascades, to model interdependence between graph nodes (as well as latent node representations). Then we present a novel deep model to learn dynamic node representations in the temporal dependency graph. This model naturally generalizes LSTM on the traditional chain-structured data. Given the dynamic node representations, we define deep interaction processes that model potential interactions between any two nodes over time. Finally we layout the maximum likelihood estimation method.

# 3.1 TEMPORAL DEPENDENCY GRAPH

![](images/28985330e9f02087653ffa00e78faa00dcc08de8cd303e6c12429b56fe3e67d4.jpg)  
Figure 2: Dynamic interactions and the corresponding dependency graph

Consider a collection of people-movie records at different time points (e.g., Dave buy Captain America's Shield toy at  $t_{6}$ ) as shown in figure 2. The people and movies form a dynamic graph in which each person or movie is a node and interactions happen over time. After one interaction occurs, we update the neural representation of the two nodes linked to this interaction; e.g., right after time  $t_{6}$ , we update the representations for David and the Captain America's Shield toy. The new neural representation of David depends on both his current and previous interactions – as a result, depending on the representations of the two nodes associated with the previous interaction. This naturally forms a dependency cascade. Similarly, we can obtain a dependency cascade for Lucy's representations. Because of the common movies David and Lucy saw and toys they bought, their dependency cascades overlap and form a cascade mixture. Formally, we denote a dynamic interaction or link at time  $t$  by  $l_{u,v,t}$  where  $u$  are  $v$  are two nodes associated with this interaction. We denote the node  $u$  at time  $t$  by  $u(t)$  and the two nodes associated with  $u$ 's precedent interaction at time  $t^{-}$  as  $u^{1}(t)$  and  $u^{2}(t)$ . Note that one of  $u^{1}(t)$  and  $u^{2}(t)$  is simply  $u(t^{-})$ . For example,  $u^{1}(t_{6})$  and  $u^{2}(t_{6})$  in figure 2 are  $u(t_{4})$  and  $w(t_{4})$ , respectively. For later usage, We denote the subgraph rooted at  $u(t)$  with  $(k - 1)$  depth as  $subgraph(u(t), k)$  shown in Figure 2.

# 3.2 DIP NEURAL UNIT

Now we present the novel neural unit to update dynamic latent representations of nodes over the temporal dependency graph. First, let us denote node  $u$ 's features or embedding (i.e., a static representation jointly learned from data) at time  $t$  by  $\mathbf{x}_{u(t)}$  and denote features of interaction  $l$  by  $x_{l}$ . The interaction feature can be empty if the interaction contains only the temporal information. The concatenation of  $\mathbf{x}_{u(t)}$  and  $\mathbf{x}_l$  is denoted by  $\hat{x}_{u(t)}$ . Let  $\Delta_{(u,t)} = t - t^{-}$  be the time interval between two consecutive interactions involving  $u$  at time  $t$  and  $t^{-}$ .

Our neural unit generalizes LSTM unit on the temporal dependency graph; we use an input gate, an output gate and two forget gates over  $\hat{x}_{u(t)}$ , dynamic representation of  $\mathbf{h}_{u^i(t)}$  and cell states  $\mathbf{c}_{u^i(t)}$  ( $i = 1,2$ ) as shown in figure 3. In addition to these gates, we introduce time gates to capture the impact of time interval  $\Delta_{(u,t)}$ .

![](images/cc80f8992582a2875c6558910e255f81bc070b374ab4148a76bcf5c68e19c383.jpg)  
Figure 3: Time-evolving graph unit updates the representation  $\mathbf{h}_{u(t)}$  and cell state  $\mathbf{c}_{u(t)}$  for node  $u$  at time  $t$  based on both the features  $\hat{x}_{u(t)}$  and the representations and cell states of the nodes  $u^1(t)$  and  $u^2(t)$  associated with  $u$ 's precedent interaction.

Specifically,  $\mathbf{h}_{u(t)}$  and  $\mathbf{c}_{u(t)}$  are updated as follows:

$$
\mathbf {z} _ {u (t)} = \sigma \left(\mathbf {W} _ {z} \hat {\mathbf {x}} _ {u (t)} + \sum_ {i = 1} ^ {2} \mathbf {R} _ {z _ {i}} \mathbf {h} _ {u ^ {i} (t)} + \mathbf {b} _ {z}\right) \qquad \qquad \mathbf {o} _ {u (t)} = \sigma \left(\mathbf {W} _ {o} \hat {\mathbf {x}} _ {u (t)} + \sum_ {i = 1} ^ {2} \mathbf {R} _ {o _ {i}} \mathbf {h} _ {u ^ {i} (t)} + \mathbf {b} _ {o}\right)
$$

$$
\mathbf {s} _ {u (t)} = \tanh \left(\mathbf {W} _ {s} \hat {\mathbf {x}} _ {u (t)} + \sum_ {i = 1} ^ {2} \mathbf {R} _ {s _ {i}} \mathbf {h} _ {u ^ {i} (t)} + \mathbf {b} _ {s}\right) \quad \mathbf {f} _ {u (t), u ^ {i} (t)} = \sigma \left(\mathbf {W} _ {f _ {i}} \hat {\mathbf {x}} _ {u (t)} + \mathbf {R} _ {f _ {i}} \mathbf {h} _ {u ^ {i} (t)} + \mathbf {b} _ {f _ {i}}\right)
$$

$$
\mathbf {g} _ {u (t), u ^ {i} (t)} = \sigma \left(\mathbf {W} _ {g _ {i}} \hat {\mathbf {x}} _ {u (t)} + \mathbf {R} _ {g _ {i}} \mathbf {h} _ {u ^ {i} (t)} + \mathbf {M} _ {g _ {i}} \Delta_ {u} + \mathbf {b} _ {g _ {i}}\right)
$$

$$
\mathbf {c} _ {u (t)} = \mathbf {z} _ {u (t)} \odot \mathbf {s} _ {u (t)} + \sum_ {i = 1} ^ {2} \mathbf {f} _ {u (t), u ^ {i} (t)} \odot \mathbf {c} _ {u ^ {i} (t)} \odot \mathbf {g} _ {u (t), u ^ {i} (t)}
$$

$$
\mathbf {h} _ {u (t)} = \mathbf {o} _ {u (t)} \odot \tanh  \left(\mathbf {c} _ {u (t)}\right)
$$

where  $\sigma$ ,  $\tanh$  and  $\odot$  represent the sigmoid function, the hyperbolic tangent function, and the Hadamard product (pointwise multiplication), respectively, and parameters in the unit including the recurrent weights  $\mathbf{R}_{z_i}, \mathbf{R}_{o_i}, \mathbf{R}_{s_i}, \mathbf{R}_{f_i}$  and  $\mathbf{R}_{g_i}$ , the projection matrices  $\mathbf{W}_z, \mathbf{W}_o, \mathbf{W}_s, \mathbf{W}_{f_i}$  and  $\mathbf{W}_{g_i}$ , the bias vectors  $\mathbf{b}_z, \mathbf{b}_o, \mathbf{b}_s, \mathbf{b}_{f_i}$  and  $\mathbf{b}_{g_i}$  and the time weight matrix  $\mathbf{M}_{g_i}$  are learned from data.

For convenience, we use DIP-UNIT  $(\cdot)$  to summarize the above equations,

$$
\mathbf {h} _ {u (t)}, \mathbf {c} _ {u (t)} = \mathrm {D I P - U N I T} \left(\hat {\mathbf {x}} _ {u (t)}, \mathbf {c} _ {u ^ {1} (t)}, \mathbf {c} _ {u ^ {2} (t)}, \mathbf {h} _ {u ^ {1} (t)}, \mathbf {h} _ {u ^ {2} (t)}, \Delta_ {u}, \Theta\right)
$$

where  $\Theta$  represent all the parameters. Similar to LSTM training where a  $k$ -hop neighborhood is often used in practice, we limit the backtracking in  $subgraph(u(t), k)$  to the computational cost.

# 3.3 DIP-UNIT AND FUSION

To model nonlinear dependency relationships at different temporal resolutions, we stack  $L$  layers of DIP-UNIT together. The output of the  $j$ -th layer is computed recursively as follows:

$$
\left(\mathbf {h} _ {u (t)} ^ {j}, \mathbf {c} _ {u (t)} ^ {j}\right) = \mathrm {D I P - U N I T} ^ {j} \left(\mathbf {h} _ {u (t)} ^ {j - 1}, \mathbf {c} _ {u ^ {1} (t)} ^ {j}, \mathbf {c} _ {u ^ {2} (t)} ^ {i}, \mathbf {h} _ {u ^ {1} (t)} ^ {j}, \mathbf {h} _ {u ^ {2} (t)} ^ {j}, \Delta_ {u}, \Theta_ {j}\right)
$$

were  $h_{u(t)}^0 = \hat{x}_{u(t)}, j = 1, \dots, L$ . To train deeper dynamic neural networks easily, we employ the residual connection as the following form:  $skip(h_{u(t)}^{j-1}, h_{u(t)}^j) = \mathbf{W}_{\mathbf{skip}} h_{u(t)}^{j-1} + h_{u(t)}$  where  $\mathbf{W}_{\mathbf{skip}}$  is a weight matrix. Motivated by ELMo (Peters et al., 2018), we fuse all internal dynamic representations from all the layers to achieve rich dynamic representations. The fusion is a weighted summation of all layers defined as follows:  $\mathbf{h}_{\mathbf{u}(t)} = \gamma^{\mathrm{task}} \sum_{j=0}^{L} \alpha_j^{\mathrm{task}} \mathbf{h}_{u(t)}^j$ , where  $\alpha_j^{\mathrm{task}}$  are task-related softmax-normalized weights and  $\gamma^{\mathrm{task}}$  is a scaling parameter.

# 3.4 SELECTION

Given an interaction  $l_{u,v,t}$ , it is reasonable to assume that not all the historical interactive nodes have the equal importance for formalizing this interaction. Thus we use an attention mechanism to select

![](images/732822c905d043df300b22ea1d51df0ea1d42e6dc78a87f44b558b97bb095acc.jpg)  
Figure 4: Neural stacking, fusion and selection.

relevant nodes to learn dynamic representations and cell states of the current node. Specifically, a co-attention mechanism is used to measure relevance of historical time-evolving patterns between subgraph  $(u(t), k)$  and subgraph  $(v(t), k)$ ,

$$
\mathbf {Q} ^ {j} = \tanh \left(\mathbf {H} _ {u} ^ {j \top} \mathbf {W} _ {Q} \mathbf {H} _ {v} ^ {j}\right)
$$

where  $\mathbf{H}_u^j = \left[\mathbf{h}_1^j,\dots ,\mathbf{h}_a^j,\dots ,\mathbf{h}_m^j\right], a\in \text{subgraph}(u(t),k), \mathbf{H}_v^j = \left[\mathbf{h}_1^j,\dots ,\mathbf{h}_b^j,\dots ,\mathbf{h}_n^j\right], b\in \text{subgraph}(v(t),k)$ , and  $\mathbf{W}_Q\in \mathbb{R}^{d\times d}$  are the weight parameters. The  $\mathbf{Q}^j$  is a co-attention affinity matrix which captures the relevance information in subgraph  $(u(t),k)$  and subgraph  $(v(t),k)$ . The co-dependent global embedding  $p_u^j,p_v^j$  are obtained by the following equations.

$$
\mathbf {p} _ {u} ^ {j} = \mathbf {H} _ {u} ^ {j} \operatorname {S o f t M a x} \left(\underset {\text {C o l W i s e}} {\operatorname {M a x}} \mathbf {Q} ^ {j}\right) \qquad \mathbf {p} _ {v} ^ {j} = \mathbf {H} _ {v} ^ {j} \operatorname {S o f t M a x} \left(\underset {\text {R o w W i s e}} {\operatorname {M a x}} (\mathbf {Q} ^ {j}) \top\right)
$$

where Max means max-pooling operation which is used to choose the most relevant information for the maximum influence (or affinitie) on nodes in the corresponding subgraph. In addition, to adjust the importance of historical nodes, two adaptive gate functions are designed for previous nodes in  $\text{subgraph}(u(t), k)$  and  $\text{subgraph}(v(t), k)$  respectively,

$$
g _ {u} (p _ {u} ^ {j}, h _ {a} ^ {j}) = \sigma (w _ {p} p _ {u} ^ {j} + w _ {h} h _ {a} ^ {j}) g _ {v} (p _ {v} ^ {j}, h _ {b} ^ {j}) = \sigma (w _ {p} p _ {v} ^ {j} + w _ {h} h _ {b} ^ {j})
$$

where the weights  $\mathbf{w}_p$ , and  $\mathbf{w}_h$  are shared by all the stacked layers. Using these gates, we adjust dynamic node representations as follows:

$$
\left(\mathbf {h} _ {u (t)} ^ {j}, \mathbf {c} _ {u (t)} ^ {j}\right) = \mathrm {D I P - U N I T} ^ {j} \left(\mathbf {h} _ {u (t)} ^ {j - 1} \odot g _ {u} \left(\mathbf {p} _ {u (t)} ^ {j - 1}, \mathbf {h} _ {u (t)} ^ {j - 1}\right), \mathbf {c} _ {u ^ {1} (t)} ^ {j}, \mathbf {c} _ {u ^ {2} (t)} ^ {j}, \mathbf {h} _ {u ^ {1} (t)} ^ {j}, \mathbf {h} _ {u ^ {2} (t)} ^ {j}, \Delta_ {u}, \Theta_ {j}\right)
$$

Similarly, we can update  $(\mathbf{h}_{v(t)}^j,\mathbf{c}_{v(t)}^j)$  based on the selection mechanism.

# 3.5 CONDITIONAL INTENSITY FUNCTION

We model the dynamic interactions as a multi-dimensional temporal point process. Specifically, we define the conditional intensity function of the temporal point process at the dimension indexed by  $(u,v)$ , given the dependant histories of non-chain structures  $H_{t}^{u,v}$  where  $H_{t}^{u,v} = \text{subgraph}(u^{1}(t),k) \cup \text{subgraph}(u^{2}(t),k) \cup \text{subgraph}(v^{1}(t),k) \cup \text{subgraph}(v^{2}(t),k)$ , as follows:

$$
\lambda^ {u, v} (t | H _ {t} ^ {u, v}) = \operatorname {S o f t P l u s} \left(\mathbf {h} _ {t} ^ {u, v} \mathbf {w} _ {\lambda} + \mathbf {w} _ {t} ^ {\top} \tau + b _ {\lambda}\right)
$$

where

$$
\boldsymbol {h} _ {t} ^ {u, v} = \left[ \mathbf {h} _ {u ^ {1} (t)} ^ {\top}, \mathbf {h} _ {u ^ {2} (t)} ^ {\top}, \mathbf {h} _ {v ^ {1} (t)} ^ {\top}, \mathbf {h} _ {v ^ {2} (t)} ^ {\top} \right], \tau = \left[ \Delta_ {u, t}, \Delta_ {v, t} \right] ^ {\top},
$$

the scalar  $b_{\lambda}$  can be viewed as a base intensity level for the occurrence of the next interaction, and the SoftPlus function is used to ensure the non-negativity of the intensity. A key step for obtaining  $H_{t}^{u,v}$  is to get the k-hop subgraphs of  $u$  and  $v$ 's direct dependants Please see Appendix.A for more details about fast obtaining k-hop subgraphs

# 3.6 PARAMETER ESTIMATION

# 3.6.1 INTERACTION PREDICTION

Given a set of interactions as  $I = \{(u_i, v_i, t_i)\}_{i=1}^{i=N}$ , we can learn the model by minimizing the negative joint log-likelihood of  $I$  as follows:  $\mathcal{L}_1 = -\sum_i \log P^{u_i, v_i} (t_i | H_{t_i}^{u_i, v_i})$  where  $P^{u_i, v_i} (t_i | H_{t_i}^{u_i, v_i})$  represents the probability of formalizing an interaction between  $u_i$  and  $v_i$  at time  $t_i$  given the dependant history of non-chain structures  $H_{t_i}^{u_i, v_i}$ . Based on the intensity definition, we have  $\mathcal{L}_1 = \sum_i - \lambda^{u_i, v_i} (t_i | H_{t_i}^{u_i, v_i}) + \int_{t_i^-}^{t_i} \Lambda(t) dt$ , where  $t_i^-$  is the most recent time point when either  $u_i$  or  $v_i$  was involved in an interaction.  $\Lambda(t) = \sum_{u, v} \lambda^{u, v}(t)$  which represents total survival probabilities for interactions that do not happen. Since the survival part does not have an analytic solution, we apply Monte Carlo to do numerical integrations. We follow the negative sampling approaches proposed by Dai et al. (2016); Trivedi et al. (2019) to accelerate the survival term calculation.

# 3.6.2 INTERACTION CLASSIFICATION

An interaction sequence with markers is denoted as  $I' = \{(u_i, v_i, t_i, y_i)\}_{i=1}^{i=N}$ , where  $y_i$  is a marker at time  $t_i$  and usually is a discrete variable. In practice, the markers have different meanings in distinct scenes. A marker can be treated as a magnitude in modeling earthquakes and aftershocks, while in financial transaction scenes, the marker can be used to label whether a transaction is a fraudulent trading or not. The joint conditional density of an interaction  $(u_i, v_i, t_i)$  with marker  $y_i$  is given as  $P^{u_i, v_i}(t_i, y_i | \hat{H}_{t_i}^{u_i, v_i})$ . By applying the Bayesian rule, the joint conditional density can be written as:  $P^{u_i, v_i}(t_i, y_i | \hat{H}_{t_i}^{u_i, v_i}) = P^{u_i, v_i}(t_i | \hat{H}_{t_i}^{u_i, v_i}) P(y_i | t_i, \hat{H}_{t_i}^{u_i, v_i})$ , where  $P^{u_i, v_i}(t_i | \hat{H}_{t_i}^{u_i, v_i})$  has the same meaning as given in subsection 3.6.1, while  $P(y_i | t_i, \hat{H}_{t_i}^{u_i, v_i})$  means the distribution of  $y_i$  given the interaction happened at  $t_i$  with interaction history  $\hat{H}_{t_i}^{u_i, v_i}$ . It should be noted that the history  $\hat{H}_{t_i}^{u_i, v_i}$  contains the information of history markers and one need to design marker-specific intensity function (Mei & Eisner, 2017b). For simplicity, we assume  $P^{u_i, v_i}(t_i, y_i | \hat{H}_{t_i}^{u_i, v_i})$  is independent on history markers. Meanwhile,  $P(y_i | t_i, \hat{H}_{t_i}^{u_i, v_i})$  can be obtained by a multinomial function:

$$
P \left(y _ {i} = c \mid \boldsymbol {h} _ {u _ {i}, v _ {i}, t _ {i}}\right) = \frac {\exp \left(\boldsymbol {V} _ {c} ^ {y} \boldsymbol {h} _ {u _ {i} , v _ {i} , t _ {i}}\right)}{\sum_ {c = 1} ^ {C} \exp \left(\boldsymbol {V} _ {c} ^ {y} \boldsymbol {h} _ {u _ {i} , v _ {i} , t _ {i}}\right)}
$$

where  $\mathbf{h}_{u_i,v_i,t_i}$  is the concatenation of  $\mathbf{h}_{u_i}$  and  $\mathbf{h}_{v_i}$  which can be regarded as dynamic representation for an interaction between  $u_i$  and  $v_i$  at  $t_i$ .  $C$  is the number of markers,  $V_c^y$  is the c-th row of matrix  $V^y$ . Then the final objective function for interaction classification can be obtained as follows:  $\mathcal{L}_2 = \mathcal{L}_1 + \mathcal{L}_{cross-entropy}$ , where  $\mathcal{L}_{cross-entropy}$  is a cross-entropy loss over marks:

$$
\mathcal {L} _ {c r o s s - e n t r o p y} = - \sum_ {i = 1} ^ {i = N} \sum_ {c = 1} ^ {C} y _ {i} \cdot l o g P (y _ {i} = c | \boldsymbol {h} _ {u _ {i}, v _ {i}, t _ {i}})
$$

# 4 RELATED WORK

Inspired by the Skip-gram (Mikolov et al., 2013) for word embedding, a series of node embedding methods based on the random walks on graphs have been proposed(Perozzi et al., 2014; Tang et al., 2015; Grover & Leskovec, 2016; Wang et al., 2016; 2017). GCN and its variants (Bruna et al., 2013; Hamilton et al., 2017a; Kipf & Welling, 2017) are a recent class of algorithms which extend convolutions from spatial domains to graph-structured domains. Meanwhile they can efficiently generate node embeddings for previously unseen data. All models above are designed for static graphs. The intuitive and popular approaches for modeling dynamic graphs are based on a sequence for graph snapshots(Goyal et al., 2018; Zhou et al., 2018; Seo et al., 2018; Yu et al., 2018), while it can be difficult to specify the appropriate aggregation granularity. Nguyen et al. (2018) adds a temporal constraint on random walk sampling, but it can't model the rich temporal information explicitly. Temporal point processes (TPPs) are an another alternative to model dynamics(Daley & Vere-Jones, 2007). Several dynamic graph modeling methods based on the TPPs (Dai et al., 2016; Trivedi et al., 2019) have been proposed. Our method DIP differs from these TPP-based methods by

the extension of the LSTM model over temporal dependency graphs, the multiple time resolution modeling via stacking and fusing, and the selection mechanism. Detailed related work are included in Appendix.C.

# 5 EXPERIMENTS

We evaluate the proposed DIP model on the task of Interaction Prediction and Interaction classification on several real-world datasets.

# 5.1 BASELINES

GraphSage(Hamilton et al., 2017a) is an inductive graph neural network framework consisting of three different aggregators which are GCN, Mean and LSTM aggregators respectively. We report the best result among these three aggregators noted as Graphsage*. What's more, for comparing with GAT(Velicković et al., 2017) we also implement a graph attention aggregator based on GraphSage. CTDNE(Nguyen et al., 2018) is a newly-proposed temporal network embedding method which is an extension of DeepWalk(Perozzi et al., 2014) by incorporating temporal order constraint when sampling sequences of walks from time-continuous graphs. DynGEM(Goyal et al., 2018) takes a sequence of static graph snapshots as inputs to learn node embeddings by a deep auto-encoder network. DeepCoevolve (Dai et al., 2016) models dynamic interaction sequences with two co-evolution recurrent neural networks. Hidden embeddings are learned for interactive nodes after each interaction. DyREP (Trivedi et al., 2019) uses a two-time scale deep temporal point process model to capture dynamics of graphs.

# 5.2 EXPERIMENTAL SETTING

We conduct all the experiments with a hyper-parameter grid search strategy. For all methods, we search the hidden vector dimension from  $\{32, 64, 128, 256\}$  and the learning rate from  $\{0.01, 0.001, 0.0005, 0.0001, 0.00001\}$ . For our DIP model, we go through  $\{1, 2, 3, 4\}$  for  $K$  and  $L$ . For Graphsage, the maximum number of 1-hop and 2-hop neighbor nodes are set to 25 and 20 respectively. The batch sizes for all candidates are  $\{100, 300, 500\}$ . All the models are trained for at most 50 epochs with an early-stop if the performance does not improve for 5 epochs. For Graphsage, DynGEM and DeepCoevolve, we use the open source codes provided by the authors. We implement the CTNDE and GAT based on the Graphsage framework, and implement DyREP based on the pytorch implementation of DeepCoevolve. After the best configuration has been found, we repeat the full experiments 5 times and report the mean results and standard deviation.

# 5.3 INTERACTION PREDICTION

# 5.3.1 DATASETS

CollegeMsg(Leskovec & Krevl, 2014) consists of sending message interactions on an online social network at the University of California, Irvine during 193 days. Ubuntu(Leskovec & Krevl, 2014) is a temporal interaction dataset extracted from the stack exchange website. An interaction between two users means one answered another's questions or replied to his/her posts. Amazon(McAuley et al., 2015) is composed of commodity rating data from amazon users. We use the Clothing subset of this dataset. MathOverflow(Leskovec & Krevl, 2014) is comprised of interactions of commenting an existing answer on the Math Overflow website. Table 1 shows the detailed dataset statistics. In this table, Sparsity indicates  $|Edge| / (|u(t)| * |v(t)|)$ . And the Duplication is the quotient of # of temporal edges divided by # of statics edges. Duplication equals 1.0 means each unique pair of participants interact only once. We also give the average time interval of two consequent interactions of a certain participant as Avg  $\Delta t$ . For each dataset, we first construct the corresponding dependency graph as described in section3.1. Then we sort these interactions by occurrence time. The first  $60\%$  interactions are adopted as training set and the next  $20\%$  interactions are used for validation. The last  $20\%$  interactions are left as test set. The Cold-start participants which only exist in validation set or test set are removed.

Table 1: Dataset Statistics  

<table><tr><td></td><td>CollegeMsg</td><td>Ubuntu</td><td>Amazon-Clothing</td><td>Math Overflow</td></tr><tr><td># Train</td><td>35902</td><td>204846</td><td>50209</td><td>58596</td></tr><tr><td># Valid</td><td>7814</td><td>39913</td><td>9195</td><td>24045</td></tr><tr><td># Test</td><td>5055</td><td>35271</td><td>7598</td><td>32705</td></tr><tr><td>Duplication</td><td>3.0569</td><td>1.886</td><td>1.000</td><td>2.819</td></tr><tr><td>Avg Δt(Hours)</td><td>27.79</td><td>575.89</td><td>3399.82</td><td>415.46</td></tr><tr><td>Sparsity</td><td>0.01181</td><td>0.00074</td><td>0.00038</td><td>0.00567</td></tr></table>

# 5.3.2 EVALUATION PROTOCOL AND RESULTS

For each interaction  $l_{u,v,t}$  in test set, we fix the first participant  $u$ , then replace the second participant  $v$  by all possible participant candidates. The conditional density  $p^{u,v}(t) = \lambda^{u,v}(t)S^{u,v}(t)$  for all candidates are first computed and then sorted by a descending order. The rank of the correct participant is finally stored and denoted as  $rank_i$  for the i-th test interaction. We report the Mean Rank defined as  $Mean\,Rank = \frac{\sum\,rank_i}{\#of\,Test\,Interaction}$ , which can represent the overall performance. Figure 5 summarizes the Mean Rank performance of all the baselines and our DIP method. DIP outperforms all baselines by  $65.84\%$ ,  $41.64\%$ ,  $10.69\%$  and  $43.99\%$  over the four datasets. When the average  $\Delta t$  is small, which means interactions are sensitive to temporal information, our model achieves the best performance. This gives the evidence of the effectiveness of explicitly modeling the temporal information. What's more, for the CollegeMsg dataset which is densest and has the most duplicated interactions, our model outperforms best again.

# 5.4 INTERACTION CLASSIFICATION

We conduct this task on an industrial dataset: Huabei Trade Data. For the performance measures, we employ KS(Kolmogorov-Smirnov) value which is a big concern of the loans provider in anti-fraud detection, as well as AUC(Area under the ROC Curve) score. This dataset consists of about 150,000 transaction records processed by Huabei during August 2018. Each transaction is initiated with three parties: the buyer, the seller and transaction details such as merchant category and transaction amount. Around  $15\%$  of the transaction are fraudulent and is labeled by a complicated Ex-Post method. In order to conducting an Ex-ante detection, we are required to find out fraudulent transaction at the time of initiating using those basic transaction features. For each interaction event, there are 11 context features that can be obtained when the trade request is created, including information about buyer types, seller types, purchased items' categories and trading platform. We use the first 10 days data as training set, the following 10 days data as validation set, the rest as test set. Note that, in this scenario, there are always users who only appear in validation/testing dataset. Thus, traditional transductive methods are not applicable on this task. Alternatively, we employ the XGBoost (Chen & Guestrin, 2016) as an additional baseline which is a popular model in the cash-out detection task(Hu et al., 2019). Table 2 compares the results. Obviously, our model outperforms all the baseline methods again. The contributions of different modules of the proposed DIP models are given in figure 7.

Table 2: Interaction classification results  

<table><tr><td></td><td>Xgboost</td><td>GraphSage*</td><td>GAT</td><td>DIP</td></tr><tr><td>AUC</td><td>0.6818 ±0.0023</td><td>0.8603 ±0.0005</td><td>0.8597 ±0.0004</td><td>0.9017 ±0.0004</td></tr><tr><td>KS</td><td>0.2536 ±0.0015</td><td>0.5934 ±0.0012</td><td>0.6018 ±0.0005</td><td>0.6703 ±0.0060</td></tr></table>

# 5.5 ABLATION STUDY

As we described in Section 3, the DIP model consists of three important components: First, it uses a Time Gate in the DIP neural unit to explicitly model the temporal information. Second, the selection mechanism enables our model to select more important historical information for interactions. Third,

![](images/4235ca4ec1e732fc94973dac242e4811c1bb654e7dc10b538a96e70617046aa2.jpg)

![](images/5192a48bb4cc5707823e239eae6e4a03d3ed504177f7fce50307c428677536fb.jpg)

![](images/4a7231dabe3a134c3ab7ced6b6ff45342dd48b6943e57225f0fbb103ec7c0f00.jpg)  
Figure 5: Mean rank results

![](images/694794c1f56ac5daab1c224489f7e652243888fe299cc88a1b5034c2357416b7.jpg)

![](images/be6999880b2ddb16f13785d6a9c14265a868f4b94ff8560de0238bbf46c68ddc.jpg)

![](images/da7edc0e3bdcf071c50a8d098be377b2c1533f9b3bdcec77d0e76d66ac42ddc2.jpg)

![](images/fc30332f7bac2c24174409d6b021ee3c89739f46ba2dea55e439ce0d1081404b.jpg)  
Figure 6: Ablation study results of the interaction prediction task

![](images/7db9b4c9e68a84564010e4871f2b667ac148c8d0c6aba796b56ac9951c1e4c22.jpg)

the Fusion of multi-layer DIP-UNIT's hidden state vector helps to extract high level feature. We investigate the contribution of each component by disabling each of them one by one, and compare the corresponding result to the full model. figure 6 and figure 7 give the detailed ablation results. FullModel in the two figures means all the three components are enabled.

![](images/d088b836c142fe4eb5e70d2ef061ecefd4e4754f4f95f51893a90ede1953de79.jpg)  
Figure 7: Ablation study results of the interaction classification task

![](images/373a81849a7c67d99ce6397b42204047c1e345b953837f0b61089049ed3a147f.jpg)

- No Time Gate: In this configuration, the time gate in DIP-UNIT is disabled. This leads to a significant drop of the Mean Rank performance. It provides a strong evidence for the effectiveness of the time gate.  
- No Selection: In this configuration the selection mechanism is disabled. Accordingly, all the historical node representations contribute equally, thus again leading to a performance drop.  
- No Fusion: In this variant, we directly use the hidden state vector of the last layer. Again, the performance degrades significantly. This demonstrates that a fusion of different layers' representations gives richer information than the last layer only.

# 6 CONCLUSIONS

In this paper, we have proposed a deep multidimensional point process approach, DIP, to learn dynamic graph representations. We generalize LSTM over temporal dependency graphs and model multiple time resolutions via stacking, selection and fusion. Experimental results show the effectiveness of the components of our neural unit and the superior performance on several datasets.

# REFERENCES

Odd Aalen, Ornulf Borgan, and Hakon Gjessing. Survival and event history analysis: a process point of view. Springer Science & Business Media, 2008.  
Amr Ahmed, Nino Shervashidze, Shravan Narayanamurthy, Vanja Josifovski, and Alexander J Smola. Distributed large-scale natural graph factorization. In Proceedings of the 22nd international conference on World Wide Web, pp. 37-48. ACM, 2013.  
Mikhail Belkin and Partha Niyogi. Laplacian eigenmaps and spectral techniques for embedding and clustering. In Advances in neural information processing systems, pp. 585-591, 2002.  
Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally connected networks on graphs. arXiv preprint arXiv:1312.6203, 2013.  
Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm SIGkdd international conference on knowledge discovery and data mining, pp. 785-794. ACM, 2016.  
Hanjun Dai, Yichen Wang, Rakshit Trivedi, and Le Song. Deep coevolutionary network: Embedding user and item features for recommendation. arXiv preprint arXiv:1609.03675, 2016.  
Daryl J Daley and David Vere-Jones. An introduction to the theory of point processes: volume II: general theory and structure. Springer Science & Business Media, 2007.  
Michael Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. neural information processing systems, pp. 3844-3852, 2016.  
Rianne Van Den Berg, Thomas N Kipf, and Max Welling. Graph convolutional matrix completion. arXiv: Machine Learning, 2017.

Nan Du, Hanjun Dai, Rakshit Trivedi, Utkarsh Upadhyay, Manuel Gomez-Rodriguez, and Le Song. Recurrent marked temporal point processes: Embedding event history to vector. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1555-1564. ACM, 2016.  
Palash Goyal, Nitin Kamra, Xinran He, and Yan Liu. Dyngem: Deep embedding method for dynamic graphs. arXiv preprint arXiv:1805.11273, 2018.  
Aditya Grover and Jure Leskovec. node2vec: Scalable feature learning for networks. knowledge discovery and data mining, pp. 855-864, 2016.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, pp. 1024-1034, 2017a.  
William L Hamilton, Jure Leskovec, and Dan Jurafsky. Diachronic word embeddings reveal statistical laws of semantic change. arXiv preprint arXiv:1605.09096, 2016.  
William L Hamilton, Rex Ying, and Jure Leskovec. Representation learning on graphs: Methods and applications. IEEE Data(base) Engineering Bulletin, 40:52-74, 2017b.  
Alan G Hawkes. Spectra of some self-exciting and mutually exciting point processes. Biometrika, 58 (1):83-90, 1971.  
Binbin Hu, Zhiqiang Zhang, Chuan Shi, Jun Zhou, Xiaolong Li, and Yuan Qi. Cash-out user detection based on attributed heterogeneous information network with a hierarchical attention mechanism. 2019.  
John Frank Charles Kingman. Poisson processes. Encyclopedia of biostatistics, 6, 2005.  
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. international conference on learning representations, 2017.  
Vivek Kulkarni, Rami Al-Rfou, Bryan Perozzi, and Steven Skiena. Statistically significant detection of linguistic change. In Proceedings of the 24th International Conference on World Wide Web, pp. 625-635. International World Wide Web Conferences Steering Committee, 2015.  
Jure Leskovec and Andrej Krevl. SNAP Datasets: Stanford large network dataset collection. http://snap.stanford.edu/data, June 2014.  
Jure Leskovec, Jon Kleinberg, and Christos Faloutsos. Graph evolution: Densification and shrinking diameters. ACM Transactions on Knowledge Discovery from Data (TKDD), 1(1):2, 2007.  
Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel. Image-based recommendations on styles and substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 43-52. ACM, 2015.  
Hongyuan Mei and Jason M Eisner. The neural hawkes process: A neurally self-modulating multivariate point process. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems 30, pp. 6754-6764. Curran Associates, Inc., 2017a.  
Hongyuan Mei and Jason M Eisner. The neural hawkes process: A neurally self-modulating multivariate point process. In Advances in Neural Information Processing Systems, pp. 6754-6764, 2017b.  
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pp. 3111-3119, 2013.  
Federico Monti, Michael M Bronstein, and Xavier Bresson. Geometric matrix completion with recurrent multi-graph neural networks. neural information processing systems, pp. 3697-3707, 2017.

Giang Hoang Nguyen, John Boaz Lee, Ryan A. Rossi, Nesreen K. Ahmed, Eunyee Koh, and Sungchul Kim. Continuous-time dynamic network embeddings. In *Companion of the The Web Conference 2018 on The Web Conference* 2018, WWW 2018, Lyon, France, April 23-27, 2018, pp. 969-976, 2018.  
Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 701-710. ACM, 2014.  
Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In Proc. of NAACL, 2018.  
Sam T Roweis and Lawrence K Saul. Nonlinear dimensionality reduction by locally linear embedding. science, 290(5500):2323-2326, 2000.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61-80, 2009.  
Youngjoo Seo, Michael Defferrard, Pierre Vandergheynst, and Xavier Bresson. Structured sequence modeling with graph convolutional recurrent networks. In International Conference on Neural Information Processing, pp. 362-373. Springer, 2018.  
Sucheta Soundarajan, Acar Tamersoy, Elias B Khalil, Tina Eliassi-Rad, Duen Horng Chau, Brian Gallagher, and Kevin Roundy. Generating graph snapshots from streaming edge data. In Proceedings of the 25th International Conference Companion on World Wide Web, pp. 109-110. International World Wide Web Conferences Steering Committee, 2016.  
Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei. Line: Large-scale information network embedding. In Proceedings of the 24th International Conference on World Wide Web, pp. 1067-1077. International World Wide Web Conferences Steering Committee, 2015.  
Joshua B Tenenbaum, Vin De Silva, and John C Langford. A global geometric framework for nonlinear dimensionality reduction. science, 290(5500):2319-2323, 2000.  
Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. Dyrep: Learning representations over dynamic graphs. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019, 2019. URL https://openreview.net/forum?id=HyePrhR5KX.  
Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.  
Daixin Wang, Peng Cui, and Wenwu Zhu. Structural deep network embedding. pp. 1225-1234, 2016.  
Xiao Wang, Peng Cui, Jing Wang, Jian Pei, Wenwu Zhu, and Shiqiang Yang. Community preserving network embedding. In Thirty-First AAAI Conference on Artificial Intelligence, 2017.  
Shuai Xiao, Mehrdad Farajtabar, Xiaojing Ye, Junchi Yan, Le Song, and Hongyuan Zha. Wasserstein learning of deep generative point process models. In Advances in Neural Information Processing Systems, pp. 3247-3257, 2017.  
Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L Hamilton, and Jure Leskovec. Graph convolutional neural networks for web-scale recommender systems. *knowledge discovery* and data mining, pp. 974-983, 2018.  
Wenchao Yu, Wei Cheng, Charu C Aggarwal, Kai Zhang, Haifeng Chen, and Wei Wang. Netwalk: A flexible deep embedding approach for anomaly detection in dynamic networks. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 2672-2681. ACM, 2018.  
Lekui Zhou, Yang Yang, Xiang Ren, Fei Wu, and Yueting Zhuang. Dynamic network embedding by modeling triadic closure process. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

Linhong Zhu, Dong Guo, Junming Yin, Greg Ver Steeg, and Aram Galstyan. Scalable temporal latent space inference for link prediction in dynamic social networks. IEEE Transactions on Knowledge and Data Engineering, 28(10):2765-2777, 2016.
