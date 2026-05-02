# LEARNING SUCCESSOR REPRESENTATIONS WITH DISTRIBUTED HEBBIAN TEMPORAL MEMORY

Anonymous authors

Paper under double-blind review

# ABSTRACT

This paper presents a novel approach to address the challenge of online hidden representation learning for decision-making under uncertainty in non-stationary, partially observable environments. The proposed algorithm, Distributed Hebbian Temporal Memory (DHTM), is based on factor graph formalism and a multicomponent neuron model. DHTM aims to capture sequential data relationships and make cumulative predictions about future observations, forming Successor Representation (SR). Inspired by neurophysiological models of the neocortex, the algorithm utilizes distributed representations, sparse transition matrices, and local Hebbian-like learning rules to overcome the instability and slow learning process of traditional temporal memory algorithms like RNN and HMM. Experimental results demonstrate that DHTM outperforms classical LSTM and performs comparably to more advanced RNN-like algorithms, speeding up Temporal Difference learning for SR in changing environments. Additionally, we compare the SRs produced by DHTM to another biologically inspired HMM-like algorithm, CSCG. Our findings suggest that DHTM is a promising approach for addressing the challenges of online hidden representation learning in dynamic environments.

# 1 INTRODUCTION

Modelling sequential data is one of the most important tasks in Artificial Intelligence as it has many applications, including decision-making and world models, natural language processing, conversational AI, time-series analysis, and video and music generation (Min et al., 2021; Eraslan et al., 2019; Dwivedi et al., 2023; Ji et al., 2020; Moerland et al., 2023). One of the classical approaches to modelling sequential data is forming a representation that stores and condenses the most relevant information about a sequence, and finding a general transformation rule of this information through the dimension of time (Lipton et al., 2015; Harshvardhan et al., 2020; Mathys et al., 2011). We refer to the class of algorithms that use this approach as Temporal Memory (TM) algorithms, as they essentially model the cognitive ability of complex living organisms to remember the experience and make future predictions based on this memory (Hochreiter & Schmidhuber, 1997; Friston et al., 2016; 2018; Parr & Friston, 2017).

This paper addresses the problem of hidden representation learning for decision-making under uncertainty, which can be formalized as agent Reinforcement Learning (RL) for a Partially Observable Markov Decision Process (POMDP) (Poupart, 2005). Inferring the hidden state in a partially observable environment is, in effect, a sequence modelling problem as it requires processing a sequence of observations to get enough information about hidden states. One of the most efficient representations of the hidden states for discrete POMDP is the Successor Representation (SR) that disentangles hidden states and goals given by the reward function (Dayan, 1993). An extension of the SR into continuous POMDP is the Successor Features framework, which employs the same idea of value function decomposition, but, instead, for features of a hidden state (Barreto et al., 2017). Temporal Memory algorithms can be leveraged to make cumulative predictions about future states and their features to form SR or SF.

The most prominent TM algorithms, like a Recurrent Neural Network (RNN) or a Hidden Markov Model (HMM), use backpropagation to capture data relationships, which is known for its instability due to recurrent non-linear derivatives. They also require having complete sequences of data at hand during the training. Although the gradient vanishing problem can be partially circumvented in a

way Receptance Weighted Key Value (RWKV) (Peng et al., 2023) or Linear Recurrent Unit (LRU) (Orvieto et al., 2023) models do, the problem of online learning is still a viable topic. In contrast to HMM, RNN models and their descendants also lack a probabilistic theory foundation, which is beneficial for modeling sequences captured from stochastic environments (Salaun et al., 2019; Zhao et al., 2020). There is little research on TM models that can be used in fully online adaptable systems interacting with partially observable stochastic environments with access only to one sequence data point at a time, a prevalent case in Reinforcement Learning (Jahromi et al., 2022).

We propose a Distributed Hebbian Temporal Memory (DHTM) algorithm based on the factor graph formalism and multi-compartment neuron model. The resulting graphical structure of our model is similar to one of the Factorial-HMM (Ghahramani & Jordan, 1995), but with a factor graph forming online during training. We also show that depending on the graphical structure, our TM can be viewed as an HMM version of either RNN or LRU regarding information propagation in time. An important feature of our model is that transition matrices for each factor are stored as different components (segments) of artificial neurons, which makes computations very efficient in the case of sparse transition matrices. Our TM forms sequence representations fully online and employs only local Hebbian-like learning rules (Hebb, 2005; Churchland & Sejnowski, 1992; Lillicrap et al., 2020), circumventing gradient drawbacks and making the learning process much faster than gradient methods.

Some key ideas for our TM algorithm are inspired by neurophysiological models of the neocortex neural circuits and pyramidal neurons (George & Hawkins, 2009; Hawkins & Ahmad, 2016; O'Reilly et al., 2021). For example, emission matrices for random variables are fixed to resemble the columnar structure of the neocortex layers, which significantly lessens the number of trainable parameters, speeding up learning and leading to sparse transition matrices. Another example is using multi-compartmental neurons with active dendritic segments as independent detectors of neuron pattern activity (London & Häusser, 2005). Each dendritic segment can be viewed as a row of an HMM state transition matrix or, more generally, a value of a discrete factor function. Thus, we don't explicitly store large transition matrices, only their non-zero parts.

The DHTM model notoriously fits Successor Features in the Reinforcement Learning setup to speed up TD learning. The proposed TM is tested as a world model (Ha & Schmidhuber, 2018; Hafner et al., 2023) for an RL agent architecture, making decisions in a simple Pinball-like environment and in a more challenging AnimalAI testbed (Crosby et al., 2020). Our algorithm outperforms a classic RNN algorithm LSTM and a more advanced RNN-like transformer algorithm RWKV in online Successor Feature formation task due to combination of fast Hebbian-like learning and sparse hidden state coding. Another advantage of our algorithm is that it allows its implementation for neuromorphic processors, as it uses only local learning rules.

Our contribution in this work is the following:

- We propose a distributed memory model DHTM based on the factor graph formalism and multicompartment neural model.  
- Our model stores sparse factor functions in neural segments, which significantly lessens the number of trainable parameters and speeds up learning.  
- The DHTM learns fully online employing only local Hebbian-like rules.  
- The DHTM model fits Successor Features in the RL setup to speed up TD learning.  
- Tested as a world model for an RL agent architecture in a Pinball environment, DHTM outperforms LSTM and RWKV in online Successor Features formation task.

# 2 BACKGROUND

This section provides basic information about some concepts necessary to follow the paper.

# 2.1 REINFORCEMENT LEARNING

In this paper, we consider decision-making in a partially observable environment, which is usually formalized as Partially Observable Decision Process (Poupart, 2005). A POMDP is defined as a tuple  $\mathcal{M} = (S,A,P,R,O,D,\gamma)$ , where  $S$ -state space,  $A$ -action space,  $P(s,a,s^i) =$

$Pr(s^i \mid s, a)$  — transition function,  $R(s)$ -reward function, O—observation space,  $D(a, s^i, o) = Pr(o \mid a, s^i)$ —sensor model and  $\gamma \in [1, 0)$ —discount factor, given a transition  $s, a \to s^i$ , where  $s \in S$ ,  $a \in A$ ,  $o \in O$ . If  $S, A, O$  are finite,  $P, D$  can be viewed as real valued matrices, otherwise, they are conditional density functions. Here we consider deterministic rewards, which depend only on the current state, i.e.  $R(s): S \to \mathbb{R}$ .

The task of RL is to find a policy  $\pi(a \mid s): S \times A \to [0,1]$ , which maximizes expected return  $G = \mathbb{E}[\sum_{t=0}^{T} \gamma^l R_t]$ , where  $T$  is an episode length. For value based methods, it is convenient to define optimal policy via Q-function:  $Q^\pi(s_t, a_t) = \mathbb{E}[\sum_{l \geq t} \gamma^l R(s_{l+1}) \mid s_t, a_t, \pi]$ . For an optimal value function  $Q^*$  an optimal policy can be defined as  $\pi(a \mid s) = \operatorname*{argmax}_{a} Q^*(s, a)$ .

# 2.2 HIDDEN MARKOV MODEL

Partially observable Markov process can be approximated by a Hidden Markov model (HMM) with hidden state space  $H$  and observation space  $O$ .  $O$  is the same as in  $\mathcal{M}$ , but  $H$  generally is not equal  $S$ . Variables  $H_{t}$  represent an unobservable (hidden) approximated state of the environment which evolves over time, and observable variables  $O_{t}$  represent observations that depend on the same time step state  $H_{t}$ , and  $h_t, o_t$  are corresponding values of this random variables. For the sake of simplicity, we suppose that actions are fully observable and information about them is included into  $H_{t}$  variables. For the process of length  $T$  with state values  $h_{1:T} = (h_1,\dots,h_T)$  and  $o_{1:T} = (o_1,\dots,o_T)$ , the Markov property yields the following factorization of the generative model:

$$
p \left(o _ {1: T}, h _ {1: T}\right) = p \left(h _ {1}\right) \prod_ {t = 2} ^ {T} p \left(h _ {t} \mid h _ {t - 1}\right) \prod_ {t = 1} ^ {T} p \left(o _ {t} \mid h _ {t}\right). \tag {1}
$$

In case of discrete hidden state, a time-independent stochastic transition matrix can be learned with Baum-Welch algorithm (Baum et al., 1970), a variant of Expectation Maximization algorithm. To compute the statistics for the expectation step, it employs the forward-backward algorithm, which is a special case of sum-product algorithm (Kschischang et al., 2001).

# 2.3 SUCCESSOR REPRESENTATION

Successor Representations are such representations of hidden states from which we can linearly infer the state value given the reward function (Dayan, 1993). Here, we assume observation and state spaces are discrete.

$$
\begin{array}{l} V \left(h _ {t} = i\right) = \mathrm {E} \left[ \sum_ {l = 0} ^ {\infty} \gamma^ {l} R _ {t + l + 1} \mid h _ {t} = i \right] = \sum_ {l = 0} ^ {\infty} \gamma^ {l} \mathrm {E} \left[ R _ {t + l + 1} \mid h _ {t} = i \right] = \\ = \sum_ {l = 0} ^ {\infty} \gamma^ {l} \sum_ {j} p \left(h _ {t + l + 1} = j \mid h _ {t} = i\right) R _ {j} = \sum_ {j} \sum_ {l = 0} ^ {\infty} \gamma^ {l} p \left(h _ {t + l + 1} = j \mid h _ {t} = i\right) R _ {j} = \sum_ {j} M _ {i j} R _ {j}, \\ \end{array}
$$

where  $\gamma$  is a discount factor, vector  $\mathrm{SR}(h = i) = \{M_{ij}\}_{j}$  is a Successor Representation of a state  $i$ , and  $M_{ij} = \sum_{l=0}^{\infty} \gamma^{l} p(h_{t+l+1} = j \mid h_{t} = i)$ .  $R_{j}$  is a reward for observing the state  $j$ . That is, SR can be computed by a TM that is able to predict future states. TM algorithms effectively predict observations only for a finite time horizon  $T$ . Therefore, in order to learn SR, a technique similar to TD learning in standard RL may be employed:

$$
\delta_ {i j} = \sum_ {l = 0} ^ {T} \gamma^ {l} p \left(h _ {t + l + 1} = j \mid h _ {t} = i\right) + \gamma^ {T + 1} \sum_ {k} M _ {k j} p \left(h _ {t + T + 1} = k \mid h _ {t} = i\right) - M _ {i j}, \tag {2}
$$

$$
M _ {i j} \leftarrow M _ {i j} + \alpha \delta_ {i j}, \tag {3}
$$

where  $\alpha \in (0,1)$  is a learning rate,  $\delta_{ij}$  TD error for SR.

In partially observable environments, however, exact state values are not known, therefore we operate with state distributions or so-called belief states (Poupart, 2005), which are inferred from observations. In that case, state value and SR are functions of hidden state variable distribution (see details in Appendix B).

![](images/bc3a3cbb2892d0a90ea3486b21ca68089d290009ad77fcf591de38502f310bd7.jpg)  
Figure 1: Partial factor graph for the DHTM. The input to the model is a sequence of binary images, each pixel is modelled as Bernoulli random variable  $O_{t}^{lm}$ , where  $l$  and  $m$  denote corresponding rows and cols of the image. The encoder block forms image categorical features  $\Phi_t^k$  in an unsupervised manner. Each feature  $\Phi$  has its own explaining hidden variable, which may depend on hidden variables of the other features and on itself from the previous time step.  $F_{c}^{k}$  and  $F_{e}^{k}$  are context and emission factors for the corresponding variables. Unary factors  $M_{t - 1}^{i}$  called messages represent accumulated information about previous time steps.

# 2.4 SPARSE DISTRIBUTED REPRESENTATIONS

In our work, we design our model to operate with sparse distributed representations (SDRs) to reflect the spatiotemporal property of cortical network activity (Perin et al., 2011). In the discrete time case, SDR is a sparse binary vector in a high-dimensional space. To encode observed dense binary patterns to SDRs, we use a biologically plausible k-WTA (k-winners take all) neural network algorithm called spatial pooler with a Hebbian-like unsupervised learning method (see details in Appendix A).

# 3 DISTRIBUTED HEBBIAN TEMPORAL MEMORY

# 3.1 FACTOR GRAPH MODEL

Distributed Hebbian Temporal Memory is based on the sum-product belief propagation algorithm in a factor graph (see Figure 1). Analogously to Factorial-HMM (Ghahramani & Jordan, 1997), we divide the hidden space  $H$  into subspaces  $H^k$ . There are four sets of random variables (RV) in the model:  $H_{t-1}^i$ —latent variables representing hidden states from the previous time step (context),  $H_t^k$ —latent variables for the current time step,  $\Phi_t^k$ —feature variables, and  $O_t^{lm}$ —observable variables. Except for  $O_t^{lm}$ , all random variables have a categorical distribution. In contrast,  $O_t^{lm}$ , are Bernoulli variables because they represent pixels from a binary input image observation. RV state values are denoted as corresponding lowercase letters:  $h_{t-1}^i$ ,  $h_t^k$ ,  $\varphi_t^k$ ,  $o_t^{lm}$ .

Each variable  $\Phi_t^k$  is considered independent and has a separate graphical model for increased computational efficiency. However, hidden variables of the same time step are statistically interdependent in practice. We introduce their interdependence through a segment computation trick that goes beyond the standard sum-product algorithm (see Eq. 7).

The model also has three types of factors:  $M_{t-1}^{i}$ —messages from previous time steps,  $F_{c}^{k}$ —context factor (generalized transition matrix),  $F_{e}^{k}$ —emission factor. We assume that messages  $M_{t-1}^{i}$  include posterior information from the time step  $t - 1$ , therefore we don't depict observable variables for previous time steps in Figure 1.

![](images/74f8e282cc9773a7596ff4db17c56b8a7e71ee612deca8c33b12c69a1d109b71.jpg)  
Figure 2: Neuronal implementation of the DHTM. Random variables are represented by cell clusters (white circles), where each cell corresponds to a state and its spike frequency—to the probability of the state  $p(h_{t}^{k})$ . Cell's dendritic segments  $seg(k)$  correspond to context factor values  $f_{l}$  for a particular combination of states (active presynaptic cells)  $rec(l)$ . Segments' excitations  $E_{l}$  are combined to determine cell's spike frequency  $p(h_{t}^{k})$ . Segment's synaptic weights reflect specificity of  $rec(l)$  combination for the segment. Emission factors  $F_{e}^{k}$  are fixed and represented by minicolumns inside a variable.

![](images/d942ceffa3c72b69d2af559727daaddcb509735b4e45af3652f80bcb77a60d21.jpg)

Further, we discuss only the upper block of the graph, which is DHTM itself. The lower block—an encoder—is described in the Appendix A. The only requirement for the encoder is that its output should be represented as states of categorical variables (features) for the current observation.

# 3.2 NEURAL IMPLEMENTATION

The main routine of the DHTM is to estimate distributions of currently hidden state variables given by the equation 4, the computational flow of which is schematically depicted in Figure 2:

$$
p \left(h _ {t} ^ {k}\right) \propto \sum_ {\left\{h _ {t - 1} ^ {i}: i \in \omega_ {k} \right\}} \prod_ {i \in \omega_ {k}} M _ {t - 1} ^ {i} \left(h _ {t - 1} ^ {i}\right) F _ {c} ^ {k} \left(h _ {t} ^ {k}, \left\{h _ {t - 1} ^ {i}: i \in \omega_ {k} \right\}\right), \tag {4}
$$

where  $\omega_{k} = i_{1},\ldots ,i_{n}$  -set of previous time step RV indexes included in  $F_{c}^{k}$  factor,  $(n + 1)$  -factor size.

For computational purposes, we translate the problem to the neural network architecture with Hebbian-like learning (for biological interpretation of the model, see Appendix C). As can be seen from Figure 2, every RV can be viewed as a set of spiking neurons representing the RV's states, that is,  $p(h_{t}^{k}) = p(c_{t}^{j} = 1)$ , where  $j$  — index of a neuron corresponding to the state  $h_{t}^{k}$ . Cell activity is binary  $c_{t}^{j} \in \{0,1\}$  (spike/no-spike), and the probability might be interpreted as a spike rate. Factors  $F_{c}^{k}$  and  $M_{t - 1}^{i}$  can be represented as vectors, where elements are factor values for all possible combinations of RV states included in the factor. Let's denote elements of the vectors as  $f_{l}$  and  $m_{u}$  correspondingly, where  $l$  corresponds to a particular combination of  $k, h_{t}^{k}, h_{t - 1}^{i_{1}}, \ldots, h_{t - 1}^{i_{n_{l}}}$  state values and  $u$  indexes all neurons representing states of previous time step RVs.

Drawing inspiration from biological neural networks with active dendrites, we group a neuron's connections (dendrites) into segments. A segment acts as an independent computational unit that detects a particular input pattern (a context state) defined by its own receptive field. In our model, a segment links together factor value  $f_{l}$ , the computational graph shown in Figure 2, and the excitation  $E_{l}$  induced by the segment  $l$  to the cell it is attached to. The segment is active, i.e.,  $s_{l} = 1$  if all its presynaptic cells are active; otherwise,  $s_{l} = 0$ . Computationally, a segment transmits its factor value  $f_{l}$  to a cell it is attached to if the context matches the corresponding state combination.

We can now rewrite equation 4 as the following:

$$
p \left(h _ {t} ^ {k}\right) \propto \sum_ {l \in \operatorname {s e g} (j)} L _ {l} f _ {l} ^ {k}, \tag {5}
$$

where  $L_{l} = \prod_{u\in \mathrm{rec}(l)}m_{u}$  is segment's likelihood as long as messages are normalized,  $\operatorname {seg}(j)$  -- indexes of segments that are attached to cell  $j$ ,  $\operatorname {rec}(l)$  --indexes of presynaptic cells that constitute receptive field of a segment with index  $l$ .

Initially, all factor entries are zero, meaning cells have no segments. As learning proceeds, new non-zero connections grouped into segments are grown. In equation 5 we benefit from having sparse factor value vectors because its complexity depends linearly on the amount of non-zero components. And that's usually the case in our model due to one-step Monte-Carlo learning and specific form of emission factors  $F_{e}^{k}$ :

$$
F _ {e} ^ {k} \left(h _ {t} ^ {k}, o _ {t} ^ {k}\right) = \mathbb {I} \left[ h _ {t} ^ {k} \in \operatorname {c o l} \left(\varphi_ {t} ^ {k}\right) \right], \tag {6}
$$

where  $\mathbb{I}$  —indicator function,  $\operatorname{col}(\varphi_t^k)$  is a set of hidden states connected to the feature state  $\varphi_t^k$  that forms a column. The form of emission factor is inspired by presumably columnar structure of the neocortex and was shown to induce sparse transition matrix in HMM (George et al., 2021).

Segment likelihood  $L_{l}$ , resulting from the sum-product algorithm, is calculated as if presynaptic cells are independent. However, it's not usually the case for sparse factors. To take into account, approximately, their interdependence, we substitute the following equation for segment log-likelihood:

$$
\log L _ {l} = \log \sum_ {u \in \operatorname {r e c} (l)} w _ {u l} m _ {u} + \sum_ {u \in \operatorname {r e c} (l)} \left(1 - w _ {u l}\right) \log m _ {u} - \log n _ {l}, \tag {7}
$$

where  $w_{pl}$  — synapse efficiency or neuron specificity for segment, such that  $w_{ul} = p(s_l = 1|c_{t - 1}^u = 1)$ , and  $n_l$ -number of cells in segment's receptive field.

The idea that underlies the formula is to approximate between two extreme cases:

-  $p(s_{l} = 1|c_{t - 1}^{u} = 1) \to 1$  for all  $u$ , which means that all cells in the receptive field are dependent and are part of one cluster, i.e., they fire together. In that case, it should be  $p(s_{l}) = m_{u}$  for any  $u$ , but we also reduce prediction variance by averaging between different  $u$ .  
-  $p(s_{l} = 1|c_{t - 1}^{u} = 1) \to 0$  for all  $u$  means that presynaptic cells don't form a cluster. In that case, segment activation probability is just a product of the activation probability of each cell.

The resulting equation for belief propagation in DHTM is the following:

$$
p \left(h _ {t} ^ {k}\right) = p \left(c _ {t} ^ {j} = 1\right) = \operatorname * {s o f t m a x} _ {j \in \text {c e l l s} \left[ H _ {t} ^ {k} \right]} \left(\max  _ {l \in s e g (j)} \left(E _ {l}\right)\right), \tag {8}
$$

where  $E_{l} = \log f_{l} + \log L_{l}$ , cells  $[H_t^k ]$  — indexes of cells that represent states for  $H_{t}^{k}$  variable. Here, we also approximate logarithmic sum with max operation inspired by the neurophysiological model of segment aggregation by cell (Stuart & Spruston, 2015).

The next step after computing  $p(h_t^k)$  distribution parameters is to incorporate information about current observations  $p(h_t^k \mid o_t^k) \propto p(h_t^k) \mathbb{I}[h_t^k \in \mathrm{col}(o_t^k)]$ . After that, the learning step is performed. The step for closing the loop of our TM algorithm is to assign the posterior for the current step  $p(h_t^k \mid o_t^k)$  to  $M_{t-1}^i$ .

DHTM learns  $f_{l}$  and  $w_{ul}$  weights by Monte-Carlo Hebbian-like updates. First,  $h_{t-1}^{i}$  and  $h_{t}^{k}$  are sampled from their posterior distributions:  $p(h_{t-1}^{i} \mid o_{t-1}^{i}) \propto M_{t-1}^{i}$  and  $p(h_{t}^{k} \mid o_{t}^{k})$  correspondingly. Then  $f_{l}$  is updated according to the segment's  $s_{l}$  and its cell's  $c_{t}^{j}$  activity so that  $f_{l}$  is proportional to several coincidences  $s_{l} = c_{t}^{j} = 1$  during the recent past, i.e., cell and its segment are active at the same time step. It's similar to Baum-Welch's update rule (Baum et al., 1970) for the transition matrix in HMM, which, in effect, counts transitions from one state to another, but, in our case, the previous state (context) is represented by a group of RVs, not just one hidden RV.

Weights  $w_{ul}$  are also updated by the Hebbian rule to reflect the specificity of a presynaptic  $u$  for activating a segment  $l$ . That is, they are targeted to represent probability  $p(s_l = 1 \mid c_{t-1}^u = 1)$  that segment  $s_l$  is active, given cell  $u$  was active at the previous time-step. We could learn it by counting activation coincidences and mismatches. But in our algorithm it is approximated as exponential moving average of segment's  $s_l$  frequency activation, given  $c_{t-1}^u = 1$ :  $\Delta w_{ul} = \alpha \cdot \mathbb{I}[c_{t-1}^u = 1] \cdot (\mathbb{I}[s_l = 1] - w_{ul})$ , where  $\alpha \in [0,1)$  — learning rate.

# 3.3 AGENT ARCHITECTURE

We incorporate DHTM as a part of an RL agent. The agent consists of a DHTM memory model, an SF mapping from hidden space, and a feature reward function. The memory model aims to speed

Algorithm 1 General agent training procedure  
1: for episode=1..n do  
2: RESET_MEMORY()  
3: action  $\leftarrow$  null  
4: while (not terminal) and (steps  $<$  max_steps) do  
5: obs, reward  $\leftarrow$  STEP()  
6: features  $\leftarrow$  ENCODE(PREPROCESS(obs))  
7: OBSERVE(features, action)  
8: REINFORCE(reward, features)  
9: action  $\leftarrow$  SAMPLE_ACTION()  
10: ACT(action)  
11: end while  
12: end for

up SF learning by predicting cumulative future distributions of feature variables  $\Phi$  according to equation 17. As shown in equation 13, SF representations are learned to estimate state value. The  $r(\varphi_t^k)$  reward function is also learned during interaction with the environment and, combined with SF representations, is used to estimate the action value function.

The agent training procedure is outlined in Algorithm 1. For each episode, the memory state is reset to a fixed initial message with RESET MEMORY() and action variable is initialized with null value. An observation image returned by an environment (obs) is first preprocessed to get events, mimicking a simple event-based camera with a floating threshold determined from the average difference between the current and previous step image intensities. The resulting events are encoded to SDRs with a biologically inspired spatial pooling encoder described in Appendix A. In OBSERVE() routine, the memory learns to predict next feature states as described in Section 3 and SF learning happens according to equation 16. An agent learns associations to feature states and rewards in line 8:

$$
r _ {i} ^ {k} \leftarrow r _ {i} ^ {k} + \alpha \mathbb {I} \left[ \varphi_ {t} ^ {k} = i \right] \left(R _ {t} - r _ {i} ^ {k}\right) \tag {9}
$$

where  $\alpha$  is a learning rate,  $R_{t}$  —a reward for the current time step.

We include actions into the model by forcing some of the hidden variables  $H_{t}^{k}$  to represent actions. That is, we assume that information about action is included in the hidden state of the model. For example, if we have 4 actions, we set 4 states for one of the hidden variables and set its state from observation of the action. We form on-policy SFs, i.e. relying on policy iteration theorem.

An agent has a softmax policy over predicted values:  $\pi (a_{t}\mid o_{0:t}) = \mathrm{softmax}(V[p(h_{t + 1}\mid o_{0:t},a_{t})])$ . We use the model to predict the hidden state distribution for every action in the next timestep  $t + 1$  and then estimate its value according to equation 13.

# 4 EXPERIMENTS

We test our model in a reinforcement learning task in a pinball-like 2D environment, where successor features are easy to interpret, and in a more challenging AnimalAI 3D environment. This section shows how different memory models affect SF learning and an RL agent's adaptability. In our work, we compare the proposed DHTM model with LSTM (Hochreiter & Schmidhuber, 1997), RWKV (Peng et al., 2023), and CSCG (George et al., 2021) (see Appendix E for the details).

# 4.1 PINBALL

The first, classic maze, test is designed in the Pinball environment (see Appendix F for details) to qualitatively assess SFs formed by different TMs for random policy (see Fig. 3). Ball is controlled by the agent able to apply a momentum in four opposite directions. The ball and terminal state are separated by a wall with a door on the right. Each episode is maximum of 30 steps. Memories are tested in two regimes: 5-step planning (i.e. using equation 17 only) and prediction only (equation 18). As can be seen from the heatmaps, only DHTM yields adequate value functions. However, as can be seen from the learning curves, surprise of DHTM is higher than of the other memories. LSTM's learning curve is much flatter than of the others. Five-step DHTM planning gives more

![](images/7673f38caa5a4b26d1ab6fb6de7d099257c2281f6a4573fdaadded6484db55f6.jpg)  
Figure 3: Results of 2D maze random policy experiment in the Pinball environment. Surprise learning curves for DHTM, LSTM, RWKV and CSCG. Heatmaps represent value functions for DHTM and LSTM.

![](images/35f77d9c174a0d8cbe46df2d01bdc5366e3ce7b634f2dd9a0376430af9f2c764.jpg)  
Figure 4: Surprise comparison for various memory models including DHTM (ours), LSTM, RWKV, and Factorial version of CSCG (fchmm). The SFs generated by normalized five-step prediction models are used to calculate surprise for three future time steps.

abrupt value function in comparison to prediction regime, as it usually requires more than five steps to reach the goal. Heatmaps for other baselines can be found in Appendix G.

The second test is to show how TM can enhance adaptation in changing environments. For that experiment, we use two configurations of the Pinball environment shown in Figure 7-A. We narrow the action space to three momentum vectors: vertical, 30 degrees left and 30 degrees right from the vertical axis. Each time step, the agent gets a small negative reward and a large positive reward if the ball enters the force field in the centre. The episode finishes when the ball enters the rewarding force field or the maximum number of steps is reached. Each trial is run for 500 episodes, each a maximum of 15 steps long, and we average the results over three trials for each parameter set and memory model.

We test the accuracy of five-step SF representations by measuring their pseudo-surprise, which is surprise computed for observed states on different time steps after SF was predicted with respect to normalized SF (more details in Appendix D). In all experiments, the encoder outputs five variables  $\Phi$  with 50 states each. As can be seen from Figure 4, SRs produced by our memory model (dhtm) give lower surprise than SRs of LSTM (lstm) and RWKV (rwkv), and is on par with SRs produced by Factorial version of CSCG (fchmm), which is just several CSCGs trained in parallel to enable handling of multiple variables outputted by encoder.

Then, we test how the number of prediction steps affects the agent's adaptability in the Pinball environment. In the first 500 episodes, the agent is trained to reach the target in the centre, as shown in Figure 7-A, then the target is blocked by a random force that applies force in perpendicular direction to the ball's movement. The results show that an agent that uses five prediction steps during n-step TD learning of SF faster adapts to the changes in the environment in comparison to 1-step TD learning for SF, as seen from Figure 5-A.

![](images/2a7ca14d18ef7258c41e211c97850ae77e5359c5e1086d51b33dd10c36be81cb.jpg)  
Figure 5: A. Comparison of agent's adaptability during changes in the environment with different prediction steps during n-step TD learning of SF. At the 500th episode, the environment changes its configuration, shown in Figure 7-A. B. AnimalAI changing food position experiment. Left picture is DHTM reward curves each averaged over five trials for two cases: SF formed by 7-step planning using DHTM and SF is predicted using TD learned weights and DHTM inferred belief states. At the 300th episode, the food is moved to the opposite corridor (see Fig. 7-C).

# 4.2 ANIMALAI

We designed an experiment in AnimalAI environment shown on Figure 7-C. There are two corridors, one of which contains food (yellow circle). The agent makes a decision at the start of the trial, having three options: go to the left corridor, go to the right and stay turning. After the decision is made, the agent follows a fixed strategy, which brings it either to the right corridor or to the left, and it observes its movement and actions. An episode ends when strategy is executed. Each time step, agent gets small negative reward and big positive reward only if reaches food. After 300 episodes, food is placed to the other corridor. Reward curves averaged over five trials for each setup are presented in Figure 5-B. There are two cases on the plot: SF is formed by prediction (equation 18) or planned (equation 17). The results for DHTM show that planning allows much faster adaptation to the change of the rewarding food position.

# 5 CONCLUSION

In this paper, we introduce a novel probabilistic Factorial-HMM-like algorithm DHTM for learning an observation sequence model in stochastic environments that uses local Hebbian-like learning rules, which renders it apt for running on neuromorphic processors. DHTM is scalable to multiple feature variables as it employs sparse distributed representations and sparse factor function implementation using segments, which biologically plausible multicomponent neural models inspire. In contrast to methods that use Monte-Carlo trajectory sampling for future states probability estimation, our method is able to perform belief propagation, so each prediction step adds constant amount of computations. We show that our memory model can quickly learn the observation sequences representation and the transition dynamics. The DHTM produces more accurate n-step Successor Features than LSTM and RWKV, which speeds up n-step TD learning of the SF in Reinforced Learning tasks with the changing environment.

One of the limitations of the DHTM is that its temporal context is random, as it is formed on the fly. That is, the mechanism of context formation doesn't allow generalizations. That is why we are forced to use feature space inferred from observations for value function decomposition, to soften this problem. Nevertheless, we believe that forming Successor Features combined with two level hierarchy of DHTM layers may provide the next step to circumvent this limitation, which directs of our further research. Another limitation is the maximum number of variables per factor. The amount of segments in use grows exponentially with the number of variables per factor, especially in noisy environments. Solving this issue would require to modify segment excitation or growth algorithms.

# REFERENCES

André Barreto, Will Dabney, Rémi Munos, Jonathan J Hunt, Tom Schaul, Hado P van Hasselt, and David Silver. Successor features for transfer in reinforcement learning. Advances in neural information processing systems, 30, 2017.

Leonard E Baum, Ted Petrie, George Soules, and Norman Weiss. A maximization technique occurring in the statistical analysis of probabilistic functions of markov chains. The annals of mathematical statistics, 41(1):164-171, 1970.  
Edward Beeching, Jilles Debangoye, Olivier Simonin, and Christian Wolf. Godot reinforcement learning agents. arXiv preprint arXiv:2112.03636, 2021.  
PENG Bo. Blinkdl/rwkv-lm: 0.01, August 2021. URL https://doi.org/10.5281/ zenodo.5196577.  
Patricia Smith Churchland and Terrence Joseph Sejnowski. The computational brain. MIT press, 1992.  
Matthew Crosby, Benjamin Beyret, Murray Shanahan, José Hernández-Orallo, Lucy Cheke, and Marta Halina. The animal-ai testbed and competition. In Hugo Jair Escalante and Raia Hadsell (eds.), Proceedings of the NeurIPS 2019 Competition and Demonstration Track, volume 123 of Proceedings of Machine Learning Research, pp. 164–176. PMLR, 08–14 Dec 2020.  
Yuwei Cui, Subutai Ahmad, and Jeff Hawkins. The htm spatial pooler—a neocortical algorithm for online sparse distributed coding. Frontiers in Computational Neuroscience, 11:111, 2017. ISSN 1662-5188. doi: 10.3389/fncom.2017.00111. URL https://www.frontiersin.org/article/10.3389/fncom.2017.00111.  
Peter Dayan. Improving generalization for temporal difference learning: The successor representation. Neural computation, 5(4):613-624, 1993.  
Damir Dobric, Andreas Pech, Bogdan Ghita, and Thomas Wennekers. On the importance of the newborn stage when learning patterns with the spatial pooler. SN Computer Science, 3(2):179, 2022.  
Yogesh K Dwivedi, Nir Kshetri, Laurie Hughes, Emma Louise Slade, Anand Jeyaraj, Arpan Kumar Kar, Abdullah M Baabdullah, Alex Koohang, Vishnupriya Raghavan, Manju Ahuja, et al. "so what if chatgpt wrote it?" multidisciplinary perspectives on opportunities, challenges and implications of generative conversational ai for research, practice and policy. International Journal of Information Management, 71:102642, 2023.  
Gökcen Eraslan, Žiga Avsec, Julien Gagneur, and Fabian J Theis. Deep learning: new computational modelling techniques for genomics. Nature Reviews Genetics, 20(7):389-403, 2019.  
Karl Friston, Thomas FitzGerald, Francesco Rigoli, Philipp Schwartenbeck, Giovanni Pezzulo, et al. Active inference and learning. Neuroscience & Biobehavioral Reviews, 68:862-879, 2016.  
Karl J Friston, Richard Rosch, Thomas Parr, Cathy Price, and Howard Bowman. Deep temporal models and active inference. *Neuroscience & Biobehavioral Reviews*, 90:486-501, 2018.  
Dileep George and Jeff Hawkins. Towards a mathematical theory of cortical micro-circuits. PLoS computational biology, 5(10):e1000532, 2009.  
Dileep George, Rajeev V. Rikhye, Nishad Gothoskar, J. Swaroop Guntupalli, Antoine Dedieu, and Miguel Lázaro-Gredilla. Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps. Nature Communications, 12(11):2392, Apr 2021. ISSN 2041-1723. doi: 10.1038/s41467-021-22559-5.  
Z. Ghahramani and M.I. Jordan. Factorial Hidden Markov Models. Machine Learning, 29(2-3): 245-273, 1997. ISSN 0885-6125. doi: 10.1023/a:1007425814087.  
Zoubin Ghahramani and Michael Jordan. Factorial hidden markov models. Advances in neural information processing systems, 8, 1995.  
David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.  
Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

GM Harshvardhan, Mahendra Kumar Gourisaria, Manjusha Pandey, and Siddharth Swarup Rautaray. A comprehensive survey and analysis of generative models in machine learning. Computer Science Review, 38:100285, 2020.  
Jeff Hawkins and Subutai Ahmad. Why neurons have thousands of synapses, a theory of sequence memory in neocortex. Frontiers in Neural Circuits, 10, March 2016. ISSN 1662-5110. doi: 10.3389/fncir.2016.00023. URL http://journal.frontiersin.org/Article/10.3389/fncir.2016.00023/abstract.  
Donald Olding Hebb. The organization of behavior: A neuropsychological theory. Psychology press, 2005.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9:1735-80, 12 1997. doi: 10.1162/neco.1997.9.8.1735.  
Mehdi Jafarnia Jahromi, Rahul Jain, and Ashutosh Nayyar. Online learning for unknown partially observable mdps. In International Conference on Artificial Intelligence and Statistics, pp. 1712-1732. PMLR, 2022.  
Shulei Ji, Jing Luo, and Xinyu Yang. A comprehensive survey on deep music generation: Multi-level representations, algorithms, evaluations, and future directions. arXiv preprint arXiv:2011.06801, 2020.  
F.R. Kschischang, B.J. Frey, and H.-A. Loeliger. Factor graphs and the sum-product algorithm. IEEE Transactions on Information Theory, 47(2):498-519, 2001. doi: 10.1109/18.910572.  
Petr Kuderov, Evgenii Dzhivelikian, and Aleksandr I Panov. Stabilize sequential data representation via attraction module. In International Conference on Brain Informatics, pp. 83-95. Springer, 2023.  
Timothy P. Lillicrap, Adam Santoro, Luke Harris, Colin J. Akerman, and Geoffrey Hinton. Backpropagation and the brain. Nature Reviews Neuroscience, 21(6):335-346, Jun 2020. ISSN 1471-003X, 1471-0048. doi: 10.1038/s41583-020-0277-3.  
Zachary C Lipton, John Berkowitz, and Charles Elkan. A critical review of recurrent neural networks for sequence learning. arXiv preprint arXiv:1506.00019, 2015.  
Michael London and Michael Häusser. Dendritic computation. Annu. Rev. Neurosci., 28:503-532, 2005.  
Christoph Mathys, Jean Daunizeau, Karl J Friston, and Klaas E Stephan. A bayesian foundation for individual learning under uncertainty. Frontiers in human neuroscience, 5:39, 2011.  
Bonan Min, Hayley Ross, Elior Sulem, Amir Pouran Ben Veyseh, Thien Huu Nguyen, Oscar Sainz, Eneko Agirre, Ilana Heintz, and Dan Roth. Recent advances in natural language processing via large pre-trained language models: A survey. ACM Computing Surveys, 2021.  
James Mnatzaganian, Ernest Fokoué, and Dhireesha Kudithipudi. A mathematical formalization of hierarchical temporal memory's spatial pooler. Frontiers in Robotics and AI, 3, 2017. ISSN 2296-9144. URL https://www.frontiersin.org/articles/10.3389/frobt.2016.00081.  
Thomas M Moerland, Joost Broekens, Aske Plaat, Catholijn M Jonker, et al. Model-based reinforcement learning: A survey. Foundations and Trends® in Machine Learning, 16(1):1-118, 2023.  
V. Mountcastle. The columnar organization of the neocortex. *Brain*, 120(4):701-722, April 1997.  
ISSN 14602156. doi: 10.1093/brain/120.4.701. URL https://academic.oup.com/brain/article-lookup/doi/10.1093/brain/120.4.701.  
Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. Resurrecting recurrent neural networks for long sequences. arXiv preprint arXiv:2303.06349, 2023.

Matthias Oster, Rodney Douglas, and Shih-Chii Liu. Computation with spikes in a winner-take-all network. Neural Computation, 21(9):2437-2465, 09 2009. doi: 10.1162/neco.2009.07-08-829.  
Randall C. O'Reilly, Jacob L. Russin, Maryam Zolfaghar, and John Rohrlich. Deep predictive learning in neocortex and pulvinar. Journal of Cognitive Neuroscience, 33(6):1158-1196, May 2021. ISSN 0898-929X. doi: 10.1162/jocn_a_01708.  
Thomas Parr and Karl J Friston. Working memory, attention, and salience in active inference. Scientific reports, 7(1):14678, 2017.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32, pp. 8024-8035. Curran Associates, Inc., 2019. URL http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf.  
Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023.  
Rodrigo Perin, Thomas K Berger, and Henry Markram. A synaptic organizing principle for cortical neuronal groups. Proceedings of the National Academy of Sciences, 108(13):5419-5424, 2011.  
Pascal Poupart. Exploiting structure to efficiently solve large scale partially observable Markov decision processes. Citeseer, 2005.  
Achille Salaun, Yohan Petetin, and François Desbouvries. Comparing the modeling powers of rnn and hmm. In 2019 18th IEEE international conference on machine learning and applications (icmla), pp. 1496-1499. IEEE, 2019.  
Jochen F. Staiger and Carl C. H. Petersen. Neuronal circuits in barrel cortex for whisker sensory perception. Physiological Reviews, 101(1):353-415, 2021. doi: 10.1152/physrev.00019.2019. URL https://doi.org/10.1152/physrev.00019.2019. PMID: 32816652.  
Greg J. Stuart and Nelson Spruston. Dendritic integration: 60 years of progress. Nature Neuroscience, 18(12):1713-1721, Dec 2015. ISSN 1546-1726. doi: 10.1038/nn.4157. URL https://doi.org/10.1038/nn.4157.  
Jingyu Zhao, Feiqing Huang, Jia Lv, Yanjie Duan, Zhen Qin, Guodong Li, and Guangjian Tian. Do rnn and lstm have long memory? In International Conference on Machine Learning, pp. 11365-11375. PMLR, 2020.
