# RECURRENT INDEPENDENT MECHANISMS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We explore the hypothesis that learning modular structures which reflect the dynamics of the environment can lead to better generalization and robustness to changes that only affect a few of the underlying causes. We propose Recurrent Independent Mechanisms (RIMs), a new recurrent architecture in which multiple groups of recurrent cells operate with nearly independent transition dynamics, communicate only sparingly through the bottleneck of attention, and compete with each other so they are updated only at time steps where they are most relevant. We show that this leads to specialization amongst the RIMs, which in turn allows for remarkably improved generalization on tasks where some factors of variation differ systematically between training and evaluation.

# 1 INDEPENDENT MECHANISMS

Physical processes in the world often have a modular structure which human cognition appears to exploit, with complexity emerging through combinations of simpler subsystems. Machine learning seeks to uncover and use regularities in the physical world. Although these regularities manifest themselves as statistical dependencies, they are ultimately due to dynamic processes governed by causal physical phenomena. These processes are mostly evolving independently and only interact sparsely. For instance, we can model the motion of two balls as separate independent mechanisms even though they are both gravitationally coupled to Earth as well as (weakly) to each other. Only occasionally will they strongly interact via collisions.

The notion of independent or autonomous mechanisms has been influential in the field of causal inference. A complex generative model, temporal or not, can be thought of as the composition of independent mechanisms or "causal" modules. In the causality community, this is often considered a prerequisite for being able to perform localized interventions upon variables determined by such models (Pearl, 2009). It has been argued that the individual modules tend to remain robust or invariant even as other modules change, e.g., in the case of distribution shift (Scholkopf et al., 2012; Peters et al., 2017). This independence is not between the random variables being processed but between the description or parametrization of the mechanisms: learning about one should not tell us anything about another, and adapting one should not require also adapting another. One may hypothesize that if a brain is able to solve multiple problems beyond a single i.i.d. (independent and identically distributed) task, they may exploit the existence of this kind of structure by learning independent mechanisms that can flexibly be reused, composed and re-purposed.

In the dynamic setting, we think of an overall system being assayed as composed of a number of fairly independent subsystems that evolve over time, responding to forces and interventions. An agent needs not devote equal attention to all subsystems at all times: only those aspects that significantly interact need to be considered jointly when deciding or planning (Bengio, 2017). Such sparse interactions can reduce the difficulty of learning since few interactions need to be considered at a time, reducing unnecessary interference when a subsystem is adapted. Models learned this way may better capture the compositional generative (or causal) structure of the world, and thus better generalize across tasks where a (small) subset of mechanisms change while most of them remain invariant (Simon, 1991; Peters et al., 2017; Parascandolo et al., 2018). The central question motivating our work is how a gradient-based deep learning approach can discover a representation of high-level variables which favour forming independent but sparsely interacting recurrent mechanisms in order to benefit from the modularity and independent mechanisms assumption.

![](images/e90b23bc49c8666b4dc05af7e1bee34c079fee3f24f877ee8d82af22064cad1e.jpg)  
Figure 1: Illustration of Recurrent Independent Mechanisms (RIMs). A single step under the proposed model occurs in four stages (left figure shows two steps). In the first stage, individual RIMs produce a query which is used to read from the current input. In the second stage, an attention based competition mechanism is used to select which RIMs to activate (right figure) based on encoded visual input (blue RIMs are active, based on an attention score, white RIMs remain inactive). In the third stage, individual activated RIMs follow their own default transition dynamics while non-activated RIMs remain unchanged. In the fourth stage, the RIMs sparsely communicate information between themselves, also using key-value attention.

![](images/0530e45be25599f1c00f4c0586119ce18efb97855f5368b494f12dcf02ba83fd.jpg)

Why do Models Succeed or Fail in Capturing Independent Mechanisms? While universal approximation theorems apply in the limit of large i.i.d. data sets, we are interested in the question of whether models can learn independent mechanisms from finite data in possibly changing environments, and how to implement suitable inductive biases. As the simplest case, we can consider training an RNN consisting of  $k$  completely independent mechanisms which operate on distinct time steps. How difficult would it be for an RNN (whether vanilla or LSTM or GRU) to correctly model that the true distribution has completely independent processes? For the hidden states to truly compartmentalize these different processes, a fraction  $\frac{k - 1}{k}$  of the connections would need to be set to exactly zero weight. This fraction approaches  $100\%$  as  $k$  approaches infinity. When sample complexity or out-of-distribution generalization matter, we argue that having an inductive bias which favors this form of modularity and dynamic recombination could be greatly advantageous, compared to static fully connected monolithic architectures.

# 2 RIMS WITH SPARSE INTERACTIONS

Our approach to modelling a dynamical system of interest divides the overall model into  $k$  small subsystems (or modules), each of which is recurrent in order to be able to capture the dynamics in the observed sequences. We refer to these subsystems as Recurrent Independent Mechanisms (RIMs), where each RIM has distinct functions that are learned automatically from data<sup>1</sup>. We refer to RIM  $k$  at time step  $t$  as having vector-valued state  $h_{t,k}$ , where  $t = 1, \dots, T$ . Each RIM has parameters  $\theta_k$ , which are shared across all time steps.

At a high level (see Fig. 1), we want each RIM to have its own independent dynamics operating by default, and occasionally to interact with other relevant RIMs and selected elements of the encoded input. The total number of parameters can be kept small since RIMs can specialize on simple sub-problems, similar to Parascandolo et al. (2018), and operate on few key/value variables at a time selected using an attention mechanism, as suggested by the inductive bias from Bengio (2017). This specialization and modularization not only has computational and statistical advantages (Baum & Haussler, 1989), but also prevents individual RIMs from dominating the computation and thus facilitates factorizing the computation into easy to recombine but simpler elements. We expect this to lead to more robust systems than training one big homogeneous system (Schmidhuber, 2018). Moreover, modularity and the independent mechanisms hypothesis (Peters et al., 2017; Bengio et al., 2019) also has the desirable implication that a RIM should maintain its own independent functionality even as other RIMs are changed. A more detailed account of the desiderata for the model is given in Appendix A.

# 2.1 KEY-VALUE ATTENTION TO PROCESS SETS OF NAMED INTERCHANGEABLE VARIABLES

Each RIM should be activated and updated when the input is relevant to it. We thus utilize competition to allocate representational and computational resources, using an attention mechanism which selects

and then activates only a subset of the RIMs for each time step. As argued by Parascandolo et al. (2018), this tends to produce independence among learned mechanisms, provided the training data has been generated by a set of independent physical mechanisms. In contrast to Parascandolo et al. (2018), we use an attention mechanism for this purpose. The introduction of content-based soft-attention mechanisms (Bahdanau et al., 2014) has opened the door to neural networks which operate on sets of typed interchangeable objects. This idea has been remarkably successful and widely applied to most recent Transformer-style multi-head dot product self attention models (Vaswani et al., 2017; Santoro et al., 2018), achieving new state-of-the-art results in many tasks. Soft-attention uses the product of a query (or read key) represented as a matrix  $Q$  of dimensionality  $N_{r} \times d$ , with  $d$  the dimension of each key, with a set of  $N_{o}$  objects each associated with a key (or write-key) as a row in matrix  $K^{T}$  ( $N_{o} \times d$ ), and after normalization with a softmax yields outputs in the convex hull of the values (or write-values)  $V_{i}$  (row  $i$  of matrix  $V$ ). The result is

$$
\mathrm {A t t e n t i o n} (Q, K, V) = \mathrm {s o f t m a x} \left(\frac {Q K ^ {T}}{\sqrt {d}}\right) V,
$$

where the softmax is applied to each row of its argument matrix, yielding a set of convex weights. As a result, one obtains a convex combination of the values in the rows of  $V$ . If the attention is focused on one element for a particular row (i.e., the softmax is saturated), this simply selects one of the objects and copies its value to row  $j$  of the result. Note that the  $d$  dimensions in the key can be split into heads which then have their own attention matrix and write values computed separately.

When the inputs and outputs of each RIM are a set of objects or entities (each associated with a key and value vector), the RIM processing becomes a generic object-processing machine which can operate on subsymbolic "variables" in a sense analogous to variables in a programming language: as interchangeable arguments of functions, albeit with a distributed representation both for they name or type and for their value. Because each object has a key embedding (which one can understand both as a name and as a type), the same RIM processing can be applied to any variable which fits an expected "distributed type" (specified by a query vector). Each attention head then corresponds to a typed argument of the function computed by the RIM. When the key of an object matches the query of head  $k$ , it can be used as the  $k$ -th input vector argument for the RIM. Whereas in regular neural networks (without attention) neurons operate on fixed variables (the neurons which are feeding them from the previous layer), the key-value attention mechanisms make it possible to select on the fly which variable instance (i.e. which entity or object) is going to be used as input for each of the arguments of the RIM dynamics, with a different set of query embeddings for each RIM head. These inputs can come from the external input or from the output of other RIMs. So, if the individual RIMs can represent these functions with typed arguments, then they can bind to whatever input is currently available and best suited according to its attention score: the "input attention" mechanism would look at the candidate input object's key and evaluate if its "type" matches with what this RIM expects (specified with the corresponding query).

# 2.2 SELECTIVE ACTIVATION OF RIMS AS A FORM OF TOP-DOWN MODULATION

The proposed model learns to dynamically select those RIMs for which the current input is relevant. RIMs are triggered as a result of interaction between the current state of the RIM and input information coming from the environment. At each step, we select the top- $k_{A}$  (out of  $k_{T}$ ) RIMs in terms of their attention score for the real input. Intuitively, the RIMs must compete on each step to read from the input, and only the RIMs that win this competition will be able to read from the input and have their state updated. In our use of key-value attention, the queries come from the RIMs, while the keys and values come from the current input. This differs from the mechanics of Vaswani et al. (2017); Santoro et al. (2018), with the modification that the parameters of the attention mechanism itself are separate for each RIM rather than produced on the input side as in Transformers. The input attention for a particular RIM is described as follows.

![](images/537c060998afb7e0b2b5069bda49906d34343a997ae547f7c641c9c818d56b90.jpg)  
Figure 2: Individual RIMs produce a query which is used to read from the current input. An attention based competition mechanism is used to select which RIMs to activate, based on the softmax scores. Here, 2 RIMs (shown in blue) are activated.

The input  $x_{t}$  at time  $t$  can be seen as an output of the encoder parameterized by a neural network (for ex. CNN in case of visual observations) i.e.,  $X = C N N(x_{t})$ . Then, linear transformations are used to construct keys  $(K = X W^{e})$ , values  $(V = X W^{v})$ , and queries  $(Q = h_{t}W_{k}^{q}$ , one per RIM attention head).  $W^{v}$  is a simple matrix mapping an input element to the corresponding value vector for the weighted attention and  $W^{e}$  is similarly a weight matrix which maps the input to the keys.  $W_{k}^{q}$  is a per-RIM weight matrix which maps from the RIM's hidden state to its queries.

$$
A _ {k} ^ {(i n)} = \operatorname {s o f t m a x} \left(\frac {h _ {t} W _ {k} ^ {q} \left(X W ^ {e}\right) ^ {T}}{\sqrt {d _ {e}}}\right) X W ^ {v}, \text {w h e r e} \quad \theta_ {k} ^ {(i n)} = \left(W _ {k} ^ {q}, W ^ {e}, W ^ {v}\right). \tag {1}
$$

Based on the softmax values in (1), we select the top  $k_{A}$  RIMs (out of the total  $k_{T}$  RIMs) (see. Fig 2) to be activated for each step, which have the highest attention on the input and call this set  $S_{t}$ . Here, the selection of top  $k_{A}$  RIMs is differentiable via the use of a sparse softmax, also successfully used in (Ke et al., 2018). Since the query for a particular RIM is a function of the hidden state of that RIM, this enables individual RIM to attend only to the part of the input that is currently relevant for that particular RIM, thus enabling selective attention based on a top-down attention process (see. Fig 1).

# 2.3 INDEPENDENT RIM DYNAMICS

Now, consider the default transition dynamics which we apply for each RIM independently and during which no information passes between RIMs. We use  $\tilde{h}$  for the hidden state after the independent dynamics are applied. The hidden states of RIMs which are not activated (we refer to the activated set as  $S_{t}$ ) remain unchanged, acting like untouched memory elements, i.e.,  $h_{t+1,k} = h_{t,k} \forall k \notin S_{t}$ . Note that the gradient still flows through a RIM on a step where it is not activated. For the RIMs that are activated, we run a per-RIM independent transition dynamics. The form of this is somewhat flexible, but we opted to use either a GRU (Chung et al., 2015) or an LSTM (Hochreiter & Schmidhuber, 1997). We generically refer to these independent transition dynamics as  $D_{k}$ , and we emphasize that each RIM has its own separate parameters. Aside from being RIM-specific, the internal operation of the LSTM and GRU remains unchanged, and active RIMs are updated by

$$
\tilde {h} _ {t, k} = D _ {k} (h _ {t, k}) = L S T M (h _ {t, k}, A _ {k} ^ {(i n)}; \theta_ {k} ^ {(D)}) \quad \forall k \in \mathcal {S} _ {t}
$$

as a function of the attention mechanism  $A_{k}^{(in)}$  applied on the current input, described in the previous sub-section.

# 2.4 COMMUNICATION BETWEEN RIMS

Although the RIMs operate independently by default, the attention mechanism allows sharing of information among the RIMs. Specifically, we allow the activated RIMs to read from all other RIMs (activated or not). The intuition behind this is that non-activated RIMs are not related to the current input, so their value needs not change. However they may still store contextual information relevant for activated RIMs later on. For this communication between RIMs, we use a residual connection as in Santoro et al. (2018) to prevent vanishing or exploding gradients over long sequences. Using parameters  $\theta_{k}^{(c)} = (\tilde{W}_{k}^{q},\tilde{W}_{k}^{e},\tilde{W}_{k}^{v})$  , we employ

$$
Q _ {t, k} = \tilde {W} _ {k} ^ {q} \tilde {h} _ {t, k}, \forall k \in \mathcal {S} _ {t} \qquad K _ {t, k} = \tilde {W} _ {k} ^ {e} \tilde {h} _ {t, k}, \forall k \qquad V _ {t, k} = \tilde {W} _ {k} ^ {v} \tilde {h} _ {t, k}, \forall k
$$

$$
h _ {t + 1, k} = \operatorname {s o f t m a x} \left(\frac {Q _ {t , k} \left(K _ {t , :}\right) ^ {T}}{\sqrt {d _ {e}}}\right) V _ {t,:} + \tilde {h} _ {t, k} \forall k \in \mathcal {S} _ {t}.
$$

Number of Parameters. RIMs can be used as a drop-in replacement for an LSTM/GRU layer. There is a subtlety that must be considered for successful integration. If the total size of the hidden state is kept the same, integrating RIMs drastically reduces the total number of recurrent parameters in the model (because of having a block-sparse structure) but RIMs also adds new parameters to the model through the addition of the attention mechanisms although these are rather in small number.

# 3 RELATED WORK

Neural Turing Machine (NTM) and Relational Memory Core (RMC): the NTM (Graves et al., 2014a) updates independent memory cells using an attention mechanism to perform targeted read

and write operations. RIMs share a key idea with NTMs: that input information should only impact a sparse subset of the memory by default, while keeping most of the memory unaltered. RMC (Santoro et al., 2018) uses a multi-head attention mechanism to share information between multiple memory elements. We encourage the RIMs to remain separate as much as possible, whereas Santoro et al. (2018) allow information between elements to flow on each step in an unconstrained way. Instead, each RIM has its own default dynamics, while in RMC, all the processes interact with each other.

Separate Recurrent Models: EntNet (Henaff et al., 2016) and IndRNN (Li et al., 2018) can be viewed as a set of separate recurrent models. In IndRNN, each recurrent unit has completely independent dynamics, whereas EntNet uses an independent gate for writing to each memory slot. RIMs use different recurrent models (with separate parameters), but we allow the RIMs to communicate with each other sparingly using an attention mechanism.

Modularity and Neural Networks: A network can be composed of several modules, each meant to perform a distinct function, and hence can be seen as a combination of experts (Jacobs et al., 1991; Bottou & Gallinari, 1991; Ronco et al., 1997; Reed & De Freitas, 2015; Andreas et al., 2016; Parascandolo et al., 2018; Rosenbaum et al., 2017; Fernando et al., 2017; Shazeer et al., 2017; Kirsch et al., 2018; Rosenbaum et al., 2019) routing information through a gated activation of modules. These works generally assume that only a single expert is active at a particular time step. In the proposed method, multiple RIMs can be active, interact and share information.

Computation on demand: There are various architectures (El Hhi & Bengio, 1996; Koutnik et al., 2014; Chung et al., 2016; Neil et al., 2016; Jernite et al., 2016; Krueger et al., 2016) where parts of the RNN's hidden state are kept dormant at times. The major differences to our architecture are that (a) we modularize the dynamics of recurrent cells (using RIMs), and (b) we also control the inputs of each module (using transformer style attention), while many previous gating methods did not control the inputs of each module, but only whether they should be executed or not.

# 4 EXPERIMENTS

The main goal of our experiments is to show that the use of RIMs improves generalization across changing environments and/or in modular tasks, and to explore how it does so. Our goal is not to outperform highly optimized baselines; rather, we want to show the versatility of our approach by applying it to a range of diverse tasks, focusing on tasks that involve a changing environment. We organize our results by the capabilities they illustrate: we address generalization based on temporal patterns, based on objects, and finally consider settings where both of these occur together.

# 4.1 RIMS IMPROVE GENERALIZATION BY SPECIALIZING OVER TEMPORAL PATTERNS

We first show that when RIMs are presented with sequences containing distinct and generally independent temporal patterns, they are able to specialize so that different RIMs are activated on different patterns. RIMs generalize well when we modify a subset of the patterns (especially those unrelated to the class label) while most recurrent models fail to generalize well to these variations.

![](images/ace2392bca1f9b1c240a58a8bb2bdf1ed5415abb678bafb8bb01fd0f4941e95e.jpg)  
Figure 3: Visualizing Activation Patterns. For the copying task, one can see that the RIM activation pattern is distinct during the dormant part of the sequence in the middle (activated RIMs black, non-activated white). X-axis=time, Y-axis=RIMs activation bit.

Copying Task: First we turn our attention to the task of receiving a short sequence of characters, then receiving blank inputs for a large number of steps, and then being asked to reproduce the original sequence. We can think of this as consisting of two temporal patterns which are independent: one where the sequence is received and another "dormant" pattern where no input is provided. As an example of out-of-distribution generalization, we find that using RIMs, we can extend the length of this dormant phase from 50 during training to 200 during testing and retain perfect performance (Table 1), whereas baseline methods including LSTM, NTM, and RMC substantially degrade. In addition, we find that this result is robust to the number of RIMs used as well as to the number of

Table 1: Performance on the copying task (left) and Sequential MNIST resolution generalization (right). While all of the methods are able to learn to copy for the length seen during training, the RIMs model generalizes to sequences longer than those seen during training whereas the LSTM, RMC, and NTM degrade much more. On sequential MNIST, both the proposed and the Baseline models were trained on  $14 \times 14$  resolution but evaluated at different resolutions (averaged over 3 trials).  

<table><tr><td rowspan="2">Copying
kT</td><td rowspan="2">kA</td><td rowspan="2">hsize</td><td rowspan="2">Train(50) 
CE</td><td rowspan="2">Test(200) 
CE</td><td colspan="3">Sequential MNIST</td><td rowspan="2">16 x 16 
Accuracy</td><td rowspan="2">19 x 19 
Accuracy</td><td rowspan="2">24 x 24 
Accuracy</td></tr><tr><td>kT</td><td>kA</td><td>hsize</td></tr><tr><td rowspan="4">RIMs</td><td>6</td><td>4</td><td>600</td><td>0.00</td><td>0.00</td><td>6</td><td>6</td><td>600</td><td>85.5</td><td>30.9</td></tr><tr><td>6</td><td>3</td><td>600</td><td>0.00</td><td>0.00</td><td>6</td><td>5</td><td>600</td><td>88.3</td><td>22.1</td></tr><tr><td>6</td><td>2</td><td>600</td><td>0.00</td><td>0.00</td><td>6</td><td>4</td><td>600</td><td>90.0</td><td>38.1</td></tr><tr><td>5</td><td>2</td><td>500</td><td>0.00</td><td>0.00</td><td>-</td><td>-</td><td>300</td><td>86.8</td><td>25.2</td></tr><tr><td rowspan="2">LSTM</td><td>-</td><td>-</td><td>300</td><td>0.00</td><td>4.32</td><td>-</td><td>-</td><td>600</td><td>84.5</td><td>21.9</td></tr><tr><td>-</td><td>-</td><td>600</td><td>0.00</td><td>3.56</td><td>-</td><td>-</td><td>-</td><td>89.2</td><td>23.5</td></tr><tr><td>NTM</td><td>-</td><td>-</td><td>-</td><td>0.00</td><td>2.54</td><td>-</td><td>-</td><td>-</td><td>89.58</td><td>27.75</td></tr><tr><td>RMC</td><td>-</td><td>-</td><td>-</td><td>0.00</td><td>0.13</td><td>-</td><td>-</td><td>-</td><td>87.2</td><td>19.8</td></tr><tr><td>Transformers</td><td>-</td><td>-</td><td>-</td><td>0.00</td><td>0.54</td><td>-</td><td>-</td><td>-</td><td>91.2</td><td>22.9</td></tr></table>

RIMs activated per-step. Our ablation results (Appendix D.1) show that all major components of the RIMs model are necessary to achieve this generalization. This is evidence that RIMs can specialize over distinct patterns in the data and improve generalization to settings where these patterns change.

Sequential MNIST Resolution Task: RIMs are motivated by the hypothesis that generalization performance can benefit from modules which only activate on relevant parts of the sequence. For further evidence that RIMs can achieve this out-of-distribution, we consider the task of classifying MNIST digits as sequences of pixels (Krueger et al., 2016) and assay generalization to images of resolutions different from those seen during training. Our intuition is that the RIMs model should have distinct subsets of the RIMs activated for pixels with the digit and empty pixels. RIMs should generalize better to higher resolutions by keeping RIMs dormant which store pixel information over empty regions of the image.

Results: Table 1 shows the result of the proposed model on the Sequential MNIST Resolution Task. If the train and test sequence lengths agree, both models achieve comparable test set performance. However, RIMs model is relatively robust to changing the sequence length (by changing the image resolution), whereas the LSTM performance degraded more severely. This can be seen as a more involved analogue of the copying task, as MNIST digits contain large empty regions. It is essential that the model be able to store information and pass gradients through these regions. The RIMs outperform strong baselines such as Transformers, EntNet, RMC, and (DNC) (Graves et al., 2016).

# 4.2 RIMS LEARN TO SPECIALIZE OVER OBJECTS AND GENERALIZE BETWEEN THEM

We have shown that RIMs can specialize over temporal patterns. We now turn our attention to assaying whether RIMs can specialize to objects, and show improved generalization to cases where we add or remove objects at test time.

Bouncing Balls Environment: We consider a synthetic "bouncing balls" task in which multiple balls (of different masses and sizes) move using basic Newtonian physics (Van Steenkiste et al., 2018). What makes this task particularly suited to RIMs is that the balls move independently most of the time, except when they collide. During training, we predict the next frame at each time step using teacher forcing (Williams & Zipser, 1989). We can then use this model to generate multi-step rollouts. As a preliminary experiment, we train on sequences of length 51 (the previous standard), using a binary cross entropy loss when predicting the next frame. We consider LSTMs as baselines. We then produce rollouts, finding that RIMs are better able to predict future motion (Figure 4).

We take this further by evaluating RIMs on environments where the test setup is different from the training setup. First we consider training with 4 balls and evaluating on an environment with 6-8 balls. Second, we consider training with 6-8 balls and evaluating with just 4 balls. Robustness in these settings requires a degree of invariance w.r.t. the number of balls.

In addition, we consider a task where we train on 4 balls and then evaluate on sequences where part the visual space is occluded by a "curtain." This allows us to assess the ability of balls to be tracked (or remembered) through the occluding region. Our experimental results on these generalization tasks

![](images/d1ac54b7abf5cef3abdde37b31d4af4632a10a333fdc4e221a87f78dfb33fa1b.jpg)  
Figure 4: Handling Novel Out-of-Distribution Variations. We study the performance of RIMs compared to an LSTM baseline (4 left plots). The first 15 frames of ground truth (yellow, orange) are fed in and then the system is rolled out for the next 35 time steps (blue, purple). During the rollout phase, RIMs perform better than the LSTMs in accurately predicting the dynamics of the balls as reflected by the lower Cross Entropy (CE) [blue for RIMs, purple for LSTMs]. Notice the substantially better out-of-distribution generalization of RIMs when testing on a number of objects different from the one seen during training. (2nd to 4th plot). We also show (right plot) improved out-of-distribution generalization (F1 score) as compared to LSTM and RMC (Santoro et al., 2018) on another partial observation video prediction task. X-axis = number of balls. For these experiments, the RIMs and baselines get an input image at each time step (see Appendix D.5, figure. 14 for magnified image as well as more details). Here, TTO refers to the time travelling oracle upper bound baseline, that does not model the dynamics, and has access to true dynamics.

(Figure 4) show that RIMs substantially improve over an LSTM baseline. We found that increasing the capacity of the LSTM from 256 to 512 units did not substantially change the performance gap, suggesting that the improvement from RIMs is not primarily related to capacity.

Environment with Novel Distractors: We next consider an object-picking reinforcement learning task from BabyAI (Chevalier-Boisvert et al., 2018) in which an agent must retrieve a specific object in the presence of distractors. We use a partially observed formulation of the task, where the agent only sees a small number of squares ahead of it. These tasks are difficult to solve (Chevalier-Boisvert et al., 2018) with standard RL algorithms, due to (1) the partial observability of the environment and (2) the sparsity of the reward, given that the agent receives a reward only after reaching the goal. During evaluation, we introduce new distractors to the environment which were not observed during training.

Figure 5 shows that RIMs outperform LSTMs on this task (details in appendix). When evaluating with known distractors, the RIM model achieves perfect performance while the LSTM struggles. When evaluating in an environment with novel unseen distractors the RIM doesn't achieve perfect performance but strongly outperforms the LSTM. An LSTM with a single memory flow may struggle to keep the distracting elements separate from elements which are necessary for the task, while the RIMs model uses attenuated LSTMs.

![](images/92ea07dd3fde9c249f691b64e5c328fb3fcb3b5f12676d7447eb2ac773158417.jpg)  
Figure 5: Robustness to Novel Distractors.: Left: performance of the proposed method (blue) compared to an LSTM baseline (red) in solving the object picking task in the presence of distractors. Right: performance of proposed method and the baseline when novel distractors are added.

tion to control which RIMs receive information at each step as well as what information they receive (as a function of their hidden state). This "top-down" attention results in a diminished representation of the distractor, not only enhancing the target visual information, but also suppressing irrelevant information.

# 4.3 RIMS IMPROVE GENERALIZATION IN COMPLEX ENVIRONMENTS

We have investigated how RIMs use specialization to improve generalization to changing important factors of variation in the data. While these improvements have often been striking, it raises a question: what factors of variation should be changed between training and evaluation? One setting where factors of variation change naturally is in reinforcement learning, as the data received from an environment changes as the agent learns and improves. We conjecture that when applied to reinforcement learning, an agent using RIMs may be able to learn faster as its specialization leads to improved generalization to previously unseen aspects of the environment. To investigate this we use an RL agent trained using Proximal Policy Optimization (PPO) (Schulman et al., 2017) with a recurrent network producing the policy. We employ an LSTM as a baseline, and compare results to

the RIMs architecture. This was a simple drop-in replacement and did not require changing any of the hyperparameters for PPO. We experiment on the whole suite of Atari games and find that simply replacing the LSTM with RIMs greatly improves performance (Figure 6).

![](images/44af4fd8060a07e45361a49d91cacb5663f87e386abb60f43868236c44e6167f.jpg)  
Figure 6: RIMs-PPO relative score improvement over LSTM-PPO baseline (Schulman et al., 2017) across all Atari games averaged over 3 trials per game. In both cases, PPO was used with the exact same settings, and the only change is the choice of recurrent architecture. More detailed experiments with learning curves as well as comparisons with external baselines are in Appendix C.

There is also an intriguing connection between the selective activation in RIMs and the concept of affordances from cognitive psychology (Gibson, 1977; Cisek & Kalaska, 2010). To perform well in environments with a dynamic combination of risks and opportunities, an agent should be ready to adapt immediately, executing actions which are at least partially prepared. This suggests agents should process sensory information in a contextual manner, building representations of potential actions that the environment currently affords. For instance, in Demon Attack, one of the games where RIMs exhibit strong performance gains, the agent must quickly choose between targeting distant aliens to maximize points and avoiding fire from close-by aliens to avoid destruction (indeed both types of aliens are always present, but which is relevant depends on the player's position). We hypothesize that in cases like this, selective activation of RIMs allows the agent to rapidly adapt its information processing to the types of actions relevant to the current context.

# 4.4 DISCUSSION AND ABLATIONS

Sparse Activation is necessary, but works for a wide range of hyperparameters: On the copying task, we tried a wide variety of sparsity levels for different numbers of RIMs, and found that using a sparsity level between  $30\%$  to  $70\%$  performed optimally, suggesting that the sparsity hyperparameter is fairly flexible (refer to Table 4, 5 in appendix). On Atari we found that using  $k_{A} = 5$  slightly improved over results compared with  $k_{A} = 4$ , but both had similar performance across the vast majority of games.

Input-attention is necessary: We study the scenario where we remove the input attention process (i.e the top-down competition between different RIMs) but still allow the RIMs to communicate with attention. We found that this degraded results substantially on Atari but still outperformed the LSTM baseline. See (Figure 21) in appendix for more details.

Communication between RIMs improves performance: For copying and sequential MNIST, we performed an ablation where we remove the communication between RIMs and varied the number of RIMs and the number of activated RIMs (Refer to Table 4 in appendix.). We found that the communication between RIMs is essential for good performance.

# 5 CONCLUSION

Many systems of interest comprise multiple dynamical processes that operate relatively independently and only occasionally have meaningful interactions. Despite this, most machine learning models employ the opposite inductive bias, i.e., that all processes interact. This can lead to poor generalization and lack of robustness to changing task distributions. We have proposed a new architecture, Recurrent Independent Mechanisms (RIMs), in which we learn multiple recurrent modules that are independent by default, but interact sparingly. For the purposes of this paper, we note that the notion of RIMs is not limited to the particular architecture employed here. The latter is used as a vehicle to assay and validate our overall hypothesis (cf. Appendix A), but better architectures for the RIMs model can likely be found.

# REFERENCES

Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 39-48, 2016.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.  
Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018.  
Eric B Baum and David Haussler. What size net gives valid generalization? In Advances in neural information processing systems, pp. 81-90, 1989.  
Yoshua Bengio. The consciousness prior. arXiv preprint arXiv:1709.08568, 2017.  
Yoshua Bengio, Tristan Deleu, Nasim Rahaman, Rosemary Ke, Sébastien Lachapelle, Olexa Bilaniuk, Anirudh Goyal, and Christopher Pal. A meta-transfer objective for learning to disentangle causal mechanisms. arXiv:1901.10912, 2019.  
Léon Bottou and Patrick Gallinari. A framework for the cooperation of learning algorithms. In Advances in neural information processing systems, pp. 781-788, 1991.  
Matthew Botvinick and Todd Braver. Motivation and cognitive control: from behavior to neural mechanism. Annual review of psychology, 66, 2015.  
Michael M Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric deep learning: going beyond euclidean data. IEEE Signal Processing Magazine, 34(4):18-42, 2017.  
Maxime Chevalier-Boisvert and Lucas Willems. Minimalistic gridworld environment for openai gym, 2018.  
Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Salem Lahlou, Lucas Willems, Chitwan Saharia, Thien Huu Nguyen, and Yoshua Bengio. Babyai: First steps towards grounded language learning with a human in the loop. arXiv preprint arXiv:1810.08272, 2018.  
Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron C Courville, and Yoshua Bengio. A recurrent latent variable model for sequential data. In Advances in neural information processing systems, pp. 2980-2988, 2015.  
Junyoung Chung, Sungjin Ahn, and Yoshua Bengio. Hierarchical multiscale recurrent neural networks. arXiv preprint arXiv:1609.01704, 2016.  
Paul Cisek and John F Kalaska. Neural mechanisms for interacting with a world full of action choices. Annual review of neuroscience, 33:269-298, 2010.  
Emily Denton and Rob Fergus. Stochastic video generation with a learned prior. arXiv preprint arXiv:1802.07687, 2018.  
Robert Desimone and Jody Duncan. Neural mechanisms of selective visual attention. Annual Review of Neuroscience, 18:193-222, 1995.  
A. Dickinson. Actions and habits: the development of behavioural autonomy. Philosophical Transactions of the Royal Society B: Biological Sciences, 308(1135):67-78, 1985. ISSN 0080-4622. doi: 10.1098/rstb.1985.0010.  
Salah El Hhi and Yoshua Bengio. Hierarchical recurrent neural networks for long-term dependencies. In Advances in neural information processing systems, pp. 493-499, 1996.

Chrisantha Fernando, Dylan Banarse, Charles Blundell, Yori Zwols, David Ha, Andrei A Rusu, Alexander Pritzel, and Daan Wierstra. Pathnet: Evolution channels gradient descent in super neural networks. arXiv preprint arXiv:1701.08734, 2017.  
James J Gibson. The theory of affordances. Hilldale, USA, 1(2), 1977.  
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1263-1272. JMLR.org, 2017.  
Anirudh Goyal, Riashat Islam, Daniel Strouse, Zafarali Ahmed, Matthew Botvinick, Hugo Larochelle, Sergey Levine, and Yoshua Bengio. Infobot: Transfer and exploration via the information bottleneck. arXiv preprint arXiv:1901.10902, 2019a.  
Anirudh Goyal, Shagun Sodhani, Jonathan Binas, Xue Bin Peng, Sergey Levine, and Yoshua Bengio. Reinforcement learning with competitive ensembles of information-constrained primitives. arXiv preprint arXiv:1906.10667, 2019b.  
Alex Graves, Greg Wayne, and Ivo Danihelka. Neural tuning machines. arXiv preprint arXiv:1410.5401, 2014a.  
Alex Graves, Greg Wayne, and Ivo Danihelka. Neural tuning machines. CoRR, abs/1410.5401, 2014b.  
Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, et al. Hybrid computing using a neural network with dynamic external memory. Nature, 538(7626):471, 2016.  
David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.  
Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. arXiv preprint arXiv:1811.04551, 2018.  
Mikael Henaff, Jason Weston, Arthur Szlam, Antoine Bordes, and Yann LeCun. Tracking the world state with recurrent entity networks. arXiv preprint arXiv:1612.03969, 2016.  
Geoffrey E Hinton, Sara Sabour, and Nicholas Frosst. Matrix capsules with em routing. 2018.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Robert A Jacobs, Michael I Jordan, Steven J Nowlan, Geoffrey E Hinton, et al. Adaptive mixtures of local experts. Neural computation, 3(1):79-87, 1991.  
Yacine Jernite, Edouard Grave, Armand Joulin, and Tomas Mikolov. Variable computation in recurrent neural networks. arXiv preprint arXiv:1611.06188, 2016.  
Nan Rosemary Ke, Anirudh Goyal, Olexa Bilaniuk, Jonathan Binas, Michael C Mozer, Chris Pal, and Yoshua Bengio. Sparse attentive backtracking: Temporal credit assignment through reminding. In Advances in Neural Information Processing Systems, pp. 7640-7651, 2018.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and Richard Zemel. Neural relational inference for interacting systems. arXiv preprint arXiv:1802.04687, 2018.  
Louis Kirsch, Julius Kunze, and David Barber. Modular networks: Learning to decompose neural computation. In Advances in Neural Information Processing Systems, pp. 2408-2418, 2018.  
Wouter Kool and Matthew Botvinick. Mental labour. Nature human behaviour, 2(12):899-908, 2018.  
Ilya Kostrikov. Pytorch implementations of reinforcement learning algorithms, 2018.

Jan Koutnik, Klaus Greff, Faustino Gomez, and Juergen Schmidhuber. A clockwork rnn. arXiv preprint arXiv:1402.3511, 2014.  
David Krueger, Tegan Maharaj, János Kramár, Mohammad Pezeshki, Nicolas Ballas, Nan Rosemary Ke, Anirudh Goyal, Yoshua Bengio, Aaron Courville, and Chris Pal. Zoneout: Regularizing rnns by randomly preserving hidden activations. arXiv preprint arXiv:1606.01305, 2016.  
Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao. Independently recurrent neural network (indrnn): Building a longer and deeper rnn. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5457-5466, 2018.  
Daniel Neil, Michael Pfeiffer, and Shih-Chii Liu. Phased LSTM: Accelerating recurrent network training for long or event-based sequences. In Advances in neural information processing systems, pp. 3882-3890, 2016.  
Giambattista Parascandolo, Niki Kilbertus, Mateo Rojas-Carulla, and Bernhard Scholkopf. Learning independent causal mechanisms. In Proceedings of the 35th International Conference on Machine Learning (ICML), pp. 4033-4041, 2018.  
Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, New York, NY, 2nd edition, 2009.  
Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. *Elements of Causal Inference - Foundations and Learning Algorithms*. MIT Press, Cambridge, MA, USA, 2017. ISBN 978-0-262-03731-0.  
David Raposo, Adam Santoro, David Barrett, Razvan Pascanu, Timothy Lillicrap, and Peter Battaglia. Discovering objects and their relations from entangled scene representations. arXiv preprint arXiv:1702.05068, 2017.  
Scott Reed and Nando De Freitas. Neural programmer-interpreters. arXiv preprint arXiv:1511.06279, 2015.  
Eric Ronco, Henrik Gollee, and Peter J Gawthrop. Modular neural networks and self-decomposition. Technical Report CSC-96012, 1997.  
Clemens Rosenbaum, Tim Klinger, and Matthew Riemer. Routing networks: Adaptive selection of non-linear functions for multi-task learning. arXiv preprint arXiv:1711.01239, 2017.  
Clemens Rosenbaum, Ignacio Cases, Matthew Riemer, and Tim Klinger. Routing networks and the challenges of modular and compositional computation. arXiv preprint arXiv:1904.12774, 2019.  
Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.  
Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. Dynamic routing between capsules. In Advances in neural information processing systems, pp. 3856-3866, 2017.  
Adam Santoro, David Raposo, David G Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, and Timothy Lillicrap. A simple neural network module for relational reasoning. In Advances in neural information processing systems, pp. 4967-4976, 2017.  
Adam Santoro, Ryan Faulkner, David Raposo, Jack W. Rae, Mike Chrzanowski, Theophane Weber, Daan Wierstra, Oriol Vinyals, Razvan Pascanu, and Timothy P. Lillicrap. Relational recurrent neural networks. CoRR, abs/1806.01822, 2018. URL http://arxiv.org/abs/1806.01822.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61-80, 2008.  
Jürgen Schmidhuber. One big net for everything. arXiv preprint arXiv:1802.08864, 2018.  
Bernhard Schölkopf, Dominik Janzing, Jonas Peters, Eleni Sgouritsa, Kun Zhang, and Joris Mooij. On causal and anticausal learning. In J. Langford and J. Pineau (eds.), Proceedings of the 29th International Conference on Machine Learning (ICML), pp. 1255-1262, New York, NY, USA, 2012. Omnipress.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.  
Herbert A Simon. The architecture of complexity. In *Facets of systems science*, pp. 457-476. Springer, 1991.  
Shagun Sodhani, Anirudh Goyal, Tristan Deleu, Yoshua Bengio, Sergey Levine, and Jian Tang. Learning powerful policies by using consistent dynamics model. arXiv preprint arXiv:1906.04355, 2019.  
Andrea Tacchetti, H Francis Song, Pedro AM Mediano, Vinicius Zambaldi, Neil C Rabinowitz, Thore Graepel, Matthew Botvinick, and Peter W Battaglia. Relational forward models for multi-agent learning. arXiv preprint arXiv:1809.11044, 2018.  
Yee Teh, Victor Bapat, Wojciech M Czarnecki, John Quan, James Kirkpatrick, Raia Hadsell, Nicolas Heess, and Razvan Pascanu. Distral: Robust multitask reinforcement learning. In Advances in Neural Information Processing Systems, pp. 4496-4506, 2017.  
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026-5033. IEEE, 2012.  
Sjoerd Van Steenkiste, Michael Chang, Klaus Greff, and Jürgen Schmidhuber. Relational neural expectation maximization: Unsupervised discovery of objects and their interactions. arXiv preprint arXiv:1802.10353, 2018.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pp. 5998-6008, 2017.  
H. L. F. von Helmholtz. Handbuch der physiologischen Optik, volume III. Voss, 1867.  
Erich von Holst and Horst Mittelstaedt. Das reafferenzprinzip. Naturwissenschaften, 37(20):464-476, Jan 1950. doi: 10.1007/BF00622503.  
Nicholas Watters, Daniel Zoran, Theophane Weber, Peter Battaglia, Razvan Pascanu, and Andrea Tacchetti. Visual interaction networks: Learning a physics simulator from video. In Advances in neural information processing systems, pp. 4539-4547, 2017.  
Ronald J Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural computation, 1(2):270-280, 1989.
