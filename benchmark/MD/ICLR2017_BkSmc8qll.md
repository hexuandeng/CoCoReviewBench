# DYNAMIC NEURAL TURING MACHINE WITH CONTINUOUS AND DISCRETE ADDRESSING SCHEMES

Caglar Gulcehre*, Sarath Chandar*, Kyunghyun Cho†, Yoshua Bengio*

* University of Montreal, name.lastname@umontreal.ca

† New York University, name.lastname@nyu.edu

# ABSTRACT

In this paper, we extend neural Turing machine (NTM) into a dynamic neural Turing machine (D-NTM) by introducing a trainable memory addressing scheme. This addressing scheme maintains for each memory cell two separate vectors, content and address vectors. This allows the D-NTM to learn a wide variety of location-based addressing strategies including both linear and nonlinear ones. We implement the D-NTM with both continuous, differentiable and discrete, non-differentiable read/write mechanisms. We investigate the mechanisms and effects for learning to read and write to a memory through experiments on Facebook bAbI tasks using both a feedforward and GRU-controller. The D-NTM is evaluated on a set of Facebook bAbI tasks and shown to outperform NTM and LSTM baselines. We also provide further experimental results on sequential MNIST, associative recall and copy tasks.

# 1 INTRODUCTION

Designing general-purpose learning algorithms is one of the long-standing goals of artificial intelligence. Despite the success of deep learning in this area (see, e.g., (Goodfellow et al., 2016)) there are still a set of complex tasks that are not well addressed by conventional neural networks. Those tasks often require a neural network to be equipped with an explicit, external memory in which a larger, potentially unbounded, set of facts need to be stored. They include, but are not limited to, episodic question-answering (Weston et al., 2015b; Hermann et al., 2015; Hill et al., 2015), compact algorithms (Zaremba et al., 2015), dialogue (Serban et al., 2016; Vinyals & Le, 2015) and video caption generation (Yao et al., 2015).

Recently two promising approaches based on neural networks to this type of tasks have been proposed. Memory networks (Weston et al., 2015b) explicitly store all the facts, or information, available for each episode in an external memory (as continuous vectors) and use the attention-based mechanism to index them when returning an output. On the other hand, neural Turing machines (NTM, (Graves et al., 2014)) read each fact in an episode and decides whether to read, write the fact or do both to the external, differentiable memory.

A crucial difference between these two models is that the memory network does not have a mechanism to modify the content of the external memory, while the NTM does. In practice, this leads to easier learning in the memory network, which in turn resulted in it being used more in real tasks (Bordes et al., 2015; Dodge et al., 2015). On the contrary, the NTM has mainly been tested on a series of small-scale, carefully-crafted tasks such as copy and associative recall. The NTM, however, is more expressive, precisely because it can store and modify the internal state of the network as it processes an episode.

The original NTM supports two modes of addressing (which can be used simultaneously.) They are content-based and location-based addressing. We notice that the location-based strategy is based on linear addressing. The distance between each pair of consecutive memory cells is fixed to a constant. We address this limitation, in this paper, by introducing a learnable address vector for each memory cell of the NTM with least recently used memory addressing mechanism, and we call this variant a dynamic neural Turing machine (D-NTM).

We evaluate the proposed D-NTM on the full set of Facebook bAbI task (Weston et al., 2015b) using either continuous, differentiable attention or discrete, non-differentiable attention (Zaremba & Sutskever, 2015) as an addressing strategy. Our experiments reveal that it is possible to use the discrete, non-differentiable attention mechanism, and in fact, the D-NTM with the discrete attention and GRU controller outperforms the one with the continuous attention. After we published our paper on arXiv, a new extension of NTM called DNC (Graves et al., 2016) has also provided results on bAbI task as well.

We also provide results on sequential-MNIST and algorithmic tasks proposed by (Graves et al., 2014) in order to investigate the ability of our model when dealing with long-term dependencies<sup>1</sup>

# Our Contributions

1. We propose a generalization of Neural Turing Machine called a dynamic neural Turing machine (D-NTM) which employs a learnable and location-based addressing.  
2. We demonstrate the application of neural Turing machines on a more natural and less toyish task: episodic question-answering besides the toy tasks. We provide detailed analysis of our model on this task.  
3. We propose to use the discrete attention mechanism and empirically show that, it can outperform the continuous attention based addressing for episodic QA task.  
4. We propose a curriculum strategy for our model with the feedforward controller and discrete attention that improves our results significantly.

# 2 DYNAMIC NEURAL TURING MACHINE

The proposed dynamic neural Turing machine (D-NTM) extends the neural Turing machine (NTM, Graves et al., 2014)) which has a modular design. The NTM consists of two main modules, a controller and, a memory. The controller, which is often implemented as a recurrent neural network, issues a command to the memory so as to read, write to and erase a subset of memory cells. Although the memory was originally envisioned as an integrated module, it is not necessary, and the memory may be an external, black box (Zaremba & Sutskever, 2015).

# 2.1 CONTROLLER

At each time step  $t$ , the controller (1) receives an input value  $\mathbf{x}^t$ , (2) addresses and reads the memory and creates the content vector  $\phi^t$ , (3) erases/writes a portion of the memory, (4) updates its own hidden state  $\mathbf{h}_t$ , and (5) outputs a value  $\mathbf{y}^t$  (if needed.) In this paper, we use both a gated recurrent unit (GRU, (Cho et al., 2014)) and a feedforward-controller to implement the controller such that for a GRU controller

$$
\mathbf {h} ^ {t} = \operatorname {G R U} \left(\mathbf {x} ^ {t}, \mathbf {h} ^ {t - 1}, \phi^ {t}\right) \tag {1}
$$

or for a feedforward-controller

$$
\mathbf {h} ^ {t} = \sigma \left(\mathbf {x} ^ {t}, \phi^ {t}\right). \tag {2}
$$

# 2.2 MEMORY

We use a rectangular matrix  $\mathbf{M} \in \mathbb{R}^{N \times d_h}$  to denote  $N$  memory cells. Unlike the original NTM, we partition each memory cell vector into two parts:

$$
\mathbf {M} = [ \mathbf {A}; \mathbf {C} ].
$$

The first part  $\mathbf{A}$  is a learnable address matrix, and the second  $\mathbf{C}$  a content matrix. In other words, each memory cell  $\mathbf{m}_i$  is now

$$
\mathbf {m} _ {i} = \left[ \mathbf {a} _ {i}; \mathbf {c} _ {i} \right].
$$

The address part  $\mathbf{a}_i$  is considered a model parameter that is updated during training. During inference, the address part is not overwritten by the controller and remains constant. On the other hand, the content part  $\mathbf{c}_i$  is both read and written by the controller both during training and inference. At the beginning of each episode, the content part  $\mathbf{C}$  is refreshed to be an all-zero matrix. This introduction of the learnable address portion for each memory cell allows the model to learn sophisticated location-based addressing strategies. A similar addressing mechanism is also explored in (Reed & de Freitas, 2015) in the context of learning program traces.

# 2.3 MEMORY ADDRESSING

Memory addressing in the D-NTM is equivalent to computing an  $N$ -dimensional address vector. The D-NTM computes three such vectors for respectively reading  $\mathbf{w}^t$ , erasing  $\mathbf{e}^t$  and writing  $\mathbf{u}^t$ .

![](images/51e7908240ff8a076083ea565fde41951f9b97d4d1fc6ca34cd100ef02abeaa9.jpg)  
Figure 1: A graphical illustration of the proposed dynamic neural Turing machine with the recurrent-controller. The controller receives the fact as a continuous vector encoded by a recurrent neural network, computes the read and write weights for addressing the memory. If the D-NTM automatically detects that a query has been received, it returns an answer and terminates.

Specifically for writing, the controller further computes a candidate memory content vector  $\bar{\mathbf{m}}^t$  based on its current hidden state of the controller  $\mathbf{h}^t$  and the input of the controller scaled with a scalar  $\alpha^t$  which is a function of the hidden state and the input of the controller as well, see Eqn 4.

$$
\alpha^ {t} = \mathrm {f} \left(h ^ {t}, x ^ {t}\right), \tag {3}
$$

$$
\bar {\mathbf {m}} ^ {t} = \operatorname {R e L U} \left(\mathbf {W} _ {m} \mathbf {h} ^ {t} + \alpha^ {t} \mathbf {W} _ {x} \mathbf {x} ^ {t} + \mathbf {b} _ {\mathbf {m}}\right). \tag {4}
$$

Reading With the read vector  $\mathbf{w}^t$ , the memory content vector  $\phi^t$  is retrieved by

$$
\phi^ {t} = \left(\mathbf {w} ^ {t}\right) ^ {\top} \mathbf {M} ^ {t - 1}, \tag {5}
$$

where  $\mathbf{w}^t$  is a row vector.

Erasing and Writing Given the erase, write and candidate memory content vectors  $(\mathbf{e}^t, u_j^t$ , and  $\bar{\mathbf{m}}^t$  respectively) generated by a simple MLP conditioned on the hidden state of the controller  $\mathbf{h}^t$ , the memory matrix is updated by,

$$
\mathbf {m} _ {j} ^ {t} = \left(1 - \mathbf {e} ^ {t} u _ {j} ^ {t}\right) \odot \mathbf {m} _ {j} ^ {t - 1} + u _ {j} ^ {t} \bar {\mathbf {m}} ^ {t}. \tag {6}
$$

where the subscript  $j$  in  $\mathbf{m}_j^t$  denotes the  $j$ -th row of the memory matrix  $\mathbf{M}^t$ .

No Operation (NOP) As found in (Joulin & Mikolov, 2015), an additional NOP action might be beneficial for the controller not to access the memory once in a while. We model this situation by designating one memory cell as a NOP cell. Reading or writing from this memory cell is ignored.

# 2.4 LEARNING

Once the proposed D-NTM is executed, it returns the output distribution  $p(\mathbf{y}|\mathbf{x}_1,\dots ,\mathbf{x}_T)$ . As a result, we define a cost function as the negative log-likelihood:

$$
C (\theta) = \frac {1}{N} \sum_ {n = 1} ^ {N} - \log p \left(\mathbf {y} ^ {n} \mid \mathbf {x} _ {1} ^ {n}, \dots , \mathbf {x} _ {T} ^ {n}\right), \tag {7}
$$

where  $\theta$  is a set of all the parameters. As the proposed D-NTM, just like the original NTM, is fully differentiable end-to-end, we can compute the gradient of this cost function using backpropagation and learn the parameters of the model with a gradient-based optimization algorithm, such as stochastic gradient descent, to train it end-to-end.

# 3 ADDRESSING MECHANISM

# 3.1 ADDRESS VECTORS

Each of the address vectors (read, write and erase) is computed in an identical manner which we describe here.

First, the controller computes a key vector:

$$
\mathbf {k} ^ {t} = \mathbf {W} _ {k} ^ {\top} \mathbf {h} ^ {t} + \mathbf {b} _ {k},
$$

where  $\mathbf{W}_k$  and  $\mathbf{b}_k$  are the parameters for this specific head (either read, write or erase.) Also, the sharpening factor  $\beta_{t}$  is computed:

$$
\beta_ {t} = \operatorname {s o f t p l u s} \left(\mathbf {u} _ {\beta} ^ {\top} \mathbf {h} ^ {t} + b _ {\beta}\right) + 1.
$$

$\mathbf{u}_{\beta}$  and  $b_{\beta}$  are the parameters of the sharpening  $\beta_{t}$ .

The address vector is then computed by

$$
z _ {i} ^ {t} = \beta^ {t} S \left(\mathbf {k} ^ {t}, \mathbf {m} _ {i} ^ {t}\right) \tag {8}
$$

$$
w _ {i} ^ {t} = \frac {\exp \left(z _ {i} ^ {t}\right)}{\sum_ {j} \exp \left(z _ {j} ^ {t}\right)}, \tag {9}
$$

where the similarity function  $S$  is defined as

$$
S \left(\mathbf {x}, \mathbf {y}\right) = \frac {\mathbf {x} \cdot \mathbf {y}}{\left(\left\| \mathbf {x} \right\| \left\| \mathbf {y} \right\| + \epsilon\right)}.
$$

# 3.2 MULTI-STEP ADDRESSING

At each time-step, controller may require more than one-step in order to access to the memory. The original NTM addresses this by implementing multiple sets of read, erase and write heads. In this paper, we explore an option of allowing each head to operate more than once at each time step, similar to the multi-hop mechanism from the end-to-end memory network (Sukhbaatar et al., 2015).

# 3.3 DYNAMIC LEAST RECENTLY USED ADDRESSING

We introduce a memory addressing schema that can learn to put more emphasis on the least recently used (LRU) memory locations. As observed in (Santoro et al., 2016; Rae et al., 2016), we find it easier to learn the write operations with the use of LRU addressing.

To learn a LRU based addressing, first we compute the exponentially moving averages of the logits  $(\mathbf{z}_t)$  as  $\mathbf{v}_t$ ,  $\mathbf{v}_t \gets 0.1\mathbf{v}_{t-1} + 0.9\mathbf{z}_t$ . We rescale the accumulated  $\mathbf{v}_t$  with  $\gamma_t$ , such that the controller adjusts the influence of how much previously written memory locations should effect the attention weights of a particular time-step. Next, we subtract  $\mathbf{v}_t$  from  $\mathbf{z}_t$  in order to reduce the weights of previously read or written memory locations.  $\gamma_t$  is a shallow MLP with a scalar output and it is conditioned on the hidden state of the controller.  $\gamma_t$  is parametrized with the parameters  $\mathbf{u}_\gamma$  and  $\mathbf{b}_\gamma$ ,

$$
\gamma_ {t} = \operatorname {s i g m o i d} \left(\mathbf {u} _ {\gamma} \mathbf {h} _ {t} + \mathbf {b} _ {\gamma}\right), \tag {10}
$$

$$
\mathbf {w} _ {t} \leftarrow \operatorname {s o f t m a x} \left(\mathbf {z} _ {t} - \gamma_ {t} \mathbf {v} _ {t - 1}\right). \tag {11}
$$

This addressing method increases the weights of the least recently used rows of the memory. The magnitude of the influence of the least-recently used memory locations is being learned and adjusted with  $\gamma_{t}$ . Our LRU addressing is dynamic due to the model's ability to switch between pure content-based addressing and LRU. During the training, we do not backpropagate through  $\mathbf{v}_{t}$ . Due to the dynamic nature of this addressing mechanism, it can be used for both read and write operations. If needed, the model will automatically learn to disable LRU while reading from the memory.

# 4 GENERATING DISCRETE ADDRESS VECTORS

In this section, we describe the discrete attention based addressing strategy.

Discrete Addressing Let us use  $\mathbf{w}$  to denote an address vector (either read, write or erase) at time  $t$ . By definition in Eq. (8), every element in this address vector is positive and sums up to one. In other words, we can treat this vector as the probabilities of a categorical distribution  $\mathcal{C}(\mathbf{w})$  with  $\dim(\mathbf{w})$  choices:

$$
p (j) = w _ {j},
$$

where  $w_{j}$  is the  $j$ -th element of  $\mathbf{w}$ . We can readily sample from this categorical distribution and form an one-hot vector  $\tilde{\mathbf{w}}$  such that

$$
\tilde {w} _ {k} = I (k = j),
$$

where  $j\sim \mathcal{C}(\mathbf{w})$  , and  $I$  is an indicator function.

Training We use this sampling-based strategy for all the heads during training. This clearly makes the use of backpropagation infeasible to compute the gradient, as the sampling procedure is not differentiable. Thus, we use REINFORCE (Williams, 1992) together with the three variance reduction techniques—global baseline, input-dependent baseline and variance normalization—suggested in (Mnih & Gregor, 2014).

Let us define  $R(\mathbf{x}) = \log p(\mathbf{y}|\mathbf{x}_1,\dots ,\mathbf{x}_T)$  as a reward. We first center and re-scale the reward by

$$
\tilde {R} (\mathbf {x}) = \frac {R (\mathbf {x}) - b}{\sqrt {\sigma^ {2} + \epsilon}},
$$

where  $b$  and  $\sigma$  is running average and standard deviation of  $R$ . We can further center it for each input  $x$  separately, i.e.,

$$
\tilde {R} (\mathbf {x}) \leftarrow \tilde {R} (\mathbf {x}) - b (\mathbf {x}),
$$

where  $b(\mathbf{x})$  is computed by a baseline network which takes as input  $\mathbf{x}$  and predicts its estimated reward. The baseline network is trained to minimize the Huber loss (Huber, 1964) between the true reward  $\widetilde{R} (\mathbf{x})^{*}$  and the predicted reward  $b(\mathbf{x})$ . We use the Huber loss, which is defined by

$$
H _ {\delta} (x) = \left\{ \begin{array}{l l} x ^ {2} & \text {f o r} | x | \leq \delta , \\ \delta (2 | x | - \delta), & \text {o t h e r w i s e}, \end{array} \right.
$$

due to its robustness. As a further measure to reduce the variance, we regularize the negative entropy of all those category distributions to facilitate a better exploration during training (Xu et al., 2015).

Then, the cost function for each training example is approximated as

$$
\begin{array}{l} C ^ {n} (\theta) = - \log p (\mathbf {y} | \mathbf {x} _ {1: T}, \tilde {w} _ {1: J}, \tilde {u} _ {1: J}, \tilde {e} _ {1: J}) \\ - \sum_ {j = 1} ^ {J} \tilde {R} (\mathbf {x} ^ {n}) \left(\log p \left(\tilde {w} _ {j} | \mathbf {x} _ {1: T}\right) + \log p \left(\tilde {u} _ {j} | \mathbf {x} _ {1: T}\right) + \log p \left(\tilde {e} _ {j} | \mathbf {x} _ {1: T}\right)\right) \\ - \lambda_ {H} \sum_ {j = 1} ^ {J} \left(\mathcal {H} \left(w _ {j} \mid \mathbf {x} _ {1: T}\right) + \mathcal {H} \left(u _ {j} \mid \mathbf {x} _ {1: T}\right) + \mathcal {H} \left(e _ {j} \mid \mathbf {x} _ {1: T}\right)\right). \\ \end{array}
$$

where  $J$  is the number of addressing steps,  $\lambda_{H}$  is the entropy regularization coefficient, and  $\mathcal{H}$  denotes the entropy.

Inference Once training is over, we switch to a deterministic strategy. We simply choose an element of  $\mathbf{w}$  with the largest value to be the index of the target memory cell, such that

$$
\tilde {w} _ {k} = \mathbf {I} (k = \operatorname {a r g m a x} (\mathbf {w})).
$$

Curriculum Learning for the Discrete Attention Training discrete attention with feed-forward controller and REINFORCE is challenging. We propose to use a curriculum strategy for training with the discrete attention in order to tackle this problem. For each minibatch, we sample  $\pi$  from a binomial distribution with the probability  $p^t$ ,  $\pi^t \sim \mathrm{Bin}(p^t)$ . The model will either use the discrete or the continuous-attention based on the  $\pi^t$ . We start the training procedure with  $p^0 = 1$  and during the training  $p^t$  is annealed to 0 by setting  $p^t = \frac{p^0}{\sqrt{1 + t}}$ .

We can rewrite the weights  $\mathbf{w}_t$  as in Equation 12, where it is expressed as the combination of continuous attention weights  $\bar{\mathbf{w}}^t$  and discrete attention weights  $\tilde{\mathbf{w}}^t$  with  $\pi^t$  being a binary variable that chooses to use one of them,

$$
\mathbf {w} ^ {t} \leftarrow \pi^ {t} \bar {\mathbf {w}} ^ {t} + (1 - \pi^ {t}) \tilde {\mathbf {w}} ^ {t}. \tag {12}
$$

By using this curriculum learning strategy, at the beginning of the training, the model learns to use the memory mainly with the continuous attention. As we anneal the  $p^t$ , the model will rely more on the discrete attention.

# 5 REGULARIZING DYNAMIC NEURAL TURING MACHINES

When the controller of D-NTM is a powerful recurrent neural network, it is important to regularize training of the D-NTM so as to avoid suboptimal solutions in which the D-NTM ignores the memory and works as a simple recurrent neural network.

Read-Write Consistency Regularizer One such suboptimal solution we have observed in our preliminary experiments with the proposed D-NTM is that the D-NTM uses the address part A of the memory matrix simply as an additional weight matrix, rather than as a means to accessing the content part C. We found that this pathological case can be effectively avoided by encouraging the read head to point to a memory cell which has also been pointed by the write head. This can be implemented as the following regularization term:

$$
R _ {\mathrm {r w}} (\mathbf {w}, \mathbf {u}) = \lambda \sum_ {t ^ {\prime} = 1} ^ {T} | | 1 - \left(\frac {1}{t ^ {\prime}} \sum_ {t = 1} ^ {t ^ {\prime}} \mathbf {u} _ {t}\right) ^ {\top} \mathbf {w} _ {t ^ {\prime}} | | _ {2} ^ {2} \tag {13}
$$

In the equations above,  $\mathbf{u}_t$  is the write and  $\mathbf{w}_t$  is the read weights.

Next Input Prediction as Regularization Temporal structure is a strong signal that should be exploited by the controller based on a recurrent neural network. We exploit this structure by letting the controller predict the input in the future. We maximize the predictability of the next input by the controller during training. This is equivalent to minimizing the following regularizer:

$$
R _ {\mathrm {p r e d}} (\mathbf {W}) = - \log p (\mathbf {f} _ {t + 1} | \mathbf {f} _ {t}, \mathbf {w} _ {t}, \mathbf {u} _ {t}, \mathbf {M} _ {t}; \mathbf {W}))
$$

where  $f_{t}$  is the current input and  $f_{t + 1}$  is the input at next timestep. We found this regularizer to be effective in our preliminary experiments and use it for bAbI tasks.

# 6 RELATED WORK

A recurrent neural network (RNN), which is used as a controller in the proposed D-NTM, has an implicit memory in the form of recurring hidden states. Even with this implicit memory, a vanilla RNN is however known to have difficulties in storing information for long time-spans (Bengio et al., 1994; Hochreiter, 1991). Long short-term memory (LSTM, (Hochreiter & Schmidhuber, 1997)) and gated recurrent units (GRU, (Cho et al., 2014)) have been found to address this issue. However all these models based solely on RNNs have been found to be limited when they are used to solve, e.g., algorithmic tasks and episodic question-answering.

In addition to the finite random access memory of the neural Turing machine, based on which the D-NTM is designed, other data structures have been proposed as external memory for neural networks. In (Sun et al., 1997; Grefenstette et al., 2015; Joulin & Mikolov, 2015), a continuous, differentiable stack was proposed. In (Zaremba et al., 2015; Zaremba & Sutskever, 2015), grid and tape storages are used. These approaches differ from the NTM in that their memory is unbounded and can grow indefinitely. On the other hand, they are often not randomly accessible.

Memory networks (Weston et al., 2015b) form another family of neural networks with external memory. In this class of neural networks, information is stored explicitly as it is (in the form of its continuous representation) in the memory, without being erased or modified during an episode. Memory networks and their variants have been applied to various tasks successfully (Sukhbaatar et al., 2015; Bordes et al., 2015; Dodge et al., 2015; Xiong et al., 2016). Miller et al. (2016) have also independently proposed the idea of having separate key and value vectors for memory networks.

Another related family of models is the attention-based neural networks. Neural networks with continuous or discrete attention over an input have shown promising results on a variety of challenging tasks, including machine translation (Bahdanau et al., 2015; Luong et al., 2015), speech recognition (Chorowski et al., 2015), machine reading comprehension (Hermann et al., 2015) and image caption generation (Xu et al., 2015).

The latter two, the memory network and attention-based networks, are however clearly distinguishable from the D-NTM by the fact that they do not modify the content of the memory.

# 7 EXPERIMENTS

We provide experimental results to demonstrate the abilities of our model, first on Facebook bAbI task (Weston et al., 2015a). We give detailed analysis and experimental results on this task. We also compare different variations of NTM on bAbI tasks. We have performed experiments on sequential permuted MNIST (Le et al., 2015) and on toy tasks to compare other published models on these tasks with a recurrent controller. The details of our experiments are provided in the supplementary material.

# 7.1 EPISODIC QUESTION-ANSWERING: BABI TASKS

In this section, we evaluate the proposed D-NTM on the recently proposed episodic question-answering task called Facebook bAbI. We use the dataset with 10k training examples per sub-task provided by Facebook. For each episode, the D-NTM reads a sequence of factual sentences followed by a question, all of which are given as natural language sentences. The D-NTM is expected to store and retrieve relevant information in the memory in order to answer the question based on the presented facts. Exact implementation details and hyper-parameter settings are provided in the appendix.

# 7.1.1 GOALS

The goal of this experiment is three-fold. First, we present for the first time the performance of a memory-based network that can both read and write dynamically on the Facebook bAbI tasks<sup>3</sup>. We aim to understand whether a model that has to learn to write an incoming fact to the memory, rather than storing it as it is, is able to work well, and to do so, we compare both the original NTM and proposed D-NTM against an LSTM-RNN.

Second, we investigate the effect of having to learn how to write. The fact that the NTM needs to learn to write likely has adverse effect on the overall performance, when compared to, for instance, end-to-end memory networks (MemN2N, (Sukhbaatar et al., 2015)) and dynamic memory network  $(\mathrm{DMN}+)$ , (Xiong et al., 2016)) both of which simply store the incoming facts as they are. We quantify this effect in this experiment. Lastly, we show the effect of the proposed learnable addressing scheme.

We further explore the effect of using a feedforward controller instead of the GRU controller. In addition to the explicit memory, the GRU controller can use its own internal hidden state as the memory. On the other hand, the feedforward controller must solely rely on the explicit memory, as it is the only memory available.

# 7.1.2 RESULTS AND ANALYSIS

In Table 1, we first observe that the NTMs are indeed capable of solving this type of episodic question-answering better than the vanilla LSTM-RNN. Although the availability of explicit memory in the NTM has already suggested this result, we note that this is the first time neural Turing machines have been used in this specific task.

All the variants of NTM with the GRU controller outperform the vanilla LSTM-RNN. However, not all of them perform equally well. First, it is clear that the proposed dynamic NTM (D-NTM) using the GRU controller outperforms the original NTM with the GRU controller (NTM, CBA only NTM vs. continuous D-NTM, Discrete D-NTM). As discussed earlier, the learnable addressing scheme of the D-NTM allows the controller to access the memory slots by location in a potentially nonlinear way. We expect it to help with tasks that have non-trivial access patterns, and as anticipated, we see a large gain with the D-NTM over the original NTM in the tasks of, for instance, 12 - Conjunction and 17 - Positional Reasoning.

Among the recurrent variants of the proposed D-NTM, we notice significant improvements by using discrete addressing over using continuous addressing. We conjecture that this is due to certain types of tasks that require precise/sharp retrieval of a stored fact, in which case continuous addressing is in disadvantage over discrete addressing. This is evident from the observation that the D-NTM with discrete addressing significantly outperforms that with continuous addressing in the tasks of 8 -

<table><tr><td>Task</td><td>LSTM</td><td>MemN2N</td><td>DMN+</td><td>1-step LBA* NTM</td><td>1-step CBA NTM</td><td>1-step Soft D-NTM</td><td>1-step Discrete D-NTM</td><td>3-steps LBA* NTM</td><td>3-steps CBA NTM</td><td>3-steps Soft D-NTM</td><td>3-steps Discrete D-NTM</td></tr><tr><td>1</td><td>0.00</td><td>0.00</td><td>0.00</td><td>16.30</td><td>16.88</td><td>5.41</td><td>6.66</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>2</td><td>81.90</td><td>0.30</td><td>0.30</td><td>57.08</td><td>55.70</td><td>58.54</td><td>56.04</td><td>61.67</td><td>59.38</td><td>46.66</td><td>62.29</td></tr><tr><td>3</td><td>83.10</td><td>2.10</td><td>1.10</td><td>74.16</td><td>55.00</td><td>74.58</td><td>72.08</td><td>83.54</td><td>65.21</td><td>47.08</td><td>41.45</td></tr><tr><td>4</td><td>0.20</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>5</td><td>1.20</td><td>0.80</td><td>0.50</td><td>1.46</td><td>20.41</td><td>1.66</td><td>1.04</td><td>0.83</td><td>1.46</td><td>1.25</td><td>1.45</td></tr><tr><td>6</td><td>51.80</td><td>0.10</td><td>0.00</td><td>23.33</td><td>21.04</td><td>40.20</td><td>44.79</td><td>48.13</td><td>54.80</td><td>20.62</td><td>11.04</td></tr><tr><td>7</td><td>24.90</td><td>2.00</td><td>2.40</td><td>21.67</td><td>21.67</td><td>19.16</td><td>19.58</td><td>7.92</td><td>37.70</td><td>7.29</td><td>5.62</td></tr><tr><td>8</td><td>34.10</td><td>0.90</td><td>0.00</td><td>25.76</td><td>21.05</td><td>12.58</td><td>18.46</td><td>25.38</td><td>8.82</td><td>11.02</td><td>0.74</td></tr><tr><td>9</td><td>20.20</td><td>0.30</td><td>0.00</td><td>24.79</td><td>24.17</td><td>36.66</td><td>34.37</td><td>37.80</td><td>0.00</td><td>39.37</td><td>32.50</td></tr><tr><td>10</td><td>30.10</td><td>0.00</td><td>0.00</td><td>41.46</td><td>33.13</td><td>52.29</td><td>50.83</td><td>56.25</td><td>23.75</td><td>20.00</td><td>20.83</td></tr><tr><td>11</td><td>10.30</td><td>0.10</td><td>0.00</td><td>18.96</td><td>31.88</td><td>31.45</td><td>4.16</td><td>3.96</td><td>0.28</td><td>30.62</td><td>16.87</td></tr><tr><td>12</td><td>23.40</td><td>0.00</td><td>0.00</td><td>25.83</td><td>30.00</td><td>7.70</td><td>6.66</td><td>28.75</td><td>23.75</td><td>5.41</td><td>4.58</td></tr><tr><td>13</td><td>6.10</td><td>0.00</td><td>0.00</td><td>6.67</td><td>5.63</td><td>5.62</td><td>2.29</td><td>5.83</td><td>83.13</td><td>7.91</td><td>5.00</td></tr><tr><td>14</td><td>81.00</td><td>0.10</td><td>0.20</td><td>58.54</td><td>59.17</td><td>60.00</td><td>63.75</td><td>61.88</td><td>57.71</td><td>58.12</td><td>60.20</td></tr><tr><td>15</td><td>78.70</td><td>0.00</td><td>0.00</td><td>36.46</td><td>42.30</td><td>36.87</td><td>39.27</td><td>35.62</td><td>21.88</td><td>36.04</td><td>40.26</td></tr><tr><td>16</td><td>51.90</td><td>51.80</td><td>45.30</td><td>71.15</td><td>71.15</td><td>49.16</td><td>51.35</td><td>46.15</td><td>50.00</td><td>46.04</td><td>45.41</td></tr><tr><td>17</td><td>50.10</td><td>18.60</td><td>4.20</td><td>43.75</td><td>43.75</td><td>17.91</td><td>16.04</td><td>43.75</td><td>56.25</td><td>21.25</td><td>9.16</td></tr><tr><td>18</td><td>6.80</td><td>5.30</td><td>2.10</td><td>3.96</td><td>47.50</td><td>3.95</td><td>3.54</td><td>47.50</td><td>47.50</td><td>6.87</td><td>1.66</td></tr><tr><td>19</td><td>90.30</td><td>2.30</td><td>0.00</td><td>75.89</td><td>71.51</td><td>73.74</td><td>64.63</td><td>61.56</td><td>63.65</td><td>75.88</td><td>76.66</td></tr><tr><td>20</td><td>2.10</td><td>0.00</td><td>0.00</td><td>1.25</td><td>0.00</td><td>2.70</td><td>3.12</td><td>0.40</td><td>0.00</td><td>3.33</td><td>0.00</td></tr><tr><td>Avg.Err.</td><td>36.41</td><td>4.24</td><td>2.81</td><td>31.42</td><td>33.60</td><td>29.51</td><td>27.93</td><td>32.85</td><td>32.76</td><td>24.24</td><td>21.79</td></tr></table>

Table 1: Test error rates  $(\%)$  on the 20 bAbI QA tasks for models using 10k training examples with the GRU and feedforward controller. FF stands for the experiments that are conducted with feedforward controller. Let us, note that LBA* refers to NTM that uses both LBA and CBA. In this table, we compare multi-step vs single-step addressing, original NTM with location based+content based addressing vs only content based addressing, and discrete vs continuous addressing on bAbI.

Lists/Sets and 11 - Basic Coreference. Furthermore, this is in line with an earlier observation in (Xu et al., 2015), where discrete addressing was found to generalize better in the task of image caption generation.

In Table 2, we also observe that the D-NTM with the feedforward controller and discrete attention performs worse than LSTM and D-NTM with continuous-attention. However, when the proposed curriculum strategy from Sec. 4 is used, the average test error drops from 68.30 to 37.79.

We empirically found training of the feedforward controller more difficult than that of the recurrent controller. We train our feedforward controller based models four times longer (in terms of the number of updates) than the recurrent controller based ones in order to ensure that they are converged for most of the tasks. On the other hand, the models trained with the GRU controller overfit on bAbI tasks very quickly. For example, on tasks 3 and 16 the feedforward controller based model underfits (i.e., high training loss) at the end of the training, whereas with the same number of units the model with the GRU controller can overfit on those tasks after 3,000 updates only.

When our results are compared to the variants of the memory network Weston et al. (2015b) (MemN2N and DMN+), we notice a significant performance gap. We attribute this gap to the difficulty in learning to manipulate and store a complex input.

<table><tr><td>Task</td><td>FF Soft D-NTM</td><td>FF Discrete D-NTM</td><td>FF Discrete* D-NTM</td></tr><tr><td>1</td><td>4.38</td><td>81.67</td><td>14.79</td></tr><tr><td>2</td><td>27.5</td><td>76.67</td><td>76.67</td></tr><tr><td>3</td><td>71.25</td><td>79.38</td><td>70.83</td></tr><tr><td>4</td><td>0.00</td><td>78.65</td><td>44.06</td></tr><tr><td>5</td><td>1.67</td><td>83.13</td><td>17.71</td></tr><tr><td>6</td><td>1.46</td><td>48.76</td><td>48.13</td></tr><tr><td>7</td><td>6.04</td><td>54.79</td><td>23.54</td></tr><tr><td>8</td><td>1.70</td><td>69.75</td><td>35.62</td></tr><tr><td>9</td><td>0.63</td><td>39.17</td><td>14.38</td></tr><tr><td>10</td><td>19.80</td><td>56.25</td><td>56.25</td></tr><tr><td>11</td><td>0.00</td><td>78.96</td><td>39.58</td></tr><tr><td>12</td><td>6.25</td><td>82.5</td><td>32.08</td></tr><tr><td>13</td><td>7.5</td><td>75.0</td><td>18.54</td></tr><tr><td>14</td><td>17.5</td><td>78.75</td><td>24.79</td></tr><tr><td>15</td><td>0.0</td><td>71.42</td><td>39.73</td></tr><tr><td>16</td><td>49.65</td><td>71.46</td><td>71.15</td></tr><tr><td>17</td><td>1.25</td><td>43.75</td><td>43.75</td></tr><tr><td>18</td><td>0.24</td><td>48.13</td><td>2.92</td></tr><tr><td>19</td><td>39.47</td><td>71.46</td><td>71.56</td></tr><tr><td>20</td><td>0.0</td><td>76.56</td><td>9.79</td></tr><tr><td>Avg.Err.</td><td>12.81</td><td>68.30</td><td>37.79</td></tr></table>

Table 2: Test error rates (%) on the 20 bAbI QA tasks for models using 10k training examples with feedforward controller.

We also provide further experiments investigating different extensions on D-NTM in the appendix.

# 7.2SEQUENTIAL  $p$  MNIST

In sequential MNIST task, the pixels of the MNIST digits are provided to the model in scan line order, left to right and top to bottom (Le et al., 2015). At the end of sequence of pixels, the model predicts the label of the digit in the sequence of pixels. We experiment D-NTM on the variation of sequential MNIST where the order of the pixels is randomly shuffled, we call this task as permuted MNIST ( $p$ MNIST). An important contribution of this task to our paper, in particular, is to measure the model's ability to perform well when dealing with long-term dependencies. We report our results in Table  $3^4$ , we observe improvements over other models that we compare against. In Table 3, "discrete addressing with MAB" refers to D-NTM model using REINFORCE with baseline computed from moving averages of the reward. Discrete addressing with IB refers to D-NTM using REINFORCE with input-based baseline.

# 7.3 NTM TOY TASKS

We explore the possibility of using D-NTM to solve algorithmic tasks such as copy and associative recall tasks. We train our model on the same lengths of sequences that is experimented in (Graves et al., 2014). We report our results in Table 4. We find out that D-NTM using continuous-attention can successfully learn the "Copy" and "Associative Recall" tasks.

In Table 4, we train our model on sequences of the same length as the experiments in (Graves et al., 2014) and test the model on the sequences of the maximum length seen during the training. We consider model to be successful on copy or associative recall if its validation cost (binary cross-entropy) is lower than 0.02 over the sequences of maximum length seen during the training. We set the threshold to 0.02 to determine whether a model is successful on a task. Because empirically we observe that the models have higher validation costs perform badly in terms of generalization over the longer sequences. "D-NTM discrete" model in this table is trained with REINFORCE using moving averages to estimate the baseline.

<table><tr><td></td><td>Test Acc</td></tr><tr><td>D-NTM discrete MAB</td><td>89.6</td></tr><tr><td>D-NTM discrete IB</td><td>92.3</td></tr><tr><td>D-NTM cont.</td><td>93.4</td></tr><tr><td>NTM</td><td>90.9</td></tr><tr><td>I-RNN (Le et al., 2015)</td><td>82.0</td></tr><tr><td>Zoneout (Krueger et al., 2016)</td><td>93.1</td></tr><tr><td>LSTM (Krueger et al., 2016)</td><td>89.8</td></tr><tr><td>Unitary-RNN (Arjovsky et al., 2015)</td><td>91.4</td></tr><tr><td>Recurrent Dropout (Krueger et al., 2016)</td><td>92.5</td></tr></table>

<table><tr><td></td><td>Copy Tasks</td><td>Associative Recall</td></tr><tr><td>D-NTM cont.</td><td>Success</td><td>Success</td></tr><tr><td rowspan="2">D-NTM discrete NTM</td><td>Success</td><td>Failure</td></tr><tr><td>Success</td><td>Success</td></tr></table>

Table 4: NTM Toy Tasks.

Table 3: Sequential  $p$  MNIST.

# 8 CONCLUSION AND FUTURE WORK

In this paper we extend neural Turing machines (NTM) by introducing a learnable addressing scheme which allows the NTM to be capable of performing highly nonlinear location-based addressing. This extension, to which we refer by dynamic NTM (D-NTM), is extensively tested with various configurations, including different addressing mechanisms (continuous vs. discrete) and different number of addressing steps, on the Facebook bAbI tasks. This is the first time an NTM-type model was tested on this task, and we observe that the NTM, especially the proposed D-NTM, performs better than vanilla LSTM-RNN. Furthermore, the experiments revealed that the discrete, discrete addressing works better than the continuous addressing with the GRU controller, and our analysis reveals that this is the case when the task requires precise retrieval of memory content.

Our experiments show that the NTM-based models can be weaker than other variants of memory networks which do not learn but have an explicit mechanism of storing incoming facts as they are. We conjecture that this is due to the difficulty in learning how to write, manipulate and delete the content of memory. Despite this difficulty, we find the NTM-based approach, such as the proposed D-NTM,

to be a better, future-proof approach, because it can scale to a much longer horizon (where it becomes impossible to explicitly store all the experiences.)  
On  $p$  MNIST task, we show that our model can outperform other similar type of approaches proposed to deal with the long-term dependencies. On copy and associative recall tasks, we show that our model can solve the algorithmic problems that are proposed to solve with NTM type of models.  
The success of both the learnable address and the discrete addressing scheme suggests two future research directions. First, we should try both of these schemes in a wider array of memory-based models, as they are not specific to the neural Turing machines. Second, the proposed D-NTM needs to be evaluated on a diverse set of applications, such as text summarization (Rush et al., 2015), visual question-answering (Antol et al., 2015) and machine translation, in order to make a more concrete conclusion.

# REFERENCES

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: visual question answering. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pp. 2425-2433, 2015.  
Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. arXiv preprint arXiv:1511.06464, 2015.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In Proceedings Of The International Conference on Representation Learning (ICLR 2015), 2015.  
Yoshua Bengio, Patrice Simard, and Paolo Frasconi. Learning long-term dependencies with gradient descent is difficult. Neural Networks, IEEE Transactions on, 5(2):157-166, 1994.  
Antoine Bordes, Nicolas Usunier, Sumit Chopra, and Jason Weston. Large-scale simple question answering with memory networks. arXiv preprint arXiv:1506.02075, 2015.  
Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.  
Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio. Attention-based models for speech recognition. arXiv preprint arXiv:1506.07503, 2015.  
Tim Coolijmans, Nicolas Ballas, César Laurent, and Aaron Courville. Recurrent batch normalization. arXiv preprint arXiv:1603.09025, 2016.  
Jesse Dodge, Andreea Gane, Xiang Zhang, Antoine Bordes, Sumit Chopra, Alexander Miller, Arthur Szlam, and Jason Weston. Evaluating prerequisite qualities for learning end-to-end dialog systems. CoRR, abs/1511.06931, 2015.  
Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep learning. Book in preparation for MIT Press, 2016. URL http://www.deeplearningbook.org.  
Alex Graves, Greg Wayne, and Ivo Danihelka. Neural tuning machines. arXiv preprint arXiv:1410.5401, 2014.  
Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, et al. Hybrid computing using a neural network with dynamic external memory. Nature, 538(7626): 471-476, 2016.  
Edward Grefenstette, Karl Moritz Hermann, Mustafa Suleyman, and Phil Blunsom. Learning to transduce with unbounded memory. In Advances in Neural Information Processing Systems, pp. 1819-1827, 2015.  
Caglar Gulcehre, Marcin Moczulski, Misha Denil, and Yoshua Bengio. Noisy activation functions. arXiv preprint arXiv:1603.00391, 2016.  
Karl Moritz Hermann, Tomáš Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. arXiv preprint arXiv:1506.03340, 2015.

Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston. The goldilocks principle: Reading children's books with explicit memory representations. arXiv preprint arXiv:1511.02301, 2015.  
Sepp Hochreiter. Untersuchungen zu dynamischen neuronalen netzen. Diploma, Technische Universität München, pp. 91, 1991.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation, 9(8): 1735-1780, 1997.  
Peter J. Huber. Robust estimation of a location parameter. Ann. Math. Statist., 35(1):73-101, 03 1964.  
Armand Joulin and Tomas Mikolov. Inferring algorithmic patterns with stack-augmented recurrent nets. In Advances in Neural Information Processing Systems, pp. 190-198, 2015.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014.  
David Krueger, Tegan Maharaj, János Kramár, Mohammad Pezeshki, Nicolas Ballas, Nan Rosemary Ke, Anirudh Goyal, Yoshua Bengio, Hugo Larochelle, Aaron Courville, et al. Zoneout: Regularizing rnns by randomly preserving hidden activations. arXiv preprint arXiv:1606.01305, 2016.  
Quoc V Le, Navdeep Jaitly, and Geoffrey E Hinton. A simple way to initialize recurrent networks of rectified linear units. arXiv preprint arXiv:1504.00941, 2015.  
Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. In Proceedings Of The Conference on Empirical Methods for Natural Language Processing (EMNLP 2015), 2015.  
Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, and Jason Weston. Key-value memory networks for directly reading documents. CoRR, abs/1606.03126, 2016. URL http://arxiv.org/abs/1606.03126.  
Andriy Mnih and Karol Gregor. Neural variational inference and learning in belief networks. arXiv preprint arXiv:1402.0030, 2014.  
Jack W Rae, Jonathan J Hunt, Tim Harley, Ivo Danihelka, Andrew Senior, Greg Wayne, Alex Graves, and Timothy P Lillicrap. Scaling memory-augmented neural networks with sparse reads and writes. In Advances in NIPS. 2016.  
Scott Reed and Nando de Freitas. Neural programmer-interpreters. arXiv preprint arXiv:1511.06279, 2015.  
Alexander M. Rush, Sumit Chopra, and Jason Weston. A neural attention model for abstractive sentence summarization. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, EMNLP 2015, Lisbon, Portugal, September 17-21, 2015, pp. 379-389, 2015.  
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. One-shot learning with memory-augmented neural networks. arXiv preprint arXiv:1605.06065, 2016.  
Iulian V Serban, Alessandro Sordoni, Yoshua Bengio, Aaron Courville, and Joelle Pineau. Building end-to-end dialogue systems using generative hierarchical neural network models. In Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI-16), 2016.  
Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. arXiv preprint arXiv:1503.08895, 2015.  
Guo-Zheng Sun, C. Lee Giles, and Hsing-Hen Chen. The neural network pushdown automaton: Architecture, dynamics and training. In Adaptive Processing of Sequences and Data Structures, International Summer School on Neural Networks, pp. 296-345, 1997.  
Oriol Vinyals and Quoc Le. A neural conversational model. arXiv preprint arXiv:1506.05869, 2015.  
Jason Weston, Antoine Bordes, Sumit Chopra, and Tomas Mikolov. Towards ai-complete question answering: a set of prerequisite toy tasks. arXiv preprint arXiv:1502.05698, 2015a.  
Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. In Proceedings Of The International Conference on Representation Learning (ICLR 2015), 2015b. In Press.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229-256, 1992.  
Caiming Xiong, Stephen Merity, and Richard Socher. Dynamic memory networks for visual and textual question answering. CoRR, abs/1603.01417, 2016.  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In Proceedings Of The International Conference on Representation Learning (ICLR 2015), 2015.  
Li Yao, Atousa Torabi, Kyunghyun Cho, Nicolas Ballas, Christopher Pal, Hugo Larochelle, and Aaron Courville. Describing videos by exploiting temporal structure. In Computer Vision (ICCV), 2015 IEEE International Conference on. IEEE, 2015.  
Wojciech Zaremba and Ilya Sutskever. Reinforcement learning neural tuning machines. CoRR, abs/1505.00521, 2015.  
Wojciech Zaremba, Tomas Mikolov, Armand Joulin, and Rob Fergus. Learning simple algorithms from examples. arXiv preprint arXiv:1511.07275, 2015.
