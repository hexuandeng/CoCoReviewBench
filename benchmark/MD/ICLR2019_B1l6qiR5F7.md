# ORDERED NEURONS: INTEGRATING TREE STRUCTURES INTO RECURRENT NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recurrent neural network (RNN) models are widely used for processing sequential data governed by a latent tree structure. Previous work shows that RNN models (especially Long Short-Term Memory (LSTM) based models) could learn to exploit the underlying tree structure. However, its performance consistently lags behind that of tree-based models. This work proposes a new inductive bias Ordered Neurons, which enforces an order of updating frequencies between hidden state neurons. We show that the ordered neurons could explicitly integrate the latent tree structure into recurrent models. To this end, we propose a new RNN unit: ON-LSTM, which achieve good performances on four different tasks: language modeling, unsupervised parsing, targeted syntactic evaluation, and logical inference.

# 1 INTRODUCTION

Natural language is usually presented in a sequential format, but the underlying structure of language is not strictly sequential. Linguists agree on a set of rules, or syntax, that governs this structure (Sandra & Taft, 2014), and the structure also dictates how the words compose to form components of sentences. This structure is usually tree-like, despite its presented form. Despite being discovered by linguistics, the real origin of the latent structure is unclear. Some theories point out that this could be related to an underlying mechanism of human cognition (Chomsky & Lightfoot, 2002). This possibility brings more interest in studying the latent structure with artificial neural network approaches, which are inspired by information processing and communication patterns in biological nervous systems.

From a practical point of view, integrating tree structure into a language model is also important for different reasons:

1. to obtain a hierarchical representation with increasing levels of abstraction, a key feature of deep neural networks (Bengio et al., 2009; LeCun et al., 2015; Schmidhuber, 2015);  
2. to capture complex linguistic phenomena, like the long-term dependency problem (Tai et al., 2015) and the compositional effects (Socher et al., 2013);  
3. to provide shortcut for gradient back-propagation (Chung et al., 2016).

Developing deep neural networks that can leverage syntactic knowledge, or at least some tree structure (Williams et al., 2018; Shi et al., 2018), to form better semantic representations have received a great deal of attention in recent years (Shen et al., 2017; Jacob et al., 2018; Bowman et al., 2016; Choi et al., 2018; Yogatama et al., 2016).

One straightforward way of obtaining the tree structure is through a supervised syntactic parser. Trees produced by these parsers have been used to guide the composition of word semantics into sentence semantics (Socher et al., 2013; Bowman et al., 2015), or even to help next word prediction given previous words (Wu et al., 2017). However, supervised parsers are limiting for several reasons: 1) few languages have comprehensive annotated data for supervised parser training; 2) in available language data, syntax rules tend to be broken (e.g. in tweets); and 3) languages change over time with use, so syntax rules may evolve.

On the other hand, learning the tree structure in an unsupervised manner from available data remains an open problem. Many such attempts suffer from inducing trivial structure (e.g., a left-branching or right-branching tree structure (Williams et al., 2018)), or the difficulty in training caused by resort to RL (Yogatama et al., 2016). Further, some methods are relatively complex to implement and train, like the PRPN from Shen et al. (2017).

Recurrent neural networks (RNNs) have proven highly effective at the task of language modeling (Merity et al., 2017; Melis et al., 2017). RNNs implicitly impose a chain structure on the data. This chain structure may seem at odds with the latent non-sequential structure of language and poses several difficulties for the processing of natural language data with deep learning methods, such as capturing long-term dependencies (Bengio et al., 2009), achieving good generalization (Bowman et al., 2015), handling negation (Socher et al., 2013), etc. Meanwhile, some evidence exists that an RNN with sufficient capacity has the potential to encode such a tree structure implicitly (Kuncoro et al., 2018). But, the question remains: Would imposing a tree-structure inductive prior on the model architecture result in better models of language?

In this work, we introduce a new inductive bias for recurrent neural networks: Ordered Neurons. This inductive bias enforces a dependency between the neurons that reflects the life cycle of information stored inside each neuron. In other words, some high-ranking neurons store long-term information, while low-ranking neurons store short-term information. To avoid a fixed division between high-ranking and low-ranking neurons, we further propose a new activation function  $\text{cumax}()$  to actively allocate neurons to store long/short-term information. Based on the  $\text{cumax}()$  and the LSTM architecture, we have designed a new model, ON-LSTM, that enables RNN models to perform tree-like compositions without breaking its sequential form. Our model achieve good performance on four tasks: language modeling, unsupervised constituency parsing, targeted syntactic evaluation (Marvin & Linzen, 2018) and logical inference (Bowman et al., 2015). The result on unsupervised constituency parsing task suggests that the proposed inductive bias aligns with the syntax principles proposed by human experts. The experiments also show that ON-LSTM performs better than standard LSTM models in terms of long-term dependency and longer sequence generalization.

# 2 RELATED WORK

There has been prior work leveraging tree structures for natural language tasks in the literature. Socher et al. (2010); Alvarez-Melis & Jaakkola (2016); Zhou et al. (2017); Zhang et al. (2015) uses labeled data from a treebank to perform supervised learning for inferring parse trees. Socher et al. (2013); Tai et al. (2015) explicitly models the tree-structure using parsing information from an external parser. Later, Bowman et al. (2016) used supervised signals from a parser (Klein & Manning, 2003) to train a stack-augmented neural network.

Theoretically, RNNs and LSTMs can model data produced by context-free grammars and context-sensitive grammars (Gers & Schmidhuber, 2001). However, recent results suggest that introducing structure information into an LSTM model is beneficial. Kuncoro et al. (2018) showed that RNNGs (Dyer et al., 2016), which have an explicit bias to model the syntactic structures, outperform LSTMs on the subject-verb agreement task (Linzen et al., 2016). In our paper, we run a more extensive suite of grammatical tests provided by Marvin & Linzen (2018). Bowman et al. (2014; 2015) also demonstrate that these recursive structures work better for downstream, predictive tasks if the data was generated with such a structure. Interestingly, Shi et al. (2018) suggests that the prescribed grammar tree may not be ideal, but some sort of hierarchical structure, perhaps task dependent, might help. However, the problem of efficiently learning such structures from data remains an open question.

One possible solution would be to develop models with varying time-scales of recurrence as a way of emulating this hierarchy. There has been precedence for such models: El Hihi & Bengio (1996); Schmidhuber (1991); Lin et al. (1998) describe models that model data at different, pre-determined time-scales. More recently, Koutnik et al. (2014) segments an RNN hidden state with different time-scales for updating called the Clockwork RNN. These approaches typically make a strong assumption about the regularity of the hierarchy involved in modelling the data. Chung et al. (2016) proposed a method that, unlike the Clockwork RNN, would learn the multi-scale hierarchical recurrence. However, the model still has a pre-determined depth to the hierarchy, depending on the number of layers it was parameterised with.

In models developed specifically for language modelling, there has been precedent for incorporating syntactic structure for the task (Roark, 2001; Charniak, 2001; Chelba & Jelinek, 2000). More recently, Yogatama et al. (2018) implicitly learned structure by using a stack-like memory. While they did not perform analysis on its ability to induce a parse tree, the authors perform the Linzen et al. (2016) test on their model. Shen et al. (2017) introduced the Parsing-Reading-Predict Networks (PRPN) model, which attempts to perform parsing with only a language modelling signal. The model uses self-attention to compose previous states. They introduced a new value, syntactic distance, to control the range of attention. This value is then found to correspond to the depth of the parse tree. However, the added complexity in using the PRPN model makes it unwieldy in practice.

# 3 ORDERED NEURONS

![](images/5a5ff16d6872b80e8003a42fecdf8e735203b33640327c817193f266583b7e7a.jpg)  
(a) Constituency tree

![](images/38a9b63b0ada99a3a8727da4ce6300ed7c3250def2cdb13a8776bac62a9dcf6e.jpg)  
(b) Block view  
Figure 1: The relationship between a constituency parse tree and an ON-LSTM. Given a sequence of tokens  $(x_{1}, x_{2}, x_{3})$ , their constituency-based parse tree is illustrated in (a). (b) provides a block view of the tree structure, where S and VP node strides across more then one time step. The representation for high-ranking nodes should be relatively consistent across multiple time steps. (c) visualization of the ratio of updated neurons for each group of neurons at each time step. At each time step, given the input word, darker grey blocks are completely updated, lighter grey blocks are partially updated. The three groups of neurons have different update frequencies. Higher groups update less frequently and lower groups update more frequently.

![](images/7d703bf7ff73f8406784cbefb91071a2a237d2c2ac0f4f4abfcf8adce20857f9.jpg)  
(c) ON-LSTM cell states

Given a sequence of tokens  $x_{1}, \ldots, x_{T}$  governed by a latent tree structure as shown in Figure 1(a), our goal is to infer the unobserved structure from observed tokens and compute a hidden state  $h_{t}$  for each time step  $t$ . One ideal interpretation for  $h_{t}$  is that it represents all nodes on the path between current leaf node  $x_{t}$  to the root node S. As shown in Figure 1(c),  $h_{t}$  contains representations for all constituents that include the current token  $x_{t}$ , even when the respective constituent is only partially observed. We can also further assume that different nodes are represented by different chunks of adjacent neurons in the hidden states. However, while the dimension of hidden states is fixed, the numbers of nodes on the path are different across different time steps and sentences. Thus, allowing the model to actively allocate different numbers of neurons to each node would allow more flexibility.

In our model, high-ranking nodes contain long-term/global information that will last anywhere from several time steps to the entire sentence, while low-ranking nodes contain only short-term/local information that only last one or a few time steps, as shown in Figure 1(b). It is also therefore important to allow the model to actively control the updating frequency of neurons to differentiate long/short-term information.

Given these requirements, we introduce a new inductive bias: ordered neurons to enable dynamic allocation of neurons to represent different time-scale dependencies by controlling the update frequency of neurons. The ordered neurons make the assumption that:

- A order should exist between neurons: the high-ranking neurons store long-term information, while the low-ranking neurons store short-term information. To erase (or update) high-ranking neurons, the model should first erase (or update) all lower-ranking neurons.  
- This ordering is independent of the data, thus we can enforce it on hidden states as an inductive bias.

In other words, some neurons always update more (or less) frequently than the others, and that order is pre-determined as part of the model architecture.

# 4 ON-LSTM

In this section, we introduce a new RNN unit ON-LSTM, as an implementation of ordered neurons. The new model shares a similar architecture with the standard LSTM model:

$$
f _ {t} = \sigma \left(W _ {f} x _ {t} + U _ {f} h _ {t - 1} + b _ {f}\right) \tag {1}
$$

$$
i _ {t} = \sigma \left(W _ {i} x _ {t} + U _ {i} h _ {t - 1} + b _ {i}\right) \tag {2}
$$

$$
o _ {t} = \sigma \left(W _ {o} x _ {t} + U _ {o} h _ {t - 1} + b _ {o}\right) \tag {3}
$$

$$
\hat {c} _ {t} = \tanh  \left(W _ {c} x _ {t} + U _ {c} h _ {t - 1} + b _ {c}\right) \tag {4}
$$

$$
h _ {t} = o _ {t} \circ \tanh  \left(c _ {t}\right) \tag {5}
$$

The only difference with the standard LSTM is that we exclude the update function for cell state  $c_{t}$  and replace it with a new update rule that will be explained in the following sections. The forget gates  $f_{t}$  and input gates  $i_{t}$  are used to control the erasing and writing operation on cell states  $c_{t}$ , as before. Since the gates in the standard LSTM do not impose a topology on the individual units in the gates, in general, the behavior of the individual cells does not reflect an ordering.

# 4.1 ACTIVATION FUNCTION: cmax()

To enforce an order to the update frequency, we introduce a new activation function:

$$
\hat {g} = \operatorname {c u m a x} (\dots) = \operatorname {c u m s u m} (\operatorname {s o f t m a x} (\dots)) \tag {6}
$$

The vector  $\hat{g}$  can be seen as the expectation of a binary gate  $g = (0,\dots,0,1,\dots,1)$ . This binary gate split the cell state into two segments: the 0-segment and the 1-segment. Thus, the model can apply different update rules on the two segments to differentiate long/short-term information. The index for the first 1 in  $g$  is parametrised as:

$$
p (d) = \operatorname {s o f t m a x} (\dots) \tag {7}
$$

This discrete variable  $d$  represents the split point between the two segments. We can further compute the probability of the  $k$ -th value being 1, by evaluating the probability of the disjunction of any of the values before the  $k$ -th being the split point:  $d \leq k = (d = 0) \lor (d = 1) \lor \dots \lor (d = k)$ . Since the categories are mutually exclusive, we can do this by computing the cumulative distribution function,

$$
p \left(g _ {k} = 1\right) = p (d \leq k) = \sum_ {i \leq k} p (d = i) \tag {8}
$$

Ideally,  $g$  should take the form of discrete values. Unfortunately, computing gradient through a discrete value is not trivial, so in practice we use a relaxation in the form of computing the quantity  $p(d \leq k)$  by computing a cumulative sum of the softmax. As  $g_{k}$  is binary, this is equivalent to computing  $\mathbb{E}[g_k]$ . Hence,  $\hat{g} = \mathbb{E}[g]$ .

# 4.2 STRUCTURED GATING MECHANISM

Based on the  $\mathrm{cumax}()$  function, we introduce a master forget gate  $\tilde{f}_t$  and a master input gate  $\tilde{i}_t$ :

$$
\tilde {f} _ {t} = \operatorname {c u m a x} \left(W _ {\tilde {f}} x _ {t} + U _ {\tilde {f}} h _ {t - 1} + b _ {\tilde {f}}\right) \tag {9}
$$

$$
\tilde {i} _ {t} = 1 - \operatorname {c u m a x} \left(W _ {\bar {i}} x _ {t} + U _ {\bar {i}} h _ {t - 1} + b _ {\bar {i}}\right) \tag {10}
$$

where the values in master forget gate are constrained to monotonously increase from 0 to 1, and those in master input gate monotonously decrease from 1 to 0. These gates serve as a high-level control unit for the update operations of cell states. Using the master gates, we define a new update rule,

$$
\omega_ {t} = \tilde {f} _ {t} \circ \tilde {i} _ {t} \tag {11}
$$

$$
\hat {f} _ {t} = f _ {t} \circ \omega_ {t} + \left(\tilde {f} _ {t} - \omega_ {t}\right) = \tilde {f} _ {t} \circ \left(f _ {t} \circ \tilde {i} _ {t} + 1 - \tilde {i} _ {t}\right) \tag {12}
$$

$$
\hat {i} _ {t} = i _ {t} \circ \omega_ {t} + \left(\tilde {i} _ {t} - \omega_ {t}\right) = \tilde {i} _ {t} \circ \left(i _ {t} \circ \tilde {f} _ {t} + 1 - \tilde {f} _ {t}\right) \tag {13}
$$

$$
c _ {t} = \hat {f} _ {t} \circ c _ {t - 1} + \hat {i} _ {t} \circ \hat {c} _ {t} \tag {14}
$$

To explain the intuition behind the new update rule, we make the assumption that the master gates are binary.

- The master forget gate  $\tilde{f}_t$  controls the erasing behavior of the model. Suppose  $\tilde{f}_t = (0, \dots, 0, 1, \dots, 1)$  and the split point is  $d_t^f$ . Given the Eq. (12) and (14), the information stored in the first  $d_t^f$  neurons of the previous cell states  $c_{t-1}$  will be completely erased. Assuming that the model learned the constituency parse as pictured in Figure 1(c), this has the effect of completing previous constituents. A large number of zeroed neurons, i.e., a large  $d_t^f$ , represents the end of a high-level constituent in a constituent-based parse tree, as most of the information will be discarded. Conversely, a small  $d_t^f$  conveys the end of a low-level constituent as high-level information is kept for further processing.  
- The master input gate  $\tilde{i}_t$  is meant to control the writing behavior of model. Suppose  $\tilde{i}_t = (1, \dots, 1, 0, \dots, 0)$  and the split point is  $d_t^i$ . Given Eq. (13) and (14), a large  $d_t^i$  means that the current input  $x_t$  contains long-term information that needs to be preserved for several time steps. Conversely, a small  $d_t^i$  means that the current input  $x_t$  just provides local information that could be erased by  $\tilde{f}_t$  in the next few time steps.  
- The product of two master gates  $\omega_{t}$  represents the overlap of  $\tilde{f}_{t}$  and  $\tilde{i}_{t}$ . When the overlap exists ( $\exists k, \omega_{tk} > 0$ ), the segment is further controlled by the  $f_{t}$  and  $i_{t}$  in standard LSTM model to enable more fine-grained operations. This segment of neurons is related to the incomplete constituents that contain some previous words and the current input word  $x_{t}$ . For example, in figure 1, the word  $x_{3}$  belongs to the constituents  $S$  and  $VP$ . At this time step, the overlap  $\omega_{3}$  would cover the related blocks of neurons, such that these neurons could be partial updated.

As the master gates only focus on coarse-grained control, modeling them with the same dimensions as the hidden states is computationally expensive and unnecessary. In practice, we parameterize  $\tilde{f}_t$  and  $\dot{i}_t$  to be  $D_m = \frac{D}{C}$  dimension vectors, where  $D$  is the dimension of hidden state, and  $C$  is a chunk size factor. We repeat each dimension  $C$  times, before the element-wise multiplication with  $f_t$  and  $\dot{i}_t$ . The downsizing significantly reduces the number of extra parameters that we add to standard LSTM. This behavior means that every unit within each  $C$ -sized chunk receives the same gating behavior from the master gates.

# 5 EXPERIMENTS

We evaluate the proposed model on four tasks: language modeling, unsupervised constituency parsing, targeted syntactic evaluation (Marvin & Linzen, 2018), and logical inference (Bowman et al., 2015).

# 5.1 LANGUAGE MODELING

Word-level language modeling is a macroscopic evaluation of the model's ability to deal with various linguistic phenomena (e.g. co-occurrence, syntactic structure, verb-subject agreement, etc). We evaluate our model by measuring perplexity on the Penn TreeBank (PTB) (Marcus et al., 1993; Mikolov, 2012) task.

For fair comparison, we closely follow the model hyper-parameters, regularization and optimization techniques introduced in AWD-LSTM (Merit et al., 2017). Our model uses a three-layer ON-LSTM model with 1150 units in the hidden layer and an embedding of size 400. For master gates, the downsize factor  $C = 10$ . The total number of parameters was slightly increased from 24 millions to 25 millions with additional matrices for computing master gates. We manually searched some of the dropout values for ON-LSTM based on the validation performance. The values used for dropout on the word vectors, the output between LSTM layers, the output of the final LSTM layer, and embedding dropout where (0.5, 0.3, 0.45, 0.1) respectively. A weight-dropout of 0.45 was applied to the recurrent weight matrices.

As shown in table 1, our model performs better than the standard LSTM while sharing the same number of layers, embedding dimensions, and hidden states units. Recall that the master gates only

<table><tr><td>Model</td><td>Parameters</td><td>Validation</td><td>Test</td></tr><tr><td>Zaremba et al. (2014) - LSTM (large)</td><td>66M</td><td>82.2</td><td>78.4</td></tr><tr><td>Gal &amp; Ghahramani (2016) - Variational LSTM (large, MC)</td><td>66M</td><td>-</td><td>73.4</td></tr><tr><td>Kim et al. (2016) - CharCNN</td><td>19M</td><td>-</td><td>78.9</td></tr><tr><td>Merit et al. (2016) - Pointer Sentinel-LSTM</td><td>21M</td><td>72.4</td><td>70.9</td></tr><tr><td>Grave et al. (2016) - LSTM</td><td>-</td><td>-</td><td>82.3</td></tr><tr><td>Grave et al. (2016) - LSTM + continuous cache pointer</td><td>-</td><td>-</td><td>72.1</td></tr><tr><td>Inan et al. (2016) - Variational LSTM (tied) + augmented loss</td><td>51M</td><td>71.1</td><td>68.5</td></tr><tr><td>Zilly et al. (2016) - Variational RHN (tied)</td><td>23M</td><td>67.9</td><td>65.4</td></tr><tr><td>Zoph &amp; Le (2016) - NAS Cell (tied)</td><td>54M</td><td>-</td><td>62.4</td></tr><tr><td>Shen et al. (2017) - PRPN-LM</td><td>-</td><td>-</td><td>62.0</td></tr><tr><td>Melis et al. (2017) - 4-layer skip connection LSTM (tied)</td><td>24M</td><td>60.9</td><td>58.3</td></tr><tr><td>Merit et al. (2017) - AWD-LSTM - 3-layer LSTM (tied)</td><td>24M</td><td>60.0</td><td>57.3</td></tr><tr><td>ON-LSTM - 3-layer (tied)</td><td>25M</td><td>58.29 ± 0.10</td><td>56.17 ± 0.12</td></tr><tr><td>Yang et al. (2017) - AWD-LSTM-MoS*</td><td>22M</td><td>56.5</td><td>54.4</td></tr></table>

Table 1: Single model perplexity on validation and test sets for the Penn Treebank language modeling task. Models noting tied use weight tying on the embedding and softmax weights. Model noting * focus on improving the softmax component of RNN language model. Their contribution is orthogonal to ours.

controls how information is stored in different neurons. Therefore, it is interesting to note that we can improve the performance of RNN model without skip connections or a significant increase in the number of parameters.

# 5.2 UNSUPERVISED CONSTITUENCY PARSING

The unsupervised constituency parsing task compares the latent stree structure induced by the model with those annotated by human experts. Following the experiment settings proposed in Htut et al. (2018), we take our best model for the language modeling task, and test it on WSJ10 dataset and WSJ test set. WSJ10 has 7422 sentences, filtered from the WSJ dataset with the constraint of 10 words or less, after the removal of punctuation and null elements (Klein & Manning, 2002). The WSJ test set contains 2416 sentences with various lengths. It is worth noting that the WSJ10 test set contains sentences from the training, validation, and test set of the PTB dataset, while WSJ test uses the same set of sentences as the PTB test set.

To generate a tree structure from the trained model and a sentence, we initialise the hidden states with 0, then feed the sentence into the model as in language modeling task. For each time step, we compute an estimation of  $d_t^f$ :

$$
\hat {d} _ {t} = \mathbb {E} \left[ d _ {t} ^ {f} \right] = \sum_ {k = 1} ^ {D _ {m}} k p \left(y _ {t} = k\right) = \sum_ {k = 1} ^ {D _ {m}} \sum_ {i = 1} ^ {k} p \left(y _ {t} = k\right) = D _ {m} - \sum_ {k = 1} ^ {D _ {m}} \tilde {f} _ {t k} \tag {15}
$$

Given  $\hat{d}_t$ , we can use the parsing algorithm proposed in Shen et al. (2017) for unsupervised constituency parsing.

The performance is shown in Table 2. The 2nd-layer of ON-LSTM model achieves state-of-the-art unsupervised constituency parsing results on the WSJ test set, while the 1st and 3rd layer of ON-LSTM do not perform as good. One possible interpretation is that only the first and last layers focus on fine-tuning the input and output vectors with local information, thus do not need to learn the entire tree structure. Since the WSJ test set contains sentence of various lengths which they are unobserved during training, we find that ON-LSTM provides better generalization and robustness toward longer sentences than previous models. We also see that ON-LSTM model can provide strong results for phrase detection, including ADJP (adjective phrases), PP (prepositional phrases), and NP (noun phrases). This feature could benefit many downstream tasks, like question-answering, named entity recognition, co-reference detection, etc.

# 5.3 TARGETED SYNTACTIC EVALUATION

Targeted syntactic evaluation is proposed in Marvin & Linzen (2018). The task evaluates language models along three different structure-sensitive linguistic phenomenon: subject-verb agreement, re

<table><tr><td rowspan="2">Model</td><td rowspan="2">Training Data</td><td rowspan="2">Training Object</td><td rowspan="2">Vocab Size</td><td colspan="4">Parsing F1</td><td rowspan="2">Depth WSJ</td><td rowspan="2">Accuracy on ADJP</td><td rowspan="2">WSJ NP</td><td rowspan="2">WSJ PP</td><td rowspan="2">by Tag INTJ</td></tr><tr><td>WSJ10 μ(σ)</td><td>max</td><td>WSJ μ(σ)</td><td>max</td></tr><tr><td>PRPN-UP</td><td>AllNLI Train</td><td>LM</td><td>76k</td><td>66.3 (0.8)</td><td>68.5</td><td>38.3 (0.5)</td><td>39.8</td><td>5.8</td><td>28.7</td><td>65.5</td><td>32.7</td><td>0.0</td></tr><tr><td>PRPN-LM</td><td>AllNLI Train</td><td>LM</td><td>76k</td><td>52.4 (4.9)</td><td>58.1</td><td>35.0 (5.4)</td><td>42.8</td><td>6.1</td><td>37.8</td><td>59.7</td><td>61.5</td><td>100.0</td></tr><tr><td>PRPN-UP</td><td>WSJ Train</td><td>LM</td><td>15.8k</td><td>62.2 (3.9)</td><td>70.3</td><td>26.0 (2.3)</td><td>32.8</td><td>5.8</td><td>24.8</td><td>54.4</td><td>17.8</td><td>0.0</td></tr><tr><td>PRPN-LM</td><td>WSJ Train</td><td>LM</td><td>10k</td><td>70.5 (0.4)</td><td>71.3</td><td>37.4 (0.3)</td><td>38.1</td><td>5.9</td><td>26.2</td><td>63.9</td><td>24.4</td><td>0.0</td></tr><tr><td>ON-LSTM 1st-layer</td><td>WSJ Train</td><td>LM</td><td>10k</td><td>35.2(4.1)</td><td>42.8</td><td>20.0(2.8)</td><td>24.0</td><td>5.6</td><td>38.1</td><td>23.8</td><td>18.3</td><td>100.0</td></tr><tr><td>ON-LSTM 2nd-layer</td><td>WSJ Train</td><td>LM</td><td>10k</td><td>65.1(1.7)</td><td>66.8</td><td>47.7(1.5)</td><td>49.4</td><td>5.6</td><td>46.2</td><td>61.4</td><td>55.4</td><td>0.0</td></tr><tr><td>ON-LSTM 3rd-layer</td><td>WSJ Train</td><td>LM</td><td>10k</td><td>54.0(3.9)</td><td>57.6</td><td>36.6(3.3)</td><td>40.4</td><td>5.3</td><td>44.8</td><td>57.5</td><td>47.2</td><td>0.0</td></tr><tr><td>300D ST-Gumbel</td><td>AllNLI Train</td><td>NLI</td><td>-</td><td>-</td><td>-</td><td>19.0 (1.0)</td><td>20.1</td><td>-</td><td>15.6</td><td>18.8</td><td>9.9</td><td>59.4</td></tr><tr><td>w/o Leaf GRU</td><td>AllNLI Train</td><td>NLI</td><td>-</td><td>-</td><td>-</td><td>22.8 (1.6)</td><td>25.0</td><td>-</td><td>18.9</td><td>24.1</td><td>14.2</td><td>51.8</td></tr><tr><td>300D RL-SPINN</td><td>AllNLI Train</td><td>NLI</td><td>-</td><td>-</td><td>-</td><td>13.2 (0.0)</td><td>13.2</td><td>-</td><td>1.7</td><td>10.8</td><td>4.6</td><td>50.6</td></tr><tr><td>w/o Leaf GRU</td><td>AllNLI Train</td><td>NLI</td><td>-</td><td>-</td><td>-</td><td>13.1 (0.1)</td><td>13.2</td><td>-</td><td>1.6</td><td>10.9</td><td>4.6</td><td>50.0</td></tr><tr><td>CCM</td><td>WSJ10 Full</td><td>-</td><td>-</td><td>-</td><td>71.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DMV+CCM</td><td>WSJ10 Full</td><td>-</td><td>-</td><td>-</td><td>77.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UML-DOP</td><td>WSJ10 Full</td><td>-</td><td>-</td><td>-</td><td>82.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Random Trees</td><td>-</td><td>-</td><td>-</td><td>-</td><td>34.7</td><td>21.3 (0.0)</td><td>21.4</td><td>5.3</td><td>17.4</td><td>22.3</td><td>16.0</td><td>40.4</td></tr><tr><td>Balanced Trees</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>21.3 (0.0)</td><td>21.3</td><td>4.6</td><td>22.1</td><td>20.2</td><td>9.3</td><td>55.9</td></tr><tr><td>Left Branching</td><td>-</td><td>-</td><td>-</td><td>28.7</td><td>28.7</td><td>13.1 (0.0)</td><td>13.1</td><td>12.4</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Right Branching</td><td>-</td><td>-</td><td>-</td><td>61.7</td><td>61.7</td><td>16.5 (0.0)</td><td>16.5</td><td>12.4</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

Table 2: Unlabeled parsing F1 results evaluated on full WSJ10 and WSJ test set. Our language model has three layers, each of them provides a sequence of  $\hat{d}_t$ . We provide the parsing performance for all layers. Results with RL-SPINN and ST-Gumbel are evaluated on the full WSJ (Williams et al., 2017). PRPN models are evaluated on WSJ test set (Htut et al., 2018). We run the model with 5 different random seeds to calculate the average F1. The Accuracy columns represent the fraction of ground truth constituents of a given type that corresponds to constituents in the model parses. We use the model with the best F1 score to report ADJP, NP, PP, and INTJ. WSJ10 baselines are from Klein & Manning (2002, CCM), Klein & Manning (2005, DMV+CCM), and Bod (2006, UML-DOP). As the WSJ10 baselines are trained using additional information such as POS tags and dependency parser, they are not strictly comparable with the latent tree learning results. Italics mark results that are worse than the random baseline.

flexive anaphora and negative polarity items. Given a large number of minimally different pairs of English sentences, each consisting of a grammatical and an ungrammatical sentence, a language model should assign a higher probability to a grammatical sentence than an ungrammatical one.

Using the released codebase<sup>1</sup> and the same settings proposed in Marvin & Linzen (2018), we train both the ON-LSTM and LSTM language models on a 90 million word subset of Wikipedia. The RNN LMs has two layers of 650 units, a batch size of 128, a dropout rate of 0.2, a learning rate of 20.0, and was trained for 40 epochs. The input embedding was 200 dimensions and the output embedding was 650 dimensions.

Table 3 shows that ON-LSTM perform better on long-term dependency cases, while LSTM is better on short-term ones. This is possibly due to the relatively small number of units in the hidden states, which is insufficient to take into account both long and short-term information. We also notice that the results for NPI test cases have unusually high variance across different hyper-parameters. This result maybe due to the non-syntactic cues discussed in Marvin & Linzen (2018). Despite this, ON-LSTM actually achieves better perplexity on the validation.

# 5.4 LOGICAL INFERENCE

We also analyze the model's performance on the logical inference task described in Bowman et al. (2015). This task is based on a language that has a vocabulary of six words and three logical operations, or, and, not. There are seven mutually exclusive logical relations that describe the relationship between two sentences: two types of entailment, equivalence, exhaustive and non-exhaustive con

<table><tr><td></td><td>ON-LSTM</td><td>LSTM</td></tr><tr><td colspan="3">Short-Term Dependency</td></tr><tr><td colspan="3">SUBJECT-VERB AGREEMENT:</td></tr><tr><td>Simple</td><td>0.99</td><td>1.00</td></tr><tr><td>In a sentential complement</td><td>0.95</td><td>0.98</td></tr><tr><td>Short VP coordination</td><td>0.89</td><td>0.92</td></tr><tr><td>In an object relative clause</td><td>0.84</td><td>0.88</td></tr><tr><td>In an object relative (no that)</td><td>0.78</td><td>0.81</td></tr><tr><td colspan="3">REFLEXIVE ANAPHORA:</td></tr><tr><td>Simple</td><td>0.89</td><td>0.82</td></tr><tr><td>In a sentential complement</td><td>0.86</td><td>0.80</td></tr><tr><td colspan="3">NEGATIVE POLARITY ITEMS:</td></tr><tr><td>Simple (grammatical vs. intrusive)</td><td>0.18</td><td>1.00</td></tr><tr><td>Simple (intrusive vs. ungrammatical)</td><td>0.50</td><td>0.01</td></tr><tr><td>Simple (grammatical vs. ungrammatical)</td><td>0.07</td><td>0.63</td></tr><tr><td colspan="3">Long-Term Dependency</td></tr><tr><td colspan="3">SUBJECT-VERB AGREEMENT:</td></tr><tr><td>Long VP coordination</td><td>0.74</td><td>0.74</td></tr><tr><td>Across a prepositional phrase</td><td>0.67</td><td>0.68</td></tr><tr><td>Across a subject relative clause</td><td>0.66</td><td>0.60</td></tr><tr><td>Across an object relative clause</td><td>0.57</td><td>0.52</td></tr><tr><td>Across an object relative (no that)</td><td>0.54</td><td>0.51</td></tr><tr><td colspan="3">REFLEXIVE ANAPHORA:</td></tr><tr><td>Across a relative clause</td><td>0.57</td><td>0.58</td></tr><tr><td colspan="3">NEGATIVE POLARITY ITEMS:</td></tr><tr><td>Across a relative clause (grammatical vs. intrusive)</td><td>0.59</td><td>0.95</td></tr><tr><td>Across a relative clause (intrusive vs. ungrammatical)</td><td>0.20</td><td>0.00</td></tr><tr><td>Across a relative clause (grammatical vs. ungrammatical)</td><td>0.11</td><td>0.04</td></tr></table>

Table 3: Overall accuracy for the ON-LSTM and LSTM on each test case. "Long-term dependency" means that an unrelated phrase (or a clause) exist between the targeted pair of words, while "short-term dependency" means there is no such distraction.

tradiction, and two types of semantic independence. Similar to the natural language inference task, this logical inference task requires the model to predict the correct label for given pair of sentences. The train/test split is as described in the original codebase $^2$ , and  $10\%$  of training set is set aside as the validation set.

![](images/0067cb591064c48ef55722ebd3260cc7a6e1a9fd96c5a4bf6c22a5b63581d137.jpg)  
Figure 2: Test accuracy of the models, trained on short sequences  $(\leq 6)$  in logic data. The horizontal axis indicates the length of the sequence, and the vertical axis indicates the accuracy of models performance on the corresponding test set.

We evaluate the ON-LSTM and the standard LSTM on this dataset. Given a pair of sentences  $(s_1, s_2)$ , we feed both sentences into an RNN encoder, taking the last hidden state  $(h_1, h_2)$  as the sentence embedding. The concatenation of  $(h_1, h_2, h_1 \circ h_2, \mathrm{abs}(h_1 - h_2))$  is used as input to a multi-layer classifier, which gives a probability distribution over seven labels. In our experiment, the RNN models were parameterised with 400 units in one hidden layer, and the input embedding size was 128. A dropout of 0.2 was applied between different layers. Both models are trained on sequences with 6 or less logical operations and tested on sequences with at most 12 operations.

Figure 2 shows the performance of ON-LSTM and standard LSTM on the logical inference task. While both models achieve nearly  $100\%$  accuracy on short sequences  $(\leq 3)$ , ON-LSTM attains better performance on sequences longer

then 3. The performance gap continues to increase on longer sequences  $(\geq 7)$  that were not present during training. Hence, the ON-LSTM model shows better generalization while facing structured data with various lengths and comparing to the standard LSTM. However, a recursive neural network model can achieve stronger performance on this dataset (Bowman et al., 2015), since they have structure information as input. We also include the result of RRNet from Jacob et al. (2018), which can induce the latent tree structure from downstream tasks. However, the results may not be comparable, because the hyper-parameters for training were not provided. The repetitive composition using the same function is better suited for this synthetic task.

# 6 CONCLUSION

In this paper, we propose the ordered neuron inductive bias. This unifies modelling tree structures and RNNs, through separately allocating hidden state neurons with long and short-term information. Based on this idea, we propose a new RNN unit, the ON-LSTM, which includes a new gating mechanism and a new activation function  $\text{cumax}(\cdot)$ . The model's results on unsupervised constituency parsing result shows that the ON-LSTM induces the latent structure of natural language in a way that is coherent with human expert annotation. The inductive bias also enables ON-LSTM to achieve good performance on language modeling, long-term dependency, and logical inference tasks.

# REFERENCES

David Alvarez-Melis and Tommi S Jaakkola. Tree-structured decoding with doubly-recurrent neural networks. 2016.  
Yoshua Bengio et al. Learning deep architectures for ai. Foundations and trends® in Machine Learning, 2(1):1-127, 2009.  
Rens Bod. An all-subtrees approach to unsupervised parsing. In Proceedings of the 21st International Conference on Computational Linguistics and the 44th annual meeting of the Association for Computational Linguistics, pp. 865-872. Association for Computational Linguistics, 2006.  
Samuel R Bowman, Christopher Potts, and Christopher D Manning. Recursive neural networks can learn logical semantics. arXiv preprint arXiv:1406.1827, 2014.  
Samuel R Bowman, Christopher D Manning, and Christopher Potts. Tree-structured composition in neural networks without tree-structured architectures. arXiv preprint arXiv:1506.04834, 2015.  
Samuel R Bowman, Jon Gauthier, Abhinav Rastogi, Raghav Gupta, Christopher D Manning, and Christopher Potts. A fast unified model for parsing and sentence understanding. arXiv preprint arXiv:1603.06021, 2016.  
Eugene Charniak. Immediate-head parsing for language models. In Proceedings of the 39th Annual Meeting on Association for Computational Linguistics, pp. 124-131. Association for Computational Linguistics, 2001.  
Ciprian Chelba and Frederick Jelinek. Structured language modeling. Computer Speech & Language, 14(4):283-332, 2000.  
Jihun Choi, Kang Min Yoo, and Sang-goo Lee. Learning to compose task-specific tree structures. In Proceedings of the 2018 Association for the Advancement of Artificial Intelligence (AAAI). and the 7th International Joint Conference on Natural Language Processing (ACL-IJCNLP), 2018.  
Noam Chomsky and David W Lightfoot. Syntactic structures. Walter de Gruyter, 2002.  
Junyoung Chung, Sungjin Ahn, and Yoshua Bengio. Hierarchical multiscale recurrent neural networks. arXiv preprint arXiv:1609.01704, 2016.  
Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A Smith. Recurrent neural network grammars. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 199-209, 2016.

Salah El Hihi and Yoshua Bengio. Hierarchical recurrent neural networks for long-term dependencies. In Advances in neural information processing systems, pp. 493-499, 1996.  
Yarin Gal and Zoubin Ghahramani. A theoretically grounded application of dropout in recurrent neural networks. In Advances in neural information processing systems, pp. 1019-1027, 2016.  
Felix A Gers and E Schmidhuber. Lstm recurrent networks learn simple context-free and context-sensitive languages. IEEE Transactions on Neural Networks, 12(6):1333-1340, 2001.  
Edouard Grave, Armand Joulin, and Nicolas Usunier. Improving neural language models with a continuous cache. arXiv preprint arXiv:1612.04426, 2016.  
Phu Mon Htut, Kyunghyun Cho, and Samuel R Bowman. Grammar induction with neural language models: An unusual replication. arXiv preprint arXiv:1808.10000, 2018.  
Hakan Inan, Khashayar Khosravi, and Richard Socher. Tying word vectors and word classifiers: A loss framework for language modeling. arXiv preprint arXiv:1611.01462, 2016.  
Athul Paul Jacob, Zhouhan Lin, Alessandro Sordoni, and Yoshua Bengio. Learning hierarchical structures on-the-fly with a recurrent-recursive model for sequences. In Proceedings of The Third Workshop on Representation Learning for NLP, pp. 154–158, 2018.  
Yoon Kim, Yacine Jernite, David Sontag, and Alexander M Rush. Character-aware neural language models. In AAAI, pp. 2741-2749, 2016.  
Dan Klein and Christopher D Manning. A generative constituent-context model for improved grammar induction. In Proceedings of the 40th Annual Meeting on Association for Computational Linguistics, pp. 128-135. Association for Computational Linguistics, 2002.  
Dan Klein and Christopher D Manning. Accurate unlexicalized parsing. In Proceedings of the 41st Annual Meeting on Association for Computational Linguistics-Volume 1, pp. 423-430. Association for Computational Linguistics, 2003.  
Dan Klein and Christopher D Manning. Natural language grammar induction with a generative constituent-context model. Pattern recognition, 38(9):1407-1419, 2005.  
Jan Koutnik, Klaus Greff, Faustino Gomez, and Juergen Schmidhuber. A clockwork rnn. arXiv preprint arXiv:1402.3511, 2014.  
Adhiguna Kuncoro, Chris Dyer, John Hale, Dani Yogatama, Stephen Clark, and Phil Blunsom. Lstms can learn syntax-sensitive dependencies well, but modeling structure makes them better. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), volume 1, pp. 1426-1436, 2018.  
Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436-444, 2015.  
Tsungnan Lin, Bill G Horne, Peter Tino, and C Lee Giles. Learning long-term dependencies is not as difficult with narx recurrent neural networks. Technical report, 1998.  
Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. Assessing the ability of lstms to learn syntax-sensitive dependencies. arXiv preprint arXiv:1611.01368, 2016.  
Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313-330, 1993.  
Rebecca Marvin and Tal Linzen. Targeted syntactic evaluation of language models. arXiv preprint arXiv:1808.09031, 2018.  
Gábor Melis, Chris Dyer, and Phil Blunsom. On the state of the art of evaluation in neural language models. arXiv preprint arXiv:1707.05589, 2017.  
Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.

Stephen Merity, Nitish Shirish Keskar, and Richard Socher. Regularizing and Optimizing LSTM Language Models. arXiv preprint arXiv:1708.02182, 2017.  
Tomáš Mikolov. Statistical language models based on neural networks. Presentation at Google, Mountain View, 2nd April, 2012.  
Brian Roark. Probabilistic top-down parsing and language modeling. Computational linguistics, 27 (2):249-276, 2001.  
Dominiek Sandra and Marcus Taft. Morphological Structure, Lexical Representation and Lexical Access (RLE Linguistics C: Applied Linguistics): A Special Issue of Language and Cognitive Processes. Routledge, 2014.  
Jürgen Schmidhuber. Neural sequence chunkers. 1991.  
Jürgen Schmidhuber. Deep learning in neural networks: An overview. Neural networks, 61:85-117, 2015.  
Yikang Shen, Zhouhan Lin, Chin-Wei Huang, and Aaron Courville. Neural language modeling by jointly learning syntax and lexicon. arXiv preprint arXiv:1711.02013, 2017.  
Haoyue Shi, Hao Zhou, Jiaze Chen, and Lei Li. On tree-based neural sentence modeling. arXiv preprint arXiv:1808.09644, 2018.  
Richard Socher, Christopher D Manning, and Andrew Y Ng. Learning continuous phrase representations and syntactic parsing with recursive neural networks. In Proceedings of the NIPS-2010 Deep Learning and Unsupervised Feature Learning Workshop, volume 2010, pp. 1-9, 2010.  
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pp. 1631-1642, 2013.  
Kai Sheng Tai, Richard Socher, and Christopher D Manning. Improved semantic representations from tree-structured long short-term memory networks. arXiv preprint arXiv:1503.00075, 2015.  
Adina Williams, Nikita Nangia, and Samuel R Bowman. A broad-coverage challenge corpus for sentence understanding through inference. arXiv preprint arXiv:1704.05426, 2017.  
Adina Williams, Andrew Drozdov*, and Samuel R Bowman. Do latent tree learning models identify meaningful structure in sentences? Transactions of the Association of Computational Linguistics, 6:253-267, 2018.  
Shuangzhi Wu, Dongdong Zhang, Nan Yang, Mu Li, and Ming Zhou. Sequence-to-dependency neural machine translation. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), volume 1, pp. 698-707, 2017.  
Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, and William W Cohen. Breaking the softmax bottleneck: A high-rank rnn language model. arXiv preprint arXiv:1711.03953, 2017.  
Dani Yogatama, Phil Blunsom, Chris Dyer, Edward Grefenstette, and Wang Ling. Learning to compose words into sentences with reinforcement learning. arXiv preprint arXiv:1611.09100, 2016.  
Dani Yogatama, Yishu Miao, Gabor Melis, Wang Ling, Adhiguna Kuncoro, Chris Dyer, and Phil Blunsom. Memory architectures in recurrent neural network language models. 2018.  
Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. Recurrent neural network regularization. arXiv preprint arXiv:1409.2329, 2014.  
Xingxing Zhang, Liang Lu, and Mirella Lapata. Top-down tree long short-term memory networks. arXiv preprint arXiv:1511.00060, 2015.  
Ganbin Zhou, Ping Luo, Rongyu Cao, Yijun Xiao, Fen Lin, Bo Chen, and Qing He. Generative neural machine for tree structures. CoRR, 2017.

Julian Georg Zilly, Rupesh Kumar Srivastava, Jan Koutnik, and Jürgen Schmidhuber. Recurrent highway networks. arXiv preprint arXiv:1607.03474, 2016.  
Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. arXiv preprint arXiv:1611.01578, 2016.
