# TREE-STRUCTURED DECODING WITH DOUBLY-RECURRENT NEURAL NETWORKS

David Alvarez-Melis & Tommi S. Jaakkola

Computer Science and Artificial Intelligence Lab MIT

{davidam, tommi}@csail.mit.edu

# ABSTRACT

We propose a neural network architecture for generating tree-structured objects from encoded representations. The core of the method is a doubly recurrent neural network that models separately the width and depth recurrences across the tree, and combines them inside each cell to generate an output. The topology of the tree is explicitly modeled, allowing the network to predict both content and topology of the tree when decoding. That is, given only an encoded vector representation, the network is able to simultaneously generate a tree from it and predict labels for the nodes. We test this architecture in an encoder-decoder framework, where we train a network to encode a sentence as a vector, and then generate a tree structure from it. The experimental results show the effectiveness of this architecture at recovering latent tree structure in sequences and at mapping sentences to simple functional programs.

# 1 INTRODUCTION

Recurrent neural networks have become extremely popular for modeling structured data. Key to their success is their ability to learn long-range temporal dependencies, their flexibility, and ease of customization. These architectures are a natural choice for modeling sequences; the notion of time—consecutive, linear inputs—in intrinsic to their design (Williams & Zipser, 1995; Hochreiter & Schmidhuber, 1997). Indeed, it is for sequential data that these architectures have achieved greater success, with groundbreaking performance in language modeling (Zaremba et al., 2015), machine translation (Sutskever et al., 2014) and conversational agents (Vinyals & Le, 2015), among other applications.

Although sequences are easy to manipulate and arise frequently in practice, they are by no means the only kind of structured data. Certain objects are more naturally modeled as trees or graphs, such as natural language parses, programs and hierarchical structures in bioinformatics. Even in tasks where the data is apparently sequential, there might be an underlying hierarchical structure that guides the generation. Language, for example, is known to be compositional: the meaning of a sentence is composed from the individual meanings of its parts Frege (1892). Thus, methods that construct sentences compositionally have the potential of generating language more naturally and efficiently than their sequential counterparts.

The flexibility and success of recurrent neural networks in modeling and generating sequential data has prompted efforts to adapt them to non-sequential data. Recent work has pioneered the application of neural architectures to hierarchical structures, albeit in limited settings. Most of this work has assumed that either the full tree structure is given (Socher et al., 2012; Tai et al., 2015) or at least the nodes are (Socher & Lin, 2011; Chen & Manning, 2014; Kiperwasser & Goldberg, 2016). In the former scenario, the network must simply aggregate the node information in a manner that is coherent with the tree structure, while in the latter, generation essentially becomes an attachment problem: deciding which pairs of nodes to join with an edge until a tree is formed.

Full decoding with structure, that is, generating a tree-structured object from nothing more than a vector representation, is a much harder problem and was until very recently mostly unexplored. Clearly, the main challenge of this paradigm lies in the fact that both the topology and the content (node and/or edge labels) of the tree must be generated. That is, without knowing in advance its

size, node degree or depth, one must generate a tree that is both topologically feasible and whose predicted node labels are coherent. There have been efforts to adapt RNNs to this context too. In fact, most current approaches to structured decoding rely on extensions and variations of sequential architectures. Since there is no obvious way to de-sequentialize these, previous work has relied on artifacts to simulate depth, such as padding trees with special tokens (Dong & Lapata, 2016) or using alternating RNNs coupled with external classifiers to predict branching (Zhang et al., 2016).

In this work, we propose a novel architecture tailored specifically to tree-structured decoding. At the heart of our approach is a doubly-recurrent (breadth and depth-wise recurrent) neural network which separately models the flow of information between parent and children nodes, and between siblings. Each of these relationships is modeled with a recurrent module whose hidden states are updated upon observing node labels. Every node in the tree receives two hidden states, which are then combined and used to predict a label for that node. Besides maintaining separate but simultaneous fraternal and paternal recurrences, the proposed architecture departs from previous methods in that it explicitly models tree topology. Each node in the network has modules that predict, based on the cell state, whether the node is terminal, both in terms of depth and width. Decoupling these decisions from the label prediction allows for a more concise formulation, which does not require artificial tokens to be added to the tree to simulate branching.

We test this novel architecture in an encoder-decoder framework, where we couple it with sequential encoders to predict tree structure from encoded vector representations of sequences. The experimental results show the effectiveness of this approach at recovering latent structure in flattened string representations of trees (Section 4.1) and at mapping from natural language descriptions of simple programs to a abstract syntax trees (Section 4.2).

To summarize, the main contributions of this paper are as follows:

- We propose a novel neural network architecture specifically tailored to tree-structured decoding, which maintains separate depth and width recurrences states and combines them to obtain hidden states for every node in the tree.  
- We equip this novel architecture with a mechanism to predict tree topology explicitly, as opposed to implicitly by adding nodes with special tokens.  
- We show experimentally that the proposed method is capable of recovering trees from encoded representations and that it outperforms state-of-the-art methods in a task consisting of mapping sentences to simple functional programs.

# 2 RELATED WORK

Recursive Neural Networks. Recursive neural networks (Socher & Lin, 2011; Socher et al., 2012) were proposed to model data with hierarchical structures, such as parsed scenes and natural language sentences. Though they have been most successfully applied to encoding objects when their tree-structured representation is given (Socher et al., 2013), their original formulation by Socher & Lin (2011) also showed how they can be used to predict the structure (edges), albeit for the case where nodes are given. Thus, besides their limited applicability due to their assumption of binary trees, recursive neural networks are not useful for fully generating trees from scratch.

Tree-structured encoders. The Tree-LSTM of Tai et al. (2015) is a generalization of long short-term memory networks (Hochreiter & Schmidhuber, 1997) to tree-structured inputs. Their model constructs a sentence representation bottom-up, obtaining at every step the representation of a node in the tree from the representations of its children. In this sense, this model can be seen as a generalization of recursive neural networks to trees with degree potentially greater than two, with the additional long-range dependency modeling provided by LSTMs. They propose two methods for aggregating the states of the children, depending on the type of underlying tree: N-ary trees or trees with unknown and potentially unbounded branching factor. For the former the maintain  $N$  different memory cells and hidden states, while for the latter they sum the children's states (after application of individual forget gates). TreeLSTMs have shown promising results for compositional encoding of structured data, though by construction they cannot be used for decoding, since they operate on a given tree structure.

Tree-structured decoders. Proposed only very recently, most tree-structured decoders rely on stacked on intertwined RNNs, and use heuristic methods for topological decisions during generation. Closest to our method is the Top-down Tree LSTM of Zhang et al. (2016), which generates a tree from an encoded representation. Their method relies on 4 independent LSTMs, which act in alternation—as opposed to simultaneously in our approach—yielding essentially a standard LSTM which changes the weights it uses based on the position of the current node. In addition, in their method children are faced with asymmetric parent input: “younger” children receive information from the parent state only through the previous sibling’s state. Though most of their experiments focus on the case where the nodes are given, they mention how to use their method for full prediction by introducing additional binary classifiers which predict which of their four LSTMs is to be used. These classifiers are trained in isolation after the main architecture has been trained. Contrary to this approach, our method can be trained end-to-end in only one pass, has a simpler formulation and explicitly incorporates topological prediction as part of the functioning of each neuron.

A similar approach is proposed by Dong & Lapata (2016). They propose SEQ2TREE, an encoder-decoder architecture that maps sentences to tree structures. For the decoder, they rely on hierarchical use of an LSTM, similar to Tai et al. (2015), but in the opposite direction: working top-down from the root of the tree. To decide when to change levels in the hierarchy, they augment the training trees with nonterminal nodes labeled with a special token  $< n >$ , which when generated during decoding trigger the branching out into a lower level in the tree. Similar to our method, they feed nodes with hidden representations of their parent and sibling, but they do so by concatenating both states and running them through a single recurrent unit, as opposed to our method, where these two sources of information are handled separately. A further difference is that our method does not require artificial nodes with special tokens to be added to the tree, which results in smaller trees.

Hierarchical Neural Networks for Parsing. Neural networks have also been recently introduced to the problem of natural language parsing (Chen & Manning, 2014; Kiperwasser & Goldberg, 2016). In this problem, the task is to predict a parse tree over a given sentence. For this, Kiperwasser & Goldberg (2016) use recurrent neural networks as a building block, and compose them recursively to obtain a tree-structured encoder. Starting from the leaves (words) they predict a parse tree with a projective bottom-up strategy, which sequentially updates the encoded vector representation of the tree and uses it to guide edge-attaching decisions. Though conceptually similar to our approach, their method relies on having access to the nodes of the tree (words) and only predicts its topology, so—similar to recursive neural networks—it cannot be used for a fully generative decoding.

# 3 DOUBLY RECURRENT NEURAL NETWORKS

Generating a tree-structured object from scratch using only an encoded representation poses several design challenges. First, one must decide in which order to generate the tree. If the nodes on the decoder side were given (such as in parsing), it would be possible to generate a tree bottom-up from these nodes (e.g. as Kiperwasser & Goldberg 2016 do). In the setting we are interested in, however, not even the nodes are known when decoding, so the natural choice is a top-down decoder, which starting from an encoded representation generates the root of the tree and then recursively generates the children (if any) of every node.

The second challenge arises from the asymmetric hierarchical nature of trees. Unlike the sequence-to-sequence setting where encoding and decoding can be achieved with analogous procedures, when dealing with tree-structured data these two involve significantly different operations. For example, an encoder that processes a tree bottom-up using information of a node's children to obtain its representation cannot be simply reversed and used as a decoder, since when generating the tree top-down, nodes have to be generated before their children are.

An additional design constraint comes from deciding what information to feed to each node. For sequences, the choice is obvious: a node should receive information from the node preceding or succeeding it (or both), i.e. there is a one-dimensional flow of information. In trees, there is an evident flow of information from parent to children (or vice-versa), but when generating nodes in a top-down order it seems unnatural to generate children in isolation: the label of one of them will likely influence what the other children states might be. For example, in the case of parse trees, generating a verb will reduce the chances of other verbs occurring in that branch of children.

With these considerations in mind, we propose an architecture tailored to tree decoding from scratch: top-down, recursive and doubly-recurrent, i.e. where both ancestral (parent-to-children) and fraternal (sibling-to-sibling) flow of information is modeled with recurrent modules. Thus, the building block of a doubly recurrent neural network (DRNN) is a cell with two types of input states, one coming from its parent, updated and passed on to its descendants, and another one received from its previous sibling, $^{1}$  updated and passed on to the next one. We model the flow of information in the two directions with separate recurrent modules, which can be vanilla RNN, LSTM, GRUs or variations thereof.

Formally, let  $\mathcal{T} = \{\mathcal{V},\mathcal{E},\mathcal{X}\}$  be a connected labeled tree, where  $\mathcal{V}$  is the set of nodes,  $\mathcal{E}$  the set of edges and  $\mathcal{X}$  are node values. Let  $g^{a}$  and  $g^{f}$  be functions which apply one step of the two separate RNNs. For a node  $i\in \mathcal{V}$  with parent  $p(i)$  and previous sibling  $s(i)$ , the ancestral and fraternal hidden states are updated via

$$
\mathbf {h} _ {i} ^ {a} = g ^ {a} \left(\mathbf {h} _ {p (i)} ^ {a}, \mathbf {x} _ {p (i)}\right) \tag {1}
$$

$$
\mathbf {h} _ {i} ^ {f} = g ^ {f} \left(\mathbf {h} _ {s (i)} ^ {f}, \mathbf {x} _ {s (i)}\right) \tag {2}
$$

where  $\mathbf{x}_{s(j)},\mathbf{x}_{p(i)}$  are the vectors representing the previous sibling's and parent's values, respectively. Once the hidden depth and width states have been updated with these observed labels, they are combined to obtain a predictive hidden state:

$$
\mathbf {h} _ {i} ^ {(p r e d)} = \tanh  \left(\mathbf {U} ^ {f} \mathbf {h} _ {i} ^ {f} + \mathbf {U} ^ {a} \mathbf {h} _ {i} ^ {a}\right) \tag {3}
$$

where  $\mathbf{U}^f\in \mathbb{R}^{n\times D_w}$  and  $\mathbf{U}^a\in \mathbb{R}^{n\times D_d}$  are learnable parameters. This state contains combined information of the node's neighborhood in the tree, and is used to predict a label for it. In its simplest form, the network could compute the output of node  $i$  by sampling from distribution

$$
\mathbf {o} _ {i} = \operatorname {s o f t m a x} \left(\mathbf {W h} _ {i} ^ {(p r e d)}\right) \tag {4}
$$

In the next section, we propose a slight modification to (4) whereby topological information is included in the computation of cell output. After the node's output symbol  $\mathbf{x}_i$  has been obtained by sampling from  $\mathbf{o}_i$ , the cell passes  $\mathbf{h}_i^a$  to all its children and  $\mathbf{h}_i^f$  to the next sibling (if any), enabling them to apply Eqs (1) and (2) to realize their states. This procedure continues recursively, until termination conditions (explained in the next section) cause the procedure to halt.

# 3.1 TOPOLOGICAL PREDICTION

As mentioned before, the central issue with free-form tree construction is to predict the topology of the tree. When constructing the tree top-down, for each node we need to decide: (i) whether it is a leaf node (and thus it should not produce offspring) and (ii) whether there should be additional siblings produced after it. Answering these two questions for every node allows us to construct a tree from scratch and eventual stop growing it.

Sequence decoders typically rely on special tokens to terminate generation (Sutskever et al., 2014). The token is added to the vocabulary and treated as a regular word. During training, the examples are padded with this token at the end of the sequence, and during testing, generation of this token signals termination. This ideas has been adopted by most tree decoders (Dong & Lapata, 2016). There are two important downsides of using a padding strategy for topology prediction in trees. First, the size of the tree can grow considerably. While in the sequence framework only one stopping token is needed, a tree with  $n$  nodes might need up to  $O(n)$  additional padding nodes to be added. This can have important effects in training speed. The second reason is that a single stopping token selected competitively with other tokens requires one to continually update the associated parameters in response to any changes in the distribution over ordinary tokens so as to maintain topological control.

![](images/9ef940bde7d5b67147d6571ae8f4efdc2f49aff21340a81be03a623293945fc1.jpg)  
Figure 1: Left: A cell of the doubly-recurrent neural network corresponding to node  $i$  with parent  $p$  and sibling  $s$ . Right: Structure-unrolled DRNN network in an encoder-decoder setting. The nodes are labeled in the order in which they are generated. Solid (dashed) lines indicate ancestral (fraternal) connections. Crossed arrows indicate production halted by the topology modules.

![](images/a18a36b715b9ef100cdbb5dd093d44871ce894924ad2533ef76e742867d616f7.jpg)

Based on these observations, we propose an alternative approach to stopping, in which topological decisions are made explicitly (as opposed to implicitly, with stopping tokens). For this, we use the predictive hidden state of the node  $\mathbf{h}^{(pred)}$  with a projection and sigmoid activation:

$$
p _ {i} ^ {a} = \sigma \left(\mathbf {u} ^ {a} \cdot \mathbf {h} _ {i} ^ {(p r e d)}\right) \tag {5}
$$

The value  $p_i^a \in [0,1]$  is interpreted as the probability that node  $i$  has children. Analogously, we can obtain a probability of stopping fraternal branch growth after the current node as follows:

$$
p _ {i} ^ {f} = \sigma \left(\mathbf {u} ^ {f} \cdot \mathbf {h} _ {i} ^ {(p r e d)}\right) \tag {6}
$$

Note that these stopping strategies depart from the usual padding methods in a fundamental property: the decision to stop is made before instead of in conjunction with the label prediction. The rationale behind this is that the label of a node will likely be influenced not only by its context, but also by the type of node (terminal or non-terminal) where it is to be assigned. This is the case in language, for example, where syntactic constraints restrict the type of words that can be found in terminal nodes. For this purpose, we include the topological information as inputs to the label prediction layer. Thus, (4) takes the form

$$
\mathbf {o} _ {i} = \operatorname {s o f t m a x} \left(\mathbf {W h} _ {i} ^ {(p r e d)} + \alpha_ {i} \mathbf {v} ^ {a} + \varphi_ {i} \mathbf {v} ^ {f}\right) \tag {7}
$$

where  $\alpha_{i},\varphi_{i}\in \{0,1\}$  are binary variables indicating the topological decisions and  $\mathbf{v}^a,\mathbf{v}^f$  are learnable offset parameters. During training, we use gold values in (7), i.e.  $\alpha_{i} = 1$  if node  $i$  has children and  $\varphi_{i} = 1$  if it has a succeeding sibling. During testing, these values are obtained from  $p^a,p^f$  by sampling or beam-search. A schematic representation of internal structure of a DRNN cell and the flow of information in a tree are shown in Figure 1.

# 3.2 TRAINING DRNNS

We train DRNNs with (reverse) back-propagation through structure BPTS (Goller & Kuechler, 1996). In the forward pass, node outputs are computed in a top-down fashion on the structure-unrolled version of the network, following the natural<sup>3</sup> dependencies of the tree. We obtain error signal at the node level from the two types of prediction: label and topology. For the former, we compute cross entropy loss of  $\mathbf{o}_i$  with respect to the true label of the node  $\mathbf{x}_i$ . For the topological values  $p_i^a$  and  $p_i^f$  we compute binary cross entropy loss with respect to gold topological indicators  $\alpha_i, \varphi_i \in \{0,1\}$ . In the backward pass, we proceed in the reverse (bottom-up) direction, feeding into

![](images/02770c201bc9cfadcf3801111e34de6aa221becbf537254b9e80fda54765ca47.jpg)  
Figure 2: Trees generated by the DRNN decoder trained on subset of size  $N$  of the synthetic dataset, for a test example with description "ROOT B W F J V".

![](images/0e06d4ee0a00238ac8738090f7fa9f0c4bca1f3c1163b0dad5b4d6fca3942695.jpg)

![](images/c296738c4b233ffc72655378132c414618f49bbefacc57b5db157ee9c8bbbb22.jpg)

![](images/d1481615d8f647973c34f6d0a7356bdc89b16602bbe3739538f95649e1c29f80.jpg)

![](images/c9f5cd9b224b18bd6894cbe1f5506f4e493f0684d5fea9785c75b8631a45a542.jpg)

every node the gradients received from child and sibling nodes and computing internally gradients with respect to both topology and label prediction. Further details on the backpropagation flow are provided in the Appendix.

Note that the way BPTS is computed for implies and underlying decoupled loss function

$$
\mathcal {E} \left(\widehat {\mathbf {x}} _ {i}\right) = \sum_ {i \in \mathcal {V}} \mathcal {L} ^ {\text {l a b e l}} \left(\mathbf {x} _ {i}, \widehat {\mathbf {x}} _ {i}\right) + \mathcal {L} ^ {\text {t o p o}} \left(\mathbf {p} _ {i}, \widehat {\mathbf {p}} _ {i}\right) \tag {8}
$$

The decoupled nature of this loss allows us to weigh these two objectives differently, to emphasize either topology or label prediction accuracy. Investigating the effect of this is left for future work.

As is common with sequence generation, during training we perform teacher forcing: after predicting the label of a node and its corresponding loss, we replace it with its gold value, so that children and siblings receive the correct label for that node. Analogously, we compute the probabilities  $p^a$  and  $p^f$ , obtain a loss with respect to actual values, and replace these for ground truth variables  $\alpha_i, \varphi_i$  for all downstream computations. Addressing this exposure bias by mixing ground truth labels with model predictions during training (Venkatraman et al., 2015) or by incremental hybrid losses (Ranzato et al., 2016) is left as an avenue for future work.

# 4 EXPERIMENTS

To validate the proposed method, we first test it on a synthetic dataset, where we evaluate its ability to recover simple trees from flattened string representations. Then, we test it on a real task consisting of recovering functional programs from a natural language description.

# 4.1 SYNTHETIC TREE RECOVERY

For our first set of experiments we generate a toy dataset consisting of simple labeled trees. To isolate the effect of label content from topological prediction, we take a simple vocabulary consisting of the 26 letters of the English alphabet. We generate trees in a top-down fashion, conditioning the label and topology of every node on the state of its ancestors and siblings. For simplicity, we use a Markovian assumption on these dependencies, modeling the probability of a node's label as depending only on the label of its parent and the last sibling generated before it (if any). Conditioned on these two inputs, we model the label of the node as coming from a multinomial distribution over the alphabet with a dirichlet prior. To generate the topology of the tree, we model the probability of a node having children and a next-sibling as depending only on the label of the node and the depth of the tree. Further details on the construction of the dataset are provided in the Appendix.

For each tree we generate a string representation by traversing it in breadth-first preorder, starting from the root. The labels of the nodes are concatenated into a string in the order in which they were visited, resulting in a string of  $|\mathcal{T}|$  symbols. We create a dataset of 5,000 trees with this procedure, and split it randomly into train, validation and test sets (with a  $80\%,10\%,10\%$  split). The characteristics of the dataset are summarized in Table 3.

The task now consists of learning a mapping from strings to trees, and using this learned mapping to recover the tree structure of the test set examples, given only their flattened representation. To do so, we use an encoder-decoder framework, where the strings are mapped to a fixed-size vector representation using a recurrent neural network. For the decoder, we use a DRNN with LSTM modules, which given the encoded representation generates a tree.

![](images/74c4b89312296f2f55a86858629076e1acba7498bbbcf56f31e99d9ae6e9a306.jpg)  
Figure 3: Left: F1-Score for models trained on randomly sampled subsets of varying size, averaged over 5 repetitions. Right: Node (first column) and edge (second) precision as a function of tree size.

![](images/865233de3c06ad26eb6bf3c477a451adc982b0cfe50f96c43366f705511a95ae.jpg)

![](images/b9115a1d5c73b530955a28ca78eae7fa5ab56e759cfa379c71ebf88a7ae25d88.jpg)  
Figure 4: Example recipe from the IFTTT dataset. The description (above) is a user-generated natural language explanation of the if-this-then-that program (below).

Measuring performance only in terms of exact recovery would likely yield near-zero accuracies for most trees. Instead, we opt for a finer-grained metric of tree similarity that gives partial credit for correctly predicted subtrees. Treating tree generation as a retrieval problem, we evaluate the quality of the predicted tree in terms of the precision and recall of generating nodes and edges present in the gold tree. Thus, we penalize both missing and superfluous components.

We choose the hyper-parameters with cross-validation on the validation set. Full training details are provided in the Appendix. Figure 3 shows the results on the test set. Training on the full data yields F1-Scores of close  $75\%$  for node retrieval, and slightly less for edge retrieval. This difference can be explained by correct nodes being generated in the wrong part of the tree, such as in the example shown in Figure 2. The second plot in Figure 3 shows that although small trees are recovered more accurately, precision decays slowly with tree size.

# 4.2 MAPPING SENTENCES TO FUNCTIONAL PROGRAMS

Tree structures arise naturally in the context of programs. A typical compiler takes human-readable source code (expressed as sequences of characters) and transforms it into an executable abstract syntax tree (AST). Source code, however, is already semi-structured. Mapping natural language sentences directly into executable programs is an open problem, which has received considerable interest in natural language processing community (Kate et al., 2005; Branavan et al., 2009).

The IFTTT dataset (Quirk et al., 2015) is a simple testbed for language-to-program mapping. It consists of if this-then-that programs (called recipes) crawled from the IFTTT website<sup>4</sup>, paired with natural language descriptions of their purpose. The recipes consist of a trigger and an action, each defined in terms of a channel (e.g., "Facebook"), a function (e.g., "Post a status update") and potentially arguments and parameters. An example of a recipe and its description are shown in Figure 4. The data is user-generated and extremely noisy, which makes the task significantly challenging.

Table 1: Results on the IFTTT task. Left: non english and unintelligible examples removed (2,262 recipes). Right: examples for which at least  $3+$  turkers agree with gold (758 recipes).  

<table><tr><td>Method</td><td>Channel</td><td>+Func</td><td>F1</td></tr><tr><td>retrieval</td><td>36.8</td><td>25.4</td><td>49.0</td></tr><tr><td>phrasal</td><td>27.8</td><td>16.4</td><td>39.9</td></tr><tr><td>sync</td><td>26.7</td><td>15.4</td><td>37.6</td></tr><tr><td>classifier</td><td>64.8</td><td>47.2</td><td>56.5</td></tr><tr><td>posclass</td><td>67.2</td><td>50.4</td><td>57.7</td></tr><tr><td>SEQ2SEQ</td><td>68.8</td><td>50.5</td><td>60.3</td></tr><tr><td>SEQ2TREE</td><td>69.6</td><td>51.4</td><td>60.4</td></tr><tr><td>GRU-DRNN</td><td>70.1</td><td>51.2</td><td>62.7</td></tr><tr><td>LSTM-DRNN</td><td>74.9</td><td>54.3</td><td>65.2</td></tr></table>

<table><tr><td>Method</td><td>Channel</td><td>+Func</td><td>F1</td></tr><tr><td>retrieval</td><td>43.3</td><td>32.3</td><td>56.2</td></tr><tr><td>phrasal</td><td>37.2</td><td>23.5</td><td>45.5</td></tr><tr><td>sync</td><td>36.5</td><td>23.5</td><td>45.5</td></tr><tr><td>classifier</td><td>79.3</td><td>66.2</td><td>65.0</td></tr><tr><td>posclass</td><td>81.4</td><td>71.0</td><td>66.5</td></tr><tr><td>SEQ2SEQ</td><td>87.8</td><td>75.2</td><td>73.7</td></tr><tr><td>SEQ2TREE</td><td>89.7</td><td>78.4</td><td>74.2</td></tr><tr><td>GRU-DRNN</td><td>89.9</td><td>77.6</td><td>74.1</td></tr><tr><td>LSTM-DRNN</td><td>90.1</td><td>78.2</td><td>77.4</td></tr></table>

We approach this task using an encoder-decoder framework. We use a standard RNN encoder, either an LSTM or a GRU (Cho et al., 2014), to map the sentence to a vector representation, and we use a DRNN decoder to generate the AST representation of the recipe. We use the original data split, which consists of 77,495 training, 5,171 development and 4,294 test examples.

For evaluation, we use the same metrics as Quirk et al. (2015). Given how noisy the dataset is, computing exact accuracy is problematic. Instead, they propose evaluating the generated AST in terms of F1-score on the set of recovered productions. In addition, they compute accuracy at the channel level (when both channels are predicted correctly) and at the function level (both channels and both functions are predicted correctly). We compare our methods against the various extraction and phrased-based machine translation baselines of Quirk et al. (2015) and the methods of Dong & Lapata (2016): SEQ2SEQ, a sequence-to-sequence model trained on flattened representations of the AST, and SEQ2TREE, a token-driven hierarchical RNN. Following Quirk et al. (2015), we report results on two noise-filtered subsets of the data: one with all non-english and unintelligible recipes removed and the other one with recipes for which at least three humans agreed with the gold AST. The results are shown in Table 1. In both subsets, DRNNs perform on par or above previous approaches, with LSTM-DRNN achieving significantly better results. The improvement is particularly evident in terms of F1-score, which is the only metric used by previous approaches that measures global tree reconstruction accuracy. To better understand the quality of the predicted trees beyond the function level (i.e. (b) in Figure 4), we computed node accuracy on the arguments level. Our best performing model, LSTM-DRNN, achieves a Macro F1 score of  $51\%$  (0.71 precision, 0.40 recall) over argument nodes, which shows that the model is reasonably successful at predicting structure even beyond depth three.

# 5 DISCUSSION AND FUTURE WORK

We have presented doubly recurrent neural networks, a natural extension of (sequential) recurrent architectures to tree-structured objects. This architecture models the information flow in a tree with two separate recurrent modules: one carrying ancestral information (received from parent and passed on to offspring) and the other carrying fraternal (passed from sibling to sibling). The topology of the tree is modeled explicitly and separately from the label prediction, with modules that given the state of a node predict whether it has children and/or siblings. The experimental results show that the proposed method is able to predict reasonable tree structures from encoded vector representations. Despite the simple structure of the IFTTT trees, the results on that task suggest a promising direction of using DRNNs for generating programs or executable queries from natural language.

Another compelling application of this new architecture is machine translation. In follow-up work, we are investigating the use of DRNNs for compositional decoding, training on dependency-parsed sentences to generate both words and parse trees at test time. We can scale the approach by resorting to batch processing in GPU. This is possible since forward and backward propagation are computed sequentially along tree traversal paths so that inputs and hidden states of parents and siblings can be grouped into tensors and operated in batch.

# REFERENCES

Srk Branavan, Harr Chen, Luke S. Zettlemoyer, and Regina Barzilay. Reinforcement learning for mapping instructions to actions. Proc. Jt. Conf. 47th Annu. Meet. ACL 4th Int. Jt. Conf. Nat. Lang. Process. AFNLP Vol. 1-Volume 1, (August):82-90, 2009. ISSN 1742206X. doi: 10.3115/1687878.1687892.  
Danqi Chen and Christopher D Manning. A Fast and Accurate Dependency Parser using Neural Networks. Proc. 2014 Conf. Empir. Methods Nat. Lang. Process., (i):740-750, 2014. URL https://cs.stanford.edu/\~danqi/papers/emnlp2014.pdf.  
Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, and Yoshua Bengio. On the Properties of Neural Machine Translation: EncoderDecoder Approaches. Proc. SSST-8, Eighth Work. Syntax. Semant. Struct. Stat. Transl., pp. 103-111, 2014. URL http://arxiv.org/pdf/1409.1259v2.pdf.  
Li Dong and Mirella Lapata. Language to Logical Form with Neural Attention. In ACL, pp. 33-43, 2016. doi: 10.18653/v1/P16-1004. URL http://arxiv.org/abs/1601.01280.  
Gottlob Frege. Über Sinn und Bedeutung. Zeitschrift für Philos. und Philos. Krit., (1):25-50, 1892.  
Christoph Goller and Andreas Kuechler. Learning task-dependent distributed representations by backpropagation through structure. In Int. Conf. Neural Networks, pp. 347-352, 1996. ISBN 0-7803-3210-5. doi: 10.1109/ICNN.1996.548916.  
Sepp Hochreiter and Jurgen Jürgen Schmidhuber. Long short-term memory. *Neural Comput.*, 9(8): 1-32, 1997. ISSN 0899-7667. doi: 10.1162/neco.1997.9.8.1735.  
Rj Kate, Yw Wong, and Rj Mooney. Learning to transform natural to formal languages. In Proc. Natl. Conf. Artif. Intell., volume 20, pp. 1062-1068, 2005. ISBN 1-57735-236-x. URL http://www.aaai.org/Library/AAAI/2005/aaai05-168.php.  
Diederik Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. Int. Conf. Learn. Represent., pp. 1-13, 2014. URL http://arxiv.org/abs/1412.6980.  
Eliyahu Kiperwasser and Yoav Goldberg. Easy-First Dependency Parsing with Hierarchical Tree LSTMs. TACL, 2016. URL https://www.transacl.org/ojs/index.php/tacl/article/viewFile/798/208.  
Jeffrey Pennington, Richard Socher, and Christopher D Manning. GloVe: Global Vectors for Word Representation. In Proc. 2014 Conf. Empir. Methods Nat. Lang. Process., 2014.  
Chris Quirk, Raymond Mooney, and Michel Galley. Language to Code: Learning Semantic Parsers for If-This-Then-That Recipes. ACL-IJCNLP, (July):878-888, 2015. URL http://www.aclweb.org/anthology/P15-1085.  
Marc'Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence Level Training with Recurrent Neural Networks. In ICLR, pp. 1-15, 2016. URL http://arxiv.org/abs/1511.06732.  
R Socher and Cc Lin. Parsing natural scenes and natural language with recursive neural networks. In EMNLP, pp. 129-136, 2011. ISBN 9781450306195. doi: 10.1007/978-3-540-87479-9.  
Richard Socher, Brody Huval, Christopher D Manning, and Andrew Y Ng. Semantic Compositionality through Recursive Matrix-Vector Spaces. In EMNLP, number Mv, pp. 1201-1211, 2012. ISBN 9781937284435.  
Richard Socher, Alex Perelygin, and Jy Wu. Recursive deep models for semantic compositionality over a sentiment treebank. Proc. ..., pp. 1631-1642, 2013. ISSN 1932-6203. doi: 10.1371/journal.pone.0073791.  
Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. Sequence to sequence learning with neural networks. In NIPS, pp. 9, 2014. ISBN 1409.3215. URL http://arxiv.org/abs/1409.3215.

Kai Sheng Tai, Richard Socher, and Christopher D. Manning. Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks. In Proc. 53rd Annu. Meet. Assoc. Comput. Linguist. 7th Int. Jt. Conf. Nat. Lang. Process., pp. 1556-1566, 2015. ISBN 9781941643723. URL http://arxiv.org/abs/1503.0075.  
Arun Venkatraman, Martial Hebert, and J Andrew Bagnell. Improving Multi-step Prediction of Learned Time Series Models. Twenty-Ninth AAAI Conf. Artif. Intell., pp. 3024-3030, 2015.  
Oorio Vinyls and Quoc V. Le. A Neural Conversational Model. arXiv, 37, 2015.  
Ronald J. Williams and David Zipser. Gradient-based learning algorithms for recurrent networks and their computational complexity. Back-propagation Theory, Archit. Appl., pp. 433-486, 1995. doi: 10.1080/02673039508720837.  
Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. Recurrent Neural Network Regularization. *ICLR*, pp. 1-8, 2015. URL http://arxiv.org/abs/1409.2329.  
Xingxing Zhang, Liang Lu, and Mirella Lapata. Top-down Tree Long Short-Term Memory Networks. In NAACL-HLT-2016, pp. 310-320, 2016.
