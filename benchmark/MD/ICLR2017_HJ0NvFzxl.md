# LEARNING GRAPHICAL STATE TRANSITIONS

Daniel D. Johnson

Department of Computer Science

Harvey Mudd College

301 Platt Boulevard

ddjohnson@hmc.edu

# ABSTRACT

Graph-structured data is important in modeling relationships between multiple entities, and can be used to represent states of the world as well as many data structures. Li et al. (2016) describe a model known as a Gated Graph Sequence Neural Network (GGS-NN) that can transform graph-structured inputs into output sequences. In this work I introduce a set of graph-based transformations, which I combine to construct a versatile extension of GGS-NNs that uses graph-structured data as an intermediate representation. The model can learn to construct and modify graphs in sophisticated ways based on textual input, and also to use the graphs to produce a variety of outputs. For example, the model successfully learns to solve almost all of the bAbI tasks (Weston et al., 2016), and also discovers the rules governing graphical formulations of a simple cellular automaton and a family of Turing machines.

# 1 INTRODUCTION

Many different types of data can be formulated using a graph structure. One form of data that lends itself to a graphical representation is data involving relationships (edges) between entities (nodes). Abstract maps of places and paths between them also have a natural graph representation, where places are nodes and paths are edges. In addition, many data structures can be expressed in graphical form, including linked lists and binary trees.

Substantial research has been done on producing output when given graph-structured input (Kashima et al., 2003; Shervashidze et al., 2011; Perozzi et al., 2014; Bruna et al., 2013; Duvenaud et al., 2015). Of particular relevance to this work are Graph Neural Networks (Gori et al., 2005; Scarselli et al., 2009), or GNNs, which extends recursive neural networks by assigning states to each node in a graph based on the states of adjacent nodes. Recently Li et al. (2016) have modified GNNs to use gated state updates and to produce output sequences. The resulting networks, called GG-NNs and GGS-NNs, are successful at solving a variety of tasks with graph-structured input.

This work further builds upon GG-NNs and GGS-NNs by allowing graph-structured intermediate representations, as well as graph-structured outputs. This is accomplished using a more flexible graph definition, along with a set of graph transformations which take a graph and other information as input and produce a modified version of the graph. Combining these transformations with a recurrent input model yields the Gated Graph Transformer Neural Network model (GGT-NN), which incrementally constructs a graph given natural language input, and can either produce a final graph representing its current state, or use the graph to produce a natural language output.

Extending GG-NNs in this way opens up a wide variety of applications. Since many types of data can be naturally expressed as a graph, it is possible to train a GGT-NN model to manipulate a meaningful graphical internal state. In this paper I demonstrate the GGT-NN model on the bAbI task dataset, which contains a set of stories about the state of the world. By encoding this state as a graph, a GGT-NN can learn to update the world state based on the input sentences and answer questions based on its internal graph. I also demonstrate that this architecture can learn complex update rules by training it to model a simple 1D cellular automaton and arbitrary 4-state Turing machines. This requires the network to learn how to transform its internal state based on the rules of each task.

# 1.1 GRU

Gated Recurrent Units (GRU) are a type of recurrent network cell introduced by Cho et al. (2014). Each unit uses a reset gate  $r$  and an update gate  $z$ , and updates according to

$$
\mathbf {r} ^ {(t)} = \sigma \left(\mathbf {W} _ {r} \mathbf {x} ^ {(t)} + \mathbf {U} _ {r} \mathbf {h} ^ {(\mathbf {t} - \mathbf {1})} + \mathbf {b} _ {r}\right) \qquad \qquad \mathbf {z} ^ {(t)} = \sigma \left(\mathbf {W} _ {z} \mathbf {x} ^ {(t)} + \mathbf {U} _ {z} \mathbf {h} ^ {(\mathbf {t} - \mathbf {1})} + \mathbf {b} _ {z}\right)
$$

$$
\widetilde {\mathbf {h}} ^ {(t)} = \phi (\mathbf {W} \mathbf {x} + \mathbf {U} (\mathbf {r} ^ {(t)} \odot \mathbf {h} ^ {(\mathbf {t} - \mathbf {1})}) + \mathbf {b})) \qquad \mathbf {h} ^ {(t)} = \mathbf {z} \odot \mathbf {h} ^ {(t - 1)} + (1 - \mathbf {z}) \odot \widetilde {\mathbf {h}} ^ {(t)}
$$

where  $\sigma$  is the logistic sigmoid function,  $\phi$  is an activation function (here tanh is used),  $\mathbf{x}^{(t)}$  is the input vector at timestep  $t$ ,  $\mathbf{h}^{(t)}$  is the hidden output vector at timestep  $t$ , and  $\mathbf{W}$ ,  $\mathbf{U}$ ,  $\mathbf{W}_r$ ,  $\mathbf{U}_r$ ,  $\mathbf{W}_z$ ,  $\mathbf{U}_z$ ,  $\mathbf{b}$ ,  $\mathbf{b}_r$  and  $\mathbf{b}_z$  are learned weights and biases. Note that  $\odot$  denotes elementwise multiplication.

# 1.2 GG-NN AND GGS-NN

Gated Graph Neural Networks (GG-NN) are a form of graphical neural network models described by Li et al. (2016). In a GG-NN, a graph  $\mathcal{G} = (\mathcal{V},\mathcal{E})$  consists of a set  $V$  of nodes  $v$  with unique values  $1,\ldots ,|\mathcal{V}|$  and a set  $\mathcal{E}$  of directed edges  $e = (v,v^{\prime})\in \mathcal{V}\times \mathcal{V}$  oriented from  $v$  to  $v^{\prime}$ . Each node has an annotation  $\mathbf{x}_v\in \mathbb{R}^N$  and a hidden state  $\mathbf{h}_v\in \mathbb{R}^D$ . Additionally, each edge has a type  $y_{e}\in \{1,\dots ,M\}$ . Initially,  $\mathbf{h}_v^{(1)}$  is set to the annotation  $\mathbf{x}_v$  padded with zeros. Then nodes exchange information for some fixed number of timesteps  $T$  according to the propagation model

$$
\mathbf {h} _ {v} ^ {(1)} = \left[ \mathbf {x} _ {v} ^ {\top}, \mathbf {0} \right] ^ {\top} \quad \mathbf {a} _ {v} ^ {(t)} = \mathbf {A} _ {v:} ^ {\top} \left[ \mathbf {h} _ {1} ^ {(t - 1) \top} \dots \mathbf {h} _ {| \mathcal {V} |} ^ {(t - 1) \top} \right] ^ {\top} \tag {1}
$$

$$
\mathbf {z} _ {v} ^ {(t)} = \sigma \big (\mathbf {W} _ {z} \mathbf {a} _ {v} ^ {(t)} + \mathbf {U h} _ {v} ^ {(t - 1)} \big) \qquad \qquad \mathbf {r} _ {v} ^ {(t)} = \sigma \big (\mathbf {W} _ {r} \mathbf {a} _ {v} ^ {(t)} + \mathbf {U} _ {r} \mathbf {h} _ {v} ^ {(t - 1)} \big)
$$

$$
\widetilde {\mathbf {h} _ {v} ^ {(t)}} = \tanh  (\mathbf {W a} _ {v} ^ {(t)} + \mathbf {U} (\mathbf {r} _ {v} ^ {(t)} \odot \mathbf {h} _ {v} ^ {(t - 1)})) \quad \mathbf {h} _ {v} ^ {(t)} = (1 - \mathbf {z} _ {v} ^ {(t)}) \odot \mathbf {h} _ {v} ^ {(t - 1)} + \mathbf {z} _ {v} ^ {(t)} \odot \widetilde {\mathbf {h} _ {v} ^ {(t)}}.
$$

Here  $\mathbf{a}_v^{(t)}$  represents the information received by each node from its neighbors in the graph, and the matrix  $\mathbf{A} \in \mathbb{R}^{D|\mathcal{V}| \times 2D|\mathcal{V}|}$  has a specific structure that determines how nodes communicate. The first half of  $\mathbf{A}$ , denoted  $\mathbf{A}^{(out)} \in \mathbb{R}^{D|\mathcal{V}| \times D|\mathcal{V}|}$ , corresponds to outgoing edges, whereas the second half  $\mathbf{A}^{(in)} \in \mathbb{R}^{D|\mathcal{V}| \times D|\mathcal{V}|}$  corresponds to incoming edges.

Each edge type  $y$  corresponds to specific forward and backward propagation matrices  $\mathbf{P}_y, \mathbf{P}_y' \in \mathbb{R}^{D \times D}$  which determine how to propagate information across an edge of that type in each direction. The  $D \times D$ -sized submatrix of  $\mathbf{A}^{(out)}$  in position  $i, j$  contains  $\mathbf{P}_y$  if an edge of type  $y$  connects nodes  $n_i$  to  $n_j$ , or  $\mathbf{0}$  if no such edge connects in that direction. Similarly, the  $D \times D$ -sized submatrix of the matrix  $\mathbf{A}^{(in)}$  in position  $i, j$  contains  $\mathbf{P}_y'$  if an edge of type  $y$  connects nodes  $n_j$  to  $n_i$ , or  $\mathbf{0}$  if no such edge connects in that direction.  $\mathbf{A}_{v:} \in \mathbb{R}^{D \times 2D|\mathcal{V}|}$  is the submatrix of  $\mathbf{A}$  corresponding to node  $v$ . Thus, multiplication by  $\mathbf{A}_{v:}$  in 1 is equivalent to taking the following sum:

$$
\mathbf {a} _ {v} ^ {(t)} = \sum_ {v ^ {\prime} \in \mathcal {V}} \left(\sum_ {y = 1} ^ {M} s _ {\text {e d g e}} \left(v, v ^ {\prime}, y\right) \odot \mathbf {P} _ {y} + s _ {\text {e d g e}} \left(v ^ {\prime}, v, y\right) \odot \mathbf {P} _ {y} ^ {\prime}\right) \mathbf {h} _ {v ^ {\prime}} ^ {(t - 1)} \tag {2}
$$

where  $s_{\mathrm{edge}}(v,v',y)$  is 1 if  $e = (v,v') \in \mathcal{E}$  and  $y_e = y$ , and 0 otherwise.

The output from a GG-NN is flexible depending on the task. For node selection tasks, a node score  $o_v = g(\mathbf{h}_v^{(T)}, \mathbf{x}_v)$  is given for each node, and then a softmax operation is applied. Graph-level outputs are obtained by combining an attention mechanism  $i$  and a node representation function  $j$ , both implemented as neural networks, to produce the output representation

$$
\mathbf {h} _ {\mathcal {G}} = \tanh  \left(\sum_ {v \in \mathcal {V}} \sigma (i \left(\mathbf {h} _ {v} ^ {(T)}, \mathbf {x} _ {v}\right)) \odot \tanh  (j \left(\mathbf {h} _ {v} ^ {(T)}, \mathbf {x} _ {v}\right))\right) \tag {3}
$$

Gated Graph Sequence Neural Networks (GGS-NN) are an extension of GG-NNs to sequential output  $\mathbf{o}^{(1)},\ldots ,\mathbf{o}^{(K)}$ . At each output step  $k$ , the annotation matrix  $\mathcal{X}$  is given by  $\mathcal{X}^{(k)} = [\mathbf{x}_1^{(k)},\dots,\mathbf{x}_{|\mathcal{V}|}^{(k)}]^\top \in \mathbb{R}^{|\mathcal{V}|\times L_{\mathcal{V}}}$ . A GG-NN  $\mathcal{F}_{\mathbf{o}}$  is trained to predict an output sequence  $\mathbf{o}^{(k)}$  from  $\mathcal{X}^{(k)}$ , and another GG-NN  $\mathcal{F}_{\mathbf{X}}$  is trained to predict  $\mathcal{X}^{(k + 1)}$  from  $\mathcal{X}^{(k)}$ . Prediction of the output at each step is performed as in a normal GG-NN, and prediction of  $\mathcal{X}^{(k + 1)}$  from the set of all final hidden states  $\mathcal{H}^{(k,T)}$  (after  $T$  propagation steps of  $\mathcal{F}_{\mathbf{X}}$ ) occurs according to the equation

$$
\mathbf {x} _ {v} ^ {(k + 1)} = \sigma \left(j \left(\mathbf {h} _ {v} ^ {(k, T)}, \mathbf {x} _ {v} ^ {(k)}\right)\right).
$$

# 2 DIFFERENTIABLE GRAPH TRANSFORMATIONS

In this section, I describe some modifications to the graph structure to make it fully differentiable, and then propose a set of transformations which can be applied to a graph structure in order to transform it. In particular, I redefine a graph  $\mathcal{G} = (\mathcal{V},\mathcal{C})\in \Gamma$  as a set  $V$  of nodes  $v$  with unique values  $1,\ldots ,|\mathcal{V}|$ , and a connectivity matrix  $\mathcal{C}\in \mathbb{R}^{|\mathcal{V}|\times |\mathcal{V}|\times Y}$ , where  $Y$  is the number of possible edge types. As before, each node has an annotation  $\mathbf{x}_v\in \mathbb{R}^N$  and a hidden state  $\mathbf{h}_v\in \mathbb{R}^D$ . However, there is an additional constraint that  $\sum_{j = 1}^{N}x_{v,j} = 1$ . One can then interpret  $x_{v,j}$  as the level of belief that  $v$  should have type  $j$  out of  $N$  possible node types. Each node also has a strength  $s_v\in [0,1]$ . This represents the level of belief that node  $v$  should exist, where  $s_v = 1$  means the node exists, and  $s_v = 0$  indicates that the node should not exist and thus should be ignored.

Similarly, elements of  $\mathcal{C}$  are constrained to the range  $[0,1]$ , and thus one can interpret  $\mathcal{C}_{v,v',y}$  as the level of belief that there should be a directed edge of type  $y$  from  $v$  to  $v'$ . (Note that it is possible for there to be edges of multiple types between the same two nodes  $v$  and  $v'$ , i.e. it is possible for  $\mathcal{C}_{v,v',y} = \mathcal{C}_{v,v',y'} = 1$  where  $y \neq y'$ .)

# 2.1 NODE ADDITION

The node addition transformation  $\mathcal{T}_{\mathrm{add}}: \Gamma \times \mathbb{R}^{\alpha} \to \Gamma$  takes as input a graph  $\mathcal{G}$  and an input vector  $\mathbf{a} \in \mathbb{R}^{\alpha}$ , and produces a graph  $\mathcal{G}'$  with additional nodes. The annotation and strength of each new node is determined by a function  $f_{\mathrm{add}}: \mathbb{R}^{\alpha} \times \mathbb{R}^{\beta} \to \mathbb{R} \times \mathbb{R}^{N} \times \mathbb{R}^{\beta}$ , where  $\alpha$  is the length of the input vector,  $\beta$  is the length of the internal state vector, and as before  $N$  is the number of node types. The new nodes are then produced according to

$$
\left(s _ {\left| V _ {\mathcal {G}} \right| + i}, \mathbf {x} _ {\left| V _ {\mathcal {G}} \right| + i}, \mathbf {h} _ {i}\right) = f _ {\mathrm {a d d}} (\mathbf {a}, \mathbf {h} _ {i - 1}), \tag {4}
$$

starting with  $\mathbf{h}_0$  initialized to some learned initial state, and recurrently computing  $s_v$  and  $\mathbf{x}_v$  for each new node, up to some maximum number of nodes. Based on initial experiments, I found that implementing  $f_{\mathrm{add}}$  as a GRU layer followed by 2 hidden tanh layers was effective, although other recurrent networks would likely be similarly effective. The node hidden states  $\mathbf{h}_v$  are initialized to zero. The recurrence should be computed as many times as the maximum number of nodes that might be produced. The recurrent function  $f_{\mathrm{add}}$  can learn to output  $s_v = 0$  for some nodes to create fewer nodes, if necessary.

# 2.2 NODE STATE UPDATE

The node state update transformation  $\mathcal{T}_{\mathbf{h}}: \Gamma \times \mathbb{R}^{\alpha} \to \Gamma$  takes as input a graph  $\mathcal{G}$  and an input vector  $\mathbf{a} \in \mathbb{R}^{\alpha}$ , and produces a graph  $\mathcal{G}'$  with updated node states. This is accomplished by performing a GRU-style update for each node, where the input is a concatenation of  $\mathbf{a}$  and that node's annotation vector  $\mathbf{x}_v$  and the state is the node's hidden state, according to

$$
\mathbf {r} _ {v} = \sigma \left(\mathbf {W} _ {r} [ \mathbf {a x} _ {v} ] + \mathbf {U} _ {r} \mathbf {h} _ {v} + \mathbf {b} _ {r}\right), \quad \mathbf {z} _ {v} = \sigma \left(\mathbf {W} _ {z} [ \mathbf {a x} _ {v} ] + \mathbf {U} _ {z} \mathbf {h} _ {v} + \mathbf {b} _ {z}\right),
$$

$$
\widetilde {\mathbf {h}} _ {v} ^ {\prime} = \tanh  \left(\mathbf {W} [ \mathbf {a x} _ {v} ] + \mathbf {U} (\mathbf {r} \odot \mathbf {h} _ {v}) + \mathbf {b}\right), \qquad \mathbf {h} _ {v} ^ {\prime} = \mathbf {z} _ {v} \odot \mathbf {h} _ {v} ^ {\prime} + (1 - \mathbf {z} _ {v}) \odot \widetilde {\mathbf {h}} _ {v} ^ {\prime}
$$

# 2.2.1 DIRECT REFERENCE UPDATE

For some tasks, performance can be improved by providing information to nodes of a particular type only. For instance, if the input is a sentence, and one word of that sentence directly refers to a node type (e.g., if nodes of type 1 represent Mary, and Mary appears in the sentence), it can be helpful to allow all nodes of type 1 to perform an update using this information. To accomplish this,  $\mathcal{T}_{\mathbf{h}}$  can be modified to take node types into account. (This modification is denoted  $\mathcal{T}_{\mathbf{h},\mathrm{direct}}$ .) Instead of a single vector  $\mathbf{a} \in \mathbb{R}^{\alpha}$ , the direct-reference transformation takes in  $\mathbf{A} \in \mathbb{R}^{N \times \alpha}$ , where  $\mathbf{A}_n \in \mathbb{R}^\alpha$  is the input vector for nodes with type  $n$ . The update equations then become

$$
\mathbf {a} _ {v} = \mathbf {x} _ {v} \mathbf {A}
$$

$$
\mathbf {r} _ {v} = \sigma \left(\mathbf {W} _ {r} [ \mathbf {a} _ {v} \mathbf {x} _ {v} ] + \mathbf {U} _ {r} \mathbf {h} _ {v} + \mathbf {b} _ {r}\right), \quad \mathbf {z} _ {v} = \sigma \left(\mathbf {W} _ {z} [ \mathbf {a} _ {v} \mathbf {x} _ {v} ] + \mathbf {U} _ {z} \mathbf {h} _ {v} + \mathbf {b} _ {z}\right),
$$

$$
\widetilde {\mathbf {h}} _ {v} ^ {\prime} = \tanh  \left(\mathbf {W} \left[ \mathbf {a} _ {v} \mathbf {x} _ {v} \right] + \mathbf {U} \left(\mathbf {r} \odot \mathbf {h} _ {v}\right) + \mathbf {b}\right), \quad \mathbf {h} _ {v} ^ {\prime} = \mathbf {z} _ {v} \odot \mathbf {h} _ {v} ^ {\prime} + (1 - \mathbf {z} _ {v}) \odot \widetilde {\mathbf {h}} _ {v} ^ {\prime}
$$

# 2.3 EDGE UPDATE

The edge update transformation  $\mathcal{T}_{\mathcal{C}}: \Gamma \times \mathbb{R}^{\alpha} \to \Gamma$  takes a graph  $\mathcal{G}$  and an input vector  $\mathbf{a} \in \mathbb{R}^{\alpha}$ , and produces a graph  $\mathcal{G}'$  with updated edges. For each pair of nodes  $(v, v')$ , the update equations are

$$
\mathbf {c} _ {v, v ^ {\prime}} = f _ {\text {s e t}} \left(\mathbf {a}, \mathbf {x} _ {v}, \mathbf {h} _ {v}, \mathbf {x} _ {v ^ {\prime}}, \mathbf {h} _ {v ^ {\prime}}\right)
$$

$$
\mathbf {r} _ {v, v ^ {\prime}} = f _ {\text {r e s e t}} \left(\mathbf {a}, \mathbf {x} _ {v}, \mathbf {h} _ {v}, \mathbf {x} _ {v ^ {\prime}}, \mathbf {h} _ {v ^ {\prime}}\right)
$$

$$
\mathcal {C} _ {v, v ^ {\prime}} ^ {\prime} = \left(1 - \mathcal {C} _ {v, v ^ {\prime}}\right) \odot \mathbf {c} _ {v, v ^ {\prime}} + \mathcal {C} _ {v, v ^ {\prime}} \odot \left(1 - \mathbf {r} _ {v, v ^ {\prime}}\right).
$$

The functions  $f_{\mathrm{set}}, f_{\mathrm{reset}}: \mathbb{R}^{\alpha \times 2N \times 2D} \to [0,1]^Y$  are implemented as neural networks. (In my experiments, I used a simple 2-layer fully connected network.)  $\mathbf{c}_{v,v',y}$  gives the level of belief in [0, 1] that an edge from  $v$  to  $v'$  of type  $y$  should be created if it does not exist, and  $\mathbf{r}_{v,v',y}$  gives the level of belief in [0, 1] that an edge from  $v$  to  $v'$  of type  $y$  should be removed if it does. Setting both to zero results in no change for that edge, and setting both to 1 toggles the edge state.

# 2.4 PROPAGATION

The propagation transformation  $\mathcal{T}_{\mathrm{prop}}: \Gamma \to \Gamma$  takes a graph  $\mathcal{G} = \mathcal{G}^{(0)}$  and runs a series of  $T$  propagation steps (as in GG-NN), returning the resulting graph  $\mathcal{G}' = \mathcal{G}^{(T)}$ . The GG-NN propagation step is extended to handle node and edge strengths, as well as to allow more processing to occur to the information transferred across edges. The full propagation equations for step  $t$  are

$$
\mathbf {a} _ {v} ^ {(t)} = \sum_ {v ^ {\prime} \in \mathcal {V}} s _ {v ^ {\prime}} \sum_ {y = 1} ^ {M} \mathcal {C} _ {v, v ^ {\prime}, y} \odot f _ {y} ^ {\mathrm {f w d}} \left(\mathbf {x} _ {v ^ {\prime}}, \mathbf {h} _ {v ^ {\prime}} ^ {(t - 1)}\right) + \mathcal {C} _ {v ^ {\prime}, v, y} \odot f _ {y} ^ {\mathrm {b w d}} \left(\mathbf {x} _ {v ^ {\prime}}, \mathbf {h} _ {v ^ {\prime}} ^ {(t - 1)}\right) \tag {5}
$$

$$
\mathbf {z} _ {v} ^ {(t)} = \sigma \left(\mathbf {W} _ {z} \left[ \mathbf {a} _ {v} ^ {(t)} \mathbf {x} _ {v} \right] + \mathbf {U h} _ {v} ^ {(t - 1)} + \mathbf {b} _ {z}\right) \tag {6}
$$

$$
\mathbf {r} _ {v} ^ {(t)} = \sigma \left(\mathbf {W} _ {r} \left[ \mathbf {a} _ {v} ^ {(t)} \mathbf {x} _ {v} \right] + \mathbf {U} _ {r} \mathbf {h} _ {v} ^ {(t - 1)} + \mathbf {b} _ {r}\right) \tag {7}
$$

$$
\widetilde {\mathbf {h} _ {v} ^ {(t)}} = \tanh  \left(\mathbf {W} \left[ \mathbf {a} _ {v} ^ {(t)} \mathbf {x} _ {v} \right] + \mathbf {U} \left(\mathbf {r} _ {v} ^ {(t)} \odot \mathbf {h} _ {v} ^ {(t - 1)}\right) + \mathbf {b} _ {h}\right) \tag {8}
$$

$$
\mathbf {h} _ {v} ^ {(t)} = \left(1 - \mathbf {z} _ {v} ^ {(t)}\right) \odot \mathbf {h} _ {v} ^ {(t - 1)} + \mathbf {z} _ {v} ^ {(t)} \odot \widetilde {\mathbf {h} _ {v} ^ {(t)}}. \tag {9}
$$

Equation 5 has been adjusted in the most significant manner (relative to 2). In particular,  $s_{v'}$  restricts propagation so that nodes with low strength send less information to adjacent nodes,  $s_{\mathrm{edge}}$  has been replaced with  $\mathcal{C}$  to allow edges with fractional strength, and the propagation matrices  $\mathbf{P}_y, \mathbf{P}_y'$  have been replaced with arbitrary functions  $f_y^{\mathrm{fwd}}, f_y^{\mathrm{bwd}}: \mathbb{R}^N \times \mathbb{R}^D \to \mathbb{R}^\alpha$ , where  $\alpha$  is the length of the vector  $\mathbf{a}$ . I used a fully connected layer to implement each function in my experiments. Equations 6, 7, and 8 have also been modified slightly to add a bias term.

# 2.5 AGGREGATION

The aggregation transformation  $\mathcal{T}_{\mathrm{repr}}: \Gamma \to \mathbb{R}^{\alpha}$  produces a graph-level representation vector from a graph. It functions very similarly to the output representation of a GG-NN, given in equation 3, but is modified slightly to take into account node strengths. As in GG-NN, both  $i$  and  $j$  are neural networks, and in practice a single fully connected layer appears to be adequate for both.

$$
\mathbf {h} _ {\mathcal {G}} = \tanh \left(\sum_ {v \in \mathcal {V}} s _ {v} \sigma (i (\mathbf {h} _ {v} ^ {(T)}, \mathbf {x} _ {v})) \odot \tanh (j (\mathbf {h} _ {v} ^ {(T)}, \mathbf {x} _ {v}))\right).
$$

# 3 GATED GRAPH TRANSFORMER NEURAL NETWORK (GGT-NN)

Combining a series of these transformations yields a Gated Graph Transformer Neural Network (GGT-NN). Depending on the configuration of the transformations, a GGT-NN can take textual or graph-structured input, and produce textual or graph-structured output. Here I describe one particular GGT-NN configuration, designed to build and modify a graph based on a sequence of input sentences, and then produce an answer to a query.

For each sentence  $k$ , each word is converted to a one-hot vector  $\mathbf{w}_l^{(k)}$ , and the sequence of words (of length  $L$ ) is passed through a GRU layer to produce a sequence of partial-sentence representation vectors  $\mathbf{p}_l^{(k)}$ . The full sentence representation vector  $\mathbf{i}^{(k)}$  is initialized to the last partial

Algorithm 1 Graph Transformation Pseudocode  
1:  $\mathcal{G}\gets \emptyset$  11:  $\mathcal{G}\gets \mathcal{T}_{\mathrm{add}}(\mathcal{G},[\mathbf{i}^{(k)}\mathbf{h}_{\mathcal{G}}^{\mathrm{add}}])$    
2: for  $k$  from 1 to  $K$  do 12:  $\mathcal{G}\gets \mathcal{T}_{\mathcal{C}}(\mathcal{G},\mathbf{i}^{(k)})$    
3:  $\mathcal{G}\gets \mathcal{T}_{\mathrm{h}}(\mathcal{G},\mathbf{i}^{(k)})$  13: end for   
4: if direct reference enabled then 14:  $\mathcal{G}\gets \mathcal{T}_{\mathrm{h}}^{\mathrm{query}}(\mathcal{G},\mathbf{i}^{\mathrm{query}})$    
5:  $\mathcal{G}\gets \mathcal{T}_{\mathrm{h,direct}}(\mathcal{G},\mathbf{D}^{(k)})$  15: if direct reference enabled then   
6: end if 16:  $\mathcal{G}\gets \mathcal{T}_{\mathrm{h,direct}}^{\mathrm{query}}(\mathcal{G},\mathbf{D}^{\mathrm{query}})$    
7: if intermediate propagation enabled then 17: end if   
8:  $\mathcal{G}\gets \mathcal{T}_{\mathrm{prop}}(\mathcal{G})$  18:  $\mathcal{G}\gets \mathcal{T}_{\mathrm{prop}}^{\mathrm{query}}(\mathcal{G})$    
9: end if 19:  $\mathbf{h}_{\mathcal{G}}^{\mathrm{answer}}\gets \mathcal{T}_{\mathrm{repr}}^{\mathrm{query}}(\mathcal{G})$    
10:  $\mathbf{h}_{\mathcal{G}}^{\mathrm{add}}\gets \mathcal{T}_{\mathrm{repr}}(\mathcal{G})$  20: return  $f_{\mathrm{output}}(\mathbf{h}_{\mathcal{G}}^{\mathrm{answer}})$

representation vector  $\mathbf{p}_L^{(k)}$ . Furthermore, a direct-reference input matrix  $\mathbf{D}^{(k)}$  is set to the sum of partial representation vectors corresponding to the words that directly reference a node type, i.e.  $\mathbf{D}_n^{(k)} = \sum_{l\in R_n}\mathbf{p}_l^{(k)}$  where  $R_{n}$  is the set of words in sentence  $k$  that directly refer to node type  $n$ .

Next, a series of graph transformations are applied, as depicted in Algorithm 1. Depending on the task, direct reference updates and per-sentence propagation can be enabled or disabled. The output function  $f_{\mathrm{output}}$  will depend on the specific type of answer desired. If the answer is a single word,  $f_{\mathrm{output}}$  can be a multilayer perceptron followed by a softmax operation. If the answer is a sequence of words,  $f_{\mathrm{output}}$  can use a recurrent network (such as a GRU) to produce a sequence of outputs. Note that transformations with different superscripts ( $T_{\mathrm{h}}$  and  $T_{\mathrm{h}}^{\mathrm{query}}$ , for instance) refer to similar transformations with different learned weights.

# 3.1 SUPERVISION

As with many supervised models, one can evaluate the loss based on the likelihood of producing an incorrect answer, and then minimize the loss by backpropagation. However, based on initial experiments, the model appeared to require additional supervision to extract meaningful graph-structured data. To provide this additional supervision, I found it beneficial to provide the correct graph at each timestep and train the network to produce that graph. This occurs in two stages, first when new nodes are proposed, and then when edges are adjusted. For the edge adjustment, the edge loss between a correct edge matrix  $\mathcal{C}^*$  and the computed edge matrix  $\mathcal{C}$  is given by

$$
\mathcal {L} _ {\text {e d g e}} = \sum \mathcal {C} ^ {*} \odot \ln (\mathcal {C}) + (1 - \mathcal {C} ^ {*}) \odot \ln (1 - \mathcal {C}).
$$

The node adjustment is slightly more complex. Multiple nodes are added in each timestep, and those nodes are added in some order, but the order of the nodes is arbitrary. The order in which the nodes are created does not matter, only the existence of the nodes is important. Thus it should be possible for the network to determine the optimal ordering of the nodes. In fact, this is important because there is no guarantee that the nodes will be ordered consistently in the training data.

Vinyals et al. (2016) demonstrate a simple method for training a network to output unordered sets: the network produces a sequence of outputs, and these outputs are compared with the closest ordering of the training data, i.e., the ordering of the training data which would produce the smallest loss when compared with the network output. Vinyals et al. show that when using this method, the network arbitrarily chooses an ordering which may not be the optimal ordering for the task. However, in this case any ordering should be sufficient, and I found the arbitrary orderings selected in this way to work well in practice. In particular, letting  $s_{\pi(v)}^*$  and  $\mathbf{x}_{\pi(v)}^*$  denote the correct strength and annotations of node  $v$  under ordering  $\pi$ , the loss becomes

$$
\mathcal {L} _ {\text {n o d e}} = \max  _ {\pi} \sum_ {v = | \mathcal {V} _ {\text {o l d}} | + 1} ^ {| \mathcal {V} _ {\text {n e w}} |} s _ {\pi (v)} ^ {*} \ln \left(s _ {v}\right) + \left(1 - s _ {\pi (v)} ^ {*}\right) \ln \left(1 - s _ {v}\right) + \mathbf {x} _ {\pi (v)} ^ {*} \ln (\mathbf {x} _ {v}).
$$

At this point the correct values  $\mathcal{C}^*$ ,  $s_{\pi(v)}^*$  and  $\mathbf{x}_{\pi(v)}^*$  are substituted into the graph for further processing. Note that only the edges and the new nodes are replaced by the supervision. The hidden states of all existing nodes are propagated without adjustment.

<table><tr><td rowspan="2">Task</td><td colspan="2">Direct reference</td><td colspan="2">No direct reference</td></tr><tr><td>Accuracy</td><td>No. ex. req. ≥ 95%</td><td>Accuracy</td><td>No. ex. req. ≥ 95%</td></tr><tr><td>1 - Single Supporting Fact</td><td>100%</td><td>100</td><td>99.3%</td><td>1000</td></tr><tr><td>2 - Two Supporting Facts</td><td>100%</td><td>250</td><td>94.3%</td><td>FAIL</td></tr><tr><td>3 - Three Supporting Facts</td><td>98.7%</td><td>1000</td><td>88.0%</td><td>FAIL</td></tr><tr><td>4 - Two Arg. Relations</td><td>98.8%</td><td>1000</td><td>97.8%</td><td>1000</td></tr><tr><td>5 - Three Arg. Relations</td><td>87.2%</td><td>FAIL</td><td>80.2%</td><td>FAIL</td></tr><tr><td>6 - Yes/No Questions</td><td>100%</td><td>100</td><td>92.3%</td><td>FAIL</td></tr><tr><td>7 - Counting</td><td>100%</td><td>250</td><td>94.4%</td><td>FAIL</td></tr><tr><td>8 - Lists/Sets</td><td>100%</td><td>250</td><td>96.7%</td><td>1000</td></tr><tr><td>9 - Simple Negation</td><td>100%</td><td>250</td><td>88.4%</td><td>FAIL</td></tr><tr><td>10 - Indefinite Knowledge</td><td>96.6%</td><td>1000</td><td>71.4%</td><td>FAIL</td></tr><tr><td>11 - Basic Coreference</td><td>100%</td><td>100</td><td>99.8%</td><td>1000</td></tr><tr><td>12 - Conjunction</td><td>99.9%</td><td>500</td><td>99.3%</td><td>1000</td></tr><tr><td>13 - Compound Coref.</td><td>100%</td><td>100</td><td>99.2%</td><td>1000</td></tr><tr><td>14 - Time Reasoning</td><td>97.8%</td><td>1000</td><td>44.9%</td><td>1000</td></tr><tr><td>15 - Basic Deduction</td><td>99.1%</td><td>500</td><td>100%</td><td>500</td></tr><tr><td>16 - Basic Induction</td><td>100%</td><td>100</td><td>100%</td><td>500</td></tr><tr><td>17 - Positional Reasoning</td><td>88.9%</td><td>FAIL</td><td>51.3%</td><td>FAIL</td></tr><tr><td>18 - Size Reasoning</td><td>97.9%</td><td>1000</td><td>89.4%</td><td>FAIL</td></tr><tr><td>19 - Path Finding</td><td>100%</td><td>500</td><td>29.4%</td><td>FAIL</td></tr><tr><td>20 - Agent&#x27;s Motivations</td><td>100%</td><td>250</td><td>99.0%</td><td>250</td></tr></table>

Table 1: Performance of GGT-NN on the bAbI tasks. "No. ex. req. ≥ 95%" refers to the number of training examples required before the network was able to reach 95% accuracy or better on the task.

# 4 EXPERIMENTS

# 4.1 BABITASKS

I evaluated the GGT-NN model on the bAbI tasks, a set of simple natural-language tasks, where each task is structured as a sequence of sentences followed by a query (Weston et al., 2016). The generation procedure for the bAbI tasks includes a "Knowledge" object after each sentence, representing the current state of knowledge after that sentence. I exposed this knowledge object in graph format, and used this to train a GGT-NN in supervised mode. The knowledge object provides names for each node type, and direct reference was performed based on these names: if a word in the sentence matched a node type name, it was parsed as a direct reference to all nodes of that type. For details on this graphical format, see Appendix A.

# 4.1.1 RESULTS

I trained two versions of the GGT-NN model for each task: one with and one without direct reference. Tasks 3 and 5, which involve a complex temporal component, were trained with intermediate propagation, whereas all of the other tasks were not because the structure of the tasks made such complexity unnecessary. Most task models were configured to output a single word, but task 19 was configured using a GRU to output multiple words, and task 8 (the listing task) was configured to output a strength for each possible word to allow multiple words to be selected without having to consider ordering.

Results are shown in Table 1. The GGT-NN model with direct reference performed very well on the majority of the tasks, reaching accuracies of at least  $95\%$  in all but two tasks, and reaching  $100\%$  accuracy in the majority of the tasks. Additionally, for many of the tasks, the model was able to reach  $95\%$  accuracy using 500 or fewer of the 1000 training examples. The two exceptions were task 5 (Three Arg. Relations) and task 17 (Positional Reasoning), for which the model was not able to attain a high accuracy. Task 5 involves sophisticated temporal reasoning, and thus requires a complex graphical structure to model accurately. Task 17 has a larger number of possible entities than the other tasks: each entity consists of a color (chosen from five options) and a shape (chosen from four shapes), for a total of 20 unique entities that must be represented separately. It is likely that these additional complexities caused the network performance to suffer.

Of particular interest is the performance of the GGT-NN model with direct reference on task 19, the pathfinding task. Previous models, such as the end-to-end memory networks described by Sukhbaatar et al. (2015), have struggled to learn this task. On the other hand, GGS-NN models were able to successfully learn the pathfinding task, but required the input to be preprocessed into graphical form even during testing (Li et al., 2016). The current results demonstrate that the proposed GGT-NN model is able to solve the pathfinding task when given textual input.

In general, the GGT-NN model with direct reference performs better than the model without it (see Table 1). Although the model without direct reference reaches  $95\%$  accuracy on more than half of the tasks, it fails to reach  $95\%$  accuracy on multiple other tasks. Additionally, when compared to the direct-reference model, it requires more training examples in order to reach the accuracy threshold. This indicates that, although the model can be used without direct reference, adding direct reference greatly improves the training of the model.

# 4.2 RULE DISCOVERY

To demonstrate the power of GGT-NN to model a wide variety of graph-based problems, I applied the GGT-NN to two additional tasks. In each task, a sequence of data structures were transformed into a graphical format, and the GGT-NN was tasked with predicting the data for the next timestep based on the current timestep. No additional information was provided as textual input; instead, the network was tasked with learning the rules governing the evolution of the graph structure over time.

# 4.2.1 CELLULAR AUTOMATON TASK

The first task used was a 1-dimensional cellular automaton, specifically the binary cellular automaton known as Rule 30 (Wolfram, 2002). Rule 30 acts on an infinite set of cells, each with a binary state (either 0 or 1). At each timestep, each cell deterministically changes state based on its previous state and the states of its neighbors. In particular, the update rules are

<table><tr><td>Current neighborhood</td><td>111</td><td>110</td><td>101</td><td>100</td><td>011</td><td>010</td><td>001</td><td>000</td></tr><tr><td>Next value</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td></tr></table>

Cell states can be converted into graphical format by treating the cells as a linked list. Each of the cells is represented by a node with edges connecting it to the cell's neighbors, and a value edge is used to indicate whether the cell is 0 or 1. This format is described in more detail in Appendix A.

# 4.2.2 TURING MACHINES

The second task was simulating an arbitrary 2-symbol 4-state Turing machine. A Turing machine operates on an infinite tape of cells, each containing a symbol from a finite set of possible symbols. It has a head, which points at a particular cell and can read and write the symbol at that cell. It also has an internal state, from a finite set of states. At each timestep, based on the current state and the contents of the cell at the head, the machine writes a new symbol, changes the internal state, and can move the head left or right or leave it in place. The action of the machine depends on a finite set of rules, which specify the actions to take for each state-symbol combination. Note that the version of Turing machine used here has only 2 symbols, and requires that the initial contents of the tape be all 0 (the first symbol) except for finitely many 1s (the second symbol).

When converting a Turing machine to graphical format, the tape of the machine is modeled as a linked list of cells. Additionally, each state of the machine is denoted by a state node, and edges between these nodes encode the transition rules. There is also a head node, which connects both to the current cell and to the current state of the machine. See Appendix A for more details.

# 4.2.3 ANALYSIS

The GGT-NN model was trained on 1000 examples of the Rule 30 automaton with different initial states, each of which simulated 7 timesteps of the automaton, and 20,000 examples of Turing machines with different rules and initial tape contents, each of which simulated 6 timesteps of the Turing machine. Performance was then evaluated on 1000 new examples generated with the same format. The models were evaluated by picking the most likely graph generated by the model, and

Original Task Generalization: 20 Generalization: 30  

<table><tr><td>Automaton</td><td>100.0%</td><td>87.0%</td><td>69.5%</td></tr><tr><td>Turing</td><td>99.9%</td><td>90.4%</td><td>80.4%</td></tr></table>

Table 2: Accuracy of GGT-NN on the Rule 30 Automaton and Turing Machine tasks.  

<table><tr><td>1000 iterations</td><td>2000 iterations</td><td>3000 iterations</td><td>7000 iterations</td><td>Ground truth</td></tr><tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr></table>

Figure 1: Visualization of network performance on the Rule 30 Automaton task. Top node (purple) represents zero, bottom node (blue) represents 1, and middle nodes (green, orange, and red) represent individual cells. Blue edges indicate adjacent cells, and gold edges indicate the value of each cell. Three timesteps occur between each row.

comparing it with the correct graph. The percent accuracy denotes the fraction of the examples for which these two graphs were identical at all timesteps. In addition to evaluating the performance on identical tasks, the generalization ability of the models was also assessed. The same trained models were evaluated on versions of the task with 20 and 30 timesteps of simulation.

Results are shown in Table 2. The models successfully learned the assigned tasks, reaching high levels of accuracy for both tasks. Additionally, the models show the ability to generalize to large inputs, giving a perfect output in the majority of extended tasks. For visualization purposes, Figure 1 shows the model at various stages of training when evaluated starting with a single 1 cell.

# 5 CONCLUSION

The results presented here show that GGT-NNs are able to successfully model a wide variety of tasks using graph-structured states and potentially could be useful in solving many other types of problems. The specific GGT-NN model described here can be used as-is for tasks consisting of a sequence of input sentences and graphs, optionally followed by a query. In addition, due to the modular nature of GGT-NNs, it is possible to reconfigure the order of the transformations to produce a model suitable for a different task. As one example, Appendix B describes a version of the model that uses the full sequence of sentence graphs while computing the answer to the query, instead of basing the answer on the final graph only.

One downside of the current model is that the time and space required to train the model increase very quickly as the complexity of the task increases, which limits the model's applicability. It would be very advantageous to develop optimizations that would allow the model to train faster and with smaller space requirements.

There are exciting potential uses for the GGT-NN model. One particularly interesting application would be using GGT-NNs to extract graph-structured information from unstructured textual descriptions. More generally, the graph transformations provided here may allow machine learning to interoperate more flexibly with other data sources and processes with structured inputs and outputs.

# ACKNOWLEDGMENTS

I would like to thank Harvey Mudd College for computing resources. I would also like to thank the developers of the Theano library, which I used to run my experiments. This work used the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number ACI-1053575.

# REFERENCES

Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally connected networks on graphs. arXiv preprint arXiv:1312.6203, 2013.  
Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.  
David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. In Advances in Neural Information Processing Systems, pp. 2224-2232, 2015.  
Marco Gori, Gabriele Monfardini, and Franco Scarselli. A new model for learning in graph domains. In Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2005., volume 2, pp. 729-734. IEEE, 2005.  
Hisashi Kashima, Koji Tsuda, and Akihiro Inokuchi. Marginalized kernels between labeled graphs. In ICML, volume 3, pp. 321-328, 2003.  
Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. *ICLR*, 2016.  
Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 701-710. ACM, 2014.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61-80, 2009.  
Nino Shervashidze, Pascal Schweitzer, Erik Jan van Leeuwen, Kurt Mehlhorn, and Karsten M Borgwardt. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research, 12(Sep):2539-2561, 2011.  
Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. End-to-end memory networks. In Advances in neural information processing systems, pp. 2440-2448, 2015.  
Theano Development Team. Theano: A Python framework for fast computation of mathematical expressions. arXiv e-prints, abs/1605.02688, May 2016. URL http://arxiv.org/abs/1605.02688.  
John Towns, Timothy Cockerill, Maytal Dahan, Ian Foster, Kelly Gaither, Andrew Grimshaw, Victor Hazlewood, Scott Lathrop, Dave Lifka, Gregory D Peterson, et al. XSEDE: accelerating scientific discovery. Computing in Science & Engineering, 16(5):62-74, 2014.  
Oriol Vinyals, Samy Bengio, and Manjunath Kudlur. Order matters: Sequence to sequence for sets. *ICLR*, 2016.  
Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M Rush, Bart van Merrienboer, Armand Joulin, and Tomas Mikolov. Towards ai-complete question answering: A set of prerequisite toy tasks. ICLR, 2016.  
Stephen Wolfram. A new kind of science, volume 5. Wolfram media Champaign, 2002.

1. John grabbed the milk.  
2. John travelled to the bedroom.  
3. Sandra took the football.  
4. John went to the garden.  
5. John let go of the milk.  
6. Sandra let go of the football.  
7. John got the football.  
8. John grabbed the milk.

Where is the milk?

![](images/78e0c10a00b29d6b507fb345b203600fa6d4a8dbe953f63ae9eb539625c80e11.jpg)  
Figure 2: Diagram of one sample story from the bAbI dataset (Task 2), along with a graphical representation of the knowledge state after the italicized sentence.
