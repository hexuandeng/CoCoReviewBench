# DEEP LEARNING WITH DYNAMIC COMPUTATION GRAPHS

Moshe Looks, Marcello Herreshoff, DeLesley Hutchins & Peter Norvig

Google Inc.

{madscience,marcelloh,delesley,pnorvig}@google.com

# ABSTRACT

Neural networks that compute over graph structures are a natural fit for problems in a variety of domains, including natural language (parse trees) and cheminformatics (molecular graphs). However, since the computation graph has a different shape and size for every input, such networks do not directly support batched training or inference. They are also difficult to implement in popular deep learning libraries, which are based on static data-flow graphs. We introduce a technique called dynamic batching, which not only batches together operations between different input graphs of dissimilar shape, but also between different nodes within a single input graph. The technique allows us to create static graphs, using popular libraries, that emulate dynamic computation graphs of arbitrary shape and size. We further present a high-level library<sup>1</sup> of compositional blocks that simplifies the creation of dynamic graph models. Using the library, we demonstrate concise and batch-wise parallel implementations for a variety of models from the literature.

# 1 INTRODUCTION

Training deep neural networks directly on minimally pre-processed corpora has led to many recent performance breakthroughs, mainly on problems in domains such as vision (Krizhevsky et al., 2012) and natural language (Bahdanau et al., 2015) where the inputs can be cast as dense  $n$ -dimensional arrays (henceforth tensors), or sequences of tensors. These successes exploit the effectiveness of training via gradient descent on mini-batches of tens to hundreds of inputs, implemented using the parallel SIMD capabilities of modern GPUs (Oh & Jung, 2004) and multi-core CPUs (Vanhoucke et al., 2011). This, in turn has led to a proliferation of libraries making it easier to train and deploy such models, by expressing them in terms of differentiable data-flow graphs over tensors (Abadi et al., 2016; Theano Development Team, 2016; Collobert et al., 2011).

However, there is also a long history of neural networks that compute over structures such as parse trees (Pollack, 1990), logical terms (Goller & Kuchler, 1996), and molecular graphs (Bianucci et al., 2000). In these models, each distinct input has a different computation graph structure; we say that they use dynamic computation graphs (DCGs). Such models continue to be developed and have recently yielded superior results on problems such as sentiment classification and semantic relatedness (Tai et al., 2015; Li et al., 2015), question-answering (Andreas et al., 2016), and screening of chemical compounds (Kearnes et al., 2016). Despite these successes, most practitioners avoid DCGs for implementation reasons. For example, Bowman et al. (2016) assert that "because TreeRNNs use a different model structure for each sentence ... efficient batching is impossible in standard implementations". Moreover, even if efficient batching were possible in principle, current libraries such as TensorFlow (Abadi et al., 2016) assume that the data-flow graph is static (i.e. is the same for each input) and impose a significant cost to graph construction, which makes it infeasible to build a new graph for each input.

Section 2 introduces dynamic batching, which enables efficient batching for training and inference with DCGs. Dynamic batching runs DCGs efficiently with existing libraries that only support static data-flow graphs; e.g. the same static graph can run a TreeRNN over any parse tree. We present empirical results for our implementation in TensorFlow. Section 3 presents a combinator library for concisely implementing models with DCGs using dynamic batching. Section 4 concludes.

# 2 DYNAMIC BATCHING

In deep learning libraries like TensorFlow, computations are manually batched. The computation is expressed as a static graph of mathematical operations, such as  $y = \sigma (x\cdot w + c)$ , which are polymorphic in batch size; an input  $x$  of dimensions  $(b,n)$  will yield an output of dimensions  $(b,m)$ , where  $b$  is the batch size. With DCGs, the graph of operations is not static, but is assumed to be different for every input, so multiple inputs no longer naturally batch together in the same way. The dynamic batching algorithm overcomes this difficulty. Given a set of computation graphs as input, each of which has a different size and topology, it will rewrite the graphs by batching together all instances of the same operation that occur at the same depth in the graph. The rewriting process inserts additional concat and gather operations to move data between the batched operations; the indices to gather encode the topology of the original input graphs.

We distinguish between individual operations appearing as nodes in the underlying data-flow graph, such as addition or matrix-multiply, and small sub-graphs that conceptually act as functions over tensors, such as a feed-forward layer or LSTM cell. We refer to the former as "ops", and to the latter as "operations." Operations, (i.e. sub-graphs), form the building-blocks from which neural networks with DCGs are composed; dynamic batching schedules operations, not ops. Our algorithm requires that all operations which might be used be specified in advance, and it enumerates them for scheduling purposes. For example, a binary TreeRNN for NLP parse trees has two operations: embedding table lookups for words at the leaves of the tree, and RNN cells for the non-terminals.

The inputs and outputs of operations have tensor types. Each input or output may have a different type, but all types must be fixed and fully specified in advance. A tensor type consists of a shape,  $x_{1}, \ldots, x_{n}$ , together with a scalar data type (e.g., float32). The inputs to an operation shall be tensors of dimension  $(b, x_{1}, \ldots, x_{n})$ , where  $b$  is the batch size and  $x_{1} \ldots x_{n}$  is the shape of corresponding input tensor type. The outputs must all be tensors of dimension  $(b, y_{1}, \ldots, y_{m})$ , where  $y_{1}, \ldots, y_{m}$  is the shape of the corresponding output tensor type. Operations must be polymorphic with respect to the batch size, because it will change each time the operation is invoked, depending on the topologies of the input graphs. However, their tensor types are fixed, so that it is possible to assign a known tensor type to each edge in the input computation graph.

The dynamic batching algorithm takes a directed acyclic computation graph as input. A batch of multiple input graphs can be treated as a single disconnected graph. Source nodes are constant tensors, and non-source nodes are operations. Edges connect one of the outputs of a node to one of the inputs of another node. Scheduling is performed using a greedy algorithm:

- Assign a depth to each node in the graph. Nodes with no dependencies (constants) are assigned depth zero. Nodes with only dependencies of depth zero, are assigned depth one, nodes whose dependencies have a maximum depth of one get assigned depth two, etc.  
- Insert pass-through (identity) operations so that an operation at depth  $d + 1$  only refers to results at depth  $d$ .  
- Batch together all nodes invoking the same operation at the same depth into a single node.  
- Concatenate all outputs which have the same depth and tensor type. The order of concatenation corresponds to the order in which the dynamic batching operations were enumerated.  
- Assign a label  $(d, t, i)$  to each edge in the graph, where  $d$  is the depth,  $t$  is the tensor type, and  $i$  is the integer index into the (concatenated) outputs for  $d, t$ . The schedule for the graph consists of the indices  $i$  for each edge, which are grouped together by depth and operation.

In our TensorFlow implementation, each dynamic operation is instantiated once in the static data-flow graph. The inputs to each operation are tf.gather ops, and the outputs are fed into tf.train ops, as described above. These TensorFlow ops are then placed within a tf.train_loop. Each iteration of the loop will evaluate all of the operations at a particular depth. The loop maintains state variables for each tensor type  $t$ , and feeds the output of concat for tensor type  $t$  and iteration  $d$  into the input of the gathers at tensor type  $t$  and iteration  $d + 1$ . The indices for gather at iteration  $d$  are drawn from the edge labels  $i$  for depth  $d$  in the schedule. The initial values for the state variables at iteration/depth 0 are the constants in the input graph.

Dynamic batching allows us to construct a static TensorFlow graph that contains a single instance of each operation, yet can emulate input graphs of arbitrary size and topology where operations may

![](images/6ff7859d50196c028c1493ce6ee04bc2724e84258cf231352c82b4cb7336f83e.jpg)  
Figure 1: The static data-flow graph created by dynamic batching for a binary TreeRNN over parse trees (left), and input graph corresponding to the parse tree  $((word_{1}, word_{3}), word_{5})$  (right).

appear an arbitrary number of times. The TensorFlow concat, gather, and while_loop ops are all differentiable, so gradients calculations and back-propagation do not require any additional code.

For example, a binary TreeRNN as described above yields a TensorFlow data-flow graph with a tf meanwhile_loop whose body is shown on the left of Figure 1. Here each gather has an additional input (the indices for the given op at the given depth) which picks out which elements the operations are to be called with. The long downward arrows are the pass-throughs. The algorithm consumes a tree such as the one shown on the right of Figure 1 and turns it into inputs for the gather operations at each depth (here depth is the loop counter for the tf meanwhile_loop.)

# 2.1 EXPERIMENTAL RESULTS

We have implemented dynamic batching as part of a new library, TensorFlow Fold, and designed a synthetic speed benchmark to compare it with manual batching in native TensorFlow. The benchmark uses the same underlying kernels and execution engine in both cases. Native TensorFlow cannot batch together trees of different shapes so, for testing purposes, we use a batch of random binary trees, all of which have the same shape. These test results thus represent a best-case scenario, in which all operations can be batched together perfectly. For the manual batching tests, we construct a static data-flow graph of operations corresponding to the shape of the tree. For the dynamic batching tests, we traverse each tree to construct a schedule, as described above.

The leaves of the tree are lookups into an embedding table, while the non-terminals implement a variant of the Tree-LSTM (Tai et al., 2015) equations. The tree size is 128, with a state size of 1024 for the LSTM. The CPU tests were run on a Dell z620 workstation with dual 8-core Intel Xeon processors (32 hardware threads), and the GPU tests were done using a consumer Nvidia GeForce GTX-1080 card. We compare manual batching, dynamic batching where all trees have the same shape, and dynamic batching where each tree has a different shape (the column marked "full dynamic"). There is no measurable penalty for dealing with trees of different shapes.

The test results shown in Table 1 emphasize the importance of batching, especially on GPUs. TensorFlow will launch a GPU kernel for every node in the tree, so there is a fixed overhead, proportional to the size of the tree, that dominates execution for small batch sizes. TensorFlow does not begin to saturate the GPU until relatively large batch sizes – 1024 or higher. The difference in speed between fully-batched and unbatched is over  $160\mathrm{x}$ .

Dynamic batching has less kernel invocation overhead because the data-flow graph is smaller. Dynamic batching instantiates each operation only once, and invokes it once for each depth, so the number of kernel invocations is  $\log(n)$ , rather than  $n$ , where  $n$  is tree size. Dynamic batching thus achieves substantial speedups even at batch size 1, because it batches operations at the same depth within a single tree.

Table 1: Inference timing benchmark; times are wall-clock averages in seconds  

<table><tr><td rowspan="2">batch-size</td><td colspan="2">manual</td><td colspan="2">dynamic</td><td colspan="2">full dynamic</td><td rowspan="2">cost ratio</td><td rowspan="2">speedup ratio</td></tr><tr><td>batch</td><td>tree</td><td>batch</td><td>tree</td><td>batch</td><td>tree</td></tr><tr><td>(CPU) 1024</td><td>14.62</td><td>0.014</td><td>18.68</td><td>0.018</td><td>18.37</td><td>0.017</td><td>1.27</td><td>28.86</td></tr><tr><td>512</td><td>7.54</td><td>0.014</td><td>9.84</td><td>0.019</td><td>9.57</td><td>0.018</td><td>1.30</td><td>27.68</td></tr><tr><td>256</td><td>4.14</td><td>0.016</td><td>5.22</td><td>0.020</td><td>5.25</td><td>0.020</td><td>1.26</td><td>25.23</td></tr><tr><td>128</td><td>2.48</td><td>0.019</td><td>2.95</td><td>0.023</td><td>3.08</td><td>0.024</td><td>1.18</td><td>21.47</td></tr><tr><td>64</td><td>1.64</td><td>0.025</td><td>1.76</td><td>0.027</td><td>1.78</td><td>0.027</td><td>1.06</td><td>18.55</td></tr><tr><td>32</td><td>1.27</td><td>0.039</td><td>1.05</td><td>0.032</td><td>1.10</td><td>0.034</td><td>0.82</td><td>14.94</td></tr><tr><td>1</td><td>0.52</td><td>0.517</td><td>0.26</td><td>0.258</td><td>0.26</td><td>0.262</td><td>0.49</td><td>1.97</td></tr><tr><td>(GPU) 1024</td><td>0.978</td><td>0.0009</td><td>1.590</td><td>0.0015</td><td>1.617</td><td>0.0015</td><td>1.62</td><td>101.79</td></tr><tr><td>512</td><td>0.530</td><td>0.0010</td><td>0.715</td><td>0.0013</td><td>0.721</td><td>0.0014</td><td>1.34</td><td>114.15</td></tr><tr><td>256</td><td>0.312</td><td>0.0012</td><td>0.323</td><td>0.0012</td><td>0.340</td><td>0.0013</td><td>1.03</td><td>120.86</td></tr><tr><td>128</td><td>0.236</td><td>0.0018</td><td>0.164</td><td>0.0012</td><td>0.178</td><td>0.0013</td><td>0.69</td><td>115.05</td></tr><tr><td>64</td><td>0.193</td><td>0.0030</td><td>0.093</td><td>0.0014</td><td>0.106</td><td>0.0016</td><td>0.48</td><td>96.40</td></tr><tr><td>32</td><td>0.153</td><td>0.0047</td><td>0.061</td><td>0.0019</td><td>0.074</td><td>0.0023</td><td>0.40</td><td>68.79</td></tr><tr><td>1</td><td>0.161</td><td>0.1608</td><td>0.038</td><td>0.0376</td><td>0.036</td><td>0.0359</td><td>0.23</td><td>4.47</td></tr></table>

However, the extra concat and gather ops that dynamic batching inserts do have a cost. The "cost ratio" column above shows the ratio between dynamic and manual batching, in the case where all trees in the batch have the same shape. The cost is only  $20\%$  for inference on GPUs with batch-size 1, but rises to  $60\%$  for training with backpropagation. The cost is mainly visible at large batch sizes, because it is balanced by the benefit of within-tree batching at smaller sizes.

Even with the cost, dynamic batching yields a 120x speedup over using a batch size of 1 on GPU, and 28x on CPU. The "speedup ratio" column above shows the ratio between the per-tree time for dynamic batching on random shapes ("full dynamic"), versus manual batching with a batch size of 1. Note that using a batch size of 1 is not actually feasible for TensorFlow, because TensorFlow has a large graph construction overhead, which is not included in these measurements, but it may apply to other libraries that lack such overhead.

# 3 A COMBINATOR LIBRARY FOR NEURAL NETWORKS

In addition to dynamic batching, the TensorFlow Fold library provides a set of combinators that simplify the task of constructing neural networks for DCGs. The design of the library was inspired by functional programming techniques such as `parser combinators` (Hutton & Meijer, 1996) and `arrows` (Hughes, 2000). In a combinator library computations are structured compositionally, by plugging together simpler computations in various ways. The basic unit of computation in TensorFlow Fold is a block, essentially a function from input to output. In a typical DCG model, the input is a graph or tree of some kind, and the output is a vector, which can be attached to a loss for training.

For example, consider a model where the inputs are sequences of words, of varying lengths, and the output is a sentence vector. Our library provides several different ways of handling sequences. Given a simpler block  $f$  that operates on elements of the sequence, or  $g$  on pairs of elements, we define the following combinators:

- Map  $(f)$ : yields  $[f(x_1), f(x_2), \ldots, f(x_n)]$ . Applies  $f$  to each element of the sequence, e.g., embedding each of the words of a sentence into  $\mathbb{R}^N$ .  
- Fold  $(g, z)$ : yields  $g(\ldots g(g(z, x_1), x_2), \ldots x_n)$ . Applies  $g$  sequentially in a leftward chain, e.g. running an RNN over a sequence. By default  $z = 0$ .  
- Reduce  $(g)$ : yields  $g(\text{Reduce}([x_1, \ldots, x_{\lfloor n/2 \rfloor}]), \text{Reduce}([x_{\lfloor n/2 \rfloor + 1}, \ldots, x_n]))$ . Applies  $g$  in a balanced tree, e.g., max or sum-pooling over the elements.

Note that it is not necessary to pad or truncate sequences to the same length; dynamic batching handles sequences of differing lengths.

# 3.1 TYPE SYSTEM

Blocks are statically typed; each block has an input type and an output type. Types are inferred where possible, but must be explicitly specified in some cases. A type is one of the following:

- Input denotes objects in the host language (Python), such as trees and dictionaries.  
- Tensor<sub>dtype, shape</sub> denotes tensors of a particular dtype and shape.  
-  $\text{Tuple}(t_1, \ldots, t_n)$ , denotes a tuple of values of types  $t_1, \ldots, t_n$ .  
- Sequence(t), denotes a sequence of elements of type  $t$ , of any length.  
- Void is the unit type.

For example Sequence(Sequence(Tuple(Tensor(float32, [], Tensor(int8, [3, 4]))) denotes jagged arrays whose elements are pairs (float32, int8 $^{3 \times 4}$ ).

# 3.2 BLOCKS AND COMBINATORS

Blocks are composed hierarchically; a block expression is always a tree. The non-terminals in the tree are combinators such as Map and Fold, which take simpler blocks as arguments. The leaves of the tree are atomic blocks, which include the following:

- Scalar: Input  $\rightarrow$  Tensor Convert a Python scalar to a tensor.  
- Tensor: Input  $\rightarrow$  Tensor Convert a NumPy array to a tensor.  
- Function  $(h): [\text{Tensor or Tuple}(\text{Tensor}, \ldots)] \to [\text{Tensor or Tuple}(\text{Tensor}, \ldots)]$

- Defines an operation  $h$  (see Section 2) over tensors. Operations with multiple inputs and outputs use tuples of tensors.  
- InputTransform(h): Input  $\rightarrow$  Input

Applies a user-defined Python function  $h$  to pre-process the input.

In addition to the sequence combinators described above, important combinators in the library include the following:

-  $b_{1} >> b_{2}$ : Function composition; the output of  $b_{1}$  is fed to the input of  $b_{2}$ .  
- Record  $\left\{\left\{l_{1}: b_{1}, \ldots, l_{n}: b_{n}\right\}\right\} : I n p u t \rightarrow T u p l e\left(t_{1}, \ldots, t_{n}\right)$

Takes a Python dictionary or tuple as input, and applies each block  $b_{i}$  to the field labeled  $l_{i}$ , to yield an object of type  $t_{i}$ . Returns a tuple of the results for all fields.

- OneOf  $(b_{1},\ldots b_{n})\colon$  Input  $\rightarrow t$  
Conditionally dispatches on its input to one of the blocks  $b_{1}, \ldots, b_{n}$ .  
- Optional  $(b): \text{Input} \to t$  
Applies  $b$  if the input is not None, otherwise returns zeros. A special case of OneOf.  
-  $\operatorname{Al1Of}\left( {{b}_{1},\ldots {b}_{n}}\right)  : {t}_{0} \rightarrow$  Tuple  $\left( {{t}_{1},\ldots {t}_{n}}\right)$  
Passes its input of type  $t_0$  to each of the blocks  $b_{1},\ldots b_{n}$ , returning a tuple of results.

# 3.3 PIPELINES

Assume we have a set of (text, label) pairs as input and wish to predict the label from the text. The text consists of words, and we want to use an array of pretrained word embeddings (word_matrix) and corresponding dictionary mapping words to indices (word_idx). We call word_idx.get(word) to obtain the index of word in word_matrix, or None if word is unknown.

We start by creating a block which embeds each word into a continuous space:

![](images/8fce5b1c966c50097ec3052ac0f41f8f902ba9b35ec156dbe583a25d464bb505.jpg)  
Figure 2: Block architectures for a pipeline (Section 3.3), feed-forward attention (Section 3.4), binary Tree-LSTMs (Section 3.5), and the weave module for molecule graphs (Section 3.6).

![](images/856fe8bfc7bcfa9a79aa3e844d4e1de1a04c36e1431aaca58d678219906122e6.jpg)

![](images/c08fc77982e6412e32e63e8fcfbc22bc09a3fdd8eea6b6a214f7694dc6f0da99.jpg)

![](images/bae48cc4fb30a9a2629b6395b01688b8c7cb0e32140ec227ccf35d65186dedbd.jpg)

```txt
word2vec = (InputTransform(word_idx.get) >> Optional(Scalar('int32')) >> Function(EMBEDding(initializer=word_matrix)))
```

This block uses an InputTransform to get the index of a word, which is passed to an Optional block that converts the scalar index to a tensor (or 0 if None). This in turn gets passed to an Embedding operation, which performs a lookup into an embedding table.

With word2vec in hand, we can define text2vec, which embeds sentences:

split  $=$  InputTransform(str.split)   
rnn_cell  $\equiv$  Concat()  $>>$  Function(FC(d,activation  $\equiv$  tf.nnrelu))   
text2vec  $=$  split  $>>$  Map(word2vec)  $>>$  Fold(rnn_cell)

We use an InputTransform to split the string into words. Then we map the words to vectors with word2vec, and combine the word vectors with a simple RNN, which uses a single fully connected layer FC with  $d$  hidden units.

Assume there are  $n$  labels; we use a linear layer with  $n$  outputs to get unscaled logits:

```r
text2logs = text2vec >> Function(FC(n, activation=None))
```

For training, we create a Record block to convert the label to a tensor as well, and calculate loss:

record  $=$  Record([('text', text2logits), ('label', Scalar('int32'))])   
loss  $=$  record >> Function(tf.nnsparse softmax.Cross_entropy)

Finally, we create a Compiler, which validates a block, performs type-checking, and sets up dynamic batching in TensorFlow. Outputs of a compiled block are available as TensorFlow tensors, so training now proceeds as it would for any other TensorFlow model:

```python
compiler = Compiler.create(loss)  
cross_entropy = Compiler.output_tensors[0]  
train_op = tf.train.AdamOptimizer().minimize(cross_entropy)
```

# 3.4 COMPLEX COMPOSITIONS

Recently, Raffel & Ellis (2016) have introduced an attention model for feed-forward neural networks. The model generalizes average-pooling and is defined as:

$$
e _ {t} = a \left(h _ {t}\right), \alpha_ {t} = \frac {\exp \left(e _ {t}\right)}{\sum_ {k = 1} ^ {T} \exp \left(e _ {k}\right)}, c = \sum_ {t = 1} ^ {T} \alpha_ {t} h _ {t} \tag {1}
$$

where  $a$  is a learnable function.

In this model, the block architecture is not a simple pipeline (i.e. a composition using  $\gg$ ) but instead forms a directed acyclic graph, as illustrated in Figure 2. A Composition block allows blocks to be composed into DAGs.

attention  $=$  Composition()   
with attentionscope(): h  $=$  attention.input exp_e  $=$  Map(a  $\gg$  Function(tf.exp)).reads(h) z  $=$  (Sum())  $\gg$  Broadcast().reads(exp_e) alpha  $=$  ZipWith(Function(tf.div)).reads(exp_e, z) c  $=$  (ZipWith(Function(tf.mul))  $\gg$  Sum().reads(alpha,h) attention.output.reads(c)

Within a composition scope, blocks may be wired together with reads, provided no directed cycles are formed. The input and output properties a composition are used to define the overall inputs and outputs of the block. This example introduces several additional block types:

- Sum is a specialization of Reduce that performs elementwise addition.  
- ZipWith is a variant of Map that accepts  $n$  sequences as input and applies an  $n$ -ary function  $f$  elementwise (stopping when the end of the shortest input sequence is reached).  
- Broadcast creates a Sequence(t) from a single  $t$ , repeating the same element endlessly.

# 3.5 RECURSIVE DEFINITIONS

$N$ -ary Tree-LSTMs (Tai et al., 2015, sec. 3.2) generalize LSTMs from 1 to  $N$  previous states. In Tai et al. (2015, sec. 5.1) they are applied to classify sentences from the Stanford Sentiment Treebank. This corpus consists of binarized constituency parse trees of one-sentence movie reviews, where every node has a sentiment label. At the leaves of the tree, words are mapped to word-embedding vectors which serve as the input to a binary tree-LSTM with 0 for the previous states. At the internal nodes, the LSTM takes 0 as input, and previous states from its two children. More formally,

$$
h _ {w o r d} = T r e e L S T M (E m b e d d i n g (w o r d), 0, 0) \tag {2}
$$

$$
h _ {\text {l e f t}, \text {r i g h t}} = \operatorname {T r e e L S T M} \left(0, h _ {\text {l e f t}}, h _ {\text {r i g h t}}\right) \tag {3}
$$

where  $TreeLSTM(x, h_{left}, h_{right})$  is a learnable function corresponding to Tai et al. (2015) eqs. 9-14 with  $N = 2$ . Since a tree is a recursive data type, a model that processes trees must be recursively defined, as illustrated by the cycle in Figure 2. A ForwardDeclaration allows the creation of recursive models:

```python
expr = ForwardDeclaration()
word = AllOf(Record(['word', word2vec]), Zeros((state_size, state_size))
pair = AllOf(Zeros(embeding_size), Record(['left', expr]), ('right', expr]))
expr_def = (OneOf(key_fn=len, case_blocks=[(1, word), (2, pair)])) >> TreeLSTM(state_size))
expr.solve_to(expr_def)
```

A forward declaration like expr is not itself a block, but may be called (using the expr() syntax) to create references - i.e. blocks which refer to the declaration. The subsequent call to resolve_to then updates all the references to refer to expr_def.

The Zeros block does what you would expect, and the word2vec block is as defined in Section 3.3.

# 3.5.1 EXPERIMENTAL RESULTS

Here we briefly report on some experiments with our implementation of  $N$ -ary Tree-LSTMs for sentiment analysis. We used constituency Tree-LSTMs with tuned Glove vectors for word embedding, which achieved the best results of all sentiment models presented in Tai et al. (2015). In addition to this specific model, we have explored several novel variants. In particular, Tai et al. (2015) employed non-recurrent dropout and L2 weight regularization. We eliminated weight regularization

Table 2: Test set accuracies on the Stanford Sentiment Treebank  

<table><tr><td>model</td><td>fine-grained</td><td>binary</td></tr><tr><td>Tai et al. (2015)</td><td>51.0 (0.5)</td><td>88.0 (0.3)</td></tr><tr><td>Munkhdalai &amp; Yu (2016a)</td><td>52.8</td><td>89.7</td></tr><tr><td>Munkhdalai &amp; Yu (2016b)</td><td>53.1</td><td>89.3</td></tr><tr><td>Ours (Single Model)</td><td>52.3 (0.7)</td><td>89.4 (0.4)</td></tr><tr><td>Ours (Ensemble)</td><td>53.6</td><td>90.2</td></tr></table>

Table 3: Lines of code comparison  

<table><tr><td>model</td><td>ours</td><td>original</td><td>ratio</td></tr><tr><td>Feed-Forward Attention</td><td>26</td><td>71</td><td>0.37</td></tr><tr><td>Tree-LSTM</td><td>119</td><td>219</td><td>0.54</td></tr><tr><td>Graph Convolutions</td><td>32</td><td>44</td><td>0.73</td></tr></table>

in favor of the recurrent dropout scheme introduced by Semeniuta et al. (2016) and increased the LSTM state size from 150 to 300, leaving all other hyperparameters unchanged.

Results are shown in Table 2, including the best previously reported results. Fine-grained accuracy is measured for all trees and calculated based on the five possible labels. Binary accuracy is measured only for trees with non-neutral sentiment, and is based on negative vs. positive classification. The numbers in parentheses are standard deviations. Tai et al. (2015) report five independent runs, our results are based on thirty independent runs. $^{5}$  Noting the small size of this dataset (8544/1101/2210 trees for train/dev/test), we further evaluated an ensemble consisting of these thirty independently trained models; this variant sets a new state-of-the-art on both subtasks.

# 3.6 GRAPH CONVOLUTIONS

As a final example, we have used the Fold library to implement the graph convolution model introduced by Kearnes et al. (2016) for molecules, which are represented as undirected graphs of atoms. The code is more complex than our previous examples because it involves nested Composition blocks, and is given in Appendix A.

# 4 DISCUSSION

Neural architectures with dynamic computation graphs suffer from inefficient batching and poor tooling. Dynamic batching solves the former problem in full generality, we believe for the first time. The SPINN architecture (Bowman et al., 2016) is an alternative stack-based approach that also enables efficient batching with DCGs, but it is limited to binary trees, and requires padding/truncation to handle trees of different sizes. The Fold library addresses the tooling problem by providing a high-level combinator library which is intended to make it easy for practitioners to rapidly develop and iterate on architectures with DCGs.

The experimental results presented in section 2.1 quantify the impact of dynamic batching. The impact of the combinator library is harder to demonstrate quantitatively. One way to approach this (with a large grain of salt) is by comparing lines of code, which we do in Table 3, vs. the original author's sources. See Appendix B for details on the comparison protocol. Of course, a very short implementation is suboptimal if it comes at the cost of flexibility. The results in Section 3.5.1 show that models from the literature can be reimplemented in Fold, then extended to achieve superior performance. We suspect that other models with DCGs will have quite a bit of "head room" as well, due to simply having less work done tuning them compared with more mainstream architectures.

# REFERENCES

Martin Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. arXiv, 1603.04467, 2016.  
Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Learning to compose neural networks for question answering. In *NAACL*, 2016.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In ICLR, 2015.  
Anna Maria Bianucci, Alessio Micheli, Alessandro Sperduti, and Antonina Starita. Application of cascade correlation networks for structures to chemistry. Applied Intelligence, 2000.  
Samuel R. Bowman, Jon Gauthier, Abhinav Rastogi, Raghav Gupta, Christopher D. Manning, and Christopher Potts. A fast unified model for parsing and sentence understanding. In *NAACL*, 2016.  
Ronan Collobert, Koray Kavukcuoglu, and Clément Farabet. Torch7: A Matlab-like environment for machine learning. In *BigLearn*, NIPS Workshop, 2011.  
Christoph Goller and Andreas Kuchler. Learning task-dependent distributed representations by backpropagation through structure. In ICNN, 1996.  
John Hughes. Generalising monads to arrows. Science of Computer Programming, 2000.  
Graham Hutton and Erik Meijer. Monadic parser combinators. Technical Report NOTTCS-TR-96-4, 1996.  
Steven Kearnes, Kevin McCloskey, Marc Berndl, Vijay Pande, and Patrick Riley. Molecular graph convolutions: moving beyond fingerprints. Journal of Computer-Aided Molecular Design, 2016.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.  
Jiwei Li, Minh-Thang Luong, Dan Jurafsky, and Eudard Hovy. When are tree structures necessary for deep learning of representations? arXiv, 1503.00185, 2015.  
Tsendsuren Munkhdalai and Hong Yu. Neural semantic encoders. arXiv, 1607.04315, 2016a.  
Tsendsumen Munkhdalai and Hong Yu. Neural tree indexers for text understanding. arXiv, 1607.04492, 2016b.  
Kyoung-Su Oh and Keechul Jung. GPU implementation of neural networks. Pattern Recognition, 2004.  
Jordan B Pollack. Recursive distributed representations. Artificial Intelligence, 1990.  
Colin Raffel and Daniel PW Ellis. Feed-forward networks with attention can solve some long-term memory problems. In *ICLR (Workshop Track)*, 2016.  
Stanislau Semeniuta, Aliaksei Severyn, and Erhardt Barth. Recurrent dropout without memory loss. arXiv, 1603.05118, 2016.  
Kai Sheng Tai, Richard Socher, and Christopher D Manning. Improved semantic representations from tree-structured long short-term memory networks. In *NAACL*, 2015.  
Theano Development Team. Theano: A Python framework for fast computation of mathematical expressions. arXiv, 1605.02688, 2016.  
Vincent Vanhoucke, Andrew Senior, and Mark Z. Mao. Improving the speed of neural networks on CPUs. In Deep Learning and Unsupervised Feature Learning, NIPS Workshop, 2011.
