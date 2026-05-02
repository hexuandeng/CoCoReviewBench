# ADAPTIVE CONTROL FLOW IN TRANSFORMERS IMPROVES SYSTEMATIC GENERALIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Despite successes across a broad range of applications, Transformers have limited capability in systematic generalization. The situation is especially frustrating in the case of algorithmic tasks, where they often fail to find intuitive solutions that can be simply expressed in terms of attention patterns. In the end, it is often all about routing the right information to the right node/operation at the right time in the grid represented by Transformer columns. To facilitate the learning of useful control flow, we propose two modifications to the Transformer architecture, copy gate and geometric attention. Our novel Transformer Control Flow (TCF) achieves  $100\%$  length generalization accuracy on the classic compositional table lookup task, as well as near-perfect accuracy on the simple arithmetic task and a new variant of ListOps testing for computational depth generalization. TCF's attention and gating patterns tend to be interpretable.

# 1 INTRODUCTION

Neural networks (NNs) typically do not generalize to a systematically different test set, even when they can easily solve the training set. Examples of systematic generalization (Fodor et al., 1988) include generalization to sequences longer than those seen during training - productivity, and algorithmic combinations of previously learned rules - systematicity. Despite recent efforts (Bahdanau et al., 2019; Korrel et al., 2019; Lake, 2019; Li et al., 2019; Russian et al., 2019; Csordás et al., 2021), systematic generalization generally remains unsolved (Fodor & McLaughlin, 1990; Lake & Baroni, 2018; Liska et al., 2018; Greff et al., 2020; Hupkes et al., 2020). On some datasets, the best performing models are neuro-symbolic hybrids (Chen et al., 2020; Liu et al., 2020) using task-specific symbolic functions. However, their applicability to other datasets remains limited (Furrer et al., 2020; Shaw et al., 2020). A big question is: which type of architectural inductive bias encourages the training process to select "good" solutions which systematically generalize?

The popular Transformer (Vaswani et al., 2017) architecture also suffers from such problems (Chaabouni et al., 2021; Csordás et al., 2021; Ontañón et al., 2021). Frustratingly, Transformers fail to generalize in many algorithmic tasks (e.g. Liska et al. (2018); Dubois et al. (2020)), even those that have intuitive solutions which can be simply expressed in terms of Transformer attention patterns. Indeed, given an input sequence of length  $N$  and a Transformer encoder of depth  $T$ , solving an algorithmic task is often all about routing the right information to the right node/operation at the right time in the  $T$ -by-  $N$  grid represented by Transformer columns. Effectively the task is to learn to draw an adaptive control flow on the canvas of Transformer columns. In fact, recent work by Weiss et al. (2021) introduced a programming language called RASP which is specifically designed to express solutions to sequence processing problems, and which has a direct equivalent to the operations in Transformers. However, this work also shows that making Transformers learn a solution expressed in RASP is only possible with intermediate supervision of attention patterns. In some cases, even such supervision fails (Weiss et al., 2021). In general, Transformers fail to find easily interpretable and/or symbolic solutions to algorithmic tasks. We conversely hypothesize that attention-based NNs able to find intuitive solutions (achieving interpretable attention patterns) can improve systematic generalization.

Here we argue that regular Transformers lack some basic ingredients for learning such "intuitive" solutions to algorithmic problems. As a remedy, we propose simple architectural modifications to help them learn data routing. As a first step towards validating our model, we focus on the popular length

generalization task of compositional table lookup (CTL; Liska et al. (2018); Hupkes et al. (2019); Dubois et al. (2020)), as well as two more complex tasks: a simple arithmetic task and a variant of ListOps (Nangia & Bowman, 2018) which are designed to test NNs' ability for compositional generalization. Our novel Transformer Control Flow (TCF) achieves  $100\%$  generalization accuracy (never reported before; Dubois et al. (2020)) on the CTL task, and obtains nearly perfect accuracy on both the proposed simple arithmetic and ListOps tasks. We show that the attention and gating patterns of TCF tend to be interpretable as plausible control flows.

# 2 IMPROVING TRANSFORMERS FOR LEARNING ADAPTIVE CONTROL FLOW

We argue that the following components are needed to build Transformers capable of learning adaptive control flow. First, composing known operations in an arbitrary order requires that all operations are available at every computational step. This can be easily achieved by sharing the weights of the layers, as is done in Universal Transformers (Dehghani et al., 2019). Second, the network should be sufficiently deep, at least as deep as the deepest data dependency in the computational graph (e.g., in case of a parse tree, this is the depth of the tree). Otherwise, multiple operations would be fused into a single layer and hinder natural and elegant compositions. Third, inputs in some columns should be kept unchanged until the right point in time when they can be processed in a later step/layer. The regular Transformer lacks a mechanism for skipping the whole transformation step by simply copying the input to the next step/layer. We propose a special gating function, copy gate, to implement such a mechanism (Sec. 2.1). Finally, many algorithmic tasks require combining a number of local computations in the right order. This typically implies that at a given time, attention should not focus on all possible matches, but only on the closest match. We propose and investigate a new type of attention with the corresponding inductive bias (Sec. 2.2), which we call geometric attention. We refer to the resulting new Transformer as Transformer Control Flow (TCF).

# 2.1 COPY GATE: LEARNING TO SKIP OPERATIONS (VERTICAL FLOW)

Each layer of the regular Transformer consists of one self-attention and one feedforward block. The input to each of these blocks are directly connected to the corresponding output via a residual connection (Srivastava et al., 2015; He et al., 2016). However, such a connection does not allow for skipping the transformation of the entire layer and simply passing the unchanged input to the next layer. Here we propose to add an explicit gate, which we call copy gate, to facilitate such a behavior.

We consider a  $T$ -layer Transformer encoder and an input sequence of length  $N$ . Since each layer corresponds to one computational step, we often refer to a layer as a step  $t$ . We denote the Transformer state of column  $i$  in layer  $t$  as  $\pmb{h}^{(i,t)} = \mathbf{H}_{t,i} \in \mathbb{R}^d$  where  $d$  is the state size, and  $\mathbf{H}_t \in \mathbb{R}^{N \times d}$  denotes the states of all  $N$  columns in layer  $t$ . In the copy gate-augmented Transformer, each column  $i$  in layer  $(t + 1)$  processes the input  $\mathbf{H}_t$  like in the regular Transformer:

$$
\boldsymbol {a} ^ {(i, t + 1)} = \operatorname {L a y e r N o r m} (\operatorname {M u l t i H e a d A t t e n t i o n} \left(\boldsymbol {h} ^ {(i, t)}, \mathbf {H} _ {t}, \mathbf {H} _ {t}\right) + \boldsymbol {h} ^ {(i, t)}) \tag {1}
$$

$$
\hat {\boldsymbol {h}} ^ {(i, t + 1)} = \operatorname {L a y e r N o r m} \left(\mathrm {F F N} ^ {\text {d a t a}} \left(\boldsymbol {a} ^ {(i, t + 1)}\right)\right) \tag {2}
$$

but the output is gated as:

$$
\boldsymbol {g} ^ {(i, t + 1)} = \sigma \left(\operatorname {F F N} ^ {\text {g a t e}} \left(\boldsymbol {a} ^ {(i, t + 1)}\right)\right) \tag {3}
$$

$$
\boldsymbol {h} ^ {(i, t + 1)} = \boldsymbol {g} ^ {(i, t + 1)} \odot \hat {\boldsymbol {h}} ^ {(i, t + 1)} + (1 - \boldsymbol {g} ^ {(i, t + 1)}) \odot \boldsymbol {h} ^ {(i, t)} \tag {4}
$$

We use the basic two-layer feedforward block (Vaswani et al., 2017) for both  $\mathrm{FFN^{data}}$  and  $\mathrm{FFN^{gate}}$  which transforms input  $\pmb{x} \in \mathbb{R}^d$  to:

$$
\operatorname {F F N} (\boldsymbol {x}) = \boldsymbol {W} _ {2} \max  \left(\boldsymbol {W} _ {1} \boldsymbol {x} + \boldsymbol {b} _ {1}, 0\right) + \boldsymbol {b} _ {2} \tag {5}
$$

but with separate parameters and different dimensionalities: for  $\mathrm{FFN}^{\mathrm{data}}$ ,  $W_1^{\mathrm{data}} \in \mathbb{R}^{d_{\mathrm{FF}} \times d}$ ,  $W_2^{\mathrm{data}} \in \mathbb{R}^{d \times d_{\mathrm{FF}}}$ , while for  $\mathrm{FFN}^{\mathrm{gate}}$ ,  $W_1^{\mathrm{gate}}$ ,  $W_2^{\mathrm{gate}} \in \mathbb{R}^{d \times d}$ , with biases  $b_1^{\mathrm{data}} \in \mathbb{R}^{d_{\mathrm{FF}}}$  and  $b_2^{\mathrm{data}}$ ,  $b_1^{\mathrm{gate}}$ ,  $b_2^{\mathrm{gate}} \in \mathbb{R}^d$ .

When the gate is closed i.e.  $g^{(i,t + 1)} = 0$  in Eq. 4, the entire transformation is skipped and the input is copied over to the next layer  $h^{(i,t + 1)} = h^{(i,t)}$ . Crucially, we parameterize the gate (Eq. 3) as a function of the output of the self-attention (Eq. 1), such that the decision to copy or to transform the

![](images/c8f4b2b8f77f1eec9a07caceaedd372fdd83a1a7fd9f1c5198f12cbfeed6069d.jpg)  
Figure 1: Left: an ideal sequence of computations in a Transformer for an example CTL task. Right: the order of source positions for each target, counting from the closest, used in geometric attention.

![](images/91fddc2d7259e34d6d5d66d8718d4a31f7992bd4dd0848ebc459c32ef431dcc0.jpg)

input for each column depends on the states of all columns. This is a crucial difference compared to previously proposed gating in Transformers which are solely motivated by the training stability (Parisotto et al., 2020) or a common practice from convolution based models (Chaabouni et al., 2021).

The bias of the gate  $b_{2}^{\mathrm{gate}}$  is initialized to  $-3$  (Hochreiter & Schmidhuber, 1997). This ensures that initially no update happens, and creates a better gradient flow between layers. It also encourages the model to skip layers, unless they have an important contribution in the corresponding step.

# 2.2 GEOMETRIC ATTENTION: LEARNING TO ATTEND TO THE CLOSEST MATCH (HORIZONTAL FLOW)

We propose geometric attention designed to attend to the closest matching element. Like in regular self-attention, given an input sequence  $\left[\pmb{x}^{(1)},\pmb{x}^{(2)},\dots,\pmb{x}^{(N)}\right]$  with  $\pmb{x}^{(i)}\in \mathbb{R}^{d_{\mathrm{in}}}$ , each input is projected to key  $\pmb{k}^{(i)}\in \mathbb{R}^{d_{\mathrm{key}}}$ , value  $\pmb{v}^{(i)}\in \mathbb{R}^{d_{\mathrm{value}}}$ , query  $\pmb{q}^{(i)}\in \mathbb{R}^{d_{\mathrm{key}}}$  vectors, and the dot product is computed for each key/query combination. In our geometric attention, the dot product is followed by a sigmoid function to obtain a score between 0 and 1:

$$
\boldsymbol {P} _ {i, j} = \sigma \left(\boldsymbol {k} ^ {(j) \top} \boldsymbol {q} ^ {(i)}\right) \tag {6}
$$

which will be treated as a probability of the key at (source) position  $j$  matching the query at (target) position  $i$ . These probabilities are finally converted to the attention scores  $A_{i,j}$  as follows:

$$
\boldsymbol {A} _ {i, j} = \boldsymbol {P} _ {i, j} \prod_ {k \in \mathbb {S} _ {i, j}} (1 - \boldsymbol {P} _ {i, k}) \tag {7}
$$

where  $\mathbb{S}_{i,j}$  denotes the set of all (source) indices which are closer to  $i$  than  $j$  to  $i$ , and when two indices have the same distance to  $i$ , we consider the one which is to the right of  $i$  (i.e., greater than  $i$ ) to be closer, i.e.,

$$
\mathbb {S} _ {i, j} = \left\{ \begin{array}{l l} k \in \{1, \dots , N \} \backslash \{i, j \}: | i - k | \leq | i - j |, & \text {i f} i <   j \\ k \in \{1, \dots , N \} \backslash \{i, j \}: | i - k | <   | i - j |, & \text {i f} j <   i \end{array} \right. \tag {8}
$$

In addition, we explicitly zero-out the diagonal by setting  $A_{i,i} = 0$  for all  $i = 1,\dots,N$ . The ordering of source indices is illustrated in Figure 1/Right. The resulting scores  $A_{i,j}$  are the attention scores used to compute the weighted average of the value vectors.

By using the term  $(1 - P_{i,k})$  in Eq. 7, when there is a match, it downscales any other matches which are more distant. Two recent works (Brooks et al., 2021; Banino et al., 2021) use such a parameterized geometric distribution in the form of Eq. 7 (see Sec. 6 on related work).

The resulting attention function has a complexity of  $O(N^2)$ , similar to the regular self-attention used in Transformers (Vaswani et al., 2017). Eq. 7 can be implemented in a numerically stable way in log-space. The products can then be calculated using cumulative sums, subtracting the elements for the correct indices in each position.

Directional encoding. In practice, we augment Eq. 6 by an additional directional encoding. In fact, the only positional information available in the geometric attention presented above is the ordering used to define the product in Eqs. 7-8. In practice, we found it crucial to augment the score computation of Eq. 6 with an additional directional information. The directional information we introduce is encoded as a scalar  $D_{i,j}$  for each target/source position pair  $(i,j)$  computed as:

$$
\boldsymbol {D} _ {i, j} = \left\{ \begin{array}{l l} \boldsymbol {W} _ {\mathrm {L R}} \boldsymbol {h} ^ {(i)} + b _ {\mathrm {L R}}, & \text {i f} i \leq j \\ \boldsymbol {W} _ {\mathrm {R L}} \boldsymbol {h} ^ {(i)} + b _ {\mathrm {R L}}, & \text {i f} i > j \end{array} \right. \tag {9}
$$

where  $\pmb{h}^{(i)}\in \mathbb{R}^d$  denotes the input/state at position  $i$  and  $W_{\mathrm{LR}}$ ,  $W_{\mathrm{RL}}\in \mathbb{R}^{1\times d}$ ,  $b_{\mathrm{LR}}$ ,  $b_{\mathrm{RL}}\in \mathbb{R}$  are trainable parameters. This directional information is integrated into the score computation of Eq. 6 as follows (akin to how Dai et al. (2019) introduce the relative positional encoding as an extra term in the computation of attention scores):

$$
\boldsymbol {P} _ {i, j} = \sigma \left(\alpha \left(\boldsymbol {h} ^ {(i)} \boldsymbol {W} _ {q} + \boldsymbol {u}\right) ^ {\top} \boldsymbol {W} _ {k, E} \boldsymbol {h} ^ {(j)} + \beta \boldsymbol {D} _ {i, j} + \gamma\right) \tag {10}
$$

where the matrix  $W_{q} \in \mathbb{R}^{d_{\mathrm{head}} \times d}$  maps the states to queries,  $u \in \mathbb{R}^{d_{\mathrm{head}}}$  is a bias for queries,  $W_{k,E} \in \mathbb{R}^{d_{\mathrm{head}} \times d}$  maps states to keys (we note that  $d_{\mathrm{head}}$  is typically the size of the key, query and value vectors for each head,  $d_{\mathrm{head}} = \frac{d}{n_{\mathrm{heads}}}$ ), and  $\alpha, \beta, \gamma \in \mathbb{R}$  are learned scaling coefficients and bias, initialized to  $\alpha = \frac{1}{\sqrt{d_{\mathrm{head}}}}$ ,  $\beta = 1$ ,  $\gamma = 0$ . Using this additional directional information, each query (position  $i$ ) can potentially learn to restrict its attention to either the left or right side.

# 3 EXPERIMENTS

We evaluate the proposed methods on three tasks: the standard compositional table lookup (Liska et al., 2018; Hupkes et al., 2019) and two tasks we propose: simple arithmetic and a variant of ListOps (Nangia & Bowman, 2018). In all cases, the task is designed to test NNs' ability for compositional generalization: the model has to learn to apply operations seen during training in a longer/deeper compositional form.

# 3.1 COMPOSITIONAL TABLE LOOKUP

Task. Compositional table lookup task (Liska et al., 2018; Hupkes et al., 2019; Dubois et al., 2020) is constructed from a set of symbols and unitary functions defined over these symbols. Each example in the task is defined with one input symbol and a list of functions to be applied sequentially, i.e. the first function is applied to the input symbol, and the resulting output becomes the input to the second function, and so forth. There are 8 possible symbols. Each symbol is traditionally represented by a 3-bit bitstring (Liska et al., 2018), however, in practice, they are simply processed as one token (Dubois et al., 2020). The functions are bijective and randomly generated. Each function is represented by a letter. An example input is '101 d a b', which corresponds to the expression  $b(a(d(101)))$ ; the model has to predict the correct output symbol. We note that there exists a sequence-to-sequence variant of this task (Dubois et al., 2020) where the model has to predict all intermediate steps (thus trained with intermediate supervision). We directly predict the final output. An ideal model should be able to solve this task independently of the presentation order, i.e., whether the task is encoded as '101 d a b' or 'b a d 101'. We thus study both forward (former) and backward (latter) variants of the task. To evaluate systematic generalization, the train/valid/test sets have different numbers of compositions: samples of up to 5/6-8/9-10 operations respectively. No previous work has reported the perfect accuracy on this task using a NN. We refer the readers to Sec. 6 for further details on the previous work.

Results. We consider four different baselines: an LSTM (Hochreiter & Schmidhuber, 1997), DNC (Graves et al., 2016; Csordás & Schmidhuber, 2019), Universal Transformers (Vaswani et al., 2017; Dehghani et al., 2019), and its relative position variants (the variant from Csordás et al. (2021) to be specific). For Transformers, the prediction is based on the last column in the final layer. Results are shown in Table 1. The LSTM and DNC perform well in the forward variant, achieving perfect generalization for longer sequences, but fail on the backward variant. In contrast, basic Transformers fail in both cases.

By introducing the copy gate (Sec. 2.1), the relative Transformer is able to solve the forward task, but not the backward one. Our analysis showed that the network learns to attend to the last operation based on the relative position information. Since the result is read from the last column, this position changes with the sequence length. The model thus fails to generalize to such arbitrary offsets. To remediate this issue, we introduce a simple mechanism to let the model choose between absolute and relative positional encodings at each position (see Appendix A). The resulting model effectively manages to make use of the absolute position for the prediction, and to perform well in both directions. However, such a combination of absolute/relative positional encoding might be an overly specific bias. A more generic solution, geometric attention (Sec. 2.2), also achieves perfect generalization, and was found easier to train. We present the corresponding visualization of our model in Sec. 4.

Table 1: Accuracy on compositional table lookup dataset.  

<table><tr><td rowspan="2">Model</td><td colspan="2">IID</td><td colspan="2">Longer</td></tr><tr><td>Forward</td><td>Backward</td><td>Forward</td><td>Backward</td></tr><tr><td>LSTM</td><td>1.00 ± 0.00</td><td>0.59 ± 0.03</td><td>1.00 ± 0.00</td><td>0.22 ± 0.03</td></tr><tr><td>DNC</td><td>1.00 ± 0.00</td><td>0.57 ± 0.06</td><td>1.00 ± 0.00</td><td>0.18 ± 0.02</td></tr><tr><td>Transformer</td><td>1.00 ± 0.00</td><td>0.82 ± 0.39</td><td>0.13 ± 0.01</td><td>0.12 ± 0.01</td></tr><tr><td>+ rel</td><td>1.00 ± 0.00</td><td>1.00 ± 0.00</td><td>0.23 ± 0.05</td><td>0.13 ± 0.01</td></tr><tr><td>+ rel + gate</td><td>1.00 ± 0.00</td><td>1.00 ± 0.00</td><td>0.99 ± 0.01</td><td>0.19 ± 0.04</td></tr><tr><td>+ abs/rel + gate</td><td>1.00 ± 0.00</td><td>1.00 ± 0.00</td><td>0.98 ± 0.02</td><td>0.98 ± 0.03</td></tr><tr><td>+ geom. att.</td><td>0.96 ± 0.04</td><td>0.93 ± 0.06</td><td>0.16 ± 0.02</td><td>0.15 ± 0.02</td></tr><tr><td>+ geom. att. + gate (TCF)</td><td>1.00 ± 0.00</td><td>1.00 ± 0.00</td><td>1.00 ± 0.00</td><td>1.00 ± 0.00</td></tr></table>

# 3.2 SIMPLE ARITHMETICS

In order to validate the success of the proposed model on a task which involves more complex data flows and operations, we propose a simple arithmetic task.

Task. The task is to execute an arithmetic expression consisting of nested modulo 10 additions and multiplications. This requires the model to process tree-structured data flows, which is presumably more difficult than the sequential processing required for the CTL task. Each operation is surrounded by brackets, such that the boundaries of operations are easy to determine. For example  $\left( (4 \times 7) + 2 \right)'$  should evaluate to  $'0'$  (30 modulo 10). The expressions are generated randomly. The tree depth is up to 5 for the training set, 6 for the validation and 7 and 8 for the test set. The depth is measured as the number of operations without the leaves, so the example above has a depth of 2. The length of all examples are limited to a maximum of 50 tokens.

Results. Table 2 shows the results. Even though all considered models performs well on the IID validation data, none except the TCF performs well on the generalization test set, which achieves a near-perfect accuracy of  $98\%$ . We also note that the TCF learns very fast: while all other models required about  $200\mathrm{K}$  steps to converge, the TCF already achieves the near-perfect accuracy at  $50\mathrm{K}$  steps of training.

Table 2: Performance of different models on simple arithmetic dataset. All models are trained  ${200}\mathrm{\;K}$  iterations, except the TCF which we stopped training at  ${100}\mathrm{\;K}$  . We also report the performance of all models after  ${50}\mathrm{\;K}$  iterations,where it can be seen that our model converges significantly faster compared to the others.  

<table><tr><td></td><td>IID (1..5)</td><td colspan="2">Test (7..8)</td></tr><tr><td></td><td>200 K</td><td>200 K</td><td>50 K</td></tr><tr><td>LSTM</td><td>0.99 ± 0.00</td><td>0.74 ± 0.02</td><td>0.72 ± 0.01</td></tr><tr><td>Transformer</td><td>0.98 ± 0.01</td><td>0.47 ± 0.01</td><td>0.29 ± 0.01</td></tr><tr><td>+ rel</td><td>1.00 ± 0.00</td><td>0.77 ± 0.04</td><td>0.40 ± 0.05</td></tr><tr><td>+ abs/rel + gate</td><td>1.00 ± 0.01</td><td>0.80 ± 0.16</td><td>0.73 ± 0.15</td></tr><tr><td>+ geom. att. + gate (TCF)</td><td>1.00 ± 0.00</td><td>0.98 ± 0.01</td><td>0.98 ± 0.01</td></tr></table>

# 3.3 LISTOPS

We also evaluate our model on a variant of the ListOps task (Nangia & Bowman, 2018) which is a popular task commonly used to evaluate parsing abilities of neural networks (Havrylov et al., 2019; Shen et al., 2019; Tay et al., 2021; Irie et al., 2021).

Task. The task consists of executing nested list operations written in prefix notation. The elements of a list are either one digit number (from 0 to 9) or recursively another list. The operations are min, max, median and sum. The sum is modulo 10 and the median is followed by the floor function

such that the output of any function/operation is between 0 and 9. For example: [MED 4 8 5 [MAX 8 4 9 ]] should return 6. There are two well known variants of ListOps: the original version by Nangia & Bowman (2018) and the "Long Range Arena" version by Tay et al. (2021) which differ from each other in terms of the maximum number of arguments in each function and the maximum sequence length in terms of number of tokens. In both versions, there is no strict control on the depth of samples in the data: there is simply a certain pre-defined probability such that each position-element in the list is expanded to start another list (which can increase the tree depth). This is not suitable for evaluating systematic generalization in terms of compositionality (over the problem depth). We instead propose to generate clean train, valid, and test splits with disjoint depths. In our settings, the training set contains samples with depths of up to 5, the validation set only contains the depth-7 samples and finally, the test set contains depth-7 and 8 samples. Also importantly, we make sure that a depth- $K$  sample effectively requires computation until the depth- $K$  (without such a control, min, max, and med operations can potentially find the output without checking/executing all its arguments). By dissociating the splits by the depth, we can clearly identify models which fail to compositionally generalize. Apart from the depth specifications, all train-valid/test sets share the same following settings: the maximum sequence length is 50 tokens, the probability of recursively sampling another function inside a list is  $30\%$  at each position, and the maximum number of arguments for a function is 5. The train set consists of 1M, the valid and test sets of 1K sequences.

Results. Table 3 shows the results. Similarly to compositional table lookup and simple arithmetic tasks, the baseline LSTM and Transformers do not generalize well to the test set consisting of deeper problems, while they achieve a near perfect accuracy on IID data. In contrast, our model achieves near-perfect generalization.

Table 3: Performance of different models on balanced ListOps dataset. All models are trained  $200\mathrm{K}$  iterations, except all +gate variants which converge after  $100\mathrm{K}$  steps.  

<table><tr><td></td><td>IID (1..5)</td><td>Test (7..8)</td></tr><tr><td>LSTM</td><td>0.99 ± 0.00</td><td>0.71 ± 0.03</td></tr><tr><td>Transformer</td><td>0.98 ± 0.00</td><td>0.74 ± 0.03</td></tr><tr><td>+ rel</td><td>0.98 ± 0.01</td><td>0.79 ± 0.04</td></tr><tr><td>+ abs/rel + gate</td><td>1.00 ± 0.01</td><td>0.90 ± 0.06</td></tr><tr><td>+ geom. att. + gate (TCF)</td><td>1.00 ± 0.00</td><td>0.99 ± 0.01</td></tr></table>

# 4 ANALYSIS

In this section, we provide some visualizations of attention and gating patterns of the TCF and the corresponding analyses. For more visualizations, we refer the readers to Appendix C.

Compositional Table Lookup. Figure 2 shows the gating and attention patterns of the TCF model for an example of the backward presentation task. As shown in Fig. 2/Bottom, the gates of different columns open sequentially one after another, when the input is available for them. Fig. 2/Top shows the corresponding attention maps. Each column attends to the neighboring one, waiting for its computation to be finished. The behavior of the last column is different: it always attends to the second position of the sequence, which corresponds to the last operation to be performed.

ListOps. We can also identify how the data is processed by the TCF in ListOps. We observe that different attention heads play different roles. We highlight the core observations in Figure 3. The input for this example is: [SM [MED [MIN 1 7 4 [MAX 2 4 0 8 9 ]] 7] 5 [MED 8 5 8] 0 7]. First of all, we find that there is a head (head 13 in Figure 3, first row) which seems to be responsible for connecting operators and their arguments: the operands/arguments of an operation attend to the operator. In step 0 ( $t = 0$  in the figure), we can recognize that the operations at the deepest level, namely MAX and the second MED have all the arguments ready (as is shown by vertical lines on the columns corresponding to MAX and MED). The model certainly identifies that these two operations are ready to be processed and that they can be processed in parallel (these arguments-to-operation attention patterns remain for a few steps). We note that at this stage, the last

![](images/7e184e2122d18593d452e92b8fac521a6516448548e73880d8db6a9c4cc14f74.jpg)  
Figure 2: Example visualization of TCF. For other models, see Appendix C. Top: Attention map for different steps. The x/y-axis corresponds to source/target positions respectively. Each position focuses on the column to the right, except the last one where the result is read from, which focuses on the last operation. The focus becomes clear only once the result is available. Bottom: gate activations for different steps/layers. The gates remain closed until the data dependencies are satisfied.

![](images/155a81d8811a25042f7e47ddb6bbe7d9b81f54f2903007f415894ab7c31af4c5.jpg)  
Figure 3: Example visualization of TCF on ListOps. The top row shows head 13 in different steps, which controls which arguments used in which step. The bottom row shows different heads in different key steps. Please refer to Sec. 4 for the step-by-step description. More visualizations are provided in the appendix: Fig. 9 shows the max of attention over all heads for all steps, Fig. 10 shows all steps of head 13, and Fig. 11 shows the corresponding gates.

arguments of MIN is not ready yet ([MIN 1 7 4 [MAX 2 4 0 8 9 ]]). We can see that only arguments which are already ready (1 7 4) attend to the operator (see the column of MIN). In step 1 ( $t = 1$ , 2nd row), we can see that head 5 copies the expected result of MAX, 9 to the column of the operator (we note that certainly this only requires one step as 9 is always the result of MAX when it is one of the arguments of MAX). Similarly in step 2, head 7 (2nd row) seems to copy the result of the second MED, 8 to the operator column. In step 3 ( $t = 3$ , 1st row), we can recognize that the result of MAX is marked as an argument for MIN in head 13 which is responsible for communication between operators and its arguments. This is shown by the new attention which appears at  $t = 3$  in head 13 from the source position MAX to the target position MIN (a pattern which is not visible at  $t = 2$ ). In head 3,  $t = 6$  (2nd row), the expected result of MIN, which is 1, is copied to the operator, similarly to the patterns we observed above for MAX and MED. In head 13,  $t = 6$  (1st row), we can observe

that all arguments for the first MED are also now recognized (the result of MIN which is 1, and 7). Finally in  $t = 7$  (2nd row), two heads, head 3 and head 5 seem to copy/gather two inputs needed to compute the corresponding median, 1 and 7, to the operator column. A complete visualization for further steps can be found in Appendix C.2. We noticed that some of the heads do not seem to play a key role. Here we focused on interpreting those which seem to participate in the main computation. For ListOps, we note that we also partially find the attention patterns described above in the baseline Transformer with relative positional encoding, at least on some examples we have inspected, which also explains its rather high accuracy.

# 5 DISCUSSION

Learning adaptive serialization. The TCF architecture could be also understood as doing adaptive serialization of the problem. A key requirement for reusable computation is the decomposition of the problem in reusable building blocks, typically in sequential steps. The granularity of the decomposition determines the reusability: fusing operations in a single step makes the processing faster (less steps), but also more specialized. Learning the most granular solutions is thus preferable for generalization. At the same time, not all processing should happen serially: branches of the computation graph that does not have common data dependencies can be processed independently in parallel, which we empirically observe in our TCF in the ListOps example (Sec. 4). This enables the architecture to get away with the number of computation steps reflecting the depth of the computation graph instead of the length of the input.

Bottom up approach for improving model architectures. Transformers have seen tremendous successes across various application domains (Devlin et al., 2019; Brown et al., 2020; Dosovitskiy et al., 2021). Impressive results have been reported when they are scaled up with a large amount of data (Brown et al., 2020). At the same time, there are simple tasks like those we highlight in this work, which demonstrate that the Transformer architecture still has to be improved. In fact, in algorithmic tasks, it is often the case that a sub-optimal choice of the architecture/optimization method makes the model fall back to memorization. We argue that it is crucial to look at isolated problems which test specific generalization capability. This calls for approaching the problem bottom up: building on toy tasks that focus on individual aspects of generalization, and using them for improving models.

# 6 RELATED WORK

Gating inside Transformers. Several prior works have proposed to use some form of gating inside Transformer architectures. However, the copy gate we propose is different from those previously proposed (Parisotto et al., 2020; Chaabouni et al., 2021) as it satisfies two important properties. First, our copy gate allows the model to skip the entire Transformer layer (i.e both the self-attention and the feedforward blocks) when the gate is closed. Second, the gate function is conditioned on the attention output such that the decision of opening or closing depends on the information from all columns. While Parisotto et al. (2020) have proposed multiple gating variants in the goal of stabilizing Transformers for reinforcement learning, none of the proposed approaches can produce this behavior. Empirically, we also tried out a few other gating variants which do not satisfy the two properties above; we found them not to improve over the regular Transformers in our preliminary experiments on compositional table lookup. Chaabouni et al. (2021) also make use of "gating" in Transformers. To be more specific, they use the gated linear unit activation function (Dauphin et al., 2017) commonly used in convolutional NNs. They import such an activation function to their Transformer models as it is part of the model (Dessi & Baroni, 2019) which has been reported to outperform RNN baselines on a systematic generalization task. Unlike our copy gate or Parisotto et al. (2020)'s gating, such a gating activation does not have the "residual" term (i.e. a closed gate zeros out the input) which allows the model to skip a transformation.

Parameterized geometric distributions. Two recent works (Brooks et al., 2021; Banino et al., 2021) have used a form of parameterized geometric distribution (PGD; in the form of Eq. 7). Brooks et al. (2021) have used such a geometric distribution to parameterize the movement of a pointer on a sequence of instructions. Banino et al. (2021) have used it to implement adaptive computation time

(Schmidhuber, 2012; Graves, 2016). We use the PGD to obtain a generic attention mechanism as a replacement to the standard self-attention used in Transformers (Vaswani et al., 2017).

Compositional table lookup. CTL has been proposed by as a task for evaluating compositional ability of neural networks (Liska et al., 2018). Previous works evaluated RNNs, RNNs with attention, and Transformer architectures on this task with limited success (Hupkes et al., 2019; Dubois et al., 2020). Dubois et al. (2020) have proposed a special attention mechanism to augment the recurrent architecture. While they obtain good performance on the forward presentation order, the proposed model has still failed in the backward presentation order. In contrast, two of the approaches we proposed (Sec. 3.1) achieve  $100\%$  generalization accuracy on this task in both presentation orders.

Positional encodings. Many previous works have focused on improving positional encoding for self-attention. Most notably, the relative positional encoding (Shaw et al., 2018; Dai et al., 2019) has been found useful in improving systematic generalization of Transformers (Csordás et al., 2021). In this work, we also present two new approaches related to positional encoding. One is the gated combination of absolute and relative positional encoding (Sec. 3.1; details in Appendix A). We show that absolute positional encoding can complement the relative positional encoding. It enables the model to always attend to a specific position, as is needed for CTL task in the last step, while the gating allows it to use relative positional encoding for other positions/steps. Second, we introduce directional encoding to augment the geometric attention. Unlike positional encoding which can overfit to a range of positions seen during training, the direction information is found to be robust and to be a crucial augmentation to the geometric attention.

# 7 CONCLUSION

We proposed a new view on the internal operations of Transformers, introducing a dynamic dataflow architecture between Transformer columns. This overcomes two shortcomings of traditional Transformers: the problem of keeping and routing data in unaltered fashion, which we solve by an additional copy gate, and the problem of learning length-independent attention patterns, which we solve by geometric attention. Our new model, TCF, generalizes to longer lengths on the popular compositional lookup table task in both forward and backward directions. TCF also achieves near perfect performance in simple arithmetic and ListOps tasks in settings which test systematic generalization in terms of computational depth. In general, the gates and the attention maps collectively make the architecture more interpretable compared to the baselines.

Reproducibility Statement. We will release all code and datasets used to produce the results reported in this work in a public repository. The corresponding code is already included in the supplemental material of this submission, including the code used to produce tables and plots presented in this work.

# REFERENCES

Dzmitry Bahdanau, Harm de Vries, Timothy J O'Donnell, Shikhar Murty, Philippe Beaudoin, Yoshua Bengio, and Aaron Courville. CLOSURE: Assessing systematic generalization of CLEVR models. In ViGIL workshop, NeurlPS, Vancouver, Canada, December 2019.  
Andrea Banino, Jan Balaguer, and Charles Blundell. Pondernet: Learning to ponder. Preprint arXiv:2107.05407, 2021.  
Ethan A. Brooks, Janarthanan Rajendran, Richard L. Lewis, and Satinder Singh. Reinforcement learning of implicit and explicit control flow instructions. In Proc. Int. Conf. on Machine Learning (ICML), pp. 1082-1091, Virtual only, July 2021.  
Tom B Brown et al. Language models are few-shot learners. In Proc. Advances in Neural Information Processing Systems (NeurIPS), Virtual only, December 2020.  
Rahma Chaabouni, Roberto Dessì, and Eugene Kharitonov. Can transformers jump around right in natural language? assessing performance transfer from scan. Preprint arXiv:2107.01366, 2021.

Xinyun Chen, Chen Liang, Adams Wei Yu, Dawn Song, and Denny Zhou. Compositional generalization via neural-symbolic stack machines. In Proc. Advances in Neural Information Processing Systems (NeurIPS), Virtual only, December 2020.  
Róbert Csordás and Jürgen Schmidhuber. Improving differentiable neural computers through memory masking, de-allocation, and link distribution sharpness control. In Int. Conf. on Learning Representations (ICLR), New Orleans, LA, USA, May 2019.  
Róbert Csordás, Kazuki Irie, and Jürgen Schmidhuber. The devil is in the detail: Simple tricks improve systematic generalization of transformers. In Proc. Conf. on Empirical Methods in Natural Language Processing (EMNLP), Punta Cana, Dominican Republic, November 2021.  
Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. In Proc. Association for Computational Linguistics (ACL), pp. 2978-2988, Florence, Italy, 2019.  
Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In Proc. Int. Conf. on Machine Learning (ICML), pp. 933-941, Sydney, Australia, August 2017.  
Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In Int. Conf. on Learning Representations (ICLR), New Orleans, LA, USA, May 2019.  
Roberto Dessi and Marco Baroni. CNNs found to jump around more skillfully than RNNs: Compositional generalization in seq2seq convolutional networks. In Proc. Association for Computational Linguistics (ACL), pp. 3919-3923, Florence, Italy, July 2019.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Proc. North American Chapter of the Association for Computational Linguistics on Human Language Technologies (NAACL-HLT), pp. 4171-4186, Minneapolis, MN, USA, June 2019.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In Int. Conf. on Learning Representations (ICLR), Virtual only, May 2021.  
Yann Dubois, Gautier Dagan, Dieuwke Hupkes, and Elia Bruni. Location attention for extrapolation to longer sequences. In Proc. Association for Computational Linguistics (ACL), pp. 403-413, Virtual only, July 2020.  
Jerry Fodor and Brian P McLaughlin. Connectionism and the problem of systematicity: Why smolensky's solution doesn't work. Cognition, 35(2):183-204, 1990.  
Jerry A Fodor, Zenon W Pylyshyn, et al. Connectionism and cognitive architecture: A critical analysis. Cognition, 28(1-2):3-71, 1988.  
Daniel Furrer, Marc van Zee, Nathan Scales, and Nathanael Schärli. Compositional generalization in semantic parsing: Pre-training vs. specialized architectures. Preprint arXiv:2007.08970, 2020.  
Alex Graves. Adaptive computation time for recurrent neural networks. In Int. Conf. on Learning Representations (ICLR) Workshop Track, Vancouver, Canada, April 2016.  
Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwinska, Sergio Gomez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John P. Agapiou, Adrià Puigdomènech Badia, Karl Moritz Hermann, Yori Zwols, Georg Ostrovski, Adam Cain, Helen King, Christopher Summerfield, Phil Blunsom, Koray Kavukcuoglu, and Demis Hassabis. Hybrid computing using a neural network with dynamic external memory. Nature, 538(7626): 471-476, 2016.  
Klaus Greff, Sjoerd van Steenkiste, and Jürgen Schmidhuber. On the binding problem in artificial neural networks. Preprint arXiv:2012.05208, 2020.

Stephen José Hanson. A stochastic version of the delta rule. Physica D: Nonlinear Phenomena, 42 (1-3):265-272, 1990.  
Serhii Havrylov, German Kruszewski, and Armand Joulin. Cooperative learning of disjoint syntax and semantics. In Proc. North American Chapter of the Association for Computational Linguistics on Human Language Technologies (NAACL-HLT), pp. 1118-1128, Minneapolis, USA, June 2019.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, Las Vegas, NV, USA, June 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, pp. 1735-1780, 1997.  
Dieuwke Hupkes, Anand Singh, Kris Korrel, German Kruszewski, and Elia Bruni. Learning compositionally through attentive guidance. In Proc. Int. Conf. on Computational Linguistics and Intelligent Text Processing, La Rochelle, France, April 2019.  
Dieuwke Hupkes, Verna Dankers, Mathijs Mul, and Elia Bruni. Compositionality decomposed: How do neural networks generalise? Journal of Artificial Intelligence Research, pp. 757-795, 2020.  
Kazuki Irie, Imanol Schlag, Róbert Csordás, and Jürgen Schmidhuber. Going beyond linear transformers with recurrent fast weight programmers. Preprint arXiv:2106.06295, 2021.  
Kris Korrel, Dieuwke Hupkes, Verna Dankers, and Elia Bruni. Transcoding compositionally: Using attention to find more generalizable solutions. In Proc. BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP, ACL, pp. 1-11, Florence, Italy, 2019.  
Brenden M Lake. Compositional generalization through meta sequence-to-sequence learning. In Proc. Advances in Neural Information Processing Systems (NeurIPS), pp. 9788-9798, Vancouver, Canada, December 2019.  
Brenden M. Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In Proc. Int. Conf. on Machine Learning (ICML), pp. 2873-2882, Stockholm, Sweden, July 2018.  
Yuanpeng Li, Liang Zhao, Jianyu Wang, and Joel Hestness. Compositional generalization for primitive substitutions. In Proc. Conf. on Empirical Methods in Natural Language Processing and Int.Joint Conf. on Natural Language Processing (EMNLP-IJCNLP), pp. 4292-4301, Hong Kong, China, November 2019.  
Adam Liska, German Kruszewski, and Marco Baroni. Memorize or generalize? searching for a compositional RNN in a haystack. In AEGAP Workshop ICML, Stockholm, Sweden, July 2018.  
Qian Liu, Shengnan An, Jian-Guang Lou, Bei Chen, Zeqi Lin, Yan Gao, Bin Zhou, Nanning Zheng, and Dongmei Zhang. Compositional generalization by learning analytical expressions. In Proc. Advances in Neural Information Processing Systems (NeurIPS), Virtual only, December 2020.  
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Int. Conf. on Learning Representations (ICLR), New Orleans, LA, USA, May 2019.  
Nikita Nangia and Samuel R. Bowman. Listops: A diagnostic dataset for latent tree learning. In Proc. North American Chapter of the Association for Computational Linguistics on Human Language Technologies (NAACL-HLT), pp. 92-99, New Orleans, USA, June 2018.  
Santiago Ontañón, Joshua Ainslie, Vaclav Cvicek, and Zachary Fisher. Making transformers solve compositional tasks. Preprint arXiv:2108.04378, 2021.  
Emilio Parisotto, H. Francis Song, Jack W. Rae, Razvan Pascanu, Caglar Gülçehre, Siddhant M. Jayakumar, Max Jaderberg, Raphaël Lopez Kaufman, Aidan Clark, Seb Noury, Matthew Botvinick, Nicolas Heess, and Raia Hadsell. Stabilizing transformers for reinforcement learning. In Proc. Int. Conf. on Machine Learning (ICML), volume 119, pp. 7487-7498, Virtual only, July 2020.

Jake Russian, Jason Jo, Randall C O'Reilly, and Yoshua Bengio. Compositional generalization in a deep seq2seq model by separating syntax and semantics. Preprint arXiv:1904.09708, 2019.  
Jürgen Schmidhuber. Self-delimiting neural networks. Technical Report IDSIA-08-12, arXiv:1210.0118v1, The Swiss AI Lab IDSIA, 2012.  
Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. In Proc. North American Chapter of the Association for Computational Linguistics on Human Language Technologies (NAACL-HLT), pp. 464–468, New Orleans, Louisiana, USA, June 2018.  
Peter Shaw, Ming-Wei Chang, Panupong Pasupat, and Kristina Toutanova. Compositional generalization and natural language variation: Can a semantic parsing approach handle both? Preprint arXiv:2010.12725, 2020.  
Yikang Shen, Shawn Tan, Seyed Arian Hosseini, Zhouhan Lin, Alessandro Sordoni, and Aaron C. Courville. Ordered memory. In Proc. Advances in Neural Information Processing Systems (NeurIPS), pp. 5038-5049, Vancouver, Canada, December 2019.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929-1958, 2014.  
Rupesh K Srivastava, Klaus Greff, and Jürgen Schmidhuber. Training very deep networks. In Proc. Advances in Neural Information Processing Systems (NIPS), pp. 2368-2376, Montreal, Canada, December 2015.  
Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for efficient transformers. In Int. Conf. on Learning Representations (ICLR), Virtual only, May 2021.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proc. Advances in Neural Information Processing Systems (NIPS), pp. 5998-6008, Long Beach, CA, USA, December 2017.  
Gail Weiss, Yoav Goldberg, and Eran Yahav. Thinking like Transformers. In Proc. Int. Conf. on Machine Learning (ICML), pp. 11080-11090, Virtual only, July 2021.
