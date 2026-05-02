# MONOTONIC CHUNKWISE ATTENTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Sequence-to-sequence models with soft attention have been successfully applied to a wide variety of problems, but their decoding process incurs a quadratic time and space cost and is inapplicable to real-time sequence transduction. To address these issues, we propose Monotonic Chunkwise Attention (MoChA), which adaptively splits the input sequence into small chunks over which soft attention is computed. We show that models utilizing MoChA can be trained efficiently with standard backpropagation while allowing online and linear-time decoding at test time. When applied to online speech recognition, we obtain state-of-the-art results and match the performance of a model using an offline soft attention mechanism. In document summarization experiments where we do not expect monotonic alignments, we show significantly improved performance compared to a baseline monotonic attention-based model.

# 1 INTRODUCTION

Sequence-to-sequence models (Sutskever et al., 2014; Cho et al., 2014) with a soft attention mechanism (Bahdanau et al., 2015) have been successfully applied to a plethora of sequence transduction problems (Luong et al., 2015; Xu et al., 2015; Chorowski et al., 2015; Wang et al., 2017; See et al., 2017). In their most familiar form, these models process an input sequence with an encoder recurrent neural network (RNN) to produce a sequence of hidden states, referred to as a memory. A decoder RNN then autoregressively produces the output sequence. At each output timestep, the decoder is directly conditioned by an attention mechanism, which allows the decoder to refer back to entries in the encoder's hidden state sequence. This use of the encoder's hidden states as a memory gives the model the ability to bridge long input-output time lags (Raffel & Ellis, 2015), which provides a distinct advantage over sequence-to-sequence models lacking an attention mechanism (Bahdanau et al., 2015). Furthermore, visualizing where in the input the model was attending to at each output timestep produces an input-output alignment which provides valuable insight into the model's behavior.

As originally defined, soft attention inspects every entry of the memory at each output timestep, effectively allowing the model to condition on any arbitrary input sequence entry. This flexibility comes at a distinct cost, namely that decoding with a soft attention mechanism has a quadratic time and space cost  $\mathcal{O}(TU)$ , where  $T$  and  $U$  are the input and output sequence lengths respectively. This precludes its use on very long sequences, e.g. summarizing extremely long documents. In addition, because soft attention considers the possibility of attending to every entry in the memory at every output timestep, it must wait until the input sequence has been processed before producing output. This makes it inapplicable to real-time sequence transduction problems. Raffel et al. (2017) recently pointed out that these issues can be mitigated when the input-output alignment is monotonic, i.e. the correspondence between elements in the input and output sequence does not involve reordering. This property is present in various real-world problems, such as speech recognition and synthesis, where the input and output share a natural temporal order (see, for example, fig. 2). In other settings, the alignment only involves local reorderings, e.g. machine translation for certain language pairs (Birch et al., 2008).

Based on this observation, Raffel et al. (2017) introduced an attention mechanism that explicitly enforces a hard monotonic input-output alignment, which allows for online and linear-time decoding. However, the hard monotonicity constraint also limits the expressivity of the model compared to soft attention (which can induce an arbitrary soft alignment). Indeed, experimentally it was shown

![](images/b887451db21ae3fcda226beee04746c3e96d428308c25479b360981674445f9b.jpg)  
(a) Soft attention.

![](images/efecf21c437c5ca155a4a3604ec9701ff4efaf0d246c1b62118807619f9e301c.jpg)  
(b) Hard monotonic attention.  
(c) Monotonic chunkwise attention.  
Figure 1: Schematics of the attention mechanisms discussed in this paper. Each node represents the possibility of the model attending to a given memory entry (horizontal axis) at a given output timestep (vertical axis). (a) In soft attention, the model assigns a probability (represented by the shade of gray of each node) to each memory entry at each output timestep. The context vector is computed as the weighted average of the memory, weighted by these probabilities. (b) At test time, monotonic attention inspects memory entries from left-to-right, choosing whether to move on to the next memory entry (shown as nodes with  $\times$ ) or stop and attend (shown as black nodes). The context vector is hard-assigned to the memory entry that was attended to. At the next output timestep, it starts again from where it left off. (c) MoChA utilizes a hard monotonic attention mechanism to choose the endpoint (shown as nodes with bold borders) of the chunk over which it attends. The chunk boundaries (here, with a window size of 3) are shown as dotted lines. The model then performs soft attention (with attention weighting shown as the shade of gray) over the chunk, and computes the context vector as the chunk's weighted average.

![](images/8f7bdf51436b71b6a039a62090a07598b6c208a106bcc01982a0fb006156c3c1.jpg)

that the performance of sequence-to-sequence models utilizing this monotonic attention mechanism lagged behind that of standard soft attention.

In this paper, we aim to close this gap by introducing a novel attention mechanism which retains the online and linear-time benefits of hard monotonic attention while allowing for soft alignments. Our approach, which we dub "Monotonic Chunkwise Attention" (MoChA), allows the model to perform soft attention over small chunks of the memory preceding where a hard monotonic attention mechanism has chosen to attend. It also has a training procedure which allows it to be straightforwardly applied to existing sequence-to-sequence models and trained with standard backpropagation. We show experimentally that MoChA effectively closes the gap between monotonic and soft attention on online speech recognition and provides a  $20\%$  relative improvement over monotonic attention on document summarization (a task which does not exhibit monotonic alignments). These benefits incur only a modest increase in the number of parameters and computational cost. We also provide a discussion of related work and ideas for future research using our proposed mechanism.

# 2 DEFINING MOCHA

To develop our proposed attention mechanism, we will first review the sequence-to-sequence framework and the most common form of soft attention used with it. Because MoChA can be considered a generalization of monotonic attention, we then re-derive this approach and point out some of its shortcomings. From there, we show how soft attention over chunks can be straightforwardly added to hard monotonic attention, giving us the MoChA attention mechanism. We also show how MoChA can be trained efficiently with respect to the mechanism's expected output, which allows us to use standard backpropagation.

# 2.1 SEQUENCE-TO-SEQUENCE MODELS

A sequence-to-sequence model is one which transduces an input sequence  $\mathbf{x} = \{x_{1},\dots,x_{T}\}$  to an output sequence (potentially of a different modality)  $\mathbf{y} = \{y_{1},\dots,y_{U}\}$ . Typically, the input sequence is first converted to a sequence of hidden states  $\mathbf{h} = \{h_1,\dots,h_T\}$  by an encoder recurrent neural network (RNN):

$$
h _ {j} = \operatorname {E n c o d e r R N N} \left(x _ {j}, h _ {j - 1}\right) \tag {1}
$$

A decoder RNN then updates its hidden state autoregressively and an output layer (typically using a softmax nonlinearity) produces the output sequence:

$$
s _ {i} = \operatorname {D e c o d e r R N N} \left(y _ {i - 1}, s _ {i - 1}, c _ {i}\right) \tag {2}
$$

$$
y _ {i} = \operatorname {O u t p u t} \left(s _ {i}, c _ {i}\right) \tag {3}
$$

where  $s_i$  is the decoder's state and  $c_i$  is a "context" vector which is computed as a function of the encoder hidden state sequence  $\mathbf{h}$ . Note that  $c_i$  is the sole conduit through which the decoder has access to information about the input sequence.

In the originally proposed sequence-to-sequence framework (Sutskever et al., 2014), the context vector is simply set to the final encoder hidden state, i.e.  $c_{i} = h_{T}$ . It was subsequently found that this approach exhibits degraded performance when transducing long sequences (Bahdanau et al., 2015). Instead, it has become standard to use an attention mechanism which treats the hidden state sequence as a (soft-)addressable memory whose entries are used to compute the context vector  $c_{i}$ . In the following subsections, we discuss three such approaches for computing  $c_{i}$ ; otherwise, the sequence-to-sequence framework remains unchanged.

# 2.2 STANDARD SOFT ATTENTION

Currently, the most commonly used attention mechanism is the one originally proposed in (Bahdanau et al., 2015). At each output timestep  $i$ , this approach proceeds as follows: First, an unnormalized scalar "energy" value  $e_{i,j}$  is produced for each memory entry:

$$
e _ {i, j} = \operatorname {E n e r g y} \left(h _ {j}, s _ {i - 1}\right) \tag {4}
$$

A common choice for Energy  $(\cdot)$  is

$$
\operatorname {E n e r g y} \left(h _ {j}, s _ {i - 1}\right) := v ^ {\top} \tanh  \left(W _ {h} h _ {j} + W _ {s} s _ {i - 1} + b\right) \tag {5}
$$

where  $W_{h} \in \mathbb{R}^{d \times \dim(h_{j})}$ ,  $W_{s} \in \mathbb{R}^{d \times \dim(s_{i-1})}$ ,  $b \in \mathbb{R}^{d}$  and  $v \in \mathbb{R}^{d}$  are learnable parameters and  $d$  is the hidden dimensionality of the energy function. Second, these energy scalars are normalized across the memory using the softmax function to produce weighting values  $\alpha_{i,j}$ :

$$
\alpha_ {i, j} = \frac {\exp \left(e _ {i , j}\right)}{\sum_ {k = 1} ^ {T} \exp \left(e _ {i , k}\right)} = \operatorname {s o f t m a x} \left(e _ {i,:}\right) _ {j} \tag {6}
$$

Finally, the context vector is computed as a simple weighted average of  $\mathbf{h}$ , weighted by  $\alpha_{i,:}$ :

$$
c _ {i} = \sum_ {j = 1} ^ {T} \alpha_ {i, j} h _ {j} \tag {7}
$$

We visualize this soft attention mechanism in fig. 1a.

Note that in order to compute  $c_{i}$  for any output timestep  $i$ , we need to have computed all of the encoder hidden states  $h_{j}$  for  $j \in \{1, \dots, T\}$ . This implies that this form of attention is not applicable to online/real-time sequence transduction problems, because it needs to have observed the entire input sequence before producing any output. Furthermore, producing each context vector  $c_{i}$  involves computing  $T$  energy scalar terms and weighting values. While these operations can typically be parallelized, this nevertheless results in decoding having a  $\mathcal{O}(TU)$  cost in time and space.

# 2.3 MONOTONIC ATTENTION

To address the aforementioned issues with soft attention, Raffel et al. (2017) proposed a hard monotonic attention mechanism whose attention process can be described as follows: At output timestep  $i$ , the attention mechanism begins inspecting memory entries starting at the memory index it is intended to be at the previous output timestep, referred to as  $t_{i-1}$ . It then computes an unnormalized energy scalar  $e_{i,j}$  for  $j = t_{i-1}, t_{i-1} + 1, \ldots$  and passes these energy values into a logistic sigmoid function  $\sigma(\cdot)$  to produce "selection probabilities"  $p_{i,j}$ . Then, a discrete attend/don't attend decision  $z_{i,j}$  is sampled from a Bernoulli random variable parameterized by  $p_{i,j}$ . In total, so far we have

$$
e _ {i, j} = \text {M o n o t o n i c E n e r g y} \left(s _ {i - 1}, h _ {j}\right) \tag {8}
$$

$$
p _ {i, j} = \sigma \left(e _ {i, j}\right) \tag {9}
$$

$$
z _ {i, j} \sim \operatorname {B e r n o u l l i} \left(p _ {i, j}\right) \tag {10}
$$

As soon as  $z_{i,j} = 1$  for some  $j$ , the model stops and sets  $t_i = j$  and  $c_{i} = h_{t_{i}}$ . This process is visualized in fig. 1b. Note that because this attention mechanism only makes a single pass over the memory, it has a  $\mathcal{O}(\max (T,U))$  (linear) cost. Further, in order to attend to memory entry  $h_j$ , the encoder RNN only needs to have processed input sequence entries  $x_{1},\ldots ,x_{j}$ , which allows it to be used for online sequence transduction. Finally, note that if  $p_{i,j}\in \{0,1\}$  (a condition which is encouraged, as discussed below) then the greedy assignment of  $c_{i} = h_{t_{i}}$  is equivalent to marginalizing over possible alignment paths.

Because this attention process involves sampling and hard assignment, models utilizing hard monotonic attention can't be trained with backpropagation. To remedy this, Raffel et al. (2017) propose training with respect to the expected value of  $c_{i}$  by computing the probability distribution over the memory induced by the attention process. This distribution takes the following form:

$$
\alpha_ {i, j} = p _ {i, j} \left((1 - p _ {i, j - 1}) \frac {\alpha_ {i , j - 1}}{p _ {i , j - 1}} + \alpha_ {i - 1, j}\right) \tag {11}
$$

The context vector  $c_{i}$  is then computed as a weighted sum of the memory as in eq. (7). Equation (11) can be explained by observing that  $(1 - p_{i,j-1})\alpha_{i,j-1} / p_{i,j-1}$  is the probability of attending to memory entry  $j-1$  at the current output timestep  $(\alpha_{i,j-1})$  corrected for the fact that the model did not attend to memory entry  $j$  (by multiplying by  $(1 - p_{i,j-1})$  and dividing by  $p_{i,j-1}$ ). The addition of  $\alpha_{i-1,j}$  represents the additional possibility that the model attended to entry  $j$  at the previous output timestep, and finally multiplying it all by  $p_{i,j}$  reflects the probability that the model selected memory item  $j$  at the current output timestep  $i$ . Note that this recurrence relation is not parallelizable across memory indices  $j$  (unlike, say, softmax), but fortunately substituting  $q_{i,j} = \alpha_{i,j} / p_{i,j}$  produces the first-order linear difference equation  $q_{i,j} = (1 - p_{i,j-1})q_{i,j-1} + \alpha_{i-1,j}$  which has the following solution (Kelley & Peterson, 2001):

$$
q _ {i,:} = \operatorname {c u m p r o d} \left(1 - p _ {i,:}\right) \operatorname {c u m s u m} \left(\frac {\alpha_ {i - 1 , :}}{\operatorname {c u m p r o d} \left(1 - p _ {i , :}\right)}\right) \tag {12}
$$

where  $\operatorname{cumprod}(\mathbf{x}) = [1, x_1, x_1x_2, \ldots, \prod_i^{|x| - 1}x_i]$  and  $\operatorname{cumsum}(\mathbf{x}) = [x_1, x_1 + x_2, \ldots, \sum_i^{|x|}x_i]$ . Because the cumulative sum and product can be computed in parallel (Ladner & Fischer, 1980), models can still be trained efficiently with this approach.

Note that training is no longer online or linear-time, but the proposed solution is to use this "soft" monotonic attention for training and use the hard monotonic attention process at test time. To encourage discreteness, Raffel et al. (2017) used the common approach of adding zero-mean, unit-variance Gaussian noise to the logistic sigmoid function's activations, which causes the model to learn to produce effectively binary  $p_{i,j}$ . If  $p_{i,j}$  are binary,  $z_{i,j} = \mathbb{1}(p_{i,j} > .5)$ , so in practice sampling is eschewed at test-time in favor of simple thresholding. Separately, it was observed that switching from the softmax nonlinearity to the logistic sigmoid resulted in optimization issues due to saturating and sensitivity to offset. To mitigate this, a slightly modified energy function was used:

$$
\text {M o n o t o n i c E n e r g y} \left(s _ {i - 1}, h _ {j}\right) = g \frac {v ^ {\top}}{\| v \|} \tanh  \left(W _ {s} s _ {i - 1} + W _ {h} h _ {j} + b\right) + r \tag {13}
$$

where  $g, r$  are learnable scalars and  $v, W_s, W_h, b$  are as in eq. (5). Further discussion of these modifications is provided in (Raffel et al., 2017) appendix G.

# 2.4 MONOTONIC CHUNKWISE ATTENTION

While hard monotonic attention provides online and linear-time decoding, it nevertheless imposes two significant constraints on the model: First, that the decoder can only attend to a single entry in memory at each output timestep, and second, that the input-output alignment must be strictly monotonic. These constraints are in contrast to standard soft attention, which allows a potentially arbitrary and smooth input-output alignment. Experimentally, it was shown that performance degrades somewhat on all tasks tested in (Raffel et al., 2017). Our hypothesis is that this degradation stems from the aforementioned constraints imposed by hard monotonic attention.

Algorithm 1 MoChA decoding process (test time). During training, lines 4-19 are replaced with eqs. (20) to (26) and  $y_{i - 1}$  is replaced with the ground-truth output at timestep  $i - 1$ .  
1: Input: memory h of length  $T$ , chunk size  $w$   
2: State:  $s_0 = 0, t_0 = 1, i = 1, y_0 = \text{StartOfSequence}$   
3: while  $y_{i-1} \neq \text{EndOfSequence do}$  // Produce output tokens until end-of-sequence token is produced  
4: for  $j = t_{i-1}$  to  $T$  do // Start inspecting memory entries  $h_j$  left-to-right from where we left off  
5:  $e_{i,j} = \text{MonotonicEnergy}(s_{i-1}, h_j)$  // Compute attention energy for  $h_j$   
6:  $p_{i,j} = \sigma(e_{i,j})$  // Compute probability of choosing  $h_j$   
7: if  $p_{i,j} \geq 0.5$  then // If  $p_{i,j}$  is larger than 0.5, we stop scanning the memory  
8:  $v = j - w + 1$  // Set chunk start location  
9: for  $k = v$  to  $j$  do // Compute chunkwise softmax energies over a size- $w$  chunk before  $j$   
10:  $u_{i,k} = \text{ChunkEnergy}(s_{i-1}, h_k)$   
11: end for  
12:  $c_i = \sum_{k=v}^{j} \frac{\exp(u_{i,k})}{\sum_{l=v}^{j} \exp(u_{i,l})} h_k$  // Compute softmax-weighted average over the chunk  
13:  $t_i = j$  // Remember where we left off for the next output timestep  
14: break // Stop scanning the memory  
15: end if  
16: end for  
17: if  $p_{i,j} < 0.5, \forall j \in \{t_{i-1}, t_{i-1} + 1, \ldots, T\}$  then  
18:  $c_i = \vec{0}$  // If we scanned the entire memory without stopping, set  $c_i$  to a vector of zeros  
19: end if  
20:  $s_i = \text{DecoderRNN}(s_{i-1}, y_{i-1}, c_i)$  // Update output RNN state based on the new context vector  
21:  $y_i = \text{Output}(s_i, c_i)$  // Output a new symbol using the softmax output layer  
22:  $i = i + 1$   
23: end while

To remedy these issues, we propose a novel attention mechanism which we call MoChA, for Monotonic Chunkwise Attention. The core of our idea is to allow the attention mechanism to perform soft attention over small "chunks" of memory preceding where a hard monotonic attention mechanism decides to stop. This facilitates some degree of softness in the input-output alignment, while retaining the online decoding and linear-time complexity benefits.

At test time, we follow the hard monotonic attention process of section 2.3 in order to determine  $t_i$  (the location where the hard monotonic attention mechanism decides to stop scanning the memory at output timestep  $i$ ). However, instead of setting  $c_i = h_{t_i}$ , we allow the model to perform soft attention over the length- $w$  window of memory entries preceding and including  $t_i$ :

$$
v = t _ {i} - w + 1 \tag {14}
$$

$$
u _ {i, k} = \operatorname {C h u n k E n e r g y} \left(s _ {i - 1}, h _ {k}\right), k \in \{v, v + 1, \dots , t _ {i} \} \tag {15}
$$

$$
c _ {i} = \sum_ {k = v} ^ {t _ {i}} \frac {\exp \left(u _ {i , k}\right)}{\sum_ {l = v} ^ {t _ {i}} \exp \left(u _ {i , l}\right)} h _ {k} \tag {16}
$$

where  $\mathrm{ChunkEnergy}(\cdot)$  is an energy function analogous to eq. (5), which is distinct from the MonotonicEnergy  $(\cdot)$  function. MoChA's attention process is visualized in fig. 1c. Note that MoChA allows for nonmonotonic alignments; specifically, it allows for reordering of the memory entries  $h_v,\ldots ,h_{t_i}$ . Including soft attention over chunks only increases the runtime complexity by the constant factor  $w$ , and decoding can still proceed in an online fashion. Furthermore, using MoChA only incurs a modest increase in the total number of parameters (corresponding to adding the second attention energy function  $\mathrm{ChunkEnergy}(\cdot)$ ). For example, in the speech recognition experiments described in section 3.1, the total number of model parameters only increased by about  $1\%$ . Finally, we point out that setting  $w = 1$  recovers hard monotonic attention. For completeness, we show the decoding algorithm for MoChA in full in algorithm 1.

During training, we proceed in a similar fashion as with monotonic attention, namely training the model using the expected value of  $c_{i}$  based on MoChA's induced probability distribution (which we denote  $\beta_{i,j}$ ). This can be computed as

$$
\beta_ {i, j} = \sum_ {k = j} ^ {j + w - 1} \left(\alpha_ {i, k} \exp \left(u _ {i, j}\right) / \sum_ {l = k - w + 1} ^ {k} \exp \left(u _ {i, l}\right)\right) \tag {17}
$$

The sum over  $k$  reflects the possible positions at which the monotonic attention could have stopped scanning the memory in order to contribute probability to  $\beta_{i,j}$  and the term inside the summation represents the softmax probability distribution over the chunk, scaled by the monotonic attention probability  $\alpha_{i,k}$ . Computing each  $\beta_{i,j}$  in this fashion is expensive due to the nested summation. Fortunately, there is an efficient way to compute  $\beta_{i,j}$  for  $j \in \{1,\dots,T\}$  in parallel: First, for a sequence  $\mathbf{x} = \{x_1,\ldots ,x_T\}$  we define

$$
\operatorname {M o v i n g S u m} (\mathbf {x}, b, f) _ {n} := \sum_ {m = n - (b - 1)} ^ {n + f - 1} x _ {m} \tag {18}
$$

This function can be computed efficiently, for example, by convolving  $\mathbf{x}$  with a length-  $(f + b - 1)$  sequence of 1s and truncating appropriately. Now, we can compute  $\beta_{i}$ : efficiently as

$$
\beta_ {i,:} = \exp \left(u _ {i,:}\right) \text {M o v i n g S u m} \left(\frac {\alpha_ {i , :}}{\text {M o v i n g S u m} \left(\exp \left(u _ {i , :}\right) , w , 1\right)}, 1, w\right) \tag {19}
$$

Putting it all together produces the following algorithm for computing  $c_{i}$  during training:

$$
e _ {i, j} = \text {M o n o t o n i c E n e r g y} \left(s _ {i - 1}, h _ {j}\right) \tag {20}
$$

$$
\epsilon \sim \mathcal {N} (0, 1) \tag {21}
$$

$$
p _ {i, j} = \sigma \left(e _ {i, j} + \epsilon\right) \tag {22}
$$

$$
\alpha_ {i,:} = p _ {i,:} \operatorname {c u m p r o d} \left(1 - p _ {i,:}\right) \operatorname {c u m s u m} \left(\frac {\alpha_ {i - 1 , :}}{\operatorname {c u m p r o d} \left(1 - p _ {i , :}\right)}\right) \tag {23}
$$

$$
u _ {i, j} = \operatorname {C h u n k E n e r g y} \left(s _ {i - 1}, h _ {j}\right) \tag {24}
$$

$$
\beta_ {i,:} = \exp \left(u _ {i,:}\right) \text {M o v i n g S u m} \left(\frac {\alpha_ {i , :}}{\text {M o v i n g S u m} \left(\exp \left(u _ {i , :}\right) , w , 1\right)}, 1, w\right) \tag {25}
$$

$$
c _ {i} = \sum_ {j = 1} ^ {T} \beta_ {i, j} h _ {j} \tag {26}
$$

Equations (20) to (23) reflect the (unchanged) computation of the monotonic attention probability distribution, eqs. (24) and (25) compute MoChA's probability distribution, and finally eq. (26) computes the expected value of the context vector  $c_{i}$ . In summary, we have developed a novel attention mechanism which allows computing soft attention over small chunks of the memory, whose locations are set adaptively. This mechanism has an efficient training-time algorithm and enjoys online and linear-time decoding at test time. We attempt to quantify the resulting speedup compared to soft attention with a synthetic benchmark in appendix B.

# 3 EXPERIMENTS

To test out MoChA, we applied it to two exemplary sequence transduction tasks: Online speech recognition and document summarization. Speech recognition is a promising setting for MoChA because it induces a naturally monotonic input-output alignment, and because online decoding is often required in the real world. Document summarization, on the other hand, does not exhibit a monotonic alignment, and we mostly include it as a way of testing the limitations of our model. We emphasize that in all experiments, we took a strong baseline sequence-to-sequence model with standard soft attention and changed only the attention mechanism; all hyperparameters, model structure, training approach, etc. were kept exactly the same. This allows us to isolate the effective difference in performance caused by switching to MoChA. Of course, this may be an artificially low estimate of the best-case performance of MoChA, due to the fact that it may benefit from a somewhat different hyperparameter setting. We leave eking out the best-case performance for future work.

Specifically, for MoChA we used eq. (13) for both the MonotonicEnergy and the ChunkEnergy functions. Following (Raffel et al., 2017), we initialized  $g = 1 / \sqrt{d}$  ( $d$  being the attention energy function hidden dimension) and tuned initial values for  $r$  based on validation set performance, using  $r = -4$  for MoChA on speech recognition,  $r = 0$  for MoChA on summarization, and  $r = -1$  for our monotonic attention baseline on summarization. We similarly tuned the chunk size  $w$ : For

speech recognition, we were surprised to find that all of  $w \in \{2, 3, 4, 6, 8\}$  performed comparably and thus chose the smallest value of  $w = 2$ . For summarization, we found  $w = 8$  to work best. We demonstrate empirically that even these small window sizes give a significant boost over hard monotonic attention ( $w = 1$ ) while incurring only a minor computational penalty. In all experiments, we report metrics on the test set at the training step of best performance on a validation set.

# 3.1 ONLINE SPEECH RECOGNITION

![](images/bc9f2feb16d0b3a2537c57dde3c29da8dbab7e066d324bb0dd39ed1c775d2296.jpg)

![](images/b142162f89142e17ad53561b623bbe482f913fac5b9931695fde76c7e4cf119d.jpg)

![](images/a582642ccdd0b0459c0bf862cb8fd6f8816482b17027276d2dd7bf2f6b09a849.jpg)

![](images/3acab97634668e5c38ca00325c80c5ecd96492e2f6120444b66a531d4bf1a952.jpg)  
Figure 2: Attention alignments plots and speech utterance feature sequence for the speech recognition task.

First, we apply MoChA in its natural setting, i.e. a domain where we expect roughly monotonic alignments: $^{1}$  Online speech recognition on the Wall Street Journal (WSJ) corpus (Paul & Baker, 1992). The goal in this task is to produce the sequence of words spoken in a recorded speech utterance. In this setting, RNN-based models must be unidirectional in order to satisfy the online requirement. We use the model of (Raffel et al., 2017), which is itself based on that of (Zhang et al., 2016). Full model and training details are provided in appendix A.1, but as a broad overview, the network ingests the spoken utterance as a mel-filterbank spectrogram which is passed to an encoder consisting of convolution layers, convolutional LSTM layers, and unidirectional LSTM layers. The decoder is a single unidirectional LSTM, which attends to the encoder state sequence via either MoChA or a standard soft attention mechanism. The decoder produces a sequence of distributions over character and word-delimiter tokens. Performance is measured in terms of word error rate (WER) after segmenting characters output by the model into words based on the produced word delimiter tokens. None of the models we report integrated a separate language model.

We show the results of our experiments, along with those obtained by prior work, in table 1. MoChA was able to beat the state-of-the-art by a large margin (20% relative). Because the performance of MoChA and the soft attention baseline was so close, we ran 8 repeat trials for both attention mechanisms and report the best, average, and standard deviation of word error rates across these trials. We found MoChA-based models to have slightly higher variance across trials, which resulted in it having a lower best WER but a slightly higher mean WER compared to soft attention (though the difference in means was not statistically significant for  $N = 8$  under an unpaired Student's t-test). This is the first time, to our knowledge, that an online attention mechanism matched the performance of standard (offline) soft attention. To get an idea of the behavior of the different attention mechanisms, we show attention alignments for an example from the WSJ validation set in fig. 2. As expected, the alignment looks roughly the same for all attention mechanisms. We note especially that MoChA is indeed taking advantage of the opportunity to produce a soft attention distribution over each length-2 chunk.

Since we empirically found the small value of  $w = 2$  to be sufficient to realize these gains, we carried out a few additional experiments to confirm that they can indeed be attributed to MoChA. First, the use of a second independent attention energy function

ChunkEnergy(\cdot) incurs a modest increase in parameter count – about  $1\%$  in our speech recognition model. To ensure the improved performance was not due to this parameter increase, we also re-trained the monotonic attention baseline with an energy function with a doubled hidden dimensionality (which produces a comparable increase in the number of parameters in a natural way).

<table><tr><td colspan="2">Prior Result</td><td>WER</td></tr><tr><td colspan="2">(Raffel et al., 2017) (CTC baseline)</td><td>33.4%</td></tr><tr><td colspan="2">(Luo et al., 2016) (Reinforcement Learning)</td><td>27.0%</td></tr><tr><td colspan="2">(Wang et al., 2016) (CTC)</td><td>22.7%</td></tr><tr><td colspan="2">(Raffel et al., 2017) (Monotonic Attention)</td><td>17.4%</td></tr><tr><td>Attention Mechanism</td><td>Best WER</td><td>Average WER</td></tr><tr><td>Soft Attention (offline)</td><td>14.2%</td><td>14.6 ± 0.3%</td></tr><tr><td>MoChA, w = 2</td><td>13.9%</td><td>15.0 ± 0.6%</td></tr></table>

Table 1: Word error rate on the Wall Street Journal test set. Our results (bottom) reflect the statistics of 8 trials.  

<table><tr><td>Mechanism</td><td>R-1</td><td>R-2</td></tr><tr><td>Soft Attention (offline)</td><td>39.11</td><td>15.76</td></tr><tr><td>Hard Monotonic Attention</td><td>31.14</td><td>11.16</td></tr><tr><td>MoChA, w = 8</td><td>35.46</td><td>13.55</td></tr></table>

Table 2: ROUGE F-scores for document summarization on the CNN/Daily Mail dataset. The soft attention baseline is our reimplementation of (See et al., 2017).

Across eight trials, the difference in performance (a decrease of  $0.3\%$  WER) was not significant compared to the baseline and was dwarfed by the gains achieved by MoChA. We also trained the  $w = 2$  MoChA model with half the attention energy hidden dimensionality (which similarly reconciles the parameter difference) and found it did not significantly undercut our gains, increasing the WER by only  $0.2\%$  (not significant over eight trials). Separately, one possible benefit of MoChA is that the attention mechanism can access a larger window of the input when producing the context vectors. An alternative approach towards this end would be to increase the temporal receptive field of the convolutional front-end, so we also re-trained the monotonic attention baseline with this change. Again, the difference in performance (an increase of  $0.3\%$  WER) was not significant over eight trials. These additional experiments reinforce the benefits of using MoChA for online speech recognition.

# 3.2 DOCUMENT SUMMARIZATION

Having proven the effectiveness of MoChA in the comfortable setting of speech recognition, we now test its limits in a task without a monotonic input/output alignment. Raffel et al. (2017) experimented with sentence summarization on the Gigaword dataset, which frequently exhibits monotonic alignments and involves short sequences (sentence-length sequences of words). They were able to achieve only slightly degraded performance with hard monotonic attention compared to a soft attention baseline. As a result, we turn to a more difficult task where hard monotonic attention struggles more substantially due to the lack of monotonic alignments: Document summarization on the CNN/Daily Mail corpus (Nallapati et al., 2016). While we primarily study this problem because it has the potential to be challenging, online and linear-time attention could also be beneficial in real-world scenarios where very long bodies of text need to be summarized as they are being created (e.g. producing a summary of a speech as it is being given).

The goal of this task is to produce a sequence of "highlight" sentences from a news article. As a baseline model, we chose the "pointer-generator" network (without the coverage penalty) of (See et al., 2017). For full model architecture and training details, refer to appendix A.2. As a brief summary, input words are converted to a learned embedding and passed into the model's encoder, consisting of a single bidirectional LSTM layer. The decoder is a unidirectional LSTM with an attention mechanism whose state is passed to a softmax layer which produces a sequence of distributions over the vocabulary. The model is augmented with a copy mechanism, which interpolates linearly between using the softmax output layer's word distribution, or a distribution of word IDs weighted by the attention distribution at a given output timestep. We tested this model with standard soft attention (as used in (See et al., 2017)), hard monotonic attention, and MoChA with  $w = 8$ .

The results are shown in table 2. We found that using a hard monotonic attention mechanism degraded performance substantially (nearly 8 ROUGE-1 points), likely because of the strong reordering required by this task. However, MoChA was able to effectively halve the gap between monotonic and soft attention, despite using the modest chunk size of  $w = 8$ . We consider this an encouraging indication of the benefits of being able to deal with local reorderings.

# 4 RELATED WORK

A similar model to MoChA is the "Neural Transducer" (Jaitly et al., 2015), where the input sequence is pre-segmented into equally-sized non-overlapping chunks and attentive sequence-to-sequence transduction is performed over each chunk separately. The full output sequence is produced by marginalizing out over possible end-of-sequence locations for the sequences generated from each chunk. While our model also performs soft attention over chunks, the locations of our chunks are set adaptively by a hard monotonic attention mechanism rather than fixed, and it avoids the marginalization over chunkwise end-of-sequence tokens.

Chorowski et al. (2015) proposes a similar idea, wherein the range over which soft attention is computed at each output timestep is limited to a fixed-sized window around the memory index of maximal attention probability at the previous output timestep. While this also produces soft attention over chunks, our approach differs in that the chunk boundary is set by an independent hard monotonic attention mechanism. This difference resulted in Chorowski et al. (2015) using a very large chunk size of 150, which effectively prevents its use in online settings and incurs a significantly higher computational cost than our approach which only required small values for  $w$ .

A related class of non-attentive sequence transduction models which can be used in online settings are connectionist temporal classification (Graves et al., 2006), the RNN transducer (Graves, 2012), segment-to-segment neural transduction (Yu et al., 2016), and the segmental RNN (Kong et al., 2015). These models are distinguished from sequence-to-sequence models with attention mechanisms by the fact that the decoder does not condition directly on the input sequence, and that decoding is done via a dynamic program. A detailed comparison of this class of approaches and attention-based models is provided in (Prabhavalkar et al., 2017), where it is shown that attention-based models perform best in speech recognition experiments. Further, Hori et al. (2017) recently proposed jointly training a speech recognition model with both a CTC loss and an attention mechanism. This combination encouraged the model to learn monotonic alignments, but Hori et al. (2017) still used a standard soft attention mechanism which precludes the model's use in online settings.

Finally, we note that there have been a few other works considering hard monotonic alignments, e.g. using reinforcement learning (Zaremba & Sutskever, 2015; Luo et al., 2016; Lawson et al., 2017), by using separately-computed target alignments (Aharoni & Goldberg, 2016) or by assuming a strictly diagonal alignment (Luong et al., 2015). We suspect that these approaches may confer similar benefits from adding chunkwise attention.

# 5 CONCLUSION

We have proposed MoChA, an attention mechanism which performs soft attention over adaptively-located chunks of the input sequence. MoChA allows for online and linear-time decoding, while also facilitating local input-output reordering. Experimentally, we showed that MoChA obtains state-of-the-art performance on an online speech recognition task, and that it substantially outperformed a hard monotonic attention-based model on document summarization. In future work, we are interested in applying MoChA to additional problems with (approximately) monotonic alignments, such as speech synthesis (Wang et al., 2017) and morphological inflection (Aharoni & Goldberg, 2016). We would also like to investigate ways to allow the chunk size  $w$  to also vary adaptively.

# REFERENCES

Martin Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G. Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: A system for large-scale machine learning. In Operating Systems Design and Implementation, 2016.  
Roee Aharoni and Yoav Goldberg. Sequence to sequence transduction with hard monotonic attention. arXiv preprint arXiv:1611.01487, 2016.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In International Conference on Learning Representations, 2015.

Alexandra Birch, Miles Osborne, and Philipp Koehn. Predicting success in machine translation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pp. 745-754. Association for Computational Linguistics, 2008.  
William Chan, Navdeep Jaitly, Quoc V. Le, and Oriol Vinyals. Listen, attend and spell: A neural network for large vocabulary conversational speech recognition. In International Conference on Acoustics, Speech and Signal Processing, 2016.  
Kyunghyun Cho, Bart van Merrienboer, Caglar Gulçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Conference on Empirical Methods in Natural Language Processing, 2014.  
Jan Chorowski and Navdeep Jaitly. Towards better decoding and language model integration in sequence to sequence models. arXiv preprint arXiv:1612.02695, 2017.  
Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio. Attention-based models for speech recognition. In Conference on Neural Information Processing Systems, 2015.  
Alex Graves. Sequence transduction with recurrent neural networks. arXiv preprint arXiv:1211.3711, 2012.  
Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In International Conference on Machine learning, 2006.  
Gael Guennebaud, Benoit Jacob, Philip Avery, Abraham Bachrach, Sebastien Barthelemy, et al. Eigen v3. http://eigen.tuxfamily.org, 2010.  
Takaaki Hori, Shinji Watanabe, Yu Zhang, and William Chan. Advances in joint CTC-attention based end-to-end speech recognition with a deep CNN encoder and RNN-LM. In Interspeech, 2017.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, 2015.  
Navdeep Jaitly, David Sussillo, Quoc V. Le, Oriol Vinyals, Ilya Sutskever, and Samy Bengio. A neural transducer. arXiv preprint arXiv:1511.04868, 2015.  
Walter G. Kelley and Allan C. Peterson. *Difference equations: an introduction with applications*. Academic Press, 2001.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Lingpeng Kong, Chris Dyer, and Noah A. Smith. Segmental recurrent neural networks. arXiv preprint arXiv:1511.06018, 2015.  
Richard E. Ladner and Michael J. Fischer. Parallel prefix computation. Journal of the ACM (JACM), 27(4):831-838, 1980.  
Dieterich Lawson, George Tucker, Chung-Cheng Chiu, Colin Raffel, Kevin Swersky, and Navdeep Jaitly. Learning hard alignments with variational inference. arXiv preprint arXiv:1705.05524, 2017.  
Yuping Luo, Chung-Cheng Chiu, Navdeep Jaitly, and Ilya Sutskever. Learning online alignments with continuous rewards policy gradient. arXiv preprint arXiv:1608.01281, 2016.  
Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. Effective approaches to attention-based neural machine translation. In Conference on Empirical Methods in Natural Language Processing, 2015.

Ramesh Nallapati, Bowen Zhou, Cicero Nogueira dos Santos, Caglar Gulgêhre, and Bing Xiang. Abstractive text summarization using sequence-to-sequence RNNs and beyond. In Conference on Computational Natural Language Learning, 2016.  
Douglas B. Paul and Janet M. Baker. The design for the Wall Street Journal-based CSR corpus. In Workshop on Speech and Natural Language, 1992.  
Rohit Prabhavalkar, Kanishka Rao, Tara Sainath, Bo Li, Leif Johnson, and Navdeep Jaitly. A comparison of sequence-to-sequence models for speech recognition. In Interspeech, 2017.  
Colin Raffel and Daniel P. W. Ellis. Feed-forward networks with attention can solve some long-term memory problems. arXiv preprint arXiv:1512.08756, 2015.  
Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, and Douglas Eck. Online and linear-time attention by enforcing monotonic alignments. In International Conference on Machine Learning, 2017.  
Abigail See, Peter J. Liu, and Christopher D. Manning. Get to the point: Summarization with pointer-generator networks. arXiv preprint arXiv:1704.04368, 2017.  
Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, 2014.  
Chong Wang, Dani Yogatama, Adam Coates, Tony Han, Awni Hannun, and Bo Xiao. Lookahead convolution layer for unidirectional recurrent neural networks. In Workshop Extended Abstracts of the 4th International Conference on Learning Representations, 2016.  
Yuxuan Wang, RJ Skerry-Ryan, Daisy Stanton, Yonghui Wu, Ron J. Weiss, Navdeep Jaitly, Zongheng Yang, Ying Xiao, Zhifeng Chen, Samy Bengio, Quoc Le, Yannis Agiomyrgiannakis, Rob Clark, and Rif A. Sauros. Tacotron: Towards end-to-end speech synthesis. arXiv preprint arXiv:1703.10135, 2017.  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In International Conference on Machine Learning, 2015.  
Lei Yu, Jan Buys, and Phil Blunsom. Online segment to segment neural transduction. In Conference on Empirical Methods in Natural Language Processing, 2016.  
Wojciech Zaremba and Ilya Sutskever. Reinforcement learning neural tuning machines. arXiv preprint arXiv:1505.00521, 2015.  
Yu Zhang, William Chan, and Navdeep Jaitly. Very deep convolutional networks for end-to-end speech recognition. arXiv preprint arXiv:1610.03022, 2016.
