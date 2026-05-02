# A COMPAR-AGGREGATE MODEL FOR MATCHING TEXT SEQUENCES

Shuohang Wang

School of Information Systems

Singapore Management University

shwang.2014@phdis.smu.edu.sg

Jing Jiang

School of Information Systems

Singapore Management University

jingjiang@smu.edu.sg

# ABSTRACT

Many NLP tasks including machine comprehension, answer selection and text entailment require the comparison between sequences. Matching the important units between sequences is a key to solve these problems. In this paper, we present a general "compare-aggregate" framework that performs word-level matching followed by aggregation using Convolutional Neural Networks. We particularly focus on the different comparison functions we can use to match two vectors. We use four different datasets to evaluate the model. We find that some simple comparison functions based on element-wise operations can work better than standard neural network and neural tensor network.

# 1 INTRODUCTION

Many natural language processing problems involve matching two or more sequences to make a decision. For example, in textual entailment, one needs to determine whether a hypothesis sentence can be inferred from a premise sentence (Bowman et al., 2015). In machine comprehension, given a passage, a question needs to be matched against it in order to find the correct answer (Richardson et al., 2013; Tapaswi et al., 2016). Table 1 gives two example sequence matching problems. In the first example, a passage, a question and four candidate answers are given. We can see that to get the correct answer, we need to match the question against the passage and identify the last sentence to be the answer-bearing sentence. In the second example, given a question and a set of candidate answers, we need to find the answer that best matches the question. Because of the fundamental importance of comparing two sequences of text to judge their semantic similarity or relatedness, sequence matching has been well studied in natural language processing.

With recent advances of neural network models in natural language processing, a standard practice for sequence modeling now is to encode a sequence of text as an embedding vector using models such as RNN and CNN. To match two sequences, a straightforward approach is to encode each sequence as a vector and then to combine the two vectors to make a decision (Bowman et al., 2015; Feng et al., 2015). However, it has been found that using a single vector to encode an entire sequence is not sufficient to capture all the important information from the sequence, and therefore advanced techniques such as attention mechanisms and memory networks have been applied to sequence matching problems (Hermann et al., 2015; Hill et al., 2016; Rocktäschel et al., 2015).

A common trait of a number of these recent studies on sequence matching problems is the use of a "compare-aggregate" framework (Wang & Jiang, 2016b; He & Lin, 2016; Parikh et al., 2016). In such a framework, comparison of two sequences is not done by comparing two vectors each representing an entire sequence. Instead, these models first compare vector representations of smaller units such as words from these sequences and then aggregate these comparison results to make the final decision. For example, the match-LSTM model proposed by Wang & Jiang (2016b) for textual entailment first compares each word in the hypothesis with an attention-weighted version of the premise. The comparison results are then aggregated through an LSTM. He & Lin (2016) proposed a pairwise word interaction model that first takes each pair of words from two sequences and applies a comparison unit on the two words. It then combines the results of these word interactions using a similarity focus layer followed by a multi-layer CNN. Parikh et al. (2016) proposed a decomposable attention model for textual entailment, in which words from each sequence are compared with an

Plot: ... Aragorn is crowned King of Gondor and taking Arwen as his queen before all present at his coronation bowing before Frodo and the other Hobbits. The Hobbits return to the Shire where Sam marries Rosie Cotton. ...

Qustion: Where does Sam marry Rosie?

Candidate answers: 0) Grey Havens. 1) Gondor. 2) The Shire. 3) Erebor. 4) Mordor.

Question: can i have auto insurance without a car

Ground-truth answer: yes, it be possible have auto insurance without own a vehicle. you will purchase what be call a name ...

Another candidate answer: insurance not be a tax or merely a legal obligation because auto insurance follow a car...

Table 1: The example on the left is a machine comprehension problem from MovieQA, where the correct answer here is The Shire. The example on the right is an answer selection problem from InsuranceQA.

attention-weighted version of the other sequence to produce a series of comparison vectors. The comparison vectors are then aggregated and fed into a feed forward network for final classification.

Although these studies have shown the effectiveness of such a "compare-aggregate" framework for sequence matching, there are at least two limitations with these previous studies: (1) Each of the models proposed in these studies is tested on one or two tasks only, but we hypothesize that this general framework is effective on many sequence matching problems. There has not been any study that empirically verifies this. (2) More importantly, these studies did not pay much attention to the comparison function that is used to compare two small textual units. Usually a standard feedforward network is used (Hu et al., 2014; Wang & Jiang, 2016b) to combine two vectors representing two units that need to be compared, e.g., two words. However, based on the nature of these sequence matching problems, we essentially need to measure how semantically similar the two sequences are. Presumably, this property of these sequence matching problems should guide us in choosing more appropriate comparison functions. Indeed He & Lin (2016) used cosine similarity, Euclidean distance and dot product to define the comparison function, which seem to be better justifiable. But they did not systematically evaluate these similarity or distance functions or compare them with a standard feedforward network.

In this paper, we argue that the general "compare-aggregate" framework is effective for a wide range of sequence matching problems. We present a model that follows this general framework and test it on four different datasets, namely, MovieQA, InsuranceQA, WikiQA and SNLI. The first three datasets are for Question Answering, but the setups of the tasks are quite different. The last dataset is for textual entailment. More importantly, we systematically present and test six different comparison functions. We find that overall a comparison function based on element-wise subtraction and multiplication works the best on the four datasets.

The contributions of this work are twofold: (1) Using four different datasets, we show that our model following the "compare-aggregate" framework is very effective when compared with the state-of-the-art performance on these datasets. (2) We conduct systematic evaluation of different comparison functions and show that a comparison function based on element-wise operations, which is not widely used for word-level matching, works the best across the different datasets. We believe that these findings will be useful for future research on sequence matching problems. We have also made our code available online.<sup>1</sup>

# 2 METHOD

In this section, we propose a general model following the "compare-aggregate" framework for matching two sequences. This general model can be applied to different tasks. We focus our discussion on six different comparison functions that can be plugged into this general "compare-aggregate" model. In particular, we hypothesize that two comparison functions based on element-wise operations, SUB and MULT, are good middle ground between highly flexible functions using standard neural network models and highly restrictive functions based on cosine similarity and/or Euclidean

![](images/86921a0b3f9c30473f58522dae062da357aed99271a114b53b615afac9dd035b.jpg)  
Figure 1: The left hand side is an overview of the model. The right hand side shows the details about the different comparison functions. The rectangles in dark represent parameters to be learned.  $\times$  represents matrix multiplication.

![](images/d1b7c5e4653e487d7c43cf3147e2528f26b1caaeeb8aaf0631306292b5132605.jpg)  
(1)  $NTN$

![](images/53b76199a77ffd50b96270eeef569bfb2080683bfffb498d1ffa03d7364d3dba.jpg)  
(2)  $NN$

![](images/053826f0561338cad5118fc277fec4ff48bc4c11607157c41518cc1702b99f8d.jpg)  
(3) Eucos

![](images/0d38954628d670b6fd4cea792584f61f281378019c8b10482311d9650d485d9d.jpg)  
(4) Sub

![](images/a8cb3a38658a6443445580699cf4081074eb80d61d5f216de8e9224bf83b1acb.jpg)  
(5) Mult

distance. As we will show in the experiment section, these comparison functions based on elementwise operations can indeed perform very well on a number of sequence matching problems.

# 2.1 PROBLEM DEFINITION AND MODEL OVERVIEW

The general setup of the sequence matching problem we consider is the following. We assume there are two sequences to be matched. We use two matrices  $\mathbf{Q} \in \mathbb{R}^{d \times Q}$  and  $\mathbf{A} \in \mathbb{R}^{d \times A}$  to represent the word embeddings of the two sequences, where  $Q$  and  $A$  are the lengths of the two sequences, respectively, and  $d$  is the dimensionality of the word embeddings. In other words, each column vector of  $\mathbf{Q}$  or  $\mathbf{A}$  is an embedding vector representing a single word. Given a pair of  $\mathbf{Q}$  and  $\mathbf{A}$ , the goal is to predict a label  $y$ . For example, in textual entailment,  $\mathbf{Q}$  may represent a premise and  $\mathbf{A}$  a hypothesis, and  $y$  indicates whether  $\mathbf{Q}$  entails  $\mathbf{A}$  or contradicts  $\mathbf{A}$ . In question answering,  $\mathbf{Q}$  may be a question and  $\mathbf{A}$  a candidate answer, and  $y$  indicates whether  $\mathbf{A}$  is the correct answer to  $\mathbf{Q}$ .

We treat the problem as a supervised learning task. We assume that a set of training examples in the form of  $(\mathbf{Q},\mathbf{A},y)$  is given and we aim to learn a model that maps any pair of  $(\mathbf{Q},\mathbf{A})$  to  $y$ .

An overview of our model is shown in Figure 1. The model can be divided into the following four layers:

1. Preprocessing: We use a preprocessing layer (not shown in the figure) to process  $\mathbf{Q}$  and  $\mathbf{A}$  to obtain two new matrices  $\overline{\mathbf{Q}}\in \mathbb{R}^{l\times Q}$  and  $\overline{\mathbf{A}}\in \mathbb{R}^{l\times A}$ . The purpose is to obtain a new embedding vector for each word in each sequence that captures some contextual information in addition to the word itself. For example,  $\overline{\mathbf{q}}_i\in \mathbb{R}^l$ , which is the  $i^{\mathrm{th}}$  column vector of  $\overline{\mathbf{Q}}$ , encodes the  $i^{\mathrm{th}}$  word in  $\mathbf{Q}$  together with its context in  $\mathbf{Q}$ .  
2. Attention: We apply a standard attention mechanism on  $\overline{\mathbf{Q}}$  and  $\overline{\mathbf{A}}$  to obtain attention weights over the column vectors in  $\overline{\mathbf{Q}}$  for each column vector in  $\overline{\mathbf{A}}$ . With these attention weights, for each column vector  $\overline{\mathbf{a}}_j$  in  $\overline{\mathbf{A}}$ , we obtain a corresponding vector  $\mathbf{h}_j$ , which is an attention-weighted sum of the column vectors of  $\overline{\mathbf{Q}}$ .  
3. Comparison: We use a comparison function  $f$  to combine each pair of  $\overline{\mathbf{a}}_j$  and  $\mathbf{h}_j$  into a vector  $\mathbf{t}_j$ .

4. Aggregation: We use a CNN layer to aggregate the sequence of vectors  $\mathbf{t}_j$  for the final classification.

Although this model follows more or less the same framework as the model proposed by Parikh et al. (2016), our work has some notable differences. First, we will pay much attention to the comparison function  $f$  and compare a number of options, including a some uncommon ones based on elementwise operations. Second, we apply our model to four different datasets representing four different tasks to evaluate its general effectiveness for sequence matching problems. There are also some other differences from the work by Parikh et al. (2016). For example, we use a CNN layer instead of summation and concatenation for aggregation. Our attention mechanism is one-directional instead of two-directional.

In the rest of this section we will present the model in detail. We will focus mostly on the comparison functions we consider.

# 2.2 PREPROCESSING AND ATTENTION

Our preprocessing layer uses a recurrent neural network to process the two sequences. We use a modified version of LSTM/GRU in which we keep only the input gates for remembering meaningful words:

$$
\overline {{\mathbf {Q}}} = \sigma (\mathbf {W} ^ {\mathrm {i}} \mathbf {Q} + \mathbf {b} ^ {\mathrm {i}} \otimes \mathbf {e} _ {Q}) \odot \tanh  (\mathbf {W} ^ {\mathrm {u}} \mathbf {Q} + \mathbf {b} ^ {\mathrm {u}} \otimes \mathbf {e} _ {Q}),
$$

$$
\overline {{\mathbf {A}}} = \sigma \left(\mathbf {W} ^ {\mathrm {i}} \mathbf {A} + \mathbf {b} ^ {\mathrm {i}} \otimes \mathbf {e} _ {A}\right) \odot \tanh  \left(\mathbf {W} ^ {\mathrm {u}} \mathbf {A} + \mathbf {b} ^ {\mathrm {u}} \otimes \mathbf {e} _ {A}\right), \tag {1}
$$

where  $\odot$  is element-wise multiplication, and  $\mathbf{W}^{\mathrm{i}}$ ,  $\mathbf{W}^{\mathrm{u}} \in \mathbb{R}^{l \times d}$  and  $\mathbf{b}^{\mathrm{i}}$ ,  $\mathbf{b}^{\mathrm{u}} \in \mathbb{R}^{l}$  are parameters to be learned. The outer product  $(\cdot \otimes \mathbf{e}_X)$  produces a matrix or row vector by repeating the vector or scalar on the left for  $X$  times.

The attention layer is built on top of the resulting  $\overline{\mathbf{Q}}$  and  $\overline{\mathbf{A}}$  as follows:

$$
\mathbf {G} = \operatorname {s o f t m a x} \left(\left(\mathbf {W} ^ {\mathrm {g}} \overline {{\mathbf {Q}}} + \mathbf {b} ^ {\mathrm {g}} \otimes \mathbf {e} _ {Q}\right) ^ {\mathrm {T}} \overline {{\mathbf {A}}}\right),
$$

$$
\mathbf {H} = \overline {{\mathbf {Q}}} \mathbf {G}, \tag {2}
$$

where  $\mathbf{W}^{\mathrm{g}}\in \mathbb{R}^{l\times l}$  and  $\mathbf{b}^{\mathrm{g}}\in \mathbb{R}^{l}$  are parameters to be learned,  $\mathbf{G}\in \mathbb{R}^{Q\times A}$  is the attention weight matrix, and  $\mathbf{H}\in \mathbb{R}^{l\times A}$  are the attention-weighted vectors. Specifically,  $\mathbf{h}_j$ , which is the  $j^{\mathrm{th}}$  column vector of  $\mathbf{H}$ , is a weighted sum of the column vectors of  $\overline{\mathbf{Q}}$  and represents the part of  $\mathbf{Q}$  that best matches the  $j^{\mathrm{th}}$  word in  $\mathbf{A}$ . Next we will combine  $\mathbf{h}_j$  and  $\overline{\mathbf{a}}_j$  using a comparison function.

# 2.3 COMPARISON

The goal of the comparison layer is to match each  $\overline{\mathbf{a}}_j$ , which represents the  $j^{\text{th}}$  word and its context in  $\mathbf{A}$ , with  $\mathbf{h}_j$ , which represents a weighted version of  $\mathbf{Q}$  that best matches  $\overline{\mathbf{a}}_j$ . Let  $f$  denote a comparison function that transforms  $\overline{\mathbf{a}}_j$  and  $\mathbf{h}_j$  into a vector  $\mathbf{t}_j$  to represent the comparison result.

A natural choice of  $f$  is a standard neural network layer that consists of a linear transformation followed by a non-linear activation function. For example, we can consider the following choice:

$$
\text {N E U R A L N E T (N N) :} \quad \mathbf {t} _ {j} = f (\overline {{\mathbf {a}}} _ {j}, \mathbf {h} _ {j}) = \operatorname {R e L U} \left(\mathbf {W} \left[ \begin{array}{l} \overline {{\mathbf {a}}} _ {j} \\ \mathbf {h} _ {j} \end{array} \right] + \mathbf {b}\right), \tag {3}
$$

where matrix  $\mathbf{W} \in \mathbb{R}^{l \times 2l}$  and vector  $\mathbf{b} \in \mathbb{R}^l$  are parameters to be learned.

Alternatively, another natural choice is a neural tensor network (Socher et al., 2013) as follows:

$$
\text {N E U R A L T E N S O R N E T (N T N)}: \quad \mathbf {t} _ {j} = f (\bar {\mathbf {a}} _ {j}, \mathbf {h} _ {j}) = \operatorname {R e L U} \left(\bar {\mathbf {a}} _ {j} ^ {\mathrm {T}} \mathbf {T} ^ {[ 1 \dots l ]} \mathbf {h} _ {j} + \mathbf {b}\right), \tag {4}
$$

where tensor  $\mathbf{T}^{[1\ldots l]}\in \mathbb{R}^{l\times l\times l}$  and vector  $\mathbf{b}\in \mathbb{R}^l$  are parameters to be learned.

However, we note that for many sequence matching problems, we intend to measure the semantic similarity or relatedness of the two sequences. So at the word level, we also intend to check how similar or related  $\overline{\mathbf{a}}_j$  is to  $\mathbf{h}_j$ . For this reason, a more natural choice used in some previous work is

<table><tr><td></td><td colspan="3">MovieQA</td><td colspan="3">InsuranceQA</td><td colspan="3">WikiQA</td><td colspan="3">SNLI</td></tr><tr><td></td><td>train</td><td>dev</td><td>test</td><td>train</td><td>dev</td><td>test</td><td>train</td><td>dev</td><td>test</td><td>train</td><td>dev</td><td>test</td></tr><tr><td>#Q</td><td>9848</td><td>1958</td><td>3138</td><td>13K</td><td>1K</td><td>1.8K*2</td><td>873</td><td>126</td><td>243</td><td>549K</td><td>9842</td><td>9824</td></tr><tr><td>#c</td><td>5</td><td>5</td><td>5</td><td>50</td><td>500</td><td>500</td><td>10</td><td>9</td><td>10</td><td>-</td><td>-</td><td>-</td></tr><tr><td>#w in P</td><td>873</td><td>866</td><td>914</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>#w in Q</td><td>10.6</td><td>10.6</td><td>10.8</td><td>7.2</td><td>7.2</td><td>7.2</td><td>6.5</td><td>6.5</td><td>6.4</td><td>14</td><td>15.2</td><td>15.2</td></tr><tr><td>#w in A</td><td>5.9</td><td>5.6</td><td>5.5</td><td>92.1</td><td>92.1</td><td>92.1</td><td>25.5</td><td>24.7</td><td>25.1</td><td>8.3</td><td>8.4</td><td>8.3</td></tr></table>

Table 2: The statistics of different data sets. Q:question/hypothesis, C:candidate answers for each question, A:answer/hypothesis, P:plot, w:word (average).

Euclidean distance or cosine similarity between  $\overline{\mathbf{a}}_j$  and  $\mathbf{h}_j$ . We therefore consider the following definition of  $f$ :

$$
\text {E U C L I D E A N} + \text {C O S I N E (E U C C O S)}: \quad \mathbf {t} _ {j} = f (\overline {{\mathbf {a}}} _ {j}, \mathbf {h} _ {j}) = \left[ \begin{array}{l} \| \overline {{\mathbf {a}}} _ {j} - \mathbf {h} _ {j} \| _ {2} \\ \cos (\overline {{\mathbf {a}}} _ {j}, \mathbf {h} _ {j}) \end{array} \right]. \tag {5}
$$

Note that with EUCOS, the resulting vector  $\mathbf{t}_j$  is only a 2-dimensional vector. Although EUCOS is a well-justified comparison function, we suspect that it may lose some useful information from the original vectors  $\overline{\mathbf{a}}_j$  and  $\mathbf{h}_j$ . On the other hand, NN and NTN are too general and thus do not capture the intuition that we care mostly about the similarity between  $\overline{\mathbf{a}}_j$  and  $\mathbf{h}_j$ .

To use something that is a good compromise between the two extreme cases, we consider the following two new comparison functions, which operate on the two vectors in an element-wise manner. These functions have been used previously by Tai et al. (2015).

$$
\text {S U B T R A C T I O N (S U B)}: \quad \mathbf {t} _ {j} = f (\overline {{\mathbf {a}}} _ {j}, \mathbf {h} _ {j}) = (\overline {{\mathbf {a}}} _ {j} - \mathbf {h} _ {j}) \odot (\overline {{\mathbf {a}}} _ {j} - \mathbf {h} _ {j}), \tag {6}
$$

$$
\text {M U L T I P L C A T I O N} (\text {M U L T}): \quad \mathbf {t} _ {j} = f (\overline {{\mathbf {a}}} _ {j}, \mathbf {h} _ {j}) = \overline {{\mathbf {a}}} _ {j} \odot \mathbf {h} _ {j}. \tag {7}
$$

Note that the operator  $\odot$  is element-wise multiplication. For both comparison functions, the resulting vector  $\mathbf{t}_j$  has the same dimensionality as  $\overline{\mathbf{a}}_j$  and  $\mathbf{h}_j$ .

We can see that SUB is closely related to Euclidean distance in that Euclidean distance is the sum of all the entries of the vector  $\mathbf{t}_j$  produced by SUB. But by not summing up these entries, SUB preserves some information about the different dimensions of the original two vectors. Similarly, MULT is closely related to cosine similarity but preserves some information about the original two vectors.

Finally, we consider combining SUB and MULT followed by an NN layer as follows:

$$
\mathrm {S u b M u l t} + \mathrm {N N}: \quad \mathbf {t} _ {j} = f (\overline {{\mathbf {a}}} _ {j}, \mathbf {h} _ {j}) = \operatorname {R e L U} (\mathbf {W} \left[ \begin{array}{c} \left(\overline {{\mathbf {a}}} _ {j} - \mathbf {h} _ {j}\right) \odot \left(\overline {{\mathbf {a}}} _ {j} - \mathbf {h} _ {j}\right) \\ \overline {{\mathbf {a}}} _ {j} \odot \mathbf {h} _ {j} \end{array} \right] + \mathbf {b}). \tag {8}
$$

In summary, we consider six different comparison functions: NN, NTN, EUCOS, SUB, MULT and SUBMULT+NN. Among these functions, the last three (SUB, MULT and SUBMULT+NN) have not been widely used in previous work for word-level matching.

# 2.4 AGGREGATION

After we apply the comparison function to each pair of  $\overline{\mathbf{a}}_j$  and  $\mathbf{h}_j$  to obtain a series of vectors  $\mathbf{t}_j$ , finally we aggregate these vectors using a one-layer CNN (Kim, 2014):

$$
\mathbf {r} = \operatorname {C N N} \left(\left[ \mathbf {t} _ {1}, \dots , \mathbf {t} _ {A} \right]\right). \tag {9}
$$

$\mathbf{r} \in \mathbb{R}^{nl}$  is then used for the final classification, where  $n$  is the number of windows in CNN.

# 3 EXPERIMENTS

In this section, we evaluate our model on four different datasets representing different tasks. The first three datasets are question answering tasks while the last one is on textual entailment. The statistics of the four datasets are shown in Table 2. We will fist introduce the task settings and the way we customize the "compare-aggregate" structure to each task. Then we will show the baselines for the different datasets. Finally, we discuss the experiment results shown in Table 3.

<table><tr><td rowspan="2">Models</td><td colspan="2">MovieQA</td><td colspan="3">InsuranceQA</td><td colspan="2">WikiQA</td><td colspan="2">SNLI</td></tr><tr><td>dev</td><td>test</td><td>dev</td><td>test1</td><td>test2</td><td>MAP</td><td>MRR</td><td>train</td><td>test</td></tr><tr><td>Cosine Word2Vec</td><td>46.4</td><td>45.63</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Cosine TFIDF</td><td>47.6</td><td>47.36</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SSCB TFIDF</td><td>48.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>IR model</td><td>-</td><td>-</td><td>52.7</td><td>55.1</td><td>50.8</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CNN with GESD</td><td>-</td><td>-</td><td>65.4</td><td>65.3</td><td>61.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Attentive LSTM</td><td>-</td><td>-</td><td>68.9</td><td>69.0</td><td>64.8</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>IARNN-Occam</td><td>-</td><td>-</td><td>69.1</td><td>68.9</td><td>65.1</td><td>0.7341</td><td>0.7418</td><td>-</td><td>-</td></tr><tr><td>IARNN-Gate</td><td>-</td><td>-</td><td>70.0</td><td>70.1</td><td>62.8</td><td>0.7258</td><td>0.7394</td><td>-</td><td>-</td></tr><tr><td>CNN-Cnt</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.6520</td><td>0.6652</td><td>-</td><td>-</td></tr><tr><td>ABCNN</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.6921</td><td>0.7108</td><td>-</td><td>-</td></tr><tr><td>CubeCNN</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.7090</td><td>0.7234</td><td>-</td><td>-</td></tr><tr><td>W-by-W Attention match-LSTM</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>85.3</td><td>83.5</td></tr><tr><td>LSTMN</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>92.0</td><td>86.1</td></tr><tr><td>Decomp Attention</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>88.5</td><td>86.3</td></tr><tr><td rowspan="2">EBIM+TreeLSTM</td><td rowspan="2">-</td><td rowspan="2">-</td><td rowspan="2">-</td><td rowspan="2">-</td><td rowspan="2">-</td><td rowspan="2">-</td><td rowspan="2">-</td><td>90.5</td><td>86.8</td></tr><tr><td>93.0</td><td>88.3</td></tr><tr><td>NN</td><td>31.6</td><td>-</td><td>76.8</td><td>74.9</td><td>72.4</td><td>0.7102</td><td>0.7224</td><td>89.3</td><td>86.3</td></tr><tr><td>NTN</td><td>31.6</td><td>-</td><td>75.6</td><td>75.0</td><td>72.5</td><td>0.7349</td><td>0.7456</td><td>91.6</td><td>86.3</td></tr><tr><td>EUCOS</td><td>71.9</td><td>-</td><td>70.6</td><td>70.2</td><td>67.9</td><td>0.6740</td><td>0.6882</td><td>87.1</td><td>84.0</td></tr><tr><td>SUB</td><td>64.9</td><td>-</td><td>70.0</td><td>71.3</td><td>68.2</td><td>0.7019</td><td>0.7151</td><td>89.8</td><td>86.8</td></tr><tr><td>MULT</td><td>66.4</td><td>-</td><td>76.0</td><td>75.2</td><td>73.4</td><td>0.7433</td><td>0.7545</td><td>89.7</td><td>85.8</td></tr><tr><td>SUBMULT+NN</td><td>72.1</td><td>72.9</td><td>77.0</td><td>75.6</td><td>72.3</td><td>0.7332</td><td>0.7477</td><td>89.4</td><td>86.8</td></tr></table>

Table 3: Experiment Results

# 3.1 TASK-SPECIFIC MODEL STRUCTURES

In all these tasks, we use matrix  $\mathbf{Q} \in \mathbb{R}^{d \times Q}$  to represent the question or premise and matrix  $\mathbf{A}_k \in \mathbb{R}^{d \times A_k}$  ( $k \in [1, K]$ ) to represent the  $k^{\text{th}}$  answer or the hypothesis. For the machine comprehension task MovieQA (Tapaswi et al., 2016), there is also a matrix  $\mathbf{P} \in \mathbb{R}^{d \times P}$  that represents the plot of a movie. Here  $Q$  is the length of the question or premise,  $A_k$  the length of the  $k^{\text{th}}$  answer, and  $P$  the length of the plot.

For the SNLI (Bowman et al., 2015) dataset, the task is text entailment, which identifies the relationship (entailment, contradiction or neutral) between a premise sentence and a hypothesis sentence. Here  $K = 1$ , and there are exactly two sequences to match. The actual model structure is what we have described before.

For the InsuranceQA (Feng et al., 2015) dataset, the task is an answer selection task which needs to select the correct answer for a question from a candidate pool. For the WikiQA (Yang et al., 2015) datasets, we need to rank the candidate answers according to a question. For both tasks, there are  $K$  candidate answers for each question. Let us use  $\mathbf{r}_k$  to represent the resulting vector produced by Eqn. 9 for the  $k^{\mathrm{th}}$  answer. In order to select one of the  $K$  answers, we first define  $\mathbf{R} = [\mathbf{r}_1,\mathbf{r}_2,\dots ,\mathbf{r}_K]$ . We then compute the probability of the  $k^{\mathrm{th}}$  answer to be the correct one as follows:

$$
p (k | \mathbf {R}) = \operatorname {s o f t m a x} \left(\mathbf {w} ^ {\mathrm {T}} \tanh  \left(\mathbf {W} ^ {\mathrm {s}} \mathbf {R} + \mathbf {b} ^ {\mathrm {s}} \otimes \mathbf {e} _ {K}\right) + b \otimes \mathbf {e} _ {K}\right), \tag {10}
$$

where  $\mathbf{W}^{\mathrm{s}}\in \mathbb{R}^{l\times nl}$ ,  $\mathbf{w}\in \mathbb{R}^l$ ,  $\mathbf{b}^{\mathrm{s}}\in \mathbb{R}^{l}$ ,  $b\in \mathbb{R}$  are parameters to be learned.

For the machine comprehension task MovieQA, each question is related to Plot Synopses written by fans after watching the movie and each question has five candidate answers. So for each candidate answer there are three sequences to be matched: the plot  $\mathbf{P}$ , the question  $\mathbf{Q}$  and the answer  $\mathbf{A}_k$ . For each  $k$ , we first match  $\mathbf{Q}$  and  $\mathbf{P}$  and refer to the matching result at position  $j$  as  $\mathbf{t}_{j}^{\mathrm{q}}$ , as generated by one of the comparison functions  $f$ . Similarly, we also match  $\mathbf{A}_k$  with  $\mathbf{P}$  and refer to the matching result at position  $j$  as  $\mathbf{t}_{k,j}^{\mathrm{a}}$ . We then define

$$
\mathbf {t} _ {k, j} \quad = \quad \left[ \begin{array}{c} \mathbf {t} _ {j} ^ {\mathbf {q}} \\ \mathbf {t} _ {k, j} ^ {\mathbf {a}} \end{array} \right],
$$

and

$$
\mathbf {r} _ {k} = \operatorname {C N N} \left(\left[ \mathbf {t} _ {k, 1}, \dots , \mathbf {t} _ {k, P} \right]\right).
$$

To select an answer from the  $K$  candidate answers, again we use Eqn. 10 to compute the probabilities.

# 3.2 BASELINES

Here, we will introduce the baselines for each dataset. We did not re-implement these models but simply took the reported performance for the purpose of comparison.

SNLI:  $\bullet$  W-by-W Attention: The model by Rocktäschel et al. (2015), who first introduced attention mechanism into text entailment.  $\bullet$  match-LSTM: The model by Wang & Jiang (2016b), which concatenates the matched words as the inputs of an LSTM.  $\bullet$  LSTMN: Long short-term memory-networks proposed by Cheng et al. (2016).  $\bullet$  Decomp Attention: Another "compare-aggregate" model proposed by Parikh et al. (2016).  $\bullet$  EBIM+TreeLSTM: The state-of-the-art model proposed by Chen et al. (2016) on the SNLI dataset.

InsuranceQA: IR model: This model by Bendersky et al. (2010) learns the concept information to help rank the candidates. CNN with GESD: This model by Feng et al. (2015) uses Euclidean distance and dot product between sequence representations built through convolutional neural networks to select the answer. Attentive LSTM: Tan et al. (2016) used soft-attention mechanism to select the most important information from the candidates according to the representation of the questions. IARNN-Occam: This model by Wang et al. (2016) adds regularization on the attention weights. IARNN-Gate: This model by Wang et al. (2016) uses the representation of the question to build the GRU gates for each candidate answer.

WikiQA: IARNN-Occam and IARNN-Gate as introduced before. CNN-Cnt: This model by Yang et al. (2015) combines sentence representations built by a convolutional neural network with logistic regression. ABCNN: This model is Attention-Based Convolutional Neural Network proposed by Yin et al. (2015). CubeCNN proposed by He & Lin (2016) builds a CNN on all pairs of word similarity.

MovieQA: All the baselines we consider come from Tapaswi et al. (2016)'s work:  $\bullet$  Cosine Word2Vec: A sliding window is used to select the answer according to the similarities computed through Word2Vec between the sentences in plot and the question/answer.  $\bullet$  Cosine TFIDF: This model is similar to the previous method but uses bag-of-word with tfidf scores to compute similarity.  $\bullet$  SSCB TFIDF: Instead of using the sliding window method, a convolutional neural network is built on the sentence level similarities.

# 3.3 ANALYSIS OF RESULTS

We use accuracy as the evaluation metric for the datasets MovieQA, InsuranceQA and SNLI, as there is only one correct answer or one label for each instance. For WikiQA, there may be multiple correct answers, so evaluation metrics we use are Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR).

We observe the following from the results. (1) Overall, we can find that our general "compare-aggregate" structure achieves the best performance on MovieQA, InsuranceQA, WikiQA datasets and very competitive performance on the SNLI dataset. Especially for the InsuranceQA dataset, with any comparison function we use, our model can outperform all the previous models. (2) The comparison method SUBMULT+NN is the best in general. (3) Some simple comparison functions can achieve better performance than the neural networks or neural tensor network comparison functions. For example, the simplest comparison function EUCOS achieves nearly the best performance in the MovieQA dataset, and the element-wise comparison functions, which do not need parameters can achieve the best performance on the WikiQA data set.

# 3.4 FURTHER ANALYSES

To further explain how our model works, we visualize the max values in each dimension of the convolutional layer. We use two examples shown in Table 1 from MovieQA and InsuranceQA data

![](images/3640d14a0df8d3457e6565193c90c474c7a808c24094553e777d705c1412a4b4.jpg)

![](images/872f05dd9e7ba7a981f369e3282f384ddd9397a6a496b8bd2fe98456d4daf901.jpg)  
Figure 2: An visualization of the largest value of each dimension in the convolutional layer of CNN. The top figure is an example from the data set MovieQA with CNN window size 5. The bottom figure is an example from the data set InsuranceQA with CNN window size 3.

sets respectively. In the top of Figure 2, we can see that the plot words that also appear in either the question or the answer will draw more attention by the CNN. We hypothesize that if the nearby words in the plot can match both the words in question and the words in one answer, then this answer is more likely to be the correct one. Similarly, the bottom one of Figure 2 also shows that the CNN will focus more on the matched word representations. If the words in one answer continuously match the words in the question, this answer is more likely to be the correct one.

# 4 RELATED WORK

We review related work in three types of general structures for matching sequences.

Siamense network: These kinds of models use the same structure, such as RNN or CNN, to build the representations for the sequences separately and then use them for classification. Then cosine similarity (Feng et al., 2015; Yang et al., 2015), element-wise operation (Tai et al., 2015; Mou et al., 2016) or neural network-based combination Bowman et al. (2015) are used for sequence matching.

Attentive network: Soft-attention mechanism (Bahdanau et al., 2014) has been widely used for sequence matching in machine comprehension (Hermann et al., 2015), text entailment (Roktäschel et al., 2015) and question answering (Tan et al., 2016). Instead of using the final state of RNN to represent a sequence, these studies use weighted sum of all the states for the sequence representation.

Compare-Aggregate network: This kind of framework is to perform the word level matching (Wang & Jiang, 2016a; Parikh et al., 2016; He & Lin, 2016; Trischler et al., 2016). Our work is under this framework. But our structure is different from previous models and our model can be applied on different tasks. Besides, we analyzed different word-level comparison functions separately.

# 5 CONCLUSIONS

In this paper, we systematically analyzed the effectiveness of a "compare-aggregate" model on four different datasets representing different tasks. Moreover, we compared and tested different kinds of word-level comparison functions and found that some element-wise comparison functions can outperform the others. According to our experiment results, many different tasks can share the same "compare-aggregate" structure. In the future work, we would like to test its effectiveness on multi-task learning.

# REFERENCES

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In Proceedings of the International Conference on Learning Representations, 2014.  
Michael Bendersky, Donald Metzler, and W Bruce Croft. Learning concept importance using a weighted dependence model. In Proceedings of the third ACM international conference on Web search and data mining, pp. 31-40. ACM, 2010.  
Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. A large annotated corpus for learning natural language inference. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2015.  
Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, and Hui Jiang. Enhancing and combining sequential and tree lstm for natural language inference. arXiv preprint arXiv:1609.06038, 2016.  
Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2016.  
Minwei Feng, Bing Xiang, Michael R Glass, Lidan Wang, and Bowen Zhou. Applying deep learning to answer selection: A study and an open task. In 2015 IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), pp. 813-820. IEEE, 2015.  
Hua He and Jimmy Lin. Pairwise word interaction modeling with deep neural networks for semantic similarity measurement. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 937-948, 2016.  
Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. In Proceedings of the Conference on Advances in Neural Information Processing Systems, pp. 1693-1701, 2015.  
Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston. The Goldilocks principle: Reading children's books with explicit memory representations. In Proceedings of the International Conference on Learning Representations, 2016.  
Baotian Hu, Zhengdong Lu, Hang Li, and Qingcai Chen. Convolutional neural network architectures for matching natural language sentences. In Advances in Neural Information Processing Systems, pp. 2042-2050, 2014.  
Yoon Kim. Convolutional neural networks for sentence classification. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2014.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the International Conference on Learning Representations, 2015.  
Lili Mou, Rui Men, Ge Li, Yan Xu, Lu Zhang, Rui Yan, and Zhi Jin. Natural language inference by tree-based convolution and heuristic matching. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2016.

Ankur P Parikh, Oscar Tackström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model for natural language inference. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2016.  
Jeffrey Pennington, Richard Socher, and Christopher D Manning. GloVe: Global vectors for word representation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2014.  
Matthew Richardson, Christopher JC Burges, and Erin Renshaw. MCTest: A challenge dataset for the open-domain machine comprehension of text. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2013.  
Tim Rocktäschel, Edward Grefenstette, Karl Moritz Hermann, Tomáš Kočisky, and Phil Blunsom. Reasoning about entailment with neural attention. In Proceedings of the International Conference on Learning Representations, 2015.  
Richard Socher, Alex Perelygin, Jean Y Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the conference on empirical methods in natural language processing, 2013.  
Kai Sheng Tai, Richard Socher, and Christopher D Manning. Improved semantic representations from tree-structured long short-term memory networks. In Proceedings of the Conference on Association for Computational Linguistics, 2015.  
Ming Tan, Cicero dos Santos, Bing Xiang, and Bowen Zhou. Improved representation learning for question answer matching. In Proceedings of the Conference on Association for Computational Linguistics, 2016.  
Makarand Tapaswi, Yukun Zhu, Rainer Stiefelhagen, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. MovieQA: Understanding stories in movies through question-answering. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, 2016.  
Adam Trischler, Zheng Ye, Xingdi Yuan, Jing He, Phillip Bachman, and Kaheer Suleman. A parallel-hierarchical model for machine comprehension on sparse data. In Proceedings of the Conference on Association for Computational Linguistics, 2016.  
Bingning Wang, Kang Liu, and Jun Zhao. Inner attention based recurrent neural networks for answer selection. In Proceedings of the Conference on Association for Computational Linguistics, 2016.  
Shuohang Wang and Jing Jiang. Machine comprehension using match-lstm and answer pointer. arXiv preprint arXiv:1608.07905, 2016a.  
Shuohang Wang and Jing Jiang. Learning natural language inference with LSTM. In Proceedings of the Conference on the North American Chapter of the Association for Computational Linguistics, 2016b.  
Yi Yang, Wen-tau Yih, and Christopher Meek. Wikiqa: A challenge dataset for open-domain question answering. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2015.  
Wenpeng Yin, Hinrich Schütze, Bing Xiang, and Bowen Zhou. Abcnn: Attention-based convolutional neural network for modeling sentence pairs. arXiv preprint arXiv:1512.05193, 2015.
