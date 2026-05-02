# EFFICIENT SUMMARIZATION WITH READ-AGAIN AND COPY MECHANISM

Wenyuan Zeng†, Wenjie Luo‡, Sanja Fidler†, Raquel Urtasun‡

†Tsinghua University, ‡University of Toronto

cengwy13@mails.tsinghua.edu.cn

{wenjie, fidler, urtasun}@cs.toronto.edu

# ABSTRACT

Encoder-decoder models have been widely used to solve sequence to sequence prediction tasks. However current approaches suffer from two shortcomings. First, the encoders compute a representation of each word taking into account only the history of the words it has read so far, yielding suboptimal representations. Second, current decoders utilize large vocabularies in order to minimize the problem of unknown words, resulting in slow decoding times. In this paper we address both shortcomings. Towards this goal, we first introduce a simple mechanism that first reads the input sequence before committing to a representation of each word. Furthermore, we propose a simple copy mechanism that is able to exploit very small vocabularies and handle out-of-vocabulary words. We demonstrate the effectiveness of our approach on the Gigaword dataset and DUC competition outperforming the state-of-the-art.

# 1 INTRODUCTION

Encoder-decoder models have been widely used in sequence to sequence tasks such as machine translation (Cho et al. (2014); Sutskever et al. (2014)). They consist of an encoder which represents the whole input sequence with a single feature vector. The decoder then takes this representation and generates the desired output sequence. The most successful models are LSTM and GRU as they are much easier to train than vanilla RNNs.

In this paper we are interested in summarization where the input sequence is a sentence/paragraph and the output is a summary of the text. Several encoding-decoding approaches have been proposed (Rush et al. (2015); Hu et al. (2015); Chopra et al. (2016)). Despite their success, it is commonly believed that the intermediate feature vectors are limited as they are created by only looking at previous words. This is particularly detrimental when dealing with large input sequences. Bi-directional RNNs (Schuster & Paliwal (1997); Bahdanau et al. (2014)) try to address this problem by computing two different representations resulting of reading the input sequence left-to-right and right-to-left. The final vectors are computed by concatenating the two representations. However, the word representations are computed with limited scope.

The decoder employed in all these methods outputs at each time step a distribution over a fixed vocabulary. In practice, this introduces problems with rare words (e.g., proper nouns) which are out of vocabulary. To alleviate this problem, one could potentially increase the size of the decoder vocabulary, but decoding becomes computationally much harder, as one has to compute the soft-max over all possible words. Gulcehre et al. (2016), Nallapati et al. (2016) and Gu et al. (2016) proposed to use a copy mechanism that dynamically copy the words from the input sequence while decoding. However, they lack the ability to extract proper embeddings of out-of-vocabulary words from the input context. Bahdanau et al. (2014) proposed to use an attention mechanism to emphasize specific parts of the input sentence when generating each word. However the encoder problem still remains in this approach.

In this work, we propose two simple mechanisms to deal with both encoder and decoder problems. We borrowed intuition from human readers which read the text multiple times before generating summaries. We thus propose a 'Read-Again' model that first reads the input sequence before committing to a representation of each word. The first read representation then biases the second read

representation and thus allows the intermediate hidden vectors to capture the meaning appropriate for the input text. We show that this idea can be applied to both LSTM and GRU models. Our second contribution is a copy mechanism which allows us to use much smaller decoder vocabulary sizes resulting in much faster decoding. Our copy mechanism also allows us to construct a better representation of out-of-vocabulary words. We demonstrate the effectiveness of our approach in the challenging Gigaword dataset and DUC competition showing state-of-the-art performance.

# 2 RELATED WORK

# 2.1 SUMMARIZATION

In the past few years, there has been a lot of work on extractive summarization, where a summary is created by composing words or sentences from the source text. Notable examples are Neto et al. (2002), Erkan & Radev (2004), Wong et al. (2008), Filippova & Altun (2013) and Colmenares et al. (2015). As a consequence of their extractive nature the summary is restricted to words (sentences) in the source text.

Abstractive summarization, on the contrary, aims at generating consistent summaries based on understanding the input text. Although there has been much less work on abstractive methods, they can in principle produce much richer summaries. Abstractive summarization is standardized by the DUC2003 and DUC2004 competitions (Over et al. (2007)). Some of the prominent approaches on this task include Banko et al. (2000), Zajic et al. (2004), Cohn & Lapata (2008) and Woodsend et al. (2010). Among them, the TOPIARY system (Zajic et al. (2004)) performs the best in the competitions amongst non neural net based methods.

Very recently, the success of deep neural networks in many natural language processing tasks (Collobert et al. (2011)) has inspired new work in abstractive summarization. Rush et al. (2015) propose a neural attention model with a convolutional encoder to solve this task. Hu et al. (2015) build a large dataset for Chinese text summarization and propose to feed all hidden states from the encoder into the decoder. More recently, Chopra et al. (2016) extended Rush et al. (2015)'s work with an RNN decoder, and Nallapati et al. (2016) proposed an RNN encoder-decoder architecture for summarization. Both techniques are currently the state-of-the-art on the DUC competition. However, the encoders exploited in these methods lack the ability to encode each word condition on the whole text, as an RNN encodes a word into a hidden vector by taking into account only the words up to that time step.

In contrast, in this work we propose a 'Read-Again' encoder-decoder architecture, which enables the encoder to understand each input word after reading the whole sentence. Our encoder first reads the text, and the results from the first read help represent the text in the second pass over the source text. Our second contribution is a simple copy mechanism that allows us to significantly reduce the decoder vocabulary size resulting in much faster inference times. Furthermore our copy mechanism allows us to handle out-of-vocabulary words in a principled manner. Finally our experiments show state-of-the-art performance on the DUC competition.

# 2.2 NEURAL MACHINE TRANSLATION

Our work is also closely related to recent work on neural machine translation, where neural encoder-decoder models have shown promising results (Kalchbrenner & Blunsom (2013); Cho et al. (2014); Sutskever et al. (2014)). Bahdanau et al. (2014) further developed an attention mechanism in the decoder in order to pay attention to a specific part of the input at every generating time-step. Our approach also exploits an attention mechanism during decoding.

# 2.3 OUT-OF-VOCABULARY AND COPY MECHANISM

Dealing with Out-Of-Vocabulary words (OOVs) is an important issue in sequence to sequence approaches as we cannot enumerate all possible words and learn their embeddings since they might not be part of our training set. Luong et al. (2014) address this issue by annotating words on the source, and aligning OOVs in the target with those source words. Recently, Vinyls et al. (2015) propose Pointer Networks, which calculate a probability distribution over the input sequence instead

![](images/967f9a1b676466739b68c0eca96e163734b6cbbb54152b29a9847a235c0ea739.jpg)  
Figure 1: Read-Again Model

![](images/99f425f10fbdae7164c4f3cb780c14e5e4accfd71500544076e3f7c0b54a9cd9.jpg)

of predicting a token from a pre-defined dictionary. Cheng & Lapata (2016) develop a neural-based extractive summarization model, which predicts the targets from the input sequences. Gulcehre et al. (2016); Nallapati et al. (2016) add a hard gate to allow the model to decide wether to generate a target word from the fixed-size dictionary or from the input sequence. Gu et al. (2016) use a softmax operation instead of the hard gating. This softmax pointer mechanism is similar to our decoder. However, our decoder can also extract different OOVs' embedding from the input text instead of using a single  $<\mathrm{UNK}>$  embedding to represent all OOVs. This further enhances the model's ability to handle OOVs.

# 3 THE READ AGAIN MODEL

Text summarization can be formulated as a sequence to sequence prediction task, where the input is a longer text and the output is a summary of that text. In this paper we develop an encoder-decoder approach to summarization. The encoder is used to represent the input text with a set of continuous vectors, and the decoder is used to generate a summary word by word.

In the following, we first introduce our 'Read-Again' model for encoding sentences. The idea behind our approach is very intuitive and is inspired by how humans do this task. When we create summaries, we first read the text and then we do a second read where we pay special attention to the words that are relevant to generate the summary. Our 'Read-Again' model implements this idea by reading the input text twice and using the information acquired from the first read to bias the second read. This idea can be seamlessly plugged into LSTM and GRU models. Our second contribution is a copy mechanism used in the decoder. It allows us to reduce the decoder vocabulary size dramatically and can be used to extract a better embedding for OOVs. Fig. 1(a) gives an overview of our model.

# 3.1 ENCODER

We first review the typical encoder used in machine translation (e.g., Sutskever et al. (2014); Bahdanau et al. (2014)). Let  $x = \{x_{1},x_{2},\dots ,x_{n}\}$  be the input sequence of words. An encoder sequentially reads each word and creates the hidden representation  $h_i$  by exploiting a recurrent neural network (RNN)

$$
h _ {i} = \operatorname {R N N} \left(\mathbf {x} _ {\mathrm {i}}, h _ {i - 1}\right), \tag {1}
$$

where  $\mathbf{x_i}$  is the word embedding of  $x_{i}$ . The hidden vectors  $h = \{h_1,h_2,\dots ,h_n\}$  are then treated as the feature representations for the whole input sentence and can be used by another RNN to decode and generate a target sentence. Although RNNs have been shown to be useful in modeling sequences, one of the major drawback is that  $h_i$  depends only on past information i.e.,  $\{x_{1},\dots ,x_{i}\}$ . However, it is hard (even for humans) to have a proper representation of a word without reading the whole input sentence.

Following this intuition, we propose our 'Read-Again' model where the encoder reads the input sentence twice. In particular, the first read is used to bias the second more attentive read. We apply this idea to two popular RNN architectures, i.e. GRU and LSTM, resulting in better encodings of the

![](images/023e9fd23771944621aaf59b3cecdfc2a13d184148429cf5d6e8843441b08bec.jpg)  
(a) GRU Read-Again Encoder

![](images/09350be92839a1653958be9a16b878beb3c07ba7dd0cc771ed3d832e7db41aec.jpg)  
(b) LSTM Read-Again Encoder  
Figure 2: Read-Again Model

input text. Note that although other alternatives, such as bidirectional RNN exist, the hidden states from the forward RNN lack direct interactions with the backward RNN, and thus forward/backward hidden states still cannot utilize the whole sequence. Besides, although we only use our model in a uni-directional manner, it can also be easily adapted to the bidirectional case. We now describe the two variants of our model.

# 3.1.1 GRUREAD-AGAIN

We read the input sentence  $\{x_{1}, x_{2}, \dots, x_{n}\}$  for the first-time using a standard GRU

$$
h _ {i} ^ {1} = \operatorname {G R U} ^ {1} \left(\mathbf {x} _ {\mathrm {i}}, h _ {i - 1} ^ {1}\right), \tag {2}
$$

where the function  $GRU^1$  is defined as,

$$
z _ {i} = \sigma \left(W _ {z} \left[ \mathbf {x} _ {\mathbf {i}}, h _ {i - 1} ^ {1} \right]\right) \tag {3}
$$

$$
r _ {i} = \sigma \left(W _ {r} \left[ \mathbf {x} _ {\mathbf {i}}, h _ {i - 1} ^ {1} \right]\right)
$$

$$
\widetilde {h} _ {i} ^ {1} = \tanh  \left(W _ {h} \left[ \mathbf {x} _ {\mathbf {i}}, r _ {i} \odot h _ {i - 1} ^ {1} \right]\right)
$$

$$
h _ {i} ^ {1} = \left(1 - z _ {i}\right) \odot h _ {i - 1} ^ {1} + z _ {i} \odot \widetilde {h} _ {i} ^ {1}
$$

It consists of two gatings  $z_{i}, r_{i}$ , controlling whether the current hidden state  $h_{i}^{1}$  should be directly copied from  $h_{i-1}^{1}$  or should pass through a more complex path  $\widetilde{h}_{i}^{1}$ .

Given the sentence feature vector  $h_n^1$ , we then compute an importance weight vector  $\alpha_i$  of each word for the second reading. We put the importance weight  $\alpha_i$  on the skip-connections as shown in Fig. 2(a) to bias the two information flows: If the current word  $x_i$  has a very small weight  $\alpha_i$ , then the second read hidden state  $h_i^2$  will mostly take the information directly from the previous state  $h_{i - 1}^2$ , ignoring the influence of the current word. If  $\alpha_i$  is close to 1 then it will be similar to a standard GRU, which is only influenced from the current word. Thus the second reading has the following update rule

$$
h _ {i} ^ {2} = \left(1 - \alpha_ {i}\right) \odot h _ {i - 1} ^ {2} + \alpha_ {i} \odot \mathrm {G R U} ^ {2} \left(\mathbf {x} _ {\mathbf {i}}, h _ {i - 1} ^ {2}\right), \tag {4}
$$

where  $\odot$  means element-wise product. We compute the importance weights by attending  $h_i^1$  with  $h_n^1$  as follows

$$
\alpha_ {i} = \tanh  \left(W _ {e} h _ {i} ^ {1} + U _ {e} h _ {n} ^ {1} + V _ {e} \mathbf {x} _ {\mathbf {i}}\right), \tag {5}
$$

where  $W_{e}$ ,  $U_{e}$ ,  $V_{e}$  are learnable parameters. Note that  $\alpha_{i}$  is a vector representing the importance of each dimension in the word embedding. Empirically, we find that using a vector is better than a single value. We hypothesize that this is because different dimensions represent different semantic meanings, and a single value lacks the ability to model the variances among these dimensions.

Combining this with the standard GRU update rule

$$
\mathrm {G R U} ^ {2} \left(\mathbf {x _ {i}}, h _ {i - 1} ^ {2}\right) = \left(1 - z _ {i}\right) \odot h _ {i - 1} ^ {2} + z _ {i} \odot \widetilde {h} _ {i} ^ {2},
$$

![](images/f7fcc9b69c983c2c83a9080c6df1a0e31c89bb118fe72deebcb581cc596cc43a.jpg)  
Figure 3: Hierarchical Read-Again

we can simplify the updating rule Eq. (4) to get

$$
h _ {i} ^ {2} = \left(1 - \alpha_ {i} \odot z _ {i}\right) \odot h _ {i - 1} ^ {2} + \left(\alpha_ {i} \odot z _ {i}\right) \odot \widetilde {h} _ {i} ^ {2} \tag {6}
$$

This equations show that our 'read-again' model on GRU is equivalent to replace the GRU cell with a more general gating mechanism that also depends on the feature representation of the whole sentence computed from the first reading pass. We argue that adding this global information could help direct the information flow for the forward pass resulting in a better encoder.

# 3.1.2 LSTM READ-AGAIN

We now apply the 'Read-Again' idea to the LSTM architecture as shown in Fig. 2(b). Our first reading is performed by an  $LSTM^{1}$  defined as

$$
f _ {i} = \sigma \left(W _ {f} \left[ \mathbf {x} _ {\mathbf {i}}, h _ {i - 1} \right]\right) \tag {7}
$$

$$
i _ {i} = \sigma \left(W _ {i} \left[ \mathbf {x} _ {\mathbf {i}}, h _ {i - 1} \right]\right)
$$

$$
o _ {i} = \sigma \left(W _ {o} [ \mathbf {x} _ {\mathrm {i}}, h _ {i - 1} ]\right)
$$

$$
\widetilde {C _ {i}} = \tanh  \left(W _ {C} \left[ \mathbf {x} _ {\mathbf {i}}, h _ {i - 1} \right]\right)
$$

$$
C _ {i} = f _ {t} \odot C _ {i - 1} + i _ {i} \odot \widetilde {C _ {i}}
$$

$$
h _ {i} = o _ {i} \odot \tanh  (C _ {i})
$$

Different from the GRU architecture, LSTM calculates the hidden state by applying a non-linear activation function to the cell state  $C_i$ , instead of a linear combination of two paths used in the GRU. Thus for our second read, instead of using skip-connections, we make the gating functions explicitly depend on the whole sentence vector computed from the first reading pass. We argue that this helps the encoding of the second reading  $LSTM^2$ , as all gating and updating increments are also conditioned on the whole sequence feature vector  $(h_i^1, h_n^1)$ . Thus

$$
h _ {i} ^ {2} = \operatorname {L S T M} ^ {2} \left(\left[ \mathbf {x} _ {\mathbf {i}}, h _ {i} ^ {1}, h _ {n} ^ {1} \right], h _ {i - 1} ^ {2}\right), \tag {8}
$$

# 3.1.3 READING MULTIPLE SENTENCES

In this section we extend our 'Read-Again' model to the case where the input sequence has more than one sentence. Towards this goal, we propose to use a hierarchical representation, where each sentence has its own feature vector from the first reading pass. We then combine them into a single vector to bias the second reading pass. We illustrate this in the context of two input sentences, but it is easy to generalize to more sentences. Let  $\{x_{1}, x_{2}, \dots, x_{n}\}$  and  $\{x_{1}', \dots, x_{m}'\}$  be the two input sentences. The first RNN reads these two sentences independently to get two sentence feature vectors  $h_{n}^{1}$  and  $h_{m}^{1}$  respectively.

Here we investigate two different ways to handle multiple sentences. Our first option is to simply concatenate the two feature vectors to bias our second reading pass:

$$
h _ {i} ^ {2} = \operatorname {R N N} ^ {2} \left(\left[ \mathbf {x} _ {\mathrm {i}}, h _ {i} ^ {1}, h _ {n} ^ {1}, h _ {m} ^ {\prime 1} \right], h _ {i - 1} ^ {2}\right) \tag {9}
$$

$$
h _ {i} ^ {\prime 2} = \mathrm {R N N} ^ {2} ([ \mathbf {x _ {i} ^ {\prime}}, h _ {i} ^ {\prime 1}, h _ {n} ^ {1}, h _ {m} ^ {\prime 1} ], h _ {i - 1} ^ {\prime 2})
$$

where  $h_0^2$  and  $h_0'^2$  are initial zero vectors. Feeding  $h_n^1$ ,  $h_m'^1$  into the second RNN provides more global information explicitly and helps acquire long term dependencies.

The second option we explored is shown in Fig. 3.1.2. In particular, we use a non-linear transformation to get a single feature vector  $h_{global}$  from both sentence feature vectors:

$$
h _ {g l o b a l} = \tanh  \left(W _ {r} \cdot h _ {n} ^ {1} + U _ {r} \cdot h _ {m} ^ {\prime 1} + v _ {r}\right) \tag {10}
$$

The second reading pass is then

$$
\widetilde {h _ {i} ^ {2}} = \operatorname {R N N} ^ {2} \left(\left[ \mathbf {x} _ {\mathbf {i}}, h _ {i} ^ {1}, h _ {n} ^ {1}, h _ {\text {g l o b a l}} \right], h _ {i - 1} ^ {2}\right) \tag {11}
$$

$$
\widetilde {h _ {i} ^ {\prime 2}} = \mathrm {R N N} ^ {2} ([ \mathbf {x _ {i} ^ {\prime}}, h _ {i} ^ {\prime 1}, h _ {m} ^ {\prime 1}, h _ {g l o b a l} ], h _ {i - 1} ^ {\prime 2})
$$

Note that this is more easily scalable to more sentences. In our experiments both approaches perform similarly.

# 3.2 DECODER WITH COPY MECHANISM

In this paper we argue that only a small number of common words are needed for generating a summary in addition to the words that are present in the source text. We can consider this as a hybrid approach which combines extractive and abstractive summarization. This has two benefits: first it allows us to use a very small vocabulary size, speeding up inference. Furthermore, we can create summaries which contain OOVs if they are present in the source text.

Our decoder reads the vector representations of the input text using an attention mechanism, and generates the target summary word by word. We use an LSTM as our decoder, with a fixed-size vocabulary dictionary  $Y$  and learnable word embeddings  $\mathbf{Y} \in \mathbf{R}^{|Y| \times dim}$ . At time-step  $t$  the LSTM generates a summary word  $y_{t}$  by first computing the current hidden state  $s_{t}$  from the previous hidden state  $s_{t-1}$ , previous summary word  $y_{t-1}$  and current context vector  $c_{t}$

$$
s _ {t} = L S T M \left(\left[ \mathbf {y} _ {\mathbf {t} - \mathbf {1}}, c _ {t} \right], s _ {t - 1}\right), \tag {12}
$$

where the context vector  $c_{t}$  is computed with an attention mechanism on the encoder hidden states:

$$
c _ {t} = \sum_ {i = 1} ^ {n} \beta_ {i t} h _ {i} ^ {2}. \tag {13}
$$

The attention score  $\beta_{it}$  at time-step  $t$  on the  $i$ -th word is computed via a soft-max over  $o_{it}$ , where

$$
o _ {i t} = \operatorname {a t t} \left(s _ {t - 1}, h _ {i} ^ {2}\right) = v _ {a} ^ {T} \tanh  \left(W _ {a} s _ {t - 1} + U _ {a} h _ {i} ^ {2}\right), \tag {14}
$$

with  $v_{a}, W_{a}, U_{a}$  learnable parameters.

A typical way to treat OOVs is to encode them with a single shared embedding. However, different OOVs can have very different meanings, and thus using a single embedding for all OOVs will confuse the model. This is particularly detrimental when using small vocabulary sizes. Here we address this issue by deriving the representations of OOVs from their corresponding context in the input text. Towards this goal, we change the update rule of  $\mathbf{y}_{\mathbf{t} - 1}$ . In particular, if  $y_{t - 1}$  belongs to a word that is in our decoder vocabulary we take its representation from the word embedding, otherwise if it appears in the input sentence as  $x_{i}$  we use

$$
\mathbf {y} _ {\mathbf {t} - \mathbf {1}} = \mathbf {p} _ {\mathbf {i}} = \tanh  \left(W _ {c} h _ {i} ^ {2} + b _ {c}\right) \tag {15}
$$

where  $W_{c}$  and  $b_{c}$  are learnable parameters. Since  $h_{i}^{2}$  encodes useful context information of the source word  $x_{i}, p_{i}$  can be interpreted as the semantics of this word extracted from the input sentence. Furthermore, if  $y_{t-1}$  does not appear in the input text, nor in  $Y$ , then we represent  $\mathbf{y}_{\mathbf{t}-1}$  using the <UNK> embedding.

Given the current decoder's hidden state  $s_t$ , we can generate the target summary word  $y_t$ . As shown in Fig. 1(b), at each time step during decoding, the decoder outputs a distribution over generating words from  $Y$ , as well as over copying a specific word  $x_i$  from the source sentence.

<table><tr><td>#Input</td><td>Model</td><td>Size</td><td>Rouge-1</td><td>Rouge-2</td><td>Rouge-L</td></tr><tr><td rowspan="9">1 sent</td><td>ABS (baseline)</td><td>69K</td><td>24.12</td><td>10.24</td><td>22.61</td></tr><tr><td>GRU (baseline)</td><td>69K</td><td>26.79</td><td>12.03</td><td>25.14</td></tr><tr><td>Ours-GRU</td><td>69K</td><td>27.26</td><td>12.28</td><td>25.48</td></tr><tr><td>Ours-LSTM</td><td>69K</td><td>27.82</td><td>12.74</td><td>26.01</td></tr><tr><td>GRU (baseline)</td><td>15K</td><td>24.67</td><td>11.30</td><td>23.28</td></tr><tr><td>Ours-GRU</td><td>15K</td><td>25.04</td><td>11.40</td><td>23.47</td></tr><tr><td>Ours-LSTM</td><td>15K</td><td>25.30</td><td>11.76</td><td>23.71</td></tr><tr><td>Ours-GRU (C)</td><td>15K</td><td>27.41</td><td>12.58</td><td>25.74</td></tr><tr><td>Ours-LSTM (C)</td><td>15K</td><td>27.37</td><td>12.64</td><td>25.69</td></tr><tr><td rowspan="2">2 sent</td><td>Ours-Opt-1 (C)</td><td>15K</td><td>27.95</td><td>12.65</td><td>26.10</td></tr><tr><td>Ours-Opt-2 (C)</td><td>15K</td><td>27.96</td><td>12.65</td><td>26.18</td></tr></table>

Table 1: Different Read-Again Model. Ours denotes Read-Again models. C denotes copy mechanism. Ours-Opt-1 and Ours-Opt-2 are the models described in section 3.1.3. Size denotes the size of decoder vocabulary in a model.

# 3.3 LEARNING

We jointly learn our encoder and decoder by maximizing the likelihood of decoding the correct word at each time step. We refer the reader to the experimental evaluation for more details.

# 4 EXPERIMENTAL EVALALUATION

In this section, we show results of abstractive summarization on Gigaword (Graff & Cieri (2003); Naples et al. (2012)) and DUC2004 (Over et al. (2007)) datasets. Our model can learn a meaningful re-reading weight distribution for each word in the input text, putting more emphasis on important verb and nous, while ignoring common words such as prepositions. As for the decoder, we demonstrate that our copy mechanism can successfully reduce the typical vocabulary size by a factor 5 while achieving much better performance than the state-of-the-art, and by a factor of 30 while maintaining the same level of performance. In addition, we provide an analysis and examples of which words are copied during decoding.

Dataset and Evaluation Metric: We use the Gigaword corpus to train and evaluate our models. Gigaword is a news corpus where the title is employed as a proxy for the summary of the article. We follow the same pre-processing steps of Rush et al. (2015), which include filtering, PTB tokenization, lower-casing, replacing digit characters with #, replacing low-frequency words with UNK and extracting the first sentence in each article. This results in a training set of 3.8M articles, a validation set and a test set each containing 400K articles. The average sentence length is 31.3 words for the source, and 8.3 words for the summaries. Following the standard protocol we evaluate ROUGE score on 2000 random samples from the test set. As for evaluation metric, we use full-length F1 score on Rouge-1, Rouge-2 and Rouge-L, following Chopra et al. (2016) and Nallapati et al. (2016), since these metrics are less bias to the outputs' length than full-length recall scores.

**Implementation Details:** We implement our model in Tensorflow and conduct all experiments on a NVIDIA Titan X GPU. Our models converged after 2-3 days of training, depending on model size. Our RNN cells in all models have 1 layer, 512-dimensional hidden states, and 512-dimensional word embeddings. We use dropout rate of 0.2 in all activation layers. All parameters, except the biases are initialized uniformly with a range of  $\sqrt{3 / d}$ , where  $d$  is the dimension of the hidden state (Sussillo & Abbott (2014)). The biases are initialized to 0.1. We use plain SGD to train the model with gradient clipped at 10. We start with an initial learning rate of 2, and halve it every epoch after first 5 epochs. Our max epoch for training is 10. We use a mini-batch size of 64, which is shuffled during training.

# 4.1 QUANTITATIVE EVALUATION

Results on Gigaword: We compare the performances of different architectures and report ROUGE scores in Tab. 1. Our baselines include the ABS model of Rush et al. (2015) with its proposed

<table><tr><td>Models</td><td>Size</td><td>Rouge-1</td><td>Rouge-2</td><td>Rouge-L</td></tr><tr><td>ZOPIARY (Zajic et al. (2004))</td><td>-</td><td>25.12</td><td>6.46</td><td>20.12</td></tr><tr><td>ABS (Rush et al. (2015))</td><td>69K</td><td>26.55</td><td>7.06</td><td>23.49</td></tr><tr><td>ABS+ (Rush et al. (2015))</td><td>69K</td><td>28.18</td><td>8.49</td><td>23.81</td></tr><tr><td>RAS-LSTM (Chopra et al. (2016))</td><td>69K</td><td>27.41</td><td>7.69</td><td>23.06</td></tr><tr><td>RAS-Elman (Chopra et al. (2016))</td><td>69K</td><td>28.97</td><td>8.26</td><td>24.06</td></tr><tr><td>big-words-lvt2k-1sent (Nallapati et al. (2016))</td><td>69K</td><td>28.35</td><td>9.46</td><td>24.59</td></tr><tr><td>big-words-lvt5k-1sent (Nallapati et al. (2016))</td><td>200K</td><td>28.61</td><td>9.42</td><td>25.24</td></tr><tr><td>Ours-GRU (C)</td><td>15K</td><td>29.08</td><td>9.20</td><td>25.25</td></tr><tr><td>Ours-LSTM (C)</td><td>15K</td><td>29.89</td><td>9.37</td><td>25.93</td></tr><tr><td>Ours-Opt-2 (C)</td><td>15K</td><td>29.74</td><td>9.44</td><td>25.94</td></tr></table>

Table 2: Rouge-N limited-length recall on DUC2004. Size denotes the size of decoder vocabulary in a model.

vocabulary size as well as an attention encoder-decoder model with uni-directional GRU encoder. We allow the decoder to generate variable length summaries. As shown in Tab. 1 our Read-Again models outperform the baselines on all ROUGE scores, when using both 15K and 69K sized vocabularies. We also observe that adding the copy mechanism further helps to improve performance: Even though the decoder vocabulary size of our approach with copy (15K) is much smaller than ABS (69K) and GRU (69K), it achieves a higher ROUGE score. Besides, our Multiple-Sentences model achieves the best performance.

Evaluation on DUC2004: DUC 2004 (Over et al. (2007)) is a commonly used benchmark on summarization task consisting of 500 news articles. Each article is paired with 4 different human-generated reference summaries, capped at 75 characters. This dataset is evaluation-only. Similar to Rush et al. (2015), we train our neural model on the Gigaword training set, and show the models' performances on DUC2004. Following the convention, we also use ROUGE limited-length recall as our evaluation metric, and set the capping length to 75 characters. We generate summaries with 15 words using beam-size of 10. As shown in Table 2, our method outperforms all previous methods on Rouge-1 and Rouge-L, and is comparable on Rouge-2. Furthermore, our model only uses 15k decoder vocabulary, while previous methods use 69k or 200k.

Importance Weight Visualization: As we described in the section before,  $\alpha_{i}$  is a high-dimension vector representing the importance of each word  $x_{i}$ . While the importance of a word is different over each dimension, by averaging we can still look at general trends of which word is more relevant. Fig. 4 depicts sample sentences with the importance weight  $\alpha_{i}$  over input words. Words such as the,  $a$ , 's, have small  $\alpha_{i}$ , while words such as aeronautics, resettled, impediments, which carry more information have higher values. This shows that our read-again technique indeed extracts useful information from the first reading to help bias the second reading results.

![](images/503d8b1b46f72977e9359ad5d10301db5d16fa35716f1e42893fe1cba0e86b21.jpg)  
Figure 4: Weight Visualization. Black indicates high weight

# 4.2 DECODER VOCABULARY SIZE

Table 3 shows the effect on our model of decreasing the decoder vocabulary size. We can see that when using the copy mechanism, we are able to reduce the decoder vocabulary size from 69K to 2K, with only 2-3 points drop on ROUGE score. This contrasts the models that do not use the copy mechanism. This is possibly due to two reasons. First, when faced with OOVs during decoding

<table><tr><td></td><td colspan="2">Rouge-1</td><td colspan="2">Rouge-2</td><td colspan="2">Rouge-L</td></tr><tr><td>Size</td><td>Ours-LSTM</td><td>Ours-LSTM (C)</td><td>Ours-LSTM</td><td>Ours-LSTM (C)</td><td>Ours-LSTM</td><td>Ours-LSTM (C)</td></tr><tr><td>2K</td><td>14.39</td><td>24.21</td><td>6.46</td><td>11.27</td><td>13.74</td><td>23.09</td></tr><tr><td>5K</td><td>20.61</td><td>26.83</td><td>9.67</td><td>12.66</td><td>19.58</td><td>25.31</td></tr><tr><td>15K</td><td>25.30</td><td>27.37</td><td>11.76</td><td>12.64</td><td>23.74</td><td>25.69</td></tr><tr><td>30K</td><td>26.86</td><td>27.49</td><td>11.93</td><td>12.75</td><td>25.16</td><td>25.77</td></tr><tr><td>69K</td><td>27.82</td><td>27.89</td><td>12.73</td><td>12.69</td><td>26.01</td><td>26.03</td></tr></table>

Table 3: ROUGE Evaluation for Models with Different Decoder Vocabulary Size. Ours denotes Read-Again. C denotes copy mechanism.  

<table><tr><td>Decoder-Size</td><td>2k</td><td>5k</td><td>15k</td><td>30k</td><td>69k</td></tr><tr><td>Ours-LSTM</td><td>0.076</td><td>0.081</td><td>0.111</td><td>0.161</td><td>0.356</td></tr><tr><td>Ours-LSTM (C)</td><td>0.084</td><td>0.090</td><td>0.123</td><td>0.171</td><td>0.376</td></tr></table>

time, our model can extract their meanings from the input text. Second, equipped with a copy mechanism, our model can generate OOVs as summary words, maintaining its expressive ability even with a small decoder vocabulary size. Tab. 4 shows the decoding time as a function of vocabulary size. As computing the soft-max is usually the bottleneck for decoding, reducing vocabulary size dramatically reduces the decoding time from 0.38 second per sentence to 0.08 second.

Tab. 5 provides some examples of visualization of the copy mechanism. Note that we are able to copy key words from source sentences to improve the summary. From these examples we can see that our model is able to copy different types of rare words, such as special entities' names in case 1 and 2, rare nouns in case 3 and 4, adjectives in case 5 and 6, and even rare verbs in the last example. Note that in the third example, when the copy model's decoder uses the embedding of headmaster as its first input, which is extracted from the source sentence, it generates the same following sentence as the no-copy model. This probably means that the extracted embedding of headmaster is closely related to the learned embedding of teacher.

Table 4: Decoding Time (s) per Sentence of Models with Different Decoder Size  

<table><tr><td>Input:</td><td>air new zealand said friday it had reached agreement to buy a ## percent interest in australia&#x27;s ansett holdings limited for ## million australian -lrb- ## million us dollars -rrb-.</td></tr><tr><td>Golden:</td><td>urgent air new zealand buys ## percent of australia&#x27;s ansett airlines</td></tr><tr><td>No Copy:</td><td>air nz to buy ## percent stake in australia&#x27;s &lt;unk&gt;</td></tr><tr><td>Copy:</td><td>air nz to buy ## percent stake in ansett</td></tr><tr><td>Input:</td><td>yemen&#x27;s ruling party was expected wednesday to nominate president ali abdullah saleh as its candidate for september&#x27;s presidential election, although saleh insisted he is not bluffing about bowing out.</td></tr><tr><td>Golden:</td><td>the ###### gmt news advisory</td></tr><tr><td>No Copy:</td><td>yemen&#x27;s ruling party expected to nominate president as presidential candidate</td></tr><tr><td>Copy:</td><td>yemen&#x27;s ruling party expected to nominate saleh as presidential candidate</td></tr><tr><td>Input:</td><td>a ##-year-old headmaster who taught children in care homes for more than ## years was jailed for ## years on friday after being convicted of ## sexual assaults against his pupils.</td></tr><tr><td>Golden:</td><td>britain : headmaster jailed for ## years for paedophilia</td></tr><tr><td>No Copy:</td><td>teacher jailed for ## years for sexually abusing children</td></tr><tr><td>Copy:</td><td>headmaster jailed for ## years for sexually abusing children</td></tr><tr><td>Input:</td><td>singapore&#x27;s rapidly ageing population poses the major challenge to fiscal policy in the ##st century, finance minister richard hu said, and warned against european-style state &lt;unk&gt;.</td></tr><tr><td>Golden:</td><td>ageing population to pose major fiscal challenge toSingapore</td></tr><tr><td>No Copy:</td><td>finance minister warns against &lt;unk&gt; state</td></tr><tr><td>Copy:</td><td>s pore&#x27;s ageing population poses challenge to fiscal policy</td></tr><tr><td>Input:</td><td>angola is planning to refit its ageing soviet-era fleet of military jets in russian factories, a media report said on tuesday.</td></tr><tr><td>Golden:</td><td>angola to refit jet fighters in russia : report</td></tr><tr><td>No Copy:</td><td>angola to &lt;unk&gt; soviet-era fleet</td></tr><tr><td>Copy:</td><td>angola to refit military fleet in russia</td></tr></table>

Table 5: Visualization of Copy Mechanism

# 5 CONCLUSION

In this paper we have proposed two simple mechanisms to alleviate the problems of current encoder-decoder models. Our first contribution is a 'Read-Again' model which does not form a representation of the input word until the whole sentence is read. Our second contribution is a copy mechanism that can handle out-of-vocabulary words in a principled manner allowing us to reduce the decoder vocabulary size and significantly speed up inference. We have demonstrated the effectiveness of our approach in the context of summarization and shown state-of-the-art performance. In the future, we plan to tackle summarization problems with large input text. We also plan to exploit our findings in other tasks such as machine translation.

# REFERENCES

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.  
Michele Banko, Vibhu O Mittal, and Michael J Witbrock. Headline generation based on statistical translation. In Proceedings of the 38th Annual Meeting on Association for Computational Linguistics, pp. 318-325. Association for Computational Linguistics, 2000.  
Jianpeng Cheng and Mirella Lapata. Neural summarization by extracting sentences and words. arXiv preprint arXiv:1603.07252, 2016.  
Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.  
Sumit Chopra, Michael Auli, Alexander M Rush, and SEAS Harvard. Abstractive sentence summarization with attentive recurrent neural networks. arXiv preprint arXiv:1602.06023, 2016.  
Trevor Cohn and Mirella Lapata. Sentence compression beyond word deletion. In Proceedings of the 22nd International Conference on Computational Linguistics-Volume 1, pp. 137-144. Association for Computational Linguistics, 2008.  
Ronan Collobert, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel Kuksa. Natural language processing (almost) from scratch. Journal of Machine Learning Research, 12(Aug):2493-2537, 2011.  
Carlos A Colmenares, Marina Litvak, Amin Mantrach, and Fabrizio Silvestri. Heads: Headline generation as sequence prediction using an abstract feature-rich space. 2015.  
Günes Erkan and Dragomir R Radev. Lexrank: Graph-based lexical centrality as salience in text summarization. Journal of Artificial Intelligence Research, 22:457-479, 2004.  
Katja Filippova and Yasemin Altun. Overcoming the lack of parallel data in sentence compression. In EMNLP, pp. 1481-1491. CiteSeer, 2013.  
David Graff and Christopher Cieri. English giga-word, 2003. Linguistic Data Consortium, Philadelphia, 2003.  
Jiatao Gu, Zhengdong Lu, Hang Li, and Victor OK Li. Incorporating copying mechanism in sequence-to-sequence learning. arXiv preprint arXiv:1603.06393, 2016.  
Caglar Gulcehre, Sungjin Ahn, Ramesh Nallapati, Bowen Zhou, and Yoshua Bengio. Pointing the unknown words. arXiv preprint arXiv:1603.08148, 2016.  
Baotian Hu, Qingcai Chen, and Fangze Zhu. Lcsts: A large scale chinese short text summarization dataset. arXiv preprint arXiv:1506.05865, 2015.  
Nal Kalchbrenner and Phil Blunsom. Recurrent continuous translation models. In EMNLP, volume 3, pp. 413, 2013.

Minh-Thang Luong, Ilya Sutskever, Quoc V Le, Oriol Vinyls, and Wojciech Zaremba. Addressing the rare word problem in neural machine translation. arXiv preprint arXiv:1410.8206, 2014.  
Ramesh Nallapati, Bowen Zhou, Ca glar Gulçehre, and Bing Xiang. Abstractive text summarization using sequence-to-sequence rnns and beyond. 2016.  
Courtney Naples, Matthew Gormley, and Benjamin Van Durme. Annotated gigaword. In Proceedings of the Joint Workshop on Automatic Knowledge Base Construction and Web-scale Knowledge Extraction, pp. 95-100. Association for Computational Linguistics, 2012.  
Joel Larocca Neto, Alex A Freitas, and Celso AA Kaestner. Automatic text summarization using a machine learning approach. In *Brazilian Symposium on Artificial Intelligence*, pp. 205-215. Springer, 2002.  
Paul Over, Hoa Dang, and Donna Harman. Duc in context. Information Processing & Management, 43(6):1506-1520, 2007.  
Alexander M Rush, Sumit Chopra, and Jason Weston. A neural attention model for abstractive sentence summarization. arXiv preprint arXiv:1509.00685, 2015.  
Mike Schuster and Kuldip K Paliwal. Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing, 45(11):2673-2681, 1997.  
David Sussillo and LF Abbott. Random walk initialization for training very deep feedforward networks. arXiv preprint arXiv:1412.6558, 2014.  
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pp. 3104-3112, 2014.  
Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. In Advances in Neural Information Processing Systems, pp. 2692-2700, 2015.  
Kam-Fai Wong, Mingli Wu, and Wenjie Li. Extractive summarization using supervised and semi-supervised learning. In Proceedings of the 22nd International Conference on Computational Linguistics-Volume 1, pp. 985-992. Association for Computational Linguistics, 2008.  
Kristian Woodsend, Yansong Feng, and Mirella Lapata. Generation with quasi-synchronous grammar. In Proceedings of the 2010 conference on empirical methods in natural language processing, pp. 513-523. Association for Computational Linguistics, 2010.  
David Zajic, Bonnie Dorr, and Richard Schwartz. Bbn/umd at duc-2004: Topiary. In Proceedings of the HLT-NAACL 2004 Document Understanding Workshop, Boston, pp. 112-119, 2004.