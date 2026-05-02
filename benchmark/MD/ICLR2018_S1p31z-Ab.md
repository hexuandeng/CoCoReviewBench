# DEEP CONTEXTUALIZED WORD REPRESENTATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pretrained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pretrained network is crucial, allowing downstream models to mix different types of semi-supervision signals.

# 1 INTRODUCTION

Pretrained word representations (Mikolov et al., 2013; Pennington et al., 2014) are a key component in many neural language understanding models. However, learning high quality representations can be challenging. They should ideally model both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). In this paper, we introduce a new type of deep contextualized word representation that directly addresses both challenges, can be easily integrated into existing models, and significantly improves the state of the art in every considered case across a range of challenging language understanding problems.

Our representations differ from traditional word embeddings in that each word is assigned a representation that is a function of the entire input sentence. We use vectors derived from a bidirectional LSTM that is trained with a coupled language model (LM) objective on a large text corpus. For this reason, we call them ELMo (Embeddings from Language Models) representations. Unlike previous approaches for learning contextualized word vectors (Peters et al., 2017; McCann et al., 2017), ELMo representations are deep, in the sense that they are a function of all of the internal layers of the biLM. More specifically, we learn a linear combination of the vectors stacked above each input word for each end task, which markedly improves performance over just using the top LSTM layer.

Combining the internal states in this manner allows for very rich word representations. We show that, for example, the higher-level LSTM states capture context-dependent aspects of word meaning (e.g., they can be used without modification to perform well on supervised word sense disambiguation tasks) while lower-level states model aspects of syntax (e.g., they can be used to do part-of-speech tagging). Simultaneously exposing all of these signals can be highly beneficial, as models can learn to select the types of semi-supervision that are most useful for each end task.

Extensive experiments demonstrate that ELMo representations work extremely well in practice. We first show that they can be easily added to existing models for six diverse and challenging language understanding problems, including textual entailment, question answering and sentiment analysis. The addition of ELMo representations alone significantly improves the state of the art in every case, including up to  $20\%$  relative error reductions. For tasks where direct comparisons are possible, ELMo outperforms CoVe (McCann et al., 2017), which computes contextualized representations using a neural machine translation encoder. Finally, an analysis of both ELMo and CoVe reveals that deep representations outperform those derived from just the top layer of an LSTM. Our trained models and code will be made publicly available, and we expect that ELMo will provide similar gains for many other NLP problems.<sup>1</sup>

# 2 RELATED WORK

Due to their ability to capture syntactic and semantic information of words from large scale unlabeled text, pretrained word vectors (Turian et al., 2010; Mikolov et al., 2013; Pennington et al., 2014) are a standard component of most state-of-the-art NLP architectures, including for question answering (Wang et al., 2017), textual entailment (Chen et al., 2017) and semantic role labeling (He et al., 2017). However, these approaches for learning word vectors only allow a single, context-independent representation for each word. Another line of research focuses on global methods for learning sentence and document encoders from unlabeled data (e.g., Le & Mikolov, 2014; Kiros et al., 2015; Hill et al., 2016; Conneau et al., 2017), where the goal is to build one representation for an entire text sequence. In contrast, as we will see Section 3, ELMo representations are associated with individual words, but also encode the larger context in which they appear.

Previously-proposed methods overcome some of the shortcomings of traditional word vectors by either enriching them with subword information (e.g., Wieting et al., 2016; Bojanowski et al., 2017) or learning separate vectors for each word sense (e.g., Neelakantan et al., 2014). Our approach also benefits from subword units through the use of character convolutions, and we seamlessly incorporate multi-sense information into downstream tasks without explicitly training to predict predefined sense classes.

Other recent work has also focused on learning context-dependent representations. context2vec (Melamud et al., 2016) uses a bidirectional Long Short Term Memory (LSTM; Hochreiter & Schmidhuber, 1997) to encode the context around a pivot word. Other approaches for learning contextual embeddings include the pivot word itself in the representation and are computed with the encoder of either a supervised neural machine translation (MT) system (CoVe; McCann et al., 2017) or an unsupervised language model (Peters et al., 2017). Both of these approaches benefit from large datasets, although the MT approach is limited by the size of parallel corpora. In this paper, we take full advantage of access to plentiful monolingual data, and train our biLM on a corpus with approximately 30 million sentences (Chelba et al., 2014). We also generalize these approaches to deep contextual representations, which we show work well across a broad range of diverse NLP tasks.

Previous work has also shown that different layers of deep biRNNs encode different types of information. For example, introducing multi-task syntactic supervision (e.g., part-of-speech tags) at the lower levels of a deep LSTM can improve overall performance of higher level tasks such as dependency parsing (Hashimoto et al., 2017) or CCG super tagging (Søgaard & Goldberg, 2016). In an RNN-based encoder-decoder machine translation system, Belinkov et al. (2017) showed that the representations learned at the first layer in a 2-layer LSTM encoder are better at predicting POS tags then second layer. Finally, the top layer of an LSTM for encoding word context (Melamud et al., 2016) has been shown to learn representations of word sense. We show that similar signals are also induced by the modified language model objective of our ELMo representations, and it can be very beneficial to learn models for downstream tasks that mix these different types of semi-supervision.

Similar to computer vision where representations from deep CNNs pretrained on ImageNet are fine tuned for other tasks (Krizhevsky et al., 2012; Shelhamer et al., 2015), Dai & Le (2015) and Ramachandran et al. (2017) pretrain encoder-decoder pairs and then fine tune with task specific supervision. In contrast, after pretraining the biLM with unlabeled data, we fix the weights and add additional task-specific model capacity, allowing us to leverage large, rich and universal biLM representations for cases where downstream training data size dictates a smaller supervised model.

# 3 ELMO: EMBEDDINGS FROM LANGUAGE MODELS

This section details how we compute ELMo representations and use them to improve NLP models. We first present our biLM approach (Sec. 3.1) and then show how ELMo representations are computed on top of them (Sec. 3.2). We also describe how to add ELMo to existing neural NLP architectures (Sec. 3.3), and the details of how the biLM is pretrained (Sec. 3.4).

# 3.1 BIDIRECTIONAL LANGUAGE MODELS

Given a sequence of  $N$  tokens,  $(t_1,t_2,\dots,t_N)$ , a forward language model computes the probability of the sequence by modeling the probability of token  $t_k$  given the history  $(t_1,\dots,t_{k - 1})$ :

$$
p \left(t _ {1}, t _ {2}, \dots , t _ {N}\right) = \prod_ {k = 1} ^ {N} p \left(t _ {k} \mid t _ {1}, t _ {2}, \dots , t _ {k - 1}\right).
$$

Recent state-of-the-art neural language models (Józefowicz et al., 2016; Melis et al., 2017; Merity et al., 2017) compute a context-independent token representation  $\mathbf{x}_k^{LM}$  (via token embeddings or a CNN over characters) then pass it through  $L$  layers of forward LSTMs. At each position  $k$ , each LSTM layer outputs a context-dependent representation  $\vec{\mathbf{h}}_k^{LM,j}$  where  $j = 1, \dots, L$ . The top layer LSTM output,  $\vec{\mathbf{h}}_k^{LM,L}$ , is used to predict the next token  $t_{k+1}$  with a Softmax layer.

A backward LM is similar to a forward LM, except it runs over the sequence in reverse, predicting the previous token given the future context:

$$
p (t _ {1}, t _ {2}, \dots , t _ {N}) = \prod_ {k = 1} ^ {N} p (t _ {k} \mid t _ {k + 1}, t _ {k + 2}, \dots , t _ {N}).
$$

It can be implemented in an analogous way to a forward LM, with each backward LSTM layer  $j$  in a  $L$  layer deep model producing representations  $\overleftarrow{\mathbf{h}}_k^{LM,j}$  of  $t_k$  given  $(t_{k + 1},\ldots ,t_N)$ .

A biLM combines both a forward and backward LM. Our formulation jointly maximizes the log likelihood of the forward and backward directions:

$$
\sum_ {k = 1} ^ {N} \left(\log p (t _ {k} \mid t _ {1}, \dots , t _ {k - 1}; \Theta_ {x}, \overleftrightarrow {\Theta} _ {L S T M}, \Theta_ {s}) + \log p (t _ {k} \mid t _ {k + 1}, \dots , t _ {N}; \Theta_ {x}, \overleftarrow {\Theta} _ {L S T M}, \Theta_ {s})\right).
$$

We tie the parameters for both the token representation  $(\Theta_{x})$  and Softmax layer  $(\Theta_{s})$  in the forward and backward direction while maintaining separate parameters for the LSTMs in each direction. Overall, this formulation is similar to the approach of Peters et al. (2017), with the exception that we share some weights between directions instead of using completely independent parameters. In the next section, we depart from previous work by introducing a new approach for learning word representations that are a linear combination of the biLM layers.

# 3.2 ELMo

ELMo is a task specific combination of the intermediate layer representations in the biLM. For each token  $t_k$ , a  $L$ -layer biLM computes a set of  $2L + 1$  representations

$$
R _ {k} = \left\{\mathbf {x} _ {k} ^ {L M}, \overrightarrow {\mathbf {h}} _ {k} ^ {L M, j}, \overleftarrow {\mathbf {h}} _ {k} ^ {L M, j} \mid j = 1, \dots , L \right\} = \left\{\mathbf {h} _ {k} ^ {L M, j} \mid j = 0, \dots , L \right\},
$$

where  $\mathbf{h}_k^{LM,0}$  is the token layer and  $\mathbf{h}_k^{LM,j} = [\overrightarrow{\mathbf{h}}_k^{LM,j};\overleftarrow{\mathbf{h}}_k^{LM,j}]$ , for each biLSTM layer.

For inclusion in a downstream model, ELMo collapses all layers in  $R$  into a single vector,  $\mathbf{ELM o}_k = E(R_k;\theta_e)$ . In the simplest case, ELMo just selects the top layer,  $E(R_{k}) = \mathbf{h}_{k}^{LM,L}$ , as in TagLM (Peters et al., 2017) and CoVe (McCann et al., 2017). Across the tasks considered, the best performance was achieved by weighting all biLM layers with softmax-normalized learned scalar weights  $\mathbf{s} = S o f t m a x(\mathbf{w})$ :

$$
E \left(R _ {k}; \mathbf {w}, \gamma\right) = \gamma \sum_ {j = 0} ^ {L} s _ {j} \mathbf {h} _ {k} ^ {L M, j}. \tag {1}
$$

The scalar parameter  $\gamma$  allows the task model to scale the entire ELMo vector and is of practical importance to aid the optimization process (see the Appendix for details). Considering that the activations of each biLM layer have a different distribution, in some cases it also helped to apply layer normalization (Ba et al., 2016) to each biLM layer before weighting.

# 3.3 USING BILMS FOR SUPERVISED NLP TASKS

Given a pre-trained biLM and a supervised architecture for a target NLP task, it is a simple process to use the biLM to improve the task model. All of the architectures considered in this paper use RNNs, although the method is equally applicable to CNNs.

We first consider the lowest layers of the supervised model without the biLM. Most RNN based NLP models (including every model in this paper) share a common architecture at the lowest layers, allowing us to add ELMo in a consistent, unified manner. Given a sequence of tokens  $(t_1,\dots ,t_N)$ , it is standard to form a context-independent token representation  $\mathbf{x}_k$  for each token position using pretrained word embeddings and optionally character-based representations (typically from a CNN). Then, one or more layers of bidirectional RNNs compute a context-sensitive representation  $\mathbf{h}_k$  for each token position  $k$ , where  $\mathbf{h}_k$  is the concatenation  $[\vec{\mathbf{h}}_k;\vec{\mathbf{h}}_k]$  of the forward and backward RNNs.

To add ELMo to the supervised model, we first freeze the weights of the biLM and then concatenate the ELMo vector  $\mathbf{ELM o}_{k}$  with  $\mathbf{x}_k$  and pass the ELMo enhanced representation  $[\mathbf{x}_k;\mathbf{ELM o}_k]$  into the task RNN. For some tasks (e.g., SNLI, SQuAD), we observe further improvements by also including ELMo at the output of the task RNN by replacing  $\mathbf{h}_k$  with  $[\mathbf{h}_k;\mathbf{ELM o}_k]$ . As the remainder of the supervised model remains unchanged, these additions can happen within the context of more complex neural models. For example, see the SNLI experiments in Sec. 4 where a bi-attention layer follows the biLSTMs, or the coreference resolution experiments where a clustering model is layered on top of the biLSTMs that compute embeddings for text spans.

Finally, we found it beneficial to add a moderate amount of dropout to ELMo (Srivastava et al., 2014) and in some cases to regularize the ELMo weights by adding  $\lambda \| \mathbf{w} - \frac{1}{L + 1}\| _2^2$  to the loss. This regularization term imposes an inductive bias on the ELMo weights to stay close to an average of all biLM layers.

# 3.4 PRE-TRAINED BIDIRECTIONAL LANGUAGE MODEL ARCHITECTURE

The pre-trained biLMs in this paper are similar to the architectures in Jozefowicz et al. (2016) and Kim et al. (2015), but modified to support joint training of both directions and to include a residual connection between LSTM layers. We focus on biLMs trained at large scale in this work, as Peters et al. (2017) highlighted the importance of using biLMs over forward-only LMs and large scale training. To balance overall language model perplexity with model size and computational requirements for downstream tasks while maintaining a purely character-based input representation, we halved all embedding and hidden dimensions from the single best model CNN-BIG-LSTM in Jozefowicz et al., 2016). The resulting model uses 2048 character n-gram convolutional filters followed by two highway layers (Srivastava et al., 2015) and a linear projection down to a 512 dimension token representation. Each recurrent direction uses two LSTM layers with 4096 units and 512 dimension projections. The average forward and backward perplexities on the 1B Word Benchmark (Chelba et al., 2014) is 39.7, compared to 30.0 for the forward CNN-BIG-LSTM. Generally, we found the forward and backward perplexities to be approximately equal, with the backward value slightly lower.

Fine tuning on task specific data resulted in significant drops in perplexity and an increase in downstream task performance in some cases. This can be seen as a type of domain transfer for the biLM. As a result, in most cases we used a fine-tuned biLM in the downstream task. See the Appendix for details.

# 4 EVALUATION

Table 1 shows the performance of ELMo across a diverse set of six benchmark NLP tasks. In every task considered, simply adding ELMo establishes a new state-of-the-art result, with relative error reductions ranging from  $6 - 20\%$  over strong base models. This is a very general result across a diverse set model architectures and language understanding tasks. In the remainder of this section we provide high-level sketches of the individual task results; see the Appendix for full experimental details.

Table 1: Test set comparison of ELMo enhanced neural models with state-of-the-art single model baselines across six benchmark NLP tasks. The performance metric varies across tasks – accuracy for SNLI and SST-5;  $\mathrm{F}_1$  for SQuAD, SRL and NER; average  $\mathrm{F}_1$  for Coref. Due to the small test sizes for NER and SST-5, we report the mean and standard deviation across five runs with different random seeds. The "increase" column lists both the absolute and relative improvements over our baseline.  

<table><tr><td>TASK</td><td colspan="2">PREVIOUS SOTA</td><td>OUR BASELINE</td><td>ELMO + BASELINE</td><td>INCREASE (ABSOLUTE/RELATIVE)</td></tr><tr><td>SNLI</td><td>McCann et al. (2017)</td><td>88.1</td><td>88.0</td><td>88.7 ± 0.17</td><td>0.7 / 5.8%</td></tr><tr><td>SQuAD2</td><td>r-net Wang et al. (2017)</td><td>84.3</td><td>81.1</td><td>85.3</td><td>4.2 / 22.2%</td></tr><tr><td>SRL</td><td>He et al. (2017)</td><td>81.7</td><td>81.4</td><td>84.6</td><td>3.2 / 17.2%</td></tr><tr><td>Coref</td><td>Lee et al. (2017)</td><td>67.2</td><td>67.2</td><td>70.4</td><td>3.2 / 9.8%</td></tr><tr><td>NER</td><td>Peters et al. (2017)</td><td>91.93 ± 0.19</td><td>90.15</td><td>92.22 ± 0.10</td><td>2.06 / 21%</td></tr><tr><td>SST-5</td><td>McCann et al. (2017)</td><td>53.7</td><td>51.4</td><td>54.7 ± 0.5</td><td>3.3 / 6.8%</td></tr></table>

Textual entailment Textual entailment is the task of determining whether a "hypothesis" is true, given a "premise". The Stanford Natural Language Inference (SNLI) corpus (Bowman et al., 2015) provides approximately 550K hypothesis/premise pairs. Our baseline, the ESIM sequence model from Chen et al. (2017), uses a biLSTM to encode the premise and hypothesis, followed by a matrix attention layer, a local inference layer, another biLSTM inference composition layer, and finally a pooling operation before the output layer. Overall, adding ELMo to the ESIM model improves accuracy by an average of  $0.7\%$  across five random seeds, increasing the single model state-of-the-art by  $0.6\%$  over the CoVe enhanced model from McCann et al. (2017). A five member ensemble pushes the overall accuracy to  $89.3\%$ , exceeding the previous ensemble best of  $88.9\%$  (Gong et al., 2017) - see Appendix for details.

Question answering The Stanford Question Answering Dataset (SQuAD) (Rajpurkar et al., 2016) contains  $100\mathrm{K}+$  crowd sourced question-answer pairs where the answer is a span in a given Wikipedia paragraph. Our baseline model (Clark & Gardner, 2017) is an improved version of the Bidirectional Attention Flow model in Seo et al. (BiDAF; 2017). It adds a self-attention layer after the bidirectional attention component, simplifies some of the pooling operations and substitutes the LSTMs for gated recurrent units (GRUs; Cho et al., 2014). After adding ELMo to the baseline model, test set  $\mathbf{F}_1$  improved by  $4.2\%$  from  $81.1\%$  to  $85.3\%$ , improving the single model state-of-the-art by  $1.0\%$ .

Semantic role labeling A semantic role labeling (SRL) system models the predicate-argument structure of a sentence, and is often described as answering "Who did what to whom". SRL is a challenging NLP task as it requires jointly extracting the arguments of a predicate and establishing their semantic roles. He et al. (2017) modeled SRL as a BIO tagging problem and used an 8-layer deep biLSTM with forward and backward directions interleaved, following Zhou & Xu (2015). As shown in Table 1, when adding ELMo to a re-implementation of He et al. (2017) the single model test set  $\mathrm{F}_1$  jumped  $3.2\%$  from  $81.4\%$  to  $84.6\%$  – a new state-of-the-art on the OntoNotes benchmark (Pradhan et al., 2013), even improving over the previous best ensemble result by  $1.2\%$  (see Table 10 in the Appendix).

Coreference resolution Coreference resolution is the task of clustering mentions in text that refer to the same underlying real world entities. Our baseline model is the end-to-end span-based neural model of Lee et al. (2017). It uses a biLSTM and attention mechanism to first compute span representations and then applies a softmax mention ranking model to find coreference chains. In our experiments with the OntoNotes coreference annotations from the CoNLL 2012 shared task (Pradhan et al., 2012), adding ELMo improved the average  $\mathrm{F}_1$  by  $3.2\%$  from 67.2 to 70.4, establishing a new state of the art, again improving over the previous best ensemble result by  $1.6\%$ $\mathrm{F}_1$  (see Table 11 in the Appendix).

Named entity extraction The CoNLL 2003 NER task (Sang & Meulder, 2003) consists of newswire from the Reuters RCV1 corpus tagged with four different entity types (PER, LOC, ORG, MISC). Fol

Table 2: Development set performance for SQuAD, SNLI and SRL comparing using all layers of the biLM (with different choices of regularization strength  $\lambda$ ) to just the top layer.

<table><tr><td rowspan="2">Task</td><td rowspan="2">Baseline</td><td rowspan="2">Last Only</td><td colspan="2">All layers</td></tr><tr><td>λ=1</td><td>λ=0.001</td></tr><tr><td>SQuAD</td><td>80.8</td><td>82.5</td><td>83.6</td><td>84.8</td></tr><tr><td>SNLI</td><td>88.1</td><td>89.1</td><td>89.3</td><td>89.5</td></tr><tr><td>SRL</td><td>81.6</td><td>84.1</td><td>84.6</td><td>84.8</td></tr></table>

Table 3: Development set performance for SQuAD, SNLI and SRL when including ELMo at different locations in the supervised model.  

<table><tr><td>Task</td><td>Input Only</td><td>Input &amp; Output</td><td>Output Only</td></tr><tr><td>SQuAD</td><td>84.2</td><td>84.8</td><td>83.7</td></tr><tr><td>SNLI</td><td>88.9</td><td>89.5</td><td>88.7</td></tr><tr><td>SRL</td><td>84.7</td><td>84.3</td><td>80.9</td></tr></table>

lowing recent state-of-the-art systems (Lample et al., 2016; Peters et al., 2017), the baseline model is a biLSTM-CRF based sequence tagger. It forms a token representation by concatenating pre-trained word embeddings with a character-based CNN representation, passes it through two layers of biLSTMs, and then computes the sentence conditional random field (CRF) loss (Lafferty et al., 2001) during training and decodes with the Viterbi algorithm during testing, similar to Collobert et al. (2011). As shown in Table 1, our ELMo enhanced biLSTM-CRF achieves  $92.22\%$ $\mathrm{F_1}$  averaged over five runs. The key difference between our system and the previous state of the art from Peters et al. (2017) is that we allowed the task model to learn a weighted average of all biLM layers, whereas Peters et al. (2017) only use the top biLM layer. As shown in Sec. 5.1, using all layers instead of just the last layer improves performance across multiple tasks.

Sentiment analysis The fine-grained sentiment classification task in the Stanford Sentiment Treebank (SST-5; Socher et al., 2013) involves selecting one of five labels (from very negative to very positive) to describe a sentence from a movie review. The sentences contain diverse linguistic phenomena such as idioms, named entities related to film, and complex syntactic constructions (e.g., negations) that are difficult for models to learn directly from the training dataset alone. Our baseline model is the biattentive classification network (BCN) from McCann et al. (2017), which also held the prior state-of-the-art result when augmented with CoVe embeddings. Replacing CoVe with ELMo in the BCN model results in a  $1.0\%$  absolute accuracy improvement over the state of the art.

# 5 ANALYSIS

This section provides an ablation analysis to validate our chief claims and to elucidate some interesting aspects of ELMo representations. Sec. 5.1 shows that using deep contextual representations in downstream tasks improves performance over previous work that uses just the top layer, regardless of whether they are produced from a biLM or MT encoder, and that ELMo representations provide the best overall performance. Sec. 5.3 explores the different types of contextual information captured in biLMs and confirms that syntactic information is better represented at lower layers while semantic information is captured a higher layers, consistent with MT encoders. It also shows that our biLM consistently provides richer representations than CoVe. Additionally, we analyze the sensitivity to where ELMo is included in the task model (Sec. 5.2), training set size (Sec. 5.4), and visualize the ELMo learned weights across the tasks (Sec. 5.5).

# 5.1 ALTERNATE LAYER WEIGHTING SCHEMES

There are many alternatives to Equation 1 for combining the biLM layers. Previous work on contextual representations use only the last layer, whether it be from a biLM (Peters et al., 2017) or an MT encoder (CoVe; McCann et al., 2017). The choice of the regularization parameter  $\lambda$  is also important, as large values such as  $\lambda = 1$  effectively reduce the weighting function to a simple average over the layers, while smaller values (e.g.,  $\lambda = 0.001$ ) allows the layer weights to vary.

Table 2 compares these alternatives for SNLI, SRL and SQuAD. Including representations from all layers improves overall performance over just using the last layer, and including contextual representations from the last layer improves performance over the baseline. For example, in the case of SQuAD, using just the last biLM layer improves development  $\mathrm{F}_1$  by  $1.7\%$  over the baseline. Aver

Table 4: Nearest neighbors to "play" using GloVe and the context embeddings from a biLM.  

<table><tr><td></td><td>Source</td><td>Nearest Neighbors</td></tr><tr><td>GloVe</td><td>play</td><td>playing, game, games, played, players, plays, player, Play, football, multiplayer</td></tr><tr><td rowspan="2">biLM</td><td>Chico Ruiz made a spec-tacular play on Alusik &#x27;s grounder {...}</td><td>Kieffer , the only junior in the group , was commended for his ability to hit in the clutch , as well as his all-round excellent play .</td></tr><tr><td>Olivia De Havilland signed to do a Broadway play for Garson {...}</td><td>{...} they were actors who had been handed fat roles in a successful play , and had talent enough to fill the roles competently , with nice understatement .</td></tr></table>

aging all biLM layers instead of using just the last layer improves  $\mathrm{F}_1$  another  $1.1\%$  (comparing "Last Only" to  $\lambda = 1$  columns), and allowing the task model to learn individual layer weights improves  $\mathrm{F}_1$  another  $1.2\%$  ( $\lambda = 1$  vs.  $\lambda = 0.001$ ). A small  $\lambda$  is preferred in most cases with ELMo, although for NER, a task with a smaller training set, the results are insensitive to  $\lambda$  (not shown).

The overall trend is similar with CoVe but with smaller increases over the baseline. In the case of SNLI, weighting all layers with  $\lambda = 1$  improves development accuracy from 88.2 to  $88.7\%$  over using just the last layer. SRL  $\mathrm{F}_1$  increased a marginal  $0.1\%$  to 82.2 for the  $\lambda = 1$  case compared to using the last layer only.

# 5.2 WHERE TO INCLUDE ELMO?

All of the task architectures in this paper include word embeddings only as input to the lowest layer biRNN. However, we find that including ELMo at the output of the biRNN in task-specific architectures improves overall results for some tasks. As shown in Table 3, including ELMo at both the input and output layers for SNLI and SQuAD improves over just the input layer, but for SRL (and coreference resolution, not shown) performance is highest when it is included at just the input layer. One possible explanation for this result is that both the SNLI and SQuAD architectures use attention layers after the biRNN, so introducing ELMo at this layer allows the supervised model to attend directly to the biLM's internal representations. In the SRL case, the task-specific context representations are likely more important than those from the biLM.

# 5.3 WHAT INFORMATION IS CAPTURED BY THE BILM'S REPRESENTATIONS?

Since adding ELMo improves task performance over word vectors alone, the biLM's contextual representations must encode information generally useful for NLP tasks that is not captured in word vectors. Intuitively, the biLM must be disambiguating the meaning of words using their context. Consider "play", a highly polysemous word. The top of Table 4 lists nearest neighbors to "play" using GloVe vectors. They are spread across several parts of speech (e.g., "played", "playing" as verbs, and "player", "game" as nouns) but concentrated in the sports-related senses of "play". In contrast, the bottom two rows show nearest neighbor sentences from the SemCor dataset (see below) using the biLM's context representation of "play" in the source sentence. In these cases, the biLM is able to disambiguate both the part of speech and word sense in the source sentence.

These observations can be quantified using an approach similar to Belinkov et al. (2017). To isolate the information encoded by the biLM, the representations are used to directly make predictions for a fine grained word sense disambiguation (WSD) task and a POS tagging task. Using this approach, it is also possible to compare to CoVe, and across each of the individual layers.

Word sense disambiguation Given a sentence, we can use the biLM representations to predict the sense of a target word using a simple 1-nearest neighbor approach, similar to Melamud et al. (2016). To do so, we first use the biLM to compute representations for all words in SemCor 3.0, our training corpus (Miller et al., 1994), and then take the average representation for each sense. At test time, we again use the biLM to compute representations for a given target word and take the nearest neighbor sense from the training set, falling back to the first sense from WordNet for lemmas not observed during training.

Table 5: All-words fine grained WSD  $F_{1}$ . For CoVe and the biLM, we report scores for both the first and second layer biLSTMs.  

<table><tr><td>Model</td><td>F1</td></tr><tr><td>WordNet 1st Sense Baseline</td><td>65.9</td></tr><tr><td>Raganato et al. (2017a)</td><td>69.9</td></tr><tr><td>Iacobacci et al. (2016)</td><td>70.1</td></tr><tr><td>CoVe, First Layer</td><td>59.4</td></tr><tr><td>CoVe, Second Layer</td><td>64.7</td></tr><tr><td>biLM, First layer</td><td>67.4</td></tr><tr><td>biLM, Second layer</td><td>69.0</td></tr></table>

Table 6: Test set POS tagging accuracies for PTB. For CoVe and the biLM, we report scores for both the first and second layer biLSTMs.  

<table><tr><td>Model</td><td>Acc.</td></tr><tr><td>Collobert et al. (2011)</td><td>97.27</td></tr><tr><td>Ma &amp; Hovy (2016)</td><td>97.55</td></tr><tr><td>Ling et al. (2015)</td><td>97.78</td></tr><tr><td>CoVe, First Layer</td><td>93.3</td></tr><tr><td>CoVe, Second Layer</td><td>92.8</td></tr><tr><td>biLM, First Layer</td><td>97.0</td></tr><tr><td>biLM, Second Layer</td><td>95.8</td></tr></table>

![](images/9afccfd966c1f232a0bbe23b59a8e6a0cc0b7c8e03ea6b62205dbed4544a6796.jpg)  
Figure 1: Comparison of baseline vs. ELMo performance for SNLI and SRL as the training set size is varied from  $0.1\%$  to  $100\%$ .

![](images/297ef601c9e09c5e9ceec57afc33632d6dbd90a96482732a706eb78f70edaa53.jpg)  
Figure 2: Visualization of softmax normalized biLM layer weights across tasks and ELMo locations. Normalized weights less than  $1/3$  are hatched with horizontal lines and those greater then  $2/3$  are speckled.

Table 5 compares WSD results using the evaluation framework from Raganato et al. (2017b) across the same suite of four test sets in Raganato et al. (2017a). Overall, the biLM top layer representations have  $\mathrm{F}_1$  of 69.0 and are better at WSD then the first layer. This is competitive with a state-of-the-art WSD-specific supervised model using hand crafted features (Iacobacci et al., 2016) and a task specific biLSTM that is also trained with auxiliary coarse-grained semantic labels and POS tags (Raganato et al., 2017a). The CoVe biLSTM layers follow a similar pattern to those from the biLM (higher overall performance at the second layer compared to the first); however, our biLM outperforms the CoVe biLSTM, which trails the WordNet first sense baseline.

POS tagging To examine whether the biLM captures basic syntax, we used the context representations as input to a linear classifier that predicts POS tags with the Wall Street Journal portion of the Penn Treebank (PTB) (Marcus et al., 1993). As the linear classifier adds only a tiny amount of model capacity, this is direct test of the biLM's representations. Similar to WSD, the biLM representations are competitive with carefully tuned, task specific biLSTMs with character representations (Ling et al., 2015; Ma & Hovy, 2016). However, unlike WSD, accuracies using the first biLM layer are higher than the top layer, consistent with results from deep biLSTMs in multi-task training (Søgaard & Goldberg, 2016; Hashimoto et al., 2017) and MT (Belinkov et al., 2017). CoVe POS tagging accuracies follow the same pattern as those from the biLM, and just like for WSD, the biLM achieves higher accuracies than the CoVe encoder.

Implications for supervised tasks Taken together, these experiments confirm different layers in the biLM represent different types of information and explain why including all biLM layers is important for the highest performance in downstream tasks. In addition, the biLM's representations are more transferable to WSD and POS tagging than those in CoVe, which helps illustrate why ELMo outperforms CoVe in downstream tasks.

# 5.4 SAMPLE EFFICIENCY

Adding ELMo to a model increases the sample efficiency considerably, both in terms of number of parameter updates to reach state-of-the-art performance and the overall training set size. For

example, the SRL model reaches a maximum development  $\mathrm{F}_1$  after 486 epochs of training without ELMo. After adding ELMo, the model exceeds the baseline maximum at epoch 10, a 98% relative decrease in the number of updates needed to reach the same level of performance.

In addition, ELMo-enhanced models use smaller training sets more efficiently than models without ELMo. Figure 1 compares the performance of baselines models with and without ELMo as the percentage of the full training set is varied from  $0.1\%$  to  $100\%$ . Improvements with ELMo are largest for smaller training sets and significantly reduce the amount of training data needed to reach a given level of performance. In the SRL case, the ELMo model with  $1\%$  of the training set has about the same  $\mathrm{F}_1$  as the baseline model with  $10\%$  of the training set.

# 5.5 VISUALIZATION OF LEARNED WEIGHTS

Figure 2 visualizes the softmax-normalized learned layer weights across the tasks. At the input layer, in all cases, the task model favors the first biLSTM layer, with the remaining emphasis split between the token layer and top biLSTM in task specific ways. For coreference and SQuAD, the first LSTM layer is strongly favored, but the distribution is less peaked for the other tasks. It is an interesting question for future work to understand why the first biLSTM layer is universally favored. The output layer weights are relatively balanced, with a slight preference for the lower layers.

# 6 CONCLUSION AND FUTURE WORK

We have introduced a general approach for learning high-quality deep context-dependent representations from biLMs, and shown large improvements when applying ELMo to a broad range of NLP tasks. Through ablations and other controlled experiments, we have also confirmed that the biLM layers efficiently encode different types of syntactic and semantic information about words-in-context, and that using all layers improves overall task performance.

Our approach raises several interesting questions for future work, broadly organized into two themes.

"What is the best training regime for learning generally useful NLP representations?" By choosing a biLM training objective, we benefit from nearly limitless unlabeled text and can immediately apply advances in language modeling, an active area of current research. However, it's possible that further decreases in LM perplexity will not translate to more transferable representations, and that other objective functions might be more suitable for learning general purpose representations.

"What is the best way to use deep contextual representations for other tasks?" Our method of using a weighted average of all layers from the biLM is simple and empirically successful. However, a deeper fusion of the biLM layers with a target NLP architecture may lead to further improvements.

# REFERENCES

Jimmy Ba, Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. CoRR, abs/1607.06450, 2016.  
Yonatan Belinkov, Nadir Durrani, Fahim Dalvi, Hassan Sajjad, and James R. Glass. What do neural machine translation models learn about morphology? In ACL, 2017.  
Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. Enriching word vectors with subword information. TACL, 5:135-146, 2017.  
Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2015.  
Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, and Phillip Koehn. One billion word benchmark for measuring progress in statistical language modeling. CoRR, abs/1312.3005, 2014.  
Qian Chen, Xiao-Dan Zhu, Zhen-Hua Ling, Si Wei, Hui Jiang, and Diana Inkpen. Enhanced LSTM for natural language inference. In ACL, 2017.

Jason Chiu and Eric Nichols. Named entity recognition with bidirectional LSTM-CNNs. In TACL, 2016.  
Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, and Yoshua Bengio. On the properties of neural machine translation: Encoder-decoder approaches. In SSST@EMNLP, 2014.  
Christopher Clark and Matthew Gardner. Advancing multi-paragraph reading comprehension. arXiv preprint, 2017.  
Kevin Clark and Christopher D. Manning. Deep reinforcement learning for mention-ranking coreference models. In EMNLP, 2016.  
Ronan Collobert, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel P. Kuksa. Natural language processing (almost) from scratch. In JMLR, 2011.  
Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, and Antoine Bordes. Supervised learning of universal sentence representations from natural language inference data. In EMNLP, 2017.  
Andrew M. Dai and Quoc V. Le. Semi-supervised sequence learning. In NIPS, 2015.  
Greg Durrett and Dan Klein. Easy victories and uphill battles in coreference resolution. In EMNLP, 2013.  
Yarin Gal and Zoubin Ghahramani. A theoretically grounded application of dropout in recurrent neural networks. In NIPS, 2016.  
Yichen Gong, Heng Luo, and Jian Zhang. Natural language inference over interaction space. CoRR, abs/1709.04348, 2017.  
Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, and Richard Socher. A joint many-task model: Growing a neural network for multiple nlp tasks. In EMNLP, 2017.  
Luheng He, Kenton Lee, Mike Lewis, and Luke S. Zettlemoyer. Deep semantic role labeling: What works and what's next. In ACL, 2017.  
Felix Hill, Kyunghyun Cho, and Anna Korhonen. Learning distributed representations of sentences from unlabelled data. In HLT-NAACL, 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation, 9, 1997.  
Ignacio Iacobacci, Mohammad Taher Pilehvar, and Roberto Navigli. Embeddings for word sense disambiguation: An evaluation study. In ACL, 2016.  
Rafal Józefowicz, Wojciech Zaremba, and Ilya Sutskever. An empirical exploration of recurrent network architectures. In ICML, 2015.  
Rafal Józefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. CoRR, abs/1602.02410, 2016.  
Yoon Kim, Yacine Jernite, David Sontag, and Alexander M Rush. Character-aware neural language models. In AAAI 2016, 2015.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.  
Jamie Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Skip-thought vectors. In NIPS, 2015.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Ankit Kumar, Ozan Irsoy, Peter Ondruska, Mohit Iyyer, Ishaan Gulrajani James Bradbury, Victor Zhong, Romain Paulus, and Richard Socher. Ask me anything: Dynamic memory networks for natural language processing. In ICML, 2016.

John D. Lafferty, Andrew McCallum, and Fernando Pereira. Conditional random fields: Probabilistic models for segmenting and labeling sequence data. In ICML, 2001.  
Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. Neural architectures for named entity recognition. In *NAACL-HLT*, 2016.  
Quoc V. Le and Tomas Mikolov. Distributed representations of sentences and documents. In ICML, 2014.  
Kenton Lee, Luheng He, Mike Lewis, and Luke S. Zettlemoyer. End-to-end neural coreference resolution. In EMNLP, 2017.  
Wang Ling, Chris Dyer, Alan W. Black, Isabel Trancoso, Ramon Fernandez, Silvio Amir, Luís Marujo, and Tiago Luís. Finding function in form: Compositional character models for open vocabulary word representation. In EMNLP, 2015.  
Xuezhe Ma and Eduard H. Hovy. End-to-end sequence labeling via bi-directional LSTM-CNNs-CRF. In ACL, 2016.  
Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of english: The penn treebank. Computational Linguistics, 19:313-330, 1993.  
Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. Learned in translation: Contextualized word vectors. CoRR, abs/1708.00107, 2017.  
Oren Melamud, Jacob Goldberger, and Ido Dagan. context2vec: Learning generic context embedding with bidirectional LSTM. In CoNLL, 2016.  
Gábor Melis, Chris Dyer, and Phil Blunsom. On the state of the art of evaluation in neural language models. CoRR, abs/1707.05589, 2017.  
Stephen Merity, Nitish Shirish Keskar, and Richard Socher. Regularizing and optimizing LSTM language models. CoRR, abs/1708.02182, 2017.  
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In NIPS, 2013.  
George A. Miller, Martin Chodorow, Shari Landes, Claudia Leacock, and Robert G. Thomas. Using a semantic concordance for sense identification. In HLT, 1994.  
Tsendsuren Munkhdalai and Hong Yu. Neural tree indexers for text understanding. In EACL, 2017.  
Arvind Neelakantan, Jeevan Shankar, Alexandre Passos, and Andrew McCallum. Efficient non-parametric estimation of multiple embeddings per word in vector space. In EMNLP, 2014.  
Martha Palmer, Paul Kingsbury, and Daniel Gildea. The proposition bank: An annotated corpus of semantic roles. Computational Linguistics, 31:71-106, 2005.  
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. Glove: Global vectors for word representation. In EMNLP, 2014.  
Matthew E. Peters, Waleed Ammar, Chandra Bhagavatula, and Russell Power. Semi-supervised sequence tagging with bidirectional language models. In ACL, 2017.  
Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Olga Uryupina, and Yuchen Zhang. Conll-2012 shared task: Modeling multilingual unrestricted coreference in ontonotes. In EMNLP-ConNLL Shared Task, 2012.  
Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Hwee Tou Ng, Anders Björkelund, Olga Uryupina, Yuchen Zhang, and Zhi Zhong. Towards robust linguistic analysis using onthonotes. In CoNLL, 2013.  
Alessandro Raganato, Claudio Delli Bovi, and Roberto Navigli. Neural sequence learning models for word sense disambiguation. In EMNLP, 2017a.

Alessandro Raganato, Jose Camacho-Collados, and Roberto Navigli. Word sense disambiguation: A unified evaluation framework and empirical comparison. In EACL, 2017b.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100, 000+ questions for machine comprehension of text. In EMNLP, 2016.  
Prajit Ramachandran, Peter Liu, and Quoc Le. Improving sequence to sequence learning with unlabeled data. In EMNLP, 2017.  
Erik F. Tjong Kim Sang and Fien De Meulder. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In CoNLL, 2003.  
Min Joon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. Bidirectional attention flow for machine comprehension. In ICLR, 2017.  
Evan Shelhamer, Jonathan Long, and Trevor Darrell. Fully convolutional networks for semantic segmentation. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3431-3440, 2015.  
Richard Socher, Alex Perelygin, Jean Y Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In EMNLP, 2013.  
Anders Søgaard and Yoav Goldberg. Deep multi-task learning with low level tasks supervised at lower layers. In ACL, 2016.  
Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15:1929-1958, 2014.  
Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. Training very deep networks. In NIPS, 2015.  
Joseph P. Turian, Lev-Arie Ratinov, and Yoshua Bengio. Word representations: A simple and general method for semi-supervised learning. In ACL, 2010.  
Wenhui Wang, Nan Yang, Furu Wei, Baobao Chang, and Ming Zhou. Gated self-matching networks for reading comprehension and question answering. In ACL, 2017.  
John Wieting, Mohit Bansal, Kevin Gimpel, and Karen Livescu. Charagram: Embedding words and sentences via character n-grams. In EMNLP, 2016.  
Sam Wiseman, Alexander M. Rush, and Stuart M. Shieber. Learning global features for coreference resolution. In HLT-NAACL, 2016.  
Matthew D. Zeiler. Adadelta: An adaptive learning rate method. CoRR, abs/1212.5701, 2012.  
Jie Zhou and Wei Xu. End-to-end learning of semantic role labeling using recurrent neural networks. In ACL, 2015.  
Peng Zhou, Zhenyu Qi, Suncong Zheng, Jiaming Xu, Hongyun Bao, and Bo Xu. Text classification improved by integrating bidirectional LSTM with two-dimensional max pooling. In COLING, 2016.
