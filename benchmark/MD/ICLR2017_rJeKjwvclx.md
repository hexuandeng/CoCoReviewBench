# DYNAMIC COATTENTION NETWORKS FOR QUESTION ANSWERING

Caiming Xiong\*, Victor Zhong\* Richard Socher

Salesforce Research

Palo Alto, CA 94301, USA

{cxiong, vzhong, rsocher}@salesforce.com

# ABSTRACT

Several deep learning models have been proposed for question answering. However, due to their single-pass nature, they have no way to recover from local maxima corresponding to incorrect answers. To address this problem, we introduce the Dynamic Coattention Network (DCN) for question answering. The DCN first fuses co-dependent representations of the question and the document in order to focus on relevant parts of both. Then a dynamic pointing decoder iterates over potential answer spans. This iterative procedure enables the model to recover from initial local maxima corresponding to incorrect answers. On the Stanford question answering dataset, a single DCN model improves the previous state of the art from  $71.0\%$  F1 to  $75.9\%$ , while a DCN ensemble obtains  $80.4\%$  F1.

# 1 INTRODUCTION

Question answering (QA) is a crucial task in natural language processing that requires both natural language understanding and world knowledge. Previous QA datasets tend to be high in quality due to human annotation, but small in size (Berant et al., 2014; Richardson et al., 2013). Hence, they did not allow for training data-intensive, expressive models such as deep neural networks.

To address this problem, researchers have developed large-scale datasets through semi-automated techniques (Hermann et al., 2015; Hill et al., 2015). Compared to their smaller, hand-annotated counterparts, these QA datasets allow the training of more expressive models. However, it has been shown that they differ from more natural, human annotated datasets in the types of reasoning required to answer the questions (Chen et al., 2016).

Recently, Rajpurkar et al. (2016) released the Stanford Question Answering dataset (SQuAD), which is orders of magnitude larger than all previous hand-annotated datasets and has a variety of qualities that culminate in a natural QA task. SQuAD has the desirable quality that answers are spans in a reference document. This constrains answers to the space of all possible spans. However, Rajpurkar et al. (2016) show that the dataset retains a diverse set of answers and requires different forms of logical reasoning, including multi-sentence reasoning.

We introduce the Dynamic Coattention Network (DCN), illustrated in Fig. 1, an end-to-end neural network for question answering. The model consists of a coattentive encoder that captures the interactions between the question and the document, as well as a dynamic pointing decoder that alternates between estimating the start and end of the answer span. Our single model obtains an F1 of  $75.9\%$  compared to the best published result of  $71.0\%$  (Yu et al., 2016). In addition, our ensemble model obtains an F1 of  $80.4\%$  compared to the second best result of  $78.1\%$  on the official SQuAD leaderboard.<sup>1</sup>

# 2 DYNAMIC COATTENTION NETWORKS

![](images/e5838e2a0d677b85eb837060295e6b2786aabd8b49337f5d06f06ef44a203d02.jpg)  
Figure 1 illustrates an overview of the DCN. We first describe the encoders for the document and the question, followed by the coattention mechanism and the dynamic decoder which produces the answer span.  
Figure 1: Overview of the Dynamic Coattention Network.

# 2.1 DOCUMENT AND QUESTION ENCODER

Let  $(x_1^Q, x_2^Q, \ldots, x_n^Q)$  denote the sequence of word vectors corresponding to words in the question and  $(x_1^D, x_2^D, \ldots, x_m^D)$  denote the same for words in the document. Using an LSTM (Hochreiter & Schmidhuber, 1997), we encode the document as:  $d_t = \mathrm{LSTM}_{enc}(d_{t-1}, x_t^D)$ . We define the document encoding matrix as  $D = [d_1 \ldots d_n d_\varnothing] \in \mathbb{R}^{\ell \times (m+1)}$ . We also add a sentinel vector  $d_\varnothing$  (Merit et al., 2016), which we later show allows the model to not attend to any particular word in the input.

The question embeddings are computed with the same LSTM to share representation power:  $q_{t} = \mathrm{LSTM}_{enc}\left(q_{t - 1},x_{t}^{Q}\right)$ . We define an intermediate question representation  $Q^{\prime} = [q_{1}\dots q_{m}q_{\emptyset}]\in \mathbb{R}^{\ell \times (n + 1)}$ . To allow for variation between the question encoding space and the document encoding space, we introduce a non-linear projection layer on top of the question encoding. The final representation for the question becomes:  $Q = \tanh \left(W^{(Q)}Q^{\prime} + b^{(Q)}\right)\in \mathbb{R}^{\ell \times (n + 1)}$ .

# 2.2 COATTENTION ENCODER

We propose a coattention mechanism that attends to the question and document simultaneously, similar to (Lu et al., 2016), and finally fuses both attention contexts. Figure 2 provides an illustration of the coattention encoder.

We first compute the affinity matrix, which contains affinity scores corresponding to all pairs of document words and question words:  $L = D^{\top}Q \in \mathbb{R}^{(m + 1)\times (n + 1)}$ . The affinity matrix is normalized row-wise to produce the attention weights  $A^Q$  across the document for each word in the question, and column-wise to produce the attention weights  $A^D$  across the question for each word in the document:

$$
A ^ {Q} = \operatorname {s o f t m a x} (L) \in \mathbb {R} ^ {(m + 1) \times (n + 1)} \text {a n d} A ^ {D} = \operatorname {s o f t m a x} \left(L ^ {\top}\right) \in \mathbb {R} ^ {(n + 1) \times (m + 1)} \tag {1}
$$

Next, we compute the summaries, or attention contexts, of the document in light of each word of the question.

$$
C ^ {Q} = D A ^ {Q} \in \mathbb {R} ^ {\ell \times (n + 1)}. \tag {2}
$$

![](images/2284861280ee86a8360607cd89e4c364bc7bb6e8a24bb8b216b47670b2c5ef12.jpg)  
Figure 2: Coattention encoder. The affinity matrix  $L$  is not shown here. We instead directly show the normalized attention weights  $A^{D}$  and  $A^{Q}$ .

We similarly compute the summaries  $QA^{D}$  of the question in light of each word of the document. Similar to Cui et al. (2016), we also compute the summaries  $C^Q A^D$  of the previous attention contexts in light of each word of the document. These two operations can be done in parallel, as is shown in Eq. 3. One possible interpretation for the operation  $C^Q A^D$  is the mapping of question encoding into space of document encodings.

$$
C ^ {D} = \left[ Q; C ^ {Q} \right] A ^ {D} \in \mathbb {R} ^ {2 \ell \times (m + 1)}. \tag {3}
$$

We define  $C^D$ , a co-dependent representation of the question and document, as the coattention context. We use the notation  $[a; b]$  for concatenating the vectors  $a$  and  $b$  horizontally.

The last step is the fusion of temporal information to the coattention context via a bidirectional LSTM:

$$
u _ {t} = \operatorname {B i - L S T M} \left(u _ {t - 1}, u _ {t + 1}, \left[ d _ {t}; c _ {t} ^ {D} \right]\right) \in \mathbb {R} ^ {2 \ell}. \tag {4}
$$

We define  $U = [u_{1},\dots ,u_{m}]\in \mathbb{R}^{\ell \times m}$ , which provides a foundation for selecting which span may be the best possible answer, as the coattention encoding.

# 2.3 DYNAMIC POINTING DECODER

Due to the nature of SQuAD, an intuitive method for producing the answer span is by predicting the start and end points of the span (Wang & Jiang, 2016). However, given a question-document pair, there may exist several intuitive answer spans within the document, each corresponding to a local maxima. We propose an iterative technique to select an answer span by alternating between predicting the start point and predicting the end point. This iterative procedure allows the model to recover from initial local maxima corresponding to incorrect answer spans.

Figure 3 provides an illustration of the Dynamic Decoder, which is similar to a state machine whose state is maintained by an LSTM-based sequential model. During each iteration, the decoder updates its state taking into account the coattention encoding corresponding to current estimates of the start and end positions, and produces, via a multilayer neural network, new estimates of the start and end positions.

Let  $h_i, s_i$ , and  $e_i$  denote the hidden state of the LSTM, the estimate of the position, and the estimate of the end position during iteration  $i$ . The LSTM state update is then described by Eq. 5.

$$
h _ {i} = \operatorname {L S T M} _ {d e c} \left(h _ {i - 1}, \left[ u _ {s _ {i - 1}}; u _ {e _ {i - 1}} \right]\right) \tag {5}
$$

where  $u_{s_{i-1}}$  and  $u_{e_{i-1}}$  are the representations corresponding to the previous estimate of the start and end positions in the coattention encoding  $U$ .

![](images/aad4d8252b6886a804e63a3a0c772254e9a0a45199dc54451b9c7e1f94797ddc.jpg)  
Figure 3: Dynamic Decoder. Blue denotes the variables and functions related to estimating the start position whereas red denotes the variables and functions related to estimating the end position.

Given the current hidden state  $h_i$ , previous start position  $u_{s_{i-1}}$ , and previous end position  $u_{e_{i-1}}$ , we estimate the current start position and end position via Eq. 6 and Eq. 7.

$$
s _ {i} = \underset {t} {\operatorname {a r g m a x}} (\alpha_ {1}, \dots , \alpha_ {m}) \tag {6}
$$

$$
e _ {i} = \underset {t} {\operatorname {a r g m a x}} \left(\beta_ {1}, \dots , \beta_ {m}\right) \tag {7}
$$

where  $\alpha_{t}$  and  $\beta_{t}$  represent the start score and end score corresponding to the  $t$ th word in the document. We compute  $\alpha_{t}$  and  $\beta_{t}$  with separate neural networks. These networks have the same architecture but do not share parameters.

Based on the strong empirical performance of Maxout Networks (Goodfellow et al., 2013) and Highway Networks (Srivastava et al., 2015), especially with regards to deep architectures, we propose a Highway Maxout Network (HMN) to compute  $\alpha_{t}$  as described by Eq. 8. The intuition behind using such model is that the QA task consists of multiple question types and document topics. These variations may require different models to estimate the answer span. Maxout provides a simple and effective way to pool across multiple model variations.

$$
\alpha_ {t} = \operatorname {H M N} _ {\text {s t a r t}} \left(u _ {t}, h _ {i}, u _ {s _ {i - 1}}, u _ {e _ {i - 1}}\right) \tag {8}
$$

Here,  $u_{t}$  is the coattention encoding corresponding to the  $t$ th word in the document.  $\mathrm{HMN}_{start}$  is illustrated in Figure 4. The end score,  $\beta_{t}$ , is computed similarly to the start score  $\alpha_{t}$ , but using a separate  $\mathrm{HMN}_{end}$ .

We now describe the HMN model:

$$
\operatorname {H M N} \left(u _ {t}, h _ {i}, u _ {s _ {i - 1}}, u _ {e _ {i - 1}}\right) = \max  \left(W ^ {(3)} \left[ m _ {t} ^ {(1)}; m _ {t} ^ {(2)} \right] + b ^ {(3)}\right) \tag {9}
$$

$$
r = \tanh  \left(W ^ {(D)} \left[ h _ {i}; u _ {s _ {i - 1}}; u _ {e _ {i - 1}} \right]\right) \tag {10}
$$

$$
m _ {t} ^ {(1)} = \max  \left(W ^ {(1)} [ u _ {t}; r ] + b ^ {(1)}\right) \tag {11}
$$

$$
m _ {t} ^ {(2)} = \max  \left(W ^ {(2)} m _ {t} ^ {(1)} + b ^ {(2)}\right) \tag {12}
$$

where  $r\in \mathbb{R}^{\ell}$  is a non-linear projection of the current state with parameters  $W^{(D)}\in \mathbb{R}^{\ell \times 5\ell}$ ,  $m_t^{(1)}$  is the output of the first maxout layer with parameters  $W^{(1)}\in \mathbb{R}^{p\times \ell \times 6\ell}$  and  $b^{(1)}\in \mathbb{R}^{p\times \ell}$ , and  $m_t^{(2)}$  is the output of the second maxout layer with parameters  $W^{(2)}\in \mathbb{R}^{p\times \ell \times \ell}$  and  $b^{(2)}\in \mathbb{R}^{p\times \ell}$ .  $m_t^{(1)}$  and  $m_t^{(2)}$  are fed into the final maxout layer, which has parameters  $W^{(3)}\in \mathbb{R}^{p\times 1\times 2\ell}$ , and  $b^{(3)}\in \mathbb{R}^p$ .  $p$  is the pooling size of each maxout layer. The max operation computes the maximum value over the first dimension of a tensor. We note that there is a highway connection between the output of the first maxout layer and the last maxout layer.

To train the network, we minimize the cumulative softmax cross entropy of the start and end points across all iterations. The iterative procedure halts when both the estimate of the start position and the estimate of the end position no longer change, or when a maximum number of iterations is reached. Details can be found in Section 4.1

![](images/ef183ab2ba0b80086469e83dff1bc9441282c50e7868bc661840eda8ae07d39e.jpg)  
Figure 4: Highway Maxout Network. Dotted lines denote highway connections.

# 3 RELATED WORK

Statistical QA Traditional approaches to question answering typically involve rule-based algorithms or linear classifiers over hand-engineered feature sets. Richardson et al. (2013) proposed two baselines, one that uses simple lexical features such as a sliding window to match bags of words, and another that uses word-distances between words in the question and in the document. Berant et al. (2014) proposed an alternative approach in which one first learns a structured representation of the entities and relations in the document in the form of a knowledge base, then converts the question to a structured query with which to match the content of the knowledge base. Wang & McAllester (2015) described a statistical model using frame semantic features as well as syntactic features such as part of speech tags and dependency parses. Chen et al. (2016) proposed a competitive statistical baseline using a variety of carefully crafted lexical, syntactic, and word order features.

Neural QA Neural attention models have been widely applied for machine comprehension or question-answering in NLP. Hermann et al. (2015) proposed an AttentiveReader model with the release of the CNN/Daily Mail cloze-style question answering dataset. Hill et al. (2015) released another dataset stemming from the children's book and proposed a window-based memory network. Kadlec et al. (2016) presented a pointer-style attention mechanism but performs only one attention step. Sordoni et al. (2016) introduced an iterative neural attention model and applied it to cloze-style machine comprehension tasks.

Recently, Rajpurkar et al. (2016) released the SQuAD dataset. Different from cloze-style queries, answers include non-entities and longer phrases, and questions are more realistic. For SQuAD, Wang & Jiang (2016) proposed an end-to-end neural network model that consists of a Match-LSTM encoder, originally introduced in Wang & Jiang (2015), and a pointer network decoder (Vinyals et al., 2015); Yu et al. (2016) introduced a dynamic chunk reader, a neural reading comprehension model that extracts a set of answer candidates of variable lengths from the document and ranks them to answer the question.

Lu et al. (2016) proposed a hierarchical co-attention model for visual question answering, which achieved state of the art result on the COCO-VQA dataset (Antol et al., 2015). In (Lu et al., 2016), the co-attention mechanism computes a conditional representation of the image given the question, as well as a conditional representation of the question given the image.

Inspired by the above works, we propose a dynamic coattention model (DCN) that consists of a novel coattentive encoder and dynamic decoder. In our model, instead of estimating the start and end positions of the answer span in a single pass (Wang & Jiang, 2016), we iteratively update the

start and end positions in a similar fashion to the Iterative Conditional Modes algorithm (Besag, 1986).

# 4 EXPERIMENTS

# 4.1 IMPLEMENTATION DETAILS

We train and evaluate our model on the SQuAD dataset. To preprocess the corpus, we use the tokenizer from Stanford CoreNLP (Manning et al., 2014). We use as GloVe word vectors pretrained on the 840B Common Crawl corpus (Pennington et al., 2014). We limit the vocabulary to words that are present in the Common Crawl corpus and set embeddings for out-of-vocabulary words to zero. Empirically, we found that training the embeddings consistently led to overfitting and subpar performance, and hence only report results with fixed word embeddings.

We use a max sequence length of 600 during training and a hidden state size of 200 for all recurrent units, maxout layers, and linear layers. For the dynamic decoder, we set the maximum number of iterations to 4 and use a maxout pool size of 16. We use dropout to regularize our network during training (Srivastava et al., 2014), and optimize the model using ADAM (Kingma & Ba, 2014). All models are implemented and trained with Chainer (Tokui et al., 2015).

# 4.2 RESULTS

Evaluation on the SQuAD dataset consists of two metrics. The exact match score (EM) calculates the exact string match between the predicted answer and a ground truth answer. The F1 score calculates the overlap between words in the predicted answer and a ground truth answer. Because a document-question pair may have several ground truth answers, the EM and F1 for a document-question pair is taken to be the maximum value across all ground truth answers. The overall metric is then computed by averaging over all document-question pairs. The official SQuAD evaluation is hosted on CodaLab<sup>2</sup>. The training and development sets are publicly available while the test set is withheld.

<table><tr><td>Model</td><td>Dev EM</td><td>Dev F1</td><td>Test EM</td><td>Test F1</td></tr><tr><td colspan="5">Ensemble</td></tr><tr><td>DCN (Ours)</td><td>70.3</td><td>79.4</td><td>71.2</td><td>80.4</td></tr><tr><td>Microsoft Research Asia *</td><td>-</td><td>-</td><td>69.4</td><td>78.3</td></tr><tr><td>Allen Institute *</td><td>69.2</td><td>77.8</td><td>69.9</td><td>78.1</td></tr><tr><td>Singapore Management University *</td><td>67.6</td><td>76.8</td><td>67.9</td><td>77.0</td></tr><tr><td>Google NYC *</td><td>68.2</td><td>76.7</td><td>-</td><td>-</td></tr><tr><td colspan="5">Single model</td></tr><tr><td>DCN (Ours)</td><td>65.4</td><td>75.6</td><td>66.2</td><td>75.9</td></tr><tr><td>Microsoft Research Asia *</td><td>65.9</td><td>75.2</td><td>65.5</td><td>75.0</td></tr><tr><td>Google NYC *</td><td>66.4</td><td>74.9</td><td>-</td><td>-</td></tr><tr><td>Singapore Management University *</td><td>-</td><td>-</td><td>64.7</td><td>73.7</td></tr><tr><td>Carnegie Mellon University *</td><td>-</td><td>-</td><td>62.5</td><td>73.3</td></tr><tr><td>Dynamic Chunk Reader (Yu et al., 2016)</td><td>62.5</td><td>71.2</td><td>62.5</td><td>71.0</td></tr><tr><td>Match-LSTM (Wang &amp; Jiang, 2016)</td><td>59.1</td><td>70.0</td><td>59.5</td><td>70.3</td></tr><tr><td>Baseline (Rajpurkar et al., 2016)</td><td>40.0</td><td>51.0</td><td>40.4</td><td>51.0</td></tr><tr><td>Human (Rajpurkar et al., 2016)</td><td>81.4</td><td>91.0</td><td>82.3</td><td>91.2</td></tr></table>

Table 1: Leaderboard performance at the time of writing (Nov 4 2016). * indicates that the model used for submission is unpublished. - indicates that the development scores were not publicly available at the time of writing.

The performance of the Dynamic Coattention Network on the SQuAD dataset, compared to other submitted models on the leaderboard  $^{3}$ , is shown in Table 1. At the time of writing, our single-model DCN ranks first at  $66.2\%$  exact match and  $75.9\%$  F1 on the test data among single-model submissions. Our ensemble DCN ranks first overall at  $71.6\%$  exact match and  $80.4\%$  F1 on the test data.

The DCN has the capability to estimate the start and end points of the answer span multiple times, each time conditioned on its previous estimates. By doing so, the model is able to explore local maxima corresponding to multiple plausible answers, as is shown in Figure 5.

![](images/0436a488f4ccd01f75b2d7904d76abbc50b7ab96abc9d1da89263e8011c3d618.jpg)  
Question 1: Who recovered Tolbert's fumble?

![](images/6c47290117ab97dd548201cbd038cc8ba173cd65514500bd69240dd41a8b4e0b.jpg)  
Question 2: What did the Kenyan business people hope for when meeting with the Chinese?

![](images/712ca6576178627c84647ed9d6a9485765649ce6ad6d77bbfa3349995cfde97a.jpg)  
Question 3: What kind of weapons did Tesla's treatise concern?  
Figure 5: Examples of the start and end conditional distributions produced by the dynamic decoder. Odd (blue) rows denote the start distributions and even (red) rows denote the end distributions.  $i$  indicates the iteration number of the dynamic decoder. Higher probability mass is indicated by darker regions. The offset corresponding to the word with the highest probability mass is shown on the right hand side. The predicted span is underlined in red, and a ground truth answer span is underlined in green.

For example, Question 1 in Figure 5 demonstrates an instance where the model initially guesses an incorrect start point and a correct end point. In subsequent iterations, the model adjusts the start point, ultimately arriving at the correct start point in iteration 3. Similarly, the model gradually shifts probability mass for the end point to the correct word.

Question 2 shows an example in which both the start and end estimates are initially incorrect. The model then settles on the correct answer in the next iteration.

![](images/d02ab4318f8edf87951a222d42779b0e435f66912cf09a4fdc3b05f75bb2a0ad.jpg)  
Figure 6: Performance of the DCN for various lengths of documents, questions, and answers. The blue dot indicates the mean F1 at given length. The vertical bar represents the standard deviation of F1s at a given length.

![](images/f97ba0d3b2e15543177416d4a9a7162d530a387ff4ddd813664d8d694cfb6443.jpg)

![](images/11bdbe767f4bed4e03f132052bf77634687e8face7210049cd93a95a0e294e05.jpg)

While the dynamic nature of the decoder allows the model to escape initial local maxima corresponding to incorrect answers, Question 3 demonstrates a case where the model is unable to decide between multiple local maxima despite several iterations. Namely, the model alternates between the answers "charged particle beam" and "particle beam weapons" indefinitely. Empirically, we observe that the model, trained with a maximum iteration of 4, takes 2.7 iterations to converge to an answer on average.

Model Ablation The performance of our model and its ablations on the SQuAD development set is shown in Table 2. On the decoder side, we experiment with various pool sizes for the HMN maxout layers, using a 2-layer MLP instead of a HMN, and forcing the HMN decoder to a single iteration. Empirically, we achieve the best performance on the development set with an iterative HMN

<table><tr><td>Model</td><td>Dev EM</td><td>Dev F1</td></tr><tr><td colspan="3">Dynamic Coattention Network (DCN)</td></tr><tr><td>pool size 16 HMN</td><td>65.4</td><td>75.6</td></tr><tr><td>pool size 8 HMN</td><td>64.4</td><td>74.9</td></tr><tr><td>pool size 4 HMN</td><td>65.2</td><td>75.2</td></tr><tr><td>DCN with 2-layer MLP instead of HMN</td><td>63.8</td><td>74.4</td></tr><tr><td>DCN with single iteration decoder</td><td>63.7</td><td>74.0</td></tr><tr><td>DCN with Wang &amp; Jiang (2016) attention</td><td>63.7</td><td>73.7</td></tr></table>

Table 2: Single model ablations on the development set.

with pool size 16, and find that the model consistently benefits from a deeper, iterative decoder network. On the encoder side, replacing the coattention mechanism with an attention mechanism similar to Wang & Jiang (2016) by setting  $C^D$  to  $C^Q$  in equation 3 results in a 1.9 point F1 drop. This suggests that, at an additional cost of a softmax computation and a dot product, the coattention mechanism provides a simple and effective means to better encode the document and question sequences.

Performance across length One point of interest is how the performance of the DCN varies with respect to the length of document. Intuitively, we expect the model performance to deteriorate with longer examples, as is the case with neural machine translation (Luong et al., 2015). However, as in shown in Figure 6, there is no notable performance degradation for longer documents and questions contrary to our expectations. This suggests that the coattentive encoder is largely agnostic to long documents, and is able to focus on small sections of relevant text while ignoring the rest of the (potentially very long) document. We do note a performance degradation with longer answers. However, this is intuitive given the nature of the evaluation metric. Namely, it becomes increasingly challenging to compute the correct word span as the number of words increases.

![](images/6a463c19df0bc1557c73351908da535d8b7927365b4a446a4b9a61eaa4123c7e.jpg)  
Figure 7: Performance of the DCN across question types. The height of each bar represents the mean F1 for the given question type. The lower number denotes how many instances in the dev set are of the corresponding question type.

Performance across question type Another natural way to analyze the performance of the model is to examine its performance across question types. In Figure 7, we note that the mean F1 of DCN exceeds those of previous systems (Wang & Jiang, 2016; Yu et al., 2016) across all question types. The DCN, like other models, is adept at "when" questions and struggles with the more complex "why" questions.

Breakdown of F1 distribution Finally, we note that the DCN performance is highly bimodal. On the development set, the model perfectly predicts (100% F1) an answer for 62.2% of examples and predicts a completely wrong answer (0% F1) for 16.3% of examples. That is, the model picks out partial answers only 21.5% of the time. Upon qualitative inspections of the 0% F1 answers, some of which are shown in Appendix A.2, we observe that when the model is wrong, its mistakes tend to have the correct "answer type" (eg. person for a "who" question, method for a "how" question) and the answer boundaries encapsulate a well-defined phrase.

# 5 CONCLUSION

We proposed the Dynamic Coattention Network, an end-to-end neural network architecture for question answering. The DCN consists of a coattention encoder which learns co-dependent representations of the question and of the document, and a dynamic decoder which iteratively estimates the answer span. We showed that the iterative nature of the model allows it to recover from initial local maxima corresponding to incorrect predictions. On the SQuAD dataset, the DCN achieves the state of the art results at  $75.9\%$  F1 with a single model and  $80.4\%$  F1 with an ensemble. The DCN significantly outperforms all other models.

# ACKNOWLEDGMENTS

We thank Kazuma Hashimoto and Bryan McCann for their help and insights.

# REFERENCES

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE International Conference on Computer Vision, pp. 2425-2433, 2015.  
Jonathan Berant, Vivek Srikumar, Pei-Chun Chen, Abby Vander Linden, Brittany Harding, Brad Huang, Peter Clark, and Christopher D Manning. Modeling biological processes for reading comprehension. In EMNLP, 2014.  
Julian Besag. On the statistical analysis of dirty pictures. Journal of the Royal Statistical Society. Series B (Methodological), pp. 259-302, 1986.  
Danqi Chen, Jason Bolton, and Christopher D Manning. A thorough examination of the cnn/daily mail reading comprehension task. arXiv preprint arXiv:1606.02858, 2016.  
Yiming Cui, Zhipeng Chen, Si Wei, Shijin Wang, Ting Liu, and Guoping Hu. Attention-over-attention neural networks for reading comprehension. arXiv preprint arXiv:1607.04423, 2016.  
Ian J Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron C Courville, and Yoshua Bengio. Maxout networks. ICML (3), 28:1319-1327, 2013.  
Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems, pp. 1693-1701, 2015.  
Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston. The goldilocks principle: Reading children's books with explicit memory representations. arXiv preprint arXiv:1511.02301, 2015.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Rudolf Kadlec, Martin Schmid, Ondrej Bajgar, and Jan Kleindienst. Text understanding with the attention sum reader network. arXiv preprint arXiv:1603.01547, 2016.

Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh. Hierarchical question-image co-attention for visual question answering. arXiv preprint arXiv:1606.00061, 2016.  
Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.  
Christopher D Manning, Mihai Surdeanu, John Bauer, Jenny Rose Finkel, Steven Bethard, and David McClosky. The stanford corenlp natural language processing toolkit. In ACL (System Demonstrations), pp. 55-60, 2014.  
Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.  
Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In EMNLP, volume 14, pp. 1532-43, 2014.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250, 2016.  
Matthew Richardson, Christopher JC Burges, and Erin Renshaw. Mctest: A challenge dataset for the open-domain machine comprehension of text. In EMNLP, volume 3, pp. 4, 2013.  
Alessandro Sordoni, Phillip Bachman, and Yoshua Bengio. Iterative alternating neural attention for machine reading. arXiv preprint arXiv:1606.02245, 2016.  
Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929-1958, 2014.  
Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. Highway networks. arXiv preprint arXiv:1505.00387, 2015.  
Seiya Tokui, Kenta Oono, Shohei Hido, and Justin Clayton. Chainer: a next-generation open source framework for deep learning. In Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Twenty-ninth Annual Conference on Neural Information Processing Systems (NIPS), 2015.  
Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. In Advances in Neural Information Processing Systems, pp. 2692-2700, 2015.  
Hai Wang and Mohit Bansal Kevin Gimpel David McAllester. Machine comprehension with syntax, frames, and semantics. Volume 2: Short Papers, pp. 700, 2015.  
Shuohang Wang and Jing Jiang. Learning natural language inference with LSTM. arXiv preprint arXiv:1512.08849, 2015.  
Shuohang Wang and Jing Jiang. Machine comprehension using match-lstm and answer pointer. arXiv preprint arXiv:1608.07905, 2016.  
Y. Yu, W. Zhang, K. Hasan, M. Yu, B. Xiang, and B. Zhou. End-to-End Reading Comprehension with Dynamic Answer Chunk Ranking. ArXiv eprints, October 2016.  
Yang Yu, Wei Zhang, Kazi Hasan, Mo Yu, Bing Xiang, and Bowen Zhou. End-to-end answer chunk extraction and ranking for reading comprehension. arXiv preprint arXiv:1610.09996v2, 2016.
