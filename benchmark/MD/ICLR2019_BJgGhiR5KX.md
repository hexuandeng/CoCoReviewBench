# LEARNING CROSS-LINGUAL SENTENCE REPRESENTATIONS VIA A MULTI-TASK DUAL-ENCODER MODEL

Anonymous authors

Paper under double-blind review

# ABSTRACT

Neural language models have been shown to achieve an impressive level of performance on a number of language processing tasks. The majority of these models, however, are limited to producing predictions for only English texts due to limited amounts of labeled data available in other languages. One potential method for overcoming this issue is learning cross-lingual text representations that can be used to transfer the performance from training on English tasks to non-English tasks, despite little to no task-specific non-English data. In this paper, we explore a natural setup for learning cross-lingual sentence representations: the dual-encoder. We provide a comprehensive evaluation of our cross-lingual representations on a number of monolingual, cross-lingual, and zero-shot/few-shot learning tasks, and also give an analysis of different learned cross-lingual embedding spaces.

# 1 INTRODUCTION

There has been a significant amount of recent work on developing models that can produce sentence representations that are useful for a number of language processing tasks (Kiros et al., 2015; Conneau et al., 2017; Subramanian et al., 2018; Logeswaran & Lee, 2018; Cer et al., 2018). However, these models are trained on largely monolingual data, and can thus only be used for tasks in a single language. A promising direction for extending the previous models to multiple languages is learning cross-lingual embedding spaces (Schwenk et al., 2017; Eriguchi et al., 2018; Singla et al., 2018), which could be used to transfer performance in one language to others.

We develop a novel approach for cross-lingual representation learning by combining the dual-encoder architectures used for learning sentence representations (Logeswaran & Lee, 2018; Cer et al., 2018) and for bi-text retrieval (Guo et al., 2018). By doing so, we learn representations that maintain state-of-the-art performance in tasks for a source language while simultaneously obtaining state-of-the-art performance in zero-shot learning tasks for a target language. For a given source-target language pair, we construct a multi-task training scheme using native source language tasks, native target language tasks, and a bridging source-target translation task to learn sentence representations that are aligned between the source and target languages. We then evaluate the learned representations on several monolingual and cross-lingual tasks, and also provide a graph-based analysis of the learned representations.

We find that multi-task training using additional monolingual tasks improves performance over models that only make use of parallel data on both cross-lingual semantic textual similarity (STS) (Cer et al., 2017) and Søgaard et al. (2018)'s cross-lingual eigen-similarity metric. The results show that the addition of monolingual data actually improves the embedding alignment of sentences and their translations. Furthermore, we find that cross-lingual training with additional monolingual data leads to far better transfer learning performance, and we show that our cross-lingual representations outperform state-of-the-art zero-shot learning systems in sentiment classification and natural language inference.

# 2 MULTI-TASK DUAL-ENCODER MODEL

The core of our approach is the idea of modeling various tasks as ranking input-response pairs by encoding them via two encoders, with the crucial task for learning cross-lingual representations

![](images/c58bfc13bec8e6e76328c2bfa74ee1cb6ef7dc0e7b6b53df167834a068be03e3.jpg)  
Figure 1: Multi-task dual-encoder model. It consists of a group of native tasks in each language and a bridging task using translation pair data. The encoders in the gray box all share their parameters, and thus constitute  $g$ . The Quick Thought task can be treated as a variation of the dual-encoder model, where we combine sentence-predecessor and sentence-successor models.

being translation ranking. For translation ranking, as well as for our other tasks, we take an input sentence  $s_i^I$  and an associated response sentence  $s_i^R$ , and we seek to rank  $s_i^R$  over all other possible response sentences  $s_j^R \in S^R$ . To do so, we model the conditional probability  $P(s_i^R \mid s_i^I)$  as:

$$
P \left(s _ {i} ^ {R} \mid s _ {i} ^ {I}\right) = \frac {e ^ {\phi \left(s _ {i} ^ {I} , s _ {i} ^ {R}\right)}}{\sum_ {s _ {j} ^ {R} \in \mathcal {S} ^ {R}} e ^ {\phi \left(s _ {i} ^ {R} , s _ {j} ^ {R}\right)}}, \quad \phi \left(s _ {i} ^ {I}, s _ {j} ^ {R}\right) = g ^ {I} \left(s _ {i} ^ {I}\right) ^ {\top} g ^ {R} \left(s _ {j} ^ {R}\right) \tag {1}
$$

Where  $g^{I}$  and  $g^{R}$  are the input and response sentence encoding functions that compose the dual-encoder. Since the normalization term in equation 1 is computationally intractable, we follow the approaches of Henderson et al. (2017) and instead choose to model an approximate conditional probability  $\widetilde{P}(s_{i}^{R} \mid s_{i}^{I})$ :

$$
\widetilde {P} \left(s _ {i} ^ {R} \mid s _ {i} ^ {I}\right) = \frac {e ^ {\phi \left(s _ {i} ^ {I} , s _ {i} ^ {R}\right)}}{\sum_ {j = 1 , j \neq i} ^ {K} e ^ {\phi \left(s _ {i} ^ {R} , s _ {j} ^ {R}\right)}} \tag {2}
$$

Where  $K$  denotes the size of a single batch of training examples, and the  $s_j^R$  correspond to the response sentences associated with the other input sentences in the same batch as  $s_i^I$ . We parametrize  $g^I$  and  $g^R$  as deep neural networks that are trained to minimize the negative log-likelihood of  $\widetilde{P}(s_i^R \mid s_i^I)$  for each task.

In order to produce a single sentence encoding function  $g$  that can be evaluated on downstream tasks, we share several layers between the input and response encoders and treat the final output of these shared layers as  $g$ . Additionally, these layers are modeled after the Universal Sentence Encoder (USE) model of Cer et al. (2018), since it is the state-of-the-art model that is most amenable to our setup. To learn cross-lingual representations, we train  $g$  on several symmetric tasks<sup>1</sup> for the source-target language pairs English-French (en-fr), English-Spanish (en-es), and English-German (en-de). The resulting model structure is illustrated in Figure 1.

# 2.1 ENCODER ARCHITECTURE

Word and Character Embeddings. As part of the training process for learning the cross-lingual sentence encoding function  $g$ , we learn embeddings for the words and characters present in the training data for a given source-target language pair. Word embeddings are learned end-to-end. Character embeddings are learned in a similar manner, but with the added stipulation that we consider character n-gram embeddings instead of single character embeddings by using a single feedforward layer with

tanh activation on top of character n-grams. Each word in an input sentence then obtains a character embedding representation by having its character n-gram representations summed together. To have the sentence encoder  $g$  leverage the word and character embeddings together in a computationally efficient way, we sum the word and character embeddings before using them as input to  $g$ .

Transformer Encoder. The actual architecture of the shared encoder  $g$  consists of three layers of transformer stacks, which contain the feed-forward and multi-head attention sub-layers described in Vaswani et al. (2017). The transformer encoder output is a variable-length sequence at each stack. We average encodings of all sequence positions in the final layer as the final sentence encoding. This embedding is then fed into different sets of feedforward layers that are used for each task.

# 2.2 MULTI-TASK TRAINING SETUP

To learn a function  $g$  that is capable of strong cross-lingual matching and transfer learning performance for a source-target language pair while also maintaining monolingual downstream task performance, we employ four unique task types for each language pair. Specifically, we employ a conversation response prediction task, a quick thought task, a natural language inference task, and a bridging task - translation ranking. Six total tasks are used in training, as the first two tasks are mirrored across languages.

Conversation Response Prediction. We model the conversation response prediction task in the same manner as Yang et al. (2018). We minimize the negative log-likelihood of  $\widetilde{P}(s_i^R \mid s_i^I)$ , where  $s_i^I$  is a single comment and  $s_i^R$  is its associated response comment. For the response side, we model  $g^R(s_i^R)$  as two fully-connected feedforward layers of size 320 and 512 with  $\tanh$  activation on top of  $g(s_i^R)$ . For the input side, however, we simply let  $g^I(s_i^I) = g(s_i^I)$ , as we noticed in early experiments that letting the optimization of the conversational response task more directly influence the parameters of the underlying sentence encoder  $g$  led to better downstream task performance.

Quick Thought. We use a modified version of the Quick Thought task detailed by Logeswaran & Lee (2018). We minimize the sum of the negative log-likelihoods of  $\widetilde{P}(s_i^R \mid s_i^I)$  and  $\widetilde{P}(s_i^P \mid s_i^I)$ , where  $s_i^I$  is a sentence taken from an article and  $s_i^P$  and  $s_i^R$  are its predecessor and successor sentences respectively. For this task, we model all three of  $g^P(s_i^P)$ ,  $g^I(s_i^I)$ , and  $g^R(s_i^R)$  using separate, fully-connected feedforward layers of size 320 and 512 with tanh activation on top of  $g$ , as we did for  $g^R(s_i^R)$  in our conversational modeling task.

Natural Language Inference (NLI). We also include an English-only natural language inference task based on Bowman et al. (2015). For this task, we first encode an input sentence  $s_i^I$  and its corresponding response hypothesis  $s_i^R$  into vectors  $u_1$  and  $u_2$  using  $g$ . The vectors  $u_1, u_2$  are then used to construct a feature vector  $(u_1, u_2, |u_1 - u_2|, u_1 * u_2)$ , where  $(\cdot)$  represents concatenation and  $*$  represents element-wise multiplication. The form of this feature vector is derived from the original experiments of Bowman et al. (2015). This feature vector is then fed into a single feedforward layer of size 512 that is used to perform the 3-way NLI classification.

Translation Ranking. Our translation task setup is identical to the one used by Guo et al. (2018) for bi-text retrieval. We minimize the negative log-likelihood of  $\widetilde{P}(s_i \mid t_i)$ , where  $(s_i, t_i)$  is a source-target translation pair. Since the translation task is intended to align the sentence representations for the source and target languages, we do not use any kind of task-specific feedforward layers and instead use  $g$  as both  $g^I$  and  $g^R$ . Following Guo et al. (2018), we append 5 translations that are similar to the correct translation to each training batch as "hard-negatives". We did not see additional gains from using more than 5 hard-negatives.

# 3 EXPERIMENTS

# 3.1 CORPORA

We draw upon multiple, openly available data sources and training corpora for the training of the tasks mentioned above. Specifically, we use extracted and preprocessed comments from Reddit for conversational response prediction, multilingual dumps of Wikipedia for Quick Thought, a bi-text-retrieval-based translation corpus for translation ranking, and the Stanford Natural Language Infer

ence data for natural language inference. Our data preprocessing procedures are described in the supplementary material.

Reddit. We preprocess the Reddit data extracted by Al-Rfou et al. (2016) into 600 million input-response comment pairs for training our conversation response prediction task. We also translate this data using the Google neural machine translation (NMT) system of Wu et al. (2016).

Wikipedia. To get native, non-English data, we extract triplets of contiguous sentences from English, French, Spanish, and German articles take from Wikipedia. Our final extracted corpus of Wikipedia sentence triplets consists of 127.9, 49.5, 29.8, and 49.3 million triplets for English, French, Spanish, and German respectively, which we use to train our Quick Thought task.

Stanford Natural Language Inference (SNLI). The NLI data we use is taken from the Stanford Natural Language Inference (SNLI) dataset of Bowman et al. (2015), which consists of 570K sentence pairs associated with one of three labels: entailment, contradiction, or neutral. The corpus is split into training (550K), validation (10K), and testing sets (10K).

Translation. The data for training the translation task is constructed using a system similar to the approach described by Guo et al. (2018). The final constructed corpus contains around 600M en-fr pairs, 470M en-es pairs and 500M en-de pairs.

# 3.2 MODEL CONFIGURATION

In all of our experiments, multi-task training is done by cycling through the different tasks (translation pairs, Reddit, Wikipedia, NLI) and performing an optimization step for a single task at a time. We train all of our models with a batch size of 100 using stochastic gradient descent with a learning rate of 0.008. All of our models are trained for 30 million steps or until they converge. All input text is tokenized prior to being used for training. We build a vocab containing 200 thousand unigram tokens with 10 thousand hash buckets for out-of-vocabulary tokens. The character n-gram vocab contains 200 thousand hash buckets used for 3 and 4 grams. Both the word and character n-gram embedding sizes are 320. All hyperparameters are tuned based on preliminary experiments on a development set. Finally, as an additional training heuristic, we multiply the gradients to the word and character embeddings by a factor of 100. We found that using this embedding gradient multiplier alleviated vanishing gradient issues and greatly improved training.

We compare the proposed cross-lingual multi-task models with baseline models that are trained using only the translation ranking task, which we dub as the "translation-ranking" models.

# 3.3 MODEL PERFORMANCE ON ENGLISH DOWNSSTREAM TASKS

We first evaluated our cross-lingual multi-task models on several downstream English tasks to verify the impact of adapting the Universal Sentence Encoder model to include cross-lingual language tasks and translation data. These results are summarized in Table 1. We note that multi-task training does not hinder the effectiveness of our encoder on English tasks, as the multi-task models are close to state-of-the-art in each of the downstream tasks. For the Text REtrieval Conference (TREC) eval, we actually find that our multi-task models outperform the previous state-of-the-art models by a sizable amount.

Table 1 also includes the results for our translation-ranking models on the same downstream English tasks. We include these results mainly to gauge the level of semantic information that can be learned from using only translation pair data. As expected, the performance of the translation-ranking models is significantly worse than that of the multi-task models.

# 3.4 CROSS-LINGUAL RETRIEVAL

We also evaluate both the multi-task and translation-ranking models' efficacy in performing crosslingual retrieval by using held-out translation pair data. Following Henderson et al. (2017), we use precision at  $\mathrm{N}(\mathrm{P}@\mathrm{N})$  as the evaluation metric by checking if a source sentence's target translation ranks (where ranking is done using dot product) in the top  $N$  scored candidates when considering  $K$  other randomly selected target sentences. Unlike Henderson et al. (2017), we set  $K$  to be 999 instead

Table 1: Performance on classification transfer tasks.  

<table><tr><td>Model</td><td>MR</td><td>CR</td><td>SUBJ</td><td>MPQA</td><td>TREC</td><td>SST</td><td>STS Bench (dev / test)</td></tr><tr><td colspan="8">Cross-lingual Multi-task Models</td></tr><tr><td>en-fr</td><td>77.9</td><td>82.9</td><td>95.5</td><td>89.3</td><td>95.3</td><td>84.0</td><td>0.803 / 0.763</td></tr><tr><td>en-es</td><td>80.1</td><td>85.9</td><td>94.6</td><td>86.5</td><td>96.2</td><td>85.2</td><td>0.809 / 0.770</td></tr><tr><td>en-de</td><td>78.8</td><td>84.0</td><td>95.9</td><td>87.6</td><td>96.1</td><td>85.0</td><td>0.802 / 0.764</td></tr><tr><td colspan="8">Translation-ranking Models</td></tr><tr><td>en-fr</td><td>68.7</td><td>79.3</td><td>87.0</td><td>81.8</td><td>89.4</td><td>74.2</td><td>0.668 / 0.558</td></tr><tr><td>en-es</td><td>67.7</td><td>75.7</td><td>83.5</td><td>86.0</td><td>94.4</td><td>72.6</td><td>0.669 / 0.631</td></tr><tr><td>en-de</td><td>67.8</td><td>75.2</td><td>84.4</td><td>83.6</td><td>86.8</td><td>74.6</td><td>0.673 / 0.632</td></tr><tr><td colspan="8">State-of-the-art Models</td></tr><tr><td>InferSent</td><td>81.1</td><td>86.3</td><td>92.4</td><td>90.2</td><td>88.2</td><td>84.6</td><td>0.801 / 0.758</td></tr><tr><td>Skip-Thought LN</td><td>79.4</td><td>83.1</td><td>93.7</td><td>89.3</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Quick-Thought</td><td>82.4</td><td>86.0</td><td>94.8</td><td>90.2</td><td>92.4</td><td>87.6</td><td>-</td></tr><tr><td>USE Transformer</td><td>81.4</td><td>87.4</td><td>93.9</td><td>87.0</td><td>92.5</td><td>85.4</td><td>0.814 / 0.782</td></tr></table>

of 99. This is because using  $K = 99$  results in all metrics quickly shooting up to  $99\%$ , which leads us to believe that choosing the correct translation out of only 100 samples is too easy.

Table 3 summarizes the P@N metric of the multi-task models and translation-ranking models for  $N = 1,3,10$ . The dual-encoder translation-ranking model remains as a strong baseline for finding the true translation, with P@1 up to  $97.8\%$  for end-retrieval task. The multi-task model performs almost identical to the translation-ranking model in all metrics, which provides some empirical justification that it is possible to maintain embedding space alignment despite optimizing for native tasks in each individual language.

Table 2: Precision at N (P@N) result on a hold-out testing dataset for en-fr, en-es and en-de. Models attempt to predict the true translation target for a source sentence against 999 randomly targets.

<table><tr><td>Language Pair (source-target)</td><td>Model</td><td>P@1</td><td>P@3</td><td>P@10</td></tr><tr><td rowspan="2">en-fr</td><td>Multi-task</td><td>95.4</td><td>96.4</td><td>97.1</td></tr><tr><td>Translation-ranking</td><td>95.1</td><td>96.0</td><td>96.7</td></tr><tr><td rowspan="2">en-es</td><td>Multi-task</td><td>87.5</td><td>91.2</td><td>93.5</td></tr><tr><td>Translation-ranking</td><td>88.8</td><td>91.6</td><td>93.3</td></tr><tr><td rowspan="2">en-de</td><td>Multi-task</td><td>97.5</td><td>98.2</td><td>99.3</td></tr><tr><td>Translation-ranking</td><td>97.8</td><td>98.7</td><td>99.2</td></tr></table>

# 3.5 MULTILINGUAL STS

We further test whether our learned cross-lingual representations can also perform well in their associated non-English language tasks by evaluating semantic textual similarity (STS) performance on French, Spanish, and German.

To evaluate Spanish-Spanish (es-es) STS, we use SemEval-2017 task 1 (STS17) track 3 of Cer et al. (2017), which contains 250 Spanish sentence pairs with human labeled similarity scores. We also evaluate Spanish-English (es-en) STS by using the track 4(a) task $^{2}$ , which contains 250 en-es sentence pairs.

Beyond English and Spanish, however, there are no standard STS datasets available for other languages. As such, we evaluate on a translated version of the STS Benchmark dataset from Cer et al. (2017) for French, Spanish, and German. We use Google's translation system to translate the STS Benchmark sentences to French, Spanish and German. We believe that the results on our pseudo

multilingual STS Benchmark dataset are expected to still be a reasonable indicator of multilingual semantic similarly performance.

Following Cer et al. (2018), we first compute the sentence encodings  $u, v$  of an STS sentence pair, and then score the sentence pair similarity based on the angular distance between the two vectors,  $-\arccos \left( \frac{uv}{||u||||v||} \right)$ . Table 3 shows the Pearson's correlation coefficient of the STS tasks for all models. The first column shows the trained model performance on original English STS Benchmark data. Columns 2 to 4 show the performance on the other languages. All multi-task models remain strong on the translated STS tasks, with around 0.77 for dev and 0.74 for test in all languages. Lastly, columns 5 and 6 show the results of en-es models on STS17 tasks. The un-tuned multi-task models achieve 0.827 for the es-es task and 0.769 for the es-en task. As a point of reference, we also list the two best performing STS systems, Tian et al. (2017) (ECNU) and Wu et al. (2017) (BIT), reported from Cer et al. (2017). Our results are very close to these state-of-the-art feature engineered and mixed systems.

Table 3: Pearson's correlation coefficients on translated STS Benchmark and STS17 tasks. The first column shows the results on the original STS Benchmark data in English.  

<table><tr><td rowspan="2">Model</td><td colspan="4">Translated STS Benchmark (dev / test)</td><td colspan="2">STS17</td></tr><tr><td>en-en</td><td>fr-fr</td><td>es-es</td><td>de-de</td><td>es-es</td><td>es-en</td></tr><tr><td>Multi-task en-fr</td><td>0.803 / 0.763</td><td>0.777 / 0.738</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Trans.-ranking en-fr</td><td>0.668 / 0.558</td><td>0.641 / 0.579</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Multi-task en-es</td><td>0.809 / 0.770</td><td>-</td><td>0.779 / 0.744</td><td>-</td><td>0.827</td><td>0.769</td></tr><tr><td>Trans.-ranking en-es</td><td>0.669 / 0.631</td><td>-</td><td>0.622 / 0.611</td><td>-</td><td>0.642</td><td>0.587</td></tr><tr><td>Multi-task en-de</td><td>0.802 / 0.764</td><td>-</td><td>-</td><td>0.768 / 0.722</td><td>-</td><td>-</td></tr><tr><td>Trans.-ranking en-de</td><td>0.673 / 0.632</td><td>-</td><td>-</td><td>0.630 / 0.526</td><td>-</td><td>-</td></tr><tr><td>ECNU</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.856</td><td>0.813</td></tr><tr><td>BIT</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.846</td><td>0.749</td></tr></table>

# 4 ZERO-SHOT CLASSIFICATION

To evaluate the transfer learning capabilities of our models, we examine how well the multi-task and translation-ranking encoders perform on zero-shot and few-shot classification tasks.

# 4.1 MULTILINGUAL NLI

We first evaluate the zero-shot classification performance of our multi-task models on multilingual natural language inference (NLI) tasks. We make use of the professionally translated French and Spanish subsets of SNLI created by Agić & Schluter (2017) for the cross-lingual zero-shot evaluation. There are 1000 examples in the translated subsets for each language. To evaluate, we simply feed the French or Spanish examples into the pre-trained English NLI sub-network of our cross-lingual models.

Table 4 lists the accuracy on the English SNLI test set (10k sentence pairs) for all models and the accuracy on the French and Spanish translated subsets (1k sentence pairs) for our en-fr and enes models. The original English SNLI accuracies are around  $84\%$  for all of our multi-task models, indicating that English SNLI performance remains stable in the multi-task training setting. The zero-shot accuracy on the translated subsets of SNLI are around  $74\%$  for both of French and Spanish.

Row 4 shows the zero-shot French NLI performance of the multi-task model of Eriguchi et al. (2018), which is a state-of-the-art zero-shot NLI classifiers based on multilingual NMT embeddings. Our en-fr multi-task model shows comparable performance to the NMT-based model in both English and French.

# 4.2 AMAZON REVIEW

Zero-shot Learning. We also conduct a zero-shot evaluation based on the Amazon review data extracted by Prettenhofer & Stein (2010). We preprocess the Amazon reviews and convert the data

Table 4: Zero-shot classification accuracy (%) on SNLI dataset.  

<table><tr><td>Model</td><td>en (10k)</td><td>fr (1k)</td><td>es (1k)</td></tr><tr><td>Multi-task en-fr</td><td>83.7</td><td>74.0</td><td>-</td></tr><tr><td>Multi-task en-es</td><td>83.6</td><td>-</td><td>74.5</td></tr><tr><td>Multi-task en-de</td><td>84.3</td><td>-</td><td>-</td></tr><tr><td>Eriguchi et al. (2018) (NMT en-fr)</td><td>84.4</td><td>73.9</td><td>-</td></tr></table>

into a sentiment classification task by considering reviews with strictly more than three stars as positive and strictly less than three stars as negative, in the same manner as Prettenhofer & Stein (2010). Each review contains a summary field and a text field, which we concatenate to produce a single input. As the multi-task models are trained with sentence lengths clipped to 64, we only take the first 64 tokens from the the concatenated text as the input. There are 6000 training reviews in English, which we split into  $90\%$  for training and  $10\%$  for development.

We first encode inputs using the pre-trained multi-task and translation-ranking encoders and feed the encoded vectors into a 2-layer feed-forward network culminating in a softmax layer. We use layers of size 512 and tanh activation functions in each layer. We use Adam for optimization with an initial learning rate of 0.0005 and a learning rate decay of 0.9 at every epoch during training. We use a batch size of 16 and train for 20 total epochs in all experiments. We freeze the cross-lingual encoder during training. The model architecture and parameters are tuned on the development set.

We first train the classifier on English data, and then evaluate it on the 6000 French and German Amazon review test examples. The results are summarized in Table 5. The accuracy on the English test set is  $87.4\%$  for the en-fr model and  $87.1\%$  for the en-de model, with the zero-shot accuracy being above  $80\%$  for both models. The translation-ranking models again perform worse on all metrics. Once again we compare the proposed model with Eriguchi et al. (2018), and find that our zero-shot performance has a reasonable gain on the fr test set<sup>3</sup>.

Table 5: Zero-shot sentiment classification accuracy(%) on target language Amazon review test data after training on only English Amazon review data.  

<table><tr><td>Model</td><td>en</td><td>fr</td><td>de</td></tr><tr><td>Multi-task en-fr</td><td>87.4</td><td>82.3</td><td>-</td></tr><tr><td>Translation-ranking en-fr</td><td>74.4</td><td>66.3</td><td>-</td></tr><tr><td>Multi-task en-de</td><td>87.1</td><td>-</td><td>81.0</td></tr><tr><td>Translation-ranking en-de</td><td>73.8</td><td>-</td><td>67.0</td></tr><tr><td>Eriguchi et al. (2018) (NMT en-fr)</td><td>83.2</td><td>81.3</td><td>-</td></tr></table>

Few-shot Learning. We further evaluate the proposed multi-task models via few-shot learning, by training on English reviews and only a portion of French and German reviews. Our few-shot models are compared with baselines of training on French and German reviews only. Table 6 shows the classification accuracy of the few-shot models, where the second row shows the percent of French and German data that is used when training each model. With as little as  $20\%$  of the French or German training data, the few-shot models perform nearly as good as the baseline models trained on  $100\%$  of the French and German data. Adding more French and German training data leads to further improvements in few-shot model performance, with the few-shot models reaching  $85.8\%$  accuracy in French and  $84.5\%$  accuracy in German when using all of the French and German data.

# 5 ANALYSIS OF CROSS-LINGUAL EMBEDDING SPACES

Motivated by the recent work of Søgaard et al. (2018) studying the graph structure of multilingual word representations, we perform a similar analysis for our learned cross-lingual sentence representations. To do so, we take  $N$  samples of size  $K$  from en-fr, en-es, and en-de translation data and then encode these samples using the corresponding multi-task and translation-ranking models. We then

Table 6: Sentiment classification accuracy(%) on target language Amazon review test data after training on English Amazon review data and a portion of French of German data. The second row shows the percent of French (fr) or German (de) data is used for training in each model.  

<table><tr><td rowspan="2">Model</td><td colspan="6">fr</td><td colspan="6">de</td></tr><tr><td>0%</td><td>10%</td><td>20%</td><td>40%</td><td>80%</td><td>100%</td><td>0%</td><td>10%</td><td>20%</td><td>40%</td><td>80%</td><td>100%</td></tr><tr><td>Few-shot</td><td>82.3</td><td>84.4</td><td>84.4</td><td>84.8</td><td>85.2</td><td>85.8</td><td>81.0</td><td>81.6</td><td>83.3</td><td>84.0</td><td>84.7</td><td>84.5</td></tr><tr><td>Baseline</td><td>-</td><td>79.2</td><td>80.0</td><td>82.7</td><td>84.3</td><td>84.9</td><td>-</td><td>75.5</td><td>77.7</td><td>81.6</td><td>83.5</td><td>84.4</td></tr></table>

compute pairwise distance matrices within each sampled set of encodings, and use these distance matrices to construct graph Laplacians<sup>4</sup>. Finally, we obtain the similarity  $\Psi(S,T)$  between each model's source and target language embedding subsets by comparing the eigenvalues of the source language graph Laplacians to the eigenvalues of the target language graph Laplacians as follows:

$$
\Psi (S, T) = \frac {1}{N} \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {K} \left(\lambda_ {j} \left(L _ {i} ^ {(s)}\right) - \lambda_ {j} \left(L _ {i} ^ {(t)}\right)\right) ^ {2} \tag {3}
$$

Where  $L_{i}^{(s)}$  and  $L_{i}^{(t)}$  refer to the graph Laplacians of the source language and target language sentences obtained from the  $i^{th}$  sample of source-target translation pairs. A smaller value of  $\Psi(S,T)$  indicates higher eigen-similarity of the source language and target language embedding subsets. Following Søgaard et al. (2018) we use a sample size of  $K = 10$  translation pairs, but we choose to use  $N = 1000$  samples instead of  $N = 10$  (as was done in their work) since we found  $\Psi(S,T)$  to have very high variance at  $N = 10$ . The computed values of  $\Psi(S,T)$  for our multi-task and translation-ranking models are summarized in Table 7.

Table 7: Average eigen-similarity values of source and target embedding subsets for multi-task and translation-ranking models.  

<table><tr><td>Model</td><td>en-fr</td><td>en-es</td><td>en-de</td></tr><tr><td>multi-task</td><td>0.592</td><td>0.526</td><td>0.761</td></tr><tr><td>translation-ranking</td><td>1.036</td><td>0.572</td><td>2.187</td></tr></table>

We find that the source and target embedding subsets constructed from the multi-task models exhibit greater average eigen-similarity than those resulting from the translation-ranking models for all source-target language pairs. This result is not necessarily intuitive, since one might expect the translation-ranking model to optimize more for alignment. Given that eigen-similarity correlates with the better performance of the multi-task models in almost all tasks, a potential direction for future work could be to introduce regularization penalties based on graph similarity in multitask training. Interestingly, we also observe that the eigen-similarity gaps between the multi-task and translation-ranking models are not uniform across language pairs (although it may be that translation-ranking requires even more training). Thus, another direction could be to further study differences in the difficulty of aligning different source-target language embeddings.

# 6 CONCLUSION

In this work, we explored a straightforward framework for training cross-lingual, multi-task dual-encoder models. We showed that by training English-French, English-Spanish, and English-German multi-task models using our setup, we can achieve near-state-of-the-art or state-of-the-art performance in a variety of English tasks while also being able to produce similar caliber results in zero-shot transfer learning tasks for other languages. Finally, we note that the fact that multi-task training can actually improve performance on some downstream English tasks (TREC) is particularly interesting, and believe that there are many possibilities for future explorations of cross-lingual model training.

# REFERENCES

Zeljko Agić and Natalie Schluter. Baselines and test data for cross-lingual inference. arXiv preprint arXiv:1704.05347, 2017.  
Rami Al-Rfou, Marc Pickett, Javier Snader, Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil. Conversational contextual cues: The case of personalization and history for response ranking. CoRR, abs/1606.00372, 2016. URL http://arxiv.org/abs/1606.00372.  
Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 632-642. Association for Computational Linguistics, 2015. doi: 10.18653/v1/D15-1075. URL http://www.aclweb.org/anthology/D15-1075.  
Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. Semeval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pp. 1-14, Vancouver, Canada, August 2017. Association for Computational Linguistics. URL http://www.aclweb.org/anthology/S17-2001.  
Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil. Universal sentence encoder. CoRR, abs/1803.11175, 2018. URL http://arxiv.org/abs/1803.11175.  
Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, and Antoine Bordes. Supervised learning of universal sentence representations from natural language inference data. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 670-680. Association for Computational Linguistics, 2017. URL http://aclweb.org/anthology/D17-1070.  
Akiko Eriguchi, Melvin Johnson, Orhan First, Hideto Kazawa, and Wolfgang Macherey. Zero-shot cross-lingual classification using multilingual neural machine translation. arXiv preprint arXiv:1809.04686, 2018.  
Dan Gillick. Sentence boundary detection and the problem with the u.s. In Proceedings of Human Language Technologies: The 2009 Annual Conference of the North American Chapter of the Association for Computational Linguistics, Companion Volume: Short Papers, NAACL-Short '09, pp. 241-244, Stroudsburg, PA, USA, 2009. Association for Computational Linguistics. URL http://dl.acm.org/citation.cfm?id=1620853.1620920.  
Mandy Guo, Qinlan Shen, Yinfei Yang, Heming Ge, Daniel Cer, Gustavo Hernandez Abrego, Keith Stevens, Noah Constant, Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil. Effective parallel corpus mining using bilingual sentence embeddings. CoRR, abs/1807.11906, 2018.  
Matthew Henderson, Rami Al-Rfou, Brian Strope, Yun-Hsuan Sung, László Lukács, Ruiqi Guo, Sanjiv Kumar, Balint Miklos, and Ray Kurzweil. Efficient natural language response suggestion for smart reply. CoRR, abs/1705.00652, 2017. URL http://arxiv.org/abs/1705.00652.  
Ryan Kiros, Yukun Zhu, Ruslan R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Skip-thought vectors. In Advances in neural information processing systems, pp. 3294-3302, 2015.  
Lajanugen Logeswaran and Honglak Lee. An efficient framework for learning sentence representations. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJvJXZb0W.  
Peter Prettenhofer and Benno Stein. Cross-Language Text Classification using Structural Correspondence Learning. In 48th Annual Meeting of the Association of Computational Linguistics (ACL 10), pp. 1118-1127. Association for Computational Linguistics, July 2010. URL http://www.aclweb.org/anthology/P10-1114.

Holger Schwenk, Ke Tran, Orhan Firat, and Matthijs Douze. Learning joint multilingual sentence representations with neural machine translation. CoRR, abs/1704.04154, 2017. URL http://arxiv.org/abs/1704.04154.  
Karan Singla, Dogan Can, and Shrikanth Narayanan. A multi-task approach to learning multilingual representations. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 214-220. Association for Computational Linguistics, 2018. URL http://aclweb.org/anthology/P18-2035.  
Anders Søgaard, Sebastian Ruder, and Ivan Vulic. On the limitations of unsupervised bilingual dictionary induction. CoRR, abs/1805.03620, 2018. URL http://arxiv.org/abs/1805.03620.  
Sandeep Subramanian, Adam Trischler, Yoshua Bengio, and Christopher J Pal. Learning general purpose distributed sentence representations via large scale multi-task learning. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=B18WgG-CZ.  
Junfeng Tian, Zhiheng Zhou, Man Lan, and Yuanbin Wu. Ecnu at semeval-2017 task 1: Leverage kernel-based traditional nlp features and neural networks to build a universal model for multilingual and cross-lingual semantic textual similarity. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pp. 191-197, 2017.  
Jakob Uszkoreit, Jay M. Ponte, Ashok C. Popat, and Moshe Dubiner. Large scale parallel document mining for machine translation. In Proceedings of the 23rd International Conference on Computational Linguistics, COLING '10, pp. 1101-1109, Stroudsburg, PA, USA, 2010. Association for Computational Linguistics. URL http://dl.acm.org/citation.cfm?id=1873781.1873905.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pp. 5998-6008, 2017.  
Hao Wu, Heyan Huang, Ping Jian, Yuhang Guo, and Chao Su. Bit at semeval-2017 task 1: Using semantic information space to evaluate semantic textual similarity. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pp. 77-84, 2017.  
Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, ukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyls, Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google's neural machine translation system: Bridging the gap between human and machine translation. CoRR, abs/1609.08144, 2016. URL http://arxiv.org/abs/1609.08144.  
Yinfei Yang, Steve Yuan, Daniel Cer, Sheng-Yi Kong, Noah Constant, Petr Pilar, Heming Ge, Yunhsuan Sung, Brian Strope, and Ray Kurzweil. Learning semantic textual similarity from conversations. In Proceedings of The Third Workshop on Representation Learning for NLP, pp. 164-174. Association for Computational Linguistics, 2018. URL http://aclweb.org/anthology/W18-3022.  
X.-D. Zhang. The Laplacian eigenvalues of graphs: a survey. ArXiv e-prints, November 2011.
