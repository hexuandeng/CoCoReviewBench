# FINDING A JACK-OF-ALL-TRADES:  
AN EXAMINATION OF SEMI-SUPERVISED LEARNING IN READING COMPREHENSION

Rudolf Kadlec*, Ondrej Bajgar*, Peter Hrincar & Jan Kleindienst

IBM Watson

V Parku 4, 140 00 Prague, Czech Republic

{rudolf_kadlec,obajgar,phrincar,jankle}@cz.ibm.com

# ABSTRACT

Deep learning has proven useful on many NLP tasks including reading comprehension. However it requires a lot of training data which are not available in some domains of application. Hence we examine the possibility of using data-rich domains to pre-train models and then apply them in domains where training data are harder to get. Specifically, we train a neural-network-based model on two context-question-answer datasets, the BookTest and CNN/DM, and we monitor transfer to subsets of bAbI, a set of artificial tasks designed to test specific reasoning abilities, and of SQuAD, a question-answering dataset which is much closer to real-world applications. Our experiments show very limited transfer if the model isn't shown any training examples from the target domain however the results are promising if the model is shown at least a few target-domain examples. Furthermore we show that the effect of pre-training is not limited to word embeddings.

# 1 INTRODUCTION

Machine intelligence has had some notable successes, however often in narrow domains which are sometimes of little practical use to humans – for instance games like chess (Campbell et al., 2002) or Go (Silver et al., 2016). If we aimed to build a general AI that would be able to efficiently assist humans in a wide range of settings, we would want it to have a much larger set of skills – among them would be an ability to understand human language, to perform common-sense reasoning and to be able to generalize its abilities to new situations like humans do.

If we want to achieve this goal through Machine Learning, we need data to learn from. A lot of data if the task at hand is complex - which is the case for many useful tasks. One way to achieve wide applicability would be to provide training data for each specific task we would like the machine to perform. However it is unrealistic to obtain a sufficient amount of training data for some domains - it may for instance require expensive human annotation or all domains of application may be difficult to predict in advance - while the amount of training data in other domains is practically unlimited, (e.g. in language modelling or Cloze-style question answering).

The way to bridge this gap – and to achieve the aforementioned adaptability – is transfer learning (Pan & Yang, 2010) and closely related semi-supervised learning (Zhu & Goldberg, 2009) which would allow the system to acquire a set of skills on domains where data are abundant and then use these skills to succeed on previously unseen domains. Despite how important generalization is for general AI, a lot of research keeps focusing on solving narrow tasks.

In this paper we would like to examine transfer of learnt skills and knowledge within the domain of text comprehension - a field that has lately attracted a lot of attention within

the NLP community (Hermann et al., 2015; Hill et al., 2015; Kobayashi et al., 2016; Kadlec et al., 2016b; Chen et al., 2016; Sordoni et al., 2016; Dhingra et al., 2016; Trischler et al., 2016; Weissenborn, 2016; Cui et al., 2016b;a; Li et al., 2016; Shen et al., 2016). Specifically, we would like to address the following research questions:

1. Whether we could train models on natural-language tasks where data are abundant and transfer the learnt skills to tasks where in-domain training data may be difficult to obtain. We will first look into what reasoning abilities a model learns from two large-scale reading-comprehension datasets and then check whether it can transfer its skills to real world tasks. Spoiler: both these transfers are very poor if we allow no training at all on the target task.  
2. Whether pre-training on large-scale datasets does help if we allow the model to train on a small sample of examples from the target tasks - here the results are much more positive.  
3. Finally we examine whether the benefits of pre-training are concentrated in any particular part of the model - namely the word-embedding part or the context encoder (the reasoning part). It turns out that pre-training is useful for both components.

Our results do not improve current state of the art in any of the studied tasks, however they show a clear positive effect of large-dataset pre-training on the performance of our baseline machine-learning model. Previous studies of transfer learning (Mou et al., 2016) and semi-supervised learning (Dai & Le, 2015) in NLP focused mainly on text classification. To our knowledge this work is the first study of transfer learning in reading comprehension and we hope it will stimulate further work in this important area.

We will first briefly introduce the datasets we will be using on the pre-training and target sides, then our baseline model and afterwards in turn describe the method and results of each of the three experiments.

# 2 DATASETS

# 2.1 PRE-TRAINING DATASETS

We have mentioned that for the model pre-training, we would want to use a task where training data are abundant. An example of such task is context-dependent cloze-style-question answering since the training data for this task can be generated automatically from a suitable corpus. We will use two such pre-training datasets in our experiments: the BookTest (Bajgar et al., 2016) and the CNN/Daily Mail (CNN/DM) news dataset (Hermann et al., 2015).

The task associated with both datasets is to answer a cloze-style question (i.e. fill in a blank in a sentence) the answer to which needs to be inferred from a context document provided with the question.

# 2.1.1 BOOKTEST

In the BookTest dataset, the context document is formed from 20 consecutive sentences from a book. The question is then formed by omitting a common noun or a named entity from the subsequent 21st sentence. Among datasets of this kind, the BookTest is among the largest with more than 14 million training examples.

# 2.1.2 CNN/DAILY MAIL

In the CNN/DM dataset the context document is formed from a news article while the cloze-style question is formed by removing a named entity from one of the short summary sentences which often appear at the top of the article.

To stop the model from using world knowledge from outside the context article (and hence truly test the comprehension of the article), all named entities have been replaced

by anonymous tags, which are further shuffled for each example. This may make the comprehension more difficult, however since the answer is always one of the anonymized entities it also reduces the number of possible answers making guessing easier.

# 2.2 TARGET DATASETS

# 2.2.1 BABI

The first target dataset are the bAbI tasks (Weston et al., 2016) - a set of artificial tasks each of which is designed to test a specific kind of reasoning. This toy dataset will allow us to observe what particular skills the model may be learning from each of the three training datasets.

For our experiments we will be using an architecture designed to select one word from the context document as the answer. Hence we have selected Tasks 1,2,3,4,5,11,12,13,14 and 16 which fulfill this requirement and added task 15 which required a slight modification. Furthermore because both pre-training datasets are cloze-style we converted also the bAbI task questions into cloze style (e.g. "Where is John?" to "John is in the XXXXX.")

Furthermore for the models pre-trained on CNN/DM we anonymized the tasks in a way similar to the pre-training dataset - i.e. we replaced all names of characters and also all words that can appear as answers for the given tasks by anonymous tags in the style of CNN/DM. This allows even models that have not seen any training examples from the target domain to have a go at the questions.

Full details about these alterations can be found in the Appendix A.

# 2.2.2 SQUAD

Secondly, we will look on transfer to the SQuAD dataset (Rajpurkar et al., 2016), where the associated task may be already useful in the real world. Although cloze-style questions have the huge advantage in the possibility of being automatically generated from a suitable corpus – the path taken by CNN/DM and the BookTest – humans want to ask a question in (truly) natural language, they would use proper question, not its cloze-style substitute. This brings us to the importance of transfer from the data-rich cloze-style training to the domain of proper questions where data are much scarcer due to the need of human annotation.

The SQuAD dataset is a great dataset to use for this. As opposed to the bAbI tasks, the goal of this dataset is actually a problem whose solving would be useful to humans - answering natural questions based on an natural language encyclopedic knowledge base.

For our experiments we selected only a subset of the SQuAD train and development examples where the answer is only a single word. We do this because it is an inherent assumption of our machine learning model. This way we extracted 28,346 training examples out of the original 100,000 examples and 3,233 development examples out of 10,570.

# 3 MACHINE LEARNING MODEL: AS READER

As a machine learning model for our experiments we use the Attention Sum Reader (AS Reader) (Kadlec et al., 2016b). The AS Reader is simple to implement while it achieves strong performance on several text comprehension tasks (Kadlec et al., 2016b; Bajgar et al., 2016; Chu et al., 2016). Since the AS Reader is a building block of many recent text-comprehension models (Trischler et al., 2016; Sordoni et al., 2016; Dhingra et al., 2016; Cui et al., 2016b;a; Shen et al., 2016; Munkhdalai & Yu, 2016) it is representative of current research in this field.

A high level structure of the AS Reader is shown in Figure 1. The words from the document and the question are first converted into vector embeddings using a look-up matrix. The document is then read by a bidirectional GRU network (Cho et al., 2014). A concatenation of the hidden states of the forward and backward GRUs at each word is then used as a contextual embedding of this word, intuitively representing the context in which the word is

appearing. We can also understand it as representing the set of questions to which this word may be an answer.

Similarly the question is read by a bidirectional GRU but in this case only the final hidden states are concatenated to form the question embedding.

The attention over each word in the context is then calculated as the dot product of its contextual embedding with the question embedding. This attention is then normalized by the softmax function and summed across all occurrences of each answer candidate. The candidate with most accumulated attention is selected as the final answer.

For a more detailed description of the model including equations check Kadlec et al. (2016b).

![](images/e0249e6c031c242c2e7385a9319ebf50a5a2a5daf16ab7554cd8e52c3c2232d5.jpg)  
Figure 1: Structure of the AS Reader model.

# 4 EXPERIMENTS: TRANSFER LEARNING IN TEXT COMPREHENSION

Now let us turn in more detail to the three kinds of experiments that we performed.

# 4.1 PRE-TRAINED WITHOUT TARGET ADJUSTMENT

In the first experiment we tested how a model trained on one of the large-scale pre-training datasets performs on the bAbI tasks without any opportunity to train on bAbI. Since the BookTest and CNN/DM tasks involve only cloze-style questions, we can't expect a model trained on them to answer natural ?-style questions. Hence we did not study the transfer to SQuAD in this case, only the transfer to the (cloze-converted) bAbI tasks.

# 4.1.1 METHOD

First we tested how our AS Reader architecture (Kadlec et al., 2016b) can handle the tasks if trained directly on the bAbI training data for each task. Then we tested the degree of transfer from the BookTest and CNN/DM data to the 11 selected bAbI tasks.

In the first part of the experiment we trained a separate instance of the AS Reader on the 10,000-example version of the bAbI training data for each of the 11 tasks (for more details see Appendix B). On 8 of them the architecture was able to learn the task with accuracy at least  $95\%$  (results for each task can be found in Table 4 in Appendix C<sup>1</sup>. Hence if given

Table 1: The mean performance evaluated on 11 bAbI tasks. The first two columns show two random baselines, the following three columns show performance of the AS Reader trained on different datasets, the last column shows the results of DMN+ (Xiong et al., 2016), the state of the art model on bAbI 10k dataset. For more detailed results listing per task accuracies see Appendix C.  

<table><tr><td>Model</td><td>Rnd.</td><td>Rnd. cand.</td><td colspan="3">AS Reader</td><td>DMN+</td></tr><tr><td>Train dataset</td><td>not trained</td><td>bAbI 10k</td><td>BookTest 14M</td><td>CNN/DM 1.2M</td><td>bAbI 10k</td><td>bAbI 10k</td></tr><tr><td>bAbI mean (11 tasks)</td><td>6.1</td><td>29.9</td><td>34.8</td><td>38.1</td><td>92.7</td><td>95.7</td></tr></table>

appropriate training the AS Reader is capable of the reasoning needed to solve most of the selected bAbI tasks. Now when we know that the AS Reader is powerful enough to learn the target tasks we can turn to transfer from the two large-scale datasets.

The main part of this first experiment was then straightforward: we pre-trained multiple models on the BookTest and CNN/DM datasets and then simply evaluated them on the test datasets of the 11 selected bAbI tasks.

# 4.1.2 RESULTS

Table 1 summarizes the results of this experiment. Both the models trained on the BookTest and those trained on the CNN/DM dataset achieve much lower accuracy than the models trained directly on each individual bAbI task. However there is some transfer between the tasks since the AS Reader trained on either BT or CNN/DM outperforms a random baseline<sup>2</sup> and even an improved baseline which selects the most frequent word from the context which also appears as an answer in the training data for this task.

The results also show that the models trained on CNN/DM perform somewhat better on most tasks than the BookTest models. This may be due to the fact that bAbI tasks generally require the model to summarize information from the context document. This is also what the CNN/DM dataset is testing. On the other hand the BookTest requires prediction of a possible continuation of a story, where the required kind of reasoning is much less clear but certainly different from pure summarization. Another explanation for better performance of CNN/DM models might be that they solve slightly simpler task since the candidate answers were already pre-selected in the entity anonymization step.

Readers interested in how the training-dataset size affects this kind of transfer can check (Kadlec et al., 2016a) where we show that the target-task performance is a bit better if we use the large BookTest as opposed to its smaller subset, the Children's Book Test (Hill et al., 2015).

Conclusions from this experiment are that the skills learned from two large-scale datasets generalize surprisingly poorly to even simple toy tasks. This may make us ask whether most teams' focus on solving narrow tasks is truly beneficial if the skills learnt on these tasks generalize this poorly. However it also brings us to our next experiment, where we try to provide some help to the struggling pre-trained models.

# 4.2 PRE-TRAINED WITH TARGET ADJUSTMENT

After showing that the skills learnt from the BookTest and CNN/DM datasets are by themselves insufficient for solving the toy tasks, the next natural question is whether they are useful if helped by training on a small sample of examples from the target task. We call this additional phase of training target adjustment. For this experiment we again use the bAbI tasks, however we also test transfer to a subset of SQuAD dataset that is much closer to real-world natural-language question answering.

![](images/8996008a493f5bc63c3ab8393c856c49b7fa548bb550946ecb33e460c5d81639.jpg)  
(a)

![](images/cf896106ddeccc06e09e27f1cb8c6581fa6a14192c7f1356d6f254b51a68a669.jpg)  
(b)  
Figure 2: Sub-figure (a) shows the average across the 11 bAbI tasks of the best-validation model's test accuracy. (b) shows the test accuracy on SQuAD of each model we trained (the points) and the lines join the accuracies of the best-validation models for each training size.

The results presented in this and the following section are based on training 3701 model instances.

# 4.2.1 METHOD

Common to bAbI and SQuAD datasets. In this experiment we started with a pretrained model which we used in the previous experiment. However after it finished training on one of the large pre-training datasets, we allowed it to train on subsets of training examples from the target dataset of various sizes. We tried training four different pre-trained models and also, for comparison, four randomly-initialized models with the same hyperparameters (see Appendix C.1 for details). The experiment with each task-model couple was run on 4 different data samples of each size which were randomly drawn from the training dataset of the task to account for variations between these random samples - which may be substantial given the small sample size. $^3$

bAbI. For each of these models we observed the test accuracy at the best-validation epoch and compared this number between the randomly initialized and pre-trained models. Validation was done using 100 examples which were set aside from the task's original 10k training data. We perform the experiment with models pre-trained on BT and also on CNN/DM.

SQuAD subset. In the SQuAD experiment we trained the model on a subset of the original training dataset where answers were only single words and we report the best-validation accuracy on a development set filtered in the same way. This experiment was performed only with the models pre-trained on BookTest.

# 4.2.2 RESULTS

The results of these experiments are summarized in Figures 2 and 3.

bAbI. Sub-figure 2a shows mean test accuracy of the models that achieved the best validation result for each single task. The results for both BookTest and CNN/DM experiments confirm positive effect of pre-training compared to randomly initialized baseline. Figure 3 shows performance on selected bAbI tasks where pre-training has clearly positive effect, such plot for each of the target tasks is provided in Appendix C.3.1.

![](images/3b0c616d4e18fbf71cce10fca7839888f2cbb475cc8d714df607df47188dbaf6.jpg)  
Figure 3: Example of 3 bAbI tasks where pre-training seems to help. Note that the task may be easier for the CNN/DM models due to answer anonymization which restricts the choice of possible answers.

![](images/d929e55ac3115e7712293e5d807b0f1b56b6695cb8992f7d9c8f4ff6bf44f14f.jpg)

![](images/ec31371b25352ab20b63d04a04e319e9b8fdb8bb5137bc6b2a8880bb786ec4df.jpg)

Note that CNN/DM can not be directly compared to BookTest results due to entity anonymization that seems to simplify the task when the model is trained on smaller datasets.

Since our evaluation methodology with different training set sizes is novel we can compare our result only to MemN2N (Sukhbaatar et al., 2015) on 1k dataset. MemN2N is the only weakly supervised model that reports accuracy when trained on less than 10k examples. MemN2N achieves average accuracy  $93.2\%$  on the eleven selected tasks. This is substantially better than both our random baseline  $(78.0\%)$  and BookTest pre-trained model  $(79.5\%)$ , however our model is not tuned towards this task. One important conceptual difference is that AS Reader processes the whole context as one sequence of words, whereas MemN2N receives context split into single sentences, which simplifies the task.

SQuAD subset. The results of SQuAD experiment also confirm positive effect of pretraining, see Sub-figure 2b, for now compare just lines showing performance of the fully pre-trained model and the randomly initialized model - the meaning of the remaining two lines shall become clear in the next section.

We should note that performance of our model is not competitive with the state of the art models on this dataset. For instance the DCR model (Yu et al., 2016) trained on our SQuAD subset achieves validation accuracy  $74.9\%$  in this task which is better than our randomly initialized  $(35.4\%)$  and pre-trained  $(51.6\%)$  models<sup>6</sup>. However, the DCR model is designed specifically for the SQuAD task, for instance it utilizes features that are not used by our model.

# 4.3 PARTIALLY PRE-TRAINED MODEL

Since our previous experiment confirmed positive effect of pre-training if followed by target-domain adjustment we wondered which part of the model contains the knowledge transferable to new domains. To examine this we performed the following experiment.

# 4.3.1 METHOD

Our machine learning model - the AS Reader - consists of two main parts: the word-embedding look-up and the bidirectional GRUs used to encode the document and question (see Figure 1). Therefore a natural question was what the contribution of each of these parts is.

To test this we created two models out of each pre-trained model used in the previous experiment. The first model variant uses the pre-trained word embeddings from the original

Table 2: Performance of AS Reader with different sets of pre-trained weights on selected bAbI tasks when trained on just 100 examples. The first row shows performance of randomly initialized baseline model. The following three rows show increase in accuracy (measured in percent absolute) when the model is initialized with pre-trained weights.  

<table><tr><td>bAbI Task
Model variant</td><td>1.</td><td>5.</td><td>11.</td><td>14.</td></tr><tr><td>Random init</td><td>53%</td><td>66%</td><td>71%</td><td>33%</td></tr><tr><td>Δ Pre-trained encoders</td><td>+6</td><td>+25</td><td>+4</td><td>+2</td></tr><tr><td>Δ Pre-trained embeddings</td><td>+17</td><td>+6</td><td>+8</td><td>+8</td></tr><tr><td>Δ Pre-trained full</td><td>+34</td><td>+22</td><td>+14</td><td>+13</td></tr></table>

model while the GRU encoders are randomly initialized. We say that this model has pretrained embeddings. The second model variant uses the opposite setting where the word embeddings are randomly initialized while the encoders are taken form a pre-trained model. We call this pre-trained encoders.

bAbI. For this experiment we selected only a subset of tasks with training set of 100 examples where there was significant difference in accuracy between randomly-initialized and pre-trained models. For evaluation we use the same methodology as in the previous experiment, that is, we report accuracy of the best-validation model averaged over 4 training splits.

SQuAD subset. We evaluated both model variants on all training sets from the previous SQuAD experiment using the same methodology.

# 4.3.2 RESULTS

bAbI. Table 2 shows improvement of pre-trained models over a randomly initialized baseline. In most cases (all except Task 5) the fully pre-trained model achieved the best accuracy.

SQuAD subset. The accuracies of the four model variants are plotted in Figure 2b together with results of the previous SQuAD experiment. The graph shows that both pre-trained embeddings and pre-trained encoders alone improve performance over the randomly initialized baseline, however the fully pre-trained model is always the best.

The overall result of this experiment is that both the pre-training of the word embeddings and pre-training of the encoder parameters are important since the fully pre-trained model outperforms both partially pre-trained variants.

# 5 CONCLUSION

Our experiments show that transfer from two large cloze-style question-answering datasets to our two target tasks is surprisingly poor, if the models aren't provided with any examples from the target domain. However we show that models that pre-trained models perform significantly better than a randomly initialized model if they are shown at least a few training examples from the target domain. The usefulness of pre-trained word embeddings is well known in the NLP community however we show that the power of our pre-trained model does not lie just in the embeddings. This suggests that once the text-comprehension community agrees on sufficiently versatile model, much larger parts of the model could start being reused than just the word-embeddings.

The generalization of skills from a training domain to new tasks is an important ingredient of any system we would want to call intelligent. This work is an early step to explore this direction.

# REFERENCES

Ondrej Bajgar, Rudolf Kadlec, and Jan Kleindienst. Embracing data abundance: BookTest Dataset for Reading Comprehension. arXiv preprint arXiv:1610.00956, 2016.  
Murray Campbell, A Joseph Hoane, and Feng-hsiung Hsu. Deep blue. Artificial intelligence, 134(1):57-83, 2002.  
Danqi Chen, Jason Bolton, and Christopher D. Manning. A Thorough Examination of the CNN / Daily Mail Reading Comprehension Task. In Association for Computational Linguistics (ACL), 2016.  
Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. Empirical Methods in Natural Language Processing (EMNLP), 2014. URL http://arxiv.org/abs/1406.1078v3.  
Zewei Chu, Hai Wang, Kevin Gimpel, and David Mcallester. Broad Context Language Modeling as Reading Comprehension. 2016.  
Yiming Cui, Zhipeng Chen, Si Wei, Shijin Wang, Ting Liu, and Guoping Hu. Attention-over-Attention Neural Networks for Reading Comprehension. 2016a. URL http://arxiv.org/abs/1607.04423.  
Yiming Cui, Ting Liu, Zhipeng Chen, Shijin Wang, and Guoping Hu. Consensus Attention-based Neural Networks for Chinese Reading Comprehension. 2016b.  
Andrew M. Dai and Quoc V. Le. Semi-supervised Sequence Learning. NIPS, 2015. ISSN 10495258. URL http://arxiv.org/abs/1511.01432.  
Bhuwan Dhingra, Hanxiao Liu, William W. Cohen, and Ruslan Salakhutdinov. Gated-Attention Readers for Text Comprehension. 2016. URL http://arxiv.org/abs/1606.01549.  
Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, Adrià Puigdomènech Badia, Karl Moritz Hermann, Yori Zwols, Georg Ostrovski, Adam Cain, Helen King, Christopher Summerfield, Phil Blunsom, Koray Kavukcuoglu, and Demis Hassabis. Hybrid computing using a neural network with dynamic external memory. Nature, 2016. ISSN 0028-0836. doi: 10.1038/nature20101.  
Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems, pp. 1684-1692, 2015.  
Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston. The goldilocks principle: Reading children's books with explicit memory representations. arXiv preprint arXiv:1511.02301, 2015.  
Rudolf Kadlec, Ondrej Bajgar, and Jan Kleindienst. From Particular to General: A Preliminary Case Study of Transfer Learning in Reading Comprehension. _MAIN Workshop at NIPS_, 2016a.  
Rudolf Kadlec, Martin Schmid, Ondrej Bajgar, and Jan Kleindienst. Neural Text Understanding with Attention Sum Reader. Proceedings of ACL, 2016b.  
Sosuke Kobayashi, Ran Tian, Naoaki Okazaki, and Kentaro Inui. Dynamic Entity Representation with Max-pooling Improves Machine Reading. Proceedings of the North American Chapter of the Association for Computational Linguistics and Human Language Technologies (NAACL-HLT), 2016.  
Peng Li, Wei Li, Zhengyan He, Xuguang Wang, Ying Cao, Jie Zhou, and Wei Xu. Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering. 2016.

Lili Mou, Zhao Meng, Rui Yan, Ge Li, Yan Xu, Lu Zhang, and Zhi Jin. How Transferable are Neural Networks in NLP Applications? EMNLP, 2016.  
Tsendsuren Munkhdalai and Hong Yu. Reasoning with Memory Augmented Neural Networks for Language Comprehension. 2016.  
Sinno Jialin Pan and Qiang Yang. A Survey on Transfer Learning. IEEE Transactions on Knowledge and Data Engineering, 22(10):1345-1359, oct 2010. ISSN 1041-4347. doi: 10.1109/TKDE.2009.191. URL http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=5288526.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ Questions for Machine Comprehension of Text. (ii), 2016. URL http://arxiv.org/abs/1606.05250.  
Yelong Shen, Po-Sen Huang, Jianfeng Gao, and Weizhu Chen. ReasoNet: Learning to Stop Reading in Machine Comprehension. 2016. URL http://arxiv.org/abs/1609.05284.  
David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587): 484-489, 2016. ISSN 0028-0836. doi: 10.1038/nature16961. URL http://dx.doi.org/10. 1038/nature16961.  
Alessandro Sordoni, Phillip Bachman, and Yoshua Bengio. Iterative Alternating Neural Attention for Machine Reading. 2016.  
Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-To-End Memory Networks. pp. 1-11, 2015. URL http://arxiv.org/abs/1503.08895.  
Adam Trischler, Zheng Ye, Xingdi Yuan, and Kaheer Suleman. Natural Language Comprehension with the EpiReader. 2016. URL http://arxiv.org/abs/1606.02270.  
Dirk Weissenborn. Separating Answers from Queries for Neural Reading Comprehension. 2016. URL http://arxiv.org/abs/1607.03316.  
Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M Rush, Bart Van Merri, Armand Joulin, and Tomas Mikolov. Towards ai-complete question answering: A set of prerequisite toy tasks. 2016.  
Caiming Xiong, Stephen Merity, and Richard Socher. Dynamic Memory Networks for Visual and Textual Question Answering. ICML, 2016. URL http://arxiv.org/abs/1603.01417.  
Yang Yu, Wei Zhang, Kazi Hasan, Mo Yu, Bing Xiang, and Bowen Zhou. End-to-End Reading Comprehension with Dynamic Answer Chunk Ranking. (1), 2016. URL http://arxiv.org/abs/1610.09996.  
Xiaojin Zhu and Andrew B Goldberg. Introduction to semi-supervised learning. Synthesis lectures on artificial intelligence and machine learning, 3(1):1-130, 2009.
