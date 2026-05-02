# DISCOVERY OF NATURAL LANGUAGE CONCEPTS IN INDIVIDUAL UNITS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Although deep convolutional networks have achieved improved performance in many natural language tasks, they have been treated as black boxes because they are difficult to interpret. Especially, little is known about how they represent language in their intermediate layers. In an attempt to understand the representations of deep convolutional networks trained on language tasks, we show that individual units are selectively responsive to specific morphemes, words, and phrases, rather than responding to arbitrary and uninterpretable patterns. In order to quantitatively analyze such intriguing phenomenon, we propose a concept alignment method based on how units respond to replicated text. We conduct analyses with different architectures on multiple datasets for classification and translation tasks and provide new insights into how deep models understand natural language.

# 1 INTRODUCTION

Understanding and interpreting how deep neural networks process natural language is a crucial and challenging problem. While deep neural networks have achieved state-of-the-art performances in neural machine translation (NMT) (Sutskever et al., 2014; Cho et al., 2014; Kalchbrenner et al., 2016; Vaswani et al., 2017), sentiment classification tasks (Zhang et al., 2015; Conneau et al., 2017) and many more, the sequence of non-linear transformations makes it difficult for users to make sense of any part of the whole model. Because of their lack of interpretability, deep models are often regarded as hard to debug and unreliable for deployment, not to mention that they also prevent the user from learning about how to make better decisions based on the model's outputs.

An important research direction toward interpretable deep networks is to understand what their hidden representations learn and how they encode informative factors when solving the target task. Among them, studies including Bau et al. (2017); Fong & Vedaldi (2018); Olah et al. (2017; 2018) have researched on what information is captured by individual or multiple units in visual representations learned for image recognition tasks. These studies showed that some of the individual units are selectively responsive to specific visual concepts, as opposed to getting activated in an uninterpretable manner. By analyzing individual units of deep networks, not only were they able to obtain more fine-grained insights about the representations than analyzing representations as a whole, but they were also able to find meaningful connections to various problems such as generalization of network (Morcos et al., 2018) or generating explanations for the decision of the model (Zhou et al., 2018a; Olah et al., 2018; Zhou et al., 2018b).

Since these studies of unit-level representations have mainly been conducted on models learned for computer vision-oriented tasks, little is known about the representation of models learned from natural language processing (NLP) tasks. Several studies that have previously analyzed individual units of natural language representations assumed that they align a predefined set of specific concepts, such as sentiment present in the text (Radford et al., 2017), text lengths, quotes and brackets (Karpathy et al., 2015). They discovered the emergence of certain units that selectively activate to those specific concepts. Building upon these lines of research, we consider the following question: What natural language concepts are captured by each unit in the representations learned from NLP tasks?

To answer this question, we newly propose a simple but highly effective concept alignment method that can discover which natural language concepts are aligned to each unit in the representation. Here we use the term unit to refer to each channel in convolutional representation, and natural language concepts to refer to the grammatical units of natural language that preserve meanings; i.e.

# Unit 108: [legal] [law] [legislative]

Better legal protection for accident victims.  
These rights are guaranteed under[law]  
This should be guaranteed by [law]  
This legislative proposal is unusual.  
Animal feed must be safe for animal health.

# Unit 711: should would not can

- That would not be democratic.  
That would be cheap and it would not be right.  
This is not how it should be in a democracy.  
I hope that you would not want that!  
- Europe cannot and must not tolerate this.

Figure 1: We discover the most activated sentences and aligned concepts to the units in hidden representations of deep convolutional networks. Aligned concepts appear frequently in most activated sentences, implying that those units respond selectively to specific natural language concepts.

morphemes, words, and phrases. Our approach first identifies the most activated sentences per unit and breaks those sentences into these natural language concepts. It then aligns specific concepts to each unit by measuring activation value of replicated text that indicates how much each concept contributes to the unit activation. This method also allows us to systematically analyze the concepts carried by units in diverse settings, including depth of layers, the form of supervision, and data-specific or task-specific dependencies.

The contributions of this work can be summarized as follows:

- We show that the units of deep CNNs learned in NLP tasks could act as a natural language concept detector. Without any additional labeled data or re-training process, we can discover, for each unit of the CNN, natural language concepts including morphemes, words and phrases that are present in the training data.  
- We systematically analyze what information is captured by units in representation in multiple settings by varying network architectures, tasks, and datasets. We use VD-CNN (Conneau et al., 2017) for sentiment and topic classification tasks on Yelp Reviews, AG News (Zhang et al., 2015), and DBpedia ontology dataset (Lehmann et al., 2015) and ByteNet (Kalchbrenner et al., 2016) for translation tasks on Europarl (Koehn, 2005) and News Commentary (Tiedemann, 2012) datasets.  
- We also analyze how aligned natural language concepts evolve as the layer gets deeper. As part of our analysis, we show that our interpretation of learned representations could be utilized at designing network architectures with fewer parameters but with comparable performance to baseline models.

# 2 RELATED WORK

# 2.1 INTERPRETATION OF INDIVIDUAL UNITS IN DEEP MODELS

Recent work on interpreting hidden representations at unit-level were mostly motivated from their counterparts in computer vision. In computer vision community, Zhou et al. (2015) retrieved image samples with the highest unit activation, for each of units in a CNN trained on image recognition tasks. They used these retrieved samples to show that visual concepts like color, texture and object parts are aligned to specific units, and the concepts were aligned to units by human annotators. Bau et al. (2017) introduced BRODEN dataset, which consists of pixel-level segmentation labels for diverse visual concepts and then analyzed the correlation between activation of each unit and such visual concepts. In their work, although aligning concepts which absent from BRODEN dataset requires additional labeled images or human annotation, they showed that some individual units respond to specific visual concepts.

On the other hand, Erhan et al. (2009); Olah et al. (2017); Simonyan et al. (2013) discovered visual concepts aligned to each unit by optimizing a random initial image to maximize the unit activation by gradient descent. In these cases, the resulting interpretation of each unit is in the form of optimized images, and not in the natural language form as the aforementioned ones. However, these continuous interpretation results make it hard for further quantitative analyses of discrete properties of representations, such as quantifying characteristics of representations in layer-wise (Bau et al., 2017) and correlations between the interpretability of a unit and regularization (Zhou et al., 2018a). Nevertheless, these methods have the advantage that the results are not constrained to a predefined set of concepts, giving flexibility as to which concepts are captured by each unit.

In the NLP domain, studies including Karpathy et al. (2015) and Tang et al. (2017) analyzed the internal mechanisms of deep models used for NLP and found intriguing properties that appear in units of hidden representations. Among those studies, the closest one to ours is Radford et al. (2017), who defined a unit as each element in the representation of an LSTM learned for language modeling and found that the concept of sentiment was aligned to a particular unit. Compared with these previous studies, we focus on discovering a much wider variety of natural language concepts, including any morphemes, words, and phrases all found in the training data. To the best our knowledge, this is the first attempt to discover concepts among all that exist in the form of natural language from the training corpus. By extending the scope of detected concepts to meaningful building blocks of natural language, we provide insights into how various linguistic features are encoded by the hidden units of deep representations.

# 2.2 ANALYSIS OF DEEP REPRESENTATIONS LEARNED FOR NLP TASKS

Most previous work that analyzes the learned representation of NLP tasks focused on constructing downstream tasks that predict concepts of interest. A common approach is to measure the performance of a regression/classification model that predicts the concept of interest to see whether those concepts are encoded in representation of a input sentence. For example, Conneau et al. (2018); Adi et al. (2017); Zhu et al. (2018) proposed several probing tasks to test whether the (non-)linear regression model can predict well the syntactic or semantic information from the representation learned on translation tasks or the skip-thought or word embedding vectors. Shi et al. (2016); Belinkov et al. (2017) constructed regression tasks that predict labels such as voice, tense, part-of-speech tag, and morpheme from the encoder representation of the learned model in translation task.

Compared with previous work, our contributions can be summarized as follows. (1) By identifying the role of the individual units, rather than analyzing the representation as a whole, we provide more fine-grained understanding of how the representations encode informative factors in training data. (2) Rather than limiting the linguistic features within the representation to be discovered, we focus on covering concepts of fundamental building blocks of natural language (morphemes, words, and phrases) present in the training data, providing more flexible interpretation results without relying on a predefined set of concepts. (3) Our concept alignment method does not need any additional labeled data or re-training process, so it can always provide deterministic interpretation results using only the training data.

# 3 APPROACH

We focus on convolutional neural networks (CNNs), particularly their character-level variants. CNNs have shown great success on various natural language applications, including translation, language modeling, and sentence classification (Kalchbrenner et al., 2016; Kim et al., 2016; Zhang et al., 2015; Conneau et al., 2017). Compared to deep architectures based on fully connected layers, CNNs are natural candidates for unit-level analysis because their channel-level representations are reported to work as templates for detecting concepts (Bau et al., 2017).

Our approach for aligning natural language concepts to units is summarized as follows. We first train a CNN model for each natural language task and retrieve training sentences that highly activate specific units. Interestingly, we discover morphemes, words, and phrases that appear dominantly within these retrieved sentences, implying that those concepts have a significant impact on the activation value of the unit. Then, we find a set of concepts which attribute a lot to the unit activation by measuring activation value of each replicated candidate concept, and align them to unit.

# 3.1 THE MODEL AND THE TASK

We analyze representations learned on three classification and four translation datasets shown in Table 1. Training details for each dataset are available in Appendix B. We then focus on the representations in each encoder layer of ByteNet and convolutional layer of VDCNN, because as Mou et al. (2016) pointed out, the representation of the decoder (the output layer in the case of classification) is specialized for predicting the output of the target task rather than for learning the semantics of the input text.

<table><tr><td>Dataset</td><td>Task</td><td>Model</td><td># of Layers</td><td># of Units</td></tr><tr><td>AG News</td><td>Ontology Classification</td><td>VDCNN</td><td>4</td><td>[64, 128, 256, 512]</td></tr><tr><td>DBpedia</td><td>Topic Classification</td><td>VDCNN</td><td>4</td><td>[64, 128, 256, 512]</td></tr><tr><td>Yelp Review</td><td>Polarity Classification</td><td>VDCNN</td><td>4</td><td>[64, 128, 256, 512]</td></tr><tr><td>WMT17&#x27; EN-DE</td><td>Translation</td><td>ByteNet</td><td>15</td><td>[1024] for all</td></tr><tr><td>WMT14&#x27; EN-FR</td><td>Translation</td><td>ByteNet</td><td>15</td><td>[1024] for all</td></tr><tr><td>WMT14&#x27; EN-CS</td><td>Translation</td><td>ByteNet</td><td>15</td><td>[1024] for all</td></tr><tr><td>EN-DE Europarl-v7</td><td>Translation</td><td>ByteNet</td><td>15</td><td>[1024] for all</td></tr></table>

Table 1: Datasets and model descriptions used in our analysis.

# 3.2 TOP  $K$  ACTIVATED SENTENCES PER UNIT

Once we train a CNN model for a given task, we feed again all sentences in the training data to the CNN and measure the activation in the unit of interest. The dimension of sentence representation is  $l \times d$ , where  $l$  is the length of the activation map and  $d$  is the number of units per layer. That is, the activation of each of  $d$  units is  $l$ -dimensional. For each unit, we retrieve top  $K$  training sentences with the highest mean activation over the  $l$  entries of the vector. Interestingly, some natural language patterns such as morphemes, words, phrases frequently appear in the retrieved sentences, implying that those concepts might have a large attribution to the activation value of that unit.

# 3.3 CONCEPT ALIGNMENT WITH REPLICATED TEXT

We propose a simple approach for identifying the concepts as follows. For constructing candidate concepts, we parse each of top  $K$  sentences with a constituency parser (Kitaev & Klein, 2018). Within the constituency-based parse tree, we define candidate concepts as all terminal and non-terminal nodes (e.g. from sentence John hit the balls, we obtain candidate concepts as  $\{John, hit\}$ , the balls, hit the balls, John hit the balls\}). We also break each word into morphemes using a morphological analysis tool (Virpioja et al., 2013) and add them to candidate concepts (e.g. from word balls, we obtain morphemes  $\{ball, s\}$ ). We repeat this process for every top  $K$  sentence and build a set of candidate concepts for unit  $u$ , which is denoted as  $\mathcal{C}_u = \{c_1, \dots, c_N\}$ , where  $N$  is the number of candidate concepts of the unit.

Next, we measure how each candidate concept attributes to the unit's activation value. We create a synthetic sentence by replicating each candidate concept so that its length is identical to the average length of all training sentences (e.g. candidate concept the ball is replicated as the ball the ball the ball...). Replicated sentences are denoted as  $\mathcal{R} = \{r_1,\dots,r_N\}$ , and each  $r_n\in \mathcal{R}$  is forwarded to CNN, and their activation value of unit  $u$  is measured as  $a_{u}(r_{n})\in \mathbb{R}$ , which is averaged over  $l$  entries. Finally, the degree of alignment (DoA) between a candidate concept  $c_{n}$  and a unit  $u$  is defined as follows:

$$
\mathrm {D o A} _ {u, c _ {n}} = a _ {u} \left(r _ {n}\right) \tag {1}
$$

In short, the DoA measures the extent to which unit  $u$ 's activation is sensitive to the presence of candidate concept  $c_{n}$ . If a candidate concept  $c_{n}$  appears in the top  $K$  sentences and unit's activation value  $a_{u}$  is responsive to  $c_{n}$  a lot, then  $\mathsf{DoA}_{u,c_n}$  gets large, suggesting that candidate concept  $c_{n}$  is strongly aligned to unit  $u$ .

Finally, for each unit  $u$ , we define a set of its aligned concepts  $\mathcal{C}_u^* = \{c_1^*,\dots,c_M^*\}$  as  $M$  candidate concepts with the largest DoA values in  $C_u$ . Depending on how we set  $M$ , we can detect different numbers of concepts per unit. In this experiments, we set  $M$  to 3.

# 4 EXPERIMENTS

# 4.1 EVALUATION OF CONCEPT ALIGNMENT

To quantitatively evaluate how well our approach aligns concepts, we measure how selectively each unit responds to the aligned concept. Motivated by Morcos et al. (2018), we define the concept selectivity of a unit  $u$ , to which a set of concepts  $\mathcal{C}_u^*$  that our alignment method detects, as follows:

$$
\operatorname {S e l} _ {u} = \frac {\mu_ {+} - \mu_ {-}}{\max  _ {s \in \mathcal {S}} a _ {u} (s) - \min  _ {s \in \mathcal {S}} a _ {u} (s)} \tag {2}
$$

![](images/e01082e1cc8e265341a87a06140057b3141aed99febba2d830a02f1402616ef1.jpg)  
Figure 2: Mean and variance of selectivity values over all units in representation learned for each dataset. Sentences including the concepts that our alignment method discovers always activate units significantly more than random sentences. See section 4.1 for details.

where  $S$  denotes all sentences in training set, and  $\mu_{+} = \frac{1}{|\mathcal{S}_{+}|}\sum_{s\in \mathcal{S}_{+}}a_{u}(s)$  is the average value of unit activation when forwarding a set of sentences  $S_{+}$ , which is defined as one of the following:

- replicate:  $S_{+}$  contains the sentences created by replicating each concept in  $C_u^*$ . As before, the sentence length is set as the average length of all training sentences for fair comparison.  
- inclusion:  $S_{+}$  contains the training sentences that include at least one concept in  $\mathcal{C}_u^*$ .  
- random:  $S_{+}$  contains randomly sampled sentences from the training data.

In contrast,  $\mu_{-} = \frac{1}{|\mathcal{S}_{-}|}\sum_{s\in \mathcal{S}_{-}}a_{u}(s)$  is the average value of unit activation when forwarding  $\mathcal{S}_{-}$ , which consists of sentences that do not include any concept in  $\mathcal{C}_u^*$ .

Intuitively, if unit  $u$ 's activation is highly sensitive to  $\mathcal{C}_u^*$  (i.e. those found by our alignment method) and if it is not to other factors, then  $\mathrm{Sel}_u$  gets large; otherwise,  $\mathrm{Sel}_u$  is near 0.

Figure 2 shows the mean and variance of selectivity values for all units learned in each dataset for the three  $S_{+}$ categories. Consistent with our intuition, in all datasets, the mean selectivity of the replicate set is the highest with a significant margin, that of inclusion set is the runner-up, and that of the random set is the lowest. These results support our claim that our method is successful to align concepts in which the unit responds selectively.

# 4.2 CONCEPT ALIGNMENT OF UNITS

Figure 3 shows examples of the top  $K$  sentences and the aligned concepts that are discovered by our method, for selected units. For each unit, we find the top  $K = 10$  sentences that activate the most in the several encoding layer of ByteNet and VDCNN, and select some of them (only up to five sentences are shown due to space constraints). We observe that some patterns appear frequently within the top  $K$  sentences. For example, in the top  $K$  sentences that activate unit 124 of 0th layer of ByteNet, the concepts of  $('', '')$ ,  $'-'$  appear in common, while the concepts of soft, software, wi appear frequently in the sentences for unit 19 of 1st layer of VDCNN. These results qualitatively show that individual units are selectively responsive to specific natural language concepts.

More interestingly, we discover that many units could capture specific meanings or syntactic roles beyond superficial, low-level patterns. For example, unit 690 of the 14th layer in ByteNet captures (what, who, where) concepts, all of which play the similar grammatical role. On the other hand, unit 224 of the 14th layer in ByteNet and unit 53 of the 0th layer in VDCNN each captures semantically similar concepts, with the ByteNet unit detecting the meaning of certainty in knowledge (sure, know, aware) and the VDCNN unit detecting years (1999, 1969, 1992). This suggests that, although we train character-level CNNs with feeding sentences as the form of discrete symbols (i.e. character indices), individual units could capture natural language concepts sharing similar semantic or grammatical role.

We note that there are units that detect concepts more abstract than just morphemes, words, or phrases, and for these units our method tends to align relevant lower-level concepts. For example, in units 477 and 244 of the 3rd layer in VDCNN, while each aligned concept emerges only once in the top  $K$  sentences, all top  $K$  sentences have similar nuances like positive and negative sentiments. In these cases, our method does capture relevant phrase-level concepts (e.g., very disappointing, absolute worst place), indicating that the higher-level nuance (e.g., negativity) is indirectly captured.

We also note that, because the number of morphemes, words and phrases present in training corpus is usually much greater than the number of units per layer, we do not expect to always align any

![](images/aac186f2aaf757316966c60cb4f53d2de36aa785483af878f5b1bee880694c66.jpg)  
Figure 3: Examples of top activated sentences and aligned concepts for some units in the several encoding layers of ByteNet and VDCNN. For each unit, aligned concept and its presence in top  $K$  sentences are painted by the same color. [#] symbol denotes morpheme concept. See section 4.2 for details.

natural language concepts in the corpus to one of the units. Our approach thus tends to find concepts that are considered as more important than others for solving the target task.

Overall, these results suggest how input sentences are represented in the hidden representation of the CNN as follows:

- Several units in the CNN learned on NLP tasks respond selectively to specific natural language concepts, rather than getting activated in an uninterpretable way. This means that these units can serve as detectors for specific natural language concepts.  
- There are units capturing syntactically or semantically related concepts, suggesting that they model the meaning or grammatical role shared between those concepts, as opposed to superficially modeling each natural language symbol.

# 4.3 CONCEPT DISTRIBUTION IN LAYERS

Using the concept alignments found earlier, we can visualize how concepts are distributed across layers. Figure 4 shows the concepts of the units in the 0th, 1st, 2nd, 3rd layer of VDCNN learned on AG-News dataset, and 0th, 5th, 9th and 14th layer of the ByteNet encoder learned on English-to-German Europarl dataset with their number of aligned units. For each layer, we sort concepts in a decreasing order by the number of aligned units and show 30 concepts most aligned. Recall that,

![](images/292988f70247bd59c1e82917c9944c202defa0b57e83583aee4ecff670f4f0d2.jpg)

![](images/b74a060c22b4e563bb9185a4e09f51f7b5976441bacd5666973aaebfc72efc52.jpg)  
Task: ag | Layer-02

![](images/1beb841adb7691f8700f2d490a872b31f6c56ddbf92d92198e22696509aeba61.jpg)  
Task: en-de-europarl | Layer-00

![](images/db7df412d511cee9fdba9ba35bd24396b56b6fcac97a9a6963ecbe800b43b57f.jpg)  
Task: en-de-europarl | Layer-09  
Figure 4: 30 concepts selected by the number of aligned units in four encoding layers in VDCNN learned on AG-News (top two rows), and ByteNet learned on the Europarl translation dataset (bottom two rows). [#] symbol denotes morpheme concept. See section 4.3 for details.

![](images/8555581cbdd9c7a8469b79548e4925143b6d55143c57bd79c2b35b507b7dfd1d.jpg)  
Task: ag | Layer-01

![](images/f20b49b4671de68157e645f5e770983e6dbb25118cb3ae22ef707960aef87d13.jpg)  
Task: ag | Layer-03

![](images/95a8554d814e0d1ffd63d3d7b986c4c62e687768e7d19108d86d6ee6d46594c2.jpg)  
Task: en-de-europarl | Layer-04

![](images/ae4ef72abea7383777e909ff99260990ad219b330462cca485a35c5bff2d85aa.jpg)  
Task: en-de-europarl | Layer-14

since we align concepts for each unit, there are concepts aligned to multiple units simultaneously. Concept distribution for other datasets are available in appendix C.

Overall, we find that data and task-specific concepts are likely to be aligned to many units. In AG News, since the task is to classify given sentences into following categories; World, Sports, Business and Science/Tech, concepts related to these topics commonly emerge. Similarly, we can see that units learned for Europarl dataset focus to encode some key words in the training corpus.

# 4.4 HOW DOES CONCEPT GRANULARITY EVOLVE WITH LAYER?

In computer vision tasks, visual concepts captured by units in CNN representations learned for image recognition tasks evolve with layer depths; color, texture concepts are emergent in earlier layers and more abstract concepts like parts and objects are emergent in deeper layers. To confirm that it also holds for representations learned in NLP tasks, we divide granularity of natural language concepts to morpheme, word and  $N$ -gram phrase  $(N = 2,3,4,5)$ , and observe the number of units that they are aligned in different layers.

Figure 5 shows this trend, where in lower layers such as the 0th layer, less phrase concepts but more morphemes and words are detected. This is because we use a character-level CNN, whose receptive fields of convolution may not be large enough to detect lengthy phrases. Further, interestingly in translation cases, we observe that aligned concepts significantly change in shallower layers (e.g. from the 0th to the 4th), but do not change much from middle to deeper layers (e.g. from the 5th to the 14th).

Thus, it remains for us to answer the following question: for the representations learned on translation datasets, why does concept granularity not evolve much in deeper layers? One possibility is that the capacity of the network is large enough so that the representations in middle layers could be sufficiently informative to solve the task. To validate this hypothesis, we re-train ByteNet from scratch while varying only layer depth of the encoder and fixing other conditions. We record their BLEU scores on the validation data as shown in Figure 6. The performance of the translation model does not change much with more than six encoder layers, but it significantly drops at the models with fewer than 4 encoder layers. This trend coincides with the result from Figure 5 that the evolution of concept granularity stops around middle-to-higher layers. This shared pattern suggests that about six encoder layers are enough to encode informative factors in the given datasets to perform optimally on the translation task. In deeper models, this may suggest that the middle layer's representation

![](images/6a17e26102d44f6517ced9194127a289629ea9dca3549b715aa5f1c1530c3c22.jpg)  
Figure 5: Aligned concepts are divided into six different levels of granularity: morphemes, words and N-gram phrases  $(N = 2,3,4,5)$ . The number of units aligned to each concept are shown layerwise across multiple datasets and tasks. Note that the number of units increases with layers in the classification models (i.e. [64, 128, 256, 512]), but in translation the number is constant (i.e. 1024) across all layers.

![](images/94c6371727329f626f98a614a316e019c55a293e2e70a20047906cf3f6910411.jpg)  
Figure 6: BLEU scores on the validation data for three translation models. We train ByteNet from scratch on each translation dataset by varying the number of encoder layers.

may be already informative enough to encode the input text, and our result may partly coincide with that of Mou et al. (2016), which shows that representation of intermediate layers is more transferable than that of deeper layers in language tasks, unlike in computer vision where deeper layers are usually more useful and discriminative.

# 5 CONCLUSION

We proposed a simple but highly effective concept alignment method for character-level CNNs to confirm that each unit of the hidden layers serves as detectors of natural language concepts. Using this method, we analyzed the characteristics of units with multiple datasets on classification and translation tasks. Consequently, we shed light on how deep representations capture natural language, and how they vary with various conditions.

An interesting future direction is to extend the concept coverage from natural language to more abstract forms such as sentence structure, nuance and tone. Another direction is to quantify the properties of individual units in other models widely used in NLP tasks. In particular, combining our definition of concepts with the attention mechanism (e.g. Bahdanau et al. (2015)) could be a promising direction, because it can reveal how the representations are attended by the model to capture concepts, helping us better understand the decision making process of popular deep models.

# REFERENCES

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mane, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaojiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015.  
Yossi Adi, Einat Kermany, Yonatan Belinkov, Ofer Lavi, and Yoav Goldberg. Fine-grained analysis of sentence embeddings using auxiliary prediction tasks. *ICLR*, 2017.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. *ICLR*, 2015.  
David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. In CVPR, 2017.  
Yonatan Belinkov, Nadir Durrani, Fahim Dalvi, Hassan Sajjad, and James Glass. What do neural machine translation models learn about morphology? In ACL, 2017.  
Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. In EMNLP, 2014.  
Alexis Conneau, Holger Schwenk, Loic Barrault, and Yann Lecun. Very deep convolutional networks for text classification. In EACL, 2017.  
Alexis Conneau, Germán Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni. What you can cram into a single \$\&!#\* vector: Probing sentence embeddings for linguistic properties. In ACL, 2018.  
Dumitru Erhan, Yoshua Bengio, Aaron Courville, and Pascal Vincent. Visualizing higher-layer features of a deep network. University of Montreal, 2009.  
Ruth Fong and Andrea Vedaldi. Net2vec: Quantifying and explaining how concepts are encoded by filters in deep neural networks. In CVPR, 2018.  
Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099, 2016.  
Andrej Karpathy, Justin Johnson, and Li Fei-Fei. Visualizing and understanding recurrent networks. arXiv preprint arXiv:1506.02078, 2015.  
Yoon Kim, Yacine Jernite, David Sontag, and Alexander M Rush. Character-aware neural language models. In AAAI, 2016.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *ICLR*, 2015.  
Nikita Kitaev and Dan Klein. Constituency parsing with a self-attentive encoder. ACL, 2018.  
Philipp Koehn. Europarl: A parallel corpus for statistical machine translation. In MT summit, volume 5, pp. 79-86, 2005.  
Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick Van Kleef, Soren Auer, et al. Dbpedia-a large-scale, multilingual knowledge base extracted from wikipedia. Semantic Web, 6(2):167-195, 2015.  
Ari S. Morcos, David G.T. Barrett, Neil C. Rabinowitz, and Matthew Botvinick. On the importance of single directions for generalization. In ICLR, 2018.  
Lili Mou, Zhao Meng, Rui Yan, Ge Li, Yan Xu, Lu Zhang, and Zhi Jin. How transferable are neural networks in nlp applications? In EMNLP, 2016.

Chris Olah, Alexander Mordvintsev, and Ludwig Schubert. Feature visualization. Distill, 2017. doi: 10.23915/distill.00007. https://distill.pub/2017/features-visualization.  
Chris Olah, Arvind Satyanarayan, Ian Johnson, Shan Carter, Ludwig Schubert, Katherine Ye, and Alexander Mordvintsev. The building blocks of interpretability. Distill, 2018. doi: 10.23915/distill.00010. https://distill.pub/2018/building-blocks.  
Alec Radford, Rafal Jozefowicz, and Ilya Sutskever. Learning to generate reviews and discovering sentiment. arXiv preprint arXiv:1704.01444, 2017.  
Franois Role and Mohamed Nadif. Handling the impact of low frequency events on co-occurrence based measures of word similarity. KDIR, 2011.  
Xing Shi, Inkit Padhi, and Kevin Knight. Does string-based neural mt learn source syntax? In EMNLP, 2016.  
Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.  
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In NIPS, 2014.  
Zhiyuan Tang, Ying Shi, Dong Wang, Yang Feng, and Shiyue Zhang. Memory visualization for gated recurrent neural networks in speech recognition. In ICASSP, 2017.  
Jrg Tiedemann. Parallel data, tools and interfaces in opus. In LREC. ELRA, 2012. ISBN 978-2-9517408-7-7.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Llion Jones, Jakob Uszkoreit, Aidan N Gomez, and Lukasz Kaiser. Attention is all you need. In NIPS, 2017.  
Sami Virpioja, Peter Smit, Stig-Arne Gronroos, and Mikko Kurimo. Morfessor 2.0: Python implementation and extensions for morfessor baseline. In Aalto University publication series. Department of Signal Processing and Acoustics, Aalto University, 2013.  
Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In NIPS, 2015.  
Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. Object detectors emerge in deep scene cnns. In ICLR, 2015.  
Bolei Zhou, David Bau, Aude Oliva, and Antonio Torralba. Interpreting deep visual representations via network dissection. IEEE TPAMI, 2018a.  
Bolei Zhou, Yiyou Sun, David Bau, and Antonio Torralba. Interpretable basis decomposition for visual explanation. In ECCV, 2018b.  
Xunjie Zhu, Tingfeng Li, and Gerard Melo. Exploring semantic properties of sentence embeddings. In ACL, 2018.
