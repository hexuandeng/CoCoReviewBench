# WORDS ARE ALL YOU NEED? LANGUAGE AS AN APPROXIMATION FOR REPRESENTATIONAL SIMILARITY

Anonymous authors

Paper under double-blind review

# ABSTRACT

Human similarity judgments are a powerful supervision signal for machine learning applications based on techniques such as contrastive learning, information retrieval, and model alignment, but classical methods for collecting human similarity judgments are too expensive to be used at scale. Recent methods propose using pre-trained deep neural networks (DNNs) to approximate human similarity, but pre-trained DNNs may not be available for certain domains (e.g., medical images, low-resource languages) and their performance in approximating human similarity has not been extensively tested. We conducted an evaluation of 611 pre-trained models across three domains – images, audio, video – and found that there is a large gap in performance between human similarity judgments and pre-trained DNNs. To address this gap, we propose a new class of similarity approximation methods based on language. To collect the language data required by these new methods, we also developed and validated a novel adaptive tag collection pipeline. We find that our proposed language-based methods are significantly cheaper, in the number of human judgments, than classical methods, but still improve performance over the DNN-based methods. Finally, we also develop 'stacked' methods that combine language embeddings with DNN embeddings, and find that these consistently provide the best approximations for human similarity across all three of our modalities. Based on the results of this comprehensive study, we provide a concise guide for researchers interested in collecting or approximating human similarity data. To accompany this guide, we also release all of the similarity and language data, a total of 206,339 human judgments, that we collected in our experiments, along with a detailed breakdown of all modeling results.

# 1 INTRODUCTION

Similarity judgments have long been used as a tool for studying human representations, both in cognitive science (Shepard, 1980; 1987; Tversky, 1977; Tenenbaum & Griffiths, 2001), as well as in neuroscience, as exemplified by the rich literature on the representational similarity between humans and machines (Schrimpf et al., 2020; Kell et al., 2018; Linsley et al., 2017; Langlois et al., 2021; Yamins et al., 2014)). Recent research in machine learning suggests that incorporating human similarity judgments in model training can play an important role in a variety of paradigms such as human alignment (Esling et al., 2018), contrastive learning (Khosla et al., 2020), information retrieval (Parekh et al., 2020), and natural language processing (Gao et al., 2021).

However, building a large dataset based on human similarity judgments is very expensive and often infeasible since the number of judgments required is quadratic in the number of stimuli – for  $N$  stimuli,  $O(N^2)$  judgments are required<sup>1</sup>. For example, to fully quantify the similarity of all possible dyadic pairs of 50,000 images, one needs to collect on the order of 1.25 billion  $(\sim \frac{50000^2}{2})$  human similarity judgments. Thus, human judgments are the main bottleneck for machine-learning methods based on similarity. For this reason, the majority of available human similarity datasets are small by machine learning standards (up to a few thousand objects).

Advancements in deep learning have brought an alternative approach that does not require extensive collection of human judgments. Specifically, the idea is to use the similarity between hidden representations in pre-trained deep neural networks (DNNs) to approximate human similarity (Peterson et al., 2018; Jha et al., 2020; Marjieh et al., 2022; Hebart et al., 2020; Roads & Love, 2021). Some of these methods also suggest fine-tuning representations on a small training set of human similarity judgments (Peterson et al., 2018). This, in turn, results in a significant reduction in the number of required human judgments down to  $O(1)$  (given the pre-trained model). While such methods are promising, they still require access to strong pre-trained models which may not necessarily be available in all domains (e.g., medical datasets, niche modalities, low-resource languages, etc.). In addition, representations obtained from neural networks may not always overlap with human similarity representations, given that the models can be trained for different objectives (i.e., their embeddings may be poor approximations for human similarity).

A comprehensive comparison to assess which models perform well in predicting human similarity across different modalities is currently lacking in the literature. To this end, one of our main contributions in this paper is providing a first-of-its-kind large-scale evaluation of over 600 publicly-available pre-trained models as approximations for human similarity judgments on three modalities (images, audio, video). Our experiments reveal that there is a large gap in performance between the  $O(1)$  DNN methods and the classical  $O(N^2)$  similarity method we used as the baseline.

To address this gap, we propose a new class of  $O(N)$  methods to efficiently and accurately approximate human similarity based on language. This is motivated by a long line of research in cognitive science suggesting that language is an extremely efficient way for humans to communicate information about their sensory environment (Murphy, 2004; Zaslavsky et al., 2018; Piantadosi et al., 2011; Jaeger & Levy, 2006). This in turn suggests that we can use textual descriptors to approximate similarity judgments across different modalities. Moreover, such textual descriptors can be collected

![](images/fcc25374cb43a4225148aa66c634e7cdf4bb38bcb8560e3288ce6ac0784000f2.jpg)  
Figure 1: Comparing human similarity scores gathered through crowdsourcing with ML pipelines. We used data from three modalities: images, audio, and video. For each modality, we extracted deep model embeddings and gathered human captions and tags. Word- and language-embedding models, as well as simple word-frequency analysis, were used to predict human similarity judgments.

at the cost of  $O(N)$  human judgments (as people describe individual stimuli rather than pairs), which renders this method scalable.

We consider two approaches for approximating similarity from text data. One approach is to use pre-trained Large Language Models (LLM) to produce vector embeddings of the textual descriptions, and then use a measure of distance between these embeddings to approximate human similarity. This method is more domain-agnostic than the  $O(1)$  deep learning methods as it only requires access to a pre-trained LLM regardless of the modality of the original dataset. However, there are some cases where the domain may be out-of-distribution for all available LLMs (e.g., niche technical fields), or where no LLMs are available at all (e.g., low-resource languages). In such cases, the other approach is to use Word-Frequency Analysis (WFA) methods from classical text processing literature (Barrios et al., 2016; Rouge, 2004; Beel et al., 2016),

As for the textual descriptions themselves, we consider two types, namely, free-text captions and concise word tags. Collecting captions for machine learning datasets is a well-established practice and can easily be done through crowdsourcing platforms. On the other hand, there is no consensus on best practices for collecting tags without a pre-existing taxonomy (i.e., open-set labels). To address this, we propose a novel adaptive tag mining pipeline called Sequential Transmission Evaluation Pipeline (STEP-Tag) which we describe in Section 2.2.4. As we will show, STEP-Tag allows to collect meaningful, diverse, and high-quality word tags for target stimuli in an online crowdsourcing environment.

Finally, we propose one additional set of hybrid approximation methods that combine sensory information with textual descriptions while still requiring  $O(N)$  human judgments. For this approach, we propose to stack the embeddings derived from both domain-specific models (e.g., output from the last layer of an image classifier) with the LLM embedding of the respective textual description. When multi-modal models are available, we can similarly leverage the joint embedding of both the stimulus and its textual description.

We evaluate all of these novel and existing methods across multiple modalities. We test the relative contributions of linguistic and sensory information in approximating human similarity and show that our proposed language-based methods provide both accurate and efficient approximations across modalities, even though they do not require a trained modality-specific deep learning model. Crucially, with this large-scale evaluation, we are able for the first time to provide researchers with a comprehensive guide of the tools to use for approximating human similarity at scale.

To summarize, our contributions are as follows:

- We conduct a comprehensive comparison of human similarity approximation methods.  
- We propose a novel modality-agnostic method for approximating similarity based on text and show that it is both efficient and competitive in terms of performance.  
- We propose STEP-Tag, a novel adaptive tagging pipeline, and show that it is effective for crowdsourcing high-quality and diverse sets of word tags.  
- We synthesize our findings into a detailed guide for researchers interested in approximating human similarity judgments at scale.  
- We collect and release ground-truth and approximated versions of a large behavioral dataset  $(N = 1,492)$  across three different domains (images, audio, video), including two text-approximated similarity matrices for 1,000 audio clips and 1,000 video clips.

# 2 DATASETS

# 2.1 STIMULI

Throughout this work, we considered five stimulus datasets across three different modalities – images, audio, and video – consisting of a total of 31,320 dyadic pairs labeled with similarity.

Images For images, we considered three datasets of common objects introduced in Peterson et al. (2018) – namely, animals, furniture, and vegetables – each consisting of 7,140 dyadic pairs (all unique pairs over 120 images).

Audio For audio, we used the RAVDESS corpus (Livingstone & Russo (2018), released under a CC Attribution license), which consists of semantically neutral sentences spoken by 24 US American actors to convey a specific target emotion. To construct a 1,000-recording subset, we selected 3 emotions per speaker per sentence. We randomly omitted 104 emotional stimuli and included all 96 neutral recordings (the dataset only contains 2 neutral recordings per speaker per sentence). To construct the subset composed of 4,950 dyadic pairs (all unique pairs over 100 recordings), we randomly selected  $\sim 13$  recordings per emotion from the 1,000.

Video Finally, for the video dataset, we considered the Mini-Kinetics-200 dataset (Xie et al. (2018), released under a CC BY 4.0 International License), which contains a large set of short video clips of human activities from 200 activity classes. Specifically, we focused on the validation split, which contains 5,000 videos in total. To construct our 1,000-video dataset, we sampled 5 random videos from each of the 200 activity categories. The 100-video subset (4,950 dyadic pairs) used in the similarity judgment collection experiment was then generated by sampling 100 random stimuli from the 1,000 list.

# 2.2 HUMAN JUDGMENT COLLECTION

# 2.2.1 PARTICIPANTS

We collected data from  $N = 1,492$  US participants for the new behavioral experiments reported in this paper. Participants were recruited anonymously from Amazon Mechanical Turk and provided informed consent under an approved IRB protocol prior to participating in our studies. Participants earned 9-12 USD per hour, and each session lasted less than 30 minutes. To help recruit reliable participants, we required that participants are at least 18 years of age, reside in the United States and have participated in more than 5,000 previous tasks with a  $99\%$  approval rate (see Supplementary Section B for additional details about the behavioral experiments). All experiments were implemented with the Dallinger and PsyNet frameworks designed for automation of large-scale behavioral research (Harrison et al., 2020). In Supplementary Section A.1, we include the data that was collected, instructions used, and code for replication of the behavioral experiments. We also provide the code for computational experiments and analysis.

# 2.2.2 SIMILARITY JUDGMENTS

We collected two batches of pairwise similarity judgements, one for each of the audio and video subsets, and were provided access to the similarity matrices for the three image datasets by the authors of Peterson et al. (2018). For each pair we collected  $\sim 5$  similarity judgments to average out inter-rater noise.

# 2.2.3 CAPTIONS

We collected free-text captions for the video and audio datasets. Captions for the image datasets were already collected by Marjieh et al. (2022) and used here with permission. For each stimulus, we collected  $\sim 10$  captions.

# 2.2.4 TAGS

We propose a novel adaptive tag pipeline for simultaneous data collection and evaluation called Sequential Transmission Evaluation Pipeline (STEP) and apply it in the context of semantic tag mining (STEP-Tag). Our paradigm, STEP-Tag, allows researchers to efficiently collect high-quality word tags for a given stimulus (Figure 2) and extends existing crowdsourcing text-mining techniques (Von Ahn & Dabbish, 2008; 2004; Krishna et al., 2017; Law et al., 2007) by integrating ideas from transmission chain experiments (Kirby et al., 2008; Griffiths & Kalish, 2005). In STEP-Tag, participants adaptively create tags for a set of target stimuli and simultaneously evaluate the annotations made by previous participants. In each trial, participants are first given a stimulus (e.g., an image or audio fragment) and rate the relevance of tags that were created by other participants (on a 5-interval Likert scale) or flag a tag if they find it inappropriate (with tags removed if more than two people flag the tag). Next, participants are also given the opportunity to add new tags if they feel a relevant tag that describes the stimulus is missing. The results of the annotation procedure of one participant then

![](images/e4f1137d4c86ed1dcc9e99938b69632655e61d351a6eb2ec701a950a0a085671.jpg)  
Figure 2: STEP-Tag, our novel tag-mining paradigm. We ran an adaptive process in which results of one iteration are used as inputs for subsequent iterations. In every iteration, participants can add a new tag, rate the relevance of existing tags or flag tags that are inappropriate.

propagate to the next participant (additional details about the paradigm, and screenshots are provided in Supplementary Section B.6). Ultimately, as the process unfolds over many iterations, meaningful tags are extracted and validated by multiple participants, enabling efficient open-label collection of a desired dataset.

To validate STEP-Tag, we compared it against several baselines: (i) randomly selecting only a single high-rated tag from the last iteration of STEP-Tag per stimulus, (ii) using tags only from the first iteration of STEP-Tag (equivalent to non-adaptive tag collection), and (iii) using class labels instead of tags. We found that tags produced after multiple iterations of STEP-Tag outperformed all three baselines in terms of quality (i.e., downstream performance for similarity reconstruction) and diversity (see Supplementary Section B.6.1).

# 3 MODELS

# 3.1 DNN-BASED METHODS

We tested a wide range of pre-trained ML models that do not rely on text (overall we tested 611 models) and compared their internal representations to human similarity judgments and text-based predictions (Figure 1A). We compiled our model pool by leveraging pre-trained model repositories (or zoos) available online. In particular, for images we use 569 pre-trained models from the pytorch-image-models package timm (Wightman, 2019), for audio we use 36 pre-trained models available in the torchaudio package (Yang et al., 2021) (see also Supplementary Figure 10 for an analysis of layer depth), and for video we use 6 pre-trained models available from the PyTorchVideo package (Fan et al., 2021). Because of the recent success of multimodal training, we additionally included 9 multimodal models based on CLIP from OpenAI's public implementation (https://github.com/openai/CLIP) for the image datasets, and compared them to "stacked" representations (i.e., concatenating embeddings from separate image and text models).

# 3.2 LLM-BASED METHODS

# 3.2.1 TAGS

To embed tags we used ConceptNet Numberbatch (CNNB) which is a word-embedding model trained on the ConceptNet knowledge graph that leverages other popular word embedding models such as word2vec and GloVe (Speer et al., 2017). We experimented with several algorithms for computing similarity between sets (or multi-sets) of tags and share the details in Supplementary Section C.1.2.

# 3.2.2 CAPTIONS

To embed captions, we used four pre-trained LLMs from HuggingFace (Wolf et al., 2020): 'bert-base-uncased', 'deberta-xlarge-mnli', 'sup-simcse-bert-base-uncased', and 'sup-simcse-roberta-large'. SimCSE is a pre-training procedure that uses semantic entailment in a contrastive learning objective (Gao et al., 2021). According to BERTScore (Zhang et al., 2020), the latter three models are ranked in the top 40 models in terms of correlation with human evaluations on certain tasks, with 'deberta-xlarge-mnli' ranked first. However, in our experiments, we found that embedding similarity computed from 'sup-simcse-roberta-large' has the highest correlation with human similarity judgments out of the four models. For SimCSE-based models, we used representations from the (final) embedding layer (where the SimCSE contrastive objective is actually applied). For the other two models, we computed embeddings from every layer, but restricted the main analysis to embeddings from the penultimate layers. This was done in order to be consistent with our procedure for DNNs.

# 3.3 STACKING METHODS

We produce stacked representations for each modality by concatenating the single best-performing (see Figure 3) LLM's embeddings with the embeddings from the five best-performing DNNs into a single set of long embeddings. Since the two sets of embeddings come from different spaces, we add a single tunable hyperparameter for rescaling the LLM embeddings. This hyperparameter can be set manually, but we use a small number of ground-truth similarity judgments (we use dyadic pairs for just 20 stimuli) to optimize it automatically.

# 3.4 WORD FREQUENCY ANALYSIS (WFA) METHODS

The aim of the WFA methods is to enable similarity approximation from language using traditional embedding-free techniques. Such techniques are particularly useful for low-resource languages or cross-cultural comparisons (Cowen & Keltner, 2017; Barrett, 2020), for which pre-trained models are lacking, as they work solely on the basis of the text itself. The WFA methods we considered included measuring co-occurrence, Rouge score, bm25s, and tfidf. We provide details on each of these procedures in Supplementary Section C.2.

# 3.5 PERFORMANCE METRIC

We quantified performance by computing the Pearson correlation  $r$  between approximated similarity scores and the ground-truth human similarity scores for all the unique dyadic pairs in a dataset. We compared the performance of the different prediction methods to the inter-rater reliability (IRR) of participants, which serves as an approximate upper-bound on performance. Following Peterson et al. (2018), we computed IRR for each human similarity matrix using the split-half correlation method with a Spearman-Brown correction (Brown, 1910).

# 4 RESULTS

Figure 3 summarizes the performance of the various techniques across the three modalities. Note that the image modality results in Figure 3A are averaged across the three image datasets and only show the top 50 methods for this modality due to space constraints. Figure 3D shows the mean performance of the methods of each type for each modality. When viewing these results, a clear hierarchy emerges. While no approximation methods can perfectly match the ground-truth pairwise similarity, (see the gap between the methods and IRR), stacked ones get close and are consistently more aligned with human similarity than other methods across all three modalities. Text-based methods come next in this hierarchy, followed by DNN-based ones.

The pre-eminence of stacked results suggests that LLMs and DNNs capture at least some different sources of variance in human similarity judgments. This is reinforced by our surprising finding that stacked representations from CLIP, a state-of-the-art jointly pre-trained multi-modal model, do not outperform stacked representations from independently trained models. We hypothesize that this happens because information is lost from both modalities when optimizing for a joint embedding. However, we note that the modest size of the performance gap between stacked and LLMs/DNNs,

![](images/f92a0e0d1027f7919d5c2a3b19848904db0b1fc738ddd8e16111c4807dc2ec61.jpg)

![](images/0b9b482d33180173fcacab951ef982ce005b573d571d22fd234dccc798d0533e.jpg)

![](images/cc32e1b8edceea80a75883a9f000d17cf836b697e345102608382990240528aa.jpg)

![](images/e0234c28b383c7dfbd8b6bb3b145e0fa859b95ae09abc2f527d62c55692b9052.jpg)  
Figure 3: Correlation to human similarity. A: Top 50 models averaged over the 3 image datasets. B: Audio dataset. C: Video dataset. Each DNN model bar averages over multiple variants of the same architecture; the dots overlaid on the bars indicate the average correlation of each particular variant of the architecture. D: Average for each method type for each modality. The error bars are standard deviations.

suggests that there is also significant overlap between aspects of human similarity captured by language and perception.

To investigate the effect of architecture and downstream task (e.g., classification) performance on alignment of DNNs with human similarity, for the image modality we compared similarity approximation performance against the number of model parameters on a log scale (Figure 4A) and ImageNet classification performance (Deng et al., 2009) (Figure 4B). Overall, we found a positive correlation between similarity approximation performance and the number of model parameters

$(r = 0.39, p < 0.001)$  and a smaller but still significant positive correlation with performance on ImageNet  $(r = 0.26, p < 0.001)$ . There were some notable exceptions with particularly high ImageNet performance but low similarity performance, such as the image transformer BEiT (Bao et al., 2021).

Finally, we leverage both DNN-based methods and our proposed language-based methods to approximate similarity matrices that would otherwise require an unaffordable number of human similarity judgments to collect all dyadic pairs. Specifically, we approximate the two similarity matrices corresponding to all 1,000 audio clips and 1,000 video clips in our datasets using every method listed for each of those modalities in Figure 3. We provide visualizations of the resulting matrices at https://words-are-all-you-need.s3.amazon.com/index.html. We note that to exhaustively collect all dyadic pairs with five judgments per pair would normally require roughly 2.5 million human judgments for each of these matrices.

![](images/1d4c018d353102d409b2f97c9b177a16a0c87eea50f77659b8b9c12098931155.jpg)

![](images/023fa2e9b09736dcebf563307fce45638d074bd504e226fd62b78993dcda03b7.jpg)  
Figure 4: Correlation to human similarity judgments as a function of A: number of model parameters; and B: ImageNet accuracy.

# 5 DISCUSSION

In this work, we compared novel and existing methods for approximating human similarity judgments. Based on the results outlined above, we are now able to provide researchers with a best-practice guide to collecting similarity datasets. Our guide is based on two bottlenecks that researchers may face: one is the limit on the number of judgments that can be collected (e.g., due to cost) and the second is the availability of pre-trained models (i.e., either DNNs or LLMs). Our results make it clear that deep learning can provide good approximations for human similarity. In fact, when both pre-trained LLMs and DNNs are available, stacking their representations is consistently the best approach. However,

![](images/71f0f63c1efc2ef02055465c78a6a034d3da47e0438674b738bfe02d1766d9cb.jpg)  
Figure 5: Guide to collecting and estimating human similarity judgments at scale.

even when neither type of pre-trained models are available, we suggest that classical word-frequency analysis methods still provide researchers with an efficient and competitive method for approximating human similarity. Our guide, comprehensively covering these and other cases, is laid out in Figure 5.

One limitation of this work is that while similarity proxies generated from our pipeline can support ML datasets, they are also at risk of baking in high-level human biases that can lead to adverse societal implications, such as amplifying race and gender gaps. Researchers should devote utmost care to what they choose to incorporate in their training objective. On the positive side, we believe that our approach paves the way for the study of cross-cultural variation of human semantic representations by providing efficient tools for crowdsourcing high-quality semantic descriptors across languages. This is particularly relevant for low-resource languages, where our tag-mining techniques can work even with the absence of pre-trained ML models (Thompson et al., 2020; Barrett, 2020). Taken together, our results showcase how we can leverage language to make machine representations more human-like. Moreover, it highlights the importance of combining machine learning and cognitive science approaches for mutually advancing both fields. In particular, we believe that the methodologies adopted in this work have the potential to greatly advance basic research on naturalistic representations in cognitive science.

# REFERENCES

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems, 33:12449-12460, 2020.  
Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. data2vec: A general framework for self-supervised learning in speech, vision and language, 2022.  
Hangbo Bao, Li Dong, and Furu Wei. BEiT: BERT pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021.  
H Clark Barrett. Towards a cognitive science of the human: cross-cultural approaches and their urgency. Trends in Cognitive Sciences, 24(8):620-638, 2020.

Federico Barrios, Federico López, Luis Argerich, and Rosa Wachenchauzer. Variations of the similarity function of textrank for automated summarization. arXiv preprint arXiv:1602.03606, 2016.  
Joeran Beel, Bela Gipp, Stefan Langer, and Corinna Breitinger. Paper recommender systems: a literature survey. International Journal on Digital Libraries, 17(4):305-338, 2016.  
William Brown. Some experimental results in the correlation of mental abilities 1. British Journal of Psychology, 1904-1920, 3(3):296-322, 1910.  
Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al. WavLM: Large-scale self-supervised pre-training for full stack speech processing. arXiv preprint arXiv:2110.13900, 2021.  
Alan S Cowen and Dacher Keltner. Self-report captures 27 distinct categories of emotion bridged by continuous gradients. Proceedings of the National Academy of Sciences, 114(38):E7900-E7909, 2017.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248-255. IEEE, 2009.  
Philippe Esling, Adrien Bitton, et al. Generative timbre spaces: regularizing variational auto-encoders with perceptual metrics. arXiv preprint arXiv:1805.08501, 2018.  
Haoqi Fan, Tullie Murrell, Heng Wang, Kalyan Vasudev Alwala, Yanghao Li, Yilei Li, Bo Xiong, Nikhila Ravi, Meng Li, Haichuan Yang, Jitendra Malik, Ross Girshick, Matt Feiszli, Aaron Adcock, Wan-Yen Lo, and Christoph Feichtenhofer. PyTorchVideo: A deep learning library for video understanding. In Proceedings of the 29th ACM International Conference on Multimedia, 2021. https://pytorchvideo.org/.  
Christoph Feichtenhofer. X3d: Expanding architectures for efficient video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 203-213, 2020.  
Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. SlowFast networks for video recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6202-6211, 2019.  
Tianyu Gao, Xingcheng Yao, and Danqi Chen. SimCSE: Simple contrastive learning of sentence embeddings. arXiv preprint arXiv:2104.08821, 2021.  
Thomas L Griffiths and Michael L Kalish. A bayesian view of language evolution by iterated learning. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 27, 2005.  
Peter Harrison, Raja Marjieh, Federico Adolfi, Pol van Rijn, Manuel Anglada-Tort, Ofer Tchernichovski, Pauline Larrouy-Maestri, and Nori Jacoby. Gibbs sampling with people. Advances in Neural Information Processing Systems, 33:10659-10671, 2020.  
Martin N Hebart, Charles Y Zheng, Francisco Pereira, and Chris I Baker. Revealing the multidimensional mental representations of natural objects underlying human similarity judgements. Nature Human Behaviour, 4(11):1173-1185, 2020.  
Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451-3460, 2021.  
T Jaeger and Roger Levy. Speakers optimize information density through syntactic reduction. Advances in Neural Information Processing Systems, 19, 2006.  
Kevin G Jamieson and Robert D Nowak. Low-dimensional embedding using adaptively selected ordinal data. In 2011 49th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pp. 1077-1084. IEEE, 2011.

Aditi Jha, Joshua Peterson, and Thomas L Griffiths. Extracting low-dimensional psychological representations from convolutional neural networks. arXiv preprint arXiv:2005.14363, 2020.  
Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The Kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.  
Alexander JE Kell, Daniel LK Yamins, Erica N Shook, Sam V Norman-Haignere, and Josh H McDermott. A task-optimized neural network replicates human auditory behavior, predicts brain responses, and reveals a cortical processing hierarchy. Neuron, 98(3):630-644, 2018.  
Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in Neural Information Processing Systems, 33:18661-18673, 2020.  
Simon Kirby, Hannah Cornish, and Kenny Smith. Cumulative cultural evolution in the laboratory: An experimental approach to the origins of structure in human language. Proceedings of the National Academy of Sciences, 105(31):10681-10686, 2008.  
Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International Journal of Computer Vision, 123(1):32-73, 2017.  
Thomas Langlois, Haicheng Zhao, Erin Grant, Ishita Dasgupta, Tom Griffiths, and Nori Jacoby. Passive attention in artificial neural networks predicts human visual selectivity. Advances in Neural Information Processing Systems, 34, 2021.  
Edith LM Law, Luis Von Ahn, Roger B Dannenberg, and Mike Crawford. TagATune: A game for music and sound annotation. In ISMIR, volume 3, pp. 2, 2007.  
Kristin Lemhöfer and Mirjam Broersma. Introducing lextale: A quick and valid lexical test for advanced learners of english. Behavior research methods, 44(2):325-343, 2012.  
Drew Linsley, Sven Eberhardt, Tarun Sharma, Pankaj Gupta, and Thomas Serre. What are the visual features underlying human versus machine vision? In Proceedings of the IEEE International Conference on Computer Vision Workshops, pp. 2706-2714, 2017.  
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10012-10022, 2021.  
Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. arXiv preprint arXiv:2201.03545, 2022.  
Steven R Livingstone and Frank A Russo. The Ryerson audio-visual database of emotional speech and song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in north american english. PloS one, 13(5):e0196391, 2018.  
Raja Marjieh, Ilia Sucholutsky, Theodore R Sumers, Nori Jacoby, and Thomas L Griffiths. Predicting human similarity judgments using large language models. arXiv preprint arXiv:2202.04728, 2022.  
Alice E Milne, Roberta Bianco, Katarina C Poole, Sijia Zhao, Andrew J Oxenham, Alexander J Billig, and Maria Chait. An online headphone screening test based on dichotic pitch. Behavior Research Methods, 53(4):1551-1562, 2021.  
Gregory Murphy. The big book of concepts. MIT press, 2004.  
Zarana Parekh, Jason Baldridge, Daniel Cer, Austin Waters, and Yinfei Yang. Crisscrossed captions: Extended intramodal and intermodal semantic similarity judgments for MS-COCO. arXiv preprint arXiv:2004.15020, 2020.  
Joshua C Peterson, Joshua T Abbott, and Thomas L Griffiths. Evaluating (and improving) the correspondence between deep neural networks and human representations. Cognitive Science, 42 (8):2648-2669, 2018.

Steven T Piantadosi, Harry Tily, and Edward Gibson. Word lengths are optimized for efficient communication. Proceedings of the National Academy of Sciences, 108(9):3526-3529, 2011.  
Mirco Ravanelli, Titouan Parcollet, Peter Plantinga, Aku Rouhe, Samuele Cornell, Loren Lugosch, Cem Subakan, Nauman Dawalatabad, Abdelwahab Heba, Jianyuan Zhong, Ju-Chieh Chou, Sung-Lin Yeh, Szu-Wei Fu, Chien-Feng Liao, Elena Rastorgueva, François Grondin, William Aris, Hwidong Na, Yan Gao, Renato De Mori, and Yoshua Bengio. SpeechBrain: A general-purpose speech toolkit, 2021. arXiv:2106.04624.  
Brett D Roads and Bradley C Love. Enriching ImageNet with human similarity judgments and psychological embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3547-3557, 2021.  
Lin CY Rouge. A package for automatic evaluation of summaries. In Proceedings of Workshop on Text Summarization of ACL, Spain, 2004.  
Martin Schrimpf, Jonas Kubilius, Ha Hong, Najib J Majaj, Rishi Rajalingham, Elias B Issa, Kohitij Kar, Pouya Bashivan, Jonathan Prescott-Roy, Franziska Geiger, et al. Brain-Score: Which artificial neural network for object recognition is most brain-like? BioRxiv, pp. 407007, 2020.  
Roger N Shepard. Multidimensional scaling, tree-fitting, and clustering. Science, 210(4468):390-398, 1980.  
Roger N Shepard. Toward a universal law of generalization for psychological science. Science, 237 (4820):1317-1323, 1987.  
Robyn Speer, Joshua Chin, and Catherine Havasi. Conceptnet 5.5: An open multilingual graph of general knowledge. In Thirty-first AAAI Conference on Artificial Intelligence, 2017.  
Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pp. 6105-6114. PMLR, 2019.  
Joshua B Tenenbaum and Thomas L Griffiths. Generalization, similarity, and bayesian inference. Behavioral and brain sciences, 24(4):629-640, 2001.  
Bill Thompson, Seán G Roberts, and Gary Lupyan. Cultural influences on word meanings revealed through large-scale semantic alignment. Nature Human Behaviour, 4(10):1029–1038, 2020.  
Amos Tversky. Features of similarity. Psychological review, 84(4):327, 1977.  
Luis Von Ahn and Laura Dabbish. Labeling images with a computer game. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pp. 319-326, 2004.  
Luis Von Ahn and Laura Dabbish. Designing games with a purpose. Communications of the ACM, 51(8):58-67, 2008.  
Johannes Wagner, Andreas Triantafyllopoulos, Hagen Wierstorf, Maximilian Schmitt, Felix Burkhardt, Florian Eyben, and Björn W. Schuller. Dawn of the transformer era in speech emotion recognition: closing the valence gap, 2022.  
Shu wen Yang, Po-Han Chi, Yung-Sung Chuang, Cheng-I Jeff Lai, Kushal Lakhotia, Yist Y. Lin, Andy T. Liu, Jiatong Shi, Xuankai Chang, Guan-Ting Lin, Tzu-Hsien Huang, Wei-Cheng Tseng, Ko tik Lee, Da-Rong Liu, Zili Huang, Shuyan Dong, Shang-Wen Li, Shinji Watanabe, Abdelrahman Mohamed, and Hung yi Lee. SUPERB: Speech Processing Universal PERformance Benchmark. In Proc. Interspeech 2021, pp. 1194-1198, 2021. doi: 10.21437/Interspeech.2021-1775.  
Ross Wightman. PyTorch image models. https://github.com/rwrightman/pytorch-image-models, 2019.  
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumont, Clement Delangue, Anthony Moi, Pierrick Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38-45. Association for Computational Linguistics, 2020.

Kevin JP Woods, Max H Siegel, James Traer, and Josh H McDermott. Headphone screening to facilitate web-based auditory experiments. Attention, Perception, & Psychophysics, 79(7): 2064-2072, 2017.  
Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. Rethinking spatiotemporal feature learning: Speed-accuracy trade-offs in video classification. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 305–321, 2018.  
Daniel Yamins. An optimization-based approach to understanding sensory systems. The Cognitive Neurosciences, 4(V1):381, 2020.  
Daniel LK Yamins, Ha Hong, Charles F Cadieu, Ethan A Solomon, Darren Seibert, and James J DiCarlo. Performance-optimized hierarchical models predict neural responses in higher visual cortex. Proceedings of the National Academy of Sciences, 111(23):8619-8624, 2014.  
Yao-Yuan Yang, Moto Hira, Zhaoheng Ni, Anjali Chourdia, Artyom Astafurov, Caroline Chen, Ching-Feng Yeh, Christian Puhrsch, David Pollack, Dmitriy Genzel, Donny Greenberg, Edward Z. Yang, Jason Lian, Jay Mahadeokar, Jeff Hwang, Ji Chen, Peter Goldsborough, Prabhat Roy, Sean Narethiran, Shinji Watanabe, Soumith Chintala, Vincent Quenneville-Bélair, and Yangyang Shi. Torchaudio: Building blocks for audio and speech processing. arXiv preprint arXiv:2110.15018, 2021.  
Noga Zaslavsky, Charles Kemp, Terry Regier, and Naftali Tishby. Efficient compression in color naming and its evolution. Proceedings of the National Academy of Sciences, 115(31):7937-7942, 2018.  
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SkeHuCVFDr.
