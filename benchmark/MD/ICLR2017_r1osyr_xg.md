# FUZZY PARAPHRASES IN LEARNING WORD REPRESENTATIONS WITH A CORPUS AND A LEXICON

Yuanzhi Ke & Masafumi Hagiwara

Department of Information and Computer Science

Keio University

Hiyoshi 3-14-1, Kohokuku, Yokohama City, Kanagawa, Japan

{enshi, hagiwara}@soft.ics.keio.ac.jp

# ABSTRACT

There is a not carefully addressed issue in the previous works using lexicons or ontologies to train or improve distributed word representations: For polysemous words and utterances changing meaning in different contexts, their paraphrases or related entities in a lexicon or an ontology are unreliable and sometimes deteriorate the learning of word representations. Thus, we propose an approach to address the problem. We consider each paraphrase of a word in a lexicon not fully a paraphrase, but a fuzzy member (fuzzy paraphrase) in the paraphrase set whose membership (i.e., degree of truth) depends on the contexts. Then we propose an efficient method to use the fuzzy paraphrases to learn word embeddings. We approximately estimate the local membership of paraphrases, and train word embeddings using a lexicon jointly by replacing the words in the contexts with their paraphrases randomly subjected to the membership of each paraphrase. The experimental results show that our method is efficient, overcomes the weakness of the previous related works in extracting semantic information and outperforms the previous works of learning word representations using lexicons.

# 1 INTRODUCTION

There have been many works and models to estimate the distributed representations of words, i.e. the word embeddings for a corpus (Bengio et al., 2003; Mnih & Hinton, 2007; 2009; Collobert et al., 2011; Huang et al., 2012; Mikolov, 2012; Mikolov et al., 2013b;a;c; Pennington et al., 2014; Bojanowski et al., 2016). Benefiting from the works, high quality word embeddings can be estimated efficiently nowadays.

Word embeddings are reported useful and improve the performance of the machine learning algorithms for many natural language processing tasks such as name entity recognition and chunking (Turian et al., 2010), text classification (Socher et al., 2012; Le & Mikolov, 2014; Kim, 2014; Joulin et al., 2016), topic extraction (Das et al., 2015; Li et al., 2016), and machine translation (Zaremba et al., 2014; Sutskever et al., 2014).

Nevertheless, there is still room for improvement. For example, for the fine-grained sentiment analysis tasks such as predicating the number of stars of a review, the reported accuracy is much lower than the other text classification tasks (Joulin et al., 2016). It indicates the needs of word representations that embed the semantic information more efficiently.

Bojanowski et al. (2016) attempt to improve word embeddings by involving character level information. There is a big improvement on syntactic questions in the word analogical reasoning task introduced by Mikolov et al. (2013a). However, the accuracy for the semantic part is not improved in the reported results.

Some works (Yu & Dredze, 2014; Xu et al., 2014; Faruqui et al., 2015; Bollegala et al., 2016) try to estimate better word embeddings by using a lexicon or an ontology. The idea is simple: because a lexicon or an ontology contains well-defined relations about words, word embeddings of high quality can be learned from it, or we can refine trained word embeddings using the lexicon or the ontology.

However, an issue is not well addressed in the previous works using lexicons to learn word embeddings: For a polysemous word or utterance, its paraphrase in a lexicon or an ontology is not always its paraphrase in different contexts. For example, we can replace the word "Earth" in the sentence "Earth goes around the sun" with its paraphrase "Terra", however the same word "Earth" in the sentence "I fill the hole with earth" cannot be replaced with "Terra". Henceforth, the lexicon or the ontology is sometimes unreliable and deteriorates the learning of word embeddings for the polysemous words and utterances.

In this paper, we propose a method to learn word embeddings using both a corpus and a lexicon that is able to alleviate the bad effect of polysemy, by estimating the degree of truth of each paraphrase in the lexicon. Our method for estimating is simple, efficient and easy to be combined with the previous learning algorithms on the basis of co-occurrences of words. The experimental results show that our method is efficient and outperforms the previous works.

# 2 RELATED WORKS

# 2.1 WORKS ON LEARNING WORD EMBEDDINGS FOR A CORPUS

The first approaches learning word embeddings use n-gram model (Bengio et al., 2003; Collobert et al., 2011; Huang et al., 2012) and recurrent neural networks (Mikolov, 2012). Recently, more efficient methods like continuous bag-of-words (CBOW) model and skip-gram (SG) model (Mikolov et al., 2013b;a), also called word2vec, provide a more efficient way to learn word embeddings based on the local co-occurrences of words in a corpus. Continuous bag-of-words tries to maximize the log probability of a word given its context, while skip-gram tries to maximize the log probability of the words in the context given a word. Negative sampling is an efficient algorithm to train them, that approximately maximizes the log probability for the targets by doing a logistic regression to discriminate the target word from noise randomly drawn from a noise distribution.

Bojanowski et al. (2016) extend word2vec using character-level information. They achieve a considerable improvement for rare words and the syntactic part of the word analogical reasoning task (Mikolov et al., 2013a). However, they fail to improve the performance for the semantic part of the task.

Other works attempt to train word embeddings via global information (Huang et al., 2012; Pennington et al., 2014) and report improvement than the other works that use only local information. However the global information is still limited by the corpus. Some other works follow another approach that uses a lexicon or an ontology to improve the word embeddings.

# 2.2 WORKS ON LEARNING WORD EMBEDDINGS USING LEXICS

The models proposed by Yu & Dredze (2014), and Bollegala et al. (2016) jointly learn word embeddings from a corpus and a semantic lexicon. The method proposed by Yu & Dredze (2014) called jointRCM improves word representations by maximizing the similarity of the word representations of the paraphrase pairs in the lexicon jointly with word2vec. The works by Bollegala et al. (2016) also improve the word embeddings by minimizing the distance of the word representations of related words. But they use not only the synonyms, but also the global information in corpus and other relationships such as antonyms and hyphenyms.

Xu et al. (2014) propose models called R-Net and C-Net. R-Net, for a triplet of words (head, relation, tail) in a knowledge graph, minimizes the distance of the embedding vector of the tail word, from the sum of the vectors of the head word and the relation. On the other hand, C-Net makes a word less similar than the other words share the same category if the size of the category is large. They jointly train the R-Net and C-Net with skip-gram.

Faruqui et al. (2015) concentrate on refining pre-trained word embeddings using semantic lexicons. However, as also pointed out by Bollegala et al. (2016) as incompatibilities between the corpus and the lexicon, some features extracted from the corpus that are not contained in the corpus, such as those from idioms or new words not contained in the lexicon, are improperly removed.

# 2.3 A NOT WELL ADDRESSED TRAP IN LEARNING WORD REPRESENTATIONS USING LEXICONS

In most of the previous works using lexicons to learn word representations, weight coefficients are used to control the input from the lexicon. These coefficients are manually optimized with a separated dataset. However, it cannot address the problem that the reliability of the paraphrases in a lexicon depends on different words and different contexts because a word or an utterance can have several meanings. For example, though "Terra" is the paraphrase of "Earth" in "Earth goes around the sun", but obviously not its paraphrase in "I fill the hole with earth".

# 3 THE PROPOSED METHOD

# 3.1 LEARNING WORD EMBEDDINGS WITH FUZZY PARAPHRASES

Our method is based on two ideas. First, if the meanings of word  $j$  and word  $k$  are ideally the same, they can replace each other in a text without changing the meaning and all the other implicit features of the text. Henceforth, we presume that we can learn word embeddings using both a corpus and a lexicon by learning the original texts and those that some words are replaced with their paraphrases at the same time in the ideal case: When we train word embeddings by predicating a word  $i$  by the context containing word  $j$  like word2vec (Mikolov et al., 2013b;a), the probability of word  $i$  keeps unchanged in that case even though word  $j$  is replaced by word  $k$ .

Secondly, as described in section 1 and 2, the paraphrases of a word in a lexicon are not always the paraphrases in a certain text but depend on the contexts because of polysemous words and utterances. Henceforth, if we simply consider all the paraphrases of a word in the lexicon fully the paraphrases for the whole corpus, they deteriorate the word embeddings of some words and texts. To avoid that, we consider each paraphrase of a word in the lexicon not fully included in the paraphrase set, but a fuzzy member with a grade of membership (i.e. a degree of truth). Then we reject some paraphrases for some contexts subjected to their membership.

![](images/f51439479b8f5797e664a50ab0efbad703d71b53ebfc98ca6324583701d5c761.jpg)  
Figure 1: Architecture of the proposed model.

For a text  $T$ , denote  $w_{i}$  the  $i$ th word in  $T$ ,  $c$  the context window,  $w_{j}$  a word in the context window,  $L_{w_{j}}$  the paraphrase set of  $w_{j}$  in the lexicon  $L$ ,  $w_{k}$  the  $k$ th fuzzy paraphrase in  $L_{w_{j}}$ , and  $x_{jk}$  the membership of  $w_{k}$  for  $w_{j}$ , based on the CBOW model (Mikolov et al., 2013a) and the two ideas, we

propose a model called continuous bag-of-fuzzy-paraphrases (CBOFP) to train word embeddings using both a corpus and a lexicon, by maximizing not only the probability of a word for a given context, but also the probability after some of the words in the context are replaced by their paraphrases randomly subjected to a function of the membership of each paraphrase:

$$
\sum_ {w _ {i} \in T} ^ {T} \sum_ {(i - c) \leq j \leq (i + c)} \left[ \log p \left(w _ {i} \mid w _ {j}\right) + \sum_ {w _ {k} \in L _ {w _ {j}}} ^ {L _ {w _ {j}}} f \left(x _ {j k}\right) \log p \left(w _ {i} \mid w _ {k}\right) \right] \tag {1}
$$

The function  $f(x_{jk})$  of the membership  $x_{jk}$  returns 1 or 0 for different paraphrases of different contexts and reduces the probabilities of the bad replacements that deteriorate the word embeddings by returning 0 more for the paraphrases that have lower grades of membership. The model can be considered as a revised CBOW model with an additional layer whose output is weighted by  $f(x_{jk})$  as shown in Figure 1.

# 3.2 MEMBERSHIP ESTIMATION

If we want the control function  $f(x_{jk})$  to reject bad replacements perfectly,  $f(x_{jk})$  or the membership  $x_{jk}$  should consider all of the contexts because the similarity of the paraphrases depends on not only themselves but also the other contexts. However, it is not easy to train such a function.

Looking for a control function that is easy to train, we notice that if two words are more often to be translated to the same word in another language, the replacement of them are less likely to change the meaning of the original sentence. Thus, we use a function of the bilingual similarity (denoted as  $S_{jk}$ ) as the membership function without considering the other contexts:

$$
x _ {j k} = g \left(S _ {j k}\right) \tag {2}
$$

There have been works about calculating the similarity of words using such bilingual information and a lexicon called the paraphrase database (PPDB) provides scores of the similarity of paraphrases (Ganitkevitch et al., 2013; Pavlick et al., 2015b;a) on the basis of bilingual features. We scale the similarity score of the paraphrase  $w_{k}$  to [0, 1] in PPDB2.0 as the membership, and draw the values of  $f(x_{jk})$  from a Bernoulli distribution subjected to the membership calculated in this way. Denote  $S_{jk}$  the similarity score of word  $w_{j}$  and  $w_{k}$  in PPDB2.0, the value of  $f(x_{jk})$  is drawn from the Bernoulli distribution:

$$
f \left(x _ {j k}\right) \sim \text {B e r n o u l l i} \left(x _ {j k}\right) \tag {3}
$$

$$
x _ {j k} = \frac {S _ {j k}}{\max  _ {j \in T , k \in L} S _ {j k}} \tag {4}
$$

We find the control function defined above is efficient in the experiments as described later in section 4.

# 3.3 TRAINING

Hence we do not need to train  $f(x_{jk})$  using the method described above. The model can be trained by negative sampling proposed by Mikolov et al. (2013b): For word  $w_{O}$  and a word  $w_{I}$  in its context, denote  $A_{I}$  as the set of the paraphrases for  $w_{I}$  accepted by  $f(x_{jk})$ , we maximize  $\log p(w_{O}|w_{I})$  by distinguishing the noise words from a noise distribution  $P_{n}(w)$  from  $w_{O}$  and its accepted paraphrases in  $A_{I}$  by logistic regression:

$$
\log p \left(w _ {O} \mid w _ {I}\right) = \log \sigma \left(v _ {w _ {O}} ^ {\prime \mathrm {T}} v _ {w _ {I}}\right) + \sum_ {i = 1} ^ {n} E _ {w _ {i}} \sim P _ {n} (w) [ \log \sigma \left(- v _ {w _ {i}} ^ {\prime \mathrm {T}} v _ {w _ {I}}\right), w _ {i} \neq w _ {O}, w _ {i} \notin A _ {I} \tag {5}
$$

Here,  $n$  is the number of total negative samples.  $\sigma(x)$  is a sigmoid function,  $\sigma(x) = 1 / (1 + e^{-x})$ .

Table 1: Different types of relationships of paraphrases in PPDB2.0(Pavlick et al., 2015b;a)  

<table><tr><td>Relationship</td><td>Description</td></tr><tr><td>Equivalence</td><td>X is the same as Y</td></tr><tr><td>Forward Entailment</td><td>X is more specific than/is a type of Y</td></tr><tr><td>Reverse Entailment</td><td>X is more general than/ encompasses Y</td></tr><tr><td>Exclusion</td><td>X is the opposite of Y / X is mutually exclusive with Y</td></tr><tr><td>OtherRelated</td><td>X is related in some other way to Y</td></tr><tr><td>Independent</td><td>X is not related to Y</td></tr></table>

# 3.4 DIFFERENT TYPES OF PARAPHRASES AND THE PARAPHRASE SET FOR EACH WORD

In PPDB2.0, there are 6 relationships for paraphrases on the basis of the thesis of MacCartney (2009). For word  $X$  and word  $Y$ , the different relationships of them defined in PPDB2.0 are shown in Table 1. We see that they are far more than we need as some of them are not the conventional "paraphrases" that can replace each other. Only the paraphrases of equivalence, forward entailment and reverse entailment are used in our method. For each word in the vocabulary, the paraphrases equal to it or entailed by it are put into its paraphrase set for learning. For example, denote each paraphrase in PPDB2.0 as (headword, tailword, relationship), for word  $A$ ,  $B$ ,  $C$ ,  $D$ ,  $E$ , if there are paraphrases ( $A$ ,  $B$ , Equivalence), ( $A$ ,  $C$ , ForwardEntailment), ( $D$ ,  $A$ , ReverseEntailment), ( $A$ ,  $E$ , Independent), the paraphrase set for  $A$  is ( $B$ ,  $C$ ,  $D$ ), and  $E$  is discarded.

# 4 EXPERIMENTS AND RESULTS

# 4.1 THE CORPUS, LEXICON AND PARAMETERS

In the experiments, we used enwiki9 as the corpus to train our model. It contains the first one billion bytes in the English Wikipedia. After removing the meta-data, tags, hyperlinks, references, URL encoded characters and converting uppercase letters, spaces and spell digits, the corpus contains 123,353,508 tokens. Among them, there are 218,317 different words.

PPDB2.0 (Pavlick et al., 2015b;a) is used as the lexicon. It contains more than 100 million paraphrases and 26 thousand manually rated phrase pairs. Only the paraphrase pairs whose relationships are equivalence, forward entailment, or reverse entailment are used for our method in the experiments as described in 3.4.

200-dimension word embeddings are trained using our method for the experiments. The context window is set to 8, the number of negative samples is set to 25, the total number of iterations is set to 15 for training. We made an implementation of the proposed method learning enwiki9 and using PPDB2.0 as the lexicon available online<sup>2</sup>.

# 4.2 BASELINES

As baselines to compare with our proposed method, we use word2vec (Mikolov et al., 2013b;a) (Marked as CBOW and SG, for continuous bag-of-word and skip-gram, respectively), word2vec enriched with subword information (Bojanowski et al., 2016) (Marked as Enriched CBOW and Enriched SG), GloVe (Pennington et al., 2014), which are widely used to extract word embeddings from a corpus. We also compare our method with the other works using a lexicon to improve word embeddings, which are jointRCM (Yu & Dredze, 2014), jointReps (Bollegala et al., 2016), RC-Net (Xu et al., 2014), and the method to retrofit pre-trained word embeddings using a lexicon (Marked as Retro) (Faruqui et al., 2015).

Table 2: Comparison against the works learning word embedding for a corpus  

<table><tr><td></td><td>Semantic [%]</td><td>Syntactic [%]</td><td>Total[%]</td></tr><tr><td>Ours</td><td>73.29</td><td>59.44</td><td>65.85</td></tr><tr><td>CBOW</td><td>72.65</td><td>59.25</td><td>65.33</td></tr><tr><td>SG</td><td>72.26</td><td>55.37</td><td>63.04</td></tr><tr><td>Enriched CBOW</td><td>33.08</td><td>75.39</td><td>56.19</td></tr><tr><td>Enriched SG</td><td>61.66</td><td>64.48</td><td>63.20</td></tr><tr><td>GloVe</td><td>66.35</td><td>43.46</td><td>53.80</td></tr></table>

We used the public online available source code of word2vec $^3$ , word2vec enriched with subword information $^4$ , GloVe $^5$ , jointRCM $^6$ , jointReps $^7$ , and Retro $^8$  to build them for the experiments. But for jointRCM and jointReps, the results in our experiments are one percent the number of the reported results in the papers. It is unreasonably low, even though the corpus in our experiments is smaller than those used in their experiments. Thus we use the reported results in the papers to compare with our method. The reported results of jointRCM are achieved using the New York Time 1994-97 subset from Gigaword v5.0 (Parker et al., 2011) containing 518,103,942 tokens. The reported results of jointReps are achieved with ukWaC $^9$  containing 2 billion tokens. For RC-Net, there are no publicly available implementations unfortunately. We report the published results in their paper that are also achieved with enwiki9 to compare with ours. For the other baselines, 200-dimension word embeddings are trained using enwiki9. The context window is set to 8. For word2vec and that enriched with subword information, the number of negative samples is set to 25. The word embeddings trained by CBOW and SG are both used for Retro respectively, and marked as Retro (CBOW) and Retro (SG).

# 4.3 WORD ANALOGICAL REASONING TASK

The word analogical reasoning task is introduced by Mikolov et al. (2013a). Given a quaternion of words  $(w_{A}, w_{B}, w_{C}, w_{D})$  that  $w_{A}$  and  $w_{B}$  have the similar relationship with that of  $w_{C}$  and  $w_{D}$ , the objective is to predict  $w_{D}$  on the basis of  $w_{A}, w_{B}$  and  $w_{C}$ . Given the word embedding  $v_{A}, v_{B}$  and  $v_{C}$  for  $w_{A}, w_{B}$  and  $w_{C}$ , it can be solved by finding the word whose word embedding is the closest to  $v_{B} - v_{A} + v_{C}$ . The dataset is separated into two parts: the semantic part and the syntactic part. The semantic part is about analogical reasoning via semantic relationships, such as predicting the capital of a country. The syntactic part is about syntactic relationships, such as predicting the adverb form for an adjective.

In Table 2, we compare our method with the works using only corpus to train word embeddings. Our method gets the best overall accuracy and the best for the semantic part. For the syntactic part, our method fails to outperform the word2vec enriched with character-level subword information that is reported powerful for the syntactic part.

In Table 3, we compare our method with the previous works using the lexicon to improve the word embeddings. The numbers in the brackets are the differences from the accuracies achieved by the models they base on. For our method and jointRCM, it is CBOw. For jointReps, it is GloVe. For RC-Net, it is SG. For Retro, we report the results retrofitting the word embeddings trained by CBOw and SG, respectively.

Table 3: Comparison against the previous works learning word embedding using a lexicon  

<table><tr><td></td><td>Semantic [%]</td><td>Syntactic [%]</td><td>Total[%]</td></tr><tr><td>Ours</td><td>73.29 (+0.64)</td><td>59.44 (+0.19)</td><td>65.85 (+0.52)</td></tr><tr><td>JointRCM</td><td>-</td><td>29.9 (-30)</td><td>-</td></tr><tr><td>JointReps</td><td>61.46 (-4.89)</td><td>69.33 (+25.87)</td><td>65.76 (+11.96)</td></tr><tr><td>RC-Net</td><td>34.36 (-37.9)</td><td>44.42 (-10.95)</td><td>-</td></tr><tr><td>Retro (CBOW)</td><td>53.88 (-18.77)</td><td>61.31 (+2.06)</td><td>57.94 (-7.39)</td></tr><tr><td>Retro (SG)</td><td>50.66 (-21.6)</td><td>59.78 (+4.41)</td><td>55.64 (-7.4)</td></tr></table>

Table 4: The accuracy of our method and the original CBOW (Mikolov et al., 2013a) in word analogical reasoning task under different corpus size  

<table><tr><td></td><td></td><td>Ours</td><td>CBOW</td><td>Difference</td></tr><tr><td rowspan="3">Text8 (17M Tokens)</td><td>Semantic[%]</td><td>46.35</td><td>46.72</td><td>-0.37</td></tr><tr><td>Syntactic[%]</td><td>42.13</td><td>41.90</td><td>+0.23</td></tr><tr><td>Total[%]</td><td>43.88</td><td>43.91</td><td>-0.03</td></tr><tr><td rowspan="3">Enwiki9 (123M Tokens)</td><td>Semantic[%]</td><td>73.29</td><td>72.65</td><td>+0.64</td></tr><tr><td>Syntactic[%]</td><td>59.44</td><td>59.25</td><td>+0.19</td></tr><tr><td>Total[%]</td><td>65.72</td><td>65.33</td><td>+0.52</td></tr></table>

We see that while the other works perform worse in the semantic part than the model they base on, ours outperform CBOw and outperform the other works in the semantic part. Benefited from alleviating the bad influence of polysemous words and utterances, our method successfully improves the word embeddings in representing semantic information using a lexicon while the other works fail to achieve it. Our method also achieves the best overall accuracy. But for syntactic part, the result of our method using enwiki9 is not as good as the reported result of jointReps using ukWaC.

# 4.4 EFFECTS OF THE SIZE OF THE CORPUS

To see how the size of the corpus affects the performance of our method, we also used a smaller corpus called Text8<sup>10</sup> to learn word embeddings using our methods and then run the word analogical reasoning task using the trained word embeddings. Text8 contains the words in the first 100 million bytes of English Wikipedia. There are 16,718,843 tokens in it and 71,291 different words among them.

We compare the difference of the results by our method and CBOW using the different corpora in Table 4. We see that our method does not achieve obvious improvement over CBOW for text8, but outperforms CBOW for enwiki9. It indicates that our method is weak at small corpus. It is because we use a probabilistic method that requires plenty of samples. By increasing the size of the corpus, our method achieves more improvement.

Table 5: Comparison of the learning speed  

<table><tr><td></td><td>Ours</td><td>CBOW</td><td>Enriched CBOW</td><td>JointRCM</td></tr><tr><td>Time Cost</td><td>3m13s</td><td>3m6s</td><td>18m</td><td>-</td></tr></table>

# 4.5 THE LEARNING SPEED

We compare the learning speed on our machine of our method training 200-dimension word embeddings on text8 against CBOw and the other related works on the basis of CBOw in Table 5. 20 threads were used to train the word embeddings for every model. Unfortunately, the public implementation of jointRCM by the original authors<sup>11</sup> fails to run correctly on our machine, and there is no reported learning speed. We see that there is almost no loss for our method in learning speed in comparison with CBOw while the word2vec enriched with subword information is obviously slower.

# 5 CONCLUSION

We figure out an issue that is not paid enough attention to in the previous works using lexicons to improve word embeddings: Because some words and utterances have multiple meanings, a paraphrase of a word in a lexicon may not be a paraphrase actually in a certain context. Then we propose a method to avoid the trap: We treat the lexicon as a fuzzy set, approximately estimate the membership of the paraphrases, and learn word embeddings using both a corpus and a lexicon by replacing the words in the context with the paraphrases randomly subjected to their grades of membership.

By comparison with the previous works in the word analogical reasoning task, it has been shown that our method overcomes the weakness of the previous related works in extracting the semantic features, outperforms the previous works and keeps fast.

The results using corpora in different sizes show that the proposed method works better with a larger corpus but less effectively with a small corpus. We are looking for another robust method to control the replacements of the paraphrases that keeps efficient for small corpora.

# REFERENCES

Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. A neural probabilistic language model. *journal of machine learning research*, 3(Feb):1137-1155, 2003.  
Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. Enriching word vectors with subword information. arXiv preprint arXiv:1607.04606, 2016.  
Danushka Bollegala, Alsuhaibani Mohammed, Takanori Maehara, and Ken-Ichi Kawarabayashi. Joint word representation learning using a corpus and a semantic lexicon. In Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI'16), 2016.  
Ronan Collobert, Jason Weston, Leon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel Kuksa. Natural language processing (almost) from scratch. Journal of Machine Learning Research, 12(Aug):2493-2537, 2011.  
Rajarshi Das, Manzil Zaheer, and Chris Dyer. Gaussian lda for topic models with word embeddings. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Association for Computational Linguistics, 2015.  
Manaal Faruqui, Jesse Dodge, Sujay K. Jauhar, Chris Dyer, Eduard Hovy, and Noah A. Smith. Retrofitting word vectors to semantic lexicons. In Proceedings of NAACL, 2015.

Juri Ganitkevitch, Benjamin Van Durme, and Chris Callison-Burch. PPDB: The paraphrase database. In Proceedings of NAACL-HLT, pp. 758-764, Atlanta, Georgia, June 2013. Association for Computational Linguistics.  
Eric H Huang, Richard Socher, Christopher D Manning, and Andrew Y Ng. Improving word representations via global context and multiple word prototypes. In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics: Long Papers-Volume 1, pp. 873-882. Association for Computational Linguistics, 2012.  
Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. arXiv preprint arXiv:1607.01759, 2016.  
Yoon Kim. Convolutional neural networks for sentence classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014.  
Quoc V Le and Tomas Mikolov. Distributed representations of sentences and documents. In the 31st International Conference on Machine Learning (ICML 2014), volume 14, pp. 1188-1196, 2014.  
Shaohua Li, Tat-Seng Chua, Jun Zhu, and Chunyan Miao. Generative topic embedding: a continuous representation of documents. In the 54th annual meeting of the Association for Computational Linguistics (ACL 2016). Association for Computational Linguistics, 2016.  
Bill MacCartney. NATURAL LANGUAGE INFERENCE. PhD thesis, Stanford University, 2009.  
Tomas Mikolov. Statistical Language Models Based on Neural Networks. PhD thesis, Brno University of Technology, 2012.  
Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. In ICLR Workshop, 2013a.  
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pp. 3111-3119, 2013b.  
Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic regularities in continuous space word representations. In Proceedings of NAACL HLT, volume 13, pp. 746-751, 2013c.  
Andriy Mnih and Geoffrey Hinton. Three new graphical models for statistical language modelling. In Proceedings of the 24th international conference on Machine learning, pp. 641-648. ACM, 2007.  
Andriy Mnih and Geoffrey E Hinton. A scalable hierarchical distributed language model. In Advances in neural information processing systems, pp. 1081-1088, 2009.  
Robert Parker, David Graff, Junbo Kong, Ke Chen, and Kazuaki Maeda. English gigaword fifth edition. Technical report, Linguistic Data Consortium, 2011.  
Ellie Pavlick, Johan Bos, Malvina Nissim, Charley Beller, Benjamin Van Durme, and Chris Callison-Burch. Adding semantics to data-driven paraphrasing. In *Association for Computational Linguistics*, Beijing, China, July 2015a. Association for Computational Linguistics.  
Ellie Pavlick, Pushpendre Rastogi, Juri Ganitkevich, Benjamin Van Durme, and Chris Callison-Burch. Ppdb 2.0: Better paraphrase ranking, fine-grained entailment relations, word embeddings, and style classification. In Association for Computational Linguistics, Beijing, China, July 2015b. Association for Computational Linguistics.  
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pp. 1532-1543, 2014.  
Richard Socher, Brody Huval, Christopher D Manning, and Andrew Y Ng. Semantic compositionality through recursive matrix-vector spaces. In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pp. 1201-1211. Association for Computational Linguistics, 2012.

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems 27 (NIPS 2014), 2014.  
Joseph Turian, Lev Ratinov, and Yoshua Bengio. Word representations: a simple and general method for semi-supervised learning. In Proceedings of the 48th annual meeting of the association for computational linguistics (ACL 2010), pp. 384-394. Association for Computational Linguistics, 2010.  
Chang Xu, Yalong Bai, Jiang Bian, Bin Gao, Gang Wang, Xiaoguang Liu, and Tie-Yan Liu. Rc-net: A general framework for incorporating knowledge into word representations. In Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management, pp. 1219-1228. ACM, 2014.  
Mo Yu and Mark Dredze. Improving lexical embeddings with semantic knowledge. In the 52nd Annual Meeting of the Association for Computational Linguistics (ACL2014), pp. 545-550. Association for Computational Linguistics, 2014.  
Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. Recurrent neural network regularization. CoRR, abs/1409.2329, 2014.