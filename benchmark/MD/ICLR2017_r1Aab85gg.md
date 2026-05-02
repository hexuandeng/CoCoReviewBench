# OFFLINE BILINGUAL WORD VECTORS WITHOUT A DICTIONARY

Samuel L. Smith, David H. P. Turban, Nils Y. Hammerla & Steven Hamblin

babylon health

London, UK

{samuel.smith, nils.hammerla, steven.hamblin}@babylonhealth.com

dt382@cam.ac.uk

# ABSTRACT

Usually bilingual word vectors are trained "online"; whereby both languages are embedded simultaneously in a single space. Mikolov showed that they can also be found "offline"; whereby two pre-trained embeddings are aligned by a linear transformation, using dictionaries compiled from expert knowledge. In this work, we prove that the linear transformation between the two spaces should be orthogonal, and can be found using canonical correlation analysis. We introduce an "inverted softmax" for identifying translation pairs, with which we improve the precision @1 of Mikolov's original mapping from  $34\%$  to  $43\%$ , when translating a test set composed of both common and rare English words into Italian. Furthermore, we can learn the transformation without expert bilingual signal by constructing a "pseudo-dictionary" using either cognate words or a corpus of aligned sentences. Finally, we extend our method to identify the true translations of English sentences from a corpus of 200k Italian sentences with a precision @1 of  $68\%$ .

# 1 INTRODUCTION

Monolingual word vectors embed language in a high-dimensional vector space, such that the similarity of two words is defined by their proximity in this space (Mikolov et al. (2013b)). They enable us to train sophisticated classifiers to interpret free flowing text (Kim (2014)), and have helped improve the performance of generative language models (Mikolov et al. (2010)). However they require independent models to be trained for each language. Crucially, training text obtained in one language cannot improve the performance of classifiers trained in another, unless the text is explicitly translated. Increasing interest is now focused on bilingual vectors, in which words are aligned by their meaning, irrespective of the language of origin. Such vectors may drive improvements in machine translation (Zou et al. (2013)), and enable language-agnostic text classifiers (Klementiev et al. (2012)). Bilingual word vectors can also be of higher quality than vectors trained on monolingual corpora alone (Faruqui & Dyer (2014)).

These bilingual vectors are normally trained "online", whereby both languages are learnt together in a shared space (Lauly et al. (2014); Gouws et al. (2015)). Typically these algorithms will exploit two sources of monolingual text alongside a smaller bilingual corpus of aligned sentences. This bilingual signal is used to provide a regularisation term, which penalises the embeddings if similar words in the two languages do not lie nearby in the vector space. However Mikolov et al. (2013a) showed that bilingual word vectors can also be obtained "offline". Two sets of word vectors in different languages were first obtained independently, and then a linear matrix  $W$  was trained using a dictionary to map word vectors from the "source" language into the "target" language. Remarkably, this simple procedure was able to translate a test set of English words into Spanish with 33% precision.

To develop an intuition for these two approaches, we note that the similarity of two word vectors is defined by their cosine similarity,  $\cos (\theta_{ij}) = y_i^T x_j / |y_i||x_j|$ . The vectors have no intrinsic meaning, it is only the angles between vectors which are meaningful. This is closely analogous to asking a cartographer to draw a map of England with no compass. The map will be correct, but she does not know which direction is north, so the angle of rotation will be random. Two maps drawn by two such cartographers will be identical, except that one will be rotated by an unknown angle with

![](images/f128470ea43c22fefab32504f7bc9a681dfadc850b47984744b3b72ced33e719.jpg)  
Figure 1: A 2D plane through an English-Italian semantic space, before and after applying CCA on the pre-trained word vectors discussed below, using a training dictionary of 5000 translation pairs. The examples above were not used during training, but CCA aligns the translations remarkably well.

respect to the other. There are two ways the cartographers could align their maps. They could draw the maps together, thus ensuring that landmarks are placed nearby on both maps during "training". Or they could draw their maps independently, and then compare the two afterwards; rotating one map with respect to the other until the major cities are aligned. We note that the more similar the intrinsic geometry of the two maps, the more accurately this rotation will align the space.

In this work, we prove that a self-consistent linear transformation between two vector spaces should be orthogonal. Intuitively the transformation is a rotation, which aligns the underlying semantic space of both languages. This rotation can be found using canonical correlation analysis (CCA). If the word vectors are normalised CCA maximises the cosine similarity of translation pairs in the training dictionary. We build on the work of Dinu et al. (2014), introducing an "inverted softmax" to identify translation pairs. Using the same word vectors, training dictionary and test set provided by Dinu, we improve the precision of Mikolov's mapping from  $34\%$  to  $43\%$  when translating from English to Italian, and from  $25\%$  to  $37\%$  when translating from Italian to English. The semantic space obtained by this procedure is illustrated in figure 1. We can exploit the robustness of CCA, by discarding the training dictionary and forming a pseudo-dictionary from the word strings which appear identically in both languages ("cognate words"). Remarkably, this pseudo dictionary achieves translation precision of  $40\%$  and  $34\%$  respectively on the same test set, despite the absence of an expert bilingual signal. Finally, we form simple sentence vectors by summing and normalising over word vectors, and we obtain bilingual sentence vectors by applying CCA to a second pseudo-dictionary formed from a bilingual corpus of aligned text. The transformation obtained aligns the underlying word vectors, achieving a translation precision of  $43\%$  and  $38\%$ , en-par with the expert dictionary above. Remarkably, we can also use these sentence vectors to retrieve the correct translations of an English sentence from a bag of 200k Italian candidate sentences with  $68\%$  precision.

# 2 OFFLINE BILINGUAL LANGUAGE VECTORS

# 2.1 PREVIOUS WORK

Offline word vectors were first proposed by Mikolov et al. (2013a). They obtained a small dictionary of paired words from Google Translate, whose word vectors we denote  $\{y_i, x_i\}_{i=1}^n$ . Next, they applied  $W$  to the source language and used stochastic gradient descent to minimise the squared reconstruction error,

$$
\min  _ {W} \sum_ {i = 1} ^ {n} \left| \left| y _ {i} - W x _ {i} \right| \right| ^ {2}. \tag {1}
$$

After training, any word vector in the source language can be mapped to the target by calculating  $y_{e} = Wx$ . The similarity between a source vector  $x$  and a target vector  $y_{t}$  can then be evaluated by their cosine similarity  $\cos(\theta_{te}) = y_t^T y_e / |y_t||y_e|$ . Astonishingly, this simple procedure achieved 33% accuracy when translating unseen words from English into Spanish, using a training dictionary of

5k common English words and their Spanish translations, and word vectors trained using word2vec on the WMT11 text datasets. Translations were found by a simple nearest neighbour procedure.

We note that the cost function above is solved by the method of least squares, as realised by Dinu et al. (2014). They did not modify this cost function, but proposed an adapted method of retrieving translation pairs which was more accurate when translating words from English to Italian. Faruqui & Dyer (2014) obtained bilingual word vectors using CCA. They did not attempt any translation tasks, but showed that the combination of CCA and dimensionality reduction improved the performance of monolingual vectors on standard evaluation tasks. They hypothesise that this procedure projects out directions in the vector space which describe language specific irregularities, generating vectors which describe universal "meaning". More recently, Xing et al. (2015) argued that Mikolov's linear matrix should be orthogonal, and introduced an approximate procedure composed of gradient descent updates and repeated applications of the singular value decomposition (SVD). CCA has been extended to map 59 languages into a single shared space (Ammar et al. (2016)), and non-linear "deep CCA" has been introduced (Lu et al. (2015)). The goal of this work is to unify and enhance these approaches, in order to demonstrate that offline methods can be trained without a training dictionary using either cognate words or a bilingual corpus of aligned text.

# 2.2 THE SIMILARITY MATRIX AND THE ORTHOGONAL TRANSFORM

To prove that a self-consistent linear mapping between semantic spaces must be orthogonal, we form the similarity matrix,  $S = YW X^T$ .  $X$  and  $Y$  are the word vector matrices in each language, where each row in the matrix contains a single word vector, denoted by lower case  $x$  and  $y$ . Consequently the matrix element,

$$
\begin{array}{l} S _ {i j} = y _ {i} ^ {T} W x _ {j} (2) \\ = y _ {i} \cdot \left(W x _ {j}\right), (3) \\ \end{array}
$$

evaluates the similarity between the  $j^{th}$  source word and the  $i^{th}$  target word. The matrix  $W$  maps the source language into the target language. The largest value in a column of the similarity matrix gives the most similar target word to a particular source word, while the largest value in a row gives the most similar source word to a given target word. However we could also form a second similarity matrix  $S' = XQY^T$ , such that the matrix  $Q$  maps the target language back into the source. The matrix element,

$$
\begin{array}{l} S _ {j i} ^ {\prime} = x _ {j} ^ {T} Q y _ {i} (4) \\ = x _ {j} \cdot \left(Q y _ {i}\right), (5) \\ \end{array}
$$

also evaluates the similarity between the  $\mathbf{j}^{th}$  source word and the  $\mathbf{i}^{th}$  target word. To be self-consistent, we require  $S' = S^T$ . However  $S^T = X W^T Y^T$ , and therefore the matrix  $Q = W^T$ . If  $W$  maps the source language into the target, then  $W^T$  maps the target language back into the source.

When we map a source word into the target language, we should be able to map it back into the source language and re-obtain the original vector.  $x \sim W^T y$  and  $y \sim W x$  and thus  $x \sim W^T W x$ . This expression should hold for any word vector  $x$  and thus we conclude that the transformation  $W$  should be an orthogonal matrix  $O$  satisfying  $O^T O = I$ , where  $I$  denotes the identity matrix. Orthogonal transformations preserve the vector norm. If we normalise  $X$  and  $Y$ , then  $OX$  and  $OT Y$  are also normalised. Consequently the matrix element,  $S_{ij} = y_i \cdot (Ox_j) = |y_i||Ox_j| \cos(\theta_{ij}) = \cos(\theta_{ij})$ . The similarity matrix  $S = YOX^T$  directly computes the cosine similarity between all possible pairs of source and target words under the orthogonal transformation  $O$ .

# 2.3 THE ORTHOGONAL PROCRUSTES PROBLEM AND CCA

We now seek to infer the orthogonal transformation  $O$  from a dictionary  $\{y_i, x_i\}_{i=1}^n$  of source words and their translations in the target language. Since we predict the similarity of two vectors by evaluating  $S_{ij} = \cos(\theta_{ij})$ , we ought to learn the transformation by maximising the cosine similarity of translation pairs in the dictionary,

$$
\max  _ {O} \sum_ {i = 1} ^ {n} y _ {i} ^ {T} O x _ {i}, \text {s u b j e c t} O ^ {T} O = I. \tag {6}
$$

There is not an intuitive analytic solution to this problem, but an analytic solution does exist to the closely related "orthogonal Procrustes problem", which minimises the squared reconstruction error subject to an orthogonal constraint (Schönemann (1966)),

$$
\min  _ {O} \sum_ {i = 1} ^ {n} \left\| y _ {i} - O x _ {i} \right\| ^ {2}, \text {s u b j e c t} O ^ {T} O = I. \tag {7}
$$

The solution proceeds as follows. We form two ordered matrices  $Y_{D}$  and  $X_{D}$  from the dictionary, such that the  $i^{th}$  row of  $\{X_{D},Y_{D}\}$  corresponds to the source and target language word vectors of the  $i^{th}$  pair in the dictionary. We then compute the SVD of

$$
M = Y _ {D} ^ {T} X _ {D} = U \Sigma V ^ {T}. \tag {8}
$$

This step is highly efficient, since  $M$  is a square matrix with the same dimensionality as the word vectors, not the length of dictionary.  $U$  and  $V$  are composed of columns of orthonormal vectors, while  $\Sigma$  is a diagonal matrix containing the singular values. The Procrustes problem is solved by the orthogonal matrix  $O = UV^T$ . Thus the optimised similarity matrix,

$$
S = Y U V ^ {T} X ^ {T}. \tag {9}
$$

$$
S _ {i j} = y _ {i} ^ {T} U V ^ {T} x _ {j} \tag {10}
$$

$$
= \left(U ^ {T} y _ {i}\right) \cdot \left(V ^ {T} x _ {j}\right). \tag {11}
$$

We map both the source and target languages into a single shared vector space, by applying the transformation matrix  $V^T$  to the source language and  $U^T$  to the target language. This procedure is an example of CCA (Hardoon et al. (2004)). It minimises the squared reconstruction error, subject to an orthogonal constraint. However both  $X$  and  $Y$  are normalised, while  $O$  preserves the vector norm. We note that,

$$
\left\| y _ {i} - O x _ {i} \right\| ^ {2} = \left| y _ {i} \right| ^ {2} + \left| x _ {i} \right| ^ {2} - 2 y _ {i} ^ {T} O x _ {i} \tag {12}
$$

$$
\propto A - y _ {i} ^ {T} O x _ {i}. \tag {13}
$$

$A$  is a constant, and so the cost functions given in equations 6 and 7 are equivalent. Thus when  $X_{D}$  and  $Y_{D}$  are normalised, CCA will maximise the mean cosine similarity between pairs in the dictionary (subject to the orthogonal constraint). Therefore CCA provides a numerically exact solution to the cost function proposed by Xing et al. (2015), just as the method of least squares provides a numerically exact solution to the cost function proposed by Mikolov et al. (2013a).

The solution to the orthogonal Procrustes problem does not use the matrix of singular values  $\Sigma$ . However that does not mean that these singular values do not carry relevant information. All of the singular values are positive, and each singular value  $s_i$  is uniquely associated to a pair of normalised vectors  $u_i$  and  $v_i$  from the matrices  $U$  and  $V$ . Standard implementations of the SVD return the singular values in descending order. The larger the singular value, the more rapidly the mean cosine similarity of the dictionary decreases if the corresponding vectors are distorted. Thus the vectors corresponding to large singular values carry significant information which aids translation between the two languages, while vectors corresponding to small singular values carry irrelevant information. We can perform dimensionality reduction by neglecting the vectors  $\{u_i, v_i\}$  which arise from the smallest singular values, and projecting our word vector matrices  $X$  and  $Y$  into a lower dimensional space within which the translation pairs are highly correlated. This is trivial to implement by simply dropping the final few rows of  $U^T$  and  $V^T$ , and we will show below that it leads to a small improvement in the translation performance.

# 2.4 THE INVERTED SOFTMAX

Mikolov et al. (2013a) predicted the translation of a source word  $x_{j}$  by finding the target word  $y_{i}$  closest to  $Wx_{j}$ . In our formalism, this corresponds to finding the largest entry in the  $j^{th}$  column of the similarity matrix. To estimate our confidence in this prediction, we could form the softmax,

$$
P _ {j \rightarrow i} = \frac {e ^ {\beta S _ {i j}}}{\sum_ {m} e ^ {\beta S _ {m j}}}. \tag {14}
$$

To learn the "inverse temperature"  $\beta$ , we maximise the log probability over the training dictionary,

$$
\max  _ {\beta} \sum_ {\text {p a i r s} i j} \ln \left(P _ {j \rightarrow i}\right). \tag {15}
$$

This sum should be performed only over valid translation pairs. Dinu et al. (2014) demonstrated that nearest neighbour retrieval is flawed, since it suffers from the presence of "hubs". Hubs are words which appear as the nearest neighbour target word to many different source words, driving down the translation performance. We propose that the hubness problem is mitigated by inverting the softmax, and normalising the probability over source words rather than target words.

$$
P _ {j \rightarrow i} = \frac {e ^ {\beta S _ {i j}}}{\alpha_ {j} \sum_ {n} e ^ {\beta S _ {i n}}}. \tag {16}
$$

Intuitively, rather than asking whether the source word translates to the candidate target word, we assess the probability that the candidate target word translates back into the source word. We then select the target word which maximises this probability. If the  $i^{th}$  target word is a hub, then the denominator in equation 16 will be large, preventing this target word from being selected. The vector  $\alpha$  ensures normalisation. The sum over  $n$  should run over all source words in the vocabulary. However to reduce the computational cost, we only perform this sum over  $n_s$  sample words, chosen randomly from the vocabulary. Unless explicitly stated,  $n_s = 1500$ .

# 2.5 PSEUDO DICTIONARIES

# 2.5.1 COGNATE WORDS

Our method requires a training dictionary of paired vectors, which is used to infer the orthogonal map  $O$  and the inverse temperature  $\beta$ , and also as a validation set during dimensionality reduction. Typically this dictionary is obtained by translating common source words into the target language using Google Translate, which was constructed using expert human knowledge. However most European languages share a large number of words composed of identical character strings (cognates). Words like "London", "DNA" and "Tortilla". It is probable that cognates across two languages share similar meanings (Simard et al. (1993)). We can extract these cognates and form a "pseudo-dictionary", compiled without any expert bilingual knowledge. Below we show that this pseudo dictionary is sufficient to successfully translate between English and Italian with high precision.

# 2.5.2 ALIGNED SENTENCES

The Europarl corpus is composed of aligned sentences in a number of European languages (Koehn (2005)). Gouws et al. (2015) showed that such corpora could be used alongside monolingual text sources to learn online bilingual vectors. To date, offline bilingual vectors have only been obtained from dictionaries. To learn the orthogonal transformation from aligned sentences, we define the vector  $q$  of a source language sentence by the normalised sum of the word vectors  $x_{i}$  contained in that sentence,

$$
q = \frac {\sum_ {i} x _ {i}}{\left| \sum_ {i} x _ {i} \right|}. \tag {17}
$$

Similarly, the vector  $w$  of a target language sentence is defined by the normalised sum of word vectors  $y_{i}$ . We can now view the aligned text corpus as a dictionary of paired sentences  $\{w_{i}, q_{i}\}$ , from which we can form two dictionary matrices  $W_{D}$  and  $Q_{D}$ . We obtain the orthogonal transformation  $O$  by performing CCA on the matrix  $M = W_{D}^{T} Q_{D}$ , and use this transformation to translate the individual words in the test set.

This simple procedure embeds words and sentences in the same vector space. The sentence embedding can be thought of as the "average word" that the sentence conveys. Intuitively, each aligned sentence pair gives us weak information about a possible word pair in the dictionary. By combining a large number of such sentence pairs, we obtain sufficient information to align the vector spaces and infer the translations of individual words. However, we will go on to show that this orthogonal transformation can be used, not only to retrieve the translations of words between languages, but also to retrieve the translations of sentences between languages with remarkably high accuracy.

Table 1: Translation performance using the expert training dictionary, English into Italian.  

<table><tr><td>Precision</td><td>Mikolov et al.</td><td>Dinu et al.</td><td>CCA</td><td>+ inverted softmax</td><td>+ dimensionality reduction</td></tr><tr><td>@1</td><td>0.338</td><td>0.385</td><td>0.369</td><td>0.417</td><td>0.431</td></tr><tr><td>@5</td><td>0.483</td><td>0.564</td><td>0.527</td><td>0.587</td><td>0.607</td></tr><tr><td>@10</td><td>0.539</td><td>0.639</td><td>0.579</td><td>0.655</td><td>0.664</td></tr></table>

Table 2: Translation performance using the expert training dictionary, Italian into English.  

<table><tr><td>Precision</td><td>Mikolov et al.</td><td>Dinu et al.</td><td>CCA</td><td>+ inverted softmax</td><td>+ dimensionality reduction</td></tr><tr><td>@1</td><td>0.249</td><td>0.246</td><td>0.322</td><td>0.373</td><td>0.380</td></tr><tr><td>@5</td><td>0.410</td><td>0.454</td><td>0.496</td><td>0.577</td><td>0.585</td></tr><tr><td>@10</td><td>0.474</td><td>0.541</td><td>0.557</td><td>0.631</td><td>0.636</td></tr></table>

# 3 EXPERIMENTS

We perform our experiments using the same word vectors, training dictionary and test dictionary provided by Dinu et al. (2014) $^{1}$ . The word vectors were trained using word2vec, and then the 200k most common words in both the English and Italian corpora were extracted. The English word vectors were trained on the WackyPedia/ukWaC and BNC corpora, while the Italian word vectors were trained on the WackyPedia/itWaC corpus. The training dictionary comprises 5k common English words and their Italian translations, while the test set is composed of 1500 English words and their Italian translations. This test set is split into five sets of 300. The first 300 words arise from the most common 5k words in the English corpus, the next 300 from the 5k-20k most common words, followed by bins for the 20k-50k, 50k-100k, and 100k-200k most common words. This enables us to evaluate how word frequency affects the translation performance. Some of the Italian words have both male and female forms, and we follow Dinu in considering either form a valid translation.

We report results using our own methods outlined above, as well as results using the methods proposed by Mikolov and Dinu. We compute results for Mikolov's method by applying the method of least squares as a numerically exact solution to equation 1, and retrieving translation pairs by nearest neighbour retrieval. We compute results for Dinu's method using the source code they provided alongside their manuscript and using 5k "pivot elements".

# 3.1 EXPERIMENTS USING THE EXPERT TRAINING DICTIONARY

In tables 1 and 2 we present the translation performance of our methods when translating the test set between English and Italian, using the expert training dictionary provided by Dinu. We also evaluate Mikolov and Dinu's methods for comparison. All the methods are more accurate when translating from English to Italian. This is unsurprising given that some English words in the test set can translate to either the male or female form of the Italian word. In the third column we evaluate the performance of CCA with nearest neighbour retrieval. This already provides a marked improvement on Mikolov's mapping, especially when translating from Italian into English. In the following two columns we apply first the inverted softmax, and then dimensionality reduction. The hyper-parameters of these procedures were optimised on the training dictionary. Combining both procedures improves the precision @1 to  $43\%$  and  $38\%$  when translating from English to Italian, or Italian to English respectively. These results are a significant improvement on previous work.

In table 3 we present the precision @1 as a function of word frequency. We achieve remarkably high precision when translating common words. This performance drops off for less common words, presumably either because there is insufficient monolingual data to learn high quality word vectors for rare words, or because the linguistic similarities between rare words across languages are less pronounced than the similarities between common words.

Table 3: Translation precision @1 from English to Italian using the expert training dictionary. We achieve  $68\%$  precision on test cases selected from the 5k most common English words in the ukWaC, Wikipedia and BNC corpora. The precision falls for less common words.  

<table><tr><td>Word frequency</td><td>Mikolov et al.</td><td>Dinu et al.</td><td>CCA</td><td>+ inverted softmax</td><td>+ dimensionality reduction</td></tr><tr><td>0-5k</td><td>0.607</td><td>0.650</td><td>0.637</td><td>0.690</td><td>0.690</td></tr><tr><td>5-20k</td><td>0.463</td><td>0.540</td><td>0.510</td><td>0.580</td><td>0.610</td></tr><tr><td>20-50k</td><td>0.280</td><td>0.350</td><td>0.323</td><td>0.380</td><td>0.403</td></tr><tr><td>50-100k</td><td>0.193</td><td>0.217</td><td>0.2</td><td>0.230</td><td>0.253</td></tr><tr><td>100-200k</td><td>0.147</td><td>0.163</td><td>0.173</td><td>0.203</td><td>0.200</td></tr></table>

Table 4: Translation performance using the pseudo dictionary of cognate words.  

<table><tr><td>Precision</td><td>English to Italian</td><td>Italian to English</td></tr><tr><td>@1</td><td>0.399</td><td>0.343</td></tr><tr><td>@5</td><td>0.576</td><td>0.566</td></tr><tr><td>@10</td><td>0.631</td><td>0.624</td></tr></table>

# 3.2 EXPERIMENTS USING COGNATE WORDS

In the preceding section, we reported our performance using an orthogonal transformation learned on an expert training dictionary of 5k common English and Italian words. We now report our performance when we do not use this dictionary, and instead construct a pseudo dictionary from the list of words in the English and Italian vocabularies which are composed of exactly the same character string. Remarkably, 47074 such cognate words appear in both vocabularies. We simply extract the corresponding vectors to form the dictionary matrices  $X_{D}$  and  $Y_{D}$ .

We exhibit our results in table 4, where we evaluate the performance of our method (CCA + inverted softmax + dimensionality reduction), when translating either from English to Italian or from Italian to English. Even when using this pseudo dictionary prepared with no expert bilingual knowledge, we still achieve a mean translation performance @1 of  $40\%$  from English to Italian on our test set.

# 3.3 EXPERIMENTS ON THE EUROPARL CORPUS OF ALIGNED SENTENCES

The English-Italian Europarl corpus comprises 2 million English sentences and their Italian translations, taken from the proceedings of the European parliament (Koehn (2005)). As outlined earlier, we can form simple sentence vectors in the word vector space by summing and normalising over the words contained in a sentence. These sentence vectors can be used in two different tasks. First, we can use the Europarl corpus as a training dictionary, whereby the matrices  $X_{D}$  and  $Y_{D}$  are formed from the sentence vectors of translation pairs. By applying CCA to the first 500k sentences in this "phrase dictionary", we obtain a set of bilingual word vectors from which we can retrieve translations of individual words. We exhibit the translation performance of this approach in table 5. We achieve  $42.8\%$  precision @1 when translating from English into Italian and  $37.5\%$  precision when translating from Italian into English, comparable to the accuracy achieved using the expert word dictionary on the same test set. It is difficult to compare the two approaches, since they require different training data. However our performance appears competitive with Bilbowa, a leading method for learning bilingual vectors online from monolingual corpora and aligned text (Gouws et al. (2015)).

Second, we can apply our orthogonal transformation to retrieve the Italian translation of an English sentence, or vice versa. To achieve this, we hold back the final 200k English and Italian sentences

Table 5: Translation performance, using the Europarl corpus as a phrase dictionary.  

<table><tr><td>Precision</td><td>English to Italian</td><td>Italian to English</td></tr><tr><td>@1</td><td>0.428</td><td>0.375</td></tr><tr><td>@5</td><td>0.589</td><td>0.563</td></tr><tr><td>@10</td><td>0.647</td><td>0.620</td></tr></table>

Table 6: "Translation" precision, when seeking to retrieve the true translation of an English sentence from a bag of 200k Italian sentences, or vice versa, averaged over 5k samples. We first obtain bilingual word vectors, either by applying CCA to the word dictionary provided by Dinu, or by constructing a phrase dictionary from Europarl. We set  $n_s = 12800$  in the inverted softmax.  

<table><tr><td>Precision</td><td>Word dictionary: English to Italian</td><td>Word dictionary: Italian to English</td><td>Phrase dictionary: English to Italian</td><td>Phrase dictionary: Italian to English</td></tr><tr><td>@1</td><td>0.546</td><td>0.429</td><td>0.678</td><td>0.486</td></tr><tr><td>@5</td><td>0.727</td><td>0.622</td><td>0.825</td><td>0.679</td></tr><tr><td>@10</td><td>0.782</td><td>0.692</td><td>0.862</td><td>0.745</td></tr></table>

from our 500k sample of Europarl, and attempt to retrieve the true translation of a given sentence in this test set. We obtain the orthogonal transformation by performing CCA on either the expert word dictionary provided by Dinu, or on the phrase dictionary formed from the first 300k sentences from Europarl. Our results are provided in table 6. Remarkably, given no information except the sentence vectors, we are able to retrieve the correct translation of an English sentence with  $67.8\%$  precision. This is particularly surprising, given that we are using the simplest possible definition of a sentence vector, which has no information about word order or sentence length. It is likely that we could improve on these results if we used higher quality sentence vectors which take these factors into account (Le & Mikolov (2014); Kiros et al. (2015)), although we might lose the ability to simultaneously align the underlying word vector space.

When training this procedure, the inverse temperature  $\beta$  diverged, and the "translation" performance from English to Italian significantly exceeded the performance from Italian to English. This implies that sentence retrieval from Italian to English is best achieved by simple nearest neighbours, and indeed this improved the performance from Italian to English from  $48.6\%$  to  $65.6\%$ , on par with the performance from English to Italian. This suggests that the optimal retrieval approach would be able to tune continuously between the conventional softmax and the inverted softmax.

# 4 SUMMARY

We have proved that the optimal linear transformation between word vector spaces should be orthogonal, and can be obtained using CCA on a dictionary of translation pairs. We can use this orthogonal transformation to obtain bilingual word vectors, from which we can predict the translations of previously unseen words. We introduced a novel "inverted softmax" which significantly increased the accuracy of our predicted translations. Combining CCA with the inverted softmax and dimensionality reduction, we improved the translation precision of Mikolov's original linear mapping from  $34\%$  to  $43\%$ , when translating a test set composed of both common and rare English words into Italian. This was achieved using a training dictionary of 5k English words and their Italian translations. Replacing this training dictionary with a pseudo-dictionary acquired from cognate words, we showed that we still achieved  $40\%$  precision, demonstrating that it is possible to obtain bilingual vector spaces without an expert bilingual signal. There are currently a number of approaches to obtaining offline bilingual word vectors in the literature. Our work shows they can all be unified.

Finally, we defined a set of simple sentence vectors. Using these vectors, we were able to obtain offline bilingual word vectors without a dictionary, using the Europarl corpus of aligned text. We achieved  $43\%$  precision when translating our test set from English into Italian under this approach, comparable to our results using a dictionary of words, and competitive with online approaches which use aligned text as the bilingual signal. We demonstrated that we could also use our sentence vectors to retrieve the true translation of an English sentence from a bag of 200k Italian candidate sentences with  $68\%$  precision, a striking result worthy of further investigation.

# ACKNOWLEDGMENTS

We would like to thank babylon health for supporting this research. We are indebted to Dinu et al. for providing their source code, pre-trained word vectors, and a training and test dictionary of English and Italian words, and to Philipp Koehn for compiling the Europarl corpus.

# REFERENCES

Waleed Ammar, George Mulcaire, Yulia Tsvetkov, Guillaume Lample, Chris Dyer, and Noah A Smith. Massively multilingual word embeddings. arXiv:1602.01925, 2016.  
Georgiana Dinu, Angeliki Lazaridou, and Marco Baroni. Improving zero-shot learning by mitigating the hubness problem. arXiv:1412.6568, 2014.  
Manaal Faruqui and Chris Dyer. Improving vector space word representations using multilingual correlation. In Proceedings of the 2014 conference of the Association for Computational Linguistics. Association for Computational Linguistics, 2014.  
Stephan Gouws, Yoshua Bengio, and Greg Corrado. Bilbowa: Fast bilingual distributed representations without word alignments. In Proceedings of The 32nd International Conference on Machine Learning, pp. 748-756, 2015.  
David R Hardoon, Sandor Szedmak, and John Shawe-Taylor. Canonical correlation analysis: An overview with application to learning methods. Neural computation, 16(12):2639-2664, 2004.  
Yoon Kim. Convolutional neural networks for sentence classification. arXiv:1408.5882, 2014.  
Ryan Kiros, Yukun Zhu, Ruslan R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Skip-thought vectors. In Advances in neural information processing systems, pp. 3294-3302, 2015.  
Alexandre Klementiev, Ivan Titov, and Binod Bhattacharai. Inducing crosslingual distributed representations of words. 2012.  
Philipp Koehn. Europarl: A parallel corpus for statistical machine translation. In MT summit, volume 5, pp. 79-86, 2005.  
Stanislas Lauly, Hugo Larochelle, Mitesh Khapra, Balaraman Ravindran, Vikas C Raykar, and Amrita Saha. An autoencoder approach to learning bilingual word representations. In Advances in Neural Information Processing Systems, pp. 1853-1861, 2014.  
Quoc V Le and Tomas Mikolov. Distributed representations of sentences and documents. In ICML, volume 14, pp. 1188-1196, 2014.  
Ang Lu, Weiran Wang, Mohit Bansal, Kevin Gimpel, and Karen Livescu. Deep multilingual correlation for improved word embeddings. In Proceedings of NAACL, 2015.  
Tomas Mikolov, Martin Karafiat, Lukas Burget, Jan Cernocky, and Sanjeev Khudanpur. Recurrent neural network based language model. In Interspeech, volume 2, pp. 3, 2010.  
Tomas Mikolov, Quoc V Le, and Ilya Sutskever. Exploiting similarities among languages for machine translation. arXiv:1309.4168, 2013a.  
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pp. 3111-3119, 2013b.  
Peter H Schonemann. A generalized solution of the orthogonal procrustes problem. Psychometrika, 31(1), 1966.  
Michel Simard, George F Foster, and Pierre Isabelle. Using cognates to align sentences in bilingual corpora. In Proceedings of the 1993 conference of the Centre for Advanced Studies on Collaborative research: distributed computing, pp. 1071-1082. IBM Press, 1993.  
Chao Xing, Dong Wang, Chao Liu, and Yiye Lin. Normalized word embedding and orthogonal transform for bilingual word translation. In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 1006-1011, 2015.  
Will Y Zou, Richard Socher, Daniel M Cer, and Christopher D Manning. Bilingual word embeddings for phrase-based machine translation. In EMNLP, pp. 1393-1398, 2013.