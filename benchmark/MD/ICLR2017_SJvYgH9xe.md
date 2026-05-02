# AUTOMATIC RULE EXTRACTION FROM LONG SHORT TERM MEMORY NETWORKS

W. James Murdoch *

Department of Statistics

UC Berkeley

Berkeley, CA 94709, USA

jmurdoch@berkeley.edu

Arthur Szlam

Facebook AI Research

New York City, NY, 10003

aszlam@fb.com

# ABSTRACT

Although deep learning models have proved effective at solving problems in natural language processing, the mechanism by which they come to their conclusions is often unclear. As a result, these models are generally treated as black boxes, yielding no insight of the underlying learned patterns. In this paper we consider Long Short Term Memory networks (LSTMs) and, using the recently introduced WikiMovies dataset, we demonstrate a new approach for tracking the importance of a given input to the LSTM for a given output. We then use frequent patterns of important words to automatically construct a simple, rule-based classifier which mimics the output of the fitted LSTM.

# 1 INTRODUCTION

Neural network language models, especially recurrent neural networks (RNN), are now standard tools for natural language processing. Amongst other things, they are used for translation Sutskever et al. (2014), language modelling Jozefowicz et al. (2016), and question answering Hewlett et al. (2016). In particular, the Long Short Term Memory (LSTM) Hochreiter & Schmidhuber (1997) architecture has become a basic building block of neural NLP. Although LSTM's are often the core of state of the art systems, their operation is not well understood. Besides the basic desire from a scientific viewpoint to clarify their workings, it is often the case that it is important to understand why a machine learning algorithm made a particular choice. Moreover, LSTM's can be costly to run in production compared to discrete models with lookup tables and pattern matching.

In this work, we describe a novel method for visualizing the importance of specific inputs for determining the output of an LSTM. In the question-answering domain, we show that simple phrase patterns can be extracted from a trained LSTM such that simply matching the patterns provides equivalent performance.

# 2 RELATED WORK

There are two lines of related work on visualizing LSTMs. First, Hendrik et al. (2016) and Karpathy et al. (2016) analyse the movement of the raw gate activations over a sequence. Karpathy et al. (2016) is able to identify co-ordinates of  $c_t$  that correspond to semantically meaningful attributes such as whether the text is in quotes and how far along the sentence a word is. However, most of the cell co-ordinates are harder to interpret, and in particular, it is often not obvious from their activations which inputs are important for specific outputs.

Another approach that has emerged in the literature Alikaniotis et al. (2016) Denil et al. (2015) Bansal et al. (2016) is for each word in the document, looking at the norm of the derivative of the loss function with respect to the embedding parameters for that word. This bridges the gap between high-dimensional cell state and low-dimensional outputs. These techniques are general- they are applicable to visualizing the importance of sets of input coordinates to output coordinates of any

differentiable function. In this work, we describe techniques that are designed around the structure of LSTM's, and show that they can give better results in that setting.

# 3 LSTMs FOR ANSWER EXTRACTION

We first review LSTMs before introducing a LSTM-based model for doing question-answering from text. As our intent is to understand the underlying mechanisms, rather than optimize for prediction accuracy, we avoid overly complex architectures. This model is a simplified version of the work in Li et al. (2016).

# 3.1 LONG SHORT TERM MEMORY NETWORKS

Over the past few years, LSTMs have become a core component of neural NLP systems. Given a sequence of word embeddings  $x_{1},\ldots ,x_{T}\in \mathbb{R}^{d}$ , an LSTM processes one word at a time, keeping track of cell and state vectors  $\{(c_i,h_i)\}_{i = 1}^T$  which contain information in the sentence up to word  $i$ .  $h_t$  and  $c_{t}$  are computed as a function of  $x_{t},c_{t - 1}$  using the below updates

$$
h _ {t} = o _ {t} \odot \tanh  \left(c _ {t}\right) \tag {1}
$$

$$
c _ {t} = f _ {t} c _ {t - 1} + i _ {t} \tilde {c} _ {t} \tag {2}
$$

$$
f _ {t} = \sigma \left(W _ {f} x _ {t} + V _ {f} h _ {t - 1} + b _ {f}\right) \tag {3}
$$

$$
i _ {t} = \sigma \left(W _ {i} x _ {t} + V _ {i} h _ {t - 1} + b _ {i}\right) \tag {4}
$$

$$
o _ {t} = \sigma \left(W _ {o} x _ {t} + V _ {o} h _ {t - 1} + b _ {o}\right) \tag {5}
$$

$$
\tilde {c} _ {t} = \tanh  \left(W _ {c} x _ {t} + V _ {c} h _ {t - 1} + b _ {c}\right) \tag {6}
$$

As a standard LSTM only considers information to the left of a given word, it is common to also compute an LSTM over the reversed sequence, concatenating the two results. This bidirectional LSTM concatenates output from the forward LSTM  $\overleftarrow{f}$ , which reads the document from  $x_{1}$  to  $x_{T}$  with a backward LSTM  $\overleftarrow{f}$ , which reads the document from  $x_{T}$  to  $x_{1}$ .

$$
\overrightarrow {h _ {t}} = \overrightarrow {L S T M} \left(x _ {t}\right) \tag {7}
$$

$$
\overleftarrow {h _ {t}} = \overleftarrow {L S T M} \left(x _ {t}\right) \tag {8}
$$

$$
h _ {t} = \left[ \overrightarrow {h _ {t}}, \overleftarrow {h _ {t}} \right] = \operatorname {B i L S T M} \left(x _ {t}\right) \tag {9}
$$

# 3.2 CONDITIONING ON QUESTIONS

Given a pair of question  $x_1^q, \ldots, x_N^q$  and document  $x_1^d, \ldots, x_T^d$ , various approaches have been proposed for extracting the answer to a question from the document. In our model, we first compute an embedding for the question using a bidirectional LSTM. Then, for each word  $t$  in the document, we augment the word embedding  $x_t$  with the computed question embedding. This is equivalent to adding an additional term which is linear in the question embedding into the gate equations 3-6, allowing the patterns an LSTM absorbs to be directly conditioned upon the question at hand.

$$
h _ {t} ^ {q} = \operatorname {B i L S T M} \left(x _ {t} ^ {q}\right) \tag {10}
$$

$$
h _ {t} = \operatorname {B i L S T M} \left(x _ {t} ^ {d} \| h _ {N} ^ {q}\right) \tag {11}
$$

# 3.3 LABEL EXTRACTION

In the problems under consideration, the answer is generally contained within the provided document. Having run the above model over a document while conditioning on a question, we are given a sequence of outputs  $h_1, \ldots, h_T$ . In order to identify the answer, for each word in the document we

compute

$$
p _ {t} = \operatorname {S o f t M a x} \left(\left[ \begin{array}{l} P \\ Q \end{array} \right] h _ {t}\right) = \frac {e ^ {P h _ {t}}}{e ^ {P h _ {t}} + e ^ {Q h _ {t}}}, \tag {12}
$$

where  $P^T$  and  $Q^T$  are vectors of the same dimension as the hidden state.

# 4 WORD IMPORTANCE SCORES IN LSTMS

We can decompose the numerator of the softmax output in (12) into a product of factors, each corresponding to the contribution of a word. Thus we can assign importance scores to words according to their contribution to the LSTM's prediction.

# 4.1 DECOMPOSING THE OUTPUT OF A LSTM

Define

$$
\beta_ {t, j} = \exp \left(P \left(o _ {t} \odot (\tanh  \left(c _ {j}\right) - \tanh  \left(c _ {j - 1}\right)\right)\right), \tag {13}
$$

so that

$$
\exp (P h _ {t}) = \exp \left( \right.\sum_ {j = 1} ^ {t} P \left(o _ {t} \odot \left(\tanh  \left(c _ {j}\right) - \tanh  \left(c _ {j - 1}\right)\right)\right) = \prod_ {j = 1} ^ {t} \beta_ {t, j}.
$$

As  $\tanh(c_j) - \tanh(c_{j-1})$  can be viewed as the update resulting from word  $j$ , so  $\beta_{t,j}$  can be interpreted as the multiplicative contribution to  $p_t$  by word  $j$ .

# 4.2 AN ADDITIVE DECOMPOSITION OF THE LSTM CELL

We will show below that the  $\beta_{t,j}$  capture some notion of the importance of a word to the LSTM's output. However, the  $\beta_{t,j}$  terms fail to account for how the information contributed by word  $j$  is affected by the LSTM's forget gates between words  $j$  and  $t$ . Consequently, we empirically found that the importance scores from this approach often yield a considerable amount of false positives. A more nuanced approach is obtained by considering the additive decomposition of  $c_{t}$  in Equation (14), where each term  $e_{t,j}$  can be interpreted as the contribution to the cell state  $c_{t}$  by word  $j$ . By iterating the equation  $c_{t} = f_{t}c_{t-1} + i_{t}\tilde{c}_{t}$ , we get that

$$
c _ {t} = \sum_ {i = 1} ^ {t} \left(\prod_ {j = i + 1} ^ {t} f _ {j}\right) i _ {i} \tilde {c} _ {i} = \sum_ {i = 1} ^ {t} e _ {t, i} \tag {14}
$$

This suggests a natural definition of an alternative score to the  $\beta_{t,j}$ , corresponding to augmenting the  $c_{j}$  terms with products of forget gates to reflect the upstream changes made to  $c_{j}$  after initially processing word  $j$ .

$$
\begin{array}{l} \gamma_ {t, j} = P \left(o _ {t} \odot \left(\tanh  \left(\sum_ {k = 1} ^ {j} e _ {t, k}\right) - \tanh  \left(\sum_ {k = 1} ^ {j - 1} e _ {t, k}\right)\right)\right) (15) \\ = P \left(o _ {t} \odot \left(\tanh  \left(\left(\prod_ {k = j + 1} ^ {t} f _ {k}\right) c _ {j}\right) - \tanh  \left(\left(\prod_ {k = j} ^ {t} f _ {k}\right) c _ {j - 1}\right)\right)\right) (16) \\ \end{array}
$$

# 5 APPROXIMATING AN LSTM USING PATTERN MATCHING

LSTMs are generally regarded as black box algorithms whose predictions can not be easily explained. We introduce a highly interpretable pattern-matching algorithm which we can extract from a trained LSTM.

<table><tr><td>Question</td><td>Going the Distance was written by who?</td><td>Who acted in 3 Strikes?</td><td>Who was Master of the House directed by?</td></tr><tr><td>Pattern</td><td>by written by</td><td>. film stars</td><td>film directed by</td></tr><tr><td>Article excerpt</td><td>Going the Distance is a 2010 American romantic comedy film directed by Nanette Burstein and written by Geoff Latulippe.</td><td>3 Strikes is a 2000 American screwball comedy film, written and directed by DJ Pooh, The film stars Brian Hooks as Rob Douglas</td><td>Master of the House is a 1925 Danish silent comedy film drama directed and written by ac-claimed filmmaker Carl Theodor Dreyer</td></tr></table>

Table 1: Examples of predictions made by pattern matching classifier. Pattern words are highlighted, answers are italicized

# 5.1 PATTERN MATCHING ALGORITHM

Given a list of patterns, our algorithm sequentially searches for each pattern. Once a matching pattern is found, the first entity within  $n_e$  words is returned, and the rest of the patterns are ignored. The criteria for matching a pattern are that the words contained within the pattern occur in order within the document, with at most  $n_p$  words between each consecutive pair. Example predictions are given in 1.

In contrast to a LSTM, which is very opaque, it is worth noting that the proposed model is very interpretable. For instance, when searching for a writer in a Wikipedia article, looking for patterns such as "written by", followed by an entity, seems like a very sensible approach to take. Moreover, this algorithm can be executed quickly on a single CPU, rather than the specialized GPU hardware required for LSTMs.

# 5.2 EXTRACTING FREQUENT PATTERNS OF IMPORTANT WORDS

We now describe how to use the introduced variable importance scores to extract patterns from a trained LSTM, which are then used in the above pattern matching algorithm. The techniques described here are for the forward portion of the bidirectional LSTM - the backward portion can be dealt with analogously. For a given answer located at word  $t$ , we use the sequence of importance scores  $\{\beta_{j,t}\}_{j=1}^{t}$  from (13),  $\{\gamma_{j,t}\}_{j=1}^{t}$  from (15), or the norms of the gradients of input word w.r.t. output, as in Alikaniotis et al. (2016); Denil et al. (2015); Bansal et al. (2016). For this discussion, suppose we choose the  $\beta$ . Then we choose a cutoff constant  $c$ , and extract an ordered list of important words  $\{i_1, \dots, i_r\}$  by choosing all words  $j$  with  $\beta_{j,t} > c$ ; in the experiments below we select  $c = 1.1$ . Finally, we extract sequences of consecutive important words ending at  $i_r$ , or  $\{(i_k)_{k=j}^r\}_{j=1}^r$ . For questions where the answer is an entity, the answer may occur only a few times in the entire corpus, making it impossible to extract generic patterns. Thus, if  $i_r = t$ , we also include contextual patterns of the form  $\{(i_k)_{k=j}^{r-1}\}_{j=1}^{r-1}$ .

As demonstrated in Karpathy et al. (2016), LSTMs are known to keep track of location information in the cell state. For questions inquiring about movies, in some groups of questions it was the case that the LSTM learned to exploit the enforced Wikipedia article structure by frequently returning the first word of the article. Thus, we also kept track of cases where the answer was the first word of the article, and that word was identified as significant, regardless of what the word was.

For each question, we use each of these three approaches separately and concatenate them to get a list of patterns. Repeating this process for each question, we compute aggregate counts of patterns. The final list of patterns is computed by taking all patterns occurring more than ten times. Example extracted patterns are shown in 2

<table><tr><td></td><td>Answer-based</td><td>Contextual</td><td>Location</td></tr><tr><td>Question</td><td>Music of the Heart, when was it released?</td><td>Who&#x27;s the writer of Experiment Perilous?</td><td>Gary David Goldberg directed which movies?</td></tr><tr><td>Article Excerpt</td><td>Music of the Heart is a 1999 dramatic film.</td><td>Experiment Perilous is a 1944 melodrama set at the turn of the 20th century. The film is based on a 1943 novel by Margaret Carpenter</td><td>Must Love Dogs is a 2005 romantic comedy film based on Claire Cook&#x27;s eponymous 2002 novel.</td></tr><tr><td>Example Extracted Patterns</td><td>1999, a 1999, is a 1999</td><td>by, novel by, 1943 novel by</td><td>Must Love Dogs, Beginning of Document</td></tr></table>

Table 2: Patterns extracted from articles. Words identified as important are highlighted

# 5.3 ORDERING PATTERNS

Having identified a list of patterns to search for, the order in which we search for them can have considerable effect on the final accuracy. Given a list of patterns, they can be arranged into a list of trees, where one pattern is a child of another if it can be constructed by adding a word on to its beginning. We then search through each tree, where the order is determined by the ratio of the pattern frequency of the tree's root to its overall frequency. Within a tree, we conduct a recursive search, according to the following logic. If all of a node's children have been explored, then we examine that node's pattern. If not, then we recursively search its children in the order given by the ratio of the pattern frequency to the overall frequency of the first word.

# 6 EXPERIMENTS

We now present the results of our experiments.

# 6.1 WIKIMOVIES

A recent line of work Li et al. (2016) Hewlett et al. (2016) Rajpurkar et al. (2016) Miller et al. (2016) has focused on neural network techniques for extracting answers directly from documents. Previous work had focused on Knowledge Bases (KBs), and techniques to map questions to logical forms suitable for querying them. Although they are effective within their domain, KBs are inevitably incomplete, and are thus an unsatisfactory solution to the general problem of question-answering. Wikipedia, in contrast, has enough information to answer a far broader array of questions, but is not as easy to query. Originally introduced in Miller et al. (2016), the WikiMovies dataset consists of questions about movies paired with Wikipedia articles.

WikiMovies consists of more than 100,000 questions about movies, paired with relevant Wikipedia articles. It was constructed using the pre-existing dataset MovieLens, paired with templates extracted from the SimpleQuestions dataset Bordes et al. (2015), a open-domain question answering dataset based on Freebase. They then selected a set of Wikipedia articles about movies by identifying a set of movies from OMDb that had an associated article by title match, and kept the title and first section for each article. We use the pre-defined splits into train, validation and test sets, containing 96k, 10k and 10k questions, respectively.

# 6.2 TRAINING DETAILS

We implemented our LSTM model in Torch using default hyperparameters for weight initializations. All documents and questions were pre-processed so that multiple word entities were concatenated into a single word. For a given question, relevant articles were found by first extracting from the question a set of rare words of frequency less than 1000, then returning a list of Wikipedia articles

<table><tr><td>Model</td><td>Test accuracy</td></tr><tr><td>KV-MemNN IE</td><td>68.3</td></tr><tr><td>KV-MemNN Doc</td><td>76.2</td></tr><tr><td>LSTM</td><td>72.8</td></tr><tr><td>Manual Pattern Matching</td><td>72.9</td></tr><tr><td>Automatic Pattern Matching</td><td>68.3</td></tr><tr><td>Automatic Cell-Difference Pattern Matching</td><td>65.8</td></tr><tr><td>Gradient Pattern Matching</td><td>66.7</td></tr></table>

Table 3: Test results on WikiMovies, measured in % hits@1. See section 6.3 for further descriptions of the models.

containing any of those words. The word and hidden representations of the LSTM were both set to dimension 100. The model was optimized using adam Kingma & Ba (2015) with the default learning rate of 0.001 for five epochs, at which point the performance on the validation set ceased to improve. For the pattern matching classifiers, all patterns were extracted by examining questions in the validation set, and pattern-matching parameters  $n_p = 3$  and  $n_e = 5$  were used.

# 6.3 RESULTS

We report results on seven different models in Tables 3 and 4. We show the results from Miller et al. (2016), which fit a key-value memory network (KV-MemNN) on representations from information extraction (IE) and raw text (Doc). Next, we report the results of the LSTM described in Section 3. Finally, we show the results of using four variants of the pattern matching algorithm described in Section 5: using patterns extracted using the additive decomposition (automatic RE), difference in cells approaches (automatic cell-difference) and gradient importance scores (gradient), as discussed in Section 2. A manual pattern matching model (manual RE) was constructed by choosing a reordered subset of the patterns used in the automatic model and generalizing the patterns for movie to year to search for any number, as discussed below in Section 7.2. Performance is reported using the accuracy of the top hit over all possible answers (all entities), i.e. the hits@1 metric.

As shown in 3, the manually adjusted patterns are able to approximate an LSTM's performance with a gain of  $0.1\%$ , indicating that on this dataset we are able to produce essentially equivalent performance overall, although there is variation across different question types. Moreover, the automatic approach is only  $4.5\%$  lower than the LSTM, which is still quite high for such a simple model. It is also worth commenting that the automatic pattern matching approach produces similar performance to the KV-MemNN IE model, which is built upon the sizeable amount of effort from the information extraction community. Finally, the additive cell decomposition patterns outperform both the gradient and cell-difference patterns, providing further evidence that they are a more sensible importance measure.

# 7 DISCUSSION

# 7.1 LEARNED PATTERNS

We now examine learned patterns for some of the document categories, shown in 7.1. These patterns qualitatively match what would be reasonably expected, providing further validation of our approach. Moreover, they reveal some interesting patterns. In the movie to language category, for instance, the word 'interchangeably' is the top pattern, followed by a list of languages. Upon closer examination, it turns out that many articles contain phrases of the form "X and Y are spoken interchangeably in the film", where X and Y are languages. This is an excellent example of an LSTM discovering patterns which, at first glance, are not obvious.

<table><tr><td></td><td>KV-MemNNIE</td><td>KV-MemNNDoc</td><td>LSTM</td><td>ManualRE</td><td>AutoRE</td><td>Auto CellDiff RE</td><td>GradientRE</td></tr><tr><td>Actor to Movie</td><td>66</td><td>83</td><td>84</td><td>77</td><td>78</td><td>77</td><td>76</td></tr><tr><td>Director to Movie</td><td>78</td><td>91</td><td>84</td><td>82</td><td>82</td><td>83</td><td>81</td></tr><tr><td>Writer to Movie</td><td>72</td><td>91</td><td>90</td><td>86</td><td>87</td><td>88</td><td>87</td></tr><tr><td>Tag to Movie</td><td>35</td><td>49</td><td>47</td><td>39</td><td>38</td><td>38</td><td>38</td></tr><tr><td>Movie to Year</td><td>75</td><td>89</td><td>78</td><td>86</td><td>66</td><td>65</td><td>72</td></tr><tr><td>Movie to Writer</td><td>61</td><td>64</td><td>77</td><td>67</td><td>62</td><td>46</td><td>47</td></tr><tr><td>Movie to Actor</td><td>64</td><td>64</td><td>72</td><td>69</td><td>68</td><td>70</td><td>70</td></tr><tr><td>Movie to Director</td><td>76</td><td>79</td><td>79</td><td>82</td><td>82</td><td>83</td><td>83</td></tr><tr><td>Movie to Genre</td><td>84</td><td>86</td><td>66</td><td>74</td><td>64</td><td>61</td><td>66</td></tr><tr><td>Movie to Votes</td><td>92</td><td>92</td><td>67</td><td>50</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Movie to Rating</td><td>75</td><td>92</td><td>81</td><td>92</td><td>92</td><td>92</td><td>92</td></tr><tr><td>Movie to Language</td><td>62</td><td>84</td><td>45</td><td>52</td><td>52</td><td>34</td><td>34</td></tr><tr><td>Movie to Tags</td><td>47</td><td>48</td><td>44</td><td>47</td><td>47</td><td>45</td><td>39</td></tr></table>

Table 4: Results broken down by question category. See section 6.3 for further descriptions of the models.  

<table><tr><td>Category</td><td>Top Patterns</td></tr><tr><td>Movie to language</td><td>interchangeably, english, italian, swedish,french, german, japanese</td></tr><tr><td>Movie to actors</td><td>drama * starring, comedy * starring, film * di-rected * starring, directed * starring, comedy *film * starring, drama * film * starring, ameri-can * film * starring</td></tr><tr><td>Movie to year</td><td>1938, is * 1999, 1999, 1988, 1981, 1978, is *1997, 1997</td></tr><tr><td>Movie to writer</td><td>drama * written * by, directed * written * by, by* written * by, film * written * by, written * by,same * by, screenplay * by, written * directed *by</td></tr><tr><td>Movie to director</td><td>is * film * woody allen, film * woody allen,woody allen, drama * directed * by, written* directed * by, comedy * directed * by, is *directed * by, film * directed * by</td></tr><tr><td>Movie to genre</td><td>american * comedy, comedy, french * drama,american * drama, - * drama, is * drama, is * a* drama, a * drama</td></tr><tr><td>Writer to movie</td><td>Beginning of document</td></tr></table>

Table 5: Top patterns for selected question categories, * denotes a number of words between zero and three

<table><tr><td>Additive cell decomposition</td><td>Difference in cell values</td><td>Gradient</td></tr><tr><td>west is west is a 2010 
british comedy - drama 
film , which is a sequel 
to the 1999 comedy ” 
east is east ” , it stars 
aqib khan , om puri , linda bassett , 
ila arun and jimi mistry ,</td><td>West 
IS 
West 
is 2010 british comedy 
- 
drama 
film , which is , sequel 
to the 1999 comedy ” ” it 
stars aqib 
khan , om puri , 
linda bassett , ila arun and jimi 
mistry .</td><td>west is west is 
a 2010 british 
comedy - drama 
film , which is a 
sequel to the 
1999 
com-
edy” east is east 
”, it stars 
aqib 
khan , om puri , 
linda bassett , ila arun and 
jimi mistry ,</td></tr></table>

Table 6: Comparison of importance scores acquired by two different approaches, conditioning on the question "the film west is west starred which actors?" Bigger and darker means more important.

# 7.2 INCORPORATING PRIOR KNOWLEDGE

One of the benefits of understanding how a model works is that it is easier to incorporate prior information. For instance, for the 'year to movie' questions, 7.1 shows that the LSTM is learning to identify particular years, e.g. 2012, as being important features. However, under standard word-based models, each year has an independent embedding, so jointly learning an 'is number' across tens of year embeddings is quite challenging. This suggests the simple solution of adding an 'is number' feature. By adding such a feature to our pattern-based classifier, we were able to improve the performance on this class of questions by  $20\%$ , allowing us to outperform the LSTM model by  $8\%$ .

# 7.3 COMPARISON BETWEEN TWO WORD IMPORTANCE MEASURES

In the pattern matching results, the two different word importance scores perform comparably across most questions, with the exception of "movie to writer" questions. However, as discussed before, the difference in cells technique fails to account for how the updates resulting from word  $j$  are affected by the LSTM's forget gates between when the word is initially processed and the answer. Consequently, we empirically found that without the interluding forget gates to dampen cell movements, the variable importance scores were far noisier than in additive cell decomposition approach. A side-by-side comparison is given in 6. Under the additive cell decomposition, it identifies the phrase 'it stars', as well as the actor's name Aqib Khan as being important, a sensible conclusion. On the other hand, the difference in cells approach yields widely changing importance scores, which are challenging to interpret. The gradient based scores are smoothed than the difference in cells, although they still have some noise.

# 8 CONCLUSION

In this paper, we introduced a novel approach for approximating an LSTM using a simple, rules-based classifier. To extract patterns, we introduce two new word importance scores by decomposing the numerator of a softmax output into a product of factors, with each factor denoting the contribution of a particular word. We compare pattern extractions using these new scores against pattern extractions using the norm of the gradient of an input w.r.t. the output scores. We find that on the WikiMovies dataset, the gradient norm method gives superior results to the naive decomposition in (13), but inferior results to the more sophisticated decomposition in (15) that takes into account the activations of the forget gate. We believe that this represents an exciting new paradigm for analysing the behaviour of LSTM's. In future work, we hope to further exploit these insights to better understand and improve upon neural models in other natural language processing problems.

# REFERENCES

Dimitrios Alikaniotis, Helen Yannakoudakis, and Marek Rei. Automatic text scoring using neural networks. In Association for Computational Linguistics, 2016.  
Trapit Bansal, David Belanger, and Andrew McCallum. Ask the gru: Multi-task learning for deep text recommendations. In ACM international conference on Recommender Systems (RecSys), 2016.  
Antoine Bordes, Nicolas Usunier, Sumit Chopra, and Jason Weston. Large-scale simple question answering with memory networks. arXiv preprint arXiv:1506.02075, 2015.  
Misha Denil, Alban Demiraj, and Nando de Freitas. Extraction of salient sentences from labelled documents. In arXiv preprint: https://arxiv.org/abs/1412.6815, 2015.  
Strobelt Hendrik, Sebastian Gehrmann, Bernd Huber, Hanspeter Pfister, and Alexander M. Rush. Visual analysis of hidden state dynamics in recurrent neural networks. In arXiv, 2016.  
Daniel Hewlett, Alexandre Lacoste, Llion Jones, Illia Polosukhin, Andrew Fandrianto, Jay Han, Matthew Kelcey, and David Berthelot. Wikireading: A novel large-scale language understanding task over wikipedia. In Association for Computational Linguistics, 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.  
Andrej Karpathy, Justin Johnson, and Fei-Fei Li. Visualizing and understanding recurrent neural networks. In ICLR Workshop, 2016.  
Diederick P. Kingma and Jimmy Lei Ba. Adam: A method for stochastic optimization. In International Conference for Learning Representations, 2015.  
Peng Li, Wei Li, Zhengyan He, Xuguang Wang, Ying Cao, Jie Zhou, and Wei Xu. Dataset and neural recurrent sequence labeling model for open-domain factoid question answering. In arXiv, 2016.  
Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, and Jason Weston. Key-value memory networks for directly reading documents. In Empirical Methods for Natural Language Processing, 2016.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. In Empirical Methods for Natural Language Processing, 2016.  
Ilya Sutskever, Oriol Vinyals, and Quoc Le. Sequence to sequence learning with neural networks. In Neural Information Processing Systems, 2014.
