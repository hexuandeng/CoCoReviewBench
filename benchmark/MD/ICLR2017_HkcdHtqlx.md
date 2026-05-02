# GATED-ATTENTION READERS FOR TEXT COMPREHENSION

Bhuwan Dhingra\* Hanxiao Liu\* Zhilin Yang, William W. Cohen & Ruslan Salakhutdinov

School of Computer Science

Carnegie Mellon University

{bdhingra,hanxiaol,zhiliny,wcohen,rsalakhu}@cs.cmu.edu

# ABSTRACT

In this paper we study the problem of answering cloze-style questions over documents. Our model, the Gated-Attention (GA) Reader, integrates a multi-hop architecture with a novel attention mechanism, which is based on multiplicative interactions between the query embedding and the intermediate states of a recurrent neural network document reader. This enables the reader to build query-specific representations of tokens in the document for accurate answer selection. The GA Reader obtains state-of-the-art results on three benchmarks for this task—the CNN & Daily Mail news stories and the Who Did What dataset. The effectiveness of multiplicative interaction is demonstrated by an ablation study, and by comparing to alternative compositional operators for implementing the gated-attention.

# 1 INTRODUCTION

A recent trend to measure progress towards machine reading is to test a system's ability to answer questions about a document it has to comprehend. Towards this end, several large-scale datasets of cloze-style questions over a context document have been introduced recently, which allow the training of supervised machine learning systems (Hermann et al., 2015; Hill et al., 2015; Onishi et al., 2016). Such datasets can be easily constructed automatically and the unambiguous nature of their queries provides an objective benchmark to measure a system's performance at text comprehension.

Deep learning models have recently been shown to outperform traditional shallow approaches on text comprehension tasks (Hermann et al., 2015). The success of many recent models can be attributed primarily to two factors: (1) Multi-hop architectures allow a (Weston et al., 2014; Sordoni et al., 2016; Shen et al., 2016), model to scan the document and the question iteratively for multiple passes. (2) Attention mechanisms, (Weston et al., 2014; Chen et al., 2016; Hermann et al., 2015) borrowed from the machine translation literature (Bahdanau et al., 2014), allow the model to focus on appropriate subparts of the context document. Intuitively, the multi-hop architecture allows the reader to incrementally refine token representations, and the attention mechanism re-weights different parts in the document according to their relevance to the query.

The effectiveness of multi-hop reasoning and attentions have been explored orthogonally so far in the literature. In this paper, we focus on combining both in a complementary manner, by designing a novel attention mechanism which gates the evolving token representations across hops. More specifically, unlike existing models where the query attention is applied either token-wise (Hermann et al., 2015; Kadlec et al., 2016; Chen et al., 2016; Hill et al., 2015) or sentence-wise (Weston et al., 2014; Sukhbaatar et al., 2015) to allow weighted aggregation, the Gated-Attention (GA) module proposed in this work allows the query to directly interact with each dimension of the token embeddings at the semantic-level, and is applied layer-wise as information filters during the multi-hop representation learning process. Such a fine-grained attention enables our model to learn conditional token representations with respect to the given question, leading to accurate answer selections.

We show in our experiments that the proposed GA reader, despite its relative simplicity, consistently improves over a variety of strong baselines on three benchmark datasets<sup>1</sup>. Our key contribution,

the GA module, provides a significant improvement when the dataset size is large. Qualitatively, visualization of the attentions at intermediate layers of the GA reader shows that in each layer the GA reader attends to distinct salient aspects of the query which help in determining the answer.

# 2 RELATED WORK

The cloze-style QA task involves tuples of the form  $(d, q, a, \mathcal{C})$ , where  $d$  is a document (context),  $q$  is a query over the contents of  $d$ , in which a phrase is replaced with a placeholder, and  $a$  is the answer to  $q$ , which comes from a set of candidates  $\mathcal{C}$ . In this work we consider datasets where each candidate  $c \in \mathcal{C}$  has at least one token which also appears in the document. The task can then be described as: given a document-query pair  $(d, q)$ , find  $a \in \mathcal{C}$  which answers  $q$ . Below we provide an overview of representative neural network architectures which have been applied to this problem.

LSTMs with Attention: Several architectures introduced in (Hermann et al., 2015) employ LSTM units to compute a combined document-query representation  $g(d, q)$ , which is used to rank the candidate answers. Their techniques include the DeepLSTM Reader which performs a single forward pass through the concatenated (document, query) pair to obtain  $g(d, q)$ ; the Attentive Reader which first computes a document vector  $d(q)$  by a weighted aggregation of words according to attentions based on  $q$ , and then combines  $d(q)$  and  $q$  to obtain their joint representation  $g(d(q), q)$ ; and the Impatient Reader where the document representation is built incrementally. The architecture of the Attentive Reader has been simplified recently in Stanford Attentive Reader, where shallower recurrent units were used with a bilinear form for the query-document attention (Chen et al., 2016).

Attention Sum: The Attention-Sum (AS) Reader (Kadlec et al., 2016) uses two bi-directional GRU networks (Cho et al., 2014) to encode both  $d$  and  $q$  into vectors, similar to Stanford AR. A probability distribution over the entities in  $d$  is obtained by computing dot products between  $q$  and the entity embeddings and taking a softmax. An aggregation scheme named pointer-sum attention is further applied to sum the probabilities of the same entity, so that frequent entities the document will be favored compared to rare ones. Building on the AS Reader, the Attention-over-Attention (AoA) Reader (Cui et al., 2016) introduces a two-way attention mechanism where the query and the document are mutually attentive to each other.

Mulit-hop Architectures: Memory Networks (MemNets) were proposed in (Weston et al., 2014), where each sentence in the document is encoded to a memory by aggregating nearby words. Attention over the memory slots given the query is used to compute an overall memory and to renew the query representation over multiple iterations, allowing certain types of reasoning over the salient facts in the memory and the query. Neural Semantic Encoders (NSE) (Munkhdalai & Yu, 2016a) extended MemNets by introducing a write operation which can evolve the memory over time during the course of reading. Iterative reasoning has been found effective in several more recent models, including the Iterative Attentive Reader (Sordoni et al., 2016) and ReasoNet (Shen et al., 2016). The latter allows a dynamic number of reasoning steps and is trained with reinforcement learning.

Other related works include Dynamic Entity Representation network (DER) (Kobayashi et al., 2016), which builds dynamic representations of the candidate answers while reading the document, and accumulates the information about an entity by max-pooling. EpiReader (Trischler et al., 2016) consists of two networks, where one proposes a small set of candidate answers, and the other reranks the proposed candidates conditioned on the query and the context. (Bajgar et al., 2016) showed a  $10\%$  improvement on the CBT corpus (Hill et al., 2015) by training the AS Reader on an augmented training set of about 14 million examples, making a case for community to exploit data abundance. The focus of this paper, however, is on designing models which exploit the available data efficiently.

# 3 GATED-ATTENTION READER

# 3.1 MOTIVATION

Our proposed GA readers perform multiple hops over the document (context), similar to the Memory Networks architecture (Sukhbaatar et al., 2015). Multi-hop architectures mimic the multi-step comprehension process of human readers, and have shown promising results in several recent models for text comprehension (Sordoni et al., 2016; Kumar et al., 2015; Shen et al., 2016). The contextual representations in GA readers, namely the embeddings of words in the document, are iteratively

refined across hops until reaching a final attention-sum module (Kadlec et al., 2016) which maps the contextual representations in the last hop to a probability distribution over candidate answers.

The attention mechanism has been introduced recently to model human focus, leading to significant improvement in machine translation and image captioning (Bahdanau et al., 2014; Mnih et al., 2014). In reading comprehension tasks, ideally, the semantic meanings carried by the contextual embeddings should be aware of the query across hops. As an example, human readers are able to keep the question in mind during multiple passes of reading, to successively mask away information irrelevant to the query. However, existing neural network readers are restricted to either attend to tokens (Hermann et al., 2015; Chen et al., 2016) or entire sentences (Weston et al., 2014), with the assumption that certain sub-parts of the document are more important than others. In contrast, we propose a finer-grained model which attends to components of the semantic representation being built up by the GRU. The new attention mechanism, called gated-attention, is implemented based on multiplicative interactions between the query and the contextual embeddings, and is applied per hop to act as fine-grained information filters during the multi-step reasoning. The filters weigh individual components of the vector representation of each token in the document separately.

The design of gated-attention layers is motivated by the effectiveness of multiplicative interaction among vector-space representations, e.g., in various types of recurrent units (Hochreiter & Schmidhuber, 1997; Wu et al., 2016) and in relational learning (Yang et al., 2014; Kiros et al., 2014). While other types of compositional operators are possible, such as concatenation or addition (Mitchell & Lapata, 2008), we find that multiplication has strong empirical performance (section 4.4). Intuitively, multiplicative interaction  $e \odot q$  between two word embeddings  $e$  and  $q$  adjusts the semantic meaning of  $e$  towards  $q$ , keeping the compositionality of the original embeddings preserved. $^{2}$

# 3.2 MODEL DETAILS

Several components of the model use a Gated Recurrent Unit (GRU) (Cho et al., 2014) which maps an input sequence  $X = [x_{1}, x_{2}, \ldots, x_{T}]$  to an output sequence  $H = [h_{1}, h_{2}, \ldots, h_{T}]$  as follows:

$$
r _ {t} = \sigma \left(W _ {r} x _ {t} + U _ {r} h _ {t - 1} + b _ {r}\right),
$$

$$
z _ {t} = \sigma \left(W _ {z} x _ {t} + U _ {z} h _ {t - 1} + b _ {z}\right),
$$

$$
\tilde {h} _ {t} = \tanh  \left(W _ {h} x _ {t} + U _ {h} \left(r _ {t} \odot h _ {t - 1}\right) + b _ {h}\right),
$$

$$
h _ {t} = (1 - z _ {t}) \odot h _ {t - 1} + z _ {t} \odot \tilde {h} _ {t}.
$$

where  $\odot$  denotes the Hadamard product or the element-wise multiplication.  $r_t$  and  $z_{t}$  are called the reset and update gates respectively, and  $\tilde{h}_t$  the candidate output. A Bi-directional GRU (Bi-GRU) processes the sequence in both forward and backward directions to produce two sequences  $[h_1^f,h_2^f,\dots ,h_T^f ]$  and  $[h_1^b,h_2^b,\dots ,h_T^b ]$ , which are concatenated at the output

$$
\stackrel {\longleftrightarrow} {\operatorname {G R U}} (X) = \left[ h _ {1} ^ {f} \| h _ {T} ^ {b}, \dots , h _ {T} ^ {f} \| h _ {1} ^ {b} \right] \tag {1}
$$

where  $\overleftrightarrow{\mathrm{GRU}}(X)$  denotes the full output of the Bi-GRU obtained by concatenating each forward state  $h_i^f$  and backward state  $h_{T - i + 1}^b$  at time-step  $i$  given the input  $X$ . Note  $\overleftrightarrow{\mathrm{GRU}}(X)$  is a matrix in  $\mathbb{R}^{2n_h\times T}$  where  $n_h$  stands for the number of hidden units in GRU.

Let  $X^{(0)} = [x_1^{(0)}, x_2^{(0)}, \ldots, x_{|D|}^{(0)}]$  denote the token embeddings of the document, which are also inputs at layer 1 for the document reader below, and  $Y = [y_1, y_2, \ldots, y_{|Q|}]$  denote the token embeddings of the query. Here  $|D|$  and  $|Q|$  denote the document and query lengths respectively.

# 3.2.1 MULTI-HOP ARCHITECTURE

Figure 1 illustrates the Gated-Attention (GA) reader. The model reads the document and the query over  $K$  horizontal layers, where layer  $k$  receives the contextual embeddings  $X^{(k - 1)}$  of the document from the previous layer. The document embeddings are transformed by taking the full output of a document Bi-GRU (indicated in blue in Figure 1):

$$
D ^ {(k)} = \overleftrightarrow {\mathrm {G R U}} _ {D} ^ {(k)} (X ^ {(k - 1)}) \tag {2}
$$

$$
{ } ^ { 2 } e _ { 1 } \odot q + e _ { 2 } \odot q = ( e _ { 1 } + e _ { 2 } ) \odot q , \forall e _ { 1 } , e _ { 2 } .
$$

![](images/06ad53eec3eb387c4aa0da29fce43ba3c5cca27f2911c02bbb0a72f89f5bc8e6.jpg)  
Figure 1: Gated-Attention Reader. Dashed lines represent dropout connections.

At the same time, a layer-specific query representation is computed as the full output of a separate query Bi-GRU (indicated in green in Figure 1):

$$
Q ^ {(k)} = \stackrel {\longleftrightarrow} {\mathrm {G R U}} _ {Q} ^ {(k)} (Y) \tag {3}
$$

Next, Gated-Attention is applied to  $D^{(k)}$  and  $Q^{(k)}$  to compute inputs for the next layer  $X^{(k)}$ .

$$
X ^ {(k)} = \operatorname {G A} \left(D ^ {(k)}, Q ^ {(k)}\right) \tag {4}
$$

where GA is defined in the following subsection.

# 3.2.2 GATED-ATTENTION MODULE

For brevity, let us drop the superscript  $k$  in this subsection as we are focusing on a particular layer. For each token  $d_{i}$  in  $\bar{D}$ , the GA module forms a token-specific representation of the query  $\tilde{q}_i$  using soft attention, and then multiplies the query representation element-wise with the document token representation. Specifically, for  $i = 1,\dots ,|D|$ :

$$
\alpha_ {i} = \operatorname {s o f t m a x} \left(Q ^ {\top} d _ {i}\right) \tag {5}
$$

$$
\tilde {q} _ {i} = Q \alpha_ {i}
$$

$$
x _ {i} = d _ {i} \odot \tilde {q} _ {i} \tag {6}
$$

In equation (6) we use the multiplication operator to model the interactions between  $d_{i}$  and  $\tilde{q}_i$ . In the experiments section, we also report results for other choices of gating functions, including addition  $x_{i} = d_{i} + \tilde{q}_{i}$  and concatenation  $x_{i} = d_{i}\| \tilde{q}_{i}$ .

# 3.2.3 ANSWER PREDICTION

Let  $q_{\ell}^{(K)} = q_{\ell}^{f}\| q_{T - \ell +1}^{b}$  be an intermediate output of the final layer query Bi-GRU at the location  $\ell$  of the cloze token in the query, and  $D^{(K)} = \overleftrightarrow{\mathrm{GRU}}_D^{(K)}(X^{(K - 1)})$  be the full output of final layer document Bi-GRU. To obtain the probability that a particular token in the document answers the query, we take an inner-product between these two, and pass through a softmax layer:

$$
s = \operatorname {s o f t m a x} \left(\left(q _ {\ell} ^ {(K)}\right) ^ {T} D ^ {(K)}\right) \tag {7}
$$

where vector  $s$  defines a probability distribution over the  $|D|$  tokens in the document. The probability of a particular candidate  $c \in \mathcal{C}$  as being the answer is then computed by aggregating the probabilities of all document tokens which appear in  $c$  and renormalizing over the candidates:

$$
\Pr (c \mid d, q) \propto \sum_ {i \in \mathbb {I} (c, d)} s _ {i} \tag {8}
$$

Table 1: Dataset statistics.  

<table><tr><td></td><td>CNN</td><td>Daily Mail</td><td>CBT-NE</td><td>CBT-CN</td><td>WDW-Strict</td><td>WDW-Relaxed</td></tr><tr><td># train</td><td>380,298</td><td>879,450</td><td>108,719</td><td>120,769</td><td>127,786</td><td>185,978</td></tr><tr><td># validation</td><td>3,924</td><td>64,835</td><td>2,000</td><td>2,000</td><td>10,000</td><td>10,000</td></tr><tr><td># test</td><td>3,198</td><td>53,182</td><td>2,500</td><td>2,500</td><td>10,000</td><td>10,000</td></tr><tr><td># vocab</td><td>118,497</td><td>208,045</td><td>53,063</td><td>53,185</td><td>347,406</td><td>308,602</td></tr><tr><td>max doc length</td><td>2,000</td><td>2,000</td><td>1,338</td><td>1,338</td><td>3,085</td><td>3,085</td></tr></table>

where  $\mathbb{I}(c,d)$  is the set of positions where a token in  $c$  appears in the document  $d$ . This aggregation operation is the same as the pointer sum attention applied in the AS Reader (Kadlec et al., 2016).

Finally, the candidate with maximum probability is selected as the predicted answer:

$$
a ^ {*} = \operatorname {a r g m a x} _ {c \in \mathcal {C}} \Pr (c | d, q). \tag {9}
$$

During the training phase, model parameters of the GA reader are updated w.r.t. a cross-entropy loss between the predicted probabilities and the true answers.

# 3.2.4 FURTHER ENHANCEMENTS

Character-level Embeddings: Given a token  $w$  from the document or query, its vector space representation is computed as  $x = L(w) || C(w)$ .  $L(w)$  retrieves the word-embedding for  $w$  from a lookup table  $L \in \mathbb{R}^{|V| \times n_l}$ , whose rows hold a vector for each unique token in the vocabulary. We also utilize a character composition model  $C(w)$  which generates an orthographic embedding of the token. Such embeddings have been previously shown to be helpful for tasks like Named Entity Recognition (Yang et al., 2016) and dealing with OOV tokens at test time (Dhingra et al., 2016). The embedding  $C(w)$  is generated by taking the final outputs  $z_{n_c}^f$  and  $z_{n_c}^b$  of a Bi-GRU applied to embeddings from a lookup table of characters in the token, and applying a linear transformation:

$$
z = z _ {n _ {c}} ^ {f} | | z _ {n _ {c}} ^ {b}
$$

$$
C (w) = W z + b
$$

Question Evidence Common Word Feature (qe-comm): (Li et al., 2016) recently proposed a simple token level indicator feature which significantly boosts reading comprehension performance in some cases. For each token in the document we construct a one-hot vector  $f_{i} \in \{0,1\}^{2}$  indicating whether that token is present in the query or not. It can be incorporated into the GA reader by assigning a feature lookup table  $F \in \mathbb{R}^{n_F \times 2}$  (we use  $n_F = 2$ ), taking the feature embedding  $e_{i} = f_{i}^{\overline{T}}F$  and appending it to the inputs of the last layer document BiGRU as,  $x_{i}^{(K)} \| f_{i}$  for all  $i$ . We conducted several experiments both with and without this feature and observed some interesting trends, which are discussed below. Henceforth, we refer to this feature as the qe-comm feature or just feature.

# 4 EXPERIMENTS AND RESULTS

# 4.1 DATASETS

We evaluate the GA reader on five large-scale datasets recently proposed in the literature. The first two, CNN and Daily Mail news stories consist of articles from the popular CNN and Daily Mail websites (Hermann et al., 2015). A query over each article is formed by removing an entity from the short summary which follows the article. Further, entities within each article were anonymized to make the task purely a comprehension one. N-gram statistics, for instance, computed over the entire corpus are no longer useful in such an anonymized corpus.

The next two datasets are formed from two different subsets of the Children's Book Test (CBT) $^{4}$  (Hill et al., 2015). Documents consist of 20 contiguous sentences from the body of a popular children's book, and queries are formed by deleting a token from the  $21^{\text{st}}$  sentence. We only focus on

Table 2: Hyperparameter settings for each dataset. dim() indicates hidden state size of GRU.  

<table><tr><td>Hyperparameter</td><td>CNN</td><td>Daily Mail</td><td>CBT-NE</td><td>CBT-CN</td><td>WDW-Strict</td><td>WDW-Relaxed</td></tr><tr><td>Dropout</td><td>0.2</td><td>0.1</td><td>0.4</td><td>0.4</td><td>0.3</td><td>0.3</td></tr><tr><td>dim(GRU*)</td><td>256</td><td>256</td><td>128</td><td>128</td><td>128</td><td>128</td></tr></table>

subsets where the deleted token is either a common noun (CN) or named entity (NE) since simple language models already give human-level performance on the other types (cf. (Hill et al., 2015)).

The final dataset we evaluate on is Who Did What $^{5}$  (WDW) (Onishi et al., 2016), constructed from the LDC English Gigaword newswire corpus. First, article pairs which appeared around the same time and with overlapping entities are chosen, and then one article forms the document and a cloze query is constructed from the other. Missing tokens are always person named entities. Questions which are easily answered by simple baselines are filtered out, to make the task more challenging. There are two versions of the training set—a small but focused "Strict" version and a large but noisy "Relaxed" version. We report results on both settings which share the same validation and test sets. Statistics of all the datasets used in our experiments are summarized in Table 1.

# 4.2 IMPLEMENTATION DETAILS

Our model was implemented using the Theano (Theano Development Team, 2016) and Lasagne $^{6}$  Python libraries. We used stochastic gradient descent with ADAM updates for optimization, which combines classical momentum and adaptive gradients (Kingma & Ba, 2014). The batch size was 32 and the initial learning rate was  $5 \times 10^{-4}$  which was halved every epoch after the second epoch. The same setting is applied to all models and datasets. We also used gradient clipping with a threshold of 10 to stabilize GRU training (Pascanu et al., 2012). We set the number of layers  $K$  to be 3 for all experiments, which outperformed  $K < 3$  on the validation sets. We obtain similar performance using  $K = 4$  and  $K = 3$  on WDW. The number of hidden units for the character GRU was set to 50. The remaining two hyperparameters—size of document and query GRUs, and dropout rate—were tuned on the validation set, and their optimal values are shown in Table 2. In general, the optimal GRU size increases and the dropout rate decreases as the corpus size increases.

The word lookup table was initialized with  $100d$  GloVe vectors $^{7}$  (Pennington et al., 2014) and OOV tokens at test time were assigned unique random vectors. We empirically observed that initializing with pre-trained embeddings gives higher performance compared to random initialization for all datasets. We do not use the character composition model for CNN and Daily Mail, since entities (and hence candidate answers) are anonymized to generic tokens in these datasets. For other datasets the character lookup table was randomly initialized with  $50d$  vectors. All other parameters were initialized to their default values as specified in the Lasagne library.

# 4.3 PERFORMANCE COMPARISON

Tables 3 and 5 show a comparison of the performance of GA Reader with previously published results on WDW and CNN, Daily Mail, CBT datasets respectively. The numbers reported for GA Reader are for single best models, though we compare to both ensembles and single models from prior work. GA Reader-- refers to an earlier version of the model, unpublished but described in a preprint, with the following differences—(1) it does not utilize token-specific attentions within the GA module, as described in equation (5), (2) it does not use a character composition model, (3) it is initialized with word embeddings pretrained on the corpus itself rather than GloVe. GA Reader (+feature) refers to the model enhanced with qe-comm features (sec 3.2.4).

Interestingly, we observe that feature engineering leads to significant improvements for WDW and CBT datasets, but not for CNN and Daily Mail datasets. We note that anonymization of the latter datasets means that there is already some feature engineering (it adds hints about whether a token is an entity), and these are much larger than the other four. In machine learning it is common to see the effect of feature engineering diminish with increasing data size.

Table 3: Validation/Test accuracy (\%) on WDW dataset for both "Strict" and "Relaxed" settings. Results marked with  $\dagger$  are cf previously published works.  

<table><tr><td rowspan="2">Model</td><td colspan="2">Strict</td><td colspan="2">Relaxed</td></tr><tr><td>Val</td><td>Test</td><td>Val</td><td>Test</td></tr><tr><td>Human †</td><td>-</td><td>84.0</td><td>-</td><td>-</td></tr><tr><td>Attentive Reader †</td><td>-</td><td>53.0</td><td>-</td><td>55.0</td></tr><tr><td>AS Reader †</td><td>-</td><td>57.0</td><td>-</td><td>59.0</td></tr><tr><td>Stanford AR †</td><td>-</td><td>64.0</td><td>-</td><td>65.0</td></tr><tr><td>NSE †</td><td>66.5</td><td>66.2</td><td>67.0</td><td>66.7</td></tr><tr><td>GA Reader-- †</td><td>-</td><td>57.0</td><td>-</td><td>60.0</td></tr><tr><td>GA Reader</td><td>67.8</td><td>67.0</td><td>66.4</td><td>66.3</td></tr><tr><td>GA Reader (+feature)</td><td>70.1</td><td>69.5</td><td>70.9</td><td>70.6</td></tr></table>

Table 4: Performance of different gating functions on WDW dataset, without using the qe-comm feature.  

<table><tr><td rowspan="2">Gating Function</td><td colspan="2">Accuracy</td></tr><tr><td>Val</td><td>Test</td></tr><tr><td>Sum</td><td>62.9</td><td>62.1</td></tr><tr><td>Concatenate</td><td>63.1</td><td>61.1</td></tr><tr><td>Multiply</td><td>67.8</td><td>67.0</td></tr></table>

Table 5: Validation/Test accuracy (%) on CNN, Daily Mail and CBT. Results marked with  $\dagger$  are cf previously published works. Results marked with  $\ddagger$  were obtained by training on a larger training set. Best performance on standard training sets is in bold, and on larger training sets in italics.  

<table><tr><td rowspan="2">Model</td><td colspan="2">CNN</td><td colspan="2">Daily Mail</td><td colspan="2">CBT-NE</td><td colspan="2">CBT-CN</td></tr><tr><td>Val</td><td>Test</td><td>Val</td><td>Test</td><td>Val</td><td>Test</td><td>Val</td><td>Test</td></tr><tr><td>Humans (query) †</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>52.0</td><td>-</td><td>64.4</td></tr><tr><td>Humans (context + query) †</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>81.6</td><td>-</td><td>81.6</td></tr><tr><td>LSTMs (context + query) †</td><td>-</td><td>-</td><td>-</td><td>-</td><td>51.2</td><td>41.8</td><td>62.6</td><td>56.0</td></tr><tr><td>Deep LSTM Reader †</td><td>55.0</td><td>57.0</td><td>63.3</td><td>62.2</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Attentive Reader †</td><td>61.6</td><td>63.0</td><td>70.5</td><td>69.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Impatient Reader †</td><td>61.8</td><td>63.8</td><td>69.0</td><td>68.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MemNets †</td><td>63.4</td><td>66.8</td><td>-</td><td>-</td><td>70.4</td><td>66.6</td><td>64.2</td><td>63.0</td></tr><tr><td>AS Reader †</td><td>68.6</td><td>69.5</td><td>75.0</td><td>73.9</td><td>73.8</td><td>68.6</td><td>68.8</td><td>63.4</td></tr><tr><td>DER Network †</td><td>71.3</td><td>72.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Stanford AR (relabeling) †</td><td>73.8</td><td>73.6</td><td>77.6</td><td>76.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Iterative Attentive Reader †</td><td>72.6</td><td>73.3</td><td>-</td><td>-</td><td>75.2</td><td>68.6</td><td>72.1</td><td>69.2</td></tr><tr><td>EpiReader †</td><td>73.4</td><td>74.0</td><td>-</td><td>-</td><td>75.3</td><td>69.7</td><td>71.5</td><td>67.4</td></tr><tr><td>AoA Reader †</td><td>73.1</td><td>74.4</td><td>-</td><td>-</td><td>77.8</td><td>72.0</td><td>72.2</td><td>69.4</td></tr><tr><td>ReasoNet †</td><td>72.9</td><td>74.7</td><td>77.6</td><td>76.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>NSE †</td><td>-</td><td>-</td><td>-</td><td>-</td><td>78.2</td><td>73.2</td><td>74.3</td><td>71.9</td></tr><tr><td>MemNets (ensemble) †</td><td>66.2</td><td>69.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>AS Reader (ensemble) †</td><td>73.9</td><td>75.4</td><td>78.7</td><td>77.7</td><td>76.2</td><td>71.0</td><td>71.1</td><td>68.9</td></tr><tr><td>Stanford AR (relabeling,ensemble) †</td><td>77.2</td><td>77.6</td><td>80.2</td><td>79.2</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Iterative Attentive Reader (ensemble) †</td><td>75.2</td><td>76.1</td><td>-</td><td>-</td><td>76.9</td><td>72.0</td><td>74.1</td><td>71.0</td></tr><tr><td>EpiReader (ensemble) †</td><td>-</td><td>-</td><td>-</td><td>-</td><td>76.6</td><td>71.8</td><td>73.6</td><td>70.6</td></tr><tr><td>AS Reader (+BookTest) †‡</td><td>-</td><td>-</td><td>-</td><td>-</td><td>80.5</td><td>76.2</td><td>83.2</td><td>80.8</td></tr><tr><td>AS Reader (+BookTest,ensemble) †‡</td><td>-</td><td>-</td><td>-</td><td>-</td><td>82.3</td><td>78.4</td><td>85.7</td><td>83.7</td></tr><tr><td>GA Reader--</td><td>73.0</td><td>73.8</td><td>76.7</td><td>75.7</td><td>74.9</td><td>69.0</td><td>69.0</td><td>63.9</td></tr><tr><td>GA Reader</td><td>77.9</td><td>77.9</td><td>81.5</td><td>80.9</td><td>74.9</td><td>70.8</td><td>71.8</td><td>69.0</td></tr><tr><td>GA Reader (+feature)</td><td>77.3</td><td>76.9</td><td>80.7</td><td>80.0</td><td>76.8</td><td>72.5</td><td>73.1</td><td>69.6</td></tr></table>

Comparing with prior work, on the WDW dataset the basic version of the GA Reader outperforms all previously published models when trained on the Strict setting. By adding the qe-comm feature the performance increases by  $2.5\%$  and  $4.3\%$  on the Strict and Relaxed settings respectively to set a new state of the art on this dataset. On the CNN and Daily Mail datasets the GA Reader leads to an improvement of  $3.2\%$  and  $4.3\%$  respectively over the best previous single models. They also outperform previous ensemble models, setting a new state of that art for both datasets. For CBT-NE, GA Reader with the qe-comm feature outperforms all previous single and ensemble models except

![](images/c041b81b58c6b2d4878f5364990a18f428327c1c9f83ef7e32e223c7d94a6893.jpg)

![](images/c288aeb04cc1e3e306e15761c2846003202c7b9ac4bb3f2bfbb266ec333e9041.jpg)

![](images/a26bd9ad286711975968a9058aaf3435cf149f960000f5b51ed3cc66be1691d5.jpg)

![](images/b2ebe631435668a0f6a407dd54966ded8ab72a95bc299b63297e69f8d6da45b7.jpg)  
Figure 2: Performance in accuracy with and without the Gated-Attention module over different amounts of training data.  $p$ -values for an exact one-sided McNemar's test are given inside the parentheses for each setting.

the NSE (Munkhdalai & Yu, 2016b), and the AS Reader trained on the much larger BookTest Corpus (Bajgar et al., 2016). Lastly, on CBT-CN the GA Reader with the qe-comm feature outperforms all previously published single models except the NSE.

# 4.4 GATED ATTENTION ANALYSIS

In this section we do an ablation study to see the effect of Gated Attention. We compare the GA Reader as described here to a model which is exactly the same in all aspects, except that it passes document embeddings  $D^{(k)}$  in each layer directly to the inputs of the next layer without using the GA module. In other words  $X^{(k)} = D^{(k)}$  for all  $k > 0$ . This model ends up using only one query GRU at the output layer for selecting the answer from the document. We compare these two variants both with and without the qe-comm feature on CNN and WDW datasets for three subsets of the training data - 50%, 75% and 100%. Test set accuracies for these settings are shown in Figure 2. On CNN when tested without feature engineering, we observe that GA provides a significant boost in performance compared to without GA. When tested with the feature it still gives an improvement, but the improvement is significant only with 100% training data. On WDW-Strict, which is a third of the size of CNN, without the feature we see an improvement when using GA versus without using GA, which becomes significant as the training set size increases. When tested with the feature on WDW, for a small data size without GA does better than with GA, but as the dataset size increases they become equivalent. We conclude that Gated Attention provides a boost in the absence of feature engineering, or as the training set size increases.

Next we look at the question of how to gate intermediate document reader states from the query, i.e. what operation to use in equation 6. Table 4 shows the performance on WDW dataset for three common choices - sum ( $x = d + q$ ), concatenate ( $x = d \| q$ ) and multiply ( $x = d \odot q$ ). Empirically we find that element-wise multiplication does significantly better than the other two, which justifies our motivation to "filter" out document features which are irrelevant to the query.

# 4.5 ATTENTION VISUALIZATION

To gain an insight into the reading process employed by the model we analyzed the attention distributions at intermediate layers of the reader. Figure 3 shows an example from the validation set of WDW dataset (several more are in the Appendix). In each figure, the left and middle plots visualize attention over the query (equation 5) for candidates in the document after layers 1 & 2 respectively. The right plot shows attention over candidates in the document of cloze placeholder (XXX) in the query at the final layer. The full document, query and correct answer are shown at the bottom.

A generic pattern observed in these examples is that in intermediate layers, candidates in the document (shown along rows) tend to pick out salient tokens in the query which provide clues about the cloze, and in the final layer the candidate with the highest match with these tokens is selected as the answer. In Figure 3 there is a high attention of the correct answer on financial regulatory standards in the first layer, and on us president in the second layer. The incorrect answer, in contrast, only attends to one of these aspects, and hence receives a lower score in the final layer despite the n-gram overlap it has with the cloze token in the query. Importantly, different layers tend to

Figure 3: Layer-wise attention visualization of GA Reader trained on WDW-Strict. See text for details.  
![](images/9be60ae2ad733c83c1bb2d1a025d04d19e8c043f3a0b4694ea0cec5c04157478.jpg)  
DOC: japan  s a said frd h wll c for stong monitoring of intnancial at the g20 summit next week in London . we will h ave to emphatly argue that the founal of the intemational monetary fund ( imf ) is weak and that we must establish financial regulations and supervis n , " as oled t alieal sion . other world leaders have also pushed for striter regulatons of risky and unstrained investment practices and instr uments blamed for triggering the current global economic crisis . japan officially agreed in february to lend up to 100 billion dollars to the inf to provide fncial lifelines to emerging economies hit hard by the worldwide downturn . us treasury secretary timothy geithner has said president barack obama would discuss new global financial regulatory standards at the londom summit.  
QRY: <beg> us president barack obama will push higher financial regulatory standards for across the globe at the upcoming g20 summit in london , XXX said thursday <end> ANS: timothy geithner

focus on different tokens in the query, which supports the hypothesis that the multi-hop architecture of GA Reader is able to combine distinct pieces of information to answer the query.

# 5 CONCLUSION

We presented the Gated-Attention reader for answering cloze-style questions over documents. The GA reader features a novel multiplicative gating mechanism, combined with a multi-hop architecture. Our model achieves state-of-the-art performance on several large-scale benchmark datasets with more than  $4\%$  improvements over competitive baselines. Our model design is backed up by an ablation study showing statistically significant improvements of using Gated Attention as information filters. We also showed empirically that multiplicative gating is superior to addition and concatenation operations for implementing gated-attention, though a theoretical justification remains part of future research goals. Analysis of document and query attentions in intermediate layers of the reader further reveals that the model iteratively attends to different aspects of the query to arrive at the final answer. In this paper we have focused on text comprehension, but we believe that the Gated-Attention mechanism may benefit other tasks as well where multiple sources of information interact. Concurrent to our work (Chu et al., 2016) have also shown the effectiveness of GA Readers on the LAMBADA dataset (Paperno et al., 2016) for language modeling.

# REFERENCES

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.  
Ondrej Bajgar, Rudolf Kadlec, and Jan Kleindienst. Embracing data abundance: Booktest dataset for reading comprehension. arXiv preprint arXiv:1610.00956, 2016.  
Danqi Chen, Jason Bolton, and Christopher D Manning. A thorough examination of the cnn/daily mail reading comprehension task. arXiv preprint arXiv:1606.02858, 2016.  
Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.  
Zewei Chu, Hai Wang, Kevin Gimpel, and David McAllester. Broad context language modeling as reading comprehension. arXiv preprint arXiv:1610.08431, 2016.

Yiming Cui, Zhipeng Chen, Si Wei, Shijin Wang, Ting Liu, and Guoping Hu. Attention-over-attention neural networks for reading comprehension. arXiv preprint arXiv:1607.04423, 2016.  
Bhuwan Dhingra, Zhong Zhou, Dylan Fitzpatrick, Michael Muehl, and William W Cohen. Tweet2vec: Character-based distributed representations for social media. ACL, 2016.  
Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems, pp. 1684-1692, 2015.  
Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston. The goldilocks principle: Reading children's books with explicit memory representations. arXiv preprint arXiv:1511.02301, 2015.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Rudolf Kadlec, Martin Schmid, Ondrej Bajgar, and Jan Kleindienst. Text understanding with the attention sum reader network. arXiv preprint arXiv:1603.01547, 2016.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Ryan Kiros, Richard Zemel, and Ruslan R Salakhutdinov. A multiplicative model for learning distributed text-based attribute representations. In Advances in Neural Information Processing Systems, pp. 2348-2356, 2014.  
Sosuke Kobayashi, Ran Tian, Naoaki Okazaki, and Kentaro Inui. Dynamic entity representations with max-pooling improves machine reading. In NAACL-HLT, 2016.  
Ankit Kumar, Ozan Irsoy, Jonathan Su, James Bradbury, Robert English, Brian Pierce, Peter Ondruska, Ishaan Gulrajani, and Richard Socher. Ask me anything: Dynamic memory networks for natural language processing. arXiv preprint arXiv:1506.07285, 2015.  
Peng Li, Wei Li, Zhengyan He, Xuguang Wang, Ying Cao, Jie Zhou, and Wei Xu. Dataset and neural recurrent sequence labeling model for open-domain factoid question answering. arXiv preprint arXiv:1607.06275, 2016.  
Jeff Mitchell and Mirella Lapata. Vector-based models of semantic composition. In ACL, pp. 236-244, 2008.  
Volodymyr Mnih, Nicolas Heess, Alex Graves, et al. Recurrent models of visual attention. In Advances in Neural Information Processing Systems, pp. 2204-2212, 2014.  
Tsendsuren Munkhdalai and Hong Yu. Neural semantic encoders. arXiv preprint arXiv:1607.04315, 2016a.  
Tsendsuren Munkhdalai and Hong Yu. Reasoning with memory augmented neural networks for language comprehension. arXiv preprint arXiv:1610.06454, 2016b.  
Takeshi Onishi, Hai Wang, Mohit Bansal, Kevin Gimpel, and David McAllester. Who did what: A large-scale person-centered cloze dataset. EMNLP, 2016.  
Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context. arXiv preprint arXiv:1606.06031, 2016.  
Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. arXiv preprint arXiv:1211.5063, 2012.  
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pp. 1532-1543, 2014. URL http://www.aclweb.org/anthology/D14-1162.  
Yelong Shen, Po-Sen Huang, Jianfeng Gao, and Weizhu Chen. Reasonet: Learning to stop reading in machine comprehension. arXiv preprint arXiv:1609.05284, 2016.

Alessandro Sordoni, Phillip Bachman, and Yoshua Bengio. Iterative alternating neural attention for machine reading. arXiv preprint arXiv:1606.02245, 2016.  
Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. End-to-end memory networks. In Advances in Neural Information Processing Systems, pp. 2431-2439, 2015.  
Theano Development Team. Theano: A Python framework for fast computation of mathematical expressions. arXiv e-prints, abs/1605.02688, May 2016. URL http://arxiv.org/abs/1605.02688.  
Adam Trischler, Zheng Ye, Xingdi Yuan, and Kaheer Suleman. Natural language comprehension with the epireader. arXiv preprint arXiv:1606.02270, 2016.  
Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. arXiv preprint arXiv:1410.3916, 2014.  
Yuhuai Wu, Saizheng Zhang, Ying Zhang, Yoshua Bengio, and Ruslan Salakhutdinov. On multiplicative integration with recurrent neural networks. arXiv preprint arXiv:1606.06630, 2016.  
Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Learning multi-relational semantics using neural-embedding models. arXiv preprint arXiv:1411.4072, 2014.  
Zhilin Yang, Ruslan Salakhutdinov, and William Cohen. Multi-task cross-lingual sequence tagging from scratch. arXiv preprint arXiv:1603.06270, 2016.
