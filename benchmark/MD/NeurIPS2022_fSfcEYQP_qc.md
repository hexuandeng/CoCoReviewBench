# A Neural Corpus Indexer for Document Retrieval

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Current state-of-the-art document retrieval solutions mainly follow an index-retrieve paradigm, where the index is hard to be optimized for the final retrieval target. In this paper, we aim to show that an end-to-end deep neural network unifying training and indexing stages can significantly improve the recall performance of traditional methods. To this end, we propose Neural Corpus Indexer (NCI), a sequence-to-sequence network that generates relevant document identifiers directly for a designated query. To optimize the recall performance of NCI, we invent a prefix-aware weight-adaptive decoder architecture, and leverage tailored techniques including query generation, semantic document identifiers and consistency-based regularization. Empirical studies demonstrated the superiority of NCI on a commonly used academic benchmark, achieving  $+51.9\%$  relative improvement on  $\mathrm{NQ320k}$  dataset compared to the best baseline.

# 1 Introduction

14 Document retrieval and ranking are two key stages for a standard web search engine [44, 25]. First, the document retrieval stage retrieves candidate documents relevant to the query, and then, the ranking stage gives a more precise ranking score for each document. The ranking stage is often fulfilled by a deep neural network, taking each pair of query and document as input and predicting their relevance score. Nevertheless, a precise ranking model is very costly, while typically only a hundred or thousand candidates per query are affordable in an online system. Therefore, the recall performance of document retrieval stage is very crucial to the effectiveness of web search engine.

Existing document retrieval methods can be divided into two categories, namely term-based and semantic-based approaches [16]. Term-based retrieval approaches [6, 46] build an inverted index for the entire web corpus, but they hardly capture document semantics and fail to retrieve similar documents in different wordings. Thus, semantic-based approaches [44, 27] are proposed to alleviate this discrepancy. First, they learn dense representations for both queries and documents through a twin-tower architecture; then Approximate Nearest Neighbor (ANN) search is applied to retrieve relevant documents for the designated query. Despite of their success in real applications, these approaches can not fully leverage the power of deep neural networks for the following reasons. First, a single embedding vector has limited capacity to memorize all semantics in a document, and it performs even worse than term-based methods in applications that heavily rely on exact match [28]. Second, the model is unable to incorporate deep query-document interactions. Because ANN algorithms theoretically require a strong assumption for the Euclidean space, we have to adopt simple functions such as cosine similarity to capture the query-document interactions [14].

Given the above limitations, several research works have explored end-to-end models that directly retrieve relevant candidates without using an explicit index. Gao et al. [14] proposed a Deep Retrieval (DR) framework for item recommendation, which learned a retrievable structure with historical user-item interactions. Nevertheless, it is more challenging to design a universal model for semantic text retrieval, as we need to leverage the power of both pre-trained language models and deep retrieval

networks simultaneously. Tay et al. [39] proposed Differentiable Search Index (DSI), a text-to-text model that maps queries directly to relevant docids. To the best of our knowledge, this is the first attempt to propose a differentiable index for semantic search. However, the vanilla decoder network in DSI hardly captures the hierarchical semantics of document identifiers, and the model is pruned to over-fitting with limited training data. Furthermore, Bevilacqua et al. [2] proposed SEAL by leveraging all n-grams in a passage as its identifiers. But for long documents, it is hard to enumerate all possible n-grams. In general, the recall performance of end-to-end document retrieval remains a large room to be improved.

In this paper, we show that the traditional text retrieval frameworks can be fundamentally changed by a unified deep neural network with tailored designs. To this end, we propose a Neural Corpus Indexer (NCI), which supports end-to-end document retrieval by a sequence-to-sequence neural network. The model takes user query as input, generates query embedding through the encoder, and outputs the identifiers of relevant documents using the decoder. It can be trained by ground-truth and augmented query-document pairs. During inference, the top  $N$  documents are retrieved directly via beam search based on the decoder. Designing and training such a model is non-trivial, so we propose several crucial techniques to ensure its effectiveness. First, to get sufficient query-document pairs for training, we leverage a query generation network to obtain possible pairs of queries and documents. Second, we utilize the hierarchical  $k$ -means algorithm to generate a semantic identifier for each document. Third, we design a prefix-aware weight-adaptive decoder to replace the vanilla one in a sequence-to-sequence architecture. Specifically, the same token will be assigned different embedding vectors at different positions in the identifiers, while another transformer-based adaptive module is applied to the classification weights for token prediction in the context of a certain prefix. This makes the classifiers customized to different prefixes when decoding along the hierarchical tree structure. Besides, a consistency-based regularization loss is taken for training both encoder and decoder networks to mitigate the over-fitting problem.

Our NCI design solves the limitations of traditional index-retrieve pipelines from multiple perspectives. On one hand, a whole neural network model replaces the traditional inverted index or vector search solutions. It can be optimized end-to-end using realistic query-document pairs, which fully capture both term-based and semantic-based features and is adaptive to the changing of workloads. On the other hand, the model is able to capture deep interactions between queries and documents via encoder-decoder attention, which enlarges the capacity of vector-based representations. Moreover, NCI achieves much better ranking results than ANN-based approaches as it is optimized directly by the final target. Thus, it can be served as an end-to-end retrieval solution and release the burden of re-ranking for a long candidate list. In addition to the superior performance, the invention of Neural Corpus Indexer is also promising from the perspective of system design. As nowadays, ranking and query-answering modules are already implemented by neural networks, NCI finishes the last piece of puzzle for the next-generation information retrieval system based on a unified differentiable model architecture. This reduces the dependency among different sub-modules, while the process of system deployment and maintenance could be greatly eased.

# Our contributions are highlighted as follows.

- For the first time, we demonstrate that an end-to-end differentiable document retrieval model can significantly outperform both inverted index and dense retrieval solutions. This finding will inspire research on further steps towards the next-generation search systems, for instance, unifying informational retrieval, ranking and question answering in a single differentiable framework.  
- We design a sequence-to-sequence model, named Neural Corpus Indexer (NCI), which generates relevant document identifiers directly for a specific query. In our experiments, the proposed NCI model improves the state-of-the-art performance of existing methods by a significant margin, achieving  $+51.9\%$  and  $+19.2\%$  relative enhancement for Recall@1 and Recall@10 respectively on NQ320k dataset. Also, NCI itself can achieve a competitive MRR score without using an explicit ranking model.  
- We propose a novel decoder architecture, namely prefix-aware weight-adaptive (PAWA) decoder, to generate document identifiers. As verified by ablation studies, this invention is very crucial for NCI to achieve an outstanding performance. Moreover, query generation, semantic document identifiers and consistency-based regularization are all accountable for the superior capability of Neural Corpus Indexer.

# 2 Related work

In this section, we briefly introduce the related works and leave more discussions about the traditional web search techniques in the Appendix A.

Sparse retrieval. Traditional document retrieval methods are based on Sparse Retrieval, which is built upon inverted index with term matching metrics such as TF-IDF [35], query likelihood [24] or BM25 [34]. In industry-scale web search, BM25 is a difficult-to-beat baseline owing to its outstanding trade-off between accuracy and efficiency. In recent years, there are some attempts to incorporate the power of neural networks into inverted index. The Standalone Neural Ranking Model (SNRM) [45] learns high-dimensional sparse representations for query and documents, which enables the construction of inverted index for efficient document retrieval. Doc2Query [31] predicts relevant queries to augment the content of each document before building the BM25 index, and DocT5Query [30] improves the performance of query generation by the pre-trained language model T5 [3]. Furthermore, DeepCT [6] calculates context-aware term importance through neural networks to improve the term matching metrics of BM25.

Dense retrieval. Another line of research lies in Dense Retrieval, which presents query and documents in dense vectors and models their similarities with inner product or cosine similarity. These methods benefit from recent progresses of pre-trained language models, such as BERT [11] and RoBERTa [26] to obtain dense representations for queries and documents. At inference time, efficient Approximate Nearest Neighbor (ANN) search algorithms, such as k-dimensional trees [1], locality-sensitive hashing [7], and graph-based indexes (e.g., HNSW [29], DiskANN [21] and SPANN [5]) can be utilized to retrieve relevant documents within a sublinear time. Besides, Luan et al. [28] analyze the limited capacity of dual encoders, and propose a combination of sparse and dense retrieval methods with multi-vector encoding to achieve better search quality.

Autoregressive retrieval. The other way to approach retrieval is utilizing an end-to-end autoregressive models. Firstly, several efforts have been done on entity linking [10, 9, 8], which can be regarded as a special type of retrieval task, e.g., using an entity to ask the posed question. Recently, different from the entity linking task, Tay et al. [39] proposed the DSI (differentiable search index) to generate relevant document identifiers directly according to the query. Bevilacqua et al. [2] employ the autoregressive model to generate the relevant words for a query and utilize the generated string to retrieve relevant documents. Besides, the Deep Retrieval (DR) [14] approach for recommendation is also related to this category, which learns a deep retrievable network with user-item clicks and gets rid of the ANN algorithms based on the Euclidean space assumption.

Pre-trained language models. Recently, pre-trained Language Models (LMs), such as BERT [11] and RoBERTa [26], have led to a revolution in web search techniques. The representation vectors for all documents can be calculated and indexed offline. In the online serving stage, it calculates the representation vector for the input query and applies a crossing layer to calculate the relevance score between each query and document. The crossing layer usually adopts simple operators such as cosine similarity or a single feed-forward layer to retain a high efficiency. Gao et al. [12] find that a standard LMs' internal attention structure is not ready-to-use for dense encoders and propose the Condenser to improve the performance of dense retrieval. Moreover, ANCE [43] leverages hard negatives to improve the effectiveness of contrastive learning, which generates better text representations for the retrieval tasks.

# 3 Neural corpus indexer

The neural corpus indexer (NCI) is a sequence-to-sequence neural network model. The model takes query as input and outputs the most relevant document identifier (docid), which can be trained by a large collection of <query, docid> pairs. The documents are encoded into semantic docids by the hierarchical  $k$ -means algorithm [17], which makes similar documents have "close" identifiers in the hierarchical tree. As shown in Figure 1, NCI is composed of three components, including Query Generation, Encoder and Prefix-Aware Weight-Adaptive (PAWA) Decoder. Query generation is implemented by a sequence-to-sequence transformer model [41] that takes as an input the document terms and produces a query as output [31]. The encoder, following the standard transformer architecture, is composed of  $N_{1}$  stacked transformer blocks, which outputs the representation for an input query. For the decoder network, we stack  $N_{2}$  transformer layers. To better align with the hierarchical nature of

![](images/20aebdc9d22f9d0bfdd74b2b5e93d66fd33372657a88b59b7e1c42c10887527d.jpg)  
(a) Preprocessing

![](images/4f089d1a123cdea298f92267928941316db69d32b0a053b241d6fe643faabdaf.jpg)  
Figure 1: Overview of Neural Corpus Indexer (NCI). (a) Preprocessing. Each document is represented by a semantic identifier via hierarchical  $k$ -means. (b) Query Generation. Queries are generated for each document based on the content. (c) The training pipeline of NCI. The model is trained over augmented <query, docid> pairs through a standard transformer encoder and the proposed Prefix-Aware Weight-Adaptive (PAWA) Decoder.  
(c) Training pipeline of Neural Corpus Indexer

the semantic identifiers, we propose a weight adaptation mechanism based on another transformer to make the decoder aware of semantic prefixes. At inference time, the top  $N$  relevant documents can be easily obtained via beam search. Due to the hierarchical property of semantic identifiers, it is relatively easy to constrain the beam search decoding process on the prefix tree so that only valid identifiers will be generated.

# 3.1 Representing document with semantic identifiers

NCI generates document identifiers solely based on the input query without explicit document content, which is difficult when the size of the corpus is very large. Thus, we aim to inject useful priors into the identifiers, so that the semantic information of documents can be considered in the tree-based decoding process. In other words, we hope the documents with similar information have close docids to facilitate the learning process of NCI. To achieve this, we leverage the hierarchical  $k$ -means algorithm to encode documents. As shown in Figure 1(a), given a collection of documents to be indexed, all documents are first classified into  $k$  clusters. For each cluster with more than  $c$  documents, the  $k$ -means algorithm is applied recursively. For each cluster containing  $c$  documents or less, each document is assigned a number starting from 0 to at most  $c - 1$ . In this way, we organize all documents into a tree structure  $T$  with root  $r_0$ . Each document is associated with one leaf node with a deterministic routing path  $l = \{r_0, r_1, \dots, r_m\}$  from the root, where  $r_i$  represents for an internal node for level  $i$ , and  $r_m$  is the leaf node. For simplicity, we set  $k = 10$  and  $c = 10$  in all experiments, leaving the optimization of these hyper-parameters to future work. The detailed procedure of hierarchical  $k$ -means will be described in Algorithm 1 in the appendix.

# 3.2 Query generation

One challenge of generating document identifiers by single query input is how to make the identifiers aware of the document semantics. Since the content of each document is not explicitly known at inference, it must be incorporated into the model parameters during training. To facilitate the training process, we generate a bunch of queries with a query generation module and bind the information of document content through training the sequence-to-sequence model with generated queries and their document identifiers. We adopt a standard sequence-to-sequence transformer [41] based on the implementation of Doc2Query<sup>1</sup>, which takes as an input the document terms and produces relevant queries via random sampling. Note that we use random sampling instead of beam search to ensure the diversity of generated queries.

# 3.3 Prefix-aware weight-adaptive decoder

The probability of generating a document identifier can be written as follows:

$$
p (l | x, \theta) = \prod_ {i = 1} ^ {m} p \left(r _ {i} \mid x, r _ {1}, r _ {2}, \dots , r _ {i - 1}, \theta_ {i}\right), \tag {1}
$$

where  $r_i$  is the  $i$ -th token in the current identifier;  $x$  is the representation output from encoder;  $\theta$  denotes the total parameters and  $\theta_i$  is the parameter for the  $i$ -th step.

This probability can be modeled by a transformer-based decoder. For an internal node with level  $i$ , the probability is calculated by:

$$
h _ {i} = \operatorname {T r a n s f o r m e r D e c o d e r} \left(x, h _ {1}, h _ {2}, h _ {i - 1}; \theta_ {i}\right), \tag {2}
$$

$$
p \left(r _ {i} \mid x, r _ {1}, r _ {2}, \dots , r _ {i - 1}, \theta_ {i}\right) = \operatorname {S o f t m a x} \left(h _ {i} W\right). \tag {3}
$$

Here  $h_i$  is the hidden representation for step  $i$ , which is calculated by a multi-head attention over encoder representation  $x$  and token representations of previous decoding steps. The linear classification weight is denoted by  $W \in \mathbb{R}^{d \times v}$ ,  $d$  is the hidden dimension size and  $v$  is the vocabulary size of identifiers.

![](images/b95f7b530f03988c54b357dda8ec15300ef996c77f171bab662384a088c6f080.jpg)  
Figure 2: Overview of the Prefix-Aware Weight-Adaptive (PAWA) Decoder.

As the encoder and decoder utilize distinct vocabulary spaces, we do not share the embedding space for their tokens. Different from a standard decoding task, the meanings of the same token appearing at different places of the same identifier are different, as they correspond to different clusters in the hierarchical tree structure. For instance, the “ $5_{2}$ ” and “ $5_{3}$ ” of the same identifier “ $3_{1}5_{2}5_{3}$ ” correspond to different semantic meanings. Moreover, the same token in the same position may have different semantics with different prefixes. For example, in identifiers “ $1_{1}1_{2}5_{3}$ ” and “ $2_{1}4_{2}5_{3}$ ”, the same token “ $5_{3}$ ” has different semantics in two different identifiers, as they are routed from different prefix paths. The two properties of the hierarchical semantic identifiers motivate us to design the novel Prefix-Aware Weight-Adaptor (PAWA) decoder.

Unlike a standard transformer decoder, the probabilities at different tree levels, such as \( p(r_i|x,r_{1..i-1},\theta_i) \) and \( p(r_j|x,r_{1..j-1},\theta_j) \) when \( i \neq j \), do not share parameters with each other. To distinguish different semantic levels, we concatenate the position and token values as input for each decoding step, as shown in the left corner of Figure 2. Specifically, we have "\( (1,3)(2,5)(3,5)" \) for the semantic identifier "3\u5\u5\u3", while "\( (2,5)" \) and "\( (3,5)" represent different tokens in the vocabulary space. As the token embedding and linear classification layers share the same weights, the same token value in different positions would correspond to different model parameters. Moreover, to reflect the influence of different prefixes, we expect the linear classification layer to be aware of different prefixes for predicting a specific token. Concretely, instead of using the same projection weight \( W \) in the linear classification layer, we employ the prefix-aware adaptive weights for each token classifier, which can be calculated by another transformer decoder,

$$
W _ {a d a} ^ {i} = \text {A d a p t i v e D e c o d e r} (e; r _ {1}, r _ {2}, \dots , r _ {i - 1}) W _ {i} \tag {4}
$$

where  $e$  is the query embedding vector taken as initial input to the transformer decoder;  $\{r_t|t\in (1,2,\dots i - 1)\}$  are prefix tokens before the  $i$ -th position, AdaptiveDecoder stacks  $N_{3}$  transformer decoding layers with dimension  $d$ , and  $W_{ada}^{i}\in \mathbb{R}^{d\times v}$  is the adapted weight matrix for the corresponding classifier. Finally, the  $i$ -th token in the given prefix can be predicted by Softmax  $(h_iW_{ada}^i)$ .

For instance, to predict the third tokens in the identifiers “ $(1,3)(2,1)(3,5)$ ” and “ $(1,2)(2,4)(3,5)$ ” respectively, the corresponding adaptive weights are derived separately for different prefixes “ $(1,3)(2,1)$ ” and “ $(1,2)(2,4)$ ”. As we already know the previous tokens for each position in the teacher forcing setting, the prefix-aware adaptive weights can be calculated and trained in parallel in different positions while adding little burden to the entire model.

# 3.4 Training and inference

Consistency-based regularization. To alleviate over-fitting, we employ a consistency-based regularization loss for training each decoding step. Given an input query  $q$ , we denote the model probabilities predicted by two independent forward pass as  $p_1(r_i|E(q), r_{1,\dots,i-1}, \theta_i)$  and  $p_r(r_i|E(q), r_{1,\dots,i-1}, \theta_i)$  respectively, where  $E(\cdot)$  denotes the encoder network. The consistency-based regularization loss tries to regularize the model prediction by minimizing the bidirectional Kullback-Leibler (KL) Divergence between two output probabilities with random dropout. The regularization loss of query  $q$  for the  $i$ -th decoding step is defined as,

$$
\begin{array}{l} \mathcal {L} _ {r e g} ^ {i} = \frac {1}{2} \left(\sum_ {i = 1} ^ {m} \mathcal {D} _ {K L} \left(p _ {1} \left(r _ {i} \mid E (q), r _ {1, \dots , i - 1}, \theta_ {i}\right) \| p _ {2} \left(r _ {i} \mid E (q), r _ {1, \dots , i - 1}, \theta_ {i}\right)\right) \right. \tag {5} \\ \left. + \mathcal {D} _ {K L} \left(p _ {2} (r _ {i} | E (q), r _ {1, \dots , i - 1}, \theta_ {i}) \| p _ {1} (r _ {i} | E (q), r _ {1, \dots , i - 1}, \theta_ {i})\right)\right). \\ \end{array}
$$

Training loss. Given a set of training examples  $\mathcal{D} = \{(q,d)\}$  composed of queries (training queries and augmented queries) and document identifiers, the loss function can be written as follows:

$$
\mathcal {L} (\theta) = \sum_ {(q, d) \in \mathcal {D}} \left(\log p (d | E (q), \theta) + \alpha \sum_ {i} \cdot \mathcal {L} _ {r e g} ^ {i}\right), \tag {6}
$$

where  $p(d|E(q),\theta)$  denotes the probability of generating  $d$  with  $q$  as the input. The first part is the seq2seq cross-entropy loss with teacher forcing and the second part is the consistency-based regularization loss summed by all decoding steps. The whole process formulates a sequence-to-sequence neural network, which can be optimized end-to-end via gradient descent. The hyperparameter  $\alpha$  denotes a scaling factor of regularization loss, which will be analyzed in Section 4.4.

Inference via beam search. In the inference stage, we calculate the query embedding through the encoder network and then perform the beam search on the decoder network. Due to the hierarchical nature of docid, it is convincing to constrain the beam search decoding process with a prefix tree, which in turn only generates the valid identifiers. The time complexity of beam search is  $O(LBF)$ , where  $L$  is the max length of identifiers (the depth of tree),  $B$  is the beam size and  $F$  is the max fanout of the tree (10 in our experiments). Given a balanced tree structure built by a corpus with  $M$  documents, the average time complexity for beam search is  $O(B\log M)$ . We leave detailed descriptions of the constrained beam search algorithm in Appendix B.

# 4 Experiments

In this section, we empirically verify the performance of NCI and the effectiveness of each component on the document retrieval task, which generates a ranking list of documents in response to a query. In the following, we discuss the datasets and evaluation protocol in Section 4.1, describe the implementation details and baseline methods in Section 4.2, and present empirical results and analyses in Section 4.3 and 4.4 respectively.

# 4.1 Datasets & evaluation metrics

Datasets. Following DSI [39] and SEAL [2], we conduct our experiments on the Natural Questions [23] dataset. Natural Questions (NQ) dataset was introduced by Google in 2019 [23]. The version we use is often referred to as NQ320k, which consists of  $320k$  query-document pairs, where the documents are gathered from Wikipedia pages and the queries are natural language questions. We use its predetermined training and validation split for evaluation.

Metrics. We use widely accepted metrics for information retrieval, including Recall@N and Mean Reciprocal Rank (MRR). Recall@N measures how often the desired document is hit by the top-N retrieved candidates. MRR calculates the reciprocal of the rank at which the first relevant document is retrieved. A high recall means that the ground truth document is contained in the retrieved candidate list, while a high MRR indicates that the corresponding document has already been ranked at the top position without a need for re-ranking.

# 4.2 Implementation details

Hierarchical semantic identifier. For semantic identifiers, we apply hierarchical  $k$ -means algorithm over the document embeddings obtained through a 12 layers BERT model with pre-trained parameters provided by the HuggingFace [42]. For each hierarchical layer, we employ the default  $k$ -means algorithm implemented in scikit-learn [32] with  $k = 10$ . For simplicity, the recursion terminal condition is also set as  $c = 10$ .

Query generation. We leverage the pre-trained model docT5query [30] for query generation. We provide all documents in the NQ320k dataset to predict augmented query-document pairs. For each document, we generate 10 queries with the first 512 input tokens of the document as the input and constrain the maximum length of the generated query as 64.

Training and inference. Neural Corpus Indexer (our approach) are implemented with python 3.6.10, PyTorch 1.8.1 and HuggingFace transformers 3.4.0. We utilize the parameters of the T5 pre-trained model [3] to initialize the encoder and randomly initialize the PAWA decoder. All NCI experiments are based on a learning rate  $2 \times 10^{-4}$  for encoder and  $1 \times 10^{-4}$  for decoder with a batch size 16 per

Table 1: Performance comparison on NQ320k retrieval task  

<table><tr><td>Method</td><td>Recall@1</td><td>Recall@10</td><td>Recall@100</td><td>MRR@100</td></tr><tr><td>Neural Corpus Indexer (Ours)</td><td>88.72</td><td>95.84</td><td>97.43</td><td>91.59</td></tr><tr><td>DSI (T5-Base)</td><td>27.40</td><td>56.60</td><td>-</td><td>-</td></tr><tr><td>DSI (T5-XXL)</td><td>40.40</td><td>70.30</td><td>-</td><td>-</td></tr><tr><td>SEAL (BART-Base)</td><td>26.55</td><td>53.61</td><td>72.67</td><td>35.64</td></tr><tr><td>ANCE (FirstP)</td><td>51.33</td><td>80.33</td><td>91.78</td><td>61.71</td></tr><tr><td>ANCE (MaxP)</td><td>52.63</td><td>80.38</td><td>91.31</td><td>62.84</td></tr><tr><td>BERT + BruteForce</td><td>28.65</td><td>53.42</td><td>73.16</td><td>36.60</td></tr><tr><td>BERT + ANN (Faisss)</td><td>27.92</td><td>53.63</td><td>73.01</td><td>37.08</td></tr><tr><td>BM25 + DocT5Query</td><td>58.39</td><td>75.76</td><td>89.51</td><td>64.53</td></tr><tr><td>BM25</td><td>30.23</td><td>47.02</td><td>68.54</td><td>36.26</td></tr></table>

GPU. We set the scaling factor of the consistency-based regularization loss as  $\alpha = 0.015$ , and the dropout ratio is 0.1. For inference, we apply the partial beam search algorithm to the trained seq2seq model. We set the length penalty and the beam size as 0.3 and 100 respectively. All experiments are based on a cluster of NVIDIA V100 GPUs with 32GB memory. Each job takes 8 GPUs, resulting in a total batch size of 128 ( $16 \times 8$ ).

Baselines. We evaluate BM25 on both raw documents and those augmented by DocT5Query. The performance of DSI [38] is referred from its original paper as the implementation has not been open-sourced. To avoid the difference in data processing, we reproduce SEAL [2] and ANCE [43] by their official implementations. We leave the detailed settings in Appendix C.

# 4.3 Results

In Table 1, we compare the empirical results of NCI and corresponding baselines. On the NQ320k dataset, the proposed NCI model outperforms all baselines by a large margin across four different metrics. Compared with the state-of-the-art model, NCI improves  $51.9\%$  on Recall@1,  $19.2\%$  on Recall@10,  $6.2\%$  on Recall@100, and  $41.9\%$  on MRR@100 relatively. It is worth noting that we are the first to verify the superiority of deep text retrieval over traditional sparse and dense retrieval methods. Previous deep text retrieval methods (i.e., DSI and SEAL) obtain relatively poor results even with a very large model size (e.g., T5-XXL). Consistent with previous studies, BM25 is an efficient and effective baseline. It even outperforms BERT-based dense retrieval solutions, perhaps owning to its capability to retrieve precise documents based on exact match. Further, we notice that query generation plays a key role in boosting the retrieval performance. With query generation, the BM25 + DocT5Query method achieves much higher performance than its vanilla version. ANCE also achieves competitive performance after fine-tuned by the training pairs, but the performance is far lower than our proposed NCI model. Moreover, the Recall@1 and MRR@100 metrics of NCI are outstanding, indicating that more than  $90\%$  of the queries can be fulfilled without re-ranking on the retrieved document list. This shows the potential of NCI to be served as an end-to-end solution that replaces the entire index-retrieve-rank pipeline in traditional web search engines.

Furthermore, to study the effect of each component, we report ablation results on NQ320k dataset in Table 2. In general, all five components are able to improve the performance of document retrieval, which are detailed below.

w/o query generation. This configuration removes the query generation module for data augmentation. Remarkably, the query generation boosts the performance greatly. The result is aligned with our expectation because training with the generated queries allows the model to be agnostic to the

Table 2: Ablation Study on NQ320k retrieval task  

<table><tr><td>Method</td><td>Recall@1</td><td>Recall@10</td><td>Recall@100</td><td>MRR@100</td></tr><tr><td>Neural Corpus Indexer (Ours)</td><td>88.72</td><td>95.84</td><td>97.43</td><td>91.59</td></tr><tr><td>w/o query generation</td><td>53.63</td><td>67.84</td><td>78.43</td><td>59.16</td></tr><tr><td>w/o PAWA decoder</td><td>87.01</td><td>95.27</td><td>97.18</td><td>90.79</td></tr><tr><td>w/o semantic id</td><td>87.22</td><td>95.34</td><td>97.25</td><td>90.85</td></tr><tr><td>w/o regularization</td><td>87.34</td><td>95.42</td><td>97.27</td><td>90.89</td></tr><tr><td>w/o constrained beam search</td><td>87.41</td><td>95.71</td><td>97.32</td><td>90.84</td></tr></table>

semantic meaning of each documents. Besides, although training on  $\langle \text{doc-content}, \text{docid} \rangle$  pairs like DSI [38] also make the model aware of the semantic meaning of each documents, we argue that training with generated queries is able to avoid the distribution shift problem, which also benefit the generalization performance.

w/o PAWA decoder. This configuration removes the adaptive decoder layer in Equation (4) and leverages shared weights with token embedding for the linear classification layer. We notice that the prefix-aware weight-adaptive decoder has a noticeable influence on the performance, which indicates that, instead of borrowing the vanilla transformer decoder, it is necessary to design a tailored decoder architecture for the task of semantic identifier generation.

w/o semantic id. This configuration replaces the semantic identifier of each document to a random generated one. We find a relative drop in the model performance on all four metrics, demonstrating that the semantic identifiers derived by hierarchical  $k$ -means have injected useful priors. We conjecture that the performance enhancement would be more significant on a larger document corpus.

w/o regularization. There is a performance drop on all four metrics without using consistency-based regularization loss. The reason is that the decoder network is prone to over-fitting. By making the prediction results for two augmented versions of the decoder to be consistent, the decoder model becomes more generalizable and resistant to over-fitting.

w/o constrained beam search. This configuration disables the validating constraint in beam search. In other words, the decoder network does not have a tree-based prior structure. Instead, all tokens in the vocabulary can be generated in each decoding step. We observe a performance drop on four evaluation metrics. This indicates that it is difficult to remember all information of valid identifiers in the network, and an explicit prior could be helpful for improving the quality of beam search.

![](images/a731b387ee3272ae6780080df2a7cb50f2a2caa1056c3f8f1fe616ef7c3503b8.jpg)  
Figure 3: Learning curves of NCI with different model capacities

Table 3: NCI with different number of layers in PAWA adapter and different regularization hyper-parameter  $\alpha$  in loss function  

<table><tr><td>Setting</td><td>Recall@1</td><td>Recall@10</td><td>Recall@100</td><td>MRR@100</td></tr><tr><td>#layer = 0</td><td>87.01</td><td>95.27</td><td>97.18</td><td>90.79</td></tr><tr><td>#layer = 1</td><td>88.54</td><td>95.62</td><td>97.16</td><td>91.44</td></tr><tr><td>#layer = 2</td><td>88.56</td><td>95.67</td><td>97.28</td><td>91.48</td></tr><tr><td>#layer = 4</td><td>88.65</td><td>95.72</td><td>97.54</td><td>91.51</td></tr><tr><td>#layer = 6</td><td>88.72</td><td>95.84</td><td>97.43</td><td>91.59</td></tr><tr><td>#layer = 8</td><td>85.31</td><td>94.17</td><td>96.34</td><td>89.25</td></tr></table>

# 4.4 Analysis

Model capacity. Figure 3 compares the learning curves of NCI with different model capacities, which are identical to the small, base, and large settings of ordinary T5 [33]. We observe that with the increase of model size, NCI convergences more quickly with fewer epochs. At convergence, the small model achieves a relatively lower recall@1. Instead, both the base and large models achieve similar results after sufficient training epochs. This implies that the model capacity has a critical impact on the retrieval performance, and the capacity of base model seems to be enough to memorize all documents in  $\mathrm{NQ320k}$  dataset. For a larger corpus, one may need to increase the model size to obtain satisfactory performance.

Layer number of PAWA adapter. We study the influence of the number of transformer layers in the PAWA adapter and choose the layer number from  $\{0,1,2,4,6,8\}$ . The results are summarized in Table 3. We notice that with the increasing of layer number, i.e. from 0 to 6, the overall performance is consistently improved under four metrics, except the Recall@100. But when the number of layer achieves 8, the performance is dropped significantly. We attribute that to the overfitting caused by a large PAWA adapter. Therefore, we adopt the design with 6 layer adapter in NCI.

Retrieved documents and their semantics identifiers. To verify the effectiveness of retrieval as well as the semantic identifiers learned by hierarchical  $k$ -means, we analyze the retrieval results of NCI for some exemplar queries. To illustrate, we select four queries denoted by  $A-1, A-2, B-1$  and  $B-2$ , where two queries inside the same group are semantically similar, and the queries in different groups correspond to distinct topics. In Figure 4(a) and 4(b), we show the probabilities of retrieved

![](images/830d633a605e98b209447ff37d883b8661023a7329a994eb90f1c5226d3f4977.jpg)  
(a) Query Group A

![](images/90cfdcd525a7263a7a1946c6dc0f0cc15dae2c82828a9fdbf20af9dcbb923679.jpg)  
Figure 4: Analyses of retrieved documents with semantic identifiers. (a) The probabilities of retrieved documents for Query Group A; (b) Query Group B. (c) The t-SNE visualization of BERT-based document embeddings.  
(b) Query Group B

![](images/0e661a5123ece0e27a54aab71cffc58c001a873d3098fa24d244599172904b7b.jpg)  
(c) Document Embeddings

documents for each query in group  $A$  and  $B$  respectively. The digits along x-axis denote the four-bit prefixes for semantic identifiers of retrieved documents, and the y-axis stands for their probabilities. We notice that similar queries result in close document distributions, while dissimilar queries in different groups result in un-overlapped document collections. In addition, the documents retrieved by the same group of queries have close prefixes for the identifiers, e.g., 6030, 6032, 6033, 6034 in group  $A$  and 7511, 7514, 7516 in group  $B$ . Also, we visualize BERT-based document embeddings by t-SNE [40] in Figure 4(c), in which each color represents the corresponding documents for a specific query. As shown in the figure, these documents naturally form two clusters with respect to different query groups. Thus, we conclude that the semantic document identifiers generated by the hierarchical  $k$ -means algorithm have positive effects on the retrieval performance.

Efficiency Analysis. We use an NVIDIA V100-32G GPU to analyze the efficiency of NCI. As the inference speed is influenced by both model capacity and beam size, we report the latency and throughput measures for multiple settings in Table 4. As NCI is an end-to-end retrieval method and achieves competitive performance without re-ranking, the latency and throughput are already affordable for some near-real-time applications. Furthermore, we can leverage other techniques to improve the efficiency of NCI, which will be discussed in the later section.

# 5 Limitation & Future Works

Despite the significant breakthrough, the current implementation of NCI still suffers from several limitations before deployment in an industrial web search system. Firstly, it requires a much larger model capacity for extending NCI to the web scale. Secondly, its inference speed needs to be improved to serve online queries in real time. Thirdly, it is difficult to update the model-based index when new documents are added to the system. In future works, we may tackle these problems from four aspects. (1) The architecture of sparsely-gated Mixture of Expert (MoE) [36] can be employed to enhance the model capacity. (2) Documents can be grouped into semantic clusters, and then NCI is used to retrieve relevant cluster identifiers. In this way, all documents in relevant clusters can be retrieved efficiently. (3) Model compression techniques, like weight quantization [20] and knowledge distillation [18], can be further taken to speed up inference. (4) We plan to explore a hybrid solution by building another index that serves new documents through traditional indexing algorithms.

# 6 Conclusion

In this work, we introduce a novel learning paradigm that unifies the learning and indexing stages by an end-to-end deep neural network framework. The proposed Neural Corpus Indexer (NCI) retrieves the identifiers of relevant documents directly for an input query, which can be optimized end-to-end with augmented query-document pairs. To optimize the recall and ranking performance, we invent the tailored prefix-aware weight-adaptive decoder. Empirically, we evaluate NCI on  $\mathrm{NQ320k}$  dataset and demonstrate its outstanding recall and MRR performance over state-of-the-art solutions.

Table 4: Efficiency analysis  

<table><tr><td>Model size</td><td>Beam size</td><td>Latency (ms)</td><td>Throughput (queries / s)</td></tr><tr><td>Small</td><td>10</td><td>78.46</td><td>58.48</td></tr><tr><td>Base</td><td>10</td><td>115.17</td><td>52.55</td></tr><tr><td>Large</td><td>10</td><td>188.60</td><td>43.39</td></tr><tr><td>Small</td><td>100</td><td>216.01</td><td>6.12</td></tr><tr><td>Base</td><td>100</td><td>269.31</td><td>5.62</td></tr><tr><td>Large</td><td>100</td><td>356.07</td><td>4.75</td></tr></table>

# References

[1] Jon Louis Bentley. Multidimensional binary search trees used for associative searching. Communications of the ACM, 18(9):509-517, 1975.  
[2] Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Wen-tau Yih, Sebastian Riedel, and Fabio Petroni. Autoregressive search engines: Generating substrings as document identifiers. arXiv preprint arXiv:2204.10628, 2022.  
[3] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.  
[4] Wei-Cheng Chang, X Yu Felix, Yin-Wen Chang, Yiming Yang, and Sanjiv Kumar. Pre-training tasks for embedding-based large-scale retrieval. In International Conference on Learning Representations, 2019.  
[5] Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong Li, Mao Yang, and Jingdong Wang. Spann: Highly-efficient billion-scale approximate nearest neighbor search. arXiv preprint arXiv:2111.08566, 2021.  
[6] Zhuyun Dai and Jamie Callan. Context-aware sentence/passage term importance estimation for first stage retrieval. arXiv preprint arXiv:1910.10687, 2019.  
[7] Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab S Mirrokni. Locality-sensitive hashing scheme based on p-stable distributions. In Proceedings of the twentieth annual symposium on Computational geometry, pages 253-262, 2004.  
[8] Nicola De Cao, Wilker Aziz, and Ivan Titov. Highly parallel autoregressive entity linking with discriminative correction. arXiv preprint arXiv:2109.03792, 2021.  
[9] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. Autoregressive entity retrieval. arXiv preprint arXiv:2010.00904, 2020.  
[10] Nicola De Cao, Ledell Wu, Kashyap Popat, Mikel Artetxe, Naman Goyal, Mikhail Plekhanov, Luke Zettlemoyer, Nicola Cancedda, Sebastian Riedel, and Fabio Petroni. Multilingual autoregressive entity linking. arXiv preprint arXiv:2103.12528, 2021.  
[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), 2019.  
[12] Luyu Gao and Jamie Callan. Condenser: a pre-training architecture for dense retrieval. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 981-993, 2021.  
[13] Luyu Gao, Zhuyun Dai, and Jamie Callan. Coil: Revisit exact lexical match in information retrieval with contextualized inverted list. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3030-3042, 2021.  
[14] Weihao Gao, Xiangjun Fan, Chong Wang, Jiankai Sun, Kai Jia, Wenzhi Xiao, Ruofan Ding, Xingyan Bin, Hui Yang, and Xiaobing Liu. Deep retrieval: Learning a retrievable structure for large-scale recommendations. arXiv preprint arXiv:2007.07203, 2020.  
[15] Jiafeng Guo, Yixing Fan, Qingyao Ai, and W Bruce Croft. A deep relevance matching model for ad-hoc retrieval. In Proceedings of the 25th ACM international on conference on information and knowledge management, pages 55-64, 2016.  
[16] Tonglei Guo, Jiafeng Guo, Yixing Fan, Yanyan Lan, Jun Xu, and Xueqi Cheng. A comparison between term-based and embedding-based methods for initial retrieval. In China Conference on Information Retrieval, pages 28-40. Springer, 2018.

[17] John A Hartigan and Manchek A Wong. Algorithm as 136: A k-means clustering algorithm. Journal of the royal statistical society. series c (applied statistics), 28(1):100-108, 1979.  
[18] Geoffrey Hinton, Oriol Vinyals, Jeff Dean, et al. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2(7), 2015.  
[19] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management, pages 2333-2338, 2013.  
[20] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2704-2713, 2018.  
[21] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar Krishnawamy, and Rohan Kadekodi. Diskann: Fast accurate billion-point nearest neighbor search on a single node. Advances in Neural Information Processing Systems, 32, 2019.  
[22] Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39-48, 2020.  
[23] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453-466, 2019.  
[24] John Lafferty and Chengxiang Zhai. Document language models, query models, and risk minimization for information retrieval. In Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval, pages 111-119, 2001.  
[25] Canjia Li, Andrew Yates, Sean MacAvaney, Ben He, and Yingfei Sun. Parade: Passage representation aggregation for document reranking. arXiv preprint arXiv:2008.09093, 2020.  
[26] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT pretraining approach. CoRR, abs/1907.11692, 2019.  
[27] Wenhao Lu, Jian Jiao, and Ruofei Zhang. Twinbert: Distilling knowledge to twin-structured compressed bert models for large-scale retrieval. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pages 2645–2652, 2020.  
[28] Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. Sparse, dense, and attentional representations for text retrieval. Transactions of the Association for Computational Linguistics, 9:329-345, 2021.  
[29] Yu A Malkov and Dmitry A Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence, 42(4):824-836, 2018.  
[30] Rodrigo Nogueira, Jimmy Lin, and AI Epistemic. From doc2query to doctttttquery. Online preprint, 2019.  
[31] Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. Document expansion by query prediction. arXiv preprint arXiv:1904.08375, 2019.  
[32] Fabian Pedregosa, Gáel Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in python. the Journal of machine learning research, 12:2825-2830, 2011.

[33] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.  
[34] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. Now Publishers Inc, 2009.  
[35] Stephen E Robertson and Steve Walker. On relevance weights with little relevance information. In Proceedings of the 20th annual international ACM SIGIR conference on Research and development in information retrieval, pages 16-24, 1997.  
[36] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. International Conference on Learning Representations (ICLR), 2017.  
[37] Yelong Shen, Xiaodong He, Jianfeng Gao, Li Deng, and Grégoire Mesnil. Learning semantic representations using convolutional neural networks for web search. In Proceedings of the 23rd international conference on world wide web, pages 373-374, 2014.  
[38] Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, and Che Zheng. Synthesizer: Rethinking self-attention in transformer models. arXiv preprint arXiv:2005.00743, 2020.  
[39] Yi Tay, Vinh Q Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al. Transformer memory as a differentiable search index. arXiv preprint arXiv:2202.06991, 2022.  
[40] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.  
[41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), pages 5998-6008, 2017.  
[42] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumont, Clement Delangue, Anthony Moi, Pierrick Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface's transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771, 2019.  
[43] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In International Conference on Learning Representations, 2020.  
[44] Wei Yang, Haotian Zhang, and Jimmy Lin. Simple applications of bert for ad hoc document retrieval. arXiv preprint arXiv:1903.10972, 2019.  
[45] Hamed Zamani, Mostafa Dehghani, W Bruce Croft, Erik Learned-Miller, and Jaap Kamps. From neural re-ranking to neural ranking: Learning a sparse representation for inverted indexing. In Proceedings of the 27th ACM international conference on information and knowledge management, pages 497-506, 2018.  
[46] Shengyao Zhuang, Hang Li, and G. Zuccon. Deep query likelihood model for information retrieval. In ECIR, 2021.
