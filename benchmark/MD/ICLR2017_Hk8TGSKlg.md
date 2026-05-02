# REASONING WITH MEMORY AUGMENTED NEURAL NETWORKS FOR LANGUAGE COMPREHENSION

Tsendsuren Munkhdalai & Hong Yu

University of Massachusetts

{tsendsuren.munkhdalai,hong.yu}@umassmed.edu

# ABSTRACT

Hypothesis testing is an important cognitive process that supports human reasoning. In this paper, we introduce a new computational hypothesis testing framework that is based on memory augmented neural networks. Our approach involves a hypothesis testing loop that reconsiders and progressively refines a previously formed hypothesis in order to generate new hypotheses to test. We apply the proposed approach to language comprehension task by using Neural Semantic Encoders (NSE). Our NSE models achieve the state-of-the-art results showing an absolute improvement of  $1.2\%$  to  $2.6\%$  accuracy over previous results obtained by single and ensemble systems on standard machine comprehension benchmarks such as the Children's Book Test (CBT) and Who-Did-What (WDW) news article datasets.

# 1 INTRODUCTION

Formulating new hypotheses and testing them is a cognitive process that supports human reasoning and intelligence. This hypothesis testing process involves selective attention, working memory and cognitive control (Just and Carpenter, 1992; Polk and Seifert, 2002). Attention and working memory are engaged in order to maintain, manipulate, and update new hypotheses. Cognitive control is required to inspect and ignore incorrect hypotheses. Inspired by the hypothesis testing process in the human brain and to support dynamic reasoning of machines, in this work, we introduce a reasoning approach that is based on memory augmented neural networks (MANN). A new hypothesis is formed by regressing the original statement (i.e. query in context of QA). Then the hypothesis is tested against reality (i.e. data or document story). If the model is satisfied with the current test response or the hypothesis is true, the reasoning process is halted and the answer is found. Otherwise, another hypothesis is formulated by refining the previous one and the process is repeated until the answer is found.

While the idea of modeling hypothesis testing with MANN remains a generic reasoning framework and is applicable to several AI tasks, we apply this approach to cloze-type QA by using Neural Semantic Encoders (NSE). NSE is a flexible MANN architecture and has shown a notable success on several language understanding tasks ranging from sentence classification to language inference and machine translation (Munkhdalai and Yu, 2016). NSE has read, compose and write modules to manipulate external memories and it has introduced a concept of shared and multiple memory accesses, which has shown to be effective for sequence transduction problems.

Cloze-type question answering (QA) is a clever way to assess the ability of human and machine to comprehend natural language. This type of tasks are attractive for the natural language processing (NLP) community because the test sets or datasets can be generated without requiring expert's supervision, in an automatic manner, which is typically useful in training and testing artificial intelligent (AI) systems that can understand human language. In cloze-type QA setup, the machine is first presented with a text document containing a fact set and then it is asked to output answer for a query related to the document.

With the recent development of large-scale Cloze-type QA datasets (Hermann et al., 2015; Hill et al., 2015; Onishi et al., 2016) and deep neural network methods, remarkable advances have been made in order to solve the problem in an end-to-end fashion. The existing neural network approaches can be broadly divided into single-step or multi-step comprehension depending on how documenting

reading and answer inference processes are modeled. By mimicking human readers for a deeper reasoning, the multi-step comprehension systems have shown promising results (Hermann et al., 2015; Hill et al., 2015; Trischler et al., 2016; Sordoni et al., 2016). However, the current multiturn models are designed with the predefined number of computational hops for inference while the difficulty of document and query pairs can vary. Some of the query-document pairs require a shallow reasoning like word or sentence level matching, but for some of them a deeper document level reasoning or a complex semantic understanding is crucial.

The proposed NSE comprehension models perform a reasoning process we called hypothesis-test loop. In each step, a new hypothesis for the correct answer is formed by a query regression. Then the hypothesis is checked and if it is true, the model halts the reasoning process (or the hypothesis-test loop) to give the correct answer. To this end, unlike previous methods with fixed computation our models introduce halting procedure in the hypothesis-test loop. When trained with classic back-propagation algorithm, our NSE models show consistent improvements over state-of-the-art baselines on two cloze-type QA datasets.

# 2 RELATED WORK

Recently, several large-scale datasets for machine comprehension have been introduced, including the cloze-type QA (Hermann et al., 2015; Hill et al., 2015; Onishi et al., 2016). Consequently there is an increasing interest in developing neural network approaches to solve the problem in an end-to-end fashion. The existing models for cloze-type QA can be categorized into single-step or multi-step approach depending on their comprehension process.

# 2.1 SINGLE-STEP COMPREHENSION

Singe-step comprehension methods read input document once with a single computational hop to make answer prediction. The reading process mainly involves context modeling with bi-directional recurrent neural networks (RNN) and selective focusing with attention mechanism. Hermann et al. (2015) introduced a CNN/Daily News QA task along with a set of baseline models such as Attentive Reader and Impatient Reader. The Attentive Reader model reads the document and the query with bi-directional LSTM (BiLSTM) networks and selects a query-relevant context by attending over the document. Chen et al. (2016) re-designed the Attentive Reader model (so called Stanford Attentive Reader) and examined the CNN/Daily News QA task. They found out that roughly  $25\%$  of all queries in Hermann et al. (2015)'s dataset is unanswerable and the recent neural network approaches have obtained the ceiling performance on this task. Kadlec et al. (2016) proposed an Attention Sum (AS) reader model that first attends over the document and then aggregates all the attention score for the same candidate answer to select the highest scoring candidate as correct answer.

However, for complex document and query pairs with deeper semantic association machine reading only once may not be enough and multiple reading and checking is crucial to perform deeper reasoning.

# 2.2 MULTI-STEP COMPREHENSION

Similar to a human reader, multi-step comprehension methods read the document and the query before making the final prediction. This kind of comprehension is mostly achieved by implementing an external memory and an attention mechanism to retrieve the query-relevant information from the memory throughout time steps. Hermann et al. (2015)'s Impatient Reader revisits the document states with the attention mechanism whenever it reads the next query word in each time scale. Hill et al. (2015) extended multi-hop Memory Networks (Sukhbaatar et al., 2015) with self-supervision to retrieve supporting memory. EpiReader performs a two-stage computation (Trischler et al., 2016). First it chooses a set of most probable answers with the AS model and forms new queries by replacing an answer placeholder in the original query with the candidate answer words. Second EpiReader runs an entailment estimation between the document and the new query pairs to predict the correct answer. Gated Attention (GA) reader extends the AS model with document-gating and iterative reading of the document and the query (Dhingra et al., 2016). The query representation is used as gating for the document and the iterative reading is accomplished by having separate bi-directional gated recurrent unit (GRU) networks for each computational hop. Iterative Alternative Attention

(IAA) reader (Sordoni et al., 2016) is a multi-step comprehension model which uses a GRU network to search for correct answers from a document. In this model, the GRU network is expected to collect evidence from the document and the query that assists prediction in the last time step. Cui et al. (2016) introduced attention-over-attention loss by computing a word level query-document matching matrix. This model provides a fine-grained word-level supervision signal which seems to help model training.

Our proposed model performs multiple computational steps for deeper reasoning. Unlike the previous work, in our model the number of steps to revise the document is not predefined and it is dynamically adapted for a particular document and query pair. Furthermore, we define novel ways to substitute query words with a word chosen from the document (i.e. regression process) and to check a hypothesis that the selected document word actually compliments the query (i.e. check process). When the hypothesis is true, our model halts the reading process and outputs the word chosen from the document as the correct answer. NSE is used as a controller for the whole process throughout the reasoning steps.

Among the aforementioned models, EpiReader seems to be the most relevant one to our language comprehension models. However, having an entailment estimation introduces a constraint in EpiReader which limits its application. Our model is generic and can be useful in different tasks other than machine comprehension such as language-based conversational tasks, knowledge inference and link prediction. As EpiReader has tightly integrated two-stage neural network modules, the performance directly depends on the first stage. If the first module misses out or fails to choose enough candidates, no correct answer can be found. Our model has no such issue and is not constrained in forming new queries.

# 3 PROPOSED APPROACH

Our dataset consists of tuples  $(D, Q, A, a)$ , where  $D$  is the document (i.e. passage) serving as a fact for the query  $Q$ ,  $A$  is a set of candidate answers and  $a \in A$  is the true answer. The document and the query are sequences of tokens drawn from a vocabulary  $V$ . We train a model to predict the correct answer  $a$  from the candidate set  $A$  for given a pair of query and document.

The main components of our proposed model are shown in Figure 1. First the query and the document memory are initialized via context embedding (omitted from the figure). The memories are then processed with memory read and write operations throughout the hypothesis-test loop. In each step of the loop, the read module formulates a new hypothesis by updating the query memory with relevant content from the document memory. Then the write module tests whether the new hypothesis is true by inspecting the current query and the document states. Selecting relevant content from the document to regress the old query is essentially an inference (i.e. prediction) in our model. Intuitively the input query is regressed toward (becoming) a complete query containing the correct answer word within it. If the write module thinks that the query is complete and the correct answer is found, it halts the hypothesis-test loop. The write module also supervises the query state transitions and retains the right to roll back the new query changes during the reasoning process. To avoid an overconfident prediction and to halt the reasoning process, we explore two different strategies: (a) query gating and (b) adaptive computation.

Like a human reader, the model reads the document multiple times, formulates a hypothetical answer for the query and tests it against the story of the document throughout the hypothesis-test steps. Once satisfied with the response of the current hypothesis, the model outputs the response as the correct answer.

# 3.1 MEMORY INITIALIZATION

Instead of using raw word embeddings, we initialize the document and the query memories via context embedding in order to inform the memory slots of contextual information from text passages. Two BiLSTM networks are applied to the document and the query sequences separately, as:

$$
M _ {0} ^ {q} = B i L S T M ^ {q} (Q) \tag {1}
$$

$$
M ^ {d} = B i L S T M ^ {d} (D) \tag {2}
$$

![](images/77712739b46cd9c1c59dbcca632f7fe4a9ca06918252378c8ec292a2a5e46ebe.jpg)

![](images/95f4bd24ab0f83b40ac6157b015ebc2fb21838f7aab2e3f78d55feb854ef881f.jpg)  
(a)  
(b)  
Figure 1: High-level architectures of our proposed models: the NSE Query Gating model (a) and the NSE Adaptive Computation model (b). The query memory of the former is gated to the next step by the write module whereas the query memory in the latter is updated and passed to the next step without gating and the write module is trained to halt the hypothesis-test loop. r: read, c: compose and w: write module.

where  $M_0^q \in R^{k \times |Q|}$  and  $M^d \in R^{k \times |D|}$  are the memory representations.  $k$  is the size of BiLSTM hidden layer. The query memory is evolved over time while the document memory is not updated and rather serves as a fact set for the query.

Note that in our model the context embedding network can be any network accepting word embeddings as input, such as multilayer perceptron (MLP) or convolutional neural network. We choose BiLSTM because it is able to learn the word-centered context representation effectively by reading text with LSTMs from both left and right and concatenating resulting hidden vectors.

# 3.2 HYPOTHESIS TESTING

The query and the document memories are further processed through an iterative process called hypothesis-test loop. In each step of the loop, the query memory  $M^q$  is updated with content from the document memory to form a new query (i.e. hypothesis formulation). The new query is then checked against the document facts and used to make an answer prediction (i.e. hypothesis testing). The NSE read, compose and write modules collectively perform the following overall process.

Read: this module takes in the query and the document states  $s_{t-1}^q, s_{t-1}^d \in R^k$  from the previous time step  $t-1$  as input and computes the alignment vectors  $l_t^q \in R^{|Q|}, l_t^d \in R^{|D|}$ , the new query and the document states  $s_t^q, s_t^d \in R^k$  and the query memory key vector  $z_t^q \in R^{|Q|}$  as follows:

$$
r _ {t} = \operatorname {r e a d} ^ {L S T M} \left(\left[ s _ {t - 1} ^ {q}; s _ {t - 1} ^ {d} \right]\right) \tag {3}
$$

$$
l _ {t} ^ {q} = r _ {t} ^ {\top} M _ {t - 1} ^ {q} \tag {4}
$$

$$
s _ {t} ^ {q} = \operatorname {s o f t m a x} \left(l _ {t} ^ {q}\right) ^ {\top} M _ {t - 1} ^ {q} \tag {5}
$$

$$
z _ {t} ^ {q} = \operatorname {s i g m o i d} \left(l _ {t} ^ {q}\right) \tag {6}
$$

$$
l _ {t} ^ {d} = s _ {t} ^ {q \top} M ^ {d} \tag {7}
$$

$$
s _ {t} ^ {d} = \operatorname {s o f t m a x} \left(l _ {t} ^ {d}\right) ^ {\top} M ^ {d} \tag {8}
$$

Intuitively depending on the previously retrieved document content as well as the previous query states, the read module retrieves from the document a word to be relocated to the query and computes the positions of this word in the query. Since the document word can be located in multiple different positions in the query, sigmoid function is used to normalize the alignment vector  $l_{t}^{q}$ . The document state  $s_{t}^{d}$  represents the newly retrieved word and the key vector  $z_{t}^{q}$  defines its new position in the query memory. Because such a decision is made sequentially in every step, we equip the read module with a LSTM network (i.e. readLSTM). We initialize the document and the query states  $s_0^d, s_0^q$  with the last hidden states of the query and the document BiLSTM networks.

Compose: the compose module combines the current query and document states  $s_t^q, s_t^d$  and the current hidden state of the read module  $r_t \in R^k$  as:

$$
c _ {t} = \operatorname {c o m p o s e} ^ {M L P} \left(s _ {t} ^ {q}, s _ {t} ^ {d}, r _ {t}\right) \tag {9}
$$

The resulting single vector  $c_{t} \in R^{k}$  is passed to the write module for subsequent process. The compose module can be viewed as a feature extractor from the current document and query pair. By taking in the hidden state  $r_{t}$ , the compose module also informs the write module of the read module's current decision.

Write: this module accepts the outputs of the read module and updates the query memory as follows:

$$
M _ {t} ^ {q} = M _ {t - 1} ^ {q} z _ {t} ^ {q} + s _ {t} ^ {d} \left(\mathbf {1} - z _ {t} ^ {q}\right) \tag {10}
$$

where  $\mathbf{1}$  is a matrix of ones. Note that the values of the key vector  $z_{t}^{q}$  ranges from zero to one and zero (or near zero) values indicate the query position where the new document word is written.

The write module is also responsible for checking the new hypothesis in order to decide whether to halt the hypothesis-test loop for the final answer or to continue. We explore two different strategies to be discussed below. Both methods take in the output of the compose module and employs the LSTM to make the sequential decision.

# 3.2.1 QUERY GATING

Figure 1 (a) shows the overall architecture of our model with query gating mechanism. In this model instead of making a hard decision on halting the loop (i.e. stop reading), the write module performs a word-level query gating as:

$$
w _ {t} = \operatorname {w r i t e} ^ {\text {L S T M}} \left(c _ {t}\right) \tag {11}
$$

$$
g _ {t} ^ {q} = \operatorname {s i g m o i d} \left(w _ {t} ^ {\top} M _ {t - 1} ^ {q}\right) \tag {12}
$$

$$
M _ {t} ^ {q} = M _ {t} ^ {q} \left(\mathbf {1} - g _ {t} ^ {q}\right) + M _ {t - 1} ^ {q} g _ {t} ^ {q} \tag {13}
$$

where  $\mathbf{1}$  is a matrix of ones. The key part in the above equation is obtaining the gating weights. This is accomplished by comparing each memory slot with the hidden vector  $w_{t} \in R^{k}$  and normalizing resulting scores with sigmoid function. Therefore, the gating vector  $g_{t}^{q} \in R^{|Q|}$  have ones for preserving and zeros for erasing content of the old query.

This can be seen as a memory gating process which prevents the model from forgetting the old query information. Note that even if the query memory is updated with the document content given by the read module, the write module makes the final decision based on features extracted by the compose module. In other words if the write module decides to keep the old query information, the changes in the new query are simply ignored and the same query from the previous time step is passed along to the next step. The number of steps  $T$  in the hypothesis-test loop is a hyperparameter in this model. Therefore, in this setup the write module is expected to lock the query state with its gating mechanism as soon as the hypothesis is true.

Table 1: Statistics of the datasets. train (s): train strict, train (r): train relaxed and cands: candidates.  

<table><tr><td rowspan="2"></td><td colspan="4">WDW</td><td colspan="4">CBT-NE</td><td colspan="3">CBT-CN</td></tr><tr><td>train (s)</td><td>train (r)</td><td>dev</td><td>test</td><td>train</td><td>dev</td><td>test</td><td>train</td><td>dev</td><td>test</td><td></td></tr><tr><td># queries</td><td>127,786</td><td>185,978</td><td>10,000</td><td>10,000</td><td>108,719</td><td>2,000</td><td>2,500</td><td>120,769</td><td>2,000</td><td>2,500</td><td></td></tr><tr><td>avg. # cands</td><td>3.5</td><td>3.5</td><td>3.4</td><td>3.4</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td><td></td></tr><tr><td>avg. # tokens</td><td>365</td><td>378</td><td>325</td><td>326</td><td>433</td><td>412</td><td>424</td><td>470</td><td>448</td><td>461</td><td></td></tr><tr><td>vocab size</td><td>308,602</td><td></td><td>347,406</td><td></td><td></td><td>53,063</td><td></td><td></td><td>53,185</td><td></td><td></td></tr></table>

# 3.2.2 ADAPTIVE COMPUTATION

In this model, the write module is equipped with a termination head as shown in Figure 1 (b). Particularly, the write module with the termination head decides its willingness to continue or finish the computation in each step.

We define a probabilistic framework for halting. Our approach is similar to the input and output handling mechanism of Neural Random-Access Machines (Kurach et al., 2016). In each time step, the write module outputs a termination score  $e_t$  as follows:

$$
w _ {t} = \operatorname {w r i t e} ^ {\text {L S T M}} \left(c _ {t}\right) \tag {14}
$$

$$
e _ {t} = \operatorname {s i g m o i d} \left(o ^ {\top} w _ {t}\right) \tag {15}
$$

where  $o \in R^k$  is a trainable vector that projects the hidden state  $w_t$  to a scalar value. Then the probability to halt the hypothesis-test loop after  $t$  steps is

$$
p _ {t} = e _ {t} \prod_ {i = 1} ^ {t - 1} \left(1 - e _ {i}\right) \tag {16}
$$

We also introduce a hyperparameter  $T$  for the maximum number of permitted steps. If the model runs out of the time without halting the process (after  $T$  steps), we force the model to output the final answer in step  $T$ . In this case, the probability to stop reading is

$$
p _ {T} = 1 - \sum_ {i = 1} ^ {T - 1} p _ {i} \tag {17}
$$

# 3.3 ANSWER PREDICTION

In step  $t$ , the query-to-document alignment score  $l_t^d$  is used to compute the probability  $P(a|Q,D)$  that the answer  $a$  is correct given the document and the query. In particular, we adapt the pointer sum attention mechanism (Kadlec et al., 2016) as

$$
P _ {t} (a | Q, D) = v ^ {\top} \operatorname {s o f t m a x} \left(l _ {t} ^ {d}\right) \tag {18}
$$

where  $v \in R^{|D|}$  is a mask denoting the positions of the answer token  $a$  in the document (ones for the token and zeros otherwise).

For the query gating model, we use the probability  $P_{T}(a|Q,D)$  produced in the last step  $T$  to choose the correct answer. For the second model, we incorporate the termination score  $p_t$  and redefine the probability as

$$
P (a | Q, D) = \sum_ {i = 1} ^ {T} \left(p _ {i} \cdot P _ {i} (a | Q, D)\right) \tag {19}
$$

We then train the models to minimize cross-entropy loss between the predicted probabilities and correct answers.

# 4 EXPERIMENTS

We evaluated our models on two large-scale datasets: Childrens Book Test (CBT) (Hill et al., 2015) and Who-Did-What (WDW) (Onishi et al., 2016). We focused on these tasks because there is

Table 2: Model comparison on the CBT dataset.  

<table><tr><td rowspan="2">Model</td><td colspan="2">CBT-NE</td><td colspan="2">CBT-CN</td></tr><tr><td>dev</td><td>test</td><td>dev</td><td>test</td></tr><tr><td>Human (context + query) (Hill et al., 2015)</td><td>-</td><td>81.6</td><td>-</td><td>81.6</td></tr><tr><td>LSTMs (context + query) (Hill et al., 2015)</td><td>51.2</td><td>41.8</td><td>62.6</td><td>56.0</td></tr><tr><td>MemNNs (window mem. + self-sup.) (Hill et al., 2015)</td><td>70.4</td><td>66.6</td><td>64.2</td><td>63.0</td></tr><tr><td>AS Reader (Kadlec et al., 2016)</td><td>73.8</td><td>68.6</td><td>68.8</td><td>63.4</td></tr><tr><td>GA Reader (Dhingra et al., 2016)</td><td>74.9</td><td>69.0</td><td>69.0</td><td>63.9</td></tr><tr><td>EpiReader (Trischler et al., 2016)</td><td>75.3</td><td>69.7</td><td>71.5</td><td>67.4</td></tr><tr><td>IAA Reader (Sordoni et al., 2016)</td><td>75.2</td><td>68.6</td><td>72.1</td><td>69.2</td></tr><tr><td>AoA Reader (Cui et al., 2016)</td><td>77.8</td><td>72.0</td><td>72.2</td><td>69.4</td></tr><tr><td>MemNN (window mem. + self-sup. + ensemble) (Hill et al., 2015)</td><td>70.4</td><td>66.6</td><td>64.2</td><td>63.0</td></tr><tr><td>AS Reader (ensemble) (Kadlec et al., 2016)</td><td>74.5</td><td>70.6</td><td>71.1</td><td>68.9</td></tr><tr><td>EpiReader (ensemble) (Trischler et al., 2016)</td><td>76.6</td><td>71.8</td><td>73.6</td><td>70.6</td></tr><tr><td>IAA Reader (ensemble) (Sordoni et al., 2016)</td><td>76.9</td><td>72.0</td><td>74.1</td><td>71.0</td></tr><tr><td>NSE (T = 1)</td><td>76.2</td><td>71.1</td><td>72.8</td><td>69.7</td></tr><tr><td>NSE Query Gating (T = 2)</td><td>76.6</td><td>71.5</td><td>72.3</td><td>70.7</td></tr><tr><td>NSE Query Gating (T = 6)</td><td>77.0</td><td>71.4</td><td>73.0</td><td>72.0</td></tr><tr><td>NSE Query Gating (T = 9)</td><td>78.0</td><td>72.6</td><td>73.5</td><td>71.2</td></tr><tr><td>NSE Query Gating (T = 12)</td><td>77.7</td><td>72.2</td><td>74.3</td><td>71.9</td></tr><tr><td>NSE Adaptive Computation (T = 2)</td><td>77.1</td><td>72.1</td><td>72.8</td><td>71.2</td></tr><tr><td>NSE Adaptive Computation (T = 12)</td><td>78.2</td><td>73.2</td><td>74.2</td><td>71.4</td></tr></table>

still a large gap between the human and the machine performances on CBT and WDW, which is in contrast to the CNN/Daily News QA datasets covered in Section 2. The CBT dataset was constructed from the children book domain whereas the WDW corpus was built from the news article domain (the English Gigaword corpus); therefore we think that the two datasets are quite representative for evaluation of our models. Furthermore the CBT dataset comes with two difficult tasks depending on the type of answer words to be predicted: named entity (CBT-NE) and common nouns (CBT-CN). The WDW training set has two different setups with strict and relaxed baseline suppression. Table 1 summarizes some important statistics of the datasets.

# 4.1 TRAINING DETAILS

We chose one-layer LSTM networks for the read and the write modules and an MLP with single-layer for the composition module. We used stochastic gradient descent with an Adam optimizer to train the models. The initial learning rate  $(lr)$  was set to 0.0005 for CBT-CN or 0.001 for other tasks. A pre-trained 300-D Glove 840B vectors (Pennington et al., 2014) were used to initialize the word embedding layer $^2$ ; therefore the embedding layer size is 300. The hidden layer size of the context embedding BiLSTM nets  $k = 436$ . The embeddings for out-of-vocabulary words and the model parameters were randomly initialized from the uniform distribution over  $[-0.1, 0.1]$ . The gradient clipping threshold was set to 15. The models were regularized by applying  $20\%$  dropouts to the embedding layer $^3$ . We used the batch size  $n = 32$  for the CBT dataset and  $n = 25$  for the WDW dataset and early stopping with a patience of 1 epoch. For the WDW dataset, we anonymized the answer candidates by following the work of Onishi et al. (2016) and Hermann et al. (2015).

We run a hyperparameter search over  $k = \{256, 368, 436, 512\}$ ,  $lr = \{0.0005, 0.001\}$  and  $l_{2}$  decay  $= \{0.0001, 0.00005, 0.00001\}$  on the CBT dev sets to come up with the current setting for training. Among these parameters, the  $l_{2}$  weight decay regularizer did not seem to help and thus it was not applied. We did not tune the dropouts.

We used the following batching heuristic in order to speedup the training. We created a temporary example pool by randomly sampling from the training set and sorted them according to the length of the document. Then the first  $n$  examples ( $n = 32$  or  $n = 25$ ) of the ordered pool were put into the

Table 3: Model comparison on the WDW dataset.  

<table><tr><td rowspan="2">Model</td><td colspan="2">Strict</td><td colspan="2">Relaxed</td></tr><tr><td>dev</td><td>test</td><td>dev</td><td>test</td></tr><tr><td>Human (Onishi et al., 2016)</td><td>-</td><td>84.0</td><td>-</td><td>-</td></tr><tr><td>Attentive Reader (Hermann et al., 2015)</td><td>-</td><td>53.0</td><td>-</td><td>55.0</td></tr><tr><td>AS Reader (Kadlec et al., 2016)</td><td>-</td><td>57.0</td><td>-</td><td>59.0</td></tr><tr><td>GA Reader (Dhingra et al., 2016)</td><td>-</td><td>57.0</td><td>-</td><td>60.0</td></tr><tr><td>Stanford Attentive Reader (Chen et al., 2016)</td><td>-</td><td>64.0</td><td>-</td><td>65.0</td></tr><tr><td>NSE (T=1)</td><td>65.1</td><td>65.5</td><td>66.4</td><td>65.3</td></tr><tr><td>NSE Query Gating (T=2)</td><td>65.4</td><td>65.1</td><td>65.7</td><td>65.5</td></tr><tr><td>NSE Query Gating (T=6)</td><td>65.5</td><td>65.7</td><td>65.6</td><td>65.8</td></tr><tr><td>NSE Query Gating (T=9)</td><td>65.8</td><td>65.8</td><td>65.8</td><td>65.9</td></tr><tr><td>NSE Query Gating (T=12)</td><td>65.2</td><td>65.5</td><td>65.7</td><td>65.4</td></tr><tr><td>NSE Adaptive Computation (T=2)</td><td>65.3</td><td>65.4</td><td>66.2</td><td>66.0</td></tr><tr><td>NSE Adaptive Computation (T=12)</td><td>66.5</td><td>66.2</td><td>67.0</td><td>66.7</td></tr></table>

same batch, without replacement, and the rest of the pool was replaced back in the training set. This is performed until there is not enough training examples to create a new pool. Finally the documents in the same batch are padded with special symbol  $<pad>$  to the same length. The batches were regenerated in every epoch to prevent the model from learning a simple mapping function. We also padded queries in the same batch with the special symbol to an equal length.

# 4.2 RESULTS

In Tables 2 and 3 we compared the performance of our models with all published models (as baselines) including their ensemble variations. We report the performance of our query gating model at the varying number of hypothesis-test steps  $T = \{1,2,6,9,12\}$ .  $T$  for our adaptive computation model was set to 2 or 12. When  $T = 1$ , both models update the memory only once and do not have enough time to read it back and thus they are reduced to the same model called NSE (i.e. NSE  $T = 1$ ). Typically when  $T$  is greater than one, the proposed halting strategies result in two different models.

Our query gating model achieves  $72.6\%$  and  $72.0\%$  accuracy on the CBT tasks outperforming all the previous baselines. The performance varies for the different number of steps. For the larger number of allowed steps, the query gating model tends to overfit on the CBT dataset as the performance gap between the dev and test sets increases. Our NSE Adaptive Computation model sets the best score on CBT-NE task with  $73.2\%$  accuracy. Overall our NSE models bring modest improvements of  $1.2\%$  on CBT-NE and  $2.6\%$  on CBT-CN tasks and now the performance differences between the human and the machines are only  $8.4\%$  and  $9.6\%$ .

On the WDW task our single model, NSE with adaptive computation, scores  $2.2\%$  and  $1.7\%$  higher than the previous best result of Stanford Attentive Reader. The NSE Query Gating model obtains its best result when  $T = 9$ . Comparing to the proposed variations of NSE, our Adaptive Computation model is more robust and sets new state-of-the-art performance on three out of four tasks by effectively deciding when to halt the reasoning processing with the termination head. Memory locking (to prevent from forgetting) did not show to be as effective as the termination-based approach on this task. In Appendix A, we visualize query regression process in both models and observe that while new queries are generated, query words relevant to the correct answer are not overwritten by the memory write module.

The performance of the NSE model with  $T = 1$  matches that of the previous single systems. In general given a small number of permitted steps, the proposed models tend to less overfit but the final performance is not high. As the number of permitted steps increases both dev and test accuracy improves yielding an overall higher performance. However, this holds up to a certain point and with a large permitted steps we no longer observe a significant performance improvement in terms of testing accuracy. For example, NSE Query Gating model with  $T = 15$  (not included in the table) achieved  $79.2\%$  dev accuracy and  $71.4\%$  test accuracy on CBT-NE task, showing the highest

accuracy on the dev set yet massively overfitting on the test set. Furthermore it becomes expensive to train a model with a large number of allowed steps.

It is worth noting that even if the proposed models in this work were trained using classic end-to-end back-propagation, they can easily be trained with reinforcement learning methods, such as the REINFORCE algorithm (Williams, 1992), for further evaluation. Such an adaptation is particularly straightforward for NSE Adaptive Computation because this model has already incorporated a probabilistic termination head in it.

# 5 CONCLUSION

Inspired by cognitive process of hypothesis testing in human brain, we proposed a reasoning approach based memory augmented neural networks and applied it to language comprehension. Our proposed NSE models with dynamic reasoning have achieved the state-of-the-art results on two machine comprehension tasks. In order to halt reasoning process we explored two different strategies to be fully trained with classic back-propagation algorithm: query memory gating which prevents from forgetting old query and adaptive computation with termination head. The NSE adaptive computation model has shown to be effective in the experiments. Our proposed models can be trained using reinforcement learning. We plan to apply our approach to other AI tasks, such language-based conversational tasks, link prediction and knowledge inference.

# REFERENCES

Marcel A Just and Patricia A Carpenter. A capacity theory of comprehension: individual differences in working memory. Psychological review, 99(1):122, 1992.  
Thad A Polk and Colleen M Seifert. Cognitive modeling. MIT Press, 2002.  
Tsendsuren Munkhdalai and Hong Yu. Neural semantic encoders. arXiv preprint arXiv:1607.04315, 2016.  
Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. In NIPS, 2015.  
Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston. The goldilocks principle: Reading children's books with explicit memory representations. arXiv preprint arXiv:1511.02301, 2015.  
Takeshi Onishi, Hai Wang, Mohit Bansal, Kevin Gimpel, and David McAllester. Who did what: A large-scale person-centered cloze dataset. In EMNLP 2016, 2016.  
Adam Trischler, Zheng Ye, Xingdi Yuan, and Kaheer Suleman. Natural language comprehension with the epireader. arXiv preprint arXiv:1606.02270, 2016.  
Alessandro Sordoni, Phillip Bachman, and Yoshua Bengio. Iterative alternating neural attention for machine reading. arXiv preprint arXiv:1606.02245, 2016.  
Danqi Chen, Jason Bolton, and Christopher D Manning. A thorough examination of the cnn/daily mail reading comprehension task. In ACL 2016, 2016.  
Rudolf Kadlec, Martin Schmid, Ondrej Bajgar, and Jan Kleindienst. Text understanding with the attention sum reader network. In ACL 2016, 2016.  
Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. End-to-end memory networks. In NIPS 2015, pages 2431-2439, 2015.  
Bhuwan Dhingra, Hanxiao Liu, William W Cohen, and Ruslan Salakhutdinov. Gated-attention readers for text comprehension. arXiv preprint arXiv:1606.01549, 2016.  
Yiming Cui, Zhipeng Chen, Si Wei, Shijin Wang, Ting Liu, and Guoping Hu. Attention-over-attention neural networks for reading comprehension. arXiv preprint arXiv:1607.04423, 2016.  
Karol Kurach, Marcin Andrychowicz, and Ilya Sutskever. Neural random-access machines. *ICLR* 2016, 2016.

Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In EMNLP, volume 14, pages 1532-1543, 2014.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.
