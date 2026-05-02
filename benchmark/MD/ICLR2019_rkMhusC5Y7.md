# LEARNING TO COORDINATE MULTIPLE REINFORCEMENT LEARNING AGENTS FOR DIVERSE QUERY RE-FORMULATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose a method to efficiently learn diverse strategies in reinforcement learning for query reformulation in the tasks of document retrieval and question answering. In the proposed framework an agent consists of multiple specialized sub-agents and a meta-agent that learns to aggregate the answers from sub-agents to produce a final answer. Sub-agents are trained on disjoint partitions of the training data, while the meta-agent is trained on the full training set. Our method makes learning faster, because it is highly parallelizable, and has better generalization performance than strong baselines, such as an ensemble of agents trained on the full data. We show that the improved performance is due to the increased diversity of reformulation strategies.

# 1 INTRODUCTION

Reinforcement learning has proven effective in several language processing tasks, such as machine translation (Wu et al., 2016; Ranzato et al., 2015; Bahdanau et al., 2016), question-answering (Wang et al., 2017a; Hu et al., 2017), and text summarization (Paulus et al., 2017). In reinforcement learning efficient exploration is key to achieve good performance. The ability to explore in parallel a diverse set of strategies often speeds up training and leads to a better policy (Mnih et al., 2016; Osband et al., 2016).

In this work, we propose a simple method to achieve efficient parallelized exploration of diverse policies, inspired by hierarchical reinforcement learning (Singh, 1992; Lin, 1993; Dietterich, 2000; Dayan & Hinton, 1993). We structure the agent into multiple sub-agents, which are trained on disjoint subsets of the training data. Sub-agents are co-ordinated by a meta-agent, called aggregator, that groups and scores answers from the sub-agents for each given input. Unlike sub-agents, the aggregator is a generalist since it learns a policy for the entire training set.

We argue that it is easier to train multiple sub-agents than a single generalist one since each sub-agent only needs to learn a policy that performs well for a subset of examples. Moreover, specializing agents on different partitions of the data encourages them to learn distinct policies, thus giving the aggregator the possibility to see answers from a population of diverse agents. Learning a single policy that results in an equally diverse strategy is more challenging.

Since each sub-agent is trained on a fraction of the data, and there is no communication between them, training can be done faster than training a single agent on the full data. Additionally, it is easier to parallelize than applying existing distributed algorithms such as asynchronous SGD or A3C (Mnih et al., 2016), as the sub-agents do not need to exchange weights or gradients. After training the sub-agents, only their actions need to be sent to the aggregator.

We build upon the works of Nogueira & Cho (2017) and Buck et al. (2018b). Therefore, we evaluate the proposed method on the same tasks they used: query reformulation for document retrieval and question-answering. We show that it outperforms a strong baseline of an ensemble of agents trained on the full dataset. We also found that performance and reformulation diversity are correlated (Sec. 5.5).

Our main contributions are the following:

- A simple method to achieve more diverse strategies and better generalization performance than a model average ensemble.  
- Training can be easily parallelized in the proposed method.  
- An interesting finding that contradicts our, perhaps naive, intuition: specializing agents on semantically similar data does not work as well as random partitioning. An explanation is given in Appendix F.

# 2 RELATED WORK

The proposed approach is inspired by the mixture of experts, which was introduced more than two decades ago (Jacobs et al., 1991; Jordan & Jacobs, 1994) and has been a topic of intense study since then. The idea consists of training a set of agents, each specializing in some task or data. One or more gating mechanisms then select subsets of the agents that will handle a new input. Recently, Shazeer et al. (2017) revisited the idea and showed strong performances in the supervised learning tasks of language modeling and machine translation. Their method requires that output vectors of experts are exchanged between machines. Since these vectors can be large, the network bandwidth becomes a bottleneck. They used a variety of techniques to mitigate this problem. Anil et al. (2018) later proposed a method to further reduce communication overhead by only exchanging the probability distributions of the different agents. Our method, instead, requires only scalars (rewards) and short strings (original query, reformulations, and answers) to be exchanged. Therefore, the communication overhead is small.

Previous works used specialized agents to improve exploration in RL (Dayan & Hinton, 1993; Singh, 1992; Kaelbling et al., 1996). For instance, Stanton & Clune (2016) and Conti et al. (2017) use a population of agents to achieve a high diversity of strategies that leads to better generalization performance and faster convergence. Rusu et al. (2015) use experts to learn subtasks and later merge them into a single agent using distillation (Hinton et al., 2015).

The experiments are often carried out in simulated environments, such as robot control (Brockman et al., 2016) and video-games (Bellemare et al., 2013). In these environments, rewards are frequently available, the states have low diversity (e.g. same image background), and responses are normally fast (60 frames per second). We, instead, evaluate our approach on tasks whose inputs (queries) and states (documents and answers) are diverse because they are in natural language, and the environment responses are slow (0.5-5 seconds per query).

Somewhat similarly motivated is the work of Serban et al. (2017). They train many heterogeneous response models and further train an RL agent to pick one response per utterance.

# 3 METHOD

# 3.1 TASK

We describe the proposed method using a generic end-to-end search task. The problem consists in learning to reformulate a query so that the underlying retrieval system can return a better result.

Following Nogueira & Cho (2017) and Buck et al. (2018b) we frame the task as a reinforcement learning problem, in which the query reformulation system is an RL-agent that interacts with an environment that provides answers and rewards. The goal of the agent is to generate reformulations such that the expected returned reward (i.e., correct answers) is maximized. The environment is treated as a black-box, i.e., the agent does not have direct access to any of its internal mechanisms. Figure 1-(b) illustrates this framework.

# 3.2 SYSTEM

Figure 1-(c) illustrates the new agent. An input query  $q_{0}$  is given to the  $N$  sub-agents. A sub-agent is any system that accepts as input a query and returns a corresponding reformulation. Thus, sub-agents can be heterogeneous.

![](images/ed442cb6ddc25d5dab5644dfbcd677c4a083db66ab8a8f638e2f3cddbc625400.jpg)  
(a)

![](images/bf74551a66a2935d48d479c12c36032b670f8086552eb871da0884f5555d943f.jpg)  
(b)

![](images/3789d8864967272548b2b9e11f1f4ad7cb2faea510b318c57d2ba81db0e8a995.jpg)  
(c)  
Figure 1: a) A vanilla search system. The query  $q_{0}$  is given to the system which outputs a result  $a_{0}$ . b) The search system with a reformulator. The reformulator queries the system with  $q_{0}$  and its reformulations  $\{q_{1},\dots q_{N}\}$  and receives back the results  $\{a_0,\dots,a_N\}$ . A selector then decides the best result  $a_{i}$  for  $q_{0}$ . c) The proposed system. The original query is reformulated multiple times by different reformulators. Reformulations are used to obtain results from the search system, which are then sent to the aggregator, which picks the best result for the original query based on a learned weighted majority voting scheme. Reformulators are independently trained on disjoint partitions of the dataset thus increasing the variability of reformulations.

Here we train each sub-agent on a partition of the training set. The  $i$ -th agent queries the underlying search system with the reformulation  $q_{i}$  and receives a result  $a_{i}$ . The set  $\{(q_i,a_i)|0\leq i\leq N\}$  is given to the aggregator, which then decides which result will be final.

# 3.3 SUB-AGENTS

The first step for training the new agent is to partition the training set. We randomly split it into equal-sized subsets. For an analysis of how other partitioning methods affect performance, see Appendix F. In our implementation, a sub-agent is a sequence-to-sequence model (Sutskever et al., 2014; Cho et al., 2014) trained on a partition of the dataset. It receives as an input the original query  $q_{0}$  and outputs a list of reformulated queries  $(q_{i})$  using beam search.

Each reformulation  $q_{i}$  is given to the same environment that returns a list of results  $(a_{i}^{1},..,a_{i}^{K})$  and their respective rewards  $(r_{i}^{1},..,r_{i}^{K})$ . We then use REINFORCE (Williams, 1992) to train the sub-agent. At training time, instead of using beam search, we sample reformulations.

Note that we also add the identity agent (i.e., the reformulation is the original query) to the pool of sub-agents.

# 3.4 META-AGENT: AGGREGATOR

The aggregator receives as inputs  $q_{0}$  and a list of candidate results  $(a_{i}^{1},..a_{i}^{K})$  for each reformulation  $q_{i}$ . We first compute the set of unique results  $a_{j}$  and two different scores for each result: the accumulated rank score  $s_{j}^{A}$  and the relevance score  $s_{j}^{R}$ .

The accumulated rank score is computed as  $s_j^A = \sum_{i=1}^{N} \frac{1}{\mathrm{rank}_{i,j}}$ , where  $\mathrm{rank}_{i,j}$  is the rank of the j-th result when retrieved using  $q_i$ . The relevance score  $s_j^R$  is the prediction that the result  $a_j$  is relevant to query  $q_0$ . It is computed as:

$$
s _ {j} ^ {R} = \sigma \left(W _ {2} \operatorname {R e L U} \left(W _ {1} z _ {j} + b _ {1}\right) + b _ {2}\right), \tag {1}
$$

where

$$
z _ {j} = f _ {\mathrm {C N N}} \left(q _ {0}\right) \left\| f _ {\mathrm {B O W}} \left(a _ {j}\right) \right\| f _ {\mathrm {C N N}} \left(q _ {0}\right) - f _ {\mathrm {B O W}} \left(a _ {j}\right) \left\| f _ {\mathrm {C N N}} \left(q _ {0}\right) \odot f _ {\mathrm {B O W}} \left(a _ {j}\right), \right. \tag {2}
$$

$W_{1} \in \mathbb{R}^{4D \times D}$  and  $W_{2} \in \mathbb{R}^{D \times 1}$  are weight matrices,  $b_{1} \in \mathbb{R}^{D}$  and  $b_{2} \in \mathbb{R}^{1}$  are biases. The symbol  $||$  denotes the concatenation operation,  $\sigma$  is the sigmoid function, and ReLU is a Rectified Linear Unit function (Nair & Hinton, 2010). The function  $f_{\mathrm{CNN}}$  is implemented as a CNN encoder followed by average pooling over the sequence (Kim, 2014). The function  $f_{\mathrm{BOW}}$  is the average word embeddings of the result. At test time, the top-K answers with respect to  $s_{j} = s_{j}^{A}s_{j}^{R}$  are returned.

We train the aggregator with stochastic gradient descent (SGD) to minimize the cross-entropy loss:

$$
L = - \sum_ {j \in J ^ {*}} \log \left(s _ {j} ^ {R}\right) - \sum_ {j \notin J ^ {*}} \log \left(1 - s _ {j} ^ {R}\right), \tag {3}
$$

where  $J^{*}$  is the set of indexes of the ground-truth results. The architecture details and hyperparameters can be found in Appendix B.

# 4 DOCUMENT RETRIEVAL

We now present experiments and results in a document retrieval task. In this task, the goal is to rewrite a query so that the number of relevant documents retrieved by a search engine increases.

# 4.1 ENVIRONMENT

The environment receives a query as an action, and it returns a list of documents as an observation/state and a reward computed using a list of ground truth documents. We use Lucene<sup>2</sup> in its default configuration<sup>3</sup> as our search engine. The input is a query and the output is a ranked list of documents.

# 4.2 DATASETS

To train and evaluate the models, we use three datasets:

TREC-CAR: Introduced by Dietz & Ben (2017), in this dataset the input query is the concatenation of a Wikipedia article title with the title of one of its section. The ground-truth documents are the paragraphs within that section. The corpus consists of all of the English Wikipedia paragraphs, except the abstracts. The released dataset has five predefined folds, and we use the first four as a training set (approx. 3M queries), and the remaining as a validation set (approx. 700k queries). The test set is the same used to evaluate the submissions to TREC-CAR 2017 (approx. 1,800 queries).

JEOPARDY: This dataset was introduced by Nogueira & Cho (2016). The input is a Jeopardy! question. The ground-truth document is a Wikipedia article whose title is the answer to the question. The corpus consists of all English Wikipedia articles.

MSA: Introduced by Nogueira & Cho (2017), this dataset consists of academic papers crawled from Microsoft Academic API. A query is the title of a paper and the ground-truth answer consists of the papers cited within. Each document in the corpus consists of its title and abstract.

# 4.3 REWARD

Since the main goal of query reformulation is to increase the proportion of relevant documents returned, we use recall as the reward:  $\mathbb{R}@\mathbb{K} = \frac{|D_K \cap D^*|}{|D^*|}$ , where  $D_K$  are the top- $K$  retrieved documents and  $D^*$  are the relevant documents. We also experimented using as a reward other metrics such as NDCG, MAP, MRR, and R-Precision but these resulted in similar or slightly worse performance than Recall@40. Despite the agents optimizing for Recall, we report the main results in MAP as this is a more commonly used metric in information retrieval. For results in other metrics, see Appendix A.

<table><tr><td></td><td>TREC-CAR</td><td>Jeopardy</td><td>MSA</td><td>Training (Days)</td><td>FLOPs (×1018)</td></tr><tr><td>Lucene</td><td>9.4</td><td>7.1</td><td>3.1</td><td colspan="2">N/A</td></tr><tr><td>PRF</td><td>9.8</td><td>12.2</td><td>3.4</td><td colspan="2">N/A</td></tr><tr><td>RM3</td><td>10.2</td><td>12.6</td><td>3.1</td><td colspan="2">N/A</td></tr><tr><td>RL-RNN (Nogueira &amp; Cho, 2017)</td><td>10.8</td><td>15.0</td><td>4.1</td><td>10</td><td>2.3</td></tr><tr><td>RL-10-Ensemble</td><td>10.9</td><td>16.1</td><td>4.4</td><td>10</td><td>23.0</td></tr><tr><td>RL-RNN Greedy + Aggregator</td><td>10.9</td><td>21.2</td><td>4.5</td><td>10</td><td>2.3</td></tr><tr><td>RL-RNN 20 Sampled + Aggregator</td><td>11.1</td><td>21.5</td><td>4.6</td><td>10</td><td>2.3</td></tr><tr><td>RL-RNN 20 Beam + Aggregator</td><td>11.0</td><td>21.4</td><td>4.5</td><td>10</td><td>2.3</td></tr><tr><td>RL-10-Full</td><td>12.2</td><td>28.4</td><td>4.9</td><td>1</td><td>2.3</td></tr><tr><td>RL-10-Bagging</td><td>12.2</td><td>28.7</td><td>5.0</td><td>1</td><td>2.3</td></tr><tr><td>RL-10-Sub</td><td>12.3</td><td>29.7</td><td>5.5</td><td>1</td><td>2.3</td></tr><tr><td>RL-10-Sub (Pretrained)</td><td>12.5</td><td>29.8</td><td>5.4</td><td>10*+1</td><td>4.6</td></tr><tr><td>RL-10-Full (Extra Budget)</td><td>12.9</td><td>30.3</td><td>5.6</td><td>10</td><td>23.0</td></tr></table>

Table 1: MAP scores on the test sets of the document retrieval datasets. Similar results hold for other metrics (see Appendix A). *The weights of the agents are initialized from a single model pretrained for 10 days on the full training set.

# 4.4 BASELINES

LUCENE: We give the original query to Lucene and use the retrieved documents as results.

PRF: This is the pseudo relevance feedback method (Rocchio, 1971). We expand the original query with terms from the documents retrieved by the Lucene search engine using the original query. The top-N TF-IDF terms from each of the top-K retrieved documents are added to the original query, where N and K are selected by a grid search on the validation data.

RELEVANCE MODEL (RM3): This is our implementation of the relevance model for query expansion (Lavrenko & Croft, 2001). The probability of adding a term  $t$  to the original query is given by:

$$
P (t | q _ {0}) = (1 - \lambda) P ^ {\prime} (t | q _ {0}) + \lambda \sum_ {d \in D _ {0}} P (d) P (t | d) P (q _ {0} | d), \tag {4}
$$

where  $P(d)$  is the probability of retrieving the document  $d$ , assumed uniform over the set,  $P(t|d)$  and  $P(q_0|d)$  are the probabilities assigned by the language model obtained from  $d$  to  $t$  and  $q_0$ , respectively.  $P'(t|q_0) = \frac{\mathrm{tf}(t \in q)}{|q|}$ , where  $\mathrm{tf}(t,d)$  is the term frequency of  $t$  in  $d$ . We set the interpolation parameter  $\lambda$  to 0.65, which was the best value found by a grid-search on the development set.

We use a Dirichlet smoothed language model (Zhai & Lafferty, 2001) to compute a language model from a document  $d \in D_0$ :

$$
P (t | d) = \frac {\operatorname {t f} (t , d) + u P (t | C)}{| d | + u}, \tag {5}
$$

where  $u$  is a scalar constant ( $u = 1500$  in our experiments), and  $P(t|C)$  is the probability of  $t$  occurring in the entire corpus  $C$ .

We use the  $N$  terms with the highest  $P(t|q_0)$  in an expanded query, where  $N = 100$  was the best value found by a grid-search on the development set.

RL-RNN: This is the sequence-to-sequence model trained with reinforcement learning from Nogueira & Cho (2017). The reformulated query is formed by appending new terms to the original query. The terms are selected from the documents retrieved using the original query. The agent is trained from scratch.

![](images/673377159dc5cfc29743dd44c1af8efcedc9a1006102ca8337f2780a6d522471.jpg)  
Figure 2: Overall system's performance for different number of sub-agents.

![](images/eb795b69b930d4006e93947b6d8e60cd4472b8d8b3b2da24992ed1262911f60d.jpg)

![](images/d9599f9d7a59e6f7f37d15c8385855ee8e37c7feb5c32473dbf467943bcbded7.jpg)

RL-N-ENSEMBLE: We train  $N$  RL-RNN agents with different initial weights on the full training set. At test time, we average the probability distributions of all the  $N$  agents at each time step and select the token with the highest probability, as done by Sutskever et al. (2014).

# 4.5 PROPOSED MODELS

We evaluate the following variants of the proposed method:

RL-N-FULL: We train  $N$  RL-RNN agents with different initial weights on the full training set. The answers are obtained using the best (greedy) reformulations of all the agents and are given to the aggregator.

RL-N-BAGGING: This is the same as RL-N-Full but we construct the training set of each RL-RNN agent by sampling with replacement D times from the full training set, which has a size of D. This is known as the bootstrap sample and leads to approximately  $63\%$  unique samples, the rest being duplicates.

RL-N-SUB: This is the proposed agent. It is similar to RL-N-Full but the multiple sub-agents are trained on random partitions of the dataset (see Figure 1-(c)).

# 4.6 RESULTS

A summary of the document retrieval results is shown in Table 1. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and 2.7 TFLOPS as an estimate of the single-precision floating-point of a K80 GPU.

Since the sub-agents are frozen during the training of the aggregator, we pre-compute all  $(q_0,q_i,a_i,r_i)$  tuples from the training set, thus avoiding sub-agent or environment calls. This reduces its training time to less than 6 hours  $(0.06\times 10^{18}$  FLOPs). Since this cost is negligible when compared to the sub-agents', we do not include it in the table.

The proposed methods (RL-10-{Sub, Bagging, Full}) have  $20 - 60\%$  relative performance improvement over the standard ensemble (RL-10-Ensemble) while training ten times faster. More interestingly, RL-10-Sub has a better performance than the single-agent version (RL-RNN), uses the same computational budget, and trains on a fraction of the time. Lastly, we found that RL-10-Sub (Pretrained) has the best balance between performance and training cost across all datasets.

For an analysis of the aggregator's contribution to the overall performance, see Appendix C.

NUMBER OF SUB-AGENTS: We compare the performance of the full system (reformulators + aggregator) for different numbers of agents in Figure 2. The performance of the system is stable across all datasets after more than ten sub-agents are used, thus indicating the robustness of the proposed method. For more experiments regarding training stability, see Appendix D.

<table><tr><td rowspan="2"></td><td colspan="2">Dev</td><td colspan="2">Test</td><td rowspan="2">Training (Days)</td><td rowspan="2">FLOPs (×1018)</td></tr><tr><td>F1</td><td>Oracle</td><td>F1</td><td>Oracle</td></tr><tr><td>BiDAF (Seo et al., 2016)</td><td>37.9</td><td>-</td><td>34.6</td><td>-</td><td>N/A</td><td></td></tr><tr><td>R3 (Wang et al., 2017a)</td><td>-</td><td>-</td><td>55.3</td><td>-</td><td>N/A</td><td></td></tr><tr><td>Re-Ranker (Wang et al., 2017b)</td><td>-</td><td>-</td><td>60.6</td><td>-</td><td>N/A</td><td></td></tr><tr><td>AQA (Buck et al., 2018b)</td><td>47.4</td><td>56.0</td><td>45.6</td><td>53.8</td><td>10</td><td>4.6</td></tr><tr><td>AQA-10-Sub</td><td>51.7</td><td>66.8</td><td>49.0</td><td>61.5</td><td>1</td><td>4.6</td></tr><tr><td>AQA-10-Full</td><td>51.0</td><td>61.2</td><td>48.4</td><td>58.7</td><td>1</td><td>4.6</td></tr><tr><td>AQA-10-Full (extra budget)</td><td>51.4</td><td>61.3</td><td>50.5</td><td>58.9</td><td>10</td><td>46.0</td></tr></table>

Table 2: Main result on the question-answering task (SearchQA dataset). We did not include the training cost of the aggregator (0.2 days,  $0.06 \times 10^{18}$  FLOPs).

# 5 QUESTION-ANSWERING

To further assess the effectiveness of the proposed method, we conduct experiments in a question-answering task, comparing our agent with the active question answering agent proposed by Buck et al. (2018b).

The environment receives a question as an action and returns an answer as an observation, and a reward computed against a ground truth answer. We use BiDAF as the question-answering system (Seo et al., 2016). Given a question, it outputs an answer span from a list of snippets. We use as a reward the token level F1 score on the answer (see Section 5.3 for its definition).

We follow Buck et al. (2018b) to train BiDAF. We emphasize that BiDAF's parameters are frozen when we train and evaluate the reformulation system. Training and evaluation are performed on the SearchQA dataset (Dunn et al., 2017). The data contains Jeopardy! clues as questions. Each clue has a correct answer and a list of 50 snippets from Google's top search results. The training, validation and test sets contain 99,820, 13,393 and 27,248 examples, respectively.

# 5.1 BASELINES AND BENCHMARKS

We compare our agent against the following baselines and benchmarks:

BIDAF: The original question is given to the question-answering system without any modification (see Figure 1-(a)).

RE-RANKER AND  $\mathbf{R}^3$ : Re-Ranker is the best model from Wang et al. (2017b). They use an answer re-ranking approach to reorder the answer candidates generated by a base Q&A model,  $\mathbf{R}^3$  (Wang et al., 2017a). We report both systems' results as a reference. To the best of our knowledge, they are currently the best systems on SearchQA.  $\mathbf{R}^3$  alone, without re-ranking, outperforms BiDAF by about 20 F1 points.

AQA: This is the best model from Buck et al. (2018b). It consists of a reformulator and a selector. The reformulator is a subword-based sequence-to-sequence model that produces twenty reformulations of an input question using beam search. The reformulations and their answers are given to the selector which then chooses one of the answers as final (see Figure 1-(b)). The reformulator is pretrained on translation and paraphrase data.

# 5.2 PROPOSED METHODS

AQA-N-{FULL, SUB}: Similar to the RL-N-{Full, Sub} models, we use AQA reformulators as the sub-agents followed by an aggregator to create AQA-N-Full and AQA-N-Sub models, whose sub-agents are trained on the full and random partitions of the dataset, respectively. For the training and hyperparameter details, see Appendix B.2.

<table><tr><td>Method</td><td>pCos ↓</td><td>pBLEU ↓</td><td>PINC ↑</td><td>Length Std ↑</td><td>F1 ↑</td><td>Oracle ↑</td></tr><tr><td>AQA</td><td>66.4</td><td>45.7</td><td>58.7</td><td>3.8</td><td>47.7</td><td>56.0</td></tr><tr><td>AQA-10-Full</td><td>29.5</td><td>26.6</td><td>79.5</td><td>9.2</td><td>51.0</td><td>61.2</td></tr><tr><td>AQA-10-Sub</td><td>14.2</td><td>12.8</td><td>94.5</td><td>11.7</td><td>51.4</td><td>61.3</td></tr></table>

Table 3: Diversity scores of reformulations from different methods. For pBLEU and pCos, lower values mean higher diversity. Notice that higher diversity scores are associated with higher F1 and oracle scores.

# 5.3 EVALUATION METRICS

F1: We use the macro-averaged F1 score as the main metric. It measures the average bag of tokens overlap between the prediction and ground truth answer. We take the F1 over the ground truth answer for a given question and then average over all of the questions.

ORACLE: Additionally, we present the oracle performances, which are from a perfect aggregator that predicts  $s_j^R = 1$  for relevant answers and  $s_j^R = 0$ , otherwise.

# 5.4 RESULTS

Results are presented in Table 2. The proposed method (AQA-10-{Full, Sub}) have both better F1 and oracle performances than the single-agent AQA method, while training in one-tenth of the time. Even when the ensemble method is given ten times more training time (AQA-10-Full, extra budget), our method achieves a higher performance.

The best model outperforms BiDAF, which is used in our environment, by almost 16 F1 points. In absolute terms, the proposed method does not reach the performance of the Re-Ranker or underlying  $\mathbf{R}^3$  system. It is important to realize, though, that these are orthogonal issues: any Q&A system, including  $\mathbf{R}^3$ , could be used as environments, including re-ranking post-processing. We leave this as a future work.

ORIGINAL QUERY CONTRIBUTION: We observe a drop in F1 of approximately  $1\%$  when the original query is removed from the pool of reformulations, which shows that the gains come mostly from the multiple reformulations and not from the aggregator falling back on selecting the original query.

# 5.5 QUERY DIVERSITY

Here we evaluate how query diversity and performance are related. For that, we use four metrics (defined in Appendix E): pCos, pBLEU, PINC, and Length Std.

Table 3 shows that the multiple agents trained on partitions of the dataset (AQA-10-Sub) produce more diverse queries than a single agent with beam search (AQA) and multiple agents trained on the full training set (AQA-10-Full). This suggests that its higher performance can be partly attributed to the higher diversity of the learned policies.

# 6 CONCLUSION

We proposed a method to build a better query reformulation system by training multiple sub-agents on partitions of the data using reinforcement learning and an aggregator that learns to combine the answers of the multiple agents given a new query. We showed the effectiveness and efficiency of the proposed approach on the tasks of document retrieval and question answering. One interesting orthogonal extension would be to introduce diversity on the beam search decoder (Vijayakumar et al., 2016; Li et al., 2016), thus shedding light on the question of whether the gains come from the increased capacity of the system due to the use of the multiple agents, the diversity of reformulations, or both.

# REFERENCES

Rohan Anil, Gabriel Pereyra, Alexandre Passos, Robert Ormandi, George E Dahl, and Geoffrey E Hinton. Large scale distributed neural network training through online distillation. arXiv preprint arXiv:1804.03235, 2018.  
Dzmitry Bahdanau, Philemon Brakel, Kelvin Xu, Anirudh Goyal, Ryan Lowe, Joelle Pineau, Aaron Courville, and Yoshua Bengio. An actor-critic algorithm for sequence prediction. arXiv preprint arXiv:1607.07086, 2016.  
Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. J. Artif. Intell. Res.(JAIR), 47:253-279, 2013.  
Leo Breiman. Bagging predictors. Machine learning, 24(2):123-140, 1996a.  
Leo Breiman. Bias, variance, and arcing classifiers (technical report 460). Statistics Department, University of California, 1996b.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.  
Christian Buck, Jannis Bulian, Massimiliano Ciaramita, Wojciech Gajewski, Andrea Gesmundo, Neil Houlsby, and Wei Wang. Analyzing language learned by an active question answering agent. arXiv preprint arXiv:1801.07537, 2018a.  
Christian Buck, Jannis Bulian, Massimiliano Ciaramita, Andrea Gesmundo, Neil Houlsby, Wojciech Gajewski, and Wei Wang. Ask the right questions: Active question reformulation with reinforcement learning. In Proceedings of ICLR, 2018b.  
Boxing Chen and Colin Cherry. A systematic comparison of smoothing techniques for sentence-level bleu. In Proceedings of the Ninth Workshop on Statistical Machine Translation, pp. 362-367, 2014.  
David L Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies-Volume 1, pp. 190-200. Association for Computational Linguistics, 2011.  
Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.  
Edoardo Conti, Vashisht Madhavan, Felipe Petroski Such, Joel Lehman, Kenneth O Stanley, and Jeff Clune. Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents. arXiv preprint arXiv:1712.06560, 2017.  
Peter Dayan and Geoffrey E Hinton. Feudal reinforcement learning. In Advances in neural information processing systems, pp. 271-278, 1993.  
Thomas G Dietterich. Hierarchical reinforcement learning with the maxq value function decomposition. J. Artif. Intell. Res.(JAIR), 13(1):227-303, 2000.  
Laura Dietz and Gamari Ben. Trec car: A data set for complex answer retrieval. http://trec-car.cs.unh.edu, 2017.  
Matthew Dunn, Levent Sagun, Mike Higgins, Ugur Guney, Volkan Cirik, and Kyunghyun Cho. Searchqa: A new q&a dataset augmented with context from a search engine. arXiv preprint arXiv:1704.05179, 2017.  
Michael Fairbank and Eduardo Alonso. The divergence of reinforcement learning algorithms with value-iteration and function approximation. arXiv preprint arXiv:1107.4606, 2011.  
Yoav Freund. Boosting a weak learning algorithm by majority. Information and computation, 121 (2):256-285, 1995.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Minghao Hu, Yuxing Peng, and Xipeng Qiu. Reinforced mnemonic reader for machine comprehension. CoRR, abs/1705.02798, 2017.  
Kai Hui, Andrew Yates, Klaus Berberich, and Gerard de Melo. Pacrr: A position-aware neural in model for relevance matching. arXiv preprint arXiv:1704.03940, 2017.  
Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79-87, 1991.  
Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. Neural computation, 6(2):181-214, 1994.  
Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. Journal of artificial intelligence research, 4:237-285, 1996.  
Yoon Kim. Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882, 2014.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Victor Lavrenko and W Bruce Croft. Relevance based language models. In Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval, pp. 120-127. ACM, 2001.  
Jiwei Li, Will Monroe, and Dan Jurafsky. A simple, fast diverse decoding algorithm for neural generation. arXiv preprint arXiv:1611.08562, 2016.  
Long-Ji Lin. Reinforcement learning for robots using neural networks. Technical report, Carnegie Mellon Univ Pittsburgh PA School of Computer Science, 1993.  
Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529, 2015.  
Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, pp. 1928-1937, 2016.  
Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th international conference on machine learning (ICML-10), pp. 807-814, 2010.  
Rodrigo Nogueira and Kyunghyun Cho. End-to-end goal-driven web navigation. In Advances in Neural Information Processing Systems, pp. 1903-1911, 2016.  
Rodrigo Nogueira and Kyunghyun Cho. Task-oriented query reformulation with reinforcement learning. arXiv preprint arXiv:1704.04572, 2017.  
Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via bootstrapped dqn. In Advances in neural information processing systems, pp. 4026-4034, 2016.  
Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.

Matteo Pirotta, Marcello Restelli, and Luca Bascetta. Adaptive step-size for policy gradient methods. In Advances in Neural Information Processing Systems, pp. 1394-1402, 2013.  
Marc'Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with recurrent neural networks. arXiv preprint arXiv:1511.06732, 2015.  
Joseph John Rocchio. Relevance feedback in information retrieval. The SMART retrieval system: experiments in automatic document processing, pp. 313-323, 1971.  
Andrei A Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, and Raia Hadsell. Policy distillation. arXiv preprint arXiv:1511.06295, 2015.  
David Sculley. Web-scale k-means clustering. In Proceedings of the 19th international conference on World wide web, pp. 1177-1178. ACM, 2010.  
Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. Bidirectional attention flow for machine comprehension. arXiv preprint arXiv:1611.01603, 2016.  
Iulian V Serban, Chinnadhurai Sankar, Mathieu Germain, Saizheng Zhang, Zhouhan Lin, Sandeep Subramanian, Taesup Kim, Michael Pieper, Sarath Chandar, Nan Rosemary Ke, et al. A deep reinforcement learning chatbot. arXiv preprint arXiv:1709.02349, 2017.  
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.  
Satinder P Singh. Reinforcement learning with a hierarchy of abstract models. In AAAI, pp. 202-207, 1992.  
Christopher Stanton and Jeff Clune. Curiosity search: producing generalists by encouraging individuals to continually explore and acquire skills throughout their lifetime. *PloS one*, 11(9):e0162235, 2016.  
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pp. 3104-3112, 2014.  
JN Tsitsiklis and B Van Roy. An analysis of temporal-difference learning with function approximation technical. Technical report, Report LIDS-P-2322). Laboratory for Information and Decision Systems, Massachusetts Institute of Technology, 1996.  
Ashwin K Vijayakumar, Michael Cogswell, Ramprasath R Selvaraju, Qing Sun, Stefan Lee, David Crandall, and Dhruv Batra. Diverse beam search: Decoding diverse solutions from neural sequence models. arXiv preprint arXiv:1610.02424, 2016.  
Shuohang Wang, Mo Yu, Xiaoxiao Guo, Zhiguo Wang, Tim Klinger, Wei Zhang, Shiyu Chang, Gerald Tesauro, Bowen Zhou, and Jing Jiang. R3: Reinforced reader-ranker for open-domain question answering. arXiv preprint arXiv:1709.00023, 2017a.  
Shuohang Wang, Mo Yu, Jing Jiang, Wei Zhang, Xiaoxiao Guo, Shiyu Chang, Zhiguo Wang, Tim Klinger, Gerald Tesauro, and Murray Campbell. Evidence aggregation for answer re-ranking in open-domain question answering. arXiv preprint arXiv:1711.05116, 2017b.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.  
Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.  
Chengxiang Zhai and John Lafferty. A study of smoothing methods for language models applied to ad hoc information retrieval. In Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval, pp. 334-342. ACM, 2001.
