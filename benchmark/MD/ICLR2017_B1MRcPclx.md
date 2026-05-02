# QUERY-REDUCTION NETWORKS FOR QUESTION ANSWERING

Minjoon Seo $^{1}$ , Sewon Min $^{3}$ , Ali Farhadi $^{1,2}$  & Hannaneh Hajishirzi $^{1}$

University of Washington<sup>1</sup>, Allen Institute for Artificial Intelligence<sup>2</sup>, Seoul National University<sup>3</sup> {minjoon, ali, hannaneh}@cs.washington.edu, shmsw25@snu.ac.kr

# ABSTRACT

In this paper, we study the problem of question answering when reasoning over multiple facts is required. We propose Query-Reduction Network(QRN), a variant of Recurrent Neural Network (RNN) that effectively handles both short-term (local) and long-term (global) sequential dependencies to reason over multiple facts. QRN considers the context sentences as a sequence of state-changing triggers, and reduces the original query to an easier-to-answer query as it observes each trigger (context sentence) through time. Our experiments show that QRN produces the state-of-the-art results in bAbI QA and dialog tasks, and in a real goal-oriented dialog dataset. In addition, QRN formulation allows parallelization on RNN's time axis, saving an order of magnitude in time complexity for training and inference.

# 1 INTRODUCTION

In this paper, we address the problem of question answering (QA) when reasoning over multiple facts is required. For example, consider we know that Frogs eat insects and Flies are insects. Then answering Do frogs eat flies? requires reasoning over both of the above facts. Question answering, more specifically context-based QA, has been extensively studied in Machine comprehension (Richardson et al., 2013; Hermann et al., 2015; Hill et al., 2016; Rajpurkar et al., 2016) in NLP. However, most of the datasets used for machine comprehension are primarily focused on lexical and syntactic understanding, and hardly concentrate on inference over multiple facts. Recently, several datasets primarily focused on testing multi-hop reasoning have emerged. Among them are story-based QA (Weston et al., 2016) and the dialog task (Bordes and Weston, 2016), each of which requires several different types of multi-hop reasoning.

Recurrent Neural Network (RNN) and its variants, such as Long Short-Term Memory (LSTM) (Hochreiter and Schmidhuber, 1997) and Gated Recurrent Unit (GRU) (Cho et al., 2014), are popular choices for modeling natural language. However, when used for multi-hop reasoning in question answering, purely RNN-based models have shown to perform poorly (Weston et al., 2016). This is largely due to the fact that RNN's internal memory is inherently unstable over a long term. For this reason, most recent approaches in the literature have mainly relied on global attention mechanism and shared external memory (Sukhbaatar et al., 2015; Peng et al., 2015; Xiong et al., 2016; Graves et al., 2016). The attention mechanism allows these models to focus on a single sentence in each layer. They can sequentially read multiple relevant sentences from the memory with multiple layers to perform multi-hop reasoning. However, one major drawback of these standard attention mechanisms is that they are insensitive to the time step (memory address) of the sentences when accessing them.

Our proposed model, Query-Reduction Network $^1$  (QRN), is a single recurrent unit that addresses the long-term dependency problem of most RNN-based models by simplifying the recurrent update, while taking the advantage of RNN's capability to model sequential data (Figure 1). QRN considers the context sentences as a sequence of state-changing triggers, and transforms (reduces) the original query to an easier-to-answer query as it observes each trigger through time. For instance in Figure 1b, the original question, Where is the apple?, cannot be directly answered by any single sentence from the story. After observing the first sentence, Sandra got the apple there, QRN transforms the original question to a reduced query Where is Sandra?, which is presumably

![](images/00753289d86fab02045144f3de574e0fea568ad90d441bdb7b654dff3c71c250.jpg)  
(a) QRN unit

![](images/9b3a17488e605c00b27862a601bfd63590cdbcbcfe9bc7515f6f36e5a1e00679.jpg)  
(b) 2-layer QRN

![](images/907e8619750680d859c30b90cafbd43cfac8502f53e33683eff771f2cd3cbf12.jpg)  
(c) Overview  
Figure 1: (1a) QRN unit, (1b) 2-layer QRN on 5-sentence story, and (1c) entire QA system (QRN and input / output modules).  $\mathbf{x},\mathbf{q},\hat{\mathbf{y}}$  are the story, question and predicted answer in natural language, respectively.  $\mathbf{x} = \langle \mathbf{x}_1,\dots ,\mathbf{x}_T\rangle ,\mathbf{q},\hat{\mathbf{y}}$  are their corresponding vector representations (upright font).  $\alpha$  and  $\rho$  are update gate and reduce functions, respectively.  $\hat{\mathbf{y}}$  is assigned to be  $\mathbf{h}_5^2$ , the local query at the last time step in the last layer. Also, red-colored text is the inferred meanings of the vectors to aid the understanding of the model.

easier to answer than the original question given the context provided by the first sentence.2 Unlike RNN-based models, QRN's candidate state  $(\tilde{\mathbf{h}}_t$  in Figure 1a) does not depend on the previous hidden state  $(\mathbf{h}_{t - 1})$ . Compared to memory-based approaches (Weston et al., 2015; Sukhbaatar et al., 2015; Peng et al., 2015; Kumar et al., 2016; Xiong et al., 2016), QRN can better encode locality information because it does not use a global memory access controller (circle nodes in Figure 2), and the query updates are performed locally.

In short, the main contribution of QRN is threefold. First, QRN is a simple variant of RNN that reduces the query given the context sentences in a differentiable manner. Second, QRN is situated between the attention mechanism and RNN, effectively handling time dependency and long-term dependency problems of each technique, respectively. Hence it is well-suited for sequential data with both local and global interactions (note that QRN is not the replacement of RNN, which is arguably better for modeling complex local interactions). Third, unlike most RNN-based models, QRN can be parallelized over time by computing candidate reduced queries  $(\tilde{\mathbf{h}}_t)$  directly from local input queries  $(\mathbf{q}_t)$  and context sentence vectors  $(\mathbf{x}_t)$ . In fact, the parallelizability of QRN implies that QRN does not suffer from the vanishing gradient problem of RNN, hence effectively addressing the long-term dependency. We experimentally demonstrate these contributions by achieving the state-of-the-art results on story-based QA and interactive dialog datasets.

# 2 MODEL

In story-based QA (or dialog dataset), the input is the context as a sequence of sentences (story or past conversations) and a question in natural language (equivalent to the user's last utterance in the dialog). The output is the predicted answer to the question in natural language (the system's next utterance in the dialog). The only supervision provided during training is the answer to the question.

In this paper we particularly focus on end-to-end solutions i.e., the only supervision comes from questions and answers, and we restrain from using manually defined rules or external language resources, such as lexicon or dependency parser. Let  $\langle x_1,\ldots ,x_T\rangle$  denote the sequence of sentences, where  $T$  is the number of sentences in the story, and let  $\pmb{q}$  denote the question. Let  $\hat{\pmb{y}}$  denote the predicted answer, and  $\pmb{y}$  denote the true answer. Our proposed system for end-to-end QA task is divided into three modules (Figure 1c): input module, QRN layers, and output module.

Input module. Input module maps each sentence  $\mathbf{x}_t$  and the question  $\mathbf{q}$  to  $d$ -dimensional vector space,  $\mathbf{x}_t \in \mathbb{R}^d$  and  $\mathbf{q}_t \in \mathbb{R}^d$ . We adopt a previous solution for the input module (details in Section 5).

QRN layers. QRN layers use the sentence vectors and the question vector from the input module to obtain the predicted answer in vector space,  $\hat{\mathbf{y}}\in \mathbb{R}^d$ . A QRN layer refers to the recurrent application of a QRN unit, which can be considered as a variant of RNN with two inputs, two outputs, and a

hidden state (reduced query), all of which operate in vector space. The details of the QRN module is explained throughout this section (2.1, 2.2).

Output module. Output module maps  $\hat{\mathbf{y}}$  obtained from QRN to a natural language answer  $\hat{\pmb{y}}$ . Similar to the input module, we adopt a standard solution for the output module (details in Section 5).

We first formally define the base model of a QRN unit, and then we explain how we connect the input and output modules to it (Section 2.1). We also present a few extensions to the network that can improve QRN's performance (Section 2.2). Finally, we show that QRN can be parallelized over time, giving computational advantage over most RNN-based models by one order of magnitude (Section 3).

# 2.1 QRN UNIT

As an RNN-based model, QRN is a single recurrent unit that updates its hidden state (reduced query) through time and layers. Figure 1a depicts the schematic structure of a QRN unit, and Figure 1b demonstrates how layers are stacked. A QRN unit accepts two inputs (local query vector  $\mathbf{q}_t \in \mathbb{R}^d$  and sentence vector  $\mathbf{x}_t \in \mathbb{R}^d$ ), and two outputs (reduced query vector  $\mathbf{h}_t \in \mathbb{R}^d$ , which is similar to the hidden state in RNN, and the sentence vector  $\mathbf{x}_t$  from the input without modification). The local query vector is not necessarily identical to the original query (question) vector  $\mathbf{q}$ . In order to compute the outputs, we use update gate function  $\alpha: \mathbb{R}^d \times \mathbb{R}^d \to [0,1]$  and reduce function  $\rho: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d$ . Intuitively, the update gate function measures the relevance between the sentence and the local query and is used to update the hidden state. The reduce function transforms the local query input to a candidate state which is a new reduced (easier) query given the sentence. The outputs are calculated with the following equations:

$$
z _ {t} = \alpha \left(\mathbf {x} _ {t}, \mathbf {q} _ {t}\right) = \sigma \left(\mathbf {W} ^ {(z)} \left(\mathbf {x} _ {t} \circ \mathbf {q} _ {t}\right) + b ^ {(z)}\right) \tag {1}
$$

$$
\tilde {\mathbf {h}} _ {t} = \boldsymbol {\rho} \left(\mathbf {x} _ {t}, \mathbf {q} _ {t}\right) = \tanh  \left(\mathbf {W} ^ {(\mathbf {h})} \left[ \mathbf {x} _ {t}; \mathbf {q} _ {t} \right] + \mathbf {b} ^ {(\mathbf {h})}\right) \tag {2}
$$

$$
\mathbf {h} _ {t} = z _ {t} \tilde {\mathbf {h}} _ {t} + (1 - z _ {t}) \mathbf {h} _ {t - 1} \tag {3}
$$

where  $z_{t}$  is the scalar update gate,  $\tilde{\mathbf{h}}_t$  is the candidate reduced query, and  $\mathbf{h}_t$  is the final reduced query at time step  $t$ ,  $\sigma(\cdot)$  is sigmoid activation,  $\tanh(\cdot)$  is hyperbolic tangent activation (applied element-wise),  $\mathbf{W}^{(z)} \in \mathbb{R}^{1 \times d}$ ,  $\mathbf{W}^{(\mathbf{h})} \in \mathbb{R}^{d \times 2d}$  are weight matrices,  $b^{(z)} \in \mathbb{R}$ ,  $\mathbf{b}^{(\mathbf{h})} \in \mathbb{R}^d$  are bias terms,  $\circ$  is element-wise vector multiplication, and  $[;]$  is vector concatenation along the row. As a base case,  $\mathbf{h}_0 = \mathbf{0}$ . Here we have explicitly defined  $\alpha$  and  $\rho$ , but they can be any reasonable differentiable functions.

The update gate is similar to the global attention mechanism (Sukhbaatar et al., 2015; Xiong et al., 2016) in that it measures the similarity between the sentence (a memory slot) and the query. However, a significant difference is that the update gate is computed using sigmoid  $(\sigma)$  function on the current memory slot only (hence internally embedded within the unit), whereas the global attention is computed using softmax function over the entire memory (hence globally defined). The update gate can be rather considered as local sigmoid attention.

Stacking layers We just showed the single-layer case of QRN, but QRN with multiple layers is able to perform reasoning over multiple facts more effectively, as shown in the example of Figure 1b. In order to stack several layers of QRN, the outputs of the current layer are used as the inputs to the next layer. That is, using superscript  $k$  to denote the current layer's index (assuming 1-based indexing), we let  $\mathbf{q}_t^{k+1} = \mathbf{h}_t^k$ . Note that  $\mathbf{x}_t$  is passed to the next layer without any modification, so we do not put a layer index on it.

Bi-direction. So far we have assumed that QRN only needs to look at past sentences, whereas often times, query answers can depend on future sentences. For instance, consider a sentence "John dropped the football" at time  $t$ . Then, even if there is no mention about the "football" in the past (at time  $i < t$ ), it can be implied that "John" has the "football" at the current time  $t$ . In order to incorporate the future dependency, we obtain  $\vec{\mathbf{h}}_t$  and  $\vec{\mathbf{h}}_t$  in both forward and backward directions, respectively, using Equation 3. We then add them together to get  $\mathbf{q}_t$  for the next layer. That is,

$$
\mathbf {q} _ {t} ^ {k + 1} = \overrightarrow {\mathbf {h}} _ {t} ^ {k} + \overleftarrow {\mathbf {h}} _ {t} ^ {k} \tag {4}
$$

for layer indices  $1 \leq k \leq K - 1$ . Note that the variables  $\mathbf{W}^{(z)}, b^{(z)}, \mathbf{W}^{(\mathbf{h})}, \mathbf{b}^{(\mathbf{h})}$  are shared between the two directions.

Connecting input and output modules. Figure 1c depicts how QRN is connected with the input and output modules. In the first layer of QRN,  $\mathbf{q}_t^1 = \mathbf{q}$  for all  $t$ , where  $\mathbf{q}$  is obtained from the input module by processing the natural language question input  $\pmb{q}$ .  $\mathbf{x}_t$  is also obtained from  $\pmb{x}_t$  by the same input module. The output at the last time step in the last layer is passed to the output module. That is,  $\hat{\mathbf{y}} = \mathbf{h}_t^K$  where  $K$  represents the number of layers in the network. Then the output module gives the predicted answer  $\hat{\pmb{y}}$  in natural language.

# 2.2 EXTENSIONS

Here we introduce a few extensions of QRN, and later in our experiments, we test QRN's performance with and without each of these extensions.

Reset gate. Inspired by GRU (Cho et al., 2014), we found that it is useful to allow the QRN unit to reset (nullify) the candidate reduced query (i.e.,  $\tilde{\mathbf{h}}_t$ ) when necessary. For this we use a reset gate function  $\beta : \mathbb{R}^d \times \mathbb{R}^d \to [0,1]$ , which can be defined similarly to the update gate function:

$$
r _ {t} = \beta \left(\mathbf {x} _ {t}, \mathbf {q} _ {t}\right) = \sigma \left(\mathbf {W} ^ {(r)} \left(\mathbf {x} _ {t} \circ \mathbf {q} _ {t}\right) + b ^ {(r)}\right) \tag {5}
$$

where  $\mathbf{W}^{(r)}\in \mathbb{R}^{1\times d}$  is a weight matrix, and  $b^{(r)}\in \mathbb{R}$  is a bias term. Equation 3 is rewritten as

$$
\mathbf {h} _ {t} = z _ {t} r _ {t} \tilde {\mathbf {h}} _ {t} + (1 - z _ {t}) \mathbf {h} _ {t - 1}. \tag {6}
$$

Note that we do not use the reset gate in the last layer.

Vector gates. As in LSTM and GRU, update and reset gates can be vectors instead of scalar values for fine-controlled gating. For vector gates, we modify the row dimension of weights and biases in Equation 1 and 5 from 1 to  $d$ . Then we obtain  $\mathbf{z}_t, \mathbf{r}_t \in \mathbb{R}^d$  (instead of  $z_t, r_t \in \mathbb{R}$ ), and these can be element-wise multiplied ( $\circ$ ) instead of being broadcasted in Equation 3 and 6.

# 3 PARALLELIZATION

An important advantage of QRN is that the recurrent updates in Equation 3 and 5 can be computed in parallel across time. This is in contrast with most RNN-based models that cannot be parallelized, where computing the candidate hidden state at time  $t$  explicitly requires the previous hidden state. In QRN, the final reduced queries  $(\mathbf{h}_t)$  can be decomposed into computing over candidate reduced queries  $(\tilde{\mathbf{h}}_t)$ , without looking at the previous reduced query. Here we primarily show that the query update in Equation 3 can be parallelized by rewriting the equation with matrix operations. The extension to Equation 5 is straightforward. The proof for QRN with vector gates is shown in Appendix B. The recursive definition of Equation 3 can be explicitly written as

$$
\mathbf {h} _ {t} = \sum_ {i = 1} ^ {t} \left[ \prod_ {j = i + 1} ^ {t} 1 - z _ {j} \right] z _ {i} \tilde {\mathbf {h}} _ {i} = \sum_ {i = 1} ^ {t} \exp \left\{\sum_ {j = i + 1} ^ {t} \log \left(1 - z _ {j}\right) \right\} z _ {i} \tilde {\mathbf {h}} _ {i} \tag {7}
$$

where  $\exp (\cdot)$  and  $\log (\cdot)$  are exponential and logarithm functions, respectively. Let  $b_{i} = \log (1 - z_{i})$  for brevity. Then we can rewrite Equation 7 as the following equation:

$$
\left( \begin{array}{c} \mathbf {h} _ {1} ^ {\top} \\ \mathbf {h} _ {2} ^ {\top} \\ \mathbf {h} _ {3} ^ {\top} \\ \vdots \\ \mathbf {h} _ {T} ^ {\top} \end{array} \right) = \left[ \exp \left\{\left( \begin{array}{c c c c c} 0 & - \infty & - \infty & \dots & - \infty \\ b _ {2} & 0 & - \infty & \dots & - \infty \\ b _ {2} + b _ {3} & b _ {3} & 0 & \dots & - \infty \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \sum_ {j = 2} ^ {T} b _ {j} & \sum_ {j = 3} ^ {T} b _ {j} & \sum_ {j = 4} ^ {T} b _ {j} & \dots & 0 \end{array} \right) \right\} \right] \left( \begin{array}{c} z _ {1} \tilde {\mathbf {h}} _ {1} ^ {\top} \\ z _ {2} \tilde {\mathbf {h}} _ {2} ^ {\top} \\ z _ {3} \tilde {\mathbf {h}} _ {3} ^ {\top} \\ \vdots \\ z _ {T} \tilde {\mathbf {h}} _ {T} ^ {\top} \end{array} \right) \tag {8}
$$

Let  $\mathbf{H} = [\mathbf{h}_1^\top ;\dots ;\mathbf{h}_T^\top ]$  be a  $T$  -by-  $d$  matrix where the transposes  $(\top)$  of the column vectors  $\mathbf{h}_t$  are concatenated across row. We similarly define  $\tilde{\mathbf{H}}$  from  $\tilde{\mathbf{h}}_t$ . Also, let  $\mathbf{z} = [z_{1};\ldots ;z_{T}]$  and  $\mathbf{b} = [0;b_{2};\ldots ;b_{T}]$  be column vectors (note that we use 0 instead of  $b_{1}$ ). Then Equation 8 is:

$$
\mathbf {H} = \left[ \mathbf {L} \circ \exp \left(\mathbf {L} \left[ \mathbf {B} \circ \mathbf {L} ^ {\prime} \right]\right) \right] \left[ \mathbf {Z} \circ \tilde {\mathbf {H}} \right] \tag {9}
$$

where  $\mathbf{L},\mathbf{L}^{\prime}\in \mathbb{R}^{T\times T}$  are lower and strictly lower triangular matrices of 1's, respectively,  $\circ$  is elementwise multiplication, and  $\mathbf{B}$  is a matrix where  $T$  b's are tiled across the column, i.e.  $\mathbf{B} = [\mathbf{b},\dots ,\mathbf{b}]\in$

![](images/bb473ace7c5ff69b07ede6a971946c31a6f1474586688711ea04abef726f385f.jpg)  
(a) QRN

![](images/9d17d662bf627eccaee8b8fdf04e0f0d4622270ce322e306390f9539e316eef1.jpg)  
Figure 2: The schematics of QRN and the two state-of-the-art models, End-to-End Memory Networks (N2N) and Improved Dynamic Memory Networks  $(\mathrm{DMN}+)$ , simplified to emphasize the differences among the models. AGRU is a variant of GRU where the update gate is replaced with soft attention, proposed by Kumar et al. (2016). For QRN and  $\mathrm{DMN}+$ , only forward direction arrows are shown.

![](images/6a18cbac4b03d48e3cca6822dc10cee53c8ec9f2bd4cbefa4da872e2913cf874.jpg)  
(b) N2N (Sukhbaatar et al., 2015)  
(c) DMN+ (Xiong et al., 2016)

$\mathbb{R}^{T\times T}$ , and similarly  $\mathbf{Z} = [\mathbf{z},\dots ,\mathbf{z}]\in \mathbb{R}^{T\times d}$ . All implicit operations are matrix multiplications. With reasonable  $N$  (batch size),  $d$  and  $T$  (e.g.  $N,d,T = 100$ ), matrix operations in Equation 9 can be comfortably computed in most modern GPUs.

# 4 RELATED WORK

Here, we primarily describe the most related approaches, which perform multi-hop reasoning over a given context (either story or dialog) in an end-to-end fashion (sketched in Figure 2). In our experiments we show that QRNs outperform all these models.

QRN is inspired by RNN-based models with gating mechanism, such as LSTM (Hochreiter and Schmidhuber, 1997) and GRU (Cho et al., 2014). While GRU and LSTM use the previous hidden state and the current input to obtain the candidate hidden state, QRN only uses the current two inputs to obtain the candidate reduced query (equivalent to candidate hidden state). We conjecture that this not only gives computational advantage via parallelization, but also makes training easier, i.e., avoiding vanishing gradient (which is critical for long-term dependency), overfitting (by simplifying the model), and converging to local minima.

End-to-end Memory Network (N2N) (Sukhbaatar et al., 2015) uses external memory with multi-layer attention mechanism to focus on sentences that are relevant to the question. There are two key differences between N2N and our QRN. First, N2N summarizes the entire memory in each layer to control the attention in the next layer (circle nodes in Figure 2b). Instead, QRN does not have any controller node (Figure 2a) and is able to focus on relevant sentences through the update gate that is internally embodied within its unit. Second, N2N adds time-dependent trainable weights to the sentence representations to model the time dependency of the sentences (as discussed in Section 1). QRN does not need such additional weights as its inherent RNN architecture allows QRN to effectively model the time dependency. Neural Reasoner (Peng et al., 2015) and Gated End-to-end Memory Network (Perez and Liu, 2016)) are variants of MemN2N that share its fundamental characteristics.

Improved Dynamic Memory Network (DMN+) (Xiong et al., 2016) uses the hybrid of the attention mechanism and the RNN architecture to model the sequence of sentences. It consists of two distinct GRUs, one for the time axis (rectangle nodes in Figure 2c) and one for the layer axis (circle nodes in Figure 2c). Note that the update gate of the GRU for the time axis is replaced with external softmax attention weights. DMN+ uses the time-axis GRU to summarize the entire memory in each layer, and then the layer-axis GRU controls the attention weights in each layer. In contrast, QRN is simply a single recurrent unit without any controller node.

# 5 EXPERIMENTS

# 5.1 DATA

bAbI story-based QA dataset bAbI story-based QA dataset (Weston et al., 2016) is composed of 20 different tasks (Appendix A), each of which has 1,000 (1k) synthetically-generated story-question

pair. A story can be as short as two sentences and as long as  $200+$  sentences. A system is evaluated on the accuracy of getting the correct answers to the questions. The answers are single words or lists (e.g. "football, apple"). Answering questions in each task requires selecting a set of relevant sentences and applying different kinds of logical reasoning over them. The dataset also includes 10k training data (for each task), which allows training more complex models. Note that DMN+ (Xiong et al., 2016) only reports on the 10k dataset.

bAbI dialog dataset bAbI dialog dataset (Bordes and Weston, 2016) consists of 5 different tasks (Table 3), each of which has 1k synthetically-generated goal-oriented dialogs between a user and the system in the domain of restaurant reservation. Each dialog is as long as 96 utterances and comes with external knowledge base (KB) providing information of each restaurant. The authors also provide Out-Of-Vocabulary (OOV) version of the dataset, where many of the words and KB keywords in test data are not seen during training. A system is evaluated on the accuracy of its response to each utterance of the user, choosing from up to 2500 possible candidate responses. A system is required not only to understand the user's request but also refer to previous conversations in order to obtain the context information of the current conversation.

DSTC2 (Task 6) dialog dataset Bordes and Weston (2016) transformed the Second Dialog State Tracking Challenge (DSTC2) dataset (Henderson et al., 2014) into the same format as the bAbI dialog dataset, for the measurement of performance on a real dataset. Each dialog can be as long as  $800+$  utterances, and a system needs to choose from 2407 possible candidate responses for each utterance of the user. Note that the evaluation metric of the original DSTC2 is different from that of the transformed DSTC2, so previous work on the original DSTC2 should not be directly compared to our work. We will refer to this transformed DSTC2 dataset by "Task 6" of dialog dataset.

# 5.2 MODEL DETAILS

Input Module. In the input module, we are given sentences (previous conversations in dialog)  $\mathbf{x}_t$  and a question (most recent user utterance)  $\mathbf{q}$ , and we want to obtain their vector representations,  $\mathbf{x}_t$ ,  $\mathbf{q} \in \mathbb{R}^d$ . We use a trainable embedding matrix  $\mathbf{A} \in \mathbb{R}^{d \times V}$  to encode the one-hot vector of each word  $\mathbf{x}_{tj}$  in each sentence  $\mathbf{x}_t$  into a  $d$ -dimensional vector  $\mathbf{x}_{tj} \in \mathbb{R}^d$ . Then the sentence representation  $\mathbf{x}_t$  is obtained by Position Encoder (Weston et al., 2015). The same encoder with the same embedding matrix is also used to obtain the question vector  $\mathbf{q}$  from  $\mathbf{q}$ .

Output Module for story-based QA. In the output module, we are given the vector representation of the predicted answer  $\hat{\mathbf{y}}$  and we want to obtain the natural language form of the answer,  $\hat{\mathbf{y}}$ . We use a  $V$ -way single-layer softmax classifier to map  $\hat{\mathbf{y}}$  to a  $V$ -dimensional sparse vector,  $\hat{\mathbf{v}} = \text{softmax}(\mathbf{W}^{(y)}\hat{\mathbf{y}}) \in \mathbb{R}^V$ , where  $\mathbf{W}^{(y)} \in \mathbb{R}^{V \times d}$  is a weight matrix. Then the final answer  $\hat{\mathbf{y}}$  is simply the argmax word in  $\hat{\mathbf{v}}$ . To handle questions with multiple-word answers, we consider each of them as a single word that contains punctuation such as space and comma, and put it in the vocabulary.

Output Module for dialog. We use a fixed number single-layer softmax classifiers, each of which is similar to that of the sotry-based QA model, to sequentially output each word of the system's response. While it is similar in spirit to the RNN decoder (Cho et al., 2014), our output module does not have a recurrent hidden state or gating mechanism. Instead, it solely uses the final output of the QRN,  $\hat{\mathbf{y}}$ , and the current word output to influence the prediction of the next word among possible candidates.

Training. We withhold  $10\%$  of the training for development. We use the hidden state size of 50 by deaf. Batch sizes of 32 for bAbI story-based QA 1k, bAbI dialog and DSTC2 dialog, and 128 for bAbI QA 10k are used. The weights in the input and output modules are initialized with zero mean and the standard deviation of  $1 / \sqrt{d}$ . Weights in the QRN unit are initialized using techniques by Glorot and Bengio (2010), and are tied across the layers. Forget bias of 2.5 is used for update gates (no bias for reset gates). L2 weight decay of 0.0005 (0.001 for dialog) is used for all weights. The loss function is the cross entropy between  $\hat{\mathbf{v}}$  and the one-hot vector of the true answer. The loss is minimized by stochastic gradient descent for maximally 500 epochs, but training is early stopped if the loss on the development data does not decrease for 50 epochs. The learning rate is controlled by AdaGrad (Duchi et al., 2011) with the initial learning rate of 0.1 (0.5 for dialog). Since the model is sensitive to the weight initialization, we repeat each training procedure 10 times (50 times for 10k) with the new random initialization of the weights and report the result on the test data with the lowest loss on the development data.

<table><tr><td rowspan="3">Task</td><td colspan="6">1k</td><td colspan="6">10k</td></tr><tr><td colspan="4">Previous works</td><td colspan="2">QRN</td><td colspan="5">Previous works</td><td>QRN</td></tr><tr><td>LSTM</td><td>N2N</td><td>DMN+†</td><td>GMemN2N</td><td>2r</td><td>3r</td><td>N2N</td><td>DMN+</td><td>GMemN2N</td><td>DNC</td><td>6r200</td><td></td></tr><tr><td># Failed</td><td>20</td><td>10</td><td>16</td><td>10</td><td>7</td><td>5</td><td>3</td><td>1</td><td>3</td><td>2</td><td>1</td><td></td></tr><tr><td>Average error rates</td><td>51.3</td><td>15.2</td><td>33.2</td><td>12.7</td><td>9.9</td><td>11.3</td><td>4.2</td><td>2.8</td><td>3.7</td><td>3.8</td><td>0.67</td><td></td></tr><tr><td rowspan="3" colspan="4">Task</td><td colspan="4">Plain</td><td colspan="5">With Match</td></tr><tr><td colspan="2">Previous works</td><td colspan="2">QRN</td><td colspan="3">Previous works</td><td colspan="2">QRN</td></tr><tr><td>N2N</td><td>GMemN2N</td><td>2r</td><td>2r100</td><td>N2N+</td><td colspan="2">GMemN2N+</td><td>2r+</td><td></td></tr><tr><td colspan="4">bAbI dialog Average error rates</td><td>13.9</td><td>14.3</td><td>5.5</td><td>5.5</td><td>6.7</td><td colspan="2">5.4</td><td>1.5</td><td></td></tr><tr><td colspan="4">bAbI dialog (OOV) Average error rates</td><td>30.3</td><td>27.9</td><td>11.1</td><td>11.1</td><td>11.2</td><td colspan="2">10.3</td><td>2.3</td><td></td></tr><tr><td colspan="4">DSTC2 dialog Average error rates</td><td>58.9</td><td>52.6</td><td>49.5</td><td>48.9</td><td>59.0</td><td colspan="2">51.3</td><td>49.3</td><td></td></tr></table>

Table 1: (top) bAbI QA dataset (Weston et al., 2016): number of failed tasks and average error rates  $(\%)$ .  $\dagger$  is obtained from github.com/therne/dmn-tensorflow. (bottom) bAbI dialog and DSTC2 dialog dataset (Bordes and Weston, 2016) average error rates  $(\%)$  of QRN and previous work (LSTM, N2N, DMN+, GMemN2N, and DNC). For QRN, the first number (1, 2, 3) indicates the number of layers, 'r' means the reset gate is used, and the last number (100, 200), if exists, indicates the dimension of the hidden state, where the default value is 50. ' $+$ ' indicates that 'match' (See Appendix for details) is used. The task-wise results are shown in Appendices: Table 2 (bAbI QA) and Table 3 (dialog datasets). See Section 5.3 for details.

# 5.3 RESULTS.

We compare our model with baselines and previous state-of-the-art models on story-based and dialog tasks (Table 1). These include LSTM (Hochreiter and Schmidhuber, 1997), End-to-end Memory Networks (N2N) (Sukhbaatar et al., 2015), Dynamic Memory Networks (DMN+) (Xiong et al., 2016), Gated End-to-end Memory Networks (GMemN2N) (Perez and Liu, 2016), and Differentiable Neural Computer (DNC) (Graves et al., 2016).

Story-based QA. Table 1(top) reports the summary of results of our model (QRN) and previous work on bAbI QA (task-wise results are shown in Table 2 in Appendix). In 1k data, QRN's '2r' (2 layers + reset gate +  $d = 50$ ) outperforms all other models by a large margin  $(2.8 + \%)$ . In 10k dataset, the average accuracy of QRN's '6r200' (2 layers + reset gate +  $d = 200$ ) model outperforms all previous models by a large margin  $(2.1 + \%)$ , achieving a nearly perfect score of  $99.3\%$ .

Dialog. Table 1(bottom) reports the summary of the results of our model (QRN) and previous work on bAbI dialog and Task 6 dialog (task-wise results are shown in Table 3 in Appendix). As done in previous work (Bordes and Weston, 2016; Perez and Liu, 2016), we also report results when we use 'Match' for dialogs. 'Match' is the extension to the model which additionally takes as input whether each answer candidate matches with context (more details on Appendix). QRN outperforms previous work by a large margin  $(2.0 + \%)$  in every configuration (plain vs OOV, plain vs match, synthetic (Task 1-5) vs real (Task 6)).

Ablations. We test four types of ablations (also discussed in Section 2.2): number of layers (1, 2, 3, or 6), reset gate (r), and gate vectorization (v) and the dimension of the hidden vector (50, 100). We show a subset of combinations of the ablations for bAbI QA in Table 1 and Table 2; other combinations performed poorly and/or did not give interesting observations. According to the ablation results, we infer that: (a) When the number of layers is only one, the model lacks reasoning capability. In the case of 1k dataset, when there are too many layers (6), it seems correctly training the model becomes increasingly difficult. In the case of 10k dataset, many layers (6) and hidden dimensions (200) helps reasoning, most notably in difficult task such as task 16. (b) Adding the reset gate helps. (c) Including vector gates hurts in 1k datasets, as the model either overfits to the training data or converges to local minima. On the other hand, vector gates in bAbI story-based QA 10k dataset sometimes help. (d) Increasing the dimension of the hidden state to 100 in the dialog's Task 6 (DSTC2) helps, while there is not much improvement in the dialog's Task 1-5. It can be hypothesized that a larger hidden state is required for real data.

Parallelization. We implement QRN with and without parallelization in TensorFlow (Abadi et al., 2016) on a single Titan X GPU to qunitify the computational gain of the parallelization. For QRN without parallelization, we use the RNN library provided by TensorFlow. QRN with parallelization gives 6.2 times faster training and inference than QRN without parallelization on average. We expect that the speedup can be even higher for datasets with larger context.

Interpretations. An advantage of QRN is that the intermediate query updates are interpretable. Figure 1 shows intermediate local queries  $(\mathbf{q}_t^k)$  interpreted in natural language, such as "Where is

<table><tr><td colspan="2"></td><td colspan="3">Layer 1</td><td colspan="3">Layer 2</td><td colspan="3">Layer 1</td><td colspan="3">Layer 2</td></tr><tr><td colspan="2">Task 2: Two Supporting Facts</td><td>z1</td><td>r1</td><td>r-1</td><td>z2</td><td>Task 15: Deduction</td><td>z1</td><td>r1</td><td>r-1</td><td>z2</td><td></td><td></td><td></td></tr><tr><td colspan="2">Sandra picked up the apple there.</td><td>0.95</td><td>0.89</td><td>0.98</td><td>0.00</td><td>Mice are afraid of wolves.</td><td>0.11</td><td>0.99</td><td>0.13</td><td>0.78</td><td></td><td></td><td></td></tr><tr><td colspan="2">Sandra dropped the apple.</td><td>0.83</td><td>0.05</td><td>0.92</td><td>0.01</td><td>Gertrude is a mouse.</td><td>0.77</td><td>0.99</td><td>0.96</td><td>0.00</td><td></td><td></td><td></td></tr><tr><td colspan="2">Daniel grabbed the apple there.</td><td>0.88</td><td>0.93</td><td>0.98</td><td>0.00</td><td>Cats are afraid of sheep.</td><td>0.01</td><td>0.99</td><td>0.07</td><td>0.03</td><td></td><td></td><td></td></tr><tr><td colspan="2">Sandra travelled to the bathroom.</td><td>0.01</td><td>0.18</td><td>0.63</td><td>0.02</td><td>Winona is a mouse.</td><td>0.14</td><td>0.85</td><td>0.77</td><td>0.05</td><td></td><td></td><td></td></tr><tr><td colspan="2">Daniel went to the hallway.</td><td>0.01</td><td>0.24</td><td>0.62</td><td>0.83</td><td>Sheep are afraid of wolves.</td><td>0.02</td><td>0.98</td><td>0.27</td><td>0.05</td><td></td><td></td><td></td></tr><tr><td colspan="6">Where is the apple? hallway</td><td colspan="8">What is Gertrude afraid of? wolf</td></tr><tr><td>Task 3</td><td colspan="3">Layer 1</td><td>Layer 2</td><td colspan="3"></td><td colspan="3">Layer 1</td><td colspan="3">Layer 2</td></tr><tr><td>Displaying options</td><td>z1</td><td>r1</td><td>r-1</td><td>z2</td><td colspan="3">Task 6 DSTC2 dialog</td><td>z1</td><td>r1</td><td>r-1</td><td>z2</td><td></td><td></td></tr><tr><td>resto-paris-expen-frech-8stars?</td><td>0.00</td><td>1.00</td><td>0.96</td><td>0.91</td><td colspan="3">Spanish food.</td><td>0.84</td><td>0.07</td><td>0.00</td><td>0.82</td><td></td><td></td></tr><tr><td>Do you have something else?</td><td>0.41</td><td>0.99</td><td>0.00</td><td>0.00</td><td colspan="3">You are looking for a spanish restaurant right?</td><td>0.98</td><td>0.02</td><td>0.49</td><td>0.75</td><td></td><td></td></tr><tr><td>Sure let me find another option.</td><td>1.00</td><td>0.00</td><td>0.00</td><td>0.12</td><td colspan="3">Yes.</td><td>0.01</td><td>1.00</td><td>0.33</td><td>0.13</td><td></td><td></td></tr><tr><td>resto-paris-expen-frech-5stars?</td><td>0.00</td><td>1.00</td><td>0.96</td><td>0.91</td><td colspan="3">What part of town do you have in mind?</td><td>0.20</td><td>0.73</td><td>0.41</td><td>0.11</td><td></td><td></td></tr><tr><td>No this does not work for me.</td><td>0.00</td><td>0.00</td><td>0.14</td><td>0.00</td><td colspan="3">I don&#x27;t care.</td><td>0.00</td><td>1.00</td><td>0.02</td><td>0.00</td><td></td><td></td></tr><tr><td>Sure let me find an other option.</td><td>1.00</td><td>0.00</td><td>0.00</td><td>0.12</td><td colspan="3">What price range would you like?</td><td>0.72</td><td>0.46</td><td>0.52</td><td>0.72</td><td></td><td></td></tr><tr><td colspan="5">What do you think of this? resto-paris-expen-french-4stars</td><td colspan="9">I don&#x27;t care. API CALL spanish R-location R-price</td></tr></table>

Figure 3: (top) bAbI QA dataset (Weston et al., 2016) visualization of update and reset gates in QRN '2r' model (bottom two) bAbI dialog and DSTC2 dialog dataset (Bordes and Weston, 2016) visualization of update and reset gates in QRN '2r' model. Note that the stories can have as many as  $800+$  sentences; we only show part of them here. More visualizations are shown in Figure 4 (bAbI QA) and Figure 5 (dialog datasets).

Sandra?". In order to obtain these, we place a decoder on the input question embedding  $\mathbf{q}$  and add its loss for recovering the question to the classification loss (similarly to Peng et al. (2015)). We then use the same decoder to decode the intermediate queries. This helps us understand the flow of information in the networks. In Figure 1, the question Where is apple? is transformed into Where is Sandra? at  $t = 1$ . At  $t = 2$ , as Sandra dropped the apple, the apple is no more relevant to Sandra. We obtain Where is Daniel? at time  $t = 3$ , and it is propagated until  $t = 5$ , where we observe a sentence (fact) that can be used to answer the query.

Visualization. Figure 3 shows visualization of the (scalar) magnitudes of update and reset gates on story sentences and dialog utterances. More visualizations are shown in Appendices: Figure 4 and Figure 5. In Figure 3, we observe high values on facts that provide information to answer question (the system's next utterance for dialog). In QA Task 2 example (top left), we observe high update gate values in the first layer on facts that state who has the apple, and in the second layer, the high update gate values are on those that inform where that person went to. We also observe that the forward reset gate at  $t = 2$  in the first layer  $(\vec{r}_2^1)$  is low, which is signifying that apple no more belongs to Sandra. In dialog Task 3 (bottom left), the model is able to infer that three restaurants are already recommended so that it can recommend another one. In dialog Task 6 (bottom), the model focuses on the sentences containing Spanish, and does not concentrate much on other facts such as I don't care.

# 6 CONCLUSION

In this paper, we introduce Query-Reduction Network (QRN) to answer context-based questions and carry out conversations with users that require multi-hop reasoning. We show the state-of-the-art results in the three datasets of story-based QA and dialog. We model a story or a dialog as a sequence of state-changing triggers and compute the final answer to the question or the system's next utterance by recurrently updating (or reducing) the query. QRN is situated between the attention mechanism and RNN, effectively handling time dependency and long-term dependency problems of each technique, respectively. It addresses the long-term dependency problem of most RNNs by simplifying the recurrent update, in which the candidate hidden state (reduced query) does not depend on the previous state. Moreover, QRN can be parallelized and can address the well-known problem of RNN's vanishing gradients.

# REFERENCES

Martin Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467, 2016.  
Antoine Bordes and Jason Weston. Learning end-to-end goal-oriented dialog. arXiv preprint arXiv:1605.07683, 2016.  
Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnN encoder-decoder for statistical machine translation. In EMNLP, 2014.  
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. JMLR, 12, 2011.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In JMLR, 2010.  
Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, et al. Hybrid computing using a neural network with dynamic external memory. Nature, 2016.  
Matthew Henderson, Blaise Thomson, and Jason Williams. The second dialog state tracking challenge. In 15th Annual Meeting of the Special Interest Group on Discourse and Dialogue, volume 263, 2014.  
Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. In NIPS, 2015.  
Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston. The goldilocks principle: Reading children's books with explicit memory representations. In ICLR, 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Ankit Kumar, Ozan Irsoy, Jonathan Su, James Bradbury, Robert English, Brian Pierce, Peter Ondruska, Ishaan Gulrajani, and Richard Socher. Ask me anything: Dynamic memory networks for natural language processing. In ICML, 2016.  
Baolin Peng, Zhengdong Lu, Hang Li, and Kam-Fai Wong. Towards neural network-based reasoning. arXiv preprint arXiv:1508.05508, 2015.  
Julien Perez and Fei Liu. Gated end-to-end memory networks. arXiv preprint arXiv:1610.04211, 2016.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. In EMNLP, 2016.  
Raymond Reiter. Knowledge in Action. MIT Press, 1st edition, 2001.  
Matthew Richardson, Christopher JC Burges, and Erin Renshaw. MCTest: A challenge dataset for the open-domain machine comprehension of text. In EMNLP, 2013.  
Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In NIPS, 2015.  
Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. In ICLR, 2015.  
Jason Weston, Antoine Bordes, Sumit Chopra, and Tomas Mikolov. Towards ai-complete question answering: A set of prerequisite toy tasks. In ICLR, 2016.  
Caiming Xiong, Stephen Merity, and Richard Socher. Dynamic memory networks for visual and textual question answering. In ICML, 2016.
