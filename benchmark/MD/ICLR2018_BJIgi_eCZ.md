# FUSIONNET: FUSING VIA FULLY-AWARE ATTENTION WITH APPLICATION TO MACHINE COMPREHENSION

Anonymous authors

Paper under double-blind review

# ABSTRACT

This paper introduces a new neural structure called FusionNet, which extends existing attention approaches from three perspectives. First, it puts forward a novel concept of "history of word" to characterize attention information from the lowest word-level embedding up to the highest semantic-level representation. Second, it introduces an improved attention scoring function that better utilizes the "history of word" concept. Third, it proposes a fully-aware multi-level attention mechanism to capture the complete information in one text (such as a question) and exploit it in its counterpart (such as context or passage) layer by layer. We apply FusionNet to the Stanford Question Answering Dataset (SQuAD) and it achieves the first position for both single and ensemble model on the official SQuAD leaderboard at the time of writing (Oct. 4th, 2017). Meanwhile, we verify the generalization of FusionNet with two adversarial SQuAD datasets and it sets up the new state-of-the-art on both datasets: on AddSent, FusionNet increases the best F1 metric from  $46.6\%$  to  $51.4\%$ ; on AddOneSent, FusionNet boosts the best F1 metric from  $56.0\%$  to  $60.7\%$ .

# 1 INTRODUCTION

Teaching machines to read, process and comprehend text and then answer questions is one of key problems in artificial intelligence. Figure 1 gives an example of the machine reading comprehension task. It feeds a machine with a piece of context and a question, and teaches it to find a correct answer to the question. This requires the machine to possess high capabilities in comprehension, inference and reasoning. This is considered as a challenging task in artificial intelligence and has already attracted numerous research efforts from the neural network and natural language processing communities. Many neural network models have been proposed for this challenge and they generally frame this problem as a machine reading comprehension (MRC) task (Hochreiter & Schmidhuber, 1997; Wang et al., 2017; Seo et al., 2017; Shen et al., 2017; Xiong et al., 2017; Weissenborn et al., 2017; Chen et al., 2017).

Context: The Alpine Rhine is part of the Rhine, a famous European river. The Alpine Rhine begins in the most western part of the Swiss canton of Graubünden, and later forms the border between Switzerland to the West and Liechtenstein and later Austria to the East. On the other hand, the Danube separates Romania and Bulgaria.

Question: What is the other country the Rhine separates Switzerland to?

Answer: Liechtenstein

Figure 1: Question-answer pair for a passage discussing Alpine Rhine.

The key innovation in recent models lies in how to ingest information in the question and characterize it in the context, in order to provide an accurate answer to the question. This is often modeled as attention in the neural network community, which is a mechanism to attend the question into the context so as to find the answer related to the question. Some (Chen et al., 2017; Weissenborn et al., 2017) attend the word-level embedding from the question to context, while some (Wang et al., 2017) attend the high level representation in the question to augment the context. However we observed that none of the existing approaches has captured full information in the context or the question, which could be vital for complete information digestion. Take image recognition as an example, information in various levels of representations can capture different aspects of details in an image: pixel, stroke and shape. We argue that this hypothesis also holds in language understanding and

MRC. In other words, an approach that utilize all the information from word embedding level up to the highest level representation would be substantially beneficial for understanding both the question and the context, hence yielding more accurate answers.

However, the ability to consider all layers of representation is often limited by the difficulty to make the neural model learn well, as model complexity will surge beyond capacity. We conjectured this is why previous literature tailored their models to only consider partial information. To alleviate this challenge, we propose an improved attention scoring function utilizing all layers of representation with less training burden. This leads to an attention that thoroughly captures the complete information between the question and the context. With this fully-aware attention, we put forward a multi-level attention mechanism to understand the information in question, and exploit it layer by layer on the context side. All of these innovations are integrated into a new end-to-end structure called FusionNet in Figure 4, with details described in Section 3.

We submitted FusionNet to SQuAD (Rajpurkar et al., 2016), a machine reading comprehension dataset. At the time of writing (Oct. 4th, 2017), our model ranked in the first place in both single model and ensemble model categories. The ensemble model achieves an exact match (EM) score of  $78.8\%$  and F1 score of  $85.9\%$ . Furthermore, we have tested FusionNet against adversarial SQuAD datasets (Jia & Liang, 2017). Results show that FusionNet outperforms existing state-of-the-art architectures in both datasets: on AddSent, FusionNet increases the best F1 metric from  $46.6\%$  to  $51.4\%$ ; on AddOneSent, FusionNet boosts the best F1 metric from  $56.0\%$  to  $60.7\%$ . This demonstrated the exceptional performance of FusionNet.

# 2 MACHINE COMPREHENSION & FULLY-AWARE ATTENTION

In this section, we briefly introduce the task of machine comprehension, and a conceptual architecture that summarizes recent advances in machine reading comprehension. Then, we introduce a novel concept called History of Word. History of Word characterizes the importance of capturing all levels of information to fully understand the text. Finally, a light-weight implementation for History of Word, Fully-aware Attention, is proposed.

# 2.1 TASK DESCRIPTION

In machine comprehension, given a context and a question, the machine needs to read and understand the context, then find the answer to the question. The context is described as a sequence of word tokens:  $\mathbf{C} = \{w_1^C,\dots ,w_m^C\}$ , and the question as:  $\mathbf{Q} = \{w_{1}^{Q},\ldots ,w_{n}^{Q}\}$ , where  $m$  is the number of words in the context, and  $n$  is the number of words in the question. In general,  $m\gg n$ . The answer  $\mathbf{A}\mathbf{n}\mathbf{s}$  can have different forms depending on the task. In the SQuAD dataset (Rajpurkar et al., 2016), the answer  $\mathbf{A}\mathbf{n}\mathbf{s}$  is guaranteed to be a contiguous span in the context  $C$ , e.g.,  $\mathbf{A}\mathbf{n}\mathbf{s} = \{w_i^C,\dots ,w_{i + k}^C\}$ .

# 2.2 CONCEPTUAL ARCHITECTURE FOR MACHINE READING COMPREHENSION

In all state-of-the-art architectures for machine reading comprehension, a recurring pattern is the following process. Given two sets of vectors, A and B, we enhance or modify every single vector in set A with the information from set B. We call this a fusion process, where set B is fused into set A. Fusion process is commonly based on attention (Bahdanau et al., 2015), but some are not. Major improvements in recent MRC work lie in how they design the fusion process.

A conceptual architecture illustrating state-of-the-art architectures is shown in Figure 2, which consists of three components.

- Input vectors: Embedding vectors for each word in the context and the question.  
- Integration components: The rectangular box. It is usually implemented using RNN such as LSTM (Hochreiter & Schmidhuber, 1997) or GRU (Chung et al., 2014).  
- Fusion processes: The numbered arrow (1), (2), (2'), (3), (3'). The set pointing outward is fused into the set being pointed to.

<table><tr><td>Architectures</td><td>(1)</td><td>(2)</td><td>(2&#x27;)</td><td>(3)</td><td>(3&#x27;)</td></tr><tr><td>Match-LSTM (Wang &amp; Jiang, 2016)</td><td>✓</td><td></td><td></td><td></td><td rowspan="2">(3)</td></tr><tr><td>DCN (Xiong et al., 2017)</td><td>✓</td><td></td><td></td><td>✓</td></tr><tr><td>FastQA (Weissenborn et al., 2017)</td><td>✓</td><td></td><td></td><td></td><td rowspan="7">(2)</td></tr><tr><td>FastQAExt (Weissenborn et al., 2017)</td><td>✓</td><td>✓</td><td></td><td>✓</td></tr><tr><td>BiDAF (Seo et al., 2017)</td><td>✓</td><td></td><td></td><td>✓</td></tr><tr><td>RaSoR (Lee et al., 2016)</td><td>✓</td><td></td><td>✓</td><td></td></tr><tr><td>DrQA (Chen et al., 2017)</td><td>✓</td><td></td><td></td><td></td></tr><tr><td>MPCM (Wang et al., 2016)</td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>Mnemonic Reader (Hu et al., 2017)</td><td>✓</td><td>✓</td><td></td><td>✓</td></tr><tr><td>R-net (Wang et al., 2017)</td><td>✓</td><td></td><td>✓</td><td></td><td>Context Question</td></tr></table>

Table 1: A summarized view on the fusion processes used in several state-of-the-art architectures.

Figure 2: A conceptual architecture illustrating recent advances in MRC.

There are three main types of fusion processes in recent advanced architectures. Table 1 shows what fusion processes are used in different state-of-the-art architectures. We now discuss them in detail.

(1) Word-level fusion. By providing the direct word information in question to the context, we can quickly zoom in to more related regions in the context. However, it may not be helpful if a word has different semantic meaning based on the context. Many word-level fusions are not based on attention, e.g., (Hu et al., 2017; Chen et al., 2017) appends binary features to context words, indicating whether each context word appears in the question.  
(2) High-level fusion. By informing the context about the semantic information in the question, it could help to find the correct answer. But high-level information is more imprecise than word information, which may cause models to be less aware of details.  
(2') High-level fusion (Alternative). Similarly, we could also fuse high-level concept of  $Q$  into the word-level of  $C$ .  
(3) Self-boosted fusion. Since the context can be long, and distant parts of text may rely on each other to fully understand the content, recent advances proposed to fuse the context into itself. As context contains excessive information, one common choice is to perform self-boosted fusion after fusing the question  $Q$ . This allows us to be more aware of the regions related to the question.  
(3') Self-boosted fusion (Alternative). Another choice is to directly condition the self-boosted fusion process on the question  $Q$ , such as the coattention mechanism proposed in (Xiong et al., 2017). Then we can perform self-boosted fusion before fusing question information.

A common trait of existing fusion mechanisms is that none of them employs all levels of representation jointly. In the follows, we claim that employing all levels of representation is crucial to achieving better text understanding.

# 2.3 FULLY-AWARE ATTENTION ON HISTORY OF WORD

Consider the illustration shown in Figure 3. As we read through the context, each input word will gradually transform into a more abstract representation, e.g., becoming low-level and then high-level concepts. Altogether, they form the history of each word in our mental flow. For a human, we utilize the history of word so frequently but we often neglect its importance. For example, to answer the question in Figure 3 correctly, we need to focus on both the high-level concept of forms the border and the word-level information of Alpine Rhine. If we focus only on the high-level concepts, we will confuse Alpine Rhine with Danube since both are European rivers that separates countries. Therefore we hypothesize that the entire history-of-word is important to fully understand the text.

In neural architectures, we define the history of the  $i$ -th word,  $\mathrm{HoW}_i$ , to be the concatenation of all the representations generated for this word. This may include word embedding, multiple inter

![](images/ac1283dcff85f74db22649040214675d52387ea8f13fd9574597920d2c7e2ff9.jpg)  
Figure 3: Illustrations of the history-of-word for the example shown in Figure 1. Utilizing the entire history-of-word is crucial for the full understanding of the context.

mediate and output hidden vectors in RNN, and corresponding representation vectors in any further layers. To incorporate history-of-word into a wide range of neural models, we present a light-weight implementation, Fully-aware Attention.

Attention can be applied in different scenarios. To be more conclusive, we focus on attention applied to fusing information from one body to another. Consider two sets of hidden vectors for words in text bodies A and B:  $\{h_1^A,\dots ,h_m^A\}$ $\{h_1^B,\dots ,h_n^B\} \subset \mathbb{R}^d$  . Their associated history-of-word are,

$$
\left\{\mathrm {H o W} _ {1} ^ {A}, \dots , \mathrm {H o W} _ {m} ^ {A} \right\}, \left\{\mathrm {H o W} _ {1} ^ {B}, \dots , \mathrm {H o W} _ {n} ^ {B} \right\} \subset \mathbb {R} ^ {d _ {h}},
$$

where  $d_h \gg d$ . Fusing body B to body A via standard attention means for every  $h_i^A$  in body A,

1. Compute an attention score  $S_{ij} = S(\pmb{h}_i^A, \pmb{h}_j^B) \in \mathbb{R}$  for each  $\pmb{h}_j^B$  in body B.  
2. Form the attention weight  $\alpha_{ij}$  through softmax:  $\alpha_{ij} = \exp (S_{ij}) / \sum_k\exp (S_{ik})$  
3. Concatenate  $\pmb{h}_i^A$  with the summarized information,  $\hat{\pmb{h}}_i^A = \sum_j \alpha_{ij} \pmb{h}_j^B$ .

In fully-aware attention, we replace attention score computation with the history-of-word.

$$
S \left(\boldsymbol {h} _ {i} ^ {A}, \boldsymbol {h} _ {j} ^ {B}\right) \Longrightarrow S \left(\operatorname {H o W} _ {i} ^ {A}, \operatorname {H o W} _ {j} ^ {B}\right).
$$

This allows us to be fully aware of the complete understanding of each word. Ablation study in Section 4.4 demonstrates that this light-weight enhancement offers a decent improvement in performance.

To fully utilize history-of-word in attention, we need a suitable attention scoring function  $S(\pmb{x}, \pmb{y})$ . A commonly used function is multiplicative attention (Britz et al., 2017):  $\pmb{x}^T U^T V \pmb{y}$ , leading to

$$
S _ {i j} = (\mathrm {H o W} _ {i} ^ {A}) ^ {T} U ^ {T} V (\mathrm {H o W} _ {j} ^ {B}),
$$

where  $U, V \in \mathbb{R}^{k \times d_h}$ , and  $k$  is the attention hidden size. However, we suspect that two large matrices interacting directly will make the neural model harder to train. Therefore we propose to constrain the matrix  $U^T V$  to be symmetric. A symmetric matrix can always be decomposed into  $U^T D U$ , thus

$$
S _ {i j} = (\mathrm {H o W} _ {i} ^ {A}) ^ {T} U ^ {T} D U (\mathrm {H o W} _ {j} ^ {B}),
$$

where  $U \in \mathbb{R}^{k \times d_h}$ ,  $D \in \mathbb{R}^{k \times k}$  and  $D$  is a diagonal matrix. Symmetric form retains the ability to give high attention score between dissimilar  $\mathrm{HoW}_i^A$ ,  $\mathrm{HoW}_j^B$ . Additionally, we marry nonlinearity with the symmetric form to provide richer interaction among different parts of the history-of-word. The final formulation for attention score is

$$
S _ {i j} = f (U (\mathrm {H o W} _ {i} ^ {A})) ^ {T} D f (U (\mathrm {H o W} _ {j} ^ {B})),
$$

where  $f(x)$  is an activation function applied element-wise. In the following context, we employ  $f(x) = \max(0, x)$ . A detailed ablation study in Section 4 demonstrates its advantage over many alternatives.

![](images/5e403eb7586ec57ac20e967f34ebd8997850ee378b265d1f0fdd677223ab6d06.jpg)  
Figure 4: An illustration of FusionNet architecture. Each upward arrow represents one layer of BiLSTM. Each circle to the right is a detailed illustration of the corresponding component in FusionNet.  
Circle 1: Fully-aware attention between  $C$  and  $Q$  to obtain question information in different levels.  
Circle 2: Concatenate all concepts in  $C$  with multi-level  $Q$  information, then pass through BiLSTM.  
Circle 3: Fully-aware attention on the context  $C$  itself to obtain related distant information.  
Circle 4: Concatenate the understanding vector of  $C$  with self-attention information, then pass through BiLSTM.

# 3 FULLY-AWARE FUSION NETWORK

# 3.1 END-TO-END ARCHITECTURE

Based on fully-aware attention, we propose an end-to-end architecture, fully-aware fusion network (FusionNet). Given text A and B, FusionNet fuses information from text B to text A and generates two set of vectors

$$
U _ {A} = \left\{\boldsymbol {u} _ {1} ^ {A}, \dots , \boldsymbol {u} _ {m} ^ {A} \right\}, \quad U _ {B} = \left\{\boldsymbol {u} _ {1} ^ {B}, \dots , \boldsymbol {u} _ {n} ^ {B} \right\}.
$$

In the following, we consider the special case where text A is context  $C$  and text B is question  $Q$ . An illustration for FusionNet is shown in Figure 4. It consists of the following components.

Input Vectors. First, each word in  $C$  and  $Q$  is transformed into an input vector  $\mathbf{w}$ . We utilize the 300-dim GloVe embedding (Pennington et al., 2014) and 600-dim contextualized vector (McCann et al., 2017). In the SQuAD task, we also include 12-dim POS embedding, 8-dim NER embedding and a normalized term frequency for context  $C$  as suggested in (Chen et al., 2017). Together  $\{\mathbf{w}_1^C, \dots, \mathbf{w}_m^C\} \subset \mathbb{R}^{900 + 20 + 1}$ , and  $\{\mathbf{w}_1^Q, \dots, \mathbf{w}_n^Q\} \subset \mathbb{R}^{900}$ .

Fully-aware Multi-level Fusion: Word-level. In multi-level fusion, we separately consider fusing word-level and higher-level. Word-level fusion inform  $C$  about what kind of words are in  $Q$ . It is illustrated as arrow (1) in Figure 2. For this component, we follow the approach in (Chen et al., 2017) First, a feature vector  $\mathrm{em}_i$  is created for each word in  $C$  to indicate whether the word occurs in the question  $Q$ . Second, attention-based fusion on GloVe embedding  $g_i$  is used

$$
\hat {\pmb {g}} _ {i} ^ {C} = \sum_ {j} \alpha_ {i j} \pmb {g} _ {j} ^ {Q}, \quad \alpha_ {i j} \propto \exp (S (\pmb {g} _ {i} ^ {C}, \pmb {g} _ {j} ^ {Q})), \quad S (\pmb {x}, \pmb {y}) = \mathrm {R e L U} (W \pmb {x}) ^ {T} \mathrm {R e L U} (W \pmb {y}),
$$

where  $W \in \mathbb{R}^{300 \times 300}$ . Fully-aware attention is not employed. The attention here merely takes back similar embedding. The enhanced input vector for context is now  $\tilde{\boldsymbol{w}}_i^C = [\boldsymbol{w}_i^C; \mathrm{em}_i; \hat{\boldsymbol{g}}_i^C]$ .

Reading. In the reading component, we use separate bidirectional LSTM (BiLSTM) to form low-level and high-level concepts for  $C$  and  $Q$ .

$$
\boldsymbol {h} _ {1} ^ {C l}, \ldots , \boldsymbol {h} _ {m} ^ {C l} = \operatorname {B i L S T M} (\tilde {\boldsymbol {w}} _ {1} ^ {C}, \ldots , \tilde {\boldsymbol {w}} _ {m} ^ {C}), \quad \boldsymbol {h} _ {1} ^ {Q l}, \ldots , \boldsymbol {h} _ {n} ^ {Q l} = \operatorname {B i L S T M} (\boldsymbol {w} _ {1} ^ {Q}, \ldots , \boldsymbol {w} _ {n} ^ {Q}),
$$

$$
\boldsymbol {h} _ {1} ^ {C h}, \ldots , \boldsymbol {h} _ {m} ^ {C h} = \operatorname {B i L S T M} (\boldsymbol {h} _ {1} ^ {C l}, \ldots , \boldsymbol {h} _ {m} ^ {C l}), \quad \boldsymbol {h} _ {1} ^ {Q h}, \ldots , \boldsymbol {h} _ {n} ^ {Q h} = \operatorname {B i L S T M} (\boldsymbol {h} _ {1} ^ {Q l}, \ldots , \boldsymbol {h} _ {n} ^ {Q l}).
$$

Hence low-level and high-level concept  $\pmb{h}^l, \pmb{h}^h \in \mathbb{R}^{250}$  are created for each word.

Question Understanding. In the Question Understanding component, we apply a new BiLSTM taking in both  $\pmb{h}^{Ql}, \pmb{h}^{Qh}$  to obtain the final question representation  $U_{Q}$ :

$$
U _ {Q} = \left\{\boldsymbol {u} _ {1} ^ {Q}, \dots , \boldsymbol {u} _ {n} ^ {Q} \right\} = \operatorname {B i L S T M} \left(\left[ \boldsymbol {h} _ {1} ^ {Q l}; \boldsymbol {h} _ {1} ^ {Q h} \right], \dots , \left[ \boldsymbol {h} _ {n} ^ {Q l}; \boldsymbol {h} _ {n} ^ {Q h} \right]\right).
$$

where  $\{\pmb{u}_i^Q\in \mathbb{R}^{250}\}_{i = 1}^n$  are the understanding vectors for  $Q$

Fully-aware Multi-level Fusion: Higher-level. This component fuses all higher level information in the question  $Q$  to the context  $C$  through fully-aware attention on history-of-word. Since the proposed attention scoring function for fully-aware attention is constrained to be symmetric, we need to identify the common history-of-word for both  $C, Q$ . This yields

$$
\operatorname {H o W} _ {i} ^ {C} = \left[ \boldsymbol {g} _ {i} ^ {C}; \boldsymbol {c} _ {i} ^ {C}; \boldsymbol {h} _ {i} ^ {C l}; \boldsymbol {h} _ {i} ^ {C h} \right], \operatorname {H o W} _ {i} ^ {Q} = \left[ \boldsymbol {g} _ {i} ^ {Q}; \boldsymbol {c} _ {i} ^ {Q}; \boldsymbol {h} _ {i} ^ {Q l}; \boldsymbol {h} _ {i} ^ {Q h} \right] \in \mathbb {R} ^ {1 4 0 0},
$$

where  $g_{i}$  is the GloVe embedding and  $c_{i}$  is the CoVe embedding. Then we fuse low, high, and understanding-level information from  $Q$  to  $C$  via fully-aware attention. Different sets of attention weights are calculated through attention function  $S^{l}(\boldsymbol{x},\boldsymbol{y}), S^{h}(\boldsymbol{x},\boldsymbol{y}), S^{u}(\boldsymbol{x},\boldsymbol{y})$  to combine low, high, and understanding-level of concept. All three functions are the proposed symmetric form with nonlinearity in Section 2.3, but are parametrized by independent parameters to attend to different regions for different level.

1. Low-level fusion:  $\hat{\pmb{h}}_i^{Cl} = \sum_j\alpha_{ij}^l\pmb {h}_j^{Ql}$ $\alpha_{ij}^{l}\propto \exp (S^{l}(\mathrm{HoW}_{i}^{C},\mathrm{HoW}_{j}^{Q}))$  
2. High-level fusion:  $\hat{\pmb{h}}_i^{Ch} = \sum_j\alpha_{ij}^h\pmb {h}_j^{Qh}$ $\alpha_{ij}^{h}\propto \exp (S^{h}(\mathrm{HoW}_{i}^{C},\mathrm{HoW}_{j}^{Q}))$  
3. Understanding fusion:  $\hat{\pmb{u}}_i^C = \sum_j\alpha_{ij}^u\pmb {u}_j^Q$ $\alpha_{ij}^{u}\propto \exp (S^{u}(\mathrm{HoW}_{i}^{C},\mathrm{HoW}_{j}^{Q}))$

This multi-level attention mechanism captures different levels of information independently, while taking all levels of information into account. A new BiLSTM is applied to obtain the representation for  $C$  fully fused with information in the question  $Q$ :

$$
V _ {C} = \{\boldsymbol {v} _ {1} ^ {C}, \dots , \boldsymbol {v} _ {m} ^ {C} \} = \mathrm {B i L S T M} ([ \boldsymbol {h} _ {1} ^ {C l}; \boldsymbol {h} _ {1} ^ {C h}; \hat {\boldsymbol {h}} _ {1} ^ {C l}; \hat {\boldsymbol {h}} _ {1} ^ {C h}; \hat {\boldsymbol {u}} _ {1} ^ {C} ], \dots , [ \boldsymbol {h} _ {m} ^ {C l}; \boldsymbol {h} _ {m} ^ {C h}; \hat {\boldsymbol {h}} _ {m} ^ {C l}; \hat {\boldsymbol {h}} _ {m} ^ {C h}; \hat {\boldsymbol {u}} _ {m} ^ {C} ]).
$$

Fully-aware Self-boosted Fusion. We now use self-boosted fusion to consider distant parts in the context, as illustrated by arrow (3) in Figure 2. Again, we achieve this via fully-aware attention on history-of-word. We identify the history-of-word to be

$$
\mathrm {H o W} _ {i} ^ {C} = \left[ \boldsymbol {g} _ {i} ^ {C}; \boldsymbol {c} _ {i} ^ {C}; \boldsymbol {h} _ {i} ^ {C l}; \boldsymbol {h} _ {i} ^ {C h}; \hat {\boldsymbol {h}} _ {i} ^ {C l}; \hat {\boldsymbol {h}} _ {i} ^ {C h}; \hat {\boldsymbol {u}} _ {i} ^ {C}; \boldsymbol {v} _ {i} ^ {C} \right] \in \mathbb {R} ^ {2 4 0 0}.
$$

We then perform fully-aware attention,  $\hat{\pmb{v}}_i^C = \sum_j\alpha_{ij}^s\pmb {v}_j^C$ $\alpha_{ij}^{s}\propto \exp (S^{s}(\mathrm{HoW}_{i}^{C},\mathrm{HoW}_{j}^{C}))$

The final context representation is obtained by

$$
U _ {C} = \left\{\boldsymbol {u} _ {1} ^ {C}, \dots , \boldsymbol {u} _ {m} ^ {C} \right\} = \operatorname {B i L S T M} \left(\left[ \boldsymbol {v} _ {1} ^ {C}; \hat {\boldsymbol {v}} _ {1} ^ {C} \right], \dots , \left[ \boldsymbol {v} _ {m} ^ {C}; \hat {\boldsymbol {v}} _ {m} ^ {C} \right]\right).
$$

where  $\{\pmb{u}_i^C\in \mathbb{R}^{250}\}_{i = 1}^m$  are the understanding vectors for  $C$ .

After these components in FusionNet, we have created the understanding vectors,  $U_{C}$ , for the context  $C$ , which are fully fused with the question  $Q$ . We also have the understanding vectors,  $U_{Q}$ , for the question  $Q$ .

# 3.2 APPLICATION IN MACHINE COMPREHENSION

We focus particularly on the output format in SQuAD (Rajpurkar et al., 2016) where the answer is always a span in the context. The output of FusionNet are the understanding vectors for both  $\mathbf{C}$  and  $\mathbf{Q}$ ,  $U_{C} = \{\pmb{u}_{1}^{C},\dots,\pmb{u}_{m}^{C}\}$ ,  $U_{Q} = \{\pmb{u}_{1}^{Q},\dots,\pmb{u}_{n}^{Q}\}$ .

We then use them to find the answer span in the context. Firstly, a single summarized question understanding vector is obtained through  $\pmb{u}^Q = \sum_i\beta_i\pmb{u}_i^Q$ , where  $\beta_{i}\propto \exp (\pmb{w}^{T}\pmb{u}_{i}^{Q})$  and  $\pmb{w}$  is a trainable vector. Then we attend for the span start using the summarized question understanding vector  $\pmb{u}^Q$ ,

$$
P _ {i} ^ {S} \propto \exp ((\pmb {u} ^ {Q}) ^ {T} W _ {S} \pmb {u} _ {i} ^ {C}),
$$

where  $W_{S} \in \mathbb{R}^{d \times d}$  is a trainable matrix. To use the information of span start when we attend for the span end, we combine the context understanding vector for the span start with  $\boldsymbol{u}^Q$  through a GRU (Chung et al., 2014),  $\boldsymbol{v}^Q = \mathrm{GRU}(\boldsymbol{u}^Q, \sum_i P_i^S \boldsymbol{u}_i^C)$ , where  $\boldsymbol{u}^Q$  is taken as the memory and  $\sum_i P_i^S \boldsymbol{u}_i^C$  as the input in GRU. Finally we attend for the end of the span using  $\boldsymbol{v}^Q$ ,

$$
P _ {i} ^ {E} \propto \exp ((\boldsymbol {v} ^ {Q}) ^ {T} W _ {E} \boldsymbol {u} _ {i} ^ {C}),
$$

where  $W_{E}\in \mathbb{R}^{d\times d}$  is another trainable matrix.

Training. During training, we maximize the log probabilities of the ground truth span start and end,  $\sum_{k}(\log (P_{i_k^s}^S) + \log (P_{i_k^e}^E))$ , where  $i_k^s, i_k^e$  are the answer span for the  $k$ -th instance.

Prediction. We predict the answer span to be  $i^s, i^e$  with the maximum  $P_{i^s}^S P_{i^e}^E$  under the constraint  $0 \leq i^e - i^s \leq 15$ .

# 4 EXPERIMENTS

In this section, we first present the datasets used for evaluation. Then we compare our end-to-end FusionNet model with existing machine reading models. Finally, we conduct experiments to validate the effectiveness of our proposed components. Detailed experimental settings can be found in Appendix C.

# 4.1 DATASETS

We focus on the SQuAD dataset (Rajpurkar et al., 2016) to train and evaluate our model. SQuAD is a popular machine comprehension dataset consisting of  $100,000+$  questions created by crowd workers on 536 Wikipedia articles. Each context is a paragraph from an article and the answer to each question is guaranteed to be a span in the context.

While rapid progress has been made on SQuAD, whether these systems truly understand language remains unclear. In a recent paper, Jia & Liang (2017) proposed several adversarial schemes to test the understanding of the systems. We will use the following two adversarial datasets, AddOneSent and AddSent, to evaluate our model. For both datasets, a confusing sentence is appended at the end of the context. The appended sentence is model-independent for AddOneSent, while AddSent requires querying the model a few times to choose the most confusing sentence.

# 4.2 MAIN RESULTS

We submitted our model to SQuAD for evaluation on the hidden test set. We also test the model on the adversarial SQuAD datasets. Two official evaluation criteria are used: Exact Match (EM) and F1 score. EM measures how many predicted answers exactly match the correct answer, while F1 score measures the weighted average of the precision and recall at token level. The evaluation results for our model and other competing approaches are shown in Table 2. $^{1}$  Additional comparisons with state-of-the-art models in the literature can be found in Appendix A.

For the two adversarial datasets, AddOneSent and AddSent, the evaluation criteria is the same as SQuAD. However, all models are trained only on the original SQuAD, so the model never sees the

<table><tr><td></td><td>Test Set</td></tr><tr><td>Single Model</td><td>EM/F1</td></tr><tr><td>LR Baseline (Rajpurkar et al., 2016)</td><td>40.4/51.0</td></tr><tr><td>Match-LSTM (Wang &amp; Jiang, 2016)</td><td>64.7/73.7</td></tr><tr><td>BiDAF (Seo et al., 2017)</td><td>68.0/77.3</td></tr><tr><td>SEDT (Liu et al., 2017)</td><td>68.2/77.5</td></tr><tr><td>RaSoR (Lee et al., 2016)</td><td>70.8/78.7</td></tr><tr><td>DrQA (Chen et al., 2017)</td><td>70.7/79.4</td></tr><tr><td>ReasoNet (Shen et al., 2017)</td><td>70.6/79.4</td></tr><tr><td>R.Mnemonic Reader (Hu et al., 2017)</td><td>73.2/81.8</td></tr><tr><td>DCN+</td><td>74.9/82.8</td></tr><tr><td>R-net (Wang et al., 2017)</td><td>75.7/83.5</td></tr><tr><td>FusionNet</td><td>76.0/83.9</td></tr><tr><td>Ensemble Model</td><td></td></tr><tr><td>ReasoNet (Shen et al., 2017)</td><td>75.0/82.3</td></tr><tr><td>MEMEN (Pan et al., 2017)</td><td>75.4/82.7</td></tr><tr><td>R.Mnemonic Reader (Hu et al., 2017)</td><td>77.7/84.9</td></tr><tr><td>R-net (Wang et al., 2017)</td><td>78.2/85.2</td></tr><tr><td>DCN+</td><td>78.7/85.6</td></tr><tr><td>FusionNet</td><td>78.8/85.9</td></tr><tr><td>Human (Rajpurkar et al., 2016)</td><td>82.3/91.2</td></tr></table>

Table 2: The performance of FusionNet and competing approaches on SQuAD hidden test set at the time of writing (Oct. 4th, 2017).

<table><tr><td>AddSent</td><td>EM/F1</td></tr><tr><td>LR Baseline</td><td>17.0/23.2</td></tr><tr><td>Match-LSTM (E)</td><td>24.3/34.2</td></tr><tr><td>BiDAF (E)</td><td>29.6/34.2</td></tr><tr><td>SEDT (E)</td><td>30.0/35.0</td></tr><tr><td>Mnemonic Reader (S)</td><td>39.8/46.6</td></tr><tr><td>Mnemonic Reader (E)</td><td>40.7/46.2</td></tr><tr><td>ReasoNet (E)</td><td>34.6/39.4</td></tr><tr><td>FusionNet (E)</td><td>46.2/51.4</td></tr></table>

Table 3: Comparison on AddSent. (S: Single model, E: Ensemble)  

<table><tr><td>AddOneSent</td><td>EM / F1</td></tr><tr><td>LR Baseline</td><td>22.3 / 30.4</td></tr><tr><td>Match-LSTM (E)</td><td>34.8 / 41.8</td></tr><tr><td>BiDAF (E)</td><td>40.7 / 46.9</td></tr><tr><td>SEDT (E)</td><td>40.0 / 46.5</td></tr><tr><td>Mnemonic Reader (S)</td><td>48.5 / 56.0</td></tr><tr><td>Mnemonic Reader (E)</td><td>48.7 / 55.3</td></tr><tr><td>ReasoNet (E)</td><td>43.6 / 49.8</td></tr><tr><td>FusionNet (E)</td><td>54.7 / 60.7</td></tr></table>

Table 4: Comparison on AddOneSent. (S: Single model, E: Ensemble)

adversarial datasets during training. The results for AddSent and AddOneSent are shown in Table 3 and Table 4, respectively. $^2$

From the results, we can see that our models not only perform well on the original SQuAD dataset, but also outperform all previous models by more than  $5\%$  in EM score on the adversarial datasets. This shows that FusionNet is better at language understanding of both context and question.

# 4.3 COMPARISON ON ATTENTION FUNCTION

In this experiment, we compare the performance of different attention scoring functions  $S(\pmb{x}, \pmb{y})$  for fully-aware attention. We utilize the end-to-end architecture presented in Section 3.1. Fully-aware attention is used in two places, fully-aware multi-level fusion: higher level and fully-aware self-boosted fusion. Word-level fusion remains unchanged. Based on the discussion in Section 2.3, we consider the following formulations for comparison:

1. Additive attention (MLP) (Bahdanau et al., 2015):  $\boldsymbol{s}^T \tanh(W_1 \boldsymbol{x} + W_2 \boldsymbol{y})$ .  
2. Multiplicative attention:  $\boldsymbol{x}^T\boldsymbol{U}^T\boldsymbol{V}\boldsymbol{y}$ .  
3. Scaled multiplicative attention:  $\frac{1}{\sqrt{k}}\pmb{x}^T\pmb{U}^T\pmb{V}\pmb{y}$ , where  $k$  is the attention hidden size. It is proposed in (Vaswani et al., 2017).  
4. Scaled multiplicative with nonlinearity:  $\frac{1}{\sqrt{k}} f(U\pmb{x})^T f(V\pmb{y})$  
5. Our proposed symmetric form:  $\pmb{x}^T\pmb{U}^T\pmb{D}\pmb{U}\pmb{y}$ , where  $D$  is diagonal.  
6. Proposed symmetric form with nonlinearity:  $f(U\pmb{x})^T Df(U\pmb{y})$ .

We consider the activation function  $f(x)$  to be  $\max(0, x)$ . The results of various attention functions on SQuAD development set are shown in Table 5. It is clear that the symmetric form consistently outperforms all alternatives. We attribute this gain to the fact that symmetric form has a single large matrix  $U$ . All other alternatives have two large parametric matrices. During optimization, these two parametric matrices would interfere with each other and it will make the entire optimization

<table><tr><td>Attention Function</td><td>EM/F1</td></tr><tr><td>Additive (MLP)</td><td>71.8/80.1</td></tr><tr><td>Multiplicative</td><td>72.1/80.6</td></tr><tr><td>Scaled Multiplicative</td><td>72.4/80.7</td></tr><tr><td>Scaled Multiplicative + ReLU</td><td>72.6/80.8</td></tr><tr><td>Symmetric Form</td><td>73.1/81.5</td></tr><tr><td>Symmetric Form + ReLU</td><td>75.3/83.6</td></tr></table>

<table><tr><td colspan="2">Configuration
C, Q Fusion Self C</td><td>Dev EM / F1</td></tr><tr><td>High-level</td><td rowspan="4">None</td><td>64.6 / 73.2</td></tr><tr><td>FA High-level</td><td>73.3 / 81.4</td></tr><tr><td>FA All-level</td><td>72.3 / 80.7</td></tr><tr><td>FA Multi-level</td><td>74.6 / 82.7</td></tr><tr><td>FA Multi-level</td><td>Normal
FA</td><td>74.4 / 82.6
75.3 / 83.6</td></tr></table>

Table 5: Comparison of different attention functions  $S\left( {\mathbf{x},\mathbf{y}}\right)$  on SQuAD dev set.

Table 6: Comparison of different configurations demonstrates the effectiveness of history-of-word.

process challenging. Besides, by constraining  $U^T V$  to be a symmetric matrix  $U^T D U$ , we retain the ability for  $x$  to attend to dissimilar  $y$ . Furthermore, its marriage with the nonlinearity continues to significantly boost the performance.

# 4.4 EFFECTIVENESS OF HISTORY-OF-WORD

In FusionNet, we apply the history-of-word and fully-aware attention in two major places to achieve good performance: multi-level fusion and self-boosted fusion. In this section, we present experiments to demonstrate the effectiveness of our application. In the experiments, we fix the attention function to be our proposed symmetric form with nonlinearity due to its good performance shown in Section 4.3. The results are shown in Table 6, and the details for each configuration can be found in Appendix B.

High-level is a vanilla model where only the high-level information is fused from  $Q$  to  $C$  via standard attention. When placed in the conceptual architecture (Figure 2), it only contains arrow (2) without any other fusion processes.

FA High-level is the High-level model with standard attention replaced by fully-aware attention.

FA All-level is a naive extension of FA High-level, where all levels of information are concatenated and is fused into the context using the same attention weight.

FA Multi-level is our proposed Fully-aware Multi-level fusion, where different levels of information are attended under separate attention weight.

Self  $C =$  None means we do not make use of self-boosted fusion.

Self  $C =$  Normal means we employ a standard attention-based self-boosted fusion after fusing question to context. This is illustrated as arrow (3) in the conceptual architecture (Figure 2).

Self  $C = \mathbf{FA}$  means we enhance the self-boosted fusion with fully-aware attention.

High-level vs. FA High-level. From Table 6, we can see that High-level performs poorly as expected. However enhancing this vanilla model with fully-aware attention significantly increases the performance by more than  $8\%$ . The performance of FA High-level already outperforms many state-of-the-art MRC models. This clearly demonstrates the power of fully-aware attention.

FA All-level vs. FA Multi-level. Next, we consider models that fuse all levels of information from question  $\mathbf{Q}$  to context  $\mathbf{C}$ . FA All-level is a naive extension of FA High-level, but its performance is actually worse than FA High-level. However, by fusing different parts of history-of-word in  $\mathbf{Q}$  independently as in FA Multi-level, we are able to further improve the performance.

Self  $C$  options. We have achieved decent performance without self-boosted fusion. Now, we compare adding normal and fully-aware self-boosted fusion into the architecture. Comparing None and Normal in Table 6, we can see that the use of normal self-boosted fusion is not very effective under our improved  $C$ ,  $Q$  Fusion. Then by comparing with  $FA$ , it is clear that through the enhancement of fully-aware attention, the enhanced self-boosted fusion can provide considerable improvement.

Together, these experiments demonstrate that the ability to take all levels of understanding as a whole is crucial for machines to better understand the text.

# 5 CONCLUSIONS

In this paper, we describe a new deep learning model FusionNet with its application to machine comprehension. FusionNet proposes a novel attention mechanism with following three contributions: 1. the concept of "history of words" to build the attention using complete information from lowest word-level embedding up to the highest semantic-level representation; 2. a new scoring function to effectively and efficiently fuse information between question and context; 3. a fully-aware multilevel fusion to exploit information layer by layer discriminatingly. We applied FusionNet to MRC task and experimental results show that FusionNet outperforms existing machine reading models on both SQuAD dataset and the adversarial SQuAD dataset. We believe FusionNet is a general and improved attention mechanism and can be applied to many tasks. Our future work is to study its capability in other NLP problems.

# REFERENCES

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. *ICLR*, 2015.  
Denny Britz, Anna Goldie, Thang Luong, and Quoc Le. Massive exploration of neural machine translation architectures. arXiv preprint arXiv:1703.03906, 2017.  
Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. Reading wikipedia to answer open-domain questions. arXiv preprint arXiv:1704.00051, 2017.  
Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.  
Yarin Gal and Zoubin Ghahramani. A theoretically grounded application of dropout in recurrent neural networks. In NIPS, 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 1997.  
Minghao Hu, Yuxing Peng, and Xipeng Qiu. Reinforced mnemonic reader for machine comprehension. arXiv preprint arXiv:1705.02798, 2017.  
Robin Jia and Percy Liang. Adversarial examples for evaluating reading comprehension systems. EMNLP, 2017.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Kenton Lee, Shimi Salant, Tom Kwiatkowski, Ankur Parikh, Dipanjan Das, and Jonathan Berant. Learning recurrent span representations for extractive question answering. arXiv preprint arXiv:1611.01436, 2016.  
Rui Liu, Junjie Hu, Wei Wei, Zi Yang, and Eric Nyberg. Structural embedding of syntactic trees for machine comprehension. arXiv preprint arXiv:1703.00572, 2017.  
B. McCann, J. Bradbury, C. Xiong, and R. Socher. Learned in Translation: Contextualized Word Vectors. arXiv preprint arXiv:1708.00107, 2017.  
Boyuan Pan, Hao Li, Zhou Zhao, Bin Cao, Deng Cai, and Xiaofei He. Memen: Multi-layer embedding with memory networks for machine comprehension. arXiv preprint arXiv:1707.09098, 2017.  
Jeffrey Pennington, Richard Socher, and Christopher Manning. Glove: Global vectors for word representation. In EMNLP, 2014.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. EMNLP, 2016.  
Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. Bidirectional attention flow for machine comprehension. In ICLR, 2017.

Yelong Shen, Po-Sen Huang, Jianfeng Gao, and Weizhu Chen. Reasonet: Learning to stop reading in machine comprehension. In KDD, 2017.  
Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. JMLR, 2014.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. 2017.  
Shuohang Wang and Jing Jiang. Machine comprehension using match-lstm and answer pointer. arXiv preprint arXiv:1608.07905, 2016.  
Wenhui Wang, Nan Yang, Furu Wei, Baobao Chang, and Ming Zhou. Gated self-matching networks for reading comprehension and question answering. In ACL, 2017.  
Zhiguo Wang, Haitao Mi, Wael Hamza, and Radu Florian. Multi-perspective context matching for machine comprehension. arXiv preprint arXiv:1612.04211, 2016.  
Dirk Weissenborn, Georg Wiese, and Laura Seiffe. Making neural qa as simple as possible but not simpler. In CoNLL, 2017.  
Caiming Xiong, Victor Zhong, and Richard Socher. Dynamic coattention networks for question answering. *ICLR*, 2017.
