# COMPOSITIONAL CONTINUAL LANGUAGE LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Motivated by the human's ability to continually learn and gain knowledge over time, several research efforts have been pushing the limits of machines to constantly learn while alleviating catastrophic forgetting Kirkpatrick et al. (2017b); significant drop of a machine skill accessed/gained far earlier in time. Most of the existing methods have been focusing on label prediction tasks to study continual learning. Humans, however, naturally interact and learn from natural language statements and instructions which is far less studied from continual learning angle. One of the key skills that enables humans to excel at learning language efficiently is ability to produce novel composition. To learn and complete new tasks, robots need to continually learn novel objects and concepts in a linguistic form which requires compositionality for efficient learning. Inspired by that, in this paper, we propose a method for compositional continual learning of sequence-to-sequence models. Experimental results show that the proposed method has significant improvement over state of the art methods, and it enables knowledge transfer and prevents catastrophic forgetting, resulting in more than  $85\%$  accuracy up to 100 stages, compared with less  $50\%$  accuracy for baselines. It also shows significant improvement in a machine translation task. This is the first work to combine continual learning and compositionality for natural language instruction learning and machine translation, and we hope this work will make robots more helpful in various tasks.

# 1 INTRODUCTION

Continual Learning is a key element of human intelligence that enables us to accumulate knowledge from a never ending stream of data. From machine learning perspective, there is no guarantee that information accessed at a current task to be revisited later in future tasks. This leads to what is known as Catastrophic Forgetting (McCloskey & Cohen, 1989; McClelland et al., 1995); significant drop in previously obtained knowledge of an AI system as it learns new information and gets less/no exposure to old information. Several approaches have been proposed to bridge the map between machine and human continual learning skills with catastrophic forgetting being the central problem. Existing continual learning methods have focused mostly on classification tasks (e.g. (Rebuffi et al., 2017; Lopez-Paz & Ranzato, 2017; Shin et al., 2017; Li & Hoiem, 2016; Shmelkov et al., 2017; Triki et al., 2017; Li & Hoiem, 2016; Triki et al., 2017; Rusu et al., 2016; Lee et al., 2017; Elhoseiny et al., 2018; Kirkpatrick et al., 2017c; Zenke et al., 2017b; Chaudhry et al., 2018)). However, humans learn language by leveraging systematic compositionality; the algebraic capacity to understand and produce large amount of novel combinations from known components (Chomsky, 1957; Montague, 1970). Compositional generalization is critical in human cognition (Minsky, 1986; Lake et al., 2017). It also helps humans acquire language from a small amount of data, and expand vocabulary sequentially (Biemiller, 2001). Modeling continual language learning with improved compositional understanding is at the heart of this paper. In contrast to humans' ability to acquire this skill continually, State-of-the-art continual learning approaches fail to achieve the expected generalization. Table 1 and Figure 2 show the performance of state-of-the-art approaches (Kirkpatrick et al., 2017a; Aljundi et al., 2018) when tested in instruction learning and machine translation tasks. This highlights the lack of generalization of these approaches, designed after classification tasks, on sequence generation language tasks and the importance of studying the design of continual learning methods for language learning.

In this paper, we address the question of how compositionality could enable better continual language learning. We show that designing compositional continual learning approach significantly re

<table><tr><td rowspan="2">Method\Stage</td><td colspan="3">Transfer</td><td colspan="3">Forget</td><td colspan="3">Long-forget</td></tr><tr><td>1</td><td>10</td><td>100</td><td>1</td><td>10</td><td>100</td><td>1</td><td>10</td><td>100</td></tr><tr><td>Standard</td><td>2.3</td><td>0.2</td><td>0.0</td><td>30.8</td><td>0.9</td><td>0.0</td><td>30.8</td><td>11.2</td><td>7.9</td></tr><tr><td>Compositional</td><td>98.8</td><td>15.0</td><td>0.0</td><td>99.3</td><td>71.7</td><td>0.7</td><td>99.3</td><td>85.5</td><td>47.4</td></tr><tr><td>EWC</td><td>2.8</td><td>0.2</td><td>0.0</td><td>35.0</td><td>1.0</td><td>0.2</td><td>35.0</td><td>11.5</td><td>11.1</td></tr><tr><td>MAS</td><td>0.6</td><td>0.2</td><td>0.0</td><td>20.0</td><td>0.8</td><td>0.1</td><td>20.0</td><td>10.8</td><td>9.8</td></tr><tr><td>Proposed</td><td>99.9</td><td>99.8</td><td>90.7</td><td>100.0</td><td>99.9</td><td>89.5</td><td>100.0</td><td>100.0</td><td>86.0</td></tr></table>

Table 1: Mean of evaluation accuracy (\%) on instruction learning tasks (Section 4 for details). Baselines include Compositional (Li et al., 2019), EWC (Kirkpatrick et al., 2017a), and MAS (Aljundi et al., 2018). Please refer to Table 3 in Appendix for more results and standard deviations.

duces catastrophic forgetting in Natural Language instructions and machine translation tasks. From language perspective, continual language learning is important due to changing and growing vocabulary. Continual language learning may facilitate a variety of applications in NLP systems. For example, it enables a robot to keep on learning new tasks via natural language instruction, a conversational agent to adapt to new conversation topics quickly, and a neural machine translation system to expand its vocabulary continually.

More concretely, we address the challenge of open and growing vocabulary problem with continual learning. Continual learning of language is less studied in deep neural networks especially with ever-growing vocabulary set. It requires optimizing over two objectives. First, previously learned knowledge should be transferred and combined with new knowledge. Second, the learned model should resist catastrophic forgetting (Kirkpatrick et al., 2017b), where a model adapted to a new distribution no longer works on the original one. To achieve these objectives, we use compositionality to separate semantics and syntax of an input sentence, so that we can convert label prediction algorithm to sequence to sequence algorithm for continual learning.

The contributions of this paper can be summarized as follows.

- We propose a new scenario of continual learning which handles sequence-to-sequence tasks common in language learning.  
- We propose an approach to convert label prediction continual learning algorithm to sequence-to-sequence continual learning algorithm by leveraging compositionality. To our knowledge, this is the first work for applying compositionality to continual learning of sequence-to-sequence tasks, targeting at both knowledge transfer to later stages and catastrophic forgetting prevention on previous stages.  
- Experiments show that the proposed method has significant improvement over multiple state-of-the-art baselines in both knowledge transfer and catastrophic forgetting prevention with almost  $85\%$  accuracy up to 100 stages on language instruction tasks. It also shows significant improvement in a machine translation task. The source codes are included in supplementary material, and will be publicly available upon acceptance.

# 2 RELATED WORK

Our work is closely related to compositionality, continual learning or lifelong learning. Here, we briefly review some related work in these areas.

Compositionality Compositional generalization is critical in human cognition Minsky (1986); Lake et al. (2017), and it helps humans acquire language from a small amount of data, and expand vocabulary sequentially Biemiller (2001). Therefore, researchers have been studying how to enable human-level compositionality in neural networks for systematic behaviour (Wong & Wang, 2007; Brakel & Frank, 2009), counting ability (Rodriguez & Wiles, 1998; Weiss et al., 2018) and sensitivity to hierarchical structure (Linzen et al., 2016). Recently, people proposed multiple related tasks (Lake & Baroni, 2018; Loula et al., 2018; Lake et al., 2019) and methods (Lake & Baroni, 2018; Loula et al., 2018; Kliegl & Xu, 2018) with different kinds of RNN models and attention mechanisms. Though these methods enable generalization when the training and test sentences

have small difference, it has been an open problem (Yang et al., 2019) to reach human-level compositionality generalization. More recently, Li et al. (2019) proposed an entropy regularization method that achieves high performance on several NLP tasks. By leveraging the compositional learning approach, we propose the continual learning algorithm by encoding compositionality into DNN. To our knowledge, our work is the first to apply compositionality to continual learning in DNN.

Continual learning Continual learning or lifelong learning involves multiple stages. Each stage has a set of classes and corresponding data, and the training can only access the data in the current stage. Based on the way for overcoming catastrophic forgetting, continual learning work may be categorized into data-based and model-based approaches. In data-based approaches, some methods store previous data either with replay buffer (Rebuffi et al., 2017; Lopez-Paz & Ranzato, 2017) or generative model (Shin et al., 2017); other approaches (Li & Hoiem, 2016; Shmelkov et al., 2017; Triki et al., 2017), employ the new task data to estimate and preserve the model behavior on previous tasks, mostly via a knowledge distillation loss as proposed in Learning without Forgetting (Li & Hoiem, 2016). These approaches are typically applied to a sequence of tasks with different output spaces. To reduce the effect of distribution difference between tasks, (Triki et al., 2017) propose to incorporate a shallow auto-encoder to further control the changes to the learned features, while (Aljundi et al., 2017) train a model for every task (an expert) and use auto-encoders to help determine the most related expert at test time given an example input. In model-based approaches, some methods dynamically increase model size for the growing information (Rusu et al., 2016; Xu & Zhu, 2018); other methods (Fernando et al., 2017; Lee et al., 2017; Kirkpatrick et al., 2017c; Zenke et al., 2017b) focus on the parameters of the network. The key idea is to define an importance weight  $\omega_{k}$  for each parameter  $\theta_{k}$  in the network indicating the importance of this parameter to the previous tasks. When training a new task, network parameters with high importance are discouraged from being changed. In Elastic Weight Consolidation, (Kirkpatrick et al., 2017c) estimate the importance weights  $\Omega$  based on the inverse of the Fisher Information matrix. (Zenke et al., 2017b) propose Synaptic Intelligence, an online continual model where  $\Omega$  is defined by the contribution of each parameter to the change in the loss, and weights are accumulated for each parameter during training. Memory Aware Synapses (Aljundi et al., 2018) measures  $\Omega$  by the effect of a change in the parameter to the function learned by the network, rather than to the loss. This allows to estimate the importance weights not only in an online fashion but also without the need for labels. Finally, Incremental Moment Matching (Lee et al., 2017) is a scheme to merge models trained for different tasks. Model-based methods seem particularly well suited for our setup, given that we work with an embedding instead of disjoint output spaces. In this paper, we propose a method with minimal increase of model structure in each stage, and we leverage compositionality with explainable mechanisms that align with human learning.

Multi-stage continual learning has been mainly explored with classification tasks (Zenke et al., 2017a; Nguyen et al., 2017; Aljundi et al., 2019). Conventional continual learning algorithms are designed for fixed size input and label classification output. However in many tasks, such as language, both input and output are sequence. Our approach bridges the gap by using compositionality in language.

# 3 CONTINUAL LEARNING WITH COMPOSITIONALITY

# 3.1 PROBLEM DEFINITION

Conventional continual learning algorithms are designed after fixed size input and label classification output. However in many tasks, such as language, both input and output are sequences and bridging this gap between continual learning and sequence-to-sequence models is at the heart of our work. We facilitate more accurate continual sequence-to-sequence artificial learner by proposing an approach that can leverage Label Prediction Continual Learning (LP-CL) compositionally into Sequence-to-Sequence Continual Learning (S2S-CL).

LP-CL: Label Prediction Continual Learning In LP-CL, we consider a word to label mapping problem, with input word  $x$  and corresponding output label  $y$ . In initial learning stage,  $y$  takes one of  $K$  classes:  $y \in V_{\mathrm{init}} = \{c_1, c_2, \ldots, c_K\}$ . In continual learning stage,  $y$  takes a new class:  $y \in V_{\mathrm{cont}} = \{c_{K+1}\}$ . In test,  $y$  takes all previous classes:  $y \in V_{\mathrm{init}} \cup V_{\mathrm{cont}}$ . For example, in

language instruction task, input  $x$  is a primitive word, and output  $Y$  is the corresponding action symbol; in word-level machine translation, input  $x$  is an English content word, and output  $Y$  is the corresponding French word. In initial training stage, we have multiple input word and output symbol pairs. In continual learning stage, we have a new input and output word pair. We train a model in initial training stage, and do not use the data any longer. We then switch to the data in continual learning stage, and continually updating the model. In test stage, we evaluate whether model can predict labels from both initial and continual learning stages. We denote label prediction continual learning model (LP-CL) as  $P(y|x;\theta)$ .

S2S-CL: Sequence to Sequence Continual Learning For sequence to sequence continual learning (S2S-CL), we consider sequential input  $X = x_{1},x_{2},\ldots ,x_{n}$  and output  $Y = y_{1},y_{2},\ldots ,y_{m}$ . Each output label  $y_{i}, i\in \{1,\dots,m\}$  is from the corresponding label set in label prediction problem. We want to make a model  $P(Y|X)$  for sequence to sequence continual learning.

Our goal is to facilitate better Sequence to Sequence Continual Learning (S2S-CL) capability quantified as  $P(Y|X)$  by leveraging access and joint-learning with Label Prediction Continual Learning (LP-CL) model,  $P(y|x;\theta)$ .

# 3.2 USE LP-CL ALGORITHM FOR S2S-CL WITH COMPOSITIONALITY

The core idea of this work is to use compositionality to separate semantics and syntax, so that we can convert label prediction algorithm to sequence to sequence algorithm for continual learning. In Kirkpatrick et al. (2017a), continual learning can be probabilistically defined as follows.

$$
\log P (\theta | \mathcal {D}) = \log P \left(\mathcal {D} _ {T} | \theta\right) + \log P \left(\theta \mid \mathcal {D} _ {1 \dots T - 1}\right) - \log P \left(\mathcal {D} _ {T}\right)
$$

Here,  $\log P(\mathcal{D}_T|\theta)$  is the negative of loss function in task  $T$ , and  $\log P(\theta|\mathcal{D}_{1\dots T-1})$  is regularization related to parameters learned during  $1\dots T-1$ . In this work, we have two parts of parameters  $\psi = \theta, \phi$  for semantics  $\theta$  and syntax  $\phi$ . With compositionality (Li et al., 2019), we make  $\theta$  and  $\phi$  conditionally independent given input of data  $\mathcal{D}_{1\dots T-1}$ .

$$
\begin{array}{l} \log P (\psi | \mathcal {D}) = \log P (\mathcal {D} _ {T} | \psi) + \log P (\theta , \phi | \mathcal {D}) - \log P (\mathcal {D} _ {T}) \\ \log P (\psi | \mathcal {D}) = \log P (\mathcal {D} _ {T} | \psi) + \log P (\theta | \mathcal {D} _ {1 \dots T - 1}) + \log P (\phi | \mathcal {D} _ {1 \dots T - 1}) - \log P (\mathcal {D} _ {T}) \\ \end{array}
$$

We assume syntax  $\phi$  do not change over time, so we realize regularization  $\log P(\phi | \mathcal{D}_{1\dots T-1})$  by freezing  $\phi$  during learning in task  $T$ . We use label prediction continual learning algorithm for regularization  $\log P(\theta | \mathcal{D}_{1\dots T-1})$ .

Based on the above arguments, we derive the proposed approach. To use label prediction algorithm in sequence to sequence problem, we need to extract label prediction problem from sequence to sequence model. Language is generally composed of semantics  $p$  and syntax  $f$ , so that we decompose an input sequence to them with compositionality.

In this task, input  $X$  is a word sequence, and output  $Y$  is a label sequence. To better compositional access to word sequences, we consider  $X$  has two types of information: which labels are present  $(X^p)$ , and how the labels should be ordered  $(X^f)$ .  $Y$  is constructed by the output label types  $(Y^p)$ , and output label order  $(Y^f)$ . We can use a sequence of attention maps on input index for the output order  $Y^f$ .  $Y^f$  functionally depends only on  $X^f$ . Given  $Y^f$ ,  $Y^p$  depending only on  $X^p$ . For an intuitive example, in language instruction example, output order of actions depends only on input function words (syntax), and given the order, each output action (semantic) only depends on the corresponding input primitive. In machine translation, output order only depends on input part-of-speech information (syntax), and given the order, each output word label (semantics) only depends on the corresponding input word.

$$
\begin{array}{l} P (Y | X) = P \left(Y ^ {f}, Y ^ {p} \mid X ^ {f}, Y ^ {p}\right) \\ = P \left(Y ^ {f} \mid X ^ {f}, Y ^ {p}\right) P \left(Y ^ {p} \mid Y ^ {f}, X ^ {f}, Y ^ {p}\right) \\ = P \left(Y ^ {f} \mid X ^ {f}\right) P \left(Y ^ {p} \mid Y ^ {f}, X ^ {p}\right) \\ \end{array}
$$

Since we aim to enable LP continual learning to communicate with S2S continual learning (our goal), we decompose output sequence to labels. We assume that the labels  $y_{1}, \ldots, y_{m}$  are conditionally independent given output order  $Y^{f}$  and semantic information  $X^{p}$ . We then use total probability and further design that  $x_{i}^{p}$  depends only on  $y_{j}^{f}$  and  $X^{p}$ , which can be implemented by attention mechanism. With label prediction component,  $y_{j}^{p}$  depends only on input word  $x_{i}^{p}$ .

$$
\begin{array}{l} P (Y | X) = P \left(Y ^ {f} | X ^ {f}\right) \prod_ {j = 1} ^ {m} P \left(y _ {j} ^ {p} | Y ^ {f}, X ^ {p}\right) \\ = P \left(Y ^ {f} \mid X ^ {f}\right) \prod_ {j = 1} ^ {m} \sum_ {i = 1} ^ {n} P \left(x _ {i} ^ {p} \mid Y ^ {f}, X ^ {p}\right) P \left(y _ {j} ^ {p} \mid x _ {i} ^ {p}, Y ^ {f}, X ^ {p}\right) \\ = P (Y ^ {f} | X ^ {f}) \prod_ {j = 1} ^ {m} \sum_ {i = 1} ^ {n} P (x _ {i} ^ {p} | y _ {j} ^ {f}, X ^ {p}) P (y _ {j} ^ {p} | x _ {i} ^ {p}) \\ \end{array}
$$

$P(x_{i}^{p}|y_{j}^{f},X^{p})$  is an operation to apply attention map  $y_{j}^{f}$  on value vector  $X^p$ , so that it does not have parameters. Let  $\theta$  be the parameter for label prediction module  $P(y_{j}^{p}|x_{i}^{p};\theta)$ , and  $\phi$  be the parameter for attention map generator  $P(Y^{f}|X^{f};\phi)$ .

$$
P (Y | X) = P (Y ^ {f} | X ^ {f}; \phi) \prod_ {j = 1} ^ {m} \sum_ {i = 1} ^ {n} P _ {\mathrm {a t t}} (x _ {i} ^ {p} | y _ {j} ^ {f}, X ^ {p}) P (y _ {j} ^ {p} | x _ {i} ^ {p}; \theta)
$$

Since the continual learning stage contains only new semantic word, we may expect the syntactic information  $Y^{f}$  and  $X^{f}$  are not new, so we can just freeze  $\phi$  during continual learning stage.  $\theta$  is the parameter for label prediction module. Therefore, we can use label prediction continual learning model (LP-CL) to enable compositional sequence to sequence continual learning (S2S-CL) as we detail in the next subsection.

# 3.3 DISENTANGLE SEMANTIC AND SYNTACTIC REPRESENTATIONS

Our S2S-CL approach is inspired from the idea decomposing syntactic and semantic representation from the compositional sequence-to-sequence by Li et al. (2019). Note that (Li et al., 2019) is not a continual learning approach but shows how compositionality can be encoded in sequence-to-sequence models. In this section we briefly describe the how (Li et al., 2019) enable such disentanglement and in the following section we detail our continual learning approach while modeling the compositional characteristic continually. In (Li et al., 2019), the method disentangles syntactic and semantic representations. It processes an input sentence by generating a sequence of attention maps from the syntactic representation and then use the attended semantic representation to generate an output sequence.

Suppose there are input  $x$  and output  $y$ .  $x$  contains a sequence of words, where each input word is from an input vocabulary of size  $U$ .  $y$  contains a sequence of output symbols, where each output symbol is from an output vocabulary of size  $V$ . Both vocabularies contain an end-of-sentence symbol which appears at the end of  $x$  and  $y$ , respectively. The model output  $\hat{y}$  is a prediction for  $y$ . Suppose both input words and output symbols are in one-hot representation, i.e.,

$$
x = [ x _ {1}, \ldots , x _ {n} ] \in \{0, 1 \} ^ {U \times n}, \qquad \qquad y = [ y _ {1}, \ldots , y _ {m} ] \in \{0, 1 \} ^ {V \times m}
$$

To disentangle information, an input sentence  $x$  is converted to semantic representation  $p$  and syntactic representation  $f$ . Specifically, each word is encoded with two embeddings.

$$
p _ {i} = \operatorname {E m b} _ {p} (x _ {i}) \in \mathbb {R} ^ {k _ {p}}, \quad f _ {i} = \operatorname {E m b} _ {f} (x _ {i}) \in \mathbb {R} ^ {k _ {f}}
$$

Then, they are concatenated to form two representations for the entire input sequence, i.e.,

$$
p = [ p _ {1}, \ldots , p _ {n} ] \in \mathbb {R} ^ {k _ {p} \times n}, \qquad \qquad f = [ f _ {1}, \ldots , f _ {n} ] \in \mathbb {R} ^ {k _ {f} \times n}
$$

Entropy regularization is introduced to achieve disentanglement by regularizing the  $L_{2}$  norm of the representations  $\mathcal{L}_{\mathrm{regularize}} = L_2(p) + L_2(f)$ , and then adding noise to the representations.

$$
p ^ {\prime} = p + \alpha \epsilon_ {p} \in \mathbb {R} ^ {k _ {p} \times n}, \epsilon_ {p} \sim \mathcal {N} (0, I), \quad f ^ {\prime} = f + \alpha \epsilon_ {f} \in \mathbb {R} ^ {k _ {f} \times n}, \epsilon_ {f} \sim \mathcal {N} (0, I)
$$

$f^{\prime}$  is fed to a sequence-to-sequence module for decoding. At each step  $j$ , the decoder generates  $b_{j} \in \mathbb{R}^{n}$ , and attention map  $a_{j}$  is obtained with Softmax. With the attention map, weighted average  $v_{j}$  on noised semantic representations  $p^{\prime}$  is computed. Then it is fed to a fully connected one-layer network  $f_{\mathrm{predict}}$  to get score  $l_{j}$ , and a Softmax is used to compute the output distribution  $\hat{y}_j$ . The decoding ends if  $\arg \max \hat{y}_j$  is an end-of-sentence symbol.

$$
a _ {j} = \operatorname {S o f t m a x} \left(b _ {j}\right), \quad v _ {j} = a _ {j} p ^ {\prime} \in \mathbb {R} ^ {k _ {p}}, \quad l _ {j} = f _ {\text {p r e d i c t}} \left(v _ {j}\right) \in \mathbb {R} ^ {V}, \quad \hat {y} _ {j} = \operatorname {S o f t m a x} \left(l _ {j}\right)
$$

The cross entropy of  $y$  and  $\hat{y}$  is used as prediction loss  $\mathcal{L}_{\mathrm{prediction}}$ , and the final loss  $\mathcal{L}$  is the combination of prediction loss and entropy regularization loss.  $\lambda$  is regularization weight.

$$
\mathcal {L} _ {\text {p r e d i c t i o n}} = \sum_ {j = 1} ^ {m} \operatorname {C r o s s E n t r o p y} \left(y _ {j}, \hat {y} _ {j}\right), \quad \mathcal {L} = \mathcal {L} _ {\text {p r e d i c t i o n}} + \lambda \mathcal {L} _ {\text {r e g u l a r i z e}}
$$

# 3.4 LABEL PREDICTION ALGORITHM FOR CONTINUAL LANGUAGE LEARNING

In language problem, it is natural to use non-parametric algorithm as label prediction continual learning algorithm, because each word is usually associated with embeddings. In each stage, since the original method uses two embeddings for a word, we append the word embeddings for the new word in the stage for semantic, syntactic and action embeddings (Fig. 1). We freeze the old embedding parameters and only learn the newly added ones in the stage.

![](images/eadf96cfebe40af35f2311cfbbf9b1883318aa25823ebf9d3aff5d3a54e555c1.jpg)  
Figure 1: Illustration for the first continual learning stage. Left is input word embedding (we only show one of two input word embeddings for simplicity). Middle is model architecture. Right is output action embedding. Parameters and data for the input word and output action embeddings of previous stage are in blue (filled boxes, solid lines), and for the new stage are in red (unfilled boxes, dashed lines). Other parts of the network are in black (unfilled boxes, solid lines).

# 4 EXPERIMENTS

We evaluate the proposed method in a continual learning task with multiple stages. The first stage is a standard process in which we train a model with combinations of multiple words in various sentence structures. In each of continual stage, we add a new input word and corresponding new output symbol. The training dataset contains only one sample, whose input is a sentence with the new word, and output is a sequence with the new symbol. For each stage, we can only use the data in that stage, and have no access to data in previous or future stages.

We have two objectives in continual learning. We want previously learned knowledge to be transferred and combined with new knowledge (transfer learning), and an updated model to work on previous data (catastrophic forgetting prevention). We evaluate transfer learning by testing whether the model works on data where the new word appears with old ones (Transfer). We evaluate catastrophic forgetting prevention by testing whether the model works on data that only contain words up to the last stage (Forget). We are also interested in preventing long-term catastrophic forgetting, because it is more difficult than preventing short-term one. Thus, we test whether the new model works on the evaluation dataset in the initial stage (Long-forget).

Baselines. We designed baseline methods for compositionality Sequence-to-Sequence continual learning to validate our approach since, to our knowledge, this is the first work for continual learning of natural language instructions and machine translation. We applied standard sequence-to-sequence model (Standard) to our continual setting, and also and the compositional generalization method (Compositional) (Li et al., 2019). We also compare with state of the art continual learning baselines. To fit in the experimental setting, we focus on those that do not use replay buffer, and require minimum model structure extension, so that we added EWC (Kirkpatrick et al., 2017a) and MAS (Aljundi et al., 2018) as comparable baselines due to their popularity and competitive performance in label prediction setting. The detailed implementation of the baseline and proposed methods can be found in Appendix B.

Metric. We use accuracy as metric for both instruction learning and machine translation experiments. A prediction is correct if and only if it is completely identical to the ground truth. We run all experiments for five times with different random seeds.

![](images/b468986588e1180189854c4f4bcbe2faaf4ecc38a5ee0d8b88cdaa54898f3558.jpg)  
(a) Transfer.

![](images/a05bf45d6eaf1feeda0ab6123db786a9ebe2817980f3b3b446dcd4d660b7d406.jpg)  
(d) Transfer.

![](images/fb71c536e3ef82b5f35e6af1bc9d1a268afb6852be4afe9eae14f6485113d722.jpg)  
(b) Forget.

![](images/67652a46a1f746778c9f2086d370e0848ba77a058ebbebeeb1298aa6e5145e4a.jpg)  
(e) Forget.

![](images/c8d1bdb324b06b06beb048605c5b4076826b279ffeb3c1f1d42ea501e6d34c41.jpg)  
(c) Long-forget.

![](images/81e3224c54a14234b2189f4ac0fc2385bcd934a005f2ec3a8085c29826aafaeb.jpg)  
(f) Long-forget.  
Figure 2: Mean of evaluation accuracy  $(\%)$  for all methods (best viewed in color). Baselines include Compositional (Li et al., 2019), EWC (Kirkpatrick et al., 2017a), and MAS (Aljundi et al., 2018). The proposed method is significantly better than all baselines. Please refer to Figure 3 and Figure 4 in Appendix for details and deviations.

Instruction Learning We first experiment on instruction learning task using SCAN dataset (Lake & Baroni, 2017). The task is summarized in Table 2 in Appendix. The details of dataset generation is in Appendix A. The results are in Figure 2 (left) and Table 1 (more details on Table 3 in Appendix). The proposed method has significantly better results than the baselines. It maintains high accuracy up to 100 stages for both transferring knowledge from previous stages to future stages, and catastrophic forgetting prevention. On the other hand, baseline methods drop performance over time. Methods without compositionality (EWC, MAS) reduces quickly, maybe because they are not designed for transferring knowledge, and since the representations are entangled, all parameters are quickly changed, causing catastrophic forgetting. Compositional method is better, but still drops, maybe because the parameters for syntax are changed over time. This experiment shows the advantage of the proposed method over baselines.

Machine Translation We also investigated whether the proposed approach works for other continual language learning problems. As an example, we conduct a proof-of-concept experiment for machine translation. We modified the English-French translation task in (Lake & Baroni, 2018). In each continual learning stage, we add an additional English-French word pair, in the format ("I am ENGLISH", "je suis FRENCH"). Neither English word nor French word appears in previous stages. This pair is used as training data in the stage, but test data contains other patterns. Appendix A provides more details on dataset and model configuration. The result is shown in Figure 2 (right) and Table 4 in Appendix. It shows that the proposed approach has stable and significantly higher performance than baselines. For Transfer and Forget evaluation, the baseline methods drop quickly. However, for Long-forget evaluation, they keep positive accuracy over time. This means the baseline methods have ability to learn knowledge and remember for long time, but they are not as strong as the proposed method. This experiment shows that the proposed approach has promise to be applied to real-world tasks.

# 5 DISCUSSIONS

# 5.1 ATTENTION MAP VISUALIZATION

We hope to use compositionality for continual learning, so we want to find whether the model works in the expected mechanism. We visualize activations of attention maps on the evaluation data in the first continual stage (Figure 3).

![](images/489683e3cac4ce16c0d4d6ef18b36bb423fbb7fcda44b4a4136d9163787e2737.jpg)  
(a) Transfer.

![](images/32d7c7087ccdb1b59fbda76409ce15fe84333a26c714f014d628089ff89a4775.jpg)  
(b) Forget.  
Figure 3: Visualization of attention maps. The horizontal and vertical dimensions are the input and output position sequences respectively. The figures show that the model identifies the appropriate input to output position mapping. This indicates that the proposed method successfully leverages compositionality in continual learning.

![](images/c9f9ada41f3e6837133d5105dc5a8b45b0c7583467619364fed9a9967ed1985e.jpg)  
(c) Long-forget.

The visualization shows that, for each output action, the attention is on the corresponding input word. Also, for the output end-of-sentence symbol, it is on the input end-of-sentence symbol. It is consistent with the original work, and the way humans apply compositionality. This indicates that the proposed method may be applicable to other tasks where humans use compositionality.

# 5.2 EMBEDDING VISUALIZATION

We visualize how the new embedding parameters fit in the space with predefined dimensions, and accommodate with previously learned parameters. The visualization of attention maps explains the syntactic information, and we are also interested in semantic information.

We use t-SNE (Maaten & Hinton, 2008) to project high dimensional embeddings to two dimensional space for visualization. Our analysis focuses on semantic embedding, because it reflects how new information is encoded in the model. Since action embedding shares much information with semantic embedding, and syntactic embedding is not supposed to contain new information because grammar does not change over stages, we leave them in Appendix D.

![](images/979e823ee30d6db0ab7426ae7faa165f4ed2ff0215e78bb35a87ae1b33d2d266.jpg)  
(a) Stage 1-25.

![](images/f1e5dfbe4cf05af41f6a160a6da6b06103e9beebaef93f937e0fb4001a81645a.jpg)  
(b) Stage 26-50.

![](images/8b9c5c811e84446347d39202a603a3d7cee93c314a9e217270d6d03ff2cc4619.jpg)  
(c) Stage 51-75.

![](images/eb0c47344ea17ac945bb96a82024add77acc6e9bf3d7bc2ba40e73968a0b4cf9.jpg)  
(d) Stage 76-100.  
Figure 4: Embedding visualization for semantic embeddings. We see two phases. In (1-50), embeddings explore outside space. In (51-100), embeddings squeeze into the explored space.

Figure 4 shows two phases in the continual learning experiment. The first phase is from the first stage to around stage 50, where the new embeddings explore outside space. The second phase is the rest of the stages, where the embeddings squeeze into the explored space, maybe because exploring becomes expensive with the dense population under regularization.

# 6 CONCLUSION

In this paper, we propose an approach to use label prediction continual learning algorithm for sequence-to-sequence continual learning problem by leveraging compositionality. To our knowledge, this is the first work to combine continual learning and compositionality for sequence-to-sequence learning. Experiments show that the proposed method has significantly better results than baseline methods, and it maintains almost more than  $85\%$  accuracy for both transfer learning and catastrophic forgetting prevention up to 100 stages. The results demonstrate that language compositionality helps continual learning of natural language instruction both efficiently and effectively. We hope this work will advance the communication between humans and robots, and make robots more helpful in various tasks.

# REFERENCES

Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: A system for large-scale machine learning. In 12th {USENIX} Symposium on Operating Systems Design and Implementation (\{OSDI\} 16), pp. 265-283, 2016.  
Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars. Expert gate: Lifelong learning with a network of experts. In CVPR, 2017.  
Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. Memory aware synapses: Learning what (not) to forget. In ECCV, 2018.  
Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. Online continual learning with no task boundaries. arxiv1903.08671v2, 2019.  
Andrew Biemiller. Teaching vocabulary: Early, direct, and sequential. American Educator, 2001.  
Philémon Brakel and Stefan Frank. Strong systematicity in sentence processing by simple recurrent networks. In 31th Annual Conference of the Cognitive Science Society (COGSCI-2009), pp. 1599-1604. Cognitive Science Society, 2009.  
Arslan Chaudhry, Marc'Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with a-gem. arXiv preprint arXiv:1812.00420, 2018.  
Noam Chomsky. Syntactic structures. Walter de Gruyter, 1957.  
Mohamed Elhoseiny, Francesca Babiloni, Rahaf Aljundi, Marcus Rohrbach, Manohar Paluri, and Tinne Tuytelaars. Exploring the challenges towards lifelong fact learning. In *Asian Conference on Computer Vision*, pp. 66-84. Springer, 2018.  
Chrisantha Fernando, Dylan Banarse, Charles Blundell, Yori Zwols, David Ha, Andrei A Rusu, Alexander Pritzel, and Daan Wierstra. Pathnet: Evolution channels gradient descent in super neural networks. arXiv preprint arXiv:1701.08734, 2017.  
J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, and et al. Overcoming catastrophic forgetting in neural networks. PNAS, 2017a.  
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, pp. 201611835, 2017c.  
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521-3526, 2017b.  
Markus Kliegl and Wei Xu. More systematic than claimed: Insights on the scan tasks. ICLR Workshop, 2018.  
Brenden Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In International Conference on Machine Learning, pp. 2879-2888, 2018.  
Brenden M Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. arXiv preprint arXiv:1711.00350, 2017.  
Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building machines that learn and think like people. Behavioral and Brain Sciences, 40, 2017.  
Brenden M Lake, Tal Linzen, and Marco Baroni. Human few-shot learning of compositional instructions. arXiv preprint arXiv:1901.04587, 2019.

Sang-Woo Lee, Jin-Hwa Kim, Jaehyun Jun, Jung-Woo Ha, and Byoung-Tak Zhang. Overcoming catastrophic forgetting by incremental moment matching. In Advances in Neural Information Processing Systems, pp. 4652-4662, 2017.  
Yuanpeng Li, Liang Zhao, Jianyu Wang, and Joel Hestness. Compositional generalizatin for primitive substitutions. EMNLP, 2019. URL https://bit.ly/2xloECo.  
Zhizhong Li and Derek Hoiem. Learning without forgetting. In European Conference on Computer Vision, pp. 614-629. Springer, 2016.  
Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. Assessing the ability of lstms to learn syntax-sensitive dependencies. Transactions of the Association for Computational Linguistics, 4:521-535, 2016.  
David Lopez-Paz and MarcAurelio Ranzato. Gradient episodic memory for continual learning. In NIPS, 2017.  
Joao Loula, Marco Baroni, and Brenden M Lake. Rearranging the familiar: Testing compositional generalization in recurrent networks. arXiv preprint arXiv:1807.07545, 2018.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(Nov):2579-2605, 2008.  
James L McClelland, Bruce L McNaughton, and Randall C O'reilly. Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. *Psychological review*, 102(3):419, 1995.  
Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. Psychology of learning and motivation, 24:109-165, 1989.  
Marvin Minsky. Society of mind. Simon and Schuster, 1986.  
Richard Montague. Universal grammar. Theoria, 36(3):373-398, 1970.  
Cuong V Nguyen, Yingzhen Li, Thang D Bui, and Richard E Turner. Variational continual learning. arXiv preprint arXiv:1710.10628, 2017.  
S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert. icarl: Incremental classifier and representation learning. In CVPR, 2017.  
Paul Rodriguez and Janet Wiles. Recurrent neural networks can learn to implement symbol-sensitive counting. In Advances in Neural Information Processing Systems, pp. 87-93, 1998.  
Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.  
Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. Continual learning with deep generative replay. In Advances in Neural Information Processing Systems, pp. 2990-2999, 2017.  
Konstantin Shmelkov, Cordelia Schmid, and Karteek Alahari. Incremental learning of object detectors without catastrophic forgetting. In The IEEE International Conference on Computer Vision (ICCV), 2017.  
Amal Rannen Triki, Rahaf Aljundi, Mathew B Blaschko, and Tinne Tuytelaars. Encoder based lifelong learning. arXiv preprint arXiv:1704.01920, 2017.  
Gail Weiss, Yoav Goldberg, and Eran Yahav. On the practical computational power of finite precision rnns for language recognition. arXiv preprint arXiv:1805.04908, 2018.  
Francis CK Wong and William SY Wang. Generalisation towards combinatorial productivity in language acquisition by simple recurrent networks. In 2007 International Conference on Integration of Knowledge Intensive Multi-Agent Systems, pp. 139-144. IEEE, 2007.

Ju Xu and Zhanxing Zhu. Reinforced continual learning. In Advances in Neural Information Processing Systems, pp. 899-908, 2018.  
Guangyu Robert Yang, Madhura R Joglekar, H Francis Song, William T Newsome, and Xiao-Jing Wang. Task representations in neural networks trained to perform many cognitive tasks. Nature neuroscience, pp. 1, 2019.  
Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 3987-3995. JMLR.org, 2017a.  
Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pp. 3987-3995. PMLR, 06-11 Aug 2017b.
