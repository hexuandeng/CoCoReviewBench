# CAN NEURAL NETWORKS UNDERSTAND LOGICAL ENTAILMENT?

Anonymous authors

Paper under double-blind review

# ABSTRACT

We introduce a new dataset of logical entailments for the purpose of measuring models' ability to capture and exploit the structure of logical expressions against an entailment prediction task. We use this task to compare a series of architectures which are ubiquitous in the sequence-processing literature, in addition to a new model class—PossibleWorldNets—which computes entailment as a "convolution over possible worlds". Results show that convolutional networks present the wrong inductive bias for this class of problems relative to LSTM RNNs, tree-structured neural networks outperform LSTM RNNs due to their enhanced ability to exploit the syntax of logic, and PossibleWorldNets outperform all benchmarks.

# 1 INTRODUCTION

This paper seeks to answer two questions: "Can neural networks understand logical formulae well enough to detect entailment?", and, more generally, "Which architectures are best at inferring, encoding, and relating features in a purely structural sequence-based problem?" In answering these questions, we aim to better understand the inductive biases of popular architectures with regard to structure and abstraction in sequence data. Such understanding would help pave the road to agents and classifiers that reason structurally, in addition to reasoning on the basis of essentially semantic representations. In this paper, we provide a testbed for evaluating some aspects of neural networks' ability to reason structurally and abstractly. We use it to compare a variety of popular network architectures and a new model we introduce, called PossibleWorldNet.

Neural network architectures lie at the heart of a variety of applications. They are practically ubiquitous across vision tasks (LeCun et al., 1995; Krizhevsky et al., 2012; Simonyan & Zisserman, 2014) and natural language understanding, from machine translation (Kalchbrenner & Blunsom, 2013; Sutskever et al., 2014; Bahdanau et al., 2014) to textual entailment (Bowman et al., 2015; Rocktäschel et al., 2015) via sentiment analysis (Socher et al., 2013; Kalchbrenner et al., 2014) and reading comprehension (Hermann et al., 2015; Hill et al., 2015; Rajpurkar et al., 2016). They have been used to synthesise programs (Ling et al., 2016; Parisotto et al., 2016; Devlin et al., 2017) or internalise algorithms (Graves et al., 2016; Grefenstette et al., 2015; Joulin & Mikolov, 2015; Kaiser & Sutskever, 2015; Reed & De Freitas, 2015). They form the basis of reinforcement learning agents capable of playing video games (Mnih et al., 2015), difficult perfect information games (Silver et al., 2016; Tian & Zhu, 2015), and navigating complex environments from raw pixels (Mirowski et al., 2016). An important question in this context is to find the inductive and generalisation properties of different neural architectures, particularly towards the ability to capture structure present in the input, an ability that might be important for many language and reasoning tasks. However, there is little work on studying these inductive biases in isolation by running these models on tasks that are primarily or purely about sequence structure, which we intend to address.

The paper's contribution is three-fold. First, we introduce a new dataset for training and evaluating models. Second, we provide a thorough evaluation of the existing neural models on this dataset. Third, inspired by the semantic (model-theoretic) definition of entailment, we propose a variant of the TreeNet that evaluates the formulas in multiple different "possible worlds", and which significantly outperforms the benchmarks. The structure of this paper is as follows. In Section 2, we introduce the new dataset and describe a generic data generation process for entailment datasets, which offers certain guarantees against the presence of superficial exploitable biases. In Section 3, we describe a series of baseline models used to validate the dataset, benchmarks from which we will

derive our analyses of popular model architectures, and also introduce our new neural model, the PossibleWorldNet. In Section 4, we describe the structure of experiments, from which we obtained the results presented and discussed in Section 5. We offer a brief survey of related work in Section 6, before making concluding remarks in Section 7.

# 2 DATASET CREATION

Formal logics provide a symbolic toolkit for encoding and examining patterns of reasoning. They are structural calculi aiming to codify the norms of correct thought. The meanings of such statements are invariant to what the particular propositions stand for: to understand the entailment  $(p \wedge q) \models q$ , we only need to understand the semantics of—or related syntactic rules governing—a finite set of logical connectives, while  $p$  and  $q$  are meaningless arbitrary symbols selected to stand for distinct propositions. In other words, the problem of determining whether an entailment holds is a purely structural sequence-based problem: to evaluate whether an entailment is true, only the meaning of—or inference rules governing—the connectives is relevant. Everything else only has meaning via its place in the structure specified by an expression. These qualities suggest that detecting logical entailment is an excellent task for measuring the ability of models to capture, understand, or exploit structure. We present in this paper a generic process for generating entailment datasets, explained in detail in Appendix A, for any given logical system. In the specific dataset—generated through this process—presented in this section, we will focus on propositional logic, which is decidable but requires a worst case of  $O(2^n)$  operations (e.g. resolution steps, truth table rows), where  $n$  is the number of unique propositional variables, to verify entailment.

Our dataset<sup>1</sup>  $\mathcal{D}$  is composed of triples of the form  $(A, B, A \vDash B)$ , where  $A$  and  $B$  are formulas of propositional logic, and  $A \vDash B$  is 1 if  $A$  entails  $B$ , and 0 otherwise. For example, the data point  $(p \wedge q, q, 1)$  is positive because  $p \wedge q$  entails  $q$ , whereas  $(q \lor r, r, 0)$  is negative because  $q \lor r$  does not entail  $r$ . Entailment is primarily a semantic notion:  $A$  entails  $B$  if every model in which  $A$  is true is also a model in which  $B$  is true.

We impose various requirements on the dataset, to rule out superficial structural differences between  $\mathcal{D}^+$  and  $\mathcal{D}^-$  that can be easily exploited by "trivial" baselines. We impose the following high level constraints on our data through the generative process, explained in detail in Appendix A: our classes must be balanced, and formulas in positive and negative examples must have the same distribution over length. Furthermore, we attempt to ensure that there are no recognisable differences in the distributions of lexical or syntactic features between the positive and negative examples. It would not be acceptable, for example, if a typical  $B$  formula in a positive entailment  $(A,B,1)$  had more disjunctions than a  $B'$  formula in a negative entailment  $(A',B',0)$ .

If we simply sample formulas  $A$  and  $B$  and evaluate whether  $A \models B$ , there are significant differences between the distributions of formulas for the positive and negative examples, which models can learn to exploit without needing to understand the structure of the problem. To avoid these issues, we use a different approach, that satisfies the above requirements. We sample 4-tuples of formulas  $(A_{1}, B_{1}, A_{2}, B_{2})$  such that:

$$
A _ {1} \vDash B _ {1} \qquad A _ {2} \vDash B _ {2} \qquad A _ {1} \not \vDash B _ {2} \qquad A _ {2} \not \vDash B _ {1}
$$

Here, each of the four formulas appears in one positive entailment and one negative entailment. This way, we minimise crude structural differences between the positive and negative examples. Here is a simple example (although the actual dataset has much longer formulas) of such a 4-tuple of datapoints:

$$
p \vDash p \lor q \qquad \neg p \wedge \neg q \vDash \neg q \qquad p \nVdash \neg q \qquad \neg p \wedge \neg q \nVdash p \lor q
$$

To generate these 4-tuples, we first generate pairs  $(A,B)$  such that  $A\models B$ . (To test if  $A\models B$ , we test whether  $A\land \neg B$  is satisfiable, using minisat (Sorensson & Een, 2005)). Then we search through the set of pairs, looking for pairs of pairs,  $(A_{1},B_{1})$  and  $(A_{2},B_{2})$ , such that  $A_{1}\nVdash B_{2}$  and  $A_{2}\nVdash B_{1}$ . We present, in Appendix A, the full details of this generative process, its constraints and guarantees, and how we used particular baselines to validate the data.

Table 1: Dataset Statistics  

<table><tr><td></td><td>Size</td><td>Mean # Vars</td><td>Mean # Ops</td><td>Mean Length</td><td>Mean 2#Vars</td></tr><tr><td>Train</td><td>100,000</td><td>4.5</td><td>5.3</td><td>11.3</td><td>52.2</td></tr><tr><td>Validate</td><td>5,000</td><td>5.1</td><td>6.8</td><td>13.0</td><td>75.7</td></tr><tr><td>Test (easy)</td><td>5,000</td><td>5.2</td><td>6.9</td><td>13.1</td><td>81.0</td></tr><tr><td>Test (hard)</td><td>5,000</td><td>5.8</td><td>17.4</td><td>31.5</td><td>184.4</td></tr></table>

# 2.1 SPLITTING THE DATASET

We divided the dataset into train, validation, test (easy), and test (hard). We produced train, validation, and test (easy) by generating one large set of 4-tuples, and splitting them into groups of sizes 100000, 5000, and 5000. The difficulty of evaluating an entailment depends on the number of propositional variables and the number of operators in the two formulas. In training, validation, and test (easy), we sample the number of propositional variables uniformly between 1 and 10 (there are 26 propositional variables in total:  $a$  to  $z$ ). In test (hard), we sample uniformly between 5 and 10. Our formula sampling method takes a parameter specifying the desired number of operators in the formula. In training, validation, and test (easy), the number of operators in a formula is sampled uniformly between 1 and 10. In our hard test set, the number of operators in a formula is sampled uniformly between 15 and 20. See Table 1 for detailed statistics of the dataset sections, including the average difficulty (based on a complexity of  $\mathcal{O}(2^{\# \text{Vars}})$ ) of sequents in each fold.

In order to test models' ability to generalise to new unseen formulas, we pruned out cases where formulas seen in validation and test were  $\alpha$ -equivalent (equivalent up to renaming of symbols) to formulas seen in training. So, for example, if it had seen  $p \models (\neg q \wedge p)$  in training, we did not want  $r \models (\neg s \wedge r)$  to appear in either the test or validation sets. To do this, we converted all formulas to de-Bruijn form (see Pierce (2002), Chapter 6), and filtered out formulas in validation and test whose de-Bruijn form was identical to one of those in training. This prevents the system from being able to simply memorise examples it has seen in training.

# 2.2 DATA AUGMENTATION THROUGH SYMBOLIC VOCABULARY PERMUTATION

As discussed above, the logical connectives  $(\vee, \wedge, \ldots)$  are the only elements of the language in each dataset that have consistent implicit semantics across expressions. In this sense, two entailments  $p \wedge q \models q$  and  $a \wedge b \models b$  should ideally be treated as identical by the model. To encourage models to capture this invariance, we add an optional data processing layer during training (not testing) whereby symbols are consistently replaced by other symbols of the same type within individual entailments before being input to the network according to the process described below. This is achieved by randomly sampling a permutation of  $a, \ldots, z$  (the propositional variables used) for every training example, and applying this permutation to the left and right sequents. This process is analogous to augmenting image classification training with random reflections and crops.

# 3 MODELS

In this section, we first describe a couple of baseline models that verify the basic difficulty of the dataset, followed by a description of benchmark models which are commonly used (with some variation) in a variety of problems, and finally by a description of our new model, PossibleWorldNet.

# 3.1 BASELINES

The classes in the dataset are balanced in training, validation, and both test sets, so a random baseline (and a constant, majority-class predicting baseline) will obtain an accuracy of  $50\%$  on the test sets.

We define two neural baselines which, we believe, should not be able to perform competitively on this task, but may do better than random. The first is a linear bag of words (Linear BoW) model which embeds each symbol to a vector, and averages them, to produce a representation of each side

of the sequent. These representations are then passed through a linear layer:

$$
P (A \vDash B) = \sigma \left(W \cdot \operatorname {c o n c a t} (g (A), g (B)) + b\right) \quad \text {w h e r e} \quad g (X) = \frac {1}{| X |} \sum_ {x \in X} \operatorname {e m b e d} (x)
$$

The second is a similar architecture, where the final linear layer is replaced with a multi-layer perceptron (MLP BoW):

$$
P (A \vDash B) = \sigma (\operatorname {M L P} (\operatorname {c o n c a t} (g (A), g (B)))) \quad \text {w h e r e} \quad g (X) = \frac {1}{| X |} \sum_ {x \in X} \operatorname {e m b e d} (x)
$$

In both of these cases, the baselines are expected to have limited performance since they can only capture entailment by modelling the contribution of symbols individually, rather than by modelling structure, since the summation in  $g$  destroys all structural information (including word order). We use these results to provide an indication of the difficulty of the dataset.

# 3.2 BENCHMARKS

We present here a series of benchmark models, not only to serve the purpose of being grounds for comparison for new models tested against this dataset, but also to compare and contrast the performance of fairly ubiquitous model architectures on this purely syntactic problem.

We distinguish two categories of models: encoding models and relational models. Encoding models, with exceptions specified below, jointly learn an encoding function  $f$  and an MLP, such that given a sequent  $A \models B$ , the model expresses

$$
P (A \vDash B) = \sigma (\operatorname {M L P} (\operatorname {c o n c a t} (f (A), f (B))))
$$

In this sense  $f$  produces a representation of each side of the sequent which contains all the information needed for the MLP to decide on entailment. In contrast, relational models will observe the pair of expressions and make a decision, perhaps by traversing both expressions, or by relating substructure of one expression to that of the other. These models express a more general formulation

$$
P (A \vDash B) = \sigma \left(f (A, B)\right).
$$

# 3.2.1 ENCODER BENCHMARKS

The first encoder benchmark implemented is a Deep Convolutional Network Encoder (ConvNet Encoders), akin to architectures described in the convolutional networks for text literature (Kalchbrenner et al., 2014; Zhang et al., 2015; Kim et al., 2016). Here, the encoder function  $f$  is a stack of one dimensional convolutions over sequence symbols embedded by an embedding operation embedSeq, interleaved with max pooling layers every  $k$  layers (which is a model hyperparameter), followed by  $n$  (also a hyperparameter) fully connected layers:

$$
f (X) = \operatorname {M L P} \left(\operatorname {C o n v 1 D} _ {n} (\dots \operatorname {m a x P o o l} \left(\operatorname {C o n v 1 D} _ {k} (\dots \operatorname {C o n v 1 D} _ {1} (\operatorname {e m b e d S e q} (X)) \dots)\right) \dots)\right)
$$

The second and third encoder benchmarks are an LSTM (Hochreiter & Schmidhuber, 1997) encoder network (LSTM Encoders), and its bidirectional LSTM variant (BiDirLSTM Encoders). For the LSTM encoder, we embed the sequence symbols, and run an LSTM RNN over them, ignoring the output until the final state:

$$
f (X) = h _ {\text {f i n a l}} \quad \text {w h e r e} \quad h _ {\text {f i n a l}} = \operatorname {L S T M} (\operatorname {e m b e d S e q} (X))
$$

For the bidirectional variant, two separate LSTM RNNs  $\mathrm{LSTM}^{\leftarrow}$  and  $\mathrm{LSTM}^{\rightarrow}$  are run over the sequence in opposite directions. Their respective final states are concatenated to form a representation of the expression:

$$
\begin{array}{l} f (X) = \operatorname {c o n c a t} \left(h _ {\text {f i n a l}} ^ {\leftarrow}, h _ {\text {f i n a l}} ^ {\rightarrow}\right) \quad \text {w h e r e} \quad h _ {\text {f i n a l}} ^ {\leftarrow} = \operatorname {L S T M} ^ {\leftarrow} (\operatorname {e m b e d S e q} (X)) \\ \text {a n d} \quad h _ {\text {f i n a l}} ^ {\rightarrow} = \operatorname {L S T M} ^ {\rightarrow} (\operatorname {e m b e d S e q} (X)) \\ \end{array}
$$

The benchmarks described thus far do not explicitly condition on structure, even when it is known, as they are designed to traverse a sequence from left to right and model dependencies in the data implicitly. In contrast, we now consider encoder benchmarks which rely on the provision of the

syntactic structure of the sequence they encode, and exploit it to determine the order of composition. This inductive bias, which may be incorrect in certain domains (e.g., where no syntax is defined) or difficult to achieve in domains such as natural language text (where syntactic structure is latent and ambiguous), is easy to achieve for logic (where the syntax is known). The experiments below will seek to demonstrate whether is a helpful inductive architectural bias.

The fourth and fifth encoding benchmarks are (tree) recursive neural networks (Tai et al., 2015; Le & Zuidema, 2015; Zhu et al., 2015; Allamanis et al., 2016), also known as TreeRNNs. These recursively encode the logical expression using the parse structure<sup>3</sup>, where leaf nodes of the tree (propositional variables) are embedded as learnable vectors, and each logical operator then combines one or more of these embedded values to produce a new embedding. For example, the expression  $(\neg a) \lor b$  is parsed as the tree with leaves  $a$  and  $b$ , a unary node  $\neg$  (with input the embedding of  $a$ ), and a binary node  $\lor$  (with inputs the embeddings of  $\neg a$  and  $b$ ). Following Allamanis et al. (2016), the fourth encoding benchmark is a simple TreeRNN (TreeNet Encoders), where each operator 'op' concatenates its inputs to a vector  $x$ , and produces the output

$$
p = \frac {h}{\| h \| _ {2}} \quad \text {w h e r e} \quad h = W _ {1} ^ {\mathrm {o p}} x + W _ {2} ^ {\mathrm {o p}} \sigma (W _ {3} ^ {\mathrm {o p}} x + b _ {3} ^ {\mathrm {o p}}) + b _ {1} ^ {\mathrm {o p}}.
$$

The fifth and final encoding benchmark (TreeLSTM Encoders) is a variant of TreeRNNs which adapts LSTM cell updates. This helps capture long range dependencies and propagate gradient within the tree. Our implementation follows Tai et al. (2015), modified to have per-op parameters as per TreeRNNs (see, also, the work by Le & Zuidema (2015) and Zhu et al. (2015)).

# 3.2.2 RELATIONAL BENCHMARKS

In addition to these encoding benchmarks, we define a pair of relational benchmarks, following Rocktäschel et al. (2015). We will traverse the entire sequent with LSTM RNNs or bidirectional LSTM RNNs but concatenating the left hand side and right hand side sequences into a single sequence separated by a held-out symbol (effectively standing for  $\models$ ). For the LSTM variant (LSTM Traversal), the model is:

$$
P (A \vDash B) = \sigma (\mathrm {M L P} \left(h _ {\text {f i n a l}}\right)) \quad \text {w h e r e} \quad h _ {\text {f i n a l}} = \operatorname {L S T M} (\operatorname {e m b e d S e q} (\operatorname {j o i n} (A, “ \vDash ”, B)))
$$

For the bidirectional case (BiDirLSTM Traversal), the extension is

$$
P (A \vDash B) = \sigma (\mathrm {M L P} (h _ {\text {f i n a l}} ^ {\leftrightarrow})) \quad \text {w h e r e} \quad h _ {\text {f i n a l}} ^ {\leftrightarrow} = \operatorname {c o n c a t} (h _ {\text {f i n a l}} ^ {\leftarrow}, h _ {\text {f i n a l}} ^ {\rightarrow})
$$

$$
\text {w i t h} \quad h _ {\text {f i n a l}} ^ {\leftarrow} = \operatorname {L S T M} ^ {\leftarrow} (\operatorname {e m b e d S e q} (X))
$$

$$
\text {a n d} \quad h _ {\text {f i n a l}} ^ {\rightarrow} = \operatorname {L S T M} ^ {\rightarrow} (\operatorname {e m b e d S e q} (X))
$$

# 3.3 THE POSSIBLEWORLDNET

In this section, we introduce our new model. Inspired by the semantic (model-theoretic) definition of entailment, we propose a variant on TreeNets that evaluates the pair of formulas in different "possible worlds".

Entailment is, first and foremost, a semantic notion. Given a set  $\mathcal{W}$  of worlds,

$$
A \vDash B \text {i f f} w \in \mathcal {W}, s a t (w, A) \text {i m p l i e s} s a t (w, B)
$$

Here  $\text{sat} : \text{World} \times \text{Formula} \to \text{Bool}$  indicates whether a formula is satisfied in a particular world.

We shall first define a variant of  $sat$  that produces integers, and then define another variant that operates on real values. First, define  $sat_2: World \times Formula \to \{0,1\}$ :

$$
s a t _ {2} (w, A) = \mathbb {1} (s a t (w, A))
$$

Using  $sat_2$ , we can redefine entailment as:

$$
A \models B \text {i f f} \forall w \in \mathcal {W} s a t _ {2} (w, A) \leq s a t _ {2} (w, B)
$$

Assume we have a finite set of worlds  $\mathcal{W} = \{w_1, \dots, w_n\}$ ; then we can recast as:

$$
P (A \vDash B) = \prod_ {i = 1} ^ {n} \mathbb {1} \left(s a t _ {2} \left(w _ {i}, A\right) \leq s a t _ {2} \left(w _ {i}, B\right)\right) \tag {1}
$$

We are going to produce a relaxation of Proposition 1 by replacing  $sat_2$  and  $\leq$  with continuous functions. Assume we have a variant of  $sat_2$  that produces vectors of real values:

$$
s a t _ {3}: W o r l d \times F o r m u l a \to \mathbb {R} ^ {d}
$$

Assume we have a function  $f: \mathbb{R}^d \times \mathbb{R}^d \to [0,1]$  that generalises  $\leq$  to vectors of real values. Now we can rewrite as:

$$
P (A \vDash B) = \prod_ {i = 1} ^ {n} f \left(s a t _ {3} \left(w _ {i}, A\right), s a t _ {3} \left(w _ {i}, B\right)\right) \tag {2}
$$

In our neural model,  $f$  is implemented by a simple linear layer using learnable weights  $W_{f}$  and  $b_{f}$ :

$$
f (x, y) = \sigma \left(W _ {f} \cdot \operatorname {c o n c a t} (x, y) + b _ {f}\right)
$$

We use a set of random vectors to represent our worlds  $\{w_{1},\dots,w_{n}\}$ , where  $w_{i}\in \mathbb{R}^{k}$  is a vector of length  $k$  of values drawn uniformly randomly. We implement  $sat_3$  using a simplified TreeNN (see Section 3.2) as described below. Since  $sat_3$  depends on the particular world  $w_{i}$  we are currently evaluating, we add an additional parameter to the TreeNN so that the embedder has access to the current world  $w_{i}$ . We add an additional weight matrix  $W_4^{op}$  so that propositional variables can learn which aspect of the current world to focus on. If the formula is of the form  $op(l,r)$ , where  $op$  is nullary (a propositional variable), unary (e.g., negation), or binary (e.g., conjunction), and  $l$  and  $r$  are the embeddings of the constituents of the expression, then

$$
s a t _ {3} (w _ {i}, o p (l, r)) = \frac {h}{\| h \| _ {2}} \quad \text {w h e r e} \quad h = \left\{ \begin{array}{l l} W _ {4} ^ {o p} w _ {i} & \text {w h e r e o p i s n u l l a r y (l e a f)} \\ W _ {1} ^ {o p} x + b _ {1} ^ {o p} & \text {o t h e r w i s e} \end{array} \right.
$$

where  $x = \mathrm{concat}(l,r)$

To evaluate whether  $A \models B$ , the PossibleWorldNet generates a set of imagined "worlds", and then evaluates  $A$  and  $B$  in each of those worlds. It is a form of "convolution over possible worlds". As we will see in Section 5, the quality of the model increases steadily as we increase the number of imagined worlds.

This architecture was inspired by semantic (model-theoretic) approaches to detecting entailment, but it does not encode any constraint on propositional logic in particular or formal logic in general. The procedure of evaluating sentences in multiple worlds, and combining those evaluations in one product, is just what "entailment" means; so we speculate that an architecture like this should, in principle, be equally applicable to other logics (e.g., intuitionistic logic, modal logics, first-order logic) and also to non-formal entailments in natural language sentences.

Abstracting away from the particular interpretation of these vectors as "worlds", this method generates  $n$  copies of the model with shared weights, one for each vector  $w_{i}$ ; each nullary operator learns a different projection on  $w_{i}$ . It makes predictions via a linear layer combining two representations, and then takes the product of the predictions as the overall prediction.

# 4 EXPERIMENTAL SETUP

For each encoder benchmark architecture, the parameters of the encoders for the left and right hand sides of the sequent are shared. The MLP which performs binary classification to detect entailment based on the expression representations produced by the encoders is model-specific (re-initialised for each model) and jointly trained. Symbol embedding matrices are also model-specific, shared across encoders, and jointly trained.

We implemented all architectures in TensorFlow (Abadi et al., 2016). We optimised all models with Adam (Kingma & Ba, 2014). We grid searched across learning rates in  $[1\mathrm{e} - 5, 1\mathrm{e} - 4, 1\mathrm{e} - 3]$ , minibatch sizes in [64, 128], and trained each model thrice with different random seeds. Per architecture, we grid-searched across specific hyperparameters as follows. We searched across 2 and 3

layer MLPs wherever an MLP existed in a benchmark, and across layer sizes in [32, 64] for MLP hidden layers, embedding sizes, and RNN cell size (where applicable). Additionally for convolutional networks, we searched across a number of convolutional layers in [4, 6, 8], across kernel size in [5, 7, 9], across number of channels in [32, 64], and across pooling interval in [0, 5, 3, 1] (where 0 indicates no pooling). Finally, for all models, we ran them with and without the symbol permutation data augmentation technique described in Section 2.2.

As a result of the grid search, we selected the best model for each architecture against validation results, and record training, validation, and all test accuracies for the associated time step, which we present below.

# 5 RESULTS AND DISCUSSION

Experimental results are shown in Table 2. The test scores of the best performing overall model are indicated in bold. The test scores of the best performing model which does not have privileged access to the syntax or semantics of the logic (i.e. excluding TreeRNN-based models) are italicised. The best benchmark test results are underlined.

Table 2: Propositional Logic Model Accuracy.  

<table><tr><td></td><td>model</td><td>train</td><td>valid</td><td>test (easy)</td><td>test (hard)</td></tr><tr><td rowspan="2">baselines</td><td>Linear BoW</td><td>49.8</td><td>53.1</td><td>52.2</td><td>49.6</td></tr><tr><td>MLP BoW</td><td>62.3</td><td>59.2</td><td>58.2</td><td>52.6</td></tr><tr><td rowspan="7">benchmarks</td><td>ConvNet Encoders</td><td>71.7</td><td>65.9</td><td>66.5</td><td>56.8</td></tr><tr><td>LSTM Encoders</td><td>86.1</td><td>82.3</td><td>81.1</td><td>69.4</td></tr><tr><td>BiDirLSTM Encoders</td><td>75.6</td><td>70.5</td><td>69.5</td><td>63.5</td></tr><tr><td>TreeNet Encoders</td><td>92.5</td><td>91.5</td><td>88.9</td><td>86.0</td></tr><tr><td>TreeLSTM Encoders</td><td>95.8</td><td>94.5</td><td>93.4</td><td>89.4</td></tr><tr><td>LSTM Traversal</td><td>73.8</td><td>71.2</td><td>70.6</td><td>64.7</td></tr><tr><td>BiDirLSTM Traversal</td><td>76.2</td><td>69.1</td><td>69.8</td><td>57.9</td></tr><tr><td>new model</td><td>PossibleWorldNet</td><td>99.7</td><td>99.1</td><td>99.3</td><td>97.3</td></tr></table>

We observe that the baselines are doing better than random (8.2 points above for the easy test set, for the MLP BoW, and 2.6 above random for the hard test set). This indicates that there are some small number of exploitable regularities at the symbolic level in this dataset, but that they do not provide significant information.

The baseline results show that convolution networks and BiDirLSTMs encoders obtain relatively mediocre results compared to other models, as do LSTM and BiDirLSTM Traversal models. LSTM encoders is the best performing model which does not have privileged access to the syntax trees. Their success relative to BiDirLSTMs Encoders could be due to their reduced number of parameters guarding against overfitting, and rendering them easier to optimise, but it is plausible BiDirLSTMs Encoders would perform similarly with a more fine-grained grid search. Both tree-based models take the lead amongst the benchmarks, with the TreeLSTM being the best performing benchmark overall on both test sets. For most models except baselines, the symbol permutation data augmentation yielded 2-3 point increase in accuracy on weaker models (BiDirLSTM encoders and traversals, an convolutional networks) and between 7-15 point increases for the Tree-based models. This indicates that this data augmentation strategy is particularly well fitted for letting structure-aware models capture, at the representational level, the arbitrariness of symbols indicating unbound variables.

Overall, these results show clearly that models that exploit structure in problems where it is provided, unambiguous, and a central feature of the task, outperform models which must implicitly model the structure of sequences. LSTM-based encoders provide robust and competitive results, although bidirectionality is not necessarily always the obvious choice due to optimisation and overfitting problems. Perhaps counter-intuitively, given the results of Rocktäschel et al. (2015), traversal models do not outperform encoding models in this pair-of-sequences traversal problem, indicating that they may be better at capturing the sort of long-range dependencies need to recognise textual entailment better than they are at capturing structure in general.

We conclude, from these benchmark results, that tree structured networks may be a better choice for domains with unambiguous syntax, such as analysing formal languages or programs. For domains such as natural language understanding, both convolutional and recurrent network architectures have had some success, but our experiments indicate that this may be due to the fact that existing tasks favour models which capture representational or semantic regularities, and do not adequately test for structural or syntactic reasoning. In particular, the poor performance of convolutional nets on this task serves as a useful indicator that while they present the right inductive bias for capturing structure in images, where topological proximity usually indicates a joint semantic contribution (pixels close by are likely to contribute to the same "part" of an image, such as an edge or pattern), this inductive bias does not carry over to sequences particularly well (where dependencies may be significantly more sparse, structured, and distant).

The best performing model overall is the PossibleWorldNet, which achieves significantly higher results than the other models, with  $99.3\%$  accuracy on test (easy), and  $97.3\%$  accuracy on test (hard). This is as to be expected, as it has the strongest inductive bias. This inductive bias has two components. First, the model has knowledge of the syntactic structure of the expression, since it is a variant of a TreeNet. Second, inspired by the definition of semantic (model-theoretic) entailment in general, the model evaluates the pair of formulas in lots of different situations ("possible worlds") and combines the various results together in a product<sup>5</sup>.

The quality of the PossibleWorldNet depends directly on the number of "possible worlds" it considers (see Figure 1). As we increase the number of possible worlds, the validation error rate goes down steadily. Note that the data-efficiency also increases as we increase the number of worlds. This is because adding worlds to the model does not increase the number of model parameters—it just increases the number of different "possibilities" that are considered.

![](images/268a823a842eaa05e34326da90b86aa1b3009d3cfe2226a38b6f2228ca9ef94d.jpg)  
Figure 1: The quality of the PossibleWorldNet as we vary the number of possible worlds

In propositional logic, of course, if we are allowed to generate every single truth-value assignment, then it is trivial to detect entailment by checking each one. In our dataset, with 26 propositional variables, there are  $2^{26}$  possible truth-value assignments. The PossibleWorldNet considers at most 256 different worlds, which is only  $0.0003\%$  of the total set. What is surprising is that it is able to achieve this level of accuracy while only considering a small fraction of the total number of possible worlds. We speculate that the model is able to do this by collapsing lots of broadly similar worlds into one overarching world.

# 6 RELATED WORK

Zaremba et al. (2014) show how a neural architecture can be used to optimise matrix expressions. They generate all expressions up to a certain depth, group them into equivalence classes, and train

a recursive neural network classifier to detect whether two expressions are in the same equivalence class. They use a recursive neural network (Socher et al., 2012) to guide the search for an optimised equivalent expression. There are two major differences between this work and ours. First, the classifier is predicting whether two matrix expressions (e.g.  $A$  and  $(A^T)^T$ ) compute the same values; this is an equivalence relation, while entailment is a partial order. Second, their dataset consists of matrix expressions containing at most one variable, while our formulas contain many variables.

Allamanis et al. (2016) use a recursive neural network to learn whether two expressions are equivalent. They tested on two datasets: propositional logic and polynomials. There are two main differences between their approach and ours. First, we consider entailment while they consider equivalence; equivalence is a symmetric relation, while entailment is not symmetric. Second, we consider entailment as a relational classification problem: given a pair of expressions  $A$  and  $B$ , predict whether  $A$  entails  $B$ . In their paper, by contrast, they generate a set of  $k$  equivalence-classes of formulas with the same truth-conditions, and ask the network to predict which of these  $k$  classes a single formula falls into. Their task is more specific: their network is unable to classify a formula from a new equivalence class that has not been seen during training.

Recognizing textual entailment (RTE) between natural language sentences is a central task in natural language processing. (See Dagan et al. (2006); for a recent dataset, see Bowman et al. (2015)). Some approaches (e.g., Wang & Jiang (2015) and Rocktäschel et al. (2015)) use LSTMs with attention, while others (e.g., Yin et al. (2015)) use a convolutional neural network with attention. Of course, recognizing entailment between natural language sentences is a very different task from recognizing entailment between logical formulas. Evaluating an entailment between natural language sentences requires understanding the meaning of the non-logical terms in the sentence. For example, the inference from "An ice skating rink placed outdoors is full of people" to "A lot of people are in an ice skating park" requires knowing the non-logical semantic information that an outdoors ice skating rink is also an ice skating park.

Current neural models do not always understand the structure of the sentences they are evaluating. In Bowman et al. (2015), all the neural models they considered wrongly claimed that "A man wearing padded arm protection is being bitten by a German shepherd dog" entails "A man bit a dog". We believe that isolating the purely structural sub-problem will be useful because only networks that can reliably predict entailment in a purely formal setting, such as propositional (or first-order) logic, will be capable of getting these sorts of examples consistently correct.

# 7 CONCLUSION

In this paper, we have introduced a new process for generating datasets for the purpose of recognising logical entailment. This was used to compare benchmarks and a new model on a task which is primarily about understanding and exploiting structure. We have established two clear results on the basis of this task. First, and perhaps most intuitively, architectures which make explicit use of structure will perform significantly better than those which must implicitly capture it. Second, the best model is the one that has a strong architectural bias towards capturing the possible world semantics of entailment. In addition to these two points, experimental results also shed some light on the relative abilities of implicit structure models—namely LSTM and Convolution network-based architectures—to capture structure, showing that convolutional networks may not present the right inductive bias to capture and exploit the heterogeneous and deeply structured syntax in certain sequence-based problems, both for formal and natural languages.

This conclusion is to be expected: the most successful models are those with the most prior knowledge about the generic structure of the task at hand. But our dataset throws new light on this unsurprising thought, by providing a new data-point on which to evaluate neural models' ability to understand structural sequence problems. Logical entailment, unlike textual entailment, depends only on the meaning of the logical operators, and of the place particular arbitrarily-named variables hold within a structure. Here, we have a task in which a network's understanding of structure can be disentangled from its understanding of the meaning of words.

# REFERENCES

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467, 2016.  
Miltiadis Allamanis, Pankajan Chanthirasegaran, Pushmeet Kohli, and Charles Sutton. Learning continuous semantic representations of symbolic expressions. arXiv preprint arXiv:1611.01423, 2016.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.  
Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. A large annotated corpus for learning natural language inference. arXiv preprint arXiv:1508.05326, 2015.  
Ido Dagan, Oren Glickman, and Bernardo Magnini. The Pascal recognising textual entailment challenge. In Machine learning challenges. evaluating predictive uncertainty, visual object classification, and recognising textual entailment, pp. 177-190. Springer, 2006.  
Jacob Devlin, Jonathan Uesato, Surya Bhupatiraju, Rishabh Singh, Abdel-rahman Mohamed, and Pushmeet Kohli. Robustfill: Neural program learning under noisy i/o. arXiv preprint arXiv:1703.07469, 2017.  
Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, et al. Hybrid computing using a neural network with dynamic external memory. Nature, 538 (7626):471-476, 2016.  
Edward Grefenstette, Karl Moritz Hermann, Mustafa Suleyman, and Phil Blunsom. Learning to transduce with unbounded memory. In Advances in Neural Information Processing Systems, pp. 1828-1836, 2015.  
Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems, pp. 1693-1701, 2015.  
Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston. The goldilocks principle: Reading children's books with explicit memory representations. arXiv preprint arXiv:1511.02301, 2015.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Armand Joulin and Tomas Mikolov. Inferring algorithmic patterns with stack-augmented recurrent nets. In Advances in neural information processing systems, pp. 190-198, 2015.  
Lukasz Kaiser and Ilya Sutskever. Neural gpus learn algorithms. arXiv preprint arXiv:1511.08228, 2015.  
Nal Kalchbrenner and Phil Blunsom. Recurrent continuous translation models. In EMNLP, volume 3, pp. 413, 2013.  
Nal Kalchbrenner, Edward Grefenstette, and Phil Blunsom. A convolutional neural network for modelling sentences. arXiv preprint arXiv:1404.2188, 2014.  
Yoon Kim, Yacine Jernite, David Sontag, and Alexander M Rush. Character-aware neural language models. In AAAI, pp. 2741-2749, 2016.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.

Phong Le and Willem Zuidema. Compositional distributional semantics with long short term memory. arXiv preprint arXiv:1503.02510, 2015.  
Yann LeCun, Yoshua Bengio, et al. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 3361(10):1995, 1995.  
Wang Ling, Edward Grefenstette, Karl Moritz Hermann, Tomáš Kočisky, Andrew Senior, Fumin Wang, and Phil Blunsom. Latent predictor networks for code generation. arXiv preprint arXiv:1603.06744, 2016.  
Piotr Mirowski, Razvan Pascanu, Fabio Viola, Hubert Soyer, Andy Ballard, Andrea Banino, Misha Denil, Ross Goroshin, Laurent Sifre, Koray Kavukcuoglu, et al. Learning to navigate in complex environments. arXiv preprint arXiv:1611.03673, 2016.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.  
Emilio Parisotto, Abdel-rahman Mohamed, Rishabh Singh, Lihong Li, Dengyong Zhou, and Pushmeet Kohli. Neuro-symbolic program synthesis. arXiv preprint arXiv:1611.01855, 2016.  
Benjamin C Pierce. Types and programming languages. MIT press, 2002.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250, 2016.  
Scott Reed and Nando De Freitas. Neural programmer-interpreters. arXiv preprint arXiv:1511.06279, 2015.  
Tim Roktäschel, Edward Grefenstette, Karl Moritz Hermann, Tomáš Kocisky, and Phil Blunsom. Reasoning about entailment with neural attention. arXiv preprint arXiv:1509.06664, 2015.  
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484-489, 2016.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Richard Socher, Brody Huval, Christopher D Manning, and Andrew Y Ng. Semantic compositionality through recursive matrix-vector spaces. In Proceedings of the 2012 joint conference on empirical methods in natural language processing and computational natural language learning, pp. 1201-1211. Association for Computational Linguistics, 2012.  
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pp. 1631-1642, 2013.  
Niklas Sorensson and Niklas Een. Minisat v1. 13-a sat solver with conflict-clause minimization. SAT, 2005(53):1-2, 2005.  
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pp. 3104-3112, 2014.  
Kai Sheng Tai, Richard Socher, and Christopher D Manning. Improved semantic representations from tree-structured long short-term memory networks. arXiv preprint arXiv:1503.00075, 2015.  
Yuandong Tian and Yan Zhu. Better computer go player with neural network and long-term prediction. arXiv preprint arXiv:1511.06410, 2015.  
Shuohang Wang and Jing Jiang. Learning natural language inference with LSTM. arXiv preprint arXiv:1512.08849, 2015.

Wenpeng Yin, Hinrich Schütze, Bing Xiang, and Bowen Zhou. Abcnn: Attention-based convolutional neural network for modeling sentence pairs. arXiv preprint arXiv:1512.05193, 2015.

Wojciech Zaremba, Karol Kurach, and Rob Fergus. Learning to discover efficient mathematical identities. In Advances in Neural Information Processing Systems, pp. 1278-1286, 2014.

Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In Advances in neural information processing systems, pp. 649-657, 2015.

Xiaodan Zhu, Parinaz Sobihani, and Hongyu Guo. Long short-term memory over recursive structures. In International Conference on Machine Learning, pp. 1604-1612, 2015.
