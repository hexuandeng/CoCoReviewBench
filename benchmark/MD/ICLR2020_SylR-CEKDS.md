# MODELING QUESTION ASKING USING NEURAL PROGRAM GENERATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

People ask questions that are far richer, more informative, and more creative than current AI systems. We propose a neural program generation framework for modeling human question asking, which represents questions as formal programs and generates programs with an encoder-decoder based deep neural network. From extensive experiments using an information-search game, we show that our method can ask optimal questions in synthetic settings, and predict which questions humans are likely to ask in unconstrained settings. We also propose a novel grammar-based question generation framework trained with reinforcement learning, which is able to generate creative questions without supervised data.

# 1 INTRODUCTION

People can ask rich, creative questions to learn efficiently about their environment. Question asking is central to human learning yet it is a tremendous challenge for computational models. There is always an infinite set of possible questions that one can ask, leading to challenges both in representing the space of questions and in searching for the right question to ask.

Machine learning has been used to address aspects of this challenge. Traditional methods have used heuristic rules designed by humans (Heilman and Smith, 2010; Chali and Hasan, 2015), which are usually restricted to a specific domain. Recently, neural network approaches have also been proposed, including retrieval methods which select the best question from past experience (Mostafazadeh et al., 2016) and encoder-decoder frameworks which map visual or linguistic inputs to questions (Serban et al., 2016; Mostafazadeh et al., 2016; Yuan et al., 2017; Yao et al., 2018). While effective in some settings, these approaches are heavily data-driven, limiting the diversity of generated questions and requiring large training sets for different goals and contexts. There is still a large gap between how people and machines ask questions.

Recent work has aimed to narrow this gap by taking inspiration from cognitive science. For instance, Lee et al. (2018) incorporates aspects of "theory of mind" (Premack and Woodruff, 1978) in question asking by simulating potential answers to the questions, but the approach relies on imperfect agents for natural language understanding which may lead to error propagation. Related to our approach, Rothe et al. (2017) proposed a powerful question-asking framework by modeling questions as symbolic programs, but their algorithm relies on hand-designed program features and requires expensive calculations to ask questions.

We use "neural program generation" to bridge symbolic program generation and deep neural networks, bringing together some of the best qualities of both approaches. Symbolic programs provide a compositional "language of thought" (Fodor, 1975) for creatively synthesizing which questions to ask, allowing the model to construct new ideas based on familiar building blocks. Compared to natural language, programs are precise in their semantics, have clearer internal structure, and require a much smaller vocabulary, making them an attractive representation for question answering systems as well (Johnson et al., 2017; Mao et al., 2019). Deep neural networks allow for rapid question-synthesis using encoder-decoder modeling, eliminating the need for the expensive symbolic search and feature evaluations in Rothe et al. (2017). Together, the questions can be synthesized quickly and evaluated formally for quality (e.g. the expected information gain), which as we show can be used to train question asking systems using reinforcement learning.

In this paper, we develop a neural program generation model for asking questions in an information-search game similar to "Battleship" used in previous work (Gureckis and Markant, 2009; Rothe et al.,

![](images/1bb018905fb157d780eba35b65cc26e94fba1a4787f16e08e668733da76a91eb.jpg)  
groundtruth board

![](images/181da0ded7fa2804f45a4560de8e10eae1c63110585371e7058052f8c75780fe.jpg)  
partly revealed board  
Figure 1: The Battleship task. Blue, red, and purple tiles are ships, dark gray tiles are water, and light gray tiles are hidden. The agent can see a partly revealed board, and should ask a question to more seek information about the hidden board. Example questions and their program format counterparts are shown on the right. We recommend viewing this figure in color.

How long is the red ship?

(size Red)

Is purple ship horizontal?

$\left( { =  = \text{ (orient Purple ) }H}\right)$

Do all three ships have the same size?

$= = =$  （map（x（sizex））（setAllShips)）

example questions

2018; 2017). The model uses a convolutional encoder to represent the game state, and a Transformer decoder (Vaswani et al., 2017) for generating questions. Building on the work of Rothe et al. (2017), the model uses a grammar-enhanced question asking framework, such that questions as programs are formed through derivation using a context free grammar. Importantly, we show that the model can be trained from human demonstrations of good questions using supervised learning, or without demonstrations of good questions using reinforcement learning. We evaluate the model on several aspects of human question asking, including reasoning about optimal questions in synthetic scenarios, density estimation based on free-form question asking, and creative generation of genuinely new questions.

To summarize, our paper makes three main contributions: 1) We propose a neural network for modeling human question-asking behavior, 2) We propose a novel reinforcement learning framework for generating creative human-like questions by exploiting the power of programs, and 3) We evaluate our methods extensively through three different experiments exploring different qualities of human question asking.

# 2 RELATED WORK

Question generation has attracted attention from the machine learning community. Early research mostly explored rule-based methods which strongly depend on human-designed rules (Heilman and Smith, 2010; Chali and Hasan, 2015). Recent methods for question generation adopt deep neural networks, especially using the encoder-decoder framework, and can generate questions without hand-crafted rules. These methods are mostly data-driven, which use pattern recognition to map inputs to questions. Researchers have worked on generating questions from different types of inputs such as knowledge base facts (Serban et al., 2016), pictures (Yuan et al., 2017), and text for reading comprehension (Mostafazadeh et al., 2016; Yao et al., 2018). However aspects of human question-asking remain beyond reach, including the goal-directed and flexible qualities that people demonstrate when asking new questions.

Several recent papers draw inspiration from cognitive science to generate more human-like, goal-oriented questions. Research from Rothe et al. (2017) and Lee et al. (2018) generate questions by sampling from a candidate set based on goal-oriented metrics. This paper extends the work of Rothe et al. (2017) to overcome the limitation of the candidate set, and generate creative, goal-oriented programs with neural networks.

Our work also builds on neural network approaches to program synthesis. Those approaches often draw inspiration from computer architecture, using neural networks to simulate stacks, memory, and controllers in differentiable form (Reed and De Freitas, 2016; Graves et al., 2014). Other models incorporate Deep Reinforcement Learning (DRL) to optimize the generated programs in a goal oriented environment, such as generating SQL queries which can correctly perform a specific database processing task (Zhong et al., 2018). Recent work has also proposed ways to incorporate explicit grammar information into the program synthesis process. Yin and Neubig (2017) design a special module to capture the grammar information as a prior, which can be used during generation. Bunel et al. (2018) uses DRL to explicitly encourage the generation of semantically correct programs. Our work differs from these in two aspects. First, our goal is to generate informative human-like questions instead of simply correct programs. Second, we more deeply integrate grammar information in our framework, which directly generates programs based on the grammar.

![](images/d4ff27008806d88e16ea4c91b47c437761dfe87ec6b5d81f78a52923a592ae9f.jpg)  
Figure 2: Neural program generation. Figure (a) shows the network architecture. The board is represented as a grid of one-shot vectors and is embedded with a convolutional neural network. The board embedding and a sequence of symbols are inputted to a Transformer decoder (Vaswani et al., 2017) to generate output vectors (details in section 4.2). PE means positional embeddings, and WE means word embeddings. (b) shows the derivation steps for program “(> (size Blue) 3)” using CFG. Non-terminals are shown as bold-faced, and terminals are shown in italic. The production rules used are shown next to each arrow.

# 3 BATTLESHIP TASK

In this paper, we work with a task used in previous work for studying human information search (Gureckis and Markant, 2009) as well as question asking (Rothe et al., 2018). The task is based on an information search game called "Battleship", in which a player aims to resolve the hidden layout of the game board based on the revealed information (Figure 1). There are three ships with different colors (blue, red, and purple) placed on a game board consisting of  $6 \times 6$  grid of tiles. Each ship can be either horizontal or vertical, and takes 2, 3 or 4 tiles long. All tiles are initially turned over (light grey in Figure 1), and the player can flip one tile at a time to reveal an underlying color (either a ship color, or dark grey for water). The goal of the player is to determine the configuration of the ships (positions, sizes, orientations) in the least number of flips.

In the modified version of this task studied in previous work (Rothe et al., 2018; 2017), the player is presented with a partly revealed game board, and is required to ask a natural language question to gain information about the underlying configuration. As shown in Figure 1, the player can only see the partly revealed board, and might ask questions such as "How long is the red ship?" In this paper, we present this task to our computational models, and ask the models to generate questions about the game board.

Rothe et al. (2017) designed a powerful context free grammar (CFG) to describe the questions in the Battleship domain. The grammar represents questions in a LISP program-like format, which consists of a set of primitives (like numbers, colors, etc.) and a set of functions over primitives (like arithmetic operators, comparison operators, and other functions related to the game board). Another research by (Rothe et al., 2018) shows that it captures the full range of questions that people asked in an extensive dataset, mainly because the majority of this grammar is general functions which make it flexible enough. The grammar is able to generate an infinite set of other possible questions beyond collected human questions, capturing key notions of compositionality and computability. Figure 1 provides some examples of produced programs. The full grammar is provided in Appendix B.

# 4 NEURAL PROGRAM GENERATION FRAMEWORK

This section introduces our approach to program generation with neural networks and Deep Reinforcement Learning (DRL). We will first introduce the neural architecture, and then explain how it works with different training and inference strategies.

# 4.1 NEURAL NETWORK ARCHITECTURE

The neural network we use is shown in Figure 2(a). It includes a Convolutional Neural Network (CNN) for encoding the input board, and a Transformer (Vaswani et al., 2017) decoder for estimating the symbol distribution or selecting actions in different settings.

Encoder. The game board  $x \in \{0,1\}^{6 \times 6 \times 5}$  is a 6x6 grid with five channels, one for each tile color, with the color encoded as a one-hot vector in each grid location. A simple CNN with one layer of filters is used to encode the board. Intuitively, many questions are related to specific positions, thus the position information should be recoverable from the encoding. On the other hand, some features of the board are translation-invariant, such as whether a ship is blocked by another ship. In order to capture the position-sensitive information as well as the translation-invariant patterns, three convolution operations with different filter sizes ( $1 \times 1$ ,  $3 \times 3$ , and  $5 \times 5$ ) are performed in parallel on the same input. The inputs are padded accordingly to make sure the feature maps have the same width and height. Then three feature maps are concatenated together along the dimension of output channels, and passed through a linear projection.

Formally, the outputs of the convolutions  $\mathbf{c}$  can be obtained by

$$
\mathbf {c} = \operatorname {R e L U} \left(\left[ \operatorname {C o n v} _ {1} (x); \operatorname {C o n v} _ {3} (x); \operatorname {C o n v} _ {5} (x) \right]\right) \tag {1}
$$

where  $\mathrm{Conv}_k$  denotes a convolution operation on a  $k\times k$  filter,  $\mathrm{ReLU}(\cdot)$  means applying a ReLU activation, and  $[A;B]$  means the concatenation of matrices  $A$  and  $B$ . Then  $\mathbf{c}\in \mathbb{R}^{6\times 6\times 3C_{out}}$  is projected to the encoder output  $\mathbf{e}\in \mathbb{R}^{6\times 6\times M}$  by matrix  $W_{o}^{e}\in \mathbb{R}^{3C_{out},M}$ , where  $C_{out}$  is the number of out channels of each convolution, and  $M$  is the length of encoded vectors.

Decoder. We use the decoder from the Transformer model (Vaswani et al., 2017). With an input sequence of length  $L$ , the decoder computes the hidden states through several stacked Decoder Attention Layers. Each layer is composed by three sub-layers, a self-attention module, an attention over the encoded board, and a fully connected feed-forward network. Residual connections are employed around each sub-layer, followed by a layer normalization (Ba et al., 2016). After  $N$  layers of attention modules, a final output layer transforms the hidden states to the output vectors  $\mathbf{y}_i \in \mathbb{R}^{N_o}$  at every position  $i$  from 1 to  $L$ , where  $N_{o}$  is the output size. As shown later, the outputs can be interpreted differently in different settings.

Given the output from encoder  $\mathbf{e}$ , and the hidden representation  $\mathbf{h}^{n - 1}$  from Decoder Attention Layer  $n - 1$ , each layer computes the hidden representation as

$$
\mathbf {g} ^ {n} = \operatorname {L N} (\operatorname {S e l f - A T T} (\mathbf {h} ^ {n - 1}) + \mathbf {h} ^ {n - 1})
$$

$$
\mathbf {v} ^ {n} = \operatorname {L N} \left(\operatorname {A T T} \left(\mathbf {g} ^ {n}, \mathbf {e}\right) + \mathbf {g} ^ {n}\right) \tag {2}
$$

$$
\mathbf {h} ^ {n} = \operatorname {L N} \left(\operatorname {F C} \left(\mathbf {v} ^ {n}\right) + \mathbf {v} ^ {n}\right)
$$

where  $\mathrm{LN}(\cdot)$  means layer normalization (Ba et al., 2016),  $\mathrm{FC}(\cdot)$  is a fully connected layer,  $\mathrm{ATT}(\cdot)$  and Self-ATT  $(\cdot)$  are multi-head attention mechanisms, which computes the attention over the output of encoder  $\mathbf{e}$ , and the attention over the input  $\mathbf{h}^{n - 1}$  itself, respectively. They are defined as follows

$$
\operatorname {A T T} \left(\mathbf {g} ^ {n}, \mathbf {e}\right) = \operatorname {M u l t i - A T T} \left(\mathbf {g} ^ {n}, \mathbf {e}, \mathbf {e}\right)
$$

$$
\operatorname {S e l f - A T T} \left(\mathbf {h} ^ {n - 1}\right) = \operatorname {M u l t i - A T T} \left(\mathbf {h} ^ {n - 1}, \mathbf {h} ^ {n - 1}, \mathbf {h} ^ {n - 1}\right) \tag {3}
$$

Multi-ATT( $\cdot$ ) is the multi-head attention mechanism described in the paper by Vaswani et al. (2017), which is a concatenation of multiple standard attention mechanisms with inputs projected using different matrices. A multi-head attention with  $n$  heads is defined as

$$
\begin{array}{l} \text {M u l t i - A T T} (Q, K, V) = W ^ {o} [ \text {A t t e n t i o n} (W _ {1} ^ {Q} Q, W _ {1} ^ {K} K, W _ {1} ^ {V} V); \dots ; \\ \left. \operatorname {A t t e n t i o n} \left(W _ {n} ^ {Q} Q, W _ {n} ^ {K} K, W _ {n} ^ {V} V\right) \right] \\ \end{array}
$$

where

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right) V \tag {5}
$$

is the scaled dot-product attention operation.  $Q, K, V$  are a set of vectors called queries, keys, and values, respectively, and  $d_{k}$  is the dimension of queries and keys.

After  $N$  layers, we apply a linear projection and a softmax activation to  $\mathbf{h}^N$  to get the output vectors  $\mathbf{y}_1,\dots ,\mathbf{y}_L$ .

# 4.2 TRAINING AND INFERENCE

The model is compatible with both supervised and reinforcement training.

![](images/47f2e16ccca59eb423f1002f650de27858851acfa3aec5f533e7d2352d92cacc.jpg)

![](images/590e78713c5458a14397d480be7143fdabe414bb7f3fd73c7a4dc4b496df2667.jpg)

![](images/614529e184b5b36d84891001911b1ed5a9888060bbf3b4fc87bc925042533968.jpg)

![](images/4ad91e64ca95893a04dcb9c4f414e839310147b616fee449ed5762dacf54bce6.jpg)

![](images/14b05e813fa5eb629f97eb3fe9d58cdde9e85dac05f591049c10a2c9c35995e5.jpg)  
Figure 3: Design of the 3 tasks in experiment 1. The goal of task (a) is to find the color which has the least number of visible tiles; the goal of task (b) to find the location and color of the missing tile; (c) is the compositionality task with 5 questions as known question types, and another one (in dotted box) as held out question type. The format of generated question is shown alongside the title of each task, where X, Y and Z are variables. The accuracy of supervised model for task (a) and (b) are given below each task.

![](images/6188d50fd3dd156050458037b14fe44d89dbdf4145a4ae3814801bfb1e537af7.jpg)

![](images/7cdfb5e4c18776836dd43c1d6c4cdea7384467c5b87f59ae85f02b5e4d6f2649.jpg)

![](images/6a85789f681b2af46ebfb96170bc4053da6f1629d19677dfb9b5b3314bc91fed.jpg)

Supervised training. In the supervised setting, the goal is to model the distribution of questions present in the training set. Each output  $\mathbf{y}_i \in \mathbb{R}^{N_o}$  is a symbol at position  $i$  in the program, where  $N_o$  is the number of different symbols in the grammar. The model is trained with symbol-level cross entropy loss, and can be used to calculate the log-likelihood of a given sequence, or to generate a question symbol-by-symbol from left to right. Generation works as follows. Suppose at step  $t$ , a sequence of length  $t$  along with the encoded board is presented to the decoder. The model predicts the vector  $\mathbf{y}_t$  which represents the probability of each symbol to be chosen as next. Then we sample a symbol at location  $t + 1$  and execute the decoder again with the new sequence, until an <eos> symbol is generated or maximum length is reached.

Sequence-based RL. The framework can be adapted to generate a sequence of symbols without stepwise supervision, such that reward is provided only after the entire question is generated.

Grammar-enhanced RL. Finally, the framework can be used with a novel grammar-enhanced RL training procedure. Figure 2(b) illustrates the process of generating a program from the context-free grammar specified in Rothe et al. (2017). Beginning from the start symbol “ $\mathbb{A}$ ”, at each step a production rule is chosen and applied to one of the non-terminals in the current string. The choice of rule is modeled as a Markov Decision Process, and we solve it with DRL. Each state is a partially derived string passed to the decoder, and we use the first output  $\mathbf{y}_1 \in \mathbb{R}^{N_o}$  to represent the probability of selecting each production rule from all possible  $N_o$  rules. After the rule is applied, the new string is passed back into the decoder, repeating until only terminals are contained in the sequence. We adopt the leftmost derivation here to avoid the ambiguity of parsing order, so at each step the left-most non-terminal will be replaced.

# 5 EXPERIMENTS

# 5.1 REASONING IN SYNTHETIC SETTINGS

In the first experiment, we designed three tasks to evaluate whether the model can learn simple rules and reasoning strategies. These tasks include counting the number of visible ship tiles, locating a missing ship tile, and generalizing both strategies to unseen scenario types using compositionality. Figure 3 illustrates the three tasks we designed in this experiment by providing some examples of each task.

# 5.1.1 TASK DESCRIPTIONS

- Counting task. Models must select the ship color with the least number of visible tiles on the board. Each board has a unique answer, and models respond by generating a program "(topleft (coloredTiles X))" where X is a ship color. 4000 examples are used for training, and another 1000 examples are used for testing.

Table 1: Accuracy (%) on the compositionality task using different numbers of training examples from the held out question type.  

<table><tr><td># of training examples</td><td>0</td><td>10</td><td>50</td><td>100</td><td>200</td><td>400</td><td>800</td></tr><tr><td>Acc. on held out question type</td><td>0.0</td><td>2.0</td><td>39.0</td><td>69.5</td><td>81.0</td><td>92.0</td><td>96.0</td></tr><tr><td>Acc. on known question types</td><td>96.6</td><td>97.3</td><td>97.1</td><td>96.0</td><td>96.3</td><td>97.8</td><td>96.1</td></tr><tr><td>Acc. classify on held out question type</td><td>33.0</td><td>37.0</td><td>49.0</td><td>75.5</td><td>88.0</td><td>94.0</td><td>99.5</td></tr></table>

- Missing tile task. Models must select the ship that is missing a tile and identify which tile is missing. All ships are completely revealed except one, which is missing exactly one tile. Models respond by generating "  $(==$  (color Y) X) " where X is a color and Y is a location on the board. The number of training and test examples are the same as the counting task.  
- Compositionality task. Models must combine both of the above strategies to find the missing tile of the ship with the least visible tiles. Outputs are produced as "(Z (coloredTiles X))" where X is a color and Z is either topleft or bottomright. Each board has a unique answer.

This task further evaluates compositionality by withholding question types from training. With three values for  $\mathbf{X}$  and two for  $\mathbb{Z}$ , there are six possible question types and one is picked as the "held out" type. The other five "known" question types have 800 training examples. For the held out question type, the number of training examples is varied from 0 to 800, to test how much data is needed for generalization. Another 200 new boards of each question type is used for evaluation.

More information about the model hyperparameters and training procedures are provided in Appendix A.1.

# 5.1.2 RESULTS AND DISCUSSION

We train our model in a fully supervised fashion. Accuracy for the counting and missing tile tasks is shown in Figure 3. The full neural program generation model shows strong reasoning abilities, achieving high accuracy for both the counting and missing tile tasks, respectively. We also perform ablation analysis of the encoder filters of the model, and provide the results in Appendix C.

The results for the compositionality task are summarized in Table 1. When no training data regarding the held out question type is provided, the model cannot generalize to situations systematically different from training data, exactly as pointed out in previous work on the compositional skills of encoder-decoder models (Lake and Baroni, 2018). However, when the number of additional training data increases, the model quickly incorporates the new question type while maintaining high accuracy on the familiar question tasks. On the last row of Table 1, we compare our model with another version where the decoder is replaced by two linear transformation operations which directly classify the ship type and location (details in Appendix A.1). This model has  $33.0\%$  transfer accuracy on compositional scenarios never seen during training. This suggests that the model has the potential to generalize to unseen scenarios if the task can be decomposed to subtasks and combined together.

# 5.2 ESTIMATING THE DISTRIBUTION OF HUMAN QUESTIONS

In this experiment, we examine if the neural network has the capability of capturing the distribution of human questions as a conditioned language model.

# 5.2.1 DATA COLLECTION

To train the model, we need to construct a training set of paired game boards and questions. Instead of laboriously collecting a large number of real human questions, and translating them into programs by hand, we construct the dataset by sampling from a previous computational model of human question asking (Rothe et al., 2017). More precisely, we randomly generate a large number of game boards and sample  $K$  questions given each board. For generating the boards, we first uniformly sample the configuration of three ships, and randomly cover arbitrary number of tiles, with the restriction that at least one ship tile is observable. Next we randomly sample  $K$  programs for each board with importance sampling based on the cognitive model proposed by Rothe et al. (2017), which models

Table 2: Log-likelihood (LL) on two evaluation set of different version models.  

<table><tr><td>Model</td><td>LL on sampled data</td><td>LL on human data</td></tr><tr><td>Full model</td><td>-3.197</td><td>-7.124</td></tr><tr><td>no pretrain</td><td>-3.217</td><td>-7.280</td></tr><tr><td>LSTM decoder</td><td>-3.222</td><td>-9.013</td></tr><tr><td>MLP encoder</td><td>-3.385</td><td>-7.475</td></tr><tr><td>decoder only</td><td>-3.401</td><td>-8.434</td></tr></table>

Table 3: Log-likelihood (LL) on different split of sampled evaluation set based on the uncertainty of the board. More comparisons are provided in Appendix B Table 8.  

<table><tr><td>Model</td><td>LL on low entropy</td><td>LL on mid entropy</td><td>LL on high entropy</td></tr><tr><td>Full model</td><td>-2.990</td><td>-3.190</td><td>-3.414</td></tr><tr><td>decoder only</td><td>-3.312</td><td>-3.397</td><td>-3.494</td></tr></table>

the probability of a question under a given context as

$$
p (q; \theta) = - \exp (\varepsilon (q; \theta)) / Z \tag {6}
$$

where  $\varepsilon (\cdot)$  is a parameterized energy function for estimating the likelihood of a question being asked by human, which considers multiple features such as question informativeness, complexity, answer type, etc.  $Z$  is a normalization constant.

We also randomly generate a larger set of questions to pretrain the decoder component of the model as a "language model" over questions, enabling it to better capture the grammatical structure of possible questions. Details regarding the model hyperparameters, training procedure, and pre-training procedure are provided in Appendix A.2.

# 5.2.2 RESULTS AND DISCUSSION

We evaluate the log-likelihood of reference questions generated by our full model as well as some lesioned variants of the full model, including a model without pretraining, a model with the Transformer decoder replaced by an LSTM decoder, a model with the convolutional encoder replaced by a simple MLP encoder, and a model that only has a decoder (unconditional language model). Though the method from Rothe et al. (2017) also works on this task, here we cannot compare with their method for two reasons. One is that our dataset is constructed using their method, so the likelihood of their method should be an upper bound in our evaluation setting. Additionally, they can only approximate the log-likelihood due to an intractable normalizing constant, and thus it difficult to directly compare with our methods.

Two different evaluation sets are used, one is sampled from the same process on new boards, the other is a small set of questions collected from human annotators. In order to calculate the log-likelihood of human questions, we use translated versions of these questions that were used in previous work (Rothe et al., 2017), and filtered some human questions that score poorly according to the generative model used for training the neural network (Appendix A.2).

A summary of the results is shown in Table 2. The full model performs best on both datasets, suggesting that pretraining, the Transformer decoder, and the convolutional encoder are all important components of the approach. However, we find that the model without an encoder performs reasonably well too, even out-performing the full model with a LSTM-decoder on the human-produced questions. This suggests that while contextual information from the board leads to improvements, it is not the most important factor for predicting human questions. To further investigate the role of contextual information, we conduct another analysis to determine whether or not the model can utilize board information effectively.

Intuitively, if there is little uncertainty about the locations of the ships, observing the board is critical since there are fewer good questions to ask. To examine this factor, we divide the scenarios based on the entropy of the hypothesis space of possible ship locations into a low entropy set (bottom  $30\%$ ), medium entropy set ( $40\%$  in the middle), and high entropy set (top  $30\%$ ). We evaluate different models on the split sets of sampled data and report the results in Table 3. When entropy is low, the models with access to the board has substantially higher log-likelihood than the model without

Table 4: Evaluation results of experiment 3. Our grammar enhanced model is compared with a supervised trained baseline from experiment 2, a sequence generative RL baseline, and a text-based model. The models are compared in terms of average energy value, average expected information gain (EIG) value, the ratio of EIG value greater than  $0.9/0$ , number of unique questions generated, and number of unique novel questions generated (by "novel" we mean questions not presented in the human dataset). The EIG of text-based model is calculated based on the program form of the generated questions.  

<table><tr><td>Model</td><td>avg. EIG</td><td>EIG&gt;0.9</td><td>EIG&gt;0</td><td>#unique</td><td>#unique novel</td></tr><tr><td>text-based</td><td>0.928</td><td>62.80%</td><td>76.95%</td><td>-</td><td>-</td></tr><tr><td>supervised</td><td>1.033</td><td>51.65%</td><td>84.55%</td><td>137</td><td>9</td></tr><tr><td>sequence RL</td><td>1.235</td><td>58.60%</td><td>75.20%</td><td>167</td><td>52</td></tr><tr><td>grammar enhanced RL</td><td>1.266</td><td>84.75%</td><td>91.35%</td><td>141</td><td>129</td></tr></table>

encoder. If the entropy is high, the importance of the encoder is reduced. Together, this implies that our model can capture important context-sensitive characteristics of how people ask questions.

# 5.3 QUESTION GENERATION

In this experiment, we evaluate our reinforcement learning framework proposed in Section 4.2 on its ability of generating novel questions from scratch, without providing a large set of question demonstrations.

The reward for training the reinforcement agent is calculated based on the energy value of the generated question  $q$ . We transform the energy value to a proper range for reward by  $-\varepsilon(q) / 10$  and clamp it between  $-1$  and  $1$ . The model is optimized with the REINFORCE algorithm (Williams, 1992). A baseline for REINFORCE is established simply as the average of the rewards in a batch. In order to produce higher-quality questions, we manually tune the information-related parameter of the energy function from Rothe et al. (2017) to make it more information-seeking in this experiment. This process is described in Appendix A.2.

We compare the models on their ability to generate diverse questions with high expected information gain (EIG). We compare our program-based framework with a simple text-based model, which has the same architecture but is trained with supervision on the text-based question dataset collected by (Rothe et al., 2017). We also compare with the supervised program-based model from the last experiment. Finally, we implement a sequence-based reinforcement learning agent that specifies the program without direct access to the grammatical rules. For this alternative RL agent, we find it necessary to pretrain for 500 epochs with stepwise supervision.

# 5.3.1 RESULTS AND DISCUSSION

The models are evaluated on 2000 randomly sampled boards, and the results are shown in Table 4. Note that any ungrammatical questions are excluded when we calculate the number of unique questions. First, when the text-based model is evaluated on new contexts,  $96.3\%$  of the questions it generates were included in the training data. We also find that the average EIG and the ratio of  $\mathrm{EIG} > 0$  is worse than the supervised model trained on programs. Some of these deficiencies are due to the very limited text-based training data, but using programs instead can help overcome these limitations. With the program-based framework, we can sample new boards and questions to create a much larger dataset with executable program representations. This self-supervised training helps to boost performance, especially when combined with grammar-enhanced RL.

From the table, the grammar-enhanced RL model is able to generate informative and creative questions. It can be trained from scratch without examples of human questions, and produces many novel questions with high EIG. In contrast, the supervised model rarely produces new questions beyond the training set. The sequence-level RL model is also comparatively weak at generating novel questions, perhaps because it is also pre-trained on human questions. It also more frequently generates ungrammatical questions.

We also provide examples in Figure 4 to show the diversity of questions generated by the grammar enhanced model, and more in the supplementary materials. Figure 4a shows novel questions the model produces, which includes clever questions such as "Where is the bottom right of all the purple and blue tiles?" or "What is the size of the blue ship minus the purple ship?", while it can also

![](images/2e35453b1d3b082405a39ac3cd71b65d54a1e4a9a996eca77f814e60ffa4173f.jpg)  
(a) Novel questions  
Where is the bottom right of all purple and blue ships? (bottomright (union (coloredTiles Purple) (coloredTiles Blue))  
EIG: 1.76  
Energy: -5.76

![](images/861fb6608a15fa0d2ffe593ac00593d4f1ccb74d77dc21bbefdb0b89f4ed615f.jpg)  
What is the size of blue ship minus the size of purple ship?  
( - (size Blue) (setSize (coloredTiles Purple)))  
EIG: 2.02 Energy:-6.43

![](images/a39e1364da7f75eebecdde7ba460b36f5ee60380f8cceb37a6f7d6f85ee4ffe9.jpg)  
Is the blue ship shorter than the blue ship? (< ({setSize (coloredTiles Blue) (size Blue))  
EIG: 0.00 Energy:3.23

![](images/a820a31aa73459b6789bf7ffd1a47fd5a58ca1176eb4979f36bdd7cb49708f17.jpg)  
(b) Conditioned on question type  
Location Question:  
Where is the last tile of blue ship? (bottomright (coloredTiles Blue))  
Composed Question:

![](images/80a1312de8d354ce9502d257454ae274581f0f55fe673518a18635529977df5c.jpg)  
Is blue ship horizontal and also purple ship longer than 3? (and  $= = H$  orient Blue)  $(>3$  {(setSize {coloredTiles Purple}))  
What is the size of the blue ship? (setSize (coloredTiles Blue)) What is the size of the purple ship? (size Purple)  
Which column is the first part of the blue ship? (coll (topless (coloredTiles Blue))) What is the orientation of the blue ship? (orient Blue)  
Human:  
Do the red and purple ships touch? (touch Purple Red) How long is the purple ship? (size Purple)  
Is blue ship horizontal? (= orient Blue)  
Does the blue ship has 4 tiles long?  $(= (size Blue)4)$  
Is there a color block at 4-1?  
(not  $==$  (color 4-1) Water))  
$(++)$  (map {lambda x (= (size x) 2)) (set AllShips))) 0  
Are there any ships in row 1?  
$(++)$  (map lambda y (and  $= =$  (row L y) 1) (not  $= =$  (color y) Water))) (set AllTiles)) 0  
Is part of a ship on tile 4-6?  
(not  $==$  (color 4-6) Water))  
Figure 4: Examples of model-generated questions. The natural language translations of the question programs are provided for interpretation. (a) shows three novel questions generated by the grammar enhanced model, (b) shows an example of how the model generates different type of questions by conditioning the input to the decoder, (c) shows questions generated by our model as well as human annotators.

![](images/345b889698f45b79e81d7b2cd475b5dbda86290aa78ab43237a53880803fdc01.jpg)  
(c) Comparison with human questions  
Our model:  
What is the size of the blue ship plus the size of the red ship?  
$(+(\text{size Blue})$  (setSize coloredTiles Red))

![](images/811cb9e6d49d8781cb9fc64faba14f5c65389a7a62c77061988ed262e67642a0.jpg)  
Our model:  
Is the blue ship vertical?  
$= =$  (orient Blue) V)  
What is the size of the blue ship? (setSize coloredTiles Blue)  
What is the size of the blue ship? (size Blue)  
Where is the last part of the red ship?  
(bottomright (coloredTiles Red))  
Is the red ship parallel to the blue ship?  $(= =$  (orient Red) (orient Blue))  
Human:  
Do the red and purple ships touch? (touch Red Purple)  
Where does the blue ship start?  
(topless (coloredTiles Blue)) Is any ship two tiles long?

sometimes generates meaningless questions such as "Is the blue ship shorter than itself?" Additional examples of generated questions are provided in Appendix B.

With the grammar enhanced framework, we can also guide the model to ask different types of questions, consistent with the goal-directed nature and flexibility of human question asking. The model can be queried for certain types of questions by providing different start conditions to the model. Instead of starting derivation from the start symbol “A”, we can start derivation from a intermediate state such as “B” for Boolean questions or a more complicated “(and B B)” for composition of two Boolean questions. In Figure 4b, we show examples where the model is asked to generate four specific types of questions: true/false questions, number questions, location-related questions, and compositional true/false questions. We see that the model can flexibly adapt to new constraints and generate meaningful questions.

In Figure 4c, we compare the model generated questions with human questions, each randomly-sampled from the model outputs and the human dataset. These examples again demonstrate that our model is able to generate clever and human-like questions. However, we also find that people sometimes generate questions with quantifiers such as "any" and "all", which are operationalized in program form with lambda functions. These questions are complicated in representation and not favored by our model, showing a current limitation in our model's capacity.

# 6 CONCLUSION

This paper introduced a neural program generation framework for modeling human behavior in a rich question asking task, and generating creative human-like questions with grammar-enhanced reinforcement learning. Programs provide models with a "machine language of thought" for compositional thinking, and neural networks provide an efficient means of question generation. We demonstrate the effectiveness of our method in extensive experiments covering a range of human question asking abilities.

The current model has important limitations. It cannot generalize to systematically different scenarios, and it sometimes generates meaningless questions. We plan to further explore the model's compositional abilities in future work. Another promising direction is to model question asking and question answering jointly within one framework, which could guide the model to a richer sense of the question semantics. We would also like to use our framework in dialog systems and open-ended question asking scenarios, allowing such systems to synthesize informative and creative questions.

# REFERENCES

Jimmy Ba, Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. CoRR, abs/1607.06450, 2016.  
Rudy Bunel, Matthew Hausknecht, Jacob Devlin, Rishabh Singh, and Pushmeet Kohli. Leveraging grammar and reinforcement learning for neural program synthesis. In International Conference on Learning Representations, 2018.  
Yllias Chali and Sadid A Hasan. Towards topic-to-question generation. Computational Linguistics, 41(1):1-20, 2015.  
Jerry A. Fodor. The Language of Thought. Harvard University Press, 1975.  
Alex Graves, Greg Wayne, and Ivo Danihelka. Neural tuning machines. arXiv preprint arXiv:1410.5401, 2014.  
Todd Gureckis and Doug Markant. Active learning strategies in a spatial concept learning game. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 31, 2009.  
Michael Heilman and Noah A Smith. Good question! statistical ranking for question generation. In Human Language Technologies: The Annual Conference of the North American Chapter of the Association for Computational Linguistics, pages 609-617. Association for Computational Linguistics, 2010.  
Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Judy Hoffman, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Inferring and executing programs for visual reasoning. In Proceedings of the IEEE International Conference on Computer Vision, pages 2989-2998, 2017.  
Brenden M Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In International Conference on Machine Learning, 2018.  
Sang-Woo Lee, Youngjoo Heo, and Byoung-Tak Zhang. Answerer in questioner's mind: Information theoretic approach to goal-oriented visual dialog. In Advances in neural information processing systems, 2018.  
Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B Tenenbaum, and Jiajun Wu. The neuro-symbolic concept learner: Interpreting scenes, words, and sentences from natural supervision. In International Conference on Learning Representations, 2019.  
Nasrin Mostafazadeh, Ishan Misra, Jacob Devlin, C. Lawrence Zitnick, Margaret Mitchell, Xiaodong He, and Lucy Vanderwende. Generating natural questions about an image. In Annual Meeting of the Association for Computational Linguistics, pages 1802-1813, 2016.  
David Premack and Guy Woodruff. Does the chimpanzee have a theory of mind? Behavioral and brain sciences, 1(4):515-526, 1978.  
Marc'Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with recurrent neural networks. In International Conference on Learning Representations, 2016.  
Scott Reed and Nando De Freitas. Neural programmer-interpreters. In International Conference on Learning Representation, 2016.  
Anselm Rothe, Brenden M Lake, and Todd Gureckis. Question asking as program generation. In Advances in Neural Information Processing Systems, pages 1046-1055, 2017.  
Anselm Rothe, Brenden M Lake, and Todd M Gureckis. Do people ask good questions? Computational Brain & Behavior, 1(1):69-89, 2018.  
Iulian Vlad Serban, Alberto García-Durán, Caglar Gulcehre, Sungjin Ahn, Sarath Chandar, Aaron Courville, and Yoshua Bengio. Generating factoid questions with recurrent neural networks: The 30m factoid question-answer corpus. In Annual Meeting of the Association for Computational Linguistics, pages 588-598, 2016.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998-6008, 2017.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.  
Kaichun Yao, Libo Zhang, Tiejian Luo, Lili Tao, and Yanjun Wu. Teaching machines to ask questions. In International Joint Conferences on Artificial Intelligence, pages 4546-4552, 2018.  
Pengcheng Yin and Graham Neubig. A syntactic neural model for general-purpose code generation. In Annual Meeting of the Association for Computational Linguistics, pages 440-450, 2017.  
Xingdi Yuan, Tong Wang, Caglar Gulcehre, Alessandro Sordoni, Philip Bachman, Sandeep Subramanian, Saizheng Zhang, and Adam Trischler. Machine comprehension by text-to-text neural question generation. In Workshop on Representation Learning for NLP, 2017.  
Victor Zhong, Caiming Xiong, and Richard Socher. Seq2sql: Generating structured queries from natural language using reinforcement learning. CoRR, abs/1709.00103, 2018.
