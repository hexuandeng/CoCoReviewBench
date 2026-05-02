# EMERGENT WORLD REPRESENTATIONS: EXPLORING A SEQUENCE MODEL TRAINED ON A SYNTHETIC TASK

Anonymous authors

Paper under double-blind review

# ABSTRACT

Language models show a surprising range of capabilities, but the source of their apparent competence is unclear. Do these networks just memorize a collection of surface statistics, or do they rely on internal representations of the process that generates the sequences they see? We investigate this question by applying a variant of the GPT model to the task of predicting legal moves in a simple board game, Othello. Although the network has no a priori knowledge of the game or its rules, we uncover evidence of an emergent nonlinear internal representation of the board state. Interventional experiments indicate this representation can be used to control the output of the network and create "latent saliency maps" that can help explain predictions.

# 1 INTRODUCTION

Recent language models have shown an intriguing range of capabilities. Networks trained on a simple "next-word" prediction task are apparently capable of many other things, such as solving logic puzzles or writing basic code. Yet how this type of performance emerges from sequence predictions remains a subject of current debate.

Some have suggested that training on a sequence modeling task is inherently limiting. The arguments range from philosophical (Bender & Koller, 2020) to mathematical (Merrill et al., 2021). A common theme is that seemingly good performance might result from memorizing "surface statistics," i.e., a long list of correlations that do not reflect a causal model of the process generating the sequence. But relying on spurious correlations may lead to problems on out-of-distribution data (Bender et al., 2021; Floridi & Chiriatti, 2020).

On the other hand, some tantalizing clues suggest language models may do more than collect spurious correlations, building interpretable world models—that is, understandable models of the process producing the sequences they are trained on. Recent evidence suggests language models can develop internal representations for very simple concepts, such as color, direction Abdou et al. (2021); Patel & Pavlick (2022), or tracking boolean states during synthetic tasks (Li et al., 2021) (see Related Work (section 6) for more detail).

The question remains, then, of how we might investigate the emergence of world models in more complex domains. One possibility comes from Toshniwal et al. (2021), who explore language models trained on chess move sequences. These models learn to predict legal moves with high accuracy. Furthermore, by analyzing predicted moves, the paper shows that the model appears to track the board state. The authors stop short, however, of investigating the form of any internal representations. Such an investigation will be the focus of this paper.

# 1.1 THE GAME OF OTHELLO AS TESTBED FOR INTERPRETABILITY

Toshniwal et al. (2021)'s observations suggest a new approach to studying the representations learned by sequence models. If we think of a board as the "world," then games provide us with an appealing experimental testbed to explore world representations of moderate complexity. As our setting, we choose the popular game of Othello ( Figure 1), which is simpler than chess. This setting allows us to

investigate world representations in a highly controlled context, where both the task and sequence being modeled are synthetic and well-understood.

As a first step, we train a language model (a GPT variant we call Othello-GPT) to extend partial game transcripts (a list of moves made by players) with legal moves. The model has no a priori knowledge of the game or its rules. All it sees during training is a series of tokens derived from the game transcripts. Each token represents a tile where players place their discs. Note that we do not explicitly train the model to make strategically good moves or to win the game. Nonetheless, our model is able to generate legal Othello moves with high accuracy.

Our next step is look for world representations that might be used by the network. In Othello, the "world" consists of the current board position. A natural question is if, within the model, we can identify a representation of the board state involved in producing its next move predictions. To study this question, we train a set of probes, i.e., classifiers which allow us to infer the board state from the internal network activations. This type of probing has become a standard tool for analyzing neural networks (Alain & Bengio, 2016; Tenney et al., 2019; Belinkov, 2016).

Using this probing methodology, we find evidence for an emergent world representation. In particular, we show that a non-linear probe is able to predict the board state with high accuracy (section 3). (Linear probes, however, produce poor results.) This probe defines an internal representation of the board state. We then provide evidence that this representation plays a causal role in the network's predictions. Our main tool is an intervention technique that modifies internal activations so that they correspond to counterfactual board states.

We also discuss how knowledge of the internal world model can be used as an interpretability tool. Using our activation-intervention technique, we create latent saliency maps, which provide insight into how the network makes a given prediction. These maps are built by performing attribution at a high-level setting (the board) rather than a low-level one (individual input tokens or moves).

To sum up, we present four contributions: (1) we provide evidence for an emergent world model in a GPT variant trained to produce legal moves in Othello; (2) we compare the performance of linear and non-linear probing approaches, and find that non-linear probes are superior in this context; (3), we present an intervention technique that suggests that, in certain situations, the emergent world model can be used to control the network's behavior; and (4) we show how probes can be used to produce latent saliency maps to shed light on the model's predictions.

# 2 "LANGUAGE MODELING" OF OTHELLO GAME TRANSCRIPTS

Our approach for investigating internal representations of language models is to narrow our focus from natural language to a more controlled synthetic setting. We are partly inspired by the fact that language models show evidence of learning to make valid chess moves simply by observing game transcripts in training data (Srivastava et al., 2022). We choose the game Othello, which is simpler than chess, but maintains a sufficiently large game tree to avoid memorization. Our strategy is to see what, if anything, a GPT variant learns simply by observing game transcripts, without any a priori knowledge of rules or board structure.

The game is played on an 8x8 board where two players alternate placing white or black discs on the board tiles. The object of the game is to have the majority of one's color discs on the board at the end of the game. Othello makes a natural testbed for studying emergent world representations since the game tree is far too large to memorize, but the rules and state are significantly simpler than chess.

The following subsections describe how we train a system with no prior knowledge of Othello to predict legal moves with high accuracy. The system itself is not our end goal; instead, it serves as our object of study.

# 2.1 DATASETS: "CHAMPIONSHIP" AND "SYNTHETIC"

We use two sets of training data for the system, which we call "championship" and "synthetic". Each captures different objectives, namely data quality vs. quantity. While limited in size, championship data reflects strategic moves by expert human players. The synthetic data set is far larger, consisting of legal but otherwise random moves.

![](images/8a54302e8e2e8f43ae625ee86de05f0c4cae1e820af5cf27dd7ac793401cb1fc.jpg)  
$\mathrm{A}$  
Figure 1: A visual explanation of Othello rules, from left to right: (A) The board is initialized with four discs (two black and two white) placed in the center of the board. (B) Black always moves first. Every move must flip one or more opponent discs by outflanking—or sandwiching—the opponent disc(s). (C) The opponent repeats this process. A game ends when there are no more legal moves.

![](images/5a76f4a4330439447e8626f49495ee0661430e877707460c892570b52aa9d4f5.jpg)  
B  
C

![](images/4bc9af1034c06b2196daeaa1b43b44ce053b68a144f6027fb095dd04eb688a85.jpg)

Our championship dataset is produced by collecting Othello championship games from two online sources<sup>2</sup>, containing 7,605 and 132,921 games, respectively. They are combined and split randomly by  $8:2$  into training and validation sets. The games in this dataset were produced by matches where human players presumably made moves with a strategic intent to win. Following this, we generate a synthetic dataset with 20 million games for training and 3,796,010 games for validation. We compute this dataset by uniformly sampling leaves from the Othello game tree. Its data distribution is different from the championship games, reflecting no strategy.

# 2.2 MODEL AND TRAINING

Our goal is to study how much Othello-GPT can learn from pure sequence information, so we provide as few inductive biases as possible. (Note the contrast with a system like AlphaZero (Silver et al., 2018), where the goal was to win highly competitive chess games.) We therefore use only sequential tile indices as input to our model. For example, A4 and H6 are indexed as the 4rd and the 62st word in our vocabulary, respectively. Each game is treated as a sentence tokenized with a vocabulary of 60 words, where each word corresponds to one of the 60 tiles on which players put discs, excluding the 4 tiles in the center (Figure 1).

We trained an 8-layer GPT model (Radford et al., 2018; 2019; Brown et al., 2020) with an 8-head attention mechanism and a 512-dimensional hidden space. The training was performed in an autoregressive fashion. For each partial game  $\{y_t\}_{t=0}^{T-1}$ , the computation process starts from indexing a trainable word embedding consisting of the 60 vectors, each for one word, to get  $\{x_t^0\}_{t=0}^{T-1}$ . They are then sequentially processed by 8 multi-head attention layers. We denote the intermediate feature for the  $t$ -th token after the  $l$ -th layer as  $x_t^l$ . By employing a causal mask, only the features at the immediately preceding layer and earlier time steps  $x_{\leqslant t}^{l-1}$  are visible to  $x_t^l$ . Finally,  $x_{T-1}^8$  goes through a linear classifier to predict logits for  $\hat{y}_T$ . We minimize the cross-entropy loss between ground-truth move and predicted logits by gradient descent.

The model starts from randomly initialized weights, including in the word embedding layer. Though there are geometrical relationships between the 60 words (e.g., C4 is below B4), this inductive bias is not explicitly given to the model but rather left to be learned.

# 2.3 OTHELLO-GPT USUALLY PREDICTS LEGAL MOVES

We now evaluate how well the model's predictions adhere to the rules of Othello. For each game in the validation set, which was not seen during training, and for each step in the game, we ask Othello-GPT to predict the next legal move conditioned by the partial game before that move. We then calculate the error rate by checking if the top-1 prediction is legal. The error rate is  $0.01\%$  for Othello-GPT trained on the synthetic dataset and  $5.17\%$  for Othello-GPT trained on the championship dataset. For comparison, the untrained Othello-GPT has an error rate of  $93.29\%$ . The main takeaway is that Othello-GPT does far better than chance in predicting legal moves when trained on both datasets.

We discuss reasons for the difference between the error rates for the synthetic and championship models later in the paper.

A potential explanation for these results may be that Othello-GPT is simply memorizing all possible transcripts. To test for this possibility, we created a skewed dataset of 20 million games to replace the training set of synthetic dataset. At the beginning of every game, there are four possible opening moves: C4, D3, E6 and F5. This means the lowest layer of the game tree (first move) has four nodes (the four possible opening moves). For our skewed dataset, we truncate one of these nodes (C4), which is equivalent to removing a quarter of the whole game tree. Othello-GPT trained on the skewed dataset still yields an error rate of  $0.02\%$ . Since Othello-GPT has seen none of these test sequences before, pure sequence memorization cannot explain its performance<sup>3</sup>.

If the performance of Othello-GPT is not due to memorization, what is it doing? We now turn to this question by probing for internal representations of the game state.

# 3 EXPLORING INTERNAL REPRESENTATIONS WITH PROBES

We seek to understand if Othello-GPT computes internal representations of the game state. One standard tool for this task is a "probe" (Alain & Bengio, 2016; Belinkov, 2016; Tenney et al., 2019). A probe is a classifier or regressor whose input consists of internal activations of a network, and which is trained to predict a feature of interest, e.g., part of speech or parse tree depth (Hewitt & Manning, 2019). If we are able to train an accurate probe, it suggests that a representation of the feature is encoded in the network's activations.

In our case, we want to know whether Othello-GPT's internal activations contain a representation of the current board state. To study this question, we train probes that predict the board state from the network's internal activations after a given sequence of moves. Note that the board state—whether each tile is empty or holds a black or white disc—is generally a nonlinear function of the input tokens. On the other hand, it is straightforward to write a program to compute this function, it makes a natural probe target.<sup>4</sup>

We take the autoregressive features  $x_{t}^{l}$  that summarize the partial sequence  $y \leqslant t$  as the input to the probe and study results from different layers  $l$ . The output  $p_{\theta}(x_{t}^{l})$  is a 3-way categorical the probability distribution. We randomly split pairs of internal representation and ground-truth tile state by 8:2 into training and validation set. Error rates on validation set are reported. A best random guess yields  $52.95\%$ , if the probe always guess the tile is empty.

# 3.1 LINEAR PROBES HAVE HIGH ERROR RATES

Our first result is that linear classifier probes have poor relative accuracy. Its function can be written as  $p_{\theta}(x_t^l) = \mathrm{softmax}(W x_t^l)$  where  $\theta = \{W \in \mathbb{R}^{F \times 3}\}$ .  $F$  is the number of dimensions of input  $x_t^l$ . As Table 1 shows, error rates never dip below  $20\%$ . As a baseline, we have included probes trained on a randomly initialized network<sup>5</sup> We can see that there is only a marginal improvement in accuracy when we move to probing a fully-trained network. This result suggests that if there is an internal representation of the board state, it does not have a simple linear form.

# 3.2 NONLINEAR PROBES HAVE LOWER ERROR RATES

Given the poor performance of linear probes, it is natural to ask whether a nonlinear probes would have higher accuracy. Moving up one notch of complexity, we apply a 2-layer MLP as a probe. This technique has been used successfully in other language model probing work, e.g., Conneau et al. (2018); Cao et al. (2021); Hernandez & Andreas (2021). Its function can be written as

Table 1: Error rates (%) of linear probes on Othello-GPTs trained on different datasets across different layers ( $x^i$  represents internal representations after the  $i$ -th layer).  

<table><tr><td></td><td>x1</td><td>x2</td><td>x3</td><td>x4</td><td>x5</td><td>x6</td><td>x7</td><td>x8</td></tr><tr><td>Randomized</td><td>26.7</td><td>27.1</td><td>27.6</td><td>28.0</td><td>28.3</td><td>28.5</td><td>28.7</td><td>28.9</td></tr><tr><td>Championship</td><td>24.2</td><td>23.8</td><td>23.7</td><td>23.6</td><td>23.6</td><td>23.7</td><td>23.8</td><td>24.3</td></tr><tr><td>Synthetic</td><td>21.9</td><td>20.5</td><td>20.4</td><td>20.6</td><td>21.1</td><td>21.6</td><td>22.2</td><td>23.1</td></tr></table>

$p_{\theta}(x_t^l) = \mathrm{softmax}(W_1\mathrm{ReLU}(W_2x_t^l))$  where  $\theta = \{W_1\in \mathbb{R}^{H\times 3},W_2\in \mathbb{R}^{F\times H}\}$ .  $H$  is the number of hidden dimensions for the nonlinear probes.

The probe accuracy for trained networks, shown in Table 2, is significantly better than the linear probe in absolute terms. By contrast, the baseline (probing a randomized network with nonlinear probes) shows almost no improvement over the linear case. These results indicate that the probe may be recovering a nontrivial representation of board state in the network's activations. In section 4, we describe intervention experiments validating this hypothesis.

Table 2: Error rates (%) of nonlinear probes on Othello-GPTs trained across layers.  

<table><tr><td></td><td>x1</td><td>x2</td><td>x3</td><td>x4</td><td>x5</td><td>x6</td><td>x7</td><td>x8</td></tr><tr><td>Randomized</td><td>25.5</td><td>25.4</td><td>25.5</td><td>25.8</td><td>26.0</td><td>26.2</td><td>26.2</td><td>26.4</td></tr><tr><td>Championship</td><td>12.8</td><td>10.3</td><td>9.5</td><td>9.4</td><td>9.8</td><td>10.5</td><td>11.4</td><td>12.4</td></tr><tr><td>Synthetic</td><td>11.3</td><td>7.5</td><td>4.8</td><td>3.4</td><td>2.4</td><td>1.8</td><td>1.7</td><td>4.6</td></tr></table>

# 4 VALIDATING PROBES WITH INTERVENTIONAL EXPERIMENTS

Our nonlinear probe accuracies suggest that Othello-GPT computes information reflecting the board state. It's not obvious, however, whether that information is causal for the model's predictions. To investigate this issue, we evaluate whether the representations uncovered through section 3 play a causal role in Othello-GPT's predictions. In the following section, we adhere to Belinkov (2016)'s recommendation, performing a set of interventional experiments to determine the causal relationship between model predictions and the emergent world representations.

To figure out whether the board state information affects the network's predictions, we influence internal activations during Othello-GPT's calculation and measure the resulting effects. At a high level, the interventions are as follows: given a set of activations from the Othello-GPT, a probe predicts a baseline board state  $B$ . We save the move predictions associated with  $B$ , then modify these activations such that our probe reports an updated board state  $B'$ . Through our protocol, only a single tile  $s$  distinguishes  $B'$  from  $B$ 's board state (an example of which can be seen in Figure 2. This small modification results in a different set of possible legal moves for  $B'$ . If the new predictions match our expectations for  $B'$  and not  $B$  we conclude the representation had a causal effect on the model.

# 4.1 INTERVENTION TECHNIQUE

To implement an intervention that changes the predicted state from a board position  $B$  to a modified version  $B'$  we must decide (a) which layers to modify activations in, and (b) how to modify those activations. The first question is subtle. Given the causal attention mechanism of GPT-2, modifying activations for only one layer is unlikely to be effective as later layer computations incorporate information from prior board representations unaffected by our intervention. Instead, we select an initial layer  $L_s$  then modify it and subsequent layers' activations (see Figure 2 (C)). Our modification uses a simple gradient descent method on the probe's class score for the particular tile  $s$  whose state is being modified.

Figure 2 illustrates an intervention on a single feature  $x$  into  $x'$  such that the corresponding board state  $B$  is updated to match the desired  $B'$ . We observe the effectiveness of these interventions by probing the intervened  $x'$  or  $x$  at later layers (see Appendix D), as well as the change in next-step prediction in (see subsection 4.2). Consistent with the training process of probes  $p_{\theta}$ , we use cross entropy loss between the probe-predicted probability distribution and the desired board state, but

![](images/c87effc6471ba67c25e99805e12860ee03fe0cbf35815df8598fb2ded9d4b586.jpg)

![](images/bcd674fc412c4b7f2896cc1b051cbba26d954e891ee10b2e97fc17fa1157c36e.jpg)

![](images/7e339d75ab50aa0b0c4298347c3de157355b0626b3dac600477b1f64fbae03c1.jpg)  
Figure 2: (A) explains how we intervene on a board tile. Here, we only want to flip one tile, e.g. F6, from white to black. In (B), four views present an Othello game in progression, which can be reliably probed from an internal representation  $x$ . The lower left board represents the model's perceived world state prior to intervention. The upper left board shows the model's predictions for legal moves given this state. Post-intervention, the model's world state is updated—F6's state has been switched from white to black (lower right), leading to a different set of legal move predictions (upper right). Note that two tiles (F6) are highlighted in the world state boards. This is the tile that we "intervene" on, changing from white to black. (C) Shows our proposed intervention scheme. Light blue indicates unmodified activations; dark blue represents activations affected by intervention. Starting from a predefined layer, we intervene at the temporally-last token (shown in (A)). We replace original internal representations with the post-intervention one and resume computation for the next one layer. Part of the misinformation gets corrected (light blue), but we alternate this intervening and computation process until the last layer, from which the next-step prediction is made.

![](images/0ccefc4b559d04c71802ca79cc610ffb720b76fb9b16829aec516f34c228c115.jpg)

![](images/3aff40258b1741979d2e663f943d55a7a6bfeab30158abdb1b9870eab233ba2f.jpg)

![](images/a70c4fe028b5b55bb72be04c1227929c06f8b02a454216bff7feaa2068585265.jpg)

rather than optimize probe weights  $\theta$ , we optimize  $x$  for intervention<sup>6</sup>:

$$
x ^ {\prime} \leftarrow x - \alpha \frac {\partial \mathcal {L} _ {\mathrm {C E}} \left(p _ {\theta} (x) , B ^ {\prime}\right)}{\partial x}.
$$

At timestep  $T$ , the internal activations of an  $L$ -layer Othello-GPT can be viewed as an  $L \times T$  grid of activation vectors. Our intervention process will work by running Othello-GPT sequentially, but using gradient descent to modify key activation vectors at the last timestep so that their board state class scores change. Note that if we change activations only at a middle layer, activations at higher layers are directly affected by pre-intervention information. Therefore, we sequentially intervene  $\{x_{T - 1}^l\}_{l = L_s}^L$  at the last timestep, on all activations starting from a preset layer  $L_{s}$  until the final layer, illustrated in Figure 2.

# 4.2 EVIDENCE FOR A CAUSAL ROLE FOR THE REPRESENTATION

To systematically evaluate if this world representation is causal for model predictions, we create two evaluation benchmarks. Each consists of 1000 intervention cases: one factual, one counterfactual. A test case in these benchmarks consists of a triplet of a partial game, a targeted board tile, and a

![](images/f4677b51f9cc2a04a387250f8ddc50d1d5dbfeaeab387ac2e093e45206cac7c5.jpg)  
Figure 3: Intervention experiment results. Red dashed lines represent average number of errors by testing pre-intervention predictions on post-intervention ground-truths, representing a null intervention method for contrast. The shaded area represents the  $95\%$  confidence interval.

target state. For each case, we will give the partial game to Othello-GPT and perform the intervention described in the previous section. That is, we extract its activations in the middle of the computation process, modify them to change the representation of the targeted board tile into the target state, give back the modified world representation and let it make prediction with this new world state.

In the counterfactual case, we specifically ask whether the model's world representation can represent arrangements of tiles on a board that are unreachable during legal Othello play, i.e., states that do not correspond to any legal sequence of moves. If the model can make correct predictions about such states, it helps rule out the possibility that our probes might have learned to merely project a sequence-oriented internal state to a board-based world model that the probes have hallucinated. If Othello-GPT can make correct predictions about counterfactual states, it provides evidence for an internal representation capable of representing a board rather than just a sequence.

To measure how well the prediction is aligned with ground-truth legal moves, we calculate a prediction set by comparing the prediction probability for each tile with  $\frac{1}{2N}$ , where  $N$  is the number of legal post-intervention moves. Then, we calculate an error per case (a sum of false positives and false negatives, shown in Figure 3)<sup>7</sup>. For both benchmarks, nonlinear probes with  $L_S = 5$  give the best result: average errors of 0.22 and 0.18 respectively. Interventions based on linear probes all give worse than baseline results. Compared to baseline errors (2.23 and 1.76), the proposed intervention technique is effective even under counterfactual board states, suggesting the emergent world representations are causal to model predictions.

# 5 LATENT SALIENCY MAPS: Attribution VIA INTERVENTION

The intervention technique of the previous section provides insight into the predictions of Othello-GPT. We might also use it to create visualizations which contextualize Othello-GPT's predictions in terms of the board state. The basic idea is simple. For any tile on the board, we ask how much the network's prediction would change if we applied the intervention of the previous section to change the state of that tile. This will yield a value per tile, positive or negative, corresponding to its saliency in the prediction (see algorithm 1). We then create a visualization of the board where tiles are colored according to their saliency. Because this map is based on the network's latent space rather than its input, we call it a latent saliency map.

Figure 4 shows latent saliency maps for the synthetic and championship versions of Othello-GPT. The two diagrams show a clear pattern. The synthetic Othello-GPT shows high saliency for precisely those tiles that are required to make a move legal. In almost all cases, other tiles have lower saliency values. Even without knowing how synthetic-GPT was trained, an experienced Othello player might be able to guess its goal. The latent saliency maps for the championship version, however, are more complex. Although tiles that relate directly to legality typically have high values, many other tiles show high saliency as well. This pattern makes sense, too. Expert moves rely on complex global features of the board. The difference between the latent saliency maps for the two versions of Othello-GPT suggests that the visualization technique is providing useful information about the two systems.

![](images/d7d824e839e498f0f91704e157ebb12000f3a1c68f668a5eba721940b958fb07.jpg)  
A

![](images/d18bb9a4001172ce8096a3d94ec9e0d0539418b1268969ea007867d4490b9dfc.jpg)

![](images/c602cc28244501a130e4bd51851e9dc9ce02ec48e73064aaa70f2ae1f8f344d3.jpg)

![](images/cc4b3856f59ac59f7508feb75a57b21114e6a0f20db41d46fc11dbabb7f879ca.jpg)

![](images/45f2db4489e467d29dbee7a7c8d7a3166aa6953d72a970f4cdd81c1724dd454e.jpg)

![](images/0c98416a553c37d23a1cd0ca3ac251ea77a5376be271ea89ca58db25fec4c32c.jpg)  
B

![](images/ca5fd2eae95e2b8c4f32c6d8b61b206c8adeaafe07f604ae000215c47222f4ef.jpg)

![](images/d613c318e892f3b9363697ea935d30c9c7fd0f55c69d0a00f2e8032f25726439.jpg)

![](images/40813d9656301d454ff1b1ea5a7f113a607b14bf08a99ebb27008a8aa582abb2.jpg)

![](images/6de08ec0fc00b7ca39d8a37b0b4093679e91329aaf904a29565d06e583ec3900.jpg)

![](images/597c83b2f0e83a944c0e78e6a4c5631fdebafec0b7404a3262ce87d26abbf12c.jpg)

![](images/ab55eeaca4ef8f40b1d88f5d1a64aa767713e0fb42227e0770a29c619b0c340a.jpg)

![](images/52c3df0099c250de56a1eaf8df8c88dc35b5a64bdf6f9e0281feafc7e01a3889.jpg)

![](images/ae07026e1e15ce645841540a9eea42f8fb31a7498f7ae086e773cfc6a5095d1d.jpg)

![](images/163fa44b037bcaecd398d8872f8ee97fa39897cefeff81846f28e0506781004d.jpg)

![](images/23c48be79638572bebd69e0d175f9527a9ab7d3f7f1b8fb9bbfb83362769594b.jpg)  
Figure 4: Latent saliency maps: Each subplot shows a different game state, and the top-1 prediction by the model is enclosed in a black box. Colors (red is high, blue is low) indicate the contribution of a square's state to this prediction. The contribution is higher when changing the internal representation of this square makes the prediction less likely. The values are normalized by subtracting the mean of the board. (A) Latent saliency maps for Othello-GPT trained on the synthetic dataset, where the model learns legal moves. (B) Latent saliency maps for Othello-GPT trained on the championship dataset. Rather than learning rules, this Othello-GPT learns to make strategically good moves.

![](images/87562be58fb5f200fb66eb2bd44b16947c78c5629e84b97555b8b0e28895e71e.jpg)

![](images/e465c1d12ace0f1977ef7149f1a33f34fd4454222708a8921f32757bbdcf1192.jpg)

![](images/86159cc76d400459858aae507db0ac2ad279d11cf8ba61c0c2c31c63348df731.jpg)

![](images/709d41c05a116a29f298d38ad9732a4f0ae862bbea50412c58387a6ff0ee0888.jpg)

Algorithm 1: Attribution via Intervention on Othello-GPT  
Inputs:  
 $B$  the current board state  
p a legal next move which we try to attribute  
Output:  
 $\{S_s\}_{s \in B}$  assigned sensitivity values for p  
 $p_0 \gets f_p(x_{t-1})$   
for  $s \in B$  do  
 $\begin{array}{l} \tilde{x}_{t-1} \gets \text{Intervention}(x_{t-1}, s) \\ p_s \gets f_p(\tilde{x}_{t-1}) \\ S_s \gets p_0 - p_s \end{array}$

# 6 RELATED WORK

Our work fits into a general line of investigation into world representations created by sequence models. For example, (Li et al., 2021) fine-tune two language models on synthetic natural language tasks (Long et al., 2016) and find evidence that semantic information about the underlying world state is at least weakly encoded in the activations of the network. More direct evidence of a faithful

representation of 3D color space comes from Abdou et al. (2021), who examine activations in the BERT model and find a geometric connection to a standard 3D color space. Another study by (Patel & Pavlick, 2022) shows that language models can learn to map conceptual domains, e.g., direction and color, onto a grounded world representation via prompting techniques (Brown et al., 2020). These investigations operate in natural language domains, but investigate relatively simple world models.

Another related stream of work concerns neural networks that learn board games. There is a long history of work in AI to learn game moves, but in general, these systems have been given some a priori knowledge of the structure of the game. Even one of the most general-purpose gameplaying engines, AlphaZero (Silver et al., 2018), has built-in knowledge of basic board structure and game rules (although, intriguingly, it seems to develop interpretable models of various strategic concepts (McGrath et al., 2021; Forde et al., 2022)).

Closer to the work described here—and a major motivation for our research—is Toshniwal et al. (2021), included in BIG-bench (Srivastava et al., 2022), which trains a language model on chess transcripts. They show strong evidence that transformer networks are building a representation of internal board state, but they stop short at investigating what form that representation takes. Our work can be seen as building on this line of research, with a focus on the geometry of internal representations.

The intervention technique we use in section 4 follows an approach of steering model output while keeping the model frozen. It is related to the ideas behind plug-and-play controllable text generation for autoregressive (Dathathri et al., 2019; Qin et al., 2020; Krause et al., 2020) and diffusion (Li et al., 2022) language models by optimizing the likelihood of the desired attribute and the fluency of generated texts at the same time. These methods naturally involve a trade-off and require several forward and backward passes to generate. Our proposed intervention method stands out by only working on internal representations and requires only one forward pass.

Finally, latent saliency maps can be viewed as a generalization of the TCAV (testing with concept activation vectors) approach Kim et al. (2018); Ghorbani et al. (2019); Koh et al. (2020). In the TCAV setting, attribution is performed via directional derivatives. This is essentially a linearization of the gradient-descent optimization used in our attribution maps.

# 7 CONCLUSION

Our experiments provide evidence that Othello-GPT maintains a representation of game board states—that is, the Othello "world"—to produce sequences it was trained on. This representation appears to be nonlinear in an essential way, as supported by the results of our linear probe and nonlinear probe experiments. Further, we find that these representations can be causally linked to how the model makes its predictions. Understanding of the internal representations of a sequence model is interesting in its own right, but may also be helpful in deeper interpretations of the network.

We have also described how interventional experiments may be used to create a "latent saliency map", which gives a picture, in terms of the Othello board, of how the network has made a prediction. Applied to two versions of Othello-GPT that were trained on different data sets, the latent saliency maps highlight the dramatic differences between underlying representations of the Othello-GPT trained on synthetic dataset and its counterpart trained on championship dataset.

There are several potential lines of future work. One natural extension would be to perform the same type of investigations with other, more complex games. It would also be interesting to compare the strategies learned by a sequence model trained on game transcripts with those of a model trained with a priori knowledge of Othello. One way to study this question is to compare latent saliency maps of Othello-GPT with standard saliency maps of an Othello-playing program which has the actual board state as input.

More broadly, it would be interesting to know how our results generalize to models trained on natural language. One stepping stone might be to look at language models whose training data has included game transcript. Will we see similar representation of board state? For more complex natural language tasks, can we find meaningful world representations? The tools described in this paper—nonlinear probes, layerwise interventions, and latent saliency maps—may yet prove useful in natural language settings.

# REFERENCES

Mostafa Abdou, Artur Kulmizev, Daniel Hershcovich, Stella Frank, Ellie Pavlick, and Anders Søgaard. Can language models encode perceptual structure without grounding? a case study in color. arXiv preprint arXiv:2109.06129, 2021.  
Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.  
Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, pp. 1-12, 2016.  
Emily M Bender and Alexander Koller. Climbing towards nlu: On meaning, form, and understanding in the age of data. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 5185-5198, 2020.  
Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pp. 610-623, 2021.  
Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.  
Steven Cao, Victor Sanh, and Alexander M Rush. Low-complexity probing via finding subnetworks. arXiv preprint arXiv:2104.03514, 2021.  
Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni. What you can cram into a single vector: Probing sentence embeddings for linguistic properties. arXiv preprint arXiv:1805.01070, 2018.  
Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, and Rosanne Liu. Plug and play language models: A simple approach to controlled text generation. arXiv preprint arXiv:1912.02164, 2019.  
Luciano Floridi and Massimo Chiriatti. Gpt-3: Its nature, scope, limits, and consequences. *Minds and Machines*, 30(4):681–694, 2020.  
Jessica Zosa Forde, Charles Lovering, George Konidaris, Ellie Pavlick, and Michael L Littman. Where, when & which concepts does alphazero learn? lessons from the game of hex. In AAAI Workshop on Reinforcement Learning in Games, volume 2, 2022.  
Amirata Ghorbani, James Wexler, James Y Zou, and Been Kim. Towards automatic concept-based explanations. Advances in Neural Information Processing Systems, 32, 2019.  
Evan Hernandez and Jacob Andreas. The low-dimensional linear geometry of contextualized word representations. arXiv preprint arXiv:2105.07109, 2021.  
John Hewitt and Christopher D Manning. A structural probe for finding syntax in word representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4129-4138, 2019.  
Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In International conference on machine learning, pp. 2668-2677. PMLR, 2018.  
Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In International Conference on Machine Learning, pp. 5338-5348. PMLR, 2020.  
Ben Krause, Akhilesh Deepak Gotmare, Bryan McCann, Nitish Shirish Keskar, Shafiq Joty, Richard Socher, and Nazneen Fatema Rajani. Gedi: Generative discriminator guided sequence generation. arXiv preprint arXiv:2009.06367, 2020.

Belinda Z Li, Maxwell Nye, and Jacob Andreas. Implicit representations of meaning in neural language models. arXiv preprint arXiv:2106.00737, 2021.  
Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori B Hashimoto. Diffusion improves controllable text generation. arXiv preprint arXiv:2205.14217, 2022.  
Reginald Long, Panupong Pasupat, and Percy Liang. Simpler context-dependent logical forms via model projections. arXiv preprint arXiv:1606.05378, 2016.  
Thomas McGrath, Andrei Kapishnikov, Nenad Tomašev, Adam Pearce, Demis Hassabis, Been Kim, Ulrich Paquet, and Vladimir Kramnik. Acquisition of chess knowledge in alphazero. arXiv preprint arXiv:2111.09259, 2021.  
William Merrill, Yoav Goldberg, Roy Schwartz, and Noah A Smith. Provable limitations of acquiring meaning from ungrounded form: What will future language models understand? Transactions of the Association for Computational Linguistics, 9:1047-1060, 2021.  
Roma Patel and Ellie Pavlick. Mapping language models to grounded conceptual spaces. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=gJcEM8sxHK.  
Lianhui Qin, Vered Shwartz, Peter West, Chandra Bhagavatula, Jena Hwang, Ronan Le Bras, Antoine Bosselut, and Yejin Choi. Back to the future: Unsupervised backprop-based decoding for counterfactual and abductive commonsense reasoning. arXiv preprint arXiv:2010.05906, 2020.  
Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.  
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.  
David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419): 1140-1144, 2018.  
Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022.  
Ian Tenney, Dipanjan Das, and Ellie Pavlick. Bert rediscovers the classical nlp pipeline. arXiv preprint arXiv:1905.05950, 2019.  
Shubham Toshniwal, Sam Wiseman, Karen Livescu, and Kevin Gimpel. Learning chess blindfolded: Evaluating language models on state tracking. arXiv preprint arXiv:2102.13249, 2021.
