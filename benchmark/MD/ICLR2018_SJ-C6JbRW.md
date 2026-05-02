# MASTERING THE DUNGEON: GROUNDED LANGUAGE LEARNING BY MECHANICAL TURKER DESCENT

Anonymous authors

Paper under double-blind review

# ABSTRACT

Contrary to most natural language processing research, which makes use of static datasets, humans learn language interactively, grounded in an environment. In this work we propose an interactive learning procedure called Mechanical Turker Descent (MTD) that trains agents to execute natural language commands grounded in a fantasy text adventure game. In MTD, Turkers compete to train better agents in the short term, and collaborate by sharing their agents' skills in the long term. This results in a gamified, engaging experience for the Turkers and a better quality teaching signal for the agents compared to static datasets, as the Turkers naturally adapt the training data to the agent's abilities.

# 1 INTRODUCTION

Natural language processing often relies on static benchmark datasets, which are used to measure progress of the field. Human language, however, is not static but emerges from communication and interaction with an environment. A dynamic environment brings several benefits: the ability for teachers and learners to control the data distribution according to the learner's abilities (Bengio et al., 2009), and the ability to pair learning language with the ability to act (Kiela et al., 2016).

We propose a general framework for interactive language learning called Mechanical Turker Descent (MTD). MTD is a competitive gamified setting. In each round of MTD, each Turker trains their own agent to compete with other Turkers' agents to win the bonus, where we design a simple and robust mechanism to evaluate the agents' performance by using the other Turkers' data. Due to the engaging nature of the competitive setting, Turkers are incentivized to create the best curriculum of training examples for their agents (not too easy, not too hard – but just right), and are also found to provide more training examples given the same amount of payment. At the same time, MTD is also a collaborative setting, where Turkers' data are merged after each round and shared in the next round. As a result, the agents improve their language abilities by interaction with humans and the environment in the long term.

As a concrete example of MTD, we develop a game interface called GraphWorld. The world is represented as a set of objects, along with directed typed edges indicating the relations between them. The set of possible actions of an agent are then defined as updates to the graph structure. Based on the GraphWorld interface, we build a text adventure game called Mastering the Dungeon, where humans control a dragon in a dungeon with various objects (e.g. elven sword), containers (e.g. treasure chest), locations (e.g. tower), and non-player characters (e.g. trolls). Turkers give training example pairs  $(x,y)$ , where  $x$  is a natural language command and  $y$  is an action sequence. The task is formulated as a language grounding problem where agents are trained to learn the mapping from  $x$  to  $y$ . For agents, we consider standard Seq2Seq models (Sutskever et al., 2014) with attention mechanism (Bahdanau et al., 2014), and also propose an architecture called Action-Centric Seq2Seq (AC-Seq2Seq), which gives improved performance by encoding the graph context and set of implementable actions.

In our experiments, we set up variants of MTD along with a baseline of static data collection on Mechanical Turk. Results show that an agent trained via MTD substantially outperforms an agent trained by the baseline data on a fixed held-out test set by up to 8.4 points in accuracy and 13.2 points in hits@1. Moreover, our ablation study shows that being engaging to humans and matching the training data distribution with the agent abilities are two important factors leading to the effectiveness of MTD.

![](images/c44444475332e1df8a7b7a8b4a469edc8300dfa3e144aeb8fc38d701e3c97b5e.jpg)  
Round  $i$  of MTD  
Figure 1: The competitive-collaborative Mechanical Turker Descent (MTD) algorithm. In each round Turkers are competitive to produce the best training data. However, in subsequent rounds they share all the data from the previous rounds so they are collaborative in the long term. The shared datasets are omitted here for simplicity.

# 2 RELATED WORK

Research into language learning can be divided into work that studies static datasets and work that studies grounding in an environment where learning agents can act. It is generally easier to collect natural language datasets for the former fixed case. Static datasets such as visual question answering (Antol et al., 2015) provide grounding into images, but no possibility for language learning through interaction. Some works utilize a geographical environment such as a maze but still employ static datasets (Artzi & Zettlemoyer, 2013).

It has been argued that virtual embodiment of agents is a viable long-term strategy for artificial intelligence research and the learning of natural language semantics, particularly in the form of games which also contain human players (Kiela et al., 2016). Grounding language in an interactive environment is an active area of research, however a number of recent works employ synthetic, templated language only (Sukhbaatar et al., 2015; Yu et al., 2017; Bordes et al., 2010; Hermann et al., 2017; Mikolov et al., 2015; Chaplot et al., 2017). Some works that do utilize real natural language and interaction include Wang et al. (2016), where language is learnt to solve block puzzles, and Wang et al. (2017) where language is learnt to draw voxel images, which are both quite different to our case of studying text adventure games. Other works study text adventure games, like we do, but without the communication element (He et al., 2016; Narasimhan et al., 2015).

Many methods that collect natural language for learning utilize Amazon's Mechanical Turk, as we do. However, the overwhelming majority collect data both in a static (rather than interactive) fashion, and by using the standard scheme of a fixed payment per training example; this includes those works mentioned previously. We use such collection schemes as our baseline to compare to the Mechanical Turker Descent (MTD) algorithm we introduce in this paper.

There are some systems that have attempted to apply competitive, collaborative and/or gamification strategies to collect data, notably the ESP game (Von Ahn & Dabbish, 2004), which is an image annotation tool where users are paired and have to "read each others mind" to agree on the contents of an image. ReferItGame (Kazemzadeh et al., 2014) and Peekaboom (Von Ahn et al., 2006) have similar ideas but for localizing objects. In a completely different field, Foldit is an online game where players compete to manipulate proteins (Eiben et al., 2012). In comparison, our approach, Mechanical Turker Descent, is not specific to a particular task and can be applied across a wide range of machine learning problems, whilst more directly optimizing the quality of data for learning.

# 3 ALGORITHM: MECHANICAL TURKER DESCENT

The Mechanical Turker Descent (MTD) algorithm is a general method for collecting training data. It is designed to be engaging for human labelers and to collect high quality training data, avoiding common pitfalls of other data collection schemes. We first describe it in the general case, and subsequently in Section 4.1 we describe how we apply it to our particular game engine scenario.

MTD consists of  $N$  human labelers (Turkers) who all use a common interface for data collection, and a sequence of rounds of labeling, where feedback is given to the labelers after each round. Before the first round, we initialize two datasets  $D_{train\_all}$  and  $D_{test\_all}$ , which could either be (i) empty; or (ii) initial sets of data collected outside of the MTD algorithm. Both  $D_{train\_all}$  and  $D_{test\_all}$  are shared by all the labelers and updated each round.

Each round consists of the following steps, also summarized in Figure 1:

1. At the beginning of the round, each of the  $N$  Turkers provides a set of labeled examples in the form of  $(x,y)$  pairs, giving  $N$  datasets  $D_{1},\ldots ,D_{N}$ . In our experiments we consider two settings for data collection: either (i) a fixed number of examples, or (ii) as many examples as the Turker can provide within a fixed time limit (with a lower bound on the number of examples). We find that (ii) is a more natural setup in order to avoid idle time due to stragglers, and to encourage individual engagement and efficiency.  
2. In the next step,  $N$  separate models are trained, one for each Turker, each using the same learning algorithm, but different data. For Turker  $i$ , a model  $M_{i}$  is trained on the dataset  $D_{i} \cup D_{train\_all}$ .  
3. Each Turker  $i$  is assigned a score  $S_{i}$  for the quality of their labeling based on the performance of their model  $M_{i}$ . The model  $M_{i}$  is evaluated using accuracy (or some other evaluation metric) on the evaluation dataset  $(D_{1} \cup D_{2} \cup \dots \cup D_{N} \cup D_{test\_all}) \setminus D_{i}$ , i.e. using the shared test set along with all other Turker's data other than their own. Let  $|D_{m}| = \min_{i} |D_{i}|$  be the size of the smallest dataset. We propose to normalize the metric by the size of the datasets to avoid bias towards any one Turker's dataset. The score of Turker  $i$  is computed as:

$$
S _ {i} = \frac {\sum_ {j \neq i} | D _ {m} | A c c (M _ {i} , D _ {j}) + | D _ {t e s t \_ a l l} | A c c (M _ {i} , D _ {t e s t \_ a l l})}{(N - 1) \cdot | D _ {m} | + | D _ {t e s t \_ a l l} |} \tag {1}
$$

where  $Acc(m, d)$  measures the accuracy for model  $m$  on dataset  $d$ . The scores of the Turkers are made visible via a high-score table of performance. A paid bonus is awarded to the Turkers who have the top scoring entries in the table. This is an explicit gamification setting designed to engage and motivate Turkers to achieve higher scores and thus provide higher quality data.

4. The data from all Turkers collected in this round is added to the shared datasets. More specifically, we split the dataset  $(D_{1} \cup \dots \cup D_{N})$  randomly into two subsets  $D_{train\_cur}$  and  $D_{test\_cur}$ , and update the shared datasets to make them available to all Turker's models on the subsequent round:  $D_{train\_all} \gets D_{train\_all} \cup D_{train\_cur}$  and  $D_{test\_all} \gets D_{test\_all} \cup D_{test\_cur}$ . At this point, the process repeats.

MTD is a competitive-collaborative algorithm. In each round, Turkers are incentivized to provide better data than their competitors. However, evaluation of quality is measured by performance on datasets from competing Turkers, making it inherently collaborative: they must agree on a common "language" of examples, i.e. they must follow a similar distribution. Further, on subsequent rounds, due to Step 4, they all share the same data they collected together, hence they are incentivized to work together in the longer term. One can make an analogy with the publication model of the research community, where researchers competitively write papers to be accepted at conferences (which is like a round of MTD), whilst using each other's ideas to build subsequent research for the next conference (which is like subsequent rounds of MTD).

Why is this a good idea? Our algorithm simultaneously brings two advantages over standard data collection procedures. Firstly it gamifies the data collection process, which is known to be more engaging to labelers (Von Ahn & Dabbish, 2004). Secondly, our approach avoids many of the common pitfalls of conventional data collection, leading to high quality data:

- Avoids examples being too easy In standard data collection, there is nothing preventing new training examples being too easy. Many similar training examples may have already been collected, and a model may only need a subset of them to do well on the rest. In MTD, there is no incentive to add easy examples as these will not improve the Turker's trained model, which negatively affects their score (position in the leaderboard). In addition, since the data is also used to evaluate other Turkers' models, providing easy examples will lead to higher scores of other Turkers, yielding a competitive disadvantage.  
- Avoids examples being too hard In standard data collection, there is nothing preventing new training examples being too hard for the model to generalize from. In MTD, there is no incentive to add too hard examples, as these will also not improve the Turker's model and their score.  
- Human-curated curriculum In MTD, there is incentive to provide examples that are "just right" for the model to generalize well to new examples. As the model should be improving on each round, this also incentivizes Turkers to provide a curriculum (Bengio et al., 2009) of harder and harder examples that are suitable for the model as it improves. Since the Turker acts as the model's teacher they are essentially defining the curriculum as teachers do for students. Choosing the best examples for the model to see next is also related to active learning (Cohn et al., 1994) except in our case this is chosen on the teacher's, rather than the learner's side.  
- MTD is not easily exploitable/gameable Mechanical Turk data collection is notorious for providing poor results unless the instructions and setup are very carefully crafted (Goodman et al., 2013). MTD's scoring system is resistant to a number of attacks designed to game it. Firstly, collusion is difficult as Turkers are randomly grouped into a set of  $N$  participants with no ability to communicate or to find out who the other participants are. Even if collusion does occur between e.g. a pair of Turkers, as  $N$  is expected to be large ( $N = 30$  in our experiments) and the evaluation scores are averaged by Turker such an attack is of small influence (cf. Eq. 1). Secondly, if a Turker seeks to create an evaluation set that reduces other Turkers scores (by acting on their own or via collusion) e.g. by creating hard-to-classify examples, these examples are importantly also the ones their own model is trained on. Hence, this strategy is actually more likely to deteriorate their own model's performance, while having relatively small influence on the performance of others. In general optimal performance is found by cooperating with others to some degree (making examples somewhat similar) whilst being competitive (trying to make more and/or more useful examples than their competitors).

There are also a number of extensions one could consider to MTD, we describe some of them in Appendix I.

# 4 GAME ENVIRONMENT: MASTERING THE DUNGEON

In this section we describe a general game interface called GraphWorld that we employ in our experiments. It is designed to be modular and extensible. Being a game, it provides an engaging interface between agents and humans for data collection, and focuses on research into grounded agents that learn to both communicate and act.

The underlying representation (grounding) in GraphWorld is a graph where each concept, object, location and actor is a node in the graph, and labeled edges represent relations between them. For example, paths between locations are edges with "path_to" labels, an agent is in the location which is connected to it with a "contained_by" edge, and movement involves altering the latter "contained_by" edge to another location. Similarly, objects have various properties: food, drink, wearable, wieldable, container, and so on. Each action in the game (if it can be executed, depending on the graph state) leads to a new state which is a change in the graph structure. Every action hence has a set of prequisites (e.g. is there a path in the graph that makes this move action possible) followed by a transformation of the graph that executes it.

Here, this underlying representation is used to generate a classic text adventure game, Mastering the Dungeon, in a fantasy setting with swords, castles and trolls, but it is a general formalism that could be used to build other games as well. The actions include moving, picking up objects, eating,

```txt
> look  
You are in the forest.  
A troll is here.  
There is a rusty sword, a glass of beer, and a mace here.  
There is a path to the cavern.  
> hit troll  
You hit the troll! The troll is dead!!!  
> go cavern  
You are in the cavern.  
An orc is here.  
There is an axe, a treasure chest, a crossbow, and three apples here.  
There are paths to the forest and the tower.  
> get apple  
Done.  
> eat apple  
Yum.  
> inventory  
You are carrying nothing.  
> get crossbow  
Done.  
> put crossbow in treasure chest  
You put a crossbow in the treasure chest.
```

![](images/c72cdbe2ce67fcdbf6cf3dd31bbcbe9d83336503b5d7a0ced84e5657379ac05a.jpg)  
Figure 2: Example gameplay from GraphWorld (left), part of the underlying graph representation (top right) and the set of actions possible within the game (bottom right).

```txt
Available Actions: look examine <thing> go <room> follow <agent> get/drop <object> eat/drink <object> wear/remove <object> wield/unwield <object> hit <agent> put <object> in <container> get <object> from <container> give <object> to <agent> take <object> from <agent>
```

etc. We implemented a total of 15 action types, which closely follow those of classical online text adventure MUD (multi-user dungeon) games such as DikuMUD<sup>1</sup>. The full list is given in Fig 2 along with an example of execution of the game and the underlying graph structure representation.

What is appealing about the GraphWorld formalism is that the grounding is extensible with (i) new actions, which can be coded by simply providing new transformations of the graph, and (ii) with more concepts - new locations, objects and actors can easily be added. This means that the (grounded) language in the simulation can easily grow, which is important for language research where small restricted dictionaries in simulations keep the research synthetic in nature (Weston et al., 2015). Here, we explore the mapping of natural language to grounded actions within GraphWorld, but the framework allows the study of other language and reasoning phenomena as well.

# 4.1 MTD FOR GRAPHWORLD

We investigate the learning problem of mapping from a natural language command  $x$  to a sequence of actions  $y$  in GraphWorld, for example "enter the bedchamber and toss your armor on the bed" maps to "go bedroom; remove helmet; put helmet on bed; remove chestplate; put chestplate on bed". We set up the MTD game as follows: each Turker is a player who is given a companion pet dragon that they can provide commands to. The player has to "train their dragon" by issuing it commands in natural language which it has to execute, and their goal is to train their dragon to perform better than their competitors, just as described in Section 3 in the general MTD case.

The particular interface for the Turkers we chose is the text adventure game itself, where they can type actions. To simplify the experience for novice gamers and first time users, at each step in the game, we list the set of possible actions so that the Turker can simply select one of them. At any stage (after any number of actions) they can enter "teach" to indicate that the last sequence entered will be the set of actions for a new training example (or else "reset" if they want to discard their current sequence). After entering "teach" the Turker provides the natural language command that should result in that set of actions. The natural language command and a representation of the state of the world become the input  $x$  and the actions that should be executed become the output  $y$  for the training example  $(x, y)$ . For representing the state of the world we simply store the entire graph, different models can then make use of that in different ways (e.g. represent it as features).

Data collection is performed within a randomized adventure game world (randomized for each training example) consisting of 3 locations, 3 agents, 14 objects (weapons, food, armor and others) and 2 containers, where locations and paths are randomized. We employ 30 Turkers on each round, and consider two settings: (i) ask them to create 10 examples each round; or (ii) ask them to create at

least 10 examples each round (but they can create more) with a maximum time of 40 minutes. The length of the action sequence is constrained to be at most 4. For each example added, the existing trained model from the previous round is executed and the Turker is told if the model gets the example correct already (which implies that the example is possibly "too easy"). This can help the Turker enter useful examples for the subsequent model to train on. The pay is  $3 per hit, and there is a$ 15 bonus for the top scoring Turker in the leaderboard, $10 for second place, and $5 for third. The leaderboard scores and bonus awarded (if any) are emailed at the end of each round. At that point, Turkers can sign up for the next round, which does not necessarily have to employ the same Turkers, but we did observe a significant amount of return players. We perform 5 rounds of MTD.

A natural comparison for MTD is the traditional method of data collection: simply pay Turkers per example collected. We choose the total pay to sum to the same as as the base pay plus bonuses for MTD, so the same dollar amount is spent. We ran this also as 5 rounds, but each round is effectively the same, as no model feedback is involved, and no leaderboard or bonuses are emailed. We also made sure that new Turkers were recruited, without prior game play experience of MTD, so as to avoid bias.

# 5 MODEL: AC-SEQ2SEQ

Our agent aims to learn a mapping from natural language command  $x$  to action sequence  $y$ . We treat this as a supervised learning problem. In the following text, we use "model" and "agent" interchangeably.

A natural baseline is a sequence-to-sequence (Seq2Seq) model (Sutskever et al., 2014) with attention mechanism (Bahdanau et al., 2014). In this section we propose the Action-Centric Seq2Seq (AC-Seq2Seq) model as an improvement over Seq2Seq, which takes advantage of the grounded nature of our task. AC-Seq2Seq shares the same encoder architecture with Seq2Seq, in our case a bidirectional GRU (Chung et al., 2014). The encoder encodes a sequence of word embeddings into a sequence of hidden states. AC-Seq2Seq has the following additional properties: it models (i) the notion of actions with arguments (using an action-centric decoder), (ii) which arguments have been used in previous actions (by maintaining counts); and (iii) which actions are possible given the current world state (by constraining the set of possible actions in the decoder). Details are provided below.

# Compositional Action Representation

Let  $\mathcal{A}$  denote the action space. Each action in the action space  $a \in \mathcal{A}$  can be denoted as  $a = (type, arg_1, arg_2)$ , which specifies a composition of an action type and two arguments. For example, the action take elven sword from troll is denoted as (take_from, elven_sword, troll). For actions with one argument,  $arg_2$  is set as none; i.e., go tower is denoted as (go_tower, none).

AC-Seq2Seq utilizes a compositional representation for each action  $a$ . More specifically, we concatenate an action type embedding with two argument embeddings, i.e.,  $\mathbf{a} = [\mathbf{e}_{type}, \mathbf{e}_{arg_1}, \mathbf{e}_{arg_2}]$ .

A compositional action representation is data-efficient because different actions share common action type and/or argument representations. For example, it is easier for the model to generalize to get elven sword after seeing get treasure chest because the get representations are shared. In contrast, the baseline Seq2Seq model treats each action in the action sequence as atomic, which neglects their compositional nature<sup>2</sup>.

# Action-Centric Decoder

First we describe how we vectorize the input before introducing the decoder formulations. Consider a decoding step  $j$ . For each action  $a = (type, \arg_1, \arg_2)$ , we employ the two argument embeddings  $\mathbf{e}_{\arg_1}$  and  $\mathbf{e}_{\arg_2}$  as query vectors to attend over encoder hidden states respectively, and concatenate the two attention results, denoted as  $att_a$ . Let  $count_{a,j}$  be the number of occurrences of the two arguments  $\arg_1$  and  $\arg_2$  in previous decoding steps from 1 to  $j - 1$ . Let  $location_j$  be the current location (e.g., cavern). We then use a graph context vector  $env_{a,j}$  to encode  $count_{a,j}$  and  $location_j$  by concatenating their learnt embeddings.

A key difference between Seq2Seq and AC-Seq2Seq is that instead of using a single vector representation (hidden state) at each time step to predict an action, AC-Seq2Seq maintains a set of action-centric hidden states. More specifically, we maintain a hidden state  $\mathbf{h}_{a,j}$  for action  $a$  at decoding step  $j$ . The hidden states are updated as follows

$$
\mathbf {h} _ {a, j} = G R U ([ \mathbf {a}; a t t _ {a}; e n v _ {a, j} ], \mathbf {h} _ {a, j - 1})
$$

In other words, we concatenate an action representation  $\mathbf{a}$ , an attention result  $att_{a}$ , and a graph context vector  $env_{a,j}$  as the input. A GRU is employed to update the hidden states for each action, and the weights of the GRU are shared among actions.

Given the model parameter  $\mathbf{w}$ , the probability distribution over the action space at decoding step  $j$  can be written as

$$
P _ {a, j} = \frac {\exp \mathbf {w} ^ {\top} \mathbf {h} _ {a , j}}{\sum_ {a ^ {\prime} \in \mathcal {A}} \exp \mathbf {w} ^ {\top} \mathbf {h} _ {a ^ {\prime} , j}}
$$

The above action-centric formulation allows us to leverage the compositionality of action representations described in Section 5. Moreover, such an action-centric view enables better matching between the input natural language commands and the action arguments because the attention mechanisms are conditioned on actions. For example, one can tie the embeddings of tower in the natural language command and tower in the action go tower so that it is possible for the model to learn go tower even without seeing the word tower before.

Action Space Decoding Constraint During decoding, we constrain the set of possible actions to be only among the valid actions given the current world state. For example, it is not valid to go tower if the dragon is in the tower, or there is no path to the tower from the current location. This constraint is applied to both Seq2Seq and AC-Seq2Seq in our experiments.

# 6 EXPERIMENTS

We employ the environment and MTD settings described in Section 4.1. For all the results in this section, we train the agents for 10 runs and report the mean and standard deviation. To study the effects of interactive learning, we compare the following learning procedures:

- MTD is our proposed algorithm. The Turkers are asked to create at least 10 examples per round (but they can create more) in a maximum time of 40 minutes, repeated for 5 rounds.  
- MTD ablations: We consider two possible ablations of the MTD algorithm:

- MTD limit has a limit on the number of examples. The Turkers are asked to create exactly 10 examples per round. Our hypothesis is that Turkers are willing to create more examples to win the game (be higher on the leaderboard), hence MTD limit should be worse than MTD.  
- MTD limit w/o model is MTD limit without model feedback. The Turkers are not informed about the model predictions and thus cannot adapt the data distribution according to the agent abilities. Our hypothesis is thus that MTD and MTD limit should outperform this method.

- Collaborative-only baseline is the conventional static data collection method where Turkers are asked to create 10 examples given a fixed amount of payment. Total payment is set to be equal to the MTD variants.

During online deployment of the MTD algorithm, AC-Seq2Seq models are trained each round and deployed to inform the Turkers about model predictions, to evaluate the agents' performance and to produce the leaderboard ranking. We combine the shared test sets  $D_{test\_all}$  from all of the above settings and an initial pilot study dataset (see Appendix E) to form a held-out test set for all methods. We train Seq2Seq models using the same training data collected using AC-Seq2Seq models $^3$ , and evaluate them on the same held-out test set.

<table><tr><td>Method</td><td>Accuracy</td><td>hits@1</td><td>F1</td></tr><tr><td>Training AC-Seq2Seq</td><td></td><td></td><td></td></tr><tr><td>MTD</td><td>0.418 ± 0.010</td><td>0.461 ± 0.033</td><td>0.701 ± 0.009</td></tr><tr><td>MTD limit</td><td>0.402 ± 0.009</td><td>0.431 ± 0.033</td><td>0.682 ± 0.007</td></tr><tr><td>MTD limit w/o model</td><td>0.386 ± 0.010</td><td>0.419 ± 0.053</td><td>0.682 ± 0.007</td></tr><tr><td>Collaborative-only baseline</td><td>0.334 ± 0.015</td><td>0.329 ± 0.034</td><td>0.644 ± 0.012</td></tr><tr><td>Training Seq2Seq</td><td></td><td></td><td></td></tr><tr><td>MTD</td><td>0.261 ± 0.005</td><td>0.026 ± 0.002</td><td>0.589 ± 0.008</td></tr><tr><td>MTD limit</td><td>0.241 ± 0.003</td><td>0.024 ± 0.003</td><td>0.569 ± 0.006</td></tr><tr><td>MTD limit w/o model</td><td>0.229 ± 0.003</td><td>0.020 ± 0.002</td><td>0.554 ± 0.005</td></tr><tr><td>Collaborative-only baseline</td><td>0.219 ± 0.005</td><td>0.032 ± 0.003</td><td>0.525 ± 0.010</td></tr></table>

Table 1: Main evaluation results. Interactive learning (MTD) outperforms static learning (collaborative-only baseline). Our model (AC-Seq2Seq) outperforms a vanilla Seq2Seq.

![](images/28da03c8d7805c5d9e45892ce6d83798cf9702ed0f0c58388c24b8080c3d9c29.jpg)  
Figure 3: Learning curves of different methods. Red lines and black lines correspond to AC-Seq2Seq and Seq2Seq respectively.

We report three metrics: accuracy, hits@1 and F1. Accuracy is determined by the ratio of test examples for which the action sequence predicted by the model leads to a correct end state as defined by the underlying graph. To compute hits@1 for each test example  $x$ , we randomly sample 99 additional examples in the test set and compute the rank of  $y$  from within that list; hits@k for larger  $k$  are given in the Appendix, Fig. 3. F1 is defined at the action level and averaged over examples. The results are given in Table 1.

MTD outperforms static data collection (collaborative-only baseline) substantially and consistently on all the metrics for both models. The improvement over the collaborative baseline is up to 8.4 points in accuracy and 13.2 points in hits@1. This indicates that MTD is effective at collecting high-quality data and thus training better agents. Unsolicited feedback from Turkers also indicates their high level of engagement, see the Appendix H for details.

The ablation study shows that MTD outperforms MTD limit, which shows that through an engaging, gamified setting, Turkers have higher incentives to create more examples in order to win the competition, and create  $30\%$  more examples on average compared to MTD limit. Both MTD and MTD limit outperform MTD limit w/o model. This clearly indicates that model feedback contributes to better agent performance. This also justifies our argument that dynamic coordination between training data distribution and agent abilities is important, avoiding too easy or too hard examples.

Lastly, AC-Seq2Seq outperforms Seq2Seq by a large margin of up to 15.7 points in accuracy, demonstrating that the inductive biases based on the GraphWorld action space are important. Similar trends can also be observed in Fig 3, where we plot the learning curves of agents trained with different learning procedures. We examine the relative contribution of (i) tracking which arguments have

been used in previous actions and (ii) which actions are possible given the current world state (by constraining the set of possible actions in the decoder) in a separate ablation study in Appendix G, and find that these lead to improved performance.

# 7 CONCLUSIONS

We studied the interactive learning of situated language, specifically training agents to act within a text adventure game environment given natural language commands from humans. To train such agents, we proposed a general interactive learning framework called Mechanical Turker Descent (MTD) where Turkers train agents both collaboratively and competitively. Experiments show that (i) interactive learning based on MTD is more effective than learning with static datasets; (ii) there are two important factors for its effectiveness: it is engaging to Turkers, and it produces training data distributions that match agent's capabilities. Going forward, we hope to apply these same techniques to learn more complex language tasks in richer domains.

# REFERENCES

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. VQA: Visual Question Answering. In Proceedings of the IEEE International Conference on Computer Vision, pp. 2425-2433, 2015.  
Yoav Artzi and Luke Zettlemoyer. Weakly supervised learning of semantic parsers for mapping instructions to actions. Transactions of the Association for Computational Linguistics, 1:49-62, 2013.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.  
Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pp. 41-48. ACM, 2009.  
Antoine Bordes, Nicolas Usunier, Ronan Collobert, and Jason Weston. Towards understanding situated natural language. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, pp. 65-72, 2010.  
Devendra Singh Chaplot, Kanthashree Mysore Sathyendra, Rama Kumar Pasumarthi, Dheeraj Rajagopal, and Ruslan Salakhutdinov. Gated-attention architectures for task-oriented language grounding. arXiv preprint arXiv:1706.07230, 2017.  
Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.  
David Cohn, Les Atlas, and Richard Ladner. Improving generalization with active learning. Machine learning, 15(2):201-221, 1994.  
Christopher B Eiben, Justin B Siegel, Jacob B Bale, Seth Cooper, Firas Khatib, Betty W Shen, Barry L Stoddard, Zoran Popovic, and David Baker. Increased diels-alderase activity through backbone remodeling guided by foldit players. Nature biotechnology, 30(2):190-192, 2012.  
Joseph K Goodman, Cynthia E Cryder, and Amar Cheema. Data collection in a flat world: The strengths and weaknesses of mechanical turk samples. Journal of Behavioral Decision Making, 26(3):213-224, 2013.  
Ji He, Jianshu Chen, Xiaodong He, Jianfeng Gao, Lihong Li, Li Deng, and Mari Ostendorf. Deep reinforcement learning with an action space defined by natural language. 2016.  
Karl Moritz Hermann, Felix Hill, Simon Green, Fumin Wang, Ryan Faulkner, Hubert Soyer, David Szepesvari, Wojtek Czarnecki, Max Jaderberg, Denis Teptyashin, et al. Grounded language learning in a simulated 3d world. arXiv preprint arXiv:1706.06551, 2017.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara L Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, pp. 787-798, 2014.  
Douwe Kiela, Luana Bulat, Anita L Vero, and Stephen Clark. Virtual embodiment: A scalable long-term strategy for artificial intelligence research. arXiv preprint arXiv:1610.07432, 2016.  
Tomas Mikolov, Armand Joulin, and Marco Baroni. A roadmap towards machine intelligence. arXiv preprint arXiv:1511.08130, 2015.  
Karthik Narasimhan, Tejas Kulkarni, and Regina Barzilay. Language understanding for text-based games using deep reinforcement learning. arXiv preprint arXiv:1506.08941, 2015.  
Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.  
Sainbayar Sukhbaatar, Arthur Szlam, Gabriel Synnaeve, Soumith Chintala, and Rob Fergus. Mazebase: A sandbox for learning from games. arXiv preprint arXiv:1511.07401, 2015.  
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pp. 3104-3112, 2014.  
Luis Von Ahn and Laura Dabbish. Labeling images with a computer game. In Proceedings of the SIGCHI conference on Human factors in computing systems, pp. 319-326. ACM, 2004.  
Luis Von Ahn, Ruoran Liu, and Manuel Blum. Peekaboom: a game for locating objects in images. In Proceedings of the SIGCHI conference on Human Factors in computing systems, pp. 55-64. ACM, 2006.  
Sida I Wang, Percy Liang, and Christopher D Manning. Learning language games through interaction. arXiv preprint arXiv:1606.02447, 2016.  
Sida I Wang, Samuel Ginn, Percy Liang, and Christopher D Manning. Naturalizing a programming language via interactive learning. arXiv preprint arXiv:1704.06956, 2017.  
Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M Rush, Bart van Merrienboer, Armand Joulin, and Tomas Mikolov. Towards ai-complete question answering: A set of prerequisite toy tasks. arXiv:1502.05698, 2015.  
Haonan Yu, Haichao Zhang, and Wei Xu. A deep compositional framework for human-like language acquisition in virtual environment. arXiv preprint arXiv:1703.09831, 2017.
