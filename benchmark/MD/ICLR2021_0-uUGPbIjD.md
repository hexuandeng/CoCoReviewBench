# HUMAN-LEVEL PERFORMANCE IN NO-PRESS DIPLOMACY VIA EQUILIBRIUM SEARCH

Anonymous authors

Paper under double-blind review

# ABSTRACT

Prior AI breakthroughs in complex games have focused on either the purely adversarial or purely cooperative settings. In contrast, Diplomacy is a game of shifting alliances that involves both cooperation and competition. For this reason, Diplomacy has proven to be a formidable research challenge. In this paper we describe an agent for the no-press variant of Diplomacy that combines supervised learning on human data with one-step lookahead search via external regret minimization. External regret minimization techniques have been behind previous AI successes in adversarial games, most notably poker, but have not previously been shown to be successful in large-scale games involving cooperation. We show that our agent greatly exceeds the performance of past no-press Diplomacy bots, is unexploitable by expert humans, and achieves a rank of 23 out of 1,128 human players when playing anonymous games on a popular Diplomacy website.

# 1 INTRODUCTION

A primary goal for AI research is to develop agents that can act optimally in real-world multi-agent interactions (i.e., games). In recent years, AI agents have achieved expert-level or even superhuman performance in benchmark games such as backgammon (Tesauro, 1994), chess (Campbell et al., 2002), Go (Silver et al., 2016; 2017; 2018), poker (Moravecik et al., 2017; Brown & Sandholm, 2017; 2019b), and real-time strategy games (Berner et al., 2019; Vinyals et al., 2019). However, previous large-scale game AI results have focused on either purely competitive or purely cooperative settings. In contrast, real-world games, such as business negotiations, politics, and traffic navigation, involve a far more complex mixture of cooperation and competition. In such settings, the theoretical grounding for the techniques used in previous AI breakthroughs falls apart.

In this paper we augment neural policies trained through imitation learning with regret minimization search techniques, and evaluate on the benchmark game of no-press Diplomacy. Diplomacy is a longstanding benchmark for research that features a rich mixture of cooperation and competition. Like previous researchers, we evaluate on the widely played no-press variant of Diplomacy, in which communication can only occur through the actions in the game (i.e., no cheap talk is allowed).

Specifically, we begin with a blueprint policy that approximates human play in a dataset of Diplomacy games. We then improve upon the blueprint during play by approximating an equilibrium for the current phase of the game, assuming all players (including our agent) play the blueprint for the remainder of the game. Our agent then plays its part of the computed equilibrium. The equilibrium is computed via external regret matching (ERM) (Blackwell et al., 1956; Hart & Mas-Colell, 2000).

Search via ERM has led to remarkable success in poker. However, ERM only converges to a Nash equilibrium in two-player zero-sum games and other special cases, and ERM was never previously shown to produce strong policies in a mixed cooperative/competitive game as complex as no-press Diplomacy. Nevertheless, we show that our agent exceeds the performance of prior agents and for the first time convincingly surpasses human-level performance in no-press Diplomacy. Specifically, we show that our agent soundly defeats previous agents, that our agent is far less exploitable than previous agents, that an expert human cannot exploit our agent even in repeated play, and, most importantly, that our agent achieves a score of  $25.6\%$  when playing anonymously with humans on a popular Diplomacy website, compared to an average human score of  $14.3\%$ .

# 2 BACKGROUND AND RELATED WORK

Search has previously been used in almost every major game AI breakthrough, including backgammon (Tesauro, 1994), chess (Campbell et al., 2002), Go (Silver et al., 2016; 2017; 2018), poker (Moravcik et al., 2017; Brown & Sandholm, 2017; 2019b), and Hanabi (Lerer et al., 2020). A major exception is real-time strategy games (Vinyals et al., 2019; Berner et al., 2019). Similar to SPARTA as used in Hanabi (Lerer et al., 2020), our agent conducts one-ply lookahead search (i.e., changes the policy just for the current game turn) and thereafter assumes all players play according to the blueprint. Similar to the Pluribus poker agent (Brown & Sandholm, 2019b), our search technique uses external regret matching to compute an approximate equilibrium. In a manner similar to the sampled best response algorithm of Anthony et al. (2020), we sample a limited number of actions from the blueprint policy rather than search over all possible actions, which would be intractable.

Learning effective policies in games involving cooperation and competition has been studied extensively in the field of multi-agent reinforcement learning (MARL) (Shoham et al., 2003). Nash-Q and CE-Q applied Q learning for general sum games by using Q values derived by computing Nash (or correlated) equilibrium values at the target states (Hu & Wellman, 2003; Greenwald et al., 2003). Friend-or-foe Q learning treats other agents as either cooperative or adversarial, where the Nash Q values are well defined Littman (2001). The recent focus on "Deep" MARL has led to learning rules from game theory such as fictitious play and regret minimization being adapted to Deep reinforcement learning (Heinrich & Silver, 2016; Brown et al., 2019), as well as work on game-theoretic challenges of mixed cooperative/competitive settings such as social dilemmas and multiple equilibria in the MARL setting (Leibo et al., 2017; Lerer & Peysakhovich, 2017; 2019).

Diplomacy in particular has served for decades as a benchmark for multi-agent AI research (Kraus & Lehmann, 1988; Kraus et al., 1994; Kraus & Lehmann, 1995; Johansson & Haard, 2005; Ferreira et al., 2015). Recently, Paquette et al. (2019) applied imitation learning (IL) via deep neural networks on a dataset of more than 150,000 Diplomacy games. This work greatly improved the state of the art for no-press Diplomacy, which was previously a handcrafted agent (van Hal, 2013). Paquette et al. (2019) also tested reinforcement learning (RL) in no-press Diplomacy via Advantage Actor-Critic (A2C) (Mnih et al., 2016). Anthony et al. (2020) introduced sampled best response policy iteration, a self-play technique, which further improved upon the performance of Paquette et al. (2019).

# 2.1 DESCRIPTION OF DIPLOMACY

The rules of no-press Diplomacy are complex; a full description is provided by Paquette et al. (2019). No-press Diplomacy is a seven-player zero-sum board game in which a map of Europe is divided into 75 provinces. 34 of these provinces contain supply centers (SCs), and the goal of the game is for a player to control a majority (18) of the SCs. Each players begins the game controlling three or four SCs and an equal number of units.

The game consists of three types of phases: movement phases in which each player assigns an order to each unit they control, retreat phases in which defeated units retreat to a neighboring province, and adjustment phases in which new units are built or existing units are destroyed.

During a movement phase, a player assigns an order to each unit they control. A unit's order may be to hold (defend its province), move to a neighboring province, convoy a unit over water, or support a neighboring unit's hold or move order. Support may be provided to units of any player. We refer to a tuple of orders, one order for each of a player's units, as an action. That is, each player chooses one action each turn. There are an average of 26 valid orders for each unit (Paurette et al., 2019), so the game's branching factor is massive and on some turns enumerating all actions is intractable.

Importantly, all actions occur simultaneously. In live games, players write down their orders and then reveal them at the same time. This makes the game an imperfect-information game in which an optimal policy may need to be stochastic in order to prevent predictability.

Diplomacy is designed in such a way that cooperation with other players is almost essential in order to achieve victory, even though only one player can ultimately win.

A game may end in a draw on any turn if all remaining players agree. Draws are a common outcome among experienced players because players will often coordinate to prevent any individual from reaching 18 centers. The two most common scoring systems for draws are draw-size scoring (DSS), in which all surviving players equally split a win, and sum-of-squares scoring (SoS), in which

player  $i$  receives a score of  $\frac{C_i^2}{\sum_{j\in\mathcal{N}}C_j^2}$ , where  $C_i$  is the number of SCs that player  $i$  controls (Fogel, 2020). Throughout this paper we use SoS scoring except in anonymous games against humans where the human host chooses a scoring system.

# 2.2 EXTERNAL REGRET MATCHING

External Regret Matching (ERM) (Blackwell et al., 1956; Hart & Mas-Colell, 2000) is an iterative algorithm that converges to a Nash equilibrium (NE) (Nash, 1951) in two-player zero-sum games and other special cases, and converges to a coarse correlated equilibrium (CCE) (Hannan, 1957) in general.

We consider a game with  $\mathcal{N}$  players where each player  $i$  chooses an action  $a_{i}$  from a set of actions  $\mathcal{A}_i$ . We denote the joint action as  $a = (a_1, a_2, \ldots, a_N)$ , the actions of all players other than  $i$  as  $a_{-i}$ , and the set of joint actions as  $\mathcal{A}$ . After all players simultaneously choose an action, player  $i$  receives a reward of  $v_i(a)$  (which can also be represented as  $v_i(a_i, a_{-i})$ ). Players may also choose a probability distribution over actions, where the probability of action  $a_{i}$  is denoted  $\pi_i(a_i)$  and the vector of probabilities is denoted  $\pi_i$ .

Normally, each iteration of ERM has a computational complexity of  $\Pi_{i\in \mathcal{N}}|\mathcal{A}_i|$ . In a seven-player game, this is typically intractable. We therefore use a sampled form of ERM in which each iteration has a computational complexity of  $\sum_{i\in \mathcal{N}}|\mathcal{A}_i|$ . We now describe this sampled form of ERM.

Each agent  $i$  maintains an external regret value for each action  $a_{i} \in \mathcal{A}_{i}$ , which we refer to simply as regret. The regret on iteration  $t$  is denoted  $R_{i}^{t}(a_{i})$ . Initially, all regrets are zero. On each iteration  $t$  of ERM,  $\pi_{i}^{t}(a_{i})$  is set according to

$$
\pi_ {i} ^ {t} \left(a _ {i}\right) = \left\{ \begin{array}{l l} \frac {\max  \left\{0 , R _ {i} ^ {t} \left(a _ {i}\right) \right\}}{\sum_ {a ^ {\prime} \in \mathcal {A} _ {i}} \max  \left\{0 , R _ {i} ^ {t} \left(a _ {i} ^ {\prime}\right) \right\}} & \text {i f} \sum_ {a _ {i} ^ {\prime} \in \mathcal {A} _ {i}} \max  \left\{0, R _ {i} ^ {t} \left(a _ {i} ^ {\prime}\right) \right\} > 0 \\ \frac {1}{| \mathcal {A} _ {i} |} & \text {o t h e r w i s e} \end{array} \right. \tag {1}
$$

Next, each player samples an action  $a_{i}^{*}$  from  $\mathcal{A}_i$  according to  $\pi_i^t$  and all regrets are updated such that

$$
R _ {i} ^ {t + 1} \left(a _ {i}\right) = R _ {i} ^ {t} \left(a _ {i}\right) + v _ {i} \left(a _ {i}, a _ {- i} ^ {*}\right) - \sum_ {a _ {i} ^ {\prime} \in \mathcal {A} _ {i}} \pi_ {i} ^ {t} \left(a _ {i} ^ {\prime}\right) v _ {i} \left(a _ {i} ^ {\prime}, a _ {- i} ^ {*}\right) \tag {2}
$$

ERM guarantees that  $R_{i}^{t}(a_{i}) \in \mathcal{O}(\sqrt{t})$ . If  $R_{i}^{t}(a_{i})$  grows sublinearly for all players' actions, as in ERM, then the average policy over all iterations converges to a NE in two-player zero-sum games and in general the empirical distribution of players' joint policies converges to a CCE as  $t \to \infty$ .

In order to improve empirical performance, we use linear ERM (Brown & Sandholm, 2019a), which weighs updates on iteration  $t$  by  $t$ . We also use optimism (Syrgkanis et al., 2015), in which the most recent iteration is counted twice when computing regret. Additionally, the action our agent ultimately plays is sampled from the final iteration's policy, rather than the average policy over all iterations. This reduces the risk of sampling a non-equilibrium action due to insufficient convergence. In theory sampling from the final iteration may increase exploitability, but this technique has been used successfully in past poker agents (Brown & Sandholm, 2019b).

# 3 AGENT DESCRIPTION

Our agent is composed of two major components. The first is a blueprint policy and state-value function trained via imitation learning on human data. The second is a search algorithm that utilizes the blueprint. This algorithm is executed on every turn, and approximates an equilibrium policy (for all players, not just the agent) for the current turn via ERM, assuming that the blueprint is played by all players for the remaining game beyond the current turn.

# 3.1 SUPERVISED LEARNING

We construct a blueprint policy via imitation learning on a corpus of 46,148 Diplomacy games collected from online play, building on the methodology and model architecture described by Paquette

et al. (2019) and Anthony et al. (2020). A blueprint policy and value function estimated from human play is ideal for performing search in a general-sum game, because it is likely to realistically approximate state values and other players' actions when playing with humans. Our blueprint supervised model is based on the DipNet agent from Paquette et al. (2019), but we make a number of modifications to the architecture and training.

We trained the blueprint policy using only a subset of the data used by Paquette et al. (2019), specifically those games obtained from webdiplomacy.net. For this subset of the data, we obtained metadata about the press variant (full-press vs. no-press) which we add as a feature to the model, and anonymized player IDs for the participants in each game. Using the IDs, we computed ratings  $s_i$  for each player  $i$  and only trained the policy on actions from players with above-average ratings. Appendix A describes our method for computing these ratings.

Our model closely follows the architecture of Paquette et al. (2019), with additional dropout of 0.4 between GNN encoder layers. We model sets of build orders as single tokens because there are a small number of build order combinations and it is tricky to predict sets auto-regressively with teacher forcing. We adopt the encoder changes of Anthony et al. (2020), but do not adopt their relational order decoder because it is more expensive to compute and leads to only marginal accuracy improvements after tuning dropout.

We make a small modification to the encoder GNN architecture that improves modeling. In addition to the standard residual that skips the entire GNN layer, we replace the graph convolution $^2$  with the sum of a graph convolution and a linear layer. This allows the model to learn a hierarchy of features for each graph node (through the linear layer) without requiring a concomitant increase in graph smoothing (the GraphConv). The resulting GNN layer computes (modification in red)

$$
x _ {i + 1} = \operatorname {D r o p o u t} \left(\operatorname {R e L U} \left(\operatorname {B N} \left(\operatorname {G r a p h C o n v} \left(x _ {i}\right) + \mathbf {A x} _ {\mathrm {i}}\right)\right)\right) + x _ {i}. \tag {3}
$$

where  $A$  is a learned linear transformation.

Finally, we achieve a substantial improvement in order prediction accuracy using a featurized order decoder. Diplomacy has over 13,000 possible orders, many of which will be observed infrequently in the training data. Therefore, by featurizing the orders by the order type, and encodings of the source, destination, and support locations, we observe improved prediction accuracy.

Specifically, in a standard decoder each order  $o$  has a learned representation  $e_o$ , and for some board encoding  $x$  and learned order embedding  $e_o$ ,  $P(o) = \text{softmax}(x \cdot e_o)$ . With order featurization, we use  $\tilde{e}_o = e_o + Af_o$ , where  $f_o$  are static order features and  $A$  is a learned linear transformation. The order featurization we use is the concatenation of the one-hot order type with the board encodings for the source, destination, and support locations. We found that representing order location features by their location encodings works better than one-hot locations, presumably because the model can learn more state-contextual features. $^3$

We add an additional value head to the model immediately after the dipnet encoder, that is trained to estimate the final SoS scores given a board situation. The value head is an MLP with one hidden layer that takes as input the concatenated vector of all board position encodings. A softmax over powers' SoS scores is applied at the end to enforce that all players' SoS scores sum to 1.

# 3.2 EQUILIBRIUM SEARCH

The policy that is actually played results from a search algorithm which utilizes the blueprint policy. Let  $s$  be the current state of the game. On each turn, the search algorithm computes an equilibrium for a subgame and our agent plays according to its part of the equilibrium solution for its next action.

Conceptually, the subgame is a well-defined game that begins at state  $s$ . The set of actions available to each player is a subset of the possible actions in state  $s$  in the full game, and are referred to as the

Table 1: Effect of model and training data changes on supervised model quality. Our final blueprint model improves modeling accuracy by about  $2\%$  and achieves a 1v6 score of  $20\%$  against 6 of the original DipNet model.  

<table><tr><td>Model</td><td>Policy Accuracy</td><td>SoS v. T=0.5</td><td>DipNet T=0.1</td></tr><tr><td>DipNet (Paquette et al. (2019))</td><td>60.5%4</td><td>0.143</td><td></td></tr><tr><td>+ combined build orders &amp; encoder dropout</td><td>62.0%</td><td></td><td></td></tr><tr><td>+ encoder changes from Anthony et al. (2020)</td><td>62.4%</td><td>0.150</td><td>0.198</td></tr><tr><td>switch to webdiplomacy training data only</td><td>61.3%5</td><td>0.175</td><td>0.206</td></tr><tr><td>+ output featurization</td><td>62.0%</td><td>0.184</td><td>0.188</td></tr><tr><td>+ improved GNN layer</td><td>62.4%</td><td>0.183</td><td>0.205</td></tr><tr><td>+ merged GNN trunk</td><td>62.9%</td><td>0.199</td><td>0.202</td></tr></table>

![](images/6718849d135d92a9a070e5c95bf5c9f6ab81ff5d821043653e3646411da33f83.jpg)  
Figure 1: Left: Score of SearchBot using different numbers of sampled subgame actions  $M_{i}$ , against 6 DipNet agents ((Paquette et al., 2019) at temperature 0.1). A score of  $14.3\%$  would be a tie. Even when sampling only two actions, SearchBot dramatically outperforms our blueprint, which achieves a score of  $20.2\%$ . Sampling a single action leads to poor performance due to all-hold actions, which is fixed if these actions are explicitly excluded. Right: The effect of different rollout lengths on SearchBot performance.

![](images/122c00188c7cffd25b4b353657ed7c95e60dc82ea6ec222abbd6024f86623b01.jpg)

subgame actions. Each player  $i$  chooses a subgame action  $a_{i}$ , resulting in joint subgame action  $a$ . After  $a$  is taken, the players make no further decisions in the subgame. Instead, the players receive a reward corresponding to the players sampling actions according to the blueprint policy  $\pi^b$  for the remaining game.

The subgame actions for player  $i$  are the  $M_{i}$  highest-probability actions according to the blueprint model.  $M_{i}$  is a hyperparameter that is proportional to the number of units controlled by player  $i$ . The effect of different choices for  $M_{i}$  is plotted in Figure 1 (left).

Rolling out  $\pi^b$  to the end of the game is very expensive, so in practice we instead roll out  $\pi^b$  for a small number of turns (usually 2 or 3 movement phases in our experiments) until state  $s'$  is reached, and then use value for  $s'$  from the blueprint's value network as the reward vector. Figure 1 (right) shows the performance of our search agent using different rollout lengths. We do not observe improved performance for rolling out farther than 3 or 4 movement phases.

We compute a policy for each agent by running the sampled regret matching algorithm described in Equation 1 and Equation 2. The search algorithm typically required between 2 minutes and 20 minutes per turn using a single Volta GPU and 8 CPU cores, depending on the hyperparameters used for the game. Details on the hyperparameters we used are provided in Appendix F.

# 4 RESULTS

Using the techniques described in Section 3, we developed an agent we call SearchBot. Our experiments focus on two formats. The first evaluates SearchBot playing against the population of human players on a popular Diplomacy website. The second measures the exploitability of SearchBot. We also show head-to-head performance against prior bots in Appendix E.

# 4.1 PERFORMANCE AGAINST A POPULATION OF HUMAN PLAYERS

The ultimate test of an AI system is how well it performs in the real world with humans. To measure this, we had SearchBot anonymously play no-press Diplomacy games on the popular Diplomacy website webdiplomacy.net. Since there are 7 players in each game, average human performance is a score of  $14.3\%$ . In contrast, SearchBot scored  $25.6\% \pm 4.8\%$ . If the bot's performance for each of the 7 powers is weighed equally, this score increases to  $27.0\% \pm 5.3\%$ . The agent's performance is shown in Table 2 and a detailed breakdown is presented in Table 5 in the appendix.

In addition to raw score, we measured SearchBot's performance using the Ghost-Rating system (Anthony, 2020), which is a Diplomacy rating system inspired by the Elo system that accounts for the relative strength of opponents and that is used to semi-officially rank players on webdiplomacy.net. Among no-press Diplomacy players on the site, our agent ranked 23 out of 1,128 players with a Ghost-Rating of 176.0 as of September 30th, 2020.

Table 2: Performance of our agent in anonymous games against humans on webdiplomacy.net. Average human performance is  $14.3\%$ . Score in the case of draws was determined by the rules of the joined game. The  $\pm$  shows one standard error. A breakdown of performance per power is provided in Table 5 in Appendix F.  

<table><tr><td>Power</td><td>Bot Score</td><td>Human Mean</td><td>Games</td><td>Wins</td><td>Draws</td><td>Losses</td></tr><tr><td>All Games</td><td>25.6% ± 4.8%</td><td>14.3%</td><td>50</td><td>7</td><td>16</td><td>27</td></tr><tr><td>Normalized By Power</td><td>27.0% ± 5.3%</td><td>14.3%</td><td>50</td><td>7</td><td>16</td><td>27</td></tr></table>

Details on the setup for experiments are provided in Appendix F.

# 4.2 EXPLOITABILITY

While performance of an agent within a population of human players is the most important metric, that metric alone does not capture how the population of players might adapt to the agent's presence. For example, if our agent is extremely strong then over time other players might adopt the bot's playstyle. As the percentage of players playing like the bot increases, other players might adopt a policy that seeks to exploit this playstyle. Thus, if the bot's policy is highly exploitable then it might eventually do poorly even if it initially performs well against the population of human players.

This can partly be interpreted through an evolutionary lens using the notion of an evolutionarily stable strategy (ESS) (Taylor & Jonker, 1978; Smith, 1982), which is a refinement of Nash equilibrium (Nash, 1951). If our agent's policy is an ESS (or Nash equilibrium), then a population of players all playing the agent's policy could not be "invaded" by a different policy. That is, no other policy could do better than tie against the population's policy.

Motivated by this, we measure the exploitability of our agent. Exploitability of a policy profile  $\pi$  (denoted  $e(\pi)$ ) measures worst-case performance when all but one agents follows  $\pi$ . Formally, the exploitability of  $\pi$  is defined as  $e(\pi) = \sum_{i \in \mathcal{N}} \max_{\pi_i} v_i(\pi_i, \pi_{-i}) / N$ , where  $\pi_{-i}$  denotes the policies of all players other than  $i$ . Agent  $i$ 's best response to  $\pi_{-i}$  is defined as  $BR(\pi_{-i}) = \arg \max_{\pi_i} v_i(\pi_i, \pi_{-i})$ .

We estimate our agent's full-game exploitability in two ways: by training an RL agent to best respond to the bot, and by having expert humans repeatedly play against six copies of the bot. We also measure the 'local' exploitability in the search subgame and show that it converges to an approximate Nash equilibrium.

# 4.2.1 PERFORMANCE AGAINST A BEST-RESPONDING AGENT

When the policies of all players but one are fixed, the game becomes a Markov Decision Process (MDP) (Howard, 1960) for the non-fixed player because the actions of the fixed players can be viewed as stochastic transitions in the "environment". Thus, we can estimate the exploitability of  $\pi$  by first training a best response policy  $BR(\pi_{-i})$  for each agent  $i$  using any single-agent RL algorithm, and then computing  $\sum_{i\in \mathcal{N}}v_i(BR(\pi_{-i}),\pi_{-i}) / N$ . Since the best response RL policy

![](images/acef1c842a82023432207aa13884fb52f4e4ccd3613364680f03dd8b318c51bb.jpg)  
Figure 2: Score of the exploiting agent against the blueprint and SearchBot-clone as a function of training time. We report the average of six runs. The shaded area corresponds to three standard errors. We use temperature 0.5 for both agents as it minimizes exploitability for the blueprint. Since SearchBot-clone is trained through imitation learning of SearchBot, the exploitability of SearchBot is almost certainly lower than SearchBot-clone.

will not be an exact best response (which is intractable to compute in a game as complex as no-press Diplomacy) this only gives us a lower-bound estimate of the exploitability.

Following other work on environments with huge action spaces (Vinyals et al., 2019; Berner et al., 2019), we use a distributed asynchronous actor-critic RL approach to optimize the exploiter policy (Espeholt et al., 2018). We use the same architecture for the exploiter agent as for the fixed model. Moreover, to simplify the training we initialize the exploiter agent from the fixed model.

We found that training becomes unstable when the policy entropy gets too low. The standard remedy is to use an entropy regularization term. However, due to the immense action space, an exact computation of the entropy term,  $E_{a}\log p_{\theta}(a)$ , is infeasible. Instead, we optimize a surrogate loss that gives an unbiased estimate of the gradient of the entropy loss (see Appendix C). We found this to be critical for the stability of the training.

Training an RL agent to exploit SearchBot is prohibitively expensive. Even when choosing hyperparameters that would result in the agent playing as fast as possible, SearchBot typically requires at least a full minute in order to act each turn. Instead, we collect a dataset of self-play games of SearchBot and train a supervised agent on this dataset. The resulting agent, which we refer to as SearchBot-clone, is weaker than SearchBot but requires only a single pass through the neural network in order to act on a turn. By training an agent to exploit SearchBot-clone, we can obtain a (likely) upper bound on what the performance would be if a similar RL agent were trained against SearchBot. We report the reward of the exploiter agents against the blueprint and SearchBot-clone agents in Figure 2.

# 4.2.2 PERFORMANCE AGAINST EXPERT HUMAN EXPLOITERS

In addition to training a best-responding agent, we also invited the 1st and 2nd place finishers in the 2017 World Diplomacy Convention (widely considered the world championship for full-press Diplomacy) to play games against six copies of our agent. The purpose was to determine whether the human experts could discover exploitable weaknesses in the bot.

The humans played games against three types of bots: DipNet (Paquette et al., 2019) (with temperature set to 0.5), our blueprint agent (with temperature set to 0.5), and SearchBot. In total, the participants played 35 games against each bot; each of the seven powers was controlled by a human player five times, while the other six powers were controlled by identical copies of the bot. The performance of the humans is shown in Table 3. While the sample size is relatively small, the results suggest that our agent is less exploitable than prior bots.

# 4.2.3 EXPLOITABILITY IN LOCAL SUBGAME

We first investigate the exploitability of our agent in the local subgame defined by a given board state, sampled actions, and assumed blueprint policy for the rest of the game. We simulate 7 games between a search agent and 6 DipNet agents, and plot the total exploitability of the average strategy of the search procedure as a function of the number of ERM iterations, as well as the exploitability of the blueprint policies. Utilities  $u_{i}$  are computed using Monte Carlo roll-

Table 3: Performance of one expert human playing against six bots under repeated play. A score less than  $14.3\%$  means the human is unable to exploit the bot. Five games were played for each power for each agent, for a total of 35 games per agent. For each power, the human first played all games against DipNet, then the blueprint model described in Section 3.1, and then finally SearchBot.  

<table><tr><td>Power</td><td>1 Human vs. 6 DipNet</td><td>1 Human vs. 6 Blueprint</td><td>1 Human vs. 6 SearchBot</td></tr><tr><td>All Games</td><td>39.1%</td><td>22.5%</td><td>5.7%</td></tr></table>

![](images/151c94fc18368d8991421b4a0d199c45c0f0b1d606df0e740a27932b4774daa0.jpg)  
Figure 3: Left: Distance of the CFR average strategy from equilibrium as a function of the CFR iteration, computed as the sum of all agents' exploitability in the matrix game in which CFR is employed. CFR converges to an approximate equilibrium, while the blueprint policy has only slightly lower exploitability than the uniform distribution over the 50 sampled CFR actions (i.e. CFR iteration 1). Right: Comparison of convergence of individual strategies to the average of two independently computed strategies. The similarity of these curves suggests that independent ERM computations lead to compatible equilibria. Note: Exploitability is averaged over all phases in 7 simulated games; more detailed results are provided in the Appendix.

![](images/a3fcefc134408ddd5490307f5f8e4f8bc94fd966e422c294761ce13aece2f198.jpg)

outs with the same blueprint as CFR, and total exploitability for a joint policy  $\pi$  is computed as  $e(\pi) = \sum_{i}\max_{a_{i}\in \mathcal{A}_{i}}u_{i}(a_{i},\pi_{-i}) - u_{i}(\pi)$ . The exploitability curves aggregated over all phases are shown in Figure 3 (left) and broken down by phase in the Appendix.

In Figure 3 (right), we verify that the average of policies from multiple independent executions of ERM also converges to an approximate Nash. For example, it is possible that if each agent independently running ERM converged to a different incompatible equilibrium and played their part of it, then the joint policy of all the agents would not be an ESS. However we observe that the exploitability of the average of policies closely matches the exploitability of the individual policies.

# 5 CONCLUSIONS

No-press Diplomacy is a complex game involving both cooperation and competition that poses major theoretical and practical challenges for past AI techniques. Nevertheless, our AI agent achieves human-level performance in this game with a combination of supervised learning on human data and one-ply search using external regret minimization. The massive improvement in performance from conducting search just one action deep matches a larger trend seen in other games, such as chess, Go, poker, and Hanabi, in which search dramatically improves performance. While external regret minimization has been behind previous AI breakthroughs in purely competitive games, it was never previously shown to be successful in a complex game involving cooperation. The success of ERM in no-press Diplomacy suggests that its use is not limited to purely adversarial games.

Our work points to several avenues for future research. SearchBot conducts search only for the current turn. In principle, this search could extend deeper into the game tree using counterfactual regret minimization (CFR) (Zinkevich et al., 2008). However, the size of the subgame grows exponentially with the depth of the subgame. Developing search techniques that scale more effectively with the depth of the game tree may lead to substantial improvements in performance. Another direction is combining our search technique with reinforcement learning. Combining search with reinforcement learning has led to tremendous success in perfect-information games (Silver et al., 2018) and more recently in two-player zero-sum imperfect-information games as well (Brown et al., 2020). Finally, it remains to be seen whether similar search techniques can be developed for variants of Diplomacy that allow for coordination between agents.

# REFERENCES

Thomas Anthony. Ghost-ratings, 2020. URL https://sites.google.com/view/webdipinfo/ghost-ratings.  
Thomas Anthony, Tom Eccles, Andrea Tacchetti, János Kramár, Ian Gemp, Thomas C Hudson, Nicolas Porcel, Marc Lanctot, Julien Pérolat, Richard Everett, et al. Learning to play no-press diplomacy with best response policy iteration. arXiv preprint arXiv:2006.04635, 2020.  
Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemysław Debiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, et al. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680, 2019.  
David Blackwell et al. An analog of the minimax theorem for vector payoffs. Pacific Journal of Mathematics, 6(1):1-8, 1956.  
Noam Brown and Tuomas Sandholm. Superhuman AI for heads-up no-limit poker: Libratus beats top professionals. Science, pp. eao1733, 2017.  
Noam Brown and Tuomas Sandholm. Solving imperfect-information games via discounted regret minimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 1829-1836, 2019a.  
Noam Brown and Tuomas Sandholm. Superhuman AI for multiplayer poker. Science, pp. eaay2400, 2019b.  
Noam Brown, Adam Lerer, Sam Gross, and Tuomas Sandholm. Deep counterfactual regret minimization. In International Conference on Machine Learning, pp. 793-802, 2019.  
Noam Brown, Anton Bakhtin, Adam Lerer, and Qucheng Gong. Combining deep reinforcement learning and search for imperfect-information games. arXiv preprint arXiv:2007.13544, 2020.  
Murray Campbell, A Joseph Hoane Jr, and Feng-hsiung Hsu. Deep Blue. Artificial intelligence, 134 (1-2):57-83, 2002.  
Arpad E Elo. The rating of chessplayers, past and present. Arco Pub., 1978.  
Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. arXiv preprint arXiv:1802.01561, 2018.  
Andre Ferreira, Henrique Lopes Cardoso, and Luis Paulo Reis. Dipblue: A diplomacy agent with strategic and trust reasoning. In ICAART International Conference on Agents and Artificial Intelligence, Proceedings, 2015.  
Brandon Fogel. To whom tribute is due: The next step in scoring systems, 2020. URL http://windycityweasels.org/wp-content/uploads/2020/04/2020-03-To-Whom-Tribute-Is-Due-The-Next-Step-in-Scoring-Systems.pdf.  
Amy Greenwald, Keith Hall, and Roberto Serrano. Correlated q-learning. In ICML, volume 20, pp. 242, 2003.  
James Hannan. Approximation to bayes risk in repeated play. Contributions to the Theory of Games, 3:97-139, 1957.  
Sergiu Hart and Andreu Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. Econometrica, 68(5):1127-1150, 2000.  
Johannes Heinrich and David Silver. Deep reinforcement learning from self-play in imperfect-information games. arXiv preprint arXiv:1603.01121, 2016.  
Ralf Herbrich, Tom Minka, and Thore Graepel. Trueskill™: a bayesian skill rating system. In Advances in neural information processing systems, pp. 569-576, 2007.

Ronald A Howard. Dynamic programming and markov processes. 1960.  
Junling Hu and Michael P Wellman. Nash q-learning for general-sum stochastic games. Journal of machine learning research, 4(Nov):1039-1069, 2003.  
Stefan J Johansson and Fredrik Hård. Tactical coordination in no-press diplomacy. In International Joint Conference on Autonomous Agents and Multiagent Systems, pp. 423-430, 2005.  
Sarit Kraus and Daniel Lehmann. Diplomat, an agent in a multi agent environment: An overview. In IEEE International Performance Computing and Communications Conference, pp. 434-435. IEEE Computer Society, 1988.  
Sarit Kraus and Daniel Lehmann. Designing and building a negotiating automated agent. Computational Intelligence, 11(1):132-171, 1995.  
Sarit Kraus, Eithan Ephrati, and Daniel Lehmann. Negotiation in a non-cooperative environment. Journal of Experimental & Theoretical Artificial Intelligence, 3(4):255-281, 1994.  
Joel Z Leibo, Vinicius Zambaldi, Marc Lanctot, Janusz Marecki, and Thore Graepel. Multi-agent reinforcement learning in sequential social dilemmas. arXiv preprint arXiv:1702.03037, 2017.  
Adam Lerer and Alexander Peysakhovich. Maintaining cooperation in complex social dilemmas using deep reinforcement learning. arXiv preprint arXiv:1707.01068, 2017.  
Adam Lerer and Alexander Peysakhovich. Learning existing social conventions via observationally augmented self-play. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, pp. 107-114. ACM, 2019.  
Adam Lerer, Hengyuan Hu, Jakob Foerster, and Noam Brown. Improving policies via search in cooperative partially observable games. In AAAI Conference on Artificial Intelligence, 2020.  
Michael L Littman. Friend-or-foe q-learning in general-sum games. In ICML, volume 1, pp. 322-328, 2001.  
Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pp. 1928-1937, 2016.  
Matej Moravčík, Martin Schmid, Neil Burch, Viliam Lisý, Dustin Morrill, Nolan Bard, Trevor Davis, Kevin Waugh, Michael Johanson, and Michael Bowling. Deepstack: Expert-level artificial intelligence in heads-up no-limit poker. Science, 356(6337):508-513, 2017.  
John Nash. Non-cooperative games. Annals of mathematics, pp. 286-295, 1951.  
Philip Paquette, Yuchen Lu, Seton Steven Bocco, Max Smith, O-G Satya, Jonathan K Kummerfeld, Joelle Pineau, Satinder Singh, and Aaron C Courville. No-press diplomacy: Modeling multi-agent gameplay. In Advances in Neural Information Processing Systems, pp. 4474-4485, 2019.  
Yoav Shoham, Rob Powers, and Trond Grenager. Multi-agent reinforcement learning: a critical survey. Web manuscript, 2, 2003.  
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484, 2016.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. Nature, 550(7676):354, 2017.  
David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419):1140-1144, 2018.  
John Maynard Smith. Evolution and the Theory of Games. Cambridge university press, 1982.

Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E Schapire. Fast convergence of regularized learning in games. In Advances in Neural Information Processing Systems, pp. 2989-2997, 2015.  
Peter D Taylor and Leo B Jonker. Evolutionary stable strategies and game dynamics. Mathematical biosciences, 40(1-2):145-156, 1978.  
Gerald Tesauro. TD-Gammon, a self-teaching backgammon program, achieves master-level play. Neural computation, 6(2):215-219, 1994.  
Jason van Hal. Diplomacy AI - Albert, 2013. URL https://sites.google.com/site/diplomacyai/.  
Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michael Mathieu, Andrew Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. Nature, 575(7782):350-354, 2019.  
Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione. Regret minimization in games with incomplete information. In Advances in neural information processing systems, pp. 1729-1736, 2008.
