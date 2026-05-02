# CAN DEEP REINFORCEMENT LEARNING SOLVE ERDOS-SELFRIDGE-SPENCER GAMES?

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep reinforcement learning has achieved many recent successes, but our understanding of its strengths and limitations is hampered by the lack of rich environments in which we can fully characterize optimal behavior, and correspondingly diagnose individual actions against such a characterization. Here we consider a family of combinatorial games, arising from work of Erdos, Selfridge, and Spencer, and we propose their use as environments for evaluating and comparing different approaches to reinforcement learning. These games have a number of appealing features: they are challenging for current learning approaches, but they form (i) a low-dimensional, simply parametrized environment where (ii) there is a linear closed form solution for optimal behavior from any state, and (iii) the difficulty of the game can be tuned by changing environment parameters in an interpretable way. We use these Erdos-Selfridge-Spencer games not only to compare different algorithms, but also to compare approaches based on supervised and reinforcement learning, to analyze the power of multi-agent approaches in improving performance, and to evaluate generalization to environments outside the training set.

# 1 INTRODUCTION

Deep reinforcement learning has seen many remarkable successes over the past few years (Mnih et al., 2015) (Silver et al., 2017). But developing learning algorithms that are robust across tasks and policy representations remains a challenge. Standard benchmarks like MuJoCo and Atari provide rich settings for experimentation, but the specifics of the underlying environments differ from each other in many different ways, and hence determining the principles underlying any particular form of sub-optimal behavior is difficult. Optimal behavior in these environments is generally complex and not fully characterized, so algorithmic success is generally associated with high scores, making it hard to analyze where errors are occurring in any sort of fine-grained sense.

An ideal setting for studying the strengths and limitations of reinforcement learning algorithms would be (i) a simply parametrized family of environments where (ii) optimal behavior can be completely characterized, (iii) the inherent difficulty of computing optimal behavior is tightly controlled by the underlying parameters, and (iv) at least some portions of the parameter space produce environments that are hard for current algorithms. To produce such a family of environments, we look in a novel direction – to a set of two-player combinatorial games with their roots in work of Erdos and Selfridge (Erdos & Selfridge, 1973), and placed on a general footing by Spencer (1994). Roughly speaking, these Erdos-Selfridge-Spencer (ESS) games are games in which two players take turns selecting objects from some combinatorial structure, with the feature that optimal strategies can be defined by potential functions derived from conditional expectations over random future play.

These ESS games thus provide an opportunity to capture the general desiderata noted above, with a clean characterization of optimal behavior and a set of instances that range from easy to very hard as we sweep over a simple set of tunable parameters. We focus in particular on one of the best-known games in this genre, Spencer's attacker-defender game (also known as the "tenure game"; Spencer, 1994), in which — roughly speaking — an attacker advances a set of pieces up the levels of a board, while a defender destroys subsets of these pieces to try prevent any of them from reaching the final level (Figure 1). An instance of the game can be parametrized by two key quantities. The first is the number of levels  $K$ , which determines both the size of the state space and the approximate length of the game; the latter is directly related to the sparsity of win/loss signals as rewards. The second

![](images/29a70901d0055bb6f4cf60ff4496eb732d0d7cbc5ccde0ff08cef33293731a7e.jpg)  
Figure 1: One turn in an ESS Attacker-Defender game. The attacker proposes a partition  $A, B$  of the current game state, and the defender chooses one set to destroy (in this case  $A$ ). Pieces in the remaining set  $(B)$  then move up a level to form the next game state.

![](images/e870cf082a8248d1cd6e1526d5d9474790e3b0024005ededa3c083915670833e.jpg)

![](images/b4399058b7f75cdbff6e1f1df5944d77d6eb26f2088857c7a7fa4ddacc0e2f4f.jpg)

![](images/0e714d33d5064cf803833283288679774313b910bfdbcffdb38d31caa3613573.jpg)

quantity is a potential function  $\phi$ , whose magnitude characterizes whether the instance favors the defender or attacker, and how much "margin of error" there is in optimal play.

The environment therefore allows us to study learning by the defender, or by the attacker, or in a multi-agent formulation where the defender and attacker are learning concurrently. Because we have a move-by-move characterization of optimal play, we can go beyond simple measures of reward based purely on win/loss outcomes and use supervised learning techniques to pinpoint the exact location of the errors in a trajectory of play. In the process, we are able to develop insights about the robustness of solutions to changes in the environment. These types of analyses have been long-standing goals, but they have generally been approached much more abstractly, given the difficulty in characterizing step-by-step optimally in non-trivial environments such as this one.

The main contributions of this work are thus the following:

1. The development of these combinatorial games as environments for studying the behavior of reinforcement learning algorithms, with sensitive control over the difficulty of individual instances using a small set of natural parameters.  
2. A comparison of the performance of an agent trained using deep RL to the performance of an agent trained using supervised learning on move-by-move decisions. Exploiting the fact that we can characterize optimal play at the level of individual moves, we find an intriguing phenomenon: while the supervised learning agent is, not surprisingly, more accurate on individual move decisions than the deep RL agent, the deep RL agent generates a higher rate of reward than the supervised learning agent.  
3. An investigation of the way in which the success of one of the two players (defender or attacker) in training turns out to depend crucially on the algorithm being used to implement the other player. We explore properties of this other player's algorithm, and also properties of multitagent learning, that lead to more robust policies.

This is a largely empirical paper, building on a theoretically grounded environment derived from a combinatorial game. We present learning and generalization experiments for a variety of commonly used model architectures and learning algorithms. We aim to show that despite the simple structure of the game, it provides both significant challenges for standard reinforcement learning approaches and a number of tools for precisely understanding those challenges.

# 2 ERDOS-SELFRIDGE-SPENCER GAME

We first introduce the Erdos-Selfridge-Spencer Attacker Defender Games, a family of games that has two particularly attractive properties for being a test bed for deep reinforcement learning: the ability to continuously vary the difficulty of the environment through two parameters, and the existence of a closed form solution that is expressible as a linear model.

# 2.1 ERDOS-SELFRIDGE-SPENCER (ESS) ATTACKER DEFENDER GAME

An Attacker-Defender game involves two players: an attacker who moves pieces, and a defender who destroys pieces. An instance of the game has  $K + 1$  levels, (from 0 to  $K$ ) and  $N$  pieces that are initialized across these levels. The attacker's goal is to get at least one of their pieces to level  $K$ , and

the defender's is to destroy all  $N$  pieces before this can happen. Each turn, the attacker proposes a partition  $A, B$  of the pieces still in play. The defender then chooses one of the sets to destroy and remove from play. All pieces in the other set are moved up a level. The game ends when either one or more pieces reach level  $K$ , or when all pieces are destroyed. Figure 1 shows one turn of play.

With this setup, varying the number of levels  $K$  or the number of pieces  $N$  changes the difficulty for the attacker or the defender. One of the most striking aspects of the Attacker-Defender game is that we can make this tradeoff precise, and en route to doing so, also identify a linear optimal policy. We start with a special case, where we can directly think of the game difficulty in terms of the number of levels  $K$  and the number of pieces  $N$ .

Theorem 1. Consider an instance of the Attacker-Defender game with  $K$  levels and  $N$  pieces, with all  $N$  pieces starting at level 0. Then if  $N < 2^K$ , the defender can always win.

The proof (Spencer, 1994) uses Erdos's probabilistic method and is as follows: for any attacker strategy, assume the defender plays randomly. Let  $T$  be a random variable for the number of pieces that reach level  $K$ . Then  $T = \sum T_{i}$  where  $T_{i}$  is the indicator that piece  $i$  reaches level  $K$ .

But then  $E[T] = \sum_{i} E[T_i] = \sum_{i} 2^{-K}$ ; as the defender is playing randomly, any piece has probability  $1/2$  of advancing a level and  $1/2$  of being destroyed. As all the pieces start at level 0, they must advance  $K$  levels to reach the top, which happens with probability  $2^{-K}$ . But now, by choice of  $N$ , we have that  $\sum_{i} 2^{-K} = N 2^{-K} < 1$ . Since  $T$  is an integer random variable,  $E[T] < 1$  implies that the distribution of  $T$  has nonzero mass at 0 - in other words there is some set of choices for the defender that guarantees destroying all pieces.

This proof as given is an 'existential' one, but we can turn it into a concrete optimal strategy. Extending the argument above, we note that a piece at level  $l$  has a  $2^{-(K - l)}$  chance of survival. We can thus define a potential function on states:

Definition 1. Potential Function: Given a game state as a  $K$  dimensional vector  $S = (n_{1},\dots,n_{K})$ , with  $n_i$  the number of pieces at level  $i$ , we define the potential of the state as  $\phi(S) = \sum_{i=1}^{K} n_i 2^{-(K-i)}$ .

Note that this is a linear function on the input state, expressible as  $\phi(S) = w^T \cdot S$  for  $w$  a vector with  $w_l = 2^{-(K - l)}$ . We can then in fact recast Theorem 1 as follows:

Theorem 2. Consider an instantiation of the Attacker-Defender game with  $K$  levels and  $N$  pieces, with pieces placed anywhere on the board, and let the initial state be  $S_0$ . Then

(a) If  $\phi(S_0) < 1$ , the defender can always win  
(b) If  $\phi(S_0) \geq 1$ , the attacker can always win.

One way to prove this is by directly extending the proof of Theorem 1, with  $E[T] = \sum_{i}E[T_i] = \sum_{i}2^{-(K - i_l)}$  where  $i_{l}$  is the level of piece  $i$ . After noting that  $\sum_{i}2^{-(K - i_{l})} = \phi (S_{0}) < 1$  by our definition of the potential function and choice of  $S_0$ , we finish off as in Theorem 1.

This definition of the potential function gives a natural, concrete strategy for the defender: the defender simply destroys whichever of  $A, B$  has higher potential. If  $\phi(S_0) < 1$ , then this strategy guarantees that any subsequent state  $S$  will also have  $\phi(S) < 1$ . Assume without loss of generality that  $\phi(B) \leq \phi(A)$ . The defender will spare set  $B$ .

Since  $\phi(B) \leq \phi(A)$  and  $\phi(A) + \phi(B) = \phi(S) < 1$ , the next state has potential  $2\phi(B)$  (double the potential of  $B$  as all pieces move up a level) which is also less than 1.

If  $\phi(S_0) \geq 1$ , we can devise a similar optimal strategy for the attacker. The attacker picks two sets  $A, B$  such that both has potential  $\geq 1/2$ . The fact that this can be done is shown in Theorem 3, and in Spencer (1994). Then regardless of which of  $A, B$  is destroyed, the other, whose pieces all move up a level, doubles its potential, and thus all subsequent states  $S$  maintain  $\phi(S) \geq 1$ , resulting in an eventual win for the attacker.

# 3 RELATED WORK

The Atari benchmark (Mnih et al., 2015) is a well known set of tasks, ranging from easy to solve (Breakout, Pong) to very difficult (Montezuma's Revenge). Duan et al. (2016) proposed a set of

continuous environments, implemented in the MuJoCo simulator Todorov et al. (2012). An advantage of physics based environments is that they can be varied continuously by changing physics parameters (Rajeswaran et al., 2016), or by randomizing rendering (Tobin et al., 2017). Deepmind Lab (Beattie et al., 2016) is a set of 3D navigation based environments. OpenAI Gym (Brockman et al., 2016) contains both the Atari and MuJoCo benchmarks, as well as classic control environments like Cartpole (Stephenson, 1909) and algorithmic tasks like copying an input sequence. The difficulty of algorithmic tasks can be easily increased by increasing the length of the input. Our proposed benchmark merges properties of both the algorithmic tasks and physics-based tasks, letting us increase difficulty by discrete changes in length or continuous changes in potential.

# 4 DEEP REINFORCEMENT LEARNING ON THE ATTACKER-DEFENDER GAME

From Section 2, we see that the Attacker-Defender games are a family of environments with a difficulty knob that can be continuously adjusted through the start state potential  $\phi(S_0)$  and the number of levels  $K$ . In this section, we describe a set of baseline results on Attacker-Defender games that motivate the exploration in the remainder of this paper. Remarkably, while the optimal policy can be expressed with a linear model, we find that in practice deep networks improve performance significantly (Figure 15 in the Appendix). We thus focus our attention on neural network models.

We set up the Attacker-Defender environment as follows: the game state is represented by a  $K + 1$  dimensional vector for levels 0 to  $K$ , with coordinate  $l$  representing the number of pieces at level  $l$ . For the defender agent, the input is the concatenation of the partition  $A, B$ , giving a  $2(K + 1)$  dimensional vector. The game start state  $S_0$  is initialized randomly from a distribution over start states of a certain potential. The policy is a fully connected neural network with two hidden layers of width 300.

# 4.1 DEFENDER AGENT PERFORMANCE WITH VARYING DIFFICULTY

We first look at training a defender agent against an attacker that randomly chooses between (mostly) playing optimally, and (occasionally) playing suboptimally, with the Disjoint Support Strategy. This strategy unevenly partitions the occupied levels between  $A$ ,  $B$  so that one set has higher potential than the other, with the proportional difference between the two sets being sampled randomly. Note that this strategy gives rise to very different states  $A$ ,  $B$  (uneven potential, disjoint occupied levels) than the optimal strategy, and we find that the model learns a much more generalizable policy when mixing between the two (Section 6).

When testing out reinforcement learning, we have two choices of difficulty parameters. The potential of the start state,  $\phi(S_0)$ , changes how optimally the defender has to play, with values close to 1 giving much less leeway for mistakes in valuing the two sets. Changing  $K$ , the number of levels, directly affects the sparsity of the reward, with higher  $K$  resulting in longer games and less feedback. Additionally,  $K$  also greatly increases the number of possible states and game trajectories (see Theorem 4).

We evaluate Proximal Policy Optimization (PPO) (Schulman et al., 2017), Advantage Actor Critic (A2C) (Mnih et al., 2016), and Deep Q-Networks (DQN) (Mnih et al., 2015), using the OpenAI Baselines implementations (Hesse et al., 2017). Algorithms are evaluated on varying start state potential and  $K$ . Each algorithm is run with 3 random seeds, and in all plots we show minimum, mean, and maximum performance.

Results are shown in Figures 2, 3. Note that all algorithms show variation in performance across different settings of potentials and  $K$ , and show noticeable drops in performance with harder difficulty settings. When varying potential in Figure 2 both PPO and A2C show larger variance than DQN, though PPO mostly matches or beats DQN in performance. When varying  $K$ , PPO shows less variance than DQN. A2C shows the greatest variance and worst performance out of all three methods.

# 5 SUPERVISED LEARNING

One remarkable aspect of the Attacker-Defender game is that not only do we have an easily expressible optimal policy, but we know the ground truth on a per move basis. We can thus compare RL to a

![](images/4a617b3e708117cfd14fd723b59bc86648f781d7e417afee1ce6a9f3c9509d3f.jpg)  
Defender trained with PPO for varying potential,  $K = 15$

![](images/871b913d1dcfb8c87196192c3575d7c8d00965534ca0e620bfd177fca34809be.jpg)  
Defender trained with PPO for varying potential,  $K = 20$  
Training steps

![](images/cdae55752ac04cd6ad362792c2437003c986767dd9ba3f7466ce4f7526fa9d2a.jpg)  
Defender trained with A2C for varying potential,  $K = 15$

![](images/5ea2bd6a6b06be5b760d6ad4ffb7afb3c74f0f80e44863f6a7b3721689dc1879.jpg)  
Defender trained with A2C for varying potential,  $K = 20$  
Training steps  
Figure 2: Training defender agent with PPO, A2C and DQN for varying values of potentials. The first row shows varying potential for fixed  $K$ , and the second row shows varying  $K$  for fixed potential. For lower  $K$ , DQN performs relatively consistently across different potential values, though not quite matching PPO – left and right panes, row 2. A2C tends to fare worse than both PPO and DQN.

![](images/eb71878ee3a90598d75ab97d5efa583c6c87c884b17efe0b2c2d0aaca7031584.jpg)  
Defender trained with DQN for varying potential,  $K = 15$

![](images/f5189e4579476a229515f9df06f9a274099b6fbe6e7b00caf774850ac37b93ee.jpg)  
Defender trained with DQN for varying potential,  $K = 20$  
Training steps

![](images/9f51d6ce4e64c351941950f9ce4f7e069cce4cc88de2f8bdd49bcd73cc22f0b0.jpg)  
Defender trained with PPO for varying K, potential 0.95

![](images/b621aeabed8d909389e7d00bf14a758a12c79af8985977c294068fb90641cacc.jpg)  
Defender trained with PPO for varying K, potential 0.99  
Figure 3: Training defender agent with PPO, A2C and DQN for varying values of  $K$ . The first two rows show varying potential for fixed  $K$ , and the second two rows show varying  $K$  for fixed potential. All three algorithms show a noticeable variation in performance over different difficulty settings, though we note that PPO seems to be more robust to longer episodes. A2C tends to fare worse than both PPO and DQN.

![](images/958063cb1200ba06f6227df22822ddd38bd44f91e6a82eedf08869eee2374fc6.jpg)  
Defender trained with A2C for varying K, potential 0.95

![](images/d1268fae0b2c09a4719bfb80a565abaf0003e53f7fc1e17bda647e52ffbaef56.jpg)  
Defender trained with A2C for varying K, potential 0.99

![](images/1dcb10ce926c20ac746412f91110a1c3b0acaa1ac553e68cfde162467bcd48d4.jpg)  
Defender trained with DQN for varying K, potential 0.95

![](images/47aec7ee5049cd3459a6cef7c4f04ae7eb7b2bdf8620b37933cde600e1b5e69c.jpg)  
Defender trained with DQN for varying K, potential 0.99

![](images/5ef707b044071db7f9e2d91517c5567107894bd29fabe59cf23dfc9602deee0c.jpg)  
RL vs Supervised Learning on Rewards and Per Move Performance for varying K

![](images/757592fade4221e9a4c5db99c1b6df48d17770b99855e8bd4e88a46d330e116e.jpg)  
Supervised Learning and RL performance on per move correct and rewards,  $K = 20$  
Figure 4: Plots showing reward and correct actions for RL vs Supervised Learning. The left plot is performance of the final models over varying  $K$ , and the right plot is performance of the  $K = 20$  models during training. The RL policy consistently achieves larger reward, and the gap grows as  $K$  increases. However, even as its performance decreases, the supervised network continues to achieve high per move accuracy.

![](images/b5067ee44b8e6edda179c85a2c004c09bf03abca4f043b780756114156de36e3.jpg)  
Proportion of Correct Actions vs distance to End Game  
Figure 5: Proportion of correct actions for RL and Supervised Learning as a function of the moves remaining before the end of the game. We see that RL is more accurate than supervised learning at predicting the right action for the final couple of moves, and then drops quickly to a constant, whereas supervised learning is less accurate right at the very end and drops more slowly but much further, having lower accuracy than RL for many of the earlier moves.

Supervised Learning setup, where we classify the correct action on a large set of sampled states. To carry out this test in practice, we first train a defender policy with reinforcement learning, saving all observations seen to a dataset. We then train a supervised network (with the same architecture as the defender policy) to classify the optimal action. This ensures both methods see the same number of unique data points, keeping the comparison fair. We then test the supervised network on how well it can play. The results, shown in Figure 4 are surprising. Reinforcement learning is better at playing the game, but does worse at predicting optimal moves.

This contrast forms an interesting counterpart to recent findings of Silver et al. (2017), who in the context of Go also compared reinforcement learning to supervised approaches. A key distinction is that their supervised work was relative to non-optimal human play, whereas in our domain we are able to compare to provably optimal play. We conjecture that reinforcement learning is learning to focus most on moves that matter for winning. To investigate this phenomenon, we compare the per move accuracy of reinforcement learning and supervised learning based on distance of the move from the end of the game. We find that RL is better at the final couple of moves, and then consistently

![](images/7582b12d466a38b958336b014d31b6605b1016525c23da4e994839973a57c49d.jpg)  
Figure 6: Plot showing overfitting to opponent strategies. A defender agent is trained on the optimal attacker, and then tested on (a) another optimal attacker environment (b) the disjoint support attacker environment. The left pane shows the resulting performance drop when switching to testing on the same opponent strategy as in training to a different opponent strategy. The right pane shows the result of testing on an optimal attacker vs a disjoint support attacker during training. We see that performance on the disjoint support attacker converges to a significantly lower level than the optimal attacker.

![](images/13542a65cf5f8b7da877ba55469a6feaec58f8c15522b74980a59f36b4461c6a.jpg)

better in most of the earlier parts of the game. Supervised learning beats RL for about one third of the latter part of the game, likely where potentials are easier to differentiate, but at the cost of better play earlier in the game.

# 6 GENERALIZATION AND MULTIAGENT LEARNING

Returning to our RL Defender Agent, we study the robustness of its learned policy. So far, we have trained the defender against a randomized but hard coded attacker, which does not guarantee generalization to all attackers. We investigate this in Figure 6, where we first train a defender agent on the optimal attacker, then test on the disjoint support attacker. We notice a large drop in performance when switching from the optimal attacker to the disjoint support attacker. As we know there exists an optimal policy which generalizes perfectly across all attacker strategies, this result suggests that the defender is overfitting to the particular attacker strategy.

# 6.1 TRAINING AN ATTACKER AGENT

One way to mitigate this overfitting issue is to set up a method of also training the attacker, with the goal of training the defender against a fixed learned attacker, or even better, in the multiagent setting. However, determining the correct setup to train the attacker agent first requires devising a tractable parametrization of the action space. A naive implementation of the attacker would be to have the policy output how many pieces should be allocated to  $A$  for each of the  $K + 1$  levels (as described in Spencer (1994)). This can grow exponentially in  $K$ , which is clearly impractical. To address this, we first prove a theorem that enables us to show that we can parametrize an optimal attacker with a much smaller action space.

Theorem 3. For any Attacker-Defender game with  $K$  levels, start state  $S_0$  and  $\phi(S_0) \geq 1$ , there exists a partition  $A, B$  such that  $\phi(A) \geq 0.5$ ,  $\phi(B) \geq 0.5$ , and for some  $l$ ,  $A$  contains pieces of level  $i > l$ , and  $B$  contains all pieces of level  $i < l$ .

Proof. For each  $l \in \{1, 2, \dots, K + 1\}$ , let  $A_{l}$  be the set of all pieces from levels  $K$  to  $l$ , with  $A_{K + 1} = \emptyset$ . We have  $\phi(A_{i + 1}) \leq \phi(A_{i})$ ,  $\phi(A_{K + 1}) = 0$ , and  $\phi(A_{1}) = \phi(S_{0}) \geq 1$ . Thus, there exists an  $l$  such that  $\phi(A_{l + 1}) < 0.5$  and  $\phi(A_{l}) > 0.5$ .

![](images/620ad058b8747b7e4416b40e033b81c27b1d622273f0bd963f310d1ab76aea3d.jpg)

![](images/5c9f6ab57977aeb806e13a01490b6a4625505de2c2304f584b4a8e9ff6cc4914.jpg)

![](images/7e7c52f2e0c8fabf9c1c31161664d3c1648e75dbe74bd77c6f08ac192abad3dc.jpg)

![](images/9b15e44f0a65cf7fb3ddaa22140de31f1f19b954fade0ef25708f9aaf1911ea2.jpg)  
Figure 7: Performance of PPO and A2C on training the attacker agent for different difficulty settings. DQN performance was very poor (reward  $< -0.8$  at  $K = 5$  with best hyperparams). We see much greater variation of performance with changing  $K$ , which now affects the sparseness of the reward as well as the size of the action space. There is less variation with potential, but we see a very high performance variance (top right pane) with lower (harder) potentials.

![](images/f9c7fd720d408895178d5f9ccdd25f196c52f756496439e03ef22373a978cf52.jpg)

![](images/d19cfa616e465fe5c29277cef2737c780cc4fc43c96726b416ce3e533c2a3cd1.jpg)

Since  $A_{l}$  only contains pieces from levels  $K$  to  $l + 1$ , potentials  $\phi(A_{l + 1})$  and  $\phi(A_{l})$  are both integer multiples of  $2^{-(K - l + 1)}$ , the value of a piece in level  $l$ . Letting  $\phi(A_{l + 1}) = n \cdot 2^{-(K - l + 1)}$  and  $\phi(A_{l}) = m \cdot 2^{-(K - l + 1)}$ , we are guaranteed that level  $l$  has  $m - n$  pieces, and that we can move  $k < m - n$  pieces from  $A_{l + 1}$  to  $A_{l}$  such that the potential of the new set equals 0.5.

This theorem gives a different attacker parametrization. The attacker outputs a level  $l$ . The environment assigns all pieces before level  $l$  to  $A$ , all pieces after level  $l$  to  $B$ , and splits level  $l$  among  $A$  and  $B$  to keep the potentials of  $A$  and  $B$  as close as possible. Theorem 3 guarantees the optimal policy is representable, and the action space linear in  $K$  instead of exponential in  $K$ .

With this setup, we train an attacker agent against the optimal defender with PPO, A2C, and DQN. The DQN results were very poor, and so we show results for just PPO and A2C. In both algorithms we found there was a large variation in performance when changing  $K$ , which now affects both reward sparsity and action space size. We observe less outright performance variability with changes in potential for small  $K$  but see an increase in the variance (Figure 7).

# 6.2 LEARNING THROUGH MULTIAGENT PLAY

With this attacker training, we can now look at learning in a multiagent setting. We first explore the effects of varying the potential and  $K$  as shown in Figure 8. Overall, we find that the attacker fares worse in multiagent play than in the single agent setting. In particular, note that in the top left pane of Figure 8, we see that the attacker loses to the defender even with  $\phi(S_0) = 1.1$  for  $K = 15$ . We can compare this to Figure 7 where with PPO, we see that with  $K = 15$ , and potential 1.1, the single agent attacker succeeds in winning against the optimal defender.

![](images/6f9287acb0e1c583013f2b6b5021fb13642e37ba3ef950412752dea668fd15a4.jpg)

![](images/2ae1006f60eda5935d4d9104a1b2f16a57cd432c8735ab09f7dd5e276c0d62d8.jpg)  
Figure 8: Performance of attacker and defender agents when learning in a multiagent setting. In the top panes, solid lines denote attacker performance and dashed lines defender performance. In the bottom panes, dashed lines are attacker and solid lines defender. The rewards are reflected around 0 as the Attacker-Defender game is a zero sum game. The sharp changes in performance correspond to the times we switch which agent is training. We note that the defender performs much better in the multiagent setting: comparing the top and bottom left panes, we see far more variance and lower performance of the attacker compared to the defender performance below. Furthermore, the attacker loses to the defender for potential 1.1 at  $K = 15$ , despite winning against the optimal defender in Figure 7. We also see (right panes) that the attacker has higher variance and sharper changes in its performance even under conditions when it is guaranteed to win.

![](images/74100e187f9f4911a54983d4ca3fab533159b6ff7d61bf61dac62b98e1211c13.jpg)

![](images/faee49307ccc1ef75045303e4128c2d8ac5e4dbda0e823407d9c0d548e897f42.jpg)

![](images/39fa87863e3907130b1a6e6350e25bb0dc1025b6f3ac62c7e5703b12275e7142.jpg)  
Figure 9: Results for generalizing to different attacker strategies with single agent defender and multiagent defender. The left pane shows a single agent defender trained on the optimal attacker and then tested on the disjoint support attacker and a multiagent defender also tested on the disjoint support attacker for different values of  $K$ . We see that multiagent defender generalizes better to this unseen strategy than the single agent defender.

![](images/950e02188ebe21d9cf78c74ae523722e2fd6a3343125a64c2a5f8382c6b5b251.jpg)

# 6.3 SINGLE AGENT AND MULTIAGENT GENERALIZATION ACROSS OPPONENT STRATEGIES

Finally, we return again to our defender agent, and test generalization between the single and multiagent settings. We train a defender agent in the single agent setting against the optimal attacker, and test on an attacker that only uses the Disjoint Support strategy. We also test a defender trained in the multiagent setting (which has never seen any hardcoded strategy of this form) on the Disjoint Support attacker. The results are shown in Figure 9. We find that the defender trained as part of a multiagent setting generalizes noticeably better than the single agent defender.

# 7 CONCLUSION

In this paper, we have proposed Erdos-Selfridge-Spencer games as rich environments for investigating reinforcement learning, exhibiting continuously tunable difficulty and an exact combinatorial characterization of optimal behavior. We have demonstrated that algorithms can exhibit wide variation in performance as we tune the game's difficulty, and we use the characterization of optimal behavior to expose intriguing contrasts between performance in supervised learning and reinforcement learning approaches. Having reformulated the results to enable a trainable attacker, we have also been able to explore insights on overfitting, generalization, and multiagent learning. We also develop further results in the Appendix, including an analysis of catastrophic forgetting, generalization across different values of the game's parameters, and a method for investigating measures of the model's confidence. We believe that this family of combinatorial games can be used as a rich environment for gaining further insights into deep reinforcement learning.

# REFERENCES

Charles Beattie, Joel Z Leibo, Denis Teplyashin, Tom Ward, Marcus Wainwright, Heinrich Kuttler, Andrew Lefrancq, Simon Green, Víctor Valdés, Amir Sadik, et al. Deepmind lab. arXiv preprint arXiv:1612.03801, 2016.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym, 2016.  
Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning, pp. 1329-1338, 2016.

Paul Erdos and John Selfridge. On a combinatorial game. Journal of Combinatorial Theory, 14: 298-301, 1973.  
Christopher Hesse, Matthias Plappert, Alec Radford, John Schulman, Szymon Sidor, and Yuhuai Wu. Openai baselines. https://github.com/openai/baselines, 2017.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, Feb 2015. ISSN 0028-0836. URL http://dx.doi.org/10.1038/nature14236.  
Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. arXiv preprint arxiv:1602.01783, 2016.  
Aravind Rajeswaran, Sarvjeet Ghotra, Sergey Levine, and Balaraman Ravindran. Epopt: Learning robust neural network policies using model ensembles. arXiv preprint arXiv:1610.01283, 2016.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arxiv:1707.06347, 2017.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. Mastering the game of go without human knowledge. Nature, 550(7676):354-359, Oct 2017. ISSN 0028-0836. URL http://dx.doi.org/10.1038/nature24270.  
Joel Spencer. Randomization, derandomization and antirandomization: Three games. Theoretical Computer Science., 131:415-429, 09 1994.  
Andrew Stephenson. Lxxi. on induced stability. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 17(101):765-766, 1909.  
Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. arXiv preprint arXiv:1703.06907, 2017.  
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on, pp. 5026-5033. IEEE, 2012.

![](images/9b09755619cb03cb7dfd263a9de4284baa0dcb93972b140342f3bcde0b6fbb93.jpg)  
Figure 10: On the left we train on different potentials and test on potential 0.99. We find that training on harder games leads to better performance, with the agent trained on the easiest potential generalizing worst and the agent trained on a harder potential generalizing best. This result is consistent across different choices of test potentials. The right pane shows the effect of training on a larger  $K$  and testing on smaller  $K$ . We see that performance appears to be inversely proportional to the difference between the train  $K$  and test  $K$ .

![](images/f759f51897635ac05fd82d175f65b25b0f86f23a7c5e8a0a8f134fa8bdb8343e.jpg)
