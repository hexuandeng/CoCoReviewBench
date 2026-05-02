# DISCOVERING DIVERSE MULTI-AGENT STRATEGIC BEHAVIOR VIA REWARD RANDOMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose a simple, general and effective technique, Reward Randomization for discovering diverse strategic policies in complex multi-agent games. Combining reward randomization and policy gradient, we derive a new algorithm, Reward-Randomized Policy Gradient (RPG). RPG is able to discover multiple distinctive human-interpretable strategies in challenging temporal trust dilemmas, including grid-world games and a real-world game Agario, where multiple equilibria exist but standard multi-agent policy gradient algorithms always converge to a fixed one with a sub-optimal payoff for every player even using state-of-the-art exploration techniques. Furthermore, with the set of diverse strategies from RPG, we can (1) achieve higher payoffs by fine-tuning the best policy from the set; and (2) obtain an adaptive agent by using this set of strategies as its training opponents.

# 1 INTRODUCTION

Games have been a long-standing benchmark for artificial intelligence, which prompts persistent technical advances towards our ultimate goal of building intelligent agents like humans, from Shannon's initial interest in Chess (Shannon, 1950) and IBM DeepBlue (Campbell et al., 2002), to the most recent deep reinforcement learning breakthroughs in Go (Silver et al., 2017), Dota II (OpenAI et al., 2019) and Starcraft (Vinyals et al., 2019). Hence, analyzing and understanding the challenges in various games also become critical for developing new learning algorithms for even harder challenges.

Most recent successes in games are based on decentralized multi-agent learning (Brown, 1951; Singh et al., 2000; Lowe et al., 2017; Silver et al., 2018), where agents compete against each other and optimize their own rewards to gradually improve their strategies. In this framework, Nash Equilibrium (NE) (Nash, 1951), where no player could benefit from altering its strategy unilaterally, provides a general solution concept and serves as a goal for policy learning and has attracted increasingly significant interests from AI researchers (Heinrich & Silver, 2016; Lanctot et al., 2017; Foerster et al., 2018; Kamra et al., 2019; Han & Hu, 2019; Bai & Jin, 2020; Perolat et al., 2020): many existing works studied how to design practical multi-agent reinforcement learning (MARL) algorithms that can provably converge to an NE in Markov games, particularly in the zero-sum setting.

Despite the empirical success of these algorithms, a fundamental question remains largely unstudied in the field: even if an MARL algorithm converges to an NE, which equilibrium will it converge to? The existence of multiple NEs is extremely common in many multi-agent games. Discovering as many NE strategies as possible is particularly important in practice not only because different NEs can produce drastically different payoffs but also because when facing unknown players who are trained to play an NE strategy, we can gain advantage by identifying which NE strategy the opponent is playing and choosing the most appropriate response. Unfortunately, in many games where multiple distinct NEs exist, the popular decentralized policy gradient algorithm (PG), which has led to great successes in numerous games including Dota II and Stacraft, always converge to a particular NE with non-optimal payoffs and fail to explore more diverse modes in the strategy space.

Consider an extremely simple example, a 2-by-2 matrix game Stag-Hunt (Rousseau, 1984; Skyrms, 2004), where two pure strategy NEs exist: a "risky" cooperative equilibrium with the highest payoff for both agents and a "safe" non-cooperative equilibrium with strictly lower payoffs. We show, from both theoretical and practical perspectives, that even in this simple matrix-form game, PG fails to discover the high-payoff "risky" NE with high probability. The intuition is that the neighborhood that makes policies converge to the "risky" NE can be substantially small comparing to the entire

policy space. Therefore, an exponentially large number of exploration steps are needed to ensure PG discovers the desired mode. We propose a simple technique, Reward Randomization (RR),

which can help PG discover the "risky" cooperation strategy in the stag-hunt game with theoretical guarantees. The core idea of RR is to directly perturb the reward structure of the multi-agent game of interest, which is typically low-dimensional. RR directly alters the landscape of different strategy modes in the policy space and therefore makes it possible to easily discover novel behavior in the perturbed game (Fig. 1). We call this new PG variant Reward-Rando

![](images/5b91daae1c4c388c0da6ac19f377a28af2d26aad9cee61d178397b33293fb551.jpg)  
Figure 1: Intuition of Reward Randomization

To further illustrate the effectiveness of RPG, we introduce three Markov games – two gridworld games and a real-world online game Agario. All these games have multiple NEs including both “risky” cooperation strategies and “safe” non-cooperative strategies. We empirically show that even with state-of-the-art exploration techniques, PG fails to discover the “risky” cooperation strategies. In contrast, RPG discovers a surprisingly diverse set of human-interpretable strategies in all these games, including some non-trivial emergent behavior. Importantly, among this set are policies achieving much higher payoffs for each player compared to those found by PG. This “diversity-seeking” property of RPG also makes it feasible to build adaptive policies: by re-training an RL agent against the diverse opponents discovered by RPG, the agent is able to dynamically alter its strategy between different modes, e.g., either cooperate or compete, w.r.t. its test-time opponent's behavior.

# 2 A MOTIVATING EXAMPLE: STAG HUNT

The stag-hunt game was originally introduced in Rousseau's work, "A discourse on inequality" (Rousseau, 1984): a group of hunters are tracking a big stag silently; now a hare shows up, each hunter should decide whether to keep tracking the stag or kill the hare immediately. This leads to the 2-by-2 matrix-form stag-hunt game in Tab. 1 with two actions for each agent, Stag (S) and Hare (H). There are two pure strategy NEs: the Stag NE, where both agents choose S and receive a high payoff  $a$  (e.g.,  $a = 4$ ) both agents choose H and receive a lower payoff  $d$  (e.g.,  $d = 1$ ). The Stag one agent defects, they still receives a decent reward  $b$  (e.g.,  $b = 3$ ) for each the other agent with an S action may suffer from a big loss  $c$  for being hurt

Formally, let  $A = \{\mathrm{S},\mathrm{H}\}$  denote the action space,  $\pi_i(\theta_i)$  denote the policy for agent  $i$  ( $i \in \{1,2\}$ ) parameterized by  $\theta_{i}$ , i.e.,  $P[\pi_i(\theta_i) = \mathrm{S}] = \theta_i$  and  $P[\pi_i(\theta_i) = \mathrm{H}] = 1 - \theta_i$ , and  $R(a_{1},a_{2};i)$  denote the payoff for agent  $i$  when agent 1 takes action  $a_1$  and agent 2 takes action  $a_2$ . Each agent  $i$  optimizes its expected utility  $U_{i}(\pi_{1},\pi_{2}) = \mathbb{E}_{a_{1}\sim \pi_{1},a_{2}\sim \pi_{2}}[R(a_{1},a_{2};i)]$ . Using the standard policy gradient algorithm, a typical learning procedure is to repeatedly take the following two steps until convergence<sup>1</sup>: (1) estimate gradient  $\nabla_{i} = \nabla U_{i}(\pi_{1},\pi_{2})$  via self-play; (2) update the policies by  $\theta_{i} \gets \theta_{i} + \alpha \nabla_{i}$  with learning rate  $\alpha$ . Although PG is widely used in practice, the following theorem shows in certain scenarios, unfortunately, the probability that PG converges to the Stag NE is low.

<table><tr><td></td><td>Stag</td><td>Hare</td></tr><tr><td>Stag</td><td>a, a</td><td>c, b</td></tr><tr><td>Hare</td><td>b, c</td><td>d, d</td></tr></table>

Table 1: The stag-hunt game,  $a > b \geq d > c$ .  
and the Hare NE, where NE is "risky" because if t ing the hare alone while gry (e.g.,  $c = -10$

![](images/c023a735c9786f747560864fa2eab12eceea996c75497e2c3c0997c6cbc95439.jpg)  
Figure 2: PPO in stag hunt, with  $a = 4$ ,  $b = 3$ ,  $d = 1$  and various  $c$  (10 seeds).

Theorem 1. Suppose  $a - b = \epsilon (d - c)$  for some  $0 < \epsilon < 1$  and initialize  $\theta_{1},\theta_{2}\sim \mathrm{Unif}[0,1]$ . Then the probability that PG discovers the high-payoff NE is upper bounded by  $O(\epsilon)$ .

Theorem 1 shows when the risk is high (i.e.,  $c$  is low), then the probability of finding the Stag NE via PG is very low. Note this theorem applies to random initialization, which is standard in RL.

Remark: One needs at least  $N = \overline{\Omega}\left(\frac{1}{\epsilon}\right)$  restarts to ensure a constant success probability.

Fig. 2 shows empirical studies: we select 4 value assignments, i.e.,  $c \in \{-5, -20, -50, -100\}$  and  $a = 4$ ,  $b = 3$ ,  $d = 1$ , and run a state-of-the-art PG method, proximal policy optimization (PPO) (Schulman et al., 2017), on these games. The Stag NE is rarely reached, and, as  $c$  becomes smaller, the probability of finding the Stag NE significantly decreases. Peysakhovich & Lerer (2018b) provided a theorem of similar flavor without analyzing the dynamics of the learning algorithm whereas we explicitly

characterize the behavior of PG. They studied a prosocial reward-sharing scheme, which transforms the reward of both agents to  $R(a_{1},a_{2};1) + R(a_{1},a_{2};2)$ . Reward sharing can be viewed as a special case of our method and, as shown in Sec. 5, it is insufficient for solving complex temporal games.

# 2.1 REWARD RANDOMIZATION IN THE MATRIX-FORM STAG-HUNT GAME

Thm. 1 suggests that the utility function  $R$  highly influences what strategy PG might learn. Taking one step further, even if a strategy is difficult to learn with a particular  $R$ , it might be easier in some other function  $R'$ . Hence, if we can define an appropriate space  $\mathcal{R}$  over different utility functions and draw samples from  $\mathcal{R}$ , we may possibly discover desired novel strategies by running PG on some sampled utility function  $R'$  and evaluating the obtained policy profile on the original game with  $R$ . We call this procedure Reward Randomization (RR).

Concretely, in the stag-hunt game,  $R$  is parameterized by 4 variables  $(a_{R}, b_{R}, c_{R}, d_{R})$ . We can define a distribution over  $\mathbb{R}^4$ , draw a tuple  $R' = (a_{R'}, b_{R'}, c_{R'}, d_{R'})$  from this distribution, and run PG on  $R'$ . Denote the original stag-hunt game where the Stag NE is hard to discover as  $R_0$ . Reward randomization draws  $N$  perturbed tuples  $R_1, \ldots, R_N$ , runs PG on each  $R_i$ , and evaluates each of the obtained strategies on  $R_0$ . The theorem below shows it is highly likely that the population of the  $N$  policy profiles obtained from the perturbed games contains the Stag NE strategy.

Theorem 2. For any Stag-Hunt game, suppose in the  $i$ -th run of RR we randomly generate  $a_{R_i}, b_{R_i}, c_{R_i}, d_{R_i} \sim \mathrm{Unif}[-1,1]$  and initialize  $\theta_1, \theta_2 \sim \mathrm{Unif}[0,1]$ , then with probability at least  $1 - \exp(-\Omega(N))$ , the aforementioned randomization procedure discovers the high-payoff NE.

Here we use the uniform distribution as an example. Other distributions may also help in practice. Comparing Thm. 2 and Thm. 1, RR significantly improves standard PG w.r.t. success probability.

Remark: For the scenario studied in Thm. 1, to achieve a  $(1 - \epsilon)$  success probability,  $PG$  requires at least  $N = \Omega\left(\frac{1}{\epsilon}\right)$  random restarts. For the same scenario,  $RR$  only requires to repeat at most  $N = O\left(\log (1 / \epsilon)\right)$  times to achieve a  $(1 - \epsilon)$  success probability. This is an exponential improvement.

RR can also be applied to NE selection in other matrix-form games using an evaluation function  $E(\pi_1, \pi_2)$ . For example, we can set  $E(\pi_1, \pi_2) = U_1(\pi_1, \pi_2) + U_2(\pi_1, \pi_2)$  for a prosocial NE, or look for Pareto-optimal NEs by setting  $E(\pi_1, \pi_2) = \beta U_1(\pi_1, \pi_2) + (1 - \beta)U_2(\pi_1, \pi_2)$  with  $0 \leq \beta \leq 1$ .

# 3 RPG: REWARD-RANDOMIZED POLICY GRADIENT

We now use RL terminologies and consider the 2-player setting for simplicity. Extension to more agents is straightforward (Appx. B.3). Consider a 2-agent Markov game  $M$  defined by  $(S, \mathcal{O}, \mathcal{A}, R, P)$ , where  $\mathcal{S}$  is the state space;  $\mathcal{O} = \{o_i : s \in S, o_i = O(s, i), i \in \{1, 2\}\}$  is the observation space, where agent  $i$  receives its own observation  $o_i = O(s; i)$  (in the fully observable setting,  $O(s, i) = s$ );  $\mathcal{A}$  is the action space for each agent;  $R(s, a_1, a_2; i)$  is the reward function for agent  $i$ ; and  $P(s'|s, a_1, a_2)$  is transition probability from state  $s$  to state  $s'$  when agent  $i$  takes action  $a_i$ . Each agent has a policy  $\pi_i(o_i; \theta_i)$  which produces a (stochastic) action and is parameterized by  $\theta_i$ . In the decentralized RL framework, each agent  $i$  optimizes its expected accumulative reward  $U_i(\theta_i) = \mathbb{E}_{a_1 \sim \pi_1, a_2 \sim \pi_2}[\sum_t \gamma^t R(s^t, a_1^t, a_2^t; i)]$  with some discounted factor  $\gamma$ .

Reward randomization can be applied to a Markov game  $M$  similarly: if the original reward function in  $M$  poses difficulties for PG to discover some particular strategy, we can then define a reward function space  $\mathcal{R}$ , train a population of policy profiles in parallel with sampled reward functions from  $\mathcal{R}$  and select the desired strategy by evaluating the obtained policy profiles in the original game  $M$ . Formally, suppose we consider learning the optimal policies  $\pi_1^{\star}(\theta_1)$  and  $\pi_2^{\star}(\theta_2)$  in a particular a Markov game  $M = (\mathcal{S},\mathcal{O},\mathcal{A},R,P)$  w.r.t. some strategy evaluation function  $E(\pi_1,\pi_2)$ . We then define a proper subspace  $\mathcal{R}$  over possible reward functions  $R:S\times \mathcal{A}\times \mathcal{A}\to \mathbb{R}$  and use  $M(R^{\prime}) = (S,\mathcal{O},\mathcal{A},R^{\prime},P)$  to denote the induced Markov game by replacing the original reward function  $R$  with another  $R^{\prime}\in \mathcal{R}$ . To apply reward randomization, we draw  $N$  samples  $R^{(1)},\ldots ,R^{(N)}$  from  $\mathcal{R}$ , run PG to learn a pair of policy  $\pi_1^{(i)},\pi_2^{(i)}$  on each induced game  $M(R^{(i)})$ , and pick the desired policy profile  $\pi_1^{(k)},\pi_2^{(k)}$  by calculating  $E$  in the original game  $M$ . Lastly, we can fine-tune the policies  $\pi_1^{(k)},\pi_2^{(k)}$  in  $M$  for better practical performance (see discussion below). We call this learning procedure, Reward-Randomized Policy Gradient (RPG), which is summarized in Algo. 1.

Reward-function space: In general, the possible space for a valid reward function is intractably huge. However, in practice, almost all the games designed by human have low-dimensional reward

Algorithm 1: RPG: Reward-Randomized Policy Gradient  
Input: original game  $M$ , search space  $\mathcal{R}$ , evaluation function  $E$ , population size  $N$ ; draw samples  $\{R^{(1)}, \ldots, R^{(N)}\}$  from  $\mathcal{R}$ ;  $\{\pi_1^{(i)}, \pi_2^{(i)}\} \gets \mathrm{PG}$  on induced games  $\{M(R^{(i)})\}_i$  in parallel; // RR phase select the best candidate  $\pi_1^{(k)}, \pi_2^{(k)}$  by  $k = \arg \max_i E(\pi_1^{(i)}, \pi_2^{(i)})$ ; // evaluation phase  $\pi_1^*, \pi_2^* \gets$  fine-tune  $\pi_1^{(k)}, \pi_2^{(k)}$  on  $M$  via PG (if necessary); // fine-tuning phase return  $\pi_1^*, \pi_2^*$ ;

structures based on objects or events, so that we can (almost) always formulate the reward function in a linear form  $R(s, a_1, a_2; i) = \phi(s, a_1, a_2; i)^T w$  where  $\phi(s, a_1, a_2; i)$  is a low-dimensional feature vector and  $w$  is some weight. For example, in navigation games (Mirowski et al., 2016; Lowe et al., 2017; Wu et al., 2018), the reward is typically set to the negative distance from the target location  $L_T$  to the agent's location  $L_A$  plus a success bonus, so the feature vector  $\phi(s, a)$  can be written as a 2-dimensional vector  $[||L_T - L_A||_2, \mathbb{I}(L_T = L_A)]$ ; in real-time strategy games (Wu & Tian, 2016; Vinyals et al., 2017; OpenAI et al., 2019), the feature vector is typically related to the bonus points for destroying each type of units; in robotics manipulation (Levine et al., 2016; Li et al., 2020; Yu et al., 2019),  $\phi$  is often related to the distance between the robot/object and its target position; in general multi-agent games (Lowe et al., 2017; Leibo et al., 2017; Baker et al., 2020),  $\phi$  could contain each agent's individual reward as well as the joint reward over each team, which also enables the representation of different prosociality levels for the agents by varying the weight  $w$ . Hence, a simple and general design principle for  $\mathcal{R}$  is to fix the object/event-based feature vector  $\phi$  while only randomize the weight  $w$ , i.e.,  $\mathcal{R} = \{R_w : R_w(s, a_1, a_2; i) = \phi(s, a_1, a_2; i)^T w, \|w\|_{\infty} \leq C_{\max}\}$ . Hence, the overall search space remains a similar structure as the original game  $M$  but contains a diverse range of preferences over different feature dimensions. Notably, since the optimal strategy is invariant to the scale of the reward function  $R$ , theoretically any  $C_{\max} > 0$  results in the same search space. However, in practice, the scale of reward may significantly influence MARL training stability, so we typically ensure the chosen  $C_{\max}$  to be compatible with the PG algorithm in use.

Fine tuning: There are two benefits: (1) the policies found in the perturbed game may not remain an equilibrium in the original game, so fine-tuning ensures convergence; (2) in practice, fine-tuning could further help escape a suboptimal mode via the noise in PG (Ge et al., 2015; Kleinberg et al., 2018). We remark that a practical issue for fine-tuning is that when the PG algorithm adopts the actor-critic framework (e.g., PPO), we need an additional critic warm-start phase, which only trains the value function while keeps the policy unchanged, before the fine-tuning phase starts. This warm-start phase significantly stabilizes policy learning by ensuring the value function is fully functional for variance reduction w.r.t. the reward function  $R$  in the original game  $M$  when estimating policy gradients.

# 3.1 LEARNING TO ADAPT WITH DIVERSE OPPONENTS

In addition to the final policies  $\pi_1^{\star},\pi_2^{\star}$  , another benefit from RPG is that the population of  $N$  policy profiles contains diverse strategies (more in Sec. 5). With a diverse set of strategies, we can build an adaptive agent by training with a random opponent policy sampled from the set per episode, so that the agent is forced to behave differently based on its opponent's behavior. For simplicity, we consider learning an adaptive pol

icy  $\pi_1^a (\theta^a)$  for agent 1. The procedure remains the same for agent 2. Given a policy population  $\Pi_{2} = \{\pi_{2}^{(1)},\dots ,\pi_{2}^{(N)}\}$  obtained during the RR phase, we can construct a mixed strategy by randomly sampling a policy  $\pi_2^\prime$  from  $\Pi_{2}$  in every training episode and run PG to learn  $\pi_1^a$  by competing against this constructed mixed strategy. The procedure is summarized in Algo. 2. Note that this method does not apply to the one-shot game setting (i.e., horizon is 1) because the adaptive agent does not have any prior knowledge about its opponent's identity before the game starts.

Implementation: We train an RNN policy for  $\pi_1^a (\theta^a)$ . It is critical that the policy input does not directly reveal the opponent's identity, so that it is forced to identify the opponent strategy through what it has observed. On the contrary, when adopting an actor-critic PG framework (Lowe et al., 2017), it is extremely beneficial to include the identity information in the critic input, which makes

Algorithm 2: Learning to Adapt  
Input: game  $M$  , policy set  $\Pi_2$  , initial  $\pi_1^a$  repeat draw a policy  $\pi_2^\prime$  from  $\Pi_{2}$  evaluate  $\pi_1^a$  and  $\pi_2^\prime$  on  $M$  and collect data; update  $\theta^a$  via PG if enough data collected;   
until enough iterations;   
return  $\pi_1^a (\theta^a)$

![](images/fa6682699cec35498497581438790af7d9a7f3fff71ec33f6393fe605c754160.jpg)  
(a) Basic elements

![](images/f633b32a0ed47e9f45eb263a256303d3df977f77ce798bd1d6b36a34db3eab85.jpg)  
Figure 5: Agario: (a) a simplified 2-player setting; (b) basic motions: split, hunt script cells, merge.  
(b) Common behavior: Split, Hunt and Merge

critic learning substantially easier and significantly stabilizes training. We also utilize a multi-head architecture adapted from the multi-task learning literature (Yu et al., 2019), i.e., use a separate value head for each training opponent, which empirically results in the best training performance.

# 4 TESTBEDS FOR RPG: TEMPORAL TRUST DILEMMAS

In this section, we present three 2-player Markov games as testbeds for RPG. All of these games have a diverse range of NE strategies including both "risky" cooperative NEs with high payoffs but hard to discover and "safe" non-cooperative NEs with lower payoffs. We call them temporal trust dilemmas.

Gridworlds: We consider two games adapted from Peysakhovich & Lerer (2018b), Monster-Hunt (Fig. 3) and Escalation (Fig. 4). Both games have a 5-by-5 grid and symmetric rewards.

Monster-Hunt contains a monster and two apples. Apples are static while the monster keeps moving towards its closest agent. If a single agent meets the monster, it loses a penalty of 2; if two agents catch the monster together, they both earn a bonus of 5. Eating an apple always raises a bonus of 2. Whenever an apple is eaten or the monster meets an agent, the entity will respawn randomly. The optimal payoff can only be achieved when both agents precisely catch the monster simultaneously.

Escalation contains a lit grid. When two agents both step on the lit grid, they both get a bonus of 1 and a neighboring grid will be lit up in the next timestep. If only one agent steps on the lit grid, it gets a penalty of  $0.9L$ , where  $L$  denotes the consecutive cooperation steps until that timestep, and the lit grid will respawn randomly. Agents need to stay together on

![](images/efcf67b9619d4c5faa8d71694fe61e8382d288fbbd68c39c5a31f8aec266a652.jpg)  
Figure 3: Monster-Hunt

![](images/f290cb04e8cb5b3d861b874bfeb1011000c56067e4a4a408f30147459376ca7a.jpg)  
Figure 4: Escalation

the lit grid to achieve the maximum payoff despite of the growing penalty. There are multiple NEs: for each  $L$ , that both agents cooperate for  $L$  steps and then leave the lit grid jointly forms an NE.

Agario is a popular multiplayer online game. Players control cells in a Petri dish to gain as much mass as possible by eating smaller cells while avoiding being eaten by larger ones. Larger cells move slower. Each player starts with one cell but can split a sufficiently large cell into two, allowing them to control multiple cells (Wikipedia, 2020). We consider a simplified scenario (Fig. 5) with 2 players (agents) and tiny script cells, which automatically runs away when an agent comes by. There is a low-risk non-cooperative strategy, i.e., two agents stay away from each other and hunt script cells independently. Since the script cells move faster, it is challenging for a single agent to hunt them. By contrast, two agents can cooperate to encircle the script cells to accelerate hunting. However, cooperation is extremely risky for the agent with less mass: two agents need to stay close to cooperate but the larger agent may defect by eating the smaller one and gaining an immediate big bonus.

# 5 EXPERIMENT RESULTS

We use PPO (Schulman et al., 2017) for training. Training episodes for RPG are accumulated over all perturbed games. Evaluation results are averaged over 100 episodes in gridworlds and 1000 episodes in Agario. We repeat all the experiments with 3 seeds and use  $X(Y)$  to denote mean  $X$  with standard deviation  $Y$  in all tables. Since all our discovered (approximate) NEs are symmetric for both players, we simply take  $E(\pi_1, \pi_2) = U_1(\pi_1, \pi_2)$  as our evaluation function and only measure the reward of agent  $I$  in all experiments for simplicity. More details can be found in appendix.

# 5.1 GRIDWORLD GAMES

**Monster-Hunt:** Each agent's reward is determined by three features per timestep: (1) whether two agents catch the monster together; (2) whether the agent steps on an apple; (3) whether

Monster moves towards the closest agent  
![](images/afae6c4ba9c1a64afea915849a90daf46a0b709f146fd027fdce4764ef6471be.jpg)  
(a) Strategy w.  $w = [5,0,0]$  and  $w = [5,0,2]$  (by chance)

![](images/53713ee0d0fb2d1c84aec29f2fddeee3e94daceb167ef7a18ea38103095552ca.jpg)  
Figure 6: Emergent cooperative (approximate) NE strategies found by RPG in Monster-Hunt  
(b) The final strategy after fine-tuning

the agent meets the monster alone. Hence, we write  $\phi(s, a_1, a_2; i)$  as a 3-dimensional  $0/1$  vector with one dimension for one feature. The original game corresponds to  $w = [5, 2, -2]$ .

We set  $C_{\mathrm{max}} = 5$  for sampling  $w$ . We compare RPG with a collection of baselines, including standard PG (PG), PG with shared reward (PG+SR), population-based training (PBT), which trains the same amount of parallel PG policies as RPG, as well as popular exploration methods, i.e., count-based exploration (PG+CNT) (Tang et al., 2017) and MAVEN (Mahajan et al., 2019). We also consider an additional baseline, DIAYN (Eysenbach et al., 2019), which discovers diverse skills using an trajectory-based diversity reward. For a fair comparison, we use DIAYN to first pretrain diverse policies (conceptually similar to the RR phase), then evaluate the rewards for every pair of obtained

![](images/bef05691b1bb8d960d8e4c5ec02470cc93e112f98f53e60b27802fec6d2771a0.jpg)  
Figure 7: Full process of RPG in Monster-Hunt

policies to select the best policy pair (i.e., evaluation phase, shown with the dashed line in Fig. 7), and finally fine-tune the selected policies until convergence (i.e., fine-tuning phase). The results of RPG and the 6 baselines are summarized in Fig. 7, where RPG consistently discovers a strategy with a significantly higher payoff. Note that the strategy with the optimal payoff may not always directly emerge in the RR phase, and there is neither a particular value of  $w$  constantly being the best candidate: e.g., in the RR phase,  $w = [5,0,2]$  frequently produces a sub-optimal cooperative strategy (Fig. 6(a)) with a reward lower than other  $w$  values, but it can also occasionally lead to the optimal strategy (Fig. 6(b)). Whereas, with the fine-tuning phase, the overall procedure of RPG always produces the optimal solution. We visualize both two emergent cooperative strategies in Fig. 6: in the sub-optimal one (Fig. 6(a)), two agents simply move to grid (1,1) together, stay still and wait for the monster, while in the optimal one (Fig. 6(b)), two agents meet each other first and then actively move towards the monster jointly, which further improves hunting efficiency.

Escalation: We can represent  $\phi(s, a_1, a_2; i)$  as 2-dimensional vector containing (1) whether two agents are both in the lit grid and (2) the total consecutive cooperation steps. The original game corresponds to  $w = [1, -0.9]$ . We set  $C_{\mathrm{max}} = 5$  and show the total number of cooperation steps per episode for several selected  $w$  values throughout training in Fig. 8, where RR is able to discover different NE strategies. Note that  $w = [1, 0]$  has already produced the strategy with the optimal payoff in this game, so the fine-tuning phase is no longer needed.

![](images/785a3f5d8c67ad0048907af16683f86bcfce27f126ca84d1e2407bc153f73338.jpg)  
Figure 8: RR in Escalation

# 5.2 2-PLAYER GAMES IN Agario

There are two different settings of Agario: (1) the standard setting, i.e., an agent gets a penalty of  $-x$  for losing a mass  $x$ , and (2) the more challenging aggressive setting, i.e., no penalty for mass loss. Note in both settings: (1) when an agent eats a mass  $x$ , it always gets a bonus of  $x$ ; (2) if an agent loses all the mass, it immediately dies while the other agent can still play in the game. The aggressive setting promotes agent interactions and typically leads to more diverse strategies in practice. Since both settings strictly define the penalty function for mass loss, we do not randomize this reward term. Instead, we consider two other factors: (1) the bonus for eating the other agent; (2) the prosocial level of both agents. We use a 2-dimensional vector  $w = [w_0, w_1]$ , where  $0 \leq w_0, w_1 \leq 1$ , to denote a particular reward function such that (1) when eating a cell of mass  $x$  from the other agent, the bonus is  $w_0 \times x$ , and (2) the final reward is a linear interpolation between  $R(\cdot; i)$  and  $0.5(R(\cdot; 0) + R(\cdot; 1))$  w.r.t.  $w_1$ , i.e., when  $w_1 = 0$ , each agent optimizes its individual reward while when  $w_1 = 1$ , two agents have a shared reward. The original game in both Agario settings corresponds to  $w = [1, 0]$ .

Standard setting: PG in the original game  $(w = [1,0])$  leads to a typical trust-dilemma dynamics: the two agents first learn to hunt and occasionally Cooperate (Fig. 9(a)), i.e., eat a script cell with the other agent close by; then accidentally one agent Attacks the other agent (Fig. 9(b)), which yields a big immediate bonus and makes the policy aggressive; finally policies converge to the non-cooperative

![](images/3c1e6a2bc918c267243303b5d76cf61550604f96be14550d2d5eb3d3e1f8e673.jpg)  
(a) Cooperate

![](images/d026bdd6f3b3aa2c0dc2acc29f982d5f06cac76c464fa4c4c6bd5b9ea6a85240.jpg)  
Figure 9: Emergent strategies in standard Agario: (a) agents cooperate to hunt efficiently; (b) a larger agent breaks the cooperation by attacking the other.  
(b) Attack

Table 2: Results in the standard setting of Agario. PBT: population training of parallel PG policies; RR: best policy in the RR phase  $(w = [1,1])$ ; RPG: fine-tuned policy; RND: PG with RND bonus in the original game.  

<table><tr><td></td><td>PBT</td><td>RR</td><td>RPG</td><td>RND</td></tr><tr><td>Rew.</td><td>3.8(0.3)</td><td>3.8(0.2)</td><td>4.3(0.2)</td><td>2.8(0.3)</td></tr><tr><td>#Coop.</td><td>1.9(0.2)</td><td>2.2(0.1)</td><td>2.0(0.3)</td><td>1.3(0.2)</td></tr><tr><td>#Hunt</td><td>0.6(0.1)</td><td>0.4(0.0)</td><td>0.7(0.0)</td><td>0.6(0.1)</td></tr></table>

Table 3: Results in the aggressive setting of Agario: PBT: population training of parallel PG policies; RR:  $w = [0,0]$  is the best candidate via RR; RPG: fine-tuned policy; RND: PG with RND bonus.  

<table><tr><td></td><td>PBT</td><td>w=[0.5,1]</td><td>w=[0,1]</td><td>w=[0,0]</td><td>RPG</td><td>RND</td></tr><tr><td>Rew.</td><td>3.3(0.2)</td><td>4.8(0.6)</td><td>5.1(0.4)</td><td>6.0(0.5)</td><td>8.9(0.3)</td><td>3.2(0.2)</td></tr><tr><td>#Attack</td><td>0.4(0.0)</td><td>0.7(0.2)</td><td>0.3(0.1)</td><td>0.5(0.1)</td><td>0.9(0.1)</td><td>0.4(0.0)</td></tr><tr><td>#Coop.</td><td>0.0(0.0)</td><td>0.6(0.6)</td><td>2.3(0.3)</td><td>1.6(0.1)</td><td>2.0(0.2)</td><td>0.0(0.0)</td></tr><tr><td>#Hunt</td><td>0.7(0.1)</td><td>0.6(0.3)</td><td>0.3(0.0)</td><td>0.7(0.0)</td><td>0.9(0.1)</td><td>0.7(0.0)</td></tr></table>

![](images/067c6cd4fb05b54d9f282a32e86d13679877cea9ccf5bd4678b682be0654c7e3.jpg)  
Figure 10: Sacrifice strategy,  $w = [1,1]$ , aggressive setting.

![](images/98db24156f62430a661c2755754327d5cb600d7d6bd58bee6832c11d278b6b72.jpg)  
Figure 11: Perpetual strategy,  $w = [0.5, 1]$  (by chance), aggressive setting, i.e., two agents mutually sacrifice themselves. One agent first splits to sacrifice a part of its mass to the larger agent while the other agent also does the same thing later to repeat the sacrifice cycle.

equilibrium where both agents keep apart and hunt alone. The quantitative results are shown in Tab. 2. Baselines include population-based training (PBT) and a state-the-art exploration method for high-dimensional state, Random Network Distillation (RND) (Burda et al., 2019). RND and PBT occasionally learns cooperative strategies while RR stably discovers a cooperative equilibrium with  $w = [1,1]$ , and the full RPG further improves the rewards. Interestingly, the best strategy obtained in the RR phase even has a higher Cooperate frequency than the full RPG: fine-tuning transforms the strong cooperative strategy to a more efficient strategy, which has a better balance between Cooperate and selfish Hunt and produces a higher average reward.

Aggressive setting: Similarly, we apply RPG in the aggressive setting and show results in Tab. 3. Neither PBT nor RND was able to find any cooperative strategies in the aggressive game while RPG stably discovers a cooperative equilibrium with a significantly higher reward. We also observe a diverse set of complex strategies in addition to normal Cooperate and Attack. Fig. 10 visualizes the Sacrifice strategy derived with  $w = [1,1]$ : the smaller agent rarely hunts script cells; instead, it waits in the corner for being eaten by the larger agent to contribute all its mass to its partner. Fig. 11 shows another surprisingly novel emergent strategy by  $w = [0.5,1]$ : each agent first hunts individually to gain enough mass; then one agent splits into smaller cells while the other agent carefully eats a portion of the split agent; later on, when the agent who previously lost mass gains sufficient mass, the larger agent similarly splits itself to contribute to the other one, which completes the (ideally) never-ending loop of partial sacrifice. We name this strategy Perpetual for its conceptual similarity to the perpetual motion machine. Lastly, the best strategy is produced by  $w = [0,0]$  with a balance between Cooperate and Perpetual: they cooperate to hunt script cells to gain mass efficiently and quickly perform mutual sacrifice as long as their mass is sufficiently large for split-and-eat. Hence, although the RPG policy has relatively lower Cooperate frequency than the policy by  $w = [0,1]$ , it yields a significantly higher reward thanks to a much higher Attack (i.e., Sacrifice) frequency.

# 5.3 LEARNING ADAPTIVE POLICIES

**Monster-Hunt:** We select policies trained by 8 different  $w$  values in the RR phase and use half of them for training the adaptive policy and the remaining half as hidden opponents for evaluation. We also make sure that both training and evaluation policies cover the following 4 strategy modes: (1)  $M(onster)$ : the agent always moves towards the monster; (2)  $M(onster)-Alone$ : the agent moves

towards the monster but also tries to keeps apart from the other agent; (3)  $M(onster)-Coop$ : the agent seeks to hunt the monster together with the other agent; (4)  $Apple$ : the agent only eats apple. The evaluation results are shown in Tab. 4, where the adaptive policy successfully exploits all the test-time opponents, including  $M(onster)-Alone$ , which was trained to actively avoids the other agent.

<table><tr><td>Oppo.</td><td>M.</td><td>M-Coop.</td><td>M-Alone.</td><td>Apple.</td></tr><tr><td>#C-H</td><td>16.3(19.2)</td><td>20.9(0.8)</td><td>14.2(18.0)</td><td>2.7(1.0)</td></tr><tr><td>#S-H</td><td>1.2(0.4)</td><td>0.4(0.1)</td><td>2.2(1.2)</td><td>2.2(1.4)</td></tr><tr><td>#Apple</td><td>12.4(7.3)</td><td>3.3(0.8)</td><td>10.9(7.0)</td><td>13.6(3.8)</td></tr></table>

Table 4: Stats. of the adaptive agent in Monster-Hunt with hold-out test-time opponents. #C(oop.)-H(unt): both agents catch the monster; #S(ingle)-H(unt): the adaptive agent meets the monster alone; #Apple: apple eating. The adaptive policy successfully exploits different opponents and rarely meets the monster alone.

Agario: We show the trained agent can choose to cooperate or compete adaptively in the standard setting. We pick 2 cooperative policies (i.e., Cooperate preferred,  $w = [1,0]$ ) and 2 competitive policies (i.e., Attack preferred,  $w = [1,1]$ ) and use half of them for training and the other half for testing. For a hard challenge at test time, we switch the opponent within an episode, i.e., we use a cooperative opponent in the first half and then immediately switch to a competitive one, and vice versa. So, a desired policy should adapt quickly at halftime. Tab. 5 compares the second-half behavior of the adaptive agent with the oracle pure-competitive/cooperative agents. The rewards of the adaptive agent is close to the oracle: even with half-way switches, the trained policy is

<table><tr><td>Agent</td><td>Adapt.</td><td>Coop.</td><td>Comp.</td></tr><tr><td colspan="4">Opponent: Cooperative → Competitive</td></tr><tr><td>#Attack</td><td>0.2(0.0)</td><td>0.3(0.0)</td><td>0.1(0.1)</td></tr><tr><td>Rew.</td><td>0.7(0.7)</td><td>-0.2(0.6)</td><td>0.8(0.5)</td></tr></table>

<table><tr><td colspan="4">Opponent: Competitive → Cooperative</td></tr><tr><td>#Coop.</td><td>1.0(0.3)</td><td>1.4(0.4)</td><td>0.3(0.4)</td></tr><tr><td>Rew.</td><td>2.5(0.7)</td><td>3.6(1.2)</td><td>1.1(0.7)</td></tr></table>

Table 5: Adaptation test in Agario. Opponent type is switched half-way per episode. #Attack, #Coop.: episode statistics; Rew.: agent reward. Adaptive agents' rewards are close to oracles.

able to exploit the cooperative opponent while avoid being exploited by the competitive one.

# 6 RELATED WORK AND DISCUSSIONS

Our core idea is reward perturbation. In game theory, this is aligned with the quantal response equilibrium (McKelvey & Palfrey, 1995), a smoothed version of NE obtained when payoffs are perturbed by a Gumbel noise. In RL, reward shaping is popular for learning desired behavior in various domains (Ng et al., 1999; Babes et al., 2008; Devlin & Kudenko, 2011), which inspires our idea for finding diverse strategic behavior. By contrast, state-space exploration methods (Pathak et al., 2017; Burda et al., 2019; Eysenbach et al., 2019; Sharma et al., 2020) only learn low-level primitives without strategy-level diversity (Baker et al., 2020). RR trains a set of policies, which is aligned with the population-based training in MARL (Jaderberg et al., 2017; 2019; Vinyals et al., 2019; Long et al., 2020). RR is conceptually related to domain randomization (Tobin et al., 2017) with the difference that we train separate policies instead of a single universal one, which suffers from mode collapse (see appendix). Besides, RPG helps train adaptive policies against a set of opponents, which is related to Bayesian games (Dekel et al., 2004; Hartline et al., 2015). In RL, there are works on learning when to cooperate/compete (Littman, 2001; Peysakhovich & Lerer, 2018a; Kleiman-Weiner et al., 2016; Woodward et al., 2019; McKee et al., 2020), which is a special case of ours, or learning robust policies (Li et al., 2019; Shen & How, 2019; Hu et al., 2020), which complements our method.

Although we choose decentralized PG in this paper, RR can be combined with any other multi-agent learning algorithms for games, such as fictitious play (Robinson, 1951; Monderer & Shapley, 1996; Heinrich & Silver, 2016; Kamra et al., 2019; Han & Hu, 2019), double-oracle (McMahan et al., 2003; Lanctot et al., 2017; Wang et al., 2019; Balduzzi et al., 2019) and regularized self-play (Foerster et al., 2018; Perolat et al., 2020; Bai & Jin, 2020). Many of these works have theoretical guarantees to find an (approximate) NE but there is little work focusing on which NE strategy these algorithms can converge to when multiple NEs exist, e.g., the stag-hunt game and its variants, for which many learning dynamics fail to converge to a prevalence of the pure strategy Stag (Kandori et al., 1993; Ellison, 1993; Fang et al., 2002; Skyrms & Pemanle, 2009; Golman & Page, 2010).

In this paper, we consider stag hunt as an particular example where an "optimal" NE with a high payoff for every agent exists. In general cases, we can select a desired strategy w.r.t. an evaluation function. This is related to the problem of equilibrium refinement (or equilibrium selection) (Selten, 1965; 1975; Myerson, 1978), which aims to find a subset of equilibria satisfying desirable properties, e.g., admissibility (Banks & Sobel, 1987), subgame perfection (Selten, 1965), Pareto efficiency (Bernheim et al., 1987) or robustness against opponent's deviation from best response in security-related applications (Fang et al., 2013; An et al., 2011).

# REFERENCES

Bo An, Milind Tambe, Fernando Ordonez, Eric Shieh, and Christopher Kiekintveld. Refinement of strong stackelberg equilibria in security games. In Twenty-Fifth AAAI Conference on Artificial Intelligence, 2011.  
Monica Babes, Enrique Munoz de Cote, and Michael L Littman. Social reward shaping in the prisoner's dilemma. In Proceedings of the 7th international joint conference on Autonomous agents and multiagent systems-Volume 3, pp. 1389-1392. International Foundation for Autonomous Agents and Multiagent Systems, 2008.  
Yu Bai and Chi Jin. Provable self-play algorithms for competitive reinforcement learning. arXiv preprint arXiv:2002.04017, 2020.  
Bowen Baker, Ingmar Kanitscheider, Todor Markov, Yi Wu, Glenn Powell, Bob McGrew, and Igor Mordatch. Emergent tool use from multi-agent autocurricula, 2019.  
Bowen Baker, Ingmar Kanitscheider, Todor Markov, Yi Wu, Glenn Powell, Bob McGrew, and Igor Mordatch. Emergent tool use from multi-agent autocurricula. In International Conference on Learning Representations, 2020.  
David Balduzzi, Marta Garnelo, Yoram Bachrach, Wojciech M Czarnecki, Julien Perolat, Max Jaderberg, and Thore Graepel. Open-ended learning in symmetric zero-sum games. arXiv preprint arXiv:1901.08106, 2019.  
Jeffrey S Banks and Joel Sobel. Equilibrium selection in signaling games. *Econometrica: Journal of the Econometric Society*, pp. 647-661, 1987.  
B Douglas Bernheim, Bezalel Peleg, and Michael D Whinston. Coalition-proof Nash equilibria i. concepts. Journal of Economic Theory, 42(1):1-12, 1987.  
George W Brown. Iterative solution of games by fictitious play. Activity analysis of production and allocation, 13(1):374-376, 1951.  
Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. In International Conference on Learning Representations, 2019.  
Murray Campbell, A Joseph Hoane Jr, and Feng-hsiung Hsu. Deep blue. Artificial intelligence, 134 (1-2):57-83, 2002.  
Eddie Dekel, Drew Fudenberg, and David K Levine. Learning to play Bayesian games. Games and Economic Behavior, 46(2):282-303, 2004.  
Sam Devlin and Daniel Kudenko. Theoretical considerations of potential-based reward shaping for multi-agent systems. In *The 10th International Conference on Autonomous Agents and Multiagent Systems-Volume 1*, pp. 225–232. International Foundation for Autonomous Agents and Multiagent Systems, 2011.  
Glenn Ellison. Learning, local interaction, and coordination. *Econometrica: Journal of the Econometric Society*, pp. 1047-1071, 1993.  
Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function. In International Conference on Learning Representations, 2019.  
Christina Fang, Steven Orla Kimbrough, Stefano Pace, Annapurna Valluri, and Zhiqiang Zheng. On adaptive emergence of trust behavior in the game of stag hunt. Group Decision and Negotiation, 11(6):449-467, 2002.  
Fei Fang, Albert Xin Jiang, and Milind Tambe. Protecting moving targets with multiple mobile resources. Journal of Artificial Intelligence Research, 48:583-634, 2013.  
Jakob Foerster, Richard Y Chen, Maruan Al-Shedivat, Shimon Whiteson, Pieter Abbeel, and Igor Mordatch. Learning with opponent-learning awareness. In Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems, pp. 122-130. International Foundation for Autonomous Agents and Multiagent Systems, 2018.

Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points—online stochastic gradient for tensor decomposition. In Conference on Learning Theory, pp. 797–842, 2015.  
Russell Golman and Scott E Page. Individual and cultural learning in stag hunt games with multiple actions. Journal of Economic Behavior & Organization, 73(3):359-376, 2010.  
Jiequn Han and Ruimeng Hu. Deep fictitious play for finding Markovian Nash equilibrium in multi-agent games. arXiv preprint arXiv:1912.01809, 2019.  
Jason Hartline, Vasilis Syrgkanis, and Eva Tardos. No-regret learning in Bayesian games. In Advances in Neural Information Processing Systems, pp. 3061-3069, 2015.  
Johannes Heinrich and David Silver. Deep reinforcement learning from self-play in imperfect-information games. arXiv preprint arXiv:1603.01121, 2016.  
Hengyuan Hu, Adam Lerer, Alex Peysakhovich, and Jakob Foerster. Other-play for zero-shot coordination. arXiv preprint arXiv:2003.02979, 2020.  
Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, et al. Population based training of neural networks. arXiv preprint arXiv:1711.09846, 2017.  
Max Jaderberg, Wojciech M Czarnecki, Iain Dunning, Luke Marris, Guy Lever, Antonio Garcia Castaneda, Charles Beattie, Neil C Rabinowitz, Ari S Morcos, Avraham Ruderman, et al. Human-level performance in 3D multiplayer games with population-based reinforcement learning. Science, 364(6443):859-865, 2019.  
Nitin Kamra, Umang Gupta, Kai Wang, Fei Fang, Yan Liu, and Milind Tambe. Deep fictitious play for games with continuous action spaces. In Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems, pp. 2042-2044. International Foundation for Autonomous Agents and Multiagent Systems, 2019.  
Michihiro Kandori, George J Mailath, and Rafael Rob. Learning, mutation, and long run equilibria in games. *Econometrica: Journal of the Econometric Society*, pp. 29-56, 1993.  
Max Kleiman-Weiner, Mark K Ho, Joseph L Austerweil, Michael L Littman, and Joshua B Tenenbaum. Coordinate to cooperate or compete: abstract goals and joint intentions in social interaction. In CogSci, 2016.  
Robert Kleinberg, Yanzhi Li, and Yang Yuan. An alternative view: When does sgd escape local minima? arXiv preprint arXiv:1802.06175, 2018.  
Marc Lanctot, Vinicius Zambaldi, Audrunas Gruslys, Angeliki Lazaridou, Karl Tuyls, Julien Pérolat, David Silver, and Thore Graepel. A unified game-theoretic approach to multiagent reinforcement learning. In Advances in Neural Information Processing Systems, pp. 4190-4203, 2017.  
Joel Z Leibo, Vinicius Zambaldi, Marc Lanctot, Janusz Marecki, and Thore Graepel. Multi-agent reinforcement learning in sequential social dilemmas. In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems, pp. 464-473, 2017.  
Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. The Journal of Machine Learning Research, 17(1):1334-1373, 2016.  
Richard Li, Allan Jabri, Trevor Darrell, and Pulkit Agrawal. Towards practical multi-object manipulation using relational reinforcement learning. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2020.  
Shihui Li, Yi Wu, Xinyue Cui, Honghua Dong, Fei Fang, and Stuart Russell. Robust multi-agent reinforcement learning via minimax deep deterministic policy gradient. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 4213-4220, 2019.  
Michael L Littman. Friend-or-foe q-learning in general-sum games. In ICML, volume 1, pp. 322-328, 2001.

Qian Long, Zihan Zhou, Abhinav Gupta, Fei Fang, Yi Wu, and Xiaolong Wang. Evolutionary population curriculum for scaling multi-agent reinforcement learning. In International Conference on Learning Representations, 2020.  
Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in neural information processing systems, pp. 6379–6390, 2017.  
Anuj Mahajan, Tabish Rashid, Mikayel Samvelyan, and Shimon Whiteson. Maven: Multi-agent variational exploration. In Advances in Neural Information Processing Systems, pp. 7611-7622, 2019.  
Kevin R McKee, Ian Gemp, Brian McWilliams, Edgar A Duñez-Guzmán, Edward Hughes, and Joel Z Leibo. Social diversity and social preferences in mixed-motive reinforcement learning. arXiv preprint arXiv:2002.02325, 2020.  
Richard D McKelvey and Thomas R Palfrey. Quantal response equilibria for normal form games. Games and economic behavior, 10(1):6-38, 1995.  
H Brendan McMahan, Geoffrey J Gordon, and Avrim Blum. Planning in the presence of cost functions controlled by an adversary. In Proceedings of the 20th International Conference on Machine Learning (ICML-03), pp. 536-543, 2003.  
Piotr Mirowski, Razvan Pascanu, Fabio Viola, Hubert Soyer, Andrew J Ballard, Andrea Banino, Misha Denil, Ross Goroshin, Laurent Sifre, Koray Kavukcuoglu, et al. Learning to navigate in complex environments. arXiv preprint arXiv:1611.03673, 2016.  
Dov Monderer and Lloyd S Shapley. Potential games. Games and economic behavior, 14(1):124-143, 1996.  
Roger B Myerson. Refinements of the Nash equilibrium concept. International journal of game theory, 7(2):73-80, 1978.  
John Nash. Non-cooperative games. Annals of mathematics, pp. 286-295, 1951.  
Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In ICML, volume 99, pp. 278-287, 1999.  
OpenAI, :, Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemysław Debiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, Rafal Józefowicz, Scott Gray, Catherine Olsson, Jakub Pachocki, Michael Petrov, Henrique Pondé de Oliveira Pinto, Jonathan Raiman, Tim Salimans, Jeremy Schlatter, Jonas Schneider, Szymon Sidor, Ilya Sutskever, Jie Tang, Filip Wolski, and Susan Zhang. Dota 2 with large scale deep reinforcement learning, 2019.  
Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 16-17, 2017.  
Julien Perolat, Remi Munos, Jean-Baptiste Lespiau, Shayegan Omidshafiei, Mark Rowland, Pedro Ortega, Neil Burch, Thomas Anthony, David Balduzzi, Bart De Vylder, et al. From Poincare recurrence to convergence in imperfect information games: Finding equilibrium via regularization. arXiv preprint arXiv:2002.08456, 2020.  
Alexander Peysakhovich and Adam Lerer. Consequentialist conditional cooperation in social dilemmas with imperfect information. In International Conference on Learning Representations, 2018a.  
Alexander Peysakhovich and Adam Lerer. Prosocial learning agents solve generalized stag hunts better than selfish ones. In Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems, pp. 2043-2044. International Foundation for Autonomous Agents and Multiagent Systems, 2018b.  
Julia Robinson. An iterative method of solving a game. Annals of mathematics, pp. 296-301, 1951.  
Jean-Jacques Rousseau. A discourse on inequality. Penguin, 1984.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
R Selten. Reexamination of the perfectness concept for equilibrium points in extensive games. International Journal of Game Theory, 4(1):25-55, 1975.  
Reinhard Selten. Spieltheoretische behandlung eines oligopolmodells mit nachfragträgeit: Teil i: Bestimmung des dynamischen preisgleichgewichts. Zeitschrift für die gesamte Staatswissenschaft/Journal of Institutional and Theoretical Economics, (H. 2):301-324, 1965.  
Claude E Shannon. Xxii. programming a computer for playing chess. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 41(314):256-275, 1950.  
Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, and Karol Hausman. Dynamics-aware unsupervised discovery of skills. In International Conference on Learning Representations, 2020.  
Macheng Shen and Jonathan P How. Robust opponent modeling via adversarial ensemble reinforcement learning in asymmetric imperfect-information games. arXiv preprint arXiv:1909.08735, 2019.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. Nature, 550(7676):354-359, 2017.  
David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419): 1140-1144, 2018.  
Satinder P Singh, Michael J Kearns, and Yishay Mansour. Nash convergence of gradient dynamics in general-sum games. In UAI, pp. 541-548, 2000.  
Brian Skyrms. The stag hunt and the evolution of social structure. Cambridge University Press, 2004.  
Brian Skyrms and Robin Pemantle. A dynamic model of social network formation. In Adaptive networks, pp. 231-251. Springer, 2009.  
Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, OpenAI Xi Chen, Yan Duan, John Schulman, Filip DeTurck, and Pieter Abbeel. #exploration: A study of count-based exploration for deep reinforcement learning. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems 30, pp. 2753-2762. 2017.  
Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS), pp. 23-30. IEEE, 2017.  
Oriol Vinyals, Timo Ewalds, Sergey Bartunov, Petko Georgiev, Alexander Sasha Vezhnevets, Michelle Yeo, Alireza Makhzani, Heinrich Kuttler, John Agapiou, Julian Schrittwieser, et al. Starcraft II: A new challenge for reinforcement learning. arXiv preprint arXiv:1708.04782, 2017.  
Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michael Mathieu, Andrew Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature, 575(7782):350-354, 2019.  
Yufei Wang, Zheyuan Ryan Shi, Lantao Yu, Yi Wu, Rohit Singh, Lucas Joppa, and Fei Fang. Deep reinforcement learning for green security games with real-time information. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 1401-1408, 2019.  
Wikipedia. Agar.io, 2020. URL http://en.wikipedia.org/wiki/Agar.io. [http://en.wikipedia.org/wiki/Agar.io; accessed 3-June-2020].  
Mark Woodward, Chelsea Finn, and Karol Hausman. Learning to interactively learn and assist. arXiv preprint arXiv:1906.10187, 2019.

Yi Wu, Yuxin Wu, Georgia Gkioxari, and Yuandong Tian. Building generalizable agents with a realistic and rich 3D environment. arXiv preprint arXiv:1801.02209, 2018.  
Yuxin Wu and Yuandong Tian. Training agent for first-person shooter game with actor-critic curriculum learning. 2016.  
Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning (CoRL), 2019.

We would suggest to visit https://sites.google.com/view/staghuntrpg for example videos.
