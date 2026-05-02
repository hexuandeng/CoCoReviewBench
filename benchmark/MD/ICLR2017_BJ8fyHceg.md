# TUNING RECURRENT NEURAL NETWORKS WITH REINFORCEMENT LEARNING

Natasha Jaques<sup>12</sup>, Shixiang Gu<sup>134</sup>, Richard E. Turner<sup>3</sup>, Douglas Eck<sup>1</sup>

<sup>1</sup>Google Brain, USA  
$^{2}$ Massachusetts Institute of Technology, USA  
<sup>3</sup>University of Cambridge, UK  
4Max Planck Institute for Intelligent Systems, Germany

jaquesn@mit.edu, sg717@cam.ac.uk, ret26@cam.ac.uk, deck@google.com

# ABSTRACT

Sequence models can be trained using supervised learning and a next-step prediction objective. This approach, however, suffers from known failure modes. For example, it is notoriously difficult to ensure multi-step generated sequences have coherent global structure. Motivated by the fact that reinforcement learning (RL) can be used to impose arbitrary properties on generated data by choosing appropriate reward functions, in this paper we propose a novel approach for sequence training which combines Maximum Likelihood (ML) and RL training. We refine a sequence predictor by optimizing for some imposed reward functions, while maintaining good predictive properties learned from data. We propose efficient ways to solve this by augmenting deep Q-learning with a cross-entropy reward and deriving novel off-policy methods for RNNs from stochastic optimal control (SOC). We explore the usefulness of our approach in the context of music generation. An LSTM is trained on a large corpus of songs to predict the next note in a musical sequence. This Note-RNN is then refined using RL, where the reward function is a combination of rewards based on rules of music theory, as well as the output of another trained Note-RNN. We show that by combining ML and RL, this RL Tuner method can not only produce more pleasing melodies, but that it can significantly reduce unwanted behaviors and failure modes of the RNN.

# 1 INTRODUCTION

Generative modeling of music with deep neural networks is typically accomplished by training a Recurrent Neural Network (RNN) such as a Long Short-Term Memory (LSTM) network to predict the next note in a musical sequence (e.g. Eck & Schmidhuber (2002)). Similar to a Character RNN (Mikolov et al., 2010), these Note RNNs can be used to generate novel melodies by initializing them with a short sequence of notes, then repeatedly sampling from the model's output distribution generated to obtain the next note. While compositions generated in this way have recently garnered attention<sup>1</sup>, this type of model tends to suffer from common failure modes, such as excessively repeating notes, or producing sequences that lack a consistent theme or structure. Such sequences can appear wandering and random (see Graves (2013) for a text-based example).

Music compositions adhere to relatively well-defined structural rules, making music an interesting sequence generation challenge. For example, music theory tells that groups of notes belong to keys, chords follow progressions, and songs have consistent structures made up of musical phrases. Our research question is therefore whether such music-theory-based constraints can be learned by an RNN, while still allowing it to maintain note probabilities learned from data.

To approach this problem we propose RL Tuner, a novel sequence learning approach in which RL is used to impose structure on an RNN trained on data. We begin by training a deep  $Q$ -network (DQN) with a reward function comprising both a music-theory based reward, and the probability output

of a trained Note RNN. We show that this objective function can be related to stochastic optimal control (SOC) and derive two additional off-policy methods for refining the RNN by penalizing KL-divergence from its original policy. Thus, the RL Tuner models learn to adhere to a set of composition rules, while still maintaining information about the transition probabilities originally learned from data. We show that not only do the models successfully learn the desired behaviors, but that they produce varied compositions which are more melodic, harmonious, interesting, and rated as significantly more subjectively pleasing than those of the Note RNN. We suggest that this method of combining ML and RL could have potential applications in a number of areas as a general way to refine existing recurrent models trained on data by imposing constraints on their behavior.

# 2 BACKGROUND

# 2.1 DEEP Q-LEARNING

In RL, an agent interacts with an environment. Given the state of the environment at time  $t$ ,  $s_t$ , the agent takes an action  $a_t$  according to its policy  $\pi(a_t|s_t)$ , receives a reward  $r(s_t, a_t)$ , and the environment transitions to a new state,  $s_{t+1}$ . The agent's goal is to maximize reward over a sequence of actions, with a discount factor of  $\gamma$  applied to future rewards. The optimal deterministic policy  $\pi^*$  is known to satisfy the following Bellman optimality equation,

$$
Q \left(s _ {t}, a _ {t}; \pi^ {*}\right) = r \left(s _ {t}, a _ {t}\right) + \gamma \mathbb {E} _ {p \left(s _ {t + 1} \mid s _ {t}, a _ {t}\right)} \left[ \max  _ {a _ {t + 1}} Q \left(s _ {t + 1}, a _ {t + 1}; \pi^ {*}\right) \right] \tag {1}
$$

where  $Q^{\pi}(s_t,a_t) = \mathbb{E}_{\pi}[\sum_{t' = t}^{\infty}\gamma^{t' - t}r(s_{t'},a_{t'})]$  is the  $Q$  function of a policy  $\pi$ .  $Q$ -learning techniques (Watkins & Dayan, 1992; Sutton et al., 1999) learn this optimal  $Q$  function by iteratively minimizing the Bellman residual. The optimal policy is given by  $\pi^{*}(a|s) = \arg \max_{a}Q(s,a)$ . Deep  $Q$ -learning(Mnih et al., 2013) uses a neural network called the deep  $Q$ -network (DQN) to approximate the  $Q$  function  $Q(s,a;\theta)$ . The network parameters  $\theta$  are learned by applying stochastic gradient descent (SGD) updates with respect to the following loss function,

$$
L (\theta) = \mathbb {E} _ {\beta} \left[ \left(r (s, a) + \gamma \max  _ {a ^ {\prime}} Q \left(s ^ {\prime}, a ^ {\prime}; \theta^ {-}\right) - Q (s, a; \theta)\right) ^ {2} \right] \tag {2}
$$

where  $\beta$  is the exploration policy, and  $\theta^{-}$  is the parameters of the Target  $Q$ -network (Mnih et al., 2013) that is held fixed during the gradient computation. The moving average of  $\theta$  is used as  $\theta^{-}$  as proposed in (Lillicrap et al., 2016). Exploration can be performed with either the  $\epsilon$ -greedy method or Boltzmann sampling. Additional standard techniques such as replay memory (Mnih et al., 2013) and Deep Double  $Q$ -learning (Hasselt et al., 2015) are used to stabilize and improve learning.

# 2.2 MUSIC GENERATION WITH LSTM

Previous work with music generation using deep learning (e.g. (Eck & Schmidhuber, 2002), (Sturm et al., 2016)) has involved training an RNN to learn to predict the next note in a monophonic melody; we call this type of model a Note RNN. Often, the Note RNN is implemented using a Long Short-Term Memory (LSTM) network (Gers et al., 2000). LSTMs are networks in which each recurrent cell learns to control the storage of information through the use of an input gate, output gate, and forget gate. The first two gates control whether information is able to flow into and out of the cell, and the latter controls whether or not the contents of the cell should be reset. Due to these properties, LSTMs are better at learning long-term dependencies in the data, and can adapt more rapidly to new data (Graves, 2013). A softmax function can be applied to the final outputs of the network to obtain the probability the network places on each note, and softmax cross-entropy loss can be used to train the model via back propagation through time (BPTT) (Graves & Schmidhuber, 2005). However, as previously described, the melodies generated by this model tend to wander, and lack musical structure; we will show that they are also perceived as less musically pleasing by listeners. In the next section, we will show how to improve this model with RL.

# 3 RL TUNER DESIGN

Given a trained Note RNN, the goal is to teach it concepts about music theory, while still maintaining the information about typical musical compositions originally learned from data. To accomplish this

task, we propose RL Tuner, a novel sequence training method incorporating RL. We use an LSTM trained on data (the Note RNN) to supply the initial weights for three networks in RL Tuner: the  $Q$ -network and Target  $Q$ -network in the DQN algorithm as described in Section 2.1, and a Reward RNN. The Reward RNN is held fixed, and used to supply part of the reward value used in training.

In order to formulate musical composition as an RL problem, we treat placing the next note in the composition as taking an action. The state of the environment  $s$  consists of both the notes placed in the composition so far and the internal state of the LSTM cells of both the  $Q$ -network and the Reward RNN. To calculate the reward, we combine probabilities learned from the training data with knowledge of music theory. We define a set of music-theory based rules (described in Section 3.2) to impose constraints on the melody that the model is composing through a reward signal  $r_{MT}(a,s)$ . For example, if a note is in the wrong key, then the model receives a negative reward. However, it is necessary that the model still be "creative," rather than learning a simple composition that can easily exploit these rewards. Therefore, we use the Reward RNN — or equivalently the trained Note RNN — to compute  $\log p(a|s)$ , the log probability of a note  $a$  given a composition  $s$ , and incorporate this into the reward function. Figure 1 illustrates these ideas.

![](images/59c7d26923c2d7a1dc19c7e290dbdc017962a4a8bf410b3ba5d5014916988c21.jpg)  
Figure 1: A Note RNN is trained on MIDI files and supplies the initial weights for the  $Q$ -network and Target- $Q$ -network, and final weights for the Reward RNN.

The total reward given at time  $t$  is therefore:

$$
r (s, a) = \log p (a | s) + r _ {M T} (a, s) / c \tag {3}
$$

where  $c$  is a constant controlling the emphasis placed on the music theory reward. Given the DQN loss function in Eq. 2 and modified reward function in Eq. 3, the new loss function and learned policy for RL Tuner are,

$$
L (\theta) = \mathbb {E} _ {\beta} [ (\log p (a | s) + r _ {M T} (a, s) / c + \gamma \max  _ {a ^ {\prime}} Q \left(s ^ {\prime}, a ^ {\prime}; \theta^ {-}\right) - Q (s, a; \theta)) ^ {2} ] \tag {4}
$$

$$
\pi_ {\theta} (a | s) = \delta (a = \arg \max  _ {a} Q (s, a; \theta)). \tag {5}
$$

Thus, the modified loss function forces the model to learn that the most valuable actions are those that conform to the music theory rules, but still have high probability in the original data.

# 3.1 RELATIONSHIP TO STOCHASTIC OPTIMAL CONTROL

The technique described in Section 3 has a close connection with stochastic optimal control (SOC) (Todorov, 2006; Kappen et al., 2012; Rawlik et al., 2012). SOC defines a prior dynamics or policy, and derives a variant of the control or RL problem as performing approximate inference in a graphical model. Let  $\tau$  be a trajectory of state and action sequences,  $p(\tau)$  be a prior dynamics, and  $r(\tau)$  be the reward of the trajectory. Then, SOC introduces an additional binary variable  $b$  and defines a graphical model as  $p(\tau, b) = p(\tau)p(b|\tau)$ , where  $p(b = 1|\tau) = e^{r(\tau)/c}$  and  $c$  is the temperature variable. An approximation to  $p(\tau|b = 1)$  can be derived using the variational free-energy method, and this leads to a cost with a similar form to the RL problem previously defined, but with an additional penalty based on the Kullback-Leibler (KL) divergence from the prior trajectory,

$$
\begin{array}{l} \log p (\tau | b = 1) = \log \int p (\tau) p (b | \tau) d \tau (6) \\ \geq \mathbb {E} _ {q (\tau)} [ \log p (\tau) p (b | \tau) - \log q (\tau) ] (7) \\ = \mathbb {E} _ {q (\tau)} [ r (\tau) / c - \mathbf {K L} [ q (\tau) | | p (\tau) ] ] = L _ {v} (q) (8) \\ \end{array}
$$

where  $q(\tau)$  is the variational distribution. Rewriting the variational objective  $L_{v}(q)$  in Eq. 6 in terms of policy  $\pi_{\theta}$ , we get the following RL objective with KL-regularization,

$$
L _ {v} (\theta) = \mathbb {E} _ {\pi} \left[ \sum_ {t} r \left(s _ {t}, a _ {t}\right) / c - K L \left[ \pi_ {\theta} \left(\cdot \mid s _ {t}\right) \right| | p \left(\cdot \mid s _ {t}\right) ] \right]. \tag {9}
$$

In contrast, the objective in Section 3 is,

$$
L _ {v} (\theta) = \mathbb {E} _ {\pi} \left[ \sum_ {t} r \left(s _ {t}, a _ {t}\right) / c + \log p \left(a _ {t} \mid s _ {t}\right) \right]. \tag {10}
$$

The difference is that Eq. 9 includes an entropy regularizer, and thus a different off-policy method from  $Q$ -learning is required.  $\Psi$ -learning (Peters et al., 2010) and  $G$ -learning (Fox et al.)² are two off-policy methods for solving the KL-regularized RL problem, where additional  $\Psi$  and  $G$  functions are defined and learned instead of  $Q$ . We implement both of these algorithms as well, treating the prior policy as the conditional distribution  $p(a|s)$  defined by the trained Note RNN. To the best of our knowledge, this is the first application of KL-regularized off-policy methods with deep neural networks to sequence modeling tasks. The two methods are given below respectively,

$$
L (\theta) = \mathbb {E} _ {\beta} \left[ \left(\log p (a | s) + r _ {M T} (s, a) / c + \gamma \log \sum_ {a ^ {\prime}} e ^ {\Psi \left(s ^ {\prime}, a ^ {\prime}; \theta^ {-}\right)} - \Psi (s, a; \theta)\right) ^ {2} \right] \tag {11}
$$

$$
\pi_ {\theta} (a | s) \propto e ^ {\Psi (s, a; \theta)} \tag {12}
$$

$$
L (\theta) = \mathbb {E} _ {\beta} \left[ \left(r _ {M T} / c (s, a) + \gamma \log \sum_ {a ^ {\prime}} e ^ {\log p \left(a ^ {\prime} \mid s ^ {\prime}\right) + G \left(s ^ {\prime}, a ^ {\prime}; \theta^ {-}\right)} - G (s, a; \theta)\right) ^ {2} \right] \tag {13}
$$

$$
\pi_ {\theta} (a | s) \propto p (a | s) e ^ {G (s, a; \theta)}. \tag {14}
$$

Both methods can be seen as instances of KL-regularized deep Q-learning, and they also subsume entropy-regularized deep Q-learning by removing the  $\log p(a|s)$  term. The main difference between the two methods is the definition of the action-value functions  $\Psi$  and  $G$ . In fact  $G$ -learning can be directly derived from  $\Psi$ -learning by reparametrizing  $\Psi(s,a) = \log p(a|s) + G(s,a)$ . The  $G$ -function does not give the policy directly but instead needs to be dynamically mixed with the prior policy probabilities. While this computation is straightforward for discrete action domains as here, extensions to continuous action domains require additional considerations such as normalizability of advantage function parametrizations (Gu et al., 2016). The SOC-based derivation also has another benefit in that the stochastic policies can be directly used as an exploration strategy, instead of heuristics such as  $\epsilon$ -greedy or additive noise (Mnih et al., 2013; Lillicrap et al., 2016). The derivations for both methods are included in the appendix for completeness.

# 3.2 MUSIC-THEORY BASED REWARD

A central question of this paper is whether RL can be used to constrain a sequence learner such that the sequences it generates adhere to a desired structure. To test this hypothesis, we developed several rules that we believe describe more pleasant-sounding melodies, taking inspiration from a text on melodic composition (Gauldin, 1995). We do not claim these characteristics are exhaustive, strictly necessary for good composition, or even particularly interesting. They simply serve the purpose of guiding the model towards traditional composition structure. It is therefore crucial to apply the RL Tuner framework to retain the knowledge learned from real songs in the training data.

Following the principles set out on page 42 of Gauldin's book (Gauldin, 1995), we define the reward function  $r_{MT}(a,s)$  to encourage compositions to have the following characteristics. All notes should belong to the same key, and the composition should begin and end with the tonic note of the key; e.g. if the key is C-major, this note would be middle C. This note should occur in the first beat and last 4 beats of the composition. Unless a rest is introduced or a note is held, a single tone should not be repeated more than four<sup>3</sup> times in a row. To encourage variety, we penalize the model if the

composition is highly correlated with itself at a lag of 1, 2, or 3 beats. The penalty is applied when the auto-correlation coefficient is greater than .15. The composition should avoid awkward intervals like augmented 7ths, or large jumps of more than an octave. Gauldin also indicates good compositions should move by a mixture of small steps and larger harmonic intervals, with emphasis on the former; the reward values for intervals reflect these requirements. When the composition moves with a large interval (a 5th or more) in one direction, it should eventually be resolved by a leap back or gradual movement in the opposite direction. Leaping twice in the same direction is negatively rewarded. The highest note of the composition should be unique, as should the lowest note. Finally, the model is rewarded for playing motifs, which are defined as a succession of notes representing a short musical "idea"; in our implementation, a bar of music with three or more unique notes. Since repetition has been shown to be key to emotional engagement with music (Livingstone et al., 2012), we also sought to train the model to repeat the same motif within a composition.

# 4 RELATED WORK

Generative modeling of music with RNNs has been explored in a variety of contexts, including generating Celtic folk music (Sturm et al., 2016), or performing Blues improvisation (Eck & Schmidhuber, 2002). Other approaches have examined RNNs with richer expressivity, latent-variables for notes, or raw audio synthesis (Boulanger-Lewandowski et al., 2012; Gu et al., 2015; Chung et al., 2015). Recently, impressive performance in generating music from raw audio has been attained with convolutional neural networks with receptive fields at various time scales (Dieleman et al., 2016).

Although the application of RL to RNNs is a relatively new area, recent work has attempted to combine the two approaches. MIXER (Mixed Incremental Cross-Entropy Reinforce) (Ranzato et al., 2015) uses BLEU score as a reward signal to gradually introduce a RL loss to a text translation model. After initially training the model using cross-entropy, the training process is repeated using cross-entropy loss for the  $T - \Delta$  tokens in a sequence (where  $T$  is the length of the sequence), and using RL for the remainder of the sequence. Another approach (Bahdanau et al., 2016) applies an actor-critic method and uses BLEU score directly to train a critic network to output the value of each word, where the actor is again initialized with the policy of an RNN trained with next-step prediction. Reward-augmented maximum likelihood (Norouzi et al., 2016) augments the standard ML with a sequence-level reward function and connects it with the above RL training methods. These approaches assume that the complete task reward specification is available. They pre-train a good policy with supervised learning so that RL can be used to learn with the true task objective, since training with RL from scratch is difficult. RL Tuner instead only uses rewards to correct certain properties of the generated data, while learning most information from data. This is important since in many sequence modeling applications such as music or language generation, the true reward function is not available or imperfect and ultimately the model should rely on learning from data. The RL Tuner method provides an elegant and flexible framework for correcting undesirable behaviors of RNNs that can arise from limited training data or imperfect training algorithms.

SeqGAN (Yu et al., 2016) applies RL to an RNN by using a discriminator network — similar to those used in Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) — to classify the realism of a complete sequence, and this classifier-based reward is used as a reward signal to the RNN. The approach is applied to a number of generation problems, including music generation. Although the model obtained improved MSE and BLEU scores on the Nottingham music dataset, it is not clear how these scores map to the subjective quality of the samples (Huszár, 2015), and no samples are provided with the paper. In contrast, we provide both samples and quantitative results demonstrating that our approach improves the metrics defined by the reward function. Further, we show that RL Tuner can be used to explicitly correct undesirable behaviors of an RNN, which could be useful in a broad range of applications.

Our work also relates to stochastic optimal control (SOC), in particular the two off-policy methods,  $\Psi$ -learning (Rawlik et al., 2012) and  $G$ -learning (Fox et al.). Both approaches solve a KL-regularized RL problem, in which a term is introduced to the reward objective to penalize KL divergence from some prior policy. While our methods rely on similar derivations presented in these papers, there are some key differences. First, these techniques have not been applied to DQNs or RNNs, or as a way to fine-tune a pre-trained RNN with additional desired characteristics. Secondly, our methods have different motivations and forms from the original papers: original

$\Psi$ -learning (Rawlik et al., 2012) restricts the prior policy to be the policy at the previous iteration and solves the original RL objective with conservative, KL-regularized policy updates, similar to conservative policy gradient methods (Kakade, 2001; Peters et al., 2010; Schulman et al., 2015). The original  $G$ -learning (Fox et al.) penalizes divergence from a simple uniform prior policy in order to cope with over-estimation of target  $Q$  values, and includes scheduling for the temperature parameter  $c$ . Lastly, our work includes the  $Q$ -learning objective with additional cross-entropy reward as a comparable alternative, and provides for the first time comparisons among the three methods for incorporating prior knowledge in RL.

# 5 EXPERIMENTS

To train the Note RNN, we extract monophonic melodies from a corpus of 30,000 MIDI songs. Melodies are quantized at the granularity of a sixteenth note, so each time step corresponds to one sixteenth of a bar of music. We encode a melody using two special events plus three octaves of notes. The special events are used to introduce rests and notes with longer durations, and are encoded as  $0 = \text{note off}$ ,  $1 = \text{no event}$ . Three octaves of pitches, starting from MIDI pitch 48, are then encoded as  $2 = \text{C3}$ ,  $3 = \text{C#3}$ ,  $4 = \text{D3}$ , ...,  $37 = \text{B5}$ . For example, the sequence  $\{4,1,0,1\}$  encodes an eighth note with pitch D3, followed by an eighth note rest. As the melodies are monophonic, playing another note implicitly ends the last note that was played without requiring an explicit note off event. Thus the sequence  $\{2,4,6,7\}$  encodes a melody of four sixteenth notes: C3, D3, E3, F3. A length-38 one-hot encoding of these values is used for both network input and network output.

The Note RNN consists of one LSTM layer of 100 cells, and was trained for 30,000 iterations with a batch size of 128. Optimization was performed with Adam (Kingma & Ba, 2014), and gradients were clipped to ensure the L2 norm was less than 5. The learning rate was initially set to .5, and a momentum of 0.85 was used to exponentially decay the learning rate every 1000 steps. To regularize the network, a penalty of  $\beta = 2.5 \times 10^{-5}$  was applied to the L2 norm of the network weights. Finally, the losses for the first 8 notes of each sequence were not used to train the model, since it cannot reasonably be expected to accurately predict them with no context. The trained Note RNN eventually obtained a validation accuracy of  $92\%$  and a log perplexity score of .2536.

The learned weights of the Note RNN were used to initialize the three sub-networks in the RL Tuner model. Each RL Tuner model was trained for 1,000,000 iterations, using the Adam optimizer, a batch size of 32, and clipping gradients in the same way. The reward discount factor was  $\gamma = .5$ . The Target- $Q$ -network's weights  $\theta^{-}$  were gradually updated to be similar to those of the  $Q$ -network  $(\theta)$  according to the formula  $(1 - \eta)\theta^{-} + \eta \theta$ , where  $\eta = .01$  is the Target- $Q$ -network update rate. We replicated our results for a number of settings for the weight placed on the music-theory rewards,  $c$ ; we present results for  $c = .5$  below because we believe them to be most musically pleasing. Similarly, we replicated the results using both  $\epsilon$ -greedy and Boltzmann exploration, and present the results using  $\epsilon$ -greedy exploration below.

We compare three methods for implementing RL Tuner:  $Q$ -learning;  $\Psi$ -learning and  $G$ -learning, where the policy defined by the trained Note RNN is used as the cross entropy reward in  $Q$ -learning and the prior policy in  $G$ - and  $\Psi$ -learning. These approaches are compared to both the original performance of the Note RNN, and a model trained using only RL and no prior policy. Model evaluation is performed every 100,000 training epochs, by generating 100 compositions and assessing the average  $r_{MT}$  and  $\log p(a|s)$ .

All of the code for RL Tuner, including a checkpointed version of the trained Note RNN is available at https://github.com/natashamjaques/magenta/tree/rl-tuner.

# 6 RESULTS

Table 1 provides quantitative results in the form of performance on the music theory rules to which we trained the model to adhere; for example, we can assess the fraction of notes played by the model which belonged to the correct key, or the fraction of melodic leaps that were resolved. The statistics were computed by randomly generating 100,000 compositions from each model.

The results above demonstrate that the application of RL is able to correct almost all of the targeted "bad behaviors" of the Note RNN, while improving performance on the desired metrics. For exam-

<table><tr><td>Metric</td><td>Note RNN</td><td>Q</td><td>Ψ</td><td>G</td></tr><tr><td>Notes excessively repeated</td><td>63.3%</td><td>0.0%</td><td>0.02%</td><td>0.03%</td></tr><tr><td>Mean autocorrelation - lag 1</td><td>-.16</td><td>-.11</td><td>-.10</td><td>.55</td></tr><tr><td>Mean autocorrelation - lag 2</td><td>.14</td><td>.03</td><td>-.01</td><td>.31</td></tr><tr><td>Mean autocorrelation - lag 3</td><td>-.13</td><td>.03</td><td>.01</td><td>17</td></tr><tr><td>Notes not in key</td><td>0.1%</td><td>1.00%</td><td>0.60%</td><td>28.7%</td></tr><tr><td>Compositions starting with tonic</td><td>0.9%</td><td>28.8%</td><td>28.7%</td><td>0.0%</td></tr><tr><td>Leaps resolved</td><td>77.2%</td><td>91.1%</td><td>90.0%</td><td>52.2%</td></tr><tr><td>Compositions with unique max note</td><td>64.7%</td><td>56.4%</td><td>59.4%</td><td>37.1%</td></tr><tr><td>Compositions with unique min note</td><td>49.4%</td><td>51.9%</td><td>58.3%</td><td>56.5%</td></tr><tr><td>Notes in motif</td><td>5.9%</td><td>75.7%</td><td>73.8%</td><td>69.3%</td></tr><tr><td>Notes in repeated motif</td><td>0.007%</td><td>0.11%</td><td>0.09%</td><td>0.01%</td></tr></table>

Table 1: Statistics of music theory rule adherence based on 100,000 randomly initialized compositions generated by each model. The top half of the table contains metrics that should be near zero, while the bottom half contains metrics that should increase. Bolded entries represent significant improvements over the Note RNN baseline.

ple, the original LSTM model was extremely prone to repeating the same note; after applying RL, we see that the number of notes belonging to some excessively repeated segment has dropped from  $63\%$  to nearly  $0\%$  in all of the RL Tuner models. While the metrics for the G model did not improve as consistently, the  $Q$  and  $\Psi$  models successfully learned to play in key, resolve melodic leaps, and play motifs. The number of compositions that start with the tonic note has also increased, composition auto-correlation has decreased, and repeated motifs have increased slightly. The degree of improvement on these metrics is related to the magnitude of the reward given for the behavior. For example, a strong penalty of -100 was applied each time a note was excessively repeated, while a reward of only 3 was applied at the end of a composition for unique extrema notes (which most likely explains the lack of improvement on this metric). The reward values could be adjusted to improve the metrics further, however we found that these values produced the most pleasant compositions.

![](images/ecf672ad5484b3ad4933793483e4b761ae93fa527ae6c22cc33f998788509dd1.jpg)  
(a) Note RNN reward:  $\log p(a|s)$

![](images/db9cd6972567e801eb6f0e7257c2bd668f70119927174d28ab421d53ca06871c.jpg)  
(b) Music theory reward  
Figure 2: Average reward obtained by sampling 100 compositions every 100,000 training epochs. The three models are compared to a model trained using only the music theory rewards  $r_{MT}$ .

While the metrics indicate that the targeted behaviors of the RNN have improved, it is not clear whether the models have retained information about the training data. Figure 2a plots the average  $\log p(a|s)$  as produced by the Reward RNN for compositions generated by the models every 100,000 training epochs; Figure 2b plots the average  $r_{MT}$ . Included in the plots is an RL only model trained using only the music theory rewards, with no information about  $\log p(a|s)$ . Since each model is initialized with the weights of the trained Note RNN, we see that as the models quickly learn to adhere to the music theory constraints,  $\log p(a|s)$  falls from its initial point. For the RL only model,  $\log p(a|s)$  reaches an average of -3.65, which is equivalent to an average  $p(a|s)$  of approximately 0.026. Since there are 38 actions, this represents essentially a random policy with respect to the distribution defined by the Note RNN. Figure 2a shows that each of our models ( $Q$ ,  $\Psi$ , and  $G$ ) attain higher  $\log p(a|s)$  values than this baseline, indicating they have maintained information about the data probabilities. The  $G$ -learning implementation scores highest on this metric, at the cost of slightly lower average  $r_{MT}$ . This compromise between data probability and adherence to music theory could explain the  $G$  model's poorer performance on the music theory metrics in Table 1. Finally, while  $c = 0.5$  produced compositions that sounded better subjectively, we found that by increasing the  $c$  parameter it is possible to train all the models to have even higher average  $\log p(a|s)$ .

The question remains whether the RL-tuned models actually produce more pleasing melodies. To answer it, we conducted a user study via Amazon Mechanical Turk in which participants were asked to rate which of two randomly selected compositions they preferred on a Likert scale. A total of 192 ratings were collected; each model was involved in 92 of these comparisons. Figure 3 plots the number of comparisons in which a composition from each model was selected as the most musically pleasing. A Kruskal-Wallis H test of the ratings showed that there was a statistically significant difference between the models,  $\chi^2 (3) = 109.480$ ,  $p < 0.001$ . Mann-Whitney U post-hoc tests revealed that the compositions from all three RL Tuner models  $(Q,\Psi ,$  and  $G)$  had significantly higher

ratings than the compositions of the Note RNN,  $p < .001$ . The  $Q$  and  $\Psi$  compositions were also rated as significantly more pleasing than those of the  $G$  model, but did not differ significantly from each other. The sample compositions used for the study are available here: goo.gl/XIYt9m; we encourage readers to judge their quality for themselves.

![](images/451f20d3d2c2bba1e5421eae39716113362b76b6efa6b99dbbd88d177bed6af3.jpg)  
Figure 3: The number of times a composition from each model was selected as most musically pleasing in a comparison. Error bars reflect the standard deviation of a binomial distribution fit to binary win/loss data from each model.

Listening to the samples produced by the Note RNN reveals that they are sometimes dischordant and usually dull; the model tends to place rests frequently, repeat the same note, and produce melodies with little variation. In contrast, the melodies produced by the RL Tuner models are more varied and interesting. The  $G$  model tends to produce energetic and chaotic compositions, which include sequences of repeated notes. This repetition is likely because the G policy as defined in Eq. 14 directly mixes  $p(a|s)$  with the output of the G network, and the Note RNN strongly favours repeating notes. The most pleasant-sounding compositions are generated by the  $Q$  and  $\Psi$  models. These melodies stay firmly in key and frequently choose more harmonious interval steps, leading to melodic and pleasant compositions. However, it is clear they have retained information about the training data; for example, the sample q2.wav in the sample directory ends with a seemingly familiar riff.

# 7 DISCUSSION AND FUTURE WORK

We have derived a novel sequence learning framework which uses RL rewards to correct properties of sequences generated by an RNN, while keeping much of the information learned from supervised training on data. We proposed and evaluated three alternative techniques for achieving this, and showed promising results on music generation tasks.

In addition to the ability to train models to generate pleasant-sounding melodies, we believe our approach of using RL to refine RNN models could be promising for a number of applications. For example, it is well known that a common failure mode of RNNs is to repeatedly generate the same token. In text generation and automatic question answering, this can take the form of repeatedly generating the same response (e.g. "How are you?")  $\rightarrow$  "How are you?"  $\rightarrow$  "How are you?" ...). We have demonstrated that with our approach we can correct for this unwanted behavior, while still maintaining information that the model learned from data. Although manually writing a reward function may seem unappealing to those who believe in training models end-to-end based only on data, that approach it is limited by the quality of the data that can be collected. If the data contains hidden biases, this can lead to highly undesirable consequences. Recent research has shown that the word2vec embeddings in popular language models trained on standard corpora consistently contain the same harmful biases with respect to race and gender that are revealed by implicit association tests on humans (Caliskan-Islam et al., 2016). In contrast to relying solely on possibly biased data, our approach allows for encoding high-level domain knowledge into the RNN, providing a general, alternative tool for training sequence models.

# ACKNOWLEDGMENTS

This work was supported by Google Brain, the MIT Media Lab Consortium, and Canada's Natural Sciences and Engineering Research Council (NSERC). We thank Greg Wayne, Sergey Levine, and Timothy Lillicrap for helpful discussions on stochastic optimal control.

# REFERENCES

Bahdanau et al. An actor-critic algorithm for sequence prediction. arXiv preprint: 1607.07086, 2016.  
Boulanger-Lewandowski, Bengio, and Vincent. Modeling temporal dependencies in high-dimensional sequences: Application to polyphonic music generation and transcription. arXiv preprint:1206.6392, 2012.  
Caliskan-Islam, Bryson, and Narayanan. Semantics derived automatically from language corpora necessarily contain human biases. arXiv preprint:1608.07187, 2016.  
Chung, Kastner, Dinh, Goel, Courville, and Bengio. A recurrent latent variable model for sequential data. In NIPS, pp. 2980-2988, 2015.  
Dieleman et al. Wavenet: A generative model for raw audio. arXiv preprint:1609.03499, 2016.  
Eck and Schmidhuber. Finding temporal structure in music: Blues improvisation with LSTM recurrent networks. In Neural Networks for Signal Processing, pp. 747-756. IEEE, 2002.  
Fox, Pakman, and Tishby. Taming the noise in reinforcement learning via soft updates.  
Gauldin. A practical approach to eighteenth-century counterpoint. Waveland Pr Inc, 1995.  
Gers, Schmidhuber, and Cummins. Learning to forget: Continual prediction with LSTM. Neural computation, 12(10):2451-2471, 2000.  
Goodfellow et al. Generative adversarial nets. In NIPS, pp. 2672-2680, 2014.  
Graves. Generating sequences with recurrent neural networks. arXiv preprint:1308.0850, 2013.  
Graves and Schmidhuber. Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural Networks, 18(5):602-610, 2005.  
Gu, Ghahramani, and Turner. Neural adaptive sequential monte carlo. In NIPS, pp. 2629-2637, 2015.  
Gu, Lillicrap, Sutskever, and Levine. Continuous Deep Q-Learning with model-based acceleration. In ICML, 2016.  
Van Hasselt, Guez, and Silver. Deep reinforcement learning with double Q-learning. CoRR, abs/1509.06461, 2015.  
Huszár. How (not) to train your generative model: Scheduled sampling, likelihood, adversary? arXiv preprint:1511.05101, 2015.  
Kakade. A natural policy gradient. In NIPS, volume 14, pp. 1531-1538, 2001.  
Kappen, Gomez, and Opper. Optimal control as a graphical model inference problem. Machine learning, 87(2):159-182, 2012.  
Kingma and Ba. Adam: A method for stochastic optimization. arXiv preprint:1412.6980, 2014.  
Lillicrap et al. Continuous control with deep reinforcement learning. *ICLR*, 2016.  
Livingstone, Palmer, and Schubert. Emotional response to musical repetition. Emotion, 12(3):552, 2012.  
Mikolov et al. Recurrent neural network based language model. In Interspeech, volume 2, pp. 3, 2010.

Mnih et al. Playing atari with deep reinforcement learning. arXiv preprint:1312.5602, 2013.  
Norouzi et al. Reward augmented maximum likelihood for neural structured prediction. arXiv preprint:1609.00150, 2016.  
Peters, Müller, and Altun. Relative entropy policy search. In AAAI. Atlanta, 2010.  
Ranzato, Chopra, Auli, and Zaremba. Sequence level training with recurrent neural networks. arXiv preprint:1511.06732, 2015.  
Rawlik, Toussaint, and Vijayakumar. On stochastic optimal control and reinforcement learning by approximate inference. Proceedings of Robotics: Science and Systems VIII, 2012.  
Schulman, Levine, Moritz, Jordan, and Abbeel. Trust region policy optimization. In ICML, 2015.  
Sturm, Santos, Ben-Tal, and Korshunova. Music transcription modelling and composition using deep learning. arXiv preprint:1604.08723, 2016.  
Sutton et al. Policy gradient methods for reinforcement learning with function approximation. In NIPS, volume 99, pp. 1057-1063, 1999.  
Todorov. Linearly-solvable markov decision problems. In NIPS, pp. 1369-1376, 2006.  
Watkins and Dayan. Q-learning. Machine learning, 8(3-4):279-292, 1992.  
Yu, Zhang, Wang, and Yu. SeqGAN: Sequence generative adversarial nets with policy gradient. arXiv preprint:1609.05473, 2016.
