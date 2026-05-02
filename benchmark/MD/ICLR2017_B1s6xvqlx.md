# RECURRENT ENVIRONMENT SIMULATORS

Silvia Chiappa, Sébastien Racaniere, Daan Wierstra & Shakir Mohamed DeepMind, London

{csilvia, sracaniere, wierstra, shakir}@google.com

# ABSTRACT

Agent-based systems require multiple mechanisms with which to understand their environments. One such mechanism is the ability to imagine and deliberate upon the effect of their actions on the environment. To equip agents with this ability, we develop a system that allows accurate models of an environment to be learned, and use this model to simulate the future evolution of the environment. We introduce an action-conditional recurrent neural network that addresses three shortcomings of existing approaches: actions in our model affect transition dynamics directly; our model is able to make spatially and temporally coherent, high-dimensional image predictions for hundreds of steps into the future; our model can make future predictions of the environment without needing to generate all intermediate observations. We show that the same model is adaptable to many diverse environments, namely complex 3D mazes, 3D racing car simulators, and at least ten Atari games. We also show that our model can be used to improve exploration in complex maze environments.

# 1 INTRODUCTION

An understanding of the temporal nature of the world is an essential requirement for agent-based systems. Such an understanding can be obtained by building detailed models of temporal data streams that equip these systems with internal simulators of their environments, allowing them to anticipate the consequences of their actions and make predictions many steps into the future. The need for model building and environment simulation is widespread: in psychology, model-based predictive abilities form sensorimotor contingencies that are seen as essential for perception (O'Regan & Noë, 2001); in neuroscience, environment simulation forms part of deliberative planning systems used by the brain (Niv, 2009); and in reinforcement learning, the ability to imagine the future evolution of the environment is needed to form predictive state representations (Littman et al., 2002) and for Monte Carlo planning (Sutton & Barto, 1998). As we continue to develop richer decision-making systems, the ability to learn robust, general-purpose and scalable models of real-world environments becomes increasingly important.

Simulating an environment requires highly flexible models of temporal sequences that must possess a number of properties to be useful: the models should make predictions that are accurate over long time periods into the future; provide temporally and spatially coherent predictions that capture the causal structure and physical constraints of their environments; and allow for flexibility in the policies and action sequences that are used. In addition, these models should be general-purpose, learn successfully from diverse and realistic sets of environments, be scalable, and able to learn from high-dimensional perceptual inputs. A model that achieves these desiderata can empower agent-based systems with a vast array of abilities, including counterfactual reasoning (Pearl, 2009), intuitive physical reasoning (McCloskey, 1983), model-based exploration, episodic control (Lengyel & Dayan, 2008), intrinsic motivation (Oudeyer et al., 2007), and hierarchical control.

In this paper we develop models that can be widely deployed as environment simulators. To capture the temporal structure of data streams, most approaches form a dynamical system with a hidden Markov structure. The state-of-the-art methods build dynamical models using recurrent networks. Wahlström et al. (2015) learn dynamics models of robotic environments from pixel information by designing an action-conditional recurrent neural network and a policy using model-predictive control. Watter et al. (2015) also explore models of robotic systems from pixel information, and develop an action-conditional recurrent model using variational autoencoders. Sun et al. (2015) build models that use predictive state representations for robotic environments. Fragkiadaki et al. (2015) develop a model of physical scenes from vision, in particular of billiard balls, that uses neural networks to

![](images/68440de34f3d72e0975db0d5097a093dc7bb2ac9d8c82ab4b4dd63c1e4f35c58.jpg)  
Figure 1: Graphical model representing (a) the recurrent structure used in (Oh et al., 2015) and (b) our recurrent structure.

![](images/af12e5e1985dbf9fb34156b51b5601f6774b74c3e26b0537c5087aff41cabf35.jpg)

![](images/c9826bc139b671570cdc2ee8d340d43fd5a1a5dd928f0a45d20320d2f1016a7f.jpg)

learn invariant physical laws that allow for long-term predictions. Patraucean et al. (2015) develop a spatio-temporal model of video that uses optical flow to model the temporal dynamics. Amongst the best performing approaches is Oh et al. (2015), who develop an action-conditional recurrent network from pixel information that is able to make predictions over many time-steps in five Atari 2600 games, and will form the basis of our comparison.

These existing approaches have two principal limitations. Firstly, they are not always able or designed to make reliable long-term predictions across many different environments. Secondly, these models must make predictions of future visual inputs at every time point, which is computationally inefficient. To overcome these limitations we develop a robust action-conditional recurrent neural network and training algorithm that allows us to learn accurate, high-dimensional predictions hundreds of steps into the future (section 2). We show that our model works across diverse environments, namely 10 Atari 2600 games, a first-person game where an agent moves in randomly generated 3D mazes, and a 3D car racing simulator. We provide a detailed analysis of our approach (section 3.1), emphasising the implications of different algorithmic choices and difficulties in training. Our model makes it possible to generate roll-outs without making pixel-level predictions at every time point, which is needed for problems in jumpy and hierarchical planning (section 3.2). We demonstrate the applicability of the model for model-based exploration (section 5), discuss some of the limitations and ways to address them in future.

# 2 RECURRENT ENVIRONMENT SIMULATORS

To equip an agent with the ability to internally simulate its environment, we build a model that, given a sequence of actions  $a_1, \ldots, a_{\tau - 1} \equiv a_{1:\tau - 1}$  and corresponding observations  $\mathbf{x}_{1:\tau}$  of the environment, is able to predict the effect of subsequent actions  $a_{\tau: \tau'}$ . We restrict ourselves to deterministic environments and discrete actions, and assume the observations to be in the form of images. To be useful, this model should produce accurate predictions for long time-spans into the future, and that are both spatially and temporally coherent. Furthermore, this model should make rapid decision-making possible. For many problems, the agent is interested only in the final state of the environment after executing a sequence of actions. In such cases, a model that can predict only the final outcome would greatly reduce the computational burden and make rapid decision-making possible; such jumpy, rather than single-step, predictors have up to now only been developed in low-dimensional observation spaces.

The most successful approaches for building next-step predictors from pixel information use neural networks that, at each time-step  $t$ , create a lower-dimensional representation  $\mathbf{h}_t$  of the environment by encoding past frames through a series of convolutions, and makes a prediction  $\hat{\mathbf{x}}_t$  by decoding  $\mathbf{h}_t$  through a series of full convolutions. In Oh et al. (2015), which represents the current state-of-the-art for the Atari environment,  $\mathbf{h}_t$  is formed using either a feed-forward or a recurrent network, and combined with actions using a multiplicative interaction before the decoding is applied. Fig. 1(a) gives a graphical model representation of the recurrent network for a single time-step transition. Nodes in the graph represent random variables and links represent direct dependence between the connected variables. All dependencies are deterministic, except the link from  $\hat{\mathbf{x}}_t$  to  $\mathbf{x}_t$  which is subject to zero mean and unit variance Gaussian noise, i.e.  $p(\mathbf{x}_t|\hat{\mathbf{x}}_t) = \mathcal{N}(\hat{\mathbf{x}}_t,\mathbf{I})$ . The dashed lines indicate that only one of the two links is active, depending on whether the predicted frame  $\hat{\mathbf{x}}_{t - 1}$  or the observed frame  $\mathbf{x}_{t - 1}$  is used to form the state  $\mathbf{h}_t$ . The model is trained by maximizing the

log-likelihood over  $N$  sequences of observations  $\sum_{n}\log p(\mathbf{x}_{\tau +1:T}^{n}|\mathbf{x}_{1:\tau}^{n})$ , i.e. minimising the mean squared error between predictions and observations using stochastic gradient decent. There are three limitations of this model that we aim to address. Firstly, while the structure allows a standard recurrent architecture, such as a long short-term memory (LSTM) (Hochreiter & Schmidhuber, 1997) to be used for the state update, this only allows for an indirect dependency of the hidden state on the action, through the high-dimensional prediction  $\hat{\mathbf{x}}_t$  or observation  $\mathbf{x}_t$ . A second limitation is that, whilst achieving state-of-the-art predictions, in many games this model is only accurate for a relatively small number of time-steps. Thirdly, following from the choice of action-dependency, this model does not support a jumpy mode of prediction: to form  $\hat{\mathbf{x}}_{\tau '}$ , the model must always make predictions  $\hat{\mathbf{x}}_{\tau +1:\tau^{\prime} - 1}$ .

We overcome these limitations with the model shown in Fig. 1(b). The choice of dependency (indicated by the dashed lines) on the prediction  $\hat{\mathbf{x}}_t$  or observation  $\mathbf{x}_t$  results in three modes of use: in the observation-dependent mode, the state  $\mathbf{h}_t$  is formed using the observed frame  $\mathbf{x}_{t-1}$ ; in the prediction-dependent mode, the state is formed using the predicted frame  $\hat{\mathbf{x}}_{t-1}$ ; in the frame-independent mode, the state does not depend on the frame. By using a training scheme that uses mostly the prediction-dependent mode, we address the limitation of short-term-only accuracy. In our model, the action directly influences  $\mathbf{h}_t$ , allowing propagation of information from  $t-1$  to  $t$  that depends directly on  $a_{t-1}$ . Finally, jumpy predictions can be obtained by removing both the link from  $\hat{\mathbf{x}}_{t-1}$  and the link from  $\mathbf{x}_{t-1}$  to  $\mathbf{h}_t$ . The prediction of our model from time-step  $t-1$  to  $t$  and the objective are described by:

$$
\text {E n c o d i n g}: \mathbf {s} _ {t - 1} = \left\{ \begin{array}{l l} \operatorname {C o n v} \left(\mathbb {I} \left(\hat {\mathbf {x}} _ {t - 1}, \mathbf {x} _ {t - 1}\right)\right) & \text {F r a m e - d e p e n d e n t m o d e} \\ \mathbf {h} _ {t - 1} & \text {F r a m e - i n d e p e n d e n t m o d e} \end{array} \right. \tag {1}
$$

$$
\text {A c t i o n} \mathbf {v} _ {t} = F \left(\mathbf {h} _ {t - 1}, a _ {t - 1}\right), \tag {2}
$$

$$
\text {G a t e} \mathbf {i} _ {t} = \sigma \left(L ^ {i v} (\mathbf {v} _ {t}) + L ^ {i s} \left(\mathbf {s} _ {t - 1}\right)\right), \quad \mathbf {f} _ {t} = \sigma \left(L ^ {f v} (\mathbf {v} _ {t}) + L ^ {f s} \left(\mathbf {s} _ {t - 1}\right)\right)
$$

$$
\mathbf {o} _ {t} = \sigma \left(L ^ {o v} \left(\mathbf {v} _ {t}\right) + L ^ {o s} \left(\mathbf {s} _ {t - 1}\right)\right) \tag {3}
$$

$$
\text {C e l l} \mathbf {c} _ {t} = f _ {t} \otimes \mathbf {c} _ {t - 1} + i _ {t} \otimes \tanh  \left(L ^ {c v} (\mathbf {v} _ {t}) + L ^ {c s} \left(\mathbf {s} _ {t - 1}\right)\right) \tag {4}
$$

$$
\text {S t a t e} \quad \mathbf {h} _ {t} = o _ {t} \otimes \tanh  \left(\mathbf {c} _ {t}\right) \tag {5}
$$

$$
\text {D e c o d i n g :} \hat {\mathbf {x}} _ {t} = \operatorname {F u l l C o n v} \left(\mathbf {h} _ {t}\right) \tag {6}
$$

$$
\text {O b j e c t i v e :} \mathcal {L} = \frac {1}{N} \frac {1}{T - \tau} \sum_ {n = 1} ^ {N} \sum_ {t = \tau + 1} ^ {T} \| \mathbf {x} _ {t} ^ {n} - \hat {\mathbf {x}} _ {t} ^ {n} \| ^ {2}. \tag {7}
$$

Observations are first mapped, in equation (1), to an embedding  $\mathbf{s}_t$ . This embedding depends on which mode of the model we are employing: for single-step models,  $\mathbb{I}$  selects the observed frame (observation-dependent mode) or predicted frame (prediction-dependent mode), and for jumpy models we simply pass the previous state  $\mathbf{h}_{t - 1}$  as the embedding (frame-independent mode). Equations (2)-(5) represent the action-dependent LSTM, where  $\mathbf{h}_t$  is the state and  $\mathbf{c}_t$  the cell;  $\mathbf{i}_t,\mathbf{f}_t$  and  $\mathbf{o}_t$  are the input, forget, and output gates, respectively;  $\otimes$  the Hadamard product and  $\sigma$  the logistic sigmoid function. The functions  $L$  represent linear mappings (fully-connected linear, or dimension-preserving convolutions), and we denote them with different superscripts to indicate that they do not share parameters. For the action fusion (2), we investigated using either a multiplicative interaction  $\mathbf{W}^h\mathbf{h}_{t - 1}\otimes \mathbf{W}^a\mathbf{a}_{t - 1}$  using parameters  $\mathbf{W}$ , where  $\mathbf{a}_{t - 1}$  is the one-hot vector representation of  $a_{t - 1}$ , or a dimension-preserving convolution with different parameters for each action. We use a set of  $\tau$  warm-up actions and observations to initialise the state and cell. More details about the model structure, such as the exact forms of the encoding and decoding, and the training algorithm are given in the Appendix.

Training. We shall refer to models that use the observation-dependent or prediction-dependent modes, as single-step simulators, since these models must generate frames at every time-step to make a prediction in the future. Models trained using the frame-independent mode are jumpy simulators, since predictions at any point in the future can be made through time-step transitions in the state-space. We obtain simulators with varying short-term and long-term prediction capabilities depending on the length of the warm-up phase  $\tau$  and the length of the sequences  $T$  used, and whether

![](images/8b21bbe81b08b5a03ac0d05732f19ecb21f0b1260ac1b7adfeec4ee7f3d02f74.jpg)  
(a)

![](images/acc38dfafcbc53d74ef04b3f281ba8b93e104c1bbdbab9c07fb1afe1904f3f0f.jpg)  
(b)  
Figure 2: Prediction error on Bowling for different training schemes. (a): Prediction error vs time-steps at 500,000 parameter updates. (b) Prediction error vs parameter updates at time-step 100.

we train single-step or jumpy models. While recurrent neural networks are most often trained in an observation-dependent mode, the importance of using predictions to obtain reliable long-term roll-outs is recognized in the literature and several schemes that mix the use of  $\hat{\mathbf{x}}_{t - 1}$  and  $\mathbf{x}_{t - 1}$ , i.e. mix the use of observation-dependent and prediction-dependent modes, have been proposed (Talvitie, 2014; Bengio et al., 2015; Oh et al., 2015). In scheduled sampling (Bengio et al., 2015), a hyper-parameter  $\epsilon$  is used that is annealed from an initial value of 1 (observation-dependent mode only) to 0 (prediction-dependent mode only). Oh et al. (2015) also use a similar mixed approach with a long observation-dependent phase using  $T = 10$ , followed by two prediction-dependent phases, first using  $T = 3$ , and then using  $T = 5$ .

Our training algorithm follows in the spirit of these previous approaches. We describe the approach that we found to apply to Atari, 3D racing simulators and 3D maze environments. We used a warm-up phase of length  $\tau = 10$ , as our analysis showed that longer lengths do not significantly improve the performance. In the frame-dependent simulators, we did not backpropagate the gradient to this phase, whilst in the frame-independent simulator this is required in order to learn the encoding link – we did that only back to time-step five, to avoid using too inaccurate states and cells. We backpropagated gradients over sequences of length  $T \leq 20$ , since longer sequences were not possible due to memory constraints. Our investigation showed considerable differences in performance between sequence lengths of  $T = 10$  and  $T = 15$ , but no significant difference between lengths  $T = 15$  and  $T = 20$ . We did also use sequences of length greater than 20 by splitting them into subsequences and performing parameter updates separately. For example, to use a sequence of length 40, we split it into two successive subsequences: we performed parameter updates over the first subsequence, initialised the LSTM state and cell of the second subsequence with the final state and cell from the first subsequence, and then performed parameter updates over the second subsequence; a similar strategy is used by Zaremba et al. (2014). This typically led to more accurate long-term predictions but, in the prediction-dependent training scheme, to less accurate short-term predictions. Single-step models that use observation-dependent training only achieved accurate short-term prediction, but very inaccurate long-term prediction. Moving towards prediction-dependent training only, whilst reducing short-term accuracy, improved dramatically the ability of the model to perform accurate long-term prediction.

# 3 ENVIRONMENT SIMULATORS OF ATARI

We considered the 10 Atari games from the arcade learning environment (Bellemare et al., 2013): Freeway, Ms Pacman, Qbert, Seaquest, Space Invaders, Bowling, Breakout, Fishing Derby, Pong, and Riverraid. Of these, the first five were analyzed in Oh et al. (2015) and are used for comparison. The remaining five were chosen to better test the ability of the model, such as scrolling backgrounds (Riverraid), small/thin objects that require accurate modelling to achieve long-term prediction (lines in Fishing Derby, ball in Pong and Breakout), and sparse-reward games that require very long-term prediction (Bowling). We used training and test datasets consisting of five million and one million

![](images/e90a16793e213ecceae9f7a374397d135de3018069edf5f25f69c446d202ecc1.jpg)  
(a)

![](images/2c18c3d7da1bfb84ffe56c20ef637c46a636599ad515aa50d69f2ab8290ed674.jpg)  
(b)  
Figure 3: Prediction error on Fishing Derby vs time-steps at one million parameter updates using different number of subsequences. (a)  $100\%$  Pred. Frames training scheme. (b):  $66\%$  Pred. Frames training scheme.

$210 \times 160$  RGB images respectively, with actions chosen from a trained DQN agent according to an  $\epsilon = 0.2$ -greedy policy. Such a large number of training frames ensured that our simulators did not overfit to the data (see training and test lines in Fig. 2(b) and the discussion in the Appendix).

# 3.1 SINGLE-STEP SIMULATORS

In this section we first present our findings related to the training scheme and compare our best simulators with the one in Oh et al. (2015) using actions selected from the test data. We then evaluate our simulators further by using actions selected by humans. Our analysis indicates that using a fully-connected linear mapping for the action fusion and the LSTM core gives the best performance, therefore we use this configuration as all experiments presented in this and following sections.

Training Scheme Analysis. Fig. 2 shows the prediction error over 10,000 sequences² obtained for the game of Bowling³ using sequences of length  $T = 15$  and the six training schemes in which the internal state was formed using:

1. Only predicted frames (100% Pred. Frames) (prediction-dependent training (PDT)),  
2. Only observed frames in the first 1000 parameter updates, and only predicted frames in the subsequent parameter updates (0%-100% Pred. Frames),  
3. Observed frames for the first 5 time-steps and predicted frames for the last 10 time-steps (66% Pred. Frames),  
4. Only observed frames in the first 10,000 parameter updates; observed frames for the first 12 time-steps and predicted frames for the last 3 time-steps for the subsequent 100,000 parameters updates; observed frames for the first 10 time-steps and predicted frames for the last 5 time-steps for the remaining parameter updates (0%-20%-33% Pred. Frames). This training scheme closely resembles that used by Oh et al. (2015),  
5. Observed frames for the first 10 time-steps and predicted frames for the last 5 time-steps (33% Pred. Frames),  
6. Only observed frames (0% Pred. Frames) (observation-dependent training (ODT)).

More specifically, Fig. 2(a) shows the test error for 200 time-steps, obtained after 500,000 parameter updates using actions and warm-up frames from the test data, whilst Fig. 2(b) shows the error at time-step 100 versus parameter updates, using actions and warm-up frames from the test data (continuous lines) and training data (dashed lines). These figures clearly show that the performance

![](images/0869dfbc73069dc0ba33ecdd6e6da2eb0f9a0e1d3fd1f7f3318e92e425ede6ad.jpg)

![](images/494335050a299c3e014272f9c1167424fbfb0f9331ee98b6c89a60ffa50cfd25.jpg)

![](images/96b94ed8c4384edc348dc45e0a3f56087916e89f5d04f5229acde871fb256a5d.jpg)

![](images/6967458c6868669fcb399342913981ce6409e382433844389acbe41fdcdcd8bb.jpg)

![](images/9dd4b0eb0fa3cee8d684b4c087521d5dcd14b514529c58d896c47f2ff4764887.jpg)

![](images/c19d2965a3dad7f89bb825fb096f9c13ef229813d3a547c213141d60186715a1.jpg)

![](images/9b6179d6a9fabf6848d5af24c05e1be55b8ad902d7ebf65e4df8f7f84f4a218b.jpg)

![](images/eec04468e8039d1b1e2ce2ca50d64cc1460fd3f1e107b2b1faacc5f509a50eb3.jpg)

![](images/c60d61b2d9d3341760616574d27e5da9c55b8bcf829433fba9a372ff1953f058.jpg)

![](images/a30d0cd2e23beee064a46f85605372d4d35dcfb43cee1c954a2d6c035e90c109.jpg)

![](images/cd7677276a3037c18c8fab63b928377e2c31ad3d50d6838b7a1298af930022dd.jpg)

![](images/51e6b2257557848310aec3f289303e9d01180d57d14aa8dcc28fd834fcb78005.jpg)

![](images/6b211bec2d7592926b5e2f656702e82acd2af7a4b415abc95e0002d5bd4cee82.jpg)

![](images/30100b9a71fa48666a59e7406359e801fb86a7d9b2d11b2e9a7840045b13d8ab.jpg)

![](images/480e4aab8fbb0255a4da690b1c1aaebcab68c91c9c1ab9e4cad0a8a9101810ff.jpg)

![](images/0f47e40c9781c29a5e134784b826d829b5ebb570a949d8122565ebf80d6c5233.jpg)  
Figure 4: One example of 200 time-step ahead prediction for each of the 10 Atari games. Displayed are predicted (left) and real (right) frames at time-steps 100 and 200.

![](images/55729e6f8738582a263c576944159482583a7cd509c31f943a97ccadf7cd9934.jpg)

![](images/c0cff0cb6fb992728fb948d0caf1f5528e3cf9f1c7bb6211f122355164f5c62d.jpg)

![](images/7d7e8bffb45844d06f671d44ec00aef97af9308249ab07a2e415f3dadfbc18f9.jpg)

![](images/44e30fb5cb995ea4b322376cdf78e71e4c847d4022587bd67e6ad3935f75e662.jpg)

improves with increasing number of predicted frames in the sequence, but also that having a short observation-dependent training phase does not improve over employing prediction-dependent training only. A qualitative understanding about the difference between observation-dependent and prediction-dependent training can be obtained from the videos in Bowling-ODT $^4$  and Bowling-PDT respectively $^5$ . As we can see in Bowling-ODT, whilst the prediction is generally quite good, the score is not updated correctly, but more importantly the movement of the ball is not always correctly predicted. The training scheme used by Oh et al. (2015) also resulted in more errors in predicting both the score and the ball (see Bowling-0%-20%-33%Pred.Frames).

For other games, observation-dependent training had more drastic consequences. In Fishing Derby, this training resulted in a complete failure of the prediction after a very small number of time-steps (as shown in FDerby-ODT), whilst prediction-dependent training produced much better predictions (as shown in FDerby-PDT). In Seaquest, observation-dependent training gave rise to predictions in which existing fish disappears quickly and no new fish ever appears from the sides of the screen (see Seaquest-ODT), whilst with a prediction-dependent training new fish appear in the right location very often, even if some noise is present (see Seaquest-PDT). In this latter scheme, early in the training the model focuses on learning the exact location of the fish, but the fish is blurry and a lot of noise is present. Later in the training, the model focuses on sharpening the fish, although the fish never gets very sharp. On the other hand, in schemes using more and more observed frames, the fish is sharper and sharper but its location is less and less accurate. In other words, in schemes containing more and more observed frames, the model focuses more on details that produce accurate short-term prediction but neglects aspects that are relevant for accurate long-term prediction – this can be clearly seen in Seaquest-0%-20%-33%Pred.Frames, where the fish is quite sharp but its location is mostly wrong. This short-term-memory focus of observation-dependent training causes no generation of new objects or background in Riverraid (Riverraid-ODT), and prediction of background only after a few time-steps in Qbert (Qbert-ODT), resetting to a new life after a few time-steps in Ms Pacman

![](images/a974ea3dea1ab6eec4455360c36bf0ab7b21574cee82947bea1478b7d9bf16ee.jpg)  
(a)

![](images/87ab138ed08432dee31932cc0e9ab1c06d2d86dac31b6017b40233b46ce745d2.jpg)  
(b)  
Figure 5: (a) Prediction error for our model (continuous lines) and the model in Oh et al. (2015) (stars). (b) Prediction error for the single-step (dashed lines) versus jumpy model (continuous lines).

(MSPacman-ODT), and Space Invaders (SInvaders-ODT). Finally, it produces quite inaccurate paddle and ball prediction in Breakout (Breakout-ODT).

Even in the easier game of Freeway where observation-dependent training reaches good predictions, prediction-dependent training can more robustly model the chicken movement (see Freeway-PDT and Freeway-ODT).

However, the characteristic behavior of prediction-dependent training in which moving objects are less sharp causes problems in modelling Breakout, Ms Pacman, Qbert, and Space Invaders, and therefore better performance was obtained with mixing schemes (see Breakout-66%Pred.Frames, Ms Pacman-66%Pred.Frames, Qbert-66%Pred.Frames, and SInvaders-66%Pred.Frames).

Fig. 3 shows the prediction error obtained on Fishing Derby after one million parameter updates with sequence length of  $T = 15, 30, 75$  and 150, using 1, 2, 5 and 10 subsequences of length 15, respectively. When using prediction-dependent training (Fig. 3(a)), short-term prediction with more than one subsequence is less accurate (these findings are consistent across all games, although differences for short-term prediction are less extreme than for Fishing Derby). On the other hand, when using  $66\%$  Pred. Frames training (Fig. 3(b)), there is no difference in performance in using one or more subsequences (although using more that one subsequence can improve overall performance in some games). It is interesting to note that, for Riverraid, the use of more than one subsequence improves accuracy dramatically as it enables recovery after the agent's death (as shown in Riverraid-PDT). In conclusion, using more than one subsequence can improve performance. However, in schemes that are close to prediction-dependent only training, care must be taken to avoid too hight short-term inaccuracies.

Comparison with Oh et al. (2015). In Fig. 4, we show for each game one example of prediction at time-steps 100 and 200 using our best simulator. In Fig. 5(a), we compare the test prediction error of our simulator (continuous lines) versus that obtained using the model and training of Oh et al. (2015) (stars). As we can see, our simulator performs substantially better overall. By looking at the videos at Oh et al. 2015, it is evident that our simulator produces substantially more accurate long-term predictions: e.g., for Seaquest, Oh et al. (2015) is unable to generate new fish appearing from the right and left sides of the screen at the right frequency and the location of the fish is mostly inaccurate. Even in Freeway, where both models achieve low error, we can notice a substantial difference in the predictions as, unlike our case, Oh et al. (2015) cannot correctly predict the score. This difference in performance is largely due to our architecture and different training scheme. However, considerable difference is also due to the different architecture – in the Appendix we show how our choice of a direct action influence improves accuracy on Seaquest (Fig. 12).

Evaluation through Human Play. Whilst our simulator improves on the state-of-the-art, as shown by the evaluation using actions from the test set, it is not clear how it would perform when using

![](images/6eeba3d6e0531657493fce3b23e7d37a30c2c1a37aea9e06a20c3a88597189a5.jpg)  
(a)

![](images/7babdc25a7ec6f4c326101b9f2e898c911a9376b67440223fef5d3a70b929f22.jpg)  
(b)  
Figure 6: Salient frames extracted from (a) 500 frames of Pong and (b) 350 frames of Breakout generated using our simulator with actions taken by a human player (larger versions can be found in the Appendix).

actions from a policy other than the DQN used in training. Whilst, in general, it is unreasonable to expect the model to generalize to (sequences of) actions never chosen by the DQN, such as to moving the agent up and down the alley in Bowling (as this is never seen in the training data), in environments such as Breakout, Freeway and Pong, it would be reasonable to expect some degree of generalization to different policies. Below we show that our simulator displays some degree of generalization to human policies in Breakout and Pong and leave the discussion of Freeway to the Appendix.

In Fig. 6(a), we show some salient frames from a game of Pong played by a human for 500 time-steps (the corresponding video is available at Pong-HPlay). The game starts with score  $(2,0)$ , after which the opponent scores five times, whilst the human player scores twice. As we can see, the scoring is updated correctly and the game dynamic is accurate. In Fig. 6(b), we show some salient fames from a game of Breakout played by a human for 350 time-steps (the corresponding video is available at Breakout-HPlay). These images demonstrate generalization of the model to a human style of play also in this game.

# 3.2 JUMPY SIMULATORS

A jumpy simulator uses the model structure shown in Fig. 1(b) without the dashed links, using the frame-independent mode for training and prediction. The advantage of jumpy simulators is that, to make a prediction for any point in the future, high-dimensional images do not need to be predicted at all intermediate time-steps. This avoids having to go through the encoding and the decoding at those intermediate time-steps – in the fully-connected linear mapping, this enables saving around 60 million flops at each time-step.

To account for the difference in the structure of the core in the warm-up and generation phase, we used two separate sets of parameters. Motivated by the results obtained with the single-step simulator, we considered one or two consecutive subsequences of length 15. Our results suggest that performance is considerably better with two consecutive subsequences.

Overall, we obtained similar performance to the single-step simulator, as shown in Fig. 5(b) for three games (randomly selected videos obtained with the jumpy simulator can be seen at Bowling, FDerby, Freeway, Pong, Qbert, Riverraid, and Seaquest). However, for the games of Breakout, Ms Pacman and Space Invaders, whilst giving reasonable predictions, the jumpy simulator performed worse than the single-step simulator – notice that the best single-step simulator for these three games used a mix of observation-dependent and prediction-dependent training.

![](images/3a1a2f7239094ccd8e450be5ef4576f6b5257f20cd6a289f7caa302faeb3ad70.jpg)  
Figure 7: Salient frames highlighting coherence extracted from 700 frames of TORCS generated using our simulator with actions taken by a human player.

![](images/6d188ae6e850931f39846b1877c0550a8a0335e407b015ffb885869202ac5cc8.jpg)  
Figure 8: Predicted (left) and real (right) frames at time-steps 1, 25, 66, 158 and 200 using actions from the test data.

# 4 SIMULATORS OF 3D ENVIRONMENTS

To demonstrate the robustness of our approach to different environments, we show how our simulators perform on a 3D car racing environment called TORCS (Wymann et al., 2013) and on 3D mazes. Both environments highlight the need to learn dynamics that are spatially and temporally coherent: TORCS exposes the need to learn fast moving dynamics and consistency under motion, whilst 3D mazes are partially-observed and therefore require the simulator to build an internal representation of its surrounding using memory, as well learn basic physics, such as rotation, momentum, and the solid properties of walls. As in Atari, we evaluated the simulators using two different policies, one from the test set, and one from human-play. Specifics of the data and models are given in the Appendix.

TORCS. The data was generated using an artificial agent controlling a fast car without opponents.

When using actions from the test set (see Fig. 16 in the Appendix and the corresponding video at TORCS), the simulator was able to produce accurate predictions for up to several hundreds time-steps. As the car moved around the racing track, the simulator was able to predict the appearance of new features in the background (towers, sitting areas, lamp posts, etc.), as well as model the jerky motion of the car caused by our choices of random actions. Finally, the instruments (speedometer and rpm) were correctly displayed.

The simulator was good enough to be used interactively for several hundred frames, using actions provided by a human. This showed that the model had learnt well how to deal with the car hitting the wall on the right side of the track. Some salient frames from the game are shown in Fig. 7 (the corresponding video can be seen at TORCS-HPlay).

3D Mazes. We used an environment that consists of randomly generated 3D mazes, containing textured surfaces with occasional paintings on the walls: the mazes were all of the same size, but differed in the layout of rooms and corridors, and in the locations of paintings (see Fig. 10(b) for an example of layout).

When using actions from the test set, the simulator was able to very reasonably predict frames even after 200 steps. In Fig. 8 we compare predicted frames to the real frames at several time-steps (the corresponding video can be seen at 3DMazes). We can see that the wall layout is better predicted when walls are closer to the agent and that the depth of corridors, and far away-walls are not as long as they should be. The lighting on the ceiling is correct on all the frames shown.

When using the simulator interactively with actions provided by a human, we could test that the simulator had learnt consistent aspects of the maze: when walking into walls, the model maintained their position and layout (in a rare case, we were able to walk through a painting on the wall - paintings are rare in the data set and hence it is not unreasonable that they would not be maintained when stress testing the model in this way). When taking  $360^{\circ}$  spins, the wall configurations were the same as previously generated and not regenerated afresh, and shown in Fig. 9 (see also 3DMazes-HPLay). The coherence of the maze was good for nearby walls, but not at the end of long-corridors.

![](images/49cafaf6d52a478db9fd633d244e0f841339e21251cde6e7de825923e801dbb4.jpg)

![](images/26cc3d29149537735def04fa9f5ab967f3713a917f90f25182a851174a74444a.jpg)  
Figure 9: Salient frames highlighting wall-Layout memory after  $360^{\circ}$  spin generated using our simulator with actions taken by a human player.

![](images/4f3c535bb77ad57666820280b8a19d1c9df8a2172ad5a2f81354e90306c94436.jpg)

![](images/42dcd62b1a905d99c2d6f974422dfe5a9bf89071c4460b25d49512862feed0f7.jpg)

![](images/9da842273bbe2dd1e6a950719c01c13f4a68614f99623d2fe49c6399d354236e.jpg)

![](images/7aeb68525b50e01df7520477a429a613f00d569d8e69cdd1660fcd71e5b5c4e9.jpg)

![](images/2bf38498999152c895f9a1fa95781a3bcf660f17f67dd19990c82fddfed1ebb3.jpg)

![](images/cf22bf2b2dddf75442a430d7bd04e0cd0212c0926a42664b7f787c89bccdd138.jpg)

![](images/9a413f93ecead58b77e6b85bf36d923c62107c3a915ba7e9d94ec11e94a7c6c3.jpg)

![](images/4c936a2bd87ed0d4f9ae0490003fc698ee1bdaa827976b848c56ebd8439c0d7e.jpg)

![](images/9281af6fb4d42799e7f79b470a1880bc81f4024c9729298782468ab4c40c2c9f.jpg)

![](images/58a9bb638e93190c3d907ea3ba0f34d64b07f5eef29efb298a623f7b9d499caf.jpg)

![](images/ce9be8e184ea42d8f2842a91f711343bd2ac6f053a170a59e55a8962d04650c4.jpg)

![](images/08ff32953ae743dd6b42997885fc9e9d06463a8702bddd96358184a0320046d9.jpg)

![](images/cb1ed1ce2d4558c3cb42385f9412e4c90b350d491f16860e59823b013efa7cdb.jpg)

![](images/d4b189cbfeb36e5a452005f684de8160866c168b7d62e0a4dd36c91ecac32303.jpg)

![](images/829e231b8b94a9e4498e9f9d5d5fd889e695deb4776c1d1b342f47ee9d056024.jpg)

![](images/1d3424bda2dca28e5b832be5de748c65c9f244bf68f81cdcf2302d0466137e39.jpg)

![](images/0960c9fd6967b41fae4c3605aaca56e2b334e4aafb04885b59c9fbc6edf544d7.jpg)  
(a)

![](images/66b83658d35648535878a288fdaae905bfd9d1d3fea91403197e23a778200488.jpg)  
(b)

![](images/5d8faaa6ce35ab795c55fb4aec9f33e99bb549c6b2306a1458c1471e670d5d43.jpg)  
Figure 10: (a) Average ratio over 10 mazes (shaded is the  $68\%$  confidence interval) of area visited by the random agent and an agent using our model. (b) Typical example of paths followed by (left) the random agent and (right) our agent (see the Appendix for more examples).

# 5 MODEL-BASED EXPLORATION

The search for exploration strategies better than  $\epsilon$ -greedy is an active area of research. Various solutions have been proposed, such as density based or optimistic exploration (Auer et al., 2002). We considered a memory-based approach, where we steer the agent towards previously unobserved frames. Our aim was to be quantitatively and qualitatively better than random exploration (using dithering of 0.7, as this leads to the best possible random agent).

We used a 3D maze simulator to predict the outcome of sequences of actions, chosen with a hardcoded policy. Our algorithm (see below) did  $N$  Monte Carlo simulations with randomly selected sequences of actions of fixed length  $d$  and dithering of 0.7. At each time-step  $t$ , we stored the last 10 observed frames in an episodic memory buffer and compared predicted frames to those in memory.

for  $t = 1$  episodeLength,  $d$  do

for  $n = 1, N$  do  
Choose random actions  $A^n = a_{t:t + d - 1}$  
Predict  $\hat{x}_{t + 1:t + d}^n$

end

Follow actions in  $A^{n_0}$  where

$n_0 = \operatorname{argmax}_n\min_{j = 0,10}||\hat{x}_{t + d}^n -x_{t - j}||_2$

end

Our best results (see Fig. 10(a)) showed that our method covered  $50\%$  more of the maze area after 900 time-steps than random exploration. These results were obtained with 100 Monte-Carlo simulations, sequences of 6 actions. More details are given in the Appendix. Comparing typical paths (see Fig. 10(b)) chosen by the random explorer and by our explorer, we see the our explorer has much smoother trajectories.

This is a good local exploration strategy that leads to faster movement through corridors. To transform this into a good global exploration strategy, our explorer would have to be augmented with a better memory in order to avoid going down the same corridor twice. These sorts of smooth local exploration strategies could also be useful in navigation problems.

# 6 DISCUSSION

In this paper we have introduced an approach to simulate action-conditional dynamics and demonstrated that is highly adaptable to different environments, ranging from Atari games to 3D car racing environments and mazes. We showed state-of-the-art results on Atari, and demonstrated the feasibility of live human play in all three task families. The system is able to capture complex and long-term

interactions, and displays a sense of spatial and temporal coherence that has, to our knowledge, not been demonstrated on high-dimensional time-series data such as these.

Complex environments have compositional structure, such as independently moving objects and other phenomena that only rarely interact. In order for our simulators to better capture this compositional structure, we may need to develop specialised functional forms and memory stores that are better suited to dealing with independent representations and their interlinked interactions and relationships. More homogeneous deep network architectures such as the one presented here are clearly not optimal for these domains, as can be seen in Atari environments such as Ms Pacman where the system has trouble keeping track of multiple independently moving ghosts. Significant progress can likely be made researching this direction. Whilst the LSTM memory and our training scheme have proven to capture long-term dependencies, alternative memory structures are required in order, for example, to learn spatial coherence at a more global level than the one displayed by our model in the 3D mazes in order to do navigation.

In the case of action-conditional dynamics, the policy-induced data distribution does not cover the state space and might in fact be nonstationary over an agent's lifetime. This can cause some regions of the state space to be oversampled, whereas the regions we might actually care about the most - those just around the agent policy's state distribution - to be underrepresented. In addition, this induces biases in the data that will ultimately not enable the model learn the environment dynamics correctly. As verified from the experiments in this paper, both on live human play and model-based exploration, this problem is not yet as pressing as might be expected in some environments. However, our simulators displayed limitations and faults due to the specificities of the training data, such as for example predicting agent's death based on the recognition of a particular sequence of actions always co-occurring with death in the training data rather than on the recognition of the real causes.

Finally, one of the present limitations of our approach is that, however capable it might be, it is a deterministic model designed for deterministic environments. Clearly most real world environments involve noisy state transitions, and future work will have to address the extension of the techniques developed in this paper to more generative temporal models.

# ACKNOWLEDGMENTS

The authors would like to thank David Barber for helping with the graphical model interpretation, Alex Pritzel for preparing the DQN data, Yori Zwols and Frederic Besse for helping with the implementation of the model, and Yee Whye Teh for providing feedback on the manuscript.

# REFERENCES

P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47:235-256, 2002.  
M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. The Arcade Learning Environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 47:253-279, 2013.  
S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In Advances in Neural Information Processing Systems 28 (NIPS), pp. 1171-1179. 2015.  
K. Fragkiadaki, P. Agrawal, S. Levine, and J. Malik. Learning visual predictive models of physics for playing billiards. CoRR, abs/1511.07404, 2015.  
A. Graves. Generating sequences with recurrent neural networks. 2013. URL http://arxiv.org/abs/1308.0850.  
S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735-1780, 1997.  
M. Lengyel and P. Dayan. Hippocampal contributions to control: The third way. In Advances in Neural Information Processing Systems 20 (NIPS), pp. 889-896, 2008.  
M. L. Littman, R. S. Sutton, and S. Singh. Predictive representations of state. In Advances in Neural Information Processing Systems 14 (NIPS), pp. 1555-1561. 2002.  
M. McCloskey. Intuitive physics. Scientific American, 248(4):122-130, 1983.  
V. Mnih, A. Puigdomenech Badia, M. Mirza, A. Graves, T. P Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016.  
Y. Niv. Reinforcement learning in the brain. Journal of Mathematical Psychology, 53(3):139-154, 2009.  
J. Oh, X. Guo, H. Lee, R. L. Lewis, and S. P. Singh. Action-conditional video prediction using deep networks in Atari games. In Advances in Neural Information Processing Systems 28 (NIPS), pp. 2863-2871. 2015. URL http://papers.nips.cc/paper/

5859-action-conditional-video-prediction-using-deep-networks-in-atari-games. pdf.  
J. K. O'Regan and A. Noë. A sensorimotor account of vision and visual consciousness. Behavioral and brain sciences, 24(05):939-973, 2001.  
P.-Y. Oudeyer, F. Kaplan, and V. V. Hafner. Intrinsic motivation systems for autonomous mental development. Evolutionary Computation, IEEE Transactions on, 11(2):265-286, 2007.  
V. Patraucean, A., and R. Cipolla. Spatio-temporal video autoencoder with differentiable memory. volume abs/1511.06309, 2015.  
J. Pearl. Causality. Cambridge University Press, 2009.  
Wen Sun, Arun Venkatraman, Byron Boots, and J Andrew Bagnell. Learning to filter with predictive state inference machines. arXiv preprint arXiv:1512.08836, 2015.  
R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction. MIT Press, 1998.  
E. Talvitie. Model regularization for stable sample rollouts. In Proceedings of the Thirtieth Conference Annual Conference on Uncertainty in Artificial Intelligence (UAI-14), pp. 780-789, 2014.  
N. Wahlström, T. B. Schon, and M. P. Deisenroth. From pixels to torques: Policy learning with deep dynamical models. arXiv preprint arXiv:1502.02251, 2015.  
M. Watter, J. Springenberg, J. Boedecker, and M. Riedmiller. Embed to control: A locally linear latent dynamics model for control from raw images. In Advances in Neural Information Processing Systems 28 (NIPS), pp. 2728-2736, 2015.  
B. Wymann, E. Espie, C. Guionneau, C. Dimitrakakis, R. Coulom, and A. Sumner. Torcs: The open racing car simulator, v1.3.5. 2013. URL http://www.torcs.org.  
B. Xu, N. Wang, T. Chen, and M. Li. Empirical evaluation of rectified activations in convolutional network. 2015.  
W. Zaremba, I. Sutskever, and O. Vinyals. Recurrent neural network regularization. 2014. URL http://arxiv.org/abs/1409.2329.
