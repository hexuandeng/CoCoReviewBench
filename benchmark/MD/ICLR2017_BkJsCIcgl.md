# THE PREDICTRON: END-TO-END LEARNING AND PLANNING

David Silver*, Hado van Hasselt*, Matteo Hessel*, Tom Schaul*, Arthur Guez*, Tim Harley, Gabriel Dulac-Arnold, David Reichert, Neil Rabinowitz, Andre Barreto, Thomas Degris DeepMind, London

{silver,hado,mtthss,schaul,aguez}@google.com

# ABSTRACT

One of the key challenges of artificial intelligence is to learn models that are effective in the context of planning. In this document we introduce the predictron architecture. The predictron consists of an abstract model, formulated as a Markov reward process, that can be rolled forward multiple steps. At every step it outputs predictions about the future, formulated as value functions. The predictron is trained end-to-end directly from experience to make these value functions accurate, thereby focusing the model upon the aspects of the environment most relevant to planning. During training, we can exploit the Markov semantics of our model to make the value functions consistent both with the real environment and with each other. We applied our architecture to procedurally generated random mazes and a simulator for the game of pool. The predictron yielded significantly more accurate predictions than conventional deep neural network architectures.

# 1 INTRODUCTION

The central idea of model-based reinforcement learning (MBRL) is to decompose the RL problem into two subproblems: learning a model of the environment, and then planning with this model. The model is typically represented by a Markov reward process (MRP) or decision process (MDP). The planning component uses this model to evaluate and select among possible strategies. This is typically achieved by rolling forward the model to construct a value function that estimates cumulative reward. In prior work, the model is trained essentially independently of its use within the planner. As a result, the model is not well-matched with the overall objective of the agent. Prior deep reinforcement learning methods have successfully constructed models that can unroll near pixel-perfect reconstructions (Oh et al., 2015); but are yet to surpass state-of-the-art model-free methods (e.g., Mnih et al., 2015; 2016; Lillicrap et al., 2016) in challenging RL domains with raw inputs.

In this paper we introduce a new architecture, which we call the predictron, that integrates learning and planning into one end-to-end training procedure. At every step, a model is applied to an internal state, to produce a next state, reward, discount, and value estimate. This model is completely abstract and focuses only on those aspects that facilitate accurate value prediction. For example, to plan effectively in a game, an agent must understand how a planned trajectory will affect the score. If our model accurately predicts the score, then an optimal plan with respect to our model will also be an optimal plan for the underlying game – even if that model uses a different state space (e.g., an abstract representation of enemy positions, ignoring their shapes and colours), action space (e.g., a high-level action to move away from an enemy), rewards (e.g., a single abstract step could have a higher value than any real reward), or even time-step (e.g., a single abstract step could “jump” the agent to the end of a corridor). All we require is that trajectories through the abstract model produce scores that are consistent with trajectories through the real environment. This is achieved by training the predictron end-to-end, so as to make its value estimates as accurate as possible.

An ideal model could generalise to many different prediction tasks, rather than overfitting to a single task; and could learn from a rich variety of feedback signals, not just a single extrinsic reward. We therefore train the predictor to predict a host of different value functions for a variety of pseudo-reward functions and discount factors. These pseudo-rewards can encode any event or aspect of the environment that the agent may care about, e.g., staying alive or reaching the next room.

We focus upon the prediction task: estimating value functions in MRP environments with uncontrolled dynamics. In this case, the predictron can be implemented as a deep neural network with an MRP as a recurrent core. We also impose the semantics of MRPs. This is achieved by training the model such that the values computed at each step are consistent with the abstract MRP—that is, they satisfy the Bellman equations induced by the predictron's model. In other words, the values should be both consistent with each other, and also consistent with observed rewards. The predictron is optimised end-to-end so as to jointly learn a state representation, MRP model, and value function that best satisfies these consistency requirements.

We applied the predictron to procedurally generated random mazes, and a simulated pool domain, directly from pixel inputs. In both cases, the predictron significantly outperformed model-free algorithms with conventional deep network architectures; and was much more robust to architectural choices such as depth.

# 2 BACKGROUND

We consider environments defined by an MRP with states  $s \in S$ , e.g., a joint configuration of a robot, or a history of raw input sensors. The MRP is defined by a function,  $s', r, \gamma = p(s, \alpha)$ , where  $s'$  is the next state,  $r$  is the reward, and  $\gamma$  is the discount factor, which can for instance represent the non-termination probability for this transition. The process may be stochastic, given IID noise  $\alpha$ .

The return of an MRP is the cumulative discounted reward over a single trajectory,  $g_{t} = r_{t + 1} + \gamma_{t + 1}r_{t + 2} + \gamma_{t + 1}\gamma_{t + 2}r_{t + 3} + \ldots$ , where  $\gamma_{t}$  is a discount factor that can vary per time-step (i.e., the modeled probability of termination may differ per transition). We consider a generalisation of the MRP setting that includes vector-valued rewards  $\mathbf{r}$ , diagonal-matrix discounts  $\gamma$ , and vector-valued returns  $\mathbf{g}$ ; definitions are otherwise identical to the above. We use this bold font notation to closely match the more familiar scalar MRP case; the majority of the paper can be comfortably understood by reading all rewards as scalars, and all discount factors as scalar and constant, i.e.,  $\gamma_{t} = \gamma$ .

The value function of an MRP  $p$  is the expected return from state  $s$ ,  $v_{p}(s) = \mathbb{E}_{p}[\mathbf{g}_{t} \mid s_{t} = s]$ . In the vector case, these are known as general value functions (Sutton et al., 2011). We will say that a (general) value function  $v(\cdot)$  is consistent with environment  $p$  if and only if  $v = v_{p}$  which satisfies the following Bellman equation (Bellman, 1957),

$$
v _ {p} (s) = \mathbb {E} _ {p} \left[ \mathbf {r} + \boldsymbol {\gamma} v _ {p} (s ^ {\prime}) \right]. \tag {1}
$$

In model-based reinforcement learning (Sutton and Barto, 1998), an approximation  $m \approx p$  to the environment is learned. In the uncontrolled setting this model is normally an MRP  $s'$ ,  $\mathbf{r}$ ,  $\pmb{\gamma} = m(s, \beta)$  that maps from state  $s$  to subsequent state  $s'$  and additionally outputs rewards  $\mathbf{r}$  and discounts  $\pmb{\gamma}$ ; the model may be stochastic given an IID source of noise  $\beta$ . A (general) value function  $v(\cdot)$  is consistent with model  $m$  (or valid, (Sutton, 1995)), if and only if  $v = v_m$  where  $v_m$  satisfies a Bellman equation  $v_m(s) = \mathbb{E}_m[\mathbf{r} + \pmb{\gamma}v_m(s')]$  with respect to model  $m$ . Conventionally, MBRL methods focus on finding a value function  $v$  that is consistent with a separately learned model  $m$ .

# 3 PREDICTRON ARCHITECTURE

The predictron is composed of three main components. First, a state representation  $\mathbf{s} = f(s)$  that encodes raw observations  $s$  (or history of observations, in the partially observed setting, for example when  $f$  is a recurrent network) into an abstract (internal, hidden) state  $\mathbf{s}$ . Second, a model  $\mathbf{s}'$ ,  $\mathbf{r}$ ,  $\boldsymbol{\gamma} = m(\mathbf{s}, \boldsymbol{\beta})$  that maps from abstract state  $\mathbf{s}$  to subsequent abstract state  $\mathbf{s}'$ , rewards  $\mathbf{r}$ , and discounts  $\boldsymbol{\gamma}$ . Third, a value function  $v$  that outputs an estimate  $\mathbf{v} = v(\mathbf{s})$  of future cumulative discounted rewards from internal state  $\mathbf{s}$  onwards. The predictron is applied by unrolling its model  $m$  multiple "planning" steps to produce rewards, discounts and values (see Figure 1a).

The components of the predictron can be composed together to form many different predictions of the real returns. We use superscripts  $\bullet^k$  to indicate internal steps of the model (which have no necessary connection to time steps  $\bullet_t$  of the environment). We accumulate discount factors  $\bar{\gamma}^k$  multiplicatively along the pathway,  $\bar{\gamma}^k = \prod_{j=1}^k \gamma^j$ . The predictron return  $\mathbf{g}^k$  (henceforth abbreviated as  $preturn$ ) is the discounted cumulative reward obtained by taking  $k$  model steps, plus an estimated

![](images/0414d4dba09eaf9097fbcc6aa10b0e48d271897025d0dc79da2d79c3e0b88073.jpg)  
(a)

![](images/f17dc37c49a26d25008d78a7723e61d1e9a8b83dae858e03ac636e8f06a2c4f0.jpg)  
(b)

![](images/563b51c3cb6bbd991447ac008575c7fe3504860bcf5d199f0171735d8bfeedf8.jpg)  
Figure 1: (a) Architecture for the predictron, consisting of state representation  $f$ , MRP model  $m$ , and value function  $v$ . (b) Returns  $\mathbf{g}^k$  accumulate the discounted return along a  $k$ -step pathway through the predictron. (c) A consistent predictron satisfies a sequence of Bellman equations with respect to its internal model  $m$ . (d) The  $\lambda$ -predictron aggregates returns. Solid lines represent the predictron network for computing returns, with parameters  $\theta$ ; dotted lines represent the  $\lambda$ -network, with parameters  $\eta$ , for combining returns; see Equations (5) and (6) for precise details.

![](images/c4bf5d6b5294cf804ec5f00d264f16a6d366f99dc8cac753db3aa9845697affa.jpg)  
(c)

![](images/c674bf45d023608e15d45ad162fbe61e1fae3e485ab38b810e67f4031d75d3dd.jpg)  
(d)

value of future cumulative reward,  $\mathbf{v}^k$ , from the final step of the pathway (see Figure 1b),

$$
\mathbf {g} ^ {k} = \sum_ {j = 1} ^ {k} \bar {\gamma} ^ {j - 1} \mathbf {r} ^ {j} + \bar {\gamma} ^ {k} \mathbf {v} ^ {k}. \tag {2}
$$

The main idea of the predictor is that each return  $\mathbf{g}^k$  should accurately estimate the true value of state  $s$ . We say that a predictor  $(f, m, v)$  is consistent if all returns are equal in expectation to the true value function of the environment,  $\mathbb{E}^m[\mathbf{g}^k | s] = \mathbb{E}_p[\mathbf{g}_t | s_t = s] = v_p(s)$ . It follows that any two returns  $\mathbf{g}^{k_1}$  and  $\mathbf{g}^{k_2}$  must also be equal in expectation,  $\mathbb{E}^m[\mathbf{g}^{k_1} | s] = \mathbb{E}^m[\mathbf{g}^{k_2} | s]$ . This is equivalent to unrolling the Bellman equation on the model  $m$  by several steps (see Figure 1c).

# 4 PREDICTRON LEARNING UPDATES

We now consider how to jointly optimise the parameters  $\theta$  of all components  $f, m, v$  of the predictor. First, we will discuss how to learn from Monte Carlo returns from the real environment. We then discuss how to form a single aggregated estimate of value, consistent with the whole sequence of Bellman equations. Then, we discuss internal consistency updates that can be applied even in the absence of real data.

# 4.1 SUPERVISED (MONTE-CARLO) LEARNING WITH THE PREDICTRON

We first consider how to train the predictron using supervised learning from the outcomes of episodes in the environment. The predictron can be viewed as a (stochastic) value function approximator that produces an ensemble of (sample) values  $\mathbf{g}^0, \dots, \mathbf{g}^K$ . We can update all of these values towards a target outcome  $\mathbf{g}$ , such as the Monte Carlo return  $\mathbf{g}_t$ , by minimising a mean-squared error loss,

$$
L = \sum_ {k = 0} ^ {K} \left\| \mathbb {E} _ {p} [ \mathbf {g} ] - \mathbb {E} ^ {m} [ \mathbf {g} ^ {k} ] \right\| ^ {2}. \tag {3}
$$

This loss depends on the parameters of the value function, model, and state representation parameters, which we together denote  $\theta$ , and we can use the gradient of  $L$  to update these, e.g., by stochastic gradient descent on the sample loss  $l = \sum_{k=0}^{K} \left\| \mathbf{g} - \mathbf{g}^k \right\|^2$ ,

$$
\frac {\partial l}{\partial \boldsymbol {\theta}} = \sum_ {k = 0} ^ {K} \left(\mathbf {g} - \mathbf {g} ^ {k}\right) \frac {\partial \mathbf {g} ^ {k}}{\partial \boldsymbol {\theta}}. \tag {4}
$$

For stochastic models, two independent samples are required for  $\mathbf{g}^k$  and  $\frac{\partial\mathbf{g}^k}{\partial\theta}$  to get unbiased samples for the gradient of loss (3).

# 4.2 AGGREGATING VALUES USING THE  $\lambda$ -PREDICTRON

We now introduce a  $\lambda$ -predictron that adaptively combines a sequence of value estimates into an aggregate prediction. Specifically, we augment the predictron architecture with  $\lambda$ -parameters that aggregate over all preturns,  $\mathbf{g}^0, \dots, \mathbf{g}^K$ , using diagonal weight matrices defined by  $\lambda^0, \dots, \lambda^K$ , and output an ensemble  $\lambda$ -predict  $\mathbf{g}^\lambda$ ,

$$
\mathbf {g} ^ {\lambda} = \sum_ {k = 0} ^ {K} \bar {\lambda} ^ {k} \mathbf {g} ^ {k} \quad \text {w h e r e} \quad \bar {\lambda} ^ {K} = \prod_ {j = 0} ^ {K - 1} \lambda^ {j} \quad \text {a n d} \quad \bar {\lambda} ^ {k} = \left(\mathbf {1} - \lambda^ {k}\right) \prod_ {j = 0} ^ {k - 1} \lambda^ {j}, \text {f o r} k <   K. \tag {5}
$$

This  $\lambda$ -preturn is analogous to the  $\lambda$ -return in the forward-view TD( $\lambda$ ) algorithm (Sutton, 1988; Sutton and Barto, 1998). It may also be computed by a backward recursion through intermediate steps  $\mathbf{g}^{k,\lambda}$ ,

$$
\mathbf {g} ^ {k, \lambda} = \left(\mathbf {1} - \boldsymbol {\lambda} ^ {k}\right) \mathbf {v} ^ {k} + \boldsymbol {\lambda} ^ {k} \left(\mathbf {r} ^ {k + 1} + \boldsymbol {\gamma} ^ {k + 1} \mathbf {g} ^ {k + 1, \lambda}\right), \tag {6}
$$

where  $\mathbf{g}^{K,\lambda} = \mathbf{v}^K$  and  $\mathbf{g}^{\lambda} = \mathbf{g}^{0,\lambda}$ . Computation in the  $\lambda$ -predictron operates in a sweep, iterating first through the predictron network from  $k = 0..K$  and then back through the  $\lambda$ -network from  $k = K..0$  in a single "forward" pass of the network (see Figure 1d). Each  $\lambda^k$  weight acts as a gate on the computation of the  $\lambda$ -preturn: a value of  $\lambda^k = 0$  will truncate the  $\lambda$ -preturn at layer  $k$ , while a value of  $\lambda^k = 1$  will utilise deeper layers based on additional steps of the model  $m$ ; the final weight is always  $\lambda^K = 0$ . The individual  $\lambda^k$  weights may depend on the corresponding abstract state  $\mathbf{s}^k$  and can differ per prediction. This enables the predictron to compute to an adaptive depth (Graves, 2016) depending on the internal state and learning dynamics of the network.

The  $\lambda^k$  weights of the  $\lambda$ -preturn are adjusted by modifying the parameters  $\eta$  of the  $\lambda$ -network (dotted lines in Figure 1d) so as to minimise a Monte-Carlo loss,

$$
L = \left\| \mathbb {E} _ {p} [ \mathbf {g} ] - \mathbb {E} _ {m} [ \mathbf {g} ^ {\lambda} ] \right\| ^ {2}, \quad \frac {\partial l}{\partial \eta} = (\mathbf {g} - \mathbf {g} ^ {\lambda}) \frac {\partial \mathbf {g} ^ {\lambda}}{\partial \boldsymbol {\lambda}}. \tag {7}
$$

# 4.3 CONSISTENCY (SEMI-SUPERVISED) LEARNING WITH THE PREDICTRON

A model may be used to generate and learn from hypothetical experience. For example, the Dyna algorithm (Sutton, 1990) applies temporal-difference updates (Sutton, 1988) to transitions sampled from the model. These updates adjust the value function to be consistent with a model but do not help make the model more accurate, nor the representation of state upon which the model operates.

In the prediction, we may also use hypothetical experience, but to jointly optimise the state representation  $f$ , value function  $v$ , and model  $m$  to be consistent. We consider an update that adjusts each return  $\mathbf{g}^k$  towards the  $\lambda$ -prereturn  $\mathbf{g}^\lambda$ ; in other words, we update each individual value estimate towards the best aggregated estimate by minimizing

$$
L = \sum_ {k = 0} ^ {K} \left\| \mathbb {E} _ {m} \left[ \mathbf {g} ^ {\lambda} \right] - \mathbb {E} _ {m} \left[ \mathbf {g} ^ {k} \right] \right\| ^ {2}, \quad \frac {\partial l}{\partial \boldsymbol {\theta}} = \sum_ {k = 0} ^ {K} \left(\mathbf {g} ^ {\lambda} - \mathbf {g} ^ {k}\right) \frac {\partial \mathbf {g} ^ {k}}{\partial \boldsymbol {\theta}}. \tag {8}
$$

Here  $\mathbf{g}^{\lambda}$  is considered fixed; the parameters  $\theta$  are only updated to make  $\mathbf{g}^k$  more similar to  $\mathbf{g}^{\lambda}$ , not vice versa. This consistency update does not require any labels  $\mathbf{g}$  or samples from the environment. As a result, it can be applied to (potentially hypothetical) states that have no associated 'real' (e.g. Monte-Carlo) outcome: we update the value estimates to be self-consistent with each other. Note the similarity with the semi-supervised setting, where we may have unlabelled inputs.

# 5 EXPERIMENTS

We conducted experiments on two domains. The first domain consists of randomly generated  $20 \times 20$  mazes in which each location either is empty or contains a wall. Two locations in a maze are considered connected if they are both empty and we can reach one from the other by moving horizontally or vertically through adjacent empty cells. The goal is to predict for each of the locations on the diagonal from top-left to bottom-right of the maze whether that location is connected to the bottom-right

![](images/65030908e6537df6b23c228a3222c884aca487f133dad2ff59cd306c8065bf9d.jpg)  
Figure 2: Left: Two sample mazes from the random-maze domain. Light blue cells are empty, darker blue cells contain a wall. One maze is connected from top-left to bottom-right (indicated in black), the other is not. Right: An example trajectory in the pool domain (before downsampling). It was selected by maximising the prediction of pocketing balls, using the predictron.

![](images/b684a923d786ae64903976186e26379c5ccb567e546b14d786da95d823aea7a7.jpg)

![](images/eb4190164cf17dc5c4b9f94c0aa93c002b1d27028e7f325ae600364595ddd3ba.jpg)

![](images/f36f00a90a689dd0f21e56991e5c8c5c171d37ccf2b7c634df65c8865b7be92a.jpg)

![](images/b58d1847889ac041efbbb26e3a3bb5c66d07b454f3b02f00cba4a0c2ffaa5d2c.jpg)

![](images/9252e7ca0cb4ac75a28bf2c99c5fbf44f0f7d89511172bba919346eaf091ec6c.jpg)

corner, given the entire maze as an input image. Some of these predictions will be straightforward, for instance for locations on the diagonal that contain a wall themselves and for locations close to the bottom right. Many other predictive questions seem to require a simple algorithm, such as some form of a flood fill or search; our hypothesis is that an internal model can learn to emulate such algorithms, where naive approximation may struggle. A few example mazes are shown in Figure 2.

Our second domain is a simulation of the game of pool, using four balls and four pockets. The simulator is implemented in the physics engine Mujoco (Todorov et al., 2012). We generate sequences of RGB frames starting from a random arrangement of balls on the table. The goal is to simultaneously learn to predict future events for each of the four balls, given the first 5 RGB frames as input. These events include: collision with any other ball, collision with any boundary of the table, entering a quadrant ( $\times 4$ , for each quadrant), being located in a quadrant ( $\times 4$ , for each quadrant), and entering a pocket ( $\times 4$ , for each pocket). Each of these  $14 \times 4$  events provides a binary pseudo-reward that we combine with 5 different discount factors  $\{0, 0.5, 0.9, 0.98, 1\}$  and predict their cumulative discounted sum over various time spans. This yields a total of 280 general value functions. An example trajectory is shown in Figure 2. Additional domain details are provided in Appendix D.

# 5.1 EXPLORING THE PREDICTRON ARCHITECTURE

Our first set of experiments examines three binary dimensions that differentiate the predictron from standard deep networks. We compare eight predictron variants corresponding to the corners of the cube on the left in Figure 3; the origin of the cube represents a recurrent neural network. All variants utilise a convolutional core with 2 intermediate hidden layers (see Appendix A).

The first dimension corresponds to whether or not the prediction architecture utilises the structure of an MRP model. In the MRP case, labelled  $r$ ,  $\gamma$ , internal rewards and discounts are both learned. In the non- $r$ ,  $\gamma$  case, which corresponds to a vanilla hidden-to-hidden neural network module, internal rewards and discounts are ignored by fixing their values to  $\mathbf{r}^k = \mathbf{0}$  and  $\gamma^k = \mathbf{1}$ .

The second dimension is whether or not a  $\lambda$ -network is used to aggregate over returns. When a  $\lambda$ -network is used, a  $\lambda$ -preturn is computed as described in Section 4.2. Otherwise, intermediate returns are ignored by fixing their values to  $\lambda^k = 1$  for  $k < K$ . In this case, the overall output of the predictor is simply the maximum-depth preturn  $\mathbf{g}^K$ .

The third dimension, labelled usage weighting, defines the loss that is used to update the parameters. This loss is combined over the returns  $\mathbf{g}^k$  at each depth  $k$ . These were previously assumed to be uniformly weighted. But instead they can be weighted according to the amount that return is actually used in the  $\lambda$ -predictron's overall output, i.e., its weight  $\bar{\lambda}^k$ . For architectures without a  $\lambda$ -predictron network,  $\bar{\lambda}^k = 0$  for  $k < K$ , and  $\bar{\lambda}^K = 1$ . Then usage weighting means that we only backpropagate the final loss.

In each case, parameters were updated by supervised learning (see Appendix B for more details). Root mean squared prediction errors for each architecture, aggregated over all predictions, are shown in Figure 3. The top row corresponds to the random mazes and the bottom row to the pool domain. The main conclusion is that learning a MRP model improved performance greatly. The inclusion of  $\lambda$  weights helped as well, especially on pool. Usage weighting further improved performance.

# 5.2 COMPARING THE PREDICTRON TO OTHER DEEP NETWORKS

Our second set of experiments compares the predictron to feedforward and recurrent deep learning architectures, with and without skip connections. We compare the corners of a new cube, as depicted on the left in Figure 4, based on three different binary dimensions.

![](images/46b0ee8f09fc664da01e446a8f73dc1cad315ca0078ef50e521ec6d70d387a05.jpg)

![](images/7d404522f0579eb308cf2d2c54d16b34a1c1e4b56c29770fee1be464e72d2895.jpg)

![](images/98f2bab768000eaf0429122d0079c15aa297fae09696047dbeee27780a74bfb6.jpg)  
Figure 3: Exploring predictron variants. Aggregated prediction errors over all predictions (20 for mazes, 280 for pool) for the eight predictron variants corresponding to the cube on the left (as described in the main text), for both random mazes (top) and pool (bottom). Each line is the median of RMSE over five seeds; shaded regions encompass all seeds. The full  $(r,\gamma,\lambda)$ -prediction (red) consistently performed best.

![](images/42dcbf598b6e7e80cc8f5d20e14f42d1f531c821d8a05d0c30030ecb61ccf96b.jpg)

![](images/7502d820f97722ad09d7d057103e76e1300361ff27bb2b14a0a82ecca255e505.jpg)

![](images/cdee9a3e4d2d19f9af872660f04fd8172756e8e435f8dd7d853feadf463d2a78.jpg)

![](images/a45761d553fdec9eb97c52a6817f6866baf09488fb15d33a671771fa64af20b3.jpg)  
Figure 4: Comparing predictron to baselines. Aggregated prediction errors on random mazes (top) and pool (bottom) over all predictions for the eight architectures corresponding to the cube on the left. Each line is the median of RMSE over five seeds; shaded regions encompass all seeds. The full  $(r, \gamma, \lambda)$ -predictron (red), consistently outperformed conventional deep network architectures (black), with and without skips or and with and without weight sharing.

![](images/44898c88887c9cf2ce650c2f76a69d2dce046e22cc152c019dfdf33257117ef4.jpg)

The first dimension of this second cube is whether we use a predictron, or a (non- $\lambda$ , non- $r$ ,  $\gamma$ ) deep network that does not have an internal model and does not output or learn from intermediate predictions. We use the most effective predictron from the previous section, i.e., the  $(r,\gamma,\lambda)$ -predictron with usage weighting.

The second dimension is whether weights are shared between all cores (as in a recurrent network), or whether each core uses separate weights (as in a feedforward network). We note that the non-MRP, non- $\lambda$  variants of the predictron then correspond to standard (convolutional) feedforward and (unrolled) recurrent neural networks respectively.

The third dimension is whether we include skip connections. This is equivalent to defining the model step to output a change to the current state,  $\Delta \mathbf{s}$ , and then defining  $\mathbf{s}^{k + 1} = h(\mathbf{s}^k +\Delta \mathbf{s}^k)$ , where  $h$  is the non-linear function—in our case a ReLU,  $h(x) = \max (0,x)$ . The deep network with skip connections is a variant of ResNet (He et al., 2015).

Root mean squared prediction errors for each architecture are shown in Figure 4. All  $(r,\gamma ,\lambda)$ -predICTRs (red lines) outperformed the corresponding feedforward or recurrent neural network baselines (black lines) both in the random mazes and in pool. We also investigated the effect of

changing the depth of the networks (see Appendix C). The predictron outperformed the corresponding feedforward or recurrent baselines for all depths, with and without skip connections.

# 5.3 SEMI-SUPERVISED LEARNING BY CONSISTENCY

We now consider how to use the predictron for semi-supervised learning, training the model on a combination of labelled and unlabelled random mazes. Semi-supervised learning is important because a common bottleneck in applying machine learning in the real world is the difficulty of collecting labelled data, whereas often large quantities of unlabelled data exist.

We trained a full  $(r,\gamma ,\lambda)$ -predictron by alternating standard supervised updates with consistency updates, obtained by stochastically minimizing the consistency loss (8), on the unlabelled samples. For each supervised update we apply either 0, 1, or 9 consistency updates. Figure 5 shows that the performance improved monotonically with the number of consistency updates, measured as a function of the number of labelled samples consumed.

![](images/e3b1955989d5a409e647693248d1f3eb59d3b7285173f4a0705af1650a290d57.jpg)  
Figure 5: Semi-supervised learning. Prediction errors of the  $(r, \gamma, \lambda)$ -predICTRs (shared core, no skips) using 0, 1, or 9 consistency updates for every update with labelled data, plotted as function of the amount of labelled data consumed. Early learning performance improved with more consistency updates.

![](images/0e5a3c7fda40951fad8c51a0dde9e5a03171b7ccaf0f4cafca948f2bf4638922.jpg)

# 5.4 ANALYSIS OF ADAPTIVE DEPTH

In principle, the predictron can adapt its depth to 'think more' about some predictions than others, perhaps depending on the complexity of the underlying target. We investigate this by looking at qualitatively different prediction types in pool: ball collisions, rail collisions, pocketing balls, and entering or staying in quadrants. For each prediction type we consider several different time-spans (determined by the real-world discount factors associated with each pseudo-reward). Figure 6 shows distributions of depth for each type of prediction. The 'depth' of a predictron is here defined as the effective number of model steps. If the predictron relies fully on the very first value, this counts as 0 steps. If, instead, it learns to place equal weight on all rewards and on the final value, this counts as 16 steps. Concretely, the depth  $\pmb{d}$  can be defined recursively as  $\pmb{d}_t = \pmb{d}_t^0$  where  $\pmb{d}_t^k = \pmb{\lambda}_t^k (1 + \pmb{\gamma}_t^k \pmb{d}_t^{k+1})$  and  $\pmb{d}_t^K = \mathbf{0}$ . Note that for a single input state  $s_t$  each prediction can have a separate depth.

The depth distributions exhibit three properties. First, different types of predictions used different depths. Second, depth was correlated with the real-world discount for the first four prediction types. Third, the distributions are not strongly peaked, which implies that the depth can differ per input even for a single real-world discount and prediction type.

# 5.5 VISUALIZING THE PREDICTIONS IN THE POOL DOMAIN

We test the quality of the predictions in the pool domain to evaluate whether they are well-suited to making decisions. For each sampled pool position, we consider a set  $I$  of different initial conditions (different angles and velocity of the white ball), and ask which is more likely to lead to pocketing coloured balls. For each initial condition  $s \in I$ , we apply the  $(r, \gamma, \lambda)$ -predictron (shared cores, 16 model steps, no skip connections) to obtain predictions  $\mathbf{g}^{\lambda}$ . We sum the predictions that correspond to pocketing any ball except the white ball, and to real-world discounts  $\gamma = 0.98$  and  $\gamma = 1$ . We select the condition  $s^{*}$  that maximises this sum.

![](images/b4c9c4ec541b6a61ab43c5d89390a4ea01ec2268306d36b621baacfc18effd9a.jpg)  
Figure 6: Thinking depth. Distributions of thinking depth on pool for different types of predictions and for different real-world discounts. Depth is defined as  $\lambda^0 (1 + \gamma^1\lambda^1 (1 + \gamma^2\lambda^2 (1 + \ldots)))$

![](images/465ab374518c031ca5e80572e192cfa35f3c0fa5b797b83251c0450b6cebe80e.jpg)

![](images/e05aab0cade0366d84d39cfea5f5490fe5b3d5284fc47d7eb12381946fa31df9.jpg)

![](images/c2d628a613ab3da12b20b7e7f6e20c6d87d7b233920b16960ebbeda703e35219.jpg)

![](images/3499bab54bebe8b43561bc3d6e4761006a9d0a408d35bf6f9e18dabb430940a8.jpg)

We then roll forward the pool simulator from  $s^*$  and log the number of pocketing events. Figure 2 shows a sampled rollout, using the predictron to pick  $s^*$ . When providing the choice of 128 angles and two velocities for initial conditions  $(|I| = 256)$ , this procedure resulted in pocketing 27 coloured balls in 50 episodes. Using the same procedure with an equally deep convolutional network only resulted in 10 pocketing events. These results suggest that the lower loss of the learned  $(r, \gamma, \lambda)$ -predictron translated into meaningful improvements when informing decisions. A video of the rollouts selected by the predictron is available here: https://youtu.be/BeaLdaN2C3Q.

# 6 RELATED WORK

Lee et al. (2015) introduced a neural network architecture where classifications branch off intermediate hidden layers. An important difference with respect to the  $\lambda$ -predictron, is that the weights are hand-tuned as hyper-parameters, whereas in the predictron the  $\lambda$  weights are learnt and, more importantly, conditional on the input. Another difference is that the loss on the auxiliary classifications is used to speed up learning, but the classifications themselves are not combined into an aggregate prediction; the output of the model itself is the deepest prediction.

Value iteration networks (Tamar et al., 2016) use convolutional and max-pooling layers to represent a step of value iteration. This is somewhat similar to a  $r$ -predictron, without  $\gamma$  and  $\lambda$ , with a single-layer convolutional core that is specialised to two-dimensional domains.

Schmidhuber (2015) discusses learning abstract models, but maintains separate losses for the model and a controller, and suggests training the model unsupervised to compactly encode the entire history of observations, through predictive coding. The predictron's abstract model is instead trained end-to-end to obtain accurate values.

# 7 DISCUSSION

The predictron is a single differentiable architecture that rolls forward an internal model to estimate values. This internal model may be given both the structure and the semantics of traditional reinforcement learning models. But unlike most approaches to model-based reinforcement learning, the model is fully abstract: it need not correspond to the real environment in any human understandable fashion, so long as its rolled-forward "plans" accurately predict outcomes in the true environment.

The predictron may be viewed as a novel network architecture that incorporates several separable ideas. First, there is the idea of outputting multiple predictions from a single network, and that these predictions should be self-consistent. Second, parts of the final prediction may be output along the way, by including not just value estimates at each point, but also intermediate rewards and discounts. Third, these various predictions may be combined into a learned ensemble that outputs a final aggregate prediction for each output. Our experiments demonstrate that these differences result in much more accurate predictions of the environment than more conventional network architectures.

We have focused on prediction tasks in uncontrolled environments. However, these ideas may transfer to the control setting, for example by using the predictron as a Q-network (Mnih et al., 2015). Even more intriguing is the possibility of learning an internal MDP with abstract internal actions, rather than the MRP model considered in this paper. We aim to explore these ideas in future work.

# REFERENCES

R.Bellman.Dynamic programming.Princeton University Press,1957.  
X. Glorot, A. Bordes, and Y. Bengio. Deep sparse rectifier neural networks. In Aistats, volume 15, page 275, 2011.  
A. Graves. Adaptive computation time for recurrent neural networks. CoRR, abs/1603.08983, 2016. URL http://arxiv.org/abs/1603.08983.  
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015.  
S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
D. P. Kingma and J. B. Adam. A method for stochastic optimization. In International Conference on Learning Representation, 2015.  
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply-supervised nets. In AISTATS, volume 2, page 6, 2015.  
T. Lillicrap, J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra. Continuous control with deep reinforcement learning. In ICLR, 2016.  
V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Ried-miller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.  
V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, 2016.  
J. Oh, X. Guo, H. Lee, R. L. Lewis, and S. Singh. Action-conditional video prediction using deep networks in atari games. In Advances in Neural Information Processing Systems, pages 2863-2871, 2015.  
J. Schmidhuber. On learning to think: Algorithmic information theory for novel combinations of reinforcement learning controllers and recurrent neural world models. arXiv preprint arXiv:1511.09249, 2015.  
R. S. Sutton. Learning to predict by the methods of temporal differences. Machine Learning, 3: 9-44, 1988.  
R. S. Sutton. Integrated architectures for learning, planning and reacting based on dynamic programming. In Machine Learning: Proceedings of the Seventh International Workshop, 1990.  
R. S. Sutton. TD models: Modeling the world at a mixture of time scales. In Proceedings of the Twelfth International Conference on Machine Learning, pages 531-539, 1995.  
R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction. The MIT press, Cambridge, MA, 1998.  
R. S. Sutton, J. Modayil, M. Delp, T. Degris, P. M. Pilarski, A. White, and D. Precup. Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction. In The 10th International Conference on Autonomous Agents and Multiagent Systems-Volume 2, pages 761-768. International Foundation for Autonomous Agents and Multiagent Systems, 2011.  
A. Tamar, S. Levine, and P. Abbeel. Value iteration networks. arXiv preprint arXiv:1602.02867, 2016.  
E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026-5033. IEEE, 2012.

![](images/0d2bcb638cdd7828175523ad08731d9ebf2cb7b4df8057c13d5e0c34cfa7addb.jpg)  
Figure 7: The predictron core used in our experiments.
