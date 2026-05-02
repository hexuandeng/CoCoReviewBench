# S3TA: A SOFT, SPATIAL,SEQUENTIAL, TOP-DOWN ATTENTION MODEL

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present a soft, spatial, sequential, top-down attention model (S3TA). This model uses a soft attention mechanism to bottleneck its view of the input. A recurrent core is used to generate query vectors, which actively select information from the input by correlating the query with input- and space-dependent key maps at different spatial locations.

We demonstrate the power and interpretability of this model under two settings. First, we build an agent which uses this attention model in RL environments and show that we can achieve performance competitive with state-of-the-art models while producing attention maps that elucidate some of the strategies used to solve the task. Second, we use this model in supervised learning tasks and show that it also achieves competitive performance and provides interpretable attention maps that show some of the underlying logic in the model's decision making.

# 1 INTRODUCTION

Traditional RL agents and image classifiers rely on some combination of convolutional and fully connected components to gradually process input information and arrive at a set of policy or class logits. This sort of architecture is very effective, but does not lend itself to easy understanding of how decisions are made, what information is used and why mistakes are made. Previous efforts to visualize deep RL agents (Greydanus et al. (2017); Zahavy et al. (2016); Wang et al. (2015)) focus on generating saliency maps to understand the magnitude of policy changes as a function of a perturbation of the input. This can uncover some of the "attended" regions, but may be difficult to interpret. For example, it can't reveal certain types of behavior when the agent makes decisions based on components absent from a frame. Our mechanism provides a more direct interpretation by making the attention a core part of the network.

In this work we present a soft, spatial, sequential and top-down attention model (S3TA, pronounced SETA). This model enables us to build agents and classifiers that actively select important, task-relevant information from visual inputs by sequentially querying and receiving compressed query-dependent summaries to generate appropriate outputs. To do this, the model generates attention maps, which can uncover some of underlying decision process used to solve the task. By observing and analyzing the resulting attention maps we can make educated guesses at how the system solves a task and where and why it might be failing. In the RL domain, we observe that the attention focuses on the key components of each level: tracking the region ahead of the player, focusing on enemies and important moving objects. In supervised learning, we observed that the attention sequentially focuses on different portions of the input to build up confidence in a classification or resolve ambiguity between different class labels. We also find that our model maintains competitive performance on both learning paradigms while providing interpretability.

# 2 MODEL

Our model, outlined in Figure 1, queries a large input tensor through an attention mechanism and uses the returned compressed answer (a low dimensional summary of the input) to produce its output. We refer to this full query-answer system as an attention head. Our system can implement multiple attention heads by producing multiple queries and receiving multiple answers.

An observation  $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$  at time  $t$  (here an RGB frame of height  $H$  and width  $W$ ) is passed through a "vision core". The vision core is a multi-layer convolutional network  $\mathrm{vis}_{\theta}$  followed by a recurrent layer with state  $s_{\mathrm{vis}}(t)$  such as a ConvLSTM (Shi et al. (2015)), which produces an output tensor  $\mathbf{O}_{\mathrm{vis}} \in \mathbb{R}^{h \times w \times c}$ :

$$
\mathbf {O} _ {\text {v i s}}, \boldsymbol {s} _ {\text {v i s}} (t) = \operatorname {v i s} _ {\theta} (\boldsymbol {X} (t), \boldsymbol {s} _ {\text {v i s}} (t - 1)) \tag {1}
$$

The vision core output is then split along the channel dimension into two tensors: the "Keys" tensor  $\mathbf{K} \in \mathbb{R}^{h \times w \times C_k}$  and the "Values" tensor  $\mathbf{V} \in \mathbb{R}^{h \times w \times C_v}$ , with  $c = C_V + C_K$ . To the keys and values tensors we concatenate a spatial basis — a fixed tensor  $\mathbf{S} \in \mathbb{R}^{h \times w \times C_S}$  which encodes spatial locations (see below for details).

A recurrent neural network (RNN) with parameters  $\phi$  produces  $N$  queries, one for each attention head. The RNN sends its state  $s_{\mathrm{RNN}}$  from the previous time step  $t - 1$  into a "Query Network". The query network  $Q_{\psi}$  is a multi-layer perceptron (MLP) with parameters  $\psi$  whose output is reshaped into  $N$  query vectors  $\pmb{q}^n$  of size  $C_k + C_S$  such that they match the channel dimension of  $\mathbf{K}$ :

$$
\boldsymbol {q} ^ {1} \dots \boldsymbol {q} ^ {N} = Q _ {\psi} \left(\boldsymbol {s} _ {\mathrm {R N N}} (t - 1)\right) \tag {2}
$$

Similar to Vaswani et al. (2017), we take the inner product between each query vector  $\mathbf{q}^n$  and all spatial locations in the keys tensor  $\mathbf{K}$  to form the  $n$ -th attention logits map  $\tilde{\mathbf{A}}^n \in \mathbb{R}^{h \times w}$ :

$$
\tilde {A} _ {i, j} ^ {n} = \sum_ {c} q _ {c} ^ {n} K _ {i, j, c} \tag {3}
$$

where  $K \in \mathbb{R}^{h \times w \times C_k + C_S}$  is the concatenation along the channel dimension of  $\mathbf{K}$  and  $\mathbf{S}$ . We then take the spatial softmax to form the final normalized attention map  $A^n$ :

$$
A _ {i, j} ^ {n} = \frac {\exp \left(\tilde {A} _ {i , j} ^ {n}\right)}{\sum_ {i , j} \exp \left(\tilde {A} _ {i , j} ^ {n}\right)} \tag {4}
$$

Each attention map  $\mathbf{A}^n$  is broadcast along the channel dimension of the values tensor  $\mathbf{V}$ , pointwise multiplied with it and then summed across space to produce the  $n$ -th answer vector  $\mathbf{a}^n \in \mathbb{R}^{1 \times 1 \times C_v + C_s}$ :

$$
a _ {c} ^ {n} = \sum_ {i, j} A _ {i, j} ^ {n} V _ {i, j, c} \tag {5}
$$

where  $V \in \mathbb{R}^{h \times w \times C_v + C_S}$  is the concatenation along the channel dimension of  $\mathbf{V}$  and  $\mathbf{S}$ . Finally, the  $N$  answer vectors  $\mathbf{a}^n$ , and the  $N$  query form the input to the RNN core to produce the next RNN state  $s_{\mathrm{RNN}}(t)$  and output  $o(t)$  for this time step:

$$
\boldsymbol {o} (t), \boldsymbol {s} _ {\mathrm {R N N}} (t) = \mathrm {R N N} _ {\phi} \left(a ^ {1}, \dots , a ^ {n}, q ^ {1}, \dots , q ^ {n}, \boldsymbol {s} _ {\mathrm {R N N}} (t - 1)\right) \tag {6}
$$

The exact details for each of the networks, outputs and states are given in Section 4 and the Appendix.

It is important to emphasize several points about the proposed model. First, the model is fully differentiable due to the use of soft-attention and can be trained using back-propagation. Second, the query vectors are a function of the RNN core state alone and not the observation — this allows for a "top-down" mechanism where the RNN can actively query the input for task-relevant information rather than having to filter out large amounts of information. Third, the spatial sum (equation 5) is a severe spatial bottleneck, which forces the system to make the attention maps in such a way that information is not "blurred" out during summation.

The summation of the values tensor of shape  $h \times w \times C_v$  to an answer of shape  $1 \times 1 \times C_v$  is invariant to permutation of spatial position, which emphasizes the need for the spatial basis. Due to the spatial structure being lost during the spatial summation, the only way the RNN core can know and reason about spatial positions is by using the channels coming from the spatial basis<sup>1</sup>. We postulate that the query and answer structure can have different "modes" — the system can ask "where" ("what") something is by sending out a query with zeros in the spatial channels of the query and non-zeros in the channels corresponding to the keys (which are input dependent). It can then read the answer from the spatial channels, localizing the object of interest. Conversely it can ask "what is in this

![](images/c665e24a86167759b78cb09b9a634ba93d845c6d603fb0890f458e311a9a09b5.jpg)  
Figure 1: An outline of our proposed model. Observations pass through a (recurrent) vision core network, producing a "keys" and a "values" tensor, to both of which we concatenate a spatial basis tensor (see text for details). A recurrent network at the top sends its state from the previous time-step into a query network which produces a set of query vectors (only one is shown here for brevity). We calculate the inner product between each query vector and each location in the keys tensor, then take the spatial softmax to produce an attention map for the query. The attention map is broadcast along the channel dimension, point-wise multiplied with the values tensor and the result is then summed across space to produce an answer vector. This answer is sent to the top RNN as input to produce the output and next state of the RNN.

particular location" by zeroing out the content channels of the query and putting information on the spatial channels, reading the content channels of the answer and ignoring the spatial channels. This is not a dichotomy as the two can be mixed (e.g. "find enemies in the top left corner"), but it does point to an interesting "what" and "where" separation, which we discuss in Section 4.1.5.

# 2.1 THE SPATIAL BASIS

The spatial basis  $\mathbf{S} \in \mathbb{R}^{h \times w \times C_S}$  such that the channels at each location  $i, j$  encode information about the spatial position. Adding this information into the values of the tensor allows some spatial information to be maintained after the spatial summation (equation 5) removes the structural information. Following Vaswani et al. (2017) and Parmar et al. (2018) we use a Fourier basis type of representation. Each channel  $(u, v)$  of  $\mathbf{S}$  is an outer product of two Fourier basis vectors. We use both odd and even basis functions with several frequencies. For example, with two even functions one channel of  $\mathbf{S}$  with spatial frequencies  $u$  and  $v$  would be:

$$
\mathbf {S} _ {i, j, (u, v)} = \cos (\pi u i / h) \cos (\pi v j / w) \tag {7}
$$

where  $u, v$  are the spatial frequencies in this channel,  $i, j$  are spatial locations in the tensor and  $h, w$  are correspondingly the height and width of the tensor. We produce all the outer products such that the number of channels in  $\mathbf{S}$  is  $(U + V)^2$  where  $U$  and  $V$  are the number of spatial frequencies we use for the even and odd components (4 for both throughout this work, so 64 channels in total).

The spatial basis can also be learned as another parameter of the model — while we tested this in some cases we did not observe that this makes a big difference in performance and for brevity this is not done in this work.

# 3 RELATED WORK

There is a vast literature in recurrent attention models. They have been applied with some success to question-answering datasets (Hermann et al., 2015), text translation (Vaswani et al., 2017; Bahdanau

et al., 2014), video classification and captioning (Shan & Atanasov, 2017; Li et al., 2017), image classification and captioning (Mnih et al., 2014; Chung & Cho, 2018; Fu et al., 2017; Ablavatski et al., 2017; Xiao et al., 2015; Zheng et al., 2017; Wang et al., 2017; Xu et al., 2015; Ba et al., 2014), text classification (Yang et al., 2016; Shen & Lee, 2016), generative models (Parmar et al., 2018; Zhang et al., 2018; Kosiorek et al., 2018), object tracking (Kosiorek et al., 2017), and reinforcement learning (Choi et al., 2017). These attention mechanisms can be grouped by whether they use hard attention (e.g. Mnih et al. (2014); Ba et al. (2014); Malinowski et al. (2018)) or soft attention (e.g. Bahdanau et al. (2014)) and whether they explicitly parameterize an attention window (e.g. Jaderberg et al. (2015); Shan & Atanasov (2017)) or use a weighting mechanism (e.g. Vaswani et al. (2017); Hermann et al. (2015)).

Our work introduces a novel architecture which builds on existing methods. We use a soft key, query, and value type of attention similar to Vaswani et al. (2017) and Parmar et al. (2018), but instead of doing "self"-attention where the queries come from the input (together with the keys and values) we have a different, top-down source for them. This enables the system to be both state/context dependent and input dependent. Furthermore the output of the attention model is highly compressed and has no spatial structure (other than the one preserved using the spatial basis), unlike in "self" attention where each pixel attends to every other pixel and the structure is preserved. Finally, we apply the attention sequentially in time similar to Xu et al. (2015) but with a largely different attention mechanism.

Of existing models, the MAC model (Hudson & Manning, 2018) is the closest to ours. There are several differences between our model and MAC. First, MAC was built to solve CLEVR (Johnson et al., 2017); major parts of it are geared for that dataset. Specifically the "control" unit is built to expect a guiding question for the reasoning process — this may not always exist, such as in the case of RL or classic supervised learning where the systems needs to come up with its own queries to produce the required output. Another difference is the use of a pre-trained ResNet-101 (Wang et al., 2017) as the visual backend; we train the visual core to co-adapt with the top-down mechanism such that it learns to produce useful keys and values for different queries. Finally, MAC does not use a spatial basis. It can still reason about space to some extent through the fully connected layers, but there is not a clear separation between space and content as in our model.

# 4 ANALYSIS AND RESULTS

# 4.1 REINFORCEMENT LEARNING

We use the Arcade Learning Environment (Bellemare et al. (2013b)) to train and test our agent on 57 different Atari games.

For this experiment, the model uses a 3 layer convolutional neural network followed by a convolutional LSTM as the vision core. The RNN is an LSTM that generates a policy  $\pi$  and a baseline function  $V^{\pi}$ ; it takes as input the query and answer vectors, the previous reward and a one-hot encoding of the previous action. The query network is a three layer MLP, which takes as input the hidden state  $h$  of the LSTM from the previous time step and produces 4 attention queries. See Appendix A.1.1 for a full specification of the network sizes.

We use the Importance Weighted Actor-Learner Architecture (Espeholt et al. (2018)) training architecture to train our agents. We use an actor-critic setup and a VTRACE loss with an RMSProp optimizer (see learning parameters in Appendix A.1.1 for more details).

We compare against two models without bottlenecks to benchmark performance, both using the deeper residual network described in Espeholt et al. (2018). In the Feedforward Baseline, the output of the ResNet is used to directly produce  $\pi$  and  $V^{\pi}$ , while in the LSTM Baseline an LSTM with 256 hidden units is inserted on top of the ResNet. The LSTM also gets as input the previous action and previous reward. We find that our agent is competitive with these state-of-the-art baselines, see Table 1 for benchmark results and Appendix A.1.3 for learning curves and performance on individual levels. Our model provides an attention map which shows the parts of space which are attended to by each attention head. This gives us hints as to what information from the input is used when producing the output. Though these do not necessarily tell the whole story of decision making,

![](images/c7fc3d02acc5ff91a7f8d43ccb1e8033233920b9775ae42a5bc0199ae56a396a.jpg)  
(a) Seaquest

![](images/e98f5f8dd41a3c97d24563b9281cc08bcc7da16bb4e73b0670b7948fb0600038.jpg)  
(b) Star Gunner  
Figure 2: Basic attention patterns. Bright areas are regions of high attention. Here we show 2 of the 4 heads used (one head in each row, time goes from left to right). The model learns to attend key sprites such as the player and different enemies. Best viewed on a computer monitor. See text for more details.

they do expose some of the strategies used by the model to solve the different tasks. Here we present some of these strategies and their relationship to the task at hand. Additionally, we analyze the use of the spatial basis vs. keys in the queries as a first step towards understanding the "what" and "where" in the system. We note that all the strategies we discuss here have been observed in more than one game or task; they are reproducible across multiple runs and we postulate they are effective strategies for the solution of the task at hand.

In order to visualize the attention maps we show the original input frame and super-impose the attention map  $A^n$  for each head on it using alpha blending. This means that the bright areas in all images are the ones which are attended to, darker areas are not. We find the range of values to be such that areas which are not attended have weights very close to zero, meaning that little information is "blended" from these areas during the summation in equation 5. A more detailed analysis of the distribution of weights can be seen in Appendix A.2.1.

# 4.1.1 THE ROLE OF TOP-DOWN INFLUENCE

To test the importance of the top-down queries, we train two additional agents with modified attention mechanisms that do not receive queries from the top-level RNN but are otherwise identical to our agent. The first agent uses the same attention mechanism except that the queries are a learnable bias tensor which does not depend on the LSTM state. The second agent does away with the query mechanism entirely and forms the weights for the attention by computing the L2 norm of each key (similar to a soft version of Malinowski et al. (2018)). Both of these modifications turn the top-down attention into a bottom-up attention, where the vision network has total control over the attention weights.

We train these agents on 7 ATARI games for  $2e9$  steps and compare the performance to the agent with top-down attention. We see significant drops in performance on 6 of the 7 games. On the remaining game, Seaquest, we see substantially improved performance; the positions of the enemies follow a very specific pattern, so there is little need for sequential decision making in that environment. On these games we see a median human normalized score of  $541.1\%$  for the attention agent,  $274.7\%$  for the fixed-query agent, and  $274.5\%$  for the L2-Norm Key Agent. Mean scores are  $975.5\%$ ,  $615.2\%$  and  $561.0\%$  respectively. See Appendix A.1.4 for more details.

Table 1: Human normalized scores for experts on ATARI.  

<table><tr><td>Model</td><td>Median</td><td>Mean</td></tr><tr><td>Feedforward Baseline</td><td>284.5%</td><td>1479.5%</td></tr><tr><td>LSTM Baseline</td><td>45.0%</td><td>1222.0%</td></tr><tr><td>Attention</td><td>407.1%</td><td>1649.0%</td></tr></table>

# 4.1.2 BASIC ATTENTION PATTERNS

The most dominant pattern we observe is that the model learns to attend to task-relevant things in the scene. In most ATARI games that usually means that the player is one of the focii of attention, as well as enemies, power-ups and the score itself (which is an important factor in the calculating the value function). Figure 2 (best viewed on screen) shows several examples of these attention maps. We also recommend watching the videos posted online for additional visualizations.

# 4.1.3 FORWARD PLANNING/SCANNING

In games where there is an element of forward planning and a direct mapping between image space and world space (such as 2D top-down view games) we observe that the model learns to scan through possible paths emanating from the player character and going through possible future trajectories. Figure 3 shows a examples of this in Ms Pacman and Alien — in the both games the model scans through possible paths, making sure there are no enemies or ghosts ahead. We observe that when it does see a ghost, another path is produced or executed in order to avoid it. Again we refer the reader to the videos for a better impression of the dynamics.

# 4.1.4 "TRIP WIRES"

In many games we observe that the agent learns to place "trip-wires" at strategic points in space such that if something crosses them a specific action is taken. For example, in Space Invaders two such trip wires are following the player ship on both sides such that if a bullet crosses one of them the agent immediately evades them by moving towards the opposite direction. Another example is Breakout where we can see it working in two stages. First the attention is spread out around the general area of the ball, then focuses into a localized line. Once the ball crosses that line the agent moves towards the ball. Figure 4 shows examples of this behavior.

# 4.1.5 “WHAT” VS. “WHERE”

As discussed in Section 2, each query has two components: one interacts with the keys tensor - which is a function of the input frame and vision core state - and the other interacts with the fixed spatial basis, which encodes locations in space. Since the output of these two parts is added together via an inner product prior to the softmax, we can analyze, for each query and attention map, which part of the query is more responsible for the the attention at each point; we can contrast the "what" from the "where". For example, during a game a query may be trying to find ghosts or enemies in the scene, in which case the "what" component should dominate as these can reside in many different places. Alternatively, a query could ask about a specific location in the screen (e.g., if it plays a special role in a game), in which case we would expect the "where" part to dominate.

In order to visualize this we color code the relative dominance of each part of the query. When a specific location is more influenced by the contents part, we will color the attention red, and when it is more influenced by the spatial part, we color it blue. Intermediate values will be white. More details can be found in Appendix A.2.

![](images/ba2fa9ea2f6af65f16728d99da5479750faf3d71996e69b7cc720eb1baf34a6f.jpg)  
Figure 3: Forward planning/scanning. We observe that in games where there is a clear mapping between image space and world space and some planning is required, the model learns to scan through possible future trajectories for the player and chooses ones that are safe/rewarding. The images show two such examples from Ms Pacman and Alien. Note how the paths follow the map structure. See text for more details and videos. Bright areas are regions of high attention.

![](images/dd49822f81c5ef9e2243028b3c056e438cc091bcd820e2b29a2c06dafd4ccb5b.jpg)  
Figure 4: Trip Wires. We observe in games where there are moving balls or projectiles that the agent sets up tripwires to create an alert when the object crosses a specific point or line. The agent learns how much time it needs to react to the moving object and sets up a spot of attention sufficiently far from the player. In Breakout (top row), one can see a two level tripwire: initially the attention is spread out, but once the ball passes some critical point it sharpens to focus on a point along the trajectory, which is the point where the agent needs to move toward the ball. In Space Invaders (bottom row) we see the tripwire acting as a shield; when a projectile crosses this point the agent needs to move away from the bullet. Bright areas are regions of high attention.

![](images/d15617e27588f74d35f73ea9bede417bb223d1ba96bf5b4a7673096ad016cebd.jpg)  
Figure 5: What/Where. This figures shows a sequence of 10 frames from Enduro (arranged left-to-right) along with the what-where visualization of each of the 3 of the 4 attention heads. (stacked vertically). The top row is the input frame at that timestep. Below we visualize the relative contribution of "what" vs. "where" in different attention heads: Red areas indicate the query has more weight in the "what" section, while blue indicates the mass is in the "where" part. White areas indicate that the query is evenly balanced between what and where. We notice that the first head here scans the horizon for upcoming cars and then starts tracking them (swithing from mixed to "what"). The second head is mostly a "where" query following the car for upcoming vehicles (a "trip-wire"). The last head here mostly tracks the player car and the score (mostly "what").  
Figure 5 shows several such maps  $C$  visualized in Enduro for different query heads. As can be seen, the system uses the two modes to make its decisions, some of the heads are content specific looking for opponent cars. Some are mixed, scanning the horizon for incoming cars and when found, tracking them, and some are location based queries, scanning the area right in front of the player for anything the crosses its path (a "trip-wire" which moves with the player). Examples of this mechanism in action can be seen in the videos online.

# 4.1.6 COMPARISON WITH OTHER ATTENTION ANALYSIS METHODS

In order to demonstrate that the attention masks are an accurate representation of where the agent is looking in the image, we perform the saliency analysis presented in Greydanus et al. (2017) on both the attention agent and the baseline feedforward agent. This analysis works by introducing a small, local Gaussian blur at a single point in the image and measuring the magnitude of the change in the policy. By measuring this at every pixel in the image, one can form a response map that shows how much the agent relies on the information at every spatial point to form its policy.

To produce these maps we run a trained agent for  $>200$  unperturbed frames on a level and then repeatedly input the final frame with perturbations at different locations. We form two saliency maps  $S_{\pi}(i,j) = 0.5||\pi(\mathbf{X}_{i,j}') - \pi(\mathbf{X})||^2$  and  $S_{V^{\pi}}(i,j) = 0.5||V^{\pi}(\mathbf{X}_{i,j}') - V^{\pi}(\mathbf{X})||^2$  where  $\mathbf{X}_{i,j}'$  is the input frame blurred at point  $(i,j)$ ,  $\pi$  are the softmax policy logits and  $V^{\pi}$  is the value function. An example of these saliency maps is shown in Figure 6. We see that the saliency map (in green) corresponds well with the attention map produced by the model and we see that the agent is sensitive to points in its planned trajectory, as we discussed in Section 4.1.3. Furthermore we see the heads specialize in their influence on the model — one clearly affects the policy more where the other affects the value function.

Comparing the attention agent to the baseline agent, we see that the attention agent is sensitive to more focused areas along the possible future trajectory. The baseline agent is more focused on the area immediately in front of the player (for the policy saliency) and on the score, while the attention agent focuses more specifically on the path the agent will follow (for the policy) and on possible future longer term paths (for the value).

![](images/4c16a0284c7c0c7386b6593198ab61e68296f70d17e6e4b367ac4029c7fc2307.jpg)  
(a) Policy saliency of the baseline agent

![](images/58cbb423d078e8a5cc844a04f0fa99dc8fb73f4f335a510b215474a3487c2eac.jpg)

![](images/3c74272aecf146a545f3a2df4cd99376963dfe34a438a94b58fd916351d29a8d.jpg)  
(c) Value saliency of the baseline agent

![](images/70419ad929345468a38e82eb4db8fd0d175b619fd93bbd0b7428661ebd84f1d8.jpg)  
(b) Policy saliency of the attention agent  
(d) Value saliency of the attention agent  
Figure 6: Saliency analysis. We run saliency analysis (see text for details) for the policy and value functions for both ours and the baseline feedforward agent. We visualize saliency in green, and in the case of our model the attention weights in white. We find that in the attention agent, one can see that the policy saliency (b) corresponds to the head that is most focused on the immediate actions of Pacman, while the value saliency (d) corresponds to the head that is looking further ahead (two scales of planning/scanning behaviour). Comparing the saliency of the baseline and attention agents, the attention agent exhibits sharper saliency, which looks along specific paths and follows the contours of the map. The saliency of the baseline agent (a, c) shows the network is concerned with shorter timescales and uses the score as the most important input to the value function (in some frames the value function does look at the map, but the majority of the time it is focused on the scene). See text for details and videos.

# 4.2 SUPERVISED LEARNING

We test the S3TA mechanism on several image and video classification problems to explore its applicability to other tasks. For image classification, we present the image to the network multiple times, allowing the model to ask new queries of the same image as a function of the previous class logits.

# 4.2.1 IMAGENET

For ImageNet classification, the model needs substantially more capacity than it does for reinforcement learning. For the vision core, we use a 50-layer ResNet (He et al. (2016)) with no recurrent layer (since there is no motion to process). On top of the ResNet we use a 3-layer MLP to produce

the class logits at each timestep. The output logits are accumulated across time, adding the output of the MLP to the current logits. The Query network is a 4-layer MLP that takes as input the previous (accumulated) logits. The cross-entropy loss is applied to the accumulated class logits at the final timestep.

We ran several baselines, including a standard ResNet 50-layer model. We also create a recurrent version of this model by using a shared, 1-layer MLP to transform each time step's logits into a  $224 \times 224$  tensor that is then added to the image at the next time step.

For our model, we find that accuracy initially improves as a function of the number of tiling steps and then degrades. Our best result is for sequence length of four timesteps and achieves  $73.4\%$  top-1,  $90.9\%$  top-5 accuracy. Our findings are summarized in Table 2. For ImageNet, S3TA initially

Table 2: Performance on ImageNet Test Dataset  

<table><tr><td>Model</td><td>Top-1</td><td>Top-5</td></tr><tr><td>Resnet-50 (He et al. (2016))</td><td>75.6%</td><td>92.9%</td></tr><tr><td>Resnet-50 (our setup)</td><td>74.0%</td><td>91.1%</td></tr><tr><td>Resnet-50, Sequence Length 4</td><td>70.2%</td><td>88.6%</td></tr><tr><td>Attention + Resnet-50, Sequence Length 1</td><td>73.1%</td><td>90.1%</td></tr><tr><td>Attention + Resnet-50, Sequence Length 4</td><td>73.4%</td><td>91.0%</td></tr><tr><td>Attention + Resnet-50, Sequence Length 8</td><td>69.0%</td><td>88.0%</td></tr></table>

attends to low-level edges (mostly around the contour of the object). It will then reduce the class choices under consideration by focusing on high-level features. In the case of dogs, the attention maps first identify that a type of dog is present; correspondingly, the class probabilities will be distributed across possible dog breed choices. The model will then focus on ears, faces, snouts and other distinctive features to tell the specific breed apart, producing peaked logits. An example of this is shown in Figure 7.

The model can alter its classification decisions midway through a sequence, even when it appears to be very confident. When dealing with occlusions, the model will use other image properties to gather relevant class context. An example of this is show in Figure 8. This shows the model is able to perform meaningful sequential computation that significantly alter its classification choices.

![](images/b9b0ec12a2d682b20a4259bffedebf946a855f09dd7a0b14fcbe4aad44de3aee.jpg)

![](images/161a858e82ef226fc3fc13364c878a4eab5ccb7473d022b9466616f3e81abf4c.jpg)

![](images/94470c46eb7958c52790021f30d4a9f6cdee3a2baf253ac1af353c2b92c46877.jpg)

![](images/32edc512a38d8c912db0032b9c510345a48272ad5a80b1788cb4897059a1fa7e.jpg)

![](images/84fe17f9f4a3a087f8a789e0d24ba6084c4ebcc5fe39301fd0a83492125132a6.jpg)

![](images/68aaa02a72e7949136de41c6628cf2de021eb4a857a03c249042bbac6e4fef3e.jpg)

![](images/11538f9b6f5a5b4f61220ab8d6f9d27b7d3038a837a973007ef3e77951e3432d.jpg)  
(a) Shetland Sheepdog

![](images/ac3464e2b2b12fc7dd6fbee93bd727a213a45e6d136248b80f5a182709749efe.jpg)  
Figure 7: ImageNet classification on two dog images from ImageNet. The input image is tiled four times. From left to right, the top row shows the input image then the four attention steps. The bottom row shows the corresponding logit outputs at each timestep. By the third frame, the model is sure both images are dogs, as indicated by similar class probability distributions. The attention snaps to specific patches in the last frame to discern the specific dog breed.

![](images/b82703acf17e6162f627ec291c69e52ac1af5f3c27847e97d3ec8c1f83718734.jpg)  
(b) Chiuaua

# 4.2.2 KINETICS

Kinetics is an action recognition video dataset where the goal is to classify videos portraying different actions correctly. We ran our model on the September 2018 version of the Kinetics 600 dataset (Carreira et al., 2018). For this model our vision core is a 34 layer ResNet followed by a convolutional LSTM; the rest of the model is identical to the ImageNet model. The videos in the dataset consist of 256 frames, from which we select 32 equally spaced frames to be processed sequentially by the model. As before, the class logits are accumulated across the sequence and the last one is

![](images/caf03254db2475497bd42fa3440ea6a373636e74f8ed0a1087a553ba51a80eff.jpg)

![](images/130575a4eb4b373816d36130b77811b991c16f61cccb79220c6db478eec79fcc.jpg)

![](images/442f28c646a44292ad5dda5ec59e0735e058ea9b700fd5ade7ed331a761dca93.jpg)

![](images/9fb156a1e9b349fd4ed49dc30300e521ea2ee4cf3d624bfa9028685f308144a3.jpg)

![](images/25c543f20791c93a2d6b0aa0b12739390443148f3b82f64323461eb5480f5d6f.jpg)

![](images/6fbdc703be26eaf1495fff5b20d03a2e72949626fdec2214c54dcbbedcee52d6.jpg)

![](images/8fe9a2af6d28bd828a26de8cf562b9da28a756ccb995e8d8da9042e9b53a6b96.jpg)  
(a) Chainsaw

![](images/2d27d4b9e1e5999f6367e11a5e8a3e57c9fab87baeb53475e433c0d9f5881717.jpg)  
(b) Horse Cart  
Figure 8: Confusion on ImageNet. In the first image, the tree-filled background initially makes S3TA suspect the class is "lumbermill". However, lumbermills are buildings full of mechanical items. The attention in the final frame focuses solely on the chainsaws, which become its final class choice. In the second, the horse is occluded in this image, and so S3TA has to use other clues to distinguish between "shopping cart", "barrow", and "horse cart". In the last frame, the attention maps focus on the horse whip on the right and the wheel type.

used as the output. We achieve  $58\%$  top-1,  $82\%$  top-5 accuracy on this dataset. The state-of-the-art (Carreira et al., 2018) achieves  $71.7\%$  top-1 accuracy,  $90.4\%$  top-5 accuracy.

In the case of the Kinetics dataset, the attention model often refrains from making a class prediction until a key item appears in the video sequence. The attention maps then focus on this object while it remains in view. For instance, the attention focuses on the musical instrument a person is playing, and the policy logits the narrow down to a few probable choices. If an action sequence is a sport, then the focus is typically on the game ball. Figure 9 shows an example of this behavior.

![](images/33accc5bf78b7707373d977a87c93e7892ef8d32e3374ea2286dcab1a7d233e4.jpg)  
Figure 9: Focus on Key Items. The attention maps are disperse until a trumpet appears in view, at which point the class logits become very peaked. Bright areas are regions of high attention.

# 5 CONCLUSION

We have introduced S3TA, a model for sequential spatial top-down attention. This model learns to query its input for task-relevant information and receive spatially bottlenecked answers. The model performs well on a variety of RL and supervised learning tasks while providing some interpretability of its reasoning process.

The attention mechanism produces attention maps which can be used to visualize which parts of the input are attended to. We have seen that the agent is able to make use of a combination of "what" and "where" queries to select both regions and objects within the input depending on the task. In RL agents, we have seen that the agents are able to learn to focus on key features of the inputs, look ahead along short trajectories, and place tripwires to trigger certain behaviors. In supervised models, the model sequentially focuses on important parts of the model to build up confidence in its classification, and will hold off narrowing down its decision until key pieces of information become available. In both the RL and supervised learning paradigms, the model yields interpretable results without sacrificing performance.

# REFERENCES

Artsiom Ablavatski, Shijian Lu, and Jianfei Cai. Enriched deep recurrent visual attention model for multiple object recognition. In Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on, pp. 971-978. IEEE, 2017.  
Jimmy Ba, Volodymyr Mnih, and Koray Kavukcuoglu. Multiple object recognition with visual attention. arXiv preprint arXiv:1412.7755, 2014.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.  
Yavar Bellemare, Marc G.and Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: an evaluation platform for general agents. Journal of Artificial Intelligence Research, 47:253-279, 2013b.  
Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew Zisserman. A short note about kinetics-600. 2018.  
Jinyoung Choi, Beom-Jin Lee, and Byoung-Tak Zhang. Multi-focus attention network for efficient deep reinforcement learning. arXiv preprint arXiv:1712.04603, 2017.  
Minki Chung and Sungzoon Cho. Cram: Clued recurrent attention model. arXiv preprint arXiv:1804.10844, 2018.  
Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. IM-PALA: scalable distributed deep-rl with importance weighted actor-learner architectures. CoRR, abs/1802.01561, 2018. URL http://arxiv.org/abs/1802.01561.  
Jianlong Fu, Heliang Zheng, and Tao Mei. Look closer to see better: Recurrent attention convolutional neural network for fine-grained image recognition. In CVPR, volume 2, pp. 3, 2017.  
Sam Greydanus, Anurag Koul, Jonathan Dodge, and Alan Fern. Visualizing and understanding atari agents. CoRR, abs/1711.00138, 2017. URL http://arxiv.org/abs/1711.00138.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. 2016.  
Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems, pp. 1693-1701, 2015.  
Drew A. Hudson and Christopher D. Manning. Compositional attention networks for machine reasoning. volume abs/1803.03067, 2018.  
Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al. Spatial transformer networks. In Advances in neural information processing systems, pp. 2017-2025, 2015.  
Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, and Koray Kavukcuoglu. Population based training of neural networks. CoRR, abs/1711.09846, 2017. URL http://arxiv.org/abs/1711.09846.  
Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross B. Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1988-1997, 2017.  
Adam Kosiorek, Alex Bewley, and Ingmar Posner. Hierarchical attentive recurrent tracking. In Advances in Neural Information Processing Systems, pp. 3053-3061, 2017.  
Adam R Kosiorek, Hyunjik Kim, Ingmar Posner, and Yee Whye Teh. Sequential attend, infer, repeat: Generative modelling of moving objects. arXiv preprint arXiv:1806.01794, 2018.

Xuelong Li, Bin Zhao, and Xiaoqiang Lu. Mam-rnn: multi-level attention model based rnn for video captioning. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, 2017.  
Mateusz Malinowski, Carl Doersch, Adam Santoro, and Peter Battaglia. Learning visual question answering by bootstrapping hard attention. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 3-20, 2018.  
Volodymyr Mnih, Nicolas Heess, Alex Graves, et al. Recurrent models of visual attention. In Advances in neural information processing systems, pp. 2204-2212, 2014.  
Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Łukasz Kaiser, Noam Shazeer, and Alexander Ku. Image transformer. arXiv preprint arXiv:1802.05751, 2018.  
Mo Shan and Nikolay Atanasov. A spatiotemporal model with visual attention for video classification. arXiv preprint arXiv:1707.02069, 2017.  
Sheng-syun Shen and Hung-yi Lee. Neural attention models for sequence classification: Analysis and application to key term extraction and dialogue act detection. arXiv preprint arXiv:1604.00077, 2016.  
Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-chun Woo. Convolutional LSTM network: A machine learning approach for precipitation nowcasting. CoRR, abs/1506.04214, 2015. URL http://arxiv.org/abs/1506.04214.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pp. 5998-6008, 2017.  
Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, and Xiaou Tang. Residual attention network for image classification. arXiv preprint arXiv:1704.06904, 2017.  
Ziyu Wang, Nando de Freitas, and Marc Lanctot. *Dueling network architectures for deep reinforcement learning*. CoRR, abs/1511.06581, 2015. URL http://arxiv.org/abs/1511.06581.  
Tianjun Xiao, Yichong Xu, Kuiyuan Yang, Jiaxing Zhang, Yuxin Peng, and Zheng Zhang. The application of two-level attention models in deep convolutional neural network for fine-grained image classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 842-850, 2015.  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron C. Courville, Ruslan Salakhutdinov, Richard S. Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In ICML, 2015.  
Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 1480-1489, 2016.  
Tom Zahavy, Nir Ben-Zrihem, and Shie Mannor. Graying the black box: Understanding dqns. CoRR, abs/1602.02658, 2016. URL http://arxiv.org/abs/1602.02658.  
Han Zhang, Ian J. Goodfellow, Dimitris N. Metaxas, and Augustus Odena. Self-attention generative adversarial networks. CoRR, abs/1805.08318, 2018.  
Heliang Zheng, Jianlong Fu, Tao Mei, and Jiebo Luo. Learning multi-attention convolutional neural network for fine-grained image recognition. In Int. Conf. on Computer Vision, volume 6, 2017.
