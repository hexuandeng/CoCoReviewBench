# DNBP: DIFFERENTIABLE NONPARAMETRIC BELIEF PROPAGATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present a differentiable approach to learn the probabilistic factors used for inference by a nonparametric belief propagation algorithm. Existing nonparametric belief propagation methods rely on domain-specific features encoded in the probabilistic factors of a graphical model. In this work, we replace each crafted factor with a differentiable neural network enabling the factors to be learned using an efficient optimization routine from labeled data. By combining differentiable neural networks with an efficient belief propagation algorithm, our method learns to maintain a set of marginal posterior samples using end-to-end training. We evaluate our differentiable nonparametric belief propagation (DNBP) method on a set of articulated pose tracking tasks and compare performance with learned baselines. Results from these experiments demonstrate the effectiveness of using learned factors for tracking and suggest the practical advantage over hand-crafted approaches. The project webpage is available at: https://sites.google.com/view/diff-nbp.

![](images/a740764b6ed212457f3bfb67f7735314d32a6f677ab15c8edff6c86ab72f81bf.jpg)  
Figure 1: Architecture diagram of differentiable nonparametric belief propagation. DNBP combines domain knowledge in the form of graphical models with differentiable neural networks for tractable inference in continuous spaces. Input features from a deep neural network and the probabilistic relationships encoded in a graphical model are learned jointly in an end-to-end fashion using backpropagation. Following offline training, DNBP can be applied to unseen data without hand-tuning.

# 1 INTRODUCTION

A significant challenge for robotic applications is the ability to estimate the pose of articulated objects in high noise environments. Nonparametric belief propagation (NBP) algorithms (Sudderth et al., 2003; Isard, 2003) have proven effective for inference in visual perception tasks such as human pose tracking (Sigal et al., 2004) and articulated object tracking in robotic perception (Desingh et al., 2019; Pavlasek et al., 2020). Moreover, these algorithms are able to account for uncertainty in their estimates when environmental noise is high and show promising computational properties in practice (Desingh et al., 2019; Ortiz et al., 2021). Their adaptability to new applications, however, is limited by the need to define hand-crafted functions that describe the distinct statistical relationships in a particular dataset. Current methods that utilize NBP rely on extensive domain knowledge to parameterize these relationships. Reducing the domain knowledge required by NBP methods would enable their use in a broader range of applications.

As a form of probabilistic graphical model inference, NBP algorithms leverage domain knowledge encoded in graph-based representations, such as the Markov random field (MRF). Their capacity to

perform inference using arbitrary graphs sets them apart from other algorithms such as the recursive Bayes filter (Thrun et al., 2005) (e.g. particle filter (Godsill, 2019)) and has been shown to be important in computational perception because it allows for modeling of non-causal relationships (Sudderth et al., 2003). Data-driven approaches are an alternative for computational perception (Xiang et al., 2018; Tremblay et al., 2018). These methods generally avoid the need for extensive domain knowledge by learning from large amounts of labelled data. Data-driven approaches, however, are prone to noisy estimates and have limited capacity to represent uncertainty inherent in their estimates. In robotic applications, both of these limitations negatively impact the ability for a robot to operate effectively in unstructured environments.

In this paper, we present a differentiable nonparametric belief propagation (DNBP) method, a hybrid approach which leverages neural networks to parameterize the NBP algorithm. Through differentiable inference, DNBP leverages the explainability and robustness of probabilistic inference techniques and capitalizes on the efficiency and generalizability of data-driven approaches. Inspired by the differentiable particle filter (DPF) from Jonschkowski et al. (2018) and the pull message passing for nonparametric belief propagation (PMPNBP) algorithm (Desingh et al., 2019), we develop a differentiable nonparametric belief propagation algorithm. DNBP performs end-to-end learning of each probabilistic factor required for graphical model inference.

The effectiveness of DNBP is demonstrated on two simulated articulated tracking tasks and on a real-world hand pose tracking tasks in challenging noisy environments. An analysis of the learned probabilistic factors and resulting tracking performance is used to validate the approach. Results show that our approach can leverage the graph structure to report uncertainty about its estimates while significantly reducing the need for prior domain knowledge required by previous NBP methods. DNBP performs competitively in comparison to traditional learning-based approaches on the tracking tasks. Collectively, these results indicate that DNBP has the potential to be successfully applied to robotic perception tasks, where a notion of uncertainty in the inference is inevitable.

# 2 RELATED WORK

Belief Propagation: In the context of graphical models, inference refers to the process in which information about observed variables is used to derive the posterior distribution(s) of unobserved variables. Belief propagation (BP) is a message passing algorithm for inference on graphical models. BP computes exact marginal distributions on trees (Pearl, 1988), and has demonstrated empirical success on loopy graphs (Murphy et al., 1999; Sun et al., 2003; Lee et al., 2008; Lan et al., 2006). In order to apply inference techniques such as BP and LBP, the parameters of a graphical model (e.g. the probabilistic factors) must be fully specified. Maximum likelihood estimation (MLE) has been shown to be an effective approach for learning the parameters of a graphical model from data (Murphy, 2012; Koller & Friedman, 2009; Ping & Ihler, 2017). In contrast, this current study focuses on parameter learning for use with inference of continuous random variables.

Nonparametric Belief Propagation: For continuous spaces, such as six degrees-of-freedom object pose, exact integrals called for in BP and LBP become intractable and approximate methods for inference have been considered. Nonparametric belief propagation (NBP) methods (Isard, 2003; Sudderth et al., 2003), have been proposed which represent the inferred marginal distributions using mixtures of Gaussians and define efficient message passing approximations for inference. Isard (2003) demonstrated the effectiveness of PAMPAS using a set of synthetic visual datasets each modeled with hand-crafted factors. Sudderth et al. (2003) applied their NBP method successfully to a visual parts-based face localization task as well as a human hand tracking task (Sudderth et al., 2004). In both applications, NBP relied on factor models which were chosen based on task-level domain knowledge (e.g. valid configurations of human hands). Sigal et al. (2004) extended these NBP methods to human pose estimation and tracking using factors which were each trained separate from the inference algorithm using independent training objectives.

Ihler & McAllester (2009) described a conceptual theory of particle belief propagation, where messages being sent to inform the marginal of a particular variable could be generated using a shared proposal distribution. Following the work of Ihler and McAllester, Desingh et al. (2019) presented an efficient "pull" message passing algorithm (PMPNBP) which uses a weighted particle set to approximate messages between random variables. PMPNBP was shown to be effective on robot pose estimation tasks using hand-crafted factors. Using a similar approximation of belief propagation,

Pavlasek et al. (2020) took a step toward neural network-based potential functions by introducing a pre-trained image segmentation network to the unary factors. An important limitation of these works is they assume the probabilistic factors expressed in the graph are provided as input or rely on domain knowledge to separately model and train each function. The potential for neural networks to learn the parameters used by alternative inference techniques has been demonstrated (Do & Artières, 2010; Thompson et al., 2014; Xiong & Ruozzi, 2020). In this paper we explore the potential for a deep learning framework to be used within the inference process of NBP such that the probabilistic factors may be learned in an end-to-end fashion.

Differentiable Bayes Filtering: In the context of robot state estimation, many approaches have recently been proposed that incorporate neural networks with recursive inference algorithms in an end-to-end fashion. Haarnoja et al. (2016) introduced a differentiable Kalman filter, and Jonschkowski & Brock (2016) proposed a differentiable, histogram-based Bayes filter algorithm. Jonschkowski et al. (2018) and Karkus et al. (2018) both proposed differentiable particle filter algorithms for modeling continuous state spaces. Kloss et al. (2021) evaluate recent differentiable filtering techniques. Yi et al. (2021) propose an end-to-end learning method for inference over factor-graph models. In contrast to these methods, which model a single object body using variants of the Bayes filter, this work sets out to study the potential for NBP to be used as an algorithmic prior for modeling multi-part articulated objects. Recently, this line of research on differentiable state estimation algorithms has extended into the planning domain (Karkus et al., 2019; Wang et al., 2020; Anderson et al., 2019). Exploration of embedding DNBP within a differentiable planning system is left as future work.

# 3 BELIEF PROPAGATION

Consider a Markov Random Field (MRF) defined by the undirected graph  $\mathcal{G} = \{\mathcal{V},\mathcal{E}\}$ , where  $\mathcal{V}$  denotes a set of nodes and  $\mathcal{E}$  denotes a set of edges. An example MRF model is shown in Fig. 2b. Each node in  $\mathcal{V}$  represents an observed (grey) or unobserved (white) random variable, while each edge in  $\mathcal{E}$  represents a pairwise relationship between two random variables in  $\mathcal{V}$ . The joint probability distribution for  $\mathcal{G}$  is:

$$
p (\mathcal {X}, \mathcal {Y}) = \frac {1}{Z} \prod_ {(s, d) \in \mathcal {E}} \psi_ {s d} \left(X _ {s}, X _ {d}\right) \prod_ {d \in \mathcal {V}} \phi_ {d} \left(X _ {d}, Y _ {d}\right) \tag {1}
$$

where  $\mathcal{X} = \{X_d \mid d \in \mathcal{V}\}$  is the set of unobserved variables and  $\mathcal{Y} = \{Y_d \mid d \in \mathcal{V}\}$  is the set of corresponding observed variables. The scalar  $Z$  is a normalizing constant. For each node, the function  $\phi_d(\cdot)$  is the unary potential, describing the compatibility of  $X_d$  with a corresponding observed variable  $Y_d$ . For each edge, the function  $\psi_{sd}(\cdot)$  is the pairwise potential, describing the compatibility of neighboring variables  $X_s$  and  $X_d$ . This work considers MRF models limited to pairwise clique potentials.

Given the factorization of the joint distribution defined in Eq. (1), BP provides an algorithm for inference of the marginal posterior distributions, know as the beliefs,  $bel_{d}(X_{d})$ . BP defines a message passing scheme for calculation of the beliefs as follows:

$$
b e l _ {d} \left(X _ {d}\right) \propto \phi_ {d} \left(X _ {d}, Y _ {d}\right) \prod_ {s \in \rho (d)} m _ {s \rightarrow d} \left(X _ {d}\right) \tag {2}
$$

where  $\rho(s)$  denotes the set of neighboring nodes of  $s$ . A message from node  $s$  to  $d$  is defined as:

$$
m _ {s \rightarrow d} (X _ {d}) = \int_ {X _ {s}} \phi_ {s} \left(X _ {s}, Y _ {s}\right) \psi_ {s d} \left(X _ {s}, X _ {d}\right) \times \prod_ {u \in \rho (s) \backslash d} m _ {u \rightarrow s} \left(X _ {s}\right) d X _ {s} \tag {3}
$$

Performing inference of random variables in continuous space causes the integral in Eq. (3) to become intractable. This motivates the use of efficient algorithms that approximate the message passing scheme of Eq. (2) and Eq. (3).

# 3.1 NONPARAMETRIC BELIEF PROPAGATION

Nonparametric belief propagation (NBP) (Sudderth et al., 2003) uses Gaussian mixtures to represent the beliefs and messages for continuous random variables. Later works, including Ihler &

McAllester (2009) and Desingh et al. (2019), further improve upon the tractibility of approximate nonparametric inference by representing beliefs and messages with sets of weighted particles. These particle-based NBP methods infer an approximation of the beliefs using an iterative message passing algorithm, in which beliefs and messages are updated at each iteration  $t$ . In particular, Desingh et al. (2019) avoid the expensive message generation of NBP by approximating Eq. (3) with a "pull" strategy. A message,  $m_{s\rightarrow d}^{t}$ , outgoing from  $s$  to  $d$ , is generated by first sampling  $M$  independent samples from  $bel_d^{t - 1}(X_d)$  then reweighting and resampling from this set.

# 4 DIFFERENTIABLE NONPARAMETRIC BELIEF PROPAGATION

We propose a differentiable nonparametric belief propagation (DNBP) method. DNBP maintains a representation of the uncertainty in the estimate by efficiently approximating the marginal posterior distributions encoded in an MRF. Our method avoids the need to define hand-crafted functions for each domain by modeling the potentials needed for the computation of the distributions with neural networks that are trained end-to-end. This hybrid generative-discriminative approach leverages the strengths of both NBP and neural networks.

DNBP uses an iterative, differentiable message passing scheme to infer the beliefs over hidden variables in an MRF. DNBP approximates the belief and messages in Eq. (2) and Eq. (3) at iteration  $t$  by sets of  $N$  and  $M$  weighted particles respectively:

$$
b e l _ {d} ^ {t} \left(X _ {d}\right) = \left\{\left(\mu_ {d} ^ {(i)}, w _ {d} ^ {(i)}\right) \right\} _ {i = 1} ^ {N} \tag {4}
$$

$$
m _ {s \rightarrow d} ^ {t} = \left\{\left(\mu_ {s d} ^ {(i)}, w _ {s d} ^ {(i)}\right)\right\} _ {i = 1} ^ {M} \tag {5}
$$

DNBP relies on a "pull" message passing strategy similar to the one presented by Desingh et al. (2019). In this strategy, each iteration of the algorithm is defined in terms of a message update step and a belief update step. The message update generates a new set of message particles as a reweighted set of samples from the previous iteration's belief. Crucially, the weights associated with these updated message samples result from learned probabilistic factors as opposed to handcrafted ones. Following a message update, the belief update combines information that is incoming to each node from the newly generated messages. Pseudocode of DNBP's message and belief update schemes is included in Appendix A.1. The following sections describe the networks used to compute the message and belief updates.

**Unary Potential Functions:** According to the factorization of the MRF joint distribution in Eq. (1), each unobserved variable  $X_{d}$ , for  $d \in \mathcal{V}$ , is related to a corresponding observed variable  $Y_{d}$  by the unary potential function  $\phi_d(X_d, Y_d)$ . DNBP models each unary function with a feedforward neural network. The unary potential for a particle,  $x_{d}$ , given an observed image,  $y_{d}$ , is:

$$
\phi_ {d} \left(X _ {d} = x _ {d}, Y _ {d} = y _ {d}\right) = l _ {d} \left(x _ {d} \oplus f _ {d} \left(y _ {d}\right)\right) \tag {6}
$$

where  $f_{d}$  is a convolutional neural network,  $l_{d}$  is a fully connected neural network, and the symbol  $\oplus$  denotes concatenation of feature vectors. Details of network architectures are given in Appendix A.2, Table 1.

Pairwise Potential Functions: For any pair of hidden variables,  $X_{s}$  and  $X_{d}$ , which are connected by an edge in  $\mathcal{E}$ , a pairwise potential function,  $\psi_{sd}(X_s,X_d)$ , represents the probabilistic relationship between the two variables. DNBP models each pairwise potential using a pair of feedforward, fully connected neural networks,  $\psi_{sd}(X_s,X_d) = \{\psi_{sd}^\rho (\cdot),\psi_{sd}^{\sim}(\cdot)\}$ . The pairwise density network,  $\psi_{sd}^{\rho}(\cdot)$ , evaluates the unnormalized potential for a pair of particles. The pairwise sampling network,  $\psi_{sd}^{\sim}(\cdot)$ , is used to form samples of node  $s$  conditioned on node  $d$  and vice versa. Details of network architectures are given in the Appendix A.2, Table 1. The weight computation is detailed in the pseudocode in Appendix A.1.

Particle Diffusion: DNBP uses a learned particle diffusion model for each hidden variable, modeled as distinct feedforward neural networks,  $\tau_d^{\sim}(\cdot)$  for  $d\in \mathcal{V}$ . This diffusion model replaces the Gaussian diffusion models typically used by particle-based inference methods. At the outset of message generation at iteration  $t$ , DNBP's belief particles from iteration  $t - 1$  are resampled then passed through the diffusion model at the beginning of iteration  $t$  to form the messages used to update the distributions at iteration  $t$ .

![](images/62cd6209505d79e3cb7f189a01f99693ae61cb421e894200f31c40f467177e1e.jpg)  
(a)

![](images/8dc2a106602dcbee7e83e4386a5e901944ac9edd1aa975e082039cee8d311ed4.jpg)  
Figure 2: a) Geometry and example configuration of the double pendulum. b) Graphical model used by DNPB for the double pendulum task. c) Geometry and an example configuration of the spider structure. d) Graphical model used by DNPB for the spider task.  
(b)

![](images/8af20aac5a2062cfab48d185fbf111d32fe0ed6cbb6ea69f5422ae7b591b2f5f.jpg)  
(c)

![](images/25865bd63bb3e82274e86b5b3de1de0f0e7720d5a8a059f2f2158d4fbe2e8d7c.jpg)  
(d)

Particle Resampling: The final operation of the belief update algorithm in NBP is a weighted resampling of belief particles. This resampling operation is non-differentiable (Karkus et al., 2018; Jonschkowski et al., 2018). It follows that the iterative belief update algorithm is non-differentiable due to the resampling step. DNBP addresses the non-differentiability of the belief update algorithm by relocating the resampling and diffusion operations to the beginning of the message update algorithm. With this modification, the belief update returns a weighted set of particles approximating the marginal beliefs. The resulting belief density estimate is differentiable up to the beginning of the message update, when particles from the previous iteration were resampled. The resulting algorithm is differentiable through one belief update and message passing updates.

# 4.1 SUPERVISED TRAINING

DNBP's training approach is inspired by the work of Jonschkowski et al. (2018) with modifications to enable learning the potential functions distinct to DNBP. During training, DNBP uses a set of observation sequences, and a corresponding set of ground truth sequences. Using the observation sequences, DNBP estimates belief of each unobserved variable at each sequence step. Then, by maximizing estimated belief at the ground truth label of each unobserved variable, DNBP learns its network parameters by maximum likelihood estimation. Further details regarding the implementation of the training procedure are discussed in Appendix A.2.

Objective Function: Given a set of weighted particles representing the belief of  $X_{d}$  produced by the inference procedure at iteration  $t$ , the density of the belief can be expressed as a mixture of Gaussians, with a component centered at each particle. The density of a sample  $x_{d}$  can be computed as follows:

$$
\overline {{b e l}} _ {d} ^ {t} \left(x _ {d}\right) = \sum_ {i = 1} ^ {N} w _ {d} ^ {(i)} \cdot \mathcal {N} \left(x _ {d}; \mu_ {d} ^ {(i)}, \Sigma\right) \tag {7}
$$

DNBP defines a loss function one each hidden node  $d \in \mathcal{G}$  as:

$$
L _ {d} ^ {t} = - \log \left(\overline {{b e l}} _ {d} ^ {t} \left(x _ {d} ^ {t, *}\right)\right) \tag {8}
$$

where  $x_{d}^{t,*}$  denotes the ground truth label for node  $d$  at sequence step  $t$ . The loss for each hidden node is computed and optimized separately. At each sequence step during training, DNBP iterates through the nodes of the graph, updating each node's incoming messages and belief followed by a single optimization step of Eq. (8) using stochastic gradient descent.

# 5 RESULTS

The capability of DNBP is demonstrated on three challenging articulated tracking tasks. The first two tasks involve visually tracking the articulated joints of simulated articulated structures, as illustrated in Fig. 2. To increase the difficulty of these tasks, simulated clutter<sup>1</sup> in the form of static and

dynamic geometric shapes are rendered into the image sequences. In the second task, we evaluate DNBP on its ability to track the articulated pose of human hands. In both experiments, DNBP is directly compared to learned baseline approaches that are not NBP.

# 5.1 DATASETS

Simulated Double Pendulum: To characterize DNBP's tracking performance under chaotic motion, the double pendulum task was chosen as an initial evaluation. The double pendulum structure consists of two revolute joints connected to two rigid-body links in series (see Fig. 2a for illustration), which are acted on by gravity. The pose of the double pendulum is modeled by the 2-dimensional position of its two revolute joints, rendered as yellow circles, and one end effector. The training set on this task consists of 1024 total sequences with 20 frames per sequence while the validation set consists of 150 total sequences with 20 frames per sequence. Both training and validation sequences are split evenly among three bins of clutter ratio: none, 0 to 0.04 and 0.04 to 0.1. Of the training and validation sequences with any amount of clutter, half contain static clutter and the other half contain dynamic clutter. The held-out test set is evenly split among clutter ratio deciles from 0 to 0.95, thus contains a shift in distribution from the training set, which was limited to clutter ratios below 0.1. Each decile contains 50 sequences with 100 frames per sequence. For test sequences with any amount of clutter, half contain static clutter and the other half contain dynamic clutter.

Simulated Articulated Spider: The spider task was chosen to further characterize DNBP's performance using a structure with added articulations and a larger graphical model. As depicted in Fig. 2c, the spider is comprised of three revolute-prismatic joints, three purely revolute joints, and six rigid-body links. An example of the spider is shown in Fig. 2c, in which the joints are rendered as yellow circles and the rigid-body links are rendered as coloured rectangles. Unlike the double pendulum, which contained a stationary base joint, the spider is not tethered to any position and can move freely throughout the image under simulated joint control. The training, validation and test set for this task follow the same respective distributions of clutter as were used in the double pendulum datasets. The training set consists of 2,048 total sequences and the validation set consists of 300 sequences. The training and validation sequences are split evenly among five bins of clutter ratio: none, 0 to 0.04 and 0.04 to 0.1, 0.1 to 0.2 and 0.2 to 0.3. There are 20 frames per sequence in each of the spider datasets. Both simulated tasks use images of size  $128 \times 128$  pixels. Ground truth keypoint locations are represented as continuous valued coordinates scaled to range of  $[-1, +1]$ .

First-Person Hand Action Benchmark: The FPHAB dataset (Garcia-Hernando et al., 2018) consists of RGB-D image sequences taken from the first-person perspective. Thus, the dataset captures the pose and motion of human hands as they perform typical actions. This is a challenging dataset with extreme occlusions where complete observations of all the finger joints are rare. In total, there are 1175 distinct sequences and 105459 individual image frames. Each image is labeled with the  $3D$  position of 21 hand joints (illustration of joint relations shown in center column of Fig. 1). The best-performing hand pose estimation baseline proposed by Garcia-Hernando et al. (2018) is used for comparison in the current study. Just like Garcia-Hernando et al. (2018), DNBP uses only depth observations. To ensure fair comparisons with the FPHAB baseline, this study follows the 1:1 cross-subject training protocol as described in FPHAB.

# 5.2 IMPLEMENTATION DETAILS

On all three tasks, Adam (Kingma & Ba, 2015) is used for network optimization with a batch size of 6 and models are trained until convergence of the validation loss. DNBP is trained using 100 particles per message and tested using 200 particles per message. During training, one message update is performed at each sequence step, while two message updates are used at test time. The pairwise density, pairwise sampling and diffusion sampling processes of DNBP are defined over the relative translations between neighboring nodes. The maximum weighted particle from each marginal belief set of DNBP is used during evaluation for comparison with the ground truth.

On both simulated tasks, DNBP is compared to an LSTM recurrent neural network (Hochreiter & Schmidhuber, 1997). Both models use image inputs that are normalized channel-wise based on training set statistics. The total number of trainable parameters between LSTM and DNBP were chosen to be similar. For hand tracking, the preprocessing protocol of Xiong et al. (2019), is followed. Notably, preprocessing on the hand tracking task assumes ground truth bounding boxes to

![](images/085167082114860207307b9f33f96059fbf6e917bed15b70ca00038a7344ec59.jpg)  
Figure 3: Average error of DNBP and LSTM predictions as a function of clutter ratio and keypoint type for double pendulum tracking.

![](images/b9637a5e277dd563c4128ea0f67a5a8e429304c9f1f02a458a714ef6a8378518.jpg)  
Figure 4: Average error of DNBP and LSTM predictions as a function of clutter ratio for articulated 'spider' tracking.

ensure fair comparison with the baseline method published by Garcia-Hernando et al. (2018). Similarly, the feature extractor used by DNBP in the following experiments was designed to emulate the feature extractor of compared baseline. Details of network parameters and inspection of learned relationships are included in the Appendices A.2 and A.6.

# 5.3 PERFORMANCE METRICS

As a quantitative measure of tracking error, average Euclidean error is used. On the simulated tasks, Euclidean error is averaged over all images in the test set. On the hand tracking task, Euclidean error is averaged over all joints per frame then used to calculate the percent of frames satisfying variable error thresholds as used by Garcia-Hernando et al. (2018).

Discrete entropy (Shannon, 1948) is used as a quantitative measure of uncertainty estimated by DNBP. Discrete entropy is calculated by binning samples from each marginal belief set. For qualitative analysis of the uncertainty estimated by DNBP, samples from an approximation of the joint posterior distribution (i.e. for collection of all unobserved variables) are formed using a sequential Monte Carlo sampling approach (Naesseth et al., 2014). Visualization of these samples are formed by plotting a rendered link between each pair of keypoint samples.

# 5.4 DOUBLE PENDULUM TRACKING RESULTS

As shown in Fig. 3, the keypoint tracking error of DNBP is directly compared to that of the LSTM baseline on the held-out test set for each keypoint type (base, middle and end effector) across the full range of clutter ratios. Results from this comparison show that DNBP's average keypoint tracking error is comparable to the LSTM's corresponding error for both the mid joint and end effector keypoints, independent of clutter ratio. For the base joint keypoint, which is stationary at the center position of every image, the LSTM was able to memorize the correct position. DNBP, which diffuses particles based on the message passing scheme, does not memorize the base joint position and registers a consistently larger error which increased with clutter ratio.

DNBP provides measures of uncertainty associated with its predictions, which are generated according to the algorithmic prior of belief propagation. Next tested was the hypothesis that the DNBP model would generate increased uncertainty under conditions in which an occluding object is placed into the input images such that it covers portions of the double pendulum. This test was performed by rendering an occluding block onto a test sequence as shown in Fig. 5a-c. Under optimal conditions, in which the pendulum is minimally occluded ( $< 25\%$  by surface area), the model's output indicates a low level of uncertainty (see Fig. 5d,f,g.) for each keypoint and each frame. In contrast, under conditions in which the pendulum is occluded by the superimposed object, the model's output indicates relatively high levels of uncertainty precisely at frames in which the superimposed object occludes a portion ( $>25\%$ ) of the double pendulum (see Fig. 5e,g.). These results demonstrate that the estimate of uncertainty produced by DNBP can identify predictions which are unreliable.

![](images/12459d48d9ce2fdc365ac8623c6dd5fd9b164c889fff3fc9fecfa735adc3529d.jpg)  
Figure 5: Tracking of double pendulum by DNBP under partial occlusion (orange block). Uncertainty associated with predictions is shown as samples from the joint distribution in pink and blue (d,e,f). (g) Marginal entropy for each keypoint across test sequence; base keypoint (red), middle keypoint (green), end-effector keypoint (blue). Sequence steps highlighted by gray correspond to images in which  $>25\%$  of the pendulum is occluded.

![](images/814d4d5966024ef27020a4db21c46989cfeb3adac4cc0f717f7f158157bc517f.jpg)  
Figure 6: Comparison of articulated 'spider' tracking by LSTM (d,e,f) and DNBP (g,h,i) under cluttered conditions. Predicted and ground truth keypoints shown as yellow circles. Clutter shown as faded shapes for illustration to highlight predictions.

# 5.5 ARTICULATED SPIDER TRACKING RESULTS

After having established the basic performance characteristics of DNBP on the relatively straightforward double pendulum task, we next set out to determine DNBP's capability for tracking more complex articulated structures. To this end, the 3-arm spider structure was used as a more challenging articulated pose tracking task. Each model's performance was quantitatively assessed on the held-out test set of the articulated spider tracking task using the same approach as described for the double pendulum experiment by varying clutter ratio (Fig. 4). Similar to the results of the double pendulum experiment, average error on the spider task increases as a function of clutter ratio for both the LSTM and for DNBP. For clutter ratios between 0 and 0.25, average error for both models remains near 6 pixels then increases consistently with clutter ratio, reaching above 30 pixels of average error for clutter ratios above 0.85. As in the case of the double pendulum experiment, these results demonstrate comparable performance between LSTM and DNBP on an articulated pose tracking task.

Next, a qualitative example of tracking performance under conditions of clutter is shown in Fig. 6. In Fig. 6(a-c), the ground truth spider pose is shown amidst distracting shapes across selected frames of a test sequence with clutter ratio of 0.25. Pose predictions generated by LSTM are shown in Fig. 6(d-f) and by DNBP in (g-i). Qualitative assessment of the images shown indicates that both the LSTM and DNBP place their spider predictions in the correct region of the image. Additionally, each model is shown to correctly predict the relative positions of the three spider arms. Over the sequence, both models track the prismatic and rotational motion of each keypoint, however appear to struggle with certain keypoint predictions.

# 5.6 HUMAN HAND TRACKING RESULTS

To evaluate DNBP's capability for application to real-world tasks, the algorithm's state estimation and tracking performance was evaluated on the FPHAB dataset. This is a challenging dataset with extreme occlusions where complete observations of all the finger joints are rare. Firstly, Euclidean error between the estimated pose and the ground truth pose is measured for every frame in the test

![](images/5fc10cab224efbc164138ba11021f09f59684b00e45e35ab88add719b3dba176.jpg)  
Figure 7: Output from DNP on randomly sampled frames. See Appendix A.7 for more examples.

![](images/2fe4c24e0be8e6debe3ea85105b9a82a6e045e07e14b5656eafdb7c9ac8f2628.jpg)  
Figure 8: Quantitative comparison between DNBP and neural network baseline on hand pose tracking task of the FPHAB dataset. For each model the percent of frames with predicted pose less than a set threshold is calculated as the threshold is varied from  $0\mathrm{mm}$  to  $80\mathrm{mm}$ .

set. For this evaluation, DNBP is applied as a frame-by-frame state estimator with no tracking of its estimates over time. The quantitative results from this experiment, in terms of estimated pose accuracy, are included in Fig. 8 with direct comparison to a pure neural network baseline. The results from this experiment indicate that for error thresholds below  $50\mathrm{mm}$ , DNBP will consistently have an accuracy of  $95\%$  and above.

Following the comparison against a state of the art baseline, it was hypothesized that DNBP when applied as a tracking method would out perform its reported frame-by-frame error performance. To perform this test, DNBP was applied sequentially to each video sequence in the test set and evaluated under the same error metric. The result from this test, as shown in Fig. 8, demonstrates that DNBP does improve in terms of frame error when allowed to track its uncertainty over time, thereby agreeing with the initial hypothesis. Qualitative examples (with randomly chosen frames from a set of sequences) showing DNBP's tracking performance are shown in Fig. 11 and Appendix A.7. The tracking videos showing the DNBP's estimates and belief are included in the supplementary material and project webpage: https://sites.google.com/view/diff-nbp.

# 6 CONCLUSION

In this work, we proposed a novel formulation of belief propagation which is differentiable and uses a nonparametric representation of belief. It was hypothesized that combining maximum likelihood estimation with the nonparametric inference approach would enable end-to-end learning of the probabilistic factors needed for inference. The hypothesis was tested on both qualitative and quantitative experiments. Results demonstrate successful application of this approach. The current approach is limited by its use of non-differentiable resampling and its demand for a graph model as input. Exploration of resampling approximations and methods to learn the graphical model's factorization are left as future work.

# REFERENCES

Peter Anderson, Ayush Shrivastava, Devi Parikh, Dhruv Batra, and Stefan Lee. Chasing ghosts: Instruction following as Bayesian state tracking. In Advances in Neural Information Processing Systems, pp. 369-379, 2019.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym, 2016.  
Alex Clark. Pillow (pil fork) documentation, 2015.  
Karthik Desingh, Shiyang Lu, Anthony Opipari, and Odest Chadwicke Jenkins. Efficient nonparametric belief propagation for pose estimation and manipulation of articulated objects. Science Robotics, 4(30), 2019. doi: 10.1126/scirobotics.aaw4523.  
Trinh Minh Tri Do and Thierry Artières. Neural conditional random fields. In Yee Whye Teh and D. Mike Titterington (eds.), Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, AISTATS 2010, Chia Laguna Resort, Sardinia, Italy, May 13-15, 2010, volume 9 of JMLR Proceedings, pp. 177-184. JMLR.org, 2010.  
Guillermo Garcia-Hernando, Shanxin Yuan, Seungryul Baek, and Tae-Kyun Kim. First-person hand action benchmark with rgb-d videos and 3d hand pose annotations. In Proceedings of Computer Vision and Pattern Recognition (CVPR), 2018.  
S. Godsill. Particle filtering: the first 25 years and beyond. In International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 7760-7764. IEEE, 2019. doi: 10.1109/ICASSP.2019.8683411.  
Tuomas Haarnoja, Anurag Ajay, Sergey Levine, and Pieter Abbeel. Backprop KF: learning discriminative deterministic state estimators. In Advances in Neural Information Processing Systems, pp. 4376-4384, 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural Comput.*, 9(8):1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735.  
Alexander Ihler and David McAllester. Particle belief propagation. In Artificial Intelligence and Statistics, pp. 256-263, 2009.  
Michael Isard. PAMPAS: Real-valued graphical models for computer vision. In Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, 2003.  
Rico Jonschkowski and Oliver Brock. End-to-end learnable histogram filters. In Workshop on Deep Learning for Action and Interaction at NeurIPS, December 2016.  
Rico Jonschkowski, Divyam Rastogi, and Oliver Brock. Differentiable particle filters: End-to-end learning with algorithmic priors. In Robotics: Science and Systems (RSS), 2018. doi: 10.15607/RSS.2018.XIV.001.  
Péter Karkus, David Hsu, and Wee Sun Lee. Particle filter networks with application to visual localization. In Conference on Robot Learning (CoRL), volume 87, pp. 169-178. PMLR, 2018.  
Péter Karkus, Xiao Ma, David Hsu, Leslie Pack Kaelbling, Wee Sun Lee, and Tomás Lozano-Pérez. Differentiable algorithm networks for composable robot learning. In Robotics: Science and Systems, 2019. doi: 10.15607/RSS.2019.XV.039.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun (eds.), 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015.  
Alina Kloss, Georg Martius, and Jeannette Bohg. How to train your differentiable filter. Autonomous Robots, pp. 1-18, 2021.  
Daphne Koller and Nir Friedman. Probabilistic Graphical Models - Principles and Techniques. MIT Press, 2009. ISBN 978-0-262-01319-2.

Xiangyang Lan, Stefan Roth, Daniel P. Huttenlocher, and Michael J. Black. Efficient belief propagation with learned higher-order markov random fields. In European Conference on Computer Vision (ECCV), volume 3952 of Lecture Notes in Computer Science, pp. 269-282. Springer, 2006. doi: 10.1007/11744047\_21.  
Kuang-chih Lee, Dragomir Anguelov, Baris Sumengen, and Salih Burak Gokturk. Markov random field models for hair and face segmentation. In International Conference on Automatic Face and Gesture Recognition (FG 2008), pp. 1-6. IEEE Computer Society, 2008. doi: 10.1109/AFGR.2008.4813431.  
Kevin P. Murphy. Machine learning - a probabilistic perspective. Adaptive computation and machine learning series. MIT Press, 2012. ISBN 0262018020.  
Kevin P. Murphy, Yair Weiss, and Michael I. Jordan. Loopy belief propagation for approximate inference: An empirical study. In Conference on Uncertainty in Artificial Intelligence (UAI), pp. 467-475. Morgan Kaufmann, 1999.  
Christian A. Naesseth, Fredrik Lindsten, and Thomas B. Schön. Sequential Monte Carlo for graphical models. In Advances in Neural Information Processing Systems, pp. 1862-1870, 2014.  
Joseph Ortiz, Talfan Evans, and Andrew J. Davison. A visual introduction to gaussian belief propagation. arXiv preprint arXiv:2107.02308, 2021.  
Jana Pavlasek, Stanley Lewis, Karthik Desingh, and Odest Chadwicke Jenkins. Parts-based articulated object localization in clutter using belief propagation. In International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020.  
Judea Pearl. Chapter 4 - belief updating by network propagation. In Judea Pearl (ed.), Probabilistic Reasoning in Intelligent Systems, pp. 143 - 237. Morgan Kaufmann, San Francisco (CA), 1988. ISBN 978-0-08-051489-5. doi: https://doi.org/10.1016/B978-0-08-051489-5.50010-2.  
Wei Ping and Alexander T. Ihler. Belief propagation in conditional rbms for structured prediction. In Aarti Singh and Xiaojin (Jerry) Zhu (eds.), Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, AISTATS 2017, 20-22 April 2017, Fort Lauderdale, FL, USA, volume 54 of Proceedings of Machine Learning Research, pp. 1141-1149. PMLR, 2017.  
Claude E. Shannon. A mathematical theory of communication. Bell Syst. Tech. J., 27(3):379-423, 1948. doi: 10.1002/j.1538-7305.1948.tb01338.x.  
Leonid Sigal, Sidharth Bhatia, Stefan Roth, Michael J. Black, and Michael Isard. Tracking loose-limbed people. In Computer Vision and Pattern Recognition (CVPR), pp. 421-428. IEEE Computer Society, 2004. doi: 10.1109/CVPR.2004.252.  
Erik B. Sudderth, Alexander T. Ihler, William T. Freeman, and Alan S. Willsky. Nonparametric belief propagation. In Computer Vision and Pattern Recognition (CVPR), pp. 605-612. IEEE Computer Society, 2003. doi: 10.1109/CVPR.2003.1211409.  
Erik B Sudderth, Michael I Mandel, William T Freeman, and Alan S Willsky. Visual hand tracking using nonparametric belief propagation. In IEEE Conference on Computer Vision and Pattern Recognition Workshop (CVPRW'04), pp. 189-189, 2004.  
Jian Sun, Nanning Zheng, and Heung-Yeung Shum. Stereo matching using belief propagation. IEEE Trans. Pattern Anal. Mach. Intell., 25(7):787-800, 2003. doi: 10.1109/TPAMI.2003.1206509.  
Sebastian Thrun, Wolfram Burgard, and Dieter Fox. *Probabilistic Robotics*. MIT Press, 2005. ISBN 978-0-262-20162-9.  
Jonathan Thompson, Arjun Jain, Yann LeCun, and Christoph Bregler. Joint training of a convolutional network and a graphical model for human pose estimation. In Advances in Neural Information Processing Systems, pp. 1799-1807, 2014.  
Jonathan Tremblay, Thang To, Balakumar Sundaralingam, Yu Xiang, Dieter Fox, and Stan Birchfield. Deep object pose estimation for semantic robotic grasping of household objects. In *Conference on Robot Learning (CoRL)*, 2018.

Yunbo Wang, Bo Liu, Jiajun Wu, Yuke Zhu, Simon S. Du, Fei-Fei Li, and Joshua B. Tenenbaum. Dualsmc: Tunneling differentiable filtering and planning under continuous pomdpds. In International Joint Conference on Artificial Intelligence (IJCAI), pp. 4190-4198. ijcai.org, 2020. doi: 10.24963/ijcai.2020/579.  
Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, and Dieter Fox. PoseCNN: A convolutional neural network for 6D object pose estimation in cluttered scenes. In Robotics: Science and Systems (RSS), 2018.  
Fu Xiong, Boshen Zhang, Yang Xiao, Zhiguo Cao, Taidong Yu, Joey Zhou Tianyi, and Junsong Yuan. A2j: Anchor-to-joint regression network for 3d articulated pose estimation from a single depth image. In International Conference on Computer Vision (ICCV), 2019.  
Hao Xiong and Nicholas Ruozzi. General purpose MRF learning with neural network potentials. In Christian Bessiere (ed.), Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI 2020, pp. 2769-2776. ijcai.org, 2020. doi: 10.24963/ijcai.2020/384.  
Qianru Ye, Shanxin Yuan, and Tae-Kyun Kim. Spatial attention deep net with partial pso for hierarchical hybrid hand pose estimation. In ECCV, 2016.  
Brent Yi, Michelle Lee, Alina Kloss, Roberto Martin-Martin, and Jeannette Bohg. Differentiable factor graph optimization for learning smoothers. In International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021.
