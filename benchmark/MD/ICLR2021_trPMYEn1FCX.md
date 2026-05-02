# Generative Model-Enhanced Human Motion Prediction

Anthony Bourached

Department of Neurology  
University College London  
London, UK  
ucabab6@ucl.ac.uk

Ryan-Rhys Griffiths

Department of Physics University of Cambridge Cambridge, UK rrg27@cam.ac.uk

Robert Gray

Department of Neurology  
University College London  
London, UK  
r(gray@ucl.ac.uk

Ashwani Jha

Department of Neurology  
University College London  
London, UK  
ashwani.jha@ucl.ac.uk

Parashkey Nachev

Department of Neurology  
University College London  
London, UK  
p.nachev@ucl.ac.uk

# Abstract

The task of predicting human motion is complicated by the natural heterogeneity and compositionality of actions, necessitating robustness to distributional shifts as far as out-of-distribution (OoD). Here we formulate a new OoD benchmark based on the Human3.6M and CMU motion capture datasets, and introduce a hybrid framework for hardening discriminative architectures to OoD failure by augmenting them with a generative model. When applied to current state-of-the-art discriminative models, we show that the proposed approach improves OoD robustness without sacrificing in-distribution performance, and can facilitate model interpretability. We suggest human motion predictors ought to be constructed with OoD challenges in mind, and provide an extensible general framework for hardening diverse discriminative architectures to extreme distributional shift. The code is available at https://github.com/bouracha/OoDMotion.

# 1 Introduction

Human motion is naturally intelligible as a time-varying graph of connected joints constrained by locomotor anatomy and physiology. Its prediction allows the anticipation of actions with applications across healthcare [Geertsema et al., 2018, Kakar et al., 2005], physical rehabilitation and training [Chang et al., 2012, Webster and Celik, 2014], robotics [Koppula and Saxena, 2013b,a, Gui et al., 2018], navigation [Paden et al., 2016, Alahi et al., 2016, Bhattacharyya et al., 2018, Wang et al., 2019], manufacture [Švec et al., 2014], entertainment [Shirai et al., 2007, Rofougaran et al., 2018, Lau and Chan, 2008], and security [Kim and Paik, 2010, Ma et al., 2018].

The favoured approach to predicting movements over time has been purely inductive, relying on the history of a specific class of movement to predict its future. For example, state space models [Koller and Friedman, 2009] enjoyed early success for simple, common or cyclic motions [Taylor et al., 2007, Sutskever et al., 2009, Lehrmann et al., 2014]. The range, diversity and complexity of human motion has encouraged a shift to more expressive, deep neural network architectures [Fragkiadaki et al., 2015, Butepage et al., 2017, Martinez et al., 2017, Li et al., 2018, Mao et al., 2019, Li et al., 2020b, Cai et al., 2020], but still within a simple inductive framework.

This approach would be adequate were actions both sharply distinct and highly stereotyped. But their complex, compositional nature means that within one category of action the kinematics may

vary substantially, while between two categories they may barely differ. Moreover, few real-world tasks restrict the plausible repertoire to a small number of classes—distinct or otherwise—that could be explicitly learnt. Rather, any action may be drawn from a great diversity of possibilities—both kinematic and teleological—that shape the characteristics of the underlying movements. This has two crucial implications. First, any modelling approach that lacks awareness of the full space of motion possibilities will be vulnerable to poor generalisation and brittle performance in the face of kinematic anomalies. Second, the very notion of In-Distribution (ID) testing becomes moot, for the relations between different actions and their kinematic signatures are plausibly determinable only across the entire domain of action. A test here arguably needs to be Out-of-Distribution (OoD) if it is to be considered a robust test at all.

These considerations are amplified by the nature of real-world applications of kinematic modelling, such as anticipating arbitrary deviations from expected motor behaviour early enough for an automatic intervention to mitigate them. Most urgent in the domain of autonomous driving [Bhattacharyya et al., 2018, Wang et al., 2019], such safety concerns are of the highest importance, and are best addressed within the fundamental modelling framework. Indeed, Amodei et al. [2016] cites the ability to recognize our own ignorance as a safety mechanism that must be a core component in safe AI. Nonetheless, to our knowledge, current predictive models of human kinematics neither quantify OoD performance nor are designed with it in mind. There is therefore a need for two frameworks, applicable across the domain of action modelling: one for hardening a predictive model to anomalous cases, and another for quantifying OoD performance with established benchmark datasets. General frameworks are here desirable in preference to new models, for the field is evolving so rapidly greater impact can be achieved by introducing mechanisms that can be applied to a breadth of candidate architectures, even if they are demonstrated in only a subset. Our approach here is founded on combining a latent variable generative model with a standard predictive model, illustrated with the current state-of-the-art discriminative architecture [Mao et al., 2019, Wei et al., 2020]. Myronenko [2018], take an analogous approach, regularising an encoder-decoder model for brain tumor segmentation on magnetic resonance images by simultaneously modelling the distribution of the data using a variational autoencoder (VAE) [Kingma and Welling, 2013]. Here the aim is to achieve robust performance within a low data regime, which coincides with the demand for OoD generalisation.

In short, our contributions to the problem of achieving robustness to distributional shift in human motion prediction are as follows:

1. We provide a framework to benchmark OoD performance on the most widely used open-source motion capture datasets: Human3.6M [Ionescu et al., 2013], and CMU-Mocap<sup>1</sup>, and evaluate state-of-the-art models on it.  
2. We present a framework for hardening deep feed-forward models to OoD samples. We show that the hardened models are fast to train, and exhibit substantially improved OoD performance with minimal impact on ID performance.

We begin section 2 with a brief review of human motion prediction with deep neural networks, and of OoD generalisation using generative models. In section 3, we define a framework for benchmarking OoD performance using open-source multi-action datasets. We introduce in section 4 the discriminative models that we harden using a generative branch to achieve a state-of-the-art (SOTA) OoD benchmark. We then turn in section 5 to the architecture of the generative model and the overall objective function. Section 6 presents our experiments and results. We conclude in section 7 with a summary of our results, current limitations, and caveats, and future directions for developing robust and reliable OoD performance and a quantifiable awareness of unfamiliar behaviour.

# 2 Related Work

Deep-network based human motion prediction. Historically, sequence-to-sequence prediction using Recurrent Neural Networks (RNNs) have been the de facto standard for human motion prediction [Fragkiadaki et al., 2015, Jain et al., 2016, Martinez et al., 2017, Guo and Choi, 2019, Gopalakrishnan et al., 2019, Li et al., 2020b]. Currently, the SOTA is dominated by feed forward

models [Butepage et al., 2017, Li et al., 2018, Mao et al., 2019, Wei et al., 2020]. These are inherently faster and easier to train than RNNs. The jury is still out, however, on the optimal way to handle temporality for human motion prediction. Meanwhile, recent trends have overwhelmingly shown that graph-based approaches are an effective means to encode the spatial dependencies between joints [Mao et al., 2019, Wei et al., 2020], or sets of joints [Li et al., 2020b]. In this study, we consider the SOTA models that have graph-based approaches with a feed forward mechanism as presented by [Mao et al., 2019], and the subsequent extension which leverages motion attention, Wei et al. [2020]. We show that these may be augmented to improve robustness to OoD samples.

Generative models for Out-of-Distribution prediction and detection. Despite the power of deep neural networks for prediction in complex domains [LeCun et al., 2015], they face several challenges that limits their suitability for safety-critical applications. Amodei et al. [2016] list robustness to distributional shift as one of the five major challenges to AI safety. Deep generative models, have been used extensively for detection of OoD inputs and have been shown to generalise well in such scenarios [Hendrycks and Gimpel, 2016, Liang et al., 2017, Hendrycks et al., 2018]. While recent work has showed some failures in simple OoD detection using density estimates from deep generative models [Nalisnick et al., 2018, Daxberger and Hernandez-Lobato, 2019], they remain a prime candidate for anomaly detection [Kendall and Gal, 2017, Grathwohl et al., 2019, Daxberger and Hernandez-Lobato, 2019].

Myronenko [2018] use a Variational Autoencoder (VAE) [Kingma and Welling, 2013] to regularise an encoder-decoder architecture with the specific aim of better generalisation. By simultaneously using the encoder as the recognition model of the VAE, the model is encouraged to base its segmentations on a complete picture of the data, rather than on a reductive representation that is more likely to be fitted to the training data. Furthermore, the original loss and the VAE's loss are combined as a weighted sum such that the discriminator's objective still dominates. Further work may also reveal useful interpretability of behaviour (via visualisation of the latent space as in Bourached and Nachev [2019]), generation of novel motion [Motegi et al., 2018], or reconstruction of missing joints as in Chen et al. [2015].

# 3 Quantifying out-of-distribution performance of human motion predictors

Even a very compact representation of the human body such as OpenPose's 17 joint parameterisation Cao et al. [2018] explodes to unmanageable complexity when a temporal dimension is introduced of the scale and granularity necessary to distinguish between different kinds of action: typically many seconds, sampled at hundredths of a second. Moreover, though there are anatomical and physiological constraints on the space of licit joint configurations, and their trajectories, the repertoire of possibility remains vast, and the kinematic demarcations of teleologically different actions remain indistinct. Thus, no practically obtainable dataset may realistically represent the possible distance between instances. To simulate OoD data we first need ID data that is as small in quantity, and narrow in domain as possible. For this reason we propose to define OoD on multi-action motion capture datasets as being the scenario where only a single action, the smallest labelled subset, is available for training and hyperparameter search. In appendix A, to show that the motion categories we have chosen can actually be distinguished at the time scales on which our trajectories are encoded we train a simple classifier and show that it can separate the selected ID action from the others with high accuracy (100% precision and recall for the CMU dataset). In this way OoD performance may be considered over the remaining set of actions.

# 4 Background

Here we describe the current SOTA model proposed by Mao et al. [2019] (GCN). We then describe the extension by Wei et al. [2020] (attention-GCN) which antecodes the GCN prediction model with motion attention.

# 4.1 Problem Formulation

We are given a motion sequence  $\mathbf{X}_{1:N} = (\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, \dots, \mathbf{x}_N)$  consisting of  $N$  consecutive human poses, where  $\mathbf{x}_i \in \mathbb{R}^K$ , with  $K$  the number of parameters describing each pose. The goal is to predict the poses  $\mathbf{X}_{N+1:N+T}$  for the subsequent  $T$  time steps.

# 4.2 DCT-based Temporal Encoding

The input is transformed using Discrete Cosine Transformations (DCT). In this way each resulting coefficient encodes information of the entire sequence at a particular temporal frequency. Furthermore, the option to remove high or low frequencies is provided. Given a joint,  $k$ , the position of  $k$  over  $N$  time steps is given by the trajectory vector:  $\mathbf{x}_k = [x_{k,1},\dots,x_{k,N}]$  where we convert to a DCT vector of the form:  $\mathbf{C}_k = [C_{k,1},\dots,C_{k,N}]$  where  $C_{k,l}$  represents the  $l$ th DCT coefficient. For  $\delta_{l1}\in \mathbb{R}^N = [1,0,\dots ,0]$ , these coefficients may be computed as

$$
C _ {k, l} = \sqrt {\frac {2}{N}} \sum_ {n = 1} ^ {N} x _ {k, n} \frac {1}{\sqrt {1 + \delta_ {l 1}}} \cos \left(\frac {\pi}{2 N} (2 n - 1) (l - 1)\right). \tag {1}
$$

If no frequencies are cropped, the DCT is invertible via the Inverse Discrete Cosine Transform (IDCT):

$$
x _ {k, l} = \sqrt {\frac {2}{N}} \sum_ {l = 1} ^ {N} C _ {k, l} \frac {1}{\sqrt {1 + \delta_ {l 1}}} \cos \left(\frac {\pi}{2 N} (2 n - 1) (l - 1)\right). \tag {2}
$$

Mao et al. use the DCT transform with a graph convolutional network architecture to predict the output sequence. This is achieved by having an equal length input-output sequence, where the input is the DCT transformation of  $\mathbf{x_{k}} = [x_{k,1},\ldots ,x_{k,N},x_{k,N + 1},\ldots ,x_{k,N + T}]$ , here  $[x_{k,1},\dots,x_{k,N}]$  is the observed sequence and  $[x_{k,N + 1},\dots,x_{k,N + T}]$  are replicas of  $x_{k,N}$  (ie  $x_{k,n} = x_{k,N}$  for  $n\geq N$ ). The target is now simply the ground truth  $\mathbf{x_{k}}$ .

# 4.3 Graph Convolutional Network

Suppose  $\mathbf{C} \in \mathbb{R}^{K \times (N + T)}$  is defined on a graph with  $k$  nodes and  $N + T$  dimensions, then we define a graph convolutional network to respect this structure. First we define a Graph Convolutional Layer (GCL) that, as input, takes the activation of the previous layer  $(\mathbf{A}^{[1 - 1]})$ , where  $l$  is the current layer.

$$
G C L (\mathbf {A} ^ {[ 1 - \mathbf {1} ]}) = \mathbf {S A} ^ {[ 1 - \mathbf {1} ]} \mathbf {W} + \mathbf {b} \tag {3}
$$

where  $\mathbf{A}^{[0]} = \mathbf{C} \in \mathbb{R}^{K \times (N + T)}$ , and  $\mathbf{S} \in \mathbb{R}^{K \times K}$  is a layer-specific learnable normalised graph Laplacian that represents connections between joints,  $\mathbf{W} \in \mathbb{R}^{n^{[l - 1]} \times n^{[l]}}$  are the learnable inter-layer weightings and  $\mathbf{b} \in \mathbb{R}^{n^{[l]}}$  are the learnable biases where  $n^{[l]}$  are the number of hidden units in layer  $l$ .

# 4.4 Network Structure and Loss

The network consists of 12 Graph Convolutional Blocks (GCBs), each containing 2 GCLs with skip (or residual) connections, see figure 5. Additionally, there is one GCL at the beginning of the network, and one at the end.  $n^{[l]} = 256$ , for each layer,  $l$ . There is one final skip connection from the DCT inputs to the DCT outputs, which greatly reduces train time. The model has around 2.6M parameters. Hyperbolic tangent functions are used as the activation function. Batch normalisation is applied before each activation.

The outputs are converted back to their original coordinate system using the IDCT (equation 2) to be compared to the ground truth. The loss used for joint angles is the average  $l_{1}$  distance between the ground-truth joint angles, and the predicted ones. Thus, the joint angle loss is:

$$
\ell_ {a} = \frac {1}{K (N + T)} \sum_ {n = 1} ^ {N + T} \sum_ {k = 1} ^ {K} | \hat {x} _ {k, n} - x _ {k, n} | \tag {4}
$$

where  $\hat{x}_{k,n}$  is the predicted  $k^{th}$  joint at timestep  $n$  and  $x_{k,n}$  is the corresponding ground truth.

This is separately trained on 3D joint coordinate prediction making use of the Mean Per Joint Position Error (MPJPE), as proposed in Ionescu et al. [2013] and used in Mao et al. [2019], Wei et al. [2020]. This is defined, for each training example, as

$$
\ell_ {m} = \frac {1}{J (N + T)} \sum_ {n = 1} ^ {N + T} \sum_ {j = 1} ^ {J} \| \hat {\mathbf {p}} _ {j, n} - \mathbf {p} _ {j, n} \| ^ {2} \tag {5}
$$

where  $\hat{\mathbf{p}}_{j,n} \in \mathbb{R}^3$  denotes the predicted jth joint position in frame  $n$ . And  $\mathbf{p}_{j,n}$  is the corresponding ground truth, while  $\mathbf{J}$  is the number of joints in the skeleton.

![](images/f5dce1e22aed5f6df6d92b64a9962db059ac4f3a0cbcf9cdc54df96706e0ea8a.jpg)  
Figure 1: GCN network architecture with VGAE branch. Here  $n_z = 16$  is the number of latent variables per joint.

# 4.5 Motion attention extension

Wei et al. [2020] extend this model by summing multiple DCT transformations from different sections of the motion history with weightings learned via an attention mechanism. For this extension, the above model (the GCN) along with the anteceding motion attention is trained end-to-end. We refer to this as the attention-GCN.

# 5 Our Approach

Myronenko [2018] augment an encoder-decoder discriminative model by using the encoder as a recognition model for a Variational Autoencoder (VAE), [Kingma and Welling, 2013, Rezende et al., 2014]. Myronenko [2018] show this to be a very effective regulariser. Here, for conjugacy with the discriminator, we consider the Variational Graph Autoencoder (VGAE), proposed by Kipf and Welling [2016] as a framework for unsupervised learning on graph-structured data.

The generative model sets a precedence for information that can be modelled causally, while leaving elements of the discriminative machinery, such as skip connections, to capture correlations that remain useful for prediction but are not necessarily persuant to the objective of the generative model. In addition to performing the role of regularisation in general, we show that we gain robustness to distributional shift across similar, but different, actions that are likely to share generative properties. The architecture may be considered with the visual aid in figure 1.

# 5.1 Variational Graph Autoencoder (VGAE) Branch and Loss

Here we define the first 6 GCB blocks as our VGAE recognition model, with a latent variable  $\mathbf{z} \in \mathbb{R}^{K \times n_z} = N(\mu_{\mathbf{z}}, \sigma_{\mathbf{z}})$ , where  $\mu_{\mathbf{z}} \in \mathbb{R}^{K \times n_z}$ ,  $\sigma_{\mathbf{z}} \in \mathbb{R}^{K \times n_z}$ .  $n_z = 8$ , or 32 depending on training stability.

The KL divergence between the latent space distribution and a spherical Gaussian  $N(0,\mathbf{I})$  is given by:

$$
\ell_ {l} = K L (q (\mathbf {Z} | \mathbf {C}) | | q (\mathbf {Z})) = \frac {1}{2} \sum_ {1} ^ {n _ {z}} \left(\mu_ {\mathbf {z}} ^ {2} + \sigma_ {\mathbf {z}} ^ {2} - 1 - \log \left(\left(\sigma_ {\mathbf {z}}\right) ^ {2}\right)\right). \tag {6}
$$

The decoder part of the VGAE has the same structure as the discriminative branch; 6 GCBs. We parametrise the output neurons as  $\mu \in \mathbb{R}^{K\times (N + T)}$ , and  $log(\sigma^2)\in \mathbb{R}^{K\times (N + T)}$ . We can now model the reconstruction of inputs as samples of a maximum likelihood of a Gaussian distribution which constitutes the second term of the negative Variational Lower Bound (VLB) of the VGAE:

$$
\ell_ {G} = \log (p (\mathbf {C} | \mathbf {Z})) = - \frac {1}{2} \sum_ {n = 1} ^ {N + T} \sum_ {l = 1} ^ {K} \left(\log \left(\sigma_ {k, l} ^ {2}\right) + \log (2 \pi) + \frac {\left| C _ {k , l} - \mu_ {k , l} \right| ^ {2}}{e ^ {\log \left(\sigma_ {k , l} ^ {2}\right)}}\right), \tag {7}
$$

where  $C_{k,l}$  are the DCT coefficients of the ground truth.

# 5.2 Training

We train the entire network together with the additional of the negative VLB:

$$
\ell = \underbrace {\frac {1}{(N + T) K} \sum_ {n = 1} ^ {N + T} \sum_ {k = 1} ^ {K} \left| \hat {x} _ {k , n} - x _ {k , n} \right|} _ {\text {D i s c r i m i n i t e l o s s}} - \lambda \underbrace {\left(\ell_ {G} - \ell_ {l}\right)} _ {\text {V L B}}. \tag {8}
$$

Here  $\lambda$  is a hyperparameter of the model. The overall network is  $\approx 3.4M$  parameters. The number of parameters varies slightly as per the number of joints, K, since this is reflected in the size of the graph in each layer ( $k = 48$  for H3.6M,  $K = 64$  for CMU joint angles, and  $K = J = 75$  for CMU Cartesian coordinates). Furthermore, once trained, the generative model is not required for prediction and hence for this purpose is as compact as the original models.

# 6 Experiments

# 6.1 Datasets and Experimental Setup

Human3.6M (H3.6M) The H3.6M dataset [Ionescu et al., 2011, 2013], so called as it contains a selection of 3.6 million 3D human poses and corresponding images, consists of seven actors each performing 15 actions, such as walking, eating, discussion, sitting, and talking on the phone. Martinez et al. [2017], Mao et al. [2019], Li et al. [2020b] all follow the same training and evaluation procedure: training their motion prediction model on 6 (5 for train and 1 for cross-validation) of the actors, for each action, and evaluate metrics on the final actor, subject 5. For easy comparison to these ID baselines, we maintain the same train; cross-validation; and test splits. However, we use the single, most well-defined action (see appendix A), walking, for train and cross-validation, and we report test error on all the remaining actions from subject 5. In this way we conduct all parameter selection based on ID performance.

CMU motion capture (CMU-mocap) The CMU dataset consists of 5 general classes of actions. Similarly to [Li et al., 2018, 2020a, Mao et al., 2019] we use 8 detailed actions from these classes: 'basketball', 'basketball signal', 'directing traffic', 'jumping', 'running', 'soccer', 'walking', and 'window washing'. We use two representations, a 64-dimensional vector that gives an exponential map representation [Grassia, 1998] of the joint angle, and a 75-dimensional vector that gives the 3D Cartesian coordinates of 25 joints. We do not tune any hyperparameters on this dataset and use only a train and test set with the same split as is common in the literature [Martinez et al., 2017, Mao et al., 2019].

Model configuration We implemented the model in PyTorch [Paszke et al., 2017] using the ADAM optimiser [Kingma and Ba, 2014]. The learning rate was set to 0.0005 for all experiments where, unlike Mao et al. [2019], Wei et al. [2020], we did not decay the learning rate as it was hypothesised that the dynamic relationship between the discriminative and generative loss would make this redundant. The batch size was 16. For numerical stability, gradients were clipped to a maximum  $\ell 2$ -norm of 1 and  $log(\hat{\sigma}^2)$  and values were clamped between -20 and 3. Code for all experiments is available at the following link: https://github.com/bouracha/OoDMotion

<table><tr><td></td><td colspan="4">Walking (ID)</td><td colspan="4">Eating (OoD)</td><td colspan="4">Smoking (OoD)</td><td colspan="4">Discussion (OoD)</td></tr><tr><td>milliseconds</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td></tr><tr><td>GCN (OoD)</td><td>0.22</td><td>0.38</td><td>0.61</td><td>0.66</td><td>0.22</td><td>0.40</td><td>0.67</td><td>0.81</td><td>0.31</td><td>0.62</td><td>1.22</td><td>1.25</td><td>0.30</td><td>0.67</td><td>1.00</td><td>1.08</td></tr><tr><td>ours (OoD)</td><td>0.23</td><td>0.37</td><td>0.58</td><td>0.63</td><td>0.21</td><td>0.37</td><td>0.59</td><td>0.72</td><td>0.27</td><td>0.54</td><td>1.03</td><td>1.03</td><td>0.30</td><td>0.66</td><td>0.94</td><td>1.02</td></tr><tr><td></td><td colspan="4">Directions (OoD)</td><td colspan="4">Greeting (OoD)</td><td colspan="4">Phoning (OoD)</td><td colspan="4">Posing (OoD)</td></tr><tr><td>milliseconds</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td></tr><tr><td>GCN (OoD)</td><td>0.38</td><td>0.58</td><td>0.81</td><td>0.90</td><td>0.48</td><td>0.82</td><td>1.28</td><td>1.47</td><td>0.58</td><td>1.12</td><td>1.52</td><td>1.66</td><td>0.30</td><td>0.64</td><td>1.37</td><td>1.68</td></tr><tr><td>ours (OoD)</td><td>0.38</td><td>0.58</td><td>0.79</td><td>0.90</td><td>0.49</td><td>0.81</td><td>1.24</td><td>1.43</td><td>0.57</td><td>1.10</td><td>1.48</td><td>1.61</td><td>0.26</td><td>0.56</td><td>1.26</td><td>1.55</td></tr><tr><td></td><td colspan="4">Purchases (OoD)</td><td colspan="4">Sitting (OoD)</td><td colspan="4">Sitting Down (OoD)</td><td colspan="4">Taking Photo (OoD)</td></tr><tr><td>milliseconds</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td></tr><tr><td>GCN (OoD)</td><td>0.62</td><td>0.90</td><td>1.34</td><td>1.42</td><td>0.40</td><td>0.66</td><td>1.15</td><td>1.33</td><td>0.46</td><td>0.94</td><td>1.52</td><td>1.69</td><td>0.26</td><td>0.53</td><td>0.82</td><td>0.93</td></tr><tr><td>ours (OoD)</td><td>0.61</td><td>0.89</td><td>1.27</td><td>1.37</td><td>0.38</td><td>0.62</td><td>1.06</td><td>1.22</td><td>0.41</td><td>0.83</td><td>1.28</td><td>1.41</td><td>0.25</td><td>0.51</td><td>0.81</td><td>0.95</td></tr><tr><td></td><td colspan="4">Waiting (OoD)</td><td colspan="4">Walking Dog (OoD)</td><td colspan="4">Walking Together (OoD)</td><td colspan="4">Average (of 14 for OoD)</td></tr><tr><td>milliseconds</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td><td>80</td><td>160</td><td>320</td><td>400</td></tr><tr><td>GCN (OoD)</td><td>0.30</td><td>0.61</td><td>1.10</td><td>1.34</td><td>0.51</td><td>0.85</td><td>1.16</td><td>1.32</td><td>0.20</td><td>0.42</td><td>0.65</td><td>0.69</td><td>0.38</td><td>0.70</td><td>1.12</td><td>1.26</td></tr><tr><td>ours (OoD)</td><td>0.29</td><td>0.58</td><td>1.06</td><td>1.29</td><td>0.52</td><td>0.88</td><td>1.17</td><td>1.34</td><td>0.21</td><td>0.44</td><td>0.66</td><td>0.74</td><td>0.37</td><td>0.63</td><td>1.08</td><td>1.18</td></tr></table>

Table 1: Short-term prediction of Eucillean distance between predicted and ground truth joint angles on H3.6M.

<table><tr><td></td><td colspan="2">Walking</td><td colspan="2">Eating</td><td colspan="2">Smoking</td><td colspan="2">Discussion</td><td colspan="2">Average</td></tr><tr><td>milliseconds</td><td>560</td><td>1000</td><td>560</td><td>1000</td><td>560</td><td>1000</td><td>560</td><td>1000</td><td>560</td><td>1000</td></tr><tr><td>GCN (OoD)</td><td>0.80</td><td>0.80</td><td>0.89</td><td>1.20</td><td>1.26</td><td>1.85</td><td>1.45</td><td>1.88</td><td>1.10</td><td>1.43</td></tr><tr><td>ours (OoD)</td><td>0.66</td><td>0.72</td><td>0.90</td><td>1.19</td><td>1.17</td><td>1.78</td><td>1.44</td><td>1.90</td><td>1.04</td><td>1.40</td></tr></table>

Table 2: Long-term prediction of Eucillean distance between predicted and ground truth joint angles on H3.6M.

Baseline comparison Both Mao et al. [2019] (GCN), and Wei et al. [2020] (attention-GCN) use this same Graph Convolutional Network (GCN) architecture with DCT inputs. In particular, Wei et al. [2020] increase the amount of history accounted for by the GCN by adding a motion attention mechanism to weight the DCT coefficients from different sections of the history prior to being inputted to the GCN. We compare against both of these baselines on OoD actions. For attention-GCN we leave the attention mechanism preceding the GCN unchanged such that the generative branch of the model is reconstructing the weighted DCT inputs to the GCN, and the whole network is end-to-end differentiable.

Hyperparameter search Since a new term has been introduced to the loss function, it was necessary to determine a sensible weighting between the discriminative and generative models. In Myronenko [2018], this weighting was arbitrarily set to 0.1. It is natural that the optimum value here will relate to the other regularisation parameters in the model. Thus, we conducted random hyperparameter search for  $p_{drop}$  and  $\lambda$  in the ranges  $p_{drop} = [0, 0.5]$  on a linear scale, and  $\lambda = [10, 0.00001]$  on a logarithmic scale. For fair comparison we also conducted hyperparameter search on GCN, for values of the dropout probability  $(p_{drop})$  between 0.1 and 0.9. For each model, 25 experiments were run and the optimum values were selected on the lowest ID validation error. The hyperparameter search was conducted only for the GCN model on short-term predictions for the H3.6M dataset and used for all future experiments hence demonstrating generalisability of the architecture.

# 6.2 Results

Consistent with the literature we report short-term  $(< 500ms)$  and long-term  $(>500ms)$  predictions. In comparison to GCN, we take short term history into account (10 frames,  $400ms$ ) for both datasets to predict both short- and long-term motion. In comparison to attention-GCN, we take long term history (50 frames, 2 seconds) to predict the next 10 frames, and predict further into the future by

<table><tr><td>millisecond</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td></td><td></td></tr><tr><td>GCN (OoD)</td><td>0.40</td><td>0.67</td><td>1.11</td><td>1.25</td><td>1.63</td><td>0.27</td><td>0.55</td><td>1.14</td><td>1.42</td><td>2.18</td><td>0.31</td><td>0.62</td><td>1.05</td><td>1.24</td><td>2.49</td><td>0.42</td><td>0.73</td><td>1.72</td><td>1.98</td><td>2.66</td><td></td><td></td></tr><tr><td>ours (OoD)</td><td>0.40</td><td>0.66</td><td>1.12</td><td>1.29</td><td>1.76</td><td>0.28</td><td>0.37</td><td>1.15</td><td>1.43</td><td>2.07</td><td>0.28</td><td>0.56</td><td>0.96</td><td>1.10</td><td>2.33</td><td>0.38</td><td>0.72</td><td>1.74</td><td>2.03</td><td>2.70</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>Soccer</td><td></td><td></td><td></td><td></td><td></td><td>Walking</td><td></td><td></td><td></td><td>Washing window</td><td></td><td></td><td></td><td></td><td colspan="4">Average (of 7 for OoD)</td></tr><tr><td>milliseconds</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td></td><td></td></tr><tr><td>GCN (OoD)</td><td>0.29</td><td>0.54</td><td>1.15</td><td>1.41</td><td>2.14</td><td>0.40</td><td>0.61</td><td>0.97</td><td>1.18</td><td>1.85</td><td>0.36</td><td>0.65</td><td>1.23</td><td>1.51</td><td>2.31</td><td>0.36</td><td>0.65</td><td>1.41</td><td>1.49</td><td>2.17</td><td></td><td></td></tr><tr><td>ours (OoD)</td><td>0.28</td><td>0.53</td><td>1.07</td><td>1.27</td><td>1.99</td><td>0.38</td><td>0.54</td><td>0.82</td><td>0.99</td><td>1.27</td><td>0.35</td><td>0.63</td><td>1.20</td><td>1.51</td><td>2.26</td><td>0.34</td><td>0.62</td><td>1.35</td><td>1.41</td><td>2.10</td><td></td><td></td></tr></table>

Table 3: Eucildean distance between predicted and ground truth joint angles on CMU.  

<table><tr><td>milliseconds</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GCN (OoD)</td><td>15.7</td><td>28.9</td><td>54.1</td><td>65.4</td><td>108.4</td><td>14.4</td><td>30.4</td><td>63.5</td><td>78.7</td><td>114.8</td><td>18.5</td><td>37.4</td><td>75.6</td><td>93.6</td><td>210.7</td><td>24.6</td><td>51.2</td><td>111.4</td><td>139.6</td><td>219.7</td><td>32.3</td><td>54.8</td><td>85.9</td><td>99.3</td><td>99.9</td></tr><tr><td>ours (OoD)</td><td>16.0</td><td>30.0</td><td>54.5</td><td>65.5</td><td>98.1</td><td>12.8</td><td>26.0</td><td>53.7</td><td>67.6</td><td>103.2</td><td>18.3</td><td>37.2</td><td>75.7</td><td>93.8</td><td>199.6</td><td>25.0</td><td>52.0</td><td>110.3</td><td>136.8</td><td>200.2</td><td>29.8</td><td>50.2</td><td>83.5</td><td>98.7</td><td>107.3</td></tr><tr><td></td><td colspan="6">Soccer</td><td colspan="7">Walking</td><td colspan="6">Washing window</td><td colspan="6">Average of 7 for (OoD)</td></tr><tr><td>milliseconds</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td>80</td><td>160</td><td>320</td><td>400</td><td>1000</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GCN (OoD)</td><td>22.6</td><td>46.6</td><td>92.8</td><td>114.3</td><td>192.5</td><td>10.8</td><td>20.7</td><td>42.9</td><td>53.4</td><td>86.5</td><td>17.1</td><td>36.4</td><td>77.6</td><td>96.0</td><td>151.6</td><td>20.0</td><td>43.8</td><td>86.3</td><td>105.8</td><td>169.2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ours (OoD)</td><td>21.1</td><td>44.2</td><td>90.4</td><td>112.1</td><td>202.0</td><td>10.5</td><td>18.9</td><td>39.2</td><td>48.6</td><td>72.2</td><td>17.6</td><td>37.3</td><td>82.0</td><td>103.4</td><td>167.5</td><td>21.6</td><td>42.3</td><td>84.2</td><td>103.8</td><td>164.3</td><td></td><td></td><td></td><td></td><td></td></tr></table>

recursively applying the predictions as input to the model as in Wei et al. [2020]. In this way a single short term prediction model may produce long term predictions.

We use Euclidean distance between the predicted and ground-truth joint angles for the Euler angle representation. For 3D joint coordinate representation we use the MPJPE as used for training (equation 5). Table 1 reports the joint angle error for the short term predictions on the H3.6M dataset. Here we found the optimum hyperparameters to be  $p_{drop} = 0.5$  for GCN, and  $\lambda = 0.003$ , with  $p_{drop} = 0.3$  for our augmentation of GCN. The latter of which was used for all future experiments, where for our augmentation of attention-GCN we removed dropout altogether. On average, our model performs convincingly better both ID and OoD. Here the generative branch works well as both a regulariser for small datasets and by creating robustness to distributional shifts. We see similar and consistent results for long-term predictions in table 2.

From tables 3 and 4, we can see that the superior OoD performance generalises to the CMU dataset with the same hyperparameter settings with a similar trend of the difference being larger for longer predictions for both joint angles and 3D joint coordinates. For each of these experiments  $n_z = 8$

Table 5, shows that the effectiveness of the generative branch generalises to the very recent motion attention architecture. For attention-GCN we used  $n_z = 32$ . Here, interestingly short term predictions are poor but long term predictions are consistently better. This supports our assertion that information relevant to generative mechanisms are more intrinsic to the causal model and thus, here, when the predicted output is recursively used, more useful information is available for the future predictions.

Table 4: Mean Joint Per Position Error (MPJPE) between predicted and ground truth 3D Cartesian coordinates of joints on CMU.  

<table><tr><td></td><td colspan="4">Walking (ID)</td><td colspan="4">Eating (OoD)</td><td colspan="4">Smoking (OoD)</td><td colspan="4">Discussion (OoD)</td></tr><tr><td>milliseconds</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td></tr><tr><td>attention-GCN (OoD)</td><td>55.4</td><td>60.5</td><td>65.2</td><td>68.7</td><td>87.6</td><td>103.6</td><td>113.2</td><td>120.3</td><td>81.7</td><td>93.7</td><td>102.9</td><td>108.7</td><td>114.6</td><td>130.0</td><td>133.5</td><td>136.3</td></tr><tr><td>ours (OoD)</td><td>58.7</td><td>60.6</td><td>65.5</td><td>69.1</td><td>81.7</td><td>94.4</td><td>102.7</td><td>109.3</td><td>80.6</td><td>89.9</td><td>99.2</td><td>104.1</td><td>115.4</td><td>129.0</td><td>134.5</td><td>139.4</td></tr><tr><td></td><td colspan="4">Directions (OoD)</td><td colspan="4">Greeting (OoD)</td><td colspan="4">Phoning (OoD)</td><td colspan="4">Posing (OoD)</td></tr><tr><td>milliseconds</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td></tr><tr><td>attention-GCN (OoD)</td><td>107.0</td><td>123.6</td><td>132.7</td><td>138.4</td><td>127.4</td><td>142.0</td><td>153.4</td><td>158.6</td><td>98.7</td><td>117.3</td><td>129.9</td><td>138.4</td><td>151.0</td><td>176.0</td><td>189.4</td><td>199.6</td></tr><tr><td>ours (OoD)</td><td>107.1</td><td>120.6</td><td>129.2</td><td>136.6</td><td>128.0</td><td>140.3</td><td>150.8</td><td>155.7</td><td>95.8</td><td>111.0</td><td>122.7</td><td>131.4</td><td>158.7</td><td>181.3</td><td>194.4</td><td>203.4</td></tr><tr><td></td><td colspan="4">Purchases (OoD)</td><td colspan="4">Sitting (OoD)</td><td colspan="4">Sitting Down (OoD)</td><td colspan="4">Taking Photo (OoD)</td></tr><tr><td>milliseconds</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td></tr><tr><td>attention-GCN (OoD)</td><td>126.6</td><td>144.0</td><td>154.3</td><td>162.1</td><td>118.3</td><td>141.1</td><td>154.6</td><td>164.0</td><td>136.8</td><td>162.3</td><td>177.7</td><td>189.9</td><td>113.7</td><td>137.2</td><td>149.7</td><td>159.9</td></tr><tr><td>ours (OoD)</td><td>128.0</td><td>143.2</td><td>154.7</td><td>164.3</td><td>118.4</td><td>137.7</td><td>149.7</td><td>157.5</td><td>136.8</td><td>157.6</td><td>170.8</td><td>180.4</td><td>116.3</td><td>134.5</td><td>145.6</td><td>155.4</td></tr><tr><td></td><td colspan="4">Waiting (OoD)</td><td colspan="4">Walking Dog (OoD)</td><td colspan="4">Walking Together (OoD)</td><td colspan="4">Average (of 14 for OoD)</td></tr><tr><td>milliseconds</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td><td>560</td><td>720</td><td>880</td><td>1000</td></tr><tr><td>attention-GCN (OoD)</td><td>109.9</td><td>125.1</td><td>135.3</td><td>141.2</td><td>131.3</td><td>146.9</td><td>161.1</td><td>171.4</td><td>64.5</td><td>71.1</td><td>76.8</td><td>80.8</td><td>112.1</td><td>129.6</td><td>140.3</td><td>147.8</td></tr><tr><td>ours (OoD)</td><td>110.4</td><td>124.5</td><td>133.9</td><td>140.3</td><td>138.3</td><td>151.2</td><td>165.0</td><td>175.5</td><td>67.7</td><td>71.9</td><td>77.1</td><td>80.8</td><td>113.1</td><td>127.7</td><td>137.9</td><td>145.3</td></tr></table>

Table 5: Long-term prediction of 3D joint positions on H3.6M. Here, ours is also trained with the attention-GCN model.

# 7 Conclusion

We draw attention to the need for robustness to distributional shifts in predicting human motion, and propose a framework for its evaluation based on major open source datasets. We demonstrate that state-of-the-art discriminative architectures can be hardened to extreme distributional shifts by augmentation with a generative model, combining low in-distribution predictive error with maximal generalisability. The introduction of a surveyable latent space further provides a mechanism for model perspicuity and interpretability, and explicit estimates of uncertainty facilitate the detection of anomalies: both characteristics are of substantial value in emerging applications of motion prediction, such as autonomous driving, where safety is paramount. Our investigation argues for wider use of generative models in behavioural modelling, and shows it can be done with minimal or no performance penalty, within hybrid architectures of potentially diverse constitution.

# References

A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei, and S. Savarese. Social LSTM: Human trajectory prediction in crowded spaces. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 961-971, 2016.  
D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mané. Concrete problems in air safety. arXiv preprint arXiv:1606.06565, 2016.  
A. Bhattacharyya, M. Fritz, and B. Schiele. Long-term on-board prediction of people in traffic scenes under uncertainty. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4194-4202, 2018.  
A. Bourached and P. Nachev. Unsupervised videographic analysis of rodent behaviour. arXiv preprint arXiv:1910.11065, 2019.  
J. Butepage, M. J. Black, D. Kragic, and H. Kjellstrom. Deep representation learning for human motion prediction and classification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6158-6166, 2017.  
Y. Cai, L. Huang, Y. Wang, T.-J. Cham, J. Cai, J. Yuan, J. Liu, X. Yang, Y. Zhu, X. Shen, et al. Learning progressive joint propagation for human motion prediction. In Proceedings of the European Conference on Computer Vision (ECCV), 2020.  
Z. Cao, G. Hidalgo, T. Simon, S.-E. Wei, and Y. Sheikh. Openpose: realtime multi-person 2d pose estimation using part affinity fields. arXiv preprint arXiv:1812.08008, 2018.  
C.-Y. Chang, B. Lange, M. Zhang, S. Koenig, P. Requejo, N. Somboon, A. A. Sawchuk, and A. A. Rizzo. Towards pervasive physical rehabilitation using microsoft kinect. In 2012 6th international conference on pervasive computing technologies for healthcare (PervasiveHealth) and workshops, pages 159-162. IEEE, 2012.  
N. Chen, J. Bayer, S. Urban, and P. Van Der Smagt. Efficient movement representation by embedding dynamic movement primitives in deep autoencoders. In 2015 IEEE-RAS 15th International Conference on Humanoid Robots (Humanoids), pages 434-440. IEEE, 2015.  
E. Daxberger and J. M. Hernández-Lobato. Bayesian variational autoencoders for unsupervised out-of-distribution detection. arXiv preprint arXiv:1912.05651, 2019.  
K. Fragkiadaki, S. Levine, P. Felsen, and J. Malik. Recurrent network models for human dynamics. In Proceedings of the IEEE International Conference on Computer Vision, pages 4346-4354, 2015.  
E. E. Geertsema, R. D. Thijs, T. Gutter, B. Vledder, J. B. Arends, F. S. Leijten, G. H. Visser, and S. N. Kalitzin. Automated video-based detection of nocturnal convulsive seizures in a residential care setting. Epilepsia, 59:53-60, 2018.  
A. Gopalakrishnan, A. Mali, D. Kifer, L. Giles, and A. G. Ororbia. A neural temporal model for human motion prediction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 12116-12125, 2019.

F. S. Grassia. Practical parameterization of rotations using the exponential map. Journal of graphics tools, 3(3):29-48, 1998.  
W. Grathwohl, K.-C. Wang, J.-H. Jacobsen, D. Duvenaud, M. Norouzi, and K. Swersky. Your classifier is secretly an energy based model and you should treat it like one. arXiv preprint arXiv:1912.03263, 2019.  
L.-Y. Gui, K. Zhang, Y.-X. Wang, X. Liang, J. M. Moura, and M. Veloso. Teaching robots to predict human motion. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 562-567. IEEE, 2018.  
X. Guo and J. Choi. Human motion prediction via learning local structure representations and temporal dependencies. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 2580-2587, 2019.  
D. Hendrycks and K. Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv preprint arXiv:1610.02136, 2016.  
D. Hendrycks, M. Mazeika, and T. Dietterich. Deep anomaly detection with outlier exposure. arXiv preprint arXiv:1812.04606, 2018.  
C. Ionescu, F. Li, and C. Sminchisescu. Latent structured models for human pose estimation. In 2011 International Conference on Computer Vision, pages 2220-2227. IEEE, 2011.  
C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu. Human3. 6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE transactions on pattern analysis and machine intelligence, 36(7):1325-1339, 2013.  
A. Jain, A. R. Zamir, S. Savarese, and A. Saxena. Structural-rnn: Deep learning on spatio-temporal graphs. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5308-5317, 2016.  
M. Kakar, H. Nyström, L. R. Aarup, T. J. Nørtrup, and D. R. Olsen. Respiratory motion prediction by using the adaptive neuro fuzzy inference system (anfis). Physics in Medicine & Biology, 50(19): 4721, 2005.  
A. Kendall and Y. Gal. What uncertainties do we need in bayesian deep learning for computer vision? In Advances in neural information processing systems, pages 5574-5584, 2017.  
D. Kim and J. Paik. Gait recognition using active shape model and motion prediction. IET Computer Vision, 4(1):25-36, 2010.  
D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
D. P. Kingma and M. Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
T. N. Kipf and M. Welling. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308, 2016.  
D. Koller and N. Friedman. Probabilistic graphical models: principles and techniques. MIT press, 2009.  
H. Koppula and A. Saxena. Learning spatio-temporal structure from rgb-d videos for human activity detection and anticipation. In International conference on machine learning, pages 792-800, 2013a.  
H. S. Koppula and A. Saxena. Anticipating human activities for reactive robotic response. In IROS, page 2071. Tokyo, 2013b.  
R. W. Lau and A. Chan. Motion prediction for online gaming. In International Workshop on Motion in Games, pages 104-114. Springer, 2008.  
Y. LeCun, Y. Bengio, and G. Hinton. Deep learning. nature, 521(7553):436-444, 2015.

A. M. Lehrmann, P. V. Gehler, and S. Nowozin. Efficient nonlinear markov models for human motion. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1314-1321, 2014.  
C. Li, Z. Zhang, W. Sun Lee, and G. Hee Lee. Convolutional sequence to sequence model for human dynamics. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5226-5234, 2018.  
D. Li, C. Rodriguez, X. Yu, and H. Li. Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), March 2020a.  
M. Li, S. Chen, Y. Zhao, Y. Zhang, Y. Wang, and Q. Tian. Dynamic multiscale graph neural networks for 3d skeleton based human motion prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 214-223, 2020b.  
S. Liang, Y. Li, and R. Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. arXiv preprint arXiv:1706.02690, 2017.  
Z. Ma, X. Wang, R. Ma, Z. Wang, and J. Ma. Integrating gaze tracking and head-motion prediction for mobile device authentication: A proof of concept. Sensors, 18(9):2894, 2018.  
W. Mao, M. Liu, M. Salzmann, and H. Li. Learning trajectory dependencies for human motion prediction. In Proceedings of the IEEE International Conference on Computer Vision, pages 9489-9497, 2019.  
J. Martinez, M. J. Black, and J. Romero. On human motion prediction using recurrent neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2891-2900, 2017.  
L. McInnes, J. Healy, and J. Melville. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018.  
Y. Motegi, Y. Hijioka, and M. Murakami. Human motion generative model using variational autoencoder. International Journal of Modeling and Optimization, 8(1), 2018.  
A. Myronenko. 3d mri brain tumor segmentation using autoencoder regularization. In International MICCAI Brainlesion Workshop, pages 311-320. Springer, 2018.  
E. Nalisnick, A. Matsukawa, Y. W. Teh, D. Gorur, and B. Lakshminarayanan. Do deep generative models know what they don't know? arXiv preprint arXiv:1810.09136, 2018.  
B. Paden, M. Čáp, S. Z. Yong, D. Yershov, and E. Frazzoli. A survey of motion planning and control techniques for self-driving urban vehicles. IEEE Transactions on intelligent vehicles, 1(1):33-55, 2016.  
A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in pytorch. 2017.  
D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In International Conference on Machine Learning, pages 1278-1286, 2014.  
A. R. Rofougaran, M. Rofougaran, N. Seshadri, B. B. Ibrahim, J. Walley, and J. Karaoguz. Game console and gaming object with motion prediction modeling and methods for use therewith, Apr. 17 2018. US Patent 9,943,760.  
A. Shirai, E. Geslin, and S. Richir. Wiimedia: motion analysis methods and applications using a consumer video game controller. In Proceedings of the 2007 ACM SIGGRAPH symposium on Video games, pages 133-140, 2007.  
I. Sutskever, G. E. Hinton, and G. W. Taylor. The recurrent temporal restricted boltzmann machine. In Advances in neural information processing systems, pages 1601-1608, 2009.

P. Švec, A. Thakur, E. Raboin, B. C. Shah, and S. K. Gupta. Target following with motion prediction for unmanned surface vehicle operating in cluttered environments. Autonomous Robots, 36(4): 383-405, 2014.  
G. W. Taylor, G. E. Hinton, and S. T. Roweis. Modeling human motion using binary latent variables. In Advances in neural information processing systems, pages 1345-1352, 2007.  
Y. Wang, Z. Liu, Z. Zuo, Z. Li, L. Wang, and X. Luo. Trajectory planning and safety assessment of autonomous vehicles based on motion prediction and model predictive control. IEEE Transactions on Vehicular Technology, 68(9):8546-8556, 2019.  
D. Webster and O. Celik. Systematic review of Kinect applications in elderly care and stroke rehabilitation. Journal of neuroengineering and rehabilitation, 11(1):108, 2014.  
M. Wei, L. Miaomiao, and S. Mathieu. History repeats itself: Human motion prediction via motion attention. In ECCV, 2020.
