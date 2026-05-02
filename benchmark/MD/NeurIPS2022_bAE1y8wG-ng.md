# Few-Shot Fast-Adaptive Anomaly Detection

Anonymous Author(s)

Affiliation

Address

email

# Abstract

The ability to detect anomaly has long been recognized as an inherent human ability, yet to date, practical AI solutions to mimic such capability have been lacking. This lack of progress can be attributed to several factors. To begin with, the distribution of "abnormalities" is intractable. Anything outside of a given normal population is by definition an anomaly. This explains why a large volume of work in this area has been dedicated to modeling the normal distribution of a given task followed by detecting deviations from it. This direction is however unsatisfying as it would require modeling the normal distribution of every task that comes along, which includes tedious data collection. In this paper, we report our work aiming to handle these issues. To deal with the intractability of abnormal distribution, we leverage Energy Based Model (EBM). EBMs learn to associates low energies to correct values and higher energies to incorrect values. At its core, the EBM employs Langevin Dynamics (LD) in generating these incorrect samples based on an iterative optimization procedure, alleviating the intractable problem of modeling the world of anomalies. Then, in order to avoid training an anomaly detector for every task, we utilize an adaptive sparse coding layer. Our intention is to design a plug and play feature that can be used to quickly update what is normal during inference time. Lastly, to avoid tedious data collection, this mentioned update of the sparse coding layer needs to be achievable with just a few shots. Here, we employ a meta learning scheme that simulates such a few shot setting during training. We support our findings with strong empirical evidence.

# 1 Introduction

Anomaly detection is an important area of study in the field of artificial intelligence. It has found utility in computer vision applications such as industrial inspection [4] and video surveillance [22, 50, 32], in the context of abuse prevention such as misinformation, fraud and network intrusion detection [49, 6, 29], and others such as system health monitoring and fault detection [3, 33]. In this paper, we propose an approach for detecting anomaly in images, where we have carefully designed steps to handle some of the bigger issues that have prevented the deployment of image anomaly detection in the real-world.

Image anomaly detection can generally be defined as the identification of abnormalities in a given image. An exact definition of abnormality in this case is elusive because abnormality can be derived from any unknown distribution outside of a normal population. Many studies have hence focused on modeling the normal population instead of learning irregularities, where the goal is to capture the shared concept among all of the normal data as one or several reference models. This process usually requires investing significant efforts in curating a large set of normal samples for each task, after which anomaly is detected as deviations from the reference model(s) [1, 47]. Recent work from [40] provides algorithms that utilize only a few normal samples to train models from scratch. However, the models still have to be provisioned for each new task, which requires considerable human efforts and expertise, and thus lack the fast deployment criterion that is often time critical for real-world

applications. In view of these challenges, our goals for this work are threefold. We are interested in designing an anomaly detection system that is capable of: (G1) modeling the normal population while at the same time has a principled approach towards modeling the abnormalities; (G2) quickly adapting to a new task at inference time; and (G3) requiring only a few normal shots to update itself to the new task at hand.

For (G1), we introduce the class of Energy Based Model (EBM), which is an important family of generative models [51, 12, 46]. EBMs have been shown to demonstrate superior capability on modeling data density and localizing anomaly [15]. For our purpose, the EBM we adopted learn to assign low energy to normal samples but high energy to abnormal samples. More importantly, the abnormal samples are generated with a procedure known as Langevin Dynamics (LD) [44], which, in its original form, starts with a noise image (see App. Fig F) and gradually samples from the distribution along the direction of lower energy. This lends itself gracefully to utilizing the generated intermediate samples as negative/abnormal. The LD procedure is then coupled with maximum likelihood loss [19] that aims to maximize the energy differences between the normal and abnormal samples.

To achieve (G2), we propose an adaptive sparse coding layer that is attached to the deep feature extractor in the EBM as Figure 1 shows. The extracted deep feature is forwarded to the sparse coding layer, where the dictionary is constructed with the features of a few normal samples of the given task. In essence, the input representation has been decomposed into a linear combination of normal features with the sparsity constraint imposed. The final energy score is measured by the distance between the original and the reconstructed features (after the sparse coding layer). Under this scheme, the dictionary for a particular task is not obtained by learning, but instead is constructed by the feature representations of a few normal samples during inference. As a result, this simple "plug-and-play" trick allows the model to be adapted to novel tasks promptly without re-training. Further, we expect that the dictionary, which is formed by normal features, will not be able to explain the abnormal samples well, causing relatively high reconstruction error that lends itself for subsequent detection. As a bonus, a backward pass of energy score minimization can be used for localizing abnormal regions. We show that using gradient to localize anomalies yields superior robustness.

Towards (G3), we utilize meta learning [42, 13] to simulate the scenario of being given a new task with a few normal shots to update the dictionary, followed by training the EBM. This is accomplished by episodic training, where in each episode the model is adapted to a held back task that is given a few normal samples. To accelerate the EBM training, we introduce "learning from inpainting", a simple yet effective strategy for synthesizing hard abnormal samples quicker by starting the LD procedure with a synthesized image that is simply a normal sample with a noise patch injected as opposed to a noise image that is traditionally what is used.

We show the proposed few-shot fast-adaptive anomaly detection and localization framework is able to efficiently adapt to a novel task (e.g., a new object category or scenes from a new camera) with a few normal samples without training on both industrial inspection and video surveillance. Compared with previous methods that adapts to new task through either from scratch training in few shots [40, 45] or few-shot with few steps of gradient descent [26], the proposed framework is the first that performs task adaptation with a single forward pass and without any gradient descent. Despite the fast adaptation, we provide both qualitative and quantitative results to demonstrate that our method outperforms other adaptive frameworks and is comparable to methods that rely on large amount of normal samples.

# 2 Backgrounds

We briefly introduce two key ingredients of the proposed method: EBMs and sparse coding.

Energy-based Model In EBMs, the goal is to learn an energy function  $E_{\theta}(\mathbf{x}):\mathbb{R}^{d}\to \mathbb{R}$  which parametrizes the data density  $p_{\theta}(\mathbf{x})$  as:

$$
p _ {\theta} (\mathbf {x}) = \frac {\exp \left(- E _ {\theta} (\mathbf {x})\right)}{\int_ {\mathbf {x}} \exp \left(- E _ {\theta} (\mathbf {x})\right)}, \tag {1}
$$

where  $\theta$  is the parameter of the energy function and  $Z_{\theta} = \int_{\mathbf{x}}\exp (-E_{\theta}(\mathbf{x}))$  is the partition function. Approximating the true data distribution  $p_{\mathrm{data}}(\mathbf{x})$  is equivalent to minimizing the expected negative log-likelihood function over the data distribution, defined by the loss function:

$$
\mathcal {L} _ {\mathrm {M L}} = \mathbb {E} _ {\mathbf {x} \sim p _ {\text {d a t a}} (\mathbf {x})} [ - \log p _ {\theta} (\mathbf {x}) ] = \mathbb {E} _ {\mathbf {x} \sim p _ {\text {d a t a}} (\mathbf {x})} [ E _ {\theta} (\mathbf {x}) + \log Z _ {\theta} ]. \tag {2}
$$

![](images/c5912da76cbcfcc25ed00e0280e5da8ae8e9304f92e44e1916adec9fd287a67e.jpg)  
Figure 1: Overview of the inference stage on a new task. (a) Adapting the task-specific dictionary with K normal samples. (b) Sparse coding with three iterations as Eqn.6 shows. We also show a backward pass from the reconstruction error to localize the abnormal regions

As the computation of  $\mathcal{L}_{\mathrm{ML}}$  involves an intractable term  $Z_{\theta}$ , the common practice is to represent the gradient of  $\mathcal{L}_{\mathrm{ML}}$  as,

$$
\nabla_ {\theta} \mathcal {L} _ {\mathrm {M L}} = \mathbb {E} _ {\mathbf {x} ^ {+} \sim p _ {\text {d a t a}} (\mathbf {x}) [ \nabla_ {\theta} E _ {\theta} (\mathbf {x} ^ {+}) ]} - \mathbb {E} _ {\mathbf {x} ^ {-} \sim p _ {\theta} (\mathbf {x}) [ \nabla_ {\theta} E _ {\theta} (\mathbf {x} ^ {-}) ]}. \tag {3}
$$

This objective decreases the energy of positive data samples  $\mathbf{x}^{+}$  from the true distribution (normal samples in our use case) and increases the energy of negative samples  $\mathbf{x}^{-}$  from the model  $p_{\theta}$  (synthesized abnormal samples). In practice, the synthesized negative samples are achieved through Langevin dynamics [44], which a  $J$ -steps sampling along the direction of energy minimization is given by:

$$
\tilde {\mathbf {x}} ^ {j} = \tilde {\mathbf {x}} ^ {j - 1} - \frac {\beta}{2} \nabla_ {\mathbf {x}} E _ {\theta} \left(\tilde {\mathbf {x}} ^ {j - 1}\right) + \omega^ {k}, \quad \omega^ {k} \sim \mathcal {N} (0, \beta \mathbf {I}), \quad j = \{1, \dots , J \} \tag {4}
$$

where  $\beta$  is the step size, and the initialization  $\mathbf{x}^0$  is sampled from a predefined prior distribution. The synthesizing ability of EBMs enables generating abnormal samples to help in learning a more accurate data density, and is often touted as the one of the advantages of using an EBM.

101 Sparse coding. Approximating a signal  $\mathbf{z}\in \mathbb{R}^d$  with the sparse linear combination over a dictionary 102  $\mathbf{D}\in \mathbb{R}^{d\times k}$  can be expressed as:

$$
\min  _ {\boldsymbol {\alpha}} \frac {1}{2} \| \mathbf {z} - \mathbf {D} \boldsymbol {\alpha} \| _ {2} ^ {2} + \lambda \| \boldsymbol {\alpha} \| _ {1}, \tag {5}
$$

where  $\alpha$  is the sparse coefficients, with its sparsity  $(l_{1}$  norm) and  $\lambda$  is the weight of the sparsity constraint.  $\mathbf{D}\alpha$  is a sparse approximation to the original signal  $\mathbf{z}$ . In practice, finding the dictionary atoms and the sparse coefficients is usually formulated as an optimization problem.

In this paper, we adopt Iterative Soft Thresholding Continuation (ISTC) [20] to convert this optimization problem into linear operations with a non-linear shrinkage function, which allows sparse coding to be seamlessly integrated into the deep neural networks. To compute a sparse coefficient  $\alpha$ , ISTC performs iterations of gradient steps on reconstruction  $||\mathbf{z} - \mathbf{D}\boldsymbol{\alpha}||^2$  and a proximal projection step to increase coefficient sparsity.

Formally, initializing the coefficients at the first step  $\alpha_0$  with all zeros, each step of ISTC refines the sparse code with descending values of  $\lambda$  from  $\lambda_{\mathrm{max}}$  to  $\lambda_{\star}$ : each step of ISTC is expressed as:

$$
\boldsymbol {\alpha} _ {n + 1} = \sigma \left(\boldsymbol {\alpha} _ {n} + \mathbf {D} ^ {\top} \left(\mathbf {z} - \mathbf {D} \boldsymbol {\alpha} _ {n}\right), \lambda_ {n}\right), \quad \text {w i t h} \quad \lambda_ {n} = \lambda_ {\max } \frac {\lambda_ {\max }}{\lambda_ {\star}} ^ {- n / N}, \tag {6}
$$

where  $\sigma (\cdot ,\cdot)$  here is a shrinkage function that truncates small values (lower than  $\lambda$  ) of the coefficients to 0 to enforce sparsity, and can be easily implemented by a customized ReLU activation function:

$$
\sigma (\mathbf {z}, \lambda) = \operatorname {s g n} (\mathbf {z}) \left(\max  \left(| \mathbf {z} | - \lambda , 0\right)\right) = \operatorname {s g n} (\mathbf {z}) \operatorname {R e L U} (| \mathbf {z} | - \lambda). \tag {7}
$$

# 3 Proposed Method

In this section, we describe the proposed fast adaptive anomaly detection framework in details. In Section 3.1, we introduce the adaptive EBM which consists of a deep feature extractor followed

by an adaptive sparse coding layer. From there, we further show that utilizing larger receptive field in the sparse coding could improve training robustness (Section 3.1.1), and applying smoothed shrinkage functions could help speed up convergence (Section 3.1.2). In Section 3.2, we describe the episodic training regime on various anomaly detection tasks that mimics few-shot adaptation in the meta-testing stage while learning common knowledge across tasks. Instead of synthesizing negative samples (anomaly) directly from noise, we introduce a simple but effective "learning from inpainting" operation to accelerate the training in Section 3.3. Finally, we summarize training steps of the proposed method in Algorithm 1.

# 3.1 Adaptive Energy-based Model

An EBM is a form of generative model and it is widely used for modeling data density and sampling. While there has been recent work [15] applying EBM to anomaly detection, it still requires retraining for each new task. To efficiently adapt the EBM to novel tasks, we introduce an adaptive sparse coding layer which is conditioned on the dictionary constructed by the features of normal samples. Specifically, as illustrated in Fig 1, given an input image,  $\mathbf{x} \in \mathbb{R}^{3 \times h \times w}$ , we first obtain the corresponding feature  $\mathbf{z} \in \mathbb{R}^{d \times h' \times w'}$  from the deep feature extractor  $\Psi$  with parameters  $\theta$ , so that  $\mathbf{z} = \boldsymbol{\Psi}(\mathbf{x}; \theta)$ . All feature vectors along spatial axes of  $\mathbf{z}$  are then sparsely decomposed through the sparse coding layer over a task-specific dictionary  $\mathbf{D} \in \mathbb{R}^{d \times Kh'w'}$ , which contains the features of  $K$  normal samples of the current task as shown in the Fig 1(a). Each feature vector of the normal sample feature is then directly used as an atom in the task dictionary. The decomposed coefficients are  $\alpha = S(\mathbf{z}; \mathbf{D})$ , where  $\alpha \in \mathbb{R}^{Kh'w' \times h' \times w'}$  and  $S$  denotes the iterative sparse decomposition process of (6). By multiplying the coefficient  $\alpha$  with the dictionary  $\mathbf{D}$ , we obtain the reconstructed features  $\mathbf{z}' = \mathbf{D}\alpha$ . The sparsity regularization to  $\alpha$  is important, as it encourages input features to be reconstructed by simple combinations of dictionary atoms (normal features), so that it would be difficult for features of abnormal samples to be well-approximated, therefore producing higher reconstruction errors that make it conducive for detecting anomalies. From here, the final energy score is formulated as the mean squared error (MSE) between the original and the reconstructed features:

$$
E _ {\theta} (\mathbf {x}; \mathbf {D}) = \operatorname {M S E} \left(\mathbf {z}, \mathbf {z} ^ {\prime}\right) = \left\| \Psi (\mathbf {x}; \theta) - \mathbf {D} \mathcal {S} \left(\Psi (\mathbf {x}; \theta); \mathbf {D}\right) \right\| ^ {2}. \tag {8}
$$

In effect, Eqn. 8 depicts a conditional EBM, which is conditioned on the task-specific D formed by normal features. With the energy score, we can obtain pixe-wise anomaly localization maps through  $\nabla_{\mathbf{x}} - E_{\theta}(\mathbf{x};\mathbf{D})$ , i.e., the gradients of pixels along the direction of minimization. High gradient magnitudes indicate regions that cannot be well explained by the dictionary D. Modifications to these regions can potentially remove the anomaly and reduce the energy as in Eqn. 4. In Section 4.1 and Appendix Section B.5, we show that using the gradient (as a natural ingredient of EBMs with LD) is more robust compared with auto-encoder and reconstruction based methods to generalize well to unseen tasks (Appendix Figure C). In the following sections, we will discuss how to make the training of this adaptive structure more robust.

# 3.1.1 Sparse Coding with Receptive Field.

As discussed in Section 3.1, the input feature  $\mathbf{z}$  is represented as  $h^{\prime} \times w^{\prime}$  of  $d$ -dim feature vectors and they are treated independently while passing through the sparse coding layer. The region of the input image that affects one feature vector is determined by the receptive field of the feature extractor. The trade-off is that a small receptive field may not capture enough contextual information, while applying a large receptive field would make feature maps spatially coarse and make it hard to spot small anomaly regions. To solve this dilemma, instead of carefully tuning the receptive field of each layer of the feature extractor, we introduce a simple yet effective technique of applying the receptive field on the sparse coding layer. Specifically, as illustrated in Appendix Fig A, rather than performing sparse coding to each individual  $d$ -dim feature vectors, we apply it on  $d \times l \times l$  volumes centered around each feature vector, where  $l$  is the receptive field. This is equivalent to applying a  $l \times l$ , sliding window on spatial axes of the feature map and can be easily implemented by image to column (Im2Col) operation. Then we flatten the feature volumes into  $dl^{2}$ -dim vectors and adjust the shape of the dictionary accordingly. In this way, we are able to capture contextual information without needing to carefully tune the architecture of the feature extractor and we show in the later experiments that this technique improves the robustness of the network on different types of objects.

![](images/f22d9a6e1c2a9c41cdfd17f342b198af856439b65dfd7570cb87fc7a2325b018.jpg)  
Figure 2: Illustration of episodic training and (a) "learning by inpainting".

# 3.1.2 Shrinkage Function

The effectiveness of training the EBM for localizing anomaly regions heavily depends on the gradient propagation from later to earlier layers. It is shown in [11] that smooth activation functions like Swish [35] could be beneficial here. Notably, the gradients of the dictionary  $\mathbf{D}$  are determined by the sparse coding coefficients  $\alpha$  as shown in Eqn. 6. However, the sparsity constraint of  $\alpha$  would turn off the gradient computation of many elements in  $\mathbf{D}$  and this could be detrimental during the early stage of the training. To alleviate the sparse gradient issue, we replace the RELU-like shrinkage function in Eqn. 7 with its smoothed counterparts by introducing the Sigmoid based shrinkage functions (SigShrink). The SigShink is originally proposed for non-parametric signal estimation in [2], and can be defined as:

$$
\sigma_ {\tau} (\mathbf {z}, \lambda) = \frac {\mathbf {z}}{1 + \exp (- \tau (| \mathbf {z} | - \lambda))}, \tag {9}
$$

where  $\tau$  is the hyperparameter of smoothness. We present visualizations of the hard shrinkage function Eqn. 7 and SigShrink with different values of  $\tau$  in Fig B. Comparing to the hard shrinkage function which truncates small values into zeros, the SigShrink with a large  $\tau$  can sharply force small values to near-zeros. Therefore, the SigShrink will guarantee non-zero gradients everywhere.

# 3.2 Episodic Training

To train the proposed adaptive EBM, we perform episodic training that is widely adopted by meta-learning few-shot learners [13, 41]. Following the terminology of few-shot learning, in each training episode, the model is adapted and tested with a task sampled from the underlying task distribution. Specifically, the model is adapted to a support set of the given task, then a query set with ground truth labels is applied to evaluate the adaptation, which is used to update the model parameters. As shown in Fig 2, the support set of the  $i$ -th episode task contains a small number of  $K$  normal samples  $\{\mathbf{s}_k^i\}_{k=1}^K$ . The features  $\mathbf{z}_k^i = \boldsymbol{\Psi}(\mathbf{x}_k^i; \theta)$  of these normal samples are plugged into the dictionary  $\mathbf{D}^i \in \mathbb{R}^{d \times Kh'w'}$  corresponding to the  $i$ -th task to adapt the dictionary. After that, the adapted model is measured by a query set consisting of  $M$  normal samples  $\{\mathbf{q}_m^i\}_{m=1}^M$  and  $M$  abnormal samples  $\{\hat{\mathbf{q}}_m^i\}_{m=1}^M$ . Note that there is no actual abnormal samples given during training, instead, they are iteratively sampled from the EBM and the sampling will be discussed in details in Section 3.3. Recall that the training of EBM with contrastive divergence as in Eqn. 3 requires the estimation of energy scores of both positive samples from the true data distribution and negative samples from the modeled distribution. The positive energy can be estimated empirically with normal query set samples. The negative energy can be estimated by performing the MCMC-based sampling technique [31, 44], typically Langevin Dynamics as described in Eqn. 4. Denoting the output of Langevin dynamics (sampled abnormal samples) initialized with  $\hat{\mathbf{q}}_m^i$  as  $\mathbf{LD}(\hat{\mathbf{q}}_m^i)$ , we have the empirical estimation of the contrastive divergence of the  $i$ -th episode as:

$$
\mathcal {L} _ {\mathrm {c d}} = \frac {1}{m} \sum_ {m = 1} ^ {M} \left[ E _ {\theta} \left(\mathbf {q} _ {m} ^ {i}; \mathbf {D} ^ {i}\right) - E _ {\theta} \left(\mathbf {L D} \left(\hat {\mathbf {q}} _ {m} ^ {i}\right); \mathbf {D} ^ {i}\right) \right]. \tag {10}
$$

With the energy score equivalent to the feature reconstruction error in Eqn. 8, minimizing  $\mathcal{L}_{\mathrm{cd}}$  encourages normal features to be well-reconstructed by a sparse linear combination of dictionary atoms while the features from abnormal samples tend to produce relatively higher reconstruction errors so that they can be easily spotted.

Algorithm 1 Training steps of few-shot adaptive anomaly detection.  
1: Given: A feature extractor  $\Psi$  with parameter  $\theta$ ; a training dataset of multiple tasks with positive (normal) samples only.  
2: Given: Number of shots  $K$ ; number of query samples  $Q$ ; step size  $\beta$  of Langevin dynamics; total training episodes  $I$ ; and learning rate  $\epsilon$ .  
3: Initialize the feature extractor  $\Psi$ .  
4: for Episode  $i = 1: I$  do  
5: Sample the  $i$ -th task from the dataset, and randomly pick  $K + M$  samples to form the support set  $\{\mathbf{s}_k^i\}_{k=1}^K$  and the query set  $\{\mathbf{q}_m^i\}_{m=1}^M$ .  
6: Generate corrupted query samples  $\{\hat{\mathbf{q}}_m^i\}_{m=1}^M$  by placing random patches to  $\{\mathbf{q}_m^i\}_{m=1}^M$ .  
7: Extract the support and query sample features with  $\Psi$  and update the adaptive sparse coding layer with  $i$ -th task dictionary  $\mathbf{D}^i$ , which is constructed by support sample features. The energy function of the  $i$ -th task is now parametrized by  $E_{\theta}(\cdot, \mathbf{D}^i)$ .  
8: Obtain synthesized negative samples  $\{\mathbf{LD}(\hat{\mathbf{q}}_m^i)\}_{m=1}^M$  with the updated energy function using Langevin dynamic in (4).  
9: Obtain the final loss  $\mathcal{L}$  with  $\mathcal{L}_{\mathrm{cd}}(10)$  and  $\mathcal{L}_{\mathrm{rec}}(11)$ .  
10: Update parameters  $\theta \gets \theta - \epsilon \nabla_\theta \mathcal{L}$ .  
11: end for  
12: Return  $\Psi$  with parameter  $\theta$ .

# 3.3 Synthesizing Negative Samples

Typical EBM training with contrastive divergence conducts negative sampling from the modeled density using techniques such as Langevin Dynamics, which applies gradient descent to a noise initialization (App. Fig F) with small step size and large number of steps [12]. Such negative sampling steps can be costly and we argue that it is unnecessary in our case. Instead, we introduce a new strategy of "learning by inpainting". Starting from a positive query sample  $\mathbf{q}_m^i$ , we synthesize the corresponding negative sample  $\hat{\mathbf{q}}_m^i$  by randomly placing a small uniform noise patch on the image. The Langevin Dynamics procedure is then initialized with the resulting image instead of a noise image. As the Langevin Dynamics proceeds, synthesized abnormal samples  $\mathbf{LD}(\hat{\mathbf{q}}_m^i)$  are inpainted along the direction of "normal",  $\mathbf{q}_m^i$ , and we introduce the following reconstruction loss:

$$
\mathcal {L} _ {\mathrm {r e c}} = \frac {1}{m} \sum_ {m = 1} ^ {M} \operatorname {M S E} \left(\mathbf {L D} \left(\hat {\mathbf {q}} _ {m} ^ {i}\right), \mathbf {q} _ {m} ^ {i}\right). \tag {11}
$$

We show in Fig 2(a) that, starting from a synthesized abnormal sample, only 5 steps of Langevin dynamic would be sufficient to make it visually close to the corresponding normal sample during training, serving as "hard negatives" that further facilitates the learning. The final loss of the episodic training is simply:

$$
\mathcal {L} = \eta_ {0} \mathcal {L} _ {\text {r e c}} + \eta_ {1} \mathcal {L} _ {\text {c d}}, \tag {12}
$$

where  $\eta_0$  and  $\eta_{1}$  are balance two loss terms.

We summarize the overall training of the proposed few-shot adaptive anomaly detection method in Alg. 1.

# 4 Experiments

In this section, we conduct evaluation on the industrial inspection task with the MVtec-AD dataset [4] (Section 4.1). Even though our proposed framework is image-based, we further demonstrate its efficacy on the video anomaly detection task in Section 4.2. In Section 4.3, we show ablations and insights relating to the adaptive sparse coding components. We show additional ablations including the superiority of using gradient of EBMs over pixel-wise reconstruction to localize anomalies in App. B and we provide implementation details in App. A.

# 4.1 Industrial Inspection

The goal of this anomaly detection task is to predict whether a manufactured component contains any defects. The MVTec-AD dataset includes 15 categories of object. To demonstrate the fast adaptation

![](images/fa7330e53f300d428467eca366111df40fac2224c601cccdae70944389d84ef4.jpg)  
Figure 3: Visualizations of localized anomaly by our method.

Table 1: Numerical evaluation of anomaly localization on MVTec-AD. We report both mIoU (top rows) and AUC-ROC (bottom rows) values. Col 2-5 are upper-bound methods trained with massive normal samples.  

<table><tr><td>Category</td><td>AE (SSIM)</td><td>AE (MSE)</td><td>AnoGAN</td><td>VE-VAE</td><td>MAML-AE</td><td>HTDGM</td><td>Ours</td></tr><tr><td rowspan="2">Carpet</td><td>0.69</td><td>0.38</td><td>0.34</td><td>0.1</td><td>0.20</td><td>0.21</td><td>0.28</td></tr><tr><td>0.87</td><td>0.59</td><td>0.54</td><td>0.78</td><td>0.68</td><td>0.78</td><td>0.83</td></tr><tr><td rowspan="2">Grid</td><td>0.88</td><td>0.83</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.07</td><td>0.12</td></tr><tr><td>0.94</td><td>0.90</td><td>0.58</td><td>0.73</td><td>0.53</td><td>0.74</td><td>0.81</td></tr><tr><td rowspan="2">Leather</td><td>0.71</td><td>0.67</td><td>0.34</td><td>0.74</td><td>0.12</td><td>0.39</td><td>0.42</td></tr><tr><td>0.78</td><td>0.75</td><td>0.64</td><td>0.87</td><td>0.77</td><td>0.90</td><td>0.98</td></tr><tr><td rowspan="2">Tile</td><td>0.04</td><td>0.23</td><td>0.08</td><td>0.14</td><td>0.14</td><td>0.12</td><td>0.28</td></tr><tr><td>0.59</td><td>0.51</td><td>0.50</td><td>0.93</td><td>0.52</td><td>0.66</td><td>0.81</td></tr><tr><td rowspan="2">Wood</td><td>0.36</td><td>0.29</td><td>0.14</td><td>0.47</td><td>0.11</td><td>0.22</td><td>0.23</td></tr><tr><td>0.73</td><td>0.73</td><td>0.62</td><td>0.91</td><td>0.68</td><td>0.79</td><td>0.78</td></tr><tr><td rowspan="2">Bottle</td><td>0.15</td><td>0.22</td><td>0.05</td><td>0.07</td><td>0.02</td><td>0.25</td><td>0.23</td></tr><tr><td>0.93</td><td>0.86</td><td>0.86</td><td>0.78</td><td>0.56</td><td>0.88</td><td>0.82</td></tr><tr><td rowspan="2">Cable</td><td>0.01</td><td>0.05</td><td>0.01</td><td>0.18</td><td>0.04</td><td>0.16</td><td>0.24</td></tr><tr><td>0.82</td><td>0.86</td><td>0.78</td><td>0.90</td><td>0.74</td><td>0.86</td><td>0.87</td></tr><tr><td rowspan="2">Capsule</td><td>0.09</td><td>0.11</td><td>0.04</td><td>0.11</td><td>0.03</td><td>0.04</td><td>0.12</td></tr><tr><td>0.94</td><td>0.88</td><td>0.84</td><td>0.74</td><td>0.68</td><td>0.88</td><td>0.90</td></tr><tr><td rowspan="2">Hazelnut</td><td>0.00</td><td>0.41</td><td>0.02</td><td>0.44</td><td>0.11</td><td>0.38</td><td>0.40</td></tr><tr><td>0.97</td><td>0.95</td><td>0.87</td><td>0.98</td><td>0.72</td><td>0.95</td><td>0.94</td></tr><tr><td rowspan="2">Metal nut</td><td>0.01</td><td>0.26</td><td>0.00</td><td>0.49</td><td>0.10</td><td>0.40</td><td>0.39</td></tr><tr><td>0.89</td><td>0.86</td><td>0.76</td><td>0.94</td><td>0.78</td><td>0.84</td><td>0.87</td></tr><tr><td rowspan="2">Pill</td><td>0.07</td><td>0.25</td><td>0.17</td><td>0.18</td><td>0.10</td><td>0.14</td><td>0.22</td></tr><tr><td>0.91</td><td>0.85</td><td>0.87</td><td>0.83</td><td>0.62</td><td>0.71</td><td>0.88</td></tr><tr><td rowspan="2">Screw</td><td>0.03</td><td>0.34</td><td>0.01</td><td>0.17</td><td>0.02</td><td>0.17</td><td>0.17</td></tr><tr><td>0.96</td><td>0.96</td><td>0.80</td><td>0.97</td><td>0.55</td><td>0.81</td><td>0.83</td></tr><tr><td rowspan="2">Toothbrush</td><td>0.08</td><td>0.51</td><td>0.07</td><td>0.14</td><td>0.06</td><td>0.14</td><td>0.23</td></tr><tr><td>0.92</td><td>0.93</td><td>0.90</td><td>0.94</td><td>0.80</td><td>0.84</td><td>0.82</td></tr><tr><td rowspan="2">Transistor</td><td>0.01</td><td>0.22</td><td>0.08</td><td>0.30</td><td>0.02</td><td>0.10</td><td>0.26</td></tr><tr><td>0.90</td><td>0.86</td><td>0.80</td><td>0.93</td><td>0.76</td><td>0.82</td><td>0.85</td></tr><tr><td rowspan="2">zipper</td><td>0.10</td><td>0.13</td><td>0.01</td><td>0.06</td><td>0.04</td><td>0.16</td><td>0.12</td></tr><tr><td>0.88</td><td>0.77</td><td>0.78</td><td>0.78</td><td>0.68</td><td>0.89</td><td>0.82</td></tr></table>

capability of the proposed method, we adopt a leave-one-out training strategy. Specifically, samples of each target category are reserved for testing only, and the episodic training is performed on the remaining categories. During the training stage, the model will not see any samples from the target category. During testing, we first adapt the model to the target category with 10 randomly selected normal samples, then measure the performance with the entire testing set. We run the test 5 times, each time the model is adapted to random sets of 10 normal samples from the target category. The final result is the average of the 5 runs.

In Table 1, we first show performance of "upper-bound" methods, which train each category from scratch with massive normal samples. Specifically, [5, 4] train auto-encoders (AE) with normal samples and measure the reconstruction errors during the inference; AnoGAN [38] adopts a generative adversarial network (GAN) to learn the manifold of normal; VE-VAE [23] presents a visually explainable variational auto encode through gradient-based attention. For apple-to/apple comparison, we create a strong baseline by applying model-agnostic meta-learning [13] on an AE (denoted as MAML-AE, more details in App. Sec. A.3). Hierarchical transformation-discriminating generative models (HTDGM) [40] trains a GAN-based anomaly detector in few shots. Although HTDGM allows adapt to a new task with few normal samples, the model still needs to be trained from scratch, which takes 4000 iterations in the 10 shot setting based on their official implementation. To the best of our knowledge, the proposed method is the first to allow adapt to new tasks with a single forward pass without any training, therefore the learned parameters are shared across tasks which greatly accelerate the model deployment. All results from our methods are obtained without any data augmentation. The proposed method outperforms MAML-AE by a large margin, and performs better than HTDGM [40] in majority of tasks. Our results are even competitive with the "upper-bounds" in some categories. We show the localized anomaly regions from our method in Fig 3. Additional visualizations are in the App. Fig D.

# 4.2 Video Surveillance

In video anomaly detection, a common goal is to detect abnormal events captured by surveillance cameras (e.g., a motorcycle on the sidewalk). A model trained on videos from one camera might not

![](images/6f4921e258b9c638c1db121b68c8d6b960e3f19d6f7f96b0b106a40ea4fc283e.jpg)  
Figure 4: Visualizations of anomaly localization with video anomaly detection.

![](images/77a6285e673e9382013f87330bfbd944bf61f0aa5f4ff05bc09d1190ea7e41e7.jpg)

![](images/a6f7febe77df277ff64a09ff7dc64fb66dfd3fad14c8608e2b753011098e0e8b.jpg)

Table 2: Frame-level AUC-ROC for the video anomaly detection tasks.  

<table><tr><td>Target datasets</td><td>Methods</td><td>1-shot</td><td>5-shot</td><td>10-shot</td></tr><tr><td rowspan="3">UCSD Ped 1</td><td>r-GAN Pre-train</td><td>73.10</td><td>73.10</td><td>73.10</td></tr><tr><td>r-GAN Fine-tune</td><td>76.99</td><td>77.85</td><td>78.23</td></tr><tr><td>r-GAN MAML</td><td>80.60</td><td>81.42</td><td>82.38</td></tr><tr><td></td><td>MAML-AE</td><td>64.12</td><td>66.88</td><td>67.34</td></tr><tr><td></td><td>Ours</td><td>77.42</td><td>78.12</td><td>78.65</td></tr><tr><td rowspan="3">UCSD Ped 2</td><td>r-GAN Pre-train</td><td>81.95</td><td>81.95</td><td>81.95</td></tr><tr><td>r-GAN Fine-tune</td><td>85.64</td><td>89.66</td><td>91.11</td></tr><tr><td>r-GAN MAML</td><td>91.19</td><td>91.80</td><td>92.80</td></tr><tr><td></td><td>MAML-AE</td><td>78.24</td><td>82.04</td><td>83.30</td></tr><tr><td></td><td>Ours</td><td>91.22</td><td>92.00</td><td>92.45</td></tr><tr><td rowspan="3">CUHK Avenue</td><td>r-GAN Pre-train</td><td>71.43</td><td>71.43</td><td>71.43</td></tr><tr><td>r-GAN Fine-tune</td><td>75.43</td><td>76.52</td><td>77.77</td></tr><tr><td>r-GAN MAML</td><td>76.58</td><td>77.10</td><td>78.79</td></tr><tr><td></td><td>MAML-AE</td><td>68.72</td><td>69.67</td><td>70.01</td></tr><tr><td></td><td>Ours</td><td>80.68</td><td>83.41</td><td>84.46</td></tr><tr><td rowspan="3">Sh-Tech</td><td>r-GAN Pre-train</td><td>70.11</td><td>70.11</td><td>70.11</td></tr><tr><td>r-GAN Fine-tune</td><td>71.61</td><td>70.47</td><td>71.59</td></tr><tr><td>r-GAN MAML</td><td>74.51</td><td>75.28</td><td>77.36</td></tr><tr><td></td><td>MAML-AE</td><td>66.62</td><td>67.12</td><td>68.04</td></tr><tr><td></td><td>Ours</td><td>75.32</td><td>79.64</td><td>81.28</td></tr></table>

![](images/c0bb7906f2b82f1673d0735ac824ccd2a355e35ccde925897b8877e41b7d000e.jpg)  
Figure 5: Loss curves with smooth (SigShrink) and non-smooth (hard-shrink RELU-like) shrinkage functions.

Table 3: Comparison of different sparse coding receptive fields. We report both mIoU (left) and AUC-ROC (right) values.  

<table><tr><td>Category</td><td colspan="2">Leather</td><td colspan="2">Grid</td><td colspan="2">Hazelnut</td><td colspan="2">Cable</td><td colspan="2">Pill</td></tr><tr><td>l=1</td><td>0.41</td><td>0.98</td><td>0.11</td><td>0.80</td><td>0.36</td><td>0.91</td><td>0.21</td><td>0.85</td><td>0.10</td><td>0.85</td></tr><tr><td>l=3</td><td>0.42</td><td>0.98</td><td>0.12</td><td>0.81</td><td>0.40</td><td>0.94</td><td>0.24</td><td>0.87</td><td>0.22</td><td>0.88</td></tr></table>

Table 4: Performance w/ and w/o sparsity constraint. From left to right: mIoU; AUC-ROC; the difference of averaged reconstruction errors between abnormal/normal samples.  

<table><tr><td>Category</td><td colspan="3">Leather</td><td colspan="3">Hazelnut</td><td colspan="3">Cable</td></tr><tr><td>Ours</td><td>0.42</td><td>0.98</td><td>1.6e-4</td><td>0.40</td><td>0.94</td><td>2.4e-4</td><td>0.24</td><td>0.87</td><td>2.0e-4</td></tr><tr><td>No sparsity</td><td>0.32</td><td>0.90</td><td>0.9e-4</td><td>0.24</td><td>0.80</td><td>1.7e-4</td><td>0.12</td><td>0.68</td><td>1.5e-4</td></tr></table>

generalize well on other cameras due to different locations / mounting heights / lightning conditions, and it is not feasible to train one model for every new camera in practice. The ability to quickly adapt to new scenes is a significant contribution to the task of video surveillance. We are only aware of the work in [26] (r-GAN) that has such adaptation capability. Specifically, the model adapts to a new scene using gradient descent with several beginning frames of a query video, after which a GAN is applied to generate future frames. Anomaly is then detected via the discrepancy between predicted future frames and the original frames. Note that the MAML-AE baseline we conducted in Section 4.1 can be seen as an ablation of r-GAN on the single-frame without temporal information.

We follow the same evaluation regime as r-GAN by training with normal samples in all 13 scenes from SH-Tech [22] and testing on UCSD Pedestrian 1, UCSD Pedestrian 2 [28], and CUHK Avenue [24]. Note that since our method is image-based, it predicts the video frames independently without leveraging any temporal information as in r-GAN. In each episode, we adapt our model with a support set containing a few normal frames randomly sampled from the target scenes. In Table 2, we compare our method against r-GAN pre-trained on SH-Tech only (r-GAN Pre-train), fine-tuned on target datasets (r-GAN Fine-tune), and with one step gradient descent with meta-learning (r-GAN MAML). We also show the performance of MAML-AE as a baseline for image-based meta-learning method. In the last section of Table 2, we present intra-dataset results as well by training with 6 scenes of SH-Tech and testing on remaining 7. We follow common evaluation protocol and measure the frame-level AUC-ROC. Without leveraging temporal information and re-training (gradient descent), our method

achieves comparable results to r-GAN MAML and outperforms image-based meta-learning method by a large margin. In App B.4, we show that incorporating simple temporal information can further improve the performance.

# 4.3 Ablation Studies

Sparse coding receptive fields. To evaluate the effectiveness of using large receptive fields in the sparse coding layer, we conduct additional experiments on the MVTec-AD dataset, and select 5 representative categories with different levels of difficulties to present the comparisons with  $l = 1$  and  $l = 3$  (Sec. 3.1.1) in Table 3. Sparse coding with large receptive field clearly benefits more complex structural objects (hazelnut, cable, and capsule), while the improvements are limited for the texture objects (leather and grid), where contextual regularization is intuitively less important.

Shrinkage functions. To show the benefits of smooth shrinkage function, we plot the loss curves of models trained with smooth SigShrink (Eqn. 9) and non-smooth RELU-like shrinkage (Eqn. 7) functions in Fig 5. The model with smooth shrinkage function converges notably faster in the early training stage and achieves lower loss.

Sparsity constraint. As discussed in Section 3.1, we impose sparsity constraint to the feature decomposition in the adaptive sparse coding layer, in order to prevent abnormal features from being well-approximated by the linear combinations of normal features, so that the reconstruction errors are effective for detecting anomaly. To validate this, we conduct experiments by removing the shrinkage function  $\sigma$  in the sparse coding stage (Eqn. 6). We show comparison in Table 4 with mIoU, AUC-ROC, and the difference of averaged reconstruction errors between abnormal and normal samples. Without sparsity, the performance drops dramatically, and reconstruction errors of normal and abnormal samples become closer.

# 5 Related Work

Anomaly detection with sparse coding. Early efforts on adopting sparse coding in anomaly detection are based on optimization (with L1 penalty) [24, 50]. Recent advances on iterative sparse thresholding algorithms [9, 20] allow seamless integration of online sparse coding with deep neural networks, and [27] formulates the sparse coding as stack RNNs for video anomaly detection.

Anomaly detection with generative models. Generative models are widely utilized in anomaly detection due to the capability in modeling the density of desired data distribution. Early efforts on variational autoencoders (VAE) based methods [1, 47] are arguably having hard time calibrating uncertainties in novel samples [30], accurately localizing abnormal regions through reconstruction errors [10]. Recent efforts have explored variant generative architectures like energy-based models (EBM) [15], GANs [40], and combining VAE with EBM [10]. Various methods also exploit intra-image structures [8, 5], cross-frame consistency [25], and motion-appearance consistency in videos [32] while detecting anomaly.

Few-shot learning. Few-shot learning is extensively explored in classification tasks. Proposed methods are based on optimization [13, 37, 14, 48, 36], learning metric [41, 43] and parameter prediction [17, 34, 16]. These technologies are further applied in other tasks like image generation [7, 21] and out-of-distribution detection [39].

# 6 Conclusion

In this paper, we introduced few-shot fast-adaptive anomaly detection. We formulated our model as an energy-based model with an adaptive sparse coding layer, of which the dictionary is directly formed by normal features of a target task. We adopted episodic meta-learning to learn common knowledge across tasks, which enables few shots adaptation. We further introduced smooth shrinkage functions, sparse coding with large receptive fields, and learning by inpainting to improve and accelerate the training. Notably, when evaluating our method's performance on industrial inspection and video anomaly detection, our method is comparable and even boasts better performance than methods trained with a large amount of normal samples. Through this work, we hope to have made a significant contribution to the important problem of anomaly detection by shedding light on our findings that anomaly detection can indeed be generalized to new tasks with a few normal samples only.

Social Impact and Ethics. As a general framework for few-shot anomaly detection, the proposed method does not suffer from particular ethical concerns or negative social impacts. All datasets used are public, and we have blurred all human faces in the qualitative visualizations.

# References

[1] Jinwon An and Sungzoon Cho. Variational autoencoder based anomaly detection using reconstruction probability. *Special Lecture on IE*, 2(1):1-18, 2015.  
[2] Abdourrahmane M Atto, Dominique Pastor, and Gregoire Mercier. Smooth sigmoid wavelet shrinkage for non-parametric estimation. In 2008 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 3265-3268. IEEE, 2008.  
[3] Yuequan Bao, Zhiyi Tang, Hui Li, and Yufeng Zhang. Computer vision and deep learning-based data anomaly detection method for structural health monitoring. Structural Health Monitoring, 18(2):401-421, 2019.  
[4] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Mvtec ad-a comprehensive real-world dataset for unsupervised anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9592-9600, 2019.  
[5] Paul Bergmann, Sindy Löwe, Michael Fauser, David Sattlegger, and Carsten Steger. Improving unsupervised defect segmentation by applying structural similarity to autoencoders. arXiv preprint arXiv:1807.02011, 2018.  
[6] Richard J Bolton and David J Hand. Statistical fraud detection: A review. Statistical science, 17(3):235-255, 2002.  
[7] Louis Clouatre and Marc Demers. Figr: Few-shot image generation with reptile. arXiv preprint arXiv:1901.02199, 2019.  
[8] Niv Cohen and Yedid Hoshen. Sub-image anomaly detection with deep pyramid correspondences. arXiv preprint arXiv:2005.02357, 2020.  
[9] Ingrid Daubechies, Michel Defrise, and Christine De Mol. An iterative thresholding algorithm for linear inverse problems with a sparsity constraint. Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences, 57(11):1413-1457, 2004.  
[10] David Dehaene, Oriel Frigo, Sébastien Combrexelle, and Pierre Eline. Iterative energy-based projection on a normal data manifold for anomaly localization. *ICLR*, 2020.  
[11] Yilun Du, Shuang Li, Joshua Tenenbaum, and Igor Mordatch. Improved contrastive divergence training of energy based models. International Conference on Machine Learning, 2021.  
[12] Yilun Du and Igor Mordatch. Implicit generation and generalization in energy-based models. NeurIPS, 2019.  
[13] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, pages 1126–1135. PMLR, 2017.  
[14] Chelsea Finn, Kelvin Xu, and Sergey Levine. Probabilistic model-agnostic meta-learning. arXiv preprint arXiv:1806.02817, 2018.  
[15] Ergin Utku Genc, Nilesh Ahuja, Ibrahima J Ndiour, and Omesh Tickoo. Energy-based anomaly detection and localization. arXiv preprint arXiv:2105.03270, 2021.  
[16] Spyros Gidaris and Nikos Komodakis. Generating classification weights with gnn denoising autoencoders for few-shot learning. In IEEE Conference on Computer Vision and Pattern Recognition, pages 21-30, 2019.  
[17] Jonathan Gordon, John Bronskill, Matthias Bauer, Sebastian Nowozin, and Richard E Turner. Meta-learning probabilistic inference for prediction. arXiv preprint arXiv:1805.09921, 2018.  
[18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770-778, 2016.  
[19] Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural computation, 14(8):1771-1800, 2002.  
[20] Yuling Jiao, Bangti Jin, and Xiliang Lu. Iterative soft/hard thresholding with homotopy continuation for sparse recovery. IEEE Signal Processing Letters, 24(6):784-788, 2017.  
[21] Ming-Yu Liu, Xun Huang, Arun Mallya, Tero Karras, Timo Aila, Jaakko Lehtinen, and Jan Kautz. Few-shot unsupervised image-to-image translation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10551-10560, 2019.

[22] W. Liu, D. Lian W. Luo, and S. Gao. Future frame prediction for anomaly detection - a new baseline. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.  
[23] Wenqian Liu, Runze Li, Meng Zheng, Srikrishna Karanam, Ziyan Wu, Bir Bhanu, Richard J Radke, and Octavia Camps. Towards visually explaining variational autoencoders. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8642-8651, 2020.  
[24] Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 fps in matlab. In Proceedings of the IEEE international conference on computer vision, pages 2720-2727, 2013.  
[25] Yiwei Lu, K Mahesh Kumar, Seyed shahabeddin Nabavi, and Yang Wang. Future frame prediction using convolutional vnn for anomaly detection. In 2019 16th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS), pages 1-8. IEEE, 2019.  
[26] Yiwei Lu, Frank Yu, Mahesh Kumar Krishna Reddy, and Yang Wang. Few-shot scene-adaptive anomaly detection. In European Conference on Computer Vision, pages 125-141. Springer, 2020.  
[27] Weixin Luo, Wen Liu, and Shenghua Gao. A revisit of sparse coding based anomaly detection in stacked rnn framework. In Proceedings of the IEEE International Conference on Computer Vision, pages 341-349, 2017.  
[28] Vijay Mahadevan, Weixin Li, Viral Bhalodia, and Nuno Vasconcelos. Anomaly detection in crowded scenes. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 1975-1981. IEEE, 2010.  
[29] Biswanath Mukherjee, L Todd Heberlein, and Karl N Levitt. Network intrusion detection. IEEE network, 8(3):26-41, 1994.  
[30] Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, and Balaji Lakshminarayanan. Do deep generative models know what they don't know? arXiv preprint arXiv:1810.09136, 2018.  
[31] Radford M Neal et al. Mcmc using hamiltonian dynamics. Handbook of markov chain monte carlo, 2(11):2, 2011.  
[32] Trong-Nguyen Nguyen and Jean Meunier. Anomaly detection in video sequence with appearance-motion correspondence. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1273-1283, 2019.  
[33] Afrooz Purarjomandlangrudi, Amir Hossein Ghapanchi, and Mohammad Esmalifalak. A data mining approach for fault diagnosis: An application of anomaly detection algorithm. Measurement, 55:343-352, 2014.  
[34] Siyuan Qiao, Chenxi Liu, Wei Shen, and Alan L Yuille. Few-shot image recognition by predicting parameters from activations. In IEEE Conference on Computer Vision and Pattern Recognition, pages 7229-7238, 2018.  
[35] Prajit Ramachandran, Barret Zoph, and Quoc V Le. Searching for activation functions. arXiv preprint arXiv:1710.05941, 2017.  
[36] Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. International Conference on Learning Representations, 2016.  
[37] Andrei A Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta-learning with latent embedding optimization. International Conference on Learning Representations, 2019.  
[38] Thomas Schlegl, Philipp Seebock, Sebastian M Waldstein, Ursula Schmidt-Erfurth, and Georg Langs. Unsupervised anomaly detection with generative adversarial networks to guide marker discovery. In International conference on information processing in medical imaging, pages 146-157. Springer, 2017.  
[39] Vikash Sehwag, Mung Chiang, and Prateek Mittal. Ssd: A unified framework for self-supervised outlier detection. arXiv preprint arXiv:2103.12051, 2021.  
[40] Shelly Sheynin, Sagie Benaim, and Lior Wolf. A hierarchical transformation-discriminating generative model for few shot anomaly detection. arXiv preprint arXiv:2104.14535, 2021.  
[41] Jake Snell, Kevin Swersky, and Richard S Zemel. Prototypical networks for few-shot learning. Advances in Neural Information Processing Systems, 2017.  
[42] Ricardo Vilalta and Youssef Drissi. A perspective view and survey of meta-learning. Artificial intelligence review, 18(2):77-95, 2002.

[43] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. Matching networks for one shot learning. Advances in Neural Information Processing Systems, 2016.  
[44] Max Welling and Yee W Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pages 681–688. CiteSeer, 2011.  
[45] Jhih-Ciang Wu, Ding-Jie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu. Learning unsupervised metaformer for anomaly detection. In International Conference on Computer Vision, pages 4369-4378, 2021.  
[46] Jianwen Xie, Yang Lu, Song-Chun Zhu, and Yingnian Wu. A theory of generative convnet. In International Conference on Machine Learning, pages 2635-2644. PMLR, 2016.  
[47] Haowen Xu, Wenxiao Chen, Nengwen Zhao, Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, Dan Pei, Yang Feng, et al. Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications. In Proceedings of the 2018 World Wide Web Conference, pages 187–196, 2018.  
[48] Jaesik Yoon, Taesup Kim, Ousmane Dia, Sungwoong Kim, Yoshua Bengio, and Sungjin Ahn. Bayesian model-agnostic meta-learning. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pages 7343–7353, 2018.  
[49] Qiang Zhang, Aldo Lipani, Shangsong Liang, and Emine Yilmaz. Reply-aided detection of misinformation via bayesian deep learning. In The world wide web conference, pages 2333–2343, 2019.  
[50] Bin Zhao, Li Fei-Fei, and Eric P Xing. Online detection of unusual events in videos via dynamic sparse coding. In CVPR 2011, pages 3313–3320. IEEE, 2011.  
[51] Junbo Zhao, Michael Mathieu, and Yann LeCun. Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126, 2016.  
[52] Yang Zhao, Jianwen Xie, and Ping Li. Learning energy-based generative models via coarse-to-fine expanding and sampling. In International Conference on Learning Representations, 2020.
