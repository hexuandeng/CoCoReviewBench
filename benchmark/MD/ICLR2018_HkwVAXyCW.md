# SKIP RNN: LEARNING TO SKIP STATE UPDATES IN RECURRENT NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recurrent Neural Networks (RNNs) continue to show outstanding performance in sequence modeling tasks. However, training RNNs on long sequences often face challenges like slow inference, vanishing gradients and difficulty in capturing long term dependencies. In backpropagation through time settings, these issues are tightly coupled with the large, sequential computational graph resulting from unfolding the RNN in time. We introduce the Skip RNN model which extends existing RNN models by learning to skip state updates and shortens the effective size of the computational graph. This model can also be encouraged to perform fewer state updates through a budget constraint. We evaluate the proposed model on various tasks and show how it can reduce the number of required RNN updates while preserving, and sometimes even improving, the performance of the baseline RNN models. Source code will be made publicly available upon publication of this work.

# 1 INTRODUCTION

Recurrent Neural Networks (RNNs) have become the standard approach for practitioners when addressing machine learning tasks involving sequential data. Such success has been enabled by the appearance of larger datasets, more powerful computing resources and improved architectures and training algorithms. Gated units, such as the Long Short-Term Memory (Hochreiter & Schmidhuber, 1997) (LSTM) and the Gated Recurrent Unit (Cho et al., 2014) (GRU), were designed to deal with the vanishing gradients problem commonly found in RNNs (Bengio et al., 1994). These architectures have become popularized thanks to their impressive results in a variety of tasks such as machine translation (Bahdanau et al., 2015), language modeling (Zaremba et al., 2015) or speech recognition (Graves et al., 2013).

Some of the main limitations of RNNs are their challenging training and deployment when dealing with long sequences, due to their inherently sequential behaviour. These challenges include throughput degradation, slower convergence during training and memory leakage, even for gated architectures (Neil et al., 2016). Sequence shortening techniques, which can be seen as a sort of conditional computation (Bengio et al., 2013; Bengio, 2013; Davis & Arel, 2013) in time, can alleviate these issues. The most common approaches, such as cropping discrete signals or reducing the sampling rate in continuous signals, are based on heuristics and can be suboptimal. In contrast, we propose a model that is able to learn which samples (i.e. elements in the input sequence) need to be used in order to solve the target task. Consider a video understanding task as an example: scenes with large motion may benefit from high frame rates, whereas only a few frames are needed to capture the semantics of a mostly static scene.

The main contribution of this work is a novel modification for existing RNN architectures that allows them to skip state updates, decreasing the number of sequential operations to be performed, without requiring any additional supervision signal. This model, called Skip RNN, adaptively determines whether the state needs to be updated or copied to the next time step. We show how the network can be encouraged to perform fewer state updates by adding a penalization term during training, allowing us to train models under different computation budgets. The proposed modification is implemented on top of well-known RNN architectures, namely LSTM and GRU, and the resulting models show promising results in a series of sequence modeling tasks. In particular, the proposed Skip RNN architecture is evaluated on six sequence learning problems: an adding task,

sine wave frequency discrimination, digit classification, sentiment analysis in movie reviews, action classification in video and temporal action localization in video<sup>1</sup>.

# 2 RELATED WORK

Conditional computation has been shown to allow gradual increases in model capacity without a proportional increase in computational cost by exploiting certain computation paths for each input (Bengio et al., 2013; Liu & Deng, 2017; Almahairi et al., 2016; McGill & Perona, 2017; Shazeer et al., 2017). This idea has been extended in the temporal domain, either by learning how many times an input needs to be pondered before moving to the next one (Graves, 2016) or building RNNs whose number of layers depends on the input data (Chung et al., 2017). Some works have addressed time-dependent computation in RNNs by updating only a fraction of the hidden states based on the current hidden state and input (Jernite et al., 2017), or following periodic patterns (Koutnik et al., 2014; Neil et al., 2016). However, due to the inherently sequential nature of RNNs and the parallel computation capabilities of modern hardware, reducing the size of the matrices involved in the computations performed at each time step does not accelerate inference. The proposed Skip RNN model can be seen as form of conditional computation in time, where the computation associated to the RNN updates may or may not be executed at every time step. This is related to the UPDATE and COPY operations in hierarchical multiscale RNNs (Chung et al., 2017), but applied to the whole stack of RNN layers at the same time. This difference is key to allowing our approach to skip input samples, effectively reducing sequential computation and shielding the hidden state over longer time lags. Learning whether to update or copy the hidden state through time steps can be seen as a learnable Zoneout mask (Krueger et al., 2017) which is shared between all the units in the hidden state. Similarly, it can be interpreted as an input-dependent recurrent version of stochastic depth (Huang et al., 2016).

Selecting parts of the input signal is similar in spirit to the hard attention mechanisms that have been applied to image regions (Mnih et al., 2014), where only some patches of the input image are attended in order to generate captions (Xu et al., 2015) or detect objects (Ba et al., 2014). Our model can be understood to generate a hard temporal attention mask on the fly given the previously seen samples, deciding which time steps should be attended and operating on a subset of input samples. Subsampling input sequences has been explored for visual storylines generation (Sigurdsson et al., 2016b), although jointly optimizing the RNN weights and the subsampling mechanism is computationally unfeasible and the Expectation Maximization algorithm is used instead. Similar research has been conducted for video analysis tasks, discovering minimally needed evidence for event recognition (Bhattacharya et al., 2014) and training agents that decide which frames need to be observed in order to localize actions in time (Yeung et al., 2016; Su & Grauman, 2016). Motivated by the advantages of training recurrent models on shorter subsequences, efforts have been conducted towards learning differentiable subsampling mechanisms (Raffel & Lawson, 2017), although the computational complexity of the proposed method precludes its application to long input sequences. In contrast, our proposed method can be trained with backpropagation and does not degrade the complexity of the baseline RNNs.

Accelerating inference in RNNs is difficult due to their inherently sequential nature, leading to the design of Quasi-Recurrent Neural Networks (Bradbury et al., 2017) and Simple Recurrent Units (Lei & Zhang, 2017), which relax the temporal dependency between consecutive steps. With the goal of speeding up RNN inference, LSTM-Jump (Yu et al., 2017) augments an LSTM cell with a classification layer that will decide how many steps to jump between RNN updates. Despite its promising results on text tasks, the model needs to be trained with REINFORCE (Williams, 1992), which requires the definition of a reward signal. Determining such reward signal is not trivial and does not necessarily generalize across tasks, e.g. regression and classification tasks may require from different reward signals. Moreover, the number of tokens read between jumps, the maximum jump distance and the number of jumps allowed need to be chosen ahead of time. These hyperparameters define a reduced set of subsequences that the model can sample, instead of allowing the network to learn any arbitrary sampling scheme. Unlike LSTM-Jump, our proposed approach is differentiable,

thus not requiring any modifications to the loss function and simplifying the optimization process, and is not limited to a predefined set of sample selection patterns.

# 3 MODEL DESCRIPTION

An RNN takes an input sequence  $\mathbf{x} = (x_{1},\dots ,x_{T})$  and generates a state sequence  $\mathbf{s} = (s_1,\ldots ,s_T)$  by iteratively applying a parametric state transition model  $S$  from  $t = 1$  to  $T$ :

$$
s _ {t} = S \left(s _ {t - 1}, x _ {t}\right) \tag {1}
$$

We augment the network with a binary state update gate,  $u_{t} \in \{0,1\}$ , selecting whether the state of the RNN will be updated ( $u_{t} = 1$ ) or copied from the previous time step ( $u_{t} = 0$ ). At every time step  $t$ , the probability  $\tilde{u}_{t + 1} \in [0,1]$  of performing a state update at  $t + 1$  is emitted. The resulting architecture is depicted in Figure 1 and can be characterized as follows:

$$
u _ {t} = f _ {\text {b i n a r i z e}} \left(\tilde {u} _ {t}\right) \tag {2}
$$

$$
s _ {t} = u _ {t} \cdot S \left(s _ {t - 1}, x _ {t}\right) + \left(1 - u _ {t}\right) \cdot s _ {t - 1} \tag {3}
$$

$$
\Delta \tilde {u} _ {t} = \sigma \left(W _ {p} s _ {t} + b _ {p}\right) \tag {4}
$$

$$
\tilde {u} _ {t + 1} = u _ {t} \cdot \Delta \tilde {u} _ {t} + (1 - u _ {t}) \cdot (\tilde {u} _ {t} + \min  (\Delta \tilde {u} _ {t}, 1 - \tilde {u} _ {t})) \tag {5}
$$

where  $W_{p}$  is a weights vector,  $b_{p}$  is a scalar bias,  $\sigma$  is the sigmoid function and  $f_{binarize} : [0,1] \to \{0,1\}$  binarizes the input value. Should the network be composed of several layers, some columns of  $W_{p}$  can be fixed to 0 so that  $\Delta \tilde{u}_{t}$  depends only on the states of a subset of layers (see Section 4.3 for an example with two layers). We implement  $f_{binarize}$  as a deterministic step function  $u_{t} = \text{round}(\tilde{u}_{t})$ , although a stochastic sampling from a Bernoulli distribution  $u_{t} \sim \text{Bernoulli}(\tilde{u}_{t})$  would be possible as well.

The model formulation implements the observation that the likelihood of requesting a new input to update the state increases with the number of consecutively skipped samples. Whenever a state update is omitted, the pre-activation of the state update gate for the following time step,  $\tilde{u}_{t+1}$ , is incremented by  $\Delta \tilde{u}_t$ . On the other hand, if a state update is performed, the accumulated value is flushed and  $\tilde{u}_{t+1} = \Delta \tilde{u}_t$ .

The number of skipped time steps can be computed ahead of time. For the particular formulation used in this work, where  $f_{\text{binarize}}$  is implemented by means of a rounding function, the number of skipped samples after performing a state update at time step  $t$  is given by:

$$
N _ {s k i p} (t) = \min  \left\{n: n \cdot \Delta \tilde {u} _ {t} \geq 0. 5 \right\} - 1 \tag {6}
$$

where  $n \in \mathbb{Z}^+$ . This enables more efficient implementations where no computation at all is performed whenever  $u_t = 0$ . These computational savings are possible because  $\Delta \tilde{u}_t = \sigma(W_p s_t + b_p) = \sigma(W_p s_{t-1} + b_p) = \Delta \tilde{u}_{t-1}$  when  $u_t = 0$  and there is no need to evaluate it again, as depicted in Figure 1d.

There are several advantages in reducing the number of RNN updates. From the computational standpoint, fewer updates translates into fewer required sequential operations to process an input signal, leading to faster inference and reduced energy consumption. Unlike some other models that aim to reduce the average number of operations per step (Neil et al., 2016; Jernite et al., 2017), ours enables skipping steps completely. Replacing RNN updates with copy operations increases the memory of the network and its ability to model long term dependencies even for gated units, since the exponential memory decay observed in LSTM and GRU (Neil et al., 2016) is alleviated. During training, gradients are propagated through fewer updating time steps, providing faster convergence in some tasks involving long sequences. Moreover, the proposed model is orthogonal to recent advances in RNNs and could be used in conjunction with such techniques, e.g. normalization (Cooijmans et al., 2017; Ba et al., 2016), regularization (Zaremba et al., 2015; Krueger et al., 2017), variable computation (Jernite et al., 2017; Neil et al., 2016) or even external memory (Graves et al., 2014; Weston et al., 2014).

![](images/e104a6d2b10455ea4a424ce3678a1d18f0eaf0d3fc804cda798517ae64293e0d.jpg)  
(a)

![](images/a357d3c83dd58d83ed94185be578d84ad0008b4ef17e9be81dc1de8e69ba3639.jpg)  
(b)

![](images/3e76bafe1080ff75eaf0cb3c160efdcb73a5b5ca58ed005d2c034051f802d8d8.jpg)  
(c)

![](images/1254cba9b52fae4aca957e1ccacfc47a50646b3dd76aad45ee1370e303c61e3c.jpg)  
(d)  
Figure 1: Model architecture of the proposed Skip RNN. (a) Complete Skip RNN architecture, where the computation graph at time step  $t$  is conditioned on  $u_{t}$ . (b) Architecture when the state is updated, i.e.  $u_{t} = 1$ . (c) Architecture when the update step is skipped and the previous state is copied, i.e.  $u_{t} = 0$ . (d) In practice, redundant computation is avoided by propagating  $\Delta \tilde{u}_{t}$  between time steps when  $u_{t} = 0$ .

# 3.1 ERROR GRADIENTS

The whole model is differentiable except for  $f_{\text{binarize}}$ , which outputs binary values. A common method for optimizing functions involving discrete variables is REINFORCE (Williams, 1992), although several estimators have been proposed for the particular case of neurons with binary outputs (Bengio et al., 2013). We select the straight-through estimator (Hinton, 2012; Bengio et al., 2013), which consists in approximating the step function by the identity when computing gradients during the backward pass:

$$
\frac {\partial f _ {b i n a r i z e} (x)}{\partial x} = 1 \tag {7}
$$

This yields a biased estimator that has proven more efficient than other unbiased but high-variance estimators such as REINFORCE (Bengio et al., 2013) and has been successfully applied in different works (Courbariaux et al., 2016; Chung et al., 2017). By using the straight-through estimator as the backward pass for  $f_{\text{binarize}}$ , all the model parameters can be trained to minimize the target loss function with standard backpropagation and without defining any additional supervision or reward signal.

# 3.2 LIMITING COMPUTATION

The Skip RNN is able to learn when to update or copy the state without explicit information about which samples are useful to solve the task at hand. However, a different operating point on the trade-off between performance and number of processed samples may be required depending on the application, e.g. one may be willing to sacrifice a few accuracy points in order to run faster on machines with a low computational power, or to reduce energy impact on portable devices. The

proposed model can be encouraged to perform fewer state updates through additional loss terms, a common practice in neural networks with dynamically allocated computation (Liu & Deng, 2017; McGill & Perona, 2017; Graves, 2016; Jernite et al., 2017). In particular, we consider a cost per sample:

$$
L _ {b u d g e t} = \lambda \cdot \sum_ {t = 1} ^ {T} u _ {t} \tag {8}
$$

where  $L_{budget}$  is the cost associated to a single sequence,  $\lambda$  is the cost per sample and  $T$  is the sequence length. This formulation bears a similarity to weight decay regularization, where the network is encouraged to slowly converge towards a solution where the norm of the weights is smaller. Similarly, in this case the network is encouraged to converge towards a solution where fewer state updates are required.

Despite this formulation has been extensively studied in our experiments, different budget loss terms can be used depending on the application. For instance, a specific number of samples may be encouraged by applying an  $L_{1}$  or  $L_{2}$  loss between the target value and the number of updates per sequence,  $\sum_{t=1}^{T} u_{t}$ .

# 4 EXPERIMENTS

In the following section, we investigate the advantages of adding this state skipping to LSTMs and GRUs for a variety of tasks. In addition to the evaluation metric for each task, we also report the number of RNN state updates (i.e. the number of elements in the input sequence that are used by the model) and the number of floating point operations (FLOPs) as measures of the computational load for each model. Since skipping an RNN update results in ignoring its corresponding input, we will refer to the number of updates and the number of used samples (i.e. elements in a sequence) interchangeably. With the goal of studying the effect of skipping state updates on the learning capability of the networks, we introduce a new baseline which skips a state update with probability  $p_{skip}$ . We tune the skipping probability to obtain models that perform a similar number of state updates to the Skip RNN models.

Training is performed with Adam (Kingma & Ba, 2014), learning rate of  $10^{-4}$ ,  $\beta_{1} = 0.9$ ,  $\beta_{2} = 0.999$  and  $\epsilon = 10^{-8}$  on batches of 256. Gradient clipping (Pascanu et al., 2013) with a threshold of 1 is applied to all trainable variables. Bias  $b_{p}$  in Equation 4 is initialized to 1, so that all samples are used at the beginning of training<sup>2</sup>. The initial hidden state  $s_0$  is learned during training, whereas  $\tilde{u}_0$  is set to a constant value of 1 in order to force the first update at  $t = 1$ .

Experiments are implemented with TensorFlow<sup>3</sup> and run on a single NVIDIA K80 GPU.

# 4.1 ADDING TASK

We revisit one of the original LSTM tasks (Hochreiter & Schmidhuber, 1997), where the network is given a sequence of (value, marker) tuples. The desired output is the addition of only the two values that are marked with a 1, whereas those marked with a 0 need to be ignored. We follow the experimental setup by Neil et al. (2016), where the first marker is randomly placed among the first  $10\%$  of samples (drawn with uniform probability) and the second one is placed among the last half of samples (drawn with uniform probability). This marker distribution yields sequences where at least  $40\%$  of the samples are distractors and provide no useful information at all. However, it is worth noting that in this task the risk of missing a marker is very large as compared to the benefits of working on shorter subsequences.

<table><tr><td>Model</td><td>Task solved</td><td>State updates</td><td>Inference FLOPs</td></tr><tr><td>LSTM</td><td>Yes</td><td>100.0% ± 0.0%</td><td>2.46 × 10^6</td></tr><tr><td>LSTM (pskip = 0.2)</td><td>No</td><td>80.0% ± 0.1%</td><td>1.97 × 10^6</td></tr><tr><td>LSTM (pskip = 0.5)</td><td>No</td><td>50.1% ± 0.1%</td><td>1.23 × 10^6</td></tr><tr><td>Skip LSTM, λ = 0</td><td>Yes</td><td>81.1% ± 3.6%</td><td>2.00 × 10^6</td></tr><tr><td>Skip LSTM, λ = 10^-5</td><td>Yes</td><td>53.9% ± 2.1%</td><td>1.33 × 10^6</td></tr><tr><td>GRU</td><td>Yes</td><td>100.0% ± 0.0%</td><td>1.85 × 10^6</td></tr><tr><td>GRU (pskip = 0.02)</td><td>No</td><td>98.0% ± 0.0%</td><td>1.81 × 10^6</td></tr><tr><td>GRU (pskip = 0.5)</td><td>No</td><td>49.9% ± 0.6%</td><td>9.25 × 10^5</td></tr><tr><td>Skip GRU, λ = 0</td><td>Yes</td><td>97.9% ± 3.2%</td><td>1.81 × 10^6</td></tr><tr><td>Skip GRU, λ = 10^-5</td><td>Yes</td><td>50.7% ± 2.6%</td><td>9.40 × 10^5</td></tr></table>

Table 1: Results for the adding task, displayed as mean ± std over four different runs. The task is considered to be solved if the MSE is at least two orders of magnitude below the variance of the output distribution.

We train RNN models with 110 units each on sequences of length 50, where the values are uniformly drawn from  $\mathcal{U}(-0.5,0.5)$ . The final RNN state is fed to a fully connected layer that regresses the scalar output. The model is trained to minimize the Mean Squared Error (MSE) between the output and the ground truth. We consider that a model is able to solve the task when its MSE on a held-out set of examples is at least two orders of magnitude below the variance of the output distribution. This criterion is a stricter version of the one followed by Hochreiter & Schmidhuber (1997).

While all models learn to solve the task, results in Table 1 show that Skip RNN models are able to do so with roughly half of the updates of their corresponding counterparts. We observed that the models using fewer updates never miss any marker, since the penalization in terms of MSE would be very large (see Section B.1 for examples). This is confirmed by the poor performance of the baselines that randomly skip state updates, which are not able to solve the tasks even when the skipping probability is low. Skip RNN models learn to skip most of the samples in the  $40\%$  of the sequence where there are no markers. Moreover, most updates are skipped once the second marker is found, since all the relevant information in the sequence has been already seen. This last pattern provides evidence that the proposed models effectively learn to decide whether to update or copy the hidden state based on the input sequence, as opposed to learning biases in the dataset only. As a downside, Skip RNN models show some difficulties skipping a large number of updates at once, probably due to the cumulative nature of  $\tilde{u}_t$ .

# 4.2 MNIST CLASSIFICATION FROM A SEQUENCE OF Pixels

The MNIST handwritten digits classification benchmark (LeCun et al., 1998) is traditionally addressed with Convolutional Neural Networks (CNNs) that can efficiently exploit spatial dependencies through weight sharing. By flattening the  $28 \times 28$  images into 784-d vectors, however, it can be reformulated as a challenging task for RNNs where long term dependencies need to be leveraged (Le et al., 2015b). We follow the standard data split and set aside 5,000 training samples for validation purposes. After processing all pixels with an RNN with 110 units, the last hidden state is fed into a linear classifier predicting the digit class. All models are trained for 600 epochs to minimize cross-entropy loss.

Table 2 summarizes classification results on the test set after 600 epochs of training. Skip RNNs are not only able to solve the task using fewer updates than their counterparts, but also show a lower variation among runs and train faster (see Figure 2). We hypothesize that skipping updates make the Skip RNNs work on shorter subsequences, simplifying the optimization process and allowing the networks to capture long term dependencies more easily. A similar behavior was observed for Phased LSTM, where increasing the sparsity of cell updates accelerates training for very long sequences (Neil et al., 2016). However, the drop in performance observed in the models where the state updates are skipped randomly suggests that learning which samples to use is a key component in the performance of Skip RNN.

<table><tr><td>Model</td><td>Accuracy</td><td>State updates</td><td>Inference FLOPs</td></tr><tr><td>LSTM</td><td>0.910 ± 0.045</td><td>784.00 ± 0.00</td><td>3.83 × 107</td></tr><tr><td>LSTM (skip = 0.5)</td><td>0.893 ± 0.003</td><td>392.03 ± 0.05</td><td>1.91 × 107</td></tr><tr><td>Skip LSTM, λ = 10-4</td><td>0.973 ± 0.002</td><td>379.38 ± 33.09</td><td>1.86 × 107</td></tr><tr><td>GRU</td><td>0.968 ± 0.013</td><td>784.00 ± 0.00</td><td>2.87 × 107</td></tr><tr><td>GRU (skip = 0.5)</td><td>0.912 ± 0.004</td><td>391.86 ± 0.14</td><td>1.44 × 107</td></tr><tr><td>Skip GRU, λ = 10-4</td><td>0.976 ± 0.003</td><td>392.62 ± 26.48</td><td>1.44 × 107</td></tr><tr><td>TANH-RNN (Le et al., 2015a)</td><td>0.350</td><td>784.00</td><td>-</td></tr><tr><td>iRNN (Le et al., 2015a)</td><td>0.970</td><td>784.00</td><td>-</td></tr><tr><td>uRNN (Arjovsky et al., 2016)</td><td>0.951</td><td>784.00</td><td>-</td></tr><tr><td>sTANH-RNN (Zhang et al., 2016)</td><td>0.981</td><td>784.00</td><td>-</td></tr><tr><td>LSTM (Cooijmans et al., 2017)</td><td>0.989</td><td>784.00</td><td>-</td></tr><tr><td>BN-LSTM (Cooijmans et al., 2017)</td><td>0.990</td><td>784.00</td><td>-</td></tr></table>

Table 2: Accuracy, used samples and average FLOPs per sequence at inference on the test set of MNIST after 600 epochs of training. Results are displayed as mean ± std over four different runs.

![](images/756fa6d8cb9dff13f033b21f39fd10ee3daac2e9e839e8dcca49b73cc2103d8f.jpg)  
Figure 2: Accuracy evolution during training on the validation set of MNIST. The Skip GRU exhibits lower variance and faster convergence than the baseline GRU. A similar behavior is observed for LSTM and Skip LSTM, but omitted for clarity. Shading shows maximum and minimum over 4 runs, while dark lines indicate the mean.

The performance of RNN models on this task can be boosted through techniques like recurrent batch normalization (Cooijmans et al., 2017) or recurrent skip coefficients (Zhang et al., 2016). Cooijmans et al. (2017) show how an LSTM with specific weight initialization schemes for improved gradient flow (Le et al., 2015a; Arjovsky et al., 2016) can reach accuracy rates of up to  $0.989\%$ . Note that these techniques are orthogonal to skipping state updates and Skip RNN models could benefit from them as well.

Sequences of pixels can be reshaped back into 2D images, allowing to visualize the samples used by the RNNs as a sort of hard visual attention model (Xu et al., 2015). Examples such as the ones depicted in Figure 3 show how the model learns to skip pixels that are not discriminative, such as the padding regions in the top and bottom of images. Similarly to the qualitative results for the adding task (Section 4.1), attended samples vary depending on the particular input being given to the network.

# 4.3 TEMPORAL ACTION LOCALIZATION ON CHARADES

One of the most accurate and scalable pipelines for video analysis consists in extracting frame level features with a CNN and modeling their temporal evolution with an RNN (Donahue et al., 2015; Yue-Hei Ng et al., 2015). Videos are commonly recorded at high sampling rates, rapidly gener

![](images/5a519dc4eb443203be823f506232e020317d74c42cdfe0bd940437e09ae5b1b8.jpg)  
Figure 3: Sample usage examples for the Skip LSTM with  $\lambda = 10^{-4}$  on the test set of MNIST. Red pixels are used, whereas blue ones are skipped.

![](images/1d6f4b1f6038a72f5d61b922194602e65443a3e619249214ceb1073677c99f72.jpg)

![](images/96704558428312884caf957dd611385ea3755cf3365bd124d1bc89ad63159efe.jpg)

![](images/4820b15c8cab2c92d344c283f1f6a2f5b92937cc1ee75cc7fc800ecdc26faa3f.jpg)

ating long sequences with strong temporal redundancy that are challenging for RNNs. Moreover, processing frames with a CNN is computationally expensive and may become prohibitive for high framereates. These issues have been alleviated in previous works by using short clips (Donahue et al., 2015) or by downsampling the original data in order to cover long temporal spans without increasing the sequence length excessively (Yue-Hei Ng et al., 2015). Instead of addressing the long sequence problem at the input data level, we let the network learn which frames need to be used.

Charades (Sigurdsson et al., 2016a) is a dataset containing 9,848 videos annotated for 157 action classes in a per-frame fashion. Frames are encoded using fc7 features from the RGB stream of a Two-Stream CNN provided by the organizers of the challenge $^{4}$ , extracted at 6fps. The encoded frames are fed into two stacked RNN layers with 256 units each and the hidden state in the last RNN layer is used to compute the update probability for the Skip RNN models. Since each frame may be annotated with zero or more classes, the networks are trained to minimize element-wise binary cross-entropy at every time step. Unlike the previous sequence tagging tasks, this setup allows to evaluate the performance of Skip RNN on a task where the output is a sequence as well.

Evaluation is performed following the setup by Sigurdsson et al. (2016c), but evaluating on 100 equally spaced frames instead of 25, and results are reported in Table 3. It is surprising that the GRU baselines that randomly skip state updates perform on par with their Skip GRU counterparts for low skipping probabilities. We hypothesize that the reasons for this behavior, which was not observed in previous experiments, are twofold: there is a supervision signal at every time step and the inputs and outputs are strongly correlated in consecutive frames. On the other hand, Skip RNN models clearly outperform the random methods when fewer updates are allowed. Note that this setup is far more challenging because of the longer time spans between updates, so that properly distributing the state updates along the sequence is key to the performance of the models. Interestingly, Skip RNN models learn which frames need to be attended from RGB data and without having access to explicit motion information.

Skip GRU tends to perform fewer state updates than Skip LSTM when the cost per sample is low or none. This behavior is the opposite of the one observed in the adding task (Section 4.1), which may be related to the observation that determining the best performing gated unit depends on the task at hand Chung et al. (2014). Indeed, GRU models consistently outperform LSTM ones on this task. This mismatch in the number of used samples is not observed for large values of  $\lambda$ , as both Skip LSTM and Skip GRU converge to a comparable number of used samples.

A previous work reports better action localization performance by integrating RGB and optical flow information as an input to an LSTM, reaching  $9.60\%$  mAP (Sigurdsson et al., 2016c). This boost in performance comes at the cost of roughly doubling the number of FLOPs and memory footprint of the CNN encoder, plus requiring the extraction of flow information during a preprocessing step. Interestingly, our model learns which frames need to be attended from RGB data and without having access to explicit motion information.

<table><tr><td>Model</td><td>mAP (%)</td><td>State updates</td><td>Inference FLOPs</td></tr><tr><td>LSTM</td><td>8.40</td><td>172.9 ± 47.4</td><td>2.65 × 1012</td></tr><tr><td>LSTM (pskip = 0.75)</td><td>8.11</td><td>43.3 ± 13.2</td><td>6.63 × 1011</td></tr><tr><td>LSTM (pskip = 0.90)</td><td>7.21</td><td>17.2 ± 6.1</td><td>2.65 × 1011</td></tr><tr><td>Skip LSTM, λ = 0</td><td>8.32</td><td>172.9 ± 47.4</td><td>2.65 × 1012</td></tr><tr><td>Skip LSTM, λ = 10-4</td><td>8.61</td><td>172.9 ± 47.4</td><td>2.65 × 1012</td></tr><tr><td>Skip LSTM, λ = 10-3</td><td>8.32</td><td>41.9 ± 11.3</td><td>6.41 × 1011</td></tr><tr><td>Skip LSTM, λ = 10-2</td><td>7.86</td><td>17.4 ± 4.4</td><td>2.66 × 1011</td></tr><tr><td>GRU</td><td>8.70</td><td>172.9 ± 47.4</td><td>2.65 × 1012</td></tr><tr><td>GRU (pskip = 0.10)</td><td>8.94</td><td>155.6 ± 42.9</td><td>2.39 × 1012</td></tr><tr><td>GRU (pskip = 0.40)</td><td>8.81</td><td>103.6 ± 29.3</td><td>1.06 × 1012</td></tr><tr><td>GRU (pskip = 0.70)</td><td>8.42</td><td>51.9 ± 15.4</td><td>7.95 × 1011</td></tr><tr><td>GRU (pskip = 0.90)</td><td>7.09</td><td>17.3 ± 6.3</td><td>2.65 × 1011</td></tr><tr><td>Skip GRU, λ = 0</td><td>8.94</td><td>159.9 ± 46.9</td><td>2.45 × 1012</td></tr><tr><td>Skip GRU, λ = 10-4</td><td>8.76</td><td>100.8 ± 28.1</td><td>1.54 × 1012</td></tr><tr><td>Skip GRU, λ = 10-3</td><td>8.68</td><td>54.2 ± 16.2</td><td>8.29 × 1011</td></tr><tr><td>Skip GRU, λ = 10-2</td><td>7.95</td><td>18.4 ± 5.1</td><td>2.82 × 1011</td></tr></table>

Table 3: Mean Average Precision (mAP), used samples and average FLOPs per sequence at inference on the validation set of Charades. The number of state updates is displayed as mean ± std over all the videos in the validation set. Note that Sigurdsson et al. (2016c) evaluate on fewer frames per video.

# 5 CONCLUSION

We presented Skip RNNs as an extension to existing recurrent architectures enabling them to skip state updates thereby reducing the number of sequential operations in the computation graph. Unlike other approaches, all parameters in Skip RNN are trained with backpropagation. Experiments conducted with LSTMs and GRUs showed that Skip RNNs can match or in some cases even outperform the baseline models while relaxing their computational requirements. Skip RNNs provide faster and more stable training for long sequences and complex models, likely due to gradients being backpropagated through fewer time steps resulting in a simpler optimization task. Moreover, the introduced computational savings are better suited for modern hardware than those methods that reduce the amount of computation required at each time step (Koutnik et al., 2014; Neil et al., 2016; Chung et al., 2017).

# REFERENCES

Amjad Almahairi, Nicolas Ballas, Tim Cooijmans, Yin Zheng, Hugo Larochelle, and Aaron Courville. Dynamic capacity networks. In ICML, 2016.  
Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. In ICML, 2016.  
Jimmy Ba, Volodymyr Mnih, and Koray Kavukcuoglu. Multiple object recognition with visual attention. arXiv preprint arXiv:1412.7755, 2014.  
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In ICLR, 2015.  
Yoshua Bengio. Deep learning of representations: Looking forward. In SLSP, 2013.  
Yoshua Bengio, Patrice Simard, and Paolo Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 1994.

Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.  
Subhabrata Bhattacharya, Felix X Yu, and Shih-Fu Chang. Minimally needed evidence for complex event recognition in unconstrained videos. In ICMR, 2014.  
James Bradbury, Stephen Merity, Caiming Xiong, and Richard Socher. Quasi-recurrent neural networks. In ICLR, 2017.  
Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 2017.  
Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. In EMNLP, 2014.  
Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.  
Junyoung Chung, Sungjin Ahn, and Yoshua Bengio. Hierarchical multiscale recurrent neural networks. In ICLR, 2017.  
Tim Coolijmans, Nicolas Ballas, Cesar Laurent, Caglar Gulçehre, and Aaron Courville. Recurrent batch normalization. In ICLR, 2017.  
Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized neural networks: Training deep neural networks with weights and activations constrained to+ 1 or-1. arXiv preprint arXiv:1602.02830, 2016.  
Andrew Davis and Itamar Arel. Low-rank approximations for conditional feedforward computation in deep neural networks. arXiv preprint arXiv:1312.4461, 2013.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.  
Jeffrey Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell. Long-term recurrent convolutional networks for visual recognition and description. In CVPR, 2015.  
Alex Graves. Adaptive computation time for recurrent neural networks. arXiv preprint arXiv:1603.08983, 2016.  
Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent neural networks. In ICASSP, 2013.  
Alex Graves, Greg Wayne, and Ivo Danihelka. Neural tuning machines. arXiv preprint arXiv:1410.5401, 2014.  
Edward Grefenstette, Karl Moritz Hermann, Mustafa Suleyman, and Phil Blunsom. Learning to transduce with unbounded memory. In NIPS, 2015.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.  
Geoffrey Hinton. Neural networks for machine learning. Coursera video lectures, 2012.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 1997.  
Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In ECCV, 2016.  
Yacine Jernite, Edouard Grave, Armand Joulin, and Tomas Mikolov. Variable computation in recurrent neural networks. In ICLR, 2017.  
Yoon Kim. Convolutional neural networks for sentence classification. In EMNLP, 2014.

Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Jan Koutnik, Klaus Greff, Faustino Gomez, and Juergen Schmidhuber. A clockwork rnn. In ICML, 2014.  
David Krueger, Tegan Maharaj, János Kramár, Mohammad Pezeshki, Nicolas Ballas, Nan Rosemary Ke, Anirudh Goyal, Yoshua Bengio, Hugo Larochelle, Aaron Courville, et al. Zoneout: Regularizing rnns by randomly preserving hidden activations. In ICLR, 2017.  
Quoc V Le, Navdeep Jaitly, and Geoffrey E Hinton. A simple way to initialize recurrent networks of rectified linear units. arXiv preprint arXiv:1504.00941, 2015a.  
Quoc V Le, Navdeep Jaitly, and Geoffrey E Hinton. A simple way to initialize recurrent networks of rectified linear units. arXiv preprint arXiv:1504.00941, 2015b.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 1998.  
Tao Lei and Yu Zhang. Training rnns as fast as cnns. arXiv preprint arXiv:1709.02755, 2017.  
Lanlan Liu and Jia Deng. Dynamic deep neural networks: Optimizing accuracy-efficiency trade-offs by selective execution. arXiv preprint arXiv:1701.00299, 2017.  
Shayne Longpre, Sabeek Pradhan, Caiming Xiong, and Richard Socher. A way out of the odyssey: Analyzing and combining recent insights for lstms. arXiv preprint arXiv:1611.05104, 2016.  
Andrew L Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In ACL, 2011.  
Mason McGill and Pietro Perona. Deciding how to decide: Dynamic routing in artificial neural networks. In ICML, 2017.  
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In NIPS, 2013.  
Takeru Miyato, Andrew M Dai, and Ian Goodfellow. Adversarial training methods for semi-supervised text classification. In ICLR, 2017.  
Volodymyr Mnih, Nicolas Heess, Alex Graves, et al. Recurrent models of visual attention. In NIPS, 2014.  
Daniel Neil, Michael Pfeiffer, and Shih-Chii Liu. Phased LSTM: accelerating recurrent network training for long or event-based sequences. In NIPS, 2016.  
Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In ICML, 2013.  
Colin Raffel and Dieterich Lawson. Training a subsampling mechanism in expectation. In *ICLR Workshop Track*, 2017.  
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In ICLR, 2017.  
Evan Shelhamer, Kate Rakelly, Judy Hoffman, and Trevor Darrell. Clockwork convnets for video semantic segmentation. arXiv preprint arXiv:1608.03609, 2016.  
Gunnar Sigurdsson, Gúl Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood in homes: Crowdsourcing data collection for activity understanding. In ECCV, 2016a.  
Gunnar A Sigurdsson, Xinlei Chen, and Abhinav Gupta. Learning visual storylines with skipping recurrent neural networks. In ECCV, 2016b.

Gunnar A Sigurdsson, Santosh Divvala, Ali Farhadi, and Abhinav Gupta. Asynchronous temporal fields for action recognition. arXiv preprint arXiv:1612.06371, 2016c.  
Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. In NIPS, 2014.  
Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.  
Yu-Chuan Su and Kristen Grauman. Leaving some stones unturned: dynamic feature prioritization for activity detection in streaming video. In ECCV, 2016.  
Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In ICCV, 2015.  
Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. arXiv preprint arXiv:1410.3916, 2014.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 1992.  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In ICML, 2015.  
Serena Yeung, Olga Russakovsky, Greg Mori, and Li Fei-Fei. End-to-end learning of action detection from frame glimpses in videos. In CVPR, 2016.  
Adams Wei Yu, Hongrae Lee, and Quoc V Le. Learning to skim text. In ACL, 2017.  
Joe Yue-Hei Ng, Matthew Hausknecht, Sudheendra Vijayanarasimhan, Oriol Vinyals, Rajat Monga, and George Toderici. Beyond short snippets: Deep networks for video classification. In CVPR, 2015.  
Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. Recurrent neural network regularization. In ICLR, 2015.  
Saizheng Zhang, Yuhuai Wu, Tong Che, Zhouhan Lin, Roland Memisevic, Ruslan R Salakhutdinov, and Yoshua Bengio. Architectural complexity measures of recurrent neural networks. In NIPS, 2016.
