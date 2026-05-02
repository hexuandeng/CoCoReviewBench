# SALIENCY-GUIDED HIDDEN ASSOCIATIVE REPLAY FOR CONTINUAL LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Continual Learning (CL) is a burgeoning domain in next-generation AI, focusing on training neural networks over a sequence of tasks akin to human learning. While CL provides an edge over traditional supervised learning, its central challenge remains to counteract catastrophic forgetting and ensure the retention of prior tasks during subsequent learning. Amongst various strategies to tackle this, replay-based methods have emerged as preeminent, echoing biological memory mechanisms. However, these methods are memory-intensive, often preserving entire data samples—an approach inconsistent with humans' selective memory retention of salient experiences. While some recent works have explored the storage of only significant portions of data in episodic memory, the inherent nature of partial data necessitates innovative retrieval mechanisms. Current solutions, like inpainting, approximate full data reconstruction from partial cues, a method that diverges from genuine human memory processes. Addressing these nuances, this paper presents the Saliency-Guided Hidden Associative Replay for Continual Learning (SHARC). This novel framework synergizes associative memory with replay-based strategies. SHARC primarily archives salient data segments via sparse memory encoding. Importantly, by harnessing associative memory paradigms, it introduces a content-focused memory retrieval mechanism, promising swift and near-perfect recall, bringing CL a step closer to authentic human memory processes. Extensive experimental results demonstrate the effectiveness of our proposed method for various continual learning tasks.

# 1 INTRODUCTION

Continual learning (CL) represents a vital advancement for next-generation AI, allowing neural networks to sequentially learn tasks like humans do Parisi et al. (2019). While traditional supervised learning is well-established, CL remains in its nascent stages. The main challenge is to prevent Catastrophic Forgetting McCloskey & Cohen (1989) as agents acquire new tasks, ensuring they retain earlier knowledge. In essence, CL strives to balance updating the model with retention across a series of tasks. In order to address this problem, researchers have put forward several strategies. Replay-based methods Rebuffi et al. (2017); Aljundi et al. (2019); Arani et al. (2022), which utilizes a small memory to store previous data and reuse them when learning new tasks, have emerged as a particularly effective solution, offering superior performance and drawing inspiration from biological systems Robins (1995). However, a potential bottleneck of this approach is its memory-intensive nature, as entire data samples are conserved. This mechanism contrasts starkly with the human brain's approach to memory retention. Humans typically do not remember every detail but tend to recall fragments or the most salient features of experiences Rolls (2013). The vast storage requirements of replay-based methods and their divergence from natural memory processes necessitate exploration into more efficient and human-like strategies for continual learning.

While there are pioneering works Saha & Roy (2023); Bai et al. (2023) in replay-based CL that have begun to explore the idea of storing only the salient or partial aspects of data into episodic memory, challenges arise due to the inherent nature of partial data. Since these fragments are not directly usable as model input, an effective retrieval technique becomes indispensable. A straightforward solution is inpainting Elharrouss et al. (2020), which, through rule-based or generative models, attempts to recreate the full data from the available partial cue. This methodology, however, essentially approximates the entirety of the data by generating similar samples from a given distribution and may suffer from inaccurate retrieval under large noise or corruption (Col 3 in Figure 1). On the contrary,

![](images/73df7b79a5f20751c380c1d4921b06725ae681c588891e46e17a0582b2e9d826.jpg)  
Figure 1: Content-based (associative memory)  $\nu.s.$  generative model retrieval. Pixels from non-salient areas are masked in query images. For a fair comparison, we train an autoencoder-based inpainting model Peng et al. (2021) and a Hopfield Network Ramsauer et al. (2020) with similar number of parameters. Associative memory achieves almost perfect recall even under large noise or corruption.

the human brain, especially the hippocampus, employs associative recall for content-based memory retrieval Hopfield (1982); Ramsauer et al. (2020), achieving a remarkable recall accuracy close to perfection (Col 4 in Figure 1). As such, for systems aiming to emulate human-like continual learning, there is an evident inspiration to design techniques that mirror the associative and content-based retrieval processes inherent in human cognition.

To address the aforementioned challenges, this paper introduces the Saliency-Guided Hidden Associative Replay for Continual Learning (SHARC), marking the inception of a Continual Learning framework that seamlessly integrates associative memory into replay-based techniques. As depicted in Figure 1, SHARC distinguishes itself from existing replay-based CL methodologies in two pivotal aspects: First, rather than archiving complete samples within episodic memory, SHARC conserves only the most salient segments through sparse memory encoding. More crucially, drawing inspiration from the principles of associative memory, we have crafted a content-centric memory retrieval module that boasts swift and impeccable recall capabilities.

Our contribution includes, 1). We develop a novel neural-inspired replay-based continual learning framework to handle catastrophic forgetting. 2). We propose to leverage associative memory for efficient memory storage and recovery. 3). We demonstrate our model's efficacy and superiority with extensive experiments.

# 2 RELATED WORK

Continual Learning (CL). Catastrophic forgetting is a long-standing problem Robins (1995) in continual learning which has been recently tackled in a variety of visual tasks such as image classification Kirkpatrick et al. (2017); Rebuffi et al. (2017), object detection Zhou et al. (2020), etc.

Existing techniques in CL can be divided into three main categories Parisi et al. (2019): 1) regularization-based approaches, 2) dynamic architectures and 3) replay-based approaches. Regularization-based approaches alleviate catastrophic forgetting by either adding a regularization term to the objective function Kirkpatrick et al. (2017) or knowledge distillation over previous tasks Li & Hoiem (2017). Dynamic architecture approaches adaptively accommodate the network architecture (e.g., adding more neurons or layers) in response to new information during training. Dynamic architectures can be explicit if new network branches are grown, or implicit, if some network parameters are only available for certain tasks. Replay-based approaches alleviate the forgetting of deep neural networks by replaying stored samples from the previous history when learning new ones and have been shown to be the most effective method for mitigating catastrophic forgetting.

Replay-based CL. Replay-based methods mainly include three directions: rehearsal methods, constrained optimization, and pseudo rehearsal. Rehearsal methods directly retrieve previous samples from a limited size memory together with new samples for training Chaudhry et al. (2019); Hayes et al. (2020); Arani et al. (2022). While simple in nature, this approach is prone to overfitting the old samples from the memory. As an alternative, constrained optimization methods formulate

backward/forward transfer as constraints in the objective function. GEM Lopez-Paz & Ranzato (2017) constrains new task updates to not interfere with previous tasks by projecting the estimated gradient on the feasible region outlined by previous task gradients through first-order Taylor series approximation. A-GEM Chaudhry et al. (2018) further extended GEM and made the constraint computationally more efficient. Finally, pseudo-rehearsal methods typically utilize generative models such as GAN Goodfellow et al. (2020) or VAE Pu et al. (2016) to generate previous samples from random inputs and have shown the ability to generate high-quality images recently Robins (1995). Readers may refer to Parisi et al. (2019) for a more comprehensive survey on continual learning.

Associative Memory (AM). In general, the attractor-based mechanism Amit & Amit (1989) is typically used for the implementation of AMs, which are models that store and recall patterns. Pattern recall (associative recall) is a process whereby an associative memory, upon receiving a potentially corrupted memory query, retrieves the associated value from memory. One of the earliest and probably the most well-known associative memory are Hopfield Networks Hopfield (1982). Hopfield networks are a class of recurrent artificial neural networks that have gained prominence for their ability to model associative memory and pattern recognition. The modern Hopfield network refers to an updated version of the original Hopfield network Ramsauer et al. (2020); Krotov & Hopfield (2020). The modern Hopfield network incorporates enhancements and modifications to improve its performance and overcome some limitations of the original model. More recently, predictive coding networks Huang & Rao (2011) have provided a new perspective for the design of AM, and such works Salvatori et al. (2021); Yoo & Wood (2022) have shown strong performance on recall tasks.

# 3 PROBLEM FORMULATION

We consider supervised continual learning in this paper. Following the learning protocol in Chaudhry et al. (2018), we consider a training set  $\mathcal{D} = \{\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_T\}$  consisting of  $T$  tasks, where  $\mathcal{D}_t = \{(\mathbf{x}_i^{(t)}, \mathbf{y}_i^{(t)})\}_{i=1}^{n_t}$  contains  $n_t$  input-target pairs  $(\mathbf{x}_i^{(t)}, \mathbf{y}_i^{(t)}) \in \mathcal{X} \times \mathcal{Y}$ . While each learning task arrives sequentially, we make the assumption of locally i.i.d, i.e.,  $\forall t, (\mathbf{x}_i^{(t)}, \mathbf{y}_i^{(t)}) \stackrel{iid}{\sim} P_t$ , where  $P_t$  denotes the data distribution for task  $t$  and i.i.d for independent and identically distributed. Given such a stream of tasks, the goal is to train a learning agent  $f_\theta: \mathcal{X} \to \mathcal{Y}$ , parameterized by  $\theta$ , which can be queried at any time to predict the target  $\mathbf{y}$  given associated unseen input  $\mathbf{x}$  and task id  $t$ . Moreover, we require that such a learning agent can only store a small amount of seen samples in an episodic memory  $\mathcal{M}$  with a fixed budget. Given predictor  $f_\theta$ , the loss on the episodic memory of task  $k$  is defined as

$$
\ell \left(f _ {\boldsymbol {\theta}}, \mathcal {M} _ {k}\right) := \left| \mathcal {M} _ {k} \right| ^ {- 1} \sum_ {\left(\mathbf {x} _ {i}, k, \mathbf {y} _ {i}\right)} \phi \left(f _ {\boldsymbol {\theta}} \left(\mathbf {x} _ {\mathbf {i}}, k\right), \mathbf {y} _ {i}\right), \forall k <   t, \tag {1}
$$

where  $\phi$  can be e.g. cross-entropy or MSE. In general, a large body of replay-based continual learning methods seeks to optimize for the following loss function at  $t$ -th task

$$
\min  _ {\boldsymbol {\theta}} \mathcal {L} _ {C L} (\boldsymbol {\theta}), \text {w h e r e} \mathcal {L} _ {C L} (\boldsymbol {\theta}) = \sum_ {(\mathbf {x}, t, \mathbf {y})} \ell \left(f _ {\boldsymbol {\theta}} (\mathbf {x}, t), \mathbf {y}\right) + \sum_ {k <   t} \ell \left(f _ {\boldsymbol {\theta}}, \mathcal {M} _ {k}\right), \tag {2}
$$

which is an aggregation of the losses on the current task and replay data. After the training of task  $t$ , a subset of training samples will be stored in the episodic memory, i.e.,  $\mathcal{M} = \mathcal{M} \cup \{(\mathbf{x}_i^{(t)},\mathbf{y}_i^{(t)})\}_{i = 1}^{m_t}$  where  $m_{t}$  is the memory buffer size for the current task.

# 4 PROPOSED METHOD

In this section, we introduce our proposed Saliency-Guided Hidden Associative Replay for Continual Learning. We innovatively utilize saliency methods to select the most important channels of feature maps for each image and only store those channels in the episodic memory, thus achieving controllable and better memory efficiency. During the training phase, we leverage pattern association techniques for memory completion, where each partially stored image will be restored by a brain-inspired associative memory. An overview of our framework is shown in Figure 2.

# 4.1 SALIENCY-GUIDED MEMORY ENCODING WITH STRUCTURED SPARSITY

In this section, we discuss the memory encoding process of our method. According to the hippocampal indexing theory Teyler & Rudy (2007), there are two major characteristics of how the human brain encodes its memories. First, the encoded representations in the human hippocampus are highly sparse, meaning that only a small subset of neurons in the hippocampus is activated for each specific

![](images/93aa1153e7a76d5a116b89e9fa5cfeeab4a246e634586082868ba7f4d41763f3.jpg)  
Figure 2: Overview of our proposed SHARC framework (best viewed in color). When new data comes, a pre-trained backbone is used to extract feature maps. Then, the saliency score of the saliency map is calculated via backpropagation and we drop channels with lower saliency, thus achieving structured sparsity and memory efficiency. During memory replay, previous feature maps are retrieved via an associative memory, which essentially picks the top-1 stored feature maps based on the similarity with the query feature map.

memory. Second, the stored representations for replay are not exact reproductions (e.g., raw pixels) Ji & Wilson (2007); instead, its visual inputs originate higher in the visual processing hierarchy rather than from the primary visual cortex or the retina Insausti et al. (2017). Motivated by this, our goal is to propose a computational model for memory encoding that satisfies both characteristics.

Different from many earlier works that store raw images for replay Lopez-Paz & Ranzato (2017); Ebrahimi et al. (2021); Arani et al. (2022), we consider first encoding raw data into high-level representations and store them. Formally, our model  $\mathbf{y} = f_{\theta}(g(\mathbf{x}))$  is composed of a pre-trained backbone  $g$  and a trainable prediction head  $f_{\theta}$ . In this work, the pre-trained backbone is general and can be instantiated as various vision models such as VGG, ResNet, and ViT.

The output of  $g(\mathbf{x})$  is a tensor  $A \in \mathbb{R}^{H \times W \times K}$ , where  $H, W$  are the dimension of the feature map and  $K$  is the number of channels. To achieve sparse representation, we consider saliency-based methods Selvaraju et al. (2017) that measure the importance of the neurons by their first-order gradients. Specifically, the saliency approach computes the gradient of the score for class c, i.e.,  $f_{\theta}^{(c)}(A)$ , with respect to feature map activations  $A$ , followed by a global-average-pooling over the width and height dimension to obtain the neuron importance weights  $\alpha_k^c$ :

$$
\forall k, \alpha_ {k} ^ {c} = \frac {1}{H} \sum_ {i = 1} ^ {H} \frac {1}{W} \sum_ {j = 1} ^ {W} \partial f _ {\theta} ^ {(c)} (A) / \partial A _ {i j} ^ {k}. \tag {3}
$$

Intuitively speaking, feature maps with higher magnitudes in the saliency are more likely to be involved in the region of the target class object while those with lower magnitudes are more likely to be the non-target objects or background regions. In addition, existing work Sun et al. (2015) has proved that hidden representation learned by convolutional neural networks is highly sparse in the hidden space. In this work, we consider the saliency score as a measure and mask out feature maps with lower saliency scores. Formally, denote  $\alpha^c = [\alpha_1^c,\alpha_2^c,\dots ,\alpha_K^c ]$  , the masked feature map  $A^{\prime}$  is

$$
A ^ {\prime} = \operatorname {T R} _ {H, W, K} \left(\mathbb {1} \left\{\left| \boldsymbol {\alpha} ^ {c} \right| > Q _ {\mu} \right\} \otimes \mathbf {J} _ {H, W}\right) \odot A, \tag {4}
$$

where  $\mathbb{1}\{\cdot\}$  is the indicator function,  $Q_{\mu}$  is the threshold for masking out the bottom  $\mu$  quantile of channels.  $\mathbf{J}$  denotes the all-one matrix and  $\mathrm{TR}(\cdot)$  denotes tensor reshaping operation.

One advantage of our design is that the channel-wise operations result in structured sparsity, which is hardware-friendly and can lead to memory cost reduction instantly without further system-level efforts. To see this, we discard those channels that have a saliency score lower than the threshold, and the rest feature maps have a regular tensor shape. We only need to keep track of the channel index which is only a 1d vector and cheap to store.

# 4.2 ASSOCIATIVE MEMORY RETRIEVAL FOR REPLAY

In this section, we discuss the memory retrieval process of our method. Associative memory plays an important role in human intelligence and its mechanisms have been linked to attention in machine learning Ramsauer et al. (2020). Recently, the machine learning community's interest in associative

Algorithm 1 SHARC Training  
Require: Continual learning classifier  $f_{\theta}$  associative memory  $\mathcal{A}(\cdot ,\omega)$  training continuum  $\mathcal{D}^{train}$  dropping threshold  $\mu$  optimizer OPT, forgetting frequency  $R$  total number of tasks  $T$    
1:  $\mathcal{M}_t\gets \{\},\forall t = 1,2,\dots ,T$  ▷ Initialize episodic memory   
2: for  $t = 1$  to  $T$  do   
3:  $\tilde{\mathcal{M}}_k\gets \mathrm{OPT}_{\mathbf{x}}(\mathbf{x},\mathcal{M}_{k <   t},\boldsymbol {\omega}),\forall k <   t$  as Eq.2  $\triangleright$  Associative memory read   
4: for  $\mathcal{B}_t\sim \mathcal{D}_t^{\textit{train}}$  do   
5:  $\theta \leftarrow \mathrm{OPT}_{\theta}(\theta ,\mathcal{B}_t,\tilde{\mathcal{M}}_{k <   t})$  as Eq.2  $\triangleright$  Train the classifier   
6: for  $(x,y)\in \mathcal{B}_t$  do   
7:  $A = g(x)$    
8:  $A^{\prime} = \mathrm{TR}_{H,W,K}\big(\mathbb{1}\{| \alpha^{c}| > Q_{\mu}\} \otimes \mathbf{J}_{H,W}\big)\odot A$ $\triangleright$  Channel-wise sparsity   
9:  $\mathcal{M}_t\gets \mathcal{M}_t\cup (A',y)$ $\triangleright$  Update episodic memory   
10: end for   
11: end for   
12:  $\omega \gets \mathrm{OPT}_{\omega}(\omega ,A_t)$  as Eq.7  $\triangleright$  Associative memory write   
13: if  $t \% R == 0$  then   
14: Bayesian Training by  $\omega \leftarrow \mathrm{OPT}_{\omega}(\omega)$ $\triangleright$  Associative memory forgetting   
15: end if   
16: end for

memories has been rekindled, and several works have been proposed to achieve strong memory recall performance. However, we notice that how to leverage associative memory in the continual learning setting is under-explored.

In this work, our goal is to design a neuro-inspired continual learning method to improve memory efficiency and mitigate forgetting. To this end, associative memory becomes a natural fit for us due to its properties such as content-based retrieval, fast and efficient recall, and high noise tolerance. Content-based retrieval and noise tolerance allow us to increase the sparsity of the masked feature maps and achieve maximal memory saving. The fast recall reduces the computational overhead for memory retrieval which is critical for our method to be applied to various replay-based baselines.

Formally, an associative memory  $\mathcal{A}(\mathbf{x},\omega)$  can be implemented as a recurrent or feed-forward neural network, where  $\mathbf{x}$  and  $\omega$  denote the input and model parameters of the associative memory, respectively. Corresponding to the "memorize" and "recall" in the human brain, associative memory has read and write operations which are implemented based on an energy function. For example, the predictive-coding-based energy function Rao & Ballard (1999) is the sum of prediction errors across all network layers, i.e.,

$$
E \left(\mathbf {x} _ {0: L}, \boldsymbol {\omega} _ {0: L}\right) = \left\| \mathbf {x} _ {L} - \boldsymbol {\omega} _ {L} \right\| _ {2} ^ {2} + \lambda \sum_ {\ell = 0} ^ {L - 1} \left\| \mathbf {x} _ {\ell} - \mathcal {A} _ {\ell} \left(\mathbf {x} _ {\ell + 1}, \boldsymbol {\omega} _ {\ell}\right) \right\| _ {2} ^ {2}, \tag {5}
$$

where  $\ell$  is the layer index and  $\lambda$  is a coefficient. During training, we write the ground-truth feature map  $A$  into associative memory, by minimizing Eq. 5 w.r.t. parameter  $\omega$  while keeping the input  $\mathbf{x} = A$ . During inference, given the masked feature map  $A'$  defined in Eq. 4, we retrieve the ground-truth  $A$  for memory replay, by minimizing Eq. 5 w.r.t. input  $\mathbf{x}$  initialized as  $A'$  while keeping the parameter  $\omega$  fixed.

# 4.3 TRAINING PIPELINE

Our proposed method involves training the classifier  $f_{\theta}$  and the associative memory  $\mathcal{A}$  while maintaining a small episodic memory  $\mathcal{M}$ . The overall training procedure is shown in Algorithm 1.

Training classifier. In each incremental phase, we update the model parameters of the continual learning classifier  $f_{\theta}$  by using the new coming data and the replay samples. The key here is that we use associative memory to retrieve the "complete" feature map and then feed it to the classifier for memory replay. Formally, the training objective of the classifier can be formulated as follows

$$
\theta^ {*} = \operatorname {a r g m i n} _ {\theta} \sum_ {(\mathbf {x}, t, \mathbf {y})} \ell \left(f _ {\boldsymbol {\theta}} (g (\mathbf {x}), t), \mathbf {y}\right) + \sum_ {k <   t} \ell \left(f _ {\boldsymbol {\theta}}, \tilde {\mathcal {M}} _ {k}\right) \tag {6}
$$

where  $\tilde{\mathcal{M}}_k = \mathrm{argmin}_{\mathbf{x}}E\big(\mathbf{x}_{0:L},\boldsymbol{\omega}_{0:L}\big)$  with  $\mathbf{x}_0$  initialized as  $A_k^\prime \in \mathcal{M}_k$

where the first row is the continual learning objective as defined in Eq. 2.  $\tilde{\mathcal{M}}_k$  denotes the retrieved episodic memory, i.e., the feature maps recalled by associative memory. As mentioned earlier, the retrieval by associative memory corresponds to solving an optimization problem as shown in the second row, where  $A_{k}^{\prime}$  is the masked (before retrieval) feature map from task  $k$ .

Training associative memory. Given feature maps coming from new tasks in each incremental, we need to write those feature maps into the associative memory such that we can ask it to retrieve the complete feature map given a partial cue at a later timestamp. Formally, given feature maps  $A_{t}$  from new task  $t$ , writing them into associative memory corresponds to solving the following optimization problem

$$
\omega = \operatorname {a r g m i n} _ {\omega} E \left(\mathbf {x} _ {0: L}, \omega_ {0: L}\right) \text {w i t h} \mathbf {x} _ {0} \text {f i x e d a s} A _ {t}. \tag {7}
$$

We also proposed a memory-forgetting mechanism for associative memory to avoid potential memory overload during continual learning. As more data are observed when new tasks keep coming, it is natural for the classifier and associative memory to be biased more towards new tasks' data rather than old ones. To this end, our forgetting mechanism erases old tasks' data more times than new ones thus satisfying the inductive bias we want to introduce.

Update Episodic Memory. For simplicity, we assume that the memory is populated with the last  $m$  examples from each task, although better memory update strategies could be employed (such as building a coreset per task, reservoir sampling, etc.)

# 5 EXPERIMENT

In this section, we evaluate our proposed method SHARC on Class-IL and Task-IL CL. Both performance tables and learning curves over entire tasks are provided. In addition, we demonstrate sensitivity analyses over the masking threshold and comparison of different associative memories. All experiments are conducted on a 64-bit machine with an NVIDIA T4 Tensor Core GPU which has 320 Turing Tensor cores, 2560 CUDA cores, 16GB memory, and Intel® Xeon® Platinum 8259CL CPU @ 2.50GHz. The anonymous code of our method can be found here.

# 5.1 EXPERIMENT SETTING

Dataset. In our research, we conducted experiments on three datasets: Split CIFAR-10, Split CIFAR-100, and Split mini-ImageNet Chaudhry et al. (2019). CIFAR-10 consists of 50,000 RGB training images and 10,000 test images, categorized into 10 object classes. Similarly, CIFAR-100 extends this classification task by including 100 classes, with each class containing 600 images. ImageNet-50 comprises 50 classes with 1300 training images and 50 validation images per class. We divided the Split CIFAR-10 dataset into 5 tasks, each with 2 classes. For Split CIFAR-100 and Split mini-ImageNet, we expanded our investigation to 20 tasks, each with 5 classes.

Comparison Methods. We compare our method with several replay-based continual learning methods, including: ER, a rehearsal-based method that utilizes the average of parameter update gradients from the current task's samples alongside samples from episodic memory to update the learning agent. MER, a rehearsal-based model that harnesses the power of episodic memory Riemer et al. (2018). GEM, one ensures that valuable information from prior tasks is retained while accommodating new learning Lopez-Paz & Ranzato (2017). A-GEM, takes a step further than GEM by incorporating an adaptive mechanism that updates the model's parameters based on both the current task's gradient and the gradients of previous tasks stored in the episodic memory. CLS-ER, an innovative algorithm that utilizes a dual-memory learning mechanism to enhance performance in continual learning tasks Arani et al. (2022). DER++, a combination of rehearsal, knowledge distillation, and regularization techniques Buzzega et al. (2020).

Evaluation Metrics. We assess the classification performance using the ACC metric, which represents the average test classification accuracy across all tasks. We also measure backward transfer (BWT Lopez-Paz & Ranzato (2017)) to evaluate the impact of new learning on previous knowledge. Negative BWT indicates forgetting, so a higher value is preferable. Detailed experimental settings can be found in the appendix.

Training Details. We utilized a frozen pre-trained model ImageNet-1K, retaining only the MLP part for training. We directly employed the feature map as the input and output of the associative memory, storing it in a memory buffer. For further details, please refer to the appendix.

Table 1: Performance comparison on image classification datasets (Task-IL). The mean and standard deviation are calculated based on five runs with varying seeds.  ${}^{ + }$  denotes the corresponding method combined with our SHARC framework. In most cases, our proposed SHARC framework significantly improves the method.  

<table><tr><td rowspan="2">Buffer</td><td rowspan="2">Model</td><td colspan="2">S-CIFAR-10</td><td colspan="2">S-CIFAR-100</td><td colspan="2">S-Mini-ImgNet</td></tr><tr><td>ACC (↑)</td><td>BWT (↑)</td><td>ACC (↑)</td><td>BWT (↑)</td><td>ACC (↑)</td><td>BWT (↑)</td></tr><tr><td>-</td><td>JOINT</td><td>93.49 ± 0.61</td><td>43.14 ± 2.07</td><td>87.57 ± 0.89</td><td>67.99 ± 1.53</td><td>74.95 ± 0.7</td><td>70.02 ± 0.81</td></tr><tr><td>-</td><td>SGD</td><td>92.31 ± 0.54</td><td>-0.38 ± 0.82</td><td>85.83 ± 0.35</td><td>3.08 ± 2.14</td><td>76.2 ± 0.41</td><td>3.98 ± 0.75</td></tr><tr><td rowspan="12">200</td><td>GEM</td><td>88.44 ± 1.11</td><td>-4.6 ± 2.24</td><td>82.82 ± 0.62</td><td>0.2 ± 1.69</td><td>72.23 ± 1.26</td><td>-1 ± 1.82</td></tr><tr><td>GEM+</td><td>91.01 ± 1.04</td><td>-1.5 ± 1.31</td><td>83.88 ± 0.52</td><td>-0.04 ± 1.2</td><td>76.13 ± 0.98</td><td>3.49 ± 1.47</td></tr><tr><td>A-GEM</td><td>90.52 ± 3.29</td><td>-1.07 ± 1.57</td><td>85.33 ± 0.58</td><td>2 ± 1</td><td>75.18 ± 1.11</td><td>2.33 ± 1.5</td></tr><tr><td>A-GEM+</td><td>91.72 ± 1.02</td><td>-0.35 ± 2.08</td><td>85.55 ± 0.88</td><td>1.25 ± 0.63</td><td>76.54 ± 0.97</td><td>4.14 ± 1.71</td></tr><tr><td>ER</td><td>86.36 ± 1.33</td><td>-5.04 ± 1.72</td><td>82.55 ± 0.47</td><td>-0.71 ± 1.59</td><td>71.66 ± 1.44</td><td>-1.53 ± 2.04</td></tr><tr><td>ER+</td><td>91.48 ± 1.18</td><td>-0.93 ± 1.54</td><td>84.55 ± 0.62</td><td>0.71 ± 0.61</td><td>73.68 ± 0.59</td><td>0.71 ± 0.97</td></tr><tr><td>MER</td><td>87.32 ± 1.39</td><td>-2.3 ± 3.83</td><td>82.04 ± 0.63</td><td>-0.83 ± 1.49</td><td>71.2 ± 1.43</td><td>-1.99 ± 1.53</td></tr><tr><td>MER+</td><td>91.14 ± 1.62</td><td>-0.9 ± 2.46</td><td>84.3 ± 0.92</td><td>0.41 ± 1.26</td><td>73.54 ± 0.58</td><td>0.7 ± 1.06</td></tr><tr><td>DER++</td><td>84.94 ± 1.95</td><td>-6.45 ± 1.91</td><td>83.27 ± 0.76</td><td>0.32 ± 1.47</td><td>72.92 ± 1.09</td><td>-0.13 ± 1.44</td></tr><tr><td>DER+++</td><td>89.89 ± 1.34</td><td>-2.72 ± 2.04</td><td>84.96 ± 0.97</td><td>0.62 ± 1.44</td><td>74.59 ± 0.87</td><td>2 ± 1.27</td></tr><tr><td>CLS-ER</td><td>80.97 ± 2.11</td><td>-12.6 ± 4.39</td><td>82.97 ± 0.32</td><td>-1.95 ± 1.5</td><td>73.67 ± 1.05</td><td>-1.75 ± 0.4</td></tr><tr><td>CLS-ER+</td><td>91.39 ± 0.7</td><td>-0.94 ± 1.18</td><td>85 ± 0.41</td><td>1.27 ± 0.68</td><td>77 ± 0.45</td><td>2.7 ± 0.97</td></tr><tr><td rowspan="12">500</td><td>GEM</td><td>88.02 ± 2.61</td><td>-3.84 ± 1.19</td><td>82.81 ± 0.66</td><td>0.06 ± 1.66</td><td>73.6 ± 1.13</td><td>0.23 ± 1.27</td></tr><tr><td>GEM+</td><td>91.53 ± 1.17</td><td>-0.05 ± 1.58</td><td>84.37 ± 1.03</td><td>1.63 ± 0.79</td><td>75.56 ± 0.93</td><td>3.2 ± 1.61</td></tr><tr><td>A-GEM</td><td>90.81 ± 2.97</td><td>0.31 ± 2.81</td><td>85.44 ± 0.28</td><td>2.32 ± 0.84</td><td>75.59 ± 1.15</td><td>2.78 ± 1.61</td></tr><tr><td>A-GEM+</td><td>92.32 ± 0.67</td><td>0.22 ± 0.7</td><td>85.89 ± 0.82</td><td>3.25 ± 1.01</td><td>75.65 ± 0.86</td><td>3.21 ± 1.56</td></tr><tr><td>ER</td><td>88.05 ± 1.51</td><td>-1.88 ± 3.62</td><td>82.7 ± 0.63</td><td>-0.09 ± 0.5</td><td>71.83 ± 1.17</td><td>-1.36 ± 1.5</td></tr><tr><td>ER+</td><td>91.64 ± 0.66</td><td>-0.82 ± 0.67</td><td>84.77 ± 1.57</td><td>1.85 ± 1.72</td><td>72.94 ± 0.63</td><td>0.41 ± 1.08</td></tr><tr><td>MER</td><td>88.33 ± 1.87</td><td>-3.35 ± 1.6</td><td>82.11 ± 0.5</td><td>-0.36 ± 0.91</td><td>70.69 ± 1.07</td><td>-2.28 ± 1.41</td></tr><tr><td>MER+</td><td>91.69 ± 0.73</td><td>-0.39 ± 1.56</td><td>84.06 ± 1.33</td><td>1.43 ± 1.46</td><td>72.63 ± 0.37</td><td>-0.28 ± 1.08</td></tr><tr><td>DER++</td><td>86.73 ± 2.77</td><td>-4.79 ± 1.26</td><td>83.04 ± 0.58</td><td>0.76 ± 1.64</td><td>72.05 ± 0.87</td><td>-0.95 ± 1.52</td></tr><tr><td>DER+++</td><td>90.46 ± 1.3</td><td>-2.71 ± 1.01</td><td>85.13 ± 1.57</td><td>1.81 ± 1.01</td><td>73.85 ± 0.79</td><td>1.5 ± 1.5</td></tr><tr><td>CLS-ER</td><td>82.54 ± 3.06</td><td>-8.64 ± 5.52</td><td>81.34 ± 0.9</td><td>-2.27 ± 1.8</td><td>72.11 ± 0.38</td><td>-3.41 ± 0.88</td></tr><tr><td>CLS-ER+</td><td>90.94 ± 1.49</td><td>-2.22 ± 1.51</td><td>85.36 ± 0.83</td><td>1.82 ± 1.66</td><td>76.27 ± 0.52</td><td>2.05 ± 0.82</td></tr></table>

![](images/c7bfa8c52ac22926120a1aff12d9dd6f5be5d31f58fc797ee48271ab02c3c307.jpg)  
(a) Task-IL S-CIFAR10

![](images/8780e34745613d9cc8483b63f18be378db0ee94f70c77c9669600016555cd85f.jpg)  
Figure 3: Learning curves of multiple models with/without SHARC on S-CIFAR10 and S-CIFAR100 in Task-IL scenario. Models with/without SHARC are shown in solid/dotted lines. The buffer size for all models is 200. Methods combined with our proposed framework SHARC significantly prevails.  
(b) Task-IL S-CIFAR100

# 5.2 PERFORMANCE COMPARISON

We demonstrate the impact of our proposed SHARC framework on several state-of-the-art replay-based approaches. Naive baselines such as SGD refer to standard training, while JOINT refers to joint training on all tasks, which provides an upper bound. The experimental results are shown in Table 1 and Table 2, which contain the results in the Task-IL scenario and Class-IL scenario, respectively. Buffer size controls the budget of episodic memory and is distributed evenly to all tasks. Each slot in the buffer contains feature maps of a sample, instead of an image.

Table 1 compares six replay-based methods before and after combining them with SHARC in the Task-IL scenario. Overall, in most cases, the methods used in conjunction with SHARC offer significant improvements. Such contrast exists in all settings (different datasets, models, and buffer sizes). In particular, CLS-ER equipped with SHARC achieves a  $12.9\%$  improvement in ACC on

Table 2: Performance comparison on image classification datasets (Class-IL). The mean and standard deviation are calculated based on five runs with varying seeds.  ${}^{ + }$  denotes the corresponding method combined with our SHARC framework. In most cases, our proposed SHARC framework significantly improves the method.  

<table><tr><td rowspan="2">Buffer</td><td rowspan="2">Model</td><td colspan="2">S-CIFAR-10</td><td colspan="2">S-CIFAR-100</td><td colspan="2">S-Mini-ImgNet</td></tr><tr><td>ACC (↑)</td><td>BWT (↑)</td><td>ACC (↑)</td><td>BWT (↑)</td><td>ACC (↑)</td><td>BWT (↑)</td></tr><tr><td>-</td><td>JOINT</td><td>72.85 ± 2.18</td><td>61.26 ± 8.55</td><td>45.87 ± 1.22</td><td>45.55 ± 1.45</td><td>47.08 ± 0.77</td><td>46.25 ± 0.98</td></tr><tr><td>-</td><td>SGD</td><td>20.47 ± 0.78</td><td>-90.16 ± 0.92</td><td>8.55 ± 1.39</td><td>-78.24 ± 0.93</td><td>12.21 ± 0.75</td><td>-67.11 ± 0.77</td></tr><tr><td rowspan="12">200</td><td>GEM</td><td>27.01 ± 6.16</td><td>-76.88 ± 6.93</td><td>16.38 ± 3.06</td><td>-67.72 ± 3.84</td><td>23.76 ± 2.65</td><td>-54.22 ± 3.47</td></tr><tr><td>GEM+</td><td>36.39 ± 4.04</td><td>-63.03 ± 7.86</td><td>20.77 ± 3.22</td><td>-63.08 ± 3.33</td><td>24.76 ± 1.86</td><td>-52.31 ± 2.62</td></tr><tr><td>A-GEM</td><td>24.12 ± 6.92</td><td>-83.48 ± 3.5</td><td>12.75 ± 4.2</td><td>-73.96 ± 4.68</td><td>16.77 ± 2.05</td><td>-62.51 ± 2.71</td></tr><tr><td>A-GEM+</td><td>29.19 ± 2.31</td><td>-77.82 ± 3.52</td><td>17.55 ± 1.91</td><td>-69.61 ± 1.6</td><td>18.96 ± 1.53</td><td>-60.02 ± 2.11</td></tr><tr><td>ER</td><td>29.35 ± 7.79</td><td>-65.76 ± 10.54</td><td>14.81 ± 2.75</td><td>-70.15 ± 3.41</td><td>21.71 ± 1.61</td><td>-56.46 ± 1.88</td></tr><tr><td>ER+</td><td>33.94 ± 6.24</td><td>-62 ± 10.91</td><td>20.58 ± 1.86</td><td>-63.09 ± 1.69</td><td>23.36 ± 1.76</td><td>-54.24 ± 2.46</td></tr><tr><td>MER</td><td>30.02 ± 7.64</td><td>-61.33 ± 7.94</td><td>13.74 ± 3.33</td><td>-70.76 ± 3.48</td><td>21.46 ± 1.78</td><td>-56.59 ± 1.89</td></tr><tr><td>MER+</td><td>34.81 ± 5.44</td><td>-59.86 ± 8.47</td><td>19.46 ± 1.3</td><td>-64.09 ± 1.96</td><td>22.83 ± 1.88</td><td>-54.68 ± 2.46</td></tr><tr><td>DER++</td><td>31.55 ± 4.61</td><td>-55.68 ± 10.28</td><td>14.44 ± 6.09</td><td>-69.28 ± 6.37</td><td>23.03 ± 1.64</td><td>-54.54 ± 1.74</td></tr><tr><td>DER+++</td><td>38.53 ± 4.84</td><td>-39.15 ± 7.8</td><td>21.01 ± 1.8</td><td>-62.08 ± 2.37</td><td>24.76 ± 1.3</td><td>-52.16 ± 1.58</td></tr><tr><td>CLS-ER</td><td>27.3 ± 3.13</td><td>-56.75 ± 14.14</td><td>15.83 ± 1.84</td><td>-70.13 ± 2.49</td><td>21.77 ± 1.43</td><td>-58.55 ± 2.09</td></tr><tr><td>CLS-ER+</td><td>28.63 ± 5.43</td><td>-51.37 ± 8.46</td><td>18.77 ± 1.93</td><td>-64.2 ± 1.36</td><td>22.86 ± 1.81</td><td>-57.38 ± 2.31</td></tr><tr><td rowspan="12">500</td><td>GEM</td><td>30.12 ± 10.19</td><td>-63.83 ± 14.88</td><td>20.81 ± 5.66</td><td>-59.13 ± 6.09</td><td>30.88 ± 2.39</td><td>-45.43 ± 3.04</td></tr><tr><td>GEM+</td><td>33.52 ± 5.29</td><td>-51.17 ± 14.8</td><td>24.89 ± 1.19</td><td>-54.14 ± 3.14</td><td>29.8 ± 1.76</td><td>-46.46 ± 2.48</td></tr><tr><td>A-GEM</td><td>25 ± 7.05</td><td>-80.94 ± 2.84</td><td>12.25 ± 3.87</td><td>-74.35 ± 4.45</td><td>17.04 ± 0.91</td><td>-62.21 ± 1.11</td></tr><tr><td>A-GEM+</td><td>29.21 ± 4.61</td><td>-77.99 ± 5.2</td><td>15.02 ± 2.02</td><td>-71.12 ± 2.31</td><td>17.33 ± 1.48</td><td>-61.45 ± 2.16</td></tr><tr><td>ER</td><td>33.16 ± 8.18</td><td>-55.8 ± 12.65</td><td>19.37 ± 3.9</td><td>-61.39 ± 3.89</td><td>27.39 ± 1.52</td><td>-49.06 ± 2.18</td></tr><tr><td>ER+</td><td>36.18 ± 2.22</td><td>-53.67 ± 6.82</td><td>24.85 ± 1.41</td><td>-53.85 ± 1.38</td><td>26.69 ± 0.95</td><td>-49.48 ± 1.61</td></tr><tr><td>MER</td><td>32.96 ± 8.35</td><td>-53.78 ± 14.59</td><td>20.51 ± 3.97</td><td>-58.51 ± 3.99</td><td>26.46 ± 1.83</td><td>-49.5 ± 2.45</td></tr><tr><td>MER+</td><td>36.04 ± 2.98</td><td>-51.5 ± 2.79</td><td>23.4 ± 1.52</td><td>-53.89 ± 1.8</td><td>26.1 ± 1.22</td><td>-50.08 ± 1.35</td></tr><tr><td>DER++</td><td>34.21 ± 8.62</td><td>-43.95 ± 16.78</td><td>16.95 ± 5.86</td><td>-63.53 ± 5.67</td><td>26.48 ± 1.67</td><td>-49.82 ± 2.39</td></tr><tr><td>DER+++</td><td>30.57 ± 8.35</td><td>-45.5 ± 8</td><td>24.66 ± 1.83</td><td>-54.62 ± 2.09</td><td>25.54 ± 2.78</td><td>-50.72 ± 3.45</td></tr><tr><td>CLS-ER</td><td>26.97 ± 3.32</td><td>-41.42 ± 5.8</td><td>22.1 ± 2.04</td><td>-58.79 ± 3.31</td><td>26.74 ± 1.49</td><td>-52.2 ± 1.73</td></tr><tr><td>CLS-ER+</td><td>29.64 ± 5.37</td><td>-77.81 ± 5.74</td><td>22.5 ± 2.52</td><td>-57.04 ± 3.65</td><td>26.31 ± 1.55</td><td>-52.68 ± 0.55</td></tr></table>

![](images/3d011ac5022b20265833aa776a6fd31e9770dd0758097a1456caa662dfb7be26.jpg)  
Figure 4: Learning curves of multiple models with/without SHARC on S-CIFAR10 and S-CIFAR100 in Class-IL scenario. Models with/without SHARC are shown in solid/dotted lines. The buffer size for all models is 200.

![](images/218790098869d03e805dab31799fae7b910e5d125e8f6061040d89cf8321ed6d.jpg)

S-CIFAR10 with buffer size 200. From a methodological perspective, rehearsal-based methods (e.g., ER) offer greater improvements than constraint-based methods (e.g., GEM). As a typical example, the performance of A-GEM improves only slightly when used with SHARC on S-CIFAR100, which is reasonable since we keep the batch size of the retrieval process constant. Rehearsal-based methods can benefit more from masking because masking reduces the memory space for samples, allowing more previous samples to be reviewed. Furthermore, in most cases on S-CIFAR100 and S-MiniImgNet, the BWT increases or even becomes positive when using SHARC, indicating that SHARC is highly resistant to forgetting. As the buffer size decreases, the complexity of the task increases. Achieving good performance with smaller buffer sizes is the spirit of continual learning. Based on this consideration, we further investigate the learning curve for a minimum buffer size of 200. As shown in Figure 4, methods equipped with SHARC clearly prevail in the figure, indicating that they have been steadily improved during the learning process.

![](images/d15eb74abe4189987b2572c0822bbfec4e8444c08f743a6492d3fa0d48782fc9.jpg)  
(a) Class-IL AM

![](images/629e2f32d42a27231c90d625ab769d06da8cc5338f582ea8cf0fd5ab7225cea4.jpg)  
(b) Task-IL AM

![](images/ad9055f1cf5f0fad5b0c4e462f91d67767474133679c8018e1249148528e50b2.jpg)  
Figure 5: Average accuracy of GEM and ER using different associative memories for inpainting on S-CIFAR100. Figure (a) shows ACC of GEM using different associative memories in Class-IL scenario. Figure (b) shows ACC of ER using different associative memories in Task-IL scenario. The buffer size for all models is 200.  
(c) Class-IL SA

![](images/95bd42cc4fda0eafbb62a411217ec579ce90514b05aa2593b5a5afb7d9f493b0.jpg)  
(d) Task-IL SA

Since the corrupted feature maps after masking cannot be used for backpropagation, in order to keep the backbone network the same as the rest of the work, we froze the convolutional basis of the pre-trained ResNet18, leaving only the parameters of the fully connected layer available for training. A single-layer classifier may not be sufficient in Class-IL, causing all models to perform poorly. Still, this is enough to illustrate the effectiveness of our proposed SHARC framework. Table 2 compares six replay-based methods before and after combining them with SHARC in a Class-IL scenario. Overall, in most cases, the methods used in conjunction with SHARC offer significant improvements. In particular, DER++ equipped with SHARC achieves a  $45.5\%$  improvement in ACC on S-CIFAR100. The smaller the buffer, the more pronounced this contrast becomes. In particular, with a buffer size of 200, SHARC improves CLS-ER in ACC much greater compared to the buffer size of 500 on S-CIFAR100.

# 5.3 ASSOCIATIVE MEMORY COMPARISON

As shown in Figure 5(a) and Figure 5(b), we compare different configurations of the associative memory. We in general follow Yoo & Wood (2022) for the implementation. We found that Modern Hopfield Network Ramsauer et al. (2020) favors more towards the task-incremental setting while BayesPCN Yoo & Wood (2022) favors more towards class-incremental setting. This is potentially due to that, in BayesPCN a forgetting mechanism is implemented, which can help mitigate the memory overload of the associative memory when the samples to memorize are too many. Such design seems to play a more important role in the more challenging class-incremental setting.

# 5.4 MASKING THRESHOLD SENSITIVITY ANALYSIS

We conduct sensitivity analysis on the threshold  $Q_{\mu}$  in Eq 4. Masking threshold  $Q_{\mu}$  is defined as certain percentile value for 512 feature importance. Features with importance below the threshold will be masked.  $Q_{\mu}$  is controlled by a hyper-parameter Theta. Theta ranges between (0, 1], where 1 means that only the features with the highest importance are retained. While different methods may favor different optimal thresholds, a general sensitivity analysis is still helpful in determining optimal threshold settings. As shown in Figure 5(d), the optimal Theta in Task-IL is between 0.8 to 0.9. As shown in Figure 5(c), the optimal Theta in Class-IL is between 0.3 to 0.5. This indicates that the optimal thresholds show different trends in Task-IL and Class-IL. In particular, Task-IL requires a higher threshold to drop most features, while the opposite is true for Class-IL. This is reasonable because in Task-IL, the task ID is given as additional information. Whereas in Class-IL, the model can only get additional information from more features.

# 6 CONCLUSION

We propose SHARC, a novel framework that bridges the gap between current AI models and humans in continual learning. Combining associative memory and interpretive techniques, SHARC enables efficient, near-perfect recall of seen samples in a human-like manner. As a generic framework, SHARC can be seamlessly adapted to any replay-based approach, thus improving their performance in different continual learning scenarios. We demonstrate the effectiveness of our framework with abundant experimental results. Our proposed SHARC framework consistently improves several SOTA replay-based methods on multiple benchmark datasets.

# REFERENCES

Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. Gradient based sample selection for online continual learning. arXiv preprint arXiv:1903.08671, 2019.  
Daniel J Amit and Daniel J Amit. Modeling brain function: The world of attractor neural networks. Cambridge university press, 1989.  
Elahe Arani, Fahad Sarfraz, and Bahram Zonooz. Learning fast, learning slow: A general continual learning method based on complementary learning system. arXiv preprint arXiv:2201.12604, 2022.  
Guangji Bai, Chen Ling, Yuyang Gao, and Liang Zhao. Saliency-augmented memory completion for continual learning. In Proceedings of the 2023 SIAM International Conference on Data Mining (SDM), pp. 244-252. SIAM, 2023.  
Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. Advances in neural information processing systems, 33:15920-15930, 2020.  
Arslan Chaudhry, Marc'Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with a-gem. arXiv preprint arXiv:1812.00420, 2018.  
Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr, and Marc'Aurelio Ranzato. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486, 2019.  
Sayna Ebrahimi, Suzanne Petryk, Akash Gokul, William Gan, Joseph E Gonzalez, Marcus Rohrbach, and Trevor Darrell. Remembering for the right reasons: Explanations reduce catastrophic forgetting. Applied AI letters, 2(4):e44, 2021.  
Omar Elharrouss, Noor Almaadeed, Somaya Al-Maadeed, and Younes Akbari. Image inpainting: A review. Neural Processing Letters, 51:2007-2028, 2020.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139-144, 2020.  
Tyler L Hayes, Kushal Kafle, Robik Shrestha, Manoj Acharya, and Christopher Kanan. Remind your neural network to prevent catastrophic forgetting. In European Conference on Computer Vision, pp. 466-483. Springer, 2020.  
Donald Hebb. The organization of behavior. New York, 1949.  
John J Hopfield. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the national academy of sciences, 79(8):2554-2558, 1982.  
Yanping Huang and Rajesh PN Rao. Predictive coding. Wiley Interdisciplinary Reviews: Cognitive Science, 2(5):580-593, 2011.  
R Insausti, MP Marcos, A Mohedano-Moriano, MM Arroyo-Jimenez, M Córcoles-Parada, E Artacho-Pérula, MM Ubero-Martinez, and M Munoz-Lopez. The nonhuman primate hippocampus: neuroanatomy and patterns of cortical connectivity. The hippocampus from cells to systems: Structure, connectivity, and functional contributions to memory and flexible cognition, pp. 3-36, 2017.  
Daoyun Ji and Matthew A Wilson. Coordinated memory replay in the visual cortex and hippocampus during sleep. Nature neuroscience, 10(1):100-107, 2007.  
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114 (13):3521-3526, 2017.  
Dmitry Krotov and John Hopfield. Large associative memory problem in neurobiology and machine learning. arXiv preprint arXiv:2008.06996, 2020.

Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935-2947, 2017.  
David Lopez-Paz and Marc'Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30:6467-6476, 2017.  
Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of learning and motivation, volume 24, pp. 109-165. Elsevier, 1989.  
German I Parisi, Ronald Kemker, Jose L Part, Christopher Kanan, and Stefan Wermter. Continual lifelong learning with neural networks: A review. Neural Networks, 113:54-71, 2019.  
Jialun Peng, Dong Liu, Songcen Xu, and Houqiang Li. Generating diverse structure for image inpainting with hierarchical vq-vae. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10775–10784, 2021.  
Yunchen Pu, Zhe Gan, Ricardo Henao, Xin Yuan, Chunyuan Li, Andrew Stevens, and Lawrence Carin. Variational autoencoder for deep learning of images, labels and captions. Advances in neural information processing systems, 29:2352-2360, 2016.  
Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber, Markus Holzleitner, Milena Pavlovic, Geir Kjetil Sandve, et al. Hopfield networks is all you need. arXiv preprint arXiv:2008.02217, 2020.  
Rajesh PN Rao and Dana H Ballard. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature neuroscience, 2(1):79-87, 1999.  
Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 2001-2010, 2017.  
Matthew Riemer, Ignacio Cases, Robert Ajemian, Miao Liu, Irina Rish, Yuhai Tu, and Gerald Tesauro. Learning to learn without forgetting by maximizing transfer and minimizing interference. arXiv preprint arXiv:1810.11910, 2018.  
Anthony Robins. Catastrophic forgetting, rehearsal and pseudorehearsal. _Connection Science_, 7(2): 123-146, 1995.  
Edmund T Rolls. The mechanisms for pattern completion and pattern separation in the hippocampus. Frontiers in systems neuroscience, 7:74, 2013.  
Gobinda Saha and Kaushik Roy. Saliency guided experience packing for replay in continual learning. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 5273-5283, 2023.  
Tommaso Salvatori, Yuhang Song, Yujuan Hong, Lei Sha, Simon Frieder, Zhenghua Xu, Rafal Bogacz, and Thomas Lukasiewicz. Associative memories via predictive coding. Advances in Neural Information Processing Systems, 34:3874-3886, 2021.  
Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pp. 618-626, 2017.  
Yi Sun, Xiaogang Wang, and Xiaou Tang. Deeply learned face representations are sparse, selective, and robust. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2892-2900, 2015.  
Timothy J Teyler and Jerry W Rudy. The hippocampal indexing theory and episodic memory: updating the index. Hippocampus, 17(12):1158-1169, 2007.  
Jinsoo Yoo and Frank Wood. Bayespcn: A continually learnable predictive coding associative memory. Advances in Neural Information Processing Systems, 35:29903-29914, 2022.  
Wang Zhou, Shiyu Chang, Norma Sosa, Hendrik Hamann, and David Cox. Lifelong object detection. arXiv preprint arXiv:2009.01129, 2020.
