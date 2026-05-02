# META-LEARNING WITH LATENT EMBEDDING OPTIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Gradient-based meta-learning techniques are both widely applicable and proficient at solving challenging few-shot learning and fast adaptation problems. However, they have practical difficulties when operating on high-dimensional parameter spaces in extreme low-data regimes. We show that it is possible to bypass these limitations by learning a data-dependent latent generative representation of model parameters, and performing gradient-based meta-learning in this low-dimensional latent space. The resulting approach, latent embedding optimization (LEO), decouples the gradient-based adaptation procedure from the underlying high-dimensional space of model parameters. Our evaluation shows that LEO can achieve state-of-the-art performance on the competitive miniImageNet and tieredImageNet few-shot classification tasks. Further analysis indicates LEO is able to capture uncertainty in the data, and can perform adaptation more effectively by optimizing in latent space.

# 1 INTRODUCTION

Humans have a remarkable ability to quickly grasp new concepts from a very small number of examples or a limited amount of experience, leveraging prior knowledge and context. In contrast, traditional deep learning approaches (LeCun et al., 2015; Schmidhuber, 2015) treat each task independently and hence are often data inefficient - despite providing significant performance improvements across the board, such as for image classification (Simonyan & Zisserman, 2014; He et al., 2016), reinforcement learning (Mnih et al., 2015; Silver et al., 2017), and machine translation (Cho et al., 2014; Sutskever et al., 2014). Just as humans can efficiently learn new tasks, it is desirable for learning algorithms to quickly adapt to and incorporate new and unseen information.

Few-shot learning tasks challenge models to learn a new concept or behaviour with very few examples or limited experience (Fei-Fei et al., 2006; Lake et al., 2011). One approach to address this class of problems is meta-learning, a broad family of techniques focused on learning how to learn or to quickly adapt to new information. More specifically, optimization-based meta-learning approaches (Ravi & Larochelle, 2017; Finn et al., 2017) aim to find a single set of model parameters that can be adapted with a few steps of gradient descent to individual tasks. However, using only a few samples (typically 1 or 5) to compute gradients in a high-dimensional parameter space could make generalization difficult, especially under the constraint of a shared starting point for task-specific adaptation.

In this work we propose a new approach, named Latent Embedding Optimization (LEO), which learns a low-dimensional latent embedding of model parameters and performs optimization-based meta-learning in this space. Intuitively, the approach provides two advantages. First, the initial parameters for a new task are conditioned on the training data, which enables a task-specific starting point for adaptation. By incorporating a relation network into the encoder, this initialization can better consider the joint relationship between all of the input data. Second, by optimizing in the lower-dimensional latent space, the approach can adapt the behaviour of the model more effectively. Further, by allowing this process to be stochastic, the uncertainties and ambiguities present in the few-shot data regime can be expressed.

We demonstrate that LEO achieves state-of-the-art results on both the miniImageNet and tieredImageNet datasets, and run an ablation study and further analysis to show that both conditional parameter generation and optimization in latent space are critical for the success of the method.

# 2 MODEL

# 2.1 PROBLEM DEFINITION

We define the  $N$ -way  $K$ -shot problem using the episodic formulation of Vinyals et al. (2016). Each task instance  $\mathcal{T}_i$  is a classification problem sampled from a task distribution  $p(\mathcal{T})$ . The tasks are divided into a training meta-set of tasks,  $\mathcal{S}^{tr}$ , a validation meta-set,  $\mathcal{S}^{val}$ , and a test meta-set  $\mathcal{S}^{test}$  each with a disjoint set of target classes (i.e., a class seen during testing is not seen during training). The validation meta-set is used for model selection, and the testing meta-set is used only for final evaluation.

Each task instance  $\mathcal{T}_i\sim p(\mathcal{T})$  is composed of a training set  $\mathcal{D}^{tr}$  and validation set  $\mathcal{D}^{val}$ , and only contains  $N$  classes randomly selected from the appropriate meta-set (e.g. for a task instance in the training meta-set, the classes are a subset of those available in  $\mathcal{S}^{tr}$ ). In most setups, the training set  $\mathcal{D}^{tr} = \left\{\left(\mathbf{x}_n^k,y_n^k\right)\mid k = 1\dots K;n = 1\dots N\right\}$  contains  $K$  samples for each class. The validation set  $\mathcal{D}^{val}$  can contain several other samples from the same classes, providing an estimate of generalization performance on the  $N$  classes for this problem instance. We note that the validation set of a problem instance  $\mathcal{D}^{val}$  (used to optimize a meta-learning objective) should not be confused with the held-out validation meta-set  $\mathcal{S}^{val}$  (used for model selection).

# 2.2 MODEL-AGNOSTIC META-LEARNING

Model-agnostic meta-learning (MAML) (Finn et al., 2017) is an approach to optimization-based meta-learning that is related to our work. For some parametric model  $f_{\theta}$ , MAML aims to find a single set of parameters  $\theta$  which, using a few optimization steps, can be successfully adapted to any novel task sampled from the same distribution. For a particular task instance  $\mathcal{T}_i = (\mathcal{D}^{tr},\mathcal{D}^{val})$ , the parameters are adapted to task-specific model parameters  $\theta_i'$  by applying some differentiable function, typically an update rule of the form:

$$
\theta_ {i} ^ {\prime} = \mathcal {G} (\theta , \mathcal {D} ^ {t r}), \tag {1}
$$

where  $\mathcal{G}$  is typically implemented as a step of gradient descent on the few-shot training set  $\mathcal{D}^{tr}$ ,  $\theta_{i}^{\prime} = \theta -\alpha \nabla_{\theta}\mathcal{L}_{T_{i}}^{tr}(f_{\theta})$ . Generally, multiple sequential adaptation steps can be applied. During meta-training, the parameters  $\theta$  are updated by back-propagating through the adaptation procedure, in order to reduce errors on the validation set  $\mathcal{D}^{val}$ :

$$
\theta \leftarrow \theta - \eta \nabla_ {\theta} \sum_ {\mathcal {T} _ {i} \sim p (\mathcal {T})} \mathcal {L} _ {\mathcal {T} _ {i}} ^ {v a l} \left(f _ {\theta_ {i} ^ {\prime}}\right) \tag {2}
$$

The learning rate  $\alpha$  can also be meta-learned concurrently, in which case we refer to this algorithm as Meta-SGD (Li et al., 2017).

The approach includes the main ingredients of optimization-based meta-learning with neural networks: initialization is done by maintaining an explicit set of model parameters  $\theta$ ; the adaptation procedure, or "inner loop", takes parameters  $\theta$  as input and returns parameters  $\theta_i'$  adapted specifically for task instance  $\mathcal{T}_i$ , by using gradient descent iteratively with some learning rate schedule (Eq. 1); and termination, which is handled simply by choosing a fixed number of optimization steps in the "inner loop". MAML updates  $\theta$  by differentiating through the "inner loop" in order to minimize errors of instance-specific adapted models  $f_{\theta_i'}$  on the corresponding validation set (Eq. 2). We refer to this process as the "outer loop" of meta-learning.

In the next section we use the same procedural stages to describe fast adaptation by Latent Embedding Optimization (LEO).

# 2.3 LATENT EMBEDDING OPTIMIZATION FOR META-LEARNING

The primary contribution of this paper is to show that it is possible, and indeed beneficial, to decouple optimization-based meta-learning techniques from the high-dimensional space of model parameters. We achieve this by learning a stochastic latent space with an information bottleneck, conditioned on the input data, from which the high-dimensional parameters are generated.

Algorithm 1 Latent Embedding Optimization  
Require: Training meta-set  $\mathcal{S}^{tr} \in \mathcal{T}$   
Require: Learning rates  $\alpha$ ,  $\eta$   
1: Randomly initialize  $\phi_e, \phi_r, \phi_d$   
2: Let  $\phi = \{\phi_e, \phi_r, \phi_d, \alpha\}$   
3: while not converged do  
4: for number of tasks in batch do  
5: Sample task instance  $\mathcal{T}_i \sim \mathcal{S}^{tr}$   
6: Let  $(\mathcal{D}^{tr}, \mathcal{D}^{val}) = \mathcal{T}_i$   
7: Encode  $\mathcal{D}^{tr}$  to  $z$  using  $g_{\phi_e}$  and  $g_{\phi_r}$   
8: Decode  $z$  to initial params  $\theta_i$  using  $g_{\phi_d}$   
9: Initialize  $\mathbf{z}' = \mathbf{z}, \theta_i' = \theta_i$   
10: for number of adaptation steps do  
11: Compute training loss  $\mathcal{L}_{\mathcal{T}_i}^{tr}(f_{\theta_i'})$   
12: Perform gradient step w.r.t.  $\mathbf{z}'$   
13:  $\mathbf{z}' \gets \mathbf{z}' - \alpha \nabla_{\mathbf{z}'} \mathcal{L}_{\mathcal{T}_i}^{tr}(f_{\theta_i'})$   
14: Decode  $\mathbf{z}'$  to obtain  $\theta_i'$  using  $g_{\phi_d}$   
15: end for  
16: Compute validation loss  $\mathcal{L}_{\mathcal{T}_i}^{val}(f_{\theta_i'})$   
17: end for  
18: Perform gradient step w.r.t  $\phi$   
19:  $\phi \gets \phi - \eta \nabla_{\phi} \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{T}_i}^{val}(f_{\theta_i'})$

![](images/91f3f74850a68055051f5bbe423e8f7b2a9625fb138707ed5d9994dd815513e7.jpg)  
Figure 1: High-level intuition for LEO. While MAML operates directly in a high dimensional parameter space  $\Theta$ , LEO performs meta-learning within a low-dimensional latent space  $\mathcal{Z}$ , from which the parameters are generated.

Instead of explicitly instantiating and maintaining a unique set of model parameters  $\theta$ , as in MAML, we learn a generative distribution of model parameters which serves the same purpose. This is a natural extension: we relax the requirement of finding a single optimal  $\theta^{*} \in \Theta$  to that of approximating a data-dependent conditional probability distribution over  $\Theta$ , which can be more expressive, at the cost of potentially being more difficult to learn. The choice of architecture, composed of an encoding process, and decoding (or parameter generation) process, enables us to perform the MAML gradient-based adaptation steps (or "inner loop") in the learned, low-dimensional embedding space of the parameter generative model (Figure 1).

The high-level operation is then as follows (Algorithm 1). First, given a task instance  $\mathcal{T}_i$ , the inputs  $\{\mathbf{x}_n^k\}$  are passed through a stochastic encoder to produce a latent code  $\mathbf{z}$ , which is then decoded to parameters  $\theta_i$  using a parameter generator<sup>1</sup>. Given these instantiated model parameters, one or more adaptation steps are applied in the latent space, by differentiating the loss with respect to  $\mathbf{z}$ , taking a gradient step to get  $\mathbf{z}'$ , decoding new model parameters, and obtaining the new loss. Finally, optimized codes are decoded to produce the final adapted parameters  $\theta_i'$ , which can be used to perform the task, or compute the task-specific meta-loss. In this way, LEO incorporates aspects of model-based and optimization-based meta-learning, producing parameters that are first conditioned on the input data and then adapted by gradient descent. Accordingly, careful ablation studies are performed in Section 4 to assess the contribution of the different components in the approach.

In the following sections, we explain the LEO procedure more formally.

# 2.3.1 INITIALIZATION: GENERATING PARAMETERS CONDITIONED ON A FEW EXAMPLES

The first stage is to instantiate the model parameters that will be adapted to each task instance. Whereas MAML explicitly maintains a single set of model parameters, LEO utilises a data-dependent latent encoding which is then decoded to generate the actual initial parameters. In what follows, we describe an encoding scheme which leverages the power of relation networks to map the few-shot examples into a single latent vector; this particular design choice worked very well in the experiments we considered, but other choices for this mapping are possible.

Encoding The encoding process involves a simple feedforward mapping of each data point, followed by a relation network that considers the joint relationship between all of the data in the prob-

lem instance. The overall encoding process is defined in Eq. 3, and proceeds as follows. First, each single example from a problem instance  $\mathcal{T}_i = (\mathcal{D}^{tr},\mathcal{D}^{val})\sim p(\mathcal{T})$  is processed by an encoder network  $g_{\phi_e}\colon \mathcal{R}^{n_x}\to \mathcal{R}^{n_h}$ , which maps from input space to a code in an intermediate hidden-layer code space  $\mathcal{H}$ . Then, codes in  $\mathcal{H}$  corresponding to different training examples are concatenated pair-wise (resulting in  $(NK)^2$  pairs in the case of  $K$ -shot classification) and processed by a relation network  $g_{\phi_r}$ , in a fashion similar to Oreshkin et al. (2018) and Sung et al. (2017). The  $(NK)^2$  outputs are grouped by class and then averaged within each group to obtain the  $(2\times N)$  parameters of a probability distribution in a low-dimensional space  $\mathcal{Z} = \mathcal{R}^{n_z}$ , where  $n_z\ll \dim (\theta)$ , for each of the  $N$  classes.

Thus, given the  $K$ -shot training samples corresponding to a class  $n$ :  $\mathcal{D}_n^{tr} = \left\{\left(\mathbf{x}_n^k,y_n^k\right)\mid k = 1\dots K\right\}$  the encoder  $g_{\phi_e}$  and relation network  $g_{\phi_r}$  together parameterize a class-conditional multivariate Gaussian distribution with a diagonal covariance, which we can sample from in order to output a class-dependent latent code  $\mathbf{z}_n\in \mathcal{Z}$  as follows:

$$
\boldsymbol {\mu} _ {n} ^ {e}, \boldsymbol {\sigma} _ {n} ^ {e} = \frac {1}{N K ^ {2}} \sum_ {k _ {n} = 1} ^ {K} \sum_ {m = 1} ^ {N} \sum_ {k _ {m} = 1} ^ {K} g _ {\phi_ {r}} \left(g _ {\phi_ {e}} \left(\mathbf {x} _ {n} ^ {k _ {n}}\right), g _ {\phi_ {e}} \left(\mathbf {x} _ {m} ^ {k _ {m}}\right)\right) \tag {3}
$$

$$
\mathbf {z} _ {n} \sim q \left(\mathbf {z} _ {n} | \mathcal {D} _ {n} ^ {t r}\right) = \mathcal {N} \left(\boldsymbol {\mu} _ {n} ^ {e}, d i a g (\boldsymbol {\sigma} _ {n} ^ {e 2})\right)
$$

Intuitively, the encoder and relation network define a stochastic mapping from one or more class examples to a single code in the latent embedding space  $\mathcal{Z}$  corresponding to that class. The final latent code can be obtained as the concatenation of class-dependent codes:  $\mathbf{z} = [\mathbf{z}_1,\mathbf{z}_2,\dots ,\mathbf{z}_N]$ .

Decoding Without loss of generality, for few-shot classification, we can use the class-specific latent codes to instantiate just the top layer weights of the classifier. This allows the meta-learning in latent space to modulate the important high-level parameters of the classifier, without requiring the generator to produce very high-dimensional parameters. In this case,  $f_{\theta_i'}$  is a simple  $N$ -way linear softmax classifier, with model parameters  $\theta_i' = \{\mathbf{w}_n \mid n = 1 \dots N\}$ , and each  $\mathbf{x}_n^k$  can be either the raw input or some learned representation<sup>2</sup>.

Then, given the low-dimensional latent codes  $\mathbf{z}_n\in \mathcal{Z},n = 1\dots N$  , the decoder function  $g_{\phi_d}\colon \mathcal{Z}\to$ $\Theta$  is used to parameterize a Gaussian distribution with diagonal covariance in model parameter space  $\Theta$  , from which we can sample class-dependent parameters  $\mathbf{w}_n$  ..

$$
\boldsymbol {\mu} _ {n} ^ {d}, \boldsymbol {\sigma} _ {n} ^ {d} = g _ {\phi_ {d}} (\mathbf {z} _ {n})
$$

$$
\mathbf {w} _ {n} \sim p (\mathbf {w} | \mathbf {z} _ {n}) = \mathcal {N} \left(\boldsymbol {\mu} _ {n} ^ {d}, \operatorname {d i a g} \left(\boldsymbol {\sigma} _ {n} ^ {d ^ {2}}\right)\right) \tag {4}
$$

In other words, codes  $\mathbf{z}_n$  are mapped independently to the top-layer parameters  $\theta_{i}$  of a softmax classifier using the decoder  $g_{\phi_d}$ , which is essentially a stochastic generator of model parameters.

# 2.3.2 ADAPTATION BY LATENT EMBEDDING OPTIMIZATION (LEO) (THE "INNER LOOP")

Given the decoded parameters, we can then define the "inner loop" classification loss using the cross-entropy function, as follows:

$$
\mathcal {L} _ {\mathcal {T} _ {i}} ^ {t r} \left(f _ {\theta_ {i}}\right) = \sum_ {(\mathbf {x}, y) \in \mathcal {D} ^ {t r}} \left[ - \mathbf {w} _ {y} \cdot \mathbf {x} + \log \left(\sum_ {j = 1} ^ {N} e ^ {\mathbf {w} _ {j} \cdot \mathbf {x}}\right) \right] \tag {5}
$$

It is important to note that the decoder  $g_{\phi_d}$  is a differentiable mapping between the latent space  $\mathcal{Z}$  and the higher-dimensional model parameter space  $\Theta$ . Primarily, this allows gradient-based optimization of the latent codes with respect to the training loss, with  $\mathbf{z}_n' = \mathbf{z}_n - \alpha \nabla_{\mathbf{z}_n} \mathcal{L}_{\mathcal{T}_i}^{tr}$ . The decoder  $g_{\phi_d}$  will convert adapted latent codes  $\mathbf{z}_n'$  to effective model parameters  $\theta_i'$  for each adaptation step, which can be repeated several times, as in Algorithm 1. In addition, by backpropagating errors through the decoder, the encoder can learn to provide a data-conditioned latent encoding  $\mathbf{z}$  that produces an appropriate initialization point  $\theta_i$  for the classifier model. Stochasticity in the entire process can be efficiently handled using the reparametrization trick.

# 2.3.3 META-TRAINING STRATEGY (THE "OUTER LOOP")

For each task instance  $\mathcal{T}_i$ , the initialization and adaptation procedure produces a new classifier  $f_{\theta_i'}$  tailored to the training set  $\mathcal{D}^{tr}$  of the instance, which we can then evaluate on the validation set of that instance  $\mathcal{D}^{val}$ . During meta-training we differentiate through the "inner loop" and update the encoder and decoder network parameters:  $\phi_e$ ,  $\phi_r$ , and  $\phi_d$ . This process is analogous to the MAML "outer loop", but instead of learning a fixed set of initialization parameters we learn a parameter generator implicitly represented by the encoder and decoder networks. Meta-training is performed by minimizing the following objective:

$$
\min  _ {\phi_ {e}, \phi_ {r}, \phi_ {d}} \sum_ {\mathcal {T} _ {i} \sim p (\mathcal {T})} \left[ \mathcal {L} _ {\mathcal {T} _ {i}} ^ {v a l} \left(f _ {\theta_ {i} ^ {\prime}}\right) + \beta D _ {K L} \left(q \left(\mathbf {z} _ {n} \mid \mathcal {D} _ {n} ^ {t r}\right) \mid | p \left(\mathbf {z} _ {n}\right)\right) + \gamma \left| \right| s t o p g r a d \left(\mathbf {z} _ {n} ^ {\prime}\right) - \mathbf {z} _ {n} \| _ {2} ^ {2} \right] + R \tag {6}
$$

where  $p(\mathbf{z}_n) = \mathcal{N}(0,\mathcal{I})$ . Similar to the loss defined in (Higgins et al., 2017) we use a weighted KL-divergence term to encourage the generative model to learn a disentangled embedding, which should also simplify the LEO "inner loop" by removing correlations between latent space gradient dimensions. We found it beneficial to encourage the encoder and relation networks to produce latent codes which are close to final adapted codes.

$L_{2}$  regularization was used with all weights of the model, as well as a soft, layer-wise orthogonality constraint on decoder network weights, which encourages the dimensions of the latent code as well as the decoder network to be maximally expressive. In the case of linear encoder, relation, and decoder networks, and assuming that  $\mathcal{C}_d$  is the correlation matrix between rows of  $\phi_d$ , then the regularization term takes the following form:

$$
R = \lambda_ {1} \left(\left\| \phi_ {e} \right\| _ {2} ^ {2} + \left\| \phi_ {r} \right\| _ {2} ^ {2} + \left\| \phi_ {d} \right\| _ {2} ^ {2}\right) + \lambda_ {2} \| \mathcal {C} _ {d} - \mathcal {I} \| _ {1} \tag {7}
$$

# 2.3.4 BEYOND CLASSIFICATION AND LINEAR OUTPUT LAYERS

Thus far we have used few-shot classification as a working example to highlight our proposed method of meta-learning with latent embedding optimization. In this domain, for simplicity, we generate only a single linear output layer in a class conditional manner. However, our approach can be applied to any model  $f_{\theta_i}$  which maps observations to outputs, e.g. a nonlinear MLP or LSTM, by using a single latent code  $\mathbf{z}$  to generate the entire parameter vector  $\theta_i$  with an appropriate decoder. In the general case,  $\mathbf{z}$  is conditioned on  $\mathcal{D}^{tr}$  by passing both inputs and labels to the encoder. Furthermore, the loss  $\mathcal{L}_{\mathcal{T}_i}$  is not restricted to be a classification loss, and can be replaced by any differentiable loss function which can be computed on  $\mathcal{D}^{tr}$  and  $\mathcal{D}^{val}$  sets of a task instance  $\mathcal{T}_i$ .

# 3 RELATED WORK

The problem of few-shot adaptation has been approached in a number of different ways, such as through the conceptual frameworks of fast weights (Hinton & Plaut, 1987; Ba et al., 2016), in the context of learning-to-learn (Schmidhuber, 1987; Thrun & Pratt, 1998; Hochreiter et al., 2001; Andrychowicz et al., 2016), and through meta-learning. Many recent approaches to meta-learning can be broadly categorized as metric-based methods, which focus on learning similarity metrics for members of the same class (e.g. Koch et al., 2015; Vinyals et al., 2016; Snell et al., 2017); memory-based methods, which exploit memory architectures to either store key training examples or directly encode fast adaptation algorithms (e.g. Santoro et al., 2016; Ravi & Larochelle, 2017); and optimization-based methods, which search for regions in parameter space that are conducive to fast gradient-based adaptation to new tasks (e.g. Finn et al., 2017; 2018).

Related work has also explored the use of one neural network to produce (some fraction of) the parameters of another (Ha et al., 2016; Krueger et al., 2017), with some approaches focusing on the goal of fast adaptation. Munkhdalai et al. (2017) meta-learn an algorithm to change additive biases across deep networks conditioned on the few-shot training samples. Gidaris & Komodakis (2018) use an attention kernel to output class conditional mixing of pre-trained linear output weights for novel categories. Qiao et al. (2017) learn to output top linear layer parameters from the activations

provided by a pre-trained feature embedding, but they do not make use of gradient-based adaptation. None of the aforementioned approaches to fast adaptation explicitly learn a probability distribution over model parameters, or make use of latent variable generative models.

Approaches which use optimization-based meta-learning include MAML (Finn et al., 2017) and REPTILE (Nichol & Schulman, 2018). While MAML backpropagates the meta-loss through the "inner loop", REPTILE simplifies the computation by incorporating an  $L_{2}$  loss which updates the meta-model parameters towards the instance-specific adapted models. These approaches use the full, high-dimensional set of model parameters within the "inner loop", while Lee & Choi (2018) learn a layer-wise subspace in which to use gradient-based adaptation. However, it is not clear how these methods scale to large expressive models such as residual networks (especially given the uncertainty in the few-shot data regime), since MAML is prone to overfitting (Mishra et al., 2018).

Probabilistic meta-learning approaches such as those of Bauer et al. (2017) and Grant et al. (2018) have shown the advantages of learning Gaussian posteriors over model parameters. Concurrently with our work, Kim et al. (2018) and Finn et al. (2018) propose probabilistic extensions to MAML that are trained using a variational approximation, using simple posteriors. However, it is not immediately clear how to extend them to more complex distributions with a more diverse set of tasks. Other concurrent works have introduced deep parameter generators (Lacoste et al., 2018; Wu et al., 2018) that can better capture a wider distribution of model parameters, but do not employ gradient-based adaptation. In contrast, our approach employs both a generative model of parameters, and adaptation in a low-dimensional latent space, aided by a data-dependent initialization.

Finally, recently proposed Neural Processes (Garnelo et al., 2018a,b) bear similarity to our work: they also learn a mapping to and from a latent space that can be used for few-shot function estimation. However, coming from a Gaussian processes perspective, their work does not perform "inner loop" adaptation and is trained by optimizing a variational objective.

# 4 EVALUATION

We evaluate the proposed approach on regression problems as well as few-shot classification tasks. This evaluation is designed to answer the following key questions: (1) Is LEO capable of modeling a distribution over model parameters when faced with sources of uncertainty? (2) Can LEO learn from multimodal task distributions and will this be reflected in ambiguous problem instances, where distinct solutions are possible given more than a single mode? (3) Is LEO competitive on large-scale few-shot learning benchmarks?

# 4.1 FEW-SHOT REGRESSION

To answer the first two questions we adopt the simple regression task of Finn et al. (2018). 1D regression problems are generated in equal proportions using either a sine wave with random amplitude and phase, or a line with random slope and intercept. Inputs are sampled randomly, creating a multimodal task distribution. Crucially, random Gaussian noise with standard deviation 0.3 is added to regression targets. Coupled with the small number of training samples (5-shot), the task is challenging for 2 main reasons: (1) learning a distribution over models becomes necessary, in order to account for the uncertainty introduced by noisy labels; (2) problem instances may be likely under both modes: in some cases a sine wave may fit the data as well as a line. Faced with such ambiguity, learning a distribution over model parameters should enable sampling of several likely models, potentially from different modes.

We used a 3-layer MLP as the underlying model architecture of  $f_{\theta}$ , and we produced the entire parameter tensor  $\theta$  with the LEO generator, conditionally on  $D^{tr}$ , the few-shot training inputs concatenated with noisy labels. For further details, see Appendix A.1.

In Figure 2 we show samples from a single model trained on noisy sines and lines, with true regression targets in black and training samples marked with red circles and horizontal dashed lines. Panels (a) and (b) illustrate how LEO captures some of the uncertainty in ambiguous problem instances within each mode, especially in parts of the input space far from any training samples. Conversely, in parts which contain data, models fit the true regression target well. Interestingly, when both sines and lines could explain the data, as shown in panels (c) and (d), we observe that LEO can sample

![](images/448b78a0defbc7478a41ea7a80491725638fee0f651b3c857ac361babc4f23e9.jpg)  
Figure 2: Meta-learning with LEO of a multimodal task distribution with sines and lines, using 5-shot regression with noisy targets. Our model outputs a distribution of possible solutions, which is also multimodal in ambiguous cases. True regression targets are plotted in black, while the 5 training examples are highlighted with red circles and vertical dashed lines. Several samples from our model are plotted with dotted lines (best seen in color).

very different models, from both families, reflecting its ability to represent parametric uncertainty appropriately.

# 4.2 FEW-SHOT CLASSIFICATION

In order to answer the final question we scale up our approach to 1-shot and 5-shot classification problems defined using two commonly used ImageNet subsets.

# 4.2.1 DATASETS

The miniImageNet dataset (Vinyals et al., 2016) is a subset of 100 classes selected randomly from the ILSVRC-12 dataset (Russakovsky et al., 2014) with 600 images sampled from each class. Following the split proposed by Ravi & Larochelle (2017), the dataset is divided into training, validation, and test meta-sets, with 64, 16, and 20 classes respectively.

The tieredImageNet dataset (Ren et al., 2018) is a larger subset of ILSVRC-12 with 608 classes (779,165 images) grouped into 34 higher-level nodes in the ImageNet human-curated hierarchy (Deng et al., 2009). This set of nodes is partitioned into 20, 6, and 8 disjoint sets of training, validation, and testing nodes, and the corresponding classes form the respective meta-sets. As argued in Ren et al. (2018), this split nears the root of the ImageNet hierarchy results in a more challenging, yet realistic regime with test classes that are less similar to training classes.

# 4.2.2 PRE-TRAINED FEATURES

Two potential difficulties of using LEO to instantiate parameters with a generator network are: (1) modeling distributions over very high-dimensional parameter spaces; and (2) requiring meta-learning (and hence, gradient computation in the inner loop) to be performed with respect to a high-dimensional input space. We address these issues by pre-training a visual representation of the data and then use the generator to instantiate the parameters for the final layer - a linear softmax classifier operating on this representation. We train a 28-layer Wide Residual Network (WRN-28-10) (Zagoruyko & Komodakis, 2016a) with supervised classification using only data and classes from the training meta-set. Recent state-of-the-art approaches use the penultimate layer representation (Qiao et al., 2017; Bauer et al., 2017; Gidaris & Komodakis, 2018); however, we choose the intermediate feature representation in layer 21, given that higher layers tend to specialize to the training distribution (Yosinski et al., 2014). For details regarding the training and evaluation procedures, see Appendix A.2.

# 4.2.3 FINE-TUNING

Following the LEO adaptation procedure (Algorithm 1) we also use fine-tuning by performing a few steps of gradient-based adaptation directly in parameter space using the few-shot set  $\mathcal{D}^{tr}$ ; this fine-tuning is similar to the adaptation procedure of MAML, or Meta-SGD (Li et al., 2017) when the learning rates are learned, with the important difference that starting points of fine-tuning are

custom generated by LEO for every task instance  $\mathcal{T}_i$ . Empirically, we find that fine-tuning applies a very small change to the parameters with only a slight improvement in performance on supervised classification tasks.

# 4.3 RESULTS

<table><tr><td rowspan="2">Model</td><td colspan="2">miniImageNet test accuracy</td></tr><tr><td>1-shot</td><td>5-shot</td></tr><tr><td>Matching networks (Vinyals et al., 2016)</td><td>43.56 ± 0.84%</td><td>55.31 ± 0.73%</td></tr><tr><td>Meta-learner LSTM (Ravi &amp; Larochelle, 2017)</td><td>43.44 ± 0.77%</td><td>60.60 ± 0.71%</td></tr><tr><td>MAML (Finn et al., 2017)</td><td>48.70 ± 1.84%</td><td>63.11 ± 0.92%</td></tr><tr><td>LLAMA (Grant et al., 2018)</td><td>49.40 ± 1.83%</td><td>-</td></tr><tr><td>REPTILE (Nichol &amp; Schulman, 2018)</td><td>49.97 ± 0.32%</td><td>65.99 ± 0.58%</td></tr><tr><td>PLATIPUS (Finn et al., 2018)</td><td>50.13 ± 1.86%</td><td>-</td></tr><tr><td>Meta-SGD (our features)</td><td>54.24 ± 0.03%</td><td>70.86 ± 0.04%</td></tr><tr><td>SNAIL (Mishra et al., 2018)</td><td>55.71 ± 0.99%</td><td>68.88 ± 0.92%</td></tr><tr><td>(Gidaris &amp; Komodakis, 2018)</td><td>56.20 ± 0.86%</td><td>73.00 ± 0.64%</td></tr><tr><td>(Bauer et al., 2017)</td><td>56.30 ± 0.40%</td><td>73.90 ± 0.30%</td></tr><tr><td>(Munkhdalai et al., 2017)</td><td>57.10 ± 0.70%</td><td>70.04 ± 0.63%</td></tr><tr><td>TADAM (Oreshkin et al., 2018)</td><td>58.50 ± 0.30%</td><td>76.70 ± 0.30%</td></tr><tr><td>(Qiao et al., 2017)</td><td>59.60 ± 0.41%</td><td>73.74 ± 0.19%</td></tr><tr><td>LEO (ours)</td><td>61.76 ± 0.08%</td><td>77.59 ± 0.12%</td></tr><tr><td rowspan="2">Model</td><td colspan="2">tieredImageNet test accuracy</td></tr><tr><td>1-shot</td><td>5-shot</td></tr><tr><td>MAML (deeper net, evaluated in Liu et al. (2018))</td><td>51.67 ± 1.81%</td><td>70.30 ± 0.08%</td></tr><tr><td>Prototypical Nets (Ren et al., 2018)</td><td>53.31 ± 0.89%</td><td>72.69 ± 0.74%</td></tr><tr><td>Relation Net (evaluated in Liu et al. (2018))</td><td>54.48 ± 0.93%</td><td>71.32 ± 0.78%</td></tr><tr><td>Transductive Prop. Nets (Liu et al., 2018)</td><td>57.41 ± 0.94%</td><td>71.55 ± 0.74%</td></tr><tr><td>Meta-SGD (our features)</td><td>62.95 ± 0.03%</td><td>79.34 ± 0.06%</td></tr><tr><td>LEO (ours)</td><td>66.33 ± 0.05%</td><td>81.44 ± 0.09%</td></tr></table>

Table 1: Test accuracies on miniImageNet and tieredImageNet. For each dataset, the first set of results use convolutional networks, while the second use much deeper residual networks, predominantly in conjunction with pre-training.

The classification accuracies for LEO and other baselines are shown in Table 1. LEO sets the new state-of-the-art performance on the 1-shot and 5-shot tasks for both miniImageNet and tieredImageNet datasets, with comfortable margins. We also evaluated LEO on the "multi-view" feature representation used by Qiao et al. (2017) with miniImageNet, which involves significant data augmentation compared to the approaches in Table 1. LEO is state-of-the-art using these features as well, with  $63.97 \pm 0.20\%$  and  $79.49 \pm 0.70\%$  test accuracies on the 1-shot and 5-shot tasks respectively.

# 4.4 ABLATION STUDY

To assess the effects of different components, we also performed an ablation study, with detailed results in Table 2. To ensure a fair comparison, all approaches begin with the same pre-trained features (Section 4.2.2). The Meta-SGD case performs gradient-based adaption directly in the parameter space in the same way as MAML, but also meta-learns the inner loop learning rate (as we do for LEO). The main approach, labeled as LEO in the table, uses a stochastic parameter generator for several steps of latent embedding optimization, followed by fine-tuning steps in parameter space (see subsection 4.2.3). All versions of LEO are at or above the previous state-of-the-art on all tasks.

The largest difference in performance is between Meta-SGD and the other cases (all of which exploit a latent representation of model parameters), indicating that the low-dimensional bottleneck is critical for this application. The "conditional generator only" case (without adaptation in latent space) yields a poorer result than LEO, and even adding fine-tuning in parameter space does not recover performance; this illustrates the efficacy of the latent adaptation procedure. The importance of the data-dependent encoding is highlighted by the "random prior" case, in which the encoding process is replaced by the prior  $p(\mathbf{z}_n)$ , and performance decreases. We also find that incorporating stochasticity can be important for miniImageNet, but not for tieredImageNet, which we hypothesize

<table><tr><td rowspan="2">Model</td><td colspan="2">miniImageNet test accuracy</td><td colspan="2">tieredImageNet test accuracy</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td>Meta-SGD (our features)</td><td>54.24 ± 0.03%</td><td>70.86 ± 0.04%</td><td>62.95 ± 0.03%</td><td>79.34 ± 0.06%</td></tr><tr><td>Conditional generator only</td><td>60.33 ± 0.11%</td><td>74.53 ± 0.11%</td><td>65.17 ± 0.15%</td><td>78.77 ± 0.03%</td></tr><tr><td>Conditional generator + fine-tuning</td><td>60.62 ± 0.31%</td><td>76.42 ± 0.09%</td><td>65.74 ± 0.28%</td><td>80.65 ± 0.07%</td></tr><tr><td>Previous SOTA</td><td>59.60 ± 0.41%</td><td>76.70 ± 0.30%</td><td>57.41 ± 0.94%</td><td>72.69 ± 0.74%</td></tr><tr><td>LEO (random prior)</td><td>61.01 ± 0.12%</td><td>77.27 ± 0.05%</td><td>65.39 ± 0.10%</td><td>80.83 ± 0.13%</td></tr><tr><td>LEO (deterministic)</td><td>61.48 ± 0.05%</td><td>76.53 ± 0.24%</td><td>66.18 ± 0.17%</td><td>82.06 ± 0.08%</td></tr><tr><td>LEO (no fine-tuning)</td><td>61.62 ± 0.15%</td><td>77.46 ± 0.12%</td><td>66.14 ± 0.17%</td><td>80.89 ± 0.11%</td></tr><tr><td>LEO (ours)</td><td>61.76 ± 0.08%</td><td>77.59 ± 0.12%</td><td>66.33 ± 0.05%</td><td>81.44 ± 0.09%</td></tr></table>

Table 2: Ablation study and comparison to Meta-SGD. Unless otherwise specified, LEO stands for using the stochastic generator for latent embedding optimization followed by fine-tuning.

is because the latter is much larger. Finally, the fine-tuning steps only yield a statistically significant improvement on the 5-shot tieredImageNet task. Thus, both the data-conditional encoding and latent space adaptation are critical to the performance of LEO, with each of the proposed improvements contributing to the overall accuracy.

![](images/de9bfc6742f71a499c11b367eb0e069155ebaa65d1cb34acc5a25a9d0559ffaa.jpg)  
Figure 3: t-SNE plot of latent space codes before and after adaptation: (a) Sample-conditional codes  $\mathbf{z}_n$  (blue) and adapted codes  $\mathbf{z}_n^\prime$  (orange); (b) Same as (a) but colored by class label; (c) Same as (a) but highlighting codes  $\mathbf{z}_n$  for validation class "Jellyfish" (left) and corresponding adapted codes  $\mathbf{z}_n^\prime$  (right).

![](images/078a6a442d827dfdbd9beb41a863d0ccc4729d38c2974cd86ff85cbce30d05cd.jpg)

![](images/e54a60813696aac49f50168567f2476a95abb513b36852a004ad7321b9ca3546.jpg)

![](images/cd005f17fee4ae7cbbabf143fd5e5713710927a1f75fb1edc00d137aa3340c12.jpg)

![](images/910d6bb428a49ef1c858f2d58697b6f634cb5db25a6a89fa291ecfc90518eabc.jpg)

![](images/defd34183c56f1f83ca4c955f0b0ef0293b1ffc46f55bfc177b3c34f3e052399.jpg)

# 4.5 LATENT EMBEDDING VISUALIZATION

To qualitatively characterize the learnt embedding space, we plot codes produced by the relational encoder before and after the LEO procedure, using a 5-way 1-shot model and 1000 task instances from the validation meta-set of miniImageNet. Figure 3 shows a t-SNE projection of class conditional encoder outputs  $\mathbf{z}_n$  as well as their respective final adapted versions  $\mathbf{z}_n'$ . If the effect of LEO were minimal, we would expect latent codes to have roughly the same structure before and after adaptation. In contrast, Figure 3(a) clearly shows that latent codes change substantially during LEO, since encoder output codes form a large cluster (blue) to which adapted codes (orange) do not belong. Figure 3(b) shows the same t-SNE embedding as (a) colored by class label. Note that encoder outputs, on the left side of plot (b), have a lower degree of class conditional separation compared to  $\mathbf{z}_n'$  clusters on the right, suggesting that qualitatively different structure is introduced by the LEO procedure. We further illustrate this point by highlighting latent codes for the "Jellyfish" validation class in Figure 3(c), which are substantially different before and after adaptation.

The additional structure of adapted codes  $\mathbf{z}_n^{\prime}$  may provide an explanation for LEO's superior performance over approaches predicting parameters directly from inputs, since the decoder may not be able to produce sufficiently different weights for different classes given very similar latent codes, especially when the decoder is linear. Conversely, LEO can reduce the uncertainty of the encoder mapping, which is inherent in the few-shot regime, by adapting latent codes with a generic, gradient-based procedure.

![](images/4c60793cbcb0a50a8d63b8aba3052f80284e71d1ccc9016d3745b3a368bde32e.jpg)  
(a)

![](images/0b7e53c3f46a20166bd83a955f29fbacd251935cb2b7c81051767e4daf8c325f.jpg)  
(b)  
Figure 4: Curvature and coverage metrics for a number of different models, computed over 1000 problem instances drawn uniformly from the test meta-set. The "gen+ft" case uses the data-conditional generator, but performs adaptation (fine-tuning) in parameter space; and metaSGD only performs adaptation in parameter space. In all cases, the same feature representation was used. For all plots, the whiskers span from the  $5^{\text{th}}$  to  $95^{\text{th}}$  percentile of the observed quantities.

![](images/5d08de95f20e857a9b30fcc999b88bd24c519b32658ecce096f2cf249d98338e.jpg)  
(c)

# 4.6 CURVATURE AND COVERAGE ANALYSIS

We hypothesize that by performing the inner-loop optimization in a lower-dimensional latent space, the adapted solutions do not need to be close together in parameter space, as each latent step can cover a larger region of parameter space and effect a greater change on the underlying function. To support this intuition, we compute a number of curvature and coverage measures, shown in Figure 4.

The curvature provides a measure of the sensitivity of a function with respect to some space. If adapting in latent space allows as much control over the function as in parameter space, one would expect the curvatures to match. However, as demonstrated in Figure 4(a), the curvature for LEO in  $\mathbf{z}$  space (the absolute eigenvalues of the Hessian of the loss) is 2 orders of magnitude higher than in  $\theta$ , indicating that taking a fixed size step in  $\mathbf{z}$  space will cause the function to change more drastically than taking the same step directly in  $\theta$ . This effect is also observed in the "gen+ft" case, where the latent embedding is still used, but adaptation is performed directly in  $\theta$  space. This suggests that the latent bottleneck is responsible for this effect. Figure 4(b) shows that this is due to the expansion of space caused by the decoder. In this case the decoder is linear, and the singular values describe how much a vector projected through this decoder grows along different directions, with a value of one preserving volume. We observe that the decoder is expanding the space by at least one order of magnitude. Finally, Figure 4(c) demonstrates this effect along the specific gradient directions used in the inner loop adaptation: the small gradient steps in  $\mathbf{z}$  taken by LEO induce much larger steps in  $\theta$  space, larger than the gradient steps taken by Meta-SGD in  $\theta$  space directly. Thus, the results support the intuition that LEO is able to 'transport' models further during adaptation by performing meta-learning in the latent space.

# 5 CONCLUSIONS AND FUTURE WORK

We have introduced Latent Embedding Optimization (LEO), a meta-learning technique which uses a parameter generative model to capture the diverse range of parameters useful for a distribution over tasks, and demonstrated a new state-of-the-art result on the challenging 5-way 1- and 5-shot miniImageNet and tieredImageNet classification problems. LEO achieves this by learning a low-dimensional data-dependent latent embedding, and performing gradient-based adaptation in this space, which means that it allows for a task-specific parameter initialization and can perform adaptation more effectively.

Future work could focus on replacing the pre-trained feature extractor with one learned jointly through meta-learning. Another interesting avenue for investigation is to use LEO for reinforcement learning tasks. Conceptually, the use of expressive neural network encoders and decoders to learn complex probability distributions over model parameters may lead to more stable optimization-based meta-reinforcement learning, where the generator could, in principle, encode more than one family of policies and thus enable flexible, few-shot adaptation for several different behaviours.

# REFERENCES

Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. In Advances in Neural Information Processing Systems, pp. 3981-3989, 2016.  
Jimmy Ba, Geoffrey E Hinton, Volodymyr Mnih, Joel Z Leibo, and Catalin Ionescu. Using fast weights to attend to the recent past. In Advances in Neural Information Processing Systems, pp. 4331-4339, 2016.  
M. Bauer, M. Rojas-Carulla, J. Bartlomiej Światkowski, B. Schölkopf, and R. E. Turner. Discriminative k-shot learning using probabilistic models. ArXiv e-prints, June 2017.  
Kyunghyun Cho, Bart van Merrienboer, Caglar Gülçehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014. URL http://arxiv.org/abs/1406.1078.  
J. Deng, W. Dong, R. Socher, L. Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248-255, June 2009.  
Li Fei-Fei, Rob Fergus, and Pietro Perona. One-shot learning of object categories. IEEE transactions on pattern analysis and machine intelligence, 28(4):594-611, 2006.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, pp. 1126-1135, 2017.  
Chelsea Finn, Kelvin Xu, and Sergey Levine. Probabilistic model-agnostic meta-learning. arXiv preprint arXiv:1806.02817, 2018.  
Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and S. M. Ali Eslami. Conditional neural processes. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 1704-1713, Stockholm, Sweden, 10-15 Jul 2018a. PMLR. URL http://proceedings.mlr.press/v80/garnelo18a.html.  
Marta Garnelo, J. Schwarz, D. Rosenbaum, F. Viola, D. J. Rezende, S. M. A. Eslami, and Y. Whye  
Teh. Neural Processes. ArXiv e-prints, July 2018b.  
Spyros Gidaris and Nikos Komodakis. Dynamic few-shot visual learning without forgetting. CoRR, abs/1804.09458, 2018. URL http://arxiv.org/abs/1804.09458.  
Erin Grant, Chelsea Finn, Sergey Levine, Trevor Darrell, and Thomas Griffiths. Recasting gradient-based meta-learning as hierarchical bayes. In Proceedings of the 6th International Conference on Learning Representations (ICLR), 2018.  
David Ha, Andrew Dai, and Quoc V Le. Hypernetworks. arXiv preprint arXiv:1609.09106, 2016.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. Beta-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR*, 2017.  
Geoffrey E Hinton and David C Plaut. Using fast weights to deblur old memories. In Proceedings of the ninth annual conference of the Cognitive Science Society, pp. 177-186, 1987.  
Sepp Hochreiter, A. Steven Younger, and Peter R. Conwell. Learning to learn using gradient descent. In Proceedings of the International Conference on Artificial Neural Networks, ICANN '01, pp. 87-94, London, UK, UK, 2001. Springer-Verlag. ISBN 3-540-42486-5. URL http://dl.acm.org/citation.cfm?id=646258.684281.

T. Kim, J. Yoon, O. Dia, S. Kim, Y. Bengio, and S. Ahn. Bayesian Model-Agnostic Meta-Learning. ArXiv e-prints, June 2018.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014. URL http://arxiv.org/abs/1412.6980.  
Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. Siamese neural networks for one-shot image recognition. In ICML Deep Learning Workshop, volume 2, 2015.  
D. Krueger, C.-W. Huang, R. Islam, R. Turner, A. Lacoste, and A. Courville. Bayesian Hypernetworks. ArXiv e-prints, October 2017.  
A. Lacoste, B. Oreshkin, W. Chung, T. Boquet, N. Rostamzadeh, and D. Krueger. Uncertainty in Multitask Transfer Learning. *ArXiv e-prints*, June 2018.  
Brenden Lake, Ruslan Salakhutdinov, Jason Gross, and Joshua Tenenbaum. One shot learning of simple visual concepts. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 33, 2011.  
Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436, 2015.  
Y. Lee and S. Choi. Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace. ArXiv e-prints, January 2018.  
Zhenguo Li, Fengwei Zhou, Fei Chen, and Hang Li. Meta-sgd: Learning to learn quickly for few shot learning. CoRR, abs/1707.09835, 2017. URL http://arxiv.org/abs/1707.09835.  
Yanbin Liu, Juho Lee, Minseop Park, Saehoon Kim, and Yi Yang. Transductive propagation network for few-shot learning. arXiv preprint arXiv:1805.10002, 2018.  
Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, and Pieter Abbeel. A simple neural attentive metalearner. In Proceedings of the 6th International Conference on Learning Representations (ICLR), 2018.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, February 2015. ISSN 00280836. URL http://dx.doi.org/10.1038/nature14236.  
Tsendsumen Munkhdalai, Xingdi Yuan, Soroush Mehri, Tong Wang, and Adam Trischler. Learning rapid-temporal adaptations. CoRR, abs/1712.09926, 2017. URL http://arxiv.org/abs/1712.09926.  
Alex Nichol and John Schulman. Reptile: a scalable metalearning algorithm. arXiv preprint arXiv:1803.02999, 2018.  
B. N. Oreshkin, P. Rodriguez, and A. Lacoste. TADAM: Task dependent adaptive metric for improved few-shot learning. *ArXiv e-prints*, May 2018.  
Siyuan Qiao, Chenxi Liu, Wei Shen, and Alan Yuille. Few-shot image recognition by predicting parameters from activations. arXiv preprint arXiv:1706.03466, 2017.  
Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In Proceedings of the 5th International Conference on Learning Representations (ICLR), 2017.  
Mengye Ren, Sachin Ravi, Eleni Triantafillou, Jake Snell, Kevin Swersky, Josh B. Tenenbaum, Hugo Larochelle, and Richard S. Zemel. Meta-learning for semi-supervised few-shot classification. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=HJcSzz-CZ.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael S. Bernstein, Alexander C. Berg, and Fei-Fei Li. Imagenet large scale visual recognition challenge. CoRR, abs/1409.0575, 2014. URL http://arxiv.org/abs/1409.0575.  
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In International conference on machine learning, pp. 1842-1850, 2016.  
Jürgen Schmidhuber. Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta... hook. PhD thesis, Technische Universität München, 1987.  
Jürgen Schmidhuber. Deep learning in neural networks: An overview. Neural networks, 61:85-117, 2015.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. Mastering the game of go without human knowledge. Nature, 550:354-, October 2017. URL http://dx.doi.org/10.1038/nature24270.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. CoRR, abs/1703.05175, 2017. URL http://arxiv.org/abs/1703.05175.  
Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip H. S. Torr, and Timothy M. Hospedales. Learning to compare: Relation network for few-shot learning. CoRR, abs/1711.06025, 2017. URL http://arxiv.org/abs/1711.06025.  
Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. Sequence to sequence learning with neural networks. CoRR, abs/1409.3215, 2014. URL http://arxiv.org/abs/1409.3215.  
Sebastian Thrun and Lorien Pratt. Learning to learn: Introduction and overview. In Learning to learn, pp. 3-17. Springer, 1998.  
Oriol Vinyals, Charles Blundell, Tim Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. In Advances in Neural Information Processing Systems, pp. 3630-3638, 2016.  
T. Wu, J. Peurifoy, I. L. Chuang, and M. Tegmark. Meta-learning autoencoders for few-shot prediction. ArXiv e-prints, July 2018.  
Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neural networks? CoRR, abs/1411.1792, 2014. URL http://arxiv.org/abs/1411.1792.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In British Machine Vision Conference, 2016a.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. CoRR, abs/1605.07146, 2016b. URL http://arxiv.org/abs/1605.07146.
