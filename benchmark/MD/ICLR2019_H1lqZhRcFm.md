# UNSUPERVISED LEARNING OF THE SET OF LOCAL MAXIMA

Anonymous authors

Paper under double-blind review

# ABSTRACT

This paper describes a new form of unsupervised learning, whose input is a set of unlabeled points that are assumed to be local maxima of an unknown value function in an unknown subset of the vector space. Two functions are learned: (i) a set indicator  $c$ , which is a binary classifier, and (ii) a comparator function  $h$  that given two nearby samples, predicts which sample has the higher value. Loss terms are used to ensure that all training samples  $x$  are a local maxima, according to  $h$  and satisfy  $c(x) = 1$ . Therefore,  $c$  and  $h$  provide training signals to each other: a point  $x'$  in the vicinity of  $x$  satisfies  $c(x) = -1$  or is deemed by  $h$  to be lower in value than  $x$ . We present an algorithm, show an example where it is more efficient to use local maxima as an indicator function than to employ conventional classification, and derive a suitable generalization bound. Our experiments show that the method is able to outperform one-class classification algorithms in the task of anomaly detection and also provide an additional signal that is extracted in a completely unsupervised way.

# 1 INTRODUCTION

...from so simple a beginning endless forms most beautiful and most wonderful have been, and are being, evolved. (Darwin, 1859)

When we observe the natural world, we see the "most wonderful" forms. We do not observe the even larger quantity of less spectacular forms and we cannot see those forms that are incompatible with existence. In other words, each sample we observe is the result of optimizing some fitness or value function under a set of constraints: the alternative, lower-value, samples are removed and the samples that do not satisfy the constraints are also missing.

The same principle also holds at the sub-cellular level. For example, a gene can have many forms. Some of them are completely synonymous, while others are viable alternatives. The gene forms that become most frequent are those which are not only viable, but which also minimize the energetic cost of their expression (Farkas et al., 2018). For example, the genes that encode proteins comprised of amino acids of higher availability or that require lower expression levels to achieve the same outcome have an advantage.

The same idea, of mixing constraints with optimality, also holds for man-made objects. Consider, for example, the set of houses in a given neighborhood. Each architect optimizes the final built form to cope with various aspects, such as the maximal residential floor area, site accessibility, parking considerations, the energy efficiency of the built product, etc. What architects find most challenging, is that this optimization process needs to correspond to a comprehensive set of state and city regulations that regard, for example, the proximity of the built mass of the house to the lot's boundaries, or the compliance of the egress sizes with current fire codes.

In another instance, consider the weights of multiple networks trained to minimize the same loss on the same training data, each using a different random initialization. By the nature of the problem, the obtained weights are the local optimum of some loss optimization process. In addition, the weights are sometimes subject to constraints, e.g., by using weight normalization.

The task tackled in this paper is learning the value function and the constraints, by observing only the local maxima of the value function among points that satisfy the constraints. This is an unsupervised problem: no labels are given in addition to the samples.

Let  $\mathbb{S}$  be the set of such samples from a space  $\mathbb{X}$ . Every  $\pmb{x} \in \mathbb{S}$  satisfies  $c(\pmb{x}) = 1$  for a classifier  $c: \mathbb{X} \to \{\pm 1\}$  that models the adherence to the set of constraints (satisfies or not). Alternatively, we can think of  $c$  as a class membership function that specifies, if a given input is within the class or not. In addition, we also consider a value function  $v$ , and for every point  $\pmb{x}'$ , such that  $\| \pmb{x}' - \pmb{x} \| \leq \epsilon$ , for a sufficiently small  $\epsilon > 0$ , we have:  $v(\pmb{x}') < v(\pmb{x})$ .

This structure leads to a co-training of  $v$  and  $c$ , such that every point  $\pmb{x}'$  in the vicinity of  $\pmb{x}$  can be used either to apply the constraint  $v(\pmb{x}') < v(\pmb{x})$  on  $v$ , or as a negative training sample for  $c$ . Which constraint to apply, depends on the other function: if  $c(\pmb{x}') = 1$ , then the first constraint applies; if  $v(\pmb{x}') \geq v(\pmb{x})$ , then  $\pmb{x}'$  is a negative sample for  $c$ . Thus, the two functions provide a training signal to each other, in an unsupervised setting, somewhat similar to the way adversarial training is done in GANs (Goodfellow et al., 2014), although the situation between  $c$  and  $h$  is not adversarial. Instead, both work collaboratively to minimize similar loss functions.

# 2 RELATED WORK

The input to our method is a set of unlabeled points. The goal is to model this set. This form of input is shared with the family of methods called one-class classification (Moya et al., 1993). The main application of these methods is anomaly detection, i.e., identifying an outlier, given a set of mostly normal (the opposite of abnormal) samples (Chandola et al., 2009).

The literature on one class classification can be roughly divided into three parts. The first includes the classical methods, mostly kernel-base methods, which were applying regularization in order to model the in-class samples in a tight way (Schölkopf et al., 2001). The second group of methods, which follow the advent of neural representation learning, employ classical one-class methods to representations that are learned in an unsupervised way (Hawkins et al., 2002; Sakurada & Yairi, 2014; Xia et al., 2015; Xu et al., 2015; Erfani et al., 2016), e.g., by using autoencoders. Lastly, a few methods have attempted to apply a suitable one-class loss, in order to learn a neural network-based representation from scratch (Ruff et al., 2018).

Despite having the same structure of the input (an unlabeled training set), our method stands out of the one-class classification and anomaly detection methods we are aware of, by optimizing a specific model that disentangles two aspects of the data: one aspect is captured by a class membership function, similar to many one-class approaches; the other aspect compares pairs of samples. This dual modeling captures the notion that the samples are not nearly random samples from some class, but also the local optimum in this class. While "the local optima of in-class points" is a class by itself, a classifier-based modeling of this class would require a higher complexity than a model that relies on the structure of the class as pertaining to local maxima, as is proved, for one example, in Sec. 4. In addition to the characterization as local maxima, the factorization between the constraints and the values also assists modeling. This is reminiscent of many other cases in machine learning, where a divide and conquer approach reduces complexity. For example, using prior knowledge on the structure of the problem, helps to reduce the complexity in hierarchical models, such as LDA (Blei et al., 2003).

While we use the term "value function", and this function is learned, we do not operate in a reinforcement learning setting, where the term value is often used. Specifically, our problem is not inverse reinforcement learning (Ng & Russell, 2000) and we do not have actions, rewards, or policies.

# 3 METHOD

Recall that  $\mathbb{S}$  is the set of unlabeled training samples, and that we seek two functions  $c$  and  $v$  such that for all  $\pmb{x} \in \mathbb{S}$  it holds that: (i)  $c(\pmb{x}) = 1$ , and (ii)  $\pmb{x}$  is a local maxima of  $v$ .

For every monotonic function  $f$ , the setting we define cannot distinguish between  $v$ , and  $f \circ v$ . This ambiguity is eliminated, if we replace  $v$  by a binary function  $h$  that satisfies  $h(\pmb{x},\pmb{x}^{\prime}) = 1$  if  $v(\pmb{x}) \geq v(\pmb{x}^{\prime})$  and  $h(\pmb{x},\pmb{x}^{\prime}) = -1$  otherwise. We found that training  $h$  in lieu of  $v$  is considerably

more stable. Note that we do not enforce transitivity, when training  $h$ , and, therefore,  $h$  can be such that no underlying  $v$  exists.

# 3.1 TRAINING  $c$  AND  $h$

When training  $c$ , the training samples in  $\mathbb{S}$  are positive examples. Without additional constraints, the recovery of  $c$  is an ill-posed problem. For example, Ruff et al. (2018) add an additional constraint on the compactness of the representation space. Here, we rely on the ability to generate hard negative points<sup>1</sup>. There are two generators  $G_{c}$  and  $G_{h}$ , each dedicated to generating negative training points to either  $c$  or  $h$ , as described in Sec. 3.2 below.

The two generators are conditioned on a positive point  $\pmb{x} \in \mathbb{S}$  and each generates one negative point per each  $\pmb{x}$ :  $\pmb{x}' = G_c(\pmb{x})$  and  $\pmb{x}'' = G_h(\pmb{x})$ . The constraints on the negative points are achieved by multiplying two losses: one pushing  $c(\pmb{x}')$  to be negative, and the other pushing  $h(\pmb{x}'', \pmb{x})$  to be negative.

Let  $\ell(p, y) := -\frac{1}{2}((y + 1)\log(p) + (1 - y)\log(1 - p))$  be the binary cross entropy loss for  $y \in \{\pm 1\}$ .  $c$  and  $h$  are implemented as neural networks trained to minimize the following losses, respectively:

$$
\mathcal {L} _ {C} := \sum_ {\boldsymbol {x} \in \mathbb {S}} \ell (c (\boldsymbol {x}), 1) + \sum_ {\boldsymbol {x} \in \mathbb {S}} \ell (c \left(G _ {c} (\boldsymbol {x})\right), - 1) \ell (h \left(G _ {c} (\boldsymbol {x}), \boldsymbol {x}\right), - 1) \tag {1}
$$

$$
\mathcal {L} _ {H} := \sum_ {\boldsymbol {x} \in \mathbb {S}} \ell (h (\boldsymbol {x}, \boldsymbol {x}), 1) + \sum_ {\boldsymbol {x} \in \mathbb {S}} \ell (c \left(G _ {h} (\boldsymbol {x})\right), - 1) \cdot \ell (h \left(G _ {h} (\boldsymbol {x}), \boldsymbol {x}\right), - 1) \tag {2}
$$

The first sum in  $\mathcal{L}_C$  ensures that  $c$  classifies all positive points as positive. The second sum links the outcome of  $h$  and  $c$  for points generated by  $G_{c}$ . It is given as a multiplication of two losses. This multiplication encourages  $c$  to focus on the cases where  $h$  predicts with a higher probability that the point  $G_{c}(\pmb{x})$  is more valued than  $\pmb{x}$ .

The loss  $\mathcal{L}_H$  is mostly similar. It ensures that  $h$  has positive values when the two inputs are the same, at least at the training points. In addition, it ensures that for the generated negative points  $\pmb{x}'$ ,  $h(\pmb{x}',\pmb{x})$  is  $-1$ , especially when  $c(\pmb{x}')$  is high.

One can alternatively use a symmetric  $\mathcal{L}_H$ , by including an additional term  $\sum_{\boldsymbol{x} \in \mathbb{S}} \ell(c(G_h(\boldsymbol{x})), -1) \ell(h(\boldsymbol{x}, G_h(\boldsymbol{x})), 1)$ . This, in our experiments, leads to very similar results, and we opt for the slightly simpler version.

# 3.2 NEGATIVE POINT GENERATION

We train two generators,  $G_{c}$  and  $G_{h}$ , to produce hard negative samples for the training of  $c$  and  $h$ , respectively. The two generators both receive a point  $x \in \mathbb{S}$  as input, and generate another point in the same space  $\mathbb{X}$ . They are constructed using an encoder-decoder architecture, see Sec. 3.4 for the exact specifications.

When training  $G_{c}$ , the loss  $-\mathcal{L}_C$  is minimized. In other words,  $G_{c}$  finds, in an adversarial way, points  $x'$ , that maximize the error of  $c$  (the first term of  $\mathcal{L}_C$  does not involve  $G_{c}$  and does not contribute, when training  $G_{c}$ ).

$G_{h}$  minimizes during training the loss  $\lambda \sum_{\pmb{x}}||\pmb{x} - G_{h}(\pmb{x})|| - \mathcal{L}_{H}$ , for some parameter  $\lambda$ . Here, in addition to the adversarial term, we add a term that encourages  $G_{h}(\pmb{x})$  to be in the vicinity of  $\pmb{x}$ . This is added, since the purpose of  $h$  is to compare nearby points, allowing for the recovery of points that are local optima. In all our experiments we set  $\lambda = 1$ .

The need for two generators, instead of just one, is verified in our ablation analysis, presented in Sec. 5. One may wonder why two are needed. One reason stems from the difference in the training loss:  $h$  is learned locally, while  $c$  can be applied anywhere. In addition,  $c$  and  $h$  are challenged by different points, depending on their current state during training. By the structure of the generators, they only produce one point per input  $x$ , which is not enough to challenge both  $c$  and  $h$ .

Algorithm 1 Training  $c$  and  $h$

Require: S: positive training points;  $\lambda$ : a trade-off parameter;  $T$ : number of epochs.

1: Initialize  $c, h, G_c$  and  $G_h$  randomly.  
2: for  $i = 1, \dots, T$  do  
3: Train  $G_{c}$  for one epoch to minimize  $-\mathcal{L}_{C}$  
4: Train  $c$  for one epoch to minimize  $\mathcal{L}_C$  
5: Train  $G_{h}$  for one epoch to minimize  $\lambda \sum_{x}||x - G_{h}(x)|| - \mathcal{L}_{H}$  
6: Train  $h$  for one epoch to minimize  $\mathcal{L}_H$  
7: return  $c, h$

# 3.3 TRAINING PROCEDURE

The training procedure follows the simple interleaving scheme presented in Alg. 1. We train the networks in turns:  $G_{c}$  and then  $c$ , followed by  $G_{h}$  and then  $h$ . Since the datasets in our experiments are relatively small, each turn is done using all mini-batches of the training dataset  $\mathbb{S}$ . The ADAM optimization scheme is used with mini-batches of size 32.

The training procedure has self regularization properties. For example, assuming that  $G_{h}(\pmb{x}) \neq \pmb{x}$ ,  $\mathcal{L}_H$  as a function of  $h$ , has a trivial global minima. This solution is to assign  $h(\pmb{x}',\pmb{x})$  to 1 iff  $\pmb{x}' = \pmb{x}$ . However, for this specific  $h$ , the only way for  $G_{h}$  to maximize  $L_{H}$  is to rely on  $c$  and  $h$  being smooth and to select points  $\pmb{x}' = G_h(\pmb{x})$  that converge to  $\pmb{x}$ , at least for some points in  $x \in \mathbb{S}$ . In this case, both  $\ell(c(G_h(\pmb{x})), -1)$  and  $\ell(h(G_h(\pmb{x}),\pmb{x}), -1)$  will become high, since  $c(\pmb{x}') \approx 1$  and  $h(\pmb{x}',\pmb{x}) \approx 1$ .

# 3.4 ARCHITECTURE

In the image experiments (MNIST, CIFAR10 and GTSRB), the networks  $G_{h}$  and  $G_{c}$  employ the DCGAN architecture of Radford et al. (2015). This architecture consists of an encoder-decoder type structure, where both the encoder and the decoder have five blocks. Each encoder (resp. decoder) block consists of a 2-strided convolution (resp. deconvolution) followed by a batch norm layer, and a ReLU activation. The fifth decoder block consists of a 2-strided convolution followed by a tanh activation instead.  $c$  and  $h$ 's architectures consist of four blocks of the same structure as for the encoder. This is followed by a sigmoid activation.

For the Cancer Genome Atlas experiment, each encoder (resp. decoder) block consists of a fully connected (FC) layer, a batch norm layer and a Leaky Relay activation (slope of 0.2). Two blocks are used for the encoder and decoder. The encoder's first FC layer reduces the dimension to 512 and the second to 256. The decoder is built to mirror this.  $c$  and  $h$  consist of two blocks, where the first FC layer reduces the dimension to 512 and the second to 1. This is followed by a sigmoid activation.

# 4 ANALYSIS

We show an example in which modeling using local-maxima-points is an efficient way to model, in comparison to the conventional classification-based approach. We then extend the framework of spectral-norm bounds, which were derived in the context of classification, to the case of unsupervised learning using local maxima.

# 4.1 MODELING USING arg max  $v$  IS BENEFICIAL

While modeling with a classifier  $c$  is commonplace, modeling a set  $\mathbb{S}$  as the local maxima of a function is much less conventional. Next, we will argue that at least in some situations, it may be advantageous. We compare the complexity of a ReLU network  $W_{2}\phi (W_{1}x + b)$  modeling a set of  $m$  real numbers. Here,  $W_{1}$  and  $W_{2}$  are linear transformations,  $b$  is a vector and  $\phi (x_{1},\ldots ,x_{n}) = (\max (0,x_{1}),\dots ,\max (0,x_{n}))$  is the ReLU activation function. We show that we can capture these points exactly as the only local maxima of a one hidden layered network with  $2m$  neurons, while a classification network would require  $3m$  neurons to achieve a good enough approximation.

Theorem 1. Let  $\mathbb{S} = \{x_i\}_{i=1}^m \subset \mathbb{R}$  be any set of points such that  $x_i < x_{i+1}$  for all  $i \in \{1, \ldots, m-1\}$ . We define  $c_{\mathbb{S}}: \mathbb{R} \rightarrow \{\pm 1\}$  to be the function, such that  $c_{\mathbb{S}}(x) = 1$  if and only if  $x \in \mathbb{S}$ . Then,

1. There is a ReLU neural network  $v: \mathbb{R} \to \mathbb{R}$  of the form  $v(x) = W_2\phi(W_1x + b)$  with  $2m$  hidden neurons such that the set of local maximum points of  $v$  is  $\mathbb{S}$ .  
2. Let  $D = q \cdot D_0 \cup (1 - q) \cdot D_1$  be a distribution that samples at probability  $q$  from  $D_0$  and probability  $1 - q$  from  $D_1$ , where  $D_0$  is any distribution supported by  $\mathbb{S}$  and  $D_1$  is any distribution supported by the segment  $[x_1 - 1, x_m + 1]$ . Then, for a small enough  $\epsilon > 0$ , every ReLU neural network  $c: \mathbb{R} \to \mathbb{R}$  of the form  $c(x) = W_2\phi(W_1x + b)$ , such that  $\mathbb{E}_{x \sim D}\mathbf{1}[c(x) \neq c_{\mathbb{S}}(x)] \leq \epsilon$  has at least  $3m$  hidden neurons.

The proof can be found in Appendix A.

# 4.2 GENERALIZATION BOUND

The following lemma provides a generalization bound that expresses the generalization of learning  $c$  along with  $h$ . See Appendix B for the exact formulation and the proof.

Lemma 1 (Informal). Let  $\mathcal{V} = \{v_{\theta} : \mathbb{R}^d \to \mathbb{R} \mid \theta \in \Theta\}$  be a class of value functions and  $\mathcal{C} = \{\mathrm{sign} \circ f_{\omega} : \mathbb{R}^d \to \{-1,1\} \mid \omega \in \Omega\}$  a class of classifiers. Assume that  $v_{\theta}$  and  $f_{\omega}$  are ReLU neural networks of fixed architectures, with parameters  $\theta$  and  $\omega$  (resp.). Let  $C(g)$  is the spectral complexity of the neural network  $g$  and  $N_{\epsilon}(\pmb{x}) := \{\pmb{u} \in \mathbb{R}^d \mid \| \pmb{u} - \pmb{x} \|_2 \leq \epsilon\}$  an  $\epsilon$ -neighborhood of  $\pmb{x}$ . Let  $D$  be a distribution of positive examples. With probability at least  $1 - \delta$  over the selection of the data  $\mathbb{S} = \{x_i\}_{i=1}^{m} \stackrel{\mathrm{i.i.d.}}{\sim} D^m$ , for every  $v_{\theta} \in \mathcal{V}$  and  $c_{\omega} \in \mathcal{C}$ , we have:

$$
\begin{array}{l} \mathbb {P} _ {\boldsymbol {x}} \left[ v _ {\theta} (\boldsymbol {x}) = \max  _ {\boldsymbol {u} \in N _ {\epsilon} (\boldsymbol {x})} v _ {\theta} (\boldsymbol {u}) \text {a n d} c _ {\omega} (\boldsymbol {x}) = 1 \right] \\ \leq \frac {1}{m} \sum_ {i = 1} ^ {m} \boldsymbol {\mathcal {I}} \left[ v _ {\theta} (\boldsymbol {x} _ {i}) = \max  _ {\boldsymbol {u} \in N _ {\epsilon} (\boldsymbol {x} _ {i})} v _ {\theta} (\boldsymbol {u}) \text {a n d} c _ {\omega} (\boldsymbol {x} _ {i}) = 1 \right] + \mathcal {O} \left(\sqrt {\frac {C \left(v _ {\theta}\right) + C \left(f _ {\omega}\right) + \log \left(\frac {m}{\delta}\right)}{m}}\right) \tag {3} \\ \end{array}
$$

The above lemma shows that the probability of  $\pmb{x} \sim D$  to be a local maxima of  $v_{\theta}$  and classified as a positive example by  $c_{\omega}$ , is at most the sum of the probability of  $\pmb{x} \in \mathbb{S}$  to be a local maxima of  $v_{\theta}$  and classified as a positive example by  $c_{\omega}$  and a penalty term. The penalty in this case is of the form  $\mathcal{O}\left(\sqrt{\frac{C(v_{\theta}) + C(f_{\omega}) + \log(m / \delta)}{m}}\right)$ , where  $m$  is the number of examples in the dataset and  $C(v_{\theta}) + C(f_{\omega})$  is the sum of the spectral norms of  $v_{\theta}$  and  $f_{\omega}$ . This suggests a tradeoff between the sum of the spectral complexities of  $v_{\theta}$  and  $f_{\omega}$  and the ability to generalize. The bound is similar asymptotically to the bounds of Neyshabur et al. (2018) and Bartlett et al. for (multi-class) supervised classification. In their bound, the penalty term is of the form  $\mathcal{O}\left(\sqrt{\frac{C(f) + \log\left(\frac{m}{\delta}\right)}{m}}\right)$ , where the (multi-class) classifier is of the form  $c(\pmb{x}) = \arg \max_{i \in \{1, \dots, t\}} f(\pmb{x})_i$ , for a neural network  $f: \mathbb{R}^d \to \mathbb{R}^t$ .

Our analysis focused on the value  $v_{\theta}$  and not on the comparator  $h$ . However, the complexities of the two are expected to be similar, since a value function can be converted to a comparator by employing  $h(\pmb{x}_1, \pmb{x}_2) = \mathrm{sign}(v_{\theta}(\pmb{x}_1) - v_{\theta}(\pmb{x}_2))$ .

# 5 EXPERIMENTS

Since we share the same form of input with one-class classification, we conduct experiments using one-class classification benchmarks. These experiments both help to understand the power of our model in capturing a given set of samples, as well as study the properties of the two underlying functions  $c$  and  $h$ .

Following acceptable benchmarks in the field, specifically the experiments done by Ruff et al. (2018), we consider single classes out of multiclass benchmarks, as the basis of one-class problems. For example, in MNIST, the set  $\mathbb{S}$  is taken to be the set of all training images of a particular

Table 1: One class experiments on the MNIST and CIFAR-10 datasets. For MNIST, there is one experiment per digit, where the training samples are the training set of this digit. The reported numbers are the AUC for classifying one-vs-rest, using the test set of this digit vs. the test sets of all other digits. For CIFAR-10, the same experiment is run with a class label, instead of the digits. Reported numbers (in all tables) are averaged over 10 runs with random initializations.  

<table><tr><td>Digit</td><td>KDE (Parzen, 1962)</td><td>AnoGAN (Schlegl, 2017)</td><td>Deep SVDD (Ruff et al., 2018)</td><td>Our c</td><td>Our h</td></tr><tr><td>0</td><td>97.1</td><td>96.6</td><td>98.0</td><td>99.1</td><td>83.5</td></tr><tr><td>1</td><td>98.9</td><td>99.2</td><td>99.7</td><td>97.2</td><td>50.7</td></tr><tr><td>2</td><td>79.0</td><td>85.0</td><td>91.7</td><td>91.9</td><td>67.1</td></tr><tr><td>3</td><td>86.2</td><td>88.7</td><td>91.9</td><td>94.3</td><td>62.4</td></tr><tr><td>4</td><td>87.9</td><td>89.4</td><td>94.9</td><td>94.2</td><td>85.7</td></tr><tr><td>5</td><td>73.8</td><td>88.3</td><td>88.5</td><td>87.2</td><td>73.3</td></tr><tr><td>6</td><td>87.6</td><td>94.7</td><td>98.3</td><td>98.8</td><td>62.8</td></tr><tr><td>7</td><td>91.4</td><td>93.5</td><td>94.6</td><td>93.9</td><td>61.6</td></tr><tr><td>8</td><td>79.2</td><td>84.9</td><td>93.9</td><td>96.0</td><td>45.8</td></tr><tr><td>9</td><td>88.2</td><td>92.4</td><td>96.5</td><td>96.7</td><td>66.8</td></tr><tr><td>Airplane</td><td>61.2</td><td>67.1</td><td>61.7</td><td>74.0</td><td>48.9</td></tr><tr><td>Automobile</td><td>64.0</td><td>54.1</td><td>65.9</td><td>74.7</td><td>64.6</td></tr><tr><td>Bird</td><td>50.1</td><td>52.9</td><td>50.8</td><td>62.8</td><td>53.2</td></tr><tr><td>Cat</td><td>56.4</td><td>54.5</td><td>59.1</td><td>57.2</td><td>51.4</td></tr><tr><td>Deer</td><td>66.2</td><td>65.1</td><td>60.9</td><td>67.8</td><td>55.0</td></tr><tr><td>Dog</td><td>62.4</td><td>60.3</td><td>65.7</td><td>60.2</td><td>58.9</td></tr><tr><td>Frog</td><td>74.9</td><td>58.5</td><td>67.7</td><td>75.3</td><td>60.7</td></tr><tr><td>Horse</td><td>62.6</td><td>62.5</td><td>67.3</td><td>68.5</td><td>58.1</td></tr><tr><td>Ship</td><td>75.1</td><td>75.8</td><td>75.9</td><td>78.1</td><td>66.9</td></tr><tr><td>Truck</td><td>76.0</td><td>66.5</td><td>73.1</td><td>79.5</td><td>70.3</td></tr></table>

digit. When applying our method, we train  $h$  and  $c$  on this set. To clarify: there are no negative samples during training.

Post training, we evaluate both  $c$  and  $h$  on the one class classification task: positive points are now the MNIST test images of the same digit used for training, and negative points are the test images of all other digits. This is repeated ten times, for digits 0-9. In order to evaluate  $h$ , which is a binary function, we provide it with two replicas of the test point.

The classification ability is evaluated as the AUC obtained on this classification task. The same experiment was conducted for CIFAR-10 where instead of digits we consider the ten different class labels. The results are reported in Tab. 1, which also states the literature baseline values reported by Ruff et al. (2018). As can be seen, for both CIFAR-10 and MNIST,  $c$  strongly captures class-membership, outperforming the baseline results in most cases.  $h$  is less correlated with class membership, resulting in much lower AUC values. However, it should not come as a surprise that  $h$  does contain such information.

Indeed, the difference in shape (single input vs. two inputs) between  $c$  and  $h$  makes them different but not independent.  $c$ , as a classifier, strongly captures class membership. We can expect  $h$ , which compares two samples, to capture relative properties. In addition,  $h$ , due to the way negative samples are collected, is expected to model local changes, at a finer resolution than  $c$ . Since it is natural to expect that the samples in the training set would provide images that locally maximize some clarity score, among all local perturbations, one can expect quality to be captured by  $h$ .

To test this hypothesis, we considered positive points to be test points of the relevant one-class, and negative points to be points with varying degree of Gaussian noise added to them. We then measure using AUC, the ability to distinguish between these two classes.

As can be seen in Fig. 1,  $h$  is much better at identifying noisy images than  $c$ , for all noise levels. This property is class independent, and in Fig. 2 (Appendix C), we repeat the experiment for all test images (not just from the one class used during training), observing the same phenomenon.

![](images/e347c95e3ceca8945725221d316c2f9ce63f23b2c31b23c0d8acbe92f1440f32.jpg)  
CIFAR-10

![](images/1089c55bb4f33ed8bf76a2a2be886844bf44689afee19a667fb2524579a5e847.jpg)  
(Automobile)

![](images/8d45d2bd8beefd350aa56c9fc204341c933a9b8b7861009ce35e72a22f049edf.jpg)  
(Airplane)

![](images/0afddf7bb0739a72c97edae4fd671b177e23b886d7c2a5880253a55c4fcdf294.jpg)  
(Cat)

![](images/4cfa392c9227d4c14fe63688b776d46e31a982251d9de2dfef91ee585925f5ef.jpg)  
(Bird)

![](images/875dd79877980f2372999f37c2ef2a42a113743283fcb1e7fc89e7f1ff72a559.jpg)

![](images/94c49114849e45c1b28ccc7d01bbc6d72c1c25de2d5e215c392f99f6eff16c26.jpg)  
(Deer)

![](images/34e0730753b4ebed59ada06e9920b435d4c2f614c4c8a4600a5934d6089c7e83.jpg)  
(Dog)

![](images/e5e920530de82fdd0ce2585602cf1ef70bd6c06feb68baa26aded7a1712d853a.jpg)  
(Frog)  
(Ship)  
Figure 1: The ability to differentiate between an in-class image and an in-class image with added noise for both  $c$  (yellow) and  $h$  (blue). The x-axis is the amount of noise (SD of the Gaussian noise). The y-axis is the AUC. As can be seen, for both CIFAR-10 and MNIST,  $h$  is much more attuned to the image quality.

![](images/9b73bcb6676be61208cf0777020db441aae475488ec3d22e213749f0ca7a0743.jpg)  
(Horse)  
(Truck)

![](images/823d77860c08890c7f5e655c6926cbc06a644402bfb8c2d5307c5cbf0abbbbc0.jpg)  
MNIST

![](images/7bb4b8774f69cb6153761d94a6a9272262b63c3d88b383a1dc85e1ef74ed0ada.jpg)

![](images/00525dbf778e5e088eedf827f6ab9db4627be6c5facad06cf92c9f467972e7bb.jpg)  
(0)

![](images/2404a55a93e499f54f554ae3c6f66d5d5b28beff8cd16c455b4aac727697adcc.jpg)  
(1)

![](images/52d3d5f39f8eac122967ce76c899ecc0e927cd0adbc9516bfa84bba48cf1650f.jpg)  
(2)

![](images/1aedf1cd0f376562b5f1b302e16d5f147acac8a161f15c44685479bf08a151ee.jpg)  
(3)

![](images/41e2b8226b3ccf5a6ab62ec8b81668bbce57a09d4bb04c9354aba911eaaeccfa.jpg)  
(4)

![](images/99edd9692f1e98fae3da66adcc0dd76e70212c04e2d12e48583fb92366f734d8.jpg)  
(5)

![](images/21bfd97c74ef21fc037138f8e62ffca3a615873f5668479e4ea87d790004b355.jpg)  
(6)  
(8)

![](images/6f1509790c302b99dad64812654df8e118b8c507827e780b85929dce6e549d04.jpg)  
(7)  
(9)

Table 2: An ablation analysis on the ten CIFAR classes (shown in order, Airplane to Truck).  

<table><tr><td></td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>Baseline c</td><td>74.0</td><td>74.7</td><td>62.8</td><td>57.2</td><td>67.8</td><td>60.2</td><td>75.3</td><td>68.5</td><td>78.1</td><td>79.5</td></tr><tr><td>Baseline h</td><td>48.9</td><td>64.6</td><td>53.2</td><td>51.4</td><td>55.0</td><td>58.9</td><td>60.7</td><td>58.1</td><td>66.9</td><td>70.3</td></tr><tr><td>c only</td><td>73.0</td><td>63.8</td><td>59.1</td><td>59.6</td><td>60.4</td><td>60.7</td><td>62.8</td><td>62.1</td><td>77.2</td><td>73.3</td></tr><tr><td>h only</td><td>35.6</td><td>51.9</td><td>50.1</td><td>48.0</td><td>48.3</td><td>48.0</td><td>68.0</td><td>54.7</td><td>75.6</td><td>73.1</td></tr><tr><td>c with Gc only</td><td>73.4</td><td>74.3</td><td>61.2</td><td>58.8</td><td>66.4</td><td>59.0</td><td>72.7</td><td>70.3</td><td>77.1</td><td>75.1</td></tr><tr><td>h with Gc only</td><td>63.7</td><td>68.3</td><td>59.2</td><td>56.6</td><td>58.8</td><td>57.4</td><td>60.7</td><td>65.5</td><td>71.3</td><td>74.2</td></tr><tr><td>c with Gh only</td><td>73.2</td><td>71.2</td><td>59.6</td><td>51.7</td><td>65.4</td><td>60.9</td><td>68.3</td><td>68.9</td><td>76.7</td><td>77.2</td></tr><tr><td>h with Gh only</td><td>56.0</td><td>65.3</td><td>55.5</td><td>53.2</td><td>50.6</td><td>58.6</td><td>54.8</td><td>58.4</td><td>65.2</td><td>71.8</td></tr></table>

We employ CIFAR also to perform an ablation analysis comparing the baseline method's  $c$  and  $h$  with four alternatives: (i) training  $c$  without training  $h$ , employing only  $G_{c}$ ; (ii) training  $h$  and  $G_{h}$  without training  $c$  nor  $G_{c}$ ; (iii) training both  $h$  and  $c$  but using only the  $G_{c}$  generator for both; and (iv) training both  $h$  and  $c$  but using only the  $G_{h}$  generator for both. The results, which can be seen in Tab. 2, indicate that the complete method is superior to the variants, since it outperforms these in the vast majority of the experiments.

Next, we evaluate our method on data from the German Traffic Sign Recognition (GTSRB) Benchmark of Houben et al. (2013). The dataset contains 43 classes, from which one class (stop signs, class #15) was used by Ruff et al. (2018) to demonstrate one-class classification where the negative class is the class of adversarial samples (presumably based on a classifier trained on all 43 classes). We were not able to obtain these samples by the time of the submission. Instead, We employ the sign data in order to evaluate three other one-class tasks: (i) the conventional task, in which a class is compared to images out of all other 42 classes; (ii) class image vs. noise image, as above, using

Table 3: Results obtained on the GTSRB dataset on three one-class tasks. Reported are AUC values in percents. DS denotes Deep-SVDD by Ruff et al. (2018).  

<table><tr><td rowspan="2">Class</td><td colspan="3">(i) Multiclass</td><td colspan="3">(ii) Noise in-class</td><td colspan="3">(iii) Noise all images.</td></tr><tr><td>c</td><td>h</td><td>DS</td><td>c</td><td>h</td><td>DS</td><td>c</td><td>h</td><td>DS</td></tr><tr><td>1</td><td>92.6</td><td>77.8</td><td>86.2</td><td>61.1</td><td>62.3</td><td>61.8</td><td>55.1</td><td>58.9</td><td>44.7</td></tr><tr><td>2</td><td>78.0</td><td>75.4</td><td>71.9</td><td>75.6</td><td>96.3</td><td>74.7</td><td>71.4</td><td>92.3</td><td>51.4</td></tr><tr><td>3</td><td>78.3</td><td>79.5</td><td>65.8</td><td>71.0</td><td>95.0</td><td>66.1</td><td>79.0</td><td>98.5</td><td>50.0</td></tr><tr><td>4</td><td>79.7</td><td>81.7</td><td>63.9</td><td>89.1</td><td>97.0</td><td>66.3</td><td>71.0</td><td>82.0</td><td>53.2</td></tr><tr><td>5</td><td>79.7</td><td>79.3</td><td>73.2</td><td>90.1</td><td>95.6</td><td>48.7</td><td>72.3</td><td>84.5</td><td>56.3</td></tr><tr><td>6</td><td>73.8</td><td>66.4</td><td>81.8</td><td>91.1</td><td>85.3</td><td>88.1</td><td>75.3</td><td>75.2</td><td>62.0</td></tr><tr><td>7</td><td>91.0</td><td>90.2</td><td>73.6</td><td>93.0</td><td>94.1</td><td>84.1</td><td>58.1</td><td>72.4</td><td>55.2</td></tr><tr><td>8</td><td>82.1</td><td>75.4</td><td>74.6</td><td>93.7</td><td>93.9</td><td>51.6</td><td>71.0</td><td>82.1</td><td>56.7</td></tr><tr><td>9</td><td>80.2</td><td>84.7</td><td>73.4</td><td>92.4</td><td>93.7</td><td>54.3</td><td>70.5</td><td>81.0</td><td>53.8</td></tr><tr><td>10</td><td>85.8</td><td>74.9</td><td>79.2</td><td>82.0</td><td>93.4</td><td>88.7</td><td>71.0</td><td>84.0</td><td>57.7</td></tr><tr><td>11</td><td>81.9</td><td>81.7</td><td>82.7</td><td>93.4</td><td>93.9</td><td>65.0</td><td>78.2</td><td>78.4</td><td>68.3</td></tr><tr><td>12</td><td>86.9</td><td>84.6</td><td>54.3</td><td>78.3</td><td>92.6</td><td>89.8</td><td>70.3</td><td>89.1</td><td>64.5</td></tr><tr><td>13</td><td>88.1</td><td>82.1</td><td>60.0</td><td>84.0</td><td>91.2</td><td>74.6</td><td>78.2</td><td>79.1</td><td>60.5</td></tr><tr><td>14</td><td>93.5</td><td>93.7</td><td>57.6</td><td>82.3</td><td>85.4</td><td>78.9</td><td>76.0</td><td>77.4</td><td>63.4</td></tr><tr><td>15</td><td>98.2</td><td>93.7</td><td>71.9</td><td>67.3</td><td>81.2</td><td>65.0</td><td>54.0</td><td>64.0</td><td>49.2</td></tr><tr><td>16</td><td>87.6</td><td>90.5</td><td>71.8</td><td>59.0</td><td>78.3</td><td>90.0</td><td>55.3</td><td>63.2</td><td>55.6</td></tr><tr><td>17</td><td>92.5</td><td>96.8</td><td>76.7</td><td>73.1</td><td>83.4</td><td>83.1</td><td>58.3</td><td>67.2</td><td>55.6</td></tr><tr><td>18</td><td>99.3</td><td>85.4</td><td>64.4</td><td>73.0</td><td>92.1</td><td>77.7</td><td>87.3</td><td>97.2</td><td>50.7</td></tr><tr><td>19</td><td>79.5</td><td>79.7</td><td>52.2</td><td>68.1</td><td>81.2</td><td>90.4</td><td>62.0</td><td>78.3</td><td>57.8</td></tr><tr><td>20</td><td>92.9</td><td>92.9</td><td>52.1</td><td>76.3</td><td>78.2</td><td>81.6</td><td>52.3</td><td>63.0</td><td>74.0</td></tr><tr><td>Avg</td><td>86.1</td><td>83.3</td><td>69.4</td><td>79.7</td><td>88.2</td><td>74.0</td><td>68.3</td><td>78.4</td><td>57.0</td></tr></table>

Gaussian noise with a fixed noise level of  $\sigma = 0.2$ ; (iii) same as (ii) only that after training on one class, we evaluate on images from all classes.

The results are presented, for the first 20 classes of GTSRB, in Tab. 3. The reported results are an average over 10 random runs. On the conventional one-class task (i), both our  $c$  and  $h$  networks outperform the baseline Deep-SVDD method, with  $c$  performing better than  $h$ , as in the MNIST and CIFAR experiments. Also following the same pattern as before, the results indicate that  $h$  captures image noise better than both  $c$  and Deep-SVDD, for both the test images of the training class and the test images from all 43 classes.

In order to explore the possibility of using the method out of the context of one-class experiments and for scientific data analysis, we downloaded samples from the Cancer Genome Atlas (https://cancergenome.nih.gov/). The data contains mRNA expression levels for over 22,000 genes, measured from the blood of 9,492 cancer patients. For most of the patients, there is also survival data in days. We split the data to  $90\%$  train and  $10\%$  test.

We run our method on the entire train data and try to measure whether the functions recovered are correlated with the survival data on the test data. While, as mentioned in Sec. 1, the gene expression optimizes a fitness function, and one can claim that gene expressions that are less fit, indicate an expected shortening in longevity, this argument is speculative. Nevertheless, since survival is the only regression signal we have, we focus on this experiment.

We compare five methods: (i) the  $h$  we recover, (ii) the  $c$  we recover, (iii) the  $h$  we recover, when learning only  $h$  and not  $c$ , (iv) the  $c$  we recover, when learning only  $c$  and not  $h$ , (v) the first PCA of the expression data, (vi) the classifier of DeepSVDD. The latter is used as baseline due to the shared form of input with our method. However, we do not perform an anomaly detection experiment.

In the simplest experiment, we treat  $h$  as a unary function by replicating the single input, as done above. We call this the standard correlation experiment. However,  $h$  was trained in order to compare two local points and we, therefore, design the local correlation protocol. First, we identify for each test datapoint, the closest test point. We then measure the difference in the target value (the patient's

Table 4: Correlation between the recovered functions and the patient's survival.  

<table><tr><td rowspan="2">Method</td><td colspan="2">Local</td><td colspan="2">Standard</td></tr><tr><td>Pearson correlation</td><td>P-value</td><td>Pearson correlation</td><td>P-value</td></tr><tr><td>Our h</td><td>0.076</td><td>0.021</td><td>0.041</td><td>0.384</td></tr><tr><td>Our c</td><td>0.020</td><td>0.520</td><td>0.029</td><td>0.444</td></tr><tr><td>Our h trained without c</td><td>0.033</td><td>0.405</td><td>0.017</td><td>0.716</td></tr><tr><td>Our c trained without h</td><td>0.029</td><td>0.444</td><td>0.031</td><td>0.420</td></tr><tr><td>First PCA of mRNA expression</td><td>0.047</td><td>0.308</td><td>0.006</td><td>0.903</td></tr><tr><td>Deep-SVDD</td><td>0.021</td><td>0.510</td><td>0.032</td><td>0.410</td></tr></table>

survival) between the two datapoints, the difference in value for unary functions (e.g., for  $c$  or for the first PCA), or  $h$  computed for the two datapoints. This way vectors of the length of the number of test data points are obtained. We use the Pearson correlation between these vectors and the associated p-values as the test statistic.

The results are reported in Tab. 4. As can be seen, the standard correlation is low for all methods. However, for local correlation, which is what  $h$  is trained to recover, the  $h$  obtained when learning both  $h$  and  $c$  is considerably more correlated than the other options, obtaining a significant p-value of 0.021. Interestingly, the ability to carve out parts of the space with  $c$ , when learning  $h$  seems significant and learning  $h$  without  $c$  results in a much reduced correlation.

# 6 DISCUSSION

The current machine learning literature focuses on models that are smooth almost everywhere. The label of a sample is implicitly assumed as likely to be the same as those of the nearby samples. In contrast to this curve-based world view, we focus on the cusps. This novel world view could be beneficial also in supervised learning, e.g., in the modeling of sparse events.

Our model recovers two functions:  $c$  and  $h$ , which are different in form. This difference may be further utilized to allow them to play different roles post learning. Consider, e.g., the problem of drug design, in which one is given a library of drugs. The constraint function  $c$  can be used, post training, to filter a large collection of molecules, eliminating toxic or unstable ones. The value function  $h$  can be used as a local optimization score in order to search locally for a better molecule.

# REFERENCES

Raman Arora, Amitabh Basu, Poorya Mianjy, and Anirbit Mukherjee. Understanding deep neural networks with rectified linear units. In International Conference on Learning Representations, 2018.  
Peter L Bartlett, Dylan J Foster, and Matus J Telgarsky. Spectrally-normalized margin bounds for neural networks. In NIPS.  
David M. Blei, Andrew Y. Ng, Michael I. Jordan, and John Lafferty. Latent dirichlet allocation. Journal of Machine Learning Research, 3:2003, 2003.  
Varun Chandola, Arindam Banerjee, and Vipin Kumar. Anomaly detection: A survey. ACM computing surveys (CSUR), 41(3):15, 2009.  
Charles Darwin. On the Origin of Species by Means of Natural Selection. Murray, London, 1859. or the Preservation of Favored Races in the Struggle for Life.  
Sarah M Erfani, Sutharshan Rajasegarar, Shanika Karunasekera, and Christopher Leckie. High-dimensional and large-scale anomaly detection using a linear one-classsvm with deep learning. Pattern Recognition, 58:121-134, 2016.

Zoltán Farkas, Dorottya Kalapis, Zoltán Bódi, Béla Szamecz, Andreea Daraba, Karola Almási, Károly Kovács, Gábor Boross, Ferenc Pál, Péter Horváth, et al. Hsp70-associated chaperones have a critical role in buffering protein production costs. eLife, 7:e29845, 2018.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NIPS, pp. 2672-2680, 2014.  
Simon Hawkins, Hongxing He, Graham Williams, and Rohan Baxter. Outlier detection using replicator neural networks. In International Conference on Data Warehousing and Knowledge Discovery, pp. 170-180. Springer, 2002.  
Sebastian Houben, Johannes Stallkamp, Jan Salmen, Marc Schlipsing, and Christian Igel. Detection of traffic signs in real-world images: The German Traffic Sign Detection Benchmark. In International Joint Conference on Neural Networks, number 1288, 2013.  
David Mcallester. Simplified pac-bayesian margin bounds. In In  $COLT$ , pp. 203-215, 2003.  
M. M. Moya, M. W. Koch, and L. D. Hostetler. One-class classifier networks for target recognition applications. NASA STI/Recon Technical Report N, 93, 1993.  
Behnam Neyshabur, Srinadh Bhojanapalli, and Nathan Srebro. A PAC-bayesian approach to spectrally-normalized margin bounds for neural networks. In ICLR, 2018.  
Andrew Y Ng and Stuart J Russell. Algorithms for inverse reinforcement learning. In ICML, pp. 663-670, 2000.  
Emanuel Parzen. On estimation of a probability density function and mode. The Annals of Mathematical Statistics, 33(3):1065-1076, 09 1962.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
Lukas Ruff, Robert Vandermeulen, Nico Goernitz, Lucas Deecke, Shoaib Ahmed Siddiqui, Alexander Binder, Emmanuel Müller, and Marius Kloft. Deep one-class classification. In ICML, 2018.  
Mayu Sakurada and Takehisa Yairi. Anomaly detection using autoencoders with nonlinear dimensionality reduction. In Proceedings of the MLSDA 2014 2nd Workshop on Machine Learning for Sensory Data Analysis, pp. 4. ACM, 2014.  
Seebock Schlegl. Unsupervised anomaly detection with generative adversarial networks to guide marker discovery. IPMI, pp. 146157, 2017.  
Bernhard Schölkopf, John C. Platt, John C. Shawe-Taylor, Alex J. Smola, and Robert C. Williamson. Estimating the support of a high-dimensional distribution. Neural Computing, 13(7):1443-1471, 2001.  
Y. Xia, X. Cao, F. Wen, G. Hua, and J. Sun. Learning discriminative reconstructions for unsupervised outlier removal. In 2015 IEEE International Conference on Computer Vision (ICCV), 2015.  
Dan Xu, Elisa Ricci, Yan Yan, Jingkuan Song, and Nicu Sebe. Learning deep representations of appearance and motion for anomalous event detection. arXiv preprint arXiv:1510.01553, 2015.
