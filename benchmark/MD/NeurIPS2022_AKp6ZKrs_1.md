# Making Look-Ahead Active Learning Strategies Feasible with Neural Tangent Kernels

Anonymous Author(s)

Affiliation

Address

email

# Abstract

We propose a new method for approximating active learning acquisition strategies that are based on retraining with hypothetically-labeled candidate data points. Although this is usually infeasible with deep networks, we use the neural tangent kernel to approximate the result of retraining, and prove that this approximation works asymptotically even in an active learning setup – approximating "look-ahead" selection criteria with far less computation required. This also enables us to conduct sequential active learning, i.e. updating the model in a streaming regime, without needing to retrain the model with SGD after adding each new data point. Moreover, our querying strategy, which better understands how the model's predictions will change by adding new data points in comparison to the standard ("myopic") criteria, beats other look-ahead strategies by large margins, and achieves equal or better performance compared to state-of-the-art methods on several benchmark datasets in pool-based active learning.

# 1 Introduction

Deep learning has drastically advanced over the past decade along with a dramatic increase in the volume of available data. It is, however, time-consuming and expensive to manually annotate large datasets. Active learning attempts to alleviate this by allowing a model to "actively" request annotation of specific data points, with the expectation that a model trained on informative points will learn a better prediction than one on a random, "passively" labeled set of the same size.

The most common form of active learning is based on using an acquisition function to measure the informativeness of potential data points or batches from some unlabeled pool. Many successful acquisition functions are based on the current model's uncertainty, such as maximum entropy [1] and Bayesian Active Learning by Disagreement (BALD) [2]. These functions are based on the expectation that training on points about which the current model is uncertain will be effective at decreasing the future model's uncertainty about similar points. This assumption, however, often does not hold: uncertain points might be inherently difficult to predict (aleatoric uncertainty) or simply too difficult for the model to learn right now.

It would be helpful, then, to see how a new data point would change a model, potentially with respect to other unseen data. One such strategy is known as Expected Model Change, which approximates how much a model's parameters will change when observing a new hypothetically-labeled data point [3, 4]. This approach, though, only considers the magnitude of change in parameters, not the model's actual outputs on the data distribution, and it only looks at the size of the first SGD step with the new data point rather than considering the full trajectory of training.

We will refer to acquisition strategies which consider full retraining of a model based on hypothetical observations as look-ahead criteria. Previous attempts to use look-ahead criteria based on the change

in model output include Expected Error Reduction [5] and Expected Model Output Change [6]. These approaches have been used only with specialized models such as Naïve Bayes and Gaussian processes, where retraining on every new candidate data point is feasible. These classes of models, however, tend to not work as well as modern neural networks, meaning these approaches are outperformed by simpler acquisition functions on stronger models.

In this work, we propose an algorithm that makes it feasible to conduct active learning with lookahead acquisition strategies using general neural networks. More specifically, we approximate a neural network using its Neural Tangent Kernel (NTK) [7], which makes it possible to obtain the outputs of a retrained neural network on a candidate data point without actually retraining the neural network. At each query step, we use the local approximation of the neural network as a proxy to compute a lookahead acquisition function, from which we query a new data point. We show that in the asymptotic regime where the width of the neural network goes to infinity, this approximation agrees with the result of actually training the network, even in an iterative setup such as in active learning. Our approximation decreases the wall-clock time more than 100 times compared to the naive computation of a lookahead acquisition function, making them far more feasible in practice.

Our proposed method outperforms existing look-ahead strategies by large margins, and achieves equal or better performance compared to the state-of-the-art methods on several benchmark datasets in pool-based active learning, including MNIST [8], SVHN [9], CIFAR10 [10], and CIFAR100 [11]. The NTK approximation also makes it possible to decouple receiving new data labels and SGD training, unlike previous active learning methods, where knowing true labels does not add any information without SGD training. This is useful when annotation is fast (yet expensive) but model training is slow, since adding data to an existing NTK approximation is far faster than retraining with SGD. Sequentially adding true labels of new data gives performance substantially better than more common batch setups.

# 2 Preliminaries

Active learning. In pool-based active learning, we have a labeled set  $\mathcal{L} = \{(x^{(i)},y^{(i)})\}_{i = 1}^{\lfloor \mathcal{L}\rfloor}$  and unlabeled set  $\mathcal{U} = \{x^{(i)}\}_{i = 1}^{\lfloor \mathcal{U}\rfloor}$ , where  $x^{(i)}$ s are inputs and  $y^{(i)}$ s are corresponding labels. At each time step  $t$ , a model  $f$  parameterized by  $\theta$  is trained on the labeled set  $\mathcal{L}$  which we denoted as  $f_{\mathcal{L}}$ , and then a data point from the unlabeled set  $\mathcal{U}$  is selected to be labeled, according to an acquisition function  $A$  measuring the "information gain" of a potential data point  $x$ :

$$
x ^ {*} = \underset {x \in \mathcal {U}} {\operatorname {a r g m a x}} A (x, f _ {\mathcal {L}}, \mathcal {L}, \mathcal {U}). \tag {1}
$$

One of the most common choices of the acquisition function is maximum entropy, which computes the estimated entropy of the unknown label of  $x$ , denoted as  $Y$ :  $A_{entropy}(x, f_{\mathcal{L}}, \mathcal{L}, \mathcal{U}) \coloneqq \mathcal{H}(Y \mid x; f_{\mathcal{L}})$ . Although the maximum entropy acquisition function often works well in practice, it does not consider how the model would change with a new queried data point, and so can overly prioritize inherently difficult, uninformative points (e.g. outliers).

Expected gradient length [12] approximates the expected model change by how large the gradient of the loss becomes when adding that data point,  $\mathbb{E}_{y\sim p_{\theta}(\cdot |x)}\| \nabla \ell_{\theta}(\mathcal{L}\cup (x,y))\|$ . BADGE [4], a more recent variant, lower-bounds the gradient norm of the last layer induced by any possible label. These approaches do not consider how the change in a model actually interacts with the data distribution: large parameter changes might give relatively small changes in predictions on most data points. They also consider a single SGD update, only roughly approximating total change over training.

One approach to alleviate this limitation is expected error reduction [EER; 5], given by

$$
A _ {E E R} (x, f _ {\mathcal {L}}, \mathcal {L}, \mathcal {U}) := - \mathbb {E} _ {y \sim p _ {\theta} (\cdot | x)} \sum_ {i = 1} ^ {| \mathcal {U} |} \mathcal {H} \left(Y ^ {(i)} \mid x ^ {(i)}; f _ {\mathcal {L} ^ {+}}\right). \tag {2}
$$

Here  $f_{\mathcal{L}^+}$  refers to the model trained on  $\mathcal{L} \cup \{(x,y)\}$ , and  $\mathcal{H}(Y^{(i)}|x^{(i)};f_{\mathcal{L}^+})$  is the (estimated) entropy of the unknown label of  $x^{(i)}$  in  $\mathcal{U}$  using that model. Maximizing  $A_{EER}$  chooses the candidate point whose label has maximal mutual information to the labels of unlabeled data,  $\mathcal{I}(Y;y|f_{\mathcal{L}})$ . If  $f_{\mathcal{L}}$  is a fairly good model and the problem is fairly difficult, the numerical value of  $A_{EER}$  is likely

dominated by the sum of irreducible (aleatoric) uncertainties over the data points; it thus might be prone to noise in estimation based on those large terms.

Freytag et al. [6] propose instead expected model output change (EMOC), based on the difference in model predictions as measured by a distance  $\mathcal{D}$  (e.g. the Euclidean distance between probability vectors). The acquisition function is defined as

$$
A _ {E M O C} (x, f _ {\mathcal {L}}, \mathcal {L}, \mathcal {U}) := \mathbb {E} _ {y \sim p _ {\theta} (\cdot | x)} \sum_ {i = 1} ^ {| \mathcal {U} |} \mathcal {D} \left(f _ {\mathcal {L}} \left(x ^ {(i)}\right), f _ {\mathcal {L} ^ {+}} \left(x ^ {(i)}\right)\right). \tag {3}
$$

Intuitively, we might hope that the change in outputs roughly cancels out the irreducible (aleatoric) uncertainty, leaving us to focus primarily on the decrease in model (epistemic) uncertainty [13]. Our techniques also apply to EER, but EMOC-like criteria performed better in our initial exploration.

Despite their advantages, look-ahead acquisition functions have been in practical reach only with certain types of models: EER has been applied to Naïve Bayes [5] and Gaussian random fields [14], and EMOC only to Gaussian processes [15, 16]. Neural networks generally outperform those models, but obtaining  $f_{\mathcal{L}^+}$  with SGD training for every candidate example in  $\mathcal{U}$  is impractically expensive. In this work, we propose to approximate  $f_{\mathcal{L}^+}$  with a more computationally efficient proxy based on neural tangent kernels, making look-ahead acquisition functions feasible on neural networks.

Neural tangent kernels. Over the past few years, connections between infinitely wide neural networks trained by gradient descent and kernel methods have become increasingly clear. The NNGP approximation of Matthews et al. [17], building off a line of work stemming from Neal [18], shows that if a network's parameters are initialized with an appropriate Gaussian distribution and only the last layer is trained, the resulting function agrees with a Gaussian process with the kernel  $\mathcal{K}(x,x^{\prime}) = \mathbb{E}_{\theta}[f_{\theta}(x)f_{\theta}(x^{\prime})]$ , where the expectation is over initializations. Jacot et al. [7], building off several immediately preceding works on optimization of wide neural networks, show that infinitely wide fully-connected neural networks also follow Gaussian process behavior, with the kernel

$$
\mathcal {K} \left(x, x ^ {\prime}\right) = \mathbb {E} _ {\theta} \left[ \left\langle \frac {\partial f _ {\theta} (x)}{\partial \theta}, \frac {\partial f _ {\theta} \left(x ^ {\prime}\right)}{\partial \theta} \right\rangle \right] \tag {4}
$$

(derivatives here denoting Jacobians with respect to the vector of all parameters  $\theta$ ). The also show that this kernel remains constant over the course of training, so that these networks evolve like a linear model under kernel gradient descent. Among many important follow-ups, Arora et al. [19] expand the results to convolutional networks with finite but large widths, and Novak et al. [20] provide an implementation for nearly-arbitrary network structures. Yang [21] and Yang and Littwin [22] later showed that Jacot et al. [7]'s findings are architecturally universal, extending the domain from fully-connected networks to a large class of architectures including ResNets and Transformers.

Unfortunately, these infinite-width versions of networks tend not to generalize as well as finite networks trained by SGD. Hence, we still want to conduct active learning for finite neural networks, rather than for pure NTK models. The infinite-width NTK also tends to be a mediocre proxy for the behavior of training a finite neural network, and so using it as a proxy for  $f_{\mathcal{L}^+}$  does not tend to work as well as we would hope (e.g. Figure 1b). We will thus instead use a local linearized approximation of the neural network, based on the empirical neural tangent kernel, as introduced next.

# 3 Active Learning using NTKs

In this section, we first define a neural network, and the linear model which (in the infinite-width limit) agrees with training that network. We will prove that in the infinite-width limit, sequentially retraining on increasing datasets – as in the active learning process – is the same as if we had trained from scratch on the final dataset. In practice, however, the infinite-width NTK does not tend to agree with finite-width SGD as well as we'd hope. We thus use a local linear approximation of the network, based on the empirical NTK, to approximate retraining in our look-ahead criteria.

Notation. We mostly use the same notation as Jacot et al. [7]. We define  $f$  as a fully-connected neural network with  $L + 1$  hidden layers numbered from 0 (input) to  $L$  (output), where each layer has  $n_0, \ldots, n_L$  neurons. This network has number of parameters  $P = \sum_{l=0}^{L-1} (n_l + 1)n_{l+1}$ : each layer

has a weight matrix  $W^{(l)} \in \mathbb{R}^{n_l \times n_{l+1}}$  and a bias vector  $b^{(l)} \in \mathbb{R}^{n_{l+1}}$ . The network is defined as  $f_{\theta}$ , where  $\theta$  collects all of the parameters:  $\theta = \cup_{l=0}^{L} \theta^l$  and  $\theta^l = \operatorname{vec}(W^{(l)}, b^{(l)})$ . We denote a labeled set  $\mathcal{L}$  as defined in Section 2 and use  $\mathcal{X} = \{x : (x, y) \in \mathcal{L}\}$  and  $\mathcal{Y} = \{y : (x, y) \in \mathcal{L}\}$  to denote its inputs and labels, respectively. We write  $f_{\mathcal{D}_1} \frac{\mathcal{D}_1 \cup \mathcal{D}_2}{t=\infty} f_{\mathcal{D}_1 \cup \mathcal{D}_2}$  to mean that  $f_{\mathcal{D}_1 \cup \mathcal{D}_2}$  is trained using gradient descent until convergence on the dataset  $\mathcal{D}_2$ , starting from  $f_{\mathcal{D}_1}$ .

Unless otherwise specified, the loss function associated with gradient descent training is assumed to be the squared loss,  $\ell (\mathcal{V},f(\mathcal{X}))\coloneqq \frac{1}{2}\| \mathcal{V} - f(\mathcal{X})\| _2^2$ . This allows for far more efficient usage of NTKs, but Hui and Belkin [23] demonstrate that, with the right learning rate,  $L_{2}$  loss is just as effective as cross-entropy for many vision and natural language processing tasks.

Neural networks in the infinite width limit. Jacot et al. [7] show that in the infinite width limit, when the network  $f$  is trained using gradient descent with the squared loss on a labeled set  $\mathcal{L}$ , the outputs of the network on any arbitrary point  $x$  evolve as

$$
f _ {t} (x) = f _ {0} (x) + \mathcal {K} (x, \mathcal {X}) \mathcal {K} (\mathcal {X}, \mathcal {X}) ^ {- 1} (\mathcal {I} - e ^ {- t \mathcal {K} (\mathcal {X}, \mathcal {X})}) (\mathcal {Y} - f _ {0} (\mathcal {X})), \tag {5}
$$

where  $\mathcal{K}$  is the NTK of (4), which stays constant during training. Here we treat inputs  $\mathcal{X}$  as a matrix and labels  $\mathcal{Y}$  as a vector in corresponding order.

Consider sequentially training a network on increasing datasets  $S_{1} \subset S_{2} \subset \dots \subset S_{C}$ , warm-starting each time from the result of training on the previous dataset. We build on the results of Jacot et al. [7] to show that the final network from this process is asymptotically equivalent, in the infinite-width limit, to training the same network from scratch (cold-start) on the last, biggest dataset  $S_{C}$ . This justifies our efficient approximation of the retrained network, even after we have gone through many iterations of active learning. See Appendix A.1 for a formal statement and proof.

Theorem 3.1 (Informal). Let  $S_{1}, S_{2}, \ldots, S_{C}$  be  $C$  datasets such that for all  $i > j$ ,  $S_{j} \subset S_{i}$ . Let  $f_{0}$  be a randomly initialized network. Let  $f_{S_{1,2,\dots,C}}$  be the network resulting from starting at  $f_{0}$  and training  $f$  using gradient descent on datasets  $S_{1}, \ldots, S_{C}$  sequentially until convergence:

$$
f _ {0} \xrightarrow [ t = \infty ]{S _ {1}} f _ {S _ {1}} \xrightarrow [ t = \infty ]{S _ {2}} f _ {S _ {1, 2}} \dots \xrightarrow [ t = \infty ]{S _ {C}} f _ {S _ {1, 2, \ldots , C}}.
$$

Also, let  $f_{S_C}$  be  $f_0 \xrightarrow[T_t = \infty]{S_C} f_{S_C}$ . Assuming that  $f$  has  $L + 1$  layers such that taking  $n_0, n_1, \ldots, n_L \to \infty$  sequentially, we have for any arbitrary data point  $x$  that  $f_{S_C}(x) = f_{S_{1,2,\dots,C}}(x)$ .

NTK approximations of retrained networks. Armed with Theorem 3.1, we can now approximate the outputs of a neural network from retraining. Suppose we have a labeled set  $\mathcal{L}$  and unlabeled set  $\mathcal{U}$  as defined in Section 2. A neural network  $f_{\mathcal{L}}$  (whose layers are assumed to be infinitely wide) has been trained on  $\mathcal{L}$  as  $f_0\xrightarrow[\substack{t=\infty\\ }}f_{\mathcal{L}}$ . We are interested in characterizing the outcome after retraining this neural network using each of the data points in  $\mathcal{U}$ . In other words, for each  $x^{\prime}\in \mathcal{U}$  with a hypothetical label  $y^\prime$  and  $\mathcal{L}^{+}:= \mathcal{L}\cup (x^{\prime},y^{\prime})$ , we would like to know what  $f_{\mathcal{L}}\xrightarrow[\substack{t=\infty\\ }}f_{\mathcal{L}^{+}}$  looks like.

According to (5), the outputs of  $f_{\mathcal{L}}$  for an arbitrary data point  $x$  can be formulated as

$$
f _ {\mathcal {L}} (x) = f _ {0} (x) + \mathcal {K} (x, \mathcal {X}) \mathcal {K} (\mathcal {X}, \mathcal {X}) ^ {- 1} (\mathcal {Y} - f _ {0} (\mathcal {X})). \tag {6}
$$

Let  $\mathcal{X}^+$  and  $\mathcal{Y}^+$  be inputs and labels of  $\mathcal{L}^+$ . Then, based on Theorem 3.1, we can conclude that

$$
f _ {\mathcal {L} ^ {+}} (x) = f _ {0} (x) + \mathcal {K} (x, \mathcal {X} ^ {+}) \mathcal {K} (\mathcal {X} ^ {+}, \mathcal {X} ^ {+}) ^ {- 1} (\mathcal {Y} ^ {+} - f _ {0} (\mathcal {X} ^ {+})). \tag {7}
$$

As  $\mathcal{K}$  remains constant in the infinite-width regime, most of the relevant quantities can be reused:

$$
\mathcal {K} (x, \mathcal {X} ^ {+}) = \left[ \begin{array}{l l} \mathcal {K} (x, \mathcal {X}) & \mathcal {K} (x, x ^ {\prime}) \end{array} \right], \quad \mathcal {K} (\mathcal {X} ^ {+}, \mathcal {X} ^ {+}) = \left[ \begin{array}{l l} \mathcal {K} (\mathcal {X}, \mathcal {X}) & \mathcal {K} (\mathcal {X}, x ^ {\prime}) \\ \mathcal {K} (x ^ {\prime}, \mathcal {X}) & \mathcal {K} (x ^ {\prime}, x ^ {\prime}) \end{array} \right]. \tag {8}
$$

Remark 3.2. In the infinite width limit, we can analytically obtain the predictions of a neural network after retraining on additional data points by augmenting its corresponding NTK using (8).

Several works have shown that while being an inspiring theoretical motivation, neural networks in the infinite width limit do not work as well as their finite-width counterparts [19, 24, 25]. Although

Arora et al. [19] prove non-asymptotic bounds between these infinite width neural networks and their corresponding finite-width networks, empirical studies in [21, 26] have shown that this approximation may not be effective for practical network widths. (If it were, we wouldn't need a network at all, and would simply do all our learning with a Gaussian process based on the NTK.) Thus, we would like to be able to efficiently characterize the outputs of a finite neural network after retraining.

Lee et al. [26] showed the first-order Taylor expansion of a neural network around its initialization (a linearized neural network) has training dynamics converging to that of the neural network as the width grows. That is, let  $f_{t}$  denote the network which has been trained with gradient descent for  $t$  steps, and  $f_{t}^{lin}$  the result of training the linearized network for  $t$  steps. They prove that, under some regularity conditions and when the learning rate  $\eta$  is less than a certain threshold,

$$
\sup  _ {t \geq 0} \| f _ {t} (x) - f _ {t} ^ {\text {l i n}} (x) \| _ {2} = \sup  _ {t \geq 0} \| \Theta_ {t} - \Theta_ {0} \| _ {F} = \mathcal {O} \left(\frac {1}{\sqrt {\operatorname* {w i d t h}}}\right). \tag {9}
$$

Informally, when  $f$  is wide enough, its predictions can be approximated by those of the linearized network  $f^{lin}$ . This is attractive since  $f^{lin}$  has simple training dynamics, converging to

$$
f _ {\mathcal {L}} ^ {l i n} (x) = f _ {0} (x) + \Theta_ {0} (x, \mathcal {X}) \Theta_ {0} (\mathcal {X}, \mathcal {X}) ^ {- 1} (\mathcal {Y} - f _ {0} (\mathcal {X})), \tag {10}
$$

where  $\Theta_0$  is the empirical tangent kernel of  $f_0, \Theta_0(x,y) \coloneqq \nabla_\theta f_0(x)\nabla_\theta f_0(y)^\top$ .

In the infinite-width limit, the empirical NTK  $\Theta_t$  converges to the NTK  $\mathcal{K}$  almost surely, leading (10) to become equivalent to (6). In finite-width regimes, however, it is intuitive to expect that the local approximation may still be able to handle local retraining, such as  $f_{\mathcal{L}} \xrightarrow{\mathcal{L} \cup \{(x,y)\}} f_{\mathcal{L} \cup \{(x,y)\}}$ , even when the correspondence to the infinite-width limit  $\mathcal{K}$  is loose. We thus model retraining of a finite network by augmenting the empirical NTK as in (8), as justified by Theorem 3.1 and Remark 3.2. Specifically, we approximate the finite width network  $f_{\mathcal{L}^+}(x)$  defined by  $f_{\mathcal{L}} \xrightarrow{\mathcal{L}^+} f_{\mathcal{L}^+}$  using

$$
f _ {\mathcal {L} ^ {+}} (x) \approx f _ {\mathcal {L} ^ {+}} ^ {\text {l i n}} (x) = f _ {\mathcal {L}} (x) + \Theta_ {\mathcal {L}} (x, \mathcal {X} ^ {+}) \Theta_ {\mathcal {L}} \left(\mathcal {X} ^ {+}, \mathcal {X} ^ {+}\right) ^ {- 1} \left(\mathcal {Y} ^ {+} - f _ {\mathcal {L}} (\mathcal {X} ^ {+})\right) \tag {11}
$$

where  $\Theta_{\mathcal{L}}$  is the empirical NTK obtained from the parameters of  $f_{\mathcal{L}}$ .

Efficient computation. To avoid computing the full empirical NTK with the shape of  $LC \times LC$  where  $L$  is the size of the labeled set and  $C$  is the number of classes, we use the "single-logit" approximation as explained in Wei et al. [27] (Section 2.3) in which we only compute the corresponding NTK of the first logit of the network, i.e.  $\Theta(x,y) = \nabla_{\theta} f^{(1)}(x)^{\top} \nabla_{\theta} f^{(1)}(y) \otimes I_C$  where  $f^{(1)}(x)$  refers to the first neuron of the output of  $f$  on the datapoint  $x$ . For a kernel regression, we can take the kronecker product out, which decreases the memory and time complexity of computing the NTK by an order of  $\mathcal{O}(C^2)$ .

To compute (11), we must solve linear systems with  $\Theta_{\mathcal{L}}(\mathcal{X}^{+},\mathcal{X}^{+})$ . This can also be done efficiently using the block structure of  $\Theta_{\mathcal{L}}(\mathcal{X}^{+},\mathcal{X}^{+})$ : letting  $\mathbf{v} = \Theta_{\mathcal{L}}(\mathcal{X},\mathcal{X})\Theta_{\mathcal{L}}(\mathcal{X},x^{\prime})$  and  $u = \Theta_{\mathcal{L}}(x^{\prime},x^{\prime}) - \Theta_{\mathcal{L}}(x^{\prime},\mathcal{X})\Theta_{\mathcal{L}}(\mathcal{X},\mathcal{X})^{-1}\Theta_{\mathcal{L}}(\mathcal{X},x^{\prime})$ ,

$$
f _ {\mathcal {L} ^ {+}} ^ {l i n} (x) = f _ {\mathcal {L}} ^ {l i n} (x) + \frac {1}{u} \left(\Theta_ {\mathcal {L}} (x, \mathcal {X}) \mathbf {v} - \Theta_ {\mathcal {L}} \left(x, x ^ {\prime}\right)\right) \left(\mathbf {v} ^ {T} \left(\mathcal {Y} - f _ {\mathcal {L}} (\mathcal {X})\right) - \left(y ^ {\prime} - f _ {\mathcal {L}} \left(x ^ {\prime}\right)\right)\right). \tag {12}
$$

Thus, rather than inverting  $\Theta_{\mathcal{L}}(\mathcal{X}^{+},\mathcal{X}^{+})$  for each  $x\in \mathcal{U}$ , we can invert  $\Theta_{\mathcal{L}}(\mathcal{X},\mathcal{X})$  only once, then use some matrix multiplications to find  $f_{\mathcal{L}^+}^{lin}(x)$  for each query point.

Time complexity. Let  $L$  be the size of the labeled set,  $U$  of the unlabeled set,  $P$  the number of model parameters, and  $E$  the number of training epochs. We assume that  $P > L$  as we are using overparameterized models. The dominant term for naive SGD retraining is  $\mathcal{O}(LUPE)$  whereas the proposed method using block structure in (12) is dominated by computing  $\Theta_{\mathcal{L}}(\mathcal{X}_{\mathcal{U}},\mathcal{X}_{\mathcal{L}})$  in  $\mathcal{O}(LUP)$  time if  $U > L$ , otherwise  $\mathcal{O}(L^2 P)$  time for  $\Theta_{\mathcal{L}}(\mathcal{X}_{\mathcal{L}},\mathcal{X}_{\mathcal{L}})$ . If  $U > L$ , then the proposed method is faster by an order of  $\mathcal{O}(E)$ . In the other case, if  $L > EU$ , one can use the conjugate gradient kernel regression solvers as in Rudi et al. [28] and Gardner et al. [29] that further reduce the time complexity of our proposed method to  $\mathcal{O}(L\sqrt{L} P)$ , which is faster than the naive SGD training by an order of  $\mathcal{O}(UE / \sqrt{L})$ . We provide a more detailed analysis in Appendix B.

Algorithm 1: Active learning using NTKs  
Input: Initialize a model  $f_{0}$  , labeled pool  $\mathcal{L}_0$  , and unlabeled pool  $\mathcal{U}_0$    
Train  $f_{0}$  on the initial labeled set  $\mathcal{L}_0$  , using SGD, to obtain  $f_{\mathcal{L}_0}$    
for  $t = 0$  to  $T - 1$  do Compute  $\Theta_{\mathcal{L}_t}(\mathcal{X}_{\mathcal{U}_t},\mathcal{X}_{\mathcal{L}_t})$ $\Theta_{\mathcal{L}_t}(\mathcal{X}_{\mathcal{L}_t},\mathcal{X}_{\mathcal{L}_t})$  and  $\Theta_{\mathcal{L}_t}(\mathcal{X}_{\mathcal{L}_t},\mathcal{X}_{\mathcal{L}_t})^{-1}$  for  $x^{\prime}$  in  $\mathcal{U}_t$  do Estimate the label for  $x^{\prime}$  as  $y^\prime = \mathrm{argmax}f_{\mathcal{L}_t}(x')$  Approximate  $f_{\mathcal{L}_t\cup (x',y')}$  using (11) and (12) for faster computation Track  $x^{*}$  as the point with maximal  $A_{MLMOC}(x',f_{\mathcal{L}_t},\mathcal{L}_t,\mathcal{U}_t)$  we've seen so far end for Obtain the label  $y^{*}$  for  $x^{*}$  from the oracle Update the labeled set  $\mathcal{L}_{t + 1} = \mathcal{L}_t\cup \{(x^*,y^*)\}$  and unlabeled set  $\mathcal{U}_{t + 1} = \mathcal{U}_t\setminus \{(x^*,y^*)\}$  Train  $f_{\mathcal{L}_t}$  on  $\mathcal{L}_{t + 1}$  , using SGD, to obtain  $f_{\mathcal{L}_{t + 1}}$    
end for

MLMOC querying. Instead of the EMOC acquisition function (3), we define a new function more suitable for a linearized network, which we term Most Likely Model Output Change (MLMOC):

$$
A _ {M L M O C} \left(x ^ {\prime}, f _ {\mathcal {L}}, \mathcal {L} _ {t}, \mathcal {U} _ {t}\right) := \sum_ {x \in \mathcal {U}} \| f _ {\mathcal {L}} (x) - f _ {\mathcal {L} ^ {+}} ^ {\text {l i n}} (x)) \| _ {2} \tag {13}
$$

where  $\mathcal{L}^{+} = \mathcal{L}\cup \{(x^{\prime},y^{\prime})\}$  with  $y^\prime = \mathrm{argmax}f_{\mathcal{L}}(x^{\prime})$  . We only consider the most likely pseudolabel, instead of the expectation; this saves a significant amount of computation, by a factor of the number of classes, but empirically does not hurt accuracy. We also suspect that, when  $f_{\mathcal{L}}$  is already reasonably accurate, we can "trust" the most likely label more than we can trust accurate estimation of probabilities for low-probability losses, especially when training networks with  $L_{2}$  loss. Algorithm 1 shows pseudo-code for the proposed algorithm.

Sequential query strategy. In the active learning literature, each query of new data points is essentially always followed by retraining the underlying model. This, however, may not be practical for the cases where annotation is quick but SGD training of a large neural network is slow. Although there has been significant effort towards batch selection strategies based on finding diverse examples to query in a batch [e.g. 4, 30, 31], which among other benefits can minimize the number of times we must retrain with SGD, they cannot fully take advantage of low-latency annotations if they are available. (This may be the case if, for instance, labeling tasks can be quickly pushed out to a pool of human labelers available for many such annotation tasks at once, or other cases where a limited number of queries are desired to avoid expense, damage to a system being measured, or so on, but those individual measurements can still be taken promptly.)

Our proposed linearized network can take advantage of sequentially added annotations without requiring SGD retraining steps. Given an empirical NTK  $\Theta$ , adding another training point  $(x, y)$  requires nothing more than adding a row and column to the kernel. As a result, computing  $f^{lin}$  and following  $A_{MLMOC}$  with an additional data point  $(x, y)$  is almost instant. This approximation can utilize the information from the label  $y$  directly while batch selection strategies cannot. (After adding a substantial number of points, we will want to retrain, but it is not needed after every new data point.) We experimentally demonstrates the effectiveness of this sequential query strategy in Section 5.

# 4 Related Work

Two main approaches in pool-based active learning are uncertainty and representation-based methods. The general goal of the uncertainty-based methods is to query the most "informative" data points, thus, their acquisition functions vary depending on how to measure the informativeness of a data point. As simple but effective uncertainty-based querying methods, posterior probability [32, 33], entropy [1], margin sampling [34] and least confident [12], have been widely used in practice. They do not, however, take into account how a candidate data point would change a model, or how this change would interact with unseen examples. More advanced querying methods have thus been proposed as discussed in Section 2. There are also Bayesian uncertainty-based methods, particularly Bayesian Active Learning by Disagreement (BALD) [2], which measures the mutual information

between a new data point and model parameters, which Gal et al. [35] propose to estimate efficiently with Monte Carlo dropout networks [36].

The goal of the representation-based methods is to query examples that are the most "representative" among the unlabeled set  $\mathcal{U}$ , hoping that doing well on those examples leads to doing well on the whole unseen dataset. Sener and Savarese [37] define a core set as a batch of images that minimize the distance between unlabeled and labeled images if added to the labeled pool. Kirsch et al. [31] further expand BALD to a batch setting using a greedy algorithm. Biryik et al. [30] propose to use determinantal point processes to target diversity of the queried examples while maintaining informativeness. BADGE [4] measures the uncertainty using a gradient norm but also encourage the diversity of queried images using k-means++ seeding [38].

Other approaches include that of Yoo and Kweon [39], who employ an additional loss prediction module to an existing neural network. However, the additional loss computation module can make training unstable (as we observed in some experiments). Tran et al. [40] improves BALD by adding a variational autoencoder and auxiliary classifier generative adversarial network, from which a synthesized data point  $x'$  similar to the queried image  $x^*$  is generated.

Our proposed method is different from the previous works in that it "looks one step ahead" instead of relying on the current state of a model. Perhaps the most similar proposal to ours is that of Borsos et al. [41], who propose using the infinite NTK to solve a bi-level optimization problem for semi-supervised batch active learning. Aside from the difference in our objective functions, however, using the infinite NTK to approximate a neural network in active learning does not seem to be a good choice; as the size of training set increases, the error of the approximation can be quite large. We explore the difference between infinite and empirical NTKs experimentally in Section 5.

# 5 Experiment Results

Datasets. To demonstrate the effectiveness of the proposed method, we provide experimental results on three benchmark datasets for classification tasks: MNIST [8], SVHN [9], CIFAR10 [10], and CIFAR100 [11]. We provide the detailed description of each dataset in Appendix C.

Implementation. We implement a pipeline for the proposed method using PyTorch [42] and Jax [43]; our neural networks  $f$  are implemented in PyTorch, whereas the linearized models  $f^{lin}$  are implemented in Jax with the neural-tangents library [44]. We employ a ResNet18 [45] and WideResNet [46] with one or two layers and maximum width of 640, which is wide enough for a linearized neural network to be a good approximation, while still being powerful enough for these datasets. As we will see in our experiments, even for narrower networks, the proposed method outperforms random acquisition strategy by large margins.

As mentioned earlier, we use  $L_{2}$  loss, rather than cross-entropy; this allows for a faster NTK approximation, as in the formulas of Section 3, rather than requiring differential equation solvers. Appropriately tuned  $L_{2}$  loss is as effective as cross-entropy [23]. To ensure that the empirical NTK of the training data  $(\mathcal{X}_{\mathcal{L}})$  is positive semidefinite, in batch normalization layers, we freeze the current running statistics. We implement LL4AL [39] and BADGE [4] based on their publicly available code. We provide anonymous code for the full pipeline in the supplementary material.

Query scheme. Following active learning literature, for each query step, we randomly draw a subset from an unlabeled set, and query a fixed number of examples from the subset. To build a batch, we take the points with the highest  $A_{MLMOC}$  scores, which we found to perform well without any batch diversity requirements. At each cycle, we query 20, 1000, 1000, and 1000 new data points on MNIST, SVHN, CIFAR10, and CIFAR100 from the subset of 4000, 6000, 6000, and 6000 unlabeled data points, and initialize the labeled set  $\mathcal{L}$  with 100, 1000, 1000, and 10000 data points.

Making look-ahead acquisitions feasible. In Figure 1a, we demonstrate the NTK approximation indeed makes it feasible to use look-ahead acquisition functions. We run the MLMOC acquisition function with naïve SGD retraining (Naive) and proposed NTK approximation with (NTK) and without (NTK w/o Block) block computation in (12) on MNIST. For the naïve version, we retrain the neural network for 15 epochs using SGD. The line plots represent accuracy whereas bar plots represent wall-clock time to query data, using a NVIDIA V100 GPU. The NTK approximation is not

![](images/0da2def3cd70d3c03e7eb0a47d2f064fe5b520be045f307ff32a17fb42edad2d.jpg)  
(a) Naive look-ahead acquisition versus NTK approximation. Bars show runtime per cycle.  
Figure 1: Comparisons of several related approaches on MNIST.

![](images/3d70a9558b47bc10895a8c8057b8183dc56bc3a8b3a5afbb5df1cdd058ed587c.jpg)  
(b) MLMOC on various models, along with random-acquisition baselines.

only computationally efficient (more than 100 times faster than Naïve) but also yields significantly more accurate models. Although one might expect that Naïve is an upper bound for the NTK approximation, as it actually "looks ahead," in reality it is hard to retrain a neural network until convergence for every candidate data point; a "proper" look-ahead method should perhaps look at an ensemble of several training runs, with different learning rate schedules, etc.

Comparison of look-ahead acquisitions. Having seen that the empirical NTK effectively approximates (even outperforms) the Naïve look-ahead acquisition function, we now show the superiority of the empirical NTK over existing look-ahead methods. In Figure 1b, we provide four look-ahead methods using MLMOC acquisition function along with their corresponding random baselines (Random). NN refers to a neural network-based model whereas GP refers to a model based on Gaussian process. Instead of using empirical NTK as we propose (denoted as NN NTK, blue), Borsos et al. [41] use infinite NTK (NN Inf NTK, orange). Käding et al. [15] approximate SGD retraining using only one-step SGD update (NN 1-step, green) whereas Käding et al. [16] exactly compute look-ahead acquisition using GP for regression problems. We modify their model for classification tasks, use an infinite NTK, and denote it as GP (purple). As shown in Figure 1b, ours is the only of these methods to outperform random baselines here. Neither infinite NTK nor 1-step approximation are good approximation for retraining steps, especially as more data points are added. Also, although GP [16] works well on regression tasks, it struggles on classification tasks.

Comparison with state-of-the-art. In Figure 2, we compare the proposed NTK active learning to state-of-the-art methods – Random, Entropy, Margin sampling (Margin) [47], BADGE [4], and LL4AL [39] – on SVHN, CIFAR10, and CIFAR100 with different architectures; we provide additional results in Appendix D. For clarity, we show the difference between each method and Random. The numbers below Random line give the accuracies of Random at each cycle. As mentioned earlier, training LL4AL is often unstable due to loss computation module; in particular, it performs too poorly to show on the same plot for Figure 2b and Figure 2c. We show the instability of training LL4AL with varying learning rate in Appendix E. Each experiment is run for six different seeds; lines show the mean performance, and shading shows a  $95\%$  confidence interval for the mean. Figure 2(a) shows that the NTK method outperforms all the state-of-the-art methods throughout training on SVHN. A similar pattern is observed on CIFAR10 and CIFAR100; NTK is consistently comparable to or better than the best competitor methods, far outperforming existing look-ahead approaches.

Sequential query strategy. As mentioned in Section 3, one great advantage of the NTK approximation is that it can exploit additional annotations without needing to run SGD training for a neural network, which cannot be done in existing active learning methods for neural networks. On MNIST, Figure 3a compares sequential NTK where the oracle provides true label for every new queried data point, to batch NTK where annotation is done batch-wise as with a normal active learning setting. Sequential NTK outperforms batch NTK; especially, sequential NTK reaches more than  $94\%$  in one cycle, and in cycle 3, it has already reached  $96\%$  accuracy, where batch NTK plateaus. We expect the sequential NTK can be useful in practice when annotation is much faster than SGD training.

![](images/67817bdc3ffd0a175a8abcf679f66c093e3ca4c00b5c3727bc7cfe3471819fa6.jpg)  
(a) SVHN: 1-layer WideResNet

![](images/cd8a061fb6f732ffd384cf1e08f7a7cd135a0d14fdc68a0a640dca00514d7981.jpg)  
(b) CIFAR10: 2-layer WideResNet

![](images/0085bb8aee236dfffac84e18373bf2eb2f7a14f785ee7b86dbe10a3dc1138b5a.jpg)  
(c) CIFAR100: ResNet18

![](images/6ce7fc042064e1e47c8234536511c832be98658cb15655eef9d624423b7e5b4b.jpg)  
Figure 2: Comparison of the-state-of-the-art active learning methods on various benchmark datasets. Vertical axis shows difference from random acquisition, whose accuracy is shown in text.  
(a) Accuracies of sequential versus batch querying strategies, compared to a random baseline.  
Figure 3: Further comparisons of algorithm variants on MNIST.

![](images/2670a9384c1189c3d923a13014086d55d124dad205d73559eda82d4a44316bb2.jpg)  
(b) Our method's improvements in accuracy over random acquisition for WideResNets of various width.

Varying model widths. The NTK approximation gets closer to the dynamics of a neural network as the width of the neural network increases. But, since very wide networks are computationally expensive in practice, it is important to empirically demonstrate that the NTK approximation works even with reasonably narrow networks. To this end, we vary the maximum width of the WideResNet with one block layer we use for the experiments as shown in Figure 3b. We can observe that regardless of the maximum width, the NTK method significantly outperforms Random strategy until convergence, indicating the approximation is good enough for active learning purposes.

# 6 Conclusion

We propose an algorithm to make look-ahead acquisition strategies feasible for active learning with fairly general neural networks. We prove that the outputs of wide enough neural networks can be efficiently approximated using a linearized network using empirical neural tangent kernels. We empirically show that this approximation is at least 100 times faster than naive computation of look-ahead strategies, while even being more accurate. Armed with this approximation, we significantly outperform previous look-ahead strategies, and achieve equal or better performance compared to the state-of-the-art methods on four benchmark datasets in active learning regime. We also propose a new sequential active learning algorithm that decouples querying and SGD training for the first time.

One limitation of the NTK approximation is that it is still slower than common "myopic" active learning methods such as entropy and BADGE. Compared to our proposed method where it takes  $\mathcal{O}(L\sqrt{LP})$  using conjugate gradient kernel regression solvers, entropy requires  $O(UP)$  time (using notations in Section 3). Improving time complexity of the NTK approximation would be an interesting future work along with devising better look-ahead acquisition functions that improve performance.

# References

[1] Dan Wang and Yi Shang. "A New Active Labeling Method for Deep Learning." IJCNN. 2014, pp. 112-119.  
[2] Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Mate Lengyel. "Bayesian Active Learning for Classification and Preference Learning" (2011). arXiv: 1112.5745.  
[3] Burr Settles, Mark Craven, and Soumya Ray. “Multiple-instance Active Learning.” 2007, pp. 1289–1296.  
[4] Jordan T Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal. "Deep batch active learning by diverse, uncertain gradient lower bounds." ICLR (2020).  
[5] Nicholas Roy and Andrew McCallum. "Toward Optimal Active Learning through Monte Carlo Estimation of Error Reduction." 2001, pp. 441-448.  
[6] Alexander Freytag, Erik Rodner, and Joachim Denzler. "Selecting Influential Examples: Active Learning with Expected Model Output Changes." ECCV. 2014, pp. 562-577.  
[7] Arthur Jacot, Franck Gabriel, and Clément Hongler. "Neural Tangent Kernel: Convergence and Generalization in Neural Networks." NeurIPS. 2018.  
[8] Li Deng. "The Mnist Database of Handwritten Digit Images for Machine Learning Research." IEEE Signal Processing Magazine (2012), pp. 141-142.  
[9] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. "Reading Digits in Natural Images with Unsupervised Feature Learning." NIPSW (2011).  
[10] Alex Krizhevsky. "Learning Multiple Layers of Features from Tiny Images" (2009).  
[11] Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. "CIFAR-100 (Canadian Institute for Advanced Research)"().  
[12] Burr Settles. "Active Learning Literature Survey" (2009).  
[13] Vu-Linh Nguyen, Sébastien Destercke, and Eyke Hüllermeier. "Epistemic Uncertainty Sampling." ICDS. 2019.  
[14] Xiaojin Zhu, John Lafferty, and Zoubin Ghahramani. "Combining Active Learning and Semi-supervised Learning Using Gaussian Fields and Harmonic Functions." ICMLW. Vol. 3. 2003.  
[15] Christoph Käding, Erik Rodner, Alexander Freytag, and Joachim Denzler. "Active and Continuous Exploration with Deep Neural Networks and Expected Model Output Changes" (2016).  
[16] Christoph Käding, Erik Rodner, Alexander Freytag, Oliver Mothes, Björn Barz, Joachim Denzler, and Carl Zeiss AG. “Active Learning for Regression Tasks with Expected Model Output Changes.” BMVC. 2018, p. 103.  
[17] Alexander G de G Matthews, Mark Rowland, Jiri Hron, Richard E Turner, and Zoubin Ghahramani. "Gaussian Process Behaviour in Wide Deep Neural Networks." *ICLR*. 2018.  
[18] Radford M Neal. “Priors for Infinite Networks.” Bayesian Learning for Neural Networks. 1996.  
[19] Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, and Ruosong Wang. "On Exact Computation with an Infinitely Wide Neural Net." NeurIPS. 2019.  
[20] Roman Novak, Lechao Xiao, Jiri Hron, Jaehoon Lee, Alexander A Alemi, Jascha Sohl-Dickstein, and Samuel S Schoenholz. “Neural Tangents: Fast and Easy Infinite Neural Networks in Python.” *ICLR*. 2020.  
[21] Greg Yang. Tensor Programs II: Neural Tangent Kernel for Any Architecture. 2020. arXiv: 2006.14548.  
[22] Greg Yang and Etai Littwin. Tensor Programs IIb: Architectural Universality of Neural Tangent Kernel Training Dynamics. 2021. arXiv: 2105.03703.  
[23] Like Hui and Mikhail Belkin. "Evaluation of Neural Architectures Trained with Square Loss vs Cross-entropy in Classification Tasks." ICLR (2021).  
[24] Zhiyuan Li, Ruosong Wang, Dingli Yu, Simon S Du, Wei Hu, Ruslan Salakhutdinov, and Sanjeev Arora. "Enhanced Convolutional Neural Tangent Kernels" (2019). arXiv: 1911.00809.  
[25] Eran Malach, Pritish Kamath, Emmanuel Abbe, and Nathan Srebro. "Quantifying the Benefit of Using Differentiable Learning over Tangent Kernels." ICML. 2021. arXiv: 2103.01210.  
[26] Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. "Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent." NeurIPS. 2019.

[27] Alexander Wei, Wei Hu, and Jacob Steinhardt. "More Than a Toy: Random Matrix Models Predict How Real-World Neural Representations Generalize." arXiv preprint arXiv:2203.06176 (2022).  
[28] Alessandro Rudi, Luigi Carratino, and Lorenzo Rosasco. "Falkon: An optimal large scale kernel method." Advances in neural information processing systems 30 (2017).  
[29] Jacob Gardner, Geoff Pleiss, Kilian Q Weinberger, David Bindel, and Andrew G Wilson. "Gpytorch: Blackbox matrix-matrix gaussian process inference withgpu acceleration." Advances in neural information processing systems 31 (2018).  
[30] Erdem Biryik, Kenneth Wang, Nima Anari, and Dorsa Sadigh. "Batch Active Learning Using Determinantal Point Processes." NeurIPS (2019).  
[31] Andreas Kirsch, Joost Van Amersfoort, and Yarin Gal. "Batchbald: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning." 2019, pp. 7026-7037.  
[32] David D Lewis and Jason Catlett. "Heterogeneous Uncertainty Sampling for Supervised Learning." Machine learning proceedings 1994. 1994, pp. 148-156.  
[33] David D Lewis and William A Gale. “A Sequential Algorithm for Training Text Classifiers.” SIGIR. 1994, pp. 3–12.  
[34] Tobias Scheffer, Christian Decomain, and Stefan Wrobel. "Active Hidden Markov Models for Information Extraction." ISIDA. 2001.  
[35] Yarin Gal, Riashat Islam, and Zoubin Ghahramani. "Deep Bayesian Active Learning with Image Data." ICML. 2017, pp. 1183-1192.  
[36] Yarin Gal and Zoubin Ghahramani. “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.” ICML. 2016, pp. 1050–1059.  
[37] Ozan Sener and Silvio Savarese. "Active Learning for Convolutional Neural Networks: A Core-set Approach." ICLR (2018).  
[38] David Arthur and Sergei Vassilvitskii. "K-means++: the Advantages of Careful Seeding." SODA. Philadelphia, PA, USA, 2007, pp. 1027-1035.  
[39] Donggeun Yoo and In So Kweon. “Learning Loss for Active Learning.” CVPR. 2019, pp. 93–102.  
[40] Toan Tran, Thanh-Toan Do, Ian Reid, and Gustavo Carneiro. "Bayesian Generative Active Deep Learning." ICML. 2019, pp. 6295-6304.  
[41] Zalán Borsos, Marco Tagliasacchi, and Andreas Krause. "Semi-supervised Batch Active Learning via Bilevel Optimization." ICASSP. IEEE. 2021.  
[42] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS. 2019, pp. 8024–8035.  
[43] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs. Version 0.2.5. 2018. URL: http://github.com/google/jax.  
[44] Roman Novak, Lechao Xiao, Jiri Hron, Jaehoon Lee, Alexander A. Alemi, Jascha Sohl-Dickstein, and Samuel S. Schoenholz. "Neural Tangents: Fast and Easy Infinite Neural Networks in Python." *ICLR*. 2020.  
[45] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image Recognition." CVPR. 2016.  
[46] Sergey Zagoruyko and Nikos Komodakis. "Wide Residual Networks." BMVC (2016).  
[47] Dan Roth and Kevin Small. "Margin-based Active Learning for Structured Output Spaces." European Conference on Machine Learning. Springer. 2006, pp. 413-424.
