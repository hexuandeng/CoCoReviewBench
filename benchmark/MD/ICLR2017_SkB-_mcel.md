# CENTRAL MOMENT DISCREPANCY (CMD) FOR DOMAIN-INVARIANT REPRESENTATION LEARNING

Werner Zellinger, Edwin Lughofer & Susanne Saminger-Platz*

Department of Knowledge-Based Mathematical Systems

Johannes Kepler University Linz, Austria

{werner.zellinger, edwin.lughofer, susanne.saminger-platz}@jku.at

Thomas Grubinger & Thomas Natschlager†

Data Analysis Systems

Software Competence Center Hagenberg, Austria

{thomas.grubinger, thomas.natschlaeger}@scch.at

# ABSTRACT

The learning of domain-invariant representations in the context of domain adaptation with neural networks is considered. In particular a new regularization method (CMD) is proposed that is based on differences of higher order central moments. CMD is used to minimize the domain discrepancy of the latent feature representations explicitly in the hidden activation space. In contrast to standard approaches, e.g. "Maximum Mean Discrepancy" (MMD), computationally expensive distance- and kernel matrix computations are unnecessary. We define CMD to be an empirical estimate of a new metric introduced in this paper. We prove that convergence of probability distributions on compact intervals w.r.t. to the new metric implies convergence in distribution of the respective random variables. We test our approach on two different benchmark data sets for object recognition (Office) and sentiment analysis of product reviews (Amazon reviews). CMD achieves state-of-the-art performance on most domain adaptation tasks of Office and outperforms networks trained with MMD, variational fair autoencoders and domain adversarial neural networks on Amazon reviews. In addition, a post-hoc parameter sensitivity analysis shows that the new approach is stable w.r.t. parameter changes in a certain interval. The source code is publicly available<sup>1</sup>.

# 1 INTRODUCTION

The collection and preprocessing of large amounts of data for new domains is often time consuming and expensive. This in turn limits the application of state-of-the-art methods like deep neural network architectures that require large amounts of data. To remedy this problem, often, data from related domains can be used to improve the prediction model in the new domain. This paper addresses the particularly important and challenging domain-invariant representation learning task of unsupervised domain adaptation (Glorot et al., 2011; Li et al., 2014; Pan et al., 2011; Ganin et al., 2016). In unsupervised domain adaptation, the training data consists of labeled data from the source domain(s) and unlabeled data from the target domain. In practice, this setting is quite common, as in many applications the collection of input data is cheap but the collection of labels is expensive. Typical examples include image analysis tasks and sentiment analysis where labels have to be collected manually.

Recent research shows that domain adaptation approaches work particularly well with (deep) neural networks, which produce outstanding results on some domain adaptation data sets (Ganin et al., 2016; Sun & Saenko, 2016; Li et al., 2016; Aljundi et al., 2015; Long & Wang, 2015; Li et al.,

2015; Zhuang et al., 2015; Louizos et al., 2016). The most successful methods have in common that they encourage similarity between the latent network representations w.r.t. the different domains. This similarity is often enforced by minimizing a certain distance between the networks domain-specific hidden activations. Three outstanding approaches for the choice of the distance function are the Proxy  $\mathcal{A}$ -distance (Ben-David et al., 2010), the Kullback-Leibler (KL) divergence Kullback & Leibler (1951), applied to the mean of the activations (Zhuang et al., 2015), and the Maximum Mean Discrepancy (Gretton et al., 2012, MMD).

Two of them, the MMD and the KL-divergence approach, can be viewed as the matching of statistical moments. The KL-divergence approach is based on mean (first raw moment) matching. Using the Taylor expansion of the Gaussian kernel, most MMD-based approaches can be viewed as minimizing a certain distance between the sums of all raw moments (Li et al., 2015).

These interpretations motivate us to match the central moments of the hidden activation distributions in a pair-wise manner emphasizing an explicit moment matching in the hidden activation space.

The contributions of this paper are as follows:

- We introduce the central moment discrepancy (CMD) in the field of domain-invariant representation learning.  
- Probability theoretic analysis is used to proof that CMD is a metric.  
- We additionally prove that convergence of probability distributions on compact intervals w.r.t. to the new metric implies convergence in distribution of the respective random variables. This means that minimizing the CMD metric between probability distributions causes the cumulative distribution functions of the random variables to converge to each other.  
- We overcome computationally expensive kernel matrix computations, as e.g. in MMD-based approaches, while emphasizing explicit moment matching in the hidden activation space.  
- We achieve state-of-the-art performance on most domain adaptation tasks of Office and outperform networks trained with MMD, variational fair autoencoders and domain adversarial neural networks on Amazon reviews.  
- A parameter sensitivity analysis shows that CMD is insensitive to parameter changes in a certain interval. Consequently, no additional hyper-parameter search has to be performed.

# 2 HIDDEN ACTIVATION MATCHING

We consider the unsupervised domain adaptation setting (Glorot et al., 2011; Li et al., 2014; Pan et al., 2011; Ganin et al., 2016) with an input space  $\mathcal{X}$  and a label space  $\mathcal{Y}$ . Two distributions over  $\mathcal{X} \times \mathcal{Y}$  are given: the labeled source domain  $D_S$  and the unlabeled target domain  $D_T$ . Two corresponding samples are given: the source sample  $S = (X_S, Y_S) = \{(x_i, y_i)\}_{i=1}^n \stackrel{\text{i.i.d.}}{\sim} (D_S)^n$  and the target sample  $T = X_T = \{x_i\}_{i=1}^m \stackrel{\text{i.i.d.}}{\sim} (D_T)^m$ . The goal of the unsupervised domain adaptation setting is to build a classifier  $f : \mathcal{X} \to \mathcal{Y}$  with a low target risk  $R_T(f) = \Pr_{(x,y) \sim D_T} (f(x) \neq y)$  while no information about the labels in  $D_T$  is given.

We focus our studies on neural network classifiers  $f_{\theta}:\mathcal{X}\to \mathcal{Y}$  with parameters  $\theta \in \Theta$ , the input space  $\mathcal{X} = \mathbb{R}^D$  with input dimension  $D$ , and, the label space  $\mathcal{V} = [0,1]^{|C|}$  with the cardinality  $|C|$  of the set of classes  $C$ . We further assume a network output  $f_{\theta}(x)\in [0,1]^{|C|}$  of an example  $x\in \mathbb{R}^D$  to be normalized by the softmax-function  $\sigma :\mathbb{R}^{|C|}\rightarrow [0,1]^{|C|}$  with  $\sigma (z)_j = \frac{e^{z_j}}{\sum_{k = 1}^{|C|}e^{z_k}}$  with  $z = \{z_1,\ldots ,z_{|C|}\}$ . We focus on bounded activation functions  $g_{H}:\mathbb{R}\to [a,b]^{N}$  for the hidden layer  $H$  with  $N$  hidden nodes, e.g. the hyperbolic tangent or the sigmoid function. Unbounded activation functions, e.g. rectified linear units or exponential linear units, can be used if batch normalization is applied. Using the loss function  $l:\Theta \times \mathcal{X}\times \mathcal{Y}\to \mathbb{R}$ , e.g. cross-entropy  $l(\theta ,x,y) = -\sum_{i\in C}y_i\log (f_\theta (x)_i)$ , and the sample set  $(X,Y)\subset \mathbb{R}^{D}\times [0,1]^{|C|}$ , we define the objective function as

$$
\min  _ {\theta \in \Theta} \mathbf {E} (l (\theta , X, Y)) \tag {1}
$$

![](images/a8f7f36ed035ce18463349ea86c655767990a1fabc167927ec69f241d3e3dfc3.jpg)  
Figure 1: Schematic sketch of a three layer neural network trained with backpropagation based on objective 2.  $\nabla_{\theta}$  refers to the gradient w.r.t.  $\theta$

![](images/9722b0563b718c3f5c22e8f1cc3a221d4604c73aa651ff16928233a3d24e1a70.jpg)  
Figure 2: Hidden activation distributions for a simple one-layer classification network with sigmoid activation functions and five hidden nodes trained with the standard objective 1 (left) and objective 2 that includes the domain discrepancy minimization (right). The approach of this paper was used as domain-regularizer. Dark gray: activations of the source domain, light gray: activations of the target domain.

![](images/16a7418b5c0c255b2157398f0020e2935c21733cd18c9b83da405120ad69ef00.jpg)

where  $\mathbf{E}$  denotes the empirical expectation, i.e.  $\mathbf{E}(l(\theta ,X,Y)) = \frac{1}{|(X,Y)|}\sum_{(x,y)\in (X,Y)}l(\theta ,x,y)$ . Let us denote the source hidden activations by  $A_H(\theta ,X_S) = g_H(\theta_H^T A_{H'}(\theta ,X_S))\subset [a,b]^N$  and the target hidden activations by  $A_{H}(\theta ,X_{T}) = g_{H}(\theta_{H}^{T}A_{H^{\prime}}(\theta ,X_{T}))\subset [a,b]^{N}$ , for the hidden layer  $H$  with  $N$  hidden nodes and parameter  $\theta_H$ , and the hidden layer  $H^{\prime}$  before  $H$ .

One fundamental assumption of most unsupervised domain adaptation networks is that the source risk  $R_{S}(f)$  is a good indicator for the target risk  $R_{T}(f)$ , when the domain-specific latent space representations are similar (Ganin et al., 2016). This similarity can be enforced by matching the distributions of the hidden activations  $A_{H}(\theta, X_{S})$  and  $A_{H}(\theta, X_{T})$  of higher layers  $H$ . Recent state-of-the-art approaches define a domain regularizer  $d: ([a, b]^{N})^{n} \times ([a, b]^{N})^{m} \to [0, \infty)$ , that gives a measure for the domain discrepancy in the activation space  $[a, b]^{N}$ . The domain regularizer is added to the objective by means of an additional weighting parameter  $\lambda$ .

$$
\min  _ {\theta \in \Theta} \mathbf {E} (l (\theta , X _ {S}, Y _ {S})) + \lambda \cdot d \left(A _ {H} (\theta , X _ {S}), A _ {H} (\theta , X _ {T})\right) \tag {2}
$$

Fig. 1 shows a sketch of the described architecture and Fig. 2 shows the hidden activations of a simple neural network optimized by Eq. 1 (left) and Eq. 2 (right). It can be seen that similar activation distributions can be obtained when being optimized on the basis of the domain regularized objective.

# 3 RELATED WORK

Recently, many suitable measures  $d$  for objective 2 have been proposed. One approach is the Proxy  $\mathcal{A}$ -distance given by  $\hat{d}_{\mathcal{A}} = 2(1 - 2\epsilon)$ , where  $\epsilon$  is the generalization error on the problem of discriminating between source and target samples (Ben-David et al., 2010). Ganin et al. (2016) compute the value  $\epsilon$  by means of a neural network classifier that is simultaneously trained with the original network by means of a gradient reversal layer. They call their approach domain-adversarial neural

networks. Unfortunately, a new classifier has to be used in this approach including the need of new parameters, additional computation times and validation procedures.

Another approach is to make use of the MMD as domain regularizer.

$$
\operatorname {M M D} (X, Y) ^ {2} = \mathbf {E} (\mathbf {E} (K (X, X))) - 2 \mathbf {E} (\mathbf {E} (K (X, Y))) + \mathbf {E} (\mathbf {E} (K (Y, Y))) \tag {3}
$$

where  $\mathbf{E}(\mathbf{E}(K(X,Y))) = \frac{1}{|X|\cdot|Y|}\sum_{k\in K(X,Y)}k$  is the empirical expectation of the kernel products  $k$  between all examples in  $X$  and  $Y$  stored by the kernel matrix  $K(X,Y)$ . A suitable choice of the kernel seems to be the Gaussian kernel  $e^{-\beta \| x - y\|^2}$  (Louizos et al., 2016; Li et al., 2015; Tzeng et al., 2014). This approach has two major drawbacks: the need of the kernel matrix computation  $K(X,Y)$ , which becomes inefficient (resource-intensive) in case of large data sets, and an additional kernel parameter  $\beta$  that has to be tuned. The tuning of  $\beta$  is sophisticated since it has to be performed in an unsupervised way (no target samples are available).

The two approaches MMD and the Proxy  $\mathcal{A}$ -distance have in common that they do not minimize the domain discrepancy explicitly in the space of the hidden activations. In contrast, the authors in Zhuang et al. (2015) do so by minimizing a modified version of the Kullback-Leibler divergence of the mean activations (MKL). That is, for samples  $X,Y\subset \mathbb{R}^N$

$$
\operatorname {M K L} (X, Y) = \sum_ {i = 1} ^ {N} \mathbf {E} (X) _ {i} \log \frac {\mathbf {E} (X) _ {i}}{\mathbf {E} (Y) _ {i}} + \mathbf {E} (Y) _ {i} \log \frac {\mathbf {E} (Y) _ {i}}{\mathbf {E} (X) _ {i}} \tag {4}
$$

with  $\mathbf{E}(X)_i$  being the  $i$ -th coordinate of the empirical expectation  $\mathbf{E}(X) = \frac{1}{|X|}\sum_{x\in X}x$ . This approach is fast to compute and has an explicit interpretation in the activation space. Empirical observations (Subsec. 5.4) show that minimizing the distance between only the first moment (mean) of the activation distributions can be improved by minimizing also a distance between higher order moments.

As noted in the introduction, our approach is motivated by the fact that the MMD and the KL-divergence approach can be seen as matching statistical moments of the hidden activations  $A_{H}(\theta ,X_{S})$  and  $A_{H}(\theta ,X_{T})$ . In particular, MMD-based approaches that use the Gaussian kernel are equivalent to minimizing a certain distance between the sums of all moments of the hidden activation distributions (Li et al., 2015).

By matching the central moments of the activations  $A_{H}(\theta, X_{S})$  and  $A_{H}(\theta, X_{T})$  in a pair-wise manner, we overcome the need of time-consuming kernel computations, overcome the parameter sensitivity issues of the spread of the Gaussian kernel and emphasize an explicit domain divergence minimization in the activation space.

# 4 CENTRAL MOMENT DISCREPANCY (CMD)

In this section we first propose a new distance function CMD on probability distributions on compact intervals. The definition is extended by two theorems that identify CMD as a metric and analyze a convergence property. The final domain regularizer is then defined as an empirical estimate of CMD. The proofs of the theorems are given in the appendix.

Definition 1 (CMD metric). Let  $X$  and  $Y$  be bounded random vectors with respective probability distributions  $p$  and  $q$ . Let  $\mathbb{E}(X^k)$  and  $\mathbb{E}(Y^k)$  for  $k \in \mathbb{N}$  be the expectations of  $X^k$  and  $Y^k$ , i.e. the  $k$ -th moments of  $X$  and  $Y$ . The central moment discrepancy metric (CMD) is defined by

$$
C M D (p, q) = \| \mathbb {E} (X) - \mathbb {E} (Y) \| _ {2} + \sum_ {k = 2} ^ {\infty} \left\| \mathbb {E} \left(\left(X - \mathbb {E} (X)\right) ^ {k}\right) - \mathbb {E} \left(\left(Y - \mathbb {E} (Y)\right) ^ {k}\right) \right\| _ {2} \tag {5}
$$

The first central moment is zero and the second to fourth central moments are called central variance, central skewness and central kurtosis. It is easy to see that  $\mathrm{CMD}(p,q)\geq 0$ ,  $\mathrm{CMD}(p,q) = \mathrm{CMD}(q,p)$ ,  $\mathrm{CMD}(p,q)\leq \mathrm{CMD}(p,r) + \mathrm{CMD}(r,q)$  and  $p = q\Rightarrow \mathrm{CMD}(p,q) = 0$ . The following theorem shows the last property for CMD to be a metric on the set of bounded random variables.

Theorem 1. Let  $p$  and  $q$  be two probability distributions on a compact interval and let  $CMD$  be defined as in (5), then

$$
C M D (p, q) = 0 \Rightarrow p = q
$$

Our approach is to minimize the discrepancy between the domain-specific hidden activation distributions by minimizing the CMD. Thus, in the optimization procedure, we increasingly expect to see the domain-specific cumulative distribution functions becoming closer and closer to each other. This characteristic can be expressed by the concept of convergence in distribution and it is shown in the following theorem.

Theorem 2. Let  $p_n$  and  $p$  be probability distributions on a compact interval and let CMD be defined as in (5), then

$$
C M D (p _ {n}, p) \rightarrow 0 \Rightarrow p _ {n} \xrightarrow {d} p
$$

where  $\xrightarrow{d}$  denotes convergence in distribution.

We define the final central moment discrepancy regularizer as an empirical estimate of the CMD metric and limit the number of central moments by a new parameter  $K$ .

Definition 2 (CMD regularizer). Let  $X$  and  $Y$  be bounded random samples independent and identically distributed from two probability distributions  $p$  and  $q$  and let  $CMD$  be the central moment discrepancy metric as defined in (5). The central moment discrepancy regularizer  $CMD_K$  is defined as an empirical estimate of  $CMD$  by

$$
C M D _ {K} (X, Y) = \| \mathbf {E} (X) - \mathbf {E} (Y) \| _ {2} + \sum_ {k = 2} ^ {K} \| c _ {k} (X) - c _ {k} (Y) \| _ {2} \tag {6}
$$

where  $\mathbf{E}(X) = \frac{1}{|X|}\sum_{x\in X}x$  is the empirical expectation computed on the sample  $X$  and  $c_{k}(X) = \mathbf{E}((x - \mathbf{E}(X))^{k}), k\geq 2$  is the sample central moment of  $X$  of order  $k$ .

We would like to underline that the training of neural networks with Eq. (2) and the CMD regularizer (6) can be easily realized by gradient descent algorithms. The gradients of the CMD regularizer are simple aggregations of derivatives of the standard functions  $g_{H}$ ,  $x^{k}$  and  $\| .\| _2$ .

# 5 EXPERIMENTS

Our experimental evaluations are based on two benchmark datasets for domain adaptation, Amazon reviews and Office, described in Subsec. Datasets. The experimental setup is discussed in Subsec. Experimental Setup and our experiments are conducted in Subsec. Experiments. The experiments aim at providing evidence regarding the following aspects: Subsec. Amazon reviews on the prediction accuracy of our approach on sentiment analysis data of product reviews, Subsec. Office on the prediction accuracy on image classification data, and, Subsec. Parameter Sensitivity on the accuracy sensitivity w.r.t. parameter changes of  $K$  for CMD and  $\beta$  for MMD.

# 5.1 DATASETS

Amazon reviews: For our first experiment we use the Amazon reviews data set with the same preprocessing as used by Chen et al. (2012); Ganin et al. (2016); Louizos et al. (2016). The data set contains product reviews of four different product categories: books,DVD disks, kitchen appliances and electronics. Reviews are encoded in 5000 dimensional feature vectors of bag-of-words unigrams and bigrams with binary labels: 0 if the product is ranked by  $1 - 3$  stars and 1 if the product is ranked by 4 or 5 stars. From the four categories we obtain twelve domain adaptation tasks (each category serves once as source category and once as target category). Using the same data splits as previous works for every task, we have 2000 labeled source examples and 2000 unlabeled target examples for training and between 3000 and 6000 examples for testing.

Office: The second experiment is based on the computer vision classification data set from Saenko et al. (2010) with images from three distinct domains: amazon (A), webcam (W) and dslr (DSLR). This data set is a de facto standard for domain adaptation algorithms in computer vision. Amazon, the largest domain, is a composition of 2817 images and its corresponding 31 classes. Following previous works we assess the performance of our method across all six possible transfer tasks (source domain  $\rightarrow$  target domain): A  $\rightarrow$  W, D  $\rightarrow$  W, W  $\rightarrow$  D, A  $\rightarrow$  D, D  $\rightarrow$  A and W  $\rightarrow$  A.

# 5.2 EXPERIMENTAL SETUP

Amazon Reviews: For the Amazon reviews experiment, we use a similar architecture as Ganin et al. (2016) with one dense hidden layer with 50 hidden nodes, sigmoid activation functions and softmax output function. Three neural networks are trained by means of Eq. 2: (a) a base model without domain regularization  $(\lambda = 0)$ , (b) the MMD as domain-regularizer  $d$  and (c) with  $\mathrm{CMD}_K$  as domain-regularizer. These models are additionally compared with the state-of-the-art models VFAE (Louizos et al., 2016) and DANN (Ganin et al., 2016). The models (a),(b) and (c) are trained with similar setup as of Louizos et al. (2016) and Ganin et al. (2016).

For the  $\mathrm{CMD}_K$  regularizer, the  $\lambda$  parameter of Eq. 2 is set to 1, i.e. the pure regularizer is used as objective. Five moments ( $K = 5$ ) are used because of its comprehensive geometrical interpretations. No additional hyper-parameter search is performed.

For the MMD regularizer we use the Gaussian kernel with parameter  $\beta$ . We performed a hyperparameter search for  $\beta$  and  $\lambda$  which has to be performed in an unsupervised way (no labels in the target domain). We use a variant of the reverse cross-validation approach proposed by Zhong et al. (2010) in which we initialize the model-weights of the reverse classifier by the weights of the first learned classifier (see Ganin et al. (2016) for details). Thereby, the parameter  $\lambda$  is tuned on 10 values between 0.1 and 500 on a logarithmic scale. The parameter  $\beta$  is tuned on 10 values between 0.01 and 10 on a logarithmic scale. Without this parameter search, no competitive prediction accuracy results could be obtained.

Since, we have to deal with sparse data, we rely on the Adagrad optimizer (Duchi et al., 2011). For all evaluations, the default parametrization is used as implemented in Keras (Chollet, 2015). All evaluations are repeated 10 times based on different shuffles of the data, and the mean accuracies and standard deviations are analyzed.

Office: Since, the office dataset is rather small with only 2817 images in its largest domain, we used the latent representations of the convolution neural network VGG16 of Simonyan & Zisserman (2014). In particular we trained a classifier with one hidden layer, 256 hidden nodes and sigmoid activation function on top of the output of the first dense layer in the network. We again trained one base model without domain regularization and a  $\mathrm{CMD}_K$  regularized version with  $K = 5$  and  $\lambda = 1$ .

We follow the standard training protocol for this data set and use all available source and target examples during training. Using this "fully-transductive" protocol, we compare our method with other state-of-the-art approaches including DLID (Chopra et al., 2013), DDC (Tzeng et al., 2014), DAN (Long & Wang, 2015), Deep CORAL (Sun & Saenko, 2016) and DANN (Ganin et al., 2016) based on fine-tuning of the baseline model AlexNet (Krizhevsky et al., 2012) and we further compare our method to LSSA (Aljundi et al., 2015), CORAL (Sun et al., 2016) and AdaBN (Li et al., 2016) based on the fine-tuning of InceptionBN (Ioffe & Szegedy, 2015).

As an alternative to Adagrad for non-sparse data, we use the Adadelta optimizer from Zeiler (2012). Again the default parametrization from Keras is used. We handle unbalances between source and target sample by randomly down-sampling (up-sampling) of the source sample. In addition, we ensure a sub-sampled source batch that is balanced w.r.t. the class labels.

Since all hyper-parameters are set such that they have simple geometric interpretations, no hyperparameter search has to be performed.

All experiments are repeated 10 times with randomly shuffled data sets and random initializations.

# 5.3 RESULTS

Amazon Reviews: Table 1 shows the classification accuracies of four models: The Source Only model is the shallow neural network trained with objective 1 and it serves as a base model for the domain adaptation improvements. The models MMD and CMD are trained with the same architecture except using objective 2 with  $d$  as the domain-regularizer MMD and  $\mathrm{CMD}_5$  respectively. VFAE is the variational fair auto encoder of Louizos et al. (2016), including a slightly modified version of the MMD regularizer for faster computations, and DANN is the domain-adversarial neural networks model of Ganin et al. (2016). The last two columns are taken directly from their publications.

As one can observe in Table 1, our accuracy of CMD is the highest in 9 out of 12 domain adaptation tasks, whereas on the remaining 3 it is the second best method. However, the difference in accuracy compared to the best method is smaller than the standard deviation over all data shuffles.

Table 1: Prediction accuracy ± standard deviation on the Amazon reviews dataset. The last two columns are taken directly from Louizos et al. (2016) and Ganin et al. (2016).  

<table><tr><td>Source→Target</td><td>Source Only</td><td>MMD</td><td>CMD</td><td>VFAE</td><td>DANN</td></tr><tr><td>books→dvd</td><td>.787 ± .004</td><td>.796 ± .008</td><td>.805 ± .007</td><td>.799</td><td>.784</td></tr><tr><td>books→electronics</td><td>.714 ± .009</td><td>.758 ± .018</td><td>.787 ± .007</td><td>.792</td><td>.733</td></tr><tr><td>books→kitchen</td><td>.745 ± .006</td><td>.787 ± .019</td><td>.813 ± .008</td><td>.816</td><td>.779</td></tr><tr><td>dvd→books</td><td>.746 ± .019</td><td>.780 ± .018</td><td>.795 ± .005</td><td>.755</td><td>.723</td></tr><tr><td>dvd→electronics</td><td>.724 ± .011</td><td>.766 ± .025</td><td>.797 ± .010</td><td>.786</td><td>.754</td></tr><tr><td>dvd→kitchen</td><td>.765 ± .012</td><td>.796 ± .019</td><td>.830 ± .012</td><td>.822</td><td>.783</td></tr><tr><td>electronics→books</td><td>.711 ± .006</td><td>.733 ± .017</td><td>.744 ± .008</td><td>.727</td><td>.713</td></tr><tr><td>electronics→dvd</td><td>.719 ± .009</td><td>.748 ± .013</td><td>.763 ± .006</td><td>.765</td><td>.738</td></tr><tr><td>electronics→kitchen</td><td>.844 ± .005</td><td>.857 ± .007</td><td>.860 ± .004</td><td>.850</td><td>.854</td></tr><tr><td>kitchen→books</td><td>.699 ± .014</td><td>.740 ± .017</td><td>.756 ± .006</td><td>.720</td><td>.709</td></tr><tr><td>kitchen→dvd</td><td>.734 ± .011</td><td>.763 ± .011</td><td>.775 ± .005</td><td>.733</td><td>.740</td></tr><tr><td>kitchen→electronics</td><td>.833 ± .004</td><td>.844 ± .007</td><td>.854 ± .003</td><td>.838</td><td>.843</td></tr><tr><td>average</td><td>.752 ± .009</td><td>.781 ± .015</td><td>.798 ± .007</td><td>.784</td><td>.763</td></tr></table>

# Office:

Table 2 shows the classification accuracy of different models trained on the Office dataset. Note that some of the methods (LSSA, CORAL and AdaBN) are evaluated based on the InceptionBN model, which shows higher accuracy than the base model (VGG16) of our method in most tasks. However, our method outperforms related state-of-the-art methods on all except two tasks, on which it performs similar. We improve the previous state-of-the-art method AdaBN (Li et al., 2016) by more than  $3.2\%$  in average accuracy.

Fig. 3 shows the  $t$ -SNE embeddings (Maaten & Hinton, 2008) of the hidden activations of a Source Only model and a CMD-based model trained on the A→W task. It can be seen that the activations of the target domain (stars) in the left image (Source Only model) are mixed especially in the middle region. Class-specific clusters of the target domain are better noticeable in the right image (CMD version). For example consider the "mouse"-class (black). The activations of this class lie in the mixed middle region for the Source Only model. The accuracy of this model is  $57\%$  for the "mouse"-class in this experiment. For the domain-regularized version, the "mouse"-activations are easier separable to the activations of the other classes. The CMD-based model achieves  $100\%$  accuracy in this experiment for the "mouse"-class. The "mouse" performance gain is induced by the central moment discrepancy of the amazon and webcam hidden activations since the CMD regularizer is the only difference of the two models.

Table 2: Prediction accuracy ± standard deviation on the Office dataset. The first 10 rows are taken directly from the papers of Ganin et al. (2016) and Li et al. (2016). The models DLID - DANN are based on the AlexNet model, LSSA - AdaBN are based on the InceptionBN model and our method (CMD) is based on the VGG16 model.  

<table><tr><td>Method</td><td>A→W</td><td>D→W</td><td>W→D</td><td>A→D</td><td>D→A</td><td>W→A</td><td>average</td></tr><tr><td>AlexNet</td><td>.616</td><td>.954</td><td>.990</td><td>.638</td><td>.511</td><td>.498</td><td>.701</td></tr><tr><td>DLID</td><td>.519</td><td>.782</td><td>.899</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DDC</td><td>.618</td><td>.950</td><td>.985</td><td>.644</td><td>.521</td><td>.522</td><td>.707</td></tr><tr><td>Deep CORAL</td><td>.664</td><td>.957</td><td>.992</td><td>.668</td><td>.528</td><td>.515</td><td>.721</td></tr><tr><td>DAN</td><td>.685</td><td>.960</td><td>.990</td><td>.670</td><td>.540</td><td>.531</td><td>.729</td></tr><tr><td>DANN</td><td>.730</td><td>.964</td><td>.992</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>InceptionBN</td><td>.703</td><td>.943</td><td>1.00</td><td>.705</td><td>.601</td><td>.579</td><td>.755</td></tr><tr><td>LSSA</td><td>.677</td><td>.961</td><td>.984</td><td>.713</td><td>.578</td><td>.578</td><td>.749</td></tr><tr><td>CORAL</td><td>.709</td><td>.957</td><td>.998</td><td>.719</td><td>.590</td><td>.602</td><td>.763</td></tr><tr><td>AdaBN</td><td>.742</td><td>.957</td><td>.998</td><td>.731</td><td>.598</td><td>.574</td><td>.767</td></tr><tr><td>VGG16</td><td>.676 ± .006</td><td>.961 ± .003</td><td>.992 ± .002</td><td>.739 ± .009</td><td>.582 ± .005</td><td>.578 ± .004</td><td>.755</td></tr><tr><td>CMD</td><td>.770 ± .006</td><td>.963 ± .004</td><td>.992 ± .002</td><td>.796 ± .006</td><td>.638 ± .007</td><td>.633 ± .006</td><td>.799</td></tr></table>

![](images/038e93c7a2cf1623b5cc9d72960733badb9dbe66415b04bb9f32d7fbcc815db7.jpg)  
Figure 3: t-SNE embedding (Maaten & Hinton, 2008) of hidden activations of two networks trained on the office dataset  $(\mathrm{A}\rightarrow \mathrm{W})$ . The networks are trained without domain regularization (left) and with the CMD regularizer (right). The circles show the embeddings of the activations of the source task and the stars correspond to the activations of the target task. Each color identifies a different class. The class visualized by the black opaque 'o's and  $^+$  s, for source and target respectively, correspond to the "mouse"-class. The network trained with objective 1 achieves an accuracy of 0.57 on the "mouse"-class where the training with objective 2 yields 1.0 classification accuracy for this class.

![](images/721dc8161f1d4a0a87acb69efa8113cba244f8d2034ccaeae2c3ac8216ffe98f.jpg)

# 5.4 PARAMETER SENSITIVITY

The sensitivity experiment aims at providing evidence regarding the accuracy sensitivity of the  $\mathrm{CMD}_K$  regularizer w.r.t. parameter changes of  $K$ . The claim is that the accuracy of CMD-based networks does not depend strongly on the choice of  $K$  in a range around its default value 5. Note that  $K = 5$  was chosen a priori as the first five moments have clear geometrical interpretations (see Subsec. Experimental Setup).

We analyze the classification accuracy of a CMD-based network trained on all tasks of the Amazon reviews experiment. We performed a grid-search for the two regularization hyper-parameters  $\lambda$  and  $K$ . We chose empirically a representative stable region for each parameter, [0.3, 3] for  $\lambda$  and  $\{1,\dots ,7\}$  for  $K$ . Since we want to analyze the sensitivity w.r.t.  $K$ , we averaged over the  $\lambda$ -dimension resulting in one accuracy value per  $K$  for each of the 12 tasks. Each accuracy is transformed in a accuracy ratio value by dividing it with the accuracy of  $K = 5$ . Thus, for each  $K$  and task we get one value representing the ratio between the obtained accuracy (for this  $K$  and task) and the accuracy of  $K = 5$ .

The same procedure was performed with MMD regularization with  $\lambda \in [5,45]$  and the Gaussian kernel parameter  $\beta \in [0.3,1.7]$ . We calculated the accuracy ratio to the accuracy of  $\beta = 1.2$ , since it shows the highest mean accuracy w.r.t. all tasks.

Fig. 4 shows the accuracy ratio values for each task and parameter value for CMD and MMD. The left figure (CMD) shows that the accuracy ratios between  $K = 5$  and  $K \in \{3,4,6,7\}$  are lower than  $0.5\%$  which underpins the claim.

For  $K = 1$  and  $K = 2$  higher ratio values are obtained. In addition, for these two values, many tasks show worse accuracy than obtained by  $K \in \{3,4,5,6,7\}$ . From this we additionally conclude that higher values of  $K$  are preferable to  $K = 1$  and  $K = 2$  for the Amazon reviews tasks.

The right figure shows the results for the MMD regularizer in comparison. It can be clearly recognized that the accuracy of the MMD network is more sensitive to parameter changes than the CMD regularized version w.r.t. the intervals under consideration. This underpins the assumption that the CMD-based version is less sensitive to parameter changes than the MMD version.

# 6 CONCLUSION AND OUTLOOK

In this paper we presented CMD, a new, mathematically well founded method for learning domain-invariant representations. Similar to other state-of-the-art approaches (MMD, KL-distance, Proxy

![](images/e9e96d2bb9cdec113e4078364be4eb304bd2999e58c37ee8f1d3e31a1e4db693.jpg)  
Figure 4: Sensitivity of classification accuracy of CMD (left) and MMD (right) on the Amazon reviews dataset. Each line represent accuracy ratio values for one task. The ratio values are obtained by averaging over the ranges [0.3, 3] and [5, 45] for the weighting parameter  $\lambda$  for CMD and MMD, respectively. The resulting values are divided by the accuracy of  $K = 5$  for the CMD and  $\beta = 1.2$  for the MMD.  $\beta = 1.2$  was chosen as it shows the largest mean accuracy over all tasks. The mean accuracy over all tasks is shown by the thick black dashed line.

![](images/320ce3ba0af453c276178366955b30b0e29242d2318150e2378ed2f6cf15e6fc.jpg)

$\mathcal{A}$ -distance) our method minimizes the domain discrepancy of the latent feature representations. By using probability theoretic analysis we proved that CMD is a metric and that convergence in CMD implies convergence in distribution for probability distributions on compact intervals. Our approach is computationally more efficient than MMD, as no expensive kernel matrix computations are required. Our approach only depends on the number of moments  $K$ . Empirical results demonstrate that our methods yields state-of-the-art performance with its default value  $K = 5$  and that the classification accuracy is not sensitive on the particular choice of  $K$  for  $K \geq 3$ . This is an additional advantage to other approaches, as no computationally expensive hyper-parameter selection is required. The experiments further identify CMD to be the new state-of-the-art for most tasks in the Office benchmark data set and to be preferable to MMD, VFAE and DANN in most tasks of the Amazon reviews benchmark data set.

In this paper we evaluated our widely applicable domain-invariant representation learning method on unsupervised domain adaptation tasks for classification. In the future we want to extend our evaluations to other domain-invariant learning tasks, e.g. supervised domain adaptation, domain generalization and regression problems.

# A THEOREM PROOFS

Theorem 1. Let  $p$  and  $q$  be two probability distributions on a compact interval and let CMD be defined as in (5), then

$$
C M D (p, q) = 0 \Rightarrow p = q
$$

Proof. Let  $X$  and  $Y$  be two random vectors that have probability distributions  $p$  and  $q$  respectively. Let  $\hat{X} = X - \mathbb{E}(X)$  and  $\hat{Y} = Y - \mathbb{E}(Y)$  be the mean centered random variables. From  $\mathrm{CMD}(p, q) = 0$  it follows that  $\mathbb{E}(\hat{X}^k) = \mathbb{E}(\hat{Y}^k), \forall k \in \mathbb{N}$ , i.e., all moments of the bounded random variables  $\hat{X}$  and  $\hat{Y}$  are equal. Therefore, the moment generating functions of  $\hat{X}$  and  $\hat{Y}$  are equal and it follows from theorem 10.4 of Grinstead & Snell (2012) that  $\hat{X} = \hat{Y}$ . Since  $\mathbb{E}(X) = \mathbb{E}(Y)$ , it follows that  $X = Y$ .

Theorem 2. Let  $p_n$  and  $p$  be probability distributions on a compact interval and let CMD be defined as in (5), then

$$
C M D (p _ {n}, p) \rightarrow 0 \Rightarrow p _ {n} \xrightarrow {d} p
$$

where  $\xrightarrow{d}$  denotes convergence in distribution.

Proof. Let  $X_{n}$  and  $X$  be random vectors that have probability distributions  $p_{n}$  and  $p$  respectively. Let  $\hat{X} = X - \mathbb{E}(X)$  and  $\hat{X}_{n} = X_{n} - \mathbb{E}(X_{n})$  be the mean centered random variables.

From  $\mathrm{CMD}(X_n,X)\to 0$  it follows that  $\mathbb{E}(\hat{X}_n^k)\to \mathbb{E}(\hat{X}^k),\forall k\in \mathbb{N}$  . Applying theorem 30.2

from Billingsley (2008) yields  $\hat{X}_n \stackrel{d}{\to} \hat{X}$ . With  $\mathbb{E}(X_n) \stackrel{d}{\to} \mathbb{E}(X)$ , we obtain  $X_n \stackrel{d}{\to} X$ .

# ACKNOWLEDGMENTS

The research reported in this paper has been supported by the Austrian Ministry for Transport, Innovation and Technology, the Federal Ministry of Science, Research and Economy, and the Province of Upper Austria in the frame of the COMET center SCCH.

# REFERENCES

Rahaf Aljundi, Rémi Emonet, Damien Muselet, and Marc Sebban. Landmarks-based kernelized subspace alignment for unsupervised domain adaptation. In International Conference on Computer Vision and Pattern Recognition, pp. 56-63, 2015.  
Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from different domains. Machine learning, 79(1-2):151-175, 2010.  
Patrick Billingsley. Probability and measure. John Wiley & Sons, 2008.  
Minmin Chen, Zhixiang Xu, Kilian Weinberger, and Fei Sha. Marginalized denoising autoencoders for domain adaptation. International Conference on Machine Learning, pp. 767-774, 2012.  
François Chollet. Keras: Deep learning library for theano and tensorflow, 2015.  
Sumit Chopra, Suhrid Balakrishnan, and Raghuraman Gopalan. Dlid: Deep learning for domain adaptation by interpolating between domains. International Conference on Machine Learning Workshop on Challenges in Representation Learning, 2013.  
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul):2121-2159, 2011.  
Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. Journal of Machine Learning Research, 17(Jan):1-35, 2016.  
Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Domain adaptation for large-scale sentiment classification: A deep learning approach. In International Conference on Machine Learning, pp. 513-520, 2011.  
Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 13(Mar):723-773, 2012.  
Charles Miller Grinstead and James Laurie Snell. Introduction to probability. American Mathematical Soc., 2012.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pp. 448-456, 2015.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Solomon Kullback and Richard A Leibler. On information and sufficiency. Annals of Mathematical Statistics, 22(1):79-86, 1951.  
Yanghao Li, Naiyan Wang, Jianping Shi, Jiaying Liu, and Xiaodi Hou. Revisiting batch normalization for practical domain adaptation. arXiv preprint arXiv:1603.04779, 2016.  
Yujia Li, Kevin Swersky, and Richard Zemel. Unsupervised domain adaptation by domain invariant projection. In Neural Information Processing Systems Workshop on Transfer and Multitask Learning, 2014.

Yujia Li, Kevin Swersky, and Richard Zemel. Generative moment matching networks. In International Conference on Machine Learning, pp. 1718-1727, 2015.  
Mingsheng Long and Jianmin Wang. Learning transferable features with deep adaptation networks. International Conference on Machine Learning, 1:97-105, 2015.  
Christos Louizos, Kevin Swersky, Yujia Li, Max Welling, and Richard Zemel. The variational fair auto encoder. International Conference on Learning Representations, 2016.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(Nov):2579-2605, 2008.  
Sinno Jialin Pan, Ivor W Tsang, James T Kwok, and Qiang Yang. Domain adaptation via transfer component analysis. IEEE Transactions on Neural Networks, 22(2):199-210, 2011.  
Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell. Adapting visual category models to new domains. In European Conference on Computer Vision, pp. 213-226. Springer, 2010.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. International Conference on Learning Representations, 2014.  
Baochen Sun and Kate Saenko. Deep coral: Correlation alignment for deep domain adaptation. arXiv preprint arXiv:1607.01719, 2016.  
Baochen Sun, Jiashi Feng, and Kate Saenko. Return of frustratingly easy domain adaptation. In Thirteen AAAI Conference on Artificial Intelligence, 2016.  
Eric Tzeng, Judy Hoffman, Ning Zhang, Kate Saenko, and Trevor Darrell. Deep domain confusion: Maximizing for domain invariance. arXiv preprint arXiv:1412.3474, 2014.  
Matthew D Zeiler. Adadelta: an adaptive learning rate method. arXiv preprint arXiv:1212.5701, 2012.  
Erheng Zhong, Wei Fan, Qiang Yang, Olivier Verscheure, and Jiangtao Ren. Cross validation framework to choose amongst models and datasets for transfer learning. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 547-562. Springer, 2010.  
Fuzhen Zhuang, Xiaohu Cheng, Ping Luo, Sinno Jialin Pan, and Qing He. Supervised representation learning: Transfer learning with deep autoencoders. In International Joint Conference on Artificial Intelligence, 2015.