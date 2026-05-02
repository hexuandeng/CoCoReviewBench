# LEARNING DEEP REPRESENTATIONS BY MUTUAL INFORMATION ESTIMATION AND MAXIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

In this work, we perform unsupervised learning of representations by maximizing mutual information between an input and the output of a deep neural network encoder. Importantly, we show that structure matters: incorporating knowledge about locality of the input to the objective can greatly influence a representation's suitability for downstream tasks. We further control characteristics of the representation by matching to a prior distribution adversarily. Our method, which we call Deep InfoMax (DIM), outperforms a number of popular unsupervised learning methods and competes with fully-supervised learning on several classification tasks. DIM opens new avenues for unsupervised learning of representations and is an important step towards flexible formulations of representation-learning objectives for specific end-goals.

# 1 INTRODUCTION

One core objective of deep learning is to discover "good" representations, and the simple idea explored here is to train a representation-learning function (i.e., an encoder) to maximize the mutual information (MI) between its inputs and outputs. MI is notoriously difficult to compute, particularly in continuous and high-dimensional settings. Fortunately, recent advances enable effective computation of MI between high dimensional input/output pairs of deep neural networks (Belghazi et al., 2018). We leverage MI estimation for representation learning, but we will also show that, depending on the downstream task, maximizing MI between the complete input and the encoder output (i.e., global MI) is often not sufficient for learning useful representations. Rather, structure matters: maximizing the average MI between the representation and local regions of the input can greatly improve the representation's quality for, e.g., classification tasks, while global MI plays a stronger role in the ability to reconstruct the full input given the representation.

Usefulness of a representation is not just a matter of information content: representational characteristics like independence also play an important role (Gretton et al., 2012; Hyvarinen & Oja, 2000; Hinton, 2002; Schmidhuber, 1992; Bengio et al., 2013; Thomas et al., 2017). We therefore combine MI maximization with prior matching in a manner similar to adversarial autoencoders (AAE, Makhzani et al., 2015) to constrain representations according to desired statistical properties. This approach is closely related to the infomax optimization principle (Linsker, 1988; Bell & Sejnowski, 1995), so we call our method Deep InfoMax (DIM). Our main contributions are the following:

- We formalize Deep InfoMax (DIM), which simultaneously estimates and maximizes the mutual information between input data and learned high-level representations.  
- Our mutual information maximization procedure can prioritize global or locally-consistent information, which we show can be used to tune the suitability of learned representations for classification or reconstruction-style tasks.  
- We use adversarial learning (à la Makhzani et al., 2015) to constrain the representation to have desired statistical characteristics specific to a prior.  
- We introduce two new measures of representation quality, one based on Mutual Information Neural Estimation (MINE, Belghazi et al., 2018) and a neural dependency measure (NDM) based on the work by Brakel & Bengio (2017), and we use these to bolster our comparison of DIM to different unsupervised methods.

# 2 RELATED WORK

There are many popular methods for learning representations. Some of the older methods, such as independent component analysis (ICA, Bell & Sejnowski, 1995) and self-organizing maps (Kohonen, 1998) generally cannot represent complex relationships in data. Other approaches include deep volume-preserving maps (Dinh et al., 2014; 2016), deep clustering (Xie et al., 2016; Chang et al., 2017), noise as targets (NAT, Bojanowski & Joulin, 2017), self-supervised methods (Doersch & Zisserman, 2017), and co-learning (Dosovitskiy et al., 2016; Sajjadi et al., 2016).

Generative models are also commonly used as methods for building representations (Vincent et al., 2010; Kingma et al., 2014; Salimans et al., 2016; Rezende et al., 2016; Donahue et al., 2016), and mutual information (MI) plays an important role in their representational quality. In generative models that rely on reconstruction (e.g., denoising, variational, and adversarial autoencoders, Vincent et al., 2008; Rifai et al., 2012; Kingma & Welling, 2013; Makhzani et al., 2015), the negative reconstruction error can be related to the MI in the encoder as,

$$
\mathcal {I} _ {e} (X, Y) \geq \mathcal {H} _ {e} (Y) - \mathcal {R} _ {e, d} (X), \tag {1}
$$

where  $X$  and  $Y$  are random variables corresponding to the input and an intermediate representation (e.g., the bottleneck),  $\mathcal{R}_{e,d}(X)$  is the reconstruction error of  $X$ , and  $\mathcal{H}_e(Y)$  is the marginal entropy of the encoder output. Thus, models with reconstruction-type objectives provide some guarantees on the amount of information encoded in their intermediate representations. Similar guarantees exist for bi-directional adversarial models (Dumoulin et al., 2016; Donahue et al., 2016), which adversarily train an encoder / decoder to match their respective joint distributions (which typically increases the marginal entropy in Equation 1) or to minimize the reconstruction error (Chen et al., 2016).

Mutual-information estimation Mutual information-based objectives have a long tradition in unsupervised learning of features. The infomax optimization principle (Linsker, 1988; Bell & Sejnowski, 1995), as prescribed for neural networks, advocates maximizing MI between the input and output. This is the basis of numerous ICA algorithms, some of which are nonlinear (Hyvärinen & Pajunen, 1999; Almeida, 2003), but none of which are general enough to apply to deep networks. Mutual information neural estimation (MINE, Belghazi et al., 2018) learns a neural estimate of the MI of continuous variables, is strongly consistent, and can be used to learn better implicit bi-directional generative models. Deep InfoMax (DIM) follows MINE in this regard, though we find that the generator is unnecessary. In addition, it is not necessary to use the exact KL-based formulation of MI: for example, a simple classifier based on the Jensen-Shannon divergence (JSD) is both more stable and provides better results. This is good as the KL-based (continuous) MI is unbounded, while the JSD is bounded. DIM generally admits a variety of MI estimators (which we will show), but more importantly: our method can also leverage local structure in the input, which can be used to improve suitability of representations for classification.

Leveraging known structure in the input in MI maximization-based objectives is nothing new (Becker, 1992; 1996; Wiskott & Sejnowski, 2002), and some very recent works also follow this intuition. Unsupervised clustering and segmentation is attainable by maximizing the MI between images associated by transforms or spatial proximity (Ji et al., 2018). Mostly independent of our work, contrastive predictive coding (CPC, Oord et al., 2018) uses a MI estimate-based approach to perform prediction on held-out patches (e.g., in the context of image). DIM, in contrast, uses MI w.r.t. a global summary vector (i.e., a single representation for the complete input), and requires only a scoring function for (local patch, global vector) pairs. Our work further looks at the suitability of representations across two different MI maximization objectives (local vs. global), a flexibility we believe is necessary for training representations intended for different end-goals.

# 3 DEEP INFOMAX

Here we outline the general setting of training an encoder to maximize mutual information between its input and output. Let  $\mathcal{X}$  and  $\mathcal{Y}$  be the domain and range of a continuous and (almost everywhere) differentiable parametric function,  $E_{\psi}:\mathcal{X}\to \mathcal{Y}$  with parameters  $\psi$  (e.g., a neural network). These parameters define a family of encoders,  $\mathcal{E}_{\Phi} = \{E_{\psi}\}_{\psi \in \Psi}$  over  $\Psi$ . Assume that we are given a set of training examples on an input space,  $\mathcal{X}$ :  $\mathbf{X} := \{x^{(i)}\in \mathcal{X}\}_{i = 1}^{N}$ , with empirical probability distribution  $\mathbb{P}$ . We define  $\mathbb{U}_{\psi ,\mathbb{P}}$  to be the marginal distribution induced by pushing samples from  $\mathbb{P}$  through  $E_{\psi}$ .

![](images/6626e874b254d1c6bab60d8919c8758104a26ea4c2df9ac177873a6cea64fbac.jpg)  
Figure 1: The base encoder model in the context of image data. An image (in this case) is encoded using a convnet until reaching a feature map of  $M \times M$  feature vectors corresponding to  $M \times M$  input patches. These vectors are summarized into a single feature vector,  $Y$ . Our goal is to train this network such that useful information about the input is easily extracted from the high-level features.

![](images/34469c54657082edc505bf8241d22c9145d3984eb84a8ba42353ccb5b5c23832.jpg)  
Figure 2: Deep InfoMax (DIM) with a global  $\mathbf{MI}(X;Y)$  objective. Here, we pass both the high-level feature vector,  $Y$ , and the lower-level  $M\times M$  feature map (see Figure 1) through a discriminator to get the score. Fake samples are drawn by combining the same feature vector with a  $M\times M$  feature map from another image.

I.e.,  $\mathbb{U}_{\psi, \mathbb{P}}$  is the distribution over encodings  $y \in \mathcal{V}$  produced by sampling observations  $x \sim \mathcal{X}$  and then sampling  $y \sim E_{\psi}(x)$ .

An example encoder for image data is given in Figure 1, which will be used in the following sections, but this approach can easily be adapted for temporal data. Similar to the infomax optimization principle (Linsker, 1988), we assert our encoder should be trained according to the following criteria:

- Mutual information maximization: Find the set of parameters,  $\psi$ , such that the mutual information,  $\mathcal{I}(X;E_{\psi}(X))$ , is maximized. Depending on the end-goal, this maximization can be done over the complete input,  $X$ , or some structured or "local" subset.  
- Statistical constraints: Depending on the end-goal for the representation, the marginal  $\mathbb{U}_{\psi, \mathbb{P}}$  should match a prior distribution,  $\mathbb{V}$ . Roughly speaking, this can be used to encourage the output of the encoder to have desired characteristics (e.g., independence).

The formulation of these two objectives covered below we call Deep InfoMax (DIM).

![](images/3a3611499224f0a9be67171b4f1a79fc050f07d46e9c05f58729a076bbe9c3e3.jpg)  
Figure 3: Maximizing mutual information between local features and global features. First we encode the image to a feature map that reflects some structural aspect of the data, e.g. spatial locality, and we further summarize this feature map into a global feature vector (see Figure 1). We then concatenate this feature vector with the lower-level feature map at every location. A score is produced for each local-global pair through an additional function (see the Appendix A.1 for details).

# 3.1 MUTUAL INFORMATION ESTIMATION AND MAXIMIZATION

Our basic mutual information maximization framework is presented in Figure 2. The approach follows Mutual Information Neural Estimation (MINE, Belghazi et al., 2018), which estimates mutual information by training a classifier to distinguish between samples coming from the joint,  $\mathbb{J}$ , and the product of marginals,  $\mathbb{M}$ , of random variables  $X$  and  $Y$ . MINE uses a lower-bound to the MI based

on the Donsker-Varadhan representation (DV, Donsker & Varadhan, 1983) of the KL-divergence,

$$
\mathcal {I} (X; Y) := \mathcal {D} _ {K L} (\mathbb {J} | | \mathbb {M}) \geq \widehat {\mathcal {I}} _ {\omega} ^ {(D V)} (X; Y) := \mathbb {E} _ {\mathbb {J}} [ T _ {\omega} (x, y) ] - \log \mathbb {E} _ {\mathbb {M}} \left[ e ^ {T _ {\omega} (x, y)} \right], \tag {2}
$$

where  $T_{\omega}:\mathcal{X}\times \mathcal{Y}\to \mathbb{R}$  is a discriminator function modeled by a neural network with parameters  $\omega$

At a high level, we optimize  $E_{\psi}$  by simultaneously estimating and maximizing  $\mathcal{I}(X,E_{\psi}(X))$

$$
(\hat {\omega}, \hat {\psi}) _ {G} = \underset {\omega , \psi} {\arg \max } \widehat {\mathcal {I}} _ {\omega} (X; E _ {\psi} (X)), \tag {3}
$$

where the subscript  $G$  denotes "global" for reasons that will be clear later. However, there are some important differences that distinguish our approach from MINE. First, because the encoder and mutual information estimator are optimizing the same objective and require similar computations, we share layers between these functions, so that  $E_{\psi} = f_{\psi}\circ C_{\psi}$  and  $T_{\psi ,\omega} = D_{\omega}\circ g\circ (C_{\psi},E_{\psi})^{1}$ , where  $g$  is a function that combines the encoder output with the lower layer.

Second, as we are primarily interested in maximizing MI, and not concerned with its precise value, we can rely on non-KL divergences which may offer favourable trade-offs. For example, one could define a Jensen-Shannon MI estimator (following the formulation of Nowozin et al., 2016),

$$
\widehat {\mathcal {L}} _ {\omega , \psi} ^ {(\mathrm {J S D})} (X; E _ {\psi} (X)) := \mathbb {E} _ {\mathbb {P}} \left[ - \operatorname {s p} \left(- T _ {\psi , \omega} \left(x, E _ {\psi} (x)\right)\right) \right] - \mathbb {E} _ {\mathbb {P} \times \tilde {\mathbb {P}}} [ \operatorname {s p} \left(T _ {\psi , \omega} \left(x ^ {\prime}, E _ {\psi} (x)\right)\right) ], \tag {4}
$$

where  $x$  is an input sample,  $x'$  is an input sampled from  $\tilde{\mathbb{P}} = \mathbb{P}$ , and  $\mathrm{sp}(z) = \log(1 + e^z)$  is the softplus function. This estimator amounts to the familiar binary cross-entropy, which is well-understood in terms of neural network optimization and which we find works better in practice (e.g., more stable) than the DV-based objective. Intuitively, the Jensen-Shannon-based estimator should behave similarly as the DV-based estimator in Eq. 2, since both act like classifiers whose objectives maximize the expected log-ratio of the joint over the product of marginals.

Noise-Contrastive Estimation (NCE, Gutmann & Hyvarinen, 2010; 2012) can be used to formulate a bound on MI (Oord et al., 2018), which can be used in the context of DIM by maximizing:

$$
\widehat {\mathcal {I}} _ {\omega , \psi} ^ {\left(\mathrm {N C E}\right)} (X; E _ {\psi} (X)) := \mathbb {E} _ {\mathbb {P}} \left[ T _ {\psi , \omega} \left(x, E _ {\psi} (x)\right) - \mathbb {E} _ {\mathbb {P}} \left[ \log \sum_ {x ^ {\prime}} e ^ {T _ {\psi , \omega} \left(x ^ {\prime}, E _ {\psi} (x)\right)} \right] \right]. \tag {5}
$$

For DIM, a key difference between the DV, JSD, and NCE-based formulations is whether an expectation over  $\mathbb{P} / \tilde{\mathbb{P}}$  appears inside or outside of a log. In fact, the JSD-based objective mirrors the original NCE formulation in Gutmann & Hyvarinen (2010), which phrased unnormalized density estimation as binary classification between the data distribution and a noise distribution. DIM sets the noise distribution to the product of marginals over  $X / Y$ , and the data distribution to the true joint. The NCE formulation in Eq. 5 follows NCE as used in the language modeling community (Mnih & Kavukcuoglu, 2013), which replaces binary classification with a related set of multiclass problems. In practice, implementations of these estimators appear quite similar and can reuse most of the same code. We investigate JSD and NCE in our experiments, and find that using NCE often outperforms JSD on downstream tasks, though this effect diminishes with more challenging data. However, as we show in the App. (A.2), NCE requires a large number of negative samples (samples from  $\tilde{\mathbb{P}}$ ) to be competitive. We generate negative samples using all combinations of global and local features at all locations of the relevant feature map, across all images in a batch. For a batch of size  $B$ , that gives  $O(B \times M^2)$  negative samples per positive example, which quickly becomes cumbersome with increasing batch size. We found that DIM with the JSD loss is relatively insensitive to the number of negative samples, and in fact outperforms NCE as the number of negative samples becomes smaller.

# 3.2 LOCAL MUTUAL INFORMATION MAXIMIZATION

The objective in Eq. 3 can be used to maximize MI between input and output, but ultimately this may be undesirable depending on the task. For example, trivial pixel-level noise is useless for image classification, so a representation may not benefit from encoding this information (e.g., in zero-shot learning, transfer learning, etc.). In order to obtain a representation more suitable for classification, we can instead maximize the average MI between the high-level representation and local patches of

the image. Because the same representation is encouraged to have high MI with all the patches, this favours encoding aspects of the data that are shared across patches.

Suppose the feature vector is of limited capacity (number of units and range) and assume the encoder does not support infinite output configurations. For maximizing the MI between the whole input and the representation, the encoder can pick and choose what type of information in the input is passed through the encoder, such as noise specific to local patches or pixels. However, if the encoder passes information specific to only some parts of the input, this does not increase the MI with any of the other patches that do not contain said noise. This encourages the encoder to prefer information that is shared across the input, and this hypothesis is supported in our experiments below.

Our local DIM framework is presented in Figure 3. First we encode the input to a feature map,  $C_{\psi}(x) \coloneqq \{C_{\psi}^{(i)}\}_{i=1}^{M \times M}$  that reflects useful structure in the data (e.g., spatial locality), indexed in this case by  $i$ . Next, we summarize this local feature map into a global feature,  $E_{\psi}(x) = f_{\psi} \circ C_{\psi}(x)$ . We then define our MI estimator on global/local pairs, maximizing the average estimated MI:

$$
(\hat {\omega}, \hat {\psi}) _ {L} = \underset {\omega , \psi} {\arg \max } \frac {1}{M ^ {2}} \sum_ {i = 1} ^ {M ^ {2}} \hat {\mathcal {L}} _ {\omega , \psi} \left(C _ {\psi} ^ {(i)} (X); E _ {\psi} (X)\right). \tag {6}
$$

We found success optimizing this "local" objective with multiple easy-to-implement architectures, and further implementation details are provided in the App. (A.1).

# 3.3 MATCHING REPRESENTATIONS TO A PRIOR DISTRIBUTION

Absolute magnitude of information is only one desirable property of a representation; depending on the application, good representations can be compact (Gretton et al., 2012), independent (Hyvärinen & Oja, 2000; Hinton, 2002; Dinh et al., 2014; Brakel & Bengio, 2017), disentangled (Schmidhuber, 1992; Rifai et al., 2012; Bengio et al., 2013; Chen et al., 2018; Gonzalez-Garcia et al., 2018), or independently controllable (Thomas et al., 2017). DIM imposes statistical constraints onto learned representations by implicitly training the encoder so that the push-forward distribution,  $\mathbb{U}_{\psi,\mathbb{P}}$ , matches a prior,  $\mathbb{V}$ . This is done (see Figure 6 in the App. A.1) by training a discriminator,  $D_{\phi}:\mathcal{V}\to \mathbb{R}$ , to estimate the divergence,  $\mathcal{D}(\mathbb{V}||\mathbb{U}_{\psi,\mathbb{P}})$ , then training the encoder to minimize this estimate:

$$
(\hat {\omega}, \hat {\psi}) _ {P} = \underset {\psi} {\arg \min } \underset {\phi} {\arg \max } \widehat {\mathcal {D}} _ {\phi} (\mathbb {V} | | \mathbb {U} _ {\psi , \mathbb {P}}) = \mathbb {E} _ {\mathbb {V}} [ \log D _ {\phi} (y) ] + \mathbb {E} _ {\mathbb {P}} [ \log (1 - D _ {\phi} (E _ {\psi} (x))) ]. \tag {7}
$$

This approach is similar to what is done in adversarial autoencoders (AAE, Makhzani et al., 2015), but without a generator. It is also similar to noise as targets (Bojanowski & Joulin, 2017), but trains the encoder to match the noise implicitly rather than using a priori noise samples as targets.

All three objectives - global and local MI maximization and prior matching - can be used together, and doing so we arrive at our complete objective for Deep InfoMax (DIM):

$$
\underset {\omega_ {1}, \omega_ {2}, \psi} {\arg \max } \left(\alpha \widehat {\mathcal {I}} _ {\omega_ {1}, \psi} (X; E _ {\psi} (X)) + \frac {\beta}{M ^ {2}} \sum_ {i = 1} ^ {M ^ {2}} \widehat {\mathcal {I}} _ {\omega_ {2}, \psi} \left(X ^ {(i)}; E _ {\psi} (X)\right)\right) + \underset {\psi} {\arg \min } \underset {\phi} {\arg \max } \gamma \widehat {\mathcal {D}} _ {\phi} (\mathbb {V} | | \mathbb {U} _ {\psi , \mathbb {P}}), \tag {8}
$$

where  $\omega_{1}$  and  $\omega_{2}$  are the discriminator parameters for the global and local objectives, respectively, and  $\alpha$ ,  $\beta$ , and  $\gamma$  are hyperparameters. We will show below that choices in these hyperparameters affect the learned representations in meaningful ways. As an interesting aside, we also show in the App. (A.7) that this prior matching can be used alone to train a generator of image data.

# 4 EXPERIMENTS

We test Deep InfoMax (DIM) on four imaging datasets to evaluate its representational properties:

- CIFAR10 and CIFAR100 (Krizhevsky & Hinton, 2009): two small-scale labeled datasets composed of  $32 \times 32$  images with 10 and 100 classes respectively.  
- Tiny ImageNet: A reduced version of ImageNet (Krizhevsky & Hinton, 2009) images scaled down to  $64 \times 64$  with a total of 200 classes.

- STL-10 (Coates et al., 2011): a dataset derived from ImageNet composed of  $96 \times 96$  images with a mixture of 100000 unlabeled training examples and 500 labeled examples per class. We use data augmentation with this dataset, taking random  $64 \times 64$  crops and flipping horizontally during unsupervised learning.  
CelebA (Yang et al., 2015, Appendix A.4 only): An image dataset composed of faces labeled with 40 binary attributes. This dataset evaluates DIM's ability to capture information that is more fine-grained than the class label and coarser than individual pixels.

For our experiments, we compare DIM against various unsupervised methods: Variational AutoEncoders (VAE, Kingma & Welling, 2013), Adversarial AutoEncoders (AAE, Makhzani et al., 2015), BiGAN (a.k.a. adversarially learned inference with a deterministic encoder: Donahue et al., 2016; Dumoulin et al., 2016), and Noise As Targets (NAT, Bojanowski & Joulin, 2017). See the App. (A.1) for details of the neural net architectures used in the experiments.

# 4.1 HOW DO WE EVALUATE THE QUALITY OF A REPRESENTATION?

Evaluation of representations is case-driven and relies on various proxies. Linear separability is commonly used as a proxy for disentanglement and mutual information (MI) between representations and class labels. Unfortunately, this will not show whether the representation has high MI with the class labels when the representation is not disentangled. Other works (Bojanowski & Joulin, 2017) have looked at transfer learning classification tasks by freezing the weights of the encoder and training a small fully-connected neural network classifier using the representation as input. Others still have more directly measured the MI between the labels and the representation (Rifai et al., 2012; Chen et al., 2018), which can also reveal the representation's degree of entanglement.

Class labels have limited use in evaluating representations, as we are often interested in information encoded in the representation that is unknown to us. Implicit models give us two new metrics for evaluating representation quality. First, we can use mutual information neural estimation (MINE, Belghazi et al., 2018) to more directly measure the MI between the input and output of the encoder.

Second, we can directly measure the independence of the representation using a discriminator. Given a batch of representations, we generate a factor-wise independent distribution with the same per-factor marginals by randomly shuffling each factor along the batch dimension. A similar trick has been used for learning maximally independent representations for sequential data (Brakel & Bengio, 2017). We can train a discriminator to estimate the KL-divergence between the original representations (joint distribution of the factors) and the shuffled representations (product of the marginals) using the Donsker-Varadhan representation as with MINE (see Figure 11). The higher the KL divergence, the more dependent the factors. We call this evaluation method Neural Dependency Measure (NDM) and show that it is sensible and empirically consistent in the App. (A.5).

To summarize, we use the following metrics for evaluating representations. For each of these, the encoder is held fixed unless noted otherwise:

- Linear classification using a support vector machine (SVM). This is simultaneously a proxy for MI of the representation with linear separability.  
- Non-linear classification using a single hidden layer neural network (200 units) with dropout. This is a proxy on MI of the representation with the labels separate from linear separability as measured with the SVM above.  
- Semi-supervised learning (STL-10 here), that is, fine-tuning the complete encoder by adding a small neural network on top of the last convolutional layer (matching architectures with a standard fully-supervised classifier).  
- MS-SSIM (Wang et al., 2003), using a decoder trained on the  $L_{2}$  reconstruction loss. This is a proxy for the total MI between the input and the representation and can indicate the amount of encoded pixel-level information.  
- Mutual information neural estimate (MINE),  $\widehat{I}_{\rho}(X,E_{\psi}(x))$  , between the input,  $X$  , and the output representation,  $E_{\psi}(x)$  , by training a discriminator with parameters  $\rho$  to maximize the DV estimator of the KL-divergence.  
- Neural dependency measure (NDM) using a second discriminator that measures the KL between  $E_{\psi}(x)$  and a batch-wise shuffled version of  $E_{\psi}(x)$ .

Table 1: Classification accuracy (top 1) results on CIFAR10 and CIFAR100. DIM(L) (i.e., with the local-only objective) outperforms all other unsupervised methods presented by a wide margin. In addition, DIM(L) approaches or even surpasses a fully-supervised classifier with similar architecture. DIM with the global-only objective is competitive with some models across tasks, but falls short when compared to generative models and DIM(L) on CIFAR100. Fully-supervised classification results are provided for comparison.

<table><tr><td rowspan="2">Model</td><td colspan="3">CIFAR10</td><td colspan="3">CIFAR100</td></tr><tr><td>conv</td><td>fc (1024)</td><td>Y(64)</td><td>conv</td><td>fc (1024)</td><td>Y(64)</td></tr><tr><td>Fully supervised</td><td colspan="3">75.39</td><td colspan="3">42.27</td></tr><tr><td>VAE</td><td>60.71</td><td>60.54</td><td>54.61</td><td>37.21</td><td>34.05</td><td>24.22</td></tr><tr><td>AAE</td><td>59.44</td><td>57.19</td><td>52.81</td><td>36.22</td><td>33.38</td><td>23.25</td></tr><tr><td>BiGAN</td><td>62.57</td><td>62.74</td><td>52.54</td><td>37.59</td><td>33.34</td><td>21.49</td></tr><tr><td>NAT</td><td>56.19</td><td>51.29</td><td>31.16</td><td>29.18</td><td>24.57</td><td>9.72</td></tr><tr><td>DIM(G)</td><td>52.2</td><td>52.84</td><td>43.17</td><td>27.68</td><td>24.35</td><td>19.98</td></tr><tr><td>DIM(L) (JSD)</td><td>73.25</td><td>73.62</td><td>66.96</td><td>48.13</td><td>45.92</td><td>39.60</td></tr><tr><td>DIM(L) (NCE)</td><td>75.21</td><td>75.57</td><td>69.13</td><td>49.74</td><td>47.72</td><td>41.61</td></tr></table>

Table 2: Classification accuracy (top 1) results on Tiny ImageNet and STL-10. For Tiny ImageNet, DIM with the local objective outperforms all other models presented by a large margin, and approaches accuracy of a fully-supervised classifier similar to the Alexnet architecture used here.

<table><tr><td></td><td colspan="3">Tiny ImageNet</td><td colspan="4">STL-10 (random crop pretraining)</td></tr><tr><td></td><td>conv</td><td>fc (4096)</td><td>Y(64)</td><td>conv</td><td>fc (4096)</td><td>Y(64)</td><td>SS</td></tr><tr><td>Fully supervised</td><td colspan="3">36.60</td><td colspan="4">68.7</td></tr><tr><td>VAE</td><td>18.63</td><td>16.88</td><td>11.93</td><td>58.27</td><td>56.72</td><td>46.47</td><td>68.65</td></tr><tr><td>AAE</td><td>18.04</td><td>17.27</td><td>11.49</td><td>59.54</td><td>54.47</td><td>43.89</td><td>64.15</td></tr><tr><td>BiGAN</td><td>24.38</td><td>20.21</td><td>13.06</td><td>71.53</td><td>67.18</td><td>58.48</td><td>74.77</td></tr><tr><td>NAT</td><td>13.70</td><td>11.62</td><td>1.20</td><td>64.32</td><td>61.43</td><td>48.84</td><td>70.75</td></tr><tr><td>DIM(G)</td><td>11.32</td><td>6.34</td><td>4.95</td><td>42.03</td><td>30.82</td><td>28.09</td><td>51.36</td></tr><tr><td>DIM(L) (JSD)</td><td>33.54</td><td>36.88</td><td>31.66</td><td>72.86</td><td>70.85</td><td>65.93</td><td>76.96</td></tr><tr><td>DIM(L) (NCE)</td><td>34.21</td><td>38.09</td><td>33.33</td><td>72.57</td><td>70.00</td><td>67.08</td><td>76.81</td></tr></table>

For the neural network classification evaluation above, we performed experiments on all datasets except CelebA, while for other measures we only looked at CIFAR10. For all classification tasks, we built separate classifiers on the high-level vector representation  $(Y)$ , the output of the previous fully-connected layer (fc) and the last convolutional layer (conv). Model selection for the classifiers was done by averaging the last 100 epochs of optimization, and the dropout rate and decaying learning rate schedule was set uniformly to alleviate over-fitting on the test set across all models.

# 4.2 REPRESENTATION LEARNING COMPARISON ACROSS MODELS

In the following experiments,  $\mathrm{DIM}(\mathrm{G})$  refers to DIM with a global-only objective ( $\alpha = 1, \beta = 0, \gamma = 1$ ) and  $\mathrm{DIM}(\mathrm{L})$  refers to DIM with a local-only objective ( $\alpha = 0, \beta = 1, \gamma = 0.1$ ), the latter chosen from the results of an ablation study presented in the App. (A.4). For the prior, we chose a compact uniform distribution on  $[0, 1]^{64}$ , which worked better in practice than other priors, such as Gaussian, unit ball, or unit sphere.

Classification comparisons Our classification results can be found in Tables 1 and 2. In general, DIM with the local objective,  $\mathrm{DIM}(\mathrm{L})$ , outperformed all models presented here by a significant margin on all datasets, regardless of which layer the representation was drawn from.  $\mathrm{DIM}(\mathrm{L})$  performs as well as or outperforms a fully-supervised classifier without fine-tuning, which indicates that the representations are nearly as good or better than the raw pixels given the model constraints. This supports the hypothesis that our local DIM objective is suitable for extracting class information.

Extended comparisons Tables 3 shows results on linear separability, reconstruction (MS-SSIM), mutual information, and independence (NDM) with the CIFAR10 dataset. For linear classifier results (SVC), we trained five support vector machines with a simple hinge loss for each model, averaging the

Table 3: Extended comparisons on CIFAR10. Linear classification results using SVM are over five runs. MS-SSIM is estimated by training a separate decoder using the fixed representation as input and minimizing the  $L2$  loss with the original input. Mutual information estimates were done using MINE and the neural dependence measure (NDM) were trained using a discriminator between unshuffled and shuffled representations. NDM measures for DIM are the measures with the sigmoid function applied at estimation and without in parentheses.  

<table><tr><td rowspan="2">Model</td><td colspan="4">Proxies</td><td colspan="2">Neural Estimators</td></tr><tr><td>SVM (conv)</td><td>SVM (fc)</td><td>SVM (Y)</td><td>MS-SSIM</td><td>Iρ(X,Y)</td><td>NDM</td></tr><tr><td>VAE</td><td>53.83 ± 0.62</td><td>42.14 ± 3.69</td><td>39.59 ± 0.01</td><td>0.72</td><td>93.02</td><td>1.62</td></tr><tr><td>AAE</td><td>55.22 ± 0.06</td><td>43.34 ± 1.10</td><td>37.76 ± 0.18</td><td>0.67</td><td>87.48</td><td>0.03</td></tr><tr><td>BiGAN</td><td>56.40 ± 1.12</td><td>38.42 ± 6.86</td><td>44.90 ± 0.13</td><td>0.46</td><td>37.69</td><td>24.49</td></tr><tr><td>NAT</td><td>48.62 ± 0.02</td><td>42.63 ± 3.69</td><td>39.59 ± 0.01</td><td>0.29</td><td>6.04</td><td>0.02</td></tr><tr><td>DIM(G)</td><td>46.8 ± 2.29</td><td>28.79 ± 7.29</td><td>29.08 ± 0.24</td><td>0.49</td><td>49.63</td><td>0.35(9.96)</td></tr><tr><td>DIM(L+G)</td><td>57.55 ± 1.442</td><td>45.56 ± 4.18</td><td>18.63 ± 4.79</td><td>0.53</td><td>101.65</td><td>0.5(22.89)</td></tr><tr><td>DIM(L)</td><td>63.25 ± 0.86</td><td>54.06 ± 3.6</td><td>49.62 ± 0.3</td><td>0.37</td><td>45.09</td><td>0.18(9.18)</td></tr></table>

Table 4: Augmenting NCE-based DIM with additional structural information – adding coordinate prediction tasks or occluding input patches when computing the global feature vector in DIM can improve the classification accuracy, particularly with the highly-compressed global features.  

<table><tr><td rowspan="2">Model</td><td colspan="3">CIFAR10</td><td colspan="3">CIFAR100</td></tr><tr><td>Y(64)</td><td>fc (1024)</td><td>conv</td><td>Y(64)</td><td>fc (1024)</td><td>conv</td></tr><tr><td>DIM</td><td>70.65</td><td>73.33</td><td>77.46</td><td>44.27</td><td>47.96</td><td>49.90</td></tr><tr><td>DIM (coord)</td><td>71.56</td><td>73.89</td><td>77.28</td><td>45.37</td><td>48.61</td><td>50.27</td></tr><tr><td>DIM (occlude)</td><td>72.87</td><td>74.45</td><td>76.77</td><td>44.89</td><td>47.65</td><td>48.87</td></tr><tr><td>DIM (coord + occlude)</td><td>73.99</td><td>75.15</td><td>77.27</td><td>45.96</td><td>48.00</td><td>48.72</td></tr></table>

test accuracy. For MINE, we used a decaying learning rate schedule, which helped reduce variance in estimates and provided faster convergence.

MS-SSIM correlated well with the MI estimate provided by MINE, indicating that these models encoded pixel-wise information well. As our prior matching was done using a sigmoid function on the representation,  $Y$ , we measured NDM with and without (in parentheses) this nonlinearity. Overall, all models showed much lower dependence than BiGAN, indicating the marginal of the encoder output is not matching to the generator's spherical Gaussian input prior. For MI, reconstruction-based models like VAE and AAE have high scores, and we found that combining local and global DIM objectives had very high scores ( $\alpha = 0.5$ ,  $\beta = 0.1$  is presented here as  $\mathrm{DIM}(\mathrm{L} + \mathrm{G})$ ). For more in-depth analyses, please see the ablation studies and the nearest-neighbor analysis in the App. (A.3, A.4).

# 4.3 ADDING COORDINATE INFORMATION AND OCCLUSIONS

Maximizing MI between global and local features is not the only way to leverage image structure. We consider augmenting DIM by adding input occlusion when computing global features and by adding auxiliary tasks which maximize MI between local features and absolute or relative spatial coordinates given a global feature. These additions improve classification results (see Table 4).

For occlusion, we randomly occlude part of the input when computing the global features, but compute local features using the full input. Maximizing MI between occluded global features and unoccluded local features aggressively encourages the global features to encode information which is shared across the entire image. For coordinate prediction, we maximize the model's ability to predict the coordinates  $(i,j)$  of a local feature  $c_{(i,j)} = C_{\psi}^{(i,j)}(x)$  after computing the global features  $y = E_{\psi}(x)$ . To accomplish this, we maximize  $\mathbb{E}[\log p_{\theta}((i,j)|y,c_{(i,j)})]$  (i.e., minimize the cross-entropy). We can extend the task to maximize conditional MI given global features  $y$  between pairs of local features  $(c_{(i,j)},c_{(i',j')})$  and their relative coordinates  $(i - i',j - j')$ . This objective can be written as  $\mathbb{E}[\log p_{\theta}((i - i',j - j')|y,c_{(i,j)},c_{(i',j')})]$ . We use both these objectives in our results.

Additional implementation details can be found in the App. (A.6). Roughly speaking, our input occlusions and coordinate prediction tasks can be interpreted as generalizations of inpainting (Pathak et al., 2016) and context prediction (Doersch et al., 2015) tasks which have previously been proposed

for self-supervised feature learning. Augmenting DIM with these tasks helps move our method further towards learning representations which encode images (or other types of inputs) not just in terms of compressing their low-level (e.g. pixel) content, but in terms of distributions over relations among higher-level features extracted from their lower-level content.

# 5 CONCLUSION

In this work, we introduced Deep InfoMax (DIM), a new method for learning unsupervised representations by maximizing mutual information, allowing for representations that contain locally-consistent information across structural "locations" (e.g., patches in an image). This provides a straightforward and flexible way to learn representations that perform well on a variety of tasks. We believe that this is an important direction in learning higher-level representations.

# REFERENCES

Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. Deep variational information bottleneck. arXiv preprint arXiv:1612.00410, 2016.  
Luís B Almeida. Linear and nonlinear ica based on mutual information. The Journal of Machine Learning Research, 4:1297-1318, 2003.  
Martin Arjovsky and Léon Bottou. Towards principled methods for training generative adversarial networks. In International Conference on Learning Representations, 2017.  
Suzanna Becker. An information-theoretic unsupervised learning algorithm for neural networks. University of Toronto, 1992.  
Suzanna Becker. Mutual information maximization: models of cortical self-organization. Network: Computation in neural systems, 7(1):7-31, 1996.  
Ishmael Belghazi, Aristide Baratin, Sai Rajeswar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, and R Devon Hjelm. Mine: mutual information neural estimation. arXiv preprint arXiv:1801.04062, ICML'2018, 2018.  
Anthony J Bell and Terrence J Sejnowski. An information-maximization approach to blind separation and blind deconvolution. Neural computation, 7(6):1129-1159, 1995.  
Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE Trans. Pattern Analysis and Machine Intelligence (PAMI), 35(8):1798-1828, 2013.  
Piotr Bojanowski and Armand Joulin. Unsupervised learning by predicting noise. arXiv preprint arXiv:1704.05310, 2017.  
Philemon Brakel and Yoshua Bengio. Learning independent features with adversarial nets for non-linear ica. arXiv preprint arXiv:1710.05050, 2017.  
Jianlong Chang, Lingfeng Wang, Gaofeng Meng, Shiming Xiang, and Chunhong Pan. Deep adaptive image clustering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5879-5887, 2017.  
Tian Qi Chen, Xuechen Li, Roger Grosse, and David Duvenaud. Isolating sources of disentanglement in variational autoencoders. arXiv preprint arXiv:1802.04942, 2018.  
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Infogan: Interpretable representation learning by information maximizing generative adversarial nets. In Advances in neural information processing systems, pp. 2172-2180, 2016.  
Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 215-223, 2011.

Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.  
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. arXiv preprint arXiv:1605.08803, 2016.  
Carl Doersch and Andrew Zisserman. Multi-task self-supervised visual learning. In The IEEE International Conference on Computer Vision (ICCV), 2017.  
Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In Proceedings of the IEEE International Conference on Computer Vision, 2015.  
Jeff Donahue, Philipp Krahenbuhl, and Trevor Darrell. Adversarial feature learning. arXiv preprint arXiv:1605.09782, 2016.  
M.D Donsker and S.R.S Varadhan. Asymptotic evaluation of certain markov process expectations for large time, iv. Communications on Pure and Applied Mathematics, 36(2):183-212, 1983.  
Alexey Dosovitskiy, Philipp Fischer, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with exemplar convolutional neural networks. IEEE transactions on pattern analysis and machine intelligence, 38(9):1734-1747, 2016.  
Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Alex Lamb, Martin Arjovsky, Olivier Mastropietro, and Aaron Courville. Adversarily learned inference. arXiv preprint arXiv:1606.00704, 2016.  
Abel Gonzalez-Garcia, Joost van de Weijer, and Yoshua Bengio. Image-to-image translation for cross-domain disentanglement. arXiv preprint arXiv:1805.09730, 2018.  
Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 13(Mar):723-773, 2012.  
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein gans. arXiv preprint arXiv:1704.00028, 2017.  
Michael Gutmann and Aapo Hyvarinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, pp. 297-304, 2010.  
Michael U Gutmann and Aapo Hyvarinen. Noise-contrastive estimation of unnormalized statistical models, with applications to natural image statistics. Journal of Machine Learning Research, 13 (Feb):307-361, 2012.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. Openreview, 2016.  
Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural computation, 14(8):1771-1800, 2002.  
R Devon Hjelm, Athul Paul Jacob, Tong Che, Adam Trischler, Kyunghyun Cho, and Yoshua Bengio. Boundary-seeking generative adversarial networks. In International Conference on Learning Representations, 2018.  
Aapo Hyvarinen and Erkki Oja. Independent component analysis: algorithms and applications. Neural networks, 13(4):411-430, 2000.  
Aapo Hyvarinen and Petteri Pajunen. Nonlinear independent component analysis: Existence and uniqueness results. Neural Networks, 12(3):429-439, 1999.

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Xu Ji, João F Henriques, and Andrea Vedaldi. Invariant information distillation for unsupervised image segmentation and clustering. arXiv preprint arXiv:1807.06653, 2018.  
Diederik Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Diederik P Kingma, Shakir Mohamed, Danilo Jimenez Rezende, and Max Welling. Semi-supervised learning with deep generative models. In Advances in Neural Information Processing Systems, pp. 3581-3589, 2014.  
Teuvo Kohonen. The self-organizing map. Neurocomputing, 21(1-3):1-6, 1998.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Ralph Linsker. Self-organization in a perceptual network. IEEE Computer, 21(3):105-117, 1988. doi: 10.1109/2.36. URL https://doi.org/10.1109/2.36.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.  
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan Frey. Adversarial autoencoders. arXiv preprint arXiv:1511.05644, 2015.  
Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge? In International Conference on Machine Learning, pp. 3478-3487, 2018.  
Andriy Mnih and Koray Kavukcuoglu. Learning word embeddings efficiently with noise-contrastive estimation. In Advances in neural information processing systems, pp. 2265-2273, 2013.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-gan: Training generative neural samplers using variational divergence minimization. In Advances in Neural Information Processing Systems, pp. 271-279, 2016.  
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. Context encoders: Feature learning by inpainting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
Danilo Jimenez Rezende, Shakir Mohamed, Ivo Danihelka, Karol Gregor, and Daan Wierstra. One-shot generalization in deep generative models. arXiv preprint arXiv:1603.05106, 2016.  
Salah Rifai, Pascal Vincent, Xavier Muller, Xavier Glorot, and Yoshua Bengio. Contractive autoencoders: Explicit invariance during feature extraction. In Proceedings of the 28th International Conference on International Conference on Machine Learning, pp. 833-840. Omnipress, 2011.  
Salah Rifai, Yoshua Bengio, Aaron Courville, Pascal Vincent, and Mehdi Mirza. Disentangling factors of variation for facial expression recognition. In European Conference on Computer Vision, pp. 808-822. Springer, 2012.  
Mehdi Sajjadi, Mehran Javanmardi, and Tolga Tasdizen. Regularization with stochastic transformations and perturbations for deep semi-supervised learning. In Advances in Neural Information Processing Systems, pp. 1163-1171, 2016.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, pp. 2234-2242, 2016.  
Jürgen Schmidhuber. Learning factorial codes by predictability minimization. *Neural Computation*, 4(6):863-879, 1992.  
Valentin Thomas, Jules Pondard, Emmanuel Bengio, Marc Sarfati, Philippe Beaudoin, Marie-Jean Meurs, Joelle Pineau, Doina Precup, and Yoshua Bengio. Independently controllable features. arXiv preprint arXiv:1708.01289, 2017.  
Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th international conference on Machine learning, pp. 1096-1103. ACM, 2008.  
Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of machine learning research, 11(Dec):3371-3408, 2010.  
Zhou Wang, Eero P Simoncelli, and Alan C Bovik. Multiscale structural similarity for image quality assessment. In Signals, Systems and Computers, 2004. Conference Record of the Thirty-Seventh Asilomar Conference on, volume 2, pp. 1398-1402. IEEE, 2003.  
Laurenz Wiskott and Terrence J Sejnowski. Slow feature analysis: Unsupervised learning of invariances. Neural computation, 14(4):715-770, 2002.  
Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. In International conference on machine learning, pp. 478-487, 2016.  
Shuo Yang, Ping Luo, Chen-Change Loy, and Xiaou Tang. From facial parts responses to face detection: A deep learning approach. In Proceedings of the IEEE International Conference on Computer Vision, pp. 3676-3684, 2015.  
Fisher Yu, Yinda Zhang, Shuran Song, Ari Seff, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.
