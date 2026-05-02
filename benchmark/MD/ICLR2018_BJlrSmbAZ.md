# BAYESIAN UNCERTAINTY ESTIMATION FOR BATCH NORMALIZED DEEP NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep neural networks have led to a series of breakthroughs, dramatically improving the state-of-the-art in many domains. The techniques driving these advances, however, lack a formal method to account for model uncertainty. While the Bayesian approach to learning provides a solid theoretical framework to handle uncertainty, inference in Bayesian-inspired deep neural networks is difficult. In this paper, we provide a practical approach to Bayesian learning that relies on a regularization technique found in nearly every modern network, batch normalization. We show that training a deep network using batch normalization is equivalent to approximate inference in Bayesian models, and we demonstrate how this finding allows us to make useful estimates of the model uncertainty. Using our approach, it is possible to make meaningful uncertainty estimates using conventional architectures without modifying the network or the training procedure. Our approach is thoroughly validated in a series of empirical experiments on different tasks and using various measures, showing it to outperform baselines on a majority of datasets with strong statistical significance.

# 1 INTRODUCTION

Deep learning has dramatically advanced the state of the art in a number of domains, and now surpasses human-level performance for certain tasks such as recognizing the contents of an image (He et al., 2015) and playing Go (Silver et al., 2017). But, despite their unprecedented discriminative power, deep networks are prone to make mistakes. Sometimes, the consequences of mistakes are minor – misidentifying a food dish or a species of flower (Liu et al., 2016) may not be life threatening. But deep networks can already be found in settings where errors carry serious repercussions such as autonomous vehicles (Chen et al., 2016) and high frequency trading. In medicine, we can soon expect automated systems to screen for skin cancer (Esteva et al., 2017), breast cancer (Shen, 2017), and to diagnose biopsies (Djuric et al., 2017). As autonomous systems based on deep learning are increasingly deployed in settings with the potential to cause physical or economic harm, we need to develop a better understanding of when we can be confident in the estimates produced by deep networks, and when we should be less certain.

Standard deep learning techniques used for supervised learning lack methods to account for uncertainty in the model, although sometimes the classification network's output vector is mistakenly understood to represent the model's uncertainty. The lack of a confidence measure can be especially problematic when the network encounters conditions it was not exposed to during training. For example, if a network trained to recognize dog breeds is given an image of a cat, it may predict it to belong to a breed of small dog with high probability. When exposed to data outside of the distribution it was trained on, the network is forced to extrapolate, which can lead to unpredictable behavior. In such cases, if the network can provide information about its uncertainty in addition to its point estimate, disaster may be avoided. This work focuses on estimating such predictive uncertainties in deep networks (Figure 1).

The Bayesian approach provides a solid theoretical framework for modeling uncertainty (Ghahramani, 2015), which has prompted several attempts to extend neural networks (NN) into a Bayesian setting. Most notably, Bayesian neural networks (BNNs) have been studied since the 1990's (Neal, 2012). Although they are simple to formulate, BNNs require substantially more computational resources than their non-Bayesian counterparts, and inference is difficult. Importantly, BNNs do

![](images/dddb1c659c13f5332d7b36701e97c5cd473ea405fe46d04abff9ffc712c428d5.jpg)

![](images/6189ba29b629f7abbb792057d2d6b818271b14089d1d0e369803d943dee58aab.jpg)

![](images/a9bbef9ca851ec402d69cc1d207afbde08cb056502c07ab41e9c52fc9e256d91.jpg)  
Figure 1: We propose a method to estimate uncertainty in any network using batch normalization (MCBN). Here, we show results on a toy dataset from networks with three hidden layers (30 units per layer). The solid line is the predictive mean of 500 stochastic forward passes. The outer area depicts the model's uncertainty as the  $95\%$  CI of the predictive distribution for each  $x$  value (inner shaded area is  $50\%$  CI). On the right, we show a similar plot using dropout to estimate uncertainty (MCDO) (Gal & Ghahramani, 2015). The bottom row depicts a minimally useful baseline - the same networks but with a constant uncertainty (CUBN, CUDO).

![](images/8d735f77c99ac15abd4909a6425545753d2ee1b5d4c618afd801a33c564540fc.jpg)

not scale well and struggle to compete with modern deep learning architectures. Recently, Gal & Ghahramani (2015) developed a practical solution to obtain uncertainty estimates by casting dropout training in conventional deep networks as an approximate Bayesian model. They showed that any network trained with dropout is an approximate Bayesian model, and uncertainty estimates can be obtained by computing the variance on multiple predictions with different dropout masks.

This technique, called Monte Carlo Dropout (MCDO), has a very attractive quality: it can be applied to existing NNs without any modification to the architecture or the way the network is trained. Uncertainty estimates come (nearly) for free. However, in recent years dropout has fallen out of favor, limiting MCDO's utility. Google's Inception network, which won ILSVRC in 2014, did not use dropout (Szegedy et al., 2015), nor did the ILSVRC 2015 winner, Microsoft's residual learning network (He et al., 2016). In place of traditional techniques like dropout, most modern networks such as Inception and ResNet have adopted other regularization techniques. In particular, batch normalization (BN) has become widespread thanks to its ability to stabilize learning with improved generalization (Ioffe & Szegedy, 2015).

An interesting aspect of BN is that the mini-batch statistics used for training each iteration depend on randomly selected batch members. We exploit this stochasticity and show that training using batch normalization, like dropout, is equivalent to approximate inference in Bayesian models<sup>1</sup>. We demonstrate how this finding allows us to make meaningful estimates of the model uncertainty in a technique we call Monte Carlo Batch Normalization (MCBN) (Figure 1). The method we propose makes no simplifying assumptions on the use of batch normalization, and applies to any network using BN as it appears in practical applications.

We validate our approach by empirical experiments on eight standard datasets used for uncertainty estimation. We measure uncertainty quality relative to a baseline of fixed uncertainty, and show that MCBN outperforms the baseline on nearly all datasets with strong statistical significance. We also show that the uncertainty quality of MCBN is on par with that of MCDO. As a practical demonstration of MCBN, we apply our method to estimate segmentation uncertainty using a conventional segmentation network (Badrinarayanan et al., 2015). Finally, as part of our evaluation, we make contributions to the methodology of measuring uncertainty quality by defining performance bounds on existing metrics and proposing a new visualization that provides an intuitive understanding of uncertainty quality.

# 2 RELATED WORK

Bayesian models provide a natural framework for modeling uncertainty, and several approaches have been developed to adapt NNs to Bayesian reasoning. A common approach is to place a prior distribution (often a Gaussian) over each weight. For infinite weights, the resulting model corresponds to a Gaussian process (Neal, 1995), and for a finite number of weights it corresponds to a Bayesian neural network (MacKay, 1992). Although simple to formulate, inference in BNNs is difficult (Gal, 2016). Therefore, focus has shifted to techniques to approximate the posterior distribution, leading to approximate BNNs. Methods based on variational inference (VI) typically rely on a fully factorized approximate distribution (Kingma & Welling, 2014; Hinton & Van Camp, 1993) but these methods do not scale easily. To alleviate these difficulties, Graves (2011) proposed a model using sampling methods to estimate a factorized posterior. Another approach, probabilistic backpropagation (PBP), also estimates a factorized posterior based on expectation propagation (Hernández-Lobato & Adams, 2015).

Deep Gaussian Processes (DGPs) formulate GPs as Bayesian models capable of working on large datasets with the aid of a number of strategies to address scaling and complexity requirements (Bui et al., 2016). The authors compare DGP with a number of state-of-the-art approximate BNNs, showing superior performance in terms of RMSE and uncertainty quality $^{2}$ . Another recent approach to Bayesian learning, Bayesian hypernetworks, use a neural network to learn a distribution of parameters over another neural network (Krueger et al., 2017). Although these recent techniques address some of the difficulties with approximate BNNs, they all require modifications to the architecture or the way networks are trained, as well as specialized knowledge from practitioners.

Recently, Gal (2016) showed that a network trained with dropout implicitly performs the VI objective. Therefore any network trained with dropout can be treated as an approx. Bayesian model by making multiple predictions as forward passes through the network while sampling different dropout masks for each prediction. An estimate of the posterior can be obtained by computing the mean and variance of the predictions. This technique, referred to here as MCDO, has been empirically demonstrated to be competitive with other approx. BNN methods and DGPs in terms of RMSE and uncertainty quality (Li & Gal, 2017). However, as the name implies, MCDO depends on dropout. While once ubiquitous in training deep learning models, dropout has largely been replaced by batch normalization in modern networks, limiting its usefulness.

# 3 METHOD

The methodology of this work is to pose a deep network trained with batch normalization as a Bayesian model in order to obtain uncertainty estimates associated with its predictions. In the following, we briefly introduce Bayesian models and a variational approximation to it using Kullback-Leibler (KL) divergence following Gal & Ghahramani (2015). We continue by showing a batch normalized deep network can be seen as an approximate Bayesian model. Then, by employing theoretical insights as well as empirical analysis, we study the induced prior on the parameters when using batch normalization. Finally, we describe the procedure we use for estimating uncertainty of batch normalized deep networks' output.

# 3.1 BAYESIAN MODELING

We assume a finite training set  $\mathbf{D} = \{(\mathbf{x}_i,\mathbf{y}_i)\}_{i = 1:N}$  where each  $(\mathbf{x}_i,\mathbf{y}_i)$  is a sample-label pair. Using  $\mathbf{D}$ , we are interested in learning an inference function  $f_{\omega}(\mathbf{x},\mathbf{y})$  with parameters  $\omega$ . In deterministic models, the estimated label  $\hat{\mathbf{y}}$  is obtained as follows:

$$
\hat {\mathbf {y}} = \underset {\mathbf {y}} {\arg \max} f _ {\boldsymbol {\omega}} (\mathbf {x}, \mathbf {y})
$$

We assume  $f_{\omega}(\mathbf{x}, \mathbf{y}) = p(\mathbf{y}|\mathbf{x}, \omega)$  (e.g. in soft-max classifiers), and is normalized to a proper probability distribution. In Bayesian modeling, in contrast to finding a point estimate of the model parameters, the idea is to estimate an (approximate) posterior distribution of the model parameters

$p(\boldsymbol {\omega}|\mathbf{D})$  to be used for probabilistic prediction:

$$
p (\mathbf {y} | \mathbf {x}, \mathbf {D}) = \int f _ {\boldsymbol {\omega}} (\mathbf {x}, \mathbf {y}) p (\boldsymbol {\omega} | \mathbf {D}) d \boldsymbol {\omega}
$$

The predicted label,  $\hat{\mathbf{y}}$ , can then be accordingly obtained by sampling  $p(\mathbf{y}|\mathbf{x},\mathbf{D})$  or takings its maxima.

Variational Approximation In approximate Bayesian modeling, it is a common approach to learn a parametrized approximating distribution  $q_{\theta}(\omega)$  that minimizes  $\mathrm{KL}(q_{\theta}(\omega)||p(\omega|\mathbf{D}))$ ; the Kullback-Leibler (KL) divergence of posterior w.r.t. its approximation, instead of the true posterior. Minimizing this KL divergence is equivalent to the following minimization while being free of the data term  $p(\mathbf{D})^3$ :

$$
\mathcal {L} _ {\mathrm {V A}} (\boldsymbol {\theta}) := - \sum_ {i = 1} ^ {N} \int q _ {\boldsymbol {\theta}} (\boldsymbol {\omega}) \ln f _ {\boldsymbol {\omega}} (\mathbf {x} _ {i}, \mathbf {y} _ {i}) \mathrm {d} \boldsymbol {\omega} + \mathrm {K L} (q _ {\boldsymbol {\theta}} (\boldsymbol {\omega}) | | p (\boldsymbol {\omega}))
$$

Using Monte Carlo integration to approximate the integral with one realized  $\hat{\omega}_i$  for each sample  $i^4$ , and optimizing over mini-batches of size  $M$ , the approximated objective becomes:

$$
\hat {\mathcal {L}} _ {\mathrm {V A}} (\boldsymbol {\theta}) := - \frac {N}{M} \sum_ {i = 1} ^ {M} \ln f _ {\boldsymbol {\omega} _ {i}} \left(\mathbf {x} _ {i}, \mathbf {y} _ {i}\right) + \operatorname {K L} \left(q _ {\boldsymbol {\theta}} (\boldsymbol {\omega}) \mid \mid p (\boldsymbol {\omega})\right) \tag {1}
$$

The first term is the data likelihood and the second term is divergence of the model prior w.r.t. the approximated distribution.

We now describe the optimization procedure of a deep network with batch normalization and draw the resemblance to the approximate Bayesian modeling in Eq (1).

# 3.2 BATCH NORMALIZED DEEP NETS AS BAYESIAN MODELING

The inference function of a feed-forward deep network with  $L$  layers can be described as:

$$
f _ {\boldsymbol {\omega}} (\mathbf {x}) = \mathbf {W} ^ {L} a (\mathbf {W} ^ {L - 1} \dots a (\mathbf {W} ^ {2} a (\mathbf {W} ^ {1} \mathbf {x}))
$$

where  $a(.)$  is an element-wise nonlinearity function and  $\mathbf{W}^l$  is the weight vector at layer  $l$ . Furthermore, we denote the input to layer  $l$  as  $\mathbf{x}^l$  with  $\mathbf{x}^1 = \mathbf{x}$  and we then set  $\mathbf{h}^l = \mathbf{W}^l\mathbf{x}^l$ . Parenthesized super-index for matrices (e.g.  $\mathbf{W}^{(j)}$ ) and vectors (e.g.  $x^{(j)}$ ) indicates  $j$ th row and element respectively. Super-index  $u$  refers to a specific unit at layer  $l$ , (e.g.  $\mathbf{W}^u = \mathbf{W}^{l,(j)}, h^u = h^{l,(j)}$ ).<sup>5</sup>

Batch Normalization Each layer of a deep network is constructed by several linear units whose parameters are the rows of the weight matrix  $\mathbf{W}$ . Batch normalization is a unit-wise operation proposed in Ioffe & Szegedy (2015) to standardize the distribution of each unit's input. It essentially converts a unit's output  $h^u$  in the following way:

$$
\hat {h} ^ {u} = \frac {h ^ {u} - \mathbb {E} \left[ h ^ {u} \right]}{\sqrt {\operatorname {V a r} \left[ h ^ {u} \right]}}
$$

where the expectations are computed over the training set<sup>6</sup>. However, often in deep networks, the weight matrices are optimized using back-propagated errors calculated on mini-batches of data. Therefore, during training, the estimated mean and variance on the mini-batch  $\mathbf{B}$  is used, which we denote by  $\mu_{\mathrm{B}}$  and  $\sigma_{\mathrm{B}}$  respectively. This makes the inference at training time for a sample  $\mathbf{x}$  a stochastic process, varying based on other samples in the mini-batch.

Loss Function and Optimization Training deep networks with mini-batch optimization involves a (regularized) risk minimization with the following form:

$$
\mathcal {L} _ {\mathrm {R R}} (\boldsymbol {\omega}) := \frac {1}{M} \sum_ {i = 1} ^ {M} l \left(\hat {\mathbf {y}} _ {i}, \mathbf {y} _ {i}\right) + \Omega (\boldsymbol {\omega})
$$

Where the first term is the empirical loss on the training data and the second term is a regularization penalty acting as a prior on model parameters  $\omega$ . If the loss  $l$  is cross-entropy for classification or sum-of-squares for regression problems (assuming i.i.d. Gaussian noise on labels), the first term is equivalent to minimizing the negative log-likelihood:

$$
\mathcal {L} _ {\mathrm {R R}} (\boldsymbol {\omega}) := - \frac {1}{M \tau} \sum_ {i = 1} ^ {M} \ln f _ {\boldsymbol {\omega}} (\mathbf {x} _ {i}, \mathbf {y} _ {i}) + \Omega (\boldsymbol {\omega}).
$$

with  $\tau = 1$  for classification. In a batch normalized network the model parameters are  $\{\mathbf{W}^{1:L},\gamma^{1:L},\beta^{1:L},\pmb{\mu}_{\mathbf{B}}^{1:L},\pmb{\sigma}_{\mathbf{B}}^{1:L}\}$ . If we decouple the learnable parameters  $\theta = \{\mathbf{W}^{1:L},\gamma^{1:L},\beta^{1:L}\}$  from the stochastic parameters  $\omega = \{\pmb{\mu}_{\mathbf{B}}^{1:L},\pmb{\sigma}_{\mathbf{B}}^{1:L}\}$ , we get the following objective at each step of the mini-batch optimization of a batch normalized network:

$$
\mathcal {L} _ {\mathrm {R R}} (\boldsymbol {\theta}) := - \frac {1}{M \tau} \sum_ {i = 1} ^ {M} \ln f _ {\{\boldsymbol {\theta}, \hat {\omega} _ {i} \}} \left(\mathbf {x} _ {i}, \mathbf {y} _ {i}\right) + \Omega (\boldsymbol {\theta}) \tag {2}
$$

where  $\hat{\omega}_i$  is the mean and variances for sample  $i$ 's mini-batch at a certain training step. Note that while  $\hat{\omega}_i$  formally needs to be i.i.d. for each training example, a batch normalized network samples the stochastic parameters once per training step (mini-batch). For a large number of epochs, however, the distribution of sampled batch members for a given training example converges to the i.i.d. case.

Comparing Eq. (1) and Eq. (2) reveals that the optimization objectives are identical, if there exists a prior  $p(\boldsymbol{\omega})$  corresponding to  $\Omega(\boldsymbol{\theta})$  such that  $\frac{\partial}{\partial \theta} \mathrm{KL}(q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) || p(\boldsymbol{\omega})) = N \tau \frac{\partial}{\partial \theta} \Omega(\boldsymbol{\theta})$ . In a batch normalized network,  $q_{\boldsymbol{\theta}}(\boldsymbol{\omega})$  corresponds to the joint distribution of the normalization parameters  $\mu_{\mathbf{B}}^{1:L}, \sigma_{\mathbf{B}}^{1:L}$ , as implied by the repeated sampling from  $\mathbf{D}$  during training. This is an approximation of the true posterior, where we have restricted the posterior to lie within the domain of our parametric network and source of randomness. With that we can use a pre-trained batch normalized network to estimate the uncertainty of its prediction using the inherent stochasticity of BN. Before that, we briefly discuss what Bayesian prior is induced in a typical batch normalized network.

# 3.3 PRIOR  $p(\omega)$

The purpose of  $\Omega(\theta)$  is to reduce variance in deep networks. L2-regularization, also referred to as weight decay ( $\Omega(\theta) = \lambda \sum_{l=1:L} ||W^l||^2$ ), is a popular technique in deep learning. The induced prior from L2-regularization is studied in Appendix 6.5. Under some approximations as outlined in the Appendix, we find that BN for a deep network with FC layers and ReLU activations induce Gaussian distributions over BN unit's means and standard deviations, centered around the population values given by  $\mathbf{D}$  (Eq. (6), details in Appendix 6.3). Factorizing this distribution across all stochastic parameters and assuming Gaussian priors, we find the approximate corresponding priors:

$$
p (\mu_ {\mathbf {B}} ^ {u}) = \mathcal {N} (0, \frac {J _ {l - 1} x ^ {2}}{2 N \tau \lambda_ {l}})
$$

$$
p (\sigma_ {\mathbf {B}} ^ {u}) = \mathcal {N} (\mu_ {p}, \sigma_ {p} ^ {2})
$$

where  $J_{l-1}$  is the dimensionality of the layer's inputs and  $x$  is the average input over  $\mathbf{D}$  for all input units. In the absence of scale and shift transformations from the previous BN layer, it converges towards an exact prior for large training datasets and deep networks (under the assumptions of the factorized distribution). The mean and variance for the BN unit's standard deviation,  $\mu_p$  and  $\sigma_p^2$ , have no relevance for the reconciliation of the optimization objectives of Eq. (1) and (2).

# 3.4 PREDICTIVE UNCERTAINTY IN BATCH NORMALIZED DEEP NETS

In the absence of the true posterior we rely on the approximate posterior to express an approximate predictive distribution:

$$
p ^ {*} (\mathbf {y} | \mathbf {x}, \mathbf {D}) := \int f _ {\boldsymbol {\omega}} (\mathbf {x}, \mathbf {y}) q _ {\boldsymbol {\theta}} (\boldsymbol {\omega}) d \boldsymbol {\omega}
$$

Following Gal & Ghahramani (2015) we estimate the first and second moment of the predictive distribution empirically (see Appendix 6.4 for details). For regression, the first two moments are:

$$
\mathbb {E} _ {p ^ {*}} [ \mathbf {y} ] \approx \frac {1}{T} \sum_ {i = 1} ^ {T} f _ {\tilde {\omega} _ {i}} (\mathbf {x})
$$

$$
\operatorname {C o v} _ {p ^ {*}} [ \mathbf {y} ] \approx \tau^ {- 1} \mathbf {I} + \frac {1}{T} \sum_ {i = 1} ^ {T} f _ {\hat {\omega} _ {i}} (\mathbf {x}) ^ {\intercal} f _ {\hat {\omega} _ {i}} (\mathbf {x}) - \mathbb {E} _ {p ^ {*}} [ \mathbf {y} ] ^ {\intercal} \mathbb {E} _ {p ^ {*}} [ \mathbf {y} ]
$$

where each  $\hat{\omega}_i$  corresponds to sampling the net's stochastic parameters  $\omega = \{\pmb{\mu}_{\mathbf{B}}^{1:L}, \pmb{\sigma}_{\mathbf{B}}^{1:L}\}$  the same way as during training. Sampling  $\hat{\omega}_i$  therefore involves sampling a batch  $\mathbf{B}$  from the training set and updating the parameters in the BN units, just as if we were taking a training step with  $\mathbf{B}$ . Recall that from a VA perspective, training the network amounted to minimizing  $\mathrm{KL}(q_{\theta}(\omega) || p(\omega | \mathbf{D}))$  wrt  $\pmb{\theta}$ . Sampling  $\hat{\omega}_i$  from the training set, and keeping the size of  $\mathbf{B}$  consistent with the mini-batch size used during training, ensures that  $q_{\theta}(\omega)$  during inference remains identical to the approximate posterior optimized during training.

After each update of the net's stochastic parameters, we take a forward pass with input  $\mathbf{x}$ , producing output  $f_{\hat{\omega}_i}(\mathbf{x})$ . After  $T$  such stochastic forward passes, we compute the mean and sample variance of outputs to find the mean  $\mathbb{E}_{p^*}[\mathbf{y}]$  and variance  $\mathrm{Cov}_{p^*}[\mathbf{y}]$  of the approximate predictive distribution. Note that  $\mathrm{Cov}_{p^*}[\mathbf{y}]$  also requires addition of constant variance from observation noise,  $\tau^{-1}\mathbf{I}$ .

The network is trained just as a regular BN network. The difference is in using the trained network for prediction. Instead of replacing  $\omega = \{\pmb{\mu}_{\mathbf{B}}^{1:L}, \pmb{\sigma}_{\mathbf{B}}^{1:L}\}$  with population values from  $\mathbf{D}$ , we update these parameters stochastically, once for each forward pass.

The form of  $p^*$  can be approximated by a Gassuan for each output dimension (for regression). We assume bounded domains for each input dimension, wide layers throughout the network, and a unimodal distribution of weights centered at 0. By the Liapounov CLT condition, the first layer then receives approximately Gaussian inputs (a proof can be found in Lehmann (1999)). Having sampled  $\mu_{\mathbf{B}}^{u}$  and  $\sigma_{\mathbf{B}}^{u}$  from a mini-batch, each BN unit's output is bounded. CLT thereby continues to hold for deeper layers, including  $f_{\omega}(\mathbf{x}) = \mathbf{W}^{L}\mathbf{x}^{L}$ . A similar motivation for a Gaussian approximation of Dropout has been presented by Wang & Manning (2013).

The actual form of  $p^*$  is likely to be highly multimodal, as can be seen immediately from  $f_{\omega}(\mathbf{x}) = \mathbf{W}^{L}\mathbf{x}^{L}$  with elements in  $\mathbf{x}^{L}$  normalized, scaled and shifted differently. Gal & Ghahramani (2015) note the multimodality as well, since MCDO implies a bimodal variational distribution over each weight matrix column.

# 4 EXPERIMENTS AND RESULTS

We assess the uncertainty quality of MCBN quantitatively and qualitatively. Our quantitative analysis relies on eight standard regression datasets, listed in Table 1. Publicly available from the UCI Machine Learning Repository (University of California, 2017) and Delve (Ghahramani, 1996), these datasets have been used to benchmark comparative models in recent related literature (see Hernández-Lobato & Adams (2015), Gal & Ghahramani (2015), Bui et al. (2016) and Li & Gal (2017)). We report results using standard metrics, and also propose useful upper and lower bounds to normalize these metrics for a more meaningful interpretation in Section 4.2.

<table><tr><td>Dataset name</td><td>N</td><td>Q</td><td>Target Feature</td></tr><tr><td>Boston Housing</td><td>506</td><td>13</td><td></td></tr><tr><td>Concrete Compressive Strength</td><td>1,030</td><td>8</td><td></td></tr><tr><td>Energy Efficiency</td><td>768</td><td>8</td><td>Heating Load</td></tr><tr><td>Kinematics 8nm</td><td>8,192</td><td>8</td><td></td></tr><tr><td>Power Plant</td><td>9,568</td><td>4</td><td></td></tr><tr><td>Protein Tertiary Structure</td><td>45,730</td><td>9</td><td></td></tr><tr><td>Wine Quality (Red)</td><td>1,599</td><td>11</td><td></td></tr><tr><td>Yacht Hydrodynamics</td><td>308</td><td>6</td><td></td></tr></table>

Table 1: Properties of the eight regression datasets used to evaluate MCBN.  $N$  is the dataset size and  $Q$  is the n.o. input features. Only one target feature was used. In cases where the raw datasets contain more than one target feature, the feature used is specified by target feature.

Our qualitative results consist of three parts. First, in Figure 1 we demonstrate that MCBN produces reasonable uncertainty bounds on a toy dataset in the style of (Karpathy, 2015). Second, we develop a new visualization of uncertainty quality by plotting test errors sorted by predicted variance in Figure 2. Finally, we apply MCBN to SegNet (Kendall et al., 2015), demonstrating the benefits of MCBN in an existing batch normalized network.

# 4.1 METRICS

We evaluate uncertainty quality based on two metrics, described below: Predictive Log Likelihood (PLL) and Continuous Ranked Probability Score (CRPS). We also propose upper and lower bounds for these metrics which can be used to normalize them and provide a more meaningful interpretation.

Predictive Log Likelihood (PLL) Predictive Log Likelihood is a widely accepted metric for uncertainty quality, used as the main uncertainty quality metric for regression (e.g. (Hernández-Lobato & Adams, 2015), (Gal & Ghahramani, 2015), (Bui et al., 2016) and (Li & Gal, 2017)). A key property is that PLL makes no assumptions about the form of the distribution. The measure is defined for a probabilistic model  $f_{\omega}(\mathbf{x})$  and a single observation  $(\mathbf{y}_i, \mathbf{x}_i)$  as:

$$
\operatorname {P L L} \left(f _ {\boldsymbol {\omega}} (\mathbf {x}), \left(\mathbf {y} _ {i}, \mathbf {x} _ {i}\right)\right) = \log p \left(\mathbf {y} _ {i} \mid f _ {\boldsymbol {\omega}} \left(\mathbf {x} _ {i}\right)\right)
$$

where  $p(\mathbf{y}_i|f_\omega (\mathbf{x}_i))$  is the model's predicted PDF evaluated at  $\mathbf{y}_i$ , given the input  $x_{i}$ . A more detailed description is given in Appendix 6.4. The metric is unbounded and maximized by a perfect prediction (mode at  $\mathbf{y}_i$ ) with no variance. As the predictive mode moves away from  $\mathbf{y}_i$ , increasing the variance tends to increase PLL (by maximizing probability mass at  $\mathbf{y}_i$ ). While PLL is an elegant measure, it has been criticized for allowing outliers to have an overly negative effect on the score (Selten, 1998).

Continuous Ranked Probability Score (CRPS) Continuous Ranked Probability Score is a less sensitive measure that takes the full predicted PDF into account. A prediction with low variance that is slightly offset from the true observation will receive a higher score form CRPS than PLL. In order for CRPS to be analytically tractable, we need to assume a Gaussian unimodal predictive distribution. CRPS is defined as

$$
\operatorname {C R P S} \left(f _ {\omega} \left(x _ {i}\right), \left(y _ {i}, x _ {i}\right)\right) = \int_ {- \infty} ^ {\infty} (F (y) - \mathbb {1} (y \geq y _ {i})) ^ {2} d y
$$

where  $F(y)$  is the predictive CDF, and  $\mathbb{1}(y \geq y_i) = 1$  if  $y \geq y_i$  and 0 otherwise (for univariate distributions) (Gneiting & Raftery, 2007). CRPS is interpreted as the sum of the squared area between the CDF and 0 where  $y < y_i$  and between the CDF and 1 where  $y \geq y_i$ . A perfect prediction with no variance yields a CRPS of 0; for all other cases the value is larger. CRPS has no upper bound.

# 4.2 BENCHMARK MODELS AND NORMALIZED METRICS

In order to establish a lower bound on useful performance for uncertainty estimates, we define a baseline that predicts constant variance regardless of input. This benchmark model produces identical point estimates as MCBN, which yield the same predictive means. The variance is set to a fixed value that optimizes CRPS on validation data. This model reflects our best guess of constant

variance on test data - any improvement in uncertainty quality from MCBN would indicate a sensible estimate of uncertainty. We call this model Constant Uncertainty BN (CUBN). Implementing MCDO as a comparative model, we similarly define a baseline for dropout, Constant Uncertainty Dropout (CUDO). The difference in variance modeling between MCBN, CUBN, MCDO and CUDO are visualized in plots of uncertainty bounds on toy data in Figure 1.

For a probabilistic model  $f$ , an upper bound on uncertainty performance can also be defined for CRPS and PLL. For each observation  $(y_i, x_i)$ , a value for the predictive variance  $T_i$  can be chosen that maximizes PLL or minimizes CRPS<sup>8</sup>. Using CUBN as a lower bound and the optimized CRPS score as the upper bound, uncertainty estimates can be normalized between these bounds (1 indicating optimal performance, and 0 indicating performance on par with fixed uncertainty). We call this normalized measure  $\overline{\mathrm{CRPS}} = \frac{\mathrm{CRPS}(f,(y_i,x_i)) - \mathrm{CRPS}(f_{CU},(y_i,x_i))}{\min_T \mathrm{CRPS}(f,(y_i,x_i)) - \mathrm{CRPS}(f_{CU},(y_i,x_i))} \times 100$ , and the PLL analogue  $\overline{\mathrm{PLL}} = \frac{\mathrm{PLL}(f,(y_i,x_i)) - \mathrm{PLL}(f_{CU},(y_i,x_i))}{\max_T \mathrm{PLL}(f,(y_i,x_i)) - \mathrm{PLL}(f_{CU},(y_i,x_i))} \times 100$ . This normalized measure gives an intuitive understanding of how close a Bayesian model is to estimating the perfect uncertainty for each prediction.

We also evaluate CRPS and PLL for an adaptation of the authors' implementation of Multiplicative Normalizing Flows (MNF) for variational Bayesian networks (Louizos & Welling, 2017). This is a recent model specialized to allow a more flexible posterior what is achievable by e.g. MCDO's bimodal variational over weight columns. MNF uses auxiliary variables on which the posterior is a latent. By applying normalizing flows to the auxiliary variable such that it can take on complex distributions, the approximate posterior becomes highly flexible.

# 4.3 TEST SETUP

Our evaluation of MCBN and MCDO is largely comparable to that of Hernández-Lobato & Adams (2015), in that we use similar datasets and metrics. This setup was later also followed by Gal & Ghahramani (2015), where we in comparison implement a different hyperparameter selection, allow for a larger range of dropout rates, and use larger networks with two hidden layers.

With the exception of Protein Tertiary Structure $^9$ , all our models share a similar architecture: two hidden layers with 50 units each, using ReLU activations. Input and output data were normalized during training. Results were averaged over five random splits of  $20\%$  test and  $80\%$  training and cross-validation (CV) data. For each split, 5-fold CV by grid search with a RMSE minimization objective was used to find training hyperparameters and optimal n.o. epochs. For BN-based models, the hyperparameter grid consisted of a weight decay factor ranging from 0.1 to  $1^{-15}$  by a log 10 scale, and a batch size range from 32 to 1024 by a log 2 scale. For DO-based models, the hyperparameter grid consisted of the same weight decay range, and dropout probabilities in  $\{0.2, 0.1, 0.05, 0.01, 0.005, 0.001\}$ . DO-based models used a batch size of 32 in all evaluations.

The model with optimal training hyperparameters was used to optimize  $\tau$  numerically. This optimization was made in terms of average CV CRPS for MCBN, CUBN, MCDO, and CUDO respectively, before evaluation on the test data.

All estimates for the predictive distribution were obtained by taking 500 stochastic forward passes through the network, throughout training and testing. The implementation was done with Tensor-Flow. The Adam optimizer was used to train all networks, with a learning rate of 0.001. The extensive part of the experiments (i.e. training and cross validation) was done on Amazon web services using 3000 machine-hours. All code necessary for reproducing both the quantitative and qualitative results is released in an anonymous github repository (https://github.com/iclr-mcbn/mcbn).

# 4.4 TEST RESULTS

A summary of the results measuring uncertainty quality of MCBN, MCDO and MNF are provided in Table 2. Tests are run over eight datasets using 5 random 80-20 splits of the data with 5 different random seeds each split. We report CRPS and PLL, expressed as a percentage, which reflects how close the model is to the upper bound. The upper bounds and lower bounds for each metric are de

<table><tr><td rowspan="2">Dataset</td><td colspan="5">CRPS</td><td colspan="5">PLL</td></tr><tr><td>MCBN</td><td>MCDO</td><td colspan="2">MNF</td><td>MCBN</td><td colspan="2">MCDO</td><td colspan="2">MNF</td><td></td></tr><tr><td>Boston</td><td>8.50 ***</td><td>3.06 ***</td><td colspan="2">8.30 ***</td><td>10.49 ***</td><td colspan="2">5.51 ***</td><td colspan="2">3.58 ***</td><td></td></tr><tr><td>Concrete</td><td>3.91 ***</td><td>0.93 *</td><td colspan="2">6.05 ***</td><td>-36.36 **</td><td colspan="2">10.92 ***</td><td colspan="2">9.71 ***</td><td></td></tr><tr><td>Energy</td><td>5.75 ***</td><td>1.37 ns</td><td colspan="2">3.45 ns</td><td>10.89 ***</td><td colspan="2">-14.28 *</td><td colspan="2">2.62 ns</td><td></td></tr><tr><td>Kin8nm</td><td>2.85 ***</td><td>1.82 ***</td><td colspan="2">1.01 *</td><td>1.68 ***</td><td colspan="2">-0.26 ns</td><td colspan="2">-0.44 ns</td><td></td></tr><tr><td>Power</td><td>0.24 ***</td><td>-0.44 ***</td><td colspan="2">-0.83 ***</td><td>0.33 **</td><td colspan="2">3.52 ***</td><td colspan="2">-1.38 ***</td><td></td></tr><tr><td>Protein</td><td>2.66 ***</td><td>0.99 ***</td><td colspan="2">TBU</td><td>2.56 ***</td><td colspan="2">6.23 ***</td><td colspan="2">TBU</td><td></td></tr><tr><td>Wine (Red)</td><td>0.26 **</td><td>2.00 ***</td><td colspan="2">TBU</td><td>0.19 *</td><td colspan="2">2.91 ***</td><td colspan="2">TBU</td><td></td></tr><tr><td>Yacht</td><td>-56.39 ***</td><td>21.42 ***</td><td colspan="2">-54.18 ***</td><td>45.58 ***</td><td colspan="2">-41.54 ns</td><td colspan="2">71.18 ***</td><td></td></tr></table>

Table 2: Uncertainty quality measured on eight datasets. MCBN, MCDO and MNF are compared over 5 random 80-20 splits of the data with 5 different random seeds each split. Reported values are uncertainty metrics CRPS and PLL normalized to a lower bound of constant variance and upper bound that maximizes the metric. CRPS and PLL are expressed as a percentage, reflecting how close the model is to the upper bound. We check to see if CRPS and PLL significantly exceed the baseline using a one sample t-test (significance level indicated by *s). Best performer versus their baseline for each dataset and metric is marked by bold. See text for further details.

![](images/01d20428a5a9e30671a2ef1aa3bed76cc84680fa78adca8df4e9890fafaae860.jpg)

![](images/0d81e4bf94e13e98eacc1208169ee8886d5928ef92fc18a0db50eadd85a6dff7.jpg)

![](images/371d8c52b68821b78f4ebda6105eb30f197cbcd5f77293ff0c52cfded42168cc.jpg)

![](images/3700c8ec348b356fd4fe000a5ebdef05e6e9b995efcf3303fdfa57dec54b3fdc.jpg)  
Figure 2: Errors in predictions (gray dots) sorted by estimated uncertainty on select datasets. The shaded areas show MCBN's (blue) and MCDO's (red) model uncertainty (light area  $95\%$  CI, dark area  $50\%$  CI). Gray dots show absolute prediction errors on the test set, and the gray line depicts a running mean of the errors. The dashed line indicates the optimized constant uncertainty. A correlation between estimated uncertainty (shaded area) and mean error (gray) indicates the uncertainty estimates are meaningful for estimating errors. See Appendix for complete results.

![](images/bd0b5618c74f6bd9306291583e67f06e4c9b187ce7630f8764cebb84ad293ca5.jpg)

![](images/234051fbf5ab4df3482aa575136bda32c92458ddab1b508ec7dbeeefa15a434b.jpg)

scribed in Section 4.2. We check to see if the reported values of  $\overline{\mathrm{CRPS}}$  and  $\overline{\mathrm{PLL}}$  significantly exceed the lower bound models (CUBN and CUDO) using a one sample t-test, where the significance level is indicated by *s. Further details from the experiment are available in Appendix 6.6.

In Figure 2, we provide a novel visualization of uncertainty quality visualization in regression datasets. Errors in the model predictions are sorted by estimated uncertainty. The shaded areas show the model uncertainty and gray dots show absolute prediction errors on the test set. A gray line depicts a running mean of the errors. The dashed line indicates the optimized constant uncertainty. In these plots, we can see a correlation between estimated uncertainty (shaded area) and mean error (gray). This trend indicates that the model uncertainty estimates can recognize samples with larger (or smaller) potential for predictive errors.

Qualitative results for Bayesian SegNet using MCBN was produced by using the main CamVid model in Kendall et al. (2015). The pre-trained model was obtained from the online model zoo and was used without modification. 10 instances of mini-batches with size 6 were used to estimate the mean and variance of MCBN. Qualitative results can be found in Figure 3 depicting intuitive

![](images/be305042fbd9a4d62b736ca6a4eb854780c91c08d9bb5e66e5d315cdfec73476.jpg)

![](images/ab2db1243ab3a1e0b6ec7d40440cba4e539dbd5c612f96f5385fce706c845c57.jpg)

![](images/2d4a8044903d8ee664b4f020195a52297569087a1fb40e9d8b35b127ecf53596.jpg)  
Figure 3: Results applying MCBN to Bayesian SegNet (Kendall et al., 2015). In the upper left, a scene from the CamVid driving scenes dataset. In the upper right, the Bayesian estimated segmentation. In the lower left, estimated uncertainty using MCBN for the car class. In the lower right, the estimated uncertainty of MCBN for all 11 classes.

uncertainty at object boundaries. Quantitative measures on various segmentation datasets can be obtained and is beyond the scope of this work.

We provide additional experimental results in Appendix 6.6. In Tables 3 and 4, we show the mean CRPS and PLL values for MCBN and MCDO. These results indicate that MCBN performs on par with MCDO across several datasets. In Table 6 we provide RMSE results of the MCBN and MCDO networks in comparison with non-stochastic BN and DO networks. These results indicate that the procedure of multiple forward passes in MCBN and MCDO show slight improvements in the predictive accuracy of the network.

# 5 DISCUSSION

The results presented in Table 2 and Appendix 6.6 indicate that MCBN generates meaningful uncertainty estimates which correlate with actual errors in the model's prediction. We show statistically significant improvements over CUBN in the majority of the datasets, both in terms of CRPS and PLL. The visualizations in Figure 2 and in Appendix 6.6 show clear correlations between the estimated model uncertainty and actual errors produced by the network. We perform the same experiments using MCDO, and find that MCBN generally performs on par with MCDO. Looking closer, in terms of CRPS, MCBN performs better than MCDO in more cases than not. However, care must be used when comparing different models. The learned network parameters are different, leading to different predictive means which can confound direct comparison.

The results on the Yacht Hydrodynamics dataset seem contradictory. The CRPS score for MCBN is extremely negative, while the PLL score is extremely positive. The opposite trend is observed for MCDO. To add to the puzzle, the visualization in Figure 2 depicts an extremely promising uncertainty estimation that models the predictive errors with high fidelity. We hypothesize that this strange behavior is due to the small size of the data set, which only contains 60 test samples, or due to the Gaussian assumption of CRPS. There is also a large variability in the model's accuracy on this dataset, which further confounds the measurements for such limited data.

One might criticize the overall quality of the uncertainty estimates of MCBN and MCDO based on the magnitude of the CRPS and PLL scores in Table 2. The scores rarely exceed  $10\%$  improvement over the lower bound. However, we caution that these measures should be taken in context. The upper bound is very difficult to achieve in practice (it is optimized for each test sample individually),

and the lower bound is a quite reasonable estimate for uncertainty. We have further compared against the recent work of Louizos & Welling (2017), and find comparable results to their MNF-based variational technique specifically targeted to increase the flexibility of the approximate posterior.

Our approximation of the implied prior in Appendix 6.5 also provides a new interpretation of the empirical evidence that significantly lower  $\lambda$  should be used in batch normalized networks (Ioffe & Szegedy, 2015). From a VA perspective, too strong a regularization for a given dataset size could be seen as constraining the prior distribution of BN units' means, effectively narrowing the approximate posterior.

In this work, we have shown that training a deep network using batch normalization is equivalent to approximate inference in Bayesian models. Using our approach, it is possible to make meaningful uncertainty estimates using conventional architectures without modifying the network or the training procedure. We show evidence that the uncertainty estimates from MCBN correlate with actual errors in the model's prediction, and are useful for practical tasks such as regression or semantic image segmentation. Our experiments show that MCBN yields an improvement over the baseline of optimized constant uncertainty on par with MCDO and MNF. Finally, we make contributions to the evaluation of uncertainty quality by suggesting new evaluation metrics based on useful baselines and upper bounds, and proposing a new visualization tool which gives an intuitive visual explanation of uncertainty quality. Finally, it should be noted that, over the past few years, batch normalization has become an integral part of most-if-not-all cutting edge deep networks which signifies the relevance of our work for estimating model uncertainty.

# REFERENCES

Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. arXiv preprint arXiv:1511.00561, 2015.  
Thang D. Bui, Daniel Hernández-Lobato, Yingzhen Li, José Miguel Hernández-Lobato, and Richard E. Turner. Deep Gaussian Processes for Regression using Approximate Expectation Propagation. In ICML, 2016.  
Xiaozhi Chen, Kaustav Kundu, Ziyu Zhang, Huimin Ma, Sanja Fidler, and Raquel Urtasun. Monocular 3d object detection for autonomous driving. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2147-2156, 2016.  
Ugljesa Djuric, Gelareh Zadeh, Kenneth Aldape, and Phedias Diamandis. Precision histology: how deep learning is poised to revitalize histomorphology for personalized cancer care. npj Precision Oncology, 1(1):22, 2017.  
Andre Esteva, Brett Kuprel, Roberto A. Novoa, Justin Ko, Susan M. Swetter, Helen M. Blau, and Sebastian Thrun. Dermatologist-level classification of skin cancer with deep neural networks. Nature, Feb 2017.  
Yarin Gal. Uncertainty in Deep Learning. PhD thesis, University of Cambridge, 2016.  
Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML, 48:1-10, 2015.  
Zoubin Ghahramani. Delve Datasets. University of Toronto, 1996. URL http://www.cs.toronto.edu/\~delve/data/kin/desc.html.  
Zoubin Ghahramani. Probabilistic machine learning and artificial intelligence. Nature, 521(7553): 452-459, May 2015.  
Tilmann Gneiting and Adrian E Raftery. Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association, 102(477):359-378, 2007.  
Alex Graves. Practical Variational Inference for Neural Networks. NIPS, 2011.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pp. 1026-1034, 2015.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Jose Miguel Hernández-Lobato and Ryan Adams. Probabilistic backpropagation for scalable learning of bayesian neural networks. In International Conference on Machine Learning, pp. 1861-1869, 2015.  
Geoffrey E Hinton and Drew Van Camp. Keeping the neural networks simple by minimizing the description length of the weights. In Proceedings of the sixth annual conference on Computational learning theory, pp. 5-13. ACM, 1993.  
Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. Arxiv, 2015. URL http://arxiv.org/abs/1502.03167.  
Andrej Karpathy. Convnetjs demo: toy 1d regression, 2015. URL http://cs.stanford.edu/people/karpathy/convnetjs/demo/regression.html.  
Alex Kendall, Vijay Badrinarayanan, and Roberto Cipolla. Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding. CoRR, abs/1511.0, 2015. URL http://arxiv.org/abs/1511.02680.  
Diederik P Kingma and Max Welling. Auto-Encoding Variational Bayes. In ICLR, 2014.  
David Krueger, Chin-Wei Huang, Riashat Islam, Ryan Turner, Alexandre Lacoste, and Aaron Courville. Bayesian hypernetworks. arXiv preprint arXiv:1710.04759, 2017.  
Erich Leo Lehmann. Elements of Large-Sample Theory. Springer Verlag, New York, 1999. ISBN 0387985956.  
Yingzhen Li and Yarin Gal. Dropout Inference in Bayesian Neural Networks with Alpha-divergences. arXiv, 2017.  
Xiao Liu, Tian Xia, Jiang Wang, Yi Yang, Feng Zhou, and Yuanqing Lin. Fully convolutional attention networks for fine-grained recognition. arXiv preprint arXiv:1603.06765, 2016.  
Christos Louizos and Max Welling. Multiplicative normalizing flows for variational Bayesian neural networks. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 2218-2227, International Convention Centre, Sydney, Australia, 06-11 Aug 2017. PMLR. URL http://proceedings.mlr.press/v70/luozos17a.html.  
David JC MacKay. A practical bayesian framework for backpropagation networks. Neural computation, 4(3):448-472, 1992.  
Radford M Neal. BAYESIAN LEARNING FOR NEURAL NETWORKS. PhD thesis, University of Toronto, 1995.  
Radford M Neal. Bayesian learning for neural networks, volume 118. Springer Science & Business Media, 2012.  
Reinhard Selten. Axiomatic characterization of the quadratic scoring rule. Experimental Economics, 1(1):43-62, 1998.  
Li Shen. End-to-end training for whole image breast cancer diagnosis using an all convolutional design. arXiv preprint arXiv:1708.09427, 2017.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. Mastering the game of go without human knowledge. Nature, 550(7676):354-359, Oct 2017.

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1-9, 2015.  
Irvine University of California. UC Irvine Machine Learning Repository, 2017. URL https://archive.ics.uci.edu/ml/index.html.  
Sida I Wang and Christopher D Manning. Fast dropout training. Proceedings of the 30th International Conference on Machine Learning, 28:118-126, 2013. URL http://machinelearning.wustl.edu/mlpapers/papers/wang13a.
