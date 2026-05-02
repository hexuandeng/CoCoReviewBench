# VARIATIONAL NETWORK QUANTIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We formulate the preparation of a neural network for pruning and few-bit quantization as a variational inference problem. We introduce a quantizing prior that leads to a multi-modal, sparse posterior distribution over weights and further derive a differentiable KL approximation for this prior. After training with Variational Network Quantization (VNQ), weights can be replaced by deterministic quantization values with small to negligible loss of task accuracy (including pruning by setting weights to 0). Our method does not require fine-tuning after quantization. We show results for ternary quantization on LeNet-5 (MNIST) and DenseNet-121 (CIFAR-10).

# 1 INTRODUCTION

Parameters of a trained network commonly exhibit high degrees of redundancy (Denil et al., 2013) which implies an over-parametrization of the network. Network compression methods implicitly or explicitly aim at the systematic reduction of redundancy in neural network models while at the same time retaining a high level of task accuracy. Besides architectural approaches, such as SqueezeNet (Iandola et al., 2016) or MobileNets (Howard et al., 2017), many compression methods either perform some form of pruning or quantization. Pruning is the removal of irrelevant units (weights, neurons or convolutional filters) (LeCun et al., 1990). Relevance of weights is often determined by the absolute value ("magnitude based pruning" (Han et al., 2016; 2017; Guo et al., 2016)), but more sophisticated methods have been known for decades, e.g. based on second-order (Optimal Brain Damage (LeCun et al., 1990) and Optimal Brain Surgeon Hassibi & Stork (1993)) or ARD (automatic relevance determination, a Bayesian framework for determining the relevance of weights, (MacKay, 1995; Neal, 1995; Karaletsos & Ratsch, 2015)). Quantization is the reduction of the bit-precision of weights, activations or even gradients, which is particularly desirable from a hardware perspective (Sze et al., 2017). Methods range from fixed bit-width computation (e.g. 12-bit fixed point) to aggressive quantization such as binarization of weights and activations (Courbariaux et al., 2016; Rastegari et al., 2016; Zhou et al., 2016; Hubara et al., 2016). Few-bit quantization (2 to 6 bits) is often performed by k-means clustering of trained weights with subsequent fine-tuning of the cluster centers (Han et al., 2016). Pruning and quantization have been shown to work well in conjunction (Han et al., 2016). In so-called "ternary" networks, weights are either negative, zero or positive which also allows for simultaneous pruning and few-bit quantization (Li et al., 2016; Zhu et al., 2016).

Our work is closely related to some recent Bayesian methods for network compression (Ullrich et al., 2017; Molchanov et al., 2017; Louizos et al., 2017; Neklyudov et al., 2017) that learn a posterior distribution over network weights under a sparsity-inducing prior. The posterior distribution over network parameters allows identifying redundancies through three means: (1) weights with an expected value very close to zero and (2) weights with a large variance can be pruned as they do not contribute much to the overall computation. (3) the posterior variance over non-pruned parameters can be used to determine the required bit-precision (quantization noise can be made as large as implied by the posterior uncertainty). Additionally, variational Bayesian inference is known to automatically reduce parameter redundancy by penalizing overly complex models.

In this paper we present Variational Network Quantization, a Bayesian network compression method for simultaneous pruning and few-bit quantization of weights. We extend previous work introducing a multi-modal quantizing prior that penalizes weights of low variance unless they lie close to one of the target values for quantization. As a result, weights are either drawn to one of the quantization target values or they are assigned large variance values—see Fig. 1. After training, our method yields a

Bayesian Neural Network with a multi-modal posterior over weights (typically with one mode fixed at 0), which is the basis for subsequent pruning and quantization. However, posterior uncertainties can also be interesting for network introspection and analysis, as well as for obtaining uncertainty estimates over network predictions Gal & Ghahramani (2015); Gal (2016); Depeweg et al. (2016; 2017). After pruning and hard quantization, and without the need for additional fine-tuning, our method yields a deterministic feed-forward neural network with heavily quantized weights. Our method is applicable to pre-trained networks but can also be used for training from scratch. Target values for quantization can either be manually fixed or they can be learned during training via hierarchical Bayesian inference. We demonstrate our results for the case of ternary quantization on LeNet-5 (MNIST) and DenseNet-121 (CIFAR-10).

![](images/f0f98486b0fa7e63e501c6d166126ef3451383c4141156477de5c4ddc927b55c.jpg)  
(a) Pre-trained network. No obvious clusters are visible in the network trained without VNQ. No regularization was used during pre-training.

![](images/d64445bd0746feeda65b7f755ad8b6694fd4a20ae68b481043fe10be860dae16.jpg)  
(b) Soft-quantized network after VNQ training. Weights tightly cluster around the quantization target values.  
Figure 1: Distribution of weights (means  $\theta$  and log-variance  $\log \sigma^2$ ) before and after VNQ training of LeNet-5 on MNIST (validation accuracy before:  $99.2\%$  vs. after 195 epochs:  $99.31\%$ ). Top row: scatter plot of weights (blue dots) per layer. Means were initialized from pre-trained deterministic network, variances with  $\log \sigma^2 = -8$ . Bottom row: corresponding density<sup>1</sup>. Red shaded areas show the funnel-shaped "basins of attraction" induced by the quantizing prior. Target values for ternary quantization (the codebook) have also been learned. After training, weights with small (absolute) expected value or large variance (inside the area marked by the dotted line, corresponding to  $\log \alpha_{ij} \leq \log T_\alpha = 2$ ) are pruned and remaining weights are quantized without loss in accuracy.

# 2 PRELIMINARIES

# 2.1 WHY BAYES FOR COMPRESSION?

Bayesian inference can be well motivated from an information-theoretic treatment of (lossy) compression (Cover & Thomas, 2006; Tishby et al., 2000; Genewein et al., 2015). In particular, Bayesian inference automatically penalizes overly complex parametric models, an effect known as "Bayesian Occams Razor" in Bayesian model selection (MacKay, 2003; Genewein & Braun, 2014). The same effect leads to automatic regularization in variational Bayesian inference over model parameters (Grünwald, 2007; Graves, 2011) (see Molchanov et al. (2017), where the authors show that Sparse Variational Dropout (Sparse VD) successfully prevents a network from fitting a unstructured data, that is a random labeling). This is particularly interesting since regularization is the basis for compression and is thought to be key for generalization (MacKay, 2003; Grünwald, 2007). The automatic regularization effect is based on maximizing model evidence, where model parameters are marginalized. A very complex model might have a parameter setting that achieves extremely good likelihood given the data, however, since the model evidence is based on the average or marginal likelihood, overly complex models are penalized for having many parameter settings with poor likelihood. The argument that Bayesian methods search for optimal model structure can also be made from an information-theoretic point-of-view by investigating the equivalence of variational inference and the Minimum description length (MDL) principle Rissanen (1978); Grünwald (2007); Graves (2011); Louizos et al. (2017). The evidence lower bound (ELBO, see Eq. (1)), which is maximized

in variational inference, is the sum of two terms: one, the average message length required to transmit outputs (labels) to a receiver that knows the inputs and the posterior over model parameters and two, the average message length to transmit the posterior parameters to a receiver that knows the prior over parameters:

$$
\mathcal {L} ^ {\text {E L B O}} = \underbrace {\text {n e g . r e c o n s t r . e r r o r}} _ {- \mathcal {L} ^ {E}} + \underbrace {\text {n e g . K L d i v e r g e n c e}} _ {- \mathcal {L} ^ {C} = \text {e n t r o p y - c r o s s e n t r o p y}},
$$

compare Eq. (1). Maximizing the ELBO minimizes the total message length:  $\max \mathcal{L}^{\mathrm{ELBO}} = \min \mathcal{L}^{E} + \mathcal{L}^{C}$ , leading to an optimal trade-off between short description length of the data and the model (thus minimizing the sum of error cost  $\mathcal{L}^{E}$  and model complexity cost  $\mathcal{L}^{C}$ ). Interestingly, MDL dictates the use of probabilistic models since they are in general "more compressible" compared to deterministic models: high uncertainty over parameters is rewarded by the entropy term in  $\mathcal{L}^{C}$  higher uncertainty allows the quantization noise to be higher, thus requiring lower bit-precision for a parameter (the bits back argument (Hinton & Van Camp, 1993; Louizos et al., 2017)).

# 2.2 VARIATIONAL BAYES AND REPARAMETRIZATION

Let  $\mathcal{D}$  be a dataset of  $N$  pairs  $(x_{n},y_{n})_{n = 1}^{N}$  and  $p(y|x,w)$  be a parameterized model that predicts outputs  $y$  given inputs  $x$  and parameters  $w$ . A Bayesian neural network models a (posterior) distribution over parameters  $w$  instead of just a point-estimate. The posterior is given by Bayes' rule:  $p(w|\mathcal{D}) = p(\mathcal{D}|w)p(w) / p(\mathcal{D})$ , where  $p(w)$  is the prior over parameters. Computation of the true posterior is in general intractable. Common approaches to approximate inference in neural networks are for instance: MCMC methods pioneered in (Neal, 1995) and later refined e.g. via stochastic gradient Langevin dynamics (Welling & Teh, 2011), or variational approximations to the true posterior (Graves, 2011), Bayes by Backprop (Blundell et al., 2015), Expectation Backpropagation (Soudry et al., 2014), Probabilistic Backpropagation (Hernandez-Lobato & Adams, 2015). In the latter methods the true posterior is approximated by a parameterized distribution  $q_{\phi}(w)$ . Variational parameters  $\phi$  are optimized by minimizing the Kullback-Leibler divergence between the true and the approximate posterior  $D_{\mathrm{KL}}(q_{\phi}(w)||p(w|\mathcal{D}))$ . Since computation of the true posterior is intractable, minimizing this KL divergence is approximately performed by maximizing the so-called "evidence lower bound" (ELBO) or "negative variational free energy" (Kingma & Welling, 2014):

$$
\begin{array}{l} \mathcal {L} (\phi) = \underbrace {\sum_ {n = 1} ^ {N} \mathbb {E} _ {q _ {\phi} (w)} [ \log p \left(y _ {n} \mid x _ {n} , w\right) ]} _ {L _ {\mathcal {D}} (\phi)} - D _ {\mathrm {K L}} \left(q _ {\phi} (w) | | p (w)\right) (1) \\ \simeq \mathcal {L} ^ {\mathrm {S G V B}} (\phi) = \frac {N}{M} \sum_ {m = 1} ^ {M} \log p \left(\tilde {y} _ {m} \mid \tilde {x} _ {m}, f \left(\phi , \epsilon_ {m}\right)\right) - D _ {\mathrm {K L}} \left(q _ {\phi} (w) | | p (w)\right) (2) \\ \end{array}
$$

where we have used the Reparameterization Trick $^2$  (Kingma & Welling, 2014) in Eq. (2) to get an unbiased, differentiable, minibatch-based Monte Carlo estimator of the expected log likelihood  $L_{\mathcal{D}}(\phi)$ . A mini-batch of data is denoted by  $(\tilde{x}_m,\tilde{y}_m)^M_{m = 1}$ . Additionally, and in line with similar work (Molchanov et al., 2017; Louizos et al., 2017; Neklyudov et al., 2017), we use the Local Reparameterization Trick (Kingma et al., 2015) to further reduce variance of the stochastic ELBO gradient estimator, which locally marginalizes weights at each layer and instead samples directly from the distribution over pre-activations (which can be computed analytically). See Appendix A.2 for more details on the Local Reparametrization. Commonly, the prior  $p(w)$  and the parametric form of the posterior  $q_{\phi}(w)$  are chosen such that the KL divergence term can be computed analytically (e.g. a fully factorized Gaussian prior and posterior, known as the mean-field approximation). Due to the particular choice of prior in our work, a closed-form expression for the KL divergence cannot be obtained but instead we use a differentiable approximation (see Sec. 3.3).

# 2.3 VARIATIONAL DROPOUT

Dropout (Srivastava et al., 2014) is a method originally introduced for regularization of neural networks, where activations are stochastically dropped (i.e. set to zero) with a certain probability  $p$

during training. It was shown that dropout, i.e. multiplicative noise on inputs, is equivalent to having noisy weights and vice versa (Wang & Manning, 2013; Kingma et al., 2015). Multiplicative Gaussian noise  $\xi_{ij} \sim \mathcal{N}(1, \alpha = \frac{p}{1 - p})$  on a weight  $w_{ij}$  induces a Gaussian distribution

$$
w _ {i j} = \theta_ {i j} \xi_ {i j} = \theta_ {i j} (1 + \sqrt {\alpha} \epsilon_ {i j}) \sim \mathcal {N} \left(\theta_ {i j}, \alpha \theta_ {i j} ^ {2}\right) \tag {3}
$$

with  $\epsilon_{ij} \sim \mathcal{N}(0,1)$ . In standard (Gaussian) dropout training, the dropout rates  $\alpha$  (or  $p$  to be precise) are fixed and the expected log likelihood  $L_{\mathcal{D}}(\phi)$  (first term in Eq. (1)) is maximized with respect to the means  $\theta$ . Kingma et al. (2015) show that Gaussian dropout training is mathematically equivalent to maximizing the ELBO (both terms in Eq. (1)), under a prior  $p(w)$  and fixed  $\alpha$  where the KL term does not depend on  $\theta$ :

$$
\mathbb {E} _ {q _ {\alpha}} \left[ L _ {\mathcal {D}} (\theta) \right] - \mathcal {L} (\alpha , \theta) = D _ {\mathrm {K L}} \left(q _ {\alpha} (w) \mid \mid p (w)\right), \tag {4}
$$

where the dependencies on  $\alpha$  and  $\theta$  of the terms in Eq. (1) have been made explicit. The only prior that meets this requirement is the scale invariant log-uniform prior:

$$
p \left(\log \left| w _ {i j} \right|\right) = \text {c o n s t .} \Leftrightarrow p \left(\left| w _ {i j} \right|\right) \propto \frac {1}{\left| w _ {i j} \right|}. \tag {5}
$$

Using this result, it is straightforward to learn individual dropout-rates  $\alpha_{ij}$  per weight, by including  $\alpha$  into the set of variational parameters  $\phi = (\theta ,\alpha)$ . This procedure was introduced in (Kingma et al., 2015) under the name "Variational Dropout". With the choice of a log-uniform prior (Eq. (5)) and a factorized Gaussian approximate posterior  $q_{\phi}(w_{ij}) = \mathcal{N}(\theta_{ij},\alpha \theta_{ij}^2)$  (Eq. (3)) the KL term in Eq. (1) is not analytically tractable, but the authors of Kingma et al. (2015) present an approximation

$$
- D _ {\mathrm {K L}} \left(q _ {\phi} \left(w _ {i j}\right) | | p \left(w _ {i j}\right)\right) \approx \operatorname {c o n s t .} + 0. 5 \log \alpha + c _ {1} \alpha + c _ {2} \alpha^ {2} + c _ {3} \alpha^ {3}, \tag {6}
$$

see original publication for numerical values of  $c_{1}, c_{2}, c_{3}$ . Note that due to the mean-field approximation, where the posterior over all weights factorizes into a product over individual weights  $q_{\phi}(w) = \prod q_{\phi}(w_{ij})$ , the KL divergence factorizes into a sum of individual KL divergences  $D_{\mathrm{KL}}(q_{\phi}(w)||p(w)) = \sum D_{\mathrm{KL}}(q_{\phi}(w_{ij})||p(w_{ij}))$ .

# 2.4 PRUNING UNITS WITH LARGE DROPOUT RATES

Learning dropout rates is interesting for network compression since neurons or weights with very high dropout rates  $p \to 1$  can very likely be pruned without loss in accuracy. However, as the authors of Sparse Variational Dropout (sparse VD) (Molchanov et al., 2017) report, the approximation in Eq. (6) is only accurate for  $\alpha \leq 1$  (corresponding to  $p \leq 0.5$ ). For this reason, the original variational dropout paper restricted  $\alpha$  to values smaller or equal to 1, which are unsuitable for pruning. Molchanov et al. (2017) propose an improved approximation, which is very accurate on the full range of  $\log \alpha$ :

$$
- D _ {\mathrm {K L}} \left(q _ {\phi} \left(w _ {i j}\right) \| p \left(w _ {i j}\right)\right) \approx \operatorname {c o n s t .} + k _ {1} S \left(k _ {2} + k _ {3} \log \alpha_ {i j}\right) - 0. 5 \log \left(1 + \alpha_ {i j} ^ {- 1}\right) = F _ {\mathrm {K L}, \mathrm {L U}} \left(\theta_ {i j}, \sigma_ {i j}\right), \tag {7}
$$

with  $k_{1} = 0.63576$ ,  $k_{2} = 1.87320$  and  $k_{3} = 1.48695$  and  $S$  denoting the sigmoid function. Additionally, the authors propose to use an additive, instead of a multiplicative noise reparameterization, which significantly reduces variance in the gradient  $\frac{\partial\mathcal{L}^{\mathrm{SGVB}}}{\partial\theta_{ij}}$  for large  $\alpha_{ij}$ . To achieve this, the multiplicative noise term is replaced with an exactly equivalent additive noise term  $\sigma_{ij}\epsilon_{ij}$  with  $\sigma_{ij}^{2} = \alpha_{ij}\theta_{ij}^{2}$  and the set of variational parameters becomes  $\phi = (\theta ,\sigma)$ :

$$
w _ {i j} = \theta_ {i j} \underbrace {(1 + \sqrt {\alpha} \epsilon_ {i j})} _ {\text {m u l t . n o i s e}} = \theta_ {i j} \underbrace {+ \sigma_ {i j} \epsilon_ {i j}} _ {\text {a d d . n o i s e}} \sim \mathcal {N} \left(\theta_ {i j}, \sigma_ {i j} ^ {2}\right), \quad \epsilon \sim \mathcal {N} (0, 1). \tag {8}
$$

After Sparse VD training, pruning is performed by thresholding  $\alpha_{ij} = \frac{\sigma_{ij}^2}{\theta_{ij}^2}$ , which translates into a threshold for the variance-to-mean ratio (also known as the index of dispersion, a limit-case of the Fano factor). In Molchanov et al. (2017) a threshold of  $\log \alpha = 3$  is used, which roughly corresponds to  $p > 0.95$ . Pruning weights that lie above a threshold of  $T_{\alpha}$  leads to

$$
\frac {\sigma_ {i j} ^ {2}}{\theta_ {i j} ^ {2}} \geq T _ {\alpha} \Leftrightarrow \sigma_ {i j} ^ {2} \geq T _ {\alpha} \theta_ {i j} ^ {2}, \tag {9}
$$

which means effectively that weights with large variance but also weights of lower variance and a mean  $\theta_{ij}$  close to zero are pruned. A visualization of the pruning threshold can also be seen in Fig 1 (the "central funnel", i.e. the area marked by the red dotted lines for a threshold for  $T_{\alpha} = 2$ ). Sparse VD training can be performed from random initialization or with pre-trained networks by initializing the means  $\theta_{ij}$  accordingly. In Bayesian Compression (Louizos et al., 2017) and Structured Bayesian Pruning (Neklyudov et al., 2017) Sparse VD has been extended to include group-sparsity constraints, which allows for pruning of whole neurons or convolutional filters (via learning their corresponding dropout rates).

# 2.5 SPARSITY INDUCING PRIORS

The prior  $p(w)$  can be used to induce sparsity into the posterior by having high density at zero and heavy tails. There is a well known family of such distributions: scale-mixtures of normals (Andrews & Mallows, 1974; Louizos et al., 2017; Ingraham & Marks, 2017):

$$
w \sim \mathcal {N} (0, z ^ {2}); \quad z \sim p (z),
$$

where the scales of  $w$  are random variables. A well-known example is the spike-and-slab prior (Mitchell & Beauchamp, 1988), which has a delta-spike at zero and a slab over the real line. Gal & Ghahramani (2015); Kingma et al. (2015) show how Dropout (Srivastava et al., 2014) implies a spike-and-slab distribution over weights. The log uniform prior used in Sparse VD (Eq. (5)) can also be derived as a marginalized scale-mixture of normals

$$
p \left(w _ {i j}\right) \propto \int \frac {1}{| z |} \mathcal {N} \left(w _ {i j} \mid 0, z ^ {2}\right) d z = \frac {1}{\left| w _ {i j} \right|}; \quad p (z) \propto \frac {1}{| z |}, \tag {10}
$$

also known as the normal-Jeffreys prior (Figueiredo, 2002). Louizos et al. (2017) discuss how the log-uniform prior can be seen as a continuous relaxation of the spike-and-slab prior and how the alternative formulation through the normal-Jeffreys distribution can be used to couple the scales of weights that belong together and thus learn dropout rates for whole neurons or convolutional filters, which is the basis for Bayesian Compression (Louizos et al., 2017) and Structured Bayesian Pruning (Neklyudov et al., 2017).

# 3 VARIATIONAL NETWORK QUANTIZATION

We formulate the preparation of a neural network for a post-training quantization step as a variational inference problem. To this end, we introduce a multi-modal, quantizing prior and train by maximizing the ELBO (Eq. (2)) under a mean-field approximation of the posterior (i.e. a fully factorized Gaussian). The goal of our algorithm is to achieve soft quantization, that is learning a posterior distribution such that the accuracy-loss introduced by post-training quantization is small. Our variational posterior approximation and training procedure is similar to Kingma et al. (2015) and Molchanov et al. (2017) with the crucial difference of using a quantizing prior that drives weights towards the target values for quantization.

# 3.1 A QUANTIZING PRIOR

The log uniform prior (Eq. (5)) can be viewed as a continuous relaxation of the spike-and-slab prior with a spike at location 0 (Louizos et al., 2017). We use this insight to formulate a quantizing prior, a continuous relaxation of a "multi-spike-and-slab" prior which has multiple spikes at locations  $c_k$ ,  $k \in \{1..K\}$ . Each spike location corresponds to one target value for subsequent quantization. The quantizing prior allows weights of low variance only at the locations of the quantization target values  $c_k$ . The effect of using such a quantizing prior during Variational Network Quantization is shown in Fig. 1. After training, most weights of low variance are distributed very closely around the quantization target values  $c_k$  and can thus be replaced by the corresponding value without significant loss in accuracy. Weights of large variance can be pruned. Additionally, we typically fix one of the spike locations to zero, e.g.  $c_2 = 0$ , which allows to prune weights with an  $\alpha_{ij}$  threshold (see Eq. (9)) as in sparse Variational Dropout (Molchanov et al., 2017). Following the interpretation of the log uniform prior  $p(w)$  as a marginal over the scale-hyperparameter  $z$ , we extend Eq. (10) with a hyper-prior over locations

$$
p \left(w _ {i j}\right) = \int \mathcal {N} \left(w _ {i j} \mid m, z\right) p _ {z} (z) p _ {m} (m) \mathrm {d} z \mathrm {d} m \quad p _ {m} (m) = \sum_ {k} p _ {k} \delta \left(m - c _ {k}\right), \tag {11}
$$

with  $p(z) \propto |z|^{-1}$ . The location prior  $p_m(m)$  is a mixture of weighted delta distributions located at the quantization values  $c_k$ . Marginalizing over  $m$  yields the quantizing prior

$$
p \left(w _ {i j}\right) \propto \sum_ {k} p _ {k} \int \frac {1}{| z |} \mathcal {N} \left(w _ {i j} \mid c _ {k}, z\right) \mathrm {d} z = \sum_ {k} p _ {k} \frac {1}{\left| w _ {i j} - c _ {k} \right|}. \tag {12}
$$

In our experiments we use  $K = 3$ ,  $p_k = 1 / K$  and  $c_2 = 0$  unless indicated otherwise.

# 3.2 POST-TRAINING QUANTIZATION

Equation (9) implies that using a threshold on  $\alpha_{ij}$  as a pruning criterion is equivalent to pruning weights whose value does not differ significantly from zero:

$$
\theta_ {i j} ^ {2} \leq \frac {\sigma_ {i j} ^ {2}}{T _ {\alpha}} \Leftrightarrow \theta_ {i j} \in \left(- \frac {\sigma_ {i j}}{\sqrt {T _ {\alpha}}}, \frac {\sigma_ {i j}}{\sqrt {T _ {\alpha}}}\right). \tag {13}
$$

To be precise,  $T_{\alpha}$  specifies the width of a scaled standard-deviation band  $\pm \sigma_{ij} / \sqrt{T_{\alpha}}$  around the mean  $\theta_{ij}$ . If the value zero lies within this band, the weight is assigned the value 0. For instance, a pruning threshold which implies  $p \geq 0.95$  corresponds to a variance band of approximately  $\sigma_{ij} / 4$ . An equivalent interpretation is that a weight is pruned if the likelihood for the value 0 under the weight posterior exceeds the threshold given by the standard-deviation band (Eq. (13)):

$$
\mathcal {N} \left(0 \mid \theta_ {i j}, \sigma_ {i j} ^ {2}\right) \geq \mathcal {N} \left(\theta_ {i j} \pm \frac {\sigma_ {i j}}{\sqrt {T _ {\alpha}}} \mid \theta_ {i j}, \sigma_ {i j} ^ {2}\right) = \frac {1}{\sqrt {2 \pi} \sigma_ {i j}} e ^ {- \frac {1}{2 T _ {\alpha}}}. \tag {14}
$$

Following this interpretation we can design a maximum  $a$ -posteriori (MAP) quantization scheme: to each weight we assign the quantized values  $c_k$  with the highest likelihood under the posterior. Since weight posteriors are Gaussian, this translates into minimizing the squared distance between the mean  $\theta_{ij}$  and the quantized values  $c_k$ :

$$
\arg \max  _ {k} \mathcal {N} \left(c _ {k} \mid \theta_ {i j}, \sigma_ {i j} ^ {2}\right) = \arg \max  _ {k} e ^ {- \frac {\left(c _ {k} - \theta_ {i j}\right) ^ {2}}{2 \sigma_ {i j} ^ {2}}} = \arg \min  _ {k} \left(c _ {k} - \theta_ {i j}\right) ^ {2} \tag {15}
$$

Additionally, the pruning rate can be increased by assigning a hard 0 to all weights that exceed the pruning threshold  $T_{\alpha}$  (see Eq. (9)) before performing the MAP assignment to quantize the nonpruned weights described above.

# 3.3 KL DIVERGENCE APPROXIMATION

Under the quantizing prior (Eq. (12)) the KL divergence between the mean-field posterior and prior  $D_{\mathrm{KL}}(q_{\phi}(w)||p(w))$  is analytically intractable. Similar to Kingma et al. (2015); Molchanov et al. (2017) we use a differentiable approximation  $F_{\mathrm{KL}}(\theta, \sigma, c)^3$ , composed of a small number of differentiable functions to keep the computational effort low during training. We now present the approximation for a reference codebook  $c = [-r, 0, r]$ ,  $r = 0.2$ , however later we show how the approximation can be used for arbitrary ternary, symmetric codebooks as well. The basis of our approximation is the approximation  $F_{\mathrm{KL,LU}}$  introduced by Molchanov et al. (2017) for the KL divergence between a log uniform prior and a Gaussian posterior (see Eq. (7)) which is centered around zero. We observe that a weighted mixture of shifted versions of  $F_{\mathrm{KL,LU}}$  can be used to approximate the KL divergence for our multi-modal quantizing prior (Eq. (12)) (which is composed of shifted versions of the log uniform prior). In a nutshell, we shift one version of  $F_{\mathrm{KL}}$  to each codebook entry  $c_k$  and then use  $\theta$ -dependent Gaussian windowing functions  $\Omega(\theta)$  to mix the shifted approximations (see more details in the Appendix A.3). The approximation for the KL divergence between a Gaussian posterior and our multi-modal quantizing prior is given as

$$
F _ {\mathrm {K L}} (\theta , \sigma , c) = \underbrace {\sum_ {k : c _ {k} \neq 0} \Omega \left(\theta - c _ {k}\right) \mathrm {F} _ {\mathrm {K L} , \mathrm {L U}} \left(\theta - c _ {k} , \sigma\right)} _ {\text {l o c a l b e h a v i o r}} + \underbrace {\Omega_ {0} (\theta) \mathrm {F} _ {\mathrm {K L} , \mathrm {L U}} \left(\theta , \sigma\right)} _ {\text {g l o b a l b e h a v i o r}} \tag {16}
$$

with

$$
\Omega (\theta) = \exp \left(- \frac {1}{2} \frac {\theta^ {2}}{\tau^ {2}}\right) \quad \Omega_ {0} (\theta) = 1 - \sum_ {k: c _ {k} \neq 0} \Omega \left(\theta - c _ {k}\right) \tag {17}
$$

We use  $\tau = 0.075$  in our experiments. Illustrations of the approximation, including a comparison against the ground-truth computed via Monte Carlo sampling are shown in Fig. 2. Over the range of  $\theta$ - and  $\sigma$ -values relevant to our method, the maximum absolute deviation from the ground-truth is 1.07. See Fig. 4 in the Appendix for a more detailed quantitative evaluation of our approximation.

![](images/b482fa8ad38568f67f93cc5e4e0b47951ad8250926d92c817941cd03d1e99522.jpg)

![](images/cb3e74fe71f3597cadafd12723f6d4bef5feee202575c64aa9469477a1ee443d.jpg)

![](images/bbbf5382b79b183c5f0bf7f5cb0c047ae3aeb35878871513d405240c314ccc53.jpg)  
Figure 2: Approximation to the analytically intractable KL divergence  $D_{\mathrm{KL}}(q_{\phi}||p)$ , constructed by shifting and mixing known approximations to the KL divergence between the posterior a log uniform prior. Top row: Shifted versions of the known approximation (Eq. (7)) in color and the ground truth KL approximation (computed via Monte Carlo sampling)  $\mathrm{D}_{\mathrm{KL}}^{\mathrm{MC}}(q_{\phi}||p)$  in black. Middle row: weighting functions  $\Omega(\theta)$  that mix the shifted known approximation to form the final approximation  $F_{\mathrm{KL}}$  shown in the bottom row (gold), compared against the ground-truth (MC sampled). Each column corresponds to a different value of  $\sigma$ . A comparison between ground-truth and our approximation over a large range of  $\sigma$  and  $\theta$  values is shown in the Appendix in Fig. 4. Note that since the priors are improper, KL approximation and ground-truth can only be compared up to an additive constant  $C$  - the constant is irrelevant for network training but has been chosen in the plot such that ground-truth and approximation align for large values of  $\theta$ .

![](images/538fdc50debee1a6b7eaeae561588d30f6bd8428f7e51ff27de5365171e3a6af.jpg)

![](images/7d28cf9aa03d2f47aaa7134c6b7b21e7dddd7ed44fc66060190a89b14349453b.jpg)

![](images/669c3e8a6cdf013f172050bee4e0f96fe6096c92762bfd0623476fe0ce9c2385.jpg)  
$\theta$

![](images/43dabecf031e7012981bb347efd4b495a30afeaa637e20e05c431ba4eeb8c659.jpg)

![](images/949983e577b91b0065710fa1103e50a18a9ed030eb33a45cbc09d553ebb1b90b.jpg)

![](images/5c52b9259342b603da421bab53e0a995c77e64e58f43670a37e8c87d32104eb2.jpg)  
$\theta$

This KL approximation in Eq. (16), developed for the reference codebook  $c_{r} = [-r,0,r]$ , can be reused for any symmetric ternary codebook  $c_{a} = [-a,0,a]$ ,  $a\in \mathbb{R}^{+}$ , since  $c_{a}$  can be represented with the reference codebook and a positive scaling factor  $s$ ,  $c_{a} = s c_{r}$ ,  $s = a / r$ . As derived in the Appendix (A.4), this re-scaling translates into a multiplicative re-scaling of the variational parameters  $\theta$  and  $\sigma$ . The KL divergence between the posterior  $q_{\phi}(w)$  and a prior based on the codebook  $c_{a}$  is thus given by  $D_{KL}(q_{\phi}(w)||p_{c_a}(w))\approx F_{\mathrm{KL}}(\theta /s,\sigma /s,c_r)$ . This result allows learning the quantization level  $a$  during training as well.

# 4 EXPERIMENTS

In our experiments we train with VNQ and then first prune via thresholding  $\log \alpha_{ij} \geq \log T_{\alpha} = 3$ . Remaining weights are then quantized by minimizing the squared distance to the quantization values  $c_k$  (corresponding to a MAP quantization, see Sec. 3.2). We use warm-up (Sønderby et al., 2016), that is we multiply the KL divergence term (Eq. (2)) with a factor  $\beta$ , where  $\beta = 0$  during the first few epochs and then linearly ramps up to  $\beta = 1$ . To improve stability of VNQ training we ensure through clipping that  $\log \sigma_{ij}^2 \in (-10, 1)$  and  $\theta_{ij} \in (0.223\sigma \pm -a,)$  (which corresponds to a shifted  $\log \alpha$  threshold of 3, that is we clip  $\theta_{ij}$  if it lies left of the  $-a$  funnel or right of the  $+a$  funnel, compare Fig. 1). When learning codebook values  $a$  during training, we use a lower learning rate for

adjusting the codebook, otherwise we observe a tendency for codebook values to collapse in early stages of training (a similar observation was made by Ullrich et al. (2017)). Additionally, we ensure  $a \leq 0.05$  by clipping.

# 4.1 LENET-5 ON MNIST

We demonstrate our method with LeNet-5 $^4$  (LeCun et al., 1998) on the MNIST handwritten digits dataset. Images are pre-processed by subtracting the mean and dividing by the standard-deviation over the training set. For the pre-trained network we run 5 epochs on a randomly initialized network (Glorot initialization, Adam optimizer), which leads to a validation accuracy of  $99.2\%$ . We initialize means  $\theta$  with the pre-trained weights and variances with  $\log \sigma^2 = -8$ . The warm-up factor  $\beta$  is linearly increased from 0 to 1 during the first 15 epochs. VNQ training runs for a total of 195 epochs with a batch-size of 128, the learning rate is linearly decreased from 0.001 to 0 and the learning rate for adjusting the codebook parameter  $a$  uses a learning rate that is 100 times lower. Results are shown in Table 1, a visualization of the distribution over weights after VNQ training is shown in Fig. 1.

Table 1: Results on LeNet-5 (MNIST), showing the error on the validation set, the percentage of non-pruned weights and the bit-precision per parameter. Original is our pre-trained LeNet-5. We show results after VNQ (no P&Q) where weights were deterministically replaced by the (full-precision) means  $\theta$  and for VNQ with subsequent pruning and quantization. We also show results of non-ternary or pruning-only methods: Deep Compression (Han et al., 2016), Soft weight-sharing (Ullrich et al., 2017), Sparse VD (Molchanov et al., 2017), Bayesian Compression (Louizos et al., 2017) and Stuctured Bayesian Pruning (Neklyudov et al., 2017).

<table><tr><td>Method</td><td>val. error [%]</td><td>|w≠0| [w]</td><td>bits</td></tr><tr><td>Original</td><td>0.8</td><td>100</td><td>32</td></tr><tr><td>VNQ (no P&amp;Q)</td><td>0.69</td><td>100</td><td>32</td></tr><tr><td>VNQ + P&amp;Q</td><td>0.83</td><td>24.5</td><td>2</td></tr><tr><td>Deep Compression (P&amp;Q)</td><td>0.74</td><td>8</td><td>10 - 13</td></tr><tr><td>Soft weight-sharing (P&amp;Q)</td><td>0.97</td><td>3</td><td>3</td></tr><tr><td>Sparse VD (P)</td><td>0.75</td><td>0.7</td><td>-</td></tr><tr><td>Bayesian Comp. (P)</td><td>1.0</td><td>0.6</td><td>7 - 18</td></tr><tr><td>Structured BP (P)</td><td>0.86</td><td>-</td><td>-</td></tr></table>

We find that VNQ training sufficiently prepares a network for pruning and quantization with negligible loss in accuracy and without requiring subsequent fine-tuning. Compared to pruning methods that do not consider few-bit quantization in their objective, we achieve significantly lower pruning rates. This is an interesting observation since our method is based on a similar objective (e.g. compared to Sparse VD) but with the addition of forcing non-pruned weights to tightly cluster around the quantization levels. Few-bit quantization severely limits network capacity. Perhaps this capacity limitation must be countered by pruning fewer weights. Our pruning rates are roughly in line with other papers on ternary quantization, e.g. Zhu et al. (2016), who report sparsity levels between  $30\%$  and  $50\%$  with their ternary quantization method. Note that a direct comparison between pruning, quantizing and ternarizing methods is difficult and depends on many factors such that a fair computation of the compression rate that does not implicitly favor certain methods is hardly possible within the scope of this paper. For instance, compression rates for pruning methods are typically reported under the assumption of a CSC storage format which would not fully account for the compression potential of a sparse ternary matrix. We thus chose not to report any measures for compression rates, however for the methods listed in Table 1, they can easily be found in the literature.

# 4.2 DENSENET-121 ON CIFAR-10

Our second experiment uses a modern DenseNet-121 Huang et al. (2017) ( $k = 12$ , with bottleneck) on CIFAR-10 (Krizhevsky & Hinton, 2009). The training procedure is identical to the procedure on MNIST with the following exceptions: we use a batch-size of 64 epochs, warmup is linearly ramped

up from 0 to 1 over the first 20 epochs, the learning rate of 0.005 is kept constant for the first 50 epochs and then linearly decreased until training stops at epoch 300. Results are shown from epoch 150. We pre-train a deterministic DenseNet (reaching validation accuracy of  $93.19\%$ ) to initialize VNQ training. Results are shown in Table 2. A visualization of the distribution over weights after VNQ training is shown in the Appendix Fig. 3.

Table 2: Results on DenseNet-121 (CIFAR-10), showing the error on the validation set, the percentage of non-pruned weights and the bit-precision per parameter. Original denotes the pre-trained network. We show results after VNQ training without pruning and quantization (weights were deterministically replaced by the (full-precision) means  $\theta$ ), and VNQ with subsequent pruning and quantization (in the condition (w/o 1) we use full-precision means for the weights in the first layer and do not prune and quantize this layer).

<table><tr><td>Method</td><td>val error [%]</td><td>|w≠0| / |w| [%]</td><td>bits</td></tr><tr><td>Original</td><td>6.81</td><td>100</td><td>32</td></tr><tr><td>VNQ (no P&amp;Q)</td><td>8.45</td><td>100</td><td>32</td></tr><tr><td>VNQ + P&amp;Q (w/o 1)</td><td>8.52</td><td>55</td><td>2 (32)</td></tr><tr><td>VNQ + P&amp;Q</td><td>10.92</td><td>55</td><td>2</td></tr></table>

We generally observe lower levels of sparsity for DenseNet, compared to LeNet. This might be due to the fact that DenseNet already has an optimized architecture which removed a lot of redundant parameters from the start. In line with previous publications we find that the first and last layer of the network are most sensitive to pruning and quantization. However, in contrast to many other methods that do not quantize these layers (e.g. Zhu et al. (2016)), we find that after sufficient training, the final layer can be pruned and quantized without loss in accuracy and the first layer can also be pruned and quantized with a small loss in accuracy (see Table 2).

# 5 RELATED WORK

Our method is an extension of Sparse VD (Molchanov et al., 2017), originally used for network pruning. In contrast we use a quantizing prior, leading multi-modal posterior suitable for few-bit quantization and pruning. Bayesian Compression and Structured Bayesian Pruning Louizos et al. (2017); Neklyudov et al. (2017) extend Sparse VD to prune whole neurons or filters via group-sparsity constraints. Additionally, in Louizos et al. (2017) the required bit-precision per layer is determined via posterior uncertainty. In contrast to our method, Bayesian Compression does not explicitly enforce clustering of weights during training and thus requires bit-widths in the range between 5 and 18 bits. Extending our method to include group-constraints for pruning is an interesting direction for future work. Another Bayesian method for simultaneous network quantization and pruning is soft weight-sharing (SWS) Ullrich et al. (2017), which uses a Gaussian mixture model prior (and a KL term without trainable parameters such that the KL term reduces to the prior entropy). SWS acts like a probabilistic version of k-means clustering with the advantage of automatic collapse of unnecessary mixture components. Similar to learning the codebooks in our method, soft weight-sharing learns the prior from the data, a technique known as empirical Bayes (see also ARD (Karaletsos & Ratsch, 2015)). We cannot directly compare against soft weight-sharing since the authors do not report results on ternary networks. Gal et al. (2017) learn dropout rates by using a continuous relaxation of dropout's discrete masks (via the concrete distribution). The authors learn layer-wise dropout rates, which do not allow for dropout-rate-based pruning. We have experimented with using the concrete distribution for learning codebooks for quantization with promising early results but we have generally observed lower pruning rates or lower accuracy compared to VNQ. A non-probabilistic state-of-the-art method for network ternarization is Trained Ternary Quantization Zhu et al. (2016) which uses full-precision shadow weights during training, but quantized forward passes. Additionally it learns a (non-symmetric) scaling per layer for the non-zero quantization values, similar to our learned quantization level  $a$ . While the method achieves impressive accuracy, the sparsity and thus pruning rates are rather low (between  $50\%$  and  $70\%$  sparsity) and the first and last layer need to be kept with full precision.

# 6 DISCUSSION

A potential shortcoming of our method is the KL divergence approximation (Sec. 3.3). While the approximation is reasonably good on the relevant range of  $\theta$ - and  $\sigma$ -values, there is still room for improvement which could have the benefit that weights are drawn even more tightly onto the quantization levels, resulting in lower accuracy loss after quantization and pruning. Since the KL approximation only needs to be computed once and an arbitrary amount of ground-truth data can be produced, it should be possible to improve upon the approximation presented here, at least by some brute-force function approximation, e.g. a neural network, polynomial or kernel regression. The main difficulty is that the resulting approximation must be differentiable and must not introduce significant computational overhead since the approximation is evaluated once for each network parameter in each gradient step.

Compared to similar methods that only consider network pruning, our pruning rates are significantly lower. This does not seem to be a particular problem of our method since other papers on network ternarization report similar sparsity levels (Li et al. (2016) roughly achieve between  $30\%$  and  $50\%$  sparsity). The reason for this might be that heavily quantized networks have much lower capacity compared to full-precision networks. This limited capacity might require that the network compensates by effectively using more weights such that the pruning rates become significantly lower. Similar trends have also been observed with binary networks, where drops in accuracy could be prevented by increasing the number of neurons (with binary weights) per layer. Principled experiments to test the trade-off between low bit-precision and sparsity rates would be an interesting direction for future work. One starting point could be to test our method with more quantization levels (e.g. 5, 7 or 9) and investigate how this affects the pruning rate.

# REFERENCES

David F Andrews and Colin L Mallows. Scale mixtures of normal distributions. Journal of the Royal Statistical Society. Series B (Methodological), pp. 99-102, 1974.  
Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural networks. arXiv preprint arXiv:1505.05424, 2015.  
Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized neural networks: Training deep neural networks with weights and activations constrained to+ 1 or-1. arXiv preprint arXiv:1602.02830, 2016.  
Thomas M Cover and Joy A Thomas. Elements of information theory. John Wiley & Sons, 2006.  
Misha Denil, Babak Shakibi, Laurent Dinh, Nando de Freitas, et al. Predicting parameters in deep learning. In Advances in Neural Information Processing Systems, pp. 2148-2156, 2013.  
Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, and Steffen Udluft. Learning and policy search in stochastic dynamical systems with bayesian neural networks. arXiv preprint arXiv:1605.07127, 2016.  
Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, and Steffen Udluft. Uncertainty decomposition in bayesian neural networks with latent variables. arXiv preprint arXiv:1706.08495, 2017.  
Mário Figueiredo. Adaptive sparseness using jeffreys prior. In Advances in neural information processing systems, pp. 697-704, 2002.  
Yarin Gal. Uncertainty in deep learning. PhD thesis, PhD thesis, University of Cambridge, 2016.  
Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. arXiv:1506.02142, 2015.  
Yarin Gal, Jiri Hron, and Alex Kendall. Concrete dropout. arXiv preprint arXiv:1705.07832, 2017.  
Tim Genewein and Daniel A Braun. Occam's razor in sensorimotor learning. Proceedings of the Royal Society of London B: Biological Sciences, 281(1783):20132952, 2014.

Tim Genewein, Felix Leibfried, Jordi Grau-Moya, and Daniel Alexander Braun. Bounded rationality, abstraction, and hierarchical decision-making: An information-theoretic optimality principle. Frontiers in Robotics and AI, 2:27, 2015.  
Alex Graves. Practical variational inference for neural networks. In Advances in Neural Information Processing Systems, pp. 2348-2356, 2011.  
Peter D Grünwald. The minimum description length principle. MIT press, 2007.  
Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. In Advances In Neural Information Processing Systems, pp. 1379-1387, 2016.  
Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. *ICLR* 2016, 2016.  
Song Han, Jeff Pool, Sharan Narang, Huizi Mao, Shijian Tang, Erich Elsen, Bryan Catanzaro, John Tran, and William J Dally. Dsd: Regularizing deep neural networks with dense-sparse-dense training flow. ICLR 2017, 2017.  
Babak Hassibi and David G Stork. Second order derivatives for network pruning: Optimal brain surgeon. In Advances in Neural Information Processing Systems, pp. 164-171, 1993.  
Jose Miguel Hernandez-Lobato and Ryan Adams. Probabilistic backpropagation for scalable learning of bayesian neural networks. In International Conference on Machine Learning, pp. 1861-1869, 2015.  
Geoffrey E Hinton and Drew Van Camp. Keeping the neural networks simple by minimizing the description length of the weights. In Proceedings of the sixth annual conference on Computational learning theory, pp. 5-13. ACM, 1993.  
Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.  
Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. 2017.  
Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Quantized neural networks: Training neural networks with low precision weights and activations. arXiv preprint arXiv:1609.07061, 2016.  
Forrest N Iandola, Song Han, Matthew W Moskewicz, Khalid Ashraf, William J Dally, and Kurt Keutzer. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and  $10.5\mathrm{mb}$  model size. arXiv preprint arXiv:1602.07360, 2016.  
John Ingraham and Debora Marks. Variational inference for sparse and undirected models. In International Conference on Machine Learning, pp. 1607-1616, 2017.  
Theofanis Karaletsos and Gunnar Ratsch. Automatic relevance determination for deep generative models. arXiv preprint arXiv:1505.07765, 2015.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. *ICLR*, 2014.  
Diederik P Kingma, Tim Salimans, and Max Welling. Variational dropout and the local reparameterization trick. In Advances in Neural Information Processing Systems, pp. 2575-2583, 2015.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.  
Yann LeCun, John S. Denker, and Sara A. Solla. Optimal brain damage. In D. S. Touretzky (ed.), Advances in Neural Information Processing Systems, pp. 598-605. 1990.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.

Fengfu Li, Bo Zhang, and Bin Liu. Ternary weight networks. arXiv preprint arXiv:1605.04711, 2016.  
Christos Louizos, Karen Ullrich, and Max Welling. Bayesian compression for deep learning. Advances in Neural Information Processing Systems, 2017.  
David JC MacKay. Probable networks and plausible predictions - a review of practical bayesian methods for supervised neural networks. Network: Computation in Neural Systems, 6(3):469-505, 1995.  
David JC MacKay. Information theory, inference and learning algorithms. Cambridge university press, 2003.  
Toby J Mitchell and John J Beauchamp. Bayesian variable selection in linear regression. Journal of the American Statistical Association, 83(404):1023-1032, 1988.  
Dmitry Molchanov, Arsenii Ashukha, and Dmitry Vetrov. Variational dropout sparsifies deep neural networks. ICML 2017, 2017.  
Radford M Neal. *Bayesian Learning for Neural Networks*. PhD thesis, PhD thesis, University of Toronto, 1995.  
Kirill Neklyudov, Dmitry Molchanov, Armenii Ashukha, and Dmitry Vetrov. Structured bayesian pruning via log-normal multiplicative noise. arXiv preprint arXiv:1705.07283, 2017.  
Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. Xnor-net: Imagenet classification using binary convolutional neural networks. In European Conference on Computer Vision, pp. 525-542. Springer, 2016.  
Jorma Rissanen. Modeling by shortest data description. Automatica, 14(5):465-471, 1978.  
Casper Kaae Sønderby, Tapani Raiko, Lars Maaløe, Søren Kaae Sønderby, and Ole Winther. How to train deep variational autoencoders and probabilistic ladder networks. arXiv preprint arXiv:1602.02282, 2016.  
Daniel Soudry, Itay Hubara, and Ron Meir. Expectation backpropagation: Parameter-free training of multilayer neural networks with continuous or discrete weights. In Advances in Neural Information Processing Systems, pp. 963-971, 2014.  
Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of machine learning research, 15(1):1929-1958, 2014.  
Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, and Joel Emer. Efficient processing of deep neural networks: A tutorial and survey. arXiv preprint arXiv:1703.09039, 2017.  
Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000.  
Karen Ullrich, Edward Meeds, and Max Welling. Soft weight-sharing for neural network compression. *ICLR* 2017, 2017.  
Sida Wang and Christopher Manning. Fast dropout training. In Proceedings of the 30th International Conference on Machine Learning (ICML-13), pp. 118-126, 2013.  
Max Welling and Yee W Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th International Conference on Machine Learning (ICML-11), pp. 681-688, 2011.  
Shuchang Zhou, Yuxin Wu, Zekun Ni, Xinyu Zhou, He Wen, and Yuheng Zou. Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients. arXiv preprint arXiv:1606.06160, 2016.  
Chenzhuo Zhu, Song Han, Huizi Mao, and William J Dally. Trained ternary quantization. arXiv preprint arXiv:1612.01064, 2016.
