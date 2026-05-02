# DISENTANGLING LEARNING REPRESENTATIONS WITH DENSITY ESTIMATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Disentangled learning representations have promising utility in many applications, but they currently suffer from serious reliability issues. We present Gaussian Channel Autoencoder (GCAE), a method which achieves reliable disentanglement via scalable non-parametric density estimation of the latent space. GCAE avoids the curse of dimensionality of density estimation by disentangling subsets of its latent space with the Dual Total Correlation (DTC) metric, thereby representing its high-dimensional latent joint distribution as a collection of many low-dimensional conditional distributions. In our experiments, GCAE achieves highly competitive and reliable disentanglement scores compared with state-of-the-art baselines.

# 1 INTRODUCTION

The notion of disentangled learning representations was introduced by Bengio et al. (2013) - it is meant to be a robust approach to feature learning when trying to learn more about a distribution of data  $X$  or when downstream tasks for learned features are unknown. Since then, disentangled learning representations have been proven to be extremely useful in the applications of natural language processing Jain et al. (2018), content and style separation John et al. (2018), drug discovery Polykovskiy et al. (2018); Du et al. (2020), fairness Sarhan et al. (2020), and more.

Density estimation of learned representations is an important ingredient to competitive disentangle-ment methods. Bengio et al. (2013) state that representations  $\mathbf{z} \sim Z$  which are disentangled should maintain as much information of the input as possible while having components which are mutually invariant to one another. Mutual invariance motivates seeking representations of  $Z$  which have independent components extracted from the data, necessitating some notion of  $p(\mathbf{z})$ .

Leading unsupervised disentanglement methods, namely  $\beta$ -VAE Higgins et al. (2016), FactorVAE Kim & Mnih (2018), and  $\beta$ -TCVAE Chen et al. (2018) all learn  $p(\mathbf{z})$  via the same variational Bayesian framework Kingma & Welling (2013), but they approach making  $p(\mathbf{z})$  independent with different angles.  $\beta$ -VAE indirectly promotes independence in  $p(\mathbf{z})$  via enforcing low  $D_{\mathrm{KL}}$  between the representation and a factorized Gaussian prior,  $\beta$ -TCVAE encourages representations to have low Total Correlation (TC) via an ELBO decomposition and importance weighted sampling technique, and FactorVAE reduces TC with help from a monolithic neural network estimate. Other well-known unsupervised methods are Annealed  $\beta$ -VAE Burgess et al. (2018), which imposes careful relaxation of the information bottleneck through the VAE  $D_{\mathrm{KL}}$  term during training, and DIP-VAE I & II Kumar et al. (2017), which directly regularize the covariance of the learned representation.

While these VAE-based disentanglement methods have been the most successful in the field, Locatello et al. (2019) point out serious reliability issues shared by all. In particular, increasing disentanglement pressure during training doesn't tend to lead to more independent representations, there currently aren't good unsupervised indicators of disentanglement, and no method consistently dominates the others across all datasets. Locatello et al. (2019) stress the need to find the right inductive biases in order for unsupervised disentanglement to truly deliver.

We seek to make disentanglement more reliable and high-performing by incorporating new inductive biases into our proposed method, Gaussian Channel Autoencoder (GCAE). We shall explain them in more detail in the following sections, but to summarize: GCAE avoids the challenge of representing high-dimensional  $p(\mathbf{z})$  via disentanglement with Dual Total Correlation (rather than TC) and the

DTC criterion is augmented with a scale-dependent latent variable arbitration mechanism. This work makes the following contributions:

- Analysis of the TC and DTC metrics with regard to the curse of dimensionality which motivates use of DTC and a new feature-stabilizing arbitration mechanism  
- GCAE, a new form of noisy autoencoder (AE) inspired by the Gaussian Channel problem, which permits application of non-parametric density estimation methods in the latent space  
- Experiments which demonstrate competitive performance of GCAE against leading disentanglement baselines on multiple datasets using existing metrics

# 2 BACKGROUND AND INITIAL FINDINGS

To estimate  $p(\mathbf{z})$ , we introduce a discriminator-based method which applies the density-ratio trick and the Radon-Nikodym theorem to estimate density of samples from an unknown distribution. We shall demonstrate in this section the curse of dimensionality in density estimation and the necessity for representing  $p(\mathbf{z})$  as a collection of conditional distributions.

The optimal discriminator neural network introduced by Goodfellow et al. (2014a) satisfies:

$$
\underset {D} {\arg \max } \mathbb {E} _ {\mathbf {x} _ {r} \sim X _ {r e a l}} \left[ \log D (\mathbf {x}) \right] + \mathbb {E} _ {\mathbf {x} _ {f} \sim X _ {f a k e}} \left[ \log \left(1 - D (\mathbf {x} _ {f})\right) \right] \triangleq D ^ {*} (\mathbf {x}) = \frac {p _ {r e a l} (\mathbf {x})}{p _ {r e a l} (\mathbf {x}) + p _ {f a k e} (\mathbf {x})}
$$

where  $D(\mathbf{x})$  is a discriminator network trained to differentiate between "real" samples  $\mathbf{x}_r$  and "fake" samples  $\mathbf{x}_f$ . Given the optimal discriminator  $D^{*}(\mathbf{x})$ , the density-ratio trick can be applied to yield  $\frac{p_{real}(\mathbf{x})}{p_{fake}(\mathbf{x})} = \frac{D^{*}(\mathbf{x})}{1 - D^{*}(\mathbf{x})}$ . Furthermore, the discriminator can be supplied conditioning variables to represent a ratio of conditional distributions Goodfellow et al. (2014b); Makhzani et al. (2015).

Consider the case where the "real" samples come from an unknown distribution  $\mathbf{z} \sim Z$  and the "fake" samples come from a known distribution  $\mathbf{u} \sim U$ . Permitted that both  $p(\mathbf{z})$  and  $p(\mathbf{u})$  are finite and  $p(\mathbf{u})$  is nonzero on the sample space of  $p(\mathbf{z})$ , the optimal discriminator can be used to retrieve the unknown density  $p_Z(z) = \frac{D^*(z)}{1 - D^*(z)} p_U(z)$ . In the special case where  $\mathbf{u}$  is a uniformly distributed variable, this "transfer" of density through the optimal discriminator can be seen as an application of the Radon-Nikodym derivative of  $\mathbf{z}$  with reference to the Lebesgue measure. Throughout the rest of this work, we employ discriminators with uniform noise and the density-ratio trick in this way to recover unknown distributions.

![](images/e461cc072a12d237b5b072789d61bfef21cc7837ae579916ffd0b750e84191bb.jpg)  
(a) Joint distributions

![](images/11c47188661c8e9465fe0ca1f8821b2b9c5d237a20e56d1b909ac14fc43d2496.jpg)  
Figure 1: KL Divergence between the true and estimated distributions as training iteration and distribution dimensionality increase. Training parameters are kept the same between both experiments.  
(b) Conditional distributions

This technique can be employed to recover the probability density of an  $m$ -dimensional isotropic Gaussian distribution. While it works well in low dimensions ( $m \leq 8$ ), the method inevitably fails

as  $m$  increases. Figure 1a depicts several experiments of increasing  $m$  in which the KL-divergence of the true and estimated distributions are plotted with training iteration. When number of data samples is finite and the dimension  $m$  exceeds a certain threshold, the probability of there being any uniform samples in the neighborhood of the Gaussian samples swiftly approaches zero, causing the density-ratio trick to fail.

This is a well-known phenomenon called the curse of dimensionality of non-parametric density estimation. In essence, as the dimensionality of a joint distribution increases, concentrated joint data quickly become isolated within an extremely large space. The limit  $m \leq 8$  is consistent with the limits of other methods such as kernel density estimation (Parzen-Rosenblatt window).

Fortunately, the same limitation does not apply to conditional distributions of many jointly distributed variables. Figure 1b depicts a similar experiment of the first in which  $m - 1$  variables are independent Gaussian distributed, but the last variable  $\mathbf{z}_m$  follows the distribution  $\mathbf{z}_m \sim \mathcal{N}(\mu = (m - 1)^{-\frac{1}{2}} \sum_{i=1}^{m-1} \mathbf{z}_i$ ,  $\sigma^2 = \frac{1}{m}$ ) (i.e., the last variable is Gaussian distributed with its mean as the sum of observations of the other variables). The marginal distribution of each component is Gaussian, just like the previous example. While it takes more training iterations to bring the KL-divergence between the true and estimated conditional distribution to zero, it is not limited by the curse of dimensionality. Hence, we assert that conditional distributions can be used to capture complex relationships between subsets of many jointly distributed variables while avoiding the curse of dimensionality.

# 3 METHODOLOGY

# ANALYSIS OF DUAL TOTAL CORRELATION

Recent works encourage disentanglement of the latent space by enhancing the Total Correlation (TC) either indirectly Higgins et al. (2016); Kumar et al. (2017) or explicitly Kim & Mnih (2018); Chen et al. (2018). TC is a metric of multivariate statistical independence that is strictly non-negative and zero if and only if all elements of  $\mathbf{z}$  are independent.

$$
\operatorname {T C} (\mathbf {z}) = \mathbb {E} _ {\mathbf {z}} \log \frac {p (\mathbf {z})}{\prod_ {i} p (\mathbf {z} _ {i})} = \sum_ {i} h (\mathbf {z} _ {i}) - h (\mathbf {z})
$$

Locatello et al. (2019) evaluate many TC-based methods and conclude that minimizing their measures of TC during training often does not lead to VAE  $\mu$  (used for representation) with low TC. We note that computing  $\mathrm{TC}(\mathbf{z})$  requires knowledge of the joint distribution  $p(\mathbf{z})$ , which can be very challenging to model in high dimensions. We hypothesize that the need for a model of  $p(\mathbf{z})$  is what leads to the observed reliability issues of these TC-based methods.

Consider another metric for multivariate statistical independence, Dual Total Correlation (DTC). Like TC, DTC is strictly non-negative and zero if and only if all elements of  $\mathbf{z}$  are independent.

$$
\operatorname {D T C} (\mathbf {z}) = \mathbb {E} _ {\mathbf {z}} \log \frac {\prod_ {i} p (\mathbf {z} _ {i} | \mathbf {z} _ {\forall j \neq i})}{p (\mathbf {z})} = h (\mathbf {z}) - \sum_ {i} h (\mathbf {z} _ {i} | \mathbf {z} _ {\forall j \neq i})
$$

At face value, it appears that  $\mathrm{DTC}(\mathbf{z})$  also requires knowledge of the joint density  $p(\mathbf{z})$ . However, observe an equivalent form of DTC manipulated for the  $i$ -th variable:

$$
\mathrm {D T C} (\mathbf {z}) = \mathrm {D T C} _ {i} (\mathbf {z}) \triangleq h (\mathbf {z}) - h \left(\mathbf {z} _ {i} \mid \mathbf {z} _ {j \neq i}\right) - \sum_ {j \neq i} h \left(\mathbf {z} _ {j} \mid \mathbf {z} _ {k \neq j}\right) = h \left(\mathbf {z} _ {j \neq i}\right) - \sum_ {j \neq i} h \left(\mathbf {z} _ {j} \mid \mathbf {z} _ {k \neq j}\right). \tag {1}
$$

Here, the  $i$ -th variable only contributes to DTC through each set of conditioning variables  $\mathbf{z}_{\forall k \neq i}$ . Hence, when computing the derivative  $\frac{\partial \mathrm{DTC}_i(\mathbf{z})}{\partial \mathbf{z}_i}$ , no representation of  $p(\mathbf{z})$  is required - only the conditional entropies  $h(\mathbf{z}_j | \mathbf{z}_{k \neq j})$  are necessary. Hence, we observe that the curse of dimensionality can be avoided through gradient descent on the DTC metric, making it more attractive for disentanglement. However, while one only needs the conditional entropies to compute gradient for DTC,

the conditional entropies alone don't tell one how close  $\mathbf{z}$  is to having independent elements. To overcome this, we define the summed information loss  $\mathcal{L}_{\Sigma I}$ :

$$
\mathcal {L} _ {\Sigma I} \triangleq \sum_ {i} I (\mathbf {z} _ {i}; \mathbf {z} _ {\forall j \neq i}) = \left[ \sum_ {i} h (\mathbf {z} _ {i}) - h (\mathbf {z} _ {i} | \mathbf {z} _ {j \neq i}) \right] + h (\mathbf {z}) - h (\mathbf {z}) = \operatorname {T C} (\mathbf {z}) + \operatorname {D T C} (\mathbf {z}). \tag {2}
$$

If gradients of each  $I(\mathbf{z}_i;\mathbf{z}_j)$  are taken only with respect to  $\mathbf{z}_j$ , then the gradients are equal to  $\frac{\partial\mathrm{DTC}(\mathbf{z})}{\partial\mathbf{z}}$ , avoiding use of any derivatives of estimates of  $p(\mathbf{z})$ . Furthermore, minimizing one metric is equivalent to minimizing the other:  $\mathrm{DTC}(\mathbf{z}) = 0\Leftrightarrow \mathrm{TC}(\mathbf{z}) = 0\Leftrightarrow \mathcal{L}_{\Sigma I} = 0$ . In our experiments, we estimate  $h(\mathbf{z}_i)$  with batch estimates  $\mathbb{E}_{\mathbf{z}_j}p(\mathbf{z}_i|\mathbf{z}_j)$ , requiring no further hyperparameters. Details on how we implement the information functional are available in appendix A.1.

# EXCESS ENTROPY POWER LOSS

We found it very helpful to "stabilize" disentangled features by attaching a feature-scale dependent term to each  $I(\mathbf{z}_i;\mathbf{z}_{\forall j\neq i})$ . The entropy power of a latent variable  $\mathbf{z}_i$  is strictly positive and grows analogously with the variance of  $\mathbf{z}_i$ . Hence, we define the excess entropy power loss:

$$
\mathcal {L} _ {\mathrm {E E P}} (\mathbf {z}) \triangleq \frac {1}{2 \pi e} \sum_ {i} \left[ I \left(\mathbf {z} _ {i}; \mathbf {z} _ {\forall j \neq i}\right) \cdot e ^ {2 h \left(\mathbf {z} _ {i}\right)} \right], \tag {3}
$$

which weighs each component of the  $\mathcal{L}_{\Sigma I}$  loss with the marginal entropy power of each  $i$ -th latent variable. While  $\mathcal{L}_{\mathrm{EEP}}$  has biased gradient with respect to descending  $\mathcal{L}_{\Sigma I}$ , this inductive bias has been extremely helpful in consistently yielding high disentanglement scores. Partial derivatives are taken with respect to the  $\mathbf{z}_{\forall j \neq i}$  subset only, so the marginal entropy power only weighs each component. An ablation study with  $\mathcal{L}_{\mathrm{EEP}}$  can be found in appendix C. The name "Excess Entropy Power" is inspired by DTC's alternative name, excess entropy.

# GAUSSIAN CHANNEL AUTOENCODER

![](images/cb1c4a7a1c2afa856988f3124495ce64b2c6d0d21d87ba78cb3175e0f4e259c4.jpg)  
Figure 2: Depiction of the proposed method, GCAE. Gaussian noise with variance  $\sigma^2$  is added to the latent space, smoothing the representations for gradient-based disentanglement with  $\mathcal{L}_{\mathrm{EEP}}$ . Discriminators use the density-ratio trick to represent the conditional distributions of each latent element given observations of all other elements, capturing complex dependencies between subsets of the variables whilst avoiding the curse of dimensionality.

We begin with a Gaussian Channel Autoencoder (GCAE), composed of a coupled encoder  $\phi : X \to Z_{\phi}$  and decoder  $\psi : Z \to \hat{X}$ , to extract a compressed, noise-resistant representation of the data  $X \in \mathbb{R}^n$  in the GCAE latent space  $Z \in \mathbb{R}^m$ . We assume  $m \ll n$ , as is typical with autoencoder models. The output of the encoder has a scaled softsign activation function, restricting  $Z_{\phi} \in (-3, 3)^m$ . The

latent space is subjected to Gaussian noise of the form  $Z = Z_{\phi} + \nu_{\sigma}$ , where each  $\nu_{\sigma} \sim \mathcal{N}(0, \sigma^2 I)$  and  $\sigma$  is a controllable hyperparameter. The Gaussian noise has the effect of "smoothing" the latent space, ensuring that  $p(\mathbf{z})$  is continuous and finite with respect to the uniform density (Lebesgue measure). This guarantees the existence of the Radon-Nikodym derivative with reference to uniform samples  $\mathbf{u} \sim \mathrm{Unif}(-4, 4)$ , forming the backbone of our disentanglement method.

The loss function for training GCAE is:

$$
\mathcal {L} _ {\mathrm {G C A E}} = \mathbb {E} _ {\mathbf {x}, \nu_ {\sigma}} \left[ \frac {1}{n} \| \hat {\mathbf {x}} - \mathbf {x} \| _ {2} ^ {2} + \lambda \mathcal {L} _ {\mathrm {E E P}} (\mathbf {z}) \right], \tag {4}
$$

where  $\lambda$  is a hyperparameter to control the strength of regularization, and  $\nu_{\sigma}$  is the Gaussian noise injected in the latent space with the scale hyperparameter  $\sigma$ . The two terms have the following intuitions: the mean squared error (MSE) of reconstructions ensures  $\mathbf{z}$  captures information of the input while  $\mathcal{L}_{\mathrm{EEP}}$  encourages representations to be mutually independent.

# 4 EXPERIMENTS

We evaluate the performance of GCAE against the leading disentanglement baselines  $\beta$ -VAE Higgins et al. (2016), FactorVAE Kim & Mnih (2018), and  $\beta$ -TCVAE Chen et al. (2018). In this study we focus on the Mutual Information Gap (MIG) metric Chen et al. (2018) for two reasons: it is one of the few metrics which does not rely on the training of an auxiliary classifier (which can introduce spurious score variance), and its definition aligns most strongly with the original definition of disentangled learning representations Bengio et al. (2013). MIG is defined as follows:

$$
\operatorname {M I G} (\mathbf {z}) \triangleq \frac {1}{K} \sum_ {k = 1} ^ {K} \frac {1}{H (\mathbf {v} _ {k})} \left(I (\mathbf {z} _ {a}; \mathbf {v} _ {k}) - I (\mathbf {z} _ {b}; \mathbf {v} _ {k})\right),
$$

where  $K$  is the number of data generating factors,  $H(\mathbf{v}_k)$  is the discrete entropy of the  $k$ -th data generating factor, and  $\mathbf{z}_a$  and  $\mathbf{z}_b$  (where  $a \neq b$ ) are the latent elements which share the most and next-most information with  $\mathbf{v}_k$ , respectively. In essence, MIG measures the average gap in information between the latent feature which is most selective for a unique data generating factor and the latent feature which is second runner up. MIG is a normalized metric on  $[0,1]$ , and higher scores indicate better capturing and disentanglement of the data generating factors. This metric aligns most closely with the original definition of disentanglement in that it rewards representations which are maximally invariant to one another in the data whilst discarding as little information of the data as is practical Bengio et al. (2013).

We consider two datasets which cover different data modalities. The Beamsynthesis dataset Yeats et al. (2022) is a collection of 360 timeseries data from a linear particle accelerator beamforming simulation. The waveforms are 1000 values in total and are made of two independent data generating factors: duty cycle (continuous) and frequency (categorical). The dSprites dataset Higgins et al. (2016) is a collection of 737280 synthetic images of simple white shapes on a black background. Each  $64 \times 64$  image consists of a single shape generated from the following independent factors: shape (categorical), scale (continuous), orientation (continuous),  $x$ -position (continuous), and  $y$ -position (continuous).

All experiments are run using the PyTorch framework Paszke et al. (2019) using 5 GPUs, and all methods are trained with the same number of iterations. Hyperparameters such as model architecture and optimizer are held constant across all models in each experiment (with the exception of the dual latent parameters required by VAE models). Latent space dimension is fixed at  $m = 10$  for all experiments. More details on experiment setup can be found in appendix B.

In general, increasing  $\lambda$  and  $\sigma$  led to lower  $\mathcal{L}_{\Sigma I}$  but higher MSE at the end of training. Figure 3a depicts this relationship for Beamsynthesis and dSprites. Increasing  $\sigma$  tends to shift ending loss values towards increased independence (according to  $\mathcal{L}_{\Sigma I}$ ) but slightly worse reconstruction error. This is consistent with the well studied theory of the Gaussian channel - as the relative noise level increases, the information capacity of a constrained channel decreases. The tightly grouped samples

![](images/311915214376e6d9355b1a15dee45bef35d560edd5fa25ca10261ddd2148eb13.jpg)  
(a) Scatter plot of  $\log (\mathcal{L}_{\Sigma I})$  vs. MSE for GCAE with  $\sigma = \{0.1, 0.2, 0.3\}$  on Beamsynthesis and dSprites. Higher  $\sigma$  and lower  $\log (\mathcal{L}_{\Sigma I})$  (through increased disentanglement pressure) tend to increase MSE. However, the increase in MSE subsides as the model becomes disentangled.

![](images/6bc572d9827222e7f087ffba1da7fd1d98c25cc2a4ff4cb37be38da785eb28f4.jpg)  
(b) Scatter plot of  $\log (\mathcal{L}_{\Sigma I})$  vs. MIG for GCAE with  $\sigma = \{0.1, 0.2, 0.3\}$  on Beamsynthesis and dSprites. There is a moderate negative relationship between  $\log (\mathcal{L}_{\Sigma I})$  and MIG  $(r = -0.823)$ , suggesting that  $\log (\mathcal{L}_{\Sigma I})$  is a reasonable unsupervised indicator of disentanglement.

in the lower right of the plot correspond with  $\lambda = 0$  and incorporating any  $\lambda > 0$  leads to a decrease in  $\mathcal{L}_{\Sigma I}$  and increase in MSE. As  $\lambda$  is increased further the MSE increases only slightly as the average  $\mathcal{L}_{\Sigma I}$  decreases significantly.

Figure 3b plots the relationship between end-of-training  $\mathcal{L}_{\Sigma I}$  values with MIG evaluation scores for both Beamsynthesis and dSprites. Our experiments depict a moderate negative relationship with correlation coefficient  $-0.823$ . These results suggest that  $\mathcal{L}_{\Sigma I}$  is a reasonable unsupervised indicator of successful disentanglement. This is very helpful in the practical setting where one likely does not have access to the ground truth data factors.

# EFFECT OF  $\lambda$  AND  $\sigma$  ON DISENTANGLEMENT

In this experiment, we plot the distributional MIG scores of GCAE as the latent space noise level  $\sigma$  and disentanglement strength  $\lambda$  vary on Beamsynthesis and dSprites. In each figure, each dark line plots the mean of the MIG scores while the shaded area fills one standard deviation of reported scores around the mean.

(a) Beamsynthesis  
![](images/2a15eb8a7617b93a3b03106fc0496a5be9a0dfbec9fb46e4041d0bc0d2168568.jpg)  
Figure 4: Effect of  $\mathcal{L}_{\Sigma I}$  and  $\sigma$  on MIG. Noise levels  $\sigma = \{0.2, 0.3\}$  are preferable for reliable MIG performance. KEY: Dark lines - mean scores. Shaded areas - one standard deviation.

![](images/0eb85e69a7c33f3635527406462c455ff963985e27225d1a3fb5256633d34964.jpg)  
(b) dSprites

Figure 4a depicts the MIG scores of GCAE on the Beamsynthesis dataset. All  $\sigma$  levels exhibit low MIG ( $\sim 0.07$ ) when  $\lambda$  is set to zero. The model is well-fit to the data, but the representation is highly redundant and entangled, causing the "gap" for each factor to be low. However, whenever  $\lambda > 0$  the

disentanglement performance increases significantly, with average performance of all three noise levels averaging at 0.35. There appears to be a slight preference for higher noise levels, as  $\sigma = 0.1$  generally has higher variance and lower disentanglement scores.

Figure 4b depicts the MIG scores of GCAE on the dSprites dataset. Similar to the previous experiment with Beamsynthesis, no disentanglement pressure leads to very low MIG scores  $(\sim 0.03)$ , but introducing  $\lambda > 0$  significantly boosts MIG performance to 0.35 for  $\sigma = \{0.2, 0.3\}$ . Here, there is a clear preference for larger  $\sigma$ ;  $\sigma = \{0.2, 0.3\}$  reliably lead to high scores with very little variance.

# COMPARISON OF GCAE WITH LEADING DISENTANGLEMENT METHODS

We incorporate experiments with leading VAE-based baselines. Each solid line represents the mean of MIG scores for each method and the shaded areas represent one standard deviation around the mean.

![](images/d7115cc7b1eb7e3c1b769ae4b83f4131c76a8a013b90689d9b9af8ada8dd2afe.jpg)  
(a) Beamsynthesis

![](images/7643e98aab4659f598f6f9e8d386b63bc3f128cbe1aa328d69e23b7fe14a9afd.jpg)  
Figure 5: MIG comparison of GCAE with VAE baselines on Beamsynthesis (left) and dSprites (right). GCAE  $\lambda$  is plotted on the lower axis, and VAE-based method regularization strength  $\beta$  is plotted on the upper axis. KEY: Dark lines - mean scores. Shaded areas - one standard deviation.  
(b) dSprites

Figure 5a depicts the distributional MIG performance of all considered methods on Beamsynthesis. When no disentanglement pressure is applied, MIG scores for all methods average at  $\sim 0.07$ . When disentanglement pressure is applied  $(\lambda, \beta > 0)$ , the distributional MIG scores of all methods increase. GCAE methods score highest ( $\sim 0.4$ ) with low relative variance.  $\beta$ -TCVAE consistently scores second-highest at ( $\sim 0.25$ ) with moderate variance. FactorVAE and  $\beta$ -VAE exhibit similar low performance ( $\sim 0.12$ ), but FactorVAE has much higher variance in its MIG scores.

Figure 5b shows a similar experiment for dSprites. Applying disentanglement pressure significantly increases MIG scores. GCAE achieves highest average overall scores at 0.37 with very little variance.  $\beta$ -VAE achieves the second-highest top score with extremely little variance but only for a very narrow range of  $\beta$ .  $\beta$ -TCVAE scores very high on average for a wide range of  $\beta$  but with large variance in scores. FactorVAE had the lowest average scores with moderate variance. We attribute the lack of performance of FactorVAE to the encroaching curse of dimensionality that the discriminator must overcome at  $m = 10$ .

# 5 DISCUSSION

Overall, the results indicate that GCAE is a highly competitive disentanglement method. It achieves the highest average MIG scores on the Beamsynthesis and dSprites datasets, and it has very low variance in MIG scores when  $\sigma = \{0.2, 0.3\}$ . The hyperparameters are highly transferable, as  $\lambda \in [0.1, 0.5]$  works well on multiple datasets and minimal domain knowledge of the task is required. For example, GCAE was the highest-performing while using the same data preprocessing (mean and standard deviation normalization) whereas VAE-based methods require domain knowledge in the form of a Gaussian (Beamsynthesis) or Bernoulli (dSprites) decoder. We also find that  $\mathcal{L}_{\Sigma I}$  is

a reasonable indicator of disentanglement performance which can be very useful for selecting a model when data generating factors for a task are unknown. Furthermore, we note that it would be trivial to convert GCAE to a generative model once it is disentangled - each  $p(\mathbf{z}_i)$  can be measured independently in the interval  $[-4, 4]$ , and the corresponding inverse cumulative distribution function can be constructed and sampled from.

While GCAE performs well in many scenarios, it has several limitations. In contrast to the VAE optimization process which is very robust Kingma & Welling (2013), the optimization of  $m$  discriminators is sensitive to hyperparameters such as learning rate and optimizer. Training  $m$  discriminators requires a lot of data and computation, and the quality of the learned representation depends heavily on the quality of the conditional densities stored in the discriminators. Increasing the latent space noise  $\sigma$  assists with gradient descent on  $\mathcal{L}_{\Sigma I}$  and generally leads to improved disentanglement outcomes, but it limits the corresponding information capacity of the latent space.

Future work involves establishing best practices for optimizing the discriminators, characterizing the performance of GCAE as a generative model, and exploring partitioning the latent space into chunks rather than individual elements.

# 6 CONCLUSION

We have presented Gaussian Channel Autoencoder (GCAE), a new disentanglement method which employs Gaussian noise and non-parametric density estimation in the latent space to achieve reliable, high-performing disentanglement outcomes. GCAE avoids the curse of dimensionality of density estimation by minimizing the Dual Total Correlation (DTC) metric with a weighted information functional to capture disentangled data generating factors. The method is shown to consistently outcompete existing SOTA baselines on the MIG metric on the Beamsynthesis and dSprites datasets.

# REFERENCES

Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798-1828, 2013.  
Christopher P Burgess, Irina Higgins, Arka Pal, Loic Matthew, Nick Watters, Guillaume Desjardins, and Alexander Lerchner. Understanding disentangling in  $\beta$ -vae. arXiv preprint arXiv:1804.03599, 2018.  
Ricky TQ Chen, Xuechen Li, Roger B Grosse, and David K Duvenaud. Isolating sources of disentanglement in variational autoencoders. Advances in neural information processing systems, 31, 2018.  
Yuanqi Du, Xiaojie Guo, Amarda Shehu, and Liang Zhao. Interpretable molecule generation via disentanglement learning. In Proceedings of the 11th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics, pp. 1-8, 2020.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014a.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014b.  
Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. 2016.  
Sarthak Jain, Edward Banner, Jan-Willem van de Meent, Iain J Marshall, and Byron C Wallace. Learning disentangled representations of texts with application to biomedical abstracts. In Proceedings of the Conference on Empirical Methods in Natural Language Processing. Conference on Empirical Methods in Natural Language Processing, volume 2018, pp. 4683. NIH Public Access, 2018.

Vineet John, Lili Mou, Hareesh Bahuleyan, and Olga Vechtomova. Disentangled representation learning for non-parallel text style transfer. arXiv preprint arXiv:1808.04339, 2018.  
Hyunjik Kim and Andriy Mnih. Disentangling by factorising. In International Conference on Machine Learning, pp. 2649-2658. PMLR, 2018.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. Self-normalizing neural networks. In Proceedings of the 31st international conference on neural information processing systems, pp. 972-981, 2017.  
Abhishek Kumar, Prasanna Sattigeri, and Avinash Balakrishnan. Variational inference of disentangled latent concepts from unlabeled observations. arXiv preprint arXiv:1711.00848, 2017.  
Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, and Olivier Bachem. Challenging common assumptions in the unsupervised learning of disentangled representations. In international conference on machine learning, pp. 4114-4124. PMLR, 2019.  
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan Frey. Adversarial autoencoders. arXiv preprint arXiv:1511.05644, 2015.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32: 8026-8037, 2019.  
Daniil Polykovskiy, Alexander Zhebrak, Dmitry Vetrov, Yan Ivanenko, Vladimir Aladinskiy, Polina Mamoshina, Marine Bozdaganyan, Alexander Aliper, Alex Zhavoronkov, and Artur Kadurin. Entangled conditional adversarial autoencoder for de novo drug discovery. Molecular pharmaceutics, 15(10):4398-4405, 2018.  
Mhd Hasan Sarhan, Nassir Navab, Abouzar Eslami, and Shadi Albarqouni. Fairness by learning orthogonal disentangled representations. In European Conference on Computer Vision, pp. 746-761. Springer, 2020.  
Eric Yeats, Frank Liu, David Womble, and Hai Li. Nashae: Disentangling representations through adversarial covariance minimization. arXiv preprint arXiv:2209.10677, 2022.
