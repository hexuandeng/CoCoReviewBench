# AMORTISED MAP INFERENCE FOR IMAGE SUPER-RESOLUTION

Casper Kaae Sønderby $^{12*}$ , Jose Caballero $^{1}$ , Lucas Theis $^{1}$ , Wenzhe Shi $^{1}$  & Ferenc Huszár $^{1}$

casperkaae@gmail.com, {jcaballero,ltheis,wshi,fhuszar}@twitter.com

$^{1}$ Twitter Cortex, London, UK

$^{2}$ University of Copenhagen, Denmark

# ABSTRACT

Image Super-resolution (SR) is an undetermined inverse problem, where a large number of plausible high-resolution images can explain the same downsampled image. Most current single image SR methods use empirical risk minimisation, often with a pixel-wise mean squared error (MSE) loss. However, the outputs from such methods tend to be blurry, over-smoothed and generally appear implausible. A more desirable approach would employ Maximum a Posteriori (MAP) inference, preferring solutions that always have a high probability under the image prior, and thus appear more plausible. Direct MAP estimation for SR is non-trivial, as it requires us to build a model for the image prior from samples. Here we introduce new methods for amortised MAP inference whereby we calculate the MAP estimate directly using a convolutional neural network. We first introduce a novel neural network architecture that performs a projection to the affine subspace of valid SR solutions ensuring that the high resolution output of the network is always consistent with the low resolution input. We show that, using this architecture, the amortised MAP inference problem reduces to minimising the cross-entropy between two distributions, similar to training generative models. We propose three methods to solve this optimisation problem: (1) Generative Adversarial Networks (GAN) (2) denoiser-guided SR which backpropagates gradient-estimates from denoising to train the network, and (3) a baseline method using a maximum-likelihood-trained image prior. Our experiments show that the GAN based approach performs best on real image data, achieving particularly good results in photo-realistic texture SR.

# 1 INTRODUCTION

Image super-resolution (SR) is the underdetermined inverse problem of estimating a high resolution (HR) image given the corresponding low resolution (LR) input. This problem has recently attracted significant research interest due to the potential of enhancing the visual experience in many applications while limiting the amount of raw pixel data that needs to be stored or transmitted. While SR has many applications in for example medical diagnostics or forensics (Nasrollahi & Moeslund, 2014, and references therein), here we are primarily motivated to improve the perceptual quality when applied to natural images. Most current single image SR methods use empirical risk minimisation, often with a pixel-wise mean squared error (MSE) loss (Dong et al., 2016; Shi et al., 2016). However, MSE, and convex loss functions in general, are known to have limitations when presented with uncertainty in multimodal and nontrivial distributions such as distributions over natural images. In SR, a large number of plausible images can explain the LR input and the Bayes-optimal behaviour for any MSE trained model is to output the mean of the plausible solutions weighted according to their posterior probability. For natural images this averaging behaviour leads to blurry and over-smoothed outputs that generally appear implausible, i.e. the produced estimates have low probability under the natural image prior.

An idealised method for our applications would use a full-reference perceptual loss function that describes the sensitivity of the human visual perception system to different distortions. However the

![](images/9a78c8f30dd7d9946a64830e04c45c1db4dec4943c691358ca0bbbb21a0e1798.jpg)  
Figure 1: Illustration of the SR problem via a toy example. Two-dimensional HR data  $y = [y_{1},y_{2}]$  is drawn from a Swiss-roll distribution (in gray). Downsampling is modelled as  $x = \frac{y_1 + y_2}{2}$ . a) Given observation  $x = 0.5$ , valid SR solutions lie along the line  $y_{2} = 1 - y_{1}(- - - )$ . The red shading illustrates the magnitude of the posterior  $p_{Y|X = 0.5}$ . Bayes-optimal estimates under MSE and MAE as well as the MAP estimate given  $x = 0.5$  are marked with labels. The MAP estimates for different values of  $x\in [-8,8]$  are also shown  $(-\bullet -)$ . b) Trained model outputs for  $x\in [-8,8]$  and estimated gradients from a denoising function trained on  $p_Y$ . Note the AffGAN  $(-\bullet-)$  and AffDG  $(-\bullet-)$  models fit the posterior mode well whereas the MSE  $(-\bullet-)$  and MAE  $(-\bullet-)$  model outputs generally fall in low probability regions.

![](images/b0b75b7f7f6344980be3f3221517f3009b6e8ecfbe0f16f59bf7f3e714684910.jpg)  
Table 1: Directly estimated cross-entropy  $\mathbb{H}[q_{\theta},p_{Y}]$  values. The AffGAN and AffDG achieves cross-entropy values close to the MAP solution confirming that they minimize the desired quantity. The MSE and MAE models performs worse since they do not minimize the cross-entropy. Further the models using affine projections (Aff) performs better than the soft constrained models.

<table><tr><td></td><td>H[qθ,pY]</td><td>lMSE(x,Ay)</td></tr><tr><td>MAP</td><td>3.15</td><td>-</td></tr><tr><td>MSE</td><td>9.10</td><td>1.25·10-2</td></tr><tr><td>MAE</td><td>6.30</td><td>4.04·10-2</td></tr><tr><td>AffGAN</td><td>4.10</td><td>0.0</td></tr><tr><td>SoftGAN</td><td>4.25</td><td>8.87·10-2</td></tr><tr><td>AffDG</td><td>3.81</td><td>0.0</td></tr><tr><td>SoftDG</td><td>4.19</td><td>1.01·10-1</td></tr></table>

most widely used loss functions MSE and the related peak-signal-to-noise-ratio (PSNR) metric have been shown to correlate poorly with human perception of image quality (Laparra et al., 2016; Wang et al., 2004). Improved perceptual quality metrics have been proposed, the most popular being structural similarity (SSIM) (Wang et al., 2004) and its multi-scale variants (Wang et al., 2003). See also (Laparra et al., 2010; 2016; Bruna et al., 2016) for more recent research. Although the correlation of these metrics with human perception has improved, they still do not provide a fully satisfactory alternative to MSE for training of neural networks (NN) for SR.

In lieu of a satisfactory perceptual loss function, we leave the empirical risk minimisation framework and present methods based only on natural image statistics. In this paper we argue that a desirable approach is to employ amortised Maximum a Posteriori (MAP) inference, preferring solutions that have a high posterior probability and thus high probability under the image prior while keeping the computational benefits of amortised inference. To motivate why MAP inference is desirable consider the toy problem in Figure 1a, where the HR data is two-dimensional  $y = [y_{1},y_{2}]$  and distributed according to the Swiss-roll density. The LR observation is defined as the average of the two pixels  $x = \frac{y_1 + y_2}{2}$ . Consider observing a LR data point  $x = 0.5$ : the set of possible HR solutions is the line  $y_{1} = 2x - y_{2}$ , more generally an affine subspace, which is shown by the dashed line in Figure 1a. The posterior distribution  $p(y|x)$  is thus degenerate, and corresponds to a slice of the prior along this line, as shown by the red shading. If one minimise MSE or Mean Absolute Error (MAE), the Bayes-optimal solution will lie at the mean or the median along the line, respectively. This example illustrates that MSE and MAE can produce output with very low probability under that data prior whereas MAP inference would always find the mode which by definition is in a high-probability region. See Section 5.6 for a discussion of possible limitations of the MAP inference approach.

Our first contribution is a convolutional neural networks (CNN) architecture designed to exploit the structure of the SR problem. Image downsampling is a linear transformation, and can be modelled as a strided convolution. As Figure 1a illustrates, the set of HR images  $y$  that are compatible with any LR image  $x$  span an affine subspace. We show that by using specifically chosen linear convolution and deconvolution layers we can implement a projection to this affine subspace. This ensures that our CNNs always output estimates that are consistent with the inputs. The affine projection layer can be added to any CNN, or indeed, any other trainable SR algorithm. Using this architecture we show that training the model for MAP inference reduces to minimising the cross-entropy  $\mathbb{H}[q_G,p_Y]$  between the HR data distribution  $p_{Y}$  and the implied distribution  $q_{G}$  of the model's output when

evaluated at random LR images. As a result, we don't need corresponding HR and LR image pairs any more, and training becomes more akin to training generative models. However direct minimisation of the cross-entropy is not possible and in this paper we present three approaches, all depending on projecting the model output to the affine subspace of valid solution, to approximate it directly from data:

1. We present a variant of the Generative Adversarial Networks (GAN) (Goodfellow et al., 2014) which approximately minimises the Kullback-Leibler divergence (KL) and cross-entropy between  $q_{G}$  and  $p_{Y}$ . Our analysis provides theoretical grounding for using GANs in image SR (Ledig et al., 2016). We also introduce a trick that we call instance noise that can be generally applied to address the instability of training GANs.  
2. We employs denoising as a way to capture natural image statistics. Bayes-optimal denoising approximately learn to take a gradient step along the log-probability of the data distribution (Alain & Bengio, 2014; Rasmus et al., 2015; Greff et al., 2016). These gradient estimates from denoising can be directly backpropagated through the network to minimise cross-entropy between  $q_{G}$  and  $p_{Y}$  via gradient descent.  
3. We present an approach where the probability density of data is directly modelled via a generative model trained by maximum likelihood. We use a differentiable generative model based on PixelCNNs (Oord et al., 2016) and Mixture of Conditional Gaussian Scale Mixtures (MCGSM, Theis et al., 2012) whose performance we believe is very close to the-state-of-the-art in this category.

In section 5 we empirically demonstrate the behaviour of the proposed methods on both the two dimensional toy dataset and on real image datasets.

# 2 RELATED WORK

The GAN framework was introduced by Goodfellow et al. (2014) which also showed that these models minimise the Shannon-Jensen Divergence between  $q_{G}$  and  $p_{Y}$  under certain conditions. In Section 3.2, we present an update rule that corresponds to minising  $\mathrm{KL}[q_{G}\| p_{Y}]$ . Recently, Nowozin et al. (2016) presented a more general treatment that connects GANs to  $f$ -divergence minimisation. In parallel to our contributions, theoretical work by Mohamed & Lakshminarayanan (2016) presented a unifying view on learning in GAN-style algorithms, of which our variant can be regarded as special case. The focus of several recent papers on GANs were algorithmic tricks to improve their stability (Radford et al., 2015; Salimans et al., 2016). In Section 3.2.1 we introduce another such trick we call instance noise. We discuss theoretical motivations for this and compare it to one-sided label smoothing proposed by Salimans et al. (2016). Recently, several attempts have been made to improve perceptual quality of image SR using deep representations of natural images. Bruna et al. (2016) and Li & Wand (2016) measure the Euclidean distance in the nonlinear feature space of a deep NN pre-trained to perform object classification. Dosovitskiy & Brox (2016) and Ledig et al. (2016) use a similar approach and also add an adversarial loss term. Unpublished work by Garcia (2016) explored combining GANs with an  $L_{1}$  penalty between the LR input and the down-sampled output. We note that the soft  $L_{2}$  or  $L_{1}$  penalties used in these methods can be interpreted as assuming Gaussian and Laplace observation noise. In contrast, our approach assumes no observation noise and satisfies the consistency of inputs and outputs exactly by using an affine projection as explained in Section 3.1. In other work, Larsen et al. (2015) proposed to replace the pixel-wise MSE used for training of variational autoencoders with a learned metric from the GAN discriminator. Our denoiser based method exploits a fundamental connection between probabilistic modelling and learning to denoise (see e.g. Vincent et al., 2008; Alain & Bengio, 2014; Vincent, 2011; Särelä & Valpola, 2005; Rasmus et al., 2015): a Bayes-optimal denoiser can be used to estimate the gradient of the log probability of data. To our knowledge this work is the first time that the output of a denoiser is explicitly back-propagated to train another network.

# 3 THEORY

Consider a function  $f_{\theta}(x)$  parametrised by  $\theta$  which maps a LR observation  $x$  to a HR estimate  $\hat{y}$ . Most current SR methods optimise model parameters via empirical risk minimization:

$$
\underset {\theta} {\operatorname {a r g m i n}} \mathbb {E} _ {y, x} [ \ell (y, f _ {\theta} (x)) ] \tag {1}
$$

Where  $y$  is the true target and  $\ell$  is some loss function. The loss function is typically a simple convex function most often MSE  $\ell_{\mathrm{MSE}}(y,\hat{y}) = \| y - \hat{y}\| _2^2$  as in (Dong et al., 2016; Shi et al., 2016). Here, we seek to perform MAP inference instead. For a single LR observation the MAP estimate is

$$
\hat {y} (x) = \underset {y} {\operatorname {a r g m a x}} \log p _ {Y \mid X} (y \mid x) \tag {2}
$$

Instead of calculating  $\hat{y}$  for each  $x$  separately we perform amortised inference, i.e. we would like to train the SR function  $f_{\theta}(x)$  to calculate the MAP estimate. A natural loss function for learning the parameters  $\theta$  is the average log-posterior:

$$
\underset {\theta} {\operatorname {a r g m a x}} \mathbb {E} _ {x} \log p _ {Y | X} \left(f _ {\theta} (x) | x\right), \tag {3}
$$

where the expectation is taken over the distribution of LR observations  $x$ . This loss depends on the unknown posterior distribution  $p_{Y|X}$ . We proceed by decomposing the log-posterior using Bayes' rule as follows.

$$
\underset {\theta} {\operatorname {a r g m a x}} \left\{\underbrace {\mathbb {E} _ {x} \log p _ {X \mid Y} (x \mid f _ {\theta} (x))} _ {\text {L i k e l i h o o d}} + \underbrace {\mathbb {E} _ {x} \log p _ {Y} \left(f _ {\theta} (x)\right)} _ {\text {P r i o r}} - \underbrace {\mathbb {E} _ {x} \log p _ {X} (x)} _ {\text {M a r g i n a l L i k e l i h o o d}} \right\}. \tag {4}
$$

# 3.1 HANDLING THE LIKELIHOOD TERM

Notice that the last term of Eqn. (4), the marginal likelihood, does not depend on  $\theta$ , so we only have to deal with the likelihood and image prior. The observation model in SR can be described as follows.

$$
x = A \hat {y}, \tag {5}
$$

where  $A$  is a linear transformation used for image downsampling. In general,  $A$  can be modelled as a strided two-dimensional convolution. Therefore, the likelihood term in Eqn. (4) is degenerate  $p(x|f_{\theta}(x)) = \delta (x - Af_{\theta}(x))$ , and Eqn. (4) can be rewritten as constrained optimisation:

$$
\operatorname {a r g m a x} \quad \mathbb {E} _ {x} [ \log p _ {Y} (f _ {\theta} (x)) ] \tag {6}
$$

$$
\begin{array}{c} \theta \\ \forall x: A f _ {\theta} (x) = x \end{array}
$$

To satisfy the constraints, we introduce a parametric function class that always guarantees  $Af_{\theta}(x) = x$ . Specifically, we propose to use functions of the form

$$
g _ {\theta} (x) = \Pi_ {x} ^ {A} f _ {\theta} (x) = \left(I - A ^ {+} A\right) f _ {\theta} (x) + A ^ {+} x \tag {7}
$$

where  $f_{\theta}$  is an arbitrary mapping from LR to HR space,  $\Pi_x^A$  a projection to the affine subspace  $\{y:yA = x\}$ , and  $A^{+}$  is the Moore-Penrose pseudoinverse of  $A$ , which satisfies  $AA^{+}A = A$  and  $A^{+}AA^{+} = A^{+}$ . Conveniently, if  $A$  is a strided two-dimensional convolution, then  $A^{+}$  becomes a deconvolution or up-convolution, which is a standard operation used in deep learning (e.g. Shi et al., 2016). It is important to stress that the optimal deconvolution  $A^{+}$  is not simply the transpose of  $A$ , Figure 2 illustrates the upsampling kernel  $(A^{+})$  that corresponds to a Gaussian downsampling kernel  $(A)$ . For any  $A$  the deconvolution  $A^{+}$  can be easily found, here we used numerical methods as detailed in Appendix B. Intuitively,  $A^{+}x$  can be thought of as a baseline SR solution, while  $(I - A^{+}A)f_{\theta}$  is the residual. The operation  $(I - A^{+}A)$  is a projection to the null-space of  $A$ , therefore when we downsample the residual  $(I - A^{+}A)f_{\theta}$  we are guaranteed to get 0 no matter what  $f_{\theta}$  is. By using functions of this form we can turn Eqn. (6) into an unconstrained optimization problem.

$$
\underset {\theta} {\operatorname {a r g m a x}} \mathbb {E} _ {x} \log p _ {Y} \left(\Pi_ {x} ^ {A} f _ {\theta} (x)\right) \tag {8}
$$

Interestingly, the objective above can be expressed in terms of the probability distribution of the model output  $q_{\theta}(y) \coloneqq \int \delta \left(y - \Pi_x^A f_{\theta}(x)\right)p_X(x)dx$  as follows.

$$
\underset {\theta} {\operatorname {a r g m a x}} \mathbb {E} _ {x} \log p _ {Y} \left(\Pi_ {x} ^ {A} f _ {\theta} (x)\right) = \underset {\theta} {\operatorname {a r g m a x}} \mathbb {E} _ {\hat {y} \sim q _ {\theta}} \log p _ {Y} (\hat {y}) = \underset {\theta} {\operatorname {a r g m i n}} \mathbb {H} [ q _ {\theta}, p _ {Y} ], \tag {9}
$$

where  $\mathbb{H}[q,p]$  denotes the cross-entropy between  $q$  and  $p$  and we used  $\mathbb{H}[q_{\theta},p_Y] = \mathbb{E}_{\hat{y}\sim q_\theta}[-\log p_Y(\hat{y})]$ . To minimise this objective, we do not need matched input-output pairs as in empirical risk minimisation. Instead we need to match the marginal distribution of reconstructed images  $q_{\theta}$  to that of the distribution of HR images. In this respect, the problem becomes more akin to unsupervised learning or generative modelling. In the following sections we present three approaches to finding the optimal  $\theta$  utilising the properties of the affine projection.

# 3.2 AFFINE PROJECTED GENERATIVE ADVERSARIAL NETWORKS

Generative Adversarial Networks (Goodfellow et al., 2014) consist of a generator  $G$  that turns noise sampled from some distribution  $z \sim p_Z$  into images  $G(z)$  via a parametric mapping, and a discriminator  $D$  that learns to distinguish between real and synthetic images. The generator and discriminator are updated in tandem resulting in the generative distribution  $q_G$  moving closer to the distribution of real data  $p_Y$ . The behaviour of GANs depends on the specifics of how the generator and the discriminator are trained. We use the following objective functions for  $D$  and  $G$ :

$$
\mathcal {L} (D; G) = - \mathbb {E} _ {y \sim p _ {Y}} \log D (y) - \mathbb {E} _ {z \sim p _ {Z}} \log (1 - D (G (z)), \tag {10}
$$

$$
\mathcal {L} (G; D) = \mathbb {E} _ {z \sim p _ {Z}} \log \frac {D (G (z))}{1 - D (G (z))}.
$$

The algorithm iterates two steps: first, it updates  $D$  by lowering  $\mathcal{L}(D;G)$  keeping  $G$  fixed, then it updates  $G$  by lowering  $\mathcal{L}(G;D)$  keeping  $D$  fixed. It can be shown that this amounts to minimising  $\mathrm{KL}[q_G\| p_Y]$ , where  $q_{G}$  is the distribution of samples generated by  $G$ . See Appendix A for a proof In the context of SR, the affine projected SR function  $\Pi_x^A f_\theta$  takes the role of the generator. Instead of noise, the generator is now fed low-resolution images  $x\sim p_X$ . Leaving everything else unchanged, we can deploy the GAN algorithm to minimise  $\mathrm{KL}[q_{\theta}\| p_{Y}]$ . We call this algorithm affine projected GAN or AffGAN for short. Similarly, we introduce notation SoftGAN to denote the GAN algorithm without the affine projection, which instead uses an additional soft-constraint  $\ell_{LR} = \mathrm{MAE}(x,A\hat{y})$  as in (Garcia, 2016). Note that the difference between the cross-entropy and the KL divergence is the entropy of  $q_{\theta}$ :  $\mathbb{H}[q_{\theta},p_{Y}] - \mathrm{KL}[q_{\theta}\| p_{Y}] = \mathbb{H}[q_{\theta}]$ . Hence, we can expect AffGAN to favour approximate MAP solutions that lead to higher entropy and thus more diverse solutions overall.

# 3.2.1 INSTANCE NOISE

The theory suggests that GANs should be a convergent algorithm. If a unique optimal discriminator exists and it is reached by optimising  $D$  to perfection at each step, technically the whole algorithm corresponds to gradient descent on an estimate of  $\mathrm{KL}[q_{\theta} \| p_{Y}]$  with respect to  $\theta$ . In practice, however, GANs tend to be highly unstable. So where does the theory go wrong? We think the main reason for the instability of GANs stems from  $q_{\theta}$  and  $p_{Y}$  being concentrated distributions whose support does not overlap. The distribution of natural images  $p_{Y}$  is often assumed to concentrate on or around a low-dimensional manifold. In most cases,  $q_{\theta}$  is degenerate and manifold-like by construction, such as in AffGAN. Therefore, odds are that especially before convergence is reached,  $q_{\theta}$  and  $p_{Y}$  can be perfectly separated by several  $D$ s violating a condition for the convergence proof. We try to remedy this problem by adding instance noise to both SR and true image samples. This amounts to minimising the divergence  $d_{\sigma}(q_{\theta}, p_{Y}) = \mathrm{KL}[p_{\sigma} * q_{\theta} \| p_{\sigma} * p_{Y}]$ , where  $p_{\sigma} * q_{\theta}$  denotes convolution of  $q_{\theta}$  with the noise distribution  $p_{\sigma}$ . The noise level  $\sigma$  can be annealed during training, and the noise allows us to safely optimise  $D$  until convergence in each iteration. The trick is related to one-sided label noise introduced by Salimans et al. (2016), however without introducing a bias in the optimal discriminator, and we believe it is a promising technique for stabilising GAN training in general. For more details please see Appendix C

# 3.3 DENOISER GUIDED SUPER-RESOLUTION

To optimise the criterion Eqn. (6) via gradient descent we need its gradient with respect to  $\theta$ :

$$
\frac {\partial}{\partial \theta} \mathbb {E} _ {x} [ \log p \left(\Pi_ {x} ^ {A} f _ {\theta} (x)\right) ] = \mathbb {E} _ {x} \left[ \left. \frac {\partial}{\partial y} \log p (y) \right| _ {y = \Pi_ {x} ^ {A} f _ {\theta} (x)} \cdot \Pi_ {x} ^ {A} \frac {\partial}{\partial \theta} f _ {\theta} (x) \right] \tag {11}
$$

Here  $\frac{\partial}{\partial\theta} f_{\theta}$  are the gradients of the SR function which can be calculated via back-propagation whereas  $\frac{\partial}{\partial y}\log p_Y(y)$  requires estimation since  $p_{Y}$  is unknown. We use results from (Alain & Bengio, 2014; Sarelă & Valpola, 2005) showing that in the limit of infinitesimal Gaussian noise, optimal denoising functions can be used to estimate this gradient:

$$
f _ {\sigma} ^ {*} = \underset {f} {\operatorname {a r g m i n}} \mathbb {E} _ {y \sim p _ {Y}} \ell_ {M S E} (f (y + \sigma \epsilon), y) \Rightarrow \frac {f ^ {*} (y) - y}{\sigma^ {2}} \approx \frac {\partial}{\partial y} \log p _ {Y} (y), \tag {12}
$$

where  $\epsilon \sim \mathcal{N}(0, I)$  is Gaussian white noise,  $f_{\sigma}^{*}$  is the Bayes-optimal denoising function for noise level  $\sigma$ . Using these results we can maximise Eqn. (9) by first training a neural network to denoise samples from  $p_{Y}$  and then backpropagate the gradient estimates from Eqn. (12) via the chain rule in Eqn. (11) to update  $\theta$ . Well call this method AffDG, as it uses the affine subspace projection and is guided by the gradient from the DAE. Similar to above we'll call the similar algorithm soft-enforcing Eqn. (5) SoftDG.

# 3.4 DENSITY GUIDED SUPER-RESOLUTION

As a more direct baseline model for amortised MAP inference we fit a tractable, yet powerful density model to  $p_{Y}$  using maximum likelihood, and then use cross entropy with respect to the generative model to approximate Eqn. (9). We use a deep generative model similar to the pixelCNN (Oord et al., 2016) but with a continuous (and differentiable) MCGSM (Theis et al., 2012) likelihood. These type of models are state-of-the-art in density estimation, are relatively fast to evaluate and produce visually interesting samples (Oord et al., 2016). We call this method AffLL, as it uses the affine projection and is guided by the log-likelihood of a density model.

# 4 EXPERIMENTS

We designed our experiments to address the following questions:

- Are the methods proposed in Section 3 successful at minimising cross-entropy?  $\rightarrow$  Section 5.1  
- Does the affine projection layer hurt the performance of CNNs for image SR?  $\rightarrow$  Section 5.2  
- Do the proposed methods produce perceptually superior SR results?  $\rightarrow$  Sections 5.3-5.5

We initially illustrate the behaviour of the proposed algorithms on data where exact MAP inference is computationally tractable. Here the HR data  $y = [y_{1},y_{2}]$  is drawn from a two-dimensional noisy Swiss-roll distribution and the one-dimensional LR data  $x$  is simply the average of the two HR pixels. Next we tested the proposed algorithm in a series of experiments on natural images using  $4\times$  downsampling.. For the first dataset, we took random crops from HR images containing grass texture. SR of random textures is known to be very hard using MSE or MAE loss functions. Finally, we tested the proposed models on real image data of faces (Celeb-A) and natural images (ImageNet). All models were convolution neural networks implemented using Theano (Team et al., 2016) and Lasagne (Dieleman et al., 2015). We refer to Appendix D for full experimental details.

# 5 RESULTS AND DISCUSSION

# 5.1 2D MAP INFERENCE: SWISS-ROLL

In this experiment we wanted to demonstrate that AffGAN and AffDG are indeed minimising the MAP objective in Eqn. (9). For this we used the two-dimensional toy problem where  $p_{Y}$  can be evaluated using brute-force Monte Carlo. Figure 1b) shows the outputs for  $x = [-8,8]$  for models trained with different criterion. The AffGAN and AffDG solutions largely fit the dominant mode similar to MAP inference. For the MSE and MAE models the output generally falls in regions with low prior density. Table 1 shows the cross-entropy  $\mathbb{H}[q_{\theta},p_{Y}]$  achieved by different methods, averaged over 10 independent trials with random initialisation. The cross-entropy values for the GAN and DAE based models are relatively close to the optimal MAP solution, which in this case we can find in a brute-force way. As expected the MSE and MAE models perform worse as these models do not minimize  $\mathbb{H}[q_{\theta},p_{Y}]$ . We also calculated the average MSE between the network input and the downsampled network output. For the affine projected models, this error is exactly 0. The soft constrained models only approximately satisfy this constraint, even after extensive training (Table 1 second column). Further, we observe that the affine projected models generally found a lower cross-entropy  $\mathbb{H}[q_{\theta},p_{Y}]$  when compared to soft-constrained versions.

# 5.2 AFFINE PROJECTED NETWORKS: PROOF OF CONCEPT USING MSE CRITERION

Adding the affine projection  $\Pi_x^A$  restricts the class of functions that the SR network can model, so it is important to verify that the network is still capable of achieving the same performance in

![](images/a832f5681cf1e3bfad2d8b651b3d4d9a7b4d2c8e581d834dc87b65ef97de5cba.jpg)  
Figure 2: CelebA performance for MSE models during training. The distance between HR model output  $\hat{y}$  and true HR image  $y$  using MSE in a) and SSIM in b). MSE in LR space between input  $x$  and down-sampled model output  $A\hat{y}$  in c). The tuple in the legend indicate: ((F)fixed / (T)rainable affine projection, (T)trained / (R)random initialised affine projections). The models using pre-trained affine projections (fixed: , trainable: always performs better in all metrics compared to models using either random initialized affine projections (or no projection ). Further, a fixed pre-trained affine projection ensures the best consistency between input and down-sampled output as seen in figure c).  $A$  (top) and  $A^{+}$  (bottom) kernels of the affine projection are seen in d).

![](images/3a4be8cad0b2866abe2242136bdf129f32f36b151bc361755dbb96ed713183d5.jpg)

![](images/79373312d02ca2f81e552e9f3a14263528f752ce2c2accf10dc4ba52482f89db.jpg)

![](images/b4b26b708a99f64bbb6209902fc8571b5c2858d44b0938cf9ae0eea37763222e.jpg)

SR as unconstrained CNN architectures. To test this, we trained CNNs with and without affine projections to perform SR on the CelebA dataset using MSE as the objective function. Results are shown in Figure 2. First note that when using affine projections, a randomly initialised network starts learning from a lower initial loss as the low-frequency components of the network output already match those of the target image. We observed that the affine projected networks generally train faster than unconstrained ones. Furthermore, the affine projected networks tend to find a better solution as measured by MSE and SSIM (Figure 2a-b). To investigate which aspects of the network architecture are responsible for the improved performance, we evaluated two further models: In one variant, we initialise the affine projected CNN to implement the correct projection, but then treat  $A^{+}$  as a trainable parameter. In the final variant, we keep the architecture the same, but initialise the final deconvolution layer  $A^{+}$  randomly and allow it to be trained. We found that initialising  $A^{+}$  to the correct Moore-Penrose inverse is important, and we get the similar results irrespective of whether or not it is fixed during training. Figure 2c shows the error between the network input and the downsampled network output. We can see that the exact affine projected network keeps this error at virtually 0.0 (up to numerical precision), whereas any other network will violate this consistency. In Figure 2d we show the downsampling kernel  $A$  and the corresponding optimal kernel for  $A^{+}$ .

# 5.3 GRASS TEXTURES

Random textures are known to be hard models using MSE loss function. Figure 3 shows  $4 \times \mathrm{SR}$  of grass texture patches using identical affine projected CNNs trained with different loss functions. When randomly initialised, affine projected CNNs always produce an output with the correct low-frequency components, as illustrated by the third panel labelled Affinit in Figure 3. The AffGAN model produces clearly the sharpest images, and we found the images to be plausible given the LR inputs. Notice that the reconstruction is not perfect pixel-by-pixel, but it has the correct statistical properties for the human visual system to recognise it as grass texture. The AffDG and AffLL models both produced blurry results which we where unable to improve upon using various optimization methods. Due to these findings we choose not to perform any further experiments with these models and concentrate on AffGAN instead. We refer to Appendix E for discussion of the results of these models.

# 5.4 CELEBA FACES

In Figure 4 the SR results are seen for several models trained using different loss functions. The MSE trained models outputs somewhat generic and over-smoothed images as expected. For the GAN models the global content is correct for both the affine projected and soft constrained models. Comparing the AffGAN and SoftGAN outputs the AffGAN model produces slightly sharper images

![](images/5cdfdbed54c6b99b4838832e4572df65f8c585faefdb1029038c427d9d9c4262.jpg)  
Figure 3:  $4 \times$  SR of grass textures. Top row shows LR model input  $x$ , true HR image  $y$  and model outputs according to figure legend. Bottom row shows zoom in on except from the images in the top row. The AffGAN image is much sharper than the somewhat blurry AffMSE image. Note that both the AffDG and AffLL produces very blurry results. The Affinit shows the output from an untrained affine projected model, i.e. the baseline solution, illustrating the effect of the upsampling using  $A^{+}$ .

which however also seem to contain slightly more high frequency noise. We observed some colour drifting for the soft constrained models. Table 2 shows quantitative results for the same four models where, in terms of PSNR and SSIM, the MSE model achieves the best scores as expected. The consistency between input and output clearly shows that the models using the affine projections satisfy Eqn. (5) better than the soft constrained versions for both MSE and GAN losses.

![](images/451aa7d02a998602f584f2d8099a4c9c068e80b15cfe26c28b1168f58785d653.jpg)  
Figure 4:  $4 \times$  SR of CelebA faces. Model input  $x$ , target  $y$  and model outputs according to figure legend. Both the AffGAN and SoftGAN produces clearly shaper images than the blurry MSE outputs. We found that AffGAN outputs slightly sharper images compared to SoftGAN, however also with slightly more high-frequency noise.

<table><tr><td></td><td>SSIM</td><td>PSNR</td><td>\( {\ell }_{MSE}\left( {x,A\widehat{y}}\right) \)</td></tr><tr><td>MSE</td><td>0.90</td><td>26.30</td><td>\( {8.0} \cdot  {10}^{-5} \)</td></tr><tr><td>AffMSE</td><td>0.91</td><td>26.53</td><td>\( {1.6} \cdot  {10}^{-{10}} \)</td></tr><tr><td>SoftGAN</td><td>0.76</td><td>21.11</td><td>\( {2.3} \cdot  {10}^{-3} \)</td></tr><tr><td>AffGAN</td><td>0.81</td><td>23.02</td><td>\( {9.1} \cdot  {10}^{-{10}} \)</td></tr></table>

Table 2: PSNR, SSIM and MSE scores for the CelebA dataset. In terms of PSNR and SSIM in HR space the MSE trained models achieves the best scores as expected and the AffGAN performs better than the SoftGAN. Considering  $\ell_{MSE}(x,A\hat{y})$  the models using the affine projections (Aff) clearly show better consistency between input  $x$  and down sampled model output  $A\hat{y}$  than models not using the projection.

# 5.5 NATURAL IMAGES

In Figure 5 we show the results for  $4 \times \mathrm{SR}$  from  $32 \times 32$  to  $128 \times 128$  pixels for AffGAN trained on natural images from ImageNET. For most of the images the results are sharp and corresponds well with the LR input. However we still see the high-frequency noise present in most GAN results in some of the images. Interestingly the snake depicted in the third column is super resolved into water which is obviously wrong but still a very plausible image considering the LR input image. Further, water will likely have a higher density under the image prior than snakes which suggests that the GAN model dreams up reasonable data.

![](images/12c2b97decdf6affcc1f683710bb311a2d8932e6ff8009665c1bc86296434719.jpg)  
Figure 5:  $4 \times$  SR from  $32 \times 32$  to  $128 \times 128$  using AffGAN on the ImageNET. AffGAN outputs (top row), true HR images  $y$  (middle row), model input  $x$  (bottom row). Generally the AffGAN produces plausible outputs which are however still easily distinguishable from true images. Interestingly the snake depicted in the third column is super resolved into water which is obviously wrong but still a very plausible image considering the LR input image.

# 5.6 CRITICISM AND FUTURE DIRECTIONS

One argument against MAP inference is that the mode of a distribution is dependent on the representation: transforming a variable through an invertible transformation and performing MAP inference in the transformed space may lead to different answers depending on the transformation. As an extreme example, consider transforming a continuous random scalar  $Y$  with its cumulative distribution function  $F = \mathbb{P}(Y \leq \cdot)$ . The resulting variable  $F(Y)$  is uniformly distributed, so any value in the interval  $(0, 1]$  can be the mode. Thus, the MAP estimate is not unique if one allows for alternative representations, and there is no guarantee that the MAP estimate in 24-bit RGB pixel representation which we seek in this paper is in any way special. One may arrive at a different solution when performing MAP estimation in the feature space of a convolutional neural network, or even if merely an alternative colour space is used. Interestingly, AffGAN is more resilient to coordinate transformations: Eqn. (10) includes the extra term  $\mathbb{H}[q_{\theta}]$  which is effected by transformations the same way as  $\mathbb{H}[q_{\theta}, p_{Y}]$ . The second argument relates to the assumption that MAP estimates appear plausible. Although by definition the mode lies in a high-probability region, it does not guarantee that its appearance is anything like that of a random sample. Consider for example data drawn from a  $d$ -dimensional standard Normal distribution. Due to concentration of measure, as  $d$  increases the norm of a typical sample will be approximately  $\sqrt{d}$  with very high probability. The mode, however, has a norm of 0. In this sense, the mode of the distribution is highly atypical. Indeed human observers can easily tell apart a typical sample from the noise distribution and the mode, but would have a hard time noticing the difference between two random samples. This argument suggests that sampling from the posterior  $p_{Y|X}$  may be a good or even preferable way to obtain plausible reconstructions. It is possible to extend AffGAN to perform approximate Bayesian inference by providing additional noise to the generator network, somewhat similarly to (Denton et al., 2015). In Appendix F we show that this stochastic version of AffGAN can be seen as performing amortized variational inference such as in Variational Autoencoders (Kingma & Welling, 2014).

# 6 CONCLUSION

In this work we developed methods for approximate MAP inference in SR. We first introduced an architectural restriction to neural networks projecting the model output to the affine subspace of valid solutions. We then proposed three methods, based on GANs, denoising or density models, for amortised MAP inference in SR using this affine projection. In high dimensions we empirically found that the GAN based approach, AffGAN produced the most visually appealing results. Our work follows successful demonstrations of GAN-based algorithms for image SR (Ledig et al., 2016), and we provide additional theoretical motivation for why this approach makes sense. In future work we plan to focus on a stochastic extension of AffGAN which can be seen as performing amortised variational inference.

# REFERENCES

Guillaume Alain and Yoshua Bengio. What regularized auto-encoders learn from the data-generating distribution. Journal of Machine Learning Research, 15(1):3563-3593, 2014.  
Joan Bruna, Pablo Sprechmann, and Yann LeCun. Super-resolution with deep convolutional sufficient statistics. International Conference on Learning Representations, 2016.  
Emily L Denton, Soumith Chintala, Rob Fergus, et al. Deep generative image models using a Laplacian Pyramid of adversarial networks. In Advances in Neural Information Processing Systems, pp. 1486-1494, 2015.  
Sander Dieleman, Jan Schlüter, Colin Raffel, Eben Olson, Søren Kaae Sønderby, Daniel Nouri, and Eric Battenberg and. Lasagne: First release., 2015. URL http://dx.doi.org/10.5281/zenodo.27878.  
Chao Dong, Chen Change Loy, Kaiming He, and Xiaou Tang. Image super-resolution using deep convolutional networks. IEEE Transactions on Pattern Analysis & Machine Intelligence, pp. 295-307, 2016.  
Alexey Dosovitskiy and Thomas Brox. Generating images with perceptual similarity metrics based on deep networks. arXiv preprint arXiv:1602.02644, 2016.  
David Garcia. Open source code. retrieved on 22 Sept 2016, 2016. URL https://github.com/david-gpu/srez.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pp. 2672-2680, 2014.  
Klaus Greff, Antti Rasmus, Mathias Berglund, Tele Hotloo Hao, Jürgen Schmidhuber, and Harri Valpola. Tagger: Deep unsupervised perceptual grouping. In Advances in Neural Information Processing Systems, 2016.  
Gao Huang, Zhuang Liu, and Kilian Q Weinberger. Densely connected convolutional networks. arXiv preprint arXiv:1608.06993, 2016.  
Ferenc Huszár. An alternative update rule for generative adversarial networks. Unpublished note (retrieved on 7 Oct 2016), 2016. URL http://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/.  
Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In The International Conference on Learning Representations, 2014.  
Valero Laparra, Jordi Mu noz Mari, and Jesus Malo. Divisive normalization image quality metric revisited. J. Opt. Soc. Am. A, pp. 852-864, 2010.  
Valero Laparra, Johannes Balle, Alexander Berardino, and Eero P Simoncelli. Perceptual image quality assessment using a normalized laplacian pyramid. In Proc. IS&T Int'l Symposium on Electronic Imaging, Conf. on Human Vision and Electronic Imaging, 2016.  
Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, and Ole Winther. Autoencoding beyond pixels using a learned similarity metric. In Proceedings of The 33rd International Conference on Machine Learning, pp. 1558-1566, 2015.  
Christian Ledig, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. Photo-realistic single image super-resolution using a generative adversarial network. arXiv preprint arXiv:1609.04802, 2016.  
Chuan Li and Michael Wand. Combining markov random fields and convolutional neural networks for image synthesis. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.  
Shakir Mohamed and Balaji Lakshminarayanan. Learning in implicit generative models. arXiv preprint arXiv:1610.03483, 2016.  
Kamal Nasrollahi and Thomas B. Moeslund. Super-resolution: a comprehensive survey. Machine Vision and Applications, pp. 1423-1468, 2014.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-GAN: Training generative neural samplers using variational divergence minimization. arXiv preprint arXiv:1606.00709, 2016.

Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In Proceedings of The 33rd International Conference on Machine Learning, pp. 1747-1756, 2016.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In International Conference on Learning Representations, 2015.  
Antti Rasmus, Mathias Berglund, Mikko Honkala, Harri Valpola, and Tapani Raiko. Semi-supervised learning with ladder networks. In Advances in Neural Information Processing Systems, pp. 3546-3554, 2015.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, 2016.  
Jaakko Särelä and Harri Valpola. Denoising source separation. Journal of Machine Learning Research, pp. 233-272, 2005.  
Wenzhe Shi, Jose Caballero, Ferenc Huszar, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1874-1883, 2016.  
The Theano Development Team, Rami Al-Rfou, Guillaume Alain, Amjad Almahairi, Christof Angermueller, Dzmitry Bahdanau, Nicolas Ballas, Frédéric Bastien, Justin Bayer, Anatoly Belikov, et al. Theano: A python framework for fast computation of mathematical expressions. arXiv preprint arXiv:1605.02688, 2016.  
Lucas Theis and Matthias Bethge. Generative image modeling using spatial lstms. In Advances in Neural Information Processing Systems, pp. 1927-1935, 2015.  
Lucas Theis, Reshad Hosseini, and Matthias Bethge. Mixtures of conditional gaussian scale mixtures applied to multiscale image representations. PLoS ONE, 2012.  
Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, pp. 1661-1674, 2011.  
Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th International Conference on Machine Learning, pp. 1096-1103, 2008.  
Zhou Wang, Eero P Simoncelli, and Alan C Bovik. Multiscale structural similarity for image quality assessment. In Conference Record of the 27th Asilomar Conference on Signals, Systems and Computers, volume 2, pp. 1398-1402, 2003.  
Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, pp. 600-612, 2004.
