# NEURAL PHOTO EDITING WITH INTROSPECTIVE ADVERSARIAL NETWORKS

Andrew Brock, Theodore Lim,& J.M. Ritchie

School of Engineering and Physical Sciences

Heriot-Watt University

Edinburgh, UK

{ajb5, t.lim, j.m.ritchie}@hw.ac.uk

Nick Weston

Renishaw plc

Research Ave, North

Edinburgh, UK

Nick.Weston@renishaw.com

# ABSTRACT

We present the Neural Photo Editor, an interface for exploring the latent space of generative image models and making large, semantically coherent changes to existing images. Our interface is powered by the Introspective Adversarial Network, a hybridization of the Generative Adversarial Network and the Variational Autoencoder designed for use in the editor. Our model makes use of a novel computational block based on dilated convolutions, and Orthogonal Regularization, a novel weight regularization method. We validate our model on CelebA, SVHN, and ImageNet, and produce samples and reconstructions with high visual fidelity.

# 1 INTRODUCTION

Recent advances in generative models for images have enabled the training of neural networks that produce image samples and interpolations with high visual fidelity. Two key methods, the Variational Autoencoder (VAE)(Kingma & Welling, 2014) and Generative Adversarial Network (GAN)(Goodfellow et al., 2014), have shown great promise for use in modeling the complex, high-dimensional distributions of natural images. VAEs are probabilistic graphical models that learn to maximize a variational lower bound on the likelihood of the data by projecting into a learned latent space, then reconstructing samples from that space. GANs learn a generative model by training one network, the "discriminator," to distinguish between real and generated data, while simultaneously training a second network, the "generator," to produce samples which the discriminator cannot distinguish from real data. Both approaches can be used to generate images by sampling in a low-dimensional learned latent space, but each comes with its own set of benefits and drawbacks.

VAEs have stable training dynamics, but when trained using elementwise  $\mathcal{L}_2$  distance as a reconstruction objective produce blurry images, as the learned model tends to be conservative in its predictions. Using the intermediate activations of a pre-trained discriminative neural network as features for comparing reconstructions to originals (Lamb et al., 2016) mollifies this effect, but requires labels in order to train the discriminative network in a supervised fashion.

By contrast, GANs have unstable and often oscillatory training dynamics, but produce images with sharper, more photorealistic features. Basic GANs lack an inference mechanism, though techniques to adversarially train an inference network (Dumoulin et al., 2016) (Donahue et al., 2016) have recently been developed, as well as a hybridization that uses the VAE's inference network (Larsen et al., 2015).

Standard procedure for evaluating these models involves generating random samples or reconstructions, and interpolating between generated images. Achieving a specific change in the model output, such as changing an unsmiling face to a smiling face, usually requires that the learned latent space be augmented during training with a set of labeled attributes, such that interpolating along a latent (e.g. "smile") vector produces a specific change. In the fully unsupervised setting, there is no guarantee that a particular latent variable will directly control a semantically meaningful output feature.

In this paper, we present the Neural Photo Editor, a novel interface for exploring the latent space of generative models. Our method makes it possible to produce specific semantic changes in the output image by use of a "contextual paintbrush" that indirectly modifies the latent vector, even for highly

![](images/356aa6a4497d4c1ee60b8fe9317db201787136144dea0be25db08cbbce4ed204.jpg)  
Figure 1: The Neural Photo Editor. The original image is center. The red and blue tiles are visualizations of the latent space, and can be directly manipulated as well.

![](images/c109cce2fbfb7a99f968e2e8694fb07e043fe9244108cae9e99eabe6778e1d9a.jpg)

![](images/779eb9f5342168accc05b2934ba847beae57684c558a52e164255ab82f12d9e0.jpg)

entangled latent vectors. By applying a simple interpolating mask, we enable this same exploration for existing photos despite errors in the model's reconstruction.

Complementary to the Neural Photo Editor, we present the Introspective Adversarial Network (IAN), a hybridization of the VAE and the GAN that leverages the power of the adversarial objective while maintaining the efficient inference mechanism of the VAE. Our model makes use of a novel inception-style convolutional block based on dilated convolutions (Yu & Koltun, 2016) and Orthogonal Regularization, a novel technique for regularizing weights in convolutional neural networks. Qualitative experiments on CelebA (Liu et al., 2015), SVHN (Netzer et al., 2011) and Imagenet(Russakovsky et al., 2015) demonstrate the sampling, reconstructing, and interpolating capabilities of the IAN, while competitive performance on the semi-supervised SVHN classification task quantitatively demonstrates its inference capabilities.

# 2 NEURAL PHOTO EDITING

Standard methods for exploring the latent space of a generative model involve interpolating between two samples or directly manipulating the latent space. The latter works well when the network is provided descriptive labels during training, but is far less effective when the network is trained in a wholly unsupervised fashion, as there is no guarantee that individual latent vectors will correspond to semantically meaningful features.

We present an interface, shown in Figure 1, that allows for a more intuitive exploration of a generative model by indirectly manipulating the latent space with a "contextual paintbrush." The key idea is simple: a user selects a paint brush size and color (as with a typical image editor) and paints on the output image. Instead of changing individual pixels, the interface backpropagates the difference between the local image patch and the requested color, and takes a gradient descent step in the latent space to minimize that difference. This step results in globally coherent changes that are semantically meaningful in the context of the requested color change.

For example, if a user has an image of a person with light skin, dark hair, and a widow's peak, by painting a dark color on the forehead, the system will automatically add hair in the requested area. Similarly, if a user has a photo of a person with a closed-mouth smile, the user can produce a toothy grin by painting bright white over the target's mouth. This method is non-iterative in the sense that a single gradient descent step is taken every time the user requests a change, and runs smoothly in real-time on a modest laptop GPU.

This technique enables exploration of samples generated by the network, but fails when applied directly to existing photos, as it relies on the manipulated image being completely controlled by the latent variables. Reconstructing images that have passed through a representational bottleneck (i.e.

![](images/50738efa4c155f955c28a761a3d0324dae4f170ffb8d710ea6d0b62322504042.jpg)

![](images/d9bc1bba9ab712c422542bd9636ef0b254698ff3411facdc72a5fdeb26ebf8d0.jpg)

![](images/72a400aff1b3c7eff7dfc3b263968f019dbcbc867d3d44c3a7b69c196a550802.jpg)

![](images/10559fddea9f7c0eefa742134f7ec67d4ce2437acb3b57b0293acc0eea3c1771.jpg)  
Figure 2: Visualizing the interpolation mask. Top, left to right: Reconstruction, reconstruction error, original image. Bottom: Modified reconstruction,  $\Delta$ , output.

![](images/eed88762c2d5f379532ae94d9ef9e00fe87b5e6e067862f9ca8c77fb1f4bc105.jpg)

![](images/f02c14c64f5de32d2a90195e3e3ea78899d71dfbb42f9c5b39a1df3c204b40a6.jpg)

with an autoencoder) is difficult, and certain to produce reconstructions which, lacking pixel-perfect accuracy, are useless for making small changes to natural images.

To circumvent this, we introduce a simple masking technique that allows a user to edit images by modifying a given photo's reconstruction, then transferring those changes to the original image. We take the output image to be a sum of the reconstruction, and a masked combination of the requested pixel-wise changes and the reconstruction error:

$$
Y = \hat {X} + M \Delta + (1 - M) (X - \hat {X})
$$

Where  $X$  is the original image,  $\hat{X}$  is the model's reconstruction of  $X$ , and  $\Delta$  is the difference between the modified reconstruction and  $\hat{X}$ . The mask  $M$  is the channel-wise mean of the absolute value of  $\Delta$ , smoothed with a Gaussian filter  $g$  and truncated to be between 0 and 1:

$$
M = \min  (g (| \bar {\Delta} |), 1)
$$

A visualization of the masking technique is shown in Figure 2. The mask is designed to allow changes to the reconstruction to show through based on the magnitude of those changes. As such, the system will successfully transfer changes so long as the reconstruction is aligned with the original image and the changes are smooth and plausible. This method adds minimal computational cost to the underlying latent space exploration and produces convincing changes of features including hair color and style, skin tone, and facial expression. A video of the interface in action is available online. $^1$

# 3 INTROSPECTIVE ADVERSARIAL NETWORKS

Complementary to the Neural Photo Editor, we introduce the Introspective Adversarial Network (IAN), a novel hybridization of the VAE and GAN motivated by the need for an image model with photorealistic outputs that learns descriptive features while still achieving high-quality reconstructions. There is typically a design tradeoff between these two goals related to the size of the latent space: a higher-dimensional latent space (i.e. a wider representational bottleneck) tends to learn less descriptive features (producing less smooth interpolations) but produce higher quality reconstructions, as its effective compression ratio is lower.

We thus seek techniques to improve the capacity of the latent space without increasing its dimensionality. Similar to VAE/GAN (Larsen et al., 2015), we use the decoder network of the autoencoder

![](images/da390025d64885139f08ae2fd2898a8a04f0563e6730eaef07f82214f69c1379.jpg)  
Figure 3: The Introspective Adversarial Network (IAN).

as the generator network of the GAN, but instead of training a separate discriminator network, we combine the encoder and discriminator into a single network. Similar also to DeePSiM (Dosovitskiy & Brox, 2016), we use three distinct loss functions:

-  $\mathcal{L}_1$  pixel-wise reconstruction loss, which we prefer to the  $\mathcal{L}_2$  reconstruction loss for its higher average gradient.  
- Feature-wise reconstruction loss, evaluated as the  $\mathcal{L}_2$  difference between the original and reconstruction in the space of the hidden layers of the discriminator.  
- Ternary Adversarial Loss, a modification of the Adversarial loss that forces the discriminator to label a sample as real, generated, or reconstructed (as opposed to a binary real vs. generated label).

Central to the IAN is the idea that features learned by a discriminatively trained network tend to be more expressive those learned by an encoder network trained via maximum likelihood (i.e. more useful on semi-supervised tasks), and thus better suited for inference. As the Neural Photo Editor relies on high-quality reconstructions, the inference capacity of the underlying model is critical. Accordingly, we use the discriminator of the GAN as a feature extractor for an inference subnetwork, which isimplemented as a fully-connected layer on top of the final convolutional layer of the discriminator.

The discriminator is updated solely using the ternary adversarial loss. The encoder subnetwork and generator are trained to produce random images (from  $Z_{rand}$ ) which the discriminator cannot distinguish from real images  $X$ , as well as reconstructions  $\hat{X}$  (from  $\hat{Z}_{IAF}$ ) which are simultaneously photorealistic, and similar to the original image in the pixel space and the space of the intermediate activations of the discriminator  $(f(X))$ . The IAN architecture is depicted in Figure 3.

The loss functions for each network thus become:

$$
\mathcal {L} _ {G} = \log (D (X | G (Z _ {\text {r a n d}}))) + \log (D (X | \hat {X})) + \| X - \hat {X} \| _ {1} + \| f (X) - f (\hat {X}) \| _ {2}
$$

$$
\mathcal {L} _ {D} = \left(1 - \log \left(D \left(G \left(Z _ {\text {r a n d}}\right) \mid G \left(Z _ {\text {r a n d}}\right)\right)\right)\right) + \left(1 - \log \left(D \left(\hat {X} \mid \hat {X}\right)\right)\right) + \log (D (X | X))
$$

Where  $D(A|B)$  indicates the probability of class  $A$  assigned by the discriminator to an input  $B$ .

# 3.1 FEATURE-WISE AND TERNARY ADVERSARIAL LOSS

Comparing the outputs of intermediate encoder/discriminator layers was originally inspired by Discriminative Regularization (Lamb et al., 2016), though we note that Feature Matching (Salimans et al., 2016) is designed to operate in a similar fashion, but without the guidance of an inference mechanism to match latent values  $Z$  to particular values of  $f(G(Z))$ . We find that using this loss to

complement the pixel-wise difference results in sharper reconstructions that better preserve higher frequency features and edges.

We note that it is possible for the discriminator to get an early lead on the generator and slow training, perhaps because it learns a small subset of features (artifacts in the generator's output) that distinguish real and generated samples, reducing the range of features the generator can learn from the discriminator. The ternary adversarial loss, where the discriminator attempts to assign one of three labels to a sample, presents a more difficult task for the discriminator, and reduces the likelihood of it learning such a small feature space by forcing it to distinguish between reconstructions and random samples. We posit that this also leads to the discriminator ultimately learning a richer feature space, contributing to consistent sample quality.

We additionally experiment with using the style loss of neural style transfer (Gatys et al., 2015) for reconstruction comparison, and find that it improves early training but comes at a prohibitive memory and computational cost, and do not make use of it.

# 3.2 IAF WITH RANDOMIZED MADE

We take advantage of the auto-encoding nature of our architecture and implement the MADE (Germain et al., 2015) variant of Inverse Autoregressive Flow (IAF) (Kingma et al., 2016). IAF improves the flexibility of the model's approximate posterior over the latents by introducing nonlinear dependencies between elements of  $\hat{Z}$  through an autoregressive whitening procedure. Adding these dependencies can be seen as adding nonlinear off-diagonal elements to the covariance matrix of the posterior approximation (which would otherwise be diagonal), which we hypothesize will improve the capacity of the latent space and therefore the quality of reconstructions.

In initial experiments, we found that implementing full MADE (including shuffling the ordering and connectivity masks) significantly reduced results quality, perhaps because the shuffling injected an undesirable amount of internal covariate shift. We found that using only a single initial shuffle and orthogonally initializing (Saxe et al., 2014) but not training the MADE worked best, suggesting that IAF whitening can be performed using any random autoregressive function of the latents. IANs trained with this form of IAF have higher pixel-wise reconstruction accuracy on held-out data, and produce reconstructions with higher visual quality, as shown in Figure 4(b).

# 3.3 ARCHITECTURE

Our model has the same basic structure as DCGAN (Radford et al., 2015), augmented with Residual (He et al., 2016) Multiscale Dilated Convolution (MDC) blocks between successive upsampling layers in the generator, an autoregressive RGB block at the output of the generator, and Minibatch Discrimination (Salimans et al., 2016) in the discriminator. We found that using Batch Normalization (Ioffe & Szegedy, 2015) and Adam (Kingma & Ba, 2014) were essential to successfully training IANs. Code containing all experiments and exact architectural details is available online.

We found that when designing IANs, maintaining the "balance of power" between the generator and the discriminator to be key. In particular, we found that if we made the discriminator too expressive (i.e. by inserting MDC blocks in between downsampling layers) it would quickly out-learn the generator and achieve near-perfect accuracy, resulting in a significant slow-down in training. We thus maintain an "improvement ratio" rule of thumb, where every layer we add to the discriminator is accompanied by an addition of three layers in the generator.

We put an especial focus on representational bottlenecks in our design. Just as aggressive down-sampling in the early layers of a classification network can hamper performance, we posit that aggressive upsampling in the final layers of the generator can reduce performance by hampering the backpropagation of error. This supposition guides us to add more expressivity (by means of additional MDC blocks) in the later layers of the generator compared to the early layers.

![](images/adfffa1892a1100c00f8e864c30d99a0ce25adfb79aa9691aa2c1297e741f5e1.jpg)  
(a)

![](images/16a46c00baa5330bf605e78f6f876b6e6d29b167f4e97aa2e40f95bac23402f5.jpg)

![](images/67f8716d3458977065a9c0a1ef64785faa1066e9201065080d65eb6b09d6810d.jpg)

![](images/914009aca00fc8d0bed2b41fe4d95de7db08fbb1cc6f07e712a5b206089796ec.jpg)

![](images/0aaec952fe3ff70ac67a5691344a5c48fc3a41d89174c56c0e8ce142ec48ce1f.jpg)  
Figure 4: (a) Multiscale Dilated Convolution Block, (b) IAF comparisons. Left: Original, Middle: Reconstruction with IAF, Right: Reconstruction without IAF.

![](images/058f1f073c9a8c77d5eb7ce6cddaa2234a6d4d2355f1df7375819c38aabd2426.jpg)  
(b)

![](images/be05bfa105d5b941d5bb8fb2c7ff04dcf64405ca4b818ba840e8548b33640e92.jpg)

# 3.3.1 MULTISCALE DILATED CONVOLUTION BLOCKS

We propose a novel Inception-style (Szegedy et al., 2016) convolutional block motivated by the ideas that image features naturally occur at multiple scales, and that a network's expressivity is proportional to the range of functions it can represent divided by its total number of parameters. The Multiscale Dilated Convolution (MDC) block applies a single  $3 \times 3$  filter at multiple dilation factors, then performs a weighted elementwise sum of each dilated filter's output, allowing the network to simultaneously learn a set of features and the relevant scales at which those features occur with a minimal increase in parameters.

As shown in Figure 4(a), each block is parameterized by a bank of N 3x3 filters  $W$ , applied with S different factors of dilation, a 1x1 convolution with weights taken as the mean of the main 3x3 filter, and a set of  $\mathrm{N}^* (\mathrm{S} + 1)$  weights  $k$ , which relatively weight the output of each filter at each scale.

We performed initial explorations by integrating MDC blocks in a 16-layer ResNet for CIFAR-100 (Krizhevsky & Hinton, 2009), and found that the network converged to  $71\%$  test accuracy within just 10 epochs,  $3\%$  higher than an identical network using only 3x3 convolutions despite a mere  $0.1\%$  increase in parameters. Though this is not a new state-of-the-art, we noted the high performance-to-depth ratio and the fast training time, and immediately integrated the blocks into our architecture, leaving full discriminative validation to future work.

# 3.3.2 RGB MODELING

Typically, the output layer of an image-generating neural network consists of 3 HxW channels corresponding to the R-G-B color channels, which are each fed to a squashing nonlinearity and mapped to an 8-bit color scheme.

PixelRNNs(van den Oord et al., 2016) changed this by allowing the network to specify a 256-dimensional discrete distribution, with each dimension corresponding to a unique unsigned 8-bit value, reasoning that allowing the network to specify a flexible discrete color distribution would improve image quality. We adopt a similar view, but note that outputting a  $3^{*}\mathrm{H}^{*}\mathrm{W}^{*}256$  dimensional output from a convolutional layer is computationally expensive. Instead, we reason that we need not specify a full discrete distribution, but can instead output the shape parameters of a flexible continuous PDF, such as a beta distribution, which then allows us to impose priors on the desired shape of that distribution, rather than having the network concentrate all of its probability mass on a single point. In practice, we find that training the network to specify a beta distribution rather than a single point works well even without enforcing a prior on the shape of the distribution, reducing the likelihood of "washed-out" samples with poor contrast and improving the fidelity of reconstructions.

As in PixelRNNs, we autoregressively specify the color channels, meaning that the R channel is dependent on the output of the last hidden layer, the G channel is dependent on the last hidden layer and the R channel, and the B channel is dependent on the last hidden layer and both the R and G channels.

![](images/66cd34f786aba48ed142a04f2b03324673fa597ea04dcd18f297a95e02f6c9cb.jpg)  
Figure 5: CelebA, ImageNet, and SVHN samples.

![](images/b02cac350c68f6e71e83d1ce77579782f8339e924fab9b1680d45e345e850d93.jpg)  
Figure 6: CelebA, ImageNet, and SVHN Reconstructions and Interpolations. The outermost images are originals, the adjacent images are reconstructions.

![](images/b2134fe999d39a16775ae1621602c47b29e6979e8d9d7fea1a6fabc2d96b6a25.jpg)

![](images/57948da7de86811f0d77d494f954bcd27622611796ce16072e7c883ffd8105d5.jpg)

![](images/6ee90e6a5ec482397ecb5ef480d50fe28f17f2522692a285e73dd0625919f193.jpg)

![](images/9115761695a1c736e3e99c449b8d29df4822ad0f1077c4435c90d953ca4bac2d.jpg)

![](images/f2344e1de65a3aafdb9fe69f2eab914210cbf867f9681c321e47fa63376e4b2e.jpg)

![](images/ff994625916cbe410a41c711e5937fdb55ba5d343d95919bc7dd98c7065ebc09.jpg)  
Figure 7: Evaluating proposed modifications for reconstruction. From left to right: Original image, full model, model with binary loss, model with binary loss and no orthogonal regularization, model with binary loss, no orthogonal regularization, and no MDC or RGB blocks.

![](images/b79740bd84d1e9cae0f1fdc28d1b51f2e2f89c05e8fe0abf68b11b05cbc92439.jpg)

![](images/204a639a4aa624458874cc74c2f3db4c77fa061381b286c5369bf0e544b99099.jpg)

![](images/7fe1cf54e62ff8928fb5a97a54b1fb0a645b3937f84c8027084ee416898555e8.jpg)

![](images/83b26182c802881e070635cf56cd07a23ec6a8dd89566e915fb3c69fc13ae2d9.jpg)

# 3.4 ORTHOGONAL REGULARIZATION

Orthogonality is a desirable quality in ConvNet filters, partially because multiplication by an orthogonal matrix leaves the norm of the original matrix unchanged. This property is valuable in deep or recurrent networks, where repeated matrix multiplication can result in signals vanishing or exploding. We note the success of initializing weights with orthogonal matrices (Saxe et al., 2014), and posit that maintaining orthogonality throughout training is also desirable. To this end, we propose a simple weight regularization technique, Orthogonal Regularization, that encourages weights to be orthogonal by pushing them towards the nearest orthogonal manifold. We add a cost term to our objective function:

$$
\mathcal {L} _ {\text {o r t h o}} = \Sigma (| W W ^ {T} - I |)
$$

Where  $\Sigma$  indicates a sum across all filter banks,  $W$  is a bank of convolution kernels and  $I$  is the identity matrix.

We found that applying Orthogonal Regularization to our CIFAR-100 testbed immediately improved test accuracy by  $2\%$  without any other changes, and we thus make use of it in all of our experiments. We also experimented with a more extreme form of regularization by reparameterizing the filters as 3D rotation matrices described by three learned rotation angles per filter, resulting in guaranteed orthogonality and a  $66\%$  reduction in the number of parameters in each convolutional layer. We find this parameterization to significantly reduce model performance, and thus stick to encouraging orthogonality through regularization, rather than enforcing it through reparameterization.

# 4 EXPERIMENTS

We qualitatively evaluated the IAN on 64x64 CelebA (Liu et al., 2015), 32x32 SVHN (Netzer et al., 2011) and 128x128 Imagenet(Russakovsky et al., 2015). Our models are implemented in Theano (Team, 2016) with Lasagne (Dieleman et al., 2015). Samples from the IAN, shown in Figure 5, display the visual fidelity typical of adversarially trained networks. The IAN demonstrates high quality reconstructions, shown in Figure 6, and smooth, plausible interpolations, even between drastically different samples.

# 4.1 EVALUATING MODIFICATIONS

Quantitatively evaluating the suitability of a model for use with the Neural Photo Editor is difficult, and we rely on qualitative visual inspection to determine whether our proposed modifications improve results. For use in our application, reconstructions must be photorealistic, align facial landmarks, and separately reconstruct individual facial features accurately.

Reconstructions across a variety of model configurations are compared in Figure 7. While all reconstructions are, in some sense, semantically similar to the original (and no reconstructions are

<table><tr><td>Method</td><td>Error rate</td></tr><tr><td>KNN (as reported in Zhao et al. (2015))</td><td>77.93%</td></tr><tr><td>TSVM (Vapnik, 1998)</td><td>66.55%</td></tr><tr><td>VAE (M1 + M2) (Kingma et al., 2014)</td><td>36.02%</td></tr><tr><td>SWWAE without dropout (Zhao et al., 2015)</td><td>27.83%</td></tr><tr><td>SWWAE with dropout (Zhao et al., 2015)</td><td>23.56%</td></tr><tr><td>DCGAN + L2-SVM (Radford et al., 2015)</td><td>22.18%(±1.13%)</td></tr><tr><td>ALI (Dumoulin et al., 2016)</td><td>19.14%(±0.50%)</td></tr><tr><td>SDGM (Maaløe et al., 2016)</td><td>16.61%(±0.24%)</td></tr><tr><td>Improved-GAN (Salimans et al., 2016)</td><td>8.11%(±1.3%)</td></tr><tr><td>IAN (ours)</td><td>18.50%(±0.38%)</td></tr></table>

Table 1: Error rates on Semi-Supervised SVHN with 1000 training examples.

identical), the quality of reconstructions visibly increases from right to left, with the progressive addition of modifications. The pixel-wise reconstruction accuracy (both on these images and on average across a held-out validation set) also increases from right to left, with the exception of the far right entries (no modifications) which have higher pixel-wise accuracy than their immediate neighbors.

Examining adjacent reconstruction pairs reveals trends for each modification. Switching from ternary to binary adversarial loss results in reduction of the alignment of facial features, clearly shown in the hairline and pose of the upper reconstruction. Removing orthogonal regularization reduces similarity on a number of facial landmarks, most notably in the eyes and the upper reconstruction's expression. Removing the MDC and RGB blocks results in a general reduction in both reconstruction fidelity and visual quality. Note that for these experiments we only use MDC and RGB blocks in the final layer of the generator, so the change in visual quality is not due to an increase in the number of layers.

# 4.2 SEMI-SUPERVISED LEARNING WITH SVHN

We quantitatively evaluate the inference abilities of our architecture by applying it to the semi-supervised SVHN classification task. Our procedure follows that of (Radford et al., 2015) and (Dumoulin et al., 2016): We train an L2-SVM to classify SVHN data, using the output of the fully-connected layer of the encoder subnetwork as input features to the SVM. We report the average test error and standard deviation across 100 different SVMs, each trained on 1000 random examples from the training set. Our performance, as shown in Table 1, is competitive with other networks evaluated in this fashion, achieving  $18.5\%$  mean classification accuracy.

We note that the Improved-GAN (Salimans et al., 2016) architecture is evaluated in a different manner, where the discriminator's objective is directly augmented with a classification objective. We evaluated IAN using the architecture and training regime from Improved-GAN and achieved comparable results (8% error), though as our modifications did not improve results over feature-matching-based Improved-GAN we do not claim this with our main results.

# 5 CONCLUSION

We introduced the Neural Photo Editor, a novel interface for exploring the learned latent space of generative models and for making specific semantic changes to natural images. Our interface makes use of the Introspective Adversarial Network, a hybridization of the VAE and GAN that outputs high fidelity samples and reconstructions, and achieves competitive performance in a semi-supervised classification task. The IAN makes use of Multiscale Dilated Convolution Blocks and Orthogonal Regularization, two improvements designed to improve model expressivity and training stability for adversarial networks.

# ACKNOWLEDGMENTS

This research was made possible by grants and support from Renishaw plc and the Edinburgh Centre For Robotics. The work presented herein is also partially funded under the European H2020 Programme BEACONING project, Grant Agreement nr. 687676.

# REFERENCES

S. Dieleman, J. Schlüter, C. Raffel, E. Olson, S.K. Sønderby, D. Nouri, and E. Battenberg. Lasagne: First release., 2015. URL http://dx.doi.org/10.5281/zenodo.27878.  
Jeff Donahue, Philipp Krahenbuhl, and Trevor Darrell. Adversarial feature learning. arXiv preprint arXiv:1605.09782, 2016.  
A. Dosovitskiy and T. Brox. Generating images with perceptual similarity metrics based on deep networks. arXiv Preprint arXiv:1602.02644, 2016.  
V. Dumoulin, I. Belghazi, B. Poole, A. Lamb, M. Arjovsky, O. Mastropietro, and A. Courville. Adversarily learned inference. arXiv Preprint arXiv: 1606.0070, 2016.  
L.A. Gatys, A.S. Ecker, and M. Bethge. A neural algorithm of artistic style. arXiv Preprint arXiv: 1508.06576, 2015.  
M. Germain, K. Gregor, I. Murray, and H Larochelle. Made: Masked autoencoder for distribution estimation. arXiv Preprint arXiv: 1502.03509, 2015.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pp. 2672-2680, 2014.  
K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. arXiv Preprint arXiv: 1603.05027, 2016.  
S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML 2015, 2015.  
Diederik P Kingma, Shakir Mohamed, Danilo Jimenez Rezende, and Max Welling. Semi-supervised learning with deep generative models. In Advances in Neural Information Processing Systems, pp. 3581-3589, 2014.  
D.P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv Preprint arXiv: 1412.6980, 2014.  
D.P. Kingma and M. Welling. Auto-encoding variational bayes. In *ICLR* 2014, 2014.  
D.P. Kingma, T. Salimans, and M. Welling. Improving variational inference with inverse autoregressive flow. arXiv Preprint arXiv: 1606.04934, 2016.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images, 2009.  
Alex Lamb, Vincent Dumoulin, and Aaron Courville. Discriminative regularization for generative models. arXiv preprint arXiv:1602.03220, 2016.  
Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, and Ole Winther. Autoencoding beyond pixels using a learned similarity metric. arXiv preprint arXiv:1512.09300, 2015.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision, pp. 3730-3738, 2015.  
Lars Maaloe, Casper Kaae Sønderby, Søren Kaae Sønderby, and Ole Winther. Auxiliary deep generative models. arXiv preprint arXiv:1602.05473, 2016.  
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervised feature learning, volume 2011, pp. 4. Granada, Spain, 2011.

Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211-252, 2015.  
T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training gans. arXiv Preprint arXiv: 1606.03498, 2016.  
A.M. Saxe, J. L. McClelland, and S. Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. In ICLR 2014, 2014.  
C. Szegedy, S. Ioffe, and V. Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. arXiv Preprint arXiv: 1602.07261, 2016.  
The Theano Development Team. Theano: A python framework for fast computation of mathematical expressions. arXiv Preprint arXiv: 1605.02688, 2016.  
A. van den Oord, N. Kalchbrenner, and K. Kavukcuoglu. Pixel recurrent neural networks. arXiv Preprint arXiv: 1601.06759, 2016.  
Vladimir N. Vapnik. Statistical Learning Theory. Wiley-Interscience, 1998.  
F. Yu and V. Koltun. Multi-scale context aggregation by dilated convolutions. In ICLR 2016, 2016.  
Junbo Zhao, Michael Mathieu, Ross Goroshin, and Yann Lecun. Stacked what-where auto-encoders. arXiv preprint arXiv:1506.02351, 2015.