# HOW MANY DATA ARE AUGMENTATIONS WORTH?

# AN INVESTIGATION INTO SCALING LAWS, INVARIANCE, AND IMPLICIT REGULARIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Despite the clear performance benefits of data augmentations, little is known about why they are so effective. In this paper, we disentangle several key mechanisms through which data augmentations operate. Establishing an exchange rate between augmented and additional real data, we find that in out-of-distribution testing scenarios, augmentations which yield samples that are diverse, but inconsistent with the data distribution can be even more valuable than additional training data. Moreover, we find that data augmentations which encourage invariances can be more valuable than invariance alone, especially on small and medium sized training sets. Following this observation, we show that augmentations induce additional stochasticity during training, effectively flattening the loss landscape.

# 1 INTRODUCTION

Even with the proliferation of large-scale image datasets, deep neural networks for computer vision represent highly flexible model families and often contain orders of magnitude more parameters than the size of their training sets. As a result, large models trained on limited datasets still have the capacity for improvement. To make up for this data shortage, standard operating procedure involves diversifying training data by augmenting samples with randomly applied transformations that preserve semantic content. These augmented samples expand the volume of data available for training, resulting in downstream performance benefits that one might expect from a larger dataset. However, the now profound significance of data augmentation (DA) for boosting performance suggests that its benefits may be more nuanced than previously believed.

In addition to adding extra samples, augmentation promotes invariance by encouraging models to make consistent predictions across augmented views of each sample. The need to incorporate invariances in neural networks has motivated the development of architectures that are explicitly constrained to be equivariant to transformations (Weiler & Cesa, 2019; Finzi et al., 2020). If the downstream effects of data augmentations were attributable solely to invariance, then we could replace DA with explicit model constraints. However, if explicit constraints cannot replicate the benefits of augmentation, then augmentations may affect training dynamics beyond imposing constraints.

Finally, augmentation may improve training by serving as an extra source of stochasticity. Under DA, randomization during training comes not only from randomly selecting samples from the dataset to form batches but also from sampling transformations with which to augment data (Fort et al., 2022). Stochastic optimization is associated with benefits in non-convex problems wherein the optimizer can bias parameters towards flatter minima (Jastrzewski et al., 2017; Geiping et al., 2021; Liu et al., 2021a).

In this paper, we re-examine the role of data augmentation. In particular, we quantify the effects of data augmentation in expanding available training data, promoting invariance, and acting as a source of stochasticity during training. In summary:

- We quantify the relationship between augmented views of training samples and extra data, evaluating the benefits of augmentations and the number of samples rises. We find that augmentations can confer comparable benefits to independently drawn samples on in-domain test sets and even stronger benefits on out-of-distribution testing.

- We observe that models that learn invariances via data augmentation provide additional regularization compared to invariant architectures and we show that invariances that are uncharacteristic of the data distribution still benefit performance.  
- We then clarify the regularization benefits gained from augmentations through measurements of flatness and gradient noise showing how DA exhibits flatness-seeking behavior.

# 2 RELATED WORK

Data Augmentations in Computer Vision. Data augmentations have been a staple of deep learning, used to deform handwritten digits as early as Yaeger et al. (1996) and LeCun et al. (1998), or to improve oversampling on class-imbalanced datasets (Chawla et al., 2002). These early works hypothesize that data augmentations are necessary to prevent overfitting when training neural networks since they typically contain many more parameters than training data points (LeCun et al., 1998).

We restrict our study to augmentations which act on a single sample and do not modify labels. Namely, we study augmentations which can be written as  $(T(\mathbf{x}),y)$ , where  $(\mathbf{x},y)$  denotes an input-label pair, and  $T \sim T$  is a random transformation sampled from a distribution of such transformations. For a broad and thorough discussion on image augmentations, their categorization, and applications to computer vision, see Shorten & Khoshgoftaar (2019) and Xu et al. (2022). We consider basic geometric (random crops, flips, perspective) and photometric (jitter, blur, contrast) transformations, and common augmentation policies, such as AutoAug (Cubuk et al., 2019a), RandAug (Cubuk et al., 2019b), AugMix (Hendrycks et al., 2020) and TrivialAug (Müller & Hutter, 2021) which combine basic augmentations.

Understanding the Role of Augmentation and Invariance. Works such as Hernández-García & König (2018) propose that data augmentations (DA) induce implicit regularization. Empirical evaluations describe useful augmentations as "label preserving", namely they do not significantly change the conditional probability over labels (Taylor & Nitschke, 2018). Gontijo-Lopes et al. (2020b;a) investigate empirical notions of consistency and diversity. They measure consistency (referred to as affinity) as the performance of models trained without augmentation on augmented validation sets. They also measure diversity as the ratio of training loss of a model trained with and without augmentations and conclude that strong data augmentations should be both consistent and diverse, an effect also seen in Kim et al. (2021). In contrast to Gontijo-Lopes et al. (2020b), Marcu & Prügel-Bennett (2021) find that the value of data augmentations cannot be measured by how much they deform the data distribution. Other work proposes to learn invariances parameterized as augmentations from the data (Benton et al., 2020), investigates the number of samples required to learn an invariance (Balestriero et al., 2022b), uncovers the tendency of augmentations to sacrifice performance on some classes in exchange for gains on others (Balestriero et al., 2022a), or argues that data augmentations cause models to misrepresent uncertainty (Kapoor et al., 2022).

Theoretical investigations in Chen et al. (2020) formalize data augmentations as label-preserving group actions and discuss an inherent invariance-variance trade-off. Variance regularization also arises when modeling augmentations for kernel classifiers (Dao et al., 2019). For a binary classifier with finite VC dimension, the bound on expected risk can be reduced through additional data generated via augmentations until inconsistency between augmented and real data distributions overwhelms would-be gains (He et al., 2019b). The regularizing effect of data augmentations is investigated in LeJeune et al. (2019) who propose a model under which continuous augmentations increase the smoothness of neural network decision boundaries. Rajput et al. (2019) similarly find that linear classifiers trained with sufficient augmentations can approximate the maximum margin solution. Hanin & Sun (2021) relate data augmentations to stochastic optimization. A different angle towards understanding invariances through data augmentations is presented in Zhu et al. (2021), where the effect of DA in increasing the theoretical sample cover of the distribution is investigated, and augmentations can reduce the amount of data required, if they "cover" the real distribution.

Stochastic Optimization and Neural Network Training. The implicit regularization of SGD is regarded as an essential component for neural network generalization (An, 1996; Neyshabur et al., 2017). Stochastic training which randomizes gradients can drive parameters into flatter minima, associated with superior generalization (Jastrzebski et al., 2017; Huang et al., 2020; Liu et al., 2021a). In fact, Geiping et al. (2021) find that neural networks trained with non-stochastic full batch gradient descent require explicit flatness-seeking regularizers in order to achieve comparable test accuracy.

![](images/c248cec9d805d3d20ad401cdfbfd28a74533eee4b4a640901e96e34c2140b5b1.jpg)  
Figure 1: Power laws  $f(x) = ax^{-c} + b$  for select augmentations applied randomly and the gain in terms of effective extra samples from Equation (1). Fitted curves marked in solid colors, with extrapolated regions dashed. Left: Number of base samples (from CINIC-10) on the logarithmic horizontal axis compared to validation accuracy. The scaling behavior of each augmentation is closely matched by these power laws. Right: Number of base samples compared to effective extra data, showing how the benefits of each data augmentation scale as the model is trained on more and more data. Policies that are strong but inconsistent, such as TrivialAug, reach the highest peak benefit (at 50000 base samples TrivialAug generates effectively 100000 extra samples), but also fall off faster than consistent augmentations, such as horizontal flips, which provide benefits up to 350000 base samples.

![](images/6c47d1f3048b9bd8c0a03f2799d74152f50390667ac801960b024b77b282e44b.jpg)

Data augmentations provide an additional source of stochasticity during training on top of batch sampling, which we will investigate in this work.

We fuse together the above three topics and explore the role of data augmentations play in learning invariance and in increasing stochasticity during late stages of training. In doing so, we fill in several gaps in the literature discussed in this section. Unlike other works which measure the effectiveness of data augmentations in terms of accuracy boosts for fixed sample sizes, we compare the benefits of augmentations to those achieved by instead collecting more data. While other works have studied the role of data augmentations in learning invariance, we find that even invariances which have no relationship to invariances in the training data distribution are still effective. Finally, we develop an understanding of batch augmentation by showing that stochastically applied augmentations increase gradient noise during training, leading to qualitatively distinct minima.

# 3 AUGMENTATIONS AS ADDITIONAL DATA

A central role of data augmentation is to serve as extra data and expand limited datasets used for training large models. In this section, we quantify this property, conducting a series of experiments culminating in measurements exchange rates, which indicate exactly how much data an augmentation is worth – the number of additional data samples which would yield the same performance gain as the augmentation. Such exchange rates constitute a novel angle for quantifying the practical benefits of data augmentations and conceptualize qualitative notions of consistency, diversity, and robustness to distribution shifts, and allow us to observe the properties of data augmentations over a wider range of dataset sizes. We conduct these experiments on subsets of the CINIC-10 dataset (Darlow et al., 2018), a drop-in replacement for CIFAR-10 (Krizhevsky, 2009), which contains  $\sim 200000$  samples. This setup allows us to train models with augmented data on dataset subsets similar to CIFAR-10, but compare to reference models trained without augmentations on larger datasets. We start with ResNet-18 architectures, evaluating their exchange rate behavior, but consider other architectures in Section 3.1.2. Our dataset setup is quite specific to CIFAR-10/CINIC-10, revisiting this experiment e.g. for ImageNet would require running a pairing such as ImageNet/JFT-300 (Sun et al., 2017), which would be prohibitive for the number of experiments in this work. Nevertheless we include several validations on CIFAR-100, EMNIST, MNIST in Appendix B to verify that the behaviors observed are not specific to CIFAR-10.

# 3.1 EXCHANGE RATES: HOW MANY SAMPLES ARE AUGMENTATIONS WORTH?

We visualize the validation accuracies for a range of models trained with select augmentations in Figure 1 (left). The validation behavior of these models can be well described by power laws of the form  $f(x) = ax^{-c} + b$ , describing the relationship between number of samples and validation

accuracy. From these power laws, we derive the exchange rates of various augmentations compared to the reference curve of un-augmented models  $f_{\mathrm{ref}} = a'x^{-c'} + b'$ . For an augmentation policy described by  $f_{\mathrm{aug}} = ax^{-c} + b$ , we define its exchange rate via

$$
v _ {\text {E f f e c t i v e E x t r a S a m p l e s f r o m A u g m e n t a t i o n s}} (x) = f _ {\text {r e f}} ^ {- 1} \left(f _ {\text {a u g}} (x)\right) - x \tag {1}
$$

for a base dataset size  $x$ . We visualize this quantity in Figure 1 (right). This metric measures exactly the gain in accuracy between the un-augmented and augmented power laws shown on the left-hand side and converts this quantity into additional samples by evaluating on how much more data the reference models would need to realize the same accuracy as the augmented model.

A core property of augmentations that becomes evident in this analysis is the relationship between consistency of augmented samples with the underlying data distribution (Taylor & Nitschke, 2018; Gontijo-Lopes et al., 2020b; He et al., 2019b) and diversity of extra data (Gontijo-Lopes et al., 2020b). In Figure 1, we find that an augmentation strategy, such as TrivialAug (Müller & Hutter, 2021) (a policy consisting of a random draw from a table of 14 common photo- and geometric transforms, applied in tandem with horizontal flips and random crops), shown in orange, is highly diverse, generating a large amount of effective extra samples when the number of base samples is small. However, this policy also falls off quickest as more base samples are added: The policy is ultimately inconsistent with the underlying data distributions and when enough real samples are present, gains through augmentation deteriorate. On the flip side, augmenting with only horizontal flips, shown in green, is less diverse, and hence yields limited impact for smaller dataset sizes. However, the augmentation is much more consistent with the data distribution, and as such horizontal flips are beneficial even for large dataset sizes.

Takeaway: The impact of augmentations is linked to dataset size; diverse but inconsistent augmentations provide large gains for smaller sizes but are hindrances at scale. Estimation of the value of future data collection should take this effect of augmentations on scaling into account.

# 3.1.1 MEASURING DIVERSITY AND CONSISTENCY DIRECTLY

We can disambiguate the effects of consistency and diversity further by analyzing these augmentations for a moment not as randomly applied augmentations, but as fixed enlargements of the dataset. In Figure 2 (right), we see that a single repetition of the dataset generated by TrivialAug is detrimental to performance, while the same effect for a single random horizontal flip of the dataset is minimal. As such, TrivialAug is inconsistent with the data distribution. On the other hand, evaluating the gains realized by applying the augmentation policy multiple times, we also see that TrivialAug leads to significantly more diversity in this controlled experiment, compared to random flips (Figure 2, left).

Another way to look at Figure 2 is again as a measurement of extra data. Simply replacing each base sample by a single augmented sample is not a benefit, yet we quickly find that large gains can be attributes simply to the duplication and quadruplication of existing data.

![](images/f3e7931a3f4c13c46b09fb90b0a869c85797d3f54b84155fa173e3c31a76be73.jpg)  
Figure 2: Exchange Rates via Equation (1) when creating larger datasets from a fixed number of applications of a data augmentation to base samples. Left: Exchange rate for random horizontal flips as the augmentation policy is repeated. Right: Exchange Rates for TrivialAug. Even a few fixed repetitions of existing samples generate most of the effective additional data observed in Figure 1.

![](images/5d7f3ff6cbdaddf25c86b817d7f4138755925f55e788ac3faabad0e0a01454be.jpg)

![](images/02dcdd8585ef76a64433b3181db96e77e649a8770180e2edd415b642eea4f25b.jpg)

![](images/62a96d78393520ae6959021cb73918ccb38d9bb77c38000b2a684c94d754d36e.jpg)

![](images/38299859c80a5f5cdaf549b24e9a35d03092fb0f8fbc6ed6e395488460e1d783.jpg)  
Figure 4: Exchange Rates as Figure 1, but evaluated on CIFAR-10 validation (left) and CIFAR-10-C (right). Note that for CIFAR-10-C, inconsistent augmentations are worth more than any number of additional in-domain samples. Augmentations really are worth much more, under slight (left) and large (right) distribution shifts.

![](images/6e51b8335e2f0978b230bd3af6aadea0e5d0d4df2b9cfe00f466033a9b9d89cf.jpg)  
Figure 3: Exchange Rates via Equation (1) for TrivialAug when modifying the model architecture. Rates for various widths of a ResNet-18 (left). 64 is the default in other parts of this work. Exchange Rates for select vision architectures (right) with similar parameter counts. Exchange rates behave similarly for a range of models architectures, and benefits increase as models widen.

# 3.1.2 EXCHANGE RATES FOR MODEL SCALING AND MODEL VARIATIONS

Figure 3 evaluates the effective samples gained from augmentation as model width increases (left) and model architecture changes (right). For both plots, the references  $f_{\mathrm{ref}}$  are based on the unaugmented models with that same configuration of width or architecture. We find that model width of the evaluated ResNet-18 reliably correlates with the gains from augmentations to larger dataset sizes. Evaluating different architectures, we find that while global behavior is similar for all models, the exact exchange rates are similar for the convolutional architectures of ResNet, PyramidNet (Han et al., 2017) and VGG (Simonyan & Zisserman, 2014), mirroring their closely related inductive biases. On the other hand, from the two transformer architectures, we find that a notably limited benefit from augmentations versus real data for the Swin-Transformer (Liu et al., 2021b; 2022), but a large benefit for the ConvMixer (Trockman & Kolter, 2022). In direct comparison of both architectures, the Swin-Transformer contains a large number of features designed specifically for vision tasks, whereas ConvMixer is much more general, so that their differing inductive biases are again reflected in Figure 3.

Takeaway: Relative gains through augmentations as sample sizes scale are broadly consistent across model widths and architectures, but absolute gains depend on architecture.

# 3.2 EXCHANGE RATES IN OUT-OF-DISTRIBUTION SETTINGS

A benefit of extra data generated from augmentations that is often underappreciated is illustrated in Figure 4, showing exchange rates of the same models trained on CINIC-10 as analyzed in Figure 1, but evaluated on the CIFAR-10 validation set (left). Comparing CINIC-10 and CIFAR-10, both datasets are nearly indistinguishable using simple summary statistics (Darlow et al., 2018), yet there is a minor distribution shift caused by different image processing protocols during dataset curation. In this setting, diverse augmentations are now quickly on-par with models trained on many more samples. Figure 5

![](images/440e0ba852854a9710758063cf5bb2aac2fb78e9170e253f0c05f88310f3e2db.jpg)  
Figure 5: Exchange Rates as Figure 2 for fixed repetitions of the data generated with TrivialAug, but evaluated on CIFAR-10 validation (left) and CIFAR-10-C (right). We find that even for this mild OOD shift, replacing each sample with a single augmented view wins as soon as more than 75000 samples are used. For CIFAR-10-C, even a single few repetition is worth more than any almost number of additional in-domain samples.

![](images/bc27305063f28943150e12f2fd791db20846ca590accc81bb4f1b2a0cebc334d.jpg)

shows that this effect is apparent, even if base samples are replaced by a fixed number of augmented samples. Four augmented samples are quickly worth more than additional real samples, and with enough base samples, even replacing each sample by a fixed augmented version is beneficial. These effects can be exaggerated by evaluating on the CIFAR-10-C dataset of common corruptions applied to CIFAR-10 (Hendrycks & Dietterich, 2018). There, we quickly find that evaluating exchange rates for this large distribution shift, that the stronger augmentations evaluated, quickly produces benefits beyond what a collection containing even substantial amounts of in-domain data would achieve.

The increased diversity in these augmentations broadens the support of the data distribution, and the support of the CIFAR-10 dataset appears to be well contained within the transformed data. In practical applications, actively broadening the support of the data distribution in the face of uncertainty is advantageous as suddenly each augmentation is effectively worth much more data.

Takeaway: Data augmentations broaden the support of training data, which significantly extends their usefulness even on larger dataset scales in OOD testing scenarios.

# 4 AUGMENTATIONS AND INVARIANCE

The success of augmentations is often attributed to the invariances encoded into the model by enforcing the assignment of identical labels across transformations of each training sample. If the success of data augmentations can be attributed solely to invariance, then we can build exactly invariant models that achieve comparable accuracy when trained without data augmentation. Several works propose such mechanisms for constraining neural network layers to be invariant, and we will leverage these in our study.

# 4.1 INVARIANT NEURAL

# NETWORKS WITHOUT AUGMENTATION

![](images/426efd636afe71078f454e0752a588f3f8ca7c3eb1a4c9e0d0ec043e0073c9af.jpg)  
Figure 6: Exchange Rates for horiz. flips of ResNet and various invariant architectures. All models have an equal number of base parameters and are based on a Resnet-18 template. Invariant architectures show opposite scaling behavior to augmentations.

In order to probe the benefits of invariance without augmentations, we evaluate the following three methods for constructing invariant networks, for the case of invariance to horizontal flips:

Prediction averaging: Insert augmented views of a sample into the model and average the corresponding predictions (Simonyan & Zisserman, 2014). We use this procedure during both training and inference, and refer to the approach applied to a ResNet as flip-invariant ResNet-18. Note that this method still involves passing augmented data into the model.

Orbit Selection: An invariance can also be enforced via orbit selection (Gandikota et al., 2021). Here, an orbit mapping uniquely selects a single element from the group of transformation before the

![](images/a2d454183ce3c637871afdc3ce165c6fd39d967c784a18595e0b42b70614222f.jpg)  
Figure 7: Out-of-distribution augmentations still boost performance. Left: Test error (log-scale) as a function of training samples when test images are rotations of training images. Right: Test accuracy on rotated samples from CIFAR-10 that were not used for training. All experiments performed on rotated CIFAR-10 samples with the ResNet-18 architecture. Bars mark standard error over 5 trials.

![](images/da513411dda24a3dc113e300c40af3974500b0d6dfe50d61c9772c879fd08539.jpg)

data is passed into the model.

E2CNN: General E(2)-Equivariant Steerable CNN (E2CNN) (Weiler & Cesa, 2019) constrains convolutional kernels to reflect a group equivariance. We instantiate and E2CNN with the same architecture as the ResNet-18 studied so far.

In Section 3, we saw that horizontal flips are consistent with the CIFAR-10 distribution, so it may not be surprising that horizontal flip invariant networks perform better than those trained with neither augmentations nor invariance constraints. Moreover, networks trained with data augmentations outperform invariant models for small and medium sample sizes. However, we observe in Figure 6 that all investigated invariant architectures (red, green, blue) catch up to models trained with random augmentations (in orange) as we increase the number of base samples. We will see in Section 5 that data augmentations serve as flatness-seeking regularizers, and the benefits of this additional regularization wane as the number of samples increases.

# 4.2 OUT-OF-DISTRIBUTION AUGMENTATIONS STILL IMPROVE PERFORMANCE

Previously, we observed the performance benefits of data augmentations which promote invariances consistent with the data distribution, or approximately so. But can it still be useful to augment our data with a transformation that generates samples completely outside the support of the data distribution and which are inconsistent with any label? To answer this question, we construct a synthetic dataset in which the exact invariances are known.

We begin by randomly sampling a single base image from each CIFAR-10 class. We then construct 10 classes by rotating each of the base images, so that all samples in a class correspond to rotations of a single image. Thus, the classification task at hand is to determine which base image was rotated to form the test sample. We randomly sample rotations from each of these classes to serve as training data and another disjoint set of rotations to serve as test data. We then use horizontal flip and random crop data augmentations to generate out-of-distribution samples, since horizontally flipped or cropped image views cannot be formed merely via rotation. Note that this experiment is distinct from typical covariate shift setups where the distribution of data domains differs, but the support is far from disjoint and may even be identical.

In Figure 7, we see that these out-of-distribution augmentations are beneficial nonetheless. Notably, random crops, which can generate significantly more unique views than horizontal flips, yield massive performance boosts for identifying rotated images, even though we know that the cropped samples are out-of-distribution. We also see in this figure that random crops are especially useful if we instead use as our test set rotations of samples from CIFAR-10 that were not used for training. Specifically, we assign a base test image and its rotations the same label as the base image from the training set with the same CIFAR-10 label. This experiment supports the observations from Section 3 that augmentations can be particularly beneficial for OOD generalization.

Takeaway: Comparing invariant architectures to augmentations, we find that augmentations dominate on smaller scales, but invariant architectures catch up in the large-sample regime. Augmentations can provide benefits even on apparently unrelated invariances, which is particularly helpful for OOD generalization.

![](images/69bca04ef8ea4730fdf204df9adffcb3c98c22c59eb349b08adaa466f77942fc.jpg)  
Figure 8: Randomly applied augmentations significantly increase stochasticity late in training but decrease stochasticity early. Standard deviation of gradient across epochs for different augmentations and different mini-batch sampling strategies. Shown is the mean over 10 runs, and shaded regions represent one standard error.

# 5 AUGMENTATIONS AS A SOURCE OF STOCHASTICITY DURING TRAINING

Typical loss functions are summed over training samples. During optimization, gradients are computed using small mini-batches of random samples, resulting in stochasticity. Augmentations increase the number of available data, often so much that we never sample the same data twice.

Since data augmentations expand and diversify the training set, they may serve as additional sources of stochasticity during optimization. If data augmentations do increase the variety of gradients, they could as a result cause us to find qualitatively different minima. Stochastic optimization is thought to be associated with flat minima of the loss landscape which are in turn associated with superior generalization (Jastrzebski et al., 2017; Huang et al., 2020; Liu et al., 2021a). This flatness-seeking behavior may be the effect of both the augmented loss function and also how we sample it.

To put this hypothesis to the test, we measure the standard deviation of gradients during optimization for models trained with and without data augmentations, and we quantify the flatness of the corresponding minima. We construct experiments that disentangle the augmented loss function from the additional stochasticity produced by sampling augmentations. We consider a "same batch" strategy in which gradient updates are averaged over multiple views of a single image, resulting in lower stochasticity. We also consider "fixed views" where we repeat a frozen set of augmented views per element of the training set, as in Figure 2.

# 5.1 MEASURING STOCHASTICITY

To measure stochasticity, we train a model on a given training set and augmentation strategy, and we freeze the model every 10 epochs to estimate the standard deviation (formally the norm of parameter-wise standard deviations) of its gradients over randomly sampled batches comprising 128 base images, the same batch size used during training. That is, we measure the square root of the average squared distance between a randomly sampled batch gradient and the mean gradient. We adopt a filter-normalized distance function (Li et al., 2018; Huang et al., 2020) to account for invariances in neural networks whereby shrinking the parameters in convolutional filters may not effect the network's output but may make the model more sensitive to parameter perturbations of a fixed size.

In Figure 8, we see that non-augmented datasets actually yield noisier gradients early in training, but this noise vanishes rapidly as over-fitting occurs. In contrast, randomly applied data augmentations result in flatter curves, indicating that the added diversity of views available for sampling preserves stochasticity later in training. We also see that for each augmentation policy, applying augmentations randomly results in the most stochasticity late in training, while including multiple random views in the same batch (Hoffer et al., 2020; Fort et al., 2022) results in less. Sampling augmentations from a fixed set of four views per sample (denoted "fixed views") results in even less stochasticity, and including each of the four views in every batch results in the least stochasticity (denoted "fixed views", "same batch"). This ordering, which holds across all data augmentations we try, is consistent with the intuition that more randomness in augmentation leads to more stochasticity in training, notably only manifesting during later epochs. We will now see that the late-training stochasticity we measure correlates strongly with the flatness of the minima these optimization procedures find.

# 5.2 MEASURING FLATNESS

We adopt the flatness measurements from Huang et al. (2020) as these measurements are non-local, do not require Hessian computations which are dubious for non-smooth ReLU networks, and they are consistent with our filter-normalized gradient standard deviation measurements. Specifically, we measure the average filter-normalized distance in random directions from the trained model parameters before we reach a loss function value of 1.0, where loss is evaluated on the non-augmented dataset. Under this metric, larger values correspond to flatter minima as parameters can be perturbed further without increasing loss. We use the same ResNet-18 models trained in the stochasticity experiments above with the same exact augmentation setups.

Investigating Figure 8 and Figure 9, (see also Table 4), we observe that flatness correlates strongly with late-training stochasticity. Models trained without augmentation or with non-random augmentation, where all views are seen in each batch, are less stochastic at the end of training and find sharper minima. While previous works have associated SGD with flatness-seeking behavior (Jastrzebski et al., 2017; Geiping et al., 2021), data augmentations appear to contribute to this phenomenon. Simply put, training with randomized data augmentations finds flatter minima, and models trained with strong data augmentations lie at especially flat minima.

Takeaway: Randomly applied augmentations provide benefits beyond invariance by flattening the loss landscape, which is reflected in both measurements of flatness after training and measurements of gradient noise late in the training.

# 5.3 DATASET SCALING AND FLATNESS

Figure 9 directly measures flatness for several dataset scales. We first notice that base models become flatter (with respect to their base samples) as the number of samples increases. Surprisingly, stronger augmentations can produce this effect quicker and raise flatness values even for lower sample sizes. As a notable example, TrivialAug produces models that remain relatively flat for all sample sizes considered. Furthermore, all augmentations converge to similar flatness in the sample size limit, as regularization becomes less relevant in the large data regime. Over all plots we can even correlate the

![](images/fb534140cd657042db398b45b60c482a45a469fb0beab7387232f30e65ce4aa4.jpg)  
Figure 9: Left: Flatness for augmented models trained on several dataset sizes from Figure 1. All strategies converge to similar levels of flatness when scaling.

number of samples gained through augmentations and flatness and find that for all weaker augmentations, flatness of the solution is strongly correlated with the number of extra samples that are gained from the augmentation.

Takeaway: Strong data augmentations flatten the loss landscape to levels otherwise only reached with significantly larger datasets.

# 6 CONCLUSION

Data augmentations have a profound impact on the performance of neural networks, but their precise role has not been well understood; for example, if augmentations are simply a heuristic for learning certain symmetries, should we not prefer to directly encode these symmetries through advances in group equivariant networks? Through the lens of exchange rates and power laws, we observe the gains through augmentations as datasets scale and domains change. We find that augmentations dominating invariant architectures on smaller scales, but, scale in opposite ways. Augmentations are further distinguished from invariances in the way they can improve performance even out-of-distribution. Ultimately we find that we can connect these findings to the regularization effect induced by data augmentations, which we also measure, showing how augmentations flatten the loss landscape. This work promotes an all-encompassing understanding of neural network training, shedding light on the nuanced but significant role of data augmentation in the success of deep learning.

# ETHICS STATEMENT

We foresee no direct negative societal consequences from this work. We do think that data augmentations are beneficial, especially in applications with only limited data, or where data curation is expensive. We argue that knowing how to exchange a smaller (but verified and curated) dataset for a larger dataset that is not augmented, but also due to its size less curated is hopefully helpful to the community.

# REPRODUCIBILITY STATEMENT

We use an academic cluster with NVIDIA RTX4000 cards and also NVIDIA GTX2080ti cards. Each job is scheduled on a single GPU and the default setting of 60000 gradient steps takes roughly an hour to train and evaluate. Including all preliminary experiments we estimate a total usage of about 400 GPU days for this project. To replicate all experiments in the main body without repeated trials, we estimate a requirement of about 10 GPU days. We provide code with the supplementary material to do so.

# REFERENCES

Guozhong An. The effects of adding noise during backpropagation training on a generalization performance. Neural computation, 8(3):643-674, 1996. (p. 2)  
Randall Balestriero, Leon Bottou, and Yann LeCun. The effects of regularization and data augmentation are class dependent. arXiv preprint arXiv:2204.03632, 2022a. (p. 2)  
Randall Balestriero, Ishan Misra, and Yann LeCun. A data-augmentation is worth a thousand samples: Exact quantification from analytical augmented sample moments. arXiv preprint arXiv:2202.08325, 2022b. (p. 2)  
Gregory Benton, Marc Finzi, Pavel Izmailov, and Andrew G Wilson. Learning Invariances in Neural Networks from Training Data. In Advances in Neural Information Processing Systems, volume 33, pp. 17605-17616. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/hash/cc8090c4d2791ccd9cd2cb3c24296190-Abstract.html. (p. 2)  
Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16:321-357, 2002. (p. 2)  
Shuxiao Chen, Edgar Dobriban, and Jane Lee. A Group-Theoretic Framework for Data Augmentation. In Advances in Neural Information Processing Systems, volume 33, pp. 21321-21333. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/hash/f4573fc71c731d5c362f0d7860945b88-Abstract.html. (p. 2)  
Gregory Cohen, Saeed Afshar, Jonathan Tapson, and André van Schaik. EMNIST: An extension of MNIST to handwritten letters. arXiv:1702.05373 [cs], February 2017. doi: 10.48550/arXiv.1702.05373. (p. 15)  
Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V. Le. AutoAugment: Learning Augmentation Policies from Data. arXiv:1805.09501 [cs, stat], April 2019a. URL http://arxiv.org/abs/1805.09501. (p. 2, 15)  
Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. RandAugment: Practical automated data augmentation with a reduced search space. arXiv:1909.13719 [cs], November 2019b. URL http://arxiv.org/abs/1909.13719. (p. 2, 15)  
Tri Dao, Albert Gu, Alexander Ratner, Virginia Smith, Chris De Sa, and Christopher Re. A Kernel Theory of Modern Data Augmentation. In Proceedings of the 36th International Conference on Machine Learning, pp. 1528-1537. PMLR, May 2019. URL https://proceedings.mlr.press/v97/dao19b.html. (p. 2)

Luke N. Darlow, Elliot J. Crowley, Antreas Antoniou, and Amos J. Storkey. CINIC-10 is not ImageNet or CIFAR-10. arXiv:1810.03505 [cs, stat], October 2018. URL http://arxiv.org/abs/1810.03505. (p. 3, 5, 15)  
Marc Finzi, Samuel Stanton, Pavel Izmailov, and Andrew Gordon Wilson. Generalizing convolutional neural networks for equivariance to lie groups on arbitrary continuous data. In International Conference on Machine Learning, pp. 3165-3176. PMLR, 2020. (p. 1)  
Stanislav Fort, Andrew Brock, Razvan Pascanu, Soham De, and Samuel L. Smith. Drawing Multiple Augmentation Samples Per Image During Training Efficiently Decreases Test Error. arXiv:2105.13343 [cs], February 2022. URL http://arxiv.org/abs/2105.13343. (p. 1, 8)  
Kanchana Vaishnavi Gandikota, Jonas Geiping, Zorah Lähner, Adam Czaplinski, and Michael Moeller. Training or architecture? how to incorporate invariance in neural networks. arXiv preprint arXiv:2106.10044, 2021. (p. 6)  
Jonas Geiping, Micah Goldblum, Phil Pope, Michael Moeller, and Tom Goldstein. Stochastic Training is Not Necessary for Generalization. In International Conference on Learning Representations, September 2021. URL https://openreview.net/forum?id=ZBESeIUB5k. (p. 1, 2, 9)  
Raphael Gontijo-Lopes, Sylvia Smullin, Ekin Dogus Cubuk, and Ethan Dyer. Tradeoffs in Data Augmentation: An Empirical Study. In International Conference on Learning Representations, September 2020a. URL https://openreview.net/forum?id=ZcKPwuhG6wy. (p. 2)  
Raphael Gontijo-Lopes, Sylvia J. Smullin, Ekin D. Cubuk, and Ethan Dyer. Affinity and Diversity: Quantifying Mechanisms of Data Augmentation. arXiv:2002.08973 [cs, stat], June 2020b. URL http://arxiv.org/abs/2002.08973. (p. 2, 4)  
Dongyoon Han, Jiwhan Kim, and Junmo Kim. Deep Pyramidal Residual Networks. arXiv:1610.02915 [cs], September 2017. URL http://arxiv.org/abs/1610.02915. (p. 5)  
Boris Hanin and Yi Sun. How Data Augmentation affects Optimization for Linear Regression. In Advances in Neural Information Processing Systems, volume 34, pp. 8095-8105. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper/2021/bit/442b548e816f05640dec68f497ca38ac-Abstract.html. (p. 2)  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. arXiv:1512.03385 [cs], December 2015. URL http://arxiv.org/abs/1512.03385. (p. 14)  
Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. Bag of Tricks for Image Classification with Convolutional Neural Networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 558-567, 2019a. URL https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html. (p. 14)  
Zhuoxun He, Lingxi Xie, Xin Chen, Ya Zhang, Yanfeng Wang, and Qi Tian. Data Augmentation Revisited: Rethinking the Distribution Gap between Clean and Augmented Data. arXiv:1909.09148 [cs, stat], November 2019b. URL http://arxiv.org/abs/1909.09148. (p. 2, 4)  
Dan Hendrycks and Thomas Dietterich. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. In International Conference on Learning Representations, September 2018. URL https://openreview.net/forum?id=HJz6tiCqYm. (p. 6, 15)  
Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, and Balaji Lakshminarayanan. AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty. arXiv:1912.02781 [cs, stat], February 2020. URL http://arxiv.org/abs/1912.02781. (p. 2, 15)

Alex Hernández-García and Peter König. Further Advantages of Data Augmentation on Convolutional Neural Networks. In Artificial Neural Networks and Machine Learning - ICANN 2018, Lecture Notes in Computer Science, pp. 95-103, Cham, 2018. Springer International Publishing. ISBN 978-3-030-01418-6. doi: 10.1007/978-3-030-01418-6_10. (p. 2)  
Elad Hoffer, Tal Ben-Nun, Itay Hubara, Niv Giladi, Torsten Hoefler, and Daniel Soudry. Augment Your Batch: Improving Generalization Through Instance Repetition. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8126-8135, June 2020. doi: 10.1109/CVPR42600.2020.00815. (p. 8)  
W Ronny Huang, Zeyad Emam, Micah Goldblum, Liam Fowl, Justin K Terry, Furong Huang, and Tom Goldstein. Understanding generalization through visualizations. 2020. (p. 2, 8, 9)  
Stanisław Jastrzejbski, Zachary Kenton, Devansh Arpit, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey. Three factors influencing minima in sgd. arXiv preprint arXiv:1711.04623, 2017. (p. 1, 2, 8, 9)  
Sanyam Kapoor, Wesley J Maddox, Pavel Izmailov, and Andrew Gordon Wilson. On uncertainty, tempering, and data augmentation in bayesian classification. arXiv preprint arXiv:2203.16481, 2022. (p. 2)  
Jaehyung Kim, Dongyeop Kang, Sungsoo Ahn, and Jinwoo Shin. What Makes Better Augmentation Strategies? Augment Difficult but Not too Different. In International Conference on Learning Representations, September 2021. URL https://openreview.net/forum?id=Ucx3DQbC9GH. (p. 2)  
Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. 2009. URL https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf. (p. 3, 15)  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998. (p. 2, 15)  
Daniel LeJeune, Randall Balestriero, Hamid Javadi, and Richard G. Baraniuk. Implicit Rugosity Regularization via Data Augmentation. arXiv:1905.11639 [cs, stat], October 2019. URL http://arxiv.org/abs/1905.11639. (p. 2)  
Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 6391-6401, 2018. (p. 8)  
Tianyi Liu, Yan Li, Song Wei, Enlu Zhou, and Tuo Zhao. Noisy gradient descent converges to flat minima for nonconvex matrix factorization. In International Conference on Artificial Intelligence and Statistics, pp. 1891-1899. PMLR, 2021a. (p. 1, 2, 8)  
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9992-10002, Montreal, QC, Canada, October 2021b. IEEE. ISBN 978-1-66542-812-5. doi: 10.1109/ICCV48922.2021.00986. URL https://ieeexplore.ieee.org/document/9710580/. (p. 5)  
Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, and Baining Guo. Swin Transformer V2: Scaling Up Capacity and Resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12009-12019, 2022. URL https://openaccess.thecvf.com/content/ CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_ Resolution_CVPR_2022_paper.html. (p. 5)  
Antonia Marcu and Adam Prügel-Bennett. On the Effects of Data Distortion on Model Analysis and Training. arXiv:2110.13968 [cs], October 2021. URL http://arxiv.org/abs/2110.13968. (p. 2)  
Samuel G. Müller and Frank Hutter. TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation. arXiv:2103.10158 [cs], August 2021. URL http://arxiv.org/abs/2103.10158. (p. 2, 4, 15)

Behnam Neyshabur, Ryota Tomioka, Ruslan Salakhutdinov, and Nathan Srebro. Geometry of optimization and implicit regularization in deep learning. arXiv preprint arXiv:1705.03071, 2017. (p. 2)  
Shashank Rajput, Zhili Feng, Zachary Charles, Po-Ling Loh, and Dimitris Papailiopoulos. Does Data Augmentation Lead to Positive Margin? In Proceedings of the 36th International Conference on Machine Learning, pp. 5321-5330. PMLR, May 2019. URL https://proceedings.mlr.press/v97/rajput19a.html. (p. 2)  
Connor Shorten and Taghi M Khoshgoftaar. A survey on image data augmentation for deep learning. Journal of big data, 6(1):1-48, 2019. (p. 2)  
Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556 [cs], September 2014. URL http://arxiv.org/abs/1409.1556. (p. 5, 6, 14)  
Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting Unreasonable Effectiveness of Data in Deep Learning Era. In Proceedings of the IEEE International Conference on Computer Vision, pp. 843-852, 2017. URL https://openaccess.thecvf.com/content.iccv_2017/html/Sun_Revisiting-Unreasonable_Effectiveness_ICCV_2017_paper.html. (p. 3)  
Luke Taylor and Geoff Nitschke. Improving Deep Learning with Generic Data Augmentation. In 2018 IEEE Symposium Series on Computational Intelligence (SSCI), pp. 1542-1547, November 2018. doi: 10.1109/SSCI.2018.8628742. (p. 2, 4)  
Asher Trockman and J. Zico Kolter. Patches Are All You Need? arXiv.2201.09792, January 2022. doi: 10.48550/arXiv.2201.09792. (p. 5, 14)  
Maurice Weiler and Gabriele Cesa. General e (2)-equivariant steerable cnns. Advances in Neural Information Processing Systems, 32, 2019. (p. 1, 7)  
Mingle Xu, Sook Yoon, Alvaro Fuentes, and Dong Sun Park. A Comprehensive Survey of Image Augmentation Techniques for Deep Learning. (arXiv:2205.01491), May 2022. doi: 10.48550/arXiv.2205.01491. (p. 2)  
Larry Yaeger, Richard Lyon, and Brandyn Webb. Effective training of a neural network character classifier for word recognition. In Proceedings of the 9th International Conference on Neural Information Processing Systems, NIPS'96, pp. 807-813, Cambridge, MA, USA, December 1996. MIT Press. (p. 2)  
Sicheng Zhu, Bang An, and Furong Huang. Understanding the Generalization Benefit of Model Invariance from a Data Perspective. In Advances in Neural Information Processing Systems, volume 34, pp. 4328-4341. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper/2021/hash/2287c6b8641dd2d21ab050eb9ff795f3-Abstract.html. (p. 2)
