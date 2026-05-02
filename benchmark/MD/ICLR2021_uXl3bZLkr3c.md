# TENT: FULLY TEST-TIME ADAPTATION BY ENTROPY MINIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

To generalize to new and different data during testing, a model must adapt itself. We highlight the setting of fully test-time adaptation given only unlabeled target data and the model parameters. We propose test-time entropy minimization (tent): we optimize for model confidence as measured by the entropy of its predictions. During testing, we adapt the model features by estimating normalization statistics and optimizing channel-wise affine transformations. Tent improves robustness to corruptions for image classification on ImageNet and CIFAR-10/100, and demonstrates the feasibility of target-only domain adaptation for digit classification from SVHN to MNIST/MNIST-M/USPS and semantic segmentation from GTA to Cityscapes.

# 1 INTRODUCTION

Deep networks can achieve high accuracy on training and testing data from the same distribution, as evidenced by tremendous benchmark progress (Krizhevsky et al., 2012; Simonyan & Zisserman, 2015; He et al., 2016). However, generalization to new and different data is limited (Hendrycks & Dietterich, 2019; Recht et al., 2019; Geirhos et al., 2018). Accuracy suffers when the training (source) data differ from the testing (target) data, a condition known as dataset shift (Quionero-Candela et al., 2009). Models can be sensitive to shifts during testing that were not known during training, whether natural variations or corruptions, such as unexpected weather or sensor degradation. Nevertheless, it can be necessary to deploy a model on different data distributions, so adaptation is needed.

During testing, the model must adapt given only its parameters and the target data. This fully test-time adaptation setting cannot rely on source data or supervision. Neither is practical when the model first encounters new testing data, before it can be collected and annotated, as inference must go on. Real-world usage motivates fully test-time adaptation by data, computation, and task needs:

1. Availability. A model might be distributed without source data for bandwidth, privacy, or profit.  
2. Efficiency. It might not be computationally practical to (re-)process source data during testing.  
3. Accuracy. A model might be too inaccurate without adaptation to serve its purpose.

We take the entropy of model predictions during testing as our adaptation objective. We call this the test entropy and name our method tent after it. Entropy is related to error, as more confident predictions are all-in-all more correct (Figure 1). On corrupted data, entropy is an indicator of the severity of the corruption, with a strong rank correlation to the loss for image classification (Figure 2).

To minimize entropy, tent normalizes and transforms the model features on target data by estimating statistics and optimizing affine parameters. This choice of low-dimensional, channel-wise feature modulation is efficient to update during testing, even for online adaptation. Tent does not restrict or alter model training: it is independent of the source data given the model parameters. If the model can be run, it can be adapted. Most importantly, tent effectively reduces not just entropy but error.

Our results evaluate robustness to common image corruptions and accuracy under domain shift for digit recognition. For reference results with more data and optimization, we evaluate methods for robust training, domain adaptation, and self-supervised learning given the labeled training data. Tent achieves less error given only the test data. Our analysis supports entropy as an objective, backs the generality of tent across architectures, and varies the amount of parameters and data for adaptation.

![](images/8e00e865f79b46582264bc3d4c6cc9567c1bd7c35d5eda7deb0cf485d30c934a.jpg)  
Figure 1: Predictions with lower entropy have lower error rates on corrupted CIFAR-100-C. Certainty can serve as supervision during testing.

![](images/0777020e1ab888afaae973c1d0561426dc0cb6159f354b515d9a0dd8717e7aad.jpg)  
Figure 2: More corruption causes higher loss and entropy for the model on this data. Entropy can measure test shift without training data or labels.

# Our contributions

- We highlight the setting of fully test-time adaptation with only target data and no source data. We suggest benchmarking adaptation with offline and online use of target data.  
- We examine entropy as an adaptation objective, and propose tent: a test-time entropy minimization scheme to reduce generalization error by reducing the entropy of model predictions.  
- For robustness to corruptions, tent reaches  $44\%$  error on ImageNet-C, better than the state-of-the-art for robust training  $(49.6\%)$  and the strong baseline of test-time normalization  $(51.7\%)$ .  
- For domain adaptation, tent is capable of target-only adaptation for digit classification and semantic segmentation, and even rivals methods that use source data and more optimization on SVHN  $\rightarrow$  MNIST/USPS.

# 2 SETTING: FULLY TEST-TIME ADAPTATION

Adaptation addresses generalization from source to target. A model  $f_{\theta}(x)$  with parameters  $\theta$  trained on source data and labels  $x^{s}, y^{s}$  may not generalize when tested on shifted target data  $x^{t}$ . Table 1 summarizes adaptation settings, their required data, and types of losses. Our fully test-time adaptation setting uniquely requires only the model  $f_{\theta}$  and unlabeled target data  $x^{t}$ .

Existing adaptation settings extend training given more data and supervision. Transfer learning by fine-tuning (Donahue et al., 2014; Yosinski et al., 2014) needs target labels to (re-)train with a supervised loss  $L(x^{t},y^{t})$ . Without target labels, our setting denies this supervised training. Domain adaptation (DA) (Quionero-Candela et al., 2009; Saenko et al., 2010; Ganin & Lempitsky, 2015; Tzeng et al., 2015) needs both the source and target data to train with a cross-domain loss  $L(x^{s},x^{t})$ . Test-time training (TTT) (Sun et al., 2019b) needs the source data to jointly train with a supervised loss and an unsupervised loss  $L(x^{s})$ . Without source, our setting denies source supervision  $L(x^{s},y^{s})$  for joint training across domains (DA) or losses (TTT). These settings have their purposes, but do not cover all practical cases when source, target, or supervision are not simultaneously available.

Unexpected target data during testing requires test-time adaptation. TTT and our setting adapt the model by optimizing an unsupervised loss during testing  $L(x^{t})$ . During training, TTT jointly optimizes this same loss on source data  $L(x^{s})$  with a supervised loss  $L(x^{s},y^{s})$ , to ensure the parameters  $\theta$  are shared across losses for compatibility with adaptation by  $L(x^{t})$ . Fully test-time adaptation is independent of the training data and training loss given the parameters  $\theta$ . By not changing training, our setting has the potential to require less data and computation for adaptation.

# 3 METHOD:TESTENTROPYMINIMIZATIONVIAFEATUREMODULATION

We adapt the model through test-time optimization to minimize the entropy of its predictions by modulating its features. We call this adaptation method tent, for test entropy. Tent requires a compatible model, an objective to minimize (Section 3.1), and parameters to optimize over (Section 3.2) to fully define the algorithm (Section Section 3.3). Figure 3 outlines our fully test-time adaptation method.

Table 1: Adaptation settings differ by their data and therefore losses during training and testing. Of the source  ${}^{s}$  and target  ${}^{t}$  data  $x$  and labels  $y$  ,our fully test-time setting only needs the target data  ${x}^{t}$  .  

<table><tr><td>setting</td><td>source data</td><td>target data</td><td>train loss</td><td>test loss</td></tr><tr><td>fine-tuning</td><td>-</td><td>xt, yt</td><td>L(xt, yt)</td><td>-</td></tr><tr><td>domain adaptation</td><td>xs, ys</td><td>xt</td><td>L(xs, ys) + L(xs, xt)</td><td>-</td></tr><tr><td>test-time training</td><td>xs, ys</td><td>xt</td><td>L(xs, ys) + L(xs)</td><td>L(xt)</td></tr><tr><td>fully test-time adaptation</td><td>-</td><td>xt</td><td>-</td><td>L(xt)</td></tr></table>

![](images/519e21e6894466780bfa10d01e7074d346386d7c9b524a812be0658ea2e0a09d.jpg)  
Figure 3: Method overview. Tent does not alter training (a), but minimizes entropy during testing (b) over a constrained modulation  $\Delta$ , given the model parameters  $\theta$  and unlabeled target data  $x^t$ .

Compatibility The model must be trained for the supervised task, probabilistic, and differentiable. No supervision is provided during testing, so the model must already be trained. Measuring the entropy of predictions requires a distribution over predictions, so the model must be probabilistic. Gradients are required for fast iterative optimization, so the model must be differentiable.

Choice of Model and Task We choose image classifiers as representative models for supervised learning with deep networks. These models are probabilistic by their prediction of softmax class distributions and end-to-end differentiable by their design.

# 3.1 ENTROPY OBJECTIVE

Our test-time objective  $L(x_{t})$  is the entropy  $H(\hat{y})$  of the model prediction during testing  $\hat{y} = f_{\theta}(x_{t})$ . In particular, we measure the Shannon entropy (Shannon, 1948),  $H(\hat{y}) = -\sum_{c} p(\hat{y}_{c}) \log p(\hat{y}_{c})$  for the probability  $\hat{y}_{c}$  of class  $c$ . As a measure of the task predictions,  $H(\hat{y} = f_{\theta}(x))$  is a function of the model parameters, and therefore the supervised training. In this way entropy is task-general, as it is defined for any probabilistic task, but at the same time task-specific, as it varies with the task training.

In contrast, proxy tasks for self-supervised learning are not directly related to the supervised task. Proxy tasks derive a self-supervised label  $y'$  from the input  $x_{t}$  without the task label  $y$ . For concreteness, examples of these proxies include rotation prediction (Gidaris et al., 2018), context prediction (Doersch et al., 2015), and cross-channel auto-encoding (Zhang et al., 2017). Too much progress on a proxy task could interfere with performance on the supervised task, and self-supervised adaptation methods have to limit or mix updates accordingly (Sun et al., 2019b;a). As such, care is needed to choose a proxy compatible with the domain and task, to design the architecture for the proxy model, and to balance optimization between the task and proxy objectives.

Our test entropy objective does not present these difficulties. Entropy is an unsupervised objective that depends only on  $\hat{y}$ . No further effort or choice is needed to adopt it. On the other hand, supervised learning and self-supervised learning take more effort in annotation or design choices.

# 3.2 MODULATION PARAMETERS

The model parameters  $\theta$  are a natural choice for test-time optimization, and these are the choice of prior work for train-time entropy minimization in semi-supervised (Grandvalet & Bengio, 2005), few-shot (Dhillon et al., 2020), and domain adaptation (Carlucci et al., 2017) regimes. However,  $\theta$  is the only representation of the training/source data in our setting, and altering  $\theta$  could cause the model to drift from the training task. Furthermore,  $f$  can be nonlinear and  $\theta$  can be high dimensional, making optimization too sensitive and inefficient for test-time usage.

For stability and efficiency, we instead only update feature modulations that are linear (scales and shifts), and low-dimensional (channel-wise). Figure 4 shows the two steps of our modulations: normalization by statistics and transformation by parameters. Normalization centers and standardizes

$$
\mathrm {I N} \xrightarrow {\mu} \begin{array}{c c c c} \mu & \sigma & \gamma & \beta \\ \longrightarrow & \odot & \odot & \otimes \\ \longrightarrow & \odot & \odot & \oplus \end{array} \longrightarrow \mathrm {O U T} \quad \left| \begin{array}{l l} \text {n o r m a l i z a t i o n} & \mu \leftarrow \mathbb {E} [ x _ {t} ], \sigma^ {2} \leftarrow \mathbb {E} [ (\mu - x _ {t}) ^ {2} ] \\ \text {t r a n s f o r m a t i o n} & \gamma \leftarrow \gamma + \partial H / \partial \gamma , \beta \leftarrow \beta + \partial H / \partial \beta \end{array} \right.
$$

Figure 4: Tent modulates features during testing by estimating normalization statistics  $\mu, \sigma$  and optimizing transformation parameters  $\gamma, \beta$ . Normalization and transformation apply channel-wise scales and shifts to the features. The statistics and parameters are updated on target data without use of source data. In practice, adapting  $\gamma, \beta$  is efficient because they make up  $<1\%$  of model parameters.

the input  $x$  into  $\bar{x} = (x - \mu) / \sigma$  by its mean  $\mu$  and standard deviation  $\sigma$ . Transformation turns  $\bar{x}$  into the output  $x' = \gamma \bar{x} + \beta$  by affine parameters for scale  $\gamma$  and shift  $\beta$ . Note that the statistics  $\mu, \sigma$  are estimated from the data while the parameters  $\gamma, \beta$  are optimized by the loss.

For implementation, we simply repurpose the normalization layers of the source model by updating their normalization statistics and affine parameters during testing for all layers and channels.

# 3.3 ALGORITHM

Initialization The optimizer collects the affine transformation parameters  $\{\gamma_{l,k},\beta_{l,k}\}$  for each normalization layer  $l$  and channel  $k$  in the source model. The remaining parameters  $\theta \setminus \{\gamma_{l,k},\beta_{l,k}\}$  are fixed. The normalization statistics  $\{\mu_{l,k},\sigma_{l,k}\}$  from the source data are discarded.

Iteration Each step updates the normalization statistics and transformation parameters on a batch of test data. The normalization statistics are updated by moving averages on the test data for each layer in turn, during the forward pass. The transformation parameters  $\gamma, \beta$  are updated by their gradient of the prediction entropy  $\nabla H(\hat{y})$ , during the backward pass. For online adaptation, the forward pass is repeated once the model has been updated, to improve inference for every test point. This only needs  $2 \times$  the inference time plus  $1 \times$  the gradient time per test point vs. the standard  $1 \times$  inference time of the unadapted model.

Termination For online adaptation, no termination is necessary, and iteration continues as long as there is test data. For offline adaptation, we update for a single epoch, as it is simple and efficient. Inference is then repeated for the whole test set. Of course, it is possible to extend adaptation by continuing to optimize for multiple epochs.

# 4 EXPERIMENTS

We evaluate tent for corruption robustness on CIFAR-10/CIFAR-100 and ImageNet, and for domain adaptation on digit adaptation from SVHN to MNIST/MNIST-M/USPS. Our implementation is in PyTorch (Paszke et al., 2019) with the pyc1s library (Radosavovic et al., 2019). Our (anonymous) code is included with this submission, and the code will be released for publication.

Datasets We run on image classification datasets for corruption and domain adaptation conditions. For large-scale experiments we choose ImageNet (Russakovsky et al., 2015), with 1,000 classes, a training set of 1.2 million, and a validation set of 50,000. For experiments at an accessible scale we choose CIFAR-10/CIFAR-100 (Krizhevsky, 2009), with 10/100 classes, a training set of 50,000, and a test set of 10,000. For domain adaptation we choose SVHN (Netzer et al., 2011) as source and MNIST (LeCun et al., 1998)/MNIST-M (Ganin & Lempitsky, 2015)/USPS (Hull, 1994) as targets, with ten classes for the digits 0–9. SVHN has color images of house numbers from street views with a training set of 73,257 and test set of 26,032. MNIST/MNIST-M/USPS have handwritten digits with a training sets of 60,000/60,000/7,291 and test sets of 10,000/10,000/2,007.

Models For corruption experiments we use residual networks (He et al., 2016) with 26 layers (R-26) on CIFAR-10/100 and 50 layers (R-50) on ImageNet. For domain adaptation experiments we use the same R-26 architecture. For fair comparison, all methods in each experimental condition share the same architecture.

Our networks are equipped with batch normalization (Ioffe & Szegedy, 2015). For the source model without adaptation, the normalization statistics are estimated during training on the source data. For all test-time adaptation methods, we estimate these statistics during testing on the target data, as also done in concurrent work on adaptation by normalization (Schneider et al., 2020; Nado et al., 2020).

Table 2: Corruption benchmark on CIFAR-10-C and CIFAR-100-C for the highest severity. Tent has least error, with less optimization than domain adaptation (RG, UDA-SS) and test-time training (TTT), and improves on test-time norm (BN).  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Source</td><td rowspan="2">Target</td><td colspan="2">Error (%)</td></tr><tr><td>C10-C</td><td>C100-C</td></tr><tr><td>Source</td><td>train</td><td></td><td>40.8</td><td>67.2</td></tr><tr><td>RG</td><td>train</td><td>train</td><td>18.3</td><td>38.9</td></tr><tr><td>UDA-SS</td><td>train</td><td>train</td><td>15.2</td><td>44.0</td></tr><tr><td>TTT</td><td>train</td><td>test</td><td>17.5</td><td>45.0</td></tr><tr><td>BN</td><td></td><td>test</td><td>17.3</td><td>42.6</td></tr><tr><td>PL</td><td></td><td>test</td><td>15.7</td><td>41.2</td></tr><tr><td>Tent (ours)</td><td></td><td>test</td><td>14.3</td><td>37.3</td></tr></table>

![](images/8a182fb54d8ad46e49d994d760e9009216deb4100c053264d3ec395e8f8b38e1.jpg)  
Figure 5: Corruption benchmark on ImageNet-C: error for each type averaged over severity levels. Tent improves on the prior state-of-the-art, adversarial noise training (Rusak et al., 2020), by fully test-time adaptation without altering training.

**Optimization** We optimize the modulation parameters  $\gamma, \beta$  following the training hyperparameters for the source model with few changes. On ImageNet we optimize by SGD with momentum, while on other datasets we optimize by Adam (Kingma & Ba, 2015). We lower the batch size (BS) to reduce memory usage, then lower the learning rate (LR) by the same factor to compensate (Goyal et al., 2017). On ImageNet, we set BS = 64 and LR = 0.00025, and on other datasets we set BS = 128 and LR = 0.001. We shuffle the test data to avoid ordering effects, and control for shuffling by sharing the order across methods.

Baselines We compare to domain adaptation, self-supervision, normalization, and pseudo-labeling:

- source applies the trained classifier to the test data without adaptation,  
- adversarial domain adaptation (RG) reverses the gradients of a domain classifier on source and target to optimize for a domain-invariant representation (Ganin & Lempitsky, 2015),  
- self-supervised domain adaptation (UDA-SS) jointly trains self-supervised rotation and position tasks on source and target to optimize for a shared representation (Sun et al., 2019a),  
- test-time training (TTT) jointly trains for supervised and self-supervised tasks on source, then keeps training the self-supervised task on target during testing (Sun et al., 2019b),  
- test-time normalization (BN) updates batch normalization statistics (Ioffe & Szegedy, 2015) on the target data during testing (Schneider et al., 2020; Nado et al., 2020),  
- pseudo-labeling (PL) tunes a confidence threshold, assigns predictions over the threshold as labels, and then optimizes the model to these pseudo-labels before testing (Lee, 2013).

Only test-time normalization (BN), pseudo-labeling (PL), and tent (ours) are fully test-time adaptation methods. See Section 2 for an explanation and contrast with domain adaptation and test-time training.

# 4.1 ROBUSTNESS TO CORRUPTIONS

To benchmark robustness to corruption, we make use of common image corruptions (See appendix Section A for examples). The CIFAR-10/100 and ImageNet datasets are turned into the CIFAR-10/100-C and ImageNet-C corruption benchmarks by duplicating their test/validation sets and applying 15 types of corruptions at five severity levels (Hendrycks & Dietterich, 2019).

Tent improves error more with less data and computation. Table 2 reports errors averaged over corruption types at the severest level of corruption. On CIFAR-10/100-C we compare all methods, including those that require joint training across domains or losses, given the convenient size of this dataset. Adaptation is offline for fair comparison across this mixture of methods. Tent improves on the fully test-time adaptation baselines (BN, PL) but also the domain adaptation (RG, UDA-SS) and test-time training (TTT) methods that need the source data and several epochs of optimization.

Tent consistently improves error across corruption types. Figure 5 plots the errors for each corruption type averaged over levels on ImageNet-C. We compare the most efficient methods—source, normalization, and tent—given the large-scale size of the source data ( $>1$  million images) that other methods rely on and the 75 target combinations of corruption types and levels. Tent and BN adapt online to rival the efficiency of inference without adaptation. Note that many conditions have

Table 3: Digit domain adaptation from SVHN to MNIST/MNIST-M/USPS. Target-only adaptation is not only feasible, but more efficient. Tent always improves on normalization (BN), and in a 2/3 cases achieves less error than domain adaptation (RG, UDA-SS) without joint training on source & target.  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Source</td><td rowspan="2">Target</td><td rowspan="2">Epochs</td><td colspan="3">Error (%)</td></tr><tr><td>MNIST</td><td>MNIST-M</td><td>USPS</td></tr><tr><td>Source</td><td>train</td><td></td><td>0</td><td>18.2</td><td>39.7</td><td>19.3</td></tr><tr><td>RG</td><td>train</td><td>train</td><td>10</td><td>15.0</td><td>33.4</td><td>18.9</td></tr><tr><td>UDA-SS</td><td>train</td><td>train</td><td>30</td><td>39.0</td><td>44.1</td><td>22.8</td></tr><tr><td>BN</td><td></td><td>test</td><td>1</td><td>15.7</td><td>39.7</td><td>18.0</td></tr><tr><td>Tent (ours)</td><td></td><td>test</td><td>1</td><td>10.0</td><td>37.0</td><td>16.3</td></tr><tr><td>Tent (ours)</td><td></td><td>test</td><td>10</td><td>8.2</td><td>36.8</td><td>14.4</td></tr></table>

over  $50\%$  error even after adaptation by normalization, but tent nevertheless further reduces the error. Tent reaches the least error for all corruptions types without harming the error on the original test set.

Tent reaches new state-of-the-art without extending training. The state-of-the-art methods for robustness extend training with adversarial noise (ANT) (Rusak et al., 2020) for  $49.6\%$  error or mixtures of data augmentations (AugMix) (Hendrycks et al., 2020) for  $51.7\%$  error. Combined with stylization from external images (SIN) (Geirhos et al., 2019), ANT+SIN reaches  $47.4\%$  error. Tent reaches a new state-of-the-art of  $44.0\%$  error by online adaptation and  $42.3\%$  error by offline adaptation. This requires just one gradient per test point, without further optimization on the training set, much less external images.

Among fully test-time adaptation methods, tent reduces the error beyond test-time normalization for  $18\%$  relative improvement. In concurrent work, Schneider et al. (2020) report  $49.3\%$  error for test-time normalization, for which tent still gives  $14\%$  relative improvement.

# 4.2 TARGET-ONLY DOMAIN ADAPTATION

For unsupervised domain adaptation, we adopt the established setting of digit adaptation (Ganin & Lempitsky, 2015; Tzeng et al., 2015; 2017). In particular we experiment with adaptation from SVHN to MNIST/MNIST-M/USPS. Recall that unsupervised domain adaptation makes simultaneous use the labeled source data and unlabeled target data, while our fully test-time adaptation setting denies use of source data.

Tent adapts to target without source. Table 3 reports the target errors for domain adaptation and fully test-time adaptation methods. Test-time normalization (BN) marginally improves, while adversarial domain adaptation (RG) and self-supervised domain adaptation (UDA-SS) improve more by joint training on source and target. Tent always has lower error than the source model and BN, and it achieves the lowest error in 2/3 cases, even in just one epoch without use of source data.

Tent needs less computation, but still improves with more. Tent reduces computation by (1) only using target data and (2) only taking one gradient per point. RG & UDA-SS use the source data (SVHN train), which is  $\sim 7\times$  the size of the target data (MNIST test), and optimize for 10 epochs. Tent adapts with  $\sim 70\times$  less computation. When tent is adapted for more epochs, its error improves further while still only using one-seventh the amount of data.

Tent scales to semantic segmentation. To show scalability to large models and inputs, we evaluate target-only adaptation for semantic segmentation (pixel-wise classification) on a domain shift from simulated source data to real target data. The source is GTA (Richter et al., 2017), a visually-sophisticated video game in an urban environment, and the target is Cityscapes (Cordts et al., 2016), an urban autonomous driving dataset. The model is HRNet-W18, a fully convolutional network (Shelhamer et al., 2017) in the high-resolution network family (Wang et al., 2020). The target mIoU scores (higher is better) for source, BN, and tent are  $28.8\%$ ,  $31.4\%$ , and  $35.8\%$  with offline optimization by Adam. See the appendix for a qualitative example of adapting to a single image from target (Section C).

![](images/e89f1619e80e5428515d1783881214e83e45303c7a919afe6a9d03bc3030faa7.jpg)  
Figure 6: Tent reduces the entropy and loss. We plot changes in entropy  $\Delta H$  and loss  $\Delta L$  for all of CIFAR-100-C. Change in entropy rank-correlates with change in loss: note the dark diagonal and the rank correlation coefficient of 0.22.  
Figure 7: Adapted features on CIFAR-100-C with Gaussian noise (front) and reference features without corruption (back). Corruption disperses features from the reference, but BN brings them back. Tent is less like the reference, and more like an oracle that optimizes on target labels.

![](images/68c12349d8d77efc99fca780199351d0018798ed20025bbb6b12e58aa9b69a45.jpg)  
(a) Source

![](images/3bd125b292f1f4a4d83b08143a6c70d3f3f0107c5e93613c7f65198fd849ff17.jpg)  
(b) BN

![](images/b9971f36fa04b49d2959f3baf110817d6f86e78c7ea7b6468d7bb7c30a33df6f.jpg)  
(c) Tent

![](images/f6d73708cc66339c404009c36c117e07cef68b814bd44e6d8e00d40c41b543af.jpg)  
(d) Oracle

# 4.3 ANALYSIS

Tent reduces entropy and error. Figure 6 verifies tent does indeed reduce the entropy and the task loss (softmax cross-entropy). We plot changes in entropy and loss on CIFAR-100-C for all 75 corruption type/level combinations. Both axes are normalized by the maximum entropy of a prediction (log 100) and clipped to  $\pm 1$ . Most points have lower entropy and error after adaptation.

Tent needs feature modulation. We ablate the normalization and transformation steps of feature modulation. Not updating normalization increases errors, and can fail to improve over BN and PL. Updating the full model parameters  $\theta$  never improves over the unadapted source model.

Tent modulation differs from normalization. Modulation normalizes and transforms, but what is the combined effect? Figure 7 contrasts adapted features on corrupted data against reference features on uncorrupted data. We examine the source model, normalization, tent, and an oracle that optimizes on the target labels. Normalization more closely resembles the reference, but tent is not closer still. Instead, tent adjusts features more like the oracle. Differences of feature means confirm this pattern. This suggests a different, task-specific effect. (Figure 9 in the appendix shows more layers.)

Alternative Architectures In principle, tent is architecture agnostic. To gauge its generality, we evaluate new architectures based on self-attention (SAN) (Zhao et al., 2020) and equilibrium solving (MDEQ) (Bai et al., 2020) for corruption robustness on CIFAR-100-C. Table 4 reports the errors for the source model, BN adaptation, and tent adaptation. Tent reduces their error with the same settings as convolutional residual networks, in spite of their distinct architectures.

Table 4: Tent adapts new architectures on CIFAR-100-C without tuning. Results are error  $(\%)$  

<table><tr><td colspan="3">SAN (pair)</td><td colspan="3">SAN-10 (patch)</td><td colspan="3">MDEQ (large)</td></tr><tr><td>Source</td><td>BN</td><td>Tent</td><td>Source</td><td>BN</td><td>Tent</td><td>Source</td><td>BN</td><td>Tent</td></tr><tr><td>55.3</td><td>39.7</td><td>36.7</td><td>48.0</td><td>31.8</td><td>29.2</td><td>53.3</td><td>44.9</td><td>41.7</td></tr></table>

# 5 RELATED WORK

We relate tent to existing adaptation, entropy minimization, and feature modulation methods.

Train-Time Adaptation Domain adaptation methods train a joint model of the source and target by cross-domain losses  $L(x^{s},x^{t})$ . These losses optimize feature alignment (Gretton et al., 2009; Sun et al., 2017), adversarial invariance (Ganin & Lempitsky, 2015; Tzeng et al., 2017), or shared proxy tasks (Sun et al., 2019a). While they are effective in their setting, they do not apply when joint use of source and target is denied. Tent adapts entirely to target without joint modeling of source.

Recent "source-free" methods (Li et al., 2020; Kundu et al., 2020) also adapt without source data. Both rely on generative modeling and offline optimization of multiple models, and Kundu et al. (2020) has to alter source training. Tent does not require generative modeling by adversarial optimization, nor does it alter training. Tent simply adapts a single discriminative model by entropy minimization. This makes it much more computationally efficient and capable of online adaptation.

Test-Time Adaptation Tent adapts by test-time optimization and normalization.

Test-time training (TTT) (Sun et al., 2019b) and tent both optimize at test-time, but differ in their requirements. Self-supervision relies on proxy tasks with automatic labels, such as recognizing rotations of an image. Therefore TTT depends on the choice of proxy task (indeed, Sun et al. (2019b) caution that the proxy must be "both well-defined and non-trivial in the new domain"). Our test entropy loss is measured on the supervised task predictions without any proxy task. Furthermore, TTT must alter source training to include its self-supervised loss. Tent applies to any given model without altering its training or architecture.

Tent estimates the means and variances for batch normalization on the target data. Aligning feature statistics is common for domain adaptation (Gretton et al., 2009; Sun et al., 2017). For batch normalization, Li et al. (2017); Carlucci et al. (2017) separate the source and target statistics when training. In concurrent work, Schneider et al. (2020); Nado et al. (2020) estimate target statistics alone during testing, and show this boosts robustness to input corruptions. Tent builds on test-time normalization to further reduce generalization error.

Entropy Minimization Entropy minimization is a key regularizer for domain adaptation (Carlucci et al., 2017; Saito et al., 2019; Roy et al., 2019), semi-supervised learning (Grandvalet & Bengio, 2005; Lee, 2013; Berthelot et al., 2019), and few-shot learning (Dhillon et al., 2020). Regularizing entropy penalizes decision boundaries at high densities in the data distribution to thereby improve accuracy for distinct classes (Grandvalet & Bengio, 2005). These methods all regularize entropy during training in concert with other supervised and unsupervised losses on additional data. Tent is the first method to minimize entropy during testing, for adaptation to corruption and domain shift, without other losses or data. Entropic losses are common; our contribution is to exhibit the effectiveness of entropy as the sole loss for fully test-time adaptation.

Feature Modulation Modulation adjusts a model so that it varies with its input. Following signal processing usage, the modulation is usually simpler than the model, for example in its lower dimensionality. We optimize feature modulations instead of the full model for stable and efficient adaptation. We choose modulation by channel-wise affine transformation, because of its effectiveness in tandem with normalization (Ioffe & Szegedy, 2015; Wu & He, 2018), and for its ability to condition features on the input (Perez et al., 2018). These normalization and conditioning methods optimize the modulation at train time by a supervised loss, but keep it fixed during testing. We optimize the modulation at test time by an unsupervised loss, so that it can adapt to different targets.

# 6 CONCLUSION

Tent reduces generalization error for fully-test time adaptation by entropy minimization. This is remarkable in that its entropy objective, while unsupervised, is defined by the supervised model training. In effect, it seems that the model has learned enough to supervise itself on shifted data. While there are still gaps in accuracy on corruptions and different domains, and therefore more adaptation is needed, this is an encouraging step. Improvements from tent show that a model knows more than it can infer in one go: it generalizes more by adapting to feedback from its own predictions. Our fully test-time adaptation setting and experiments should encourage more exploration of what models may already know about the data distribution, and how this can be turned into further self-improvement.

# REFERENCES

Shaojie Bai, Vladlen Koltun, and J Zico Kolter. Multiscale deep equilibrium models. arXiv preprint arXiv:2006.08656, 2020.  
David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. Mixmatch: A holistic approach to semi-supervised learning. In NeurIPS, 2019.  
Fabio Maria Carlucci, Lorenzo Porzi, Barbara Caputo, Elisa Ricci, and Samuel Rota Bulo. Autodial: Automatic domain alignment layers. In 2017 IEEE International Conference on Computer Vision (ICCV), pp. 5077-5085. IEEE, 2017.  
Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In CVPR, 2016.  
Guneet Singh Dhillon, Pratik Chaudhari, Avinash Ravichandran, and Stefano Soatto. A baseline for few-shot image classification. In ICLR, 2020.  
Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015.  
J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In ICML, 2014.  
Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In ICML, 2015.  
Robert Geirhos, Carlos RM Temme, Jonas Rauber, Heiko H Schutt, Matthias Bethge, and Felix A Wichmann. Generalisation in humans and deep neural networks. In NeurIPS, 2018.  
Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. Imagenet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. In International Conference on Learning Representations, 2019.  
Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018.  
Priya Goyal, Piotr Dár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
Yves Grandvalet and Yoshua Bengio. Semi-supervised learning by entropy minimization. In NeurIPS, 2005.  
A. Gretton, AJ. Smola, J. Huang, M. Schmittfull, KM. Borgwardt, and B. Schölkopf. Covariate shift and local learning by distribution matching. In *Dataset Shift in Machine Learning*, pp. 131-160. MIT Press, Cambridge, MA, USA, 2009.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, June 2016.  
Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In ICLR, 2019.  
Dan Hendrycks, Norman Mu, Ekin D Cubuk, Barret Zoph, Justin Gilmer, and Balaji Lakshminarayanan. Augmix: A simple data processing method to improve robustness and uncertainty. In ICLR, 2020.  
Jonathan J. Hull. A database for handwritten text recognition research. TPAMI, 1994.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.  
A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. NeurIPS, 25, 2012.  
Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

Jogendra Nath Kundu, Naveen Venkat, R Venkatesh Babu, et al. Universal source-free domain adaptation. In CVPR, pp. 4544-4553, 2020.  
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Dong-Hyun Lee. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In ICML Workshop on challenges in representation learning, 2013.  
Rui Li, Qianfen Jiao, Wenming Cao, Hau-San Wong, and Si Wu. Model adaptation: Unsupervised domain adaptation without source data. In CVPR, June 2020.  
Yanghao Li, Naiyan Wang, Jianping Shi, Jiaying Liu, and Xiaodi Hou. Revisiting batch normalization for practical domain adaptation. In ICLRW, 2017.  
Zachary Nado, Shreyas Padhy, D Sculley, Alexander D'Amour, Balaji Lakshminarayanan, and Jasper Snoek. Evaluating prediction-time batch normalization for robustness under covariate shift. arXiv preprint arXiv:2006.10963, 2020.  
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. NeurIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019.  
Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In AAAI, 2018.  
Joaquin Quionero-Candela, Masashi Sugiyama, Anton Schwaighofer, and Neil D Lawrence. Dataset shift in machine learning. MIT Press, Cambridge, MA, USA, 2009.  
Ilija Radosavovic, Justin Johnson, Saining Xie, Wan-Yen Lo, and Piotr Dólar. On network design spaces for visual recognition. In ICCV, 2019.  
Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do ImageNet classifiers generalize to ImageNet? In ICML, 2019.  
Stephan R Richter, Zeeshan Hayden, and Vladlen Koltun. Playing for benchmarks. In ICCV, 2017.  
Subhankar Roy, Aliaksandr Siarohin, Enver Sangineto, Samuel Rota Bulo, Nicu Sebe, and Elisa Ricci. Unsupervised domain adaptation using feature-whitening and consensus loss. In CVPR, 2019.  
Evgenia Rusak, Lukas Schott, Roland S Zimmermann, Julian Bitterwolf, Oliver Bringmann, Matthias Bethge, and Wieland Brendel. A simple way to make neural networks robust against diverse image corruptions. In ECCV, 2020.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. ImageNet large scale visual recognition challenge. IJCV, 2015.  
Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell. Adapting visual category models to new domains. In European conference on computer vision, pp. 213-226. Springer, 2010.  
Kuniaki Saito, Donghyun Kim, Stan Sclaroff, Trevor Darrell, and Kate Saenko. Semi-supervised domain adaptation via minimax entropy. In ICCV, 2019.  
Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bringmann, Wieland Brendel, and Matthias Bethge. Improving robustness against common corruptions by covariate shift adaptation. arXiv preprint arXiv:2006.16971, 2020.  
C.E. Shannon. A mathematical theory of communication. Bell system technical journal, 27, 1948.  
Evan Shelhamer, Jonathan Long, and Trevor Darrell. Fully convolutional networks for semantic segmentation. PAMI, 2017.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

Baochen Sun, Jiashi Feng, and Kate Saenko. Correlation alignment for unsupervised domain adaptation. In Domain Adaptation in Computer Vision Applications, pp. 153-171. Springer, 2017.  
Yu Sun, Eric Tzeng, Trevor Darrell, and Alexei A Efros. Unsupervised domain adaptation through self-supervision. arXiv preprint arXiv:1909.11825, 2019a.  
Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A Efros, and Moritz Hardt. Test-time training for out-of-distribution generalization. arXiv preprint arXiv:1909.13231, 2019b.  
Eric Tzeng, Judy Hoffman, Trevor Darrell, and Kate Saenko. Simultaneous deep transfer across domains and tasks. In ICCV, 2015.  
Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell. Adversarial discriminative domain adaptation. In CVPR, 2017.  
Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, et al. Deep high-resolution representation learning for visual recognition. PAMI, 2020.  
Yuxin Wu and Kaiming He. Group normalization. In ECCV, 2018.  
Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neural networks? In NeurIPS, 2014.  
Richard Zhang, Phillip Isola, and Alexei A Efros. Split-brain autoencoders: Unsupervised learning by cross-channel prediction. In CVPR, 2017.  
Hengshuang Zhao, Jiaya Jia, and Vladlen Koltun. Exploring self-attention for image recognition. In CVPR, 2020.
