# THE ULTIMATE COMBO: BOOSTING ADVERSARIAL EXAMPLE TRANSFERABILITY BY COMPOSING DATA AUGMENTATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Transferring adversarial examples from surrogate (ML) models to evade target models is a common method for evaluating adversarial robustness in black-box settings. Researchers have invested substantial efforts to enhance transferability. Chiefly, attacks leveraging data augmentation have been found to help adversarial examples generalize better from surrogate to target models. Still, prior work has explored a limited set of augmentation techniques and their composition. To fill the gap, we conducted a systematic, comprehensive study of how data augmentation affects transferability. Particularly, we explored ten augmentation techniques of six categories originally proposed to help ML models generalize to unseen benign samples, and assessed how they influence transferability, both when applied individually and when composed. Our extensive experiments with the ImageNet dataset showed that simple color-space augmentations (e.g., color to greyscale) outperform the state of the art when combined with standard augmentations, such as translation and scaling. Additionally, except for two methods that may harm transferability, we found that composing augmentation methods impacts transferability monotonically (i.e., more methods composed  $\rightarrow$ $\geq$  transferability)—the best composition we found significantly outperformed the state of the art (e.g.,  $95.6\%$  vs.  $90.9\%$  average transferability from normally trained surrogates to other normally trained models). We provide intuitive, empirically supported explanations for why certain augmentations fail to improve transferability.

# 1 INTRODUCTION

Adversarial examples—variants of benign inputs minimally perturbed to induce misclassification at test time—have emerged as a profound challenge to machine learning (ML) (Biggio et al., 2013; Szegedy et al., 2014), calling its use in security- and safety-critical systems into question (e.g., Eykholt et al. (2018)). Many attacks have been proposed to generate adversarial examples in white-box settings, where adversaries are familiar with all the particularities of the attacked model (Papernot et al., 2016). By contrast, black-box attacks enable evaluating the vulnerability of ML in realistic settings, without access to the model (Papernot et al., 2016).

Attacks exploiting the transferability-property of adversarial examples (Szegedy et al., 2014) have received special attention. Namely, as adversarial examples produced against one model are often misclassified by others, transferability-based attacks produce adversarial examples against surrogate (a.k.a. substitute) white-box models to mislead black-box ones. To measure the risk of adversarial examples in black-box settings more accurately, researchers have proposed varied methods to enhance transferability beyond that of white-box attacks (e.g., Lin et al. (2020); Liu et al. (2017)).

Notably, attacks using data augmentation, such as translations (Dong et al., 2019) and scaling of pixel values (Lin et al., 2020), as a means to improve the generalizability of adversarial examples across models have accomplished state-of-the-art transferability rates. Still, to the best of our knowledge, previous transferability-based attacks have studied only four augmentation methods (see Section 3.1), out of many proposed in the data-augmentation literature (Shorten & Khoshgoftaar, 2019), primarily for reducing model overfitting. Hence, the extent to which different data-augmentation types boost transferability, either individually or when combined, remains largely unknown.

To fill the gap, we conducted a systematic, comprehensive study of how data-augmentation methods influence transferability. Specifically, alongside techniques considered in previous work, we studied how ten augmentation techniques pertaining to six categories impact transferability when applied individually or composed (Section 3). Integrating augmentation methods into attacks via a flexible framework we propose (Algorithm 1), we conducted extensive experiments using an ImageNet-compatible dataset and ten models, and measured transferability in diverse settings, including with and without defenses (Sections 4 and 5). Our results offer several interesting insights:

- Simple color-space augmentations outperform state-of-the-art transferability-based attacks when composed with standard augmentations (Section 5.1).  
- Transferability has a mostly monotonic relationship with data-augmentation techniques. Except for two augmentation methods that may harm transferability, composing additional augmentation methods either improves the preserves transferability (Section 5.2).  
- Out of  $2^{7}$  compositions explored, the best composition we found, ULTIMATECOMBO, outperforms state-of-the-art attacks by a large margin (Section 5.3).  
- We show empirical support to conjectures we raise concerning when data-augmentation techniques may be counterproductive to transferability (Section 5.4).

# 2 BACKGROUND AND RELATED WORK

We now review prior evasion attacks on ML (incl. via transferability), and defenses against them.

# 2.1 EVASION ATTACKS

Researchers have proposed numerous evasion attacks against ML models in general, and deep neural networks (DNNs) in particular. Many attacks assume adversaries have white-box access to models—i.e., adversaries know models' architectures and weights (e.g., Goodfellow et al. (2015); Szegedy et al. (2014); Carlini & Wagner (2017)). These attacks typically leverage first- or second-order optimizations to generate adversarial examples models would misclassify. For example, given an input  $x$  of class  $y$ , model weights  $\theta$ , and a loss function  $J$ , the Fast Gradient Sign method (FGSM) of Goodfellow et al. (2015), crafts an adversarial example  $\hat{x}$  using the loss gradients  $\nabla_{x}J(x,y,\theta)$ :

$$
\hat {x} = x + \epsilon * \operatorname {s i g n} \left(\nabla_ {x} J (x, y, \theta)\right)
$$

where  $\mathrm{sign}(\cdot)$  maps real numbers to  $-1, 0$ , or  $1$ , depending on their sign.

Following FGSM, researchers proposed various advanced attacks. Notably, iterative FGSM (I-FGSM) of Kurakin et al. (2017b) performs multiple gradient-ascent steps, updating  $\hat{x}$  iteratively to produce adversarial examples:

$$
\hat {x} _ {t + 1} = \operatorname {P r o j} _ {x} ^ {\epsilon} \left(\hat {x} _ {t} + \alpha \cdot \operatorname {s i g n} \left(\nabla_ {x} J (\hat {x} _ {t}, y, \theta)\right)\right)
$$

where  $\mathrm{Proj}_x^\epsilon (\cdot)$  projects the perturbation into  $\ell_{\infty}$ -norm  $\epsilon$ -ball centered at  $x$ ,  $\alpha$  is the step size, and  $\hat{x}_0 = x$ . The attacks we study in this work are based on I-FGSM.

In real-world settings, adversaries often do not have white-box access to victim models. Hence, researchers also studied black-box attacks in which adversaries only have query access to models. Certain attack types, such as score- and boundary-based attacks perform multiple queries, often in the order of several thousands, to produce adversarial examples (e.g., Brendel et al. (2018); Ilyas et al. (2019)) By contrast, attacks leveraging transferability (e.g., Goodfellow et al. (2015); Szegedy et al. (2014)) avoid querying victim models when generating adversarial examples. Instead, transferability-based attacks create adversarial examples against surrogate white-box models, that are also likely misclassified by other black-box models.

Attempts to explain the transferability phenomenon attribute it to gradient norm of the target model (i.e., its susceptibility to attacks), the smoothness of classification boundaries, and, primarily, the alignment of gradient directions between the surrogate and target models (Demontis et al., 2019; Yang et al., 2021). Said differently, for adversarial examples to transfer, the gradient directions of surrogates need to be similar to those of target models (i.e., attain high cosine similarity).

# Algorithm 1 MI-FGSM with data augmentation

Input: Benign sample  $x$ ; ground-truth label  $y$ ; loss function  $J(\cdot)$ ; model parameters  $\theta$ ; iterations #  $T$ ; momentum parameter  $\mu$ ; perturbation size  $\epsilon$ ; data-augmentation method  $D(\cdot)$ .

Output: Adversarial example  $\hat{x}$

1:  $\alpha = \epsilon / T$  
2:  $\hat{x}_0 = x$  
3:  $g_0 = 0$  
4: for  $t = 0$  to  $T - 1$  do  
5:  $\bar{g}_{t + 1} = \frac{1}{m}\sum_{i = 0}^{m - 1}\nabla_{x}\left(J\left(D(\hat{x}_{t})_{i},y,\theta\right)\right)$  
6:  $g_{t + 1} = \mu \cdot g_t + \frac{\bar{g}_{t + 1}}{\|\bar{g}_{t + 1}\|_1}$  
7:  $\hat{x}_{t + 1} = \operatorname{Proj}_x^\epsilon \left( \hat{x}_t + \alpha \cdot \operatorname{sign}(g_{t + 1}) \right)$  
8: return  $\hat{x} = \hat{x}_T$

Initialize adversarial example  
Initialize momentum

Expected loss gradient on augmented samples

Gradient with momentum  
Update adversarial example

Enhancing adversarial example transferability has become an active research area. One category of methods integrates momentum into attacks such as I-FGSM to avoid surrogate-specific optima and saddle points which may hinder transferability (e.g., Dong et al. (2018); Lin et al. (2020)). Another attack category employs specialized losses, such as reducing the variance of intermediate activations (Huang et al., 2019) or the mean loss of model ensembles (Liu et al., 2017), to enhance transferability. Lastly, a prominent family of attacks leverages data augmentation to enhance the generalizability of adversarial examples between models. For instance, Dong et al. (2019) boosted transferability by integrating random translations into I-FGSM, producing translation-invariant attacks, while Xie et al. (2019) improved transferability using random crops, creating size-invariant attacks. Evasion attacks incorporating data augmentation attain state-of-the-art transferability rates (Lin et al., 2020; Wang et al., 2021a). Nonetheless, prior work has only considered a restricted set of data-augmentation methods for enhancing transferability. By contrast, our work aims to investigate the role of data augmentation at enhancing transferability more systematically, by exploring how a more comprehensive set of augmentation types and their compositions affect transferability.

# 2.2 DEFENSES

Various defenses have been proposed to mitigate evasion attacks. Adversarial training—a procedure integrating correctly labeled adversarial examples in training—is one of the most practical and effective methods for enhancing adversarial robustness (e.g., Goodfellow et al. (2015); Tramér et al. (2018)). Other defense methods sanitize inputs prior to classification (e.g., Guo et al. (2018)); attempt to detect attacks (see Tramer (2022)); or seek to certify robustness in  $\epsilon$ -balls around inputs (e.g., Cohen et al. (2019); Salman et al. (2019)). Following standard practices in the literature (Wang et al., 2021a), we evaluate transferability-based attacks against a representative set of these defense.

# 3 DATA AUGMENTATION FOR ENHANCING TRANSFERABILITY

Data augmentation is traditionally used during training to reduce overfitting and improve generalizability (Shorten & Khoshgoftaar, 2019). Inspired by the original use, transferability-based attacks adopted data augmentation to limit overfitting to surrogate models and produce adversarial examples likely to generalize and be misclassified by victim models. Algorithm 1 depicts a general framework for integrating data augmentation into I-FGSM with momentum (MI-FGSM). In the framework, a method  $D(\cdot)$  augments the attack with  $m$  variants of the estimated adversarial example at each iteration. Consequently, the adversarial perturbation found by the attack increases the expected loss over transformed counterparts of the benign sample  $x$  (i.e., the distribution set by  $D(\cdot)$  given  $x$ ). Note that  $D(\cdot)'s$  output may include  $x$ .

The framework in Algorithm 1 is flexible, and can admit any data-augmentation method. We use it to describe previous attacks employing data augmentation and to systematically explore new ones. Next, we detail previous attacks, describe data augmentation methods we adopt for the first time to enhance transferability, and explain how these can be combined for best performance.

# 3.1 PREVIOUS ATTACKS LEVERAGING DATA AUGMENTATION

Previous work explored the following augmentation methods to set  $D(\cdot)$ .

Translations Using random translations of inputs, Dong et al. (2019) proposed a translation-invariant attack to promote transferability. They also offered an optimization to reduce the attack's time and space complexity by simply convolving the model's gradients (w.r.t. non-translated inputs) with a Gaussian kernel. While we use this optimization in the implementation for the interest of efficiency, we highlight that the attack can be well-captured by our framework.

Diverse Inputs Xie et al. (2019) proposed a size-invariant attack. Their augmentation procedure samples random crops from  $\hat{x}_t$  that are later resized per the model's input dimensionality.

Scaling Pixels Lin et al. (2020) showed that adversarial perturbations invariant to scaling pixel values transfer with higher success between DNNs. In their case,  $D(\cdot)$  produces  $m$  samples such that  $D(x)_i = \frac{x}{2^i}$  for  $i\in \{0,1,\dots,m - 1\}$ , where  $m = 5$  by default.

Admix Wang et al. (2021a) assumed that the adversary has a gallery of images from different classes and adopted augmentations similar to MixUp (Zhang et al., 2018a). For each sample  $x'$  from the gallery, Admix augments attacks with  $m$  (typically set to 5) samples, such that  $D(x, x')_i = \frac{1}{2^i} \cdot (\hat{x}_t + \eta \cdot x')$ , where  $i \in \{0, 1, \dots, m-1\}$ , and  $\eta \in [0, 1]$  is set to 0.2 by default. Notably, Admix degenerates to pixel scaling when  $\eta = 0$ .

The leading transferability-based attacks compose (1) diverse inputs, scaling, and translations (Lin et al.'s (2020) DST-MI-FGSM attack); or (2) Admix, diverse inputs, and translation (Wang et al.'s (2021a) Admix-DT-MI-FGSM attack). We describe how these compositions operate in Section 3.3.

# 3.2 NEW AUGMENTATIONS FOR ENHANCING TRANSFERABILITY

While prior work studied the effect of spatial transformations (i.e., translations and diverse inputs), pixel scaling, and mixing on transferability, a substantially wider range of data-augmentation methods exist (Shorten & Khoshgoftaar, 2019). Yet, the impact of these on transferability remains unknown. To fill the gap, we examined Shorten & Khoshgoftaar's (2019) recent survey on data augmentation for reducing overfitting in deep learning and identified ten representative methods of six categories that may boost transferability. We present them in what follows, one category at a time.

# 3.2.1 COLOR-SPACE TRANSFORMATIONS

Potentially the simplest of all augmentation types are those applied in color-space. Given images represented as three-channel tensors, methods in this category manipulate pixel values only based on information encoded in the tensors. We evaluate four color-space transformations.

Color Jitter (CJ) This transformation applies random color manipulation (Wu et al., 2015). In this work, we consider random adjustments of pixel values within a pre-defined range in terms of hue, contrast, saturation, and brightness around original values.

Fancy Principle Component Analysis (fPCA) Used in AlexNet (Krizhevsky et al., 2017), fPCA adds noise to the image proportionally to the variance in each channel. Given an RGB image, fPCA adds the following quantity to each image pixel:

$$
\left[ \mathbf {p} _ {1}, \mathbf {p} _ {2}, \mathbf {p} _ {3} \right] \left[ \alpha_ {1} \lambda_ {1}, \alpha_ {2} \lambda_ {2}, \alpha_ {3} \lambda_ {3} \right] ^ {T},
$$

where  $p_i$  and  $\lambda_i$  are the  $i^{\text{th}}$  eigenvector and eigenvalue of the  $3 \times 3$  covariance matrix of RGB pixels, respectively, and  $\alpha_i$  is sampled once per image from Gaussian distribution  $\mathcal{N}(0, 0.1)$ .

Channel shuffle (CS) Included in ShuffleNet training (Zhang et al., 2018b), CS simply swaps the orders of the image's RGB channels at random.

Greyscale (GS) This simple augmentation converts images into greyscale (replicating it three times to obtain an RGB representation). Mathematically, the conversion is calculated by  $\omega_{R} \cdot x^{R} + \omega_{G} \cdot x^{G} + \omega_{B} \cdot x^{B}$ , where  $x^{R}, x^{G}$ , and  $x^{B}$ , correspond to the RGB channels, respectively, and  $\omega_{R}, \omega_{G}$ , and  $\omega_{B}$ , all  $\in [0,1]$ , denote the channel weights, and sum up to 1.

# 3.2.2 RANDOM ERASING

Inspired by dropout regularization, random erasing (RE) helps ML models focus on descriptive features of images and promote robustness to occlusions (Zhong et al., 2020). To do so, randomly selected rectangular regions in images are replaced by masks composed of random pixel values. Similarly to RE, CutOut masks out regions of inputs to improve DNNs' accuracy (DeVries & Taylor, 2017). The main difference from e is that CutOut uses fixed masking values, and may perform less aggressive masking when selected regions lie outside the image.

# 3.2.3 KERNEL FILTERS

Convolving images with kernels of different types can produce certain effects, such as blurring (via Gaussian kernels), sharpening (via edge filters), or edge enhancement. We study the effect of sharpening (Sharp) on transferability with edge-enhancement filters.

# 3.2.4 MIXING IMAGES

As a form of vicinal risk minimization, some augmentation methods mix images together, creating virtual examples for training. The MixUp method, the cornerstone behind Admix, computes weighted sums of images (Zhang et al., 2018a). By contrast, we consider CutMix, which replaces a region within one image with a region from another image picked from a gallery (Yun et al., 2019).

# 3.2.5 NEURAL TRANSFER

Augmentations building upon neural transfer (NeuTrans) preserve images' semantic content while changing their style. We consider Gatys et al.'s (2015) generative model for artistic style transfer. Particularly, we use their approach to transfer image styles to that of Picasso's 1907 self-portrait.

# 3.2.6 META-LEARNING-INSPIRED AUGMENTATIONS

Meta-learning is a subfield of ML studying how ML algorithms can optimize other learning algorithms (Hospedales et al., 2021). In the context of data augmentation, algorithms such as AutoAugment have been proposed to train controllers to select an appropriate augmentation method to avoid overfitting (Cubuk et al., 2019). We use the pre-trained AutoAugment controller, encoded as a recurrent neural network, to select augmentation methods and their magnitude from a set of 13 augmentation methods.

# 3.3 COMPOSING AUGMENTATIONS

There are two primary ways to compose data-augmentation methods in attack algorithms, namely: parallel and serial composition. In parallel composition, augmentation methods are applied independently on the input, and their outputs are aggregated (i.e., taking their union) to augment attacks. By contrast, serial composition applies augmentation methods sequentially, one after the other, where the first method operates on the original sample, and each of the subsequent augmentation functions is applied on its predecessor's output. State-of-the-art attacks (i.e., DST-MI-FGSM and Admix-DT-MI-FGSM) use serial composition. By contrast, in this work, we consider a substantially larger number of augmentation methods, which may lead to prohibitive memory requirements in the case of serial composition. Additionally, because the order of applying certain augmentations matters (e.g., GS then CutMix leads to different outcome that CutMix followed by GS), exploring a meaningful number of serial compositions (out of an order of 10! possibilities) becomes virtually impossible. Accordingly, we mainly consider parallel composition between data-augmentation methods. The only settings we use serial composition are when combining augmentations with translation, scaling, and diverse inputs, for consistency with prior work (e.g., Wang et al. (2021a)). A few serial compositions we tested were significantly outperformed by their parallel counterparts. While inconclusive, the results hint that serially composing augmentations may not be a promising direction for enhancing transferability.

# 4 EXPERIMENTAL SETUP

Now we turn to the setup of our experiments, including the data, models, and attack configurations.

Data Following prior work (Dong et al., 2019; Gao et al., 2020), we used an ImageNet-compatible dataset (Russakovsky et al., 2015) originally collected for the NeurIPS 2017 evasion-attack competition. The dataset contains 1,000 images, covering 430 ImageNet classes, with a maximum of five images per class. These images are mostly classified correctly—the normally trained classifiers we considered (see below) obtain  $96.7\%$  benign accuracy.

Models We used ten DNNs to transfer attacks from (as surrogates) and to (as targets). Six models were normally trained, while others were adversarially trained. Specifically, for normally trained models, we selected: Inception-v3 (Inc-v3) (Szegedy et al., 2016)); Inception-v4 (Inc-v4); Inception-ResNet-v2 (IncRes-v2) (Szegedy et al., 2017)); ResNet-v2-50 (Res-50); ResNet-v2-101 (Res-101); and ResNet-v2-152 (Res-152) (He et al., 2016). For adversarially trained models, we selected: Inception-v3-adv (Inc-v3\_adv) (Kurakin et al., 2017a); ens3-Inception-v3 (Inc-v3\_ens3); ens4-Inception-v3 (Inc-v3\_ens4); and ens-adv-Inception-ResNet-v2 (IncRes-v2\_ens) (Tramér et al., 2018). These models constitute a superset of ones considered in prior work (Wang et al., 2021a). We obtained the models' PyTorch implementations and weights from a public GitHub repository.

Attack Parameters We tested standard attack configurations, in line with prior work (Dong et al., 2018; Wang et al., 2021a). More precisely, we evaluated MI-FGSM-based attacks, bounded in  $\ell_{\infty}$ -norm, with a maximum perturbation norm of  $\epsilon = \frac{16}{255}$ . The attacks were not targeted toward specific classes. Accordingly, we evaluated them via transferability rates—the percentages of attempts at which adversarial examples created against surrogates were misclassified by victims. We set the MI-FGSM decay factor  $\mu = 1.0$  (similarly to Wang et al. (2021b)), and the number of iterations  $T = 10$ . We used DST-MI-FGSM and Admix-DT-MI-FGSM, two state-of-the-art transferability-based attacks, as baselines. The exact parameters used for all augmentation methods are reported in Appendix A. Using our parameters with the baselines, we managed to reproduce and improve transferability rates reported in prior work (Wang et al., 2021a).

We measured transferability rates from each normally trained model, as surrogate, to every other model, as target. Furthermore, for adversarially trained target models, we employed Liu et al.'s (2017) method and evaluated transferability from an ensemble of normally trained models as surrogates. Specifically, we considered an ensemble containing Inc-v4, Res-50, Res-101, and Res-152.

# 5 EXPERIMENTAL RESULTS

This section summarizes our findings. We start by evaluating individual augmentation methods and standard combinations with scaling, diverse inputs, and translations (Section 5.1). We then turn to analyzing all possible compositions between different augmentation types to assess whether transferability typically improves when considering additional augmentations (Section 5.2). Our analysis helped us identify the best performing composition for boosting transferability, denoted by ULTIMATECOMBO, outperforming state-of-the-art attacks. Section 5.3 reports rigorous comparisons between ULTIMATECOMBO and the baselines, including against defended models. Finally, we help develop intuition for when augmentations may or may not help improve transferability (Section 5.4).

# 5.1 COLOR-SPACE AUGMENTATIONS OUTPERFORM THE STATE OF THE ART

Initially, we evaluated transferability integrating a single augmentation at a time in attacks, or when composing individual augmentations with diverse inputs, scaling, and translation (DST), as is standard (Lin et al., 2020; Wang et al., 2021a). We found that considering each of the ten augmentations individually does not lead to competitive performance with the baselines. However, composing individual augmentations with DST enhanced transferability markedly. Surprisingly, augmentations in color-space fared particularly well, outperforming the baselines and advanced augmentation methods in most cases. Specifically, composing GS with DST (GS-DST-MI-FGSM attack) performed

best in this setting. Table 1 reports the transferability rates from four normally trained models to other models. It can be immediately seen that GS-DST-MI-FGSM attains higher transferability than the baselines (93.6% vs.  $\leq 90.9\%$ , on avg.). The same trend holds when transferring adversarial examples from normally trained models and the ensemble of models to adversially trained victims (Tables 2 and 3, respectively). According to a paired t-test, the differences between GS-DST-MI-FGSM and the baselines across different surrogate and target models were statistically significant ( $p < 0.01$ ). Transferability results using other surrogate models are included in Appendix B.

Table 1: Transferability rates (\%) from normally trained surrogates (rows) to normally trained target models (columns) using four attacks. All attacks are black-box, except for when the surrogate and target models are the same (i.e., white-box attacks).  

<table><tr><td>Model</td><td>Attack</td><td>Inc-v3</td><td>Inc-v4</td><td>Res-50</td><td>Res-101</td><td>Res-152</td><td>IncRes-v2</td></tr><tr><td rowspan="4">Inc-v3</td><td>Admix-DT-MI-FGSM</td><td>99.5</td><td>92.3</td><td>88.5</td><td>87.0</td><td>85.3</td><td>90.9</td></tr><tr><td>DST-MI-FGSM</td><td>100.0</td><td>92.9</td><td>89.5</td><td>87.2</td><td>86.4</td><td>91.2</td></tr><tr><td>GS-DST-MI-FGSM</td><td>100.0</td><td>95.6</td><td>93.7</td><td>91.8</td><td>90.9</td><td>94.9</td></tr><tr><td>ULTIMATECOMBO</td><td>100.0</td><td>98.0</td><td>95.1</td><td>94.3</td><td>92.7</td><td>97.1</td></tr><tr><td rowspan="4">Inc-v4</td><td>Admix-DT-MI-FGSM</td><td>93.7</td><td>99.3</td><td>86.7</td><td>84.9</td><td>84.7</td><td>89.6</td></tr><tr><td>DST-MI-FGSM</td><td>94.4</td><td>100.0</td><td>90.2</td><td>88.1</td><td>88.2</td><td>92.8</td></tr><tr><td>GS-DST-MI-FGSM</td><td>96.5</td><td>100.0</td><td>94.1</td><td>92.5</td><td>93.0</td><td>95.4</td></tr><tr><td>ULTIMATECOMBO</td><td>98.1</td><td>99.9</td><td>94.8</td><td>95.0</td><td>94.6</td><td>96.8</td></tr><tr><td rowspan="4">Res-101</td><td>Admix-DT-MI-FGSM</td><td>82.6</td><td>78.1</td><td>93.5</td><td>97.4</td><td>93.9</td><td>79.3</td></tr><tr><td>DST-MI-FGSM</td><td>86.7</td><td>83.2</td><td>97.3</td><td>99.9</td><td>96.6</td><td>84.9</td></tr><tr><td>GS-DST-MI-FGSM</td><td>89.0</td><td>84.8</td><td>97.6</td><td>99.8</td><td>97.7</td><td>87.6</td></tr><tr><td>ULTIMATECOMBO</td><td>93.0</td><td>90.4</td><td>98.1</td><td>99.7</td><td>97.8</td><td>91.8</td></tr><tr><td rowspan="4">IncRes-v2</td><td>Admix-DT-MI-FGSM</td><td>93.8</td><td>91.9</td><td>91.1</td><td>90.6</td><td>89.5</td><td>98.9</td></tr><tr><td>DST-MI-FGSM</td><td>95.8</td><td>94.2</td><td>93.5</td><td>92.0</td><td>92.4</td><td>99.8</td></tr><tr><td>GS-DST-MI-FGSM</td><td>96.5</td><td>95.6</td><td>95.5</td><td>94.2</td><td>94.7</td><td>100.0</td></tr><tr><td>ULTIMATECOMBO</td><td>98.2</td><td>97.1</td><td>96.3</td><td>96.5</td><td>95.7</td><td>100.0</td></tr></table>

Table 2: Transferability rates (\%) from normally trained surrogates (rows) to adversarially trained target models (columns) using four attacks.  

<table><tr><td>Model</td><td>Attack</td><td>Inc-v3adv</td><td>Inc-v3ens3</td><td>Inc-v3ens4</td><td>IncRes-v2ens</td></tr><tr><td rowspan="4">Inc-v3</td><td>Admix-DT-MI-FGSM</td><td>84.6</td><td>84.3</td><td>83.5</td><td>70.8</td></tr><tr><td>DST-MI-FGSM</td><td>81.3</td><td>81.2</td><td>77.7</td><td>61.4</td></tr><tr><td>GS-DST-MI-FGSM</td><td>87.3</td><td>88.5</td><td>85.5</td><td>72.2</td></tr><tr><td>ULTIMATECOMBO</td><td>88.2</td><td>88.7</td><td>86.7</td><td>72.6</td></tr><tr><td rowspan="4">Inc-v4</td><td>Admix-DT-MI-FGSM</td><td>82.7</td><td>83.3</td><td>81.3</td><td>73.7</td></tr><tr><td>DST-MI-FGSM</td><td>80.6</td><td>81.8</td><td>80.8</td><td>70.5</td></tr><tr><td>GS-DST-MI-FGSM</td><td>87.6</td><td>89.5</td><td>87.2</td><td>78.0</td></tr><tr><td>ULTIMATECOMBO</td><td>88.6</td><td>89.4</td><td>88.4</td><td>78.2</td></tr><tr><td rowspan="4">Res-101</td><td>Admix-DT-MI-FGSM</td><td>79.5</td><td>80.4</td><td>78.6</td><td>71.2</td></tr><tr><td>DST-MI-FGSM</td><td>78.9</td><td>78.7</td><td>76.7</td><td>68.7</td></tr><tr><td>GS-DST-MI-FGSM</td><td>82.3</td><td>83.7</td><td>81.1</td><td>73.2</td></tr><tr><td>ULTIMATECOMBO</td><td>83.5</td><td>86.7</td><td>82.8</td><td>76.8</td></tr><tr><td rowspan="4">IncRes-v2</td><td>Admix-DT-MI-FGSM</td><td>89.0</td><td>89.0</td><td>88.7</td><td>87.1</td></tr><tr><td>DST-MI-FGSM</td><td>87.2</td><td>89.2</td><td>86.4</td><td>82.9</td></tr><tr><td>GS-DST-MI-FGSM</td><td>91.1</td><td>92.2</td><td>90.0</td><td>88.2</td></tr><tr><td>ULTIMATECOMBO</td><td>92.2</td><td>92.6</td><td>92.0</td><td>88.5</td></tr></table>

# 5.2 THE MONOTONICITY OF TRANSFERABILITY WHEN ADDING AUGMENTATIONS

We wanted to evaluate whether transferability is monotonic in the number of augmentation types considered—i.e., whether composing more techniques increases, or at least does not harm, transferability. To this end, we selected the best performing augmentation method of each of the six categories presented in Section 3.2 as well as DST-MI-FGSM, and evaluated all  $2^{7}$  (=128) compositions possible (per Section 3.3). More precisely, we tested every possible combination of GS, CutOut, Sharp, NeuTrans, AutoAugment, Admix, and DST-MI-FGSM. Given a composition, we

Table 3: Transferability rates (\%) from an ensemble of normally trained surrogates (containing Incv4, Res-50, Res-101 and Res-152) to adversarially trained target models.  

<table><tr><td>Attack</td><td>Inc-v3adv</td><td>Inc-v3ens3</td><td>Inc-v3ens4</td><td>IncRes-v2ens</td></tr><tr><td>DST-MI-FGSM</td><td>89.0</td><td>90.0</td><td>87.6</td><td>82.4</td></tr><tr><td>Admix-DT-MI-FGSM</td><td>90.1</td><td>90.5</td><td>89.4</td><td>84.7</td></tr><tr><td>GS-DST-MI-FGSM</td><td>92.4</td><td>93.5</td><td>92.5</td><td>88.7</td></tr><tr><td>ULTIMATECOMBO</td><td>93.6</td><td>95.2</td><td>93.7</td><td>91.2</td></tr></table>

produced adversarial examples against the Inc-v3 DNN as surrogate, and computed the expected transferability rate against all other nine DNNs, both normally and adversarially trained. Then, for every pair of attacks differing only in whether a single augmentation method was incorporated in the composition, we tested whether adding the augmentation method improved transferability.

The results reflected a mostly monotonic relationship between transferability and augmentations. Except for NeuTrans and Sharp, which sometimes harmed transferability when considered within a composition, adding augmentation method increased or preserved transferability. Notably, comparing all compositions enabled us to find that a composition of all seven augmentation methods except for NeuTrans attained the best transferability. We call this composition the ULTIMATECOMBO.

# 5.3 THE MOST EFFECTIVE COMBINATION

We evaluated ULTIMATECOMBO extensively, testing transferability to normally and adversarially trained DNNs. As shown in Table 1, ULTIMATECOMBO obtained higher transferability to normally trained models than the baselines (95.6% vs. ≤90.9% avg. transferability) and GS-DST-MI-FGSM, when normally trained models were used as surrogates. Furthermore, ULTIMATECOMBO achieved the best performance also when transferring attacks from normally trained to adversarially trained DNNs (Table 2; 86.0% vs. ≤81.7% avg. transferability). Transferring adversarial examples crafted by ULTIMATECOMBO using an ensemble of models increased transferability further (Table 3; 93.4% avg. transferability). Per a paired t-test, the differences between ULTIMATECOMBO and the baselines over all pairs of surrogates and targets considered are statistically significant ( $p < 0.01$ ).

Table 4: Transferability rates  $(\%)$  from an ensemble of normally trained surrogates (Inc-v4, Res-50, Res-101 and Res-152) to models defended by provable methods or input transformations.  

<table><tr><td>Attack</td><td>Bit-Red</td><td>NRP</td><td>RS</td><td>ARS</td></tr><tr><td>DST-MI-FGSM</td><td>85.3</td><td>40.7</td><td>84.2</td><td>39.8</td></tr><tr><td>Admix-DT-MI-FGSM</td><td>86.4</td><td>39.4</td><td>86.6</td><td>43.0</td></tr><tr><td>ULTIMATECOMBO</td><td>87.5</td><td>47.7</td><td>88.4</td><td>43.5</td></tr></table>

Besides adversarially trained models, we evaluated ULTIMATECOMBO's transferability against four defenses. Two defenses, bit reduction (Bit-Red) (Xu et al., 2018) and neural representation purification (NRP) (Naseer et al., 2020), transform inputs to sanitize adversarial perturbations. The two others, randomized smoothing (RS) (Cohen et al., 2019) and randomized smoothing with adversarial training (ARS) (Salman et al., 2019) offer provable robustness guarantees. We used the defenses with default parameters (see Appendix C), and transferred adversarial examples crafted against the ensemble of normally trained models. Results are shown in Table 4. Similar to other settings, here too, ULTIMATECOMBO outperformed the baselines (66.8% vs. ≤63.9% avg. transferability).

# 5.4 WHEN DO AUGMENTATIONS FAIL TO IMPROVE TRANSFERABILITY?

While augmentation methods mostly increased transferability, in some cases they were counterproductive. Particularly, NeuTrans and Sharp decreased transferability when composed with certain methods. We conducted simple experiments as a preliminary assessment of two conjectures we had concerning when augmentations may harm transferability.

First, we expected augmentation methods that harm model accuracy on benign samples to be less conducive for transferability. As DNNs fail to generalize to samples produced by such augmentation

Table 5: Benign accuracy (%) after applying data augmentation methods. Rows are sorted in a descending order of average transferability.  

<table><tr><td>Augmentation</td><td>Inc-v3</td><td>Inc-v4</td><td>Res-50</td><td>Res-101</td><td>Res-152</td><td>IncRes-v2</td><td>Avg.</td></tr><tr><td>None</td><td>96.2</td><td>97.4</td><td>94.5</td><td>96.3</td><td>95.8</td><td>99.8</td><td>96.7</td></tr><tr><td>CS</td><td>94.0</td><td>95.7</td><td>95.3</td><td>94.6</td><td>95.4</td><td>99.5</td><td>95.8</td></tr><tr><td>fPCA</td><td>91.6</td><td>96.8</td><td>89.9</td><td>92.8</td><td>93.6</td><td>99.4</td><td>94.0</td></tr><tr><td>CJ</td><td>90.0</td><td>92.3</td><td>90.3</td><td>90.3</td><td>91.4</td><td>96.8</td><td>91.9</td></tr><tr><td>Admix</td><td>86.7</td><td>91.6</td><td>86.8</td><td>88.9</td><td>89.7</td><td>94.6</td><td>89.7</td></tr><tr><td>CutOut</td><td>86.5</td><td>89.2</td><td>85.7</td><td>87.2</td><td>88.6</td><td>92.3</td><td>88.2</td></tr><tr><td>GS</td><td>86.6</td><td>90.3</td><td>84.7</td><td>87.6</td><td>86.5</td><td>92.7</td><td>88.1</td></tr><tr><td>AutoAugment</td><td>82.9</td><td>86.2</td><td>82.1</td><td>84.3</td><td>84.4</td><td>89.8</td><td>85.0</td></tr><tr><td>Sharp</td><td>69.5</td><td>87.3</td><td>71.5</td><td>76.7</td><td>75.5</td><td>90.6</td><td>78.5</td></tr><tr><td>NeuTrans</td><td>24.4</td><td>25.4</td><td>24.2</td><td>27.0</td><td>24.0</td><td>32.7</td><td>26.3</td></tr></table>

methods, we anticipated that adversarial perturbations informed by augmented samples would not be informative. To support this conjecture, we tested the normally trained DNNs' accuracy on benign samples transformed by each augmentation method. As can be seen from Table 5, the methods least conducive for transferability (NeuTrans and Sharp) harmed the DNN accuracy the most  $(6.5\% - 58.7\%)$  lower accuracy than other methods), supporting our conjecture.

Table 6: Cosine similarities between gradients of benign images computed on Inc-v3 after applying augmentation methods composed with DST-MI-FGSM, and gradients of other normally trained models on benign images. Rows are sorted in a descending order of average cosine similarity.  

<table><tr><td>Augmentation</td><td>Inc-v4</td><td>Res-50</td><td>Res-101</td><td>Res-152</td><td>IncRes-v2</td><td>Avg.</td></tr><tr><td>CutOut</td><td>0.568</td><td>0.583</td><td>0.581</td><td>0.574</td><td>0.591</td><td>0.579</td></tr><tr><td>CS</td><td>0.565</td><td>0.578</td><td>0.576</td><td>0.570</td><td>0.590</td><td>0.576</td></tr><tr><td>None</td><td>0.564</td><td>0.575</td><td>0.573</td><td>0.568</td><td>0.586</td><td>0.573</td></tr><tr><td>Admix</td><td>0.563</td><td>0.575</td><td>0.573</td><td>0.567</td><td>0.586</td><td>0.573</td></tr><tr><td>CJ</td><td>0.560</td><td>0.575</td><td>0.573</td><td>0.568</td><td>0.584</td><td>0.572</td></tr><tr><td>GS</td><td>0.559</td><td>0.572</td><td>0.569</td><td>0.563</td><td>0.582</td><td>0.569</td></tr><tr><td>AutoAugment</td><td>0.558</td><td>0.569</td><td>0.567</td><td>0.562</td><td>0.579</td><td>0.567</td></tr><tr><td>fPCA</td><td>0.560</td><td>0.568</td><td>0.566</td><td>0.561</td><td>0.578</td><td>0.567</td></tr><tr><td>NeuTrans</td><td>0.546</td><td>0.556</td><td>0.554</td><td>0.549</td><td>0.565</td><td>0.554</td></tr><tr><td>Sharp</td><td>0.548</td><td>0.548</td><td>0.545</td><td>0.540</td><td>0.558</td><td>0.548</td></tr></table>

Prior work demonstrated that gradient alignment between surrogates and targets is needed for transferability (Demontis et al., 2019). Thus, we expected augmentation methods that estimate target model gradients more accurately to increase transferability further. To assess this conjecture, we evaluated the cosine similarity between the gradients of the Inc-v3 model while using augmentations composed with DST applied to benign samples, and the gradients of other normally trained models on (untransformed) benign samples. The results (Table 6) show some support to the conjecture—NeuTrans and Sharp led to lower cosine similarities with target models' gradients. Yet, the differences in cosine similarities between augmentation methods were small ( $\leq 0.031$ , on avg.).

# 6 CONCLUSION

Our study uncovered a mostly monotonic relationship between data-augmentation methods and transferability, and helped us identify a composition of data-augmentation methods, ULTIMATE-COMBO, that outperforms previously proposed methods when integrated into attacks. The resulting attack should be considered as a standard baseline in follow-up work on transferability. In addition for assessing the vulnerability of ML models in black-box settings, it would be interesting to evaluate whether the ULTIMATECOMBO-based attack advances methods leveraging adversarial examples for defensive purposes, by deceiving adversaries (e.g., to attain privacy (Cherepanova et al., 2021; Shetty et al., 2018)). Our work also puts forward conjectures for when augmentation techniques are expected to improve transferability, and offers some empirical support. In the future, it would be informative to develop formal explanations.

# REPRODUCIBILITY STATEMENT

In the interest of reproducibility, we make our code publicly available at the following repository: https://tinyurl.com/UltimateComboICLR.

# REFERENCES

Battista Biggio, Igino Corona, Davide Maiorca, Blaine Nelson, Nedim Šrndić, Pavel Laskov, Giorgio Giacinto, and Fabio Roli. Evasion attacks against machine learning at test time. In Proc. ECML/PKDD, 2013.  
Wieland Brendel, Jonas Rauber, and Matthias Bethge. Decision-based adversarial attacks: Reliable attacks against black-box machine learning models. In Proc. ICLR, 2018.  
Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In Proc. IEEE S&P, 2017.  
Valeriia Cherepanova, Micah Goldblum, Harrison Foley, Shiyuan Duan, John Dickerson, Gavin Taylor, and Tom Goldstein. Lowkey: Leveraging adversarial attacks to protect social media users from facial recognition. In Proc. ICLR, 2021.  
Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In Proc. ICML, 2019.  
Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le. AutoAugment: Learning augmentation policies from data. In Proc. CVPR, 2019.  
Ambra Demontis, Marco Melis, Maura Pintor, Matthew Jagielski, Battista Biggio, Alina Oprea, Cristina Nita-Rotaru, and Fabio Roli. Why do adversarial attacks transfer? Explaining transferability of evasion and poisoning attacks. In Proc. USENIX Security, 2019.  
Terrance DeVries and Graham W Taylor. Improved regularization of convolutional neural networks with cutout. arXiv preprint 1708.04552, 2017.  
Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, and Jianguo Li. Boosting adversarial attacks with momentum. In Proc. CVPR, 2018.  
Yinpeng Dong, Tianyu Pang, Hang Su, and Jun Zhu. Evading defenses to transferable adversarial examples by translation-invariant attacks. In Proc. CVPR, 2019.  
Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Chaowei Xiao, Atul Prakash, Tadayoshi Kohno, and Dawn Song. Robust physical-world attacks on deep learning visual classification. In Proc. CVPR, 2018.  
Lianli Gao, Qilong Zhang, Jingkuan Song, Xianglong Liu, and Heng Tao Shen. Patch-wise attack for fooling deep neural network. In Proc. ECCV, 2020.  
Leon A Gatys, Alexander S Ecker, and Matthias Bethge. A neural algorithm of artistic style. arXiv preprint 1508.06576, 2015.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In Proc. ICLR, 2015.  
Chuan Guo, Mayank Rana, Moustapha Cisse, and Laurens Van Der Maaten. Countering adversarial images using input transformations. In Proc. ICLR, 2018.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In Proc. ECCV, 2016.  
Timothy Hesperides, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey. IEEE PAMI, 44(9):5149-5169, 2021.  
Qian Huang, Isay Katsman, Horace He, Zeqi Gu, Serge Belongie, and Ser-Nam Lim. Enhancing adversarial example transferability with an intermediate level attack. In Proc. ICCV, 2019.

Andrew Ilyas, Logan Engstrom, and Aleksander Madry. Prior convictions: Black-box adversarial attacks with bandits and priors. In Proc. ICLR, 2019.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. ImageNet classification with deep convolutional neural networks. CACM, 60(6):84-90, 2017.  
Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial machine learning at scale. 2017a.  
Alexey Kurakin, Ian J Goodfellow, and Samy Bengio. Adversarial examples in the physical world. In Proc. ICLRW. 2017b.  
Jiadong Lin, Chuanbiao Song, Kun He, Liwei Wang, and John E Hopcroft. Nesterov accelerated gradient and scale invariance for adversarial attacks. In Proc. ICLR, 2020.  
Yanpei Liu, Xinyun Chen, Chang Liu, and Dawn Song. Delving into transferable adversarial examples and black-box attacks. In Proc. ICLR, 2017.  
Muzammal Naseer, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Fatih Porikli. A self-supervised approach for adversarial robustness. In Proc. CVPR, 2020.  
Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z Berkay Celik, and Ananthram Swami. The limitations of deep learning in adversarial settings. In Proc. Euro S&P, 2016.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. ImageNet large scale visual recognition challenge. IJCV, 115(3):211-252, 2015.  
Hadi Salman, Jerry Li, Ilya Razenshteyn, Pengchuan Zhang, Huan Zhang, Sebastien Bubeck, and Greg Yang. Provably robust deep learning via adversarially trained smoothed classifiers. In Proc. NeurIPS, 2019.  
Rakshith Shetty, Bernt Schiele, and Mario Fritz. A4NT: Author attribute anonymity by adversarial training of neural machine translation. In Proc. USENIX Security, 2018.  
Connor Shorten and Taghi M Khoshgoftaar. A survey on image data augmentation for deep learning. Journal of big data, 6(1):1-48, 2019.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In Proc. ICLR, 2014.  
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proc. CVPR, 2016.  
Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alexander A Alemi. Inception-v4, Inception-ResNet and the impact of residual connections on learning. In Proc. AAAI, 2017.  
Florian Tramer. Detecting adversarial examples is (nearly) as hard as classifying them. In Proc. ICML, 2022.  
Florian Tramér, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel. Ensemble adversarial training: Attacks and defenses. In Proc. ICLR, 2018.  
Xiaosen Wang, Xuanran He, Jingdong Wang, and Kun He. Admix: Enhancing the transferability of adversarial attacks. In Proc. ICCV, 2021a.  
Xiaosen Wang, Jiadong Lin, Han Hu, Jingdong Wang, and Kun He. Boosting adversarial transferability through enhanced momentum. In Proc. BMVC, 2021b.  
Ren Wu, Shengen Yan, Yi Shan, Qingqing Dang, and Gang Sun. Deep image: Scaling up image recognition. arXiv preprint 1501.02876, 2015.  
Cihang Xie, Zhishuai Zhang, Yuyin Zhou, Song Bai, Jianyu Wang, Zhou Ren, and Alan L Yuille. Improving transferability of adversarial examples with input diversity. In Proc. CVPR, 2019.

Weilin Xu, David Evans, and Yanjun Qi. Feature squeezing: Detecting adversarial examples in deep neural networks. In Proc. NDSS, 2018.  
Zhuolin Yang, Linyi Li, Xiaojun Xu, Shiliang Zuo, Qian Chen, Pan Zhou, Benjamin Rubinstein, Ce Zhang, and Bo Li. TRS: Transferability reduced ensemble via promoting gradient diversity and model smoothness. In Proc. NeurIPS, 2021.  
Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. CutMix: Regularization strategy to train strong classifiers with localizable features. In Proc. ICCV, 2019.  
Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In Proc. ICLR, 2018a.  
Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. ShuffleNet: An extremely efficient convolutional neural network for mobile devices. In Proc. CVPR, 2018b.  
Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In Proc. AAAI, 2020.
