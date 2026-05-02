# A PERTURBATION ANALYSIS OF INPUT TRANSFORMATIONS FOR ADVERSARIAL ATTACKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

The existence of adversarial examples, or intentional mis-predictions constructed from small changes to correctly predicted examples, is one of the most significant challenges in neural network research today. Ironically, many new defenses are based on a simple observation—the adversarial inputs themselves are not robust and small perturbations to the attacking input often recover the desired prediction. While the intuition is somewhat clear, a detailed understanding of this phenomenon is missing from the research literature. This paper presents a comprehensive experimental analysis of when and why perturbation defenses work and potential mechanisms that could explain their effectiveness (or ineffectiveness) in different settings.

# 1 INTRODUCTION

Adversarial examples are synthesized inputs to a machine learning model to induce an intentional mistake. The existence of such examples is widely known and extensively studied (Szegedy et al., 2014; Goodfellow et al., 2014; Biggio et al., 2013; Papernot et al., 2016; Carlini & Wagner, 2017). Interestingly enough, there is a growing body of evidence that suggest such adversarial examples are themselves not robust, namely, small perturbations sometimes will recover original, desired prediction (Dziugaite et al., 2016; Roth et al., 2019). Many new defense techniques explicitly leverage this property and prior techniques can be retrospectively interpreted as perturbations of the input images. However, a detailed understanding of this phenomenon is lacking from the research literature including: (1) what types of perturbations work, (2) whether all attacks exhibit this same property, and (3) possible counter-measures attackers can employ to defeat perturbation defenses.

We start with a simple experimental model where every example is passed through a lossy channel (whether stochastic or deterministic) prior to model inference. This channel induces a small perturbation to the input. This perturbation should be small enough as not to affect the prediction accuracy of normal examples, but large enough to dominate any adversarial attack. We can interpret a large number of recent defenses in this model including: feature squeezing (Xu et al., 2017), frequency or JPEG compression (Dziugaite et al., 2016), randomized smoothing (Cohen et al., 2019), and perturbation of network structure or the inputs randomly (Jafarnia-Jahromi et al., 2018; Zhang & Liang, 2019; Guo et al., 2017).

The main trade-off is choosing the strength (lossiness) of such channel to carefully mitigate prediction errors over true examples but maximize recovery of the adversarial examples. Our experiments suggest that this trade-off is surprisingly consistent across very different families of input perturbations, where the relationship between channel distortion (the  $\ell_2$  distance between channel input and output) and robustness is largely the same. Our experiments provide a detailed study of when such recovery is possible and the underlying mechanisms at work.

The objective of this analysis is not to demonstrate a new defense but to argue that many recent defense proposals are all based on a similar underlying mechanism of perturbation—and potentially suggest that they are all vulnerable to same types of attack strategies. In fact, we can devise a generic attacker that attacks a particularly strong lossy channel, an additive Laplace noise channel, and attacks designed on this channel are often successful against other defenses. This result implies that for many input perturbation defenses the attacker need not be fully adaptive, i.e., they do not need to know exactly what kind of transformation is used to defend the network. This analysis also highlights a curious perturbation-theoretic property of state-of-the-art neural networks, namely, recoverable

adversarial examples tend to exhibit higher instability (the more unstable the adversarial example, the easier it is to restore its correct label) from both a first-order and second-order analysis.

# 2 RELATED WORK

Much of the community's current understanding of adversarial sensitivity in neural networks is based on the seminal work by Szegedy et al. (2014). Multiple contemporaneous works also studied different aspects of this problem, postulating linearity and over-parametrization as possible explanations (Goodfellow et al., 2014; Biggio et al., 2013). Since the beginning of this line of work, the connection between compression and adversarial robustness has been recognized. The main defense strategies include: the idea of defensive network distillation<sup>1</sup> (Papernot et al., 2015), quantizing inputs using feature squeezing (Xu et al., 2017), the thermometer encoding as another form of quantization (Buckman et al., 2018), JPEG compression harnessed by (Dziugaite et al., 2016; Guo et al., 2017; Das et al., 2017; 2018; Aydemir et al., 2018; Liu et al., 2019). Other line of research leveraged connection between randomization and adversarial robustness: Pixel Deflection (Prakash et al., 2018), random resizing and padding of the input image (Xie et al., 2017), and total variance minimization (Guo et al., 2017). In our work we unify the methods based on compression and randomization that are applied to the input images.

While all of the aforementioned defenses were later broken (Carlini & Wagner, 2017; Athalye et al., 2018), it is important to understand why these approaches afforded any form of robustness. The community actually lacks consensus on this point: Szegedy et al. (2014) suggest that neural networks have blind spots, Xu et al. (2017) suggest that quantization makes the adversarial search space smaller, Buckman et al. (2018) suggest the linearity is the main culprit and quantized inputs break up linearity.

Zhang & Liang (2019) inject random Gaussian noise into an image and then discretize it. This method fits into the noisy channel framework with different definitions of  $C(x)$ . They show improved performance of this combined model in the non-adaptive setting. Our experiments find that simply injecting Gaussian noise (or even Uniform or Laplace noise) is an equally effective defense method. We also show that the discretization is not essential to good performance if the level of noise is appropriately tuned. The imprecise channel defense in a neural network is also related to the idea of gradient masking or gradient obfuscation, i.e., a hard to differentiate layer (Papernot et al., 2017). In this work, the backward pass computation is perturbed to make it difficult for a gradient-based attack to synthesize an adversarial image (while the forward pass is kept the same). We implement both non-adaptive attacks and adaptive that can observe the channel and take an approximate gradient through it. We further focus our study on the families of white-box attacks proposed by Carlini & Wagner (2017), and their adaptive variants.

Noise injection can be much more powerful than regularization or a dataset augmentation method. The dropout algorithm can be seen as applying noise to the hidden units. The dropout randomization (Feinman et al., 2017) was used to create a defense that was not completely broken and required a high distortion added to the adversarial examples (Carlini & Wagner, 2017). Many new defenses propose randomization through noise injection without considering the adversarial training (Zhang & Liang, 2019; Cohen et al., 2019). The work on injection of noise into inputs and each of the layers of neural networks by Liu et al. (2018) is a strong heuristic that led to defenses with theoretical guarantees. The random smoothing provides a certified robustness by utilizing inequalities from the differential privacy literature (Lecuyer et al., 2018). Cohen et al. (2019) improve the theoretical bounds of methods that randomly smooth the input examples. Recent work focuses on combination of randomized smoothing with adversarial training and achieve state of the art in terms of the provable robustness (Salman et al., 2019).

# 3 LOSSY CHANNEL MODEL

We consider convolutional neural networks that take  $w \times h$  (width times height) RGB digital images as input, giving an example space of  $\mathcal{X} \in (255)^{w \times h \times 3}$ , where  $(z)$  denotes the integer numbers from 0 to  $z$ . We consider a discrete label space of  $k$  classes represented as a confidence value  $\mathcal{Y} \in [0,1]^k$ .

Neural networks are parametrized functions (by a weight vector  $\theta$ ) between the example and label spaces  $f(x; \theta): \mathcal{X} \mapsto \mathcal{Y}$ .

An adversarial input  $x_{adv}$  is a perturbation of a correctly predicted example  $x$  that is incorrectly predicted by  $f$ .

$$
f (x) \neq f \left(x _ {a d v}\right)
$$

The distortion is the  $\ell_2$  error between the original example and the adversarial one:

$$
\delta_ {a d v} = \| x - x _ {a d v} \| _ {2} ^ {2}
$$

# 3.1 MODEL

Approximating  $f(\cdot)$  with a less precise version  $\bar{f}(\cdot)$  can counter-intuitively make it more robust (Dziugaite et al., 2016):

$$
f (x) = \bar {f} (x _ {a d v})
$$

Intuitively, a lossy version of  $f$  introduces noise into a prediction which dominates the strategic perturbations found by an adversarial attack procedure. It turns out that we can characterize a number of popular defense methodologies with this basic framework.

Let  $x$  be an example and  $f$  be a trained neural network. Precise evaluation means running  $f(x)$  and observing the predicted label. Imprecise evaluation involves first transforming  $x$  through a deterministic or stochastic noise process  $C(x) = C[x' \mid x]$ , and then evaluating the neural network

$$
y = f (x ^ {\prime}) \quad x ^ {\prime} \sim C (x)
$$

We can think of  $C(x)$  as a noisy channel (as in signal processing). The distortion of a  $C(x)$  is the expected  $\ell_2$  reconstruction error:

$$
\delta_ {c} = \mathbf {E} [ \| C (x) - x \| _ {2} ^ {2} ],
$$

which is a measure of how much information is lost passing the example through a channel.

This paper shows that there is a subtle trade-off between  $\delta_{c}$  and  $\delta_{adv}$ . In particular, we can find  $\delta_{c}$  such that  $\delta_{c} >> \delta_{adv}$  and  $f(x) = f(C(x_{adv}))$ . We show that compression and randomization based techniques exhibit this property.

# EXAMPLE OF DETERMINISTIC CHANNEL

When  $C(x)$  is deterministic it can be thought of as a lossy compression technique. Essentially, we run the following operation on each input example:

$$
x ^ {\prime} = \operatorname {c o m p r e s s} (x)
$$

One form of compression for CNNs is color-depth compression. Most common image classification neural network architectures convert the integer valued inputs into floating point numbers. We abstract this process with the norm function that for each pixel  $n \in (255)$  maps it to a real number  $v \in [0,1]$  by normalizing the value and the corresponding denorm function that retrieves the original integer value (where  $\lfloor \cdot \rfloor$  denotes the nearest integer function)  $^2$ :

$$
\operatorname {n o r m} (n) := \frac {n}{2 5 5} \qquad \operatorname {d e n o r m} (v) := \lfloor 2 5 5 * v \rfloor
$$

This process is normally reversible  $v = \mathrm{norm}(\mathrm{denorm}(v))$ , but we can artificially make this process lossy. Consider a parametrized  $C(\cdot)$  version of the color-depth compression function:

$$
C (v, b) := \frac {1}{2 ^ {b} - 1} \cdot \lfloor (2 ^ {b} - 1) * v \rceil
$$

By decreasing  $b$  by  $\Delta b$  we reduce the fidelity of representing  $v$  by a factor of  $2^{\Delta b}$  (for the  $b$  bits of precision).

# EXAMPLE OF STOCHASTIC PERTURBATION

The channel model is particularly interesting when  $C(x)$  is stochastic. Randomization has also been noted to play a big role in strong defenses in prior work (Madry et al., 2017; Zhang & Liang, 2019; Cohen et al., 2019). For example, we could add independent random noise to each input pixel:

$$
x ^ {\prime} = x + \epsilon
$$

We consider two schemes, Gaussian  $\epsilon \sim N(0,\sigma)$  and additive Uniform noise  $\epsilon \sim U(-B,B)$  which add independent noise to each pixel.

One of the advantages of randomization is that an adversary cannot anticipate how the particular channel  $C$  will transform an input before prediction. However, there is another subtle advantage to randomization. Randomized approaches can partially recover their loss in accuracy due to imprecision by averaging over multiple instances of the perturbation. In classification problems, we can take the most frequent label seen after  $T$  perturbation trials:

$$
\bar {f} (x) = \arg \max  _ {1.. k} \sum_ {i} ^ {T} f (x + \epsilon)
$$

# 3.2 PERTURBATION ANALYSIS

While the intuition is that the channel's perturbations dominate strategically placed distortions in an adversarial example, the underlying mathematical mechanism of why recovery is sometimes possible is less clear. We start with the hypothesis that synthesized adversarial examples are unstable predictions—meaning that small perturbations to the input space can change confidence values drastically. How do we quantify instability?

Let  $f(x)$  be a function that maps an image to a single class confidence value (i.e., a scalar output). We want to understand how  $f(x)$  changes if  $x$  is perturbed by  $\epsilon$ . We can apply a Taylor expansion of  $f$  around the given example  $x$ :

$$
f (x + \epsilon) \approx f (x) + \epsilon^ {T} \nabla_ {x} f (x) + \frac {1}{2} \epsilon^ {T} \nabla_ {x} ^ {2} f (x) \epsilon + \dots
$$

where  $\nabla_x f(x)$  denotes the gradient of the function  $f$  with respect to  $x$  and  $\nabla_x^2 f(x)$  denotes the Hessian of the function  $f$  with respect to  $x$ . The magnitude of the change in confidence is governed by the Taylor series terms in factorially decreasing importance.  $\| \epsilon \| _2$  is exactly the distortion measure  $\delta_{c}$  described at the beginning of Section 3.1. Thus, the expression is bounded in terms of the operator norm, or the maximal change in norm that could be induced, of each of the terms:

$$
\epsilon^ {T} \nabla_ {x} f (x) + \frac {1}{2} \epsilon^ {T} \nabla_ {x} ^ {2} f (x) \epsilon + \dots <   \delta_ {c} M _ {1} (x) + \frac {1}{2} \delta_ {c} ^ {2} M _ {2} (x) + \dots
$$

As  $\nabla_x f(x)$  is a vector, this is simply the familiar  $\ell_2$  norm, and for the second order term this is the maximal eigenvalue:

$$
M _ {1} (x) = \| \nabla_ {x} f (x) \| _ {2} M _ {2} (x) = \lambda_ {m a x} (\nabla_ {x} ^ {2} f (x))
$$

When  $M_1$  and  $M_2$  are larger this means there is a greater propensity to change the prediction for small perturbations. We will show that experimentally for certain types of attacks the  $M_1$  and  $M_2$  values around adversarial examples show signs of instability compared to those around natural examples—suggesting a mathematical mechanism of why recovery is possible.

# 4 EXPERIMENTS

Our experiments evaluate the efficacy of imprecision based defenses in a number of different adversarial problem settings.

# 4.1 EXPERIMENTAL SETUP

We run our experiments using ResNet-18 on CIFAR-10 and ResNet-50 on ImageNet dataset using P-100 GPUs (16GB memory). We explore a number of different attacks that are implemented in

the foolbox library (Rauber et al., 2017). In each experiment we measure the test accuracy  $(\%)$ , the confidence of predictions, and distances between the original images and either their adversarial counterparts or the recovered images after applying one of the defenses. We present our results for non-targeted attacks; if the adversary is successful it induces any misclassification. We experiment with many gradient-based attacks provided in the foolbox library that explore different optimization algorithms and distance measures, for instance:

- LBFGS minimizes the distance between the input image and the adversarial example as well as the cross-entropy between the predictions for the adversarial and the input image; introduced by Szegedy et al. (2014).  
- Carlini-Wagner  $L_{2}$  (C&W  $L_{2}$ ) is a generalization of the LBFGS attack that is devised after exhaustive search over possible space of: norms, loss functions, box optimization procedures, etc. (Carlini & Wagner, 2017).  
- BIM  $L_{1}$  is a modified version of the Basic Iterative Method that minimizes the  $L_{1}$  distance (Kurakin et al., 2016).  
- FGSM adds the sign of the gradient to the image, gradually increasing the magnitude until the image is misclassified Goodfellow et al. (2014).  
- PGD  $L_{\infty}$  the Projected Gradient Descent Attack that is an iterative version of the FGSM attack; we use the version that minimizes the  $L_{\infty}$  distance (Madry et al., 2017).

We extend the Carlini-Wagner  $L_{2}$  attack in its adaptive version so that the gradients are not obfuscated. We approximate the gradients for the backward pass on the compression layers as an identity function, similarly to (He et al., 2017; Athalye et al., 2018).

# 1. ALL PERTURBATION DEFENSES ARE SIMILAR

In the first experiment, we select 1000 random images from ImageNet and CIFAR-10 datasets. For each of these images we generate an adversarial attack with popular white-box gradient-based methods. Then, we run the adversarial images through different channels: Frequency Compression (FC), Color Depth reduction (CD), Uniform noise (Unif), Gaussian Noise (Gauss), SVD-based compression (SVD), Identity (Iden). The identity channel just passes through the adversarial image with no modification. We measure the accuracy of  $f(C(x))$ , which indicates the ability of the imprecise channels to recover the original label. We present the results in Figure 1 and also in the Supplement in Figure 5 (for all images in the test CIFAR-10 and dev ImageNet sets) and in Table 3 (for different channel parameters and five attacks).

When there is an identity channel, the adversarial attack is always successful. However, each of the imprecise channels is able to recover a substantial portion of original labels from the adversarial examples. It is important to note that we are evaluating these attacks in the setting, where any mis-classification is considered a success and the adversary is not aware of the defense. Importantly, all the channels are comparable in their performance. This suggests that any form of imprecision with the right error magnitude is effective at defending against these types of attacks. We observe that the attacks that incur higher distortions such as FGSM or LBFGS decrease the accuracy more than the iterative attacks such as C&W  $L_{2}$  or PGD  $L_{\infty}$ . This is because the iterative attacks find the adversarial images that are closer to the original images in terms of the corresponding distance measure that they optimize for in the input space. The key is to ensure that the error introduced by the imprecise channels is big enough to dominate the adversarial perturbations but small enough to generate valid predictions.

Figure 1 illustrates this relationship (see also a detailed analysis for a single image presented in Figure 6 in the Appendix). For five different imprecise channels, we plot the channel distortion against the accuracy for CIFAR-10 and ImageNet datasets. We analyze FC, CD, Unif, Gauss, and SVD channels. The curves are qualitatively similar in the low noise regime, but they show more differences when higher distortions are incurred. The lowest accuracy across the datasets for high distortions is observed for content-preserving FC and SVD compression channels. The Gaussian and Uniform channels have very similar trends. They outperform other channels on ImageNet for large distortions but are less performant on low-resolution CIFAR-10 images, where the CD channel achieves higher accuracy.

![](images/b2318780f0beb66e00a16ca3726c664060160b624033a442c4fd5459a7ef77f7.jpg)  
Figure 1: We plot the channel distortion against test accuracy (\%). The distortion of the imprecise channels has to be large enough to recover the correct label but not so large that it degrades model performance. The base test accuracy is about  $93.5\%$  for CIFAR-10 and  $83.5\%$  for ImageNet on 1000 randomly chosen images (results for the full test CIFAR-10 set and the full dev ImageNet set can be found in the Appendix in Figure 5). The experiment is run for the C&W  $L_{2}$  attack with 100 iterations and the PGD attack with 40 iterations.

![](images/8e8cde9fe07b520d5ccd83aa6260734eeb2e58e3e90faf5c2d245da162687e98.jpg)

![](images/38de7ba990596ddd331497d58cf7c26ef9327f0fcd902a2c3a978a34afc324b0.jpg)

![](images/c7597ab335fba68b995d2be4821ec91a97bf6f3e1207ea7727b52d60ec0fefbe.jpg)

![](images/4227bd208d81ec31544731921f5eb718f89da9596bda66730cbc923750b5b467.jpg)

# 2. ACCURACY OF PERTURBATION DEFENSES ON CLEAN DATA

One pitfall of the imprecise channel defense is that it introduces errors whether or not there are any adversarial examples. The errors act as an upper-bound for the best possible test accuracy we can get under adversarial perturbations. Balancing this trade-off is the key parameter in leveraging perturbation-based defenses. For frequency compression, color depth reduction, and uniform noise injection, we compare the test accuracy for different levels of imprecision. Table 1 shows the results for all test images from CIFAR-10 on the ResNet-18 architecture, for three of the imprecise channels, and for different noise settings. We present results for six noisy channels and full CIFAR-10 and ImageNet datasets in the Appendix in Figure 12.

Table 1: On CIFAR-10 with ResNet-18, we measure the max test accuracy for imprecise channels without any adversarial perturbation. This signifies the amount of accuracy we sacrifice with respect to the baseline test accuracy of the model (without any perturbations of the images) which is  $93.56\%$ .  

<table><tr><td>FC (%)</td><td>Acc. (%)</td><td>CD (bits)</td><td>Acc. (%)</td><td>Uniform (ε)</td><td>Acc. (%)</td></tr><tr><td>1</td><td>93.5</td><td>8</td><td>93.4</td><td>0.009</td><td>93.52</td></tr><tr><td>10</td><td>93.42</td><td>6</td><td>93.3</td><td>0.03</td><td>92.59</td></tr><tr><td>50</td><td>91.6</td><td>4</td><td>91.9</td><td>0.07</td><td>85.2</td></tr><tr><td>75</td><td>79.53</td><td>2</td><td>87.4</td><td>0.1</td><td>70.67</td></tr></table>

The test accuracy of the models can be increased by training with compression, e.g., by using FFT based convolutions with  $50\%$  compression in the frequency domain increases the accuracy to  $92.32\%$ .

# 3. ATTACKS ARE TRANSFERABLE

Many input transformation defenses are broken. If the attacker has full knowledge of the defense, it is possible to construct an attack that is impervious to the defense. This is called the adaptive setting. Since the underlying mechanisms of input transformations are similar, we find that an attacker does not need to be fully adaptive. The attacker can assume a particular strong defense and that same adversarial input often transfers to other defenses. We narrow the attacker to a single adaptive step (for details see Section B.6 in the Appendix). Even in this weak adaptive setting, the deterministic channels are fully broken but the randomized channels retain relatively high accuracy above  $23.8\%$ . We show in the Appendix in Figure 11 that the randomized defenses can also be broken when the adversary is given an unlimited number of adaptive steps.

Table 2 shows that FC attacked images do not transfer well to other defenses; the maximum drop in accuracy of the model protected by other defenses is  $14.8\%$ . In general, Laplace attacked images transfer the best to other defenses and decrease the accuracy of the defense models by at least  $46.86\%$  (for Laplace itself) and the accuracy of the model protected by the Uniform noisy channel drops by  $53.44\%$ . Most adversarial images (against a given defense) transfer very well to the FC defense,

Table 2: Transferability of the adversarial images created against a given noisy channel denoted as  $A$  (adaptive attack specified in the first column) to the defense protected with a noisy channel denoted as  $D$  (the defense with a noisy channel specified in the first row). Each result represents a recovery (%) of the adversarial examples (generated for  $A$ ) to correct labels after applying the defense  $(D)$ . We use  $30\%$  FC compression,  $50\%$  SVD compression, 4 bit values in CD, 0.03 noise level for Gauss and Laplace, and 0.04 noise level for the Uniform channel. We use 2000 images from the CIFAR-10 test set and 100 attack iterations with 5 binary steps to find the  $c$  value (with initial  $c$  value set to 0.01) for the adaptive C&W  $L_{2}$  attack. The baseline test accuracy is  $93.56\%$ .

<table><tr><td>A\D</td><td>FC</td><td>CD</td><td>SVD</td><td>Gauss</td><td>Uniform</td><td>Laplace</td></tr><tr><td>FC</td><td>0.20</td><td>80.75</td><td>83.05</td><td>81.15</td><td>79.65</td><td>78.70</td></tr><tr><td>CD</td><td>3.85</td><td>0.70</td><td>43.60</td><td>47.30</td><td>60.45</td><td>62.35</td></tr><tr><td>SVD</td><td>1.99</td><td>47.96</td><td>0.77</td><td>46.52</td><td>62.87</td><td>65.75</td></tr><tr><td>Gauss</td><td>4.45</td><td>48.70</td><td>44.80</td><td>51.50</td><td>61.75</td><td>60.15</td></tr><tr><td>Uniform</td><td>3.45</td><td>30.30</td><td>30.60</td><td>30.15</td><td>48.05</td><td>51.55</td></tr><tr><td>Laplace</td><td>3.05</td><td>23.35</td><td>24.60</td><td>23.80</td><td>39.15</td><td>46.70</td></tr></table>

i.e. an adversarial image against any defense (e.g. CD, SVD, Gauss, Uniform, or Laplace) is also adversarial against the FC defense. The adversarial images generated against the Uniform defense show better transfer to other defenses in comparison to the adversarial images generated against the Gaussian defense. This is because the higher noise level is applied in the Uniform defense. We observe analogous trends for the ImageNet dataset and present the results in the supplement (Tables 5 and 6).

![](images/ddc859f7f449db3e857adb31d9d5a7c00a3204a7b3d01602bc8187ec9e457ac7.jpg)

![](images/c337137402bcd06b2d81a3a3a520a71955e5c3a71b151935fd81b732f284d2c3.jpg)  
Figure 2: The  $L_{2}$  norm of the gradients for the original and adversarial classes with respect to the original and adversarial input images. The norm of the gradient w.r.t the original class on the original image is small-indicating a stable prediction. Adversarial images have a higher norm for both the adversarial class and the original class indicating higher instability.

# 4. ADVERSARIAL EXAMPLES ARE UNSTABLE

The key question is why adversarial examples are more sensitive to perturbations than natural inputs, when there is evidence that from an input perspective they are statistically indistinguishable. Our experiments suggest that this sensitivity arises from the optimization process that generates adversarial inputs. Based on the operator-norm analysis in the text, we plot those values for both adversarial and natural images. Figure 2 shows the  $\ell_2$  norm of the input gradient w.r.t the correct class and adversarial class for the original image and the adversarial image. For the original image, the gradient w.r.t the original class is very small (i.e., a stable prediction).

This analysis extends to higher orders as well. Our results in Figure 3 show that the adversarial inputs lead to noticeably higher Hessian spectrum than the original inputs. This suggests that the model predictions for the adversarial inputs are less stable than for the original images. Thus, perturbations of the adversarial images with some form of noise can easily change the classification outcome while the prediction for the original images are much more robust and do not lead to such unstable predictions.

![](images/2da99505d0e6fd75c232797cc6be743788d0e82a983eace692e35713d0cc9628.jpg)  
Figure 3: The top eigenvalues of the Hessians with respect to (w.r.t.) the input 1024 images from the CIFAR-10 dataset trained on the ResNet-18 architecture. We plot the histogram that shows counts of magnitudes for the eigenvalues. Yao et al. (2018) show analysis of the Hessian w.r.t. parameters and we extend it to analyze Hessian w.r.t. inputs.

![](images/466994bca97c95a82ca401ee88d6315adccffd26c5dfd76750edfd21b8c2dd99.jpg)  
Figure 4: Non-adaptive attack and the recoverable ranges in terms of the c parameter in the C&W  $L_{2}$  attack. We systematically change the c parameter and keep the parameters for the channels unchanged. We use the VGG-16 network and 1024 images from the CIFAR-10 dataset. The test accuracy of the model without any noise layers and on the clean data is  $85.23\%$ .

# 5. RECOVERABLE RANGES

In Figure 4 we present the accuracy of different channels as the  $c$  parameter in the C&W algorithm is systematically increased. For example, Carlini & Wagner L2 attack with  $c = 0.1$  causes the accuracy after the Gauss channel to drop from about  $85\%$  to  $67\%$ , the empty channel accuracy drops to  $10\%$ , and the accuracy of the simple RGB brightness reduction method drops to  $53\%$ . The Laplace channel gives the highest accuracy for high values of the  $c$  parameter (above 1.0). The CD, FC, SVD, Gauss, and Uniform channels show similar trends.

We also add a related approach which is the RSE (Random Self-Ensemble) network with 0.2 noise level in the first layer and 0.1 noise level in the remaining layers (as recommended in Liu et al. (2018)). This defense does better for lower distortion levels (c value below 0.1) than other noisy channels, but then its accuracy deterioration is faster for higher distortion levels. We also include a very simple channel that reduces brightness of an image by subtracting an arbitrary value from each pixel. The comparison between a very complex approach and simple input transformation is informative—as they largely follow the same trends.

We present a detailed analysis of recoverable ranges for an ImageNet example in Appendix in Figure 6. The recovery range shrinks as we increase the strength of the attack that also incurs higher distortion of the adversarial image.

# 5 CONCLUSION, LIMITATIONS, AND FUTURE WORK

There is a growing body of evidence to suggest that the attacks themselves are not robust since small changes to the adversarial input often recover the original label. In hindsight, this is an obvious corollary to the very existence of adversarial examples—by definition they are relatively close to correctly predicted examples in the input space. Random perturbations of the input can dominate the strategically placed perturbations synthesized by an attack. In fact, results are consistent across both deterministic and stochastic channels that degrade the fidelity of the input example. This paper put forth a detailed experimental study illustrating the conditions under which the true label can be recovered.

The current trend in the community leads towards certified defenses that give robust theoretical guarantees and eschew the adversarial arms race (Salman et al., 2019). Our paper caters to the need of unifying the input transformation and perturbation methods, characterizing their common aspects, and preventing future proposals that fall into the same family of weak techniques. Looking on the bright side, eliminating the known unsuccessful defenses leaves an uncharted territory for new methods that could lead from an experimental success to its theoretical resultant.

# REFERENCES

Anish Athalye, Nicholas Carlini, and David A. Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In ICML, 2018.  
Ayse Elvan Aydemir, Alptekin Temizel, and Tugba Taskaya-Temizel. The effects of JPEG and JPEG2000 compression on attacks using adversarial examples. CoRR, abs/1803.10418, 2018. URL http://arxiv.org/abs/1803.10418.  
Battista Biggio, Igino Corona, Davide Maiorca, Blaine Nelson, Nedim Šrndić, Pavel Laskov, Giorgio Giacinto, and Fabio Roli. Evasion attacks against machine learning at test time. In Joint European conference on machine learning and knowledge discovery in databases, pp. 387-402. Springer, 2013.  
Jacob Buckman, Aurko Roy, Colin Raffel, and Ian Goodfellow. Thermometer encoding: One hot way to resist adversarial examples. In International Conference on Learning Representations, 2018.  
N. Carlini and D. Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy (SP), pp. 39-57, May 2017.  
Nicholas Carlini and David Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods. In Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security, pp. 3-14. ACM, 2017.  
Jeremy M Cohen, Elan Rosenfeld, and J Zico Kolter. Certified adversarial robustness via randomized smoothing. arXiv preprint arXiv:1902.02918, 2019.  
Nilaksh Das, Madhuri Shanbhogue, Shang-Tse Chen, Fred Hohman, Li Chen, Michael E. Kounavis, and Duen Horng Chau. Keeping the bad guys out: Protecting and vaccinating deep learning with JPEG compression. CoRR, abs/1705.02900, 2017. URL http://arxiv.org/abs/1705.02900.  
Nilaksh Das, Madhuri Shanbhogue, Shang-Tse Chen, Fred Hohman, Siwei Li, Li Chen, Michael E. Kounavis, and Duen Horng Chau. Shield: Fast, practical defense and vaccination for deep learning using JPEG compression. CoRR, abs/1802.06816, 2018. URL http://arxiv.org/abs/1802.06816.  
Adam Dziedzic, John Paparrizos, Sanjay Krishnan, Aaron Elmore, and Michael Franklin. Band-limited training and inference for convolutional neural networks. ICML, 2019.  
Gintare Karolina Dziugaite, Zoubin Ghahramani, and Daniel M Roy. A study of the effect of jpg compression on adversarial images. arXiv preprint arXiv:1608.00853, 2016.  
Reuben Feinman, Ryan R Curtin, Saurabh Shintre, and Andrew B Gardner. Detecting adversarial samples from artifacts. arXiv preprint arXiv:1703.00410, 2017.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Chuan Guo, Mayank Rana, Moustapha Cisse, and Laurens van der Maaten. Countering Adversarial Images using Input Transformations. arXiv e-prints, art. arXiv:1711.00117, Oct 2017.  
Warren He, James Wei, Xinyun Chen, Nicholas Carlini, and Dawn Song. Adversarial example defense: Ensembles of weak defenses are not strong. In 11th USENIX Workshop on Offensive Technologies (WOOT 17), 2017.  
Mehdi Jafarnia-Jahromi, Tasmin Chowdhury, Hsin-Tai Wu, and Sayandev Mukherjee. Ppd: Permutation phase defense against adversarial examples in deep learning. arXiv preprint arXiv:1812.10049, 2018.  
Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio. Adversarial examples in the physical world. CoRR, abs/1607.02533, 2016.

Mathias Lecuyer, Vaggelis Atlidakis, Roxana Geambasu, Daniel Hsu, and Suman Jana. Certified Robustness to Adversarial Examples with Differential Privacy. arXiv e-prints, art. arXiv:1802.03471, Feb 2018.  
Xuanqing Liu, Minhao Cheng, Huan Zhang, and Cho-Jui Hsieh. Towards robust neural networks via random self-ensemble. In Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss (eds.), Computer Vision - ECCV 2018, pp. 381-397, Cham, 2018. Springer International Publishing. ISBN 978-3-030-01234-2.  
Zihao Liu, Qi Liu, Tao Liu, Nuo Xu, Xue Lin, Yanzhi Wang, and Wujie Wen. Feature distillation: Dnn-oriented jpeg compression against adversarial examples. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.  
Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, and Ananthram Swami. Distillation as a defense to adversarial perturbations against deep neural networks. arXiv preprint arXiv:1511.04508, 2015.  
Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z Berkay Celik, and Ananthram Swami. The limitations of deep learning in adversarial settings. In 2016 IEEE European Symposium on Security and Privacy (EuroS&P), pp. 372-387. IEEE, 2016.  
Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z Berkay Celik, and Ananthram Swami. Practical black-box attacks against machine learning. In Proceedings of the 2017 ACM on Asia conference on computer and communications security, pp. 506-519. ACM, 2017.  
Aaditya Prakash, Nick Moran, Solomon Garber, Antonella DiLillo, and James Storer. Deflecting adversarial attacks with pixel deflection. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.  
Jonas Rauber, Wieland Brendel, and Matthias Bethge. Foolbox: A python toolbox to benchmark the robustness of machine learning models. arXiv preprint arXiv:1707.04131, 2017.  
Kevin Roth, Yannic Kilcher, and Thomas Hofmann. The odds are odd: A statistical test for detecting adversarial examples. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 5498-5507, Long Beach, California, USA, 09-15 Jun 2019. PMLR. URL http://proceedings.mlr.press/v97/roth19a.html.  
Hadi Salman, Greg Yang, Jerry Li, Pengchuan Zhang, Huan Zhang, Ilya P. Razenshteyn, and Sébastien Bubeck. Provably robust deep learning via adversarially trained smoothed classifiers. CoRR, abs/1906.04584, 2019. URL http://arxiv.org/abs/1906.04584.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. CoRR, abs/1312.6199, 2014.  
Cihang Xie, Jianyu Wang, Zhishuai Zhang, Zhou Ren, and Alan Yuille. Mitigating adversarial effects through randomization. arXiv preprint arXiv:1711.01991, 2017.  
Weilin Xu, David Evans, and Yanjun Qi. Feature squeezing: Detecting adversarial examples in deep neural networks. arXiv preprint arXiv:1704.01155, 2017.  
Zhewei Yao, Amir Gholami, Qi Lei, Kurt Keutzer, and Michael W Mahoney. Hessian-based analysis of large batch training and robustness to adversaries. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems 31, pp. 4949-4959. Curran Associates, Inc., 2018.  
Yuchen Zhang and Percy Liang. Defending against whitebox adversarial attacks via randomized discretization. AISTATS, 2019.
