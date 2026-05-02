# DISTRIBUTED MOMENTUM FOR BYZANTINE-RESILIENT STOCHASTIC GRADIENT DESCENT

Anonymous authors

Paper under double-blind review

# ABSTRACT

Byzantine-resilient Stochastic Gradient Descent (SGD) aims at shielding model training from Byzantine faults, be they ill-labeled training datapoints, software/hardware bugs, or malicious worker nodes in a distributed setting. Two recent attacks have been challenging state-of-the-art defenses though, often successfully precluding the model from even fitting the training set. The main identified weakness of current defenses is their requirement of a sufficiently low variance-norm ratio for the stochastic gradients. We propose a practical method which, despite increasing the variance, reduces the variance-norm ratio, mitigating the identified weakness. We assess the effectiveness of our method over 736 different training configurations, comprising the 2 state-of-the-art attacks and 6 defenses. For confidence and reproducibility purposes, each configuration is run 5 times, with seeds 1 to 5, totalling 3680 runs. In our experiments, when the attack is effective enough to decrease the highest observed top-1 cross-accuracy by at least  $20\%$  compared to the unattacked run, our technique systematically increases back the highest observed accuracy, and is able to recover at least  $20\%$  in more than  $60\%$  of the cases.

# 1 INTRODUCTION

Stochastic Gradient Descent (SGD) is one of the main driving forces behind the successes of machine learning. Scaling SGD can mean aggregating more but inevitably less well-sanitized data, and distributing the training over several machines, making SGD even more vulnerable to Byzantine faults: corrupted/malicious training datapoints, software bugs, etc. Many Byzantine-resilient techniques have been proposed to keep SGD safer from these faults, e.g. Alistarh et al. (2018); Damaskinos et al. (2018); Yang & Bajwa (2019b); TianXiang et al. (2019); Bernstein et al. (2019); Yang & Bajwa (2019a); Yang et al. (2019); Rajput et al. (2019); Muñoz-González et al. (2019). These techniques mainly use the same adversarial model (Figure 2): a central, trusted parameter server distributing gradient computations to several workers, a minority of which is controlled by an adversary and can submit arbitrary gradients.

Two families of defense techniques can be distinguished. The first employs redundancy schemes, inspired by coding theory. This approach has strong resilience guarantees, but its requirement to share data between workers makes this approach unsuitable for several classes of applications, e.g. when data cannot be shared for privacy, scalability or legal reasons. The second family uses statistically-robust aggregation schemes, and is the focus of this paper. The underlying idea is simple. At each training step, the server aggregates the stochastic gradients computed by the workers into one gradient, using a function called a Byzantine-resilient Gradient Aggregation Rule (GAR). These statistically-robust GARs are designed to produce at each step a gradient that is expected to decrease the loss.

Intuitively, one can think of this second family as different formulations of the multivariate median. In particular, if the non-Byzantine gradients were all equal at each step, any different (adversarial) gradient would be rejected by each of these medians, and no attack would succeed. But due to their stochastic nature, the non-Byzantine gradients are different: their variance is strictly positive. Formal guarantees on any given statistically-robust GAR typically require that the variance-norm ratio, the ratio between the variance of the non-Byzantine gradients and the norm of the expected non-Byzantine gradient, remains below a

![](images/fad62d9c1343c8fe40ee94ae9bde7c2a1621274f62d68b54f1513b583c04382d.jpg)  
Figure 1: We report on the highest measured top-1 cross-accuracy while training under either of the two studied, state-of-the-art attacks. [a, b]: a convolutional model (Section 4.1) for CIFAR-10 under the attack from Baruch et al. (2019), and [c, d]: a fully connected model for Fashion-MNIST (Xiao et al., 2017) under the attack from Xie et al. (2019a). Roughly half the workers implements the attack in [a, c], and a quarter does in [b, d]; see Section 4.1. Each experiment is run 5 times. The dotted blue line is the median of the maximum top-1 cross-accuracy of the 5 runs without attack, and the boxes aggregate the maximum top-1 cross-accuracies obtained under attack with each 5 runs of the 6 studied defenses. Over 736 different combinations of attacks, defenses, datasets, etc (totalling of 3680 runs), our method consistently obtain at least similar, if not substantially better performances (lower minimal loss, higher maximal top-1 cross-accuracy) than the standard formulation. Notably, our formulation obtains these results with no additional computational complexity.

![](images/d8f2e3950b26a542909af1bd213ce9619e30e1a9fbc059a1e36c38804ee0671a.jpg)

![](images/80baf65a24086b5c3ebf3e5f41dffdc1f9c4cdd63c8e3a24b8c7d3b368a6ac5b.jpg)

![](images/73507314ae1da4f82a295edade05395ccc76be3bd42946af559f209cac9e31d9.jpg)

certain constant (constant which depends on the GAR itself and fixed hyperparameters). Intuitively, this notion of variance-norm ratio can be comprehended quite analogously to the inverse of the signal-to-noise ratio (i.e. the "noise-to-signal" ratio) in signal processing.

However, Baruch et al. (2019) noted that an attack could send gradients that are close to non-Byzantine outlier gradients, building an apparent majority of gradients that could be sufficiently far from the expected non-Byzantine gradient to increase the loss. This can happen against most statistically-robust GARs in practice, as the variance-norm ratio is often too large for them. Two recent attacks (Baruch et al., 2019; Xie et al., 2019a) were able to exploit this fact to substantially hamper the training process (which our experiments confirm).

The work presented here aims at (substantially) improving the resilience of statistically robust GARs "also in practice", by reducing the variance-norm ratio of the gradients received by the server. We do that by taking advantage of an old technique normally used for acceleration: momentum. This technique is regularly applied at the server, but instead we propose to confer it upon each distributed worker, effectively making the Byzantine-resilient GAR aggregate accumulated gradients. Crucially, there is no computational complexity attached to our reformulation: it only reorders operations in existing (distributed) algorithms.

Contributions. Our main contributions can be summarized as follows:

- A reformulation of classical/Nesterov momentum which can significantly improve the effectiveness (Figure 1) of any statistically-robust Gradient Aggregation Rule (GAR). We formally analyze the impact of our reformulation on the variance-norm ratio of the aggregated gradients, ratio on which the studied GARs assume an upper bound.  
- An extensive and reproducible<sup>1</sup> set of experiments substantiating the effectiveness of our reformulation of momentum in improving existing defenses against state-of-the-art attacks.

Paper Organization. Section 2 provides the necessary background. Section 3 presents our distributed momentum scheme and provides some intuitions on its effects. Formal developments of these intuitions are given in the appendix. Section 4 describes our experimental settings in details, before presenting and analysing some of our experimental results. The appendix reports on the entirety of our experiments, and details how they can be reproduced (in one command, graphs included). Section 5 discusses related and future work.

# 2 BACKGROUND

# 2.1 BYZANTINE DISTRIBUTED SGD

Stochastic Gradient Descent (SGD). We consider the classical problem of optimizing a non-convex, differentiable loss function  $Q: \mathbb{R}^d \to \mathbb{R}$ , where  $Q(\theta_t) \triangleq \mathbb{E}_{x \sim \mathcal{D}}[q(\theta_t, x)]$  for a fixed data distribution  $\mathcal{D}$ . Ideally, we seek  $\theta^*$  such that  $\theta^* = \arg \min_{\theta} (Q(\theta))$ .

We employ mini-batch SGD optimization. Starting from initial parameter  $\theta_0\in \mathbb{R}^d$ , at every step  $t\geq 0$ ,  $b$  samples  $\left(x_{t}^{(1)}\ldots x_{t}^{(b)}\right)$  are sampled from  $\mathcal{D}$  to estimate one stochastic gradient  $g_{t}\triangleq \frac{1}{b}\sum_{k = 1}^{b}\nabla q\left(\theta_{t},x_{t}^{(k)}\right)\approx \nabla Q\left(\theta_{t}\right)$ . This stochastic gradient is then used to update the parameters  $\theta_t$ , with:  $\theta_{t + 1} = \theta_t - \alpha_t g_t$ . The sequence  $\alpha_{t} > 0$  is called the learning rate.

Classical and Nesterov momentum One field-tested amendment to mini-batch SGD is classical momentum (Polyak, 1964), where each gradient keeps an exponentially-decreasing effect on every subsequent update. Formally:  $\theta_{t + 1} = \theta_t - \alpha_t\sum_{u = 0}^t\mu^{t - u}g_u$ , with  $0 < \mu < 1$ . Nesterov (1983) proposed another revision. Noting  $v_{t}$  the velocity vector,  $v_{0} = 0$ , formally:

$$
v _ {t + 1} = \mu v _ {t} + \frac {1}{b} \sum_ {k = 1} ^ {b} \nabla q \left(\theta_ {t} - \alpha_ {t} v _ {t}, x _ {t} ^ {(k)}\right)
$$

$$
\theta_ {t + 1} = \theta_ {t} - \alpha_ {t} v _ {t + 1}
$$

Compared to classical momentum, the gradient is estimated at  $\theta_{t} - \alpha_{t}v_{t}$  instead of  $\theta_{t}$ .

# Distributed SGD with Byzantine workers.

We follow the parameter server model (Li et al., 2014): one single process (the parameter server) holding the parameter vector  $\theta_t \in \mathbb{R}^d$ , and  $n$  other (the workers) estimating gradients. Among these  $n$  workers, up to  $f < n$  are said Byzantine, i.e. adversarial. Unlike the other  $n - f$  honest workers, these  $f$  Byzantine workers can submit arbitrary gradients (Figure 2).

At each step  $t$ , the parameter server receives  $n$  different gradients  $g_{t}^{(1)} \ldots g_{t}^{(n)}$ , among which  $f$  are arbitrary (submitted by the Byzantine workers). So the update equation becomes:  $\theta_{t+1} = \theta_t - \alpha_t G_t$ , where:

![](images/506a26fc0403b32e5ac73e93af515396b11bcef12f160d76579b03ba8c5bcf78.jpg)  
Figure 2: A parameter server setup with  $n = 8$  workers, among which  $f = 3$  are Byzantine (i.e., adversarial) workers. A black line represents a bidirectional communication channel.

$$
G _ {t} \triangleq \sum_ {u = 0} ^ {t} \mu^ {t - u} F \left(g _ {u} ^ {(1)}, \dots , g _ {u} ^ {(n)}\right) \tag {1}
$$

Function  $F$  is called a Gradient Aggregation Rule (GAR). In non-Byzantine settings, averaging is used; formally:  $F\big(g_t^{(1)},\ldots ,g_t^{(n)}\big) = \frac{1}{n}\sum_{i = 1}^{n}g_t^{(i)}$ . In the presence of Byzantine workers, a more robust aggregation is performed with a Byzantine-resilient GAR. Sections 2.2 and 2.3 respectively describe the 6 existing GARs and 2 attacks studied in this paper.

Adversarial Model. The goal of the adversary is to impede the learning process, which is defined as the maximization of the loss  $Q$  or, more judiciously for the image classification tasks tackled in this paper, as the minimization $^2$  of the model's top-1 cross-accuracy. The adversary cannot directly overwrite  $\theta_t$  at the parameter server. The adversary only submits  $f$  arbitrary gradients to the server per step, via the  $f$  Byzantine workers it controls $^3$ . We assume an omniscient adversary. In particular, the adversary knows the GAR used by the parameter server and, at each step, the adversary can generate Byzantine gradients dependent on the honest gradients submitted at the same step and any previous step.

# 2.2 BYZANTINE-RESILIENT GARS

We briefly present below the 6 studied GARs. A more formal presentation, along with the theoretical guarantees of some of these GARs, is provided in the appendix (Section A). Let  $n$  be the number of gradients the parameter server received from the  $n$  workers (Figure 2), and let  $f$  be the maximum number of Byzantine gradients the GAR must be able to tolerate.

Median (Yin et al., 2018). The coordinate-wise median of the  $n$  received gradients.

Krum (Blanchard et al., 2017). Each received gradient is assigned a score. The score of gradient  $x$  is the sum of the squared  $\ell_2$ -distances between  $x$  and the  $n - f - 2$  closest gradients to  $x$ . The aggregated gradient is then the arithmetic mean of the  $n - f - 2$  gradients with the smallest scores. This variant is called Multi-Krum in the original paper.

Trimmed Mean (Yin et al., 2018). The coordinate-wise trimmed-mean of the  $n$  received gradients. The trimmed-mean of a vector of  $n$  values is the arithmetic mean, after the  $f$  smallest and the  $f$  largest values have been discarded, of the remaining values.

Phocas (Xie et al., 2018b). The coordinate-wise arithmetic mean of the  $n - f$  closest values to the coordinate-wise trimmed-mean.

MeaMed (Xie et al., 2018a). Same as Phocas, but with median replacing trimmed-mean.

Bulyan (El-Mhamdi et al., 2018). This is a composite aggregation rule that iterates on another GAR in a first selection phase. Bulyan uses Krum, so this first phase selects  $n - 2f - 2$  gradients, at each iteration removing the highest scoring gradient. The aggregated gradient is the coordinate-wise arithmetic mean of the  $n - 4f - 2$  closest values to the (coordinate-wise) median of the selected gradients.

# 2.3 STUDIED ATTACKS

The two state-of-the-art attacks, that recently appeared in the literature, follow the same core principle. Let  $\varepsilon \in \mathbb{R}_{\geq 0}$  be a non-negative factor, and  $a_{t} \in \mathbb{R}^{d}$  an attack vector which value depends on the actual attack used (see below for possible values of  $a_{t}$ ). At each step  $t$ , each of the  $f$  Byzantine workers submits the same Byzantine gradient:  $\overline{g_t} + \varepsilon a_t$ , where  $\overline{g_t}$  is an approximation of the real gradient  $\nabla Q(\theta_t)$  at step  $t$ . The value of  $\varepsilon$  is fixed (see below).

A Little is Enough (Baruch et al., 2019). In this attack, each Byzantine worker submits  $\overline{\mathcal{G}_t} + \varepsilon a_t$ , with  $a_t \triangleq -\sigma_t$  the opposite of the coordinate-wise standard deviation of the honest gradient distribution  $\mathcal{G}_t$ . Our experiments use  $\varepsilon = 1.5$ , as proposed by the original paper.

Fall of Empires (Xie et al., 2019a). Each Byzantine worker submits  $(1 - \varepsilon)\overline{g_t}$ , i.e.,  $a_{t} \triangleq -\overline{g_{t}}$ . The original paper tested  $\epsilon \in \{-10, -1, 0, 0.1, 0.2, 0.5, 1, 10, 100\}$ , our experiments use  $\varepsilon = 1.1$ , corresponding in the notation of the original paper to  $\epsilon \triangleq -(1 - \varepsilon) = -(1 - 1.1) = 0.1$ .

# 3 MOMENTUM AT THE WORKERS

Intuitively, the Byzantine-resilient GARs (Section 2.2) rely on the honest gradients being sufficiently clumped (formalized in e.g. Equation 3 and Equation 4). In the edge case where every honest gradient is equal (i.e. no stochastic noise), no attack can affect the learning: there is by assumption a strict majority of identical honest gradients. On the contrary when the honest gradients are "spread", i.e. their variance is large enough compared to their norms, the attack vectors can form a majority by relying on a few outlier (but honest) gradients (Baruch et al., 2019), and so substantially influence the aggregated gradient.

Momentum makes the parameters  $\theta_{t}$  travel down the loss function with inertia, accumulating both the real gradient  $\nabla Q(t)$  and the error (i.e. here, the stochastic noise)  $g_{t} - \nabla Q(t)$ . Intuitively, the accumulation of errors grows at a moderate rate, as past errors can be partially compensated by future ones. But when consecutive  $\nabla Q(t)$  have sufficiently low solid angles, past real gradients do not compensate future real gradients: the norm of  $G_{t}$  can grow "faster" (for each new step) than its variance, mitigating the potential impact of an attack.

# 3.1 FORMULATION

From the formulation of momentum SGD in a distributed setting (Equation 1):

$$
G _ {t} \triangleq \sum_ {u = 0} ^ {t} \mu^ {t - u} F \left(g _ {u} ^ {(1)}, \dots , g _ {u} ^ {(n)}\right)
$$

we instead confer the momentum computation on the workers:

$$
G _ {t} \triangleq F \left(\underbrace {\sum_ {u = 0} ^ {t} \mu^ {t - u} g _ {u} ^ {(1)}} _ {G _ {t} ^ {(1)}}, \dots , \underbrace {\sum_ {u = 0} ^ {t} \mu^ {t - u} g _ {u} ^ {(n)}} _ {G _ {t} ^ {(n)}}\right) \tag {2}
$$

Notations. In the remaining of this paper, we call the original formulation (momentum) at the server, and the proposed, revised formulation (momentum) at the worker(s). The quantities  $G_{t}^{(1)} \ldots G_{t}^{(n)}$  will be called the submitted gradients (at step  $t$ ). At step  $t$ , the variance-norm ratio is computed on the honest subset of:  $g_{t}^{(1)} \ldots g_{t}^{(n)}$ , if momentum at the server is employed, otherwise  $G_{t}^{(1)} \ldots G_{t}^{(n)}$ , if momentum at the workers is used instead.

Formal analysis. The formal analysis of the impact of our technique on the variance-norm ratio of the aggregated gradients is available in the appendix, Section B.2.

# 4 EXPERIMENTS

Our experiments cover 2 models, 4 datasets, the 6 studied defenses under each of the 2 state-of-the-art attacks $^{5}$ , different fractions of Byzantine workers (either half or a quarter), using Nestorov instead of classical momentum, plus unattacked settings where each worker is honest and the GAR is mere averaging. Since our theoretical results (Section B.1) suggest that smaller learning rates may reduce the variance-norm ratio, two learning rate schedules (an optimal and a smaller one) are also tested. For reproducibility and confidence in the empirical benefits of our reformulation, we test every combination of the hyperparameters mentioned above, and each combination is repeated 5 times with fixed seeds 1 to 5 (totally 3680 runs).

The tools we developed to implement our reformulation captures  $\sim 20$  metrics, including the evolution of the average loss, top-1 cross-accuracy and variance-norm ratio of the submitted gradients. In this section and Section D, we specifically report on these 3 metrics.

# 4.1 EXPERIMENTAL SETUP

We use a compact notation to define the models: L(#outputs) for a fully-connected linear layer, R for ReLU activation, S for log-softmax, C(#channels) for a fully-connected 2D convolutional layer (kernel size 3, padding 1, stride 1), M for 2D-maxpool (kernel size 2), B for batch-normalization, and D for dropout (with fixed probability 0.25).

We use the models from respectively Baruch et al. (2019) and Xie et al. (2019a):

<table><tr><td></td><td>Fully connected</td><td>Convolutional</td></tr><tr><td>Model</td><td>(784)-L(100)-R-L(10)-R-S</td><td>(3, 32×32)-C(64)-R-B-C(64)-R-B-M-D-C(128)-R-B-C(128)-R-B-M-D-L(128)-R-D-L(10)-S</td></tr><tr><td>Datasets</td><td>MNIST, Fashion MNIST (83 samples/gradient)</td><td>CIFAR-10, CIFAR-100 (50 samples/gradient)</td></tr><tr><td>#workers</td><td>n = 51 f ∈ {24, 12}</td><td>n = 25 f ∈ {11, 5}</td></tr></table>

For model training, we use the negative log likelihood loss and respectively  $10^{-4}$  and  $10^{-2}$ $\ell_2$ -regularization for the fully connected and convolutional models. We also clip gradients, en

suring their norms remain respectively below 2 and 5 for the fully connected and convolutional models. Regarding evaluation, we use the top-1 cross-accuracy over the whole test set.

Datasets are pre-processed before training. MNIST receives the same pre-processing as in Baruch et al. (2019): an input image normalization with mean 0.1307 and standard deviation 0.3081. Fashion MNIST, CIFAR-10 and CIFAR-100 are all expanded with horizontally flipped images. For both CIFAR-10 and CIFAR-100, a per-channel normalization with means 0.4914, 0.4822, 0.4465 and standard deviations 0.2023, 0.1994, 0.2010 (Liu, 2019) has been applied.

We denote by  $f$  the number of Byzantine workers either to the maximum for which Krum can be used (roughly an half:  $f = \left\lfloor \frac{n - 3}{2} \right\rfloor$ ), or the maximum for Bulyan (roughly a quarter,  $f = \left\lfloor \frac{n - 3}{4} \right\rfloor$ ). The attack factors  $\varepsilon_{t}$  (Section 2.3) are set to constants proposed in the literature, namely  $\varepsilon_{t} = 1.5$  for Baruch et al. (2019) and  $\varepsilon_{t} = 1.1$  for Xie et al. (2019a). We also experiment two different learning rates. The first and largest is selected so as to maximize the performance (highest final cross-accuracy and accuracy gain per step) of the model trained without Byzantine workers. The second and smallest is chosen so as to minimize the performance loss under attack, without substantially impacting the final accuracy when trained without Byzantine workers. The fully connected and convolutional models are trained respectively with  $\mu = 0.9$  and  $\mu = 0.99$ . These values were obtained by trial and error, to maximize the accuracy gain per step when there is no attack.

Reproducibility. Particular care has been taken to make our results reproducible. Each of the 5 runs per experiment are respectively seeded with seed 1 to 5. For instance, this implies that two experiments with same seed and same model also starts with the same parameters  $\theta_0$ . To further reduce the sources of non-determinism, the CuDNN backend is configured in deterministic mode (our experiments ran on two GeForce GTX 1080 Ti) with benchmark mode turned off. We also used log-softmax + nll loss, which is equal to softmax + cross-entropy loss, but with improved numerical stability on PyTorch. We provide our code along with a script reproducing all of our results, both the experiments and the graphs, in one command. Details, including software and hardware dependencies, are available in the appendix.

# 4.2 EXPERIMENTAL RESULTS

This section reports on the evolution of the average loss, top-1 cross-accuracy and variance-norm ratio of the submitted gradients. Section D in the appendix includes all our results.

One first important remark is that our new formulation either obtain similar, or (subtantly) increased maximum top-1 cross-accuracy measured, compared to the standard formulation in the exact same settings. Namely, in only 4 pairs of runs (0.23% of all the tested pairs) did our formulation lead to a decreased maximum top-1 cross-accuracy. Also, these decreases were only observed with the fully connected model, using Krum against Xie et al. (2019a), and for each of these 4 runs using any of the 4 other seeds made the decrease disappear.

In all our experiments, we observe a strong correlation between higher top-1 cross-accuracies and lower average losses; e.g. see Figure 4. The two state-of-the-art attacks decreased the accuracy by at least  $20\%$ , compared to the unattacked case (see "No attack" in Figure 3), in  $25.80\%$  and  $70.80\%$  of the runs with respectively the fully connected and convolutional models.

Focusing on the convolutional model, when roughly an half of the workers are Byzantine, both attacks actually succeed in decreasing the accuracy by at least  $20\%$  in  $100\%$  of our runs. Our technique manages to recover at least  $10\%$  and  $20\%$  in respectively  $79.75\%$  and  $49.25\%$  of these runs. When roughly a quarter of the workers are Byzantine, the attacks decrease the accuracy by at least  $20\%$  in  $46.46\%$  of our runs. Our technique then manages to recover at least  $20\%$  in  $95.07\%$  of these runs. Figure 3 shows a fraction of these runs.

Technically, our reformulation aims at reducing the variance-norm ratio of the aggregated gradients. Intuitively, this ratio is expected to increase as the loss decreases; more correctly as the norm of the gradient decreases. For instance, Figure 5 displays the variance-norm ratios of Trimmed Mean and Bulyan using the same settings as in Figure 4. At least before the final cross-accuracies are reached, our technique consistently decreases the variance-norm ratio of the aggregated gradients. Also, we consistently observed in the experiments that reducing the learning rate indeed reduces the variance-norm ratio (e.g. Figure 5,  $t \geq 1500$ ).

![](images/02e672e5a536e71e267920b12d2f772c53e188c6e4af2dd57775dec81f9cdd9b.jpg)

![](images/764f6172222a0164f053764c86de8c92d3b18fb0d6a4dacba1f1875c681a05fd.jpg)

![](images/ef5176a2630fea22cc3a35226ad3a821ba03163b4446040a33774ca564d45cb3.jpg)  
Figure 3: CIFAR-10 and the convolutional model (Section 4.1), with  $n = 25$ ,  $f = 5$  and  $\alpha_{t} = 0.01$  if  $t < 1500$  else  $\alpha_{t} = 0.001$ , under attack from (Baruch et al., 2019). Each line and colored surface correspond to respectively the average and standard deviation of the top-1 cross-correct accuracy over 5 seeded runs. Only two parameters change between graphs: where momentum is computed (at the server or at the workers), and which flavor of momentum is employed.

![](images/077c3c5ebce0f75ddf8ab91b0f09976f3923243c86e3d32f2f22d55a9420c8c2.jpg)

![](images/db2b85f65cea36f75729e71c298dcd2a6827bfcc555af8dde7e05031269ebb31.jpg)

![](images/e56ca57363bb049794aa6b080880aceb87f770c456a46c51089819c1e26f9eb4.jpg)

![](images/887e58f3180c918e22c1b2d833347f39c885772b607ec95ee8bbc57f99230247.jpg)  
Figure 4: Accuracy and average loss, CIFAR-100 and the convolutional model, with  $n = 25$ ,  $f = 5$  and  $\alpha_{t} = 0.01$  if  $t < 1500$  else  $\alpha_{t} = 0.001$ , under attack from (Xie et al., 2019a).

![](images/5d5babefb590b4d411fd991209d49001ffc69757efad86c968317c0f32884d3f.jpg)

![](images/7b13c6d9e5292f1d8a14a472f684ac476c613b190102380c8daee441278df56e.jpg)  
Figure 5: Same settings as in Figure 4, variance-norm ratios of Trimmed Mean and Bulyan with momentum at the workers. "sample" corresponds to the variance-norm ratio of the sampled gradients, and "submit" to the variance-norm ratio of the submitted gradients.

![](images/e645d3138c24efd1baf77596f4e8a1328d1baca0172d7483109ae41a0d3738c9.jpg)

# 5 RELATED AND FUTURE WORK

Alternative Byzantine-resilient Approaches. The Byzantine abstraction is a very general fault model that has long been studied in distributed computing (Lamport et al., 1982). The standard, golden solution for Byzantine fault tolerance is the state machine replication approach (Schneider, 1990). This approach is however based on replication, which is known to be unsuitable for distributed machine learning and stochastic gradient descent.

Chen et al. (2018) was the first Byzantine-resilient mechanism based on a redundancy scheme rather than statistical robustness. While the proposed mechanism is not vulnerable to the attacks discussed in this paper, it induces (due to the redundancy) substantial computational costs compared to statistically-robust techniques. Rajput et al. (2019) combines Chen et al. (2018) with statistically-robust GARs into a hybrid system, and achieves an improved aggregation time. Under the requirements of both redundancy-based schemes and statistically-robust GARs, Rajput et al. (2019) can significantly decrease the voting power of the adversary, and can consequently also deflect attacks in these cases. Xie et al. (2019b), and its follow-up (Xie, 2019), introduced the concept of suspicion-based fault-tolerance: the parameter server uses the loss of each received gradient to assign it a score. The lowest scoring gradients are then filtered-out, and the remaining gradients are averaged and used to update the model.

Momentum-based Variance Reduction. Our algorithm is different from Cutkosky & Orabona (2019), as instead of reducing the variance of the gradients, we actually increase it (Equation 6). What we seek to reduce is the variance-norm ratio, which is the key quantity for any Byzantine-resilient GAR approximating a high-dimensional median, e.g. Krum, Median, as well as Yang & Bajwa (2019b;a); Chen et al. (2017); Muñoz-González et al. (2019) $^6$ . Some of the ideas introduced in Cutkosky & Orabona (2019) could nevertheless help further improve Byzantine resilience. For instance, introducing an adaptive learning rate which decreases depending on the curvature of the parameter trajectory is an appealing approach to further reduce the variance-norm ratio (Equation 8). The computation of momentum at the workers has also been used in the literature for the purpose of gradient compression (Lin et al., 2018). These techniques are nevertheless not (meant to be) Byzantine-resilient.

Future Work. The theoretical condition to reduce the variance-norm ratio of the submitted gradients (compared to the variance-norm ratio of the sampled gradients at the same step), in Section B.2, shows that momentum at the workers is a double-edged sword. The problem is that  $s_t$  can become negative: the norm of the momentum gradient would then be decreased, increasing the variance-norm ratio. While the ability to cross narrow, local minima is recognized as an accelerator (Goh, 2017), for the purpose of Byzantine-resilience we want to ensure momentum at the workers does not increase the variance-norm ratio (compared to the variance-norm ratio of the sampled gradients at the same step). The theoretical condition for this purpose is given in Equation 7. One simple amendment would then be to use momentum at the workers when Equation 7 is satisfied, and fallback to computing it at the server otherwise. Also, a more complex, possible future approach could be to dynamically adapt the momentum factor  $\mu$ , decreasing it as the curvature increases.

Asynchronous SGD. We focused in this work on the synchronous setting, which received most of the attention in the Byzantine-resilient literature. Yet, we do not see any issue that would prevent our work from being applied in asynchronous settings. Specifically, combining our idea with a filtering scheme such as Kardam (Damaskinos et al., 2018) is in principle possible, as this filter and momentum commute. However, further analysis of the interplay between the dynamics of stale gradients and the dynamics of momentum remain necessary.

Byzantine Servers. While most of the research on Byzantine-resilience gradient descent has focused on the workers' side, assuming a reliable server, recent efforts have started tackling Byzantine servers (El-Mhamdi et al., 2020). Our reduction of the variance-norm ratio strengthens the gradient aggregation phase, which is necessary whether we deal with Byzantine workers or Byzantine servers. An interesting open question is whether the dynamics of momentum could positively affect the model drift between different parameter servers in a Byzantine context. Any quantitative answer to this question could enable the use of our method in fully decentralised Byzantine resilient gradient descent.

# REFERENCES

Dan Alistarh, Zeyuan Allen-Zhu, and Jerry Li. Byzantine stochastic gradient descent. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, 3-8 December 2018, Montréal, Canada, pp. 4618-4628, 2018.  
Moran Baruch, Gilad Baruch, and Yoav Goldberg. A little is enough: Circumventing defenses for distributed learning. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, 8-14 December 2019, Long Beach, CA, USA, 2019.  
Jeremy Bernstein, Jiawei Zhao, Kamyar Azizzadenesheli, and Anima Anandkumar. signsgd with majority vote is communication efficient and fault tolerant. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019, 2019.  
Peva Blanchard, El-Mahdi El-Mhamdi, Rachid Guerraoui, and Julien Stainer. Machine learning with adversaries: Byzantine tolerant gradient descent. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, 4-9 December 2017, Long Beach, CA, USA, pp. 119-129, 2017.  
Léon Bottou. Online learning and stochastic approximations. Online learning in neural networks, 17(9):142, 1998.  
Lingjiao Chen, Hongyi Wang, Zachary B. Charles, and Dimitris S. Papailiopoulos. DRACO: byzantine-resilient distributed training via redundant gradients. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, July 10-15, 2018, pp. 902-911, 2018.  
Yudong Chen, Lili Su, and Jiaming Xu. Distributed statistical machine learning in adversarial settings: Byzantine gradient descent. CoRR, abs/1705.05491, 2017.  
Ashok Cutkosky and Francesco Orabona. Momentum-based variance reduction in nonconvex SGD. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC, Canada, pp. 15210-15219, 2019. URL http://papers.nips.cc/paper/9659-momentum-based-variance-reduction-in-non-convex-sgd.  
Georgios Damaskinos, El-Mahdi El-Mhamdi, Rachid Guerraoui, Rhicheek Patra, and Mahsa Taziki. Asynchronous byzantine machine learning (the case of SGD). In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, July 10-15, 2018, pp. 1153-1162, 2018.  
El-Mahdi El-Mhamdi, Rachid Guerraoui, and Sébastien Rouault. The hidden vulnerability of distributed learning in byzantium. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, July 10-15, 2018, pp. 3518-3527, 2018.  
El-Mahdi El-Mhamdi, Rachid Guerraoui, Arsany Guirguis, Lê Nguyen Hoang, and Sébastien Rouault. Genuinely distributed byzantine machine learning. In Yuval Emek and Christian Cachin (eds.), PODC '20: ACM Symposium on Principles of Distributed Computing, Virtual Event, Italy, August 3-7, 2020, pp. 355-364. ACM, 2020. doi: 10.1145/3382734.3405695. URL https://doi.org/10.1145/3382734.3405695.  
Gabriel Goh. Why momentum really works. Distill, 2017. doi: 10.23915/distill.00006. URL http://distill.pub/2017/momentum.  
Leslie Lamport, Robert E. Shostak, and Marshall C. Pease. The byzantine generals problem. ACM Trans. Program. Lang. Syst., 4(3):382-401, 1982. doi: 10.1145/357172.357176.

Mu Li, David G. Andersen, Jun Woo Park, Alexander J. Smola, Amr Ahmed, Vanja Josifovski, James Long, Eugene J. Shekita, and Bor-Yiing Su. Scaling distributed machine learning with the parameter server. In 11th USENIX Symposium on Operating Systems Design and Implementation, OSDI '14, Broomfield, CO, USA, October 6-8, 2014, pp. 583-598, 2014.  
Yujun Lin, Song Han, Huizi Mao, Yu Wang, and Bill Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=SkhQHMWOW.  
Kuang Liu. Train cifar-10 with pytorch, 2019. URL https://github.com/kuangliu/pytorch-cifar/blob/ab908327d44bf9b1d22cd333a4466e85083d3f21/main.py#L33.  
Luis Muñoz-González, Kenneth T Co, and Emil C Lupu. Byzantine-robust federated machine learning through adaptive model averaging. arXiv preprint arXiv:1909.05125, 2019.  
Yurii Nesterov. A method for solving a convex programming problem with convergence rate o(1/k2). Soviet Mathematics Doklady, 27:372-367, 1983.  
Boris Polyak. Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4:1-17, 12 1964. doi: 10.1016/0041-5553(64)90137-5.  
Shashank Rajput, Hongyi Wang, Zachary Charles, and Dimitris Papailiopoulos. Detox: A redundancy-based framework for faster and more robust gradient aggregation. Neural Information Processing Systems, 2019.  
Fred B Schneider. Implementing fault-tolerant services using the state machine approach: A tutorial. ACM Computing Surveys (CSUR), 22(4):299-319, 1990.  
WANG TianXiang, ZhongLong ZHENG, TANG ChangBing, and PENG Hao. Aggregation rules based on stochastic gradient descent in byzantine consensus. In 2019 IEEE 8th Joint International Information Technology and Artificial Intelligence Conference (ITAIC), pp. 317-324. IEEE, 2019.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.  
Cong Xie. Zeno++: robust asynchronous SGD with arbitrary number of byzantine workers. CoRR, abs/1903.07020, 2019.  
Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Generalized byzantine-tolerant SGD. CoRR, abs/1802.10116, 2018a. URL http://arxiv.org/abs/1802.10116.  
Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Phocas: dimensional byzantine-resilient stochastic gradient descent. CoRR, abs/1805.09682, 2018b. URL http://arxiv.org/abs/1805.09682.  
Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Fall of empires: Breaking byzantine-tolerant SGD by inner product manipulation. In Proceedings of the Thirty-Fifth Conference on Uncertainty in Artificial Intelligence, UAI 2019, Tel Aviv, Israel, July 22-25, 2019, pp. 83, 2019a.  
Cong Xie, Sanmi Koyejo, and Indranil Gupta. Zeno: Distributed stochastic gradient descent with suspicion-based fault-tolerance. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, pp. 6893-6901, 2019b.  
Zhixiong Yang and Waheed U Bajwa. Bridge: Byzantine-resilient decentralized gradient descent. arXiv preprint arXiv:1908.08098, 2019a.  
Zhixiong Yang and Waheed U Bajwa. Byrdie: Byzantine-resilient distributed coordinate descent for decentralized learning. IEEE Transactions on Signal and Information Processing over Networks, 2019b.

Zhixiong Yang, Arpita Gang, and Waheed U Bajwa. Adversary-resilient inference and machine learning: From distributed to decentralized. arXiv preprint arXiv:1908.08649, 2019.

Dong Yin, Yudong Chen, Kannan Ramchandran, and Peter L. Bartlett. Byzantine-robust distributed learning: Towards optimal statistical rates. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, July 10-15, 2018, pp. 5636-5645, 2018.
