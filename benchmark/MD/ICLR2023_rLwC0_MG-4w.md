# DENOISING DIFFUSION ERROR CORRECTION CODES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Error correction code (ECC) is an integral part of the physical communication layer, ensuring reliable data transfer over noisy channels. Recently, neural decoders have demonstrated their advantage over classical decoding techniques. However, recent state-of-the-art neural decoders suffer from high complexity and lack the important iterative scheme characteristic of many legacy decoders. In this work, we propose to employ denoising diffusion models for the soft decoding of linear codes at arbitrary block lengths. Our framework models the forward channel corruption as a series of diffusion steps that can be reversed iteratively. Three contributions are made: (i) a diffusion process suitable for the decoding setting is introduced, (ii) the neural diffusion decoder is conditioned on the number of parity errors, which indicates the level of corruption at a given step, (iii) a line search procedure based on the code's syndrome obtains the optimal reverse diffusion step size. The proposed approach demonstrates the power of diffusion models for ECC and is able to achieve state of the art accuracy, outperforming the other neural decoders by sizable margins, even for a single reverse diffusion step. Our code is attached as supplementary material.

# 1 INTRODUCTION

Reliable digital communication is of major importance in the modern information age and involves the design of codes that can be robustly decoded despite noisy transmission channels. The target decoding is defined by the NP-hard maximum likelihood rule, and the efficient decoding of commonly employed families of codes, such as algebraic block codes, remains an open problem.

Recently, powerful learning-based techniques have been introduced. Model-free decoders (O'Shea & Hoydis, 2017; Gruber et al., 2017; Kim et al., 2018) employ generic neural networks and may potentially benefit from the application of powerful deep architectures that have emerged in recent years in various fields. A Transformer-based decoder that is able to incorporate the code into the architecture has been recently proposed by Choukroun & Wolf (2022). It outperforms existing methods by sizable margins, at a fraction of their time complexity. The decoder's objective in this model is to predict the noise corruption, to recover the transmitted codeword (Bennatan et al., 2018).

Deep generative neural networks have shown significant progress over the last years. Denoising Diffusion Probabilistic Models (DDPM) (Ho et al., 2020b) are an emerging class of likelihood-based generative models. Such methods use diffusion models and denoising score matching to generate new samples, for example, images (Dhariwal & Nichol, 2021) or speech (Chen et al., 2020a). The DDPM model learns to perform a reversed diffusion process on a Markov chain of latent variables, and generates samples by gradually removing noise from a given signal.

One major drawback of model-free approaches is the high space/memory requirement and time complexity that hamper its deployment on constrained hardware. Moreover, the lack of an iterative solution means that both highly and slightly corrupted codewords go through the same computationally demanding neural decoding procedure.

In this work, we consider the error correcting code paradigm via the prism of diffusion processes. The channel codeword corruption can be viewed as an iterative forward diffusion process to be reversed via an adapted DDPM. As far as we can ascertain, this is the first adaptation of diffusion models to error correction codes.

Beyond the conceptual novelty, we make three technical contributions: (i) our framework is based on an adapted diffusion process that simulates the coding and transmission processes, (ii) we further condition the denoising model on the number of parity-check errors, as an indicator of the signal's level of corruption, and (iii) we propose a line-search procedure that minimizes the denoised code syndrome, in order to provide an optimal step size for the reverse diffusion.

Applied to a wide variety of codes, our method outperforms the state-of-the-art learning-based solutions by very large margins, employing extremely shallow architectures. Furthermore, we show that even a single reverse diffusion step with a controlled step size can outperform concurrent methods.

# 2 RELATED WORKS

The emergence of deep learning for communication and information theory applications has demonstrated the advantages of neural networks in many tasks, such as channel equalization, modulation, detection, quantization, compression, and decoding (Ibnkahla, 2000). Model-free decoders employ general neural network architectures (Cammerer et al., 2017; Gruber et al., 2017; Kim et al., 2018; Bennatan et al., 2018). However, the exponential number of possible codewords makes the decoding of large codes unfeasible. Bennatan et al. (2018) preprocess the channel output to allow the decoder to remain provably invariant to the transmitted codeword and to eliminate risks of overfitting. Model-free approaches generally make use of multilayer perceptron networks or recurrent neural networks to simulate the iterative process existing in many legacy decoders (Gruber et al., 2017; Kim et al., 2018; Bennatan et al., 2018). However, many architectures have difficulties in learning the code or analyzing the reliability of the output, and require prohibitive parameterization or expensive graph permutation preprocessing (Bennatan et al., 2018).

Recently, Choukroun & Wolf (2022) proposed the Error Correction Code Transformer (ECCT), obtaining SOTA performance. The model embeds the signal elements into a high-dimensional space where analysis is more efficient, while the information about the code is integrated via a masked self-attention mechanism.

Diffusion Probabilistic Models were first introduced by Sohl-Dickstein et al. (2015), who presented the idea of using a slow iterative diffusion process to break the structure of a given distribution while learning the reverse neural diffusion process, in order to restore the structure in the data. Song & Ermon (2019) proposed a new score-based generative model, building on the work of Hyvarinen & Dayan (2005), as a way of modeling a data distribution using its gradients, and then sampling using Langevin dynamics (Welling & Teh, 2011).

The DDPM method of Ho et al. (2020b) is a generative model based on the neural diffusion process that applies score matching for image generation. Song et al. (2020b) leverage techniques from stochastic differential equations to improve the sample quality obtained by score-based models; Song et al. (2020a) and Nichol & Dhariwal (2021a) propose methods for improving sampling speed; Nichol & Dhariwal (2021a) and Sahara et al. (2021) demonstrated promising results on the difficult ImageNet generation task, using upsampling diffusion models. Several extensions to other fields, such as audio (Kong et al., 2020; Chen et al., 2020b), have been proposed.

# 3 BACKGROUND

We provide in this section the necessary background on error correction coding and DDPM.

Coding We assume a standard transmission that uses a linear code  $C$ . The code is defined by the binary generator matrix  $G$  of size  $k \times n$  and the binary parity check matrix  $H$  of size  $(n - k) \times n$  defined such that  $GH^T = 0$  over the order 2 Galois field  $GF(2)$ .

The input message  $m \in \{0,1\}^k$  is encoded by  $G$  to a codeword  $x \in C \subset \{0,1\}^n$  satisfying  $Hx = 0$  and transmitted via a Binary-Input Symmetric-Output channel, e.g., an AWGN channel. Let  $y$  denote the channel output represented as  $y = x_s + \varepsilon$ , where  $x_s$  denotes the Binary Phase Shift Keying (BPSK) modulation of  $x$  (i.e., over  $\{\pm 1\}$ ), and  $\varepsilon$  is a random noise independent of the transmitted  $x$ . The main goal of the decoder  $f: \mathbb{R}^n \to \mathbb{R}^n$  is to provide a soft approximation  $\hat{x} = f(y)$  of the codeword.

We follow the preprocessing of Bennatan et al. (2018); Choukroun & Wolf (2022), in order to remain provably invariant to the transmitted codeword and to avoid overfitting. The preprocessing transforms  $y$  to a vector of dimensionality  $2n - k$  defined as

$$
\tilde {y} = h (y) = \left[ | y |, s (y) \right], \tag {1}
$$

where,  $[\cdot ,\cdot ]$  denotes vector concatenation,  $|y|$  denotes the absolute value (magnitude) of  $y$  and  $s(y)\in \{0,1\}^{n - k}$  denotes the binary code syndrome. The syndrome is obtained via the  $GF(2)$  multiplication of the binary mapping of  $y$  with the parity check matrix such that

$$
s (y) = H y _ {b} := H \operatorname {b i n} (y) := H \left(0. 5 (1 - \operatorname {s i g n} (y))\right). \tag {2}
$$

The induced parameterized decoder  $\epsilon_{\theta}:\mathbb{R}^{2n - k}\to \mathbb{R}^n$  with parameters  $\theta$  aims to predict the multiplicative noise denoted as  $\tilde{\varepsilon}$  and defined such that  $y = x_{s}\cdot \tilde{\varepsilon}$ . The final soft prediction takes the form  $\hat{x} = y\cdot \epsilon_{\theta}(|y|,Hy_b)$ .

Denoising Diffusion Probability Model (DDPM) Ho et al. (2020a) assume a data distribution  $x_0 \sim q(x)$  and a Markovian noisig process  $q$  that gradually adds noise to the data to produce noisy samples  $\{x_i\}_{i=1}^T$ . Each step of the corruption process adds Gaussian noise according to some variance schedule given by  $\beta_t$  such that

$$
q \left(x _ {t} \mid x _ {t - 1}\right) \sim \mathcal {N} \left(x _ {t}; \sqrt {1 - \beta_ {t}} x _ {t - 1}, \beta_ {t} I\right)) \tag {3}
$$

$$
x _ {t} = \sqrt {1 - \beta_ {t}} x _ {t - 1} + \sqrt {\beta_ {t}} z _ {t - 1}, z _ {t - 1} \sim \mathcal {N} (0, I).
$$

$q(x_{t}|x_{0})$  can be expressed as a Gaussian distribution such that, with  $\alpha_{t}\coloneqq 1 - \beta_{t}$  and  $\bar{\alpha}_t\coloneqq \prod_{s = 0}^t\alpha_s$  , we have

$$
q \left(x _ {t} \mid x _ {0}\right) \sim \mathcal {N} \left(x _ {t}; \sqrt {\bar {\alpha} _ {t}} x _ {0}, (1 - \bar {\alpha} _ {t}) I\right) \tag {4}
$$

$$
x _ {t} = \sqrt {\bar {\alpha} _ {t}} x _ {0} + \varepsilon \sqrt {1 - \bar {\alpha} _ {t}}, \varepsilon \sim \mathcal {N} (0, I).
$$

The intractable reverse diffusion process  $q(x_{t-1}|x_t)$  approaches a diagonal Gaussian distribution as  $\beta_t \underset{t \to \infty}{\longrightarrow} 0$  (Sohl-Dickstein et al., 2015) and can be approximated using a neural network  $p_\theta(x_t)$  in order to predict the Gaussian statistics. The model is trained by stochastically optimizing the random terms of the variational lower bound of the negative log-likelihood function.

One can find via Bayes' theorem that the posterior  $q(x_{t-1}|x_t, x_0)$  is also Gaussian, making the objective a sum of tractable KL divergences between Gaussians. Ho et al. (2020a) found a more practical objective, defined via the training of a model  $\epsilon_\theta(x_t, t)$  that predicts the additive noise  $\varepsilon$  from Eq. 4 as follows

$$
\mathcal {L} _ {D D P M} (\theta) = \mathbb {E} _ {t \sim \mathcal {U} [ 1, T ], x _ {0} \sim q (x), \varepsilon \sim \mathcal {N} (0, I)} \| \varepsilon - \epsilon_ {\theta} (x _ {t}, t) \| ^ {2}. \tag {5}
$$

The distribution  $q(x_{T})$  is assumed to be a nearly isotropic Gaussian distribution, such that sampling  $x_{T}$  is trivial. Thus, the reverse diffusion process is given by the following iterative process

$$
x _ {t - 1} = \frac {1}{\sqrt {\alpha_ {t}}} \left(x _ {t} - \frac {1 - \alpha_ {t}}{\sqrt {1 - \bar {\alpha} _ {t}}} \epsilon_ {\theta} (x _ {t}, t)\right). \tag {6}
$$

# 4 DENOISING DIFFUSION ERROR CORRECTION CODES

We present the elements of the proposed denoising diffusion for decoding and the proposed architecture, together with its training procedure. An illustration of the coding setting and the proposed decoding framework are given in Figure 1.

# 4.1 DATA TRANSMISSION AS A FORWARD DIFFUSION PROCESS

Given a codeword  $x_0$  sampled from the Code distribution  $x_0 \sim q(x)$ , we propose to define the codeword transmission procedure  $y = x_0 + \sigma \varepsilon$  as a forward diffusion process adding a small amount

![](images/f448e8f34bc0cb2a448fe57d51a803f4ba90053c8f2f0bce3bc46b1e00d7da1a.jpg)  
Figure 1: Illustration of the communication system. We train a parameterized iterative decoder  $\epsilon_{\theta}$  conditioned on the number of parity check errors. The decoding is performed iteratively through the reverse diffusion process, as described in this paper.

of Gaussian noise to the sample in  $t$  steps with  $t \in (0, \dots, T)$ , where the step sizes are controlled by a variance schedule  $\{\beta_{t}\}_{t=0}^{T}$ . In our setting, we propose the following unscaled forward diffusion

$$
q \left(x _ {t} := y \mid x _ {t - 1}\right) \sim \mathcal {N} \left(x _ {t}; x _ {t - 1}, \beta_ {t} I\right). \tag {7}
$$

Thus, for a given received word  $y$  and a corresponding  $t$ , we consider  $y$  as a codeword that has been corrupted gradually, such that for  $\varepsilon \sim \mathcal{N}(0,I)$

$$
\begin{array}{l} y := x _ {t} = x _ {0} + \sigma \varepsilon , \varepsilon \sim \mathcal {N} (0, I) \\ = x _ {0} + \sqrt {\bar {\beta} _ {t}} \varepsilon , \tag {8} \\ \sim \mathcal {N} (x _ {t}; x _ {0}, \bar {\beta} _ {t} I), \\ \end{array}
$$

where  $\bar{\beta}_t = \sum_{i=1}^t \beta_i$  and  $\sigma$  defines the level of corruption of the AWGN channel. Thus, the transmission of data over noisy communication channels can be defined as a modified iterative diffusion process to be reversed for decoding.

# 4.2 DECODING AS A REVERSE DIFFUSION PROCESS

Following Bayes' theorem, the posterior  $q(x_{t-1}|x_t,x_0)$  is a Gaussian such that  $q(x_{t}|x_{t-1},x_{0})\sim \mathcal{N}(x_{t};\tilde{\mu}_{t}(x_{t},x_{0}),\tilde{\beta}_{t}I)$ , where, according to Eq. 8, we have

$$
\tilde {\mu} _ {t} \left(x _ {t}, x _ {0}\right) = \frac {\bar {\beta} _ {t}}{\bar {\beta} _ {t} + \beta_ {t}} x _ {t} + \frac {\beta_ {t}}{\bar {\beta} _ {t} + \beta_ {t}} x _ {0} = x _ {t} - \frac {\sqrt {\bar {\beta} _ {t}} \beta_ {t}}{\bar {\beta} _ {t} + \beta_ {t}} \varepsilon , \text {a n d} \tilde {\beta} _ {t} = \frac {\bar {\beta} _ {t} \beta_ {t}}{\bar {\beta} _ {t} + \beta_ {t}}. \tag {9}
$$

The full derivation is given in the Appendix A. Similarly to (Sohl-Dickstein et al., 2015; Ho et al., 2020b), we wish to approximate the intractable Gaussian reverse diffusion process  $q(x_{t}|x_{t - 1})$  such that

$$
q \left(x _ {t} \mid x _ {t - 1}\right) \approx p _ {\theta} \left(x _ {t} \mid x _ {t - 1}\right) \sim \mathcal {N} \left(x _ {t}; \mu_ {\theta} \left(x _ {t}, t\right), \tilde {\beta} _ {t} I\right), \tag {10}
$$

with fixed variance  $\tilde{\beta}_t$ . Following the simplified objective of Ho et al. (2020b), one would adapt the negative log-likelihood approximation such that the decoder predicts the additive noise of the adapted diffusion process and

$$
\mathcal {L} (\theta) = \mathbb {E} _ {t \sim \mathcal {U} [ 1, T ], x _ {0} \sim q (x), \varepsilon \sim \mathcal {N} (0, I)} [ \| \varepsilon - \epsilon_ {\theta} (x _ {0} + \sqrt {\bar {\beta} _ {t}} \varepsilon , t) \| ^ {2} ]. \tag {11}
$$

One interesting property of the syndrome-based approach of Bennatan et al. (2018) is that, similarly to denoising diffusion models, in order to retrieve the original codeword, the decoder's objective is to predict the channel's noise. However, the syndrome-based approach enforces the prediction of the multiplicative noise  $\tilde{\varepsilon}$  instead of the additive noise  $\varepsilon$ , in contrast to classic diffusion models. We note, however, that the exact value of the multiplicative noise is not important for hard decoding, but only its sign since  $x_{s} = \mathrm{sign}(y\tilde{\varepsilon})$ .

Therefore, we propose to learn the hard (i.e., the sign) prediction of the multiplicative noise using the binary cross entropy loss as a surrogate objective, such that

$$
\mathcal {L} (\theta) = - \mathbb {E} _ {t, x _ {0}, \varepsilon} \log \left(\epsilon_ {\theta} \left(x _ {0} + \sqrt {\bar {\beta} _ {t}} \varepsilon , t\right), \tilde {\varepsilon} _ {b}\right), \tag {12}
$$

where the target binary multiplicative noise is defined as  $\tilde{\varepsilon}_b = \mathrm{bin}\big(x_0(x_0 + \sqrt{\bar{\beta}_t}\varepsilon)\big)$ .

![](images/ea99aec1ae415238262347c6e0a58a68d45024ef485565d83ba1ceb67a87a48a.jpg)  
Figure 2: Reverse diffusion dynamics on a (3,1) repetition code. The two points represent the two only signed codewords:  $\pm (1,1,1)$ . The colors are defined by Maximum Likelihood decoding. Evidently, the denoising diffusion model reverses noisy codes towards the right distribution. An illustration of the forward process for this code is provided in Appendix B.

![](images/46ec1c4986ac7adbe90ef286c211b385f1987d157b075b64ab3eb07a8ffcd4e1.jpg)  
Figure 3: Influence of the noise or  $E_{b}N_{0}$  (normalized SNR) on the number of parity check errors for several codes. The greater the noise, the higher the number of parity check errors, which demonstrates that the syndrome conveys information about the level of noise.

# 4.3 DENOISING VIA PARITY CHECK CONDITIONING

The reverse denoising process of traditional DDPM is conditioned by the time step. Thus, by sampling Gaussian noise, which is assumed as equivalent to step  $t = T$ , one can fully reverse the diffusion by up to  $T$  iterations. In our case, we are not interested in a generative model, but in an exact iterative denoising scheme, where the original signal is only corrupted to a measured extent.

Moreover, a given noisy code conveys information about the level of noise via its syndrome, since  $s(y) = Hy = Hx + Hz = Hz$ . Fig.3 illustrates the impact of noise on the number of parity check errors. As we can see, one can approximate an injective function between the number of parity check errors and the amount of noise. Therefore, we suggest conditioning the diffusion decoder according to the number of parity check errors  $e_t$ , such that  $e_t \coloneqq e(x_t) = \sum_{i=1}^{n-k} s(x_t)_i \in \{0, \dots, n-k\}$ . The resulting training objective is now given by

$$
\mathcal {L} (\theta) = - \mathbb {E} _ {t, x _ {0}, \varepsilon} \log \left(\epsilon_ {\theta} \left(x _ {0} + \bar {\beta} _ {t} ^ {1 / 2} \varepsilon , \boldsymbol {e} _ {t}\right), \tilde {\varepsilon} _ {b}\right). \tag {13}
$$

Following this logic, the number of required denoising steps  $T = n - k$  is set as the maximum number of parity check errors. Similarly to the classical DDPM training procedure, sampling a time step  $t \sim \mathcal{U}(0, \dots, T)$  produces noise, which in turn induces a certain number of parity errors.

Denoting by BCE the binary cross-entropy loss, the training procedure of our method is given in Alg. 1. The framework assumes a random "time" sampling, producing a noise and then a syndrome to be corrected. Note that, our model-free solution is invariant to the transmitted codeword, and the diffusion decoding can be trained with one single codeword (Alg. 1 line 1).

Since the denoising model predicts the multiplicative noise  $\tilde{\varepsilon}$ , at inference time it needs to be transformed into its additive counterpart  $\varepsilon$  in order to perform the gradient step in the original additive diffusion process domain. We obtain the additive noise by subtracting the modulated predicted codeword  $\mathrm{sign}(\hat{x})$  from the noisy signal, such that

$$
\hat {\varepsilon} = y - \operatorname {s i g n} (\hat {x}) = y - \operatorname {s i g n} (\hat {\bar {\varepsilon}} y). \tag {14}
$$

Therefore, following Eq. 9, at inference time the reverse process is given by

$$
x _ {t - 1} = x _ {t} - \frac {\sqrt {\bar {\beta} _ {t}} \beta_ {t}}{\bar {\beta} _ {t} + \beta_ {t}} \left(x _ {t} - \operatorname {s i g n} \left(x _ {t} \hat {\varepsilon}\right)\right) = x _ {t} - \frac {\sqrt {\bar {\beta} _ {t}} \beta_ {t}}{\bar {\beta} _ {t} + \beta_ {t}} \left(x _ {t} - \operatorname {s i g n} \left(x _ {t} \epsilon_ {\theta} \left(x _ {t}, e _ {t}\right)\right)\right) \tag {15}
$$

<table><tr><td colspan="2">Algorithm 1: DDECC training procedure.</td></tr><tr><td>1:</td><td>x0 ∈ C</td></tr><tr><td>2:</td><td>Input: Parity check matrix H, noise schedule β1, ..., βT</td></tr><tr><td>3:</td><td>repeat</td></tr><tr><td>4:</td><td>t ~ U({1,..., T})</td></tr><tr><td>5:</td><td>ε ~ N(0,I)</td></tr><tr><td>6:</td><td>xt = x0 + √βtε = x0ε</td></tr><tr><td>7:</td><td>Take gradient descent step on: BCE(εθ(xt, et), bin(ε))</td></tr><tr><td>8:</td><td>until converged</td></tr></table>

<table><tr><td colspan="2">Algorithm 2: DDECC sampling procedure</td></tr><tr><td>1:</td><td>Input: Parity check matrix H, channel&#x27;s output y</td></tr><tr><td>2:</td><td>for n-k iterations do</td></tr><tr><td>3:</td><td>γ = e(bin(y))</td></tr><tr><td>4:</td><td>if γ = 0 then</td></tr><tr><td>5:</td><td>return bin(y)</td></tr><tr><td>6:</td><td>ˆ̂ = εθ(y,γ);ˆ̂ = y - sign(ˆ̂y)</td></tr><tr><td>7:</td><td>Get λ according to Eq. 16</td></tr><tr><td>8:</td><td>y = y - λ√βγβγ/βγ+βγˆ̂</td></tr><tr><td>9:</td><td>return bin(y)</td></tr></table>

The inference procedure is defined in Alg.2. If the syndrome is non-zero, we predict the multiplicative noise, extract the corresponding additive noise, and perform the reverse step. We illustrate in Fig.2 the reverse diffusion dynamics (gradient field) for a  $(3,1)$  repetition code, i.e.,  $G = (1,1,1)$ .

# 4.4 SYNDROME-BASED LINE SEARCH FOR REVERSE DIFFUSION STEP SIZE

One major limitation of the generative neural diffusion process is the large number of diffusion steps required - generally a thousand - in order to generate high-quality samples. Several methods proposed faster sampling procedures in order to accelerate data generation via schedule subsampling or step size correction (Nichol & Dhariwal, 2021b; San-Roman et al., 2021). In our configuration, one can assess the quality of the denoised signal via the value of its syndrome, i.e., the number of parity check errors, while a zero syndrome means a valid codeword.

Therefore, we propose to find the optimal step size  $\lambda$  by solving the following optimization problem

$$
\lambda^ {*} = \underset {\lambda \in \mathbb {R} ^ {+}} {\arg \min } \| s \left(x _ {t} - \lambda \frac {\sqrt {\bar {\beta} _ {t}} \beta t}{\bar {\beta} _ {t} + \beta_ {t}} \hat {\varepsilon}\right) \| _ {1}, \tag {16}
$$

where  $s(\cdot)$  denotes the syndrome computed over  $GF(2)$  as in Eq. 2.

While many line-search (LS) methods exist in numerical optimization (Nocedal & Wright, 2006), since the objective is highly non-differentiable, we suggest adopting a grid search procedure such that the search space becomes restricted to  $\lambda \in I$  where  $I$  is a predefined discrete segment. This parallelizable procedure reduces the number of iterations by a sizable factor, as shown in Section 5.

# 4.5 ARCHITECTURE AND TRAINING

The state-of-the-art ECCT architecture of Choukroun & Wolf (2022) is used as  $\epsilon_{\theta}$ . In this architecture, the capacity of the model is defined according to the chosen embedding dimension  $d$  and the number of self-attention layers  $N$ . In order to condition the network by the number of parity errors  $e_t \in \{0, \dots, n - k\}$ , we employ a  $d$  dimensional one hot encoding multiplied via Hadamard product with the initial elements' embedding of the ECCT. Denoting the ECCT's embedding of the  $i$  element as  $\phi_i$ , the new embedding is defined as  $\tilde{\phi}_i = \phi_i \odot \psi(e_t), \forall i$ , where  $\psi$  denotes the  $n - k$  one hot embedding. As a transformation of the syndrome,  $e_t$  remains also invariant to the codeword.

The discrete grid search of  $\lambda$  is uniformly sampled over  $I = [1, 20]$  with 20 samples, in order to find the optimal step size. A denser or a code adaptive sampling may improve the results, according to a predefined computation-speed trade-off. We show the distribution of optimal  $\lambda$  in Appendix C.

The Adam optimizer (Kingma & Ba, 2014) is used with 128 samples per mini-batch, for 2000 epochs, with 1000 mini-batches per epoch. The noise scheduling is constant and set to  $\beta_{t} = 0.01, \forall t$ . We initialized the learning rate to  $10^{-4}$  coupled with a cosine decay scheduler down to  $5 \cdot 10^{-6}$  at the end of training. No warmup (Xiong et al., 2020) was employed.

Training and experiments were performed on a 12GB Titan V GPU. The total training time ranged from 12 to 24 hours depending on the code length, and no optimization of the self-attention mecha

BP-based results are obtained after  $L = 5$  BP iterations in first row (i.e. 10-layer neural network) and at convergence results in second row are obtained after  $L = 50$  BP iterations (i.e., 100-layer neural network). Our performance is presented for six different architectures: for  $N = \{2,6\}$  and  $d = \{32,64,128\}$ . The presented results are obtained with the LS procedure.

Table 1: A comparison of the negative natural logarithm of Bit Error Rate (BER) for three normalized SNR values (4,5,6) of our method with literature baselines. Higher is better.  

<table><tr><td rowspan="2">Method</td><td colspan="2">BP</td><td colspan="2">ARBP</td><td colspan="2">ECCT N=2</td><td colspan="2">ECCT N=6</td><td colspan="2">Ours N=2</td><td colspan="2">Ours N=6</td></tr><tr><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td></tr><tr><td rowspan="3">Polar(64,32)</td><td>3.52</td><td>4.04</td><td>4.48</td><td>4.77</td><td>6.30</td><td>8.19</td><td>4.27</td><td>5.44</td><td>6.95</td><td>5.71</td><td>7.63</td><td>9.94</td></tr><tr><td>4.26</td><td>5.38</td><td>6.50</td><td>5.57</td><td>7.43</td><td>9.82</td><td>4.57</td><td>5.86</td><td>7.50</td><td>6.48</td><td>8.60</td><td>11.43</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>4.87</td><td>6.2</td><td>7.93</td><td>6.99</td><td>9.44</td><td>12.32</td></tr><tr><td rowspan="3">Polar(64,48)</td><td>4.15</td><td>4.68</td><td>5.31</td><td>5.25</td><td>6.96</td><td>9.00</td><td>4.92</td><td>6.46</td><td>8.41</td><td>5.82</td><td>7.81</td><td>10.24</td></tr><tr><td>4.74</td><td>5.94</td><td>7.42</td><td>5.41</td><td>7.19</td><td>9.30</td><td>5.14</td><td>6.78</td><td>8.9</td><td>6.15</td><td>8.20</td><td>10.86</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>5.36</td><td>7.12</td><td>9.39</td><td>6.36</td><td>8.46</td><td>11.09</td></tr><tr><td rowspan="3">Polar(128,64)</td><td>3.38</td><td>3.80</td><td>4.15</td><td>4.02</td><td>5.48</td><td>7.55</td><td>3.51</td><td>4.52</td><td>5.93</td><td>4.47</td><td>6.34</td><td>8.89</td></tr><tr><td>4.10</td><td>5.11</td><td>6.15</td><td>4.84</td><td>6.78</td><td>9.30</td><td>3.83</td><td>5.16</td><td>7.04</td><td>5.12</td><td>7.36</td><td>10.48</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>4.04</td><td>5.52</td><td>7.62</td><td>5.92</td><td>8.64</td><td>12.18</td></tr><tr><td rowspan="3">Polar(128,86)</td><td>3.80</td><td>4.19</td><td>4.62</td><td>4.81</td><td>6.57</td><td>9.04</td><td>4.30</td><td>5.58</td><td>7.34</td><td>5.36</td><td>7.45</td><td>10.22</td></tr><tr><td>4.49</td><td>5.65</td><td>6.97</td><td>5.39</td><td>7.37</td><td>10.13</td><td>4.49</td><td>5.90</td><td>7.75</td><td>5.75</td><td>8.16</td><td>11.29</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>4.75</td><td>6.25</td><td>8.29</td><td>6.31</td><td>9.01</td><td>12.45</td></tr><tr><td rowspan="3">Polar(128,96)</td><td>3.99</td><td>4.41</td><td>4.78</td><td>4.92</td><td>6.73</td><td>9.30</td><td>4.56</td><td>5.98</td><td>7.93</td><td>5.39</td><td>7.62</td><td>10.45</td></tr><tr><td>4.61</td><td>5.79</td><td>7.08</td><td>5.27</td><td>7.44</td><td>10.2</td><td>4.69</td><td>6.20</td><td>8.30</td><td>5.88</td><td>8.33</td><td>11.49</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>4.88</td><td>6.58</td><td>8.93</td><td>6.31</td><td>9.12</td><td>12.47</td></tr><tr><td rowspan="3">LDPC(49,24)</td><td>5.30</td><td>7.28</td><td>9.88</td><td>6.05</td><td>8.13</td><td>11.68</td><td>4.51</td><td>6.07</td><td>8.11</td><td>5.74</td><td>8.13</td><td>11.30</td></tr><tr><td>6.23</td><td>8.19</td><td>11.72</td><td>6.58</td><td>9.39</td><td>12.39</td><td>4.58</td><td>6.18</td><td>8.46</td><td>5.91</td><td>8.42</td><td>11.90</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>4.71</td><td>6.38</td><td>8.73</td><td>6.13</td><td>8.71</td><td>12.10</td></tr><tr><td rowspan="3">LDPC(121,60)</td><td>4.82</td><td>7.21</td><td>10.87</td><td>5.22</td><td>8.31</td><td>13.07</td><td>3.88</td><td>5.51</td><td>8.06</td><td>4.98</td><td>7.91</td><td>12.70</td></tr><tr><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>3.89</td><td>5.55</td><td>8.16</td><td>5.02</td><td>7.94</td><td>12.72</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>3.93</td><td>5.66</td><td>8.51</td><td>5.17</td><td>8.31</td><td>13.30</td></tr><tr><td rowspan="3">LDPC(121,70)</td><td>5.88</td><td>8.76</td><td>13.04</td><td>6.45</td><td>10.01</td><td>14.77</td><td>4.63</td><td>6.68</td><td>9.73</td><td>6.11</td><td>9.62</td><td>15.10</td></tr><tr><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>4.64</td><td>6.71</td><td>9.77</td><td>6.28</td><td>10.12</td><td>15.57</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>4.67</td><td>6.79</td><td>9.98</td><td>6.40</td><td>10.21</td><td>16.11</td></tr><tr><td rowspan="3">LDPC(121,80)</td><td>6.66</td><td>9.82</td><td>13.98</td><td>7.22</td><td>11.03</td><td>15.90</td><td>5.27</td><td>7.59</td><td>10.08</td><td>6.92</td><td>10.74</td><td>15.10</td></tr><tr><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>5.29</td><td>7.63</td><td>10.90</td><td>7.17</td><td>11.21</td><td>16.31</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>5.30</td><td>7.65</td><td>11.03</td><td>7.41</td><td>11.51</td><td>16.44</td></tr><tr><td rowspan="3">MacKay(96,48)</td><td>6.84</td><td>9.40</td><td>12.57</td><td>7.43</td><td>10.65</td><td>14.65</td><td>4.95</td><td>6.67</td><td>8.94</td><td>6.88</td><td>9.86</td><td>13.40</td></tr><tr><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>5.04</td><td>6.80</td><td>9.23</td><td>7.10</td><td>10.12</td><td>14.21</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>5.17</td><td>7.07</td><td>9.64</td><td>7.38</td><td>10.72</td><td>14.83</td></tr><tr><td rowspan="3">CCSDS(128,64)</td><td>6.55</td><td>9.65</td><td>13.78</td><td>7.25</td><td>10.99</td><td>16.36</td><td>4.35</td><td>6.01</td><td>8.30</td><td>6.34</td><td>9.80</td><td>14.40</td></tr><tr><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>4.41</td><td>6.09</td><td>8.49</td><td>6.65</td><td>10.40</td><td>15.46</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>4.59</td><td>6.42</td><td>9.02</td><td>6.88</td><td>10.90</td><td>15.90</td></tr><tr><td rowspan="3">BCH(63,36)</td><td>3.72</td><td>4.65</td><td>5.66</td><td>4.33</td><td>5.94</td><td>8.21</td><td>3.79</td><td>4.87</td><td>6.35</td><td>4.42</td><td>5.91</td><td>8.01</td></tr><tr><td>4.03</td><td>5.42</td><td>7.26</td><td>4.57</td><td>6.39</td><td>8.92</td><td>4.05</td><td>5.28</td><td>7.01</td><td>4.62</td><td>6.24</td><td>8.44</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>4.21</td><td>5.50</td><td>7.25</td><td>4.86</td><td>6.65</td><td>9.10</td></tr><tr><td rowspan="3">BCH(63,45)</td><td>4.08</td><td>4.96</td><td>6.07</td><td>4.80</td><td>6.43</td><td>8.69</td><td>4.47</td><td>5.88</td><td>7.81</td><td>5.16</td><td>7.02</td><td>9.75</td></tr><tr><td>4.36</td><td>5.55</td><td>7.26</td><td>4.97</td><td>6.90</td><td>9.41</td><td>4.66</td><td>6.16</td><td>8.17</td><td>5.41</td><td>7.49</td><td>10.25</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>4.79</td><td>6.39</td><td>8.49</td><td>5.60</td><td>7.79</td><td>10.93</td></tr><tr><td rowspan="3">BCH(63,51)</td><td>4.34</td><td>5.29</td><td>6.35</td><td>4.95</td><td>6.69</td><td>9.18</td><td>4.60</td><td>6.05</td><td>8.05</td><td>5.20</td><td>7.08</td><td>9.65</td></tr><tr><td>4.5</td><td>5.82</td><td>7.42</td><td>5.17</td><td>7.16</td><td>9.53</td><td>4.78</td><td>6.34</td><td>8.49</td><td>5.46</td><td>7.57</td><td>10.51</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>5.01</td><td>6.72</td><td>9.03</td><td>5.66</td><td>7.89</td><td>11.01</td></tr></table>

nism was employed. Per epoch, the training time was in the range of 19-40 and 40-102 seconds for the  $N = 2,6$  architectures, respectively.

# 5 EXPERIMENTS

To evaluate our method, we train the proposed architecture with three classes of linear block codes: Low-Density Parity Check (LDPC) codes (Gallager, 1962), Polar codes (Arikan, 2008) and Bose-Chaudhuri-Hocquenghem (BCH) codes (Bose & Ray-Chaudhuri, 1960). All parity check matrices are taken from Helmling et al. (2019).

The proposed architecture is defined solely by the number of encoder layers  $N$  and the dimension of the embedding  $d$ . We compare our method with the BP algorithm (Pearl, 1988), the recent Autoregressive hyper-network BP of Nachmani & Wolf (2021) (AR BP) and the SOTA ECCT (Choukroun & Wolf, 2022). Since our decoder is based on the ECCT, the contribution of the diffusion model

The other columns represent the mean and standard deviation of the number of iterations of the reverse process until convergence, i.e., convergence to zero syndrome.

Table 2: A comparison between the line search procedure and the regular reverse diffusion. The  $\Delta$  column denotes the difference between the logarithm of Bit Error Rate (BER) for three normalized SNR values (i.e.,  $\Delta = -\log (\mathrm{BER}_{LS}) + \log (\mathrm{BER}_{Reg}))$ .  

<table><tr><td rowspan="2">Method</td><td colspan="3">Δ N=2</td><td colspan="3">Δ N=6</td><td colspan="3">#It. Reg. N=2</td><td colspan="3">#It. Reg. N=6</td><td colspan="3">#It. LS N=2</td><td colspan="3">#It. LS N=6</td></tr><tr><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td><td>4</td><td>5</td><td>6</td></tr><tr><td rowspan="3">Polar(64,48)</td><td>-0.02</td><td>0.11</td><td>0.16</td><td>0.01</td><td>0.06</td><td>0.47</td><td>5.9 ± 4.9</td><td>3.3 ± 3.8</td><td>1.5 ± 2.7</td><td>5.7 ± 4.6</td><td>3.2 ± 3.8</td><td>1.5 ± 2.7</td><td>1.4 ± 2.3</td><td>0.7 ± 1.0</td><td>0.4 ± 0.5</td><td>1.2 ± 1.7</td><td>0.7 ± 0.9</td><td>0.4 ± 0.6</td></tr><tr><td>0.01</td><td>0.14</td><td>0.30</td><td>-0.03</td><td>0.12</td><td>0.43</td><td>5.8 ± 4.8</td><td>3.2 ± 3.8</td><td>1.5 ± 2.7</td><td>5.7 ± 4.6</td><td>3.2 ± 3.7</td><td>1.5 ± 2.7</td><td>1.3 ± 2.0</td><td>0.7 ± 0.9</td><td>0.4 ± 0.5</td><td>0.9 ± 1.0</td><td>0.6 ± 0.4</td><td>0.5 ± 0.5</td></tr><tr><td>0.00</td><td>0.16</td><td>0.40</td><td>-0.01</td><td>0.07</td><td>0.71</td><td>5.8 ± 4.7</td><td>3.2 ± 3.8</td><td>1.5 ± 2.7</td><td>5.7 ± 4.6</td><td>3.2 ± 3.7</td><td>1.5 ± 2.7</td><td>1.2 ± 1.8</td><td>0.7 ± 0.8</td><td>0.4 ± 0.5</td><td>1.0 ± 1.0</td><td>0.6 ± 0.4</td><td>0.5 ± 0.5</td></tr><tr><td rowspan="3">Polar(128,86)</td><td>-0.10</td><td>-0.11</td><td>-0.20</td><td>-0.20</td><td>-0.13</td><td>-0.21</td><td>16.5 ± 10.9</td><td>9.1 ± 7.3</td><td>4.7 ± 4.9</td><td>13.4 ± 7.5</td><td>8.3 ± 5.8</td><td>4.6 ± 4.6</td><td>4.0 ± 9.1</td><td>1.4 ± 3.4</td><td>0.8 ± 1.0</td><td>2.9 ± 4.4</td><td>1.6 ± 2.1</td><td>1.0 ± 1.2</td></tr><tr><td>-0.06</td><td>-0.04</td><td>0.00</td><td>-0.30</td><td>-0.34</td><td>-0.33</td><td>15.5 ± 10.0</td><td>8.8 ± 6.8</td><td>4.6 ± 4.8</td><td>13.1 ± 6.9</td><td>8.3 ± 5.7</td><td>4.6 ± 4.6</td><td>3.4 ± 8.3</td><td>1.3 ± 3.2</td><td>0.8 ± 0.9</td><td>1.3 ± 2.4</td><td>1.0 ± 0.6</td><td>0.7 ± 0.5</td></tr><tr><td>-0.09</td><td>-0.06</td><td>-0.10</td><td>-0.32</td><td>-0.21</td><td>-0.40</td><td>14.7 ± 9.2</td><td>8.6 ± 6.4</td><td>4.6 ± 4.7</td><td>13.0 ± 6.6</td><td>8.3 ± 5.7</td><td>4.6 ± 4.6</td><td>2.7 ± 6.8</td><td>1.2 ± 2.3</td><td>0.8 ± 0.7</td><td>1.2 ± 1.9</td><td>0.9 ± 0.5</td><td>0.7 ± 0.5</td></tr><tr><td rowspan="3">Polar(128,96)</td><td>-0.10</td><td>-0.13</td><td>-0.20</td><td>-0.11</td><td>0.04</td><td>-0.16</td><td>12.6 ± 8.8</td><td>6.5 ± 5.9</td><td>3.1 ± 3.9</td><td>10.5 ± 6.4</td><td>6.1 ± 5.0</td><td>3.1 ± 3.8</td><td>3.6 ± 7.4</td><td>1.2 ± 2.7</td><td>0.6 ± 0.8</td><td>2.2 ± 3.4</td><td>1.2 ± 1.4</td><td>0.7 ± 0.8</td></tr><tr><td>-0.09</td><td>-0.10</td><td>-0.10</td><td>-0.19</td><td>0.13</td><td>0.14</td><td>11.8 ± 8.0</td><td>6.3 ± 5.5</td><td>3.1 ± 3.8</td><td>10.32 ± 6.11</td><td>6.1 ± 4.9</td><td>3.1 ± 3.8</td><td>2.7 ± 6.0</td><td>1.1 ± 2.0</td><td>0.6 ± 0.6</td><td>1.2 ± 2.0</td><td>0.9 ± 0.5</td><td>0.6 ± 0.5</td></tr><tr><td>-0.16</td><td>-0.12</td><td>-0.20</td><td>-0.15</td><td>0.00</td><td>-0.13</td><td>11.1 ± 7.3</td><td>6.2 ± 5.2</td><td>3.1 ± 3.8</td><td>10.2 ± 5.9</td><td>6.1 ± 4.9</td><td>3.1 ± 3.8</td><td>2.0 ± 4.4</td><td>1.0 ± 1.3</td><td>0.6 ± 0.5</td><td>1.1 ± 1.5</td><td>0.9 ± 0.5</td><td>0.6 ± 0.5</td></tr><tr><td rowspan="3">LDPC(49,24)</td><td>0.06</td><td>0.03</td><td>0.23</td><td>-0.05</td><td>-0.31</td><td>-0.16</td><td>11.5 ± 7.0</td><td>7.4 ± 5.7</td><td>4.4 ± 4.5</td><td>10.9 ± 6.3</td><td>7.3 ± 5.4</td><td>4.4 ± 4.5</td><td>2.2 ± 5.1</td><td>1.0 ± 1.9</td><td>0.7 ± 0.7</td><td>2.2 ± 3.6</td><td>1.3 ± 1.5</td><td>0.8 ± 0.8</td></tr><tr><td>0.05</td><td>-0.06</td><td>0.20</td><td>-0.13</td><td>-0.12</td><td>-0.20</td><td>11.4 ± 7.0</td><td>7.4 ± 5.7</td><td>4.4 ± 4.5</td><td>10.9 ± 6.3</td><td>7.3 ± 5.4</td><td>4.4 ± 4.5</td><td>2.1 ± 4.8</td><td>1.0 ± 1.9</td><td>0.7 ± 0.6</td><td>1.5 ± 3.5</td><td>0.9 ± 1.1</td><td>0.7 ± 0.5</td></tr><tr><td>0.07</td><td>-0.10</td><td>0.20</td><td>-0.12</td><td>-0.13</td><td>-0.41</td><td>11.4 ± 7.0</td><td>7.4 ± 5.6</td><td>4.4 ± 4.5</td><td>10.9 ± 6.3</td><td>7.3 ± 5.4</td><td>4.4 ± 4.5</td><td>2.1 ± 4.8</td><td>1.0 ± 1.9</td><td>0.7 ± 0.6</td><td>1.4 ± 3.4</td><td>0.9 ± 1.1</td><td>0.7 ± 0.5</td></tr><tr><td rowspan="3">LDPC(121,80)</td><td>-0.23</td><td>-0.03</td><td>-0.40</td><td>-0.33</td><td>-0.46</td><td>-0.95</td><td>12.5 ± 7.9</td><td>7.3 ± 5.0</td><td>4.0 ± 3.8</td><td>11.4 ± 5.6</td><td>7.2 ± 4.7</td><td>4.0 ± 3.8</td><td>2.7 ± 7.5</td><td>1.0 ± 1.6</td><td>0.7 ± 0.5</td><td>1.2 ± 3.0</td><td>0.9 ± 0.4</td><td>0.7 ± 0.4</td></tr><tr><td>-0.13</td><td>-0.10</td><td>-0.30</td><td>-0.15</td><td>-0.42</td><td>0.81</td><td>12.5 ± 7.9</td><td>7.3 ± 4.9</td><td>4.0 ± 3.8</td><td>11.4 ± 5.8</td><td>7.2 ± 4.7</td><td>4.0 ± 3.8</td><td>2.7 ± 7.7</td><td>1.0 ± 1.7</td><td>0.7 ± 0.5</td><td>1.4 ± 3.7</td><td>0.9 ± 0.6</td><td>0.7 ± 0.4</td></tr><tr><td>-0.10</td><td>-0.17</td><td>-0.20</td><td>-0.28</td><td>-0.21</td><td>-0.27</td><td>12.4 ± 7.82</td><td>7.2 ± 4.9</td><td>4.0 ± 3.8</td><td>11.4 ± 5.6</td><td>7.2 ± 4.7</td><td>4.0 ± 3.8</td><td>3.1 ± 6.9</td><td>1.3 ± 1.7</td><td>0.8 ± 0.6</td><td>1.3 ± 3.2</td><td>0.9 ± 0.5</td><td>0.7 ± 0.4</td></tr><tr><td rowspan="3">MacKay(96,48)</td><td>-0.23</td><td>-0.16</td><td>-0.50</td><td>-0.09</td><td>0.29</td><td>0.00</td><td>15.3 ± 7.9</td><td>10.2 ± 5.5</td><td>6.4 ± 4.5</td><td>14.4 ± 5.7</td><td>10.0 ± 5.0</td><td>6.4 ± 4.4</td><td>2.8 ± 8.1</td><td>1.2 ± 2.6</td><td>0.9 ± 0.7</td><td>2.2 ± 2.9</td><td>1.5 ± 0.9</td><td>1.1 ± 0.6</td></tr><tr><td>-0.20</td><td>-0.14</td><td>-0.30</td><td>-0.17</td><td>-0.21</td><td>0.00</td><td>15.3 ± 7.8</td><td>10.2 ± 5.4</td><td>6.4 ± 4.5</td><td>14.3 ± 5.7</td><td>10.0 ± 5.0</td><td>6.4 ± 4.4</td><td>2.6 ± 7.8</td><td>1.2 ± 2.5</td><td>0.9 ± 0.7</td><td>1.3 ± 2.6</td><td>1.0 ± 0.5</td><td>0.9 ± 0.3</td></tr><tr><td>-0.19</td><td>-0.17</td><td>-0.20</td><td>-0.19</td><td>-0.33</td><td>-0.13</td><td>15.2 ± 7.7</td><td>10.2 ± 5.4</td><td>6.4 ± 4.5</td><td>14.3 ± 5.6</td><td>10.0 ± 5.0</td><td>6.4 ± 4.4</td><td>2.6 ± 7.6</td><td>1.2 ± 2.4</td><td>0.9 ± 0.6</td><td>1.2 ± 2.3</td><td>1.0 ± 0.4</td><td>0.9 ± 0.3</td></tr><tr><td rowspan="3">CCSDS(128,64)</td><td>-0.19</td><td>-0.58</td><td>-0.40</td><td>-0.31</td><td>-0.36</td><td>0.52</td><td>20.8 ± 11.4</td><td>13.1 ± 6.3</td><td>3.8 ± 4.9</td><td>18.2 ± 6.6</td><td>12.8 ± 5.4</td><td>8.4 ± 4.9</td><td>4.7 ± 13.4</td><td>1.4 ± 4.1</td><td>1.0 ± 0.8</td><td>1.7 ± 4.7</td><td>1.1 ± 0.7</td><td>0.3 ± 0.3</td></tr><tr><td>-0.21</td><td>-0.30</td><td>-0.60</td><td>-0.23</td><td>-0.42</td><td>0.31</td><td>20.6 ± 11.2</td><td>13.1 ± 6.2</td><td>2.8 ± 4.9</td><td>18.1 ± 6.4</td><td>12.8 ± 5.4</td><td>8.4 ± 4.9</td><td>4.4 ± 12.8</td><td>1.3 ± 3.1</td><td>0.8 ± 0.7</td><td>1.6 ± 4.1</td><td>1.1 ± 0.6</td><td>0.3 ± 0.3</td></tr><tr><td>-0.27</td><td>-0.42</td><td>-0.40</td><td>-0.26</td><td>-0.11</td><td>0.37</td><td>20.6 ± 11.0</td><td>13.1 ± 6.2</td><td>8.4 ± 4.9</td><td>18.1 ± 6.3</td><td>12.8 ± 5.4</td><td>8.4 ± 4.9</td><td>4.2 ± 12.3</td><td>1.3 ± 3.4</td><td>1.0 ± 0.6</td><td>1.6 ± 3.7</td><td>1.1 ± 0.5</td><td>1.0 ± 0.3</td></tr><tr><td rowspan="3">BCH(63,36)</td><td>-0.01</td><td>0.02</td><td>-0.04</td><td>0.01</td><td>0.00</td><td>0.09</td><td>12.6 ± 8.0</td><td>7.8 ± 6.7</td><td>4.3 ± 5.1</td><td>11.9 ± 7.5</td><td>7.6 ± 6.4</td><td>4.3 ± 5.0</td><td>3.7 ± 7.2</td><td>1.4 ± 3.4</td><td>0.7 ± 1.3</td><td>4.3 ± 6.8</td><td>2.0 ± 3.6</td><td>1.0 ± 1.8</td></tr><tr><td>-0.01</td><td>0.02</td><td>0.13</td><td>-0.11</td><td>-0.05</td><td>0.03</td><td>12.4 ± 7.9</td><td>7.7 ± 6.6</td><td>4.3 ± 5.1</td><td>11.8 ± 7.4</td><td>7.5 ± 6.3</td><td>4.3 ± 5.0</td><td>3.3 ± 6.8</td><td>1.3 ± 3.1</td><td>0.7 ± 1.1</td><td>1.6 ± 5.6</td><td>1.1 ± 2.4</td><td>0.7 ± 0.8</td></tr><tr><td>0.04</td><td>0.16</td><td>0.23</td><td>-0.11</td><td>-0.02</td><td>0.04</td><td>12.1 ± 7.6</td><td>7.6 ± 6.5</td><td>4.3 ± 5.0</td><td>11.7 ± 7.3</td><td>7.5 ± 6.3</td><td>4.3 ± 5.0</td><td>2.5 ± 5.5</td><td>1.1 ± 2.2</td><td>0.7 ± 0.8</td><td>2.5 ± 5.6</td><td>1.1 ± 2.3</td><td>0.7 ± 0.8</td></tr><tr><td rowspan="3">BCH(63,51)</td><td>0.04</td><td>0.28</td><td>0.94</td><td>0.09</td><td>0.42</td><td>1.16</td><td>4.8 ± 4.1</td><td>2.6 ± 3.4</td><td>1.2 ± 2.4</td><td>4.7 ± 4.0</td><td>2.6 ± 3.4</td><td>1.2 ± 2.4</td><td>1.8 ± 2.9</td><td>0.7 ± 1.3</td><td>0.3 ± 0.6</td><td>1.6 ± 2.6</td><td>0.7 ± 1.0</td><td>0.3 ± 0.6</td></tr><tr><td>0.06</td><td>0.31</td><td>1.13</td><td>0.06</td><td>0.28</td><td>1.19</td><td>4.8 ± 4.1</td><td>2.6 ± 3.4</td><td>1.2 ± 2.4</td><td>4.7 ± 4.0</td><td>2.6 ± 3.4</td><td>1.2 ± 2.4</td><td>1.6 ± 2.7</td><td>0.7 ± 1.2</td><td>0.3 ± 0.5</td><td>1.3 ± 2.1</td><td>0.6 ± 1.0</td><td>0.3 ± 0.5</td></tr><tr><td>0.04</td><td>0.34</td><td>1.02</td><td>0.02</td><td>0.34</td><td>1.04</td><td>4.8 ± 4.1</td><td>2.6 ± 3.4</td><td>1.2 ± 2.4</td><td>4.7 ± 4.0</td><td>2.6 ± 3.4</td><td>1.2 ± 2.3</td><td>1.6 ± 2.6</td><td>0.7 ± 1.1</td><td>0.3 ± 0.5</td><td>1.4 ± 2.3</td><td>0.6 ± 1.0</td><td>0.3 ± 0.5</td></tr></table>

![](images/170a7099c0dccad5062f8393c1d9c5d07c3a61b1a540b438a64aabdc01c0cc79.jpg)  
(a)

![](images/242534d3a6ba901e34b0d476c49daf043d4cc08803f66371ff1ddfb225143e4a.jpg)  
(b)

![](images/184a02b0c4127c7dca7a5d8011f759a79790abb2e458fdf727c9472bbbd124e2.jpg)  
(c)

![](images/9cbf9f3ef6446beb3875ca8b3d3107f9c31f8201ad5244abd4a23b1258b0b87d.jpg)  
Figure 4: BER comparison between the ECCT  $N = 6$ ,  $d = 32$  and the proposed DDECCT, for the Rayleigh fading channel for (a) Polar(64,32), (b) BCH(63,36), and (c) LDPC(49,24) codes.  
(a)

![](images/667ed12c47a3fd2ff519ee4091a5f5c36c3a591aeea13f5b89c53f9f39b82741.jpg)  
(b)

![](images/4c1c24871c784ba1bc58b468d753b3ba228a0cedeeb8bf2e75d5a250f4ab0e6f.jpg)  
Figure 5: Performance comparison between ECCT and DDECCT  $N = 6, d = 32$  for various values of normalized SNR for (a) Polar(512,384), (b) LDPC (529,440) codes.  
Figure 6: BER vs the number of iterations (up to  $n - k$ ) for regular and line search reverse diffusion.

scheme is pertinent in comparing our results with ECCT since they have similar architectures and capacities. Our method's overhead over ECCT is by a factor of the number of diffusion steps  $L$ , i.e., a complexity of  $\mathcal{O}(LN(d^2 (2n - k) + hd))$ , where  $h$  is the complexity of the self-attention module.

Note that LDPC codes are designed specifically for BP-based decoding (Richardson et al., 2001).

The results are reported as bit error rates (BER) for different normalized SNR values  $(EbN0)$ . We follow the testing benchmark of (Nachmani & Wolf, 2019; Choukroun & Wolf, 2022). During testing, our decoder decodes at least  $10^{5}$  random codewords, to obtain at least 500 frames with errors at each SNR value. All baseline results were obtained from the corresponding papers.

The results are reported in Tab. 1, where we present the negative natural logarithm of the BER. For each code, we present the results of the BP-based competing methods for 5 and 50 iterations (first and second rows), corresponding to a neural network with 10 and 100 layers, respectively. As in (Choukroun & Wolf, 2022), our framework's performance with Line Search (LS) as described in Section 4.4 is evaluated for six different architectures, with  $N = \{2,6\}$  and  $d = \{32,64,128\}$ , respectively (first to third rows).

As can be seen, our approach outperforms the current SOTA results (obtained by ECCT) by extremely large margins on several codes, at a fraction of the capacity. Especially for shallow models, the difference can be an order of magnitude. Performance is closer with short high-rate codes, for which ECCT performance is already very high. We present in Figure 5 the performance of the proposed DDECCT on larger codes. As can be seen, DDECCT can learn to efficiently decode larger codes and outperforms ECCT.

We present in Table 2 the difference in accuracy  $\Delta$  between the line search procedure and the regular reverse diffusion. We also present convergence statistics (mean and standard deviation of the number of iterations) for the regular reverse diffusion and the line search procedure. The full table with the statistics for all of the codes is given in Appendix E. Evidently, the line search procedure enables extremely fast convergence, requiring as little as one iteration for high SNR. Note that we measure the number of iterations required to reach a syndrome of zero failed checks. We do not apply early stopping to the decoding, which could reduce the average number of iterations even further if the decoder stagnates and does not converge to zero syndrome.

# 5.1 NON-GAUSSIAN CHANNEL

We test our framework on a non-Gaussian Rayleigh fading channel, which is often used for simulating the propagation environment of a signal, e.g., for wireless devices. In this fading model, the transmission of the codeword  $x \in \{0,1\}^n$  is defined as  $y = hx_{s} + z$ , where  $h$  is an  $n$ -dimensional i.i.d. Rayleigh-distributed vector with a scale parameter  $\alpha$ , and  $z \sim \mathcal{N}(0,\sigma^2 I_n)$ .

In our simulations, we assume a high scale  $\alpha = 1$  in order to easily compare and reproduce the results, while the level of Gaussian noise and the testing procedure remain the same as described in the paper. The overall variance of the transmitted codeword  $y$  in the Rayleigh channel is roughly twice the AWGN's on the tested SNR range. The results are presented in Figure 4. As can be observed, our method is still able to learn to decode, even under these very noisy fading channels.

# 5.2 BER EVOLUTION THROUGH ITERATION/TIME

We illustrate in Figure 2 the denoising process for several codes. We show how the BER decreases with time for the regular proposed method and the augmented line search procedure. We can observe the very fast convergence of the line search approach. We further provide in Appendix D the performance of the proposed framework for one, two and three iteration steps. We can see that LS enables outperforming the original ECCT, even with one step only.

# 6 CONCLUSIONS

We present a novel denoising diffusion method for the decoding of algebraic block codes. It is based on an adapted diffusion process that simulates the channel corruption we wish to reverse. The method makes use of the syndrome as a conditioning signal and employs a line-search procedure to control the step size. Since it inherits the iterative nature of the underlying process, both training and deployment are extremely efficient. Even with very low-capacity networks, the proposed approach outperforms existing neural decoders by sizable margins for a broad range of code families.

# REFERENCES

Erdal Arikan. Channel polarization: A method for constructing capacity-achieving codes. In 2008 IEEE International Symposium on Information Theory, pp. 1173-1177. IEEE, 2008.  
Amir Bennatan, Yoni Choukroun, and Pavel Kisilev. Deep learning for decoding of linear codes-a syndrome-based approach. In 2018 IEEE International Symposium on Information Theory (ISIT), pp. 1595-1599. IEEE, 2018.  
Raj Chandra Bose and Dwijendra K Ray-Chaudhuri. On a class of error correcting binary group codes. Information and control, 3(1):68-79, 1960.  
Sebastian Cammerer, Tobias Gruber, Jakob Hoydis, and Stephan ten Brink. Scaling deep learning-based decoding of polar codes via partitioning. In GLOBECOM 2017-2017 IEEE Global Communications Conference, pp. 1-6. IEEE, 2017.  
Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. arXiv preprint arXiv:2009.00713, 2020a.  
Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. arXiv:2009.00713, 2020b.  
Yoni Choukroun and Lior Wolf. Error correction code transformer. Advances in Neural Information Processing Systems (NeurIPS), 2022.  
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34, 2021.  
Robert Gallager. Low-density parity-check codes. IRE Transactions on information theory, 8(1): 21-28, 1962.  
Tobias Gruber, Sebastian Cammerer, Jakob Hoydis, and Stephan ten Brink. On deep learning-based channel decoding. In 2017 51st Annual Conference on Information Sciences and Systems (CISS), pp. 1-6. IEEE, 2017.  
Michael Helmling, Stefan Scholl, Florian Gensheimer, Tobias Dietz, Kira Kraft, Stefan Ruzika, and Norbert Wehn. Database of Channel Codes and ML Simulation Results. www.uni-kl.de/channel-codes, 2019.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv:2006.11239, 2020a.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv preprint arXiv:2006.11239, 2020b.  
Aapo Hyvärinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005.  
Mohamed Ibnkahla. Applications of neural networks to digital communications-a survey. Signal processing, 80(7):1185-1215, 2000.  
Hyeji Kim, Yihan Jiang, Ranvir Rana, Sreeram Kannan, Sewoong Oh, and Pramod Viswanath. Communication algorithms via deep learning. In Sixth International Conference on Learning Representations (ICLR), 2018.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. arXiv:2009.09761, 2020.  
Eliya Nachmani and Lior Wolf. Hyper-graph-network decoders for block codes. In Advances in Neural Information Processing Systems, pp. 2326-2336, 2019.

Eliya Nachmani and Lior Wolf. Autoregressive belief propagation for decoding block codes. arXiv preprint arXiv:2103.11780, 2021.  
Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. arXiv:2102.09672, 2021a.  
Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pp. 8162-8171. PMLR, 2021b.  
Jorge Nocedal and Stephen J. Wright. Line Search Methods, pp. 30-65. Springer New York, New York, NY, 2006. ISBN 978-0-387-40065-5. doi: 10.1007/978-0-387-40065-5_3. URL https://doi.org/10.1007/978-0-387-40065-5_3.  
Timothy J O'Shea and Jakob Hoydis. An introduction to machine learning communications systems. arXiv preprint arXiv:1702.00832, 2017.  
Judea Pearl. *Probabilistic reasoning in intelligent systems: networks of plausible inference*. Morgan Kaufmann, 1988.  
Thomas J Richardson, Mohammad Amin Shokrollahi, and Rüdiger L Urbanke. Design of capacity-approaching irregular low-density parity-check codes. IEEE transactions on information theory, 47(2):619-637, 2001.  
Chitwan Sahara, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. arXiv:arXiv:2104.07636, 2021.  
Robin San-Roman, Eliya Nachmani, and Lior Wolf. Noise estimation for generative diffusion models. arXiv preprint arXiv:2104.02600, 2021.  
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pp. 2256-2265. PMLR, 2015.  
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, 2020a.  
Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. arXiv preprint arXiv:1907.05600, 2019.  
Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv:2011.13456, 2020b.  
Max Welling and Yee W Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pp. 681-688. Citeseer, 2011.  
Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer architecture. arXiv preprint arXiv:2002.04745, 2020.
