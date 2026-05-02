# STICK-BRACKING VARIATIONAL AUTOENCODERS

# Eric Nalisnick

Department of Computer Science  
University of Calforina, Irvine  
enalisni@uci.edu

# Padhraic Smyth

Department of Computer Science  
University of Calforina, Irvine  
smyth@ics.uci.edu

# ABSTRACT

We extend Stochastic Gradient Variational Bayes (17; 27) to perform posterior inference for the weights of Stick-Breaking processes. This development allows us to define a Stick-Breaking Variational Autoencoder (SB-VAE), a Bayesian nonparametric version of the variational autoencoder (17) that has a latent representation with stochastic dimensionality. We experimentally demonstrate that the SB-VAE, and a semi-supervised variant, learn highly discriminative latent representations that often outperform the Gaussian VAE's.

# 1 INTRODUCTION

Deep generative models trained via Stochastic Gradient Variational Bayes (SGVB) (17; 27) efficiently couple the expressiveness of deep neural networks with the robustness to uncertainty of probabilistic latent variables. This combination has lead to their success in tasks ranging from image generation (9; 26) to semi-supervised learning (16; 21) to language modeling (2). Various extensions to SGVB have been proposed (3; 21; 31), but one conspicuous absence is an extension to Bayesian nonparametric processes. Using SGVB to perform inference for nonparametric distributions is quite attractive. For instance, SGVB allows for a broad class of non-conjugate approximate posteriors and thus has the potential to expand Bayesian nonparametric models beyond the exponential family distributions to which they are usually confined. Moreover, coupling nonparametric processes with neural network inference models equips the networks with automatic model selection properties such as a self-determined width, which we explore in this paper.

We make progress on this problem by first describing how to use SGVB for posterior inference for the weights of Stick-Breaking processes (12). This is not a straightforward task as the Beta distribution, the natural choice for an approximate posterior, does not have the differentiable non-centered parametrization SGVB requires. We bypass this obstacle by using the little-known Kumaraswamy distribution (19).

Using the Kumaraswamy as an approximate posterior, we then reformulate two popular deep generative models—the Variational Autoencoder (17) and its semi-supervised variant (model M2 from (16))—into their nonparametric analogs. These models can be thought of as having an infinitely wide hidden layer, using as many stick segments (latent variables) as the data requires. We experimentally show that, for datasets of natural images, stick-breaking priors improve upon previously proposed deep generative models by having a latent representation that better preserves class boundaries and provides beneficial regularization for semi-supervised learning.

# 2 BACKGROUND

We begin by reviewing the relevant background material on Variational Autoencoders (17), Stochastic Gradient Variational Bayes (also known as Stochastic Backpropagation) (17; 27), and Stick-Breaking Processes (12).

# 2.1 VARIATIONAL AUTOENCODERS

A Variational Autoencoder (VAE) is model comprised of two multilayer perceptrons: one acts as a density network (22) mapping a latent variable  $\mathbf{z}_i$  to an observed datapoint  $\mathbf{x}_i$ , and the other acts

as an inference model (32) performing the reverse mapping from  $\mathbf{x}_i$  to  $\mathbf{z}_i$ . Together the two form a computational pipeline that resembles an unsupervised autoencoder (10). The generative process can be written mathematically as

$$
\mathbf {z} _ {i} \sim p (\mathbf {z}), \mathbf {x} _ {i} \sim p _ {\theta} (\mathbf {x} | \mathbf {z} _ {i}) \tag {1}
$$

where  $p(\mathbf{z})$  is the prior and  $p_{\theta}(\mathbf{x}|\mathbf{z}_i)$  is the density network with parameters  $\theta$ . The approximate posterior of this generative process, call it  $q_{\phi}(\mathbf{z}|\mathbf{x}_i)$ , is then parametrized by the inference network (with parameters  $\phi$ ). Previous work (17; 25; 3; 20) has always chosen the prior  $p(\mathbf{z})$  and variational posterior to be Gaussian.

# 2.2 STOCHASTIC GRADIENT VARIATIONAL BAYES

The VAE's generative and variational parameters are estimated by Stochastic Gradient Variational Bayes (SGVB). SGVB is distinguished from classical variational bayes by its use of differentiable Monte Carlo expectations. To elaborate, consider SGVB's approximation of the usual evidence lowerbound (ELBO) (13):

$$
\tilde {\mathcal {L}} (\boldsymbol {\theta}, \boldsymbol {\phi}; \mathbf {x} _ {i}) = \frac {1}{S} \sum_ {s = 1} ^ {S} \log p _ {\boldsymbol {\theta}} (\mathbf {x} _ {i} | \hat {\mathbf {z}} _ {i, s}) - K L (q _ {\boldsymbol {\phi}} (\mathbf {z} | \mathbf {x} _ {i}) | | p (\mathbf {z})) \tag {2}
$$

for  $S$  samples of  $\mathbf{z}_i$ . An essential requirement of SGVB is that the latent variable be represented in a differentiable, non-centered parametrization (DNCP) (23; 15); this is what allows the gradients to be taken through the MC expectation, i.e.:

$$
\frac {\partial}{\partial \boldsymbol {\phi}} \sum_ {s = 1} ^ {S} \log p _ {\boldsymbol {\theta}} (\mathbf {x} _ {i} | \hat {\mathbf {z}} _ {i, s}) = \sum_ {s = 1} ^ {S} \frac {\partial}{\partial \hat {\mathbf {z}} _ {i , s}} \log p _ {\boldsymbol {\theta}} (\mathbf {x} _ {i} | \hat {\mathbf {z}} _ {i, s}) \frac {\partial \hat {\mathbf {z}} _ {i , s}}{\partial \boldsymbol {\phi}}.
$$

In other words,  $\mathbf{z}$  must have a functional form that deterministically exposes the variational distribution's parameters and allows the randomness to come from draws from some fixed distribution. Location-scale representations and inverse cumulative distribution functions are two examples of DNCPs. For instance, the VAE's Gaussian latent variable (with diagonal covariance matrix) is represented as  $\hat{\mathbf{z}}_i = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$  where  $\boldsymbol{\epsilon} \sim \mathrm{N}(\mathbf{0}, \mathbb{1})$ .

# 2.3 STICK-BRACKING PROCESSES

Lastly, we define stick-breaking processes with the ultimate goal of using their weights for the VAE's prior  $p(\mathbf{z})$ . A random measure is referred to as a stick-breaking prior (12) if it is of the form  $G(\cdot) = \sum_{k=1}^{\infty} \pi_k \delta_{\zeta_k}$  where  $\delta_{\zeta_k}$  is a discrete measure concentrated at  $\zeta_k \sim G_0$ , a draw from the base distribution  $G_0$  (12). The  $\pi_k$ s are random weights independent of  $G_0$ , chosen such that  $0 \leq \pi_k \leq 1$ , and  $\sum_k \pi_k = 1$  almost surely. SBPs have been termed as such because of their constructive definition known as the stick-breaking process (33). Mathematically, this definition implies that the weights can be drawn according to the following iterative procedure:

$$
\pi_ {k} = \left\{ \begin{array}{l} v _ {1} \text {i f} k = 1 \\ v _ {k} \prod_ {j <   k} \left(1 - v _ {j}\right) \text {f o r} k > 1 \end{array} \right. \tag {3}
$$

where  $\nu_{k} \sim \mathrm{Beta}(\alpha, \beta)$ . When  $\nu_{k} \sim \mathrm{Beta}(1, \alpha_{0})$ , then we have the stick-breaking construction for the Dirichlet Process (6). In this case, the name for the joint distribution over the infinite sequence of stick-breaking weights is the Griffiths, Engen and McCloskey distribution (24) with concentration parameter  $\alpha_{0}$ :  $(\pi_{1}, \pi_{2}, \ldots) \sim \mathrm{GEM}(\alpha_{0})$ .

# 3 SGVB FOR GEM RANDOM VARIABLES

Having covered the relevant background material, we now discuss the first contribution of this paper, using Stochastic Gradient Variational Bayes for the weights of a stick-breaking process. Unfortunately, performing inference for the random measure  $G(\cdot)$  is an open problem that we leave to future work. Instead we focus on performing inference for just the series of stick-breaking weights, which we will refer to as GEM random variables after their joint distribution.

As the stick-breaking weights are just a composition of Beta random variables, the main obstacle then is finding an approximate posterior for the Beta distribution with a non-centered parametrization (DNCP). When using standard Variational Bayes for the Dirichlet Process, Betas are chosen to be the variational distribution (1), but unfortunately, the Beta does not have a DNCP and thus we are forced to find an alternative. In this section we discuss three options: composing Gamma random variables, the Kumaraswamy distribution, and modification to a Logit SBP.

# 3.1 COMPOSITION OF GAMMA RANDOM VARIABLES

In the original SGVB paper (17), Kingma & Welling suggest representing the Beta distribution as a composition of Gamma random variables by using the fact  $\nu \sim \mathrm{Beta}(\alpha, \beta)$  can be sampled by drawing Gamma variables  $x \sim \mathrm{Gamma}(\alpha, 1)$ ,  $y \sim \mathrm{Gamma}(\beta, 1)$  and composing them as  $\nu = x / (x + y)$ . However, this representation still does not admit a DNCP as the Gamma distribution does not have one with respect to its shape parameter. D. Knowles (18) suggests that when the shape parameter is near zero, the following asymptotic approximation of the inverse CDF is a suitable DNCP:

$$
F ^ {- 1} (\hat {u}) \approx \frac {(\hat {u} a \Gamma (a)) ^ {\frac {1}{a}}}{b} \tag {4}
$$

for  $\hat{u} \sim \mathrm{Uniform}(0,1)$ , shape parameter  $a$ , and scale parameter  $b$ . This approximation becomes poor as  $a$  increases, however, and Knowles recommends a finite difference approximation of the inverse CDF when  $a \geq 1$ .

# 3.2 THE KUMARASWAMY DISTRIBUTION

Another candidate posterior is the little-known Kumaraswamy distribution (19). It is a two-parameter continuous distribution also on the unit interval; its density function is

$$
\operatorname {K u m a r a s w a m y} (x; a, b) = a b x ^ {a - 1} \left(1 - x ^ {a}\right) ^ {b - 1} \tag {5}
$$

for  $x \in (0,1)$  and  $a, b > 0$ . In fact, if  $a = 1$  or  $b = 1$  or both, the Kumaraswamy and Beta are equivalent, and for equivalent parameter settings, the Kumaraswamy is resembles the Beta albeit has higher entropy. The DNCP we so desire is the Kumaraswamy's closed-form inverse CDF. Samples can be drawn via the inverse transform:

$$
x \sim \left(1 - u ^ {\frac {1}{b}}\right) ^ {\frac {1}{a}} \text {w h e r e} u \sim \operatorname {U n i f o r m} (0, 1). \tag {6}
$$

Not only does the Kumaraswamy make sampling easy, its KL-divergence from the Beta can be closely approximated in closed-form.

# 3.2.1 GAUSS-LOGIT PARAMETRIZATION

Another promising parametrization is inspired by the Probit Stick-Breaking Process (29). In a two-step process, we can draw a Gaussian and then use a squashing function to map it on  $(0,1)$ :

$$
\hat {v} _ {k} = g \left(\mu_ {k} + \sigma_ {k} \epsilon\right) \tag {7}
$$

where  $\epsilon \sim \mathrm{N}(0,1)$ . In the Probit SBP,  $g(\cdot)$  is taken to be the Gaussian CDF, and it is chosen as such for posterior sampling considerations. This choice is impractical for our purposes, however, since the Gaussian CDF does not have a closed form. Instead, we chose the logistic function  $g(x) = 1 / (1 + e^{-x})$ . This parametrization has some downsides, however. The Kumaraswamy distribution can become bimodal, like the Beta, when  $a, b < 1$ , but this Gauss-Logit parametrization is strictly uni-modal.

# 4 STICK-BRACKING VARIATIONAL AUTOENCODERS

We propose the following novel modification to the VAE. Instead of drawing the latent variables from a Gaussian distribution, we draw them from the GEM distribution, making the hidden representation an infinite sequence of stick-breaking weights. Mathematically, this is written as

$$
\pi_ {i} \sim \operatorname {G E M} \left(\alpha_ {0}\right), \mathbf {x} _ {i} \sim p _ {\boldsymbol {\theta}} (\mathbf {x} | \boldsymbol {\pi} _ {i}) \tag {8}
$$

![](images/92d7b6cabefd3e453cc795a2be33254ffe4cffa517fcc19312102217f07ef6b2.jpg)  
(a) Finite Dimensional

![](images/bd10ba6487a5ae1b5556e43f0eb5ba46378bb87699cab7dcf77b84cb3debfb7a.jpg)  
(b) Infinite Dimensional

![](images/fca2d07eb85690adab0e089e3d07834cec9a7efee060ef905c0f57e17f7f5ce9.jpg)  
(c) The Stick-Breaking Variational Autoencoder.  
Figure 1: Subfigures (a) and (b) show the plate diagrams for the relevant latent variable models. Solid lines denote the generative process and dashed the inference model. Subfigure (a) shows the finite dimensional case considered in (17), and (b) shows the infinite dimensional case of our concern. Subfigure (c) shows the feedforward architecture of the Stick-Breaking Autoencoder, which is a neural-network-based parametrization of the graphical model in (b).

where  $\pi_{i}$  is the vector of stick-breaking weights and  $\alpha_0$  is the concentration parameter of the GEM distribution. A linear-time operation is required to compose the stick segments from the sampled fractions  $\nu$ :

$$
\pi_ {i} = \left(\pi_ {i, 1}, \pi_ {i, 2}, \dots , \pi_ {i, K}\right) = \left(v _ {i, 1}, v _ {i, 2} (1 - v _ {i, 1}), \dots , \prod_ {j = 1} ^ {K - 1} (1 - v _ {i, j})\right). \tag {9}
$$

The  $\nu$ 's are sampled via one of the parametrizations described in Section 3. The computation path is summarized in Figure 1 (c) with arrows denoting the direction of feedforward computation. The gray blocks represent any deterministic function that can be trained with gradient descent—i.e. one or more neural network layers. Optimization of our Stick-Breaking Variational Autoencoder (SB-VAE) is done just as for the VAE, by optimizing Equation 2 w.r.t.  $\pmb{\phi}$  and  $\pmb{\theta}$ . The KL divergence term can be computed in closed-form for all three parametrizations under consideration; the Kumaraswamy-to-Beta KL divergence is given in the appendix.

An important detail is that the  $K$ th fraction  $\nu_{i,K}$  is always set to one to ensure the stick segments sum to one. This truncation of the variational posterior does not imply that we are using a finite dimensional prior. As explained in (1), the truncation level is a variational parameter and not part of the prior model specification. Truncation-free posteriors have been proposed, but these methods use split-and-merge steps (11) or collapsed Gibbs sampling, both of which are not applicable to the models we consider. Yet, because SGVB imposes few limitations on the inference model, it is possible to have an untruncated posterior. We attempted to use a truncation-free posterior by adding extra variational parameters in an on-line fashion, initializing new weights if more than 1% of the stick remained unbroken. However, we found this made optimization slower without any increase in performance.

# 5 SEMI-SUPERVISED MODEL

We also perform experiments with the semi-supervised relative of the VAE, the M2 model described in (16). A second latent variable  $y_{i}$  is introduced that represents a class label. Its distribution is the categorical one:  $q_{\phi}(y_i|\mathbf{x}_i) = \mathrm{Cat}(y|g_y(\mathbf{x}_i))$  where  $g_{y}$  is a non-linear function of the inference network. Although  $y$ 's distribution is written as independent of  $\mathbf{z}$ , the two share parameters within the inference network and thus act to regularize one another. We assume the same factorization of the posterior and use the same objectives as in the finite dimensional version (model M2 from (16)). Since  $y_{i}$  is present for some but not all observations, semi-supervised DGMs need to be trained with different objectives depending on whether the label is present or not. If the label is present, following (16) we optimize

$$
\tilde {\mathcal {J}} (\boldsymbol {\theta}, \boldsymbol {\phi}; \mathbf {x} _ {i}, y _ {i}) = \frac {1}{S} \sum_ {s = 1} ^ {S} \log p _ {\boldsymbol {\theta}} \left(\mathbf {x} _ {i} \mid \boldsymbol {\pi} _ {i, s}, y _ {i}\right) - K L \left(q _ {\boldsymbol {\phi}} \left(\boldsymbol {\pi} _ {i} \mid \mathbf {x} _ {i}\right) \right\lVert p \left(\boldsymbol {\pi} _ {i}; \boldsymbol {\alpha} _ {0}\right)) + \log q _ {\boldsymbol {\phi}} \left(y _ {i} \mid \mathbf {x} _ {i}\right) \tag {10}
$$

where  $\log q_{\phi}(y_i|\mathbf{x}_i)$  is the log-likelihood of the label. And if the label is missing, we optimize

$$
\begin{array}{l} \tilde {\mathcal {J}} (\boldsymbol {\theta}, \boldsymbol {\phi}; \mathbf {x} _ {i}) = \frac {1}{S} \sum_ {s = 1} ^ {S} \sum_ {y _ {j}} q _ {\boldsymbol {\phi}} \left(y _ {j} \mid \mathbf {x} _ {i}\right) \left[ \log p _ {\boldsymbol {\theta}} \left(\mathbf {x} _ {i} \mid \boldsymbol {\pi} _ {i, s}, y _ {j}\right) \right] + \mathbb {H} \left[ q _ {\boldsymbol {\phi}} (y \mid \mathbf {x} _ {i}) \right] \tag {11} \\ - K L \left(q _ {\phi} \left(\pi_ {i} | \mathbf {x} _ {i}\right) \| p \left(\pi_ {i}; \alpha_ {0}\right)\right) \\ \end{array}
$$

where  $\mathbb{H}[q_{\phi}(y_i|\mathbf{x}_i)]$  is the entropy of  $y$ 's variational distribution.

# 6 RELATED WORK

To the best of our knowledge, neither SGVB nor any of the other recently proposed amortized VI methods (15; 25; 27; 34) have been used in conjunction with BNP priors. There has been work on using nonparametric posterior approximations—see the Variational Gaussian Process (34)—but in that work the variational distribution is nonparametric, not the generative model. Moreover, we are aware of no work that uses SGVB for Beta (or Beta-like) random variables<sup>1</sup>.

In regards to the autoencoder implementations we describe, they are closely related to the existing work on representation learning with adaptive latent factors—i.e. where the number of latent dimensions grows as the data necessitates. The best known model of this kind is the infinite binary latent feature model defined by the Indian Buffet Process (7); but its discrete latent variables prevent this model from admitting fully differentiable inference. Recent work that is much closer in spirit is the Infinite Restricted Boltzmann Machine (iRBM) (4), which has gradient-based learning, expands its capacity by adding hidden units, and induces a similar ordering on latent factors. The most significant difference between our Stick-Breaking Autoencoder and the iRBM is that the latter's nonparametric behavior arises from a particular definition of the energy function of the Gibbs distribution, not from an infinite dimensional Bayesian prior. Lastly, our training procedure bears some resemblance to Nested Dropout (28), which removes all hidden units to the right of some threshold index. The SB-VAE can be seen as performing soft nested dropout since the activations decrease from left to right.

# 7 EXPERIMENTS

Turning to experiments, we start by analyzing the behavior of the three parametrizations of the Stick-Breaking VAE and examining how they compare to the Gaussian VAE. We do this mainly by examining their ability to reconstruct the data (i.e. density estimation) and to preserve class structure. Following the original DGM papers (16; 17; 27), we performed unsupervised and semi-supervised tasks on the following image datasets: Frey Faces $^2$ , MNIST, MNIST+rot, and Street View House Numbers $^3$  (SVHN). MNIST+rot is a dataset we created by combining MNIST and rotated MNIST $^4$  for the purpose of testing the latent representation, under the conjecture that the rotated digits should use more latent variables than the non-rotated ones.

Complete implementation and optimization details can be found in the appendix and code repository<sup>5</sup>. In all experiments, to best isolate the effects of Gaussian vs stick-breaking latent variables, the same architecture and optimization hyperparameters were used for each model. The only difference was in the prior:  $p(\mathbf{z}) = \mathrm{N}(\mathbf{0},\mathbb{1})$  for Gaussian latent variables and  $p(\nu) = \mathrm{Beta}(1,\alpha_0)$  (Dirichlet process) for stick-breaking latent variables. We cross-validated the concentration parameter over the range  $\alpha_0 \in \{1,3,5,8\}$ . The Gaussian model's performance potentially could have been improved by cross validating its prior variance. However, the standard Normal prior is widely used as a default choice (2; 9; 16; 17; 27; 31), and we aim to experimentally demonstrate a stick-breaking prior is a competitive alternative.

![](images/de28bdb490ccb74badae5424f4cc4938f560dde560dea80568ac4f1cdcf95ba9.jpg)  
(a) Gauss VAE

![](images/44a057c018672e1c4d1a6b1b00101c5d45a987cf0a0183ec25753527a5ac15d0.jpg)  
(b) Stick-Breaking VAE  
Figure 2: Sparsity in the latent representation vs sparsity in the decoder network. The Gaussian VAE 'turns off' unused latent dimensions by setting the outgoing weights to zero (in order to dispel the sampled noise). The SB VAE, on the other hand, also has sparse representations but without decay of the associated decoder weights.

# 7.1 UNSUPERVISED

We first performed unsupervised experiments testing each model's ability to recreate the data as well as preserve the class structure (without ever having access to labels). The inference and generative models both contained one hidden layer of 200 units for Frey Faces and 500 units for MNIST and MNIST+rot. For Frey Faces, the Gauss-VAE had a 25 dimensional (factorized) distribution, and we set the truncation level of the SB-VAE also to  $K = 25$ , so the SB-VAE could use only as many latent variables as the Gauss-VAE. For the MNIST datasets, latent dimensionality/truncation-level was set at 50. Cross-validation chose  $\alpha_0 = 1$  for Frey Faces and  $\alpha_0 = 5$  for both MNISTs.

Density Estimation. In order to show each model's optimization progress, Figure 3 (a), (b), and (c) report test expected reconstruction error (i.e. the first term in the ELBO) vs training progress (epochs) for Frey Faces, MNIST, and MNIST+rot respectively. Optimization proceeds much the same in both models except that the SB-VAE learns at a slightly slower pace for all parametrizations. This is not too surprising since the recursive definition of the latent variables likely causes coupled gradients.

We compare the final converged models in Table 1, reporting the marginal likelihood of each model via the MC approximation  $\log p(\mathbf{x}_i) \approx \log \frac{1}{S} \sum_s p(\mathbf{x}_i | \hat{\mathbf{z}}_{i,s}) p(\hat{\mathbf{z}}_{i,s}) / q(\hat{\mathbf{z}}_{i,s})$  using 100 samples. The Gaussian VAE has a better likelihood than all stick-breaking implementations ( $\sim 93$  vs  $\sim 98$ ). Between the stick-breaking parametrizations, the Kumaraswamy outperforms both the Gamma and Gauss-Logit on both datasets, which is not surprising given the others' flaws (i.e. the Gamma is approximate, the Gauss-Logit is restricted). Given this result, we used the Kumaraswamy parametrization for all subsequently reported experiments.

We also investigated whether the SB-VAE is using its adaptive capacity in the manner we expect, i.e., the SB-VAE should use a larger latent dimensionality for the rotated images in MNIST+rot than it does for the non-rotated ones. We examined if this is the case by tracking how many 'breaks' it took the model to deconstruct  $99\%$  of the stick. On average, the rotated images in the training set were represented by 28.7 dimensions and the non-rotated by 27.4. Furthermore, the rotated images used more latent variables in eight out of ten classes. Although the difference is not as large as we were expecting, it is statistically significant. Moreover, the difference is made smaller by the non-rotated one digits, which use 32 dimensions on average, the most for any class. The non-rotated average decreases to 26.3 when ones are excluded.

Lastly, Figure 3 (g) shows MNIST digits drawn from the SB-VAE by sampling from the prior-i.e.  $\nu_{k}\sim \mathrm{Beta}(1,5)$ . Samples using all fifty dimensions of the truncated posterior are shown in the bottom block. Samples from Dirichlets constrained to a subset of the dimensions are shown in the two columns in order to test that the latent features are concentrating onto lower-dimensional simplices. This is indeed the case: adding a latent variable results in markedly different but still coherent samples. For instance, the second and third dimensions seem to capture the 7-class, the fourth and fifth the 6-class, and the eighth the 5-class. The seventh dimension seems to model notably thick digits.

Discriminative Qualities. The discriminative qualities of the models' latent spaces are assessed by running a k-Nearest Neighbors classifier on (sampled) MNIST latent variables. Results are shown in the table in Figure 3 (f). The SB-VAE exhibits conspicuously better performance at all choices of  $k$ , which suggests that although the Gauss-VAE converges to a better likelihood, the SB-VAE's

<table><tr><td rowspan="2">Model</td><td colspan="2">- log p(xi)</td></tr><tr><td>MNIST</td><td>MNIST+rot</td></tr><tr><td>Gauss VAE</td><td>93.93</td><td>108.40</td></tr><tr><td>Kumar-SB VAE</td><td>98.01</td><td>112.33</td></tr><tr><td>Logit-SB VAE</td><td>99.48</td><td>114.09</td></tr><tr><td>Gamma-SB VAE</td><td>100.74</td><td>113.22</td></tr></table>

Table 1: Marginal likelihood results (estimated) for Gaussian VAE and the three parametrizations of the Stick-Breaking VAE.

latent space better captures class structure. The discriminative qualities of the SB-VAE's latent space are further supported by Figures 3 (d) and (e). t-SNE was used to embed the Gaussian (e) and stick-breaking (d) latent MNIST representations into two dimensions. Digit classes (denoted by color) in the stick-breaking latent space are clustered with noticeably more cohesion and separation.

Combating Decoder Pruning. The 'component collapsing' behavior of the variational autoencoder has been well noted (5; 21): the model will set to zero the outgoing weights of latent variables that remain near the prior. Figure 2 (a) depicts this phenomenon for the Gauss-VAE by plotting the KL divergence from the prior and outgoing decoder weight norm for each latent dimension. We see the weights are only nonzero in the dimensions in which there is posterior deviation. Ostensibly the model receives only sampling noise from the dimensions that remain at the prior, and setting the decoder weights to zero quells this variance. While the behavior of the Gauss VAE is not necessarily improper, all examples are restricted to pass through the same latent variables. A sparse-coded representation—one having few active components per example (like the Gauss-VAE) but diversity of activations across examples—would likely be better.

We compare the activation patterns against the sparsity of the decoder for the SB-VAE in Figure 2 (b). Since KL-divergence doesn't directly correspond to sparsity in stick-breaking latent variables like it does for Gaussian ones, the black lines denote the average activation value per dimension. Similarly to (a), blue lines denoted the decoder weight norms, but they had to be down-scaled by a factor of 100 so they could be visualized on the same plot. The SB-VAE does not seem to have any component collapsing, which is not too surprising since the model can set latent variables to zero to deactivate decoder weights without being in the heart of the prior. We conjecture that this increased capacity is the reason stick-breaking variables demonstrate better discriminative performance in many of our experiments.

# 7.2 SEMI-SUPERVISED

We also performed semi-supervised classification, replicating and extending the experiments in the original semi-supervised DGMs paper (16). We used the MNIST, MNIST+rot, and SVHN datasets and reduced the number of labeled training examples to  $10\%$ ,  $5\%$ , and  $1\%$  of the total training set size. Labels were removed completely at random and as a result, class imbalance was all but certainly introduced. Similarly to the unsupervised setting, we compared DGMs with stick-breaking (SB-DGM) and Gaussian (Gauss-DGM) latent variables against one another and a baseline k-Nearest Neighbors classifier  $(k=5)$ . We used 50 for the latent variable dimensionality / truncation level. The MNIST networks use one hidden layer of 500 hidden units. The MNIST+rot and SVHN networks use four hidden layers of 500 units in each. The last three hidden layers have identity function skip-connections. Cross-validation chose  $\alpha_0 = 5$  for MNISTs and  $\alpha_0 = 8$  for SVHN.

Quantitative Evaluation. Table 2 shows percent error on a test set when training with the specified percentage of labeled examples. We see the the SB-DGM performs markedly better across almost all experiments. The Gauss-DGM achieves a superior error rate only on the easiest tasks: MNIST with  $10\%$  and  $5\%$  of the data labeled.

# 8 CONCLUSIONS

We have described how to employ the Kumaraswamy distribution to extend Stochastic Gradient Variational Bayes to the weights of stick-breaking Bayesian nonparametric priors. Using this

![](images/8d2fef41b7c9b4cfcc35e5ec3ac16afbf297cb2a97636f04a1f71a7f971a51a0.jpg)  
(a) Frey Faces

![](images/c3fe53ad0c2b34466c4fd8bdf1c4fc556876d0a49e476aab13529488e087fc6e.jpg)  
(b) MNIST

![](images/9bb9efd4036e99cb004fbd5bdcf336ac007c77d6799c198f474eb6b0a2f7deaf.jpg)  
(c) MNIST+rot

![](images/415b8ff61594ad00cfe4ec94f46ca30003f10f49df1d5187026034d6a3eaec66.jpg)  
(d) MNIST SB-VAE

![](images/f4e0af1250d8cd6e717686e5127dd27d5d600fb8e7933805a238474a31014411.jpg)  
(e) MNIST Gauss-VAE

<table><tr><td></td><td>k=3</td><td>k=5</td><td>k=10</td></tr><tr><td>SB-VAE</td><td>9.34</td><td>8.65</td><td>8.90</td></tr><tr><td>Gauss-VAE</td><td>28.4</td><td>20.96</td><td>15.33</td></tr><tr><td>Raw Pixels</td><td>2.95</td><td>3.12</td><td>3.35</td></tr></table>

![](images/c2e41ad8ef9ae332f55975f495aeedaad8cfeb5fd6e77ac7ccf04abec0efe403.jpg)  
(g) SB-VAE: MNIST Samples Drawn from Prior  
(h) Gauss-VAE: MNIST Samples Drawn from Prior  
Figure 3: Subfigure (a) shows test (expected) reconstruction error vs training epoch for the SB-VAE and G-VAE on the Frey Faces dataset, subfigure (b) shows the same quantities for the same models on the MNIST dataset, and subfigure (c) shows the same quantities for the same models on the MNIST+rot dataset. Subfigures (d) and (e) show t-SNE projections of the latent representations learned by the SB-VAE and Gauss-VAE respectively. Subfigure (f) shows results of a kNN classifier trained on the latent representations produced by each model. Subfigure (g) depicts samples from the SB-VAE trained on MNIST. We show the ordered, factored nature of the latent variables by sampling from Dirichlet's of increasing dimensionality. Subfigure (h) depicts samples from the Gauss-VAE trained on MNIST.

![](images/c2090e24bb4a33f7a28fec8d2697833b02009800388ce235f42003482a64c252.jpg)

(f) MNIST: kNN on latent space  

<table><tr><td rowspan="2"></td><td colspan="3">MNIST (N=45,000)</td><td colspan="3">MNIST+rot (N=70,000)</td><td colspan="3">SVHN (N=65,000)</td></tr><tr><td>10%</td><td>5%</td><td>1%</td><td>10%</td><td>5%</td><td>1%</td><td>10%</td><td>5%</td><td>1%</td></tr><tr><td>SB-DGM</td><td>4.86±.14</td><td>5.29±.39</td><td>7.34±.47</td><td>11.78±.39</td><td>14.27±.58</td><td>27.67±1.39</td><td>32.08±4.00</td><td>37.07±5.22</td><td>61.37±3.60</td></tr><tr><td>Gauss-DGM</td><td>3.95±.15</td><td>4.74±.43</td><td>11.55±2.28</td><td>21.78±.73</td><td>27.72±.69</td><td>38.13±.95</td><td>36.08±1.49</td><td>48.75±1.47</td><td>69.58±1.64</td></tr><tr><td>kNN</td><td>6.13±.13</td><td>7.66±.10</td><td>15.27±.76</td><td>18.41±.01</td><td>23.43±.01</td><td>37.98±.01</td><td>64.81±.34</td><td>68.94±.47</td><td>76.64±.54</td></tr></table>

Table 2: Percent error on three semi-supervised classification tasks with  $10\%$ ,  $5\%$ , and  $1\%$  of labels present for training. Our DGM with stick-breaking latent variables (SB-DGM) is compared with a DGM with Gaussian latent variables (Gauss-DGM), and a k-Nearest Neighbors classifier  $(k = 5)$ .

development we then defined deep generative models with infinite dimensional latent variables and showed that their latent representations are more discriminative than those of the popular Gaussian variant. Moreover, the only extra computational cost is in assembling the stick segments, a linear operation on the order of the truncation size. Not only are the ideas herein immediately useful as presented, we believe they are an important first-step to integrating black box variational inference and Bayesian nonparametrics, resulting in scalable models that have differentiable control of their capacity. In particular, we see applying SGVB to full Dirichlet processes with non-trivial base measures as an interesting next step. Furthermore, differentiable stick-breaking has the potential to increase the dynamism and adaptivity of neural networks, which has been a subject of recent interest (8), in a probabilistically principled way.

# ACKNOWLEDGEMENTS

Many thanks to Marc-Alexandre Côté and Hugo Larochelle for helpful discussions.

# REFERENCES

[1] David M Blei and Michael I Jordan. Variational inference for Dirichlet process mixtures. Bayesian Analysis, 1(1):121-143, 2006.  
[2] Samuel R Bowman, Luke Vilnis, Oriol Vinyals, Andrew M Dai, Rafal Jozefowicz, and Samy Bengio. Generating sentences from a continuous space. CoNLL, 2016.  
[3] Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. International Conference on Learning Representations (ICLR), 2016.  
[4] Marc-Alexandre Côté and Hugo Larochelle. An infinite restricted Boltzmann machine. Neural Computation, 2016.  
[5] Laurent Dinh and Vincent Dumoulin. Training neural Bayesian nets, 2014. Slides from CIFAR NCAP Summer School, August 12–16, University of Toronto, Toronto, ON.  
[6] Thomas S Ferguson. A Bayesian analysis of some nonparametric problems. The annals of statistics, pages 209-230, 1973.  
[7] Zoubin Ghahramani and Thomas L Griffiths. Infinite latent feature models and the Indian buffet process. In Advances in Neural Information Processing Systems, pages 475-482, 2005.  
[8] Alex Graves. Adaptive computation time for recurrent neural networks. arXiv preprint arXiv:1603.08983, 2016.  
[9] Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Rezende, and Daan Wierstra. Draw: A recurrent neural network for image generation. In Proceedings of the 32nd International Conference on Machine Learning, pages 1462-1471, 2015.  
[10] Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. Science, 313(5786):504-507, 2006.  
[11] Michael C Hughes, Dae Il Kim, and Erik B Sudderth. Reliable and scalable variational inference for the hierarchical Dirichlet process. In International Conference on Artificial Intelligence and Statistics, pages 370-378, 2015.  
[12] Hemant Ishwaran and Lancelot F James. Gibbs sampling methods for stick-breaking priors. Journal of the American Statistical Association, 96(453):161-173, 2001.  
[13] Michael I Jordan, Zoubin Ghahramani, Tommi S Jaakkola, and Lawrence K Saul. An introduction to variational methods for graphical models. Machine learning, 37(2):183-233, 1999.  
[14] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[15] Diederik Kingma and Max Welling. Efficient gradient-based inference through transformations between Bayes nets and neural nets. In Proceedings of the 31st International Conference on Machine Learning (ICML-14), pages 1782–1790, 2014.  
[16] Diederik P Kingma, Shakir Mohamed, Danilo Jimenez Rezende, and Max Welling. Semi-supervised learning with deep generative models. In Advances in Neural Information Processing Systems, pages 3581-3589, 2014.  
[17] Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. International Conference on Learning Representations (ICLR), 2014.  
[18] David A Knowles. Stochastic gradient variational bayes for gamma approximating distributions. arXiv preprint arXiv:1509.01631, 2015.  
[19] Ponnambalam Kumaraswamy. A generalized probability density function for double-bounded random processes. Journal of Hydrology, 46(1):79-88, 1980.  
[20] Yingzhen Li and Richard E Turner. Variational inference with renyi divergence. Neural Information Processing Systems (NIPS), 2016.  
[21] Lars Maaløe, Casper Kaae Sønderby, Søren Kaae Sønderby, and Ole Winther. Auxiliary deep generative models. International Conference on Machine Learning (ICML), 2016.  
[22] David JC MacKay and Mark N Gibbs. Density networks. Statistics and neural networks: advances at the interface. Oxford University Press, Oxford, pages 129-144, 1999.  
[23] Omiros Papaspiliopoulos. Non-centered parameterisations for data augmentation and hierarchical models. Lancaster University, 2003.  
[24] Jim Pitman. Combinatorial Stochastic Processes. Number 1875. Springer Science & Business Media, 2006.  
[25] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Proceedings of the 32nd International Conference on Machine Learning (ICML-15), pages 1530–1538, 2015.  
[26] Danilo Jimenez Rezende, Shakir Mohamed, Ivo Danihelka, Karol Gregor, and Daan Wierstra. One-shot generalization in deep generative models. International Conference on Machine Learning (ICML), 2016.  
[27] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In Proceedings of The 31st International Conference on Machine Learning, pages 1278–1286, 2014.  
[28] Oren Rippel, Michael A Gelbart, and Ryan P Adams. Learning ordered representations with nested dropout. In ICML, pages 1746-1754, 2014.  
[29] Abel Rodriguez and David B Dunson. Nonparametric bayesian models through probit stick-breaking processes. Bayesian analysis (Online), 6(1), 2011.  
[30] F. J. R. Ruiz, M. K. Titsias, and D. M. Blei. The Generalized Reparameterization Gradient. Neural Information Processing Systems (NIPS), 2016.  
[31] Tim Salimans, Diederik Kingma, and Max Welling. Markov chain monte carlo and variational inference: Bridging the gap. In Proceedings of the 32nd International Conference on Machine Learning, 2015.  
[32] Tim Salimans, David A Knowles, et al. Fixed-form variational posterior approximation through stochastic linear regression. Bayesian Analysis, 8(4):837-882, 2013.  
[33] Jayaram Sethuraman. A constructive definition of Dirichlet priors. Statistica Sinica, pages 639-650, 1994.  
[34] Dustin Tran, Rajesh Ranganath, and David M Blei. Variational Gaussian process. International Conference on Learning Representations (ICLR), 2016.
