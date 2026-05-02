# FITTING DATA NOISE IN VARIATIONAL AUTOENCODER

Anonymous authors

Paper under double-blind review

# ABSTRACT

Why does variational autoencoder(VAE) suffer from bad reconstruction, and what influence the disentanglement of VAE? This paper tries to address those issues through a noise modelling perspective. On one fold, the paper proposes the adaptive noise learning algorithms of Gaussian noise and mixture Gaussian noise assumption which empirically contributes to a better reconstruction than original VAE noise assumptions. On other fold, several generating factor properties in the idealistic VAE case are discussed and several performance indicators regarding the disentanglement and generating influence are subsequently raised to evaluate the performance of VAE model and to supervise the used factors. Theoretical analysis is reflected in the experiment results.

# 1 INTRODUCTION

From the theory of artificial intelligence perspectives, Variational AutoEncoder(VAE)s, raised by Kingma & Welling (2013) and Rezende et al. (2014), have been attracting much research attention in the recent years due to their powerful human-like abilities in extracting disentangled factors/representation(Bengio et al. (2013)) underlying data in a purely unsupervised manner and generating signals with abundant diversities in a "latent-factor-controllable" way. On the one hand, beyond most of the current machine learning regimes, VAEs are capitalized on its approaching a causal modelling and disentangled generating factor learning capability, which finely simulating human abilities, emphasized by Lake et al. (2016), of the knowledge transferring through shared causes/factors among different tasks/experiences. On the other hand, the generalization capability possessed by VAEs also well comply with the ideal mental imagery mechanism in memory and thinking.

Due to its strong capabilities on latent representation learning, signal reconstruction and new sample generation, VAE and its variants have been widely applied to wide range of applications, including disentangled representations learning of images(Higgins et al. (2016), Kulkarni et al. (2015), Mathieu et al. (2016)) and time series(Fabius & van Amersfoort (2014)), zero/one/few-shot learning(Rezende et al. (2016), Higgins et al. (2017b)) and transfer learning in reinforcement learning(Higgins et al. (2017a)), causal relationships modeling(Louizos et al. (2017)), pixel trajectory predicting(Walker et al. (2016)), joint multi-modal inference learning(Suzuki et al. (2016)), increasing diversity in imitation learning(Wang et al. (2017)), generation with memory(Li et al. (2016)) and etc.

However, the implementations of VAE always ignore the importance of noise on its reconstruction and representation learning ability and the applications of VAE has suffered from the blurred reconstructions/generations and the unstable disentangled quality (Higgins et al. (2016)). In particular, most of them assume a heuristic fixed noise that disables the model to flexible adapt to the real noise. These issues, as results, sometimes make VAE have to be an auxiliary part to only alleviate the generation shortcomings of generative adversarial network(GAN)(Larsen et al. (2015), Wang et al. (2017)).

From the perspective of VAE assumption, noise modelling places an indispensable part. Concretely, on one hand, the major factors are learnt and inferred and the noise enables the optimization. On the other hand, more importantly, since the whole learning procedure is through an unsupervised manner, the prior for major factors and noise modelling together form the core inductive bias of VAE.

![](images/3db020f99b582a41b6c02995317b403e71fe2b1bceb08d9531a4cba7b4850119.jpg)  
(a)

![](images/5db724210c824cc372ebc97faf620150d5153b67de42db0122a3a5dc5fbd7e45.jpg)  
(b)  
Figure 1: The directed graph model of noise modeling VAE. (a) is under gaussian noise assumption. (b) is under mixture of gaussian noise assumption. Solid lines denote the generative model  $p_{model}(z;W_G)p_{model}(x|z;W_G,\sigma)$  in (a) and  $p_{model}(z;W_G)p_{model}(x|z;W_G,\Sigma,\Pi)$  in (b). Dashed black lines denote the variational approximation  $q(z|x;W_E)$  to the intractable posterior  $p_{model}(z|x;W_G,\sigma)$  in (a) or  $p_{model}(z|x;W_G,\Sigma,\Pi)$  in (b).

From the perspective of real implementation, different datasets may have drastically different noise and improper modeling for noise distribution inclines to lead to incorrect model hypothesis and tend to influence the distribution learning consequence. The superiority brought by proper consideration on noise modeling has been verified in various applications like denoising(Chen et al. (2017)), background subtraction(Yong et al. (2017)), derain(Wei et al. (2017)) and medical image reconstruction and ect.

There are also lack of effective disentanglement performance metrics for VAE in real application. The traditional performance metric raised by Higgins et al. (2016) is intractable to be computed in real data and hard to provide direct feedback of the disentanglement of VAE in used currently.

To alleviate the aforementioned issues on VAE, in this study, we attempt to address the aforementioned issues through a noise modelling perspective regarding VAE. This perspective provides us a reinterpretation that the generation process of VAE can be viewed as a noise adaption process. Since the derivation of the VAE is constructed based on the maximum likelihood principle, the supplemental parameters deduced by noise distribution amelioration can be directly embedded into and learnt through optimizing the objective of VAE and general end-to-end learning algorithms can be easily deduced. In this work, we use two modeling regime for fitting noise to data: Gaussian (with its variance adaptively learnt from data) and mixture of Gaussian. In such way, we expect our VAE embedded with noise modeling regime capable of better deliver the intrinsic generalization mechanism underlying data and achieve better performance in both representation and reconstruction. Furthermore, in order to guarantee a better disentanglement of representation, the auxiliary constraints (Higgins et al. (2016)) are introduced to the objectives. Further, the factors properties regarding the Gaussian-prior-VAE are discussed mathematically and indicators for quantitatively assessing the factor disentanglement ability, factor influence degree and inference mutual information raised. Beyond the previous metrics for performance assessment which could be hardly used in practical cases, such raised indicators can help easily quantify VAE model performance w or w/o noise modelling and auxiliary constraints in real data. By qualitative (in visualization) and quantitative (in terms of the presented performance indicators), we show that overall noise modelling w & w/o auxiliary constraints are superior to original VAE by experiments on datasets, including CelebA(Liu et al. (2015)), Extended Yale Face B(Georghiades et al. (2001),Lee et al. (2005)) and MNIST(Lcun et al. (1998)). The proposed metrics can also facilitate effective discovering most "influential" latent factors to help generating and traversal in the latent space.

In summary, the contribution of this paper can be mainly summarized as follows.

- We propose the VAE model embedded with noise fitting component, which is expected to better adapt practical noise configurations in real data. Two noise modeling regimes are considered to construct the noise learning VAE model, Gaussian (with variance learned from data) and mixture of Gaussian. Such amelioration facilitates the VAE capable of always reducing the artificial intervention due to more proper guiding of noise learning. Further, auxiliary constraints can be introduced to guarantee a better disentanglement.

- We propose multiple quantitative indicators for VAE as well as their estimation methods to supervise the degree of disentanglement, to quantify the mutual information of codes/factors and original signal regarding inference/encoder network and to determine the influential factors, and provide a bunch of theorems/definitions to illustrate the idealistic properties of gaussian factors VAE. Different from the previous metrics for assessing VAE performance, these indicators can be calculated by directly implementing our proposed algorithms on any given data.  
- We substantiate the effectiveness of the proposed VAE-with-noise-modeling algorithm, as well as the proposed indicators on various datasets, and show the importance of such noise modeling consideration in both the reconstruction and disentanglement performance of VAE model as compared with the previous enumerated pre-specified noise VAEs. The mixture of gaussian(MoG) noise model with proper specified components number can further help achieve a more elaborate noise component decomposition. Also, the Gaussian and MoG noise assumption alleviate the blurry effect issue of the generated images generally encountered by previous VAE methods. The proposed indicators also facilitates an appropriate extraction on intrinsic latent factors underlying data.

The paper is organized as the following: The related work is briefly reviewed in Section 2. The proposed VAE model with noise modeling, together with its theoretical support and implementation algorithms, is introduced in Section 3. The proposed indicators for assessing the factor disentanglement ability, factor influence degree and inference mutual information, as well as their insightful theories, are given in Section 4. The experimental results are demonstrated in Section 5, and finally we provide conclusion and discussions on this work.

# 2 RELATED WORK

VAE was proposed by Kingma & Welling (2013) and Rezende et al. (2014) to implement the efficient learning and inference in directed probabilistic models regarding continuous latent variables with intractable posterior distributions and in scalable datasets. They introduced a network inference/recognition model to represent the approximate posterior distribution and utilized reparameterization trick for stochastic joint optimization of a variational lower bound containing the parameters of both the generative/decoder and inference/recognition/encoder models. While they also designed a network for parameterizing the noise for Gaussian MLP decoder, the capability of noise rectifying based on data has not been specifically emphasized, just similar to most of the latter VAE applications, which easily specify a fixed Gaussian noise (with preset variance parameter) but more or less underestimate the role of noise learning.

After being raised, many VAE variations have been proposed to boost VAE's capabilities in generation quality and/or disentanglement of the learned representation. In these methods, multiple efforts were made by improving the generative and inference network structures. Typical works along this line include the convolution/de-convolution structure raised by Kulkarni et al. (2015) and ladder structure raised by Zhao et al. (2017). Some other works advanced the mechanism under the VAE generation/inference processes. Typical works include the iterative attention generation/inference mechanism raised by Gregor et al. (2015), normalizing flow proposed by (Rezende & Mohamed (2015)) that enhanced the expressive ability of the approximate posterior and its variants (Kingma et al. (2016)).

Despite the improvement to the VAE itself, some other efforts were made by the ensemble between GAN with VAE. E.g., Larsen et al. (2015) unified GAN and VAE to obtain a better reconstruction and a high-level abstracts visual features embedding. Mathieu et al. (2016) also unified GAN and VAE but put emphasis on disentangling factors of variation. GANs without auxiliary design would learn the data distribution disregarding its noise level though suffer from unstable training and mode collapsing(Salimans et al. (2016)) while VAEs would assume a decomposition of the noise and oracle clean datapoint regarding the noise data with an auxiliary prior on the distribution regarding the factors. If the wrong specified noise in VAE is combined into GAN which inclines to conduce unexpected images with corruptions, it's reasonable that the ensemble model may achieve a better hypothesis of true data distribution.

Besides, many efforts were made by regularization on the factor distribution or factor generating effect. E.g., Makhzani et al. (2015) introduced an adversarial loss into the latent space of the autoencoder which in idealistic case could learn any kind factor/latent distribution including those contributing to the disentangled factors/representation. InfoGAN, raised by Chen et al. (2016), introduced the infomax principle to GAN by adding an auxiliary mutual information regularization which enabled the inference of GANs and led to a better disentangled representation as well.

However, most of these current VAE models need to pre-specify the noise parameter before VAE training, which inclines to make them perform not stably well especially under complex noises especially different from the subjective pre-specified noise configuration. This issue thus inspires us to introduce noise learning component into the VAE model (as depicted in Fig. 1 to ameliorate the model better fitting the real data and further boost the VAE performance in both latent factor representation and signal reconstruction.

Recently, there is a new VAE variation is proposed by Higgins et al. (2016) who introduced the  $\beta$ -VAE framework which enhanced the constraints regarding the KL-divergence of the posterior and prior distribution of VAE and showed a novel disentanglement performance. This method has obtained a better performance as compared with conventional VAE methods, especially on its flexible tuning a compromising a parameter beta between the KL-divergence term and the likelihood term (the variational lower bound). We thus select this VAE model to embed our noise modeling regime to improve its capability of adapting real noise in data. It should be noted that  $\beta$ -VAE still assumes a Gaussian noise with pre-specified variance parameter, and the noise modeling regime is thus expected to further improve its learning capability and ameliorate its performance to be more stable and robust to real data noise.

In the other perspective of the decomposition of the noise and oracle clean data, deep denoising models, including denoising autoencoder(Goodfellow et al. (2016)), stacked denoising autoencoders(Vincent et al. (2010)), denosing variational encoder(Im et al. (2017)) etc., have also shown similar ability to the proposed VAE model with noise modeling. However, those models are trained by using paired/supervised data (both corrupted and clean data) while our model just learn in a purely unsupervised manner without supervision regarding the noise/or oracle data. Just similar to traditional VAE, such implementation paradigm is more similar to human learning process, and can be better generalized into real applications lacking supervision knowledge.

# 3 VAE WITH NOISE MODELING

In traditional VAE applications, the noise is generally fixed as a Gaussian with fixed variance. In this section, we further extend such noise-specification implementation as a automatic noise adapting regime. Specifically, we take the Gaussian variance parameter as an optimization variable and integrate it into the VAE model to make the noise fitted by data, and furthermore, we ameliorate the noise as a mixture of  $\mathrm{Gaussian}^1$  to further enhance its noise modeling capability. The  $\beta$ -VAE is employed to integrate such noise modeling mechanism.

# 3.1 INTERPRET THE ORACLE GENERATION OF VAE IN NOISE MODELING PERSPECTIVE

When modelling the real-valued generation process, VAEs often assume the conditional distribution to be

$$
p _ {m o d e l} (x | z) = \mathcal {N} (x | G (z), \sigma^ {2} I _ {d}),
$$

where  $x \in \mathbb{R}^D$  is the random vector variable corresponding to input data and  $z \in \mathbb{R}^H$  represents the latent factors for implicitly generating the data.  $G$  is the generating/decoder function parameterized by neural networks and  $\sigma^2$  is the variance parameter of the Gaussian distribution.

Such VAE model can be equivalently interpreted in perspective of noise modeling. The data are formulated by the ideal generation  $G(z)$  (clean data, where  $z$  follows a prior  $p(z) = \mathcal{N}(z;0,I_H)$ ) corrupted by an additional element-wise Gaussian noise  $\varepsilon$  with variance  $\sigma^2$ . mathematics, this

saying can be expressed as the following

$$
x = G (z) + \varepsilon . \tag {1}
$$

In VAE setting, the approximate inference<sup>3</sup> method is applied to maximizing the variational lower bound of  $p_{model}(x) = \int p_{model}(x|z)p(z)dz$ ,

$$
\mathcal {L} (q) = \underset {z \sim q (z \mid x)} {\mathbb {E}} \log p _ {\text {m o d e l}} (x \mid z) - D _ {K L} (q (z \mid x) \| p (z)) \leq \log p _ {\text {m o d e l}} (x). \tag {2}
$$

In most practical applications of VAE, the noise level,  $\sigma^2$ , is generally heuristically pre-specified(Makhzani et al. (2015), Walker et al. (2016), Higgins et al. (2017a)). Since the objective of the log-likelihood needs to negotiate with the KL-divergence term in the optimization process for VAE, if the noise level cannot guarantee to comply with the real noise level underlying data, the properness of the above VAE model could not be satisfied, which naturally tends to lead to the instable performance of the method in practice (i.e. suffer from Severely Wrong Model Assumption). Such an issue has been observed in  $\beta$ -VAE(Higgins et al. (2016)), which enforces a tunable parameter  $\beta$  to more or less deliver noise variation knowledge of data in the VAE model<sup>4</sup>, while such  $\beta$  still needs to be manually pre-specified before VAE optimization process, which is generally still a challenging task for real data.

# 3.2 NOISE MODELLING WITH AUXILIARY CONSTRAINT

The entangled representation can be caused by the over-large of searching space of  $q(z|x)$ . The posterior distribution searching space determined by the neural structure could always be two large in VAE application. If the learned  $q(z) = \int q(z|x)p_{data}(x)dx$  doesn't factorize, then the VAE model in the perspective of inference network just tends to learn the entangle representation. Actually, in the VAE model, what we want is to search in the space that  $q(z)$  is possibly similar to  $p(z)^5$ . By implementing this ideal, we add auxiliary upper bound  $\mathbb{E}_{x\sim p_{data}(x)}D_{KL}(q(z|x)||p(z))$  (detailed in Theorem 5) of  $D_{KL}(q(z)||p(z))$  to the original objective. This equivalently leads to the approach of  $\beta$ -VAE raised by Higgins et al. (2016),

$$
\sup_{q(z|x)}\mathbb{E}_{x\sim p_{data}(x)}\mathcal{L}(q(z|x)) - (\beta -1)D_{KL}(q(z|x)||p(z))
$$

$$
\arg \inf  _ {G} - \log p (x | z) = \frac {\| x - G (z) \| _ {2} ^ {2}}{2 \sigma^ {2}} + \frac {1}{2} \log (2 \pi) ^ {m} | \sigma^ {2} I _ {m} | \longleftrightarrow \arg \inf  _ {G} \| x - G (z) \| _ {2} ^ {2}.
$$

$$
= \underset {x \sim p _ {\text {d a t a}} (x)} {\mathbb {E}} \underset {z \sim q (z | x)} {\mathbb {E}} \log p _ {\text {m o d e l}} (x | z) - \beta D _ {K L} (q (z | x) \| p (z)) \tag {3}
$$

where  $\beta > 1$ .

# 3.3 INTEGRATING NOISE FITTING AND VAE PARAMETER LEARNING

We can easily integrate the noise parameters into the VAE objective to make the noise as a learnable part in VAE to compensate a better performance on both representation and reconstruction of VAE.

# 3.3.1 GAUSSIAN CASE

The VAE objective (i.e., the variational lower bound) can be treated as a function of the noise variance parameters  $\sigma^2$  and networks parameters  $W$  (which parameterizes the factor  $W_{G}$ , the encoder and the posterior distribution  $W_{E}$ ),

$$
\mathcal {L} (W, \sigma , x ^ {m}) = \underset {z \sim q (z | x ^ {m}; W _ {E})} {\mathbb {E}} \log p _ {\text {m o d e l}} \left(x ^ {m} \mid z; \sigma , W _ {G}\right) - D _ {K L} \left(q \left(z \mid x ^ {m}; W _ {E}\right) \mid \mid p _ {\text {m o d e l}} (z)\right). \tag {4}
$$

Here the SGVB estimator in (Kingma & Welling (2013)),  $\tilde{\mathcal{L}}^B (W,\sigma ,x^m)$  =  $[\frac{1}{L}\sum_{l = 1}^{L}\log p_{model}(x^m |z^l;\sigma ,W_G)] - D_{KL}(q(z|x^m;W_E)||p_{model}(z))$  is used. Note that the noise variance  $\sigma$  is also taken as an optimization variable in the model, making the model capable of better adapting noise variation of data in practical cases in a totally automatic way, instead of a manually set manner.

Given multiple data points from a dataset  $\mathbf{X}$ , we can construct an estimator of the mean marginal likelihood lower bound of the full dataset, based on minibatches

$$
\tilde {\mathcal {L}} ^ {M} (W, \sigma , X ^ {M}) = \frac {1}{M} \sum_ {m = 1} ^ {M} \tilde {\mathcal {L}} ^ {B} (W, \sigma , x ^ {m}), \tag {5}
$$

where the minibatch  $X^{M} = \{x^{m}\}_{m = 1}^{M}$  is a randomly drawn sample set of  $M$  datapoints from the full dataset  $X$ . Such a lower bound also constitutes an important indicator for model evidence in latter experiment. We call it the empirical variational lower bound (EVLB) in the following.

Note that  $\tilde{\mathcal{L}}^B (W,\sigma ,x^m)\simeq \mathcal{L}(W,\sigma ,x^m)$  and we can deduce that

$$
\begin{array}{l} \tilde {\mathcal {L}} ^ {M} (W, \sigma , X ^ {M}) \simeq \underset {x \sim p _ {d a t a} (x)} {\mathbb {E}} \mathcal {L} (W, \sigma , x) \\ \leq \underset {x \sim p _ {\text {d a t a}} (x)} {\mathbb {E}} \log p _ {\text {m o d e l}} (x; W _ {G}, \sigma) \leq \underset {x \sim p _ {\text {d a t a}} (x)} {\mathbb {E}} \log p _ {\text {d a t a}} (x). \tag {6} \\ \end{array}
$$

The last inequality holds due to  $D_{KL}(p_{data}(x)||p_{model}(x;W_G,\sigma))\geq 0$

The alternative optimization strategy can be readily utilized to design the algorithm for solving the model by iteratively updating the noise parameter and the network ones. During the optimization process, the objective can be monotonically increasing, and thus the algorithm can be guaranteed to be convergent.

The algorithm is summarized as follows:

Optimization for W: gradient method for  $W$  in regard to  $\tilde{\mathcal{L}}^M (W,\sigma ,X^M)$

Optimization for  $\sigma: \sigma^2 = \frac{\sum_{l=1}^{L} \sum_{m=1}^{M} \|x^m - G(z^{m,l})\|_2^2}{dML}$ . (Close form solution in regard to  $\tilde{\mathcal{L}}^M(W, \sigma, X^M)$ .)

Direct gradient method to the transformed variable  $\log_{-}simga = \log \sigma \in \mathbb{R}$  can be implemented to lift the lower bound  $\tilde{\mathcal{L}}^M$  as a result to increase the likelihood as well.

# 3.3.2 MIXTURE OF GAUSSIAN CASE

The noise  $\varepsilon$  in Eq. (1) in real situation might be more complex than a simple Gaussian, like that existed in real photographs(Plotz & Roth (2017)). We thus try to further ameliorate the noise setting

as a mixture of gaussian(MoG) noise. Such noise modeling strategy has been widely verified to be effective in applications, like matrix factorization (Meng & Torre (2014)) and robust principal component analysis (Zhao et al. (2014)). That is, we assume that

$$
\varepsilon \sim \sum_ {k = 1} ^ {K} \pi_ {k} \mathcal {N} \left(0, \sigma_ {k} ^ {2}\right). \tag {7}
$$

Let  $c_{d} \in \{0,1\}^{K}$  be the latent indicator random one-hot variable,  $\sum_{k=1}^{K} c_{dk} = 1$ , for the MoG-noise component of pixel indexed by  $d$ . Let  $\Pi = [\pi_1, \dots, \pi_K]$  and  $\Sigma = [\sigma_1^2, \dots, \sigma_K^2]$  be the ratio and variance of each component, respectively. Let  $W_N = [\Pi, \Sigma]$ . The conditional joint distribution turns to be

$$
p _ {\text {m o d e l}} \left(c _ {d}, x _ {d} \mid z, W _ {N}, W _ {G}\right) = \prod_ {k = 1} ^ {K} \pi_ {k} ^ {c _ {d k}} \mathcal {N} \left(x _ {d} \mid G (z) _ {d}, \sigma_ {k}\right) ^ {c _ {d k}}. \tag {8}
$$

The posterior distribution  $q(z, c|x)$  can be factorized as  $q(z|x)q(c|x,z)$ , where  $q(z|x)$  will be direct learnt and the alternative of  $q(c|x,z)$ ,  $q(c|x,e)$  will be set to the last step  $p_{model}(c|x,e)$  in regard to EM procedure. The lower bound of  $\log p_{model}(x)$  is then reformulated as follows:

$$
\mathcal {L} (q (z, c | x)) = \underset {z \sim q (\tilde {z} | x)} {\mathbb {E}} \underset {c \sim q (\tilde {c} | x, \tilde {z} = z)} {\mathbb {E}} \log p _ {\text {m o d e l}} (x, c | z) + H (q (c | x, z)) - D _ {K L} (q (z | x) | | p _ {\text {m o d e l}} (z)). \tag {9}
$$

Similar to the Gaussian case, the reparameterization trick is implemented,

$$
\begin{array}{l} \mathcal {L} (q (c | x, e), W _ {N}, W _ {G}, W _ {E}, x ^ {m}) \\ = \underset {e \sim \mathcal {N} (0, 1)} {\mathbb {E}} \underset {c \sim q (\tilde {c} | x, e)} {\mathbb {E}} \log p _ {\text {m o d e l}} (x, c | \tilde {z}) + \mathcal {H} (q (c | x, \tilde {z})) - D _ {K L} (q (z | x) | | p _ {\text {m o d e l}} (z)), \tag {10} \\ \end{array}
$$

where  $\tilde{z} = En(x) + \Sigma_{z|x}^{1/2}(x)e$ .

By utilizing the SGVB estimator, we get,

$$
\begin{array}{l} \tilde {\mathcal {L}} ^ {B} (q (c | x, e), W _ {N}, W _ {G}, W _ {E}, x ^ {m}) = \lceil \frac {1}{L} \sum_ {l = 1} ^ {L} \underset {c \sim q (\tilde {c} | x ^ {m}, e ^ {(l)})} {\mathbb {E}} \log p _ {m o d e l} (x ^ {m}, c | z ^ {m, l}) \\ \left. + \mathcal {H} \left(q \left(c \mid x ^ {m}, e ^ {(l)}\right)\right) \right] - D _ {K L} \left(q \left(z \mid x ^ {m}\right) \right\rvert   | p _ {\text {m o d e l}} (z)). \tag {11} \\ \end{array}
$$

Given an input dataset  $X$ , we can then construct an estimator to the mean marginal likelihood lower bound of the full dataset, based on minibatches, as follows:

$$
\tilde {\mathcal {L}} ^ {M} \left(q (c | x, e), W _ {N}, W _ {G}, W _ {E}, X ^ {M}\right) = \frac {1}{M} \sum_ {i = 1} ^ {M} \tilde {\mathcal {L}} ^ {B} \left(q (c | x, e), W _ {N}, W _ {G}, W _ {E}, x ^ {m}\right), \tag {12}
$$

where  $z^{m,l} = En(x^{m}) + \Sigma_{z|x}{}^{1/2}(x^{m})e^{(l)}$  and the minibatch  $X^{M} = \{x^{m}\}_{i=1}^{M}$  is a randomly drawn sample of  $M$  datapoints from the full dataset  $X$ .

Then let

$$
p _ {\text {m o d e l}} ^ {\text {o l d}} \left(c _ {d}, x _ {d} \mid z _ {\text {o l d}}, W _ {N} ^ {\text {o l d}}, W _ {G} ^ {\text {o l d}}\right) = \prod_ {k = 1} ^ {K} \pi_ {k} ^ {\text {o l d} ^ {c _ {d k}}} \mathcal {N} \left(x _ {d} \mid G ^ {\text {o l d}} \left(z _ {\text {o l d}}\right) _ {d}, \sigma_ {k} ^ {\text {o l d}}\right) ^ {c _ {d k}}, \tag {13}
$$

where  $z_{old} = En^{old}(x) + \Sigma_{z|x}^{old1/2}(x)e$ , and we can get

$$
p _ {\text {m o d e l}} ^ {\text {o l d}} \left(c _ {d} \mid x, z _ {\text {o l d}}, W _ {N} ^ {\text {o l d}}, W _ {G} ^ {\text {o l d}}\right) = \frac {p _ {\text {m o d e l}} ^ {\text {o l d}} \left(c _ {d} , x _ {d} \mid z _ {\text {o l d}} , W _ {N} ^ {\text {o l d}} , W _ {G} ^ {\text {o l d}}\right)}{\sum_ {c _ {d}} p _ {\text {m o d e l}} ^ {\text {o l d}} \left(c _ {d} , x _ {d} \mid z _ {\text {o l d}} , W _ {N} ^ {\text {o l d}} , W _ {G} ^ {\text {o l d}}\right)}. \tag {14}
$$

The EM algorithm can be naturally employed to solve the model. The implementation steps are listed as follows:

# Step 1. Expectation Step.

Set  $q(c|x^{m},e^{(l)}) = p_{model}^{old}(c|x^{m},En^{old}(x^{m}) + \Sigma^{old}(x^{m})e^{(l)})$ $i = 1,\dots ,m,l = 1,\dots ,L.$

Calculate the expectation of the latent variable  $c$ :

$$
E \left(c _ {d m l k}\right) = \gamma_ {d m l k} = \frac {\pi_ {k} \mathcal {N} \left(x _ {d} ^ {m} \mid G \left(z ^ {m , l}\right) _ {d} , \sigma_ {k} ^ {2}\right)}{\sum_ {l = 1} ^ {L} \sum_ {m = 1} ^ {M} \pi_ {k} \mathcal {N} \left(x _ {d} ^ {m} \mid G \left(z ^ {m , l}\right) _ {d} , \sigma_ {k} ^ {2}\right)}, \tag {15}
$$

where  $z^{m,l}:z_{old}^{m,l} = En^{old}(x^m) + \Sigma^{old}(x^m)e^{(l)}$ .

The Objective in Maximization Step is obtained as the following,

$$
\begin{array}{l} \tilde {\mathcal {L}} ^ {M} (q (c | x, e), W _ {N}, W _ {G}, W _ {E}, x ^ {m}) = \frac {1}{M} \sum_ {i = 1} ^ {M} - D _ {K L} (q (z | x ^ {m}) | | p _ {m o d e l} (z)) \\ + \frac {1}{L} \sum_ {l = 1} ^ {L} \mathcal {H} \left(q ^ {\text {o l d}} \left(c \mid x ^ {m}, e ^ {(l)}\right)\right) + \sum_ {k = 1} ^ {K} \sum_ {d = 1} ^ {D} \gamma_ {d m l k} \left[ \frac {\left(x _ {d} ^ {m} - G \left(z ^ {m , l}\right) _ {d}\right) ^ {2}}{2 \sigma_ {k} ^ {2}} + \frac {1}{2} \log (2 \pi) \sigma_ {k} ^ {2} + \ln \pi_ {k} \right]. \tag {16} \\ \end{array}
$$

# Step 2. Maximization Step:

Fix:  $q(c|x, e)$  determined in the Expectation Step.

$$
\frac {1}{M} \sum_ {i = 1} ^ {M} - D _ {K L} (q (z | x ^ {m}) | | p _ {\text {m o d e l}} (z)) + \frac {1}{L} \sum_ {l = 1} ^ {L} \sum_ {k = 1} ^ {K} \sum_ {d = 1} ^ {D} \gamma_ {d m l k} \left[ \frac {\left(x _ {d} ^ {m} - G \left(z ^ {m , l}\right) _ {d}\right) ^ {2}}{2 \sigma_ {k} ^ {2}} + \frac {1}{2} \log (2 \pi) \sigma_ {k} ^ {2} + \ln \pi_ {k} \right]. \tag {17}
$$

Update  $[W_N],[W_G,W_E]$  by alternative optimization strategy.

Update  $\Pi, \Sigma$ : note here  $z^{m,l}: z_{old}^{m,l} = En^{old}(x^m) + \Sigma^{old}(x^m)e^{(l)}$ , and we can easily get the closed-form updating formula for these parameters:

$$
N _ {k} = \sum_ {d, m, l} \gamma_ {d m l k} \quad \pi_ {k} = \frac {N _ {k}}{\sum_ {k = 1} ^ {K} N _ {k}} \quad \sigma_ {k} ^ {2} = \frac {1}{N _ {k}} \sum_ {d, m, l} \gamma_ {d m l k} \left(x _ {d} ^ {m} - G \left(z _ {o l d} ^ {m, l}\right) _ {d}\right) ^ {2}. \tag {18}
$$

Update  $W_{G}, W_{E}$ : gradient methods with respect to  $W_{G}, W_{E}$ . Note here  $z^{m,l} = En(x^{m}) + \Sigma (x^{m})e^{(l)}$ .

The algorithm can then be summarized as follows:

1. Initialize the coefficient of  $W_{G}, W_{E} = [Encoder, \Sigma_{z|x}]$  and the coefficient of noise  $\varepsilon$ :  $\Pi, \Sigma$ .  
2. Sample  $e$  from  $\mathcal{N}(0, I_H)$  to obtain  $e_1, \dots, e_M$  [One for each element sample in the mini batch in the next step ( $L$  here is set to 1)].  
3. Sample a mini batch  $X^{M}$  from  $p_{data}(x)$ .  
4. Implement EM algorithms as aforementioned (approximate inference for  $q(c, z|x)$ ):

Expectation: calculate  $\gamma_{dmk}$ .

Maximization: update  $W_{N}$ , Update  $W_{G}, W_{E}$  with gradient methods.

5. Goto 3: Until Trigger End-Criterion.

# 3.3.3 NETWORKPARAMERIZEDGAUSSIANCASE

Here we want to give a short discussion on the difference of the proposed VAE model with noise modeling with that raised by Kingma & Welling (2013), in which they model the noise as the following parameterized network structure:

$$
x \sim \mathcal {N} (G (z), \sigma^ {2} (z))
$$

$$
G (z) = W _ {u} h (z) + b _ {u}
$$

$$
\log \sigma^ {2} (z) = W _ {\sigma} h (z) + b _ {\sigma}, \tag {19}
$$

where  $h$  represents the mapping induced by the previous network layers;  $\sigma^2(z)$  represents the diagonal of diagonal covariance matrix;  $\{W_u, W_\sigma, b_u, b_\sigma\}$  are the weights and biases of the last layer of network.

It can be observed that this model assumes that each pixel noise indexed by  $d$  has its own level determined by  $b_{\sigma d}$  and is also influenced by the deterministic part  $W_{\sigma}h(z)$ . When the noise level is shared among all data points and not influenced by the deterministic part, then it degenerates to the Gaussian assumption; if the noise has several discrete levels, it then tends to degenerate to MoG assumption on noise. However, on the one hand, the assumption used in the model inclines to make the optimization for the model difficult due to the numerical instability caused even by one zerovariance residual pixel. This is possibly why most applications on VAE have not employed such noise assumption while prefer to fix and manually set a noise level before VAE training. On the other hand, the over-parameterized noise could significantly increase the difficulty to find a better deterministic part since the model might be inclined to fit the noise hypothesis rather than learn a good  $G$ . These limitations have been empirically verified by all our experiments and can be observed in the experimental results as listed in Section 5.

# 4 GENERATING FACTOR PROPERTIES AND PERFORMANCE INDICTORS

In order to evaluate the performance of our model, we propose multiple new indicators. All these indicators can be approximately calculated from input data, which ameliorates the issue that the previous performance metric cannot be easily computed from real data (as discussed in Section 4.2.1). Roughly speaking, we try to show that idealistic VAE model $^6$  is hard to properly learn excess/extra factors by information conservation theorem(Theorem 1); factors that are possible to be learnt by idealistic VAE tend to form an equivalence class under the orthogonal transformation by Gaussian factor equivalence theorem(Theorems 2 and 3) and therefore even idealistic VAE cannot learn "semantic disentangle" representation. Subsequently, multiple meaningful performance indicators are raised: the estimation for  $D_{KL}(q(z)||p(z))$  is used for quantifying the disentanglements, and the estimation of  $\mathcal{I}_{encoder}(x;z_h)$ , used for quantifying the influential("used") factors.

# 4.1 INFORMATION CONSERVATION

Whether VAE can learn the real factors or just some fantasies that model itself makes up is an important issue for a generative model. We try to address this issue by disregarding the training procedure and direct considering the idealistic VAE's behavior through the following theorem.

Theorem 1 (Information Conservation). Suppose that  $z = (z_{1},\dots ,z_{H})$  and  $y = (y_{1},\dots ,y_{P})$  are sets of  $H$  and  $P(H\neq P)$  independent unit Gaussian random variables, respectively, then these two sets of random variables can not be the generating factor of each other. That is, there are no continuous functions  $f:\mathbb{R}^H\to \mathbb{R}^P$  and  $g:\mathbb{R}^P\to \mathbb{R}^H$  such that

$$
z = g (y) \quad a n d \quad y = f (z).
$$

The principle of the theorem is visually illustrated in Fig. 2. This theorem roughly demonstrates that the number of the learnt "used" factors of VAE can be the same as the true factors number under some assumptions such as the learnt  $q(z)$  should equal  $p(z)$  and decode/encode process is continuous and reversible. Empirically, only a small amount of unit gaussian variables regarding the factors of well disentangled VAE have been used in practical VAE application and this theorem helps provide an interpretation to explain this phenomenon. Suppose that the observed data, denoted by random variable  $x$ , is generated by  $y$  (with  $P$  independent unit gaussian random variables) with a homeomorphism mapping  $x = \phi(y)$ . VAE will be forced to learn the factor  $z$  (with  $H$  independent unit gaussian random variables) that generates the  $x$  with a homeomorphism mapping  $x = \psi(z)$ . It yields  $z = \psi^{-1} \circ \phi(y)$  and  $y = \phi^{-1} \circ \psi(z)$ . Then according to the information conservation theorem, it must hold that  $H = P$ .

![](images/3074fd8e892557309b217e9b687abc68f3fe36f8d09700b705642ddd2aee6566.jpg)  
Figure 2: The illustration of the information conservation theorem

# 4.1.1 EFFICIENT REPRESENTATION AND CLARIFICATION ON DISENTANGLEMENT

According to the information conservation theorem, the independent unit gaussian distribution assumption regarding the factor of the model facilitates the model incline to achieve most efficient coding.(i.e., the number of functional latent factors extracted from the model should be same as that of the intrinsic latent factors underlying the model). That is, no auxiliary factor tend to be learnt, though the number latent of factors sometimes is pre-specified larger than that in the idealistic VAE setting.

Here, in order to avoid the ambiguity of the terminology of disentanglement, we make the following clarification.

- The disentanglement of the learnt representation/factors in this literature refers to two parts depicted in Theorem 1:

- the factors are closer to be independent with each other,  
- the factors incline to be able to generate the oracle signal and to be inferred perfectly from the oracle signal through a continuous procedure/mapping.

- The "disentanglement" refers to the closeness of the learnt factors to the pre-specified independent factors/concpets that can generate the oracle signal and be perfect inferred through a continuous procedure/mapping such as the independent semantic/visual factors.

Therefore, the estimation for  $D_{KL}(q(z) || p(z))$  that reflects the divergence of the learnt factor distribution and the i.i.d. unit gaussian prior can be good a indicator to supervise the independence of the factors can be served to quantitatively assess the disentanglement of each extracted factor.

The "disentanglement" will be shown that is hard to be obtained in a unsupervised manner. Concretely, even in the idealistic cases, the extracted factors tend to possess the intrinsic number of latent factors of the model, while there are still possibly large variations of these factors due to it can be obtained in as proved in the next subsection.

# 4.2 GENERATOR EQUIVALENCE

Theorem 2 (Gaussian Factor Equivalence). Suppose that  $z = (z_{1},\dots ,z_{H})$  is a set of  $H$  independent unit gaussian random variables. Let  $Q\in \mathbb{R}^{H\times H}$  be an orthogonal matrix and then  $y = Qz$  is also a set  $H$  independent unit gaussian random variables. Besides,  $z$  and  $y$  can generate each other through a linear homeomorphism mapping.

This theorem implies that there are a class of unit Gaussian random variables which can generate each other and have equivalent conservation information, as indicated by the following theorem.

Theorem 3 (Linear Gaussian Factor Equivalence Class).

$$
[ z ] = \left\{y \mid y = Q z, \quad Q \in \mathbb {R} ^ {H \times H} \text {b e t h e o r t h o g o n a l m a p p i n g .} \right\}
$$

Then  $\forall y \in [z]$ ,  $y$  is a set of  $H$  independent unit gaussian random variables and can generate  $z$  through an linear homeomorphism mapping.

The theorem clarifies that if VAEs have an linear matrix multiplication freedom degree of learning the factors, then the factors in the equivalence class can all be possibly learnt.

The empirically results tally with the above analysis(see Fig. 3). Suppose the visual semantic concepts can be viewed as a set of independent Gaussian variables  $(z = (z_{rotation}, z_{gender}, z_{with - glass}, \dots)^T)$  which are desired to be captured and learnt by VAEs, while the model is also possible to learn the independent factor set  $y = (y_1, \dots)^T = Qz$  in the equivalence class  $[z]$ . This explains why changing one factor like  $y_1$  always empirically results in change in multiple visual concepts.

![](images/d2e323ebdef8c8167413df58bf9beb9457c2952d49b08e2c5823b81637c80bf8.jpg)  
(a)

![](images/230b7f4bf1ca139965b28866d6919159898b9b6da061c04586e5107e78d44ccb.jpg)  
(b)

![](images/ca09fdd6444d7b7ab9bf251f535c704a2b6205d42a4c1eb31a9705905339c607.jpg)  
(c)  
Figure 3: One-shot traversal & generating factor equivalence class demonstration. The images are generated by MoG-2  $\beta (=40)$  VAE trained on CelebA. The seed image is obtained out from the datasets. Each block represents the traversal of the generating factor from  $[-3 + z_{\text{seedh}}, +3 + z_{\text{seedh}}]$ . (a) corresponds to face color white-yellow & female-male change. (b) corresponds to face color white to yellow change. (c) corresponds to background yellow to blue change. (d) corresponds to hair color white to black & face width change. It can be seen that changing one factor results in multiple semantic factor change in a comprehensible manner which reflected analysis regarding generating factor equivalence.

![](images/ccbe2e91d10fd6ae71a4d5729362adfaccb63cad61a7dd0a2cf6c29c7d448d0a.jpg)  
(d)

This perspective also suggests that it's actually hard to obtain the disentangled representation that exactly "one-to-one" corresponds to the "independent semantic representation" even though they are in the same equivalence class. As a result, the idealistic VAE model just tends to learn the "entangle representation" if we do preset a "oracle generating factor" belonging to the equivalent class.

However, though those conclusions might be upsetting, it seems not be biology impossible. Many biological evidences have proved that actually a neuron in the brain of animals could combinationally possess several functional capabilities. E.g., Aronov et al. (2017) found that some neurons in rat's hippocampus involved in representing sound frequencies also were involved in spatial representation after training rats by a tasks that required them to use a joystick to manipulate sound in frequency continuously. We thus expect that even with such "entangling mechanism" the extracted factors by VAE could also possess rational representation capabilities and be finely interpreted.

If we assume that the most of the visual concept/factors follow the condition regarding the "disentanglement", it is rational to qualitatively measure the interpretability of extracted latent factors to infer the disentanglement. Besides, both the biological and previous empirical evidences of VAE applications(Higgins et al. (2016),Higgins et al. (2017b),Larsen et al. (2015),Mathieu et al. (2016)) have shown that such extracted factors located in the equivalent class can also finely reveal the interpretable representations underlying data.

# 4.2.1 DEFICIENCY OF THE EXISTING DISENTANGLEMENT METRIC

Higgins et al. (2016) proposed an "simulated factor" based disentanglement metric on the simulation datasets. However, this metric could be hardly calculated in the real datasets to provide direct feedback of the disentanglement of the model since it needs to pre-know the generating factors of the VAE model by default, which yet are generally hardly to know in practice. Besides, according to Gaussian generator equivalence theorem(Theorem 2) that even idealistic VAE will still learn the factors in the equivalence class, their metric can suffer severely instability to evaluate the VAE in different trials (detailed in Appendix 7.2).

# 4.3 INFORMATION CHANNEL

The mutual information<sup>7</sup> regarding the factors learned by the inference/encoder network and the signal  $x$  can be a good quantity for evaluating the generating influence<sup>8</sup>. That is,

$$
\mathcal {I} _ {\text {e n c o d e r}} (x; z) = \underset {x \sim p _ {\text {d a t a}} (x)} {\mathbb {E}} D _ {K L} (q (z | x) \| q (z)). \tag {20}
$$

In order to understand and estimate which factor of the VAE was learnt and influenced the generating process,  $\mathcal{I}_{\text{encoder}}(x;z_h)$  can be taken as a rational indicator<sup>9</sup>. If we assume that  $z_1, z_2, \dots, z_H$  is conditional independent given  $x^{10}$ , it can yield a useful result as the following.

Theorem 4 (Separation of the Mutual Information). Suppose  $z_{1}, \dots, z_{H}$  be independent unit gaussian distribution, and  $z_{1}, z_{2}, \dots, z_{H}$  be conditional independent given  $x$ . Then

$$
\mathcal {I} \left(z _ {1}, \dots , z _ {H}; x\right) = \sum_ {h = 1} ^ {H} \mathcal {I} \left(z _ {h}; x\right). \tag {21}
$$

This theorem suggests that if the learnt  $q(z)$  can factorize and the  $q(z|x)$  can factorize, then the consideration of each  $\mathcal{I}(z_h; x)$  won't be excess or lose information.

# 4.4 INDICATORS

In order to quantify the disentanglement performance as well as the  $\mathcal{I}_{\text{encoder}}(x;z)$ . We assume that  $q^{*}(z)$  is a factorized zero mean gaussian estimation for  $q(z)$ . We first propose the following relevant theorem and then provide the indicators.

Theorem 5. The terminology follows the aforementioned definitions and if the involved KL-divergence and mutual information is well defined, then

$$
\underset {x \sim p _ {\text {d a t a}} (x)} {\mathbb {E}} D _ {K L} (q (z | x) \| p (z)) = \mathcal {I} _ {\text {e n c o d e r}} (x; z) + D _ {K L} (q (z) \| p (z)). \tag {22}
$$

The theorem demonstrates that the second term in variation lower bound in Eq. (3.2) capable of controlling both the mutual information of  $x$  and  $z$  induced by the encoder network as well as the similarity of the learnt  $q(z)$  and the prior  $p(z)$  (disentanglement performance). We can then list the indicators for assessing latent factor disentanglement:

Definition 1 (Estimation for  $\mathbb{E}_{x\sim p_{data}(x)}D_{KL}(q(z|x)||p(z)))$

$$
\tilde {D} _ {K L} (q (z | x) | | p (z)) = \frac {1}{M} \sum_ {m = 1} ^ {M} D _ {K L} (q (z | x _ {m}) | | p (z)). \tag {23}
$$

Definition 2 (Estimation for  $I_{\text{encoder}}(x;z)$ ).

$$
\tilde {I} _ {\text {e n c o d e r}} (x; z) = \frac {1}{M} \sum_ {m = 1} ^ {M} D _ {K L} \left(q \left(z \mid x _ {m}\right) \mid \mid q ^ {*} (z)\right). \tag {24}
$$

Definition 3 (Estimation for  $I_{\text{encoder}}(x;z_h)$  which quantifies the influence of each factor).

$$
\tilde {I} _ {\text {e n c o d e r}} (x; z _ {h}) = \frac {1}{M} \sum_ {m = 1} ^ {M} D _ {K L} \left(q \left(z _ {h} \mid x _ {m}\right) \mid \mid q ^ {*} \left(z _ {h}\right)\right). \tag {25}
$$

Definition 4 (Estimation for  $D_{KL}(q(z)||p(z))$ ).

$$
\tilde {D} _ {K L} (q (z) | | p (z)) = \tilde {D} _ {K L} (q (z | x) | | p (z)) - \tilde {I} _ {\text {e n c o d e r}} (x; z). \tag {26}
$$

Note that the above indicators 2-4 need the value of  $q^{*}(z)$ , we then introduce how to calculate this term based on Theorem 5. Through the minimization equivalence, we know taht

$$
\min  _ {Q} \underset {x \sim p _ {\text {d a t a}} (x)} {\mathbb {E}} D _ {K L} (q (z | x) | | Q (z)) \Leftrightarrow \min  _ {Q} D _ {K L} (q (z) | | Q (z)) d z, \tag {27}
$$

the  $q*(z)$  can then be obtained from solving the following optimization problem which can be calculated by gradient method.

$$
q ^ {*} (z) = \arg \min  _ {Q} \frac {1}{M} \sum_ {m = 1} ^ {M} D _ {K L} (q (z | x _ {m}) | | Q (z)). \tag {28}
$$

# 5 EXPERIMENT

In this section, we will show experimental results to both quantitatively demonstrate the superiority of a VAE model with embedded noise modeling component as compared with that without this part, and qualitatively show the better reconstruction capability and meaningful-latent-factor-extraction capability of the ameliorated VAE model with noise modeling. The functional effects of the proposed indicators can also be verified.

The comparison method is employed as the recently proposed  $\beta$ -VAE(Higgins et al. (2016)), which has been proved to have a good reconstruction and representation capabilities as compared with traditional VAE methods due to its involvement of a tunable compromising parameter  $\beta$  between the likelihood and KL-divergence terms in VAE objective. Besides, the network parameterized Gaussian noise learning VAE model is also considered for comparison in Extended Yale B dataset. We will show the performance amelioration taken by integrating noise modeling component in these methods.

![](images/aadde119f0ddf04c14d5b184a790420c969f41289b8282707f4c6dd4b2cdd74c.jpg)  
Figure 4: Noise specification influence the learned hypothesis and the reconstruction. Blue Line: the EVBL of different specified  $\sigma^2$  VAE [correspond to pre-specified  $\sigma_{pre}^2$ $\beta$ -VAE illustrated at footnote 4 with equivalent  $\sigma^2 = \beta \sigma_{pre}^2$ ]. Green Line: the EVLB of noise learning  $\beta$ -VAE with different specified  $\beta$  [normalized to  $\sigma^2 = \beta \sigma_{learn}^2$  for convenient comparing]. Red Plus: the EVLB of noise learning VAE. Other Figures: residual (=abs(original-reconstruct)) on the testing set.

# 5.1 EXPERIMENTS ON MNIST

MNIST is a database of handwritten digits. By setting  $\beta$  as different values, we compare the performance of  $\beta$ -VAE with and without considering noise modeling components on this dataset. We specifically listed the result of  $\beta (= 1)$ -VAE in all cases. More details can be referred to in Appendix 7.3.1.

The noise specifications significantly influence the quality of final method performance in both quantity and quality, as clearly shown in Fig. (4). However, different datasets have its own noise and the relative optimal specification of noise level in practice might be really hard to be obtained. It can be seen that in most cases the  $\beta$ -VAE model with noise modeling is superior in learning a relatively better hypothesis with higher EVLB to that without this component. This can be easily interpreted by the fact that each dataset has its own level of noises, and noise modeling regime in VAE model tends to help the model better fit such noise and naturally conduct a better reconstruction result.

![](images/11a81b50977e4c990399f97274920709f6a294706dbd805a994ee9e795067589.jpg)  
(a)

![](images/a9d58a5206f5631f8f16a25efe3c1eab44c9c292b3e3af93b49c34549057b4c0.jpg)  
(b)  
Figure 5: (a)  $\tilde{D}_{KL}(q(z)||p(z))$  of different VAE models & (b) Number of normal-variance generators of different VAE models (with 128 factors.)

The noise specifications significantly influence the disentanglement and noise learning  $\beta$ -VAE with noise modeling achieves a better disentanglement quantitatively based on the proposed indicators in the perspective of  $D_{KL}(q^{*}(z)||p(z))$  [11] and the number of normal variance factors. In regard to the same normalized variance, according to figure 5a, the factor distributions of the  $\beta$ -VAEs with noise learning are closer to the prior distribution, which means it is more likely to be independent. The learnt factors with an estimated normal variance  $\sigma_{z_h}^2 \geq 0.8$  are counted to convey more information regarding the factor distribution. According to Fig. (5b), the  $\beta$ -VAEs with noise learning learn significantly more normal variance factors, in regard to the same normalized variance. The values of these indicators quantitatively show that the factors of  $\beta$ -VAE with noise learning are more likely/closer to be independent with each other while also guarantee to be a good distribution hypothesis in regard to maximum likelihood principle as depicted in Fig. (4).

The  $\beta$ -VAE with noise learning also achieves a better disentanglement qualitatively. As shown in Fig. (6) and 7, in regard to normalized variance, the  $\beta$ -VAE with noise learning learns more interpretable factors as well as more normal variance factors. Also, according to the estimation of mutual information, the influence of factor is also more balanced than that of traditional  $\beta$ -VAE. It can also be found that  $\beta$ -VAE has the ability to automatically suppress the auxiliary factors and learn the intrinsic factor dimension, already suggested by the information conservation theorem 1.

We find that  $\beta$ -VAE with noise modeling suffers from the suppression on  $I_{encoder}(x;z)$ , as depicted in Fig. (8a) and that is comprehensible since  $\beta$ -VAE is minimizing the auxiliary constraints both  $I_{encoder}(x;z) + D_{KL}(q(z)||p(z))$  based on Theorem 5.

# 5.2 EXPERIMENTS ON EXTENDED YALE FACE DATABASE B

The extended Yale Face Database B contains images of several human subjects under different poses and different illumination conditions. In this series We compare the  $\beta$ -VAEs with noise Gaussian and MoG components, and network parameterized Gaussian noise and as well as different  $\beta$  balancing the representation ability. The Table 1 quantitatively compare the performance indicators of different methods.

![](images/a2fbba384dcd550ff4f9bc08d65c4cf47169fc76631a6750206643872ad4345d.jpg)  
Figure 6: Noise learning  $\beta$ -VAE ( $\beta = 8$ , equivalent  $\sigma^2 = 0.0944$ ): estimation of  $I_{encoder}(x;z_h)$ ,  $\sigma_h^2$  and qualitatively influential factor traversals. The top pulse subgraph: the estimated mutual information  $I_{encoder}(x;z_h)$  of each factor. The bottom reverse pulse subgraph: the estimated variance  $\sigma_{z_h}^2$  of each factor. The montages: influential factor traversals. In all figures of factor traversal each montage corresponds to the traversal of a single factor while keeping others fixed to their inferred (VAE,  $\beta$ -VAE). Each row corresponds to a different seed image used to infer latent factor value in the VAE-based models.  $\beta$ -VAE and VAE traversal is over [-3, 3]. Note that all the factors with  $I_{encoder}(x;z_h)$  not close to zero ( $>0.1$ ) can be visually tell the existence of their generation effect. We select those factor traversals with visually most interpretable/comprehensive effects to present. Due to the limitation of space, the whole influential factor traversals are listed in appendix 7.4.2 The mutual information of "used" factor learnt by noise learning  $\beta$ -VAE can be found relatively balanced. It's interesting that whether the learnt estimation of  $\sigma_{z_h}^2$  takes 1 or small value is strongly correlated with whether the estimation of  $I_{encoder}(x;z_h)$  be near zero or not. The phenomenon of the multiple semantic change induced by the same learnt factor and the encoding of same semantic among different learnt factor tallies with factor equivalence class theorem 2.

![](images/2aba4c0fd495fb99361922d2c868007d49a0e4d9e787ed3bd95aaf6dd56472c0.jpg)  
Figure 7: Noise specified  $(\beta)$ -VAE with equivalent  $\sigma^2 = 0.1$ : estimation of  $I_{encoder}(x;z_h)$ ,  $\sigma_h^2$  and qualitatively influential factor traversals. Note that we select those factor traversals with visually most interpretable/comprehensive effects to present. Due to the limitation of space, the whole influential factor traversals are listed in appendix 7.4.1. The mutual information of "used" factor learnt by noise specified  $\beta$ -VAE can be found more diverse than that in figure 6. The  $\sigma_{zh}^2$  and  $I_{encoder}(x;z_h)$  value correlation is also significant. However, the effect of factor learnt by the noise specified VAE is hard to be interpreted (They maybe not independent with each other.).

$\beta$ -VAE with MoG noise modeling ( $\beta = 1$ ) learns an evidently better distribution hypothesis compared with the Gaussian one. It's comprehensive that MoG-VAE learns two different noise level component according to Fig. (9). One of them can be interpreted as the intrinsic physical Gaussian

![](images/93ef8cc0a82c635c8a8156b24c71d839b740d33fe32a0b1a1a3b7b220507a70a.jpg)  
(a)

![](images/5b95b2a49b94716d31d47b9be58e3b34c4093d6c4bf701195f4cb463b90002cb.jpg)  
(b)  
Figure 8: (a)  $\tilde{I}_{\text {encoder }}(x;z)$  of different VAE models & (b) Number of influential generators of different VAE models

Table 1: Yale Face Database B Model Comparison [with 128 Latents ( H=128 )]  

<table><tr><td>β</td><td>Noise</td><td>σk2</td><td>πk2</td><td>EVLB</td><td>DKL(q(z)||p(z))</td><td>#(σ2zh&gt;0.8)</td><td>#(I(zh; x) &gt; 0.5)(updated)</td></tr><tr><td rowspan="5">1</td><td>G</td><td>0.00040</td><td>1</td><td>76892</td><td>127.2</td><td>0</td><td>128*</td></tr><tr><td>Network</td><td>-</td><td>-</td><td>90506</td><td>127.4</td><td>0</td><td>128</td></tr><tr><td>Network(MoG-2)</td><td>[0.00011 0.0053]</td><td>[0.728 0.272]</td><td>72778</td><td>127.4</td><td>0</td><td>128</td></tr><tr><td>Network(G)</td><td>0.0016</td><td>1</td><td>57195</td><td>127.4</td><td>0</td><td>128</td></tr><tr><td>MoG-2</td><td>[0.0001 0.0029]</td><td>[0.783 0.217]</td><td>81200</td><td>127.9</td><td>0</td><td>128</td></tr><tr><td>40</td><td>MoG-2</td><td>[0.00012 0.0043]</td><td>[0.719 0.281]</td><td>72346</td><td>89.28</td><td>73</td><td>46</td></tr><tr><td>80</td><td>MoG-2</td><td>[0.00012 0.0049]</td><td>[0.697 0.302]</td><td>69074</td><td>44.25</td><td>105</td><td>23</td></tr><tr><td>120</td><td>MoG-2</td><td>[0.00013 0.0054]</td><td>[0.664 0.336]</td><td>64439</td><td>27.27</td><td>114</td><td>14</td></tr><tr><td>160</td><td>MoG-2</td><td>[0.00013 0.0059]</td><td>[0.660 0.340]</td><td>63447</td><td>23.41</td><td>116</td><td>12</td></tr></table>

The result can be influenced by the specification of the initialization of the  $\sigma_k^2$  and  $\pi_{k}$ . Variance was clipped to 0.0001 to guarantee the stable optimization. The results of Network(MoG-2) and Network(G) are derived using the fixed  $G(z)$  but changing the noise hypothesis and recalculating the noise parameters. The * result is calculated by  $\tilde{D}_{KL}(q(z|x)||p(z)) - \bar{D}_{KL}(q(z)||p(z))$  and the others in the same column are calculated by  $\tilde{D}_{KL}(q(z)||p(z))$ .

noise and the other might be the part hard to be reconstructed. However, if the noise is assumed to be one Gaussian, then it can hardly decompose such an elaborate description of noise configurations, as shown in Fig. (9).

For the parameterized VAE, although the network parameterized noise VAE achieves the highest distribution hypothesis, the qualitatively reconstruction of the network was not as good as its EVLB. The model generates the more blurred reconstruction, which can be observed on two typical faces shown in Fig. (10). We further plug the generator  $G(z)$  into MoG-2 noise hypothesis, according to the Table 1, its EVLB decreases significantly. Besides, the network parameterized noise VAE suffers severely from the numerical instability such that we could always not finish a complete training(2002 epoch) due to the its objective collapsed illustrated in Section 3.3.3. All of the aforementioned evidences suggest that the model learns a relatively dedicate hypothesis for the noise rather than the deterministic part(oracle signal).

# 5.3 EXPERIMENTS ON CELEBA

CelebA is a large-scale celebfaces attributes datasets and only its images are used in our experiments. We compare the VAEs of different specification of number of latent factors with both Gaussian and MoG noise modeling and as well as different  $\beta$  balancing the representation ability. The following Table 2 of the performance indicators shows the comparison of the model.

Table 2: CelebA Model Comparison  

<table><tr><td>β</td><td>Noise</td><td>σ2k</td><td>πk2</td><td>EVLB</td><td>DKL(q(z)||p(z))</td><td>#(σ2zh&gt;0.8)</td><td>#(Izh; x) &gt; 0.5)</td><td># latents</td></tr><tr><td>40</td><td>MoG-2</td><td>[0.0030 0.029]</td><td>[0.628 0.372]</td><td>10552</td><td>24.22</td><td>94</td><td>27</td><td>128</td></tr><tr><td>30</td><td>MoG-2</td><td>[0.0027 0.027]</td><td>[0.637 0.363]</td><td>11324</td><td>29.72</td><td>82</td><td>32</td><td>128</td></tr><tr><td>1</td><td>G</td><td>0.011</td><td>1</td><td>10015</td><td>31.94</td><td>0</td><td>32</td><td>32</td></tr></table>

The  $\bar{I}(x;z_h)$  is calculated by  $\tilde{D}_{KL}(q(z|x)||p(z)) - \bar{D}_{KL}(q(z)||p(z))$  and  $\tilde{D}_{KL}(q(z)||p(z))$  is better.

![](images/0a423cd0acfb2dd3d09d93c3c10bccdd67671095252258ac287de330bfdef92d.jpg)

![](images/7b32ac2ad69d8875b0651f7306f8ca1b32d1cc2d538e33cdd5ac2a53917aa3ec.jpg)

![](images/375306a68a8ebda1dd9070846853b9d748616a43e0102ca81cef637e416ec928.jpg)

![](images/2ff0117128e3101ea07b3938e2031147df3fdb94c9ab43828bc7ae0e69df16e9.jpg)  
Figure 9: MoG-2/Gaussian VAE Reconstruction and Residual Gaussian Components Membership

![](images/24edac0d57bd0673af0c31dce9c0234578d79c5b9f87071bfe611e5d4e629b17.jpg)

![](images/82e5f0b98671d445b8663c1bbebbee9ac25ffd79c6359e53c9ecfa149de13ae3.jpg)

![](images/35a328928c753b00cbade9543c5dabc2f95588880adff0e16ac4546c2ce8f019.jpg)  
Origin  
Figure 10: Reconstruction Visual Comparison

![](images/61f60832a1d0a22a22b6404122db3eb98d7c26f648d3be3e263e771e0f1a6058.jpg)  
Gaussian

![](images/6965e80be7547228c3c6c0fe8107053cd1ae2ba86c6f4c9aa3f6012ea8ac939b.jpg)  
MoG-2

![](images/1df4faa365827c317ffa14fad511856d59b92b543c93f3024b981b6d761f5f41.jpg)  
Network

![](images/1572e966139d7e4b8bc0e13879fb11be8675d1306247bfd14e24ee114f377bf7.jpg)  
$\sigma$  Map

The table shows that as compared with carefully specification of the generating factors number, noise modelling and auxiliary constraints make the model capable of learning both better hypothesis and disentangled representation.

The generating equivalence property is again well demonstrated by seeing Fig. (11): "Blue to Yellow" background change  $\sim$  factor 13, 96,40,45,118. "Black to White" background change  $\sim$  factor 7. Height  $\sim$  factor 37,45. Mouth Open to Close background  $\sim$  factor 8. Face direction change  $\sim$  factor 26,31. "Male to Female" change  $\sim$  factor 28. "Big to Small" face change  $\sim$  factor 63,77,82. Lighting  $\sim$  factor 73,90. Face lighting  $\sim$  factor 120,110. Glass  $\sim$  factor 73. Neck length  $\sim$  factor 102. "White to Yellow" skin color change  $\sim$  factor 102,96,28,63,82. Hair color  $\sim$  factor 120. "Half Bright Half Gloomy" background change  $\sim$  factor 110.

# 6 CONCLUSION AND PERSPECTIVE

In summary, the paper obtains the following conclusions:

- Integrating noise modeling component into a VAE model tends to evidently ameliorate the reconstruction quality of VAE and disentanglement performance from the evidence of the indictors and interpretability of the extracted latent factors obtained by.

![](images/d4da3906f2c619d34dc1e1efefa548a91e25e84c15fe91144289278704bc864e.jpg)

![](images/46a8fdac6255099d73e19835aec882a032985b5589518d8e3052893a63d5e6f3.jpg)

![](images/131487533023bd1f899ad8a085af282aa9e4469802cbf446435a7731c01be4ed.jpg)

![](images/0fccbb4aaad3eed82b4f482dfe1c7f89392744734831cbde06f56f62ad174599.jpg)  
factor 7

![](images/00c57efd8ad0d43d0d2072c1537631197f1a1fe361b7c8c6794664c0ad8a52fe.jpg)  
factor 8

![](images/86f48a2d8e03fcb0922793ec56b1db5e82c134463101dd3d7fac2158e6af6cc5.jpg)  
factor 12

![](images/dadadc40ec9f9c21ac3d30dbc2585beffc506c3f1f68e8a8936b973fa22e4848.jpg)  
factor 13

![](images/d65c321b75dc41bf4e469a3d39766d31677f6a364c3991e1e9ae81fa32cbaa37.jpg)  
factor 19

![](images/eda08afa8c29e898c58304109cae2517b472b9fa8fed135901e52ab4da72d7aa.jpg)  
factor 26

![](images/3e49d5780bc01823f8d7c491f6f2db2a89bc53c5a58d4c9ae8d57ace6e31f0fe.jpg)  
factor 28

![](images/792fdc8e6123e44efb0291f04bd6e6e5b498ab89e5dd4e13ae5a2ec3be690ba0.jpg)  
factor 29

![](images/f313f7b5e8f5c74c8696ca6df55d7a582c13cd2babb6e3515137fee20a2300ba.jpg)  
factor 31

![](images/0cd54cf8146f34a44520a88f78e566c00584a18684a756656bfb1825e04a1a0d.jpg)  
factor 37

![](images/b6d1c3bae529784f95fee82d2defa558f4ac2fe31ddac1ae17565f7ff1266bea.jpg)  
factor 40

![](images/cd270cdcd3679ce26bfc520f6b5ffa4e923d217a51e520c58603f4220a1daa0e.jpg)  
factor 45

![](images/084e9010daa4d907895dd555d57b28e07f45109df3fd307a1af0a3dd4353a026.jpg)  
factor 48

![](images/79b3bee236ec8d0eca3f182883cfd8554a3d78357cd187abc7b0d89fe28287cb.jpg)  
factor 56

![](images/6ca2ec0d39e3db474e16b0e9fe52fd436342a2433c58cd61cf23cc6382229fd1.jpg)  
factor 63

![](images/7c126cce581dcf8dedf02b5d1a25e68d15818c5ef1a090e71997081ba95d8292.jpg)  
factor 65

![](images/1b683af72d39cfa442db3e96e1376ab4b572435921d45b83b179b7fb9abf1f3a.jpg)  
factor 73

![](images/7e40d5d3ace064f9120c59a1ba4c3cd9756b7572ebbb50cc60de5ba03f3627f3.jpg)  
factor 77

![](images/917853039e195c7e63b285664340cb09f6ab819f72e204209577713fbcb416da.jpg)  
factor 81

![](images/87aaa4b36461255534f075db9cd81b5397f268baf9ebd0d4ba4335f36de6123d.jpg)  
factor 82

![](images/16e261bc1dc4beff4ea3dc921556a1fb59a08d1ee777bc839962aaaea35c7b66.jpg)  
factor 88

![](images/7ea3ecc1558d416be082bcd388f71b95bf3141fd6fd74a449f37d69a41f30261.jpg)  
factor 90  
factor 110  
Figure 11: CelebA: Generating Factors Traversal

![](images/1dcfca10478c03d84afbf4db073f884abfea069d6a2737e18665a61e59308853.jpg)  
factor 96  
factor 118

![](images/326a77d4b482618a17d4a14416981d53a46c69fd3cdf60dfbe6b6fe75d52fd1d.jpg)  
factor 102  
factor 120

-  $\beta$ -VAE with noise modelling is able to automatically attain a relatively better distribution hypothesis and help achieve a better disentanglement performance as compared to the manipulation on the pre-specified noise ( $\beta$ -)VAE.

- Further, MoG  $\beta$ -VAE and can learn a better hypothesis distribution than the  $\beta$ -VAE with Gaussian noise modeling when the data noise distribution is complex.  
- Network parameterized noise VAE learns a more blurred generation and tends to suffer from the numerical instability though it can learn a good distribution hypothesis.  
- The Gaussian prior assumption contributes to the efficient coding of VAE model, and the idealistic VAE won't learn auxiliary dimension of generating factors.  
- The learned factors of the idealistic VAE exist an equivalence class under an orthogonal linear transformation, though the semantic factors can generate the data.  
- The mutual information  $I_{encoder}(x;z_h)$  is a good indicator to help determine the "used" generating factors.

We further try to give some discussions which should be beneficial to our future works on this work.

Firstly, from the perspective of noise modeling:

- The physical noise in different practical scenarios, such as medical image processing/generating, can be taken into consideration while implementing the VAE model.  
- The noise modelling for other generative model and deep model is also an interesting direction.

From the perspective of representation learning:

- It is interesting that the topology properties of oracle signal are used to obtain the proof for the information conservation theorem. Other situation including the data has several connected components can be further considered and would uncover the efficient coding properties of discrete factors.  
- The learnt factors' variance still exists a gap to the unit Gaussian prior, and it is unsatisfactory that the auxiliary constraint suppresses the  $I_{encoder}(x;z)$ . A better mechanism that is innocuous to other part of VAE but complies  $q(z)$  to follow the prior  $p(z)$  is still required to be investigated.

# REFERENCES

Dmitriy Aronov, Rhino Nevers, and David W Tank. Mapping of a non-spatial dimension by the hippocampal/entorhinal circuit. Nature, 543(7647):719, 2017.  
Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798-1828, 2013.  
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Info-gan: Interpretable representation learning by information maximizing generative adversarial nets. CoRR, abs/1606.03657, 2016. URL http://arxiv.org/abs/1606.03657.  
Yang Chen, Xiangyong Cao, Qian Zhao, Deyu Meng, and Zongben Xu. Denoising hyperspectral image with non-iid noise structure. arXiv preprint arXiv:1702.00098, 2017.  
Thomas M Cover and Joy A Thomas. Elements of information theory. John Wiley & Sons, 2012.  
Otto Fabius and Joost R van Amersfoort. Variational recurrent auto-encoders. arXiv preprint arXiv:1412.6581, 2014.  
A.S. Georgiades, P.N. Belhumeur, and D.J. Kriegman. From few to many: Illumination cone models for face recognition under variable lighting and pose. IEEE Trans. Pattern Anal. Mach. Intelligence, 23(6):643-660, 2001.  
Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016. http://www.deeplearningbook.org.

Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, and Daan Wierstra. Draw: A recurrent neural network for image generation. arXiv preprint arXiv:1502.04623, 2015.  
Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. 2016.  
Irina Higgins, Arka Pal, Andrei A Rusu, Loic Matthew, Christopher P Burgess, Alexander Pritzel, Matthew Botvinick, Charles Blundell, and Alexander Lerchner. Darla: Improving zero-shot transfer in reinforcement learning. arXiv preprint arXiv:1707.08475, 2017a.  
Irina Higgins, Nicolas Sonnerat, Loic Matthew, Arka Pal, Christopher P Burgess, Matthew Botvinick, Demis Hassabis, and Alexander Lerchner. Scan: Learning abstract hierarchical compositional visual concepts. arXiv preprint arXiv:1707.03389, 2017b.  
Daniel Jiwoong Im, Sungjin Ahn, Roland Memisevic, Yoshua Bengio, et al. Denoising criterion for variational auto-encoding framework. In AAAI, pp. 2059–2065, 2017.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Diederik P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. In Advances in Neural Information Processing Systems, pp. 4743-4751, 2016.  
Tejas D Kulkarni, William F Whitney, Pushmeet Kohli, and Josh Tenenbaum. Deep convolutional inverse graphics network. In Advances in Neural Information Processing Systems, pp. 2539-2547, 2015.  
Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building machines that learn and think like people. Behavioral and Brain Sciences, pp. 1-101, 2016.  
Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, and Ole Winther. Autoencoding beyond pixels using a learned similarity metric. arXiv preprint arXiv:1512.09300, 2015.  
Yann Lcun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
K.C. Lee, J. Ho, and D. Kriegman. Acquiring linear subspaces for face recognition under variable lighting. IEEE Trans. Pattern Anal. Mach. Intelligence, 27(5):684-698, 2005.  
Chongxuan Li, Jun Zhu, and Bo Zhang. Learning to generate with memory. In International Conference on Machine Learning, pp. 1177-1186, 2016.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision, pp. 3730-3738, 2015.  
Christos Louizos, Uri Shalit, Joris Mooij, David Sontag, Richard Zemel, and Max Welling. Causal effect inference with deep latent-variable models. arXiv preprint arXiv:1705.08821, 2017.  
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, and Ian J. Goodfellow. Adversarial autoencoders. CoRR, abs/1511.05644, 2015. URL http://arxiv.org/abs/1511.05644.  
Michael F Mathieu, Junbo Jake Zhao, Junbo Zhao, Aditya Ramesh, Pablo Sprechmann, and Yann LeCun. Disentangling factors of variation in deep representation using adversarial training. In Advances in Neural Information Processing Systems, pp. 5040-5048, 2016.  
Deyu Meng and Fernando De La Torre. Robust matrix factorization with unknown noise. In IEEE International Conference on Computer Vision, pp. 1337-1344, 2014.  
Tobias Plotz and Stefan Roth. Benchmarking denoising algorithms with real photographs. CoRR, abs/1707.01313, 2017. URL http://arxiv.org/abs/1707.01313.  
Danilo Rezende, Ivo Danihelka, Karol Gregor, Daan Wierstra, et al. One-shot generalization in deep generative models. In International Conference on Machine Learning, pp. 1521-1529, 2016.

Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. arXiv preprint arXiv:1505.05770, 2015.  
Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. arXiv preprint arXiv:1401.4082, 2014.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, pp. 2234-2242, 2016.  
Masahiro Suzuki, Kotaro Nakayama, and Yutaka Matsuo. Joint multimodal learning with deep generative models. arXiv preprint arXiv:1611.01891, 2016.  
Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of Machine Learning Research, 11(Dec):3371-3408, 2010.  
Jacob Walker, Carl Doersch, Abhinav Gupta, and Martial Hebert. An uncertain future: Forecasting from static images using variational autoencoders. In European Conference on Computer Vision, pp. 835-851. Springer, 2016.  
Ziyu Wang, Josh Merel, Scott Reed, Greg Wayne, Nando de Freitas, and Nicolas Heess. Robust imitation of diverse behaviors. arXiv preprint arXiv:1707.02747, 2017.  
Wei Wei, Lixuan Yi, Qi Xie, Qian Zhao, Deyu Meng, and Zongben Xu. Should we encode rain streaks in video as deterministic or stochastic? In The IEEE International Conference on Computer Vision (ICCV), Oct 2017.  
Wikipedia. Mental image — wikipedia, the free encyclopedia, 2017. URL https://en.wikipedia.org/w/index.php?title=Mental_image&oldid=798962875. [Online; accessed 6-October-2017].  
Hongwei Yong, Deyu Meng, Wangmeng Zuo, and Lei Zhang. Robust online matrix factorization for dynamic background subtraction. arXiv preprint arXiv:1705.10000, 2017.  
Qian Zhao, Deyu Meng, Zongben Xu, Wangmeng Zuo, and Lei Zhang. Robust principal component analysis with complex noise. In International Conference on Machine Learning, pp. 55-63, 2014.  
Shengjia Zhao, Jiaming Song, and Stefano Ermon. Learning hierarchical features from generative models. arXiv preprint arXiv:1702.08396, 2017.
