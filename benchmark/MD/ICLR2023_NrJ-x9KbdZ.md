# YOUR DENOISING IMPLICIT MODEL IS A SUBOPTIMAL ENSEMBLE OF DENOISING PREDICTIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Denoising diffusion models construct a Markov denoising process to learn the transport from Gaussian noise distribution to the data distribution, however require thousands of denoising steps to achieve the SOTA generative performance. Denoising diffusion implicit models (DDIMs) introduce non-Markovian process to largely reduce the required steps, but its performance degenerates as the sampling steps further reducing. In this work, we show that DDIMs belong to our ensemble denoising implicit models which heavily rely on the convex ensemble of obtained denoising predictions. We propose improved DDIM (iDDIM) to demonstrate DDIMs adopt sub-optimal ensemble coefficients. The iDDIM can largely improve on DDIMs, but still deteriorates in the case of a few sampling steps. Thus we further propose generalized denoising implicit model (GDIM) that replace the ensemble prediction with a probabilistic inference conditioned on the obtained states. Then a specific instance  $t$ -GDIM that only depends on the latest state is parameterized by the conditional energy-based model (EBM) and variational sampler. The models are jointly trained with variational maximum likelihood. Extensive experiments show  $t$ -GDIM can reduces the sampling steps to only 4 and remains comparable generative quality to other generative models.

# 1 INTRODUCTION

Modern deep generative modelling focuses on learning transport from a tractable reference distribution (e.g. Gaussian) to the target distribution, and the learned transport is applied on reference samples to generate new data on the sampling stage. Among them, implicit generative model (IGM, Mohamed & Lakshminarayanan 2016) is the simplest one that directly mapping the reference samples to data through neural network. Sampling from IGMs requires only once forward evaluation of network. However, commonly used training algorithms for IGMs like generative adversarial networks (GANs, Goodfellow et al. 2014) meet the challenges of poor mode coverage and unstable optimization. The reason may be that, training the direct mapping to characterize the complex transport is difficult, since they lack of intermediate structural assumptions.

Recently, researchers focus on diffusion probabilistic model (DPM, Sohl-Dickstein et al. 2015; Ho et al. 2020), a well-specified probabilistic transport that constructs a generative Markov chain with its marginal distribution evolving from the Gaussian noise distribution into the data distribution. To accomplish it, DPM first gradually imposes Gaussian noise into the data samples with fixed noise scales, producing a Markov forward diffusion process. And the reversal of which, a Markov reverse process, is regarded as the learning target. DPM assumes the variance scale of each Gaussian forward kernel is small enough, leading to a Gaussian reverse process that is tractable for generative denoising process to learn. DPMs achieve impressive image generative quality even comparable with SOTA GANs (Dhariwal & Nichol, 2021). Nevertheless, the small noise scale assumption incurs quite long diffusion chains, resulting in far less efficient sampling process than IGMs.

To circumvent the small noise scale assumption, Song et al. (2021a) generalize the forward process in DPM to a non-Markovian one. The new forward process is represented by an inference process that, first infers the terminal state given data sample and then gradually infers the rest states along the reverse direction conditioned on data sample. A corresponding generative process is then constructed by replacing the conditional data sample with denoising predictions. The general process is proved to be an alternative sampling scheme for DPM. Song et al. (2021a) thus propose denoising

diffusion implicit model (DDIM), an implicit variant of the general process, that can speed up  $20\times$  over DPM with similar generative quality. However, it remains inferior with fewer sampling steps.

In this work, we introduce a novel perspective on DDIM that the generative process relies heavily on the convex combination of obtained denoising predictions. Thus DDIM belongs to a general class of ensemble denoising implicit models whose convex coefficients can be adjusted flexibly (Sec. 3.1). It reveals the nature of each denoising step in ensemble models is predicting the denoising target with ensemble denoising prediction. Further we introduce iDDIM, an intuition guided ensemble model that allocates more trust on the latest denoising prediction based on DDIM (Sec. 3.2). Experiments on CIFAR10 indicate iDDIM largely improves on baseline DDIM especially in the case of fewer generative iterations, and convince that DDIM adopts sub-optimal convex coefficients.

However iDDIM still fails to generate realistic samples when further reducing the sampling steps. We find the reason is that the parameterization in iDDIM is unable to obtain good denoising targets with just a few denoising steps. To obtain better denoising targets, we instead propose generalized denoising implicit model (GDIM), a general probabilistic extension to the ensemble model that replaces the ensemble denoising prediction with a probabilistic inference conditioned on obtained states (Sec. 4). Finally we provide a specific choice that only relies on the current state radically, termed  $t$ -GDIM (Sec. 4.1). Conditional energy-based models (EBM, LeCun et al. 2006) and IGMs are used to construct  $t$ -GDIM, and are jointly trained with variational maximum likelihood (Grathwohl et al., 2021) (Sec. 4.2). Moreover, the iDDIM can be regarded as an ensemble augmentation trick which leverages predictions at previous steps. Experiments on various resolution image datasets show our  $t$ -GDIM+iDDIM can largely reduce the number of sampling steps to only 4, and still achieves high generative quality comparable to diffusion models or other generative models.

# 2 BACKGROUND

DPM (Sohl-Dickstein et al., 2015) typically specifies a Markov forward diffusion process converting the data distribution  $q(\mathbf{x}_0)$  into a terminal state  $q(\mathbf{x}_T)$  that is closed to tractable prior  $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$ . It is achieved by repeated application of a Gaussian diffusion kernel  $q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$ , i.e., gradually imposing noise into data samples with fixed variance scales  $\beta_i, i = 1, \dots, T$ . Then DPM defines a generative denoising process to simulate the reverse of the forward process with Gaussian denoising kernel  $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})$ .

However, the feasibility of approximation comes up against commonly non-Gaussian reverse kernel  $q(\mathbf{x}_{t - 1}|\mathbf{x}_t)$ , unless the noise scale  $\beta_{t}$  is small enough. To keep noise scale small, DPM requires pretty long diffusion chain (  $\sim 1\mathrm{K}$  steps), largely degenerating the training and sampling efficiency. To reduce the length of sampling chain, Song et al. (2021a) introduce a class of non-Markovian forward processes indexed by  $\sigma \in \mathbb{R}_{\geq 0}^{T}$ , characterized by the following inference process:

$$
q _ {\sigma} \left(\mathbf {x} _ {1: T} \mid \mathbf {x} _ {0}\right) = q _ {\sigma} \left(\mathbf {x} _ {T} \mid \mathbf {x} _ {0}\right) \prod_ {t = 2} ^ {T} q _ {\sigma} \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {t}, \mathbf {x} _ {0}\right), \quad q _ {\sigma} \left(\mathbf {x} _ {T} \mid \mathbf {x} _ {0}\right) = \mathcal {N} \left(\mathbf {x} _ {T}; \sqrt {\alpha_ {T}} \mathbf {x} _ {0}, (1 - \alpha_ {T}) \mathbf {I}\right), \tag {1}
$$

$$
q _ {\sigma} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}, \mathbf {x} _ {0}) = \mathcal {N} (\mathbf {x} _ {t - 1}; \sqrt {\alpha_ {t - 1}} \mathbf {x} _ {0} + \sqrt {1 - \alpha_ {t - 1} - \sigma_ {t} ^ {2}} \cdot \frac {\mathbf {x} _ {t} - \sqrt {\alpha_ {t}} \mathbf {x} _ {0}}{\sqrt {1 - \alpha_ {t}}}, \sigma_ {t} ^ {2} \mathbf {I}),
$$

where the Gaussian form of  $q_{\sigma}(\mathbf{x}_{t - 1}|\mathbf{x}_t,\mathbf{x}_0)$  free the forward process from Gaussian assumption. It's proved the marginal posteriors are the same as that in DPMs:  $q_{\sigma}(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\alpha_t}\mathbf{x}_0,(1 - \alpha_t)\mathbf{I})$ , where  $\alpha_{t} = \prod_{i = 1}^{t}1 - \beta_{i}$ . Then a corresponding generative process is defined as<sup>1</sup>:

$$
p _ {\theta} (\mathbf {x} _ {0: T}) = p (\mathbf {x} _ {T}) \prod_ {t = 1} ^ {T} p _ {\theta} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}), \quad p _ {\theta} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}) = q _ {\sigma} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}, \mathbf {x} _ {0} ^ {t}), \tag {2}
$$

where  $\mathbf{x}_0^t = \pmb{f}_{\theta}(\mathbf{x}_t,t)$  denotes the denoising prediction of  $\mathbf{x}_0$  from  $\mathbf{x}_t$  to meet with Eq. (1). Song et al. (2021a) find that training Eq. (2) with variational inference (Kingma & Welling, 2014) objective is equivalent to optimizing that of DPMs, from the perspective of global optimal solution. So Eq. (2) becomes a class of alternative sampling scheme to DPMs.

Specifically, Song et al. (2021a) focus on DDIM, an implicit generative process composed of deterministic transformations  $p_{\theta}(\mathbf{x}_{t - 1}|\mathbf{x}_t)$  in the case of  $\sigma_t = 0$ . Then DDIM is trained to fit a deterministic path from  $\mathbf{x}_0$  to  $\mathbf{x}_T$  characterized by the Dirac distributions (set  $\sigma_t = 0$  in Eq. (1)):

$$
q (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}, \mathbf {x} _ {0}) = \delta (\mathbf {x} _ {t - 1} - \left[ \sqrt {\alpha_ {t - 1}} \mathbf {x} _ {0} + \sqrt {1 - \alpha_ {t - 1}} \cdot \frac {\mathbf {x} _ {t} - \sqrt {\alpha_ {t}} \mathbf {x} _ {0}}{\sqrt {1 - \alpha_ {t}}} \right]). (3)
$$

Please see Appendix A for more detailed review and discussion.

# 3 ENSEMBLE OF DENOISING PREDICTIONS

Thanks to the non-Markovian inference process, DDIM can speed up  $20\times$  over denoising DPM (DDPM, Ho et al. 2020) with similar high performance, it nevertheless degenerates when the number of sampling steps is no more than 20. In order to mitigate it, in this section, we explore how the denoising predictions  $\mathbf{x}_0^{1:T}$  are leveraged to accomplish the generative process, as  $\mathbf{x}_0^t = \pmb{f}_{\theta}(\mathbf{x}_t,t)$  is the key for implementing  $p_{\theta}(\mathbf{x}_{t - 1}|\mathbf{x}_t)$ . Our key observation is that each  $\mathbf{x}_{t - 1}$  along the generative process in DDIM depends on a specific convex combination of  $\mathbf{x}_0^{t:T}$ . This leads to a general denoising implicit model which is an ensemble of the denoising predictions  $\mathbf{x}_0^{1:T}$ . Then the experiments on CIFAR10 indicate the coefficients used in DDIM are not optimal especially when  $T$  is small.

# 3.1 ENSEMBLE DENOISING IMPLICIT MODELS

To show our novel perspective, let us define the ensemble denoising implicit models indexed by  $\omega_{t} = [\omega_{t}^{t},\ldots ,\omega_{t}^{T}]\in \mathbb{R}_{\geq 0}^{T - t + 1}$ , characterized by the deterministic transformation  $q_{\omega}(\mathbf{x}_{t - 1}|\mathbf{x}_0^{t:T},\mathbf{x}_T)$ :

$$
\mathbf {x} _ {t - 1} = B _ {t - 1} \cdot \sum_ {k = t} ^ {T} \frac {\omega_ {t} ^ {k}}{\sum_ {k = t} ^ {T} \omega_ {t} ^ {k}} \mathbf {x} _ {0} ^ {k} + C _ {t - 1} \mathbf {x} _ {T} = B _ {t - 1} \bar {\mathbf {x}} _ {0} ^ {t} + C _ {t - 1} \mathbf {x} _ {T}, \tag {4}
$$

where  $\bar{\mathbf{x}}_0^t$  denotes the convex ensemble of denoising predictions  $\mathbf{x}_0^{t:T}$ , and  $B_{t-1}, C_{t-1}$  are set to:

$$
B _ {t - 1} = \sqrt {\alpha_ {t - 1}} - \sqrt {1 - \alpha_ {t - 1}} \frac {\sqrt {\alpha_ {T}}}{\sqrt {1 - \alpha_ {T}}}, \quad C _ {t - 1} = \frac {\sqrt {1 - \alpha_ {t - 1}}}{\sqrt {1 - \alpha_ {T}}} \tag {5}
$$

to match up with the inference process (3) as shown later. We find the DDIM denoising kernel represented by the deterministic transformation  $q(\mathbf{x}_{t - 1}|\mathbf{x}_t,\mathbf{x}_0^t)$  (in the case of  $\sigma_t = 0$  in Eq. (2)):

$$
\mathbf {x} _ {t - 1} = \sqrt {\alpha_ {t - 1}} \mathbf {x} _ {0} ^ {t} + \sqrt {1 - \alpha_ {t - 1}} \cdot \frac {\mathbf {x} _ {t} - \sqrt {\alpha_ {t}} \mathbf {x} _ {0} ^ {t}}{\sqrt {1 - \alpha_ {t}}}, \tag {6}
$$

is a linear combination of  $\mathbf{x}_t$  and  $\mathbf{x}_0^t$ . Since  $\mathbf{x}_t$  is also a combination of  $\mathbf{x}_{t+1}$  and  $\mathbf{x}_0^{t+1}$ , we can recursively expand the particles  $\mathbf{x}_k$  along  $t \to T$  and obtain the following result:

Proposition 1. Denoising diffusion implicit model (6) can be reformulated as  $q(\mathbf{x}_{t - 1}|\mathbf{x}_0^{t:T},\mathbf{x}_T)$ :

$$
\mathbf {x} _ {t - 1} = \sqrt {1 - \alpha_ {t - 1}} \cdot \sum_ {k = t} ^ {T} \left(A _ {k - 1} - A _ {k}\right) \mathbf {x} _ {0} ^ {k} + \frac {\sqrt {1 - \alpha_ {t - 1}}}{\sqrt {1 - \alpha_ {T}}} \mathbf {x} _ {T}, \quad A _ {k} = \frac {\sqrt {\alpha_ {k}}}{\sqrt {1 - \alpha_ {k}}}, \tag {7}
$$

and is a specific instance of ensemble denoising implicit models (4) with  $\omega_{t}^{k} = A_{k - 1} - A_{k}$ .

We include a general proof in Appendix B.1. Proposition 1 demonstrates that the ensemble denoising implicit models are generalized DDIMs, and they are all first computing  $\bar{\mathbf{x}}_0^t$  with a convex combination of  $\mathbf{x}_0^{t:T}$  and then using linear combination with  $\mathbf{x}_T$  to obtain  $\mathbf{x}_{t-1}$ . But which  $\bar{\mathbf{x}}_0^t$  is the best for the general ensemble models? To answer this, we notice in DDIM, the denoising predictions  $\mathbf{x}_0^{t:T}$  are trained to approximate the same real sample  $\mathbf{x}_0 \sim q(\mathbf{x}_0|\mathbf{x}_T)$  given  $\mathbf{x}_{t:T}$ . So that if we let  $\mathbf{x}_0^{t:T} = \mathbf{x}_0$  in the ensemble denoising model (4), it turns into  $q(\mathbf{x}_{t-1}|\mathbf{x}_0,\mathbf{x}_T)$ :

$$
\mathbf {x} _ {t - 1} = B _ {t - 1} \mathbf {x} _ {0} + C _ {t - 1} \mathbf {x} _ {T} = \sqrt {\alpha_ {t - 1}} \mathbf {x} _ {0} + \sqrt {1 - \alpha_ {t - 1}} \cdot \frac {\mathbf {x} _ {T} - \sqrt {\alpha_ {T}} \mathbf {x} _ {0}}{\sqrt {1 - \alpha_ {T}}}. \tag {8}
$$

Equation (8) forms a deterministic path between  $\mathbf{x}_0$  and  $\mathbf{x}_T$ , which is exactly the inference process in DDIM (3), but is rewritten into a more proper equivalent form:

$$
q \left(\mathbf {x} _ {1: T} \mid \mathbf {x} _ {0}\right) = q \left(\mathbf {x} _ {T} \mid \mathbf {x} _ {0}\right) \prod_ {t = 2} ^ {T} q \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {0}, \mathbf {x} _ {T}\right). \tag {9}
$$

![](images/b07ec11e796cbde606f82e1a2153e0739ea9cd4cd7a087f16017ccba91f23c88.jpg)  
Figure 1: Design intuition (left) and graphical description (right) for improved DDIM. The ensemble model (yellow line) uses linear combination of  $\bar{\mathbf{x}}_0^{t + 1}$  and  $\mathbf{x}_T$  to obtain  $\mathbf{x}_t$ . Then, DDIM (blue line) takes a denoising step along the deterministic path from  $\mathbf{x}_t$  to  $\mathbf{x}_0^t$ , while radical ensemble model (red line) along the path from  $\mathbf{x}_T$  to  $\mathbf{x}_0^t$ . If  $\mathbf{x}_0^t$  is closer to some real  $\mathbf{x}_0$  than  $\bar{\mathbf{x}}_0^{t + 1}$ , the current denoising step of radical ensemble model (red line) becomes closer to the real path.

So in other words, the ensemble denoising implicit models are trained to fit the same target process as in DDIM, providing a class of alternative sampling schemes. And what's more, ensemble models (include DDIM) are essentially predicting the real sample  $\mathbf{x}_0$  with  $\bar{\mathbf{x}}_0^t$  at each denoising step, by means of leveraging the convex ensemble of the denoising predictions  $\mathbf{x}_0^{t:T}$ . However in practice,  $\mathbf{x}_0^{t:T}$  are commonly different, let alone be equal to  $\mathbf{x}_0 \sim q(\mathbf{x}_0|\mathbf{x}_T)$ . This leads to the ensemble denoising prediction  $\bar{\mathbf{x}}_0^t$  not always be like some  $\mathbf{x}_0$ . In this work, the proposed ensemble denoising implicit model provides a flexible way to combine  $\mathbf{x}_0^{t:T}$  for better ensemble prediction  $\bar{\mathbf{x}}_0^t$ , potentially results in a generative process closer to the target inference process.

# 3.2 SUB-OPTIMAL COEFFICIENTS IN DDIM

However, finding out the optimal coefficients in ensemble models is intractable as we know nothing about how the performance of  $\mathbf{x}_0^{t:T}$  contributes to the additive ensemble prediction  $\bar{\mathbf{x}}_0^t$ . Since denoising  $\mathbf{x}_k$  becomes more difficult along  $k \to T$ , the latest denoising prediction  $\mathbf{x}_0^t$  is intuitively more precise than  $\mathbf{x}_0^{t+1:T}$  and thus more precise than  $\bar{\mathbf{x}}_0^{t+1}$ . If we let  $\omega_t = [1, 0, \dots, 0]$ , the ensemble model (4) will only trust the latest  $\mathbf{x}_0^t$  radically, and the resulting radical ensemble model is presented as  $q(\mathbf{x}_{t-1} | \mathbf{x}_0^t, \mathbf{x}_T)$ :

$$
\mathbf {x} _ {t - 1} = B _ {t - 1} \mathbf {x} _ {0} ^ {t} + C _ {t - 1} \mathbf {x} _ {T} = \sqrt {\alpha_ {t - 1}} \mathbf {x} _ {0} ^ {t} + \sqrt {1 - \alpha_ {t - 1}} \cdot \frac {\mathbf {x} _ {T} - \sqrt {\alpha_ {T}} \mathbf {x} _ {0} ^ {t}}{\sqrt {1 - \alpha_ {T}}}. \tag {10}
$$

Inspired by the intuition (Fig. 1, left) that trusting more on  $\mathbf{x}_0^t$  may bring about a generative process closer to a real one, we introduce an improved DDIM (iDDIM) where  $\mathbf{y}_{t-1}$  and  $\mathbf{z}_{t-1}$  are computed with Eqs. (6) and (10) respectively (Fig. 1, right):

$$
\mathbf {x} _ {t - 1} = \left(1 - m _ {t}\right) \mathbf {y} _ {t - 1} + m _ {t} \mathbf {z} _ {t - 1}. \tag {11}
$$

Equation (11) actually comes from replacing  $m_t \in [0, 1]$  proportion of  $\mathbf{x}_0^{t+1:T}$  with  $\mathbf{x}_0^t$  in DDIM, so it is still an ensemble denoising implicit model. We include derivations in Appendix B.2. The iDDIM behaves as an interpolation between DDIM (6) and radical ensemble model (10). And as  $m_t \to 1$ , it allocates more trust on  $\mathbf{x}_0^t$  as expected.

In Sec. 6.1, we conduct experiments on CIFAR10 to explore how the performance of iDDIM influenced by varying  $m_{t}$ . The results demonstrate that, DDIM adopts sub-optimal coefficients and allocating higher proportion  $(\omega_{t}^{t})$  on  $\mathbf{x}_0^t$  achieves prominent improvement especially as  $T$  decreasing.

# 4 GENERALIZED DENOISING IMPLICIT MODELS

As we emphasized in the previous section, in order to fit the deterministic inference process characterized by Eq. (9), the ensemble denoising implicit models (4) are actually predicting  $\mathbf{x}_0$  with the ensemble prediction  $\bar{\mathbf{x}}_0^t$  at each step. Thus the performance of generative process largely rests with the alignment between  $\bar{\mathbf{x}}_0^t$  and  $\mathbf{x}_0$ . We have verified that carefully selecting the coefficients  $\omega_{t}$  does produce better  $\bar{\mathbf{x}}_0^t$ , however, the misalignment between  $\bar{\mathbf{x}}_0^t$  and  $\mathbf{x}_0$  still remains and is exacerbated when  $T$  further reducing (see Fig. 4). It is because  $\mathbf{x}_0^t = f_\theta (\mathbf{x}_t,t)$  is a Dirac approximation of

![](images/a867b14ce46cc8c81898dc9076e8952183632683c96eaa94db50131a3e8b9c0d.jpg)  
Figure 2: Denoising target  $\bar{\mathbf{x}}_0^t$  in the ensemble denoising implicit model and the generalized denoising implicit model. Ensemble model uses convex combination of potentially inferior  $\mathbf{x}_0^{t:T}$  to obtain blurry  $\bar{\mathbf{x}}_0^t$ , while GDIM leverages probabilistic inference  $p_{\theta}(\bar{\mathbf{x}}_0^t|\mathbf{x}_{t:T})$  to get better  $\bar{\mathbf{x}}_0^t$  directly.

$q(\mathbf{x}_0^t|\mathbf{x}_t)$ , i.e.,  $p_{\theta}(\mathbf{x}_0^t|\mathbf{x}_t) = \delta (\mathbf{x}_0^t -\mathbf{f}_{\theta}(\mathbf{x}_t,t))$ . As shown in Xiao et al. (2022), this deterministic parameterization struggles with the commonly multimodal  $q(\mathbf{x}_0^t|\mathbf{x}_t)$  as  $t\to T$ , brings about potentially inferior  $\mathbf{x}_0^t$  (see Fig. 2 for instance). As a result, ensemble models require more steps and carefully coefficients seeking to get gradually better  $\bar{\mathbf{x}}_0^t$  along the generative process. In order to obtain more realistic  $\bar{\mathbf{x}}_0^t$  at each denoising step, we propose to replace the convex ensemble of  $\mathbf{x}_0^{t:T}$  with probabilistic inference conditioned on  $\mathbf{x}_{t:T}$ , i.e.,  $p_{\theta}(\bar{\mathbf{x}}_0^t|\mathbf{x}_{t:T})$ . This leads to the generalized denoising implicit model (GDIM):

$$
p _ {\theta} \left(\mathbf {x} _ {0: T}\right) = p \left(\mathbf {x} _ {T}\right) \prod_ {t = 1} ^ {T} p _ {\theta} \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {t: T}\right), p _ {\theta} \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {t: T}\right) = \int p _ {\theta} \left(\bar {\mathbf {x}} _ {0} ^ {t} \mid \mathbf {x} _ {t: T}\right) q \left(\mathbf {x} _ {t - 1} \mid \bar {\mathbf {x}} _ {0} ^ {t}, \mathbf {x} _ {T}\right) d \bar {\mathbf {x}} _ {0} ^ {t}. \tag {12}
$$

The GDIM is a general extension to the ensemble models as they share the same spirit that, first predicting a current denoising target  $\bar{\mathbf{x}}_0^t$  given obtained states  $\mathbf{x}_{t:T}$  and then taking one denoising step to  $\mathbf{x}_{t-1}$  via deterministic transform (8). See Fig. 2 for comparison. While the benefit is that GDIM can directly predicts  $\bar{\mathbf{x}}_0^t$  via  $p_\theta(\bar{\mathbf{x}}_0^t | \mathbf{x}_{t:T})$  represented by some expressive probabilistic models, and thus the dependence on  $\mathbf{x}_{t:T}$  (corresponding to the coefficients in ensemble models) is learned adaptively. More importantly, if  $p_\theta(\bar{\mathbf{x}}_0^t | \mathbf{x}_{t:T})$  is properly trained to generate good  $\bar{\mathbf{x}}_0^t$ , the denoising process will no longer need many steps as that in ensemble models.

# 4.1 RADICAL GDIM

Notice the GDIM is autoregressive and enables us to flexible choose which  $\mathbf{x}_{t:T}$  does  $p_{\theta}(\bar{\mathbf{x}}_0^t|\mathbf{x}_{t:T})$  conditioned on. We next discuss a specific choice that only relies on  $\mathbf{x}_t$  radically, i.e.,  $p_{\theta}(\mathbf{x}_0^t|\mathbf{x}_t)$ . It seems to be a probabilistic counterpart to the radical ensemble model (10) and is termed  $t$ -GDIM:

$$
p _ {\theta} \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {t}, \mathbf {x} _ {T}\right) = \int p _ {\theta} \left(\mathbf {x} _ {0} ^ {t} \mid \mathbf {x} _ {t}\right) q \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {0} ^ {t}, \mathbf {x} _ {T}\right) \mathrm {d} \mathbf {x} _ {0} ^ {t}, \tag {13}
$$

We adopt expressive conditional energy-based model (EBM) to represent the denoising distribution:

$$
p _ {\theta} \left(\mathbf {x} _ {0} ^ {t} \mid \mathbf {x} _ {t}\right) = \frac {\exp \left(- E _ {\theta} \left(\mathbf {x} _ {0} ^ {t} , \mathbf {x} _ {t}\right)\right)}{\int \exp \left(- E _ {\theta} \left(\mathbf {x} _ {0} ^ {t} , \mathbf {x} _ {t}\right)\right) \mathrm {d} \mathbf {x} _ {0} ^ {t}} = \frac {\exp \left(- E _ {\theta} \left(\mathbf {x} _ {0} ^ {t} , \mathbf {x} _ {t}\right)\right)}{Z (\theta , \mathbf {x} _ {t})}, \tag {14}
$$

where  $E_{\theta}:\mathcal{X}\times \mathcal{X}\times \mathbb{R}\to \mathbb{R}$  denotes the joint energy over  $\mathbf{x}_0^t$  and  $\mathbf{x}_t$ , and the dependence on  $t$  is not displayed for brevity. The same inference process as that in ensemble models (9) is regarded as the learning target for  $t$ -GDIM. Then we again optimize  $\theta$  with the variational inference (Kingma & Welling, 2014) objective:

$$
\begin{array}{l} - \mathbb {E} _ {q (\mathbf {x} _ {0})} [ \log p _ {\theta} (\mathbf {x} _ {0}) ] \leq \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \left[ \log \frac {q \left(\mathbf {x} _ {1 : T} \mid \mathbf {x} _ {0}\right)}{p _ {\theta} \left(\mathbf {x} _ {0 : T}\right)} \right] \tag {15} \\ \dot {=} \sum_ {t = 1} ^ {T} \mathbb {E} _ {q \left(\mathbf {x} _ {0}, \mathbf {x} _ {t - 1}, \mathbf {x} _ {t}, \mathbf {x} _ {T}\right)} \left[ E _ {\theta} \left(T ^ {- 1} \left(\mathbf {x} _ {t - 1}; \mathbf {x} _ {T}\right), \mathbf {x} _ {t}\right) + \log Z (\theta , \mathbf {x} _ {t}) \right], \\ \end{array}
$$

where the diffeomorphism  $T(\mathbf{x}_0; \mathbf{x}_T) = \mathbf{x}_{t-1}$  stands for the deterministic transform  $q(\mathbf{x}_{t-1} | \mathbf{x}_0, \mathbf{x}_T)$ . For convenience, we use  $\mathcal{J}(\theta, t)$  to represent the energy objective at each time step  $t$ . However, computing  $Z(\theta, \mathbf{x}_t)$  requires intractable integral over the whole space, and fortunately, a more efficient

alternate is to estimate the optimizing gradients:

$$
\nabla_ {\theta} \mathcal {J} (\theta , t) = \mathbb {E} _ {q \left(\mathbf {x} _ {0}\right) q \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {0}\right) p _ {\theta} \left(\mathbf {x} _ {0} ^ {t} \mid \mathbf {x} _ {t}\right)} \left[ \nabla_ {\theta} E _ {\theta} \left(\mathbf {x} _ {0}, \mathbf {x} _ {t}\right) - \nabla_ {\theta} E _ {\theta} \left(\mathbf {x} _ {0} ^ {t}, \mathbf {x} _ {t}\right) \right]. \tag {16}
$$

Notice it resembles the gradient of maximum likelihood objective for learning EBMs (LeCun et al., 2006), and an unbiased Monte Carlo gradient estimator can be accomplished by sampling a batch of  $(\mathbf{x}_0, \mathbf{x}_t, \mathbf{x}_0^t)$  at each training iteration. See Appendix C.1 for derivations.

# 4.2 SAMPLING FROM ENERGY-BASED DENOISING DISTRIBUTION

The gradient estimator commonly suffers from sampling from unnormalized distributions, e.g. Eq. (14). Recent attempts (Du & Mordatch, 2019; Nijkamp et al., 2020b) resort to dynamic-based Markov chain Monte Carlo (MCMC), but fall into the trouble of mixing and again requires lots of sampling steps. Here we amortize the MCMC sampling into training conditional IGMs constructed by  $\mathbf{x}_0^t = G_{\phi}(\mathbf{u};\mathbf{x}_t,t),\mathbf{u}\sim \mathcal{N}(\mathbf{0},\mathbf{I})$ . In other words, the expressive conditional IGMs are used as approximate samplers  $p_{\phi}(\mathbf{x}_0^t |\mathbf{x}_t) = \int \delta (\mathbf{x}_0^t -G_{\phi}(\mathbf{u};\mathbf{x}_t,t))\mathcal{N}(\mathbf{u};\mathbf{0},\mathbf{I})\mathrm{d}\mathbf{u}$ , and are trained by minimizing the following KL divergence with respect to  $\phi$  (see Appendix C.2 for derivations):

$$
D _ {\mathrm {K L}} \left(p _ {\phi} \left(\mathbf {x} _ {0} ^ {t} \mid \mathbf {x} _ {t}\right) \mid p _ {\theta} \left(\mathbf {x} _ {0} ^ {t} \mid \mathbf {x} _ {t}\right)\right) \doteq \mathbb {E} _ {\mathcal {N} (\mathbf {u}; \mathbf {0}, \mathbf {I})} \left[ E _ {\theta} \left(G _ {\phi} \left(\mathbf {u}; \mathbf {x} _ {t}\right), \mathbf {x} _ {t}\right) \right] - \mathcal {H} \left(p _ {\phi} \left(\mathbf {x} _ {0} ^ {t} \mid \mathbf {x} _ {t}\right)\right). \tag {17}
$$

This variational approximation can be incorporated in  $\mathcal{J}(\theta, t)$  as an additional inner optimization which is similar to the variational maximum likelihood (Grathwohl et al., 2021). And the resulting nested objective is commonly handled with alternating optimization. Notice that  $\mathcal{H}(p_{\phi}(\mathbf{x}_0^t | \mathbf{x}_t))$  is the entropy of sampler and is typically difficult to optimize. If we ignore this entropy term, the nested optimization becomes similar to WGAN (Arjovsky et al., 2017):

$$
\left. \min  _ {\theta} \max  _ {\phi} \left\{\mathbb {E} _ {q \left(\mathbf {x} _ {0}, \mathbf {x} _ {t}\right) \mathcal {N} (\mathbf {u}; \mathbf {0}, \mathbf {I})} \left[ E _ {\theta} \left(\mathbf {x} _ {0}, \mathbf {x} _ {t}\right) - E _ {\theta} \left(G _ {\phi} (\mathbf {u}; \mathbf {x} _ {t}), \mathbf {x} _ {t}\right) \right] \right\}. \right. \tag {18}
$$

Therefore, we can borrow the proven optimizing technique from GANs for jointly training the conditional EBM  $p_{\theta}(\mathbf{x}_0^t | \mathbf{x}_t)$  and its variational sampler  $p_{\phi}(\mathbf{x}_0^t | \mathbf{x}_t)$ .

# 5 RELATED WORK AND DISCUSSION

Score-based generative model (SGM). As shown to have interesting connection with denoising score matching (Vincent, 2011), DPMs as well as noise conditional score networks (NCSN, Song & Ermon 2019; 2020) are usually referred to together as SGMs. Song et al. (2021c) further propose a unified forward-reverse stochastic differential equation (SDE) framework that treats them as discretizations of specific SDEs. After that, lots of works explore the intrinsic properties or numerical approximations of different SDEs to improve the generative quality and the sampling efficiency (Dockhorn et al., 2022; Jolicoeur-Martineau et al., 2021a; Lu et al., 2022; Liu et al., 2022). However, they still generate inferior samples when further reduce the number of sampling steps. It is possibly because the numerical simulation for SDEs always assumes the discretization steps are small, and the case of only few sampling iterations violates the assumption.

Accelerate sampling. Besides, there are lots of other studies focusing on accelerating the sampling process for DPMs (Kong & Ping, 2021; Watson et al., 2021; Lyu et al., 2022; Nachmani et al., 2021; Zheng et al., 2022). They are all basically orthogonal to us since they do not involve the inherent understanding of how DPMs (or DDIMs) leverage the denoising predictions  $\mathbf{x}_0^{t:T}$ . Watson et al. (2022) propose a similar method to our ensemble model, which use a combination of obtained state  $\mathbf{x}_{t:T}$  to represent DPMs, and learn the coefficients by differentiating through the sample quality. But we point out that the denoising predictions rather than obtained states are the keys in nature.

Discussion. Notice the objective for  $t$ -GDIM (18) does not depend on  $\mathbf{x}_T$ , and  $p_{\theta}(\mathbf{x}_0^t | \mathbf{x}_t)$  at different step  $t$  are trained to approximate corresponding  $q(\mathbf{x}_0 | \mathbf{x}_t)$  rather than the same  $q(\mathbf{x}_0 | \mathbf{x}_T)$ . It implies that  $t$ -GDIM may be not always going to fit the deterministic inference process (9) as GDIM supposed to. Notice the same case can also be found in DDIM, so iDDIM can be useful in  $t$ -GDIM as an ensemble augmentation for denoising targets. A potential solution to the missing dependence on  $\mathbf{x}_T$  is to incorporate explicit condition, i.e.,  $p_{\theta}(\mathbf{x}_0^t | \mathbf{x}_t, \mathbf{x}_T)$ , but may incur additional input for networks. Or we can force the training of  $p_{\theta}(\mathbf{x}_0^t | \mathbf{x}_t)$  to depend on  $\mathbf{x}_T$  implicitly. Furthermore, we find the models used by Gao et al. (2021); Xiao et al. (2022) have similar spirit to  $t$ -GDIM, but ours follows a distinct theoretical route. Inspired by them, we explore feasible methods to introduce the dependence explicitly or implicitly. Please see Appendix D for more discussions.

<table><tr><td colspan="2">T</td><td>4</td><td>10</td><td>20</td><td>50</td><td>100</td><td>1000</td></tr><tr><td rowspan="6">m</td><td>0.0</td><td>37.82</td><td>13.74</td><td>7.55</td><td>4.78</td><td>4.14</td><td>3.88</td></tr><tr><td>0.1</td><td>37.50</td><td>12.03</td><td>6.12</td><td>3.82</td><td>3.55</td><td>3.59</td></tr><tr><td>0.2</td><td>37.02</td><td>10.74</td><td>5.29</td><td>3.82</td><td>4.05</td><td>4.68</td></tr><tr><td>0.3</td><td>36.50</td><td>9.71</td><td>5.05</td><td>4.97</td><td>6.04</td><td>-</td></tr><tr><td>0.4</td><td>36.56</td><td>8.84</td><td>5.51</td><td>8.16</td><td>10.74</td><td>-</td></tr><tr><td>1.0</td><td>35.72</td><td>11.99</td><td>56.60</td><td>112.23</td><td>-</td><td>-</td></tr><tr><td colspan="2">DDPM</td><td>-</td><td>-</td><td>137.77</td><td>35.29</td><td>10.61</td><td>3.19</td></tr></table>

Table 1: CIFAR10 image generation measured in FID $\downarrow$ .  $m$  denotes the replacing ratio in iDDIM.  $m = 0.0$  and  $m = 1.0$  represent DDIM and radical ensemble model respectively. Sampling process uses the pretrained DDPM score predictor.

![](images/a3f472ad5ee0fcf50fee47c42481ce1357b852aa6b090f4401b58bc3c08b0203.jpg)  
Figure 3: Samples of iDDIM with  $T = 10,20,50,100$  and best  $m^{*}$ .

# 6 EXPERIMENTS

In this section, we conduct experiments to verify our two claims: 1). The DDIMs are ensemble denoising implicit models with sub-optimal convex coefficients. With the intuition that the latest denoising prediction  $\mathbf{x}_0^t$  is more precise, our iDDIM provides a simple but effective way to seek better coefficients. 2). Our  $t$ -GDIM can largely reduce the number of denoising steps to just a few, but still achieve comparable generative quality to more expensive diffusion-based models.

Datasets and metrics. For our iDDIMs, we conduct extensive experiments on CIFAR10 (Krizhevsky, 2009) for comparison. For our  $t$ -GDIMs which are trained similarly to conditional GAN, we additionally consider CelebA (Liu et al., 2015) with higher resolutions. We resize the images in CelebA to  $64 \times 64$  and  $128 \times 128$ , termed CelebA-64 and CelebA-128 respectively. For all datasets, only the random horizontal flipping is used for pre-processing. We use the image generation quality to characterize the performance of different methods. The image generation quality on CIFAR10 is evaluated by Frechet inception distance (FID, Heusel et al. 2017) and Inception Score (IS, Salimans et al. 2016). For higher resolutions, only FID is reported since IS is not proper.

Generative process. For all experiments, we use 1000-step linear noise schedule in DDPM to construct the complete diffusion process. Following Song et al. (2021a), quadratic timesteps selection is used to construct the generative sub-sequence. We consider various  $T$  for iDDIM experiments, while only adopt  $T = 4$  for the  $t$ -GDIM part to evaluate its performance on few sampling steps. Please see Appendix E.1 for the architecture of models or complementary experimental details.

# 6.1 SEEKING BETTER COEFFICIENTS

For simplicity, we choose  $m_t = m$  for all  $t$  in iDDIM. In Tab. 1, we show the generation quality of our iDDIM trained on CIFAR10. We find that DDIM ( $m = 0.0$ ) performs worse than iDDIM consistently for each  $T$ , if the ratio  $m$  is properly increased. More interestingly, the sample quality further becomes better when we choose higher  $m$  as  $T$  decreasing, but overly trusting the latest  $\mathbf{x}_0^t$  ( $m \to 1.0$ ) leads to worse quality and the radical ensemble model ( $m = 1.0$ ) performs bad.

In Fig. 4, we display the ensemble denoising prediction  $\bar{\mathbf{x}}_0^t$  at each sampling step of the 20-step and 4-step iDDIM generative processes for CIFAR10 image. It shows that, though  $\bar{\mathbf{x}}_0^t$  are becoming more realistic as  $t$  decreasing, the final prediction of DDIM ( $m = 0.0$ ) is still blurry. With increasing  $m$ , the ensemble prediction at each step becomes much clearer, but with too high  $m$ , especially when  $m = 1.0$  (the radical ensemble model), the predictions become distorted. Besides, in 4-step sampling process, all the ensemble predictions  $\bar{\mathbf{x}}_0^{1:T}$  are blurry regardless of  $m$ . These results suggest the coefficients used by DDIM is sub-optimal, and our iDDIM with proper  $m$  leads to better generative quality, nevertheless still fails when  $T$  is too small.

In Tab. 2, we report the generation quality of iDDIM with best tuned  $m^{*}$ , and compare iDDIM with recent proposed impressive methods for accelerating sampling. For a fair comparison, we report the results of other methods that have similar settings to ours. It demonstrates that our iDDIM achieves the best result among baseline methods in the case of  $T = 10$  and  $T = 20$ , though iDDIM is much simpler than others. When  $T$  is larger,  $t$ -GDIM is slightly worse than FastDPM. It indicates that we require more complexly coefficients seeking since more denoising predictions are involved. Figure 3 presents some randomly generated CIFAR10 samples by our iDDIM with the best  $m^{*}$ .

![](images/b036958f0437ad53c3ef3be31b530335e235224b0507fb74dc8a1fe3da383fc1.jpg)  
Figure 4: Ensemble denoising predictions  $\bar{\mathbf{x}}_0^{1:T}$  in 20-step (left) and 4-step (right) iDDIM sampling process with varying  $m$ .

Table 2: The best  ${m}^{ * }$  for iDDIM on CIFAR10,searched by traversing  $\left\lbrack  {0,1}\right\rbrack$  with 0.05 intervals.  

<table><tr><td>T</td><td colspan="2">10</td><td colspan="2">20</td><td colspan="2">50</td><td colspan="2">100</td></tr><tr><td>m*</td><td colspan="2">0.6</td><td colspan="2">0.3</td><td colspan="2">0.15</td><td colspan="2">0.1</td></tr><tr><td>IS↑ FID↓</td><td>8.85</td><td>8.24</td><td>9.09</td><td>5.05</td><td>9.27</td><td>3.61</td><td>9.21</td><td>3.55</td></tr><tr><td>DDIM (Song et al., 2021a)</td><td>8.28</td><td>13.74</td><td>8.81</td><td>7.55</td><td>8.98</td><td>4.78</td><td>9.11</td><td>4.14</td></tr><tr><td>DDPM (Ho et al., 2020)</td><td colspan="2">-</td><td>3.98</td><td>137.77</td><td>8.53</td><td>35.29</td><td>9.45</td><td>10.61</td></tr><tr><td>GGDM (Watson et al., 2022)</td><td>8.84</td><td>8.23</td><td>9.18</td><td>5.57</td><td colspan="2">-</td><td colspan="2">-</td></tr><tr><td>FastDPM (Kong &amp; Ping, 2021)</td><td>-</td><td>9.90</td><td>-</td><td>5.22</td><td>8.98</td><td>3.41</td><td>-</td><td>3.01</td></tr><tr><td>Analytic-DDIM (Bao et al., 2022)</td><td>-</td><td>14.00</td><td>-</td><td>5.81</td><td>-</td><td>4.04</td><td>-</td><td>3.55</td></tr></table>

# 6.2 SAMPLE QUALITY IN GDIM

For an overall evaluation of the proposed  $t$ -GDIM, Fig. 5 presents the qualitative samples which are generated with only 4 sampling steps. These images are of high fidelity consistently.

In Tab. 3, we present our quantitative results on CIFAR10. Here we report the FID of our  $t$ -GDIM, with iDDIM as an ensemble augmentation technique for the prediction of denoising target  $\bar{\mathbf{x}}_0^t$ . As discussed in Sec. 5, iDDIM can improve the quality of  $t$ -GDIM marginally. We provide related studies in Appendix E.2. When comparing with score-based models, our  $t$ -GDIM can largely reduce the number of function evaluations (NFE) to only 4, while achieve comparable quality. When comparing with GANs, we find our models surpass most of SOTA GANs, though we do not use any data augmentation technique. Notice DDGAN is superior to ours. We suggest the reason is that: the optimization method of  $t$ -GDIM is similar to GAN which ignores the entropy term of variational sampler in variational maximum likelihood (17) in theory, leading to poor mode coverage in practice. Please see Appendix E.2 for qualitative results. While DDGAN is based on the DPM framework, and as a result, additional noises are introduced during training, which is an important data augmentation method for GAN-based optimization. These imply the proven augmentation techniques for GAN-based optimization are useful for our  $t$ -GDIM and potentially improve the generation quality. But here we report the pure version of the proposed  $t$ -GDIM, and leave them for future work.

Table 4 present the quantitative results on CelebA-64 and CelebA-128. When comparing our model with recent few-step diffusion-based models, we find  $t$ -GDIM achieve the best quality among baseline methods with similar NFE. Surprisingly, it even surpass 1000-step DDPM especially on

![](images/e8f91e070e46feeb7f0328a0a311b649edf9104106425dc0827ecad2fd6f6ece.jpg)  
Figure 5: Qualitative samples of  $t$ -GDIM. Left: CIFAR10. Middle: CelebA-64. Right: CelebA-128.

![](images/96937f97c3c11d0b1d334ec3ab777494720d0d6191eb7b561201208c33fc0b19.jpg)

![](images/b877478749f3473ff1b45cfcb5582084fb1f5ce0f31abe96a8e46bcdb1e7953f.jpg)

Table 3: CIFAR10 image generation measured in IS↑ and FID↓.  

<table><tr><td>Method</td><td>IS↑</td><td>FID↓</td><td>NFE↓</td></tr><tr><td>Improved DDPM (Nichol &amp; Dhariwal, 2021)</td><td>-</td><td>2.90</td><td>4000</td></tr><tr><td>UDM (Kim et al., 2021)</td><td>10.1</td><td>2.33</td><td>2000</td></tr><tr><td>Likelihood SDE (Song et al., 2021b)</td><td>-</td><td>2.87</td><td>2000</td></tr><tr><td>Score SDE (VE) (Song et al., 2021c)</td><td>9.89</td><td>2.20</td><td>2000</td></tr><tr><td>DDPM (Ho et al., 2020)</td><td>9.47</td><td>3.19</td><td>1000</td></tr><tr><td>NCSN (Song &amp; Ermon, 2019)</td><td>8.87</td><td>25.3</td><td>1000</td></tr><tr><td>Adversarial DSM (Jolicoeur-Martineau et al., 2021b)</td><td>-</td><td>6.10</td><td>1000</td></tr><tr><td>VDM (Kingma et al., 2021)</td><td>-</td><td>4.00</td><td>1000</td></tr><tr><td>Recovery EBM, T6 (Gao et al., 2021)</td><td>8.30</td><td>9.58</td><td>180</td></tr><tr><td>Gotta Go Fast (Jolicoeur-Martineau et al., 2021a)</td><td>-</td><td>2.44</td><td>180</td></tr><tr><td>LSGM (Vahdat et al., 2021)</td><td>9.87</td><td>2.10</td><td>147</td></tr><tr><td>CLD-SGM (Prob. Flow) (Dockhorn et al., 2022)</td><td>-</td><td>2.71</td><td>147</td></tr><tr><td>Probability Flow (VP) (Song et al., 2021c)</td><td>9.83</td><td>3.08</td><td>140</td></tr><tr><td>DiffuseVAE, T = 100 (Pandey et al., 2022)</td><td>8.27</td><td>11.71</td><td>100</td></tr><tr><td>FastDPM, T = 50 (Kong &amp; Ping, 2021)</td><td>8.98</td><td>3.41</td><td>50</td></tr><tr><td>F-PNDM, T = 50 (Liu et al., 2022)</td><td>-</td><td>3.68</td><td>50</td></tr><tr><td>gDDIM, T = 50 (Zhang et al., 2022)</td><td>-</td><td>2.28</td><td>50</td></tr><tr><td>SNGAN+DGflow (Ansari et al., 2021)</td><td>9.35</td><td>9.62</td><td>25</td></tr><tr><td>DDGAN, T = 4 (Xiao et al., 2022)</td><td>9.63</td><td>3.75</td><td>4</td></tr><tr><td>DDPM Distillation (Luhman &amp; Luhman, 2021)</td><td>8.36</td><td>9.36</td><td>1</td></tr><tr><td>AutoGAN (Gong et al., 2019)</td><td>8.60</td><td>12.4</td><td>1</td></tr><tr><td>TransGAN (fan Jiang et al., 2021)</td><td>9.02</td><td>9.26</td><td>1</td></tr><tr><td>StyleGAN2 w/o ADA (Karras et al., 2020a)</td><td>9.18</td><td>8.32</td><td>1</td></tr><tr><td>StyleGAN2 w/ ADA (Karras et al., 2020a)</td><td>9.83</td><td>2.92</td><td>1</td></tr><tr><td>StyleGAN2 w/ Diffaug (Zhao et al., 2020)</td><td>9.40</td><td>5.79</td><td>1</td></tr><tr><td>t-GDIM (ours)</td><td>9.55</td><td>5.51</td><td>4</td></tr><tr><td>t-GDIM+iDDIM, m = 0.7 (ours)</td><td>9.50</td><td>5.24</td><td>4</td></tr></table>

Table 4: CelebA-64 and CelebA-128 image generation measured in FID↓.  

<table><tr><td>Method</td><td>CelebA-64</td><td>CelebA-128</td><td>NFE↓</td></tr><tr><td>DDPM (Ho et al., 2020)</td><td>3.26</td><td>5.65</td><td>1000</td></tr><tr><td>Recovery EBM, T6 (Gao et al., 2021)</td><td>5.98</td><td>-</td><td>180</td></tr><tr><td>F-PNDM, T = 10 (Liu et al., 2022)</td><td>7.71</td><td>-</td><td>10</td></tr><tr><td>StyleGAN2+5-step ES-DDPM (Lyu et al., 2022)</td><td>9.15</td><td>6.15</td><td>5</td></tr><tr><td>TDPM-GAN, T = 4 (Zheng et al., 2022)</td><td>3.96</td><td>-</td><td>4</td></tr><tr><td>COCO-GAN (Lin et al., 2019)</td><td>4.00</td><td>5.74</td><td>1</td></tr><tr><td>t-GDIM (ours)</td><td>3.28</td><td>4.55</td><td>4</td></tr><tr><td>t-GDIM+iDDIM, m = 0.6 (ours)</td><td>2.93</td><td>4.04</td><td>4</td></tr></table>

$128 \times 128$  resolution, which suggests that,  $t$ -GDIM has the potential to generate realistic larger scale images with few sampling steps.

# 7 CONCLUSION

We have provided an insightful perspective that DDIM is a specific instance of our ensemble denoising implicit model with sub-optimal convex coefficients. This explains why DDIM fails to achieve good generation quality with fewer sampling steps. Our iDDIM is an intuition guided modification on DDIM which simply allocates more trust on the latest denoising prediction, but can improve on DDIM largely. To further decrease the sampling steps, we propose GDIM, a general extension to ensemble model, that replaces the additive ensemble of denoising predictions to a principled probabilistic inference. Then the variational maximum likelihood is used to train  $t$ -GDIM, a specific GDIM only conditioned on the latest state at each step, and derivates more favorable GAN-based optimization methods. Extensive experiments demonstrate  $t$ -GDIM can reduce the number of sampling steps to only 4 while achieve comparable performance to other generative models. It also shows the potential to apply  $t$ -GDIM on higher resolutions, where we leave it for future work.

# REFERENCES

Abdul Fatir Ansari, Ming Liang Ang, and Harold Soh. Refining deep generative models via discriminator gradient flow. In International Conference on Learning Representations, 2021. 9  
Martín Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein gan. arXiv preprint arXiv:1701.07875, 2017. 6  
Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. International Conference on Learning Representations, 2022. 8  
Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 2021. 1  
Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Score-based generative modeling with critically-damped Langevin diffusion. International Conference on Learning Representations, 2022. 6, 9  
Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models. In Advances in Neural Information Processing Systems, 2019. 6, 18  
Yi fan Jiang, Shiyu Chang, and Zhangyang Wang. Transgan: Two transformers can make one strong gan. ArXiv, abs/2102.07074, 2021. 9  
Ruiqi Gao, Yang Song, Ben Poole, Ying Nian Wu, and Diederik P. Kingma. Learning energy-based models by diffusion recovery likelihood. International Conference on Learning Representations, 2021. 6, 9, 20  
Xinyu Gong, Shiyu Chang, Yi fan Jiang, and Zhangyang Wang. Autogan: Neural architecture search for generative adversarial networks. Proceedings of the IEEE International Conference on Computer Vision, pp. 3223-3233, 2019. 9  
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, 2014. 1  
Will Grathwohl, Jacob Kelly, Milad Hashemi, Mohammad Norouzi, Kevin Swersky, and David Kristjanson Duvenaud. No mcmc for me: Amortized sampling for fast and stable training of energy-based models. In International Conference on Learning Representations, 2021. 2, 6, 19  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, 2017. 7  
Jonathan Ho, Ajay Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 2020. 1, 3, 8, 9, 13, 21  
Alexia Jolicoeur-Martineau, Ke Li, Remi Piche-Taillefer, Tal Kachman, and Ioannis Mitliagkas. Gotta go fast when generating data with score-based models. ArXiv, abs/2105.14080, 2021a. 6, 9  
Alexia Jolicoeur-Martineau, Remi Piche-Taillefer, Rémi Tachet des Combes, and Ioannis Mitliagkas. Adversarial score matching and improved sampling for image generation. International Conference on Learning Representations, 2021b. 9  
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. Advances in Neural Information Processing Systems, 2020a. 9  
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8107-8116, 2020b. 21

Dongjun Kim, Seungjae Shin, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon. Score matching model for unbounded data score. ArXiv, abs/2106.05527, 2021. 9  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2014. 2, 5, 13  
Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. ArXiv, abs/2107.00630, 2021. 9  
Zhifeng Kong and Wei Ping. On fast sampling of diffusion probabilistic models. ArXiv, abs/2106.00132, 2021. 6, 8, 9  
Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009. 7  
Yann LeCun, Sumit Chopra, Raia Hadsell, Aurelio Ranzato, and Fu Jie Huang. A tutorial on energy-based learning. 2006. 2, 6  
Chieh Hubert Lin, Chia-Che Chang, Yu-Sheng Chen, Da-Cheng Juan, Wei Wei, and Hwann-Tzong Chen. Coco-gan: Generation by parts via conditional coordinating. Proceedings of the IEEE International Conference on Computer Vision, pp. 4511-4520, 2019. 9  
Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. International Conference on Learning Representations, 2022. 6, 9  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision, pp. 3730-3738, 2015.  
Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. ArXiv, abs/2206.00927, 2022. 6  
Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. ArXiv, abs/2101.02388, 2021. 9  
Zhaoyang Lyu, Xu Xudong, Ceyuan Yang, Dahua Lin, and Bo Dai. Accelerating diffusion models via early stop of the diffusion process. ArXiv, abs/2205.12524, 2022. 6, 9  
Shakir Mohamed and Balaji Lakshminarayanan. Learning in implicit generative models. arXiv preprint arXiv:1610.03483, 2016. 1  
Eliya Nachmani, Robin San-Roman, and Lior Wolf. Non gaussian denoising diffusion models. ArXiv, abs/2106.07582, 2021. 6  
Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. International conference on machine learning, 2021. 9  
Erik Nijkamp, Ruiqi Gao, Pavel Sountsov, Srinivas Vasudevan, Bo Pang, Song-Chun Zhu, and Ying Nian Wu. Learning energy-based model with flow-based backbone by neural transport mcmc. arXiv preprint arXiv:2006.06897, 2020a. 18  
Erik Nijkamp, Mitch Hill, Tian Han, Song-Chun Zhu, and Ying Nian Wu. On the anatomy of mcmc-based maximum likelihood learning of energy-based models. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020b. 6, 18  
Kushagra Pandey, Avideep Mukherjee, Piyush Rai, and Abhishek Kumar. Diffusevae: Efficient, controllable and high-fidelity generation from low-dimensional latents. ArXiv, abs/2201.00308, 2022.9  
George Papamakarios, Eric T. Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22:57:1-57:64, 2021. 17

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. International Conference on Medical image computing and computer-assisted intervention, 2015. 21  
Tim Salimans, Ian J. Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in Neural Information Processing Systems, 2016. 7  
Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. International conference on machine learning, 2015. 1, 2, 13  
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. International Conference on Learning Representations, 2021a. 1, 2, 3, 7, 8, 14, 21  
Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 2019. 6, 9  
Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. Advances in Neural Information Processing Systems, 2020. 6  
Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. In Advances in Neural Information Processing Systems, 2021b. 9  
Yang Song, Jascha Narain Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. International Conference on Learning Representations, 2021c. 6, 9, 21  
Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. In Advances in Neural Information Processing Systems, 2021. 9  
Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23:1661-1674, 2011. 6  
Daniel Watson, Jonathan Ho, Mohammad Norouzi, and William Chan. Learning to efficiently sample from diffusion probabilistic models. *ArXiv*, abs/2106.03802, 2021. 6  
Daniel Watson, William Chan, Jonathan Ho, and Mohammad Norouzi. Learning fast samplers for diffusion models by differentiating through sample quality. International Conference on Learning Representations, 2022. 6, 8  
Max Welling and Yee Whye Teh. Bayesian learning via stochastic gradient Langevin dynamics. In International conference on machine learning, 2011. 18  
Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion gans. International Conference on Learning Representations, 2022. 5, 6, 9, 20, 21, 22  
Qinsheng Zhang, Molei Tao, and Yongxin Chen. gddim: Generalized denoising diffusion implicit models. ArXiv, abs/2206.05564, 2022. 9  
Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han. Differentiable augmentation for data-efficient gan training. Advances in Neural Information Processing Systems, 2020. 9  
Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Truncated diffusion probabilistic models. ArXiv, abs/2202.09671, 2022. 6, 9
