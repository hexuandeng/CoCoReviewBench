# ON IMPORTANCE-WEIGHTED AUTOENCODERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

The importance weighted autoencoder (IWAE) (Burda et al., 2016) is a popular variational-inference method which achieves a tighter evidence bound (and hence a lower bias) than standard variational autoencoders by optimising a multi-sample objective, i.e. an objective that is expressible as an integral over  $K > 1$  Monte Carlo samples. Unfortunately, IWAE crucially relies on the availability of reparametrisations and even if these exist, the multi-sample objective leads to inference-network gradients which break down as  $K$  is increased (Rainforth et al., 2018). This breakdown can only be circumvented by removing high-variance score-function terms, either by heuristically ignoring them (which yields the 'sticking-the-landing' IWAE (IWAE-STL) gradient from Roeder et al. (2017)) or through an identity from Tucker et al. (2019) (which yields the 'doubly-reparametrised' IWAE (IWAE-DREG) gradient). In this work, we argue that directly optimising the proposal distribution in importance sampling as in the reweighted wake-sleep (RWS) algorithm from Bornschein & Bengio (2015) is preferable to optimising IWAE-type multi-sample objectives. To formalise this argument, we introduce an adaptive-importance sampling framework termed adaptive importance sampling for learning (AISLE) which slightly generalises the RWS algorithm. We then show that AISLE admits IWAE-STL and IWAE-DREG (i.e. the IWAE-gradients which avoid breakdown) as special cases.

# 1 INTRODUCTION

# 1.1 PROBLEM STATEMENT

Let  $x$  be some observation and let  $z$  be some latent variable taking values in some space  $\mathbb{Z}$ . These are modeled via the generative model  $p_{\theta}(z,x) = p_{\theta}(z)p_{\theta}(x|z)$  which gives rise to the marginal likelihood  $p_{\theta}(x) = \int_{\mathbb{Z}}p_{\theta}(z,x)\mathrm{d}z$  of the model parameters  $\theta$ . The latter may also be viewed as the evidence for the model parametrised by a particular value of  $\theta$ . In this work, we analyse algorithms for variational inference, i.e. algorithms which aim to

1. learn the generative model, i.e. find a value  $\theta^{\star}$  which is approximately equal to the maximum-likelihood estimate (MLE)  $\theta^{\mathrm{ML}}\coloneqq \arg \max_{\theta}p_{\theta}(x)$ ;  
2. construct a tractable variational approximation  $q_{\phi, x}(z)$  of  $p_{\theta}(z|x) = p_{\theta}(z, x) / p_{\theta}(x)$ , i.e. find the value  $\phi^{\star}$  such that  $q_{\phi^{\star}, x}(z)$  is as close as possible to  $p_{\theta}(z|x)$  in some suitable sense.

A few comments about this setting are in order. Firstly, as is common in the literature, we restrict our presentation to a single latent representation-observation pair  $(z,x)$  to avoid notational clutter - the extension to multiple independent observations is straightforward. Secondly, we assume that no parameters are shared between the generative model  $p_{\theta}(z,x)$  and the variational approximation  $q_{\phi ,x}(z)$ . This is common in neural-network applications but could be relaxed. Thirdly, our setting is general enough to cover amortised inference which is why we often refer to  $\phi$  as the parameters of an inference network.

In recent years, two classes of stochastic-gradient ascent algorithms for optimising  $(\theta ,\phi)$  - which employ  $K\geq 1$  Monte Carlo samples ('particles') to reduce errors - have been proposed.

- IWAE. The importance weighted autoencoder (IWAE) (Burda et al., 2016) optimises a joint objective for  $\theta$  and  $\phi$  (which is 'biased' for  $\theta$  though optimising  $\phi$  or increasing  $K$  decreases this bias) whose gradients are unbiasedly approximated via the Monte Carlo method. Unfortunately, as this multi-sample objective is expressible as an integral on a  $K$ -dimensional space, the signal-to-noise ratio of the IWAE  $\phi$ -gradient vanishes as  $K$  grows (Rainforth et al., 2018). Two modified IWAE  $\phi$ -gradients avoid this breakdown by removing high-variance 'score-function' terms:

- IWAE-STL. The 'sticking-the-landing' IWAE (IWAE-STL)  $\phi$ -gradient (Roeder et al., 2017) heuristically drops the problematic score-function terms from the IWAE  $\phi$ -gradient. This induces bias for the IWAE objective.  
- IWAE-DREG. The 'doubly-reparametrised' IWAE (IWAE-DREG)  $\phi$ -gradient (Tucker et al., 2019) unbiasedly removes the problematic score-function terms from the IWAE  $\phi$ -gradient using a formal identity.

- RWS. The reweighted wake-sleep (RWS) algorithm (Bornshein & Bengio, 2015) optimises two separate but 'unbiased' objectives for  $\theta$  and  $\phi$ . Its gradients are approximated by self-normalised importance sampling with  $K$  particles which induces bias (though again, optimising  $\phi$  or increasing  $K$  decreases this bias). RWS can be viewed as an adaptive importance-sampling approach which iteratively improves its proposal distribution while simultaneously optimising  $\theta$  via stochastic approximation. Crucially, RWS is not a multi-sample objective approach and hence does not require continuous reparametrisations nor do its  $\phi$ -gradients suffer from the breakdown highlighted in Rainforth et al. (2018).

Of these two methods, the IWAE is the most popular and Tucker et al. (2019) demonstrated empirically that RWS can break down, conjecturing that this is due to the fact that RWS does not optimise a joint objective (for  $\theta$  and  $\phi$ ). Meanwhile, the IWAE-STL gradient performed consistently well despite lacking a firm theoretical footing. Yet, IWAE suffers from the above-mentioned  $\phi$ -gradient breakdown and exhibited inferior empirical performance to RWS in some scenarios (Le et al., 2019). Thus, it is not clear whether the multi-sample objective approach of IWAE or the adaptive importance-sampling approach of RWS is preferable.

In this work, we argue that the adaptive importance-sampling paradigm of RWS is preferable to the multi-sample objective paradigm of IWAEs. This is because (a) the multi-sample objective crucially requires reparametrisations and, even if these are available, leads to the  $\phi$ -gradient breakdown, (b) modifications of the IWAE  $\phi$ -gradient which avoid this breakdown (i.e. IWAE-STL and IWAE-DREG) can be justified in a more principled manner by taking an RWS-type adaptive importance-sampling view.

To formalise these arguments, we slightly generalise the RWS algorithm to obtain a generic adaptive importance-sampling framework for variational inference which we term adaptive importance sampling for learning (AISLE) for ease of reference. We then show that AISLE admits not only RWS but also the IWAE-DREG and IWAE-STL gradients as special cases.

# 1.2 CONTRIBUTIONS

Importance sampling as well as the IWAE and RWS algorithms are reviewed in Section 2. Novel material is presented in Section 3, where we introduce the AISLE-framework:

- In Subsection 3.3, we show that AISLE admits RWS as a special case. In addition, we prove that the IWAE-STL gradient is in turn recovered as a special case of RWS (and hence of AISLE) via a principled and novel application of the 'double-reparametrisation' identity from Tucker et al. (2019). This indicates that the breakdown of RWS observed in Tucker et al. (2019) may not be due to its lack of a joint objective as previously conjectured (because IWAE-STL avoided this breakdown). Our work also provides a theoretical foundation for IWAE-STL which was hitherto only heuristically justified as a biased IWAE gradient.  
- In Subsection 3.4, we prove that AISLE also admits the IWAE-DREG gradient as a special case. Our derivation also makes it clear that the learning rate should be scaled as  $\mathcal{O}(K)$  for the IWAE  $\phi$ -gradient (and its modified version IWAE-DREG)

unless the gradients are normalised as implicitly done by popular optimisers such as ADAM (Kingma & Ba, 2015). In contrast, the scaling of the learning rate for AISLE is independent of  $K$ .

- In the supplementary materials, we provide some insight into the impact of the self-normalisation bias on some of the importance-sampling based gradient approximations (Appendix A) and empirically compare all algorithms discussed in this work (Appendix B).

We stress that the point of our work is not to derive new algorithms nor to establish which of the various special cases of AISLE is preferable. Indeed, while we compare all algorithms discussed in this work empirically on Gaussian models in the in the supplementary materials available with this paper, we refer the reader to Tucker et al. (2019); Le et al. (2019) for a extensive empirical comparisons of all the algorithms discussed in this work. Instead, the main message of our work is that the AISLE-type adaptive importance-sampling paradigm is preferable to the IWAE-type multi-sample objective paradigm because the former allows us to derive all the above-mentioned variants of IWAE - as well as further algorithms which do not require reparametrisations - in a principled manner (the only exception is the standard IWAE reparametrisation  $\phi$ -gradient but this variant suffers from the breakdown highlighted in Rainforth et al. (2018) and was therefore consistently outperformed by the other variants in the simulations shown in Appendix B and in Tucker et al. (2019), for  $K > 1$ ).

# 1.3 NOTATION

We assume that all (probability) measures  $p$  used in this work are absolutely continuous w.r.t. some suitable dominating measure  $\mathrm{d}z$  and with some abuse of notation, we use the same symbol for the measure and the density, i.e. we write  $p(\mathrm{d}z) = p(z)\mathrm{d}z$ . With this convention, we employ the shorthand  $p(f) \coloneqq \int_{\mathbb{Z}} f(z)p(z)\mathrm{d}z$  for the integral of some  $p$ -integrable test function  $f$ ; thus,  $p(f) = \mathbb{E}_{z \sim p}[f(z)]$  if  $p$  is a probability measure. Furthermore,  $q^{\otimes K}(z^{1:K}) \coloneqq \prod_{k=1}^{K} q(z^k)$ . We also let 0 denote vectors or matrices of 0s of some appropriate size which will be clear from the context and we let 1 be the function that takes value 1 everywhere on its domain. To keep the notation concise, we hereafter suppress dependence on the observation  $x$ , i.e. we write  $q_{\phi}(z) \coloneqq q_{\phi,x}(z)$  as well as

$$
\pi_ {\theta} (z) := p _ {\theta} (z | x) = \frac {p _ {\theta} (z , x)}{p _ {\theta} (x)} = \frac {\gamma_ {\theta} (z)}{\mathcal {Z} _ {\theta}},
$$

where  $\gamma_{\theta}(z)\coloneqq p_{\theta}(z,x)$  and where  $\mathcal{Z}_{\theta}\coloneqq p_{\theta}(x) = \int_{\mathbb{Z}}\gamma_{\theta}(z)\mathrm{d}z = \gamma_{\theta}(1)$ .

# 2 BACKGROUND

# 2.1 IMPORTANCE SAMPLING

Basic idea. We hereafter write  $\psi \coloneqq (\theta, \phi)$  and assume that the support of  $q_{\phi}$  includes the support of  $\pi_{\theta}$  so that the importance weight function  $w_{\psi}(z) \coloneqq \gamma_{\theta}(z) / q_{\phi}(z)$  is well defined. For  $\pi_{\theta}$ -integrable  $f \colon \mathbb{Z} \to \mathbb{R}$ , we can unbiasedly approximate integrals of the form

$$
\gamma_ {\theta} (f) := \int_ {\mathbb {Z}} f (z) \gamma_ {\theta} (z) \mathrm {d} z = \int_ {\mathbb {Z}} f (z) w _ {\psi} (z) q _ {\phi} (z) \mathrm {d} z = q _ {\phi} (f w _ {\psi}), \tag {1}
$$

via importance sampling using a set of  $K$  particles,  $\mathbf{z} \coloneqq (z^{1},\ldots ,z^{K}) \sim q_{\phi}^{\otimes K}$ , which are independent and identically distributed (IID) according to  $q_{\phi}$ , as

$$
\hat {\gamma} _ {\theta} \langle \phi , \mathbf {z} \rangle (f) := \frac {1}{K} \sum_ {k = 1} ^ {K} w _ {\psi} \left(z ^ {k}\right) f \left(z ^ {k}\right).
$$

Here, the notation  $\langle \phi ,\mathbf{z}\rangle$  stresses the dependence of the estimator on  $\phi$  and  $\mathbf{z}$ . Note that this is simply an application of the vanilla Monte Carlo method to the expectation from the r.h.s. of (1). Hereafter, we use the convention that  $\mathbb{E} = \mathbb{E}_{\mathbf{z}\sim q_{\phi}^{\otimes K}}$  and  $\mathrm{var}_{\mathbf{z}\sim q_{\phi}^{\otimes K}}$  denote expectation and variance w.r.t.  $\mathbf{z} = (z^{1},\dots,z^{K})\sim q_{\phi}^{\otimes K}$ .

Self-normalised importance sampling. Approximating integrals of the form

$$
\pi_ {\theta} (f) := \int_ {\mathbb {Z}} f (z) \pi_ {\theta} (z) \mathrm {d} z = \frac {\gamma_ {\theta} (f)}{\gamma_ {\theta} (1)},
$$

is slightly more complicated because the marginal likelihood  $\mathcal{Z}_{\theta} = \gamma_{\theta}(1) = p_{\theta}(x)$  is intractable. Plugging in importance-sampling approximations for both the numerator and denominator leads to the following self-normalised importance sampling estimate:

$$
\hat {\pi} _ {\theta} \langle \phi , \mathbf {z} \rangle (f) := \frac {\hat {\gamma} _ {\theta} \langle \phi , \mathbf {z} \rangle (f)}{\hat {\gamma} _ {\theta} \langle \phi , \mathbf {z} \rangle (1)} = \sum_ {k = 1} ^ {K} \frac {w _ {\psi} (z ^ {k})}{\sum_ {l = 1} ^ {K} w _ {\psi} (z ^ {l})} f (z ^ {k}).
$$

Properties. Proposition 1 summarises some well-known properties of importance-sampling approximations (see, e.g., Geweke, 1989) used throughout this work.

Proposition 1. Let  $f\colon \mathbb{Z}\to \mathbb{R}$  be  $\pi_{\theta}$ -integrable and  $\mathbf{z}\sim q_{\phi}^{\otimes K}$ . Then if  $\sup w_{\psi} < \infty$

1.  $\mathbb{E}[\hat{\gamma}_{\theta}\langle \phi ,\mathbf{z}\rangle (f)] = \gamma_{\theta}(f)$  , for any  $K\in \mathbb{N}$  
2.  $\mathbb{E}[\hat{\pi}_{\theta}\langle \phi ,\mathbf{z}\rangle (f)] = \pi_{\theta}(f) + \mathcal{O}(K^{-1})$  and  $\mathrm{var}[\hat{\pi}_{\theta}\langle \phi ,\mathbf{z}\rangle (f)] = \mathcal{O}(K^{-1}),$  
3.  $\hat{\gamma}_{\theta}\langle \phi ,\mathbf{z}\rangle (f)\to \gamma_{\theta}(f)$  and  $\hat{\pi}_{\theta}\langle \phi ,\mathbf{z}\rangle (f)\to \pi_{\theta}(f)$ , almost surely, as  $K\rightarrow \infty$

Proof. Part 1 is immediate; Part 2 is proved, e.g. in Liu (2001, p. 35); Part 3 is a direct consequence of the strong law of large numbers.  $\square$

Part 1 of Proposition 1 shows that (non self-normalised) importance-sampling approximations  $\hat{\gamma}_{\theta}\langle \phi ,\mathbf{z}\rangle (f)$  are unbiased. In particular,

$$
\widehat {\mathcal {Z}} _ {\theta} \langle \phi , \mathbf {z} \rangle := \widehat {\gamma} _ {\theta} \langle \phi , \mathbf {z} \rangle (1) = \frac {1}{K} \sum_ {k = 1} ^ {K} w _ {\psi} (z ^ {k}),
$$

is an unbiased estimate of the normalising constant  $\mathcal{Z}_{\theta} = \gamma_{\theta}(1) = p_{\theta}(x)$ . In contrast, the self-normalised importance-sampling approximation  $\hat{\pi}_{\theta}\langle \phi ,\mathbf{z}\rangle (f)$  is typically biased. However, Part 3 shows that it is still consistent and Part 2 ensures that the bias decays quickly in  $K$ .

# 2.2 IMPORTANCE WEIGHTED AUTOENCODER (IWAE)

Objective. The importance weighted autoencoder (IWAE), introduced by Burda et al. (2016), seeks to find a value  $\theta^{\star}$  of the generative-model parameters  $\theta$  which maximises a lower bound  $\mathcal{L}_{\psi}^{K}$  on the log-marginal likelihood ('evidence') which depends on the inference-network parameters  $\phi$  and the number of samples,  $K \geq 1$ ,

$$
\psi^ {\star} := \left(\theta^ {\star}, \phi^ {\star}\right) := \arg \max  _ {\psi} \mathcal {L} _ {\psi} ^ {K},
$$

$$
\mathcal {L} _ {\psi} ^ {K} := \mathbb {E} \left[ \log \widehat {\mathcal {Z}} _ {\theta} \langle \phi , \mathbf {z} \rangle \right]. \tag {2}
$$

For any finite  $K$ , optimisation of the inference-network parameters  $\phi$  tightens the evidence bound. Burda et al. (2016) prove the following properties. Firstly,  $\mathcal{L}_{\psi}^{K} \leq \log \mathcal{Z}_{\theta}$  follows from Jensen's inequality and Part 1 of Proposition 1. Secondly, again by Jensen's inequality,  $\mathcal{L}_{\psi}^{K} \leq \mathcal{L}_{\psi}^{K+1}$ . These inequalities are strict unless  $\pi_{\theta} = q_{\phi}$ . Finally, Part 3 of Proposition 1 (along with the dominated convergence theorem) shows that for any  $\phi$ ,  $\mathcal{L}_{\psi}^{K} \uparrow \log \mathcal{Z}_{\theta}$  as  $K \to \infty$ . If  $K = 1$ , the IWAE reduces to the variational autoencoder (VAE) from Kingma & Welling (2014). However, for  $K > 1$ , as pointed out in Cremer et al. (2017); Domke & Sheldon (2018), the IWAE also constitutes another VAE on an extended space based on an auxiliary-variable construction developed in Andrieu & Roberts (2009); Andrieu et al. (2010); Lee (2011) (see, e.g. Finke, 2015, for a review).

Standard reparametrisation gradient. The gradient of the IwAE objective from (2)  $\nabla_{\psi}\mathcal{L}_{\psi}^{K} = \mathbb{E}\big[\nabla_{\psi}\log \widehat{\mathcal{Z}}_{\theta}\langle \phi ,\mathbf{z}\rangle +G_{\psi}(\mathbf{z})\big]$ , with  $G_{\psi}(\mathbf{z})\coloneqq \log \widehat{\mathcal{Z}}_{\theta}\langle \phi ,\mathbf{z}\rangle \sum_{k = 1}^{K}\nabla_{\psi}\log q_{\phi}(z^{k})$ , is typically intractable. However, it could be approximated unbiasedly via a vanilla Monte Carlo approximation using a single sample point  $\mathbf{z} = (z^1,\dots,z^K)\sim q_\phi^{\otimes K}$ . Unfortunately, the term  $G_{\psi}(\mathbf{z})$  typically has such a large variance that the Monte Carlo approximation becomes impracticably noisy (Paisley et al., 2012). To remove this high-variance term, the well known reparametrisation trick (Kingma & Welling, 2014) is usually employed. It requires that the following assumption holds.

(R1) There exists a distribution  $q$  on some space  $\mathsf{E}$  and a diffeomorphism  $h_\phi \colon \mathsf{E} \to \mathsf{Z}$  such that  $e \sim q \Leftrightarrow h_\phi(e) \sim q_\phi$ .

Under R1, the gradient can alternatively be expressed as

$$
\begin{array}{l} \nabla_ {\psi} \mathcal {L} _ {\psi} ^ {K} = \mathbb {E} _ {e _ {1} ^ {1}, \ldots , e ^ {K} \sim q} ^ {\mathrm {I I D}} \big [ \nabla_ {\psi} \log \widehat {\mathcal {Z}} _ {\theta} \langle \phi , \{h _ {\phi} (e ^ {k}) \} _ {k = 1} ^ {K} \rangle \big ] \\ = \mathbb {E} _ {e ^ {1}, \dots , e ^ {K} \sim q} \left[ \sum_ {k = 1} ^ {K} \frac {w _ {\psi} \left(h _ {\phi} \left(e ^ {k}\right)\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(h _ {\phi} \left(e ^ {l}\right)\right)} \nabla_ {\psi} \log w _ {\psi} \left(h _ {\phi} \left(e ^ {k}\right)\right) \right] \\ = \mathbb {E} \left[ \sum_ {k = 1} ^ {K} \frac {w _ {\psi} \left(z ^ {k}\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(z ^ {l}\right)} \left(\nabla_ {\theta} \log \gamma_ {\theta} \left(z ^ {k}\right) \nabla_ {\psi} \left(z ^ {k}\right) - \nabla_ {\phi} \log q _ {\phi} \left(z ^ {k}\right)\right) \right], \tag {3} \\ \end{array}
$$

with

$$
\nabla_ {\psi} (z) := \nabla_ {\phi} [ \log \circ w _ {\psi^ {\prime}} \circ h _ {\phi} ] | _ {\psi^ {\prime} = \psi} (h _ {\phi} ^ {- 1} (z)).
$$

IWAE then uses a vanilla Monte Carlo estimate of (3) (using a single sample point  $\mathbf{z} \sim q_{\phi}^{\otimes K}$ ):

$$
\left[ \begin{array}{l} \widehat {\nabla} _ {\theta} ^ {\mathrm {I W A E}} \langle \phi , \mathbf {z} \rangle \\ \widehat {\nabla} _ {\phi} ^ {\mathrm {I W A E}} \langle \theta , \mathbf {z} \rangle \end{array} \right] := \sum_ {k = 1} ^ {K} \frac {w _ {\psi} \left(z ^ {k}\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(z ^ {l}\right)} \left[ \begin{array}{c} \nabla_ {\theta} \log \gamma_ {\theta} \left(z ^ {k}\right) \\ \nabla_ {\psi} \left(z ^ {k}\right) - \nabla_ {\phi} \log q _ {\phi} \left(z ^ {k}\right) \end{array} \right]. \tag {4}
$$

$\phi$ -gradient issues. Before proceeding, we state the following lemma, proved in Tucker et al. (2019, Section 8.1), which generalises of the well-known identity  $q_{\phi}(\nabla_{\phi}\log q_{\phi}) = 0$

Lemma 1 (Tucker et al. (2019)). Under  $\mathbf{R1}$ , for suitably integrable  $f_{\psi} \colon \mathbb{Z} \to \mathbb{R}$ :

$$
q _ {\phi} (f _ {\psi} \nabla_ {\phi} \log q _ {\phi}) = q _ {\phi} (\nabla_ {\phi} [ f _ {\psi^ {\prime}} \circ h _ {\phi} ] | _ {\psi^ {\prime} = \psi} \circ h _ {\phi} ^ {- 1}).
$$

![](images/072b90359ea4afeec71136bfc990402b40ea2006adf9923ed7980d5736f04aff.jpg)

We now exclusively focus on the  $\phi$ -portion of the IwAE gradient,  $\widehat{\nabla}_{\phi}^{\mathrm{IwAE}}\langle \theta ,\mathbf{z}\rangle$

Remark 1 (drawbacks of the IwAE  $\phi$ -gradient). The gradient  $\widehat{\nabla}_{\phi}^{\mathrm{IwAE}}\langle \theta, \mathbf{z} \rangle$  has three drawbacks. The last two of these are attributable to the 'score-function' terms  $\nabla_{\phi} \log q_{\phi}(z)$  in the  $\phi$ -gradient portion of (4).

- Reliance on reparametrisations. A continuous reparametrisation à la  $\mathbf{R1}$  is necessary to remove the high-variance term  $G_{\psi}(\mathbf{z})$ ; this makes it difficult to use IWAE for models with e.g. discrete latent variables  $z$  (Le et al., 2019).  
- Vanishing signal-to-noise ratio. The  $\phi$ -gradient breaks down in the sense that its signal-to-noise ratio vanishes as  $\mathbb{E}[\widehat{\nabla}_{\phi}^{\mathrm{IWAE}}\langle \theta ,\mathbf{z}\rangle ] / \mathrm{var}[\widehat{\nabla}_{\phi}^{\mathrm{IWAE}}\langle \theta ,\mathbf{z}\rangle ]^{1 / 2} = \mathcal{O}(K^{-1 / 2})$  (Rainforth et al., 2018). This follows from Part 2 of Proposition 1 since  $\widehat{\nabla}_{\phi}^{\mathrm{IWAE}}\langle \theta ,\mathbf{z}\rangle$  constitutes a self-normalised importance-sampling approximation of  $\pi_{\theta}(\nabla_{\psi} - \nabla_{\phi}\log q_{\phi}) = 0$  (the last identity follows from Lemma 1 with  $f_{\psi} = w_{\psi}$ ).  
- Inability to achieve zero variance. As pointed out in Roeder et al. (2017),  $\operatorname{var}[\widehat{\nabla}_{\phi}^{\mathrm{IWAE}}\langle \theta ,\mathbf{z}\rangle ]] > 0$  even in the ideal scenario that  $q_{\phi} = \pi_{\theta}$  despite the fact that in this case,  $w_{\psi}$  is constant and hence  $\operatorname{var}[\log \widehat{\mathcal{Z}}_{\theta}\langle \phi ,\mathbf{z}\rangle ] = 0$

Two modifications of  $\widehat{\nabla}_{\phi}^{\mathrm{WAE}}\langle \theta ,\mathbf{z}\rangle$  have been proposed which (under R1) avoid the score-function terms in (4) and hence (a) exhibit a stable signal-to-noise ratio as  $K\to \infty$  and (b) can achieve zero variance if  $q_{\phi} = \pi_{\theta}$  (because then  $\nabla_{\psi}\equiv 0$  since  $w_{\psi}$  is constant).

- IWAE-STL. The 'sticking-the-landing' IWAE (IWAE-STL) gradient proposed by Roeder et al. (2017) heuristically ignores the score function terms (this introduces bias relative to  $\overline{\nabla}_{\phi}^{\mathrm{IWAE}}\langle \phi, \mathbf{z}\rangle$  whenever  $K > 1$  as shown in Tucker et al. (2019)):

$$
\widehat {\nabla} _ {\phi} ^ {\mathrm {I W A E - S T L}} \langle \theta , \mathbf {z} \rangle := \sum_ {k = 1} ^ {K} \frac {w _ {\psi} \left(z ^ {k}\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(z ^ {l}\right)} \boldsymbol {\nabla} _ {\psi} \left(z ^ {k}\right). \tag {5}
$$

- IWAE-DREG. The 'doubly-reparametrised' IWAE (IWAE-DREG) gradient proposed by Tucker et al. (2019) removes the score-function terms through Lemma 1 (i.e. this does not introduce bias relative to  $\widehat{\nabla}_{\phi}^{\mathrm{IWAE}}\langle \phi, \mathbf{z}\rangle$ ):

$$
\widehat {\nabla} _ {\phi} ^ {\mathrm {I W A E - D R E G}} \left\langle \theta , \mathbf {z} \right\rangle := \sum_ {k = 1} ^ {K} \left(\frac {w _ {\psi} \left(z ^ {k}\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(z ^ {l}\right)}\right) ^ {2} \boldsymbol {\nabla} _ {\psi} \left(z ^ {k}\right). \tag {6}
$$

# 2.3 REWEIGHTED WAKE-SLEEP (RWS)

The reweighted wake-sleep (RWS) algorithm was proposed in Bornschein & Bengio (2015).<sup>1</sup> Letting  $\mathrm{KL}(p\| q)\coloneqq \int_{\mathbb{Z}}\log (p(z) / q(z))q(z)\mathrm{d}z$  is the Kullback-Leibler (KL)-divergence from  $p$  to  $q$ , the RWS algorithm seeks to optimise  $\psi = (\theta ,\phi)$  as

$$
\theta^ {\star} := \theta^ {\mathrm {M L}} = \arg \max  _ {\theta} \log \mathcal {Z} _ {\theta},
$$

$$
\phi^ {\star} := \arg \min  _ {\phi} \mathrm {K L} \left(\pi_ {\theta^ {\star}} \| q _ {\phi}\right).
$$

The  $\theta$ - and  $\phi$ -gradients

$$
\left[ \begin{array}{c} \nabla_ {\theta} \log \mathcal {Z} _ {\theta} \\ - \nabla_ {\phi} \operatorname {K L} (\pi_ {\theta} \| q _ {\phi}) \end{array} \right] = \pi_ {\theta} \binom {\nabla_ {\theta} \log \gamma_ {\theta}} {\nabla_ {\phi} \log q _ {\phi}}, \tag {7}
$$

are usually intractable and therefore approximated by replacing  $\pi_{\theta}$  by the self-normalised importance sampling approximation  $\hat{\pi}_{\theta}\langle \phi ,\mathbf{z}\rangle$  (note that this does not need R1):

$$
\left[ \begin{array}{l} \widehat {\nabla} _ {\theta} ^ {\mathrm {R W S}} \langle \phi , \mathbf {z} \rangle \\ \widehat {\nabla} _ {\phi} ^ {\mathrm {R W S}} \langle \theta , \mathbf {z} \rangle \end{array} \right] := \sum_ {k = 1} ^ {K} \frac {w _ {\psi} \left(z ^ {k}\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(z ^ {l}\right)} \left[ \begin{array}{l} \nabla_ {\theta} \log \gamma_ {\theta} \left(z ^ {k}\right) \\ \nabla_ {\phi} \log q _ {\phi} \left(z ^ {k}\right) \end{array} \right]. \tag {8}
$$

Since (8) relies on self-normalised importance sampling, it biased relative to (7). However, by Part 2 of Proposition 1 the bias of the  $\theta$ -gradient  $\widehat{\nabla}_{\theta}^{\mathrm{RWS}}\langle \phi ,\mathbf{z}\rangle = \widehat{\nabla}_{\theta}^{\mathrm{IWAE}}\langle \phi ,\mathbf{z}\rangle$  relative to  $\nabla_{\theta}\log \mathcal{Z}_{\theta}$  decays as  $\mathcal{O}(K^{-1})$ . Appendix A discusses the impact of the bias on the  $\phi$ -gradients.

The optimisation of  $\theta$  and  $\phi$  is carried out simultaneously. This is because (a) a better proposal  $q_{\phi}$  reduces both bias and variance of (self-normalised) importance-sampling approximations and can therefore be leveraged for reducing the bias and variance of the  $\theta$ -gradients and (b) this strategy reduces the computational cost because the same set of particles  $\mathbf{z}$  and weights  $\{w_{\psi}(z^k)\}_{k=1}^K$  is shared by both gradients. However, this simultaneous optimisation is often viewed as the main drawback of RWS because there is no joint objective (for both  $\theta$  and  $\phi$ ).

RWS-DREG. Under R1, Tucker et al. (2019) proposed the following 'doubly-reparametrised' RWS (RWS-DREG) gradient which is equal to  $\widehat{\nabla}_{\phi}^{\mathrm{RWS}}\langle \theta ,\mathbf{z}\rangle$  in expectation and is derived by applying Lemma 1 to the latter:

$$
\widehat {\nabla} _ {\phi} ^ {\mathrm {R W S - D R E G}} \langle \theta , \mathbf {z} \rangle := \sum_ {k = 1} ^ {K} \left[ \frac {w _ {\psi} \left(z ^ {k}\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(z ^ {l}\right)} - \left(\frac {w _ {\psi} \left(z ^ {k}\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(z ^ {l}\right)}\right) ^ {2} \right] \boldsymbol {\nabla} _ {\psi} \left(z ^ {k}\right). \tag {9}
$$

# 3 AISLE: A UNIFIED ADAPTIVE IMPORTANCE-SAMPLING FRAMEWORK

# 3.1 OBJECTIVE

If  $\theta$  is fixed, the RWS algorithm reduces to an adaptive importance-sampling scheme which optimises the proposal distribution by minimising the KL-divergence from the target distribution  $\pi_{\theta}$  to the proposal  $q_{\phi}$  (see, e.g., Douc et al., 2007; Cappé et al., 2008). If instead  $\phi$  is fixed, the RWS algorithm reduces to a stochastic-approximation algorithm for estimating the MLE of the generative-model parameters  $\theta$ . The advantage of optimising  $\theta$  and  $\phi$  simultaneously is that (a) Monte Carlo samples used to approximate the  $\theta$ -gradient can be re-used to approximate the  $\phi$ -gradient and (b) optimising  $\phi$  typically reduces the error (both in terms of bias and variance) of the  $\theta$ -gradient approximation.

However, adapting the proposal distribution  $q_{\phi}$  in importance-sampling schemes need not necessarily be based on minimising the KL-divergence. Numerous other techniques exist in the literature (e.g. Geweke, 1989; Evans, 1991; Oh & Berger, 1992; Richard & Zhang, 2007; Cornebise et al., 2008) and may sometimes be preferable. Indeed, another popular approach with strong theoretical support is based on minimising the  $\chi^2$ -divergence (see, e.g., Deniz Akyildiz & Miguez, 2019). Based on this insight, we slightly generalise the RWS-objective as

$$
\theta^ {\star} := \arg \max  _ {\theta} \log \mathcal {Z} _ {\theta} (= \theta^ {\mathrm {M L}}),
$$

$$
\phi^ {\star} := \arg \min  _ {\phi} \mathrm {D} _ {\mathrm {f}} \left(\pi_ {\theta^ {*}} \| q _ {\phi}\right). \tag {10}
$$

Here,  $\mathrm{Df}(p\| q)\coloneqq \int_{\mathbb{Z}}\mathrm{f}(p(z) / q(z))q(z)\mathrm{d}z$  is some f-divergence from  $p$  to  $q$ . We reiterate that alternative approaches for optimising  $\phi$  (which do not minimise f-divergences) could be used. However, we state (10) for concreteness as it suffices for the remainder of this work; we call the resulting algorithm adaptive importance sampling for learning (AISLE). We stress again that AISLE is not introduced with the aim or claim of proposing a new algorithms but to formalise the argument that the adaptive importance-sampling paradigm avoids the drawbacks from Remark 1 thus making it preferable to the multi-sample objective paradigm.

# 3.2  $\theta$ -GRADIENT

Optimisation is again performed via a stochastic gradient-ascent. The intractable  $\theta$ -gradient  $\nabla_{\theta}\log \mathcal{Z}_{\theta} = \pi_{\theta}(\nabla_{\theta}\log \gamma_{\theta})$  is approximated as in RWS, i.e. for  $\mathbf{z}\sim q_{\phi}^{\otimes K}$ :

$$
\widehat {\nabla} _ {\theta} ^ {\text {A I S L E}} \langle \phi , \mathbf {z} \rangle := \widehat {\nabla} _ {\theta} ^ {\text {R W S}} \langle \phi , \mathbf {z} \rangle = \widehat {\nabla} _ {\theta} ^ {\text {I W A E}} \langle \phi , \mathbf {z} \rangle .
$$

The  $\theta$ -gradient is thus the same for all algorithms discussed in this work although the IWAE-paradigm views it as an unbiased gradient for a biased objective while AISLE (and RWS) interpret it as a self-normalised importance-sampling (and hence biased) approximation of the gradient  $\nabla_{\theta}\log \mathcal{Z}_{\theta}$  for the 'exact' objective.

# 3.3  $\phi$ -GRADIENT SPECIAL CASE I: RWS AND IWAE-STL

The  $\phi$ -gradients depend on the particular choice of f-divergence in (10). By construction, we recover RWS as a special case of AISLE if we define the f-divergence through  $f(y) \coloneqq y \log y$  because in this case  $D_{f}(p \| q) = KL(p \| q)$  reduces to the KL-divergence. Our main contribution in this subsection is to show that a more principled application of the identity from Lemma 1 leads to the IWAE-STL gradient from (5).

To derive the AISLE  $\phi$ -gradients for this divergence we note that

$$
- \nabla_ {\phi} \operatorname {K L} \left(\pi_ {\theta} \| q _ {\phi}\right) = \pi_ {\theta} \left(\nabla_ {\phi} \log q _ {\phi}\right), \tag {11}
$$

which, under R1, by Lemma 1 with  $f_{\psi} = w_{\psi}$ , can be written as

$$
\pi_ {\theta} \left(\nabla_ {\phi} \log q _ {\phi}\right) = q _ {\phi} \left(w _ {\psi} \nabla_ {\phi} \log q _ {\phi}\right) / \mathcal {Z} _ {\theta} = q _ {\phi} \left(w _ {\psi} \boldsymbol {\nabla} _ {\psi}\right) / \mathcal {Z} _ {\theta} = \pi_ {\theta} (\boldsymbol {\nabla} _ {\psi}). \tag {12}
$$

We then obtain practical approximations of these gradients by plugging in  $\hat{\pi}_{\theta}\langle \phi ,\mathbf{z}\rangle$  for  $\pi_{\theta}$

- AISLE-KL-NOREP/RWS. Without relying on any reparametrisation, (11) yields the following gradient, which clearly equals  $\widehat{\nabla}_{\phi}^{\mathrm{RWS}}\langle \theta ,\mathbf{z}\rangle$

$$
\widehat {\nabla} _ {\phi} ^ {\text {A I S L E - K L - N O R E P}} \left\langle \theta , \mathbf {z} \right\rangle := \sum_ {k = 1} ^ {K} \frac {w _ {\psi} \left(z ^ {k}\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(z ^ {l}\right)} \nabla_ {\phi} \log q _ {\phi} \left(z ^ {k}\right). \tag {13}
$$

- AISLE-KL. Using the reparametrisation from R1, (12) yields the gradient:

$$
\widehat {\nabla} _ {\phi} ^ {\text {A I S L E - K L}} \langle \theta , \mathbf {z} \rangle := \sum_ {k = 1} ^ {K} \frac {w _ {\psi} \left(z ^ {k}\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(z ^ {l}\right)} \boldsymbol {\nabla} _ {\psi} \left(z ^ {k}\right). \tag {14}
$$

We thus arrive at the following result which demonstrates that IWAE-STL can be derived in a principled manner from AISLE, i.e. without the need for a multi-sample objective.

Proposition 2. For any  $(\theta, \phi, \mathbf{z})$ ,  $\widehat{\nabla}_{\phi}^{\mathrm{AISLE - KL}}\langle \theta, \mathbf{z}\rangle = \widehat{\nabla}_{\phi}^{\mathrm{IWAE - STL}}\langle \theta, \mathbf{z}\rangle$ .

Proposition 2 thus provides a theoretical basis for IWAE-STL which was previously viewed as an alternative gradient for IWAE for which it is biased and only heuristically justified. Furthermore, the fact that IWAE-STL exhibited good empirical performance in Tucker et al. (2019) even in an example in which RWS broke down, suggests that this breakdown may not be due to RWS' lack of optimising a joint objective as previously conjectured.

Finally, recall that Tucker et al. (2019) obtained an alternative 'doubly-reparametrised' RWS  $\phi$ -gradient  $\widehat{\nabla}_{\phi}^{\mathrm{RWS - DREG}}\langle \theta ,\mathbf{z}\rangle$  given in (9) by first replacing the exact (but intractable)  $\phi$ -gradient from (11) by the self-normalised importance-sampling approximation  $\widehat{\nabla}_{\phi}^{\mathrm{RWS}}\langle \theta ,\mathbf{z}\rangle$  and then applying the identity from Lemma 1. Note that this may result in a variance reduction but does not change the bias of the gradient estimator. In contrast, AISLE-KL is derived by first applying Lemma 1 to the exact (RWS)  $\phi$ -gradient and then approximating the resulting expression. This can potentially reduce both bias and variance.

# 3.4  $\phi$ -GRADIENT SPECIAL CASE II: IWAE-DREG

We now demonstrate that the IWAE-DREG gradient can be recovered as a special case of AISLE (up to a proportionality constant). To establish this relationship, we take  $\mathrm{f}(y) \coloneqq (y - 1)^2$  so that  $\mathrm{Df}(p\| q) = \chi^2(p\| q) \coloneqq \int_{\mathbb{Z}}([p(z) / q(z)] - 1)^2q(z)\mathrm{d}z = \int_{\mathbb{Z}}[p(z) / q(z)]p(z)\mathrm{d}z - 1$ , is the  $\chi^2$ -divergence. Minimising this divergence is natural in importance sampling since  $\chi^2(\pi_\theta \| q_\phi) = \operatorname{var}_{z \sim q_\phi}[w_\psi / \mathcal{Z}_\theta]$  is the variance of the importance weights.

To derive the AISLE  $\phi$ -gradients for this divergence we note that

$$
- \nabla_ {\phi} \chi^ {2} \left(\pi_ {\theta} \| q _ {\phi}\right) = - \pi_ {\theta} \left(\nabla_ {\phi} w _ {\psi}\right) / \mathcal {Z} _ {\theta} = \pi_ {\theta} \left(w _ {\psi} \nabla_ {\phi} \log q _ {\phi}\right) / \mathcal {Z} _ {\theta}, \tag {15}
$$

which, under  $\mathbf{R1}$ , by Lemma 1 with  $f_{\psi} = w_{\psi}^{2}$ , can be written as

$$
\begin{array}{l} \pi_ {\theta} \left(w _ {\psi} \nabla_ {\phi} \log q _ {\phi}\right) / \mathcal {Z} _ {\theta} = q _ {\phi} \left(w _ {\psi} ^ {2} \nabla_ {\phi} \log q _ {\phi}\right) / \mathcal {Z} _ {\theta} ^ {2} \\ = q _ {\phi} \left(w _ {\psi} ^ {2} \nabla_ {\phi} \left[ \log \circ w _ {\psi^ {\prime}} ^ {2} \circ h _ {\phi} \right] | _ {\psi^ {\prime} = \psi} \circ h _ {\phi} ^ {- 1}\right) / \mathcal {Z} _ {\theta} ^ {2} = \pi_ {\theta} \left(2 w _ {\psi} \nabla_ {\psi}\right) / \mathcal {Z} _ {\theta}. \tag {16} \\ \end{array}
$$

Again plugging in  $\hat{\pi}_{\theta}\langle \phi ,\mathbf{z}\rangle$  for  $\pi_{\theta}$  and  $\widehat{\mathcal{Z}}_{\theta}\langle \phi ,\mathbf{z}\rangle$  for  $\mathcal{Z}_{\theta}$  yields the following approximations.

- AISLE- $\chi^2$ -NOREP. Without relying on any reparametrisation, (15) yields the following gradient which is also proportional to the 'score gradient' from Dieng et al. (2017, Appendix G):

$$
\widehat {\nabla} _ {\phi} ^ {\text {A I S L E -} \chi^ {2} - \text {N O R E P}} \langle \theta , \mathbf {z} \rangle := K \sum_ {k = 1} ^ {K} \left(\frac {w _ {\psi} \left(z ^ {k}\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(z ^ {l}\right)}\right) ^ {2} \nabla_ {\phi} \log q _ {\phi} \left(z ^ {k}\right). \tag {17}
$$

- AISLE- $\chi^2$ . Using the reparametrisation from R1, (16) yields the gradient:

$$
\widehat {\nabla} _ {\phi} ^ {\mathrm {A I S L E -} \chi^ {2}} \langle \theta , \mathbf {z} \rangle := 2 K \sum_ {k = 1} ^ {K} \left(\frac {w _ {\psi} \left(z ^ {k}\right)}{\sum_ {l = 1} ^ {K} w _ {\psi} \left(z ^ {l}\right)}\right) ^ {2} \boldsymbol {\nabla} _ {\psi} \left(z ^ {k}\right). \tag {18}
$$

We thus arrive at the following result which demonstrates that IWAE-DREG can be derived (up to the proportionality factor  $2K$ ) in a principled manner from AISLE, i.e. without the need for a multi-sample objective.

Proposition 3. For any  $(\theta, \phi, \mathbf{z})$ ,  $\widehat{\nabla}_{\phi}^{\mathrm{AISLE}-\chi^{2}}\langle \theta, \mathbf{z} \rangle = 2K\widehat{\nabla}_{\phi}^{\mathrm{IWAE-DREG}}\langle \theta, \mathbf{z} \rangle$ .

Note that if the implementation normalises the gradients, e.g. as effectively done by ADAM (Kingma & Ba, 2015), the constant factor cancels out and AISLE- $\chi^2$  becomes equivalent to IWAE-DREG. Otherwise (e.g. in plain stochastic gradient-ascent) Proposition 3 shows that the learning rate needs to be scaled as  $\mathcal{O}(K)$  for the IWAE or IWAE-DREG  $\phi$ -gradients.

# 4 CONCLUSION

We have shown that the adaptive-importance sampling paradigm of the reweighted wake-sleep (RWS) (Bornshein & Bengio, 2015) is preferable to the multi-sample objective paradigm of importance weighted autoencoders (IWAEs) (Burda et al., 2016) because the former achieves all the goals of the latter whilst avoiding its drawbacks. To formalise this argument, we have introduced a simple, unified adaptive-importance-sampling framework termed adaptive importance sampling for learning (AISLE) (which slightly generalises the RWS algorithm) and have proved that AISLE allows us to derive the 'sticking-the-landing' IWAE (IWAE-STL) gradient from Roeder et al. (2017) and the 'doubly-reparametrised' IWAE (IWAE-DREG) gradient from Tucker et al. (2019) as special cases.

We hope that this work highlights the potential for further improving variational techniques by drawing upon the vast body of research on (adaptive) importance sampling in the computational statistics literature. Conversely, the methodological connections established in this work may also serve to emphasise the utility of the reparametrisation trick from Kingma & Welling (2014); Tucker et al. (2019) to computational statisticians.

In a companion article, we are extending the present work to the variational sequential Monte Carlo methods from Maddison et al. (2017); Le et al. (2018); Naesseth et al. (2018) and to the tensor Monte Carlo approach from Aitchison (2018).

# REFERENCES

Laurence Aitchison. Tensor Monte Carlo: particle methods for the GPU era. arXiv e-prints, art. arXiv:1806.08593, Jun 2018.  
Christophe Andrieu and Gareth O Roberts. The pseudo-marginal approach for efficient Monte Carlo computations. The Annals of Statistics, 37(2):697-725, 2009.  
Christophe Andrieu, Arnaud Doucet, and Roman Holenstein. Particle Markov chain Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(3):269-342, 2010. With discussion.  
Robert Bamler, Cheng Zhang, Manfred Opper, and Stephan Mandt. Perturbative black box variational inference. Advances in Neural Information Processing Systems (NeurIPS), pp. 5079-5088, 2017.  
Jörg Bornschein and Yoshua Bengio. Reweighted wake-sleep. In 3rd International Conference on Learning Representations (ICLR), 2015.  
Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. In 4th International Conference on Learning Representations (ICLR), 2016.  
Olivier Cappé, Randal Douc, Arnaud Guillin, Jean-Michel Marin, and Christian P Robert. Adaptive importance sampling in general mixture classes. Statistics and Computing, 18 (4):447-459, 2008.  
Julien Cornebise, Éric Moulines, and Jimmy Olsson. Adaptive methods for sequential importance sampling with application to state space models. *Statistics and Computing*, 18 (4):461-480, 2008.  
Chris Cremer, Quaid Morris, and David Duvenaud. Reinterpreting importance-weighted autoencoders. In 5th International Conference on Learning Representations (ICLR), 2017.  
Ömer Deniz Akyildiz and Joaquín Miguez. Convergence rates for optimised adaptive importance samplers. arXiv e-prints, art. arXiv:1903.12044, 2019.  
Adji Bousso Dieng, Dustin Tran, Rajesh Ranganath, John Paisley, and David Blei. Variational inference via  $\chi$  upper bound minimization. Advances in Neural Information Processing Systems (NeurIPS), pp. 2732-2741, 2017.  
Justin Domke and Daniel R Sheldon. Importance weighting and variational inference. Advances in Neural Information Processing Systems (NeurIPS), pp. 4475-4484, 2018.  
Randal Douc, Arnaud Guillin, Jean-Michel Marin, and Christian P Robert. Convergence of adaptive mixtures of importance sampling schemes. The Annals of Statistics, 35(1): 420-448, 2007.  
Michael Evans. Adaptive importance sampling and chaining. Statistical Numerical Integration, Contemporary Mathematics, 115:137-143, 1991.  
Axel Finke. On extended state-space constructions for Monte Carlo methods. PhD thesis, Department of Statistics, University of Warwick, UK, 2015.  
John Geweke. Bayesian inference in econometric models using Monte Carlo integration. *Econometrica*, 57(6):1317-1339, 1989.  
Edward L Ionides. Truncated importance sampling. Journal of Computational and Graphical Statistics, 17(2):295-311, 2008.  
Diederik P Kingma and Jimmy Lei Ba. ADAM: A method for stochastic optimization. In 3rd International Conference on Learning Representations (ICLR), 2015.  
Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. In 2nd International Conference on Learning Representations (ICLR), 2014.

Augustine Kong, Jun S. Liu, and Wing Hung Wong. Sequential imputations and Bayesian missing data problems. Journal of the American Statistical Association, 89(425):278-288, 1994.  
Tuan Anh Le, Maximilian Igl, Tom Rainforth, Tom Jin, and Frank Wood. Auto-encoding sequential Monte Carlo. In 6th International Conference on Learning Representations (ICLR), 2018.  
Tuan Anh Le, Adam R Kosiorek, N Siddharth, Yee Whye Teh, and Frank Wood. Revisiting reweighted wake-sleep for models with stochastic control flow. In Proceedings of the 35th Conference on Uncertainty in Artificial Intelligence (UAI), 2019.  
Anthony Lee. On auxiliary variables and many-core architectures in computational statistics. PhD thesis, Department of Statistics, University of Oxford, UK, 2011.  
Jun S Liu. Metropolized independent sampling with comparisons to rejection sampling and importance sampling. Statistics and Computing, 6(2):113-119, 1996.  
Jun S. Liu. *Monte Carlo Strategies in Scientific Computing*. Springer Series in Statistics. Springer, 2001.  
Chris J Maddison, John Lawson, George Tucker, Nicolas Heess, Mohammad Norouzi, Andriy Mnih, Arnaud Doucet, and Yee Whye Teh. Filtering variational objectives. Advances in Neural Information Processing Systems (NeurIPS), pp. 6573-6583, 2017.  
Christian A Naesseth, Scott W Linderman, Rajesh Ranganath, and David M Blei. Variational sequential Monte Carlo. In 21st International Conference on Artificial Intelligence and Statistics (AISTATS), 2018.  
Man-Suk Oh and James O Berger. Adaptive importance sampling in Monte Carlo integration. Journal of Statistical Computation and Simulation, 41(3-4):143-168, 1992.  
John Paisley, David Blei, and Michael Jordan. Variational Bayesian inference with stochastic search. In 29th International Conference on Machine Learning (ICML), 2012.  
Tom Rainforth, Adam R Kosiorek, Tuan Anh Le, Chris J Maddison, Maximilian Igl, Frank Wood, and Yee Whye Teh. Tighter variational bounds are not necessarily better. In Bayesian Deep Learning (NeurIPS 2018 workshop), 2018.  
Jean-François Richard and Wei Zhang. Efficient high-dimensional importance sampling. Journal of Econometrics, 141(2):1385-1411, 2007.  
Geoffrey Roeder, Yuhuai Wu, and David K Duvenaud. Sticking the landing: Simple, lowvariance gradient estimators for variational inference. Advances in Neural Information Processing Systems (NeurIPS), pp. 6925-6934, 2017.  
George Tucker, Dieterich Lawson, Shixiang Gu, and Chris J Maddison. Doubly reparameterized gradient estimators for Monte Carlo objectives. In 7th International Conference on Learning Representations (ICLR), 2019.  
Ming Xu, Matias Quiroz, Robert Kohn, and Scott A Sisson. Variance reduction properties of the reparameterization trick. In The 22nd International Conference on Artificial Intelligence and Statistics (AISTATS), pp. 2711-2720, 2019.
