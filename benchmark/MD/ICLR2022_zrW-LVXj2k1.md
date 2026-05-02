# ON THE BENEFITS OF MAXIMUM LIKELIHOOD ESTIMATION FOR REGRESSION AND FORECASTING

Anonymous authors

Paper under double-blind review

# ABSTRACT

We advocate for a practical Maximum Likelihood Estimation (MLE) approach towards designing loss functions for regression and forecasting, as an alternative to the typical approach of direct empirical risk minimization on a specific target metric. The MLE approach is better suited to capture inductive biases such as prior domain knowledge in datasets, and can output post-hoc estimators at inference time that can optimize different types of target metrics. We present theoretical results to demonstrate that our approach is competitive with any estimator for the target metric under some general conditions. In two example practical settings, Poisson and Pareto regression, we show that our competitive results can be used to prove that the MLE approach has better excess risk bounds than directly minimizing the target metric. We also demonstrate empirically that our method instantiated with a well-designed general purpose mixture likelihood family can obtain superior performance for a variety of tasks across time-series forecasting and regression datasets with different data distributions.

# 1 INTRODUCTION

The task of fitting a regression model for a response variable  $y$  against a covariate vector  $\pmb{x} \in \mathbb{R}^d$  is ubiquitous in supervised learning in both linear and non-linear settings (Lathuilière et al., 2020; Mohri et al., 2018) as well as non-i.i.d settings like multi-variate forecasting (Salinas et al., 2020; Wang et al., 2019). The end goal in regression and forecasting problems is often to use the resulting model to obtain good performance in terms of some target metric of interest on the population level (usually measured on a previously unseen test set). The mean-squared error or the mean absolute deviation are examples of common target metrics.

In this paper, our focus is on the choice of loss function that is used to train such models, which is an important question that is often overlooked, especially in the deep neural networks context where the emphasis is more on the choice of network architecture (Lathuilière et al., 2020).

Perhaps the most common method used by practitioners for choosing the loss function for a particular regression model is to directly use the target metric of interest as the loss function for empirical risk minimization (ERM) over a function class on the training set. We denote this approach for choosing a loss function as Target Metric Optimization (TMO). This is especially more common with the advent of powerful general purpose function optimizers like deep networks and has also been rigorously analyzed for simpler function classes (Mohri et al., 2018).

Target Metric Optimization seems like a reasonable approach - if the practitioner knows about the target metric of interest for prediction using the model, it seems intuitive to optimize for the same objective on the training data. Prior work (both theoretical and applied) has both advocated for and argued against TMO for regression problems. Many prominent works on regression (Goldberger et al., 1964; Lecue & Mendelson, 2016) use the TMO approach, though most of them assume that the data is well behaved (e.g. sub-Gaussian noise). In terms of applications, many recent works on time-series forecasting (Wu et al., 2020; Oreshkin et al., 2019; Sen et al., 2019) also use the TMO approach directly on the target metric. On the other hand, the robust regression literature has long advocated for not using the target metric directly for ERM in the case of contamination or heavy tailed response/covariate behaviour (Huber, 1992; Hsu & Sabato, 2016; Zhu & Zhou, 2021; Lugosi & Mendelson, 2019a; Audibert et al., 2011; Brownlees et al., 2015) on account of its suboptimal high-probability risk bounds. However, as noted in (Prasad et al., 2020), many of these methods

are either not practical (Lugosi & Mendelson, 2019a; Brownlees et al., 2015) or have sub-optimal empirical performance (Hsu & Sabato, 2016). Even more practical methods such as (Prasad et al., 2020) would lead to sufficiently more computational overhead over standard TMO.

Another well known approach for designing a loss function is Maximum Likelihood Estimation (MLE). Here one assumes that the conditional distribution of  $y$  given  $\pmb{x}$  belongs to a family of distributions  $p(y|x;\pmb{\theta})$  parameterized by  $\pmb{\theta} \in \Theta$  (McCullagh & Nelder, 2019). Then one can choose the negative log likelihood as the loss function to optimize using the training set, to obtain an estimate  $\hat{\pmb{\theta}}_{\mathrm{mle}}$ . This approach is sometimes used in the forecasting literature (Salinas et al., 2020; Davis & Wu, 2009) where the choice of a likelihood can encode prior knowledge about the data. For instance a negative binomial distribution can be used to model count data. During inference, given a new instance  $x'$ , one can output the statistic from  $p(y|x'; \hat{\pmb{\theta}}_{\mathrm{mle}})$  that optimizes the target metric, as the prediction value (Gneiting, 2011). MLE also seems like a reasonable approach for loss function design - it is folklore that the MLE is asymptotically optimal for parameter estimation, in terms of having the smallest asymptotic variance among all estimators (Heyde, 1978; Rao, 1963), when the likelihood is well-specified. However, much less is known about finite-sample, fixed-dimension analysis of MLE, which is the typical regime of interest for the regression problems we consider in this paper. An important practical advantage for MLE is that model training is agnostic to the choice of the target metric - the same trained model can output estimators for different target metrics at inference time. Perhaps the biggest argument against the MLE approach is the requirement of knowing the likelihood distribution family. We address both these topics in Section 5.

Both TMO and MLE can be viewed as offering different approaches to selecting the loss function for a given regression model. In this paper, we argue that in several settings, both from a practical and theoretical perspective, MLE might be a better approach than TMO. This result might not be immediately obvious apriori - while MLE does benefit from prior knowledge of the distribution, TMO also benefits from prior knowledge of the target metric at training time.

Our main contributions are as follows:

Competitiveness of MLE: In Section 3, we prove that under some general conditions on the family of distributions and a property of interest, the MLE approach is competitive with any estimator for the property. We show that this result can be applied to fixed design regression problems in order to prove that MLE can competitive (up to logarithmic terms) with any estimator in terms of excess square loss risk, under some assumptions.

Example Applications: In Section 4.1, we apply our general theorem to prove an excess square loss bound for the MLE estimator for Poisson regression with the identity link (Nelder & Wedderburn, 1972; Lawless, 1987). We show that these bounds can be better than those of the TMO estimator, which in this case is least-squares regression. Then in Section 4.2, we show a similar application in the context of Pareto regression i.e  $y|x$  follows a Pareto distribution. We show that MLE can be competitive with robust estimators like the one in (Hsu & Sabato, 2016) and therefore can be better than TMO (least-squares).

Empirical Results: We propose the use of a general purpose mixture likelihood family (see Section 5) that can capture a wide variety of prior knowledge across datasets, including zero-inflated or bursty data, count data, sub-Gaussian continuous data as well as heavy tailed data, through different choices of (learnt) mixture weights. Then we empirically show that the MLE approach with this likelihood can outperform ERM for many different commonly used target metrics like WAPE, MAPE and RMSE (Hyndman & Koehler, 2006) for two popular forecasting and two regression datasets. Moreover the MLE approach is also shown to have better probabilistic forecasts (measured by quantile losses (Wen et al., 2017)) than quantile regression (Koenker & Bassett Jr, 1978; Gasthaus et al., 2019; Wen et al., 2017) which is the TMO approach in this case.

# 2 PRIOR WORK ON MLE

Maximum likelihood estimators (MLE) have been studied extensively in statistics starting with the work of Wald (1949); Redner (1981), who showed the maximum likelihood estimates are asymptotically consistent for parametric families. Fahrmeir & Kaufmann (1985) showed asymptotic normality for MLE for generalized linear models. It is also known that under some regularity

assumptions, MLE achieves the Cramer-Rao lower bound asymptotically (Lehmann & Casella, 2006). However, we note that none of these asymptotic results directly yield finite sample guarantees.

Finite sample guarantees have been shown for certain problem scenarios. Geer & van de Geer (2000); Zhang (2006) provided uniform convergence bounds in Hellinger distance for maximum likelihood estimation. These ideas were recently used by Foster & Krishnamurthy (2021) to provide algorithms for contextual bandits. There are other works which study MLE for non-parametric distributions e.g., Dumbgen & Rufibach (2009) showed convergence rates for log-concave distributions. There has been some works (Sur & Candès, 2019; Bean et al., 2013; Donoho & Montanari, 2016; El Karoui, 2018) that show that MLE can be sub-optimal for high dimensional regression i.e when the dimension grows with the number of samples. In this work we focus on the setting where the dimension does not scale with the number of samples.

Our MLE results differ from the above work as we provide finite sample competitiveness guarantees. Instead of showing that the maximum likelihood estimator converges in some distance metric, we show that under some mild assumptions it can work as well as other estimators. Hence, our methods are orthogonal to known well established results in statistics.

Perhaps the closest to our work is the competitiveness result of Acharya et al. (2017), who showed that MLE is competitive when the size of the output alphabet is bounded and applied to profile maximum likelihood estimation. In contrast, our work applies for unbounded output alphabets and can provide stronger guarantees in many scenarios.

# 3 COMPETITIVENESS OF MLE

In this section, we will show that under some reasonable assumptions on the likelihood family, the MLE is competitive with any estimator in terms of estimating any property of a distribution from the family. We will then show that this result can be applied to derive bounds on the MLE in some fixed design regression settings that can be better than that of TMO. We will first setup some notation.

Notation: Given a positive semi-definite symmetric matrix  $M$ ,  $\| \pmb{x} \|_M \coloneqq \pmb{x}^T \pmb{M} \pmb{x}$  is the matrix norm of the vector  $\pmb{x}$ .  $\lambda(M)$  denotes an eigenvalue of a symmetric square matrix  $M$ ; specifically  $\lambda_{\max}(M)$  and  $\lambda_{\min}(M)$  denote the maximum and minimum eigenvalues respectively. The letter  $f$  is used to denote general distributions. We use  $p$  to denote the conditional distribution of the response given the covariate.  $\| \cdot \|_1$  will be overloaded to denote the  $\ell_1$  norm between two probability distributions for example  $\| p - p' \|_1$ .  $D_{\mathrm{KL}}(p_1; p_2)$  will be used to denote the KL-divergence between the two distributions. If  $\mathcal{Z}$  is a set equipped with a norm  $\| \cdot \|$ , then  $\mathcal{N}(\epsilon, \mathcal{Z})$  will denote an  $\epsilon$ -net i.e. any point  $z \in \mathcal{Z}$  has a corresponding point  $z' \in \mathcal{N}(\epsilon, \mathcal{Z})$  s.t.  $\| z - z' \| \leq \epsilon$ .  $\mathbb{B}_r^d$  denotes the sphere centered at the origin with radius  $r$  and  $\mathbb{S}_r^{d-1}$  denotes its surface. We define  $[n] := \{1, 2, \dots, n\}$ .

General Competitiveness: We first consider a general family of distributions  $\mathcal{F}$  over the space  $\mathcal{Z}$ . For a sample  $z \sim f$  (for  $z \in \mathcal{Z}$  and  $f \in \mathcal{F}$ ), the MLE distribution is defined as  $f_z = \operatorname{argmax}_{f \in \mathcal{F}} f(z)$ . We are interested in estimating a property  $\pi : \mathcal{F} \to \mathcal{W}$  of these distributions from an observed sample. The following definition will be required to impose some joint conditions on the distribution family and the property being estimated, that are needed for our result.

Definition 1. The tuple  $(\mathcal{F},\pi)$ , where  $\mathcal{F}$  is a set of distributions and  $\pi : \mathcal{F} \to \mathcal{W}$  a property of those distributions, is said to be  $(T,\epsilon,\delta_1,\delta_2)$ -approximable, if there exists a set of distributions  $\tilde{\mathcal{F}} \subseteq \mathcal{F}$  s.t  $|\tilde{\mathcal{F}}| \leq T$  and for every  $f \in \mathcal{F}$ , there exists a  $\tilde{f}$  such that  $\|f - \tilde{f}\|_1 \leq \delta_1$  and  $\operatorname{Pr}_{z \sim f}\left(\left\| \pi(f_z) - \pi(\tilde{f}_z) \right\| \geq \epsilon\right) \leq \delta_2$ , where  $\tilde{f}_z = \operatorname{argmax}_{\tilde{f} \in \tilde{\mathcal{F}}} \tilde{f}(z)$  and  $\mathcal{W}$  has a norm  $\| \cdot \|$ .

The above definition states that the set of distributions  $\mathcal{F}$  has a finite  $\delta$ -cover,  $\tilde{\mathcal{F}}$  in terms of the  $\ell_1$  distance. Moreover the cover is such that solving MLE on the cover and applying the property  $\pi$  on the result of the MLE is not too far from  $\pi$  applied on the MLE over the whole set  $\mathcal{F}$ . This property is satisfied trivially if  $\mathcal{F}$  is finite. We note that it is also satisfied by some commonly used set of distributions and corresponding properties. Now we state our main result.

Theorem 1. Let  $\hat{\pi}$  be an estimator such that for any  $f\in \mathcal{F}$  and  $z\sim f$ $\operatorname *{Pr}(\| \pi (f) - \hat{\pi} (z)\| \geq \epsilon)\leq \delta$  Let  $\mathcal{F}_f$  be a subset of  $\mathcal{F}$  that contains  $f$  such that with probability at least  $1 - \delta$ $f_{z}\in \mathcal{F}_{f}$  and  $(\mathcal{F}_f,\pi)$  is  $(T,\epsilon ,\delta_1,\delta_2)$ -approximable. Then the MLE based estimator satisfies the following bound,

$$
\Pr (\| \pi (f) - \pi (f _ {z}) \| \geq 3 \epsilon) \leq (T + 3) \delta + \delta_ {1} + \delta_ {2}.
$$

We provide the proof of Theorem 1 in Appendix A<sup>1</sup>. We also provide a simpler version of this result for finite distribution families as Theorem 3 in Appendix A for the benefit of the reader.

Competitiveness in Fixed Design Regression: Theorem 1 can be used to show that MLE is competitive with respect to any estimator for square loss minimization in fixed design regression. We will first formally introduce the setting. Consider a fixed design matrix  $\mathbf{X} \in \mathbb{R}^{n \times d}$  where  $n$  is the number of samples and  $d$  the feature dimension. We will work in a setting where  $n \gg d$ . The target vector is a random vector given by  $y^n \in \mathbb{R}^n$ . Let  $y_i$  be the  $i$ -th coordinate of  $y^n$  and  $\boldsymbol{x}_i$  denote the  $i$ -th row of the design matrix. We assume that the target is generated from the conditional distribution given  $\boldsymbol{x}_i$  such that,

$$
y _ {i} \sim p (\cdot | \boldsymbol {x} _ {i}; \boldsymbol {\theta} ^ {*}), \boldsymbol {\theta} ^ {*} \in \Theta .
$$

We are interested in optimizing a target metric  $\ell(\cdot, \cdot)$  given an instance of the random vector  $y^n$ . The final objective is to optimize,

$$
\min  _ {h \in \mathcal {H}} \mathbb {E} _ {y _ {i} \sim p (\cdot | \boldsymbol {x} _ {i}; \boldsymbol {\theta} ^ {*})} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \ell (y _ {i}, h (\boldsymbol {x} _ {i})) \right],
$$

where  $\mathcal{H}$  is a class of functions. In this context, we are interested in comparing two approaches.

TMO (see (Mohri et al., 2018)). This is standard empirical risk minimization on the target metric where given an instance of the random vector  $y^n$  one outputs the estimator  $\hat{h} = \min_{h\in \mathcal{H}}\frac{1}{n}\sum_{i = 1}^{n}\ell (y_i,h(\pmb {x}_i))$ .

MLE and post-hoc inference (see (Gneiting, 2011)). In this method one first solves for the parameter in the distribution family that best explains the empirical data by MLE i.e.,

$$
\hat {\boldsymbol {\theta}} _ {\mathrm {m l e}} := \underset {\boldsymbol {\theta} \in \Theta} {\operatorname {a r g m i n}} \mathcal {L} (y ^ {n}; \theta), \text {w h e r e} \mathcal {L} (y ^ {n}; \theta) := \sum_ {i = 1} ^ {n} - \log p (y _ {i} | \boldsymbol {x} _ {i}; \boldsymbol {\theta})
$$

Then during inference given a sample  $\pmb{x}_i$  the predictor is defined as,  $\tilde{h}(\pmb{x}_i) := \operatorname*{argmin}_{\hat{y}} \mathbb{E}_{y \in p(\cdot | \pmb{x}_i; \hat{\pmb{\theta}}_{\mathrm{mle}})}[\ell(y, \hat{y})]$  or in other words we output the statistic from the MLE distribution that optimizes the loss function of interest. For instance if  $\ell$  is the square loss, then  $\tilde{h}(\pmb{x}_i)$  will be the mean of the conditional distribution  $p(y | \pmb{x}_i; \hat{\pmb{\theta}}_{\mathrm{mle}})$ .

We will prove a general result using Theorem 1 when the target metric  $\ell$  is the square loss and  $\mathcal{H}$  is a linear function class. Moreover, the true distribution  $p(\cdot |\boldsymbol{x}_i;\boldsymbol{\theta}^*)$  is such that  $\mathbb{E}[y_i] = \langle \boldsymbol{\theta}^*,\boldsymbol{x}_i\rangle$  for all  $i\in [n]$  i.e we are in the linear realizable setting.

In this case the quantity of interest is the excess square loss risk given by,

$$
\mathcal {E} (\boldsymbol {\theta}) := \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {y ^ {n}} \| y ^ {n} - \boldsymbol {X} \boldsymbol {\theta} \| _ {2} ^ {2} - \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {y ^ {n}} \| y ^ {n} - \boldsymbol {X} \boldsymbol {\theta} ^ {*} \| _ {2} ^ {2} = \| \boldsymbol {\theta} - \boldsymbol {\theta} ^ {*} \| _ {\Sigma} ^ {2}, \tag {1}
$$

where  $\Sigma := (\sum_{i} \boldsymbol{x}_{i} \boldsymbol{x}_{i}^{T}) / n$  is the normalized covariance matrix, and  $\theta^{*}$  is the population minimizer of the target metric over the linear function class. Now we are ready to state the main result.

Theorem 2. Consider a fixed design regression setting where the likelihood family is parameterized by  $\theta \in \Theta \subseteq \mathbb{B}_w^d$  and  $|\mathcal{N}(\epsilon, \Theta \cap \mathbb{B}_w^d)| \leq |\mathcal{N}(\epsilon, \mathbb{B}_w^d)|$  for a small enough  $\epsilon$ . Further the following conditions hold,

1.  $D_{\mathrm{KL}}(p(y_i;\pmb {\theta});p(y_i;\pmb {\theta}'))\leq L\| \pmb {\theta} - \pmb {\theta}'\| _2.$  
2. The negative log-likelihood  $\mathcal{L}(y^n;\pmb {\theta})$  as a function of  $\pmb{\theta}$  is  $\alpha$ -strongly convex and  $\beta$ -smooth, w.p at least  $1 - \delta$ .

Further suppose there exists an estimator  $\theta_{\mathrm{est}}$  such that  $\mathcal{E}(\theta_{\mathrm{est}}) \leq (c_1 + c_2 \log(1 / \delta))^{\eta} / n$ , where  $c_1, c_2$  are problem-dependent quantities and  $\eta > 0$ . Then the MLE estimator also satisfies,

$$
\mathcal {E} (\hat {\boldsymbol {\theta}} _ {\mathrm {m l e}}) = O \left(\frac {\left(c _ {1} + c _ {2} d \left(\log n + \log \left(w L \lambda_ {\max } (\Sigma)\right) + \log \left(\beta / \alpha\right) + \log \frac {1}{\delta}\right)\right) ^ {\eta}}{n}\right)
$$

w.p at least  $1 - \delta$

We provide the proof in Appendix C. The proof involves proving the conditions in Theorem 1 and bounding the size of the cover  $T$ .

In order to better understand Theorem 2, let us consider a typical case where there exists a possibly complicated estimator such that  $\mathcal{E}(\theta_{\mathrm{est}}) = O((d + \log (1 / \delta)) / n)$ . In this case the above theorem implies that MLE will be competitive with this estimator up to a log  $n$  factor. In many cases the MLE might be much simpler to implement than the original estimator but would essentially match the same error bound. We now provide specific examples in subsequent sections.

# 4 APPLICATIONS OF COMPETITIVENESS RESULT

In this section we will specialize to two examples, Poisson regression and Pareto regression. In both these examples we show that MLE can be better than TMO through the use of our competitive result in Theorem 2.

# 4.1 POISSON REGRESSION

We work in the fixed design setting in Section 3 and assume that the conditional distribution of  $y|x$  is Poisson i.e,

$$
p \left(y _ {i} = k \mid \boldsymbol {x} _ {i}; \boldsymbol {\theta} ^ {*}\right) = \frac {\mu_ {i} ^ {k} e ^ {- \mu_ {i}}}{k !} \quad \text {w h e r e ,} \quad \mu_ {i} = \langle \boldsymbol {\theta} ^ {*}, \boldsymbol {x} _ {i} \rangle > 0, \tag {2}
$$

for all  $i \in [n]$ . Poisson regression is a popular model for studying count data regression which naturally appears in many applications like demand forecasting (Lawless, 1987). Note that here we study the version of Poisson regression with the identity link function (Nelder & Wedderburn, 1972), while another popular variant is the one with exponential link function (McCullagh & Nelder, 2019). We choose the identity link function for a fair comparison of the two approaches as it is realizable for both the approaches under the linear function class i.e. the globally optimal estimator in terms of population can be obtained by both approaches. The exponential link function would make the problem non-realizable under a linear function class for the TMO approach.

We make the following natural assumptions. Let  $\Sigma = (\sum_{i=1}^{n} \boldsymbol{x}_i \boldsymbol{x}_i^T) / n$  be the design covariance matrix as before and  $M = (\sum_{i=1}^{n} \mu_i \boldsymbol{u}_i \boldsymbol{u}_i^T) / n$ , where  $\boldsymbol{u}_i = \boldsymbol{x}_i / \| \boldsymbol{x}_i \|_2$ . Let  $\chi$  and  $\zeta$  be the condition numbers of the matrices  $M$  and  $\Sigma$  respectively.

Assumption 1. The parameter space  $\Theta$  and the design matrix  $X$  satisfy the following,

- (A1) The parameter space  $\Theta = \{\pmb {\theta}\in \mathbb{R}^d:\| \pmb {\theta}\| _2\leq w,\min (\| \pmb {\theta}\| _2^2,\langle \pmb {\theta},\pmb {x}_i\rangle)\geq \gamma >0,\forall i\in [n]\}$ .  
- (A2) The design matrix is such that  $\lambda_{\min}(\Sigma) > 0$  and  $\| \pmb{x}_i\| _2\leq R$  for all  $i\in [n]$ .  
- (A3) Let  $\lambda_{\min}(M) \geq \frac{R^2}{4n\gamma^2} (d\log(24\chi) + \log(1/\delta))$  and  $\sqrt{\lambda_{\max}(M)(d\log(24\chi) + \log(1/\delta))} \leq \sqrt{n}\lambda_{\min}(M)/16$ , for a small  $\delta \in (0,1)^2$ .

The above assumptions are fairly mild. For instance  $\lambda_{\mathrm{min}}(M)$  is  $\tilde{\Omega}(1 / d)$  for random covariance matrices (Bai & Yin, 2008) and thus much greater than the lower bound required by the above assumption. The other part of the assumption merely requires that  $\lambda_{\mathrm{min}}(M) = \tilde{\Omega}(\sqrt{d\lambda_{\mathrm{max}}(M) / n})$ .

We are interested in comparing MLE with TMO for the square loss which is just the least-squares estimator i.e  $\hat{\pmb{\theta}}_{\mathrm{ls}}\coloneqq \operatorname *{argmin}_{\pmb {\theta}\in \Theta}\frac{1}{n}\| y^n -\pmb {X}\pmb {\theta}\| _2^2$  Note that it is apriori unclear as to which approach would be better in terms of the target metric because on one hand the MLE method knows the distribution family but on the other hand TMO is explicitly geared towards minimizing the square loss during training.

Least squares analysis is typically provided for regression under sub-Gaussian noise. By adapting existing techniques (Hsu et al., 2012), we show the following guarantee for Poisson regression with least square loss. We provide a proof in Appendix D for completeness.

Lemma 1. Let  $\mu_{max} = \max_i\mu_i$ . The least squares estimator  $\hat{\theta}_{\mathrm{ls}}$  satisfies the following loss bounds w.p. at least  $1 - \delta$ ,

$$
\mathcal {E} (\hat {\boldsymbol {\theta}} _ {\mathrm {l s}}) = \left\{ \begin{array}{l l} O \big (\frac {\mu_ {m a x}}{n} \big (\log \frac {1}{\delta} + d \big) \big) & i f \mu_ {m a x} \geq (\log (1 / \delta) + d \log 6) / 2 \\ O \Big (\frac {1}{n} \big (\log \frac {1}{\delta} + d \big) ^ {2} \Big) & o t h e r w i s e \end{array} \right.
$$

Now we present our main result in this section which uses the competitiveness bound in Theorem 2 coupled with the existence of a superior estimator compared to TMO, to show that the MLE estimator can have a better bound than TMO.

In Theorem 4 (in Appendix F), under some mild assumptions on the covariates, we construct an estimator  $\theta_{\mathrm{est}}$  with the following bound for the Poisson regression setting,

$$
\mathcal {E} \left(\boldsymbol {\theta} _ {\text {e s t}}\right) \leq c \cdot \left\| \boldsymbol {\theta} ^ {*} \right\| ^ {2} \cdot \lambda_ {\max } (\Sigma) \left(\frac {d + \log \left(\frac {1}{\delta}\right)}{n}\right). \tag {3}
$$

The construction of the estimator is median-of-means tournament based along the lines of (Lugosi & Mendelson, 2019a) and therefore the estimator might not be practical. However, this immediately gives the following bound on the MLE as a corollary of Theorem 2.

Corollary 1. Under assumption 1 and the conditions of Theorem 4, the MLE estimator for the Poisson regression setting satisfies w.p at least  $1 - \delta$ ,

$$
\mathcal {E} (\hat {\boldsymbol {\theta}} _ {\mathrm {m l e}}) = O \bigg (\| \boldsymbol {\theta} ^ {*} \| ^ {2} \cdot \lambda_ {\max } (\Sigma) \frac {d (\log n + \log (w R \lambda_ {\max } (\Sigma)) + \log \chi + \log \frac {1}{\delta})}{n} \bigg).
$$

The bound in Corollary 1 can be better than the bound for  $\hat{\theta}_{\mathrm{ls}}$  in Lemma 1. In the sub-Gaussian region, the bound in Lemma 1 scales linearly with  $\mu_{max}$  which can be prohibitively large even when a few covariates have large norms. The bound for the MLE estimator in Corollary 1 has no such dependence. Further, in the sub-Exponential region the bound in Lemma 1 scales as  $\tilde{O}(d^2 / n)$  while the bound in Corollary 1 has a  $\tilde{O}(d / n)$  dependency, up to log-factors. In Appendix G, we also show that when the covariates are one-dimensional, an even sharper analysis is possible, that shows that the MLE estimator is always better than least squares in terms of excess risk.

# 4.2 PARETO REGRESSION

Now we will provide an example of a heavy tailed regression setting where it is well-known that TMO for the square loss does not perform well (Lugosi & Mendelson, 2019a). We will consider the Pareto regression setting given by,

$$
p \left(y _ {i} \mid \boldsymbol {x} _ {i}\right) = \frac {b m _ {i} ^ {b}}{y _ {i} ^ {b + 1}}, \quad m _ {i} = \frac {b - 1}{b} \left\langle \boldsymbol {\theta} ^ {*}, \boldsymbol {x} _ {i} \right\rangle \quad \text {f o r} y _ {i} \geq m _ {i} \tag {4}
$$

provided  $\langle \pmb{\theta}^*,\pmb{x}_i\rangle >\gamma$  for all  $i\in [n]$ . Thus  $y_{i}$  is Pareto given  $\pmb {x}_i$  and  $\mathbb{E}[y_i|\pmb {x}_i] = \mu_i\coloneqq \langle \pmb {\theta}^*,\pmb {x}_i\rangle$ . We will assume that  $b > 4$  such that  $4 + \epsilon$ -moment exists for  $\epsilon >0$ . As in the Poisson setting, we choose this parametrization for a fair comparison between TMO and MLE i.e in the limit of infinite samples  $\pmb{\theta}^{*}$  lies in the linear solution space for both TMO (least squares) and MLE.

As before, to apply Theorem 2 we need an estimator with a good risk bound. We use the estimator in Theorem 4 of (Hsu & Sabato, 2016), which in the fixed design pareto regression setting yields,

$$
\mathcal {E} (\pmb {\theta} _ {\mathrm {e s t}}) = \left(1 + O \left(\frac {d \log \frac {1}{\delta}}{n}\right)\right) \frac {\| \theta^ {*} \| _ {\Sigma} ^ {2}}{b (b - 2)}.
$$

Note that the above estimator might not be easily implementable, however this yields the following corollary of Theorem 2, which is a bound on the performance of the MLE estimator.

Corollary 2. Under assumptions of our Pareto regression setting, the MLE estimator satisfies w.p at least  $1 - \delta$ ,

$$
\mathcal {E} (\hat {\boldsymbol {\theta}} _ {\mathrm {m l e}}) = 1 + O \left(\frac {d ^ {2} \left(\log n + \log \zeta + \log \frac {b w R \lambda_ {\max} (\Sigma)}{\gamma} + \log \frac {1}{\delta}\right)}{n}\right) \frac {\| \theta^ {*} \| _ {\Sigma} ^ {2}}{b (b - 2)}.
$$

The proof is provided in Appendix H. It involves verifying the two conditions in Theorem 2 in the Pareto regression setting.

The above MLE guarantee is expected to be much better than what can be achieved by TMO which is least-squares. It is well established in the literature (Hsu & Sabato, 2016; Lugosi & Mendelson, 2019a) that ERM on square loss cannot achieve a  $O(\log(1/\delta))$  dependency in a heavy tailed regime; instead it can achieve only a  $O(\sqrt{1/\delta})$  rate.

# 5 CHOICE OF LIKELIHOOD AND INFERENCE METHODS

In this section we discuss some practical considerations for MLE, such as adapting to a target metric of interest at inference time, and the choice of the likelihood family.

Inference for different target metrics: In most practical settings, the trained regression model is used at inference time to predict the response variable on some test set to minimize some target metric. For the MLE based approach, once the distribution parameters are learnt, this involves using an appropriate statistic of the learnt distribution at inference time (see Section 3). For mean squared error and mean absolute error, the estimator corresponds to the mean and median of the distribution, but for several other commonly used loss metrics in the forecasting domain such as Mean Absolute Percentage Error (MAPE) and Relative Error (RE) (Gneiting, 2011; Hyndman & Koehler, 2006), this estimator corresponds to a median of a transformed distribution (Gneiting, 2011). Please see Appendix I for more details. This ability to optimize the estimator at inference time for different target metrics using a single trained model is another advantage that MLE based approaches have over TMO models that are trained individually for specific target metrics.

Mixture Likelihood: An important practical question when performing MLE-based regression is to decide which distribution family to use for the response variable. The goal is to pick a distribution family that can capture the inductive biases present in the data. It is well known that a misspecified distribution family for MLE might adversely affect generalization error of regression models (Heagerty & Kurland, 2001). At the same time, it is also desirable for the distribution family to be generic enough to cater to diverse datasets with potentially different types of inductive biases, or even datasets for which no distributional assumptions can be made in advance.

A simple approach that we observe to work particularly well in practice with regression models using deep neural networks is to assume the response variable comes from a mixture distribution, where each mixture component belongs to a different distribution family and the mixture weights are learnt along with the parameters of the distribution. More specifically, we consider a mixture distribution of  $k$  components  $p(y|\boldsymbol{x};\boldsymbol{\theta}_1,\dots,\boldsymbol{\theta}_k,w_1,\dots,w_k) = \sum_{j=1}^k w_jp_j(y|\boldsymbol{x};\boldsymbol{\theta}_j)$ , where each  $p_j$  characterizes a different distribution family, and the mixture weights  $w_j$  and distribution parameters  $\boldsymbol{\theta}_j$  are learnt together. For example, if we have reason to believe the response variable's distribution might be heavy-tailed in some datasets but Negative Binomial in other datasets (say, for count regression), then we can use a single MLE regression model with a mixture distribution of the two, that often performs well in practice for all the datasets.

Motivated by our theory, we use a three component mixture of  $(i)$  the constant 0 (zero-inflation for dealing with bi-modal sparse data),  $(ii)$  negative binomial where  $n$  and  $p$  are learnt and  $(iii)$  a Pareto distribution where the scale parameter is learnt. Our experiments in Section 6 show that this mixture shows promising performance on a variety of datasets.

This approach will increase the number of parameters and the resulting likelihood might require non-convex optimization. However, we empirically observed that with sufficiently over-parameterized networks and standard gradient-based optimizers, this is not a problem in practice at all.

# 6 EMPIRICAL RESULTS

We present empirical results on two time-series forecasting problems and two regression problems using neural networks. We will first describe our models and baselines. Our goal is to compare the MLE approach with the TMO approach for three target metrics per dataset.

<table><tr><td rowspan="2">Model</td><td colspan="3">Favorite</td><td colspan="3">M5</td></tr><tr><td>MAPE</td><td>WAPE</td><td>RMSE</td><td>MAPE</td><td>WAPE</td><td>RMSE</td></tr><tr><td>TMO(MSE)</td><td>0.6121±0.0075</td><td>0.2891±0.0023</td><td>175.3782±0.8235</td><td>0.5045±0.004</td><td>0.2839±0.0008</td><td>7.507±0.023</td></tr><tr><td>TMO(MAE)</td><td>0.3983±0.0012</td><td>0.2258±0.0006</td><td>161.4919±0.4748</td><td>0.4452±0.0005</td><td>0.266±0.0001</td><td>7.0503±0.0094</td></tr><tr><td>TMO(MAPE)</td><td>0.3199±0.0011</td><td>0.2528±0.0016</td><td>192.3823±1.3871</td><td>0.3892±0.0001</td><td>0.3143±0.0007</td><td>11.3799±0.1965</td></tr><tr><td>TMO(Huber)</td><td>0.432±0.0033</td><td>0.2366±0.0018</td><td>164.7006±0.7178</td><td>0.4722±0.0007</td><td>0.269±0.0002</td><td>7.093±0.0133</td></tr><tr><td>MLE (ZNBP)</td><td>0.3139±0.0011</td><td>0.2238±0.0009</td><td>164.6521±1.5185</td><td>0.3864±0.0001</td><td>0.2677±0.0002</td><td>7.2133±0.0152</td></tr></table>

Table 1: We provide the MAPE, WAPE and RMSE metrics for all the models on the test set of two time-series datasets. The confidence intervals provided are one standard error over 50 experiments, for each entry. TMO(<loss>) refers to TMO using the <loss>. For the MLE row, we only train one model per dataset. The same model is used to output a different statistic for each column during inference. For MAPE, we output the optimizer of MAPE given in Section I.4. For WAPE we output the median and for RMSE we output the mean.  
Table 2: We provide the MAPE, WAPE and RMSE metrics for all the models on the test set of two regression datasets. The confidence intervals provided are one standard error over 50 experiments, for each entry. TMO(<loss>) refers to TMO using the <loss>. For the MLE row, we only train one model per dataset. The same model is used to output a different statistic for each column during inference. For MAPE, we output the optimizer of MAPE given in Section I.4. For WAPE we output the median and for RMSE we output the mean.  

<table><tr><td rowspan="2">Model</td><td colspan="3">Bicycle Share</td><td colspan="3">Gas Turbine</td></tr><tr><td>MAPE</td><td>WAPE</td><td>RMSE</td><td>MAPE</td><td>WAPE</td><td>RMSE</td></tr><tr><td>TMO(MSE)</td><td>0.2503±0.0008</td><td>0.1421±0.0003</td><td>878.5815±1.3059</td><td>0.8884±0.0118</td><td>0.3496±0.0041</td><td>1.5628±0.0071</td></tr><tr><td>TMO(MAE)</td><td>0.2594±0.0011</td><td>0.1436±0.0003</td><td>901.1357±1.4943</td><td>0.774±0.0054</td><td>0.3389±0.0019</td><td>1.5789±0.0067</td></tr><tr><td>TMO(MAPE)</td><td>0.2382±0.0012</td><td>0.1469±0.0012</td><td>899.9163±4.8219</td><td>0.8108±0.0009</td><td>0.8189±0.001</td><td>3.0573±0.0019</td></tr><tr><td>TMO(Huber)</td><td>0.2536±0.0011</td><td>0.1414±0.0004</td><td>889.1173±1.9654</td><td>0.902±0.0128</td><td>0.3598±0.0049</td><td>1.5992±0.0082</td></tr><tr><td>MLE (ZNBP)</td><td>0.1969±0.0018</td><td>0.1235±0.001</td><td>767.4368±7.1274</td><td>0.9877±0.0019</td><td>0.3379±0.0004</td><td>1.4547±0.0054</td></tr></table>

Common Experimental Protocol: Now we describe the common experimental protocol on all the datasets (we get into dataset related specifics and architectures subsequently). For a fair comparison the architecture is kept the same for TMO and MLE approaches. For each dataset, we tune the hyper-parameters for the TMO(MSE) objective. Then these hyper-parameters are held fixed for all models for that dataset i.e only the output layer and the loss function is modified. We provide all the details in Appendix I.

For the MLE approach, the output layer of the models map to the MLE parameters of the mixture distribution introduced in Section 5, through link functions. The MLE output has 6 parameters, three for mixture weights, two for negative binomial component and one for the scale parameter in Pareto. The choice of the link functions and more details are specified in Appendix I.2. The loss function used is the negative log-likelihood implemented in Tensorflow Probability (Dillon et al., 2017). Note that for the MLE approach only one model is trained per dataset and during inference we output the statistic that optimizes the target metric in question. We refer to our MLE based models that employs the mixture likelihood from Section 5 as MLE (ZNBP) loss, where ZNBP refers to the mixture components Zero, Negative-Binomial and Pareto.

For TMO, the output layer of the models map to  $\hat{y}$  and we directly minimize the target metric in question. Note that this means we need to train a separate model for every target metric. Thus we have one model each for target metrics in  $\{\text{'MSE}',\text{'MAE'},\text{'MAPE'}\}$ . Further we also train a model using the Huber loss  $^3$ . In order to keep the number of parameters the same as that of MLE, we add an additional 6 neurons to the TMO models.

# 6.1 EXPERIMENTS ON FORECASTING DATASETS

We perform our experiments on two well-known forecasting datasets used in Kaggle competitions.

1. The M5 dataset (M5, 2020) consists of time series data of product sales from 10 Walmart stores in three US states. The data consists of two different hierarchies: the product hierarchy and store location hierarchy. For simplicity, in our experiments we use only the product hierarchy consisting of 3K individual time-series and 1.8K time steps.  
2. The Favorita dataset (Favorita, 2017) is a similar dataset, consisting of time series data from Corporación Favorita, a South-American grocery store chain. As above, we use the product hierarchy, consisting of 4.5k individual time-series and 1.7k time steps.

The task is to predict the values for the last 14 days all at once. The preceding 14 days are used for validation. We provide more details about the dataset generation for reproducibility in Appendix I.

Table 3: The MLE model predicts the empirical quantile of interest during inference. It is compared with Quantile regression (TMO based). The results, averaged over 50 runs along with the corresponding confidence intervals are presented.  

<table><tr><td>Model</td><td>p10QL</td><td>p90QL</td></tr><tr><td>TMO (Quantile)</td><td>0.0973±0.0002</td><td>0.0628±0.0019</td></tr><tr><td>MLE (ZNBP)</td><td>0.0788±0.0008</td><td>0.0536±0.0007</td></tr></table>

Table 4: We perform an ablation study on the Favorita dataset, where we progressively add the components of our mixture distribution. There are three MLE models in the progression: Negative Binomial (NB), Zero-Inflated Negative Binomial (ZNB) and finally ZNBP.  

<table><tr><td>Model</td><td>MAPE</td><td>WAPE</td><td>RMSE</td></tr><tr><td>MLE (NB)</td><td>0.3314+/-0.0016</td><td>0.2521+/-0.002</td><td>175.501+/-1.1928</td></tr><tr><td>MLE (ZNB)</td><td>0.3186+/-0.0011</td><td>0.2453+/-0.002</td><td>170.0075+/-1.282</td></tr><tr><td>MLE (ZNBP)</td><td>0.3139±0.0011</td><td>0.2238±0.0009</td><td>164.6521+/-1.5185</td></tr></table>

The base architecture for the baselines as well as our model is a seq-2-seq model (Sutskever et al., 2014). The encoders and decoders both are LSTM cells (Hochreiter & Schmidhuber, 1997). The architecture is illustrated in Figure 1 and described in more detail in Appendix I.

We present our experimental results in Table 1. On both the datasets the MLE model with the appropriate inference-time estimator for a metric is always better than TMO trained on the same target metric, except for WAPE in M5 where MLE's performance is only marginally worse. Note that the MLE model is always the best or second best performing model on all metrics, among all TMO models. For TMO the best performance is not always achieved for the same target metric. For instance, TMO(MAE) performs better than TMO(MSE) for the RMSE metric on the Favorita dataset. In Table 4 we perform an ablation study on the Favorita dataset, where we progressively add mixture components resulting in three MLE models: Negative Binomial, Zero-Inflated Negative Binomial and finally ZNBP. This shows that each of the components add value in this dataset.

# 6.2 EXPERIMENTS ON REGRESSION DATASETS

We perform our experiments on two standard regression datasets,

1. The Bicycle Share dataset (Bicycle, 2017) has daily counts of the total number of rental bikes. The features include time features as well as weather conditions such as temperature, humidity, windspeed etc. A random  $10\%$  of the dataset is used as test and the rest for training and validation. The dataset has a total of 730 samples.  
2. The Gas Turbine dataset (Kaya et al., 2019) has 11 sensor measurements per example (hourly) from a gas turbine in Turkey. We consider the level of NOx as our target variable and the rest as predictors. There are 36733 samples in total. We use the official train/test split. A randomly chosen  $20\%$  of the training set is used for validation. The response variable is continuous.

For all our models, the model architecture is a fully connected DNN with one hidden layer that has 32 neurons. Note that for categorical variables, the input is first passed through an embedding layer (one embedding layer per feature), that is jointly trained. We provide further details like the shape of the embedding layers in Appendix I. The architecture is illustrated in Figure 2.

We present our results in Table 2. On the Bicycle Share dataset, the MLE (ZNBP) based model performs optimally in all metrics and often outperforms the TMO models by a large margin even though TMO is a separate model per target metric. On the Gas Turbine dataset, the MLE based model is optimal for WAPE and RMSE, however it does not perform that well for the MAPE metric.

In Table 3, we compare the MLE based approach versus quantile regression (TMO based) on the Bicycle Share dataset, where the metric presented is the normalized quantile loss (Wang et al., 2019). We train the TMO model for the corresponding quantile loss directly and the predictions are evaluated on normalized quantile losses as shown in the table. The MLE based model is trained by minimizing the negative log-likelihood and then during inference we output the corresponding empirical quantile from the predicted distribution. MLE (ZNBP) outperforms TMO(Quantile) significantly.

Discussion: We compare the approaches of direct ERM on the target metric (TMO) and MLE followed by post-hoc inference time optimization for regression and forecasting problems. We prove a general competitiveness result for the MLE approach and also show theoretically that it can be better than TMO in the Poisson and Pareto regression settings. Our empirical results show that our proposed general purpose likelihood function employed in the MLE approach can uniformly perform well on several tasks across four datasets. Even though this addresses some of the concerns about choosing the correct likelihood for a dataset, some limitations still remain for example concerns about the non-convexity of the log-likelihood. We provide a more in-depth discussion in Appendix J.

Reproducibility Statement: We provide our code in the supplementary along with instructions to run it. We also provide download instructions for a processed dataset for ease of use.

# REFERENCES

Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: A system for large-scale machine learning. In 12th USENIX symposium on operating systems design and implementation (OSDI 16), pp. 265-283, 2016.  
Jayadev Acharya, Hirakendu Das, Alon Orlitsky, and Ananda Theertha Suresh. A unified maximum likelihood approach for estimating symmetric properties of discrete distributions. In International Conference on Machine Learning, pp. 11-21. PMLR, 2017.  
Jean-Yves Audibert, Olivier Catoni, et al. Robust linear least squares regression. The Annals of Statistics, 39(5):2766-2794, 2011.  
Zhi-Dong Bai and Yong-Qua Yin. Limit of the smallest eigenvalue of a large dimensional sample covariance matrix. In Advances In Statistics, pp. 108-127. World Scientific, 2008.  
Derek Bean, Peter J Bickel, Noureddine El Karoui, and Bin Yu. Optimal m-estimation in high-dimensional regression. Proceedings of the National Academy of Sciences, 110(36):14563-14568, 2013.  
Bicycle. Bicycle share dataset. https://www.kaggle.com/contactprad/bike-share-daily-data/, 2017.  
Christian Brownlees, Emilien Joly, and Gábor Lugosi. Empirical risk minimization for heavy-tailed losses. The Annals of Statistics, 43(6), Dec 2015. ISSN 0090-5364. doi: 10.1214/15-aos1350. URL http://dx.doi.org/10.1214/15-AOS1350.  
Richard A Davis and Rongning Wu. A negative binomial model for time series of counts. Biometrika, 96(3):735-749, 2009.  
Joshua V Dillon, Ian Langmore, Dustin Tran, Eugene Brevdo, Srinivas Vasudevan, Dave Moore, Brian Patton, Alex Alemi, Matt Hoffman, and Rif A Saurous. Tensorflow distributions. arXiv preprint arXiv:1711.10604, 2017.  
David Donoho and Andrea Montanari. High dimensional robust m-estimation: Asymptotic variance via approximate message passing. Probability Theory and Related Fields, 166(3):935-969, 2016.  
Lutz Dumbgen and Kaspar Rufibach. Maximum likelihood estimation of a log-concave density and its distribution function: Basic properties and uniform consistency. Bernoulli, 15(1):40-68, 2009.  
Noureddine El Karoui. On the impact of predictor geometry on the performance on high-dimensional ridge-regularized generalized robust regression estimators. *Probability Theory and Related Fields*, 170(1):95–175, 2018.  
Ludwig Fahrmeir and Heinz Kaufmann. Consistency and asymptotic normality of the maximum likelihood estimator in generalized linear models. The Annals of Statistics, 13(1):342-368, 1985.  
Favorite. Favorite forecasting dataset. https://www.kaggle.com/c/favorite-grocery-sales-forecasting/, 2017.  
Dylan J Foster and Akshay Krishnamurthy. Efficient first-order contextual bandits: Prediction, allocation, and triangular discrimination. arXiv preprint arXiv:2107.02237, 2021.  
Jan Gasthaus, Konstantinos Benidis, Yuyang Wang, Syama Sundar Rangapuram, David Salinas, Valentin Flunkert, and Tim Januschowski. Probabilistic forecasting with spline quantile function rnns. In The 22nd international conference on artificial intelligence and statistics, pp. 1901-1910. PMLR, 2019.  
Sara A Geer and Sara van de Geer. Empirical Processes in M-estimation, volume 6. Cambridge university press, 2000.

Tilmann Gneiting. Making and evaluating point forecasts. Journal of the American Statistical Association, 106(494):746-762, 2011.  
Arthur Stanley Goldberger et al. Econometric theory. Econometric theory., 1964.  
Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, volume 1. MIT Press, 2016.  
Patrick J Heagerty and Brenda F Kurland. Misspecified maximum likelihood estimates and generalised linear mixed models. Biometrika, 88(4):973-985, 2001.  
C.C. Heyde. On an optimal asymptotic property of the maximum likelihood estimator of a parameter from a stochastic process. Stochastic Processes and their Applications, 8(1): 1-9, 1978. ISSN 0304-4149. URL https://www.sciencedirect.com/science/article/pii/0304414978900649.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Daniel Hsu and Sivan Sabato. Loss minimization and parameter estimation with heavy tails. Journal of Machine Learning Research, 17(18):1-40, 2016. URL http://jmlr.org/papers/v17/14-273.html.  
Daniel Hsu, Sham M Kakade, and Tong Zhang. Random design analysis of ridge regression. In Conference on learning theory, pp. 9-1. JMLR Workshop and Conference Proceedings, 2012.  
Peter J Huber. Robust estimation of a location parameter. In *Breakthroughs in statistics*, pp. 492-518. Springer, 1992.  
Rob John Hyndman and Ann B Koehler. Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4):679-688, 2006.  
Heysem Kaya, PINAR TÜFEKCI, and Erdinc Uzun. Predicting co and no x emissions from gas turbines: novel data and a benchmark pems. Turkish Journal of Electrical Engineering & Computer Sciences, 27(6):4783-4796, 2019.  
Bernhard Klar. Bounds on tail probabilities of discrete distributions. Probability in the Engineering and Informational Sciences, 14(2):161-171, 2000.  
Roger Koenker and Gilbert Bassett Jr. Regression quantiles. *Econometrica: journal of the Econometric Society*, pp. 33-50, 1978.  
Stéphane Lathuilière, Pablo Mesejo, Xavier Alameda-Pineda, and Radu Horaud. A comprehensive analysis of deep regression. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42 (9):2065-2081, 2020. doi: 10.1109/TPAMI.2019.2910523.  
Jerald F Lawless. Negative binomial and mixedoisson regression. The Canadian Journal of Statistics/La Revue Canadienne de Statistique, pp. 209-225, 1987.  
G. Lecue and S. Mendelson. Learning subgaussian classes: Upper and minimax bounds. In S. Boucheron and N. Vayatis, editors, Topics in Learning Theory. Societe Mathematique de France, 2016.  
Erich L Lehmann and George Casella. Theory of point estimation. Springer Science & Business Media, 2006.  
Gábor Lugosi and Shahar Mendelson. Mean estimation and regression under heavy-tailed distributions: A survey. Foundations of Computational Mathematics, 19(5):1145-1190, 2019a.  
Gábor Lugosi and Shahar Mendelson. Sub-gaussian estimators of the mean of a random vector. The annals of statistics, 47(2):783-794, 2019b.  
M5. M5 forecasting dataset. https://www.kaggle.com/c/m5-forecasting-accuracy/, 2020.

Peter McCullagh and John A Nelder. Generalized linear models. Routledge, 2019.  
Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of machine learning. MIT press, 2018.  
John Ashworth Nelder and Robert WM Wedderburn. Generalized linear models. Journal of the Royal Statistical Society: Series A (General), 135(3):370-384, 1972.  
Ryan O'Donnell. Analysis of boolean functions. Cambridge University Press, 2014.  
Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-beats: Neural basis expansion analysis for interpretable time series forecasting. arXiv preprint arXiv:1905.10437, 2019.  
Adarsh Prasad, Arun Sai Suggala, Sivaraman Balakrishnan, and Pradeep Ravikumar. Robust estimation via robust gradient estimation. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 82(3):601-627, 2020.  
C.R. Rao. Criteria of estimation in large samples. Sankhya, 25, Ser A, 1963.  
Richard Redner. Note on the consistency of the maximum likelihood estimate for nonidentifiable distributions. The Annals of Statistics, pp. 225-228, 1981.  
Philippe Rigollet. High-dimensional statistics. https://ocw.mit.edu/courses/mathematics/18-s997-high-dimensional-statistics-spring-2015/lecture-notes/MIT18_S997S15_Chapter2.pdf, 2015.  
Alessandro Rinaldo. Sub-exponential concentration. http://www.stat.cmu.edu/~arinaldo/Teaching/36709/S19/Scribed_Lectures/Feb5_Aleksandr.pdf, 2019.  
David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deeper: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3): 1181-1191, 2020.  
Rajat Sen, Hsiang-Fu Yu, and Inderjit Dhillon. Think globally, act locally: A deep neural network approach to high-dimensional time series forecasting. arXiv preprint arXiv:1905.03806, 2019.  
Niranjan Srinivas, Andreas Krause, Sham M Kakade, and Matthias Seeger. Gaussian process optimization in the bandit setting: No regret and experimental design. arXiv preprint arXiv:0912.3995, 2009.  
Pragya Sur and Emmanuel J Candès. A modern maximum-likelihood theory for high-dimensional logistic regression. Proceedings of the National Academy of Sciences, 116(29):14516-14525, 2019.  
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. arXiv preprint arXiv:1409.3215, 2014.  
Abraham Wald. Note on the consistency of the maximum likelihood estimate. The Annals of Mathematical Statistics, 20(4):595-601, 1949.  
Yuyang Wang, Alex Smola, Danielle Maddix, Jan Gasthaus, Dean Foster, and Tim Januschowski. Deep factors for forecasting. In International Conference on Machine Learning, pp. 6607-6617. PMLR, 2019.  
Ruofeng Wen Wen, Kari Torkkola, and Balakrishnan Narayanaswamy. A multi-horizon quantile recurrent forecaster. In NIPS Time Series Workshop, 2017.  
Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, and Chengqi Zhang. Connecting the dots: Multivariate time series forecasting with graph neural networks. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 753-763, 2020.

Tong Zhang. From epsilon-entropy to kl-entropy: Analysis of minimum information complexity density estimation. The Annals of Statistics, 34(5):2180-2210, 2006.  
Ziwei Zhu and Wenjing Zhou. Taming heavy-tailed features by shrinkage. In International Conference on Artificial Intelligence and Statistics, pp. 3268-3276. PMLR, 2021.
