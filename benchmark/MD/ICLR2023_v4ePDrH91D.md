# ROBUST MANIFOLD ESTIMATION APPROACH FOR EVALUATING FIDELITY AND DIVERSITY

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose a robust and reliable evaluation metric for generative models by introducing topological and statistical treatments for a rigorous support manifold estimation. Existing metrics, such as Inception Score (IS), Fréchet Inception Distance (FID), and the variants of Precision and Recall (P&R), heavily rely on support manifolds that are estimated from sample features. However, the reliability of their estimation has not been seriously discussed (and overlooked) even though the quality of the evaluation entirely depends on it. In this paper, we propose Topological Precision and Recall (TopP&R, pronounced "topper"), which provides a systematic approach to estimating support manifolds, retaining only topologically and statistically important features with a certain level of confidence. This not only makes TopP&R strong for noisy features, but also provides statistical consistency. Our theoretical and experimental results show that TopP&R is robust to outliers and non-independent and identically distributed (Non-IID) perturbations, while accurately capturing the true trend of change in samples. To the best of our knowledge, this is the first evaluation metric focused on the robust estimation of the support manifold and provides its statistical consistency under noise.

# 1 INTRODUCTION

In keeping with the remarkable improvements of deep generative models (Karras et al., 2019; 2020; 2021; Brock et al., 2018; Ho et al., 2020; Kingma & Welling, 2013; Sauer et al., 2022; 2021; Kang & Park, 2020), evaluation metrics that can well measure the performance of generative models have also been continuously developed (Salimans et al., 2016; Heusel et al., 2017; Sajjadi et al., 2018; Kynkänniemi et al., 2019; Naeem et al., 2020). For instance, Inception Score (IS) (Salimans et al., 2016) measures the Kullback-Leibler divergence between the real and fake sample distributions in VGG (Simonyan & Zisserman, 2014) feature space. Fréchet Inception Score (FID)(Heusel et al., 2017) calculates the distance between the real and fake support manifolds using the estimated mean and variance under the multi-Gaussian assumption. The original Precision and Recall  $(\mathbb{P} \& \mathbb{R})$  (Sajjadi et al., 2018) and its variants (Kynkänniemi et al., 2019; Naeem et al., 2020) were the first to separately investigate the fidelity and diversity aspects of generative performance.

Considering the eminent progress of deep generative models based on these existing metrics, some may question why we need another evaluation study. In this paper, we argue that we need more reliable evaluation metrics now precisely, because deep generative models have reached sufficient maturity. To provide a more accurate and comprehensive ideas and to illuminate a new direction of improvements in the generative field, we need a more robust and reliable evaluation metric. In fact, it has been recently reported that even the most widely used evaluation metric, FID, sometimes doesn't match with the expected perceptual quality, fidelity, and diversity, which means the metrics are not always working properly (Kynkänniemi et al., 2022). In addition to this, in practice, not only do generated samples but also real data in the wild often contain lots of artifacts, and these have been shown to seriously perturb the existing evaluation metrics, giving a false sense of improvements (Naeem et al., 2020; Kynkänniemi et al., 2022).

An ideal evaluation metric must capture the real signal of the data, while being robust to noise. Note that there is an inherent tension in developing metrics that meets these goals. On one hand, the metric should be sensitive enough so that it can capture real signals lurking in data. On the other hand, it must ignore noises that hide the signal. However, sensitive metrics are inevitably susceptible to noise

![](images/eee4a9b90bf85044f7d4624bdc48db9589eed4696a55eb797dd00d7bc34a30ea.jpg)  
Figure 1: To robustly estimate the support, we use the bootstrap bandwidth  $c_{\alpha}$  to filter out topological noise (orange) and keep topological signal (skyblue). Then TopP&R is computed on this support.

![](images/7255fab85a8089d727d8cf39f3f389722ec2d38ff4a9c67437bd3e483983f620.jpg)

to some extent. To address this, one needs a systematic way to answer the following two questions: 1) what is signal and what is noise? and 2) how do we draw a line between them?

One solution can be to use the idea of statistical inference and topological data analysis (TDA). Topological data analysis (TDA) (Carlsson, 2009) is a recent and emerging field of data science that relies on topological tools to infer relevant features for possibly complex data. A key object in TDA is persistent homology, which quantifies salient topological features of data by observing them in multi-resolutions. It observes how long each homological feature would survive across different resolutions through the lens of topology and provides a way to quantify its importance; That is, if some features persist longer than others, we consider the features and the data that compose them as a topological signal and vice versa as noise. Then, statistical inference provides a systematic way to establish statistical thresholds that separate the topological signal from topological noise with statistical interpretation.

In this paper, we propose to combine these ideas to form a more robust and compact feature manifold and overcome various issues from the conventional metrics. Our main contributions are as follows: we introduce (1) an approach to directly estimate a support manifold via Kernel Density Estimator (KDE) derived under topological conditions; (2) a new metric that is robust to outliers while reliably detecting the change of distributions on various scenarios; and (3) a theoretical guarantee of consistency with robustness under very weak assumptions that is suitable for high dimensional data.

# 2 BACKGROUND

To lay the foundation for our theoretical analysis, we introduce the main idea of persistent homology and its confidence estimation techniques that bring the benefit of using topological and statistical tools for addressing uncertainty in samples. In later sections, we use these tools to analyze the effects of outliers in evaluating generative models and provide more rigorous way of scoring the samples based on the confidence level we set. For space reasons, we only provide a brief overview of the concepts that are relevant to this work and refer the reader to Section A or (Edelsbrunner & Harer, 2010; Chazal & Michel, 2021; Wasserman, 2018; Hatcher, 2002) for further details.

# 2.1 NOTATION

For any  $x$  and  $r > 0$ , we use the notation  $\mathcal{B}_d(x,r) = \{y:d(y,x) < r\}$  be the open ball in distance  $d$  of radius  $r$ . We also write  $\mathcal{B}(x,r)$  when  $d$  is understood from context. For a distribution  $P$  on  $\mathbb{R}^d$ , we let  $\mathrm{supp}(P) \coloneqq \{x \in \mathbb{R}^d : P(\mathcal{B}(x,r)) > 0 \text{ for all } r > 0\}$  be the support of  $P$ . Throughout the paper, we refer to  $\mathrm{supp}(\mathrm{P})$  as support manifold of  $P$ , or simply support, or manifold, but we don't necessarily require the (geometrical) manifold structure on  $\mathrm{supp}(\mathrm{P})$ . For a kernel function  $K: \mathbb{R}^d \to \mathbb{R}$ , a dataset  $\mathcal{X} = \{X_1, \ldots, X_n\} \subset \mathbb{R}^d$  and bandwidth  $h > 0$ , we let the kernel density estimator (KDE) as  $\hat{p}_h(x) \coloneqq \frac{1}{nh^d} \sum_{i=1}^{n} K\left(\frac{x - X_i}{h}\right)$ , and we let the average KDE as  $p_h \coloneqq \mathbb{E}[\hat{p}_h]$ . We denote by  $P, Q$  the probability distributions in  $\mathbb{R}^d$  of real data and generated samples, respectively. And we use  $\mathcal{X} = \{X_1, \ldots, X_n\} \subset \mathbb{R}^d$  and  $\mathcal{Y} = \{Y_1, \ldots, Y_m\} \subset \mathbb{R}^d$  for real data and generated samples possibly with noise, respectively.

# 2.2 CONFIDENCE BAND ESTIMATION

Statistical inference has recently been developed for topological data analysis (Chazal et al., 2013; Fasy et al., 2014). Topological data analysis consists of features reflecting topological characteristics of data, and it is of question to distinguish features that are indeed from geometrical structures and features that are insignificant or due to noise. To statistically separate topologically significant features from topological noise, we use a confidence band. Given the significance level  $\alpha$ , let confidence band  $c_{\mathcal{X}}$  be the bootstrap bandwidth of  $\| \hat{p}_h - p_h^*\|_\infty$ . Then it satisfies  $\lim \inf_{n\to \infty}\mathbb{P}\left(\| \hat{p}_h - p_h\|_\infty < c_{\mathcal{X}}\right)\geq 1 - \alpha$ , as in Proposition 4 in Section C. This confidence band can be used to determine simultaneously significant topological features while filtering out noise features. The algorithm for computing  $c_{\mathcal{X}}$  is described below.

# Algorithm 1 Confidence Band Estimator

1: #KDE: kernel density estimator  
2: # R.S.: random sample with replacement  
3: #  $k$ : number of repeats  
4: #  $\hat{\theta}$ : set of difference  
5: Given  $\mathcal{X} = \{X_1, X_2, \ldots, X_n\}$  
6:  $\hat{p} = KDE(\mathcal{X})$  
7: for iteration  $= 1,2,\dots ,k$  do  
8: # compute  $\tilde{\theta}$  with bootstrap samples  
9:  $\mathcal{X}^{*} = \mathrm{R.S.}n$  times from  $\mathcal{X}$  
10: #  $\hat{p}^*$  replaces population density  
11:  $\hat{p}^{*} = KDE(\mathcal{X}^{*})$  
12: Append  $\hat{\theta}$  with  $\sqrt{n} ||\hat{p} -\hat{p}^{*}||_{\infty}$  
13: end for  
14: #grid search for the confidence band

15: for  $q \in [\min(\hat{\theta}), \max(\hat{\theta})]$  do  
16: count = 0  
17: for  $\forall$  element  $\in \theta$  do  
18: #count significant difference  
19: if element  $>q$  then  
20: count = count + 1  
21: end if  
22: end for  
23: #define the band threshold  
24: if count/k  $\approx \alpha$  then  
25:  $q_{\alpha} = q$  
26: end if  
27: end for  
28: # define estimated confidence band  
29:  $c_{\alpha} = q_{\alpha} / \sqrt{n}$

# 3 ROBUST SUPPORT MANIFOLD ESTIMATION FOR RELIABLE EVALUATION

Current evaluation metrics for generative models typically rely on strong regularity conditions. For example, they assume samples are well-curated without outliers or adversarial perturbation, real or generative models have bounded densities, etc. However, practical scenarios are wild: both real and generated samples can be corrupted with noise from various sources, and the real data can be very sparsely distributed without density. In this work, we consider more general and practical situations, wherein both real and generated samples can have noises that come from sampling procedure, remained uncertainty due to data or model, etc.

# 3.1 TOPOLOGICAL PRECISION AND RECALL

In the ideal case where we have full access to the probability distributions  $P$  and  $Q$ , we define the precision and the recall of distributions as

$$
\operatorname {p r e c i s i o n} _ {P} (Q) := Q \left(\operatorname {s u p p} (P)\right), \quad \operatorname {r e c a l l} _ {Q} (P) := P \left(\operatorname {s u p p} (Q)\right).
$$

These correspond to the max precision and the max recall in Sajjadi et al. (2018). We tweak the precision as  $\mathrm{precision}_P(\mathcal{Y}) = Q\left(\mathrm{supp}(P)\cap \mathrm{supp}(Q)\right) / Q\left(\mathrm{supp}(Q)\right)$ , and define the precision of data points as

$$
\operatorname {p r e c i s i o n} _ {P} (\mathcal {Y}) := \frac {\sum_ {j = 1} ^ {m} 1 \left(Y _ {j} \in \operatorname {s u p p} (P) \cap \operatorname {s u p p} (Q)\right)}{\sum_ {j = 1} ^ {m} 1 \left(Y _ {j} \in \operatorname {s u p p} (Q)\right)},
$$

which is just replacing the distribution  $Q$  by the empirical distribution  $\frac{1}{m}\sum_{j = 1}^{m}\delta_{Y_j}$  of  $Y$  in the precision. We similarly define the recall of data points as

$$
\operatorname {r e c a l l} _ {Q} (\mathcal {X}) := \frac {\sum_ {i = 1} ^ {n} 1 \left(X _ {i} \in \operatorname {s u p p} (Q) \cap \operatorname {s u p p} (P)\right)}{\sum_ {i = 1} ^ {n} 1 \left(X _ {i} \in \operatorname {s u p p} (P)\right)},
$$

However, in practice,  $\mathrm{supp}(P)$  and  $\mathrm{supp}(Q)$  are not known a priori and need to be estimated, and since we allow noise, these estimates should be robust to noise. For this, we use the kernel density estimator (KDE) and the bootstrap bandwidth to robustly estimate the support. Given  $h_n > 0$  and a significance level  $\alpha \in (0,1)$ , we use the KDE  $\hat{p}_{h_n}(x) \coloneqq \frac{1}{nh_n^d} \sum_{i=1}^{n} K\left(\frac{x - X_i}{h_n}\right)$  of  $\mathcal{X}$ , and we use the bootstrap bandwidth  $c_{\mathcal{X}}$  of  $\left\| \hat{p}_{h_n} - \hat{p}_{h_n}^* \right\|_{\infty}$  from Section 2. Then we estimate the support of  $P$  by the superlevel set at  $c_{\mathcal{X}}$  as  $\mathrm{supp}(P) = \hat{p}_{h_n}^{-1}[c_{\mathcal{X}},\infty)$ . Similarly, we let  $\hat{q}_{h_m}(x) \coloneqq \frac{1}{mh_m^d} \sum_{j=1}^{m} K\left(\frac{x - Y_j}{h_m}\right)$  be the KDE of  $\mathcal{Y}$  and let  $c_{\mathcal{Y}}$  be the bootstrap bandwidth of  $\left\| \hat{q}_{h_m} - \hat{q}_{h_m}^* \right\|_{\infty}$ , and then we use  $\mathrm{supp}(Q) = \hat{q}_{h_m}^{-1}[c_{\mathcal{Y}},\infty)$ . Using the superlevel set at  $c_{\mathcal{X}}$  allows to filter out noise whose KDE values are likely to be small.

For the robust estimates of the precision, we apply the support estimates to the precision of data points, and define the topological precision (TopP) as

$$
\operatorname {T o p P} _ {\mathcal {X}} (\mathcal {Y}) := \frac {\sum_ {j = 1} ^ {m} 1 (Y _ {j} \in \operatorname {s u p p} (P) \cap \operatorname {s u p p} (Q))}{\sum_ {j = 1} ^ {m} 1 (Y _ {j} \in \operatorname {s u p p} (Q))} = \frac {\sum_ {j = 1} ^ {m} 1 (\hat {p} _ {h _ {n}} (Y _ {j}) > c _ {\mathcal {X}} , \hat {q} _ {h _ {m}} (Y _ {j}) > c _ {\mathcal {Y}})}{\sum_ {j = 1} ^ {m} 1 (\hat {q} _ {h _ {m}} (Y _ {j}) > c _ {\mathcal {Y}})}.
$$

And we similarly define the topological recall (TopR) as

$$
\operatorname {T o p R} _ {\mathcal {Y}} (\mathcal {X}) := \frac {\sum_ {i = 1} ^ {n} 1 \left(\hat {q} _ {h _ {m}} (X _ {i}) > c _ {\mathcal {Y}} , \hat {p} _ {h _ {n}} (X _ {i}) > c _ {\mathcal {X}}\right)}{\sum_ {i = 1} ^ {n} 1 \left(\hat {p} _ {h _ {n}} (X _ {i}) > c _ {\mathcal {X}}\right)}.
$$

# 3.2 BANDWIDTH ESTIMATION USING BOOTSTRAPPING

Using the bootstrap bandwidth  $c_{\mathcal{X}}$  as threshold is the key part of our estimators TopP&R for robustly estimating  $\mathrm{supp}(P)$ . As we have seen in Section 2, the bootstrap bandwidth  $c_{\mathcal{X}}$  acts as a threshold for filtering out the topological noise in topological data analysis. Analogously, using  $c_{\mathcal{X}}$  as a threshold allows to robustly estimating  $\mathrm{supp}(P)$ . In particular, when  $X_{i}$  is an outlier, its KDE value  $\hat{p}_h(X_i)$  is likely to be small, and the KDE values at the connected component generated by  $X_{i}$  is likely to be small as well. So those components from outliers are likely to be removed in the estimated support  $\hat{p}_h^{-1}[c_{\mathcal{X}},\infty)$ . Hence, the estimated support denoises topological noise from outliers and robustly estimates  $\mathrm{supp}(P)$ . See Section B for more detailed explanation.

Now that we are only left with topological features of high confidence, this allows us to draw analogies to confidence intervals in statistical analysis, where the uncertainty of the samples is treated by setting the level of confidence. In the next section, we show that TopP&R not only gives a more reliable evaluation score for generated samples but also has a good theoretical properties.

# 4 CONSISTENCY WITH ROBUSTNESS OF TOPP&R

The key properties of TopP&R is consistency with robustness. The consistency ensures that, the precision and the recall we compute from the data approaches the precision and the recall from the distribution as we have more samples. The consistency allows to investigate the precision and the recall of the full distributions only with access to finite sampled data. TopP&R achieves consistency with robustness, that is, the consistency holds with the data possibly corrupted by noise. This is due to the robust estimation of the support with the kernel density estimator with confidence bands. This section is devoted to the theoretical analysis of consistency of TopP&R with robustness.

We demonstrate the statistical model for the data and the noise. Let  $P, Q, \mathcal{X}, \mathcal{Y}$  be as in Notation in Section 2, and let  $\mathcal{X}^0, \mathcal{Y}^0$  be real data and generated data without noise.  $\mathcal{X}, \mathcal{Y}, \mathcal{X}^0, \mathcal{Y}^0$  are understood as multisets, i.e., elements can be repeated. We first assume that the uncorrupted data are IID.

Assumption 1. The data  $\mathcal{X}^0 = \{X_1^0,\dots ,X_n^0\}$  and  $\mathcal{Y}^0 = \{Y_1^0,\dots ,Y_m^0\}$  are IID from  $P$  and  $Q$ , respectively.

In practice, the data is often corrupted with noise. We consider the adversarial noise, where some fraction of data are replaced with arbitrary point cloud data.

Assumption 2. Let  $\{\rho_k\}_{k\in \mathbb{N}}$  be a sequence of nonnegative real numbers. Then the observed data  $\mathcal{X}$  and  $\mathcal{Y}$  satisfies  $|\mathcal{X}\backslash \mathcal{X}^0 | = n\rho_n$  and  $|\mathcal{Y}\backslash \mathcal{Y}^0 | = m\rho_m$ .

In the adversarial model, we control the level of noise by the fraction  $\rho$ , but do not assume other conditions such as IID or boundedness, to make our noise model very general and challenging.

For distributions, we assume that the order of probability volume decay  $P(\mathcal{B}(x,r))$  is at least  $r^d$ .

Assumption 3. For all  $x\in \operatorname {supp}(P)$  and  $y\in \operatorname {supp}(Q)$

$$
\liminf _ {r \to 0} \frac {P (\mathcal {B} (x , r))}{r ^ {d}} > 0, \quad \liminf _ {r \to 0} \frac {Q (\mathcal {B} (y , r))}{r ^ {d}} > 0.
$$

Remark 1. Assumption 3 is analogous to Assumption 2 of (Kim et al., 2019), but is weaker since the condition is pointwise on each  $x \in \mathbb{R}^d$ . And this condition is much weaker than assuming a density on  $\mathbb{R}^d$ : for example, a distribution supported on a low-dimensional manifold satisfies Assumption 3. This provides a framework suitable for high dimensional data, since many times high dimensional data lies on a low dimensional structure hence its density on  $\mathbb{R}^d$  cannot exist. See (Kim et al., 2019) for more detailed discussion.

For kernel functions, we also assume weak condition, detailed in Assumption A1 in Section C. Under the data and the noise models, TopP&R achieves consistency with robustness. That is, the estimated precision and recall is asymptotically correct with high probability even if up to a portion of  $1 / \sqrt{n}$  or  $1 / \sqrt{m}$  are replaced by adversarial noise. This is due to the robust estimation of the support with the kernel density estimator with the confidence band of the persistent homology.

Proposition 2. Suppose Assumption 1,2,3,A1 hold. Suppose  $h_n \to 0$ ,  $nh_n \to \infty$ ,  $nh_n^{-d}\rho_n^2 \to 0$ , and similar relations hold for  $h_m$ ,  $\rho_m$ . Then

$$
\left| \mathrm {T o p P} _ {\mathcal {X}} (\mathcal {Y}) - \mathrm {p r e c i s i o n} _ {P} (\mathcal {Y}) \right| \to 0, \qquad \left| \mathrm {T o p R} _ {\mathcal {Y}} (\mathcal {X}) - \mathrm {r e c a l l} _ {Q} (\mathcal {X}) \right| \to 0, \qquad \text {i n p r o b a b i l i t y}.
$$

Theorem 3. Under the same condition as in Proposition 2,

$$
\left| \mathrm {T o p P} _ {\mathcal {X}} (\mathcal {Y}) - \mathrm {p r e c i s i o n} _ {P} (Q) \right| \to 0, \qquad \left| \mathrm {T o p R} _ {\mathcal {Y}} (\mathcal {X}) - \mathrm {r e c a l l} _ {Q} (P) \right| \to 0, \qquad \text {i n p r o b a b i l i t y}.
$$

Our theoretical results in Proposition 2 and Theorem 3 are novel and important in several perspectives. These results are among the first theoretical guarantees for evaluation metrics for generative models as far as we are aware of. Also, as in Remark 1, assumptions are very weak and suitable for high dimensional data. Also, robustness to adversarial noise is provably guaranteed.

# 5 EXPERIMENTS

A good evaluation metric must correctly capture the changes of the underlying data distribution. To examine the performance of evaluation metrics, we carefully select a set of experiments for sanity checks. With toy and real image data, we check 1) how well the metric captures the true trend of underlying data distributions and 2) how well the metric resist perturbations applied to samples.

# 5.1 SANITY CHECKS WITH TOY DATA

Following Naeem et al. (2020), we first examine how well the metric reflects the trend of  $\mathcal{V}$  moving away from  $\mathcal{X}$  and whether it is suitable for finding mode-drop phenomena. In addition to these, we newly design several experiments that can highlight TopP&R's favorable theoretical properties, such as consistency with robustness, in various scenarios.

# 5.1.1 SHIFTING THE GENERATED FEATURE MANIFOLD

For this experiment, we generate samples for  $\mathcal{X} \sim \mathcal{N}(0, I)$  and  $\mathcal{Y} \sim \mathcal{N}(\mu \mathbf{1}, I)$  in  $\mathbb{R}^{32}$  where  $\mathbf{1}$  is a vector of ones and  $I$  is an identity matrix. We then examine how each metric responds to shifting  $\mathcal{Y}$  with  $\mu \in [-1, 1]$  while there are outliers at  $\mathbf{3} \in \mathbb{R}^{32}$  for both  $\mathcal{X}$  and  $\mathcal{Y}$  (Figure 2). Here, we find that both improved P&R and D&C behave pathologically when there are outliers. Since these methods are based on the k-nearest neighbor algorithm and ignore the fact that there can be outliers in both real and fake data, they inevitably overestimate the underlying support when there are outliers. For example, when  $\mathcal{X}$  lies between  $\mathcal{Y}$  and the outlier at  $y = \mathbf{3}$ , improved Recall returns a high-diversity score, even though the true supports of  $\mathcal{X}$  and  $\mathcal{Y}$  are actually far apart. In addition, P&R does not reach 1 in high dimensions even when  $\mathcal{X} = \mathcal{Y}$ . Naeem et al. (2020) circumvented these problems

![](images/1e74389bb15d015e121dc79f1513d37fcbc3f3a3110cab37a0f6eecbdddbd242.jpg)  
Figure 2: Behaviors of evaluation metrics for outliers on real and fake distribution. The horizontal axis corresponds to the value of  $\mu$ .

![](images/7689dd5f65418be1b81c1882d3b409513da0980f7ad8c97057b99b4ad2a1ed50.jpg)  
(a) Sequential mode dropping  
Figure 3: Behaviors of evaluation metrics for (a) sequential and (b) simultaneous mode dropping scenarios. The horizontal axis shows the concentration ratio on the distribution centered at  $\mu = 0$ .  
(b) Simultaneous mode dropping

by proposing D&C that always use  $\mathcal{X}$  (the real data distribution) as a reference point, which in most cases is assumed to have fewer outliers than  $\mathcal{V}$  (the fake data distribution). However, there is no guarantee that this will be the case in practice. When there is an outlier in  $\mathcal{X}$ , D&C also returns an incorrect high-fidelity score at  $\mu > 0.5$ . On the other hand, TopP&R shows a stable trend unaffected by outliers, demonstrating the robustness of our method.

# 5.1.2 SEQUENTIALLY AND SIMULTANEOUSLY DROPPING MODES

For this experiment, we consider the mixture of Gaussians with seven modes in  $\mathbb{R}^{32}$ . We simulate mode-drop phenomena by gradually dropping all but one mode from the fake distribution  $\mathcal{V}$  that is initially identical to  $\mathcal{X}$  (Figure 3). As in the illustration of mode-drop experiment, when the number of samples in a particular mode decreases, we kept the number of samples in  $\mathcal{X}$  constant so that the same amount of decreased samples are supplemented to the first mode which leads fidelity to be fixed to 1. From the result, we observe that the values of Precision fail to saturate, i.e., mainly smaller than 1, and the Density fluctuates to a value greater than 1 indicating their instability and unboundedness. In terms of diversity, Recall does not respond to the simultaneous mode drop, nor does the improved metric Coverage show a fast decay as the reference line. Compared to these methods, TopP performs well, being held at the upperbound of 1 in sequential mode dropping, and TopR also decreases closest to the reference line in simultaneous mode drops.

# 5.1.3 TOLERANCE TO NON-IID PERTURBATIONS

Robustness to perturbations is another important aspect we should consider when designing a metric. Here, we test whether  $\mathsf{TopP\&R}$  behaves stably under two variants of noise cases; 1) scatter noise: replacing  $X_{i}$  and  $Y_{j}$  with uniformly distributed noise and 2) swap noise: swapping the position between  $X_{i}$  and  $Y_{j}$ . These two cases all correspond to the adversarial noise model of Assumption 2. We set  $\mathcal{X} \sim \mathcal{N}(\mu = 0, I)$  and  $\mathcal{Y} \sim \mathcal{N}(\mu = 1, I)$  where  $\mu = 1$ , and thus an ideal evaluation metric must return zero for both fidelity and diversity scores. In both cases, we find that  $\mathsf{P\&R}$  and D&C are more sensitive while  $\mathsf{TopP\&R}$  remains relatively stable until the noise ratio reaches  $30\%$  of the total data, which is a clear example of the weakness of existing metrics to perturbation.

![](images/83975101cc2903d63756921f586594e9924e5a00374dfbf3ada1afff238ada6c.jpg)  
Figure 4: Behaviors of evaluation metrics on Non-IID perturbations. We replace a certain percentage of real and fake data (a) with random uniform noise and (b) by switching.

![](images/f0f7d381e182c72bbc8e1961ac54e1b19ee461914a14f13ec9086a0eb623724f.jpg)  
0.0

![](images/6eec475b83c7018af68687ea3a512095f714fd0b0b31be6ae05029c509bc9328.jpg)  
0.5

![](images/14b5d4f84c336397d6fecd5800fa12bb110a93d78c4121baabdf880d2d2b8b8e.jpg)  
1.0

![](images/eac5c9b7b4098d27546c06ceb63058fe012e9df3f0c32089a397bd4871941547.jpg)  
Figure 5: Behaviour of metrics with truncation trick. The horizontal axis corresponds to the value of  $\psi$  denoting the increased diversity. The images are generated via StyleGAN2 with FFHQ dataset.

# 5.2 SANITY CHECK WITH REAL DATA

Now that we have verified the metrics on toy data using Gaussians, we test them on real data. Just like in the toy experiments, we concentrate on how the metrics behave in extreme situations, such as outliers, mode-drop phenomena, perceptual distortions, and etc. For evaluation, ImageNet pretrained VGG16 with linear random projection to 32 dimension is used as an image embedding. For more experimental details, please refer to the appendix.

# 5.2.1 RESOLVING FIDELITY AND DIVERSITY

To test whether TopP&R responds appropriately to the change in the underlying distributions in real scenarios, we test the metric on the generated images of stylegan2 (Karras et al., 2020) using the truncation trick (Karras et al., 2019). As shown in Figure 5, every time the distribution is transformed by  $\psi$ , TopP&R consistently responds with bounded scores in [0, 1]. On the other hand, Density gives unbounded scores (fidelity  $>1$ ). Because Density is not capped in value, it is difficult to interpret and know exactly which value denotes the best performance (e.g., in our case, the best performance is when fidelity and diversity  $= 1$ ).

# 5.2.2 SEQUENTIALLY AND SIMULTANEOUSLY DROPPING MODES IN CIFAR-10

We conduct an additional simultaneous mode drop experiment to verify TopP&R's actual sensitiveness on the real data set (CIFAR-10). The performance of each metric (Figure 6) is measured with the identical data while simultaneously dropping the modes of nine classes of CIFAR-10. Since the number of the images dropped in each step is identical, the trend of ground truth diversity should linearly decrease. Here, P&R metric captures the simultaneous mode dropping better than D&C because this time random drop of the modes has reduced the area of the estimated fake manifold. On the other hand, TopP&R best captures the true trend of decreasing diversity on average, consistent with the toy result in Figure 3.

# 5.2.3 ROBUSTNESS TO PERTURBATIONS BY OUTLYING FEATURES

To demonstrate the robustness of our metric against the adversarial noise model of Assumption 2, we test both scatter-noise and swap noise scenarios with real data. In the experiment, following Kynkänniemi et al. (2019), we first classify inliers and outliers that are generated by StyleGAN (Karras et al., 2019). For scatter noise we add the outliers to the inliers and for swap noise we swap the real FFHQ images with generated images. Under these specific noise conditions, Precision

![](images/369ff24d3657f43a098b510e6fa69b033ac5bdc46e1a13f3a785fc567610a1ec.jpg)  
(a) Sequential mode dropping

![](images/31aa5d97919e46b84c7e07c9cdd92ec3007ee9c50036432f95fe0a3bd8b83711.jpg)  
(b) Simultaneous mode dropping

![](images/2777460f2cb16892da8475a77040343784d1aeafe280c8d7090a0f52e20509ad.jpg)  
Inlier

![](images/0cebfff483f438a95855e0fe7aa5a0cad70e991e96f96d84aef4c41a171789a5.jpg)  
Outlier

![](images/907f916fbb013f171fddaf3e120c361c600a09ea422eaad3693ed0f33dc617e5.jpg)  
Figure 6: Comparison of evaluation metrics under sequential and simultaneous mode dropping scenario with CIFAR-10.  
(a) Scatter noise

![](images/56f8fbceaada5cdf947bc6af2c2e4f7e073d365cca954bf1e3558772cdee9bf9.jpg)  
(b) Swap noise

![](images/f068989a156816a64d85b3674dd4ef9414b402ca1d8bc8687723eeb97a0751b6.jpg)  
Figure 7: Comparison of evaluation metrics on Non-IID perturbations using FFHQ dataset. We replaced certain ratio of  $\mathcal{X}$  and  $\mathcal{Y}$  (a) with outliers and (b) by exchanging features.  
Figure 8: Verification of whether TopP&R can make an accurate quantitative assessment of noisy image features. Gaussian Noise, gaussian blur, and black rectangle noise are added on the FFHQ imageset and embedded with T4096.

shows similar or even better robustness than Density (Figure 7). On the other hand, Coverage is more robust than Recall. In both cases, TopP&R shows the best performance, resistant to noise.

# 5.2.4 SENSITIVENESS TO THE NOISE INTENSITY

One of the advantages of FID (Heusel et al., 2017) is that it is good at estimating the degrees of distortion applied to the images. Similarly, we check whether the F1-score based on TopP&R provides a reasonable evaluation according to different noise levels. As illustrated in Figure 8,  $\mathcal{X}$  and  $\mathcal{Y}$  are sets of reference FFHQ features and noisy FFHQ features, respectively. The experimental results show that TopP&R actually reflects well the different degrees of distortion added to the images.

# 5.2.5 RANKING BETWEEN GENERATIVE MODELS

One of the major caveats with two-score metrics is that they make it difficult to rank between different models; e.g., which model is better? High fidelity with low diversity or low fidelity with high diversity? In the case of traditional precision and Recall, this problem could be solved by using F1-score, which is the harmonic mean of fidelity and diversity. However, unlike the traditional ones, the F1-score based on P&R or D&C does not provide a reliable or stable score due to their inherent instability and unboundedness. Thanks to its stability and robustness to various perturbations, we find that the TopP&R-based F1 score offers consistent ranking with FID (Table 1).

Table 1: Generative models ranked by FID and F1-scores based on TopP&R, D&C, and P&R, respectively. The  $\mathcal{X}$  and  $\mathcal{Y}$  are embedded with ImageNet pretrained networks.  

<table><tr><td></td><td>Model</td><td>StyleGAN2</td><td>ReACGAN</td><td>BigGAN</td><td>PDGAN</td><td>ACGAN</td><td>WGAN</td></tr><tr><td rowspan="7">VGG16 InceptionV3</td><td>FID (↓)</td><td>3.78</td><td>3.87</td><td>4.16</td><td>31.54</td><td>33.39</td><td>107.68</td></tr><tr><td>TopP&amp;R (↑)</td><td>0.9769</td><td>0.8457</td><td>0.7751</td><td>0.7339</td><td>0.6951</td><td>0.0163</td></tr><tr><td>D&amp;C (↑)</td><td>0.9626</td><td>0.9409</td><td>1.1562</td><td>0.4383</td><td>0.3883</td><td>0.1913</td></tr><tr><td>P&amp;R (↑)</td><td>0.6232</td><td>0.3320</td><td>0.3278</td><td>0.1801</td><td>0.0986</td><td>0.0604</td></tr><tr><td>TopP&amp;R (↑)</td><td>0.9754</td><td>0.5727</td><td>0.7556</td><td>0.4021</td><td>0.3463</td><td>0.0011</td></tr><tr><td>D&amp;C (↑)</td><td>0.9831</td><td>1.0484</td><td>0.9701</td><td>0.9872</td><td>0.8971</td><td>0.6372</td></tr><tr><td>P&amp;R (↑)</td><td>0.6861</td><td>0.1915</td><td>0.3526</td><td>0.0379</td><td>0.0195</td><td>0.0001</td></tr></table>

# 5.3 RELATED WORKS

# 5.3.1 PERSISTENT HOMOLOGY AND DEEP LEARNING

Topology shows various potentials in the field of deep learning by introducing a new perspective on the support estimation, a new distance function robust to the noisy information, and a technique for GAN evaluation. Chen et al. (2017) introduces a method for approximating the support of a distribution using general density estimator and the Hausdorff distance and a new visualization method for support. Chazal et al. (2011) proposes distance-to-measure, a robust Wasserstein distance function for perturbation, as an alternative to the characteristic that existing distance functions are not robust to outliers. For the evaluation, one of the recent metric called MTop-Divergence (Barannikov et al., 2021) uses the summation (or in another word statistics) of the life-length of homology to score which manifold is containing more important topological signals. While M-Top-Divergence directly use persistent homology to score the deep-learning models, we employ topology to estimate a robust and stable manifold.

# 5.3.2 EVALUATION METRICS

Various evaluation metrics for generative models have been recently proposed (Salimans et al., 2016; Heusel et al., 2017; Sajjadi et al., 2018; Kynkänniemi et al., 2019; Naeem et al., 2020; Borji, 2022). One of the earliest methods is Inception Score (IS) (Szegedy et al., 2016), which measures the divergence of generated samples on the InceptionV3 embedding space. However, IS fails to capture the simultaneous mode drop and only considers the population distribution. Fréchet Inception Distance (FID) (Heusel et al., 2017) measures the difference in the means and variances of the real and fake features. Since FID assumes the multi-Gaussian distribution of the features, if the true feature distribution is not normally distributed, the estimation becomes highly unreliable. Unlike IS and FID, which give a single score, some metrics separate the score into two components, the fidelity and diversity Sajjadi et al. (2018); Kynkänniemi et al. (2019); Naeem et al. (2020). While Topological Precision and Recall (TopP&R) falls into this category, unlike the others, it does not assume strong regularity conditions.

# 6 CONCLUSIONS

Recently, many works have been proposed to score the fidelity and diversity of generative models. However, none of them has focused on an accurate estimation of supports even though this is one of the key components in the evaluation pipeline. In this paper, we proposed topological precision and recall (TopP&R) that provides a systematical fix for robustly estimating the manifold by employing topological and statistical ideas. Our theoretical and experimental results showed that TopP&R serves as a robust and reliable evaluation metric under various noisy conditions, including mode collapse, outliers, and Non-IID perturbations.

# REFERENCES

Serguei Barannikov, Ilya Trofimov, Grigorii Sotnikov, Ekaterina Trimbach, Alexander Korotin, Alexander Filippov, and Evgeny Burnaev. *Manifold topology divergence: a framework for comparing data manifolds.* Advances in Neural Information Processing Systems, 34, 2021.  
Ali Borji. Pros and cons of gan evaluation measures: New developments. Computer Vision and Image Understanding, 215:103329, 2022.  
Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.  
Gunnar Carlsson. Topology and data. Bull. Amer. Math. Soc. (N.S.), 46(2):255-308, 2009. ISSN 0273-0979. doi: 10.1090/S0273-0979-09-01249-X. URL https://doi.org/10.1090/S0273-0979-09-01249-X.  
Frédéric Chazal and Bertrand Michel. An introduction to topological data analysis: Fundamental and practical aspects for data scientists. Frontiers Artif. Intell., 4:667963, 2021. doi: 10.3389/frai.2021.667963. URL https://doi.org/10.3389/frai.2021.667963.  
Frédéric Chazal, David Cohen-Steiner, and Quentin Mérigot. Geometric inference for probability measures. Foundations of Computational Mathematics, 11(6):733-751, 2011.  
Frédéric Chazal, Brittany Fasy, Fabrizio Lecci, Alessandro Rinaldo, Aarti Singh, and Larry Wasserman. On the bootstrap for persistence diagrams and landscapes. Modelirovanie i Analiz Informacionnyh Sistem, 20, 11 2013. doi: 10.18255/1818-1015-2013-6-111-120.  
Frédéric Chazal, Brittany Terese Fasy, Fabrizio Lecci, Alessandro Rinaldo, and Larry Wasserman. Stochastic convergence of persistence landscapes and silhouettes. J. Comput. Geom., 6(2):140-161, 2015.  
Yen-Chi Chen, Christopher R Genovese, and Larry Wasserman. Density level sets: Asymptotics, inference, and visualization. Journal of the American Statistical Association, 112(520):1684-1696, 2017.  
Herbert Edelsbrunner and John L. Harer. Computational topology. American Mathematical Society, Providence, RI, 2010. ISBN 978-0-8218-4925-5. doi: 10.1090/mbk/069. URL https://doi.org/10.1090/mbk/069. An introduction.  
Brittany Terese Fasy, Fabrizio Lecci, Alessandro Rinaldo, Larry Wasserman, Sivaraman Balakrishnan, and Aarti Singh. Confidence sets for persistence diagrams. Ann. Statist., 42(6):2301-2339, 2014. ISSN 0090-5364. doi: 10.1214/14-AOS1252. URL https://doi.org/10.1214/14-AOS1252.  
Allen Hatcher. *Algebraic topology*. Cambridge University Press, Cambridge, 2002. ISBN 0-521-79160-X; 0-521-79540-0.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840-6851, 2020.  
William B Johnson, Joram Lindenstrauss, and Gideon Schechtman. Extensions of lipschitz maps into banach spaces. *Israel Journal of Mathematics*, 54(2):129-138, 1986.  
Minguk Kang and Jaesik Park. Contragan: Contrastive learning for conditional image generation. Advances in Neural Information Processing Systems, 33:21357-21369, 2020.  
Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4401-4410, 2019.

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8110-8119, 2020.  
Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. Advances in Neural Information Processing Systems, 34, 2021.  
Jisu Kim, Jaehyeok Shin, Alessandro Rinaldo, and Larry Wasserman. Uniform convergence rate of the kernel density estimator adaptive to intrinsic volume dimension. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 3398-3407. PMLR, 09-15 Jun 2019. URL https://proceedings.mlr.press/v97/kim19e.html.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Michael R. Kosorok. Introduction to empirical processes and semiparametric inference. Springer Series in Statistics. Springer, New York, 2008. ISBN 978-0-387-74977-8. doi: 10.1007/978-0-387-74978-5. URL https://doi.org/10.1007/978-0-387-74978-5.  
Tuomas Kynkänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in Neural Information Processing Systems, 32, 2019.  
Tuomas Kynkänniemi, Tero Karras, Miika Aittala, Timo Aila, and Jaakko Lehtinen. The role of imagenet classes in fr\`echet inception distance. arXiv preprint arXiv:2203.06026, 2022.  
Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjoy Choi, and Jaejun Yoo. Reliable fidelity and diversity metrics for generative models. In International Conference on Machine Learning, pp. 7176-7185. PMLR, 2020.  
Michael H. Neumann. Strong approximation of density estimators from weakly dependent observations by density estimators from independent observations. Ann. Statist., 26(5):2014-2048, 1998. ISSN 0090-5364. doi: 10.1214/aos/1024691367. URL https://doi.org/10.1214/aos/1024691367.  
Mehdi SM, Sajjadi, Olivier Bachem, Mario Lucic, Olivier Bousquet, and Sylvain Gelly. Assessing generative models via precision and recall. Advances in Neural Information Processing Systems, 31, 2018.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.  
Axel Sauer, Kashyap Chitta, Jens Müller, and Andreas Geiger. Projected gans converge faster. Advances in Neural Information Processing Systems, 34, 2021.  
Axel Sauer, Katja Schwarz, and Andreas Geiger. Stylegan-xl: Scaling stylegan to large diverse datasets. arXiv preprint arXiv:2202.00273, 2022.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2818-2826, 2016.  
George R Terrell and David W Scott. Variable kernel density estimation. The Annals of Statistics, pp. 1236-1265, 1992.  
A.W. van der Vaart. Asymptotic Statistics. Asymptotic Statistics. Cambridge University Press, 2000.  
ISBN 9780521784504. URL https://books.google.fr/books?id=UEuQEM5RjWgC.

Hubert Wagner, Chao Chen, and Erald Vuçini. Efficient computation of persistent homology for cubical data. In Topological methods in data analysis and visualization II, pp. 91-106. Springer, 2012.  
Larry Wasserman. Topological data analysis. Annu. Rev. Stat. Appl., 5:501-535, 2018. ISSN 2326-8298. doi: 10.1146/annurev-statistics-031017-100045. URL https://doi.org/10.1146/annurev-statistics-031017-100045.
