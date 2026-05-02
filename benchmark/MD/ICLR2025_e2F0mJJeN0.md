# GEOMETRIC MEDIAN (GM) MATCHING FOR ROBUST DATA PRUNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Data pruning, the combinatorial task of selecting a small and informative subset from a large dataset, is crucial for mitigating the enormous computational costs associated with training data-hungry modern deep learning models at scale. Since large-scale data collections are invariably noisy, developing data pruning strategies that remain robust even in the presence of corruption is critical in practice. In response, we propose GM MATCHING – a herding (Welling, 2009) style greedy algorithm – that yields a  $k$ -subset such that the mean of the subset approximates the geometric median of the (potentially) noisy dataset. Theoretically, we show that GM Matching enjoys an improved  $\mathcal{O}(1 / k)$  scaling over  $\mathcal{O}(1 / \sqrt{k})$  scaling of uniform sampling; while achieving the optimal breakdown point of  $1 / 2$  even under arbitrary corruption. Extensive experiments across popular deep learning benchmarks indicate that GM Matching consistently outperforms prior state-of-the-art; the gains become more profound at high rates of corruption and aggressive pruning rates; making it a strong baseline for robust data pruning.

# 1 INTRODUCTION

Recent success of deep learning has been largely fueled by training gigantic models over vast amounts of training data (Radford et al., 2021; 2018; Brown et al., 2020; Kaplan et al., 2020; Hestness et al., 2017). Such large scale training, however is associated with enormous computational costs hindering the path to democratizing AI (Paul et al., 2021). Data pruning, the combinatorial task of downsizing a large training set into a small informative subset (Feldman, 2020; Agarwal et al., 2005; Muthukrishnan et al., 2005; Har-Peled, 2011; Feldman & Langberg, 2011), is a promising approach for reducing the enormous computational and storage costs of modern deep learning.

# EXISTING DATA PRUNING STRATEGIES

Consequently, a large body of recent works have been proposed to solve the data selection problem. At a high level, there are two main directions: One set of data pruning approaches rely on some carefully designed pruning metrics, rank the training samples based on the scores and retain a fraction of them as representative samples (super samples), used for training the downstream model. For example, (Xia et al., 2022; Joshi & Mirzasoleiman, 2023; Sorscher et al., 2022) calculate the importance score of a sample in terms of the distance from the centroid of its corresponding class marginal. Samples closer to the centroid are considered most prototypical (easy) and those far from the centroid are treated as least prototypical (hard). A second set of works reformulate this problem as minimizing a moment matching objective (Chen et al., 2010; Campbell & Broderick, 2018; Dwivedi & Mackey, 2021) that aims to select a subset whose mean closely matches that of the entire dataset.

While this work primarily focuses on spatial approaches, it is worth mentioning that the canonical importance scoring criterion have been proposed in terms gradient norm (Paul et al., 2021; Needell et al., 2014), uncertainty (Pleiss et al., 2020) and forgetfulness (Toneva et al., 2018). Typically, samples closer to the class centroid in feature space tend to have lower gradient norms, exhibit lower uncertainty, and are harder to forget during training. In contrast, samples farther from the centroid generally have higher gradient norms, greater uncertainty, and are easier to forget (Paul et al., 2021; Sorscher et al., 2022; Xia et al., 2022). Moreover, (Mirzasoleiman et al., 2020) extended the moment-matching approach to the gradient space, selecting subsets that preserve the overall gradient statistics of the full dataset.

![](images/a13af88c8686d0df61f1aafaf6df1c84ea8fee934d78adbb5d6ae3afbd8af0b5.jpg)  
(a) UNIFORM

![](images/1f8992eb5c7bb53f2cd7cc7f830b32500f88ad9b08f0ac643fb6f121f24d2caa.jpg)  
(b) EASY

![](images/0ff0a3e97c2998ccd220496df30bda4e67cf08021b1ae156e7d567406875da76.jpg)  
(c) HARD

![](images/92576e069ef535c6e9140db25fc3453da51adc43ffa27d0fd1e2ffe86215b61e.jpg)  
(d) MODERATE

![](images/f7b597f3850232fd66c40c0c42d63deb762151aa258426cdf8dbbd96372845c7.jpg)  
(e) HERDING

![](images/9ab4d7c738313d1a05503d58ff370c75c90d78115abd705d62df57e253d3e3d6.jpg)  
Figure 1: DATA PRUNING IN THE WILD: Data Pruning methods applied to samples from a multivariate Gaussian distribution (blue), with  $40\%$  replaced by an adversarial distribution (red). We subset  $10\%$  of the examples using: (UNIFORM) Random Sampling, (EASY) Selection of samples closest to the centroid. (HARD) Selection of samples farthest from the centroid. (MODERATE) Selection of samples closest to the median distance from the centroid. (HERDING) Moment Matching, (GM MATCHING) Robust Moment (GM) Matching (6). GM MATCHING yields significantly more robust (from the true distribution) subset than the other approaches.  
(f) GM MATCHING

# ROBUSTNESS VS DIVERSITY

In the ideal scenario (i.e. in absence of any corruption), hard examples are known to contribute the most in downstream generalization performance (Katharopoulos & Fleuret, 2018; Joshi et al., 2009; Huang et al., 2010; Balcan et al., 2007) as they often capture most of the usable information in the dataset (Xu et al., 2020). On the other hand, in realistic noisy scenarios involving outliers, this strategy often fails since the noisy examples are wrongly deemed informative for training (Zhang & Sabuncu, 2018; Park et al., 2024). Pruning methods specifically designed for such noisy scenarios thus propose to retain the most representative (easy) samples (Pleiss et al., 2020; Jiang et al., 2018; Har-Peled et al., 2006; Shah et al., 2020; Shen & Sanghavi, 2019). However, by only choosing samples far from the decision boundary, these methods ignore the more informative uncorrupted less prototypical samples. This can often result in sub-optimal downstream performance and in fact can also lead to degenerate solutions due to a covariance-shift problem (Sugiyama & Kawanabe, 2012); giving rise to a robustness vs diversity trade off (Xia et al., 2022). This restricts the applicability of existing pruning methods, as realistic scenarios often deviate from expected conditions, making it challenging or impractical to adjust the criteria and methods accordingly.

# Algorithm 1 GEOMETRIC MEDIAN MATCHING

Initialize: A finite collection of  $\alpha$  corrupted (Definition 1) observations  $\mathcal{D}$  defined over Hilbert space  $\mathcal{H} \in \mathbb{R}^d$ , equipped with norm  $\|\cdot\|$  and inner  $\langle \cdot \rangle$  operators; initial weight vector  $\theta_0 \in \mathcal{H}$ .

Robust Mean Estimation:  $\pmb{\mu}^{\mathrm{GM}} = \arg \min_{\mathbf{z}\in \mathcal{H}}\sum_{\mathbf{x}_i\in \mathcal{D}}\| \mathbf{z} - \mathbf{x}_i\|$

$\mathcal{D}_S\gets \emptyset$

for iterations  $t = 0,1,\dots ,k - 1$  do

$\mathbf{x}_{t + 1}\coloneqq \arg \max_{\mathbf{x}\in \mathcal{D}}\langle \pmb {\theta}_t,\mathbf{x}\rangle$ $\pmb{\theta}_{t + 1}\coloneqq \pmb {\theta}_t + \pmb{\mu}_\epsilon^{\mathrm{GM}} - \mathbf{x}_{t + 1}$ $\mathcal{D}_S\coloneqq \mathcal{D}_S\cup \mathbf{x}_{t + 1}$ $\mathcal{D}\coloneqq \mathcal{D}\setminus \mathbf{x}_{t + 1}$

end

return:  $\mathcal{D}_S$

# OVERVIEW OF OUR APPROACH

To go beyond these limitations, we study data pruning in presence of corruption. Specifically, we consider the  $\alpha$  corruption framework (Definition 1), where  $0\leq \psi < \frac{1}{2}$  fraction of the samples are allowed be arbitrarily perturbed. This allowance for arbitrary corruption enables us to generalize many practical robustness scenarios; including corrupt feature / label and adversarial attacks.

We make a key observation that, traditional pruning methods typically use the empirical mean to calculate the centroid of the samples, which then guides the selection process based on how representative those samples are. However, the empirical mean is highly susceptible to outliers – in fact, it is possible to construct a single adversarial example to arbitrarily perturb the empirical mean. As a consequence, in the presence of arbitrary corruption, the conventional distinction between easy (robust) and hard samples breaks down, leading to the selection of subsets that are significantly compromised by corruption as illustrated in Figure 1, depicting sampling from a corrupted Gaussian.

In response, we propose a data pruning strategy that fosters balanced diversity, effectively navigating various regions of the distribution while avoiding distant, noisy points. Our key idea is to replace the target moment in the standard moment matching objective with a robust surrogate – Geometric Median (Weber et al., 1929; Weiszfeld, 1937) – a classical robust estimator of the mean. In particular, we optimize over finding a subset minimizes the discrepancy between the subset's mean and the GM (Definition 3) of the (potentially noisy) dataset using greedy herding (Welling, 2009) style update rule. We call our algorithm Geometric Median Matching as described in Algorithm 1.

# CONTRIBUTIONS

Overall, our contributions can be summarized as follows:

- We systematically and formally investigate and extend data pruning in presence of corruption. In particular, we study data pruning under the gross corruption framework (Definition 1), where up to  $\frac{1}{2}$  fraction of the training examples are allowed to be arbitrarily corrupted. We note that, existing pruning heuristics (including the ones proposed for robust scenarios) break down under this strong corruption, due to empirical mean's vulnerability to corruption (Section 4, Figure 1).  
- Motivated by this key observation, we exploit the robustness property of GM (Definition 3), to design a novel robust moment matching objective (6). It aims at finding a subset such that the mean of the subset approximates the GM of the noisy dataset. We minimize over this objective using greedy herding (Welling, 2009) style update rule. We call the resulting data pruning algorithm GM MATCHING and formally describe it in Algorithm 1.  
- Leveraging classical robustness properties of GM, we show that, GM Matching converges to a bounded neighborhood of original underlying mean, at an impressive  $\mathcal{O}(1 / k)$  rate while being robust even when up to  $1 / 2$  of the samples are arbitrarily corrupted (Theorem 1).  
- Extensive experiments over CIFAR 10/100, Tiny ImageNet, across feature corruption, label noise and adversarial attacks indicate the superiority of GM Matching over existing methods. We improve over prior work almost in all settings, the gains are especially more profound (often by more than  $10\%$ ) in presence of corruption and at aggressive pruning rates; making GM Matching a strong baseline for future research in robust data pruning.

# 2 PROBLEM SETUP: ROBUST DATA PRUNING

Given a set of samples  $\mathcal{D}$ , the goal of data pruning is to select a subset of the most representative samples  $\mathcal{D}_S \subseteq \mathcal{D}$ , that can approximate the underlying distribution well. Data pruning methods achieve this by first defining a pruning criterion e.g. based on distance, uncertainty, diversity; and then selecting a subset that best satisfies these criteria to represent the full dataset effectively. If such a subset (also referred to as coreset) can be found in a compute efficient manner, then training a parametric model on the subset, typically yields similar generalization performance as training on the entire dataset while resulting in significant speed up when  $|\mathcal{D}_S| \ll |\mathcal{D}|$ . However, for machine learning systems deployed in the wild,  $\mathcal{D}$  is often noisy and imperfect due to the difficulty and expense of obtaining perfect semantic annotations for large amounts of data, adversarial attacks or simply measurement noises.

Definition 1 ( $\alpha$ -corruption). Given a set of observations from the original distribution of interest, an adversary is allowed to inspect all the samples and arbitrarily perturb up to  $\psi \in [0, \frac{1}{2})$  fraction of them. We refer to a set of samples  $\mathcal{D} = \mathcal{D}_{\mathcal{G}} \cup \mathcal{D}_{\mathcal{B}}$  as  $\alpha$ -corrupted,  $\alpha := |\mathcal{D}_{\mathcal{B}}| / |\mathcal{D}_{\mathcal{G}}| = \frac{\psi}{1 - \psi} < 1$  and  $\mathcal{D}_{\mathcal{B}}, \mathcal{D}_{\mathcal{G}}$  denote the sets of corrupt and clean samples respectively.

To this end, this work studies data pruning under the  $\alpha$ -corruption framework (Definition 1), where a fraction  $\psi \in [0,\frac{1}{2})$  of the samples can be arbitrarily corrupted - a strong corruption model (Diakonikolas et al., 2019; Acharya et al., 2022) that generalizes the popular Huber Contamination Model (Huber, 1992), as well as the notorious Byzantine Corruption (Lamport et al., 1982).

Given an  $\alpha$ -corrupted set of observations  $\mathcal{D} = \mathcal{D}_{\mathcal{G}} \cup \mathcal{D}_{\mathcal{B}}$ , the goal of ROBUST DATA PRUNING is thus to judiciously select a subset  $\mathcal{D}_S \subseteq \mathcal{D}$ ; that encapsulates the underlying clean (uncorrupted) distribution induced by subset  $\mathcal{D}_{\mathcal{G}}$  without any a-priori knowledge about the corrupted samples.

We measure the robustness of data pruning algorithms via breakdown point analysis (Donoho & Huber, 1983) – a classic tool in robust optimization to assess the resilience of an estimator.

Definition 2 (Breakdown Point). The breakdown point of an estimator is defined as the smallest fraction of contaminated data that can cause the estimator to result in arbitrarily large errors.

In the context of Definition 1, we say that an estimator achieves optimal breakdown point 1/2 (Lopuhaa et al., 1991) if it remains robust in presence of  $\alpha$ -corruption  $\forall \alpha < 1$ .

# 3 WARM UP : MOMENT MATCHING

In the uncorrupted setting i.e. when  $\mathcal{D}_{\mathcal{B}} = \emptyset$ , a natural and widely used approach for data pruning is to formulate it as the following combinatorial MOMENT MATCHING objective:

$$
\underset {\mathcal {D} _ {S} \subseteq \mathcal {D}, | \mathcal {D} _ {S} | = k} {\arg \min } \left\| \frac {1}{| \mathcal {D} |} \sum_ {\mathbf {x} _ {i} \in \mathcal {D}} \mathbf {x} _ {i} - \frac {1}{k} \sum_ {\mathbf {x} _ {i} \in \mathcal {D} _ {S}} \mathbf {x} _ {i} \right\| ^ {2} \tag {1}
$$

Observe that, (1) is an instance of the famous set function maximization problem – known to be NP hard via a reduction from  $k$ -set cover (Feige, 1998). Despite its intractability, (Mirzasoleiman et al., 2020) demonstrated a transformation into a submodular set cover problem, enabling efficient solution via greedy algorithms (Nemhauser et al., 1978; Wolsey, 1982). The greedy approach: also referred to as kernel herding (Welling, 2009; Welling & Chen, 2010) starts with a suitably chosen  $\theta_0 \in \mathbb{R}^d$ ; and iteratively adds samples via the following update rule:

$$
\mathbf {x} _ {t + 1} := \underset {\mathbf {x} \in \mathcal {D}} {\arg \max } \left\langle \boldsymbol {\theta} _ {t}, \mathbf {x} \right\rangle \tag {2}
$$

$$
\boldsymbol {\theta} _ {t + 1} := \boldsymbol {\theta} _ {t} + \left(\frac {1}{| \mathcal {D} |} \sum_ {\mathbf {x} _ {i} \in \mathcal {D}} \mathbf {x} _ {i} - \mathbf {x} _ {t + 1}\right) \tag {3}
$$

It's worth noting that this algorithm is an infinite memory, deterministic process as at each iteration  $T$ ,  $\pmb{\theta}_T$  encapsulates the entire sampling history:  $\pmb{\theta}_T = \pmb{\theta}_0 + T\pmb{\mu} - \sum_{t=1}^T \mathbf{x}_t$  where  $\pmb{\mu} = \frac{1}{|\mathcal{D}|}\sum_{\mathbf{x}_i \in \mathcal{D}}\mathbf{x}_i$ . Conceptually,  $\pmb{\theta}_T$  represents the vector pointing towards under-sampled regions of the target distribution induced by  $\mathcal{D}$  at iteration  $T$ . The algorithm's greedy selection strategy aligns each new sample with  $\pmb{\theta}$ , effectively herding new points to fill the gaps left by earlier selections. Remarkably, (Chen et al., 2010) showed that this simple greedy update rule achieves an impressive  $\mathcal{O}(1/k)$  convergence rate for (1), a quadratic improvement over random sampling where the error decreases at the rate  $\mathcal{O}(1/\sqrt{k})$ . The result holds if  $\| \mathbf{x} \| \leq R \forall \mathbf{x} \in \mathcal{D}$  for some constant  $R$  and as long as the target moment is in the relative interior of  $\mathcal{C} = \mathrm{conv}\{\mathbf{x} | \mathbf{x} \in \mathcal{D}\}$  (Proposition 1 (Chen et al., 2010)).

# 4 GEOMETRIC MEDIAN (GM) MATCHING

Despite its strong performance guarantees in the vanilla (uncorrupted) setting, we argue that the algorithm can result in arbitrarily poor solution in the noisy setting. The vulnerability can be

attributed to the estimation of target moment via empirical mean – notorious for its sensitivity to outliers. Consider a single adversarial sample:  $\mathbf{x}^{\mathcal{B}} = |\mathcal{D}|\boldsymbol{\mu}^{\mathcal{B}} - \sum_{\mathbf{x}\in \mathcal{D}\backslash \mathbf{x}^{\mathcal{B}}}\mathbf{x}$ , shifting the empirical mean to adversary chosen arbitrary target  $\boldsymbol{\mu}^{\mathcal{B}}$ . This implies that the empirical mean can't tolerate even a single grossly corrupted sample i.e. yields lowest possible asymptotic breakdown point of 0. As a consequence, optimizing over the moment matching objective (1) no longer guarantee convergence to the true underlying (uncorrupted) moment  $\boldsymbol{\mu}^{\mathcal{G}} = \mathbb{E}_{\mathbf{x}\in \mathcal{D}_{\mathcal{G}}}\mathbf{x}$ , instead the algorithm can be hijacked by a single bad sample, warping the solution towards an adversarial target.

Motivated by this key observation, a natural idea to enable ROBUST MOMENT MATCHING is to replace the empirical mean in (1) with a robust surrogate estimator of the target moment and perform greedy herding updates to match the robust surrogate. Ideally, the robust estimate  $\pmb{\mu}$  should ensure that the estimation error  $\Delta = \| \pmb {\mu} - \pmb{\mu}^{\mathcal{G}}\| \leq \delta$  remain bounded, even when the observations are  $\alpha$ -corrupted (Definition 1). Moreover, the estimate should reside inside the relative interior of  $\mathcal{C}_{\mathcal{G}} = \mathrm{conv}\{\mathbf{x}|\mathbf{x}\in \mathcal{D}_{\mathcal{G}}\}$  to ensure the linear convergence guarantee.

In the univariate setting, various robust mean estimators, such as the median and the trimmed mean, are known to achieve the optimal breakdown point  $1/2$ . A common strategy to extend these methods to the multivariate setting is to perform univariate estimation independently along each dimension. However, in high dimensions, these estimates need not lie in the convex hull of the samples and are not orthogonal equivariant and can even become degenerate in the overparameterized settings ( $n \ll d$ ) (Lopuhaa et al., 1991; Rousseeuw & Leroy, 2005). On the other hand, M-estimators are affine equivariant but have breakdown point at most  $1/(d+1)$  (Donoho & Huber, 1983).

Definition 3 (Geometric Median). Given a finite collection of observations  $\{\mathbf{x}_1,\mathbf{x}_2,\dots \mathbf{x}_n\}$  defined over Hilbert space  $\mathcal{H}\in \mathbb{R}^d$  , equipped with norm  $\| \cdot \|$  and inner  $\langle \cdot \rangle$  operators, the geometric median(or Fermat-Weber point) (Weber et al., 1929) is defined as:

$$
\boldsymbol {\mu} ^ {\mathrm {G M}} = \operatorname {G M} \left(\left\{\mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots \mathbf {x} _ {n} \right\}\right) = \underset {\mathbf {z} \in \mathcal {H}} {\arg \min } \left[ \rho (\mathbf {z}) := \sum_ {i = 1} ^ {n} \left\| \mathbf {z} - \mathbf {x} _ {i} \right\| \right] \tag {4}
$$

In this context, Geometric Median (GM) (Definition 3) – a well studied spatial estimator, known for several nice properties like rotation and translation invariance and optimal breakdown point of 1/2 under gross corruption (Minsker et al., 2015; Kemperman, 1987). Moreover, the estimate is guaranteed to lie in the relative interior of the convex hull of the majority (good) points i.e.  $\mu^{\mathrm{GM}} \in \mathcal{C}_{\mathcal{G}}$  making it a natural choice to estimate the target moment.

Computing the GM exactly, is known to be hard as linear time algorithm exists (Bajaj, 1988), making it is necessary to rely on approximation methods to estimate the geometric median (Weiszfeld, 1937; Vardi & Zhang, 2000; Cohen et al., 2016). We call a point  $\pmb{\mu}_{\epsilon}^{\mathrm{GM}} \in \mathcal{H}$  an  $\epsilon$  accurate GM if it holds:

$$
\sum_ {i = 1} ^ {n} \left\| \boldsymbol {\mu} _ {\epsilon} ^ {\mathrm {G M}} - \mathbf {x} _ {i} \right\| \leq (1 + \epsilon) \sum_ {i = 1} ^ {n} \left\| \boldsymbol {\mu} ^ {\mathrm {G M}} - \mathbf {x} _ {i} \right\| \tag {5}
$$

We then, exploit the breakdown and translation invariance property of GM and solve for the following ROBUST MOMENT MATCHING objective - a robust surrogate of (1):

$$
\underset {\mathcal {D} _ {\mathcal {S}} \subseteq \mathcal {D}, | \mathcal {D} _ {\mathcal {S}} | = k} {\arg \min } \left\| \boldsymbol {\mu} _ {\epsilon} ^ {\mathrm {G M}} - \frac {1}{k} \sum_ {\mathbf {x} _ {i} \in \mathcal {S}} \mathbf {x} _ {i} \right\| ^ {2} \tag {6}
$$

Consequently, we perform herding style greedy minimization of the error (6):

We start with a suitably chosen  $\theta_0\in \mathbb{R}^d$ ; and repeatedly perform the following updates, adding one sample at a time,  $k$  times:

$$
\mathbf {x} _ {t + 1} := \underset {\mathbf {x} \in \mathcal {D}} {\arg \max } \left\langle \boldsymbol {\theta} _ {t}, \mathbf {x} \right\rangle \tag {7}
$$

$$
\boldsymbol {\theta} _ {t + 1} := \boldsymbol {\theta} _ {t} + \left(\boldsymbol {\mu} _ {\epsilon} ^ {\mathrm {G M}} - \mathbf {x} _ {t + 1}\right) \tag {8}
$$

We refer to the resulting robust data pruning approach as GM MATCHING. For ease of exposition, let  $\theta_0 = \mu_\epsilon^{\mathrm{GM}}$ . Then, at iteration  $t = T$ , GM MATCHING is performing:

$$
\mathbf {x} _ {T + 1} = \underset {\mathbf {x} \in \mathcal {D}} {\arg \max } \left[ \left\langle \boldsymbol {\mu} _ {\epsilon} ^ {\mathrm {G M}}, \mathbf {x} \right\rangle - \frac {1}{T + 1} \sum_ {t = 1} ^ {T} \left\langle \mathbf {x}, \mathbf {x} _ {t} \right\rangle \right] \tag {9}
$$

Greedy updates in the direction that reduces the accumulated error, encourages the algorithm to explore underrepresented regions of the feature space, promoting diversity. By matching the GM rather than the empirical mean, the algorithm imposes larger penalties on outliers, which lie farther from the core distribution. This encourages GM MATCHING to prioritize samples near the convex hull of uncorrupted points  $\mathcal{C}_{\mathcal{G}} = \mathrm{conv}\{\phi_{\mathcal{B}}(\mathbf{x})|\mathbf{x} \in \mathcal{D}_{\mathcal{G}}\}$ . As a result, the algorithm promotes diversity in a balanced manner, effectively exploring different regions of the distribution while avoiding distant, noisy points, thus mitigating the robustness vs. diversity trade-off discussed in Section 1. This makes GM MATCHING an excellent choice for data pruning in the wild.

# THEORETICAL GUARANTEE

In order to theoretically characterize the convergence behavior of GM MATCHING, we first exploit the robustness property of GM (Acharya et al., 2022; Cohen et al., 2016; Chen et al., 2017) to get an upper bound on the estimation error w.r.t the underlying true mean. Next, we use the property that GM is guaranteed to lie in the interior of the convex hull of majority of the samples (Minsker et al., 2015; Boyd & Vandenberghe, 2004) which follows from the properties of convex sets. Combining these two results we establish the following convergence guarantee for GM MATCHING:

Theorem 1. Suppose that, we are given, a set of  $\alpha$ -corrupted samples  $\mathcal{D} = \mathcal{D}_{\mathcal{G}} \cup \mathcal{D}_{\mathcal{B}}$  (Definition 1) and an  $\epsilon$  approx.  $\mathrm{GM}(\cdot)$  oracle (4). Further assume that  $\| \mathbf{x}\| \leq R\forall \mathbf{x} \in \mathcal{D}$  for some constant  $R$ . Then, GM MATCHING guarantees that the mean of the selected  $k$ -subset  $\mathcal{D}_{\mathcal{S}} \subseteq \mathcal{D}$  converges to a  $\delta$ -neighborhood of the uncorrupted (true) mean  $\pmb{\mu}^{\mathcal{G}} = \mathbb{E}_{\mathbf{x} \in \mathcal{D}_{\mathcal{G}}}(\mathbf{x})$  at the rate  $\mathcal{O}\left(\frac{1}{k}\right)$  such that:

$$
\delta^ {2} = \mathbb {E} \left\| \frac {1}{k} \sum_ {\mathbf {x} _ {i} \in \mathcal {D} _ {\mathcal {S}}} \mathbf {x} _ {i} - \boldsymbol {\mu} ^ {\mathcal {G}} \right\| ^ {2} \leq \frac {8 | \mathcal {D} _ {\mathcal {G}} |}{(| \mathcal {D} _ {\mathcal {G}} | - | \mathcal {D} _ {\mathcal {B}} |) ^ {2}} \sum_ {\mathbf {x} \in \mathcal {D} _ {\mathcal {G}}} \mathbb {E} \left\| \mathbf {x} - \boldsymbol {\mu} ^ {\mathcal {G}} \right\| ^ {2} + \frac {2 \epsilon^ {2}}{(| \mathcal {D} _ {\mathcal {G}} | - | \mathcal {D} _ {\mathcal {B}} |) ^ {2}} \tag {10}
$$

This result suggests that, even in presence of  $\alpha$  corruption, the proposed algorithm GM Matching converges to a neighborhood of the true mean, where the neighborhood radius depends on two terms - the first term depends on the variance of the uncorrupted samples and the second term depends on how accurately the GM is calculated. Furthermore the bound holds  $\forall \alpha = \mathcal{D}_{\mathcal{B}} / \mathcal{D}_{\mathcal{G}} < 1$  implying GM Matching remains robust even when half of the samples are arbitrarily corrupted i.e. it achieves the optimal breakdown point of  $1/2$ . The detailed proofs are provided in Section 8.

# 5 EXPERIMENTS

In this section, we outline our experimental setup, present our key empirical findings, and discuss deeper insights into the performance of GM Matching. Due to space constraint we only present a subset of the results in the main paper. Please refer to Section 8, for additional experimental evidence.

BASELINES: To ensure reproducibility, our experimental setup is identical to (Xia et al., 2022). We compare the proposed GM Matching selection strategy against the following popular data pruning strategies as baselines for comparison: (1) Random; (2) Herding Welling (2009); (3) Forgetting Toneva et al. (2018); (4) GraNd-score Paul et al. (2021); (5) EL2N-score Paul et al. (2021); (6) Optimization-based Yang et al. (2022); (7) Self-sup.-selection Sorscher et al. (2022) and (8) Moderate (Xia et al., 2022). We do not run these baselines for be these baselines are borrowed from (Xia et al., 2020). Additionally, for further ablations we compare GM Matching with many (natural) distance based geometric pruning strategies: (UNIFORM) Random Sampling, (EASY) Selection of samples closest to the centroid; (HARD) Selection of samples farthest from the centroid; (MODERATE) (Xia et al., 2022) Selection of samples closest to the median distance from the centroid; (HERDING) Moment Matching (Chen et al., 2010), (GM MATCHING) Robust Moment (GM) Matching (6).

DATASETS AND NETWORKS: We perform extensive experiments across three popular image classification datasets - CIFAR10, CIFAR100 and Tiny-ImageNet. Our experiments span popular

CIFAR-100  

<table><tr><td>Method / Ratio</td><td>20%</td><td>30%</td><td>40%</td><td>60%</td><td>80%</td><td>100%</td><td>Mean ↑</td></tr><tr><td>Random</td><td>50.26±3.24</td><td>53.61±2.73</td><td>64.32±1.77</td><td>71.03±0.75</td><td>74.12±0.56</td><td>78.14±0.55</td><td>62.67</td></tr><tr><td>Herding</td><td>48.39±1.42</td><td>50.89±0.97</td><td>62.99±0.61</td><td>70.61±0.44</td><td>74.21±0.49</td><td>78.14±0.55</td><td>61.42</td></tr><tr><td>Forgetting</td><td>35.57±1.40</td><td>49.83±0.91</td><td>59.65±2.50</td><td>73.34±0.39</td><td>77.50±0.53</td><td>78.14±0.55</td><td>59.18</td></tr><tr><td>GraNd-score</td><td>42.65±1.39</td><td>53.14±1.28</td><td>60.52±0.79</td><td>69.70±0.68</td><td>74.67±0.79</td><td>78.14±0.55</td><td>60.14</td></tr><tr><td>EL2N-score</td><td>27.32±1.16</td><td>41.98±0.54</td><td>50.47±1.20</td><td>69.23±1.00</td><td>75.96±0.88</td><td>78.14±0.55</td><td>52.99</td></tr><tr><td>Optimization-based</td><td>42.16±3.30</td><td>53.19±2.14</td><td>58.93±0.98</td><td>68.93±0.70</td><td>75.62±0.33</td><td>78.14±0.55</td><td>59.77</td></tr><tr><td>Self-sup.-selection</td><td>44.45±2.51</td><td>54.63±2.10</td><td>62.91±1.20</td><td>70.70±0.82</td><td>75.29±0.45</td><td>78.14±0.55</td><td>61.60</td></tr><tr><td>Moderate-DS</td><td>51.83±0.52</td><td>57.79±1.61</td><td>64.92±0.93</td><td>71.87±0.91</td><td>75.44±0.40</td><td>78.14±0.55</td><td>64.37</td></tr><tr><td>GM Matching</td><td>55.93±0.48</td><td>63.08±0.57</td><td>66.59±1.18</td><td>70.82±0.59</td><td>74.63±0.86</td><td>78.14±0.55</td><td>66.01</td></tr><tr><td colspan="8">Tiny ImageNet</td></tr><tr><td>Random</td><td>24.02±0.41</td><td>29.79±0.27</td><td>34.41±0.46</td><td>40.96±0.47</td><td>45.74±0.61</td><td>49.36±0.25</td><td>34.98</td></tr><tr><td>Herding</td><td>24.09±0.45</td><td>29.39±0.53</td><td>34.13±0.37</td><td>40.86±0.61</td><td>45.45±0.33</td><td>49.36±0.25</td><td>34.78</td></tr><tr><td>Forgetting</td><td>22.37±0.71</td><td>28.67±0.54</td><td>33.64±0.32</td><td>41.14±0.43</td><td>46.77±0.31</td><td>49.36±0.25</td><td>34.52</td></tr><tr><td>GraNd-score</td><td>23.56±0.52</td><td>29.66±0.37</td><td>34.33±0.50</td><td>40.77±0.42</td><td>45.96±0.56</td><td>49.36±0.25</td><td>34.86</td></tr><tr><td>EL2N-score</td><td>19.74±0.26</td><td>26.58±0.40</td><td>31.93±0.28</td><td>39.12±0.46</td><td>45.32±0.27</td><td>49.36±0.25</td><td>32.54</td></tr><tr><td>Optimization-based</td><td>13.88±2.17</td><td>23.75±1.62</td><td>29.77±0.94</td><td>37.05±2.81</td><td>43.76±1.50</td><td>49.36±0.25</td><td>29.64</td></tr><tr><td>Self-sup.-selection</td><td>20.89±0.42</td><td>27.66±0.50</td><td>32.50±0.30</td><td>39.64±0.39</td><td>44.94±0.34</td><td>49.36±0.25</td><td>33.13</td></tr><tr><td>Moderate-DS</td><td>25.29±0.38</td><td>30.57±0.20</td><td>34.81±0.51</td><td>41.45±0.44</td><td>46.06±0.33</td><td>49.36±0.25</td><td>35.64</td></tr><tr><td>GM Matching</td><td>27.88±0.19</td><td>33.15±0.26</td><td>36.92±0.40</td><td>42.48±0.12</td><td>46.75±0.51</td><td>49.36±0.25</td><td>37.44</td></tr></table>

Table 1: No Corruption : Comparing (Test Accuracy) pruning algorithms on CIFAR-100 and Tiny-ImageNet in the uncorrupted setting. ResNet-50 is used both as proxy and for downstream classification.  
Table 2: Image Corruption : Experiments comparing pruning methods when  ${20}\%$  of the images are corrupted. ResNet-50 is used for both proxy (data pruning) and downstream training.  

<table><tr><td>Method / Selection ratio</td><td>20%</td><td>30%</td><td>40%</td><td>60%</td><td>80%</td><td>100%</td><td>Mean ↑</td></tr><tr><td colspan="8">CIFAR-100 with 20% corrupted images</td></tr><tr><td>Random</td><td>40.99±1.46</td><td>50.38±1.39</td><td>57.24±0.65</td><td>65.21±1.31</td><td>71.74±0.28</td><td>74.92±0.88</td><td>57.11</td></tr><tr><td>Herding</td><td>44.42±0.46</td><td>53.57±0.31</td><td>60.72±1.78</td><td>69.09±1.73</td><td>73.08±0.98</td><td>74.92±0.88</td><td>60.18</td></tr><tr><td>Forgetting</td><td>26.39±0.17</td><td>40.78±2.02</td><td>49.95±2.31</td><td>65.71±1.12</td><td>73.67±1.12</td><td>74.92±0.88</td><td>51.30</td></tr><tr><td>GraNd-score</td><td>36.33±2.66</td><td>46.21±1.48</td><td>55.51±0.76</td><td>64.59±2.40</td><td>70.14±1.36</td><td>74.92±0.88</td><td>54.56</td></tr><tr><td>EL2N-score</td><td>21.64±2.03</td><td>23.78±1.66</td><td>35.71±1.17</td><td>56.32±0.86</td><td>69.66±0.43</td><td>74.92±0.88</td><td>41.42</td></tr><tr><td>Optimization-based</td><td>33.42±1.60</td><td>45.37±2.81</td><td>54.06±1.74</td><td>65.19±1.27</td><td>70.06±0.83</td><td>74.92±0.88</td><td>54.42</td></tr><tr><td>Self-sup.-selection</td><td>42.61±2.44</td><td>54.04±1.90</td><td>59.51±1.22</td><td>68.97±0.96</td><td>72.33±0.20</td><td>74.92±0.88</td><td>60.01</td></tr><tr><td>Moderate-DS</td><td>42.98±0.87</td><td>55.80±0.95</td><td>61.84±1.96</td><td>70.05±1.29</td><td>73.67±0.30</td><td>74.92±0.88</td><td>60.87</td></tr><tr><td>GM Matching</td><td>47.12±0.64</td><td>59.17±0.92</td><td>63.45±0.34</td><td>71.70±0.60</td><td>74.60±1.03</td><td>74.92±0.88</td><td>63.21</td></tr><tr><td colspan="8">Tiny ImageNet with 20% corrupted images</td></tr><tr><td>Random</td><td>19.99±0.42</td><td>25.93±0.53</td><td>30.83±0.44</td><td>37.98±0.31</td><td>42.96±0.62</td><td>46.68±0.43</td><td>31.54</td></tr><tr><td>Herding</td><td>19.46±0.14</td><td>24.47±0.33</td><td>29.72±0.39</td><td>37.50±0.59</td><td>42.28±0.30</td><td>46.68±0.43</td><td>30.86</td></tr><tr><td>Forgetting</td><td>18.47±0.46</td><td>25.53±0.23</td><td>31.17±0.24</td><td>39.35±0.44</td><td>44.55±0.67</td><td>46.68±0.43</td><td>31.81</td></tr><tr><td>GraNd-score</td><td>20.07±0.49</td><td>26.68±0.40</td><td>31.25±0.40</td><td>38.21±0.49</td><td>42.84±0.72</td><td>46.68±0.43</td><td>30.53</td></tr><tr><td>EL2N-score</td><td>18.57±0.30</td><td>24.42±0.44</td><td>30.04±0.15</td><td>37.62±0.44</td><td>42.43±0.61</td><td>46.68±0.43</td><td>30.53</td></tr><tr><td>Optimization-based</td><td>13.71±0.26</td><td>23.33±1.84</td><td>29.15±2.84</td><td>36.12±1.86</td><td>42.94±0.52</td><td>46.88±0.43</td><td>29.06</td></tr><tr><td>Self-sup.-selection</td><td>20.22±0.23</td><td>26.90±0.50</td><td>31.93±0.49</td><td>39.74±0.52</td><td>44.27±0.10</td><td>46.68±0.43</td><td>32.61</td></tr><tr><td>Moderate-DS</td><td>23.27±0.33</td><td>29.06±0.36</td><td>33.48±0.11</td><td>40.07±0.36</td><td>44.73±0.39</td><td>46.68±0.43</td><td>34.12</td></tr><tr><td>GM Matching</td><td>27.19±0.92</td><td>31.70±0.78</td><td>35.14±0.19</td><td>42.04±0.31</td><td>45.12±0.28</td><td>46.68±0.43</td><td>36.24</td></tr></table>

deep nets including ResNet-18/50 (He et al., 2016), VGG-16 (Simonyan & Zisserman, 2014), ShuffleNet (Ma et al., 2018), SENet (Hu et al., 2018), EfficientNet-B0(Tan & Le, 2019).

**IMPLEMENTATION DETAILS:** For the CIFAR-10/100 experiments, we utilize a batch size of 128 and employ SGD optimizer with a momentum of 0.9, weight decay of 5e-4, and an initial learning rate of 0.1. The learning rate is reduced by a factor of 5 after the 60th, 120th, and 160th epochs, with a total of 200 epochs. Data augmentation techniques include random cropping and random horizontal flipping. In the Tiny-ImageNet experiments, a batch size of 256 is used with an SGD optimizer, momentum of 0.9, weight decay of 1e-4, and an initial learning rate of 0.1. The learning rate is decreased by a factor of 10 after the 30th and 60th epochs, with a total of 90 epochs. Random horizontal flips are applied for data augmentation. Each experiment is repeated over 5 random seeds and the variances are noted. Throughout this paper, we use Weiszfeld Solver (Weiszfeld, 1937) to compute GM approximately.

Table 3: Robustness to Label Noise: Comparing (Test Accuracy) pruning methods on CIFAR-100 and TinyImageNet datasets, under  $20\%$  and  $35\%$  Symmetric Label Corruption, at  $20\%$  and  $30\%$  selection ratio. ResNet-50 is used both as proxy and for downstream classification.  

<table><tr><td rowspan="2">Method / Ratio</td><td colspan="2">CIFAR-100 (Label noise)</td><td colspan="2">Tiny ImageNet (Label noise)</td><td rowspan="2">Mean ↑</td></tr><tr><td>20%</td><td>30%</td><td>20%</td><td>30%</td></tr><tr><td colspan="6">20% Label Noise</td></tr><tr><td>Random</td><td>34.47±0.64</td><td>43.26±1.21</td><td>17.78±0.44</td><td>23.88±0.42</td><td>29.85</td></tr><tr><td>Herding</td><td>42.29±1.75</td><td>50.52±3.38</td><td>18.98±0.44</td><td>24.23±0.29</td><td>34.01</td></tr><tr><td>Forgetting</td><td>36.53±1.11</td><td>45.78±1.04</td><td>13.20±0.38</td><td>21.79±0.43</td><td>29.33</td></tr><tr><td>GraNd-score</td><td>31.72±0.67</td><td>42.80±0.30</td><td>18.28±0.32</td><td>23.72±0.18</td><td>28.05</td></tr><tr><td>EL2N-score</td><td>29.82±1.19</td><td>33.62±2.35</td><td>13.93±0.69</td><td>18.57±0.31</td><td>23.99</td></tr><tr><td>Optimization-based</td><td>32.79±0.62</td><td>41.80±1.14</td><td>14.77±0.95</td><td>22.52±0.77</td><td>27.57</td></tr><tr><td>Self-sup.-selection</td><td>31.08±0.78</td><td>41.87±0.63</td><td>15.10±0.73</td><td>21.01±0.36</td><td>27.27</td></tr><tr><td>Moderate-DS</td><td>40.25±0.12</td><td>48.53±1.60</td><td>19.64±0.40</td><td>24.96±0.30</td><td>31.33</td></tr><tr><td>GM Matching</td><td>52.64±0.72</td><td>61.01±0.47</td><td>25.80±0.37</td><td>31.71±0.24</td><td>42.79</td></tr><tr><td colspan="6">35% Label Noise</td></tr><tr><td>Random</td><td>24.51±1.34</td><td>32.26±0.81</td><td>14.64±0.29</td><td>19.41±0.45</td><td>22.71</td></tr><tr><td>Herding</td><td>29.42±1.54</td><td>37.50±2.12</td><td>15.14±0.45</td><td>20.19±0.45</td><td>25.56</td></tr><tr><td>Forgetting</td><td>29.48±1.98</td><td>38.01±2.21</td><td>11.25±0.90</td><td>17.07±0.66</td><td>23.14</td></tr><tr><td>GraNd-score</td><td>23.03±1.05</td><td>34.83±2.01</td><td>13.68±0.46</td><td>19.51±0.45</td><td>22.76</td></tr><tr><td>EL2N-score</td><td>21.95±1.08</td><td>31.63±2.84</td><td>10.11±0.25</td><td>13.69±0.32</td><td>19.39</td></tr><tr><td>Optimization-based</td><td>26.77±0.15</td><td>35.63±0.92</td><td>12.37±0.68</td><td>18.52±0.90</td><td>23.32</td></tr><tr><td>Self-sup.-selection</td><td>23.12±1.47</td><td>34.85±0.68</td><td>11.23±0.32</td><td>17.76±0.69</td><td>22.64</td></tr><tr><td>Moderate-DS</td><td>28.45±0.53</td><td>36.55±1.26</td><td>15.27±0.31</td><td>20.33±0.28</td><td>25.15</td></tr><tr><td>GM Matching</td><td>43.33±1.02</td><td>58.41±0.68</td><td>23.14±0.92</td><td>27.76±0.40</td><td>38.16</td></tr></table>

PROXY MODEL: Needless to say, identifying sample importance is an ill-posed problem without some notion of similarity among the samples. Thus, it is common to assume access to a proxy encoder that maps the features to a separable embedding space - a property often satisfied by off-the-shelf pretrained foundation models (Hessel et al., 2021; Sorscher et al., 2022). We perform experiments across multiple choices of such proxy encoder scenarios: (A) Standard Setting: when the proxy model shares the same architecture as the model Table 1-4). Additionally, we also experiment with (B) Distribution Shift: proxy model pretrained on a different (distribution shifted) dataset( Figure 2-3) e.g. ImageNet and used to sample from CIFAR10. (C) Network Transfer: where, the proxy has a different network compared to the downstream classifier (Table 5).

# IDEAL (NO CORRUPTION) SCENARIO

Our first sets of experiments involve performing data pruning across selection ratio ranging from  $20\%$  -  $80\%$  in the uncorrupted setting. The corresponding results, presented in Table 1, indicate that while GM Matching is developed with robustness scenarios in mind, it outperforms the existing strong baselines even in the clean setting. Overall, on both CIFAR-100 and Tiny ImageNet GM Matching improves over the prior methods  $>2\%$  on an average. In particular, we note that GM Matching enjoys larger gains in the low data selection regime, while staying competitive at low pruning rates.

# CORRUPTION SCENARIOS

To understand the performance of data pruning strategies in presence of corruption, we experiment with three different sources of corruption – image corruption, label noise and adversarial attacks.

ROBUSTNESS TO IMAGE CORRUCTION: In this set of experiments, we investigate the robustness of data pruning strategies when the input images are corrupted – a popular robustness setting, often encountered when training models on real-world data (Hendrycks & Dietterich, 2019; Szegedy et al., 2013). To corrupt images, we apply five types of realistic noise: Gaussian noise, random occlusion, resolution reduction, fog, and motion blur to parts of the corrupt samples i.e. to say if  $m$  samples are corrupted, each type of noise is added to one a random  $m/5$  of them, while the other partitions are corrupted with a different noise. The results are presented in Table 2. We observe that GM Matching outperforms all the baselines across all pruning rates improving  $\approx 3\%$  across both datasets on an average. We note that, the gains are more consistent and profound in this setting over the clean setting. Additionally, similar to our prior observations in the clean setting, the gains of GM Matching are more significant at high pruning rates.

Table 4: Robustness to Adversarial Attacks. Comparing (Test Accuracy) pruning methods under PGD and GS attacks. ResNet-50 is used both as proxy and for downstream classification.  

<table><tr><td rowspan="2">Method / Ratio</td><td colspan="2">CIFAR-100 (PGD Attack)</td><td colspan="2">CIFAR-100 (GS Attack)</td><td rowspan="2">Mean ↑</td></tr><tr><td>20%</td><td>30%</td><td>20%</td><td>30%</td></tr><tr><td>Random</td><td>43.23±0.31</td><td>52.86±0.34</td><td>44.23±0.41</td><td>53.44±0.44</td><td>48.44</td></tr><tr><td>Herding</td><td>40.21±0.72</td><td>49.62±0.65</td><td>39.92±1.03</td><td>50.14±0.15</td><td>44.97</td></tr><tr><td>Forgetting</td><td>35.90±1.30</td><td>47.37±0.99</td><td>37.55±0.53</td><td>46.88±1.91</td><td>41.93</td></tr><tr><td>GraNd-score</td><td>40.87±0.84</td><td>50.13±0.30</td><td>40.77±1.11</td><td>49.88±0.83</td><td>45.41</td></tr><tr><td>EL2N-score</td><td>26.61±0.58</td><td>34.50±1.02</td><td>26.72±0.66</td><td>35.55±1.30</td><td>30.85</td></tr><tr><td>Optimization-based</td><td>38.29±1.77</td><td>46.25±1.82</td><td>41.36±0.92</td><td>49.10±0.81</td><td>43.75</td></tr><tr><td>Self-sup.-selection</td><td>40.53±1.15</td><td>49.95±0.50</td><td>40.74±1.66</td><td>51.23±0.25</td><td>45.61</td></tr><tr><td>Moderate-DS</td><td>43.60±0.97</td><td>51.66±0.39</td><td>44.69±0.68</td><td>53.71±0.37</td><td>48.42</td></tr><tr><td>GM Matching</td><td>45.41 ± 0.86</td><td>51.80 ± 1.01</td><td>49.78 ± 0.27</td><td>55.50 ± 0.31</td><td>50.62</td></tr><tr><td></td><td colspan="2">Tiny ImageNet (PGD Attack)</td><td colspan="2">Tiny ImageNet (GS Attack)</td><td></td></tr><tr><td>Method / Ratio</td><td>20%</td><td>30%</td><td>20%</td><td>30%</td><td>Mean ↑</td></tr><tr><td>Random</td><td>20.93±0.30</td><td>26.60±0.98</td><td>22.43±0.31</td><td>26.89±0.31</td><td>24.21</td></tr><tr><td>Herding</td><td>21.61±0.36</td><td>25.95±0.19</td><td>23.04±0.28</td><td>27.39±0.14</td><td>24.50</td></tr><tr><td>Forgetting</td><td>20.38±0.47</td><td>26.12±0.19</td><td>22.06±0.31</td><td>27.21±0.21</td><td>23.94</td></tr><tr><td>GraNd-score</td><td>20.76±0.21</td><td>26.34±0.32</td><td>22.56±0.30</td><td>27.52±0.40</td><td>24.30</td></tr><tr><td>EL2N-score</td><td>16.67±0.62</td><td>22.36±0.42</td><td>19.93±0.57</td><td>24.65±0.32</td><td>20.93</td></tr><tr><td>Optimization-based</td><td>19.26±0.77</td><td>24.55±0.92</td><td>21.26±0.24</td><td>25.88±0.37</td><td>22.74</td></tr><tr><td>Self-sup.-selection</td><td>19.23±0.46</td><td>23.92±0.51</td><td>19.70±0.20</td><td>24.73±0.39</td><td>21.90</td></tr><tr><td>Moderate-DS</td><td>21.81±0.37</td><td>27.11±0.20</td><td>23.20±0.13</td><td>28.89±0.27</td><td>25.25</td></tr><tr><td>GM Matching</td><td>25.98 ± 1.12</td><td>30.77 ± 0.25</td><td>29.71 ± 0.45</td><td>32.88 ± 0.73</td><td>29.84</td></tr></table>

Table 5: Network Transfer (Clean): Tiny-ImageNet Model Transfer Results. A ResNet-50 proxy is used to find important samples which are then used to train SENet and EfficientNet.  

<table><tr><td rowspan="2">Method / Ratio</td><td colspan="2">ResNet-50→SENet</td><td colspan="2">ResNet-50→EfficientNet-B0</td><td rowspan="2">Mean ↑</td></tr><tr><td>20%</td><td>30%</td><td>20%</td><td>30%</td></tr><tr><td>Random</td><td>34.13±0.71</td><td>39.57±0.53</td><td>32.88±1.52</td><td>39.11±0.94</td><td>36.42</td></tr><tr><td>Hering</td><td>34.86±0.55</td><td>38.60±0.68</td><td>32.21±1.54</td><td>37.53±0.22</td><td>35.80</td></tr><tr><td>Forgetting</td><td>33.40±0.64</td><td>39.79±0.78</td><td>31.12±0.21</td><td>38.38±0.65</td><td>35.67</td></tr><tr><td>GraNd-score</td><td>35.12±0.54</td><td>41.14±0.42</td><td>33.20±0.67</td><td>40.02±0.35</td><td>37.37</td></tr><tr><td>EL2N-score</td><td>31.08±1.11</td><td>38.26±0.45</td><td>31.34±0.49</td><td>36.88±0.32</td><td>34.39</td></tr><tr><td>Optimization-based</td><td>33.18±0.52</td><td>39.42±0.77</td><td>32.16±0.90</td><td>38.52±0.50</td><td>35.82</td></tr><tr><td>Self-sup.-selection</td><td>31.74±0.71</td><td>38.45±0.39</td><td>30.99±1.03</td><td>37.96±0.77</td><td>34.79</td></tr><tr><td>Moderate-DS</td><td>36.04±0.15</td><td>41.40±0.20</td><td>34.26±0.48</td><td>39.57±0.29</td><td>37.82</td></tr><tr><td>GM Matching</td><td>37.93±0.23</td><td>42.59±0.29</td><td>36.31±0.67</td><td>41.03±0.41</td><td>39.47</td></tr></table>

ROBUSTNESS TO LABEL CORRUCTION: Next, we consider another important corruption scenario where a fraction of the training examples are mislabeled. We conduct experiments with synthetically injected symmetric label noise (Li et al., 2022; Patrini et al., 2017; Xia et al., 2020). The results are summarized in Table 3. Encouragingly, GM Matching outperforms the baselines by  $\approx$ $12\%$ . Since, mislabeled samples come from different class - they tend to be spatially quite dissimilar, being less likely to be picked by GM matching, explaining the superior performance.

ROBUSTNESS TO ADVERSARIAL ATTACKS: Finally, we experiment with adversarial attacks that add imperceptible but adversarial noise on natural examples (Szegedy et al., 2013; Huang et al., 2010). Specifically, we employ two popular adversarial attack algorithms - PGD attack (Madry et al., 2017) and GS Attacks (Goodfellow et al., 2014) on models trained with CIFAR-100 and Tiny-ImageNet to generate adversarial examples. Following this, various pruning methods are applied to these adversarial examples, and the models are retrained on the curated subset of data. The results are summarized in Table 4. Similar to other corruption scenarios, even in this setting, GM MATCHING outperforms the baselines yielding  $\approx 3\%$  average gain over the best performing baseline.

# GENERALIZATION TO UNSEEN NETWORK / DOMAIN

Since, the input features (e.g. images) often reside on a non-separable manifold, data pruning strategies rely on a proxy model to map the samples into a separable manifold (embedding space), wherein the data pruning strategies can now assign importance scores. However, it is important for the data pruning strategies to be robust to architecture changes i.e. to say that samples selected via a

![](images/eb53d983a17927fa4bdcf3bcd7368996ffb0864f166254754849fb63bd591b45.jpg)  
(a) NO CORRUPTION

![](images/3e84a2bac7de624dd0ff8c65e614a90c5fb669052f4d81b4d55ed70f63b587fa.jpg)  
(b)  $20\%$  SYMM LABEL NOISE

![](images/5f976776ece75e05e23b5729dda2f07022eec7e407bf306a820d6d93853706dd.jpg)  
(c)  $40\%$  SYMM LABEL NOISE

![](images/418fe54575ee7b9d3029f6397f70a80a279aa40086bd2f6e11a3007d9c306681.jpg)  
Figure 2: Domain Transfer (ImageNet-1k  $\rightarrow$  CIFAR-10) Proxy : CIFAR10, corrupted with label noise is pruned using a (proxy) ResNet-18 pretrained on ImageNet-1k. A ResNet-18 is trained from scratch on the subset. We compare our method GM MATCHING with geometric pruning baselines: UNIFORM, EASY,HARD, MODERATE, HERDING.  
(a) NO CORRUPTION  
Figure 3: Domain Transfer (ImageNet-1k  $\rightarrow$  CIFAR-10) Proxy + Embedding : We train a Linear Classifier on CIFAR10; over embeddings obtained from a frozen ResNet-18 pretrained on ImageNet-1k. The dataset was pruned using the same encoder. We compare our method GM MATCHING with geometric pruning baselines: UNIFORM, EASY,HARD, MODERATE, HERDING across different label noise settings.

![](images/fbf1ec57d5f82df234dbbb133a8f6e753469c696b51880fa85ccd9a4eee6fe7c.jpg)  
(b)  $20\%$  SYMM LABEL NOISE

![](images/b5bc3041f3bb5509ba28fa5c6452d67d69a3324aa46f1b9101479b1d27dfd34d.jpg)  
(c)  $40\%$  SYMM LABEL NOISE

proxy network should generalize well when trained on unseen (during sample selection) networks / domains. We perform experiments on two such scenarios:

NETWORK TRANSFER: In this setting, the proxy model is trained on the target dataset (no distribution shift). However, the proxy architecture is different than the downstream network. In Table 5, we use a ResNet-50 proxy trained on Mini-ImageNet to sample the data. However, then we train a downstream SENet and EfficientNet-B0 on the sampled data.

DOMAIN TRANSFER: Next, we consider the setting where the proxy shares the same architecture with the downstream model. However, the proxy used to select the samples is pretrained on a different dataset (distribution shift) than target dataset. In Figure 2 we use a proxy ResNet-18 pretrained on ImageNet to select samples from CIFAR10. The selected samples are used to train a subsequent ResNet-18. In Figure 3, we additionally freeze the pretrained encoder i.e. we use ResNet-18 encoder pretrained on ImageNet as proxy. Further, we freeze the encoder and train a downstream linear classifier on top over CIFAR-10.

# 6 CONCLUSION

In this work, we formalized the problem of robust data pruning. We show that existing data pruning strategies suffer significant degradation in performance in presence of corruption. Orthogonal to existing works, we propose GM MATCHING where our goal is to find a  $k$ -subset from the noisy data such that the mean of the subset approximates the GM of the noisy dataset. We solve this meta problem using a herding style greedy approach. We theoretically justify our approach and empirically show its efficacy by comparing it against several popular benchmarks across multiple datasets. Our results indicate that GM MATCHING consistently outperforms existing pruning strategies in both clean and noisy settings making it a lucrative tool for data pruning in the wild.

# 7 REPRODUCIBILITY STATEMENT

We provide the source code implementation of the proposed algorithm as well as a notebook with a running demo on Synthetic Gaussian Dataset. The hyper-parameters and other training details to reproduce our benchmarks are provided in Section 5. Several benchmarks for existing methods were borrowed directly from prior work, in such cases the source has been appropriately cited e.g. (Xia et al., 2022). All the proofs have been stated clearly in Appendix with necessary assumptions.

# REFERENCES

Anish Acharya, Abolfazl Hashemi, Prateek Jain, Sujay Sanghavi, Inderjit S. Dhillon, and Ufuk Topcu. Robust training in high dimensions via block coordinate geometric median descent. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera (eds.), Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pp. 11145-11168. PMLR, 28-30 Mar 2022. URL https://proceedings.mlr.press/v151/acharya22a.html.  
Pankaj K Agarwal, Sariel Har-Peled, Kasturi R Varadarajan, et al. Geometric approximation via coresets. Combinatorial and computational geometry, 52(1), 2005.  
Chanderjit Bajaj. The algebraic degree of geometric optimization problems. Discrete & Computational Geometry, 3:177-191, 1988.  
Maria-Florina Balcan, Andrei Broder, and Tong Zhang. Margin based active learning. In International Conference on Computational Learning Theory, pp. 35-50. Springer, 2007.  
Stephen Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.  
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.  
Trevor Campbell and Tamara Broderick. Bayesian coreset construction via greedy iterative geodesic ascent. In International Conference on Machine Learning, pp. 698-706. PMLR, 2018.  
Yudong Chen, Lili Su, and Jiaming Xu. Distributed statistical machine learning in adversarial settings: Byzantine gradient descent. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 1(2):1-25, 2017.  
Yutian Chen, Max Welling, and Alex Smola. Super-samples from kernel herding. In Proceedings of the Twenty-Sixth Conference on Uncertainty in Artificial Intelligence, pp. 109-116, 2010.  
Michael B Cohen, Yin Tat Lee, Gary Miller, Jakub Pachocki, and Aaron Sidford. Geometric median in nearly linear time. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing, pp. 9-21, 2016.  
Ilias Diakonikolas, Gautam Kamath, Daniel Kane, Jerry Li, Ankur Moitra, and Alistair Stewart. Robust estimators in high-dimensions without the computational intractability. SIAM Journal on Computing, 48(2):742-864, 2019.  
David L Donoho and Peter J Huber. The notion of breakdown point. A festschrift for Erich L. Lehmann, 157184, 1983.  
Raaz Dwivedi and Lester Mackey. Generalized kernel thinning. arXiv preprint arXiv:2110.01593, 2021.  
Uriel Feige. A threshold of  $\ln n$  for approximating set cover. Journal of the ACM (JACM), 45(4): 634-652, 1998.  
Dan Feldman. Core-sets: Updated survey. In *Sampling techniques for supervised or unsupervised tasks*, pp. 23–44. Springer, 2020.  
Dan Feldman and Michael Langberg. A unified framework for approximating and clustering data. In Proceedings of the forty-third annual ACM symposium on Theory of computing, pp. 569-578, 2011.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Sariel Har-Peled. Geometric approximation algorithms. Number 173. American Mathematical Soc., 2011.

Sariel Har-Peled, Dan Roth, and Dav A Zimak. Maximum margin coresets for active and noise tolerant learning. 2006.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019.  
Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.  
Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Patwary, Mostofa Ali, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.  
Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7132-7141, 2018.  
Sheng-Jun Huang, Rong Jin, and Zhi-Hua Zhou. Active learning by querying informative and representative examples. Advances in neural information processing systems, 23, 2010.  
Peter J Huber. Robust estimation of a location parameter. In *Breakthroughs in statistics*, pp. 492-518. Springer, 1992.  
Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In International conference on machine learning, pp. 2304–2313. PMLR, 2018.  
Ajay J Joshi, Fatih Porikli, and Nikolaos Papanikolopoulos. Multi-class active learning for image classification. In 2009 IEEE conference on computer vision and pattern recognition, pp. 2372-2379. IEEE, 2009.  
Siddharth Joshi and Baharan Mirzasoleiman. Data-efficient contrastive self-supervised learning: Most beneficial examples for supervised learning contribute the least. In International conference on machine learning, pp. 15356-15370. PMLR, 2023.  
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.  
Angelos Katharopoulos and François Fleuret. Not all samples are created equal: Deep learning with importance sampling. In International conference on machine learning, pp. 2525-2534. PMLR, 2018.  
JHB Kemperman. The median of a finite measure on a banach space. Statistical data analysis based on the L1-norm and related methods (Neuchâtel, 1987), pp. 217-230, 1987.  
Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International conference on machine learning, pp. 1885-1894. PMLR, 2017.  
LESLIE Lamport, ROBERT SHOSTAK, and MARSHALL PEASE. The byzantine generals problem. ACM Transactions on Programming Languages and Systems, 4(3):382-401, 1982.  
Liping Li, Wei Xu, Tianyi Chen, Georgios B Giannakis, and Qing Ling. Rsa: Byzantine-robust stochastic aggregation methods for distributed learning from heterogeneous datasets. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 1544–1551, 2019.  
Shikun Li, Xiaobo Xia, Shiming Ge, and Tongliang Liu. Selective-supervised contrastive learning with noisy labels. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 316-325, 2022.  
Hendrik P Lopuhaa, Peter J Rousseeuw, et al. Breakdown points of affine equivariant estimators of multivariate location and covariance matrices. The Annals of Statistics, 19(1):229-248, 1991.

Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. Shufflenet v2: Practical guidelines for efficient cnn architecture design. In Proceedings of the European conference on computer vision (ECCV), pp. 116-131, 2018.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.  
Stanislav Minsker et al. Geometric median and robust estimation in banach spaces. Bernoulli, 21(4): 2308-2335, 2015.  
Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models. In International Conference on Machine Learning, pp. 6950-6960. PMLR, 2020.  
Shanmugavelayutham Muthukrishnan et al. Data streams: Algorithms and applications. Foundations and Trends® in Theoretical Computer Science, 1(2):117-236, 2005.  
Deanna Needell, Rachel Ward, and Nati Srebro. Stochastic gradient descent, weighted sampling, and the randomized kaczmarz algorithm. Advances in neural information processing systems, 27, 2014.  
George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. An analysis of approximations for maximizing submodular set functions—i. Mathematical programming, 14:265–294, 1978.  
Dongmin Park, Seola Choi, Doyoung Kim, Hwanjun Song, and Jae-Gil Lee. Robust data pruning under label noise via maximizing re-labeling accuracy. Advances in Neural Information Processing Systems, 36, 2024.  
Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: A loss correction approach. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1944-1952, 2017.  
Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training. Advances in Neural Information Processing Systems, 34: 20596-20607, 2021.  
Geoff Pleiss, Tianyi Zhang, Ethan Elenberg, and Kilian Q Weinberger. Identifying mislabeled data using the area under the margin ranking. Advances in Neural Information Processing Systems, 33: 17044-17056, 2020.  
Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748-8763. PMLR, 2021.  
Peter J Rousseeuw and Annick M Leroy. Robust regression and outlier detection, volume 589. John wiley & sons, 2005.  
Vatsal Shah, Xiaoxia Wu, and Sujay Sanghavi. Choosing the sample with lowest loss makes sgd robust. In International Conference on Artificial Intelligence and Statistics, pp. 2120-2130. PMLR, 2020.  
Yanyao Shen and Sujay Sanghavi. Learning with bad training data via iterative trimmed loss minimization. In International Conference on Machine Learning, pp. 5739-5748. PMLR, 2019.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. Advances in Neural Information Processing Systems, 35:19523-19536, 2022.

Masashi Sugiyama and Motoaki Kawanabe. Machine learning in non-stationary environments: Introduction to covariate shift adaptation. MIT press, 2012.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.  
Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pp. 6105-6114. PMLR, 2019.  
Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J Gordon. An empirical study of example forgetting during deep neural network learning. arXiv preprint arXiv:1812.05159, 2018.  
Yehuda Vardi and Cun-Hui Zhang. The multivariate 11-median and associated data depth. Proceedings of the National Academy of Sciences, 97(4):1423-1426, 2000.  
Alfred Weber, Carl Joachim Friedrich, et al. Alfred Weber's theory of the location of industries. The University of Chicago Press, 1929.  
Endre Weiszfeld. Sur le point pour lequel la somme des distances de n points donnés est minimum. Tohoku Mathematical Journal, First Series, 43:355-386, 1937.  
Max Welling. Herding dynamical weights to learn. In Proceedings of the 26th Annual International Conference on Machine Learning, pp. 1121-1128, 2009.  
Max Welling and Yutian Chen. Statistical inference using weak chaos and infinite memory. In Journal of Physics: Conference Series, volume 233, pp. 012005. IOP Publishing, 2010.  
Laurence A Wolsey. An analysis of the greedy algorithm for the submodular set covering problem. Combinatorica, 2(4):385-393, 1982.  
Zhaoxian Wu, Qing Ling, Tianyi Chen, and Georgios B Giannakis. Federated variance-reduced stochastic gradient descent with robustness to byzantine attacks. IEEE Transactions on Signal Processing, 68:4583-4596, 2020.  
Xiaobo Xia, Tongliang Liu, Bo Han, Chen Gong, Nannan Wang, Zongyuan Ge, and Yi Chang. Robust early-learning: Hinding the memorization of noisy labels. In International conference on learning representations, 2020.  
Xiaobo Xia, Jiale Liu, Jun Yu, Xu Shen, Bo Han, and Tongliang Liu. Moderate coreset: A universal method of data selection for real-world data-efficient deep learning. In The Eleventh International Conference on Learning Representations, 2022.  
Yilun Xu, Shengjia Zhao, Jiaming Song, Russell Stewart, and Stefano Ermon. A theory of usable information under computational constraints. arXiv preprint arXiv:2002.10689, 2020.  
Shuo Yang, Zeke Xie, Hanyu Peng, Min Xu, Mingming Sun, and Ping Li. Dataset pruning: Reducing training data by examining generalization influence. arXiv preprint arXiv:2205.09329, 2022.  
Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. Advances in neural information processing systems, 31, 2018.
