# ON THE MINIMAL SUPERVISION FOR TRAINING ANY BINARY CLASSIFIER FROM ONLY UNLABELLED DATA

Anonymous authors

Paper under double-blind review

# ABSTRACT

Empirical risk minimization (ERM), with proper loss function and regularization, is the common practice of supervised classification. In this paper, we study training arbitrary (from linear to deep) binary classifier from only unlabeled (U) data by ERM. We prove that it is impossible to estimate the risk of an arbitrary binary classifier in an unbiased manner given a single set of U data, but it becomes possible given two sets of U data with different class priors. These two facts answer a fundamental question—what the minimal supervision is for training any binary classifier from only U data. Following these findings, we propose an ERM-based learning method from two sets of U data, and then prove it is consistent. Experiments demonstrate the proposed method could train deep models and outperform state-of-the-art methods for learning from two sets of U data.

# 1 INTRODUCTION

With some properly chosen loss function (e.g., Bartlett et al., 2006; Tewari & Bartlett, 2007; Reid & Williamson, 2010) and regularization (e.g., Tikhonov, 1943; Srivastava et al., 2014), empirical risk minimization (ERM) is the common practice of supervised classification (Vapnik, 1998). Actually, ERM is used in not only supervised learning but also weakly-supervised learning. For example, in semi-supervised learning (Chapelle et al., 2006), we have very limited labeled (L) data and a lot of unlabeled (U) data, where L data share the same form with supervised learning. Thus, it is easy to estimate the risk from L data, and ERM can be done with regularization based on U data (including but not limited to Grandvalet & Bengio, 2004; Belkin et al., 2006; Mann & McCallum, 2007; Niu et al., 2013; Miyato et al., 2016; Laine & Aila, 2017; Tarvainen & Valpola, 2017; Luo et al., 2018).

Nevertheless, L data may differ from supervised learning in not only the amount but also the form. For instance, in positive-unlabeled learning (Elkan & Noto, 2008; Ward et al., 2009), all L data are from the positive class, and due to the lack of L data from the negative class it becomes impossible to estimate the risk from only L data. To this end, a two-step approach to ERM has been considered (du Plessis et al., 2014; 2015; Niu et al., 2016; Kiryo et al., 2017). Firstly, the risk is rewritten into an equivalent expression, such that it just involves the same distributions from which L and U data are sampled—this step leads to certain risk estimators. Secondly, the risk is estimated from both L and U data, and the resulted empirical training risk is minimized (e.g. by Robbins & Monro, 1951; Kingma & Ba, 2015). In this two-step approach, U data are used for risk evaluation that is the core of ERM and is mandatory in ERM, and hence risk rewrite enables ERM and is the key of success.

One step further from positive-unlabeled learning is learning from only  $U$  data without any  $L$  data. This is significantly harder than previous learning problems (cf. Figure 1). However, we would still like to train arbitrary binary classifier, in particular, deep networks (Goodfellow et al., 2016). As a consequence, we prefer ERM to clustering methods (e.g., Xu et al., 2004; Gomes et al., 2010). The critical point is how to estimate the risk from only  $U$  data, and our solution is again ERM-enabling risk rewrite in the aforementioned two-step approach. The first step should lead to an unbiased risk estimator that will be used in the second step. Subsequently, we can evaluate the empirical training and/or validation risk by plugging only  $U$  training/validation data into the risk estimator. Thus, this two-step ERM needs no  $L$  validation data for hyperparameter tuning, which is a huge advantage in training deep models nowadays. Note that given only  $U$  data, by no means could we learn the class priors (Menon et al., 2015), so that we assume all necessary class priors are also given. This is the

![](images/5f619610fdf9aaba50f9c7205fd2f64ca1e5506878201c9832481c4a8de939c8.jpg)  
(a) P component

![](images/5935752afdec8cfa004a8645968b1fbb6daa4b23296bba771e8bcd5f1ec73b91.jpg)  
(b) N component

![](images/221873a3a1170b5a843250aa9f15c4d798faf1f3f793b20a6b18df49a7a28b64.jpg)  
(c) U set1

![](images/dac329f97a27eb09eb62ab3e582a269eed9fc2abd70bd8cd45e542286f394dc4.jpg)  
(d) U set2

![](images/56b692e2a910d8feca391b459f77e356d20b18fa39161ac4d180e35bf991c3c9.jpg)

In the left panel, (a) and (b) show positive (P) and negative (N) components of the Gaussian mixture; (c) and (d) show two distributions (with class priors 0.9 and 0.4) where U training data are drawn (marked as black points). The right panel shows the test distribution (with class prior 0.3) and data (marked as blue for P and red for N), as well as four learned classifiers. In the legend, "CCN" refers to Natarajan et al. (2013), "UU-biased" means supervised learning taking larger-/smaller-class-prior U data as P/N data, "UU" is the proposed method, and "Oracle" means supervised learning from the same amount of L data. See Appendix B for more information. We can see that the UU classifier is almost identical to the Oracle classifier and much better than the other two.

Figure 1: Illustrative example of classification from a Gaussian mixture dataset.

unique type of supervision we will leverage throughout this paper, which implies that the problem of interest still belongs to weakly-supervised learning rather than unsupervised learning.

In this paper, we raise a fundamental question in weakly-supervised learning—how many sets of U data with different class priors are necessary for rewriting the risk? Our answer has two aspects:

- Risk rewrite is impossible given a single set of U data (see Theorem 2 in Sec. 3);  
- Risk rewrite becomes possible given two sets of U data (see Theorem 4 in Sec. 4).

This suggests that three class priors<sup>1</sup> are all you need to train deep models from only U data, while any two<sup>2</sup> should not be enough. The impossibility is a proof by contradiction, and the possibility is a proof by construction, following which we explicitly design an unbiased risk estimator. Therefore, with the help of this risk estimator, we propose an ERM-based learning method from two sets of U data. Thanks to the unbiasedness of our risk estimator, we derive an estimation error bound which certainly guarantees the consistency of learning (Mohri et al., 2012; Shalev-Shwartz & Ben-David, 2014).<sup>3</sup> Experiments demonstrate that the proposed method could train multilayer perceptron, AllConvNet (Springenberg et al., 2015) and ResNet (He et al., 2016) from two sets of U data; it could outperform state-of-the-art methods for learning from two sets of U data. See Figure 1 for how the proposed method works on a Gaussian mixture of two components.

# 2 PROBLEM SETTING AND RELATED WORK

Consider the binary classification problem. Let  $X$  and  $Y$  be the input and output random variables such that  $p(x, y)$  is the underlying joint density,  $p_{\mathrm{p}}(x) = p(x \mid Y = +1)$  and  $p_{\mathrm{n}}(x) = p(x \mid Y = -1)$  are the P and N class-conditional densities,  $p(x)$  is the marginal density, and  $\pi_{\mathrm{p}} = p(Y = +1)$  is the class-prior probability.

Data generation process Let  $\theta$  and  $\theta^{\prime}$  be two class priors such that  $\theta \neq \theta^{\prime}$ , and let

$$
p _ {\mathrm {t r}} (x) = \theta p _ {\mathrm {p}} (x) + (1 - \theta) p _ {\mathrm {n}} (x), \quad p _ {\mathrm {t r}} ^ {\prime} (x) = \theta^ {\prime} p _ {\mathrm {p}} (x) + (1 - \theta^ {\prime}) p _ {\mathrm {n}} (x) \tag {1}
$$

be the marginal densities from which U training data are drawn. Eq. (1) implies there are  $p_{\mathrm{tr}}(x,y)$  and  $p_{\mathrm{tr}}'(x,y)$ , whose class-conditional densities are same and equal to those of  $p(x,y)$ , and whose class priors are different, i.e.,

$$
p _ {\mathrm {t r}} (x \mid y) = p _ {\mathrm {t r}} ^ {\prime} (x \mid y) = p (x \mid y), \quad p _ {\mathrm {t r}} (Y = + 1) = \theta \neq \theta^ {\prime} = p _ {\mathrm {t r}} ^ {\prime} (Y = + 1).
$$

If we could sample L data from  $p_{\mathrm{tr}}(x,y)$  or  $p_{\mathrm{tr}}'(x,y)$ , it would reduce to supervised learning under class-prior change (Quinonero-Candela et al., 2009).

Nonetheless, the problem of interest belongs to weakly-supervised learning—U training (and validation) data are supposed to be drawn according to (1). More specifically, we have

$$
\mathcal {X} _ {\mathrm {t r}} = \left\{x _ {1}, \dots , x _ {n} \right\} \sim p _ {\mathrm {t r}} (x), \quad \mathcal {X} _ {\mathrm {t r}} ^ {\prime} = \left\{x _ {1} ^ {\prime}, \dots , x _ {n ^ {\prime}} ^ {\prime} \right\} \sim p _ {\mathrm {t r}} ^ {\prime} (x), \tag {2}
$$

where  $n$  and  $n'$  are two natural numbers as the sample sizes of  $\mathcal{X}_{\mathrm{tr}}$  and  $\mathcal{X}_{\mathrm{tr}}'$ . This is exactly same as du Plessis et al. (2013) and Menon et al. (2015) with some different names. In Menon et al. (2015),  $\theta$  and  $\theta'$  are called corruption parameters, and if we assume  $\theta > \theta'$ ,  $p_{\mathrm{tr}}(x)$  is called the corrupted  $P$  density and  $p_{\mathrm{tr}}'(x)$  is called the corrupted  $N$  density. Despite the same data generation process in (2), a vital difference between the problem settings is performance measures to be optimized.

Performance measures Let  $g: \mathbb{R}^d \to \mathbb{R}$  be an arbitrary decision function, i.e.,  $g$  may literally be any binary classifier. Let  $\ell: \mathbb{R} \to \mathbb{R}$  be the loss function, such that the value  $\ell(z)$  means the loss by predicting  $g(x)$  when the ground truth is  $y$  where  $z = yg(x)$  is the margin. The risk of  $g$  is

$$
R (g) = \mathbb {E} _ {(X, Y) \sim p (x, y)} [ \ell (Y g (X)) ] = \pi_ {\mathrm {p}} \mathbb {E} _ {\mathrm {p}} [ \ell (g (X)) ] + (1 - \pi_ {\mathrm {p}}) \mathbb {E} _ {\mathrm {n}} [ \ell (- g (X)) ], \tag {3}
$$

where  $\mathbb{E}_{\mathrm{p}}[\cdot ]$  means  $\mathbb{E}_{X\sim p_{\mathrm{p}}}[\cdot ]$  and  $\mathbb{E}_{\mathrm{n}}[\cdot ]$  means  $\mathbb{E}_{X\sim p_{\mathrm{n}}}[\cdot ]$  respectively. If  $\ell$  is the zero-one loss that is defined by  $\ell_{01}(z) = (1 - \mathrm{sign}(z)) / 2$ , the risk is named the classification error that is the standard performance measure in classification problems. A balanced version of (3) is

$$
B (g) = \frac {1}{2} \mathbb {E} _ {\mathrm {p}} [ \ell (g (X)) ] + \frac {1}{2} \mathbb {E} _ {\mathrm {n}} [ \ell (- g (X)) ], \tag {4}
$$

and if  $\ell$  is  $\ell_{01}$ , (4) is named the balanced error (Brodersen et al., 2010). The vital difference is that (3) is chosen in the current paper whereas (4) is chosen in du Plessis et al. (2013) and Menon et al. (2015) as the performance measure to be optimized.

We argue that (3) is more natural than (4) as the performance measure for binary classification. By the phrase "binary classification", we mean  $\pi_{\mathrm{p}}$  is neither very large nor very small. Otherwise, due to extreme values of  $\pi_{\mathrm{p}}$  (i.e., either  $\pi_{\mathrm{p}} \approx 0$  or  $\pi_{\mathrm{p}} \approx 1$ ), the problem under consideration should be retrieval or detection rather than binary classification. Note that  $B(g) \neq R(g)$  unless  $\pi_{\mathrm{p}} = \frac{1}{2}$  since  $g$  is arbitrary, which implies that (4) is misleading so long as (3) is the performance measure.

Related work Learning from only U data is previously regarded as discriminative clustering (Xu et al., 2004; Valizadegan & Jin, 2006; Li et al., 2009; Gomes et al., 2010; Sugiyama et al., 2014; Hu et al., 2017). Their goals are to maximize the margin or the mutual information between  $X$  and  $Y$ . As a result, they rely on the cluster assumption (Chapelle et al., 2002) and the assumption that one class corresponds to exactly one cluster. The second assumption is rarely satisfied in practice.

As mentioned earlier, learning from two sets of U data is already studied in du Plessis et al. (2013) and Menon et al. (2015). Both of them adopt (4) as the performance measure. In the former paper,  $g$  is learned by estimating  $\mathrm{sign}(p_{\mathrm{tr}}(x) - p_{\mathrm{tr}}'(x))$ . In the latter paper,  $g$  is learned by taking noisy L data from  $p_{\mathrm{tr}}(x)$  and  $p_{\mathrm{tr}}'(x)$  as clean L data from  $p_{\mathrm{p}}(x)$  and  $p_{\mathrm{n}}(x)$ , and then its threshold is moved to the correct value by post-processing. In summary, instead of ERM, they evidence the possibility of empirical balanced risk minimization, and no impossibility is proven.

Our findings are compatible with learning from label proportions (Quadrianto et al., 2009; Yu et al., 2013). In Quadrianto et al. (2009), it is proven the minimal number of U sets equals the number of classes. However, their finding only holds for the linear model, the logistic loss, and their proposed method based on mean operators. On the other hand, Yu et al. (2013) is not ERM-based; it is based on discriminative clustering together with expectation regularization (Mann & McCallum, 2007).

At first glance, our data generation process, using the names from Menon et al. (2015), looks quite similar to that of learning with noisy labels (cf. Natarajan et al., 2013). In fact, these two are fairly different, and the differences are reviewed and discussed in Menon et al. (2015) and van Rooyen & Williamson (2018). Along this line of research, just a few papers explore instance-dependent noise models (Menon et al., 2016; Cheng et al., 2017), and the vast majority of papers employ instance-independent noise models (e.g., Natarajan et al., 2013; Sukhbaatar et al., 2015; Menon et al., 2015; Liu & Tao, 2016; Goldberger & Ben-Reuven, 2017; Patrini et al., 2017; Han et al., 2018a) or have no noisy model but many heuristics (e.g., Reed et al., 2015; Jiang et al., 2018; Ren et al., 2018; Han et al., 2018b). There exist two instance-independent noise models: class-conditional noise (CCN)

in Angluin & Laird (1988) and mutually contaminated distributions (MCD) in Scott et al. (2013). Denote by  $\tilde{y}$  and  $\tilde{p}(\cdot)$  the corrupted label and distributions. Then, CCN and MCD are defined by

$$
\left( \begin{array}{c} \tilde {p} (\tilde {Y} = + 1 \mid x) \\ \tilde {p} (\tilde {Y} = - 1 \mid x) \end{array} \right) = T _ {\text {C C N}} \left( \begin{array}{c} p (Y = + 1 \mid x) \\ p (Y = - 1 \mid x) \end{array} \right) \quad \text {a n d} \quad \left( \begin{array}{c} \tilde {p} (x \mid \tilde {Y} = + 1) \\ \tilde {p} (x \mid \tilde {Y} = - 1) \end{array} \right) = T _ {\text {M C D}} \left( \begin{array}{c} p _ {\mathrm {p}} (x) \\ p _ {\mathrm {n}} (x) \end{array} \right),
$$

where both of  $T_{\mathrm{CCN}}$  and  $T_{\mathrm{MCD}}$  are 2-by-2 matrices but  $T_{\mathrm{CCN}}$  is column normalized and  $T_{\mathrm{MCD}}$  is row normalized. It has been proven in Menon et al. (2015) that CCN is a strict special case of MCD. To be clear,  $\tilde{p}(\tilde{y})$  is fixed in CCN once  $\tilde{p}(\tilde{y} \mid x)$  is specified while  $\tilde{p}(\tilde{y})$  is free in MCD after  $\tilde{p}(x \mid \tilde{y})$  is specified. Furthermore,  $\tilde{p}(x) = p(x)$  in CCN but  $\tilde{p}(x) \neq p(x)$  in MCD. Due to this covariate shift, CCN methods do not fit MCD problem setting, though MCD methods fit CCN problem setting. To the best of our knowledge, the proposed method is the first MCD method based on ERM.

# 3 LEARNING FROM ONE SET OF U DATA

From now on, we prove that knowing  $\pi_{\mathrm{p}}$  and  $\theta$  is insufficient for rewriting  $R(g)$ .

# 3.1 A BRIEF REVIEW OF ERM

To begin with, we review ERM (Vapnik, 1998) by imaging that we are given  $\mathcal{X}_{\mathrm{p}} = \{x_1,\ldots ,x_n\} \sim p_{\mathrm{p}}(x)$  and  $\mathcal{X}_{\mathrm{n}} = \{x_1',\dots,x_n'\} \sim p_{\mathrm{n}}(x)$ . Then, we would go through the following procedure:

1. Choose a surrogate loss  $\ell (z)$ , so that  $R(g)$  in Eq. (3) is defined.  
2. Choose a model  $\mathcal{G}$ , so that  $\min_{g\in \mathcal{G}}R(g)$  is achievable by ERM.  
3. Approximate  $R(g)$  by

$$
\widehat {R} _ {\mathrm {p n}} (g) = \frac {\pi_ {\mathrm {p}}}{n} \sum_ {i = 1} ^ {n} \ell \left(g \left(x _ {i}\right)\right) + \frac {1 - \pi_ {\mathrm {p}}}{n ^ {\prime}} \sum_ {j = 1} ^ {n ^ {\prime}} \ell \left(- g \left(x _ {j} ^ {\prime}\right)\right). \tag {5}
$$

4. Minimize  $\widehat{R}_{\mathrm{pn}}(g)$ , with appropriate regularization, by favorite optimization algorithm.

Here,  $\ell$  should be classification-calibrated (Bartlett et al., 2006), in order to guarantee that  $R(g;\ell)$  and  $R(g;\ell_{01})$  have the same minimizer over all measurable functions. This minimizer is the Bayes optimal classifier and denoted by  $g^{**} = \arg \min_g R(g)$ . The Bayes optimal risk  $R(g^{**})$  is usually unachievable by ERM as  $n,n^{\prime}\to \infty$ . That is why by choosing a model  $\mathcal{G}$ ,  $g^{*} = \arg \min_{g\in \mathcal{G}}R(g)$  is changed as the target to which  $\widehat{g}_{\mathrm{pn}} = \arg \min_{g\in \mathcal{G}}\widehat{R}_{\mathrm{pn}}(g)$  converges as  $n,n^{\prime}\to \infty$ . In statistical learning, the approximation error is  $R(g^{*}) - R(g^{**})$ , and the estimation error is  $R(\widehat{g}_{\mathrm{pn}}) - R(g^{*})$ . Learning is consistent if and only if the estimation error converges to zero as  $n,n^{\prime}\to \infty$ .

# 3.2 IMPOSSIBILITY OF RISK REWRITE

Recall that  $R(g)$  is approximated by (5) given  $\mathcal{X}_{\mathrm{p}}$  and  $\mathcal{X}_{\mathrm{n}}$ , which does not work given  $\mathcal{X}_{\mathrm{tr}}$  and  $\mathcal{X}_{\mathrm{tr}}^{\prime}$ . We might rewrite  $R(g)$  such that it can be approximated given  $\mathcal{X}_{\mathrm{tr}}$  and/or  $\mathcal{X}_{\mathrm{tr}}^{\prime}$ . This is known as the backward correction (Natarajan et al., 2013; Patrini et al., 2017) in learning with noisy labels.

Definition 1. We say that  $R(g)$  in (3) is rewritable given  $p_{\mathrm{tr}}$ , if and only if there exist constants  $a$  and  $b$ , such that for any  $g$  it holds that

$$
R (g) = \mathbb {E} _ {p _ {\mathrm {t r}}} [ \bar {\ell} (g (X)) ], \tag {6}
$$

where  $\mathbb{E}_{p_{\mathrm{tr}}}[\cdot ]$  means  $\mathbb{E}_{X\sim p_{\mathrm{tr}}}[\cdot ]$  and  $\bar{\ell} (z) = a\ell (z) + b\ell (-z)$  is the corrected loss function.

Theorem 2. Let  $\ell$  be  $\ell_{01}$ , or any bounded surrogate loss satisfying that

$$
0 \leq \ell (+ \infty) = \lim  _ {z \rightarrow + \infty} \ell (z) <   \lim  _ {z \rightarrow - \infty} \ell (z) = \ell (- \infty) <   + \infty . \tag {7}
$$

Assume  $p_{\mathrm{p}}$  and  $p_{\mathrm{n}}$  are almost surely separable and  $\theta$  is arbitrary. Then,  $R(g)$  is not rewritten.

Theorem 2 shows that under the separability assumption of  $p_{\mathrm{p}}$  and  $p_{\mathrm{n}}$ ,  $R(g)$  is not rewritten. As a consequence, we lack a learning objective, that is, the empirical training risk. It is even worse—we cannot access the empirical validation risk of  $g$  after it is trained by other learning methods such as discriminative clustering. In particular,  $\ell_{01}$  satisfies (7), which implies that the common practice of hyperparameter tuning is disabled by Theorem 2, since U validation data also follow  $p_{\mathrm{tr}}$ .

# 4 LEARNING FROM TWO SETS OF U DATA

From now on, we prove that knowing  $\pi_{\mathrm{p}}$ ,  $\theta$  and  $\theta'$  is sufficient for rewriting  $R(g)$ .

# 4.1 POSSIBILITY OF RISK REWRITE, AND UNBIASED RISK ESTIMATORS

We have proven that  $R(g)$  is not rewritten given  $p_{\mathrm{tr}}$ , and Quadrianto et al. (2009) has proven that  $R(g)$  can be estimated from  $\mathcal{X}_{\mathrm{tr}}$  and  $\mathcal{X}_{\mathrm{tr}}^{\prime}$ , where  $g$  is a linear model and  $\ell$  is the logistic loss. These facts motivate us to investigate the possibility of rewriting  $R(g)$ , where  $g$  and  $\ell$  are both arbitrary. $^6$

Definition 3. We say that  $R(g)$  is rewritten given  $p_{\mathrm{tr}}$  and  $p_{\mathrm{tr}}^{\prime}$ , if and only if there exist constants  $a, b, c$  and  $d$ , such that for any  $g$  it holds that

$$
R (g) = \mathbb {E} _ {p _ {\mathrm {t r}}} [ \bar {\ell} _ {+} (g (X)) ] + \mathbb {E} _ {p _ {\mathrm {t r}} ^ {\prime}} [ \bar {\ell} _ {-} (- g (X)) ], \tag {8}
$$

where  $\bar{\ell}_{+}(z) = a\ell (z) + b\ell (-z)$  and  $\bar{\ell}_{-}(z) = c\ell (z) + d\ell (-z)$  are the corrected loss functions.

Theorem 4. Assume  $\theta$  and  $\theta'$  are arbitrary but satisfy  $\theta > \theta'$ ; otherwise, swap  $p_{\mathrm{tr}}$  and  $p_{\mathrm{tr}}'$  to make sure  $\theta > \theta'$ . Then,  $R(g)$  is rewritten, by letting

$$
a = \frac {(1 - \theta^ {\prime}) \pi_ {\mathrm {p}}}{\theta - \theta^ {\prime}}, \quad b = - \frac {\theta^ {\prime} (1 - \pi_ {\mathrm {p}})}{\theta - \theta^ {\prime}}, \quad c = \frac {\theta (1 - \pi_ {\mathrm {p}})}{\theta - \theta^ {\prime}}, \quad d = - \frac {(1 - \theta) \pi_ {\mathrm {p}}}{\theta - \theta^ {\prime}}. \tag {9}
$$

Theorem (4) immediately leads to an unbiased risk estimator, namely

$$
\begin{array}{l} \widehat {R} _ {\mathrm {u u}} (g) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {(1 - \theta^ {\prime}) \pi_ {\mathrm {p}}}{\theta - \theta^ {\prime}} \ell \left(g \left(x _ {i}\right)\right) - \frac {\theta^ {\prime} \left(1 - \pi_ {\mathrm {p}}\right)}{\theta - \theta^ {\prime}} \ell \left(g \left(- x _ {i}\right)\right)\right) \tag {10} \\ + \frac {1}{n ^ {\prime}} \sum_ {j = 1} ^ {n ^ {\prime}} \left(- \frac {(1 - \theta) \pi_ {\mathrm {p}}}{\theta - \theta^ {\prime}} \ell (g (x _ {j} ^ {\prime})) + \frac {\theta (1 - \pi_ {\mathrm {p}})}{\theta - \theta^ {\prime}} \ell (- g (x _ {j} ^ {\prime}))\right). \\ \end{array}
$$

Eq. (10) is useful for both training (by plugging U training data into it) and hyperparameter tuning (by plugging U validation data into it). We hereafter refer to the process of obtaining the empirical risk minimizer of (10), i.e.,  $\widehat{g}_{\mathrm{uu}} = \arg \min_{g\in \mathcal{G}}\widehat{R}_{\mathrm{uu}}(g)$ , as unlabeled-unlabeled (UU) learning. The proposed UU learning is by nature ERM-based, and consequently  $\widehat{g}_{\mathrm{uu}}$  can be obtained by powerful stochastic optimization algorithms (e.g., Duchi et al., 2011; Kingma & Ba, 2015).

Simplification Note that (10) may require some efforts to implement. Fortunately, it can be simplified by employing  $\ell$  that satisfies a symmetric condition:

$$
\ell (z) + \ell (- z) = 1. \tag {11}
$$

Eq. (11) covers  $\ell_{01}$ , the ramp loss  $\ell_{\mathrm{ramp}}(z) = \max \{0, \min \{1, (1 - z)/2\}\}$  (du Plessis et al., 2014; Niu et al., 2016) and the sigmoid loss  $\ell_{\mathrm{sig}}(z) = 1 / (1 + \exp(z))$  (Kiryo et al., 2017). With the help of (11), we can simplify (10) as

$$
\widehat {R} _ {\mathrm {u u}} ^ {\mathrm {S y m}} (g) = \frac {1}{n} \sum_ {i = 1} ^ {n} \alpha \ell \left(g \left(x _ {i}\right)\right) + \frac {1}{n ^ {\prime}} \sum_ {j = 1} ^ {n ^ {\prime}} \alpha^ {\prime} \ell \left(- g \left(x _ {j} ^ {\prime}\right)\right) - \frac {\theta^ {\prime} \left(1 - \pi_ {\mathrm {p}}\right) + (1 - \theta) \pi_ {\mathrm {p}}}{\theta - \theta^ {\prime}}, \tag {12}
$$

where  $\alpha = (\theta' + \pi_{\mathrm{p}} - 2\theta'\pi_{\mathrm{p}}) / (\theta - \theta')$  and  $\alpha' = (\theta + \pi_{\mathrm{p}} - 2\theta\pi_{\mathrm{p}}) / (\theta - \theta')$ . Just like (10), (12) is an unbiased risk estimator, and it is easy to implement in many deep learning frameworks.

Special cases Consider some special cases of (10) by specifying  $\theta$  and  $\theta'$ . It is obvious that (10) reduces to (5) for supervised learning, if  $\theta = 1$  and  $\theta' = 0$ . Next, (10) reduces to

$$
\widehat {R} _ {\mathrm {u u}} (g) = \frac {1}{n} \sum_ {i = 1} ^ {n} \pi_ {\mathrm {p}} \ell (g (x _ {i})) - \frac {1}{n} \sum_ {i = 1} ^ {n} \pi_ {\mathrm {p}} \ell (- g (x _ {i})) + \frac {1}{n ^ {\prime}} \sum_ {j = 1} ^ {n ^ {\prime}} \ell (- g (x _ {j} ^ {\prime})),
$$

if  $\theta = 1$  and  $\theta' = \pi_{\mathrm{p}}$ , and we recover the unbiased risk estimator in positive-unlabeled learning (du Plessis et al., 2015; Kiryo et al., 2017). Additionally, (10) reduces to a fairly complicated unbiased risk estimator in similar unlabeled learning (Bao et al., 2018), if  $\theta = \pi_{\mathrm{p}}, \theta' = \pi_{\mathrm{p}}^2 / (2\pi_{\mathrm{p}}^2 - 2\pi_{\mathrm{p}} + 1)$  or vice versa. Therefore, UU learning is a very general framework in weakly-supervised learning.

Table 1: Specification of benchmark datasets, models, and optimization algorithms.  

<table><tr><td>Dataset</td><td># Train</td><td># Test</td><td># Feature</td><td>πp</td><td>Model g(x;θ)</td><td>Opt.</td></tr><tr><td>MNIST</td><td>60,000</td><td>10,000</td><td>784</td><td>0.49</td><td>5-layer FC with ReLU</td><td>SGD</td></tr><tr><td>Fashion-MNIST</td><td>60,000</td><td>10,000</td><td>784</td><td>0.50</td><td>5-layer FC with ReLU</td><td>SGD</td></tr><tr><td>SVHN</td><td>100,000</td><td>26,032</td><td>3,072</td><td>0.27</td><td>12-layer CNN with ReLU</td><td>Adam</td></tr><tr><td>CIFAR-10</td><td>50,000</td><td>10,000</td><td>3,072</td><td>0.60</td><td>32-layer ResNet with ReLU</td><td>Adam</td></tr></table>

![](images/3b94ab5e031660a4f4ffe167c9e40ef24d04b8f4ee134a9d14743f9eb0d6759c.jpg)

![](images/d69ac4f6c3869976e619603626e51482a3cfdca3e2de41a0730cc032523b6b26.jpg)

![](images/7d4a4774f2f2c168baed9271cf2d7118a66bde67c091c6c831cf61ebd3928140.jpg)

![](images/1007c381ac914beb66e50676ebab6c55dba590b07b26f6bada0c83e97c40940e.jpg)

![](images/9084080f8c44a379c939f1fbd63943af426dbd60fc820d299b9ec236b443612b.jpg)  
(a) MNIST

![](images/2c2e84f7ab4dab1eb694affb1ba13ce8c6223f1e8040df233c93b46050ed536b.jpg)  
(b) Fashion-MNIST

![](images/bc04ad19bd68f20792833e0d950fac696287be5996de61bbdc905612681bb5bd.jpg)  
(c) SVHN  
Figure 2: Experimental results of training the deep neural networks with  $\ell_{\mathrm{sig}}$ . The top/bottom row corresponds to  $(\theta, \theta') = (0.9, 0.1) / (0.8, 0.2)$ .

![](images/ca77adba5e512a34bea0a6570b039ed3eef3fdcd81d4552d79fb14e2a4bbc05a.jpg)  
(d) CIFAR-10

# 4.2 CONSISTENCY AND CONVERGENCE RATE

The consistency of UU learning is guaranteed due to the unbiasedness of (10). In what follows, we analyze the estimation error  $R(\widehat{g}_{\mathrm{uu}}) - R(g^{*})$  (see Sec. 3.1 for the definition). To this end, assume there are  $C_g > 0$  and  $C_{\ell} > 0$  such that  $\sup_{g\in \mathcal{G}}\| g\|_{\infty}\leq C_g$  and  $\sup_{|z|\leq C_g}\ell (z)\leq C_\ell$ , and assume  $\ell (z)$  is Lipschitz continuous for all  $|z|\leq C_g$  with a Lipschitz constant  $L_{\ell}$ . Let  $\Re_n(\mathcal{G})$  and  $\Re_{n^{\prime}}^{\prime}(\mathcal{G})$  be the Rademacher complexity of  $\mathcal{G}$  over  $p_{\mathrm{tr}}(x)$  and  $p_{\mathrm{tr}}'(x)$  (Mohri et al., 2012; Shalev-Shwartz & Ben-David, 2014). For convenience, denote by  $\chi_{n,n'} = \alpha /\sqrt{n} +\alpha ' / \sqrt{n'}$ .

Lemma 5. For any  $\delta >0$ , let  $C_{\delta} = \sqrt{(\ln 2 / \delta) / 2}$ , then we have with probability at least  $1 - \delta$

$$
\sup  _ {g \in \mathcal {G}} | \widehat {R} _ {\mathrm {u u}} (g) - R (g) | \leq 2 L _ {\ell} \alpha \Re_ {n} (\mathcal {G}) + 2 L _ {\ell} \alpha^ {\prime} \Re_ {n ^ {\prime}} ^ {\prime} (\mathcal {G}) + C _ {\ell} C _ {\delta} \chi_ {n, n ^ {\prime}}, \tag {13}
$$

where the probability is over repeated sampling of data for evaluating  $\widehat{R}_{\mathrm{uu}}(g)$ .

Theorem 6. For any  $\delta >0$ , let  $C_{\delta} = \sqrt{(\ln 2 / \delta) / 2}$ , then we have with probability at least  $1 - \delta$

$$
R \left(\widehat {g} _ {\mathrm {u u}}\right) - R \left(g ^ {*}\right) \leq 4 L _ {\ell} \alpha \Re_ {n} (\mathcal {G}) + 4 L _ {\ell} \alpha^ {\prime} \Re_ {n ^ {\prime}} ^ {\prime} (\mathcal {G}) + 2 C _ {\ell} C _ {\delta} \chi_ {n, n ^ {\prime}}, \tag {14}
$$

where the probability is over repeated sampling of data for training  $\widehat{g}_{\mathrm{uu}}$

Theorem 6 ensures that UU learning is consistent (and so are all the special cases): as  $n, n' \to \infty$ ,  $R(\widehat{g}_{\mathrm{uu}}) \to R(g^*)$ , since  $\Re_n(\mathcal{G}), \Re_{n'}'(\mathcal{G}) \to 0$  for all parametric models with a bounded norm such as deep networks trained with weight decay. Moreover,  $R(\widehat{g}_{\mathrm{uu}}) \to R(g^*)$  in  $\mathcal{O}_p(\chi_{n,n'})$ , where  $\mathcal{O}_p$  denotes the order in probability, for linear-in-parameter models and non-parametric kernel models in reproducing kernel Hilbert spaces with a bounded norm (Schölkopf & Smola, 2001).

# 5 EXPERIMENTS

In this section, we experimentally analyze the proposed algorithm on deep neural network models trained on various benchmark datasets. The implementation is based on Keras.

# 5.1 BENCHMARK EXPERIMENTS WITH NEURAL NETWORK MODELS

We first illustrate the operation of the proposed unbiased risk estimator and evaluate it with 3 supervised baselines: small PN, PN oracle and small PN prior-shift, where small PN/PN oracle denotes

Table 2: Mean errors (standard deviations) in percentage given inaccurate training class priors.  

<table><tr><td>Dataset</td><td>(θ, θ&#x27;)</td><td>ε = 0.8</td><td>ε = 0.9</td><td>ε = 1.0</td><td>ε = 1.1</td><td>ε = 1.2</td></tr><tr><td rowspan="3">MNIST</td><td>(0.9, 0.1)</td><td>2.90(0.12)</td><td>2.54(0.10)</td><td>2.31(0.15)</td><td>2.08(0.10)</td><td>2.10(0.06)</td></tr><tr><td>(0.8, 0.2)</td><td>3.64(0.13)</td><td>3.29(0.15)</td><td>3.01(0.12)</td><td>2.78(0.10)</td><td>2.63(0.16)</td></tr><tr><td>(0.7, 0.3)</td><td>5.15(0.23)</td><td>4.87(0.29)</td><td>4.84(0.25)</td><td>4.75(0.24)</td><td>4.68(0.28)</td></tr><tr><td rowspan="3">CIFAR-10</td><td>(0.9, 0.1)</td><td>10.50(0.36)</td><td>10.30(0.35)</td><td>10.15(0.31)</td><td>9.82(0.36)</td><td>9.81(0.35)</td></tr><tr><td>(0.8, 0.2)</td><td>11.27(0.40)</td><td>10.94(0.41)</td><td>10.77(0.41)</td><td>10.56(0.37)</td><td>10.30(0.39)</td></tr><tr><td>(0.7, 0.3)</td><td>12.50(0.59)</td><td>12.23(0.56)</td><td>11.91(0.55)</td><td>11.59(0.52)</td><td>11.38(0.48)</td></tr></table>

![](images/8cddb583914e007ed755d2cab6bd1972818bf776a9ce0626fa183e2fc7b32768.jpg)  
(a) CCN,  $\theta = 0.9$

![](images/c3feeac84b16d83ecefc3c90ae7ddef8f391f135f301d57514195d06b090f9fc.jpg)  
(b)  $\mathrm{UU},\theta = 0.9$  
Figure 3: Illustration of how moving  $\theta$  and  $\theta^{\prime}$  closer affects the classification performance of CCN and the proposed UU method. PN oracle is illustrated in black dashed line.

![](images/9ea65e57050a05292b800b7092ab03c8689314d938123354067376de7361559f.jpg)  
(c) CCN,  $\theta = 0.8$

![](images/6dc32a8efcbffaa976a7fb19e6b4213f3fef189f425999c5652b674c0ca119d6.jpg)  
(d)  $\mathrm{UU},\theta = 0.8$

fully supervised classification with  $10\% / 100\%$  training data, and small PN prior-shift means supervised classification with  $10\%$  training data under class-prior change. Note that small PN and PN oracle use fully supervised training data generated from the same distribution as the test data, which is extremely advantageous.

We test on the widely adopted benchmarks, MNIST, Fashion-MNIST, SVHN and CIFAR-10. Table 1 summarizes the specification of the datasets. For each dataset, we draw equal amount of unlabeled training data for  $\mathcal{X}_{\mathrm{tr}}$  and  $\mathcal{X}_{\mathrm{tr}}^{\prime}$  from Eq. (1), where two different pairs of training class priors are considered: (0.9, 0.1), (0.8, 0.2). The test data is directly drawn from the original joint distribution  $p(x,y)$  for evaluating the performance of the trained models.

The model and optimizer for MNIST and Fashion-MNIST were 5-layer fully connected neural network (FC) and SGD (Robbins & Monro, 1951). For SVHN and CIFAR-10, we used 12-layer all convolutional net (Springenberg et al., 2015) and 32-layer residual network (ResNet) (He et al., 2016) respectively, and the resulting objectives were minimized by Adam (Kingma & Ba, 2015). We compared the performance of the proposed risk estimator Eq. (10) with logistic loss  $\ell_{\log}(z) = \log(1 + \exp(-z))$  (Natarajan et al., 2013) and the simplified version Eq. (12) with  $\ell_{\mathrm{sig}}$  in Appendix C.2. The results show that the risk estimators with  $\ell_{\mathrm{sig}}$  and  $\ell_{\log}$  perform similarly. For simplicity, we select  $\ell_{\mathrm{sig}}$  as the surrogate loss for training (0-1 loss for testing) in the following experiments. More details on experimental setup can be found in Appendix C.1.

The experimental results are reported in Figure 2, where means and standard deviations of test risks based on the same 10 random samplings are shown. In the case of  $(\theta, \theta') = (0.9, 0.1)$ , we can see the proposed UU classifiers are comparable to the PN oracle in most cases. Note that the test class prior of SVHN is 0.27, which is farther from the corrupted UU class prior 0.5. Thus this setting is more advantageous for PN baselines without class prior shift but more challenging for UU. The results in (c) show that UU still outperforms small PN while small PN prior-shift deteriorates severely. For the harder case of  $(\theta, \theta') = (0.8, 0.2)$ , the performances of UU classifiers drop slightly, but is still comparable to small PN. The drop here can be explained by larger noise in the U sets when moving  $\theta$  and  $\theta'$  closer, and we investigate this issue in Figure 3.

Analysis of moving  $\theta$  and  $\theta^{\prime}$  closer It is intuitive that if  $\theta$  and  $\theta^{\prime}$  are closer, the U sets will be less informative. To investigate the influence of this, we conducted additional experiments on MNIST by moving  $\theta$  and  $\theta^{\prime}$  closer, where  $\theta \in \{0.9, 0.8\}$  and  $\theta^{\prime}$  is gradually moved from 0.1 to 0.5. The experimental setup is exactly same as before. We reported the means and standard deviations of the test risks over 10 trails in Figure 3. The results show that the proposed unbiased UU method works reasonably well, while the performance of CCN drops severely. It is because the marginal  $p(x)$  is shifted more distant from training to test stages as  $\theta^{\prime}$  moves closer to  $\theta$ , which will make the

Table 3: Means and standard deviations of classification errors over 10 trials in percentage. Best and comparable methods based on the t-test at the significance level  $1\%$  are highlighted in boldface.  

<table><tr><td>Dataset</td><td># Train</td><td># Test</td><td>πp</td><td>pSVM</td><td>BER</td><td>BER-FC</td><td>UU</td></tr><tr><td>pendigits</td><td>971</td><td>350</td><td>0.10</td><td>4.03(0.27)</td><td>5.51(1.35)</td><td>5.46(1.23)</td><td>1.97(0.78)</td></tr><tr><td>covtype-binary</td><td>3863</td><td>1500</td><td>0.30</td><td>14.63(1.00)</td><td>11.33(0.26)</td><td>5.17(0.57)</td><td>4.97(0.48)</td></tr><tr><td>MNIST</td><td>11640</td><td>2000</td><td>0.50</td><td>N/A</td><td>3.10(0.17)</td><td>3.03(0.25)</td><td>2.87(0.28)</td></tr><tr><td>spambase</td><td>1139</td><td>400</td><td>0.70</td><td>29.18(1.29)</td><td>11.28(1.73)</td><td>13.98(1.63)</td><td>12.53(1.00)</td></tr><tr><td>letter</td><td>532</td><td>200</td><td>0.90</td><td>15.65(4.18)</td><td>15.45(6.99)</td><td>8.45(2.92)</td><td>3.15(0.84)</td></tr><tr><td rowspan="5">USPS</td><td>971</td><td>350</td><td>0.10</td><td>5.91(1.52)</td><td>12.69(4.09)</td><td>8.57(2.40)</td><td>3.74(1.24)</td></tr><tr><td>2605</td><td>800</td><td>0.30</td><td>5.55(0.46)</td><td>5.36(0.41)</td><td>2.75(0.28)</td><td>2.63(0.18)</td></tr><tr><td>1695</td><td>600</td><td>0.50</td><td>9.27(0.61)</td><td>7.27(1.09)</td><td>5.48(1.33)</td><td>5.52(1.02)</td></tr><tr><td>1853</td><td>600</td><td>0.70</td><td>8.20(0.73)</td><td>7.48(0.65)</td><td>4.23(0.50)</td><td>4.43(0.94)</td></tr><tr><td>424</td><td>150</td><td>0.90</td><td>9.80(2.07)</td><td>14.13(2.02)</td><td>18.27(5.17)</td><td>6.20(1.33)</td></tr></table>

CCN-based risk estimator more biased. Thanks to the unbiasedness of the proposed risk estimator, UU classifiers can handle the covariate shift reasonably well. We further analyzed the influence of covariate shift by changing the sample sizes of the U sets in Appendix C.2, the experimental results are consistent.

Robustness of noisy training class priors In the above experiments, we assume the true training class priors  $\theta$  and  $\theta'$  are exactly accessible, but in practice we may only be able to approximately specify them. This motivates our study to try some cases when  $\theta$  and  $\theta'$  are misspecified, in order to simulate UU learning in the wild. We run more experiments by replacing the true training class priors  $(\theta, \theta')$  with  $(\vartheta, \vartheta') = (\epsilon \theta, \epsilon \theta')$  and give  $(\vartheta, \vartheta')$  to the learning method. The experimental setup is exactly same as before except that the training class priors are noisy. We reported the classification errors of the learned models in Table 2. The results show that the proposed method is fairly robust to the misspecification of training class priors so long as  $|\vartheta - \theta| + |\vartheta' - \theta'| \ll |\theta - \theta'|$ .

# 5.2 COMPARISON WITH STATE-OF-THE-ART METHODS

We finally compare our method with two state-of-the-art methods for dealing with two sets of U data: the proportion-SVM method (pSVM) (Yu et al., 2013) and the balanced error minimisation method (BER) (Menon et al., 2015). We downloaded the codes from the webpage of authors. Note that the pSVM method is based on maximum margin clustering (Xu et al., 2004; Valizadegan & Jin, 2006; Li et al., 2009) and the original codes of BER implement neural network by Matlab and use the second order optimization method. Considering the time for 10 runs of each experiment, we used UCI benchmarks and USPS datasets for the experiment following pSVM and BER. By re-sampling the original datasets, we test several different settings of class prior  $\pi_{\mathrm{p}}$ .

The 5-layer FC and SGD were again used for training UU classifier here. For fairness, we also implemented BER method using the same network architecture and optimizer as UU (BER-FC). The specifications of the datasets are summarized and the experimental results are reported in Table 3. We can see that the proposed method outperforms others in most cases. The closer  $\pi_{\mathrm{p}}$  is to  $\frac{1}{2}$ , the better classification performance of BER and BER-FC. In particular, in the experiments of MNIST and USPS  $(\pi_{\mathrm{p}} = 0.5)$ , BER and BER-FC can achieve comparable results as UU where pSVM falls behind. This is because the balanced error assumption holds in this case. However, in the experiments of pendigits and USPS  $(\pi_{\mathrm{p}} = 0.1, 0.9)$ , our method still works well while the performance of BER and BER-FC drops severely and become inferior to pSVM.

# 6 CONCLUSIONS

We focused on training arbitrary binary classifier, including deep networks, from only U data by ERM. We proved that this is impossible given a single set of U data, but it is possible given two sets of U data with different class priors, where all class priors necessary for training are given. This led to an unbiased risk estimator and subsequently we proposed the first ERM-based learning method from two sets of U data. Experiments demonstrated that the proposed method could even successfully train AllConvNet and ResNet, and it compared favorably with state-of-the-art methods for learning from two sets of U data.

# REFERENCES

D. Angluin and P. Laird. Learning from noisy examples. Machine Learning, 2(4):343-370, 1988.  
H. Bao, G. Niu, and M. Sugiyama. Classification from pairwise similarity and unlabeled data. In ICML, 2018.  
P. L. Bartlett, M. I. Jordan, and J. D. McAuliffe. Convexity, classification, and risk bounds. Journal of the American Statistical Association, 101(473):138-156, 2006.  
M. Belkin, P. Niyogi, and V. Sindhwani. Manifold regularization: a geometric framework for learning from labeled and unlabeled examples. Journal of Machine Learning Research, 7:2399-2434, 2006.  
K. H. Brodersen, C. S. Ong, K. E. Stephan, and J. M. Buhmann. The balanced accuracy and its posterior distribution. In ICPR, 2010.  
O. Chapelle, J. Weston, and B. Schölkopf. Cluster kernels for semi-supervised learning. In NIPS, 2002.  
O. Chapelle, B. Scholkopf, and A. Zien (eds.). Semi-Supervised Learning. MIT Press, 2006.  
J. Cheng, T. Liu, K. Ramamohanarao, and D. Tao. Learning with bounded instance- and label-dependent label noise. arXiv preprint arXiv:1709.03768, 2017.  
M. C. du Plessis, G. Niu, and M. Sugiyama. Clustering unclustered data: Unsupervised binary labeling of two datasets having different class balances. In TAAI, 2013.  
M. C. du Plessis, G. Niu, and M. Sugiyama. Analysis of learning from positive and unlabeled data. In NIPS, 2014.  
M. C. du Plessis, G. Niu, and M. Sugiyama. Convex formulation for learning from positive and unlabeled data. In ICML, 2015.  
J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12:2121-2159, 2011.  
C. Elkan and K. Noto. Learning classifiers from only positive and unlabeled data. In KDD, 2008.  
J. Goldberger and E. Ben-Reuven. Training deep neural-networks using a noise adaptation layer. In ICLR, 2017.  
R. Gomes, A. Krause, and P. Perona. Discriminative clustering by regularized information maximization. In NIPS, 2010.  
I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. MIT Press, 2016.  
Y. Grandvalet and Y. Bengio. Semi-supervised learning by entropy minimization. In NIPS, 2004.  
B. Han, J. Yao, G. Niu, M. Zhou, I. W. Tsang, Y. Zhang, and M. Sugiyama. Masking: A new perspective of noisy supervision. In NIPS, 2018a.  
B. Han, Q. Yao, X. Yu, G. Niu, M. Xu, W. Hu, I. W. Tsang, and M. Sugiyama. Co-teaching: Robust training deep neural networks with extremely noisy labels. In NIPS, 2018b.  
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.  
W. Hu, T. Miyato, S. Tokui, E. Matsumoto, and M. Sugiyama. Learning discrete representations via information maximizing self augmented training. In ICML, 2017.  
L. Jiang, Z. Zhou, T. Leung, L.-J. Li, and F.-F. Li. MentorNet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In ICML, 2018.  
D. P. Kingma and J. L. Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

R. Kiryo, G. Niu, M. C. du Plessis, and M. Sugiyama. Positive-unlabeled learning with non-negative risk estimator. In NIPS, 2017.  
S. Laine and T. Aila. Temporal ensembling for semi-supervised learning. In ICLR, 2017.  
Y.-F. Li, I. W. Tsang, J. T. Kwok, and Z.-H. Zhou. Tighter and convex maximum margin clustering. In AISTATS, 2009.  
T. Liu and D. Tao. Classification with noisy labels by importance reweighting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(3):447-461, 2016.  
Y. Luo, J. Zhu, M. Li, Y. Ren, and B. Zhang. Smooth neighbors on teacher graphs for semi-supervised learning. In CVPR, 2018.  
G. S. Mann and A. McCallum. Simple, robust, scalable semi-supervised learning via expectation regularization. In ICML, 2007.  
C. McDiarmid. On the method of bounded differences. In J. Siemons (ed.), Surveys in Combinatorics, pp. 148-188. Cambridge University Press, 1989.  
A. K. Menon, B. van Rooyen, C. S. Ong, and R. C. Williamson. Learning from corrupted binary labels via class-probability estimation. In ICML, 2015.  
A. K. Menon, B. van Rooyen, and N. Natarajan. Learning from binary labels with instance-dependent corruption. arXiv preprint arXiv:1605.00751, 2016.  
T. Miyato, S. Maeda, M. Koyama, K. Nakae, and S. Ishii. Distributional smoothing with virtual adversarial training. In ICLR, 2016.  
M. Mohri, A. Rostamizadeh, and A. Talwalkar. Foundations of Machine Learning. MIT Press, 2012.  
N. Natarajan, I. S. Dhillon, P. Ravikumar, and A. Tewari. Learning with noisy labels. In NIPS, 2013.  
G. Niu, W. Jitkrittum, B. Dai, H. Hachiya, and M. Sugiyama. Squared-loss mutual information regularization: A novel information-theoretic approach to semi-supervised learning. In ICML, 2013.  
G. Niu, M. C. du Plessis, T. Sakai, Y. Ma, and M. Sugiyama. Theoretical comparisons of positive-unlabeled learning against positive-negative learning. In NIPS, 2016.  
G. Patrini, A. Rozza, A. K. Menon, R. Nock, and L. Qu. Making deep neural networks robust to label noise: A loss correction approach. In CVPR, 2017.  
N. Quadrianto, A. J. Smola, T. S. Caetano, and Q. V. Le. Estimating labels from label proportions. Journal of Machine Learning Research, 10:2349-2374, 2009.  
J. Quinonero-Candela, M. Sugiyama, A. Schwaighofer, and N. D. Lawrence. Dataset Shift in Machine Learning. MIT Press, 2009.  
S. Reed, H. Lee, D. Anguelov, C. Szegedy, D. Erhan, and A. Rabinovich. Training deep neural networks on noisy labels with bootstrapping. In ICLR workshop, 2015.  
M. D. Reid and R. C. Williamson. Composite binary losses. Journal of Machine Learning Research, 11:2387-2422, 2010.  
M. Ren, W. Zeng, B. Yang, and R. Urtasun. Learning to reweight examples for robust deep learning. In ICML, 2018.  
H. Robbins and S. Monro. A stochastic approximation method. The Annals of Mathematical Statistics, 22(3):400-407, 1951.  
B. Scholkopf and A. Smola. Learning with Kernels. MIT Press, 2001.  
C. Scott, G. Blanchard, and G. Handy. Classification with asymmetric label noise: Consistency and maximal denoising. In  $COLT$ , 2013.

S. Shalev-Shwartz and S. Ben-David. Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press, 2014.  
J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. Striving for simplicity: The all convolutional net. In ICLR, 2015.  
N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15: 1929-1958, 2014.  
M. Sugiyama, G. Niu, M. Yamada, M. Kimura, and H. Hachiya. Information-maximization clustering based on squared-loss mutual information. *Neural Computation*, 26(1):84-131, 2014.  
S. Sukhbaatar, J. Bruna, M. Paluri, L. Bourdev, and R. Fergus. Training convolutional networks with noisy labels. In ICLR workshop, 2015.  
A. Tarvainen and H. Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In NIPS, 2017.  
A. Tewari and P. L. Bartlett. On the consistency of multi-class classification methods. Journal of Machine Learning Research, 8:1007-1025, 2007.  
A. N. Tikhonov. On the stability of inverse problems (in Russian). Doklady Akademii Nauk SSSR, 39(5):195-198, 1943.  
H. Valizadegan and R. Jin. Generalized maximum margin clustering and unsupervised kernel learning. In NIPS, 2006.  
B. van Rooyen and R. C. Williamson. A theory of learning with corrupted labels. Journal of Machine Learning Research, 18(228):1-50, 2018.  
V. N. Vapnik. Statistical Learning Theory. John Wiley & Sons, 1998.  
G. Ward, T. Hastie, S. Barry, J. Elith, and J. Leathwick. Presence-only data and the EM algorithm. Biometrics, 65(2):554-563, 2009.  
L. Xu, J. Neufeld, B. Larson, and D. Schuurmans. Maximum margin clustering. In NIPS, 2004.  
F. X. Yu, D. Liu, S. Kumar, T. Jebara, and S.-F. Chang.  $\propto$ SVM for learning with label proportions. In ICML, 2013.
