# PERTURBATION ANALYSIS OF NEURAL COLLAPSE

Anonymous authors

Paper under double-blind review

# ABSTRACT

Training deep neural networks for classification often includes minimizing the training loss beyond the zero training error point. In this phase of training, a "neural collapse" behavior has been observed: the variability of features (outputs of the penultimate layer) of within-class samples decreases and the mean features of different classes approach a certain tight frame structure. Recent works analyze this behavior via idealized unconstrained features models where all the minimizers exhibit exact collapse. However, with practical networks and datasets, the features typically do not reach exact collapse, e.g., because deep layers cannot arbitrarily modify intermediate features that are far from being collapsed. In this paper, we propose a richer model that can capture this phenomenon by forcing the features to stay in the vicinity of a predefined features matrix (e.g., intermediate features). We explore the model in the small vicinity case via perturbation analysis and establish results that cannot be obtained by the previously studied models. For example, we prove reduction in the within-class variability of the optimized features compared to the predefined input features (via analyzing gradient flow on the "central-path" with minimal assumptions), analyze the minimizers in the near-collapse regime, and provide insights on the effect of regularization hyperparameters on the closeness to collapse. We support our theory with experiments in practical deep learning settings.

# 1 INTRODUCTION

Modern classification systems are typically based on deep neural networks (DNNs), whose parameters are optimized using a large amount of labeled training data. Their training scheme often includes minimizing the training loss beyond the zero training error point (Hoffer et al., 2017; Ma et al., 2018; Belkin et al., 2019). In this terminal phase of training, a "neural collapse" (NC) behavior has been empirically observed when using either cross-entropy (CE) loss (Papyan et al., 2020) or mean squared error (MSE) loss (Han et al., 2022).

The NC behavior includes several simultaneous phenomena that evolve as the number of epochs grows. The first phenomenon, dubbed NC1, is decrease in the variability of the features (outputs of the penultimate layer) of training samples from the same class. The second phenomenon, dubbed NC2, is increasing similarity of the structure of the inter-class features' means (after subtracting the global mean) to a simplex equiangular tight frame (ETF). The third phenomenon, dubbed NC3, is alignment of the last layer's weights with the inter-class features' means. A consequence of these phenomena is that the classifier's decision rule becomes similar to nearest class center in feature space.

Many recent works attempt to theoretically analyze the NC behavior (Mixon et al., 2020; Lu & Steinerberger, 2022; Wojtowytsch et al., 2021; Fang et al., 2021; Zhu et al., 2021; Graf et al., 2021; Ergen & Pilanci, 2021; Ji et al., 2021; Galanti et al., 2021; Tirer & Bruna, 2022; Zhou et al., 2022; Thrampoulidis et al., 2022; Yang et al., 2022; Kothapalli et al., 2022). The mathematical frameworks are almost always based on variants of the unconstrained features model (UFM), proposed by Dixon et al. (2020), which treats the (deepest) features of the training samples as free optimization variables (disconnected from data or intermediate/shallow features). Typically, in these "idealized" models all the minimizers exhibit "exact collapse" (i.e., their within-class variability is exactly 0 and an exact simplex ETF structure is demonstrated) provided that arbitrary (but nonzero) level of regularization is used.

However, the features of DNNs are not free optimization variables but outputs of predetermined architectures that get training samples as input and have parameters (shared by all the samples) that are hard to optimize. Thus, usually, the deepest features demonstrate reduced "NC distance metrics" (such as within-class variability) compared to features of intermediate layers but do not exhibit convergence to an exact collapse. Indeed, as can be seen in any NC paper that presents empirical results, the decrease in the NC metrics is typically finite and stops above zero at some epoch (the margin depends on the dataset complexity, architecture, hyperparameter tuning, etc.).

In this paper, this issue is taken into account by studying a model that can force the features to stay in the vicinity of a predefined features matrix. By considering the predefined features as intermediate features of a DNN, the proposed model allows us to analyze how deep features progress from, or relate to, shallower features. We explore the model in the small vicinity case via perturbation analysis and establish results that cannot be obtained by the previously studied UFMs. Specifically, we prove reduction in the within-class variability of the optimized features compared to the predefined input features. To obtain this result (for arbitrary input features), we prove monotonic decrease of within-class variability along gradient flow on the "central-path" of a UFM with minimal assumptions (i.e., we drop the assumptions and modifications of the flow that Han et al. (2022) did to facilitate their analysis). Next, we provide a closed-form approximation for the model's minimizer. Then, focusing on the case where the input features matrix is already near collapse (e.g., the penultimate features of a well-trained DNN), we present a fine-grained analysis of our closed-form approximation, which provides insights on the effect of regularization hyperparameters on the closeness to collapse. We support our theory with experiments in practical deep learning settings.

# 2 BACKGROUND AND PROBLEM SETUP

Consider a classification task with  $K$  classes and  $n$  training samples per class. Let us denote by  $\mathbf{y}_k\in \mathbb{R}^K$  the one-hot vector with 1 in its  $k$ -th entry and by  $\mathbf{x}_{k,i}\in \mathbb{R}^p$  the  $i$ -th training sample of the  $k$ -th class. DNN-based classifiers can be typically expressed as

$$
\mathrm {D N N} _ {\Theta} (\mathbf {x}) = \mathbf {W h} _ {\theta} (\mathbf {x}) + \mathbf {b},
$$

where  $\mathbf{h}_{\pmb{\theta}}(\cdot):\mathbb{R}^p\to \mathbb{R}^d$  (with  $d\geq K$ ) is the feature mapping that is composed of multiple layers (with learnable parameters  $\pmb{\theta}$ ), and  $\mathbf{W} = [\mathbf{w}_1,\dots ,\mathbf{w}_K]^\top \in \mathbb{R}^{K\times d}$  ( $\mathbf{w}_k^\top$  denotes the  $k$ th row of  $\mathbf{W}$ ) and  $\mathbf{b}\in \mathbb{R}^K$  are the weights and bias of the last classification layer. The network's parameters  $\Theta = \{\mathbf{W},\mathbf{b},\pmb {\theta}\}$  are usually learned by empirical risk minimization

$$
\min  _ {\boldsymbol {\Theta}} \frac {1}{K n} \sum_ {k = 1} ^ {K} \sum_ {i = 1} ^ {n} \mathcal {L} \left(\mathbf {W h} _ {\boldsymbol {\theta}} \left(\mathbf {x} _ {k, i}\right) + \mathbf {b}, \mathbf {y} _ {k}\right) + \mathcal {R} (\boldsymbol {\Theta}),
$$

where  $\mathcal{L}(\cdot, \cdot)$  is a loss function (e.g., CE or MSE) and  $\mathcal{R}(\cdot)$  is a regularization term (e.g., squared  $L_{2}$ -norm).

Following the work of Mixon et al. (2020), in order to mathematically show the emergence of minimizers with NC structure, most of the theoretical papers have followed the "unconstrained features model" (UFM) approach, where the features  $\{\mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x}_{k,i})\}$  are treated as free optimization variables  $\{\mathbf{h}_{k,i}\}$ . Namely, they study problems of the form

$$
\min  _ {\mathbf {W}, \mathbf {b}, \left\{\mathbf {h} _ {k, i} \right\}} \frac {1}{K n} \sum_ {k = 1} ^ {K} \sum_ {i = 1} ^ {n} \mathcal {L} \left(\mathbf {W h} _ {k, i} + \mathbf {b}, \mathbf {y} _ {k}\right) + \mathcal {R} \left(\mathbf {W}, \mathbf {b}, \left\{\mathbf {h} _ {k, i} \right\}\right).
$$

One such example is the work in (Tirer & Bruna, 2022), which considered a setting with regularized MSE loss of the form

$$
\min  _ {\mathbf {W}, \mathbf {H}} \frac {1}{2 K n} \| \mathbf {W} \mathbf {H} - \mathbf {Y} \| _ {F} ^ {2} + \frac {\lambda_ {W}}{2 K} \| \mathbf {W} \| _ {F} ^ {2} + \frac {\lambda_ {H}}{2 K n} \| \mathbf {H} \| _ {F} ^ {2}, \tag {1}
$$

where  $\mathbf{H} = [\mathbf{h}_{1,1},\dots,\mathbf{h}_{1,n},\mathbf{h}_{2,1},\dots,\mathbf{h}_{K,n}]\in \mathbb{R}^{d\times Kn}$  is the (organized) unconstrained features matrix,  $\mathbf{Y} = \mathbf{I}_K\otimes \mathbf{1}_n^\top \in \mathbb{R}^{K\times Kn}$  (where  $\otimes$  denotes the Kronecker product) is its associated one-hot vectors matrix, and  $\lambda_W$  and  $\lambda_H$  are positive regularization hyperparameters. It was shown that

all the (global) minimizers of this bias-free UFM exhibit an orthogonal collapse, as stated in the following theorem.

Theorem 2.1 (Theorem 3.1 in (Tirer & Bruna, 2022)). Let  $d \geq K$  and define  $c \coloneqq \sqrt{\lambda_H \lambda_W}$ . If  $c \leq 1$ , then any global minimizer  $(\mathbf{W}^*, \mathbf{H}^*)$  of Eq. 1 satisfies

$$
\mathbf {h} _ {k, 1} ^ {*} = \dots = \mathbf {h} _ {k, n} ^ {*} =: \bar {\mathbf {h}} _ {k} ^ {*}, \quad \forall k \in [ K ],
$$

$$
\| \overline {{\mathbf {h}}} _ {1} ^ {*} \| _ {2} ^ {2} = \dots = \| \overline {{\mathbf {h}}} _ {K} ^ {*} \| _ {2} ^ {2} =: \rho = (1 - c) \sqrt {\frac {\lambda_ {W}}{\lambda_ {H}}},
$$

$$
\left[ \overline {{\mathbf {h}}} _ {1} ^ {*}, \dots , \overline {{\mathbf {h}}} _ {K} ^ {*} \right] ^ {\top} \left[ \overline {{\mathbf {h}}} _ {1} ^ {*}, \dots , \overline {{\mathbf {h}}} _ {K} ^ {*} \right] = \rho \mathbf {I} _ {K},
$$

$$
\mathbf {w} _ {k} ^ {*} = \sqrt {\lambda_ {H} / \lambda_ {W}} \overline {{\mathbf {h}}} _ {k} ^ {*}, \quad \forall k \in [ K ].
$$

If  $c > 1$ , then Eq. 1 is minimized by  $(\mathbf{W}^{*},\mathbf{H}^{*}) = (\mathbf{0},\mathbf{0})$

In short, the theorem states that any minimizer  $(\mathbf{W}^{*},\mathbf{H}^{*})$  of Eq. 1 obeys that  $\mathbf{H}^{*} = \overline{\mathbf{H}}\otimes \mathbf{1}_{n}^{\top}$  for some  $\overline{\mathbf{H}}\in \mathbb{R}^{d\times K}$ , and  $\mathbf{W}^{*}\overline{\mathbf{H}}\propto \overline{\mathbf{H}}^{\top}\overline{\mathbf{H}}\propto \mathbf{W}^{*}\mathbf{W}^{*\top}\propto \mathbf{I}_{K}$ . It is not hard to show that  $\overline{\mathbf{H}}^{\top}\overline{\mathbf{H}} = \rho \mathbf{I}_K$  implies that

$$
\left(\overline {{\mathbf {H}}} - \overline {{\mathbf {h}}} _ {G} ^ {*} \mathbf {1} _ {K} ^ {\top}\right) ^ {\top} \left(\overline {{\mathbf {H}}} - \overline {{\mathbf {h}}} _ {G} ^ {*} \mathbf {1} _ {K} ^ {\top}\right) = \rho \left(\mathbf {I} _ {K} - \frac {1}{K} \mathbf {1} _ {K} \mathbf {1} _ {K} ^ {\top}\right),
$$

where  $\overline{\mathbf{h}}_G^* = \frac{1}{K}\sum_{k=1}^{K}\overline{\mathbf{h}}_k^* = \frac{1}{K}\overline{\mathbf{H}}\mathbf{1}_K$  is the global mean. Namely, the "mean-subtracted features" collapse to a simplex ETF. From the structure of the problem and the theorem, we see that there are infinitely many minimizers of Eq. 1. Indeed, as can be deduced from the proof of Theorem 2.1 in (Tirer & Bruna, 2022): Taking any (partial) orthonormal matrix  $\mathbf{R} \in \mathbb{R}^{d \times K}$  (i.e.,  $\mathbf{R}^\top \mathbf{R} = \mathbf{I}_K$ ), one can construct a minimizer for Eq. 1 simply by  $\mathbf{H}^* = \sqrt{\rho(\lambda_W, \lambda_H)}\mathbf{R} \otimes \mathbf{1}_n^\top$  and  $\mathbf{W}^* = \sqrt{\lambda_H / \lambda_W}\sqrt{\rho(\lambda_W, \lambda_H)}\mathbf{R}^\top$ .

The existing literature includes other different UFM settings where all the minimizers exhibit NC structures (e.g., see (Lu & Steinerberger, 2022; Wojtowytsch et al., 2021; Zhu et al., 2021; Fang et al., 2021; Thrampoulidis et al., 2022)). However, as discussed in Section 1, all the previously studied UFMs are idealized and their results deviate from the situation in practical DNN training, where the features do not exhibit exact collapse (e.g., since deep layers cannot arbitrarily modify intermediate features that are far from being collapsed) and the setting of the hyperparameters affects the distance from NC structure.

In this paper, we consider a different model with the goal of better analyzing the real-world "near collapse" situation where "exact NC" cannot be reached. Motivated by Eq. 1, we consider the following model

$$
\min  _ {\mathbf {W}, \mathbf {H}} f (\mathbf {W}, \mathbf {H}; \mathbf {H} _ {0}) = \frac {1}{2 K n} \| \mathbf {W} \mathbf {H} - \mathbf {Y} \| _ {F} ^ {2} + \frac {\lambda_ {W}}{2 K} \| \mathbf {W} \| _ {F} ^ {2} + \frac {\lambda_ {H}}{2 K n} \| \mathbf {H} \| _ {F} ^ {2} + \frac {\beta}{2 K n} \| \mathbf {H} - \mathbf {H} _ {0} \| _ {F} ^ {2}, \tag {2}
$$

where  $\mathbf{H}_0\in \mathbb{R}^{d\times Kn}$  is an input features matrix, which is fixed, and  $\beta$  is a positive hyperparameter that controls the distance of  $\mathbf{H}$  from  $\mathbf{H}_0$ .

Let us discuss the motivation for studying this model. As before, we interpret  $\mathbf{W}$  and  $\mathbf{H}$  as the final weights and deepest features of the DNN, respectively. Clearly, for  $\mathbf{H}_0 = \mathbf{0}$  this model reduces to Eq. 1 (with  $\| \mathbf{H} \|_F^2$  regularized by  $\lambda_H + \beta$ ). Furthermore, when  $\mathbf{H}_0$  is nonzero, but already collapsed (i.e., a minimizer of Eq. 1), the following statement is straightforward.

Corollary 2.2. Let  $d \geq K$ ,  $\lambda_H \lambda_W < 1$ , and let  $(\mathbf{W}^*, \mathbf{H}^*)$  be a minimizer of Eq. 1. Then, the minimizer of  $f(\mathbf{W}, \mathbf{H}; \mathbf{H}_0 = \mathbf{H}^*)$  (in Eq. 2) is unique $^2$  and it is given by  $(\mathbf{W}^*, \mathbf{H}^*)$ .

That is, Eq. 2 allows us to pick one of the minimizers of Eq. 1 by  $\mathbf{H}_0$  and transfer its orthogonal collapse properties, which are stated in Theorem 2.1, to the minimizer of Eq. 2.

However, the usefulness of Eq. 2 comes from exploring cases with nonzero/non-collapsed  $\mathbf{H}_0$ . Indeed, while  $\mathbf{H}$  can be interpreted as the deepest features of a DNN, here we interpret  $\mathbf{H}_0$  as the features that are obtained in a shallower layer. In this case,  $1 / \beta$  can be understood as the complexity of the subnetwork (or link) from  $\mathbf{H}_0$  to  $\mathbf{H}$ . Specifically,  $\beta \ll 1$  is associated with extremely expressive yet easy-to-optimize architecture between  $\mathbf{H}_0$  and  $\mathbf{H}$ , which allows modifying  $\mathbf{H}$  almost as in Eq. 1. On the other hand,  $\beta \gg 1$  is associated with a simple architecture between  $\mathbf{H}_0$  and  $\mathbf{H}$  (e.g., a single layer) that significantly constrains  $\mathbf{H}$ . In this paper we focus on this large  $\beta$  regime, and provide mathematical reasoning for the empirical NC behavior that are not captured by previously studied UFMs, such as proving that the optimized  $\mathbf{H}$  has smaller within-class variability than  $\mathbf{H}_0$ , and analyzing how perturbations from collapse of  $\mathbf{H}_0$  can be mitigated by the minimizer of Eq. 2.

# 3 DECREASE IN WITHIN-CLASS VARIABILITY

As discussed above, while the features matrix  $\mathbf{H}$  represents the output of a DNN's penultimate layer, the input matrix  $\mathbf{H}_0$  can be interpreted as the features of a preceding layer. Several works have presented empirical settings where the within-class variability of the features, measured by some "NC1 metric", decreases across depth (Papyan et al., 2020; Tirer & Bruna, 2022; Galanti, 2022). The goal of this section is to prove such a phenomenon for the model stated in Eq. 2. The theory that we provide shows also monotonic decrease of the within-class variability (till exact collapse) along gradient flow on the "central-path" of the UFM stated in Eq. 1.

Let us begin with several definitions that will be used in this section. For a given set of  $n$  features for each of  $K$  classes,  $\{\mathbf{h}_{k,i}\}$ , we define the per-class and global means as  $\overline{\mathbf{h}}_k\coloneqq \frac{1}{n}\sum_{i = 1}^n\mathbf{h}_{k,i}$  and  $\overline{\mathbf{h}}_G\coloneqq \frac{1}{Kn}\sum_{k = 1}^k\sum_{i = 1}^n\mathbf{h}_{k,i}$ , respectively, as well as the mean features matrix  $\overline{\mathbf{H}}\coloneqq [\overline{\mathbf{h}}_1,\dots ,\overline{\mathbf{h}}_K]$ . Next, we define the within-class and between-class  $d\times d$  covariance matrices

$$
\boldsymbol {\Sigma} _ {W} (\mathbf {H}) := \frac {1}{K n} \sum_ {k = 1} ^ {K} \sum_ {i = 1} ^ {n} (\mathbf {h} _ {k, i} - \overline {{\mathbf {h}}} _ {k}) (\mathbf {h} _ {k, i} - \overline {{\mathbf {h}}} _ {k}) ^ {\top},
$$

$$
\boldsymbol {\Sigma} _ {B} (\mathbf {H}) := \frac {1}{K} \sum_ {k = 1} ^ {K} \left(\overline {{\mathbf {h}}} _ {k} - \overline {{\mathbf {h}}} _ {G}\right) \left(\overline {{\mathbf {h}}} _ {k} - \overline {{\mathbf {h}}} _ {G}\right) ^ {\top}.
$$

The within-class variability collapse (NC1) can be expressed as  $\Sigma_W(\mathbf{H})\to \mathbf{0}$  while  $\Sigma_B(\mathbf{H})\nrightarrow \mathbf{0}$ , where the limit takes place with increasing the training epoch, and  $\Sigma_B(\mathbf{H}) > 0$  filters degenerate cases such as  $\mathbf{H} = \mathbf{0}$ . Several papers considered in their experiments the metric  $\frac{1}{K}\mathrm{Tr}\left(\Sigma_W(\mathbf{H})\Sigma_B^\dagger (\mathbf{H})\right)$ , where  $\Sigma_B^\dagger$  denotes the pseudoinverse of  $\Sigma_B$  (Papyan et al., 2020; Han et al., 2022; Zhu et al., 2021). Yet, we believe that considering the metric

$$
\widetilde {\mathcal {N C}} _ {1} (\mathbf {H}) := \operatorname {T r} \left(\boldsymbol {\Sigma} _ {W} (\mathbf {H})\right) / \operatorname {T r} \left(\boldsymbol {\Sigma} _ {B} (\mathbf {H})\right) \tag {3}
$$

is more amenable for theoretical analysis while capturing the desired nondegenerate collapse behavior. Indeed, the trace of a covariance matrix equals zero if and only if the covariance matrix is a zero matrix (this follows from  $\mathrm{Cov}^2(X,Y) \leq \mathrm{Var}(X)\mathrm{Var}(Y)$ ).

Recall that the minimizer w.r.t.  $\mathbf{W}$  in Eq. 2 (and Eq. 1) has a closed-form expression that is a function of  $\mathbf{H}$ , which is given by  $\mathbf{W}^{*}(\mathbf{H}) = \mathbf{Y}\mathbf{H}^{\top}(\mathbf{H}\mathbf{H}^{\top} + n\lambda_{W}\mathbf{I}_{d})^{-1}$ . Thus, the optimization in Eq. 2 is equivalent to

$$
\mathbf {H} _ {1 / \beta} := \underset {\mathbf {H}} {\operatorname {a r g m i n}} \mathcal {L} (\mathbf {H}) + \frac {\beta}{2 K n} \| \mathbf {H} - \mathbf {H} _ {0} \| _ {F} ^ {2}
$$

where  $\mathcal{L}(\mathbf{H})\coloneqq \frac{1}{2Kn}\| \mathbf{W}^{*}(\mathbf{H})\mathbf{H} - \mathbf{Y}\|_{F}^{2} + \frac{\lambda_{W}}{2K}\| \mathbf{W}^{*}(\mathbf{H})\|_{F}^{2} + \frac{\lambda_{H}}{2Kn}\| \mathbf{H}\|_{F}^{2}$

For large  $\beta$ , the minimizer  $\mathbf{H}_{1 / \beta}$  can be viewed as a backward/implicit gradient descent update from  $\mathbf{H}_0$  with respect to the loss  $\mathcal{L}$ . This follows from rewriting the first order optimality condition as

$$
\frac {\mathbf {H} _ {1 / \beta} - \mathbf {H} _ {0}}{1 / \beta} = - K n \nabla \mathcal {L} (\mathbf {H} _ {1 / \beta}).
$$

Observing that for  $\beta \to \infty$  we have  $\mathbf{H}_{1 / \beta} \to \mathbf{H}_0$  (formally shown in Appendix B), the above equation can be written as  $\left.\frac{d\mathbf{H}_t}{dt}\right|_{t=0} = -Kn\nabla \mathcal{L}(\mathbf{H}_0)$ , where we think of  $t$  as  $\beta^{-1}$ . This naturally gives rise to the gradient flow

$$
\frac {d \mathbf {H} _ {t}}{d t} = - K n \nabla \mathcal {L} (\mathbf {H} _ {t}), \tag {4}
$$

associated with the UFM in Eq. 1. This means that results on this flow can be translated to results on the minimizer of Eq. 2 in the large  $\beta$  regime. Indeed, in Theorem 3.1 below, we show that  $\widetilde{NC}_1(\mathbf{H})$  monotonically decreases along this flow, which implies that  $\widetilde{NC}_1(\mathbf{H}_{1 / \beta}) < \widetilde{NC}_1(\mathbf{H}_0)$  for large enough  $\beta$  (see the statement in Corollary 3.2 below).

Note that a flow for an objective that is equivalent to  $\mathcal{L}(\mathbf{H})$  with  $\lambda_W = 0$  and  $\lambda_H = 0$  has been studied in (Han et al., 2022), who called it the "central path". The motivation for studying such an objective, where the optimization variable  $\mathbf{W}$  is replaced by the optimal  $\mathbf{W}^*(\mathbf{H})$ , comes from the empirical observation that  $\|\mathbf{W}^*(\mathbf{H})\mathbf{H} - \mathbf{Y}\|_F^2 - \|\mathbf{W}\mathbf{H} - \mathbf{Y}\|_F^2$  is rather small during the optimization process of practical DNNs.

We now state our result for gradient flow on the "central path" (which is proved in Appendix A).

Theorem 3.1. Assume that  $\lambda_W > 0$ ,  $\lambda_H \geq 0$ , and that  $\mathbf{H}_0$  is non-collapsed (i.e.,  $\Sigma_W(\mathbf{H}_0) \neq \mathbf{0}$ ). Then, along the gradient flow, which is stated in Eq. 4, we have that

-  $\widetilde{NC}_1(\mathbf{H}_t)$  strictly decreases along the flow until it reaches zero.  
-  $t \mapsto e^{2\lambda_H t} \operatorname{Tr}(\pmb{\Sigma}_W(\mathbf{H}_t))$  decreases along the flow. In particular, when  $\lambda_H > 0$ ,  $\operatorname{Tr}(\pmb{\Sigma}_W(\mathbf{H}_t))$  decays exponentially.  
-  $t \mapsto e^{2\lambda_H t} \operatorname{Tr}(\Sigma_B(\mathbf{H}_t))$  strictly increases along the flow.

Remark. Note that our gradient flow analysis has minimal assumptions. Unlike (Han et al., 2022), our flow does not assume zero global mean  $(\overline{\mathbf{h}}_G = \mathbf{0})$ ,  $\lambda_W = \lambda_H = 0$  and invertibility of  $\boldsymbol{\Sigma}_W$ . And most importantly, it does not include any engineered renormalization and projection of the gradient, contrary to the previous work. Thus, it is more similar to practical gradient descent optimization of DNNs. Our unmodified flow and minimal assumptions require a different, and more general, analysis with quite involved computations. $^4$

Not only does Theorem 3.1 state a monotonic decrease toward 0 in the NC1 metric, it also provides a separation between the behavior of  $\mathrm{Tr}(\pmb{\Sigma}_W)$  and  $\mathrm{Tr}(\pmb{\Sigma}_B)$  along the flow. A strict separation is observed for  $\lambda_H = 0$ :  $\mathrm{Tr}(\pmb{\Sigma}_W)$  decreases while  $\mathrm{Tr}(\pmb{\Sigma}_B)$  increases. As gradient flow is often used as a proxy for analyzing gradient descent with a small step-size (Elkabetz & Cohen, 2021), if we overlook the difference between optimizing the UFM in Eq. 1 jointly w.r.t.  $\mathbf{W}$  and  $\mathbf{H}$  and restricting the optimization to the "central path"  $(\mathbf{W}^{*}(\mathbf{H}),\mathbf{H})$ , then our theory also provides a mathematical reasoning for the experiments on gradient descent in (Tirer & Bruna, 2022) that show monotonic decrease in within-class variability.

Finally, with our interpretation of  $t$  as  $\beta^{-1}$ , the following Corollary is a direct consequence of Theorem 3.1 and the continuity of  $\nabla \mathcal{L}(\mathbf{H})$  (see Appendix B for a formal proof).

Corollary 3.2. Assume that  $\mathbf{H}_0$  is non-collapsed. Then, there exists some constant  $C = C(\mathbf{H}_0) > 0$  such that for  $\beta > C$  we have that  $\widetilde{NC_1}(\mathbf{H}_{1/\beta}) < \widetilde{NC_1}(\mathbf{H}_0)$ .

Recall that in the large  $\beta$  regime we can interpret  $\mathbf{H}$  as features of DNN that are deeper than  $\mathbf{H}_0$  but such that the architecture between  $\mathbf{H}_0$  and  $\mathbf{H}$  is extremely simple (e.g., they are features of adjacent layers) and thus the distance between them is constrained. Under this interpretation, Corollary 3.2 implies that layer-wise optimization of DNN where each time a new layer is added (so that the previous deepest features  $\mathbf{H}_{1 / \beta}$  are considered as the new  $\mathbf{H}_0$ ) will result in gradually depthwise decreasing NC1. An extension of the model in Eq. 2 that will include multiple levels of estimizable parameters may be able to provide similar reasoning to the gradual depthwise decrease in NC1 that is observed in practical DNN training, where all the layers are optimized simultaneously.

# 4 ANALYSIS OF THE NEAR-COLLAPSE REGIME

In this section, we will explore the behavior of the minimizers of Eq. 2 in the near-collapse regime. As stated in Corollary 2.2, if  $\mathbf{H}_0$  is already collapsed then the minimizer of Eq. 2 is also collapsed. This is aligned with the rationale that if we have a DNN that already exhibits collapse at some intermediate layer, we would expect the subsequent layers to maintain this collapse. $^5$  Essentially, we would like to analyze the minimizer of Eq. 2 for  $\mathbf{H}_0$  that is not already collapsed. Unfortunately, for general non-collapsed  $\mathbf{H}_0$  it is not likely that the minimizer is amenable for explicit analytical characterization. Yet, the fact that for orthogonally collapsed  $\mathbf{H}_0 = \mathbf{H}^*$  we get a unique minimizer  $(\mathbf{W}^*,\mathbf{H}^*)$  of Eq. 2, which is still characterized by Theorem 2.1, gives us a desirable setting for examining the minimizer of Eq. 2 obtained for  $\tilde{\mathbf{H}}_0 = \mathbf{H}^* +\delta \mathbf{H}_0$  (with sufficiently small  $\delta \mathbf{H}_0$ ) by exploiting our knowledge on  $(\mathbf{W}^*,\mathbf{H}^*;\mathbf{H}_0 = \mathbf{H}^*)$ . Analyzing the near-collapse setting will shed light on the way that the deviation from collapse in the input features is transferred to the optimized features, e.g., the amount of interaction within/between classes and the effects of hyperparameters. Such insights can be latter examined empirically beyond the near-collapse regime.

Let us denote by  $(\tilde{\mathbf{W}}^{*},\tilde{\mathbf{H}}^{*})$  the minimizer of  $f(\mathbf{W},\mathbf{H};\tilde{\mathbf{H}}_0)$ . We are interested in studying the dependence of  $\delta \mathbf{W} \coloneqq \tilde{\mathbf{W}}^{*} - \mathbf{W}^{*}$  and  $\delta \mathbf{H} \coloneqq \tilde{\mathbf{H}}^{*} - \mathbf{H}^{*}$  on  $\delta \mathbf{H}_0 = \tilde{\mathbf{H}}_0 - \mathbf{H}^*$  without the requirement of computing  $(\tilde{\mathbf{W}}^{*},\tilde{\mathbf{H}}^{*})$  (that lack analytical expressions). In particular, our focus is on the relation between the features  $\delta \mathbf{H}$  and  $\delta \mathbf{H}_0$  (rather than  $\delta \mathbf{W}$  and  $\delta \mathbf{H}_0$ ), both because a minimizer  $\tilde{\mathbf{H}}^*$  uniquely implies the associated  $\tilde{\mathbf{W}}^*$ , and because important aspects of NC, such as within-class variability decrease (NC1) and inter-class feature structure (NC2), consider the feature mapping rather than the last layer weights.

We begin with establishing such a result in the following theorem (which is proved in Appendix C) for  $\mathbf{H}_0$  that is not necessarily a collapsed features matrix.

The notation in the theorem is as follows. We use  $\mathrm{vec}(\cdot)$  to denote the column-stack vectorization of a matrix. The derivatives are w.r.t. the vectorized matrices  $\mathrm{vec}(\mathbf{H})$  and  $\mathrm{vec}(\mathbf{W})$ . For example,  $\nabla_H f \in \mathbb{R}^{dnK \times 1}$  stands for the derivative of  $f$  w.r.t.  $\mathrm{vec}(\mathbf{H})$ , and a second derivative w.r.t.  $\mathrm{vec}(\mathbf{W})^\top$  yields  $\nabla_W^\top \nabla_H f \in \mathbb{R}^{dnK \times Kd}$ .

Theorem 4.1. Let  $d \geq K$ , and set some  $\mathbf{H}_0$  and  $\delta \mathbf{H}_0$ . Let  $(\hat{\mathbf{W}}^*, \hat{\mathbf{H}}^*)$  be the minimizer of  $f(\mathbf{W}, \mathbf{H}; \mathbf{H}_0)$  (with  $f$  stated in Eq. 2). Let  $(\tilde{\mathbf{W}}^*, \tilde{\mathbf{H}}^*)$  be the minimizer of  $f(\mathbf{W}, \mathbf{H}; \tilde{\mathbf{H}}_0 = \mathbf{H}_0 + \delta \mathbf{H}_0)$ . Define  $\delta \mathbf{W} := \tilde{\mathbf{W}}^* - \hat{\mathbf{W}}^*$  and  $\delta \mathbf{H} := \hat{\mathbf{H}}^* - \hat{\mathbf{H}}^*$ . Then, with approximation accuracy of  $O(\| \delta \mathbf{H} \|^2, \| \delta \mathbf{W} \|^2, \| \delta \mathbf{H}_0 \|^2)$ , we have that

$$
\operatorname {v e c} (\delta \mathbf {H}) \approx \frac {\beta}{K n} \left(\nabla_ {H} ^ {\top} \nabla_ {H} f - \nabla_ {W} ^ {\top} \nabla_ {H} f (\nabla_ {W} ^ {\top} \nabla_ {W} f) ^ {- 1} \nabla_ {H} ^ {\top} \nabla_ {W} f\right) ^ {- 1} \operatorname {v e c} (\delta \mathbf {H} _ {0}),
$$

$$
\operatorname {v e c} (\delta \mathbf {W}) \approx - \frac {\beta}{K n} (\nabla_ {W} ^ {\top} \nabla_ {W} f) ^ {- 1} \nabla_ {H} ^ {\top} \nabla_ {W} f (\nabla_ {H} ^ {\top} \nabla_ {H} f - \nabla_ {W} ^ {\top} \nabla_ {H} f (\nabla_ {W} ^ {\top} \nabla_ {W} f) ^ {- 1} \nabla_ {H} ^ {\top} \nabla_ {W} f) ^ {- 1} \operatorname {v e c} (\delta \mathbf {H} _ {0}),
$$

where all the derivatives are evaluated at the point  $(\hat{\mathbf{W}}^{*},\hat{\mathbf{H}}^{*};\mathbf{H}_{0})$

In particular, for  $\beta \gg \max \{1,\lambda_H\}$  we have (with additional approximation error of  $O(\beta^{-2})$

$$
\operatorname {v e c} (\delta \mathbf {H}) \approx \left(\mathbf {I} _ {d n K} - \frac {\lambda_ {H}}{\beta} \mathbf {I} _ {d n K} - \frac {1}{\beta} \mathbf {I} _ {n K} \otimes \hat {\mathbf {W}} ^ {* \top} \hat {\mathbf {W}} ^ {*} + \frac {1}{\beta} \mathbf {Z} ^ {*}\right) \operatorname {v e c} (\delta \mathbf {H} _ {0}), \tag {5}
$$

where

$$
\begin{array}{l} \mathbf {Z} ^ {*} := \left(\mathbf {E} ^ {* ^ {\top}} + \hat {\mathbf {H}} ^ {*} \otimes \hat {\mathbf {W}} ^ {*}\right) ^ {\top} \left(\hat {\mathbf {H}} ^ {*} \hat {\mathbf {H}} ^ {* ^ {\top}} \otimes \mathbf {I} _ {K} + n \lambda_ {W} \mathbf {I} _ {d K}\right) ^ {- 1} \left(\mathbf {E} ^ {* ^ {\top}} + \hat {\mathbf {H}} ^ {*} \otimes \hat {\mathbf {W}} ^ {*}\right), \\ \mathbf {E} ^ {*} := \left[ \operatorname {v e c} \left(\mathbf {e} _ {d, 1} \mathbf {e} _ {K, 1} ^ {\top} \left(\hat {\mathbf {W}} ^ {*} \hat {\mathbf {H}} ^ {*} - \mathbf {Y}\right)\right), \dots , \operatorname {v e c} \left(\mathbf {e} _ {d, 1} \mathbf {e} _ {K, K} ^ {\top} \left(\hat {\mathbf {W}} ^ {*} \hat {\mathbf {H}} ^ {*} - \mathbf {Y}\right)\right), \operatorname {v e c} \left(\mathbf {e} _ {d, 2} \mathbf {e} _ {K, 1} ^ {\top} \left(\hat {\mathbf {W}} ^ {*} \hat {\mathbf {H}} ^ {*} - \mathbf {Y}\right)\right), \dots \right. \\ \left. \dots , \operatorname {v e c} \left(\mathbf {e} _ {d, d} \mathbf {e} _ {K, K} ^ {\top} \left(\hat {\mathbf {W}} ^ {*} \hat {\mathbf {H}} ^ {*} - \mathbf {Y}\right)\right) \right], \\ \end{array}
$$

and  $\mathbf{e}_{d,i}$  is the standard vector in  $\mathbb{R}^d$  with 1 in its  $i$ th entry (similar definition stands for  $\mathbf{e}_{K,k}$ ).

Observe that, assuming small approximation error, Theorem 4.1 states the linear operation that transforms  $\delta \mathbf{H}_0$  to  $\delta \mathbf{H}$ . We will focus on the large  $\beta$  regime that is stated in Eq. 5, where the matrix inversion can be well approximated. Furthermore, due to the vectorization operation, observe that the linear expression  $\mathrm{vec}(\delta \mathbf{H})\approx \mathbf{F}\mathrm{vec}(\delta \mathbf{H}_0)$  has the following block-based representation

$$
\left[ \begin{array}{c} \operatorname {v e c} (\delta \mathbf {H} ^ {(1)}) \\ \vdots \\ \operatorname {v e c} (\delta \mathbf {H} ^ {(K)}) \end{array} \right] \approx \left[ \begin{array}{c c c} \mathbf {F} _ {1, 1} & \dots & \mathbf {F} _ {1, K} \\ & \ddots & \\ \mathbf {F} _ {K, 1} & \dots & \mathbf {F} _ {K, K} \end{array} \right] \left[ \begin{array}{c} \operatorname {v e c} (\delta \mathbf {H} _ {0} ^ {(1)}) \\ \vdots \\ \operatorname {v e c} (\delta \mathbf {H} _ {0} ^ {(K)}) \end{array} \right], \tag {6}
$$

where  $\delta \mathbf{H}^{(k)}\coloneqq \delta \mathbf{H}[:,dn(k - 1) + 1:dnK]\in \mathbb{R}^{d\times n}$  is the sub-matrix of  $\delta \mathbf{H}$  that is composed of the columns associated with the  $k$ th class (and similarly for  $\delta \mathbf{H}_0$ ). Namely, we have that  $\mathbf{F}\in \mathbb{R}^{dnK\times dnK}$  is composed of blocks of size  $dn\times dn$ . The diagonal blocks are the "intra-class blocks". Each of them shows the effect of perturbation in a certain class in  $\mathbf{H}_0$  on the features of the same class in  $\mathbf{H}$ . The off-diagonal blocks are the "inter-class blocks". Each of them shows the effect of perturbation in a certain class in  $\mathbf{H}_0$  on the features of another class in  $\mathbf{H}$ .

Recall that for  $\mathbf{H}_0 = \mathbf{H}^*$  that is already exactly collapsed, the minimizer of  $f(\cdot; \mathbf{H}_0)$  is also collapsed, so  $\hat{\mathbf{H}}^* = \mathbf{H}^*$  in the above theorem. Importantly, in this case the matrix in Eq. 6 transforms deviation from exact collapse in the input features to deviation from exact collapse in the optimized features. Thus, we have that stronger attenuation behavior of the blocks of  $\mathbf{F}$  (e.g., small singular values) implies that the minimizer  $\hat{\mathbf{H}}^*$  is closer to exact collapse. Based on specializing Theorem 4.1 to the near-collapse case, we present in the following theorem (which is proved in Appendix D) an exact analysis of singular values of the blocks of  $\mathbf{F}$ . (The notations  $\sigma_{max}(\cdot)$  and  $\sigma_{min}(\cdot)$  stand for the largest and smallest singular vlaues of a matrix, respectively).

Theorem 4.2. Consider the setting of Theorem 4.1,  $\lambda_H\lambda_W < 1$  (assumed in Theorem 2.1),  $d > K$ ,  $\beta \gg \max \{1,\lambda_H\}$ , and the representation of Eq. 5 that is given in Eq. 6. Let  $\mathbf{H}_0$  be a collapse features matrix (minimizer of Eq. 1 for the same  $\lambda_H$ ,  $\lambda_W$  as in Eq. 2). Then, for  $k,\tilde{k} \in [K]$  with  $k \neq \tilde{k}$  we have that  $\mathbf{F}_{k,k}$  is full rank,  $\mathbf{F}_{k,\tilde{k}}$  is rank-1, and

$$
\sigma_ {m a x} (\mathbf {F} _ {k, k}) = 1,
$$

$$
\sigma_ {m i n} \left(\mathbf {F} _ {k, k}\right) = 1 - \beta^ {- 1} \sqrt {\lambda_ {H} / \lambda_ {W}},
$$

$$
\sigma_ {m a x} \left(\mathbf {F} _ {k, \tilde {k}}\right) = 2 \beta^ {- 1} \lambda_ {H} \left(1 - \sqrt {\lambda_ {H} \lambda_ {W}}\right).
$$

Remark. In Appendix D we derive expressions for the complete singular value decomposition of  $\mathbf{F}_{k,k}$  and  $\mathbf{F}_{k,\tilde{k}}$ . Our expressions for the entire spectrum of  $\mathbf{F}_{k,k}$  reveal its step-wise decreasing shape, as visualized in Figure 1 for  $\beta = 100$ ,  $K = 4$ ,  $d = 10$ ,  $n = 10$ ,  $\lambda_{W} = \sqrt{2}$  and various values of  $\lambda_{H}$ . To keep the paper concise, we state in the above theorem only the results for the maximal and minimal singular values of  $\mathbf{F}_{k,k}$ , but note that, similarly to  $\sigma_{min}(\mathbf{F}_{k,k})$ , almost all singular values decrease as  $\lambda_{H}$  increases. Even though a small portion  $(\frac{1 - K / d}{p})$  of the singular values equal 1 (as shown in our analysis in

Appendix D), we can still gain insights on the attenuation profile since generic perturbations are unlikely to concentrate in such an extremely low-dimensional subspace (and, in fact, the singular vectors associated with this subspace do not affect the within-class variability).

From Theorem 4.2 we gain the following insights on the minimizer of Eq. 2 in the near-collapse and large  $\beta$  regime. First, observe that not only do exactly collapsed minimizers have orthogonal features

![](images/6e52a97992fbcae8f4c7b7d785d86e84e69080adbc164b5ce2ffd246b7390074.jpg)  
Figure 1: The effect of  $\lambda_{H}$  on the spectrum of  $\mathbf{F}_{k,k}$ .

for different classes, but also in the near-collapse setting the intra-class blocks are much more dominant than the inter-class blocks, as follows from  $\mathbf{F}_{k,\tilde{k}}$  being rank-1 and  $\sigma_{max}(\mathbf{F}_{k,\tilde{k}}) \ll \sigma_{min}(\mathbf{F}_{k,k})$ . In other words, also before/near pure collapse, we have that the deviation from collapse in the features of a certain class is mainly due to deviation from collapse of input (preceding) features of the same class and not those of other classes. (This implies preservation of per-class near-collapse). Second, we see that the feature mapping regularization plays the major role in approaching (near-)collapse behavior. Indeed, increasing  $\lambda_{H}$  decreases the spectral values of the (more dominant) intra-class blocks  $\{\mathbf{F}_{k,k}\}$  (contrary to increasing  $\lambda_{W}$ ). Recall that reducing the singular vlaues of the blocks of  $\mathbf{F}$  implies reducing the distance of the minimizer  $\tilde{\mathbf{H}}^{*}$  from exact collapse. Third, our result on the inter-class blocks  $\{\mathbf{F}_{k,\tilde{k} \neq k}\}$  hints that the regularization of the last layer's weights (determined by  $\lambda_{W} > 0$ ) may still have a supportive effect on reaching (near-)collapse behavior by reducing the component of the deviation from collapse that is due to "crosstalk"/interference of features of different classes (e.g., when some classes are harder to be classified then others). In the sequel, we show that the above observations correlate with the NC behavior in practical settings.

# 5 EXPERIMENTS

In this section, we translate the insights that are obtained for the model in Eq. 2 to what is observed with practical DNNs and datasets. We evaluate the distance of DNN's features from exact NC using metrics that have been also used in previous works. Despite defining the metric  $\widetilde{NC}_1$  in Eq. 3, here we mainly measure within-class variability using  $NC_1 \coloneqq \frac{1}{K} \operatorname{Tr}\left(\boldsymbol{\Sigma}_W \boldsymbol{\Sigma}_B^\dagger\right)$ , where we use the definitions of Section 3. (We use this metric due to its popularity even though it is less amenable for theoretical analysis). We measure the structure of the features using

$$
N C _ {2} := \left\| \frac {(\overline {{\mathbf {H}}} - \overline {{\mathbf {h}}} _ {G} \mathbf {1} _ {K} ^ {\top}) ^ {\top} (\overline {{\mathbf {H}}} - \overline {{\mathbf {h}}} _ {G} \mathbf {1} _ {K} ^ {\top})}{\| (\overline {{\mathbf {H}}} - \overline {{\mathbf {h}}} _ {G} \mathbf {1} _ {K} ^ {\top}) ^ {\top} (\overline {{\mathbf {H}}} - \overline {{\mathbf {h}}} _ {G} \mathbf {1} _ {K} ^ {\top}) \| _ {F}} - \frac {1}{\sqrt {K - 1}} (\mathbf {I} _ {K} - \frac {1}{K} \mathbf {1} _ {K} \mathbf {1} _ {K} ^ {\top}) \right\| _ {F},
$$

where the simplex ETF is normalized to unit Frobenius norm.

The result of Section 3 provides reasoning to justify depthwise decrease in within-class variability, which has already been empirically demonstrated for end-to-end training in several papers (Papyan et al., 2020; Tirer & Bruna, 2022; Galanti, 2022) (we present such experiments in Appendix E.2). Here we show this behavior also for layer-wise training, which is better represented by our model. We consider the CIFAR-10 dataset and train an MLP with 1 to 10 hidden layers and a final classification layer. Each time, we add and train a hidden layer on top of the previous hidden layers, which are maintained fixed. Then we compute the NC1 metrics for the deepest features. Due to space limitation, the

![](images/0dd61344281bfd6071d6f9a638158ddd10d5c9f638b6ff572a727f9a78299d39.jpg)  
Figure 2: Layer-wise training of MLP on CIFAR-10.

experimental details are deferred to Appendix E.1. Figure 2 demonstrates decrease in both  $NC_{1}$  and  $\widetilde{NC}_{1}$  as we add more hidden layers on top the previous, which are maintained fixed. Note that our theory justifies such decrease for all the layers (the features are not required to be near collapse).

Next, we turn to demonstrate correlation of practical NC behavior with the insight gained in Section 4 that  $\lambda_{H}$  plays a bigger role than  $\lambda_{W}$  does in approaching NC. Based on the equivalence of  $L_{2}$ -regularization with weight decay (WD) in gradient-based methods, we can make the analogy of regularizing  $\mathbf{H}$  in Eq. 2 to WD of the weights of practical DNNs in the feature mapping layers (i.e., excluding the last layer's weights). Importantly, note that this analogy is empirically justified for plain UFMs in (Zhu et al., 2021). Under this analogy, our analysis suggests that, as long as entering the zero training error phase of training is maintained, increasing (resp. decreasing) the WD in the feature mapping layers should decrease (resp. increase) the distance from exact collapse more than increasing (resp. decreasing) the WD in the classification layer. Indeed, we empirically show this behavior below. (More experiments are presented in Appendix E.2). We note that there exists a work that empirically<sup>7</sup> shows that WD facilitates collapse (Rangamani & Banburski-Fahey, 2022), however, they do not examine the WD in feature mapping and classification layers separately.

![](images/9341e58f82cd70e43e6a3d02cb3c1330b6afc8c61e31a05a2a941980754272b5.jpg)

![](images/14ad90b9a8a3d6ea4a2a0c760d7ae0d4d1e9fa8ee9d74fed473c67310280d47f.jpg)

![](images/7a97e9eaa06a1c5b410a01c6eafba2d51f8c578342a17bb29def7750a8e8d2c7.jpg)

![](images/b12c5d1f79fe1312030ec791cb9c14051f32a6682d75c7de1b0629ab636cdb2c.jpg)

![](images/a68c0bfec04e24be45eb8557c3f2933aeda7e62c2ec87514b7c390f45962a9a8.jpg)  
Figure 3: The effect of modifying the weight decay (WD) on NC metrics for ResNet18 trained on CIFAR-10. Top: MSE loss without bias; Bottom: CE loss with bias. Observe that modifying the WD in the feature mapping increases the deviation from the baseline more than modifying the WD of the last layer.

![](images/1c1465bcb9686d58dcfea31dc172a5fca318d073bb25da788cab39192c1a8f8a.jpg)

![](images/3c8d10d511c7d5408940ccfe138307227b7134f97dbe42b39be26f17b0b3a32c.jpg)

![](images/72a843bb6511e4489c82058064728051318cd680bf12474016461e09f043adaf.jpg)

We consider the CIFAR-10 dataset and examine how modifying the regularization hyperparameters affects the NC behavior of the widely used ResNet18 (He et al., 2016) compared to a baseline setting. Specifically, as a baseline hyperparameter setting, we consider one that is used in previous works (Papyan et al., 2020; Zhu et al., 2021): default PyTorch initialization of the weights, SGD optimizer with learning rate 0.05 that is divided by 10 every 40 epochs, momentum of 0.9, and WD of 5e-4 for all the network's parameters. The modifications include: 1) doubling the WD only for the last (FC) layer; 2) doubling the WD only for feature mapping (conv) layers; 3) zeroing the WD for the last layer; and 4) zeroing the WD for feature mapping layers.

Figure 3 presents the NC1 and NC2 metrics of the (deepest) features for: (Top) MSE loss with no bias in the FC layer (similar to the analyzed model); and (Bottom) CE loss with bias in the FC layer. In all the settings, we reach zero training error at the 40 epoch approximately. The empirical results show that modifying the WD in the feature mapping layers leads to curves with larger deviations from the baseline compared to modifying the last layer's WD, which is aligned with the theory established in Section 4 (i.e., the important role of  $\lambda_{H}$  in attenuating the dominant intra-class perturbations). Reducing (zeroing) the WD in the feature mapping increases the distance from exact NC (i.e., from 0 value of the metrics), while increasing the WD decreases the gap from exact NC, as the theory predicts. The fact that sometimes (e.g., with CE loss) increasing the WD of the last layer can also decrease the gap from collapse hints that mitigating inter-class interference/correlation of features in practical deep learning settings is more significant for reaching NC than in our analysis that considers a near-collapse regime. $^{8}$  Yet, both the experiments and the theoretical study show that the regularization of the feature mapping has larger significance in approaching NC.

# 6 CONCLUSION

The features that are learned by training practical networks on real world datasets typically do not reach exact NC. In this paper, we addressed this issue by studying a model that can force the features to stay in the vicinity of a predefined features matrix. We analyzed it for the small vicinity case and established results that cannot be obtained by the previously studied (idealized) UFMs. We proved reduction in within-class variability of the optimized features compared to the input features (via analyzing gradient flow along the "central-path" of a UFM with minimal assumptions, unlike existing literature). We also presented an analysis of the model's minimizer in the near-collapse regime that provides insights on the effect of the regularization hyperparameters on the closeness to collapse, which correlate with the behavior in practical deep learning settings. We believe that our perturbation analysis approach, which is based on exploiting our knowledge on exactly collapsed minimizers of UFMs for studying non-collapse cases, can be applied to models other than the one considered in this paper, such as models with different loss functions and/or multiple levels of features and/or imbalanced data.

# REFERENCES

Mikhail Belkin, Alexander Rakhlin, and Alexandre B Tsybakov. Does data interpolation contradict statistical optimality? In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 1611-1619. PMLR, 2019.  
Omer Elkabetz and Nadav Cohen. Continuous vs. discrete optimization of deep neural networks. Advances in Neural Information Processing Systems, 34:4947-4960, 2021.  
Tolga Ergen and Mert Pilanci. Revealing the structure of deep neural networks via convex duality. In International Conference on Machine Learning, pp. 3004-3014. PMLR, 2021.  
Cong Fang, Hangfeng He, Qi Long, and Weijie J Su. Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training. Proceedings of the National Academy of Sciences, 118(43), 2021.  
Tomer Galanti. A note on the implicit bias towards minimal depth of deep neural networks. arXiv preprint arXiv:2202.09028, 2022.  
Tomer Galanti, András György, and Marcus Hutter. On the role of neural collapse in transfer learning. arXiv preprint arXiv:2112.15121, 2021.  
Florian Graf, Christoph Hofer, Marc Niethammer, and Roland Kwitt. Dissecting supervised constrastive learning. In International Conference on Machine Learning, pp. 3821-3830. PMLR, 2021.  
XY Han, Vardan Papyan, and David L Donoho. Neural collapse under mse loss: Proximity to and dynamics on the central path. In International Conference on Learning Representations, 2022.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Elad Hoffer, Itay Hubara, and Daniel Soudry. Train longer, generalize better: closing the generalization gap in large batch training of neural networks. Advances in neural information processing systems, 30, 2017.  
Wenlong Ji, Yiping Lu, Yiliang Zhang, Zhun Deng, and Weijie J Su. An unconstrained layer-peeled perspective on neural collapse. arXiv preprint arXiv:2110.02796, 2021.  
Vignesh Kothapalli, Ebrahim Rasromani, and Vasudev Awatramani. Neural collapse: A review on modelling principles and generalization. arXiv preprint arXiv:2206.04041, 2022.  
Jianfeng Lu and Stefan Steinerberger. Neural collapse under cross-entropy loss. Applied and Computational Harmonic Analysis, 2022.  
Siyuan Ma, Raef Bassily, and Mikhail Belkin. The power of interpolation: Understanding the effectiveness of sgd in modern over-parametrized learning. In International Conference on Machine Learning, pp. 3325-3334. PMLR, 2018.  
Dustin G Mixon, Hans Parshall, and Jianzong Pi. Neural collapse with unconstrained features. arXiv preprint arXiv:2011.11619, 2020.  
Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40): 24652-24663, 2020.  
Akshay Rangamani and Andrzej Banburski-Fahey. Neural collapse in deep homogeneous classifiers and the role of weight decay. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 4243-4247. IEEE, 2022.  
Christos Thrampoulidis, Ganesh R Kini, Vala Vakilian, and Tina Behnia. Imbalance trouble: Revisiting neural-collapse geometry. arXiv preprint arXiv:2208.05512, 2022.

Tom Tirer and Joan Bruna. Extended unconstrained features model for exploring deep neural collapse. In Proceedings of the 39th International Conference on Machine Learning, volume 162, pp. 21478-21505. PMLR, 2022.  
Stephan Wojtowytsch et al. On the emergence of simplex symmetry in the final and penultimate layers of neural network classifiers. Proceedings of Machine Learning Research, 145:1-21, 2021.  
Yibo Yang, Liang Xie, Shixiang Chen, Xiangtai Li, Zhouchen Lin, and Dacheng Tao. Do we really need a learnable classifier at the end of deep neural network? arXiv preprint arXiv:2203.09081, 2022.  
Jinxin Zhou, Xiao Li, Tianyu Ding, Chong You, Qing Qu, and Zhihui Zhu. On the optimization landscape of neural collapse undermse loss: Global optimality with unconstrained features. In Proceedings of the 39th International Conference on Machine Learning, volume 162, pp. 27179-27202. PMLR, 2022.  
Zhihui Zhu, Tianyu Ding, Jinxin Zhou, Xiao Li, Chong You, Jeremias Sulam, and Qing Qu. A geometric analysis of neural collapse with unconstrained features. Advances in Neural Information Processing Systems, 34:29820-29834, 2021.
