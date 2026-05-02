# DISTRIBUTIONALLY ROBUST FAIR PRINCIPAL COMPONENTS VIA GEODESIC DESCENTS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Principal component analysis is a simple yet useful dimensionality reduction technique in modern machine learning pipelines. In consequential domains such as college admission, healthcare and credit approval, it is imperative to take into account emerging criteria such as the fairness and the robustness of the learned projection. In this paper, we propose a distributionally robust optimization problem for principal component analysis which internalizes a fairness criterion in the objective function. The learned projection thus balances the trade-off between the total reconstruction error and the reconstruction error gap between subgroups, taken in the min-max sense over all distributions in a moment-based ambiguity set. The resulting optimization problem over the Stiefel manifold can be efficiently solved by a Riemannian subgradient descent algorithm with a sub-linear convergence rate. Our experimental results on real-world datasets show the merits of our proposed method over state-of-the-art baselines.

# 1 INTRODUCTION

Machine learning models are ubiquitous in our daily lives and supporting the decision-making process in diverse domains. With their flourishing applications, there also surface numerous concerns regarding the fairness of the models' outputs (Mehrabi et al., 2021). Indeed, these models are prone to biases due to various reasons (Barocas et al., 2018). First, the collected training data is likely to include some demographic disparities due to the bias in the data acquisition process (e.g., conducting surveys on a specific region instead of uniformly distributed places), or the imbalance of observed events at a specific period of time. Second, because machine learning methods only care about data statistics and are objective driven, groups that are under-represented in the data can be neglected in exchange for a better objective value. Finally, even human feedback to the predictive models can also be biased, e.g., click counts are human feedback to recommendation systems but they are highly correlated with the menu list suggested previously by a potentially biased system. Real-world examples of machine learning models that amplify biases and hence potentially cause unfairness are commonplace, ranging from recidivism prediction giving higher false positive rates for African-American<sup>1</sup> to facial recognition systems having large error rate for women<sup>2</sup>.

To tackle the issue, various fairness criteria for supervised learning have been proposed in the machine learning literature, which encourage the (conditional) independence of the model's predictions on a particular sensitive attribute (Dwork et al., 2012; Hardt et al., 2016b; Kusner et al., 2017; Chouldechova, 2017; Verma & Rubin, 2018; Berk et al., 2021). Strategies to mitigate algorithmic bias are also investigated for all stages of the machine learning pipelines (Berk et al., 2021). For the pre-processing steps, (Kamiran & Calders, 2012) proposed reweighting or resampling techniques to achieve statistical parity between subgroups; or in the training steps, fairness can be encouraged by adding constraints (Donini et al., 2018) or regularizing the original objective function (Kamishima et al., 2012; Zemel et al., 2013); and in the post-processing steps, adjusting classification threshold based on examining black-box models over a holdout dataset can be used (Hardt et al., 2016b; Wei et al., 2019).

Since biases may already exist in the raw data, it is reasonable to demand the machine learning pipeline to combat biases as early as possible. We focus in this paper on the Principal Component

Analysis (PCA), which is a fundamental dimensionality reduction technique in the early stage of the pipelines (Pearson, 1901; Hotelling, 1933). PCA finds a linear transformation that embeds the original data into a lower-dimensional subspace that maximizes the variance of the projected data. Thus, PCA is prone to amplify biases if the data variability is different between the majority and the minority subgroups, see a toy example in Figure 1. A naive approach to promote fairness is to train one independent transformation for each subgroup. However, this requires knowing the sensitive attribute of each sample at test time, which would raise disparity concerns. On the contrary, using a single transformation for all subgroups is "group-blinded" and faces no discrimination problem (Lipton et al., 2018).

Learning a fair PCA has attracted attention from many fields from machine learning, statistics to signal process. Samadi et al. (2018) and Zalcberg & Wiesel (2021) propose to find the principal components that minimize the maximum subgroup reconstruction error; the min-max formulations can be relaxed and solved as semidefinite programs. Olfat & Aswani (2019) propose to learn a transformation that minimizes the possibility of predicting the sensitive attribute from the projected data. Apart from being a dimensionality reduction technique, PCA can also be thought of as a representation learning toolkit. Viewed in this way, we can also consider a more general family of fair representation learning methods that can be applied before any further analysis steps. There are a number of works develop towards this idea (Kamiran & Calders, 2012; Zemel et al., 2013; Calmon et al., 2017; Feldman et al., 2015; Beutel et al., 2017; Madras et al., 2018; Zhang et al., 2018; Tantipongpipat et al., 2019), which apply a multitude of fairness criteria.

In addition, we also focus on the robustness criteria for the linear transformation. Recently, it has been observed that machine learning models are susceptible to small perturbations of the data (Goodfellow et al., 2014; Madry et al., 2017; Carlini & Wagner, 2017). These observations have fuelled many defenses using adversarial training (Akhtar & Mian, 2018; Chakraborty et al., 2018) and distributionally robust optimization (Rahimian & Mehrotra, 2019; Kuhn et al., 2019).

Contributions. This paper blends the ideas from the field of fairness in artificial intelligence and distributionally robust optimization. Our contributions can be described as follows.

- We propose the fair principal components which balance between the total reconstruction error and the absolute gap of reconstruction error between subgroups. Moreover, we also add a layer of robustness to the principal components by considering a min-max formulation that hedges against all perturbations of the empirical distribution in a moment-based ambiguity set.  
- We provide the reformulation of the distributionally robust fair PCA problem as a finite-dimensional optimization problem over the Stiefel manifold. We provide a Riemannian gradient descent algorithm and show that it has a sub-linear convergence rate.

Figure 1 illustrates the qualitative comparison between (fair) PCA methods and our proposed method on a 2-dimensional toy example. The majority group (blue dots) spreads on the horizontal axis, while the minority group (yellow triangles) spreads on the slanted vertical axis. The nominal PCA (red) captures the majority direction to minimize the total error, while the fair PCA of Samadi et al. (2018) returns the diagonal direction to minimize the maximum subgroup error. Our fair PCA can probe the full spectrum in between these two extremes by sweeping through our penalization parameters appropriately. If we do not penalize the error gap between subgroups, we recover the PCA method; if we penalize heavily, we recover the fair PCA of Samadi et al. (2018). Extensive numerical results on real datasets are provided in Section 5. Proofs are relegated to the appendix.

![](images/9585e97856b6cfe7debb2074600064b7a58ce4643305c0f3a02d4ea3544ccdf9.jpg)  
Figure 1: Nominal PCA (red arrow), fair PCA by Samadi et al. (2018) (green arrow), and our spectrum of fair PCA (shorter arrows). Arrows show directions and are not normalized to unit length.

# 2 FAIR PRINCIPAL COMPONENT ANALYSIS

# 2.1 PRINCIPAL COMPONENT ANALYSIS

We first briefly revisit the classical PCA. Suppose that we are given a collection of  $N$  i.i.d. samples  $\{\hat{x}_i\}_{i=1}^N$  generated by some underlying distribution  $\mathbb{P}$ . For simplicity, we assume that both the empirical and population mean are zero vectors. The goal of PCA is to find a  $k$ -dimensional linear subspace of  $\mathbb{R}^d$  that explains as much variance contained in the data  $\{\hat{x}_i\}_{i=1}^N$  as possible, where  $k < d$  is a given integer. More precisely, we parametrize  $k$ -dimensional linear subspaces by orthonormal matrices, i.e., matrices whose columns are orthogonal and have unit Euclidean norm. Given any such matrix  $V$ , the associated  $k$ -dimensional subspace is the one spanned by the columns of  $V$ . The projection matrix onto the subspace is  $VV^\top$ , and hence the variance of the projected data is given by  $\operatorname{tr}(VV^\top \Xi \Xi^\top)$ , where  $\Xi = [\hat{x}_1, \dots, \hat{x}_N] \in \mathbb{R}^{d \times N}$  is the data matrix. By a slight abuse of terminology, sometimes we refer to  $V$  as the projection matrix. The problem of PCA then reads

$$
\max  _ {V \in \mathbb {R} ^ {d \times k}, V ^ {\top} V = I _ {k}} \operatorname {t r} \left(V V ^ {\top} \Xi \Xi^ {\top}\right). \tag {1}
$$

For any vector  $X\in \mathbb{R}^d$  and orthonormal matrix  $V$ , denote by  $\ell (V,X)$  the reconstruction error, i.e.,

$$
\ell (V, X) = \| X - V V ^ {\top} X \| _ {2} ^ {2} = X ^ {\top} \left(I _ {d} - V V ^ {\top}\right) X.
$$

The problem of PCA can alternatively be formulated as a stochastic optimization problem

$$
\min  _ {V \in \mathbb {R} ^ {d \times k}, V ^ {\top} V = I _ {k}} \operatorname {E} _ {\hat {\mathbb {P}}} [ \ell (V, X) ], \tag {2}
$$

where  $\hat{\mathbb{P}}$  is the empirical distribution associated with the samples  $\{\hat{x}_i\}_{i=1}^N$  and  $X \sim \hat{\mathbb{P}}$ . It is well-known that PCA admits an analytical solution. In particular, the optimal solution to problem (2) (and also problem (1)) is given by any orthonormal matrix whose columns are the eigenvectors associated with the  $k$  largest eigenvalues of the sample covariance matrix  $\Xi \Xi^\top$ .

# 2.2 FAIR PRINCIPAL COMPONENT ANALYSIS

In the fair PCA setting, we are also given a discrete sensitive attribute  $A \in \mathcal{A}$ , where  $A$  may represent features such as race, gender or education. We consider binary attribute  $A$  and let  $\mathcal{A} = \{0,1\}$ . A straightforward idea to define fairness is to require the (strict) balance of a certain objective between the two groups. For example, this is the strategy in Hardt et al. (2016a) for developing fair supervised learning algorithms. A natural objective to balance in the PCA context is the reconstruction error. It is therefore tempted to adopt the following definition.

Definition 2.1 (Fair projection). Let  $\mathbb{Q}$  be an arbitrary distribution of  $(X,A)$ . A projection matrix  $V\in \mathbb{R}^{d\times k}$  is fair relative to  $\mathbb{Q}$  if the conditional expected reconstruction error is equal between subgroups, i.e.,

$$
\mathbb {E} _ {\mathbb {Q}} [ \ell (V, X) | A = a ] = \mathbb {E} _ {\mathbb {Q}} [ \ell (V, X) | A = a ^ {\prime} ] \qquad \forall (a, a ^ {\prime}) \in \mathcal {A} \times \mathcal {A}.
$$

Unfortunately, Definition 2.1 is too stringent: for a general probability distribution  $\mathbb{Q}$ , it is possible that there exists no fair projection matrix  $V$ .

Proposition 2.2 (Impossibility result). For any distribution  $\mathbb{Q}$  on  $\mathcal{X} \times \mathcal{A}$ , let  $S = \mathbb{E}_{\mathbb{Q}}[XX^{\top}|A = 0] - \mathbb{E}_{\mathbb{Q}}[XX^{\top}|A = 1]$ . Then, there exists a fair projection matrix  $V \in \mathbb{R}^{d \times k}$  relative to  $\mathbb{Q}$  if and only if  $\mathrm{rank}(S) \leq k$ .

One way to circumvent the impossibility result is to relax the requirement of strict balance to approximate balance. In other words, an inequality constraint of the following form is imposed:

$$
| \mathbb {E} _ {\mathbb {Q}} [ \ell (V, X) | A = a ] - \mathbb {E} _ {\mathbb {Q}} [ \ell (V, X) | A = a ^ {\prime} ] | \leq \epsilon \quad \forall (a, a ^ {\prime}) \in \mathcal {A} \times \mathcal {A},
$$

where  $\epsilon > 0$  is some prescribed fairness threshold. This approach has been adopted in other fair machine learning settings, see Donini et al. (2018) and Agarwal et al. (2019) for example.

In this paper, instead of imposing the fairness requirement as a constraint, we penalize the unfairness in the objective function. Specifically, for any projection matrix  $V$ , we define the unfairness as the absolute difference between the conditional loss between two subgroups:

$$
\mathbb {U} (V, \mathbb {Q}) \triangleq | \mathbb {E} _ {\mathbb {Q}} [ \ell (V, X) | A = 0 ] - \mathbb {E} _ {\mathbb {Q}} [ \ell (V, X) | A = 1 ] |.
$$

We thus consider the following fairnes-aware PCA problem

$$
\min  _ {V \in \mathbb {R} ^ {d \times k}, V ^ {\top} V = I _ {k}} \mathbb {E} _ {\hat {\mathbb {P}}} [ \ell (V, X) ] + \lambda \mathrm {U} (V, \hat {\mathbb {P}}), \tag {3}
$$

where  $\lambda \geq 0$  is a penalty parameter to encourage fairness. Note that for fair PCA, the dataset is  $\{(\hat{x}_i,\hat{a}_i)\}_{i = 1}^N$  and hence the empirical distribution  $\hat{\mathbb{P}}$  is given by  $\hat{\mathbb{P}} = \frac{1}{N}\sum_{i = 1}^{N}\delta_{(\hat{x}_i,\hat{a}_i)}$ .

# 3 DISTRIBUTIONALLY ROBUST FAIR PCA

The weakness of empirical distribution-based stochastic optimization has been well-documented, see (Smith & Winkler, 2006; Homem-de Mello & Bayraksan, 2014). In particular, due to overfitting, the out-of-sample performance of the decision, prediction, or estimation obtained from such a stochastic optimization model is unsatisfactory, especially in the low sample size regime. Ideally, we could improve the performance by using the underlying distribution  $\mathbb{P}$  instead of the empirical distribution  $\hat{\mathbb{P}}$ . But the underlying distribution  $\mathbb{P}$  is unavailable in most practical situations, if not all. Distributional robustification is an emerging approach to handle this issue and has been shown to deliver promising out-of-sample performance in many applications (Delage & Ye, 2010; Namkoong & Duchi, 2017; Kuhn et al., 2019; Rahimian & Mehrotra, 2019). Motivated by the success of distributional robustification, we propose a robustified version of model (3), called the distributionally robust fairness-aware PCA:

$$
\min  _ {V \in \mathbb {R} ^ {d \times k}, V ^ {\top} V = I _ {k}} \sup  _ {\mathbb {Q} \in \mathbb {B} (\hat {\mathbb {P}})} \mathbb {E} _ {\mathbb {Q}} [ \ell (V, X) ] + \lambda \mathrm {U} (V, \mathbb {Q}), \tag {4}
$$

where  $\mathbb{B}(\hat{\mathbb{P}})$  is a set of probability distributions similar to the empirical distribution  $\hat{\mathbb{P}}$  in a certain sense, called the ambiguity set. The empirical distribution  $\hat{\mathbb{P}}$  is also called the nominal distribution. Many different ambiguity sets have been developed and studied in the optimization literature, see Rahimian & Mehrotra (2019) for an extensive overview.

# 3.1 THE WASSERSTEIN-TYPE AMBIGUITY SET

To present our ambiguity set and main results, we need to introduce some definitions and notations. Definition 3.1 (Wasserstein-type divergence). The divergence W between two probability distributions  $\mathbb{Q}_1\sim (\mu_1,\Sigma_1)\in \mathbb{R}^d\times \mathbb{S}_+^d$  and  $\mathbb{Q}_2\sim (\mu_2,\Sigma_2)\in \mathbb{R}^d\times \mathbb{S}_+^d$  is defined as

$$
\mathbb {W} \left(\mathbb {Q} _ {1} \parallel \mathbb {Q} _ {2}\right) \triangleq \left\| \mu_ {1} - \mu_ {2} \right\| _ {2} ^ {2} + \operatorname {t r} \left(\Sigma_ {1} + \Sigma_ {2} - 2 \left(\Sigma_ {2} ^ {\frac {1}{2}} \Sigma_ {1} \Sigma_ {2} ^ {\frac {1}{2}}\right) ^ {\frac {1}{2}}\right).
$$

The divergence  $\mathbb{W}$  coincides with the squared type-2 Wasserstein distance between two Gaussian distributions  $\mathcal{N}(\mu_1,\Sigma_1)$  and  $\mathcal{N}(\mu_2,\Sigma_2)$  (Givens & Shortt, 1984). One can readily show that  $\mathbb{W}$  is non-negative, and it vanishes if and only if  $(\mu_{1},\Sigma_{1}) = (\mu_{2},\Sigma_{2})$ , which implies that  $\mathbb{Q}_1$  and  $\mathbb{Q}_2$  have the same first- and second-moments.

Recall that the nominal distribution is  $\hat{\mathbb{P}} = \frac{1}{N}\sum_{i = 1}^{N}\delta_{(\hat{x}_i,\hat{a}_i)}$ . For any  $a\in \mathcal{A}$ , its conditional distribution given  $A = a$  is given by

$$
\hat {\mathbb {P}} _ {a} = \frac {1}{| \mathcal {I} _ {a} |} \sum_ {i \in \mathcal {I} _ {a}} \delta_ {x _ {i}}, \quad \text {w h e r e} \quad \mathcal {I} _ {a} \triangleq \{i \in \{1, \dots , N \}: a _ {i} = a \}.
$$

We also use  $(\hat{\mu}_a,\hat{\Sigma}_a)$  to denote the empirical mean vector and covariance matrix of  $X$  given  $A = a$ :

$$
\hat {\mu} _ {a} = \mathbb {E} _ {\hat {\mathbb {P}} _ {a}} [ X ] = \mathbb {E} _ {\hat {\mathbb {P}}} [ X | A = a ] \quad \mathrm {a n d} \quad \hat {\Sigma} _ {a} + \hat {\mu} _ {a} \hat {\mu} _ {a} ^ {\top} = \mathbb {E} _ {\hat {\mathbb {P}} _ {a}} [ X X ^ {\top} ] = \mathbb {E} _ {\hat {\mathbb {P}}} [ X X ^ {\top} | A = a ].
$$

For any  $a \in \mathcal{A}$ , the empirical marginal distribution of  $A$  is denoted by  $\hat{p}_a = |\mathcal{I}_a| / N$ .

Finally, for any set  $S$ , we use  $\mathcal{P}(S)$  to denote the set of all probability distributions supported on  $S$ . For any integer  $k$ , the  $k$ -by-  $k$  identity matrix is denoted  $I_k$ .

We then define our ambiguity set as

$$
\mathbb {B} (\hat {\mathbb {P}}) \triangleq \left\{\mathbb {Q} \in \mathcal {P} (\mathcal {X} \times \mathcal {A}): \begin{array}{l} \exists \mathbb {Q} _ {a} \in \mathcal {P} (\mathcal {X}) \text {s u c h t h a t :} \\ \mathbb {Q} (\mathrm {d} x \times \mathrm {d} a) = \sum_ {a \in \mathcal {A}} \hat {p} _ {a} \mathbb {Q} _ {a} (\mathrm {d} x) \delta_ {a} (\mathrm {d} a) \\ \mathrm {W} \left(\mathbb {Q} _ {a}, \hat {\mathbb {P}} _ {a}\right) \leq \varepsilon_ {a} \quad \forall a \in \mathcal {A} \end{array} \right\}, \tag {5}
$$

where  $\mathbb{Q}_a$  is the conditional distribution of  $X|A = a$ . Intuitively, each  $\mathbb{Q} \in \mathbb{B}(\hat{\mathbb{P}})$  is a joint distribution of the random vector  $(X,A)$ , formed by taking a mixture of conditional distributions  $\mathbb{Q}_a$  with mixture weight  $\hat{p}_a$ . Each conditional distribution  $\mathbb{Q}_a$  is constrained in an  $\varepsilon_a$ -neighborhood from the nominal conditional distribution  $\hat{\mathbb{P}}_a$  with respect to the W divergence. Notice that because the loss function  $\ell$  is a quadratic function of  $X$ , the (conditional) expected losses only involve the first two moments of  $X$ , and thus prescribing the ambiguity set using W would suffice for the purpose of robustification.

# 3.2 REFORMULATION

We now present the reformulation of problem (4) under the ambiguity set  $\mathbb{B}(\hat{\mathbb{P}})$ .

Theorem 3.2 (Reformulation). Suppose that either of the following two conditions holds:

(i)  $0\leq \lambda \leq \min \{\hat{p}_a,\hat{p}_{a'}\}$  
(ii) for any  $a \in \mathcal{A}$ , the empirical second moment matrix  $\hat{M}_a = \frac{1}{N_a} \sum_{i \in \mathcal{I}_a} \hat{x}_i \hat{x}_i^\top$  satisfies  $\sum_{j=1}^{d-k} \sigma_j(\hat{M}_a) \geq \varepsilon_a$ , where  $\sigma_j(\hat{M}_a)$  is the  $j$ -th smallest eigenvalues of  $\hat{M}_a$ .

Then problem (4) is equivalent to

$$
\min  _ {V \in \mathbb {R} ^ {d \times k}, V ^ {\top} V = I _ {k}} \max  \left\{J _ {0} (V), J _ {1} (V) \right\}, \tag {6a}
$$

where for each  $(a,a^{\prime})\in \{(0,1),(1,0)\}$ , the function  $J_{a}$  is defined as

$$
J _ {a} (V) = \kappa_ {a} + \theta_ {a} \sqrt {\left\langle I _ {d} - V V ^ {\top} , \hat {M} _ {a} \right\rangle} + \vartheta_ {a ^ {\prime}} \sqrt {\left\langle I _ {d} - V V ^ {\top} , \hat {M} _ {a ^ {\prime}} \right\rangle} + \left\langle I _ {d} - V V ^ {\top}, C _ {a} \right\rangle , \tag {6b}
$$

and the parameters  $\kappa \in \mathbb{R},\theta \in \mathbb{R},\vartheta \in \mathbb{R}$  and  $C\in \mathbb{S}_+^d$  are defined as

$$
\begin{array}{l} \kappa_ {a} = (\hat {p} _ {a} + \lambda) \varepsilon_ {a} + (\hat {p} _ {a ^ {\prime}} - \lambda) \varepsilon_ {a ^ {\prime}}, \quad \theta_ {a} = 2 | \hat {p} _ {a} + \lambda | \sqrt {\varepsilon_ {a}}, \quad \vartheta_ {a ^ {\prime}} = 2 | \hat {p} _ {a ^ {\prime}} - \lambda | \sqrt {\varepsilon_ {a ^ {\prime}}}, \\ C _ {a} = (\hat {p} _ {a} + \lambda) \hat {M} _ {a} + (\hat {p} _ {a ^ {\prime}} - \lambda) \hat {M} _ {a ^ {\prime}}. \end{array} \tag {6c}
$$

We now briefly explain the steps that lead to the results in Theorem 3.2. Letting

$$
J_{0}(V) = \sup_{\mathbb{Q}\in \mathbb{B}(\hat{\mathbb{P}})}(\hat{p}_{0} + \lambda)\mathbb{E}_{\mathbb{Q}}[\ell (V,X)|A = 0] + (\hat{p}_{1} - \lambda)\mathbb{E}_{\mathbb{Q}}[\ell (V,X)|A = 1],
$$

$$
J_{1}(V) = \sup_{\mathbb{Q}\in \mathbb{B}(\hat{\mathbb{P}})}(\hat{p}_{0} - \lambda)\mathbb{E}_{\mathbb{Q}}[\ell (V,X)|A = 0] + (\hat{p}_{1} + \lambda)\mathbb{E}_{\mathbb{Q}}[\ell (V,X)|A = 1],
$$

then by expanding the term  $\mathbb{U}(V,\mathbb{Q})$  using its definition, problem (4) becomes

$$
\min_{V\in \mathbb{R}^{d\times k},V^{\top}V = I_{k}}\max \{J_{0}(V),J_{1}(V)\} .
$$

By the definition the ambiguity set  $\mathbb{B}(\hat{\mathbb{P}})$ , for any pair  $(a, a') \in \{(0,1), (1,0)\}$ , we can decompose  $J_{a}$  into two separate supremum problems as follows

$$
J_{a}(V) = \sup_{\mathbb{Q}_{a}:\mathbb{W}(\mathbb{Q}_{a},\hat{\mathbb{P}}_{a})\leq \varepsilon_{a}}(\hat{p}_{a} + \lambda)\mathbb{E}_{\mathbb{Q}_{a}}[\ell (V,X)] + \sup_{\mathbb{Q}_{1}:\mathbb{W}(\mathbb{Q}_{a^{\prime}},\hat{\mathbb{P}}_{a^{\prime}})\leq \varepsilon_{a^{\prime}}}(\hat{p}_{a^{\prime}} - \lambda)\mathbb{E}_{\mathbb{Q}_{a^{\prime}}}[\ell (V,X)].
$$

The next proposition asserts that each individual supremum in the above expression admits an analytical expression.

Proposition 3.3 (Reformulation). Fix  $a \in \mathcal{A}$ . For any  $v \in \mathbb{R}$ ,  $\varepsilon_{a} \in \mathbb{R}_{+}$ , it holds that

$$
\begin{array}{l} \sup_{\substack{\mathbb{Q}_{a}:\mathbb{W}(\mathbb{Q}_{a},\hat{\mathbb{P}}_{a})\leq \varepsilon_{a}}}  \nu \mathbb{E}_{\mathbb{Q}_{a}}[\ell (V,X)] \\ = \left\{ \begin{array}{l l} v \left(\sqrt {\left\langle I _ {d} - V V ^ {\top} , \hat {M} _ {a} \right\rangle} + \sqrt {\bar {\varepsilon} _ {a}}\right) ^ {2} & \text {i f} v \geq 0, \\ v \left(\sqrt {\left\langle I _ {d} - V V ^ {\top} , \hat {M} _ {a} \right\rangle} - \sqrt {\bar {\varepsilon} _ {a}}\right) ^ {2} & \text {i f} v <   0 \text {a n d} \left\langle I _ {d} - V V ^ {\top}, \hat {M} _ {a} \right\rangle \geq \varepsilon_ {a}, \\ 0 & \text {i f} v <   0 \text {a n d} \left\langle I _ {d} - V V ^ {\top}, \hat {M} _ {a} \right\rangle <   \varepsilon_ {a}. \end{array} \right. \\ \end{array}
$$

The proof of Theorem 3.2 now follows by applying Proposition 3.3 to each term in  $J_{a}$ , and balance the parameters to obtain (6c). A detailed proof is relegated to the appendix. In the next section, we study an efficient algorithm to solve (6a).

Remark 3.4 (Recovery of the nominal PCA). If  $\lambda = 0$  and  $\varepsilon_{a} = 0 \forall a \in \mathcal{A}$ , our formulation (4) becomes the standard PCA problem (2). In this case, our robust fair principal components reduce to the standard principal components. On the contrary, existing fair PCA methods such as Samadi et al. (2018) and Olfat & Aswani (2019) cannot recover the standard principal components.

# 4 RIEMANNIAN GRADIENT DESCENT ALGORITHM

Using Theorem 3.2, our distributionally robust fairness-aware PCA problem (4), which is an infinite-dimensional minimax problem, is reduced to the simpler finite-dimensional minimax problem (6a), where the inner problem is only a maximization over two points. Problem (6a) is, however, still challenging as it is a non-convex optimization problem over a non-convex feasible region defined by the orthogonality constraint  $V^{\top}V = I_{d}$ . The purpose of this section is to devise an efficient algorithm for solving problem (6a) to local optimality based on Riemannian optimization.

# 4.1 REPARAMETRIZATION

As mentioned above, the non-convexity of problem (6a) comes from both the objective function and the feasible region. It turns out that we can get rid of the non-convexity of the objective function via a simple change of variables. To see that, we let  $U \in \mathbb{R}^{d \times (d - k)}$  be an orthonormal matrix complement to  $V$ , that is,  $U$  and  $V$  satisfy  $U U^{\top} + V V^{\top} = I_{d}$ . Thus, we can express the objective function  $J$  via

$$
J (V) = F (U) \triangleq \max  \{F _ {0} (U), F _ {1} (U) \},
$$

where for  $(a,a^{\prime})\in \{(0,1),(1,0)\}$ , the function  $F_{a}$  is defined as

$$
F _ {a} (U) \triangleq \kappa_ {a} + \theta_ {a} \sqrt {\left\langle U U ^ {\top} , \hat {M} _ {a} \right\rangle} + \vartheta_ {a ^ {\prime}} \sqrt {\left\langle U U ^ {\top} , \hat {M} _ {a ^ {\prime}} \right\rangle} + \left\langle U U ^ {\top}, C _ {a} \right\rangle .
$$

Moreover, letting  $\mathcal{M} \triangleq \{U \in \mathbb{R}^{d \times (d - k)} : U^\top U = I_{d - k}\}$ , we can re-express problem (6a) as

$$
\min  _ {U \in \mathcal {M}} F (U). \tag {7}
$$

The feasible region  $\mathcal{M}$  of problem (7) is a Riemannian manifold, called the Stiefel manifold (Absil et al., 2007, Section 3.3.2). It is then natural to solve problem (7) by using Riemannian optimization algorithms (Absil et al., 2007). In fact, problem (6a) itself (before the change of variables) can also be seen as a Riemannian optimization problem over another Stiefel manifold. The change of variables above might seem unnecessary. Nonetheless, the upshot of problem (7) is that the objective function  $F$  is convex (in the traditional sense). This facilitates the application of the theoretical and algorithmic framework developed in Li et al. (2019) for (weakly) convex optimization over Stiefel manifolds.

# 4.2 THE RIEMANNIAN SUBGRADIENT

Note that the objective function  $F$  is non-smooth since it is defined as the maximum of two functions  $F_{0}$  and  $F_{1}$ . To apply the framework in Li et al. (2019), we need to compute the Riemannian subgradient of the objective function  $F$ . Since the Stiefel manifold  $\mathcal{M}$  is an embedded manifold in Euclidean space, the Riemannian subgradient of  $F$  at any point  $U \in \mathcal{M}$  is given by the orthogonal projection of the usual Euclidean subgradient onto the tangent space of the manifold  $\mathcal{M}$  at the point  $U$ , see Absil et al. (2007, Section 3.6.1) for example.

Lemma 4.1. For any point  $U \in \mathcal{M}$ , let $^3$ $a_U \in \arg \max_{a \in \{0,1\}} F_a(U)$  and  $a_U' = 1 - a_U$ . Then, a Riemannian subgradient of the objective function  $F$  at the point  $U$  is given by

$$
\operatorname {g r a d} F (U) = \left(I _ {d} - U U ^ {\top}\right) \left(\frac {\theta_ {a _ {U}}}{\sqrt {\langle U U ^ {\top} , \hat {M} _ {a _ {U}} \rangle}} \hat {M} _ {a _ {U}} U + \frac {\vartheta_ {a _ {U} ^ {\prime}}}{\sqrt {\langle U U ^ {\top} , \hat {M} _ {a _ {U} ^ {\prime}} \rangle}} \hat {M} _ {a _ {U} ^ {\prime}} U + 2 C _ {a _ {U}} U\right).
$$

# 4.3 RETRACTIONS

Another important instrument required by the framework in Li et al. (2019) is a retraction of the Stiefel manifold  $\mathcal{M}$ . At each iteration, the point  $U - \gamma \Delta$  obtained by moving from the current iterate  $U$  in the opposite direction of the Riemannian gradient  $\Delta$  may not lie on the manifold in general, where  $\gamma > 0$  is the stepsize. In Riemannian optimization, this is circumvented by the concept of retraction. Given a point  $U \in \mathcal{M}$  on the manifold, the Riemannian gradient  $\Delta \in T_U\mathcal{M}$  (which must lie in the tangent space  $T_U\mathcal{M}$ ) and a stepsize  $\gamma$ , the retraction map  $\mathrm{Rtr}$  defines a point  $\mathrm{Rtr}_U(-\gamma \Delta)$  which is guaranteed to lie on the manifold  $\mathcal{M}$ . Roughly speaking, the retraction  $\mathrm{Rtr}_U(\cdot)$  approximates the geodesic curve through  $U$  along the input tangential direction. For a formal definition of retractions, we refer the readers to (Absil et al., 2007, Section 4.1). In this paper, we focus on the following two commonly used retractions for Stiefel manifolds. The first one is the QR decomposition-based retraction

$$
\operatorname {R t r} _ {U} ^ {\mathrm {q f}} (\Delta) = \operatorname {q f} (U + \Delta), \quad U \in \mathcal {M}, \Delta \in T _ {U} \mathcal {M},
$$

where  $\mathrm{qf}(\cdot)$  is the Q-factor in the QR decomposition. The second one is the polar decomposition-based retraction

$$
\operatorname {R t r} _ {U} ^ {\text {p o l a r}} (\Delta) = (U + \Delta) \left(I _ {d - k} + \Delta^ {\top} \Delta\right) ^ {- \frac {1}{2}}, \quad U \in \mathcal {M}, \Delta \in T _ {U} \mathcal {M}. \tag {8}
$$

# 4.4 ALGORITHM AND CONVERGENCE GUARANTEES

Associated with any choice of retraction Rtr is a concrete instantiation of the Riemannian subgradient descent algorithm for our problem (7), which is presented in Algorithm 1.

# Algorithm 1 Riemannian Subgradient Descent for (7)

1: Input: An initial point  $U_{0}$ , a number of iterations  $\tau$  and a retraction  $\mathrm{Rtr}: (U, \Delta) \mapsto \mathrm{Rtr}_U(\Delta)$ .  
2: for  $t = 0,1,\ldots ,\tau -1$  do  
3: Find  $a_{t} \triangleq \arg \max_{a \in \{0,1\}} \{F_{a}(U_{t})\}$ .  
4: Compute the Riemannian subgradient  $\Delta_t = \mathrm{grad}F(U_t)$  using the formula

$$
\Delta_ {t} = (I - U _ {t} U _ {t} ^ {\top}) \left(\frac {\theta_ {a _ {t}}}{\sqrt {\langle U _ {t} U _ {t} ^ {\top} , \hat {M} _ {a _ {t}} \rangle}} \hat {M} _ {a _ {t}} U _ {t} + \frac {\vartheta_ {a _ {t} ^ {\prime}}}{\sqrt {\langle U _ {t} U _ {t} ^ {\top} , \hat {M} _ {a _ {t} ^ {\prime}} \rangle}} \hat {M} _ {a _ {t} ^ {\prime}} U _ {t} + 2 C _ {a _ {t}} U _ {t}\right).
$$

5: Set  $U_{t + 1} = \mathrm{Rtr}_{U_t}(-\gamma_t\Delta_t)$ , where the step-size  $\gamma_t \equiv \frac{1}{\sqrt{\tau + 1}}$  is constant.  
6: end for  
7: Output:  $U_{\tau}$ .

The specific choice of the stepsizes  $\gamma_{t}$  is motivated by the theoretical results of (Li et al., 2019).

We now study the convergence guarantee of Algorithm 1. The following lemma shows that the objective function  $F$  is Lipschitz continuous (with respect to the Riemannian metric on the Stiefel manifold  $\mathcal{M}$ ) with an explicit Lipschitz constant  $L$ .

Lemma 4.2 (Lipschitz continuity). The function  $F$  is  $L$ -Lipschitz continuous on  $\mathcal{M}$ , where  $L > 0$  is given by

$$
\begin{array}{l} L \triangleq \max  \left\{\theta_ {0} \frac {\sigma_ {\operatorname* {m a x}} (\hat {M} _ {0})}{\sqrt {\sigma_ {\operatorname* {m i n}} (\hat {M} _ {0})}}, \theta_ {1} \frac {\sigma_ {\operatorname* {m a x}} (\hat {M} _ {1})}{\sqrt {\sigma_ {\operatorname* {m i n}} (\hat {M} _ {1})}}, \vartheta_ {0} \frac {\sigma_ {\operatorname* {m a x}} (\hat {M} _ {0})}{\sqrt {\sigma_ {\operatorname* {m i n}} (\hat {M} _ {0})}}, \vartheta_ {1} \frac {\sigma_ {\operatorname* {m a x}} (\hat {M} _ {1})}{\sqrt {\sigma_ {\operatorname* {m i n}} (\hat {M} _ {1})}}, \right. \tag {9} \\ \left. 2 \sqrt {d - k} \sigma_ {\max} (C _ {0}), 2 \sqrt {d - k} \sigma_ {\max} (C _ {1}) \right\}. \\ \end{array}
$$

We now proceed to show that Algorithm 1 enjoys a sub-linear convergence rate. To state the result, we define the Moreau envelope

$$
F _ {\mu} (U) \triangleq \min  _ {U ^ {\prime} \in \mathcal {M}} \left\{F (U ^ {\prime}) + \frac {1}{2 \mu} \| U ^ {\prime} - U \| _ {F} ^ {2} \right\},
$$

where  $\| \cdot \| _F$  denotes the Frobenius norm of a matrix. Also, to measure the progress of the algorithm, we need to introduce the proximal mapping on the Stiefel manifold (Li et al., 2019):

$$
\operatorname{prox}_{\mu F}(U)\in \operatorname *{arg  min}_{U^{\prime}\in \mathcal{M}}\left\{F(U^{\prime}) + \frac{1}{2\mu}\left\| U^{\prime} - U\right\|_{F}^{2}\right\} .
$$

From Li et al. (2019, Equaton (22)), we have that

$$
\| \operatorname {g r a d} F (U) \| _ {F} \leq \frac {\left\| \operatorname {p r o x} _ {\mu F} (U) - U \right\| _ {F}}{\mu} \triangleq \operatorname {g a p} _ {\mu} (U).
$$

Therefore, the number  $\mathrm{gap}_{\mu}(U)$  is a good candidate to quantify the progress of optimization algorithms for solving problem (7).

Theorem 4.3 (Convergence guarantee). Let  $\{U_t\}_{t=1,\dots,\tau}$  be the sequence of iterates generated by Algorithm 1. Suppose that  $\mu = 1/4L$ , where  $L$  is the Lipschitz constant of  $F$  in (9). Then, we have

$$
\min _ {t = 0, \ldots , \tau} \mathsf {g a p} _ {\mu} (U _ {t}) \leq \frac {2 \sqrt {F _ {\mu} (U _ {0}) - \min _ {U} F _ {\mu} (U) + 2 L ^ {3} (L + 1)}}{(\tau + 1) ^ {1 / 4}}.
$$

# 5 NUMERICAL EXPERIMENTS

We compare our proposed method, denoted RFPCA, against two state-of-the-art methods for fair PCA: 1) FairPCA from Samadi et al. (2018)<sup>4</sup>, and 2) CFPCA from Olfat & Aswani (2019)<sup>5</sup> with both cases: only mean constraint, and both mean and covariance constraints. We consider a wide variety of datasets from UC Irvine's online Machine Learning Repository (Dua & Graff, 2017) with ranging sample sizes and number of features. Further details about the datasets can be found in Appendix B. The code for all experiments is available in supplementary materials.

We include here some details about the hyper-parameters that we search in the cross-validation steps.

- RFPCA. We notice that the neighborhood size  $\varepsilon_{a}$  should be inversely proportional to the size of subgroup  $a$ . Indeed, a subgroup with large sample size is likely to have more reliable estimate of the moment information. Then we parameterize the neighborhood size  $\varepsilon_{a}$  by a common scalar  $\alpha$ , and we have  $\varepsilon_{a} = \alpha / N_{a}$ , where  $N_{a}$  is the number of samples in group  $a$ . We search  $\alpha \in \{0.05, 0.1, 0.15\}$  and  $\lambda \in \{0., 0.5, 1., 1.5, 2.0, 2.5\}$ . For better convergence quality, we set the number of iteration for our subgradient descent algorithm to  $\tau = 1000$  and also repeat the Riemannian descent for 20 randomly generated initial point  $U_{0}$ .  
- FairPCA. According to Samadi et al. (2018), we only need tens of iterations for the multiplicative weight algorithm to provide good-quality solution; however, to ensure a fair comparison, we set the number of iterations to be 1000 for the convergence guarantee. We search the learning rate  $\eta$  of the algorithm from set of 17 values evenly spaced in [0.25, 4.25] and  $\{0.1\}$ .  
- CFPCA. Followed Olfat & Aswani (2019), for the mean-constrained version of CFPCA, we search  $\delta$  from  $\{0., 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9\}$ , and for both mean and covariance constrained version, we fix  $\delta = 0$  while searching  $\mu$  in  $\{0.0001, 0.001, 0.01, 0.05, 0.5\}$ .

Trade-offs. First, we examine the trade-off between the total reconstruction error and the gap between the subgroup error. In this experiment, we only compare our model with FairPCA and CFPCA mean-constraint version. We plot a pareto curve for each of them over the two criteria with different hyper-parameters (hyper-parameters test range are mentioned above). The whole datasets are used for training and evaluation. The results averaged over 5 runs are shown in Figure 2.

In testing methods with different principal components, we first split each dataset into training set and test set with equal size (50% each), the projection matrix of each method is learned from training set and tested over both sets. In this case, we only compare our method with traditional PCA and FairPCA method. We fix one set hyper-parameters for each method. For FairPCA, we set  $\eta = 0.1$  and for RFPCA we set  $\alpha = 0.15$ ,  $\lambda = 0.5$ , others hyper-parameters are kept as discussed before. The results are averaged over 5 different splits. Figure 3 shows the consistence of our method performing fair projections over different values of  $k$ . Our method (cross) exhibits smaller gap of subgroup errors. More results can be found in Appendix C.2.

![](images/fbe502bc4893cdfc01daafd74c6564ded552c28db748b0db70b640d4e5496451.jpg)  
Figure 2: Pareto curves on Default Credit dataset (all data) with 3 principal components

![](images/2559635d80ec3289eab995f65d043061ba853f259dc537d847a6e38b0d14f0d5.jpg)  
Figure 3: Subgroup average error with different  $k$  on Biodeg dataset (Out-of-sample).

Cross-validations. Next, we report the performance of all methods based on four criteria: absolute difference between average reconstruction error between groups (ABDiff.), average reconstruction error of all data (ARE.), and the fairness criterion defined by Olfat & Aswani (2019) with respect to a linear SVM's classifier family  $(\triangle \mathcal{F}_{Lin})^6$ . Due to the space constraint, we only include the first two criteria in the main text, see Appendix 4 for full results. In Each dataset, one feature is selected as a sensitive attribute, other features is used as the input for algorithms. To emphasize the generalization capacity of each algorithm, we split each dataset into a training set and a test set with ratio of  $30\% - 70\%$  respectively, and only extract top three principal components from the training set. We find the best hyper-parameters by 3-fold cross validation, and prioritize the one giving minimum value of the summation (ABDiff. + ARE.). The results are averaged over 10 different training-testing splits. We report the performance on both training set (In-sample data) and test set (Out-of-sample data). The details results for Out-of-sample data is given in Table 1 while one for In-sample data is reported in the appendix at Table 3.

Results. Our proposed RFPCA method outperforms on 10 out of 14 datasets in terms of the subgroup error gap ABDiff, and 8 out of 14 with the totall error ARE. criterion. There are 4 datasets that RFPCA gives the best results for both criteria, and for the remaining datasets, RFPCA has small performance gaps compared with the best method.

Table 1: Out-of-sample errors on real datasets. Bold indicates the lowest error for each dataset.  

<table><tr><td rowspan="2">Dataset</td><td colspan="2">RFPCA</td><td colspan="2">FairPCA</td><td colspan="2">CFPCA-Mean Con.</td><td colspan="2">CFPCA - Both Con.</td></tr><tr><td>ABDiff.</td><td>ARE.</td><td>ABDiff.</td><td>ARE.</td><td>ABDiff.</td><td>ARE.</td><td>ABDiff.</td><td>ARE.</td></tr><tr><td>Default Credit</td><td>0.9483</td><td>10.3995</td><td>1.4401</td><td>10.4439</td><td>0.9367</td><td>10.9451</td><td>3.3359</td><td>22.0310</td></tr><tr><td>Biodeg</td><td>23.0066</td><td>33.8571</td><td>27.5159</td><td>34.6184</td><td>29.1728</td><td>37.6052</td><td>37.9533</td><td>50.7090</td></tr><tr><td>E. Coli</td><td>1.1500</td><td>1.7210</td><td>1.5280</td><td>2.4799</td><td>1.1005</td><td>2.9466</td><td>5.1275</td><td>5.6674</td></tr><tr><td>Energy</td><td>0.0125</td><td>0.2238</td><td>0.0138</td><td>0.2225</td><td>0.1229</td><td>2.7318</td><td>0.1001</td><td>7.9511</td></tr><tr><td>German Credit</td><td>2.0588</td><td>43.9032</td><td>1.3670</td><td>44.0064</td><td>1.7845</td><td>43.9648</td><td>1.4955</td><td>49.5014</td></tr><tr><td>Image</td><td>0.7522</td><td>6.0199</td><td>1.6129</td><td>10.2616</td><td>1.1499</td><td>14.3725</td><td>4.7013</td><td>19.3356</td></tr><tr><td>Letter</td><td>0.1712</td><td>7.4176</td><td>1.2489</td><td>7.4470</td><td>0.4427</td><td>8.7445</td><td>0.5743</td><td>15.1779</td></tr><tr><td>Magic</td><td>1.8314</td><td>3.9094</td><td>2.9405</td><td>3.3815</td><td>5.5790</td><td>4.2105</td><td>8.7810</td><td>9.0064</td></tr><tr><td>Parkinsons</td><td>0.3273</td><td>5.0597</td><td>0.8678</td><td>4.9044</td><td>3.3804</td><td>5.7260</td><td>18.3312</td><td>19.7001</td></tr><tr><td>SkillCraft</td><td>0.7669</td><td>8.2828</td><td>0.7771</td><td>8.2494</td><td>1.0283</td><td>9.9484</td><td>1.2849</td><td>15.9751</td></tr><tr><td>Statlog</td><td>0.0838</td><td>3.0998</td><td>0.3356</td><td>7.9734</td><td>0.4476</td><td>10.8263</td><td>13.8437</td><td>35.8268</td></tr><tr><td>Steel</td><td>1.1472</td><td>12.5944</td><td>1.2208</td><td>12.3096</td><td>4.8710</td><td>16.4015</td><td>3.8084</td><td>25.8953</td></tr><tr><td>Taiwan Credit</td><td>0.5523</td><td>10.9845</td><td>0.5710</td><td>10.9415</td><td>0.5744</td><td>13.0437</td><td>0.9535</td><td>21.8963</td></tr><tr><td>Wine Quality</td><td>0.6359</td><td>4.2801</td><td>0.3046</td><td>6.0936</td><td>1.5020</td><td>6.1118</td><td>3.0451</td><td>10.1001</td></tr></table>

# REFERENCES

P.-A. Absil, R. Mahony, and R. Sepulchre. Optimization Algorithms on Matrix Manifolds. Princeton University Press, 2007.  
Alekh Agarwal, Miroslav Dudík, and Zhiwei Steven Wu. Fair regression: Quantitative definitions and reduction-based algorithms. In International Conference on Machine Learning, pp. 120-129. PMLR, 2019.  
Naveed Akhtar and Ajmal Mian. Threat of adversarial attacks on deep learning in computer vision: A survey. IEEE Access, 6:14410-14430, 2018.  
Solon Barocas, Moritz Hardt, and Arvind Narayanan. Fairness and machine learning. fairmlbook. org, 2019, 2018.  
Richard Berk, Hoda Heidari, Shahin Jabbari, Michael Kearns, and Aaron Roth. Fairness in criminal justice risk assessments: The state of the art. Sociological Methods & Research, 50(1):3-44, 2021.  
Alex Beutel, Jilin Chen, Zhe Zhao, and Ed H Chi. Data decisions and theoretical implications when adversarially learning fair representations. arXiv preprint arXiv:1707.00075, 2017.  
Flavio P Calmon, Dennis Wei, Bhanukiran Vinzamuri, Karthikeyan Natesan Ramamurthy, and Kush R Varshney. Optimized pre-processing for discrimination prevention. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pp. 3995-4004, 2017.  
Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy (SP), pp. 39-57. IEEE, 2017.  
Anirban Chakraborty, Manaar Alam, Vishal Dey, Anupam Chattopadhyay, and Debdeep Mukhopadhyay. Adversarial attacks and defences: A survey. arXiv preprint arXiv:1810.00069, 2018.  
Alexandra Chouldechova. Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. *Big Data*, 5(2):153–163, 2017.  
Erick Delage and Yinyu Ye. Distributionally robust optimization under moment uncertainty with application to data-driven problems. Operations Research, 58(3):595-612, 2010.  
Michele Donini, Luca Oneto, Shai Ben-David, John S Shawe-Taylor, and Massimiliano Pontil. Empirical risk minimization under fairness constraints. In Advances in Neural Information Processing Systems, pp. 2791-2801, 2018.  
Dheeru Dua and Casey Graff. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ml.  
Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through awareness. In Proceedings of the 3rd innovations in theoretical computer science conference, pp. 214-226, 2012.  
Michael Feldman, Sorelle A Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubramanian. Certifying and removing disparate impact. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 259-268, 2015.  
C.R. Givens and R.M. Shortt. A class of Wasserstein metrics for probability distributions. The Michigan Mathematical Journal, 31(2):231-240, 1984.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Moritz Hardt, Eric Price, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. In Advances in Neural Information Processing Systems 29, pp. 3315-3323, 2016a.  
Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. Advances in neural information processing systems, 29:3315-3323, 2016b.

Tito Homem-de Mello and Guzin Bayraksan. Monte Carlo sampling-based methods for stochastic optimization. Surveys in Operations Research and Management Science, 19(1):56-85, 2014.  
Harold Hotelling. Analysis of a complex of statistical variables into principal components. Journal of educational psychology, 24(6):417, 1933.  
Faisal Kamiran and Toon Calders. Data preprocessing techniques for classification without discrimination. Knowledge and Information Systems, 33(1):1-33, 2012.  
Toshihiro Kamishima, Shotaro Akaho, Hideki Asoh, and Jun Sakuma. Fairness-aware classifier with prejudice remover regularizer. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 35-50. Springer, 2012.  
Daniel Kuhn, Peyman Mohajerin Esfahani, Viet Anh Nguyen, and Soroosh Shafieezadeh-Abadeh. Wasserstein distributionally robust optimization: Theory and applications in machine learning. In Operations Research & Management Science in the Age of Analytics, pp. 130-166. INFORMS, 2019.  
Matt J Kusner, Joshua R Loftus, Chris Russell, and Ricardo Silva. Counterfactual fairness. arXiv preprint arXiv:1703.06856, 2017.  
Xiao Li, Shixiang Chen, Zengde Deng, Qing Qu, Zhihui Zhu, and Anthony Man Cho So. Weakly convex optimization over Stiefel manifold using Riemannian subgradient-type methods. arXiv preprint arXiv:1911.05047, 2019.  
Zachary Lipton, Julian McAuley, and Alexandra Chouldechova. Does mitigating ML's impact disparity require treatment disparity? In Advances in Neural Information Processing Systems, pp. 8125-8135, 2018.  
David Madras, Elliot Creager, Toniann Pitassi, and Richard Zemel. Learning adversarily fair and transferable representations. In International Conference on Machine Learning, pp. 3384-3393. PMLR, 2018.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.  
Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey on bias and fairness in machine learning. ACM Computing Surveys (CSUR), 54(6):1-35, 2021.  
Hongseok Namkoong and John C Duchi. Variance-based regularization with convex objectives. In Advances in Neural Information Processing Systems 30, pp. 2971-2980, 2017.  
Viet Anh Nguyen. Adversarial Analytics. PhD thesis, Ecole Polytechnique Fédérale de Lausanne, 2019.  
Matt Olfat and Anil Aswani. Convex formulations for fair principal component analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 663-670, 2019.  
Karl Pearson. Liii. on lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin philosophical magazine and journal of science, 2(11):559-572, 1901.  
Hamed Rahimian and Sanjay Mehrotra. Distributionally robust optimization: A review. arXiv preprint arXiv:1908.05659, 2019.  
Samira Samadi, Uthaipon Tantipongpipat, Jamie H Morgenstern, Mohit Singh, and Santosh Vempala. The price of fair PCA: One extra dimension. In Advances in Neural Information Processing Systems, pp. 10976-10987, 2018.  
James E Smith and Robert L Winkler. The optimizer's curse: Skepticism and postdecision surprise in decision analysis. Management Science, 52(3):311-322, 2006.  
Uthaipon Tantipongpipat, Samira Samadi, Mohit Singh, Jamie Morgenstern, and Santosh Vempala. Multi-criteria dimensionality reduction with applications to fairness. arXiv preprint arXiv:1902.11281, 2019.

Sahil Verma and Julia Rubin. Fairness definitions explained. In 2018 IEEE/ACM International Workshop on Software Fairness (fairware), pp. 1-7. IEEE, 2018.  
Dennis Wei, Karthikeyan Natesan Ramamurthy, and Flavio du Pin Calmon. Optimized score transformation for fair classification. arXiv preprint arXiv:1906.00066, 2019.  
Gad Zalcberg and Ami Wiesel. Fair principal component analysis and filter design. IEEE Transactions on Signal Processing, 69:4835-4842, 2021.  
Rich Zemel, Yu Wu, Kevin Swersky, Toni Pitassi, and Cynthia Dwork. Learning fair representations. In International Conference on Machine Learning, pp. 325-333. PMLR, 2013.  
Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. Mitigating unwanted biases with adversarial learning. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, pp. 335-340, 2018.
