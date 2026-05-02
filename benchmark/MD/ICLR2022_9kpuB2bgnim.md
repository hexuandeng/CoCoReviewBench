# HUBER ADDITIVE MODELS FOR NON-STATIONARY TIME SERIES ANALYSIS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Sparse additive models have shown promising flexibility and interpretability in processing time series data. However, existing methods usually assume the time series data to be stationary and the innovation is sampled from a Gaussian distribution. Both assumptions are too stringent for heavy-tailed and non-stationary time series data that frequently arise in practice, such as finance and medical fields. To address these problems, we propose an adaptive sparse Huber additive model for robust forecasting and inference (e.g., Granger causal discovery) in both non-Gaussian data and (non)stationary data. In theory, the generalization bounds of our estimator are established for both stationary and non-stationary time series data, which are independent of the widely used mixing conditions in learning theory of dependent observations. Moreover, the error bound for non-stationary time series contains a discrepancy measure for the shifts of the data distributions over time. Such a discrepancy measure can be estimated empirically and used as a penalty in our method. Experimental results on both synthetic and real-world benchmark datasets validate the effectiveness of the proposed method.

# 1 INTRODUCTION

Additive models have become one of the most powerful tools for time series analysis due to the exemplary monograph (Stone, 1985; Hastie & Tibshirani, 1990) and companion software (Chambers & Hastie, 1992). For the past two decades, the growing importance of algorithmic flexibility and interpretability motivates the development of various additive models along with theory exploration (Huang & Yang, 2004; Wang & Yang, 2007; Chu & Glymour, 2008; Song & Yang, 2010; Yang et al., 2018) and practical applications (Dominici et al., 2002; Wang & Brown, 2011; Ravindra et al., 2019; Bussmann et al., 2020). Although the aforementioned works have shown promising behaviours, the methods proposed in these works require some strict assumptions on the stochastic process, e.g., Gaussian innovation, mixing dependency or stationary distribution. However, these formerly mentioned approaches may have degraded performance when facing heavy-tailed and non-stationary time series (Raya & Murad, 1998; Qiu et al., 2015).

A number of attempts have been made to relax the strict assumptions on the stochastic process. Qiu et al. (2015) develops an elliptical vector autoregressive model for estimating heavy-tailed stationary processes with parametric convergence analysis that reduces the influence of non-Gaussian innovation. Moreover, under stationarity and decaying  $\beta$ -mixing condition, Wong et al. (2020) derive nonasymptotic estimation error of lasso without assuming special parametric form of the data generating process. Stationarity and mixing conditions are commonly adopted in many previous studies of non-i.i.d settings (Doukhan, 1994), which ensure the use of traditional complexity tools (e.g., Rademacher complexity (Bartlett & Mendelson, 2002; Mohri & Rostamizadeh, 2009), covering number (Ron, 2000; Zhou, 2002; Guo & Shi, 2011) and stability (Mohri & Rostamizadeh, 2010)) for theoretical exploration. However, the mixing and stationary conditions are too strict and not always valid. For instance, long memory models (e.g., ARFIMA) may not be mixing (Baillie, 1996; Kuznetsov & Mohri, 2020). To relax the mixing and stationary conditions, Adams & Nobel (2010) prove asymptotic guarantees for stationary ergodic sequences. Alekh & C. (2013) give generalization bounds for asymptotically stationary (mixing) processes in the case of stable on-line learning algorithms. Kuznetsov & Mohri (2014) establish learning guarantees for fully non-stationary and mixing processes. Recently, Kuznetsov & Mohri (2020) provide data-dependent generalization bounds for the non-stationary and non-mixing stochastic processes.

However, both theoretically and practically, the exploration on robust additive models for (non)stationary time series analysis is still limited. In this paper, we propose a class of sparse Huber additive models with statistical guarantees. Our main contributions are summarized as below:

- Algorithm design: We first propose a novel sparse Huber additive model (SpHAM) by integrating Huber loss and sparsity-inducing  $\ell_{2,1}$ -norm regularizer into an additive data-dependent hypothesis space. For stationary time series, this proposed algorithm can achieve robust forecasting and satisfactory inference (e.g., variable selection and Granger causal discovery) simultaneously, even under non-Gaussian innovation, e.g., heavy-tailed innovation and outliers.  
- Theoretical guarantees: We establish the function approximation error bounds of SpHAM by developing error decomposition technique and employing sequential Rademacher complexity (Rakhlin et al., 2010; 2015; Kuznetsov & Mohri, 2020). With properly selected scale parameter, the theoretical findings indicate that: a) for stationary time series, the convergence rate  $O(n^{-\frac{1}{2}})$  can be achieved, even when the innovation is non-Gaussian distribution. In fact, this error bound appears to be novel even in the stationary case because the mixing conditions are not imposed here; b) for non-stationary time series, the approximation error of our method is bounded by a discrepancy measure for the shifts of the data distributions along with time (see Theorems 2-3 for more details), which inspires us to further propose an adaptive SpHAM by penalizing such a discrepancy measure.  
- Optimization and empirical evaluations: The proposed SpHAM and adaptive SpHAM can be implemented efficiently by Difference of Convex programming (DC programming) (Tao & An, 1998) and Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) (Beck & Teboulle, 2009). Experimental results on both synthetic and real-world benchmark CauseMe (Runge et al., 2019) validate the effectiveness of the proposed method.

Related works: We compare our method with related works and show improvement from the following aspects (the characteristics of our method are in bold):

a) SpHAM for non-i.i.d data vs. Huber algorithm for i.i.d data: Huber algorithms play an important role in robust estimation and inference (Huber, 1964; Huber & Ronchetti, 2009; Loh, 2017; Feng & Wu, 2020). However, the most existing methods concern i.i.d data. Our method fills the gap in the investigation of whether Huber additive models work when data are non-i.i.d with non-Gaussian innovation.  
b) Robust forecasting vs. non-robust forecasting: Under Gaussian innovation, there has been a lot of studies on sparse additive models (Wang & Yang, 2007; Chu & Glymour, 2008; Yang et al., 2018; Bussmann et al., 2020). Differently, our method benefits from the advantageous properties of Huber algorithms in robust forecasting, and the corresponding theoretical findings in Theorem 1 verify the convergence of our method in weak moment condition.  
c) Nonstationary data vs. stationary data: In order to develop widely used complexity tools (e.g., Rademacher complexity and covering number) for non-i.i.d case, early proposed methods assume that the data satisfies (strict) stationary or various mixing conditions (Doukhan, 1994; Qiu et al., 2015; Yang et al., 2018). To overcome these limitations, our method employs sequential Rademacher complexity in theoretical analysis, and the corresponding theoretical results inspire us to modify our method to deal with non-stationary data.  
d) Function approximation analysis vs. generalization analysis: Recently, Kuznetsov & Mohri (2020) support a class of methods for nonstationary time series with generalization error analysis. In contrast, we are chiefly concerned about the function approximation error analysis, since the convergence of generalization cannot imply the convergence of function approximation for Huber loss based algorithms (Sun et al., 2019; Feng & Wu, 2020).  
e) Interpretable vs. non-interpretable: Compared with the smoothness-inducing regularized methods Kuznetsov & Mohri (2020); Feng & Wu (2020), the sparsity-inducing regularizer ensures that our method can screen out informative variables or detect the Granger causal among variables.

To better highlight the novelty of our method, Table 1 summarizes the algorithmic properties of our method and other related works, e.g., Sparse additive models for time series (TS-SpA) (Yang et al.,

Table 1: The properties of related methods  

<table><tr><td></td><td>TS-SpAM</td><td>Huber Reg.</td><td>DBF</td><td>R-Dantzig</td><td>Ours</td></tr><tr><td>Hypothesis space</td><td>Additive Spline-based</td><td>Hilbert</td><td>Kernel-based</td><td>Linear</td><td>Additive Kernel-based</td></tr><tr><td>Loss</td><td>Squared</td><td>Huber</td><td>Squared</td><td>Quantile-based</td><td>Huber</td></tr><tr><td>Mixing</td><td>Yes</td><td>i.i.d.</td><td>No</td><td>Yes</td><td>No</td></tr><tr><td>Stationarity</td><td>Yes</td><td>i.i.d.</td><td>No</td><td>Yes</td><td>No</td></tr><tr><td>Robustness</td><td>No</td><td>Yes</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Sparsity</td><td>Yes</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr></table>

2018), Huber regression (Feng & Wu, 2020), Discrepancy-based Forecasting (DBF) (Kuznetsov & Mohri, 2020) and Robust Dantzig selector-type estimator (R-Dantzig) (Qiu et al., 2015).

# 2 PRELIMINARY

Let  $\{Z^t\}_{t = -\infty}^{\infty}$  be a stochastic time series with time index  $t$ , where variable  $Z^{t} = (X^{t},Y^{t})$  takes values in the compact input space  $\mathcal{X}\subset \mathbb{R}^p$  and the output space  $\mathcal{V}\subset \mathbb{R}$ . We consider a common nonparametric model

$$
Y ^ {t} = f ^ {*} \left(X ^ {t}\right) + \varepsilon^ {t}, \mathbb {E} \left(\varepsilon^ {t}\right) = 0, \tag {1}
$$

where  $f^{*}(\cdot)$  is the ground truth function, and the innovation  $\varepsilon^t$  is i.i.d. across time  $t \in \mathbb{Z}$ . For the sake of simplicity, we denote  $\rho^t$  and  $\rho_{\mathcal{X}}^t$  as the jointed distribution of  $(X^t, Y^t)$  and the corresponding marginal distribution with respect to  $X^t$ , respectively. This setup actually covers a larger number of scenarios commonly used in practice. For instance, the case  $X^t$  contains  $p$  lagged values of  $Y^t$  (e.g.,  $X^t = Y^{(t - p)} \times \dots \times Y^{(t - 1)}$ ) corresponds to the  $p$ -order autoregressive models. Moreover, this case can be viewed as a vector autoregressive model in the sense that input  $X^t$  includes the information of multiple variables at different lagged times.

Although such a nonparametric model (1) makes very few assumptions on data generation, the related nonparametric algorithms suffer so-called "curse of dimensionality", see Fan & Gijbels (1996) for further discussion. An effective strategy for solving this problem is additive model. Usually, the additive structure is obtained by decomposing the input space  $\mathcal{X} \in \mathbb{R}^p$  into  $\mathcal{X} = \mathcal{X}_1 \times \ldots \times \mathcal{X}_p$ . Under the assumption that the ground truth admits an additive structure  $f^* = \sum_{j=1}^{p} f_j^*$ , the additive model can be defined as

$$
Y ^ {t} = f _ {1} ^ {*} \left(X _ {1} ^ {1}\right) + \dots + f _ {p} ^ {*} \left(X _ {p} ^ {t}\right) + \varepsilon^ {t}, \tag {2}
$$

where each component  $f_{j}^{*}: \mathcal{X}_{j} \to \mathbb{R}$  is a smooth function. In linear time series analysis, a weak stationarity condition (i.e., the first two moments of time series are time invariant) is preferred (Han et al., 2015; Qiu et al., 2015). In contrast, strict stationarity is primarily used if our focus is on nonlinear relationships (Fan & Yao, 2005).

Definition 1. A stochastic process  $\{Z^t\}_{t = -\infty}^{\infty}$  is strictly stationary if  $(Z^{1},\dots,Z^{t})$  and  $(Z^{1 + k},\dots,Z^{t + k})$  have the same joint distributions for any  $t\in \mathbb{Z}$  and  $k\in \mathbb{Z}$ .

Note that, if not otherwise stated, the stationarity in this paper refers to strict stationarity. Suppose that we are given  $T$  size time series data  $\{(x^t,y^t)\}_{t = 1}^T\in \mathcal{Z}^T$  which are drawn from an additive data-generating model (2). Under stationarity condition and zero-mean Gaussian innovation with finite variance, the most existing methods to learn  $f^{*}$  are usually integrating squared loss and a smoothness- or sparsity-inducing regularizer  $\Omega (\cdot)$  into a Structural Risk Minimization scheme:

$$
\min _ {f \in \mathcal {H}} \sum_ {t = 1} ^ {T} (y ^ {t} - \sum_ {j = 1} ^ {p} f _ {j} (x _ {j} ^ {t})) ^ {2} + \Omega (f),
$$

where  $\mathcal{H} := \{f_1 + \ldots + f_p : f_j \in \mathcal{H}_j, j = 1, \ldots, p\}$  is an additive hypothesis space. Some existing works construct the additive hypothesis space  $\mathcal{H}$  via spline-based basis (Yang et al., 2018) and composite neural net (Bussmann et al., 2020).

However, when facing heavy-tailed innovation, these methods may have degraded performance due to the amplification of the squared loss to large residuals. In statistic learning community, as one commonly used robust statistic, Huber loss is defined as

$$
\ell_ {\sigma} \left(f \left(x ^ {t}\right) - y ^ {t}\right) = \left\{ \begin{array}{l l} \left(f \left(x ^ {t}\right) - y ^ {t}\right) ^ {2}, & \text {i f} | f \left(x ^ {t}\right) - y ^ {t} | <   \sigma \\ 2 \sigma | f \left(x ^ {t}\right) - y ^ {t} | - \sigma^ {2}, & \text {i f} | f \left(x ^ {t}\right) - y ^ {t} | \geq \sigma , \end{array} \right. \tag {3}
$$

where  $\sigma$  is a positive hyper-parameter. Note that in the previous studies (Huber & Ronchetti, 2009; Loh, 2017), the hyper-parameter  $\sigma$  is set to be fixed according to the  $95\%$  asymptotic efficiency rule. However, Huber regression with a fixed scale parameter may not be able to learn the ground truth when the noise is asymmetric, as argued recently in Feng & Wu (2020); Sun et al. (2019). In this paper, we choose the scale parameter  $\sigma$  by relating it to the moment condition of the noise distribution, and the sample size so that the resulting regression estimator can asymptotically converge to the ground truth function.

# 3 METHOD

In this section, we first propose a sparse Huber additive model (SpHAM). Secondly, we conduct some assessments of SpHAM from a statistical learning viewpoint. Finally, with the help of our theoretical findings, we give an adaptive SpHAM for non-stationary time series forecasting.

# 3.1 SPARSE HUBER ADDITIVE MODELS

In this paper, we choose reproducing kernel Hilbert space (RKHS)  $\mathcal{H}_{K_j}, j = 1,\dots,p$ , to form the additive hypothesis space  $\mathcal{H}$ , where each  $\mathcal{H}_{K_j}$  is associated with a symmetric and positive semi-definite Mercer kernel  $K_{j}:\mathcal{X}_{j}\times \mathcal{X}_{j}\to \mathbb{R}$ . Then an additive RKHS is defined as

$$
\mathcal {H} _ {K} = \left\{f _ {1} + \dots + f _ {p}: f _ {j} \in \mathcal {H} _ {K _ {j}}, j = 1, \dots , p \right\} \tag {4}
$$

with kernel norm  $\| f\| _K^2 = \inf \{\| f\|_{K_1}^2 +\ldots +\| f\|_{K_p}^2\}$ . By integrating the Huber loss (3) and kernel norm into an additive RKHS, the kernel-based Huber additive model can be formulated as

$$
\hat {f} _ {\eta} = \sum_ {j = 1} ^ {p} \hat {f} _ {\eta , j} = \underset {f = \sum_ {j = 1} ^ {p} f _ {j}, f _ {j} \in \mathcal {H} _ {K _ {j}}} {\arg \min } \left\{\sum_ {t = 1} ^ {T} \ell_ {\sigma} \left(y ^ {t} - \sum_ {j = 1} ^ {p} f _ {j} \left(x _ {j} ^ {t}\right)\right) + \eta \sum_ {j = 1} ^ {p} \tau_ {j} \| f _ {j} \| _ {K _ {j}} ^ {2} \right\}, \tag {5}
$$

where  $\eta$  is positive regularization parameter and  $\tau_{j}$  is the weight for  $j$ -th kernel norm. The representer theorem (Wahba, 1990) ensures that  $\hat{f}_{\eta}$  can be represented as

$$
\hat {f} _ {\eta} = \sum_ {j = 1} ^ {p} \sum_ {t = 1} ^ {T} \alpha_ {t j} ^ {\eta} K _ {j} (x _ {j} ^ {t}, \cdot), \alpha_ {t j} ^ {\eta} \in \mathbb {R}.
$$

To offer the model with sparsity, we consider the following sparsity-inducing penalty

$$
\Omega (f) := \inf  \left\{\sum_ {j = 1} ^ {p} \tau_ {j} \| \alpha_ {j} \| _ {2}: f = \sum_ {j = 1} ^ {p} \sum_ {t = 1} ^ {T} \alpha_ {t j} K _ {j} \left(x _ {j} ^ {t}, \cdot\right) \right\}
$$

in an additive data dependent hypothesis space

$$
\mathcal {H} _ {Z} = \{f = \sum_ {j = 1} ^ {p} \sum_ {t = 1} ^ {T} \alpha_ {t j} K _ {j} (x _ {j} ^ {t}, \cdot): \alpha_ {t j} \in \mathbb {R} \}.
$$

Clearly, this data-dependent hypothesis spaces is the additive RKHS depending on the observations  $\{(x^t,y^t)\}_{t = 1}^T$ . Then the SpHAM can be formulated as

$$
\hat {f} = \underset {f \in \mathcal {H} _ {Z}} {\arg \min} \{\frac {1}{T} \sum_ {t = 1} ^ {T} \ell_ {\sigma} (y ^ {t} - \sum_ {j = 1} ^ {p} f _ {j} (x _ {j} ^ {t})) + \lambda \Omega (f) \}.
$$

Denote  $\mathbf{K}_j^t = (K_j(x_j^1,x_j^t),\dots,K_j(x_j^T,x_j^t))'\in \mathbb{R}^T$ $\alpha = (\alpha_1',\dots,\alpha_p')'\in \mathbb{R}^{Tp}$  and  $\alpha_{j} = (\alpha_{1j},\dots,\alpha_{Tj})'\in \mathbb{R}^{T}$ , where  $\alpha_{j}'$  here refers to the transposition of  $\alpha_{j}$  for avoiding confusion with time  $T$ . The SpHAM can be represented as

$$
\hat {f} = \sum_ {j = 1} ^ {p} \sum_ {t = 1} ^ {T} \alpha_ {t j} ^ {\lambda} K _ {j} \left(x _ {j} ^ {t}, \cdot\right) \tag {6}
$$

with

$$
\alpha^ {\lambda} = \underset {\alpha_ {j} \in \mathbb {R} ^ {T}, j = 1, \dots , p} {\arg \min } \left\{\frac {1}{T} \sum_ {t = 1} ^ {T} \ell_ {\sigma} \left(y ^ {t} - \sum_ {j = 1} ^ {p} \left(\mathbf {K} _ {j} ^ {t}\right) ^ {\prime} \alpha_ {j}\right) + \lambda \sum_ {j = 1} ^ {p} \tau_ {j} \| \alpha_ {j} \| _ {2} \right\}. \tag {7}
$$

The optimization problem (7) can be solved by Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) (Beck & Teboulle, 2009). We provide the detailed optimization procedure in Appendix E due to the space limitation.

Remark 1. The proposed method can be easily extended to the group sparse setting by replacing the direct decomposition  $\{\mathcal{X}_j\}_{j=1}^p$  with the subgroups decomposition  $\{\mathcal{X}_j\}_{j=1}^d$ , where each  $\mathcal{X}_j$  refers to the component input space concerning the interactions among variables ((Yin et al., 2012; Lin & Zhang, 2007)) or lagged times ((Nicholson et al., 2020)).

# 3.2 ASYMPTOTIC THEORY ANALYSIS

In this section, we provide the theoretical analysis of SpHAM. It is natural and necessary to investigate the following three problems:

- The problem of the function estimation calibration and convergence rates for stationary time series data (See Theorem 1 in Section 3.2.1);  
- The problem of the function estimation calibration for non-stationary time series data (See Theorem 2 in Section 3.2.2);  
- Whether the theoretical findings can inspire us to modify the SpHAM for non-stationary time series forecasting (See Theorem 3 in Section 3.3).

Due to limited space, the high-level outline and detailed proofs are provided in the Appendix A-D.

# 3.2.1 FUNCTION APPROXIMATION ANALYSIS FOR STATIONARY TIME SERIES

Assumption 1. Assume that  $|Y^t|, \forall t \in \mathbb{Z}$ , is bounded and there exists a constant  $c > 0$  such that  $\mathbb{E}|Y^t|^{1 + c} < \infty, \forall t \in \mathbb{Z}$ .

The moment condition in Assumption 1 is rather weak in the sense that the response variable  $Y^{t}$  possesses infinite variance. The same condition also applies to the distributions of the innovation  $\varepsilon^t$ , implying that heavy-tailed innovation is allowed.

Assumption 2. Let  $\kappa = \sup_{x\in \mathcal{X}}\sqrt{K_j(x,x)} <  \infty ,\forall j = 1,\dots,p.$

Some familiar kernels satisfy the requirements of Assumption 2, e.g., Gaussian kernel, Sigmoid kernel, and Logistic kernel.

The following definitions are needed for Assumption 3. For any  $j = 1, \dots, p$ , we define a kernel integral operator  $L_{K_j,T + 1}: L_2(\rho_{\mathcal{X}_j}^{T + 1}) \to L_2(\rho_{\mathcal{X}_j}^{T + 1})$  associated with the kernel  $K_j$  by

$$
L _ {K _ {j}, T + 1} (f) \bigl (x _ {j} ^ {T + 1} \bigr) = \int_ {\mathcal {X} _ {j}} K _ {j} (x _ {j} ^ {T + 1}, u _ {j}) f (u _ {j}) d \rho_ {\mathcal {X} _ {j}} ^ {T + 1} (u _ {j}).
$$

Note that  $L_{K_j,T + 1}$  is a compact and positive operator on  $L_{2}(\rho_{\mathcal{X}_{j}}^{T + 1})$ . According to Mercer theorem, we can find the corresponding normalized eigenpairs  $\{(\zeta_i^j,\psi_i^j)\}_{i\geq 1}$  such that  $\{\psi_i^j\}_{i\geq 1}$  is an orthonormal basis of  $L_{2}(\rho_{\mathcal{X}}^{T + 1})$  and  $\zeta_i^j\to 0$  as  $i\to \infty$ . Then for given  $r > 0$ , we defined the  $r$ -th power  $L_{K_j,T + 1}^r$  by

$$
L _ {K _ {j}, T + 1} ^ {r} \left(\sum_ {i \geq 1} \beta_ {i} ^ {j} \psi_ {i} ^ {j}\right) = \sum_ {i \geq 1} \beta_ {i} ^ {j} \left(\zeta_ {i} ^ {j}\right) ^ {r} \psi_ {i} ^ {j}.
$$

Assumption 3. We assume that  $f_{j}^{*}: \mathcal{X}_{j} \to \mathbb{R}, \forall j = 1, \dots, p$  is a function of the form  $f_{j}^{*} = L_{K_{j}, T + 1}^{r}(g_{j}^{*}), \forall r \in (0, \frac{1}{2}]$  with some  $g_{j}^{*} \in L_{2}(\rho_{\mathcal{X}_{j}}^{T + 1}), \forall h \in \mathbb{Z}$ .

The Assumption 3 is necessary to bridge the relation between the ground truth  $f^{*}$  and the functions in  $\mathcal{H}_K$ . Indeed, this assumption is a natural extension from i.i.d setting (Assumption 1 in Christmann & Zhou (2016)) to non-i.i.d time series.

Theorem 1. Suppose that the process  $\{Z^t\}_{t = -\infty}^{\infty}$  is stationary. Let Assumptions 1-3 be true. By taking  $\sigma = T^{m}$ ,  $\eta = T^{-\frac{1}{4r}}$  and  $\lambda = T^{-\frac{1}{4r} - m}$ , we have for any  $0 < \delta < 1$

$$
\| \hat {f} - f ^ {*} \| _ {L _ {2} \left(\rho_ {\mathcal {X}} ^ {T + 1}\right)} ^ {2} \leq \widetilde {C} \log (1 / \delta) T ^ {\Psi (m, c, r)}
$$

with confidence at least  $1 - \delta$ , where  $\widetilde{C}$  is a positive constant independently of  $T, \lambda, \eta, \delta$  and  $\sigma$  and

$$
\Psi (m, c, r) = \left\{ \begin{array}{l l} \max  \{- \frac {1}{2}, m - 1, - c m + \frac {1}{4 r} + m - 1 \}, & i f m \leq 1 - \frac {1}{4 r} \\ \max  \{- \frac {1}{2}, m - 1, - c m + \frac {1}{2 r} + 2 m - 2 \}, & i f m > 1 - \frac {1}{4 r}. \end{array} \right.
$$

Remark 2. In the stationary case, our bound appears to be novel for the following reasons: a) the result is completely independent of the various mixing conditions which are widely used in the theory analysis of non-i.i.d dependent time series (Mohri & Rostamizadeh, 2009; Yang et al., 2018; Mohri & Rostamizadeh, 2010; Guo & Shi, 2011; Qiu et al., 2015); b) compared with Yang et al. (2018), the innovation assumption is rather weak in the sense that the innovation possesses infinite variance and thus admits a heavy-tailed distribution.

Corollary 1. Suppose that the process  $\{Z^t\}_{t = -\infty}^{\infty}$  is stationary. Let all the conditions in Theorem 1 be true. We then have for any  $0 < \delta < 1$

$$
\| \hat {f} - f ^ {*} \| _ {L _ {2} (\rho_ {\mathcal {X}} ^ {T + 1})} ^ {2} \leq \widetilde {C} \log (1 / \delta) T ^ {\Psi (m, c)}
$$

with confidence at least  $1 - \delta$ , where

$$
\Psi (m, c) = \left\{ \begin{array}{l l} \max  \{- \frac {1}{2}, - c m + m - \frac {1}{2} \}, & i f m \leq \frac {1}{2} \\ \max  \{m - 1, - c m + + 2 m - 1 \}, & i f m > \frac {1}{2}. \end{array} \right.
$$

Figure 1 summarizes the convergence rates in Corollary 1 by taking different  $\sigma$  and  $c$ . If the innovation is Gaussian distribution with finite variance (i.e., Assumption 1 holds for any  $c > 1$ ), one can arbitrarily select a  $\sigma = T^m\left(0 < m < \frac{1}{2}\right)$  to obtain convergence rates  $O(n^{-\frac{1}{2}})$ . Moreover, we can see that the convergence rate will decrease as  $m$  increases. Combined with the conclusion in Lemma 1 (i.e., the equivalence relation between Huber loss based empirical risk and MSE as  $\sigma \to \infty$ ), it indicates that  $\sigma$  indeed plays a tradeoff role between algorithmic robustness and variance-reduction. For the weak moment condition (e.g.,  $0 < c < 1$ ), one may get slower convergence rates (e.g.,  $O(n^{(1 - c)m - \frac{1}{2}})$ ) or

![](images/1a0aade4255924c984d4601a450bb1b9fe1a69286fdaf9accc8f5cfd310686c7.jpg)  
Figure 1: The convergence rates under different  $\sigma$  and  $c$ .

$O(n^{(2 - c)m - 1}))$ , which also coincides with our intuitive understanding that small  $\sigma$  may be conducive to robust forecasting. Note that our method will not converge when  $\sigma$  and  $c$  are both located in the white area in Figure 1.

As a comparison, due to the non-robustness of the squared loss, the most existing convergence rates are established under the assumption that the innovation is Gaussian distribution with finite

variance (see, e.g., Yang et al. (2018); Han et al. (2015); Wang & Yang (2007); Kock & Callot (2015)). However, from Theorem 1 and Corollary 1, we see that the convergence of SpHAM can be proven under weaker moment condition, which verifies the robustness of our method. Recall the learning rate  $O(n^{\frac{2d}{2d + 1}})$  derived in Yang et al. (2018), where  $d$  is the order of smoothness of the component function  $f_{j}, j = 1,\dots,p$ . Relatively slow convergence rate  $O(n^{-\frac{1}{2}})$  we obtained indicates the sacrifice for the absence of mixing condition.

# 3.2.2 FUNCTION APPROXIMATION ANALYSIS FOR NON-STATIONARY TIME SERIES

In non-stationary time series setting, different  $Z^t$ s may follow different distributions. Thus we introduce the weights  $\mathbf{s} = \{s_1, \dots, s_T\}$  and assign them to the losses made on different sample points in terms of their relevance to forecasting the future  $Z^{T+1}$ . For given weights  $\{s_t\}_{t=1}^T$ , we define the estimator  $\hat{f}^{\mathrm{s}} = \sum_{j=1}^{p} \hat{f}_j^{\mathrm{s}}$  as the minimizer of the following weighted objective

$$
\mathcal {E} _ {\lambda} ^ {\mathbf {s}} (\hat {f} ^ {\mathbf {s}}) := \min  _ {f \in \mathcal {H} _ {Z}} \left\{\sum_ {t = 1} ^ {T} s _ {t} \ell_ {\sigma} \left(y ^ {t} - \sum_ {j = 1} ^ {p} f _ {j} \left(x _ {j} ^ {t}\right)\right) + \lambda \Omega (f) \right\}. \tag {8}
$$

Furthermore, we introduce a discrepancy measure developed describes the discrepancy between target distribution and the distribution of the sample (Kuznetsov & Mohri, 2020).

Definition 2. For any  $f \in \mathcal{H}_Z$ , the discrepancy measure with respect to Huber loss is defined as

$$
\operatorname {d i s c} (\mathbf {s}) := \sup  _ {f \in \mathcal {H} _ {Z}} \left\{\mathbb {E} \ell_ {\sigma} (f (x ^ {T + 1}) - y ^ {T + 1}) - \sum_ {t = 1} ^ {T} s _ {t} \mathbb {E} \ell_ {\sigma} (f (x ^ {t}) - y ^ {t}) \right\}.
$$

The discrepancy measure is a natural measure of the non-stationarity of the stochastic process  $\{Z^t\}_{t = -\infty}^{\infty}$  with respect to both the loss function  $\ell_{\sigma}$  and the hypothesis set  $\mathcal{H}_Z$ .

Theorem 2. Let Assumptions 1-3 be true. We assume that  $f_{j}^{*} \in \mathcal{H}_{K_{j}}, \forall j = 1, \dots, p$ . By taking  $\sigma = T^{\frac{1}{2c}}$ ,  $\lambda = T^{-1}$  and  $\eta = T^{-\frac{1}{2}}$ , we have for any  $0 < \delta < 1$

$$
\left\| \hat {f} ^ {\mathbf {s}} - f ^ {*} \right\| _ {L _ {2} \left(\rho_ {\mathcal {X}} ^ {T + 1}\right)} ^ {2} \leq \operatorname {d i s c} (\mathbf {s}) + \mathcal {E} _ {\lambda} ^ {\mathbf {s}} (\hat {f} ^ {\mathbf {s}}) + \widetilde {C} _ {1} \| \mathbf {s} \| _ {2} T ^ {\frac {1}{2 c}} + \widetilde {C} _ {2} \log (1 / \delta) T ^ {- \frac {1}{2}}.
$$

with confidence at least  $1 - \delta$ , where  $\widetilde{C}_1, \widetilde{C}_2$  are two positive constants independently of  $T, \lambda, \eta, \delta$  and  $\sigma$ .

Note that there are several existing studies towards analyzing non-stationary and non-mixing time series, see, e.g., Kuznetsov & Mohri (2020). Different from them, the error bound we derived is with respect to the function approximation error rather than generalization error, which is very crucial for Huber regression problem, since the convergence of generalization cannot imply the convergence of function approximation when the innovation is non-Gaussian (Sun et al., 2019; Feng & Wu, 2020).

# 3.3 ADAPTIVE SPARSE HUBER ADDITIVE MODEL FOR NON-STATIONARY TIME SERIES

Theorem 2 illustrates that we shall minimize the following optimization problem for non-stationary time series forecasting:

$$
\min _ {f \in \mathcal {H} _ {Z}} \left\{\sum_ {t = 1} ^ {T} s _ {t} \ell_ {\sigma} \left(y _ {t} - \sum_ {j = 1} ^ {p} f _ {j} \left(x _ {j} ^ {t}\right)\right) + \operatorname {d i s c} (\mathbf {s}) + \lambda_ {1} \Omega (f) + \lambda_ {2} T ^ {\frac {1}{2 c}} \| \mathbf {s} \| _ {2} \right\},
$$

where  $\lambda_{1}$  and  $\lambda_{2}$  are two positive regularization parameters, and  $c$  is a positive constant introduced in Assumption 1. Although the discrepancy measure  $\mathrm{disc}(\mathbf{s})$  is crucial for such an optimization problem, we cannot obtain its exact value since we do not have access to the distributions of  $Z^{t}, t \in \mathbb{Z}$ . Hence, we need to estimate the approximated discrepancy from given data. Inspired by (Kuznetsov & Mohri, 2020), one natural and necessary assumption is that there is an underlying representation relationship between distribution  $\rho^{T+1}$  and distributions  $\rho^{t}, t = 1, \dots, T$ .

Assumption 4. Denote by a probability set  $\mathbf{q}^{*} = \{q_{t}^{*}\}_{t = 1}^{T}$  with  $\sum_{t = 1}^{T}q_{t}^{*} = 1$ . We assume that the following term is sufficiently small:

$$
\operatorname {d i s c} \left(\mathbf {q} ^ {*}\right) := \sup  _ {f \in \mathcal {H} z} \left[ \mathbb {E} \ell_ {\sigma} \left(f \left(x ^ {T + 1}\right) - y ^ {T + 1}\right) - \sum_ {t = 1} ^ {T} q _ {t} \mathbb {E} \ell_ {\sigma} \left(f \left(x ^ {t}\right) - y ^ {t}\right) \right].
$$

This assumption is necessary for learning non-stationary time series, see Kuznetsov & Mohri (2020) for further discussion.

Theorem 3. Let Assumption 4 the conditions in Theorem 2 be true. We have for any  $0 < \delta < 1$

$$
\begin{array}{l} \| \hat {f} ^ {\mathbf {s}} - f ^ {*} \| _ {L _ {2} (\rho_ {\mathcal {X}} ^ {T + 1})} ^ {2} \leq \quad \operatorname {d i s c} (\mathbf {q} ^ {*}) + \sup  _ {f \in \mathcal {H} _ {Z}} \sum_ {t = 1} ^ {T} (q _ {t} ^ {*} - s _ {t}) \ell_ {\sigma} (f (x ^ {t}) - y ^ {t}) + \mathcal {E} _ {\lambda} ^ {\mathbf {s}} (\hat {f} ^ {\mathbf {s}}) \\ + \widetilde {C} _ {1} \log (1 / \delta) \left(\| \mathbf {q} ^ {*} - \mathbf {s} \| _ {2} + \| \mathbf {s} \| _ {2} T ^ {\frac {1}{2 c}}\right) + \widetilde {C} _ {2} \log (1 / \delta) T ^ {- \frac {1}{2}} \\ \end{array}
$$

with confidence at least  $1 - \delta$ , where  $\widetilde{C}_1, \widetilde{C}_2$  are two constants independently of  $T, \lambda, \eta, \delta$  and  $\sigma$ .

Note that the priori  $\mathbf{q}$  can be any distribution. Also, some optimization strategies for hyper-parameter selection can also be used for finding the underlying probability set  $\mathbf{q}^*$ , e.g., bilevel optimization scheme (Frecon et al., 2018; Franceschi et al., 2018). We leave these extensions to future work. For simplicity, this paper assumes that the distributions  $\rho^{T + 1}$  does not change drastically compared with the distributions  $\rho^t, t = 1,\dots,T$ . Thus, we consider the underlying probability  $\mathbf{q}^*$  as an uniform distribution over last  $l$  observations in this paper, where  $l > 0$  is a hyper-parameter that can be tuned via cross-validation. Finally, the optimization problem of adaptive SpHAM can be formulated as following two stages:

Step A: finding the weight  $\hat{s}$ :

$$
\hat {\mathbf {s}} = \arg \min  _ {\mathbf {s}} \left\{\sup  _ {f \in \mathcal {H} _ {Z}} \sum_ {t = 1} ^ {T} \left(q _ {t} ^ {*} - s _ {t}\right) \ell_ {\sigma} \left(f \left(x ^ {t}\right) - y ^ {t}\right) + \lambda_ {1} \| \mathbf {q} ^ {*} - \mathbf {s} \| _ {2} ^ {2} + \lambda_ {2} T ^ {\frac {1}{2 c}} \| \mathbf {s} \| _ {2} ^ {2} \right\} \tag {9}
$$

Step B: forecasting:

$$
\hat {f} ^ {\mathbf {s}} (x ^ {T + 1}) = \sum_ {j = 1} ^ {p} \sum_ {t = 1} ^ {T} \alpha_ {t j} ^ {\mathbf {s}} K (x _ {j} ^ {t}, x _ {j} ^ {T + 1}),
$$

where

$$
\alpha^ {\mathbf {s}} = \underset {\alpha_ {j} \in \mathbb {R} ^ {T}, j = 1, \dots , p} {\arg \min } \left\{\sum_ {t = 1} ^ {T} \hat {s} _ {t} \ell_ {\sigma} \left(y ^ {t} - \sum_ {j = 1} ^ {p} \left(\mathbf {K} _ {j} ^ {t}\right) ^ {\prime} \alpha_ {j}\right) + \lambda \sum_ {j = 1} ^ {p} \tau_ {j} \| \alpha_ {j} \| _ {2} \right\}. \tag {10}
$$

The optimization problems (9) can be solved by standard gradient descent, where, for each step, we employ DC programming to evaluate the supremum over  $f \in \mathcal{H}_Z$ . Similarly to the strategy for optimization problem (7), we use Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) (Beck & Teboulle, 2009) for Step B. We provide the detailed procedure in Appendix E.

# 4 EXPERIMENT

This section validates the effectiveness of SpHAM and adaptive SpHAM. In all experiments, the Gaussian kernel  $K_{j}(u,v) = \exp (-\frac{\|u - v\|_{2}^{2}}{2d^{2}}), j = 1,\dots,p$ , is employed for constructing the additive data dependent hypothesis space, where  $d > 0$  is the bandwidth. Due to limited space, we only represent the evaluations on the synthetic time series. The real-world experiments can be found in Appendix F.

# 4.1 EVALUATION ON SYNTHETIC STATIONARY DATA

We consider two synthetic examples: stationary autoregressive model and nonstationary autoregressive model:

Example A: Inspired by (Kuznetsov & Mohri, 2020), a stationary time series is generated according to the non-linear additive autoregressive model:

$$
Y ^ {t} = \frac {3}{2} \sin (\frac {\pi}{2} Y ^ {t - 2}) - \sin (\frac {\pi}{2} Y ^ {t - 3}) + \varepsilon^ {t},
$$

where the innovation  $\varepsilon^t$ s are i.i.d. drawn from Gaussian distribution  $N(0,1)$  and Student distribution with freedom 2, respectively.

Example B: Inspired by (Kuznetsov & Mohri, 2020), a time series with smooth drift is generated by

$$
Y ^ {t} = \frac {t}{4 0 0} \sin (Y ^ {t - 1}) + \epsilon^ {t}.
$$

Hyperparameter selection and evaluation criterions: Recall that SpHAM algorithm requires three hyper-parameters: regularization parameter  $\lambda$ , bandwidth of kernel  $d$  and Huber parameter  $\sigma$ . For Huber parameter  $\sigma$  selection, we set  $\sigma = T^{\frac{1}{4}}$  based on the suggestion in Theorem 1 and Corollary 1. Moreover, we make no attempt to optimize the other hyper-parameters in our method. Thus, we set the  $d = 1$ ,  $\lambda = 0.001$ ,  $\lambda_{1} = 1000$ ,  $\lambda_{2} = 30$  and  $l = 60$ . Suppose that there are  $N$ -size test samples. The evaluation criterions for forecasting used here contains Average Sample Error(ASE) =  $\frac{1}{N}\sum_{t=1}^{N}(\hat{f}(x^t) - y^t)_2^2$  and True Deviation (TD) =  $\frac{1}{N}\sum_{t=1}^{N}(\hat{f}(x^t) - f^*(x^t))_2^2$ .

For each example, we generate time series with 4000 sample points. The samples at time  $t = \{1500, 1501, \dots, 1900\}$  are used as a training set, and the samples at time  $\{1901, 1902, 1903\}$  are considered as the test set. The competing methods include standard Autoregression (AR) and TS-SpAM (Yang et al., 2018). All the evaluations are repeated for 10 times. The average results for Example A and Example B are presented in Table 2.

From the results on Example A, we see that SpHAM enjoys robust forecasting in presence of heavily-tailed  $t$  noise. Moreover, the RKHS offer our method with nonlinear forecasting, which makes our method always outperforms linear AR. The results in Example B verify the effectiveness of our method for nonstationary time series forecasting.

Table 2: The results on synthetic stationary data.  

<table><tr><td rowspan="2"></td><td rowspan="2">Methods</td><td colspan="2">Gaussian noise</td><td colspan="2">Student noise</td></tr><tr><td>ASE</td><td>TD</td><td>ASE</td><td>TD</td></tr><tr><td rowspan="3">Example A (stationary)</td><td>AR</td><td>3.4832</td><td>3.5021</td><td>21.372</td><td>3.5426</td></tr><tr><td>TS-SpAM</td><td>0.6144</td><td>0.5845</td><td>11.242</td><td>2.3499</td></tr><tr><td>SpHAM</td><td>0.6121</td><td>0.5932</td><td>10.941</td><td>1.7112</td></tr><tr><td rowspan="4">Example B (nonstationary)</td><td>AR</td><td>11.624</td><td>10.915</td><td>13.334</td><td>13.012</td></tr><tr><td>TS-SpAM</td><td>7.1213</td><td>5.8141</td><td>7.8315</td><td>7.1575</td></tr><tr><td>SpHAM</td><td>6.8677</td><td>5.5576</td><td>9.3382</td><td>7.8111</td></tr><tr><td>Adaptive SpHAM</td><td>2.1643</td><td>1.4933</td><td>4.7256</td><td>1.2986</td></tr></table>

# 4.2 EVALUATION ON BENCHMARK DATA

We test our algorithm on nonlinear Nonlinear-VAR dataset from CauseMe. The hyper-parameters are the same as we use in Section 4.1, except for  $d$  which is fine-tuned by cross validation in  $\{0.1, 0.5, 1, 2\}$ . The results in Table 3 verify the effectiveness of our method.

Table 3: The ASE on CauseMe data ( $p$  refers to the dimension of features).  

<table><tr><td>Methods</td><td>(p=3,T=300)</td><td>(p=5,T=300)</td></tr><tr><td>VAR</td><td>4.1678</td><td>5.4330</td></tr><tr><td>TS-SpAM</td><td>3.8908</td><td>4.7424</td></tr><tr><td>SpHAM</td><td>3.8240</td><td>4.7473</td></tr></table>

# 5 CONCLUSION

We propose an adaptive sparse Huber additive model by integrating Huber loss and  $\ell_{2,1}$ -norm regularizer into a additive data dependent hypothesis space. We explore, both theoretically and empirically, the ability of our method for robust estimation and inference in both non-Gaussian data and (non)stationary time series data. To our best knowledge, this is the first work on sparse Huber additive models for time series analysis. Experimental results on both synthetic and real-world data validate the effectiveness of the proposed method.

# REFERENCES

Terrence M. Adams and Andrew B. Nobel. Uniform convergence of vapnik-chervonenkis classes under ergodic sampling. The Annals of Probability, 38(4):1345-1367, 2010.  
Agarwal Alekh and Duchi John C. The generalization ability of online algorithms for dependent data. IEEE Transactions on Information Theory, 59(1):573-587, 2013.  
Richard T. Baillie. Long memory processes and fractional integration in econometrics. Journal of Econometrics, 73(1):5-59, 1996.  
Peter L. Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research, 3:463-482, 2002.  
Amir Beck and Marc Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1):183-202, 2009.  
Bart Bussmann, Jannes Nys, and Steven Latre. Neural additive vector autoregression models for causal discovery in time series data. arXiv:2010.09429v1, 2020.  
John M. Chambers and Trevor Hastie. Statistical Models in S. Wadsworth & Brooks/Cole Advanced Books & Software, 1992.  
Andreas Christmann and Ding-Xuan Zhou. Learning rates for the risk of kernel based quantile regression estimators in additive models. Analysis and Applications, 14(3):449-477, 2016.  
Tianjiao Chu and Clark Glymour. Search for additive nonlinear time series causal models. Journal of Machine Learning Research, 9:967-991, 2008.  
Francesca Dominici, Aidan McDermott, Scott L. Zeger, and Jonathan M. Samet. On the use of generalized additive models in time-series studies of air pollution and health. *American Journal of Epidemiology*, 156(3):193-203, 2002.  
Paul Doukhan. Mixing: Properties and Examples. Lecture Notes in Statistics. Springer, 1994.  
Jianqing Fan and Irene Gijbels. Local Polynomial Modelling and Its Applications. Chapman and Hall, 1996.  
Jianqing Fan and Qiwei Yao. Nonlinear time series: Nonparametric and parametric methods. Springer, 2005.  
Yunlong Feng and Qiang Wu. A statistical learning assessment of huber regression. arXiv:2009.12755v1, 2020.  
Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In International Conference on Machine learning (ICML), pp. 1563-1572, 2018.  
Jordan Frecon, Saverio Salzo, and Massimiliano Pontil. Bilevel learning of the group lasso structure. In Advances in Neural Information Processing Systems (NIPS), pp. 8301-8311, 2018.  
Zheng-Chu Guo and Lei Shi. Classification with non-i.i.d. sampling. Mathematical and Computer Modelling, 54(5):1347-1364, 2011.  
Fang Han, Huanran Lu, and Han Liu. A direct estimation of high dimensional stationary vector autoregressions. Journal of Machine Learning Research, 16:3115-3150, 2015.  
Trevor J. Hastie and Robert J. Tibshirani. Generalized additive models. London: Chapman and Hall, 1990.  
Jianhua Z. Huang and Lijian Yang. Identification of non-linear additive autoregressive models. Journal of the Royal Statistical Society. Series B (Statistical Methodology), 66(2):463-477, 2004.  
Peter J. Huber. Robust estimation of a location parameter. The Annals of Mathematical Statistics, 35 (1):73-101, 1964.

Peter J. Huber and Elvezio M. Ronchetti. Robust Statistics. Wiley, 2009.  
Anders Bredahl Kock and Laurent Callot. Oracle inequalities for high dimensional vector autoregressions. Journal of Econometrics, 186(2):325-344, 2015.  
Vitaly Kuznetsov and Mehryar Mohri. Generalization bounds for time series prediction with nonstationary processes. In International Conference on Algorithmic Learning Theory, 2014.  
Vitaly Kuznetsov and Mehryar Mohri. Discrepancy-based theory and algorithms for forecasting non-stationary time series. Annals of Mathematics and Artificial Intelligence, 88:367-399, 2020.  
Yi Lin and Hao Helen Zhang. Component selection and smoothing in multivariate nonparametric regression. The Annals of Statistics, 34(5):2272-2297, 2007.  
Po-Ling Loh. Statistical consistency and asymptotic normality for high-dimensional robust mesimators. The Annals of Statistics, 45(2):866-896, 2017.  
Mehryar Mohri and Afshin Rostamizadeh. Rademacher complexity bounds for non-i.i.d. processes. In Advances in Neural Information Processing Systems (NIPS), 2009.  
Mehryar Mohri and Afshin Rostamizadeh. Stability bounds for stationary  $\phi$ -mixing and  $\beta$ -mixing processes. In Advances in Neural Information Processing Systems (NIPS), 2010.  
William B. Nicholson, Ines Wilms, Jacob Bien, and David S. Matteson. High dimensional forecasting via interpretable vector autoregression. arXiv:1412.5250v4, 2020.  
Huitong Qiu, Sheng Xu, Fang Han, Han Liu, and Brian Caffo. Robust estimation of transition matrices in high dimensional heavy-tailed vector autoregressive processes. In Proceedings of the 32nd International Conference on Machine Learning, volume 37, pp. 1843-1851, 2015.  
Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. Online learning: random averages, combinatorial parameters, and learnability. In Advances in Neural Information Processing Systems (NIPS), pp. 1984-1992, 2010.  
Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. Sequential complexities and uniform martingale laws of large numbers. *Probability Theory and Related Fields*, 161(1):111-153, 2015.  
Khaiwal Ravindra, Preety Rattan, Suman Mor, and Ashutosh Nath Aggarwal. Generalized additive models: Building evidence of air pollution, climate change and human health. *Environment International*, 132:104987, 2019.  
Feldman Raya and Taqqu Murad. A Practical Guide to Heavy Tails: Statistical Techniques and Applications. Springer, 1998.  
Meir Ron. Nonparametric time series prediction through adaptive model selection. Machine Learning, 39(1):5-34, 2000.  
Jakob Runge, Sebastian Bathiany, Erik M. Bollt, Gustau Camps-Valls, Dim Coumou, Ethan R Doyle, Clark Glymour, Marlene Kretschmer, Miguel D. Mahecha, Jordi Muñoz-Marí, Egbert H. van Nes, Jonas Peters, Rick Quax, Markus Reichstein, Marten Scheffer, Bernhard Schölkopf, Peter L. Spirtes, George Sugihara, Jie Sun, Kun Zhang, and Jakob Zscheischler. Inferring causation from time series in earth system sciences. Nature Communications, 10, 2019.  
Qiongxia Song and Lijian Yang. Oracally efficient spline smoothing of nonlinear additive autoregression models with simultaneous confidence band. Journal of Multivariate Analysis, 101(9): 2008-2025, 2010.  
Charles J. Stone. Additive regression and other nonparametric models. The Annals of Statistics, 13 (2):689-705, 1985.  
Qiang Sun, Wen-Xin Zhou, and Jianqing Fan. Adaptive huber regression. Journal of the American Statistical Association, 115(529):254-265, 2019.  
Pham Dinh Tao and Le Thi Hoai An. A d.c. optimization algorithm for solving the trust-region subproblem. SIAM Journal on Optimization, 8:476-505, 1998.

Grace Wahba. Spline models for observational data. Society for Industrial and Applied Mathematics, 1990.  
Li Wang and Lijian Yang. Spline-backfitted kernel smoothing of nonlinear additive autoregression model. The Annals of Statistics, 35(6):2474 - 2503, 2007.  
Xiaofeng Wang and Donald E. Brown. The spatio-temporal generalized additive model for criminal incidents. In Proceedings of 2011 IEEE International Conference on Intelligence and Security Informatics, pp. 42-47, 2011.  
Kam Chung Wong, Zifan Li, and Ambuj Tewari. Lasso guarantees for  $\beta$ -mixing heavy-tailed time series. The Annals of Statistics, 48(2):1124-1142, 2020.  
Qiang Wu, Yiming Yin, and Ding-Xuan Zhou. Multi-kernel regularized classifiers. Biometrika, 23: 108-134, 2007.  
Yingxiang Yang, Adams Wei Yu, Zhaoran Wang, and Tuo Zhao. Detecting nonlinear causality in multivariate time series with sparse additive models. arXiv:1803.03919v2, 2018.  
Junming Yin, Xi Chen, and Eric P Xing. Group sparse additive models. In International Conference on Machine Learning (ICML), pp. 871-878, 2012.  
Ding-Xuan Zhou. The covering number in learning theory. Journal of Complexity, 18(3):739-767, 2002.
