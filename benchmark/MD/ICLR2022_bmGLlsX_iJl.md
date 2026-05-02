# EMFLOW: DATA IMPUTATION IN LATENT SPACE VIA EM AND DEEP FLOW MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

The presence of missing values within high-dimensional data is an ubiquitous problem for many applied sciences. A serious limitation of many available data mining and machine learning methods is their inability to handle partially missing values and so an integrated approach that combines imputation and model estimation is vital for down-stream analysis. A computationally fast algorithm, called EMFlow, is introduced that performs imputation in a latent space via an online version of Expectation-Maximization (EM) algorithm by using a normalizing flow (NF) model which maps the data space to a latent space. The proposed EMFlow algorithm is iterative, involving updating the parameters of online EM and NF alternatively. Extensive experimental results for high-dimensional multivariate and image datasets are presented to illustrate the superior performance of the EMFlow compared to a couple of recently available methods in terms of both predictive accuracy and speed of algorithmic convergence.

# 1 INTRODUCTION

Missing values are often encountered for real-world datasets and are known to adversely impact the validity of down-stream analysis. Most machine learning (ML) algorithms and statistical tools often ignore the problem of partially observed features by simply dropping entire cases with missing values, which could result into biased estimation and underestimation of parameter uncertainty (Schmitt et al., 2015; Somasundaram & Nedunchezhian, 2011; Lall, 2016).

There have been some recent attempts to develop ML methods that properly handle missing values. Early works in this field performed imputation in a supervised manner where a complete training set is needed to learning the correlation between missing and observed entries (e.g. García-Laencina et al., 2010; Rezende et al., 2014; Bertalmio et al., 2000; Xie et al., 2012; Yeh et al., 2016). However, it is common that a large collection of fully observed data is hard to acquire, which makes those data-greedy supervised algorithms less effective. On the other hand, early attempts in unsupervised imputation methods, such as the collaborative filtering (Sarwar et al., 2001), usually learn the correlation across dimensions by embedding the data into a lower dimensional latent space linearly (e.g. Little & Rubin, 2019; Audigier et al., 2016). To improve the limited representation power of those linear models, kernel-based methods are also proposed but at the cost of excessive computational load when dealing with large datasets (e.g. Sanguinetti & Lawrence, 2006; Liu et al., 2016).

More recently, several imputation methods based on deep generative models have been proposed, providing much more accurate estimates of missing values especially for high-dimensional image datasets. In this work, we introduce EMFlow that integrates the normalizing flow (NF) (Dinh et al., 2016; Rezende & Mohamed, 2015; Dinh et al., 2014; Kingma & Dhariwal, 2018) with an online version of Expectation-Maximization (EM) algorithm (Cappé & Moulines, 2009). The proposed framework is motivated by the strength and weakness of EM. As a class of iterative algorithm designed for latent variable models, EM can be applied to data imputation in an interpretable way (Little & Rubin, 2019). Additionally, the learning of EM is usually numerically stable and its convergence property has been studied extensively (e.g. Ma et al., 2000; Zhao et al., 2020; Meng et al., 1994; Wu, 1983). However, the E-step and M-step are only traceable with simple underlying distributions including multivariate Gaussian or Student's  $t$  distributions as well as their mixtures (e.g. Di Zio et al., 2007; xian Wang et al., 2004). On the other hand, NF is capable of latent representation and efficient data sampling, which makes it a convenient bridge between the data space and the

latent space. Therefore, we let EM perform imputation in the latent space where simple inter-feature dependency is assumed (i.e. the underlying distribution is multivariate Gaussian). Meanwhile, NF is used to recover the mapping between the complex inter-feature dependency in the data space and the one in the latent space learned by EM.

The inference of EMFlow adopts an iterative learning strategy that has been widely used in model-based multiple imputation methods where the initial naive imputation is refined step by step until convergence (e.g. Gondara & Wang, 2017; Buuren & Groothuis-Oudshoorn, 2010; Stekhoven & Buhlmann, 2012). Specifically, three steps are performed alternatively:

- update the density estimation of complete data including the observed and current imputed values;  
- update the base distribution (i.e.  $\mu$  and  $\Sigma$ ) in the latent space by EM; and  
- update the imputation in the data space.

Note that the first update corresponds to optimizing the complete data likelihood as well as the reconstruction error computed on the observed entries. We also derive an online version of EM such that it only consumes a batch of data at a time for parameter updates and thus can work with deep generative models smoothly. We will show that such learning schema has simpler implementation and leads to faster convergence compared to other competing methods.

The main contributions of this work are

(i) an imputation framework combining an online version of Expectation-Maximization (EM) algorithm and the normalizing flow;  
(ii) an iterative learning schema that alternatively updates the inter-feature dependency in the latent space (i.e. the base distribution) and density estimation in the data space;  
(iii) a derivation of online EM in the context of missing data imputation; and  
(iv) extensive experiments on multiple image and tabular datasets to demonstrate the imputation quality and convergence speed of the proposed framework.

# 2 RELATED WORK

Recently, the applications of deep generative models like Generative Adversarial Networks (GAN) (Goodfellow et al., 2014) have been extended to the field of missing data imputation under the assumption of Missing at Random (MAR) or Missing Completely at Random (MCAR). GAIN (Yoon et al., 2018) designs a compute data generator that performs imputation, and a discriminator to differentiate imputed and observed components with the help of a hint mechanism. MisGAN (Li et al., 2019) introduces another pair of generator-discriminator that aims to learn the distribution of missing mask. However, training GAN-based models is a notoriously challenging for its excessively complex structure and non-convex objectives. For example, MisGAN optimizes three objectives jointly involving six neural networks for imputation. Furthermore, GAN-based models do not obtain explicit density estimation that could be critical for down-stream analysis (e.g. Ferdosi et al., 2011; Zambom & Ronalddo, 2013).

Some imputation techniques based on Variational Autoencoders (VAE) (Kingma & Welling, 2013) are also developed. For example, MIWAE Mattei & Frellsen (2019) leverages the importance-weighted autoencoder (Burda et al., 2015) and optimizes a lower bound of the likelihood of observed data. But the zero imputation adopted by it can be problematic since observed entries can also be zero. It also needs quite large computational power to make the bound to be tight. EDDI (Ma et al., 2018) avoids zero imputation by introducing a permutation invariant encoder. Imputation under Missing not at Random (MNAR) has also been explored by explicitly modeling the missing mechanism (Ipsen et al., 2020; Collier et al., 2020). However, all VAE-based approaches only permit approximate density estimation no matter how expressive the inference model is.

Our proposed framework builds on the work of MCFlow (Richardson et al., 2020) that utilizes NF to learn exact density estimation on incomplete data via an iterative learning schema. The core component of MCFlow is a feed forward network that operates in the latent space and attempts to find the likeliest embedding vector in a supervised manner. Although MCFlow achieves impressive performance compared to other state-of-art methods, it remains ambiguous that how the feed forward

network exploits the correlation in the latent space. A key difference between the MCFlow and our proposed method is that we exploit the correlation in the latent space which in turn induces correlation in ambient space via NF. We also find that the inference of MCFlow often has slow convergence speed and can be unstable.

# 3 APPROACH

# 3.1 PROBLEM DEFINITION

We define the complete dataset  $\mathbf{X}$  as a collection of  $p$ -dimensional vectors  $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$  that are independent and identically distributed (i.i.d.) samples drawn from  $p_X(\cdot; \theta)$  in a  $p$ -dimensional data space  $\mathcal{X}$ . The missing pattern of  $\mathbf{x}_i$  is described by a binary mask  $\mathbf{m}_i \in \{0, 1\}^p$  such that  $\mathbf{x}_{ij}$  is missing if  $\mathbf{m}_{ij} = 1$ , and  $\mathbf{x}_{ij}$  is observed if  $\mathbf{m}_{ij} = 0$ .

Let  $\mathbf{x}^o$  and  $\mathbf{x}^m$  be the observed and missing parts of  $\mathbf{x}$ , and  $p_M(\mathbf{m}|\mathbf{x}) = p_M(\mathbf{m}|\mathbf{x}^o, \mathbf{x}^m)$  be the conditional distribution of the mask. Based on dependency between  $\mathbf{m}$  and  $(\mathbf{x}^o, \mathbf{x}^m)$ , the missing mechanism can be classified into three classes (Little & Rubin, 2019):

- MCAR:  $p_{M}(\mathbf{m}|\mathbf{x}^{o},\mathbf{x}^{m}) = p_{M}(\mathbf{m})$  
- MAR:  $p_{M}(\mathbf{m}|\mathbf{x}^{o},\mathbf{x}^{m}) = p_{M}(\mathbf{m}|\mathbf{x}^{o})$  
- MNAR: The probability of missing depends on both  $\mathbf{x}^o$  and  $\mathbf{x}^m$ .

Throughout this paper, we only focus on MCAR or MAR where the missing mechanism can be safely ignored. The typical objective of learning a latent model is to maximize the observed data log-likelihood defined as

$$
L ^ {o b s} (\theta) = \sum_ {i = 1} ^ {n} \log \int p _ {X} \left(\mathbf {x} _ {i} ^ {o}, \mathbf {x} _ {i} ^ {m}; \theta\right) d \mathbf {x} _ {i} ^ {m}. \tag {1}
$$

However, our goal in this work is to estimate  $p_X$  from incomplete data as well as obtain accurate imputation under  $p_X$ . Therefore, we attempt to learn the reconstructed data  $\widehat{\mathbf{X}} = (\widehat{\mathbf{x}}_1, \dots, \widehat{\mathbf{x}}_n)^T$  and the estimated model parameter  $\widehat{\theta}$  via

$$
\left(\widehat {\mathbf {X}}, \widehat {\theta}\right) = \underset {\mathbf {x} _ {i} \in \mathcal {X} _ {i} ^ {\prime}, \theta} {\arg \max } \sum_ {i = 1} ^ {n} \log p _ {X} \left(\mathbf {x} _ {i} \mid \theta\right) \tag {2}
$$

where  $\mathcal{X}_i' \in \mathcal{X}$  is the search space for  $\mathbf{x}_i$  where the observed locations have fixed values.

# 3.2 NORMALIZING FLOWS

To make it possible to optimize equation 2,  $p_{X}(\mathbf{x};\theta)$  needs to be specified in a parametric way that should be expressive enough, as the density is potentially complex and high-dimensional. To this end, we use NF to model  $p_{X}(\mathbf{x};\theta)$  as an invertible transformation  $f_{\psi}$  of a base distribution  $p_Z(\mathbf{z};\phi)$  in the latent space  $\mathcal{Z}$ . Under the change of variables theorem, the complete data log-density is specified as

$$
p _ {X} (\mathbf {x}; \theta) = p _ {Z} \left(f _ {\psi} ^ {- 1} (\mathbf {x}); \phi\right) \left| \det  \left(\frac {\partial f _ {\psi} ^ {- 1} (\mathbf {x})}{\partial \mathbf {x} ^ {T}}\right) \right| \tag {3}
$$

where  $\theta = (\psi ,\phi)$

$f_{\psi}$  is usually composed by a sequence of relatively simple transformations to approximate arbitrarily complex distributions with high representation power. In this work, we choose Real NVP (Dinh et al., 2016) based on affine coupling transformations as the flow model<sup>1</sup>. Recently, Teshima

et al. (2020) has shown that flow models constructed from affine coupling layers can be universal distributional approximators.

Data imputation relies on the inter-feature dependency that is usually intractable and hard to capture in the data space  $\mathcal{X}$  with the presence of missing entries. Therefore, we make the following two assumptions related to NF.

Assumption 1: The inter-feature dependency in the latent space  $\mathcal{Z}$  is simple and can be characterized by a multivariate Gaussian density, that is:

$$
p _ {Z} (\mathbf {z}; \phi) = \mathcal {N} (\mathbf {z}; \boldsymbol {\mu}, \boldsymbol {\Sigma}) \tag {4}
$$

where  $\phi = (\pmb{\mu},\pmb{\Sigma})$  consists of latent parameters; mean vector  $\pmb{\mu}$  and covariance matrix  $\pmb{\Sigma}$ .

Note that the base distribution is usually chosen as a standard Gaussian distribution (with  $\mu = 0$  and  $\pmb{\Sigma} = \pmb{I}_p$ ) in the literature for simplicity. However, the covariance  $\pmb{\Sigma}$  is essential in our imputation work to represent the inter-feature dependency.

Assumption 2: Given that the transformations involved in NF are feature-wise (e.g. Dinh et al., 2016; 2014), we expect NF to learn the mapping between the complex inter-feature dependency in the data space  $\mathcal{X}$  and the simple one in the latent space  $\mathcal{Z}$ .

# 3.3 ONLINE EM

EM is a class of iterative algorithms for latent variable models including missing data imputation(Little & Rubin, 2019). In EMFlow, it works in the latent space  $\mathcal{Z}$  where the underlying distribution is  $\mathcal{N}(\mathbf{z};\boldsymbol{\mu},\boldsymbol{\Sigma})$ . Given the embedding vectors  $\{\mathbf{z}_1,\dots ,\mathbf{z}_n\}$  where  $\mathbf{z}_i = f_{\psi}^{-1}(\mathbf{x}_i)$ , and the corresponding missing mask  $\{\mathbf{m}_1,\dots ,\mathbf{m}_n\}$ , EM aims to estimate  $(\boldsymbol{\mu},\boldsymbol{\Sigma})$  in an iterative way:

$$
\begin{array}{l} \hat {\boldsymbol {\mu}} ^ {(t + 1)} = g _ {\boldsymbol {\mu}} \left(\hat {\boldsymbol {\mu}} ^ {(t)}, \hat {\boldsymbol {\Sigma}} ^ {(t)}; \left\{\mathbf {z} _ {i}, \mathbf {m} _ {i} \right\} _ {i = 1} ^ {n}\right) \\ \widehat {\boldsymbol {\mu}} ^ {(t + 1)} = \left( \begin{array}{l l} & (t) \\ & \widehat {\boldsymbol {\mu}} ^ {(t)} \end{array} \right) \end{array} \tag {5}
$$

$$
\hat {\boldsymbol {\Sigma}} ^ {(t + 1)} = g _ {\boldsymbol {\Sigma}} \left(\hat {\boldsymbol {\mu}} ^ {(t)}, \hat {\boldsymbol {\Sigma}} ^ {(t)}; \left\{\mathbf {z} _ {i}, \mathbf {m} _ {i} \right\} _ {i = 1} ^ {n}\right)
$$

where  $(\widehat{\pmb{\mu}}^{(t)},\widehat{\pmb{\Sigma}}^{(t)})$  are the estimates at the  $t^{th}$  iteration, and  $\{g_{\pmb{\mu}}(\cdot),g_{\pmb{\Sigma}}(\cdot)\}$  denote the mappings between two consecutive iterations.

Given estimates  $(\widehat{\pmb{\mu}},\widehat{\pmb{\Sigma}})$ , the missing part  $\mathbf{z}_i^m$  is imputed by its conditional mean given the observed part  $\mathbf{z}_i^o$ :

$$
\widehat {\mathbf {z}} _ {i} ^ {m} = E \left(\mathbf {z} _ {i} ^ {m} \mid \mathbf {z} _ {i} ^ {o}; \widehat {\boldsymbol {\mu}}, \widehat {\boldsymbol {\Sigma}}\right) = \widehat {\boldsymbol {\mu}} _ {\mathbf {m} _ {i}} + \widehat {\boldsymbol {\Sigma}} _ {\mathbf {m} _ {i} \mathbf {o} _ {i}} \left(\widehat {\boldsymbol {\Sigma}} _ {\mathbf {o} _ {i} \mathbf {o} _ {i}}\right) ^ {- 1} \left(\mathbf {z} _ {i} ^ {o} - \widehat {\boldsymbol {\mu}} _ {\mathbf {o} _ {i}}\right) \tag {6}
$$

where  $\mathbf{o}_i$  is the observed mask (i.e. the complement of  $\mathbf{m}_i$ ), and the subscripts of  $(\widehat{\boldsymbol{\mu}},\widehat{\boldsymbol{\Sigma}})$  denote the slicing indexes.

When processing datasets of large volume, EM becomes impractical because it needs to read the whole data into the memory for each iteration. Following the framework introduced by Cappé & Moulines (2009), we derive an online version of EM algorithm in the context of data imputation. Let  $B \subset \{1, \dots, n\}$  denote a mini-batch of sample indexes, the online EM first obtains local estimates from a batch of data:

$$
\begin{array}{l} \widehat {\boldsymbol {\mu}} _ {l o c a l} = g _ {\boldsymbol {\mu}} \left(\widehat {\boldsymbol {\mu}} ^ {(t)}, \widehat {\boldsymbol {\Sigma}} ^ {(t)}; \left\{\mathbf {z} _ {i}, \mathbf {m} _ {i} \right\} _ {i \in B}\right) \\ \widehat {\boldsymbol {\Sigma}} = \left\{x _ {i} ^ {(t)}, \widehat {\boldsymbol {\Sigma}} ^ {(t)}; \left\{\mathbf {z} _ {i}, \mathbf {m} _ {i} \right\} _ {i \in B}\right) \end{array} \tag {7}
$$

$$
\widehat {\boldsymbol {\Sigma}} _ {l o c a l} = g _ {\boldsymbol {\Sigma}} \left(\widehat {\boldsymbol {\mu}} ^ {(t)}, \widehat {\boldsymbol {\Sigma}} ^ {(t)}; \left\{\mathbf {z} _ {i}, \mathbf {m} _ {i} \right\} _ {i \in B}\right)
$$

The global estimates are then updated in the fashion of weighted average:

$$
\begin{array}{c} \hat {\boldsymbol {\mu}} ^ {(t + 1)} = \rho_ {t + 1} \hat {\boldsymbol {\mu}} _ {\text {l o c a l}} + (1 - \rho_ {t + 1}) \hat {\boldsymbol {\mu}} ^ {(t)} \\ \widehat {\hat {\boldsymbol {\mu}}} ^ {(t + 1)} = \widehat {\hat {\boldsymbol {\mu}}} ^ {(t)} \end{array} \tag {8}
$$

$$
\widehat {\boldsymbol {\Sigma}} ^ {(t + 1)} = \rho_ {t + 1} \widehat {\boldsymbol {\Sigma}} _ {l o c a l} + (1 - \rho_ {t + 1}) \widehat {\boldsymbol {\Sigma}} ^ {(t)}
$$

![](images/38f95d83451e2528053bb6100e951817bdeb2e0c0b3f26a037beed4ee7e5d531.jpg)  
Figure 1: EMFlow architecture.

where  $(\rho_1,\rho_2,\ldots)$  are a sequence of step sizes satisfying

$$
0 <   \rho_ {t} <   1, \sum_ {i = 1} ^ {\infty} \rho_ {i} = \infty \text {a n d} \sum_ {i = 1} ^ {\infty} \rho_ {i} ^ {2} <   \infty \tag {9}
$$

In this work, we use a step size schedule defined by

$$
\rho_ {t} = C t ^ {- \gamma}, \quad t = 1, 2, \dots \tag {10}
$$

where  $C$  is a positive constant and  $\gamma \in (0.5,1]$ .

# 3.4 ARCHITECTURE AND INFERENCE

EMFlow is a composite framework that combines NF and online EM. As illustrated in Fig 1, NF is the bidirectional tunnel between the data space and the latent space, aiming to learn the complete data density  $p_{X}$ . In the latent space, the online EM estimates the inter-feature dependency of the embedding vectors and performs imputation. To address the issue that NF needs complete data vectors for computation, the incomplete data  $\mathbf{X}^{in}$  are imputed naively (e.g. median imputation for tabular datasets) at the very beginning to get the initial current imputed data  $\widehat{\mathbf{X}}$ . Afterwards, the objective in equation 2 is optimized in an iterative schema, where each iteration consists of a training phase and a re-imputation phase.

Training Phase At this phase, the current imputed data  $\widehat{\mathbf{X}}$  stay fixed, while the parameter estimates of NF (i.e.  $\widehat{\psi}$ ) and base distribution (i.e.  $\widehat{\mu}$  and  $\widehat{\Sigma}$ ) are updated in different ways.

First of all, given the current estimated base distribution  $\mathcal{N}(\cdot ;\widehat{\mu},\widehat{\Sigma})$ , the flow model  $f_{\psi}$  are learned by minimizing the negative log-likelihood of a batch of the current imputed data  $\widehat{\mathbf{X}}_B$ :

$$
L _ {1} (\psi) = - \frac {1}{| B |} \sum_ {i \in B} \log p _ {X} \left(\widehat {\mathbf {x}} _ {i}; \psi , \widehat {\boldsymbol {\mu}}, \widehat {\boldsymbol {\Sigma}}\right) \tag {11}
$$

where  $|B|$  denotes the batch size.

The computation of  $L_{1}$  requires exact likelihood evaluation that is equipped by NF. Once the flow model parameters get updated, we obtain the embedding vectors in the latent space:

$$
\mathbf {z} _ {i} = f _ {\hat {\psi}} ^ {- 1} \left(\hat {\mathbf {x}} _ {i}\right), \quad i \in B \tag {12}
$$

Although the embedding vectors are complete, they are treated as incomplete using the missing masks in the data space  $\{\mathbf{m}_i\}_{i\in B}$ , given that the invertible mapping parameterized by  $f_{\psi}$  is featurewise. Therefore, the online EM imputes the missing parts of the embedding vectors with the current global estimates  $(\widehat{\boldsymbol{\mu}},\widehat{\boldsymbol{\Sigma}})$ :

$$
\widehat {\mathbf {z}} _ {i} ^ {m} = E \left(\mathbf {z} _ {i} ^ {m} \mid \mathbf {z} _ {i} ^ {o}; \widehat {\boldsymbol {\mu}}, \widehat {\boldsymbol {\Sigma}}\right), \quad i \in B \tag {13}
$$

which results in new embedding vectors  $\{\widehat{\mathbf{z}}\}_{i\in B}$  where  $\widehat{\mathbf{z}}_i$  consists of the observed part  $\mathbf{z}_i^o$  and the imputed part  $\widehat{\mathbf{z}}_i^m$ . After the imputation, the global estimates  $(\widehat{\boldsymbol{\mu}},\widehat{\boldsymbol{\Sigma}})$  are also updated following equation 7 and equation 8.

Since the base distribution has been changed, it's necessary to update the flow model  $f_{\psi}$  again by optimizing a composite loss:

$$
L _ {2} (\psi) = - \frac {1}{| B |} \sum_ {i \in B} \left[ \log p _ {X} \left(\widetilde {\mathbf {x}} _ {i}; \psi , \widehat {\boldsymbol {\mu}}, \widehat {\boldsymbol {\Sigma}}\right) - \alpha L _ {\text {r e c}} \left(\widetilde {\mathbf {x}} _ {i}, \widehat {\mathbf {x}} _ {i}, \mathbf {m} _ {i}\right) \right] \tag {14}
$$

where  $\widetilde{\mathbf{x}}_i = f_{\psi}(\widehat{\mathbf{z}}_i)$ , and  $L_{\mathrm{rec}}(\widetilde{\mathbf{x}}_i, \widehat{\mathbf{x}}_i, \mathbf{m}_i)$  is the reconstruction error only for non-missing values:

$$
L _ {\mathrm {r e c}} \left(\widetilde {\mathbf {x}} _ {i}, \widehat {\mathbf {x}} _ {i}, \mathbf {m} _ {i}\right) = \sum_ {j = 1} ^ {p} \left(1 - \mathbf {m} _ {i j}\right) \left(\widetilde {\mathbf {x}} _ {i j} - \widehat {\mathbf {x}} _ {i j}\right) ^ {2} \tag {15}
$$

In this composite loss, the first term forces the reconstructed data vectors  $\{\widetilde{\mathbf{x}}\}_{i\in B}$  to have high likelihood in the data space, while the second term encourages  $\{\widetilde{\mathbf{x}}\}_{i\in B}$  to match the observed parts of the incomplete training data. And since  $\{\widetilde{\mathbf{x}}\}_{i\in B}$  are transformed from  $\{\widehat{\mathbf{z}}\}_{i\in B}$  via  $f_{\psi}$ , both terms are conditioned on the inter-feature dependency learned by EM. A pseudocode for this training phase is presented in Algorithm 1, and the implementation details that aim to make the training phase more stable is presented in appendix D.

# Algorithm 1 Training Phase

1: Input: Current imputation:  $\widehat{\mathbf{X}} = (\widehat{\mathbf{x}}_1, \ldots, \widehat{\mathbf{x}}_n)^T$ , missing masks  $\mathbf{M} = (\mathbf{m}_1, \ldots, \mathbf{m}_n)^T$ , initial estimates of the base distribution  $(\widehat{\pmb{\mu}}^{(0)}, \widehat{\pmb{\Sigma}}^{(0)})$ , online EM step size sequence:  $\rho_1, \rho_2, \ldots, \rho_t, \ldots$  
2: for  $t = 1$  to  $T_{\mathrm{epoch}}$  do  
3: Get a mini-batch  $\widehat{\mathbf{X}}_B = \{\widehat{\mathbf{x}}_i\}_{i\in B}$  
4: #update the flow model  
5: Compute  $L_{1}$  in equation 11.  
6: Update  $\psi$  via gradient descent  
7: #update the base distribution  
8:  $\mathbf{z}_i = f_{\psi}^{-1}(\hat{\mathbf{x}}_i),\quad i\in B$  
9: Impute in the latent space with  $(\widehat{\pmb{\mu}}^{(t - 1)},\widehat{\pmb{\Sigma}}^{(t - 1)})$  to get  $\{\widehat{\mathbf{z}}_i\}_{i\in B}$  via equation 13  
10: Obtain updated  $(\widehat{\pmb{\mu}}^{(t)},\widehat{\pmb{\Sigma}}^{(t)})$  via equation 7 and equation 8  
11: #update the flow model again  
12:  $\widetilde{\mathbf{x}}_i = f_{\psi}(\widehat{\mathbf{z}}_i),\quad i\in B$  
13: Compute  $L_{2}$  via equation 14  
14: Update  $\psi$  via gradient descent

Re-Imputation Phase After the training phase, the re-imputation phase is executed to update the current imputation  $\widehat{\mathbf{X}}$ . This procedure is similar to that in the training phase, except that all model parameters are kept fixed. As shown by the last line of Algorithm 2, the missing parts of  $\widehat{\mathbf{X}}$  are replaced with those of the reconstructed data vectors, while the observed parts are kept the same.

# Algorithm 2 Re-imputation Phase

1: Input: Current imputation:  $\widehat{\mathbf{X}} = (\widehat{\mathbf{x}}_1, \ldots, \widehat{\mathbf{x}}_n)^T$ , missing masks  $\mathbf{M} = (\mathbf{m}_1, \ldots, \mathbf{m}_n)^T$ , estimates of the base distribution  $(\widehat{\boldsymbol{\mu}}, \widehat{\boldsymbol{\Sigma}})$  from the previous training phase  
2: for  $i = 1$  to  $n$  do  
3:  $\mathbf{z}_i = f_{\psi}^{-1}(\hat{\mathbf{x}}_i)$  
4: Impute in the latent space with  $(\widehat{\pmb{\mu}},\widehat{\pmb{\Sigma}})$  to get  $\widehat{\mathbf{z}}_i$  via equation 13  
5:  $\widetilde{\mathbf{x}}_i = f_{\psi}(\widehat{\mathbf{z}}_i)$  
6: #update current imputation  
7:  $\widehat{\mathbf{x}}_i = \widehat{\mathbf{x}}_i\odot \mathbf{o}_i + \widetilde{\mathbf{x}}_i\odot \mathbf{m}_i$

Table 1: Imputation results on UCI datasets - RMSE (lower is better).  

<table><tr><td rowspan="2">Data</td><td colspan="3">MCAR</td><td colspan="3">MAR</td></tr><tr><td>EMFlow</td><td>MCFlow</td><td>GAIN</td><td>EMFlow</td><td>MCFlow</td><td>GAIN</td></tr><tr><td>News</td><td>.139 ± .001</td><td>.167 ± .0015</td><td>.197 ± .005</td><td>.172 ± .000</td><td>.181 ± .001</td><td>.271 ± .039</td></tr><tr><td>Air</td><td>.097 ± .005</td><td>.111 ± .004</td><td>.127 ± .005</td><td>.040 ± .001</td><td>.055 ± .001</td><td>.061 ± .023</td></tr><tr><td>Letter</td><td>.111 ± .001</td><td>.121 ± .000</td><td>.127 ± .001</td><td>.110 ± .001</td><td>.127 ± .001</td><td>.166 ± .040</td></tr><tr><td>Concrete</td><td>.147 ± .004</td><td>.233 ± .007</td><td>.194 ± .008</td><td>.133 ± .006</td><td>.198 ± .010</td><td>.184 ± .012</td></tr><tr><td>Review</td><td>.229 ± .003</td><td>.234 ± .004</td><td>.278 ± .005</td><td>.194 ± .004</td><td>.191 ± .004</td><td>.234 ± .014</td></tr><tr><td>Credit</td><td>.125 ± .001</td><td>.135 ± .002</td><td>.131 ± .002</td><td>.024 ± .001</td><td>.028 ± .002</td><td>.029 ± .002</td></tr><tr><td>Energy</td><td>.086 ± .001</td><td>.092 ± .001</td><td>.110 ± .002</td><td>.175 ± .002</td><td>.176 ± .002</td><td>.250 ± .019</td></tr><tr><td>CTG</td><td>.104 ± .006</td><td>.140 ± .005</td><td>.143 ± .008</td><td>.105 ± .001</td><td>.153 ± .003</td><td>.165 ± .006</td></tr><tr><td>Song</td><td>.025 ± .000</td><td>.030 ± .006</td><td>.034 ± .002</td><td>.024 ± .000</td><td>.031 ± .014</td><td>.028 ± .001</td></tr><tr><td>Wine</td><td>.076 ± .001</td><td>.098 ± .002</td><td>.097 ± .002</td><td>.102 ± .002</td><td>.124 ± .003</td><td>.135 ± .007</td></tr></table>

![](images/8dd186c5f7aa72cd796f873249f43756222b1d5d8a0d162045dd009fe0871d55.jpg)

![](images/7fd469bad3c216c57cb441ddfd101efe39bec37168cf24d33503a4ed76c572c3.jpg)

![](images/e2669b48007477d9858efe2475f8cc72bad626bfe4265eafb3e6e775e4a9cfa3.jpg)

![](images/f333c1d3c37aa483e55a7514f0cc4079ab3a29989ab818810749dc0591c619b4.jpg)  
Figure 2: Comparison of convergence speed in terms of the training loss and test set RMSE on three UCI datasets.

![](images/3a6cc4187827e4fc337d19cf8cece2620b06a4e05bf86f3d5766d7e8eb1bd588.jpg)

![](images/a7f58640219bb2699896286e8d9bfda8c33f8fefcf1fdf6f218d5ce0e47e2aeb.jpg)

# 4 EXPERIMENTS

In this section, we evaluate the performance of EMFlow on multivariate and image datasets in terms of the imputation quality and and the speed of model training. Its performance is compared to that of MCFlow, the most related competitor that has been shown to be superior to other state of art methods (Richardson et al., 2020).

Table 2: Imputation results on image datasets - RMSE (lower is better)  

<table><tr><td></td><td>Missing Rate</td><td>.1</td><td>.2</td><td>.3</td><td>.4</td><td>.5</td><td>.6</td><td>.7</td><td>.8</td><td>.9</td></tr><tr><td rowspan="4">MNIST</td><td>GAIN</td><td>.1029</td><td>.1184</td><td>.1399</td><td>.1495</td><td>.1723</td><td>.1794</td><td>.2167</td><td>.2200</td><td>.2710</td></tr><tr><td>MisGAN</td><td>.1083</td><td>.1117</td><td>.1184</td><td>.1227</td><td>.1311</td><td>.1388</td><td>.1512</td><td>.1906</td><td>.2621</td></tr><tr><td>MCFlow</td><td>.0835</td><td>.0879</td><td>.0894</td><td>.0941</td><td>.1027</td><td>.1119</td><td>.1251</td><td>.1463</td><td>.2020</td></tr><tr><td>EMFlow</td><td>.0726</td><td>.0775</td><td>.0832</td><td>.0901</td><td>.0986</td><td>.1100</td><td>.1260</td><td>.1504</td><td>.1951</td></tr><tr><td rowspan="4">CIFAR-10</td><td>GAIN</td><td>.1025</td><td>.1090</td><td>.1103</td><td>.1073</td><td>.1094</td><td>.1202</td><td>.1217</td><td>.1426</td><td>.5388</td></tr><tr><td>MisGAN</td><td>.1577</td><td>.1434</td><td>.1478</td><td>.1326</td><td>.1588</td><td>.1824</td><td>.2036</td><td>.2660</td><td>.3011</td></tr><tr><td>MCFlow</td><td>.1083</td><td>.1112</td><td>.1179</td><td>.1273</td><td>.1340</td><td>.1387</td><td>.1466</td><td>.1552</td><td>.1702</td></tr><tr><td>EMFlow</td><td>.0444</td><td>.0479</td><td>.0525</td><td>.0575</td><td>.0619</td><td>.0689</td><td>.0782</td><td>.0926</td><td>.1188</td></tr></table>

Table 3: Classification accuracy on imputed image datasets (higher is better)  

<table><tr><td></td><td>Missing Rate</td><td>.1</td><td>.2</td><td>.3</td><td>.4</td><td>.5</td><td>.6</td><td>.7</td><td>.8</td><td>.9</td></tr><tr><td rowspan="2">MNIST</td><td>MCFlow</td><td>.9894</td><td>.9878</td><td>.9878</td><td>.9871</td><td>.9840</td><td>.9806</td><td>.9659</td><td>.9331</td><td>.7732</td></tr><tr><td>EMFlow</td><td>.9894</td><td>.9884</td><td>.9882</td><td>.9878</td><td>.9860</td><td>.9824</td><td>.9696</td><td>.9253</td><td>.7502</td></tr><tr><td rowspan="2">CIFAR-10</td><td>MCFlow</td><td>.8352</td><td>.7081</td><td>.5525</td><td>.4166</td><td>.3406</td><td>.2820</td><td>.2476</td><td>.2194</td><td>.1875</td></tr><tr><td>EMFlow</td><td>.9085</td><td>.8974</td><td>.8783</td><td>.8535</td><td>.8116</td><td>.7446</td><td>.6214</td><td>.4868</td><td>.3127</td></tr></table>

To make the comparison more objective, both models use the same normalizing flow with six affine coupling layers. We also follow the authors' suggestion for the hyperparameter selection of MCFlow throughout this section. Additionally, we also present the benchmarks of other state-of-art models including GAIN (Yoon et al., 2018) and MisGAN (Li et al., 2019) for more comprehensive comparisons.

# 4.1 MULTIVARIATE DATASETS

Ten multivariate datasets from the UCI repository (Dheeru & Taniskidou, 2017) are used for evaluation. For all of them, each feature is scaled to fit inside the interval [0, 1] via min-max normalization. We simulate MCAR with a missing rate of 0.2 by removing each value independently according to a Bernoulli distribution. We also simulate MAR scenario where the missing probability of the last  $30\%$  features depends on the values of the first  $70\%$  features<sup>3</sup>.

The initial imputation is performed by randomly sampling from the observed entries of each feature. All experiments are conducted using five-fold cross validation where the test set only goes through the re-imputation phase in each iteration. The choices of hyperparameters are detailed in appendix F, where we also show that EMFlow is not sensitive to the choice of hyperparameters.

Results The imputation performance is evaluated by calculating the Root Mean Squared Error (RMSE) between the imputed and true values. As shown in Table 1, EMFlow performs constantly better than MCFlow under both MCAR and MAR settings for nearly all datasets.

Additionally, We trained EMFlow and MCFlow on the same machine with the same learning rate and batch size to compare the convergence speed. Figure 2 shows the training loss and the test set RMSE over time on three UCI datasets. It shows that EMFlow converges significantly faster than MCFlow. In fact, EMFlow converges within three iterations for most of the UCI datasets.

# 4.2 IMAGE DATASETS

We also evaluate EMFlow on MNIST and CIFAR-10. MNIST is a dataset of  $28 \times 28$  grayscale images of handwritten digits (LeCun et al., 1998), and CIFAR-10 is a dataset of  $32 \times 32$  colorful images from 10 classes (Krizhevsky et al., 2009). For both datasets, the pixel values of each image are scaled to

![](images/3b26e8666ff96ea027707a0f4d9a6db3fa47830912a202ac29328cdc6fa1a187.jpg)  
Figure 3: Sample imputed images for CIFAR-10 at missing rates of 0.5 and 0.9.

![](images/a8800c109cee0c64d975952bcf238642eafa02a9f24956769e39eef284b2ea6e.jpg)

[0, 1]. In this section, we simulate MCAR where each pixel is independently missing with various probabilities from 0.1 to 0.9.

The initial imputation is performed by nearest-neighbor sampling where a missing pixel is filled by one of its nearest observed neighbors. In our experiments, the standard 60,000/10,000 and 50,000/10,000 training-test set partitions are used for MNIST and CIFAR-10 respectively. The choices of hyperparameters are detailed in appendix F.

Results Table 2 shows the RMSE of all considered methods on both image datasets. In the case of MNIST, EMFlow and MCFlow have similar RMSE and outperform other methods, while MCFlow starts to gain slight advantage under high missing rates. In the case of CIFAR-10, EMFlow achieve much lower RMSE than all competing methods.

To further demonstrate the efficiency of EMFlow, We also compare EMFlow and MCFlow with respect to the accuracy of post-imputation classification. For this purpose, a LeNet-based model and a VGG19 model were trained on the original training sets of MNIST and CIFAR-10 respectively. These models then made predictions on the imputed test sets under different missing rates. Table 3 shows that EMFlow yields slightly better post-imputation prediction accuracy than MCFlow on MNIST, while the improvement is much more significant on CIFAR-10. We note that these findings are in good agreement with the RMSE results.

To qualitatively compare the imputation quality of EMFlow and MCFlow, Figure 3 shows sample imputed images from CIFAR-10 with MCAR missing rates at 0.5 and 0.9. The first row includes the (complete) ground truth images for reference, while the second row includes the (incomplete) observed images on which the models were trained. The last two rows showcase the reconstructed images by MCFlow and EMFlow, respectively. It's clear that EMFlow performs better than MCFlow by recovering more details and displaying sharper boundaries and cleaner background.

# 5 CONCLUSION

We propose a novel architecture EMFlow for missing data imputation. It combines the strength of the online EM and the normalizing flow to learn the density estimation in the presence of incomplete data while performing imputation. Various experiments with multivariate and image datasets show that EMFlow significantly outperforms its state-of-art competitor with respect to imputation accuracy as well as the convergence speed under a wide range of missing rates and different missing mechanisms. The accuracy of post-imputation classification on image datasets also demonstrates the superior EMFlow's ability of recovering semantic structure from incomplete data.

# REPRODUCIBILITY STATEMENT

The code and data to reproduce the results in this work are included in the supplementary materials.

# REFERENCES

Vincent Audigier, François Husson, and Julie Josse. Multiple imputation for continuous variables using a bayesian principal component analysis. Journal of statistical computation and simulation, 86(11):2140-2156, 2016.  
Marcelo Bertalmio, Guillermo Sapiro, Vincent Caseles, and Coloma Ballester. Image inpainting. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pp. 417-424, 2000.  
AE Brockwell. Universal residuals: A multivariate transformation. Statistics & probability letters, 77(14):1473-1478, 2007.  
Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519, 2015.  
Stephen Burgess, Ian R White, Matthieu Resche-Rigon, and Angela M Wood. Combining multiple imputation and meta-analysis with individual participant data. Statistics in medicine, 32(26): 4499-4514, 2013.  
S van Buuren and Karin Groothuis-Oudshoorn. mice: Multivariate imputation by chained equations in r. Journal of statistical software, pp. 1-68, 2010.  
Olivier Cappé and Eric Moulines. On-line expectation-maximization algorithm for latent data models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 71(3):593-613, 2009.  
Yilun Chen, Ami Wiesel, and Alfred O Hero. Robust shrinkage estimation of high-dimensional covariance matrices. IEEE Transactions on Signal Processing, 59(9):4097-4107, 2011.  
Mark Collier, Alfredo Nazabal, and Christopher KI Williams. Vaes in the presence of missing data. arXiv preprint arXiv:2006.05301, 2020.  
Dua Dheeru and E Karra Taniskidou. Uci machine learning repository. 2017.  
Marco Di Zio, Ugo Guarnera, and Orietta Luzi. Imputation through finite gaussian mixture models. Computational Statistics & Data Analysis, 51(11):5305-5316, 2007.  
Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.  
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. arXiv preprint arXiv:1605.08803, 2016.  
Jianqing Fan, Yingying Fan, and Jinchi Lv. High dimensional covariance matrix estimation using a factor model. Journal of Econometrics, 147(1):186-197, 2008.  
BJ Ferdosi, H Buddelmeijer, SC Trager, MHF Wilkinson, and JBTM Roerdink. Comparison of density estimation methods for astronomical datasets. *Astronomy & Astrophysics*, 531:A114, 2011.  
Pedro J García-Laencina, José-Luis Sancho-Gómez, and Aníbal R Figueiras-Vidal. Pattern classification with missing data: a review. Neural Computing and Applications, 19(2):263-282, 2010.  
Lovedeep Gondara and Ke Wang. Multiple imputation using deep denoising autoencoders. arXiv preprint arXiv:1705.02737, 2017.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.  
Niels Bruun Ipsen, Pierre-Alexandre Mattei, and Jes Frellsen. not-miwae: Deep generative modelling with missing not at random data. arXiv preprint arXiv:2006.12871, 2020.

Diederik P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible  $1 \times 1$  convolutions. arXiv preprint arXiv:1807.03039, 2018.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
Ranjit Lall. How multiple imputation makes a difference. Political Analysis, 24(4):414-433, 2016.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Steven Cheng-Xian Li, Bo Jiang, and Benjamin Marlin. Misgan: Learning from incomplete data with generative adversarial networks. arXiv preprint arXiv:1902.09599, 2019.  
Roderick JA Little and Donald B Rubin. Statistical analysis with missing data, volume 793. John Wiley & Sons, 2019.  
Xinyue Liu, Chara Aggarwal, Yu-Feng Li, Xiaugnan Kong, Xinyuan Sun, and Saket Sathe. Kernelized matrix factorization for collaborative filtering. In Proceedings of the 2016 SIAM International Conference on Data Mining, pp. 378-386. SIAM, 2016.  
Chao Ma, Sebastian Tschiatschek, Konstantina Palla, José Miguel Hernández-Lobato, Sebastian Nowozin, and Cheng Zhang. Eddi: Efficient dynamic discovery of high-value information with partial vae. arXiv preprint arXiv:1809.11142, 2018.  
Jinwen Ma, Lei Xu, and Michael I Jordan. Asymptotic convergence rate of the em algorithm for gaussian mixtures. Neural Computation, 12(12):2881-2907, 2000.  
Pierre-Alexandre Mattei and Jes Frellsen. Miwae: Deep generative modelling and imputation of incomplete data sets. In International Conference on Machine Learning, pp. 4413-4423. PMLR, 2019.  
Xiao-Li Meng. Multiple-imputation inferences with uncongenial sources of input. Statistical Science, pp. 538-558, 1994.  
Xiao-Li Meng et al. On the rate of convergence of the ccm algorithm. The Annals of Statistics, 22 (1):326-339, 1994.  
Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International Conference on Machine Learning, pp. 1530-1538. PMLR, 2015.  
Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In International conference on machine learning, pp. 1278-1286. PMLR, 2014.  
Trevor W Richardson, Wencheng Wu, Lei Lin, Beilei Xu, and Edgar A Bernal. Mcflow: Monte carlo flow models for data imputation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14205-14214, 2020.  
Guido Sanguinetti and Neil D Lawrence. Missing data in kernel pca. In European Conference on Machine Learning, pp. 751-758. Springer, 2006.  
Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th international conference on World Wide Web, pp. 285-295, 2001.  
Peter Schmitt, Jonas Mandel, and Mickael Guedj. A comparison of six methods for missing data imputation. Journal of Biometrics & Biostatistics, 6(1):1, 2015.  
RS Somasundaram and R Nedunchezhian. Evaluation of three simple imputation methods for enhancing preprocessing of data with missing values. International Journal of Computer Applications, 21(10):14-19, 2011.

Daniel J Stekhoven and Peter Buhlmann. Missforest—non-parametric missing value imputation for mixed-type data. Bioinformatics, 28(1):112-118, 2012.  
Takeshi Teshima, Isao Ishikawa, Koichi Tojo, Kenta Oono, Masahiro Ikeda, and Masashi Sugiyama. Coupling-based invertible neural networks are universal diffeomorphism approximators. arXiv preprint arXiv:2006.11469, 2020.  
CF Jeff Wu. On the convergence properties of the em algorithm. The Annals of statistics, pp. 95-103, 1983.  
Hai xian Wang, Quan bing Zhang, Bin Luo, and Sui Wei. Robust mixture modelling using multivariate t-distribution with missing information. Pattern Recognition Letters, 25(6):701-710, 2004.  
Junyuan Xie, Linli Xu, and Enhong Chen. Image denoising and inpainting with deep neural networks. Advances in neural information processing systems, 25:341-349, 2012.  
Raymond Yeh, Chen Chen, Teck Yian Lim, Mark Hasegawa-Johnson, and Minh N Do. Semantic image inpainting with perceptual and contextual losses. arXiv preprint arXiv:1607.07539, 2(3), 2016.  
Jinsung Yoon, James Jordon, and Mihaela Schaar. Gain: Missing data imputation using generative adversarial nets. In International Conference on Machine Learning, pp. 5689-5698. PMLR, 2018.  
Adriano Z Zambom and Dias Ronaldo. A review of kernel density estimation with applications to econometrics. International Econometric Review, 5(1):20-42, 2013.  
Ruofei Zhao, Yuanzhi Li, Yuekai Sun, et al. Statistical convergence of the em algorithm on gaussian mixture models. Electronic Journal of Statistics, 14(1):632-660, 2020.
