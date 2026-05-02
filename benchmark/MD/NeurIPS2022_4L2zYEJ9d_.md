# CARD: Classification and Regression Diffusion Models

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Learning the distribution of a continuous or categorical response variable  $\mathbf{y}$  given its covariates  $\mathbf{x}$  is a fundamental problem in statistics and machine learning. Deep neural network-based supervised learning algorithms have made great progress in predicting the mean of  $\mathbf{y}$  given  $\mathbf{x}$ , but they are often criticized for their ability to accurately capture the uncertainty of their predictions. In this paper, we introduce classification and regression diffusion (CARD) models, which combine a denoising diffusion-based conditional generative model and a pre-trained conditional mean estimator, to accurately predict the distribution of  $\mathbf{y}$  given  $\mathbf{x}$ . We demonstrate the outstanding ability of CARD in conditional distribution prediction with both toy examples and real world datasets, the experimental results on which show that CARD in general outperforms state-of-the-art methods, including Bayesian neural network based one, designed for uncertainty estimation, especially when the conditional distribution of  $\mathbf{y}$  given  $\mathbf{x}$  is multi-modal.

# 1 Introduction

A fundamental problem in statistics and machine learning is to predict the response variable  $\mathbf{y}$  given a set of covariates  $\mathbf{x}$ . Generally speaking,  $\mathbf{y}$  is a continuous variable for regression analysis and a categorical variable for classification. Denote  $f(\mathbf{x})\in \mathbb{R}^{C}$  as a deterministic function that transforms  $\mathbf{x}$  into a  $C$  dimensional output. Denote  $f_{c}(\mathbf{x})$  as the  $c$ -th dimension of  $f(\mathbf{x})$ . Existing methods typically assume an additive noise model: for regression analysis with  $\mathbf{y}\in \mathbb{R}^C$ , one often assumes  $\mathbf{y} = f(\mathbf{x}) + \epsilon$ ,  $\epsilon \sim \mathcal{N}(0,\Sigma)$ , while for classification with  $y\in \{1,\dots,C\}$ , one often assumes  $y = \arg \max \left(f_1(\mathbf{x}) + \epsilon_1,\ldots ,f_C(\mathbf{x}) + \epsilon_C\right)$ , where  $\epsilon_c\stackrel {iid}{\sim}\mathrm{EV}_1(0,1)$ , a standard type-1 extreme value distribution. Thus we have the expected value of  $\mathbf{y}$  given  $\mathbf{x}$  as  $\mathbb{E}[\mathbf{y}|\mathbf{x}] = f(\mathbf{x})$  in regression and  $P(y = c|\mathbf{x}) = \mathbb{E}[y = c|\mathbf{x}] = \mathrm{softmax}_c(f(\mathbf{x})) = \frac{\exp(f_c(\mathbf{x}))}{\sum_{c'=1}^{C}\exp(f_{c'}(\mathbf{x}))}$  in classification.

These additive-noise models are primarily focusing on accurately estimating the conditional mean  $\mathbb{E}[\mathbf{y}|\mathbf{x}]$ , while paying less attention to whether the noise distribution can accurately capture the uncertainty of  $\mathbf{y}$  given  $\mathbf{x}$ . For this reason, they may not work well if the distribution of  $\mathbf{y}$  given  $\mathbf{x}$  clearly deviates from the additive-noise assumption. For example, if  $p(\mathbf{y}|\mathbf{x})$  is multi-modal, which commonly happens when there are missing categorical covariates in  $\mathbf{x}$ , then  $\mathbb{E}[\mathbf{y}|\mathbf{x}]$  may not be close to any possible true values of  $\mathbf{y}$  given that specific  $\mathbf{x}$ . More specifically, consider a person whose weight, height, blood pressure, and age are known but gender is unknown, then the testosterone level of this person is likely to follow a bimodal distribution and the chance of developing breast cancer is also likely to follow a bimodal distribution. Therefore, these widely used additive-noise models, which use a deterministic function  $f(\mathbf{x})$  to characterize the conditional mean of  $\mathbf{y}$ , are inherently restrictive in their ability for uncertainty estimation.

In this paper, our goal is to accurately recover the full distribution of  $\mathbf{y}$  conditioning on  $\mathbf{x}$  given a set of  $N$  training data points, denoted as  $\mathcal{D} = \{(\mathbf{x}_i,\mathbf{y}_i)\}_{1,N}$ . To realize this goal, we consider the diffusion-based (a.k.a. score-based) generative models [23, 11, 24, 25] and inject covariate-dependence into

both the forward (inference) and reverse (generative) diffusion chains. Our method can model the conditional distribution of both continuous and categorical y variable, and the algorithms developed under this method will be collectively referred to as Classification And Regression Diffusion (CARD) models.

Diffusion-based generative models have received significant recent attention due to not only their ability to generate high-dimensional data, such as high-resolution photo-realistic images, but also their training stability. They can be understood from the perspective of score matching [12, 27] and Langevin dynamics [16, 28], as pioneered by Song and Ermon [23]. They can also be understood from the perspective of diffusion probabilistic models [22, 11], which first define a forward diffusion to transform the data into noise and then an inverse diffusion to regenerate the data from noise.

These previous methods mainly focus on unconditional generative modeling. While there exist guided-diffusion models [23, 25, 4, 17, 21] that target on generating high-resolution photo-realistic images that match the semantic meanings or content of the label, text, or corrupted-images, we focus on studying diffusion-based conditional generative modeling at a more fundamental level. In particular, our goal is to thoroughly investigate whether CARD can help accurately recovery  $p(\mathbf{y} \mid \mathbf{x}, \mathcal{D})$ , the predictive distribution of  $\mathbf{y}$  given  $\mathbf{x}$  after observing data  $\mathcal{D}$ . In other words, our focus is on regression analysis of continuous or categorical response variables given their corresponding covariates.

We summarize our main contributions as follows: 1) We show CARD, which injects covariate-dependence and a pre-trained conditional mean estimator into both the forward and reverse diffusion chains to construct a denoising diffusion probabilistic model, provides an accurate estimation of  $p(\mathbf{y} \mid \mathbf{x}, \mathcal{D})$ . 2) We provide a new metric to better evaluate how well a regression model captures the full distribution  $p(\mathbf{y} \mid \mathbf{x}, \mathcal{D})$ . 3) Experiments on standard benchmarks for regression analysis show that CARD achieves state-of-the-art results, using both existing metrics and the new one.

# 2 Methods and Algorithms for CARD

# 2.1 Problem Statement

Given the ground-truth response variable  $\mathbf{y}_0$  and its covariates  $\mathbf{x}$ , and assuming a sequence of intermediate uncertain prediction  $\mathbf{y}_{1:T}$  made by the diffusion model, the goal of supervised learning is to learn a good model such that the log-likelihood is maximized by optimizing the following ELBO:

$$
\log p _ {\theta} (\mathbf {x}, \mathbf {y} _ {0}) = \log \int p _ {\theta} (\mathbf {x}, \mathbf {y} _ {0: T}) d \mathbf {y} _ {1: T} \geq \mathbb {E} _ {q \left(\mathbf {y} _ {1: T} \mid \mathbf {x}\right)} \left[ \log \frac {p _ {\theta} \left(\mathbf {y} _ {0 : T} \mid \mathbf {x}\right)}{q \left(\mathbf {y} _ {1 : T} \mid \mathbf {y} _ {0} , \mathbf {x}\right)} \right], \tag {1}
$$

where  $q(\mathbf{y}_{1:T}|\mathbf{y}_0,\mathbf{x})$  is called the forward process or diffusion process in the concept of diffusion models [11]. Denoting  $D_{\mathrm{KL}}(q||p)$  as the Kullback-LeiblerKL (KL) divergence from distributions  $p$  to  $q$ . The above objective can be rewritten as:

$$
\mathcal {L} _ {\mathrm {E L B O}} (\mathbf {x}) := \mathcal {L} _ {0} (\mathbf {x}) + \sum_ {t = 2} ^ {T} \mathcal {L} _ {t - 1} (\mathbf {x}) + \mathcal {L} _ {T} (\mathbf {x}), \tag {2}
$$

$$
\mathcal {L} _ {0} (\mathbf {x}) := \mathbb {E} _ {q} \left[ - \log p _ {\theta} \left(\mathbf {y} _ {0} \mid \mathbf {y} _ {1}, \mathbf {x}\right) \right], \tag {3}
$$

$$
\mathcal {L} _ {t - 1} (\mathbf {x}) := \mathbb {E} _ {q} \left[ \right. D _ {\mathrm {K L}} \left( \right.q \left(\mathbf {y} _ {t - 1} \mid \mathbf {y} _ {t}, \mathbf {y} _ {0}, \mathbf {x}\right)\left. \right\rvert   | p _ {\theta} \left(\mathbf {y} _ {t - 1} \mid \mathbf {y} _ {t}, \mathbf {x}\right)\left. \right)\left. \right], \tag {4}
$$

$$
\mathcal {L} _ {T} (\mathbf {x}) := \mathbb {E} _ {q} \left[ D _ {\mathrm {K L}} \left(q \left(\mathbf {y} _ {T} \mid \mathbf {y} _ {0}, \mathbf {x}\right) \mid p \left(\mathbf {y} _ {T} \mid \mathbf {x}\right)\right) \right]. \tag {5}
$$

Here we follow the convention to assume  $\mathcal{L}_T$  does not depend on any parameter and it will be close to zero by carefully diffusing the observed response variable  $\mathbf{y}_0$  towards a pre-assumed distribution  $p(\mathbf{y}_T|\mathbf{x})$ . The rest of terms will make sure the model  $p_{\theta}(\mathbf{y}_{t - 1}|\mathbf{y}_t,\mathbf{x})$  predicts at different diffusion steps and we hope this gives us different scales in uncertainty measure. Different than vanilla diffusion model, we assume the endpoint of the diffusion as:

$$
p (\mathbf {y} _ {T}) = \mathcal {N} \left(f _ {\phi} (\mathbf {x}), \boldsymbol {I}\right). \tag {6}
$$

where  $f_{\phi}(\mathbf{x})$  is pre-knowledge of the relation between  $\mathbf{x}$  and  $\mathbf{y}_0$ , e.g., pre-trained with  $\mathcal{D}$  to approximate  $\mathbb{E}[\mathbf{y}|\mathbf{x}]$ , or  $\mathbf{0}$  if we assume the relation is unknown. With a diffusion schedule  $\{\beta_t\}_{t=1:T} \in (0,1)^T$ , we specify the forward diffusion process conditional distributions in a similar fashion as [18], but for all timesteps including  $t = 1$ :

$$
q \left(\mathbf {y} _ {t} \mid \mathbf {y} _ {t - 1}, f _ {\phi} (\mathbf {x})\right) = \mathcal {N} \left(\mathbf {y} _ {t}; \sqrt {1 - \beta_ {t}} \mathbf {y} _ {t - 1} + (1 - \sqrt {1 - \beta_ {t}}) f _ {\phi} (\mathbf {x}), \beta_ {t} \boldsymbol {I}\right), \tag {7}
$$

which admits a closed form sampling distribution with arbitrary timestep  $t$ :

$$
q \left(\mathbf {y} _ {t} \mid \mathbf {y} _ {0}, f _ {\phi} (\mathbf {x})\right) = \mathcal {N} \left(\mathbf {y} _ {t}; \sqrt {\bar {\alpha} _ {t}} \mathbf {y} _ {0} + (1 - \sqrt {\bar {\alpha} _ {t}}) f _ {\phi} (\mathbf {x}), (1 - \bar {\alpha} _ {t}) I\right), \tag {8}
$$

where  $\alpha_{t} = 1 - \beta_{t}$  and  $\bar{\alpha}_{t} = \prod_{t}\alpha_{t}$ . Note that Eq. (7) can be viewed as an interpolation between true data  $\mathbf{y}_0$  and predicted conditional expectation  $f_{\phi}(\mathbf{x})$ , that gradually changes from the former to the latter throughout the forward process.  
Such formulation corresponds to a tractable forward process posterior:

$$
q \left(\mathbf {y} _ {t - 1} \mid \mathbf {y} _ {t}, f _ {\phi} (\mathbf {x})\right) = \mathcal {N} \left(\mathbf {y} _ {t - 1}; \tilde {\mu} \left(\mathbf {y} _ {t}, f _ {\phi} (\mathbf {x})\right), \tilde {\beta} _ {t} \boldsymbol {I}\right), \tag {9}
$$

83 where

$$
\begin{array}{l} \tilde {\boldsymbol {\mu}} \big (\mathbf {y} _ {t}, f _ {\phi} (\mathbf {x}) \big) := \underbrace {\frac {\beta_ {t} \sqrt {\bar {\alpha} _ {t - 1}}}{1 - \bar {\alpha} _ {t}}} _ {\gamma_ {0}} \mathbf {y} _ {0} + \underbrace {\frac {(1 - \bar {\alpha} _ {t - 1}) \sqrt {\alpha_ {t}}}{1 - \bar {\alpha} _ {t}}} _ {\gamma_ {1}} \mathbf {y} _ {t} + \underbrace {\left(1 + \frac {(\sqrt {\bar {\alpha} _ {t}} - 1) (\sqrt {\alpha_ {t}} + \sqrt {\bar {\alpha} _ {t - 1}})}{1 - \bar {\alpha} _ {t}}\right)} _ {\gamma_ {2}} f _ {\phi} (\mathbf {x}), \\ \tilde {\beta} _ {t} := \frac {1 - \bar {\alpha} _ {t - 1}}{1 - \bar {\alpha} _ {t}} \beta_ {t}. \\ \end{array}
$$

84 We provide the derivation in Appendix A.1.

# 85 2.2 CARD for Regression

For regression problems, the goal of the reverse diffusion process is to gradually recover the distribution of the noise term, the aleatoric uncertainty inherent in the observations according to Kendall and Gal [13], enabling us to generate samples that match the true conditional  $p(\mathbf{y} \mid \mathbf{x})$ .

Following the reparameterization introduced by Ho et al. [11], we construct  $\epsilon_{\theta}(\boldsymbol{x},\boldsymbol{y}_t,f_\phi (\mathbf{x}),t)$  which is a function approximator parameterized by a deep neural network that predicts the forward diffusion noise  $\epsilon$  sampled for  $\boldsymbol{y}_t$ . The training and inference procedure can be carried out in a standard DDPM manner.

Algorithm 1 Training (Regression)  
1: pre-train  $f_{\phi}(\mathbf{x})$  that predicts  $\mathbb{E}(\mathbf{y}\mid \mathbf{x})$  with MSE   
2: repeat   
3:  $y_0\sim q(y_0)$    
4:  $t\sim \mathrm{Uniform}(\{1\dots T\})$    
5:  $\epsilon \sim \mathcal{N}(0,I)$    
6: compute noise estimation loss

$$
\mathcal {L} _ {\epsilon} = | | \boldsymbol {\epsilon} - \boldsymbol {\epsilon} _ {\boldsymbol {\theta}} (x, \sqrt {\bar {\alpha} _ {t}} \pmb {y} _ {0} + \sqrt {1 - \bar {\alpha} _ {t}} \pmb {\epsilon} + \sqrt {1 - \bar {\alpha} _ {t}} f _ {\phi} (\mathbf {x}), t) | | ^ {2}
$$

7: take numerical optimization step on:

$$
\nabla_ {\theta} \mathcal {L} _ {\epsilon}
$$

8: until convergence

Algorithm 2 Inference (Regression)  
1:  $\pmb{y}_T\sim \mathcal{N}(f_\phi (\mathbf{x}),\pmb {I})$    
2: for  $t = T$  to 1 do   
3:  $\pmb {z}\sim \mathcal{N}(\pmb {0},\pmb {I})$  if  $t > 1$  , else  $\pmb {z} = \pmb{0}$    
4: reparameterized  $\hat{\pmb{y}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\bigl (\pmb {y}_t - (1 - \sqrt{\bar{\alpha}_t})f_\phi (\mathbf{x}) - \pmb {\epsilon}_\theta (\pmb {x},\pmb {y}_t,f_\phi (\mathbf{x}),t)\sqrt{1 - \bar{\alpha}_t}\bigr)$    
5:  $\pmb{y}_{t - 1} = \gamma_0\hat{\pmb{y}}_0 + \gamma_1\pmb {y}_t + \gamma_2f_\phi (\mathbf{x}) + \pmb {z}\tilde{\beta}_t$    
6: end for   
7: return  $\pmb{y}_0$

# 2.3 CARD for Classification

We formulate the classification tasks in a similar fashion as in Section 2.2, where we:

1. replace the continuous response variable with one-hot encoded labels for  $\mathbf{y}_0$ ;  
2. replace the mean estimator with a pre-trained classifier that outputs softmax probabilities of the class labels for  $f_{\phi}(\mathbf{x})$ .

This construction no longer assumes  $y_0$  to be drawn from a categorical distribution, but instead treating each one-hot label as a class prototype. The sampling procedure would output reconstructed  $y_0$  in the range of real numbers for each dimension, instead of a vector in the probability simplex. We convert such output to a probability vector inspired by the Brier score [2], which computes the squared error between the prediction and a vector of 1s with length  $C$ , the number of classes:

$$
\hat {\mathbf {y}} = \operatorname {S o f t m a x} \left(- \left(\mathbf {y} _ {0} - \mathbf {1} _ {C}\right) ^ {2}\right). \tag {10}
$$

Intuitively, this construction would assign the class whose raw output in the sampled  $\mathbf{y}_0$  is closest to the true class, encoded by the value of 1 in the one-hot label, with the highest probability.

The stochasticity of generative model would give us a different class prototype reconstruction, which leads to the variation in the predicted probability for each class label, enabling us to prediction intervals. Such stochastic reconstruction is in a similar fashion as DALL-E 2 [20] that applies a diffusion prior to reconstruct the image embedding by conditioning on the text embedding during reverse diffusion process, which is a key step in generated image diversity.

# 3 Experiments

For the hyperparameters of CARD in both regression and classification tasks, we set the number of timesteps as  $T = 1000$ , a linear noise schedule with  $\beta_{1} = 10^{-4}$  and  $\beta_{T} = 0.02$ , same as Ho et al. [11]. For network architecture, we adopt the settings described in [30, 32]. We provide a more detailed walkthrough of experimental setup in Appendix A.2.

# 3.1 Regression

Putting aside its statistical interpretation, the word regress indicates a direction opposite to progress, suggesting a less developed state. Such semantics in fact translates well into the statistical domain, in the sense that traditional regression analysis methods often only focus on estimating  $\mathbb{E}(\mathbf{y}|\mathbf{x})$ , while leaving out all other details about  $p(\mathbf{y}|\mathbf{x})$ . In recent years, Bayesian neural networks (BNNs) have emerged as a class of models that aims at estimating the uncertainty [10, 7, 14, 26], providing a more complete picture of  $p(\mathbf{y}|\mathbf{x})$ . The metric that they use to quantify uncertainty estimation, negative log-likelihood (NLL), is computed with a Gaussian density, implying their assumption such that the conditional distributions  $p(\mathbf{y}|\mathbf{x} = x)$  for all  $x$  are Gaussian. However, this assumption is very difficult to verify for real world datasets: the covariates can be arbitrarily high-dimensional, making the feature space increasingly sparse with respect to the number of collected observations.

To accommodate the need for uncertainty estimation without imposing such restriction for the parametric form of  $p(\mathbf{y} \mid \mathbf{x})$ , we apply the following two metrics, both of which are designed to empirically evaluate the level of similarity between the learned and the true conditional distributions:

1. Prediction Interval Coverage Probability (PICP);  
2. Quantile Interval Coverage Error (QICE).

PICP has been described in Yao et al. [31], whereas QICE is a new metric proposed by us. We describe both of them in the following section.

# 3.1.1 PICP and QICE

The PICP is computed as:

$$
\operatorname {P I C P} := \frac {1}{N} \sum_ {n = 1} ^ {N} \mathbb {1} _ {y _ {n} \geq \hat {y} _ {n} ^ {\text {l o w}}} \cdot \mathbb {1} _ {y _ {n} \leq \hat {y} _ {n} ^ {\text {h i g h}}}, \tag {11}
$$

where  $\hat{y}_n^{\mathrm{low}}$  and  $\hat{y}_n^{\mathrm{high}}$  represents the low and high percentile of our choice for the predicted y outputs given the same x input. This metric measures the proportion of true observations that fall in the percentile range of the generated y samples given each x input. Intuitively, when the learned distribution represents the true distribution well, this measurement should be close to the difference between the selected low and high percentile. In this paper, we choose the  $2.5\%$  and  $97.5\%$  percentile, thus an ideal PICP value for the learned model should be  $95\%$ .

Meanwhile, there is a caveat for this metric: for example, imagine a situation where the  $2.5\%$  and  $97.5\%$  percentile of the learned distribution happens to cover the data between the  $1\%$  and  $96\%$  percentile from the true distribution. Given enough samples, we shall still obtain a PICP value close to  $95\%$ , but clearly there is a mismatch between the learned distribution and the true one.

Based on such reasoning, we propose a new empirical metric QICE, which by design can be viewed as PICP with finer granularity. To compute QICE, we first generate enough y samples given each x, and divide them into  $M$  bins with roughly equal sizes. We would obtain the corresponding quantile values at each boundary. In this paper, we set  $M = 10$ , and obtain the following 10 quantile intervals (QIs) of the generated y samples: below 10% percentile, between 10% and 20% percentile, ..., between 80% and 90% percentile, and above 90% percentile. Optimally, when the learned conditional distribution is identical to the true one, given enough samples from both learned and true distribution we shall observe about 10% of true data falling into each of these 10 QIs.

We define  $QICE$  to be the mean absolute error between the proportion of true data contained by each QI and the optimal proportion, which is  $1 / M$  for all intervals:

$$
\mathrm {Q I C E} := \frac {1}{M} \sum_ {m = 1} ^ {M} \left| r _ {m} - \frac {1}{M} \right|, \text {w h e r e} r _ {m} = \frac {1}{N} \sum_ {n = 1} ^ {N} \mathbb {1} _ {y _ {n} \geq \hat {y} _ {n} ^ {\mathrm {l o w} _ {m}}} \cdot \mathbb {1} _ {y _ {n} \leq \hat {y} _ {n} ^ {\mathrm {h i g h} _ {m}}}. \tag {12}
$$

Intuitively, under optimal scenario with enough samples, we shall obtain a QICE value of 0. Note that each  $r_m$  is indeed the PICP for the corresponding QI with boundaries at  $\hat{y}_n^{\mathrm{low}_m}$  and  $\hat{y}_n^{\mathrm{high}_m}$ . Since the true  $\mathbf{y}$  for each  $\mathbf{x}$  is guaranteed to fall into one of these QIs, we are thus able to overcome the mismatch issue described in the above example for PICP: fewer true instances falling into one QI would result in more instances captured by another QI, thus increasing the absolute error for both QIs.

QICE is similar to NLL in the sense that it also utilizes the summary statistics of the samples from the learned distribution conditional on each new  $\mathbf{x}$  to empirically evaluate how well the model fits the true data. Meanwhile, it does not assume any parametric form on the conditional distribution, making it a much more generalizable metric to measure the level of distributional match between the learned and the underlying true conditional distributions, especially when the true conditional distribution is known to be multi-modal. We will demonstrate this point through the regression toy examples.

# 3.1.2 Toy Examples

To demonstrate the effectiveness of CARD in regression tasks for not only learning the conditional mean  $\mathbb{E}(\mathbf{y}|\mathbf{x})$ , but also recreating the ground truth data generating mechanism, we first apply CARD on 8 toy examples, whose data generating functions are designed to possess different statistical characteristics: some have a uni-modal symmetric distribution for their error term (linear regression, quadratic regression, sinusoidal regression $^1$ ), others have heteroscedasticity (log-log linear regression, log-log cubic regression) or multi-modality (inverse sinusoidal regression $^2$ , 8 Gaussians $^3$ , full circle). We show that CARD can generate samples that are visually indistinguishable from the true response variables of the new covariates, as well as quantitatively match the true distribution in terms of some summary statistics.

The 8 toy examples are summarized by Table 7 in Appendix A.3. For each task, we create the dataset by sampling 10240 data points from the data generating function, and randomly split them into training and test set with a  $80\% / 20\%$  ratio.

We first examine the performance of CARD visually, by making the scatter plots of both true test data and generated data conditional on test  $\mathbf{x}$  for all 8 tasks. For tasks with uni-modal conditional

![](images/c86560050d7ad87321c328278e6afc251db80d15f0063ed21a108e9f0964cea4.jpg)

![](images/14375e3fc37cd89b18cb0a2dff5d20ec835fe4a34ee52b2fcbca5c33b1bea3fb.jpg)

![](images/2be83ac43a91f0d1f37de55441db45d56ea3be96c170d921e5e93bbb3d3e5ab3.jpg)

![](images/d3d0b12c35c84129b773acce82a460be4f42f0113aad54840664bd52818ca3bd.jpg)

![](images/fc23548e9cfdd824420ea172c2c3fba825f612080373435cd7e549977e391a19.jpg)  
Figure 1: Regression toy example scatter plots. (Top) left to right: linear regression, quadratic regression, log-log linear regression, log-log cubic regression; (Bottom) left to right: sinusoidal regression, inverse sinusoidal regression, 8 Gaussians, full circle.

![](images/e0993b028659392e4974301478d1345a4dc72300bae222dd3f475ed31cad3358.jpg)

![](images/312f991586557651b6a256c40e3f6d07e400febbe17dc9263f41b8f685577289.jpg)

![](images/d55e93c610f97a534c4d6e494ef0a1e0017df2b1cc63557db26c3903953e5167.jpg)

distribution, we fill the region between the 2.5-th and 97.5-th percentile of the generated y's, and plot the true  $\mathbb{E}(\mathbf{y}|\mathbf{x})$  along with the sample means. We observe that for all 8 tasks, the generated samples (blue dots) have blended remarkably well with the true test instances (red dots), suggesting the CARD model is capable of reconstructing the underlying data generation mechanism. For all uni-modal tasks, we note that the empirical means of generated samples are close to the true conditional means for all test x, as the green and orange line interlace with each other without a clear separation at any point. Note that we include all y samples for the computation of each conditional mean, instead of discarding any samples that might be deemed as extreme.

To quantitatively evaluate the performance of CARD, we generate  $1000\mathrm{y}$  samples for each  $\mathbf{x}$  in the test set, and compute the corresponding metrics. We conduct such procedures over 10 runs, each applying a different random seed to generate the dataset, and report the mean and standard deviation over all runs for each metric. For all tasks regardless of the form of  $p(\mathbf{y}|\mathbf{x})$ , we compute PICP and QICE. For tasks with uni-modal  $p(\mathbf{y}|\mathbf{x})$  distributions, we summarize the 1000 samples for each test  $\mathbf{x}$  by computing their mean, as an unbiased estimator to  $\mathbb{E}(\mathbf{y}|\mathbf{x})$ , and compute the root mean squared error (RMSE) between the estimated and true conditional mean. For all tasks, we obtain a mean PICP very close to the optimal  $95\%$ , and most of the tasks have a mean QICE value far less than 0.01 except log-log cubic regression, which also has a mean RMSE noticeably larger by an order of magnitude among cases with uni-modal conditional distributions. Note that the  $\mathbf{y}$  samples here have a much wider range: as  $\mathbf{x}$  increases from 0 to 10,  $\mathbf{y}$  increases from 0 to over 1200, resulting in a much more difficult task. Therefore, the metrics reported here can be viewed with relativity, and combined with the qualitative conclusions from 1. The metrics of all tasks are recorded to Table 8 in Appendix A.3.

# 3.1.3 UCI Regression Tasks

We continue to investigate our model through experiments on real world datasets. We adopt the same set of 10 UCI regression benchmark datasets [5] as well as the experimental protocol proposed in [10] and followed by [7] and [14]. The dataset information in terms of their size and number of features are summarized in 1.

Table 1: Dataset size ( $N$  observations,  $P$  features) of UCI regression tasks.  

<table><tr><td>Dataset</td><td>Boston</td><td>Concrete</td><td>Energy</td><td>Kin8nm</td><td>Naval</td><td>Power</td><td>Protein</td><td>Wine</td><td>Yacht</td><td>Year</td></tr><tr><td>(N,P)</td><td>(506,13)</td><td>(1030,8)</td><td>(768,8)</td><td>(8192,8)</td><td>(11934,16)</td><td>(9568,4)</td><td>(45730,9)</td><td>(1599,11)</td><td>(308,6)</td><td>(515345,90)</td></tr></table>

We apply multiple train-test splits with  $90\% / 10\%$  ratio in the same way (20 folds for all datasets except 5 for Protein and 1 for Year), and report the metrics by their mean and standard deviation across all splits. We compare our method to all aforementioned BNN frameworks: PBP, MC Dropout, and Deep Ensembles, as well as another deep generative model that estimates a conditional distribution sampler, GCDS[33]. Similar to BNNs, we evaluate the accuracy and predictive uncertainty estimation of CARD by reporting RMSE and NLL. Furthermore, we also report QICE for all methods to evaluate distributional matching. Since this new metric was not applied in previous methods, we re-ran the experiments for all BNNs and obtained comparable or slightly better results reported in their literature. Further details about the experiment setup for these models can be found in the Appendix. The experiment results with corresponding metrics are shown in Table 2, 3, 4, with the number of times each model achieves the best corresponding metric reported in the last row.

Table 2: RMSE of UCI regression tasks. For both  ${}^{1}\mathrm{{Kin}}8\mathrm{{nm}}$  and  ${}^{2}$  Naval dataset,we multiply the response variable by 100 to match the scale of others.  

<table><tr><td rowspan="2">Dataset</td><td colspan="5">RMSE ↓</td></tr><tr><td>PBP</td><td>MC Dropout</td><td>Deep Ensembles</td><td>GCDS</td><td>CARD (ours)</td></tr><tr><td>Boston</td><td>2.89 ± 0.74</td><td>3.06 ± 0.96</td><td>3.17 ± 1.05</td><td>2.75 ± 0.58</td><td>2.62 ± 0.63</td></tr><tr><td>Concrete</td><td>5.55 ± 0.46</td><td>5.09 ± 0.60</td><td>4.91 ± 0.47</td><td>5.39 ± 0.55</td><td>4.92 ± 0.48</td></tr><tr><td>Energy</td><td>1.58 ± 0.21</td><td>1.70 ± 0.22</td><td>2.02 ± 0.32</td><td>0.64 ± 0.09</td><td>0.69 ± 0.20</td></tr><tr><td>Kin8nm1</td><td>9.42 ± 0.29</td><td>7.10 ± 0.26</td><td>8.65 ± 0.47</td><td>8.88 ± 0.42</td><td>6.40 ± 0.19</td></tr><tr><td>Naval2</td><td>0.41 ± 0.08</td><td>0.08 ± 0.03</td><td>0.09 ± 0.01</td><td>0.14 ± 0.05</td><td>0.02 ± 0.00</td></tr><tr><td>Power</td><td>4.10 ± 0.15</td><td>4.04 ± 0.14</td><td>4.02 ± 0.15</td><td>4.11 ± 0.16</td><td>3.91 ± 0.16</td></tr><tr><td>Protein</td><td>4.65 ± 0.02</td><td>4.16 ± 0.12</td><td>4.45 ± 0.02</td><td>4.50 ± 0.02</td><td>3.71 ± 0.01</td></tr><tr><td>Wine</td><td>0.64 ± 0.04</td><td>0.62 ± 0.04</td><td>0.63 ± 0.04</td><td>0.66 ± 0.04</td><td>0.62 ± 0.04</td></tr><tr><td>Yacht</td><td>0.88 ± 0.22</td><td>0.84 ± 0.27</td><td>1.19 ± 0.49</td><td>0.79 ± 0.26</td><td>0.56 ± 0.20</td></tr><tr><td>Year</td><td>8.86± NA</td><td>8.77± NA</td><td>8.79± NA</td><td>9.20± NA</td><td>8.69± NA</td></tr><tr><td># best</td><td>0</td><td>1</td><td>1</td><td>1</td><td>8</td></tr></table>

Table 3: NLL of UCI regression tasks.  

<table><tr><td>Dataset</td><td>PBP</td><td>MC Dropout</td><td>NLL ↓ Deep Ensembles</td><td>GCDS</td><td>CARD (ours)</td></tr><tr><td>Boston</td><td>2.53 ± 0.27</td><td>2.46 ± 0.12</td><td>2.35 ± 0.16</td><td>18.66 ± 8.92</td><td>2.32 ± 0.17</td></tr><tr><td>Concrete</td><td>3.19 ± 0.05</td><td>3.21 ± 0.18</td><td>2.93 ± 0.12</td><td>13.64 ± 6.88</td><td>2.99 ± 0.12</td></tr><tr><td>Energy</td><td>2.05 ± 0.05</td><td>1.50 ± 0.11</td><td>1.40 ± 0.27</td><td>1.46 ± 0.72</td><td>1.12 ± 0.15</td></tr><tr><td>Kin8nm</td><td>-0.83 ± 0.02</td><td>-1.14 ± 0.05</td><td>-1.06 ± 0.02</td><td>-0.38 ± 0.36</td><td>-1.31 ± 0.02</td></tr><tr><td>Naval</td><td>-3.97 ± 0.10</td><td>-4.45 ± 0.38</td><td>-5.94 ± 0.10</td><td>-5.06 ± 0.48</td><td>-6.49 ± 0.01</td></tr><tr><td>Power</td><td>2.92 ± 0.02</td><td>2.90 ± 0.03</td><td>2.89 ± 0.02</td><td>2.83 ± 0.06</td><td>2.82 ± 0.02</td></tr><tr><td>Protein</td><td>3.05 ± 0.00</td><td>2.80 ± 0.08</td><td>2.89 ± 0.02</td><td>2.81 ± 0.09</td><td>2.47 ± 0.01</td></tr><tr><td>Wine</td><td>1.03 ± 0.03</td><td>0.93 ± 0.06</td><td>0.96 ± 0.06</td><td>6.52 ± 21.86</td><td>0.87 ± 0.08</td></tr><tr><td>Yacht</td><td>1.58 ± 0.08</td><td>1.73 ± 0.22</td><td>1.11 ± 0.18</td><td>0.61 ± 0.34</td><td>0.89 ± 0.07</td></tr><tr><td>Year</td><td>3.69± NA</td><td>3.42± NA</td><td>3.44± NA</td><td>3.43± NA</td><td>3.34± NA</td></tr><tr><td># best</td><td>0</td><td>0</td><td>1</td><td>1</td><td>8</td></tr></table>

Table 4: QICE (in %) of UCI regression tasks.  

<table><tr><td>Dataset</td><td>PBP</td><td>MC Dropout</td><td>QICE ↓ Deep Ensembles</td><td>GCDS</td><td>CARD (ours)</td></tr><tr><td>Boston</td><td>3.50 ± 0.88</td><td>3.82 ± 0.82</td><td>3.37 ± 0.00</td><td>11.73 ± 1.05</td><td>3.50 ± 0.89</td></tr><tr><td>Concrete</td><td>2.52 ± 0.60</td><td>4.17 ± 1.06</td><td>2.68 ± 0.64</td><td>10.49 ± 1.01</td><td>2.38 ± 0.61</td></tr><tr><td>Energy</td><td>6.54 ± 0.90</td><td>5.22 ± 1.02</td><td>3.62 ± 0.58</td><td>7.41 ± 2.19</td><td>4.82 ± 1.06</td></tr><tr><td>Kin8nm</td><td>1.31 ± 0.25</td><td>1.50 ± 0.32</td><td>1.17 ± 0.22</td><td>7.73 ± 0.80</td><td>0.90 ± 0.28</td></tr><tr><td>Naval</td><td>4.06 ± 1.25</td><td>12.50 ± 1.95</td><td>6.64 ± 0.60</td><td>5.76 ± 2.25</td><td>2.57 ± 0.71</td></tr><tr><td>Power</td><td>0.82 ± 0.19</td><td>1.32 ± 0.37</td><td>1.09 ± 0.26</td><td>1.77 ± 0.33</td><td>0.85 ± 0.18</td></tr><tr><td>Protein</td><td>1.69 ± 0.09</td><td>2.82 ± 0.41</td><td>2.17 ± 0.16</td><td>2.33 ± 0.18</td><td>0.75 ± 0.04</td></tr><tr><td>Wine</td><td>2.22 ± 0.64</td><td>2.79 ± 0.56</td><td>2.37 ± 0.63</td><td>3.13 ± 0.79</td><td>3.68 ± 0.82</td></tr><tr><td>Yacht</td><td>6.93 ± 1.74</td><td>10.33 ± 1.34</td><td>7.22 ± 1.41</td><td>5.01 ± 1.02</td><td>8.17 ± 0.85</td></tr><tr><td>Year</td><td>2.96±NA</td><td>2.43±NA</td><td>2.56±NA</td><td>1.61±NA</td><td>0.56±NA</td></tr><tr><td># best</td><td>2</td><td>0</td><td>2</td><td>1</td><td>5</td></tr></table>

We observe that CARD outperforms existing methods, often by a considerable margin (especially on larger datasets), in all metrics for most of the datasets, and is competitive with the best method for

the remaining ones: we obtain state-of-the-art results in 8 out of 10 datasets in terms of both RMSE and NLL, and 5 out of 10 for QICE. Note that although we do not explicitly optimize our model by MSE or by NLL, we still obtain better results than models trained with these objectives.

# 3.2 Classification

Similar to Lakshminarayanan et al. [14], our motivation for classification is not to achieve the state-of-the-art performance in terms of mean accuracy on the benchmark datasets, which is strongly related to network architecture design. Our goal is two-fold:

1. We aim to solve classification problems via a generative model, emphasizing its capability to improve the performance of a base classifier with deterministic outputs;  
2. We intent to introduce the idea of uncertainty to classification predictions at instance level.

As another type of supervised learning problems, classification is different than regression mainly for the response variable being discrete class labels instead of continuous values. Conventionally, the output from a classification algorithm is a point estimate, a value being casted as between 0 and 1, with the intention to align with the human cognitive intuition of probability for each class label [3]. In other words, that point estimate should express a level of confidence, or the likelihood of correctness, in that prediction by the model. Metrics like Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) have been proposed to evaluate the alignment between classifier predictions and the true correctness likelihood, and calibration methods like Platt scaling and isotonic regression have been developed to improve such alignment [9], but all are based on point estimate predictions. In the next section, we introduce an alternative in access model confidence, at the level of individual instances, specifically tailored for models with stochastic output.

# 3.2.1 Estimating Instance Level Prediction Confidence via Generative Models

We utilize the stochasticity of generative models to construct our framework of evaluating prediction uncertainty at instance level. As introduced in Section 2.3, we view each one-hot label as a class prototype, and we use the generative model to reconstruct this prototype in a stochastic fashion. The intuition is that if the learned model knows well about which class a particular instance belongs to, it would precisely reconstruct the prototype vector; otherwise if the model is unsure about a new instance based on the learned patterns, it would not be robust against the input noise sample: under the context of denoising diffusion models, the class prototype reconstruction would appear rather different from each other, given a different sample from the prior distribution at timestep  $T$ .

We propose the following framework to access model confidence: during test time, we sample  $N$  class prototype reconstructions for each instance, converting them into probability scale with Eq. (10), and make the following two computations:

1. We calculate the prediction interval width (PIW) between  $2.5\%$  and  $97.5\%$  percentile of the  $N$  predicted probabilities for the true class associated with each instance;  
2. We apply paired two-sample  $t$ -test as an uncertainty estimation method proposed in [6]: we obtain the most and second most predicted class for each instance, and test whether the difference in their mean predicted probability is statistically significant.

# 3.2.2 Classification on FashionMNIST with Prediction Uncertainty Evaluation

We demonstrate our experiment results on the FashionMNIST dataset. Following the recipe in 2.3, we first pre-train a deterministic classifier with ResNet-18 architecture for 10 epochs with a batch size of 256, and achieve a test accuracy of  $91.19\%$ . After training CARD, we obtain our instance prediction through majority vote, i.e. the most predicted class label among its  $N$  samples, and achieve an improved test accuracy of  $91.71\%$ , showing its ability to improve the prediction accuracy from the base classifier. We contextualize such performance by reporting the mean accuracy of other BNNs with LeNet CNN architecture [26] in Table 5. CARD uses a much simpler network as its image feature encoder, yet still outperforms all other models.

In Table 6, we report the mean PIW among both correct and incorrect predictions, as well as the mean accuracy among both groups rejected and not-rejected by the paired two-sample  $t$ -test ( $\alpha = 0.01$ ). We report these metrics for all test instances and for each class label along with their group accuracy.

Table 5: Comparison of mean accuracy for FashionMNIST classification task with other BNNs and ResNet-18 (our pre-trained classifier  $f_{\phi}$ ).  

<table><tr><td>Model</td><td>CMV-MF-VI</td><td>CM-MF-VI</td><td>CV-MF-VI</td><td>CM-MF-VI OPT</td><td>MF-VI</td><td>MAP</td><td>MC Dropout</td><td>MF-VI EB</td><td>ResNet-18</td><td>CARD (ours)</td></tr><tr><td>Mean Accuracy</td><td>91.10%</td><td>90.95%</td><td>88.53%</td><td>90.67%</td><td>87.04%</td><td>88.06%</td><td>87.99%</td><td>87.04%</td><td>91.19%</td><td>91.71%</td></tr></table>

Table 6: PIW (in %) and  $t$  -test results for FashionMNIST classification task.  

<table><tr><td rowspan="2">Class</td><td rowspan="2">Accuracy</td><td colspan="2">PIW</td><td colspan="2">Accuracy by t-test Status</td></tr><tr><td>Correct</td><td>Incorrect</td><td>Rejected</td><td>Not-Rejected (Count)</td></tr><tr><td>All</td><td>91.71%</td><td>2.39</td><td>22.05</td><td>92.05%</td><td>40.00% (65)</td></tr><tr><td>1</td><td>86.70%</td><td>4.35</td><td>21.75</td><td>87.13%</td><td>53.85% (13)</td></tr><tr><td>2</td><td>98.30%</td><td>0.77</td><td>13.79</td><td>98.30%</td><td>100% (1)</td></tr><tr><td>3</td><td>86.90%</td><td>3.65</td><td>20.41</td><td>87.35%</td><td>50.00% (12)</td></tr><tr><td>4</td><td>92.00%</td><td>2.80</td><td>27.58</td><td>92.25%</td><td>50.00% (6)</td></tr><tr><td>5</td><td>86.40%</td><td>3.99</td><td>33.36</td><td>87.21%</td><td>33.33% (15)</td></tr><tr><td>6</td><td>99.00%</td><td>0.83</td><td>18.36</td><td>99.00%</td><td>NA (0)</td></tr><tr><td>7</td><td>76.70%</td><td>5.57</td><td>16.00</td><td>77.56%</td><td>20.00% (15)</td></tr><tr><td>8</td><td>96.20%</td><td>1.32</td><td>29.15</td><td>96.30%</td><td>0.00% (1)</td></tr><tr><td>9</td><td>98.50%</td><td>0.57</td><td>10.79</td><td>98.50%</td><td>NA (0)</td></tr><tr><td>10</td><td>96.40%</td><td>1.29</td><td>15.30</td><td>96.49%</td><td>50.00% (2)</td></tr></table>

We observe from Table 6 that under the scope of the entire test set, mean PIW of the true class label among the correct predictions is narrower than that of the incorrect predictions by an order of magnitude, indicating that CARD is much more confident when making correct predictions; on the other hand, CARD is also aware of what it does not know. More revealing observations can be made when comparing mean PIWs across different classes: a class with more accurate predictions tends to have a sharper contrast between correct and incorrect predictions; additionally, the metric values of both types of predictions tend to be larger in a less accurate class. Meanwhile, we observe that the accuracy of test instances rejected by the  $t$ -test is much higher than that of the not-rejected ones, both among the entire set and within each class (except Class 2 with only 1 not-rejected instance).

We point out that these metrics can thus reflect how sure CARD is about its predictions, and can be used as an important indicator of whether the model prediction can be trusted or not. Therefore, it has the potential to be further applied in the human-machine collaboration domain [15, 19, 29, 8], such that one can apply such uncertainty measurement to decide if we can directly accept the model prediction, or we need to allocate the instance to humans for further evaluation.

# 4 Conclusion

In this paper, we propose Classification And Regression Diffusion (CARD) models, a class of generative models that approaches supervised learning problems from a generative modeling perspective. Without training with objectives directly related to the evaluation metrics, we achieve state-of-the-art results on benchmark regression tasks. Furthermore, CARD exhibits a strong ability to represent conditional distribution with multiple density modes. We also propose a new metric Quantile Interval Coverage Error (QICE), which can be viewed as a generalized version of negative log-likelihood in evaluating how well the model fits the data. Lastly, we introduce a framework to evaluate prediction uncertainty at instance level for classification tasks.

# References

[1] Christopher Bishop. Mixture density networks. In Aston University Neural Computing Research Group Report, 1994.  
[2] G. W. Brier. Verification of forecasts expressed in terms of probability. In Monthly weather review, 1950.  
[3] Leda Cosmides and John Tooby. Are humans good intuitive statisticians after all? rethinking some conclusions from the literature on judgment under uncertainty. In cognition, volume 58(1), pages 1-73, 1996.  
[4] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=AAWuCvzaVt.  
[5] D. Dua and C. Graff. UCI machine learning repository. 2017. URL http://archive.ics.uci.edu/ml.  
[6] Xinjie Fan, Shujian Zhang, Korawat Tanwisuth, Xiaoning Qian, and Mingyuan Zhou. Contextual dropout: An efficient sample-dependent dropout module. In International Conference on Learning Representations, 2021.  
[7] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In Proceedings of the 33rd International Conference on Machine Learning, 2016.  
[8] Ruijiang Gao, Maytal Saar-Tsechansky, Maria De-Arteaga, Ligong Han, Min Kyung Lee, and Matthew Lease. Human-ai collaboration with bandit feedback. In International Joint Conferences on Artificial Intelligence, 2021.  
[9] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, 2017.  
[10] José Miguel Hernández-Lobato and Ryan P. Adams. Probabilistic backpropagation for scalable learning of bayesian neural networks. In Proceedings of the 32nd International Conference on International Conference on Machine Learning, 2015.  
[11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 2020.  
[12] Aapo Hyvarinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005.  
[13] Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? In 31st Conference on Neural Information Processing System, 2017.  
[14] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Proceedings of the 31st Conference on Neural Information Processing Systems, 2017.  
[15] David Madras, Toniann Pitassi, and Richard Zemel. Predict responsibly: Improving fairness and accuracy by learning to defer. In Proceedings of the 32nd Conference on Neural Information Processing Systems, 2018.  
[16] Radford M Neal. MCMC using Hamiltonian dynamics. Handbook of Markov Chain Monte Carlo, page 113, 2011.  
[17] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

[18] Kushagra Pandey, Avideep Mukherjee, Piyush Rai, and Abhishek Kumar. Diffusevae: Efficient, controllable and high-fidelity generation from low-dimensional latents. arXiv preprint arXiv:2201.00308, 2022.  
[19] Maithra Raghu, Katy Blumer, Greg Corrado, Jon Kleinberg, Ziad Obermeyer, and Sendhil Mullainathan. The algorithmic automation problem: Prediction, triage, and human effort. 2019.  
[20] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.  
[21] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125, 2022.  
[22] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics, 2015.  
[23] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, pages 11918-11930, 2019.  
[24] Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. arXiv preprint arXiv:2006.09011, 2020.  
[25] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=PxTIG12RRHS.  
[26] Marcin B. Tomczak, Siddharth Swaroop, Andrew Y. K. Foong, and Richard E. Turner. Collapsed variational bounds for bayesian neural networks. In Proceedings of the 35th Conference on Neural Information Processing Systems, 2021.  
[27] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661-1674, 2011.  
[28] Max Welling and Yee W Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pages 681–688. Citeseer, 2011.  
[29] Bryan Wilder, Eric Horvitz, and Ece Kamar. Learning to complement humans. In International Joint Conferences on Artificial Intelligence, 2020.  
[30] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion gans. arXiv preprint arXiv:2112.07804, 2021.  
[31] Jiayu Yao, Weiwei Pan, Soumya Ghosh, and Finale Doshi-Velez. Quality of uncertainty quantification for bayesian neural network inference. In ICML 2019 Workshop on Uncertainty and Robustness in Deep Learning, 2019.  
[32] Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Truncated diffusion probabilistic models. arXiv preprint arXiv:2202.09671, 2022.  
[33] Xingyu Zhou, Yuling Jiao, Jin Liu, and Jian Huang. A deep generative approach to conditional sampling. Journal of the American Statistical Association, pages 1-28, 2021.
