# NEURAL CAUSAL REGULARIZATION UNDER THE INDEPENDENCE OF MECHANISMS ASSUMPTION

Mohammad Taha Bahadori<sup>1</sup>, Krzysztof Chalupka<sup>2</sup>, Edward Choi<sup>1</sup>, Robert Chen<sup>1</sup>, Walter F. Stewart<sup>3</sup>, & Jimeng Sun<sup>1</sup>

<sup>1</sup> Georgia Institute of Technology, <sup>2</sup> California Institute of Technology, <sup>3</sup> Sutter Health

# ABSTRACT

Neural networks provide a powerful framework for learning the association between input and response variables and making accurate predictions and offer promise in using the rapidly growing volume of health care data to surface causal relationships that cannot necessarily be tested in randomized clinical trials. In pursuit of models whose predictive power comes maximally from causal variables, we propose a novel causal regularizer based on the independence of mechanisms assumption. We use the causal regularizer to steer deep neural network architectures towards causally-interpretable solutions. We perform a large-scale analysis of electronic health records. Our causally-regularized algorithm outperforms its  $L_{1}$ -regularized counterpart both in predictive performance as well as causal relevance. Finally, we show that the proposed causal regularizer can be used together with representation learning algorithms to yield up to  $20\%$  improvement in the causality score of the generated hypotheses.

# 1 INTRODUCTION

In domains such as healthcare, genomics or social science there is high demand for data analysis that reveals causal relationships between independent and target variables. For example, doctors not only want models that accurately predict the status of patients, but also want to identify the factors that can change the status. The distinction between prediction and causation has at times been subject to controversy in statistics and machine learning (Breiman et al., 2001; Shmueli, 2010; Donoho, 2015). On one hand, machine learning has been focusing almost exclusively on pure prediction tasks, enjoying great commercial success. On the other hand, in many scientific domains pure prediction without consideration of the underlying causal mechanisms is considered unscientific (Shmueli, 2010). In this work, we propose a neural causal regularizer that balances causal interpretability and high predictive power.

Causal Inference: Our notion of causality follows the counterfactual framework of Pearl (2000). Thus, we will say that one random variable  $X$  causes another variable  $Y$  (which relationship we denote as  $X \rightarrow Y$ ) if intervening or experimenting on  $X$  changes the distribution of  $Y$ . Consider the problem of identifying the causal relationship between drinking red wine and heart disease (Spirtes, 2010). Regular consumption of red wine correlates with healthy heart. That might mean that drinking red wine decreases heart attack rates. But it might be, for example, that people of high socio-economic status tend to drink more wine, while at the same time tend to suffer fewer heart problems due to better living conditions. To distinguish between these two possibilities, one could implement a controlled trial in which the subjects are told to drink (or not drink) red wine, independently of any other factors—including their socio-economic status.

Such controlled trials are often undesirable or even impossible. In healthcare, it can be due to moral and regulatory reasons; in climate science for example, due to technological limitations (we don't know how to change climate). In such settings, we would like to still establish causality without resorting to experiment. Even in applications where controlled trials are possible, the large number of causal hypotheses can make it impossible to experimentally test all of them. Furthermore, in domains such as healthcare, many causal factors need to occur simultaneously to have an effect on the target variable, a scenario that we call multivariate causation. Given the exponential number of combinations of the independent variables and different transformations, it is even more difficult to explore all of these multivariate causation scenarios.

![](images/6c87df82a2147b53c43e126bf45bd440d173a0415d767d91acad36b450b3aa10.jpg)  
(a) Ground Truth Causality Score

![](images/a640e9e83d6a6a321ce7776ce1f6f706321857b6c6a386b1a0fc3b4a7c243f39.jpg)  
(b) Classification AUC  
Figure 1: Superior causality and predictive performance of the causal regularizer in our heart failure study. (a) Average causality score computed using ground truth causality labels. We compute the score for top  $k$  codes reported by three algorithms. (b and c) The causal regularizer is more stable in predictive performance measured by AUC and  $F_{1}$  metrics. See Section 3 for more details.

![](images/cc0475e3e5c4e3048189b55ada42fe2aa8d565afc2190c4a926ca8701c56faca.jpg)  
(c) Classification  $F_{1}$  Score

Analyzing causation without resorting to experiment is challenging due to unobserved confounders (such as the possible influence of socio-economic status on heart health and wine drinking). Many methods have been proposed for discovering causal relationships among multiple variables from observational data only (Chickering, 2002; Kalisch & Buhlmann, 2007; Colombo et al., 2012), demonstrating various degrees of success. These methods are based on the idea that any given set of causal relationships among multiple variables will leave in the joint distribution well-defined markers in the form of independence relationships among subsets of the variables. These methods, however, are often very sensitive to small changes in the joint distribution.

Causal Regularization: Our main idea is to design a causal regularizer to control the complexity of the statistical models and at the same time favor causal explanations. Compared to the two step procedure of (i) causal variable selection and (ii) multivariate regression, the proposed approach performs joint causal variable selection and prediction, thus avoiding the statistically sensitive hard-thresholding of the causality scores in the causal variable selection step. It allows dependencies that cannot be explained via causation to be included in the model. Our contributions are four-fold:

1. We propose a customized causality detector neural network that can accurately discriminate causal and non-causal variables in our healthcare datasets. To this end, we propose new synthetic dataset generation to train the causal structure detectors in (Chalupka et al., 2016; Lopez-Paz et al., 2016) with additional prior knowledge from the healthcare domain.  
2. We use the causality detector to construct a causal regularizer that can guide predictive models towards learning causal relationships between the independent and target variables.  
3. Given the fact that the causal regularizer seamlessly integrates with non-linear predictive models such as neural networks, we propose a new non-linear predictive model regularized by our causal regularizer, which allows neural causally predictive modeling.  
4. Finally, we demonstrate that the proposed causal regularizer can be combined with neural representation learning techniques to efficiently generate multivariate causal hypotheses.

The proposed framework scales linearly with the number of variables, as opposed to many previous causal methods. Combined with a predictive model, it efficiently screens a high-dimensional hypothesis space and proposes plausible hypotheses.

We applied the proposed algorithm to two electronic health records (EHR) datasets: Sutter Health's heart failure study data and the publicly available MIMIC III (Johnson et al., 2016) dataset. Altogether, we analyzed the influence of 17,081 independent variables on heart failure. To validate our claims, we use expert judgment as the causal ground-truth to compare our causal-predictive solutions with purely predictive solutions that do not take causality into account. As shown in Figure 1, a causally-regularized algorithm outperforms its  $L_{1}$ -regularized equivalent both in predictive performance as well as causal performance.

# 2 METHODOLOGY

In order to "inject causality" into predictive models, we use the "independence of cause and mechanisms" (ICM) assumption, which allows us to construct a neural network causality detector, as described in Section 2.1. We present a causal regularizer for linear models in Section 2.2. Using this regularizer, in Section 2.3 we propose non-linear deep neural networks to learn non-linear causal relationships between the independent and target variables. Finally, we show that the causal regularizer can efficiently explore the space of multivariate causal hypotheses and extract meaningful candidates for causality analysis.

![](images/6dc753feacc0cb55782deff8dc83c24aaa365e874c1d47bf682a0cfa65e7aebf.jpg)  
(a) Independent

![](images/66bc6cc00262140098c86ef12c76bfb03e46e86b127d9c66f96fb8156d45bf83.jpg)

![](images/e0d1dd848c90eb281d584362567af538a551a57e06ad4669aaf46cf9a874cf14.jpg)  
(b) Direct

![](images/0326dd335a0364b2f95ba67af2eac87d687f72cfb4607dc4b12029c31929a929.jpg)  
(c) Reverse

![](images/fa078c95603370fd5c15c7372abab0333de005e7823cb9772e90d25f68b86374.jpg)  
(d) Indirect

![](images/4aa629da3353e6f3323b4808e8180242a19d876f4e3b2e2fb8fa5d2baa6722b4.jpg)

![](images/edf44ebbd918c0869d2d20f07f236bad958fc1d23b7d2c4d6383d889ee520876.jpg)  
(f) Confounded Correlation

![](images/fde5cedc3f5e1b043a1113052322d91c96c5c037cf326406a5ec90b69ad2e655.jpg)  
(g) Confounded and Direct

![](images/6442eaa049adc64783347837c0b74715b6152ac19d7aeebf56b5adeafd78f550.jpg)  
(h) Confounded and Reverse  
Figure 2: Some possible causal structures between two observed and one or more hidden variables. Under the algorithmic independence assumption, we can sample from the joint distribution of  $X$  and  $Y$  in each case and train a classifier that distinguishes between these cases based on the (automatically learned) features of the joint distribution.

![](images/0c9737c91018e4fb2a5053e7b92f56ed1056ad27b0109b1e07d72dd7304124e7.jpg)  
(i) Confounded and Indirect

![](images/231e2a90fc3513454dc19678458670aa43239b5a7a550caf08e9468556b67f83.jpg)  
(e) Indirect Reverse  
(j) Confounded & Indirect Reverse

# 2.1 CAUSALITY DETECTION BASED ON INDEPENDENCE OF MECHANISMS

As we discussed in the introduction, the task of analysis of causal effect of multiple independent variables on a target variable is difficult. Our approach in this paper is to reduce the problem to analysis of the causal effect of a single independent variable  $X$  on the target variable  $Y$ , which is known as pairwise causal analysis. In the next subsections, we will describe how to use a pairwise causality detector to perform multivariate causality analysis.

In particular, we are interested in finding causal models where  $X$  causes  $Y$ , or  $Y$  causes  $X$ , or the two are confounded based on joint distribution of  $P(X,Y)$ . However, even the pairwise causality analysis is infeasible for arbitrary joint distributions. Thus, we need to resort to additional assumptions on the nature of the causal relationships. Recently several algorithms have been proposed that distinguish between the cause and effect based on the natural assumption that steps in the process that generates the data are independent from each other, see (Lemeire & Dirkx, 2006; Janzing et al., 2012; Daniusis et al., 2010; Lopez-Paz, 2016; Chalupka et al., 2016) and the references therein. In this work, we follow (Lopez-Paz et al., 2016; Chalupka et al., 2016) to describe this causality detection approach. In the next subsections, we describe our novel causal regularizer designed based on this causality detection approach and its application in non-linear causality analysis and multivariate causal hypothesis generation.

Conceptual description of the independence between the cause and the mechanism. Algorithms based on the ICM, such as (Chalupka et al., 2016; Lopez-Paz et al., 2016) do not put assumptions on the functional form of the causal relationships between the variables of interest. Instead, they are based on the following assumption on how causal mechanisms come to be. ICM states that the two processes of generation of the cause and mapping from cause to effect are in some sense independent. In our case, we assume that when  $X \to Y$  ( $X$  causes  $Y$ ), the probabilities  $\mathbb{P}(Y \mid X)$  and  $\mathbb{P}(X)$  are generated by independent higher-level distributions. This conforms to the scientific idea of Uniformitarianism (Gould, 1965) which, putting roughly, states that the laws of nature apply to all objects similarly. ICM can be described in both deterministic (Janzing & Scholkopf, 2010) and probabilistic sense (Daniusis et al., 2010); this work mainly uses the probabilistic interpretation.

ICM can be used to generate all of the possible graphical models including two observed variables  $X$  and  $Y$  and an unobserved variable  $H$  shown in Figure 2, by requiring that the probabilities in the factorization are independent from each other. The hidden variables can represent the other observed variables such as  $Z$ , critical in design of the regularizer in the next subsection.

Following the ICM, we assume that each cause-effect link in the world is probabilistic and can be described by a joint distribution  $P( \text{cause}, \text{effect} )$ . In addition, the link itself is sampled from a probabilistic hyperprior. The key assumption is on the structure of this hyperprior, namely that it decomposes into two parts  $\Pi_c$  and  $\Pi_m$  that have the following properties:

1. For each cause, effect pair, Nature samples the cause's distribution  $P_{\text{cause}}$  from a hyperprior  $\Pi_c[P_{\text{cause}}]$ .

Given the data, perform the following steps:

1. Generate data samples  $S_{i}$  for  $i = 1, \dots, n_{\text{train}}$  from  $p_{X,Y}$  according to the ten cases in Figure 2.  
2. Assign label  $y = 0$  to the cases in Figures 2b, 2d, 2g and 2i and  $y = 1$  to the rest.  
3. Train a classifier  $f: \mathcal{S} \to [0,1]$  to classify them as causation (label=1) or not-causation (label=0). Given the fact that this is a synthetic dataset, we know these labels and we can use supervised learning.  
4. On the test set, construct the test sample sets and use the classifier in step 3 to classify the example.

Algorithm 1: The algorithm for constructing the causality detector. The structure of neural network classifier is given in Appendix B.1.

2. At the same time, Nature samples the causal mechanism (the distribution of the effect conditioned on the cause)  $P_{\text{effect|cause}}$  from a hyperprior  $\Pi_m[P_{\text{effect|cause}}]$ .  
3. The hyperpriors are flat — for discrete cause and effect, they are Dirichlet distributions with  $\alpha$  uniformly equal to 1.

The last assumption is not crucial and can easily be changed if knowledge about hyperpriors in a specific domain is available. In fact, we tailor the hyperpriors to our task below. These three assumptions give us a full generative model of causal links in the world, a model under which the likelihood ratio test can be used to differentiate between the data generated from each of the ten cases shown in Fig. 2. Chalupka et al. (2016) developed an analytical likelihood ratio test that decides between the causal and anticausal cases (Figures 2b and 2c). Taking into account the confounded cases is, however, difficult or impossible to compute analytically. Nevertheless, it is possible to generate samples from the generative model defined by the ICM and train a neural network to learn to choose the max likelihood causal structure given samples from the joint  $P( \text{cause}, \text{effect})$ . This is the key idea of the causality detectors in (Lopez-Paz et al., 2016; Chalupka et al., 2016).

Mathematical description of the causality detection algorithm. Formally, suppose we have  $m$  variables  $X_{i}$ , each with dimensionality  $d_{i}$ . For each variable we observe a sample of size  $n_i$  denoted by  $S_{i} = \{(\mathbf{x}_{i,j},y_{j})\}_{j = 1}^{n_{i}}$ , where  $y_{j}$  are observations of a common target variable  $Y$ . Let  $S$  denote the set of all such samples. For each sample  $S_{i}$ , we are interested in determining the binary label  $\ell_{i}\in \{0,1\}$  which determines whether  $X_{i}$  causes  $Y$  or not. In fact, we are interested in the function approximation problem of learning the mapping  $f:S\mapsto \{0,1\}$ .

Several approaches can learn such a mapping function. When  $X$  and  $Y$  are both discrete and finite, Chalupka et al. (2016) construct the empirical joint distribution  $\widehat{p_i} = \widehat{p}(X_i, Y)$  and train a supervised neural network mapping function  $f(\widehat{p_i}) \to \ell_i$ . Lopez-Paz et al. (2016) learn the mapping  $\frac{1}{n_i} \sum_{j=1}^{n_i} \phi(\mathbf{x}_{i,j}, y_j)$  and a neural network  $f\left(\frac{1}{n_i} \sum_{j=1}^{n_i} \phi(\mathbf{x}_{i,j}, y_j)\right) \to \ell_i$ . They train both the representation leaning function  $\phi(\cdot, \cdot)$  and the classification network in a joint and supervised way.

However, it is rare to have the true causal labels  $\ell$  for training a causal detector. The key idea is to generate a synthetic dataset composed of the cases in Figure 2 based on the ICM assumption. As shown in Algorithm 1, the overall procedure is to generate samples from distributions  $p_{X,Y}$  that are one of the ten possible cases in Figure 2. We need to select the distributions such that they impose minimum number of restriction on the data and the synthetically-generated distributions have statistics as similar as possible to those of our true data of interest. For example, in our dataset, the independent variables  $X$  are counts of the number of disease codes in patients' records (cf. Section 3). Thus, we sample  $X$  from a mixture of appropriate distributions for count data: the Zipf, Poisson, Uniform, and Bernoulli distributions. The hidden variable  $H$  and the response variable  $Y$  are sampled from the Dirichlet and Bernoulli distributions. Details of our sampling procedure are provided in Appendix A.

# 2.2 THE CAUSAL REGULARIZER

As an instructive alternative to our approach, consider the two-step analysis method of first finding the variables  $X_{i}$  that are most likely causes of  $Y$  and then performing a sparse multivariate regression to select the important variables. Ideally, if the ICM holds and if we had access to the true joint distributions and could discriminate between causal and non-causal variables with perfect accuracy, the two-step procedure would be sufficient. But real-world datasets always contain noise and selection bias, which can perturb the causality scores generated by the neural network confounder detector.

The problem arises from the fact that our causality detection algorithm might give soft scores such as  $0.5 + \varepsilon$  or  $0.5 - \varepsilon$  to two variables  $X_{1}$  and  $X_{2}$ , respectively. These soft-scores can be interpreted as the probability that each variable is the cause of  $Y$ . If we use the two-step procedure, we will include  $X_{1}$  in the regression model but not  $X_{2}$ . However,  $X_{2}$  could possibly contribute more to the predictive performance in presence of other variables in the multivariate regression. In other words, any hard cut-off for the purpose of two-step causal variable selection and regression will pose the question of "what should be the best cut-off threshold?" Note that any hard cut-off will be always statistically unstable in presence of noise and selection bias.

Instead, we propose a causally regularized regression approach, where this trade-off is performed naturally via a regularization parameter. We select variables that are both potentially causal with high probability and also significantly predictive.

Causal Regularizer. Now that we have a classifier that outputs  $c_{i} = \mathbb{P}[X_{i} \text{ and } Y \text{ are not-causal}]$ , we can design the following regularizer to encourage learning a causal predictive model:

$$
\widehat {\mathbf {w}} = \underset {\mathbf {w}} {\operatorname {a r g m i n}} \left\{\frac {1}{n} \sum_ {j = 1} ^ {n} \mathcal {L} \left(\mathbf {x} _ {j}, y _ {j} \mid \mathbf {w}\right) + \lambda \sum_ {i = 1} ^ {m} c _ {i} \left| w _ {i} \right| \right\}, \tag {1}
$$

where  $\mathcal{L}(X_1, \ldots, X_n, Y|\mathbf{w})$  is the loss function of logistic regression for  $X_1, \ldots, X_n$  and  $Y$ . The first term in Eq. (1) is a multivariate analysis term, whereas the regularizer might look like a bivariate operation between each independent variable  $X_i$  and the target variable  $Y$  for  $i = 1, \ldots, p$ . However, we should note that in the design of the causal regularizer, we have implicitly included the other variables as hidden variables in the analysis. Thus we are allowed to use the regularizer together with multivariate regression. Note that the proposed causal regularizer is also a decomposable regularizer which makes analysis of its theoretical properties easier (Negahban et al., 2012).

The two-step analysis can be cast as a special case of causally predictive modeling where we use hard scores instead of soft scores. Consider the following setting:

$$
\widehat {\mathbf {w}} = \underset {\mathbf {w}} {\operatorname {a r g m i n}} \left\{\frac {1}{n} \sum_ {j = 1} ^ {n} \mathcal {L} (\mathbf {x} _ {j}, y _ {j} | \mathbf {w}) + \gamma \sum_ {i = 1} ^ {m} c _ {i} ^ {\prime} | w _ {i} | \right\},
$$

Where  $c_{i}^{\prime}$  is defined as follows:

$$
c _ {i} ^ {\prime} = \left\{ \begin{array}{l l} 1 - \varepsilon & \text {i f} c _ {i} > 1 / 2 \\ \varepsilon & \text {i f} c _ {i} \leq 1 / 2 \end{array} \right.
$$

Now, consider the limiting case of  $\varepsilon \to 0$  and  $\gamma \varepsilon \to \lambda$ . This case corresponds to the two-step procedure with  $L_{1}$  regularized logistic regression.

Note that the possibility of having a causal regularizer has been proposed in (Lopez-Paz, 2016, Page 181) and (Lopez-Paz et al., 2016), however a specific causal regularizer has never been developed and evaluated. Furthermore, note that using the score of a "causal-anticausal"-only classifier, as e.g. in (Lopez-Paz et al., 2016), cannot properly regularize a multivariate model such as logistic regression. In our proposal, the rest of the observed independent variables can be considered as hidden variables in our bivariate causality analysis which allows proper regularization. Moreover, a major novelty of our proposed causal regularizer is to do joint causal variable selection (the  $L_{1}$  regularization) and prediction, but the idea in (Lopez-Paz et al., 2016) cannot.

# 2.3 CAUSAL REGULARIZERS IN NEURAL NETWORKS

The key advantages of causal regularizer can be seen when it is used for regularizing neural networks. We demonstrate two use cases of causal regularizer as shown in Figure 3.

![](images/c37d8aac60ce14e218eefebc7a374f60fc8f25739348d26e0f981acdfe0b0fd7.jpg)  
(a) Non-linear causality analysis

![](images/0e11b2c3364da9d69bb968f1352b22fa6ef8eb8e45eed247be69b14a9f4bfc71.jpg)  
(b) Multivariate causal hypothesis generation  
Figure 3: Two use cases of the proposed causal regularizer: (a) In the proposed architecture, applying the causal regularizer allows identification of causal relationships in the non-linear settings, where the causality coefficient can change from subject to subject. (b) The causal regularizer allows us to explore the high-dimensional multi-variate combinations of the variables and identify plausible hypotheses. Here,  $g$  generates the causal regularization coefficients for the hypotheses  $h$ . The regularizer encourages the coordinates of  $h$  to be more causal.

Non-linear Modeling. The objective is to design a non-linear neural network in a way that we can still identify causality. We propose the following non-linear generalized linear model:

$$
\sigma^ {- 1} (\mathbb {E} [ Y ]) = \mathbf {w} ^ {\top} \mathbf {x} + \boldsymbol {\beta} ^ {\top} (\boldsymbol {\alpha} (E \mathbf {x}) \odot (E \mathbf {x})) + b, \tag {2}
$$

where the embedding matrix  $E \in \mathbb{R}^{q \times m}$  maps the input  $\mathbf{x} \in \mathbb{R}^m$  to a lower dimensional representation space and the symbol  $\odot$  denotes the element-wise product. The logistic sigmoid function  $\sigma^{-1}$  maps the real values to the [0, 1] interval. The term  $\mathbf{w}^\top \mathbf{x}$  acts as the skip connection and initialized by the result of logistic regression. The embedding allows dealing with very large set of discrete concepts and can be initialized via techniques such as skip-gram (Mikolov et al., 2013) or GloVe (Pennington et al., 2014). The vector  $\alpha(E\mathbf{x})$  can be computed using a multi-layer preception.

The model in Eq. (2) is a particular non-linear extension of logistic regression. We can reorder the equations to write the right hand side of Eq. (2) as  $\boldsymbol{\omega}(\mathbf{x})^{\top}\mathbf{x} + b$ , where the new regression coefficient  $\boldsymbol{\omega}$  can change with every input. Each coordinate of the new regression coefficient can be calculated as  $\omega_{i}(\mathbf{x}) = w_{i} + (\beta \odot \alpha(E\mathbf{x}))^{\top}E_{i}$ , where  $E_{i}$  denotes the  $i$ th column of the embedding matrix  $E$ . The variability of  $\omega_{i}(\mathbf{x})$  for each input  $\mathbf{x}$  enables us to perform individual causality analysis. For training, we can penalize the  $\boldsymbol{\omega}$  coefficients and minimize the following loss function

$$
\frac {1}{n} \sum_ {j = 1} ^ {n} \left\{\widetilde {\mathcal {L}} \left(\mathbf {x} _ {j}, y _ {j}\right) + \lambda \sum_ {i = 1} ^ {m} c _ {i} \left| \omega_ {i} \left(\mathbf {x} _ {j}\right) \right| \right\}, \tag {3}
$$

where  $\widetilde{\mathcal{L}}$  denotes the negative log-likelihood of the model described by Eq. (2). The change of the prediction vector with each sample  $\mathbf{x}$  can be related to the probabilistic definition of causation (Pearl, 2000) in the sense that the strength of causality may change from a subject to another one. The fact that in Eq. (2) the impact of each independent variable on the target is measured by  $\omega_{i}(\mathbf{x})$  allows us to penalize it with our regularizer and push the model to learn more causal relationships.

Multivariate Causal Hypothesis Generation. A key application of our proposed causal regularizer in conjunction with deep representation learning is to efficiently extract multivariate causal hypotheses from the data. Figure 3b shows an example of causal hypothesis generation where the hypotheses are generated via a Multilayer Perceptron (MLP). We assume that there is a representation learning network with  $K$ -dimensional output  $h(\mathbf{x}) \in \mathcal{I}^K$ , where  $\mathcal{I}$  denotes the range of the output, for example  $\mathcal{I} = (0,1)$  for sigmoid and  $\mathcal{I} = [0,\infty)$  for ReLU activation functions. Our goal is to force each dimension of  $h$  to be causal, thus  $h$  can be used as multivariate causal hypotheses. In particular, we aim at minimizing the following objective function:

$$
\frac {1}{n} \sum_ {j = 1} ^ {n} \left\{\mathcal {L} \left(\mathbf {w} ^ {\top} \boldsymbol {h} _ {j} + b\right) + \lambda \sum_ {i = 1} ^ {K} \left| g _ {i} \left(\boldsymbol {h} _ {j, i}\right) w _ {i} \right| \right\} \tag {4}
$$

Our approach is to train a causality detector based on (Lopez-Paz et al., 2016) and design the regularizer  $g(h(\mathbf{x}))$  based on its score. Then, as shown in Figure 3b, we can combine it with the neural network to regularize the coefficients of the last layer of the multilayer Perceptron which predicts the labels from  $h$ . The weights of the lower layers in  $h(\mathbf{x})$  are regularized using  $L_{1}$  regularizer to make the generated causal variables simple. To train the network, we select batches with fixed-size of 200 examples. This number is selected to be large enough such that error rate of the causality detector in (Lopez-Paz et al., 2016) becomes lower than  $2\%$ . We select the non-linearity for  $h$  to be the logistic sigmoid function, thus we use Beta distribution for generating synthetic data for training of the causality classifier.

# 3 EXPERIMENTS

We evaluate the proposed causal regularizer in Section 2.2 both in terms of their predictive and causal performance. Next, we compare the quality of the codes identified as causes of heart failure identified by different approaches. Finally, we evaluate performance of multivariate causal hypothesis generation by qualitatively analyzing the extracted hypotheses. We defer evaluation of the causality detection algorithms to Appendix A, as they are not the main contributions of this work.

# 3.1 DATA

The Sutter Health heart failure dataset consists of Electronic Health Records of middle-aged adults collected by Sutter Health for a heart failure study. From the encounter records, medication orders, procedure orders and problem lists, we extracted visit records consisting of diagnosis, medication and procedure codes. We denote the set of such codes by  $\mathcal{C}$ . Given a visit sequence  $\mathbf{v}_1,\ldots ,\mathbf{v}_T$ , we try to predict if the patient will be diagnosed with heart failure (HF) and identify the key causes of increase heart failure risk. To this end, 3,884 cases are selected and approximately 10 controls are selected for each case (28,903 controls). The case/control selection criteria are fully described in the supplementary section. Cases have index dates to denote the date they are diagnosed with HF. Controls have the same index dates as their corresponding cases. We extract diagnosis codes, medication codes and procedure codes from the 18-month window before the index date. There are in total 17,081 number of unique medical codes in this dataset.

The MIMIC III dataset (Johnson et al., 2016) is a publicly available dataset consisting of medical records of intensive care unit (ICU) patients over 11 years. We use a public query<sup>1</sup> to extract the binary mortality labels for the patients. Our goal is to use the codes in the patients' last visit to the ICU and predict their mortality outcome. Our dataset includes 46,520 patients out of whom 5810 have deceased (mortality=1). A total of 14,587 different medical codes are used in this dataset.

Feature construction. Given the sequence of visits  $\mathbf{v}_1^{(i)},\ldots ,\mathbf{v}_T^{(i)}$  for patients  $i = 1,\dots ,n$ , we create a feature vector  $\mathbf{x}_i\in \mathbb{N}_0^{|C|}$  by counting the number of codes observed in the records of the ith patient. Given the large variations in the number of codes, we logarithmically bin the count data into 16 bins. The final data is in the form of  $(\mathbf{x}_i,y_i)$  where  $y_{i}$  is ith patient's label; heart failure and mortality outcome in the Sutter and MIMIC III datasets, respectively.

Training details. Given the fact that we generate synthetic datasets for training the causality detector neural networks, we can generate as many new batches of data for training and parameter tuning purposes as required. We report the test results on a dataset of size 10,000 data points. For training and parameter tuning of the neural network model in Section 2.2, we perform the common  $75\% / 10\% / 15\%$  training/validation/test splits. Details of training the latter neural network are given in Appendix B.2.

# 3.2 EVALUATING THE PREDICTIVE PREFORMANCE OF CAUSAL REGULARIZER

In order to characterize the performance of the proposed causal regularizer, we perform penalized logistic regression with the proposed regularizer and the commonly used  $L_{1}$  regularizer. Table 1 shows the test accuracy of heart failure and mortality prediction in Sutter and MIMIC datasets, respectively. We have run each algorithm ten times and report the mean and standard deviation of the performance measures. As we can see, the proposed causal regularizer does not significantly hurt the predictive performance, whereas the two-step procedure significantly reduces the accuracy.

Table 1: Prediction accuracy results on two datasets. (mean±standard deviation)  

<table><tr><td rowspan="2">Algorithms</td><td colspan="2">Sutter</td><td colspan="2">MIMIC III</td></tr><tr><td>AUC</td><td>F1</td><td>AUC</td><td>F1</td></tr><tr><td>Causal Logistic</td><td>0.8289 ± 0.0064</td><td>0.4147 ± 0.0192</td><td>0.9772 ± 0.0022</td><td>0.7871 ± 0.0097</td></tr><tr><td>L1 Logistic</td><td>0.8289 ± 0.0054</td><td>0.4109 ± 0.0150</td><td>0.9774 ± 0.0022</td><td>0.7869 ± 0.0095</td></tr><tr><td>Two Step</td><td>0.7276 ± 0.0086</td><td>0.2686 ± 0.0134</td><td>0.9515 ± 0.0033</td><td>0.6745 ± 0.0106</td></tr></table>

![](images/afe05e54c7f325c186c6a896d19e4b58f51957d41afb1a4202ab2e66151c4916.jpg)  
(a) AUC on Sutter

![](images/cb6e0bf75ec47a12cf7c848d6290f1490097ea633e11518b92340bfd291bffaa.jpg)  
(b)  $F_{1}$  on Sutter  
Figure 4: Comparison of variable selection in logistic regression via the causal and  $L_{1}$  regularizers on two datasets and two accuracy measures. Note the stability of variable selection by the causal regularizer as the penalization coefficient varies.

![](images/b23f4a63dfb4840c875c88376301dabb962f92258fbce05d521d06080f08a7bb.jpg)  
(c) AUC on MIMIC

![](images/9610f7d140f42aba5d3ebc7f66a1fb6a60ae75680f3c5e75ccc94a32169db13a.jpg)  
(d)  $F_{1}$  on MIMIC

An interesting phenomenon, shown in Figure 4, is the relative robustness of the performance with respect to the value of the penalization parameter compared to the  $L_{1}$  regularization case. This robustness comes at no surprise, because the causal regularizer assigns very small penalization coefficients to the causal variables and as we discussed in Section 2.2, only with very high values of penalization we can force all coefficients to become zero, see Figure 7 in Appendix A.1. Moreover, this robustness can be attributed to the fact that the causal regularizer might match the true generative process of the dataset better than the flat  $L_{1}$  regularizer and puts the model under less pressure as we increase the penalization parameter. We demonstrate the predictive gain by the non-linear causal model in Figure 5a. Furthermore, the impact of changing the regularization parameter on the number of selected variables is visualized in Figure 8 in Appendix A.1.

# 3.3 EVALUATING THE CAUSAL PERFORMANCE OF CAUSAL REGULARIZER

In order to evaluate the performance of the algorithms in their ability to identify causal factors, we generate the top 100 influential factors by different methods. We ask a clinical expert to label each factor as "causal", "not-causal", and "potentially causal" and assign scores 1, 0, and 0.5 to them, respectively. Table 2 shows the average causality score by each algorithm based on the labels provided by the medical expert. As expected,  $L_{1}$  regularized logistic regression performs poorly, as it is susceptible to the impact of confounded variables. Performance of the causally regularized logistic regression is superior to the two step procedure, which suggests that picking factors that are both causal and highly predictive leads to better causality score. This result together with the predictive results in Table 1 confirms that the causal regularizer can be efficiently used for finding few causal variables that are highly predictive of the target quantity.

The advantages of the regularized approach can also be seen by the results in Table 4. We have marked many disease codes that can potentially increase the risk of heart failure. However, the predicted causality score for them is lower than 0.5 and the two-step procedure would have eliminated from the predictors set (as shown in Table 10 in Appendix C). The causal regularizer approach is able to establish a balance between the prediction and causation and produce more plausible results.

# 3.4 EVALUATING THE MULTIVARIATE CAUSAL HYPOTHESES

We evaluate the performance of the proposed causal hypothesis generation against the case when we do not use any causal regularization. We generate two lists of top 50 hypotheses using two algorithms and ask our medical expert to label each hypothesis as causal, non-causal or possibly causal with corresponding scores of 1, 0, and 0.5. The results in Figure 5b shows that the causal regularizer can increase the causality score of the hypotheses by up to  $20\%$ . We also provide a qualitative analysis of

Table 2: Average causality score on the heart failure task computed using ground truth labels. For a higher resolution of number of top codes in the list see Figure 1a.  

<table><tr><td># codes in the list</td><td>Causal Logistic</td><td>L1 Logistic</td><td>Two Step</td></tr><tr><td>Top 20</td><td>0.725</td><td>0.400</td><td>0.425</td></tr><tr><td>Top 50</td><td>0.520</td><td>0.330</td><td>0.450</td></tr><tr><td>Top 100</td><td>0.485</td><td>0.315</td><td>0.375</td></tr></table>

![](images/41ed9a4b41549bab74ae4ee3eceadd25c52c4f1d4c41efc8b5e21f227b9c6fc0.jpg)  
(a) Predictive gain by the non-linear model

![](images/8cad3d900e9a928ae3ed784b9b49086266267bb13ee745ec8243a17e10f676aa.jpg)  
(b) Accuracy of causal hypothesis generation  
Figure 5: (a) The predictive gain by the nonlinear causal model in Eq. (2) on the MIMIC III dataset. The gain is more visible when fewer features are used in the analysis because the input becomes more expressive by themselves. We select the variables in the descending order of variance. (b) Average causality score computed using ground truth causality labels for generated hypotheses. We compute the score for top  $k$  hypotheses reported by two algorithms.

Table 3: Examples of multivariate causal hypotheses generated via causal regularizer.  

<table><tr><td>Name</td><td>Conditions</td><td>Description</td></tr><tr><td>Aortic Dissection from Trauma</td><td>Dissection of aortaBurn in multiple sites of trunkAbdominal pain, lower left quadrant</td><td>This collection of diagnoses is is especially causal for heart failure, as heart failure can manifest as a complication of dissection of aorta. Dissection of aorta can present with abdominal pain, and may happen in traumatic injuries that involve burn of unspecified degree of other and multiple sites of trunk, occurring together.</td></tr><tr><td>Kidney Neoplasm and Severe Infections</td><td>Malignant neoplasm of kidneyHistory of infectious and parasitic diseasesTuberculosis of lung</td><td>Neoplasms in the kidney may lead to paraneoplastic systemic effects that may lead to heart failure. Furthermore, having concurrent severe infections such as tuberculosis can also increase the risk of heart failure.</td></tr><tr><td>Metabolic Syndrome with Concurrent Infections and Pregnancy</td><td>Metabolic syndromeTuberculosis of lungObstetrical pulmonary embolism</td><td>Metabolic syndrome co-occurring with severe infections such as tuberculosis can lead to heart failure. Obstetrical pulmonary embolisms can lead to acute heart failure.</td></tr></table>

the causal hypotheses generated by our algorithm. To this end, we pick several hypotheses and show that clinically they are meaningful. Three examples of multivariate causal hypotheses generated via causal regularizer are shown in Table 3.

# 4 CONCLUSION

We addressed the problem of exploring the high-dimensional causal hypothesis space in applications such as healthcare. We designed a causal regularizer that steers predictive algorithms towards explanations "as causal as possible". The proposed causal regularizer, based on our causality detector, does not increase the computational complexity of the  $L_{1}$  regularizer and can be seamlessly integrated with a neural network to perform non-linear causality analysis. We also demonstrated the application of the proposed causal regularizer in generating multivariate causal hypotheses. Finally, we demonstrated the usefulness of the causal regularizer in detecting the causes of heart failure using an electronic health records dataset.

# ACKNOWLEDGMENT

The authors would like to thank Frederick Eberhardt for helpful discussions. Mohammad Taha Bahadori acknowledges the previous discussions with David C. Kale and Micheal E. Hankin on the concept of causal regularizer.

Table 4: Top 30 codes with causal regularization. The coefficient is  $w_{i}$  from Eq. (1). The causality score in this table is the output of causality classifier.  

<table><tr><td>Code</td><td>Description</td><td>Coefficient</td><td>Causality</td></tr><tr><td>794.31</td><td>Nonspecific abnormal electrocardiogram [ECG] [EKG]</td><td>0.3422</td><td>0.9351</td></tr><tr><td>425.8</td><td>Cardiomyopathy in other diseases classified elsewhere</td><td>0.3272</td><td>0.2322</td></tr><tr><td>786.05</td><td>Shortness of breath</td><td>0.3124</td><td>0.5536</td></tr><tr><td>424.90</td><td>Endocarditis, valve unspecified, unspecified cause</td><td>0.3086</td><td>0.3908</td></tr><tr><td>425.4</td><td>Other primary cardiomyopathies</td><td>0.2880</td><td>0.1351</td></tr><tr><td>427.9</td><td>Cardiac dysrhythmia, unspecified</td><td>0.2531</td><td>0.9864</td></tr><tr><td>785.9</td><td>Other symptoms involving cardiovascular system</td><td>0.2377</td><td>0.8024</td></tr><tr><td>585.6</td><td>End stage renal disease</td><td>0.2225</td><td>0.3948</td></tr><tr><td>511.9</td><td>Unspecified pleural effusion</td><td>0.2218</td><td>0.0839</td></tr><tr><td>425.9</td><td>Secondary cardiomyopathy, unspecified</td><td>0.2203</td><td>0.8024</td></tr><tr><td>782.3</td><td>Edema</td><td>0.2065</td><td>0.0027</td></tr><tr><td>278.01</td><td>Morbid obesity</td><td>0.1955</td><td>0.0345</td></tr><tr><td>424.0</td><td>Mitral valve disorders</td><td>0.1948</td><td>0.0003</td></tr><tr><td>427.31</td><td>Atrial fibrillation</td><td>0.1762</td><td>1.0000</td></tr><tr><td>410.90</td><td>Acute myocardial infarction of unspecified site, episode of care unspecified</td><td>0.1756</td><td>0.2510</td></tr><tr><td>426.3</td><td>Other left bundle branch block</td><td>0.1690</td><td>0.4890</td></tr><tr><td>424.1</td><td>Aortic valve disorders</td><td>0.1649</td><td>0.0012</td></tr><tr><td>879.8</td><td>Open wound(s) (multiple) of unspecified site(s), without mention of complication</td><td>0.1645</td><td>0.6399</td></tr><tr><td>429.3</td><td>Cardiomegaly</td><td>0.1619</td><td>0.5022</td></tr><tr><td>780.60</td><td>Fever, unspecified</td><td>0.1602</td><td>0.7747</td></tr><tr><td>482.9</td><td>Bacterial pneumonia, unspecified</td><td>0.1514</td><td>0.7482</td></tr><tr><td>786.09</td><td>Other respiratory abnormalities</td><td>0.1454</td><td>0.7305</td></tr><tr><td>496</td><td>Chronic airway obstruction, not elsewhere classified</td><td>0.1403</td><td>0.9990</td></tr><tr><td>V42.0</td><td>Kidney replaced by transplant</td><td>0.1398</td><td>0.4351</td></tr><tr><td>250.03</td><td>Diabetes mellitus without mention of complication, type I [juvenile type], uncontrolled</td><td>0.1388</td><td>0.4727</td></tr><tr><td>276.51</td><td>Dehydration</td><td>0.1347</td><td>0.6738</td></tr><tr><td>403.10</td><td>Hypertensive chronic kidney disease, benign, with chronic kidney disease stages I ~ IV</td><td>0.1316</td><td>0.7488</td></tr><tr><td>250.50</td><td>Diabetes with ophthalmic manifestations, type II or unspecified type, not uncontrolled</td><td>0.1283</td><td>0.2271</td></tr><tr><td>427.89</td><td>Other specified cardiac dysrhythmias</td><td>0.1282</td><td>0.9416</td></tr><tr><td>250.51</td><td>Diabetes with ophthalmic manifestations, type I [juvenile type], not stated as uncontrolled</td><td>0.1234</td><td>0.5473</td></tr></table>

# REFERENCES

Leo Breiman et al. Statistical modeling: The two cultures. Statistical Science, 16(3):199-231, 2001.  
Krzysztof Chalupka, Frederick Eberhardt, and Pietro Perona. Estimating the causal direction and confounding of two discrete variables. arXiv Preprint, 2016.  
David M. Chickering. Optimal structure identification with greedy search. JMLR, 3:507-554, 2002.  
Diego Colombo, Marloes H Maathuis, Markus Kalisch, and Thomas S Richardson. Learning high-dimensional directed acyclic graphs with latent and selection variables. Ann. Stat., 2012.  
Povilas Daniusis, Dominik Janzing, Joris Mooij, Jakob Zscheischler, Bastian Steudel, Kun Zhang, and Bernhard Scholkopf. Inferring deterministic causal relations. In UAI, 2010.  
David Donoho. 50 years of data science. In *Princeton NJ, Tukey Centennial Workshop*, 2015.  
Stephen Jay Gould. Is uniformitarianism necessary? Am. J. Sci., 263(3):223-228, 1965.  
Dominik Janzing and Bernhard Scholkopf. Causal inference using the algorithmic markov condition. IEEE Transactions on Information Theory, 56(10):5168-5194, 2010.  
Dominik Janzing, Jonas Peters, Eleni Sgouritsa, Kun Zhang, Joris M Mooij, and Bernhard Scholkopf. On causal and anticausal learning. In ICML, pp. 1255-1262, 2012.  
Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible critical care database. Scientific data, 3, 2016.  
Markus Kalisch and Peter Buhlmann. Estimating high-dimensional directed acyclic graphs with the pc-algorithm. JMLR, 8(Mar):613-636, 2007.  
Jan Lemeire and Erik Dirkx. Causal models as minimal descriptions of multivariate systems, 2006.  
David Lopez-Paz. From dependence to causation. PhD thesis, University of Cambridge, 2016.  
David Lopez-Paz, Robert Nishihara, Soumith Chintala, Bernhard Scholkopf, and Léon Bottou. Discovering causal signals in images. arXiv preprint arXiv:1605.08179, 2016.  
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In NIPS, 2013.  
Sahand N. Negahban, Pradeep Ravikumar, Martin J. Wainwright, and Bin Yu. A unified framework for high-dimensional analysis of  $m$ -estimators with decomposable regularizers. Statist. Sci., 2012.  
Judea Pearl. Causality. Cambridge university press, 2000.  
Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. EMNLP, 2014.  
Galit Shmueli. To explain or to predict? Statistical science, pp. 289-310, 2010.  
Peter Spirtes. Introduction to causal inference. JMLR, 11(May):1643-1662, 2010.  
Martin J Wainwright and Michael I Jordan. Graphical models, exponential families, and variational inference. Foundations and Trends® in Machine Learning, 1(1-2):1-305, 2008.
