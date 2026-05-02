# SHOULD ENSEMBLE MEMBERS BE CALIBRATED?

Anonymous authors

Paper under double-blind review

# ABSTRACT

Underlying the use of statistical approaches for a wide range of applications is the assumption that the probabilities obtained from a statistical model are representative of the "true" probability that event, or outcome, will occur. Unfortunately, for modern deep neural networks this is not the case, they are often observed to be poorly calibrated. Additionally, these deep learning approaches make use of large numbers of model parameters, motivating the use of Bayesian, or ensemble approximation, approaches to handle issues with parameter estimation. This paper explores the application of calibration schemes to deep ensembles from both a theoretical perspective and empirically on a standard image classification task, CIFAR-100. The underlying theoretical requirements for calibration, and associated calibration criteria, are first described. It is shown that well calibrated ensemble members will not necessarily yield a well calibrated ensemble prediction, and if the ensemble prediction is well calibrated its performance cannot exceed that of the average performance of the calibrated ensemble members. On CIFAR-100 the impact of calibration for ensemble prediction, and associated calibration is evaluated. Additionally the situation where multiple different topologies are combined together is discussed.

# 1 INTRODUCTION

Deep learning approaches achieve state-of-the-art performance in a wide range of applications, including image classification. However, these networks tend to be overconfident in their predictions, they often exhibit poor calibration. A system is well calibrated, if when the system makes a prediction with probability of 0.6 then  $60\%$  of the time that prediction is correct. Calibration is very important in deploying system, especially in risk-sensitive tasks, such as medicine (Jiang et al., 2012), auto-driving (Bojarski et al., 2016), and economics (Gneiting et al., 2007). It was shown by Niculescu-Mizil & Caruana (2005) that shallow neural networks are well calibrated. However, Guo et al. (2017) found that more complex neural network model with deep structures do not exhibit the same behaviour. This work motivated recent research into calibration for general deep learning systems. Previous research has mainly examined calibration based on samples from the true data distribution  $\{\pmb{x}^{(i)},y^{(i)}\}_{i = 1}^{N}\sim \mathbb{P}(\pmb {x},\omega),y^{(i)}\in \{\omega_1,\dots,\omega_K\}$  (Zadrozny & Elkan, 2002; Vaicenavicius et al., 2019). This analysis relies on the limiting behaviour as  $N\to +\infty$  to define a well calibrated system

$$
\mathrm {P} (y = \hat {y} | \mathrm {P} (\hat {y} | \boldsymbol {x}; \boldsymbol {\theta}) = p) = p \Longleftrightarrow \lim  _ {N \rightarrow + \infty} \sum_ {i \in \mathcal {S} _ {j} ^ {p}} \frac {\delta (y ^ {(i)} , \hat {y} ^ {(i)})}{| \mathcal {S} _ {j} ^ {p} |} = p \tag {1}
$$

where  $S_{j}^{p} = \{i|\mathbb{P}(\hat{y}^{(i)} = j|\boldsymbol{x}^{(i)};\boldsymbol{\theta}) = p,i = 1,\dots,N\}$  and  $\hat{y}^{(i)}$  the model prediction for  $\boldsymbol{x}^{(i)}$ . However, Eq. (1) doesn't explicitly reflect the relation between  $\mathbb{P}(y = \hat{y} |\mathbb{P}(\hat{y} |\boldsymbol{x};\boldsymbol{\theta}) = p)$  and the underlying data distribution  $\mathfrak{p}(\boldsymbol{x},\boldsymbol{y})$ . In this work we examine this explicit relationship and use it to define a range of calibration evaluation criteria, including the standard sample-based criteria.

One issue with deep-learning approaches is the large number of model parameters associated with the networks. Deep ensembles (Lakshminarayanan et al., 2017) is a simple, effective, approach for handling this problem. It has been found to improve performance, as well as allowing measures of uncertainty. In recent literature there has been "contradictory" empirical observations about the relationship between the calibration of the members of the ensemble and the calibration of the final ensemble prediction (Rahaman & Thiery, 2020; Wen et al., 2020). In this paper, we examine the underlying theory and empirical results relating to calibration with ensemble methods. We found, both

theoretically and empirically, that assembling multiple calibrated models decreases the confidence of final prediction, resulting in an ill-calibrated ensemble prediction. To address this, strategies to calibrate the final ensemble prediction, rather than individual members, are required. Additionally we empirically examine the situation where the ensemble is comprised of models with different topologies, and resulting complexity/performance, requiring non-uniform ensemble averaging.

In this study, we focus on post-hoc calibration of ensemble, based on temperature annealing. Guo et al. (2017) conducted a thorough comparison of various existing post-hoc calibration methods and found that temperature scaling was a simple, fast, and often highly effective approach to calibration. However, standard temperature scaling acts globally for all regions of the input samples, i.e. all logits are scaled towards one single direction, either increasing or decreasing the distribution entropy. To address this constraint, that may hurt some legitimately confident predictions, we investigate the effect of region-specific temperatures. Empirical results demonstrate the effectiveness of this approach, with minimal increase in the number of calibration parameters.

# 2 RELATED WORK

Calibration is inherently related to uncertainty modeling. Two of the most important scopes of calibration are calibration evaluation and calibration system construction. One method to assessing calibration is the reliability diagram (Vaicenavicius et al., 2019; Brocker, 2012). Though informative, It is still desirable to have an overall metric. Widmann et al. (2019) investigate different distances in the probability simplex for estimating calibration error. Nixon et al. (2019) point out the problem of fixed spaced binning scheme, bins with few predictions may have low-bias but high-variance measurement. Calibration error measure adaptive to dense populated regions have also been proposed (Nixon et al., 2019). Vaicenavicius et al. (2019) treated the calibration evaluation as hypotheses tests. All these approaches examine calibration criteria from a sample-based perspective, rather than as a function of the underlying data distribution which is used in the theoretical analysis in this work.

There are two main approaches to calibrating systems. The first is to recalibrate the uncalibrated systems with post-hoc calibration mapping, e.g. Platt scaling (Platt et al., 1999), isotonic regression (Zadrozny & Elkan, 2002), Dirichlet calibration (Kull et al., 2017; 2019). The second is to directly build calibrated systems, via: (i) improving model structures, e.g. deep convolutional Gaussian processes (Tran et al., 2019); (ii) data augmentation, e.g. adversarial samples (Stutz et al., 2020) or Mixup (Zhang et al., 2018); (iii) minimize calibration error during training (Kumar et al., 2018). Calibration based on histogram binning (Zadrozny & Elkan, 2001), Bayesian binning (Naeini et al., 2015) and scaling binning (Kumar et al., 2019) are related to our proposed dynamic temperature scaling, in the sense that the samples are divided into regions and separate calibration mapping are applied. However, our method can preserve the property that all predictions belonging to one sample sum to 1. The region-based classifier by Kuleshov & Liang (2015) is also related to our approach.

Ensemble diversity has been proposed for improved calibration (Raftery et al., 2005; Stickland & Murray, 2020). In Zhong & Kwok (2013), ensembles of SVM, logistic regressor, boosted decision trees are investigated, where the combination weights of calibrated probabilities is based on AUC of ROC. In this work we investigate the combination of different deep neural network structures. The weights assigned to the probabilities is either optimised using AUC as in (Ashukha et al., 2020) or a likelihood-based metric.

# 3 CALIBRATION FRAMEWORK

Let  $\mathcal{X} \subseteq \mathbb{R}^d$  be the  $d$ -dimensional input space and  $\mathcal{Y} = \{\omega_1, \dots, \omega_K\}$  be the discrete output space consisting of  $K$  classes. The true underlying joint distribution for the data is  $\mathsf{p}(\boldsymbol{x}, \omega) = \mathrm{P}(\omega | \boldsymbol{x}) \mathsf{p}(\boldsymbol{x}), \boldsymbol{x} \in \mathcal{X}, \omega \in \mathcal{Y}$ . Given some training data  $\mathcal{D} \sim \mathsf{p}(\boldsymbol{x}, \omega)$ , a model  $\theta$  is trained to predict the distribution  $\mathrm{P}(\omega | \boldsymbol{x}; \theta)$  given observation features. For a calibrated system the average predicted posterior probability should equate to the average posterior of the underlying distribution for a specific probability region. Two extreme cases will always yield perfect calibration. First when the predictions that are the same, and equal to the class prior for all inputs,  $\mathrm{P}(\omega_j | \boldsymbol{x}; \theta) = \mathrm{P}(\omega_j)$ . Second the minimum Bayes' risk classifier is obtained,  $\mathrm{P}(\omega_j | \boldsymbol{x}; \theta) = \frac{\mathsf{p}(\boldsymbol{x}, \omega_j)}{\sum_{k=1}^{K} \mathsf{p}(\boldsymbol{x}, \omega_k)}$ . Note that perfect calibration doesn't imply high accuracy, as shown by the system predicting the prior distribution.

# 3.1 DISTRIBUTION CALIBRATION

A system is calibrated if the predictive probability values can accurately indicate the portion of correct predictions. Perfect calibration for a system that yields  $\mathbb{P}(\boldsymbol {\omega}|\boldsymbol {x};\boldsymbol {\theta})$  when the training and test data are obtained from the joint distribution  $\mathfrak{p}(\pmb {x},\pmb {\omega})$  can be defined as:

$$
\int_ {\boldsymbol {x} \in \mathcal {R} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon)} \mathrm {P} \left(\omega_ {j} | \boldsymbol {x}; \boldsymbol {\theta}\right) \mathrm {p} (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} = \int_ {\boldsymbol {x} \in \mathcal {R} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon)} \mathrm {P} \left(\omega_ {j} | \boldsymbol {x}\right) \mathrm {p} (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} \quad \forall p, \omega_ {j}, \epsilon \rightarrow 0 \tag {2}
$$

$$
\mathcal {R} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon) = \left\{\boldsymbol {x} \mid | \mathrm {P} \left(\omega_ {j} \mid \boldsymbol {x}; \boldsymbol {\theta}\right) - p | \leq \epsilon , \boldsymbol {x} \in \mathcal {X} \right\} \tag {3}
$$

$\mathcal{R}_j^p (\theta ,\epsilon)$  denotes the region of input space where the system predictive probability for class  $\omega_{j}$  is sufficiently close, within error of  $\epsilon$  to the probability  $p$ . A perfectly calibrated system will satisfy this expression for all regions, the expected predictive probability (left side of Eq. (2)) is identical to the expected correctness, i.e., expected true probability (right side of Eq. (2)).

$\mathcal{R}_j^p (\theta ,\epsilon)$  defines the region in which calibration is defined. For top-label calibration, only the most probable class is considered and the region defined in Eq. (3) is modified to reflect this:

$$
\tilde {\mathcal {R}} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon) = \mathcal {R} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon) \cap \left\{\boldsymbol {x} \mid \omega_ {j} = \underset {\omega} {\arg \max } \mathrm {P} (\omega | \boldsymbol {x}; \boldsymbol {\theta}), \boldsymbol {x} \in \mathcal {X} \right\} \tag {4}
$$

Eq. (4) is a strict subset of Eq. (3). As the two calibration regions are different between calibration and top-label calibration, perfect calibration doesn't imply top-label calibration, and vise versa. A simple illustrative example of this property is given in A.3. Binary classification,  $K = 2$ , is an exception to this general rule, as the regions for top-label calibration are equivalent to those for perfect calibration, i.e.  $\tilde{\mathcal{R}}_j^p(\theta, \epsilon) = \mathcal{R}_j^p(\theta, \epsilon)$ . Hence, perfect calibration is equivalent to top-label calibration for binary classification (Nguyen & O'Connor, 2015).

Eq. (2) defines the requirements for a perfectly calibrated system. It is useful to define metrics that allow how close a system is to perfect calibration to be assessed. Let the region calibration error be:

$$
\mathcal {C} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon) = \int_ {\boldsymbol {x} \in \mathcal {R} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon)} \left(\mathrm {P} \left(\omega_ {j} | \boldsymbol {x}; \boldsymbol {\theta}\right) - \mathrm {P} \left(\omega_ {j} | \boldsymbol {x}\right)\right) \mathrm {p} (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} \tag {5}
$$

This then allows two forms of expected calibration losses to be defined

$$
\mathbf {A C E} (\boldsymbol {\theta}) = \frac {1}{K} \int_ {0} ^ {1} \left| \sum_ {j = 1} ^ {K} \mathcal {C} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon) \right| \mathrm {d} p; \quad \mathbf {A C C E} (\boldsymbol {\theta}) = \frac {1}{K} \sum_ {j = 1} ^ {K} \int_ {0} ^ {1} \left| \mathcal {C} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon) \right| \mathrm {d} p \tag {6}
$$

All Calibration Error (ACE) only considers the expected calibration error for a particular probability, irrespective of the class associated with the data $^{1}$  (Hendrycks et al., 2019). Hence, All Class Calibration Error (ACCE) that requires that all classes minimise the calibration error for all probabilities is advocated by Kull et al. (2019); Kumar et al. (2019). Nixon et al. (2019) propose the Thresholded Adaptive Calibration Error (TACE) to consider only the prediction larger than a threshold, and it can be described as a special case of ACCE by replacing the integral range. Naeini et al. (2015) also propose to only consider the region with maximum error.

Though measures such as ACE and ACCE require consistency of the expected posteriors with the true distribution, for tasks with multiple classes, particularly large numbers of classes, the same weight is given to the ability of the model to assign low probabilities to highly unlikely classes, and high probabilities to the "correct" class. For systems with large numbers of classes this can yield artificially low scores. To address this problem it is more common to replace the regions in Eq. (5) with the top-label regions in Eq. (4), to give a top-label calibration error  $\tilde{C}_j^p (\theta ,\epsilon)$ . This then yields the expected top-label equivalents of ACCE and ACE, Expected Class Calibration Error (ECCE) and Expected Calibration Error (ECE). Here for example ECE by Guo et al. (2017) is expressed as

$$
\begin{array}{l} \operatorname {E C E} (\boldsymbol {\theta}) = \int_ {0} ^ {1} \left| \sum_ {j = 1} ^ {K} \int_ {\boldsymbol {x} \in \tilde {\mathcal {R}} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon)} \left(\mathrm {P} \left(\omega_ {j} | \boldsymbol {x}; \boldsymbol {\theta}\right) - \mathrm {P} \left(\omega_ {j} | \boldsymbol {x}\right)\right) \mathrm {p} (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} \right| \mathrm {d} p (7) \\ = \int_ {0} ^ {1} \mathcal {O} (\boldsymbol {\theta}, p) | \operatorname {C o n f} (\boldsymbol {\theta}, p) - \operatorname {A c c} (\boldsymbol {\theta}, p) | \mathrm {d} p (8) \\ \end{array}
$$

where  $\mathcal{O}(\pmb{\theta}, p) = \sum_{j=1}^{K} \int_{\pmb{x} \in \tilde{\mathcal{R}}_j^p(\pmb{\theta}, \epsilon)} \mathrm{p}(\pmb{x}) \, \mathrm{d}\pmb{x}$  is the fraction observations that are assigned to that particular probability and  $\mathrm{Conf}(\pmb{\theta}, p)$  and  $\mathrm{Acc}(\pmb{\theta}, p)$  are the ideal distribution accuracy and confidences from the model for that probability. For more details see the appendix.

# 3.2 SAMPLE-BASED CALIBRATION

Usually only samples from the true joint distribution are available. Any particular training set is drawn from the distribution to yield

$$
\mathcal {D} = \left\{\left\{\pmb {x} ^ {(i)}, y ^ {(i)} \right\} \right\} _ {i = 1} ^ {N}, \quad \left\{\pmb {x} ^ {(i)}, y ^ {(i)} \right\} \sim \mathrm {p} (\pmb {x}, \pmb {\omega}), \quad y ^ {(i)} \in \left\{\omega_ {1}, \dots , \omega_ {K} \right\}.
$$

The region defined in Eq. (3) is now changed to be indices of the samples:

$$
\mathcal {S} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon) = \left\{i \mid \left| \mathrm {P} \left(\omega_ {j} \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta}\right) - p \right| \leq \epsilon , \boldsymbol {x} ^ {(i)} \in \mathcal {D} \right\}, \tag {9}
$$

The sample-based version of "perfect" calibration in Eq. (2) can then be expressed as:

$$
\frac {1}{\left| \mathcal {S} _ {j} ^ {p} (\boldsymbol {\theta} , \epsilon) \right|} \sum_ {i \in \mathcal {S} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon)} \mathrm {P} \left(\omega_ {j} \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta}\right) = \frac {1}{\left| \mathcal {S} _ {j} ^ {p} (\boldsymbol {\theta} , \epsilon) \right|} \sum_ {i \in \mathcal {S} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon)} \delta \left(y ^ {(i)}, \omega_ {j}\right), \quad \forall p, \omega_ {j}, \epsilon \rightarrow 0 \tag {10}
$$

as  $N\to \infty$ . When considering finite data, in this case  $N$  samples, it is important to set  $\epsilon$  appropriately. Setting different  $\epsilon$  yields different regions and leads to different calibration results (Kumar et al., 2019). Thus it is important to specify  $\epsilon$  when defining calibration for a system.

Similarly, the distribution form of top-label calibration can be written in terms of samples as Eq. (4), with different regions considered:

$$
\tilde {S} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon) = S _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon) \cap \left\{i \mid \omega_ {j} = \arg \max  _ {\omega} \mathrm {P} \left(\omega \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta}\right), \boldsymbol {x} ^ {(i)} \in \mathcal {D} \right\} \tag {11}
$$

The sample-based calibration losses in region  $S_{j}^{p}(\theta ,\epsilon)$  can be defined based on Eq. (10). For example ACE in Eq. (6) can be expressed in its sample-based form (Hendrycks et al., 2019)

$$
\operatorname {A C E} (\boldsymbol {\theta}, \epsilon) = \frac {1}{N K} \sum_ {p \in \mathcal {P} (\epsilon)} \left| \sum_ {j = 1} ^ {K} \sum_ {i \in \mathcal {S} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon)} \left(\mathrm {P} \left(\omega_ {j} \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta}\right) - \delta \left(y ^ {(i)}, \omega_ {j}\right)\right) \right| \tag {12}
$$

where  $\mathcal{P}(\epsilon) = \{p|p = \min \{1,(2z - 1)\epsilon \} ,z\in \mathbb{Z}^{+}\}$ , and  $\mathbb{Z}^+$  is the set of positive integers. The measure of ECE relating to Eq. (7), which only considers the top regions in Eq. (11) can be defined as Guo et al. (2017)

$$
\begin{array}{l} \operatorname {E C E} (\boldsymbol {\theta}, \epsilon) = \frac {1}{N} \sum_ {p \in \mathcal {P} (\epsilon)} \left| \sum_ {j = 1} ^ {K} \sum_ {i \in \tilde {\mathcal {S}} _ {j} ^ {p} (\boldsymbol {\theta}, \epsilon)} \left(\mathrm {P} \left(\omega_ {j} \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta}\right) - \delta \left(y ^ {(i)}, \omega_ {j}\right)\right) \right| (13) \\ = \sum_ {p \in \mathcal {P} (\epsilon)} \frac {\left(\sum_ {j = 1} ^ {K} | \tilde {\mathcal {S}} _ {j} ^ {p} (\boldsymbol {\theta} , \epsilon) |\right)}{N} \left| \operatorname {C o n f} (\boldsymbol {\theta}, p) - \operatorname {A c c} (\boldsymbol {\theta}, p) \right| (14) \\ \end{array}
$$

It should be noted that for finite number of sample, the regions  $S_{j}^{p}(\theta ,\epsilon)$  and  $\tilde{S}_j^p (\theta ,\epsilon)$  derived from the samples can be different from the theoretical regions, leading to difference between theoretical calibration error measures and the values estimated from the finite samples. This is also referred to as "estimator randomness" by Vaicenavicius et al. (2019). An example is given in A.3 to illustrate this mismatch.

The simplest region specification for calibration is to set  $\epsilon = 1$ . In this case,  $|\mathcal{S}_j^p (\pmb {\theta},1)| = N$ , and the "minimum" perfect calibration requirement for a system with parameters  $\pmb{\theta}$  becomes

$$
\frac {1}{N} \sum_ {i = 1} ^ {N} P \left(\omega_ {j} \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta}\right) = \frac {1}{N} \sum_ {i = 1} ^ {N} \delta \left(y ^ {(i)}, \omega_ {j}\right), \quad \forall \omega_ {j} \tag {15}
$$

This is also referred to as global calibration in this paper. Similarly, global top-label calibration can be defined as

$$
\frac {1}{N} \sum_ {i = 1} ^ {N} \mathrm {P} \left(\hat {y} ^ {(i)} \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta}\right) = \frac {1}{N} \sum_ {i = 1} ^ {N} \delta \left(y ^ {(i)}, \hat {y} ^ {(i)}\right), \quad \hat {y} ^ {(i)} = \arg \max  _ {\omega} \mathrm {P} (\omega \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta}) \tag {16}
$$

# 4 ENSEMBLE CALIBRATION

An interesting question when using ensembles is whether calibrating the ensemble members is sufficient to ensure calibrated predictions. Initially the ensemble model will be viewed as an approximation to Bayesian parameter estimation. Given training data  $\mathcal{D}$ , the prediction of class  $\omega_{j}$  is:

$$
\begin{array}{l} \mathrm {P} \left(\omega_ {j} \mid \boldsymbol {x} ^ {*}, \mathcal {D}\right) = \mathbb {E} _ {\boldsymbol {\theta} \sim \mathrm {p} (\boldsymbol {\theta} | \mathcal {D})} \left[ \mathrm {P} \left(\omega_ {j} \mid \boldsymbol {x} ^ {*}; \boldsymbol {\theta}\right) \right] = \int \mathrm {P} \left(\omega_ {j} \mid \boldsymbol {x} ^ {*}; \boldsymbol {\theta}\right) \mathrm {p} (\boldsymbol {\theta} | \mathcal {D}) \mathrm {d} \boldsymbol {x} \\ \approx \quad P \left(\omega_ {j} \mid \boldsymbol {x} ^ {*}; \boldsymbol {\Theta}\right) = \frac {1}{M} \sum_ {m = 1} ^ {M} \mathrm {P} \left(\omega_ {j} \mid \boldsymbol {x} ^ {*}; \boldsymbol {\theta} ^ {(m)}\right); \quad \boldsymbol {\theta} ^ {(m)} \sim \mathrm {p} (\boldsymbol {\theta} | \mathcal {D}) \tag {17} \\ \end{array}
$$

where Eq. (17) is an ensemble, Monte-Carlo, approximation to the full Bayesian integration, with  $\pmb{\theta}^{(m)}$  the  $m$ -th ensemble member parameters in the ensemble  $\Theta$ . The predictions of ensemble and members are  $\hat{y}_m^* = \arg \max_{\omega}\{\mathsf{P}(\omega |\pmb{x}^*;\pmb{\theta}^{(m)})\}, \hat{y}_{\mathbb{E}}^* = \arg \max_{\omega}\left\{\frac{1}{M}\sum_{m=1}^{M}\mathsf{P}(\omega|\pmb{x}^*;\pmb{\theta}^{(m)})\right\}$ .

# 4.1 THEORETICAL ANALYSIS

For ensemble methods it is only important that the final ensemble prediction,  $\hat{y}_{\mathrm{E}}$ , is well calibrated, rather than the individual ensemble members. It is useful to examine the relationship between this ensemble prediction and the predictions from the individual models when the ensemble members are top-label calibrated. Consider a particular top-label calibration region for the ensemble prediction,  $\tilde{\mathcal{R}}^p (\Theta ,\epsilon)$ , related to Eq. (4), the following expression is true

$$
\int_ {\boldsymbol {x} \in \tilde {\mathcal {R}} ^ {p} (\boldsymbol {\Theta}, \epsilon)} \frac {1}{M} \sum_ {m = 1} ^ {M} \mathrm {P} \left(\hat {y} _ {\mathrm {E}} \mid \boldsymbol {x}; \boldsymbol {\theta} ^ {(m)}\right) \mathrm {p} (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} \leq \int_ {\boldsymbol {x} \in \tilde {\mathcal {R}} ^ {p} (\boldsymbol {\Theta}, \epsilon)} \frac {1}{M} \sum_ {m = 1} ^ {M} \mathrm {P} \left(\hat {y} _ {m} \mid \boldsymbol {x}; \boldsymbol {\theta} ^ {(m)}\right) \mathrm {p} (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} \tag {18}
$$

where the ensemble region is defined as  $\tilde{\mathcal{R}}^p (\Theta ,\epsilon) = \left\{\boldsymbol {x}\bigg||\mathrm{P}(\hat{y}_{\mathrm{E}}|\boldsymbol {x};\Theta) - p|\leq \epsilon ,\boldsymbol {x}\in \mathcal{X}\right\}$ . For all regions  $\tilde{\mathcal{R}}^p (\Theta ,\epsilon)$  the ensemble is no more confident than the average confidence of individual member predictions. This puts bounds on performance of the ensemble prediction if the resulting ensemble prediction is top-label calibrated, and the ensemble member regions yield  $\tilde{\mathcal{R}}^p (\Theta ,\epsilon)$ . Now

$$
\int_ {\boldsymbol {x} \in \tilde {\mathcal {R}} ^ {p} (\boldsymbol {\Theta}, \epsilon)} \mathrm {P} \left(\hat {y} _ {\mathrm {E}} | \boldsymbol {x}; \boldsymbol {\Theta}\right) \mathrm {p} (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} = \int_ {\boldsymbol {x} \in \tilde {\mathcal {R}} ^ {p} (\boldsymbol {\Theta}, \epsilon)} \mathrm {P} \left(\hat {y} _ {\mathrm {E}} | \boldsymbol {x}\right) \mathrm {p} (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} \tag {19}
$$

From Eq. (18) the left hand-side of this expression, the ensemble region confidence, cannot be greater than the average ensemble member confidence. If the regions associated with the ensemble prediction and members are the same, then for top-label calibrated members this average confidence is the same as the average ensemble member accuracy. Furthermore, if the ensemble prediction is top-label calibrated, then this average ensemble member accuracy bounds the ensemble prediction accuracy, meaning there is no ensemble performance gain. However in general the regions are not the same, and the ensemble prediction is not necessarily calibrated. It is possible to consider global regions. If the members of the ensemble are globally calibrated, then the ensemble prediction will be globally calibrated. However, this is not the case for global top-label calibration so Eq. (19) doesn't necessarily follow even for global regions, see A.1 for proof.

Now consider sample-calibration applying Eq. (18) at the global region level with ensemble members that are global top-label calibrated. This yields

$$
\frac {1}{N} \sum_ {i = 1} ^ {N} \frac {1}{M} \sum_ {m = 1} ^ {M} \mathrm {P} \left(\hat {y} _ {\mathrm {E}} ^ {(i)} \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta} ^ {(m)}\right) \leq \frac {1}{M} \sum_ {m = 1} ^ {M} \frac {1}{N} \sum_ {i = 1} ^ {N} \mathrm {P} \left(\hat {y} _ {m} ^ {(i)} \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta} ^ {(m)}\right) \tag {20}
$$

where  $\hat{y}_{\mathbf{E}}^{(i)}$  and  $\hat{y}_m^{(i)}$  are the ensemble and member sample  $\pmb{x}^{(i)}$  predictions and from Eq. (16)

$$
\frac {1}{N} \sum_ {i = 1} ^ {N} \delta \left(y ^ {(i)}, \hat {y} _ {m} ^ {(i)}\right) = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathrm {P} \left(\hat {y} _ {m} ^ {(i)} \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta} ^ {(m)}\right), \quad m = 1, \dots , M \tag {21}
$$

Combining Eq. (20) and Eq. (21) yields the following inequality

$$
\operatorname {C o n f} _ {\text {e n s}} = \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {1}{M} \sum_ {m = 1} ^ {M} \mathrm {P} \left(\hat {y} _ {\mathrm {E}} ^ {(i)} \mid \boldsymbol {x} ^ {(i)}; \boldsymbol {\theta} ^ {(m)}\right) \leq \frac {1}{M} \sum_ {m = 1} ^ {M} \frac {1}{N} \sum_ {i = 1} ^ {N} \delta \left(y ^ {(i)}, \hat {y} _ {m} ^ {(i)}\right) = \mathrm {A c c} _ {\mathrm {m e m}} \tag {22}
$$

Given this expression it is worth examining two scenarios, illustrated by the two inequalities in Eq. (23), that relate the inequality in Eq. (22) to the accuracy of the ensemble prediction.

$$
\operatorname {C o n f} _ {\text {e n s}} \leq \frac {1}{N} \sum_ {i = 1} ^ {N} \delta \left(y ^ {(i)}, \hat {y} _ {\mathrm {E}} ^ {(i)}\right); \quad \frac {1}{N} \sum_ {i = 1} ^ {N} \delta \left(y ^ {(i)}, \hat {y} _ {\mathrm {E}} ^ {(i)}\right) \leq \mathbf {A c c} _ {\text {m e m}} \tag {23}
$$

If the ensemble prediction is global top-label calibrated, equality on the left expression, then the ensemble performance cannot exceed that of the ensemble average, the right-hand inequality. In this scenario there is no benefit in using an ensemble. If the ensemble is not top-label calibrated and under-confident, the left inequality above, then the ensemble performance is not bounded by the right-hand inequality. In practice, there is no constraint that the ensemble prediction should be calibrated, thus ensemble prediction calibration is required even for top-label calibrated members.

In the above discussion, the ensemble members are combined with uniform weights, motivated from a Bayesian approximation perspective. When, for example, multiple different topologies are used as members of the ensemble, a non-uniform averaging of the members of the ensemble, reflecting the model complexities and performance may be useful. Using non-uniform weights, Eq. (20) is still true and the discussion of Eq. (21)-(23) still apply.

# 4.2 TEMPERATURE ANNEALING FOR ENSEMBLE CALIBRATION

Calibrating the ensemble in Eq. (23) can be performing using a function with some parameters,  $t$ ,  $f:[0,1] \to [0,1]$  for scaling probabilities. There are two modes for calibrating an ensemble:

Pre-combination Mode. the function is applied to the probabilities predicted by members, prior to combining the members to obtain ensemble prediction using a set of calibration parameters  $\mathbf{T}$ .

$$
\mathrm {P} _ {\text {p r e}} \left(\hat {y} _ {\mathrm {E}} \mid \boldsymbol {x}; \boldsymbol {\Theta}, \boldsymbol {T}\right) = \frac {1}{M} \sum_ {m = 1} ^ {M} f \left(\mathrm {P} \left(\hat {y} _ {\mathrm {E}} \mid \boldsymbol {x}; \boldsymbol {\theta} ^ {(m)}\right), \boldsymbol {t} ^ {(m)}\right) \tag {24}
$$

Post-combination Mode. the function is applied to the ensemble predicted probability after combining members' predictions.

$$
\mathrm {P} _ {\text {p o s t}} \left(\hat {y} _ {\mathrm {E}} | \boldsymbol {x}; \boldsymbol {\Theta}, \boldsymbol {t}\right) = f \left(\left(\frac {1}{M} \sum_ {m = 1} ^ {M} \mathrm {P} \left(\hat {y} _ {\mathrm {E}} | \boldsymbol {x}; \boldsymbol {\theta} ^ {(m)}\right)\right), \boldsymbol {t}\right) \tag {25}
$$

There are many functions for transforming predicted probability in the calibration literature, e.g. histogram binning, Platt scaling and temperature annealing. However, histogram binning shouldn't be adopted in the pre-combination mode as scaling function  $f$  for calibrating multi-class ensemble, as the transformed values may not yield a valid PMF.

As shown in Guo et al. (2017), temperature scaling is a simple, effective, option for the mapping function  $f$ , which scales the logit values associated with the posterior by a temperature  $t$ ,  $f(z; t) = \exp \{z / t\} / \sum_{j} \exp \{z_{j} / t\}$ . Here a single temperature is used for scaling logits for all samples. This leads to the problem that the entropy of the predictions for all regions are either increased or decreased. From Eq. (2) the temperature can be made region specific.

$$
f _ {\mathrm {d y n}} (\boldsymbol {z}; \boldsymbol {t}) = \frac {\exp \left\{\boldsymbol {z} / t _ {r} \right\}}{\sum_ {j} \exp \left\{z _ {j} / t _ {r} \right\}}, \quad \text {i f} \max  _ {i} \frac {\exp \left\{z _ {i} \right\}}{\sum_ {j} \exp \left\{z _ {j} \right\}} \in \mathcal {R} _ {r} \tag {26}
$$

To determine the optimal set of temperatures, the samples in the validation set are divided into  $R$  regions based on the ensemble predictions (e.g.  $\mathcal{R}_1 = [0,0.3)$ ,  $\mathcal{R}_2 = [0.3,0.6)$ , and  $\mathcal{R}_3 = [0.6,1]$ ). Each region has an individual temperature for scaling  $\{\mathcal{R}_r,t_r\}_{r = 1}^R$ .

# 4.3 EMPIRICAL RESULTS

Experiments were conducted on CIFAR-100 (and CIFAR-10 in the A.4). The data partition was 45,000/5,000/10,000 images for train/validation/test. We train LeNet (LeCun et al., 1998), DenseNet (Huang et al., 2017) and Wide ResNet (Zagoruyko & Komodakis, 2016) following the original training recipes in each paper (more details in A.4). The results presented are slightly lower than that in the original papers, as 5,000 images were held-out to enable calibration parameter optimisation.

![](images/09305d84b40fa87c57707847d9e6722bcef7dbbf6a1a4efb49b679562031d5a3.jpg)  
(a) LeNet 5

![](images/b4260cec7308bcec86597a62496b5600afbe9dc47e176e9cd8cd60379d786145.jpg)  
(b) DenseNet 100

![](images/f523b5ef07b5e965c13563ff470e330c0edb8dde439b4391718e1765c952cc48.jpg)  
(c) Wide ResNet 28

![](images/396cdf6f3fada36f3046c1fbd8e169403bbb017899933df1747cffca1b013532.jpg)

![](images/2339a6befaa38d800990803160705402ddea80c858cd0d7d61513bec6b4c7c82.jpg)

![](images/36e5e4d9a07076260e0a1f72b58e439456f3c174f72356624c509b8a518b8fe9.jpg)

![](images/23c8b280c29ece429c04506d5cdf03daf12d5c0b4884c489b293d81868e0dfa9.jpg)  
Figure 1: Top-label calibration error and accuracy of members (mem) and the whole ensemble (ens) on CIFAR-100 (test set) using LeNet, DenseNet and ResNet. "pre" denotes the calibration where shared temperature is applied to members before combination. The reliability curves show the calibrated members and calibrated ensembles with optimal temperature values.

![](images/c6fa6d0da42271a508afdce732ea508e82e6657ccca871bcda8e5aa82c520383.jpg)

![](images/09ebfb2a68268377dce5ab8d006ec2cb9a40d1b43a2a1fa7bfd637343b6b2a68.jpg)

Figure 1 examines the empirical performance of ensemble calibration on CIFAR-100 test set using the three trained networks. The middle row shows the ECE of ensemble members and ensemble prediction at different temperatures. The optimal calibration temperature for the ensemble prediction are consistently smaller than those associated with the ensemble members. This indicates that the ensemble predictions are less confident than those of the members, as stated in Eq. (20). The bottom row of figures show the reliability curves when the ensemble members are calibrated with optimal temperature values, and the resulting combination. It is clear that calibrating the ensemble members, using temperature, does not yield a calibrated ensemble prediction. Furthermore for all models the ensemble prediction is less confident than it should be, the line is above the diagonal. As discussed in Eq. (23), this is necessary, or the ensemble prediction is no better, which is clearly not the case for the performance plots in the top row. This ensemble performance is relatively robust to poorly calibrated ensemble members, with consistent performance over a wide range of temperatures.

![](images/7e3fc4c8b309107a6681457e91339b4acae53c41f8f9fe06b620dc4e5cf98feb.jpg)  
LEN+DSN100+DSN121+RSN

![](images/d6a50137f16d1be27f9dec670b1efb42bda693fce073cdf002bc67e14a95bc38.jpg)  
DSN100+DSN121+RSN  
Figure 2: Reliability curves of weighted combination of 4 calibrated structures, LeNet, DenseNet 100 and DenseNet 121 and wide ResNet 28-10 on CIFAR-100. The weights are estimated by Max LL. Each structure is an ensemble of 10 models.

![](images/f5e8b9b3c2f284bca3dd7680298fa9cc8759525bf5c81b98ce41be383707052d.jpg)  
DSN121+RSN

Table 1 shows the calibration performance using three temperature scaling methods, pre-, post- and dynamic post-combination. The temperatures are optimized to minimize ECE (Liang et al., 2020) on the validation data. All three methods effectively improve the ensemble prediction calibration, with the dynamic approach yielding the best performance.  

<table><tr><td>Model</td><td>Cal.</td><td>Acc.(%)</td><td>NLL</td><td>ACCE(10-4)</td><td>ACE(10-4)</td><td>ECCE(10-2)</td><td>ECE(10-2)</td></tr><tr><td rowspan="4">LEN</td><td>—</td><td>49.32</td><td>1.9759</td><td>30.68</td><td>23.81</td><td>16.47</td><td>11.83</td></tr><tr><td>pre</td><td>49.37</td><td>1.9640</td><td>22.93</td><td>8.66</td><td>13.09</td><td>3.19</td></tr><tr><td>post</td><td>49.32</td><td>1.9290</td><td>21.65</td><td>5.96</td><td>13.38</td><td>2.02</td></tr><tr><td>dyn.</td><td>49.32</td><td>1.9275</td><td>20.84</td><td>4.43</td><td>12.90</td><td>2.05</td></tr><tr><td rowspan="4">DSN 100</td><td>—</td><td>81.24</td><td>0.6704</td><td>16.50</td><td>5.63</td><td>8.73</td><td>2.48</td></tr><tr><td>pre</td><td>81.25</td><td>0.6901</td><td>16.89</td><td>6.67</td><td>8.60</td><td>1.98</td></tr><tr><td>post</td><td>81.24</td><td>0.6886</td><td>16.89</td><td>6.63</td><td>8.57</td><td>1.95</td></tr><tr><td>dyn.</td><td>81.24</td><td>0.6764</td><td>16.24</td><td>4.87</td><td>8.36</td><td>1.11</td></tr><tr><td rowspan="4">DSN 121</td><td>—</td><td>82.70</td><td>0.6307</td><td>15.76</td><td>3.63</td><td>8.65</td><td>1.74</td></tr><tr><td>pre</td><td>82.73</td><td>0.6282</td><td>15.71</td><td>3.32</td><td>8.71</td><td>1.85</td></tr><tr><td>post</td><td>82.70</td><td>0.6303</td><td>15.83</td><td>3.62</td><td>8.72</td><td>1.80</td></tr><tr><td>dyn.</td><td>82.70</td><td>0.6316</td><td>15.64</td><td>3.37</td><td>8.81</td><td>1.85</td></tr><tr><td rowspan="4">RSN</td><td>—</td><td>83.54</td><td>0.6254</td><td>17.02</td><td>7.62</td><td>9.20</td><td>3.42</td></tr><tr><td>pre</td><td>83.50</td><td>0.6143</td><td>15.33</td><td>2.36</td><td>8.88</td><td>1.81</td></tr><tr><td>post</td><td>83.54</td><td>0.6129</td><td>15.54</td><td>3.02</td><td>8.83</td><td>1.75</td></tr><tr><td>dyn.</td><td>83.54</td><td>0.6119</td><td>15.75</td><td>2.75</td><td>8.90</td><td>0.83</td></tr></table>

Finally, for the topology ensemble, weights were optimised using either maximum likelihood (Max LL) or area under curve (AUC) Zhong & Kwok (2013) (results in A.4). In Figure 2, the ensemble of calibrated structures is shown to be uncalibrated, with reliability curves typically slightly above the diagonal line. When the ensemble prediction is calibrated it can be seen that the calibration for the ensemble prediction is lower than the individual calibration errors in Table 1 ("post" lines).

Table 1: Temperature calibration techniques on CIFAR-100, calibration parameters optimized to minimize ECE on validation set. In the "pre" mode, each member is scaled with one separate temperature. "dyn." denotes dynamic temperature scaling in post-combination mode using 6 region-based temperatures. The structures investigated are LeNet, DenseNet 100, DenseNet 121 and wide ResNet 28.  

<table><tr><td rowspan="2">Weight Est.</td><td colspan="4">Comb. Weight</td><td rowspan="2">Acc. (%)</td><td rowspan="2">Ens Cal.</td><td rowspan="2">NLL</td><td rowspan="2">ACE (10-4)</td><td rowspan="2">ECE (10-2)</td></tr><tr><td>LEN</td><td>DSN100</td><td>DSN121</td><td>RSN</td></tr><tr><td rowspan="6">Max LL</td><td rowspan="2">0.02</td><td rowspan="2">0.19</td><td rowspan="2">0.30</td><td rowspan="2">0.49</td><td rowspan="2">83.75</td><td>—</td><td>0.5766</td><td>4.97</td><td>2.24</td></tr><tr><td>✓</td><td>0.5698</td><td>1.42</td><td>1.20</td></tr><tr><td rowspan="2">—</td><td rowspan="2">0.22</td><td rowspan="2">0.30</td><td rowspan="2">0.48</td><td rowspan="2">83.80</td><td>—</td><td>0.5741</td><td>3.74</td><td>2.00</td></tr><tr><td>✓</td><td>0.5714</td><td>1.52</td><td>1.29</td></tr><tr><td rowspan="2">—</td><td rowspan="2">—</td><td rowspan="2">0.44</td><td rowspan="2">0.56</td><td rowspan="2">83.86</td><td>—</td><td>0.5816</td><td>3.64</td><td>2.06</td></tr><tr><td>✓</td><td>0.5801</td><td>2.36</td><td>1.35</td></tr></table>

Table 2: Topology ensembles for CIFAR-100, optimal weights using ML estimation. Calibrations of each topology and ensemble using post-combination mode ("post" in Table 1).

# 5 CONCLUSIONS

State-of-the-art deep learning models often exhibit poor calibration performance. In this paper two aspects of calibration for these models are investigated: the theoretical definition of calibration and associated attributes for both general and top-label calibration; and the application of calibration to ensemble methods that are often used in deep-learning approaches for improved performance and uncertainty estimation. It is shown that calibrating members of the ensemble is not sufficient to ensure that the ensemble prediction is itself calibrated. The resulting ensemble predictions will be under-confident, requiring calibration functions to be optimised for the ensemble prediction, rather than ensemble members. These theoretical results are backed-up by empirical analysis on CIFAR-100 deep-learning models, with ensemble performance being robust to poorly calibrated ensemble members but requiring calibration even with well calibrated members.

# REFERENCES

Arsenii Ashukha, Alexander Lyzhov, Dmitry Molchanov, and Dmitry Vetrov. Pitfalls of in-domain uncertainty estimation and ensembling in deep learning. arXiv preprint arXiv:2002.06470, 2020.  
Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, et al. End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316, 2016.  
Jochen Brocker. Estimating reliability and resolution of probability forecasts through decomposition of the empirical score. Climate dynamics, 39(3-4):655-667, 2012.  
Tilmann Gneiting, Fadoua Balabdaoui, and Adrian E Raftery. Probabilistic forecasts, calibration and sharpness. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(2): 243-268, 2007.  
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. ICML, 2017.  
Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich. Deep anomaly detection with outlier exposure. *ICLR*, 2019.  
Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4700-4708, 2017.  
Xiaoqian Jiang, Melanie Osl, Jihoon Kim, and Lucila Ohno-Machado. Calibrating predictive model estimates to support personalized medicine. Journal of the American Medical Informatics Association, 19(2):263-274, 2012.  
Volodymyr Kuleshov and Percy S Liang. Calibrated structured prediction. In Advances in Neural Information Processing Systems, pp. 3474-3482, 2015.  
Meelis Kull, Telmo Silva Filho, and Peter Flach. Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers. In Artificial Intelligence and Statistics, pp. 623-631, 2017.  
Meelis Kull, Miquel Perello Nieto, Markus Kangsepp, Telmo Silva Filho, Hao Song, and Peter Flach. Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration. In Advances in Neural Information Processing Systems, pp. 12316-12326, 2019.  
Ananya Kumar, Percy S Liang, and Tengyu Ma. Verified uncertainty calibration. In Advances in Neural Information Processing Systems, pp. 3792-3803, 2019.  
Aviral Kumar, Sunita Sarawagi, and Ujjwal Jain. Trainable calibration measures for neural networks from kernel mean embeddings. In International Conference on Machine Learning, pp. 2805-2814, 2018.  
Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in neural information processing systems, pp. 6402-6413, 2017.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Gongbo Liang, Yu Zhang, and Nathan Jacobs. Neural network calibration for medical imaging classification using dca regularization. In ICML UDL, 2020.  
Mahdi Pakdaman Naeini, Gregory F Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Proceedings of the... AAAI Conference on Artificial Intelligence. AAAI Conference on Artificial Intelligence, volume 2015, pp. 2901. NIH Public Access, 2015.

Khanh Nguyen and Brendan O'Connor. Posterior calibration and exploratory analysis for natural language processing models. EMNLP, 2015.  
Alexandru Niculescu-Mizil and Rich Caruana. Predicting good probabilities with supervised learning. In Proceedings of the 22nd international conference on Machine learning, pp. 625-632, 2005.  
Jeremy Nixon, Michael W Dusenberry, Linchuan Zhang, Ghassen Jerfel, and Dustin Tran. Measuring calibration in deep learning. In CVPR Workshops, pp. 38-41, 2019.  
John Platt et al. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3):61-74, 1999.  
Adrian E Raftery, Tilmann Gneiting, Fadoua Balabdaoui, and Michael Polakowski. Using bayesian model averaging to calibrate forecast ensembles. Monthly weather review, 133(5):1155-1174, 2005.  
Rahul Rahaman and Alexandre H Thiery. Uncertainty quantification and deep ensembles. arXiv preprint arXiv:2007.08792, 2020.  
Asa Cooper Stickland and Iain Murray. Diverse ensembles improve calibration. ICML 2020 workshop on Uncertainty Robustness in Deep Learning, 2020.  
David Stutz, Matthias Hein, and Bernt Schiele. Confidence-calibrated adversarial training: Generalizing to unseen attacks. ICML 2020 workshop on Uncertainty Robustness in Deep Learning, 2020.  
Gia-Lac Tran, Edwin V Bonilla, John Cunningham, Pietro Michiardi, and Maurizio Filippone. Calibrating deep convolutional gaussian processes. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 1554-1563. PMLR, 2019.  
Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, and Thomas B Schon. Evaluating model calibration in classification. Proceedings of Machine Learning Research, 2019.  
Yeming Wen, Ghassen Jerfel, Rafael Muller, Michael W Dusenberry, Jasper Snoek, Balaji Lakshminarayanan, and Dustin Tran. Improving calibration of batchsemble with data augmentation. ICML 2020 workshop on Uncertainty Robustness in Deep Learning, 2020.  
David Widmann, Fredrik Lindsten, and Dave Zachariah. Calibration tests in multi-class classification: A unifying framework. In Advances in Neural Information Processing Systems, pp. 12257-12267, 2019.  
Bianca Zadrozny and Charles Elkan. Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers. In Icml, volume 1, pp. 609-616. CiteSeer, 2001.  
Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 694-699, 2002.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.  
Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. *ICLR*, 2018.  
Wenliang Zhong and James T Kwok. Accurate probability calibration for multiple classifiers. In Twenty-Third International Joint Conference on Artificial Intelligence. CiteSeer, 2013.
