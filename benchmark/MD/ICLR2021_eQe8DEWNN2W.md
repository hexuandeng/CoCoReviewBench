# CALIBRATION OF NEURAL NETWORKS USING SPLINES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Calibrating neural networks is of utmost importance when employing them in safety-critical applications where the downstream decision making depends on the predicted probabilities. Measuring calibration error amounts to comparing two empirical distributions. In this work, we introduce a binning-free calibration measure inspired by the classical Kolmogorov-Smirnov (KS) statistical test in which the main idea is to compare the respective cumulative probability distributions. From this, by approximating the empirical cumulative distribution using a differentiable function via splines, we obtain a recalibration function, which maps the network outputs to actual (calibrated) class assignment probabilities. The spline-fitting is performed using a held-out calibration set and the obtained recalibration function is evaluated on an unseen test set. We tested our method against existing calibration approaches on various image classification datasets and our spline-based recalibration approach consistently outperforms existing methods on KS error as well as other commonly used calibration measures.

# 1 INTRODUCTION

Despite the success of modern neural networks they are shown to be poorly calibrated (Guo et al. (2017)), which has led to a growing interest in the calibration of neural networks over the past few years (Kull et al. (2019); Kumar et al. (2019; 2018); Müller et al. (2019)). Considering classification problems, a classifier is said to be calibrated if the probability values it associates with the class labels match the true probabilities of correct class assignments. For instance, if an image classifier outputs 0.2 probability for the "horse" label for 100 test images, then out of those 100 images approximately 20 images should be classified as horse. It is important to ensure calibration when using classifiers for safety-critical applications such as medical image analysis and autonomous driving where the downstream decision making depends on the predicted probabilities.

One of the important aspects of machine learning research is the measure used to evaluate the performance of a model and in the context of calibration, this amounts to measuring the difference between two empirical probability distributions. To this end, the popular metric, Expected Calibration Error (ECE) (Naeini et al. (2015)), approximates the classwise probability distributions using histograms and takes an expected difference. This histogram approximation has a weakness that the resulting calibration error depends on the binning scheme (number of bins and bin divisions). Even though the drawbacks of ECE have been pointed out and some improvements have been proposed (Kumar et al. (2019); Nixon et al. (2019)), the histogram approximation has not been eliminated. $^{1}$

In this paper, we first introduce a simple, binning-free calibration measure inspired by the classical Kolmogorov-Smirnov (KS) statistical test (Kolmogorov (1933); Smirnov (1939)), which also provides an effective visualization of the degree of miscalibration similar to the reliability diagram (Niculescu-Mizil & Caruana (2005)). To this end, the main idea of the KS-test is to compare the respective classwise cumulative (empirical) distributions. Furthermore, by approximating the empirical cumulative distribution using a differentiable function via splines (McKinley & Levine (1998)), we obtain an analytical recalibration function which maps the given network outputs to the actual class assignment probabilities. Such a direct mapping was previously unavailable and the problem has been

approached indirectly via learning, for example, by optimizing the (modified) cross-entropy loss (Guo et al. (2017); Mukhoti et al. (2020); Müller et al. (2019)). Similar to the existing methods (Guo et al. (2017); Kull et al. (2019)) the spline-fitting is performed using a held-out calibration set and the obtained recalibration function is evaluated on an unseen test set.

We evaluated our method against existing calibration approaches on various image classification datasets and our spline-based recalibration approach consistently outperforms existing methods on KS error, ECE as well as other commonly used calibration measures. Our approach to calibration does not update the model parameters, which allows it to be applied on any trained network and it retains the original classification accuracy in all the tested cases.

# 2 NOTATION AND PRELIMINARIES

We abstract the network as a function  $f_{\theta}:\mathcal{D}\to [0,1]^{K}$ , where  $\mathcal{D}\subset \mathbb{R}^d$ , and write  $f_{\theta}(\mathbf{x}) = \mathbf{z}$ . Here,  $\mathbf{x}$  may be an image, or other input datum, and  $\mathbf{z}$  is a vector, sometimes known as the vector of logits. In this paper, the parameters  $\theta$  will not be relevant, and we write simply  $f$  to represent the network function. Moreover, a function of this type will be referred to as a classifier, which may be of some other kind than a neural network.

In a classification problem,  $K$  is the number of classes to be distinguished, and we call the value  $z_{k}$  (the  $k$ -th component of vector  $\mathbf{z}$ ) the score for the class  $k$ . If the final layer of a network is a softmax layer, then the values  $z_{k}$  satisfy  $\sum_{k=1}^{K} z_{k} = 1$ , and  $z_{k} \geq 0$ . Hence, the  $z_{k}$  are pseudoprobabilities, though they do not necessarily have anything to do with real probabilities of correct class assignments. Typically, the value  $y^{*} = \arg \max_{k} z_{k}$  is taken as the (top-1) prediction of the network, and the corresponding score,  $\max_{k} z_{k}$  is called the confidence of the prediction. However, the term confidence does not have any mathematical meaning in this context and we deprecate its use.

We assume we are given a set of training data  $(\mathbf{x}_i, y_i)_{i=1}^n$ , where  $\mathbf{x}_i \in \mathcal{D}$  is an input data element, which for simplicity we call an image, and  $y_i \in \mathcal{K} = \{1, \dots, K\}$  is the so-called ground-truth label. Our method also uses two other sets of data, called calibration data and test data.

It would be desirable if the numbers  $z_{k}$  output by a network represented true probabilities. For this to make sense, we posit the existence of joint random variables  $(X,Y)$ , where  $X$  takes values in a domain  $\mathcal{D} \subset \mathbb{R}^{d}$ , and  $Y$  takes values in  $\mathcal{K}$ . Further, let  $Z = f(X)$ , another random variable, and  $Z_{k} = f_{k}(X)$  be its  $k$ -th component. Note that in this formulation  $X$  and  $Y$  are joint random variables, and the probability  $P(Y \mid X)$  is not assumed to be 1 for single class, and 0 for the others.

A network is said to be calibrated if for every class  $k$

$$
P (Y = k \mid Z = \mathbf {z}) = z _ {k}. \tag {1}
$$

This can be written briefly as  $P(k \mid f(\mathbf{x})) = f_k(\mathbf{x}) = z_k$ . Thus, if the network takes input  $\mathbf{x}$  and outputs  $\mathbf{z} = f(\mathbf{x})$ , then  $z_k$  represents the probability (given  $f(\mathbf{x})$ ) that image  $\mathbf{x}$  belongs to class  $k$ .

The probability  $P(k \mid \mathbf{z})$  is difficult to evaluate, even empirically, and most metrics (such as ECE) use or measure a different notion called classwise calibration (Kull et al. (2019); Zadrozny & Elkan (2002)), defined as,

$$
P (Y = k \mid Z _ {k} = z _ {k}) = z _ {k}. \tag {2}
$$

This paper uses this definition (2) of calibration in the proposed KS metric.

Calibration and accuracy of a network are different concepts. For instance, one may consider a classifier that simply outputs the class probabilities for the data, ignoring the input  $\mathbf{x}$ . Thus, if  $f_{k}(\mathbf{x}) = z_{k} = P(Y = k)$ , this classifier  $f$  is calibrated but the accuracy is no better than the random predictor. Therefore, in calibration of a classifier, it is important that this is not done while sacrificing classification (for instance top-1) accuracy.

The top- $r$  prediction. The classifier  $f$  being calibrated means that  $f_{k}(\mathbf{x})$  is calibrated for each class  $k$ , not only for the top class. This means that scores  $z_{k}$  for all classes  $k$  give a meaningful estimate of the probability of the sample belonging to class  $k$ . This is particularly important in medical diagnosis where one may wish to have a reliable estimate of the probability of certain unlikely diagnoses.

Frequently, however, one is most interested in the probability of the top scoring class, the top-1 prediction, or in general the top- $r$  prediction. Suppose a classifier  $f$  is given with values in  $[0, 1]^K$  and

let  $y$  be the ground truth label. Let us use  $f^{(-r)}$  to denote the  $r$ -th top score (so  $f^{(-1)}$  would denote the top score; the notation follows python semantics in which  $A[-1]$  represents the last element in array  $A$ ). Similarly we define  $\max^{(-r)}$  for the  $r$ -th largest value. Let  $f^{(-r)}: \mathcal{D} \to [0,1]$  be defined as

$$
f ^ {(- r)} (\mathbf {x}) = \max  _ {k} ^ {(- r)} f _ {k} (\mathbf {x}), \quad \text {a n d} \quad y ^ {(- r)} = \left\{ \begin{array}{l l} 1 & \text {i f} y = \arg \max  _ {k} ^ {(- r)} f _ {k} (\mathbf {x}) \\ 0 & \text {o t h e r w i s e .} \end{array} \right. \tag {3}
$$

In words,  $y^{(-r)}$  is 1 if the  $r$ -th top predicted class is the correct (ground-truth) choice. The network is calibrated for the top-  $r$  predictor if for all scores  $\sigma$ ,

$$
P \left(y ^ {(- r)} = 1 \mid f ^ {(- r)} (\mathbf {x}) = \sigma\right) = \sigma . \tag {4}
$$

In words, the conditional probability that the top- $r$ -th choice of the network is the correct choice, is equal to the  $r$ -th top score.

Similarly, one may consider probabilities that a datum belongs to one of the top- $r$  scoring classes. The classifier is calibrated for being within-the-top- $r$  classes if

$$
P \left(\sum_ {s = 1} ^ {r} y ^ {(- s)} = 1 \mid \sum_ {s = 1} ^ {r} f ^ {(- s)} (\mathbf {x}) = \sigma\right) = \sigma . \tag {5}
$$

Here, the sum on the left is 1 if the ground-truth label is among the top  $r$  choices, 0 otherwise, and the sum on the right is the sum of the top  $r$  scores.

# 3 KOLMOGOROV-SMIRNOV CALIBRATION ERROR

We now consider a way to measure if a classifier is classwise calibrated, including top- $r$  and within-top- $r$  calibration. This test is closely related to the Kolmogorov-Smirnov test (Kolmogorov (1933); Smirnov (1939)) for the equality of two probability distributions. This may be applied when the probability distributions are represented by samples.

We start with the definition of classwise calibration:

$$
P (Y = k \mid f _ {k} (X) = z _ {k}) = z _ {k}. \tag {6}
$$

$$
P (Y = k, f _ {k} (X) = z _ {k}) = z _ {k} P \left(f _ {k} (X) = z _ {k}\right), \quad \text {B a y e s ’ r u l e}.
$$

This may be written more simply but with a less precise notation as

$$
P (z _ {k}, k) = z _ {k} P (z _ {k}).
$$

Motivation of the KS test. One is motivated to test the equality (or difference between) two distributions, defined on the interval  $[0, 1]$ . However, instead of having a functional form of these distributions, one has only samples from them. Given samples  $(\mathbf{x}_i, y_i)$ , it is not straightforward to estimate  $P(z_k)$  or  $P(z_k \mid k)$ , since a given value  $z_k$  is likely to occur only once, or not at all, since the sample set is finite. One possibility is to use histograms of these distributions. However, this requires selection of the bin size, and the division between bins, and the result depends on these parameters. For this reason, we abjure this solution.

The approach suggested by the Kolmogorov-Smirnov test is to compare the cumulative distributions. Thus, with  $k$  given, one tests the equality

$$
\int_ {0} ^ {\sigma} P \left(z _ {k}, k\right) d z _ {k} = \int_ {0} ^ {\sigma} z _ {k} P \left(z _ {k}\right) d z _ {k}. \tag {7}
$$

Writing  $\phi_1(\sigma)$  and  $\phi_2(\sigma)$  to be the two sides of this equation, then the KS-distance between these two distributions is  $\max_{\sigma}|\phi_1(\sigma) - \phi_2(\sigma)|$

The fact that simply the maximum is used here may suggest a lack of robustness, but this is a maximum difference between two integrals, so it reflects an accumulated difference between the two distributions. In fact, if  $z_{k}$  consistently over or under-estimates  $P(k\mid z_k)$  (which is usually the case, at least for top-1 classification), then  $P(k\mid z_k) - z_k$  has constant sign for all values of  $z_{k}$ .

It follows that  $P(z_{k}, k) - z_{k}P(z_{k})$  has constant sign and so the maximum value in the KS-distance is achieved when  $\sigma = 1$ . In this case,

$$
\mathrm {K S} = \int_ {0} ^ {1} | P (z _ {k}, k) - z _ {k} P (z _ {k}) | d z _ {k} = \int_ {0} ^ {1} | P (k | z _ {k}) - z _ {k} | P (z _ {k}) d z _ {k}, \tag {8}
$$

which is the expected difference between  $z_{k}$  and  $P(k\mid z_k)$ . This can be equivalently referred to as the expected calibration error for the class  $k$ .

![](images/fe5b3f3d4e753316e1503361f9e69b651a08ff5cde0cca53b7309a1ee0e4ae21.jpg)  
(a)

![](images/088fbd381522bec6f2d6860af37e5206971239a2be5100542055e52633a03ae0.jpg)  
(b)

![](images/0a321b7b263a983d2c4b5bbb3e934b21a9900a163007cc4d48dca75bc29272ec.jpg)  
Figure 1: Calibration graphs for an uncalibrated DenseNet-40 (Huang et al. (2017)) trained on CIFAR-10 for top-1 class with a KS error of  $5.5\%$ , and top-1 accuracy of  $92.4\%$  on the test set. Here  $(a)$  shows the plot of cumulative score and probability versus the fractile of the test set,  $(b)$  shows the same information with the horizontal axis warped so that the cumulative-score graph is a straight line. This is created as scatter plots of cumulative (score, score): blue and (score, probability): orange. If the network is perfectly calibrated, the probability line will be a straight line coincident with the (score, score) line. This shows that the network is substantially overestimating (score) the probability of the computation.  $(c)$  and  $(d)$  show plots of (non-cumulative) score and probability plotted against fractile, or score. How these plots are produced is described in section 4.  
(c)

![](images/7e9a4814a6baa4e2e1600a80ed20402ccb1b4d9cfc163399cba4871631ca8026.jpg)  
(d)

Sampled distributions. Given samples  $(\mathbf{x}_i, y_i)_{i=1}^N$ , and a fixed  $k$ , one can estimate these cumulative distributions by

$$
\int_ {0} ^ {\sigma} P \left(z _ {k}, k\right) d z _ {k} \approx \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbf {1} \left(f _ {k} \left(\mathbf {x} _ {i}\right) \leq \sigma\right) \times \mathbf {1} \left(y _ {i} = k\right), \tag {9}
$$

where  $\mathbf{1}:\mathcal{B}\to \{0,1\}$  is the function that returns 1 if the Boolean expression is true and otherwise 0. Thus, the sum is simply a count of the number of samples for which  $y_{i} = k$  and  $f_{k}(\mathbf{x}_{i})\leq \sigma$ , and so the integral represents the proportion of the data satisfying this condition. Similarly,

$$
\int_ {0} ^ {\sigma} z _ {k} P \left(z _ {k}\right) d z _ {k} \approx \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbf {1} \left(f _ {k} \left(\mathbf {x} _ {i}\right) \leq \sigma\right) f _ {k} \left(\mathbf {x} _ {i}\right). \tag {10}
$$

These sums can be computed quickly by sorting the data according to the values  $f_{k}(\mathbf{x}_{i})$ , then defining two sequences as follows.

$$
\tilde {h} _ {0} = h _ {0} = 0,
$$

$$
\tilde {h} _ {i} = \tilde {h} _ {i - 1} + \mathbf {1} \left(y _ {i} = k\right) / N, \tag {11}
$$

$$
h _ {i} = h _ {i - 1} + f _ {k} (\mathbf {x} _ {i}) / N.
$$

The two sequences should be the same, and the metric

$$
\mathrm {K S} \left(f _ {k}\right) = \max  _ {i} \left| h _ {i} - \tilde {h} _ {i} \right|, \tag {12}
$$

gives a numerical estimate of the similarity, and hence a measure of the degree of calibration of  $f_{k}$ . This is essentially a version of the Kolmogorov-Smirnov test for equality of two distributions.

Remark. All this discussion holds also when  $k < 0$ , for top-  $r$  and within-top-  $r$  predictions as discussed in section 2. In (11), for instance,  $f_{-1}(\mathbf{x}_i)$  means the top score,  $f_{-1}(\mathbf{x}_i) = \max_k(f_k(\mathbf{x}_i))$ , or more generally,  $f_{-r}(\mathbf{x}_i)$  means the  $r$ -th top score. Similarly, the expression  $y_i = -r$  means that  $y_i$  is the class that has the  $r$ -th top score.

# 4 RECALIBRATION USING SPLINES

The function  $h_i$  defined in (11) computes an empirical approximation

$$
h _ {i} \approx P (Y = k, f _ {k} (X) \leq f _ {k} (\mathbf {x} _ {i})) . \tag {13}
$$

For convenience, the value of  $f_{k}$  will be referred to as the score. We now define a continuous function  $h(t)$  for  $t \in [0,1]$  by

$$
h (t) = P (Y = k, f _ {k} (X) \leq s (t)), \tag {14}
$$

where  $s(t)$  is the  $t$ -th fractile score, namely the value that a proportion  $t$  of the scores  $f_{k}(X)$  lie below. For instance  $s(0.5)$  is the median score. So,  $h_{i}$  is an empirical approximation to  $h(t)$  where  $t = i / N$ . We now provide the basic observation that allows us to compute probabilities given the scores.

Proposition 4.1. If  $h(t) = P(Y = k, f_k(X) \leq s(t))$  as in (14) where  $s(t)$  is the t-th fractile score, then  $h'(t) = P(Y = k \mid f_k(X) = s(t))$ , where  $h'(t) = dh / dt$ .

Proof. The proof relies on the equality  $P(f_{k}(X) \leq s(t)) = t$ . In words, since  $s(t)$  is the value that a fraction  $t$  of the scores are less than or equal, the probability that a score is less than or equal to  $s(t)$ , is (obviously) equal to  $t$ . See the supplementary material for a detailed proof.

Notice  $h^\prime (t)$  allows a direct conversion from score to probability. Therefore, our idea is to approximate  $h_i$  using a differentiable function and take the derivative which would be our recalibration function.

# 4.1 SPLINE FITTING

The function  $h_i$  (shown in fig 1a) is obtained through sampling only. Nevertheless, the sampled graph is smooth and increasing. There are various ways to fit a smooth curve to it, so as to take derivatives. We choose to fit the sampled points  $h_i$  to a cubic spline and take its derivative.

Given sample points  $(u_{i},v_{i})_{i = 1}^{N}$  in  $\mathbb{R}\times \mathbb{R}$ , easily available references show how to fit a smooth spline curve that passes directly through the points  $(u_{i},v_{i})$ . A very clear description is given in McKinley & Levine (1998), for the case where the points  $u_{i}$  are equally spaced. We wish, however, to fit a spline curve with a small number of knot points to do a least-squares fit to the points. For convenience, this is briefly described here.

A cubic spline  $v(u)$  is defined by its values at certain knot points  $(\hat{u}_k, \hat{v}_k)_{k=1}^K$ . In fact, the value of the curve at any point  $u$  can be written as a linear function  $v(u) = \sum_{k=1}^{K} a_k(u) \hat{v}_k = \mathbf{a}^\top(u) \hat{\mathbf{v}}$ , where the coefficients  $a_k$  depend on  $u$ . Therefore, given a set of further points  $(u_i, v_i)_{i=1}^N$ , which may be different from the knot points, and typically more in number, least-squares spline fitting of the points  $(u_i, v_i)$  can be written as a least-squares problem  $\min_{\hat{\mathbf{v}}} \| \mathsf{A}(\mathbf{u}) \hat{\mathbf{v}} - \mathbf{v} \|^2$ , which is solved by standard linear least-squares techniques. Here, the matrix  $\mathsf{A}$  has dimension  $N \times K$  with  $N > K$ . Once  $\hat{\mathbf{v}}$  is found, the value of the spline at any further points  $u$  is equal to  $v(u) = \mathbf{a}(u)^\top \hat{\mathbf{v}}$ , a linear combination of the knot-point values  $\hat{v}_k$ .

Since the function is piecewise cubic, with continuous second derivatives, the first derivative of the spline is computed analytically. Furthermore, the derivative  $v'(u)$  can also be written as a linear combination  $v'(u) = \mathbf{a}'(u)^\top \hat{\mathbf{v}}$ , where the coefficients  $\mathbf{a}'(u)$  can be written explicitly.

Our goal is to fit a spline to a set of data points  $(u_i, v_i) = (i/N, h_i)$  defined in (11), in other words, the values  $h_i$  plotted against fractile score. Then according to Proposition 4.1, the derivative of the spline is equal to  $P(k \mid f_k(X) = s(t))$ . This allows a direct computation of the conditional probability that the sample belongs to class  $k$ .

Since the derivative of  $h_i$  is a probability one might constrain the derivative to be in the range [0, 1] while fitting splines. This can be easily incorporated because the derivative of the spline is a linear expression in  $\hat{v}_i$ . The spline fitting problem thereby becomes a linearly-constrained quadratic program (QP). However, although we tested this, in all the reported experiments, a simple least-squares solver is used without the constraints.

# 4.2 RECALIBRATION

We suppose that the classifier  $f = f_{\theta}$  is fixed, through training on the training set. Typically, if the classifier is tested on the training set, it is very close to being calibrated. However, if a classifier  $f$  is then tested on a different set of data, it may be substantially mis-calibrated. See fig 1.

![](images/62cfcf9129cacf58f11b53965dbb287ff8f273730860266185868c0de1535df6.jpg)  
Figure 2: The result of the spline calibration method, on the example given in fig 1 for top-1 calibration. A recalibration function  $\gamma : \mathbb{R} \to \mathbb{R}$  is used to adjust the scores, replacing  $f_{k}(\mathbf{x})$  with  $\gamma(f_{k}(\mathbf{x}))$  (see section 4.2). As is seen, the network is now almost perfectly calibrated when tested on the "calibration" set (top row) used to calibrate it. In bottom row, the recalibration function is tested on a further set "test". It is seen that the result is not perfect, but much better than the one in fig 1d. It is also notable that the improvement in calibration is achieved without any loss of accuracy.

Our method of calibration is to find a further mapping  $\gamma : [0,1] \to [0,1]$ , such that  $\gamma \circ f_k$  is calibrated. This is easily obtained from the direct mapping from score  $f_k(\mathbf{x})$  to  $P(k \mid f_k(\mathbf{x}))$  (refer to fig 1d). In equations,  $\gamma(\sigma) = h'(s^{-1}(\sigma))$ . The function  $h'$  is known analytically, from fitting a spline to  $h(t)$  and taking its derivative. The function  $s^{-1}$  is a mapping from the given score  $\sigma$  to its fractile  $s^{-1}(\sigma)$ . Note that, a held out calibration set is used to fit the splines and the obtained recalibration function  $\gamma$  is evaluated on an unseen test set.

To this end, given a sample  $\mathbf{x}$  from the test set with  $f_{k}(\mathbf{x}) = \sigma$ , one can compute  $h^{\prime}(s^{-1}(\sigma))$  directly in one step by interpolating its value between the values of  $h^{\prime}(f_k(\mathbf{x}_i))$  and  $h^\prime (f_k(\mathbf{x}_{i + 1}))$  where  $\mathbf{x}_i$  and  $\mathbf{x}_{i + 1}$  are two samples from the calibration set, with closest scores on either side of  $\sigma$ . Assuming the samples in the calibration set are ordered, the samples  $\mathbf{x}_i$  and  $\mathbf{x}_{i + 1}$  can be quickly located using binary search. Given a reasonable number of samples in the calibration set, (usually in the order of thousands), this can be very accurate. In our experiments, improvement in calibration is observed in the test set with no difference to the accuracy of the network (refer to fig 2d). Our code will be published to ensure reproducibility.

# 5 RELATED WORK

Modern calibration methods. In recent years, neural networks are shown to overfit to the Negative Log-Likelihood (NLL) loss and in turn produce overconfident predictions which is cited as the main reason for miscalibration (Guo et al. (2017)). To this end, modern calibration methods can be broadly categorized into 1) methods that adapt the training procedure of the classifier, and 2) methods that learn a recalibration function post training. Among the former, the main idea is to increase the entropy of the classifier to avoid overconfident predictions, which is accomplished via modifying the training loss (Kumar et al. (2018); Mukhoti et al. (2020); Seo et al. (2019)), label smoothing (Müller et al. (2019); Pereyra et al. (2017)), and data augmentation techniques (Thulasidasan et al. (2019); Yun et al. (2019); Zhang et al. (2018)).

Table 1: KS Error (in %) for top-1 prediction (with lowest in bold and second lowest underlined) on various image classification datasets and models with different calibration methods. Note, our method consistently reduces calibration error to  $< 1\%$  in almost all experiments, outperforming state-of-the-art methods.  

<table><tr><td>Dataset</td><td>Model</td><td>Uncalibrated</td><td>Temp. Scaling</td><td>Vector Scaling</td><td>MS-ODIR</td><td>Dir-ODIR</td><td>Ours (Spline)</td></tr><tr><td rowspan="5">CIFAR-10</td><td>Resnet-110</td><td>4.750</td><td>0.916</td><td>0.996</td><td>0.977</td><td>1.060</td><td>0.643</td></tr><tr><td>Resnet-110-SD</td><td>4.102</td><td>0.362</td><td>0.430</td><td>0.358</td><td>0.389</td><td>0.269</td></tr><tr><td>DenseNet-40</td><td>5.493</td><td>0.900</td><td>0.890</td><td>0.897</td><td>1.057</td><td>0.773</td></tr><tr><td>Wide Resnet-32</td><td>4.475</td><td>0.296</td><td>0.267</td><td>0.305</td><td>0.291</td><td>0.367</td></tr><tr><td>Lenet-5</td><td>5.038</td><td>0.799</td><td>0.839</td><td>0.646</td><td>0.854</td><td>0.348</td></tr><tr><td rowspan="5">CIFAR-100</td><td>Resnet-110</td><td>18.481</td><td>1.489</td><td>1.827</td><td>2.845</td><td>2.575</td><td>0.575</td></tr><tr><td>Resnet-110-SD</td><td>15.832</td><td>0.748</td><td>1.303</td><td>3.572</td><td>1.645</td><td>1.028</td></tr><tr><td>DenseNet-40</td><td>21.156</td><td>0.304</td><td>0.483</td><td>2.350</td><td>0.618</td><td>0.454</td></tr><tr><td>Wide Resnet-32</td><td>18.784</td><td>1.130</td><td>1.642</td><td>2.524</td><td>1.788</td><td>0.930</td></tr><tr><td>Lenet-5</td><td>12.117</td><td>1.215</td><td>0.768</td><td>1.047</td><td>2.125</td><td>0.391</td></tr><tr><td rowspan="2">ImageNet</td><td>Densenet-161</td><td>5.721</td><td>0.744</td><td>2.014</td><td>4.723</td><td>3.103</td><td>0.406</td></tr><tr><td>Resnet-152</td><td>6.544</td><td>0.791</td><td>1.985</td><td>5.805</td><td>3.528</td><td>0.441</td></tr><tr><td>SVHN</td><td>Resnet-152-SD</td><td>0.852</td><td>0.552</td><td>0.570</td><td>0.573</td><td>0.607</td><td>0.556</td></tr></table>

On the other hand, we are interested in calibrating an already trained classifier that eliminates the need for training from scratch. In this regard, a popular approach is Platt scaling (Platt et al. (1999)) which transforms the outputs of a binary classifier into probabilities by fitting a scaled logistic function on a held out calibration set. Similar approaches on binary classifiers include Isotonic Regression (Zadrozny & Elkan (2001)), histogram and Bayesian binning (Naeini et al. (2015); Zadrozny & Elkan (2001)), and Beta calibration (Kull et al. (2017)), which are later extended to the multiclass setting (Guo et al. (2017); Kull et al. (2019); Zadrozny & Elkan (2002)). Among these, the most popular method is temperature scaling (Guo et al. (2017)), which learns a single scalar on a held out set to calibrate the network predictions. Despite being simple and one of the early works, temperature scaling is the method to beat in calibrating modern networks. Our approach falls into this category, however, as opposed to minimizing a loss function, we obtain a recalibration function via spline-fitting, which directly maps the classifier outputs to the calibrated probabilities.

Calibration measures. Expected Calibration Error (ECE) (Naeini et al. (2015)) is the most popular measure in the literature, however, it has a weakness that the resulting calibration error depends on the histogram binning scheme such as the bin endpoints and the number of bins. Even though, some improvements have been proposed (Nixon et al. (2019); Vaicenavicius et al. (2019)), the binning scheme has not been eliminated and it is recently shown that any binning scheme leads to underestimated calibration errors (Kumar et al. (2019); Widmann et al. (2019)). Note that, there are binning-free metrics exist such as Brier score (Brier (1950)), NLL, and kernel based metrics for the multiclass setting (Kumar et al. (2018); Widmann et al. (2019)). Nevertheless, the Brier score and NLL measure a combination of calibration error and classification error (not just the calibration which is the focus). Whereas kernel based metrics, besides being computationally expensive, measure the calibration of the predicted probability vector rather than the classwise calibration error (Kull et al. (2019)) (or top-  $r$  prediction) which is typically the quantity of interest. To this end, we introduce a binning-free calibration measure based on the classical KS-test, which has the same benefits of ECE and provides effective visualizations similar to reliability diagrams. Furthermore, KS error can be shown to be a special case of kernel based measures (Gretton et al. (2012)).

# 6 EXPERIMENTS

Experimental setup. We evaluate our proposed calibration method on four different image-classification datasets namely CIFAR-10/100 (Krizhevsky et al. (2009)), SVHN (Netzer et al. (2011)) and ImageNet (Deng et al. (2009)) using LeNet (LeCun et al. (1998)), ResNet (He et al. (2016)), ResNet with stochastic depth (Huang et al. (2017)), Wide ResNet (Zagoruyko & Komodakis (2016)) and DenseNet (Huang et al. (2017)) network architectures against state-of-the-art methods that calibrate post training. We use the pretrained network  $\log_{10}^{3}$  for spline fitting where we choose

Table 2: KS Error (in %) for top-2 prediction (with lowest in bold and second lowest underlined) on various image classification datasets and models with different calibration methods. Again, our method consistently reduces calibration error to  $< 1\%$  (less than  $0.7\%$ , except for one case), in all experiments, the only one of the methods to achieve this.  

<table><tr><td>Dataset</td><td>Model</td><td>Uncalibrated</td><td>Temp. Scaling</td><td>Vector Scaling</td><td>MS-ODIR</td><td>Dir-ODIR</td><td>Ours (Spline)</td></tr><tr><td rowspan="5">CIFAR-10</td><td>Resnet-110</td><td>3.011</td><td>0.947</td><td>0.948</td><td>0.598</td><td>0.953</td><td>0.347</td></tr><tr><td>Resnet-110-SD</td><td>2.716</td><td>0.478</td><td>0.486</td><td>0.401</td><td>0.500</td><td>0.310</td></tr><tr><td>DenseNet-40</td><td>3.342</td><td>0.535</td><td>0.543</td><td>0.598</td><td>0.696</td><td>0.695</td></tr><tr><td>Wide Resnet-32</td><td>2.669</td><td>0.426</td><td>0.369</td><td>0.412</td><td>0.382</td><td>0.364</td></tr><tr><td>Lenet-5</td><td>1.708</td><td>0.367</td><td>0.279</td><td>0.409</td><td>0.426</td><td>0.837</td></tr><tr><td rowspan="5">CIFAR-100</td><td>Resnet-110</td><td>4.731</td><td>1.401</td><td>1.436</td><td>0.961</td><td>1.269</td><td>0.371</td></tr><tr><td>Resnet-110-SD</td><td>3.923</td><td>0.315</td><td>0.481</td><td>0.772</td><td>0.506</td><td>0.595</td></tr><tr><td>DenseNet-40</td><td>5.803</td><td>0.305</td><td>0.653</td><td>0.219</td><td>0.135</td><td>0.903</td></tr><tr><td>Wide Resnet-32</td><td>5.349</td><td>0.790</td><td>1.095</td><td>0.646</td><td>0.845</td><td>0.372</td></tr><tr><td>Lenet-5</td><td>2.615</td><td>0.571</td><td>0.439</td><td>0.324</td><td>0.799</td><td>0.587</td></tr><tr><td rowspan="2">ImageNet</td><td>Densenet-161</td><td>1.689</td><td>1.044</td><td>1.166</td><td>1.288</td><td>1.321</td><td>0.178</td></tr><tr><td>Resnet-152</td><td>1.793</td><td>1.151</td><td>1.264</td><td>1.660</td><td>1.430</td><td>0.580</td></tr><tr><td>SVHN</td><td>Resnet-152-SD</td><td>0.373</td><td>0.226</td><td>0.216</td><td>0.973</td><td>0.218</td><td>0.492</td></tr></table>

validation set as the calibration set, similar to the standard practice. Our final results for calibration are then reported on the test set of all datasets. Since ImageNet does not comprise of the validation set, test set is divided into two halves: calibration set and test set. We use the natural cubic spline fitting method (that is, cubic splines with linear run-out) with 6 knots for all our experiments. Further experimental details are provided in the supplementary. For baseline methods namely: Temperature scaling, Vector scaling, Matrix scaling with ODIR (Off-diagonal and Intercept Regularisation), and Dirichlet calibration, we use the implementation of Kull et al. (Kull et al. (2019)).

Results. We provide comparisons of our method using proposed KS error for the top most prediction against state-of-the-art calibration methods namely temperature scaling (Guo et al. (2017)), vector scaling, MS-ODIR and Dirichlet Calibration (Dir-ODIR) (Kull et al. (2019)) in Table 1. Our method reduces calibration error to  $1\%$  in almost all experiments performed on different datasets without any loss in accuracy. It clearly reflects the efficacy of our method irrespective of the scale of the dataset as well as the depth of the network architecture. It consistently performs better than recently introduced Dirichlet calibration and Matrix scaling with ODIR (Kull et al. (2019)) in all the experiments. The closest competitor to our method is temperature scaling, against which our method performs better in 9 out of 13 experiments. Note, in the cases where temperature scaling outperforms our method, the gap in KS error between the two methods is marginal  $(< 0.3\%)$  and our method is the second best. We provide comparisons using other calibration metrics in the supplementary.

From the practical point of view, it is also important for a network to be calibrated for top second/third predictions and so on. We thus show comparisons for top-2 prediction KS error in Table 2. An observation similar to the one noted in Table 1 can be made for the top-2 predictions as well. Our method achieves  $< 1\%$  calibration error in all the experiments. It consistently performs well especially for experiments performed on large scale ImageNet dataset where it sets new state-of-the-art for calibration. We would like to emphasize here, though for some cases Kull et al. (Kull et al. (2019)) and Vector Scaling perform better than our method in terms of top-2 KS calibration error, overall (considering both top-1 and top-2 predictions) our method performs better.

# 7 CONCLUSION

In this work, we have introduced a binning-free calibration metric based on the Kolmogorov-Smirnov test to measure classwise or (within)-top- $r$  calibration errors. Our KS error eliminates the shortcomings of the popular ECE measure and its variants while accurately measuring the expected calibration error and provides effective visualizations similar to reliability diagrams. Furthermore, we introduced a simple and effective calibration method based on spline-fitting which does not involve any learning and yet consistently yields the lowest calibration error in the majority of our experiments. We believe, the KS metric would be of wide-spread use to measure classwise calibration and our spline method would inspire learning-free approaches to neural network calibration.

# REFERENCES

Glenn W Brier. Verification of forecasts expressed in terms of probability. Monthly weather review, 78(1):1-3, 1950.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248-255. Ieee, 2009.  
Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 2012.  
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1321-1330. JMLR.org, 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4700-4708, 2017.  
A Kolmogorov. Sulla determinazione empírica di una legge di distribuzione. 1933.  
Alex Krizhevsky et al. Learning multiple layers of features from tiny images. 2009.  
Meelis Kull, Telmo Silva Filho, and Peter Flach. Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers. In Artificial Intelligence and Statistics, pp. 623-631, 2017.  
Meelis Kull, Miquel Perello Nieto, Markus Kangsepp, Telmo Silva Filho, Hao Song, and Peter Flach. Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration. In Advances in Neural Information Processing Systems, pp. 12295-12305, 2019.  
Ananya Kumar, Percy S Liang, and Tengyu Ma. Verified uncertainty calibration. In Advances in Neural Information Processing Systems, pp. 3787-3798, 2019.  
Aviral Kumar, Sunita Sarawagi, and Ujjwal Jain. Trainable calibration measures for neural networks from kernel mean embeddings. In International Conference on Machine Learning, pp. 2805-2814, 2018.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Sky McKinley and Megan Levine. Cubic spline interpolation. College of the Redwoods, 1998.  
Jishnu Mukhoti, Viveka Kulharia, Amartya Sanyal, Stuart Golodetz, Philip HS Torr, and Puneet K Dokania. Calibrating deep neural networks using focal loss. arXiv preprint arXiv:2002.09437, 2020.  
Rafael Müller, Simon Kornblith, and Geoffrey E Hinton. When does label smoothing help? In Advances in Neural Information Processing Systems, pp. 4696-4705, 2019.  
Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.  
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. 2011.  
Alexandru Niculescu-Mizil and Rich Caruana. Predicting good probabilities with supervised learning. In Proceedings of the 22nd international conference on Machine learning, 2005.

Jeremy Nixon, Michael W Dusenberry, Linchuan Zhang, Ghassen Jerfel, and Dustin Tran. Measuring calibration in deep learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 38-41, 2019.  
Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, and Geoffrey Hinton. Regularizing neural networks by penalizing confident output distributions. arXiv preprint arXiv:1701.06548, 2017.  
John Platt et al. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3):61-74, 1999.  
Seonguk Seo, Paul Hongsuck Seo, and Bohyung Han. Learning for single-shot confidence calibration in deep neural networks through stochastic inferences. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9030-9038, 2019.  
Nikolai Smirnov. On the estimation of the discrepancy between empirical curves of distribution for two independent samples. 1939.  
Sunil Thulasidasan, Gopinath Chennupati, Jeff A Bilmes, Tanmoy Bhattacharya, and Sarah Michalak. On mixup training: Improved calibration and predictive uncertainty for deep neural networks. In Advances in Neural Information Processing Systems, pp. 13888-13899, 2019.  
Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, and Thomas B Schon. Evaluating model calibration in classification. AISTATS, 2019.  
David Widmann, Fredrik Lindsten, and Dave Zachariah. Calibration tests in multi-class classification: A unifying framework. In Advances in Neural Information Processing Systems, pp. 12236-12246, 2019.  
Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE International Conference on Computer Vision, pp. 6023-6032, 2019.  
Bianca Zadrozny and Charles Elkan. Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers. In Icml, volume 1, pp. 609-616. CiteSeer, 2001.  
Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 694-699, 2002.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.  
Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In 6th International Conference on Learning Representations, ICLR 2018, 2018.
