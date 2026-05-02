# TIGHT SECOND-ORDER CERTIFCATES FOR RANDOMIZED SMOOTHING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Randomized smoothing is a popular way of providing robustness guarantees against adversarial attacks: randomly-smoothed functions have a universal Lipschitz-like bound, allowing for robustness certificates to be easily computed. In this work, we show that there also exists a universal curvature-like bound for Gaussian random smoothing: given the exact value and gradient of a smoothed function, we compute a lower bound on the distance of a point to its closest adversarial example, called the Second-order Smoothing (SoS) robustness certificate. In addition to proving the correctness of this novel certificate, we show that SoS certificates are realizable and therefore tight. Interestingly, we show that the maximum achievable benefits, in terms of certified robustness, from using the additional information of the gradient norm are relatively small: because our bounds are tight, this is a fundamental negative result. The gain of SoS certificates further diminishes if we consider the estimation error of the gradient norms, for which we have developed an estimator. We therefore additionally develop a variant of Gaussian smoothing, called Gaussian dipole smoothing, which provides similar bounds to randomized smoothing with gradient information, but with much-improved sample efficiency. This allows us to achieve (marginally) improved robustness certificates on high-dimensional datasets such as CIFAR-10 and ImageNet.

# 1 INTRODUCTION

A topic of much recent interest in machine learning has been the design of deep classifiers with provable robustness guarantees. In particular, for an  $m$ -class classifier  $h: \mathbb{R}^d \to [m]$ , the  $L_2$  certification problem for an input  $\mathbf{x}$  is to find a radius  $\rho$  such that, for all  $\delta$  with  $\| \delta \|_2 < \rho$ ,  $h(\mathbf{x}) = h(\mathbf{x} + \delta)$ . This robustness certificate serves as a lower bound on the magnitude of any adversarial perturbation of the input that can change the classification: therefore, the certificate is a security guarantee against adversarial attacks.

There are many approaches to the certification problem, including exact methods, which compute the precise norm to the decision boundary (Tjeng et al., 2019; Carlini et al., 2017; Huang et al., 2017) as well as methods for which the certificate  $\rho$  is merely a lower bound on the distance to the decision boundary (Wong & Kolter, 2018; Gowal et al., 2018; Raghunathan et al., 2018).

One approach that belongs to the latter category is Lipschitz function approximation. Recall that a function  $f: \mathbb{R}^d \to \mathbb{R}$  is  $L$ -Lipschitz if, for all  $\mathbf{x}, \mathbf{x}'$ ,  $|f(\mathbf{x}) - f(\mathbf{x}')| \leq L\|\mathbf{x} - \mathbf{x}'\|_2$ . If a classifier is known to be a Lipschitz function, this immediately implies a robustness certificate. In particular, consider a binary classification for simplicity, where we use an  $L$ -Lipschitz function  $f$  as a classifier, using the sign of  $f(\mathbf{x})$  as the classification. Then for any input  $\mathbf{x}$ , we are assured that the classification (i.e., the sign) will remain constant for all  $\mathbf{x}'$  within a radius  $|f(\mathbf{x})| / L$  of  $\mathbf{x}$ .

Numerous methods for training Lipschitz neural networks with small, known Lipschitz constants have been proposed. (Fazlyab et al., 2019; Zhang et al., 2019; Anil et al., 2019; Li et al., 2019b) It is desirable that the network be as expressive as possible, while still maintaining the desired Lipschitz property. Anil et al. (2019) in particular demonstrates that their proposed method can universally approximate Lipschitz functions, given sufficient network complexity. However, in practice, for the robust certification problem on large-scale input, randomized smoothing (Cohen et al., 2019) is the current state-of-the-art method. The key observation of randomized smoothing (as formalized by

![](images/81211922d8a5f25f57436a1aa731a502284be0994288c6755857185eadbb37f2.jpg)  
(a)

![](images/358d618dee99c92470bb71c2e7e4b4a8c71beb3f9afe4467aaaa88bf81d6affb.jpg)  
Figure 1: (a) Tight lower bound on the value of a smoothed function at  $\mathbf{x}'$  (i.e.  $p_a(\mathbf{x}')$ ) as a function of  $\|\mathbf{x}' - \mathbf{x}\|_2$ . In this example,  $p_a(\mathbf{x}) = 0.8$  and the smoothing standard deviation  $\sigma = 1$ . The red line shows the lower bound for the function, with no information about the gradient given. The blue line incorporates the additional information that  $\|\nabla_{\mathbf{x}} p_a(\mathbf{x})\|_2 = 0$ . Note that the axis at  $\Phi^{-1}(p_a(\mathbf{x})) = 0$  corresponds to  $p_a(\mathbf{x}) = 0.5$ , the decision boundary for a binary classifier. (b) Tight robustness certificates for a randomized-smoothed classifier, given the top-class value  $p_a(\mathbf{x})$  and the gradient norm  $\|\nabla_{\mathbf{x}} p_a(\mathbf{x})\|_2$ . The dashed lines show the certificates given  $p_a(\mathbf{x})$  alone. Note that the maximum possible gradient for a smoothed classifier depends on  $p_a(\mathbf{x})$  (see Equation 1).  
(b)

(Salman et al., 2019; Levine et al., 2019)) is that, for any arbitrary base classifier function  $f: \mathbb{R}^d \to [0,1]$ , the function

$$
\mathbf {x} \rightarrow \Phi^ {- 1} \left(p _ {a}\right) \quad \text {w h e r e} \quad p _ {a} (\mathbf {x}) := \underset {\epsilon \sim \mathcal {N} \left(0, \sigma^ {2} I\right)} {\mathbb {E}} f (\mathbf {x} + \epsilon) \tag {1}
$$

is  $(1 / \sigma)$ -Lipschitz, where  $\mathcal{N}(0,\sigma^2 I)$  is a  $d$ -dimensional isometric Gaussian distribution with variance  $\sigma^2$  and  $\Phi^{-1}$  is the inverse normal CDF function. As a result, given the smoothed classifier value  $p_a(\mathbf{x})$  at  $\mathbf{x}$ , one can calculate the certified radius  $\rho (\mathbf{x}) = \sigma \Phi^{-1}(p_a(\mathbf{x}))$  in which  $p_a(\mathbf{x})\geq 0.5$  (i.e.,  $\Phi^{-1}(p_a(\mathbf{x}))\geq 0$ ). This means that we can use  $p_a(\mathbf{x})\in \mathbb{R}^d\to [0,1]$  as a robust binary classifier (with one class assignment if  $p_a(\mathbf{x})\geq 0.5$ , and the other if  $p_a(\mathbf{x}) < 0.5$ ). Cohen et al. (2019) shows that this is a tight certificate result for a classifier smoothed with Gaussian noise: given the value of  $p_a(\mathbf{x})$ , there exists a base classifier function  $f$  such that, if  $p_a$  is the Gaussian-smoothed version of  $f$ , then there exists an  $\mathbf{x}'$  with  $\| \mathbf{x} - \mathbf{x}'\| _2 = \rho$  such that  $p_a(\mathbf{x}') = 0.5$ . In other words, the certificate provided by (Cohen et al., 2019) is the largest possible certificate for Gaussian smoothing, given only the value of  $p_a(\mathbf{x})$ . Previous results (Li et al., 2019a; Lecuyer et al., 2019) provided looser bounds for Gaussian smoothing.

Singla & Feizi (2020) have recently shown, for shallow neural networks, that, rather than globally bounding the (first-order) Lipschitz constant of the network, it possible to achieve larger robustness certificates by instead globally bounding the Lipschitz constant of the gradient of the network. This second-order, curvature-based method takes advantage of the fact that the gradient at  $\mathbf{x}$  can be computed easily via back-propagation, so certificates can make use of both  $f(\mathbf{x})$  and  $\nabla_{\mathbf{x}}f(\mathbf{x})$ .

This leads to a question: can we also use the gradient of a smoothed classifier  $\nabla_{\mathbf{x}}p_a(\mathbf{x})$  to improve smoothing-based certificates? In this work, we show that there is a universal curvature-like bound for all randomly-smoothed classifiers. Therefore, given  $p_a(\mathbf{x})$  and  $\nabla_{\mathbf{x}}p_a(\mathbf{x})$ , we can compute larger certificates than is possible using the value of  $p_a(\mathbf{x})$  alone. Moreover, our bound is tight in that, given only the pair  $(p_a(\mathbf{x}),\nabla_{\mathbf{x}}p_a(\mathbf{x}))$ , the certificate we provide is the largest possible certificate for Gaussian smoothing. We call our certificates "Second-order Smoothing" (SoS) certificates. As shown in Figure 1, the smoothing-based certificates which we can achieve using second-order smoothing represent relatively modest improvements compared to the first-order bounds. This is a meaningful negative result, given the tightness of our bounds, and is therefore useful in guiding (or limiting) future research into higher-order smoothing certificates. Additionally, this result shows that randomized smoothing (or, specifically, functions in the form of Equation 1) can not be used to uni-

versally approximate Lipschitz functions: all randomly smoothed functions will have the additional curvature constraint described in this work.

If the base classifier  $f$  is a neural network, computing the expectation in Equation 1 analytically is not tractable. Therefore it is standard (Lecuyer et al., 2019; Cohen et al., 2019; Salman et al., 2019) to estimate this expectation using  $N$  random samples, and bound the expectation probabilistically. The certificate is then as a high-probability, rather than exact, result, using the estimated lower bound of  $p_a(\mathbf{x})$ . In Section 3.1, we discuss empirical estimation of the gradient norm of a smoothed classifier for second-order certification. This also leads to somewhat negative result: we find that (for an unrestricted base classifier) the number of Gaussian samples required to meaningfully estimate the gradient unavoidably scales with the dimensionality  $d$  of the input. We develop an estimation scheme which tightly achieves this scaling. In order to overcome this, in Section 4, we develop a modified form of Gaussian randomized smoothing, Gaussian Dipole Smoothing, which allows for a dipole certificate, related to the second-order certificate, to be computed. Unlike the second-order certificate, however, the dipole certificate has no explicit dependence of dimensionality in its estimation, and therefore can practically scale to real-world high-dimensional datasets.

# 2 PRELIMINARIES, ASSUMPTIONS AND NOTATION

We use  $f(\mathbf{x})$  to represent a generic scalar-valued "base" function to be smoothed. In general, we assume  $f \in \mathbb{R}^d \to [0,1]$ . However, for empirical estimation results (Theorem 3), we assume that  $f$  is a "hard" base classifier:  $f \in \mathbb{R}^d \to \{0,1\}$ . This will be made clear in context. The smoothed version of  $f$  is notated as  $p_a \in \mathbb{R}^d \to [0,1]$ , defined as in equation 1.

Recall that  $\Phi$  is the normal CDF function and  $\Phi'$  is the normal PDF function. In randomized smoothing for multi-class problems, the base classifier is typically a vector-valued function  $\mathbf{f} \in \mathbb{R}^d \to \{0,1\}^m$ ,  $\sum_c \mathbf{f}_c(\mathbf{x}) = 1$ , where  $m$  is the number of classes. The final classification returned by the smoothed classifier is then given by  $a := \arg \max_c \mathbb{E}_{\epsilon} \mathbf{f}_c(\mathbf{x} + \epsilon)$ . However, in most prominent implementations (Cohen et al., 2019; Salman et al., 2019), certificates are computed using only the smoothed value for the estimated top class  $a$ , where  $a$  is estimated using a small number  $N_0$  of initial random samples, before the final value of  $p_a(\mathbf{x})$  is computed using  $N$  samples. The certificate then determines the radius in which  $p_a(\mathbf{x}')$  will remain above 0.5: this guarantees that  $a$  will remain the top class, regardless of the other logits. While some works (Lecuyer et al., 2019; Feng et al., 2020) independently estimate each smoothed logit, this incurs additional estimation error as the number of classes increases. In this work, we assume that only estimates for the top-class smoothed logit  $p_a(\mathbf{x})$  and its gradient  $\nabla_{\mathbf{x}} p_a(\mathbf{x})$  are available (although we briefly discuss the case with more estimated logits in Section 3.2). When discussing empirical estimation, we use  $\eta$  as the accepted probability of failure of an estimation method.

# 3 SECOND-ORDER SMOOTHING CERTIFICATE

We now state our main second-order robustness certificate result:

Theorem 1. For all  $\mathbf{x},\mathbf{x}^{\prime}$  with  $\| \mathbf{x} - \mathbf{x}^{\prime}\|_{2} < \rho$  and for all  $f:\mathbb{R}^{d}\to [0,1]$ ,

$$
p _ {a} \left(\mathbf {x} ^ {\prime}\right) \geq \Phi \left(\Phi^ {- 1} \left(a ^ {\prime} + p _ {a} (\mathbf {x})\right) - \frac {\rho}{\sigma}\right) - \Phi \left(\Phi^ {- 1} \left(a ^ {\prime}\right) - \frac {\rho}{\sigma}\right) \tag {2}
$$

where  $a^\prime$  is the (unique) solution to

$$
\Phi^ {\prime} \left(\Phi^ {- 1} \left(a ^ {\prime}\right)\right) - \Phi^ {\prime} \left(\Phi^ {- 1} \left(a ^ {\prime} + p _ {a} (\mathbf {x})\right)\right) = - \sigma \| \nabla_ {\mathbf {x}} p _ {a} (\mathbf {x}) \| _ {2}. \tag {3}
$$

Further, for all pairs  $(p_a(\mathbf{x}),\| \nabla_{\mathbf{x}}p_a(\mathbf{x})\| _2)$  which are possible, there exists a base classifier  $f$  and an adversarial point  $\mathbf{x}^{\prime}$  such that Equation 4 is an equality. This implies that our certificate is realizable, and therefore tight.

Note that the right-hand side of Equation 2 is monotonically decreasing with  $\rho$ : we can then compute a robustness certificate by simply setting  $p_a(\mathbf{x}') = 0.5$  and solving for the certified radius  $\rho$ . Also,  $a'$  can be computed easily, because the left-hand side of Equation 3 is monotonic in  $a'$ . Evaluated certificate values are shown in Figure 1-b, and compared with first-order certificates.

![](images/2bc203b2be84dec8b6146042875cad5031a994a4ccc5444c2ab1894c35b613e5.jpg)  
(a)  
Figure 3: Worst case base classifiers for second-order smoothing for the same value of  $p_a(\mathbf{x})$  at different values of  $\| \nabla_{\mathbf{x}}p_a(\mathbf{x})\| _2$ . The base classifier is  $f = 1$  in the blue regions and  $f = 0$  in the red regions. The point  $\mathbf{x}$  is shown as a blue dot, with the Gaussian sampled region used for calculating  $p_a(\mathbf{x})$  is approximately shown as a dashed blue circle.  $\nabla_{\mathbf{x}}p_a(\mathbf{x})$  is shown as a blue arrow. (a) The gradient takes its maximum possible value:  $\| \nabla_{\mathbf{x}}p_a(\mathbf{x})\| _2 = \sigma^{-1}\Phi '(P_{a}^{-1}(p_a(\mathbf{x}))$ . (b) The gradient has an intermediate value:  $0 < \| \nabla_{\mathbf{x}}p_a(\mathbf{x})\| _2 < \sigma^{-1}\Phi '(P_{a}^{-1}(p_a(\mathbf{x}))$ . (c) The gradient is zero:  $\| \nabla_{\mathbf{x}}p_a(\mathbf{x})\| _2 = 0$ .

![](images/e3cdbb8eb367fa9e8edf2cc81c7e27478e64da411c00e7a69df3097fdec66525.jpg)  
(b)

![](images/eefcb40669cce87515cd7da8efe290460864477cc53bfcb1aa4fb7ce0482dce6.jpg)  
(c)

All proofs are presented in Appendix A. Like in Cohen et al. (2019), we proceed by constructing the worst-case base classifier  $f$  given  $p_{a}(\mathbf{x})$  and  $\| \nabla_{\mathbf{x}}p_{a}(\mathbf{x})\|_{2}$ . This is the base classifier  $f$  which creates an adversarial point to the smoothed classifier as close as possible to  $\mathbf{x}$ , given the constraints that  $p_{a}(\mathbf{x})$  and  $\| \nabla p_{a}(\mathbf{x})\|_{2}$  are equal to their reported values. In Cohen et al. (2019), given only  $p_{a}(\mathbf{x})$ , this is simply a linear classifier. With the gradient norm, the worst case is that  $\mathbf{x}$  lies in a region with class  $a$  which is a slice between two linear decision boundaries, both perpendicular to  $\nabla p_{a}(\mathbf{x})$ . See Figure 3. Note that, by isometry and because  $\nabla p_{a}(\mathbf{x})$  is the only vector information we have, there is no benefit in certified radius to having the direction of  $\nabla p_{a}(\mathbf{x})$ : the norm is sufficient. In the case of a linear classifier the gradient takes its maximum possible value:  $\| \nabla_{\mathbf{x}}p_{a}(\mathbf{x})\|_{2} = \sigma^{-1}\Phi^{\prime}(\Phi^{-1}(p_{a}(\mathbf{x}))$ . This case is shown in Figure 3-a: if the gradient norm is equal to this value, the second-order certificate is identical to the first-order certificate (Cohen et al., 2019). However, if the gradient

![](images/d930a49e9a111867719a328470a7823fb58c57eaeae55a0fe24969692b04a2f0.jpg)  
Figure 2: Comparison of second-order smoothing certificates to standard Gaussian smoothing certificates on a selection of points from the Swiss Roll dataset. Correctly labeled points with (second-order) certificates are shown in light red and blue, and points with incorrect label or no certificate are in black. For a selection of points, shown in red/blue, the first-order certified radii shown are as red/blue rings. Increases to certified radii due to second-order smoothing shown are as light blue (light red, absent) rings around certificate radii. For both experiments,  $N = 10^{8}$ , and  $\eta = 0.001$ .

norm is smaller, then we cannot be in this worst-case linear-classifier scenario. Instead, the new "worst case" is constructed by introducing a second "wrong class" region opposite to the direction of the adversarial point (Figure 3-b). In the extreme case (Figure 3-c) where the gradient norm is zero, this is accomplished by balancing two adversarial regions in a "sandwich" around  $\mathbf{x}$ .

This "sandwich" configuration reveals the relative weakness of gradient information in improving robustness certificates: having zero gradient does not require that the adversarial regions be evenly distributed around  $\mathbf{x}$ . Rather, it is sufficient to distribute the adversarial probability mass  $1 - p_{a}(\mathbf{x})$  into just two adversarial regions. Therefore, the certified radius, even in this most extreme case, is similar to the Cohen et al. (2019) certificate in the case with half as much adversarial probability mass (the first-order certificate for  $p_{a}(\mathbf{x}) \coloneqq (1 + p_{a}(\mathbf{x})) / 2$ ). This can be seen in Figure 1-b: note that at  $p_{a}(\mathbf{x}) = 0.6$ , if the gradient norm is known to be zero, the certificate is slightly below the certificate for  $p_{a}(\mathbf{x}) = 0.8$  with no gradient information. The second-order cer-

tificate at  $p_a(\mathbf{x}) = 0.6$ ,  $\| nabla a_{\mathbf{x}}p_{a}(\mathbf{x})\|_{2} = 0$  is in fact slightly below the first-order certificate for  $p_a(\mathbf{x}) = 0.8$ , because the Gaussian noise samples throughout all of space, so the smoothed classifier decision boundary is slightly affected by the adversarial region in the opposite direction of  $\mathbf{x}$ .

Because we can explicitly construct "worst-case" classifiers which represent the equality case of Equation 2, our certificates are known to be tight: the reported certified radii are the largest possible certificates, if only  $p_a(\mathbf{x})$  and  $\| \nabla p_a(\mathbf{x}) \|_2$  are known.

In Figure 2, we show how our second-order certificate behaves on a simple, two-dimensional, nonlinearly separable dataset, the classic Swiss Roll. The increases are marginal, mostly because the certificates using standard randomized smoothing are already fairly tight. On these data, the certified radii for the two classes are nearly touching in many places along the decision boundary. However, for the blue class, which is surrounded on multiple sides by the red class, there are noticeable increases in the certified radius. This is especially true for points near the center of the blue class, which are at the "top of the hill" of the blue class probability, and therefore have smaller gradient.

# 3.1 GRADIENT NORM ESTIMATION

In order to use the second-order certificate in practice, we must first bound, with high-probability, the gradient norm  $\| \nabla_{\mathbf{x}}p_a(\mathbf{x})\| _2$  using samples from the base classifier  $f$ . Because Theorem 1 provides certificates that are strictly decreasing with  $\| \nabla_{\mathbf{x}}p_a(\mathbf{x})\| _2$ , it is only necessary to lower bound  $\| \nabla_{\mathbf{x}}p_a(\mathbf{x})\| _2$  with high probability.

Salman et al. (2019) suggests two ways of approximate the gradient vector  $\nabla_{\mathbf{x}}p_a(\mathbf{x})$  itself, both based on the following important observation:

$$
\nabla_ {\mathbf {x}} p _ {a} (\mathbf {x}) = \underset {\epsilon \sim \mathcal {N} (0, \sigma^ {2} I)} {\mathbb {E}} [ \nabla_ {\mathbf {x}} f (\mathbf {x} + \epsilon) ] = \underset {\epsilon \sim \mathcal {N} (0, \sigma^ {2} I)} {\mathbb {E}} [ \epsilon f (\mathbf {x} + \epsilon) ] / \sigma^ {2} \tag {4}
$$

These two methods are:

1. At each sampled point, one can measure the gradient of  $f$  using back-propagation, and take the mean vector of these estimates.  
2. At each sampled point, one can multiply  $f(\mathbf{x} + \epsilon)$  by the noise vector  $\epsilon$ , and take the mean vector of these estimates.

Note, however, that Salman et al. (2019) does not provide statistical bounds on these estimates: for our certificate application, we must do so. While we ultimately use an approach based on method 2, we will first briefly discuss method 1. The major obstacle to using method 1 is that it requires that the base classifier  $f$  itself to be a Lipschitz function, with a small Lipschitz constant. This can be understood from Markov's inequality. For example, consider the value of some component  $z(\mathbf{x}) \coloneqq \mathbf{u} \cdot \nabla f(\mathbf{x})$ , where  $\mathbf{u}$  is an arbitrary vector. Suppose  $N$  samples are taken, but that  $z$  is distributed such that:

$$
z (\mathbf {x} + \epsilon) = \left\{ \begin{array}{l l} 0 & \text {w i t h p r o b a b i l i t y} 1 - \frac {1}{2 N} \\ 2 N & \text {w i t h p r o b a b i l i t y} \frac {1}{2 N} \end{array} \right. \tag {5}
$$

This would be the case if  $f$  is a function that approximates a step function from 0 to 1, with a small buffer region of very high slope, for example. Note that the probability that any of the  $N$  samples measures the nonzero gradient component is  $< 0.5$ , but the expected value of this component is in fact 1.0. This example shows that, in order to accurately estimate the gradient with high probability, the number of samples used must at least scale linearly with the maximum possible value of the gradient norm for  $f$ . For un-restricted deep neural networks, Lipschitz constants are NP-hard to estimate, and upper bounds on it are typically very large (Virmaux & Scaman, 2018). Of course, we could use Lipschitz-constrained networks as described in Section 1 for the base classifier, but this would defeat the purpose of using randomized smoothing in the first place. Moreover, in standard "hard" randomized smoothing as typically implemented (Cohen et al., 2019; Salman et al., 2019), the range of  $f$  is  $\{0,1\}$ , so  $f$  is non-differentiable: therefore, this back-propagation method can not be used at all.

We therefore use method 2. This also has a fundamental limitation: consider the case where the gradient is zero, and the base classifier  $f(\mathbf{x}) = c$  everywhere, for some constant  $c$  (if considering

a "hard" base classifier, equivalently consider the base classifier as very rapidly oscillating between zero and one along some dimension). In this case, the  $N$  observed values of  $\epsilon f(\mathbf{x} + \epsilon)$  will each simply follow a  $d$ -dimensional normal distribution. In each dimension, the sample mean vector will then follow a Gaussian distribution with mean zero and variance  $c^2\sigma^2 / N$ . However, the norm-squared of the final estimated gradient, as a sum of the squares of each of these Gaussians, will be distributed by a  $\chi^2$  distribution with  $d$  degrees of freedom: the mean will concentrate at  $\frac{c^2\sigma^2d}{N}$ . Because the actual norm-squared of the mean of the samples will concentrate at a value proportional to  $d / N$ , the best we can hope for is an estimation method where the number of samples we require is proportional to  $d$ .

In fact, we are able to achieve this. In particular, we reject the naive approach of estimating each component independently, taking a union bound, and the taking the norm: not only would the error in the norm-squared scale with  $d$  as the error from each component accumulates, but there would be an additional dependence on  $d$  from the union bound: each component would have to be bounded with failure probability  $\eta / d$ , where  $\eta$  is the total failure probability for measuring the gradient norm. Note that this issue will also be encountered in method 1 above, but in that case, a loose upper bound could at least be achieved without this dependency using Jensen's inequality (the mean of the norms of the gradient is larger than the norm of the mean).

Instead, we estimate the norm-squared of the mean using a single, unbiased estimator. Note that:

$$
\| \nabla_ {\mathbf {x}} \mathbb {E} _ {\epsilon} [ f (\mathbf {x} + \epsilon) ] \| _ {2} ^ {2} = \sigma^ {- 4} \mathbb {E} _ {\epsilon} [ \epsilon f (\mathbf {x} + \epsilon) ] \cdot \mathbb {E} _ {\epsilon} [ \epsilon f (\mathbf {x} + \epsilon) ] =
$$

$$
\sigma^ {- 4} \mathbb {E} _ {\epsilon} [ \epsilon f (\mathbf {x} + \epsilon) ] \cdot \mathbb {E} _ {\epsilon^ {\prime}} [ \epsilon^ {\prime} f (\mathbf {x} + \epsilon^ {\prime}) ] = \tag {6}
$$

$$
\sigma^ {- 4} \mathbb {E} _ {\epsilon , \epsilon^ {\prime}} [ (\epsilon f (\mathbf {x} + \epsilon)) \cdot (\epsilon^ {\prime} f (\mathbf {x} + \epsilon^ {\prime})) ]
$$

In other words, we can estimate the norm-squared of the mean by taking pairs of smoothing samples, and taking the dot product of the noise vectors times the product of the sampled values. We show that this is a subexponential random variable (see Appendix), which gives us the asymptotically linear scaling of  $N$  with  $d$ , as desired:

Theorem 2. Let  $V \coloneqq \mathbb{E}_{\epsilon, \epsilon'}[(\epsilon f(\mathbf{x} + \epsilon)) \cdot (\epsilon' f(\mathbf{x} + \epsilon'))]$ , and  $\tilde{V}$  be its empirical estimate. If  $n$  pairs of samples  $(= N/2)$  are used to estimate  $V$ , then, with probability at most  $\eta$ ,  $\mathbb{E}[V] - \tilde{V} \geq t$ , where:

$$
t = \left\{ \begin{array}{l l} 4 \sigma^ {2} \sqrt {- \frac {d}{n} \ln (\eta)} & i f - 2 \ln (\eta) \leq d n \\ - \frac {4 \sqrt {2} \sigma^ {2}}{n} \ln (\eta) & i f - 2 \ln (\eta) > d n \end{array} \right. \tag {7}
$$

Note that in practice, we can use the same samples to estimate  $\| \nabla_{\mathbf{x}}p_a(\mathbf{x})\| _2$  as are used to estimate  $p_a(\mathbf{x})$ . However, this requires reducing the failure probability of each estimate to  $\eta^{\prime} = \eta /2$ , in order to use a union bound. This means that, if  $N$  is small (or  $d$  large), second-order smoothing can in fact give worse certificates than standard smoothing, because the benefit of a (loose, for  $N$  small) estimate of the gradient is less significant than the negative effect of reducing the estimate of  $p_a(\mathbf{x})$ . As shown in Figure 4-a, even for very large  $N$  and relatively small dimension, the empirical estimation significantly reduces the radii of certificates which can be calculated. See Section 5 for experimental results.

# 3.2 UPPER-BOUND AND MULTI-CLASS CERTIFICATES

We can easily convert Theorem 1 into a tight upper bound on  $p_a(\mathbf{x}')$  by simply evaluating it for  $f' = 1 - f$  (and therefore  $p_a' = 1 - p_a$ ). If estimates and gradients are available for multiple classes, it would then be possible to achieve an even larger certificate, by setting the lower bound of the top logit equal to the upper bounds of each of the other logits. Note, however, that unlike first-order smoothing works (Lecuyer et al., 2019; Feng et al., 2020) which use this approach, it is not sufficient to compare against just the "runner-up" class, because other logits may have less restrictive upper-bounds due to having larger gradients. As discussed above, gradient norm estimation can be computationally expensive, so gradient estimation for many classes may not be feasible. Also, note that while this approach would produce larger, correct certificates, we do not claim that these would be tight certificates given the value and gradient information for all classes: the "worst case" constructions we describe above for a single logit might not be simultaneously construct-able for multiple logits.

![](images/ef9f110cf63b228a674bdebfcc370ab4154efbf0474a3c013cb8e612c7cbbfbc.jpg)  
(a)  
(b)

![](images/45ac6790f1b555ff357d17b6ecb1898882ba2e21e2df112488f5c290bda2a76f.jpg)  
Figure 4: (a) Empirical second-order smoothing certificates, with  $d = 49$  (corresponding to  $7 \times 7$  MNIST experiments),  $N = 10^8$ , and  $\eta = .001$  (b) Worst case classifier for dipole smoothing.

# 4 DIPOLE SMOOTHING

For large-scale image datasets, the dependence on  $d$  in Theorem 2 can create statistical barriers. However, the general approach of second-order smoothing, especially using the discrete estimation method (method 2) described above, has an interesting interpretation: rather than using simply the mean of  $f(\mathbf{x} + \epsilon)$ , we are also using the geometrical distribution of the values of  $f(\mathbf{x} + \epsilon)$  in space to compute a larger certified bound. In particular, if we can show that points which are adversarial for the base classifier (points with  $f(\mathbf{x} + \epsilon) = 0$ ) are dispersed, then this will imply larger certificates, because it makes it impossible for a perturbation in a single direction to move  $\mathbf{x}$  towards the adversarial region. Second-order smoothing, above, is merely an example of this.

We therefore introduce Gaussian Dipole smoothing. This is a method which, like second-order smoothing, also harnesses the geometrical distribution of the values of  $f(\mathbf{x})$  to improve certificates. However, unlike second-order smoothing, there is no explicit dependence on  $d$  in the empirical dipole smoothing bound. In this method, when we sample  $f(\mathbf{x} + \epsilon)$  when estimating  $p_a(\mathbf{x})$ , we also sample  $f(\mathbf{x} - \epsilon)$ . This allows us to compute two quantities:

$$
C ^ {S} := \mathbb {E} _ {\epsilon} [ f (\mathbf {x} + \epsilon) f (\mathbf {x} - \epsilon) ]
$$

$$
C ^ {N} := \mathbb {E} _ {\epsilon} [ f (\mathbf {x} + \epsilon) - f (\mathbf {x} + \epsilon) f (\mathbf {x} - \epsilon) ] \tag {8}
$$

The certificate we can calculate is then as follows:

Theorem 3. For all  $\pmb{x}, \pmb{x}'$  with  $\| \pmb{x} - \pmb{x}' \|_2 < \rho$ , and for all  $f: \mathbb{R}^d \to [0,1]$ ,

$$
p _ {a} \left(\mathbf {x} ^ {\prime}\right) \geq \Phi \left(\Phi^ {- 1} \left(C ^ {N}\right) - \frac {\rho}{\sigma}\right) + \Phi \left(\Phi^ {- 1} \left(\frac {1 + C ^ {S}}{2}\right) - \frac {\rho}{\sigma}\right) - \Phi \left(\Phi^ {- 1} \left(\frac {1 - C ^ {S}}{2}\right) - \frac {\rho}{\sigma}\right) \tag {9}
$$

We also compute this bound by constructing the worst possible classifier. In this case, the trick is that, if two adversarial sampled points are opposite one another (i.e.,  $f(\mathbf{x} + \epsilon) = f(\mathbf{x} - \epsilon) = 0$ ) then they cannot both contribute to the same adversarial "direction". In the worst case, the "reflected" adversarial points form a plane opposite the base classifier decision boundary (See Figure 4-b). In the extreme case where  $C^N = 0$ , the "worst case" classifier is the same as for second-order smoothing.

Experimentally, we simply need to lower-bound both  $C^S$  and  $C^N$  from samples. This reduces the precision of our estimates, for two reasons: we have half as many independent samples for the same number of evaluations we must perform, and we are bounding two quantities, which requires halving the error probability for each. However, unlike second-order smoothing, there is no dependence on  $d$ : this allows for practical certificates of real-world datasets.

![](images/40dbbe8ca2d078a1de6e5db79d68fe2e145fe9cf23f5980c31b91ea81c3c3abc.jpg)

![](images/20ca3b3985d2762ccdd36e24f1a86eaeb92b8eff3217339d50ffb54e2f8ac333.jpg)

![](images/2bc22b5038f108749b914b1e678810da6371d6d7d55232bc08bc819bb6f65f35.jpg)  
Figure 5: (a-d): Experiments on  $7 \times 7$  MNIST. Reported is the distribution of the improvement (or reduction) of higher-order certificates from certificates computed using standard (first-order) randomized smoothing, for each tested image. For all,  $\sigma = 0.25$ . For (a,c), Second-order Smoothing is used. For (b,d), Gaussian dipole smoothing is used. For (a,b),  $N = 10^6$ . For (c,d),  $N = 10^8$ . (e-f): Dipole smoothing experiments, with  $N = 10^6$ ,  $\sigma = 0.25$ , on CIFAR-10 and ImageNet.

![](images/69c4f5b62cafb560805c398a36e9a025db934eb41eeb2ac9ebb0d2783e04c712.jpg)

![](images/dded6980da66d6cb2a19ac7549391bba755087a2e0e587ad848549e1a3e1013e.jpg)

![](images/990b2220dcd47e79def1d19dc3c893584bad338b821498218acd65811d51735e.jpg)

# 5 EXPERIMENTS

Experimental results are presented in Figure 5, with further results in Appendix A. Because both dipole and second-order certificates reduce the precision with which empirical quantities needed for certification can be estimated, but both provide strictly larger certificates at the population level, the key question becomes at what number of samples  $N$  does each higher-order method become beneficial. Note that in the figures, we are comparing the new methods to standard smoothing, using the same  $N$  for standard smoothing as for the new method. Due to the poor scaling of second-order certificates with dimension, we tested second-order smoothing on a low-dimensional dataset,  $7 \times 7$  MNIST. However, significant increases to certificates were not seen until  $N = 10^8$  even on this dataset. By contrast, dipole smoothing is beneficial for many samples even at  $N = 10^6$ . Because it scales to higher-dimensional data, we also tested Gaussian dipole smoothing on CIFAR-10 and ImageNet, where it leads to modest improvements in certificates.

# 6 CONCLUSION

In this work, we explored the limits of using gradient information to improve randomized smoothing certificates. In particular, we introduced second-order smoothing certificates and showed tight and realizable upper bounds on their maximum achievable benefits. We also proposed Gaussian dipole smoothing, a novel method for robustness certification, which can improve smoothing-based robustness certificates even on large-scale data sets. This introduces a broader question for future work: what other information about the spacial distribution of classes in randomized smoothing can be efficiently used to improve robustness certificates?

# REFERENCES

Cem Anil, James Lucas, and Roger Grosse. Sorting out Lipschitz function approximation. volume 97 of Proceedings of Machine Learning Research, pp. 291-301, Long Beach, California, USA, 09-15 Jun 2019. PMLR. URL http://proceedings.mlr.press/v97/anil19a.html.  
Nicholas Carlini, Guy Katz, Clark Barrett, and David L Dill. Provably minimally-distorted adversarial examples. arXiv preprint arXiv:1709.10207, 2017.  
Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 1310-1320, Long Beach, California, USA, 09-15 Jun 2019. PMLR. URL http://proceedings.mlr.press/v97/cohen19c.html.  
Mahyar Fazlyab, Alexander Robey, Hamed Hassani, Manfred Morari, and George Pappas. Efficient and accurate estimation of lipschitz constants for deep neural networks. In Advances in Neural Information Processing Systems, pp. 11427-11438, 2019.  
Huijie Feng, Chunpeng Wu, Guoyang Chen, Weifeng Zhang, and Yang Ning. Regularized training and tight certification for randomized smoothed classifier with provable robustness. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pp. 3858-3865. AAAI Press, 2020. URL https://aaaai.org/ojs/index.php/AAAI/article/view/5798.  
Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Uesato, Relja Arandjelovic, Timothy Mann, and Pushmeet Kohli. On the effectiveness of interval bound propagation for training verifiably robust models. arXiv preprint arXiv:1810.12715, 2018.  
Xiaowei Huang, Marta Kwiatkowska, Sen Wang, and Min Wu. Safety verification of deep neural networks. In International Conference on Computer Aided Verification, pp. 3-29. Springer, 2017.  
Mathias Lecuyer, Vaggelis Atlidakis, Roxana Geambasu, Daniel Hsu, and Suman Jana. Certified robustness to adversarial examples with differential privacy. In 2019 IEEE Symposium on Security and Privacy (SP), pp. 656-672. IEEE, 2019.  
Alexander Levine, Sahil Singla, and Soheil Feizi. Certifiably robust interpretation in deep learning. CoRR, abs/1905.12105, 2019. URL http://arxiv.org/abs/1905.12105.  
Bai Li, Changyou Chen, Wenlin Wang, and Lawrence Carin. Certified adversarial robustness with additive noise. In Advances in Neural Information Processing Systems, pp. 9464-9474, 2019a.  
Qiyang Li, Saminul Haque, Cem Anil, James Lucas, Roger B Grosse, and Jorn-Henrik Jacobsen. Preventing gradient attenuation in lipschitz constrained convolutional networks. In Advances in neural information processing systems, pp. 15390-15402, 2019b.  
Aditi Raghunathan, Jacob Steinhardt, and Percy S Liang. Semidefinite relaxations for certifying robustness to adversarial examples. In Advances in Neural Information Processing Systems, pp. 10877-10887, 2018.  
Hadi Salman, Jerry Li, Ilya Razenshteyn, Pengchuan Zhang, Huan Zhang, Sebastien Bubeck, and Greg Yang. Provably robust deep learning via adversarially trained smoothed classifiers. In Advances in Neural Information Processing Systems, pp. 11292-11303, 2019.  
Sahil Singla and Soheil Feizi. Second-order provable defenses against adversarial attacks. In Proceedings of the 37th International Conference on Machine Learning (preproceedings), 2020. URL https://proceedings.icml.cc/static/paper_files/icml/2020/2933-Paper.pdf.

Vincent Tjeng, Kai Y. Xiao, and Russ Tedrake. Evaluating robustness of neural networks with mixed integer programming. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=HyGidiRqtm.  
Roman Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.  
Aladin Virmaux and Kevin Scaman. Lipschitz regularity of deep neural networks: analysis and efficient estimation. In Advances in Neural Information Processing Systems, pp. 3835-3844, 2018.  
Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge University Press, 2019.  
Eric Wong and Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. In International Conference on Machine Learning, pp. 5286-5295. PMLR, 2018.  
Huan Zhang, Pengchuan Zhang, and Cho-Jui Hsieh. Recurjac: An efficient recursive algorithm for bounding jacobian matrix of neural networks and its applications. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 5757-5764, 2019.
