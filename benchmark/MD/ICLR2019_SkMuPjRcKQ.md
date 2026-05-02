# FEED-FORWARD PROPAGATION IN PROBABILISTIC NEURAL NETWORKS WITH CATEGORICAL AND MAX LAYERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Probabilistic Neural Networks take into account various sources of stochasticity: input noise, dropout, stochastic neurons, parameter uncertainties modeled as random variables. In this paper we revisit the feed-forward propagation method that allows one to estimate for each neuron its mean and variance w.r.t. mentioned sources of stochasticity. In contrast, standard NNs propagate only point estimates, discarding the uncertainty. Methods propagating also the variance have been proposed by several authors in different context. The presented view attempts to clarify the assumptions and derivation behind such methods, relate it to classical NNs and broaden the scope of its applicability. The main technical innovations are new posterior approximations for argmax and max-related transforms, that allows for applicability in networks with softmax and max-pooling layers as well as leaky ReLU activations. We evaluate the accuracy of the approximation and suggest a simple calibration. Applying the method to networks with dropout allows for faster training and gives improved test likelihoods without the need of sampling.

# 1 INTRODUCTION

Despite the massive success of Neural Networks (NNs) considered as deterministic predictors, there are many scenarios where a probabilistic treatment is highly desirable. One of the best known techniques to improve the generalization is dropout (Srivastava et al., 2014), which introduces multiplicative Bernoulli noise in the network. At test time, however, it is commonly approximated by substituting the mean value of the noise variables. Computing the expectation by Monte Carlo (MC) sampling instead leads to improved test likelihood and accuracy (Srivastava et al., 2014; Gal & Ghahramani, 2015) but is computationally expensive. A challenging problem in NNs is the sensitivity of the output to the perturbations of the input, in particular random and adversarial perturbations (Moosavi-Dezfooli et al., 2017; Fawzi et al., 2016; Rodner et al., 2016). In Fig. 1 we illustrate the point that the average of the network output under noisy input differs from propagating the clean input. It is therefore desirable to estimate the output uncertainty resulting from the uncertainty of the input. In classification networks, propagating the uncertainty of the input can impact the confidence of the classifier and its robustness as shown by Astudillo & da Silva Neto (2011). Ideally, we would like that a classifier is not overconfident when making errors, however such high confidences of wrong predictions are typically observed in NNs. Similarly, when predicting real values (e.g. optical flow estimation), it is desirable to estimate also confidences of such predictions. Taking into account uncertainties from input or dropout allows to predict output uncertainties well correlated with the test error (Kendall & Gal, 2017; Gast & Roth, 2018; Schoenholz et al., 2016). Another important problem is overfitting. Bayesian learning is a sound way of dealing with a finite training set: the parameters are considered as random variables and are determined up to an uncertainty implied by the training data. This uncertainty needs then to be propagated to predictions at the test-time.

The above scenarios motivate considering NNs with different sources of stochasticity as Bayesian networks, a class of directed probabilistic graphical models. We focus on the inference problem that consists in estimating the probability of hidden units and the outputs given the network input. While there exist elaborate inference methods for Bayesian networks (variational, mean field, Gibbs

![](images/8182dd0fb53de658ce9ad68bf7f8c8333a2ea02c4bdc856a6202f820564a013e.jpg)  
Figure 1: Illustrative example of propagating an input perturbed with Gaussian noise  $\mathcal{N}(0,0.1)$  through a fully trained LeNet. When the same image is perturbed with different samples of noise, we observe on the output empirical distributions shown as Monte Carlo (MC) histograms. Propagating the clean image results in the estimate denoted AP1 which may be away from the MC mean. Propagating means and variances results in a posterior Gaussian distribution denoted AP2. For the final class probabilities we approximate the expected value of the softmax. The methods AP1 and AP2 are formally defined in § 2. A quantitative evaluation of this experiment is given in § 5.

sampling, etc.), they are computationally demanding and can hardly be applied at the same scale as state-of-the-art NNs.

Contribution and Related Work We revisit feed-forward propagation methods that perform an approximate inference analytically by propagating means and variances of neurons through all layers of a NN, ensuring computational efficiency and differentiability. This type of propagation has been proposed by several authors under different names: uncertainty propagation (Astudillo & da Silva Neto, 2011) in a very limited setting with no learning, fast dropout training (Wang & Manning, 2013), probabilistic backpropagation (Hernández-Lobato & Adams, 2015) in the context of Bayesian learning, assumed density filtering Gast & Roth (2018). Perhaps the most general form is considered by Wang et al. (2016) and termed natural parameter networks. The local reparametrization trick (Kingma et al., 2015) can be viewed as application of the variance propagation method through one layer only and then sampling from the approximate posterior.

In these preceding works, for propagation through softmax, sampling or point-wise estimates were used while max-pooling was avoided. Ghosh et al. (2016) proposed an analytic approximation for softmax using two inequalities, but resorted to sampling noting that the approximation was not accurate. Gast & Roth (2018) introduced Dirichlet posterior to overcome the difficulty with softmax, however, the softmax is still used in the model internally. Furthermore, typically used expressions for ReLU activations involve differences of error functions and may be unstable.

We propose a latent variable view of probabilistic NNs that links them closer to their deterministic counterparts and allows us to develop better approximations. Our technical contribution includes the development of numerically suitable approximations for propagating means and variances through "multivariate" activation functions such as softmax for categorical variables and other max-related non-linearities: max-pooling and leaky ReLU. This makes the whole framework practically operational and applicable to a wider class of problems.

Experimentally, we verify the accuracy of the proposed propagation in approximating the true posterior and compare it to the standard propagation by NN, which has not been questioned before. This verification shows that the proposed scheme has better accuracy than standard propagation in all tested scenarios. We further demonstrate its potential utility in the end-to-end learning with dropout.

# 2 PROBABILISTIC NNS AND FEED-FORWARD EXPECTATION PROPAGATION

In probabilistic NNs, all units are considered to be random variables. In a typical network, units are organized by layers. There are  $l$  layers of hidden random vectors  $X^k$ ,  $k = 1, \ldots, l$  and  $X^0$  is the input layer. Each vector  $X^k$  has  $n_k$  components (layer units) denoted  $X_i^k$ . The network is modeled as a conditional Bayesian network (aka belief network, Neal (1992)) defined by the pdf

$$
p \left(X ^ {1, \dots l} \mid X ^ {0}\right) = \prod_ {k = 1} ^ {l} p \left(X ^ {k} \mid X ^ {k - 1}\right). \tag {1}
$$

We further assume that the conditional distribution  $p(X^k \mid X^{k-1})$  factorizes and depends on a linear combination of the random vector  $X^{k-1}$ ,  $p(X^k \mid X^{k-1}) = \prod_{i=1}^{n_k} p(X_i^k \mid A_i^k)$ , where  $A_i^k = (W^k X^{k-1})_i$  are activations. We will denote values of r.v.  $X^k$  by  $x^k$ , so that the event  $X^k = x^k$  can be unambiguously denoted just by  $x^k$ . Notice also that we consider biases of the units implicitly via an additional input fixed to value one. The posterior distribution of each layer  $k > 0$ , given the observations  $x^0$ , recurrently expresses as

$$
p \left(X ^ {k} \mid x ^ {0}\right) = \mathbb {E} _ {X ^ {k - 1} \mid x ^ {0}} \left[ p \left(X ^ {k} \mid X ^ {k - 1}\right) \right] = \int p \left(X ^ {k} \mid x ^ {k - 1}\right) p \left(x ^ {k - 1} \mid x ^ {0}\right) d x ^ {k - 1}. \tag {2}
$$

The posterior distribution of the last layer,  $p(X^l \mid x^0)$  is the prediction of the model.

We now explain how the standard NNs with injected noises give rise to the Bayesian networks of the form (1). Consider a deterministic nonlinear mapping applied to a "noised" activation:

$$
X ^ {k} = f \left(A ^ {k} - Z ^ {k}\right), \tag {3}
$$

where  $f\colon \mathbb{R}\to \mathbb{R}$  is applied component-wise and  $Z_{i}^{k}$  are independent real-valued random variables with a known distribution (such as the standard normal distribution). From representation (3) we can recover the conditional cdf of the belief network  $F_{X^k |X^{k - 1}}(u) = \mathbb{E}[[f(W^k X^{k - 1} - Z^k)\leq u|X^{k - 1}]]$  and the respective conditional density.

Example 1. Stochastic binary unit (Williams, 1992). Let  $Y$  be a binary valued r.v. given by  $Y = \Theta(A - Z)$ , where  $\Theta$  is the Heaviside step function and  $Z$  is noise with cdf  $F_Z$ . Then  $\mathbb{P}(Y = 1 \mid A) = F_Z(A)$ . This is easily seen from

$$
\mathbb {P} (Y = 1 \mid A) = \mathbb {P} (\Theta (A - Z) = 1 \mid A) = \mathbb {P} (Z \leq A | A) = F _ {Z} (A). \tag {4}
$$

If, for instance,  $Z$  is distributed with standard logistic distribution, then  $\mathbb{P}(Y = 1 \mid A) = S(A)$ , where  $S$  is the logistic sigmoid function  $S(a) = (1 + e^{-a})^{-1}$ .

In general, the expectation (2) is intractable to compute and the resulting posterior can have a combinatorial number of modes. However, in many cases of interest it is suitable to approximate the posterior  $p(X^k \mid x^0)$  for a given  $x^0$  with a factorized distribution  $q(X^k) = \prod_i q(X_i^k)$ . We expect that in many recognition problems, given the input image, the hidden states and the final prediction are concentrated around some specific values (unlike in generative problems, where the posterior distributions are typically multi-modal). A similar factorized approximation is made for the activations. The exact shape of distributions  $q(X_i^k)$  and  $q(A_i^k)$  can be chosen appropriately depending on the unit type: e.g., a Bernoulli distribution for binary  $X_i^k$  a Gaussian or Logistic distribution for real-valued activations  $A_i^k$ . We will rely on the fact that the mean and variance are sufficient statistics for such approximating distributions. Then, as long as we can calculate these sufficient statistics for the layer of interest, the exact shape of distributions for the intermediate outputs need not be assumed.

The information-theoretic optimal factorized approximation to the posterior  $p(X^k \mid x^0)$ , minimizing the forward KL divergence  $KL(p(X^k \mid x^0) \| q(X^k))$ , is given by marginals  $\prod_i p(X_i^k \mid x^0)$ . Furthermore, in the case when  $q(X_i^k)$  is from to the exponential family, the optimal approximation is given by matching the moments of  $q(X_i^k)$  to  $p(X_i^k \mid x^0)$ . The factorized approximation then can be computed layer-by-layer, assuming that the preceding layer was already approximated. Substituting  $q(X^{k-1})$  for  $p(X^{k-1} \mid x^0)$  in (2) results in the procedure

$$
q \left(X _ {i} ^ {k}\right) = \mathbb {E} _ {q \left(X ^ {k - 1}\right)} \left[ p \left(X _ {i} ^ {k} \mid X ^ {k - 1}\right) \right] = \int p \left(X _ {i} ^ {k} \mid x ^ {k - 1}\right) \prod_ {i} q \left(x _ {i} ^ {k - 1}\right) d x ^ {k - 1}. \tag {5}
$$

Thus we need to propagate the factorized approximation layer-by-layer, by the marginalization update (5) until we get the approximate posterior output  $q(\bar{X}^l)$ . This method is closely related to the

assumed density filtering (see Minka, 2001), in which, in the context of learning, one chooses a family of distributions that is easy to work with and "projects" the true posterior onto the family after each measurement update. Here, the projection takes place after propagating each layer for the purpose of the inference.

# 3 PROPAGATION IN BASIC LAYERS

We now consider a single layer at a time and detail how (5) is computed (approximately) for a layer consisting of a linear mapping  $A = w^{\mathsf{T}}X$  (scalar output, for clarity) and a non-linear noisy activation  $Y = f(A - Z)$ .

Linear Mapping Activation  $A$  in a typical deep network is a sum of hundreds of stochastic inputs  $X$  (from the previous layer). This justifies the assumption that  $A - Z$  (where  $Z$  is a smoothly distributed injected noise) can be approximated by a uni-modal distribution fully specified by mean and variance as e.g. normal or logistic distribution<sup>1</sup>. Knowing the statistics of  $Z$ , it is therefore sufficient to estimate the mean and the variance of the activation  $A$  given by

$$
\mu^ {\prime} = \mathbb {E} [ A ] = w ^ {\mathsf {T}} \mathbb {E} [ X ] = w ^ {\mathsf {T}} \mu , \tag {6a}
$$

$$
\sigma^ {\prime 2} = \sum_ {i j} w _ {i} w _ {j} \operatorname {C o v} [ X ] _ {i j} \approx \sum_ {i} w _ {i} ^ {2} \sigma_ {i} ^ {2}, \tag {6b}
$$

where  $\mu$  is the mean and  $\operatorname{Cov}[X]$  is the covariance matrix of  $X$ . The approximation of the covariance matrix by its diagonal is implied by the factorization assumption for the activations  $A$ .

Nonlinear Coordinate-wise Mappings Let  $A$  be a scalar r.v. with statistics  $\mu, \sigma^2$  and let  $Y = f(A - Z)$  with independent noise  $Z$ . Assuming that  $\widetilde{A} = A - Z$  is distributed normally or logistically with statistics  $\tilde{\mu}, \tilde{\sigma}^2$ , we can approximate the expectation and variance of  $Y = f(\widetilde{A})$

$$
\mu_ {i} ^ {\prime} = \mathbb {E} _ {q (\widetilde {A})} [ f (\widetilde {A}) ], \quad \sigma_ {i} ^ {\prime 2} = \mathbb {E} _ {q (\widetilde {A})} [ f ^ {2} (\widetilde {A}) ] - \mu_ {i} ^ {\prime 2} \tag {7}
$$

by analytic expressions for most of the commonly used non-linearities. For binary variables, occurring in networks with Heaviside nonlinearities, the distribution  $q(Y)$  is fully described by one parameter  $\mu_{i} = \mathbb{E}[Y]$ , and the propagation rule (5) becomes

$$
\mu_ {i} ^ {\prime} = \mathbb {E} _ {q (A)} \left[ p \left(Y = 1 \mid A ^ {k}\right) \right], \quad \sigma_ {i} ^ {\prime 2} = \mu_ {i} ^ {\prime} \left(1 - \mu_ {i} ^ {\prime}\right), \tag {8}
$$

where the variance is dependent but will be needed in propagation through other layers.

Example 2. Heaviside Nonlinearity with Noise. Consider the model  $Y = \Theta (A - Z)$ , where  $Z$  is logistic noise. The statistics of  $\widetilde{A} = A - Z$  are given by  $\tilde{\mu} = \mu$  and  $\tilde{\sigma}^2 = \sigma^2 +\sigma_S^2$ , where  $\sigma_S^2 = \pi^2 /3$  is the variance of  $Z$ . Assuming noisy activations  $\widetilde{A}$  to have logistic distribution, we obtain the mean of  $Y$  as:

$$
\mu^ {\prime} = \mathbb {E} [ \Theta (\tilde {A}) ] = \mathbb {P} (\widetilde {A} \geq 0) = \mathbb {P} \left(\frac {\widetilde {A} - \tilde {\mu}}{\tilde {\sigma} / \sigma_ {S}} \geq \frac {- \tilde {\mu}}{\tilde {\sigma} / \sigma_ {S}}\right) \doteq \mathcal {S} \left(\frac {\tilde {\mu}}{\tilde {\sigma} / \sigma_ {S}}\right) = \mathcal {S} \left(\frac {\mu}{\sqrt {\sigma^ {2} / \sigma_ {S} ^ {2} + 1}}\right), \tag {9}
$$

where the dotted equality is due to that  $-(A - \tilde{\mu})\frac{\sigma_S}{\tilde{\sigma}}$  has standard logistic distribution and that the sigmoid function  $S$  is its cdf. The variance of  $Y$  is expressed as in (8).

Example 3. Rectified Linear Unit (ReLU) Assuming the activation  $A$  to be normally distributed, the mean of  $Y = \max(0, A)$  expresses as  $\mu' = \int_{-\infty}^{\infty} \max(0, a) p(a) \mathrm{d}a = \int_0^\infty a p(a) \mathrm{d}a = \mu \Phi(\mu / \sigma) + \sigma \phi(\mu / \sigma)$ , i.e., expresses analytically using the pdf  $\phi$  and cdf  $\Phi$  of the standard normal distribution. The variance can be expressed as well. These expressions, used by Frey & Hinton (1999); Hernández-Lobato & Adams (2015) rely on function  $\Phi$ , which has limited numerical accuracy and may lead to negative output variances. In § 4.4 we propose an approximation for leaky ReLU, which is numerically stable and is suitable for ReLU as well.

Fig. 2 shows the approximations for propagation through Heaviside, ReLU and leaky ReLU nonlinearities. Note that all expectations over a smoothly distributed  $A$  result in smooth propagation functions regardless the smoothness (or lack thereof) of the original function.

![](images/ef4da4d799275eb105231cfb8150391b6965fce201b693ac30959bea41be8366.jpg)  
Figure 2: Propagation for the Heaviside function:  $Y = \llbracket A \geq 0 \rrbracket$ , ReLU:  $Y = \max(0, A)$  and leaky ReLU:  $Y = \max(\alpha A, A)$ . Red: activation function. Black: an exemplary input distribution with mean  $\mu = 3$ , variance  $\sigma^2 = 1$  shown with support  $\mu \pm 3\sigma$ . Dashed blue: the approximate mean  $\mu'$  of the output versus the input mean  $\mu$ . The variance of the output is shown as blue shaded area  $\mu' \pm 3\sigma'$ .

Summarizing, we can represent the approximate inference in networks with binary and continuous variables as a feed-forward moment propagation: given the approximate moments of  $X^{k - 1} \mid x^0$ , the moments of  $X_{i}^{k} \mid x^{0}$  are estimated via (8), (7) ignoring dependencies between  $X_{j}^{k - 1} \mid x^{0}$  on each step (as implied by the factorized approximation).

AP1 and AP2 The standard NN can be viewed as a further simplification of the proposed method: it makes the same factorization assumption but does not compute variances of the activations (6b) and propagates only the means. Consequently, a zero variance is assumed in propagation through non-linearities. In this case the expected values of mappings such as  $\Theta(A)$  and  $\mathrm{ReLU}(A)$  are just these functions evaluated at  $\mu$ . For injected noise models we obtain smoothed versions: e.g., substituting  $\sigma = 0$  in the noisy Heaviside function (9) recovers the standard sigmoid function. We thus can view standard NNs as making a simpler from of factorized inference in the same Bayesian NN model. We designate this simplification (in figures and experiments) by AP1 and the method using variances by AP2 ("AP" stands for approximation).

# 4 PROPAGATION IN CATEGORICAL AND MAX LAYERS

In this section we present our main technical contribution: propagation rules for argmax, softmax and max mappings, that are non-linear and multivariate. Similar to how sigmoid function is obtained as the expectation of the Heaviside function with injected noise in Example 2, we observe that softmax layer is the expectation of argmax with injected noise. It will follow that the standard NN with softmax layer can be viewed as AP1 approximation of argmax layer with injected noise. We propose a new approximation for the argmax posterior probability that takes into account uncertainty (variances) of the activations and enables propagation through argmax and softmax layers. Next, we observe that the maximum of several variables (used in max-pooling) can be expressed through argmax. This gives a new one-shot approximation of the expected maximum using argmax probabilities. Finally, we consider the case of leaky ReLU, which is a maximum of two correlated variables. The proposed approximations are relatively easy to compute and differentiable, which facilitates their usage in NNs.

# 4.1 ARGMAX AND SOFTMAX

The softmax function, most commonly used to model a categorical distribution, ubiquitous in classification, is defined as  $p(Y = y|x) = e^{x_y} / \sum_k e^{x_k}$ , where  $y$  is the class index. We explore the following latent variable representation known in the theory of discrete choice:  $p(Y = y|x) = \mathbb{E}[\overline{Y}_y]$ , where  $\overline{Y} \in \{0,1\}^n$  is the indicator of the noisy argmax:  $\overline{Y}_y = \llbracket \operatorname{argmax}_k(X_k + \Gamma_k) = y\rrbracket$  and  $\Gamma_k$  follows the standard Gumbel distribution. Standard NN implements the AP1 approximation of this latent model: conditioned on  $X = x$ , the expectation over latent noises  $\Gamma$  is the softmax(x).

For the AP2 approximation we need to compute the expectation w.r.t. both:  $X$  and  $\Gamma$ , or, what is the same, to compute the expectation of  $\operatorname{softmax}(X)$  over  $X$ . This task is difficult, particularly because

variances of  $X_{i}$  may differ across components. First, we derive an approximation for the expectation of argmax indicator without injected noise:

$$
\bar {Y} _ {y} = \llbracket \operatorname * {a r g m a x} _ {k} X _ {k} = y \rrbracket . \tag {10}
$$

The injected noise case can be treated by simply increasing the variance of each  $X_{i}$  by the variance of standard Gumbel distribution.

Let  $X_{k}$ ,  $k = 1, \ldots, n$  be independent, with mean  $\mu_{k}$  and variance  $\sigma_{k}^{2}$ . We need to estimate

$$
\mathbb {E} \left[ \bar {Y} _ {y} \right] = \mathbb {E} _ {X} \llbracket X _ {y} - X _ {k} \geq 0 \forall k \neq y \rrbracket , \tag {11}
$$

The vector  $U$  with components  $U_{k} = X_{y} - X_{k}$  for  $k \neq y$  is from  $\mathbb{R}^{n-1}$  with component means  $\tilde{\mu}_{k} = \mu_{y} - \mu_{k}$  and component variances  $\tilde{\sigma}_{k}^{2} = \sigma_{y}^{2} + \sigma_{k}^{2}$ . Note the components of  $U$  are not independent.

We approximate the distribution of  $U$  by the multivariate logistic distribution defined by Malik & Abraham (1973). This choice is motivated by the extrapolation of the case with two input variables. The approximation is made by shifting and rescaling the distribution in order to match the means and marginal variances. The marginal distributions of standard multivariate logistic distribution are standard logistic with zero mean and variance  $\sigma_{S}$ . Thus the approximation assumes that  $(U_k - \tilde{\mu}_k)\sigma_S / \tilde{\sigma}_k$  is standard  $(n-1)$ -variate logistic with the cdf given by  $S_{n-1}(u) = \frac{1}{1 + \sum_k e^{-u_k}}$  (Malik & Abraham, 1973, eq. 2.5). It allows us to evaluate the necessary probability:

$$
q (y) = \mathbb {E} [ \bar {Y} _ {y} ] = \mathbb {P} (U \geq 0) = \mathbb {P} \left(\frac {U _ {k} - \tilde {\mu} _ {k}}{\tilde {\sigma} _ {k} / \sigma_ {\mathcal {S}}} \geq \frac {- \tilde {\mu} _ {k}}{\tilde {\sigma} _ {k} / \sigma_ {\mathcal {S}}} \forall k \neq y\right) = \mathcal {S} _ {n - 1} \left(\frac {- \tilde {\mu} _ {k}}{\tilde {\sigma} _ {k} / \sigma_ {\mathcal {S}}}\right). \tag {12}
$$

Expanding  $\tilde{\mu},\tilde{\sigma}^2$  and noting that  $\mu_k - \mu_y = 0$  for  $y = k$ , we obtain the approximation

$$
q (y) = \left(\sum_ {k} \exp \left\{\frac {\mu_ {k} - \mu_ {y}}{\sqrt {\left(\sigma_ {k} ^ {2} + \sigma_ {y} ^ {2}\right) / \sigma_ {S} ^ {2}}} \right\}\right) ^ {- 1}. \tag {13}
$$

This approximation has linear memory complexity but requires quadratic time in the number of inputs, which may be prohibitive for applications in NNs. We can simplify it further as follows. The expression (13) simplifies when we can approximate

$$
\frac {\mu_ {k} - \mu_ {y}}{\sqrt {\left(\sigma_ {k} ^ {2} + \sigma_ {y} ^ {2}\right) / \sigma_ {S} ^ {2}}} \approx a _ {k} - a _ {y} \tag {14}
$$

with some choice of  $a_k$  for all  $k$ . In this case we obtain  $q(y) = (\operatorname{softmax}(a))_y$ . We therefore propose the approximation

$$
q = \operatorname {s o f t m a x} (a) \text {w i t h} a _ {k} = \mu_ {k} / \sqrt {\left(\sigma_ {k} ^ {2} + \frac {n \bar {\sigma} ^ {2} - \sigma_ {k} ^ {2}}{n - 1}\right) / \sigma_ {S} ^ {2}}, \tag {15}
$$

where  $\bar{\sigma}^2 = \frac{1}{n}\sum_k\sigma_k^2$  is the average variance.

Importantly, the approximation is consistent with the already obtained results for the following special cases. In the case of two input variables, for the simplified approximation with  $a_{k}$  set as (15) we have  $a_{k} = \mu_{k} / \sqrt{(\sigma_{1}^{2} + \sigma_{2}^{2}) / \sigma_{S}^{2}}$ , i.e. (14) holds as equality, and we obtain

$$
q (y = 1) = \operatorname {s o f t m a x} \left(a _ {1}, a _ {2}\right) _ {1} = \frac {e ^ {a _ {1}}}{e ^ {a _ {1}} + e ^ {a _ {2}}} = \frac {1}{1 + e ^ {a _ {2} - a _ {1}}} = \mathcal {S} \left(a _ {2} - a _ {1}\right) = \mathcal {S} \left(\frac {\tilde {\mu}}{\tilde {\sigma} / \sigma_ {S}}\right), \tag {16}
$$

which matches the approximation of the Heaviside posterior with input  $X_{1} - X_{2}$  with mean  $\tilde{\mu}$  and variance  $\tilde{\sigma}^2$ . As a consequence expectation of softmax (argmax indicator with injected noise) matches the expectation of sigmoid (Heaviside function with injected noise) given by (9).

In the case when all variances  $\sigma_k^2$  are equal:  $\sigma_{k} = \sigma$ , the approximation (15) results in

$$
q = \operatorname {s o f t m a x} \left(\frac {\mu_ {k}}{\sqrt {2} \sigma / \sigma_ {S}}\right). \tag {17}
$$

More specifically, when  $X_{k} = \mu_{k} + \Gamma_{k}$ , where  $\Gamma_{k}$  is standard Gumbel (with variance  $\pi^2 / 6 = \sigma_S^2 / 2$ ) we obtain that  $q = \mathrm{softmax}(\mu_k)$ , i.e. recover the exact expectation of noisy argmax with deterministic inputs used by AP1.

# 4.2 MAXIMUM OF TWO VARIABLES

Let us consider the function  $\max(X_1, X_2)$ , which is important for leaky ReLU and maxOut. In this case, exact expressions for the moments for the maximum of two Gaussian random variables  $X_1, X_2$  are known (Nadarajah & Kotz, 2008). Denoting  $s = (\sigma_1^2 + \sigma_2^2 - 2\operatorname{Cov}[X_1, X_2])^{\frac{1}{2}}$  and  $a = (\mu_1 - \mu_2) / s$ , the mean and variance of  $\max(X_1, X_2)$  can be expressed as:

$$
\mu^ {\prime} = \mu_ {1} \Phi (a) + \mu_ {2} \Phi (- a) + s \phi (a), \tag {18a}
$$

$$
\sigma^ {\prime 2} = \left(\sigma_ {1} ^ {2} + \mu_ {1} ^ {2}\right) \Phi (x) + \left(\sigma_ {2} ^ {2} + \mu_ {2} ^ {2}\right) \Phi (- a) + \left(\mu_ {1} + \mu_ {2}\right) s \phi (a) - \mu^ {\prime 2}. \tag {18b}
$$

These expressions involving the normal cdf  $\Phi$ , will not be used directly. We simplify them in the case of leaky ReLU and use as a reference for maximum of multiple variables. The variance can be further expressed as

$$
\sigma^ {\prime 2} = \sigma_ {1} ^ {2} \Phi (a) + \sigma_ {2} ^ {2} \Phi (- a) + s ^ {2} \left(a ^ {2} \Phi (a) + a \phi (a) - \left(a \Phi (a) + \phi (a)\right) ^ {2}\right). \tag {19}
$$

We observe that the function of one variable  $a^2\Phi(a) + a\phi(a) - (a\Phi(a) + \phi(a))^2$  is always negative, quickly vanishes with  $|a|$  increasing and is above  $-0.16$ . By neglecting it, we obtain a rather tight upper bound  $\sigma'^2 \leq \sigma_1^2\Phi(a) + \sigma_2^2(1 - \Phi(a))$ . Note that  $\Phi(a)$ , which serves as interpolating coefficient between  $\sigma_1^2$  and  $\sigma_2^2$ , is precisely the probability of the event  $X_1 > X_2$ . This suggests the idea of estimating mean and variance of max from the argmax probabilities in the multivariate case.

# 4.3 MAXIMUM OF SEVERAL VARIABLES

Let  $X_{k}$ ,  $k = 1, \ldots, n$  be independent, with mean  $\mu_{k}$  and variance  $\sigma_{k}^{2}$ . The moments of the maximum  $Y = \max_{k} X_{k}$ , assuming the distributions of  $X_{k}$  are known, can be computed by integration with the CDF of  $Y$  (Ross, 2010) given by  $F_{Y}(y) = \mathbb{P}(X_{k} \leq y \forall k) = \prod_{k} F_{X_{k}}(y)$ . However, this requires numerical 1D integration. We seek a simpler approximation. One option is to compose the maximum of  $n > 2$  variables hierarchically using maximum of two variables § 4.2 and assume that the intermediate results are distributed normally.

We propose a new non-trivial one-shot approximations for the mean and variance assuming that the argmax probabilities  $q_{k} = \mathbb{P}(X_{k}\geq X_{j}\forall j)$  are already estimated. The derivation of these approximations and proofs of their accuracy are given in § A.

Proposition 1. Assuming  $X_{k}$  are logistic  $(\mu_k, \sigma_k^2)$ , the mean of  $Y = \max_{k} X_{k}$  can be approximated (upper bounded) by

$$
\mu^ {\prime} \approx \sum_ {k} q _ {k} \hat {\mu} _ {k}, \text {w h e r e} \hat {\mu} _ {k} = \mu_ {k} + \frac {\sigma_ {k}}{q _ {k} \sigma_ {S}} H \left(q _ {k}\right), \tag {20}
$$

where  $H(q_{k})$  is the entropy of the Bernoulli distribution with probabilities  $q_{k}$ . The variance of  $Y$  can be approximated as

$$
\sigma^ {\prime 2} \approx \sum_ {k} \sigma_ {k} ^ {2} \mathcal {S} (a + b \mathcal {S} ^ {- 1} \left(q _ {k}\right)) + \sum_ {k} q _ {k} \left(\hat {\mu} _ {k} - \mu^ {\prime}\right) ^ {2}, \tag {21}
$$

where  $a = -1.33751$  and  $b = 0.886763$  are coefficients originating from a Taylor expansion.

Notice the similarity to the expressions (18a) and (19) (identifying  $q_{1}$ ,  $q_{2}$  with argmax probabilities  $\Phi(a)$ ,  $\Phi(-a)$ , resp.). Also notice that the entropy is non-negative, and thus increases  $\mu'$  when the argmax is ambiguous, as expected in the extreme value theory.

# 4.4 LEAKY RELU

$\mathsf{LReLU}$  is a popular max-related function defined as:  $Y = \max (\alpha X,X)$ . We use the exact expressions for the case of two correlated normal variables (18a) and (19). Assume that  $\alpha < 1$ , let  $X_{2} = \alpha X_{1}$  and denote  $\mu = \mu_{1}$  and  $\sigma^2 = \sigma_1^2$ . Then  $\mu_{2} = \alpha \mu$ ,  $\sigma_2^2 = \alpha^2\sigma^2$  and  $\operatorname{Cov}[X_1,X_2] = \operatorname{Cov}[X_1,\alpha X_1] = \alpha \sigma^2$ . We have  $s = \sigma (1 - \alpha)$  and  $a = (\mu_1 - \mu_2) / s = \mu (1 - \alpha) / s = \mu /\sigma$ . The mean  $\mu^\prime$  expresses as

$$
\mu^ {\prime} = \mu (\alpha + (1 - \alpha) \Phi (a)) + \sigma (1 - \alpha) \phi (a). \tag {22}
$$

The variance  $\sigma^{\prime 2}$  expresses as

$$
\begin{array}{l} \sigma^ {2} \left(\Phi (a) + \alpha^ {2} (1 - \Phi (a)) + (1 - \alpha) ^ {2} \left(a ^ {2} \Phi (a) + a \phi (a) - (a \Phi (a) + \phi (a)) ^ {2}\right)\right) (23) \\ = \sigma^ {2} \left(\alpha^ {2} + 2 \alpha (1 - \alpha) \Phi (a) + (1 - \alpha) ^ {2} \mathcal {R} (a)\right), (24) \\ \end{array}
$$

where  $\mathcal{R}(a) = a\phi(a) + (a^2 + 1)\Phi(a) - (a\Phi(a) + \phi(a))^2$  is a sigmoid-shape function of one variable. In practice we approximate  $\sigma'^2$  with the simpler function

$$
\sigma^ {\prime 2} \approx \sigma^ {2} \left(\alpha^ {2} + \left(1 - \alpha^ {2}\right) \mathcal {S} (a / t)\right), \tag {25}
$$

where  $t = 0.3758$  is set by fitting the approximation. The approximation is shown in Fig. 2.

# 5 EXPERIMENTS

In the experiments we evaluate the accuracy of the proposed approximation and compare it to the standard propagation. We also test the method in the end-to-end learning and show that with a simple calibration it achieves better test likelihoods than the state-of-the-art. Full details of the implementation, training protocols, used datasets and networks are given in  $\S$  B. The running time of AP2 is  $2\times$  more for a forward pass and  $2 - 3\times$  more for a forward-backward pass than that of AP1.

# 5.1 APPROXIMATION ACCURACY

We conduct two experiments: how well the proposed method approximates the real posterior of neurons, w.r.t. noise in the network input and w.r.t. dropout. The first case (illustrated in Fig. 1) is studied on the LeNet5 model of Lecun et al. (2001), a 5-layer net with max pooling detailed in § B.4, trained on MNIST dataset using standard methods. We set LReLU activations with  $\alpha = 0.01$  to test the proposed approximations. We estimate the ground truth statistics  $\mu^{*}$ ,  $\sigma^{*}$  of all neurons by the Monte Carlo (MC) method: drawing 1000 samples of noise per input image and collecting sample-based statistics for each neuron. Then we apply AP1 to compute  $\mu_{1}$  and AP2 to compute  $\mu_{2}$  and  $\sigma_{2}$  for each unit from the clean input and known noise variance  $\sigma_0$ . The error measure of the means  $\varepsilon_{\mu}$  is the average  $|\mu -\mu^{*}|$  relative to the average  $\sigma^{*}$ . The averages are taken over all units in the layer and over input images. The error of the standard deviation  $\varepsilon_{\sigma}$  is the geometric mean of  $\sigma /\sigma^{*}$ , representing the error as a factor from the true value (e.g., 1.0 is exact, 0.9 is under-estimating and 1.1 is over-estimating). Table 1 shows average errors per layer and points the main observation: that AP2 is more accurate than AP1 but both methods suffer from the factorization assumption. The variance computed by AP2 provides a good estimate and the estimated categorical distribution by propagating the variance through softmax is much closer to the MC estimate.

Next, we study a widely used ALL-CNN network § B.4 by Springenberg et al. (2015) trained with standard dropout on CIFAR-10. Bernoulli dropout noise with dropout rate 0.2 is applied after each activation. The accuracies of estimated statistics w.r.t. dropout noises are shown in Table 2. Here, each layer receives uncertainty propagated from preceding layers, but also new noises are mixed-in in each layer, which works in favor of the factorization assumption. The results are shown in Table 2. Observe that GT noise variance  $\sigma^{*}$  changes significantly across layers, up to 1-2 orders and AP2 gives a useful estimate. Furthermore, having estimated the average factors suggests a simple calibration.

Calibration We divide the variance in the last layer by the average factor  $\sigma / \sigma^{*}$  estimated on the validation set. With this method, denoted AP2 calibrated, we get significantly better test likelihoods in the end-to-end learning experiment.

# 5.2 ANALYTIC NORMALIZATION

The AP2 method can be used to approximate neuron statistics w.r.t. the input chosen at random from the training dataset as was proposed by Shekhotsov & Flach (2018). Instead of propagating sample instances, the method takes the dataset statistics  $(\mu^0, (\sigma^0)^2)$  and propagates them once through all network layers, averaging over spatial dimensions. The obtained neuron mean and variance are then used to normalize the output the same way as in batch normalization (Ioffe & Szegedy, 2015). This normalization leads to a better conditioned initialization and training and is batch-independent. We

<table><tr><td></td><td>Conv</td><td>LReLU</td><td>MaxPool</td><td>Conv</td><td>LReLU</td><td>MaxPool</td><td>FC</td><td>LReLU</td><td>FC</td><td>LReLU</td><td>FC</td><td>Softmax</td></tr><tr><td colspan="13">Noisy input N(0,10-4)</td></tr><tr><td>σ*</td><td>0.03</td><td>0.02</td><td>0.02</td><td>0.06</td><td>0.03</td><td>0.03</td><td>0.09</td><td>0.05</td><td>0.10</td><td>0.05</td><td>0.11</td><td></td></tr><tr><td>εμ1</td><td>0.02</td><td>0.19</td><td>0.37</td><td>0.84</td><td>0.43</td><td>0.52</td><td>1.20</td><td>0.66</td><td>1.16</td><td>0.62</td><td>1.25</td><td>KL 3.5e-4</td></tr><tr><td>εμ2</td><td>0.02</td><td>0.02</td><td>0.13</td><td>0.29</td><td>0.13</td><td>0.17</td><td>0.37</td><td>0.21</td><td>0.36</td><td>0.20</td><td>0.39</td><td>KL 3.3e-5</td></tr><tr><td>εσ2</td><td>1.00</td><td>1.05</td><td>1.25</td><td>1.06</td><td>1.06</td><td>1.12</td><td>1.09</td><td>1.10</td><td>1.03</td><td>1.04</td><td>0.96</td><td></td></tr><tr><td colspan="13">Noisy input N(0,0.01)</td></tr><tr><td>σ*</td><td>0.3</td><td>0.16</td><td>0.20</td><td>0.58</td><td>0.24</td><td>0.27</td><td>0.79</td><td>0.47</td><td>0.86</td><td>0.42</td><td>0.92</td><td></td></tr><tr><td>εμ1</td><td>0.02</td><td>0.24</td><td>0.53</td><td>1.46</td><td>0.58</td><td>0.70</td><td>1.44</td><td>0.85</td><td>1.40</td><td>0.79</td><td>1.57</td><td>KL 0.36</td></tr><tr><td>εμ2</td><td>0.02</td><td>0.02</td><td>0.21</td><td>0.65</td><td>0.21</td><td>0.31</td><td>0.61</td><td>0.37</td><td>0.67</td><td>0.34</td><td>0.72</td><td>KL 0.05</td></tr><tr><td>εσ2</td><td>1.00</td><td>1.10</td><td>1.15</td><td>1.17</td><td>1.22</td><td>1.42</td><td>1.37</td><td>1.59</td><td>1.31</td><td>1.47</td><td>1.23</td><td></td></tr></table>

Table 1: Accuracy of approximation of mean and variance statistics for each layer in a fully trained LeNet5 (MNIST) tested with noisy input. Observe the following: MC variance  $\sigma^{*}$  is growing significantly from the input to the output; both AP1 and AP2 have a significant drop of accuracy at linear (fc and conv) layers, due to factorized approximation assumption; AP2 approximation of the standard deviation is within a factor close to one, and makes a meaningful estimate, although degrading with depth; AP2 approximation of the mean is more accurate than AP1; the KL divergence from the MC class posterior is improved with AP2.  

<table><tr><td></td><td>C A</td><td>C A</td><td>C A</td><td>C A</td><td>C A</td><td>C A</td><td>C A</td><td>C A</td><td>C A</td><td>C A</td><td>C A</td><td>C P</td><td>Softmax</td></tr><tr><td>σ*</td><td>0 0.26</td><td>0.31 0.46</td><td>0.86 0.77</td><td>1.1 0.78</td><td>1.7 0.97</td><td>2.2 1.3</td><td>1.5 0.89</td><td>2 0.74</td><td>16 2.8</td><td></td><td></td><td></td><td></td></tr><tr><td>εμ1</td><td>- 0.01</td><td>0.02 0.03</td><td>0.07 0.06</td><td>0.17 0.09</td><td>0.19 0.10</td><td>0.25 0.11</td><td>0.22 0.11</td><td>0.21 0.12</td><td>0.17 0.38</td><td>KL 0.11</td><td></td><td></td><td></td></tr><tr><td>εμ2</td><td>- 0.01</td><td>0.02 0.01</td><td>0.02 0.02</td><td>0.05 0.02</td><td>0.06 0.03</td><td>0.07 0.04</td><td>0.08 0.04</td><td>0.09 0.04</td><td>0.05 0.14</td><td>KL 0.04</td><td></td><td></td><td></td></tr><tr><td>εσ2</td><td>- 1.00</td><td>1.00 1.02</td><td>0.88 0.89</td><td>0.90 0.95</td><td>0.84 0.87</td><td>0.77 0.77</td><td>0.82 0.85</td><td>0.88 0.92</td><td>0.69 0.45</td><td></td><td></td><td></td><td></td></tr></table>

Table 2: Accuracy of approximation of mean and variance statistics for each layer in All-CNN (CIFAR-10) trained and tested with dropout. The table shows accuracies after all layers ( C-convolution, A-activation, P-average pooling) and the final KL divergence. A similar effect to propagating input noise is observed: the MC variance  $\sigma^{*}$  grows with depth; a significant drop of accuracy is observed in conv and pooling layers which exploit the independence assumption.

verify the efficiency of this method for a network that includes the proposed approximations for LReLU and max pooling layers in § B.5 and use it in the end-to-end learning experiment below.

# 5.3 END-TO-END LEARNING WITH ANALYTIC DROPOUT

In this experiment we approximate the dropout analytically at training time similar to Wang & Manning (2013) but including the new approximations for LReLU and softmax layers. We compare training All-CNN network on CIFAR-10 without dropout, with standard dropout (Srivastava et al., 2014) and analytic (AP2) dropout. All three cases use exactly the same initialization, AP2 normalization as discussed above and the same learning setup. Only the learning rate is optimized individually per method § B.3. The dropout layers with dropout rate 0.2 are applied after every activation and there is no input dropout. Fig. 3 shows the progress of the three methods. The analytic dropout is efficient as a regularizer (reduces overfitting in the validation likelihood), is non-stochastic and progresses faster than standard stochastic dropout. While latter slows the training down due to increased stochasticity of the gradient, the analytic dropout smoothes the loss function and speeds the training up. This is especially visible on the training loss plot Fig. B.3. Furthermore, analytic dropout can be applied as the test-time inference method in a network trained with any variant of dropout. Table 3 shows that AP2, calibrated as proposed above, achieves the best test likelihood, significantly improving SOTA results for this network. Some additional results are given in § B.7. Differently from Wang & Manning (2013), we find that when trained with standard dropout, all test methods achieve approximately the same accuracy and only differ in likelihoods. We believe this is due to the deep CNN in our case that achieves  $100\%$  training accuracy.

We also attempted comparison with other approaches. Gaussian dropout (Srivastava et al., 2014) performed similarly to standard Bernoulli dropout. Variational dropout Kingma et al. (2015) in our implementation for convolutional networks has diverged or has not reached the accuracy of the baseline without dropout (we tried correlated and uncorrelated versions with or without local reparametrization trick and with different KL divergence factors 1, 0.1, 0.01, 0.001).

![](images/a300e0ccde2196630de82c948d0978ec9a72951f8ba9373f9c1235df9f6d6ead.jpg)  
Validation Accuracy  
Table 3: Results for All-CNN on CIFAR-10 test set: negative log likelihood (NLL) and accuracy. Left: state of the art results for this network (Gast & Roth, 2018, table 3). Middle: All-CNN trained with standard dropout (our learning schedule and analytic normalization) evaluated using different test-time methods. Observe that "AP2 calibrated" well approximates dropout: the test likelihood is better than MC-100. Right: All-CNN trained with analytic dropout (same schedule and normalization). Observe that "AP2 calibrated" achieves the best likelihood and accuracy.

![](images/a76cac85faf4edb13b2cd4402eaa6304005e5ee6fbf5f43790bdca84abbf0d29.jpg)  
Validation Loss  
Figure 3: Comparison of analytic AP2 dropout with baselines. All methods use AP2 normalization during training. Analytic dropout converges to similar values of stochastic dropout and is faster per iteration. Both methods are efficient in preventing overfitting as seen in the right plot.

<table><tr><td colspan="3">SOTA results (Gast &amp; Roth, 2018)</td></tr><tr><td>Method</td><td>NLL</td><td>Acc.</td></tr><tr><td>Dropout MC-30</td><td>0.327</td><td>90.88</td></tr><tr><td>ProbOut</td><td>0.37</td><td>91.9</td></tr></table>

<table><tr><td colspan="3">Standard dropout</td></tr><tr><td>Test method</td><td>NLL</td><td>Acc.</td></tr><tr><td>AP1</td><td>0.434</td><td>0.938</td></tr><tr><td>AP2</td><td>0.311</td><td>0.936</td></tr><tr><td>AP2 calibrated</td><td>0.214</td><td>0.937</td></tr><tr><td>MC-10</td><td>0.264</td><td>0.935</td></tr><tr><td>MC-100</td><td>0.217</td><td>0.937</td></tr><tr><td>MC-1000</td><td>0.210</td><td>0.937</td></tr></table>

<table><tr><td colspan="3">Analytic dropout</td></tr><tr><td>Test method</td><td>NLL</td><td>Acc.</td></tr><tr><td>AP1</td><td>1.86</td><td>0.940</td></tr><tr><td>AP2</td><td>0.363</td><td>0.940</td></tr><tr><td>AP2 calibrated</td><td>0.194</td><td>0.940</td></tr><tr><td>MC-10</td><td>0.546</td><td>0.919</td></tr><tr><td>MC-100</td><td>0.281</td><td>0.925</td></tr><tr><td>MC-1000</td><td>0.243</td><td>0.926</td></tr></table>

# 6 CONCLUSION

We have revisited the method for approximate inference in probabilistic neural networks that takes into account all sources of stochasticity analytically. The latent variable interpretation allows a transparent interpretation of standard propagation in NNs as the simplest approximation and the development of variance propagating approximations. We proposed new approximations to LReLU max and argmax functions. This allows analytic propagation in max pooling layers and softmax layer.

We measured the quality of the approximation of posterior. The accuracy is improved compared to standard propagation and is sufficient for several use cases such as estimating statistics over the dataset (normalization) and dropout training, where we report improved test likelihoods. We identified that the weak point of the approximation is the factorization assumption. While modeling correlations is possible (e.g. Rezende & Mohamed, 2015), it is also more expensive and we showed that a calibration of the cheap methods can give a significant improvement and is a direction for further research. Except as a final layer, argmax and softmax may occur also inside the network, in models such as capsules (Sabour et al., 2017) or multiple hypothesis (Ilg et al., 2018), etc. Further applications of the developed technique may include generative and semi-supervised learning and Bayesian model estimation.

# REFERENCES

Ramn Fernández Astudillo and Joo Paulo da Silva Neto. Propagation of uncertainty through multilayer perceptrons for robust automatic speech recognition. In *INTERSPEECH*, 2011.

Anirban DasGupta, S.N. Lahiri, and Jordan Stoyanov. Sharp fixed n bounds and asymptotic expansions for the mean and the median of a Gaussian sample maximum, and applications to the DonohoJin model. Statistical Methodology, 20:40-62, 2014.  
Alhussein Fawzi, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard. Robustness of classifiers: from adversarial to random noise. In NIPS, pp. 1632-1640. 2016.  
Brendan J. Frey and Geoffrey E. Hinton. Variational learning in nonlinear Gaussian belief networks. Neural Comput., 11(1):193-213, January 1999.  
Yarin Gal and Zoubin Ghahramani. Bayesian convolutional neural networks with Bernoulli approximate variational inference. arXiv:1506.02158, 2015.  
Jochen Gast and Stefan Roth. Lightweight probabilistic deep networks. CoRR, abs/1805.11327, 2018.  
Soumya Ghosh, Francesco Maria Delle Fave, and Jonathan S. Yedidia. Assumed density filtering methods for learning Bayesian neural networks. pp. 1589-1595, 2016.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. 2010.  
Jose Miguel Hernández-Lobato and Ryan P. Adams. Probabilistic backpropagation for scalable learning of Bayesian neural networks. In ICML, pp. 1861-1869, 2015.  
Eddy Ilg, Ozgun Cicek, Silvio Galesso, Aaron Klein, Osama Makansi, Frank Hutter, and Thomas Brox. Uncertainty estimates and multi-hypotheses networks for optical flow. In ECCV, 2018.  
Sergey Ioffe. Batch renormalization: Towards reducing minibatch dependence in batch-normalized models. CoRR, abs/1702.03275, 2017.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, volume 37, pp. 448-456, 2015.  
Alex Kendall and Yarin Gal. What uncertainties do we need in Bayesian deep learning for computer vision? In NIPS, 2017.  
Diederik P Kingma, Tim Salimans, and Max Welling. Variational dropout and the local reparameterization trick. In NIPS, pp. 2575-2583. 2015.  
Yann Lecun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. In Intelligent signal processing, pp. 306-351, 2001.  
Henrick J. Malik and Bovas Abraham. Multivariate logistic distributions. The Annals of Statistics, 1(3):588-590, 1973.  
Thomas P. Minka. Expectation propagation for approximate Bayesian inference. In Uncertainty in Artificial Intelligence, pp. 362-369, 2001.  
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Universal adversarial perturbations. In CVPR, July 2017.  
Saralees Nadarajah and Samuel Kotz. Exact distribution of the max/min of two Gaussian random variables. IEEE Trans. VLSI Syst., 16(2):210-212, 2008.  
Radford M. Neal. Connectionist learning of belief networks. Artif. Intell., 56(1):71-113, July 1992.  
Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. In ICML, pp. 1530-1538, 2015.  
Erik Rodner, Marcel Simon, Bob Fisher, and Joachim Denzler. Fine-grained recognition in the noisy wild: Sensitivity analysis of convolutional neural networks approaches. In BMVC, 2016.  
Andrew M. Ross. Computing bounds on the expected maximum of correlated normal variables. Methodology and Computing in Applied Probability, 12(1):111-138, Mar 2010.

Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. Dynamic routing between capsules. In NIPS, pp. 3856-3866. 2017.  
Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. Deep information propagation. CoRR, abs/1611.01232, 2016.  
Alexander Shekhotsov and Boris Flach. Normalization of neural networks using analytic variance propagation. In Computer Vision Winter Workshop, pp. 45-53, 2018.  
J.T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. Striving for simplicity: The all convolutional net. In ICLR (workshop track), 2015.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. JMLR, 15:1929-1958, 2014.  
Hao Wang, Xingjian SHI, and Dit-Yan Yeung. Natural-parameter networks: A class of probabilistic neural networks. In NIPS, pp. 118-126, 2016.  
Sida Wang and Christopher Manning. Fast dropout training. In ICML, pp. 118-126, 2013.  
Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3):229-256, May 1992.
