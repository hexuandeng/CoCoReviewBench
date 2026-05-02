# ESTIMATING INFORMATIVENESS OF SAMPLES WITH SMOOTH UNIQUE INFORMATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We define a notion of information that an individual sample provides to the training of a neural network, and we specialize it to measure both how much a sample informs the final weights and how much it informs the function computed by the weights. Though related, we show that these quantities have a qualitatively different behavior. We give efficient approximations of these quantities using a linearized network and demonstrate empirically that the approximation is accurate for real-world architectures, such as pre-trained ResNets. We apply these measures to several problems, such as dataset summarization, analysis of under-sampled classes, comparison of informativeness of different data sources, and detection of corrupted examples. Our work generalizes existing frameworks, but enjoys better computational properties for heavily over-parametrized models, which makes it possible to apply it to real-world networks.

# 1 INTRODUCTION

Training a deep neural network (DNN) entails extracting information from samples in a dataset and storing it in the weights of the network, so that it may be used in future inference or prediction. But how much information does a particular sample contribute to the trained model? The answer can be used to provide strong generalization bounds (if no information is used, the network is not memorizing the sample), privacy bounds (how much information the network can leak about a particular sample), and enable better interpretation of the training process and its outcome. To determine the information content of samples, we need to define and compute information. In the classical sense, information is a property of random variables, which may be degenerate for the deterministic process of computing the output of a trained DNN in response to a given input (inference). So, even posing the problem presents some technical challenges. But beyond technicalities, how can we know whether a given sample is memorized by the network and, if it is, whether it is used for inference?

We propose a notion of unique sample information that, while rooted in information theory, captures some aspects of stability theory and influence functions. Unlike most information-theoretic measures, ours can be approximated efficiently for large networks, especially in the case of transfer learning, which encompasses many real-world applications of deep learning. Our definition can be applied to either "weight space" or "function space." This allows us to study the non-trivial difference between information the weights possess (weight space) and the information the network actually uses to make predictions on new samples (function space).

Our method yields a valid notion of information without relying on the randomness of the training algorithm (e.g., stochastic gradient descent, SGD), and works even for deterministic training algorithms. Our main work-horse is a first-order approximation of the network. This approximation is accurate when the network is pre-trained (Mu et al., 2020) — as is common in practical applications — or is randomly initialized but very wide (Lee et al., 2019), and can be used to obtain a closed-form expression of the per-sample information. In addition, our method has better scaling with respect to the number of parameters than most other information measures, which makes it applicable to massively over-parametrized models such as DNNs. Our information measure can be computed without actually training the network, making it amenable to use in problems like dataset summarization.

We apply out method to remove a large portion of uninformative examples from a training set with minimum impact on the accuracy of the resulting model (dataset summarization). We also apply our method to detect unlabeled samples, which we show carry more unique information.

To summarize, our contributions are (1) We introduce a notion of unique information that a sample contributes to the training of a DNN, both in weight space and in function space, and relate it with stability of the training algorithm; (2) We provide an efficient method to compute unique information even for large networks using a linear approximation of the DNN, and without having to train a network; (3) We show applications to dataset summarization and analysis.

Prerequisites and Notation. Consider a dataset of  $n$  labeled examples  $S = \{z_i\}_{i=1}^n$ , where  $z_i = (x_i, y_i)$ ,  $x_i \in \mathcal{X}$  and  $y_i \in \mathbb{R}^k$  and a neural network model  $f_w: \mathcal{X} \mapsto \mathbb{R}^k$  with parameters  $w \in \mathbb{R}^d$ . Throughout the paper  $S_{-i} = \{z_1, \ldots, z_{i-1}, z_{i+1}, \ldots, z_n\}$  denotes the set excluding the  $i$ -th sample;  $f_{w_t}$  is often shortened to  $f_i$ ; the concatenation of all training examples is denoted by  $X$ ; the concatenation of all training labels by  $Y \in \mathbb{R}^{nk}$ ; and the concatenation of all outputs by  $f_w(X) \in \mathbb{R}^{nk}$ . The loss on the  $i$ -th example is denoted by  $\mathcal{L}_i(w)$  and is equal to  $\frac{1}{2} \|f_w(x) - y\|_2^2$  unless specified otherwise. This choice is useful when dealing with linearized models and is justified by Hui & Belkin (2020), who showed that the mean-squared error (MSE) loss is as effective as cross-entropy for classification tasks. The total loss is  $\mathcal{L}(w) = \sum_{i=1}^n \mathcal{L}_i(w) + \frac{\lambda}{2} \|w - w_0\|_2^2$ , where  $\lambda \geq 0$  is a weight decay regularization coefficient and  $w_0$  is the weight initialization point. Note that the regularization term differs from standard weight decay  $\|w\|_2^2$  and is more appropriate for linearized neural networks, as it allows us to derive the dynamics analytically (see Sec. F of the appendix). Finally, a (possibly stochastic) training algorithm is denoted with a conditional distribution  $A(w | S)$ , while its output random variable is denoted with  $A(S)$ . We use several information-theoretic quantities, such as entropy:  $H(X) = -\mathbb{E}\big[\log p(x)\big]$ , mutual information:  $I(X;Y) = H(X) + H(Y) - H(X,Y)$ , Kullback-Leibler divergence:  $\mathrm{KL}(p(x)|q(x)) = \mathbb{E}_{x \sim p(x)}[\log(p(x)/q(x))]$  and their conditional variants (Cover & Thomas, 2006). If  $y \in \mathbb{R}^m$  and  $x \in \mathbb{R}^n$ , then the Jacobian  $\frac{\partial y}{\partial x}$  is an  $m \times n$  matrix. The gradient  $\nabla_x y$  denotes transpose of the Jacobian  $\frac{\partial y}{\partial x}$ , an  $n \times m$  matrix.

# 2 RELATED WORK

Our work is related to information-theoretic stability notions (Bassily et al., 2016; Raginsky et al., 2016; Feldman & Steinke, 2018) that seek to measure the influence of a sample on the output, and to measure generalization. Raginsky et al. (2016) define information stability as  $\mathbb{E}_S\left[\frac{1}{n}\sum_{i=1}^n I(w;z_i|S_{-i})\right]$ , the expected average amount of unique (Shannon) information that weights have about an example. This, without the expectation over  $S$ , is also our starting point (eq. 1). Bassily et al. (2016) define KL-stability  $\sup_{S,S'}\mathrm{KL}(A(w|S)\parallel A(w|S'))$ , where  $S$  and  $S'$  are datasets that differ by one example, while Feldman & Steinke (2018) define average leave-one-out KL stability as  $\sup_{S}\frac{1}{n}\sum_{i=1}^{n}\mathrm{KL}(A(w|S)\parallel A(w|S_{-i}))$ . The latter closely resembles our definition (eq. 4). Unfortunately, while the weights are continuous, the optimization algorithm (such as SGD) is usually discrete. This generally makes the resulting quantities degenerate (infinite). Most works address this issue by replacing the discrete optimization algorithm with a continuous one, such as stochastic gradient Langevin dynamics (Welling & Teh, 2011) or continuous stochastic differential equations that approximate SGD (Li et al., 2017) in the limit. We aim to avoid such assumptions and give a definition that is directly applicable to real networks trained with standard algorithms. To do this, we apply a smoothing procedure to a standard discrete algorithm. The final result can still be interpreted as a valid bound on Shannon mutual information, but for a slightly modified optimization algorithm. Our definitions relate informativeness of a sample to the notion of algorithmic stability (Bousquet & Elisseeff, 2002; Hardt et al., 2015), where a training algorithm  $A$  is called stable if  $A(S)$  is close to  $A(S')$  when the datasets  $S$  and  $S'$  differ by only one sample.

To ensure our quantities are well-defined, we apply a smoothing technique which is reminiscent of a soft discretization of weight space. In Section 4, we show that a canonical discretization is obtained using the Fisher information matrix, which relates to classical results of Rissanen (1996) on optimal coding length. It also relates to the use of a post-distribution in Achille et al. (2019), who however use it to estimate the total amount of information in the weights of a network.

We use a first-order approximation (linearization) inspired by the Neural Tangent Kernel (NTK) (Jacot et al., 2018; Lee et al., 2019) to efficiently estimate informativeness of a sample. While NTK predicts that, in the limit of an infinitely wide network, the linearized model is an accurate approximation, we do not observe this on more realistic architectures and datasets. However, we show that, when using pre-trained networks as common in practice, linearization yields an accurate approximation, similarly to what is observed by Mu et al. (2020). Shwartz-Ziv & Alemi (2020) study the

total information contained by an ensemble of randomly initialized linearized networks. They notice that, while considering ensembles makes the mutual information finite, it still diverges to infinity as training time goes to infinity. On the other hand, we consider the unique information about a single example, without the need for ensembles, by considering smoothed information, which remains bounded for any time. Other complementary works study how information about an input sample propagates through the network (Shwartz-Ziv & Tishby, 2017; Achille & Soatto, 2018), rather than how much information the sample itself contains.

In terms of applications, our work is related to works that estimate influence of an example (Koh & Liang, 2017; Toneva et al., 2019; Katharopoulos & Fleuret, 2018; Ghorbani & Zou, 2019; Yoon et al., 2020). This can be done by estimating the change in weights if a sample is removed from the training set, which is addressed by several works (Koh & Liang, 2017; Golatkar et al., 2020; Wu et al., 2020). Influence functions (Cook, 1977; Koh & Liang, 2017) model removal of a sample as reducing its weight infinitesimally in the loss function, and show an efficient first-order approximation of its effect on other measures (such as test time predictions). We found influence functions to be prohibitively slow for the networks and data regimes we consider. Basu et al. (2020) found that influence functions are not accurate for large DNNs. Additionally, influence functions assume that the training has converged, which is not usually the case in practice. We instead use linearization of neural networks to estimate the effect of removing an example efficiently. We find that this approximation is accurate in realistic settings, and that the computational cost scales better with network size, making it applicable to very large neural networks.

# 3 UNIQUE INFORMATION OF A SAMPLE IN THE WEIGHTS

Consider a (possibly stochastic) training algorithm  $A(w \mid S)$  that, given a training dataset  $S$ , returns a (possibly degenerate) distribution over weights  $w$  of a classifier  $f_w$ . From an information-theoretic point of view, the amount of unique information sample  $z_i = (x_i, y_i)$  provides about the weights is given by the conditional point-wise mutual information:

$$
I (w; Z _ {i} = z _ {i} \mid \mathbf {S} _ {- i} = S _ {- i}) = \mathrm {K L} (A (w \mid \mathbf {S} = S) \| m (w \mid \mathbf {S} _ {- i} = S _ {- i})), \tag {1}
$$

where  $\mathbf{S}$  denotes the random variable whose sample is the particular dataset  $S$ , and  $m(w \mid \mathbf{S}_{-i} = S_{-i}) = \mathbb{E}_{z_i' \sim p(z)}[A(w \mid \mathbf{S} = S_{-i}, z_i')]$  is the distribution of the weights over all possible sampling of  $Z_i$ . Computing the distribution  $m(w \mid S_{-i})$  is challenging because of the high-dimensionality and the cost of training algorithm  $A(w \mid S)$  for multiple samples. One can address this problem by using the following upper bound:

$$
\begin{array}{l} \operatorname {K L} (A (w \mid S) \| m (w \mid S _ {- i})) = \operatorname {K L} (A (w \mid S) \| q (w \mid S _ {- i})) - \operatorname {K L} (m (w \mid S _ {- i}) \| q (w \mid S _ {- i})) \\ \leq \mathrm {K L} (A (w \mid S) \| q (w \mid S _ {- i})), \tag {2} \\ \end{array}
$$

which is valid for any distribution  $q(w \mid S_{-i})$ . Choosing  $q(w \mid S_{-i}) = A(w \mid S_{-i})$ , the distribution of the weights after training on  $S_{-i}$ , gives a reasonable upper bound (see Sec. A.1 for details):

$$
I (w; z _ {i} \mid S _ {- i}) \leq \mathrm {K L} (A (w \mid S) \| A (w \mid S _ {- i})). \tag {3}
$$

We call  $\mathrm{SI}(z_i,A)\triangleq \mathrm{KL}(A(w\mid S)\parallel A(w\mid S_{-i}))$  the sample information of  $z_{i}$  w.r.t. algorithm  $A$

Smoothed Sample Information. The formulation above is valid in theory, but in practice even SGD is used in a deterministic fashion by fixing the random seed, and in the end we obtain just one set of weights rather than a distribution of them. Under these circumstances, all the above KL divergences are degenerate, as they evaluate to infinity. It is common to address the problem by assuming that  $A$  is a continuous stochastic optimization algorithm, such as stochastic gradient Langevin dynamics (SGLD) or a continuous approximation of SGD which adds Gaussian noise to the gradients. However, this creates a disconnect with the practice, where such approaches do not perform at the state-of-the-art. Our definitions below aim to overcome this disconnect.

Definition 3.1 (Smooth sample information). Let  $A$  be a possibly stochastic algorithm. Following eq. (3), we define the smooth sample information with smoothing  $\Sigma$ :

$$
\boxed {\mathrm {S I} _ {\Sigma} (z _ {i}, A) = \mathrm {K L} (A _ {\Sigma} (w \mid S) \| A _ {\Sigma} (w \mid S _ {- i})).} \tag {4}
$$

where we define smoothed weights  $A_{\Sigma}(S)\triangleq A(S) + \xi$  with  $\xi \sim \mathcal{N}(0,\Sigma)$ .

Note that if the algorithm  $A$  is continuous, we can pick  $\Sigma \to 0$ , which will make  $\mathrm{SI}_{\Sigma}(z_i, A) \to \mathrm{SI}(z_i, A)$ . The following proposition shows how to compute the value of  $\mathrm{SI}_{\Sigma}$  in practice.

Proposition 3.2. Let  $A$  be a deterministic training algorithm. Then, we have:

$$
\mathrm {S I} _ {\Sigma} \left(z _ {i}, A\right) = \frac {1}{2} \left(w - w _ {- i}\right) ^ {T} \Sigma^ {- 1} \left(w - w _ {- i}\right), \tag {5}
$$

where  $w = A(S)$  and  $w_{-i} = A(S_{-i})$  are the weights obtained by training respectively with and without the training sample  $z_{i}$ . That is, the value of  $\mathrm{SI}_{\Sigma}(z_i)$  depends on the distance between the solutions obtained training with and without the sample  $z_{i}$ , rescaled by  $\Sigma$ .

The smoothing of the weights by a matrix  $\Sigma$  can be seen as a form of soft-discretization. Rather than simply using an isotropic discretization  $\Sigma = \sigma^2 I -$  since different filters have different norms and/or importance for the final output of the network - it makes sense to discretize them differently. In Sections 4 and 5 we show two canonical choices for  $\Sigma$ . One is the inverse of the Fisher information matrix, which discounts weights not used for classification, and the other is covariance of the steady-state distribution of SGD, which respects the level of SGD noise and flatness of the loss.

# 4 UNIQUE INFORMATION IN THE PREDICTIONS

$\mathrm{SI}_{\Sigma}(z_i, A)$  measures how much information an example  $z_i$  provides to the weights. Alternatively, instead of working in weight-space, we can approach the problem in function-space, and measure the informativeness of a training example for the network outputs or activations. The unique information that  $z_i$  provides to the predictions on a test example  $x$  is:

$$
I \left(z _ {i}; \widehat {y} \mid x, S _ {- i}\right) = \mathbb {E} _ {S} \operatorname {K L} \left(q (\widehat {y} \mid x, S) \| m (\widehat {y} \mid x, S _ {- i})\right),
$$

where  $x \sim p(x)$  is a previously unseen test sample,  $\widehat{y} \sim q(\cdot \mid x, S)$  is the network output on input  $x$  after training on  $S$ , and  $m(\widehat{y} \mid x, S_{-i}) = \mathbb{E}_{z_i'} q(\widehat{y} \mid x, S_{-i}, z_i')$ . Following the reasoning in the previous section, we arrive at

$$
I \left(z _ {i}; \widehat {y} \mid x, S _ {- i}\right) \leq \mathrm {K L} \left(q (\widehat {y} \mid x, S) \| q (\widehat {y} \mid x, S _ {- i})\right). \tag {6}
$$

Again, when training with a discrete algorithm and/or when the output of the network is deterministic, the above quantity may be infinite. Similar to smooth sample information, we define:

Definition 4.1 (Smooth functional sample information). Let  $A$  be a possibly stochastic training algorithm and let  $\widehat{y}$  be the prediction on example  $x$  after training on  $S$ . We define the smooth functional sample information (F-SI) as:

$$
\boxed {\mathrm {F} - \mathrm {S I} _ {\sigma} (z _ {i}, A) = \mathbb {E} _ {S} [ \mathrm {K L} (q _ {\sigma} (\widehat {y} _ {\sigma} \mid x, S) \| q _ {\sigma} (\widehat {y} _ {\sigma} \mid x, S _ {- i})) ]}, \tag {7}
$$

where  $\widehat{y}_{\sigma} = \widehat{y} (x,S) + n$  with  $n\sim \mathcal{N}(0,\sigma^2 I)$  and  $q_{\sigma}(\widehat{y}_{\sigma}\mid x,S)$  being the distribution of  $\widehat{y}_{\sigma}$ .

We now describe a first-order approximation of the value of  $\mathrm{F - SI}_{\Sigma}$  for deterministic algorithms.

Proposition 4.2. Let  $A$  be a deterministic algorithm,  $w = A(S)$  and  $w_{-i} = A(S_{-i})$  be the weights obtained training respectively with and without sample  $z_{i}$ . Then,

$$
\begin{array}{l} \operatorname {F} - \operatorname {S I} _ {\sigma} \left(z _ {i}, A\right) = \frac {1}{2 \sigma^ {2}} E _ {x \sim p (x)} \| f _ {w} (x) - f _ {w _ {- i}} (x) \| _ {2} ^ {2} (8) \\ \approx \frac {1}{2 \sigma^ {2}} (w - w _ {- i}) ^ {T} F (w) (w - w _ {- i}), (9) \\ \end{array}
$$

with  $F(w) = \mathbb{E}_x\left[\nabla_w f_w(x)\nabla_w f_w(x)^T\right]$  being the Fisher information matrix of  $q_{\sigma = 1}(\widehat{y}\mid x,S)$ .

By comparing eq. (5) and eq. (9), we see that the functional sample information is approximated by using the inverse of the Fisher information matrix to smooth the weight space. However, this smoothing is not isotropic as it depends on the point  $w$ .

# 5 EXACT SOLUTION FOR LINEARIZED NETWORKS

In this section we derive a close-form expression for  $\mathrm{SI}_{\Sigma}$  and F-SI using a linear approximation of the network around the initial weights. We show that this approximation can be computed efficiently and, as we validate empirically in Sec. 6, correlates well with the actual informativeness values. We also show that the covariance matrix of SGD's steady-state distribution is a canonical choice for the smoothing matrix  $\Sigma$  of  $\mathrm{SI}_{\Sigma}$ .

Linearized Network. Linearized neural networks are a class of neural networks obtained by taking the first-order Taylor expansion of a DNN around the initial weights (Lee et al., 2019):

$$
f _ {w} ^ {\mathrm {l i n}} (x) \triangleq f _ {w _ {0}} (x) + \nabla_ {w} f _ {w} (x) ^ {T} | _ {w = w _ {0}} (w - w _ {0}).
$$

These networks are linear with respect to their parameters  $w$ , but can be highly non-linear with respect to their input  $x$ . One of the advantages of linearized neural networks is that the dynamics of continuous-time or discrete-time gradient descent can be written analytically if the loss function is the mean squared error (MSE). In particular, for continuous-time gradient descent with constant learning rate  $\eta > 0$ , we have (Lee et al., 2019):

$$
w _ {t} = \nabla_ {w} f _ {0} (X) \Theta_ {0} ^ {- 1} \left(I - e ^ {- \eta \Theta_ {0} t}\right) \left(f _ {0} (X) - Y\right), \tag {10}
$$

$$
f _ {t} ^ {\ln} (x) = f _ {0} (x) + \Theta_ {0} (x, X) \Theta_ {0} ^ {- 1} \left(I - e ^ {- \eta \Theta_ {0} t}\right) (Y - f _ {0} (X)), \tag {11}
$$

where  $\Theta_0 = \nabla_w f_0(X)^T\nabla_w f_0(X) \in \mathbb{R}^{nk \times nk}$  is the Neural Tangent Kernel (NTK) (Jacot et al., 2018; Lee et al., 2019) and  $\Theta_0(x, X) = \nabla_w f_0(x)^T\nabla_w f_0(X)$ . The expressions for networks trained with weight decay is essentially the same (see Sec. F). To keep the notation simple, we will use  $f_w(x)$  to indicate  $f_w^{\mathrm{lin}}(x)$  from now on.

Stochastic Gradient Descent. As mentioned in Sec. 3, a popular alternative approach to make information quantities well-defined is to use continuous-time SGD, which is defined by (Li et al., 2017; Mandt et al., 2017):

$$
d w _ {t} = - \eta \nabla_ {w} \mathcal {L} _ {w} \left(w _ {t}\right) d t + \eta \sqrt {\frac {1}{b} \Lambda \left(w _ {t}\right)} d n (t), \tag {12}
$$

where  $\eta$  is the learning rate,  $b$  is the batch size,  $n(t)$  is a Brownian motion, and  $\Lambda(w_{t})$  is the covariance matrix of the per-sample gradients (see Sec. C for details). Let  $A_{\mathrm{SGD}}(w \mid S)$  be the steady-state distribution of (12), and let  $A_{\mathrm{ERM}}$  be the deterministic algorithm that returns the global minimum  $w^{*}$  of the loss  $\mathcal{L}(w)$  (for a regularized linearized network  $\mathcal{L}(w)$  is strictly convex). We now show that the non-smooth sample information  $\mathrm{SI}(z_i, A_{\mathrm{SGD}})$  is the same as the smooth sample information using SGD's steady-state covariance as the smoothing matrix and  $A_{\mathrm{ERM}}$  as the training algorithm.

Proposition 5.1. Let the loss function be regularized MSE,  $w^{*}$  be the global minimum of it, and algorithms  $A_{SGD}$  and  $A_{ERM}$  be defined as above. Assuming  $\Lambda(w)$  is approximately constant around  $w^{*}$  and SGD's steady-state covariance remains constant after removing an example, we have

$$
\mathrm {S I} \left(z _ {i}, A _ {S G D}\right) = \mathrm {S I} _ {\Sigma} \left(z _ {i}, A _ {E R M}\right) = \frac {1}{2} \left(w ^ {*} - w _ {- i} ^ {*}\right) ^ {T} \Sigma^ {- 1} \left(w ^ {*} - w _ {- i} ^ {*}\right), \tag {13}
$$

where  $\Sigma$  is the solution of

$$
H \Sigma + \Sigma H ^ {T} = \frac {\eta}{b} \Lambda \left(w ^ {*}\right), \tag {14}
$$

with  $H = (\nabla_w f_0(X) \nabla_w f_0(X)^T + \lambda I)$  being the Hessian of the loss function.

This proposition motivates the use of SGD's steady-state covariance as smoothing matrix. From equations (13) and (14) we see that SGD's steady-state covariance is proportional to the flatness of the loss at the minimum, the learning rate, and to SGD's noise, while inversely proportional to the batch size. When  $H$  is positive definite, as in our case when using weight decay, the continuous Lyapunov equation (14) has a unique solution, which can be found in  $O(d^3)$  time using the Bartels-Stewart algorithm (Bartels & Stewart, 1972). One particular case when the solution can be found analytically is when  $\Lambda(w^{*})$  and  $H$  commute, in which case  $\Sigma = \frac{\eta}{2b}\Lambda H^{-1}$ . For example, this is the case for Langevin dynamics, for which  $\Lambda(w) = \sigma^2 I$  in equation (12). In this case, we have

$$
\mathrm {S I} \left(z _ {i}, A _ {\mathrm {S G D}}\right) = \mathrm {S I} _ {\Sigma} \left(z _ {i}, A _ {\mathrm {E R M}}\right) = \frac {b}{\eta \sigma^ {2}} \left(w ^ {*} - w _ {- i} ^ {*}\right) ^ {T} H \left(w ^ {*} - w _ {- i} ^ {*}\right), \tag {15}
$$

which was already suggested by Cook (1977) as a way to measure the importance of a datum in linear regression.

Functional Sample Information. The definition in Section 4 simplifies for linearized neural networks: The step from eq. (8) to eq. (9) becomes exact, and the Fisher information matrix becomes independent of  $w$  and equal to  $F = \mathbb{E}_{x\sim p(x)}\left[\nabla_{w}f_{0}(x)\nabla_{w}f_{0}(x)^{T}\right]$ . This shows that functional sample information can be seen as weight sample information with discretization  $\Sigma$  equal to  $F^{-1}$ . The functional sample information depends on the training data distribution, which is usually unknown. We can estimate it using a validation set:

$$
\begin{array}{l} \mathrm {F} - \mathrm {S I} _ {\sigma} \left(z _ {i}, A\right) \approx \frac {1}{2 \sigma^ {2} n _ {\text {v a l}}} \sum_ {j = 1} ^ {n _ {\text {v a l}}} \left| \left| f _ {w} \left(x _ {j} ^ {\text {v a l}}\right) - f _ {w _ {- i}} \left(x _ {j} ^ {\text {v a l}}\right) \right| \right| (16) \\ = \frac {1}{2 \sigma^ {2} n _ {\mathrm {v a l}}} (w - w _ {- i}) ^ {T} \left(H _ {\mathrm {v a l}} - \lambda I\right) (w - w _ {- i}). (17) \\ \end{array}
$$

It is instructive to compare the sample weight information of (15) and functional sample information of (17). Besides the constants, the former uses the Hessian of the training loss, while the latter uses the Hessian of the validation loss (without the  $\ell_2$  regularization term). One advantage of the latter is computational cost: As demonstrated in the next section, we can use equation (16) to compute the prediction information, entirely in the function space, without any costly operation on weights. For this reason, we focus on the linearized F-SI approximation in our experiments. Since  $\sigma^{-2}$  is just a multiplicative factor in (17) we set  $\sigma = 1$ . We also focus on the case where the training algorithm  $A$  is discrete gradient descent running for  $t$  epochs (equations 10 and 11).

Efficient Implementation. To compute the proposed sample information measures for linearized neural networks, we need to compute the change in weights  $w - w_{-i}$  (or change in predictions  $f_{w}(x) - f_{w_{i}}(x)$ ) after excluding an example from the training set. This can be done without retraining using the analytical expressions of weight and prediction dynamics of linearized neural networks eq. (10) and eq. (11), which also work when the algorithm has not yet converged ( $t < \infty$ ). We now describe a series of measures to make the problem tractable. First, to compute the NTK matrix we would need to store the Jacobian  $\nabla f_{0}(x_{i})$  of all training points and compute  $\nabla_{w}f_{0}(X)^{T}\nabla_{w}f_{0}(X)$ . This is prohibitively slow and memory consuming for large DNNs. Instead, similarly to Zancato et al. (2020), we use low-dimensional random projections of per-example Jacobians to obtain provably good approximations of dot products (Achlioptas, 2003; Li et al., 2006). We found that just taking 2000 random weights coordinates per layer provides a good enough approximation of the NTK matrix. Importantly, we consider each layer separately, as different layers may have different gradient magnitudes. With this method, computing the NTK matrix takes  $O(nkd + n^{2}k^{2}d_{0})$  time, where  $d_0\approx 10^4$  is the number of sub-sampled weight indices ( $d_0\ll d$ ). We also need to recompute  $\Theta_0^{-1}$  after removing an example from the training set. This can be done in quadratic time by using rank-one updates of the inverse (see Sec. E). Finally, when  $t\neq \infty$  we need to recompute  $e^{-\eta \bar{\Theta}_0t}$  after removing an example. This can be done in  $O(n^{2}k^{2})$  time by downdating the eigendecomposition of  $\Theta_0$  (Gu & Eisenstat, 1995). Overall, the complexity of computing  $w - w_{i}$  for all training examples is  $O(n^{2}k^{2}d_{0} + n(n^{2}k^{2} + C))$ ,  $C$  is the complexity of a single pass over the training dataset. The complexity of computing functional sample information for  $m$  test samples is  $O(C + nmk^2 d_0 + n(mnk^2 + n^2 k^2))$ . This depends on the network size lightly, only through  $C$ .

# 6 EXPERIMENTS

In this section we test the validity of linearized network approximation in terms of estimating effects of removing an example and show several applications of the proposed information measures. Additional results and details are provided in the supplementary Sec. A.

Accuracy of the linearized network approximation. We measure  $\| w - w_{-i}\| _2^2$  and  $\| f_w(X_{\mathrm{val}}) - f_{w_{-i}}(X_{\mathrm{val}})\| _2^2$  for each sample  $z_{i}$  by training with and without that example. Then, instead of retraining, we use the efficient linearized approximation in Sec. 6 to estimate the same quantities and measure their correlation with the ground-truth values (Table 1). For comparison, we also estimate these quantities using influence functions (Koh & Liang, 2017). We consider two classification tasks: (a) a toy MNIST 4 vs 9 classification task and (b) Kaggle Dogs vs. Cats classification task (Kaggle, 2013), both with 1000 examples. For MNIST we consider a fully connected network with

Table 1: Pearson correlations of weight change  $\| w - w_{-i}\| _2^2$  and validation prediction change  $\| f_w(X_{\mathrm{val}}) - f_{w_{-i}}(X_{\mathrm{val}})\| _2^2$  norms computed with influence functions and linearized neural networks with their corresponding measures computed for standard neural networks with retraining.  

<table><tr><td></td><td>Reg.</td><td>Method</td><td>MNIST MLP</td><td colspan="2">MNIST CNN</td><td colspan="2">Cats and Dogs</td></tr><tr><td></td><td></td><td></td><td>scratch</td><td>scratch</td><td>pretrained</td><td>pr. ResNet-18</td><td>pr. ResNet-50</td></tr><tr><td rowspan="4">weights</td><td rowspan="2">λ = 0</td><td>Linearization</td><td>0.987</td><td>0.193</td><td>0.870</td><td>0.895</td><td>0.968</td></tr><tr><td>Infl. functions</td><td>0.935</td><td>0.319</td><td>0.736</td><td>0.675</td><td>0.897</td></tr><tr><td rowspan="2">λ = 103</td><td>Linearization</td><td>0.977</td><td>-0.012</td><td>0.964</td><td>0.940</td><td>0.816</td></tr><tr><td>Infl. functions</td><td>0.978</td><td>0.069</td><td>0.979</td><td>0.858</td><td>0.912</td></tr><tr><td rowspan="4">predictions</td><td rowspan="2">λ = 0</td><td>Linearization</td><td>0.993</td><td>0.033</td><td>0.875</td><td>0.877</td><td>0.895</td></tr><tr><td>Infl. functions</td><td>0.920</td><td>0.647</td><td>0.770</td><td>0.530</td><td>0.715</td></tr><tr><td rowspan="2">λ = 103</td><td>Linearization</td><td>0.993</td><td>0.070</td><td>0.974</td><td>0.931</td><td>0.519</td></tr><tr><td>Infl. functions</td><td>0.990</td><td>0.407</td><td>0.954</td><td>0.753</td><td>0.506</td></tr></table>

![](images/360994edae7661a9707b7d82872850b967dea0b2b773c8fa65039a0128ee6c80.jpg)  
B

![](images/8b7d15e99068d84e97c4356381f17f0edecd2fc56f2667ce0c38da8959b6ea60.jpg)

![](images/7aff67e460cb58debac8b62ddc7d099d86be449b8d3d37c8650cb240a84776e1.jpg)

![](images/b484fd0e11234af0fddfabe94a1e931adb032ad2783abd93853288a6f0533f6b.jpg)

![](images/c01f9af46d55f0a5da95536742f150f64902612d5cd7826280766681138e302b.jpg)

![](images/2bc03d35afb0389b7639334fa5d3cad2a27074b3d84be1079d991372624f7cf4.jpg)

![](images/b9eab760296096575564416a9a4f1541eb20e9eb2b641a1cd959cd24437b4de6.jpg)

![](images/eb7e59505b4e69e01f5e8b958e2d9cd0555a12d56b2955d576781f856ed3095e.jpg)

![](images/ccdff12bfb9f4ee77a90ebd90e5b6230e3b0d0322b3c9b9f5c9cca7f37047f27.jpg)

![](images/dd4a26a5542dd3ff8d67545dfaef886fbe3fce3aacf8e71703a08524fb655f33.jpg)

![](images/157bd5fbf2c3894faf99671918b90e55bdb5d9f67e62f47ec9095afb9cf455ff.jpg)

![](images/251562dd630a67f521c7d0967a83cd3b74189411c6e6c2d9d00647763d7a0125.jpg)

![](images/35fac7592af0c17e6ef92cc889b182941ed1118f8089a591fc998288e99f68e3.jpg)

![](images/e22a8dbf812466715137eb059df3b42ec8d70ce785a548113d5da6907ee9c651.jpg)

![](images/3c6662a42f446e5c3c4664d1e40a9a0de7ad4a219f910a1ae67f0289175ffad3.jpg)

![](images/74d16d105d752724e7f60a630ae51652500109e7391e93e40cb17ec13b5616b2.jpg)

![](images/97dbc29691845ababa9e25d5d833ac64483d451ba2e7005866df26d9c7ae10ba.jpg)

![](images/79f25b1b3ec1d6fbd080145ccf1a89efa6ebaa652f7a31966dd096885293356e.jpg)  
Figure 1: Functional sample information of samples in the iCassava classification task with 1000 samples, where the network is a pretrained ResNet-18. A: histogram of sample informations, B: 10 least informative samples, C: 10 most informative samples.  
C

![](images/ec43ff5752615bd528c0bc275339f47dc1df7f99dd05c759b875f920226db555.jpg)

![](images/47e24fd5c0fa21fe0477728ecd74d2daeb9ab22491ccbed4ca42d1a76dbd7b63.jpg)

![](images/5525815445add0cce8092689c5642383b7e16cb4ba43f6f2139706f1249514a0.jpg)

![](images/ded3404da9b1661a014ba53d6c328759d24bd1551c685ffcca6ef1cf922f4f2b.jpg)

![](images/409fd2f59fd6f1199007a306d434818417184e46060780c9f68c9707a6a052c5.jpg)

![](images/9efbb6fc0a7b1292c429dca9e73ed7f7c55931aacc39ef7eaf2ad3e85bb11a76.jpg)

![](images/9deb40be2b2140ddbd1cbd06469169592a5294ef2e066951e8db2fa7c04e0dc1.jpg)

![](images/6c59e95bae5d83bfcd6bbf305d1c6b26e09994db0b4f94c5df58c4d234206ebe.jpg)

![](images/bc067c7f90d9d7fe8a871562af52f102d3290bcba40d83d96d7af1246b0da411.jpg)

![](images/c8a05e082e846ffb60aa3911a0981262f617a2740904446122fc68fe5aa3aa17.jpg)

![](images/1b3d7061143eeb6f4503ddb68386ab815690cb34510938a96ff285832bcdb6bf.jpg)

![](images/4676887dae4ac682db6e014125f0228ddf50652b399236497b8242e78957033a.jpg)

![](images/c86560245cc925e1f729fe9d201e4d36b2a1935f33b83a7af20846395331e9f7.jpg)

![](images/95598c1837c84e7f5ed87fe1f92cf91479566367c6cc7c00a2a273dcfa7b2105.jpg)

a single hidden layer of 1024 ReLU units (MLP) and a small 4-layer convolutional network (CNN), either trained from scratch or pretrained on EMNIST letters (Cohen et al., 2017). For cats vs dogs classification, we consider ResNet-18 and ResNet-50 networks (He et al., 2016) pretrained on ImageNet. In both tasks, we train both with and without weight decay ( $\ell_2$  regularization). The results in Table 1 shows that linearized approximation correlates well with ground truth when the network is wide enough (MLP) and/or pretraining is used (CNN with pretraining and pretrained ResNets). This is expected, as wider networks can be approximated with linearized ones better (Lee et al., 2019), and pretraining decreases the distance from initialization, making the Taylor approximation more accurate. Adding regularization also keeps the solution close to initialization, and generally increases the accuracy of the approximation. Furthermore, in most cases linearization gives better results compared to influence functions, while also being around 30 times faster in our settings.

Which examples are informative? Fig. 1, and Fig. 4 of the supplementary, plot the top 10 least and most important samples in iCassava plant disease classification (Mwebaze et al., 2019), the MNIST 4 vs 9, and Kaggle cats vs dogs classification tasks. Especially in the case of the last two, we see that the least informative samples look typical and easy, while the most informative ones look more challenging and atypical. In case of iCassava, the informative samples are more zoomed on features that are important for classification (e.g., the plant disease spots). We observe that most samples have small unique information, possibly because they are easier or because the dataset may have many similar-looking examples. While in the case of MNIST 4 vs 9 and Cats vs. Dogs, the two classes have on average similar information scores, in Fig. 1a we see that in iCassava examples from rare classes (such as 'healty' and 'cbb') are on average more informative.

Which data source is more informative? In this experiment we train for 10-way digit classification where both the training and validation sets consists of 1000 samples, 500 from MNIST and 500 from SVHN. We compute functional sample information for a pretrained ResNet-18 network. The results presented in Fig. 2a tell that SVHN examples are much more informative than MNIST examples. This is intuitive, as SVHN examples have more variety. One can go further and use our information measures for estimating informativeness of examples of dataset  $A$  for training a classifier for task  $B$ , similar to Torralba & Efros (2011).

Detecting mislabeled examples. We expect a mislabeled example to carry more unique information, since the network needs to memorize unique features of that particular example to classify it.

![](images/9ac1d33db2526a3d8d033a0186b5362533c1d5bc2a509fb55cd5eb97ffea3323.jpg)  
(a) Comparing data sources

![](images/eaef046a65e6f4391ce142ae185745cc4bda3b4c0faa0bd44ef5817ca7f60563.jpg)  
(b) Detecting mislabeled examples

![](images/4c52ab637d5d3cf5799fbed1583470bc4e52f29dd3deb32734fc51d97f3d338b.jpg)  
Figure 2: Applications of functional sample information. (a) Different source of data for the same task (digit classification) can have vastly different amount of information. (b) As expected, samples with wrong labels carry more unique information. (c) Test accuracy as a function of the ratio of removed training examples using different strategies.  
(c) Summarizing datasets

To test this, we add  $10\%$  uniform label noise to MNIST 4 vs 9, Kaggle cats vs dogs, and iCassava classification tasks (all with 1000 examples in total), while keeping the validation sets clean. Fig. 2b plots the histogram of functional sample information for both correct and mistrabeled examples, while Fig. 6a and 6b plot that for MNIST 4 vs 9 and Kaggle cats vs dogs tasks, respectively. The results indicate that mistrabeled examples are much more informative on average. This suggests that our information measures can be used to detect outliers or corrupted examples.

Data summarization. We subsample the training dataset by removing a fraction of the least informative examples and measure the test performance of the resulting model. We expect that removing the least informative training samples should not affect the performance of the model. Note however that, since we are considering the unique information, removing one sample can increase the informativeness of another. For this reason, we consider two strategies: In one we compute the informativeness scores once, and remove a given percentage of the least informative samples. In the other we remove  $5\%$  of the least informative samples, recompute the scores, and iterate until we remove the desired number of samples. For comparison, we also consider removing informative examples ("top" baseline) and randomly selected examples ("random" baseline). The results on MNIST 4 vs 9 are shown in Fig. 2c. Indeed, removing the least informative training samples has little effect on the test error, while removing the top examples has the most impact. Also, recomputing the information scores after each removal steps ("bottom iterative") greatly improves the performance when many samples are removed, confirming that SI and F-SI are good practical measures of unique information in a sample, but also that the total information in a large group is not simply the sum of the unique information of its samples.

Detecting under-sampled sub-classes. Using CIFAR-10 images, we create a dataset of "Pets vs Deer": Pets has 4200 samples and deer 4000. The class pets consists of two unlabeled sub-classes, cats (200) and dogs (4000). Since there are relatively few cat images, we expect each to carry more unique information. Indeed, Fig. 7 in the appendix shows that this is the case, suggesting that the F-SI can help detecting when an unlabeled sub-class of a larger class is under-sampled.

# 7 CONCLUSION

There are many notions of information that are relevant to understanding the inner workings of neural networks. Recent efforts have focused on defining information in the weights or activations that do not degenerate for deterministic training. We look at the information in the training data, which ultimately affects both the weights and the activations. In particular, we focus on the most elementary case, which is the unique information contained in a sample, because it can be the foundation for understanding more complex notions. However, our approach can be readily generalized to unique information of a group of samples. Unlike most previously introduced information measures, ours is tractable even for real datasets used to train standard network architectures, and does not require restriction to limiting cases. In particular, we can approximate our quantities without requiring the limit of small learning rate (continuous training time), or the limit of infinite network width. Much remains to be done: One can consider common information, high-order information, synergistic information and other notions of information between samples. The relation among these quantities is complex in general, even for 3 variables (Williams & Beer, 2010) and is an open challenge.

# REFERENCES

Alessandro Achille and Stefano Soatto. Emergence of invariance and disentanglement in deep representations. The Journal of Machine Learning Research, 19(1):1947-1980, 2018.  
Alessandro Achille, Giovanni Paolini, and Stefano Soatto. Where is the information in a deep neural network? arXiv preprint arXiv:1905.12213, 2019.  
Dimitris Achlioptas. Database-friendly random projections: Johnson-lindenstrauss with binary coins. Journal of computer and System Sciences, 66(4):671-687, 2003.  
R. H. Bartels and G. W. Stewart. Solution of the matrix equation  $ax + xb = c$  [f4]. Commun. ACM, 15(9):820-826, September 1972. ISSN 0001-0782. doi: 10.1145/361573.361582. URL https://doi.org/10.1145/361573.361582.  
Raef Bassily, Kobbi Nissim, Adam Smith, Thomas Steinke, Uri Stemmer, and Jonathan Ullman. Algorithmic stability for adaptive data analysis. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing, pp. 1046-1059, 2016.  
Samyadeep Basu, Philip Pope, and Soheil Feizi. Influence functions in deep learning are fragile. arXiv preprint arXiv:2006.14651, 2020.  
Olivier Bousquet and André Elisseeff. Stability and generalization. Journal of machine learning research, 2(Mar):499-526, 2002.  
Gregory Cohen, Saeed Afshar, Jonathan Tapson, and André van Schaik. Emmist: an extension of mnist to handwritten letters. arXiv preprint arXiv:1702.05373, 2017.  
R Dennis Cook. Detection of influential observation in linear regression. Technometrics, 19(1): 15-18, 1977.  
Thomas M Cover and Joy A Thomas. Elements of information theory. Wiley-Interscience, 2006.  
Vitaly Feldman and Thomas Steinke. Calibrating noise to variance in adaptive data analysis. In Conference On Learning Theory, pp. 535-544, 2018.  
Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning. volume 97 of Proceedings of Machine Learning Research, pp. 2242-2251, Long Beach, California, USA, 09-15 Jun 2019. PMLR.  
Aditya Golatkar, Alessandro Achille, and Stefano Soatto. Forgetting outside the box: Scrubbing deep networks of information accessible from input-output observations. Proceedings of the European Conference on Computer Vision (ECCV), 2020.  
Ming Gu and Stanley C Eisenstat. Downdating the singular value decomposition. SIAM Journal on Matrix Analysis and Applications, 16(3):793-810, 1995.  
Moritz Hardt, Benjamin Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. arXiv preprint arXiv:1509.01240, 2015.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Like Hui and Mikhail Belkin. Evaluation of neural architectures trained with square loss vs cross-entropy in classification tasks. arXiv preprint arXiv:2006.07322, 2020.  
Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems 31, pp. 8571-8580. Curran Associates, Inc., 2018.  
Kaggle. Dogs vs. Cats, 2013. URL https://www.kaggle.com/c/dogs-vs-cats/overview.

A. Katharopoulos and F. Fleuret. Not all samples are created equal: Deep learning with importance sampling. In Proceedings of the International Conference on Machine Learning (ICML), pp. 2525-2534, 2018.  
Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 1885-1894, International Convention Centre, Sydney, Australia, 06-11 Aug 2017. PMLR.  
Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. In Advances in neural information processing systems, pp. 8572-8583, 2019.  
Ping Li, Trevor J Hastie, and Kenneth W Church. Very sparse random projections. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 287-296, 2006.  
Qianxiao Li, Cheng Tai, and E Weinan. Stochastic modified equations and adaptive stochastic gradient algorithms. In International Conference on Machine Learning, pp. 2101-2110, 2017.  
Stephan Mandt, Matthew D Hoffman, and David M Blei. Stochastic gradient descent as approximate bayesian inference. The Journal of Machine Learning Research, 18(1):4873-4907, 2017.  
Fangzhou Mu, Yingyu Liang, and Yin Li. Gradients as features for deep representation learning. arXiv preprint arXiv:2004.05529, 2020.  
Ernest Mwebaze, Timnit Gebru, Andrea Frome, Solomon Nsumba, and Jeremy Tusubira. icassava 2019 fine-grained visual categorization challenge, 2019.  
Maxim Raginsky, Alexander Rakhlin, Matthew Tsao, Yihong Wu, and Aolin Xu. Information-theoretic analysis of stability and bias of learning algorithms. In 2016 IEEE Information Theory Workshop (ITW), pp. 26-30. IEEE, 2016.  
J. J. Rissanen. Fisher information and stochastic complexity. IEEE Transactions on Information Theory, 42(1):40-47, Jan 1996. ISSN 0018-9448. doi: 10.1109/18.481776.  
Ravid Shwartz-Ziv and Alexander A Alemi. Information in infinite ensembles of infinitely-wide neural networks. volume 118 of Proceedings of The 2nd Symposium on Advances in Approximate Bayesian Inference, pp. 1-17. PMLR, 08 Dec 2020.  
Ravid Shwartz-Ziv and Naftali Tishby. Opening the black box of deep neural networks via information, 2017.  
Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J. Gordon. An empirical study of example forgetting during deep neural network learning. In International Conference on Learning Representations, 2019.  
Antonio Torralba and Alexei A Efros. Unbiased look at dataset bias. In CVPR 2011, pp. 1521-1528. IEEE, 2011.  
Max Welling and Yee W Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pp. 681-688, 2011.  
Paul L. Williams and Randall D. Beer. Nonnegative decomposition of multivariate information. CoRR, abs/1004.2515, 2010.  
Yinjun Wu, Edgar Dobriban, and Susan B Davidson. Deltagrad: Rapid retraining of machine learning models. arXiv preprint arXiv:2006.14755, 2020.  
Jinsung Yoon, Sercan Arik, and Tomas Pfister. Data valuation using reinforcement learning. 2020.  
Luca Zancato, Alessandro Achille, Avinash Ravichandran, Rahul Bhotika, and Stefano Soatto. Predicting training time without training. Advances in Neural Information Processing Systems 33, 2020.
