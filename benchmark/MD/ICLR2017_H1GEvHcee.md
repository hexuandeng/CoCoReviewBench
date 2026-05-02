# ANNEALING GAUSSIAN INTO RELU: A NEW SAMPLING STRATEGY FOR LEAKY-RELU RBM

Chun-Liang Li Siamak Ravanbakhsh Barnabás Póczos

Department of Machine Learning

Carnegie Mellon University

Pittsburgh, PA 15213, USA

{chunlial,mravanba,bapoczos}@cs.cmu.edu

# ABSTRACT

Restricted Boltzmann Machine (RBM) is a bipartite graphical model that is used as the building block in energy-based deep generative models. Due to numerical stability and quantifiability of the likelihood, RBM is commonly used with Bernoulli units. Here, we consider an alternative member of exponential family RBM with leaky rectified linear units - called leaky RBM. We first study the joint and marginal distributions of leaky RBM under different leakiness, which provides us important insights by connecting the leaky RBM model and truncated Gaussian distributions. The connection leads us to a simple yet efficient method for sampling from this model, where the basic idea is to anneal the leakiness rather than the energy; - i.e., start from a fully Gaussian/Linear unit and gradually decrease the leakiness over iterations. This serves as an alternative to the annealing of the temperature parameter and enables numerical estimation of the likelihood that are more efficient and more accurate than the commonly used annealed importance sampling (AIS). We further demonstrate that the proposed sampling algorithm enjoys faster mixing property than contrastive divergence algorithm, which benefits the training without any additional computational cost.

# 1 INTRODUCTION

In this paper, we are interested in deep generative models. There is a family of directed deep generative models which can be trained by back-propagation (e.g., Kingma & Welling, 2013; Goodfellow et al., 2014). The other family is the deep energy-based models, including deep belief network (Hinton et al., 2006) and deep Boltzmann machine (Salakhutdinov & Hinton, 2009). The building block of deep energy-based models is a bipartite graphical model called restricted Boltzmann machine (RBM). The RBM model consists of two layers, visible and hidden layers, which can model higher-order correlation of the visible units (visible layer) using the hidden units (hidden layer). It also makes the inference easier that there are no interactions between the variables in each layer.

The conventional RBM uses Bernoulli units for both the hidden and visible units (Smolensky, 1986). One extension is using Gaussian visible units to model general natural images (Freund & Haussler, 1994). For hidden units, we can also generalize Bernoulli units to the exponential family (Welling et al., 2004; Ravanbakhsh et al., 2016).

Nair & Hinton (2010) propose one special case by using Rectified Linear Unit (ReLU) for the hidden layer with the heuristic sampling procedure, which has promising performance in terms of reconstruction error and classification accuracy. Unfortunately, due to its lack of strict monotonicity, ReLU RBM does not fit within the framework of exponential family RBMs (Ravanbakhsh et al., 2016). Instead we study leaky-ReLU RBM (leaky RBM) in this work and address two important issues i) a better training (sampling) algorithm for ReLU RBM and; ii) a better quantification of leaky RBM-i.e., its performance in terms of likelihood.

We study some of the fundamental properties of leaky RBM, including its joint and marginal distributions (Section 2). By analyzing these distributions, we show that the leaky RBM is a union of truncated Gaussian distributions. In this paper we will show that training leaky RBM involves underlying positive definite constraints. Because of this, the training can diverge if these constrains

are not satisfied. This is an issue that was previously ignored in ReLU RBM, as it was mainly used for pre-training rather than generative modeling. Our contribution in this paper is three-fold: I) We systematically identify and address model constraints in leaky RBM (Section 3); II) For the training of leaky RBM, we propose a meta algorithm for sampling, which anneals leakiness during the Gibbs sampling procedure (Section 3) and empirically show that it can boost contrastive divergence with faster mixing (Section 5); III) We demonstrate the power of the proposed sampling algorithm on estimating the partition function. In particular, comparison on several benchmark datasets shows that the proposed method outperforms the conventional AIS (Salakhutdinov & Murray, 2008) (Section 4). Moreover, we provide an incentive for using leaky RBM by showing that the leaky ReLU hidden units perform better than the Bernoulli units in terms of the model log-likelihood (Section 4).

# 2 RESTRICTED BOLTZMANN MACHINE AND RELU

The Boltzmann distribution is defined as

$$
p (x) = \frac {e ^ {- E (x)}}{Z},
$$

where  $Z = \sum_{x} e^{-E(x)}$  is the partition function. Restricted Boltzmann Machine (RBM) is a Boltzmann distribution with a bipartite structure. It is also the building block for many deep models (e.g., Hinton et al., 2006; Salakhutdinov & Hinton, 2009; Lee et al., 2009), which are widely used in numerous applications (Bengio, 2009). The conventional Bernoulli RBM, models the joint probability  $p(v,h)$  for the visible units  $v \in [0,1]^I$  and the hidden units  $h \in [0,1]^J$  as  $p(v,h) \propto \exp(-E(v,h))$ , where

$$
E (v, h) = a ^ {\top} v - v ^ {\top} W h + b ^ {\top} h.
$$

The parameters are  $a \in \mathbb{R}^I$ ,  $b \in \mathbb{R}^J$  and  $W \in \mathbb{R}^{I \times J}$ . We can derive the conditional probabilities as

$$
p \left(v _ {i} = 1 \mid h\right) = \sigma \left(\sum_ {j = 1} ^ {J} W _ {i j} h _ {j} + a _ {i}\right) \quad \text {a n d} \quad p \left(h _ {j} = 1 \mid v\right) = \sigma \left(\sum_ {i = 1} ^ {I} W _ {i j} v _ {i} + b _ {j}\right), \tag {1}
$$

where  $\sigma (x) = (1 + e^{-x})^{-1}$  is the sigmoid function.

One extension of Bernoulli RBM is replacing the binary visible units by linear units  $v \in \mathbb{R}^I$  with independent Gaussian noise. The energy function in this case is given by

$$
E (v, h) = \sum_ {i = 1} ^ {I} \frac {(v _ {i} - a _ {i}) ^ {2}}{2 \sigma_ {i} ^ {2}} - \sum_ {i = 1} ^ {I} \sum_ {j = 1} ^ {J} \frac {v _ {i}}{\sigma_ {i}} W _ {i j} h _ {j} + b ^ {\top} h.
$$

To simplify the notation, we eliminate  $a_i$  and  $\sigma_i$  in this paper, and then the energy function is simplified to be  $E(v,h) = \frac{\|v\|^2}{2} - v^\top Wh + b^\top h$ . Note that the elimination does not influence the discussion and one can easily extend all the results in this paper to the model that includes  $a_i$  and  $\sigma_i$ .

The conditional distributions are as follows:

$$
p \left(v _ {i} \mid h\right) = \mathcal {N} \left(\sum_ {j = 1} ^ {J} W _ {i j} h _ {j}, 1\right) \quad \text {a n d} \quad p \left(h _ {j} = 1 \mid v\right) = \sigma \left(\sum_ {i = 1} ^ {I} W _ {i j} v _ {i} + b _ {j}\right), \tag {2}
$$

where  $\mathcal{N}(\mu, V)$  is a Gaussian distribution with mean  $\mu$  and variance  $V$ .

# 2.1 RELU RBM WITH CONTINUOUS VISIBLE UNITS

From (1) and (2), we can see that the mean of the  $p(h_j|v)$  is actually the evaluation of a sigmoid function at the response  $\sum_{i=1}^{I} W_{ij} v_i + b_j$ , which is the non-linearity of the hidden units. From this perspective, we can extend the sigmoid function to other functions and thus allow RBM to have more expressive power (Ravanbakhsh et al., 2016). Nair & Hinton (2010) propose to use rectified linear unit (ReLU) to replace conventional sigmoid hidden units. The activation function is defined as a one-sided function  $\max(0, x)$ .

However, as it has been shown in Ravanbakhsh et al. (2016), only the strictly monotonic activation functions can derive feasible joint and conditional distributions<sup>1</sup>. Therefore, we consider the leaky ReLU (Maas et al., 2013) in this paper. The activation function of leaky ReLU is defined as  $\max(cx, x)$ , where  $c \in (0, 1)$  is the leakiness parameter.

To simplify the notation, we define  $\eta_{j} = \sum_{i=1}^{I} W_{ij} v_{i} + b_{j}$ . By Ravanbakhsh et al. (2016), the conditional probability of the activation  $f$  is defined as  $p(h_{j} | v) = \exp(-D_{f}(\eta_{j} \| h_{j}) + g(h_{j}))$ , where  $D_{f}(\eta_{j} \| h_{j})$  is a Bregman Divergence and  $g(h_{j})$  is the base measure. The Bergman divergence of  $f$  is given by  $D_{f}(\eta_{j} \| h_{j}) = -\eta_{j} h_{j} + F(\eta_{j}) + F^{*}(h_{j})$ , where  $F$  with  $\frac{d}{d\eta_j} F(\eta_j) = f(\eta_j)$  is the anti-derivative of  $f$  and  $F^{*}$  is the anti-derivative of  $f^{-1}$ . We then get the conditional distributions of leaky RBM as

$$
p \left(h _ {j} \mid v\right) = \left\{ \begin{array}{l l} \mathcal {N} \left(\eta_ {j}, 1\right), & \text {i f} \eta_ {j} > 0 \\ \mathcal {N} \left(c \eta_ {j}, c\right), & \text {i f} \eta_ {j} \leq 0. \end{array} \right. \tag {3}
$$

Note that the conditional distribution of the visible unit is

$$
p \left(v _ {i} \mid h\right) = \mathcal {N} \left(\sum_ {j = 1} ^ {J} W _ {i j} h _ {j}, 1\right), \tag {4}
$$

which can also be written as  $p(v_{i}|h) = \exp \left(-D_{\tilde{f}}(\nu_{i}\| v_{i}) + g(v_{i})\right)$ , where  $\nu_{i} = \sum_{j=1} W_{ij} h_{j}$  and  $\tilde{f}(x) = x$ . By having these two conditional distributions, we can train and do inference on a leaky RBM model by using contrastive divergence (Hinton, 2002) or other algorithms (Tieleman, 2008; Tieleman & Hinton, 2009).

# 3 TRAINING AND SAMPLING FROM LEAKY RBM

First, we explore the joint and marginal distribution of the leaky RBM. Given the conditional distributions  $p(v|h)$  and  $p(h|v)$ , the joint distribution  $p(v,h)$  from the general treatment for MRF model given by Yang et al. (2012) is

$$
p (v, h) \propto \exp \left(v ^ {\top} W h - \sum_ {i = 1} ^ {I} \left(\tilde {F} ^ {*} \left(v _ {i}\right) + g \left(v _ {i}\right)\right) - \sum_ {j = 1} ^ {J} \left(F ^ {*} \left(h _ {j}\right) + g \left(h _ {j}\right)\right)\right). \tag {5}
$$

By (5), we can derive the joint distribution of the leaky-ReLU RBM as

$$
p (v, h) \propto \exp \left(v ^ {\top} W h - \frac {\| v \| ^ {2}}{2} - \sum_ {\eta_ {j} > 0} \left(\frac {h _ {j} ^ {2}}{2} + \log \sqrt {2 \pi}\right) - \sum_ {\eta_ {j} \leq 0} \left(\frac {h _ {j} ^ {2}}{2 c} + \log \sqrt {2 c \pi}\right) + b ^ {\top} h\right),
$$

and the marginal distribution as

$$
\begin{array}{l} p (v) \propto \exp \left(- \frac {\| v \| ^ {2}}{2}\right) \prod_ {\eta_ {j} > 0} \exp \left(\frac {\eta_ {j} ^ {2}}{2}\right) \prod_ {\eta_ {j} \leq 0} \left(\frac {c \eta_ {j} ^ {2}}{2}\right) \\ \propto \exp \left(- \frac {1}{2} v ^ {\top} \left(I - \sum_ {\eta_ {j} > 0} W _ {j} W _ {j} ^ {\top} - c \sum_ {\eta_ {j} \leq 0} W _ {j} W _ {j} ^ {\top}\right) v + \sum_ {\eta_ {j} > 0} b _ {j} W _ {j} ^ {\top} v + c \sum_ {\eta_ {j} \leq 0} b _ {j} W _ {j} ^ {\top} v\right). \tag {6} \\ \end{array}
$$

where  $W_{j}$  is the  $j$ -th column of  $W$ .

# 3.1 LEAKY RBM AS UNION OF TRUNCATED GAUSSIAN DISTRIBUTIONS

From (6), the marginal probability is determined by the affine constraints  $\eta_j > 0$  or  $\eta_j \leq 0$  for all  $j$ . By combinatorics, these constraints divide  $\mathbb{R}^I$  into at most  $M = \sum_{i=1}^{I} \binom{J}{i}$  convex regions  $R_1, \dots, R_M$ . An example with  $I = 2$  and  $J = 3$  is shown in Figure 2. If  $I > J$ , then we have at most  $2^J$  regions.

![](images/1cb6140d8c6cd67a7bdd9c7334022ba166f99640e4c58fb76221512da488602a.jpg)  
Figure 1: A two dimensional example with 3 hidden units.

![](images/6055e5997217d8f71d5c7699064136da46201eccfbbd4d862a77212c995e9ab0.jpg)  
Figure 2: An one dimensional example of truncated Gaussian distributions with different variances.

![](images/3fecefbf2e20c3a7f32b4798031e81b4ca5db4b03727de2d32c6bf1e01cc8cdd.jpg)  
Figure 3: A three dimensional example with 3 hidden units, where  $W_{j}$  are orthogonal to each other.

We discuss the two types of these regions. For bounded regions, such as  $R_{1}$  in Figure 2, the integration of (6) is also bounded, which results in a valid distribution. Before we discuss the unbounded cases, we define  $\Omega = I - \sum_{j=1}^{J} \alpha_{j} W_{j} W_{j}^{\top}$ , where  $\alpha_{j} = \mathbb{1}_{\eta_{j} > 0} + c \mathbb{1}_{\eta_{j} \leq 0}$ . For the unbounded region, if  $\Omega \in \mathbb{R}^{I \times I}$  is a positive definite (PD) matrix, then the probability density is proportional to a multivariate Gaussian distribution with mean  $\mu = \Omega^{-1}\left(\sum_{j=1}^{J} \alpha_{j} b_{j} W_{j}\right)$  and precision matrix  $\Omega$  (covariance matrix  $\Omega^{-1}$ ) but over an affine-constrained region. Therefore, the distribution of each unbounded region can be treated as a truncated Gaussian distribution.

On the other hand, if  $\Omega$  is not PD, and the region  $R_{i}$  contains the eigenvectors with negative eigenvalues of  $\Omega$ , the integration of (6) over  $R_{i}$  is divergent (infinite), which can not result in a valid probability distribution. In practice, with this type of parameter, when we do Gibbs sampling on the conditional distributions, the sampling will diverge. However, it is unfeasible to check exponentially many regions for each gradient update.

Theorem 1. If  $I - WW^{\top}$  is positive definite, then  $I - \sum_{j} \alpha_{j} W_{j} W_{j}^{\top}$  is also positive definite, for all  $\alpha_{j} \in [0,1]$ .

The proof is shown in Appendix 1. From Theorem 1 we can see that if the constraint  $I - WW^{\top}$  is PD, then one can guarantee that the distribution of every region is a valid truncated Gaussian distribution. Therefore, we introduce the following projection step for each  $W$  after the gradient update.

$$
\underset {\tilde {W}} {\operatorname {a r g m i n}} \| W - \tilde {W} \| _ {F} ^ {2} \tag {7}
$$

$$
\begin{array}{c c} \text {s . t .} & I - \bar {W} \bar {W} ^ {\top} \succeq 0 \end{array}
$$

Theorem 2. The above projection step (7) can be done by shrinking the singular values to be less than 1.

The proof is shown in Appendix B. The training algorithm of the leaky RBM is shown in Algorithm 1. By using the projection step (7), we could treat the leaky RBM as the union of truncated Gaussian distributions, which uses weight vectors to divide the space of visible units into several regions and use a truncated Gaussian distribution to model each region. Note that the leaky RBM model is different from Su et al. (2016), which uses a truncated Gaussian distribution to model the conditional distribution  $p(h|v)$  instead of the marginal distribution. The empirical study about the divergent values and the necessity of the projection step is shown in Appendix C.

Algorithm 1 Training Leaky RBM  
for  $t = 1,\dots ,T$  do Estimate gradient  $g_{\theta}$  by CD or other algorithms with (3) and (4), where  $\theta = \{W,a,b\}$ $\theta^{(t)}\gets \theta^{(t - 1)} + \eta g_{\theta}.$  Project  $W^{(t)}$  by (7).   
end for

# 3.2 SAMPLING FROM LEAKY-RELURBM

Gibbs sampling is the core procedure for RBM, including training, inference, and estimating the partition function (Fischer & Igel, 2012; Tieleman, 2008; Salakhutdinov & Murray, 2008). For every task, we start from randomly initializing  $v$  by an arbitrary distribution  $q$ , and iteratively sample from the conditional distributions. Gibbs sampling guarantees the procedure result in the stationary distribution in the long run for any initialized distribution  $q$ . However, if  $q$  is close to the target distribution  $p$ , it can significantly shorten the number of iterations to achieve the stationary distribution.

If we set the leakiness  $c$  to be 1, then (6) becomes a simple multivariate Gaussian distribution  $\mathcal{N}\left((I - WW^{\top})^{-1}Wb, (I - WW^{\top})^{-1}\right)$ , which can be easily sampled without Gibbs sampling. Also, the projection step (7) guarantees it is a valid Gaussian distribution. Then we decrease the leakiness with a small  $\epsilon$ , and use samples from the multivariate Gaussian distribution when  $c = 1$  as the initialization to do Gibbs sampling. Note that the distribution of each region is a truncated Gaussian distribution. When we only decrease the leakiness with a small amount, the resulted distribution is a "similar" truncated Gaussian distribution with more concentrated density. From this observation, we could expect the original multivariate Gaussian distribution serves as a good initialization. The one-dimensional example is shown in Figure 2. We then repeat this procedure until we reach the target leakiness. The algorithm can be seen as annealing the leakiness during the Gibbs sampling procedure. The meta algorithm is shown in Algorithm 2. Next, we show the proposed sampling algorithm can help both the partition function estimation and the training of leaky RBM.

Algorithm 2 Meta Algorithm for Sampling from Leaky RBM  
Sample  $v$  from  $\mathcal{N}\left((I - WW^{\top})^{-1}Wb, (I - WW^{\top})^{-1}\right)$ $c' = 1$   
for  $t = 1, \dots, T$  do  
if  $c' > c$  then  
 $c' = c' - \epsilon$   
end if  
Do Gibbs sampling by using (3) and (4) with leakiness  $c'$  end for

# 4 PARTITION FUNCTION ESTIMATION

It is known that estimating the partition function of RBM is intractable (Salakhutdinov & Murray, 2008). Existing approaches, including Salakhutdinov & Murray (2008); Grosse et al. (2013); Liu et al. (2015); Carlson et al. (2016) focus on using sampling to approximate the partition function of the conventional Bernoulli RBM instead of the RBM with Gaussian visible units and non-Bernoulli hidden units. In this paper, we focus on extending the classic annealed importance sampling (AIS) algorithm (Salakhutdinov & Murray, 2008) to leaky RBM.

Assuming that we want to estimate the partition function  $Z$  of  $p(v)$  with  $p(v) = p^{*}(v) / Z$  and  $p^{*}(v) \propto \sum_{h} \exp(-E(v, h))$ , Salakhutdinov & Murray (2008) start from a initial distribution  $p_0(v) \propto \sum_{h} \exp(-E_0(v, h))$ , where computing the partition  $Z_0$  of  $p_0(v)$  is tractable and we can draw samples from  $p_0(v)$ . They then use the "geometric path" to anneal the intermediate distribution as  $p_k(v) \propto p_k^*(v) = \sum_{h} \exp(-\beta_k E_0(v, h) - (1 - \beta_k) E(v, h))$ , where they grid  $\beta_k$  from 1 to 0. If we let  $\beta_0 = 1$ , we can draw samples  $v_k$  from  $p_k(v)$  by using samples  $v_{k-1}$  from  $p_{k-1}(v)$  for  $k \geq 1$  via Gibbs sampling. The partition function is then estimated via  $Z = \frac{Z_0}{M} \sum_{i=1}^{M} \omega^{(i)}$ , where

$$
\omega^ {(i)} = \frac {p _ {1} ^ {*} (v _ {0} ^ {(i)})}{p _ {0} ^ {*} (v _ {0} ^ {(i)})} \frac {p _ {2} ^ {*} (v _ {1} ^ {(i)})}{p _ {1} ^ {*} (v _ {1} ^ {(i)})} \dots \frac {p _ {K - 1} ^ {*} (v _ {K - 2} ^ {(i)})}{p _ {K - 2} ^ {*} (v _ {K - 2} ^ {(i)})} \frac {p _ {K} ^ {*} (v _ {K - 1} ^ {(i)})}{p _ {K - 1} ^ {*} (v _ {K - 1} ^ {(i)})},
$$

and  $\beta_{K} = 0$

Salakhutdinov & Murray (2008) use the initial distribution with independent visible units and without hidden units. Therefore, we extend Salakhutdinov & Murray (2008) to the leaky-ReLU case with  $E_0(v,h) = \frac{\|v\|^2}{2}$ , which results in a multivariate Gaussian distribution  $p_0(v)$ . Compared with the meta algorithm shown in Algorithm 2 which anneals between leakiness, the extension of Salakhutdinov & Murray (2008) anneals between energy functions.

<table><tr><td></td><td>J=5</td><td>J=10</td><td>J=20</td><td>J=30</td></tr><tr><td>Log partition function</td><td>2825.48</td><td>2827.98</td><td>2832.98</td><td>2837.99</td></tr></table>

Table 1: The true partition function for Leaky-ReLU RBM with different number of hidden units.  

<table><tr><td></td><td>J=5</td><td>J=10</td><td>J=20</td><td>J=30</td></tr><tr><td>AIS-Energy</td><td>1.76 ± 0.011</td><td>3.56 ± 0.039</td><td>7.95 ± 0.363</td><td>9.60 ± 0.229</td></tr><tr><td>AIS-Leaky</td><td>0.02 ± 0.001</td><td>0.04 ± 0.002</td><td>0.08 ± 0.003</td><td>0.13 ± 0.004</td></tr></table>

Table 2: The difference between the true partition function and the estimations of two algorithms with standard deviation.

# 4.1 STUDY ON TOY EXAMPLES

As we discussed in Section 3.1, leaky RBM with  $J$  hidden units is a union of  $2^{J}$  truncated Gaussian distributions. Here we perform a study on the leaky RBM with a small number hidden units. Since in this example the number of hidden units is small, we can integrate out all possible configurations of  $h$ . However, integrating a truncated Gaussian distribution with general affine constraints does not have analytical solutions, and several approximations have been developed (e.g., Pakman & Paninski, 2014). To compare our results with the exact partition function, we consider a special case that has the following form:

$$
p (v) \propto \exp \left(- \frac {1}{2} v ^ {\top} \left(I - \sum_ {\eta_ {j} > 0} W _ {j} W _ {j} ^ {\top} - c \sum_ {\eta_ {j} \leq 0} W _ {j} W _ {j} ^ {\top}\right) v\right). \tag {8}
$$

Compared to (6), it is equivalent to the setting where  $b = 0$ . Geometrically, every  $W_{j}$  passes through the origin. We further put the additional constraint  $W_{i} \perp W_{j}, \forall i \neq j$ . Therefore, we divide the whole space into  $2^{J}$  equally-sized regions. A three-dimensional example is shown in Figure 3. Then the partition function of this special case has the analytical form

$$
Z = \frac {1}{2 ^ {J}} \sum_ {\alpha_ {j} \in \{1, c \}, \forall j} (2 \pi) ^ {- \frac {I}{2}} \left| \left(I - \sum_ {j = 1} ^ {J} \alpha_ {j} W _ {j} W _ {j} ^ {\top}\right) ^ {- \frac {1}{2}} \right|.
$$

We randomly initialize  $W$  and use SVD to make each column orthogonal to each other. Also, we scale  $\| W_j\|$  to satisfy  $I - WW^{\top} \succeq 0$ . The leukiness parameter is set to be 0.01. For Salakhutdinov & Murray (2008) (AIS-Energy), we use  $10^{5}$  particles with  $10^{5}$  intermediate distributions. For the proposed method (AIS-Leaky), we use only  $10^{4}$  particles with  $10^{3}$  intermediate distributions. In this small problem we study the cases when the model has 5, 10, 20 and 30 hidden units and 3072 visible units. The true log partition function  $\log Z$  is shown in Table 1 and the difference between  $\log Z$  and the estimates given by the two algorithms are shown in Table 2.

From Table 1, we observe that AIS-Leaky has significantly better and more stable estimations than AIS-Energy especially when  $J$  is large. For example, when we increase  $J$  from 5 to 30, the bias (difference) of AIS-Leaky only increases from 0.02 to 0.13; however, the bias of AIS-Energy increases from 1.76 to 9.6. Moreover, we note that AIS-Leaky uses less particles and less intermediate distributions, and therefore is more computationally efficient than AIS-Energy. We further study the implicit connection between the proposed AIS-Leaky and AIS-Energy in Appendix D, which shows AIS-Leaky is a special case of AIS-Energy under certain conditions.

# 4.2 COMPARISON BETWEEN LEAKY-RELURBM AND BERNOULLI-GAUSSIAN RBM

It is known that the reconstruction error is not a proper approximation of the likelihood (Hinton, 2012). By having an accurate estimation of the partition function, we can study the power of leaky RBM when our goal is to use under the likelihood function as our objective instead of the reconstruction error.

<table><tr><td></td><td>CIFAR-10</td><td>SVHN</td></tr><tr><td>Bernoulli-Gaussian RBM</td><td>-2548.3</td><td>-2284.2</td></tr><tr><td>Leaky-ReLU RBN</td><td>-1031.1</td><td>-182.4</td></tr></table>

Table 3: The log-likelihood performance of Bernoulli-Gaussian RBM and leaky RBM.

We compare the Bernoulli-Gaussian  $\mathrm{RBM}^2$ , which has Bernoulli hidden units and Gaussian visible units. We trained both models with  $\mathrm{CD - 20^{3}}$  and momentum. For both model, we all used 500 hidden units. We initialized  $W$  by sampling from  $\mathrm{Unif}(0,0.01)$ ,  $a = 0$ ,  $b = 0$  and  $\sigma = 1$ . The momentum parameter was 0.9 and the batch size was set to 100. We tuned the learning rate between  $10^{-1}$  and  $10^{-6}$ . We studied two benchmark data sets, including CIFAR10 and SVHN. The data was normalized to have zero mean and standard deviation of 1 for each pixel. The results of the log-likelihood values are reported in Table 3.

From Table 3, leaky RBM outperforms Bernoulli-Gaussian RBM significantly. The unsatisfactory performance of Bernoulli-Gaussian RBM may be in part due to the optimization procedure. If we tune the decay schedule of the learning-rate for each dataset in an ad-hoc way, we observe the performance of Bernoulli-Gaussian RBM can be improved by  $\sim 300$  nats for both datasets. Also, increasing CD-steps brings slight improvement. The other possibility is the bad mixing during the CD iterations. The advanced algorithms Tieleman (2008); Tieleman & Hinton (2009) may help. Although Nair & Hinton (2010) demonstrate the power of ReLU in terms of reconstruction error and classification accuracy, it does not imply its superior generative capability. Our study confirms leaky RBM could have a much better generative performance compared to Bernoulli-Gaussian RBM

# 5 BETTER MIXING BY ANNEALING LEAKINESS

In this section, we show the idea of annealing between leakiness benefit the mixing in Gibbs sampling in other settings. A common procedure for comparison of sampling methods for RBM is through visualization. Here, we are interested in more quantitative metrics and the practical benefits of improved sampling. For this, we consider optimization performance as the evaluation metric.

The gradient of the log-likelihood function  $\mathcal{L}(\theta | v_{data})$  of general RBM models is

$$
\frac {\partial \mathcal {L} (\theta | v _ {d a t a})}{\partial \theta} = \mathbb {E} _ {h | v _ {d a t a}} \left[ \frac {\partial E (v , h)}{\partial \theta} \right] - \mathbb {E} _ {v, h} \left[ \frac {\partial E (v , h)}{\partial \theta} \right]. \tag {9}
$$

Since the second expectation in (9) is usually intractable, people use different algorithms (Fischer & Igel, 2012) to approximate it.

In this section, we compare two gradient approximation procedures. The first one is the conventional contrastive divergence (CD) (Hinton, 2002). The second method is using Algorithm 2 (Leaky) with the same number of mixing steps as CD. The experiment setup is the same as that of Section 4. The results are shown in Figure 4. The proposed sampling procedure is slightly better than typical CD steps. The reason is we only anneals the leakiness for 20 steps. To get accurate estimation requires thousands of steps as shown in Section 4 when we estimate the partition function. Therefore, the estimated gradient is still inaccurate. However, it still outperforms the conventional CD algorithm, which can demonstrate the better mixing power of the proposed sampling algorithm as we expect.

The drawback of using Algorithm 2 is sampling  $v$  from  $\mathcal{N}\left((I - WW^{\top})^{-1}Wb, (I - WW^{\top})^{-1}\right)$  requires computing mean, covariance and the Cholesky decomposition of the covariance matrix in every iteration, which are computationally expensive. We study a mixture algorithm by combining CD and the idea of annealing leakiness. The mixture algorithm is replacing the sampling  $v$  from  $\mathcal{N}\left((I - WW^{\top})^{-1}Wb, (I - WW^{\top})^{-1}\right)$  with sampling from the empirical data distribution. The resulted mix algorithm is almost the same as CD algorithm while it anneals the leakiness over the iterations as Algorithm 2. The results of the mix algorithm is also shown in Figure 4.

![](images/bdca4e7e5ec6be7cf1cde7be1a97501c0b6ce87821dc6ec4af9a1c2e1f8bdb8a.jpg)  
(a) SVHN

![](images/e4e2385d5939361e024488c85459b16534e6938813b8d6fecda6af649cda4d75.jpg)  
(b) CIFAR10  
Figure 4: Training leaky RBM with different sampling algorithms.

The mix algorithm is slightly worse than the original leaky algorithm, but outperforms the conventional CD algorithm. Starting from the data distribution is biased to  $\mathcal{N}\left((I - WW^{\top})^{-1}Wb, (I - WW^{\top})^{-1}\right)$ , which cause the mix algorithm perform worse than Algorithm 2. However, by sampling from the data distribution, it is as efficient as the CD algorithm (without additional computation cost). Annealing the leakiness helps the mix algorithm explore different modes of the distribution, which benefits the training. The idea could also be combined with more advanced algorithms (Tieleman, 2008; Tieleman & Hinton, 2009)<sup>4</sup>.

# 6 CONCLUSION

In this paper, we study the properties of the distributions of leaky RBM. The study links the leaky RBM model and truncated Gaussian distributions. Also, our study shows and addresses an underlying positive definite constraint of training leaky RBM. Based on our study, we further propose a meta sampling algorithm, which anneals between leakiness during the Gibbs sampling procedure. We first demonstrate the proposed sampling algorithm is more effective and more efficient in estimating the partition function than the conventional AIS algorithm. Second, we show the proposed sampling algorithm has better mixing property under the evaluation via optimization.

A few direction worth further studying. For example, one is how to speed up the naive projection step. Some potential direction is using the barrier function as shown in Hsieh et al. (2011) to avoid the projection step.

# REFERENCES

Y. Bengio. Learning deep architectures for ai. Found. Trends Mach. Learn., 2009.  
Y. Burda, R. B. Grosse, and R. Salakhutdinov. Accurate and conservative estimates of mrf log-likelihood using reverse annealing. In AISTATS, 2015.  
D. E. Carlson, P. Stinson, A. Pakman, and L. Paninski. Partition functions from rao-blackwellized tempered sampling. In ICML, 2016.  
A. Fischer and C. Igel. An introduction to restricted boltzmann machines. In CIARP, 2012.  
Y. Freund and D. Haussler. Unsupervised learning of distributions on binary vectors using two layer networks. Technical report, 1994.  
I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In ICML. 2014.

R. B. Grosse, C. J. Maddison, and R. Salakhutdinov. Annealing between distributions by averaging moments. In NIPS, 2013.  
G. E. Hinton. Training products of experts by minimizing contrastive divergence. Neural Computation, 2002.  
G. E. Hinton. A practical guide to training restricted boltzmann machines. In Neural Networks: Tricks of the Trade (2nd ed.). 2012.  
G. E. Hinton, S. Osindero, and Y.-W. Teh. A fast learning algorithm for deep belief nets. Neural Computation, 2006.  
C.-J. Hsieh, M. A. Sustik, I. S. Dhillon, and P. Ravikumar. Sparse inverse covariance matrix estimation using quadratic approximation. In NIPS, 2011.  
D. P. Kingma and M. Welling. Auto-encoding variational bayes. CoRR, 2013.  
H. Lee, R. Grosse, R. Ranganath, and A. Y. Ng. Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. In ICML, 2009.  
Q. Liu, J. Peng, A. Ihler, and J. Fisher III. Estimating the partition function by discriminance sampling. In UAI, 2015.  
A. L. Maas, A. Y. Hannun, and A. Y. Ng. Rectifier nonlinearities improve neural network acoustic models. In ICML Workshop on Deep Learning for Audio, Speech, and Language Processing, 2013.  
V. Nair and G. E. Hinton. Rectified linear units improve restricted boltzmann machines. In ICML, 2010.  
A. Pakman and L. Paninski. Exact hamiltonian monte carlo for truncated multivariate gaussians. Journal of Computational and Graphical Statistics, 2014.  
N. Parikh and S. Boyd. Proximal algorithms. Found. Trends Optim., 2014.  
S. Ravanbakhsh, B. Poczos, J. G. Schneider, D. Schuurmans, and R. Greiner. Stochastic neural networks with monotonic activation functions. In AISTATS, 2016.  
R. Salakhutdinov and G. Hinton. Deep Boltzmann machines. In AISTATS, 2009.  
R. Salakhutdinov and I. Murray. On the quantitative analysis of Deep Belief Networks. In ICML, 2008.  
P. Smolensky. Parallel distributed processing: Explorations in the microstructure of cognition, vol. 1. 1986.  
Q. Su, X. Liao, C. Chen, and L. Carin. Nonlinear statistical learning with truncated gaussian graphical models. In ICML, 2016.  
L. Theis, A. van den Oord, and M. Bethge. A note on the evaluation of generative models. In ICLR, 2016.  
T. Tieleman. Training restricted boltzmann machines using approximations to the likelihood gradient. In ICML, 2008.  
T. Tieleman and G.E. Hinton. Using Fast Weights to Improve Persistent Contrastive Divergence. In ICML, 2009.  
M. Welling, M. Rosen-Zvi, and G. E. Hinton. Exponential family harmoniums with an application to information retrieval. In NIPS, 2004.  
E. Yang, P. Ravikumar, G. I. Allen, and Z. Liu. Graphical models via generalized linear models. In NIPS, 2012.
