# AN INFORMATION-THEORETIC FRAMEWORK FOR FAST AND ROBUST UNSUPERVISED LEARNING VIA NEURAL POPULATION INFOMAX

Wentao Huang & Kechen Zhang

Department of Biomedical Engineering

Johns Hopkins University School of Medicine

Baltimore, MD 21205, USA

{whuang21,kzhang4}@jhmi.edu

# ABSTRACT

A framework is presented for unsupervised learning of representations based on infomax principle for large-scale neural populations. We use an asymptotic approximation to the Shannon's mutual information for a large neural population to demonstrate that a good initial approximation to the global information-theoretic optimum can be obtained by a hierarchical infomax method. From the initial solution, an efficient algorithm based on gradient descent of the final objective function is proposed to learn representations from the input datasets, allowing complete, overcomplete, or undercomplete bases. As confirmed by numerical experiments, our method is robust and highly efficient for extracting salient features from image datasets. Compared with the main existing methods, our algorithm has a distinct advantage in both the training speed and the robustness of unsupervised representation learning.

# 1 INTRODUCTION

How to discover the unknown structures in data is a key task for machine learning. Learning good representations from observed data is important because a clearer description helps reveal the underlying structures. Representation learning has drawn considerable attention in recent years (Bengio et al., 2013). One category of algorithms for unsupervised learning of representations is based on probabilistic models (Lewicki & Sejnowski, 2000; Hinton, 2002; Hinton & Salakhutdinov, 2006; Lee et al., 2008), e.g. maximum likelihood (ML) estimation, maximum a posteriori (MAP) probability estimation, etc. Another category of algorithms is based on reconstruction error or generative criterion (Olshausen & Field, 1996; Aharon et al., 2006; Ranzato et al., 2007; Vincent et al., 2010; Mairal et al., 2009; 2010; Goodfellow et al., 2014), which usually uses the squared error and additional constraints for the objective functions. Sometimes the reconstruction error or generative criterion can be explained also by probabilistic models (Olshausen & Field, 1997; Vincent et al., 2010).

Shannon's information theory is a powerful tool for description of stochastic systems and could be utilized to provide a characterization for good representations (Vincent et al., 2010). However, computational difficulties associated with Shannon's mutual information (MI) (Shannon, 1948) have hindered its wider applications. The Monte Carlo (MC) sampling (MacKay, 2003; Yarrow et al., 2012) is a convergent method for estimating MI with arbitrary accuracy, but its computational inefficiency makes it unsuitable for difficult optimization problems especially in the cases of high-dimensional input stimuli and large population networks. Bell and Sejnowski (Bell & Sejnowski, 1995) have directly applied the infomax (Linsker, 1988) approach to independent component analysis (ICA) of data with independent non-Gaussian components, assuming additive noise, but their method requires that the number of outputs be equal to the number of inputs. The extensions of ICA that allows overcomplete bases incurs increased algorithm complexity and difficulty in learning of parameters (Lewicki & Sejnowski, 2000; Kreutz-Delgado et al., 2003; Hyvarinen, 2005; Le et al., 2011).

Since Shannon MI is closely related to ML and MAP (Clarke & Barron, 1990; Lee et al., 2000; Huang & Zhang, 2016), the algorithms of representation learning based on probabilistic models should be amenable to information-theoretic treatment. Representation learning based on reconstruction error could also be accommodated by information theory, because the inverse of Fisher information (FI) is the Cramér-Rao lower bound on the mean square decoding error obtained with any unbiased decoder (Rao, 1945). Hence minimizing the reconstruction error potentially maximizes a lower bound on the MI (Vincent et al., 2010).

Related problems also arise in neuroscience. It has long been suggested that the real nervous systems might approach an information-theoretic optimum for neural coding and computation (Barlow, 1961; Atick, 1992; Borst & Theunissen, 1999; Pouget et al., 2000). However, in the cerebral cortex, the number of neurons is huge, with about  $10^{5}$  neurons under a square millimeter of cortical surface (Rockel et al., 1980; Carlo & Stevens, 2013). It has often been computationally intractable to precisely characterize the information coding and processing for large-scale populations.

To address all these issues, we present a framework for unsupervised learning of representations in a large-scale nonlinear feedforward model based on infomax principle with realistic biological constraints such as neuron models with Poisson noise. First we adopt an objective function based on an asymptotic formula in the large population limit for the MI between the stimuli and the neural population responses (Huang & Zhang, 2016). Since the objective function is usually nonconvex, choosing a good initial value is very important for its optimization. Starting from the initial value, we use a hierarchical infomax approach to quickly find a tentative global optimal solution for each layer by analytic methods. Finally, a fast convergence learning rule is presented for optimizing the final objective functions based on the tentative optimal solution. Our algorithm is robust and can learn complete, overcomplete or undercomplete basis vectors quickly from different image datasets. Experimental results showed that the convergence rate of our method was significantly faster than other existing methods, often by an order of magnitude. More importantly, the number of output units processed by our method can be very large, much larger than the number of inputs. As far as we know, no existing model can easily deal with this situation.

# 2 METHODS

Suppose the input  $\mathbf{x}$  is a  $K$ -dimensional vector,  $\mathbf{x} = (x_{1},\dots ,x_{K})^{T}$ , the outputs of  $N$  neurons are denoted by a vector,  $\mathbf{r} = (r_1,\dots ,r_N)^T$ . In this paper we denote random variables by upper case letters, e.g., random variables  $X$  and  $R$ , in contrast to their vector values  $\mathbf{x}$  and  $\mathbf{r}$ . Then the MI between  $X$  and  $R$  is defined by (Cover & Thomas, 2006)

$$
I (X; R) = \left\langle \ln \frac {p (\mathbf {x} | \mathbf {r})}{p (\mathbf {x})} \right\rangle_ {\mathbf {r}, \mathbf {x}}, \tag {2.1}
$$

where  $\langle \cdot \rangle_{\mathbf{r},\mathbf{x}}$  denotes the expectation with respect to the probability density function (PDF)  $p(\mathbf{r},\mathbf{x})$ . A link between MI and FI was derived by some researchers (Brunel & Nadal, 1998). If the PDF  $p(\mathbf{x})$  and  $p(\mathbf{r}|\mathbf{x})$  are twice continuously differentiable for almost every  $\mathbf{x}\in \mathbb{R}^K$ , we can prove the following more general asymptotic approximation for large  $N$  (e.g.  $N\gg K$ , Huang & Zhang, 2016),

$$
I (X; R) \simeq I _ {G} = \frac {1}{2} \left\langle \ln \left(\det  \left(\frac {\mathbf {G} (\mathbf {x})}{2 \pi e}\right)\right) \right\rangle_ {\mathbf {x}} + H (X), \tag {2.2}
$$

where  $\operatorname{det}(\cdot)$  denotes the matrix determinant and  $H(X) = -\langle \ln p(\mathbf{x}) \rangle_{\mathbf{x}}$  is the entropy,

$$
\mathbf {G} (\mathbf {x}) = \mathbf {J} (\mathbf {x}) + \mathbf {P} (\mathbf {x}), \tag {2.3}
$$

$$
\mathbf {J} (\mathbf {x}) = - \left\langle \frac {\partial^ {2} \ln p (\mathbf {r} | \mathbf {x})}{\partial \mathbf {x} \partial \mathbf {x} ^ {T}} \right\rangle_ {\mathbf {r} | \mathbf {x}}, \tag {2.4}
$$

$$
\mathbf {P} (\mathbf {x}) = - \frac {\partial^ {2} \ln p (\mathbf {x})}{\partial \mathbf {x} \partial \mathbf {x} ^ {T}}. \tag {2.5}
$$

From (2.1) and (2.2) we get the conditional entropy

$$
H (X | R) = - \left\langle \ln p (\mathbf {x} | \mathbf {r}) \right\rangle_ {\mathbf {r}, \mathbf {x}} \simeq - \frac {1}{2} \left\langle \ln \left(\det  \left(\frac {\mathbf {G} (\mathbf {x})}{2 \pi e}\right)\right) \right\rangle_ {\mathbf {x}}. \tag {2.6}
$$

Here the Fisher information matrix  $\mathbf{J}(\mathbf{x})$  is symmetric positive semidefinite and can be written also

$$
\mathbf {J} (\mathbf {x}) = \left\langle \frac {\partial \ln p (\mathbf {r} | \mathbf {x})}{\partial \mathbf {x}} \frac {\partial \ln p (\mathbf {r} | \mathbf {x})}{\partial \mathbf {x} ^ {T}} \right\rangle_ {\mathbf {r} | \mathbf {x}}. \tag {2.7}
$$

Suppose  $p(\mathbf{r}|\mathbf{x})$  is conditional independent,  $p(\mathbf{r}|\mathbf{x}) = \prod_{n=1}^{N} p(r_n|\mathbf{x};\pmb{\theta}_n)$ , where  $\pmb{\theta}_n$  denotes a vector of parameters for the  $n$ -th neuron. Then we get (see Huang & Zhang, 2016)

$$
\mathbf {J} (\mathbf {x}) = N \int_ {\Theta} p (\boldsymbol {\theta}) \mathbf {S} (\mathbf {x}; \boldsymbol {\theta}) d \boldsymbol {\theta}, \tag {2.8}
$$

$$
\mathbf {S} (\mathbf {x}; \boldsymbol {\theta}) = \left\langle \frac {\partial \ln p (r | \mathbf {x} ; \boldsymbol {\theta})}{\partial \mathbf {x}} \frac {\partial \ln p (r | \mathbf {x} ; \boldsymbol {\theta})}{\partial \mathbf {x} ^ {T}} \right\rangle_ {r | \mathbf {x}}, \tag {2.9}
$$

where  $p(\theta)$  is the population density function of parameter  $\theta$

$$
p (\boldsymbol {\theta}) = \frac {1}{N} \sum_ {n = 1} ^ {N} \delta (\boldsymbol {\theta} - \boldsymbol {\theta} _ {n}), \tag {2.10}
$$

and  $\delta (\cdot)$  denotes the Dirac delta function. It can be proved that the approximation function of MI  $I_G[p(\pmb {\theta})]$  (Eq. 2.2) is concave about  $p(\pmb {\theta})$  (Huang & Zhang, 2016). In Eq. (2.8), we can approximate the continuous integral by a discrete summation for numerical computation,

$$
\mathbf {J} (\mathbf {x}) = N \sum_ {k = 1} ^ {K _ {1}} \alpha_ {k} \mathbf {S} (\mathbf {x}; \boldsymbol {\theta} _ {k}), \tag {2.11}
$$

where the integer  $K_{1} \in \{1, \dots, N\}$ ,  $\sum_{k=1}^{K_{1}} \alpha_{k} = 1$ ,  $\alpha_{k} > 0$ ,  $k = 1, \dots, K_{1}$ . We know that the cerebral cortex usually forms functional column structures and each column is composed of neurons with the same properties (Hubel & Wiesel, 1962). Hence the positive integer  $K_{1}$  can be regarded as the number of classes in the neural population. Poisson noise model and Gaussian noise model are detailed in Appendix.

# 2.1 The Objective Function

Given the tuning curve  $f(\mathbf{x};\pmb{\theta}_n)$ , our goal is to find the optimal population distribution density of parameter vector  $\pmb{\theta}_n$  from which we can maximize the MI between the stimulus  $\mathbf{x}$  and response  $\mathbf{r}$ . By Eq. (2.2), our optimization problem becomes:

$$
\operatorname {m a x i m i z e} I _ {G} \left[ \left\{\alpha_ {k} \right\} \right] = \frac {1}{2} \left\langle \ln \left(\det \left(\frac {\mathbf {G} (\mathbf {x})}{2 \pi e}\right)\right) \right\rangle_ {\mathbf {x}} + H (X) \tag {2.12}
$$

or equivalently

$$
\operatorname {m i n i m i z e} Q _ {G} \left[ \left\{\alpha_ {k} \right\} \right] = - \frac {1}{2} \left\langle \ln \left(\det \left(\mathbf {G} (\mathbf {x})\right)\right) \right\rangle_ {\mathbf {x}}, \tag {2.13}
$$

and

$$
\text {s u b j e c t} \sum_ {k = 1} ^ {K _ {1}} \alpha_ {k} = 1, \alpha_ {k} > 0, \forall k = 1, \dots , K _ {1}, \tag {2.14}
$$

where  $\mathbf{G}(\mathbf{x})$  is given by Eq. (2.3)-(2.5) and (2.11).

Sometimes we do not know the specific form of  $p(\mathbf{x})$  and only know  $M$  samples,  $\mathbf{x}_1, \dots, \mathbf{x}_M$ , which are independent and identically distributed (i.i.d.) samples drawn from the distribution  $p(\mathbf{x})$ , then we can use the empirical average to approximate the integral in Eq. (2.13),

$$
Q _ {G} \left[ \left\{\alpha_ {k} \right\} \right] \approx - \frac {1}{2} \sum_ {m = 1} ^ {M} \ln \left(\det \left(\mathbf {G} \left(\mathbf {x} _ {m}\right)\right)\right). \tag {2.15}
$$

Since  $Q_{G}[\{\alpha_{k}\}]$  is a convex function of  $\{\alpha_{k}\}$  (Huang & Zhang, 2016), we can directly use efficient numerical methods to solve the optimal solution for small  $K$ . However, for large  $K$ , finding optimal solution by numerical methods becomes intractable. In the following we will propose another way to this problem. Instead of directly solving its density distribution  $\{\alpha_{k}\}$ , we optimize the parameters  $\{\alpha_{k}\}$  and  $\{\theta_{k}\}$  simultaneously under a framework of hierarchical infomax.

# 2.2 Hierarchical Infomax

For clarity, here we consider the Poisson neuron model and suppose the tuning curve  $f(\mathbf{x};\pmb{\theta}_n)$  is a nonlinear function, although our method is easily applicable to other neuron models and different tuning functions, e.g. sigmoid and rectified linear unit (ReLU) (Nair & Hinton, 2010; Glorot et al., 2011).

Let us consider the nonlinear function for the  $n$ -th neuron has the form,  $f(\mathbf{x}; \boldsymbol{\theta}_n) = \tilde{f}(y_n; \tilde{\boldsymbol{\theta}}_n)$ ,  $n = 1, \dots, N$ , where  $\tilde{f}(y_n; \tilde{\boldsymbol{\theta}}_n)$  is a nonlinear function,  $\boldsymbol{\theta}_n = (\mathbf{w}_n^T, \tilde{\boldsymbol{\theta}}_n^T)^T$  and  $\tilde{\boldsymbol{\theta}}_n$  are the parameter vectors, and  $\mathbf{w}_n$  is a  $K$ -dimensional weights vector,

$$
y _ {n} = \mathbf {w} _ {n} ^ {T} \mathbf {x}. \tag {2.16}
$$

In general, it is very difficult to find the optimal parameters,  $\theta_{n}$ ,  $n = 1,\dots ,N$  for the following reasons. First, the number of output neurons  $N$  is very large, usually  $N\gg K$ . Second, the tuning curve  $f(\mathbf{x};\pmb{\theta}_n)$  is a nonlinear function, which usually leads to a nonconvex optimization problem. For nonconvex optimization problems, the selection of initial values often has a great influence on the final optimization results. Our approach meets these challenges by making better use of the large number of neurons and by finding good initial values by hierarchical infomax.

We divide the nonlinear transformation into two stages, mapping first from  $\mathbf{x}$  to  $y_{n}$ , and then from  $y_{n}$  to  $\tilde{f}(y_{n};\tilde{\pmb{\theta}}_{n})$ , where  $y_{n}$  can be regarded as the membrane potential of the  $n$ -th neuron, and  $\tilde{f}(y_{n};\tilde{\pmb{\theta}}_{n})$  its firing rate. As with the real neurons, we assume the membrane potential is corrupted by noise:

$$
\check {Y} _ {n} = Y _ {n} + Z _ {n}, \tag {2.17}
$$

where  $Z_{n} \sim \mathcal{N}(0, \sigma^{2})$  is a normal distribution with mean 0 and variance  $\sigma^2$ . Then the mean membrane potential of the  $k$ -th class subpopulation with  $N_{k} = N\alpha_{k}$  neurons is given by

$$
\bar {Y} _ {k} = \frac {1}{N _ {k}} \sum_ {n = 1} ^ {N _ {k}} \check {Y} _ {k _ {n}} = Y _ {k} + \bar {Z} _ {k}, k = 1, \dots , K _ {1}, \tag {2.18}
$$

$$
\bar {Z} _ {k} \sim \mathcal {N} \left(0, N _ {k} ^ {- 1} \sigma^ {2}\right). \tag {2.19}
$$

Define vectors  $\breve{\mathbf{y}} = (\breve{y}_1,\dots ,\breve{y}_N)^T$ $\bar{\mathbf{y}} = (\bar{y}_1,\dots ,\bar{y}_{K_1})^T$  and  $\mathbf{y} = (y_{1},\dots ,y_{K_{1}})^{T}$ , where  $y_{k} = \mathbf{w}_{k}^{T}\mathbf{x}$ $(k = 1,\dots ,K_{1})$ . Notice that  $\breve{y}_n$ $(n = 1,\dots ,N)$  is also divided into  $K_{1}$  classes, the same as for  $r_n$ . If we assume  $f(\mathbf{x};\pmb {\theta}_k) = \tilde{f} (\bar{y}_k;\tilde{\pmb{\theta}}_k)$ , i.e. assuming an additive Gaussian noise for  $y_{n}$  (see Eq. 2.18), then the random variables  $X,Y,\breve{Y},\breve{Y}$  and  $R$  form a Markov chain, denoted by  $X\to Y\to \breve{Y}\to \bar{Y}\to R$ , which is demonstrated in Figure 1, and we have the following proposition (the proof is given in Appendix).

Proposition 2.1. With the random variables  $X, Y, \check{Y}, \bar{Y}$ ,  $R$  and Markov chain  $X \to Y \to \check{Y} \to \bar{Y} \to R$ , the following equations hold,

$$
I (X; R) = I (Y; R) \leq I (\bar {Y}; R), \tag {2.20}
$$

$$
I (X; R) \leq I (X; \bar {Y}) = I (X; \check {Y}), \tag {2.21}
$$

and for large  $N_{k}$ $(k = 1,\dots ,K_{1})$

$$
I (Y; R) \simeq I (\bar {Y}; R). \tag {2.22}
$$

A major advantage of incorporating membrane noise is that it facilitates finding the optimal solution by using the infomax principle. When the noise variance  $\sigma^2\rightarrow 0$ , we have  $\bar{Y}_k\to Y_k$ ,  $\tilde{f} (\bar{y}_k;\tilde{\pmb{\theta}}_k)\simeq \tilde{f} (y_k;\tilde{\pmb{\theta}}_k) = f(\mathbf{x};\pmb {\theta}_k)$  and  $I(\bar{Y};R)\simeq I(Y;R) = I(X;R)$ .

Hence, from the above we know that maximizing  $I(X;R)$  will result in maximizing  $I(Y;R)$  and maximizing  $I\left(X;\breve{Y}\right)$ . Therefore the optimal solution of  $\max I\left(X;\breve{Y}\right)$  and  $\max I(Y;R)$  will give a good initial approximation that will be very close to the optimal solution of  $\max I(X;R)$ .

Similarly, we can extend this method to multilayer neural population networks. For example, the two layers neural populations networks with outputs  $R^{(1)}$  and  $R^{(2)}$  form a Markov chain,  $X \rightarrow$

![](images/1e17e885e59421f47540b18c6ab6ca7241cbef798a1251d001382e7ad02f1b39.jpg)  
Figure 1: A neural network structure diagram for random variables  $X, Y, \breve{Y}, \breve{Y}, R$ .

$\tilde{R}^{(1)} \to R^{(1)} \to \bar{R}^{(1)} \to R^{(2)}$ , where the random variable  $\tilde{R}^{(1)}$  is similar to  $Y$ , the random variable  $R^{(1)}$  is similar to  $\breve{Y}$  and  $\bar{R}^{(1)}$  is similar to  $\bar{Y}$  in the above. Then we can prove that the optimal solution of  $\max I(X; R^{(2)})$  approximates the solutions of  $\max I(X; R^{(1)})$  and  $\max I(\tilde{R}^{(1)}; R^{(2)})$ , where  $I(\tilde{R}^{(1)}; R^{(2)}) \simeq I(\bar{R}^{(1)}; R^{(2)})$ .

More generally, consider a highly nonlinear feedforward neural network that maps the input  $\mathbf{x}$  to output  $\mathbf{z}$ , with  $\mathbf{z} = F(\mathbf{x};\pmb{\theta}) = h_{L} \circ \dots \circ h_{1}(\mathbf{x})$ , where  $h_{l}$  ( $l = 1,\dots ,L$ ) is a linear or nonlinear function (Montufar et al., 2014). We aim to find the optimal parameter  $\pmb{\theta}$  by maximizing  $I(X;Z)$ . It is usually difficult to solve the optimization problem when there are many local extremums for  $F(\mathbf{x};\pmb{\theta})$ . However, if each function  $h_l$  is easy to optimize, then we can use the above method (i.e. hierarchical infomax for neural population) to get a good initial approximation to its global optimization solution, and go from there to find the final optimal solution. This information-theoretic consideration from the neural population coding point of view may help explain why deep structure networks with unsupervised pre-training have a powerful ability for learning representations.

In the following, we will discuss the optimization procedure in two stages, namely maximizing  $I\left(X;\breve{Y}\right)$  and maximizing  $I(Y;R)$ .

# 2.2.1 The 1st Stage

In the first stage, our goal is to maximize the MI  $I\left(X; \check{Y}\right)$  and get the optimal parameters  $\mathbf{w}_n$  ( $n = 1, \dots, N$ ). Let us assume the stimulus  $\mathbf{x}$  has zero mean (if not, then let  $\mathbf{x} \gets \mathbf{x} - \langle \mathbf{x} \rangle_{\mathbf{x}}$ ) and covariance matrix  $\boldsymbol{\Sigma}_{\mathbf{x}}$ . It follows from eigendecomposition that

$$
\boldsymbol {\Sigma} _ {\mathbf {x}} = \left\langle \mathbf {x x} ^ {T} \right\rangle_ {\mathbf {x}} \approx \frac {1}{M - 1} \mathbf {X} \mathbf {X} ^ {T} = \mathbf {U} \boldsymbol {\Sigma} \mathbf {U} ^ {T}, \tag {2.23}
$$

where  $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_M]$  (see 2.15),  $\mathbf{U} = [\mathbf{u}_1, \dots, \mathbf{u}_K] \in \mathbb{R}^{K \times K}$  is an unitary orthogonal matrix and  $\boldsymbol{\Sigma}$  is a diagonal matrix with positive real numbers on the diagonal,  $\boldsymbol{\Sigma} = \mathrm{diag}\left(\sigma_1^2, \dots, \sigma_K^2\right)$ ,  $\sigma_1 \geq \dots \geq \sigma_K > 0$ .

Define

$$
\tilde {\mathbf {x}} = \boldsymbol {\Sigma} ^ {- 1 / 2} \mathbf {U} ^ {T} \mathbf {x}, \tag {2.24}
$$

$$
\tilde {\mathbf {w}} _ {k} = \boldsymbol {\Sigma} ^ {1 / 2} \mathbf {U} ^ {T} \mathbf {w} _ {k}, \tag {2.25}
$$

$$
y _ {k} = \tilde {\mathbf {w}} _ {k} ^ {T} \tilde {\mathbf {x}}, \tag {2.26}
$$

where  $k = 1,\dots ,K_{1}$  . The covariance matrix of  $\tilde{\mathbf{x}}$  is given by

$$
\boldsymbol {\Sigma} _ {\tilde {\mathbf {x}}} = \left\langle \tilde {\mathbf {x}} \tilde {\mathbf {x}} ^ {T} \right\rangle_ {\tilde {\mathbf {x}}} \approx \mathbf {I} _ {K}, \tag {2.27}
$$

and  $\mathbf{I}_K$  is a  $K\times K$  identity matrix, then from (2.2) and (A.5) we have that  $I\left(X;\breve{Y}\right) = I(\tilde{X};\breve{Y})$  and

$$
I (\tilde {X}; \check {Y}) \simeq I _ {G} ^ {\prime} = \frac {1}{2} \ln \left(\det  \left(\frac {\tilde {\mathbf {G}}}{2 \pi e}\right)\right) + H (\tilde {X}), \tag {2.28}
$$

$$
\tilde {\mathbf {G}} = N \sigma^ {- 2} \sum_ {k = 1} ^ {K _ {1}} \alpha_ {k} \tilde {\mathbf {w}} _ {k} \tilde {\mathbf {w}} _ {k} ^ {T} + \mathbf {I} _ {K}. \tag {2.29}
$$

Generally we can use the following approximation (see Huang & Zhang, 2016),

$$
p (\tilde {\mathbf {x}}) \simeq \mathcal {N} \left(0, \mathbf {I} _ {K}\right), \tag {2.30}
$$

$$
\mathbf {P} (\tilde {\mathbf {x}}) = - \frac {\partial^ {2} \ln p (\tilde {\mathbf {x}})}{\partial \tilde {\mathbf {x}} \partial \tilde {\mathbf {x}} ^ {T}} \simeq \mathbf {I} _ {K}. \tag {2.31}
$$

By the central limit theorem, the distribution of random variable  $\tilde{X}$  is closer to a normal distribution than the distribution of the original random variable  $X$ . On the other hand, the PCA models assume multivariate gaussian data whereas the ICA models assume multivariate non-gaussian data. Hence by a PCA-like whitening transformation (2.24) we can use the approximation (2.31) with the Laplace's method asymptotic expansion, which only requires that the peak be close to its mean and not that the random variable  $\tilde{X}$  be exactly Gaussian.

If there is no other constraints on the Gaussian channels of neural populations, especially the peak firing rate, then the capacity of this channel may grow indefinitely, i.e.,  $I(\tilde{X};\breve{Y}) \to \infty$ . The most common limitation on the neural populations is an energy or power constraint which can also be regarded as a limitation on the signal-to-noise ratio (SNR). The SNR for the output  $\breve{y}_n$  of the  $n$ -th neuron is given by

$$
\mathrm {S N R} _ {n} = \frac {1}{\sigma^ {2}} \left\langle \left(\mathbf {w} _ {n} ^ {T} \mathbf {x}\right) ^ {2} \right\rangle_ {\mathbf {x}} \approx \frac {1}{\sigma^ {2}} \tilde {\mathbf {w}} _ {n} ^ {T} \tilde {\mathbf {w}} _ {n}, n = 1, \dots , N. \tag {2.32}
$$

We require that

$$
\frac {1}{N} \sum_ {n = 1} ^ {N} \mathrm {S N R} _ {n} \approx \frac {1}{\sigma^ {2}} \sum_ {k = 1} ^ {K _ {1}} \alpha_ {k} \tilde {\mathbf {w}} _ {k} ^ {T} \tilde {\mathbf {w}} _ {k} \leq \rho , \tag {2.33}
$$

where  $\rho$  is a positive constant. Then by Eq. (2.28), (2.29) and (2.33), we have the following optimization problem:

$$
\operatorname {m i n i m i z e} Q _ {G} ^ {\prime} [ \hat {\mathbf {W}} ] = - \frac {1}{2} \ln \left(\det  \left(N \sigma^ {- 2} \hat {\mathbf {W}} \hat {\mathbf {W}} ^ {T} + \mathbf {I} _ {K}\right)\right), \tag {2.34}
$$

$$
\text {s u b j e c t} h = \operatorname {T r} \left(\hat {\mathbf {W}} \hat {\mathbf {W}} ^ {T}\right) - E \leq 0, \tag {2.35}
$$

where  $\operatorname{Tr}(\cdot)$  denotes matrix trace and

$$
\hat {\mathbf {W}} = \tilde {\mathbf {W}} \mathbf {A} ^ {1 / 2} = \boldsymbol {\Sigma} ^ {1 / 2} \mathbf {U} ^ {T} \mathbf {W} \mathbf {A} ^ {1 / 2} = \left[ \hat {\mathbf {w}} _ {1}, \dots , \hat {\mathbf {w}} _ {K _ {1}} \right], \tag {2.36}
$$

$$
\mathbf {A} = \operatorname {d i a g} \left(\alpha_ {1}, \dots , \alpha_ {K _ {1}}\right), \tag {2.37}
$$

$$
\mathbf {W} = \left[ \mathbf {w} _ {1}, \dots , \mathbf {w} _ {K _ {1}} \right], \tag {2.38}
$$

$$
\tilde {\mathbf {W}} = \left[ \tilde {\mathbf {w}} _ {1}, \dots , \tilde {\mathbf {w}} _ {K _ {1}} \right], \tag {2.39}
$$

$$
E = \rho \sigma^ {2}. \tag {2.40}
$$

Then we can get an optimal solution as following:

$$
\mathbf {W} = a \mathbf {U} _ {0} \boldsymbol {\Sigma} _ {0} ^ {- 1 / 2} \mathbf {V} _ {0} ^ {T}, \tag {2.41}
$$

$$
\mathbf {A} = K _ {1} ^ {- 1} \mathbf {I} _ {K _ {1}}, \tag {2.42}
$$

$$
a = \sqrt {E K _ {1} K _ {0} ^ {- 1}}, \tag {2.43}
$$

where  $K_{0}$  is a positive number,  $0 < K_{0} \leq K$ , and represents the size of the reduced dimension,

$$
\boldsymbol {\Sigma} _ {0} = \operatorname {d i a g} \left(\sigma_ {1} ^ {2}, \dots , \sigma_ {K _ {0}} ^ {2}\right), \tag {2.44}
$$

$$
\mathbf {U} _ {0} = \mathbf {U} (:, 1: K _ {0}) \in \mathbb {R} ^ {K \times K _ {0}}, \tag {2.45}
$$

$$
\mathbf {V} _ {0} = \mathbf {V} (:, 1: K _ {0}) \in \mathbb {R} ^ {K _ {1} \times K _ {0}}, \tag {2.46}
$$

and  $\mathbf{V} = [\mathbf{v}_1,\dots ,\mathbf{v}_{K_1}]$  is an  $K_{1}\times K_{1}$  unitary orthogonal matrix.

In this case, the optimal parameters  $\mathbf{w}_n$  ( $n = 1, \dots, N$ ) cluster into  $K_{1}$  classes (see Eq. 2.11) and obey an uniform discrete distribution (see also Eq. 2.57 in Section 2.2.2). Here notice that  $K_{0}$  is an unknown parameter which will be determined below.

In the case of  $K = K_{0} = K_{1}$ , the solution of  $\mathbf{W}$ , i.e. Eq. (2.41), is whitening-like filters. When  $\mathbf{V} = \mathbf{I}_K$ , the optimal matrix  $\mathbf{W}$  is the principal component analysis (PCA) whitening filters. However, when  $\mathbf{V} = \mathbf{U}$ , the optimal matrix  $\mathbf{W}$  is symmetrical and is a zero component analysis (ZCA) whitening filter. If  $K < K_{1}$ , this case leads to an overcomplete solution, whereas when  $K_{0} \leq K_{1} < K$ , the undercomplete solution arises. Since  $K_{0} \leq K_{1}$  and  $K_{0} \leq K$ ,  $Q_G'$  achieves its minimum when  $K_{0} = K$ . However, in practice other factors may limit its reaching this maximum. For example, consider the average of squared weights,

$$
\varsigma = \sum_ {k = 1} ^ {K _ {1}} \alpha_ {k} \| \mathbf {w} _ {k} \| ^ {2} = \operatorname {T r} \left(\mathbf {W A W} ^ {T}\right) = \frac {E}{K _ {0}} \sum_ {k = 1} ^ {K _ {0}} \sigma_ {k} ^ {- 2}, \tag {2.47}
$$

where  $\|\cdot\|$  denotes the Frobenius norm. The value of  $\varsigma$  is extremely large when any  $\sigma_k$  becomes vanishingly small. For real neurons these weights of connection are not allowed to be too large. Hence we impose a limitation on the weights, e.g.  $\varsigma \leq E_1$ , where  $E_1$  is a positive constant. This yields another constraint on the objective function,

$$
\tilde {h} = \frac {E}{K _ {0}} \sum_ {k = 1} ^ {K _ {0}} \sigma_ {k} ^ {- 2} - E _ {1} \leq 0. \tag {2.48}
$$

In this case, we can choose  $K_0$  by the following (see Appendix for details):

$$
K _ {0} = \arg \min  _ {K _ {0} ^ {\prime}} \left(\sqrt {\frac {\sum_ {k = 1} ^ {K _ {0} ^ {\prime}} \sigma_ {k} ^ {2}}{\sum_ {k = 1} ^ {K} \sigma_ {k} ^ {2}}} \geq \epsilon\right), \tag {2.49}
$$

where  $\epsilon$  is a positive constant,  $0 < \epsilon \leq 1$ .

Moreover, another advantage of a low-rank matrix  $\mathbf{W}$  is that it can significantly reduce overfitting for learning neural population parameters. In practice, the constraint (consw) is equivalent to the regularization term called a weight-decay term in many other optimization problems (Hoerl & Kennard, 1970; Cortes & Vapnik, 1995; Hinton, 2010), which can reduce overfitting to the training data. To prevent neural networks from over fitting, Srivastava et al. (2014) presented a technique to randomly drop units from the neural network during training, which may in fact be regarded as an attempt to reduce the rank of the weight matrix.

In this stage, we get the optimal parameter  $\mathbf{W}$  (see 2.41), but the matrix  $\mathbf{V}_0$  is uncertain and we need to go further to determine its optimal value (see section 2.3).

# 2.2.2 The 2nd Stage

For this stage, our goal is to maximize the MI  $I(Y;R)$  and get the optimal parameters  $\tilde{\theta}_n$  ( $n = 1,\dots ,N$ ). Here the input is  $\mathbf{y} = (y_{1},\dots ,y_{K_{1}})^{T}$  and the output  $\mathbf{r} = (r_1,\dots ,r_N)^T$  is also clustered into  $K_{1}$  classes. The responses of  $N_{k}$  neurons in the  $k$ -th subpopulation obey a Poisson distribution with mean  $\tilde{f} (\mathbf{e}_k^T\mathbf{y};\tilde{\pmb{\theta}}_k)$ , where  $\mathbf{e}_k$  is a unit vector with 1 in the  $k$ -th element and  $y_{k} = \mathbf{e}_{k}^{T}\mathbf{y}$ . By (2.24) and (2.26), we have that

$$
\left\langle y _ {k} \right\rangle_ {y _ {k}} = 0, \tag {2.50}
$$

$$
\sigma_ {y _ {k}} ^ {2} = \left\langle y _ {k} ^ {2} \right\rangle_ {y _ {k}} = \left\| \tilde {\mathbf {w}} _ {k} \right\| ^ {2}. \tag {2.51}
$$

Then for large  $N$  and by (2.2)-(2.5) and (2.30) we can use the following approximation,

$$
I (Y; R) \simeq \check {I} _ {F} = \frac {1}{2} \left\langle \ln \left(\det  \left(\frac {\check {\mathbf {J}} (\mathbf {y})}{2 \pi e}\right)\right) \right\rangle_ {\mathbf {y}} + H (Y), \tag {2.52}
$$

where

$$
\mathbf {J} (\mathbf {y}) = \operatorname {d i a g} \left(N \alpha_ {1} \left| g _ {1} ^ {\prime} \left(y _ {1}\right) \right| ^ {2}, \dots , N \alpha_ {K _ {1}} \left| g _ {K _ {1}} ^ {\prime} \left(y _ {K _ {1}}\right) \right| ^ {2}\right), \tag {2.53}
$$

$$
g _ {k} ^ {\prime} \left(y _ {k}\right) = \frac {\partial g _ {k} \left(y _ {k}\right)}{\partial y _ {k}}, k = 1, \dots , K _ {1}, \tag {2.54}
$$

$$
g _ {k} \left(y _ {k}\right) = 2 \sqrt {\tilde {f} \left(y _ {k} ; \tilde {\boldsymbol {\theta}} _ {k}\right)}, k = 1, \dots , K _ {1}. \tag {2.55}
$$

It is easy to get that

$$
\begin{array}{l} \breve {I} _ {F} = \frac {1}{2} \sum_ {k = 1} ^ {K _ {1}} \left\langle \ln \left(\frac {N \alpha_ {k} | g _ {k} ^ {\prime} (y _ {k}) | ^ {2}}{2 \pi e}\right) \right\rangle_ {\mathbf {y}} + H (Y) \\ \leq \frac {1}{2} \sum_ {k = 1} ^ {K _ {1}} \left\langle \ln \left(\frac {\left| g _ {k} ^ {\prime} \left(y _ {k}\right) \right| ^ {2}}{2 \pi e}\right) \right\rangle_ {\mathbf {y}} - \frac {K _ {1}}{2} \ln \left(\frac {K _ {1}}{N}\right) + H (Y), \tag {2.56} \\ \end{array}
$$

where the equality holds if and only if

$$
\alpha_ {k} = \frac {1}{K _ {1}}, k = 1, \dots , K _ {1}, \tag {2.57}
$$

which is consistent with Eq. (2.42).

On the other hand, it follows from the Jensen's inequality that

$$
\begin{array}{l} \check {I} _ {F} = \left\langle \ln \left(p (\mathbf {y}) ^ {- 1} \det  \left(\frac {\check {\mathbf {J}} (\mathbf {y})}{2 \pi e}\right) ^ {1 / 2}\right) \right\rangle_ {\mathbf {y}} \\ \leq \ln \int \det  \left(\frac {\check {\mathbf {J}} (\mathbf {y})}{2 \pi e}\right) ^ {1 / 2} d \mathbf {y}, \tag {2.58} \\ \end{array}
$$

with equality if and only if  $p(\mathbf{y})^{-1}\operatorname*{det}\left(\check{\mathbf{J}} (\mathbf{y})\right)^{1 / 2}$  is a constant, which implies that

$$
p (\mathbf {y}) = \frac {\operatorname* {d e t} \left(\check {\mathbf {J}} (\mathbf {y})\right) ^ {1 / 2}}{\int \operatorname* {d e t} \left(\check {\mathbf {J}} (\mathbf {y})\right) ^ {1 / 2} d \mathbf {y}} = \frac {\prod_ {k = 1} ^ {K _ {1}} \left| g _ {k} ^ {\prime} \left(y _ {k}\right) \right|}{\int \prod_ {k = 1} ^ {K _ {1}} \left| g _ {k} ^ {\prime} \left(y _ {k}\right) \right| d \mathbf {y}}. \tag {2.59}
$$

From (2.58) and (2.59), maximizing  $\tilde{I}_F$  yields

$$
p \left(y _ {k}\right) = \frac {\left| g _ {k} ^ {\prime} \left(y _ {k}\right) \right|}{\int \left| g _ {k} ^ {\prime} \left(y _ {k}\right) \right| d y _ {k}}, k = 1, \dots , K _ {1}. \tag {2.60}
$$

We assume that (2.60) holds, at least approximately. Hence we can let the peak of  $g_{k}^{\prime}(y_{k})$  be at  $y_{k} = \langle y_{k}\rangle_{y_{k}} = 0$  and  $\langle y_k^2\rangle_{y_k} = \sigma_{y_k}^2 = \| \tilde{\mathbf{w}}_k\| ^2$ . Then combining (2.54), (2.58) and (2.60) we can get the optimal parameters  $\tilde{\pmb{\theta}}_k$  for the nonlinear functions  $\tilde{f} (y_k;\tilde{\pmb{\theta}}_k), k = 1,\dots ,K_1$ .

# 2.3 The Final Objective Function

Section 2.2 gives the initial optimal solutions by maximizing  $I(X;Y)$  and  $I(Y;R)$ . In this section, we will discuss how to find the final optimal  $V_{0}$  and other parameters by maximizing  $I(X;R)$  from the initial optimal solutions.

First, we have

$$
\mathbf {y} = \tilde {\mathbf {W}} ^ {T} \tilde {\mathbf {x}} = a \hat {\mathbf {y}}, \tag {2.61}
$$

where  $a$  is given in (2.43) and

$$
\hat {\mathbf {y}} = \left(\hat {y} _ {1}, \dots , \hat {y} _ {K _ {1}}\right) ^ {T} = \mathbf {C} ^ {T} \hat {\mathbf {x}} = \tilde {\mathbf {C}} ^ {T} \tilde {\mathbf {x}}, \tag {2.62}
$$

$$
\hat {\mathbf {x}} = \boldsymbol {\Sigma} _ {0} ^ {- 1 / 2} \mathbf {U} _ {0} ^ {T} \mathbf {x}, \tag {2.63}
$$

$$
\mathbf {C} = \mathbf {V} _ {0} ^ {T} \in \mathbb {R} ^ {K _ {0} \times K _ {1}}, \tag {2.64}
$$

$$
\check {\mathbf {x}} = \mathbf {U} _ {0} \boldsymbol {\Sigma} _ {0} ^ {- 1 / 2} \mathbf {U} _ {0} ^ {T} \mathbf {x} = \mathbf {U} _ {0} \hat {\mathbf {x}}, \tag {2.65}
$$

$$
\tilde {\mathbf {C}} = \mathbf {U} _ {0} \mathbf {C} = \left[ \check {\mathbf {c}} _ {1}, \dots , \check {\mathbf {c}} _ {K _ {1}} \right]. \tag {2.66}
$$

Then, from the above, we get

$$
I (X; R) = I (\tilde {X}; R) \simeq \tilde {I} _ {G} = \frac {1}{2} \left\langle \ln \left(\det  \left(\frac {\mathbf {G} (\hat {\mathbf {x}})}{2 \pi e}\right)\right) \right\rangle_ {\hat {\mathbf {x}}} + H (\tilde {X}), \tag {2.67}
$$

$$
\mathbf {G} (\hat {\mathbf {x}}) = N \hat {\mathbf {W}} \hat {\boldsymbol {\Phi}} \hat {\mathbf {W}} ^ {T} + \mathbf {I} _ {K}, \tag {2.68}
$$

$$
\hat {\mathbf {W}} = \boldsymbol {\Sigma} ^ {1 / 2} \mathbf {U} ^ {T} \mathbf {W} \mathbf {A} ^ {1 / 2} = a \sqrt {K _ {1} ^ {- 1}} \mathbf {I} _ {K _ {0}} ^ {K} \mathbf {C} = \sqrt {K _ {0} ^ {- 1}} \mathbf {I} _ {K _ {0}} ^ {K} \mathbf {C}, \tag {2.69}
$$

where  $\mathbf{I}_{K_0}^K$  is a  $K\times K_{0}$  diagonal matrix with value 1 on the diagonal and

$$
\hat {\Phi} = \Phi^ {2}, \tag {2.70}
$$

$$
\Phi = \operatorname {d i a g} \left(\phi \left(\hat {y} _ {1}\right), \dots , \phi \left(\hat {y} _ {K _ {1}}\right)\right), \tag {2.71}
$$

$$
\phi \left(\hat {y} _ {k}\right) = a ^ {- 1} \left| \frac {\partial g \left(\hat {y} _ {k}\right)}{\partial \hat {y} _ {k}} \right|, \tag {2.72}
$$

$$
\hat {y} _ {k} = a ^ {- 1} y _ {k} = \mathbf {c} _ {k} ^ {T} \hat {\mathbf {x}}, k = 1, \dots , K _ {1}. \tag {2.73}
$$

Then we have

$$
\det  \left(\mathbf {G} (\hat {\mathbf {x}})\right) = \det  \left(N K _ {0} ^ {- 1} \mathbf {C} \hat {\boldsymbol {\Phi}} \mathbf {C} ^ {T} + \mathbf {I} _ {K _ {0}}\right), \tag {2.74}
$$

for large  $N$  and  $K_0 / N\to 0$

$$
\det  \left(\mathbf {G} (\hat {\mathbf {x}})\right) \approx \det  \left(\mathbf {J} (\hat {\mathbf {x}})\right) = \det  \left(N K _ {0} ^ {- 1} \mathbf {C} \hat {\boldsymbol {\Phi}} \mathbf {C} ^ {T}\right), \tag {2.75}
$$

and

$$
\tilde {I} _ {G} \approx \tilde {I} _ {F} = - Q - \frac {K}{2} \ln (2 \pi e) - \frac {K _ {0}}{2} \ln (\varepsilon) + H (\tilde {X}), \tag {2.76}
$$

$$
Q = - \frac {1}{2} \left\langle \ln \left(\det  \left(\mathbf {C} \hat {\boldsymbol {\Phi}} \mathbf {C} ^ {T}\right)\right) \right\rangle_ {\hat {\mathbf {x}}}, \tag {2.77}
$$

$$
\varepsilon = \frac {K _ {0}}{N}. \tag {2.78}
$$

Hence we can state the optimization problem as:

$$
\operatorname {m i n i m i z e} Q [ \mathbf {C} ] = - \frac {1}{2} \left\langle \ln \left(\det  \left(\mathbf {C} \hat {\boldsymbol {\Phi}} \mathbf {C} ^ {T}\right)\right) \right\rangle_ {\hat {\mathbf {x}}}, \tag {2.79}
$$

$$
\text {s u b j e c t} \mathbf {C} \mathbf {C} ^ {T} = \mathbf {I} _ {K _ {0}}. \tag {2.80}
$$

The gradient from (2.79) reads as follows:

$$
\frac {d Q [ \mathbf {C} ]}{d \mathbf {C}} = - \left\langle \left(\mathbf {C} \hat {\boldsymbol {\Phi}} \mathbf {C} ^ {T}\right) ^ {- 1} \mathbf {C} \hat {\boldsymbol {\Phi}} + \hat {\mathbf {x}} \boldsymbol {\omega} ^ {T} \right\rangle_ {\hat {\mathbf {x}}}, \tag {2.81}
$$

where  $\mathbf{C} = [\mathbf{c}_1,\dots ,\mathbf{c}_{K_1}],\pmb {\omega} = (\omega_1,\dots ,\omega_{K_1})^T$

$$
\omega_ {k} = \phi (\hat {y} _ {k}) \phi^ {\prime} (\hat {y} _ {k}) \mathbf {c} _ {k} ^ {T} \left(\mathbf {C} \hat {\boldsymbol {\Phi}} \mathbf {C} ^ {T}\right) ^ {- 1} \mathbf {c} _ {k}, k = 1, \dots , K _ {1}. \tag {2.82}
$$

In the following we will discuss how to get the optimal solution of  $\mathbf{C}$  for two specific cases.

# 2.3.1 Algorithm 1:  $K_{0} = K_{1}$

Now  $\mathbf{C}\mathbf{C}^T = \mathbf{C}^T\mathbf{C} = \mathbf{I}_{K_1}$ , then by Eq. (2.79) we have

$$
Q _ {1} [ \mathbf {C} ] = - \left\langle \sum_ {k = 1} ^ {K _ {1}} \ln \left(\phi \left(\hat {y} _ {k}\right)\right) \right\rangle_ {\hat {\mathbf {x}}}, \tag {2.83}
$$

$$
\frac {d Q _ {1} [ \mathbf {C} ]}{d \mathbf {C}} = - \left\langle \hat {\mathbf {x}} \boldsymbol {\omega} ^ {T} \right\rangle_ {\hat {\mathbf {x}}}, \tag {2.84}
$$

$$
\omega_ {k} = \frac {g ^ {\prime \prime} \left(\hat {y} _ {k}\right)}{g ^ {\prime} \left(\hat {y} _ {k}\right)}, k = 1, \dots , K _ {1}. \tag {2.85}
$$

Under the orthogonality constraints (Eq. 2.80), we can use the following update rule for learning  $\mathbf{C}$  (Edelman et al., 1998; Amari, 1999),

$$
\mathbf {C} ^ {t + 1} = \mathbf {C} ^ {t} + \mu_ {t} \frac {d \mathbf {C} ^ {t}}{d t}, \tag {2.86}
$$

$$
\frac {d \mathbf {C} ^ {t}}{d t} = - \frac {d Q _ {1} [ \mathbf {C} ^ {t} ]}{d \mathbf {C} ^ {t}} + \mathbf {C} ^ {t} \left(\frac {d Q _ {1} [ \mathbf {C} ^ {t} ]}{d \mathbf {C} ^ {t}}\right) ^ {T} \mathbf {C} ^ {t}, \tag {2.87}
$$

where the learning rate parameter  $\mu_t$  changes with the iteration count  $t$ ,  $t = 1, \dots, t_{\max}$ . Here we can use the empirical average to approximate the integral in (2.84) (ref. Eq. 2.15). We can also apply stochastic gradient descent (SGD) method for online updating  $\mathbf{C}^{t + 1}$  in (2.86).

The orthogonality constraint (Eq. 2.80) can accelerate the convergence rate. In practice, the orthogonal constraint (2.80) for objective function (2.79) is not necessary in this case. We can completely abandon the constraint condition,

$$
\text {m i n i m i z e} Q _ {2} [ \mathbf {C} ] = - \left\langle \sum_ {k = 1} ^ {K _ {1}} \ln \left(\phi \left(\hat {y} _ {k}\right)\right) \right\rangle_ {\hat {\mathbf {x}}} - \frac {1}{2} \ln \left(\det \left(\mathbf {C} ^ {T} \mathbf {C}\right)\right), \tag {2.88}
$$

where we assume rank  $(\mathbf{C}) = K_{1} = K_{0}$ . If we let

$$
\frac {d \mathbf {C}}{d t} = - \mathbf {C} \mathbf {C} ^ {T} \frac {d Q _ {2} [ \mathbf {C} ]}{d \mathbf {C}}, \tag {2.89}
$$

then

$$
\operatorname {T r} \left(\frac {d Q _ {2} [ \mathbf {C} ]}{d \mathbf {C}} \frac {d \mathbf {C} ^ {T}}{d t}\right) = - \operatorname {T r} \left(\mathbf {C} ^ {T} \frac {d Q _ {2} [ \mathbf {C} ]}{d \mathbf {C}} \frac {d Q _ {2} [ \mathbf {C} ]}{d \mathbf {C} ^ {T}} \mathbf {C}\right) \leq 0. \tag {2.90}
$$

Therefore we can use the similar update rule (see Eq. 2.86) for learning  $\mathbf{C}$ . In fact, it can also be extended to the case  $K_{0} > K_{1}$  by using the same objective function (2.88).

For the learning rate parameter  $\mu_t$  (see 2.86), we apply an adaptive method for updating it. First, calculate

$$
\mu_ {t} = \frac {v _ {t}}{\kappa_ {t}}, t = 1, \dots , t _ {\max }, \tag {2.91}
$$

$$
\kappa_ {t} = \frac {1}{K _ {1}} \sum_ {k = 1} ^ {K _ {1}} \frac {\left\| \nabla \mathbf {C} ^ {t} (: , k) \right\|}{\left\| \mathbf {C} ^ {t} (: , k) \right\|}, \tag {2.92}
$$

and  $\mathbf{C}^{t + 1}$  by (2.86) and (2.87), then calculate the value  $Q_{1}\big[\mathbf{C}^{t + 1}\big]$ . If  $Q_{1}\big[\mathbf{C}^{t + 1}\big] < Q_{1}[\mathbf{C}^{t}]$ , then let  $v_{t + 1} \gets v_t$ , continue for next iteration; otherwise, let  $v_t \gets \tau v_t$ ,  $\mu_t \gets v_t / \kappa_t$  and recalculate  $\mathbf{C}^{t + 1}$  and  $Q_{1}\big[\mathbf{C}^{t + 1}\big]$ . Here  $0 < v_1 < 1$  and  $0 < \tau < 1$  are set as constants. After getting  $\mathbf{C}^{t + 1}$  for each update, we employ a Gram-Schmidt orthonormalization process for matrix  $\mathbf{C}^{t + 1}$ , where the orthonormalization process can accelerate the convergence. However, we can discard the Gram-Schmidt orthonormalization process after iterative epochs  $t_0 (> 1)$  for more accurate optimization solution  $\mathbf{C}$ . In this case, the objective function is given by the Eq. (2.88). We can also further optimize the parameter  $b$  by gradient descent.

When  $K_{0} = K_{1}$ , the objective function (2.88)  $Q_{2}$  [C] without constraint is the same as the objective function of infomax ICA (IICA) (Bell & Sejnowski, 1995; 1997), and we can get the same optimal

solution C. Hence, in this sense, the IICA can be regarded as a special case of our method. However, our method has a wider range of application and can handle a number of generic situations. Our model is derived by neural populations with a huge number of neurons and not only for additive noise neuron model. Moreover, our method can get a faster convergence rate for training than IICA (see Section 3).

# 2.3.2 Algorithm 2:  $K_{0} \leq K_{1}$

In this case, it is computationally expensive to update  $\mathbf{C}$  by using the gradient of  $Q$  (see Eq. 2.81), since it needs to compute the inverse matrix for every  $\hat{\mathbf{x}}$ . Consider the following inequalities (see Appendix for proof).

Proposition 2.2. The following inequations hold,

$$
\left. \frac {1}{2} \left\langle \ln \left(\det  \left(\mathbf {C} \hat {\boldsymbol {\Phi}} \mathbf {C} ^ {T}\right)\right) \right\rangle_ {\hat {\mathbf {x}}} \leq \frac {1}{2} \ln \left(\det  \left(\mathbf {C} \left\langle \hat {\boldsymbol {\Phi}} \right\rangle_ {\hat {\mathbf {x}}} \mathbf {C} ^ {T}\right)\right), \right. \tag {2.93}
$$

$$
\left. \left\langle \ln \left(\det \left(\mathbf {C} \boldsymbol {\Phi} \mathbf {C} ^ {T}\right)\right) \right\rangle_ {\hat {\mathbf {x}}} \leq \ln \left(\det \left(\mathbf {C} \left\langle \boldsymbol {\Phi} \right\rangle_ {\hat {\mathbf {x}}} \mathbf {C} ^ {T}\right)\right) \right. \tag {2.94}
$$

$$
\leq \frac {1}{2} \ln \left(\det  \left(\mathbf {C} \langle \boldsymbol {\Phi} \rangle_ {\hat {\mathbf {x}}} ^ {2} \mathbf {C} ^ {T}\right)\right) \tag {2.95}
$$

$$
\leq \frac {1}{2} \ln \left(\det  \left(\mathbf {C} \left\langle \hat {\boldsymbol {\Phi}} \right\rangle_ {\hat {\mathbf {x}}} \mathbf {C} ^ {T}\right)\right), \tag {2.96}
$$

$$
\ln \left(\det  \left(\mathbf {C} \boldsymbol {\Phi} \mathbf {C} ^ {T}\right)\right) \leq \frac {1}{2} \ln \left(\det  \left(\mathbf {C} \hat {\boldsymbol {\Phi}} \mathbf {C} ^ {T}\right)\right), \tag {2.97}
$$

where  $\mathbf{C} \in \mathbb{R}^{K_0 \times K_1}$ ,  $K_0 \leq K_1$ ,  $\mathbf{C}\mathbf{C}^T = \mathbf{I}_{K_0}$ .

We use the following objective function  $\hat{Q}[\mathbf{C}]$  as a substitute for  $Q[\mathbf{C}]$  (see Appendix for detailed discussion) and write the optimization problem as:

$$
\operatorname {m i n i m i z e} \hat {Q} [ \mathbf {C} ] = - \frac {1}{2} \ln \left(\det  \left(\mathbf {C} \langle \boldsymbol {\Phi} \rangle_ {\hat {\mathbf {x}}} ^ {2} \mathbf {C} ^ {T}\right)\right), \tag {2.98}
$$

$$
\text {s u b j e c t} \mathbf {C} \mathbf {C} ^ {T} = \mathbf {I} _ {K _ {0}}. \tag {2.99}
$$

The update rule (2.86) may also apply here and a modified algorithm similar to algorithm 1 may be used for parameter learning.

# 3 EXPERIMENTAL RESULTS

We performed our all experiments using Matlab 2016a on a machine with 12 Intel CPU cores (at 2.4 GHz) and have applied our methods to natural images from Olshausen's image dataset (Olshausen & Field, 1996) and images of handwritten digits from MNIST dataset (LeCun et al., 1998). The gray level of each raw image was normalized to the range of 0 to 1. The  $M$  image patches with size  $w \times w = K$  for training are randomly sampled from the natural images. We use the Poisson neuron model with the sigmoid like tuning function  $\tilde{f}(y; \tilde{\pmb{\theta}}) = \frac{1}{4(1 + \exp(-\beta y - b))^2}$ , then  $g(y) = 2\sqrt{\tilde{f}(y; \tilde{\pmb{\theta}})} = \frac{1}{1 + \exp(-\beta y - b)}$ , where  $\tilde{\pmb{\theta}} = (\beta, b)^T$ . We can solve their initial value (see section 2.2.2),  $b_0 = 0$ ,  $\beta_0 \approx 1.81\sqrt{K_1K_0^{-1}}$ . For our experiments, we set  $\beta = 0.5\beta_0$  for  $t = 1, \dots, t_0$  and  $\beta = \beta_0$  for  $t = t_0 + 1, \dots, t_{\max}$ .

Firstly, we tested the case of  $K = K_{0} = K_{1} = 144$  and randomly sampled  $M = 10^{5}$  image patches with size  $12 \times 12$  from the Olshausen's natural images, assuming that  $N = 10^{6}$  neurons were divided into  $K_{1} = 144$  classes and  $\epsilon = 1$  (see Eq. 2.49). The input patches were preprocessed by a ZCA whitening filter (see Eq. 2.65). To test our algorithms, we chose the batch size to be equal to the number of training samples  $M$ , although we could also choose a smaller batch size. We updated the matrix  $\mathbf{C}$  from a random start, and set parameters  $t_{\mathrm{max}} = 300$ ,  $t_{0} = 50$ ,  $v_{1} = 0.4$ , and  $\tau = 0.8$  for all experiments.

In this case, the optimal solution  $\mathbf{C}$  looked similar to the optimal solution of IICA (Bell & Sejnowski, 1997). We also compared the fast ICA (FICA) (Hyvarinen, 1999), a faster algorithm than IICA.

We also tested the restricted Boltzmann machine (RBM) (Hinton, 2002; Hinton et al., 2006) for a unsupervised learning of representations, and found that it could not easily learn Gabor-like filters as trained by contrastive divergence with Olshausen's image dataset. However, an improved method by adding a sparsity constraint on the output units, e.g., sparse RBM (SRBM) (Lee et al., 2008) or sparse autoencoder (Hinton, 2010; Ng, 2011), can attain Gabor-like filters from this dataset. Similar results (i.e. Gabor-like filters) were also reproduced by the denoising autoencoders (Vincent et al., 2010), which method requires a careful choice of parameters, such as noise level, learning rate, and batch size.

In order to compare our methods, i.e. algorithm 1 (Alg.1) and algorithm 2 (Alg.2), with other methods, i.e. IICA, FICA and SRBM, we implemented these algorithms by using the same initial weights and the same training data set (i.e.  $10^{5}$  image patches preprocessed by a ZCA whitening filter). To get a good result by IICA, we must carefully select the parameters; we set the batch size as 50, the initial learning rate as 0.01, and final learning rate as 0.0001, with an exponential decay with the epoch of iterations. IICA tends to have a faster convergence rate for a bigger batch size but it may become harder to escape local minima. For FICA, we chose the nonlinearity function  $f(u) = \log \cosh(u)$  as contrast function, and for SRBM, we set the sparseness control constant  $p$  as 0.01 and 0.03. The number of epochs for iterations was set to 300 for all algorithms. Figure 2 shows the filters learnt by our methods and other methods. Each filter in Figure 2(a) corresponds to a column vector of matrix  $\check{\mathbf{C}}$  (see Eq. 2.66), where each vector for display is normalized by  $\check{\mathbf{c}}_k \gets \check{\mathbf{c}}_k / \max(|\check{c}_{1,k}|, \dots, |\check{c}_{K,k}|)$ ,  $k = 1, \dots, K_1$ . The results in Figures 2(a), 2(b) and 2(c) look very similar to one another, and slightly different from the results in Figure 2(d) and 2(e). There are no Gabor-like filters in Figure 2(f), which corresponds to SRBM with  $p = 0.03$ .

To quantify the efficiency of learning representations by the above algorithms, we calculate the coefficient entropy (CFE) for estimating coding cost (Lewicki & Olshausen, 1999; Lewicki & Sejnowski, 2000) as follows:

$$
\check {y} _ {k} = \zeta \tilde {\mathbf {w}} _ {k} ^ {T} \check {\mathbf {x}}, k = 1, \dots , K _ {1}, \tag {3.100}
$$

$$
\zeta = \frac {K _ {1}}{\sum_ {k = 1} ^ {K _ {1}} \| \breve {\mathbf {w}} _ {k} \|.} \tag {3.101}
$$

where  $\check{\mathbf{x}}$  is defined by Eq. (2.65), and  $\check{\mathbf{w}}_k$  is the corresponding optimal filter. To estimate the probability density of coefficients  $q_{k}(\check{y}_{k})$  ( $k = 1, \dots, K_{1}$ ) from the  $M$  training samples, we apply the kernel density estimation for  $q_{k}(\check{y}_{k})$  and use a normal kernel with an adaptive optimal window width. Then we define the CFE  $h$  as

$$
h = \frac {1}{K _ {1}} \sum_ {k = 1} ^ {K _ {1}} H _ {k} \left(\check {Y} _ {k}\right), \tag {3.102}
$$

$$
H _ {k} \left(\check {Y} _ {k}\right) = - \Delta \sum_ {n} q _ {k} (n \Delta) \log_ {2} q _ {k} (n \Delta), \tag {3.103}
$$

where  $q_{k}(\check{y}_{k})$  is quantized as discrete  $q_{k}(n\Delta)$  and  $\Delta$  is the step size.

Methods such as IICA and SRBM as well as our methods are feedforward structures in which information is transferred directly through a nonlinear function, e.g., the sigmoid function. Hence, we can use the amount of transmitted information to measure the results learnt by these methods. We consider a stochastic system with nonlinear transfer functions, which is a neural population with  $N$  neurons. We chose the sigmoid function as the transfer function and Gaussian noise with standard deviation set to 1 as the system noise. In this case, from (2.2), (A.2) and (A.5), we know that the proximate MI  $I_G$  is equivalent to Poisson neuron model. It follows from (2.67)-(2.78) that

$$
I (X; R) = I (\tilde {X}; R) = H (\tilde {X}) - H (\tilde {X} | R) \simeq \tilde {I} _ {G} = H (\tilde {X}) - h _ {1}, \tag {3.104}
$$

$$
\left. H (\tilde {X} | R) \simeq h _ {1} = - \frac {1}{2} \left\langle \ln \left(\det  \left(\frac {1}{2 \pi e} \left(N K _ {0} ^ {- 1} \mathbf {C} \hat {\boldsymbol {\Phi}} \mathbf {C} ^ {T} + \mathbf {I} _ {K _ {0}}\right)\right)\right) \right\rangle_ {\hat {\mathbf {x}}} \right., \tag {3.105}
$$

where we set  $N = 10^6$ . A good representation should make the MI  $I(X;R)$  as big as possible. Equivalently, for the same inputs, a good representation should make the conditional entropy (CDE)  $H(\tilde{X} | R)$  (or  $h_1$ ) as small as possible.

Figure 3 shows how CFE and CDE varied with training time. We calculated CFE and CDE by sampling once every 10 epochs from a total of 300 epochs. These results show that our algorithms

![](images/901fc82e7fdff97f127150fa9b9b95d966f19271fb5c5df51f4f499127e3046b.jpg)  
(a)

![](images/0b8eb0e1373797ff80783041a30a93c264568c63125cfb86fb73347674743135.jpg)  
(b)

![](images/66705f554e060da39b5831821f4b9d6e40f032772f797a0fafadd891324b7800.jpg)  
(c)

![](images/389d3e5fd2fea7bb1771c60ea4edbf451557ea8422326e0897e7052883ebb52d.jpg)  
(d)

![](images/cf47e96bf9503ad17898bb4794684a0b68f844e453c48ba05dd3ab7a52953c87.jpg)  
(e)

![](images/f28703acf4ae5f87b6b0484b7c94f5d76d7e6a308d1b748e2987ab4551a251c5.jpg)  
(f)

![](images/c44c7f55f1977907fcdf061874e03604476791cc6584776efc21faaec564aaac.jpg)  
(a)

![](images/23a361505ee495fb48c59c013c7384ef3d1560ec0bd474aa7a9b9862e8dce2d3.jpg)  
Figure 2: Comparison of filters obtained from  $10^{5}$  natural image patches with size  $12 \times 12$  by our methods (Alg.1 and Alg.2) and other methods. The number of output filters is  $K_{1} = 144$ . (a) and (b): Alg.1 and Alg.2, respectively. (c): IICA. (d): FICA. (e): SRBM ( $p = 0.01$ ). (f): SRBM ( $p = 0.03$ ).  
(b)

![](images/a85e7a537862a5a7c3cb8a5fb49cf2a3a02da7ddf2ac34e60e578f765271b6e5.jpg)  
(c)  
Figure 3: Comparison of quantization effects and convergence rate by coefficient entropy (ref. 3.102) and conditional entropy (ref. 3.105) corresponding to training results (filters) shown in Figure 2. The coefficient entropy (panel a) or conditional entropy (panel b and c) are shown a function of training time on a logarithmic scale. All experiments run on the same machine using Matlab. Here we sampled once every 10 epochs out of a total of 300 epochs. Notice that we set epoch number  $t_0 = 50$  for Alg.1 and Alg.2 and the start time was set to 1 second. See main text for more details.

had a fast convergence rate towards stable solutions while having CFE and CDE values similar to the algorithm of IICA, which converged much more slowly. Here the values of CFE and CDE should be as small as possible for a good representation learnt from the same data set. Notice that here we set epoch number  $t_0 = 50$  in our algorithms (see Alg.1 and Alg.2), and the start time was set to 1 second. This explains the step seen in Figure 3 (b) for Alg.1 and Alg.2 since the parameter  $\beta$  was updated when epoch number  $t = t_0$ . FICA had a convergence rate close to our algorithms but had a big CFE, which is reflected by the quality of the filter results in Figure 2. The convergence rate and CFE for SRBM were close to IICA, but SRBM had a much bigger CDE than IICA, which implies

that the information had a greater loss when passing through the system optimized by SRBM than by IICA or our methods.

From Figure 3(c) we see that the CDE (or MI  $I(X;R)$ , see Eq. 3.104 and 3.105) decreases (or increases) with the increase of the value of the sparseness control constant  $p$ . Note that a smaller  $p$  means sparser outputs. Hence, in this sense, increasing sparsity may result in sacrificing some information. On the other hand, a weak sparsity constraint may lead to failure of learning a significant result (i.e. Gabor-like filters, see Figure 2(f)), and increasing sparsity has an advantage in eliminating the impact of noise in many practical cases.

Similar situation also occurs in sparse coding (Olshausen & Field, 1997), which provides a class of algorithms for learning overcomplete dictionary representations of the input signals. However, its training is time consuming due to its expensive computational cost, although many new training algorithms have emerged (Aharon et al., 2006; Elad & Aharon, 2006; Lee et al., 2006; Mairal et al., 2009; 2010). We compared our algorithm with an up-to-date sparse coding algorithm, minibatch dictionary learning (MBDL) as given in (Mairal et al., 2009; 2010) and integrated in Python library, i.e. scikit-learn. The input data was the same as the above, i.e.  $10^{5}$  nature image patches preprocessed by a ZCA whitening filter.

We denote the optimal dictionary learnt by MBDL as  $\breve{\mathbf{B}}\in \mathbb{R}^{K\times K_1}$  for which each column represents a basis vector, then we have that

$$
\mathbf {x} \approx \mathbf {U} \boldsymbol {\Sigma} ^ {1 / 2} \mathbf {U} ^ {T} \tilde {\mathbf {B}} \mathbf {y} = \tilde {\mathbf {B}} \mathbf {y}, \tag {3.106}
$$

$$
\tilde {\mathbf {B}} = \mathbf {U} \boldsymbol {\Sigma} ^ {1 / 2} \mathbf {U} ^ {T} \tilde {\mathbf {B}}, \tag {3.107}
$$

where  $\mathbf{y} = (y_{1},\dots ,y_{K_{1}})^{T}$  is the coefficient vector.

Similarly, we can obtain a dictionary from the filter matrix  $\mathbf{C}$ . Suppose  $\operatorname{rank}(\mathbf{C}) = K_0 \leq K_1$ , then it follows from (2.61) that

$$
\hat {\mathbf {x}} = \left(a \mathbf {C} \mathbf {C} ^ {T}\right) ^ {- 1} \mathbf {C} \mathbf {y}. \tag {3.108}
$$

And by (2.63) and (3.108), we get

$$
\mathbf {x} \approx \mathbf {B} \mathbf {y} = a \mathbf {B} \mathbf {C} ^ {T} \boldsymbol {\Sigma} _ {0} ^ {- 1 / 2} \mathbf {U} _ {0} ^ {T} \mathbf {x}, \tag {3.109}
$$

$$
\mathbf {B} = a ^ {- 1} \mathbf {U} _ {0} \boldsymbol {\Sigma} _ {0} ^ {1 / 2} \left(\mathbf {C C} ^ {T}\right) ^ {- 1} \mathbf {C} = \left[ \mathbf {b} _ {1}, \dots , \mathbf {b} _ {K _ {1}} \right], \tag {3.110}
$$

where  $\mathbf{y} = \mathbf{W}^T\mathbf{x} = a\mathbf{C}^T\Sigma_0^{-1 / 2}\mathbf{U}_0^T\mathbf{x}$ , the vectors  $\mathbf{b}_1,\dots ,\mathbf{b}_{K_1}$  can be regarded as the basis vectors and the strict equality holds when  $K_{0} = K_{1} = K$ . Recall that  $\mathbf{X} = [\mathbf{x}_1,\dots ,\mathbf{x}_M] = \mathbf{US}\tilde{\mathbf{V}}^T$  (see Eq. A.6) and  $\mathbf{Y} = [\mathbf{y}_1,\dots ,\mathbf{y}_M] = \mathbf{W}^T\mathbf{X} = a\sqrt{M - 1}\mathbf{C}^T\tilde{\mathbf{V}}_0^T$ , then we can get  $\check{\mathbf{X}} = \mathbf{B}\mathbf{Y} = \sqrt{M - 1}\mathbf{U}_0\Sigma_0^{1 / 2}\tilde{\mathbf{V}}_0^T\approx \mathbf{X}$ . Hence, the Eq. (3.109) holds.

The basis vectors shown in Figure 4(a)–4(e) correspond to filters in Figure 2(a)–2(e). And Figure 4(f) illustrates the optimal dictionary  $\tilde{\mathbf{B}}$  learnt by MBDL, where we set the regularization parameter as  $\lambda = 1.2 / \sqrt{K}$ , the batch size as 50 and the total number of iterations to perform as 20000, which took about 3 hours for training. From Figure 4 we see that these basis vectors obtained by the above algorithms have local Gabor-like shapes except for those by SRBM. If rank  $(\breve{\mathbf{B}}) = K = K_{1}$ , then the matrix  $\breve{\mathbf{B}}^{-T}$  can be regarded as a filter matrix like matrix  $\breve{\mathbf{C}}$  (see Eq. 2.66). However, from the column vector of matrix  $\breve{\mathbf{B}}^{-T}$  we cannot find any local Gabor-like filter that resembles the filters shown in Figure 2. Our algorithm has less computational cost and a much faster convergence rate than the sparse coding algorithm. Moreover, the sparse coding method involves a dynamic generative model that requires relaxation and is therefore unsuitable for fast inference, whereas the feedforward framework of our model is easy for inference because it only requires evaluating the nonlinear tuning functions.

We have trained our model on the same nature image patches with a highly overcomplete setup by optimizing the objective (2.98) by Alg.2 and got Gabor-like filters. The results of 400 typical filters chosen from 1024 output filters are displayed in Figure 5(a) and corresponding base (see Eq. 3.110) are shown in Figure 5(b). In this case, we set parameters  $K_{1} = 1024$ ,  $t_{\max} = 100$ ,  $t_{0} = 50$ ,  $v_{1} = 0.4$ ,  $\tau = 0.8$ ,  $\epsilon = 0.98$  (see 2.49), and got rank  $(\mathbf{B}) = K_{0} = 82$ . Compared to the ICA-like

![](images/165aa3df3956bedafcf527577b3e36d145d7a09b8b2a6cbcfdf8fde0c965233a.jpg)  
(a)

![](images/58703f8a75c756928c2a98a0c4867cbba3b92549daa1be415c5d135e7b0f8494.jpg)  
(b)

![](images/19fd45bd837c2fe85239008d7610b85e22bdf6e74f9c80a9b7cad423ec67a3e8.jpg)  
(c)

![](images/5f00f022a8fc1a76205a1cda2e6706a4e1f374d760aa46e738054ecd46b829af.jpg)  
(d)

![](images/d2df65a9be38762151c5ef8e9aef3128904edbbdebdb61d7f85ed14ef0089129.jpg)  
(e)

![](images/a353c2de66e9ff3f5f3fbb61a7fc96569b4ce2896b6c0d52cc99eecbf4a8a35f.jpg)  
(f)  
Figure 4: Comparison of basis vectors obtained by our method and other methods. Panel (a)-(e) correspond to panel (a)-(e) in Figure 2, where the basis vectors are given by (3.110). The basis vectors in panel (f) are learnt by MBDL and given by (3.107).

results in Figure 2(a)–Figure 2(c), the average size of Gabor-like filters in Figure 5(a) is bigger, indicating that the small noise-like local structures in images have been filtered out.

We have also trained our model on 60000 images of handwritten digits from MNIST dataset (LeCun et al., 1998) and the resultant 400 typical optimal filters and bases are shown in Figure 5(c) and Figure 5(d), respectively. All parameters were the same as Figure 5(a) and Figure 5(b):  $K_{1} = 1024$ ,  $t_{\mathrm{max}} = 100$ ,  $t_0 = 50$ ,  $v_{1} = 0.4$ ,  $\tau = 0.8$  and  $\epsilon = 0.98$ , from which we got rank  $(\mathbf{B}) = K_{0} = 183$ . From these figures we can see that the salient features of the input images are reflected in these filters and bases. We could also get the similar overcomplete filters and bases by SRBM and MBDL. However, the results depended sensitively on the choice of parameters and the training took a long time.

Figure 6 shows that CFE as a function of training time for Alg.2, where Figure 6(a) corresponds to Figure 5(a)-5(b) for learning nature image patches and Figure 6(b) corresponds to Figure 5(c)-5(d) for learning MNIST dataset. We set parameters  $t_{\mathrm{max}} = 100$ ,  $t_0 = 50$  and  $\tau = 0.8$  for all experiments and varied parameter  $v_{1}$  for each experiment, with  $v_{1} = 0.2$ , 0.4, 0.6 or 0.8. These results indicate a fast convergence rate for training on different datasets. Generally, the convergence is not sensitive to the change of parameter  $v_{1}$ .

We have also performed additional tests on other image datasets and got similar results, confirming the speed and robustness of our learning method. Compared with other methods, e.g., IICA, FICA, MBDL, SRBM or sparse autoencoders etc., our method appeared to be more efficient and robust for unsupervised learning of representations. We also found that the complete or overcomplete filters and bases learnt by our methods have the local Gabor-like shapes while the results by SRBM or MBDL do not have this property.

Similar to the sparse coding method applied to image denoising (Elad & Aharon, 2006), our method (see Eq. 3.110) can also be applied to image denoising. Figure 7 shows an example of image denoising. The filters or bases were learnt by using image patches with size  $7 \times 7$  sampled from the left half of the image, and subsequently used to reconstruct the right half of the image which was

![](images/9081121903f23e30b1e76479471a54788705b57524ad0e36a2720211de9988ba.jpg)  
(a)

![](images/7a1aa74dc679642a98c4bddf229438359c2ce1cc9c81c55bf33d3c3a86553bc2.jpg)  
(b)

![](images/402e99b07b4aa7f988f15de6b69b2ae00d06642c287f50206458cdd393b4b60c.jpg)  
(c)  
Figure 5: Filters and bases obtained from Olshausen's image dataset and MNIST dataset by Algorithm 2. (a) and (b): 400 typical filters and the corresponding bases obtained from Olshausen's image dataset, where  $K_{0} = 82$  and  $K_{1} = 1024$ . (c) and (d): 400 typical filters and the corresponding bases obtained from the MNIST dataset, where  $K_{0} = 183$  and  $K_{1} = 1024$ .

![](images/c1827638974ca590f9713d43cd149f1707071b9c16d92dcc12aaebbfb65f11ba.jpg)  
(d)

distorted by Gaussian noise. A common practice for evaluating the results of image denoising is by looking at the difference between the reconstruction and the original image. If the reconstruction is perfect the difference will look like Gaussian noise. In Figure 7(c) and 7(d) a dictionary (100 bases) was learnt by MBDL and orthogonal matching pursuit was used to estimate the sparse solution  $^1$ . For our method (showed in Figure 7(b)), we first get the optimal filters parameter  $\mathbf{W}$ , where  $\mathbf{W}$  is a low rank matrix  $(K_0 < K)$ , then from the distorted image patches  $\mathbf{x}_m$ $(m = 1,\dots ,M)$ , we can get filter outputs  $\mathbf{y}_m = \mathbf{W}^T\mathbf{x}_m$  and reconstruction  $\check{\mathbf{x}}_m = \mathbf{B}\mathbf{y}_m$ , where we set parameter  $\epsilon = 0.975$  and  $K_{0} = K_{1} = 14$ . As can be seen from Figure 7, our method worked better than dictionary learning, although we only used 14 bases comparing with 100 bases used by dictionary learning. Moreover, our method is more efficient. In fact, we can get better optimal bases  $\mathbf{B}$  by a generative model using our infomax approach, which we will discuss in another paper.

![](images/4243f3c968d70db4ec27ae0318728a4f7ab9089628b8fb35a8bb46f90046ee0c.jpg)  
(a)

![](images/8dd34d845fe091855890e162f847ebf8d2e4363fc51efff07534711e1d249b4d.jpg)  
(b)  
Figure 6: CFE as a function of training time for Alg.2, with  $v_{1} = 0.2$ , 0.4, 0.6 or 0.8. In all experiments parameters were set to  $t_{\mathrm{max}} = 100$ ,  $t_{0} = 50$  and  $\tau = 0.8$ . (a): corresponding to Figure 5(a) or Figure 5(b). (b): corresponding to Figure 5(c) or Figure 5(d).

![](images/56f93bd2c62ce27bc749f4cba9f75b816a2b0ef7b62143348fe0d39c4010a8d0.jpg)  
Distorted image  
(a)

![](images/61f636f4511cb3e2bb9b67ba78b729651f7fc21ec45dd6141cd4a3b1b82d57ba.jpg)  
Algorithm 1  
(b)

![](images/dbc9ff7a76916e90b6d5760b6840e14ecc67fe9a8f00830ab99465d3e9be5fde.jpg)  
Orthogonal Matching Pursuit 1 atom  
(c)  
Figure 7: Image denoising. (a): the right half of the original image is distorted by Gaussian noise and the norm of the difference between the distorted image and the original image is 23.48. (b): image denoising by our method (Algorithm 1), with 14 bases used. (c) and (d): image denoising using dictionary learning, with 100 bases used.

![](images/8c495947f5fdb9b0c3d1b9f41ab5b26139f7a4779193bbf659f0e0646bf839c3.jpg)  
Orthogonal Matching Pursuit 2 atoms  
(d)

# 4 CONCLUSIONS

In this paper, we have presented a framework for unsupervised learning of representations via information maximization for neural populations. Information theory is a powerful tool for machine learning and it also provides a benchmark of optimization principle for neural information processing in nervous systems. In our framework, first, we give an asymptotic approximation to MI for a large-scale neural population. Then, to solve the infomax objective, we demonstrate that hierarchical infomax can provide a good approximation to the global optimal solution. Analytical solutions of the hierarchical infomax are further improved by a fast convergence algorithm based on gradient de

scent. Using this method, we can optimize a highly nonlinear function via hierarchical optimization using infomax principle.  
Moreover, from the viewpoint of information theory, the unsupervised pre-training for deep learning (Hinton & Salakhutdinov, 2006; Bengio et al., 2007) may be reinterpreted as a process of hierarchical infomax given by our framework, which might help explain why unsupervised pre-training helps deep learning (Erhan et al., 2010). In our framework, a pre-whitening step can emerge naturally by the hierarchical infomax, which might also explain why a pre-whitening step is useful for training in many learning algorithms (Coates et al., 2011; Bengio, 2012).  
Our model naturally incorporates a considerable degree of biological realism, allowing a large-scale neural population with Poisson neuron model, and taking into account of several actual biological constraints, such as membrane noise, limited energy, and bounded connection weights. We employ a technique to attain a low-rank weight matrix for optimization, so as to avoid the influence of noise and reduce overfitting during training. In our model, many parameters have been optimized, including the population density of parameters, filter weight vectors, and parameters for nonlinear tuning functions, which could not be done by many other methods.  
Our experimental results suggest that our method for unsupervised learning of representations has obvious advantages in its training speed and robustness over the main existing methods. Our model has a nonlinear feedforward structure and is convenient for fast learning and inference, which provides a simple and flexible framework for unsupervised learning of presentations and should be easily extended to training deep structure networks. In future work, it would interesting to use our method to train deep structure networks with unsupervised or supervised learning.

# ACKNOWLEDGMENTS

We thank Prof. Honglak Lee for sharing Matlab code for algorithm comparison, Prof. Shan Tan for discussions and comments and Kai Liu for helping draw Figure 1. Supported by grant NIH-NIDCD R01 DC013698.

# REFERENCES

Aharon, M., Elad, M., & Bruckstein, A. (2006). K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation. Signal Processing, IEEE Transactions on, 54(11), 4311-4322.  
Amari, S. (1999). Natural gradient learning for over- and under-complete bases in ica. Neural Comput., 11(8), 1875-1883.  
Atick, J. J. (1992). Could information theory provide an ecological theory of sensory processing? Network: Comp. Neural., 3(2), 213-251.  
Barlow, H. B. (1961). Possible principles underlying the transformation of sensory messages. *Sensory Communication*, (pp. 217-234).  
Bell, A. J. & Sejnowski, T. J. (1995). An information-maximization approach to blind separation and blind deconvolution. *Neural Comput.*, 7(6), 1129-1159.  
Bell, A. J. & Sejnowski, T. J. (1997). The "independent components" of natural scenes are edge filters. Vision Res., 37(23), 3327-3338.  
Bengio, Y. (2012). Deep learning of representations for unsupervised and transfer learning. *Unsupervised and Transfer Learning Challenges in Machine Learning*, 7, 19.  
Bengio, Y., Courville, A., & Vincent, P. (2013). Representation learning: A review and new perspectives. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 35(8), 1798-1828.  
Bengio, Y., Lamblin, P., Popovici, D., Larochelle, H., et al. (2007). Greedy layer-wise training of deep networks. Advances in neural information processing systems, 19, 153.

Borst, A. & Theunissen, F. E. (1999). Information theory and neural coding. Nature neuroscience, 2(11), 947-957.  
Brunel, N. & Nadal, J. P. (1998). Mutual information, Fisher information, and population coding. *Neural Comput.*, 10(7), 1731-1757.  
Carlo, C. N. & Stevens, C. F. (2013). Structural uniformity of neocortex, revisited. Proceedings of the National Academy of Sciences, 110(4), 1488-1493.  
Clarke, B. S. & Barron, A. R. (1990). Information-theoretic asymptotics of Bayes methods. IEEE Trans. Inform. Theory, 36(3), 453-471.  
Coates, A., Ng, A. Y., & Lee, H. (2011). An analysis of single-layer networks in unsupervised feature learning. In International conference on artificial intelligence and statistics (pp. 215-223).  
Cortes, C. & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297.  
Cover, T. M. & Thomas, J. A. (2006). Elements of Information, 2nd Edition. New York: Wiley-Interscience.  
Edelman, A., Arias, T. A., & Smith, S. T. (1998). The geometry of algorithms with orthogonality constraints. SIAM J. Matrix Anal. Appl., 20(2), 303-353.  
Elad, M. & Aharon, M. (2006). Image denoising via sparse and redundant representations over learned dictionaries. Image Processing, IEEE Transactions on, 15(12), 3736-3745.  
Erhan, D., Bengio, Y., Courville, A., Manzagol, P.-A., Vincent, P., & Bengio, S. (2010). Why does unsupervised pre-training help deep learning? The Journal of Machine Learning Research, 11, 625-660.  
Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. In International Conference on Artificial Intelligence and Statistics (pp. 315-323).  
Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. In Advances in Neural Information Processing Systems (pp. 2672–2680).  
Hinton, G. (2010). A practical guide to training restricted boltzmann machines. _Momentum_, 9(1), 926.  
Hinton, G., Osindero, S., & Teh, Y.-W. (2006). A fast learning algorithm for deep belief nets. Neural computation, 18(7), 1527-1554.  
Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural computation, 14(8), 1771-1800.  
Hinton, G. E. & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786), 504-507.  
Hoerl, A. E. & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1), 55-67.  
Huang, W. & Zhang, K. (2016). Information-theoretic bounds and approximations in neural population coding. Neural Comput, submitted, URL https://arxiv.org/abs/1611.01414.  
Hubel, D. H. & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. The Journal of physiology, 160(1), 106-154.  
Hyvarinen, A. (1999). Fast and robust fixed-point algorithms for independent component analysis. Neural Networks, IEEE Transactions on, 10(3), 626-634.  
Hyvarinen, A. (2005). Estimation of non-normalized statistical models using score matching. The Journal of Machine Learning Research, 6, 695-709.

Konstantinides, K. & Yao, K. (1988). Statistical analysis of effective singular values in matrix rank determination. Acoustics, Speech and Signal Processing, IEEE Transactions on, 36(5), 757-763.  
Kreutz-Delgado, K., Murray, J. F., Rao, B. D., Engan, K., Lee, T. S., & Sejnowski, T. J. (2003). Dictionary learning algorithms for sparse representation. Neural computation, 15(2), 349-396.  
Le, Q. V., Karpenko, A., Ngiam, J., & Ng, A. Y. (2011). Ica with reconstruction cost for efficient overcomplete feature learning. In Advances in Neural Information Processing Systems (pp. 1017-1025).  
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.  
Lee, H., Battle, A., Raina, R., & Ng, A. Y. (2006). Efficient sparse coding algorithms. In Advances in neural information processing systems (pp. 801-808).  
Lee, H., Ekanadham, C., & Ng, A. Y. (2008). Sparse deep belief net model for visual area v2. In Advances in neural information processing systems (pp. 873-880).  
Lee, T.-W., Girolami, M., Bell, A. J., & Sejnowski, T. J. (2000). A unifying information-theoretic framework for independent component analysis. Computers & Mathematics with Applications, 39(11), 1-21.  
Lewicki, M. S. & Olshausen, B. A. (1999). Probabilistic framework for the adaptation and comparison of image codes. JOSA A, 16(7), 1587-1601.  
Lewicki, M. S. & Sejnowski, T. J. (2000). Learning overcomplete representations. Neural computation, 12(2), 337-365.  
Linsker, R. (1988). Self-Organization in a perceptual network. Computer, 21(3), 105-117.  
MacKay, D. J. C. (2003). Information Theory, Inference and Learning Algorithms. Cambridge: Cambridge University Press.  
Mairal, J., Bach, F., Ponce, J., & Sapiro, G. (2009). Online dictionary learning for sparse coding. In Proceedings of the 26th annual international conference on machine learning (pp. 689-696). ACM.  
Mairal, J., Bach, F., Ponce, J., & Sapiro, G. (2010). Online learning for matrix factorization and sparse coding. The Journal of Machine Learning Research, 11, 19-60.  
Montufar, G. F., Pascanu, R., Cho, K., & Bengio, Y. (2014). On the number of linear regions of deep neural networks. In Advances in Neural Information Processing Systems (pp. 2924-2932).  
Nair, V. & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML-10) (pp. 807-814).  
Ng, A. (2011). Sparse autoencoder. CS294A Lecture notes, 72, 1-19.  
Olshausen, B. A. & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.  
Olshausen, B. A. & Field, D. J. (1997). Sparse coding with an overcomplete basis set: A strategy employed by v1? Vision Res., 37(23), 3311-3325.  
Pouget, A., Dayan, P., & Zemel, R. (2000). Information processing with population codes. Nature Reviews Neuroscience, 1(2), 125-132.  
Ranzato, M., Poultney, C., Chopra, S., & LeCun, Y. (2007). Efficient learning of sparse representations with an energy-based model. In Advances in neural information processing systems (pp. 1137-1144).  
Rao, C. R. (1945). Information and accuracy attainable in the estimation of statistical parameters. Bulletin of the Calcutta Mathematical Society, 37(3), 81-91.

Rockel, A. J., Hiorns, R. W., & Powell, T. P. (1980). The basic uniformity in structure of the neocortex. *Brain*, 103(2), 221-244.  
Shannon, C. (1948). A mathematical theory of communications. Bell System Technical Journal, 27, 379-423 and 623-656.  
Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958.  
Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., & Manzagol, P.-A. (2010). Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. The Journal of Machine Learning Research, 11, 3371-3408.  
Yarrow, S., Challis, E., & Series, P. (2012). Fisher and shannon information in finite neural populations. Neural computation, 24(7), 1740-1780.
