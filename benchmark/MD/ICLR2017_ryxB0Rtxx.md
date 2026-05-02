# IDENTITY MATTERS IN DEEP LEARNING

# MoritzHardt

Google Brain

1600 Amphitheatre Parkway,

Mountain View, CA, 94043

m@mrtz.org

# Tengyu Ma

Department of Computer Science

Princeton University

35 Olden Street, Princeton, 08540

tengyu@cs.princeton.edu

# ABSTRACT

An emerging design principle in deep learning is that each layer of a deep artificial neural network should be able to easily express the identity transformation. This idea not only motivated various normalization techniques, such as batch normalization, but was also key to the immense success of residual networks.

In this work, we put the principle of identity parameterization on a more solid theoretical footing alongside further empirical progress. We first give a strikingly simple proof that arbitrarily deep linear residual networks have no spurious local optima. The same result for feed-forward networks in their standard parameterization is substantially more delicate. Second, we show that residual networks with ReLu activations have universal finite-sample expressivity in the sense that the network can represent any function of its sample provided that the model has more parameters than the sample size.

Directly inspired by our theory, we experiment with a radically simple residual architecture consisting of only residual convolutional layers and ReLu activations, but no batch normalization, dropout, or max pool. Our model improves significantly on previous all-convolutional networks on the CIFAR10, CIFAR100, and ImageNet classification benchmarks.

# 1 INTRODUCTION

Traditional convolutional neural networks for image classification, such as AlexNet (Krizhevsky et al. (2012)), are parameterized in such a way that when all trainable weights are 0, a convolutional layer represents the 0-mapping. Moreover, the weights are initialized symmetrically around 0. This standard parameterization makes it non-trivial for a convolutional layer trained with stochastic gradient methods to preserve features that were already good. Put differently, such convolutional layers cannot easily converge to the identity transformation at training time.

This shortcoming was observed and partially addressed by Ioffe & Szegedy (2015) through batch normalization, i.e., layer-wise whitening of the input with a learned mean and covariance. But the idea remained somewhat implicit until residual networks (He et al. (2015); He et al. (2016)) explicitly introduced a reparameterization of the convolutional layers such that when all trainable weights are 0, the layer represents the identity function. Formally, for an input  $x$ , each residual layer has the form  $x + h(x)$ , rather than  $h(x)$ . This simple reparameterization allows for much deeper architectures largely avoiding the problem of vanishing (or exploding) gradients. Residual networks, and subsequent architectures that use the same parameterization, have since then consistently achieved state-of-the-art results on various computer vision benchmarks such as CIFAR10 and ImageNet.

# 1.1 OUR CONTRIBUTIONS

In this work, we consider identity parameterizations from a theoretical perspective, while translating some of our theoretical insight back into experiments. Loosely speaking, our first result underlines how identity parameterizations make optimization easier, while our second result shows the same is true for representation.

Linear residual networks. Since general non-linear neural networks, are beyond the reach of current theoretical methods in optimization, we consider the case of deep linear networks as a simplified model. A linear network represents an arbitrary linear map as a sequence of matrices  $A_{\ell} \cdots A_{2}A_{1}$ . The objective function is  $\mathbb{E}\|y - A_{\ell} \cdots A_{1}x\|^2$ , where  $y = Rx$  for some unknown linear transformation  $R$  and  $x$  is drawn from a distribution. Such linear networks have been studied actively in recent years as a stepping stone toward the general non-linear case (see Section 1.2). Even though  $A_{\ell} \cdots A_{1}$  is just a linear map, the optimization problem over the factored variables  $(A_{\ell}, \ldots, A_{1})$  is non-convex.

In analogy with residual networks, we will instead parameterize the objective function as

$$
\min  _ {A _ {1}, \dots , A _ {\ell}} \mathbb {E} \| y - (I + A _ {\ell}) \dots (I + A _ {1}) x \| ^ {2}. \tag {1.1}
$$

To give some intuition, when the depth  $\ell$  is large enough, we can hope that the target function  $R$  has a factored representation in which each matrix  $A_{i}$  has small norm. Any symmetric positive semidefinite matrix  $O$  can, for example, be written as a product  $O = O_{\ell} \cdots O_{1}$ , where each  $O_{i} = O^{1/\ell}$  is very close to the identity for large  $\ell$  so that  $A_{i} = O_{i} - I$  has small spectral norm. We first prove that an analogous claim is true for all linear transformations  $R$ . Specifically, we prove that for every linear transformation  $R$ , there exists a global optimizer  $(A_{1}, \ldots, A_{\ell})$  of (1.1) such that for large enough depth  $\ell$ ,

$$
\max  _ {1 \leq i \leq \ell} \| A _ {i} \| \leq O (1 / \ell). \tag {1.2}
$$

Here,  $\| A\|$  denotes the spectral norm of  $A$ . The constant factor depends on the conditioning of  $R$ . We give the formal statement in Theorem 2.1. The theorem has the interesting consequence that as the depth increases, smaller norm solutions exist and hence regularization may offset the increase in parameters.

Having established the existence of small norm solutions, our main result on linear residual networks shows that the objective function (1.1) is, in fact, easy to optimize when all matrices have sufficiently small norm. More formally, letting  $A = (A_{1},\ldots ,A_{\ell})$  and  $f(A)$  denote the objective function in (1.1), we can show that the gradients of vanish only when  $f(A) = 0$  provided that  $\max_i\| A_i\| \leq O(1 / \ell)$ . See Theorem 2.2. This result implies that linear residual networks have no critical points other than the global optimum. In contrast, for standard linear neural networks we only know, by work of Kawaguchi (2016) that these networks don't have local optima except the global optimum, but it doesn't rule out other critical points. In fact, setting  $A_{i} = 0$  will always lead to a bad critical point in the standard parameterization.

Universal finite sample expressivity. Going back to non-linear residual networks with ReLU activations, we can ask: How expressive are deep neural networks that are solely based on residual layers with ReLU activations? To answer this question, we give a very simple construction showing that such residual networks have perfect finite sample expressivity. In other words, a residual network with ReLU activations can easily express any functions of a sample of size  $n$ , provided that it has sufficiently more than  $n$  parameters. Note that this requirement is easily met in practice. On CIFAR 10 ( $n = 50000$ ), for example, successful residual networks often have more than  $10^6$  parameters. More formally, for a data set of size  $n$  with  $r$  classes, our construction requires  $O(n\log n + r^2)$  parameters. Theorem 3.2 gives the formal statement.

Each residual layer in our construction is of the form  $x + V\mathrm{ReLU}(Ux)$ , where  $U$  and  $V$  are linear transformations. These layers are significantly simpler than standard residual layers, which typically have two ReLU activations as well as two instances of batch normalization.

The power of all-convolutional residual networks. Directly inspired by the simplicity of our expressivity result, we experiment with a very similar architecture on the CIFAR10, CIFAR100, and ImageNet data sets. Our architecture is merely a chain of convolutional residual layers each with a single ReLU activation, but without batch normalization, dropout, or max pooling as are common in standard architectures. The last layer is a fixed random projection that is not trained. In line with our theory, the convolutional weights are initialized near 0, using Gaussian noise mainly as a symmetry breaker. The only regularizer is standard weight decay ( $\ell_2$ -regularization) and there is no need for dropout. Despite its simplicity, our architecture reaches  $6.38\%$  top-1 classification error on the CIFAR10 benchmark (with standard data augmentation). This is competitive with the best

residual network reported in He et al. (2015), which achieved  $6.43\%$ . Moreover, it improves upon the performance of the previous best all-convolutional network,  $7.25\%$ , achieved by Springenberg et al. (2014). Unlike ours, this previous all-convolutional architecture additionally required dropout and a non-standard preprocessing (ZCA) of the entire data set. Our architecture also improves significantly upon Springenberg et al. (2014) on both CIFar100 and ImageNet.

# 1.2 RELATED WORK

Since the advent of residual networks (He et al. (2015); He et al. (2016)), most state-of-the-art networks for image classification have adopted a residual parameterization of the convolutional layers. Further impressive improvements were reported by Huang et al. (2016) with a variant of residual networks, called dense nets. Rather than adding the original input to the output of a convolutional layer, these networks preserve the original features directly by concatenation. In doing so, dense nets are also able to easily encode an identity embedding in a higher-dimensional space. It would be interesting to see if our theoretical results also apply to this variant of residual networks.

There has been recent progress on understanding the optimization landscape of neural networks, though a comprehensive answer remains elusive. Experiments in Goodfellow et al. (2014) and Dauphin et al. (2014) suggest that the training objectives have a limited number of bad local minima with large function values. Work by Choromanska et al. (2015) draws an analogy between the optimization landscape of neural nets and that of the spin glass model in physics (Auffinger et al. (2013)). Soudry & Carmon (2016) showed that 2-layer neural networks have no bad differentiable local minima, but they didn't prove that a good differentiable local minimum does exist. Baldi & Hornik (1989) and Kawaguchi (2016) show that linear neural networks have no bad local minima. In contrast, we show that the optimization landscape of deep linear residual networks has no bad critical point, which is a stronger and more desirable property. Our proof is also notably simpler illustrating the power of re-parametrization for optimization. Our results also indicate that deeper networks may have more desirable optimization landscapes compared with shallower ones.

# 2 OPTIMIZATION LANDSCAPE OF LINEAR RESIDUAL NETWORKS

Consider the problem of learning a linear transformation  $R \colon \mathbb{R}^d \to \mathbb{R}^d$  from noisy measurements  $y = Rx + \xi$ , where  $\xi \in \mathcal{N}(0, \mathrm{Id}_d)$  is a  $d$ -dimensional spherical Gaussian vector. Denoting by  $\mathcal{D}$  the distribution of the input data  $x$ , let  $\Sigma = \mathbb{E}_{x \sim \mathcal{D}}[xx^\top]$  be its covariance matrix.

There are, of course, many ways to solve this classical problem, but our goal is to gain insights into the optimization landscape of neural nets, and in particular, residual networks. We therefore parameterize our learned model by a sequence of weight matrices  $A_{1},\ldots ,A_{\ell}\in \mathbb{R}^{d\times d}$

$$
h _ {0} = x, \quad h _ {j} = h _ {j - 1} + A _ {j} h _ {j - 1}, \quad \hat {y} = h _ {\ell}. \tag {2.1}
$$

Here  $h_1, \ldots, h_{\ell-1}$  are the  $\ell-1$  hidden layers and  $\hat{y} = h_\ell$  are the predictions of the learned model on input  $x$ . More succinctly, we have

$$
\hat {y} = \left(\operatorname {I d} _ {d} + A _ {\ell}\right) \dots \left(\operatorname {I d} + A _ {1}\right) x.
$$

It is easy to see that this model can express any linear transformation  $R$ . We will use  $A$  as a shorthand for all of the weight matrices, that is, the  $\ell \times d \times d$ -dimensional tensor the contains  $A_1, \ldots, A_\ell$  as slices. Our objective function is the maximum likelihood estimator,

$$
f (A, (x, y)) = \| \hat {y} - y \| ^ {2} = \| (\operatorname {I d} + A _ {\ell}) \dots (\operatorname {I d} + A _ {1}) x - R x - \xi \| ^ {2}. \tag {2.2}
$$

We will analyze the landscape of the population risk, defined as,

$$
f (A) := \mathbb {E} [ f (A, (x, y)) ].
$$

Recall that  $\| A_i \|$  is the spectral norm of  $A_i$ . We define the norm  $\| \cdot \|$  for the tensor  $A$  as the maximum of the spectral norms of its slices,

$$
\| A \| := \max  _ {1 \leq i \leq \ell} \| A _ {i} \|.
$$

The first theorem of this section states that the objective function  $f$  has an optimal solution with small  $\| \cdot \|$ -norm, which is inversely proportional to the number of layers  $\ell$ . Thus, when

the architecture is deep, we can shoot for fairly small norm solutions. We define  $\gamma := \max \{|\log \sigma_{\max}(R)|, |\log \sigma_{\min}(R)|\}$ . Here  $\sigma_{\min}(\cdot), \sigma_{\max}(\cdot)$  denote the least and largest singular values of  $R$  respectively.

Theorem 2.1. Suppose  $\ell \geq 3\gamma$ . Then, there exists a global optimum solution  $A^{\star}$  of the population risk  $f(\cdot)$  with norm

$$
\left\| A ^ {\star} \right\| \leq 2 (\sqrt {\pi} + \sqrt {3 \gamma}) ^ {2} / \ell .
$$

Here  $\gamma$  should be thought of as a constant since if  $R$  is too large (or too small), we can scale the data properly so that  $\sigma_{\mathrm{min}}(R) \leq 1 \leq \sigma_{\mathrm{max}}(R)$ . Concretely, if  $\sigma_{\mathrm{max}}(R) / \sigma_{\mathrm{min}}(R) = \kappa$ , then we can scaling for the outputs properly so that  $\sigma_{\mathrm{min}}(R) = 1 / \sqrt{\kappa}$  and  $\sigma_{\mathrm{max}}(R) = \sqrt{\kappa}$ . In this case, we have  $\gamma = \log \sqrt{\kappa}$ , which will remain a small constant for fairly large condition number  $\kappa$ . We also point out that we made no attempt to optimize the constant factors here in the analysis. The proof of Theorem 2.1 is rather involved and is deferred to Section A.

Given the observation of Theorem 2.1, we restrict our attention to analyzing the landscape of  $f(\cdot)$  in the set of  $A$  with  $\| \cdot \|$ -norm less than  $\tau$ ,

$$
\mathcal {B} _ {\tau} = \left\{A \in \mathbb {R} ^ {\ell \times d \times d}: \| A \| \leq \tau \right\}.
$$

Here using Theorem 2.1, the radius  $\tau$  should be thought of as on the order of  $1 / \ell$ . Our main theorem in this section claims that there is no bad critical point in the domain  $\mathcal{B}_{\tau}$  for any  $\tau < 1$ . Recall that a critical point has vanishing gradient.

Theorem 2.2. For any  $\tau < 1$ , we have that any critical point  $A$  of the objective function  $f(\cdot)$  inside the domain  $\mathcal{B}_{\tau}$  must also be a global minimum.

Theorem 2.2 suggests that it is sufficient for the optimizer to converge to critical points of the population risk, since all the critical points are also global minima.

Moreover, in addition to Theorem 2.2, we also have that any  $A$  inside the domain  $\mathcal{B}_{\tau}$  satisfies that

$$
\left\| \nabla f (A) \right\| _ {F} ^ {2} \geq 4 \ell (1 - \tau) ^ {\ell - 1} \sigma_ {\min } (\Sigma) ^ {2} \left(f (A) - C _ {\text {o p t}}\right). \tag {2.3}
$$

Here  $C_{\mathrm{opt}}$  is the global minimal value of  $f(\cdot)$  and  $\| \nabla f(A)\| _F$  denotes the euclidean norm<sup>1</sup> of the  $\ell \times d\times d$ -dimensional tensor  $\nabla f(A)$ . Note that  $\sigma_{\mathrm{min}}(\Sigma)$  denote the minimum singular value of  $\Sigma$ .

Equation (2.3) says that the gradient has fairly large norm compared to the error, which guarantees convergence of the gradient descent to a global minimum (Karimi et al. (2016)) if the iterates stay inside the domain  $\mathcal{B}_{\tau}$ , which is not guaranteed by Theorem 2.2 by itself.

Towards proving Theorem 2.2, we start off with a simple claim that simplifies the population risk. We also use  $\| \cdot \| _F$  to denote the Frobenius norm of a matrix.

Claim 2.3. In the setting of this section, we have,

$$
f (A) = \left\| \left(\left(\operatorname {I d} + A _ {\ell}\right) \dots \left(\operatorname {I d} + A _ {1}\right) - R\right) \Sigma^ {1 / 2} \right\| _ {F} ^ {2} + C. \tag {2.4}
$$

Here  $C$  is a constant that doesn't depend on  $A$ , and  $\Sigma^{1/2}$  denote the square root of  $\Sigma$ , that is, the unique symmetric matrix  $B$  that satisfies  $B^2 = \Sigma$ .

Proof of Claim 2.3. Let  $\operatorname{tr}(A)$  denotes the trace of the matrix  $A$ . Let  $E = (\operatorname{Id} + A_{\ell}) \ldots (\operatorname{Id} + A_{1}) - R$ . Recalling the definition of  $f(A)$  and using equation (2.2), we have

$$
\begin{array}{l} f (A) = \mathbb {E} \left[ \| E x - \xi \| ^ {2} \right] \quad (\text {b y}) \\ = \mathbb {E} \left[ \| E x \| ^ {2} + \| \xi \| ^ {2} - 2 \langle E x, \xi \rangle \right] \\ = \mathbb {E} \left[ \operatorname {t r} \left(E x x ^ {\top} E ^ {\top}\right) \right] + \mathbb {E} \left[ \| \xi \| ^ {2} \right] \quad (\text {s i n c e} \mathbb {E} [ \langle E x, \xi \rangle ] = \mathbb {E} [ \langle E x, \mathbb {E} [ \xi | x ] \rangle ] = 0) \\ = \operatorname {t r} \left(E \mathbb {E} \left[ x x ^ {\top} \right] E ^ {\top}\right) + C \quad (\text {w h e r e} C = \mathbb {E} [ x x ^ {\top} ]) \\ = \operatorname {t r} \left(E \Sigma E ^ {\top}\right) + C = \| E \Sigma^ {1 / 2} \| _ {F} ^ {2} + C. \quad (\text {s i n c e} \mathbb {E} [ x x ^ {\top} ] = \Sigma) \\ \end{array}
$$

□

Next we compute the gradients of the objective function  $f(\cdot)$  from straightforward matrix calculus. We defer the full proof to Section A.

Lemma 2.4. The gradients of  $f(\cdot)$  can be written as,

$$
\frac {\partial f}{\partial A _ {i}} = 2 \left(\mathrm {I d} + A _ {\ell} ^ {\top}\right) \dots \left(\mathrm {I d} + A _ {i + 1} ^ {\top}\right) E \Sigma \left(\mathrm {I d} + A _ {i - 1} ^ {\top}\right) \dots \left(\mathrm {I d} + A _ {1} ^ {\top}\right), \tag {2.5}
$$

where  $E = (\mathrm{Id} + A_{\ell})\ldots (\mathrm{Id} + A_{1}) - R$

Now we are ready to prove Theorem 2.2. The key observation is that each matric  $A_{j}$  has small norm and cannot cancel the identity matrix. Therefore, the gradients in equation (2.5) is a product of non-zero matrices, except for the error matrix  $E$ . Therefore, if the gradient vanishes, then the only possibility is that the matrix  $E$  vanishes, which in turns implies  $A$  is an optimal solution.

Proof of Theorem 2.2. Using Lemma 2.4, we have,

$$
\begin{array}{l} \left\| \frac {\partial f}{\partial A _ {i}} \right\| _ {F} = 2 \left\| \left(\mathrm {I d} + A _ {\ell} ^ {\top}\right) \dots \left(\mathrm {I d} + A _ {i + 1} ^ {\top}\right) E \Sigma \left(\mathrm {I d} + A _ {i - 1} ^ {\top}\right) \dots \left(\mathrm {I d} + A _ {1} ^ {\top}\right) \right\| _ {F} \quad (\text {b y}) \\ \geq 2 \prod_ {j \neq i} \sigma_ {\min } \left(\operatorname {I d} + A _ {i} ^ {\top}\right) \cdot \sigma_ {\min } (\Sigma) \| E \| _ {F} \quad \text {(b y C l a i m C . 2)} \\ \geq 2 (1 - \tau) ^ {\ell - 1} \sigma_ {\min } (\Sigma) \| E \|. \quad \text {(s i n c e} \sigma_ {\min } (\mathrm {I d} + A) \geq 1 - \| A \|) \\ \end{array}
$$

It follows that

$$
\begin{array}{l} \| \nabla f (A) \| _ {F} ^ {2} = \sum_ {i = 1} ^ {\ell} \left\| \frac {\partial f}{\partial A _ {i}} \right\| _ {F} ^ {2} \geq 4 \ell (1 - \tau) ^ {\ell - 1} \sigma_ {\min } (\Sigma) ^ {2} \| E \| ^ {2} \\ \geq 4 \ell (1 - \tau) ^ {\ell - 1} \sigma_ {\min } (\Sigma) ^ {2} (f (A) - C) \quad \text {(b y} E \text {a n d} \text {C l a i m} 2. 3) \\ \geq 4 \ell (1 - \tau) ^ {\ell - 1} \sigma_ {\min } (\Sigma) ^ {2} (f (A) - C _ {\text {o p t}}). \\ (s i n c e C _ {\mathrm {o p t}} = \min  _ {A} f (A) \geq C \text {b y C l a i m 2 . 3}) \\ \end{array}
$$

Therefore we complete the proof of equation (2.3). Finally, if  $A$  is a critical point, namely,  $\nabla f(A) = 0$ , then by equation (2.3) we have that  $f(A) = C_{\mathrm{opt}}$ . That is,  $A$  is a global minimum.

# 3 REPRESENTATIONAL POWER OF THE RESIDUAL NETWORKS

In this section we characterize the finite-sample expressivity of residual networks. We consider a residual layers with a single ReLU activation and no batch normalization. The basic residual building block is a function  $\mathcal{T}_{U,V,s}(\cdot):\mathbb{R}^k\to \mathbb{R}^k$  that is parameterized by two weight matrices  $U\in \mathbb{R}^{ \times k}$ ,  $V\in \mathbb{R}^{k\times k}$  and a bias vector  $s\in \mathbb{R}^k$ ,

$$
\mathcal {T} _ {U, V, s} (h) = V \operatorname {R e L u} (U h + s). \tag {3.1}
$$

A residual network is composed of a sequence of such residual blocks. In comparison with the full pre-activation architecture in He et al. (2016), we remove two batch normalization layers and one ReLU layer in each building block.

We assume the data has  $r$  labels, encoded as  $r$  standard basis vectors in  $\mathbb{R}^r$ , denoted by  $e_1, \ldots, e_r$ . We have  $n$  training examples  $(x^{(1)}, y^{(1)}), \ldots, (x^{(n)}, y^{(n)})$ , where  $x^{(i)} \in \mathbb{R}^d$  denotes the  $i$ -th data and  $y^{(i)} \in \{e_1, \ldots, e_r\}$  denotes the  $i$ -th label. Without loss of generality we assume the data are normalized so that  $x^{(i)} = 1$ . We also make the mild assumption that no two data points are very close to each other.

Assumption 3.1. We assume that for every  $1 \leq i < j \leq n$ , we have  $\|x^{(i)} - x^{(j)}\|^2 \geq \rho$  for some absolute constant  $\rho > 0$ .

Images, for example, can always be imperceptibly perturbed in pixel space so as to satisfy this assumption for a small but constant  $\rho$ .

Under this mild assumption, we prove that residual networks have the power to express any possible labeling of the data as long as the number of parameters is a logarithmic factor larger than  $n$ .

Theorem 3.2. Suppose the training examples satisfy Assumption 3.1. Then, there exists a residual network  $N$  (specified below) with  $O(n\log n + r^2)$  parameters that perfectly expresses the training data, i.e., for all  $i \in \{1, \ldots, n\}$ , the network  $N$  maps  $x^{(i)}$  to  $y^{(i)}$ .

It is common in practice that  $n > r^2$ , as is for example the case for the Imagenet data set where  $n > 10^6$  and  $r = 1000$ .

We construct the following residual net using the building blocks of the form  $\mathcal{T}_{U,V,s}$  as defined in equation (3.1). The network consists of  $\ell + 1$  hidden layers  $h_0, \ldots, h_\ell$ , and the output is denoted by  $\hat{y} \in \mathbb{R}^r$ . The first layer of weights matrices  $A_0$  maps the  $d$ -dimensional input to a  $k$ -dimensional hidden variable  $h_0$ . Then we apply  $\ell$  layers of building block  $\mathcal{T}$  with weight matrices  $A_j, B_j \in \mathbb{R}^{k \times k}$ . Finally, we apply another layer to map the hidden variable  $h_\ell$  to the label  $\hat{y}$  in  $\mathbb{R}^k$ . Mathematically, we have

$$
h _ {0} = A _ {0} x,
$$

$$
h _ {j} = h _ {j - 1} + \mathcal {T} _ {A _ {j}, B _ {j}, b _ {j}} (h _ {j - 1}), \quad \forall j \in \{1, \dots , \ell \}
$$

$$
\hat {y} = h _ {\ell} + \mathcal {T} _ {A _ {\ell + 1}, B _ {\ell + 1}, s _ {\ell + 1}} \left(h _ {\ell}\right).
$$

We note that here  $A_{\ell + 1} \in \mathbb{R}^{k \times r}$  and  $B_{\ell + 1} \in \mathbb{R}^{r \times r}$  so that the dimension is compatible. We assume the number of labels  $r$  and the input dimension  $d$  are both smaller than  $n$ , which is safely true in practical applications. The hyperparameter  $k$  will be chosen to be  $O(\log n)$  and the number of layers is chosen to be  $\ell = \lceil n / k \rceil$ . Thus, the first layer has  $dk$  parameters, and each of the middle  $\ell$  building blocks contains  $2k^2$  parameters and the final building block has  $kr + r^2$  parameters. Hence, the total number of parameters is  $O(kd + \ell k^2 + rk + r^2) = O(n \log n + r^2)$ .

Towards constructing the network  $N$  of the form above that fits the data, we first take a random matrix  $A_0 \in \mathbb{R}^{k \times d}$  that maps all the data points  $x^{(i)}$  to vectors  $h_0^{(i)} \coloneqq A_0 x^{(i)}$ . Here we will use  $h_j^{(i)}$  to denote the  $j$ -th layer of hidden variable of the  $i$ -th example. By Johnson-Lindenstrauss Theorem (Johnson & Lindenstrauss (1984), or see Wikipedia (2016)), with good probability, the resulting vectors  $h_0^{(i)}$ 's remain to satisfy Assumption 3.1 (with slightly different scaling and larger constant  $\rho$ ), that is, any two vectors  $h_0^{(i)}$  and  $h_0^{(j)}$  are not very correlated.

Then we construct  $\ell$  middle layers that maps  $h_0^{(i)}$  to  $h_\ell^{(i)}$  for every  $i\in \{1,\ldots ,n\}$ . These vectors  $h_\ell^{(i)}$  will be clustered into  $r$  groups according to the labels, though they are in the  $\mathbb{R}^k$  instead of in  $\mathbb{R}^r$  as desired. Concretely, we design this cluster centers by picking  $r$  random unit vectors  $q_{1},\dots,q_{r}$  in  $\mathbb{R}^k$ . We view them as the surrogate label vectors in dimension  $k$  (note that  $k$  is potentially much smaller than  $r$ ). In high dimensions (technically, if  $k > 4\log r$ ) random unit vectors  $q_{1},\dots,q_{r}$  are pair-wise uncorrelated with inner product less than  $< 0.5$ . We associate the  $i$ -th example with the target surrogate label vector  $v^{(i)}$  defined as follows,

$$
\text {i f} y ^ {(i)} = e _ {j}, \text {t h e n} v ^ {(i)} = q _ {j}. \tag {3.2}
$$

Then we will construct the matrices  $(A_{1},B_{1}),\ldots ,(A_{\ell},B_{\ell})$  such that the first  $\ell$  layers of the network maps vector  $h_0^{(i)}$  to the surrogate label vector  $v^{(i)}$ . Mathematically, we will construct  $(A_{1},B_{1}),\ldots ,(A_{\ell},B_{\ell})$  such that

$$
\forall i \in \{1, \dots , n \}, h _ {\ell} ^ {(i)} = v ^ {(i)}. \tag {3.3}
$$

Finally we will construct the last layer  $\mathcal{T}_{A_{\ell +1},B_{\ell +1},b_{\ell +1}}$  so that it maps the vectors  $q_{1},\ldots ,q_{r}\in \mathbb{R}^{k}$  to  $e_1,\dots ,e_r\in \mathbb{R}^r$

$$
\forall j \in \{1, \dots , r \}, q _ {j} + \mathcal {T} _ {A _ {\ell + 1}, B _ {\ell + 1}, b _ {\ell + 1}} (q _ {j}) = e _ {j}. \tag {3.4}
$$

Putting these together, we have that by the definition (3.2) and equation (3.3), for every  $i$ , if the label is  $y^{(i)}$  is  $e_j$ , then  $h_\ell^{(i)}$  will be  $q_j$ . Then by equation (3.4), we have that  $\hat{y}^{(i)} = q_j + \mathcal{T}_{A_{\ell+1},B_{\ell+1},b_{\ell+1}}(q_j) = e_j$ . Hence we obtain that  $\hat{y}^{(i)} = y^{(i)}$ .

The key part of this plan is the construction of the middle  $\ell$  layers of weight matrices so that  $h_{\ell}^{(i)} = v^{(i)}$ . We encapsulate this into the following informal lemma. The formal statement and the full proof is deferred to Section B.

Lemma 3.3 (Informal version of Lemma B.2). In the setting above, for (almost) arbitrary vectors  $h_0^{(1)}, \ldots, h_0^{(n)}$  and  $v^{(1)}, \ldots, v^{(n)} \in \{q_1, \ldots, q_r\}$ , there exists weights matrices  $(A_1, B_1), \ldots, (A_\ell, B_\ell)$ , such that,

$$
\forall i \in \{1, \ldots , n \}, h _ {\ell} ^ {(i)} = v ^ {(i)}.
$$

We briefly sketch the proof of the Lemma to provide intuitions, and defer the full proof to Section B. The operation that each residual block applies to the hidden variable can be abstractly written as,

$$
\hat {h} \rightarrow h + \mathcal {T} _ {U, V, s} (h). \tag {3.5}
$$

where  $h$  corresponds to the hidden variable before the block and  $\hat{h}$  corresponds to that after. We claim that for an (almost) arbitrary sequence of vectors  $h^{(1)},\ldots ,h^{(n)}$ , there exist  $\mathcal{T}_{U,V,s}(\cdot)$  such that operation (3.5) transforms  $k$  vectors of  $h^{(i)}$ 's to an arbitrary set of other  $k$  vectors that we can freely choose, and maintain the value of the rest of  $n - k$  vectors. Concretely, for any subset  $S$  of size  $k$  and any desired vector  $v^{(i)}(i\in S)$ , there exist  $U,V,s$  such that

$$
v ^ {(i)} = h ^ {(i)} + \mathcal {T} _ {U, V, s} (h ^ {(i)}) \forall i \in S
$$

$$
h ^ {(i)} = h ^ {(i)} + \mathcal {T} _ {U, V, s} \left(h ^ {(i)}\right) \forall i \notin S \tag {3.6}
$$

This claim is formalized in Lemma B.1. We can use it repeatedly to construct  $\ell$  layers of building blocks, each of which transforms a subset of  $k$  vectors in  $\{h_0^{(1)},\ldots ,h_0^{(n)}\}$  to the corresponding vectors in  $\{v^{(1)},\dots ,v^{(n)}\}$ , and maintains the values of the others. Recall that we have  $\ell = \lceil n / k\rceil$  layers and therefore after  $\ell$  layers, all the vectors  $h_0^{(i)}$ 's are transformed to  $v^{(i)}$ 's, which complete the proof sketch.

# 4 POWER OF ALL-CONVOLUTIONAL RESIDUAL NETWORKS

Inspired by our theory, we experimented with all-convolutional residual networks on standard image classification benchmarks.

# 4.1 CIFAR10 AND CIFAR100

Our architectures for CIFAR10 and CIFAR100 are identical except for the final dimension corresponding to the number of classes 10 and 100, respectively. In Table 1, we outline our architecture. Each residual block has the form  $x + C_2(\mathrm{ReLU}(C_1x))$ , where  $C_1, C_2$  are convolutions of the specified dimension (kernel width, kernel height, number of input channels, number of output channels). The second convolution in each block always has stride 1, while the first may have stride 2 where indicated. In cases where transformation is not dimensionality-preserving, the original input  $x$  is adjusted using averaging pooling and padding as is standard in residual layers.

We trained our models with the Tensorflow framework, using a momentum optimizer with momentum 0.9, and batch size is 128. All convolutional weights are trained with weight decay 0.0001. The initial learning rate is 0.05, which drops by a factor 10 and 30000 and 50000 steps. The model reaches peak performance at around  $50k$  steps, which takes about  $24h$  on a single NVIDIA Tesla K40 GPU. Our code can be easily derived from an open source implementation<sup>3</sup> by removing batch normalization, adjusting the residual components and model architecture. An important departure from the code is that we initialize a residual convolutional layer of kernel size  $k \times k$  and  $c$  output channels using a random normal initializer of standard deviation  $\sigma = 1/k^2c$ , rather than  $1/k\sqrt{c}$  used for standard convolutional layers. This substantially smaller weight initialization helped training, while not affecting representation.

A notable difference from standard models is that the last layer is not trained, but simply a fixed random projection. On the one hand, this slightly improved test error (perhaps due to a regularizing effect). On the other hand, it means that the only trainable weights in our model are those of the convolutions, making our architecture "all-convolutional".

Table 1: Architecture for CIFAR10/100 (55 convolutions, 13.5M parameters)  

<table><tr><td>variable dimensions</td><td>initial stride</td><td>description</td></tr><tr><td>3 × 3 × 3 × 16</td><td>1</td><td>1 standard conv</td></tr><tr><td>3 × 3 × 16 × 64</td><td>1</td><td>9 residual blocks</td></tr><tr><td>3 × 3 × 64 × 128</td><td>2</td><td>9 residual blocks</td></tr><tr><td>3 × 3 × 128 × 256</td><td>2</td><td>9 residual blocks</td></tr><tr><td>-</td><td>-</td><td>8 × 8 global average pool</td></tr><tr><td>256 × num_classeses</td><td>-</td><td>random projection (not trained)</td></tr></table>

![](images/ee13dffd78735dfec85582c052ddd9fb058f2f439af3abec49e7ef2b53280043.jpg)  
Figure 1: Convergence plots of best model for CIFAR10 (left) and CIFAR (100) right. One step is a gradient update with batch size 128.

![](images/e7c472ed64633a47a691911ad6f348d862a75913e212f640ee8119141932767b.jpg)

An interesting aspect of our model is that despite its massive size of 13.59 million trainable parameters, the model does not seem to overfit too quickly even though the data set size is 50000. In contrast, we found it difficult to train a model with batch normalization of this size without significant overfitting on CIFAR10.

Table 2 summarizes the top-1 classification error of our models compared with a non-exhaustive list of previous works, restricted to the best previous all-convolitional result by Springenberg et al. (2014), the first residual results He et al. (2015), and state-of-the-art results on CIFAR by Huang et al. (2016). All results are with standard data augmentation.

Table 2: Comparison of top-1 classification error on different benchmarks  

<table><tr><td>Method</td><td>CIFAR10</td><td>CIFAR100</td><td>ImageNet</td><td>remarks</td></tr><tr><td>All-CNN</td><td>7.25</td><td>32.39</td><td>41.2</td><td>all-convolutional, dropout, extra data processing</td></tr><tr><td>Ours</td><td>6.38</td><td>24.64</td><td>35.29</td><td>all-convolutional</td></tr><tr><td>ResNet</td><td>6.43</td><td>25.16</td><td>19.38</td><td></td></tr><tr><td>DenseNet</td><td>3.74</td><td>19.25</td><td>N/A</td><td></td></tr></table>

# 4.2 IMAGENET

The ImageNet ILSVRC 2012 data set has 1, 281, 167 data points with 1000 classes. Each image is resized to  $224 \times 224$  pixels with 3 channels. We experimented with an all-convolutional variant of the 34-layer network in He et al. (2015). The original model achieved  $25.03\%$  classification error. Our derived model has  $35.7M$  trainable parameters. We trained the model with a momentum optimizer (with momentum 0.9) and a learning rate schedule that decays by a factor of 0.94 every two epochs, starting from the initial learning rate 0.1. Training was distributed across 6 machines

updating asynchronously. Each machine was equipped with 8 GPUs (NVIDIA Tesla K40) and used batch size 256 split across the 8 GPUs so that each GPU updated with batches of size 32.

In contrast to the situation with CIFAR10 and CIFAR100, on ImageNet our all-convolutional model performed significantly worse than its original counterpart. Specifically, we experienced a significant amount of underfitting suggesting that a larger model would likely perform better.

Despite this issue, our model still reached  $35.29\%$  top-1 classification error on the test set (50000 data points), and  $14.17\%$  top-5 test error after 700,000 steps (about one week of training). While no longer state-of-the-art, this performance is significantly better than the  $40.7\%$  reported by Krizhevsky et al. (2012), as well as the best all-convolutional architecture by Springenberg et al. (2014). We believe it is quite likely that a better learning rate schedule and hyperparameter settings of our model could substantially improve on the preliminary performance reported here.

# 5 CONCLUSION

Our theory underlines the importance of identity parameterizations when training deep artificial neural networks. An outstanding open problem is to extend our optimization result to the non-linear case where each residual has a single ReLU activation as in our expressivity result. We conjecture that a result analogous to Theorem 2.2 is true for the general non-linear case. Unlike with the standard parameterization, we see no fundamental obstacle for such a result.

We hope our theory and experiments together help simplify the state of deep learning by aiming to explain its success with a few fundamental principles, rather than a multitude of tricks that need to be delicately combined. We believe that much of the advances in image recognition can be achieved with residual convolutional layers and ReLU activations alone. This could lead to extremely simple (albeit deep) architectures that match the state-of-the-art on all image classification benchmarks.

# REFERENCES

Antonio Auffinger, Gérard Ben Arous, and Jérôme Cerny. Random matrices and complexity of spin glasses. Communications on Pure and Applied Mathematics, 66(2):165-201, 2013.  
P. Baldi and K. Hornik. Neural networks and principal component analysis: Learning from examples without local minima. *Neural Netw.*, 2(1):53-58, January 1989. ISSN 0893-6080. doi: 10.1016/0893-6080(89)90014-2. URL http://dx.doi.org/10.1016/0893-6080(89)90014-2.  
Anna Choromanska, Mikael Henaff, Michael Mathieu, Gérard Ben Arous, and Yann LeCun. The loss surfaces of multilayer networks. In AISTATS, 2015.  
Yann N Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. In Advances in neural information processing systems, pp. 2933–2941, 2014.  
I. J. Goodfellow, O. Vinyals, and A. M. Saxe. Qualitatively characterizing neural network optimization problems. *ArXiv e-prints*, December 2014.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In arXiv prepring arXiv:1506.01497, 2015.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In Computer Vision - ECCV 2016 - 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part IV, pp. 630-645, 2016. doi: 10.1007/978-3-319-46493-0_38. URL http://dx.doi.org/10.1007/978-3-319-46493-0_38.  
Gao Huang, Zhuang Liu, and Kilian Q. Weinberger. Densely connected convolutional networks. CoRR, abs/1608.06993, 2016. URL http://arxiv.org/abs/1608.06993.

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, pp. 448-456, 2015. URL http://jmlr.org/proceedings/papers/v37/ioffe15.html.  
William B Johnson and Joram Lindenstrauss. Extensions of lipschitz mappings into a hilbert space. Contemporary mathematics, 26(189-206):1, 1984.  
H. Karimi, J. Nutini, and M. Schmidt. Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak\{\}ojasiewicz Condition. ArXiv e-prints, August 2016.  
K. Kawaguchi. Deep Learning without Poor Local Minima. ArXiv e-prints, May 2016.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
D. Soudry and Y. Carmon. No bad local minima: Data independent training error guarantees for multilayer neural networks. *ArXiv e-prints*, May 2016.  
J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. Striving for Simplicity: The All Convolutional Net. ArXiv e-prints, December 2014.  
Eric W. Weisstein. Normal matrix, from mathworld-a wolfram web resource., 2016. URL http://mathworld.wolfram.com/NormalMatrix.html.  
Wikipedia. Johnsonlindenstrauss lemma — wikipedia, the free encyclopedia, 2016. URL https://en.wikipedia.org/w/index.php?title=Johnson%E2%80%93Lindenstrauss_lemmale&oldid=743553642.
