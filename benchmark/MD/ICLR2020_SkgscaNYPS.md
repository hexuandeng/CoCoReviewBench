# THE ASYMPTOTIC SPECTRUM OF THE HESSIAN OF DNN THROUGHOUT TRAINING

Anonymous authors

Paper under double-blind review

# ABSTRACT

The dynamics of DNNs during gradient descent is described by the so-called Neural Tangent Kernel (NTK). In this article, we show that the NTK allows one to gain precise insight into the Hessian of the cost of DNNs: we obtain a full characterization of the asymptotics of the spectrum of the Hessian, at initialization and during training.

# 1. INTRODUCTION

The advent of deep learning has sparked a lot of interest in the loss surface of deep neural networks (DNN), and in particular its Hessian. However to our knowledge, there is still no theoretical description of the spectrum of the Hessian. Nevertheless a number of phenomena have been observed numerically.

The loss surface of neural networks has been compared to the energy landscape of different physical models Choromanska et al. (2015); Geiger et al. (2018); Mei et al. (2018). It appears that the loss surface of DNNs may change significantly depending on the width of the network (the number of neurons in the hidden layer), motivating the distinction between the under- and over-parametrized regimes Baity-Jesi et al. (2018); Geiger et al. (2018; 2019).

The non-convexity of the loss function implies the existence of a very large number of saddle points, which could slow down training. In particular, in Pascanu et al. (2014); Dauphin et al. (2014), a relation between the rank of saddle points (the number of negative eigenvalues of the Hessian) and their loss has been observed.

For overparametrized DNNs, a possibly more important phenomenon is the large number of flat directions Baity-Jesi et al. (2018). The existence of these flat minima is conjectured to be related to the generalization of DNNs and may depend on the training procedure Hochreiter & Schmidhuber (1997); Chaudhari et al. (2016); Wu et al. (2017).

Recent results have obtained strong convergence guarantees for shallow networks Rotskoff & Vanden-Eijnden (2018); Chizat & Bach (2018a); Mei et al. (2018) and also for deep networks in the over-parametrized regime Jacot et al. (2018); Du et al. (2019); Allen-Zhu et al. (2018).

In Jacot et al. (2018) it has been shown, using a functional approach, that in the infinite width-limit, DNNs behave like kernel methods with respect to the so-called Neural Tangent Kernel, which is determined by the architecture of the network. This strengthens the connections between neural networks and kernel methods Neal (1996); Cho & Saul (2009); Lee et al. (2018).

This raises the question: can we use these new results to gain insight into the behavior of the Hessian of the loss of DNNs, at least in the small region explored by the parameters during training?

# 1.1. CONTRIBUTIONS

Following ideas introduced in Jacot et al. (2018), we consider the training of  $L + 1$ -layered DNNs in a functional setting. For a functional cost  $\mathcal{C}$ , the Hessian of the loss  $\mathbb{R}^P \ni \theta \mapsto \mathcal{C}(F^{(L)}(\theta))$  is the sum of two  $P \times P$  matrices  $I$  and  $S$ . We show the following results for large  $P$  and for a fixed number of datapoints  $N$ :

- The first matrix  $I$  is positive semi-definite and its eigenvalues are given by the (weighted) kernel PCA of the dataset with respect to the NTK. The dominating eigenvalues are the principal components of the data followed by a high number of small eigenvalues. The "flat directions" are spanned by the small eigenvalues and the null-space (of dimension at least  $P - N$  when there is a single output). Because the NTK is asymptotically constant Jacot et al. (2018), these results apply at initialization, during training and at convergence.  
- The second matrix  $S$  can be viewed as residual contribution to  $H$ , since it vanishes as the network converges to a global minimum. We compute the limit of the first moment  $\operatorname{Tr}(S)$  and characterize its evolution during training, of the second moment  $\operatorname{Tr}\left(S^2\right)$  which stays constant during training, and show that the higher moments vanish.  
- Regarding the sum  $H = I + S$ , we show that the matrices  $I$  and  $S$  are asymptotically orthogonal to each other at initialization and during training. In particular, the moments of the matrices  $I$  and  $S$  add up:  $tr(H^k) \approx tr(I^k) + tr(S^k)$ .

These results give, for any depth and a fairly general non-linearity, a complete description of the spectrum of the Hessian in terms of the NTK at initialization and throughout training. This gives theoretical confirmation of a number of observations about the Hessian Hochreiter & Schmidhuber (1997); Pascanu et al. (2014); Dauphin et al. (2014); Chaudhari et al. (2016); Wu et al. (2017); Pennington & Bahri (2017); Geiger et al. (2018), and sheds a new light on them.

# 1.2. RELATED WORKS

The Hessian of the loss has been studied through the decomposition  $I + S$  in a number of previous works (Sagun et al. (2017); Pennington & Bahri (2017); Geiger et al. (2018)).

For least-squares and cross-entropy costs, the first matrix  $I$  is equal to the Fisher matrix Wagenaar (1998); Pascanu & Bengio (2013), whose moments have been described for shallow networks in Pennington & Worah (2018). For deep networks, the first two moments and the operator norm of the Fisher matrix for a least squares loss were computed at initialization in Karakida et al. (2018) conditionally on a certain independence assumption; our method does not require such assumptions. Note that their approach implicitly uses the NTK.

The second matrix  $S$  has been studied in Pennington & Bahri (2017); Geiger et al. (2018) for shallow networks, conditionally on a number of assumptions. Note that in the setting of Pennington & Bahri (2017), the matrices  $I$  and  $S$  are assumed to be freely independent, which allows them to study the spectrum of the Hessian; in our setting, we show that the two matrices  $I$  and  $S$  are asymptotically orthogonal to each other.

# 2. SETUP

We consider deep fully connected artificial neural networks (DNNs) using the setup and NTK parametrization of Jacot et al. (2018), taking an arbitrary nonlinearity  $\sigma \in C_b^4 (\mathbb{R})$  (i.e.  $\sigma :\mathbb{R}\to \mathbb{R}$  that is 4 times continuously differentiable function with all four derivatives bounded). The layers are numbered from 0 (input) to  $L$  (output), each containing  $n_{\ell}$  neurons for  $\ell = 0,\ldots ,L$ . The  $P = \sum_{\ell = 0}^{L - 1}(n_{\ell} + 1)n_{\ell +1}$  parameters consist of the weight matrices  $W^{(\ell)}\in \mathbb{R}^{n_{\ell +1}\times n_{\ell}}$  and bias vectors  $b^{(\ell)}\in \mathbb{R}^{n_{\ell +1}}$  for  $\ell = 0,\dots ,L - 1$ . We aggregate the parameters into the vector  $\theta \in \mathbb{R}^P$ .

The activations and pre-activations of the layers are defined recursively for an input  $x \in \mathbb{R}^{n_0}$ , setting  $\alpha^{(0)}(x; \theta) = x$ :

$$
\begin{array}{l} \tilde {\alpha} ^ {(\ell + 1)} (x; \theta) = \frac {1}{\sqrt {n _ {\ell}}} W ^ {(\ell)} \alpha^ {(\ell)} (x; \theta) + \beta b ^ {(\ell)}, \\ \alpha^ {(\ell + 1)} (x; \theta) = \sigma \big (\tilde {\alpha} ^ {(\ell + 1)} (x; \theta) \big). \\ \end{array}
$$

The parameter  $\beta$  is added to tune the influence of the bias on training<sup>1</sup>. All parameters are initialized as iid  $\mathcal{N}(0,1)$  Gaussians.

We will in particular study the network function, which maps inputs  $x$  to the activation of the output layer (before the last non-linearity):

$$
f _ {\theta} (x) = \tilde {\alpha} ^ {(L)} (x; \theta).
$$

In this paper, we will study the limit of various objects as  $n_1, \ldots, n_L \to \infty$  sequentially, i.e. we first take  $n_1 \to \infty$ , then  $n_2 \to \infty$ , etc. This greatly simplifies the proofs, but they could in principle be extended to the simultaneous limit, i.e. when  $n_1 = \ldots = n_{L-1} \to \infty$ . All our numerical experiments are done with 'rectangular' networks (with  $n_1 = \ldots = n_{L-1}$ ) and match closely the predictions for the sequential limit.

In the limit we study in this paper, the NTK is asymptotically fixed, as in Jacot et al. (2018); Allen-Zhu et al. (2018); Du et al. (2019); Arora et al. (2019). By rescaling the outputs of DNNs as the width increases, one can reach another limit where the NTK is not fixed Chizat & Bach (2018a;b); Rotskoff & Vanden-Eijnden (2018); Mei et al. (2019). The behavior of the Hessian in this other limit may be significantly different.

# 2.1. FUNCTIONAL VIEWPOINT

The network function lives in a function space  $f_{\theta} \in \mathcal{F} \coloneqq [\mathbb{R}^{n_0} \to \mathbb{R}^{n_L}]$  and we call the function  $F^{(L)}: \mathbb{R}^P \to \mathcal{F}$  that maps the parameters  $\theta$  to the network function  $f_{\theta}$  the realization function. We study the differential behavior of  $F^{(L)}$ :

- The derivative  $\mathcal{DF}^{(L)}\in \mathbb{R}^P\otimes \mathcal{F}$  is a function-valued vector of dimension  $P$ . The  $p$ -th entry  $\mathcal{D}_pF^{(L)} = \partial_{\theta_p}f_\theta \in \mathcal{F}$  represents how modifying the parameter  $\theta_p$  modifies the function  $f_{\theta}$  in the space  $\mathcal{F}$ .  
- The Hessian  $\mathcal{H}F^{(L)}\in \mathbb{R}^P\otimes \mathbb{R}^P\otimes \mathcal{F}$  is a function-valued  $P\times P$  matrix.

The network is trained with respect to the cost functional:

$$
\mathcal {C} (f) = \frac {1}{N} \sum_ {i = 1} ^ {N} c _ {i} \left(f (x _ {i})\right),
$$

for strictly convex  $c_{i}$ , summing over a finite dataset  $x_{1},\ldots ,x_{N}\in \mathbb{R}^{n_{0}}$  of size  $N$ . The parameters are then trained with gradient descent on the composition  $\mathcal{C}\circ F^{(L)}$ , which defines the usual loss surface of neural networks.

In this setting, we define the finite realization function  $Y^{(L)}: \mathbb{R}^P \to \mathbb{R}^{Nn_L}$  mapping parameters  $\theta$  to be the restriction of the network function  $f_{\theta}$  to the training set  $y_{ik} = f_{\theta,k}(x_i)$ . The Jacobian  $\mathcal{D}Y^{(L)}$  is hence an  $Nn_L \times P$  matrix and its Hessian  $\mathcal{H}Y^{(L)}$  is a  $P \times P \times Nn_L$  tensor. Defining the restricted cost  $C(y) = \frac{1}{N}\sum_{i}c_{i}(y_{i})$ , we have  $C \circ F^{(L)} = C \circ Y^{(L)}$ .

For our analysis, we require that the gradient norm  $\| \mathcal{D}C\|$  does not explode during training. The following condition is sufficient:

Definition 1. A loss  $C:\mathbb{R}^{Nn_L}\to \mathbb{R}$  has bounded gradients over sublevel sets (BGOSS) if the norm of the gradient is bounded over all sets  $U_{a} = \left\{Y\in \mathbb{R}^{Nn_{L}}:C(Y)\leq a\right\}$ .

For example, the Mean Square Error (MSE)  $C(Y) = \frac{1}{2N} \| Y^* - Y \|^2$  for the labels  $Y^* \in \mathbb{R}^{Nn_L}$  has BGOSS because  $\| \nabla C(Y) \|^2 = \frac{1}{N} \| Y^* - Y \|^2 = 2C(Y)$ . For the binary and softmax cross-entropy the gradient is uniformly bounded, see Proposition 2 in Appendix A.

# 2.2. NEURAL TANGENT KERNEL

The behavior during training of the network function  $f_{\theta}$  in the function space  $\mathcal{F}$  is described by a (multi-dimensional) kernel, the Neural Tangent Kernel (NTK)

$$
\Theta_ {k, k ^ {\prime}} ^ {(L)} (x, x ^ {\prime}) = \sum_ {p = 1} ^ {P} \partial_ {\theta_ {p}} f _ {\theta , k} (x) \partial_ {\theta_ {p}} f _ {\theta , k ^ {\prime}} (x ^ {\prime}).
$$

During training, the function  $f_{\theta}$  follows the so-called kernel gradient descent with respect to the NTK, which is defined as

$$
\partial_ {t} f _ {\theta (t)} (x) = - \nabla_ {\Theta^ {(L)}} C _ {| f _ {\theta (t)}} (x) := - \frac {1}{N} \sum_ {i = 1} ^ {N} \Theta^ {(L)} (x, x _ {i}) \nabla c _ {i} (f _ {\theta (t)} (x _ {i})).
$$

In the infinite-width limit (letting  $n_1 \to \infty, \ldots, n_{L-1} \to \infty$  sequentially) and for losses with BGOSs, the NTK converges to a deterministic limit  $\Theta^{(L)} \to \Theta_{\infty}^{(L)} \otimes Id_{n_L}$ , which is constant during training, uniformly on finite time intervals  $[0, T]$  Jacot et al. (2018). In the case of the MSE loss, the uniform convergence of the NTK was proven for  $T = \infty$  in Arora et al. (2019).

The limiting NTK  $\Theta_{\infty}^{(L)}:\mathbb{R}^{n_0}\times \mathbb{R}^{n_0}\to \mathbb{R}$  is constructed as follows:

(1) For  $f,g:\mathbb{R}\to \mathbb{R}$  and a kernel  $K:\mathbb{R}^{n_0}\times \mathbb{R}^{n_0}\rightarrow \mathbb{R}$ , define the kernel  $\mathbb{L}_K^{f,g}$ :  $\mathbb{R}^{n_0}\times \mathbb{R}^{n_0}\to \mathbb{R}$  by

$$
\mathbb {L} _ {K} ^ {f, g} (x _ {0}, x _ {1}) = \mathbb {E} _ {(a _ {0}, a _ {1})} \left[ f (a _ {0}) g (a _ {1}) \right],
$$

for  $(a_0, a_1)$  a centered Gaussian vector with covariance matrix  $(K(x_i, x_j))_{i,j=0,1}$ . For  $f = g$ , we denote by  $\mathbb{L}_K^f$  the kernel  $\mathbb{L}_K^{f,f}$ .

(2) We define the kernels  $\Sigma_{\infty}^{(\ell)}$  for each layer of the network, starting with  $\Sigma_{\infty}^{(1)}(x_0,x_1) = 1 / n_0(x_0^T x_1) + \beta^2$  and then recursively by  $\Sigma_{\infty}^{(\ell +1)} = \mathbb{L}_{\Sigma_{\infty}^{(\ell)}}^{\sigma} + \beta^2$ , for  $\ell = 1,\ldots ,L - 1$ , where  $\sigma$  is the network non-linearity.  
(3) The limiting NTK  $\Theta_{\infty}^{(L)}$  is defined in terms of the kernels  $\Sigma_{\infty}^{(\ell)}$  and the kernels  $\dot{\Sigma}_{\infty}^{(\ell)} = \mathbb{L}_{\Sigma_{\infty}^{(\ell -1)}}^{\dot{\sigma}}$ :

$$
\Theta_ {\infty} ^ {(L)} = \sum_ {\ell = 1} ^ {L} \Sigma_ {\infty} ^ {(\ell)} \dot {\Sigma} _ {\infty} ^ {(\ell + 1)} \dots \dot {\Sigma} _ {\infty} ^ {(L)}.
$$

The NTK leads to convergence guarantees for DNNs in the infinite-width limit, and connect their generalization properties to those of kernel methods Jacot et al. (2018); Arora et al. (2019).

# 2.3. GRAM MATRICES

For a finite dataset  $x_{1},\ldots ,x_{N}\in \mathbb{R}^{n_{0}}$  and a fixed depth  $L\geq 1$ , we denote by  $\tilde{\Theta}\in \mathbb{R}^{Nn_L\times Nn_L}$  the Gram matrix of  $x_{1},\ldots ,x_{N}$  with respect to the limiting NTK, defined by

$$
\tilde {\Theta} _ {i k, j m} = \Theta_ {\infty} ^ {(L)} (x _ {i}, x _ {i ^ {\prime}}) \delta_ {k m}.
$$

It is block diagonal because different outputs  $k \neq m$  are asymptotically uncorrelated.

Similarly, for any (scalar) kernel  $\mathcal{K}^{(L)}$  (such as the limiting kernels  $\Sigma_{\infty}^{(L)}, \Lambda_{\infty}^{(L)}, \Upsilon_{\infty}^{(L)}, \Phi_{\infty}^{(L)}, \Xi_{\infty}^{(L)}$  introduced later), we will use the same notation, denoting the Gram matrix of the datapoints by  $\tilde{\mathcal{K}}$ .

# 3. MAIN THEOREMS

# 3.1. HESSIAN AS  $I + S$

Using the above setup, the Hessian  $H$  of the loss  $\mathcal{C} \circ F^{(L)}$  is the sum of two terms, with the entry  $H_{p,p'}$  given by

$$
H _ {p, p ^ {\prime}} = \mathcal {H C} _ {| f _ {\theta}} (\partial_ {\theta_ {p}} F, \partial_ {\theta_ {p ^ {\prime}}} F) + \mathcal {D C} _ {| f _ {\theta}} (\partial_ {\theta_ {p}, \theta_ {p ^ {\prime}}} F).
$$

For a finite dataset, the Hessian matrix  $\mathcal{H}\left(C\circ Y^{(L)}\right)$  is equal to the sum of two matrices

$$
I = \left(\mathcal {D} Y ^ {(L)}\right) ^ {T} \mathcal {H} C \mathcal {D} Y ^ {(L)} \quad \text {a n d} \quad S = \nabla C \cdot \mathcal {H} Y ^ {(L)}
$$

where  $\mathcal{D}Y^{(L)}$  is a  $Nn_{L}\times P$  matrix,  $\mathcal{H}C$  is a  $Nn_{L}\times Nn_{L}$  matrix and  $\mathcal{H}Y^{(L)}$  is a  $P\times P\times Nn_{L}$  tensor to which we apply a scalar product (denoted by  $\cdot$ ) in its last dimension with the  $Nn_{L}$  vector  $\nabla C$  to obtain a  $P\times P$  matrix.

Our main contribution is the following theorem, which describes the limiting moments  $\operatorname{Tr}\left(H^{k}\right)$  in terms of the moments of  $I$  and  $S$ :

Theorem 1. For any loss  $C$  with BGOSS and  $\sigma \in C_b^4 (\mathbb{R})$ , in the sequential limit  $n_1\to \infty ,\ldots ,n_{L - 1}\to \infty$ , we have for all  $k\geq 1$

$$
\operatorname {T r} \left(H (t) ^ {k}\right) \approx \operatorname {T r} \left(I (t) ^ {k}\right) + \operatorname {T r} \left(S (t) ^ {k}\right).
$$

The limits of  $\operatorname{Tr}\left(I(t)^k\right)$  and  $\operatorname{Tr}\left(S(t)^k\right)$  can be expressed in terms of the  $NTK\Theta_{\infty}^{(L)}$ , the kernel  $\Upsilon_{\infty}^{(L)}$  and the non-symmetric kernels  $\Phi_{\infty}^{(L)}$ ,  $\Lambda_{\infty}^{(L)}$  defined in Appendix C:

- The moments  $\operatorname{Tr}\left(I(t)^k\right)$  converge to the following limits (with the convention that  $i_{k+1} = i_1$ ):

$$
\mathrm {T r} \left(I (t) ^ {k}\right)\rightarrow \mathrm {T r} \left(\left(\mathcal {H} C (Y (t)) \tilde {\Theta}\right) ^ {k}\right) = \frac {1}{N ^ {k}} \sum_ {i _ {1}, \ldots , i _ {k} = 1} ^ {N} \prod_ {m = 1} ^ {k} c _ {i _ {m}} ^ {\prime \prime} (f _ {\theta (t)} (x _ {i _ {m}})) \Theta_ {\infty} ^ {(L)} (x _ {i _ {m}}, x _ {i _ {m + 1}}).
$$

- The first moment  $\operatorname{Tr}(S(t))$  converges to the limit:

$$
\operatorname {T r} (S (t)) = (G (t)) ^ {T} \nabla C (Y (t)).
$$

At initialization  $(G(0),Y(0))$  form a Gaussian pair of  $N n_{L}$ -vectors, independent for differing output indices  $k = 1,\dots,n_{L}$  and with covariance  $\mathbb{E}[G_{ik}(0)Y_{i'k'}(0)] = \delta_{kk'}\Phi_{\infty}^{(L)}(x_i,x_{i'})$  for a (non-symmetric)  $N\times N$  limiting kernel  $\Phi_{\infty}^{(L)}(x_i,x_{i'})$ . During training, both vectors follow the differential equations

$$
\begin{array}{l} \partial_ {t} G (t) = - \tilde {\Lambda} \nabla C (Y (t)) \\ \partial_ {t} Y (t) = - \tilde {\Theta} \nabla C (Y (t)). \\ \end{array}
$$

- The second moment  $\operatorname{Tr}\left(S(t)^2\right)$  converges to the following limit defined in terms of the Gram matrix  $\tilde{\Upsilon}$ :

$$
\operatorname {T r} \left(S ^ {2}\right)\rightarrow \left(\nabla C (Y (t))\right) ^ {T} \tilde {\Upsilon} \nabla C (Y (t))
$$

- The higher moments  $\operatorname{Tr}\left(S(t)^k\right)$  for  $k \geq 3$  vanish.

Proof. The moments of  $I$  and  $S$  can be studied separately because the moments of their sum is asymptotically equal to the sum of their moments by Proposition 5 below. The limiting moments of  $I$  and  $S$  are respectively described by Propositions 1 and 4 below.

In the case of a MSE loss  $C(Y) = \frac{1}{2N} \| Y - Y^* \|^2$ , the first and second derivatives take simple forms  $\nabla C(Y) = \frac{1}{N} (Y - Y^*)$  and  $\mathcal{H}C(Y) = \frac{1}{N} Id_{Nn_L}$  and the differential equations can be solved to obtain more explicit formulae:

Corollary 1. For the MSE loss  $C$  and  $\sigma \in C_b^4 (\mathbb{R})$ , in the limit  $n_1,\ldots ,n_{L - 1}\to \infty$ , we have uniformly over  $[0,T]$

$$
\operatorname {T r} \left(H (t) ^ {k}\right)\rightarrow \frac {1}{N ^ {k}} \operatorname {T r} \left(\tilde {\Theta} ^ {k}\right) + \operatorname {T r} \left(S (t) ^ {k}\right)
$$

where

$$
\begin{array}{l} \operatorname {T r} \left(S (t)\right)\rightarrow - \frac {1}{N} \left(Y ^ {*} - Y (0)\right) ^ {T} \left(I d _ {N n _ {L}} + e ^ {- t \tilde {\Theta}}\right) \tilde {\Theta} ^ {- 1} \tilde {\Lambda} ^ {T} e ^ {- t \tilde {\Theta}} \left(Y ^ {*} - Y (0)\right) \\ + \frac {1}{N} G (0) ^ {T} e ^ {- t \tilde {\Theta}} (Y ^ {*} - Y (0)) \\ \mathrm {T r} \left(S (t) ^ {2}\right)\rightarrow \frac {1}{N ^ {2}} (Y ^ {*} - Y (0)) ^ {T} e ^ {- t \tilde {\Theta}} \tilde {\Upsilon} e ^ {- t \tilde {\Theta}} (Y ^ {*} - Y (0)) \\ \operatorname {T r} \left(S (t) ^ {k}\right)\rightarrow 0 \quad w h e n k > 2. \\ \end{array}
$$

In expectation we have:

$$
\mathbb {E} \left[ \operatorname {T r} (S (t)) \right]\rightarrow - \frac {1}{N} T r \left(\left(I d _ {N n _ {L}} + e ^ {- t \tilde {\Theta}}\right) \tilde {\Theta} ^ {- 1} \tilde {\Lambda} ^ {T} e ^ {- t \tilde {\Theta}} \left(\tilde {\Sigma} + Y ^ {*} Y ^ {* T}\right)\right) + \frac {1}{N} T r \left(e ^ {- t \tilde {\Theta}} \tilde {\Phi} ^ {T}\right)
$$

$$
\mathbb {E} \left[ \operatorname {T r} \left(S (t) ^ {2}\right)\right]\rightarrow \frac {1}{N ^ {2}} T r \left(e ^ {- t \tilde {\Theta}} \tilde {\Upsilon} e ^ {- t \tilde {\Theta}} \left(\tilde {\Sigma} + Y ^ {*} Y ^ {* T}\right)\right).
$$

Proof. The moments of  $I$  are constant because  $\mathcal{H}C = \frac{1}{N}Id_{Nn_L}$  is constant. For the moments of  $S$ , we first solve the differential equation for  $Y(t)$ :

$$
Y (t) = Y ^ {*} - e ^ {- t \tilde {\Theta}} (Y ^ {*} - Y (0)).
$$

Noting  $Y(t) - Y(0) = -\tilde{\Theta}\int_0^t\nabla C(s)ds$ , we have

$$
\begin{array}{l} G (t) = G (0) - \tilde {\Lambda} \int_ {0} ^ {t} \nabla C (s) d s \\ = G (0) + \tilde {\Lambda} \tilde {\Theta} ^ {- 1} (Y (t) - Y (0)) \\ = G (0) + \tilde {\Lambda} \tilde {\Theta} ^ {- 1} \left(I d _ {N n _ {L}} + e ^ {- t \tilde {\Theta}}\right) \left(Y ^ {*} - Y (0)\right) \\ \end{array}
$$

The expectation of the first moment of  $S$  then follows.

![](images/be67f13ab639e9da66d7267e401500cd54cfffd7a7fdf2dd371aa0b519bf1369.jpg)

# 3.2. MUTUAL ORTHOGONALITY OF  $I$  AND  $S$

A first key ingredient to prove Theorem 1 is the asymptotic mutual orthogonality of the matrices  $I$  and  $S$

Proposition (Proposition 5 in Appendix D). For any loss  $C$  with BGOSS and  $\sigma \in C_b^4 (\mathbb{R})$ , we have uniformly over  $[0,T]$

$$
\lim  _ {n _ {L - 1} \to \infty} \dots \lim  _ {n _ {1} \to \infty} \| I S \| _ {F} = 0.
$$

As a consequence  $\lim_{n_{L - 1}\to \infty}\dots \lim_{n_1\to \infty}\operatorname {Tr}\left([I + S]^k\right) - \left[\operatorname {Tr}\left(I^k\right) + \operatorname {Tr}\left(S^k\right)\right] = 0.$

Remark 1. If two matrices  $A$  and  $B$  are mutually orthogonal (i.e.  $AB = 0$ ) the range of  $A$  is contained in the nullspace of  $B$  and vice versa. The non-zero eigenvalues of the sum  $A + B$  are therefore given by the union of the non-zero eigenvalues of  $A$  and  $B$ . Furthermore the moments of  $A$  and  $B$  add up:  $\operatorname{Tr}\left([A + B]^k\right) = \operatorname{Tr}\left(A^k\right) + \operatorname{Tr}\left(B^k\right)$ . Proposition 5 shows that this is what happens asymptotically for  $I$  and  $S$ .

Note that both matrices  $I$  and  $S$  have large nullspaces: indeed assuming a constant width  $w = n_1 = \ldots = n_{L-1}$ , we have  $\text{Rank}(I) \leq Nn_L$  and  $\text{Rank}(S) \leq 2(L - 1)wNn_L$  (see Appendix C), while the number of parameters  $P$  scales as  $w^2$  (when  $L > 2$ ).

Figure 2 illustrates the mutual orthogonality of  $I$  and  $S$ .

![](images/d29487a1e8483899cd4402a52c3c2ad8354441e369b0cccfb948b097dc0dce13.jpg)  
FIGURE 1. Comparison of the theoretical prediction of Corollary 1 for the expectation of the first 4 moments (colored lines) to the empirical average over 250 trials (black crosses) for a rectangular network with two hidden layers of finite widths  $n_1 = n_2 = 5000$  ( $L = 3$ ) with the smooth ReLU (left) and the normalized smooth ReLU (right), for the MSE loss on scaled down 14x14 MNIST with  $N = 256$ . Only the first two moments are affected by  $S$  at the beginning of training.

![](images/347dd6aff6ba5d2d2cf35cdc05405a9836137d881a213a35a9c4c0214f333f95.jpg)

# 3.3. THE MATRIX  $S$

The matrix  $S = \nabla C\cdot \mathcal{H}Y^{(L)}$  is best understood as a perturbation to  $I$ , which vanishes as the network converges because  $\nabla C\rightarrow 0$ . To calculate its moments, we note that

$$
\mathrm {T r} \left(\nabla C \cdot \mathcal {H} Y ^ {(L)}\right) = \left(\sum_ {p = 1} ^ {P} \partial_ {\theta_ {p} ^ {2}} ^ {2} Y\right) ^ {T} \nabla C = G ^ {T} \nabla C,
$$

where the vector  $G = \sum_{k=1}^{P} \partial_{\theta_p^2}^2 Y \in \mathbb{R}^{Nn_L}$  is the evaluation of the function  $g_\theta(x) = \sum_{k=1}^{P} \partial_{\theta_p^2}^2 f_\theta(x)$  on the training set.

For the second moment we have

$$
\operatorname {T r} \left(\left(\nabla C \cdot \mathcal {H} Y ^ {(L)}\right) ^ {2}\right) = \nabla C ^ {T} \left(\sum_ {p, p ^ {\prime} = 1} ^ {P} \partial_ {\theta_ {p} \theta_ {p ^ {\prime}}} ^ {2} Y \left(\partial_ {\theta_ {p} \theta_ {p ^ {\prime}}} ^ {2} Y\right) ^ {T}\right) \nabla C = \nabla C ^ {T} \tilde {\Upsilon} \nabla C
$$

for  $\tilde{\Upsilon}$  the Gram matrix of the kernel  $\Upsilon^{(L)}(x,y) = \sum_{p,p^{\prime} = 1}^{P}\partial_{\theta_{p}\theta_{p^{\prime}}}^{2}f_{\theta}(x)\left(\partial_{\theta_{p}\theta_{p^{\prime}}}^{2}f_{\theta}(y)\right)^{T}$ .

The following proposition describes the limit of the function  $g_{\theta}$  and the kernel  $\Upsilon^{(L)}$  and the vanishing of the higher moments:

Proposition (Proposition 4 in Appendix C). For any loss  $C$  with BGOSS and  $\sigma \in C_b^4 (\mathbb{R})$ , the first two moments of  $S$  take the form

$$
\operatorname {T r} (S (t)) = G (t) ^ {T} \nabla C (t)
$$

$$
\operatorname {T r} \left(S (t) ^ {2}\right) = \nabla C (t) ^ {T} \tilde {\Upsilon} (t) \nabla C (t)
$$

- At initialization,  $g_{\theta}$  and  $f_{\theta}$  converge to a (centered) Gaussian pair with covariances

$$
\mathbb {E} \left[ g _ {\theta , k} (x) g _ {\theta , k ^ {\prime}} \left(x ^ {\prime}\right) \right] = \delta_ {k k ^ {\prime}} \Xi_ {\infty} ^ {(L)} \left(x, x ^ {\prime}\right)
$$

$$
\mathbb {E} \left[ g _ {\theta , k} (x) f _ {\theta , k ^ {\prime}} \left(x ^ {\prime}\right) \right] = \delta_ {k k ^ {\prime}} \Phi_ {\infty} ^ {(L)} \left(x, x ^ {\prime}\right)
$$

$$
\mathbb {E} [ f _ {\theta , k} (x) f _ {\theta , k ^ {\prime}} (x ^ {\prime}) ] = \delta_ {k k ^ {\prime}} \Sigma_ {\infty} ^ {(L)} (x, x ^ {\prime})
$$

and during training  $g_{\theta}$  evolves according to

$$
\partial_ {t} g _ {\theta , k} (x) = \sum_ {i = 1} ^ {N} \Lambda_ {\infty} ^ {(L)} (x, x _ {i}) \partial_ {i k} C (Y (t)).
$$

- Uniformly over any interval  $[0, T]$ , the kernel  $\Upsilon^{(L)}$  has a deterministic and fixed limit  $\lim_{n_{L-1} \to \infty} \dots \lim_{n_1 \to \infty} \Upsilon_{kk'}^{(L)}(x, x') = \delta_{kk'} \Upsilon_{\infty}^{(L)}(x, x')$  with limiting kernel:

$$
\Upsilon_ {\infty} ^ {(L)} (x, x ^ {\prime}) = \sum_ {\ell = 1} ^ {L - 1} \left(\Theta_ {\infty} ^ {(\ell)} (x, x ^ {\prime}) ^ {2} \ddot {\Sigma} _ {\infty} ^ {(\ell)} (x, x ^ {\prime}) + 2 \Theta_ {\infty} ^ {(\ell)} (x, x ^ {\prime}) \dot {\Sigma} _ {\infty} ^ {(\ell)} (x, x ^ {\prime})\right) \dot {\Sigma} _ {\infty} ^ {(\ell + 1)} (x, x ^ {\prime}) \dots \dot {\Sigma} _ {\infty} ^ {(L - 1)} (x, x ^ {\prime}).
$$

- The higher moment  $k > 2$  vanish:  $\lim_{n_{L - 1} \to \infty} \cdots \lim_{n_1 \to \infty} \operatorname{Tr}\left(S^k\right) = 0$ .

This result has a number of consequences for infinitely wide networks:

(1) At initialization, the matrix  $S$  has a finite Frobenius norm  $\| S \|_F^2 = \operatorname{Tr}(S^2) = \nabla C^T \tilde{\Upsilon} \nabla C$ , because  $\Upsilon$  converges to a fixed limit. As the network converges, the derivative of the cost goes to zero  $\nabla C(t) \to 0$  and so does the Frobenius norm of  $S$ .  
(2) In contrast the operator norm of  $S$  vanishes already at initialization (because for all even  $k$ , we have  $\| S \|_{op} \leq \sqrt[k]{\operatorname{Tr}(S^k)} \to 0$ ). At initialization, the vanishing of  $S$  in operator norm but not in Frobenius norm can be explained by the matrix  $S$  having a growing number of eigenvalues of shrinking intensity as the width grows.  
(3) When it comes to the first moment of  $S$ , Proposition 4 shows that the spectrum of  $S$  is in general not symmetric. For the MSE loss the expectation of the first moment at initialization is

$$
\mathbb {E} \left[ \operatorname {T r} (S) \right] = \mathbb {E} \left[ (Y - Y ^ {*}) ^ {T} G \right] = \mathbb {E} \left[ Y ^ {T} G \right] - (Y ^ {*}) ^ {T} \mathbb {E} [ G ] = \operatorname {T r} \left(\tilde {\Phi}\right) - 0
$$

which may be positive or negative depending on the choice of nonlinearity: with a smooth ReLU, it is positive, while for the arc-tangent or the normalized smooth ReLU, it can be negative (see Figure 1).

This is in contrast to the result obtained in Pennington & Bahri (2017); Geiger et al. (2018) for the shallow ReLU networks, taking the second derivative of the ReLU to be zero. Under this assumption the spectrum of  $S$  is symmetric: if the eigenvalues are ordered from lowest to highest,  $\lambda_{i} = -\lambda_{P - i}$  and  $\mathrm{Tr}(S) = 0$ .

These observations suggest that  $S$  has little influence on the shape of the surface, especially towards the end of training, the matrix  $I$  however has an interesting structure.

# 3.4. THE MATRIX  $I$

At a global minimizer  $\theta^{*}$ , the spectrum of  $I$  describes how the loss behaves around  $\theta^{*}$ . Along the eigenvectors of the biggest eigenvalues of  $I$ , the loss increases rapidly, while small eigenvalues correspond to flat directions. Numerically, it has been observed that the matrix  $I$  features a few dominating eigenvalues and a bulk of small eigenvalues Sagun et al. (2016; 2017); Gur-Ari et al. (2018); Papyan (2019). This leads to a narrow valley structure of the loss around a minimum: the biggest eigenvalues are the 'cliffs' of the valley, i.e. the directions along which the loss grows fastest, while the small eigenvalues form the 'flat directions' or the bottom of the valley.

Note that the rank of  $I$  is bounded by  $Nn_{L}$  and in the overparametrized regime, when  $Nn_{L} < P$ , the matrix  $I$  will have a large nullspace, these are directions along which the value of the function on the training set does not change. Note that in the overparametrized regime, global minima are not isolated: they lie in a manifold of dimension at least  $P - Nn_{L}$  and the nullspace of  $I$  is tangent to this solution manifold.

![](images/493b7c6a6b447855225281729dd7d62d86a8eb9f87f6e865b26d63cd51dbc36e.jpg)  
FIGURE 2. Illustration of the mutual orthogonality of  $I$  and  $S$ . For the 20 first eigenvectors of  $I$  (blue) and  $S$  (orange), we plot the Rayleigh quotients  $v^{T}Iv$  and  $v^{T}Sv$  (with  $L = 3$ ,  $n_{1} = n_{2} = 1000$  and the normalized ReLU on 14x14 MNIST with  $N = 256$ ). We see that the directions where  $I$  is large are directions where  $S$  is small and vice versa.

![](images/8d5d8daeb309fd7ab84da7058411b405c948c641dfd39f51d7aac4a71253e4fc.jpg)  
FIGURE 3. Plot of the loss surface around a global minimum along the first (along the y coordinate) and fourth (x coordinate) eigenvectors of  $I$ . The network has  $L = 4$ , width  $n_1 = n_2 = n_3 = 1000$  for the smooth ReLU (left) and the normalized smooth ReLU (right). The data is uniform on the unit disk. Normalizing the non-linearity greatly reduces the narrow valley structure of the loss thus speeding up training.

The matrix  $I$  is closely related to the NTK Gram matrix:

$$
\tilde {\Theta} = \mathcal {D} Y ^ {(L)} \left(\mathcal {D} Y ^ {(L)}\right) ^ {T} \text {a n d} I = \left(\mathcal {D} Y ^ {(L)}\right) ^ {T} \mathcal {H} C \mathcal {D} Y ^ {(L)}.
$$

As a result, the limiting spectrum of the matrix  $I$  can be directly obtained from the  $\mathrm{NTK}^2$

Proposition 1. For any loss  $C$  with BGOSS and  $\sigma \in C_b^4 (\mathbb{R})$ , uniformly over any interval  $[0,T]$ , the moments  $\operatorname {Tr}\left(I^k\right)$  converge to the following limit (with the convention that  $i_{k + 1} = i_1$ ):

$$
\lim  _ {n _ {L - 1} \to \infty} \dots \lim  _ {n _ {1} \to \infty} \operatorname {T r} \left(I ^ {k}\right) = \operatorname {T r} \left(\left(\mathcal {H} C (Y _ {t}) \tilde {\Theta}\right) ^ {k}\right) = \frac {1}{N ^ {k}} \sum_ {i _ {1}, \dots , i _ {k} = 1} ^ {N} \prod_ {m = 1} ^ {k} c _ {i _ {m}} ^ {\prime \prime} \left(f _ {\theta (t)} \left(x _ {i _ {m}}\right)\right) \Theta_ {\infty} ^ {(L)} \left(x _ {i _ {m}}, x _ {i _ {m + 1}}\right)
$$

Proof. It follows from  $\operatorname{Tr}\left(I^k\right) = \operatorname{Tr}\left(\left(\left(\mathcal{D}Y^{(L)}\right)^T\mathcal{H}C\mathcal{D}Y^{(L)}\right)^k\right) = \operatorname{Tr}\left(\left(\mathcal{H}C\tilde{\Theta}\right)^k\right)$  and the asymptotic of the NTK Jacot et al. (2018).

# 3.4.1. MEAN-SQUARE ERROR

When the loss is the MSE,  $\mathcal{HC}$  is equal to  $\frac{1}{N} Id_{Nn_L}$ . As a result,  $\tilde{\Theta}$  and  $I$  have the same non-zero eigenvalues up to a scaling of  $1/N$ . Because the NTK is asymptotically fixed, the spectrum of  $I$  is also fixed in the limit.

The eigenvectors of the NTK Gram matrix are the kernel principal components of the data. The biggest principal components are the directions in function space which are most favoured by the NTK. This gives a functional interpretation of the narrow valley structure in DNNs: the cliffs of the valley are the biggest principal components, while the flat directions are the smallest components.

Remark 2. As the depth  $L$  of the network increases, one can observe two regimes Poole et al. (2016); Jacot et al. (2019): Order/Freeze where the NTK converges to a constant and Chaos where the NTK converges to a Kronecker delta. In the Order/Freeze the  $Nn_{L} \times Nn_{L}$

Gram matrix approaches a block diagonal matrix with  $n_{L}$  constant blocks, and as a result  $n_{L}$  eigenvalues of  $I$  dominate the other ones, corresponding to constant directions along each outputs (this is line with the observations of Papyan (2019)). This leads to a narrow valley for the loss and slows down training. In contrast, in the Chaos regime, the NTK Gram matrix approaches a scaled identity matrix, and the spectrum of  $I$  should hence concentrate around a positive value, hence speeding up training. Figure 3 illustrates this phenomenon: with the smooth ReLU we observe a narrow valley, while with the normalized smooth ReLU (which lies in the Chaos according to Jacot et al. (2019)) the narrowness of the loss is reduced. A similar phenomenon may explain why normalization helps smoothing the loss surface and speed up training Santurkar et al. (2018); Ghorbani et al. (2019).

# 3.4.2. CROSS-ENTROPY LOSS

For a binary cross-entropy loss with labels  $Y^{*} \in \{-1, + 1\}^{N}$

$$
C (Y) = \frac {1}{N} \sum_ {i = 1} ^ {N} \log \left(1 + e ^ {- Y _ {i} ^ {*} Y _ {i}}\right),
$$

$\mathcal{HC}$  is a diagonal matrix whose entries depend on  $Y$  (but not on  $Y^{*}$ ):

$$
\mathcal {H} _ {i i} C (Y) = \frac {1}{N} \frac {1}{1 + e ^ {- Y _ {i}} + e ^ {Y _ {i}}}.
$$

The eigenvectors of  $I$  then correspond to the weighted kernel principal component of the data. The positive weights  $\frac{1}{1 + e^{-Y_i} + e^{Y_i}}$  approach  $1/3$  as  $Y_i$  goes to 0, i.e. when it is close to the decision boundary from one class to the other, and as  $Y_i \to \pm \infty$  the weight goes to zero. The weights evolve in time through  $Y_i$ , the spectrum of  $I$  is therefore not asymptotically fixed as in the MSE case, but the functional interpretation of the spectrum in terms of the kernel principal components remains.

# 4. CONCLUSION

We have given an explicit formula for the limiting moments of the Hessian of DNNs throughout training. We have used the common decomposition of the Hessian in two terms  $I$  and  $S$  and have shown that the two terms are asymptotically mutually orthogonal, such that they can be studied separately.

The matrix  $S$  vanishes in Frobenius norm as the network converges and has vanishing operator norm throughout training. The matrix  $I$  is arguably the most important as it describes the narrow valley structure of the loss around a global minimum. The eigendecomposition of  $I$  is related to the (weighted) kernel principal components of the data w.r.t. the NTK.

# REFERENCES

Zeyuan Allen-Zhu, Yanzhi Li, and Zhao Song. A Convergence Theory for Deep Learning via Over-Parameterization. CoRR, abs/1811.03962, 2018. URL http://arxiv.org/abs/1811.03962.  
Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net. arXiv preprint arXiv:1904.11955, 2019.  
Marco Baity-Jesi, Levent Sagun, Mario Geiger, Stefano Spigler, Gerard Ben Arous, Chiara Cammarota, Yann LeCun, Matthieu Wyart, and Giulio Biroli. Comparing Dynamics: Deep Neural Networks versus Glassy Systems. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80, pp. 314-323. PMLR, 10-15 Jul 2018. URL http://proceedings.mlr.press/v80/baity-jesi18a.html.  
Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina. Entropy-sgd: Biasing gradient descent into wide valleys. arXiv preprint arXiv:1611.01838, 2016.

Lenaïc Chizat and Francis Bach. On the Global Convergence of Gradient Descent for Over-parameterized Models using Optimal Transport. In Advances in Neural Information Processing Systems 31, pp. 3040-3050. Curran Associates, Inc., 2018a. URL http://papers.nips.cc/paper/7567-on-the-global-convergence-of-gradient-descent-for-over-parameterized-models-using-optimal-transport.pdf.  
Lenaic Chizat and Francis Bach. A note on lazy training in supervised differentiable programming. arXiv preprint arXiv:1812.07956, 2018b.  
Youngmin Cho and Lawrence K. Saul. Kernel Methods for Deep Learning. In Advances in Neural Information Processing Systems 22, pp. 342-350. Curran Associates, Inc., 2009. URL http://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdf.  
Anna Choromanska, Mikael Henaff, Michael Mathieu, Gérard Ben Arous, and Yann LeCun. The Loss Surfaces of Multilayer Networks. Journal of Machine Learning Research, 38: 192-204, nov 2015. URL https://arxiv.org/pdf/1412.0233.pdf.  
Yann N. Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio. Identifying and Attacking the Saddle Point Problem in High-dimensional Non-convex Optimization. In Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2, NIPS'14, pp. 2933-2941, Cambridge, MA, USA, 2014. MIT Press.  
Simon S. Du, Xiyu Zhai, Barnabás Póczos, and Aarti Singh. Gradient Descent Provably Optimizes Over-parameterized Neural Networks. 2019.  
Mario Geiger, Stefano Spigler, Stéphane d'Ascoli, Levent Sagun, Marco Baity-Jesi, Giulio Biroli, and Matthieu Wyart. The jamming transition as a paradigm to understand the loss landscape of deep neural networks. arXiv preprint arXiv:1809.09349, 2018.  
Mario Geiger, Arthur Jacot, Stefano Spigler, Franck Gabriel, Levent Sagun, Stéphane d'Ascoli, Giulio Biroli, Clément Hongler, and Matthieu Wyart. Scaling description of generalization with number of parameters in deep learning. abs/1901.01608, 2019. URL http://arxiv.org/abs/1901.01608.  
Behrooz Ghorbani, Shankar Krishnan, and Ying Xiao. An investigation into neural net optimization via hessian eigenvalue density. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 2232-2241, Long Beach, California, USA, 09-15 Jun 2019. PMLR. URL http://proceedings.mlr.press/v97/ghorbani19b.htm1.  
Guy Gur-Ari, Daniel A. Roberts, and Ethan Dyer. Gradient descent happens in a tiny subspace. CoRR, abs/1812.04754, 2018. URL http://arxiv.org/abs/1812.04754.  
Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural Computation, 9(1):1-42, 1997.  
Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural Tangent Kernel: Convergence and Generalization in Neural Networks. In Advances in Neural Information Processing Systems 31, pp. 8580-8589. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/8076-neural-tangent-kernel-convergence-and-generalization-in-neural-networks.pdf.  
Arthur Jacot, Franck Gabriel, and Clément Hongler. Freeze and chaos for dnns: an NTK view of batch normalization, checkerboard and boundary effects. CoRR, abs/1907.05715, 2019. URL http://arxiv.org/abs/1907.05715.  
Ryo Karakida, Shotaro Akaho, and Shun-Ichi Amari. Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach. jun 2018. URL http://arxiv.org/abs/1806.01316.  
Jae Hoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, and Jascha Sohl-Dickstein. Deep Neural Networks as Gaussian Processes. *ICLR*, 2018.  
Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. Proceedings of the National Academy of Sciences, 115(33): E7665-E7671, 2018.  
Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Mean-field theory of two-layers neural networks: dimension-free bounds and kernel limit. arXiv preprint arXiv:1902.06015, 2019.  
Radford M. Neal. Bayesian Learning for Neural Networks. Springer-Verlag New York, Inc., Secaucus, NJ, USA, 1996. ISBN 0387947248.  
Vardan Papyan. Measurements of three-level hierarchical structure in the outliers in the spectrum of deepnet hessenians. CoRR, abs/1901.08244, 2019. URL http://arxiv.org/

abs/1901.08244.  
Razvan Pascanu and Yoshua Bengio. Revisiting Natural Gradient for Deep Networks. jan 2013. URL http://arxiv.org/abs/1301.3584.  
Razvan Pascanu, Yann N Dauphin, Surya Ganguli, and Yoshua Bengio. On the saddle point problem for non-convex optimization. arXiv preprint, 2014. URL https://arxiv.org/pdf/1405.4604.pdf.  
Jeffrey Pennington and Yasaman Bahri. Geometry of Neural Network Loss Surfaces via Random Matrix Theory. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pp. 2798-2806. PMLR, 06-11 Aug 2017. URL http://proceedings.mlr.press/v70/pennington17a.html.  
Jeffrey Pennington and Pratik Worah. The Spectrum of the Fisher Information Matrix of a Single-Hidden-Layer Neural Network. In Advances in Neural Information Processing Systems 31, pp. 5415-5424. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/7786-the-spectrum-of-the-fisher-information-matrix-of-a-single-hidden-layer-neural-network.pdf.  
Ben Poole, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, and Surya Ganguli. Exponential expressivity in deep neural networks through transient chaos. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett (eds.), Advances in Neural Information Processing Systems 29, pp. 3360-3368. Curran Associates, Inc., 2016. URL http://papers.nips.cc/paper/6322-exponential-expressivity-in-deep-neural-networks-through-transient-chaos.pdf.  
Grant Rotskoff and Eric Vanden-Eijnden. Parameters as interacting particles: long time convergence and asymptotic error scaling of neural networks. In Advances in Neural Information Processing Systems 31, pp. 7146-7155. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/7945-parameters-as-interacting-particles-long-time-convergence-and-asymptotic-error-scaling-of-neural-networks.pdf.  
Levent Sagun, Léon Bottou, and Yann LeCun. Singularity of the hessian in deep learning. CoRR, abs/1611.07476, 2016. URL http://arxiv.org/abs/1611.07476.  
Levent Sagun, Utku Evci, V. Ugur Güney, Yann Dauphin, and Léon Bottou. Empirical Analysis of the Hessian of Over-Parametrized Neural Networks. CoRR, abs/1706.04454, 2017.  
Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How does batch normalization help optimization? In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems 31, pp. 2483-2493. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/7515-how-does-batch-normalization-help-optimization.pdf.  
Daniel Wagenaar. Information geometry of neural networks. 1998. ISSN 0302-9743.  
Lei Wu, Zhanxing Zhu, and Weinan E. Towards Understanding Generalization of Deep Learning: Perspective of Loss Landscapes. CoRR, abs/1706.10239, 2017. URL http://arxiv.org/abs/1706.10239.
