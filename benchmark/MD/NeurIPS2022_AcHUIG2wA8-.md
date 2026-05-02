# Non-Gaussian Tensor Programs

Anonymous Author(s)

Affiliation

Address

email

# Abstract

The Tensor Programs framework has produced a series of powerful results by 1) expressing any deep learning computation of concern as a principled composition of elementwise nonlinearities and matrix multiplication, and 2) inductively reasoning about the program behavior as the sizes of the matrices in the program tend to infinity. For example, this framework showed that infinitely wide neural networks of any architecture exhibit Gaussian process behavior at initialization and evolve like a kernel model during training in the so-called NTK parametrization [21, 22, 26]. Moreover, this framework yielded a novel parameterization, coined  $\mu \mathrm{P}$  [24], that for the first time enabled hyperparameter tuning for enormous networks too expensive to train more than once [25]. However, this framework has so far been limited to Gaussian initialized weights, while uniform or truncated Gaussian distributions are more prevalent in practice. This work extends Tensor Programs to general non-Gaussian initializations, thus recovering all of the above results in all practical settings.

# 1 Introduction

Just like autograd [17] empirically automates the calculation of chain rule of arbitrary computation graphs, Tensor Programs (TP) [20] has automated the theoretical calculation of infinite-width limits of the same (where width of a computation graph corresponds to the size of matrices). What previously were difficult limits to calculate, now becomes routine via TP. For example, [10] in 1994 showed that randomly initialized wide shallow neural networks are Gaussian Processes (which is called the Neural Network-Gaussian Process Correspondence, or NNGP Correspondence), but only recently this has been extended to deep perceptrons [8, 9] and more advanced architectures such as convolutional neural networks [2, 11], and each such extension requires painstaking calculations and careful applications of Law of Large Numbers and Central Limit Theorem. But with Tensor Programs, one can show that NNGP Correspondence holds for any architecture all at once [21]. Similarly, in a certain parametrization, a wide multi-layer perceptron (MLP) evolves like a linear model during training [6], but showing this for advanced architectures was very difficult. TP [22, 26] again was able to prove this behavior for any architecture. Finally, TP gave rise to the Dynamical Dichotomy Theorem [24], a classification of all natural infinite-width limits of neural networks, and led to the discovery of Maximal Update Parametrization, or  $\mu$ P. These results underlie the hyperparameter transfer technology that for the first time enabled the hyperparameter tuning of enormous neural networks too expensive to train more than once [25].

However, these results were only proven for neural networks randomly initialized with Gaussian weights. But in practice, non-Gaussian initializations such as uniform or truncated Gaussian, proliferate. Could such non-Gaussian initializations possess different behavior in wide neural networks?

In this work, we answer NO: under mild conditions, non-Gaussian iid initializations behave the same as Gaussian initializations with the same variance in wide neural networks. From the perspective of universality [14] from statistical mechanics, this is not unexpected; yet proving such results rigorously

Submitted to 36th Conference on Neural Information Processing Systems (NeurIPS 2022). Do not distribute.

has always been tricky [18]. In our work, we apply the interpolation technique of [1] to interpolate between a non-Gaussian Tensor Program in question and the corresponding Gaussian Tensor Program, and show that this continuous interpolation in the infinite-width limit has identically zero derivative, and therefore the two programs have the same limits.

This result also makes the proof of the Semicircle and Marchenko-Pastur Laws in [23] automatically valid for non-Gaussian random matrix ensembles. Likewise, it shows that the asymptotic singular value distribution of the input-output Jacobian of a random neural network does not depend on the distribution of random initialization beyond its variance.

# 2 Setup

Given initial vectors  $g^{1}, \ldots, g^{M_{0}} \in \mathbb{R}^{n}$  and initial scalars  $c^{1}, \ldots, c^{M_{0}} \in \mathbb{R}$ , consider the following iteration for  $m = M_{0} + 1, \ldots, M$  that generates new vectors  $g^{M_{0} + 1}, \ldots, g^{M} \in \mathbb{R}^{n}$  and scalars  $c^{M_{0} + 1}, \ldots, c^{M} \in \mathbb{R}$ :

$$
g _ {\alpha} ^ {m} = \sum_ {\beta = 1} ^ {n} A _ {\alpha \beta} ^ {m, T _ {m}} x _ {\beta} ^ {m}, \quad c ^ {m} = \frac {1}{n} \sum_ {\beta = 1} ^ {n} x _ {\beta} ^ {m}, \quad \text {w h e r e} x _ {\alpha} ^ {m} = \phi^ {m} \left(g _ {\alpha} ^ {1}, \dots , g _ {\alpha} ^ {m - 1}, c ^ {1}, \dots , c ^ {m - 1}\right). \tag {1}
$$

Here  $\phi^m$  is a scalar function with  $2m - 2$  arguments,  $A^m$  is an  $n\times n$  matrix, and  $T_{m}\in \{1,\top \}$  marks whether this matrix should be transposed or not. The matrices  $A^m$  for different  $m$  can possibly be the same. In this work, we shall call any computation of this form a Tensor Program, or TP.

This formulation of a Tensor Program is equivalent to  $\mathsf{NETSOR}^{\top +}$  in [23], as shown in Appendix D. As such, Eq. (1) can express any computation expressible in a DL framework such as PyTorch [13], including gradient descent iterations of neural networks of any architecture, e.g. [24, 26]. This expressivity allows one to treat a wide range of problems uniformly using just Theorem 1 below.

For example, consider the first forward pass of a simple MLP with scalar input  $\xi$  and output  $f(\xi)$ :

$$
f (\xi) = V ^ {\top} \sigma (W \sigma (\xi U)),
$$

where  $\xi \in \mathbb{R}$ ;  $U, V \in \mathbb{R}^n$ ;  $W \in \mathbb{R}^{n \times n}$ . We can express this in a TP as follows:  $g^1 = U, g^2 = nV$  are the initial vectors and  $c^1 = \xi, c^2 = 1$  are initial scalars (where  $c^2$  will just be ignored). Then the program computes

$$
g ^ {3} = A ^ {3} x ^ {3}, \quad \text {w h e r e} A ^ {3} = W, x ^ {3} = \sigma (\xi U)
$$

$$
f (\xi) = c ^ {4} = \frac {1}{n} \sum_ {\beta = 1} ^ {n} x _ {\beta} ^ {4}, \quad \text {w h e r e} x ^ {4} = (n V) \odot \sigma (g ^ {3})
$$

with  $c^3$  and  $g^4$  computed but just discarded. See [21, 22, 26, 24] for more examples.

Gaussian Tensor Programs. [20, 23, 26] most commonly work under the following setup:

Setup 1. Consider a Tensor Program with  $M$  vectors  $g^{1},\ldots ,g^{M}\in \mathbb{R}^{n}$  and scalars  $c^1,\dots ,c^M$  Suppose 1) all initial vectors  $g^1,\ldots ,g^{M_0}$  are sampled as  $(g_{\alpha}^{1},\ldots ,g_{\alpha}^{M_{0}})\sim \mathcal{N}(\mu^{in},\Sigma^{in})$  iid over  $\alpha = 1,\dots ,n$  , for some  $\mu^{in}\in \mathbb{R}^{M_0},\Sigma^{in}\in \mathbb{R}^{M_0\times M_0};2)$  all initial scalars  $c^1,\ldots ,c^{M_0}$  have almost sure limits as  $n\to \infty$  ; 3) all matrices  $A^m$  have iid entries from  $\mathcal{N}(0,n^{-1})$  ; 4) any two matrices  $A^m,A^l$  are either equal or independently sampled; 5) all the nonlinearities  $\phi^m$  are pseudo-Lipschitz.

Under this setup, the following Master theorem holds:

Theorem 1 ([23]). Consider Setup 1. Then, as  $n \to \infty$ , for any pseudo-Lipschitz  $\psi$

$$
\frac {1}{n} \sum_ {\alpha = 1} ^ {n} \psi \left(g _ {\alpha} ^ {1}, \dots , g _ {\alpha} ^ {M}, c ^ {1}, \dots , c ^ {M}\right) \stackrel {{a. s.}} {{\longrightarrow}} \stackrel {{\circledast}} {{\Psi}}, \tag {2}
$$

where  $\hat{\Psi}$  is a deterministic scalar given by a certain recurrent formula. Moreover, if all nonlinearities  $\phi^m$  are linearly bounded and  $\psi$  is quadratically bounded, we get convergence also in mean.

Non-Gaussian Tensor Programs. We shall generalize Theorem 1 to non-Gaussian distributions.

Definition 1. We say  $f: \mathbb{R}^k \to \mathbb{R}$  is polynomially smooth if it is  $C^\infty$  and its partial derivatives of any order are polynomially bounded, i.e. for any multidegree  $\alpha \geq 0$ ,  $\left| \frac{\partial^{|\alpha|}}{\partial \mathbf{x}^{\alpha}} f(\mathbf{x}) \right| \leq C (1 + |x_1|^p + \dots + |x_k|^p)$  for some  $C, p > 0$  that may depend on  $\alpha$ .

Setup 2. Consider Setup 1, but replace 3) and 5) with the following:  $3^{*}$ ) all matrices  $A^m$  have iid entries from a distribution (that could vary with  $m$ ) with zero mean, variance  $n^{-1}$ , and all higher moment existing;  $5^{*}$ ) all the nonlinearities  $\phi^m$  are polynomially smooth. We further require 6) all moments of initial scalars  $c^1, \ldots, c^{M_0}$  to exist and converge as  $n \to \infty$ .

Theorem 2 (Ours). Consider Setup 2. Then, as  $n\to \infty$  , for any polynomially smooth  $\psi$

$$
\frac {1}{n} \sum_ {\alpha = 1} ^ {n} \psi \left(g _ {\alpha} ^ {1}, \dots , g _ {\alpha} ^ {M}, c ^ {1}, \dots , c ^ {M}\right) \xrightarrow {\text {p r o b}} \stackrel {{\circledast}} {\Psi} \tag {3}
$$

with the same  $\dot{\Psi}$  as in Theorem 1 (but here convergence is in probability). Moreover, if all the nonlinearities  $\phi^m$  are linearly bounded and  $\psi$  is quadratically bounded, then the expectation of the LHS of Eq. (3) converges to the RHS.

In short, Theorem 2 relaxes matrix sampling to be non-Gaussian in general, at the cost of a) requiring more smoothness in nonlinearities, and b) weakening almost sure convergence to in-probability. Note that Setup 2 still requires initial vectors to be Gaussian, but this is not essential, as we discuss in Appendix A.

On Tensor Programs with variable dimensions. While Setup 1 and 2 assume all hidden dimensions to be equal, Theorem 1 holds also for Tensor Programs with variable dimensions, see e.g. [23]. Since our proof technique is based on interpolation between Gaussian and non-Gaussian weights, it can be straightforwardly extended to variable dimensions, as long as Theorem 1 holds in this setting.

# 3 Applications

As mentioned in the introduction, the Tensor Programs series of papers so far has proven a wide range of results, which typically have the characteristic of architectural universality, i.e., covering any neural architectures. But all of such results have assumed Gaussian weight initialization. Now, armed with Theorem 2, we show the same results hold with non-Gaussian weight initialization as well, under mild assumptions, thus extending them to other prevalent initializations such as uniform and truncated Gaussian. This demonstrates these results to be distributional universal, i.e., independent of specifics of weight sampling. Theorem 2 here acts like a drop-in replacement for Theorem 1 in their proofs, except we need to a) add more smoothness assumptions on nonlinearities, and b) any almost sure convergence is weakened to convergence in probability.

Our Theorem 2 can be applied to a very general class of neural networks as given by the following:

Setup 3. Consider a neural network whose forward pass can be expressed as Eq. (1) where  $T_{m} = 1$  for all  $m$ . Suppose 1) all the activation functions are polynomially smooth; 2) a bias vector of the output layer is initialized with zeros; 3) bias vectors of all other layers are initialized with elementwise images of iid Gaussian vectors under polynomially smooth nonlinearities; 4) weights of the input layer are initialized with elementwise images of iid Gaussian matrices of appropriate sizes under polynomially smooth nonlinearities; 5) weights of the output layer are initialized with iid Gaussians with zero mean and variance inversely proportional to the number of input neurons of this layer; 6) weights of any other layer are initialized iid from a distribution with zero mean and variance inversely proportional to the number of input neurons of this layer, with all higher moments existing.

NNGP Correspondence. The following result is an immediate consequence of our Theorem 2 and Proposition G.4 of [21]:

Corollary 1. Consider Setup 3. At initialization, as width tends to infinity, the neural network function converges weakly to a Gaussian Process (GP). Moreover, the empirical kernel associated to this network converges pointwise in probability to the kernel of this GP.

NTK correspondence. We can directly plug our Theorem 2 into the proof of [26] and obtain

Corollary 2. Consider Setup 3 and assume NTK parameterization. Then under SGD weight updates, NTK of the network at any optimization step converges pointwise in probability to a finite deterministic limit that does not depend on the timestep. Moreover, the output of the network at any timestep converges pointwise to the output of a kernel method with this limit as a kernel.

In Appendix C, we present a variant of Theorem 2 which works for non-smooth but Lipschitz nonlinearities (like ReLU or MaxPool). It allows to generalize Corollary 1 to ReLU nets, but not Corollary 2 (that's because a Tensor Program expressing the backward pass of a ReLU net has ReLU derivatives, which are not even continuous, as nonlinearities).

Random Matrix Theory. Our Theorem 2 implies the semi-circle law for non-Gaussian Wigner matrices, the Marchenko-Pastur law for  $AA^{\top}$ , where  $A$  is non-Gaussian, and Free Independence Principle for Tensor Programs with non-Gaussian initial weights, thus generalizing TP3; we discuss these results in Appendix B.

Classification of Infinite-Width Limits. Consider now an  $L$ -hidden-layer perceptron trained using stochastic gradient descent (SGD). TP4 [24] proposes the notion of an abc-parameterization, which is specified by a set of numbers  $\{a_l, b_l\}_l \cup \{c\}$ : (1) each weight is parameterized as  $W^l = n^{-a_l}w^l$  for the actual trainable parameter  $w^l$ , (2)  $w_{\alpha \beta}^l \sim \mathcal{N}(0,n^{-2b_l})$  at initialization, and (3) the SGD learning rate is taken as  $\eta n^{-c}$  for some constant  $\eta$ .

When  $c$  is too small (i.e. learning rate is too large), SGD can lead to blowup of logits; such a parameterization is called unstable. When  $c$  is too large, the function computed by the network does not change with time; such a parameterization is called trivial. The result below follows from Theorem 1:

Corollary 3 (Dynamical dichotomy; [24]). Any nontrivial stable abc-parameterization yields a (discrete-time) infinite-width limit. This limit either (1) allows the last layer embedding to evolve nontrivially, or (2) is described by kernel gradient descent in function space, but not both.

The first situation is called feature learning, as opposed to the second one, called kernel learning. Feature learning is argued to be a more sensible learning regime for a neural network.

In TP4, [24] proposed the so-called  $\mu$ -parameterization, which uniquely exhibits maximal feature learning in the infinite-width limit among all abc-parameterizations. In TP5, [25] generalized this  $\mu$ -parameterization to the Adam optimizer [7] and a wider class of architectures and used it to facilitate hyperparameter tuning of large models.

Since Dynamical dichotomy (Corollary 3) is a direct consequence of convergence of averages given by Theorem 1, we can readily apply our Theorem 2 for the same purpose:

Corollary 4 (Dynamical dichotomy; ours). Consider an  $L$ -hidden-layer perceptron trained using stochastic gradient descent (SGD) under abc-parameterization, but with  $w_{l}$  sampled independently from a distribution with zero mean, variance  $n^{-2b_l}$ , and all higher moments existing, for  $l = 2,\dots ,L$ . Then the statement of Corollary 3 holds.

The  $\mu$ -parameterization in this case is the same, and can be recovered by the same argument as in TP4. We therefore theoretically justify the hyperparameter tuning technique used in TP5 for neural networks with non-Gaussian initializations commonly used in practice.

On potential societal impacts. Our work concerns generic behavior of neural nets in the limit of infinite width and therefore does not provide any foreseen societal impacts. The only direct practical application of our work we are aware of is theoretical justification for the hyperparameter tuning method of TP5 [25] for non-Gaussian weight initializations. However, what our work provides is merely justification for this method, while the method itself existed before our work (and was well-justified for Gaussian weight initializations).

# 4 Proof of Theorem 2

Denote the matrices, the vectors, and the scalars of our non-Gaussian Tensor Program as  $\tilde{A}^{M_0 + 1},\ldots ,\tilde{A}^M,\tilde{g}^1,\ldots ,\tilde{g}^M$  , and  $\tilde{c}^1,\dots ,\tilde{c}^M$  , respectively. Consider the same Tensor Program with Gaussian weights with zero mean and the same variance  $n^{-1}$  Denote its matrices as  $A^{M_0 + 1},\ldots ,A^M$  vectors as  $g^{1},\ldots ,g^{M}$  , and scalars as  $c^1,\ldots ,c^M$  . Note that initial vectors and scalars of both programs coincide, i.e.  $g^{m} = \tilde{g}^{m}$  and  $c^{m} = \tilde{c}^{m}\forall m\in [M_{0}]$

For the sake of brevity, we will denote all the nonlinearities  $\phi^m$  by the same letter  $\phi$  in the sequel, ignoring the fact that they can be different functions with different number of arguments.

Limitations of the proof technique of Theorem 1. Let us first review the idea of Theorem 1 proof, as given in [23]. For simplicity, let us ignore the scalars in the program and assume that each vector is generated with a fresh new matrix each time (i.e. no weights are shared).

Theorem 1 is then proven by induction on  $m$ . For  $m = M_0$ , for any polynomially bounded  $\phi, \frac{1}{n}\sum_{\alpha}\phi(g_{\alpha}^{1},\ldots,g_{\alpha}^{M_{0}}) \xrightarrow{\text{a.s.}} \mathbb{E}_{Z \sim \mathcal{N}(\mu^{in},\Sigma^{in})}\phi(Z)$  by Law of Large Numbers (LLN) since  $[g_{\alpha}^{1},\ldots,g_{\alpha}^{M_{0}}] \sim \mathcal{N}(\mu^{in},\Sigma^{in})$  independently for each coordinate  $\alpha \in [n]$ .

Suppose  $\frac{1}{n}\sum_{\beta}\phi (g_{\beta}^{1},\ldots ,g_{\beta}^{m - 1})\stackrel {\mathrm{a.s.}}{\longrightarrow}\mathring{\Psi}_{\phi}$  for some  $\mathring{\Psi}_{\phi}$ , and any polynomially bounded  $\phi$ . Conditioned on all initial vectors and all previous matrices  $A^{M_0 + 1},\dots ,A^{m - 1}$ ,  $g_{\alpha}^{m} = \sum_{\beta}A_{\alpha \beta}^{m}\phi (g_{\beta}^{1},\dots ,g_{\beta}^{m - 1})$  is Gaussian as a linear combination of Gaussians. Variance of this Gaussian is  $\frac{1}{n}\sum_{\beta}\phi^2 (g_\beta^1,\dots ,g_\beta^{m - 1})$ , which converges almost surely to a deterministic limit  $\mathring{\Psi}_{\phi^2}$  by induction hypothesis. Therefore  $g_{\alpha}^{m}$  converges almost surely to a Gaussian independent on  $g^{1},\ldots ,g^{M_0}$  and  $A^{M_0 + 1},\ldots ,A^{m - 1}$ . Of course,  $\frac{1}{n}\sum_{\beta}\phi (g_{\beta}^{1},\dots ,g_{\beta}^{m})\stackrel {\mathrm{a.s.}}{\longrightarrow}\mathring{\Psi}_{\phi}$  for some (new)  $\mathring{\Psi}_{\phi}$ , and any polynomially bounded  $\phi$  then.

If we relax our assumption that no weights are shared,  $g^{1}, \ldots, g^{m-1}$  may depend on  $A^{m}$  in the induction step. This situation is treated using a Gaussian conditioning argument, which we do not discuss here. Note that in this case,  $g_{\alpha}^{m}$  may depend on  $g_{\alpha}^{1}, \ldots, g_{\alpha}^{m-1}$ .

Unfortunately, for non-Gaussian weights  $\tilde{A}$ , the same argument cannot be applied readily. Indeed,  $\tilde{g}_{\alpha}^{m} = \sum_{\beta} \tilde{A}_{\alpha \beta}^{m} \phi(\tilde{g}_{\beta}^{1}, \ldots, \tilde{g}_{\beta}^{m-1})$  is a sum of dependent non-Gaussian random variables, it does not generally have a closed-form expression as a function of  $\frac{1}{n} \sum_{\beta} \phi(\tilde{g}_{\beta}^{1}, \ldots, \tilde{g}_{\beta}^{m-1})$  for some  $\phi$ .

We therefore apply a different argument. The idea is to interpolate the weights from the Gaussian ones, for which convergence of  $\frac{1}{n}\sum_{\alpha}\psi(g_{\alpha}^{1},\ldots,g_{\alpha}^{M})$  is given by Theorem 1, to the non-Gaussian ones, and show that the above average does not change along this interpolation in the limit of large  $n$ .

Our proof. [1] proved a result similar to our Theorem 2 for approximate message passing; see Appendix E for comparison. Their argument was based on the following weight interpolation:  $A^m(t) = \sqrt{1 - t}A^m + \sqrt{t}\tilde{A}^m \forall m \in [M_0 + 1 : M]$ . We are going to use the same interpolation technique. Denote the vectors corresponding to  $A^{M_0 + 1}(t), \ldots, \tilde{A}^M(t)$  as  $g^1(t), \ldots, g^M(t)$  and denote the corresponding scalars as  $c^1(t), \ldots, c^M(t)$ . Define

$$
\Psi_ {n} (t) = \frac {1}{n} \sum_ {\alpha = 1} ^ {n} \psi \left(g _ {\alpha} ^ {1} (t), \dots , g _ {\alpha} ^ {M} (t), c ^ {1} (t), \dots , c ^ {M} (t)\right). \tag {4}
$$

We are going to prove the following proposition:

Proposition 1.  $\forall \chi \in Lip(\mathbb{R})\cap C^1 (\mathbb{R})$

$$
\sup  _ {s \in [ 0, 1 / \sqrt {2} ]} \left| \partial_ {s} \mathbb {E} \left[ \chi \left(\Psi_ {n} \left(s ^ {2}\right)\right) \right] \right| = O \left(n ^ {- 1 / 2}\right), \quad \sup  _ {s \in [ 0, 1 / \sqrt {2} ]} \left| \partial_ {s} \mathbb {E} \left[ \chi \left(\Psi_ {n} \left(1 - s ^ {2}\right)\right) \right] \right| = O \left(n ^ {- 1 / 2}\right). \tag {5}
$$

Once it is proven, we get by dominated convergence,

$$
\begin{array}{l} \limsup_{n\to \infty}\left| \mathbb{E}\left[ \chi (\Psi_{n}(1 / 2))\right] - \mathbb{E}\left[ \chi (\Psi_{n}(0))\right]\right| = \\ = \operatorname * {l i m s u p} _ {n \to \infty} \left| \int_ {0} ^ {\frac {1}{\sqrt {2}}} \partial_ {s} \mathbb {E} \left[ \chi (\Psi_ {n} (s ^ {2})) \right] d s \right| = \int_ {0} ^ {\frac {1}{\sqrt {2}}} \operatorname * {l i m} _ {n \to \infty} \left| \partial_ {s} \mathbb {E} \left[ \chi (\Psi_ {n} (s ^ {2})) \right] \right| d s = 0. (6) \\ \end{array}
$$

Therefore the limit exists and equals to zero. By a similar argument, we get  $\lim_{n\to \infty}|\mathbb{E}\left[\chi (\Psi_n(1))\right] - \mathbb{E}\left[\chi (\Psi_n(1 / 2))\right]| = 0$ . Hence  $\lim_{n\to \infty}|\mathbb{E}\left[\chi (\Psi_n(1))\right] - \mathbb{E}\left[\chi (\Psi_n(0))\right]| = 0$ .

Note that Theorem 1 implies weak convergence of  $\Psi (0)$

$$
\lim  _ {n \rightarrow \infty} \mathbb {E} [ \chi (\Psi_ {n} (0)) ] = \chi (\stackrel {\circ} {\Psi}) \quad \forall \chi \in L i p _ {b} (\mathbb {R}) \cap C ^ {1} (\mathbb {R}). \tag {7}
$$

The above then means that  $\Psi_n(1) = \tilde{\Psi}_n = \frac{1}{n}\sum_{\alpha=1}^n \psi(\tilde{g}_\alpha^1, \ldots, \tilde{g}_\alpha^M, \tilde{c}^1, \ldots, \tilde{c}^M)$  converges weakly to the same quantity,  $\tilde{\Psi}$ . Since the weak limit is deterministic, weak convergence further upgrades to convergence in probability.

Similarly, if all  $\phi$  are linearly bounded and  $\psi$  is quadratically bounded, Theorem 1 gives convergence of expectations:  $\lim_{n\to \infty}\mathbb{E}[\Psi_n(0)] = \mathring{\Psi}$ . From the previous, we have  $\lim_{n\to \infty}|\mathbb{E}[\Psi_n(1)] - \mathbb{E}[\Psi_n(0)]| = 0$  by taking  $\chi$  to be identity. Therefore  $\lim_{n\to \infty}\mathbb{E}[\tilde{\Psi}_n] = \lim_{n\to \infty}\mathbb{E}[\Psi_n(1)] = \mathring{\Psi}$ .

Proof of Proposition 1. We need the following lemma in order to validate swapping the  $s$ -derivative and expectation:

Lemma 1.  $\forall \chi \in Lip(\mathbb{R})\cap C^1 (\mathbb{R})\forall n\geq 1$

$$
\mathbb {E} \left(\sup  _ {s \in [ 0, 1 / \sqrt {2} ]} \left| \partial_ {s} \left[ \chi \left(\Psi_ {n} \left(s ^ {2}\right)\right) \right] \right|\right) <   \infty , \quad \mathbb {E} \left(\sup  _ {s \in [ 0, 1 / \sqrt {2} ]} \left| \partial_ {s} \left[ \chi \left(\Psi_ {n} \left(1 - s ^ {2}\right)\right) \right] \right|\right) <   \infty . \tag {8}
$$

For  $s \in [0, 1 / \sqrt{2}]$ , we apply Cauchy-Schwarz and get

$$
\left| \partial_ {s} \mathbb {E} \left[ \chi (\Psi_ {n} (s ^ {2})) \right] \right| = \left| \mathbb {E} \left[ \chi^ {\prime} (\Psi_ {n} (s ^ {2})) \partial_ {s} \Psi_ {n} (s ^ {2}) \right] \right| \leq \sqrt {\mathbb {E} \left([ \chi^ {\prime} (\Psi_ {n} (s ^ {2})) ] ^ {2}\right) \mathbb {E} \left([ \partial_ {s} \Psi_ {n} (s ^ {2}) ] ^ {2}\right)}. \quad (9)
$$

Similarly,  $\left| \partial_s \mathbb{E} \left[ \chi(\Psi_n(1 - s^2)) \right] \right| \leq \sqrt{\mathbb{E} \left( [\chi'(\Psi_n(1 - s^2))]^2 \right) \mathbb{E} \left( [\partial_s \Psi_n(1 - s^2)]^2 \right)}.$

Since  $\chi$  is Lipschitz and continuously differentiable,  $\mathbb{E}\left([ \chi' (\Psi_n(s^2)) ]^2\right)$  and  $\mathbb{E}\left([ \chi' (\Psi_n(1 - s^2)) ]^2\right)$  are uniformly bounded on  $s \in [0, 1 / \sqrt{2}]$ . We are going to prove the following lemma below:

Lemma 2.  $\sup_{s\in [0,1 / \sqrt{2} ]}\mathbb{E}\left([ \partial_s\Psi (s^2)]^2\right) = O(n^{-1})$  , and

$\sup_{s\in [0,1 / \sqrt{2} ]}\mathbb{E}\left([ \partial_s\Psi (1 - s^2)]^2\right) = O(n^{-1}).$

Once it is proven, the statement of Proposition 1 follows. Indeed,

$$
\sup  _ {s \in [ 0, 1 / \sqrt {2} ]} \left| \partial_ {s} \mathbb {E} \left[ \chi \left(\Psi_ {n} \left(s ^ {2}\right)\right) \right] \right| \leq \sup  _ {s \in [ 0, 1 / \sqrt {2} ]} \sqrt {\mathbb {E} \left([ \chi^ {\prime} \left(\Psi_ {n} \left(s ^ {2}\right)\right) ] ^ {2}\right) \mathbb {E} \left([ \partial_ {s} \Psi_ {n} \left(s ^ {2}\right) ] ^ {2}\right)} = O \left(n ^ {- 1 / 2}\right). \tag {10}
$$

The asymptotics of the second limit is given by a similar argument.

# 4.1 Proof of Lemma 2 in a simple scenario

To make our arguments clear, we illustrate the proof of Lemma 2 on the simplest non-trivial case of  $M_0 = 1$ ,  $M = 2$ , with no dependence on  $c^1$  or  $c^2$ :  $\Psi_n(t) = \frac{1}{n} \sum_{\alpha} \psi(g_\alpha^1, g_\alpha^2(t))$ ,  $g^2(t) = A(t) \phi(g^1)$ ,  $g^1 \sim \mathcal{N}(0, I)$ . The full proofs of Lemmas 1 and 2 can be found in Appendix F.

Consider the exact expression of  $\partial_s\Psi_n(s^2)$

$$
\partial_ {s} \Psi_ {n} \left(s ^ {2}\right) = \frac {1}{n} \sum_ {\alpha} \partial_ {2} \psi \left(g _ {\alpha} ^ {1}, \sum_ {\beta} A _ {\alpha \beta} \left(s ^ {2}\right) \phi \left(g _ {\beta} ^ {1}\right)\right) \sum_ {\beta} \partial_ {s} A _ {\alpha \beta} \left(s ^ {2}\right) \phi \left(g _ {\beta} ^ {1}\right); \tag {11}
$$

the exact expression of  $\partial_s\Psi_n(1 - s^2)$  is similar. For any finite  $n$ , Lemma 1 follows from the facts that  $\sup_{s\in [0,1]}|a(s^2)|\leq |a| + |\tilde{a} |,\sup_{s\in [0,1 / \sqrt{2} ]}|\partial_s a(s^2)|\leq |a| + |\tilde{a} |$ , initial vectors are Gaussian, initial scalars have all moments, and  $\phi$  and  $\psi$  are polynomially smooth. We do not present the full proof of Lemma 1 in the main due to space constraints.

Lemma 2 follows from the exact expression of  $\partial_s\Psi_n(s^2)$  and  $\partial_s\Psi_n(1 - s^2)$  too:

$$
\left[ \partial_ {s} \Psi_ {n} \left(s ^ {2}\right) \right] ^ {2} = \frac {1}{n ^ {2}} \sum_ {\alpha , \beta} \sum_ {\tilde {\alpha}, \tilde {\beta}} \partial_ {2} \psi \left(g _ {\alpha} ^ {1}, g _ {\alpha} ^ {2} \left(s ^ {2}\right)\right) \partial_ {2} \psi \left(g _ {\tilde {\alpha}} ^ {1}, g _ {\tilde {\alpha}} ^ {2} \left(s ^ {2}\right)\right) \phi \left(g _ {\tilde {\beta}} ^ {1}\right) \phi \left(g _ {\tilde {\beta}} ^ {1}\right) \partial_ {s} A _ {\alpha \beta} \left(s ^ {2}\right) \partial_ {s} A _ {\tilde {\alpha} \tilde {\beta}} \left(s ^ {2}\right). \tag {12}
$$

Since  $g^2(s^2) = A(s^2)\phi(g^1)$  is a function of weights, we can re-write the above expression as

$$
\left[ \partial_ {s} \Psi_ {n} \left(s ^ {2}\right) \right] ^ {2} = \frac {1}{n ^ {2}} \sum_ {\alpha , \beta} \sum_ {\tilde {\alpha}, \tilde {\beta}} f _ {\alpha , \beta , \tilde {\alpha}, \tilde {\beta}} \left(A \left(s ^ {2}\right)\right) \partial_ {s} A _ {\alpha \beta} \left(s ^ {2}\right) \partial_ {s} A _ {\tilde {\alpha} \tilde {\beta}} \left(s ^ {2}\right), \tag {13}
$$

where  $f_{\alpha ,\beta ,\tilde{\alpha},\tilde{\beta}}(A) = \partial_2\psi \left(g_\alpha^1,(A\phi (g^1))_\alpha\right)\partial_2\psi \left(g_\tilde{\alpha}^1,(A\phi (g^1))_\tilde{\alpha}\right)\phi (g_\beta^1)\phi (g_\tilde{\beta}^1).$

Since  $\mathbb{E}\left[\partial_sA_{\alpha \beta}(s^2)\partial_sA_{\tilde{\alpha}\tilde{\beta}}(s^2)\right] = 0$  if  $\alpha \neq \tilde{\alpha}$  or  $\beta \neq \tilde{\beta}$  but it is not zero otherwise, it is necessary to study these two cases separately. Let  $[\partial_s\Psi_n(s^2)]^2 = ([\partial_s\Psi_n(s^2)]^2)_{1,1} + ([\partial_s\Psi_n(s^2)]^2)_2$ , where

$$
\left(\left[ \partial_ {s} \Psi_ {n} \left(s ^ {2}\right) \right] ^ {2}\right) _ {2} = \frac {1}{n ^ {2}} \sum_ {\alpha , \beta} f _ {\alpha , \beta , \alpha , \beta} (A \left(s ^ {2}\right)) \left[ \partial_ {s} A _ {\alpha \beta} \left(s ^ {2}\right) \right] ^ {2}, \tag {14}
$$

238 and

$$
\left(\left[ \partial_ {s} \Psi_ {n} \left(s ^ {2}\right) \right] ^ {2}\right) _ {1, 1} = \frac {1}{n ^ {2}} \sum_ {\substack {\alpha , \beta , \tilde {\alpha}, \tilde {\beta}: \\ \alpha \neq \tilde {\alpha} \text {or} \beta \neq \tilde {\beta}}} f _ {\alpha , \beta , \tilde {\alpha}, \tilde {\beta}} \left(A \left(s ^ {2}\right)\right) \partial_ {s} A _ {\alpha \beta} \left(s ^ {2}\right) \partial_ {s} A _ {\tilde {\alpha} \tilde {\beta}} \left(s ^ {2}\right). \tag{15}
$$

We can write the sums over indices as sums over sets  $\Gamma^{(2)} = \{(\alpha, \beta, \tilde{\alpha}, \tilde{\beta}) : \alpha = \tilde{\alpha}, \beta = \tilde{\beta}\}$  and  $\Gamma^{(1,1)} = \{(\alpha, \beta, \tilde{\alpha}, \tilde{\beta}) : \alpha \neq \tilde{\alpha} \text{ or } \beta \neq \tilde{\beta}\}$ . For every configuration  $(\alpha, \beta, \tilde{\alpha}, \tilde{\beta}) \in \Gamma^{(1,1)}$ , enumerate the set of all (distinct) weights  $\{A_{\alpha\beta}\}_{\alpha, \beta=1}^n$  in such a way that  $a_1 = A_{\alpha\beta}$  and  $a_2 = A_{\tilde{\alpha}\tilde{\beta}}$ ; the order of the rest can be arbitrary. Similarly, for every configuration  $(\alpha, \beta, \tilde{\alpha}, \tilde{\beta}) \in \Gamma^{(2)}$ , enumerate the set of all weights  $\{A_{\alpha\beta}\}_{\alpha, \beta=1}^n$  in such a way that  $a_1 = A_{\alpha\beta} = A_{\tilde{\alpha}\tilde{\beta}}$ . Recall  $N$  is the number of all distinct weights; in our illustrative example,  $N = n^2$ . The above expressions take the following form:

$$
\left(\left[ \partial_ {s} \Psi_ {n} \left(s ^ {2}\right) \right] ^ {2}\right) _ {2} = \frac {1}{n ^ {2}} \sum_ {u \in \Gamma^ {(2)}} f _ {u} \left(a _ {1} \left(s ^ {2}\right), \dots , a _ {N} \left(s ^ {2}\right)\right) \left[ \partial_ {s} a _ {1} \left(s ^ {2}\right) \right] ^ {2}, \tag {16}
$$

245 and

$$
\left(\left[ \partial_ {s} \Psi_ {n} \left(s ^ {2}\right) \right] ^ {2}\right) _ {1, 1} = \frac {1}{n ^ {2}} \sum_ {u \in \Gamma^ {(1, 1)}} f _ {u} \left(a _ {1} \left(s ^ {2}\right), \dots , a _ {N} \left(s ^ {2}\right)\right) \partial_ {s} a _ {1} \left(s ^ {2}\right) \partial_ {s} a _ {2} \left(s ^ {2}\right). \tag {17}
$$

Since we are interested in  $\mathbb{E}\left([ \partial_s \Psi_n(s^2) ]^2\right) \leq \mathbb{E}\left([ \partial_s \Psi_n(s^2) ]^2\right)_2 + \mathbb{E}\left([ \partial_s \Psi_n(s^2) ]^2\right)_{1,1}$ , we need upper-bounds for the following expectations:  
248  
uniform on  $s \in [0,1/\sqrt{2}]$ . In our simple example, cardinalities of Gamma-terms can be easily bounded using their definitions:  $\left|\Gamma^{(2)}\right| \leq n^2$  and  $\left|\Gamma^{(1,1)}\right| \leq n^4$ .  
Let us bound  $\sup_{u\in \Gamma^{(1,1)}}\left|\mathbb{E}\left(f_u(a_1(s^2),\ldots ,a_N(s^2))\partial_s a_1(s^2)\partial_s a_2(s^2)\right)\right|$ . We start with Taylor-expanding  $f_{u}$  wrt the first two arguments up to some order  $K$  to be specified later:

$$
\begin{array}{l} f _ {u} \left(x _ {1}, \dots , x _ {N}\right) = \sum_ {k = 0} ^ {K} \frac {1}{k !} \sum_ {j _ {1}, j _ {2} \geq 0, j _ {1} + j _ {2} = k} \left[ x _ {1} \right] ^ {j _ {1}} \left[ x _ {2} \right] ^ {j _ {2}} \times \partial_ {1} ^ {j _ {1}} \partial_ {2} ^ {j _ {2}} f _ {u} \left(0, 0, x _ {3}, \dots , x _ {N}\right) + \\ + \frac {1}{K !} \sum_ {j _ {1}, j _ {2} \geq 0, j _ {1} + j _ {2} = K + 1} [ x _ {1} ] ^ {j _ {1}} [ x _ {2} ] ^ {j _ {2}} \times \int_ {0} ^ {1} \partial_ {1} ^ {j _ {1}} \partial_ {2} ^ {j _ {2}} f _ {u} (\xi x _ {1}, \xi x _ {2}, x _ {3}, \dots , x _ {N}) d \xi . \tag {20} \\ \end{array}
$$

The expectation we are interested in therefore takes the following form:

$$
\begin{array}{l} \mathbb {E} \left[ f _ {u} \left(a _ {1}, \dots , a _ {N}\right) \left[ \partial_ {s} a _ {1} \right] \left[ \partial_ {s} a _ {2} \right] \right] = \\ = \sum_{k = 0}^{K}\frac{1}{k!}\sum_{\substack{j_{1},j_{2}\geq 0,\\ j_{1} + j_{2} = k}}\mathbb{E}\left[[\partial_{s}a_{1}][\partial_{s}a_{2}][a_{1}]^{j_{1}}[a_{2}]^{j_{2}}\right]\times \mathbb{E}\left[\partial_{1}^{j_{1}}\partial_{2}^{j_{2}}f_{u}(0,0,a_{3},\ldots ,a_{N})\right] + \\ + \frac {1}{K !} \sum_ {\substack {j _ {1}, j _ {2} \geq 0, \\ j _ {1} + j _ {2} = K + 1}} \mathbb {E} \left[ \left[ \partial_ {s} a _ {1} \right] \left[ \partial_ {s} a _ {2} \right] \left[ a _ {1} \right] ^ {j _ {1}} \left[ a _ {2} \right] ^ {j _ {2}} \times \int_ {0} ^ {1} \partial_ {1} ^ {j _ {1}} \partial_ {2} ^ {j _ {2}} f _ {u} (\xi a _ {1}, \xi a _ {2}, a _ {3}, \dots , a _ {N}) d \xi \right], \tag{21} \\ \end{array}
$$

where we have omitted  $s^2$  in the arguments of  $a$ 's for brevity.

We split the expectation of the remainder term using Cauchy-Schwarz:

$$
\begin{array}{l} \left| \mathbb {E} \left[ [ \partial_ {s} a _ {1} ] [ \partial_ {s} a _ {2} ] [ a _ {1} ] ^ {j _ {1}} [ a _ {2} ] ^ {j _ {2}} \times \int_ {0} ^ {1} \partial_ {1} ^ {j _ {1}} \partial_ {2} ^ {j _ {2}} f _ {u} (\xi a _ {1}, \xi a _ {2}, a _ {3}, \dots , a _ {N}) d \xi \right] \right| \leq \\ \leq \sqrt {\mathbb {E} \left[ \left(\left[ \partial_ {s} a _ {1} \right] \left[ \partial_ {s} a _ {2} \right] \left[ a _ {1} \right] ^ {j _ {1}} \left[ a _ {2} \right] ^ {j _ {2}}\right) ^ {2} \right] \times \mathbb {E} \left[ \left(\int_ {0} ^ {1} \partial_ {1} ^ {j _ {1}} \partial_ {2} ^ {j _ {2}} f _ {u} (\xi a _ {1} , \xi a _ {2} , a _ {3} , \dots , a _ {N}) d \xi\right) ^ {2} \right]}. \tag {22} \\ \end{array}
$$

We claim that  $\mathbb{E}\left[\partial_1^{j_1}\partial_2^{j_2}f_u(0,0,a_3,\ldots ,a_N)\right]$  and  $\mathbb{E}\left[\left(\int_0^1\partial_1^{j_1}\partial_2^{j_2}f_u(\xi a_1,\xi a_2,a_3,\ldots ,a_N)d\xi\right)^2\right]$  are bounded uniformly over  $u\in \Gamma^{(1,1)}$ $\xi \in [0,1]$ , and  $s\in [0,1 / \sqrt{2} ]$ . This claim follows from Lemma 9, the main technical lemma of the present work, proven in Appendix H.4.  
We are left to bounding  $\mathbb{E}\left[[\partial_s a_1][\partial_s a_2][a_1]^{j_1}[a_2]^{j_2}\right]$  and  $\mathbb{E}\left[\left([\partial_s a_1][\partial_s a_2][a_1]^{j_1}[a_2]^{j_2}\right)^2\right]$  uniformly on  $s \in [0, 1/\sqrt{2}]$ . We prove the following Lemma in Appendix G.1:  
Lemma 3. Let  $a$  and  $\tilde{a}$  be independent random variables with zero mean, variance  $n^{-1}$ , and all higher moments existing. Define  $a(t) = \sqrt{1 - t} a + \sqrt{t}\tilde{a}$ . The following holds:

$$
\begin{array}{l} I. \mathbb {E} \left[ \partial_ {s} a (s ^ {2}) \right] = \mathbb {E} \left[ \partial_ {s} a (1 - s ^ {2}) \right] = 0 \forall s \in [ 0, 1); \\ 2. \mathbb {E} [ a (s ^ {2}) \partial_ {s} a (s ^ {2}) ] = \mathbb {E} [ a (1 - s ^ {2}) \partial_ {s} a (1 - s ^ {2}) ] = 0 \forall s \in [ 0, 1); \\ 3. \sup  _ {s \in [ 0, 1 / \sqrt {2} ]} \left(\mathbb {E} \left[ \left| [ a (s ^ {2}) ] ^ {j} \partial_ {s} a (s ^ {2}) \right| ^ {k} \right]\right) = O \left(n ^ {- (j + 1) k / 2}\right), a n d \\ \sup _ {s \in [ 0, 1 / \sqrt {2} ]} \left(\mathbb {E} \left[ \left| \left[ a (1 - s ^ {2}) \right] ^ {j} \partial_ {s} a (1 - s ^ {2}) \right| ^ {k} \right]\right) = O \left(n ^ {- (j + 1) k / 2}\right) \forall j \geq 0 \forall k \geq 0. \\ \end{array}
$$

This gives  $\mathbb{E}\left[[\partial_s a_1][\partial_s a_2][a_1]^{j_1}[a_2]^{j_2}\right] = 1_{j_1 \geq 2} 1_{j_2 \geq 2} O\left(n^{-(2 + j_1 + j_2) / 2}\right)$  and  $\mathbb{E}\left[\left([ \partial_s a_1 ] [ \partial_s a_2 ] [ a_1 ] ^ {j_1 } [ a_2 ] ^ {j_2 } \right)^2 \right] = O\left(n^{-(2 + j_1 + j_2)}\right)$ , both uniformly over  $s \in [0, 1 / \sqrt{2}]$ . This implies that the first four terms in our Taylor expansion zero-out:

$$
\sum_ {\substack {j _ {1}, j _ {2} \geq 0, \\ j _ {1} + j _ {2} = k}} \mathbb {E} \left[ \left[ \partial_ {s} a _ {1} \right] \left[ \partial_ {s} a _ {2} \right] \left[ a _ {1} \right] ^ {j _ {1}} \left[ a _ {2} \right] ^ {j _ {2}} \right] \times \mathbb {E} \left[ \partial_ {1} ^ {j _ {1}} \partial_ {2} ^ {j _ {2}} f _ {u} (0, 0, a _ {3}, \dots , a _ {N}) \right] = 1 _ {k \geq 4} O \left(n ^ {- (2 + k) / 2}\right) \tag{23}
$$

uniformly over  $s \in [0, 1 / \sqrt{2}]$ . Taking  $K = 3$ , we get:

$$
\begin{array}{l} \mathbb {E} \left[ f _ {u} \left(a _ {1}, \dots , a _ {N}\right) \left[ \partial_ {s} a _ {1} \right] \left[ \partial_ {s} a _ {2} \right] \right] \leq \\ \leq \frac{1}{3!}\sum_{\substack{j_{1},j_{2}\geq 0,\\ j_{1} + j_{2} = 4}}\sqrt{\mathbb{E}\left[(\left[\partial_{s}a_{1}\right][\partial_{s}a_{2}]\left[a_{1}\right]^{j_{1}}\left[a_{2}\right]^{j_{2}})^{2}\right]}\\ \times \mathbb{E}\left[\left(\int_{0}^{1}\partial_{1}^{j_{1}}\partial_{2}^{j_{2}}f_{u}(\xi a_{1},\xi a_{2},a_{3:N})  d\xi\right)^{2}\right] = \\ = O \left(n ^ {- (2 + 4) / 2}\right) = O \left(n ^ {- 3}\right) \tag {24} \\ \end{array}
$$

uniformly over  $u \in \Gamma^{(1,1)}$  and  $s \in [0,1/\sqrt{2}]$ . From Eq. (19),  $\mathbb{E}\left(\left([ \partial_s \Psi_n(s^2) ]^2\right)_{1,1}\right) \leq n^{-2} \left| \Gamma^{(1,1)} \right| O(n^{-3}) = O(n^{-1})$  uniformly over  $s \in [0,1/\sqrt{2}]$ .

In order to upper-bound  $\mathbb{E}\left[f_u(a_1,\dots ,a_N)[\partial_s a_1]^2\right]$  in our simple case, it suffices to directly apply Cauchy-Schwarz:

$$
\mathbb {E} \left[ f _ {u} \left(a _ {1}, \dots , a _ {N}\right) \left[ \partial_ {s} a _ {1} \right] ^ {2} \right] \leq \sqrt {\mathbb {E} \left(\left[ \partial_ {s} a _ {1} \right] ^ {4}\right) \times \mathbb {E} \left(\left[ f _ {u} \left(a _ {1} , \dots , a _ {N}\right) \right] ^ {2}\right)} = O (n ^ {- 1}) \tag {25}
$$

uniformly over  $u \in \Gamma^{(2)}$  and  $s \in [0,1/\sqrt{2}]$ . From Eq. (18),  $\mathbb{E}\left(\left[\partial_s\Psi_n(s^2)\right]^2\right)_2 \leq n^{-2}\left|\Gamma^{(2)}\right|O(n^{-1}) = O(n^{-1})$  uniformly over  $s \in [0,1/\sqrt{2}]$ . This gives  $\mathbb{E}\left(\left[\partial_s\Psi_n(s^2)\right]^2\right) = O(n^{-1})$  uniformly over  $s \in [0,1/\sqrt{2}]$ .

# 5 Related works

The Tensor Programs series discusses different applications of the (Gaussian) Master theorem proven by [20]. They include: Gaussian process behavior at initialization (TP1, [21]), convergence to a kernel method (TP2, [22]), Free Independence Principle (TP3, [23]), dynamical dichotomy and  $\mu$ -parameterization (TP4, [24]), and finally application of  $\mu$ P to hyperparameter tuning (TP5, [25]).

Neural networks converge to Gaussian processes as their width goes to infinity, as was proven by [8, 9] for fully-connected nets, and by [11, 2] for convolutional nets; see also [5]. Using the Master theorem, [21] showed that this behavior holds for a very wide class of architectures, including not only convolutional, but also graph convolutional and recurrent neural nets, ResNets, networks with batch normalization, and networks with attention.

The seminal work of [6] demonstrated that under certain parameterization, the learning dynamics a neural net converges to that of a kernel method. The corresponding kernel was called Neural Tangent Kernel, or NTK, and drawn a lot of attention in recent years. While the result of [6] was proven only for fully-connected nets with smooth activations, the Master theorem allows to generalize this result for a wider class of architectures (the same as mentioned above), see [22].

[16, 15, 19] and others studied trainability of very deep and wide neural networks using random matrix theory. Their analysis crucially relied on the assumption that hidden representations of a neural network at initialization were freely independent from the weights. TP3 [23] was among the first works to validate this assumption rigorously; see also [12].

Infinite-width behavior of a neural net depends on scaling of its hyperparameters (like initial weights variance and learning rate) with width. Dynamical dichotomy proposed in TP4 [24] is a classification of scalings that are meaningful in a certain sense. Another classification of scalings with a different notion of meaningfulness was proposed earlier by [3, 4], but only for two-layered networks.

A distribution universality property similar to our Theorem 2 was shown by [1] for approximate messaging passing. However, their model does not cover most of possible neural network computations; see Appendix E for discussion.

# 6 Limitations of our results

First, our Theorem 2 is applicable only to Tensor Programs with smooth nonlinearities, which draws out several popular activation functions like ReLU or MaxPool. Our Theorem 9 (see Appendix C) does not really solve the issue since a Tensor Program expressing the backward pass involves derivatives of the activation functions, which are not even continuous for ReLU. As a workaround, we could consider their smoothed versions (e.g. Softplus instead of ReLU) with a controllable smoothness parameter, and put this parameter very close to zero, thus getting "almost ReLU".

Second, we prove convergence in probability, which is weaker than almost sure convergence that holds in the Gaussian case. Because of this, we are only able to prove a weaker (but more traditional) version of the semi-circle law and Free Independence Principle, compared to TP3 [23]; see Appendix B.

# 7 Conclusions

We present a generalization of the Master theorem of [20] to non-Gaussian weight initializations. Our generalization allows for the same applications as the original Master theorem, thus broadening the scope of applicability of the Tensor Programs machinery.

# References

[1] Wei-Kuo Chen and Wai-Kit Lam. Universality of approximate message passing algorithms. Electronic Journal of Probability, 26:1-44, 2021.  
[2] Adrià Garriga-Alonso, Carl Edward Rasmussen, and Laurence Aitchison. Deep convolutional networks as shallow gaussian processes. arXiv preprint arXiv:1808.05587, 2018.  
[3] Eugene Golikov. Towards a general theory of infinite-width limits of neural classifiers. In International Conference on Machine Learning, pages 3617-3626. PMLR, 2020.  
[4] Eugene A Golikov. Dynamically stable infinite-width limits of neural classifiers. arXiv preprint arXiv:2006.06574, 2020.  
[5] Boris Hanin. Random neural networks in the infinite width limit as gaussian processes. arXiv preprint arXiv:2107.01562, 2021.  
[6] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in neural information processing systems, pages 8571-8580, 2018.  
[7] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
[8] Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S Schoenholz, Jeffrey Pennington, and Jascha Sohl-Dickstein. Deep neural networks as gaussian processes. arXiv preprint arXiv:1711.00165, 2017.  
[9] Alexander G de G Matthews, Jiri Hron, Mark Rowland, Richard E Turner, and Zoubin Ghahrami. Gaussian process behaviour in wide deep neural networks. In International Conference on Learning Representations, 2018.  
[10] Radford M Neal. BAYESIAN LEARNING FOR NEURAL NETWORKS. PhD thesis, University of Toronto, 1995.  
[11] Roman Novak, Lechao Xiao, Jaehoon Lee, Yasaman Bahri, Greg Yang, Jiri Hron, Daniel A Abolafia, Jeffrey Pennington, and Jascha Sohl-Dickstein. Bayesian deep convolutional networks with many channels are gaussian processes. arXiv preprint arXiv:1810.05148, 2018.  
[12] Leonid Pastur. On random matrices arising in deep neural networks. gaussian case. arXiv preprint arXiv:2001.06188, 2020.  
[13] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32, pages 8024–8035. Curran Associates, Inc., 2019.  
[14] R. K. Pathria and Paul D. Beale. Statistical mechanics. Elsevier/Academic Press, Amsterdam; Boston, 3rd ed edition, 2011.  
[15] Jeffrey Pennington, Samuel Schoenholz, and Surya Ganguli. Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice. In Advances in neural information processing systems, pages 4785-4795, 2017.  
[16] Ben Poole, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, and Surya Ganguli. Exponential expressivity in deep neural networks through transient chaos. In Advances in neural information processing systems, pages 3360-3368, 2016.  
[17] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. nature, 323(6088):533-536, 1986.  
[18] Terence Tao. Topics in random matrix theory, volume 132. American Mathematical Soc., 2012.

[19] Lechao Xiao, Yasaman Bahri, Jascha Sohl-Dickstein, Samuel Schoenholz, and Jeffrey Pennington. Dynamical isometry and a mean field theory of cnns: How to train 10,000-layer vanilla convolutional neural networks. In International Conference on Machine Learning, pages 5393-5402. PMLR, 2018.  
[20] Greg Yang. Scaling limits of wide neural networks with weight sharing: Gaussian process behavior, gradient independence, and neural tangent kernel derivation. arXiv preprint arXiv:1902.04760, 2019.  
[21] Greg Yang. Tensor programs i: Wide feedforward or recurrent neural networks of any architecture are gaussian processes. arXiv preprint arXiv:1910.12478, 2019.  
[22] Greg Yang. Tensor programs ii: Neural tangent kernel for any architecture. arXiv preprint arXiv:2006.14548, 2020.  
[23] Greg Yang. Tensor programs iii: Neural matrix laws. arXiv preprint arXiv:2009.10685, 2020.  
[24] Greg Yang and Edward J Hu. Tensor programs iv: Feature learning in infinite-width neural networks. In International Conference on Machine Learning, pages 11727-11737. PMLR, 2021.  
[25] Greg Yang, Edward J Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466, 2022.  
[26] Greg Yang and Etai Littwin. Tensor programs iib: Architectural universality of neural tangent kernel training dynamics. In International Conference on Machine Learning, pages 11762-11772. PMLR, 2021.
