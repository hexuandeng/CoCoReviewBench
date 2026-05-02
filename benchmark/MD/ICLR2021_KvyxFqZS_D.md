# GLOBAL CONVERGENCE OF THREE-LAYER NEURAL NETWORKS IN THE MEAN FIELD REGIME

Anonymous authors

Paper under double-blind review

# ABSTRACT

In the mean field regime, neural networks are appropriately scaled so that as the width tends to infinity, the learning dynamics tends to a nonlinear and nontrivial dynamical limit, known as the mean field limit. This lends a way to study large-width neural networks via analyzing the mean field limit. Recent works have successfully applied such analysis to two-layer networks and provided global convergence guarantees. The extension to multilayer ones however has been a highly challenging puzzle, and little is known about the optimization efficiency in the mean field regime when there are more than two layers.

In this work, we prove a global convergence result for unregularized feedforward three-layer networks in the mean field regime. We first develop a rigorous framework to establish the mean field limit of three-layer networks under stochastic gradient descent training. To that end, we propose the idea of a neuronal embedding, which comprises of a fixed probability space that encapsulates neural networks of arbitrary sizes. The identified mean field limit is then used to prove a global convergence guarantee under suitable regularity and convergence mode assumptions, which – unlike previous works on two-layer networks – does not rely critically on convexity. Underlying the result is a universal approximation property, natural of neural networks, which importantly is shown to hold at any finite training time (not necessarily at convergence) via an algebraic topology argument.

# 1 INTRODUCTION

Interests in the theoretical understanding of the training of neural networks have led to the recent discovery of a new operating regime: the neural network and its learning rates are scaled appropriately, such that as the width tends to infinity, the network admits a limiting learning dynamics in which all parameters evolve nonlinearly with time<sup>1</sup>. This is known as the mean field (MF) limit (Mei et al. (2018); Chizat & Bach (2018); Rotskoff & Vanden-Eijnden (2018); Sirignano & Spiliopoulos (2018); Nguyen (2019); Araujo et al. (2019); Sirignano & Spiliopoulos (2019)). The four works Mei et al. (2018); Chizat & Bach (2018); Rotskoff & Vanden-Eijnden (2018); Sirignano & Spiliopoulos (2018) led the first wave of efforts in 2018 and analyzed two-layer neural networks. They established a connection between the network under training and its MF limit. They then used the MF limit to prove that two-layer networks could be trained to find (near) global optima using variants of gradient descent, despite non-convexity (Mei et al. (2018); Chizat & Bach (2018)). The MF limit identified by these works assumes the form of gradient flows in the measure space, which factors out the invariance from the action of a symmetry group on the model. Interestingly, by lifting to the measure space, with a convex loss function (e.g. squared loss), one obtains a limiting optimization problem that is convex (Bengio et al. (2006); Bach (2017)). The analyses of Mei et al. (2018); Chizat & Bach (2018) utilize convexity, although the mechanisms to attain global convergence in these works are more sophisticated than the usual convex optimization setup in Euclidean spaces.

The extension to multilayer networks has enjoyed much less progresses. The works Nguyen (2019); Araujo et al. (2019); Sirignano & Spiliopoulos (2019) argued, heuristically or rigorously, for the existence of a MF limiting behavior under gradient descent training with different assumptions. In

fact, it has been argued that the difficulty is not simply technical, but rather conceptual (Nguyen (2019)): for instance, the presence of intermediate layers exhibits multiple symmetry groups with intertwined actions on the model. Convergence to the global optimum of the model under gradient-based optimization has not been established when there are more than two layers.

In this work, we prove a global convergence guarantee for feedforward three-layer networks trained with unregularized stochastic gradient descent (SGD) in the MF regime. After an introduction of the three-layer setup and its MF limit in Section 2, our development proceeds in two main steps:

Step 1 (Theorem 3 in Section 3): We first develop a rigorous framework that describes the MF limit and establishes its connection with a large-width SGD-trained three-layer network. Here we propose the new idea of a neuronal embedding, which comprises of an appropriate non-evolving probability space that encapsulates neural networks of arbitrary sizes. This probability space is in general abstract and is constructed according to the (not necessarily i.i.d.) initialization scheme of the neural network. This idea addresses directly the intertwined action of multiple symmetry groups, which is the aforementioned conceptual obstacle (Nguyen (2019)), thereby covering setups that cannot be handled by formulations in Araujo et al. (2019); Sirignano & Spiliopoulos (2019) (see also Section 5 for a comparison). Our analysis follows the technique from Sznitman (1991); Mei et al. (2018) and gives a quantitative statement: in particular, the MF limit yields a good approximation of the neural network as long as  $n_{\mathrm{min}}^{-1} \log n_{\mathrm{max}} \ll 1$  independent of the data dimension, where  $n_{\mathrm{min}}$  and  $n_{\mathrm{max}}$  are the minimum and maximum of the widths.

Step 2 (Theorem 8 in Section 4): We prove that the MF limit, given by our framework, converges to the global optimum under suitable regularity and convergence mode assumptions. Several elements of our proof are inspired by Chizat & Bach (2018); the technique in their work however does not generalize to our three-layer setup. Unlike previous two-layer analyses, we do not exploit convexity; instead we make use of a new element: a universal approximation property. The result turns out to be conceptually new: global convergence can be achieved even when the loss function is non-convex. An important crux of the proof is to show that the universal approximation property holds at any finite training time (but not necessarily at convergence, i.e. at infinite time, since the property may not realistically hold at convergence).

Together these two results imply a positive statement on the optimization efficiency of SGD-trained unregularized feedforward three-layer networks (Corollary 10). Certain parts of our work can be readily extended to the general multilayer case or used to obtain new global convergence guarantees in the two-layer case. We choose to keep the current paper concise with the three-layer case being a prototypical setup that conveys the most of the ideas. Complete proofs are presented in appendices.

Notations.  $K$  denotes a generic constant that may change from line to line.  $|\cdot |$  denotes the absolute value for a scalar and the Euclidean norm for a vector. For an integer  $n$ , we let  $[n] = \{1,\dots,n\}$ .

# 2 THREE-LAYER NEURAL NETWORKS AND THE MEAN FIELD LIMIT

# 2.1 THREE-LAYER NEURAL NETWORK

We consider the following three-layer network that takes as input  $x \in \mathbb{R}^d$ :

$$
\hat {\mathbf {y}} (x; \mathbf {W} (k)) = \varphi_ {3} \left(\mathbf {H} _ {3} (x; \mathbf {W} (k))\right), \tag {1}
$$

$$
\mathbf {H} _ {3} \left(x; \mathbf {W} (k)\right) = \frac {1}{n _ {2}} \sum_ {j _ {2} = 1} ^ {n _ {2}} \mathbf {w} _ {3} (k, j _ {2}) \varphi_ {2} \left(\mathbf {H} _ {2} (x, j _ {2}; \mathbf {W} (k))\right),
$$

$$
\mathbf {H} _ {2} \left(x, j _ {2}; \mathbf {W} (k)\right) = \frac {1}{n _ {1}} \sum_ {j _ {1} = 1} ^ {n _ {1}} \mathbf {w} _ {2} \left(k, j _ {1}, j _ {2}\right) \varphi_ {1} \left(\left\langle \mathbf {w} _ {1} (k, j _ {1}), x \right\rangle\right),
$$

in which  $\mathbf{W}(k) = (\mathbf{w}_1(k,\cdot),\mathbf{w}_2(k,\cdot,\cdot),\mathbf{w}_3(k,\cdot))$  is the weight<sup>2</sup> with  $\mathbf{w}_1(k,j_1) \in \mathbb{R}^d$ ,  $\mathbf{w}_2(k,j_1,j_2) \in \mathbb{R}$  and  $\mathbf{w}_3(k,j_2) \in \mathbb{R}$ ,  $\varphi_1: \mathbb{R} \to \mathbb{R}$ ,  $\varphi_2: \mathbb{R} \to \mathbb{R}$  and  $\varphi_3: \mathbb{R} \to \mathbb{R}$  are the activations. Here the network has widths  $\{n_1,n_2\}$ , and  $k \in \mathbb{N}_{\geq 0}$  denotes the (discrete) time.

We train the network with SGD w.r.t. the loss  $\mathcal{L}:\mathbb{R}\times \mathbb{R}\to \mathbb{R}_{\geq 0}$ . We assume that at each time  $k$  we draw independently a fresh sample  $z(k) = (x(k),y(k))\in \mathbb{R}^d\times \mathbb{R}$  from a training distribution  $\mathcal{P}$ . Given an initialization  $\mathbf{W}(0)$ , we update  $\mathbf{W}(k)$  according to

$$
\mathbf {w} _ {3} (k + 1, j _ {2}) = \mathbf {w} _ {3} (k, j _ {2}) - \epsilon \xi_ {3} (k \epsilon) \operatorname {G r a d} _ {3} (z (k), j _ {2}; \mathbf {W} (k)),
$$

$$
\mathbf {w} _ {2} (k + 1, j _ {1}, j _ {2}) = \mathbf {w} _ {2} (k, j _ {1}, j _ {2}) - \epsilon \xi_ {2} (k \epsilon) \operatorname {G r a d} _ {2} (z (k), j _ {1}, j _ {2}; \mathbf {W} (k)),
$$

$$
\mathbf {w} _ {1} (k + 1, j _ {1}) = \mathbf {w} _ {1} (k, j _ {1}) - \epsilon \xi_ {1} (k \epsilon) \operatorname {G r a d} _ {1} (z (k), j _ {1}; \mathbf {W} (k)),
$$

in which  $j_{1} = 1,\dots,n_{1},j_{2} = 1,\dots,n_{2},\epsilon \in \mathbb{R}_{>0}$  is the learning rate,  $\xi_{i}:\mathbb{R}_{\geq 0}\mapsto \mathbb{R}_{\geq 0}$  is the learning rate schedule for  $\mathbf{w}_i$  , and for  $z = (x,y)$  , we define

$$
\operatorname {G r a d} _ {3} (z, j _ {2}; \mathbf {W} (k)) = \partial_ {2} \mathcal {L} (y, \hat {\mathbf {y}} (x; \mathbf {W} (k))) \varphi_ {3} ^ {\prime} (\mathbf {H} _ {3} (x; \mathbf {W} (k))) \varphi_ {2} (\mathbf {H} _ {2} (x, j _ {2}; \mathbf {W} (k))),
$$

$\operatorname{Grad}_2(z, j_1, j_2; \mathbf{W}(k)) = \Delta_2^{\mathbf{H}}(z, j_2; \mathbf{W}(k)) \varphi_1(\langle \mathbf{w}_1(k, j_1), x \rangle)$ ,

$$
\operatorname {G r a d} _ {1} \left(z, j _ {1}; \mathbf {W} (k)\right) = \left(\frac {1}{n _ {2}} \sum_ {j _ {2} = 1} ^ {n _ {2}} \Delta_ {2} ^ {\mathbf {H}} \left(z, j _ {2}; \mathbf {W} (k)\right) \mathbf {w} _ {2} (k, j _ {1}, j _ {2})\right) \varphi_ {1} ^ {\prime} \left(\left\langle \mathbf {w} _ {1} (k, j _ {1}), x \right\rangle\right) x,
$$

$$
\Delta_ {2} ^ {\mathbf {H}} (z, j _ {2}; \mathbf {W} (k)) = \partial_ {2} \mathcal {L} (y, \hat {\mathbf {y}} (x; \mathbf {W} (k))) \varphi_ {3} ^ {\prime} (\mathbf {H} _ {3} (x; \mathbf {W} (k))) \mathbf {w} _ {3} (k, j _ {2}) \varphi_ {2} ^ {\prime} (\mathbf {H} _ {2} (x, j _ {2}; \mathbf {W} (k))).
$$

We note that this setup follows the same scaling w.r.t.  $n_1$  and  $n_2$ , which is applied to both the forward pass and the learning rates in the backward pass, as Nguyen (2019).

# 2.2 MEAN FIELD LIMIT

The MF limit is a continuous-time infinite-width analog of the neural network under training. To describe it, we first introduce the concept of a neuronal ensemble. Given a product probability space  $(\Omega, \mathcal{F}, P) = (\Omega_1 \times \Omega_2, \mathcal{F}_1 \times \mathcal{F}_1, P_1 \times P_2)$ , we independently sample  $C_i \sim P_i$ ,  $i = 1, 2$ . In the following, we use  $\mathbb{E}_{C_i}$  to denote the expectation w.r.t. the random variable  $C_i \sim P_i$  and  $c_i$  to denote an arbitrary point  $c_i \in \Omega_i$ . The space  $(\Omega, \mathcal{F}, P)$  is referred to as a neuronal ensemble.

Given a neuronal ensemble  $(\Omega, \mathcal{F}, P)$ , the MF limit is described by a time-evolving system with state/parameter  $W(t)$ , where the time  $t \in \mathbb{R}_{\geq 0}$  and  $W(t) = (w_{1}(t, \cdot), w_{2}(t, \cdot, \cdot), w_{3}(t, \cdot))$  with  $w_{1}: \mathbb{R}_{\geq 0} \times \Omega_{1} \to \mathbb{R}^{d}$ ,  $w_{2}: \mathbb{R}_{\geq 0} \times \Omega_{1} \times \Omega_{2} \to \mathbb{R}$  and  $w_{3}: \mathbb{R}_{\geq 0} \times \Omega_{2} \to \mathbb{R}$ . It entails the quantities:

$$
\hat {y} (x; W (t)) = \varphi_ {3} \left(H _ {3} (x; W (t))\right),
$$

$$
H _ {3} (x; W (t)) = \mathbb {E} _ {C _ {2}} \left[ w _ {3} (t, C _ {2}) \varphi_ {2} \left(H _ {2} (x, C _ {2}; W (t))\right) \right],
$$

$$
H _ {2} (x, c _ {2}; W (t)) = \mathbb {E} _ {C _ {1}} \left[ w _ {2} (t, C _ {1}, c _ {2}) \varphi_ {1} (\langle w _ {1} (t, C _ {1}), x \rangle) \right].
$$

Here for each  $t \in \mathbb{R}_{\geq 0}$ ,  $w_{1}(t,\cdot)$  is  $(\Omega_1, \mathcal{F}_1)$ -measurable, and similar for  $w_{2}(t,\cdot,\cdot)$ ,  $w_{3}(t,\cdot)$ . The MF limit evolves according to a continuous-time dynamics, described by a system of ODEs, which we refer to as the MF ODEs. Specifically, given an initialization  $W(0) = (w_{1}(0,\cdot), w_{2}(0,\cdot,\cdot), w_{3}(0,\cdot))$ , the dynamics solves:

$$
\partial_ {t} w _ {3} (t, c _ {2}) = - \xi_ {3} (t) \Delta_ {3} (c _ {2}; W (t)),
$$

$$
\partial_ {t} w _ {2} (t, c _ {1}, c _ {2}) = - \xi_ {2} (t) \Delta_ {2} (c _ {1}, c _ {2}; W (t)),
$$

$$
\partial_ {t} w _ {1} (t, c _ {1}) = - \xi_ {1} (t) \Delta_ {1} (c _ {1}; W (t)).
$$

Here  $c_{1} \in \Omega_{1}$ ,  $c_{2} \in \Omega_{2}$ ,  $\mathbb{E}_{Z}$  denotes the expectation w.r.t. the data  $Z = (X,Y) \sim \mathcal{P}$ , and for  $z = (x,y)$ , we define

$$
\Delta_ {3} \left(c _ {2}; W (t)\right) = \mathbb {E} _ {Z} \left[ \partial_ {2} \mathcal {L} \left(Y, \hat {y} (X; W (t))\right) \varphi_ {3} ^ {\prime} \left(H _ {3} (X; W (t))\right) \varphi_ {2} \left(H _ {2} (X, c _ {2}; W (t))\right) \right],
$$

$$
\Delta_ {2} \left(c _ {1}, c _ {2}; W (t)\right) = \mathbb {E} _ {Z} \left[ \Delta_ {2} ^ {H} \left(Z, c _ {2}; W (t)\right) \varphi_ {1} \left(\langle w _ {1} (t, c _ {1}), X \rangle\right) \right],
$$

$$
\Delta_ {1} \left(c _ {1}; W \left(t\right)\right) = \mathbb {E} _ {Z} \left[ \mathbb {E} _ {C _ {2}} \left[ \Delta_ {2} ^ {H} \left(Z, C _ {2}; W \left(t\right)\right) w _ {2} \left(t, c _ {1}, C _ {2}\right) \right] \varphi_ {1} ^ {\prime} \left(\langle w _ {1} \left(t, c _ {1}\right), X \rangle\right) X \right],
$$

$$
\Delta_ {2} ^ {H} (z, c _ {2}; W (t)) = \partial_ {2} \mathcal {L} (y, \hat {y} (x; W (t))) \varphi_ {3} ^ {\prime} (H _ {3} (x; W (t))) w _ {3} (t, c _ {2}) \varphi_ {2} ^ {\prime} (H _ {2} (x, c _ {2}; W (t))).
$$

In Appendix B, we show well-posedness of MF ODEs under the following regularity conditions.

Assumption 1 (Regularity). We assume that  $\varphi_{1}$  and  $\varphi_{2}$  are  $K$ -bounded,  $\varphi_{1}^{\prime}$ ,  $\varphi_{2}^{\prime}$  and  $\varphi_{3}^{\prime}$  are  $K$ -bounded and  $K$ -Lipschitz,  $\varphi_{2}^{\prime}$  and  $\varphi_{3}^{\prime}$  are non-zero everywhere,  $\partial_2\mathcal{L}(\cdot ,\cdot)$  is  $K$ -Lipschitz in the second variable and  $K$ -bounded, and  $|X|\leq K$  with probability 1. Furthermore  $\xi_{1},\xi_{2}$  and  $\xi_{3}$  are  $K$ -bounded and  $K$ -Lipschitz.

Theorem 1. Under Assumption 1, given any neuronal ensemble and an initialization  $W(0)$  such that  $\sup_{t\in [0,\infty)}|w_2(0,C_1,C_2)|$ ,  $\sup_{t\in [0,\infty)}|w_3(0,C_2)| \leq K$ , there exists a unique solution  $W$  to the MF ODEs on  $t\in [0,\infty)$ .

An example of a suitable setup is  $\varphi_{1} = \varphi_{2} = \tanh$ ,  $\varphi_{3}$  is the identity,  $\mathcal{L}$  is the Huber loss, although a non-convex sufficiently smooth loss function suffices. In fact, all of our developments can be easily modified to treat the squared loss with an additional assumption  $|Y| \leq K$  with probability 1.

So far, given an arbitrary neuronal ensemble  $(\Omega ,\mathcal{F},P)$ , for each initialization  $W(0)$ , we have defined a MF limit  $W(t)$ . The connection with the neural network's dynamics  $\mathbf{W}(k)$  is established in the next section.

# 3 CONNECTION BETWEEN NEURAL NETWORK AND ITS MEAN FIELD LIMIT

# 3.1 NEURONAL EMBEDDING AND THE COUPLING PROCEDURE

To formalize a connection between the neural network and its MF limit, we consider their initializations. In practical scenarios, to set the initial parameters  $\mathbf{W}(0)$  of the neural network, one typically randomizes  $\mathbf{W}(0)$  according to some distributional law  $\rho$ . We note that since the neural network is defined w.r.t. a set of finite integers  $\{n_1, n_2\}$ , so is  $\rho$ . We consider a family Init of initialization laws, each of which is indexed by the set  $\{n_1, n_2\}$ :

$\mathsf{Init} = \{\rho : \rho \text{ is the initialization law of a neural network of size } \{n_1, n_2\}, n_1, n_2 \in \mathbb{N}_{>0}\}$ .

This is helpful when one is to take a limit that sends  $n_1, n_2 \to \infty$ , in which case the size of this family  $|\mathrm{Init}|$  is infinite. More generally we allow  $|\mathrm{Init}| < \infty$  (for example, Init contains a single law  $\rho$  of a network of size  $\{n_1, n_2\}$  and hence  $|\mathrm{Init}| = 1$ ). We make the following crucial definition.

Definition 2. Given a family of initialization laws Init, we call  $(\Omega, \mathcal{F}, P, \{w_i^0\}_{i=1,2,3})$  a neuronal embedding of Init if the following hold:

1.  $(\Omega, \mathcal{F}, P) = (\Omega_1 \times \Omega_2, \mathcal{F}_1 \times \mathcal{F}_2, P_1 \times P_2)$  a product measurable space. As a reminder, we call it a neuronal ensemble.  
2. The deterministic functions  $w_1^0: \Omega_1 \to \mathbb{R}^d$ ,  $w_2^0: \Omega_1 \times \Omega_2 \to \mathbb{R}$  and  $w_3^0: \Omega_2 \to \mathbb{R}$  are such that, for each index  $\{n_1, n_2\}$  of Init and the law  $\rho$  of this index, if — with an abuse of notations — we independently sample  $\{C_i(j_i)\}_{j_i \in [n_i]} \sim P_i$  i.i.d. for each  $i = 1, 2$ , then

$$
\text {L a w} \left(w _ {1} ^ {0} \left(C _ {1} (j _ {1})\right), w _ {2} ^ {0} \left(C _ {1} (j _ {1}), C _ {2} (j _ {2})\right), w _ {3} ^ {0} \left(C _ {2} (j _ {2})\right), j _ {i} \in [ n _ {i} ], i = 1, 2\right) = \rho .
$$

To proceed, given Init and  $\{n_1,n_2\}$  in its index set, we perform the following coupling procedure:

1. Let  $(\Omega, \mathcal{F}, P, \{w_i^0\}_{i=1,2,3})$  be a neuronal embedding of Init.  
2. We form the MF limit  $W(t)$  (for  $t \in \mathbb{R}_{\geq 0}$ ) associated with the neuronal ensemble  $(\Omega, \mathcal{F}, P)$  by setting the initialization  $W(0)$  to  $w_{1}(0,\cdot) = w_{1}^{0}(\cdot)$ ,  $w_{2}(0,\cdot,\cdot) = w_{2}^{0}(\cdot,\cdot)$  and  $w_{3}(0,\cdot) = w_{3}^{0}(\cdot)$  and running the MF ODEs described in Section 2.2.  
3. We independently sample  $C_i(j_i) \sim P_i$  for  $i = 1,2$  and  $j_{i} = 1,\dots,n_{i}$ . We then form the neural network initialization  $\mathbf{W}(0)$  with  $\mathbf{w}_1(0,j_1) = w_1^0 (C_1(j_1))$ ,  $\mathbf{w}_2(0,j_1,j_2) = w_2^0 (C_1(j_1),C_2(j_2))$  and  $\mathbf{w}_3(0,j_2) = w_3^0 (C_2(j_2))$  for  $j_{1}\in [n_{1}]$ ,  $j_{2}\in [n_{2}]$ . We obtain the network's trajectory  $\mathbf{W}(k)$  for  $k\in \mathbb{N}_{\geq 0}$  as in Section 2.1, with the data  $z(k)$  generated independently of  $\{C_i(j_i)\}_{i = 1,2}$  and hence  $\mathbf{W}(0)$ .

We can then define a measure of closeness between  $\mathbf{W}\left(\lfloor t / \epsilon \rfloor\right)$  and  $W(t)$  for  $t\in [0,T]$ :

$$
\begin{array}{l} \mathcal {D} _ {T} \left(W, \mathbf {W}\right) = \sup  \left\{\left| \mathbf {w} _ {1} \left(\lfloor t / \epsilon \rfloor , j _ {1}\right) - w _ {1} \left(t, C _ {1} \left(j _ {1}\right)\right) \right|, \left| \mathbf {w} _ {2} \left(\lfloor t / \epsilon \rfloor , j _ {1}, j _ {2}\right) - w _ {2} \left(t, C _ {1} \left(j _ {1}\right), C _ {2} \left(j _ {2}\right)\right) \right|, \right. \\ \left| \mathbf {w} _ {3} \left(\lfloor t / \epsilon \rfloor , j _ {2}\right) - w _ {3} \left(t, C _ {2} (j _ {2})\right) \right|: t \leq T, j _ {1} \leq n _ {1}, j _ {2} \leq n _ {2} \}. \tag {2} \\ \end{array}
$$

Note that  $W(t)$  is a deterministic trajectory independent of  $\{n_1, n_2\}$ , whereas  $\mathbf{W}(k)$  is random for all  $k \in \mathbb{N}_{\geq 0}$  due to the randomness of  $\{C_i(j_i)\}_{i=1,2}$  and the generation of the training data  $z(k)$ . Similarly  $\mathcal{D}_T(W, \mathbf{W})$  is a random quantity.

The idea of the coupling procedure is closely related to the coupling argument in Sznitman (1991); Mei et al. (2018). Here, instead of playing the role of a proof technique, the coupling serves as a vehicle to establish the connection between  $W$  and  $\mathbf{W}$  on the basis of the neuronal embedding. This connection is shown in Theorem 3 below, which gives an upper bound on  $\mathcal{D}_T(W,\mathbf{W})$ .

We note that the coupling procedure can be carried out to provide a connection between  $W$  and  $\mathbf{W}$  as long as there exists a neuronal embedding for Init. Later in Section 4.1, we show that for a common initialization scheme (in particular, i.i.d. initialization) for Init, there exists a neuronal embedding. Theorem 3 applies to, but is not restricted to, this initialization scheme.

# 3.2 MAIN RESULT: APPROXIMATION BY THE MF LIMIT

Assumption 2 (Initialization of second and third layers). We assume that ess-sup  $\left|w_2^0(C_1, C_2)\right|$ , ess-sup  $\left|w_3^0(C_2)\right| \leq K$ , where  $w_2^0$  and  $w_3^0$  are as described in Definition 2.

Theorem 3. Given a family  $\mathrm{Init}$  of initialization laws and a tuple  $\{n_1, n_2\}$  that is in the index set of  $\mathrm{Init}$ , perform the coupling procedure as described in Section 3.1. Fix a terminal time  $T \in \epsilon \mathbb{N}_{\geq 0}$ . Under Assumptions 1 and 2, for  $\epsilon \leq 1$ , we have with probability at least  $1 - 2\delta$

$$
\mathscr {D} _ {T} \left(W, \mathbf {W}\right) \leq e ^ {K _ {T}} \left(\frac {1}{\sqrt {n _ {\operatorname* {m i n}}}} + \sqrt {\epsilon}\right) \log^ {1 / 2} \left(\frac {3 (T + 1) n _ {\max } ^ {2}}{\delta} + e\right) \equiv \operatorname {e r r} _ {\delta , T} \left(\epsilon , n _ {1}, n _ {2}\right),
$$

in which  $n_{\mathrm{min}} = \min \{n_1, n_2\}$ ,  $n_{\mathrm{max}} = \max \{n_1, n_2\}$ , and  $K_T = K(1 + T^K)$ .

The theorem gives a connection between  $\mathbf{W}\left(\lfloor t / \epsilon \rfloor\right)$ , which is defined upon finite widths  $n_1$  and  $n_2$ , and the MF limit  $W(t)$ , whose description is independent of  $n_1$  and  $n_2$ . It lends a way to extract properties of the neural network in the large-width regime.

Corollary 4. Under the same setting as Theorem 3, consider any test function  $\psi : \mathbb{R} \times \mathbb{R} \to \mathbb{R}$  which is  $K$ -Lipschitz in the second variable uniformly in the first variable (an example of  $\psi$  is the loss  $\mathcal{L}$ ). For any  $\delta > 0$ , with probability at least  $1 - 3\delta$ ,

$$
\sup  _ {t \leq T} | \mathbb {E} _ {Z} \left[ \psi \left(Y, \hat {\mathbf {y}} \left(X; \mathbf {W} \left(\lfloor t / \epsilon \rfloor\right)\right)\right) \right] - \mathbb {E} _ {Z} \left[ \psi \left(Y, \hat {y} \left(X; W (t)\right)\right) \right] | \leq e ^ {K _ {T}} \operatorname {e r r} _ {\delta , T} \left(\epsilon , n _ {1}, n _ {2}\right).
$$

These bounds hold for any  $n_1$  and  $n_2$ , similar to Mei et al. (2018); Araújo et al. (2019), in contrast with non-quantitative results in Chizat & Bach (2018); Sirignano & Spiliopoulos (2019). These bounds suggest that  $n_1$  and  $n_2$  can be chosen independent of the data dimension  $d$ . This agrees with the experiments in Nguyen (2019), which found width  $\approx 1000$  to be typically sufficient to observe MF behaviors in networks trained with real-life high-dimensional data.

We observe that the MF trajectory  $W(t)$  is defined as per the choice of the neuronal embedding  $(\Omega, \mathcal{F}, P, \{w_i^0\}_{i=1,2,3})$ , which may not be unique. On the other hand, the neural network's trajectory  $\mathbf{W}(k)$  depends on the randomization of the initial parameters  $\mathbf{W}(0)$  according to an initialization law from the family Init (as well as the data  $z(k)$ ) and hence is independent of this choice. Another corollary of Theorem 3 is that given the same family Init, the law of the MF trajectory is insensitive to the choice of the neuronal embedding of Init.

Corollary 5. Consider a family  $\mathsf{Init}$  of initialization laws, indexed by a set of tuples  $\{m_1,m_2\}$  that contains a sequence of indices  $\{m_{1}(m),m_{2}(m):m\in \mathbb{N}\}$  in which as  $m\to \infty$ $\min \left\{m_1(m),m_2(m)\right\}^{-1}\log \left(\max \left\{m_1(m),m_2(m)\right\}\right)\rightarrow 0.$  Let  $W(t)$  and  $\hat{W} (t)$  be two MF trajectories associated with two choices of neuronal embeddings of Init,  $(\Omega ,\mathcal{F},P,\{w_i^0\}_{i = 1,2,3})$  and  $(\hat{\Omega},\hat{\mathcal{F}},\hat{P},\{\hat{w}_i^0\}_{i = 1,2,3})$  . The following statement holds for any  $T\geq 0$  and any two positive integers  $n_1$  and  $n_2$  : if we independently sample  $C_i(j_i)\sim P_i$  and  $\hat{C}_i(j_i)\sim \hat{P}_i$  for  $j_{i}\in [n_{i}]$ $i = 1,2$  then Law  $(\mathcal{W}(n_1,n_2,T)) = \mathrm{Law}(\hat{\mathcal{W}} (n_1,n_2,T))$  , where we define  $\mathcal{W}(n_1,n_2,T)$  as the below collection w.r.t.  $W(t)$  , and similarly define  $\hat{\mathcal{W}} (n_1,n_2,T)$  w.r.t.  $\hat{W} (t)$  ..

$$
\begin{array}{l} \mathcal {W} \left(n _ {1}, n _ {2}, T\right) = \left\{w _ {1} \left(t, C _ {1} \left(j _ {1}\right)\right), w _ {2} \left(t, C _ {1} \left(j _ {1}\right), C _ {2} \left(j _ {2}\right)\right), w _ {3} \left(t, C _ {2} \left(j _ {2}\right)\right): \right. \\ \left. j _ {1} \in [ n _ {1} ], j _ {2} \in [ n _ {2} ], t \in [ 0, T ] \right\}. \\ \end{array}
$$

The proofs are deferred to Appendix C.

# 4 CONVERGENCE TO GLOBAL OPTIMA

In this section, we prove a global convergence guarantee for three-layer neural networks via the MF limit. We consider a common class of initialization: i.i.d. initialization.

# 4.1 I.I.D. INITIALIZATION

Definition 6. An initialization law  $\rho$  for a neural network of size  $\{n_1, n_2\}$  is called  $\left(\rho^1, \rho^2, \rho^3\right)$ -i.i.d. initialization (or i.i.d. initialization, for brevity), where  $\rho^1$ ,  $\rho^2$  and  $\rho^3$  are probability measures over  $\mathbb{R}^d$ ,  $\mathbb{R}$  and  $\mathbb{R}$  respectively, if  $\{\mathbf{w}_1(0, j_1)\}_{j_1 \in [n_1]}$  are generated i.i.d. according to  $\rho^1$ ,  $\{\mathbf{w}_2(0, j_1, j_2)\}_{j_1 \in [n_1], j_2 \in [n_2]}$  are generated i.i.d. according to  $\rho^2$  and  $\{\mathbf{w}_3(0, j_2)\}_{j_2 \in [n_2]}$  are generated i.i.d. according to  $\rho^3$ , and  $\mathbf{w}_1$ ,  $\mathbf{w}_2$  and  $\mathbf{w}_3$  are independent.

Observe that given  $(\rho^1, \rho^2, \rho^3)$ , one can build a family Init of i.i.d. initialization laws that contains any index set  $\{n_1, n_2\}$ . Furthermore i.i.d. initializations are supported by our framework, as stated in the following proposition and proven in Appendix D.

Proposition 7. There exists a neuronal embedding  $\left(\Omega, \mathcal{F}, P, \left\{w_i^0\right\}_{i=1,2,3}\right)$  for any family Init of initialization laws, which are  $(\rho^1, \rho^2, \rho^3)$ -i.i.d.

# 4.2 MAIN RESULT: GLOBAL CONVERGENCE

To measure the learning quality, we consider the loss averaged over the data  $Z \sim \mathcal{P}$ :

$$
\mathcal {L} (V) = \mathbb {E} _ {Z} [ \mathcal {L} (Y, \hat {y} (X; V)) ],
$$

where  $V = (v_{1}, v_{2}, v_{3})$  a set of three measurable functions  $v_{1}: \Omega_{1} \to \mathbb{R}^{d}$ ,  $v_{2}: \Omega_{1} \times \Omega_{2} \to \mathbb{R}$ ,  $v_{3}: \Omega_{2} \to \mathbb{R}$ .

Assumption 3. Consider a neuronal embedding  $\left(\Omega, \mathcal{F}, P, \left\{w_{i}^{0}\right\}_{i=1,2,3}\right)$  of the  $(\rho^{1}, \rho^{2}, \rho^{3})$ -i.i.d. initialization, and the associated MF limit with initialization  $W(0)$  such that  $w_{1}(0, \cdot) = w_{1}^{0}(\cdot)$ ,  $w_{2}(0, \cdot, \cdot) = w_{2}^{0}(\cdot, \cdot)$  and  $w_{3}(0, \cdot) = w_{3}^{0}(\cdot)$ . Assume:

1. Support: The support of  $\rho^1$  is  $\mathbb{R}^d$ .  
2. Convergence mode: There exist limits  $\bar{w}_1$ ,  $\bar{w}_2$  and  $\bar{w}_3$  such that as  $t \to \infty$ ,

$$
\begin{array}{l} \mathbb {E} \left[ (1 + | \bar {w} _ {3} (C _ {2}) |) | \bar {w} _ {3} (C _ {2}) | | \bar {w} _ {2} (C _ {1}, C _ {2}) | | w _ {1} (t, C _ {1}) - \bar {w} _ {1} (C _ {1}) | ] \rightarrow 0, \right. (3) \\ \mathbb {E} \left[ (1 + | \bar {w} _ {3} (C _ {2}) |) | \bar {w} _ {3} (C _ {2}) | | w _ {2} (t, C _ {1}, C _ {2}) - \bar {w} _ {2} (C _ {1}, C _ {2}) | ] \rightarrow 0, \right. (4) \\ \mathbb {E} \left[ (1 + | \bar {w} _ {3} (C _ {2}) |) | w _ {3} (t, C _ {2}) - \bar {w} _ {3} (C _ {2}) | \right]\rightarrow 0, (5) \\ \operatorname {e s s} - \sup  \mathbb {E} _ {C _ {2}} \left[ | \partial_ {t} w _ {2} (t, C _ {1}, C _ {2}) | \right]\rightarrow 0. (6) \\ \end{array}
$$

3. Universal approximation:  $\{\varphi_1(\langle u,\cdot \rangle):u\in \mathbb{R}^d\}$  has dense span in  $L^2 (\mathcal{P}_X)$  (the space of square integrable functions w.r.t.  $\mathcal{P}_X$  the distribution of the input  $X$ ).

Assumption 3 is inspired by the work Chizat & Bach (2018) on two-layer networks, with certain differences. Assumptions 3.1 and 3.3 are natural in neural network learning (Cybenko (1989); Chen & Chen (1995)), while we note Chizat & Bach (2018) does not utilize universal approximation. Similar to Chizat & Bach (2018), Assumption 3.2 is technical and does not seem removable. Note that this assumption specifies the mode of convergence and is not an assumption on the limits  $\bar{w}_1$ ,  $\bar{w}_2$  and  $\bar{w}_3$ . Specifically conditions (3)-(5) are similar to the convergence assumption in Chizat & Bach (2018). We differ from Chizat & Bach (2018) fundamentally in the essential supremum condition (6). On one hand, this condition helps avoiding the Morse-Sard type condition in Chizat & Bach (2018), which is difficult to verify in general and not simple to generalize to the three-layer case. On the other hand, it turns out to be a natural assumption to make, in light of Remark 9 below.

We now state the main result of the section. The proof is in Appendix D.

Theorem 8. Consider a neuronal embedding  $\left(\Omega, \mathcal{F}, P, \left\{w_{i}^{0}\right\}_{i=1,2,3}\right)$  of  $(\rho^{1}, \rho^{2}, \rho^{3})$ -i.i.d. initialization. Consider the MF limit corresponding to the network (1), such that they are coupled together by the coupling procedure in Section 3.1, under Assumptions 1, 2 and 3. For simplicity, assume  $\xi_{1}(\cdot) = \xi_{2}(\cdot) = 1$ . Further assume either:

- (untrained third layer)  $\xi_3(\cdot) = 0$  and  $w_3^0 (C_2)\neq 0$  with a positive probability, or  
- (trained third layer)  $\xi_3(\cdot) = 1$  and  $\mathcal{L}\left(w_1^0, w_2^0, w_3^0\right) < \mathbb{E}_Z[\mathcal{L}(Y, \varphi_3(0))]$ .

Then the following hold:

- Case 1 (convex loss): If  $\mathcal{L}$  is convex in the second variable, then

$$
\lim  _ {t \to \infty} \mathcal {L} \left(W \left(t\right)\right) = \inf  _ {V} \mathcal {L} \left(V\right) = \inf  _ {\tilde {y}: \mathbb {R} ^ {d} \to \mathbb {R}} \mathbb {E} _ {Z} \left[ \mathcal {L} \left(Y, \tilde {y} \left(X\right)\right) \right].
$$

- Case 2 (generic non-negative loss): Suppose that  $\partial_2\mathcal{L}(y,\hat{y}) = 0$  implies  $\mathcal{L}(y,\hat{y}) = 0$ . If  $y = y(x)$  is a function of  $x$ , then  $\mathcal{L}(W(t)) \to 0$  as  $t \to \infty$ .

Remarkably here the theorem allows for non-convex losses. A further inspection of the proof shows that we do not rely critically on any convexity property. We also remark that the same proof of global convergence should extend beyond the specific fully-connected architecture considered here. Similar to previous results on SGD-trained two-layer networks Mei et al. (2018); Chizat & Bach (2018), our current result in the three-layer case is non-quantitative.

Remark 9. Interestingly there is a converse relation between global convergence and the essential supremum condition (6): under the same setting, global convergence is unattainable if condition (6) does not hold. A similar observation was made in Wojtowytsch (2020) for two-layer ReLU networks. A precise statement and its proof can be found in Appendix E.

The following result is straightforward from Theorem 8 and Corollary 4, establishing the optimization efficiency of the neural network with SGD.

Corollary 10. Consider the neural network (1). Under the same setting as Theorem 8, in Case 1,

$$
\lim  _ {t \rightarrow \infty} \lim  _ {n _ {1}, n _ {2}} \lim  _ {\epsilon \rightarrow 0} \mathbb {E} _ {Z} \left[ \mathcal {L} \left(Y, \hat {\mathbf {y}} \left(X; \mathbf {W} \left(\lfloor t / \epsilon \rfloor\right)\right)\right)\right] = \inf  _ {f _ {1}, f _ {2}, f _ {3}} \mathcal {L} \left(f _ {1}, f _ {2}, f _ {3}\right) = \inf  _ {\tilde {y}} \mathbb {E} _ {Z} \left[ \mathcal {L} \left(Y, \tilde {y} (X)\right)\right]
$$

in probability, where the limit of the widths is such that  $\min \{n_1,n_2\}^{-1}\log (\max \{n_1,n_2\})\to 0$  In Case 2, the same holds with the right-hand side being 0.

# 4.3 HIGH-LEVEL IDEA OF THE PROOF

We give a high-level discussion of the proof. This is meant to provide intuitions and explain the technical crux, so our discussion may simplify and deviate from the actual proof.

Our first insight is to look at the second layer's weight  $w_{2}$ . At convergence time  $t = \infty$ , we expect to have zero movement and hence, denoting  $W(\infty) = (\bar{w}_1,\bar{w}_2,\bar{w}_3)$ :

$$
\Delta_ {2} \left(c _ {1}, c _ {2}; W (\infty)\right) = \mathbb {E} _ {Z} \left[ \Delta_ {2} ^ {H} \left(Z, c _ {2}; W (\infty)\right) \varphi_ {1} \left(\langle \bar {w} _ {1} (c _ {1}), X \rangle\right) \right] = 0,
$$

for  $P$ -almost every  $c_{1}, c_{2}$ . Suppose for the moment that we are allowed to make an additional (strong) assumption on the limit  $\bar{w}_{1}$ :  $\mathrm{supp}(\bar{w}_{1}(C_{1})) = \mathbb{R}^{d}$ . It implies that the universal approximation property, described in Assumption 3, holds at  $t = \infty$ ; more specifically, it implies  $\{\varphi_{1}(\langle \bar{w}_{1}(c_{1}), \cdot \rangle): c_{1} \in \Omega_{1}\}$  has dense span in  $L^{2}(\mathcal{P}_{X})$ . This thus yields

$$
\mathbb {E} _ {Z} \left[ \Delta_ {2} ^ {H} \left(Z, c _ {2}; W (\infty)\right) \middle | X = x \right] = 0,
$$

for  $\mathcal{P}$ -almost every  $x$ . Recalling the definition of  $\Delta_2^H$ , one can then easily show that

$$
\mathbb {E} _ {Z} \left[ \partial_ {2} \mathcal {L} \left(Y, \hat {y} (X; W (\infty))\right) | X = x \right] = 0.
$$

Global convergence follows immediately; for example, in Case 2 of Theorem 8, this is equivalent to that  $\partial_2\mathcal{L}(y(x),\hat{y} (x;W(\infty))) = 0$  and hence  $\mathcal{L}(y(x),\hat{y} (x;W(\infty))) = 0$  for  $\mathcal{P}$ -almost every  $x$ . In short, the gradient flow structure of the dynamics of  $w_{2}$  provides a seamless way to obtain global convergence. Furthermore there is no critical reliance on convexity.

However this plan of attack has a potential flaw in the strong assumption that  $\operatorname{supp}(\bar{w}_1(C_1)) = \mathbb{R}^d$ , i.e. the universal approximation property holds at convergence time. Indeed there are setups where it is desirable that  $\operatorname{supp}(\bar{w}_1(C_1)) \neq \mathbb{R}^d$  (Mei et al. (2018); Chizat (2019)); for instance, it is the case where the neural network is to learn some "sparse and spiky" solution, and hence the weight distribution at convergence time, if successfully trained, cannot have full support. On the other hand, one can entirely expect that if  $\operatorname{supp}(w_1(0,C_1)) = \mathbb{R}^d$  initially at  $t = 0$ , then  $\operatorname{supp}(w_1(t,C_1)) = \mathbb{R}^d$  at any finite  $t \geq 0$ . The crux of our proof is to show the latter without assuming  $\operatorname{supp}(\bar{w}_1(C_1)) = \mathbb{R}^d$ .

This task is the more major technical step of the proof. To that end, we first show that there exists a mapping  $(t,u)\mapsto M(t,u)$  that maps from  $(t,w_{1}(0,c_{1})) = (t,u)$  to  $w_{1}(t,c_{1})$  via a careful measurability argument. This argument rests on a scheme that exploits the symmetry in the network evolution. Furthermore the map  $M$  is shown to be continuous. The desired conclusion then follows from an algebraic topology argument that the map  $M$  preserves a homotopic structure through time.

# 5 DISCUSSION

The MF literature is fairly recent. A long line of works (Nitanda & Suzuki (2017); Mei et al. (2018); Chizat & Bach (2018); Rotskoff & Vanden-Eijnden (2018); Sirignano & Spiliopoulos (2018); Wei et al. (2019); Javanmard et al. (2019); Mei et al. (2019); Shevchenko & Mondelli (2019); Wojtowytsch (2020)) have focused mainly on two-layer neural networks, taking an interacting particle system approach to describe the MF limiting dynamics as Wasserstein gradient flows. The three works Nguyen (2019); Araujo et al. (2019); Sirignano & Spiliopoulos (2019) independently develop different formulations for the MF limit in multilayer neural networks, under different assumptions. The latter two works Araujo et al. (2019); Sirignano & Spiliopoulos (2019) develop MF formulations that hold exclusively for i.i.d. initializations. As shown in Araujo et al. (2019), when there are more than three layers and no biases, i.i.d. initializations cause a strong simplifying effect such that the evolution of weights in one layer becomes independent of all other layers. On the other hand, our framework is closer to the spirit of the heuristic work Nguyen (2019): the framework supports non-i.i.d. initializations which avoid the simplifying effect, as long as there exist suitable neuronal embeddings. Although our global convergence result is proven in the context of i.i.d. initializations for three-layer networks, we believe non-i.i.d. initializations could provide useful insights in the general case and leave that to another report. Our Theorem 3 provides a quantitative bound and only requires sufficiently large  $\min\{n_1, n_2\}$ , whereas Sirignano & Spiliopoulos (2019) is non-quantitative and takes an unnatural sequential limit  $n_1 \to \infty$  before  $n_2 \to \infty$ . The theorem does not assume untrained input and output weights, compared to Araujo et al. (2019). We note that Theorem 3 can be extended to general multilayer networks using the neuronal embedding idea.

Global convergence in the two-layer case with convex losses has enjoyed multiple efforts with a lot of new and interesting results (Mei et al. (2018); Chizat & Bach (2018); Javanmard et al. (2019); Rotskoff et al. (2019); Wei et al. (2019)). Our work is the first to establish a global convergence guarantee for SGD-trained three-layer networks in the MF regime. Our proof sends a new message that the crucial factor is not necessarily convexity, but rather that the whole learning trajectory maintains the universal approximation property of the function class represented by the first layer's neurons, together with the gradient flow structure of the second layer's weights. As a remark, our approach can also be applied to prove a similar global convergence guarantee for two-layer networks, removing the convex loss assumption in previous works. The recent work Lu et al. (2020) on a MF resnet model (a composition of many two-layer MF networks) and a recent update of Sirignano & Spiliopoulos (2019) essentially establish conditions of stationary points to be global optima. They however require the strong assumption of full support of the limit point. As explained in Section 4.3, we analyze the training dynamics without such assumption and in fact allow it to be violated.

Our global convergence result is non-quantitative. An important, highly challenging future direction is to develop a quantitative version of global convergence; previous works on two-layer networks Javanmard et al. (2019); Wei et al. (2019); Rotskoff et al. (2019); Chizat (2019) have done so under sophisticated modifications of the architecture and training algorithms.

Finally we remark that our insights on global convergence here can be applied to prove similar results for more general multilayer networks. The details are deferred to another report.

# REFERENCES

Zeyuan Allen-Zhu, Yanzhi Li, and Zhao Song. A convergence theory for deep learning via overparameterization. In Proceedings of the 36th International Conference on Machine Learning, volume 97, pp. 242-252, 2019. 1  
Dyego Araujo, Roberto I Oliveira, and Daniel Yukimura. A mean-field limit for certain deep neural networks. arXiv preprint arXiv:1906.00193, 2019. 1, 1, 3.2, 5  
Francis Bach. Breaking the curse of dimensionality with convex neural networks. The Journal of Machine Learning Research, 18(1):629-681, 2017. 1  
Yoshua Bengio, Nicolas L Roux, Pascal Vincent, Olivier Delalleau, and Patrice Marcotte. Convex neural networks. In Advances in neural information processing systems, pp. 123-130, 2006. 1  
Tianping Chen and Hong Chen. Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. IEEE Transactions on Neural Networks, 6(4):911-917, 1995. 4.2  
Lenaic Chizat. Sparse optimization on measures with over-parameterized gradient descent. arXiv preprint arXiv:1907.10300, 2019. 4.3, 5  
Lenaic Chizat and Francis Bach. On the global convergence of gradient descent for overparameterized models using optimal transport. In Advances in Neural Information Processing Systems, pp. 3040-3050. 2018. 1, 1, 3.2, 4.2, 4.2, 5, D.2  
Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. In Advances in Neural Information Processing Systems 32, pp. 2937-2947. 2019. 1  
George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4):303-314, 1989. 4.2  
Simon S. Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes over-parameterized neural networks. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=S1eK3i09YQ.1  
Vitaly Feldman and Jan Vondrak. Generalization bounds for uniformly stable algorithms. In Advances in Neural Information Processing Systems, pp. 9747-9757, 2018. F  
Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems 31, pp. 8580-8589. 2018. 1  
Adel Javanmard, Marco Mondelli, and Andrea Montanari. Analysis of a two-layer neural network via displacement convexity. arXiv preprint arXiv:1901.01375, 2019. 5  
Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. In Advances in Neural Information Processing Systems 32, pp. 8572-8583. 2019. 1  
Yiping Lu, Chao Ma, Yulong Lu, Jianfeng Lu, and Lexing Ying. A mean-field analysis of deep resnet and beyond: Towards provable optimization via overparameterization from depth. arXiv preprint arXiv:2003.05508, 2020. 5  
Song Mei, Andreaa Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two layers neural networks. In Proceedings of the National Academy of Sciences, volume 115, pp. 7665-7671, 2018. 1, 1, 3.1, 3.2, 4.2, 4.3, 5  
Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Mean-field theory of two-layers neural networks: dimension-free bounds and kernel limit. arXiv preprint arXiv:1902.06015, 2019. 5  
Phan-Minh Nguyen. Mean field limit of the learning dynamics of multilayer neural networks. arXiv preprint arXiv:1902.02880, 2019. 1, 1, 2.1, 3.2, 5

Atsushi Nitanda and Taiji Suzuki. Stochastic particle gradient descent for infinite ensembles. arXiv preprint arXiv:1712.05438, 2017. 5  
Iosif Pinelis. Optimum bounds for the distributions of martingales in banach spaces. The Annals of Probability, 22(4):1679-1706, 1994. F  
Iosif Pinelis and Alexander I Sakanenko. Remarks on inequalities for large deviation probabilities. Theory of Probability & Its Applications, 30(1):143-148, 1986. F, F  
Grant Rotskoff and Eric Vanden-Eijnden. Parameters as interacting particles: long time convergence and asymptotic error scaling of neural networks. In Advances in Neural Information Processing Systems 31, pp. 7146-7155. 2018. 1, 5  
Grant Rotskoff, Samy Jelassi, Joan Bruna, and Eric Vanden-Eijnden. Neuron birth-death dynamics accelerates gradient descent and converges asymptotically. In Proceedings of the 36th International Conference on Machine Learning, volume 97, pp. 5508-5517, 2019. 5  
Alexander Shevchenko and Marco Mondelli. Landscape connectivity and dropout stability of sgd solutions for over-parameterized neural networks. arXiv preprint arXiv:1912.10095, 2019. 5  
Justin Sirignano and Konstantinos Spiliopoulos. Mean field analysis of neural networks. arXiv preprint arXiv:1805.01053, 2018. 1, 5  
Justin Sirignano and Konstantinos Spiliopoulos. Mean field analysis of deep neural networks. arXiv preprint arXiv:1903.04440, 2019. 1, 1, 3.2, 5  
Alain-Sol Sznitman. Topics in propagation of chaos. In Ecole d'été de probabilités de Saint-Flour XIX—1989, pp. 165–251. Springer, 1991. 1, 3.1  
Roman Vershynin. Introduction to the non-asymptotic analysis of random matrices. arXiv preprint arXiv:1011.3027, 2010. F  
Colin Wei, Jason D Lee, Qiang Liu, and Tengyu Ma. Regularization matters: Generalization and optimization of neural nets v.s. their induced kernel. In Advances in Neural Information Processing Systems 32, pp. 9712-9724. 2019. 5  
Stephan Wojtowytsch. On the convergence of gradient descent training for two-layer relu-networks in the mean field regime. arXiv preprint arXiv:2005.13530, 2020. 9, 5  
Difan Zou, Yuan Cao, Dongruo Zhou, and Quanquan Gu. Stochastic gradient descent optimizes over-parameterized deep relu networks. arXiv preprint arXiv:1811.08888, 2018. 1
