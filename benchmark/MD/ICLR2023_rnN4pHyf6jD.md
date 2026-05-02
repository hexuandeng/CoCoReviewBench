# SCALING CONVEX NEURAL NETWORKS WITH BURERMONTEIRO FACTORIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recently, it has been demonstrated that a wide variety of (non) linear two-layer neural networks (such as two-layer perceptrons, convolutional networks, and self-attention) can be posed as equivalent convex optimization problems, with an induced regularizer which encourages low rank. However, this regularizer becomes prohibitively expensive to compute at moderate scales, impeding training convex neural networks. To this end, we propose applying the Burer-Monteiro factorization to convex neural networks, which for the first time enables a Burer-Monteiro perspective on neural networks with non-linearities. This factorization leads to an equivalent yet computationally tractable non-convex alternative with no spurious local minima. We develop a novel relative optimality bound of stationary points of the Burer-Monteiro factorization, thereby providing verifiable conditions under which any stationary point is a global optimum. Further, for the first time, we show that linear self-attention with sufficiently many heads has no spurious local minima. Our experiments demonstrate the utility and implications of the novel relative optimality bound for stationary points of the Burer-Monteiro factorization.

# 1 INTRODUCTION

It has been demonstrated that (non-linear) two-layer neural networks are equivalent to convex programs (Pilanci & Ergen, 2020; Ergen & Pilanci, 2020; Sahiner et al., 2021b; Ergen et al., 2021; Sahiner et al., 2021a). This has been observed for a variety of architectures, including multi-layer perceptrons (MLPs) (Pilanci & Ergen, 2020; Sahiner et al., 2021b), convolutional neural networks (CNNs) (Ergen & Pilanci, 2020; Sahiner et al., 2021c), and self-attention based transformers (Sahiner et al., 2022). There are definite benefits of convex training of neural networks. Most prominently that global optimality is guaranteed, which brings transparency to training neural networks.

The convex formulation of neural networks induces biases by regularization of the network weights. For linear activation, the convex model directly imposes nuclear-norm regularization which is well-known to encourage low-rank solutions (Recht et al., 2010). For ReLU activation, however, the convex model induces a type of nuclear norm which promotes sparse factorization while the left factor is constrained to an affine space (Sahiner et al., 2021b). This constrained nuclear-norm is NP-hard to compute. This impedes the utility of convex neural networks for ReLU activation.

To address this computational challenge, we seek a method which  $(i)$  inherits the per-iteration complexity of non-convex training of neural network, and  $(ii)$  inherits the optimality guarantees and transparency of convex training. To find a solution, we leverage the well-studied Burer-Monterio (BM) factorization (Burer & Monteiro, 2003), which was originally proposed as a heuristic method to improve the complexity of convex semi-definite programs (SDPs).

BM has been applied as an efficient solution strategy for problems ranging from matrix factorization (Zheng & Lafferty, 2016; Park et al., 2017; Ge et al., 2017; Gillis, 2017) to rank minimization (Mardani et al., 2013; Recht et al., 2010; Wang et al., 2017) and matrix completion (Mardani et al., 2015; Ge et al., 2017). BM has also been used for over-simplified neural networks such as (Kawaguchi, 2016; Haeffele & Vidal, 2017; Du & Lee, 2018), where optimality conditions for local minima are provided. However, no work has deployed BM factorization for practical non-linear neural networks, and no guarantees are available about the optimality of stationary points. This is likely because BM theory is not applicable to the standard non-convex ReLU networks due to non-linearity between layer weights.

Thus, our focus in this work is to adapt BM for practical two-layer (non-linear) convex neural networks. We consider three common architectures, namely MLPs, CNNs, and self-attention networks. For these scenarios, we develop verifiable relative optimality bounds for all local minima and stationary points, which are easy and interpretable. In light of these conditions, we identify useful insights about the nature of neural networks contributing to optimality. In particular, we observe that for self-attention networks all local minima coincide with the global optima if there are sufficiently many heads. The optimality guarantees also provide useful algorithmic insights, allowing one to verify whether the light-weight first-order methods such as SGD achieve the global optimum for the non-convex training of neural networks. Our experiments with image classification task indicate that this BM factorization enables layerwise training of convex CNNs, which allows for convex networks for the first time to match the performance of multi-layer end-to-end trained non-convex CNNs.

# 1.1 CONTRIBUTIONS

All in all, our contributions are summarized as follows:

- We propose the BM factorization for efficiently solving convex neural networks with ReLU activation for moderate and large scales. This is the first time BM theory has been applied to the non-linear neural network setting.  
- We derive a novel bound on the relative optimality of the stationary points of the BM factorization for neural networks.  
- Accordingly, we identify simple and verifiable conditions which guarantee a stationary point of the non-convex BM formulation achieves the global optimum of the convex neural network.  
- We yield basic insights into the fundamental nature of neural networks that contribute to optimality; e.g. that linear self-attention has no spurious local minima if it has sufficiently many heads.  
- Our experiments demonstrate that the BM factorization enables layer-wise training of two-layer convex CNNs, which allows convex optimization methods to be competitive with end-to-end training of multi-layer networks.

# 1.2 RELATED WORK

Burer-Monteiro factorization. The Burer-Monteiro (BM) factorization was first introduced in (Burer & Monteiro, 2003; 2005). There has been a long line of work studying the use of this factorization for solving SDPs (Boumal et al., 2016; Cifuentes & Moitra, 2019; Waldspurger & Waters, 2020; Erdogdu et al., 2021). Furthermore, in the rectangular matrix case, it has been shown that gradient descent converges to a global optimum of the matrix factorization problem with high probability for certain classes of matrices (Zheng & Lafferty, 2016). The BM factorization has been also studied in the rectangular case in more generic settings (Bach et al., 2008; Haeffele et al., 2014), which can be applied to some specialized neural network architectures (Haeffele & Vidal, 2017).

Nuclear norm and rank minimization. The ability of nuclear norm regularization to induce low rank has been studied extensively in compressed sensing (Candes & Recht, 2009; Recht et al., 2010; Candès & Tao, 2010). BM factorization has been applied to scale up nuclear-norm minimization (Mardani et al., 2015; 2013). It has also been deployed for low-rank matrix factorization (Cabral et al., 2013; Zhu et al., 2017; Park et al., 2017; Ge et al., 2017). The results show that all second-order critical points of the BM factorization are global optima if certain qualification conditions are met.

SGD for non-convex neural networks. It has been shown that for over-parameterized two-layer linear networks, all local minima are global minima (Kawaguchi, 2016). Accordingly, a line of work has attempted to show that gradient descent or its modifications provably find local minima and escape saddle points (Ge et al., 2015; Lee et al., 2016; Jin et al., 2017; Daneshmand et al., 2018). However, these works assume Lipschitz gradients and Hessians of the non-convex objective, which is not typically satisfied. Another line of work shows that gradient descent converges to global optima for sufficiently highly over-parameterized neural networks, with either the parameter count being a high-order polynomial of the sample count (Du et al., 2018; 2019; Arora et al., 2019), or the network

architecture being simple (Du & Lee, 2018). In practice, it has been empirically observed that SGD can converge to local maxima, or get stuck in saddle points (Du et al., 2017; Ziyin et al., 2021). For matrix factorization, it has also recently been shown that randomly initialized gradient descent on BM factorization provably converges to global minima (Ye & Du, 2021). However, only the squared loss is considered and no regularization is used.

Convex neural networks. It has been recently shown that a variety of neural networks have equivalent convex programs for training. Some of those include: ReLU networks with scalar outputs (Pilanci & Ergen, 2020), and vector-output (Sahiner et al., 2021b), convolutional networks (Ergen & Pilanci, 2020; Sahiner et al., 2021c), polynomial-activation networks (Bartan & Pilanci, 2021), batch-norm based networks (Ergen et al., 2021), Wasserstein GANs (Sahiner et al., 2021a), and self-attention networks (Sahiner et al., 2022). Despite some efforts in developing efficient solvers, convex networks are only effectively trainable at small scales (Bai et al., 2022; Mishkin et al., 2022). We observe that these convex formulations of ReLU networks enable BM factorizations for ReLU networks. Our novelty is to adapt BM factorization as a fast and scalable solution for training convex networks, with simple and verifiable conditions for global optimality.

# 2 PRELIMINARIES

We denote  $(\cdot)_{+} := \max \{0, \cdot\}$  as the ReLU non-linearity. We use superscripts, say  $\mathbf{A}^{(i_i, i_2)}$ , to denote blocks of matrices, and brackets, say  $\mathbf{A}[i_1, i_2]$ , to denote elements of matrices. We let  $\mathbf{1}$  be the vector of ones of appropriate size, and  $\mathcal{B}_H$  be the unit  $H$ -norm ball,  $\{\mathbf{u} : \| \mathbf{u} \|_H \leq 1\}$ . Unless otherwise stated, let  $F$  be a convex, differentiable function. All proofs are presented in the Appendix.

# 2.1 TWO-LAYER NEURAL NETWORKS AS CONVEX PROGRAMS

A line of work has demonstrated that two-layer neural networks are equivalent to convex optimization problems. We consider a data matrix  $\mathbf{X} \in \mathbb{R}^{n \times d}$  and consider two-layer  $\sigma$ -activation network with  $c$  outputs,  $m$  neurons, weight-decay parameter  $\beta > 0$ :

$$
p_{MLP}^{*}:= \min_{\substack{\mathbf{W}_{1}\in \mathbb{R}^{d\times m}\\ \mathbf{W}_{2}\in \mathbb{R}^{c\times m}}}F(\sigma (\mathbf{X}\mathbf{W}_{1})\mathbf{W}_{2}^{\top}) + \frac{\beta}{2}\sum_{j = 1}^{m}\| \mathbf{w}_{1j}\|_{2}^{2} + \| \mathbf{w}_{2j}\|_{2}^{2}. \tag{1}
$$

When  $\sigma$  is a linear activation and  $m\geq m^{*}$  for some  $m^{*}\leq \min \{d,c\}$ , this problem is equivalent to ((Rennie & Srebro, 2005), Section 2.2)

$$
p _ {L M L P} ^ {*} = \min  _ {\mathbf {Z} \in \mathbb {R} ^ {d \times c}} F (\mathbf {X Z}) + \beta \| \mathbf {Z} \| _ {*}, \tag {2}
$$

whereas for a ReLU activation and  $m \geq m^{*}$  for some  $m^{*} \leq nc$  ((Sahiner et al., 2021b), Thm. 3.1),

$$
p _ {R M L P} ^ {*} = \min  _ {\mathbf {Z} _ {j} \in \mathbb {R} ^ {d \times c}} F \left(\sum_ {j = 1} ^ {P} \mathbf {D} _ {j} \mathbf {X} \mathbf {Z} _ {j}\right) + \beta \sum_ {j = 1} ^ {P} \| \mathbf {Z} _ {j} \| _ {*}, \tag {3}
$$

$$
\mathbf {K} _ {j} := \left(2 \mathbf {D} _ {i} - \mathbf {I} _ {n}\right) \mathbf {X}
$$

where  $\{\mathbf{D}_j\}_{j=1}^P = \{\mathrm{diag}(\mathbb{1}\{\mathbf{X}\mathbf{u} \geq 0\}) : \mathbf{u} \in \mathbb{R}^d\}$  enumerates the possible activation patterns generated from  $\mathbf{X}$ , and the number of such patterns satisfies  $P \leq 2r\left(\frac{e(n-1)}{r}\right)^r$ , where  $r \coloneqq \operatorname{rank}(\mathbf{X})$  (Stanley et al., 2004; Pilanci & Ergen, 2020). The expression (3) also involves a constrained nuclear norm expression, which is defined as

$$
\left\| \mathbf {Z} \right\| _ {\ast , \mathrm {K}} := \min  _ {t \geq 0} t \text {s . t .} \mathbf {Z} \in t \mathcal {C} \tag {4}
$$

$$
\mathcal {C} := \operatorname {c o n v} \left\{\mathbf {Z} = \mathbf {u v} ^ {\top}: \mathbf {K u} \geq 0, \| \mathbf {u} \| _ {2} \leq 1, \| \mathbf {v} \| _ {2} \leq 1 \right\}.
$$

This norm is a quasi-nuclear norm, which differs from the standard nuclear norm in that the factorization upon which it relies imposes a constraint on its left factors. In convex ReLU neural networks, this norm enforces the existence of  $\{\mathbf{u}_k,\mathbf{v}_k\}$  such that  $\mathbf{Z} = \sum_{k}\mathbf{u}_{k}\mathbf{v}_{k}^{\top}$  and  $\mathbf{D}_j\mathbf{X}\mathbf{Z} = \sum_k(\mathbf{X}\mathbf{u}_k)_+\mathbf{v}_k^\top$  and penalizes  $\sum_{k}\| \mathbf{u}_{k}\mathbf{v}_{k}^{\top}\|_{*}$ . This norm is NP-hard to compute. A variant of these ReLU activations,

called gated ReLU activations, achieves the piecewise linearity of ReLU activations without enforcing the constraints (Fiat et al., 2019). Specifically, the ReLU gates are fixed to some  $\{\mathbf{h}_j\}_{j=1}^P$  to form

$$
\sigma \left(\mathbf {X} \mathbf {w} _ {1 j}\right) := \operatorname {d i a g} \left(\mathbb {1} \left\{\mathbf {X} \mathbf {h} _ {j} \geq 0 \right\}\right) \left(\mathbf {X} \mathbf {w} _ {1 j}\right) = \mathbf {D} _ {j} \mathbf {X} \mathbf {w} _ {1 j}. \tag {5}
$$

With gated ReLU activation, the equivalent convex program is given by ((Mishkin et al., 2022), Thm. 2.2; (Sahiner et al., 2022), e.q. (8))

$$
p _ {G M L P} ^ {*} = \min  _ {\mathbf {Z} _ {j} \in \mathbb {R} ^ {d \times c}} F \left(\sum_ {j = 1} ^ {P} \mathbf {D} _ {j} \mathbf {X} \mathbf {Z} _ {j}\right) + \beta \sum_ {j = 1} ^ {P} \| \mathbf {Z} _ {j} \| _ {*}, \tag {6}
$$

which thereby converts the constrained nuclear norm penalty to a standard nuclear norm penalty, thereby improving the complexity of the ReLU network. In addition to the multi-layer perceptron (MLP) formulation, two-layer ReLU-activation convolutional neural networks (CNNs) with global average pooling have been demonstrated to be equivalent to convex programs as well (Sahiner et al., 2021b;c; Ergen & Pilanci, 2020). The non-convex formulation is given by

$$
p _ {R C N N} ^ {*} := \min  _ {\substack {\mathbf {w} _ {1 j} \in \mathbb {R} ^ {h} \\ \mathbf {w} _ {2 j} \in \mathbb {R} ^ {c}}} \sum_ {i = 1} ^ {n} F (\sum_ {j = 1} ^ {m} \mathbf {w} _ {2 j} \mathbf {1} ^ {\top} (\mathbf {X} _ {i} \mathbf {w} _ {1 j}) _ {+}) + \frac {\beta}{2} \sum_ {j = 1} ^ {m} \| \mathbf {w} _ {1 j} \| _ {2} ^ {2} + \| \mathbf {w} _ {2 j} \| _ {2} ^ {2}, \tag{7}
$$

where samples  $\mathbf{X}_i\in \mathbb{R}^{K\times h}$  are represented by patch matrices, which hold a convolutional patch of size  $h$  in each of their  $K$  rows. It has been shown (Sahiner et al., 2021b) that as long as  $m\geq m^*$  where  $m^{*}\leq nc$ , this is equivalent to a convex program ((Sahiner et al., 2021b), Cor. 5.1)

$$
p _ {R C N N} ^ {*} = \min  _ {\mathbf {Z} _ {j} \in \mathbb {R} ^ {h \times c}} \sum_ {i = 1} ^ {n} F \left(\left(\sum_ {j = 1} ^ {P} \mathbf {1} ^ {\top} \mathbf {D} _ {j} ^ {(i)} \mathbf {X} _ {i} \mathbf {Z} _ {j}\right) ^ {\top}\right) + \beta \sum_ {j = 1} ^ {P} \| \mathbf {Z} _ {j} \| _ {* , \mathrm {K} _ {j}} \tag {8}
$$

$$
\mathbf {K} _ {j} := (2 \mathbf {D} _ {j} - \mathbf {I} _ {n K}) \mathbf {X},   \mathbf {X} := \left[ \begin{array}{c} \mathbf {X} _ {1} \\ \dots \\ \mathbf {X} _ {n} \end{array} \right]
$$

where  $\{\mathbf{D}_j\}_{j=1}^P = \{\mathrm{diag}(\mathbb{1}\{\mathbf{X}\mathbf{u} \geq 0\}) : \mathbf{u} \in \mathbb{R}^h\}$  and  $\mathbf{D}_j^{(i)} \in \mathbb{R}^{K \times K}$ . Since  $P$  is exponential of the rank of  $\mathbf{X}$ , for fixed filter size  $h$ ,  $P$  is polynomial in all other problem dimensions. Lastly, we review existing convexity results for self-attention transformers (Sahiner et al., 2022). We have the following non-convex objective for a single block of multi-head self-attention with  $m$  heads, where  $\mathbf{X}_i \in \mathbb{R}^{s \times d}$  with  $s$  tokens and  $d$  features

$$
p _ {S A} ^ {*} := \min  _ {\substack {\mathbf {W} _ {1 j} \in \mathbb {R} ^ {d \times d} \\ \mathbf {W} _ {2 j} \in \mathbb {R} ^ {d \times c}}} \sum_ {i = 1} ^ {n} F \left(\sum_ {j = 1} ^ {m} \sigma \left(\mathbf {X} _ {i} \mathbf {W} _ {1 j} \mathbf {X} _ {i} ^ {\top}\right) \mathbf {X} _ {i} \mathbf {W} _ {2 j}\right) + \frac {\beta}{2} \sum_ {j = 1} ^ {m} \| \mathbf {W} _ {1 j} \| _ {F} ^ {2} + \| \mathbf {W} _ {2 j} \| _ {F} ^ {2}, \tag{9}
$$

for which a variety of objectives  $F$  can be posed, including classification (e.g.  $F$  incorporates global average pooling followed by softmax-cross-entropy with labels) or denoising (e.g.  $F$  is a squared loss against a label matrix). In the linear activation case, as long as  $m \geq m^*$ , where  $m^* \leq \min\{d^2, dc\}$ , this is equivalent to ((Sahiner et al., 2022), Thm. 3.1)

$$
p _ {L S A} ^ {*} = \min  _ {\mathbf {Z} \in \mathbb {R} ^ {d ^ {2} \times d _ {c}}} \sum_ {i = 1} ^ {n} F \left(\sum_ {k = 1} ^ {d} \sum_ {\ell = 1} ^ {d} \mathbf {G} _ {i} [ k, \ell ] \mathbf {X} _ {i} \mathbf {Z} ^ {(k, \ell)}\right) + \beta \| \mathbf {Z} \| _ {*}, \tag {10}
$$

where  $\mathbf{G}_i\coloneqq \mathbf{X}_i^\top \mathbf{X}_i$ ,  $\mathbf{G}_i[k,l]\in \mathbb{R}$ , and  $\{\mathbf{Z}^{(k,\ell)}\in \mathbb{R}^{d\times c}\}$  are block matrices which form  $\mathbf{Z}$ . A similar formulation can be posed for ReLU and Gated ReLU activations. In this work, we show that these network architectures are amenable to the BM factorization.

# 2.2 THE BURER-MONTEIRO FACTORIZATION

First proposed by Burer & Monteiro (2003), the Burer-Monteiro (BM) factorization proposes to solve SDPs over some square matrix  $\mathbf{Q}$  in terms of rectangular factors  $\mathbf{R}$  where  $\mathbf{Q}$  is substituted by  $\mathbf{RR}^{\top}$ . It was first demonstrated that solving over  $\mathbf{R}$  does not introduce spurious local minima, given

$\mathrm{rank}(\mathbf{R}) \geq \mathrm{rank}(\mathbf{Q}^*)$  for optimal solution to the original SDP  $\mathbf{Q}^*$  (Burer & Monteiro, 2005). In general, we seek applications where we optimize over a non-square matrix  $\mathbf{Z}$ , i.e.

$$
p _ {C V X} ^ {*} := \min  _ {\mathbf {Z} \in \mathbb {R} ^ {d \times c}} F (\mathbf {Z}) \tag {11}
$$

for a convex, differentiable function  $F$ . One may approach this by factoring  $\mathbf{Z} = \mathbf{U}\mathbf{V}^{\top}$ , where  $\mathbf{U} \in \mathbb{R}^{d \times m}$ ,  $\mathbf{V} \in \mathbb{R}^{c \times m}$  for some arbitrary choice  $m$ . Then, we have an equivalent non-convex problem over  $\mathbf{R} := \begin{bmatrix} \mathbf{U} \\ \mathbf{V} \end{bmatrix}$ , for  $f(\mathbf{R}) = F(\mathbf{U}\mathbf{V}^{\top})$ :

$$
p _ {C V X} ^ {*} = \min  _ {\mathbf {R}} f (\mathbf {R}). \tag {12}
$$

Noting that (11) is convex over  $\mathbf{RR}^{\top} = \begin{bmatrix} \mathbf{UU}^{\top} & \mathbf{UV}^{\top} \\ \mathbf{VU}^{\top} & \mathbf{VV}^{\top} \end{bmatrix}$ , one may apply directly the result of Burer & Monteiro (2005) to conclude that as long as  $m \geq \mathrm{rank}(\mathbf{Z}^{*})$ , all local minima of (12) are global minima of (11) (see Appendix A.1). A major issue with these results is that  $\mathrm{rank}(\mathbf{Z}^{*})$  is not known a priori. Naively, one may simply choose  $m \geq \min\{d, c\}$  and be assured that  $m \geq \mathrm{rank}(\mathbf{Z}^{*})$ , but this approach is not satisfactory if further under-parameterization is desired. To address this issue, work from Bach et al. (2008) and Haeffele et al. (2014) demonstrates that all rank-deficient local minimizers of (12) achieve the global minimum  $p_{CVX}^{*}$  (under mild conditions, see Appendix A.2).

A long line of work has analyzed the conditions where known non-convex optimization algorithms will converge to second-order critical points (local minima) (Ge et al., 2015; Jin et al., 2017; Daneshmand et al., 2018). Under the assumption of a bounded  $f$  and its Hessian, a second-order critical point can be found by noisy gradient descent (Ge et al., 2015), or other second-order algorithms (Sun et al., 2015). Even vanilla gradient descent with random initialization has been demonstrated to almost surely converge to a local minimum for  $f$  with Lipschitz gradient (Lee et al., 2016). However, if the gradient of  $f$  is not Lipschitz-continuous, there are no guarantees that gradient descent will find a second-order critical point of (12): one may encounter a stationary point which is a saddle. For example, in the linear regression setting, i.e.

$$
f (\mathbf {R}) = \left\| \mathbf {X} \mathbf {U} \mathbf {V} ^ {\top} - \mathbf {Y} \right\| _ {F} ^ {2}, \tag {13}
$$

the gradient of  $f$  is Lipschitz continuous with respect to  $\mathbf{U}$  when  $\mathbf{V}$  is fixed and vice-versa, but not Lipschitz continuous with respect to  $\mathbf{R}$  (Mukkamala & Ochs, 2019). Thus, one may not directly apply the results of Ge et al. (2015); Sun et al. (2015); Lee et al. (2016) in this case. Instead, we seek to understand the conditions under which stationary points to (12) correspond to global optima of (11). One such condition is given in Mardani et al. (2013; 2015).

Theorem 2.1 (From (Mardani et al., 2013)). Stationary points  $\hat{\mathbf{U}},\hat{\mathbf{V}}$  of the optimization problem

$$
p ^ {*} := \min  _ {\mathbf {U}, \mathbf {V}} \frac {1}{2} \| \mathbf {U} \mathbf {V} ^ {\top} - \mathbf {Y} \| _ {F} ^ {2} + \frac {\beta}{2} \left(\| \mathbf {U} \| _ {F} ^ {2} + \| \mathbf {V} \| _ {F} ^ {2}\right) \tag {14}
$$

correspond to global optima  $\mathbf{Z}^{*} = \hat{\mathbf{U}}\hat{\mathbf{V}}^{\top}$  of the equivalent convex optimization problem

$$
p ^ {*} = \min  _ {\mathbf {Z}} \frac {1}{2} \| \mathbf {Z} - \mathbf {Y} \| _ {F} ^ {2} + \beta \| \mathbf {Z} \| _ {*} \tag {15}
$$

provided that  $\| \mathbf{Y} - \hat{\mathbf{U}}\hat{\mathbf{V}}^{\top}\|_{2}\leq \beta$

# 3 BURER-MONTEIRO FACTORIZATION FOR CONVEX NEURAL NETWORKS

# 3.1 MLPs

We first seek to compare the convex formulations of the MLP training problem (2), (3), and (6) to their BM factorizations. We describe how to find the BM factorization for any convex MLP.

Lemma 3.1. For any matrix  $\mathbf{M} \in \mathbb{R}^{n \times d_c}$ , let  $f(\mathbf{U}, \mathbf{V}) \coloneqq F(\mathbf{MUV}^\top)$  be a differentiable function. For any  $\beta > 0$  and arbitrary vector norms  $\| \cdot \|_R$  and  $\| \cdot \|_C$ , we define the Burer-Monteiro factorization

$$
p ^ {*} := \min  _ {\substack {\mathbf {U} \in \mathbb {R} ^ {d _ {c} \times m} \\ \mathbf {V} \in \mathbb {R} ^ {d _ {r} \times m}}} f (\mathbf {U}, \mathbf {V}) + \frac {\beta}{2} \left(\sum_ {j = 1} ^ {m} \| \mathbf {u} _ {j} \| _ {C} ^ {2} + \| \mathbf {v} _ {j} \| _ {R} ^ {2}\right). \tag{16}
$$

For the matrix norm  $\| \cdot \| _D$  defined as

$$
\left\| \mathbf {Z} \right\| _ {D} := \max  _ {\mathbf {R}} \operatorname {t r a c e} \left(\mathbf {R} ^ {\top} \mathbf {Z}\right) \text {s . t .} \mathbf {u} ^ {\top} \mathbf {R} \mathbf {v} \leq 1 \forall \mathbf {u} \in \mathcal {B} _ {C}, \forall \mathbf {v} \in \mathcal {B} _ {R}, \tag {17}
$$

the problem (16) is equivalent to the convex optimization problem

$$
p ^ {*} = \min  _ {\mathbf {Z} \in \mathbb {R} ^ {d _ {c} \times d _ {r}}} F (\mathbf {M Z}) + \beta \| \mathbf {Z} \| _ {D}. \tag {18}
$$

Remark 3.2. In the case of a linear MLP,  $\mathbf{M} = \mathbf{X}$ ,  $d_{c} = d$ ,  $d_{r} = c$ , and  $\| \cdot \| _D = \| \cdot \| _*$ , so using the definition of  $\| \cdot \| _D$ , in the corresponding BM factorization,  $R = 2$  and  $C = 2$  (Bach et al., 2008). For a gated ReLU network, the regularizer is still the nuclear norm, and thus the same  $R = C = 2$  regularization appears in the BM factorization. In the case of the ReLU MLP, the nuclear norm is replaced by  $\| \cdot \| _D = \sum_{j = 1}^P\| \cdot j\|_{*,\mathrm{K}_j}$ , which in the BM factorization amounts to having the constraint  $\mathbf{K}_j\mathbf{U}_j\geq \mathbf{0}$ . We accordingly express the BM factorization of convex MLPs below.

$$
p _ {L M L P} ^ {*} = \min  _ {\substack {\mathbf {U} \in \mathbb {R} ^ {d \times m} \\ \mathbf {V} \in \mathbb {R} ^ {c \times m}}} F (\mathbf {X U V} ^ {\top}) + \frac {\beta}{2} \left(\| \mathbf {U} \| _ {F} ^ {2} + \| \mathbf {V} \| _ {F} ^ {2}\right) \tag{19}
$$

$$
p _ {G M L P} ^ {*} = \min  _ {\substack {\mathbf {U} _ {j} \in \mathbb {R} ^ {d \times m} \\ \mathbf {V} _ {j} \in \mathbb {R} ^ {c \times m}}} F \left(\sum_ {j = 1} ^ {P} \mathbf {D} _ {j} \mathbf {X} \mathbf {U} _ {j} \mathbf {V} _ {j} ^ {\top}\right) + \frac {\beta}{2} \sum_ {j = 1} ^ {P} \left(\| \mathbf {U} _ {j} \| _ {F} ^ {2} + \| \mathbf {V} _ {j} \| _ {F} ^ {2}\right) \tag{20}
$$

$$
p _ {R M L P} ^ {*} = \min  _ {\substack {\mathbf {U} _ {j} \in \mathbb {R} ^ {d \times m}: (2 \mathbf {D} _ {j} - \mathbf {I} _ {n}) \mathbf {U} _ {j} \geq \mathbf {0} \\ \mathbf {V} _ {j} \in \mathbb {R} ^ {c \times m}}} F \left(\sum_ {j = 1} ^ {P} \mathbf {D} _ {j} \mathbf {X} \mathbf {U} _ {j} \mathbf {V} _ {j} ^ {\top}\right) + \frac {\beta}{2} \sum_ {j = 1} ^ {P} \left(\| \mathbf {U} _ {j} \| _ {F} ^ {2} + \| \mathbf {V} _ {j} \| _ {F} ^ {2}\right) \tag{21}
$$

To the best of our knowledge, (21) presents the first application of BM factorization to a non-linear neural network, which is enabled by the convex model (3).

In the linear case, the BM factorization (19) is identical to the original non-convex formulation of a linear MLP with  $m$  neurons. Furthermore, in the case of gated ReLU, the BM factorization when  $m = 1$  is equivalent to the original non-convex formulation. However, for ReLU activation two-layer networks, the BM factorization even when  $m = 1$  corresponds to a different (i.e. constrained, rather than ReLU activation) model than the non-convex formulation. While the original convex program is NP-hard, the computation of the cost function of the BM factorization is very simple. Thus, the per-iteration complexity of the BM factorization is much lower than for the convex ReLU MLP.

The BM factorizations of these convex MLPs are non-convex, hence finding a global minimum appears intractable. However, the following theorem demonstrates that as long as a rank-deficient local minimum to the BM factorization is obtained, it corresponds to a global optimum.

Theorem 3.3. If  $m \geq \operatorname{rank}(\mathbf{Z}^*)$ , where  $\mathbf{Z}^*$  is a minimizer of (18), all local minima of the BM factorization (16) are global minima. Furthermore, if  $F$  is twice-differentiable, any rank-deficient local minimum  $\hat{\mathbf{R}} := \begin{bmatrix} \hat{\mathbf{U}} \\ \hat{\mathbf{V}} \end{bmatrix}$  of (16) corresponds to a global minimizer  $\mathbf{Z}^* = \hat{\mathbf{U}}\hat{\mathbf{V}}^\top$  of (18).

This result demonstrates that these two-layer convex MLPs have no spurious local minima under mild conditions. However, there remains an algorithmic challenge: it is not straightforward to obtain a guaranteed local minima when the gradients of  $f$  are not Lipschitz continuous. The following result provides a general condition under which stationary points of the (16) are global optima of (18).

Theorem 3.4. For any non-negative objective function  $F$ , for a stationary  $(\hat{\mathbf{U}},\hat{\mathbf{V}})$  of (16) with corresponding  $\hat{\mathbf{Z}} = \hat{\mathbf{U}}\hat{\mathbf{V}}^{\top}$  with objective  $\hat{p}$  for (18), the relative optimality gap  $\frac{\hat{p} - p^{*}}{p^{*}}$  satisfies

$$
\frac {\hat {p} - p ^ {*}}{p ^ {*}} \leq \left(\frac {\| \nabla_ {\mathbf {Z}} F (\mathbf {M} \hat {\mathbf {Z}}) \| _ {D} ^ {*}}{\beta} - 1\right) _ {+} \tag {22}
$$

where  $\| \cdot \| _D^*$  is the dual norm of  $\| \cdot \| _D$

In the case of a linear MLP with  $\mathbf{X} = \mathbf{I}_d$ ,  $F$  is a squared-loss objective, and  $\| \nabla_{\mathbf{Z}} F(\mathbf{M}\hat{\mathbf{Z}}) \|_D^* \leq \beta$ , our result exactly replicates the result of Theorem 2.1 from Mardani et al. (2013). Furthermore, when this

condition is not exactly satisfied, (22) provides a novel result in the form of a optimality gap bound. To our knowledge, this is the first result of that generalizes the optimality conditions for stationary points from any BM factorization of a neural network. This provides an easily computable bound after solving (16) which quantifies how close a solution is to the global minimum. In the case of a ReLU MLP, the relative optimality gap is given by

$$
\frac {\hat {p} - p ^ {*}}{p ^ {*}} \leq \left(\max  _ { \begin{array}{l} j \in [ P ] \\ \mathbf {u} \in \mathcal {B} _ {2} \\ \mathbf {K} _ {j} \mathbf {u} \geq 0 \end{array} } \frac {1}{\beta} \| \nabla_ {\mathbf {Z} _ {j}} F \left(\sum_ {j ^ {\prime} = 1} ^ {P} \mathbf {D} _ {j ^ {\prime}} \mathbf {X} \hat {\mathbf {Z}} _ {j ^ {\prime}}\right) \mathbf {u} \| _ {2} - 1\right) + \tag {23}
$$

Computing this quantity amounts to solving a cone-constrained PCA problem (Deshpande et al., 2014), which can be done in polynomial-time when  $d$  is constant. We should note that some stationary points are clearly present in any problem, such as  $(\hat{\mathbf{U}},\hat{\mathbf{V}}) = (\mathbf{0},\mathbf{0})$ , so we cannot conclude that all stationary points are global optima. However, in certain cases, the optimality gap of stationary points (22) is always zero as we show next.

Theorem 3.5. A stationary point  $(\hat{\mathbf{U}},\hat{\mathbf{V}})$  of (16) is a global minimizer of (18) if  $R = C = 2$  and

$$
\operatorname {r a n k} (\hat {\mathbf {U}}) = \operatorname {r a n k} (\hat {\mathbf {V}}) = \min  \left\{d _ {c}, d _ {r} \right\}. \tag {24}
$$

Thus, for linear and gated ReLU MLPs, we can be assured that if the Burer-Monteiro factorization achieves a stationary point with full rank, it is corresponds with the global optimum of the convex program. We now can further extend these results to CNNs and self-attention architectures.

# 3.2 CNNs

Before proceeding to explore the BM factorization in the context of two-layer CNNs, we first provide a new result on an equivalent convex program for two-layer ReLU CNNs with arbitrary linear pooling operations, which extends the results of Sahiner et al. (2021b); Ergen & Pilanci (2020) on Global Average Pooling CNNs. Define  $\mathbf{P}_a \in \mathbb{R}^{a \times K}$  to be a linear pooling matrix which pools the  $K$  spatial dimensions to an arbitrary size  $a$ . Then, we express the non-convex two-layer CNN problem as

$$
p _ {C N N} ^ {*} := \min  _ {\substack {\mathbf {w} _ {1 j} \in \mathbb {R} ^ {h} \\ \mathbf {W} _ {2 j} \in \mathbb {R} ^ {c \times a}}} \sum_ {i = 1} ^ {n} F \left(\sum_ {j = 1} ^ {m} \mathbf {W} _ {2 j} \mathbf {P} _ {a} \sigma (\mathbf {X} _ {i} \mathbf {w} _ {1 j})\right) + \frac {\beta}{2} \sum_ {j = 1} ^ {m} \| \mathbf {w} _ {1 j} \| _ {2} ^ {2} + \| \mathbf {W} _ {2 j} \| _ {F} ^ {2}. \tag{25}
$$

Theorem 3.6. For  $\beta > 0$  and ReLU activation  $\sigma(\cdot) = (\cdot)_+$ , if  $m \geq m^*$  where  $m^* \leq nac$ , then (25) is equivalent to a convex optimization problem, given by

$$
p _ {C N N} ^ {*} = \min  _ {\mathbf {Z} _ {k} \in \mathbb {R} ^ {h \times a c}} \sum_ {i = 1} ^ {n} F \left(\sum_ {k = 1} ^ {P} \left[ \begin{array}{c} \operatorname {t r a c e} \left(\mathbf {P} _ {a} \mathbf {D} _ {k} ^ {(i)} \mathbf {X} _ {i} \mathbf {Z} _ {k} ^ {(1)}\right) \\ \vdots \\ \operatorname {t r a c e} \left(\mathbf {P} _ {a} \mathbf {D} _ {k} ^ {(i)} \mathbf {X} _ {i} \mathbf {Z} _ {k} ^ {(c)}\right) \end{array} \right]\right) + \beta \sum_ {k = 1} ^ {P} \| \mathbf {Z} _ {k} \| _ {*}, \tag {26}
$$

$$
\mathbf {K} _ {k} := \left(2 \mathbf {D} _ {k} - \mathbf {I} _ {n K}\right) \left[ \begin{array}{c} \mathbf {X} _ {1} \\ \dots \\ \mathbf {X} _ {n} \end{array} \right],   \mathbf {Z} _ {k} ^ {(c ^ {\prime})} \in \mathbb {R} ^ {h \times a}   \forall c ^ {\prime} \in [ c ].
$$

Thus, we provide a novel result which characterizes two-layer CNNs with arbitrary linear pooling operations as a convex program. Similar results can be shown for the linear and gated-ReLU activation cases<sup>1</sup>. With this established, we present our main results on the BM factorization for CNNs.

Lemma 3.7. The BM factorization of the convex CNN problem with ReLU activation is given as follows.

$$
p _ {R C N N} ^ {*} = \min  _ {\substack {\{\mathbf {u} _ {j k} \in \mathbb {R} ^ {h} \} _ {j = 1} ^ {m} \} _ {k = 1} ^ {P} \\ \{\mathbf {V} _ {j k} \in \mathbb {R} ^ {c \times a} \} _ {j = 1} ^ {m} \} _ {k = 1} ^ {P} \\ (2 \mathbf {D} _ {k} ^ {(i)} - \mathbf {I}) \mathbf {X} _ {i} \mathbf {u} _ {j k} \geq 0}} \sum_ {i = 1} ^ {n} F \left(\sum_ {k = 1} ^ {P} \sum_ {j = 1} ^ {m} \mathbf {V} _ {j k} \mathbf {P} _ {a} \mathbf {D} _ {k} ^ {(i)} \mathbf {X} _ {i} \mathbf {u} _ {j k}\right) + \frac {\beta}{2} \sum_ {k = 1} ^ {P} \sum_ {j = 1} ^ {m} \left(\| \mathbf {u} _ {j k} \| _ {F} ^ {2} + \| \mathbf {V} _ {j k} \| _ {F} ^ {2}\right) \tag{27}
$$

The BM factorization closely resembles the original non-convex formulation (25). Generally, (27) inherits the results of Theorems (3.3), (3.4), and (3.5); we present one such corollary here.

Corollary 3.7.1. A stationary point  $((\hat{\mathbf{u}}_{jk},\hat{\mathbf{V}}_{jk})_{j = 1}^{m})_{k = 1}^{P}$  of (27) corresponds to a global minimizer  $\hat{\mathbf{Z}}_k = \sum_{j = 1}^m\hat{\mathbf{u}}_{jk}\mathrm{vec}\left(\hat{\mathbf{V}}_{jk}\right)^\top$  of (26) provided that

$$
\left\| \right. \sum_ {i = 1} ^ {n} \nabla_ {\mathbf {Z} _ {k}} F \left(\sum_ {k ^ {\prime} = 1} ^ {P} \left[\begin{array}{c}\operatorname {t r a c e} \left(\mathbf {P} _ {a} \mathbf {D} _ {k ^ {\prime}} ^ {(i)} \mathbf {X} _ {i} \mathbf {Z} _ {k ^ {\prime}} ^ {(1)}\right)\\\vdots\\\operatorname {t r a c e} \left(\mathbf {P} _ {a} \mathbf {D} _ {k ^ {\prime}} ^ {(i)} \mathbf {X} _ {i} \mathbf {Z} _ {k ^ {\prime}} ^ {(c)}\right)\end{array}\right]\right) \mathbf {u} \| _ {2} \leq \beta , \forall k \in [ P ], \forall \mathbf {u} \in \mathcal {B} _ {2}: (2 \mathbf {D} _ {k} ^ {(i)} - \mathbf {I}) \mathbf {X} _ {i} \mathbf {u} \geq 0. \tag {28}
$$

# 3.3 MULTI-HEAD SELF-ATTENTION

We now for the first time extend BM factorization theory to self-attention networks.

Lemma 3.8. The BM factorization of the convex self-attention problem with linear activation<sup>2</sup> is given as follows.

$$
p _ {L S A} ^ {*} = \min  _ {\substack {\mathbf {U} _ {j} \in \mathbb {R} ^ {d \times d} \\ \mathbf {V} _ {j} \in \mathbb {R} ^ {d \times c}}} \sum_ {i = 1} ^ {n} F \left(\sum_ {j = 1} ^ {m} \mathbf {X} _ {i} \mathbf {U} _ {j} \mathbf {X} _ {i} ^ {\top} \mathbf {X} _ {i} \mathbf {V} _ {j}\right) + \frac {\beta}{2} \sum_ {j = 1} ^ {m} \| \mathbf {U} _ {j} \| _ {F} ^ {2} + \| \mathbf {V} _ {j} \| _ {F} ^ {2} \tag{29}
$$

In addition to inheriting all of the results of Theorems 3.3, 3.4, and 3.5, noting the equivalence of the BM factorization with the original non-convex program (9), we are the first to show conditions under which there are no spurious local minima for self-attention networks.

Corollary 3.8.1. The linear-activation self-attention network (29) has no spurious local minima as long as the number of heads satisfies  $m \geq m^{*}$  where  $m^{*} \leq \min\{d^{2}, dc\}$ . Furthermore, for any twice-differentiable objective  $F$ , if for any local minimum  $(\hat{\mathbf{U}}_{j}, \hat{\mathbf{V}}_{j})_{j=1}^{m}$  of (29), the matrix

$$
\hat {\mathbf {R}} := \left[ \begin{array}{l l l} \operatorname {v e c} \left(\hat {\mathbf {U}} _ {1}\right) & \dots & \operatorname {v e c} \left(\hat {\mathbf {U}} _ {m}\right) \\ \operatorname {v e c} \left(\hat {\mathbf {V}} _ {1}\right) & \dots & \operatorname {v e c} \left(\hat {\mathbf {V}} _ {m}\right) \end{array} \right] \in \mathbb {R} ^ {d (d + c) \times m} \tag {30}
$$

is rank-deficient, then this local minimum is also a global minimum of (10).

# 4 EXPERIMENTAL RESULTS: THE RELATIVE OPTIMALITY GAP BOUND

![](images/3b4517a476f8f5c311ef901e2f944cf714836535ed48748a9242ecbcf27a3927.jpg)  
(a)  $n = 45$

![](images/d104917b4600add50d59ecedf76dba2cda263f3c8e02a94682ecc012a97f8444.jpg)  
Figure 1: Example of three-class spiral dataset, with different number of samples  $n$ .  
(b)  $n = 150$

In this section, we illustrate the utility of our proposed relative optimality bound for stationary points in the setting of two-layer fully-connected networks. We also seek to examine how this bound changes with respect to the number of samples  $n$ , the regularization parameter  $\beta$  (which controls the sparsity of the convex solution), and the number of factors in the BM factorization

![](images/3197e23e44c09037308efb426cb492bd2fd325871b1bfa296f528a40eea5dcc7.jpg)

![](images/28497a00538f1a97bcd41d56c202e47f18294efab922d5223e2371abd293ab25.jpg)

![](images/28348eafdb7a62793e76e8bef52f050b34801a392b88eee5a6afd49c1b929801.jpg)  
(a)  $n = 15$  
(c)  $n = 75$

![](images/f5cfbc10c11ae260c1dc70a7e2e0a7cbdb230f8c1fe9cbc4e77225f4c09a1db0.jpg)  
Figure 2: Relative optimality gap of the non-convex BM factorization of a gated-ReLU two-layer MLP for three-class spiral data classification ( $d = 2$ ,  $c = 3$ ). For fixed values of  $n$ , we demonstrate how  $\beta$  and  $m$  affect relative optimality gap, both in terms of the proposed bound and the actual gap, where the global minimum is determined by convex optimization.  
(b)  $n = 45$  
(d)  $n = 150$

$m$ . We initialize a class-balanced three-class spiral data set with varied number of samples  $n$  (see Figure 1 for examples). For this dataset, we then train the gated ReLU MLP BM factorization (20) with varying number of factors  $m$ . We then compare the stationary points of these BM factorizations found by gradient descent (GD) to the global optimum, which we compute from (6).

For each stationary point of the BM factorization, we compute the relative optimality gap bound provided in our result in Theorem 3.4. We note that since  $d = 2$ ,  $c = 3$  in this case, for all  $j$ ,  $\mathrm{rank}(\mathbf{Z}_j^*) \leq 2$ , so as long as  $m \geq 2$  all local minima of the BM factorization are global minima (Burer & Monteiro, 2005; Haeffele et al., 2014). While Lee et al. (2016) demonstrated that gradient descent with a random initialization converges to a local optimum almost surely for losses  $f$  whose gradient is Lipschitz continuous, we use squared loss with one-hot-encoded class labels, for which  $f$  is not Lipschitz continuous (Mukkamala & Ochs, 2019). Thus, there is no guarantee that GD will find the global minimum. We display results over  $\beta$  for each fixed  $n$  in Figure 2. For larger values of  $\beta$ , it becomes much easier for GD to find an optimal solution. We nevertheless find that our bound gives a useful proxy for whether the BM factorization has converged to the global minimum.

We find that GD applied to the BM factorization finds "subtle" saddle points: not quite local minima, but close. Interestingly, there is only a minor relationship between the optimality gap and the size of the BM factorization  $m$ . While our optimality gap bound for  $m = 1$  is larger than larger values of  $m$  for small  $\beta$ , the actual optimality gap is nearly identical across  $m$ . This experiment further validates the need to consider stationary points of the BM factorization, rather than just local minima, to fully characterize the BM factorization for efficient solutions to convex problems.

# 5 CONCLUSION

We are the first to adapt the Burer-Monteiro (BM) factorization for two-layer convex neural networks with linear and ReLU activations, which offers new insights on their global optima. We provide a novel relative optimality bound on stationary point of the BM factorization, which provides a condition whose satisfaction guarantees a globally optimal solution.

# REFERENCES

Erling D Andersen and Knud D Andersen. The mosek interior point optimizer for linear programming: an implementation of the homogeneous algorithm. In High performance optimization, pp. 197-232. Springer, 2000.  
Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In International Conference on Machine Learning, pp. 322-332. PMLR, 2019.  
Francis Bach, Julien Mairal, and Jean Ponce. Convex sparse matrix factorizations. arXiv preprint arXiv:0812.1869, 2008.  
Yatong Bai, Tanmay Gautam, and Somayeh Sojoudi. Efficient global optimization of two-layer relu networks: Quadratic-time algorithms and adversarial training. arXiv preprint arXiv:2201.01965, 2022.  
Burak Bartan and Mert Pilanci. Neural spectrahedra and semidefinite lifts: Global convex optimization of polynomial activation neural networks in fully polynomial-time. arXiv preprint arXiv:2101.02429, 2021.  
Eugene Belilovsky, Michael Eickenberg, and Edouard Oyallon. Greedy layerwise learning can scale to imagenet. In International conference on machine learning, pp. 583-593. PMLR, 2019.  
Shobhit Bhatnagar, Deepanway Ghosal, and Maheshkumar H Kolekar. Classification of fashion article images using convolutional neural networks. In 2017 Fourth International Conference on Image Information Processing (ICIIP), pp. 1-6. IEEE, 2017.  
Nicolas Boumal, Vlad Voroninski, and Afonso Bandeira. The non-convex burer-monteiro approach works on smooth semidefinite programs. Advances in Neural Information Processing Systems, 29, 2016.  
Samuel Burer and Renato DC Monteiro. A nonlinear programming algorithm for solving semidefinite programs via low-rank factorization. Mathematical Programming, 95(2):329-357, 2003.  
Samuel Burer and Renato DC Monteiro. Local minima and convergence in low-rank semidefinite programming. Mathematical programming, 103(3):427-444, 2005.  
Ricardo Cabral, Fernando De la Torre, João P Costeira, and Alexandre Bernardino. Unifying nuclear norm and bilinear factorization approaches for low-rank matrix decomposition. In Proceedings of the IEEE international conference on computer vision, pp. 2488-2495, 2013.  
Emmanuel J Candès and Benjamin Recht. Exact matrix completion via convex optimization. Foundations of Computational mathematics, 9(6):717-772, 2009.  
Emmanuel J Candès and Terence Tao. The power of convex relaxation: Near-optimal matrix completion. IEEE Transactions on Information Theory, 56(5):2053-2080, 2010.  
Diego Cifuentes and Ankur Moitra. Polynomial time guarantees for the burer-monteiro method. arXiv preprint arXiv:1912.01745, 2019.  
Hadi Daneshmand, Jonas Kohler, Aurelien Lucchi, and Thomas Hofmann. Escaping saddles with stochastic gradients. In International Conference on Machine Learning, pp. 1155-1164. PMLR, 2018.  
Yash Deshpande, Andrea Montanari, and Emile Richard. Cone-constrained principal component analysis. Advances in Neural Information Processing Systems, 27, 2014.  
Steven Diamond and Stephen Boyd. Cvxpy: A python-embedded modeling language for convex optimization. The Journal of Machine Learning Research, 17(1):2909–2913, 2016.  
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. arXiv preprint arXiv:1605.08803, 2016.

Simon Du and Jason Lee. On the power of over-parametrization in neural networks with quadratic activation. In International conference on machine learning, pp. 1329-1338. PMLR, 2018.  
Simon Du, Jason Lee, Yuandong Tian, Aarti Singh, and Barnabas Poczos. Gradient descent learns one-hidden-layer cnn: Don't be afraid of spurious local minima. In International Conference on Machine Learning, pp. 1339-1348. PMLR, 2018.  
Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In International conference on machine learning, pp. 1675-1685. PMLR, 2019.  
Simon S Du, Chi Jin, Jason D Lee, Michael I Jordan, Aarti Singh, and Barnabas Poczos. Gradient descent can take exponential time to escape saddle points. Advances in neural information processing systems, 30, 2017.  
Murat A Erdogdu, Asuman Ozdaglar, Pablo A Parrilo, and Nuri Denizcan Vanli. Convergence rate of block-coordinate maximization burer-monteiro method for solving large sdps. Mathematical Programming, pp. 1-39, 2021.  
Tolga Ergen and Mert Pilanci. Implicit convex regularizers of cnn architectures: Convex optimization of two- and three-layer networks in polynomial time. In International Conference on Learning Representations, 2020.  
Tolga Ergen, Arda Sahiner, Batu Ozturkler, John M Pauly, Morteza Mardani, and Mert Pilanci. Demystifying batch normalization in relu networks: Equivalent convex optimization models and implicit regularization. In International Conference on Learning Representations, 2021.  
Jonathan Fiat, Eran Malach, and Shai Shalev-Shwartz. Decoupling gating from linearity. arXiv preprint arXiv:1906.05032, 2019.  
Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points—online stochastic gradient for tensor decomposition. In Conference on learning theory, pp. 797–842. PMLR, 2015.  
Rong Ge, Chi Jin, and Yi Zheng. No spurious local minima in nonconvex low rank problems: A unified geometric analysis. In International Conference on Machine Learning, pp. 1233-1242. PMLR, 2017.  
Nicolas Gillis. Introduction to nonnegative matrix factorization. arXiv preprint arXiv:1703.00663, 2017.  
Benjamin Haeffele, Eric Young, and Rene Vidal. Structured low-rank matrix factorization: Optimality, algorithm, and applications to image processing. In International conference on machine learning, pp. 2007-2015. PMLR, 2014.  
Benjamin D Haeffele and René Vidal. Global optimality in neural network training. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7331-7339, 2017.  
Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M Kakade, and Michael I Jordan. How to escape saddle points efficiently. In International Conference on Machine Learning, pp. 1724-1732. PMLR, 2017.  
Kenji Kawaguchi. Deep learning without poor local minima. Advances in neural information processing systems, 29, 2016.  
Serhat Kiliçarslan and Mete Celik. Rsigelu: A nonlinear activation function for deep neural networks. Expert Systems with Applications, 174:114805, 2021.  
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
Jason D Lee, Max Simchowitz, Michael I Jordan, and Benjamin Recht. Gradient descent only converges to minimizers. In Conference on learning theory, pp. 1246-1257. PMLR, 2016.  
Jan R Magnus and Heinz Neudecker. Matrix differential calculus with applications in statistics and econometrics. John Wiley & Sons, 2019.

Morteza Mardani, Gonzalo Mateos, and Georgios B Giannakis. Decentralized sparsity-regularized rank minimization: Algorithms and applications. IEEE Transactions on Signal Processing, 61(21): 5374-5388, 2013.  
Morteza Mardani, Gonzalo Mateos, and Georgios B Giannakis. Subspace learning and imputation for streaming big data matrices and tensors. IEEE Transactions on Signal Processing, 63(10): 2663-2677, 2015.  
Aaron Mishkin, Arda Sahiner, and Mert Pilanci. Fast convex optimization for two-layer relu networks: Equivalent model classes and cone decompositions. In Proceedings of the 39th International Conference on Machine Learning, 2022.  
Mahesh Chandra Mukkamala and Peter Ochs. Beyond alternating updates for matrix factorization with inertial bregman proximal gradient algorithms. Advances in Neural Information Processing Systems, 32, 2019.  
Dohyung Park, Anastasios Kyrillidis, Constantine Carmanis, and Sujay Sanghavi. Non-square matrix sensing without spurious local minima via the burer-monteiro approach. In Artificial Intelligence and Statistics, pp. 65-74. PMLR, 2017.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32: 8026-8037, 2019.  
Mert Pilanci and Tolga Ergen. Neural networks are convex regularizers: Exact polynomial-time convex optimization formulations for two-layer networks. In International Conference on Machine Learning, pp. 7695-7705. PMLR, 2020.  
Benjamin Recht, Maryam Fazel, and Pablo A Parrilo. Guaranteed minimum-rank solutions of linear matrix equations via nuclear norm minimization. SIAM review, 52(3):471-501, 2010.  
Jasson DM Rennie and Nathan Srebro. Fast maximum margin matrix factorization for collaborative prediction. In Proceedings of the 22nd international conference on Machine learning, pp. 713-719, 2005.  
Arda Sahiner, Tolga Ergen, Batu Ozturkler, Burak Bartan, John M Pauly, Morteza Mardani, and Mert Pilanci. Hidden convexity of wasserstein gans: Interpretable generative models with closed-form solutions. In International Conference on Learning Representations, 2021a.  
Arda Sahiner, Tolga Ergen, John M Pauly, and Mert Pilanci. Vector-output relu neural network problems are copositive programs: Convex analysis of two layer networks and polynomial-time algorithms. In ICLR, 2021b.  
Arda Sahiner, Morteza Mardani, Batu Ozturkler, Mert Pilanci, and John M Pauly. Convex regularization behind neural reconstruction. In ICLR, 2021c.  
Arda Sahiner, Tolga Ergen, Batu Ozturkler, John Pauly, Morteza Mardani, and Mert Pilanci. Unraveling attention via convex duality: Analysis and interpretations of vision transformers. In Proceedings of the 39th International Conference on Machine Learning, 2022.  
Alexander Shapiro. Semi-infinite programming, duality, discretization and optimality conditions. Optimization, 58(2):133-161, 2009.  
Richard P Stanley et al. An introduction to hyperplane arrangements. Geometric combinatorics, 13 (389-496):24, 2004.  
Ju Sun, Qing Qu, and John Wright. When are nonconvex problems not scary? arXiv preprint arXiv:1510.06096, 2015.  
Irene Waldspurger and Alden Waters. Rank optimality for the burer-monteiro factorization. SIAM journal on Optimization, 30(3):2577-2602, 2020.

Lingxiao Wang, Xiao Zhang, and Quanquan Gu. A unified computational and statistical framework for nonconvex low-rank matrix estimation. In Artificial Intelligence and Statistics, pp. 981-990. PMLR, 2017.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.  
Tian Ye and Simon S Du. Global convergence of gradient descent for asymmetric low-rank matrix factorization. Advances in Neural Information Processing Systems, 34, 2021.  
Qinqing Zheng and John Lafferty. Convergence analysis for rectangular matrix completion using burer-monteiro factorization and gradient descent. arXiv preprint arXiv:1605.07051, 2016.  
Zhihui Zhu, Qiuwei Li, Gongguo Tang, and Michael B Wakin. The global optimization geometry of low-rank matrix optimization. arXiv preprint arXiv:1703.01256, 2017.  
Liu Ziyin, Botao Li, James B Simon, and Masahito Ueda. Sgd can converge to local maxima. In International Conference on Learning Representations, 2021.
