# LINEAR CONVERGENCE OF SGD ON OVERPARAMETRIZED SHALLOW NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Despite the non-convex landscape, first-order methods can be shown to reach global minima when training overparameterized neural networks, where the number of parameters far exceeds the number of training data. In this work, we prove linear convergence of stochastic gradient descent when training a two-layer neural network with smooth activations. While the existing theory either requires a high degree of overparameterization or non-standard initialization and training strategies, e.g., training only a single layer, we show that a subquadratic scaling on the width is sufficient under standard initialization and training both layers simultaneously if the minibatch size is sufficiently large and it also grows with the number of training examples. Via the batch size, our results interpolate between the state-of-the-art subquadratic results for gradient descent and the quadratic results in the worst case.

# 1 INTRODUCTION

Our understanding of the optimization landscape of supervised learning with neural networks has vastly improved in recent years. This is in part due to the observation that overparameterization is key to overcome the pitfalls of first-order methods in general non-convex problems (Soltanolkotabi et al., 2019). Under this assumption, a line of research has established convergence of first-order methods such as gradient descent (GD) to global optimality, (Allen-Zhu et al., 2019; Kawaguchi & Huang, 2019; Du et al., 2019; Du & Lee, 2018; Zou & Gu, 2019; Brutzkus & Globerson, 2017; Song & Yang, 2019; Oymak & Soltanolkotabi, 2020), a phenomenon that has been confirmed in practice.

Empirically, as long as the width of a network scales linearly with the size of the training data (mild overparameterization), stochastic gradient descent (SGD) enjoys fast convergence to global optimality (Livni et al., 2014; Safran & Shamir, 2018; Oymak & Soltanolkotabi, 2020; Kawaguchi & Huang, 2019). Can we explain such behavior theoretically? Sadly, the available characterizations require a larger degree of overparameterization, or imposes additional assumptions, which do not hold for the algorithms that are used in practice. For example, if GD is applied exclusively to the last layer, Kawaguchi & Huang (2019) show that an ideal linear scaling of the width is sufficient to guarantee convergence. Song & Yang (2019) prove quadratic scaling when GD is applied only to the first layer.

For two-layer neural networks, when both layers are trained with GD simultaneously, state-of-the-art results show that subquadratic (not linear) scaling is enough to converge to global optimality (Anonymous). Despite being close to the ideal linear rate of overparameterization, due to computational constraints, GD is rarely used in modern applications involving huge datasets. Hence, closing the gap between theory and practice requires studying scalable first-order algorithms such as SGD. Our work focuses on mini-batch SGD, which is one of the most common algorithms for training deep models. We study convergence of SGD when it is applied to train both layers of a neural network, which is initialized with standard initialization schemes.

# Our contributions:

- We show that under proper initialization and choice of learning rate, the iterates of SGD converge to a global minimum with high probability and exponentially fast for a general non-convex problem assuming that the loss function satisfies a growth condition.  
- For the special case a two-layer neural network, we show that a subquadratic scaling on the width is sufficient under standard initialization and training both layers simultaneously, if the

Table 1: Required degree of Overparameterization for training shallow networks with global convergence guarantees.  $\mathrm{{QL}} =$  quadratic loss,  $\mathrm{{CLL}} =$  convex and Lipschitz loss,  $\mathrm{{SD}} =$  separable data. The notation  $\widetilde{\Omega }$  ignores logarithmic factors.  

<table><tr><td>Reference</td><td>Algorithm</td><td>Activation</td><td>Setting</td><td>Scaling</td></tr><tr><td>Oymak &amp; Soltanolkotabi (2020)</td><td>SGD on layer 1</td><td>ReLU</td><td>QL</td><td>\(\tilde{\Omega}(n^{2})\)</td></tr><tr><td>Song &amp; Yang (2019)</td><td>GD on layer 1</td><td>ReLU</td><td>SD</td><td>\(\tilde{\Omega}(n^{2})\)</td></tr><tr><td>Kawaguchi &amp; Huang (2019)</td><td>GD on layer 2</td><td>ReLU</td><td>CLL</td><td>\(\tilde{\Omega}(n)\)</td></tr><tr><td>Du et al. (2019)</td><td>GD</td><td>ReLU</td><td>SD+QL</td><td>\(\tilde{\Omega}(n^{6})\)</td></tr><tr><td>Zou &amp; Gu (2019)</td><td>GD</td><td>ReLU</td><td>SD+QL</td><td>\(\Omega(n^{8})\)</td></tr><tr><td>Anonymous</td><td>GD</td><td>smooth</td><td>QL</td><td>\(\tilde{\Omega}(n^{\frac{3}{2}})\) or \(\tilde{\Omega}(n^{2})\)</td></tr><tr><td>Allen-Zhu et al. (2019)</td><td>SGD</td><td>ReLU</td><td>SD+QL</td><td>\(\Omega(n^{24})\)</td></tr><tr><td>This paper</td><td>SGD</td><td>smooth</td><td>QL</td><td>\(\tilde{\Omega}(n^{\frac{3}{2}})\) or \(\tilde{\Omega}(n^{2})\)</td></tr></table>

minibatch size is sufficiently large and it also grows with the number of training examples. For constant batch size, we show that quadratic overparametrization is sufficient. Our results interpolate between subquadratic and quadratic scalings depending on the batch size.

Related work. The majority of the existing literature on overparameterization focuses on GD (Du et al., 2019; Du & Lee, 2018; Allen-Zhu et al., 2019; Zou & Gu, 2019). Allen-Zhu et al. (2019) provided theoretical bounds for deep networks trained with SGD. However, their results require an overparameterization degree that is too large, compared to what can be achieved for GD. In contrast, we study SGD and how the batch size affects the required degree of overparameterization. Chen et al. (2021) establish generalization guarantees and sufficient network width when SGD trains deep ReLU networks for binary classification, which is a different setting compared to our paper.

We study the case where SGD updates all the parameters of a shallow neural network. In contrast, a number of existing literature assume that only the parameters corresponding to some layers are updated throughout training (Oymak & Soltanolkotabi, 2020; Kawaguchi & Huang, 2019; Song & Yang, 2019). When SGD is applied only to the first layer, Oymak & Soltanolkotabi (2020) showed that quadratic scaling is sufficient for convergence with linear rate. Despite being an interesting theoretical setup, such algorithmic choice rarely happens in practice.

There are also differences regarding the choice of activation function. While ReLU can be considered as the default activation function when studying deep neural networks, its non-smoothness may be the reason why results for ReLU networks require substantially more number of parameters or additional assumptions on the data (like separability) to guarantee convergence to a global minimum. Moreover, backpropagation on ReLU networks does not correctly calculate the gradient at all points of differentiability (Kakade & Lee, 2018; Bolte & Pauwels, 2021), which raises major technical issues. In contrast, we assume a smooth activation, similar to Anonymous, which avoids such issues and achieves lower overparameterization degrees.

The authors of (Anonymous) established subquadratic scaling when GD trains a shallow neural network. In this paper, we focus on SGD, which results in substantial technical challenges. Compared to the results in (Anonymous), controlling the length of the trajectory is more involved in this paper, which requires a new analysis technique that bounds the length of the trajectory with high probability. We consider the effect of mini-batch SGD, which shows an interpolation between subquadratic and quadratic scaling. We also improve the estimates at initialization and show that more relaxed assumptions are sufficient to establish sufficient overparameterization degree.

We summarize such recent results in the overparametrization literature in Table 1.

Lazy Training. Proving fast convergence to global optimality is not a complete answer. It has been shown that despite fast convergence, it is possible that an algorithm tends towards a solution with poor performance on test data, if the training falls in the so-called Lazy Training regime (Chizat et al., 2019). Thus, any useful algorithmic framework for learning neural networks should avoid this regime, usually through careful initialization schemes. For example, despite requiring only linear overparametrization for GD, the initialization studied by Nguyen & Mondelli (2020) leads to

the lazy regime. This is the reason why we omit such result from Table 1. In this work, we show global convergence and achieve subquadratic scaling under standard initialization schemes, which empirically perform well on test data.

Polylogarithmic width is enough to obtain convergence for neural networks of arbitrary depth, according to Ji & Telgarsky (2020); Chen et al. (2021). However, in those work, convergence is understood in an ergodic sense. This is a weaker notion than strict convergence with high probability, which is the one we consider, and which better matches practical applications.

Given perfect knowledge about the underlying function that generates the labels and under the assumption that such target function has low-rank approximation, Su & Yang (2019) showed that GD achieves zero-approximation. This is different from the problem considered in our paper.

For a binary classification problem, Daniely (2020) showed that near linear network size is sufficient for SGD to memorize random examples under a variant of Xavier initialization, which is a different setting compared to our paper. For a deep neural network with pyramidal structure and smooth activations, Nguyen & Mondelli (2020) showed that subquadratic scaling is sufficient for global convergence of GD under a restrictive initialization scheme. In this paper, we establish global convergence for SGD under standard initialization.

A recent line of work uses mean-field analysis to approximate the target distribution of the weights in a neural network via their empirical distribution (Mei et al., 2019; Lu et al., 2020). Nevertheless, such results do not provide useful overparametrization degree bounds in terms of the number of samples. In contrast, our work does not require such approximations and we focus on deriving explicit sufficient overparametrization rates for global optimality of SGD.

Notation. We use  $\|\cdot\|$  to denote the Euclidean norm of a vector and Frobenius norm of a matrix. We use  $\nabla$  to represent the Jacobian of a vector-valued and gradient of a scalar-valued function. We use  $\odot$  and  $\otimes$  to represent the entry-wise Hadamard product and Kronecker product, respectively. We use lower-case bold font to denote vectors. We use calligraphic and standard fonts to represent sets and scalars, respectively. We use  $\sigma_{\min}(T)$  and  $\sigma_{\max}(T)$  to denote the smallest and largest singular values of a linear map  $T$ . We use  $[n]$  to represent  $\{1, \dots, n\}$  for an integer  $n$ . We use  $\tilde{\mathcal{O}}$  and  $\tilde{\Omega}$  to hide logarithmic factors and use  $\lesssim$  to ignore terms up to constant and logarithmic factors.

# 2 A GENERAL GLOBAL CONVERGENCE RESULT FOR SGD

In this section, we consider a general non-convex minimization problem and show that for a certain choice of learning rate and careful initialization, the iterates of SGD converge to a global minimum with high probability and exponentially fast. In Section 3, we extend our consideration to the training of a shallow neural network and find the hidden layer size, which is sufficient for SGD to converge to a global minimum i.e., its overparameterization degree.

Definition 1 (Smoothness). Let  $\beta_{\psi} > 0$ . A function  $\psi : \mathbb{R}^{d_1} \to \mathbb{R}^{d_2}$  is  $\beta_{\psi}$ -smooth, if for all  $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{d_1}$ , we have

$$
\sigma_ {\max } (\nabla \psi (\mathbf {u}) - \nabla \psi (\mathbf {v})) \leq \beta_ {\psi} \| \mathbf {u} - \mathbf {v} \|. \tag {1}
$$

Definition 2 (PL condition (Bolte et al., 2017)). A function  $\psi : \mathbb{R}^{d_1} \to \mathbb{R}$  satisfies the PL condition if there exists  $\alpha_{\psi} > 0$  such that, for all  $\mathbf{u} \in \mathbb{R}^{d_1}$ , we have

$$
\psi (\mathbf {u}) \leq \frac {\left\| \nabla \psi (\mathbf {u}) \right\| ^ {2}}{2 \alpha_ {\psi}}. \tag {2}
$$

We are now ready to state our finite-sum compositional optimization problem:

$$
\min  _ {\mathbf {w} \in \mathbb {R} ^ {d}} \left\{h (\mathbf {w}) := f \left(\Phi (\mathbf {w})\right) = \frac {1}{m} \sum_ {j = 1} ^ {m} f _ {j} \left(\Phi (\mathbf {w})\right)\right\}, \quad \Phi : \mathbb {R} ^ {d} \rightarrow \mathbb {R} ^ {\tilde {d}}, \quad f, f _ {j}: \mathbb {R} ^ {\tilde {d}} \rightarrow \mathbb {R} _ {+} \tag {3}
$$

where  $m$  denotes the number of training examples.

Assumption 1. The functions introduced in Eq. (3) satisfy the following properties: (i)  $\Phi$  is twice-differentiable and  $\beta_{\Phi}$ -smooth (Definition 1), (ii)  $f$  is twice-differentiable and  $\beta_{f}$ -smooth and (iii)  $f$  satisfies the PL condition with some  $\alpha_{f} > 0$  (Definition 2).

We study the iterates of the stochastic gradient descent (SGD) algorithm when applied to the objective function  $h$  in Eq. (3). For  $i \geq 0$ , let  $\mathcal{I}^i$  denote a random minibatch at iteration  $i$  drawn uniformly at random, independent of all previous draws. Let  $b \in [m]$  denote the minibatch size, i.e.,  $|\mathcal{I}^i| = b$  for all  $i$ . The SGD iterates are defined by a random variable  $\mathbf{w}^0$ , referred to as the initialization, and the update rule:

$$
\mathbf {w} ^ {i + 1} = \mathbf {w} ^ {i} - \lambda \frac {1}{b} \sum_ {j \in \mathcal {I} ^ {i}} \nabla h _ {j} \left(\mathbf {w} ^ {i}\right), \tag {4}
$$

where  $\lambda > 0$  is the learning rate and  $h_j(\mathbf{w}) \coloneqq f_j(\Phi(\mathbf{w})) \forall j, \mathbf{w}$ .

An important feature of the SGD iterates is that  $\frac{1}{b}\sum_{j\in \mathcal{I}^i}\nabla h_j(\Phi (\mathbf{w}^i))$  in Eq. (4) is an unbiased estimator of the gradient  $\nabla h(\mathbf{w}^i)$  given  $\mathbf{w}^i$ , i.e.,  $\mathbb{E}\left[\frac{1}{b}\sum_{j\in \mathcal{I}^i}\nabla h_j(\Phi (\mathbf{w}^i))|\mathbf{w}^i\right] = \nabla h(\mathbf{w}^i)$ . Nevertheless, this is not enough for SGD to converge to the first-order optimality. In addition, we will assume that  $f$  in Eq. (3) satisfies the growth condition (Schmidt & Roux, 2013; Vaswani et al., 2019; Cevher & Vu, 2019):

Definition 3 (Growth condition). A function  $\psi : \mathbb{R}^d \to \mathbb{R}$  with a finite-sum structure satisfies the growth condition with minibatch size  $b$  if there exists  $\eta_{\psi} > 0$  such that, for all  $\mathbf{u} \in \mathbb{R}^d$ , we have

$$
\mathbb {E} \left[ \left\| \frac {1}{c} \sum_ {j \in \mathcal {I}} \nabla \psi_ {j} (\mathbf {u}) \right\| ^ {2} \right] \leq \eta_ {\psi} \| \nabla \psi (\mathbf {u}) \| ^ {2}, \tag {5}
$$

where the expectation is over the random choice of set  $\mathcal{I}$ .

Assumption 2. In Eq. (3),  $f$  satisfies the growth condition (Definition 3) for some  $\eta_f > 0$ .

We are now ready to state the main result of this section.

Theorem 1. Let Assumptions 1 and 2 hold and let  $\zeta > 1$ . Suppose that at initialization,

$$
0 <   \mu_ {\Phi} \leq \sigma_ {\min } \left(\nabla \Phi^ {*} \left(\mathbf {w} ^ {0}\right)\right) \leq \sigma_ {\max } \left(\nabla \Phi^ {*} \left(\mathbf {w} ^ {0}\right)\right) \leq \nu_ {\Phi}, \quad h \left(\mathbf {w} ^ {0}\right) = O \left(\frac {\alpha_ {f} \mu_ {\Phi} ^ {6}}{\zeta \beta_ {\Phi} ^ {2} \eta_ {f} \nu_ {\Phi} ^ {2}}\right). \tag {6}
$$

Then, for a sufficiently small learning rate

$$
\lambda \lesssim \min  \left(\frac {\mu_ {\Phi} ^ {2}}{\eta_ {f} \left(\beta_ {\Phi} \nu_ {\Phi} ^ {2} \| \nabla f (\Phi (\mathbf {w} ^ {0})) \| + \beta_ {f} \nu_ {\Phi} ^ {4} + \beta_ {f} \mu_ {\Phi} \nu_ {\Phi} ^ {3}\right)}, \frac {\mu_ {\Phi}}{\zeta \sqrt {\eta_ {f}} \nu_ {\Phi} \left(\beta_ {\Phi} \| \nabla f (\Phi (\mathbf {w} ^ {0})) \| + \beta_ {f} \nu_ {\Phi} \mu_ {\Phi}\right)}\right) \tag {7}
$$

the iterates of  $SGD\left\{\mathbf{w}^i\right\}_{i \geq 0}$  (4) converge to a global minimizer of  $h$  (3) with the optimal value of zero, exponentially fast and with probability at least  $1 - 1/\zeta$ . The rate of convergence is given by

$$
\mathbb {E} [ h (\mathbf {w} ^ {i}) ] \leq (1 - C \lambda \alpha_ {f} \mu_ {\Phi} ^ {2}) ^ {i} \cdot h (\mathbf {w} ^ {0})
$$

for a universal constant  $C$ .

Remark 1. The second item in Eq. (6) suggests initializing close to a global minimum of the nonconvex optimization problem. This feature has precedence in the related literature, e.g., in matrix factorization (Chi et al., 2019).

The proof of Theorem 1 is deferred to Appendix B. However, in the remaining of this section we provide a sketch of the main arguments that lead to the result. The first condition in Eq. (6) is central to our arguments, and we will refer to it as the near-isometry property.

Definition 4 (Near-isometry). A linear map  $T: \mathbb{R}^{d_1} \to \mathbb{R}^{d_2}$  is  $(\mu, \nu)$ -near-isometry if there exist  $0 < \mu \leq \nu$  such that

$$
\mu \leq \sigma_ {\min } (T) \leq \sigma_ {\max } (T) \leq \nu . \tag {8}
$$

Let  $\overline{\mathbf{w}}$  denote the limit point when the SGD algorithm is run with some learning rate and let  $\nabla \Phi^{*}(\overline{\mathbf{w}})$  denote the adjoint operator of  $\nabla \Phi (\overline{\mathbf{w}})$ . Convergence of SGD is ensured with high probability due to the strong growth condition (Definition 3) along with proper learning rate and initialization. We note that  $\overline{\mathbf{w}}$  is a first-order stationary point of  $h$ . Hence we have:

$$
0 = \nabla h (\overline {{\mathbf {w}}}) = \nabla \Phi^ {*} (\overline {{\mathbf {w}}}) \nabla f (\Phi (\overline {{\mathbf {w}}})) \tag {9}
$$

Note that if  $\nabla \Phi^{*}(\overline{\mathbf{w}})$  is nonsingular, it would follow that  $\nabla f(\Phi (\overline{\mathbf{w}})) = 0$ . The PL condition (Definition 2) would then imply that  $\Phi (\overline{\mathbf{w}})$  is a global minimizer of  $f$  and hence, a global minimizer of  $h$ . With this fact in mind, our proof can be summarized in three steps: first, a careful choice of initialization will ensure that  $\nabla \Phi^{*}$  is nonsingular for all elements within a certain distance of  $\mathbf{w}^0$ . Second, we show that under small enough learning rate, the iterates of SGD remain close to the initialization  $\mathbf{w}^0$ , with high probability regardless of the number of iterations.

The third and final step will use the non-singularity of  $\nabla \Phi^{*}$  at convergence and Eq. (9) to conclude global optimality.

This is akin to the arguments in (Anonymous), however, in our case the stochasticity in the SGD updates poses a challenge for controlling the distance to initialization. We use concentration bounds on the length of the path and show that the SGD trajectory remains in the region where  $\nabla \Phi^{*}$  is non-singular, with high probability.

A crucial result for the first step of our proof has already been established in by (Anonymous). It shows that a smooth function that is near-isometry at initialization will preserve such property for all points within a certain distance.

Lemma 1 (Anonymous). Let  $\Phi$  be  $\beta_{\Phi}$ -smooth and  $\nabla \Phi^{*}(\mathbf{w}_{0})$  be a  $(\mu_{\Phi}, \nu_{\Phi})$ -near-isometry. Then

$$
f o r a l l \mathbf {w} s u c h t h a t \| \mathbf {w} - \mathbf {w} _ {0} \| \leq \frac {\mu_ {\Phi}}{2 \beta_ {\Phi}}, \quad \frac {\mu_ {\Phi}}{2} \leq \sigma_ {\min } \left(\nabla \Phi^ {*} (\mathbf {w})\right) \leq \sigma_ {\max } \left(\nabla \Phi^ {*} (\mathbf {w})\right) \leq \frac {3 \nu_ {\Phi}}{2} \tag {10}
$$

The second step in the proof of Theorem 1 is to compute the expected length of the SGD trajectory which is spent inside the ball defined in Eq. (10). We find an upper bound on this expected length depending on the initialization and learning rate, but independent of the number of iterations. Hence, under some proper initialization and learning rate, we can control the expected length of the trajectory for which Lemma 1 holds. In particular, we have

Proposition 1 (Expected length of trajectory). Let Assumptions 1 and 2 hold and let  $\zeta > 1$ . Let the random variable  $I$  denote the first iteration of SGD (Eq. (4)) such that

$$
\mathbf {w} ^ {I} \notin B := \operatorname {b a l l} \left(\mathbf {w} ^ {0}, \rho_ {\Phi}\right) := \left\{\mathbf {w}: \| \mathbf {w} - \mathbf {w} ^ {0} \| \leq \rho_ {\Phi} \right\} \tag {11}
$$

or  $I = \infty$  if the trajectory does not leave  $B$ . Suppose that  $\mathbf{w}^0$  satisfies Eq. (6) and SGD is executed with sufficiently small learning rate, which satisfies Eq. (7). An upper bound on the expected length of the SGD trajectory is given by

$$
\mathbb {E} [ \ell (I) ] \leq \frac {\mu_ {\Phi}}{2 \zeta \beta_ {\Phi}} = \frac {\rho_ {\Phi}}{\zeta}. \tag {12}
$$

We provide the sketch of the proof (see Appendix A for the complete proof). We first find an upper bound on the expected length of the trajectory in terms of the norm of gradients of  $f$ . With a proper learning rate, we find an upper bound on the norm of the gradient in terms of the expected decent of  $f$  in two consecutive iterates, which are inside the ball. We also ensure that the learning rate is sufficiently small such that  $\mathbb{E}[\| \mathbf{w}^I - \mathbf{w}^{I-1} \|]$  is bounded. Finally, under proper initialization, we obtain an upper bound on the expected length of the trajectory for the iterates inside the ball, i.e.,  $\mathbb{E}[\sum_{i=0}^{I-1} \| \mathbf{w}^i - \mathbf{w}^{i-1} \|]$ .

Remark 2. A similar phenomenon that shows bounded length of the trajectory has been observed in various settings mainly for gradient descent (Du et al., 2019; Oymak & Soltanolkotabi, 2019; Anonymous). In this paper, we focus on a compositional non-convex problem trained with SGD, which is more challenging to analyze.

Using the upper bound (12) on the expected length of the trajectory spent inside  $B = \mathrm{ball}(\mathbf{w}^0, \rho_{\Phi})$ , we can bound the probability that the SGD iterates leaves the ball  $B$ . Indeed, in order for the process to leave  $B$  starting from  $\mathbf{w}^0$ , it is required that the length of the trajectory spent inside  $B$  satisfies  $l(I) \geq \rho_{\Phi}$ . Hence, using bound (12) on  $\mathbb{E}[\ell(I)]$  together with a concentration bound (Markov inequality in our case), we can upper bound the probability of SGD iterates leaving  $B$ . Finally, under the event that the SGD iterates remain in  $B$ , an upper bound on  $\mathbb{E}[\ell(I)]$  implies the convergence of the iterates.

Remark 3. With a more involved analysis on the concentration properties of the random variable  $\ell(I)$ , it may be possible to greatly improve the dependence of the initialization and step size on  $\zeta$ .

Indeed, the current analysis assumes the worst-case scenario, where the SGD iterates either remain at the initialization, or directly leave the ball  $B$  in a straight line (this scenario indeed maximizes the probability that the process leaves the ball, given a bound on  $\mathbb{E}[\ell(I)]$ ).

Although  $\ell(I)$  is obtained as a sum of random variables, the difficulty of obtaining better concentration bounds for  $\ell(I)$  comes from the high level of dependence between all the variables involved. A better analysis would thus need to better understand how the trajectories behave inside the ball  $B$ , e.g., by bounding the variance of  $\ell(I)$ .

In the following section, we specify our result to the special case of shallow neural networks. We will show that, in the case of quadratic loss, the strong growth condition naturally holds, with a constant depending on the batch size. Moreover, using Gaussian initialization for the neural network parameters, we can control the initial smoothness and near-isometry parameters involved in Theorem 1 with high probability.

# 3 GLOBAL OPTIMALITY OF NEURAL NETWORKS TRAINED WITH SGD

Setup. We will consider the problem of training a shallow neural network with one hidden layer, input dimension  $d_0$ ,  $d_1$  hidden nodes, output dimension  $d_2$ , and quadratic loss. We denote the data and label matrices as  $X \in \mathbb{R}^{d_0 \times m}$  and  $Y \in \mathbb{R}^{d_2 \times m}$ , respectively.

Let  $W \in \mathbb{R}^{d_1 \times d_0}$  and  $V \in \mathbb{R}^{d_2 \times d_1}$  denote the parameters of the first and second layers of the network, respectively. We collect both parameters in a variable  $\Theta = (W, V) \in \mathbb{R}^{d_1 \times d_0} \times \mathbb{R}^{d_2 \times d_1}$ . In order to fit the supervised training of the network to the template studied in Section 2 (Eq. (3)) we define:

$$
\Phi (\Theta) := V \cdot \phi (W X) \in \mathbb {R} ^ {d _ {2} \times m}, \quad f _ {j} (Z) = \| Z _ {j} - Y _ {j} \| ^ {2} \in \mathbb {R} _ {+} \tag {13}
$$

where  $Z_{j}$  denotes the  $j$ -th column of a matrix  $Z$  and  $\phi : \mathbb{R} \to \mathbb{R}$  is the activation function, which is applied entry-wise. We can now write the problem as a the finite-sum:

$$
\min  _ {\theta} \left\{h (\Theta) = f (\Phi (\Theta)) = \frac {1}{m} \| V \phi (W X) - Y \| ^ {2} = \frac {1}{m} \sum_ {j = 1} ^ {m} \| V \phi (W X _ {j}) - Y _ {j} \| ^ {2} \right\} \tag {14}
$$

We will make an assumption on the Hermite norm (Definition 5) of the activation function. Our assumptions on the activation function are summarized as Assumption 3 below.

Definition 5 (Hermite norm (Olver et al., 2010)). Let  $\phi : \mathbb{R} \to \mathbb{R}$ . The Hermite norm of  $\phi$  is given by  $\| \phi \|_{\mathcal{H}} = \sqrt{\sum_{i=0}^{\infty} c_i^2}$  where  $c_i$  denotes the  $i$ -th Hermite coefficients of  $\phi$  given by:

$$
c _ {i} = \langle \phi , q _ {i} \rangle_ {\mathcal {H}} = \frac {1}{\sqrt {2 \pi}} \int \phi (x) q _ {i} (x) \exp \left(- \frac {x ^ {2}}{2}\right) d x
$$

and  $q_{i}:\mathbb{R}\to \mathbb{R}$  is the  $i$ -th Hermite polynomial (probabilist's convention) for  $i\geq 0$

Assumption 3.  $\phi$  is twice-differentiable,  $\phi (0) = 0$ ,  $\sup_{a}|\dot{\phi} (a)| = \dot{\phi}_{\max} < \infty$ ,  $\sup_{a}|\ddot{\phi} (a)| = \ddot{\phi}_{\max} < \infty$ , and  $\| \phi \|_{\mathcal{H}} < \infty$ .

The popular ReLU does not satisfy the twice-differentiability assumption. However, smooth approximations of ReLU such as the Gaussian error Linear Units (GeLU) and softmax (Hendrycks & Gimpel, 2020; Nguyen & Mondelli, 2020) have been shown to outperform ReLU in several settings and are commonly used in practice (Clevert et al., 2016; Gulrajani et al., 2017; Kumar et al., 2017; Kim et al., 2018; Xu et al., 2020). In addition, smoothened functions by a Gaussian kernel uniformly approximate the ReLU function (Nguyen & Mondelli, 2020).

Assumption 4. For all  $j \in [m]$ ,  $\| X_j \| \leq 1$ .  $\| Y \| \leq 1$ .

Remark 4. The assumption on the data is mild and common in the overparameterization literature (Li & Liang, 2018; Ji & Telgarsky, 2020). It can be enforced by data normalization.

Initialization. The initial iterate of SGD will be chosen in the following way:

$$
W ^ {0} \sim \mathcal {N} \left(0, \frac {1}{d _ {0}}\right), \quad V ^ {0} \sim \mathcal {N} \left(0, \frac {1}{d _ {1}}\right). \quad \Theta^ {0} := \left(W ^ {0}, V ^ {0}\right) \tag {15}
$$

Remark 5. The initialization in Eq. (15) matches popular initialization schemes such as LeCun (LeCun et al., 2012) and He (He et al., 2015) initializations.

We now proceed to estimate with high probability the value of  $h(\Theta^0)$ , near-isometry constants  $(\mu_{\Phi}, \nu_{\Phi})$  of  $\nabla^{*}\Phi(\Theta^{0})$  and smoothness parameter  $\beta_{\Phi}$ , which are required in Theorem 1.

Lemma 2 (Estimation of  $h(\Theta^0)$ ,  $\mu_{\Phi}, \nu_{\Phi}, \beta_{\Phi}$  (Anonymous)). Let Assumptions 3 and 4 hold, and suppose that  $\Theta^0$  follows the initialization distribution in Eq. (15). Let  $t$  be a positive integer such that  $m \simeq d_0^t$  and  $X^{*t} \in \mathbb{R}^{d_0^t \times m}$  be the matrix whose  $a$ -th column defined as  $\operatorname{vec}(x_a \otimes \dots \otimes x_a) \in \mathbb{R}^{d_0^t}$ .

For some constants  $\delta_1, \delta_2, \delta_3, k_1$ , and  $k_2$  independent of  $d_0$ ,  $d_1$  and  $m$ , with probability at least  $1 - \tilde{\psi}$  it holds that:

$$
\begin{array}{l} h (\Theta^ {0}) \leq \frac {\delta_ {3} ^ {2} k _ {1} ^ {2} k _ {2} ^ {2} \sigma_ {\max} ^ {2} (X)}{m} \\ \nu_ {\Phi} = \max  \left\{\left| c _ {0} \right| \sqrt {\left(1 + \delta_ {2}\right) d _ {1} m}, \omega_ {1} \sqrt {\left(1 + \delta_ {2}\right) \left(c _ {1} ^ {2} + c _ {\infty} ^ {2}\right) d _ {1}} \sigma_ {\max } (X) \right\} \tag {16} \\ \mu_ {\Phi} = \sqrt {(1 - \delta_ {1}) \frac {c _ {t} ^ {2}}{t !} d _ {1}} \sigma_ {\mathrm {m i n}} (X ^ {* t}) \\ \end{array}
$$

The precise expression for  $\tilde{\psi}$  is provided in Appendix  $E$ ,  $c_{i}$  is the  $i$ -th Hermite coefficients of  $\phi$  (Definition 5) and  $c_{\infty}^{2} = \sum_{i=2}^{\infty} c_{i}^{2}/i!$ .

Moreover, the map  $\Phi$  restricted to the set  $\{(V,W):\sigma_{\max}(V)\leq \chi_{\max}\}$  is smooth with constant

$$
\beta_ {\Phi} = \sqrt {2} \sigma_ {\max } (X) \left(\dot {\phi} _ {\max } + \ddot {\phi} _ {\max } \chi_ {\max }\right). \tag {17}
$$

Although the mapping  $\Phi$  is not globally smooth, Lemma 2 shows that it is smooth in a region where the largest singular value of  $V$  remains bounded. In the following lemma, we show that we can indeed bound the smoothness constant of  $\Phi$  restricted to a neighbourhood of  $V^0$  as required in Theorem 1.

Lemma 3. Let Assumption 3 hold. Let  $V^0, W^0$  be arbitrary matrices and  $\mu_{\Phi}$  be as in (16). Let

$$
\beta_ {\Phi} := \sqrt {2} \sigma_ {\max } (X) \left(\dot {\phi} _ {\max } + \sigma_ {\max } \left(V ^ {0}\right)\right) + \frac {\ddot {\phi} _ {\max } \mu_ {\Phi}}{2 \dot {\phi} _ {\max }}, \quad \rho_ {\Phi} := \frac {\mu_ {\Phi}}{2 \beta_ {\Phi}} \tag {18}
$$

The function  $\Phi$  is  $\beta_{\Phi}$ -smooth over the set:

$$
B _ {\rho_ {\Phi}} \left(V ^ {0}, W ^ {0}\right) := \left\{\left(V, W\right): \sqrt {\| V - V ^ {0} \| ^ {2} + \| W - W ^ {0} \| ^ {2}} \leq \rho_ {\Phi} \right\} \tag {19}
$$

Proof. Let

$$
\chi_ {\max } := \sigma_ {\max } \left(V ^ {0}\right) + \frac {\mu_ {\Phi}}{2 \sqrt {2} \sigma_ {\max } (X) \dot {\phi} _ {\max }} \text {,} \quad \tilde {B} _ {\chi_ {\max }} := \left\{\left(V, W\right): \sigma_ {\max } (V) \leq \chi_ {\max } \right\} \tag {20}
$$

Lemma 2 then implies that  $\Phi$  restricted to  $\tilde{B}\chi_{\mathrm{max}}$  is  $\beta_{\Phi}$ -smooth, following Eq. (17). With this choice of  $\chi_{\mathrm{max}}$  we show that  $B_{\rho_{\Phi}}(V^0,W^0)\subseteq \tilde{B}_{\chi_{\mathrm{max}}}$ , which implies the result. Note that  $\beta_{\Phi}\geq \sqrt{2}\sigma_{\mathrm{max}}(X)\dot{\phi}_{\mathrm{max}}$ , hence

$$
\chi_ {\mathrm {m a x}} \geq \sigma_ {\mathrm {m a x}} (V ^ {0}) + \frac {\mu_ {\Phi}}{2 \beta_ {\Phi}} = \sigma_ {\mathrm {m a x}} (V ^ {0}) + \rho_ {\Phi}.
$$

Suppose that  $(V,W)\in B_{\rho_{\Phi}}(V^0,W^0)$ . By Eq. (19) this implies  $\| V - V^{0}\| \leq \rho_{\Phi}$ . Then,

$$
\begin{array}{l} \sigma_ {\max } (V) \leq \sigma_ {\max } (V - V ^ {0}) + \sigma_ {\max } (V ^ {0}) \\ \leq \| V - V ^ {0} \| + \sigma_ {\max } (V ^ {0}) \\ \leq \rho_ {\Phi} + \sigma_ {\max } (V ^ {0}) \\ \leq \chi_ {\max } \cdot \\ \end{array}
$$

![](images/733ee1f5b608cd4ce2551d6d4971641cb29abf8de60e1ef8ccb757143589da9f.jpg)

In our case, as  $f$  is the quadratic loss, the growth condition (Definition 3) is satisfied with  $\eta_{f} = \frac{m}{b}$ . This is precisely the quantity that will reveal the impact of the minibatch size on the global convergence of SGD. Moreover, the quadratic loss satisfies the PL condition (Definition 2) and is smooth. All things considered, Assumptions 1 and 2 hold for our shallow neural network training setting, ensuring that Theorem 1 is valid.

We are now ready to integrate Lemma 2 and Lemma 3 together with the convergence guarantees in Theorem 1 to arrive at the sufficient degree of overparameterization required for the convergence of SGD. The following theorem finally concludes that for the shallow neural network described in Section 3, for a sufficient degree of overparameterization SGD converges to a global minimum with high probability.

Theorem 2 (Shallow network with SGD). Suppose that Assumptions 3 and 4 hold, and that  $\left(W^{0}, V^{0}\right)$  is randomly initialized as in (15). Suppose that the hidden layer width  $d_{1}$  satisfies

$$
d _ {1} = \tilde {\Omega} \left(\xi \left(\mathcal {C} _ {\delta}, t, \phi , \left\{c _ {i} \right\} _ {i \geq 0}, \zeta\right) \frac {\sigma_ {\max } ^ {2} (X) m}{\sigma_ {\min } ^ {3} \left(X ^ {* t}\right) \sqrt {b}}\right) \tag {21}
$$

where  $\mathcal{C}_{\delta}$  is a set of constants,  $\xi$  is a term independent of  $d_0,m$ . The SGD iterates converge to a global minimum exponentially fast with probability at least  $1 - \psi (\phi ,\xi ,d_0,d_1,d_2,X,\zeta)$ . See Appendix C for the exact expressions of  $\xi$  and  $\psi$  and the proof.

Finally, we provide an order analysis to understand how the sufficient overall overparameterization degree directly depends on the minibatch size. Intuitively, the sufficient overparameterization degree improves (is lower) as the minibatch size increases.

# 3.1 IMPACT OF THE MINIBATCH SIZE ON THE OVERPARAMETERIZATION DEGREE

For  $t = 1$ , the analysis requires  $m \simeq d_0$ , which is not a common setting in practice. For  $t \geq 2$  we suppose that  $m \simeq d_0^t$ , which is the case in practice. We estimate that  $\sigma_{\max}(X) \simeq \sqrt{m / d_0}$  and  $\sigma_{\min}(X^{*t}) \simeq \sqrt{m / d_0^t} \simeq 1$ , along the lines of (Oymak & Soltanolkotabi, 2020, Section 2.1). Substituting  $\sigma_{\max}(X)$  and  $\sigma_{\min}(X^{*t})$  into (21), we have

$$
d _ {1} \gtrsim \frac {m ^ {2}}{\sqrt {b} d _ {0}}. \tag {22}
$$

Therefore, the overall overparameterization degree becomes  $d_0d_1 \simeq \tilde{\Omega}(m^2/\sqrt{b})$ , which is sufficient for SGD to find a global minimum at a linear rate except with an arbitrary small probability. This fact will let us understand more clearly the effect minibatch size on the overparameterization degree.

If  $b = \tilde{\Omega}(m)$ , similar to gradient descent, a subquadratic scaling on the network width,  $d_0d_1 \simeq \tilde{\Omega}(m^{\frac{3}{2}})$ , is sufficient. In that case, an optimal linear scaling  $d_1 \simeq \tilde{O}(m)$  is sufficient when the number of input features is sufficiently large  $d_0 \simeq \tilde{\Omega}(\sqrt{m})$ .

On the other hand, when the batch size is small  $b = \tilde{O}(1)$ , we recover the standard quadratic scaling on the network width. Our analysis provides an interpolation between  $d_0d_1 \simeq \tilde{\Omega}(m^{\frac{3}{2}})$  and  $d_0d_1 \simeq \tilde{\Omega}(m^2)$  depending on  $b$ . As long as the batch size  $b \simeq \tilde{\Omega}(m^a)$  for some  $a > 0$ , we achieve a subquadratic scaling.

# 4 CONCLUSIONS AND FUTURE WORK

In this work, we prove linear convergence of stochastic gradient descent for training overparameterized two-layer neural networks with smooth activation functions, using classical initialization strategies, and where both layers are trained simultaneously. We provide a lower bound on the required over-parameterization degree for our result to hold, depending on the batch size  $b$  used to compute the stochastic gradients. More precisely, we show that using a number of parameters  $d_0d_1 = \Omega (m^2 /\sqrt{b})$  is sufficient to obtain linear convergence with high probability, providing subquadratic over-parameterization degree as long as the batch size increases with the number of data points.

In future work, we would like to relax the smoothness condition on the activation function, in order to encapsulate non-smooth activation functions such as ReLU. In addition, we would like to improve the high probability bound by analyzing more deeply the concentration properties of the random variable  $\ell(I)$ , characterizing the length of the trajectory spent in a neighborhood of the initialization. Finally, an important step would be to analyze the generalization properties of SGD through the lens of the proposed approach, in particular by analyzing in which case it leads to lazy training.

# REFERENCES

Zeyuan Allen-Zhu, Yanzhi Li, and Zhao Song. A convergence theory for deep learning via overparameterization. In International Conference on Machine Learning (ICML), 2019.  
Anonymous. Anonymous. Anonymous.  
Jérôme Bolte and Edouard Pauwels. Conservative set valued fields, automatic differentiation, stochastic gradient methods and deep learning. Mathematical Programming, 188(1):19-51, 2021.  
Jérôme Bolte, Trong Phong Nguyen, Juan Peypouquet, and Bruce W Suter. From error bounds to the complexity of first-order descent methods for convex functions. Mathematical Programming, 165: 471-507, 2017.  
Alon Brutzkus and Amir Globerson. Globally optimal gradient descent for a ConvNet with Gaussian inputs. In International Conference on Machine Learning (ICML), 2017.  
Volkan Cevher and Bäng Cöng Vu. On the linear convergence of the stochastic gradient method with constant step-size. Optimization Letters, 13:1177-1187, 2019.  
Zixiang Chen, Yuan Cao, Difan Zou, and Quanquan Gu. How much over-parameterization is sufficient to learn deep ReLU networks? In International Conference on Learning Representations (ICLR), 2021.  
Yuejie Chi, Yue M Lu, and Yuxin Chen. Nonconvex optimization meets low-rank matrix factorization: An overview. IEEE Transactions on Signal Processing (TSP), 67:5239-5269, 2019.  
Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. In Advances in neural information processing systems (NeurIPS), 2019.  
Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (ELUs). In International Conference on Learning Representations (ICLR), 2016.  
Amit Daniely. Neural networks learning and memorization with (almost) no over-parameterization. In Advances in neural information processing systems (NeurIPS), 2020.  
Simon S. Du and Jason D. Lee. On the power of over-parametrization in neural networks with quadratic activation. In International Conference on Machine Learning (ICML), 2018.  
Simon S. Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes over-parameterized neural networks. In International Conference on Learning Representations (ICLR), 2019.  
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of Wasserstein GANs. In Advances in neural information processing systems (NeurIPS), 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In Conference on Computer Vision and Pattern Recognition (CVPR), 2015.  
Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (GELU). arXiv preprint arXiv:1606.08415v4, 2020.  
Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal of American Statistical Association, 58:13-30, 1963.

Ziwei Ji and Matus Telgarsky. Polylogarithmic width suffices for gradient descent to achieve arbitrarily small test error with shallow ReLU networks. In International Conference on Learning Representations (ICLR), 2020.  
Sham Kakade and Jason D Lee. Provably correct automatic subdifferentiation for qualified programs. arXiv preprint arXiv:1809.08530, 2018.  
Kenji Kawaguchi and Jiaoyang Huang. Gradient descent finds global minima for generalizable deep neural networks of practical sizes. In Annual Allerton Conference on Communication, Control, and Computing, 2019.  
Youngjin Kim, Minjung Kim, and Gunhee Kim. Memorization precedes generation: Learning unsupervised GANs with memory networks. In International Conference on Learning Representations (ICLR), 2018.  
Abhishek Kumar, Prasanna Sattigeri, and Tom Fletcher. Semi-supervised learning with GANs: Manifold invariance with improved inference. In Advances in neural information processing systems (NeurIPS), 2017.  
Yann A. LeCun, Léon Bottou, Genevieve B. Orr, and Klaus-Robert Müller. Efficient BackProp. In Neural networks: Tricks of the Trade. Springer, 2012.  
Yuanzhi Li and Yingyu Liang. Learning overparameterized neural networks via stochastic gradient descent on structured data. In Advances in neural information processing systems (NeurIPS), 2018.  
Roi Livni, Shai Shalev-Shwartz, and Ohad Shamir. On the computational efficiency of training neural networks. In Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 1, NIPS'14, pp. 855-863, Cambridge, MA, USA, 2014. MIT Press.  
Yiping Lu, Chao Ma, Yulong Lu, Jianfeng Lu, and Lexing Ying. A mean field analysis of deep resnet and beyond: Towards provably optimization via overparameterization from depth. In International Conference on Machine Learning (ICML), 2020.  
Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Mean-field theory of two-layers neural networks: dimension-free bounds and kernel limit. In Conference on Learning Theory, 2019.  
Quynh Nguyen and Marco Mondelli. Global convergence of deep networks with one wide layer followed by pyramidal topology. In Advances in neural information processing systems (NeurIPS), 2020.  
Frank W. J. Olver, Daniel W. Lozier, Ronald F. Boisvert, and Charles W. Clark. NIST Handbook of Mathematical Functions Paperback and CD-ROM. Cambridge University Press, 2010.  
Samet Oymak and Mahdi Soltanolkotabi. Overparameterized nonlinear learning: Gradient descent takes the shortest path? In International Conference on Machine Learning (ICML), 2019.  
Samet Oymak and Mahdi Soltanolkotabi. Towards moderate overparameterization: global convergence guarantees for training shallow neural networks. IEEE Journal on Selected Areas in Information Theory, 1:84-105, 2020.  
Itay Safran and Ohad Shamir. Spurious local minima are common in two-layer relu neural networks. In International Conference on Machine Learning, pp. 4433-4441. PMLR, 2018.  
Mark Schmidt and Nicolas Le Roux. Fast convergence of stochastic gradient descent under a strong growth condition. arXiv preprint arXiv:1308.6370, 2013.  
Mahdi Soltanolkotabi, Adel Javanmard, and J. Lee. Theoretical insights into the optimization landscape of over-parameterized shallow neural networks. IEEE Transactions on Information Theory, 65:742-769, 2019.  
Zhao Song and Xin Yang. Quadratic suffices for over-parametrization via matrix Chernoff bound. arXiv preprint arXiv:1906.03593v2, 2019.  
Lili Su and Pengkun Yang. On learning over-parameterized neural networks: A functional approximation perspective. In Advances in neural information processing systems (NeurIPS), 2019.

Sharan Vaswani, Francis Bach, and Mark Schmidt. Fast and faster convergence of sgd for overparameterized models and an accelerated perceptron. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2019.

Roman Vershynin. Introduction to the Non-asymptotic Analysis of Random Matrices. Cambridge University Press, 2012.

Bing Xu, Naiyan Wang, Tianqi Chen, and Mu Li. Empirical evaluation of rectified activations in convolutional network. arXiv preprint arXiv:1505.00853v2, 2020.

Difan Zou and Quanquan Gu. An improved analysis of training over-parameterized deep neural networks. In Advances in neural information processing systems (NeurIPS), 2019.
