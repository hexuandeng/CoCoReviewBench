# PROVABLE BENEFIT OF ORTHOGONAL INITIALIZATION IN OPTIMIZING DEEP LINEAR NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

The selection of initial parameter values for gradient-based optimization of deep neural networks is one of the most impactful hyperparameter choices in deep learning systems, affecting both convergence times and model performance. Yet despite significant empirical and theoretical analysis, relatively little has been proved about the concrete effects of different initialization schemes. In this work, we analyze the effect of initialization in deep linear networks, and provide for the first time a rigorous proof that drawing the initial weights from the orthogonal group speeds up convergence relative to the standard Gaussian initialization with iid weights. We show that for deep networks, the width needed for efficient convergence to a global minimum with orthogonal initializations is independent of the depth, whereas the width needed for efficient convergence with Gaussian initializations scales linearly in the depth. Our results demonstrate how the benefits of a good initialization can persist throughout learning, suggesting an explanation for the recent empirical successes found by initializing very deep non-linear networks according to the principle of dynamical isometry.

# 1 INTRODUCTION

Through their myriad successful applications across a wide range of disciplines, it is now well established that deep neural networks possess an unprecedented ability to model complex real-world datasets, and in many cases they can do so with minimal overfitting. Indeed, the list of practical achievements of deep learning has grown at an astonishing rate, and includes models capable of human-level performance in tasks such as image recognition (Krizhevsky et al., 2012), speech recognition (Hinton et al., 2012), and machine translation (Wu et al., 2016).

Yet to each of these deep learning triumphs corresponds a large engineering effort to produce such a high-performing model. Part of the practical difficulty in designing good models stems from a proliferation of hyperparameters and a poor understanding of the general guidelines for their selection. Given a candidate network architecture, some of the most impactful hyperparameters are those governing the choice of the model's initial weights. Although considerable study has been devoted to the selection of initial weights, relatively little has been proved about how these choices affect important quantities such as rate of convergence of gradient descent.

In this work, we examine the effect of initialization on the rate of convergence of gradient descent in deep linear networks. We provide for the first time a rigorous proof that drawing the initial weights from the orthogonal group speeds up convergence relative to the standard Gaussian initialization with iid weights. In particular, we show that for deep networks, the width needed for efficient convergence for orthogonal initializations is independent of the depth, whereas the width needed for efficient convergence of Gaussian networks scales linearly in the depth.

Orthogonal weight initializations have been the subject of a significant amount of prior theoretical and empirical investigation. For example, in a line of work focusing on dynamical isometry, it was found that orthogonal weights can speed up convergence for deep linear networks (Saxe et al., 2014; Advani & Saxe, 2017) and for deep non-linear networks (Pennington et al., 2018; Xiao et al., 2018; Gilboa et al., 2019; Chen et al., 2018; Pennington et al., 2017; Tarnowski et al., 2019; Ling & Qiu, 2019) when they operate in the linear regime. In the context of recurrent neural networks, orthogonality can help improve the system's stability. A main limitation of prior work is that it has focused almost exclusively on model's properties at initialization. In contrast, our analysis

focuses on the benefit of orthogonal initialization on the entire training process, thereby establishing a provable benefit for optimization.

The paper is organized as follows. After reviewing related work in Section 2 and establishing some preliminaries in Section 3, we present our main positive result on efficient convergence from orthogonal initialization in Section 4. In Section 5, we show that Gaussian initialization leads to exponentially long convergence time if the width is too small compared with the depth.

# 2 RELATED WORK

Deep linear networks. Despite the simplicity of their input-output maps, deep linear networks define high-dimensional non-convex optimization landscapes whose properties closely reflect those of their non-linear counterparts. For this reason, deep linear networks have been the subject of extensive theoretical analysis. A line of work (Kawaguchi, 2016; Hardt & Ma, 2016; Lu & Kawaguchi, 2017; Yun et al., 2017; Zhou & Liang, 2018; Laurent & von Brecht, 2018) studied the landscape properties of deep linear networks. Although it was established that all local minima are global under certain assumptions, these properties alone are still not sufficient to guarantee global convergence or to provide a concrete rate of convergence for gradient-based optimization algorithms.

Another line of work directly analyzed the trajectory taken by gradient descent and established conditions that guarantee convergence to global minimum (Bartlett et al., 2018; Arora et al., 2018; Du & Hu, 2019). Most relevant to our work is the result of Du & Hu (2019), which shows that if the width of hidden layers is larger than the depth, gradient descent with Gaussian initialization can efficiently converge to a global minimum. Our result establishes that for Gaussian initialization, this linear dependence between width and depth is necessary, while for orthogonal initialization, the width can be independent of depth. Our negative result for Gaussian initialization also significantly generalizes the result of Shamir (2018), who proved a similar negative result for 1-dimensional linear networks.

Orthogonal weight initializations. Orthogonal weight initializations have also found significant success in non-linear networks. In the context of feedforward models, the spectral properties of a network's input-output Jacobian have been empirically linked to convergence speed (Saxe et al., 2014; Pennington et al., 2017; 2018; Xiao et al., 2018). It was found that when this spectrum concentrates around 1 at initialization, a property dubbed dynamical isometry, convergence times improved by orders of magnitude. The conditions for attaining dynamical isometry in the infinite-width limit were established by Pennington et al. (2017; 2018) and basically require that input-output map to be approximately linear and for the weight matrices to be orthogonal. Therefore the training time benefits of dynamical isometry are likely rooted in the benefits of orthogonality for deep linear networks, which we establish in this work.

Orthogonal matrices are also frequently used in the context of recurrent neural networks, for which the stability of the state-to-state transition operator is determined by the spectrum of its Jacobian (Haber & Ruthotto, 2017; Laurent & von Brecht, 2016). Orthogonal matrices can improve the conditioning, leading to an ability to learn over long time horizons (Le et al., 2015; Henaff et al., 2016; Chen et al., 2018; Gilboa et al., 2019). While the benefits of orthogonality can be quite large at initialization, little is known about whether or in what contexts these benefits persist during training, a scenario that has lead to the development of efficient methods of constraining the optimization to the orthogonal group (Wisdom et al., 2016; Vorontsov et al., 2017; Mhammedi et al., 2017). Although we do not study the recurrent setting in this work, an extension of our analysis might help determine when orthogonality is beneficial in that setting.

# 3 PRELIMINARIES

# 3.1 NOTATION

Let  $[n] = \{1,2,\dots ,n\}$ . Denote by  $I_{d}$  the  $d\times d$  identity matrix, and by  $I$  an identity matrix when its dimension is clear from context. Denote by  $\mathcal{N}(\mu ,\sigma^2)$  the Gaussian distribution with mean  $\mu$  and variance  $\sigma^2$ , and by  $\chi_k^2$  the chi-squared distribution with  $k$  degrees of freedom.

Denote by  $\| \cdot \|$  the  $\ell_2$  norm of a vector or the spectral norm of a matrix. Denote by  $\| \cdot \|_F$  the Frobenius norm of a matrix. For a symmetric matrix  $A$ , let  $\lambda_{\max}(A)$  and  $\lambda_{\min}(A)$  be its maximum and minimum eigenvalues, and let  $\lambda_i(A)$  be its  $i$ -th largest eigenvalue. For a matrix  $B \in \mathbb{R}^{m \times n}$ , let  $\sigma_i(B)$  be its  $i$ -th largest singular value ( $i = 1, 2, \ldots, \min\{m, n\}$ ), and let  $\sigma_{\max}(B) = \sigma_1(B)$ ,  $\sigma_{\min}(B) = \sigma_{\min\{m, n\}}(B)$ . Denote by  $\operatorname{vec}(A)$  the vectorization of a matrix  $A$  in column-first order. The Kronecker product between two matrices  $A \in \mathbb{R}^{m_1 \times n_1}$  and  $B \in \mathbb{R}^{m_2 \times n_2}$  is defined as

$$
A \otimes B = \left( \begin{array}{c c c} a _ {1, 1} B & \dots & a _ {1, n _ {1}} B \\ \vdots & \ddots & \vdots \\ a _ {m _ {1}, 1} B & \dots & a _ {m _ {1}, n _ {1}} B \end{array} \right) \in \mathbb {R} ^ {m _ {1} m _ {2} \times n _ {1} n _ {2}},
$$

where  $a_{i,j}$  is the element in the  $(i,j)$ -th entry of  $A$ .

We use the standard  $O(\cdot), \Omega(\cdot)$  and  $\Theta(\cdot)$  notation to hide universal constant factors. We also use  $C$  to represent a sufficiently large universal constant whose specific value can differ from line to line.

# 3.2 PROBLEM SETUP

Suppose that there are  $n$  training examples  $\{(x_k, y_k)\}_{k=1}^n \subset \mathbb{R}^{d_x} \times \mathbb{R}^{d_y}$ . Denote by  $X = (x_1, \ldots, x_n) \in \mathbb{R}^{d_x \times n}$  the input data matrix and by  $Y = (y_1, \ldots, y_n) \in \mathbb{R}^{d_y \times n}$  the target matrix. Consider an  $L$ -layer linear neural network with weight matrices  $W_1, \ldots, W_L$ , which given an input  $x \in \mathbb{R}^{d_x}$  computes

$$
f (x; W _ {1}, \dots , W _ {L}) = \alpha W _ {L} W _ {L - 1} \dots W _ {1} x, \tag {1}
$$

where  $W_{i} \in \mathbb{R}^{d_{i} \times d_{i-1}} (i = 1, \dots, L)$ ,  $d_{0} = d_{x}$ ,  $d_{L} = d_{y}$ , and  $\alpha$  is a normalization constant which will be specified later according to the initialization scheme. We study the problem of training the deep linear network by minimizing the  $\ell_{2}$  loss over training data:

$$
\ell \left(W _ {1}, \dots , W _ {L}\right) = \frac {1}{2} \sum_ {k = 1} ^ {n} \| f \left(x _ {k}; W _ {1}, \dots , W _ {L}\right) - y _ {k} \| ^ {2} = \frac {1}{2} \| \alpha W _ {L} \dots W _ {1} X - Y \| _ {F} ^ {2}. \tag {2}
$$

The algorithm we consider to minimize the objective (2) is gradient descent with random initialization, which first randomly samples the initial weight matrices  $\{W_{i}(0)\}_{i = 1}^{L}$  from a certain distribution, and then updates the weights using gradient descent: for time  $t = 0,1,2,\ldots$

$$
W _ {i} (t + 1) = W _ {i} (t) - \eta \frac {\partial \ell}{\partial W _ {i}} \left(W _ {1} (t), \dots , W _ {L} (t)\right), \quad i \in [ L ], \tag {3}
$$

where  $\eta > 0$  is the learning rate.

For convenience, we denote  $W_{j:i} = W_jW_{j-1}\dots W_i$  ( $1 \leq i \leq j \leq L$ ) and  $W_{i-1:i} = I$  ( $i \in [L]$ ). The time index  $t$  is used on any variable that depends on  $W_1, \ldots, W_L$  to represent its value at time  $t$ , e.g.,  $W_{j:i}(t) = W_j(t)\dots W_i(t)$ ,  $\ell(t) = \ell(W_1(t), \ldots, W_L(t))$ , etc.

# 4 EFFICIENT CONVERGENCE USING ORTHOGONAL INITIALIZATION

In this section we present our main positive result for orthogonal initialization. We show that orthogonal initialization enables efficient convergence of gradient descent to a global minimum provided that the hidden width is not too small.

In order to properly define orthogonal weights, we let the widths of all hidden layers be equal:  $d_{1} = d_{2} = \dots = d_{L - 1} = m$ , and let  $m \geq \max \{d_x,d_y\}$ . Note that all intermediate matrices  $W_{2},\ldots ,W_{L - 1}$  are  $m \times m$  square matrices, and  $W_{1} \in \mathbb{R}^{m\times d_{x}}$ ,  $W_{L} \in \mathbb{R}^{d_y\times m}$ . We sample each initial weight matrix  $W_{i}(0)$  independently from a uniform distribution over scaled orthogonal matrices satisfying

$$
W _ {1} ^ {\top} (0) W _ {1} (0) = m I _ {d _ {x}},
$$

$$
W _ {i} ^ {\top} (0) W _ {i} (0) = W _ {i} (0) W _ {i} ^ {\top} (0) = m I _ {m}, \quad 2 \leq i \leq L - 1, \tag {4}
$$

$$
W _ {L} (0) W _ {L} ^ {\top} (0) = m I _ {d _ {y}}.
$$

In accordance with such initialization, the scaling factor  $\alpha$  in (1) is set as  $\alpha = \frac{1}{\sqrt{m^{L - 1}d_y}}$ , which ensures  $\mathbb{E}\left[\| f(x;W_L(0),\ldots ,W_1(0))\|^2\right] = \| x\|^2$  for any  $x\in \mathbb{R}^{d_x}$ . The same scaling factor was adopted in Du & Hu (2019), which preserves the expectation of the squared  $\ell_{2}$  norm of any input.

Let  $W^{*}\in \arg \min_{W\in \mathbb{R}^{d_y\times d_x}}\| WX - Y\| _F$  and  $\ell^* = \frac{1}{2}\| W^* X - Y\| _F^2$ . Then  $\ell^*$  is the minimum value for the objective (2). Denote  $r = \mathrm{rank}(X)$ ,  $\kappa = \frac{\lambda_{\max}(X^\top X)}{\lambda_r(X^\top X)}$ , and  $\tilde{r} = \frac{\|X\|_F^2}{\|X\|^2}$ . Our main theorem in this section is the following:

Theorem 4.1. Suppose

$$
m \geq C \cdot \tilde {r} \kappa^ {2} \left(d _ {y} \left(1 + \| W ^ {*} \| ^ {2}\right) + \log (r / \delta)\right) a n d m \geq d _ {x}, \tag {5}
$$

for some  $\delta \in (0,1)$  and a sufficiently large universal constant  $C > 0$ . Set the learning rate  $\eta \leq \frac{d_y}{2L\|X\|^2}$ . Then with probability at least  $1 - \delta$  over the random initialization, we have

$$
\begin{array}{l} \ell (0) - \ell^ {*} \leq O \left(1 + \frac {\log (r / \delta)}{d _ {y}} + \| W ^ {*} \| ^ {2}\right) \| X \| _ {F} ^ {2}, \\ \ell (t) - \ell^ {*} \leq \left(1 - \frac {1}{2} \eta L \lambda_ {r} (X ^ {\top} X) / d _ {y}\right) ^ {t} (\ell (0) - \ell^ {*}), \quad t = 0, 1, 2, \dots , \\ \end{array}
$$

where  $\ell(t)$  is the objective value at iteration  $t$ .

Notably, in Theorem 4.1, the width  $m$  need not depend on the depth  $L$ . This is in sharp contrast with the result of Du & Hu (2019) for Gaussian initialization, which requires  $m \geq \tilde{\Omega} (Lr\kappa^3 d_y)$ . It turns out that a near-linear dependence between  $m$  and  $L$  is necessary for Gaussian initialization to have efficient convergence, as we will show in Section 5. Therefore the requirement in Du & Hu (2019) is nearly tight in terms of the dependence on  $L$ . These results together rigorously establish the benefit of orthogonal initialization in optimizing very deep linear networks.

If we set the learning rate optimally according to Theorem 4.1 to  $\eta = \Theta \left( \frac{d_y}{L \| X \|^2} \right)$ , we obtain that  $\ell(t) - \ell^*$  decreases by a ratio of  $1 - \Theta (\kappa^{-1})$  after every iteration. This matches the convergence rate of gradient descent on the (1-layer) linear regression problem  $\min_{W \in \mathbb{R}^{d_y \times d_x}} \frac{1}{2} \| WX - Y \|_F^2$ .

# 4.1 PROOF OF THEOREM 4.1

The proof uses the high-level framework from Du & Hu (2019), which tracks the evolution of the network's output during optimization. This evolution is closely related to a time-varying positive semidefinite (PSD) matrix (defined in (7)), and the proof relies on carefully upper and lower bounding the eigenvalues of this matrix throughout training, which in turn implies the desired convergence result.

First, we can make the following simplifying assumption without loss of generality. See Appendix B in Du & Hu (2019) for justification.

Assumption 4.1. (Without loss of generality)  $X \in \mathbb{R}^{d_x \times r}$ ,  $\operatorname{rank}(X) = r$ ,  $Y = W^* X$ , and  $\ell^* = 0$ .

Now we briefly review Du & Hu (2019)'s framework. The key idea is to look at the network's output, defined as

$$
U = \alpha W _ {L: 1} X \in \mathbb {R} ^ {d _ {y} \times n}.
$$

We also write  $U(t) = \alpha W_{L:1}(t)X$  as the output at time  $t$ . Note that  $\ell(t) = \frac{1}{2} \|U(t) - Y\|_F^2$ . According to the gradient descent update rule, we write

$$
W _ {L: 1} (t + 1) = \prod_ {i} \left(W _ {i} (t) - \eta \frac {\partial \ell}{\partial W _ {i}} (t)\right) = W _ {L: 1} (t) - \sum_ {i = 1} ^ {L} \eta W _ {L: i + 1} (t) \frac {\partial \ell}{\partial W _ {i}} (t) W _ {i - 1: 1} (t) + E (t),
$$

where  $E(t)$  contains all the high-order terms (i.e., those with  $\eta^2$  or higher). With this definition, the evolution of  $U(t)$  can be written as the following equation:

$$
\operatorname {v e c} (U (t + 1) - U (t)) = - \eta P (t) \cdot \operatorname {v e c} (U (t) - Y) + \alpha \cdot \operatorname {v e c} (E (t) X), \tag {6}
$$

where

$$
P (t) = \alpha^ {2} \sum_ {i = 1} ^ {L} \left[ \left(\left(W _ {i - 1: 1} (t) X\right) ^ {\top} \left(W _ {i - 1: 1} (t) X\right)\right) \otimes \left(W _ {L: i + 1} (t) W _ {L: i + 1} ^ {\top} (t)\right) \right]. \tag {7}
$$

Notice that  $P(t)$  is always PSD since it is the sum of  $L$  PSD matrices. Therefore, in order to establish convergence, we only need to (i) show that the higher-order term  $E(t)$  is small and (ii) prove upper and lower bounds on  $P(t)$ 's eigenvalues. For the second task, it suffices to control the singular values of  $W_{i-1:1}(t)$  and  $W_{L:i+1}(t)$  ( $i \in [L]$ ).<sup>3</sup> Under orthogonal initialization, these matrices are perfectly isometric at initialization, and we will show that they stay close to isometry during training, thus enabling efficient convergence.

The following lemma summarizes some properties at initialization.

Lemma 4.2. At initialization, we have

$$
\sigma_ {\max } \left(W _ {j: i} (0)\right) = \sigma_ {\min } \left(W _ {j: i} (0)\right) = m ^ {\frac {j - i + 1}{2}}, \quad \forall 1 \leq i \leq j \leq L, (i, j) \neq (1, L). \tag {8}
$$

Furthermore, with probability at least  $1 - \delta$ , the loss at initialization satisfies

$$
\ell (0) \leq O \left(1 + \frac {\log (r / \delta)}{d _ {y}} + \| W ^ {*} \| ^ {2}\right) \| X \| _ {F} ^ {2}. \tag {9}
$$

Proof sketch. The spectral property (8) follows directly from (4).

To prove (9), we essentially need to upper bound the magnitude of the network's initial output. This turns out to be equivalent to studying the magnitude of the projection of a vector onto a random low-dimensional subspace, which we can bound using standard concentration inequalities. The details are given in Appendix A.1.

Now we proceed to prove Theorem 4.1. We define  $B = O\left(1 + \frac{\log(r / \delta)}{d_y} + \| W^*\|^2\right) \| X\|_F^2$  which is the upper bound on  $\ell(0)$  from (9). Conditioned on (9) being satisfied, we will use induction on  $t$  to prove the following three properties  $\mathcal{A}(t), \mathcal{B}(t)$  and  $\mathcal{C}(t)$  for all  $t = 0, 1, \ldots$ :

-  $\mathcal{A}(t)$ :  $\ell(t) \leq \left(1 - \frac{1}{2}\eta L\sigma_{\min}^2(X)/d_y\right)^t\ell(0) \leq \left(1 - \frac{1}{2}\eta L\sigma_{\min}^2(X)/d_y\right)^tB.$  
-  $\mathcal{B}(t)$ :  $\sigma_{\max}(W_{j:i}(t)) \leq 1.1m^{\frac{j - i + 1}{2}}, \sigma_{\min}(W_{j:i}(t)) \geq 0.9m^{\frac{j - i + 1}{2}}$ ,  $\forall 1 \leq i \leq j \leq L, (i,j) \neq (1,L)$ .  
$\mathcal{C}(t)$  ..  $\| W_i(t) - W_i(0)\| _F\leq \frac{8\sqrt{Bd_y}\|X\|}{L\sigma_{\min}^2(X)},\quad \forall 1\leq i\leq L.$

$\mathcal{A}(0)$  and  $\mathcal{B}(0)$  are true according to Lemma 4.2, and  $\mathcal{C}(0)$  is trivially true. In order to prove  $\mathcal{A}(t)$ ,  $\mathcal{B}(t)$  and  $\mathcal{C}(t)$  for all  $t$ , we will prove the following claims for all  $t \geq 0$ :

Claim 4.3.  $\mathcal{A}(0),\ldots ,\mathcal{A}(t),\mathcal{B}(0),\ldots ,\mathcal{B}(t)\Longrightarrow \mathcal{C}(t + 1).$

Claim 4.4.  $\mathcal{C}(t) \Longrightarrow \mathcal{B}(t)$ .

Claim 4.5.  $\mathcal{A}(t),\mathcal{B}(t)\Longrightarrow \mathcal{A}(t + 1)$

The proofs of these claims are given in Appendix A. Notice that we finish the proof of Theorem 4.1 once we prove  $\mathcal{A}(t)$  for all  $t\geq 0$

# 5 EXPONENTIAL CURSE OF GAUSSIAN INITIALIZATION

In this section, we show that gradient descent with Gaussian random initialization necessarily suffers from a running time that scales exponentially with the depth of the network, unless the width becomes nearly linear in the depth. Since we mostly focus on the dependence of width and running time on depth, we will assume the depth  $L$  to be very large.

Recall that we want to minimize the objective  $\ell(W_1, \ldots, W_L) = \frac{1}{2} \|\alpha W_{L:1} X - Y\|_F^2$  by gradient descent. We assume  $Y = W^*X$  for some  $W^* \in \mathbb{R}^{d_y \times d_x}$ , so that the optimal objective value is 0. For convenience, we assume  $\|X\|_F = \Theta(1)$  and  $\|Y\|_F = \Theta(1)$ .

Suppose that at layer  $i \in [L]$ , every entry of  $W_{i}(0)$  is sampled from  $\mathcal{N}(0, \sigma_i^2)$ , and all weights in the network are independent. We set the scaling factor  $\alpha$  such that the initial output of the network does not blow up exponentially (in expectation):

$$
\mathbb {E} \left[ \| f (x; W _ {1} (0), \dots , W _ {L} (0)) \| ^ {2} \right] \leq L ^ {O (1)} \cdot \| x \| ^ {2}, \quad \forall x \in \mathbb {R} ^ {d _ {x}}. \tag {10}
$$

Note that  $\mathbb{E}\left[\| f(x;W_1(0),\ldots ,W_L(0))\|^2\right] = \alpha^2\prod_{i = 1}^{L}(d_i\sigma_i^2)\| x\|^2$ . Thus (10) means

$$
\alpha^ {2} \prod_ {i = 1} ^ {L} \left(d _ {i} \sigma_ {i} ^ {2}\right) \leq L ^ {O (1)}.
$$

We also assume that the magnitude of initialization at each layer cannot vanish with depth:

$$
d _ {i} \sigma_ {i} ^ {2} \geq \frac {1}{L ^ {\mathcal {O} (1)}}, \quad \forall i \in [ L ]. \tag {11}
$$

Note that the assumptions (10) and (11) are just sanity checks to rule out the obvious pathological cases – they are easily satisfied by all the commonly used initialization schemes in practice.

Now we formally state our main theorem in this section.

Theorem 5.1. Suppose  $\max \{d_0, d_1, \ldots, d_L\} \leq O(L^{1 - \gamma})$  for some universal constant  $0 < \gamma \leq 1$ . Then there exists a universal constant  $c > 0$  such that, if gradient descent is run with learning rate  $\eta \leq e^{cL^{\gamma}}$ , then with probability at least 0.9 over the random initialization, for the first  $e^{\Omega (L^{\gamma})}$  iterations, the objective value is stuck between  $0.4\| Y\| _F^2$  and  $0.6\| Y\| _F^2$ .

Theorem 5.1 establishes that efficient convergence from Gaussian initialization is impossible for large depth unless the width becomes nearly linear in depth. This nearly linear dependence is the best we can hope for, since Du & Hu (2019) proved a positive result when the width is larger than linear in depth. Therefore, a phase transition from untrainable to trainable happens at the point when the width and depth has a nearly linear relation. Furthermore, Theorem 5.1 generalizes the result of Shamir (2018), which only treats the special case of  $d_0 = \dots = d_L = 1$ .

# 5.1 PROOF OF THEOREM 5.1

For convenience, we define a scaled version of  $W_{i}$ : let  $A_{i} = W_{i} / (\sqrt{d_{i}}\sigma_{i})$  and  $\beta = \alpha \prod_{i=1}^{L}(\sqrt{d_{i}}\sigma_{i})$ . Then we know  $\beta \leq L^{O(1)}$  and  $\alpha W_{L:1} = \beta A_{L:1}$ , where  $A_{j:i} = A_{j}\dots A_{i}$ .

We first give a simple upper bound on  $\| A_{j:i}(0)\|$  for all  $1\leq i\leq j\leq L$ .

Lemma 5.2. With probability at least  $1 - \delta$ , we have  $\|A_{j:i}(0)\| \leq O\left(\frac{L^3}{\delta}\right)$  for all  $1 \leq i \leq j \leq L$ .

The proof of Lemma 5.2 is given in Appendix B.1. It simply uses Markov inequality and union bound.

Furthermore, a key property at initialization is that if  $j - i$  is large enough,  $\| A_{j:i}(0)\|$  will become exponentially small.

Lemma 5.3. With probability at least  $1 - e^{-\Omega (L^{\gamma})}$ , for all  $1 \leq i \leq j \leq L$  such that  $j - i \geq \frac{L}{10}$ , we have  $\| A_{j:i}(0)\| \leq e^{-\Omega (L^{\gamma})}$ .

Proof. We first consider a fixed pair  $(i,j)$  such that  $j - i \geq \frac{L}{10}$ . In order to bound  $\|A_{j:i}(0)\|$ , we first take an arbitrary unit vector  $v \in \mathbb{R}^{d_{i-1}}$  and bound  $\|A_{j:i}(0)v\|$ . We can write  $\|A_{j:i}(0)v\|^2 = \prod_{k=i}^{j}Z_k$ , where  $Z_k = \frac{\|A_{k:i}(0)v\|^2}{\|A_{k-1:i}(0)v\|^2}$ . Note that for any nonzero  $v' \in \mathbb{R}^{d_{k-1}}$  independent of  $A_k(0)$ , the distribution of  $d_k \cdot \frac{\|A_k(0)v'\|^2}{\|v'\|^2}$  is  $\chi_{d_k}^2$ . Therefore,  $Z_i, \ldots, Z_j$  are independent, and  $d_kZ_k \sim \chi_{d_k}^2$  ( $k = i, i+1, \ldots, j$ ). Recall the expression for the moments of chi-squared random variables:  $\mathbb{E}\left[Z_k^\lambda\right] = \frac{2^\lambda\Gamma(d_k/2+\lambda)}{d_k^\lambda\Gamma(d_k/2)}$  ( $\forall \lambda > 0$ ). Taking  $\lambda = \frac{1}{2}$  and using the bound  $\frac{\Gamma(a + \frac{1}{2})}{\Gamma(a)} \leq \sqrt{a - 0.1} (\forall a \geq \frac{1}{2})$  (Qi & Luo, 2012), we get  $\mathbb{E}\left[\sqrt{Z_k}\right] \leq \sqrt{\frac{2(d_k/2 - 0.1)}{d_k}} = \sqrt{1 - \frac{0.2}{d_k}} \leq 1 - \frac{0.1}{d_k}$ . Therefore we have  $\mathbb{E}\left[\sqrt{\prod_{k=i}^{j}Z_k}\right] \leq \prod_{k=i}^{j}\left(1 - \frac{0.1}{d_k}\right) \leq \left(1 - \frac{0.1}{O(L^{1-\gamma})}\right)^{j-i+1} \leq (1 - \Omega(L^{\gamma-1}))^{\frac{L}{10}} = e^{-\Omega(L^{\gamma})}$ .

Choose a sufficiently small constant  $c' > 0$ . By Markov inequality we have  $\operatorname*{Pr}\left[\sqrt{\prod_{k=i}^{j}Z_k} > e^{-c'L^{\gamma}}\right] \leq e^{c'L^{\gamma}}\mathbb{E}\left[\sqrt{\prod_{k=i}^{j}Z_k}\right] \leq e^{c'L^{\gamma}}e^{-\Omega(L^{\gamma})} = e^{-\Omega(L^{\gamma})}$ . Therefore we have shown that for any fixed unit vector  $v \in \mathbb{R}^{d_{i-1}}$ , with probability at least  $1 - e^{-\Omega(L^{\gamma})}$  we have  $\|A_{j:i}(0)v\| \leq e^{-\Omega(L^{\gamma})}$ .

Next, we use this to bound  $\| A_{j:i}(0) \|$  via an  $\epsilon$ -net argument. We partition the index set  $[d_{i-1}]$  into  $[d_{i-1}] = S_1 \cup S_2 \cup \dots \cup S_q$  such that  $|S_l| \leq L^{\gamma/2} (\forall l \in [q])$  and  $q = O\left(\frac{d_{i-1}}{L^{\gamma/2}}\right)$ . For each  $l \in [q]$ , let  $\mathcal{N}_l$  be a  $\frac{1}{2}$ -net for all the unit vectors in  $\mathbb{R}^{d_{i-1}}$  with support in  $S_l$ . Note that we can choose  $\mathcal{N}_l$  such that  $|\mathcal{N}_l| = e^{O(|S_l|)} = e^{O(L^{\gamma/2})}$ . Taking a union bound over  $\cup_{l=1}^q \mathcal{N}_l$ , we know that  $\| A_{j:i}(0)v \| \leq e^{-\Omega(L^\gamma)} \| v \|$  simultaneously for all  $v \in \cup_{l=1}^q \mathcal{N}_l$  with probability at least  $1 - (\sum_{l=1}^q |\mathcal{N}_l|)e^{-\Omega(L^\gamma)} \geq 1 - q \cdot e^{O(L^{\gamma/2})}e^{-\Omega(L^\gamma)} = 1 - e^{-\Omega(L^\gamma)}$ .

Now, for any  $u \in \mathbb{R}^{d_{i-1}}$ , we write it as  $u = \sum_{l=1}^{q} a_{l} u_{l}$  where  $a_{l}$  is a scalar and  $u_{l}$  is a unit vector supported on  $S_{l}$ . By the definition of  $\frac{1}{2}$ -net, for each  $l \in [q]$  there exists  $v_{l} \in \mathcal{N}_{l}$  such that  $\| v_{l} - u_{l} \| \leq \frac{1}{2}$ . We know that  $\| A_{j:i}(0) v_{l} \| \leq e^{-\Omega(L^{\gamma})} \| v_{l} \|$  for all  $l \in [q]$ . Let  $v = \sum_{l=1}^{q} a_{l} v_{l}$ . We have

$$
\begin{array}{l} \| A _ {j: i} (0) v \| \leq \sum_ {l = 1} ^ {q} | a _ {l} | \cdot \| A _ {j: i} (0) v _ {l} \| \leq \sum_ {l = 1} ^ {q} | a _ {l} | \cdot e ^ {- \Omega (L ^ {\gamma})} \| v _ {l} \| \leq e ^ {- \Omega (L ^ {\gamma})} \sqrt {q \cdot \sum_ {l = 1} ^ {q} a _ {l} ^ {2} \| v _ {l} \| ^ {2}} \\ = \sqrt {q} e ^ {- \Omega (L ^ {\gamma})} \| v \| = e ^ {- \Omega (L ^ {\gamma})} \| v \|. \\ \end{array}
$$

Note that  $\| u - v\| = \| \sum_{l = 1}^{q}a_{l}(u_{l} - v_{l})\| = \sqrt{\sum_{l = 1}^{q}a_{l}^{2}\|u_{l} - v_{l}\|^{2}}\leq \sqrt{\frac{1}{4}\sum_{l = 1}^{q}a_{l}^{2}} = \frac{1}{2}\| u\|$ , which implies  $\| v\| \leq \frac{3}{2}\| u\|$ . Therefore we have

$$
\begin{array}{l} \left\| A _ {j: i} (0) u \right\| \leq \left\| A _ {j: i} (0) v \right\| + \left\| A _ {j: i} (0) (u - v) \right\| \leq e ^ {- \Omega \left(L ^ {\gamma}\right)} \| v \| + \left\| A _ {j: i} (0) \right\| \cdot \| u - v \| \\ \leq e ^ {- \Omega (L ^ {\gamma})} \cdot \frac {3}{2} \| u \| + \| A _ {j: i} (0) \| \cdot \frac {1}{2} \| u \| = e ^ {- \Omega (L ^ {\gamma})} \| u \| + \| A _ {j: i} (0) \| \cdot \frac {1}{2} \| u \|. \\ \end{array}
$$

The above inequality is valid for any  $u \in \mathbb{R}^{d_{i-1}}$ . Thus we can take the unit vector  $u$  that maximizes  $\| A_{j:i}(0)u \|$ . This gives us  $\| A_{j:i}(0) \| \leq e^{-\Omega(L^{\gamma})} + \frac{1}{2} \| A_{j:i}(0) \|$ , which implies  $\| A_{j:i}(0) \| \leq e^{-\Omega(L^{\gamma})}$ .

Finally, we take a union bound over all possible  $(i,j)$ . The failure probability is at most  $L^2 e^{-\Omega (L^\gamma)} = e^{-\Omega (L^\gamma)}$ .

The following lemma shows that the properties in Lemmas 5.2 and 5.3 are still to some extent preserved after applying small perturbations on all the weight matrices.

Lemma 5.4. Suppose that the initial weights satisfy  $\| A_{j:i}(0) \| \leq O(L^3)$  for all  $1 \leq i \leq j \leq L$ , and  $\| A_{j:i}(0) \| \leq e^{-c_1 L^\gamma}$  if  $j - i \geq \frac{L}{10}$ , where  $c_1 > 0$  is a universal constant. Then for another set of matrices  $A_1, \ldots, A_L$  satisfying  $\| A_i - A_i(0) \| \leq e^{-0.6c_1 L^\gamma}$  for all  $i \in [L]$ , we must have

$$
\begin{array}{l} \left\| A _ {j: i} \right\| \leq O \left(L ^ {3}\right), \quad \forall 1 \leq i \leq j \leq L, \\ \left\| A _ {j: i} \right\| \leq O \left(e ^ {- c _ {1} L ^ {\gamma}}\right), \quad \forall 1 \leq i \leq j \leq L, j - i \geq \frac {L}{4}. \tag {12} \\ \end{array}
$$

Proof. It suffices to show that the difference  $A_{j:i} - A_{j:i}(0)$  is tiny. Let  $\Delta_i = A_i - A_i(0)$ . We have  $A_{j:i} = (A_j(0) + \Delta_j)\dots (A_{i+1}(0) + \Delta_{i+1})(A_i(0) + \Delta_i)$ . Expanding this product, except for the one term corresponding to  $A_{j:i}(0)$ , every other term has the form  $A_{j:(k_s+1)}(0)\cdot \Delta_{k_s}\cdot A_{(k_s-1):(k_s+1)}(0)\cdot \Delta_{k_s-1}\dots \Delta_{k_1}\cdot A_{(k_1-1):i}(0)$ , where  $i\leq k_1 < \dots < k_s\leq j$ . By assumption, each  $\Delta_k$  has spectral norm  $e^{-0.6c_1L^\gamma}$ , and each  $A_{j':i'}(0)$  has spectral norm  $O(L^3)$ , so we have  $\| A_{j:(k_s+1)}(0)\cdot \Delta_{k_s}\cdot A_{(k_s-1):(k_s+1)}(0)\cdot \Delta_{k_{s-1}}\dots \Delta_{k_1}\cdot A_{(k_1-1):i}(0)\| \leq (e^{-0.6c_1L^\gamma})^s (O(L^3))^{s+1}$ . Therefore we have

$$
\begin{array}{l} \| A_{j:i} - A_{j:i}(0)\| \leq \sum_{s = 1}^{j - i + 1}\binom {j - i + 1}{s}\left(e^{-0.6c_{1}L^{\gamma}}\right)^{s}\bigl(O(L^{3})\bigr)^{s + 1} \\ \leq \sum_ {s = 1} ^ {j - i + 1} L ^ {s} \left(e ^ {- 0. 6 c _ {1} L ^ {\gamma}}\right) ^ {s} \left(O (L ^ {3})\right) ^ {s + 1} \leq O (L ^ {3}) \sum_ {s = 1} ^ {\infty} \left(O (L ^ {4}) e ^ {- 0. 6 c _ {1} L ^ {\gamma}}\right) ^ {s} \leq O (L ^ {3}) \sum_ {s = 1} ^ {\infty} (1 / 2) ^ {s} = O (L ^ {3}), \\ \end{array}
$$

which implies  $\| A_{j:i}\| \leq O(L^3)$  for all  $1\leq i\leq j\leq L$

The proof of the second part of the lemma is postponed to Appendix B.2.

![](images/2db1795022dfebf8e2b0434c120ec8c00ff501d345d6864807afdd94c484ddb4.jpg)

As a consequence of Lemma 5.4, we can control the objective value and the gradient at any point sufficiently close to the random initialization.

Lemma 5.5. For a set of weight matrices  $W_{1},\ldots ,W_{L}$  with  $A_{i} = W_{i} / (\sqrt{d_{i}}\sigma_{i})$  that satisfy (12), the objective and the gradient satisfy

$$
\begin{array}{l} 0. 4 \left\| Y \right\| _ {F} ^ {2} <   \ell \left(W _ {1}, \dots , W _ {L}\right) <   0. 6 \left\| Y \right\| _ {F} ^ {2}, \\ \| \nabla_ {W _ {i}} \ell (W _ {1}, \ldots , W _ {L}) \| \leq (\sqrt {d _ {i}} \sigma_ {i}) ^ {- 1} e ^ {- 0. 9 c _ {1} L ^ {\gamma}}, \quad \forall i \in [ L ]. \\ \end{array}
$$

The proof of Lemma 5.5 is given in Appendix B.3.

Finally, we can finish the proof of Theorem 5.1 using the above lemmas.

Proof of Theorem 5.1. From Lemmas 5.2 and 5.3, we know that with probability at least 0.9, we have (i)  $\| A_{j:i}(0)\| \leq O(L^3)$  for all  $1\leq i\leq j\leq L$ , and (ii)  $\| A_{j:i}(0)\| \leq e^{-c_1L^\gamma}$  if  $(i,j)$  further satisfies  $j - i\geq \frac{L}{10}$ . Here  $c_{1} > 0$  is a universal constant. From now on we are conditioned on these properties being satisfied. We suppose that the learning rate  $\eta$  is at most  $e^{0.2c_1L^\gamma}$ .

We say that a set of weight matrices  $W_{1},\ldots ,W_{L}$  are in the "initial neighborhood" if  $\| A_{i} - A_{i}(0)\| \leq e^{-0.6c_{1}L^{\gamma}}$  for all  $i\in [L]$ . From Lemmas 5.4 and 5.5 we know that in the "initial neighborhood" the objective value is always between  $0.4\| Y\| _F^2$  and  $0.6\| Y\| _F^2$ . Therefore we have to escape the "initial neighborhood" in order to get the objective value out of this interval.

Now we calculate how many iterations are necessary to escape the "initial neighborhood." According to Lemma 5.5, inside the "initial neighborhood" each  $W_{i}$  can move at most  $\eta (\sqrt{d_i}\sigma_i)^{-1}e^{-0.9c_1L^\gamma}$  in one iteration by definition of the gradient descent algorithm. In order to leave the "initial neighborhood," some  $W_{i}$  must satisfy  $\| W_{i} - W_{i}(0)\| = \sqrt{d_{i}}\sigma_{i}\| A_{i} - A_{i}(0)\| > \sqrt{d_{i}}\sigma_{i}e^{-0.6c_{1}L^{\gamma}}$ . In order to move this amount, the number of iterations has to be at least

$$
\frac {\sqrt {d _ {i}} \sigma_ {i} e ^ {- 0 . 6 c _ {1} L ^ {\gamma}}}{\eta (\sqrt {d _ {i}} \sigma_ {i}) ^ {- 1} e ^ {- 0 . 9 c _ {1} L ^ {\gamma}}} = \frac {d _ {i} \sigma_ {i} ^ {2} e ^ {0 . 3 c _ {1} L ^ {\gamma}}}{\eta} \geq \frac {1}{L ^ {O (1)}} \cdot \frac {e ^ {0 . 3 c _ {1} L ^ {\gamma}}}{e ^ {0 . 2 c _ {1} L ^ {\gamma}}} \geq e ^ {\Omega (L ^ {\gamma})}.
$$

This finishes the proof.

![](images/4acf89e2243c42dc9bf363b1cd2b493ef26dc02785d5ee5f738d903bdb0d5308.jpg)

# 6 CONCLUSION

In this work, we studied the effect of the initialization parameter values of deep linear neural networks on the convergence time of gradient descent. We found that when the initial weights are iid Gaussian, the convergence time grows exponentially in the depth unless the width is at least as large as the depth. In contrast, when the initial weight matrices are drawn from the orthogonal group, the width needed to guarantee efficient convergence is in fact independent of the depth. These results establish for the first time a concrete proof that orthogonal initialization is superior to Gaussian initialization in terms of convergence time.

# REFERENCES

Madhu S Advani and Andrew M Saxe. High-dimensional dynamics of generalization error in neural networks. arXiv preprint arXiv:1710.03667, 2017.  
Sanjeev Arora, Nadav Cohen, Noah Golowich, and Wei Hu. A convergence analysis of gradient descent for deep linear neural networks. arXiv preprint arXiv:1810.02281, 2018.  
Peter Bartlett, Dave Helmbold, and Phil Long. Gradient descent with identity initialization efficiently learns positive definite linear transformations. In International Conference on Machine Learning, pp. 520-529, 2018.  
Minmin Chen, Jeffrey Pennington, and Samuel S Schoenholz. Dynamical isometry and a mean field theory of rnns: Gating enables signal propagation in recurrent neural networks. arXiv preprint arXiv:1806.05394, 2018.  
Simon Du and Wei Hu. Width provably matters in optimization for deep linear neural networks. In International Conference on Machine Learning, pp. 1655-1664, 2019.  
Dar Gilboa, Bo Chang, Minmin Chen, Greg Yang, Samuel S Schoenholz, Ed H Chi, and Jeffrey Pennington. Dynamical isometry and a mean field theory of lstms and grus. arXiv preprint arXiv:1901.08987, 2019.  
Eldad Haber and Lars Ruthotto. Stable architectures for deep neural networks. Inverse Problems, 34(1):014004, 2017.  
Moritz Hardt and Tengyu Ma. Identity matters in deep learning. International Conference on Learning Representations, 2016.  
Mikael Henaff, Arthur Szlam, and Yann LeCun. Recurrent orthogonal networks and long-memory tasks. arXiv preprint arXiv:1602.06662, 2016.  
Geoffrey Hinton, Li Deng, Dong Yu, George E. Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N Sainath, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. IEEE Signal Processing Magazine, 29(6):82-97, 2012.  
Kenji Kawaguchi. Deep learning without poor local minima. In Advances in Neural Information Processing Systems, pp. 586-594, 2016.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Thomas Laurent and James von Brecht. A recurrent neural network without chaos. arXiv preprint arXiv:1612.06212, 2016.  
Thomas Laurent and James von Brecht. Deep linear networks with arbitrary loss: All local minima are global. In International Conference on Machine Learning, pp. 2908-2913, 2018.  
Quoc V Le, Navdeep Jaitly, and Geoffrey E Hinton. A simple way to initialize recurrent networks of rectified linear units. arXiv preprint arXiv:1504.00941, 2015.  
Zenan Ling and Robert C Qiu. Spectrum concentration in deep residual learning: a free probability approach. IEEE Access, 7:105212-105223, 2019.  
Haihao Lu and Kenji Kawaguchi. Depth creates no bad local minima. arXiv preprint arXiv:1702.08580, 2017.  
Zakaria Mhammedi, Andrew Hellicar, Ashfaqur Rahman, and James Bailey. Efficient orthogonal parametrisation of recurrent neural networks using householder reflections. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 2401-2409. JMLR.org, 2017.

Jeffrey Pennington, Samuel Schoenholz, and Surya Ganguli. Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice. In Advances in neural information processing systems, pp. 4785-4795, 2017.  
Jeffrey Pennington, Samuel S Schoenholz, and Surya Ganguli. The emergence of spectral universality in deep networks. arXiv preprint arXiv:1802.09979, 2018.  
Feng Qi and Qiu-Ming Luo. Bounds for the ratio of two gamma functions—from wendel's and related inequalities to logarithmically completely monotonic functions. Banach Journal of Mathematical Analysis, 6(2):132-158, 2012.  
Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. International Conference on Learning Representations, 2014.  
Ohad Shamir. Exponential convergence time of gradient descent for one-dimensional deep linear neural networks. arXiv preprint arXiv:1809.08587, 2018.  
Wojciech Tarnowski, Piotr Warchol, Stanisław Jastrzbski, Jacek Tabor, and Maciej Nowak. Dynamical isometry is achieved in residual networks in a universal way for any activation function. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 2221-2230, 2019.  
Eugene Vorontsov, Chiheb Trabelsi, Samuel Kadoury, and Chris Pal. On orthogonality and learning recurrent networks with long term dependencies. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 3570-3578. JMLR.org, 2017.  
Scott Wisdom, Thomas Powers, John Hershey, Jonathan Le Roux, and Les Atlas. Full-capacity unitary recurrent neural networks. In Advances in Neural Information Processing Systems, pp. 4880-4888, 2016.  
Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.  
Lechao Xiao, Yasaman Bahri, Jascha Sohl-Dickstein, Samuel Schoenholz, and Jeffrey Pennington. Dynamical isometry and a mean field theory of cnns: How to train 10,000-layer vanilla convolutional neural networks. In International Conference on Machine Learning, pp. 5389-5398, 2018.  
Chulhee Yun, Suvrit Sra, and Ali Jabbabaie. Global optimality conditions for deep neural networks. arXiv preprint arXiv:1707.02444, 2017.  
Yi Zhou and Yingbin Liang. Critical points of linear neural networks: Analytical forms and landscape properties. 2018.
