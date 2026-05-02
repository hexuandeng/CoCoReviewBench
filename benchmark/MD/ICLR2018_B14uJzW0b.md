# NO SPURIOUS LOCAL MINIMA IN A TWO HIDDEN UNIT RELU NETWORK

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep learning models can be efficiently optimized via stochastic gradient descent, but there is little theoretical evidence to support this. A key question in optimization is to understand when the optimization landscape of a neural network is amenable to gradient-based optimization. We focus on a simple neural network: RELU network with one hidden layer consisting of two RELU units, and show that all local minimizers are global. This combined with recent work of Lee et al. (2017); Lee et al. (2016) show that gradient descent converges to the global minimizer.

# 1 INTRODUCTION

Deep learning has been used to achieve state-of-art performance on a wide variety of problems in machine learning, artificial intelligence, computer vision, and natural language processing. In all these applications, deep models often use hundreds of millions of parameters and are trained with stochastic gradient descent (or other gradient-based methods such as Adagrad (Duchi et al., 2011), Adam (Kingma and Ba, 2014)), a surprisingly simple method, and yet finds solutions with both low train and test error.

Despite the empirical success, the mathematical justification for gradient-based methods is not well-understood. Zhang et al. (2016a) empirically demonstrated that sufficiently over-parametrized networks can be efficiently optimized to near global optimality with stochastic gradient. For a two-layer network with leaky ReLU activation, Soudry and Carmon (2016) showed that gradient descent on a modified loss function can obtain a global minimum of the modified loss function; however, this does not imply reaching a global minimum of the original loss function. Under the same setting, Xie et al. (2016) showed that critical points with large "diversity" are nearly globally optimal. Choromanska et al. (2015) used several assumptions to simplify the loss function to a polynomial with i.i.d. Gaussian coefficients. They then showed that every local minima of the simplified loss has objective value comparable to the global minima. Kawaguchi (2016) used similar assumptions to show that all local minimum are global minimum in a nonlinear network. However the assumptions of Choromanska et al. (2015); Kawaguchi (2016) require independent activations, meaning that the activations of the hidden units are independent of the input and/or mutually independent, which is violated in practice.

Multiple works have been proposed to circumvent this assumption when dealing with the two-layer ReLU network  $F(x;W) = \sum_{j=1}^{K} \sigma(w_j^T x)$ , where  $\sigma = \max(0, x)$  is the ReLU activation function. Under the realizable setting (i.e., the labels are generated from a network with "teaching" parameters  $w^*$ ) and isotropic Gaussian input, Tian (2017) shows that when there is only a single ReLU node gradient descent converges to the global optimum. For  $K = 2$ , he conjectured that there are no spurious local minima, and provided a partial characterization of the critical point structure. With the same assumptions, Brutzkus and Globerson (2017) proved, for a two-layer ReLU network with a single non-overlapping convolutional filter, all local minimizers are global. Zhang et al. (2017a) show that for two-layer networks with non-standard activation functions that gradient descent converges to global minimizers.

In this paper, we focus on the case when  $K = 2$  and prove that every local minimum is global. As in previous works (Brutzkus and Globerson, 2017; Tian, 2017; Hardt and Ma, 2016), we focus on the population loss. The ReLU function is positive homogeneous, so we can rewrite the function as  $F(x;W) = v_{1}\sigma (w_{1}^{T}x) + v_{2}\sigma (w_{2}^{T}x)$  where  $w_{1}$  and  $w_{2}$  are unit vectors; for simplicity, we will

assume that  $v_{1} = v_{2} = 1$ . Using these assumptions and an additional orthogonality assumption, we prove that all local minima of the loss surface are global. Although the setting is a simplification of practical neural networks, this is a meaningful step towards understanding the success of gradient-based methods in deep learning and other non-convex optimization problems. For the non-orthogonal case, we provide a partial characterization of the critical point structure.

The paper is organized as follows: Section 2 discusses related works, and Section 3 introduces the notation and definitions. Section 4 shows our main result that all local minima are global and gives a proof sketch and the formal proofs are in Section 5. Section 6 provides some extensions to the non-orthogonal case. Section 7 presents the result of the experiments, and finally, Section 8 concludes the paper.

# 2 RELATED WORK

Single Hidden Node Networks: For a neural network with a single hidden unit and monotone activation function  $\sigma$ , numerous authors (Mei et al., 2016; Hazan et al., 2015; Kakade et al., 2011; Kalai and Sastry, 2009; Soltanolkotabi, 2017; Tian, 2017) have shown that gradient-based methods converge to the true parameter  $w^{*}$ . In the case of a single hidden unit, the loss function is weakly quasi-convex, meaning that the gradient points in the direction of  $w^{*}$ , which explains the success of gradient-based methods. For  $K > 1$  hidden units, the loss function is no longer quasi-convex, so this analysis does not easily generalize. In fact, our analysis for  $K = 2$  is considerably more involved, and requires analyzing the gradient and hessian simultaneously.

Improper Learning: On the improper learning side, Shalev-Shwartz et al. (2011) pioneered a kernel-based approach that can be used for learning a single halfspace or smoothed ReLU. This was generalized to fully-connected deep neural networks in Zhang et al. (2016b) using the recursive kernel method. Goel et al. (2016) designed a new smoothed ReLU function that is a better approximation to the ReLU. Instead of learning a neural network, these methods learn a function in a RKHS, hence improper learning. Zhang et al. (2017b) improved upon this by learning a neural network, instead of a kernel machine, via a boosting approach, and with much lower sample complexity. The disadvantages of improper learning are two-fold: 1) the sample complexity for these methods is exponentially larger than the Rademacher complexity of the network, and 2) the practical success of deep learning is intricately tied to using gradient-based training procedures, and the learnability of these networks using improper learning does not explain the success of gradient-based methods. On a related line of work, Janzamin et al. (2015) propose a method of moments estimator using tensor decomposition.

Over-Parametrization There have been several works on studying the effect of over-parametrization on the training of neural networks (Poston et al., 1991; Haeffele and Vidal, 2015). These results require the width of a hidden layer to be greater than the number of training samples, which is not the case for commonly used networks. Finally, Zhang et al. (2016a) empirically demonstrated that commonly used over-parametrized networks can be efficiently optimized to near global optimality with stochastic gradient descent.

Non-Convex Optimization: Since the loss function of neural networks is non-convex, the theory of training neural networks is closely related to the theory of non-convex optimization. Recently, there is considerable progress on convergence guarantees of first-order and second-order methods, including some applications in machine learning problems. Lee et al. (2016) and Lee et al. (2017) show gradient descent and other first-order methods converge only to local minima, and not saddle points. Jin et al. (2017) and Ge et al. (2015) show that variants of stochastic gradient method converge to local minimizers in polynomial time. Ge et al. (2016) and Ge et al. (2017) show there is no spurious local minima in matrix completion problem and non-convex low rank problems. For the phase retrieval problem, Sun et al. (2016) show that there is no spurious local minimum.

# 3 PRELIMINARIES

We study a simple two RELU hidden node network with output function

$$
F (x; w) = \sigma (w _ {1} ^ {T} x) + \sigma (w _ {2} ^ {T} x).
$$

For the duration of this paper, we will assume that  $x$  is standard normal in  $\mathbf{R}^n$  and all expectations are with respect to the standard normal. The population loss function is:

$$
L (x, W) = \frac {1}{2} \mathbf {E} \left[ \left(F (x, W) - F (x, W ^ {*})\right) ^ {2} \right]. \tag {1}
$$

Define

$$
g \left(v _ {1}, v _ {2}\right) = \mathbf {E} \left[ \sigma \left(v _ {1} ^ {T} x\right) \sigma \left(v _ {2} ^ {T} x\right) \right], \tag {2}
$$

so the loss can be rewritten as (ignoring additive constants, then multiplied by 4):

$$
f (W) = \sum_ {i, j \in \{1, 2 \}} \left(g \left(w _ {i}, w _ {j}\right) - 2 g \left(w _ {i}, w _ {j} ^ {*}\right)\right). \tag {3}
$$

From Brutzkus and Globerson (2017) we get

$$
g (u, v) = \frac {1}{2 \pi} \| u \| \| v \| (\sin \theta_ {u, v} - (\pi - \theta_ {u, v}) \cos \theta_ {u, v}). \tag {4}
$$

and

$$
\frac {\partial g}{\partial u} = \frac {1}{2 \pi} \| v \| \frac {u}{\| u \|} \sin \theta_ {u, v} + \frac {1}{2 \pi} (\pi - \theta_ {u, v}) v. \tag {5}
$$

In this paper, we study the landscape of  $f$  over the manifold  $\mathcal{R} = \{\| w_1\| = \| w_2\| = 1\}$ . The manifold gradient descent algorithm is:

$$
x _ {k + 1} = P _ {\mathcal {R}} \left(x _ {k} - \alpha \nabla_ {\mathcal {R}} f \left(x _ {k}\right)\right),
$$

where  $P_{\mathcal{R}}$  is the orthogonal projector onto the manifold  $\mathcal{R}$ , and  $\nabla_{\mathcal{R}}$  is the manifold gradient of  $f$ .

# 4 MAIN RESULT AND PROOF SKETCH

In order to analyze the global convergence of manifold gradient descent, we need a characterization of all critical points. We show that  $f(W)$  have no spurious local minimizer on the manifold  $\mathcal{R}$ .

Theorem 4.1. Assume  $\| w_1^* \| = \| w_2^* \| = 1$  and  $w_1^{*T}w_2^* = 0$ , then there is no spurious local minimizer of the objective function (3) on the manifold  $\mathcal{R} = \{\| w_1\| = \| w_2\| = 1\}$ . Furthermore, every saddle point or local maximizer has a direction of negative curvature.

The next theorem shows that manifold gradient descent with random initialization converges to the global minimizer

Theorem 4.2. With probability one, manifold gradient descent will converge to the global minimizers.

Proof. The objective function  $f$  is infinitely differentiable on manifold  $\mathcal{R}$ . Using Corollary 6 of Lee et al. (2017), manifold gradient descent will converge to a local minimizer with probability one. Since the only local minima for function  $f$  are  $w_{1} = w_{1}^{*}$ ,  $w_{2} = w_{2}^{*}$  and  $w_{1} = w_{2}^{*}$ ,  $w_{2} = w_{1}^{*}$ , manifold gradient descent converges to the true solutions.

Proof of Theorem 4.1. The proof of the main result is complicated, so let's start with a simpler case, in which both  $w_{1}$  and  $w_{2}$  are in  $\operatorname{span}\{w_1^*,w_2^*\}$ .

Proposition 4.3. Assume  $\| w_1^* \| = \| w_2^* \| = 1$ ,  $w_1^{*T}w_2^* = 0$  and  $w_{1},w_{2}\in \operatorname {span}\{w_{1}^{*},w_{2}^{*}\}$ , then there is no spurious local minimizer of the objective function (3) on the manifold  $\mathcal{R} = \{\| w_1\| = \| w_2\| = 1\}$ . Furthermore, every saddle point or local maximizer has a direction of negative curvature.

The complete proof is given in Appendix B and C, so here we just give a proof sketch.

To prove this, we need some observations. The first important observation is that we are always on manifold  $\{\| w_1\| = \| w_2\| = 1\}$ , and for each vector in the plane with fixed norm, there is only one degree of freedom, which means we can express each vector with only one variable. Thus, we can express the vectors in polar coordinates, where  $\theta_{1}$  and  $\theta_{2}$  are the angles for  $w_{1}$  and  $w_{2}$ .

The second observation is we only need to compute the gradient on the manifold and check whether it's zero. Define  $m(w_{1}) = \sin \theta_{1} \frac{\partial f}{\partial w_{11}} - \cos \theta_{1} \frac{\partial f}{\partial w_{12}}$  and  $m(w_{2}) = \sin \theta_{2} \frac{\partial f}{\partial w_{21}} - \cos \theta_{2} \frac{\partial f}{\partial w_{22}}$ . Then for  $w_{1}$  and  $w_{2}$ , the norm of the manifold gradients are  $|m(w_{1})|$  and  $|m(w_{2})|$ . Thus, we only need to check whether the value of function  $m$  is 0 and get rid of the absolute value sign.

Then we apply the polar coordinates onto the manifold gradients, and obtain:

$$
\begin{array}{l} m \left(w _ {2}\right) = \frac {1}{\pi} \left(\pi - \theta_ {w _ {1}, w _ {2}}\right) \sin \left(\theta_ {2} - \theta_ {1}\right) + \cos \theta_ {2} - \sin \theta_ {2} (6) \\ + \frac {1}{\pi} \left(\theta_ {w _ {2}, w _ {1} ^ {*}} \sin \theta_ {2} - \theta_ {w _ {2}, w _ {2} ^ {*}} \cos \theta_ {2}\right). (7) \\ \end{array}
$$

The last observation we need for this theorem is that we must divide this problem into several cases because each angle in (300) is a piecewise linear function. If we discuss each case independently, the resulting functions are linear in the angles. The details are in Appendix B. After the calculation of all cases, we found the positions of all the critical points: WLOG assume  $\theta_{1} \leq \theta_{2}$ , then there are four critical points in the 2D case:  $(\theta_{1}, \theta_{2}) = (0, \frac{\pi}{2}), (\frac{\pi}{4}, \frac{\pi}{4}), (\frac{\pi}{4}, \frac{5\pi}{4})$  and  $(\frac{5\pi}{4}, \frac{5\pi}{4})$ .

After finding all the critical points, we compute the manifold Hessian matrix for those points and show that there is a direction of negative curvature. The details can be found in Appendix C.

The next step is to reduce it to a three dimensional problem. As stated in the two-dimensional case, the gradient is in span  $\{w_{1}, w_{2}, w_{1}^{*}, w_{2}^{*}\}$ , which is four-dimensional. However, using the following lemma, we can reduce it to three dimensions and simplify the whole problem.

Lemma 4.4. If  $(w_{1}, w_{2})$  is a critical point, then there exists a set of standard orthogonal basis  $(e_{1}, e_{2}, e_{3})$  such that  $e_{1} = w_{1}^{*}$ ,  $e_{2} = w_{2}^{*}$  and  $w_{1}, w_{2}$  lies in  $\operatorname{span}\{e_{1}, e_{2}, e_{3}\}$ .

Even if we simplify the problem into three dimensional case, it still seems to be impossible to identify all critical points explicitly. Our method to analyze the landscape of the loss surface is to find the properties of critical points and then show all saddle points and local maximizers have a direction of negative curvature.

The following two lemmas captures the main geometrical properties of the critical points in three dimensional case. More detailed properties are given in Section 5.2

Lemma 4.5.

$$
\frac {\operatorname {a r c c o s} (- w _ {1 1})}{\operatorname {a r c c o s} (- w _ {2 1})} = \frac {\operatorname {a r c c o s} (- w _ {1 2})}{\operatorname {a r c c o s} (- w _ {2 2})} = - \frac {w _ {2 3}}{w _ {1 3}}. \tag {8}
$$

The ratio in Lemma 4.5 captures an important property of all critical points. For simplicity, based on D.5, we define  $k_{0} = -k$ ,  $\theta_{1} = \pi - \theta_{w_{2}, w_{1}^{*}}$  and  $\theta_{2} = \pi - \theta_{w_{2}, w_{2}^{*}}$ . Then

$$
\pi - \theta_ {w _ {1}, w _ {1} ^ {*}} = k _ {0} \theta_ {1} \tag {9}
$$

$$
\pi - \theta_ {w _ {1}, w _ {2} ^ {*}} = k _ {0} \theta_ {2}. \tag {10}
$$

Then from the properties of  $\theta_{1},\theta_{2}$  and upper bound the value of  $k_{0}$  we get

Lemma 4.6.  $\theta_{1} = \theta_{2}$

That lemma shows that  $w_{1}$  and  $w_{2}$  must be on a plane whose projection onto span  $\{w_{1}^{*}, w_{2}^{*}\}$  is the bisector of  $w_{1}^{*}$  and  $w_{2}^{*}$ . Combining this with the computation of Hessian, we conclude that we have found negative curvature for all possible critical points, which leads to the following proposition.

Proposition 4.7. Assume  $\| w_1^* \| = \| w_2^* \| = 1$ ,  $w_1^{*T}w_2^* = 0$  and  $\exists i\in [2],w_i\notin \operatorname {span}\{w_1^*,w_2^*\}$ , then there is no spurious local minimizer of the objective function (3) on the manifold  $\{\| w_1\| = \| w_2\| = 1\}$ . Furthermore, every saddle point or local maximizer has a direction of negative curvature.

Combining both Propositions 4.3 and 4.7, we have proved Theorem 4.1, which is the main result of this paper.

# 5 PROOFS

Here we provide some detailed proofs which are important for the understanding of the main theorem.

# 5.1 WHY WE ONLY NEED 3 DIMENSION

In general case, the following lemma shows we only need three dimension.

Lemma 5.1. If  $(w_{1}, w_{2})$  is a critical point, then there exists a set of standard orthogonal basis  $(e_{1}, e_{2}, e_{3})$  such that  $e_{1} = w_{1}^{*}$ ,  $e_{2} = w_{2}^{*}$  and  $w_{1}, w_{2}$  lies in  $\text{span}\{e_{1}, e_{2}, e_{3}\}$ .

Proof. If  $(w_{1}, w_{2})$  is a critical point, then

$$
\left(I - w _ {1} w _ {1} ^ {T}\right) \frac {\partial f}{\partial w _ {1}} = 0. \tag {11}
$$

where matrix  $(I - w_{1}w_{1}^{T})$  projects a vector onto the tangent space of  $w_{1}$ . Since

$$
(I - w _ {1} w _ {1} ^ {T}) w _ {1} = w _ {1} - w _ {1} = 0, \tag {12}
$$

we get

$$
\begin{array}{l} \left(I - w _ {1} w _ {1} ^ {T}\right) \frac {\partial f}{\partial w _ {1}} (13) \\ = \frac {1}{\pi} \left(I - w _ {1} w _ {1} ^ {T}\right) \left(\left(\pi - \theta_ {w _ {1}, w _ {2}}\right) w _ {2} - \left(\pi - \theta_ {w _ {1}, w _ {1} ^ {*}}\right) w _ {1} ^ {*} - \left(\pi - \theta_ {w _ {1}, w _ {2} ^ {*}}\right) w _ {2} ^ {*}\right), (14) \\ \end{array}
$$

which means that  $(\pi - \theta_{w_1, w_2})w_2 - (\pi - \theta_{w_1, w_1^*})w_1^* - (\pi - \theta_{w_1, w_2^*})w_2^*$  lies in the direction of  $w_1$ . If  $\theta_{w_1, w_2} = \pi$ , i.e.,  $w_1 = -w_2$ , then of course the four vectors have rank at most 3, so we can find the proper basis. If  $\theta_{w_1, w_2} < \pi$ , then we know that there exists a real number  $r$  such that

$$
\left(\pi - \theta_ {w _ {1}, w _ {2}}\right) w _ {2} - \left(\pi - \theta_ {w _ {1}, w _ {1} ^ {*}}\right) w _ {1} ^ {*} - \left(\pi - \theta_ {w _ {1}, w _ {2} ^ {*}}\right) w _ {2} ^ {*} + r \cdot w _ {1} = 0. \tag {15}
$$

Since  $\theta_{w_1, w_2} < \pi$ , we know that the four vectors  $w_1, w_2, w_1^*$  and  $w_2^*$  are linear dependent. Thus, they have rank at most 3 and we can find the proper basis.

# 5.2 SOME PROPERTIES OF CRITICAL POINTS

Next we will focus on the properties of critical points. Assume  $(w_{1},w_{2})$  is one of the critical points, from lemma D.1 we can find a set of standard orthogonal basis  $(e_1,e_2,e_3)$  such that  $e_1 = w_1^*$ ,  $e_2 = w_2^*$  and  $w_{1},w_{2}$  lies in span  $\{e_1,e_2,e_3\}$ . Furthermore, assume  $w_{1} = w_{11}e_{1} + w_{12}e_{2} + w_{13}e_{3}$  and  $w_{2} = w_{21}e_{1} + w_{22}e_{2} + w_{23}e_{3}$ , i.e.,  $w_{1} = (w_{11},w_{12},w_{13})$  and  $w_{2} = (w_{21},w_{22},w_{23})$ . Since we have already found out all the critical points when  $w_{13} = w_{23} = 0$ , in the following we assume  $w_{13}^{2} + w_{23}^{2}\neq 0$ .

First, we give the fundamental equation in our analysis.

Lemma 5.2.

$$
\frac {\operatorname {a r c c o s} (- w _ {1 1})}{\operatorname {a r c c o s} (- w _ {2 1})} = \frac {\operatorname {a r c c o s} (- w _ {1 2})}{\operatorname {a r c c o s} (- w _ {2 2})} = - \frac {w _ {2 3}}{w _ {1 3}}. \tag {16}
$$

Proof. Adapting from the proof of lemma D.4 and we know that

$$
\frac {w _ {2 1} - \frac {\pi - \theta_ {w _ {1} , w _ {1} ^ {*}}}{\pi - \theta_ {w _ {1} , w _ {2}}}}{w _ {1 1}} = \frac {w _ {2 2} - \frac {\pi - \theta_ {w _ {1} , w _ {2} ^ {*}}}{\pi - \theta_ {w _ {1} , w _ {2}}}}{w _ {1 2}} = \frac {w _ {2 3}}{w _ {1 3}} = k. \tag {17}
$$

Similarly, we have

$$
\frac {w _ {1 1} - \frac {\pi - \theta_ {w _ {2} , w _ {1} ^ {*}}}{\pi - \theta_ {w _ {1} , w _ {2}}}}{w _ {2 1}} = \frac {w _ {1 2} - \frac {\pi - \theta_ {w _ {2} , w _ {2} ^ {*}}}{\pi - \theta_ {w _ {1} , w _ {2}}}}{w _ {2 2}} = \frac {w _ {1 3}}{w _ {2 3}} = \frac {1}{k}. \tag {18}
$$

Taking the first component of (217) and (218) gives us

$$
w _ {2 1} = k \cdot w _ {1 1} + \frac {\pi - \theta_ {w _ {1} , w _ {1} ^ {*}}}{\pi - \theta_ {w _ {1} , w _ {2}}} \tag {19}
$$

$$
w _ {2 1} = k \cdot w _ {1 1} - k \frac {\pi - \theta_ {w _ {2} , w _ {1} ^ {*}}}{\pi - \theta_ {w _ {1} , w _ {2}}}. \tag {20}
$$

Thus,

$$
\frac {\pi - \theta_ {w _ {1} , w _ {1} ^ {*}}}{\pi - \theta_ {w _ {2} , w _ {1} ^ {*}}} = - k. \tag {21}
$$

Similarly, we get

$$
\frac {\pi - \theta_ {w _ {1} , w _ {2} ^ {*}}}{\pi - \theta_ {w _ {2} , w _ {2} ^ {*}}} = - k. \tag {22}
$$

Since  $\forall i,j\in [2],\pi -\theta_{w_i,w_j^*} = \arccos (-\theta_{w_{ij}})$ , we know that

$$
\frac {\operatorname {a r c c o s} (- w _ {1 1})}{\operatorname {a r c c o s} (- w _ {2 1})} = \frac {\operatorname {a r c c o s} (- w _ {1 2})}{\operatorname {a r c c o s} (- w _ {2 2})} = - \frac {w _ {2 3}}{w _ {1 3}}. \tag {23}
$$

![](images/aaf4535cc810f77572331703491dfd8e9e3acc2a1f0149ab9d7896c632748aa6.jpg)

Using this equation, we obtain several properties of critical points. The following two lemmas show the basic properties of critical points in three dimensional case. Completed proofs are given in Appendix B and C.

Lemma 5.3.  $\theta_{w_1,w_2} < \pi$

Lemma 5.4.  $w_{13} * w_{23} < 0$ .

These two lemmas restrict the position of critical points in some specific domains.

Then we construct a new function  $F$  in order to get more precise analysis. Define  $k_{0} = -k$ ,  $\theta_{1} = \pi - \theta_{w_{2}, w_{1}^{*}}$  and  $\theta_{2} = \pi - \theta_{w_{2}, w_{2}^{*}}$ .

$$
F (\theta) = \frac {- k _ {0} \theta}{k _ {0} \cos (k _ {0} \theta) + \cos (\theta)}, \tag {24}
$$

From the properties of that particular function and upper bound the value of  $k_{0}$  we get

Lemma 5.5.  $\theta_{1} = \theta_{2}$

That lemma shows that  $w_{1}$  and  $w_{2}$  must be on a plane whose projection onto span  $\{w_{1}^{*}, w_{2}^{*}\}$  is the bisector of  $w_{1}^{*}$  and  $w_{2}^{*}$ . Although we cannot identify the critical points explicitly, we will show these geometric properties already capture the direction of negative curvature.

# 6 ANALYSIS OF CRITICAL POINTS FOR NON-ORTHOGONAL  $W^{*}$

In this section, we partially characterize the structure of the critical points when  $w_1^*$ ,  $w_2^*$  are nonorthogonal, but form an acute angle. In other words, the angle between  $w_1^*$  and  $w_2^*$  is  $\alpha \in (0, \frac{\pi}{2})$ . Let us first consider the 2D cases, i.e., both  $w_1$  and  $w_2$  are in the span of  $w_1^*$  and  $w_2^*$ . Similar to the original problem, after the technique of changing variables (i.e., using polar coordinates and assume  $\theta_1$  and  $\theta_2$  are the angles of  $w_1$  and  $w_2$  in polar coordinates), we divide the whole plane into 4 parts, which are the angle in  $[0, \alpha]$ ,  $[\alpha, \pi]$ ,  $[\pi, \pi + \alpha]$  and  $[\pi + \alpha, 2\pi)$ . We have the following lemma:

Lemma 6.1. Assume  $\| w_1^* \| = \| w_2^* \| = 1$ ,  $w_1^{*T}w_2^* > 0$  and  $w_{1}, w_{2} \in \operatorname{span}\{w_{1}^{*}, w_{2}^{*}\}$ . When  $w_{1}$  and  $w_{2}$  are in the same part (one of four parts), the only critical points except the global minima are those when both  $w_{1}$  and  $w_{2}$  are on the bisector of  $w_{1}^{*}$  and  $w_{2}^{*}$ .

Proof. The complete proof is given in appendix E, the techniques are nearly the same as things in the original problem and a bit harder, so to be brief, we omit the proof details here.

For the three-dimensional cases cases of this new problem, it's interesting that the first few lemmatas are still true. Specifically, Lemma D.1(restated as Lemma 4.4) to Lemma D.5(restated as Lemma 4.5) are still correct. The proof is very similar to the proofs of those lemmas, except we need modification to the coefficients of terms in the expressions of the manifold gradients.

![](images/62076d6c4755fccf3096843829ce2bea0e9b765aaf1dd0dba8ac333853baad8c.jpg)  
Figure 1: Spurious Local Minima for  $K \geq 2$  ReLU Network.

# 7 EXPERIMENTS

We did experiments to verify the theoretical results. Since our results are restricted to the case of  $K = 2$  hidden units, it is also natural to investigate whether general two-layer ReLU networks also have the property that all local minima are global minima. Unfortunately as we show via numerical simulation, this is not the case. We consider the cases of  $K$  from 2 to 11 hidden units and we set the dimension  $d = K$ . For each  $K$ , the true parameters are orthogonal to each other. For each  $K$ , we run projected gradient descent with 300 different random initializations, and count the number of local minimum (critical points where the manifold Hessian is positive definite) with non-zero training error. If we reach a sub-optimal local minimum, we can conclude the loss surface exhibits spurious local minima. The bar plot showing the number of times gradient descent converged to spurious local minima is in Figure 1. From the plot, we see there is no spurious local minima from  $K = 2$  to  $K = 6$ . However for  $K \geq 7$ , we observe a clear trend that there are more spurious local minima when there are more hidden units.

# 8 CONCLUSION AND FUTURE WORK

In this paper, we provided recovery guarantee of stochastic gradient descent with random initialization for learning a two-layer neural network with two hidden nodes, unit-norm weights, ReLU activation functions and Gaussian inputs. Experiments are also done to verify our results. For future work, here we list some possible directions.

# 8.1 GENERAL CASE OF NETWORKS

This paper focused on a ReLU network with only two hidden units, . And the teaching weights must be orthogonal. Those are many conditions, in which we think there are some conditions that are not quite essential, e.g., the orthogonal assumption. In experiments we have already seen that even if they are not orthogonal, it still has some good properties such as the positions of critical points. Therefore, in the future we can further relax or abandon some of the assumptions of this paper and preserve or improve the result we have.

# 8.2 BAD LOCAL MINIMA

The neural network we discussed in this paper is in some sense very simple and far from practice, although it is already the most complex model when we want to analyze the whole loss surface. By experiments we have found that when it comes to seven hidden nodes with orthogonal true parameters,

there will be some bad local minima, i.e., there are some local minima that are not global. We believe that research in this paper can capture the characteristics of the whole loss surface and can help analyze the loss surface when there are three or even more hidden units, which may give some bounds on the performance of bad local minima and help us understand the specific non-convexity of loss surfaces.

# REFERENCES

A. Brutzkus and A. Globerson. Globally optimal gradient descent for a convnet with gaussian inputs. International Conference on Machine Learning (ICML), 2017.  
A. Choromanska, M. Henaff, M. Mathieu, G. B. Arous, and Y. LeCun. The loss surfaces of multilayer networks. In AISTATS, 2015.  
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul):2121-2159, 2011.  
R. Ge, J. D. Lee, and T. Ma. Matrix completion has no spurious local minimum. In Advances in Neural Information Processing Systems, pages 2973-2981, 2016.  
Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points—online stochastic gradient for tensor decomposition. arXiv:1503.02101, 2015.  
Rong Ge, Chi Jin, and Yi Zheng. No spurious local minima in nonconvex low rank problems: A unified geometric analysis. arXiv preprint arXiv:1704.00708, 2017.  
Surbhi Goel, Varun Kanade, Adam Klivans, and Justin Thaler. Reliably learning the relu in polynomial time. arXiv preprint arXiv:1611.10258, 2016.  
Benjamin D Haeffele and René Vidal. Global optimality in tensor factorization, deep learning, and beyond. arXiv preprint arXiv:1506.07540, 2015.  
Moritz Hardt and Tengyu Ma. Identity matters in deep learning. arXiv preprint arXiv:1611.04231, 2016.  
Elad Hazan, Kfir Levy, and Shai Shalev-Shwartz. Beyond convexity: Stochastic quasi-convex optimization. In Advances in Neural Information Processing Systems, pages 1594–1602, 2015.  
Majid Janzamin, Hanie Sedghi, and Anima Anandkumar. Beating the perils of non-convexity: Guaranteed training of neural networks using tensor methods. arXiv preprint arXiv:1506.08473, 2015.  
C. Jin, R. Ge, P. Netrapalli, S. M. Kakade, and M. I. Jordan. How to escape saddle points efficiently. arXiv preprint arXiv:1703.00887, 2017.  
Sham M Kakade, Varun Kanade, Ohad Shamir, and Adam Kalai. Efficient learning of generalized linear and single index models with isotonic regression. In Advances in Neural Information Processing Systems, pages 927-935, 2011.  
Adam Tauman Kalai and Ravi Sastry. The isotron algorithm: High-dimensional isotonic regression. In  $COLT$ , 2009.  
K. Kawaguchi. Deep learning without poor local minima. In Advances In Neural Information Processing Systems, pages 586-594, 2016.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
J. D. Lee, I. Panageas, G. Piliouras, M. Simchowitz, M. I. Jordan, and B. Recht. First-order methods almost always avoid saddle points. *ArXiv eprints*, 2017.  
Jason D Lee, Max Simchowitz, Michael I Jordan, and Benjamin Recht. Gradient descent converges to minimizers. University of California, Berkeley, 1050:16, 2016.

Song Mei, Yu Bai, and Andrea Montanari. The landscape of empirical risk for non-convex losses. arXiv preprint arXiv:1607.06534, 2016.  
Timothy Poston, C-N Lee, Y Choie, and Yonghoon Kwon. Local minima and back propagation. In Neural Networks, 1991., IJCNN-91-Seattle International Joint Conference on, volume 2, pages 173-176. IEEE, 1991.  
Shai Shalev-Shwartz, Ohad Shamir, and Karthik Sridharan. Learning kernel-based halfspaces with the 0-1 loss. SIAM Journal on Computing, 40(6):1623-1646, 2011.  
Mahdi Soltanolkotabi. Learning relus via gradient descent. arXiv preprint arXiv:1705.04591, 2017.  
D. Soudry and Y. Carmon. No bad local minima: Data independent training error guarantees for multilayer neural networks. arXiv preprint arXiv:1605.08361, 2016.  
J. Sun, Q. Qu, and J. Wright. A geometric analysis of phase retrieval. arXiv preprint arXiv:1602.06664, 2016.  
Yuandong Tian. An analytical formula of population gradient for two-layered relu network and its applications in convergence and critical point analysis. International Conference on Machine Learning (ICML), 2017.  
Bo Xie, Yingyu Liang, and Le Song. Diversity leads to generalization in neural networks. arXiv preprint arXiv:1611.03131, 2016.  
C. Zhang, S.y Bengio, M. Hardt, B. Recht, and O. Vinyals. Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530, 2016a.  
Q. Zhang, R. Panigrahy, S. Sachdeva, and A. Rahimi. Electron-proton dynamics in deep learning. arXiv preprint arXiv:1702.00458, 2017a.  
Yuchen Zhang, Jason D Lee, and Michael I Jordan. 11-regularized neural networks are improperly learnable in polynomial time. In International Conference on Machine Learning, pages 993-1001, 2016b.  
Yuchen Zhang, Jason Lee, Martin Wainwright, and Michael Jordan. On the learnability of fully-connected neural networks. In Artificial Intelligence and Statistics, pages 83-91, 2017b.
