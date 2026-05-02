# PROXIMAL BACK PROPAGATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose proximal backpropagation (ProxProp) as a novel algorithm that takes implicit instead of explicit gradient steps to update the network parameters during neural network training. Our algorithm is motivated by the step size limitation of explicit gradient descent, which poses an impediment for optimization. ProxProp is developed from a general point of view on the backpropagation algorithm, currently the most common technique to train neural networks via stochastic gradient descent and variants thereof. Specifically, we show that backpropagation of a prediction error is equivalent to sequential gradient descent steps on a quadratic penalty energy, which comprises the network activations as variables of the optimization. We further analyze theoretical properties of ProxProp and in particular prove that the algorithm yields a descent direction in parameter space and can therefore be combined with a wide variety of convergent algorithms. Finally, we devise an efficient numerical implementation that integrates well with popular deep learning frameworks. We conclude by demonstrating promising numerical results and show that ProxProp can be effectively combined with common first order optimizers such as Adam.

# 1 INTRODUCTION

In recent years neural networks have gained considerable attention in solving difficult correlation tasks such as classification in computer vision (Krizhevsky et al., 2012) or sequence learning (Sutskever et al., 2014) and as building blocks of larger learning systems (Silver et al., 2016). Training neural networks is accomplished by optimizing a nonconvex, possibly nonsmooth, nested function of the network parameters. Since the introduction of stochastic gradient descent (SGD) (Robbins & Monro, 1951; Bottou, 1991), several more sophisticated optimization methods have been studied. One such class is that of quasi-Newton methods, as for example the comparison of L-BFGS with SGD in (Le et al., 2011), Hessian-free approaches (Martens, 2010), and the Sum of Functions Optimizer in (Sohl-Dickstein et al., 2013). Several works consider specific properties of energy landscapes of deep learning models such as frequent saddle points (Dauphin et al., 2014) and well-generalizable local optima (Chaudhari et al., 2017a). Among the most popular optimization methods in currently used deep learning frameworks are momentum based improvements of classical SGD, notably Nesterov's Accelerated Gradient (Nesterov, 1983; Sutskever et al., 2013), and the Adam optimizer (Kingma & Ba, 2015), which uses estimates of first and second order moments of the gradients for parameter updates.

Nevertheless, the optimization of these models remains challenging, as learning with SGD and its variants requires careful weight initialization and a sufficiently small learning rate in order to yield a stable and convergent algorithm. Moreover, SGD often has difficulties in propagating a learning signal deeply into a network, commonly referred to as the vanishing gradient problem (Hochreiter et al., 2001).

Training neural networks can be formulated as a constrained optimization problem by explicitly introducing the network activations as variables of the optimization, which are coupled via layerwise constraints to enforce a feasible network configuration. The authors of (Carreira-Perpinañán & Wang, 2014) have tackled this problem with a quadratic penalty approach, the method of auxiliary coordinates (MAC). Closely related, (Taylor et al., 2016) introduce additional auxiliary variables to further split linear and nonlinear transfer between layers and propose a primal dual algorithm for optimization. From a different perspective, (LeCun, 1988) takes a Lagrangian approach to formulate the constrained optimization problem.

![](images/7aef1be2c4d967890145cc87c2aa811701eb3c9de26994686da1ab558e2a44cd.jpg)  
Figure 1: Notation overview. For an  $L$ -layer feed-forward network we denote the explicit layer-wise activation variables as  $z_{l}$  and  $a_{l}$ . The transfer functions are denoted as  $\phi$  and  $\sigma$ . Layer  $l$  is of size  $n_l$ .

In this work, we start from a constrained optimization point of view on the classical backpropagation algorithm. We show that backpropagation can be interpreted as a method alternating between two steps. First, a forward pass of the data with the current network weights. Secondly, an ordered sequence of gradient descent steps on a quadratic penalty energy.

Using this point of view, instead of taking explicit gradient steps to update the network parameters associated with the linear transfer functions, we propose to use implicit gradient steps (also known as proximal steps, for the definition see (6)). We prove that such a model yields a descent direction and can therefore be used in a wide variety of (provably convergent) algorithms under weak assumptions. Since an exact proximal step may be costly, we further consider a matrix-free conjugate gradient (CG) approximation, which can directly utilize the efficient pre-implemented forward and backward operations of any deep learning framework. We prove that this approximation still yields a descent direction and demonstrate the effectiveness of the proposed approach in PyTorch.

# 2 MODEL AND NOTATION

We propose a method to train a general  $L$ -layer neural network of the functional form

$$
J (\boldsymbol {\theta}; X, y) = \mathcal {L} _ {y} \left(\phi \left(\theta_ {L - 1}, \sigma \left(\phi \left(\dots , \sigma \left(\phi \left(\theta_ {1}, X\right)\right) \dots\right)\right). \right. \right. \tag {1}
$$

Here,  $J(\theta; X, y)$  denotes the training loss as a function of the network parameters  $\theta$ , the input data  $X$  and the training targets  $y$ . As the final loss function  $\mathcal{L}_y$  we choose the softmax cross-entropy for our classification experiments.  $\phi$  is a linear transfer function and  $\sigma$  an elementwise nonlinear transfer function. As an example, for fully-connected neural networks  $\theta = (W, b)$  and  $\phi(\theta, a) = W a + b\mathbf{1}$ .

While we assume the nonlinearities  $\sigma$  to be continuously differentiable functions for analysis purposes, our numerical experiments indicate that the proposed scheme extends to rectified linear units (ReLU),  $\sigma(x) = \max(0, x)$ . Formally, the functions  $\sigma$  and  $\phi$  map between spaces of different dimensions depending on the layer. However, to keep the presentation clean, we do not state this dependence explicitly. Figure 1 illustrates our notation for the fully-connected network architecture.

Throughout this paper, we denote the Euclidean norm for vectors and the Frobenius norm for matrices by  $||\cdot ||$ , induced by an inner product  $\langle \cdot ,\cdot \rangle$ . We use the gradient symbol  $\nabla$  to denote the transpose of the Jacobian matrix, such that the chain rule applies in the form "inner derivative times outer derivative". For all computations involving matrix-valued functions and their gradient/Jacobian, we uniquely identify all involved quantities with their vectorized form by flattening matrices in a column-first order. Furthermore, we denote by  $A^{*}$  the adjoint of a linear operator  $A$ .

# 3 PENALTY FORMULATION OF BACK PROPAGATION

The gradient descent iteration on a nested function  $J(\pmb{\theta}; X, y)$ ,

$$
\boldsymbol {\theta} ^ {k + 1} = \boldsymbol {\theta} ^ {k} - \tau \nabla J (\boldsymbol {\theta} ^ {k}; X, y), \tag {2}
$$

is commonly implemented using the backpropagation algorithm (Rumelhart et al., 1986). As the basis for our proposed optimization method, we derive a connection between the classical backpropagation algorithm and quadratic penalty functions of the form

$$
E (\boldsymbol {\theta}, \boldsymbol {a}, \boldsymbol {z}) = \mathcal {L} _ {y} \left(\phi \left(\theta_ {L - 1}, a _ {L - 2}\right)\right) + \sum_ {l = 1} ^ {L - 2} \frac {\gamma}{2} \| \sigma (z _ {l}) - a _ {l} \| ^ {2} + \frac {\rho}{2} \| \phi \left(\theta_ {l}, a _ {l - 1}\right) - z _ {l} \| ^ {2}. \tag {3}
$$

<table><tr><td>Algorithm 1 - Penalty formulation of BackProp</td><td>Algorithm 2 - ProxProp</td></tr><tr><td>Input: Current parameters θk.</td><td>Input: Current parameters θk.</td></tr><tr><td>// Forward pass.</td><td>// Forward pass.</td></tr><tr><td>for l = 1 to L-2 do</td><td>for l = 1 to L-2 do</td></tr><tr><td>zlk= φ(θl,k, al-1), /a0=X.</td><td>zlk= φ(θl,k, al-1), /a0=X.</td></tr><tr><td>alk= σ(ztk).</td><td>alk= σ(ztk).</td></tr><tr><td>end for</td><td>end for</td></tr><tr><td>// Perform minimization steps on (3).</td><td>// Perform minimization steps on (3).</td></tr><tr><td>@grad. step on E wrt. (θL-1, al-2)</td><td>@grad. step on E wrt. (θL-1, al-2), Eqs. 8, 12.</td></tr><tr><td>for l = L-2 to 1 do</td><td>for l = L-2 to 1 do</td></tr><tr><td>@grad. step on E wrt. zl and al-1, @grad. step on E wrt. θl.</td><td>@grad. step on E wrt. zl and al-1, Eqs. 9, 10.</td></tr><tr><td>end for</td><td>@prox step on E wrt. θl, Eq. 11.</td></tr><tr><td>Output: New parameters θk+1.</td><td>end for</td></tr><tr><td></td><td>Output: New parameters θk+1.</td></tr></table>

The approach of (Carreira-Perpinañán & Wang, 2014) is based on the minimization of (3), as under mild conditions the limit  $\rho, \gamma \to \infty$  leads to the convergence of the sequence of minimizers of  $E$  to the minimizer of  $J$  (Nocedal & Wright, 2006, Theorem 17.1). In contrast to (Carreira-Perpinañán & Wang, 2014) we do not optimize (3), but rather use a connection of (3) to the classical backpropagation algorithm to motivate a semi-implicit optimization algorithm for the original cost function  $J$ .

Indeed, the iteration shown in Algorithm 1 of forward passes followed by a sequential gradient descent on the penalty function  $E$  is equivalent to the classical gradient descent iteration.

Proposition 1. Let  $\mathcal{L}_y$ ,  $\phi$  and  $\sigma$  be continuously differentiable. For  $\rho = \gamma = 1 / \tau$  and  $\theta^k$  as the input to Algorithm 1, its output  $\theta^{k + 1}$  satisfies (2), i.e., Algorithm 1 computes one gradient descent iteration on  $J$ .

Proof. For this and all further proofs we refer to Appendix A.

![](images/e7594bf56a36effd42222a0ccdac1e960be1daa8492e3fa1ef9bf1514da7b4e6.jpg)

# 4 PROXIMAL BACK PROPAGATION

The interpretation of Proposition 1 leads to the natural idea of replacing the explicit gradient steps  $\mathbf{a}$ ,  $\mathbf{b}$  and  $\mathbf{c}$  in Algorithm 1 with other – possibly more powerful – minimization steps. We propose Proximal Backpropagation (ProxProp) as one such algorithm that takes implicit instead of explicit gradient steps to update the network parameters  $\theta$  in step  $\mathbf{c}$ . This algorithm is motivated by the step size restriction of gradient descent.

# 4.1 GRADIENT DESCENT AND PROXIMAL MAPPINGS

Explicit gradient steps pose severe restrictions on the allowed step size  $\tau$ : Even for a convex, twice continuously differentiable,  $\mathcal{L}$ -smooth function  $f: \mathbb{R}^n \to \mathbb{R}$ , the convergence of the gradient descent algorithm can only be guaranteed for step sizes  $0 < \tau < 2 / \mathcal{L}$ . The Lipschitz constant  $\mathcal{L}$  of the gradient  $\nabla f$  is in this case equal to the largest eigenvalue of the Hessian  $H$ . With the interpretation of backpropagation as in Proposition 1, gradient steps are taken on quadratic functions. As an example for the first layer,

$$
f (\theta) = \frac {1}{2} \left\| \theta X - z _ {1} \right\| ^ {2}. \tag {4}
$$

In this case the Hessian is  $H = XX^{\top}$ , which is often ill-conditioned. For the CIFAR-10 dataset the largest eigenvalue is  $6.7 \cdot 10^{6}$ , which is seven orders of magnitude larger than the smallest eigenvalue. Similar problems also arise in other layers where poorly conditioned matrices  $a_{l}$  pose limitations for guaranteeing the energy  $E$  to decrease.

The proximal mapping (Moreau, 1965) of a function  $f: \mathbb{R}^n \to \mathbb{R}$  is defined as:

$$
\operatorname {p r o x} _ {\tau f} (y) := \underset {x \in \mathbb {R} ^ {n}} {\arg \min } f (x) + \frac {1}{2 \tau} | | x - y | | ^ {2}. \tag {5}
$$

By rearranging the optimality conditions to (5) and taking  $y = x^k$ , it can be interpreted as an implicit gradient step evaluated at the new point  $x^{k + 1}$  (assuming differentiability of  $f$ ):

$$
x ^ {k + 1} := \underset {x \in \mathbb {R} ^ {n}} {\operatorname {a r g m i n}} f (x) + \frac {1}{2 \tau} | | x - x ^ {k} | | ^ {2} = x ^ {k} - \tau \nabla f \left(x ^ {k + 1}\right). \tag {6}
$$

The iterative algorithm (6) is known as the proximal point algorithm (Martinet, 1970). In contrast to explicit gradient descent this algorithm is unconditionally stable, i.e. the update scheme (6) monotonically decreases  $f$  for any  $\tau > 0$ , since it holds by definition of the minimizer  $x^{k+1}$  that  $f(x^{k+1}) + \frac{1}{2\tau} ||x^{k+1} - x^k||^2 \leq f(x^k)$ .

Thus proximal mappings yield unconditionally stable subproblems in the following sense: The update in  $\theta_{l}$  provably decreases the penalty energy  $E(\pmb {\theta},\pmb{a}^{k},z^{k})$  from (3) for fixed activations  $(\pmb {a}^k,\pmb {z}^k)$  for any choice of step size. This motivates us to use proximal steps as depicted in Algorithm 2.

# 4.2 PROXPROP

We propose to replace explicit gradient steps with proximal steps to update the network parameters of the linear transfer function. More precisely, after the forward pass

$$
z _ {l} ^ {k} = \phi \left(\theta_ {l} ^ {k}, a _ {l - 1} ^ {k}\right), \tag {7}
$$

$$
a _ {l} ^ {k} = \sigma (z _ {l} ^ {k}),
$$

we keep the explicit gradient update equations for  $z_{l}$  and  $a_{l}$ . The last layer update is

$$
a _ {L - 2} ^ {k + 1 / 2} = a _ {L - 2} ^ {k} - \tau \nabla_ {a _ {L - 2}} \mathcal {L} _ {y} \left(\phi \left(\theta_ {L - 1}, a _ {L - 2}\right)\right), \tag {8}
$$

and for all other layers,

$$
z _ {l} ^ {k + 1 / 2} = z _ {l} ^ {k} - \sigma^ {\prime} \left(z _ {l} ^ {k}\right) \left(\sigma \left(z _ {l} ^ {k}\right) - a _ {l} ^ {k + 1 / 2}\right), \tag {9}
$$

$$
a _ {l - 1} ^ {k + 1 / 2} = a _ {l - 1} ^ {k} - \nabla \left(\frac {1}{2} \| \phi (\theta_ {l}, \cdot) - z _ {l} ^ {k + 1 / 2} \| ^ {2}\right) \left(a _ {l - 1} ^ {k}\right), \tag {10}
$$

where we use  $a_{l}^{k + 1 / 2}$  and  $z_{l}^{k + 1 / 2}$  to denote the updated variables before the forward pass of the next iteration and multiplication in (9) is componentwise. However, instead of taking explicit gradient steps to update the linear transfer parameters  $\theta_{l}$ , we take proximal steps

$$
\theta_ {l} ^ {k + 1} = \underset {\theta} {\operatorname {a r g m i n}} \frac {1}{2} \left\| \phi \left(\theta , a _ {l - 1} ^ {k}\right) - z _ {l} ^ {k + 1 / 2} \right\| ^ {2} + \frac {1}{2 \tau_ {\theta}} \left\| \theta - \theta_ {l} ^ {k} \right\| ^ {2}. \tag {11}
$$

This update can be computed in closed form as it amounts to a linear solve (for details see Appendix B). While in principle one can take a proximal step on the final loss  $\mathcal{L}_y$ , for efficiency reasons we choose an explicit gradient step, as the proximal step does not have a closed form solution in many scenarios (e.g. the softmax cross-entropy loss in classification problems). Specifically,

$$
\theta_ {L - 1} ^ {k + 1} = \theta_ {L - 1} ^ {k} - \tau \nabla_ {\theta_ {L - 1}} \mathcal {L} _ {y} \left(\phi \left(\theta_ {L - 1} ^ {k}, a _ {L - 2} ^ {k}\right)\right). \tag {12}
$$

Note that we have eliminated the step sizes in the updates for  $z_{l}$  and  $a_{l-1}$  in (9) and (10), as such updates correspond to the choice of  $\rho = \gamma = \frac{1}{\tau}$  in the penalty function (3) and are natural in the sense of Proposition 1. For the proximal steps in the parameters  $\theta$  in (11) we have introduced a step size  $\tau_{\theta}$  which – as we will see in Proposition 2 below – changes the descent metric opposed to  $\tau$  which rather rescales the magnitude of the update.

We refer to one sweep of updates according to equations (7) - (12) as ProxProp, as it closely resembles the classical backpropagation (BackProp), but replaces the parameter update by a proximal mapping instead of an explicit gradient descent step. In the following subsection we analyze the convergence properties of ProxProp more closely.

# 4.2.1 CONVERGENCE OF PROXPROP

ProxProp inherits all convergence-relevant properties from the classical backpropagation algorithm, despite replacing explicit gradient steps with proximal steps: It minimizes the original network energy  $J(\pmb{\theta}; X, y)$  as its fixed-points are stationary points of  $J(\pmb{\theta}; X, y)$ , and the update direction  $\pmb{\theta}^{k+1} - \pmb{\theta}^k$  is a descent direction such that it converges when combined with a suitable optimizer. In particular, it is straightforward to combine ProxProp with popular optimizers such as Nesterov's accelerated gradient descent (Nesterov, 1983) or Adam (Kingma & Ba, 2015).

In the following, we give a detailed analysis of these properties.

Proposition 2. For  $l = 1, \dots, L - 2$ , the update direction  $\pmb{\theta}^{k + 1} - \pmb{\theta}^k$  computed by ProxProp meets

$$
\theta_ {l} ^ {k + 1} - \theta_ {l} ^ {k} = - \tau \left(\frac {1}{\tau_ {\theta}} I + (\nabla \phi (\cdot , a _ {l - 1} ^ {k})) (\nabla \phi (\cdot , a _ {l - 1} ^ {k})) ^ {*}\right) ^ {- 1} \nabla_ {\theta_ {l}} J (\pmb {\theta} ^ {k}; X, y).
$$

In other words, ProxProp multiplies the gradient  $\nabla_{\theta_l}J$  with the inverse of the positive definite, symmetric matrix

$$
M _ {l} ^ {k} := \frac {1}{\tau_ {\theta}} I + (\nabla \phi (\cdot , a _ {l - 1} ^ {k})) (\nabla \phi (\cdot , a _ {l - 1} ^ {k})) ^ {*}, \tag {13}
$$

which depends on the activations  $a_{l-1}^k$  of the forward pass. Proposition 2 has some important implications:

Proposition 3. For any choice of  $\tau >0$  and  $\tau_{\theta} > 0$ , fixed points  $\pmb{\theta}^{*}$  of ProxProp are stationary points of the original energy  $J(\pmb {\theta};X,y)$ .

Moreover, we can conclude convergence in the following sense.

Proposition 4. The ProxProp direction  $\theta^{k + 1} - \theta^k$  is a descent direction. Moreover, under the (weak) assumption that the activations  $a_{l}^{k}$  remain bounded, the angle  $\alpha^k$  between  $\nabla_{\pmb{\theta}}J(\pmb{\theta}^k;X,y)$  and  $\theta^{k + 1} - \theta^k$  remains uniformly bounded away from  $\pi$ , i.e.

$$
\cos (\alpha^ {k}) > c \geq 0, \quad \forall k
$$

for some constant  $c$ .

Proposition 4 immediately implies convergence of a whole class of algorithms that depend only on a provided descent direction. We refer the reader to (Nocedal & Wright, 2006, Chapter 3.2) for examples and more details.

Furthermore, Proposition 4 states convergence for any minimization scheme in step  $\odot$  of Algorithm 2 that induces a descent direction in parameter space and thus provides the theoretical basis for a wide range of neural network optimization algorithms.

Considering the advantages of proximal steps over gradient steps, it is tempting to also update the auxiliary variables  $a$  and  $z$  in an implicit fashion. This corresponds to a proximal step in ⑥ of Algorithm 2. However, one cannot expect an analogue version of Proposition 3 to hold anymore. For example, if the update of  $a_{L-2}$  in (8) is replaced by a proximal step, the propagated error does not correspond to the gradient of the loss function  $\mathcal{L}_y$ , but to the gradient of its Moreau envelope. Consequently, one would then minimize a different energy. While in principle this could result in an optimization algorithm with, for example, favorable generalization properties, we focus on minimizing the original network energy in this work and therefore do not further pursue the idea of implicit steps on  $a$  and  $z$ .

# 4.2.2 INEXACT SOLUTION OF PROXIMAL STEPS

As we can see in Proposition 2, the ProxProp updates differ from vanilla gradient descent by the variable metric induced by the matrices  $(M_l^k)^{-1}$  with  $M_{l}^{k}$  defined in (13). Computing the ProxProp update direction  $v_{l}^{k} := \frac{1}{\tau} (\theta_{l}^{k + 1} - \theta_{l}^{k})$  therefore reduces to solving the linear equation

$$
M _ {l} ^ {k} v _ {l} ^ {k} = - \nabla_ {\theta_ {l}} J (\boldsymbol {\theta} ^ {k}; X, y), \tag {14}
$$

which requires an efficient implementation. We propose to use a conjugate gradient (CG) method, not only because it is one of the most efficient methods for iteratively solving linear systems in general, but also because it can be implemented matrix-free: It merely requires the application of the linear operator  $M_l^k$  which consists of the identity and an application of  $(\nabla \phi(\cdot, a_{l-1}^k)) (\nabla \phi(\cdot, a_{l-1}^k))^*$ . The latter, however, is preimplemented for many linear transfer functions  $\phi$  in common deep learning frameworks, because  $\phi(x, a_{l-1}^k) = (\nabla \phi(\cdot, a_{l-1}^k))^*(x)$  is nothing but a forward-pass in  $\phi$ , and  $\phi^*(z, a_{l-1}^k) = (\nabla \phi(\cdot, a_{l-1}^k))(z)$  provides the gradient with respect to the parameters  $\theta$  if  $z$  is the backpropagated gradient up to that layer. Therefore, a CG solver is straightforward to implement in any deep learning framework using the existing, highly efficient and high level implementations of  $\phi$  and  $\phi^*$ . For a fully connected network  $\phi$  is a matrix multiplication and for a convolutional network the convolution operation.

As we will analyze in more detail in Section 5.1, we approximate the solution to (14) with a few CG iterations, as the advantage of highly precise solutions does not justify the additional computational effort in obtaining them. Using any number of iterations provably does not harm the convergence properties of ProxProp:

Proposition 5. The direction  $\tilde{v}_l^k$  one obtains from approximating the solution  $v_{l}^{k}$  of (14) with the CG method remains a descent direction for any number of iterations.

# 4.2.3 CONVERGENCE IN THE STOCHASTIC SETTING

While the above analysis considers only the full batch setting, we remark that convergence of ProxProp can also be guaranteed in the stochastic setting under mild assumptions. Assuming that the activations  $a_{l}^{k}$  remain bounded (as in Proposition 4), the eigenvalues of  $(M_l^k)^{-1}$  are uniformly contained in the interval  $[\lambda ,\tau_{\theta}]$  for some fixed  $\lambda >0$ . Therefore, our ProxProp updates fulfill Assumption 4.3 in (Bottou et al., 2016), presuming the classic stochastic gradient fulfills them. This guarantees convergence of stochastic ProxProp updates in the sense of (Bottou et al., 2016, Theorem 4.9), i.e. for a suitable sequence of diminishing step sizes.

# 5 NUMERICAL EVALUATION

ProxProp generally fits well with the API provided by modern deep learning frameworks, since it can be implemented as a network layer with a custom backward pass for the proximal mapping. We chose PyTorch for our implementation. In particular, our implementation can use the API's GPU compute capabilities; all numerical experiments reported below were conducted on an NVIDIA Titan X GPU. To directly compare the algorithms, we used our own layer for either proximal or gradient update steps (cf. step  $\mathbb{C}$  in Algorithms 1 and 2). A ProxProp layer can be seamlessly integrated in a larger network architecture, also with other parametrized layers such as BatchNormalization.

# 5.1 EXACT AND APPROXIMATE SOLUTIONS TO PROXIMAL STEPS

We study the behavior of ProxProp in comparison to classical BackProp for a supervised visual learning problem on the CIFAR-10 dataset. We train a fully connected network with architecture  $3072 - 4000 - 1000 - 4000 - 10$  and ReLU nonlinearities. As the loss function, we chose the cross-entropy between the probability distribution obtained by a softmax nonlinearity and the ground-truth labels. We used a subset of 45000 images for training while keeping 5000 images as a validation set. We initialized the parameters  $\theta_{l}$  uniformly in  $[-1 / \sqrt{n_{l - 1}}, 1 / \sqrt{n_{l - 1}}]$ , the default initialization of PyTorch.

Figure 2 shows the decay of the full batch training loss over epochs (left) and training time (middle) for a Nesterov momentum<sup>1</sup> based optimizer using a momentum of  $\mu = 0.95$  and minibatches of size 500. We used  $\tau_{\theta} = 0.05$  for the ProxProp variants along with  $\tau = 1$ . For BackProp we chose  $\tau = 0.05$  as the optimal value we found in a grid search.

![](images/6319e9194f676a5a6faff725681a45d77018bc3cbc7d2569c29308c579fa565e.jpg)  
Figure 2: Exact and inexact solvers for ProxProp compared with BackProp. Left: A more precise solution of the proximal subproblem leads to overall faster convergence, while even a very inexact solution (only 3 CG iterations) already outperforms classical backpropagation. Center & Right: While the run time is comparable between the methods, the proposed ProxProp updates have better generalization performance ( $\approx$  54% for backprop and  $\approx$  56% for ours on the test set).

![](images/11d7dda5d867186df2bab168a911af9297fcb62d315270705592e5e32f0bac0d.jpg)

![](images/519d091258a405cc80c177510f0c50c2d7caa67fb609c698afd3a1dea67ca402.jpg)

As we can see in Figure 2 using implicit steps indeed improves the optimization progress per epoch. Thanks to powerful linear algebra methods on the  $\mathrm{GPU}^2$ , the exact ProxProp solution is competitive with BackProp even in terms of runtime.

The advantage of the CG-based approximations, however, is that they generalize to arbitrary linear transfer functions in a matrix-free manner, i.e. they are independent of whether the matrices  $M_{l}^{k}$  can be formed efficiently. Moreover, the validation accuracies (right plot in Figure 2) suggest that these approximations have generalization advantages in comparison to BackProp as well as the exact ProxProp method. Finally, we found the exact solution to be significantly more sensitive to changes of  $\tau_{\theta}$  than its CG-based approximations. We therefore focus on the CG-based variants of ProxProp in the following. In particular, we can eliminate the hyperparameter  $\tau_{\theta}$  and consistently chose  $\tau_{\theta} = 1$  for the rest of this paper, while one can in principle perform a hyperparameter search just as for the learning rate  $\tau$ . Consequently, there are no additional parameters compared with gradient descent using BackProp.

# 5.2 STABILITY FOR LARGER STEP SIZES

We compare the behavior of ProxProp and BackProp for different step sizes. Table 1 shows the final full batch training loss after 50 epochs with various  $\tau$ . The ProxProp based approaches remain stable over a significantly larger range of  $\tau$ . Even more importantly, deviating from the optimal step size  $\tau$  by one order of magnitude resulted in a divergent algorithm for classical BackProp, but still provides reasonable training results for ProxProp with 3 or 5 CG iterations. These results are in accordance with our motivation developed in Section 4.1. From a practical point of view, this eases hyperparameter search over  $\tau$ .

<table><tr><td>τ</td><td>50</td><td>10</td><td>5</td><td>1</td><td>0.5</td><td>0.1</td><td>0.05</td><td>\( 5 \cdot  {10}^{-3} \)</td><td>\( 5 \cdot  {10}^{-4} \)</td></tr><tr><td>BackProp</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.524</td><td>0.091</td><td>0.637</td><td>1.531</td></tr><tr><td>ProxProp (cg1)</td><td>77.9</td><td>0.079</td><td>0.145</td><td>0.667</td><td>0.991</td><td>1.481</td><td>1.593</td><td>1.881</td><td>2.184</td></tr><tr><td>ProxProp (cg3)</td><td>94.7</td><td>0.644</td><td>0.031</td><td>\( 2 \cdot {10}^{-3} \)</td><td>0.012</td><td>1.029</td><td>1.334</td><td>1.814</td><td>2.175</td></tr><tr><td>ProxProp (cg5)</td><td>66.5</td><td>0.190</td><td>0.027</td><td>\( 3 \cdot {10}^{-4} \)</td><td>\( 2 \cdot {10}^{-3} \)</td><td>0.399</td><td>1.049</td><td>1.765</td><td>2.175</td></tr></table>

Table 1: Full batch loss for conjugate gradient versions of ProxProp and BackProp after training for 50 epochs, where “-” indicates that the algorithm diverged to NaN. The implicit ProxProp algorithms remain stable for a significantly wider range of step sizes.

![](images/fd777bc0112c9db78741efed25ff8dd169dce93c43944995b59a7d555778b35d.jpg)  
CIFAR-10, Convolutional Neural Network

![](images/66d33b25765c901e5d9e3fdd4bddbee411a43e42804a7b6748ad641a0fe9993b.jpg)  
CIFAR-10, Convolutional Neural Network  
CIFAR-10, Convolutional Neural Network

![](images/3ba8bba6896abeeadbaacbf216fc7cfe68d98ea490819c85ce51eacabed1c697.jpg)  
CIFAR-10, Convolutional Neural Network

![](images/edeeb6d1c04a7c3c648b61382d4071d79ee8913ed0ebafe33ace17934ab79adb.jpg)  
Figure 3: ProxProp as a first-order oracle in combination with the Adam optimizer. The proposed method leads to faster decrease of the full batch loss in epochs and to an overall higher accuracy on the validation set. The plots on the right hand side show data for a fixed runtime, which corresponds to a varying number of epochs for the different optimizers.

# 5.3 PROXPROP AS A FIRST-ORDER ORACLE

We show that ProxProp can be used as a gradient oracle for first-order optimization algorithms. In this section, we consider Adam (Kingma & Ba, 2015). Furthermore, to demonstrate our algorithm on a generic architecture with layers commonly used in practice, we trained on a convolutional neural network of the form:

$$
\begin{array}{l} \operatorname {C o n v} [ 1 6 \times 3 2 \times 3 2 ] \rightarrow \operatorname {R e L U} \rightarrow \operatorname {P o o l} [ 1 6 \times 1 6 \times 1 6 ] \rightarrow \operatorname {C o n v} [ 2 0 \times 1 6 \times 1 6 ] \rightarrow \operatorname {R e L U} \\ \rightarrow \operatorname {P o o l} [ 2 0 \times 8 \times 8 ] \rightarrow \operatorname {C o n v} [ 2 0 \times 8 \times 8 ] \rightarrow \operatorname {R e L U} \rightarrow \operatorname {P o o l} [ 2 0 \times 4 \times 4 ] \rightarrow \operatorname {F C} + \operatorname {S o f t m a x} [ 1 0 \times 1 \times 1 ] \\ \end{array}
$$

Here, the first dimension denotes the respective number of filters with kernel size  $5 \times 5$  and max pooling downsamples its input by a factor of two. We set the step size  $\tau = 10^{-3}$  for both BackProp and ProxProp.

The results are shown in Fig. 3. Using parameter update directions induced by ProxProp within Adam leads to a significantly faster decrease of the full batch training loss in epochs. While the running time is higher than the highly optimized backpropagation method, we expect that it can be improved through further engineering efforts. We deduce from Fig. 3 that the best validation accuracy (72.9%) of the proposed method is higher than the one obtained with classical backpropagation (71.7%). Such a positive effect of proximal smoothing on the generalization capabilities of deep networks is consistent with the observations of Chaudhari et al. (2017b). Finally, the accuracies on the test set after 50 epochs are  $70.7\%$  for ProxProp and  $69.6\%$  for BackProp which suggests that the proposed algorithm can lead to better generalization.

# 6 CONCLUSION

We have proposed proximal backpropagation (ProxProp) as an effective method for training neural networks. To this end, we first showed the equivalence of the classical backpropagation algorithm with an algorithm that alternates between sequential gradient steps on a quadratic penalty function and forward passes through the network. Subsequently, we developed a generalization of backprop, which replaces explicit gradient steps with implicit (proximal) steps, and proved that such a scheme yields a descent direction, even if the implicit steps are approximated by conjugate gradient iterations. Our numerical analysis demonstrates that ProxProp is stable across various choices of step sizes and shows promising results when compared with common stochastic gradient descent optimizers.

We believe that the interpretation of gradient descent as the alternation between forward passes and sequential minimization steps on a penalty functional provides a theoretical basis for the development of further learning algorithms.

# REFERENCES

Léon Bottou. Stochastic gradient learning in neural networks. Proceedings of Neuro-Nimes, 91(8), 1991.  
Léon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. arXiv preprint arXiv:1606.04838, 2016.  
Miguel Á. Carreira-Perpínán and Weiran Wang. Distributed optimization of deeply nested systems. In Proceedings of the 17th International Conference on Artificial Intelligence and Statistics, AISTATS, 2014.  
Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina. Entropy-SGD: Biasing gradient descent into wide valleys. In Proceedings of the 5th International Conference on Learning Representations, ICLR, 2017a.  
Pratik Chaudhari, Adam Oberman, Stanley Osher, Stefano Soatto, and Guillame Carlier. Deep Relaxation: partial differential equations for optimizing deep neural networks. arXiv preprint arXiv:1704.04932, 2017b.  
Yann N. Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. In Proceedings of the 27th International Conference on Neural Information Processing Systems, NIPS, 2014.  
Sepp Hochreiter, Yoshua Bengio, and Paolo Frasconi. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies. In Field Guide to Dynamical Recurrent Networks. IEEE Press, 2001.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations, ICLR, 2015.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. ImageNet classification with deep convolutional neural networks. In Proceedings of the 25th International Conference of Neural Information Processing Systems, NIPS, 2012.  
Quoc V Le, Adam Coates, Bobby Prochnow, and Andrew Y Ng. On optimization methods for deep learning. In Proceedings of The 28th International Conference on Machine Learning, ICML, 2011.  
Yann LeCun. A theoretical framework for back-propagation. In Proceedings of the 1988 Connectionist Models Summer School, pp. 21-28, 1988.  
James Martens. Deep learning via Hessian-free optimization. In Proceedings of the 27th International Conference on Machine Learning, ICML, 2010.

Bernard Martinet. Régularisation d'inéquations variationnelles par approximations successives. Rev. Francaise Inf. Rech. Oper., pp. 154-159, 1970.  
Jean-Jacques Moreau. Proximate et dualité dans un espace hilbertien. Bulletin de la Société mathématique de France, 93:273-299, 1965.  
Yurii Nesterov. A method of solving a convex programming problem with convergence rate  $O(1 / k^2)$ . Soviet Mathematics Doklady, 27(2):372-376, 1983.  
Jorge Nocedal and Stephen Wright. Numerical Optimization. Springer Series in Operations Research and Financial Engineering. Springer, 2006.  
Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics, 22(3):400-407, 1951.  
David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by backpropagating errors. Nature, 323(6088):533-536, 1986.  
David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484-489, 2016.  
Jascha Sohl-Dickstein, Ben Poole, and Surya Ganguli. Fast large-scale optimization by unifying stochastic gradient and quasi-newton methods. In Proceedings of The 31st International Conference on Machine Learning, ICML, 2013.  
Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. In Proceedings of the 30th International Conference on International Conference on Machine Learning, ICML, 2013.  
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Proceedings of the 27th International Conference of Neural Information Processing Systems, NIPS, 2014.  
Gavin Taylor, Ryan Burmeister, Zheng Xu, Bharat Singh, Ankit Patel, and Tom Goldstein. Training neural networks without gradients: A scalable ADMM approach. In Proceedings of the 33rd International Conference on Machine Learning, ICML, 2016.
