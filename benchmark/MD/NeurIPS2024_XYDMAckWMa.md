# Explicit Flow Matching: On The Theory of Flow Matching Algorithms with Applications

Anonymous Author(s)

Affiliation

Address

email

# Abstract

This paper proposes a novel method, Explicit Flow Matching (ExFM), for training and analyzing flow-based generative models. ExFM leverages a theoretically grounded loss function, ExFM loss (a tractable form of Flow Matching (FM) loss), to demonstrably reduce variance during training, leading to faster convergence and more stable learning. Based on theoretical analysis of these formulas, we derived exact expressions for the vector field (and score in stochastic cases) for model examples (in particular, for separating multiple exponents), and in some simple cases, exact solutions for trajectories. In addition, we also investigated simple cases of diffusion generative models by adding a stochastic term and obtained an explicit form of the expression for score. While the paper emphasizes the theoretical underpinnings of ExFM, it also showcases its effectiveness through numerical experiments on various datasets, including high-dimensional ones. Compared to traditional FM methods, ExFM achieves superior performance in terms of both learning speed and final outcomes.

# 1 Introduction

In recent years, there has been a remarkable surge in Deep Learning, wherein the advancements have transitioned from purely neural networks to tackling differential equations. Notably, Diffusion Models [16] have emerged as key players in this field. This models transform a simple initial distribution, usually a standard Gaussian distribution, into a target distribution via a solution of Stochastic Differentiable Equation (SDE) [1] or Ordinary Differentiable Equation (ODE)[2] with right-hand side representing a trained neural network. The Conditional Flow Matching (CFM) [9] technique, which we focus on in our research, is a promising approach for constructing probability distributions using conditional probability paths, which is notably a robust and stable alternative for training Diffusion Models. The development of the CFM-based approach includes various techniques and heuristics [4, 7, 13] aimed at improving convergence or quality of learning or inference. For example, in the works [19, 20, 10] it was proposed to straighten the trajectories between points by different methods, which led to serious modifications of the learning process. We refer the reader for example, to the paper [20] where different FM-based approaches are summarised, and to the paper [9] for the connection between Diffusion Models and CFM.

In our work, we introduced an approach which we called Explicit Flow Matching (ExFM), to consider the Flow Matching framework theoretically by modifying the loss and writing the explicit value of the vector field. Strictly speaking, the presented loss is a tractable form of the FM loss, see Eq. (5) of [9]. Base on this methods we can improve the convergence of the method in practical examples reducing the variance of the loss, but the main focus of our paper is on theoretical derivations.

Our method allows us to write an expression for the vector field in closed form for quite simple cases (Gaussian distributions), however, we note that Diffusion Models framework in the case of a Gaussian Mixture of two Gaussian as a target distribution is still under investigation, see recent publications [15, 8].

Our main contributions are:

1. A tractable form of the FM loss is presented, which reaches a minimum on the same function as the loss used in Conditional Flow Matching, but has a smaller variance;  
2. The explicit expression in integral form for the vector field delivering the minimum to this loss (therefore for Flow Matching loss) is presented.  
3. As a consequence, we derive expressions for the flow matching vector field and score in several particular cases (when linear conditional mapping is used, normal distribution, etc.);  
4. Analytical analysis of SGD convergence showed that our formula have better training variance on several cases;  
5. Numerical experiments show that we can achieve better learning results in fewer steps.

# 1.1 Preliminaries

Flow matching is well known method for finding a flow to connect samples from two distribution with densities  $\rho_0$  and  $\rho_{1}$ . It is done by solving continuity equation with respect to the time dependent vector field  $\overline{v} (x,t)$  and time-dependent density  $\rho (x,t)$  with boundary conditions:

$$
\left\{ \begin{array}{l} \frac {\partial \rho (x , t)}{\partial t} = - \operatorname {d i v} (\rho (x, t) \bar {v} (x, t)), \\ \rho (x, 0) = \rho_ {0} (x), \quad \rho (x, 1) = \rho_ {1} (x). \end{array} \right. \tag {1}
$$

Function  $\rho(x, t)$  is called probability density path. Typically, the distribution  $\rho_0$  is known and it is chosen for convenience reasons, for example, as standard normal distribution  $\rho(x) = \mathcal{N}(x \mid 0, I)$ . The distribution  $\rho_1$  is unknown and we only know the set of samples from it, so the problem is to approximate the vector field  $v(x, t) \approx \overline{v}(x, t)$  using these samples. To make problem (1) well defined, one usually imposes additional regularity conditions on the densities, such as smoothness. The rigorous justification of the obtained results we put in the Appendix, leaving the general formulations of theorems and ideas in the main text.

From a given vector field, we can construct a flow  $\phi_t$ , i.e., a time-dependent map, satisfying the ODE  $\frac{\partial \phi_t(x)}{\partial t} = v(\phi_t(x), t)$  with initial condition  $\phi_0(x) = x$ . Thus, one can sample a point  $x_0$  from the distribution  $\rho_0$  and then using this ODE obtain a point  $x_1 = \phi_1(x_0)$  which have a distribution approximately equal to  $\rho_1$ . For given boundary  $\rho_0$  and  $\rho_1$ , the vector field or path solutions are not the only solutions, but if we have found any solution, it will already allow us to sample from the unknown density  $rho_1$ . However, if the problem is more narrowly defined, e.g., one needs to have a map that is close to the Optimal Transport (OT) map, we have to impose additional constraints.

The problem of finding any vector field  $v$  is solved in conditional manner in the paper [9], where so-called Conditional Flow Matching (CFM) is present. Namely, the following loss function was introduced for the training a model  $v_{\theta}$  which depends on parameters  $\theta$

$$
L _ {\mathrm {C F M}} (\theta) = \mathbb {E} _ {t} \mathbb {E} _ {x _ {1}, x _ {0}} \| v _ {\theta} \left(\phi_ {t, x _ {1}} \left(x _ {0}\right), t\right) - \phi_ {t, x _ {1}} ^ {\prime} \left(x _ {0}\right) \| ^ {2}, \tag {2}
$$

where  $\phi_{t,x_1}(x_0)$  is some flow, conditioned on  $x_{1}$  (one can take  $\phi_{t,x_1}(x_0) = (1 - t)x_0 + tx_1 + \sigma_stx_0$  in the simplest case, where  $\sigma_s > 0$  is a small parameter need for this map to be invertable at any  $0\leq t\leq 1$ ). Hereinafter the dash indicates the time derivative. Time variable  $t$  is uniformly distributed:  $t\sim \mathcal{U}[0,1]$  and random variables  $x_0$  and  $x_{1}$  are distributed according to the initial and final distributions, respectively:  $x_0\sim \rho_0$ ,  $x_{1}\sim \rho_{1}$ . Below we omit specifying of the symbol  $\mathbb{E}$  the distribution by which the expectation is taken where it does not lead to ambiguity.

# 1.2 Why new method?

Model training using loss (2) have the following disadvantage: during training, due to the randomness of  $x_0$  and  $x_1$ , significantly different values can be presented for model as output value at close model

![](images/0656e381bbc36e9c0ff5da0ea220e535a23645a72729074d81d97efb47c89c66.jpg)

![](images/4bdea1be57a5522010f96edfd38a3948426a7dc4a0248933952284f95f76393c.jpg)

![](images/344692af66a5d4061bf7e79290e1fe0e36dc1409853cb88131b9f4c0f538e363.jpg)

![](images/4d60aa01df440c22e4a18a7403cf05ed9774bcbf47bf517a2c4b44272caf9647.jpg)  
Figure 1: (Left) The key novelty of our approach is that in classical CFM, highly divergent directions can appear in a small spatial area at similar times (left part). In our approach (right part) we average over these vectors, training the model on a smoothed unnoised vector field. (Right) The comparison evaluated dispersion norm over time parameter  $t$  for CFM and ExFM in matching standard Gaussian  $\rho_0 = \mathcal{N}(0, I)$  to general Gaussian  $\rho_1 = \mathcal{N}(\mu, \sigma^2 I)$  distributions. The y-axis represents the sum of dispersion vector components, denoted as  $|\mathbb{D}_{x,x_1}\Delta v(x,t)|$ . The left panel illustrates samples drawn from the  $\rho_0$  and  $\rho_1$  distributions, as well as the corresponding flows. The right panel depicts the dispersion trend over time for both CFM (black line) and ExFM (red line) objectives. The dotted lines correspond to the dispersion levels (in top-down order  $|\mathbb{D}x_1|$ ,  $|\mathbb{D}x_0|$ ,  $|\mathbb{D}x_1| / N$ ).

argument values  $(x_{t},t)$ . Indeed, a fixed point  $x_{t} = \phi_{t,x_{1}}(x_{0})$  can be obtained by an infinite set of  $x_0$  and  $x_{1}$  pairs, some of which are directly opposite, and at least for small times  $t$  the probability of these different directions may not be significantly different. At the same time, data  $\phi_{t,x_1}'(x_0)$  on which the model learns significantly different for such different positions of pairs  $x_0$  and  $x_{1}$ . Thus, the model is forced to do two functions during training: generalize and take the mathematical expectation (clean the data from noise).  
In our approach, see Fig. 1(a), we feed the model input with cleaned data with small variance. Thus, the model only needs to generalize the data, which happens much faster (in fewer training steps).  
Moreover, in the process of constructing the modified loss, we have developed the exact formula for the vector field, see Eq. (11), (34). The existence of an explicit formula for the vector field is of great importance not only from a theoretical but also from a practical point of view.

# 2 Main idea

# 2.1 Modified objective

Lets expand the last two mathematical expectations in the loss (2) and substitute variables using map  $\phi_{t,x_1}$ , passing from the point  $x_0$  to its position  $x_{t} = \phi_{t,x_{1}}(x_{0})$  at time  $t$ :

$$
\begin{array}{l} \mathbb {E} _ {x _ {1}, x _ {0}} \left\| v _ {\theta} \left(\phi_ {t, x _ {1}} \left(x _ {0}\right), t\right) - \phi_ {t, x _ {1}} ^ {\prime} \left(x _ {0}\right) \right\| ^ {2} = \iint \left\| v _ {\theta} \left(\phi_ {t, x _ {1}} \left(x _ {0}\right), t\right) - \phi_ {t, x _ {1}} ^ {\prime} \left(x _ {0}\right) \right\| ^ {2} \rho_ {0} \left(x _ {0}\right) \rho_ {1} \left(x _ {1}\right) d x _ {0} d x _ {1} \\ = \iint \left\| v _ {\theta} \left(x _ {t}, t\right) - \phi_ {t, x _ {1}} ^ {\prime} \left(\phi_ {t, x _ {1}} ^ {- 1} \left(x _ {t}\right)\right) \right\| ^ {2} \underbrace {\det  \left[ \partial \phi_ {t , x _ {1}} ^ {- 1} (x) / \partial x \Big | _ {x = x _ {t}} \right] \rho_ {0} \left(\phi_ {t , x _ {1}} ^ {- 1} \left(x _ {t}\right)\right)} _ {\rho_ {x _ {1}} \left(x _ {t}, t\right)} \rho_ {1} \left(x _ {1}\right) d x _ {t} d x _ {1} \\ = \mathbb {E} _ {x _ {1}, x _ {t} \sim \rho_ {x _ {1}} (\cdot , t)} \left\| v _ {\theta} (x _ {t}, t) - \phi_ {t, x _ {1}} ^ {\prime} \left(\phi_ {t, x _ {1}} ^ {- 1} (x _ {t})\right) \right\| ^ {2}. \tag {3} \\ \end{array}
$$

We assume, that the map  $\phi_{t,x_1}$  is invertible at each  $0 < t < 1$ , i.e. that  $\phi_{t,x_1}^{-1}(x_t)$  exists on this time interval and for all  $x_t = \{\phi_t(x_0) \mid \forall x_0 : \rho(x_0) > 0\}$ . Eq. (3) can be seen as a transition from expectation on the variable  $x_0 \sim \rho_0$  to expectation on the variable  $x_t \sim \rho_{x_1}(\cdot, t)$ , where  $\rho_{x_1}(x, t) = [\phi_{t,x_1}]_* \rho_0(x) \coloneqq \rho_0(\phi_{t,x_1}^{-1}(x)) \operatorname*{det}[\partial \phi_{t,x_1}^{-1}(x) / \partial x]$ . See paper [5] for details about the push-forward operator “*”. Our representation (3) is very similar to expression (9) of the cited paper [9], only we write it in terms of the conditional flow rather than the conditional vector field.

To obtain the modified loss, we return to end of the standard CFM loss representation in (3). It is written as the expectation over two random variables  $x_{1}$  and  $x_{t}$  having a common distribution density

$$
\left\{x _ {1}, x _ {t} \right\} \sim \rho_ {j} \left(x _ {1}, x _ {t}, t\right) = \rho_ {x _ {1}} \left(x _ {t}, t\right) \rho_ {1} \left(x _ {1}\right), \tag {4}
$$

which, generally speaking, is not factorizable. Let us rewrite this expectations in terms of two independent random variables, each of which have its marginal distribution. The marginal distribution  $\rho_{m}$  of  $x_{t}$  can be obtained via integration:

$$
\rho_ {m} \left(x _ {t}, t\right) = \int \rho_ {j} \left(x _ {1}, x _ {t}, t\right) \mathrm {d} x _ {1} = \int \rho_ {x _ {1}} \left(x _ {t}, t\right) \rho_ {1} \left(x _ {1}\right) \mathrm {d} x _ {1}, \tag {5}
$$

while the marginal distribution of  $x_{1}$  is just (unknown) function  $\rho_{1}$ . Let for convenience  $w(t,x_1,x) = \phi_{t,x_1}'(\phi_{t,x_1}^{-1}(x))^1$ . We have

$$
L _ {\mathrm {C F M}} (\theta) = \mathbb {E} _ {t, x _ {1}, x _ {t} \sim \rho_ {x _ {1}} (\cdot , t)} \left\| v _ {\theta} (x _ {t}, t) - w (t, x _ {1}, x _ {t}) \right\| ^ {2} =
$$

$$
\int_ {0} ^ {1} \iint \| v _ {\theta} (x _ {t}, t) - w (t, x _ {1}, x _ {t}) \| ^ {2} \rho_ {x _ {1}} (x, t) \rho_ {1} (x _ {1}) \mathrm {d} x _ {t} \mathrm {d} x _ {1} \mathrm {d} t =
$$

$$
\int_ {0} ^ {1} \iint \| v _ {\theta} (x _ {t}, t) - w (t, x _ {1}, x _ {t}) \| ^ {2} \left(\rho_ {x _ {1}} (x _ {t}, t) / \rho_ {m} (x _ {t}, t)\right) \rho_ {m} (x _ {t}, t) \rho_ {1} (x _ {1}) \mathrm {d} x _ {t} \mathrm {d} x _ {1} \mathrm {d} t =
$$

$$
\mathbb {E} _ {t, x _ {1}, x \sim \rho_ {m} (\cdot , t)} \| v _ {\theta} (x, t) - w (t, x _ {1}, x) \| ^ {2} \rho_ {c} (x | x _ {1}, t) / \rho_ {1} (x _ {1}), \tag {6}
$$

where we introduce a conditional distribution

$$
\rho_ {c} (x \mid x _ {1}, t) := \rho_ {x _ {1}} (x, t) \rho_ {1} (x _ {1}) / \rho_ {m} (x, t) := \rho_ {x _ {1}} (x, t) \rho_ {1} (x _ {1}) / \int \rho_ {x _ {1}} (x, t) \rho_ {1} (x _ {1}) \mathrm {d} x _ {1}. \tag {7}
$$

The key feature of the representation (6) is that the integration variables  $x_{1}$  and  $x$  are independent. Thus, we can evaluate them using Monte Carlo-like schemes in different ways. However, we go further and make a modification to this loss to reduce the variance of Monte Carlo methods.

# 2.2 New loss and exact expression for vector field

Note that so far the expression for  $L_{\mathrm{CFM}}$  have not changed, it has just been rewritten in different forms. Now we change this expression so that its numerical value, generally speaking, may be different, but the derivative of the model parameters will be the same. We introduce the following loss

$$
L _ {\mathrm {E x F M}} (\theta) = \mathbb {E} _ {t} \mathbb {E} _ {x \sim \rho_ {m}} \left\| v _ {\theta} (x, t) - \mathbb {E} _ {x _ {1} \sim \rho_ {1}} w (t, x _ {1}, x) \rho_ {c} (x | x _ {1}, t) / \rho_ {1} (x _ {1}) \right\| ^ {2} =
$$

$$
\int_ {0} ^ {1} \int \left\| v _ {\theta} (x, t) - \int w (t, x _ {1}, x) \times \rho_ {c} (x | x _ {1}, t) \mathrm {d} x _ {1} \right\| ^ {2} \rho_ {m} (x, t) \mathrm {d} x \mathrm {d} t. \tag {8}
$$

Theorem 2.1. Losses  $L_{CFM}$  in Eq. (2) and  $L_{ExFM}$  in Eq. (8) have the same derivative with respect to model parameters:

$$
\mathrm {d} L _ {C F M} (\theta) / \mathrm {d} \theta = \mathrm {d} L _ {E x F M} (\theta) / \mathrm {d} \theta . \tag {9}
$$

Proof is in the Appendix A.1.

In the presented loss  $L_{\mathrm{ExFM}}$ , the integration (outside the norm operator) proceeds on those variables on which the model depends, while inside this operator there are no other free variables. Thus, using this kind of loss, it is possible to find an exact analytical expression for the vector field for which the minimum of this loss is zero (unlike the loss  $L_{\mathrm{CFM}}$ ). Namely, we have

$$
v (x, t) = \int w (t, x _ {1}, x) \rho_ {c} (x | x _ {1}, t) \mathrm {d} x _ {1}. \tag {10}
$$

We can obtain the exact form of this vector field given the particular map  $\phi_{t,x_1}$ . For example, the following statement holds:

Corollary 2.2. Consider the linear conditioned flow  $\phi_{t,x_1}(x_0) = (1 - t)x_0 + tx_1$  which is inevitable as  $0\leq t < 1$ . Then  $w(t,x_1,x) = \frac{x_1 - x}{1 - t}$ ,  $\rho_{x_1}(x,t) = \rho_0\left(\frac{x - x_1t}{1 - t}\right)\frac{1}{(1 - t)^d}$  and the loss  $L_{ExFM}$  in Eq. (8) reaches zero value when the model of the vector field has the following analytical form

$$
v (x, t) = \int \left(x _ {1} - x\right) \rho_ {0} \left(\frac {x - x _ {1} t}{1 - t}\right) \rho_ {1} \left(x _ {1}\right) \mathrm {d} x _ {1} / \left((1 - t) \int \rho_ {0} \left(\frac {x - x _ {1} t}{1 - t}\right) \rho_ {1} \left(x _ {1}\right) \mathrm {d} x _ {1}\right). \tag {11}
$$

This is the exact value of the vector field whose flow translates the given distribution  $\rho_0$  to  $\rho_{1}$

Complete proofs are in the Appendix A.3.1. Note that the result (11) is not totally new, for example, a similar result (though in the form of a general expression rather than an explicit formula), was given in [19], Eq. (9). However, our contribution consists of both the general form (10) and practical and theoretical conclusions from it (see below).

Remark 2.3. In the case of the initial and final times  $t = 0$ , 1, Eq. (11) is noticeably simpler

$$
v (x, 0) = \mathbb {E} _ {x _ {1}} x _ {1} - x = \int x _ {1} \rho_ {1} (x _ {1}) \mathrm {d} x _ {1} - x. \quad v (x, 1) = x - \int x _ {0} \rho_ {0} (x _ {0}) \mathrm {d} x _ {0}. \tag {12}
$$

This expression for the initial velocity means that each point first tends to the center of mass of the unknown distribution  $\rho_{1}$  regardless of its initial position.

Extensions to SDE Now let the conditional map be stochastic:  $\phi_{t,x_1} = (1 - t)x_0 + tx_1 + \sigma_e(t)\epsilon$ , where  $\epsilon \sim \mathcal{N}(0,1)$ . Typically,  $\sigma_e(0) = \sigma_e(1) = 0$ , for example,  $\sigma_e(t) = t(1 - t)\sigma_e$ .

Note that this formulation covers (with appropriate selection of the  $\sigma_{e}(t)$  parameter) the case of diffusion models [20].

Then, we can write the exact solution for a so-called score and flow matching objective (see [20] for details)

$$
\mathcal {L} _ {[ \mathrm {S F} ] ^ {2} \mathrm {M}} (\theta) = \mathbb {E} \Big [ \underbrace {\| v _ {\theta} (x , t) - u _ {t} ^ {\circ} (x) \| ^ {2}} _ {\text {f l o w m a t c h i n g l o s s}} + \lambda (t) ^ {2} \underbrace {\| s _ {\theta} (x , t) - \nabla \log p _ {t} (x) \| ^ {2}} _ {\text {s c o r e m a t c h i n g l o s s}} \Big ].
$$

that corresponds to this map. In the last expression, the following explicit conditional expressions are considered in the cited paper for the case  $\sigma_{e}(t) = \sqrt{t(1 - t)}\sigma_{e}$

$$
u _ {t} ^ {\circ} (x) = \frac {1 - 2 t}{t (1 - t)} (x - (t x _ {1} + (1 - t) x _ {0})) + (x _ {1} - x _ {0}), \nabla \log p _ {t} (x) = \frac {t x _ {1} + (1 - t) x _ {0} - x}{\sigma_ {e} ^ {2} t (1 - t)}.
$$

The exact solution (our result, explicit analog of the Eq. (10) from [20]) under consideration has the form (44) and (46) and, for example for the for the Gaussian  $\rho_0$  this expressions reduced to the Eq. (49) and (50), correspondingly. See Appendix E for the details on this case.

Simple examples Consider the case of Standard Normal Distribution as  $\rho_0$  and Gaussian Mixture of two Gaussians as  $\rho_{1}$ . Vector field have a closed form (37) in this case, and we can fast numerically solve ODE for trajectories. Random generated trajectories and plot of the vector field are shown on Fig. 2 (a)-(b). Detailed explanation of this case is in the Sec. D.2. Another example is related to the case of a stochastic map in the form of Brownian Bridge, which briefly described in the last paragraph and considered in Sec. E.3.2 in details, see Fig. 2 (c)-(f). Note that at some  $\sigma_{e}$  values the trajectories are a little bit straightened in this case compared to the usual linear map, if we compare cases on the Fig. 6.

# 2.3 Training scheme based on the modified loss

Let us consider the difference between our new scheme based on loss  $L_{\mathrm{ExFM}}$  and the classical CFM learning scheme. As a basis for the implementation of the learning scheme, we take the open-source code from the works [20, 19].

Consider a general framework of numerical schemes in classical CFM. We first sample  $m$  random time variables  $t \sim \mathcal{U}[0,1]$ . Then we sample several values of  $x$ . To do this, we sample a certain number  $n$  samples  $\{x_0^i\}_{i=1}^n$  from the "noisy" distribution  $\rho_0$ , and the same number  $n$  of samples  $\{x_1^i\}_{i=1}^n$  from

![](images/3d87168f8848e76a6b2c85f55a3297f8cacfc3a70a1be82ec866700b0e0631fa.jpg)  
(a) GM trajectory- (b)ries

![](images/10499a4a2c10f47245549ed4420b3225648702a2d7972477516d0f2c7aae392c.jpg)  
(c) BB Trajecto-(d) BB VF,ries,  $\sigma_{e} = 3\sigma_{e} = 3$

![](images/2a2a940b806496c3def5ba0e7b44c2f85562abf234aaa81054dddf769f62a506.jpg)

![](images/f6a257ff2cf29c2cf7031b7a6e577447e5e504f1fca557217cae9b3563242230.jpg)

![](images/7287abdcee7d2b7248e63cc832fd292d5073c59d5eb67b665913d0295fb54c52.jpg)

![](images/a2b72dd9c1f588a18ca830386c208871d2df4fcce2a639ddef10369ce8e34ccc.jpg)  
Figure 2: Trajectories and vector field obtained in simple cases: (a)  $N = 80$  random trajectories from  $\mathcal{N}\left(\cdot \mid 0,1^2\right)$  to GM; (b) 2D plot of the vector field in this case (c)-(f)  $N = 40$  random trajectories from  $\mathcal{N}\left(\cdot \mid 0,1^2\right)$  to  $\mathcal{N}\left(\cdot \mid 2,3^2\right)$  and 2D plot of the vector field for different  $\sigma_{e}$  for the Brownian Bridge map  
(e) BB Trajecto-(f) BB VF,ries,  $\sigma_{e} = 10\sigma_{e} = 10$

the unknown distribution  $\rho_{1}$ . Then we pair them (according to some scheme), and get  $n$  samples as  $x^{j,i} = \phi_{t^j,x_1^i}(x_0^i)$  (e.g. a linear combination in the simple case of linear map:  $x^{j,i} = (1 - t^{j})x_{0}^{i} + t^{j}x_{1}^{i}$ ),  $\forall i = 1,2,\ldots ,n;\forall j = 1,2,\ldots ,m$ . Note, than one of the variable  $n$  or  $m$  (or both) can be equal to 1. At the step 2, the following discrete loss is constructed from the obtained samples

$$
L _ {\mathrm {C F M}} ^ {d} (\theta) = \sum_ {j = 1} ^ {m} \sum_ {i = 1} ^ {n} \left\| v _ {\theta} \left(x ^ {j, i}, t ^ {j}\right) - \phi_ {t ^ {j}, x _ {1} ^ {i}} ^ {\prime} \left(x _ {0} ^ {i}\right) \right\| ^ {2}. \tag {13}
$$

Finally, we do a standard gradient descent step to update model parameters  $\theta$  using this loss.

The first and last step in our algorithm is the same as in the standard algorithm, but the second step is significantly different. Namely, we additionally generate a sufficiently large number  $N \gg n \cdot m$  of samples  $\overline{x}_1$  from the unknown distribution  $\rho_1$ , sampling  $(N - n)$  new samples and adding to it the samples  $\{x_1^i\}_1^n$  that are already obtained on the previous step.

Then we form the following discrete loss which replaces the integral on  $x_{1}$  in  $L_{\mathrm{ExFM}}$  by its evaluation  $v^{d}$  by self-normalized importance sampling or rejection sampling (see Appendix B for details)

$$
L _ {\mathrm {E x F M}} ^ {d} (\theta) = \sum_ {j = 1} ^ {m} \sum_ {i = 1} ^ {n} \left\| v _ {\theta} \left(x ^ {j, i}, t ^ {j}\right) - v ^ {d} \left(x ^ {j, i}, t ^ {j}\right) \right\| ^ {2}. \tag {14}
$$

For example, if we use self-normalized importance sampling and assume that the Jacobian  $\operatorname*{det}\left[\partial \phi_{t,x_1}^{-1}(x) / \partial x\right]$  do not depend on  $x_{1}$ , we can write

$$
v ^ {d} (x, t) = \left(\sum_ {k = 1} ^ {N} w \left(t, \bar {x} _ {1} ^ {k}, x\right) \rho_ {0} \left(\phi_ {t, \bar {x} _ {1} ^ {k}} ^ {- 1} (x)\right)\right) / \sum_ {k = 1} ^ {N} \rho_ {0} \left(\phi_ {t, \bar {x} _ {1} ^ {k}} ^ {- 1} (x)\right). \tag {15}
$$

Theorem 2.4. Under mild conditions, the error variance of the integral gradient (9) using the Monte Carlo method (14) is lower than using formula (13) with the same number  $n \cdot m$  of samples for  $\{x\}$ .

Sketch of the proof is in the Appendix A.2. The steps of our scheme are formally summarized in Algorithm 1.

Particular case of linear map and Gaussian noise Let  $\phi_{t,x_1}$  be the linear flow:  $\phi_{t,x_1}(x_0) = (1 - t)x_0 + tx_1$ . and consider the case of standard normal distribution for the initial density  $\rho_0$ :  $\rho_0(x)\sim \mathcal{N}(x\mid 0,I)$ . Then in the case of using self-normalized importance sampling, we have

$$
v ^ {d} (x, t) = \sum_ {k = 1} ^ {N} \frac {\bar {x} _ {1} ^ {k} - x}{1 - t} \left(\operatorname {S o f t M a x} \left(Y ^ {1}, \dots , Y ^ {N}\right)\right) _ {k}, \quad \text {w h e r e} \quad Y ^ {k} = - \frac {1}{2} \frac {\left\| x - t \cdot \bar {x} _ {1} ^ {k} \right\| _ {\mathbb {R} ^ {d}} ^ {2}}{1 - t}. \tag {16}
$$

Here, the lower index  $k$  in SoftMax stands for the  $k$ -th component, and the SoftMax operation itself came about due to exponents in the Gaussian density as a more stable substitute for computing than directly through exponents.

Extension of other maps and initial densities  $\rho_0$  Common expression (10) can be reduced to closed form for the particular choices of density  $\rho_0$  and map  $\phi$  (consequently, expression for  $w$ ). We summarise several known approaches for which FM-based techniques can be applied in Table  $1^3$ . See Appendix C and D for derivations of formulas and for more extensions.

Table 1: Correspondence between some methods which can reduced to FM framework and our theoretical descriptions of them.  

<table><tr><td>Probability Path</td><td>q(z)</td><td>μt(z)</td><td>σt</td><td>Explicit expressions: vector field (VF) and score (S)</td></tr><tr><td>Var. Exploding [17]</td><td>ρ1(x1)</td><td>x1</td><td>σ1-t</td><td>VF: (32)</td></tr><tr><td>Var. Preserving [6]</td><td>ρ1(x1)</td><td>α1-tx1</td><td>√1-α12</td><td>VF: (31)</td></tr><tr><td>Flow Matching [9]</td><td>ρ1(x1)</td><td>tx1</td><td>tσs - t + 1</td><td>VF: (11) if σ = 0; and (26)</td></tr><tr><td>Independent CFM</td><td>ρ0(x0)ρ1(x1)</td><td>tx1 + (1 - t)x0</td><td>σ</td><td>VF: (10)</td></tr><tr><td>Schrödinger Bridge CFM [20]</td><td>ρ0(x0)ρ1(x1)</td><td>tx1 + (1 - t)x0</td><td>σ√t(1 - t)</td><td>Can be obtained by SDE using VF: (49), S:(50)</td></tr></table>

Complexity We assume that the main running time of the algorithm is spent on training the model, especially if it is quite complex. Thus, the running time of one training step depends crucially on the number  $n \cdot m$  of samples  $\{x\}$  and it is approximately the same for both algorithms: the addition of points  $\overline{x}_1$  entails only an additional calculation using formula (16), which can be done quickly and, moreover, can be simple parallelized.

# 2.4 Irreducible dispersion of gradient for CFM optimization

Ensuring the stability of optimization is vital. Let  $\Delta \theta$  be changes in parameters, obtained by SGD with step size  $\gamma /2$  applied to the functional from Eq. (13):

$$
\Delta v \left(x ^ {j, i}, t ^ {j}\right) = - \gamma \cdot \left(v \left(x ^ {j, i}, t ^ {j}\right) - v ^ {d} \left(x ^ {j, i}, t ^ {j}\right)\right). \tag {17}
$$

For simplification, we consider a function,  $v_{\theta}(x,t)$ , capable of perfectly fitting the CFM problem and providing an optimal solution for any point  $x$  and time  $t$ . For a linear conditional flow at a specific point  $x^{j,i} \sim \rho_{x_1^i}(\cdot ,t^j)$  at time  $t^j \sim U(0,1)$ , the update  $\Delta v(x^{j,i},t^j)$  can be represented as follows:

$$
\Delta v \left(x ^ {j, i}, t ^ {j}\right) = \gamma \left(x _ {1} ^ {i} - \hat {x} _ {0} ^ {i} - v \left(x ^ {j, i}, t ^ {j}\right)\right), \tag {18}
$$

where  $\hat{x}_0^i = \frac{x^{j,i} - t^j x_1^i}{1 - t^j}$ . We define the dispersion  $\mathbb{D}_{x,x_1}f(x,x_1)$  for  $x \sim \rho_{x_1}(\cdot, t)$  and  $x_1 \sim \rho_1$  as:

$$
\mathbb {D} _ {x, x _ {1}} f (x, x _ {1}) = \mathbb {E} _ {x, x _ {1}} f ^ {2} (x, x _ {1}) - (\mathbb {E} _ {x, x _ {1}} f (x, x _ {1})) ^ {2}. \tag {19}
$$

Proposition 2.5. At the time  $t = 0$ , the dispersion of update in the form (18) have the following element-wise lower bound:

$$
\mathbb {D} _ {x ^ {j, i}, x _ {1} ^ {i}} \Delta v (x ^ {j, i}, 0) = \gamma^ {2} \mathbb {D} _ {x _ {1} ^ {i}} x _ {1} ^ {i} + \gamma^ {2} \mathbb {D} _ {x ^ {j, i}, x _ {1} ^ {i}} (x ^ {j, i} + v (x ^ {j, i}, 0)) \geq \gamma^ {2} \mathbb {D} _ {x _ {1} ^ {i}} x _ {1} ^ {i}.
$$

Equality is reached when the model  $v(x^{j,i},0)$  has exact values equal to (12).

Given that the dispersion cannot be reduced with an increase in batch size, the only available option is to decrease the step size of the optimization method, i.e., reduce the learning rate slowing down the convergence. The situation is much better for the proposed loss in (14). We can express the update  $\Delta v(x^{j,i},t^j)$  in the case of ExFM objective as:

$$
\Delta v \left(x ^ {j}, t ^ {j}\right) = \gamma^ {2} \left(\sum_ {k = 1} ^ {N} x _ {1} ^ {k} \tilde {\rho} \left(x ^ {j, i} \mid x _ {1} ^ {k}, t ^ {j}\right) - x ^ {j, i} - v \left(x ^ {j, i}, t ^ {j}\right)\right), \tag {20}
$$

where  $x^{j,i} \sim \rho_{x_1^i}(\cdot, t^j)$ ,  $x_1^k \sim \rho_1$  and  $\tilde{\rho}\left(x^{j,i}|x_1^k,t^j\right) = \rho_0\left(\frac{x^{j,i} - t^j x_1^k}{1 - t^j}\right) / \sum_{k=1}^{N} \rho_0\left(\frac{x^{j,i} - t^j x_1^k}{1 - t^j}\right)$ . Similar to the derivations in the previous part, we can found simplified form for the dispersion of update at  $t = 0$ .

Proposition 2.6. At the time  $t = 0$ , the dispersion of update from (20) have the following element-wise lower bound:

$$
\mathbb {D} _ {x ^ {j, i}, x _ {1} ^ {k}} \Delta v (x ^ {j, i}, 0) = \frac {\gamma^ {2}}{N} \mathbb {D} _ {x _ {1} ^ {k}} x _ {1} ^ {k} + \gamma^ {2} \mathbb {D} _ {x ^ {j, i}, x _ {1} ^ {k}} (x ^ {j, i} + v (x ^ {j, i}, 0)) \geq \frac {\gamma^ {2}}{N} \mathbb {D} _ {x _ {1} ^ {k}} x _ {1} ^ {k}.
$$

Equality is reached when the model  $v(x^{j,i},0)$  has exact values equal to (12).

In comparison to CFM, the dispersion of the update is  $N$  times smaller than the dispersion of the target distribution and could be controlled without impeding convergence by adjusting the number of samples  $N$ . In Figure 1(b), we visually compare the dispersions of CFM and ExFM. The illustration aligns a standard normal distribution  $\mathcal{N}(0,I)$  with a shifted and scaled variant  $\mathcal{N}(\mu ,I\sigma^2)$ . ExFM yields lower dispersion throughout the range  $t\in [0,1]$ . Detailed analytical calculations of the optimal velocity  $v(x,t)$  and dispersion are provided in the Appendix G.

![](images/447477b1d48a3dca12d9b48669026f526b85cffd39d0aa74e702ecae82f70ce6.jpg)

![](images/2a353e785605eae4a32791cc261593f2173e391cfb50cf0dec4cd27d634d4297.jpg)

![](images/5b6c4f504c1b5b0968464c80832c2c82bef107cb1cac4e3e687e1b104ddc3d06.jpg)

![](images/d5b02e28c8eea92000ca35ffb72074bc60ee651c724ef259887b156e53447a8d.jpg)

![](images/850bac3c5cd5ae2b1892a0005a8293f9952d430eee7be89a1d81ea8aa5b1ce73.jpg)

![](images/9e74e25636d222e0e0e3510d6ee29f156490ba894fc6a0bd25c6a9678b19b6e2.jpg)

![](images/0678bb07f7bc476750e0b8164debd2629a93033fee78f5b363328d8f7111c4f4.jpg)

![](images/1e219dc34982587d6987ae065cb1f0bc4e44c633c182f1c735cc82fc8dfbb50b.jpg)

![](images/18b7587c9b2ea7932f88a30b7cd43a3935d22cf37a9e285db3e4727a9db3c82f.jpg)

![](images/e16379d1fa6ef43a600cfabae06acd7c14aec11f4b0e94f8b27fb9c3d0f743d0.jpg)

![](images/3e6761da5c439744ac58b72e1e17a45de49baec7a835bdc03bca93a568352973.jpg)

![](images/83e7ce187f46f4e732686e575f35e0c5748e864cd9e041af26bd1da9aef7d2d2.jpg)

![](images/4317572a86c6e9dce742113d3922d40cd67e6d500e5e397cc6192725e38a653e.jpg)

![](images/dc8eb3a52978cd72c16e6ae4e495a3408f7ec8ac98f60af8c46f62d233a4e54d.jpg)

![](images/505b0548baf04fc9c63f54483f6ea1e3af5b0f25b0efc35a5ae16c1168807b9b.jpg)  
(a) swissroll

![](images/c981c10e0058671e7ab7348f0fe53dd2c37d8d10e66aeb067a8f88a4eedeebbe.jpg)  
(b) moons

![](images/b1322cda15a57d6ae6cd2a743fb80af7f2a2a884b3e03673393c5edf776b346d.jpg)  
(c) circles

![](images/ce83964ec315ad40fdff562d3b36145503f7e5882336b1afbc39de77a8110aff.jpg)  
Figure 3: Visual comparison of methods on toy 2D data. First row are original samples, second row sampled by ExFM, third row sampled by CFM.  
(d) 2spirals

![](images/fa52db521ee8a6d4a75290f70509925f7171b68376b4b793d806dc113eca3105.jpg)  
(f) pinwheel  
checkerboard

![](images/cbf42e98ad4d6fe3028a0a6b0e265a4da9c6e59a54ac3311db0bedb17b57af00.jpg)  
(g) rings

![](images/29eedd641027aedc5c5da8521d7c58292fa32450693d48f34c935fd2a4a3b202.jpg)

Table 2: ExFM and CFM metrics comparison table on toy 2D data.  

<table><tr><td rowspan="2">DATA</td><td colspan="2">MSE TRAINING LOSS</td><td colspan="2">ENERGY DISTANCE</td></tr><tr><td>ExFM</td><td>CFM</td><td>ExFM</td><td>CFM</td></tr><tr><td>SWISSROLL</td><td>1.13E-02</td><td>2.12E+00</td><td>2.58e-03</td><td>1.07E-02</td></tr><tr><td>MOONS</td><td>9.96E-03</td><td>2.01E+00</td><td>2.74e-03</td><td>1.41E-02</td></tr><tr><td>8GAUSSIANS</td><td>2.40E-02</td><td>2.77E+00</td><td>4.90e-03</td><td>2.45E-02</td></tr><tr><td>CIRCLES</td><td>9.28E-03</td><td>2.79E+00</td><td>6.69e-04</td><td>1.32E-02</td></tr><tr><td>2SPIRALS</td><td>8.92E-03</td><td>2.34E+00</td><td>1.27e-03</td><td>8.35E-03</td></tr><tr><td>CHECKERBOARD</td><td>1.04E-02</td><td>3.12E+00</td><td>1.01e-02</td><td>1.63E-02</td></tr><tr><td>PINWHEEL</td><td>4.53E-03</td><td>2.12E+00</td><td>1.01e-03</td><td>9.22E-03</td></tr><tr><td>RINGS</td><td>8.60E-03</td><td>1.93E+00</td><td>3.55e-04</td><td>2.37E-03</td></tr></table>

Table 3: NLL comparison for ExFM, CFM and OT-CFM methods over 10 000 learning steps, mean and std taken from 10 sampling iterations.  

<table><tr><td>DATA</td><td>ExFM</td><td>CFM</td><td>OT-CFM</td></tr><tr><td>POWER</td><td>-8.51e-02 ± 4.85e-02</td><td>1.64E-01 ± 4.18E-02</td><td>5.22E-02 ± 3.92E-02</td></tr><tr><td>GAS</td><td>-5.53e+00 ± 3.66e-02</td><td>-5.00E+00 ± 2.56E-02</td><td>-5.48E+00 ± 2.90E-02</td></tr><tr><td>HEPMASS</td><td>2.16E+01 ± 6.31E-02</td><td>2.21e+01 ± 6.13e-02</td><td>2.16E+01 ± 4.32E-02</td></tr><tr><td>BSDS300</td><td>-1.29E+02 ± 8.40E-01</td><td>-1.29E+02 ± 8.97E-01</td><td>-1.32e+02 ± 6.39e-01</td></tr><tr><td>MINIBOONE</td><td>1.34e+01 ± 1.95e-04</td><td>1.42E+01 ± 1.29E-04</td><td>1.43E+01 ± 9.22E-05</td></tr></table>

# 3 Numerical Experiments

Toy 2D data We conducted unconditional density estimation among eight distributions. Additional details of the experiments see in the Appendix H. We commence the exposition of our findings by showcasing a series of classical 2-dimensional examples, as depicted in Fig. 3 and Table 2. Our observations indicate that ExFM adeptly handles complex distribution shapes is particularly noteworthy, especially considering its ability to do so within a small number of epochs. Additionally, the visual comparison underscores the evident superiority of ExFM over the CFM approach.

Tabular data We conducted unconditional density estimation on five tabular datasets, namely power, gas, hepmass, minibone, and BSDS300. Additional details of the experiments see in the Appendix H. The empirical findings obtained from the numerical experiments from Table 3 indicate a statistically significant improvement in the performance of our proposed method. Notably, ExFM demonstrates a notable acceleration in convergence rate.

High-dimensional data and additional experiments We conducted experiments on high-dimensional data, among them experiments on CIFAR10 and MNIST dataset. FID results on CIFAR10 shows slightly better score among sampled images.

Additional details of the experiments and sampled images see in the Appendix H.

Stochastic ExFM (ExFM-S) on toy 2D data We evaluated the performance of the stochastic version of ExFM (ExFM-S) with use of expressions given in Sec. E.3.2 on four standard toy datasets. The primary experimental setup follows that used in [19]. Additional details on the hyperparameters used are available in Appendix H. Based on the findings presented in Table 4, we determine that ExFM-S surpasses I-CFM on all four datasets in terms of generative performance  $(\mathcal{W}_2)$  and also outperforms in terms of OT optimality (NPE) on two of them, exhibiting similar results on the remaining datasets. It also demonstrates performance similar to OT-CFM. While ExFM-S is not as robust as the basic ExFM, it enables the matching of one dataset to another (moons  $\rightarrow$  8gaussians) as it does not necessitate the presence of an explicit formula for  $\rho_0$ . Among other things, this experiment demonstrates the feasibility of our methods when both distributions  $\rho_0$  and  $\rho_{1}$  are unknown.

Table 4: ExFM-S evaluation on four toy datasets ( $\mu \pm \sigma$  over three seeds). For comparison we take I-CFM, OT-CFM, and ExFM (no values for moons  $\rightarrow$  8gaussians due to the absence of explicit formula for  $\rho_0$ ). Performance in generative modeling ( $\mathcal{W}_2$ ) and dynamic OT optimality (NPE) is assessed. The best result for each metric is highlighted in bold. Instances where we outperform CFM are underscored.

<table><tr><td rowspan="2">Metric → Algorithm ↓Dataset →</td><td colspan="4">W2↓</td><td colspan="4">NPE↓</td></tr><tr><td>N → moons</td><td>N → 8gaussians</td><td>moons → 8gaussians</td><td>N → 2spiralns</td><td>N → moons</td><td>N → 8gaussians</td><td>moons → 8gaussians</td><td>N → 2spiralns</td></tr><tr><td>I-CFM</td><td>0.522 ± 0.015</td><td>0.647 ± 0.078</td><td>0.966 ± 0.21</td><td>1.662 ± 0.067</td><td>0.328 ± 0.051</td><td>0.209 ± 0.009</td><td>0.945 ± 0.025</td><td>0.098 ± 0.04</td></tr><tr><td>OT-CFM</td><td>0.427 ± 0.038</td><td>0.528 ± 0.053</td><td>0.569 ± 0.018</td><td>1.322 ± 0.052</td><td>0.065 ± 0.068</td><td>0.031 ± 0.018</td><td>0.074 ± 0.026</td><td>0.031 ± 0.02</td></tr><tr><td>ExFM</td><td>0.318 ± 0.010</td><td>0.445 ± 0.075</td><td>-</td><td>1.276 ± 0.043</td><td>0.382 ± 0.050</td><td>0.213 ± 0.023</td><td>-</td><td>0.069 ± 0.064</td></tr><tr><td>ExFM-S</td><td>0.486 ± 0.09</td><td>0.570 ± 0.053</td><td>0.728 ± 0.063</td><td>1.361 ± 0.181</td><td>0.35 ± 0.143</td><td>0.166 ± 0.039</td><td>0.946 ± 0.059</td><td>0.083 ± 0.059</td></tr></table>

# 4 Conclusions

The presented method introduces a new loss function in tractable form (in terms of integrals) that improves upon the existing Conditional Flow Matching approach. New loss as a function of the model parameters, reaches zero at its minimum. Thanks to this, we can: a) write an explicit expression for the vector field on which the loss minimum is achieved; b) get a smaller variance when training on the discrete version of the loss, therefore, we can learn the model faster and more accurately.

Numerical experiments conducted on toy 2D data show reliable outcomes under uniform conditions and parameters. Comparison of the absolute values of loss for the proposed method and for CFM for the same distributions show that the absolute values of loss for these models differ strikingly, by a factor of  $10^{2} - 10^{3}$ . Experiments on high-dimensional datasets also confirm the theoretical deductions about the variance reduction of our method. However, we emphasize that we do not expect to use the proposed method in its pure form. On the contrary, we expect that the theoretical implications of our formulas will contribute to the construction of better learning or inference algorithms in conjunction with other heuristics or methods.

Algebraic analysis of variance for some cases (in particular, for the case  $t = 0$  or for the case of two Gaussians as initial and final distributions) show an improvement in variance when using the new loss. However, it is rather difficult to analyze in the general case, for all times  $t$  and general distributions  $\rho_0$  and  $\rho_1$ .

Having the expression for the vector field and score in the form of integrals, we can explicitly write out their expressions for some simple cases; in the case of Gaussian distributions we can also write out the exact solution for the trajectories. Thus, our approach allows one to advance the theoretical study of FM-based and Diffusion Model-based frameworks.

# References

[1] Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions". In: arXiv preprint 2303.08797 (2023).  
[2] Michael S. Albergo and Eric Vanden-Eijnden. "Building Normalizing Flows with Stochastic Interpolants". In: International Conference on Learning Representations (ICLR) (2023).  
[3] Gabriel Cardoso et al. "BR-SNIS: Bias Reduced Self-Normalized Importance Sampling". In: Advances in Neural Information Processing Systems. Ed. by S. Koyejo et al. Vol. 35. Curran Associates, Inc., 2022, pp. 716-729. URL: https://proceedings.neurips.cc/paper_files/paper/2022/file/04bd683d5428d91c5fbb5a7d2c27064d-Paper-Conference.pdf.  
[4] Ricky T. Q. Chen and Yaron Lipman. "Riemannian Flow Matching on General Geometries". In: arXiv:2302.03660 (2023).  
[5] Ricky T. Q. Chen et al. “Neural Ordinary Differential Equations”. In: Advances in Neural Information Processing Systems. Ed. by S. Bengio et al. Vol. 31. Curran Associates, Inc., 2018. URL: https://proceedings.neurips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf.  
[6] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. 2020. arXiv: 2006.11239 [cs.LG].  
[7] Alexia Jolicoeur-Martineau, Kilian Fatras, and Tal Kachman. "Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees". In: arXiv:2309.09968 (2023). arXiv: 2309.09968 [cs.LG].  
[8] Puheng Li et al. "On the Generalization Properties of Diffusion Models". In: Advances in Neural Information Processing Systems. Ed. by A. Oh et al. Vol. 36. Curran Associates, Inc., 2023, pp. 2097-2127. URL: https://proceedings.neurips.cc/paper_files/paper/2023/file/06abed94583030dd50abe6767bd643b1-Paper-Conference.pdf.  
[9] Yaron Lipman et al. “Flow Matching for Generative Modeling”. In: The Eleventh International Conference on Learning Representations. 2023. URL: https://openreview.net/forum?id=PqvMRDCJT9t.  
[10] Xingchao Liu, Chengyue Gong, and qiang liu. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow". In: The Eleventh International Conference on Learning Representations. 2023. URL: https://openreview.net/forum?id=XVjTT1nw5z.  
[11] D. Martin et al. “A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics”. In: Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001. Vol. 2. 2001, 416–423 vol.2. DOI: 10.1109/ICCV.2001.937655.  
[12] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. "On Aliased Resizing and Surprising Subtleties in GAN Evaluation". In: CVPR. 2022.  
[13] Aram-Alexandre Pooladian et al. “Multisample Flow Matching: Straightening Flows with Minibatch Couplings”. In: Proceedings of the 40th International Conference on Machine Learning. Ed. by Andreas Krause et al. Vol. 202. Proceedings of Machine Learning Research. PMLR, July 2023, pp. 28100-28127. URL: https://proceedings.mlr.press/v202/pooladian23a.html.  
[14] Aaditya Ramdas, Nicolas García Trillos, and Marco Cuturi. "On wasserstein two-sample testing and related families of nonparametric tests". In: Entropy 19.2 (2017), p. 47.  
[15] Kulin Shah, Sitan Chen, and Adam Klivans. "Learning Mixtures of Gaussians Using the DDPM Objective". In: Thirty-seventh Conference on Neural Information Processing Systems. 2023. URL: https://openreview.net/forum?id=aig7sgdRfI.  
[16] Jascha Sohl-Dickstein et al. "Deep Unsupervised Learning using Nonequilibrium Thermodynamics". In: Proceedings of the 32nd International Conference on Machine Learning. Ed. by Francis Bach and David Blei. Vol. 37. Proceedings of Machine Learning Research. Lille, France: PMLR, July 2015, pp. 2256-2265. URL: https://proceedings.mlr.press/v37/sohl-dickstein15.html.  
[17] Yang Song and Stefano Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution". In: Neural Information Processing Systems (NeurIPS) (2019).  
[18] Gábor J Székely. "E-statistics: The energy of statistical samples". In: Bowling Green State University, Department of Mathematics and Statistics Technical Report 3.05 (2003), pp. 1-18.

[19] Alexander Tong et al. "Improving and generalizing flow-based generative models with minibatch optimal transport". In: Transactions on Machine Learning Research (2024). Expert Certification. ISSN: 2835-8856. URL: https://openreview.net/forum?id=CD9Snc73AW.  
[20] Alexander Tong et al. "Simulation-Free Schrödinger Bridges via Score and Flow Matching". In: The 27th International Conference on Artificial Intelligence and Statistics. 2024. URL: https://virtual.aistats.org/virtual/2024/poster/6691.
