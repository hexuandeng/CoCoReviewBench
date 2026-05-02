# TOPOLOGY AND GEOMETRY OF DEEP RECTIFIED NETWORK OPTIMIZATION LANDSCAPES

C. Daniel Freeman

Department of Physics

University of California at Berkeley

Berkeley, CA 94720, USA

daniel.freeman@berkeley.edu

Joan Bruna *

Courant Institute of Mathematical Sciences

New York University

New York, NY 10011, USA

bruna@cims.nyu.edu

# ABSTRACT

The loss surface of deep neural networks has recently attracted interest in the optimization and machine learning communities as a prime example of high-dimensional non-convex problem. Some insights were recently gained using spin glass models and mean-field approximations, but at the expense of strongly simplifying the nonlinear nature of the model.

In this work, we do not make any such assumption and study conditions on the data distribution and model architecture that prevent the existence of bad local minima. Our theoretical work quantifies and formalizes two important folklore facts: (i) the landscape of deep linear networks has a radically different topology from that of deep half-rectified ones, and (ii) that the energy landscape in the non-linear case is fundamentally controlled by the interplay between the smoothness of the data distribution and model over-parametrization. These results are in accordance with empirical practice and recent literature.

The conditioning of gradient descent is the next challenge we address. We study this question through the geometry of the level sets, and we introduce an algorithm to efficiently estimate the regularity of such sets on large-scale networks. Our empirical results show that these level sets remain connected throughout all the learning phase, suggesting a near convex behavior, but they become exponentially more curvy as the energy level decays, in accordance to what is observed in practice with very low curvature attractors.

# 1 INTRODUCTION

Optimization is a critical component in deep learning, governing its success in different areas of computer vision, speech processing and natural language processing. The prevalent optimization strategy is Stochastic Gradient Descent, invented by Robbins and Munro in the 50s. The empirical performance of SGD on these models is better than one could expect in generic, arbitrary non-convex loss surfaces, often aided by modifications yielding significant speedups Duchi et al. (2011); Hinton et al. (2012); Ioffe & Szegedy (2015); Kingma & Ba (2014). This raises a number of theoretical questions as to why neural network optimization does not suffer in practice from poor local minima.

The loss surface of deep neural networks has recently attracted interest in the optimization and machine learning communities as a paradigmatic example of a hard, high-dimensional, non-convex problem. Recent work has explored models from statistical physics such as spin glasses Choromanska et al. (2015), in order to understand the macroscopic properties of the system, but at the expense of strongly simplifying the nonlinear nature of the model. Other authors have advocated that the real danger in high-dimensional setups are saddle points rather than poor local minima Dauphin et al. (2014), although recent results rigorously establish that gradient descent does not get stuck on saddle points Lee et al. (2016) but merely slowed down. Other notable recent contributions are Kawaguchi (2016), which further develops the spin-glass connection from Choromanska et al. (2015), Sagun et al. (2014) which also discusses the impact of stochastic vs plain gradient, Soudry & Carmon (2016), that studies Empirical Risk Minimization for piecewise multilayer neural networks

under overparametrization (which needs to grow with the amount of available data), and Goodfellow et al. (2014), which provided insightful intuitions on the loss surface of large deep learning models and partly motivated our work. Lastly, the work Safran & Shamir (2015) studies some topological properties of homogeneous nonlinear networks and shows how overparametrization acts upon these properties, and the pioneering Shamir (2016) studied the distribution-specific hardness of optimizing non-convex objectives.

In this work, we do not make any linearity assumption and study conditions on the data distribution and model architecture that prevent the existence of bad local minima. The loss surface  $F(\theta)$  of a given model can be expressed in terms of its level sets  $\Omega_{\lambda}$ , which contain for each energy level  $\lambda$  all parameters  $\theta$  yielding a loss smaller or equal than  $\lambda$ . A first question we address concerns the topology of these level sets, i.e. under which conditions they are connected. Connected level sets imply that one can always find a descent direction at each energy level, and therefore that no poor local minima can exist. In absence of nonlinearities, deep (linear) networks have connected level sets Kawaguchi (2016). We first generalize this result to include ridge regression (in the two layer case) and provide an alternative, more direct proof of the general case. We then move to the half-rectified case and show that the topology is intrinsically different and clearly dependent on the interplay between data distribution and model architecture. Our main theoretical contribution is to prove that half-rectified single layer networks are asymptotically connected, and we provide explicit bounds that reveal the aforementioned interplay.

Beyond the question of whether the loss contains poor local minima or not, the immediate follow-up question that determines the convergence of algorithms in practice is the local conditioning of the loss surface. It is thus related not to the topology but to the shape or geometry of the level sets. As the energy level decays, one expects the level sets to exhibit more complex irregular structures, which correspond to regions where  $F(\theta)$  has small curvature. In order to verify this intuition, we introduce an efficient algorithm to estimate the geometric regularity of these level sets by approximating geodesics of each level set starting at two random boundary points. Our algorithm uses dynamic programming and can be efficiently deployed to study mid-scale CNN architectures on MNIST, CIFAR-10 and RNN models on Penn Treebank next word prediction. Our empirical results show that these models have a nearly convex behavior up until their lowest test errors, with a single connected component that becomes more elongated as the energy decays. The rest of the paper is structured as follows. Section 2 presents our theoretical results on the topological connectedness of multilayer networks. Section 3 presents our path discovery algorithm and Section 4 covers the numerical experiments.

# 2 TOPOLOGY OF LEVEL SETS

Let  $P$  be a probability measure on a product space  $\mathcal{X} \times \mathcal{Y}$ , where we assume  $\mathcal{X}$  and  $\mathcal{Y}$  are Euclidean vector spaces for simplicity. Let  $\{(x_i, y_i)\}_i$  be an iid sample of size  $L$  drawn from  $P$  defining the training set. We consider the classic empirical risk minimization of the form

$$
F _ {e} (\theta) = \frac {1}{L} \sum_ {l = 1} ^ {L} \| \Phi \left(x _ {i}; \theta\right) - y _ {i} \| ^ {2} + \kappa \mathcal {R} (\theta), \tag {1}
$$

where  $\Phi (x;\theta)$  encapsulates the feature representation that uses parameters  $\theta \in \mathbb{R}^S$  and  $\mathcal{R}(\theta)$  is a regularization term. In a deep neural network,  $\theta$  contains the weights and biases used in all layers. For convenience, in our analysis we will also use the oracle risk minimization:

$$
F _ {o} (\theta) = \mathbb {E} _ {(X, Y) \sim P} \| \Phi (X; \theta) - Y \| ^ {2} + \kappa \mathcal {R} (\theta). \tag {2}
$$

Our setup considers the case where  $\mathcal{R}$  consists on either  $\ell_1$  or  $\ell_2$  norms, as we shall describe below. They correspond to well-known sparse and ridge regularization respectively.

# 2.1 POOR LOCAL MINIMA CHARACTERIZATION FROM TOPOLOGICAL CONNECTEDNESS

We define the level set of  $F(\theta)$  as

$$
\Omega_ {F} (\lambda) = \left\{\theta \in \mathbb {R} ^ {S}; F (\theta) \leq \lambda \right\}. \tag {3}
$$

The first question we study is the structure of critical points of  $F_{e}(\theta)$  and  $F_{o}(\theta)$  when  $\Phi$  is a multi-layer neural network. In particular, we are interested to know whether  $F_{e}$  has local minima which are not global minima. This question is answered by knowing whether  $\Omega_{F}(\lambda)$  is connected at each energy level  $\lambda$ :

Proposition 2.1. If  $\Omega_F(\lambda)$  is connected for all  $\lambda$  then every local minima of  $F(\theta)$  is a global minima.

This proposition shows that a sufficient condition to prevent the existence of poor local minima is having connected level sets, but this condition is not necessary: one can have isolated local minima lying at the same energy level. This can be the case in systems that are defined up to a discrete symmetry group, such as multilayer neural networks. However, as we shall see next, this case puts the system in a brittle position, since one needs to be able to account for all the local minima (and there can be exponentially many of them as the parameter dimensionality increases) and verify that their energy is indeed equal.

# 2.2 THE LINEAR CASE

We first consider the particularly simple case where  $F$  is a multilayer network defined by

$$
\Phi (x; \theta) = W _ {K} \dots W _ {1} x, \theta = \left(W _ {1}, \dots , W _ {K}\right). \tag {4}
$$

and the ridge regression  $\mathcal{R}(\theta) = \| \theta \| ^2$ . This model defines a non-convex (and non-concave) loss  $F_{e}(\theta)$ . When  $\kappa = 0$ , it has been shown in Saxe et al. (2013) and Kawaguchi (2016) that in this case, every local minima is a global minima. We provide here an alternative proof of that result that uses a somewhat simpler argument and allows for  $\kappa > 0$  in the case  $K = 2$ .

Proposition 2.2. Let  $W_{1}, W_{2}, \ldots, W_{K}$  be weight matrices of sizes  $n_k \times n_{k+1}$ ,  $k < K$ , and let  $F_{e}(\theta)$ ,  $F_{o}(\theta)$  denote the risk minimizations using  $\Phi$  as in (4). Assume that  $n_j \geq \min(n_1, n_K)$  for  $j = 2 \ldots K-1$ . Then  $\Omega_{F_{e}}(\lambda)$  (and  $\Omega_{F_{o}}$ ) is connected for all  $\lambda$  and all  $K$  when  $\kappa = 0$ , and for  $\kappa > 0$  when  $K = 2$ ; and therefore there are no poor local minima in these cases.

This result highlights a certain mismatch between the picture of having no poor local minima and generalization. Incorporating regularization drastically changes the topology, and the fact that we are able to show connectedness only in the two-layer case with ridge regression is profound; we conjecture that extending it to deeper models requires a different regularization, perhaps using more general atomic norms Bach (2013). But we now move our interest to the nonlinear case, which is more relevant to our purposes.

# 2.3 HALF-RECTIFIED NONLINEAR CASE

We now study the setting given by

$$
\Phi (x; \theta) = W _ {K} \rho W _ {K - 1} \rho \dots \rho W _ {1} x, \theta = \left(W _ {1}, \dots , W _ {K}\right), \tag {5}
$$

where  $\rho (z) = \max (0,z)$ . The biases can be implemented by replacing the input vector  $x$  with  $\overline{x} = (x,1)$  and by rebranding each parameter matrix as

$$
\overline {{W}} _ {i} = \left( \begin{array}{c c} W _ {i} & b _ {i} \\ \hline 0 & 1 \end{array} \right)  ,
$$

where  $b_{i}$  contains the biases for each layer. For simplicity, we continue to use  $W_{i}$  and  $x$  in the following.

# 2.3.1 NONLINEAR MODELS ARE GERELY DISCONNECTED

One may wonder whether the same phenomena of global connectedness also holds in the half-rectified case. A simple motivating counterexample shows that this is not the case in general. Consider a simple setup with  $X \in \mathbb{R}^2$  drawn from a mixture of two Gaussians  $\mathcal{N}_{-1}$  and  $\mathcal{N}_1$ , and let  $Y = (X - \mu_Z) \cdot Z$ , where  $Z$  is the (hidden) mixture component taking  $\{1, -1\}$  values. Let  $\hat{Y} = \Phi(X; \{W_1, W_2\})$  be a single-hidden layer ReLU network, with two hidden units. Let  $\theta^A$  be a configuration that bisects the two mixture components, and let  $\theta^B$  the same configuration, but swapping the bisectrices. One can verify that they can both achieve arbitrarily small risk by letting

the covariance of the mixture components go to 0. However, any path that connects  $\theta^A$  to  $\theta^B$  must necessarily pass through a point in which  $W_{1}$  has rank 1, which leads to an estimator with risk at least  $1/2$ .

In fact, it is easy to see that this counter-example can be extended to any generic half-rectified architecture, if one is allowed to adversarially design a data distribution. For any given  $\Phi(X; \theta)$  with arbitrary architecture and current parameters  $\theta = (W_i)$ , let  $\mathcal{P}_{\theta} = \{\mathcal{A}_1, \ldots, \mathcal{A}_S\}$  be the underlying tessellation of the input space given by our current choice of parameters; that is,  $\Phi(X; \theta)$  is piece-wise linear and  $\mathcal{P}_{\theta}$  contains those pieces. Now let  $X$  be any arbitrary distribution with density  $p(x) > 0$  for all  $x \in \mathbb{R}^n$ , for example a Gaussian, and let  $Y \mid X \stackrel{d}{=} \Phi(X; \theta)$ . Since  $\Phi$  is invariant under a subgroup of permutations  $\theta_{\sigma}$  of its hidden layers, it is easy to see that one can find two parameter values  $\theta_A = \theta$  and  $\theta_B = \theta_{\sigma}$  such that  $F_o(\theta_A) = F_o(\theta_B) = 0$ , but any continuous path  $\gamma(t)$  from  $\theta_A$  to  $\theta_B$  will have a different tessellation and therefore won't satisfy  $F_o(\gamma(t)) = 0$ . Moreover, one can build on this counter-example to show that not only the level sets are disconnected, but also that there exist poor local minima. Let  $\theta'$  be a different set of parameters, and  $Y' \mid X \stackrel{d}{=} \Phi(X; \theta')$  be a different target distribution. Now consider the data distribution given by the mixture

$$
X \mid p (x), z \sim \operatorname {B e r n o u l l i} (\pi), Y \mid X, z \stackrel {{d}} {=} z \Phi (X; \theta) + (1 - z) \Phi (X; \theta^ {\prime}).
$$

By adjusting the mixture component  $\pi$  we can clearly change the risk at  $\theta$  and  $\theta'$  and make them different, but we preserve the status of local minima of  $\theta$  and  $\theta'$ .

This illustrates an intrinsic difficulty in the optimization landscape if one is after universal guarantees that do not depend upon the data distribution. This difficulty is non-existent in the linear case and not easy to exploit in mean-field approaches such as Choromanska et al. (2015), and shows that in general we should not expect to obtain connected level sets. However, connectedness can be recovered if one is willing to accept a small increase of energy and make some assumptions on the complexity of the regression task. Our main result shows that the amount by which the energy is allowed to increase is upper bounded by a quantity that trades-off model overparametrization and smoothness in the data distribution.

For that purpose, we start with a characterization of the oracle loss, and for simplicity let us assume  $Y \in \mathbb{R}$  and let us first consider the case with a single hidden layer and  $\ell_1$  regularization:  $\mathcal{R}(\theta) = \| \theta \|_1$ .

# 2.3.2 PRELIMINARIES

Before proving our main result, we need to introduce preliminary notation and results. We first describe the case with a single hidden layer of size  $m$ .

We define

$$
e (m) = \min  _ {W _ {1} \in \mathbb {R} ^ {m \times n}, \| W _ {1} (i) \| _ {2} \leq 1, W _ {2} \in \mathbb {R} ^ {m}} \mathbb {E} \left\{\left| \Phi (X; \theta) - Y \right| ^ {2} \right\} + \kappa \| W _ {2} \| _ {1}. \tag {6}
$$

to be the oracle risk using  $m$  hidden units with norm  $\leq 1$  and using sparse regression. It is a well known result by Hornik and Cybenko that a single hidden layer is a universal approximator under very mild assumptions, i.e.  $\lim_{m\to \infty}e(m) = 0$ . This result merely states that our statistical setup is consistent, and it should not be surprising to the reader familiar with classic approximation theory. A more interesting question is the rate at which  $e(m)$  decays, which depends on the smoothness of the joint density  $(X,Y)\sim P$  relative to the nonlinear activation family we have chosen.

For convenience, we redefine  $W = W_{1}$  and  $\beta = W_{2}$  and  $Z(W) = \max(0, WX)$ . We also write  $z(w) = \max(0, \langle w, X \rangle)$  where  $(X, Y) \sim P$  and  $w \in \mathbb{R}^{N}$  is any deterministic vector. Let  $\Sigma_{X} = \mathbb{E}_{P}XX^{T} \in \mathbb{R}^{N \times N}$  be the covariance operator of the random input  $X$ . We assume  $\| \Sigma_{X} \| < \infty$ .

A fundamental property that will be essential to our analysis is that, despite the fact that  $Z$  is nonlinear, the quantity  $[w_1, w_2]_Z \coloneqq \mathbb{E}_P\{z(w_1)z(w_2)\}$  is locally equivalent to the linear metric  $\langle w_1, w_2 \rangle_X = \mathbb{E}_P\{w_1^TXX^Tw_2\} = \langle w_1, \Sigma_Xw_2 \rangle$ , and that the linearization error decreases with the angle between  $w_1$  and  $w_2$ . Without loss of generality, we assume here that  $\| w_1 \| = \| w_2 \| = 1$ , and we write  $\| w \|_Z^2 = \mathbb{E}\{|z(w)|^2\}$ .

Proposition 2.3. Let  $\alpha = \cos^{-1}(\langle w_1, w_2 \rangle)$  be the angle between unitary vectors  $w_1$  and  $w_2$  and let  $w_m = \frac{w_1 + w_2}{\|w_1 + w_2\|}$  be their unitary bisector. Then

$$
\frac {1 + \cos \alpha}{2} \| w _ {m} \| _ {Z} ^ {2} - 2 \| \Sigma_ {X} \| \left(\frac {1 - \cos \alpha}{2} + \sin^ {2} \alpha\right) \leq [ w _ {1}, w _ {2} ] _ {Z} \leq \frac {1 + \cos \alpha}{2} \| w _ {m} \| _ {Z} ^ {2}. \tag {7}
$$

The term  $\| \Sigma_X\|$  is overly pessimistic: we can replace it by the energy of  $X$  projected into the subspace spanned by  $w_{1}$  and  $w_{2}$  (which is bounded by  $2\| \Sigma_X\|$ ). When  $\alpha$  is small, a Taylor expansion of the trigonometric terms reveals that

$$
\begin{array}{l} \frac {2}{3 \| \Sigma_ {X} \|} \langle w _ {1}, w _ {2} \rangle = \frac {2}{3 \| \Sigma_ {X} \|} \cos \alpha = \frac {2}{3 \| \Sigma_ {X} \|} (1 - \frac {\alpha^ {2}}{2} + O (\alpha^ {4})) \\ \leq \left(1 - \alpha^ {2} / 4\right) \| w _ {m} \| _ {Z} ^ {2} - \| \Sigma_ {X} \| \left(\alpha^ {2} / 4 + \alpha^ {2}\right) + O \left(\alpha^ {4}\right) \\ \leq \left[ w _ {1}, w _ {2} \right] _ {Z} + O \left(\alpha^ {4}\right), \\ \end{array}
$$

and similarly

$$
\left[ w _ {1}, w _ {2} \right] _ {Z} \leq \langle w _ {1}, w _ {2} \rangle \| w _ {m} \| _ {Z} ^ {2} \leq \| \Sigma_ {X} \| \langle w _ {1}, w _ {2} \rangle .
$$

The local behavior of parameters  $w_{1}, w_{2}$  on our regression problem is thus equivalent to that of having a linear layer, provided  $w_{1}$  and  $w_{2}$  are sufficiently close to each other. This result can be seen as a spoiler of what is coming: increasing the hidden layer dimensionality  $m$  will increase the chances to encounter pairs of vectors  $w_{1}, w_{2}$  with small angle; and with it some hope of approximating the previous linear behavior thanks to the small linearization error.

In order to control the connectedness, we need a last definition. Given a hidden layer of size  $m$  with current parameters  $W \in \mathbb{R}^{n \times m}$ , we define a "robust compressibility" factor as

$$
\delta_ {W} (n, \alpha ; m) = \min  _ {\| \gamma \| _ {0} \leq n, \sup  _ {i} | \angle (\tilde {w} _ {i}, w _ {i}) | \leq \alpha} \mathbb {E} \left\{| Y - \gamma Z (\tilde {W}) | ^ {2} + \kappa \| \gamma \| _ {1} \right\}, (n \leq m). \tag {8}
$$

This quantity thus measures how easily one can compress the current hidden layer representation, by keeping only a subset of  $n$  its units, but allowing these units to move by a small amount controlled by  $\alpha$ . It is a form of  $n$ -width similar to Kolmogorov width Donoho (2006) and is also related to robust sparse coding from Tang et al. (2013); Ekanadham et al. (2011).

# 2.3.3 MAIN RESULT

Our main result considers now a non-asymptotic scenario given by some fixed size  $m$  of the hidden layer. Given two parameter values  $\theta^A = (W_1^A, W_2^A) \in \mathcal{W}$  and  $\theta^B = (W_1^B, W_2^B)$  with  $F_o(\theta^{A,B}) \leq \lambda$ , we show that there exists a continuous path  $\gamma : [0,1] \to \mathcal{W}$  connecting  $\theta^A$  and  $\theta^B$  such that its oracle risk is uniformly bounded by  $\max(\lambda, \epsilon)$ , where  $\epsilon$  decreases with model overparametrization.

Theorem 2.4. For any  $\theta^A, \theta^B \in \mathcal{W}$  and  $\lambda \in \mathbb{R}$  satisfying  $F_o(\theta^{\{A,B\}}) \leq \lambda$ , there exists a continuous path  $\gamma : [0,1] \to \mathcal{W}$  such that  $\gamma(0) = \theta^A$ ,  $\gamma(1) = \theta^B$  and

$$
F _ {o} (\gamma (t)) \leq \max  (\lambda , \epsilon), \text {w i t h} \tag {9}
$$

$$
\epsilon = \inf  _ {n, \alpha} \left(\max  \left\{e (n), \delta_ {W _ {1} ^ {A}} (m, 0; m), \delta_ {W _ {1} ^ {A}} (m - n, \alpha ; m), \delta_ {W _ {1} ^ {B}} (m, 0; m), \delta_ {W _ {1} ^ {B}} (m - n, \alpha ; m) \right\} + C _ {1} \alpha + O \left(\alpha^ {2}\right)\right),
$$

where  $C_1$  is an absolute constant depending only on  $\kappa$  and  $P$ .

Some remarks are in order. First, our regularization term is currently a mix between  $\ell_2$  norm constraints on the first layer and  $\ell_1$  norm constraints on the second layer. We believe this is an artifact of our proof technique, and we conjecture that more general regularizations yield similar results. Next, this result uses the data distribution through the oracle bound  $e(m)$  and the covariance term. The extension to empirical risk is accomplished by replacing the probability measure  $P$  by the empirical measure  $\hat{P} = \frac{1}{L}\sum_{l}\delta((x,y) - (x_l,y_l))$ . However, our asymptotic analysis has to be carefully reexamined to take into account and avoid the trivial regime when  $M$  outgrows  $L$ . A consequence of Theorem 2.4 is that as  $m$  increases, the model becomes asymptotically connected, as proven in the following corollary.

Corollary 2.5. As  $m$  increases, the energy gap  $\epsilon$  goes to zero and therefore the level sets become connected at all energy levels.

This is consistent with the overparametrization results from Safran & Shamir (2015); Shamir (2016) and the general common knowledge amongst deep learning practitioners. Our next sections explore this question, and refine it by considering not only topological properties but also some rough geometrical measure of the level sets.

# 3 GEOMETRY OF LEVEL SETS

# 3.1 THE GREEDY ALGORITHM

The intuition behind our main result is that, for smooth enough loss functions and for sufficient overparameterization, it should be "easy" to connect two equally powerful models—i.e., two models with  $F_{o}\theta^{A,B} \leq \lambda$ . A sensible measure of this ease-of-connectedness is the normalized length of the geodesic connecting one model to the other:  $|\gamma_{A,B}(t)| / |\theta_A - \theta_B|$ . This length represents approximately how far of an excursion one must make in the space of models relative to the euclidean distance between a pair of models. Thus, convex models have a geodesic length of 1, because the geodesic is simply linear interpolation between models, while more non-convex models have geodesic lengths strictly larger than 1.

Because calculating the exact geodesic is difficult, we approximate the geodesic paths via a dynamic programming approach we call Dynamic String Sampling. We comment on alternative algorithms in Appendix A.

For a pair of models with network parameters  $\theta_{i},\theta_{j}$ , each with  $F_{e}(\theta)$  below a threshold  $L_0$ , we aim to efficiently generate paths in the space of weights where the empirical loss along the path remains below  $L_{0}$ . These paths are continuous curves belonging to  $\Omega_F(\lambda)$ -that is, the level sets of the loss function of interest.

Algorithm 1 Greedy Dynamic String Sampling  
1:  $L_0 \gets$  Threshold below which path will be found  
2:  $\Phi_1 \gets$  randomly initialize  $\theta_1$ , train  $\Phi(x_i \theta_1)$  to  $L_0$   
3:  $\Phi_2 \gets$  randomly initialize  $\theta_2$ , train  $\Phi(x_i \theta_2)$  to  $L_0$   
4: BeadList  $\leftarrow (\Phi_1, \Phi_2)$   
5: Depth  $\leftarrow 0$   
6: procedure FINDCONNECTION( $\Phi_1$ ,  $\Phi_2$ )  
7:  $t^* \gets t$  such that  $\frac{d\gamma(\theta_1, \theta_2, t)}{dt} = 0$  OR  $t = 0.5$   
8:  $\Phi_3 \gets$  train  $\Phi(x_i; t^*\theta_1 + (1 - t^*)\theta_2)$  to  $L_0$   
9: BeadList  $\leftarrow$  insert( $\Phi_3$ , after  $\Phi_1$ , BeadList)  
10: MaxError1  $\leftarrow$  max_t(F_e(t $\theta_3$ + (1 - t) $\theta_1$ ))  
11: MaxError2  $\leftarrow$  max_t(F_e(t $\theta_2$ + (1 - t) $\theta_3$ ))  
12: if MaxError1 > L0 then return FindConnection( $\Phi_1$ ,  $\Phi_3$ )  
13: if MaxError2 > L0 then return FindConnection( $\Phi_3$ ,  $\Phi_2$ )  
14: Depth  $\leftarrow$  Depth+1

The algorithm recursively builds a string of models in the space of weights which continuously connect  $\theta_{i}$  to  $\theta_{j}$ . Models are added and trained until the pairwise linearly interpolated loss, i.e.  $\max_{\mathrm{t}}\mathrm{F}_{\mathrm{e}}(\mathrm{t}\theta_{\mathrm{i}} + (1 - \mathrm{t})\theta_{\mathrm{j}})$  for  $t\in (0,1)$ , is below the threshold,  $L_{0}$ , for every pair of neighboring models on the string. We provide a cartoon of the algorithm in Appendix C.

# 3.2 FAILURE CONDITIONS AND PRACTICALITIES

While the algorithm presented will faithfully certify two models are connected if the algorithm converges, it is worth emphasizing that the algorithm does not guarantee that two models are disconnected if the algorithm fails to converge. In general, the problem of determining if two models are connected can be made arbitrarily difficult by choice of a particularly pathological geometry for the loss function, so we are constrained to heuristic arguments for determining when to stop running the algorithm. Thankfully, in practice, loss function geometries for problems of interest are not intractably difficult to explore. We comment more on diagnosing disconnections more carefully in Appendix E.

Further, if the MaxError exceeds  $L_{0}$  for every new recursive branch as the algorithm progresses, the worst case runtime scales as  $O(\exp(\mathbf{Depth}))$ . Empirically, we find that the number of new models added at each depth does grow, but eventually saturates, and falls for a wide variety of models and architectures, so that the typical runtime is closer to  $O(\mathrm{poly}(\mathbf{Depth}))$  at least up until a critical value of  $L_{0}$ .

To aid convergence, either of the choices in line 7 of the algorithm works in practice—choosing  $t^*$  at a local maximum can provide a modest increase in algorithm runtime, but can be unstable if the calculated interpolated loss is particularly flat or noisy.  $t^* = .5$  is more stable, but slower. Finally, we find that training  $\Phi_3$  to  $\alpha L_0$  for  $\alpha < 1$  in line 8 of the algorithm tends to aid convergence without noticeably impacting our numerics. We provide further implementation details in 4.

# 4 NUMERICAL EXPERIMENTS

For our numerical experiments, we calculated normalized geodesic lengths for a variety of regression and classification tasks. In practice, this involved training a pair of randomly initialized models to the desired test loss value/accuracy/perplexity, and then attempting to connect that pair of models via the Dynamic String Sampling algorithm. We also tabulated the average number of "beads", or the number intermediate models needed by the algorithm to connect two initial models. For all of the below experiments, the reported losses and accuracies are on a restricted test set. For more complete architecture and implementation details, see our GitHub page.

The results are broadly organized by increasing model complexity and task difficulty, from easiest to hardest. Throughout, and remarkably, we were able to easily connect models for every dataset and architecture investigated except the one explicitly constructed counterexample discussed in Appendix E.1. Qualitatively, all of the models exhibit a transition from a highly convex regime at high loss to a non-convex regime at low loss, as demonstrated by the growth of the normalized length as well as the monotonic increase in the number of required "beads" to form a low-loss connection.

# 4.1 POLYNOMIAL REGRESSION

We studied a 1-4-4-1 fully connected multilayer perceptron style architecture with sigmoid nonlinearities and RMSProp/ADAM optimization. For ease-of-analysis, we restricted the training and test data to be strictly contained in the interval  $x \in [0,1]$  and  $f(x) \in [0,1]$ . The number of required beads, and thus the runtime of the algorithm, grew approximately as a power-law, as demonstrated in Table 1 Fig. 1. We also provide a visualization of a representative connecting path between two models of equivalent power in Appendix D.

The cubic regression task exhibits an interesting feature around  $L_0 = .15$  in Table 1 Fig. 2, where the normalized length spikes, but the number of required beads remains low. Up until this point, the cubic model is strongly convex, so this first spike seems to indicate the onset of non-convex behavior and a concomitant radical change in the geometry of the loss surface for lower loss.

# 4.2 CONVOLUTIONAL NEURAL NETWORKS

To test the algorithm on larger architectures, we ran it on the MNIST hand written digit recognition task as well as the CIFAR10 image recognition task, indicated in Table 1, Figs. 3 and 4. Again, the data exhibits strong qualitative similarity with the previous models: normalized length remains low until a threshold loss value, after which it grows approximately as a power law. Interestingly, the MNIST dataset exhibits very low normalized length, even for models nearly at the state of the art in classification power, in agreement with the folk-understanding that MNIST is highly convex and/or "easy". The CIFAR10 dataset, however, exhibits large non-convexity, even at the modest test accuracy of  $80\%$ .

![](images/9e25ba3e563ea26927e4029e659e2c233a57f04ba675fb6bc2771f876f1bca01.jpg)

![](images/c46f9d02f14e27c934e7d86665f09c38fea7da8bd7ac52b3148d67368cefa701.jpg)

![](images/543312315a9112e0b07775ddf482d6cc568b766a90b567c33e7e9a91434692da.jpg)

![](images/c05a01035748e89750546b3431705805de294184493b11a3f224373cab88ec86.jpg)

![](images/920ed47cf0ad54d6acb2316f1fe03f4656bbc71eafea82ecfbab0ea17803c78d.jpg)

![](images/ab0cf24484c578ff7f3a949969f412a3240bab6482baf0edf7f9b7e45e323b3d.jpg)

![](images/dad94dfb9b9c32db58894529a73314448385bb02c730f7857a04936456801dd2.jpg)

![](images/f5a0d4d0105128d24e932ad3f962def48fed5b60228db672d04aef7c5c428d6d.jpg)

![](images/759e8758858e5b9adb9b06d250f758e87ccac1f5ea91ba06e15c7c7195a8fc35.jpg)  
Figure 1: (Column a) Average normalized geodesic length and (Column b) average number of beads versus loss. (1) A quadratic regression task. (2) A cubic regression task. (3) A convnet for MNIST. (4) A convnet inspired by Krizhevsky for CIFAR10. (5) A RNN inspired by Zaremba for PTB next word prediction.

![](images/db9f9afe2758d23eb68e4fcce126716b846c703581a5551c62ad805354abdd25.jpg)

# 4.3 RECURRENT NEURAL NETWORKS

To gauge the generalizability of our algorithm, we also applied it to an LSTM architecture for solving the next word prediction task on the PTB dataset, depicted in Table 1 Fig. 5. Notably, even for a

radically different architecture, loss function, and data set, the normalized lengths produced by the DSS algorithm recapitulate the same qualitative features seen in the above datasets—i.e., models can be easily connected at high perplexity, and the normalized length grows at lower and lower perplexity after a threshold value, indicating an onset of increased non-convexity of the loss surface.

# 5 DISCUSSION

We have addressed the problem of characterizing the loss surface of neural networks from the perspective of gradient descent algorithms. We explored two angles - topological and geometrical aspects - that build on top of each other.

On the one hand, we have presented new theoretical results that quantify the amount of uphill climbing that is required in order to progress to lower energy configurations in single layer ReLU networks, and proved that this amount converges to zero with overparametrization under mild conditions. On the other hand, we have introduced a dynamic programming algorithm that efficiently approximates geodesics within each level set, providing a tool that not only verifies the connectedness of level sets, but also estimates the geometric regularity of these sets. Thanks to this information, we can quantify how 'non-convex' an optimization problem is, and verify that the optimization of quintessential deep learning tasks – CIFAR-10 and MNIST classification using CNNs, and next word prediction using LSTMs – behaves in a nearly convex fashion up until they reach high accuracy levels.

That said, there is still a large number of limitations and open questions related to our framework. Amongst those, in the near future we shall concentrate on:

- Extending Theorem 2.4 to the multilayer case. We believe this is within reach, since the main analytic tool we use is that small changes in the parameters result in small changes in the covariance structure of the features. That remains the case in the multilayer case.  
- Empirical versus Oracle Risk. A big limitation of our theory is that right now it does not inform us on the differences between optimizing the empirical risk versus the oracle risk. Understanding the impact of generalization error and stochastic gradient in the ability to do small uphill climbs is a major open line of research.  
- Influence of symmetry groups. Our current model shows that under appropriate conditions, the presence of discrete symmetry groups does not prevent the loss from being connected, but does so at the expense of increasing the capacity. An important open question is whether one can significantly improve the asymptotic properties by relaxing connectedness to being connected up to discrete symmetry.  
- Improving numerics with Hyperplane method. Our current numerical experiments employ a greedy (albeit faster) algorithm to discover connected components and estimate geodesics. We plan to perform experiments using the less greedy algorithm described in Appendix A.

# ACKNOWLEDGMENTS

We would like to thank Mark Tygert for pointing out the reference to the  $\epsilon$ -nets and Kolmogorov capacity. We would also like to thank Maithra Raghu and Jascha Sohl-Dickstein for enlightening discussions, as well as Yasaman Bahri for helpful feedback on an early version of the manuscript. CDF was supported by the NSF Graduate Research Fellowship under Grant DGE-1106400.

# REFERENCES

Francis Bach. Convex relaxations of structured matrix factorizations. arXiv preprint arXiv:1309.3117, 2013.  
Anna Choromanska, Mikael Henaff, Michael Mathieu, Gérard Ben Arous, and Yann LeCun. The loss surfaces of multilayer networks. In Proc. AISTATS, 2015.  
Yann N Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. In Advances in Neural Information Processing Systems, pp. 2933–2941, 2014.

David L Donoho. Compressed sensing. IEEE Transactions on information theory, 52(4):1289-1306, 2006.  
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul):2121-2159, 2011.  
Chaitanya Ekanadham, Daniel Tranchina, and Eero P Simoncelli. Recovery of sparse translation-invariant signals with continuous basis pursuit. IEEE transactions on signal processing, 59(10): 4735-4744, 2011.  
Ian J Goodfellow, Oriol Vinyals, and Andrew M Saxe. Qualitatively characterizing neural network optimization problems. arXiv preprint arXiv:1412.6544, 2014.  
Geoffrey Hinton, N Srivastava, and Kevin Swersky. Lecture 6a overview of mini-batch gradient descent. Coursera Class, 2012.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Kenji Kawaguchi. Deep learning without poor local minima. arXiv preprint arXiv:1605.07110, 2016.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Jason D Lee, Max Simchowitz, Michael I Jordan, and Benjamin Recht. Gradient descent converges to minimizers. University of California, Berkeley, 1050:16, 2016.  
Itay Safran and Ohad Shamir. On the quality of the initial basin in overspecified neural networks. arXiv preprint arXiv:1511.04210, 2015.  
Levent Sagun, V Ugur Guney, Gerard Ben Arous, and Yann LeCun. Explorations on high dimensional landscapes. arXiv preprint arXiv:1412.6615, 2014.  
Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120, 2013.  
Ohad Shamir. Distribution-specific hardness of learning neural networks. arXiv:1609.01037, 2016.  
Daniel Soudry and Yair Carmon. No bad local minima: Data independent training error guarantees for multilayer neural networks. arXiv preprint arXiv:1605.08361, 2016.  
Gongguo Tang, Badri Narayan Bhaskar, Parikshit Shah, and Benjamin Recht. Compressed sensing off the grid. IEEE Transactions on Information Theory, 59(11):7465-7490, 2013.  
Roman Vershynin. Introduction to the non-asymptotic analysis of random matrices. arXiv preprint arXiv:1011.3027, 2010.
