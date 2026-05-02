# A VARIATIONAL INEQUALITY PERSPECTIVE ON GENERATIVE ADVERSARIAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Generative adversarial networks (GANs) form a generative modeling approach known for producing appealing samples, but they are notably difficult to train. One common way to tackle this issue has been to propose new formulations of the GAN objective. Yet, surprisingly few studies have looked at optimization methods designed for this adversarial training. In this work, we cast GAN optimization problems in the general variational inequality framework. Tapping into the mathematical programming literature, we counter some common misconceptions about the difficulties of saddle point optimization and propose to extend techniques designed for variational inequalities to the training of GANs. We apply averaging, extrapolation and a novel computationally cheaper variant that we call extrapolation from the past to the stochastic gradient method (SGD) and Adam.

# 1 INTRODUCTION

Generative adversarial networks (GANs) (Goodfellow et al., 2014) form a generative modeling approach known for producing realistic natural images (Karras et al., 2018) as well as high quality super-resolution (Ledig et al., 2017) and style transfer (Zhu et al., 2017). Nevertheless, GANs are also known to be difficult to train, often displaying an unstable behavior (Goodfellow, 2016). Much recent work has tried to tackle these training difficulties, usually by proposing new formulations of the GAN objective (Nowozin et al., 2016; Arjovsky et al., 2017). Each of these formulations can be understood as a two-player game, in the sense of game theory (Von Neumann and Morgenstern, 1944), and can be addressed as a variational inequality problem (VIP) (Harker and Pang, 1990), a framework that encompasses traditional saddle point optimization algorithms (Korpelevich, 1976).

Solving such GAN games is traditionally approached by running variants of stochastic gradient descent (SGD) initially developed for optimizing supervised neural network objectives. Yet it is known that for some games (Goodfellow, 2016, §8.2) SGD exhibits oscillatory behavior and fails to converge. This oscillatory behavior, which does not arise from stochasticity, highlights a fundamental problem: while a direct application of basic gradient descent is an appropriate method for regular minimization problems, it is not a sound optimization algorithm for the kind of two-player games of GANs. This constitutes a fundamental issue for GAN training, and calls for the use of more principled methods with more reassuring convergence guarantees.

Contributions. We point out that multi-player games can be cast as variational inequality problems and consequently the same applies to any GAN formulation posed as a minimax or non-zero-sum game. We present two techniques from this literature, namely averaging and extrapolation, widely used to solve variational inequality problems (VIP) but which have not been explored in the context of GANs before.<sup>1</sup>

We extend standard GAN training methods such as SGD or Adam into variants that incorporate these techniques (Alg. 3, 4 are new). We also explain that the oscillations of basic SGD for GAN training previously noticed (Goodfellow, 2016) can be explained by standard variational inequality optimization results and we illustrate how averaging and extrapolation can fix this issue.

We introduce a new technique, called extrapolation from the past, that only requires one gradient computation per iteration compared to extrapolation which requires to compute the gradient twice. We prove its convergence in the stochastic variational inequality setting, i.e. when applied to SGD.

Finally, we test these techniques in the context of standard GAN training. We observe a  $4\% - 6\%$  improvement on the inception score (Salimans et al., 2016) of WGAN (Arjovsky et al., 2017) and WGAN-GP (Gulrajani et al., 2017) on the CIFAR-10 dataset.

Outline. §2 presents the background on GAN and optimization, and shows how to cast this optimization as a VIP. §3 presents standard techniques to optimize variational inequalities in a batch setting as well as our new one, extrapolation from the past. §4 considers these methods in the stochastic setting, yielding three corresponding variants of SGD, and provide their respective convergence rates. §6 discusses the related work and §7 presents experimental results.

# 2 GAN OPTIMIZATION AS A VARIATIONAL INEQUALITY PROBLEM

# 2.1 GAN FORMULATIONS

The purpose of generative modeling is to generate samples from a distribution  $q_{\theta}$  that matches best the true distribution  $p$  of the data. The generative adversarial network training strategy can be understood as a game between two players called generator and discriminator. The former produces a sample that the latter has to classify between real or fake data. The final goal is to build a generator able to produce sufficiently realistic samples to fool the discriminator.

In the original GAN paper (Goodfellow et al., 2014), the GAN objective is formulated as a zero-sum game where the cost function of the discriminator  $D_{\varphi}$  is given by the negative log-likelihood of the binary classification task between real or fake data generated from  $q_{\theta}$  by the generator,

$$
\min  _ {\boldsymbol {\theta}} \max  _ {\boldsymbol {\varphi}} \mathcal {L} (\boldsymbol {\theta}, \boldsymbol {\varphi}) \stackrel {\text {d e f}} {=} - \underset {\mathbf {x} \sim p} {\mathbb {E}} [ \log D _ {\boldsymbol {\varphi}} (\mathbf {x}) ] - \underset {\mathbf {x} ^ {\prime} \sim q _ {\boldsymbol {\theta}}} {\mathbb {E}} [ \log (1 - D _ {\boldsymbol {\varphi}} (\mathbf {x} ^ {\prime})) ]. \tag {1}
$$

However Goodfellow et al. (2014) recommends to use in practice a second formulation, called non-saturating GAN. This formulation is a non-zero-sum game where the aim is to jointly minimize:

$$
\mathcal {L} ^ {(\boldsymbol {\theta})} (\boldsymbol {\theta}, \boldsymbol {\varphi}) \stackrel {\text {d e f}} {=} - \underset {\mathbf {x} ^ {\prime} \sim q _ {\boldsymbol {\theta}}} {\mathbb {E}} \log D _ {\boldsymbol {\varphi}} (\mathbf {x} ^ {\prime}) \text {a n d} \mathcal {L} ^ {(\boldsymbol {\varphi})} (\boldsymbol {\theta}, \boldsymbol {\varphi}) \stackrel {\text {d e f}} {=} - \underset {\mathbf {x} \sim p} {\mathbb {E}} \log D _ {\boldsymbol {\varphi}} (\mathbf {x}) - \underset {\mathbf {x} ^ {\prime} \sim q _ {\boldsymbol {\theta}}} {\mathbb {E}} \log (1 - D _ {\boldsymbol {\varphi}} (\mathbf {x} ^ {\prime})). \tag {2}
$$

The dynamics of this formulation have the same stationary points as the zero-sum one (1) but are claimed to provide "much stronger gradients early in learning" (Goodfellow et al., 2014).

# 2.2 EQUILIBRIUM

The minimax formulation (1) is theoretically convenient because a large literature on games studies this problem and provides guarantees on the existence of equilibria. Nevertheless, practical considerations lead the GAN literature to consider a different objective for each player as formulated in (2). In that case, the two-player game problem (Von Neumann and Morgenstern, 1944) consists in finding the following Nash equilibrium:

$$
\boldsymbol {\theta} ^ {*} \in \underset {\boldsymbol {\theta} \in \Theta} {\arg \min } \mathcal {L} ^ {(\boldsymbol {\theta})} (\boldsymbol {\theta}, \boldsymbol {\varphi} ^ {*}) \quad \text {a n d} \quad \boldsymbol {\varphi} ^ {*} \in \underset {\boldsymbol {\varphi} \in \Phi} {\arg \min } \mathcal {L} ^ {(\boldsymbol {\varphi})} (\boldsymbol {\theta} ^ {*}, \boldsymbol {\varphi}). \tag {3}
$$

Only when  $\mathcal{L}^{(\theta)} = -\mathcal{L}^{(\varphi)}$  is the game called a zero-sum game and (3) can be formulated as a minimax problem. One important point to notice is that the two optimization problems in (3) are coupled and have to be considered jointly from an optimization point of view.

Standard GAN objectives are non-convex (i.e. each cost function is non-convex), and thus such (pure) equilibria may not exist. As far as we know, not much is known about the existence of these equilibria for non-convex losses (see Heusel et al. (2017) and references therein for some results). In our theoretical analysis in §4, our assumptions (monotonicity (24) of the operator and convexity of the constraints set) imply the existence of an equilibrium.

In this paper, we focus on ways to optimize these games, assuming that an equilibrium exists. As is often standard in non-convex optimization, we also focus on finding points satisfying the necessary

stationary conditions. As we mentioned previously, one difficulty that emerges in the optimization of such games is that the two different cost functions of (3) have to be minimized jointly in  $\theta$  and  $\varphi$ . Fortunately, the optimization literature has for a long time studied so-called variational inequality problems, which generalize the stationary conditions for two-player game problems.

# 2.3 VARIATIONAL INEQUALITY PROBLEM FORMULATION

We first consider the local necessary conditions that characterize the solution of the smooth two-player game (3), defining stationary points, which will motivate the definition of a variational inequality. In the unconstrained setting, a stationary point is a couple  $(\theta^{*},\varphi^{*})$  with zero gradient:

$$
\left\| \nabla_ {\boldsymbol {\theta}} \mathcal {L} ^ {(\boldsymbol {\theta})} \left(\boldsymbol {\theta} ^ {*}, \boldsymbol {\varphi} ^ {*}\right) \right\| = \left\| \nabla_ {\boldsymbol {\varphi}} \mathcal {L} ^ {(\boldsymbol {\varphi})} \left(\boldsymbol {\theta} ^ {*}, \boldsymbol {\varphi} ^ {*}\right) \right\| = 0. \tag {4}
$$

When constraints are present, a stationary point  $(\theta^{*},\varphi^{*})$  is such that the directional derivative of each cost function is non-negative in any feasible direction (i.e. there is no feasible descent direction):

$$
\nabla_ {\boldsymbol {\theta}} \mathcal {L} ^ {(\boldsymbol {\theta})} \left(\boldsymbol {\theta} ^ {*}, \boldsymbol {\varphi} ^ {*}\right) ^ {\top} (\boldsymbol {\theta} - \boldsymbol {\theta} ^ {*}) \geq 0 \quad \text {a n d} \quad \nabla_ {\boldsymbol {\varphi}} \mathcal {L} ^ {(\boldsymbol {\varphi})} \left(\boldsymbol {\theta} ^ {*}, \boldsymbol {\varphi} ^ {*}\right) ^ {\top} (\boldsymbol {\varphi} - \boldsymbol {\varphi} ^ {*}) \geq 0, \forall (\boldsymbol {\theta}, \boldsymbol {\varphi}) \in \Theta \times \Phi . (5)
$$

Defining  $\omega \stackrel{\mathrm{def}}{=} (\theta, \varphi)$ ,  $\omega^{*} \stackrel{\mathrm{def}}{=} (\theta^{*}, \varphi^{*})$ ,  $\Omega \stackrel{\mathrm{def}}{=} \Theta \times \Phi$ , Eq. (5) can be compactly formulated as:

$$
F \left(\boldsymbol {\omega} ^ {*}\right) ^ {\top} \left(\boldsymbol {\omega} - \boldsymbol {\omega} ^ {*}\right) \geq 0, \forall \boldsymbol {\omega} \in \Omega \quad \text {w h e r e} \quad F (\boldsymbol {\omega}) \stackrel {\text {d e f}} {=} \left[ \begin{array}{l l} \nabla_ {\boldsymbol {\theta}} \mathcal {L} ^ {(\boldsymbol {\theta})} (\boldsymbol {\theta}, \boldsymbol {\varphi}) & \nabla_ {\boldsymbol {\varphi}} \mathcal {L} ^ {(\boldsymbol {\varphi})} (\boldsymbol {\theta}, \boldsymbol {\varphi}) \end{array} \right] ^ {\top}. \tag {6}
$$

These stationary conditions can be generalized to any continuous vector field: let  $\Omega \subset \mathbb{R}^d$  and  $F:\Omega \to \mathbb{R}^d$  be a continuous mapping. The variational inequality problem (Harker and Pang, 1990) (depending on  $F$  and  $\Omega$ ) is:

$$
\text {f i n d} \omega^ {*} \in \Omega \quad \text {s u c h t h a t} \quad F \left(\omega^ {*}\right) ^ {\top} \left(\omega - \omega^ {*}\right) \geq 0, \forall \omega \in \Omega . \tag {VIP}
$$

We call optimal set the set  $\Omega^{*}$  of  $\omega \in \Omega$  verifying (VIP). The intuition behind it is that any  $\omega^{*} \in \Omega^{*}$  is a fixed point of the constrained dynamic of  $F$  (constrained to  $\Omega$ ).

We have thus showed that both saddle point optimization and non-zero sum game optimization, which encompasses the large majority of GAN variants proposed in the literature, can be cast as Variational Inequality Problems. In the following section, we turn to suitable optimization techniques for such problems.

# 3 OPTIMIZATION OF VARIATIONAL INEQUALITIES (BATCH SETTING)

Let us begin by looking at techniques that were developed in the optimization literature to solve (VIP). We present the intuitions behind them as well as their performance on a simple bilinear problem (see Fig. 1). Our goal here is to provide mathematical insights into the techniques of averaging (§3.1) and extrapolation (§3.2), to inspire their application to extending other optimization algorithms. We then propose a novel variant of the extrapolation technique in §3.3 extrapolation from the past. We here treat the batch setting, i.e. considering that the operator  $F(\omega)$  as defined in Eq. 6 yields an exact full gradient. We will present extensions of these techniques to the stochastic setting later in §4.

The two standard methods studied in the VIP literature are the gradient method (Bruck, 1977) and the extragradients method (Korpelevich, 1976). The iterates of the basic gradient method are given by  $\omega_{t + 1} = P_{\Omega}[\omega_t - \eta F(\omega_t)]$  where  $P_{\Omega}[\cdot ]$  is the projection onto the constraints set (if constraints are present) associated to (VIP). These iterates are known to converge linearly under an additional assumption on the operator3 (Chen and Rockafellar, 1997), but oscillate for a bilinear operator as shown in Fig. 1. On the other hand, the uniform average of these iterates converge for any bounded monotone operator with a  $O(1 / \sqrt{t})$  rate (Nedic and Ozdaglar, 2009), motivating the presentation of averaging in §3.1. By contrast, the extragradients method (extrapolated gradient) does not require any averaging to converge for monotone operators (in the batch setting), and can even converge at the faster  $O(1 / t)$  rate (Nesterov, 2007). The idea of this method is to compute a lookahead step (see intuition on extrapolation in §3.2) in order to compute a more stable direction to follow.

# 3.1 AVERAGING

More generally, we consider a weighted averaging scheme with weights  $\rho_t \geq 0$ . This weighted averaging scheme has been proposed for the first time for (batch) VIP by Bruck (1977),

$$
\bar {\omega} _ {T} \stackrel {\text {d e f}} {=} \frac {\sum_ {t = 0} ^ {T - 1} \rho_ {t} \omega_ {t}}{S _ {T}}, \quad S _ {T} \stackrel {\text {d e f}} {=} \sum_ {t = 0} ^ {T - 1} \rho_ {t}. \tag {7}
$$

Averaging schemes can be efficiently implemented in an online fashion noticing that,

$$
\bar {\omega} _ {t} = \left(1 - \tilde {\rho} _ {t}\right) \bar {\omega} _ {t - 1} + \tilde {\rho} _ {t} \omega_ {t} \quad \text {w h e r e} \quad 0 \leq \tilde {\rho} _ {t} \leq 1. \tag {8}
$$

For instance, setting  $\tilde{\rho}_t = \frac{1}{t}$  provides uniform averaging ( $\rho_t = 1$ ) and  $\tilde{\rho}_t = 1 - \beta < 1$  provides geometric averaging also known as exponential moving averaging ( $\rho_t = \beta^t$ ). Averaging is experimentally compared with the other techniques presented in this section in Fig. 1.

In order to illustrate how averaging tackle the oscillatory behavior in game optimization, we consider a toy example where the discriminator and the generator are linear:  $D_{\varphi}(\mathbf{x}) = \varphi^T\mathbf{x}$  and  $G_{\theta}(\mathbf{z}) = \theta \mathbf{z}$  (implicitly defining  $q_{\theta}$ ). By replacing these expressions in the WGAN objective, we get the following bilinear objective:

$$
\min  _ {\boldsymbol {\theta} \in \Theta} \max  _ {\boldsymbol {\varphi} \in \Phi , | | \boldsymbol {\varphi} | | \leq 1} \boldsymbol {\varphi} ^ {T} \mathbb {E} [ \mathbf {x} ] - \boldsymbol {\varphi} ^ {T} \boldsymbol {\theta} \mathbb {E} [ \mathbf {z} ]. \tag {9}
$$

A similar task was presented by Nagarajan and Kolter (2017) where they consider a quadratic discriminator instead of a linear one, and show that gradient descent is not necessarily asymptotically stable. The bilinear objective has been extensively used (Goodfellow, 2016; Mescheder et al., 2018; Yadav et al., 2018) to highlight the difficulties of gradient descent for saddle point optimization. Yet, ways to cope with this issue have been proposed decades ago in the context of mathematical programming. Simplifying further by setting the dimension to 1 and centering the equilibrium to the origin, Eq. 9 becomes:

$$
\min  _ {\theta \in \mathbb {R}} \max  _ {\phi \in \mathbb {R}} \theta \cdot \phi \quad \text {a n d} \quad \left(\theta^ {*}, \phi^ {*}\right) = (0, 0). \tag {10}
$$

The operator associated with this minimax game is  $F(\theta, \phi) = (\phi, -\theta)$ . There are several ways to compute the discrete updates of this dynamics. The two most common ones are the simultaneous and the alternated gradient update rules,

$$
\text {S i m u l t a n e o u s} \quad \text {u p d a t e :} \left\{ \begin{array}{l} \theta_ {t + 1} = \theta_ {t} - \eta \phi_ {t} \\ \phi_ {t + 1} = \phi_ {t} + \eta \theta_ {t} \end{array} , \right. \quad \text {A l t e r n a t e d} \quad \text {u p d a t e :} \left\{ \begin{array}{l} \theta_ {t + 1} = \theta_ {t} - \eta \phi_ {t} \\ \phi_ {t + 1} = \phi_ {t} + \eta \theta_ {t + 1} \end{array} . \right. \tag {11}
$$

Interestingly, these two choices give rise to have a completely different behavior. The norm of the simultaneous updates diverges geometrically whereas the alternated iterates are bounded but do not converge to the equilibrium. As a consequence, their respective uniform average have a different behavior, as highlighted in the following proposition (more details and proof in §B.1):

Proposition 1. The simultaneous iterates diverge geometrically and the alternated iterates defined in (11) are bounded but do not converge to  $0$  as

$$
\begin{array}{l} \text {S i m u l t a n e o u s :} \theta_ {t + 1} ^ {2} + \phi_ {t + 1} ^ {2} = (1 + \eta^ {2}) (\theta_ {t} ^ {2} + \phi_ {t} ^ {2}), \quad \text {A l t e r n a t e d :} \theta_ {t} ^ {2} + \phi_ {t} ^ {2} = \Theta (\theta_ {0} ^ {2} + \phi_ {0} ^ {2}) \\ \text {w h e r e} u _ {t} = \Theta (v _ {t}) \Leftrightarrow \exists \alpha , \beta > 0: \alpha v _ {t} \leq u _ {t} \leq \beta v _ {t}. \end{array} \tag {12}
$$

The uniform average  $(\bar{\theta}_t,\bar{\phi}_t)\stackrel {def}{=}\frac{1}{t}\sum_{s = 0}^{t - 1}(\theta_s,\phi_s)$  of the simultaneous updates (resp. the alternated updates) diverges (resp. converges to 0) as,

$$
\text {S i m u l t a n e o u s :} \bar {\theta} _ {t} ^ {2} + \bar {\phi} _ {t} ^ {2} = \Theta \left(\frac {\theta_ {0} ^ {2} + \phi_ {0} ^ {2}}{\eta^ {2} t ^ {2}} (1 + \eta^ {2}) ^ {t}\right), \text {A l t e r n a t e d :} \bar {\theta} _ {t} ^ {2} + \bar {\phi} _ {t} ^ {2} = \Theta \left(\frac {\theta_ {0} ^ {2} + \phi_ {0} ^ {2}}{\eta^ {2} t ^ {2}}\right). \tag {13}
$$

This sublinear convergence result, proved in §B, underlines the benefits of averaging when the sequence of iterates is bounded (i.e. for alternated update rule). When the sequence of iterates is not bounded (i.e. for simultaneous updates) averaging fails to ensure convergence. This theorem also shows how alternated updates may have better convergence properties than simultaneous updates.

# 3.2 EXTRAPOLATION

Another technique used in the variational inequality literature to prevent oscillations is extrapolation. This concept is anterior to the extragradient method since Korpelevich (1976) mentions that the idea of extrapolated "prices" to give "stability" had been already formulated by Polyak (1963, Chap. II). The idea behind this technique is to compute the gradient at an (extrapolated) point different from the current point from which the update is performed, stabilizing the dynamics:

$$
\text {C o m p u t e} \omega_ {t + 1 / 2} = P _ {\Omega} \left[ \omega_ {t} - \eta F \left(\omega_ {t}\right) \right], \tag {14}
$$

$$
\text {P e r f o r m} \quad \omega_ {t + 1} = P _ {\Omega} \left[ \omega_ {t} - \eta F \left(\omega_ {t + 1 / 2}\right) \right]. \tag {15}
$$

Note that, even in the unconstrained case, this method is intrinsically different from Nesterov's momentum<sup>5</sup> (Nesterov, 1983, Eq. 2.2.9) because of this lookahead step for the gradient computation:

$$
\text {N e s t e r o v ’ s m e t h o d :} \quad \omega_ {t + 1 / 2} = \omega_ {t} - \eta F (\omega_ {t}), \quad \omega_ {t + 1} = \omega_ {t + 1 / 2} + \beta \left(\omega_ {t + 1 / 2} - \omega_ {t}\right). \tag {16}
$$

Nesterov's method does not converge when trying to optimize (10). One intuition explaining why extrapolation provides better convergence properties than the standard gradient method comes from Euler's method framework (see for instance (Atkinson, 2003) for more details on that topic). Actually, if we consider a first order approximation of  $\omega_{t + 1 / 2}$ , we have  $\omega_{t + 1 / 2} \approx \omega_{t + 1} + o(\eta)$  and consequently, the update step (15) is close to an implicit method step:

$$
\text {I m p l i c i t s t e p :} \quad \omega_ {t + 1} = \omega_ {t} - \eta F \left(\omega_ {t + 1}\right). \tag {17}
$$

In the literature on Euler's method, implicit methods are known to be more stable and to benefit from better convergence properties (Atkinson, 2003) than explicit methods. They are not often used in practice though since they require to solve a potentially non-linear system at each iteration.

Taking back the simplified WGAN toy example (10) from §3.1 we get the following update rules,

$$
\text {I m p l i c i t :} \left\{ \begin{array}{l} \theta_ {t + 1} = \theta_ {t} - \eta \phi_ {t + 1} \\ \phi_ {t + 1} = \phi_ {t} + \eta \theta_ {t + 1} \end{array} , \right. \quad \text {E x t r a p o l a t i o n :} \left\{ \begin{array}{l} \theta_ {t + 1} = \theta_ {t} - \eta \left(\phi_ {t} + \eta \theta_ {t}\right) \\ \phi_ {t + 1} = \phi_ {t} + \eta \left(\theta_ {t} - \eta \phi_ {t}\right) \end{array} . \right. \tag {18}
$$

In the following proposition, we will see that the respective convergence rates of the implicit method and extrapolation are highly similar. Keeping in mind that the latter has the major advantage of being more practical, this proposition clearly underlines the benefits of extrapolation (more details and proof in §B.2),

Proposition 2. The squared norm of the iterates  $N_{t}\stackrel {\text{def}}{=}\theta_{t}^{2} + \phi_{t}^{2}$ , where the update rule of  $\theta_t$  and  $\phi_t$  are defined in (18), decreases geometrically for any  $\eta < 1$  as,

$$
\text {I m p l i c t}: N _ {t + 1} = \left(1 - \eta^ {2} + \eta^ {4} + \mathcal {O} \left(\eta^ {6}\right)\right) N _ {t}, \quad \text {E x t r a p o l a t i o n}: N _ {t + 1} = \left(1 - \eta^ {2} + \eta^ {4}\right) N _ {t}. \tag {19}
$$

# 3.3 EXTRAPOLATION FROM THE PAST

One issue with extrapolation is that the algorithm "wastes" a gradient (14). Indeed we need to compute the gradient at two different positions for every single update of the parameters. We thus propose a new technique that we call extrapolation from the past which only requires to compute one gradient for every update. The idea of this technique is to store and re-use the previous extrapolated gradient to compute the new extrapolation point:

$$
\text {E x t r a p o l a t i o n} \omega_ {t + 1 / 2} = P _ {\Omega} \left[ \omega_ {t} - \eta F \left(\omega_ {t - 1 / 2}\right) \right] \tag {20}
$$

$$
\text {P e r f o r m} \quad \omega_ {t + 1} = P _ {\Omega} \left[ \omega_ {t} - \eta F \left(\omega_ {t + 1 / 2}\right) \right] \text {a n d s t o r e :} F \left(\omega_ {t + 1 / 2}\right) \tag {21}
$$

This update scheme can be related to the optimistic mirror descent (Rakhlin and Sridharan, 2013; Daskalakis et al., 2018) in the unconstrained case, (20) and (21) reduce to:

$$
\text {O p t i m i s t i c} \quad \omega_ {t + 1 / 2} = \omega_ {t - 1 / 2} - 2 \eta F \left(\omega_ {t - 1 / 2}\right) + \eta F \left(\omega_ {t - 3 / 2}\right) \tag {22}
$$

However our technique comes from a different perspective, it was motivated by VIP and inspired from the extragradient method. Furthermore our technique extends to constrained optimization as show in (20) and (21). It is not clear whether or not a single projection added to (22) provides a provably converging algorithm. Using the VIP point of view we are able to prove a linear convergence rate for a projected version of the extrapolation from the past (see details and proof of Theorem 1 in §B.3). We also extend these results to the stochastic operator setting in §4.

![](images/69ea7ddbd744f4ddbdd4db59c8c9748e71947864a1fc255a0db88e5b2e44c2f8.jpg)  
Figure 1: Comparison of the basic gradient method (as well as Adam) with the techniques presented in §3 on the optimization of (9). Only the algorithms advocated in this paper (Averaging, Extrapolation and Extrapolation from the past) converge quickly to the solution. Each marker represents 20 iterations. We compare these algorithms on a non-convex objective in §G.1.

![](images/87a2e80f2cc4dae1fb0c62c2be64898208bfc2d6b64a0852737dd98bb6898d60.jpg)  
Figure 2: Three variants of SGD using the techniques introduced in §3.

<table><tr><td>Algorithm 1 AvgSGD</td><td>Algorithm 2 AvgExtraSGD</td><td>Algorithm 3 AvgPastExtraSGD</td></tr><tr><td>Let ω0 ∈ Ω</td><td>for t = 0...T-1 do</td><td>Let ω0 ∈ Ω</td></tr><tr><td>for t = 0...T-1 do</td><td>dt← F(ωt,ξt)</td><td>for t = 0...T-1 do</td></tr><tr><td>dt← F(ωt,ξt)</td><td>ω′t← PΩ[ωt-ηt]dt</td><td>ω′t← PΩ[ωt-ηt]dt-1</td></tr><tr><td>ωt+1← PΩ[ωt-ηt]dt</td><td>dt← F(ω′t,ξt)</td><td>dt← F(ω′t,ξt)</td></tr><tr><td>end for</td><td>ωt+1← PΩ[ωt-ηt]dt</td><td>ωt+1← PΩ[ωt-ηt]dt</td></tr><tr><td rowspan="2">Return ωT← Σt=0T-1ηtωt/Σt=0T-1ηt</td><td>end for</td><td>end for</td></tr><tr><td>Return ωT← Σt=0T-1ηtω′t/Σt=0T-1ηt</td><td>Return ωT← Σt=0T-1ηtω′t/Σt=0T-1ηt</td></tr></table>

Theorem 1 (Linear convergence of extrapolation from the past). If  $F$  is  $\mu$ -strongly monotone (see §A for the definition of strong monotonicity) and  $L$ -Lipschitz, then the updates (20) and (21) with  $\eta = \frac{1}{4L}$  provide linearly converging iterates,

$$
\left\| \boldsymbol {\omega} _ {t} - \boldsymbol {\omega} ^ {*} \right\| _ {2} ^ {2} \leq \left(1 - \frac {\mu}{4 L}\right) ^ {t} \left\| \boldsymbol {\omega} _ {0} - \boldsymbol {\omega} ^ {*} \right\| _ {2} ^ {2}, \quad \forall t \geq 0. \tag {23}
$$

In comparison to the results from (Daskalakis et al., 2018) that hold only for a bilinear objective, we provide a faster convergence rate (linear vs sublinear) on the last iterate for a general (strongly monotone) operator  $F$  and any projection on a convex  $\Omega$ . One thing to notice is that the operator of a bilinear objective is not strongly monotone, but in that case one can use the standard extrapolation method (14) which converge linearly for a (constrained or not) bilinear game (Tseng, 1995, Cor. 3.3).

# 4 OPTIMIZATION OF VIP WITH STOCHASTIC GRADIENTS

In this section, we consider extensions of the techniques presented in section §3 for optimizing (VIP), to the context of a stochastic operator. In this case, at each time step we no longer have access to the exact gradient  $F(\omega)$  but to an unbiased stochastic estimate of it  $F(\omega, \xi)$  where  $\xi \sim P$  and  $F(\omega) \coloneqq \mathbb{E}_{\xi \sim P}[F(\omega, \xi)]$ . This is motivated from the GAN formulation where we only have access to a finite sample estimate of the expected gradient, computed on a mini-batch. For GANs,  $\xi$  is thus a mini-batch of points coming from the true data distribution  $p$  and the generator distribution  $q_{\theta}$ .

For our analysis, we require at least one of the two following assumptions on the stochastic operator:

Assumption 1. Bounded variance by  $\sigma^2$ :  $\mathbb{E}_{\xi}[\| F(\omega) - F(\omega, \xi)\|^2] \leq \sigma^2$ ,  $\forall \omega \in \Omega$ .

Assumption 2. Bounded expected squared norm by  $M^2$ :  $\mathbb{E}_{\xi}[\| F(\omega, \xi)\|^2] \leq M^2$ ,  $\forall \omega \in \Omega$ .

Assump. 1 is standard in stochastic variational analysis, while Assump. 2 is a stronger assumption sometimes made in stochastic convex optimization. To illustrate how strong Assump. 2 is, note that it does not hold for an unconstrained bilinear objective like in our example 10 in §3. It is thus mainly reasonable for bounded constraint sets. Note that in practice we have  $\sigma \ll M$ .

We now present and analyse three algorithms, variants of SGD that are appropriate to solve (VIP). They are given in Fig. 2. The first one Alg. 1 (AvgSGD) is the stochastic extension of the gradient method for solving (VIP); Alg. 2 (AvgExtraSGD) uses extrapolation and Alg. 3 (AvgPastExtraSGD) uses extrapolation from the past. A fourth variant (Alg.5) is proposed in §D. These three algorithms return an average of the iterates. The proofs of the theorems presented in this section are in §F.

To handle constraints such as parameter clipping (Arjovsky et al., 2017), we present a projected version of theses algorithms, where  $P_{\Omega}[\omega']$  denotes the projection of  $\omega'$  onto  $\Omega$  (see §A). Note that when  $\Omega = \mathbb{R}^d$ , the projection is the identity mapping (unconstrained setting). In order to prove the convergence of these three algorithms we will assume that  $F$  is monotone:

$$
\left(F (\boldsymbol {\omega}) - F \left(\boldsymbol {\omega} ^ {\prime}\right)\right) ^ {\top} \left(\boldsymbol {\omega} - \boldsymbol {\omega} ^ {\prime}\right) \geq 0 \quad \forall \boldsymbol {\omega}, \boldsymbol {\omega} ^ {\prime} \in \Omega . \tag {24}
$$

If  $F$  can be written as (6), it implies that the cost functions are convex.

Assumption 3.  $F$  is monotone and  $\Omega$  is a compact convex set, such that  $\max_{\omega,\omega'\in\Omega}\|\omega-\omega'\|^2 \leq R^2$ . In that setting the quantity  $g(\omega) := \max_{\omega\in\Omega} F(\omega^*)^\top(\bar{\omega}^* - \omega)$  is well defined and is equal to 0 if and only if  $\omega^*$  is a solution of (VIP). Moreover, if we are optimizing a zero-sum game, we have  $\omega = (\theta, \varphi)$ ,  $\Omega = \Theta \times \Phi$  and  $F(\theta, \varphi) = [\nabla_\theta \mathcal{L}(\theta, \varphi) - \nabla_\varphi \mathcal{L}(\theta, \varphi)]^\top$ . Hence, the quantity  $h(\theta, \varphi) := \max_{\theta\in\Theta} \mathcal{L}(\theta, \varphi^*) - \min_{\varphi\in\Phi} \mathcal{L}(\theta^*, \varphi)$  is well defined and equal to 0 if and only if  $(\theta^*, \varphi^*)$  is a Nash equilibrium of the game. The two functions  $g$  and  $h$  are called merit functions (more details on the concept of merit functions in §C). In the following, we call,

$$
\operatorname {E r r} (\boldsymbol {\omega}) \stackrel {\text {d e f}} {=} \left\{ \begin{array}{l l} \max  _ {\boldsymbol {\theta}, \boldsymbol {\varphi} \in \Theta \times \Phi} \mathcal {L} \left(\boldsymbol {\theta}, \boldsymbol {\varphi} ^ {*}\right) - \mathcal {L} \left(\boldsymbol {\theta} ^ {*}, \boldsymbol {\varphi}\right) & \text {i f} F (\boldsymbol {\theta}, \boldsymbol {\varphi}) = \left[ \nabla_ {\boldsymbol {\theta}} \mathcal {L} (\boldsymbol {\theta}, \boldsymbol {\varphi}) - \nabla_ {\boldsymbol {\varphi}} \mathcal {L} (\boldsymbol {\theta}, \boldsymbol {\varphi}) \right] ^ {\top} \\ \max  _ {\boldsymbol {\omega} \in \Omega} F \left(\boldsymbol {\omega} ^ {*}\right) ^ {\top} \left(\bar {\boldsymbol {\omega}} ^ {*} - \boldsymbol {\omega}\right) & \text {o t h e r w i s e .} \end{array} \right. \tag {25}
$$

Averaging. Alg. 1 (AvgSGD) presents the stochastic gradient method with averaging, which reduces to the standard (simultaneous) SGD updates for the two-player games used in the GAN literature, but returning an average of the iterates.

Theorem 2. Under Assump. 1, 2 and 3, SGD with averaging (Alg. 1) with a constant step-size gives,

$$
\mathbb {E} \left[ \operatorname {E r r} \left(\bar {\omega} _ {T}\right) \right] \leq \frac {R ^ {2}}{2 \eta T} + \eta \frac {M ^ {2} + \sigma^ {2}}{2} \quad \text {w h e r e} \quad \bar {\omega} _ {T} \stackrel {\text {d e f}} {=} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \omega_ {t}, \quad \forall T \geq 1. \tag {26}
$$

Thm. 2 uses a similar proof as (Nemirovski et al., 2009). The constant term  $\eta(M^2 + \sigma^2)/2$  in (26) is called the variance term. This type of bound is standard in stochastic optimization. We also provide in §F a similar rate with an extra log factor when  $\eta_t = \frac{\eta}{\sqrt{t}}$ . We show that this variance term is smaller than the one of SGD with prediction method (Yadav et al., 2018) in §E.

Extrapolations. Alg. 2 (AvgExtraSGD) adds an extrapolation step compared to Alg. 1 in order to reduce the oscillations due to the game between the two players. A theoretical consequence is that it has a smaller variance term than (26). As discussed previously, Assump. 2 made in Thm. 2 for the convergence of Alg. 1 is very strong in the unbounded setting. One advantage of SGD with extrapolation is that Thm. 3 does not require this assumption.

Theorem 3. (Juditsky et al., 2011, Thm. 1) Under Assump. 1 and 3, if  $\mathbb{E}_{\xi}[F]$  is  $L$ -Lipschitz, then SGD with extrapolation and averaging (Alg. 2) using a constant step-size  $\eta \leq \frac{1}{\sqrt{3}L}$  gives,

$$
\mathbb {E} \left[ \operatorname {E r r} \left(\bar {\omega} _ {T}\right) \right] \leq \frac {R ^ {2}}{\eta T} + \frac {7}{2} \eta \sigma^ {2} \quad \text {w h e r e} \quad \bar {\omega} _ {T} \stackrel {\text {d e f}} {=} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \omega_ {t} ^ {\prime}, \quad \forall T \geq 1. \tag {27}
$$

Since in practice  $\sigma \ll M$ , the variance term in (27) is significantly smaller than the one in (26). To summarize, SGD with extrapolation provides better convergence guarantees but requires two gradient computations and samples per iteration. This motivates our new method, Alg. 3 (AvgPastExtraSGD) which uses extrapolation from the past and achieves the best of both worlds.

Theorem 4. Under Assump. 1 and 3, if  $\mathbb{E}_{\xi}[F]$  is  $L$ -Lipschitz then SGD with extrapolation from the past using a constant step-size  $\eta \leq \frac{1}{2\sqrt{3}L}$ , gives that the averaged iterates converge as,

$$
\mathbb {E} \left[ \operatorname {E r r} \left(\bar {\omega} _ {T}\right) \right] \leq \frac {R ^ {2}}{\eta T} + \frac {1 3}{2} \eta \sigma^ {2} \quad \text {w h e r e} \quad \bar {\omega} _ {T} \stackrel {\text {d e f}} {=} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \omega_ {t} ^ {\prime} \quad \forall T \geq 1. \tag {28}
$$

The bounds is similar to the one provided in Thm. 3 but each iteration of Alg. 3 is computationally half the cost of an iteration of Alg. 2.

# 5 COMBINING THE TECHNIQUES WITH ESTABLISHED ALGORITHMS

In the previous sections we presented several techniques that converge on a simple bilinear example. These techniques can be combined in practice with existing algorithms. We propose to combine them to two standard algorithms used for training deep neural networks: the Adam optimizer (Kingma and Ba, 2015) and the SGD optimizer (Robbins and Monro, 1951). Note that in the case of a two-player game (3), the previous results can be generalized to gradient updates with a different step-size for each player by simply rescaling the objectives  $\mathcal{L}^{(\theta)}$  and  $\mathcal{L}^{(\varphi)}$  by a different scaling factor. A detailed pseudo-code for Adam with extrapolation step (Extra-Adam) is given in Algorithm 4.

Algorithm 4 Extra-Adam: proposed Adam with extrapolation step.  
input: step-size  $\eta$  decay rates for moment estimates  $\beta_{1},\beta_{2}$  , access to the stochastic gradients  $\nabla \ell_t(\cdot)$  and to the projection  $P_{\Omega}[\cdot ]$  onto the constraint set  $\Omega$  , initial parameter  $\omega_0$  , averaging scheme  $(\rho_{t})_{t\geq 1}$    
for  $t = 0\dots T - 1$  do Option 1: Standard extrapolation. Sample new minibatch and compute stochastic gradient:  $g_{t}\gets \nabla \ell_{t}(\omega_{t})$    
Option 2: Extrapolation from the past Load previously saved stochastic gradient:  $g_{t} = \nabla \ell_{t - 1 / 2}(\omega_{t - 1 / 2})$  Update estimate of first moment for extrapolation:  $m_{t - 1 / 2}\leftarrow \beta_1m_{t - 1} + (1 - \beta_1)g_t$  Update estimate of second moment for extrapolation:  $v_{t - 1 / 2}\leftarrow \beta_2v_{t - 1} + (1 - \beta_2)g_t^2$  Correct the bias for the moments:  $\hat{m}_{t - 1 / 2}\leftarrow m_{t - 1 / 2} / (1 - \beta_1^{2t - 1}),\hat{v}_{t - 1 / 2}\leftarrow v_{t - 1 / 2} / (1 - \beta_2^{2t - 1})$  Perform extrapolation step from iterate at time t:  $\omega_{t - 1 / 2}\gets P_{\Omega}[\omega_t - \eta \frac{m_{t - 1 / 2}}{\sqrt{v_{t - 1 / 2}} + \epsilon} ]$  Sample new minibatch and compute stochastic gradient:  $g_{t + 1 / 2}\leftarrow \nabla \ell_{t + 1 / 2}(\omega_{t + 1 / 2})$  Update estimate of first moment:  $m_t\gets \beta_1m_{t - 1 / 2} + (1 - \beta_1)g_{t + 1 / 2}$  Update estimate of second moment:  $v_{t}\gets \beta_{2}v_{t - 1 / 2} + (1 - \beta_{2})g_{t + 1 / 2}^{2}$  Compute bias corrected for first and second moment:  $\hat{m}_t\leftarrow m_t / (1 - \beta_1^{2t}),\hat{v}_t\leftarrow v_t / (1 - \beta_2^{2t})$  Perform update step from the iterate at time t:  $\omega_{t + 1}\gets P_{\Omega}[\omega_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t + \epsilon}]$    
end for   
Output:  $\omega_{T - 1 / 2},\omega_T$  or  $\bar{\omega}_T = \sum_{t = 0}^{T - 1}\rho_{t + 1}\omega_{t + 1 / 2} / \sum_{t = 0}^{T - 1}\rho_{t + 1}$  (see (8) for online averaging)

# 6 RELATED WORK

The extragradients method is the standard algorithm to optimize variational inequalities. This algorithm has been originally introduced by Korpelevich (1976) and extended by Nesterov (2007) and Nemirovski (2004). Stochastic versions of the extragradients have been recently analyzed (Juditsky et al., 2011; Yousefian et al., 2014; Iusem et al., 2017) for stochastic variational inequalities with bounded constraints. A linearly convergent variance reduced version of the stochastic gradient method has been proposed by Palaniappan and Bach (2016) for strongly monotone variational inequalities.

Several methods to stabilize GANs consist in transforming a zero-sum formulation into a more general game that can no longer be cast as a saddle point problem. This is the case of the non-saturating formulation of GANs (Goodfellow et al., 2014; Fedus et al., 2018), the DCGANs (Radford et al., 2016), the gradient penalty for WGANs (Gulrajani et al., 2017). Yadav et al. (2018) propose an optimization method for GANs based on AltSGD using a momentum based step on the generator. Daskalakis et al. (2018) proposed a method inspired from game theory. Li et al. (2017) suggest to dualize the GAN objective to reformulate it as a maximization problem and Mescheder et al. (2017) propose to add the norm of the gradient in the objective and provide an interesting perspective on GANs, interpreting the training as the search of a two-player game equilibrium. A study of the continuous version of two player games has been conducted by Ratliff et al. (2016). Interesting non-convex results were proved, for a new notion of regret minimization, by Hazan et al. (2017) and in the context of GANs by Grnarova et al. (2018).

The technique of unrolling steps proposed by Metz et al. (2017) can be confused with extrapolation but is actually fundamentally different: the perspective is try to construct the "true generator objective

function" unrolling for  $K$  steps the updates of the generator and then update the discriminator. Nevertheless the fact that this "true generator function" may not be found with a satisfying accuracy may lead to a different behavior than the one expected.

Regarding the averaging technique, some recent work appear to have already successfully used geometric averaging (7) for GANs in practice, but only briefly mention it (Karras et al., 2018; Mescheder et al., 2018). By contrast the present work formally motivates and justifies the use of averaging for GANs by relating them to the VIP perspective, and sheds light on its underlying intuitions in §3.1. Another independent work (Yazici et al., 2018) made a similar attempt but in the context of regret minimization in games. Mertikopoulos et al. (2018) also independently explored extrapolation providing asymptotic convergence results (i.e. without any rate of convergence) in the context of coherent saddle point. The coherence assumption is slightly weaker than monotonicity.

# 7 EXPERIMENTS

Our goal in this experimental section is not to provide new state-of-the-art results with architectural improvements or a new GAN formulation but to show that using the techniques (with theoretical guarantees in the monotone case) that we introduced earlier allow us to optimize standard GANs in a better way. These techniques, which are orthogonal to the design of new formulations of GAN optimization objectives, and to architectural choices, can potentially be used for the training of any type of GAN. We will compare the following optimization algorithms: baselines are SGD and Adam using either simultaneous updates on the generator and on the discriminator (denoted SimAdam and SimSGD) or  $k$  updates on the discriminator alternated with 1 update on the generator (denoted AltSGD{k} and AltAdam{k})<sup>8</sup>. Variants that use extrapolation are denoted ExtraSGD (Alg. 2) and ExtraAdam (Alg. 4). Variants using extrapolation from the past are PastExtraSGD (Alg. 3) and PastExtraAdam (Alg. 4). We also present results using as output the averaged iterates, adding Avg as a prefix of the algorithm name when we use (uniform) averaging.

# 7.1 BILINEAR SADDLE POINT (STOCHASTIC)

We evaluate the performance of the various stochastic algorithms first on a simple  $(n = 10^3, d = 10^3)$  finite sum bilinear objective (a monotone operator) constrained to  $[-1, 1]^d$ :

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \left(\boldsymbol {\theta} ^ {\top} \boldsymbol {M} ^ {(i)} \boldsymbol {\varphi} + \boldsymbol {\theta} ^ {\top} \boldsymbol {a} ^ {(i)} + \boldsymbol {\varphi} ^ {\top} \boldsymbol {b} ^ {(i)}\right) \tag {29}
$$

solved by  $(\pmb{\theta}^{*},\varphi^{*})$  s.t.  $\left\{ \begin{array}{l}\bar{M}\varphi^{*} = -\bar{a}\\ \bar{M}^{T}\pmb{\theta}^{*} = -\bar{b} \end{array} \right.$

where  $\bar{\pmb{a}}\stackrel {\mathrm{def}}{=}\frac{1}{n}\sum_{i = 1}^{n}\pmb{a}^{(i)}$ $\bar{\pmb{b}}\stackrel {\mathrm{def}}{=}\frac{1}{n}\sum_{i = 1}^{n}\pmb{b}^{(i)}$  and  $\bar{M}\stackrel {\mathrm{def}}{=}\frac{1}{n}\sum_{i = 1}^{n}M^{(i)}$  . Marices and vectors  $M_{kj}^{(i)},a_k^{(i)},b_k^{(i)};1\leq i\leq n,1\leq j,k\leq d$  were randomly generated, but ensuring that  $(\theta^{*},\varphi^{*})$  would belong to  $[-1,1]^d$  . Results are shown in Fig. 3. We can see that AvgSGD and AvgPastExtraSGD perform the best on this task.

![](images/8e4813e3d75900371b5ba15cba994e385f1a34d14c189c0e5cc7eb73b85d69a6.jpg)  
Figure 3: Performance of the considered stochastic optimization algorithms on the bilinear problem (29). Each method uses its respective optimal step-size found by grid-search.

# 7.2 WGAN AND WGAN-GP ON CIFAR10

We now evaluate the proposed techniques in the context of GAN training, which is a challenging stochastic optimization problem where the objectives of both players are non-convex. We focus on the more dovanced Adam variants of optimization algorithms (see Alg. 4 for Adam with extrapolation) and compare them for training a fixed DCGAN architecture (Radford et al., 2016) on the CIFAR10 dataset (Krizhevsky and Hinton, 2009) for two different training objectives: WGAN with weight

<table><tr><td>Model</td><td colspan="3">WGAN</td><td colspan="2">WGAN-GP</td></tr><tr><td>Method</td><td>no averaging</td><td>uniform avg</td><td>EMA</td><td>no averaging</td><td>uniform avg</td></tr><tr><td>SimAdam</td><td>6.05 ± .12</td><td>5.83 ± .16</td><td>6.08 ± .10</td><td>6.00 ± .07</td><td>6.01 ± .08</td></tr><tr><td>AltAdam5</td><td>5.45 ± .08</td><td>5.72 ± .06</td><td>5.49 ± .05</td><td>6.25 ± .05</td><td>6.51 ± .05</td></tr><tr><td>ExtraAdam</td><td>6.38 ± .09</td><td>6.38 ± .20</td><td>6.37 ± .08</td><td>6.22 ± .04</td><td>6.35 ± .05</td></tr><tr><td>PastExtraAdam</td><td>5.98 ± 0.15</td><td>6.07 ± 0.19</td><td>6.01 ± 0.11</td><td>6.27 ± 0.06</td><td>6.23 ± 0.13</td></tr><tr><td>OptimAdam</td><td>5.74 ± 0.10</td><td>5.80 ± 0.08</td><td>5.78 ± 0.05</td><td>-</td><td>-</td></tr></table>

Table 1: Best inception scores (averaged over 5 runs) achieved on CIFAR10 for every considered Adam variant. OptimAdam is the related Optimistic Adam (Daskalakis et al., 2018) algorithm. EMA denotes exponential moving average (with  $\beta = 0.999$ , see Eq. 8). We see that the techniques of extrapolation and averaging consistently enable improvements over the baselines (in italic).

![](images/e20127f66fed392cc3655b0e0883d65f26748b945ffa025df927abd64d33ec6c.jpg)  
Figure 4: Left: Mean and standard deviation of the inception score computed over 5 runs for each method on WGAN trained on CIFAR10. To keep the graph readable we show only SimAdam but AltAdam performs similarly. Middle: Samples from a generator trained as a WGAN using ExtraAdam. Right: WGAN-GP trained on CIFAR10: mean and standard deviation of the inception score computed over 5 runs for each method using the best performing learning rate plotted over wall-clock time; all experiments were run on a NVIDIA Quadro GP100 GPU. We see that ExtraAdam converges faster than the Adam baselines.

![](images/8d998a7d98e5c4682dda056e536044e5722c6cb9adc3326c0c4aecd66d916dca.jpg)

![](images/93d3feae16ae7d331adb45bf77f4a7f68f35570a014c956bfff5cf71e69ecccb.jpg)

clipping (constrained) (Arjovsky et al., 2017), and a WGAN-GP objective (Gulrajani et al., 2017) (a non-zero sum game). Models are evaluated using the inception score (Salimans et al., 2016). For each algorithm we did an extensive search over the hyperparameters of Adam ( $\beta_{1} = 0.5$  and  $\beta_{2} = 0.9$  performed best for all). We ran each with 5 different random seeds for 500,000 iterations.

Table 1 reports the best inception score achieved on this problem by each considered method. We see that the techniques of extrapolation and averaging consistently enable improvements over the baselines (see §G.3 for more experiments on averaging). Fig. 4 shows training curves for each method (for their best performing learning rate), as well as samples from an ExtraAdam-trained WGAN. For training WGAN, using an extrapolation step with Adam (ExtraAdam) outperformed all other methods. For training WGAN-GP, the best results are achieved with uniform averaging of AltAdam5. However its iterations require to update the discriminator 5 times for every generator update. With a small drop in best final score, ExtraAdam can train WGAN-GP significantly faster (see Fig. 4 right) as the discriminator and generator are updated only twice. We also observed that methods based on extrapolation are less sensitive to the choice of learning rate and can be used with higher learning rates with less degradation; see App. §G.2 for more details.

# 8 CONCLUSION

We newly addressed GAN objectives in the framework of variational inequality. We tapped into the optimization literature to provide more principled sound techniques to optimize such games. We leveraged these techniques to develop practical optimization algorithms suitable for a wide range of GAN training objectives (including non-zero sum games and projections onto constraints). We experimentally verified that this could yield better trained models, achieving to our knowledge the best inception score when optimizing a WGAN objective on the reference unmodified DCGAN architecture (Radford et al., 2016). The presented techniques address a fundamental problem in GAN training in a principled way, and are orthogonal to the design of new GAN architectures and objectives. They are thus likely to be widely applicable, and benefit future development of GANs.

# REFERENCES

M. Arjovsky, S. Chintala, and L. Bottou. Wasserstein generative adversarial networks. In ICML, 2017.  
K. E. Atkinson. An introduction to numerical analysis. John Wiley & Sons, 2003.  
S. Boyd and L. Vandenberghe. Convex optimization. Cambridge university press, 2004.  
R. E. Bruck. On the weak convergence of an ergodic iteration for the solution of variational inequalities for monotone operators in hilbert space. Journal of Mathematical Analysis and Applications, 1977.  
G. H. Chen and R. T. Rockafellar. Convergence rates in forward-backward splitting. SIAM Journal on Optimization, 1997.  
G. P. Crespi, A. Guerraggio, and M. Rocca. Minty variational inequality and optimization: Scalar and vector case. In Generalized Convexity, Generalized Monotonicity and Applications, 2005.  
C. Daskalakis, A. Ilyas, V. Syrgkanis, and H. Zeng. Training GANs with optimism. In  $ICLR$ , 2018.  
W. Fedus, M. Rosca, B. Lakshminarayanan, A. M. Dai, S. Mohamed, and I. Goodfellow. Many paths to equilibrium: GANs do not need to decrease a divergence at every step. In *ICLR*, 2018.  
G. Gidel, T. Jebara, and S. Lacoste-Julien. Frank-wolfe algorithms for saddle point problems. In AISTATS, 2017.  
I. Goodfellow. Nips 2016 tutorial: Generative adversarial networks. arXiv:1701.00160, 2016.  
I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In NIPS, 2014.  
P. Grinarova, K. Y. Levy, A. Lucchi, T. Hofmann, and A. Krause. An online learning approach to generative adversarial networks. In ICLR, 2018.  
I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville. Improved training of wasserstein GANs. In NIPS, 2017.  
P. T. Harker and J.-S. Pang. Finite-dimensional variational inequality and nonlinear complementarity problems: a survey of theory, algorithms and applications. Mathematical programming, 1990.  
E. Hazan, K. Singh, and C. Zhang. Efficient regret minimization in non-convex games. In ICML, 2017.  
M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In NIPS, 2017.  
A. Iusem, A. Jofre, R. I. Oliveira, and P. Thompson. Extragradient method with variance reduction for stochastic variational inequalities. SIAM Journal on Optimization, 2017.  
A. Juditsky, A. Nemirovski, and C. Tauvel. Solving variational inequalities with stochastic mirror-prox algorithm. Stochastic Systems, 2011.  
T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive growing of gans for improved quality, stability, and variation, 2018.  
D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In ICLR, 2015.  
G. Korpelevich. The extragradient method for finding saddle points and other problems. Matecon, 12, 1976.  
A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Master's thesis, University of Toronto, Canada, 2009.  
T. Larsson and M. Patriksson. A class of gap functions for variational inequalities. Math. Program., 1994.

C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, A. Acosta, A. P. Aitken, A. Tejani, J. Totz, Z. Wang, et al. Photo-realistic single image super-resolution using a generative adversarial network. In CVPR, 2017.  
Y. Li, A. Schwing, K.-C. Wang, and R. Zemel. Dualing GANs. In NIPS, 2017.  
P. Mertikopoulos, H. Zenati, B. Lecouat, C.-S. Foo, V. Chandrasekhar, and G. Piliouras. Mirror descent in saddle-point problems: Going the extra (gradient) mile. arXiv, 2018.  
L. Mescheder, S. Nowozin, and A. Geiger. The numerics of GANs. In NIPS, 2017.  
L. Mescheder, A. Geiger, and S. Nowozin. Which training methods for GANs do actually converge? In ICML, 2018.  
L. Metz, B. Poole, D. Pfau, and J. Sohl-Dickstein. Unrolled generative adversarial networks. In ICLR, 2017.  
V. Nagarajan and J. Z. Kolter. Gradient descent GAN optimization is locally stable. In NIPS, 2017.  
A. Nedic and A. Ozdaglar. Subgradient methods for saddle-point problems. J Optim Theory Appl, 2009.  
A. Nemirovski. Prox-method with rate of convergence  $O(1 / t)$  for variational inequalities with Lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM J. Optim., 2004.  
A. Nemirovski, A. Juditsky, G. Lan, and A. Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on optimization, 2009.  
Y. Nesterov. Introductory Lectures On Convex Optimization. Springer, 1983.  
Y. Nesterov. Dual extrapolation and its applications to solving variational inequalities and related problems. Math. Program., 2007.  
S. Nowozin, B. Cseke, and R. Tomioka. f-GAN: Training generative neural samplers using variational divergence minimization. In NIPS, 2016.  
B. Palaniappan and F. Bach. Stochastic variance reduction methods for saddle-point problems. In NIPS, 2016.  
B. T. Polyak. Gradient methods for minimizing functionals. Zhurnal Vychislitel'noi Matematiki i Matematicheskoi Fiziki, 1963.  
A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In ICLR, 2016.  
A. Rakhlin and K. Sridharan. Online learning with predictable sequences. In  $COLT$ , 2013.  
L. J. Ratliff, S. A. Burden, and S. S. Sastry. On the characterization of local nash equilibria in continuous games. 2016.  
H. Robbins and S. Monro. A stochastic approximation method. The Annals of Mathematical Statistics, 1951.  
T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training GANs. In NIPS, 2016.  
I. Sutskever. Training recurrent neural networks. PhD thesis, 2013.  
P. Tseng. On linear convergence of iterative methods for the variational inequality problem. Journal of Computational and Applied Mathematics, 1995.  
J. Von Neumann and O. Morgenstern. Theory of games and economic behavior. Princeton University Press, 1944.

A. Yadav, S. Shah, Z. Xu, D. Jacobs, and T. Goldstein. Stabilizing adversarial nets with prediction methods. In ICLR, 2018.  
Y. Yazici, C.-S. Foo, S. Winkler, K.-H. Yap, G. Piliouras, and V. Chandrasekhar. The unusual effectiveness of averaging in gan training. arXiv preprint arXiv:1806.04498, 2018.  
F. Yousefian, A. Nedic, and U. V. Shanbhag. Optimal robust smoothing extragradients algorithms for stochastic variational inequality problems. In CDC. IEEE, 2014.  
J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In ICCV, 2017.
