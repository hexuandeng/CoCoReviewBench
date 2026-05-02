# CERTIFIABLE DISTRIBUTIONAL ROBUSTNESS WITH PRINCIPLED ADVERSARIAL TRAINING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Neural networks are vulnerable to adversarial examples and researchers have proposed many heuristic attack and defense mechanisms. We take the principled view of distributionally robust optimization, which guarantees performance under adversarial input perturbations. By considering a Lagrangian penalty formulation of perturbation of the underlying data distribution in a Wasserstein ball, we provide a training procedure that augments model parameter updates with worst-case perturbations of training data. For smooth losses, our procedure provably achieves moderate levels of robustness with little computational or statistical cost relative to empirical risk minimization. Furthermore, our statistical guarantees allow us to efficiently certify robustness for the population loss. For imperceptible perturbations, our method matches or outperforms heuristic approaches.

# 1 INTRODUCTION

Consider the classical supervised learning problem, in which we minimize the expected loss  $\mathbb{E}_{P_0}[\ell (\theta ;Z)]$  over a parameter  $\theta \in \Theta$ , where  $Z\sim P_0$  is a distribution on a space  $\mathcal{Z}$  and  $\ell$  is a loss function. In many systems, robustness to changes in the data-generating distribution  $P_{0}$  is desirable, whether they be from covariate shifts, changes in the underlying domain (Ben-David et al., 2010), or adversarial attacks (Goodfellow et al., 2015; Kurakin et al., 2016). As deep networks become prevalent in modern performance-critical systems (e.g. perception for self-driving cars, automated detection of tumors), model failure increasingly leads to life-threatening situations; in these systems, it is irresponsible to deploy models whose robustness we cannot certify.

However, recent works have shown that neural networks are vulnerable to adversarial examples; seemingly imperceptible perturbations to data can lead to misbehavior of the model, such as misclassifications of the output (Goodfellow et al., 2015; Kurakin et al., 2016; Moosavi-Dezfooli et al., 2016; Nguyen et al., 2015). Subsequently, many researchers have proposed adversarial attack and defense mechanisms (Rozsa et al., 2016; Papernot et al., 2016a;b;c; Tramér et al., 2017; Carlini & Wagner, 2017; Madry et al., 2017; He et al., 2017). While these works provide an initial foundation for adversarial training, there are no guarantees on whether proposed white-box attacks can find the most adversarial perturbation and whether there is a class of attacks such defenses can successfully prevent. On the other hand, verification of deep networks using SMT solvers (Katz et al., 2017a;b; Huang et al., 2017) provides formal guarantees on robustness but is NP-hard in general; this approach requires prohibitive computational expense even on small networks.

We take the perspective of distributionally robust optimization and provide an adversarial training procedure with provable guarantees on its computational and statistical performance. We postulate a class  $\mathcal{P}$  of distributions around the data-generating distribution  $P_0$  and consider the problem

$$
\underset {\theta \in \Theta} {\text {m i n i m i z e}} \sup  _ {P \in \mathcal {P}} \mathbb {E} _ {P} [ \ell (\theta ; Z) ]. \tag {1}
$$

The choice of  $\mathcal{P}$  influences robustness guarantees and computability; we develop robustness sets  $\mathcal{P}$  with computationally efficient relaxations that apply even when the loss  $\ell$  is non-convex. We provide an adversarial training procedure that, for smooth  $\ell$ , enjoys convergence guarantees similar to non-robust approaches while certifying performance even for the worst-case population loss  $\sup_{P\in \mathcal{P}}\mathbb{E}_P[\ell (\theta ;Z)]$ . On a simple implementation in Tensorflow, our method takes  $5 - 10\times$  as long as stochastic gradient methods for empirical risk minimization (ERM), matching runtimes for other

adversarial training procedures (Goodfellow et al., 2015; Kurakin et al., 2016; Madry et al., 2017). We show that our procedure—which learns to protect against adversarial perturbations in the training dataset—generalizes, allowing us to train a model that prevents attacks to the test dataset.

We briefly overview our approach. Let  $c: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}_+ \cup \{\infty\}$ , where  $c(z, z_0)$  is the "cost" for an adversary to perturb  $z_0$  to  $z$  (we typically use  $c(z, z_0) = \|z - z_0\|_p^2$  with  $p \geq 1$ ). We consider the robustness region  $\mathcal{P} = \{P: W_c(P, P_0) \leq \rho\}$ , a  $\rho$ -neighborhood of the distribution  $P_0$  under the Wasserstein metric  $W_c(\cdot, \cdot)$  (see Section 2 for a formal definition). For deep networks and other complex models, this formulation of problem (1) is intractable with arbitrary  $\rho$ . Instead, we consider its Lagrangian relaxation for a fixed penalty parameter  $\gamma \geq 0$ , resulting in the reformulation

$$
\underset {\theta \in \Theta} {\text {m i n i m i z e}} \left\{F (\theta) := \sup  _ {P} \left\{\mathbb {E} _ {P} [ \ell (\theta ; Z) ] - \gamma W _ {c} (P, P _ {0}) \right\} = \mathbb {E} _ {P _ {0}} \left[ \phi_ {\gamma} (\theta ; Z) \right] \right\} \tag {2a}
$$

$$
\text {w h e r e} \phi_ {\gamma} (\theta ; z _ {0}) := \sup  _ {z \in \mathcal {Z}} \left\{\ell (\theta ; z) - \gamma c (z, z _ {0}) \right\}. \tag {2b}
$$

(See Proposition 1 for a rigorous statement of these equalities.) Here, we have replaced the usual loss  $\ell(\theta; Z)$  by the robust surrogate  $\phi_{\gamma}(\theta; Z)$ ; this surrogate (2b) allows adversarial perturbations of the data  $z$ , modulated by the penalty  $\gamma$ . We typically solve the penalty problem (2) with  $P_0$  replaced by the empirical distribution  $\widehat{P}_n$ , as  $P_0$  is unknown (we refer to this as the penalty problem below).

The key feature of the penalty problem (2) is that moderate levels of robustness—in particular, defense against imperceptible adversarial perturbations—are achievable at essentially no computational or statistical cost for smooth losses  $\ell$ . Specifically, for large enough penalty  $\gamma$  (by duality, small enough robustness  $\rho$ ), the function  $z \mapsto \ell(\theta; z) - \gamma c(z, z_0)$  in the robust surrogate (2b) is strongly concave and hence easy to optimize if  $\ell(\theta, z)$  is smooth in  $z$ . Consequently, stochastic gradient methods applied to problem (2) have similar convergence guarantees as for non-robust methods (ERM). In Section 3, we provide a certificate of robustness for any  $\rho$ ; we give an efficiently computable data-dependent upper bound on the worst-case loss  $\sup_{P: W_c(P, P_0) \leq \rho} \mathbb{E}_P[\ell(\theta; Z)]$ . That is, the worst-case performance of the output of our principled adversarial training procedure is guaranteed to be no worse than this certificate. Our bound is tight when  $\rho = \widehat{\rho}_n$ , the achieved robustness for the empirical objective. These results suggest advantages of networks with smooth activations rather than ReLU's. We experimentally verify our results in Section 4 and show that, even for non-smooth losses, we match or achieve state-of-the-art performance on a variety of adversarial attacks.

Robust optimization and adversarial training The standard robust-optimization approach minimizes losses of the form  $\sup_{u\in \mathcal{U}}\ell (\theta ;z + u)$  for some uncertainty set  $\mathcal{U}$  (Ben-Tal et al., 2009; Ratliff et al., 2006; Xu et al., 2009). Unfortunately, this approach is intractable except for specially structured losses, such as the composition of a linear and simple convex function (Ben-Tal et al., 2009; Xu et al., 2009; 2012). Nevertheless, this robust approach underlies recent advances in adversarial training (Szegedy et al., 2013; Goodfellow et al., 2015; Papernot et al., 2016b; Carlini & Wagner, 2017; Madry et al., 2017), which heuristically perturb data during a stochastic optimization procedure.

One such heuristic uses a locally linearized loss function (proposed with  $p = \infty$  as the "fast gradient sign method" (Goodfellow et al., 2015)):

$$
\Delta_ {x _ {i}} (\theta) := \underset {\| \eta \| _ {p} \leq \epsilon} {\operatorname {a r g m a x}} \left\{\nabla_ {x} \ell \left(\theta ; \left(x _ {i}, y _ {i}\right)\right) ^ {T} \eta \right\} \text {a n d p e r t u r b} x _ {i} \rightarrow x _ {i} + \Delta_ {x _ {i}} (\theta). \tag {3}
$$

One form of adversarial training simply trains on these perturbed losses (Goodfellow et al., 2015; Kurakin et al., 2016), and many others perform iterated variants (Papernot et al., 2016b; Tramère et al., 2017; Carlini & Wagner, 2017; Madry et al., 2017). Madry et al. (2017) observe that these procedures attempt to optimize the objective  $\mathbb{E}_{P_0}[\sup_{\| u\| _p\leq \epsilon}\ell (\theta ;Z + u)]$ , a constrained version of the penalty problem (2). This notion of robustness is typically intractable: the inner supremum is generally non-concave in  $u$ , so it is unclear whether model-fitting with these techniques converges, and there are possibly worst-case perturbations these techniques do not find. Indeed, it is NP-hard to find worst-case perturbations when deep networks use ReLU activations, suggesting difficulties for fast and iterated heuristics (see Lemma 2 in Appendix B). Smoothness, which can be obtained in standard deep architectures with exponential linear units (ELU's) (Clevert et al., 2015), allows us to find Lagrangian worst-case perturbations with low computational cost.

Distributionally robust optimization To situate the current work, we review some of the substantial body of work on robustness and learning. The choice of  $\mathcal{P}$  in the robust objective (1) affects

both the richness of the uncertainty set we wish to consider as well as the tractability of the resulting optimization problem. Previous approaches to distributional robustness have considered finite-dimensional parametrizations for  $\mathcal{P}$ , such as constraint sets for moments, support, or directional deviations (Chen et al., 2007; Delage & Ye, 2010; Goh & Sim, 2010), as well as non-parametric distances for probability measures such as  $f$ -divergences (Ben-Tal et al., 2013; Bertsimas et al., 2013; Miyato et al., 2015; Lam & Zhou, 2015; Duchi et al., 2016; Namkoong & Duchi, 2016), and Wasserstein distances (Blanchet et al., 2016; Esfahani & Kuhn, 2015; Shafieezadeh-Abadeh et al., 2015). In contrast to  $f$ -divergences (e.g.  $\chi^2$ - or Kullback-Leibler divergences) which are effective when the support of the distribution  $P_0$  is fixed, a Wasserstein ball around  $P_0$  includes distributions  $Q$  with different support and allows (in a sense) robustness to unseen data.

Many authors have studied tractable classes of uncertainty sets  $\mathcal{P}$  and losses  $\ell$ . For example, Ben-Tal et al. (2013) and Namkoong & Duchi (2017) use convex optimization approaches for  $f$ -divergence balls. For worst-case regions  $\mathcal{P}$  formed by Wasserstein balls, Esfahani & Kuhn (2015), Shafieezadeh-Abadeh et al. (2015) and Blanchet et al. (2016) show how to convert the saddle-point problem (1) to a regularized ERM problem, but this is possible only for a limited class of convex losses  $\ell$  and costs  $c$ . In this work, we treat a much larger class of losses and costs and provide direct solution methods for a Lagrangian relaxation of the saddle-point problem (1).

# 2 PROPOSED APPROACH

Our approach is based on the following simple insight: assume that the function  $z \mapsto \ell(\theta; z)$  is smooth, meaning there is some  $L$  for which  $\nabla_z \ell(\theta; \cdot)$  is  $L$ -Lipschitz. Then for any  $c: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}_+ \cup \{\infty\}$  1-strongly convex in its first argument, a Taylor expansion yields

$$
\ell (\theta ; z ^ {\prime}) - \gamma c (z ^ {\prime}, z _ {0}) \leq \ell (\theta ; z) - \gamma c (z, z _ {0}) + \left\langle \nabla_ {z} \left(\ell (\theta ; z) - \gamma c (z, z _ {0})\right), z ^ {\prime} - z \right\rangle + \frac {L - \gamma}{2} \| z - z ^ {\prime} \| _ {2} ^ {2}. \tag {4}
$$

For  $\gamma \geq L$  this is the first-order condition for  $(\gamma -L)$ -strong concavity of  $z\mapsto (\ell (\theta ;z) - \gamma c(z,z_0))$ . Thus, whenever the loss is smooth enough in  $z$  and the penalty  $\gamma$  is large enough (corresponding to less robustness), computing the surrogate (2b) is a strongly-concave optimization problem.

We leverage the insight (4) to show that as long as we do not require too much robustness, this strong concavity approach (4) provides a computationally efficient and principled approach for robust optimization problems (1). Our starting point is a duality result for the minimax problem (1) and its Lagrangian relaxation for Wasserstein-based uncertainty sets, which makes the connections between distributional robustness and the "lazy" surrogate (2b) clear. We then show (Section 2.1) how stochastic gradient descent methods can efficiently find minimizers (in the convex case) or approximate stationary points (when  $\ell$  is non-convex) for our relaxed robust problems.

Wasserstein robustness and duality Wasserstein distances define a notion of closeness between distributions. Let  $\mathcal{Z} \subset \mathbb{R}^m$ , and let  $(\mathcal{Z}, \mathcal{A}, P_0)$  be a probability space. Let the transportation cost  $c: \mathcal{Z} \times \mathcal{Z} \to [0, \infty)$  be nonnegative, lower semi-continuous, and satisfy  $c(z, z) = 0$ . For example, for a differentiable convex  $h: \mathcal{Z} \to \mathbb{R}$ , the Bregman divergence  $c(z, z_0) = h(z) - h(z_0) - \langle \nabla h(z_0), z - z_0 \rangle$  satisfies these conditions. For probability measures  $P$  and  $Q$  supported on  $\mathcal{Z}$ , let  $\Pi(P, Q)$  denote their couplings, meaning measures  $M$  on  $\mathcal{Z}^2$  with  $M(A, \mathcal{Z}) = P(A)$  and  $M(\mathcal{Z}, A) = Q(A)$ . The Wasserstein distance between  $P$  and  $Q$  is

$$
W _ {c} (P, Q) := \inf  _ {M \in \Pi (P, Q)} \mathbb {E} _ {M} [ c (Z, Z ^ {\prime}) ].
$$

For  $\rho \geq 0$  and distribution  $P_0$ , we let  $\mathcal{P} = \{P:W_c(P,P_0)\leq \rho \}$ , considering the Wasserstein form of the robust problem (1) and its Lagrangian relaxation (2) with  $\gamma \geq 0$ . The following duality result due to Blanchet & Murthy (2016) gives the equality (2) for the relaxation and an analogous result for the problem (1). We give an alternate proof in Appendix C.1 for convex, continuous cost functions.

Proposition 1. Let  $\ell : \Theta \times \mathcal{Z} \to \mathbb{R}$  and  $c: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}_+$  be continuous. Let  $\phi_{\gamma}(\theta; z_0) = \sup_{z \in \mathcal{Z}} \{\ell(\theta; z) - \gamma c(z, z_0)\}$  be the robust surrogate (2b). For any distribution  $Q$  and any  $\rho > 0$ ,

$$
\sup  _ {P: W _ {c} (P, Q) \leq \rho} \mathbb {E} _ {P} [ \ell (\theta ; Z) ] = \inf  _ {\gamma \geq 0} \left\{\gamma \rho + \mathbb {E} _ {Q} [ \phi_ {\gamma} (\theta ; Z) ] \right\}, \tag {5}
$$

and for any  $\gamma \geq 0$ , we have

$$
\sup  _ {P} \left\{\mathbb {E} _ {P} [ \ell (\theta ; Z) ] - \gamma W _ {c} (P, Q) \right\} = \mathbb {E} _ {Q} \left[ \phi_ {\gamma} (\theta ; Z) \right]. \tag {6}
$$

# Algorithm 1 Distributionally robust optimization with adversarial training

INPUT: Sampling distribution  $P_0$ , constraint sets  $\Theta$  and  $\mathcal{Z}$ , stepsize sequence  $\{\alpha_t > 0\}_{t=0}^{T-1}$  for  $t = 0, \dots, T-1$  do

Sample  $z^t \sim P_0$  and find an  $\epsilon$ -approximate maximizer  $\widehat{z}^t$  of  $\ell(\theta^t; z) - \gamma c(z, z^t)$

$$
\theta^ {t + 1} \leftarrow \operatorname {P r o j} _ {\Theta} \left(\theta^ {t} - \alpha_ {t} \nabla_ {\theta} \ell \left(\theta^ {t}; \widehat {z} ^ {t}\right)\right)
$$

Leveraging the insight (4), we give up the requirement that we wish a prescribed amount  $\rho$  of robustness (solving the worst-case problem (1) for  $\mathcal{P} = \{P:W_c(P,P_0)\leq \rho \}$ ) and focus instead on the Lagrangian penalty problem (2) and its empirical counterpart

$$
\underset {\theta \in \Theta} {\operatorname {m i n i m i z e}} \left\{F _ {n} (\theta) := \sup  _ {P} \left\{\mathbb {E} [ \ell (\theta ; Z) ] - \gamma W _ {c} (P, \widehat {P} _ {n}) \right\} = \mathbb {E} _ {\widehat {P} _ {n}} \left[ \phi_ {\gamma} (\theta ; Z) \right] \right\}. \tag {7}
$$

# 2.1 OPTIMIZING THE ROBUST LOSS BY STOCHASTIC GRADIENT DESCENT

We now develop stochastic gradient-type methods for the relaxed robust problem (7), making clear the computational benefits of relaxing the strict robustness requirements of formulation (5). We begin with assumptions we require, which roughly quantify the amount of robustness we can provide.

Assumption A. The function  $c: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}_+$  is continuous. For each  $z_0 \in \mathcal{Z}$ ,  $c(\cdot, z_0)$  is 1-strongly convex with respect to the norm  $\|\cdot\|$ .

To guarantee that the robust surrogate (2b) is tractably computable, we also require a few smoothness assumptions. Let  $\| \cdot \|_{*}$  be the dual norm to  $\| \cdot \|$ ; we abuse notation by using the same norm  $\| \cdot \|$  on  $\Theta$  and  $\mathcal{Z}$ , though the specific norm is clear from context.

Assumption B. The loss  $\ell :\Theta \times \mathcal{Z}\to \mathbb{R}$  satisfies the Lipschitzian smoothness conditions

$$
\left\| \nabla_ {\theta} \ell (\theta ; z) - \nabla_ {\theta} \ell \left(\theta^ {\prime}; z\right) \right\| _ {*} \leq L _ {\theta \theta} \| \theta - \theta^ {\prime} \|, \left\| \nabla_ {z} \ell (\theta ; z) - \nabla_ {z} \ell \left(\theta ; z ^ {\prime}\right) \right\| _ {*} \leq L _ {z z} \| z - z ^ {\prime} \|,
$$

$$
\left\| \nabla_ {\theta} \ell (\theta ; z) - \nabla_ {\theta} \ell \left(\theta ; z ^ {\prime}\right) \right\| _ {*} \leq L _ {\theta z} \| z - z ^ {\prime} \|, \left\| \nabla_ {z} \ell (\theta ; z) - \nabla_ {z} \ell \left(\theta^ {\prime}; z\right) \right\| _ {*} \leq L _ {z \theta} \| \theta - \theta^ {\prime} \|.
$$

These properties guarantee both (i) the well-behavedness of the robust surrogate  $\phi_{\gamma}$  and (ii) its efficient computability. Making point (i) precise, Lemma 1 shows that if  $\gamma$  is large enough and Assumption B holds, the surrogate  $\phi_{\gamma}$  is still smooth. Throughout, we assume  $\Theta \subseteq \mathbb{R}^d$ .

Lemma 1. Let  $f: \Theta \times \mathcal{Z} \to \mathbb{R}$  be differentiable and  $\lambda$ -strongly concave in  $z$  with respect to the norm  $\| \cdot \|$ , and define  $\bar{f}(\theta) = \sup_{z \in \mathcal{Z}} f(\theta, z)$ . Let  $\mathbf{g}_{\theta}(\theta, z) = \nabla_{\theta} f(\theta, z)$  and  $\mathbf{g}_{\mathbf{z}}(\theta, z) = \nabla_{z} f(\theta, z)$ , and assume  $\mathbf{g}_{\theta}$  and  $\mathbf{g}_{\mathbf{z}}$  satisfy Assumption B with  $\ell(\theta; z)$  replaced with  $f(\theta, z)$ . Then  $\bar{f}$  is differentiable, and letting  $z^{\star}(\theta) = \operatorname{argmax}_{z \in \mathcal{Z}} f(\theta, z)$ , we have  $\nabla \bar{f}(\theta) = \mathbf{g}_{\theta}(\theta, z^{\star}(\theta))$ . Moreover,

$$
\left\| z ^ {\star} \left(\theta_ {1}\right) - z ^ {\star} \left(\theta_ {2}\right) \right\| \leq \frac {L _ {z \theta}}{\lambda} \left\| \theta_ {1} - \theta_ {2} \right\| a n d \left\| \nabla \bar {f} (\theta) - \nabla \bar {f} \left(\theta^ {\prime}\right) \right\| _ {\star} \leq \left(L _ {\theta \theta} + \frac {L _ {\theta z} L _ {z \theta}}{\lambda}\right) \| \theta - \theta^ {\prime} \|.
$$

See Section C.2 for the proof. Fix  $z_0 \in \mathcal{Z}$  and focus on the  $\ell_2$ -norm case where  $c(z, z_0)$  satisfies Assumption A with  $\| \cdot \|_2$ . Noting that  $f(\theta, z) \coloneqq \ell(\theta, z) - \gamma c(z, z_0)$  is  $(\gamma - L_{zz})$ -strongly concave from the insight (4) (with  $L \coloneqq L_{zz}$ ), let us apply Lemma 1. Under Assumptions A, B,  $\phi_\gamma(\cdot; z_0)$  then has  $L = L_{\theta \theta} + \frac{L_{\theta z} L_{z\theta}}{[\gamma - L_{zz}]_+}$ -Lipschitz gradients, and

$$
\nabla_ {\theta} \phi_ {\gamma} (\theta ; z _ {0}) = \nabla_ {\theta} \ell (\theta ; z ^ {\star} (z _ {0}, \theta)) \text {w h e r e} z ^ {\star} (z _ {0}, \theta) = \operatorname * {a r g m a x} _ {z \in \mathcal {Z}} \{\ell (\theta ; z) - \gamma c (z, z _ {0}) \}.
$$

This motivates Algorithm 1, a stochastic-gradient approach for the penalty problem (7). The benefits of Lagrangian relaxation become clear here: for  $\ell(\theta;z)$  smooth in  $z$  and  $\gamma$  large enough, gradient ascent on  $\ell(\theta^t;z) - \gamma c(z,z^t)$  in  $z$  converges linearly and we can compute (approximate)  $\widehat{z}^t$  efficiently (we initialize our inner gradient ascent iterations with the sampled natural example  $z^t$ ).

Convergence properties of Algorithm 1 depend on the loss  $\ell$ . When  $\ell$  is convex in  $\theta$  and  $\gamma$  is large enough that  $z \mapsto (\ell(\theta; z) - \gamma c(z, z_0))$  is concave for all  $(\theta, z_0) \in \Theta \times \mathcal{Z}$ , we have a stochastic monotone variational inequality, which is efficiently solvable (Juditsky et al., 2011; Chen et al., 2014) with convergence rate  $1 / \sqrt{T}$ . When the loss  $\ell$  is nonconvex in  $\theta$ , the following theorem guarantees convergence to a stationary point of problem (7) at the same rate when  $\gamma \geq L_{zz}$ . Recall that  $F(\theta) = \mathbb{E}_{P_0}[\phi_\gamma(\theta; Z)]$  is the robust surrogate objective for the Lagrangian relaxation (2).

Theorem 2 (Convergence of Nonconvex SGD). Let Assumptions  $A$  and  $B$  hold with the  $\ell_2$ -norm and let  $\Theta = \mathbb{R}^d$ . Let  $\Delta_F \geq F(\theta^0) - \inf_{\theta} F(\theta)$ . Assume  $\mathbb{E}[\| \nabla F(\theta) - \nabla_{\theta} \phi_{\gamma}(\theta, Z) \|_2^2] \leq \sigma^2$ , and take constant step sizes  $\alpha = \sqrt{\frac{2\Delta_F}{L_\phi \sigma^2 T}}$  where  $L_\phi \coloneqq L_{\theta \theta} + \frac{L_{\theta z} L_{z\theta}}{\gamma - L_{zz}}$ . Then Algorithm 1 satisfies

$$
\frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {E} \left[ \left\| \nabla F (\theta^ {t}) \right\| _ {2} ^ {2} \right] - \frac {2 L _ {\theta \mathbf {z}} ^ {2}}{\gamma - L _ {\mathbf {z z}}} \epsilon \leq \sigma \sqrt {8 \frac {L _ {\phi} \Delta_ {F}}{T}}.
$$

See Section C.3 for the proof. We make a few remarks. First, the condition  $\mathbb{E}[\| \nabla F(\theta) - \nabla_{\theta}\phi_{\gamma}(\theta ,Z)\| _2^2 ]\leq \sigma^2$  holds (to within a constant factor) whenever  $\| \nabla_{\theta}\ell (\theta ,z)\| _2\leq \sigma$  for all  $\theta ,z$ . Theorem 2 shows that the stochastic gradient method achieves the rates of convergence on the penalty problem (7) achievable in standard smooth non-convex optimization (Ghadimi & Lan, 2013). The accuracy parameter  $\epsilon$  has a fixed effect on optimization accuracy, independent of  $T$ : approximate maximization has limited effects.

Key to the convergence guarantee of Theorem 2 is that the loss  $\ell$  is smooth in  $z$ : the inner supremum (2b) is NP-hard to compute for non-smooth deep networks (see Lemma 2 in Section B for a proof of this for ReLU's). The smoothness of  $\ell$  is essential so that a penalized version  $\ell(\theta, z) - \gamma c(z, z_0)$  is concave in  $z$  (which can be approximately verified by computing Hessians  $\nabla_{zz}^2 \ell(\theta, z)$  for each training datapoint), allowing computation and our coming certificates of optimality. Replacing ReLU's with sigmoidals or ELU's (Clevert et al., 2015) allows us to apply Theorem 2, making distributionally robust optimization tractable for deep learning.

In supervised learning scenarios, we are often interested in adversarial perturbations only to feature vectors (and not labels). Letting  $Z = (X,Y)$  where  $X$  denotes the feature vector (covariates) and  $Y$  the label, this is equivalent to defining the Wasserstein cost function  $c: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}_+ \cup \{\infty\}$  by

$$
c \left(z, z ^ {\prime}\right) := c _ {x} \left(x, x ^ {\prime}\right) + \infty \cdot \mathbf {1} \left\{y \neq y ^ {\prime} \right\} \tag {8}
$$

where  $c_{x}:\mathcal{X}\times \mathcal{X}\to \mathbb{R}_{+}$  is the transportation cost for the feature vector  $X$ . All of our results suitably generalize to this setting with minor modifications to the robust surrogate (2b) and the above assumptions (see Section D). Similarly, our distributionally robust framework (2) is general enough to consider adversarial perturbations to only an arbitrary subset of coordinates in  $Z$ . For example, it may be appropriate in certain applications to hedge against adversarial perturbations to a small fixed region of an image (Brown et al., 2017). By suitably modifying the cost function  $c(z,z^{\prime})$  to take value  $\infty$  outside this small region, our general formulation covers such variants.

# 3 CERTIFICATE OF ROBUSTNESS AND GENERALIZATION

From results in the previous section, Algorithm 1 provably learns to protect against adversarial perturbations of the form (7) on the training dataset. Now, we show that such procedures generalize, allowing us to prevent attacks on the test set. Our subsequent results hold uniformly over the space of parameters  $\theta \in \Theta$ , including  $\theta_{\mathrm{WRM}}$ , the output of the stochastic gradient descent procedure in Section 2.1. Our first main result, presented in Section 3.1, gives a data-dependent upper bound on the population worst-case objective  $\sup_{P:W_c(P,P_0)\leq \rho}\mathbb{E}_P[\ell (\theta ;Z)]$  for any arbitrary level of robustness  $\rho$ ; this bound is optimal for  $\rho = \widehat{\rho}_n$ , the level of robustness achieved for the empirical distribution by solving (7). Our bound is efficiently computable and hence certifies a level of robustness for the worst-case population objective. Second, we show in Section 3.2 that adversarial perturbations on the training set (in a sense) generalize: solving the empirical penalty problem (7) guarantees a similar level of robustness as directly solving its population counterpart (2).

# 3.1 CERTIFICATE OF ROBUSTNESS

Our main result in this section is a data-dependent upper bound for the worst-case population objective:  $\sup_{P:W_c(P,P_0)\leq \rho}\mathbb{E}_P[\ell (\theta ;Z)]\leq \gamma \rho +\mathbb{E}_{\widehat{P}_n}[\phi_\gamma (\theta ;Z)] + O(1 / \sqrt{n})$  for all  $\theta \in \Theta$ , with high probability. To make this rigorous, fix  $\gamma >0$ , and consider the worst-case perturbation, typically called the transportation map or Monge map (Villani, 2009),

$$
T _ {\gamma} (\theta ; z _ {0}) := \underset {z \in \mathcal {Z}} {\operatorname {a r g m a x}} \left\{\ell (\theta ; z) - \gamma c (z, z _ {0}) \right\}. \tag {9}
$$

Under our assumptions,  $T_{\gamma}$  is easily computable when  $\gamma \geq L_{zz}$ . Letting  $\delta_z$  denote the point mass at  $z$ , Proposition 1 shows the empirical maximizers of the Lagrangian formulation (6) are attained by

$$
P _ {n} ^ {*} (\theta) := \underset {P} {\operatorname {a r g m a x}} \left\{\mathbb {E} _ {P} [ \ell (\theta ; Z) ] - \gamma W _ {c} (P, \widehat {P} _ {n}) \right\} = \frac {1}{n} \sum_ {i = 1} ^ {n} \delta_ {T _ {\gamma} (\theta , Z _ {i})} \quad \text {a n d} \tag {10}
$$

$$
\widehat {\rho} _ {n} (\theta) := W _ {c} (P _ {n} ^ {*} (\theta), \widehat {P} _ {n}) = \mathbb {E} _ {\widehat {P} _ {n}} [ c (T _ {\gamma} (\theta ; Z), Z) ].
$$

Our results imply, in particular, that the empirical worst-case loss  $\mathbb{E}_{P_n^*}[\ell (\theta ;Z)]$  gives a certificate of robustness to (population) Wasserstein perturbations up to level  $\widehat{\rho}_n$ .  $\mathbb{E}_{P_n^* (\theta)}[\ell (\theta ;Z)]$  is efficiently computable via (10), providing a data-dependent guarantee for the worst-case population loss.

Our bound relies on the usual covering numbers for the model class  $\{\ell(\theta; \cdot) : \theta \in \Theta\}$  as the notion of complexity (e.g. van der Vaart & Wellner, 1996), so, despite the infinite-dimensional problem (7), we retain the same uniform convergence guarantees typical of empirical risk minimization. Recall that for a set  $V$ , a collection  $v_1, \ldots, v_N$  is an  $\epsilon$ -cover of  $V$  in norm  $\|\cdot\|$  if for each  $v \in V$ , there exists  $v_i$  such that  $\|v - v_i\| \leq \epsilon$ . The covering number of  $V$  with respect to  $\|\cdot\|$  is

$$
N (V, \epsilon , \| \cdot \|) := \inf  \left\{N \in \mathbb {N} \mid \text {t h e r e i s a n} \epsilon \text {- c o v e r o f} V \text {w i t h r e s p e c t t o} \| \cdot \| \right\}.
$$

For  $\mathcal{F} \coloneqq \{\ell(\theta, \cdot) : \theta \in \Theta\}$  equipped with the  $L^{\infty}(\mathcal{Z})$  norm  $\|f\|_{L^{\infty}(\mathcal{Z})} \coloneqq \sup_{z \in \mathcal{Z}} |f(z)|$ , we state our results in terms of  $\|\cdot\|_{L^{\infty}(\mathcal{Z})}$ -covering numbers of  $\mathcal{F}$ . To ease notation, we let

$$
\epsilon_ {n} (t) := \gamma b _ {1} \sqrt {\frac {M _ {\ell}}{n}} \int_ {0} ^ {1} \sqrt {\log N (\mathcal {F} , M _ {\ell} \epsilon , \| \cdot \| _ {L ^ {\infty} (\mathcal {Z})})} d \epsilon + b _ {2} M _ {\ell} \sqrt {\frac {t}{n}}
$$

where  $b_{1}, b_{2}$  are numerical constants.

We are now ready to state the main result of this section. We first show from the duality result (6) that we can provide an upper bound for the worst-case population performance for any level of robustness  $\rho$ . For  $\rho = \widehat{\rho}_n(\theta)$  and  $\theta = \theta_{\mathrm{WRM}}$ , this certificate is (in a sense) tight as we see below.

Theorem 3. Assume  $|\ell(\theta; z)| \leq M_{\ell}$  for all  $\theta \in \Theta$  and  $z \in \mathcal{Z}$ . Then, for a fixed  $t > 0$  and numerical constants  $b_1, b_2 > 0$ , with probability at least  $1 - e^{-t}$ , simultaneously for all  $\theta \in \Theta$ , and  $\rho \geq 0$ ,

$$
\sup  _ {P: W _ {c} (P, P _ {0}) \leq \rho} \mathbb {E} _ {P} [ \ell (\theta ; Z) ] \leq \gamma \rho + \mathbb {E} _ {\widehat {P} _ {n}} [ \phi_ {\gamma} (\theta ; Z) ] + \epsilon_ {n} (t). \tag {11}
$$

In particular, if  $\rho = \widehat{\rho}_n(\theta)$  then with probability at least  $1 - e^{-t}$ , for all  $\theta \in \Theta$

$$
\begin{array}{l} \sup  _ {P: W _ {c} (P, P _ {0}) \leq \widehat {\rho} _ {n} (\theta)} \mathbb {E} _ {P} [ \ell (\theta ; Z) ] \leq \mathbb {E} _ {\widehat {P} _ {n}} [ \phi_ {\gamma} (\theta ; Z) ] + \gamma \widehat {\rho} _ {n} (\theta) + \epsilon_ {n} (t) \\ = \sup  _ {P: W _ {c} (P, \widehat {P} _ {n}) \leq \widehat {\rho} _ {n} (\theta)} \mathbb {E} _ {P} [ \ell (\theta ; Z) ] + \epsilon_ {n} (t). \tag {12} \\ \end{array}
$$

See Section C.4 for its proof. We now give a concrete variant of Theorem 3 for Lipschitz functions. When  $\Theta$  is finite-dimensional  $(\Theta \subset \mathbb{R}^d)$ , Theorem 3 provides a robustness guarantee scaling linearly with  $d$  despite the infinite-dimensional Wasserstein penalty. Assuming there exist  $\theta_0 \in \Theta$ ,  $M_{\theta_0} < \infty$  such that  $|\ell(\theta_0;z)| \leq M_{\theta_0}$  for all  $z \in \mathcal{Z}$ , we have the following corollary (see proof in Section C.5).

Corollary 1. Let  $\ell (\cdot ;z)$  be  $L$ -Lipschitz with respect to some norm  $\| \cdot \|$  for all  $z\in \mathcal{Z}$ . Assume that  $\Theta \subset \mathbb{R}^d$  satisfies  $\mathrm{diam}(\Theta) = \sup_{\theta ,\theta '\in \Theta}\| \theta -\theta '\| < \infty$ . Then, the bounds (11) and (12) hold with

$$
\epsilon_ {n} (t) = b _ {1} \sqrt {\frac {d (L \mathrm {d i a m} (\Theta) + M _ {\theta_ {0}})}{n}} + b _ {2} (L \mathrm {d i a m} (\Theta) + M _ {\theta_ {0}}) \sqrt {\frac {t}{n}}
$$

for some numerical constants  $b_{1}, b_{2} > 0$ .

A key consequence of the bound (11) is that  $\gamma \rho + \mathbb{E}_{\widehat{P}_n}[\phi_\gamma(\theta;Z)]$  certifies robustness for the worst-case population objective for any  $\rho$  and  $\theta$ . For a given  $\theta$ , this certificate is tightest at the achieved level of robustness  $\widehat{\rho}_n(\theta)$ , as noted in the refined bound (12) which follows from the duality result

$$
\underbrace {\mathbb {E} _ {\widehat {P} _ {n}} \left[ \phi_ {\gamma} (\theta ; Z) \right]} _ {\text {s u r r o g a t e l o s s}} + \underbrace {\gamma \widehat {\rho} _ {n} (\theta)} _ {\text {r o b u s t n e s s}} = \sup  _ {P: W _ {c} (P, \widehat {P} _ {n}) \leq \widehat {\rho} _ {n} (\theta)} \mathbb {E} _ {P} \left[ \ell (\theta ; Z) \right] = \mathbb {E} _ {P _ {n} ^ {*} (\theta)} \left[ \ell (\theta ; Z) \right]. \tag {13}
$$

(See Section C.4 for a proof of these equalities.) We expect  $\theta_{\mathrm{WRM}}$ , the output of Algorithm 1, to be close to the minimizer of the surrogate loss  $\mathbb{E}_{\widehat{P}_n}[\phi_\gamma (\theta ;Z)]$  and therefore have the best guarantees. Most importantly, the certificate (13) is easy to compute via expression (10): as noted in Section 2.1, the mappings  $T(\theta ,Z_{i})$  are efficiently computable for large enough  $\gamma$ , and  $\widehat{\rho}_n = \mathbb{E}_{\widehat{P}_n}[c(T(\theta ,Z),Z)]$ .

# 3.2 GENERALIZATION OF ADVERSARIAL EXAMPLES

We can also show that the level of robustness on the training set generalizes. Our starting point is Lemma 1, which shows that  $T_{\gamma}(\cdot; z)$  is smooth under Assumptions A and B:

$$
\left\| T _ {\gamma} \left(\theta_ {1}; z\right) - T _ {\gamma} \left(\theta_ {2}; z\right) \right\| \leq \frac {L _ {z \theta}}{\left[ \gamma - L _ {z z} \right] _ {+}} \| \theta_ {1} - \theta_ {2} \| \tag {14}
$$

for all  $\theta_{1},\theta_{2}$ , where we recall that  $L_{zz}$  is the Lipschitz constant of  $\nabla_z\ell (\theta ;z)$ . Leveraging this smoothness, we show that  $\widehat{\rho}_n(\theta) = \mathbb{E}_{\widehat{P}_n}[c(T_\gamma (\theta ;Z),Z)]$ , the level of robustness achieved for the empirical problem, concentrates uniformly around its population counterpart.

Theorem 4. Let  $\mathcal{Z} \subset \{z \in \mathbb{R}^m : \|z\| \leq M_z\}$  so that  $\|Z\| \leq M_z$  almost surely and assume either that (i)  $c(\cdot, \cdot)$  is  $L_{\mathbb{C}}$ -Lipschitz over  $\mathcal{Z}$  with respect to the norm  $\|\cdot\|$  in each argument, or (ii) that  $\ell(\theta, z) \in [0, M_\ell]$  and  $z \mapsto \ell(\theta, z)$  is  $\gamma L_{\mathbb{C}}$ -Lipschitz for all  $\theta \in \Theta$ .

If Assumptions  $A$  and  $B$  hold, then with probability at least  $1 - e^{-t}$

$$
\sup  _ {\theta \in \Theta} | \mathbb {E} _ {\widehat {P} _ {n}} [ c (T _ {\gamma} (\theta ; Z), Z) ] - \mathbb {E} _ {P _ {0}} [ c (T _ {\gamma} (\theta ; Z), Z) ] | \leq 4 B \sqrt {\frac {1}{n} \left(t + \log N \left(\Theta , \frac {[ \gamma - L _ {z z} ] _ {+} t}{4 L _ {c} L _ {z \theta}} , \| \cdot \|\right)\right)}. \tag {15}
$$

where  $B = L_{\mathsf{c}}M_{\mathsf{z}}$  under assumption (i) and  $B = M_{\ell} / \gamma$  under assumption (ii).

See Section C.6 for the proof. For  $\Theta \subset \mathbb{R}^d$ , we have  $\log N(\Theta, \epsilon, \| \cdot \|) \leq d \log (1 + \frac{\mathrm{diam}(\Theta)}{\epsilon})$  so that the bound (28) gives the usual  $\sqrt{d / n}$  generalization rate for the distance between adversarial perturbations and natural examples. Another consequence of Theorem 4 is that  $\widehat{\rho}_n(\theta_{\mathrm{WRM}})$  in the certificate (12) is positive as long as the loss  $\ell$  is not completely invariant to data. To see this, note from the optimality conditions for  $T_{\gamma}(\theta; Z)$  that  $\mathbb{E}_{P_0}[c(T_{\gamma}(\theta; Z), Z)] = 0$  iff  $\nabla_z \ell(\theta; z) = 0$  almost surely, and hence for large enough  $n$ , we have  $\widehat{\rho}_n(\theta) > 0$  by the bound (28).

# 4 EXPERIMENTS

Our technique for distributionally robust optimization with adversarial training extends beyond supervised learning. To that end, we present empirical evaluations on supervised and reinforcement learning tasks where we compare performance with empirical risk minimization (ERM) and, where appropriate, models trained with the fast-gradient method (3) (FGM) (Goodfellow et al., 2015), its iterated variant (IFGM) (Kurakin et al., 2016), and the projected-gradient method (PGM) (Madry et al., 2017). PGM augments stochastic gradient steps for the parameter  $\theta$  with projected gradient ascent over  $x\mapsto \ell (\theta ;x,y)$ , iterating (for data point  $x_{i},y_{i}$ )

$$
\Delta x _ {i} ^ {t + 1} (\theta) := \underset {\| \eta \| _ {p} \leq \epsilon} {\operatorname {a r g m a x}} \left\{\nabla_ {x} \ell \left(\theta ; x _ {i} ^ {t}, y _ {i}\right) ^ {T} \eta \right\} \text {a n d} x _ {i} ^ {t + 1} := \Pi_ {\mathcal {B} _ {\epsilon , p} \left(x _ {i} ^ {t}\right)} \left\{x _ {i} ^ {t} + \alpha_ {t} \Delta x _ {i} ^ {t} (\theta) \right\} \tag {16}
$$

for  $t = 1,\ldots ,T_{\mathrm{adv}}$  , where  $\Pi$  denotes projection onto  $\mathcal{B}_{\epsilon ,p}(x_i)\coloneqq \{x:\| x - x_i\| _p\leq \epsilon \}$

The adversarial training literature (e.g. Goodfellow et al. (2015)) usually considers  $\| \cdot \|_{\infty}$ -norm attacks, which allow imperceptible perturbations to all input features. In most scenarios, however, it is reasonable to defend against weaker adversaries that instead perturb influential features more. We consider this setting and train against  $\| \cdot \|_2$ -norm attacks. Namely, we use the squared Euclidean cost for the feature vectors  $c_x(x,x') \coloneqq \| x - x' \|_2^2$  and define the overall cost as the covariate-shift adversary (8) for WRM (Algorithm 1), and we use  $p = 2$  for FGM, IFGM, PGM training in all experiments; we still test against adversarial perturbations with respect to the norms  $p = 2,\infty$ . We use  $T_{\mathrm{adv}} = 15$  iterations for all iterative methods (IFGM, PGM, and WRM) in training and attacks.

In Section 4.1, we visualize differences between our approach and ad-hoc methods to illustrate the benefits of certified robustness. In Section 4.2 we consider a supervised learning problem for MNIST where we adversarially perturb the test data. Finally, we consider a reinforcement learning problem in Section 4.3, where the Markov decision process used for training differs from that for testing.

WRM enjoys the theoretical guarantees of Sections 2 and 3 for large  $\gamma$ , but for small  $\gamma$  (large adversarial budgets), WRM becomes a heuristic like other methods. In Appendix A.4, we compare WRM

![](images/b5edd90ca1901206f30a7ad74253e1ffe878030fb95d2a4ebbb2bdb3779cd51b.jpg)  
(a) ReLU model

![](images/4a6dbfba96279f671b867e514ac9ae5cd1d51cc7379b5f38dabbb6644f3739f5.jpg)  
(b) ELU model  
Figure 1. Experimental results on synthetic data. Training data are shown in blue and red. Classification boundaries are shown in yellow, purple, and green for ERM, FGM, and and WRM respectively. The boundaries are shown with the training data as well as separately with the true class boundaries.

![](images/ea349ef37b95a3402adc134d9a1e127d32455f62de66faffe40f7f462504438d.jpg)

![](images/93766ad722e2944f6b196c96ee2f2136e7e653c78319cb51cb42e9f94fe5f3f0.jpg)

![](images/d5d4513e61fb8a154425764c769cbf84987d9ff189ed90b074555a10bbe7b682.jpg)

with other methods on attacks with large adversarial budgets. In Appendix A.5, we further compare WRM—which is trained to defend against  $\| \cdot \| _2$ -adversaries—with other heuristics trained to defend against  $\| \cdot \|_{\infty}$ -adversaries. WRM matches or outperforms other heuristics against imperceptible attacks, while it underperforms for attacks with large adversarial budgets.

# 4.1 VISUALIZING THE BENEFITS OF CERTIFIED ROBUSTNESS

For our first experiment, we generate synthetic data  $Z = (X,Y) \sim P_0$  by  $X_i \stackrel{\mathrm{id}}{\sim} \mathsf{N}(0_2,I_2)$  with labels  $Y_i = \mathrm{sign}(\|x\|_2 - \sqrt{2})$ , where  $X \in \mathbb{R}^2$  and  $I_2$  is the identity matrix in  $\mathbb{R}^2$ . Furthermore, to create a wide margin separating the classes, we remove data with  $\|X\|_2 \in (\sqrt{2}/1.3, 1.3\sqrt{2})$ . We train a small neural network with 2 hidden layers of size 4 and 2 and either all ReLU or all ELU activations between layers, comparing our approach (WRM) with ERM and the 2-norm FGM. For our approach we use  $\gamma = 2$ , and to make fair comparisons with FGM we use

$$
\epsilon^ {2} = \widehat {\rho} _ {n} \left(\theta_ {\mathrm {W R M}}\right) = W _ {c} \left(P _ {n} ^ {*} \left(\theta_ {\mathrm {W R M}}\right), \widehat {P} _ {n}\right) = \mathbb {E} _ {\widehat {P} _ {n}} \left[ c \left(T \left(\theta_ {\mathrm {W R M}}, Z\right), Z\right) \right], \tag {17}
$$

for the fast-gradient perturbation magnitude  $\epsilon$ , where  $\theta_{\mathrm{WRM}}$  is the output of Algorithm 1.

Figure 1 illustrates the classification boundaries for the three training procedures over the ReLU-activated (Figure 1(a)) and ELU-activated (Figure 1(b)) models. Since  $70\%$  of the data are of the blue class  $(\| X\| _2\leq \sqrt{2} /1.3)$ , distributional robustness favors pushing the classification boundary outwards; intuitively, adversarial examples are most likely to come from pushing blue points outwards across the boundary. ERM and FGM suffer from sensitivities to various regions of the data, as evidenced by the lack of symmetry in their classification boundaries. For both activations, WRM pushes the classification boundaries further outwards than ERM or FGM. However, WRM with ReLU's still suffers from sensitivities (e.g. radial asymmetry in the classification surface) due to the lack of robustness guarantees. WRM with ELU's provides a certified level of robustness, yielding an axisymmetric classification boundary that hedges against adversarial perturbations in all directions.

Recall that our certificates of robustness on the worst-case performance given in Theorem 3 applies for any level of robustness  $\rho$ . In Figure 2 (a), we plot our certificate (11) against the out-of-sample (test) worst-case performance  $\sup_{P:W_c(P,P_0)\leq \rho}\mathbb{E}_P[\ell (\theta ;Z)]$ . Since the worst-case loss is hard to evaluate directly, we solve its Lagrangian relaxation (6) for different values of  $\gamma_{\mathrm{adv}}$ . For each  $\gamma_{\mathrm{adv}}$  we consider the distance to adversarial examples in the test dataset

$$
\widehat {\rho} _ {\text {t e s t}} (\theta) := \mathbb {E} _ {\widehat {P} _ {\text {t e s t}}} [ c (T _ {\gamma_ {\text {a d v}}} (\theta , Z), Z) ], \tag {18}
$$

where  $\widehat{P}_{\mathrm{test}}$  is the test distribution,  $c(z,z') \coloneqq \|x - x'\|_2^2 + \infty \cdot \mathbf{1}\{y \neq y'\}$  as before, and  $T_{\gamma_{\mathrm{adv}}}(\theta, Z) = \operatorname{argmax}_z \{\ell(\theta;z) - \gamma_{\mathrm{adv}} c(z,Z)\}$  is the adversarial perturbation of  $Z$  (Monge map)

![](images/93c6fbd48805fd81bd2bb0d9df928b72e51f6cc78cb16ad6416e93b708fe7399.jpg)  
(a) Synthetic data

![](images/caf50ca3101d1e617a278d72453d95ab5efc7dfba6ff970c9fbd9b20c37e82f5.jpg)  
(b) MNIST  
Figure 2. Empirical comparison between certificate of robustness (11) (blue) and out-of-sample (test) worst-case performance (red) for the experiments with (a) synthetic data and (b) MNIST. The statistical error term  $\epsilon_{n}(t)$  is omitted from the certificate. The vertical bar indicates the achieved level of robustness on the training set  $\widehat{\rho}_{n}(\theta_{\mathrm{WRM}})$ .

for the model  $\theta$ . The worst-case losses on the test dataset are then given by

$$
\mathbb {E} _ {\widehat {P} _ {\text {t e s t}}} \left[ \phi_ {\gamma_ {\text {a d v}}} \left(\theta_ {\text {W R M}}; Z\right) \right] + \gamma_ {\text {a d v}} \widehat {\rho} _ {\text {t e s t}} \left(\theta_ {\text {W R M}}\right) = \sup  _ {P: W _ {c} \left(P, P _ {\text {t e s t}}\right) \leq \widehat {\rho} _ {\text {t e s t}} \left(\theta_ {\text {W R M}}\right)} \mathbb {E} _ {P} \left[ \ell \left(\theta_ {\text {W R M}}; Z\right) \right].
$$

As anticipated, our certificate is almost tight near the achieved level of robustness  $\widehat{\rho}_n(\theta_{\mathrm{WRM}})$  for WRM (10) and provides a performance guarantee even for other values of  $\rho$ .

# 4.2 LEARNING A MORE ROBUST CLASSIFIER

We now consider a standard benchmark—training a neural network classifier on the MNIST dataset. The network consists of  $8 \times 8$ ,  $6 \times 6$ , and  $5 \times 5$  convolutional filter layers with ELU activations followed by a fully connected layer and softmax output. We train WRM with  $\gamma = 0.04\mathbb{E}_{\widehat{P}_n}[\|X\|_2]$  and for the other methods we choose  $\epsilon$  as the level of robustness achieved by WRM (17). In the figures, we scale the budgets  $1/\gamma_{\mathrm{adv}}$  and  $\epsilon_{\mathrm{adv}}$  for the adversary with  $C_p := \mathbb{E}_{\widehat{P}_n}[\|X\|_p]$ .

First, in Figure 2(b) we again illustrate the validity of our certificate of robustness (11) for the worst-case test performance for arbitrary level of robustness  $\rho$ . We see that our certificate provides a performance guarantee for out-of-sample worst-case performance.

We now compare adversarial training techniques. All methods achieve at least  $99\%$  test-set accuracy, implying there is little test-time penalty for the robustness levels  $(\epsilon$  and  $\gamma)$  used for training. It is thus important to distinguish the methods' abilities to combat attacks. We test performance of the five methods (ERM, FGM, IFGM, PGM, WRM) under PGM attacks (16) with respect to 2- and  $\infty$ -norms. In Figure 3(a) and (b), all adversarial methods outperform ERM, and WRM offers more robustness even with respect to these PGM attacks. Training with the Euclidean cost still provides robustness to  $\infty$ -norm fast gradient attacks. We provide further evidence in Appendix A.1.

Next we study stability of the loss surface with respect to perturbations to inputs. We note that small values of  $\widehat{\rho}_{\mathrm{test}}(\theta)$ , the distance to adversarial examples (18) correspond to small magnitudes of  $\nabla_{z}\ell (\theta ;z)$  in a neighborhood of the nominal input, which ensures stability of the model. Figure 4(a) shows that  $\widehat{\rho}_{\mathrm{test}}$  differs by orders of magnitude between the training methods (models  $\theta = \theta_{\mathrm{ERM}},\theta_{\mathrm{FGM}},\theta_{\mathrm{IFGM}},\theta_{\mathrm{PGM}},\theta_{\mathrm{WRM}}$ ); the trend is nearly uniform over all  $\gamma_{\mathrm{adv}}$ , with  $\theta_{\mathrm{WRM}}$  being the most stable. Thus, we see that our adversarial-training method defends against gradient-exploiting attacks by reducing the magnitudes of gradients near the nominal input.

In Figure 4(b) we provide a qualitative picture by adversarially perturbing a single test datapoint until the model misclassifies it. Specifically, we again consider WRM attacks and we decrease  $\gamma_{\mathrm{adv}}$  until each model misclassifies the input. The original label is 8, whereas on the adversarial examples IFGM predicts 2, PGM predicts 0, and the other models predict 3. WRM's "misclassifications"

![](images/367277f00a0b629a46e79f68a9d88f04720c09d9587a5d61cd8cacf8709ad6b6.jpg)  
(a) Test error vs.  $\epsilon_{\mathrm{adv}}$  for  $\| \cdot \| _2$  attack

![](images/8976fefa198f5b9c80fbff8e2301d5329a895cd0008392796db6cb43058ff7cf.jpg)  
(b) Test error vs.  $\epsilon_{\mathrm{adv}}$  for  $\| \cdot \|_{\infty}$  attack

![](images/012b9cce79e2c77fe51cfe66023d0dfddb8f1de6c0e44b8afbec7cb7891835ce.jpg)  
Figure 3. PGM attacks on the MNIST dataset. (a) and (b) show test misclassification error vs. the adversarial perturbation level  $\epsilon_{\mathrm{adv}}$  for the PGM attack with respect to Euclidean and  $\infty$  norms respectively. The vertical bar in (a) indicates the perturbation level used for training the FGM, IFGM, and PGM models as well as the estimated radius  $\sqrt{\widehat{\rho}_n(\theta_{\mathrm{WRM}})}$ . For MNIST,  $C_2 = 9.21$  and  $C_\infty = 1.00$ .  
(a)  $\widehat{\rho}_{test}$  vs.  $1 / \gamma_{\mathrm{adv}}$  
Figure 4. Stability of the loss surface. In (a), we show the average distance of the perturbed distribution  $\widehat{\rho}_{\mathrm{test}}$  for a given  $\gamma_{\mathrm{adv}}$ , an indicator of local stability to inputs for the decision surface. The vertical bar in (a) indicates the  $\gamma$  we use for training WRM. In (b) we visualize the smallest WRM perturbation (largest  $\gamma_{\mathrm{adv}}$ ) necessary to make a model misclassify a datapoint. More examples are in Appendix A.2.

![](images/559b4ac6184c34c310d4bed01e7b3228e3430f634faa7833eb06bcb9df38da97.jpg)  
Original

![](images/b024d346365e1a2c8845326fefda943f1fb96fb9683aa792c0d7f3a8441fd31f.jpg)  
ERM

![](images/911e74bf70f49f29fb9b100ddfd17306bc24f6f553f47abd1ed9f69bb229f4db.jpg)  
FGM

![](images/11a3a77242c04576793e2b7bea8c2d024f7a42984663f2da97051c6b7a469d79.jpg)  
IFGM  
(b) Perturbations on a test datapoint

![](images/7f6d66c44646d32c31c0d9b91ecd68cb2b5f087e99aad929361b5b08522c081d.jpg)  
PGM

![](images/476e97d3fdc5e360f70eaddcc68dd09bc6810d41ed78022a3da60b63c3e09bb0.jpg)  
WRM

appear consistently reasonable to the human eye (see Appendix A.2 for examples of other digits); WRM defends against gradient-based exploits by learning a representation that makes gradients point towards inputs of other classes. Together, Figures 4(a) and (b) depict our method's defense mechanisms to gradient-based attacks: creating a more stable loss surface by reducing the magnitude of gradients and improving their interpretability.

# 4.3 ROBUST MARKOV DECISION PROCESSES

For our final experiments, we consider distributional robustness in the context of Q-learning, a model-free reinforcement learning technique. We consider Markov decision processes (MDPs)  $(\mathcal{S},\mathcal{A},P_{sa},r)$  with state space  $S$ , action space  $\mathcal{A}$ , state-action transition probabilities  $P_{sa}$ , and rewards  $r:\mathcal{S}\to \mathbb{R}$ . The goal of a reinforcement-learning agent is to maximize (discounted) cumulative rewards  $\sum_t\lambda^t\mathbb{E}[r(s^t)]$  (with discount factor  $\lambda$ ); this is analogous to minimizing  $\mathbb{E}_P[\ell (\theta ;Z)]$  in supervised learning. Robust MDP's consider an ambiguity set  $\mathcal{P}_{sa}$  for state-action transitions. The goal is maximizing the worst-case realization  $\inf_{P\in \mathcal{P}_{sa}}\sum_t\lambda^t\mathbb{E}_P[r(s^t)]$ , analogous to problem (1).

In a standard MDP, Q-learning learns a quality function  $Q: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$  via the iterations

$$
Q \left(s ^ {t}, a ^ {t}\right) \leftarrow Q \left(s ^ {t}, a ^ {t}\right) + \alpha_ {t} \left(r \left(s ^ {t}\right) + \lambda \max  _ {a} Q \left(s ^ {t + 1}, a\right) - Q \left(s ^ {t}, a ^ {t}\right)\right) \tag {19}
$$

such that  $\operatorname{argmax}_a Q(s, a)$  is (eventually) the optimal action in state  $s$  to maximize cumulative reward. In scenarios where the underlying environment has a continuous state-space and we represent  $Q$  with a differentiable function (e.g. Mnih et al. (2015)), we can modify the update (19) with an adversarial state perturbation to incorporate distributional robustness. Namely, we draw the nominal

<table><tr><td>Environment</td><td>Regular</td><td>Robust</td></tr><tr><td>Original</td><td>399.7 ± 0.1</td><td>400.0 ± 0.0</td></tr><tr><td colspan="3">Easier environments</td></tr><tr><td>Light</td><td>400.0 ± 0.0</td><td>400.0 ± 0.0</td></tr><tr><td>Long</td><td>400.0 ± 0.0</td><td>400.0 ± 0.0</td></tr><tr><td>Soft g</td><td>400.0 ± 0.0</td><td>400.0 ± 0.0</td></tr><tr><td colspan="3">Harder environments</td></tr><tr><td>Heavy</td><td>150.1 ± 4.7</td><td>334.0 ± 3.7</td></tr><tr><td>Short</td><td>245.2 ± 4.8</td><td>400.0 ± 0.0</td></tr><tr><td>Strong g</td><td>189.8 ± 2.3</td><td>398.5 ± 0.3</td></tr></table>

Table 1. Episode length over 1000 trials (mean ± standard error)

![](images/a19bf507ced5a95623071fe569b12f5740d82146ed08f2306d0270ee9dd8e13c.jpg)  
Figure 5. Episode lengths during training. The environment caps episodes to 400 steps.

state-transition update  $\widehat{s}^{t + 1}\sim p_{sa}(s^t,a^t)$ , and proceed with the update (19) using the perturbation

$$
s ^ {t + 1} \leftarrow \underset {s} {\operatorname {a r g m i n}} \left\{r (s) + \lambda \max  _ {a} Q (s, a) + \gamma c (s, \hat {s} ^ {t + 1}) \right\}. \tag {20}
$$

For large  $\gamma$ , we can again solve problem (20) efficiently using gradient descent. This procedure provides robustness to uncertainties in state-action transitions. For tabular Q-learning, where we represent  $Q$  only over a discretized covering of the underlying state-space, we can either neglect the second term in the update (20) and, after performing the update, round  $s^{t+1}$  as usual, or we can perform minimization directly over the discretized covering. In the former case, since the update (20) simply modifies the state-action transitions (independent of  $Q$ ), standard results on convergence for tabular Q-learning (e.g. Szepesvári & Littman (1999)) apply under these adversarial dynamics.

We test our adversarial training procedure in the classic cart-pole environment, where the goal is to balance a pole on a cart by moving the cart left or right. The environment caps episode lengths to 400 steps and ends the episode prematurely if the pole falls too far from the vertical or the cart translates too far from its origin. We use the reward  $r(\beta) \coloneqq \exp \{-|\beta|\}$ , where  $\beta$  is the angle of the pole from the vertical. Furthermore, we use a tabular representation for  $Q$  with 30 discretized states for  $\beta$  and 15 for its time-derivative  $\hat{\beta}$  (we perform the update (20) without the  $Q$ -dependent term). The action space is binary: push the cart left or right with a fixed force. Due to the nonstationary, policy-dependent radius for the Wasserstein ball, an analogous  $\epsilon$  for the fast-gradient method (or other variants) is not well-defined. Thus, we only compare with an agent trained on the nominal MDP. We test both models with perturbations to the physical parameters. Namely, we shrink/magnify the pole's mass by 2, shrink/magnify the pole's length by 2, and shrink/magnify the strength of gravity  $g$  by 5. The dynamics of the system are such that the heavy, short, and strong-gravity cases are more physically unstable than the original environment, whereas their counterparts are less unstable.

Table 1 shows the performance of the trained models over the original MDP and all of the perturbed MDPs. Both models perform similarly over easier environments, but the robust model greatly outperforms in harder environments. Interestingly, as shown in Figure 4, the robust model also learns more efficiently than the nominal model in the original MDP. We hypothesize that a potential side-effect of robustness is that adversarial perturbations encourage better exploration of the environment.

# 5 CONCLUSION

Explicit distributional robustness of the form (5) is intractable except in limited cases. We provide a method for efficiently guaranteeing distributional robustness with a simple form of adversarial data perturbation. Using only assumptions about the smoothness of the loss function  $\ell$ , we prove that our method enjoys strong statistical guarantees and fast optimization rates for a large class of problems. The NP-hardness of certifying robustness for ReLU networks, coupled with our empirical success and theoretical certificates for smooth networks in deep learning, suggest that using smooth networks may be preferable if we wish to guarantee robustness. Empirical evaluations indicate that our methods are in fact robust to perturbations in the data, and they outperform less-principled adversarial training techniques. The major benefit of our approach is its simplicity and wide applicability across many models and machine-learning scenarios.

# REFERENCES

P. L. Bartlett and S. Mendelson. Rademacher and Gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research, 3:463-482, 2002.  
S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. Vaughan. A theory of learning from different domains. Machine Learning, 79:151-175, 2010.  
A. Ben-Tal, L. E. Ghaoui, and A. Nemirovski. Robust Optimization. Princeton University Press, 2009.  
A. Ben-Tal, D. den Hertog, A. D. Waegenaere, B. Melenberg, and G. Rennen. Robust solutions of optimization problems affected by uncertain probabilities. Management Science, 59(2):341-357, 2013.  
D. Bertsimas, V. Gupta, and N. Kallus. Data-driven robust optimization. arXiv:1401.0212 [math.OC], 2013. URL http://arxiv.org/abs/1401.0212.  
P. Billingsley. Convergence of Probability Measures. Wiley, Second edition, 1999.  
J. Blanchet and K. Murthy. Quantifying distributional model risk via optimal transport. arXiv:1604.01446 [math.PR], 2016.  
J. Blanchet, Y. Kang, and K. Murthy. Robust Wasserstein profile inference and applications to machine learning. arXiv:1610.05627 [math.ST], 2016.  
J. F. Bonnans and A. Shapiro. *Perturbation analysis of optimization problems*. Springer Science & Business Media, 2013.  
S. Boucheron, O. Bousquet, and G. Lugosi. Theory of classification: a survey of some recent advances. *ESAIM: Probability and Statistics*, 9:323-375, 2005.  
S. Boucheron, G. Lugosi, and P. Massart. Concentration Inequalities: a Nonasymptotic Theory of Independence. Oxford University Press, 2013.  
T. Brown, D. Mane, A. Roy, M. Abadi, and J. Gilmer. Adversarial patch. In Machine Learning and Computer Security Workshop, Neural Information Processing Systems, 2017.  
N. Carlini and D. Wagner. Towards evaluating the robustness of neural networks. In Security and Privacy (SP), 2017 IEEE Symposium on, pp. 39-57. IEEE, 2017.  
X. Chen, M. Sim, and P. Sun. A robust optimization perspective on stochastic programming. Operations Research, 55(6):1058-1071, 2007.  
Y. Chen, G. Lan, and Y. Ouyang. Accelerated schemes for a class of variational inequalities. arXiv:1403.4164 [math.OC], 2014.  
D.-A. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289, 2015.  
E. Delage and Y. Ye. Distributionally robust optimization under moment uncertainty with application to data-driven problems. Operations Research, 58(3):595-612, 2010.  
J. C. Duchi, P. W. Glynn, and H. Namkoong. Statistics of robust optimization: A generalized empirical likelihood approach. arXiv:1610.03425 [stat.ML], 2016. URL https://arxiv.org/abs/1610.03425.  
P. M. Esfahani and D. Kuhn. Data-driven distributionally robust optimization using the Wasserstein metric: Performance guarantees and tractable reformulations. arXiv:1505.05116 [math.OC], 2015.  
S. Ghadimi and G. Lan. Stochastic first- and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization, 23(4):2341-2368, 2013.  
J. Goh and M. Sim. Distributionally robust optimization and its tractable approximations. Operations Research, 58(4):902-917, 2010.  
I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. In International Conference on Learning Representations, 2015.  
W. He, J. Wei, X. Chen, N. Carlini, and D. Song. Adversarial example defenses: Ensembles of weak defenses are not strong. arXiv:1706.04701 [cs.LG], 2017.  
X. Huang, M. Kwiatkowska, S. Wang, and M. Wu. Safety verification of deep neural networks. In International Conference on Computer Aided Verification, pp. 3-29. Springer, 2017.  
A. Juditsky, A. Nemirovski, and C. Tauvel. Solving variational inequalities with the stochastic mirror-prox algorithm. Stochastic Systems, 1(1):17-58, 2011.  
G. Katz, C. Barrett, D. Dill, K. Julian, and M. Kochenderfer. Reluplex: An efficient SMT solver for verifying deep neural networks. arXiv:1702.01135 [cs.AI], 2017a.

G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer. Towards proving the adversarial robustness of deep neural networks. arXiv:1709.02802 [cs.LG], 2017b.  
A. Kurakin, I. Goodfellow, and S. Bengio. Adversarial machine learning at scale. arXiv:1611.01236 [cs.CV], 2016.  
H. Lam and E. Zhou. Quantifying input uncertainty in stochastic optimization. In Proceedings of the 2015 Winter Simulation Conference. IEEE, 2015.  
D. Luenberger. Optimization by Vector Space Methods. Wiley, 1969.  
A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. arXiv:1706.06083 [stat.ML], 2017.  
T. Miyato, S.-i. Maeda, M. Koyama, K. Nakae, and S. Ishii. Distributional smoothing with virtual adversarial training. arXiv:1507.00677 [stat.ML], 2015.  
V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Ried-miller, A. K. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.  
S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard. Deepfool: a simple and accurate method to fool deep neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2574-2582, 2016.  
H. Namkoong and J. C. Duchi. Stochastic gradient methods for distributionally robust optimization with  $f$ -divergences. In Advances in Neural Information Processing Systems 29, 2016.  
H. Namkoong and J. C. Duchi. Variance regularization with convex objectives. In Advances in Neural Information Processing Systems 30, 2017.  
A. Nguyen, J. Yosinski, and J. Clune. Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 427-436, 2015.  
N. Papernot, P. McDaniel, I. Goodfellow, S. Jha, Z. B. Celik, and A. Swami. Practical black-box attacks against deep learning systems using adversarial examples. arXiv:1602.02697 [cs.CR], 2016a.  
N. Papernot, P. McDaniel, S. Jha, M. Fredrikson, Z. B. Celik, and A. Swami. The limitations of deep learning in adversarial settings. In Security and Privacy (EuroS&P), 2016 IEEE European Symposium on, pp. 372-387. IEEE, 2016b.  
N. Papernot, P. McDaniel, X. Wu, S. Jha, and A. Swami. Distillation as a defense to adversarial perturbations against deep neural networks. In *Security and Privacy (SP)*, 2016 IEEE Symposium on, pp. 582-597. IEEE, 2016c.  
N. Parikh and S. Boyd. Proximal algorithms. Foundations and Trends in Optimization, 1(3):123-231, 2013.  
N. Ratliff, J. A. Bagnell, and M. Zinkevich. Maximum margin planning. In Proceedings of the 23rd International Conference on Machine Learning, 2006.  
R. T. Rockafellar and R. J. B. Wets. Variational Analysis. Springer, New York, 1998.  
A. Rozsa, M. Gunther, and T. E. Boult. Towards robust deep neural networks with bang. arXiv:1612.00138 [cs.CV], 2016.  
S. Shafieezadeh-Abadeh, P. M. Esfahani, and D. Kuhn. Distributionally robust logistic regression. In Advances in Neural Information Processing Systems, pp. 1576–1584, 2015.  
C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. arXiv:1312.6199 [cs.CV], 2013.  
C. Szepesvári and M. L. Littman. A unified analysis of value-function-based reinforcement-learning algorithms. Neural computation, 11(8):2017-2060, 1999.  
F. Tramer, A. Kurakin, N. Papernot, D. Boneh, and P. McDaniel. Ensemble adversarial training: Attacks and defenses. arXiv:1705.07204 [stat.ML], 2017.  
A. W. van der Vaart and J. A. Wellner. Weak Convergence and Empirical Processes: With Applications to Statistics. Springer, New York, 1996.  
C. Villani. Optimal Transport: Old and New. Springer, 2009.  
H. Xu, C. Caramenis, and S. Mannor. Robustness and regularization of support vector machines. The Journal of Machine Learning Research, 10:1485-1510, 2009.  
H. Xu, C. Caramanis, and S. Mannor. A distributional interpretation of robust optimization. Mathematics of Operations Research, 37(1):95-110, 2012.
