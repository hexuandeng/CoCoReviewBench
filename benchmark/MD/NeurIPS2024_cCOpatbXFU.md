# Investigating Variance Definitions for Stochastic Mirror Descent with Relative Smoothness

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Mirror Descent is a popular algorithm, that extends Gradients Descent (GD) beyond the Euclidean geometry. One of its benefits is to enable strong convergence guarantees through smooth-like analyses, even for objectives with exploding or vanishing curvature. This is achieved through the introduction of the notion of relative smoothness, which holds in many of the common use-cases of Mirror descent. While basic deterministic results extend well to the relative setting, most existing stochastic analyses require additional assumptions on the mirror, such as strong convexity (in the usual sense), to ensure bounded variance. In this work, we revisit Stochastic Mirror Descent (SMD) proofs in the (relatively-strongly-) convex and relatively-smooth setting, and introduce a new (less restrictive) definition of variance which can generally be bounded (globally) under mild regularity assumptions. We then investigate this notion in more details, and show that it naturally leads to strong convergence guarantees for stochastic mirror descent. Finally, we leverage this new analysis to obtain convergence guarantees for the Maximum Likelihood Estimator of a Gaussian with unknown mean and variance.

# 1 Introduction

The central problem of this paper is to solve optimization problems of the following form:

$$
\min  _ {x \in C} f (x), \text {w h e r e} f (x) = \mathbb {E} \left[ f _ {\xi} (x) \right], \tag {1}
$$

where  $C$  is a closed convex subset of  $\mathbb{R}^d$ , and  $f_{\xi}$  are differentiable convex functions (stochasticity is on the variable  $\xi$ ). The problems that we will consider typically arise from machine-learning use-cases, meaning that the dimension  $d$  can be very large. Therefore, first-order methods are popular for solving these problems, since they usually scale well with the dimension.

In standard machine learning setups, computing a gradient of  $f$  is very costly (or even impossible), since it requires computing gradients for all individual examples in the dataset. Yet, gradients of  $f_{\xi}$  are relatively cheap, and arbitrarily high precisions are generally not required. This makes Stochastic Gradient Descent (SGD) the method of choice [4]. Using a step-size  $\eta > 0$ , the SGD update from point  $x \in \mathbb{R}^d$  can be written as  $x_{\mathrm{SGD}}^{+} = \arg \min_{u \in C} \left\{ \eta \nabla f_{\xi}(x)^{\top} u + \frac{1}{2} \| u - x \|^2 \right\}$ .

While the standard Euclidean geometry leading to Gradient Descent (GD) fits many use-cases quite well, several applications are better solved with Mirror Descent (MD), a generalization of GD which allows to better capture the geometry of the problem. For instance, the Kullback-Leibler divergence might be better suited to discriminating between probability distributions than the (squared) Euclidean norm, and this is something that one can leverage using MD with entropy as a mirror. As a matter of fact, many standard algorithms can be interpreted as MD, i.e., as generalized first-order methods. This is for instance the case in statistics, where Expectation Minimization and Maximum A Posteriori

estimators can be interpreted as running MD with specific mirror and step-sizes [15, 17]. Mirror descent can also be used to solve Poisson inverse problems, which have many applications in astronomy and medicine [3], to reduce the communication cost of distributed algorithms [24, 12], or to solve convex quartic problems [6]. In the online learning community as well, many standard algorithms such as Exponential Weight Updates or Follow-The-Regularized-Leader can be interpreted as running mirror descent [21, 13]. There are still many open questions regarding the convergence guarantees for most of the algorithms mentioned above. Therefore, progress on the understanding of MD can lead to a plethora of results on these applications, and more generally to a more consistent theory for Majorization-Minimization algorithms. This paper is a stepping stone in this direction.

Let us now introduce the mirror map, or potential function  $h$ , together with the Bregman divergence with respect to  $h$ , which is defined for  $x, y \in \operatorname{dom} h$  as  $D_h(x, y) = h(x) - h(y) - \nabla h(y)^\top (x - y)$ . We now introduce the Stochastic Mirror Descent (SMD) update, which can be found in its deterministic form in, e.g., Nemirovskij and Yudin [22]. SMD consists in replacing the squared Euclidean norm from the SGD update by the Bregman divergence with respect to the mirror map  $h$ :

$$
x ^ {+} (\eta , \xi) = \arg \min  _ {u \in C} \left\{\eta \nabla f _ {\xi} (x) ^ {\top} u + D _ {h} (u, x) \right\}. \tag {2}
$$

Note that since  $D_{\| \cdot \| ^2}(x,y) = \| x - y\| ^2$ , one can recover SGD by taking  $h = \frac{1}{2}\| \cdot \| ^2$ . In this sense, SMD can be viewed as standard SGD, but changing the way distances are computed, and so the geometry of the problem. Yet, this change significantly complicates the convergence analysis of the method, since the Bregman divergence, in general: (i) does not satisfy the triangular inequality, (ii) is not symmetric, (iii) is not translation-invariant, (iv) is not convex in its second argument.

This means that analyzing mirror descent methods requires quite some care, and that many standard (S)GD results do not extend to the mirror setting. For instance, one can prove that mirror descent cannot be accelerated in general [8]. Similarly, applying techniques such as variance-reduction requires additional assumptions [7]. To ensure that  $x^{+}(\eta ,\xi)$  exists and is unique, we first make the following blanket assumption throughout the paper:

Assumption 1. Function  $h: \mathbb{R}^d \to \mathbb{R} \cup \{\infty\}$  is twice continuously differentiable and strictly convex on  $C$ . For every  $y \in \mathbb{R}^d$ , the problem  $\min_{x \in C} h(x) - x^\top y$  has a unique solution, which lies in  $\operatorname{int} C$  and all  $f_\xi$  are convex.

Note that the regularity assumption on  $h$  could be relaxed, as discussed in Section 3, but we choose a rather strong one to make sure all the objects we will manipulate are well-defined. Interestingly, while mirror descent changes the way distances are computed to move away from the Euclidean geometry, standard analyses of mirror descent methods, and in particular in the online learning community, still require strong convexity and Lipschitz continuity with respect to norms [5, Chapter 4]. It is only recently that a relative smoothness assumption was introduced to study mirror descent [2, 20], together with the corresponding relative strong convexity.

Definition 1. The function  $f$  is said to be  $L$ -relatively smooth and  $\mu$ -relatively strongly convex with respect to  $h$  if for all  $x, y \in C$ :  $\mu D_h(x, y) \leq D_f(x, y) \leq LD_h(x, y)$ . To lighten notation, we will omit the dependence on  $h$  and simply write that  $f$  is  $L$ -rel.-smooth unless clearly specified.

Definition 1 extends the standard smooth and strongly convex assumptions that correspond to the case  $h = \frac{1}{2} \| \cdot \|^{2}$ , so that for all  $x \in C$ ,  $\nabla^2 h(x) = I$  the identity matrix. These assumptions allow MD analyses to generalize standard GD analyses, and in particular to obtain similar linear and sublinear rates, with constant step-size and conditions adapted to the relative assumptions.

While the basic deterministic setting is now well-understood under relative assumptions, a good understanding of the stochastic setting remains elusive. In particular, as we will see in more details in the related work section, all existing proofs somehow require the mirror  $h$  to be globally strongly convex with respect to a norm, or have non-vanishing variance. The only case that can be analyzed tightly is under interpolation (there exists a point that minimizes all stochastic functions), or when using Coordinate Descent instead of SMD [10, 11]. This is a major weakness, as the goal of relative smoothness is precisely to avoid comparisons to norms. Indeed, even when these "absolute" regularity assumptions hold, the smoothness and strong convexity constants are typically very loose, and the theory is not representative of the observed behaviour of the algorithms.

However, as hinted at earlier, this was expected: acceleration is notoriously hard to achieve for mirror descent (and even impossible in general [8]), and variance reduction typically encounters the same

problems [7]. For stochastic updates, this comes from the fact that it is impossible to disentangle the stochastic gradient from the effect of the curvature of  $h$  at the point at which it is applied.

Contribution and outline. The main contribution of this paper is to introduce a new analysis for mirror descent, with a variance notion which is provably bounded under mild regularity assumptions: typically, the same as those required for the deterministic case. We introduce our new variance notion, and compare it with standard ones from the literature in Section 2. This new analysis is both simpler and tighter than existing ones, as shown in Section 3. Finally, we use our results to analyse the convergence of the Maximum Likelihood and Maximum A Posteriori estimators for a Gaussian with unknown mean and variance in Section 4, and show that it is the first generic stochastic mirror descent analysis that obtains meaningful finite-time convergence guarantees in this case.

# 2 Variance Assumptions

We now focus on the various variance assumptions under which Stochastic Mirror Descent is analyzed. Some manipulations require technical lemmas, such as the duality property of the Bregman divergence or the Bregman co-coercivity lemma, which can be found in Appendix A.

We start by introducing our variance definition, prove a few good properties for it, and then compare it with the existing ones to highlight their shortcomings. The two key properties we would like to ensure (and which are not satisfied by other definitions) are: (i) boundedness without strong convexity of  $h$  or restricting the SMD iterates, and (ii) finiteness for  $\eta \to 0$  (with the appropriate scaling).

# 2.1 New variance definition

Let  $\eta > 0$ , and recall that  $x^{+}(\eta, \xi)$  is the result of a SMD step from  $x$  using function  $f_{\xi}$  with step-size  $\eta$  (Equation (2)). From now on, when clear from the context, we will simply denote this point  $x^{+}$ . Yet, although the dependence is now implicit, do keep in mind that  $x^{+}$  is a stochastic quantity that is not independent from  $\xi$  nor  $\eta$ , as this is critical in most results. Under Assumption 1,  $x^{+}$  writes:

$$
\nabla h \left(x ^ {+}\right) = \nabla h (x) - \eta \nabla f _ {\xi} (x). \tag {3}
$$

Similarly, we denote by  $\overline{x^{+}}$  the deterministic Mirror Descent update, which is such that  $\nabla h(\overline{x^{+}}) = \nabla h(x) - \eta \nabla f(x)$ . We also introduce  $h^{*}: y \mapsto \arg \max_{x \in C} x^{\top}y - h(x)$  the convex conjugate of  $h$ , which verifies  $\nabla h^{*}(\nabla h(x)) = x$ . Let us now define the key function

$$
f _ {\eta} (x) = f (x) - \frac {1}{\eta} \mathbb {E} \left[ D _ {h} \left(x, x ^ {+}\right) \right]. \tag {4}
$$

Definition 2. We define the variance of the stochastic mirror descent iterates given by (2) as  $\sigma_{\star, \eta}^2 = \frac{1}{\eta} \sup_{x \in C} (f(x_\star) - f_\eta(x)) = \frac{f^\star - f_\eta^\star}{\eta}$ , where  $f^\star$  and  $f_\eta^\star$  are respectively the inf. of  $f$  and  $f_\eta$ .

We now state various bounds on  $\sigma_{\star ,\eta}^{2}$ , to help understand its behaviour. We start by positivity, which is an essential property that justifies the square in the definition.

Proposition 2.1 (Positivity). For all  $\eta >0$ ,  $\sigma_{\star ,\eta}\geq 0$

This result follows from  $f_{\eta}(x) \leq f(x)$ , since  $D_h(x, x^+) \geq 0$  for all  $x \in C$  by convexity of  $h$ .

Stochastic functions after a step. We first upper bound  $\sigma_{\star ,\eta}^{2}$  directly in terms of  $f_{\xi}$ .

Proposition 2.2. If  $f_{\xi}$  is  $L$ -rel.-smooth and  $\eta \leq 1 / L$ , then  $\sigma_{\star, \eta}^2 \leq \frac{1}{\eta} \left( f(x_{\star}) - \min_{x \in C} \mathbb{E}[f_{\xi}(x^{+})] \right)$ .

Proof. Since  $D_h(x, x^+) = \langle \nabla h(x^+) - \nabla h(x), x^+ - x \rangle - D_h(x^+, x)$ , then  $D_h(x, x^+) = -\eta \nabla f_{\xi}(x)^{\top}(x^+ - x) - D_h(x^+, x) = \eta \left( D_{f_{\xi}}(x^+, x) - f_{\xi}(x^+) + f_{\xi}(x) \right) - D_h(x^+, x)$ . The relative smoothness of  $f_{\xi}$  and the step-size condition imply that  $\eta D_{f_{\xi}}(x^+, x) \leq D_h(x^+, x)$ , leading to  $\frac{1}{\eta} D_h(x, x^+) \leq f_{\xi}(x) - f_{\xi}(x^+)$ , and the result follows.

This bound offers a new point of view on the variance, which can be bounded as the difference between the optimum of  $f$ , and the optimum of a related function, in which we make one mirror descent step before evaluating each  $f_{\xi}$ .

Finiteness. Proposition 2.2 implies the following:

Corollary 2.3. If  $f_{\xi}$  is  $L$ -relatively-smooth w.r.t.  $h$  and admits a minimum  $x_{\star}^{\xi} \in \operatorname{int} C$  a.s., then for all  $\eta \leq 1 / L$ ,  $\sigma_{\star, \eta}^{2} \leq \frac{f(x_{\star}) - \mathbb{E}[f_{\xi}(x_{\star}^{\xi})]}{\eta}$ . In particular,  $\sigma_{\star, \eta}^{2}$  is finite.

This result directly comes from the fact that  $\min_{x\in C}\mathbb{E}\left[f_{\xi}(x^{+})\right]\geq \mathbb{E}\left[\min_{x\in C}f_{\xi}(x^{+})\right]\geq$ $\mathbb{E}\left[f_{\xi}(x_{\star}^{\xi})\right]$ . It shows that the standard regularity assumptions for the convergence of stochastic mirror descent guarantee that the variance as introduced in Definition 2 remains bounded. This is a strong result, that justifies the supremum in the variance definition. Indeed, most other variance definitions require additional assumptions for the variance to remain bounded after the supremum. Instead, we globalize the variance definition, by taking the supremum over the right quantity to ensure that it remains bounded over the whole domain without having to explicitly assume it.

Note that the bound from Corollary 2.3 has already been investigating in other settings for stochastic optimization [19], as discussed in Section 2.2. While useful to show boundedness, this bound has a major drawback, which is that it explodes when the step-size  $\eta$  vanishes. This does not reflect what happens in practice, which is why we investigate finer bounds on  $\sigma_{\star ,\eta}^{2}$ .

Gradient norm at optimum. A usual way of formulating variance is to express it as the norm of the difference between stochastic gradients and the deterministic gradients. While the previous bounds highlight dependencies on the gradient steps (through evaluations at  $x^{+}$ ), none of them really corresponds to "the size of the stochastic gradients at optimum". The key subtlety is that when using mirror descent, it is important to also specify the point at which these gradients are applied, and the following proposition gives a bound of this flavor on  $\sigma_{\star ,\eta}^2$ . In this section,  $x_{\eta}$  denotes the minimizer of  $f_{\eta}$  when it exists and is in int  $C$ . Otherwise, unless explicitly stated, results involving  $x_{\eta}$  can be replaced by a limit for  $x \to x_{\eta}$ .

Proposition 2.4. If  $f$  is  $L$ -rel.-smooth,  $\eta \leq 1 / L$  and  $x_{\star} \in \operatorname{int} C$ ,  $\sigma_{\star,\eta}^2 \leq \frac{1}{\eta^2} \mathbb{E}\left[D_h\left(\overline{x_\eta^+}, x_\eta^+\right)\right]$ .

This can be considered as the Mirror Descent equivalent of  $\mathbb{E}\left[\| \nabla f_{\xi}(x_{\star})\|^{2}\right]$ . Yet, a key difference is that stochastic gradients are evaluated at point  $x_{\eta}$  instead of  $x_{\star}$ , and  $\nabla f(x_{\eta}) \neq 0$  in general.

Proof. For all  $x$ , applying the duality property of the Bregman divergence leads to:

$$
\begin{array}{l} \mathbb {E} \left[ D _ {h} (x, x ^ {+}) \right] = \mathbb {E} \left[ D _ {h ^ {*}} (\nabla h (x ^ {+}), \nabla h (x)) \right] = \mathbb {E} \left[ D _ {h ^ {*}} (\nabla h (x) - \eta \nabla f _ {\xi} (x), \nabla h (x)) \right] \\ = \mathbb {E} \left[ D _ {h ^ {*}} (\nabla h (x) - \eta \nabla f (x), \nabla h (x)) \right] + \mathbb {E} \left[ D _ {h ^ {*}} (\nabla h (x) - \eta \nabla f _ {\xi} (x), \nabla h (x) - \eta \nabla f (x)) \right] \\ = \mathbb {E} \left[ \right. D _ {h ^ {*}} (\nabla h (x) - \eta [ \nabla f (x) - \nabla f (x _ {\star}) ], \nabla h (x)) ] + \mathbb {E} \left[ D _ {h} \left(\overline {{x ^ {+}}}, x ^ {+}\right)\right], \\ \end{array}
$$

where the last equality comes from the Bregman bias-variance decomposition Lemma [23]. We then use the Bregman cocoercivity Lemma [7] to obtain:  $\mathbb{E}[D_h(x,x^+)]\leq \eta D_f(x,x_\star) + \mathbb{E}\left[D_h\left(\overline{x^+},x^+\right)\right]$ . All these technical results can be found in Appendix A. In the end,  $f_{\eta}(x)\geq f(x_{\star}) - \frac{1}{\eta}\mathbb{E}\left[D_h(\overline{x^+},x^+)\right]$ , and this is in particular true for  $x = x_{\eta}$ .

Limit behaviour. A first observation is that both the  $D_{h}(x,x^{+})$  term in the definition of  $f_{\eta}$  and our variance definition are scaled by  $\eta^{-1}$ . Yet, they remain finite when  $\eta \to 0$ . While this is clear in the Euclidean setting, this property holds more generally, as shown in the two following results.

Proposition 2.5. Let  $x \in C$  and  $\eta_0 > 0$  s.t.  $\mathbb{E}D_h(x,x^+(\eta_0,\xi)) < \infty$ . Then,  $f_{\eta}(x) \xrightarrow{\eta \to 0} f(x)$ .

Note that uniform convergence of  $f_{\eta}$  to  $f$  would require that there exists  $\eta > 0$  such that  $\sup_{x \in C} D_h(x, x^+)$  is finite, which we cannot guarantee in general (it does not hold for  $f = g = \frac{1}{2} \| \cdot \|^2$  defined on  $\mathbb{R}^d$  for instance). Denote  $\| x \|_A^2 = x^\top Ax$ , then:

Proposition 2.6 (Small step-sizes limit). If  $f_{\xi}$  are  $L$ -rel.-smooth and  $f$  has a unique minimizer  $x_{\star}$  and for some  $\eta_0 > 0$ ,  $x_{\eta} = \arg \min f_{\eta}(x)$  exists and is in int  $C$  for  $\eta \leq \eta_0$ ,

$$
\lim  _ {\eta \rightarrow 0} \sigma_ {\star , \eta} ^ {2} = \lim  _ {\eta \rightarrow 0} \frac {1}{\eta^ {2}} \mathbb {E} \left[ D _ {h} \left(x _ {\star} ^ {+}, x _ {\star}\right)\right] = \frac {1}{2} \mathbb {E} \left[ \| \nabla f _ {\xi} \left(x _ {\star}\right) \| _ {\nabla^ {2} h \left(x _ {\star}\right) ^ {- 1}} ^ {2} \right]. \tag {5}
$$

This variance is actually the best we can hope for in the Bregman setting, which indicates the relevance of Definition 2. Indeed, this term exactly corresponds to the variance one would obtain when making infinitesimal SMD steps from  $x_{\star}$ , i.e., the norm of the stochastic gradients at optimum in the geometry given by  $\nabla^2 h(x_{\star})^{-1}$ .

# 2.2 Standard Assumptions

We now compare Definition 2 with several variance assumptions from the literature. Note that they typically "only" require the bounds to hold for all iterates over the trajectory. However, in the absence of proof that the iterates stay in certain regions of the space, suprema over the whole domain are required for all variance definitions.

Euclidean case. Let us now take a step back and look at the Euclidean case,  $h = \frac{1}{2} \| \cdot \|^{2}$ , and assume that  $f$  is  $L$ -smooth. Writing Equation (3) with this specific  $h$  and replacing  $x_{\eta}$  by a supremum, we obtain  $\sigma_{\star ,\eta}^2 \leq \sup_{x\in C}\mathbb{E}\left[\frac{1}{2}\| \nabla f(x) - \nabla f_\xi (x)\| ^2\right]$ , which is a common though debatable variance assumption. Indeed, it involves a maximum over the domain, and is in particular not bounded in general even for simple examples like Linear Regression. Yet, we can recover another standard variance assumption by assuming the smoothness of all  $f_{\xi}$  [9], which writes  $\sigma_{\star ,\eta}^2 \leq \mathbb{E}\left[\| \nabla f_\xi (x_\star)\| ^2\right]$ .

This result is obtained by writing that  $\| \nabla f_{\xi}(x)\|^{2}\leq 2\| \nabla f_{\xi}(x) - \nabla f_{\xi}(x_{\star})\|^{2} + 2\| \nabla f_{\xi}(x_{\star})\|^{2}$ , and bounding the first term using smoothness. In particular, we see that standard Euclidean variance definitions are natural bounds of  $\sigma_{\star ,\eta}^2$ . Detailed derivations can be found in Appendix B.

Divergence between stochastic and deterministic gradients. An early variance definition for SMD in the relative setting comes from Hanzely and Richtárik [10], who define  $\sigma_{\mathrm{sym}}^2$  as:

$$
\sigma_ {\mathrm {s y m}} ^ {2} = \frac {1}{\eta} \sup  _ {x \in C} \mathbb {E} \left[ \left\langle \nabla f (x) - \nabla f _ {\xi} (x), x ^ {+} - \overline {{x ^ {+}}} \right\rangle \right] = \frac {1}{\eta^ {2}} \sup  _ {x \in C} \mathbb {E} \left[ D _ {h} \left(x ^ {+}, \overline {{x ^ {+}}}\right) + D _ {h} \left(\overline {{x ^ {+}}}, x ^ {+}\right) \right],
$$

where we recall that  $\overline{x^{+}}$  is such that  $\nabla h(\overline{x^{+}}) = \nabla h(x) - \eta \nabla f(x)$ . We remark two main things when comparing  $\sigma_{\mathrm{sym}}^2$  with Proposition 2.4: (i)  $\sigma_{\star ,\eta}^2$  is not symmetrized, and contains only one of the two terms, and (ii) the bound only needs to hold at  $x_{\eta}$  instead of for all  $x\in C$ . As a result, we directly obtain that  $\sigma_{\star ,\eta}^2\leq \sigma_{\mathrm{sym}}^2$ , and  $\sigma_{\mathrm{sym}}^2$  is actually infinite in most cases, whereas  $\sigma_{\star ,\eta}^2$  is usually finite, as seen above.

Stochastic gradients at optimum. Dragomir et al. [7] define the variance as:

$$
\sigma_ {D E H} ^ {2} = \sup _ {x \in C} \frac {1}{2 \eta^ {2}} \mathbb {E} \left[ D _ {h ^ {*}} (\nabla h (x) - 2 \eta \nabla f _ {\xi} (x _ {\star}), \nabla h (x)) \right] = \sup _ {x \in C} \mathbb {E} \left[ \| \nabla f _ {\xi} (x _ {\star}) \| _ {\nabla^ {2} h ^ {*} (z (x))} ^ {2} \right],
$$

where  $z(x) \in [\nabla h(x), \nabla h(x) - \eta \nabla f_{\xi}(x_{\star})]$ . The main interest of this definition is that stochastic gradients are only taken at  $x_{\star}$ . In particular, this variance is 0 in case there is interpolation (all stochastic functions share a common minimum). However, this quantity can blow up if  $h$  is not strongly convex, since in this case  $\nabla^2 h^*$  is not upper bounded (indeed, smoothness of the conjugate is ensured by strong convexity of the primal function [14]). Following similar derivations, but after the supremum has been taken, we arrive at:

Proposition 2.7. If  $f$  is  $L$ -relatively-smooth w.r.t.  $h$ , then for  $\eta < 1/(2L)$  and some  $z_{\eta} \in [\nabla h(x_{\eta}), \nabla h(x_{\eta}) - \eta \nabla f_{\xi}(x_{\star})]$ , the variance can be bounded as  $\sigma_{\star, \eta}^2 \leq \mathbb{E}\left[\| \nabla f_{\xi}(x_{\star}) \|_{\nabla^2 h^{*}(z_{\eta})}^2\right]$ .

In particular, we obtain a finite bound without having to restrict the space.

Functions variance. Another variance definition that appears in the SGD literature is of the form  $f(x_{\star}) - \mathbb{E}\left[f_{\xi}(x_{\star}^{\xi})\right]$ , using the optima of the stochastic functions [19]. Unfortunately, the results derived with this definition do not obtain a vanishing variance term when  $\eta \to 0$ , unlike most other variance definitions, and contrary to what is observed in practice, that smaller step-sizes reduce the variance. The vanishing variance term can be obtained by rescaling by  $1 / \eta$  (so considering  $\left(f(x_{\star}) - \mathbb{E}\left[f_{\xi}(x_{\star}^{\xi})\right]\right) / \eta$  instead), but this variance definition would explode for  $\eta \to 0$ . This is because using such a definition would come down to performing the supremum step within the expectation from Proposition 2.2, using that  $f_{\xi}(x^{+}) \geq f_{\xi}(x_{\star}^{\xi})$ , which is a very crude bound. Instead, Corollary 2.3 directly shows that our variance definition is tighter than this one, and in particular (i) it is bounded for all  $\eta > 0$ , (ii) it remains finite as  $\eta \to 0$  even with the proper rescaling (Proposition 2.6).

Relation to  $c$ -transform. Mirror descent can be viewed as an alternate minimization method on transforms of  $f$  [18]. This point of view subsumes many methods, including the Newton Method or Mirror Descent. Central to their analysis is the notion of  $c$ -transform  $f^c(y) = \sup_{x \in C} f(x) - c(x, y)$ , a standard quantity from optimal transport [25]. It turns out that for  $\eta \leq 1/L$ ,  $f_\eta$  is actually linked to the  $c$ -transform as  $f_\eta(x) = \mathbb{E}\left[f_\xi^c(x^+)\right]$ , where we use the cost  $c(x, y) = \frac{1}{\eta} D_h(x, y)$ . Since  $f(x_\star) = f^c(x_\star) = \arg \min_{x \in C} f(x^+)$ , denoting  $\mathcal{T}_c(g) = g^c(\nabla h^*(\nabla h(x) - \eta \nabla g(x)))$ , we have that  $\sigma_{\star, \eta}^2 = \frac{1}{\eta} (\min_{x \in C} \mathcal{T}_c(\mathbb{E}[f_\xi])(x) - \min_{x \in C} \mathbb{E}[\mathcal{T}_c(f_\xi)](x))$ . We recognize the structure of a variance, as the difference between an operator applied to the expectation of a random variable, and the expectation of the operator applied to the random variable. Yet, compared to standard (Euclidean) analyses of SGD, it does not simply correspond to the variance of the stochastic gradients (at optimum), and bears a more complex form.

In this section, we have highlighted the connections with other definitions, and argued that  $f_{\eta}$  (and its minimum) is a relevant quantity. In particular, Definition 2 is the only definition that allows boundedness of the variance notion both after a supremum step over the iterates (and without strong convexity of  $h$ ) and in the  $\eta \to 0$  limit with the proper rescaling.

# 3 Convergence Analysis

Now that we have (extensively) investigated  $\sigma_{\star,\eta}^2$ , and the various interpretations that come from different bounds, we are ready to state the convergence results. Some proofs in this section are just sketched, but complete derivations can be found in Appendix C.

# 3.1 Relatively Strongly Convex setting.

Recall that  $f_{\eta}^{\star} = \inf_{x \in C} f_{\eta}(x)$ . Starting from an arbitrary  $x^{(0)}$ , the sequence  $(x^{(k)})_{k \geq 0}$  is built as  $x^{(k+1)} = (x^{(k)})^{+}$  for  $k \in \{0, T\}$  for some  $T > 0$

Theorem 3.1. If  $f$  is  $\mu$ -relatively-strongly-convex with respect to  $h$ , under a constant step-size  $\eta$ , the iterates obtained by SMD (Equation (3)) verify

$$
\eta \left[ \mathbb {E} \left[ f _ {\eta} \left(x ^ {(T)}\right) \right] - f _ {\eta} ^ {\star} \right] + \mathbb {E} \left[ D _ {h} \left(x _ {\star}, x ^ {(T + 1)}\right) \right] \leq (1 - \eta \mu) ^ {T + 1} D _ {h} \left(x _ {\star}, x ^ {(0)}\right) + \frac {\eta \sigma_ {\star , \eta} ^ {2}}{\mu}. \tag {6}
$$

Note that the (relatively) strongly-convex theorem has a standard form, and recovers usual MD results if we remove the variance, and standard SGD results if we take  $h = \frac{1}{2} \| \cdot \| ^2$ .

Proof of Theorem 3.1. We start from a variation of Dragomir et al. [7, Lemma 4]:

$$
\begin{array}{l} \mathbb {E} \left[ D _ {h} \left(x _ {\star}, x ^ {+}\right) \right] - D _ {h} \left(x _ {\star}, x\right) + \eta D _ {f} \left(x _ {\star}, x\right) = - \eta \left[ f (x) - f \left(x _ {\star}\right) \right] + \mathbb {E} \left[ D _ {h} \left(x, x ^ {+}\right) \right] (7) \\ = \eta \left[ f \left(x _ {\star}\right) - \left(f (x) - \frac {1}{\eta} \mathbb {E} \left[ D _ {h} \left(x, x ^ {+}\right) \right]\right) \right] = \eta \left[ f \left(x _ {\star}\right) - f _ {\eta} (x) \right] (8) \\ = - \eta \left[ f _ {\eta} (x) - f _ {\eta} ^ {\star} \right] + \eta \left[ f \left(x _ {\star}\right) - f _ {\eta} ^ {\star} \right]. (9) \\ \end{array}
$$

Using that  $D_{f}(x_{\star},x)\geq \mu D_{h}(x_{\star},x)$  , and remarking that  $f(x_{\star}) - f_{\eta}^{\star} = \eta \sigma_{\star ,\eta}^{2}$  , we obtain:

$$
\eta \left[ f _ {\eta} (x) - f _ {\eta} ^ {\star} \right] + \mathbb {E} \left[ D _ {h} \left(x _ {\star}, x ^ {+}\right) \right] \leq (1 - \eta \mu) D _ {h} \left(x _ {\star}, x\right) + \eta^ {2} \sigma_ {\star , \eta} ^ {2}. \tag {10}
$$

At this point, we can neglect the  $\eta\left[f_{\eta}(x) - f_{\eta}^{\star}\right] \geq 0$  terms and chain the inequalities for  $x = x^{(t)}$  for  $t$  from 0 to  $T$  to obtain the result.

This proof is quite simple, and naturally follows from Lemma C.1. One can also note that relative smoothness of  $f$  is not required to obtain Theorem 3.1, which has no condition on the step-size. This is not a typo, but reflects the fact that step-size conditions are needed to obtain a bounded variance. Indeed, the variance as defined here entangles aspects tied with the error due to discretization (which is usually dealt with using smoothness), and the error due to stochasticity. This is natural, as the stochastic noise vanishes in the continuous limit ( $\eta \to 0$ ). Besides, the magnitude of the updates depends both on where the stochastic gradient is applied and on the step-size. Yet, the simplicity of

the proof is partly due to this entanglement, meaning that we have deferred some of the complexity to the bounding of the variance term.

Also note that Theorem 3.1 uses constant step-sizes, but Equation (10) can be used with time-varying step-sizes, as is done for instance in the proof of Theorem 4.3. A variant of Theorem 3.1 in which the discretization error is partly removed from the notion of variance writes:

Corollary 3.2. Let  $f$  be  $\mu$ -strongly-convex and  $L$ -relatively-smooth with respect to  $h$ , and  $f_{+}^{\star} = \inf_{x\in C}\mathbb{E}\left[f_{\xi}(x^{+})\right]$ . If  $\eta \leq 1 / L$ , the SMD iterates (Equation (3)) with constant step-size  $\eta$  verify

$$
\eta \left[ \mathbb {E} \left[ f _ {\xi} ((x ^ {(T)}) ^ {+}) \right] - f _ {+} ^ {\star} \right] + \mathbb {E} \left[ D _ {h} (x _ {\star}, x ^ {(T + 1)}) \right] \leq (1 - \eta \mu) ^ {T + 1} D _ {h} (x _ {\star}, x ^ {(0)}) + \frac {\eta}{\mu} \left[ \frac {f (x _ {\star}) - f _ {+} ^ {\star}}{\eta} \right].
$$

This alternate version is obtained using that  $f_{\eta}(x) \geq \mathbb{E}\left[f_{\xi}(x^{+})\right]$ , a key step from the proof of Proposition 2.2 (see (8)). In the deterministic case,  $f_{+}^{\star} = f(x_{\star})$ , and we recover standard results.

# 3.2 Convex setting.

Let us now consider the convex case, meaning that  $\mu = 0$ .

Theorem 3.3. If  $f$  is convex, the iterates obtained by SMD using a constant step-size  $\eta > 0$  verify

$$
\frac {1}{T + 1} \sum_ {k = 0} ^ {T} \mathbb {E} \left[ f _ {\eta} \left(x ^ {(k)}\right) - f _ {\eta} ^ {\star} + D _ {f} \left(x _ {\star}, x ^ {(k)}\right) \right] \leq \frac {D _ {h} \left(x _ {\star} , x ^ {(0)}\right)}{\eta (T + 1)} + \eta \sigma_ {\star , \eta} ^ {2}. \tag {11}
$$

This theorem is obtained by summing Equation (9) for  $x = x^{(k)}$  for all  $k \in \{1, \dots, T\}$  and rearranging the terms. Note that varying step-size results can be obtained in the same way.

This case differs from standard convex analyses, in that we obtain a control on  $f_{\eta}(x^{(k)}) - f_{\eta}^{\star} + D_{f}(x_{\star},x^{(k)})$  instead of the usual  $f(x^{(k)}) - f(x_{\star})$ . One of the main consequences is that we cannot get a control on the average iterate since Bregman divergences are in general not convex in their second argument, and  $f_{\eta}$  is not necessarily convex. This non-standard result is a direct consequence of our choice of variance definition, but it is actually a quantity that naturally arises in the analysis. Note that a variant involving  $f_{+}^{\star}$  can be obtained in the same lines as Corollary 3.2.

Controlling  $f_{\eta}$ . The results in this section do not directly control the function gap  $f(x) - f^{*}$ , but rather the transformed one  $f_{\eta}(x) - f_{\eta}^{\star}$ . Yet, the continuity result (in  $\eta$ ) from Proposition 2.5 shows that the bounds we provide can still be interpreted as relevant function values for small  $\eta$ .

**Controlling  $D_{f}(x_{\star}, x^{(k)})$ .** An interesting property of  $D_{f}(x_{\star}, x^{(k)})$  is that it can be linked with the size of the gradients of  $f$ , as shown by the following result.

Proposition 3.4. If  $\nabla f(x_{\star}) = 0$  and  $f$  is  $L$ -relatively smooth with respect to  $h$  then for all  $x \neq x_{\star}$ ,  $D_{f}(x_{\star}, x) \geq L D_{h^{\star}} \left( \nabla h(x_{\star}) + \frac{\nabla f(x)}{L}, \nabla h(x_{\star}) \right) > 0$ .

This is a Bregman equivalent of controlling the gradient squared norm, with the additional benefit that the reference point at which we apply the gradient is the optimum  $x_{\star}$ . Besides, Proposition 3.4 shows that  $D_{f}(x_{\star},x) > 0$  for  $x \neq x_{\star}$  without requiring  $f$  to be strictly convex (only  $h$ ).

Minimal assumptions on  $h$ . Note that the theorems in this section do not actually require  $h$  to satisfy Assumption 1, but only that iterations can be written in the form of Equation 3 (which is guaranteed by Assumption 1). While Assumption 1 allows for instance to use the Bregman cocoercivity lemma with any points, or ensures that  $\nabla^2 h$  is well-defined, which we leverage extensively in Section 2, our theorems are much more general than this, and include applications such as proximal gradient mirror descent (next remark) or the MAP for Gaussian Parameters Estimation (next section).

Stochastic Mirror Descent with a Proximal term. Note that our results can be directly extended to handle a proximal term (similarly to the Euclidean proximal gradient algorithm), to handle composite objectives of the form  $f + g$  (and in particular projections, for cases in which  $g$  is the indicator of a convex set). More details can be found in Appendix E.

# 4 MAP For Gaussian Parameters Estimation.

So far, we have proposed new variance definitions for the analysis of stochastic mirror descent, and we have shown that they compare favorably to existing ones, while leading to simple convergence proofs. In this section, we investigate the open problem formulated by Le Priol et al. [17], which is to find non-asymptotic convergence guarantees for the KL-divergence of the Maximum A Posteriori (MAP) estimator. In particular, this example highlights the relevance of the infimum step on  $f_{\eta}$ , since it gives the first generic analysis that obtains meaningful finite time convergence rates.

# 4.1 MAP and MLE of exponential families.

We now rapidly review the formalism of exponential families. More details can be found in Le Priol et al. [17], and Wainwright et al. [26, Chapter 3]. Let  $X$  be a random variable, and  $T$  a deterministic function, then the density of an exponential family for a sample  $x$  writes  $p_{\theta}(x) = p(x|\theta) = \exp (\langle \theta ,T(x)\rangle -A(\theta))$ , where  $A$  is often referred to as the log-partition function. In this case,  $\theta$  is called the natural parameter, and  $T$  is the sufficient statistic. Function  $A$  is convex, and we can thus establish a form of duality through convex conjugacy. The entropy writes  $A^{*}(\mu) = \max_{\theta^{\prime}\in \Theta}\langle \mu ,\theta^{\prime}\rangle -A(\theta^{\prime})$ . Parameter  $\mu$  is called the mean parameter, and the standard MAP estimator can be derived for  $n_0\in \mathbb{N}$ ,  $\mu_0\in \mathbb{R}$  as  $\mu_{\mathrm{MAP}}^{(n)} = \frac{n_0\mu^{(0)} + \sum_{i = 1}^n T(X_i)}{n_0 + n}$ . The Maximum Likelihood Estimator (MLE) corresponds to taking  $n_0 = 0$ . An interesting observation is that  $\mu_{\mathrm{MAP}}^{(n)}$  can be obtained recursively for  $n > 0$ , as  $\mu_{\mathrm{MAP}}^{(0)} = \mu^{(0)},\eta_n = (n + n_0)^{-1},\mu_{\mathrm{MAP}}^{(n + 1)} = \mu_{\mathrm{MAP}}^{(n)} - \eta_n\nabla g_{X_n}(\mu_{\mathrm{MAP}}^{(n)})$ , with  $\nabla g_{X_n}(\mu) = \mu -T(X_n)$ . In terms of primal variable  $\theta^{(n)} = \nabla A^{*}(\mu_{\mathrm{MAP}}^{(n)})$ , the MAP writes:

$$
\nabla A \left(\theta^ {(n + 1)}\right) = \nabla A \left(\theta^ {(n)}\right) - \eta \nabla f _ {X _ {n}} \left(\theta^ {(n)}\right), \tag {12}
$$

where  $f_{X_n}(\theta) = A(\theta) - \langle \theta, T(X_n) \rangle$ , so that  $f(\theta) = A(\theta) - \langle \theta, \mu_\star \rangle$ . We recognize stochastic mirror descent iterations, with mirror  $A$  and stochastic gradients  $\nabla f_X$ . Similar results on the MLE can be obtained by taking  $n_0 = 0$ . This key observation implies that convergence guarantees on the MAP and the MLE can be deduced from stochastic mirror descent convergence guarantees.

While this appears as an appealing way to obtain convergence guarantees for the MAP, Le Priol et al. [17] observe that none of the existing SMD results obtain meaningful rates for the convergence of the MAP for general exponential families. In particular, none of them recover the  $O(1/n)$  asymptotic convergence rate for estimating a Gaussian with unknown mean and covariance.

This is due to the variance definitions used in the existing analyses, that all have issues (not uniformly bounded over the domain, not decreasing with the step-size...) as discussed in Section 2. Our analysis fixes this problem, and thus yields finite-time guarantees for the MAP estimator for the estimation of a Gaussian with unknown mean and covariance. This shows the relevance of Assumption 2.

# 4.2 Full Gaussian (unknown mean and covariance)

The main problem studied in Le Priol et al. [17] is that of the one-dimensional full-Gaussian case, where the goal is to estimate the mean and covariance of a Gaussian from i.i.d. samples  $X_{1},\ldots ,X_{n}\sim \mathcal{N}(m_{\star},\Sigma_{\star})$  with  $\Sigma_{\star} > 0$ . Note that although notation  $\boldsymbol{\Sigma}$  is usually reserved for the covariance matrix of a multivariate Gaussian, we use it for a scalar value here to highlight the distinction with  $\sigma_{\star ,\eta}^{2}$ , the variance from stochastic mirror descent. In this case, the sufficient statistics write  $T(X) = (X,X^2)$ , and the log-partition and entropy functions are, up to constants,  $A(\theta) = \frac{\theta_1^2}{-4\theta_2} -\frac{1}{2}\log (-\theta_2)$ ,  $A^{*}(\mu) = -\frac{1}{2}\log (\mu_{2} - \mu_{1}^{2})$ , for  $\theta \in \Theta = \mathbb{R}\times \mathbb{R}_{-}^{*}$  and  $\mu \in \{(u,v),u^2 < v\}$ . The goal is to estimate  $D_A(\theta ,\theta_\star)$ , for which Le Priol et al. [17] show that only partial solutions exist: results are either asymptotic, or rely on the objective being (approximately) quadratic. Note that there is a relationship between natural parameters, mean parameters, and  $(m,\Sigma^2)$ , the mean and covariance of the Gaussian we would like to estimate. In the following, we will often abuse notations, and write for instance  $D_{A}(\tilde{\theta},\theta)$  in terms of  $(m,\Sigma^2)$  and  $(\tilde{m},\tilde{\Sigma}^2)$  rather than  $\theta$  and  $\tilde{\theta}$ . We now state a few results, for which detailed derivations can be found in Appendix F. More specifically:

$$
D _ {A} (\tilde {\theta}, \theta) = - \frac {1}{2} \log \left(\frac {\Sigma^ {2}}{\tilde {\Sigma} ^ {2}}\right) - \frac {\tilde {\Sigma} ^ {2} - \Sigma^ {2}}{2 \tilde {\Sigma} ^ {2}} + \frac {(\tilde {m} - m) ^ {2}}{2 \tilde {\Sigma} ^ {2}}.
$$

The update formulas for the parameters are given by:

$$
m ^ {+} = (1 - \eta) m + \eta X, \quad \left(\Sigma^ {2}\right) ^ {+} = (1 - \eta) \left[ \Sigma^ {2} + \eta (m - X) ^ {2} \right]. \tag {13}
$$

Therefore, MAP iterations are well-defined although  $A$  does not verify Assumption 1.

Proposition 4.1. The iterations (12) are well-defined for  $\eta < 1$  in the sense that if  $\theta^{(n)}\in \Theta = \mathbb{R}\times \mathbb{R}_{-}^{*}$ , then  $\nabla A(\theta^{(n)}) - \eta \nabla f_{X_n}(\theta^{(n)})\in \mathrm{Range}(\nabla A)$  almost surely, so that  $\theta^{(n + 1)}\in \Theta$  is well-defined almost surely. Besides,  $f_{\xi}$  is 1-relatively-smooth and 1-relatively-strongly-convex with respect to  $A$ .

This result is a direct consequence of the fact that  $D_{f_{\xi}} = D_{f} = D_{A}$  for all  $\xi$ , and the fact that  $\nabla A(\theta) - \eta \nabla f_{X_n}(\theta) = (1 - \eta)\nabla A(\theta) + \eta T(X_n) \in \{(u,v), u^2 < v\}$  if  $\nabla A(\theta) \in \{(u,v), u^2 < v\}$ . Proposition 4.1 means that we can apply Theorem 3.1, so the next step is to bound the variance  $\sigma_{\star,\eta}^{2}$ .

$$
\left. \right. f _ {\eta} (\theta) - f \left(\theta_ {\star}\right) = \frac {1}{2 \eta} \mathbb {E} \left[ \log \left((1 - \eta) \left(1 + \eta \frac {(m - X) ^ {2}}{\Sigma^ {2}}\right)\right)\right] - \frac {1}{2} \log \left(\frac {\Sigma_ {\star} ^ {2}}{\Sigma^ {2}}\right). \tag {14}
$$

We now use this expression to to lower bound  $f_{\eta}^{\star}$  and so upper bound  $\sigma_{\star ,\eta}^{2}$ .

Lemma 4.2. Let  $(m_{\eta}, \Sigma_{\eta}^{2})$  be the minimizer of  $f_{\eta}$ . Then, for  $\eta < 1/3$ ,  $m_{\eta} = m_{\star}$ ,  $\Sigma_{\star}^{2} \geq \Sigma_{\eta}^{2} \geq (1 - 3\eta)\Sigma_{\star}^{2}$ . In particular, the variance  $\sigma_{\star,\eta}^{2}$  verifies  $\sigma_{\star,\eta}^{2} \leq -\frac{1}{2\eta}\log(1 - 3\eta)$ . For  $1/3 < \eta \leq 1 - \varepsilon$ ,  $\sigma_{\star,\eta}^{2} \leq c_{\varepsilon}$ , where  $c_{\varepsilon}$  is a numerical constant that only depends on  $\varepsilon$ .

Note that we show in this example that  $\Sigma_{\eta}^{2}$  is arbitrarily close to  $\Sigma_{\star}^{2}$  as  $\eta \to 0$ , which is expected.

Theorem 4.3. Let  $\Gamma \geq 0$  be a numerical constant and  $\Gamma = 0$  if  $n_0 > 3$ . The MAP estimator satisfies:

$$
\mathbb {E} \left[ D _ {A} \left(\theta_ {\star}, \theta^ {(n)}\right) \right] \leq \frac {n _ {0} D _ {A} \left(\theta_ {\star} , \theta^ {(0)}\right) + \frac {3}{2} \log \left(1 + \frac {n + 1}{n _ {0}}\right) + \Gamma}{n + n _ {0}}.
$$

Numerical constants are not optimized. Theorem 4.3 gives an anytime result on the convergence of the MAP estimator for all  $n \geq 0, n_0 \geq 1$  directly from the general SMD convergence theorem. Yet, the open problem from Le Priol et al. [17] is not completely solved still, as discussed below.

Reverse KL bound. We obtain a bound on  $D_A(\theta_\star, \theta^{(n)})$ , instead of  $D_A(\theta^{(n)}, \theta_\star) = f(\theta) - f(\theta_\star)$ .  $D_A(\theta^{(n)}, \theta_\star)$  can be controlled asymptotically thanks to the bound on  $f_\eta(\theta^{(n)}) - f_\eta(\theta_\eta)$ , and  $f_\eta \to f$  when  $\eta = 1/n \to 0$ , but we might also be able to exploit this control over the course of the iterations.

Asymptotic convergence. Theorem 4.3 leads to a  $O(\log n / n)$  asymptotic convergence rate instead of the expected  $O(1 / n)$  [17]. This indicates that the  $f_{\eta_n}(\theta^{(n)}) - f_{\eta_n}^*$  terms should not be neglected. Indeed,  $\theta^{(n)}$  actually has a lot of structure in this example, since  $\nabla A(\theta^{(n)}) = \frac{1}{n}\sum_{k = 1}^{n}T(X_k)$ . The SMD analysis is oblivious to this structure, hence the gap. Note that we can get rid of the  $\log n$  factor and recover the right  $O(1 / n)$  rate from the same analysis by using a slightly different estimator than the MAP (or MLE). This is done by setting the step-size as  $\eta_{n} = \frac{2}{n + 1}$  for  $n > 1$ , and the analysis of this variant follows Lacoste-Julien et al. [16], as detailed in Appendix F.3.

The special case of the MLE. The MLE corresponds to  $n_0 = 0$ , which is not handled in our analysis since the first step corresponds to  $\eta = 1$ , which necessarily results in  $\theta_2^{(1)} = -\infty$  (which corresponds to  $\Sigma^2 = 0$ , as can be seen from (13)). If we consider that mirror descent is run from  $\theta^{(1)}$ , then we obtain  $\mathbb{E}\left[D_A(\theta_\star, \theta^{(1)})\right] = \infty$  in general, where the expectation is over the value of the first sample drawn. Therefore, we need to start the SMD analysis at  $\theta^{(2)}$  to fit the MLE into this framework, and in particular we need to be able to evaluate  $\mathbb{E}\left[D_A(\theta_\star, \theta^{(2)})\right]$ . This is further discussed in Appendix F.4.

# 5 Conclusion

This paper introduces a new notion of variance for the analysis of stochastic mirror descent. This notion, based on the fact that a certain function  $f_{\eta}$  admits a minimum, is less restrictive than existing ones, has the right asymptotic scaling with the step-size and is bounded regardless of the trajectory of the iterates without further assumptions.

We strongly believe that our analysis of SMD opens up new perspectives. As an example, we use our SMD results to show convergence of the MAP for estimating a Gaussian with unknown mean and covariance. As evidenced in Le Priol et al. [17], all existing generic analyses of stochastic mirror descent failed to obtain such results.

# References

[1] H. H. Bauschke, J. M. Borwein, et al. Legendre functions and the method of random bregman projections. Journal of convex analysis, 4(1):27-67, 1997.  
[2] H. H. Bauschke, J. Bolte, and M. Teboulle. A descent lemma beyond lipschitz gradient continuity: first-order methods revisited and applications. Mathematics of Operations Research, 42(2):330-348, 2017.  
[3] M. Bertero, P. Boccacci, G. Desiderà, and G. Vicidomini. Image deblurring with poisson data: from cells to galaxies. Inverse Problems, 25(12):123006, 2009.  
[4] L. Bottou. Large-scale machine learning with stochastic gradient descent. In Proceedings of COMPSTAT'2010: 19th International Conference on Computational Statistics Paris France, August 22-27, 2010 Keynote, Invited and Contributed Papers, pages 177-186. Springer, 2010.  
[5] S. Bubeck et al. Convex optimization: Algorithms and complexity. Foundations and Trends® in Machine Learning, 8(3-4):231–357, 2015.  
[6] R.-A. Dragomir and Y. Nesterov. Convex quartic problems: homogenized gradient method and preconditioning. arXiv preprint arXiv:2306.17683, 2023.  
[7] R. A. Dragomir, M. Even, and H. Hendriks. Fast stochastic bregman gradient methods: Sharp analysis and variance reduction. In International Conference on Machine Learning, pages 2815-2825. PMLR, 2021.  
[8] R.-A. Dragomir, A. B. Taylor, A. d'Aspremont, and J. Bolte. Optimal complexity and certification of bregman first-order methods. Mathematical Programming, pages 1-43, 2021.  
[9] R. M. Gower, N. Loizou, X. Qian, A. Sailanbayev, E. Shulgin, and P. Richtárik. Sgd: General analysis and improved rates. In International conference on machine learning, pages 5200–5209. PMLR, 2019.  
[10] F. Hanzely and P. Richtárik. Fastest rates for stochastic mirror descent methods. Computational Optimization and Applications, 79:717-766, 2021.  
[11] H. Hendrikx, F. Bach, and L. Massoulie. Dual-free stochastic decentralized optimization with variance reduction. Advances in neural information processing systems, 33:19455-19466, 2020.  
[12] H. Hendrikx, L. Xiao, S. Bubeck, F. Bach, and L. Massoulie. Statistically preconditioned accelerated gradient method for distributed optimization. In International conference on machine learning, pages 4203-4227. PMLR, 2020.  
[13] D. Hoeven, T. Erven, and W. Kotlowski. The many faces of exponential weights in online learning. In Conference On Learning Theory, pages 2067-2092. PMLR, 2018.  
[14] S. Kakade, S. Shalev-Shwartz, A. Tewari, et al. On the duality of strong convexity and strong smoothness: Learning applications and matrix regularization. Unpublished Manuscript, http://ttic.uchicago.edu/shai/papers/KakadeShalevTewari09.pdf, 2(1):35, 2009.  
[15] F. Kunstner, R. Kumar, and M. Schmidt. Homeomorphic-invariance of em: Non-asymptotic convergence in  $\mathrm{kl}$  divergence for exponential families via mirror descent. In International Conference on Artificial Intelligence and Statistics, pages 3295-3303. PMLR, 2021.  
[16] S. Lacoste-Julien, M. Schmidt, and F. Bach. A simpler approach to obtaining an o (1/t) convergence rate for the projected stochastic subgradient method. arXiv preprint arXiv:1212.2002, 2012.  
[17] R. Le Priol, F. Kunstner, D. Scieur, and S. Lacoste-Julien. Convergence rates for the map of an exponential family and stochastic mirror descent—an open problem. arXiv preprint arXiv:2111.06826, 2021.  
[18] F. Léger and P.-C. Aubin-Frankowski. Gradient descent with a general cost. arXiv preprint arXiv:2305.04917, 2023.

[19] N. Loizou, S. Vaswani, I. H. Laradji, and S. Lacoste-Julien. Stochastic polyak step-size for sgd: An adaptive learning rate for fast convergence. In International Conference on Artificial Intelligence and Statistics, pages 1306-1314. PMLR, 2021.  
[20] H. Lu, R. M. Freund, and Y. Nesterov. Relatively smooth convex optimization by first-order methods, and applications. SIAM Journal on Optimization, 28(1):333-354, 2018.  
[21] B. McMahan. Follow-the-regularized-leader and mirror descent: Equivalence theorems and 11 regularization. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pages 525–533. JMLR Workshop and Conference Proceedings, 2011.  
[22] A. S. Nemirovskij and D. B. Yudin. Problem complexity and method efficiency in optimization. 1983.  
[23] D. Pfau. A generalized bias-variance decomposition for bregman divergences. Unpublished Manuscript, 2013.  
[24] O. Shamir, N. Srebro, and T. Zhang. Communication-efficient distributed optimization using an approximate newton-type method. In International conference on machine learning, pages 1000-1008. PMLR, 2014.  
[25] C. Villani et al. Optimal transport: old and new, volume 338. Springer, 2009.  
[26] M. J. Wainwright, M. I. Jordan, et al. Graphical models, exponential families, and variational inference. Foundations and Trends® in Machine Learning, 1(1-2):1-305, 2008.
