# Unbalanced Optimal Transport through Non-negative Penalized Linear Regression

Anonymous Author(s)

Affiliation

Address

email

# Abstract

This paper addresses the problem of Unbalanced Optimal Transport (UOT) in which the marginal conditions are relaxed (using weighted penalties in lieu of equality) and no additional regularization is enforced on the OT plan. In this context, we show that the corresponding optimization problem can be reformulated as a non-negative penalized linear regression problem. This reformulation allows us to propose novel algorithms inspired from inverse problems and nonnegative matrix factorization. In particular, we consider majorization-minimization which leads in our setting to efficient multiplicative updates for a variety of penalties. Furthermore, we derive for the first time an efficient algorithm to compute the regularization path of UOT with quadratic penalties. The proposed algorithm provides a continuity of piece-wise linear OT plans converging to the solution of balanced OT (corresponding to infinite penalty weights). We perform several numerical experiments on simulated and real data illustrating the new algorithms, and provide a detailed discussion about more sophisticated optimization tools that can further be used to solve OT problems thanks to our reformulation.

# 1 Introduction

Optimal Transport (OT) theory provides powerful tools for comparing probability distributions and has been successfully employed in a wide range of machine learning applications such as supervised learning (Frogner et al., 2015), clustering (Ho et al., 2017), generative modelling (Arjovsky et al., 2017), domain adaptation (Courty et al., 2017), learning of structured data (Maretic et al., 2019; Vayer et al., 2019) or natural language processing (Kusner et al., 2015), among many others. One reason for those recent successes is the introduction of entropy-regularized OT that can be solved with the efficient Sinkhorn-Knopp matrix scaling algorithm (Cuturi, 2013). However, the classical OT problem seeks the optimal cost to transport all the mass from a source distribution to a target one (Villani, 2009), greatly limiting its use in scenarios where the measures have different masses or when they contain noisy observations or outliers.

Unbalanced Optimal Transport (UOT) (Benamou, 2003) has been introduced to tackle this shortcoming, allowing some mass variation in the transportation problem. It is expressed as a relaxation of the Kantorovich formulation (Kantorovich, 1942) by penalizing the divergence between the marginals of the transportation plan and the given distributions. Several divergences can be considered, such as the Kullback-Leibler (KL) divergence (Frogner et al., 2015; Liero et al., 2018), the  $\ell_1$  norm corresponding to the partial optimal transport problem (Caffarelli and McCann, 2010; Figalli, 2010), or the squared  $\ell_2$  norm (Benamou, 2003). Regarding numerical solutions, Chizat et al. (2018) considered an entropic-regularized version of UOT leading to a class of scaling algorithms in the vein of the Sinkhorn-Knopp approach (Sinkhorn and Knopp, 1967). The introduction of this entropic regularization improves the scalability of OT, but involves a spreading of the mass and a loss of sparsity in the OT plan. When a

sparse transport plan is sought, the convergence is slowed down, necessitating the use of acceleration strategies (Thibault et al., 2021). Regarding UOT with the (squared)  $\ell_2$  norm, Blondel et al. (2018) showed that the resulting OT plan is sparse and proposed to use an efficient L-BFGS-B algorithm (Byrd et al., 1995) to address this case. Note that the L-BFGS-B method can be used to solve UOT with differentiable divergences even without the entropic-regularization on the OT plan that induces the Sinkhorn-like iterations. Finally, also note that, as for balanced OT, UOT can be solved more efficiently when the data has a specific structure, such as unidimensional distributions (Bonneel and Coeurjolly, 2019) or distributions supported on trees (Sato et al., 2020).

Contributions. In this paper, we show after some preliminaries that UOT can be recast as a convex penalized linear regression problem with non-negativity constraints (Section 2.2). The main interest of this reformulation resides in the fact that non-negative linear regression has been extensively studied in inverse problems and machine learning, offering a large panel of tools for devising new numerical algorithms. Our reformulation involves a design/dictionary matrix that is structured and sparse. Leveraging this structure, we propose two new families of algorithms for solving the exact (i.e., without regularization of the plan) UOT problem in Section 3

We first derive in Section 3.1 a new Majorization-Minimization (MM) algorithm for solving UOT with Bregman divergences, and more specifically KL and  $\ell_2$ -penalized UOT. The MM approach results in multiplicative updates that have appealing features: i) they are easy to implement, ii) have low complexity per iteration and can be instantiated on GPU, iii) ensure monotonicity of the objective function and inherit existing convergence results. Our methodology is inspired by well-known algorithms in image restoration (Richardson, 1972; De Pierro, 1993) and non-negative matrix factorization (NMF) (Lee and Seung, 2001; Dhillon and Sra, 2005; Fevtte and Idier, 2011). Interestingly, the resulting multiplicative updates bear a similarity with the celebrated Sinkhorn scaling algorithm, with some key differences that are discussed.

Next, we derive in Section 3.2 an efficient algorithm to compute the regularization path in  $\ell_2$ -penalized UOT. To do so, we build on our proposed reformulation and more precisely on the fact that  $\ell_2$ -penalized UOT can be reformulated as a weighted Lasso problem. We propose a new methodology inspired by LARS (Efron et al., 2004; Hastie et al., 2004), which, to the best of our knowledge, is the first regularization path algorithm for OT problems. It brings a novel understanding of the properties of the evolution of the support of OT plans, besides the practical interest of computing the complete regularization path when hyperparameter validation is necessary.

Our new families of algorithms (MM for general UOT, LARS for  $\ell_2$ -penalized UOT) are showcased in the numerical experiments of Section 4. Python implementation of the algorithms, provided in supplementary, will be released with MIT license on GitHub. The connection between UOT and linear regression that we reveal in the paper opens the door to further fruitful developments and in particular to more efficient algorithms, thanks to the large literature dealing with non-negative penalized linear regression. We discuss those possible research directions in Section 5, before concluding the paper.

Notations. Vectors such as  $\pmb{m}$  are written with lower case and bold font, with coefficients  $m_{i}$  or  $[m]_i$ , according to context. The  $|\mathcal{A}|$ -dimensional sub-vector with indexes in set  $\mathcal{A}$  is written  $\pmb{m}_{\mathcal{A}}$ . Matrices such as  $M$  are written with upper case and bold font, with coefficients  $M_{i,j}$ . We introduce a vectorization operator defined by  $\pmb{m} = \mathrm{vec}(M) = [M_{1,1}, M_{1,2}, \dots, M_{n,m-1}, M_{n,m}]^{\top}$ , i.e., the concatenation of the rows of the matrix, following the Numpy/C memory convention.  $\mathbb{1}_n$  is a vector of  $n$  ones and  $M \geq 0$  denotes entry-wise non-negativity. Finally,  $D_{\varphi}$  is the Bregman divergence generated by the strictly convex and differentiable function  $\varphi$ , i.e.,  $D_{\varphi}(\pmb{u},\pmb{v}) = \sum_{i}[\varphi(u_i) - \varphi(v_i) - \varphi'(v_i)(u_i - v_i)]$ .

# 2 Reformulation of UOT as non-negative penalized linear regression

# 2.1 Background on Optimal Transport

Let us consider two clouds of points  $\mathbf{X} = \{\pmb{x}_i\}_{i=1}^n$  and  $\mathbf{Y} = \{\pmb{y}_j\}_{j=1}^m$ . Let  $\pmb{a} \in \mathbb{R}_n^+$  and  $\pmb{b} \in \mathbb{R}_m^+$  be two discrete distributions of mass on  $\mathbf{X}$  and  $\mathbf{Y}$ , such that  $a_i$  (resp.  $b_j$ ) is the mass at  $\pmb{x}_i$  (resp.  $\pmb{y}_j$ ). The balanced OT problem, as defined by Kantorovich [1942], is a linear problem that computes the

87 minimum cost of moving  $\pmb{a}$  to  $\pmb{b}$ :

$$
\mathrm {O T} (\boldsymbol {a}, \boldsymbol {b}) = \min  _ {\boldsymbol {T} \geq 0} \langle \boldsymbol {C}, \boldsymbol {T} \rangle \quad \text {s u c h t h a t (s . t .)} \quad \boldsymbol {T} \mathbb {1} _ {m} = \boldsymbol {a}, \boldsymbol {T} ^ {\top} \mathbb {1} _ {n} = \boldsymbol {b} \tag {1}
$$

where  $\langle \cdot ,\cdot \rangle$  is the Frobenius inner product,  $T\in \mathbb{R}_{n\times m}^{+}$  is the transport plan and  $C\in \mathbb{R}_{n\times m}^{+}$  is the cost matrix. The entry  $C_{i,j}$  of  $C$  represents the cost of moving point  $\pmb {x}_i$  to  $\pmb {y}_j$ . The Wasserstein 1- distance (also known as the earth mover's distance) is obtained for  $C_{i,j} = \| x_i - y_j\|$ . The constraints on the transport plan  $T$  require that  $\| a\| _1 = \| b\| _1$  and that all the mass from  $\pmb{a}$  is transported to  $\pmb{b}$ . These constraints can be alleviated through relaxation, leading to UOT (Benamou, 2003):

$$
\operatorname {U O T} ^ {\lambda} (\boldsymbol {a}, \boldsymbol {b}) = \min  _ {\boldsymbol {T} \geq 0} \langle \boldsymbol {C}, \boldsymbol {T} \rangle + \lambda_ {1} D _ {\varphi} (\boldsymbol {T} \mathbb {1} _ {m}, \boldsymbol {a}) + \lambda_ {2} D _ {\varphi} (\boldsymbol {T} ^ {\top} \mathbb {1} _ {n}, \boldsymbol {b}). \tag {2}
$$

The deviations from the true margins are penalized by means of a given Bregman divergence  $D_{\varphi}$ , as introduced in Chizat et al. (2018), where  $\lambda_{1}$  and  $\lambda_{2}$  are hyperparameters that represent the strengths of penalization. Note that balanced OT (I) is recovered when  $\lambda_{1} = \lambda_{2} \to \infty$ . Furthermore, when  $\lambda_{1}$  or  $\lambda_{2} \to \infty$ , we recover semi-relaxed OT (Rabin et al. 2014). In practice, authors often set  $\lambda_{1} = \lambda_{2} = \lambda$  for UOT in order to reduce the necessity of hyperparameter tuning. Various divergences have been considered in the literature. The  $\ell_{1}$  norm gives rise to so-called partial optimal transport (Caffarelli and McCann, 2010). The squared  $\ell_{2}$  norm provides a sparse and smooth transport plan (Blondel et al., 2018) when introducing a strongly convex term in Eq. (2). Chizat et al. (2018) derive efficient algorithms to solve Eq. (2) for several divergences by adding an additional regularization term  $\lambda_{\mathrm{reg}} D_{\varphi} (T, ab^{\top})$ . In particular, entropic regularization is obtained when the KL divergence is used, promoting a dense transport plan unlike exact UOT.

# 2.2 Reformulation of UOT

105 UOT cast as regression. Let  $t = \operatorname{vec}(T)$ ,  $c = \operatorname{vec}(C)$  and  $y^\top = [a^\top, b^\top]$ . Problem (2) can be re-written as

$$
\min  _ {\boldsymbol {t} \geq 0} F _ {\lambda} (\boldsymbol {t}) \stackrel {\text {d e f}} {=} \frac {1}{\lambda} \boldsymbol {c} ^ {\top} \boldsymbol {t} + D _ {\varphi} (\boldsymbol {H} \boldsymbol {t}, \boldsymbol {y}) \tag {3}
$$

and as such be expressed as a non-negative penalized linear regression problem, where the design matrix  $\pmb{H} = [H_r^\top, H_c^\top]^\top$  is the concatenation of the matrices  $\pmb{H}_r$  and  $\pmb{H}_c$  that compute sums of the rows and columns of  $\pmb{T}$ , respectively (see expressions in Section A.1 of the supplementary material). Note that, for the sake of simplicity, we consider here  $\lambda_1 = \lambda_2 = \lambda$  but this hypothesis could be easily alleviated for a given family of divergences (see Sec. 5 for a discussion). Important features of Eq. 3 should be discussed. First,  $F_{\lambda}(t)$  is convex thanks to the convexity of Bregman divergences w.r.t. their first argument. Second,  $\pmb{H}$  is very structured and sparse (with a ratio of only  $\frac{1}{m + n}$  non-zero coefficients) which will allow for more efficient computations and updates than with a dense  $\pmb{H}$ . Finally, since  $\pmb{t} \geq 0$  and  $\pmb{c} \geq 0$ , the linear term can be expressed as  $\frac{1}{\lambda} c^\top t = \frac{1}{\lambda} \sum_i c_i t_i = \frac{1}{\lambda} \sum_i c_i |t_i|$ . This corresponds to a weighted  $\ell_1$  regularization, promoting sparsity in  $\pmb{t}$  and hence in the transport plans. Note that the "sparse" regularization is here controlled by  $\frac{1}{\lambda}$  (instead of  $\lambda$  in classical penalized linear regression), meaning that the sparsity promoting term will be more aggressive for small  $\lambda$ .

Solving problem (3). Problems of the form of Eq. (3) are well-known in inverse problems and NMF. In inverse problems,  $t$  typically acts as a clean image degraded by operator  $H$  (e.g., a convolution) and noise. The data fitting term  $D_{\varphi}(\mathbf{H}t, \mathbf{y})$  captures assumptions about the noise corrupting the observed image  $\mathbf{y}$ . Sparsity is a common regularizer of  $t$ . In NMF, given a set of nonnegative samples  $\{\mathbf{y}_l\}$  one wants to learn a non-negative dictionary  $H$  and non-negative lower-dimensional embeddings  $\{t_l\}$  such that  $\mathbf{y}_l \approx \mathbf{H}t_l$  (Lee and Seung, 1999). Updating the latter involves optimization problems of form (3) (with or without sparse regularization). In contrast to problem (3), the data fitting term is more commonly  $D_{\varphi}(\mathbf{y}, \mathbf{H}t)$  instead of  $D_{\varphi}(\mathbf{H}t, \mathbf{y})$  in inverse problems and NMF. This is because the former is a log-likelihood in disguise for the mean-parametrized exponential family, and takes important noise models as special cases, such as Poisson, additive Gaussian or multiplicative Gamma noise (Févotte and Idier, 2011). Using such penalizations with reversed arguments would be possible in our case as well but we stick to the now standard formulation of (Liero et al., 2018; Chizat et al., 2018) for simplicity.

In the next section, we will first leverage a classical family of algorithms in inverse problems and NMF, namely MM, to obtain new algorithms for KL and  $\ell_2$ -penalized UOT (possibly with entropic regularization in the first case). Second, we will leverage results about non-negative Lasso to design an efficient algorithm to compute the regularization path of  $\ell_2$ -penalized UOT.

# 3 Novel numerical solvers for UOT

# 3.1 Majorization-Minimization (MM) for UOT

General MM framework. MM algorithms have been around a long time in inverse problems and NMF to solve problems of form (3). Classical algorithms for NMF such as (Lee and Seung, 2001) have built on seminal MM algorithms for inverse problems such as (Richardson, 1972; De Pierro, 1993). Subsequent works in NMF such as (Dhillon and Sra, 2005; Fevtte and Idier, 2011; Yang and Oja, 2011) have further contributed novel MM algorithms for larger classes of problems, including larger families of divergences. In a nutshell, MM consists in iteratively building and minimizing an upper bound of the objective function which is tight at the current parameter estimate (and referred to as auxiliary function), see Hunter and Lange (2004); Sun et al. (2017) for tutorials. In NMF, a common approach consists of alternating the updates of the dictionary  $H$  and of the embeddings. In our case,  $H$  is fixed and we may use the results of (Dhillon and Sra, 2005) to build an auxiliary function for term  $D_{\varphi}(Ht, y)$ , to which we may simply add the linear term  $c^{\dagger} t / \lambda$  to obtain a valid auxiliary function for  $F_{\lambda}(t)$ . Let  $\tilde{t}$  denote the current estimate of  $t$ ,  $\tilde{Z}_{i,j} = \frac{H_{i,j} \tilde{t}_j}{\sum_l H_{i,l} t_l}$  and

$$
G _ {\lambda} (\boldsymbol {t}, \tilde {\boldsymbol {t}}) = \sum_ {i, j} \tilde {Z} _ {i, j} \varphi \left(\frac {H _ {i , j} t _ {j}}{\tilde {Z} _ {i , j}}\right) + \sum_ {j} \left[ \frac {c _ {j}}{\lambda} - \sum_ {i} H _ {i, j} \varphi^ {\prime} \left(y _ {i}\right) \right] t _ {j} + c s t, \tag {4}
$$

where  $cst = \sum_{i}[\varphi'(y_i)y_i - \varphi(y_i]$ . Then,  $G_{\lambda}(t,\tilde{t})$  is an auxiliary function for  $F_{\lambda}(t)$ , i.e.,  $\forall t$ ,  $G_{\lambda}(t,\tilde{t}) \geq F_{\lambda}(t)$  and  $G_{\lambda}(\tilde{t},\tilde{t}) = F_{\lambda}(\tilde{t})$ . Let  $\pmb{t}^{(k+1)} = \mathrm{argmin}_{t \geq 0} G_{\lambda}(\pmb{t},\pmb{t}^{(k)})$ , then  $F_{\lambda}(\pmb{t}^{(k)}) = G_{\lambda}(\pmb{t}^{(k)},\pmb{t}^{(k)}) \geq G_{\lambda}(\pmb{t}^{(k+1)},\pmb{t}^{(k)}) \geq F_{\lambda}(\pmb{t}^{(k+1)})$ , producing a descent algorithm over  $F$ . The trick to obtain  $G$  is to apply Jensen inequality to  $\varphi(\sum_{j} H_{i,j} t_j) = \varphi(\sum_{j} \tilde{Z}_{i,j} \frac{H_{i,j}}{\tilde{Z}_{i,j}} t_j) \leq \sum_{j} \tilde{Z}_{i,j} \varphi(\frac{H_{i,j}}{\tilde{Z}_{i,j}} t_j)$ , thanks to the convexity of  $\varphi$ , see details in (Dhillon and Sra, 2005). We provide below the resulting algorithms for the KL and  $\ell_2$  penalizations, with detailed computations available in Section A.2 of the supplementary.

MM for KL-penalized UOT. The KL divergence is obtained with  $\varphi(y) = y\log y - y$ . Minimizing  $G_{\lambda}(t,t^{(k)})$  in that case leads to following multiplicative update:

$$
t _ {j} ^ {(k + 1)} = t _ {j} ^ {(k)} \exp \left(\frac {\left[ \boldsymbol {H} ^ {\top} \log (\boldsymbol {y}) - \boldsymbol {H} ^ {\top} \log \left(\boldsymbol {H} \boldsymbol {t} ^ {(k)}\right) \right] _ {j} - \frac {1}{\lambda} c _ {j}}{\left[ \boldsymbol {H} ^ {\top} \mathbb {1} \right] _ {j}}\right). \tag {5}
$$

Owing to the structure of  $t$  and  $H$ , the update can be re-written in the following matrix form:

$$
\boldsymbol {T} ^ {(k + 1)} = \operatorname {d i a g} \left(\frac {\boldsymbol {a}}{\boldsymbol {T} ^ {(k)} \mathbb {1} _ {m}}\right) ^ {\frac {1}{2}} \left(\boldsymbol {T} ^ {(k)} \odot \exp \left(- \frac {\boldsymbol {C}}{2 \lambda}\right)\right) \operatorname {d i a g} \left(\frac {\boldsymbol {b}}{\boldsymbol {T} ^ {(k) \top} \mathbb {1} _ {n}}\right) ^ {\frac {1}{2}}, \tag {6}
$$

where  $\odot$  is entrywise multiplication and divisions are taken entrywise as well. The multiplicative update (6) is remarkably similar to the well-known Sinkhorn-Knopp algorithm that has been used in numerous OT problems involving KL regularization. But instead of two separate steps for the left and right scaling, Eq. (6) applies these scalings simultaneously in a unique update using the diagonal matrices (and a form of geometrical average). Also note how the scaling factor  $\exp\left(-\frac{C}{2\lambda}\right)$  penalizes along iterations the coefficients of the transport plan with large costs.

MM for  $\ell_2$ -penalized UOT. The quadratic loss is obtained with  $\varphi(y) = \frac{y^2}{2}$ . In that case, minimizing  $G_{\lambda}(t, t^{(k)})$  s.t. non-negativity leads to following multiplicative update:

$$
\boldsymbol {T} ^ {(k + 1)} = \boldsymbol {T} ^ {(k)} \odot \frac {\operatorname* {m a x} \left(0 , \boldsymbol {a} \mathbb {1} _ {m} ^ {\top} + \mathbb {1} _ {n} \boldsymbol {b} ^ {\top} - \frac {1}{\lambda} \boldsymbol {C}\right)}{\boldsymbol {T} ^ {(k)} \mathbf {O} _ {m} + \mathbf {O} _ {n} \boldsymbol {T} ^ {(k)}} \quad \text {w i t h} \quad \mathbf {O} _ {\ell} = \mathbb {1} _ {\ell} \mathbb {1} _ {\ell} ^ {\top}. \tag {7}
$$

Interestingly enough, update (7) prunes any coefficient  $T_{i,j}$  in  $\pmb{T}$  such that  $a_i + b_j - \frac{1}{\lambda} C_{i,j} < 0$  from the very first iteration, providing a useful certificate on the support of the solution.

# 3.2 Regularization path for  $\ell_2$ -penalized UOT

Let us focus on the case where  $D_{\varphi}$  is a quadratic divergence. As mentioned in Section 2.2, Eq. (3) is then a positive weighted Lasso problem, allowing us to derive the first regularization path algorithm for computing the whole set of solutions for a varying  $\lambda$  from 0 to  $+\infty$ . Note that the path's extreme point recovers the balanced OT solution. We show that the path is piecewise linear in  $1 / \lambda$  between changes in the active set  $\mathcal{A} = \mathrm{supp}(t^{\lambda})$ , where  $t^{\lambda} = \mathrm{vec}(T^{\lambda})$  and  $T^{\lambda}$  is the OT plan for given hyperparameter  $\lambda$ . The main steps of the algorithm are roughly as follows: given a current solution  $(\lambda_k, T^{\lambda_k})$  and a current active set  $\mathcal{A}_k$ , we look for the next value  $\lambda_{k+1} > \lambda_k$  such that the active set changes (i.e.,  $\mathcal{A}_{k+1} \neq \mathcal{A}_k$ ), either because one component enters or leaves the active set. We describe our algorithm below.

KKT conditions of the  $\ell_2$ -penalized UOT problem. The Lagrangian for problem (3) writes:

$$
L _ {\lambda} (\boldsymbol {t}, \gamma) = \frac {1}{\lambda} \boldsymbol {c} ^ {\top} \boldsymbol {t} + \frac {1}{2} (\boldsymbol {H} \boldsymbol {t} - \boldsymbol {y}) ^ {\top} (\boldsymbol {H} \boldsymbol {t} - \boldsymbol {y}) - \gamma^ {\top} \boldsymbol {t} \tag {8}
$$

where  $\gamma$  represents the Lagrange parameters. We denote  $m = H^{\top}y = \mathrm{vec}(a\mathbb{1}_{m}^{\top} + \mathbb{1}_{n}b^{\top})$ . KKT optimality conditions state that i)  $\nabla_{t}L_{\lambda}(t,\lambda) = \frac{1}{\lambda} c + H^{\top}Ht - m - \gamma = 0$  (stationarity condition), ii)  $\gamma \odot t = 0$  (complementary condition) and iii)  $\gamma \geq 0$  (feasibility condition).

Piecewise linearity of the path. Assume that, at iteration  $k$ , we know the current active set  $\mathcal{A} = \mathcal{A}_k$  and we look for  $t_{\mathcal{A}}^{\lambda}$  (the other values of  $t_{\mathcal{A}}$  being 0). Let  $H_{\mathcal{A}}$ ,  $m_{\mathcal{A}}$  and  $c_{\mathcal{A}}$  denote the corresponding sub-matrix and vectors (see Appendix A.3 for rigorous definitions). Because of the complementary condition, we have  $\gamma_{\mathcal{A}} = 0$ . Using  $\lambda = \lambda_k + \epsilon$ , with  $\epsilon > 0$  small enough to ensure that the active set remains the same, the stationarity condition writes:

$$
\boldsymbol {H} _ {\mathcal {A}} ^ {\top} \boldsymbol {H} _ {\mathcal {A}} t _ {\mathcal {A}} ^ {\lambda} = \boldsymbol {m} _ {\mathcal {A}} - \frac {1}{\lambda} \boldsymbol {c} _ {\mathcal {A}} \Rightarrow \boldsymbol {t} _ {\mathcal {A}} ^ {\lambda} = \tilde {\boldsymbol {m}} _ {\mathcal {A}} - \frac {1}{\lambda} \tilde {\boldsymbol {c}} _ {\mathcal {A}} \tag {9}
$$

with  $\tilde{m}_{\mathcal{A}} = (H_{\mathcal{A}}^{\top}H_{\mathcal{A}})^{-1}m_{\mathcal{A}}$  and  $\tilde{c}_{\mathcal{A}} = (H_{\mathcal{A}}^{\top}H_{\mathcal{A}})^{-1}c_{\mathcal{A}}$ . Eq. (9) shows that the optimal  $t_{\mathcal{A}}^{\lambda}$  (and hence  $t^{\lambda}$ ) can be solved for any  $\lambda \in [\lambda_k, \lambda_{k+1}]$ , i.e., when the active set  $\mathcal{A}$  remains the same, by solving a linear problem. It also reveals the piecewise linearity in  $\lambda^{-1}$  of the path when  $\mathcal{A}$  is fixed. As expected, balanced OT is recovered when  $\lambda \to \infty$ .

Finding  $(\lambda_{k + 1},\mathcal{A}_{k + 1})$  given  $(\lambda_k,\mathcal{A}_k)$ . Given a current solution  $(\lambda_k,t^{\lambda_k})$  and  $\lambda = \lambda_{k} + \epsilon$ , we increase the  $\epsilon$  until we reach a change in the set of active components. This happens whenever the first of the following two situations occurs.

- One component in  $\mathcal{A}$  becomes inactive. In that case, we remove the index  $i \in \mathcal{A}$  with the smallest  $\lambda_r > \lambda_k$  that violates the constraint. In such case,  $[\tilde{m}_{\mathcal{A}}]_i = [\tilde{c}_{\mathcal{A}}]_i / \lambda$  and we may write

$$
\lambda_ {r} = \min  _ {> \lambda_ {k}} \left(\frac {\tilde {c} _ {\mathcal {A}}}{\tilde {m} _ {\mathcal {A}}}\right) \tag {10}
$$

where  $\min_{> \lambda_k}$  indicates the minimum value in the vector greater than  $\lambda_k$  and the division is entrywise.

- One component in  $\bar{\mathcal{A}}$  becomes active. This occurs when the KKT positivity constraint  $\gamma_{\bar{A}} \geq 0$  becomes violated. Assume this happens at index  $i \in \bar{\mathcal{A}}$  for the smallest value  $\lambda_a > \lambda_k$  of  $\lambda$ . In such case, the stationarity condition outside the active set can be rewritten:

where  $\tilde{m}$  (resp.  $\tilde{c}$ ) equals  $\tilde{m}_{\mathcal{A}}$  (resp.  $\tilde{c}_{\mathcal{A}}$ ) on  $\mathcal{A}$  and zero on  $\bar{\mathcal{A}}$ .

$$
\left[ \frac {1}{\lambda} \boldsymbol {c} _ {\bar {A}} + \left[ \boldsymbol {H} ^ {\top} \boldsymbol {H} \left(\tilde {\boldsymbol {m}} + \frac {1}{\lambda} \tilde {\boldsymbol {c}}\right) \right] _ {\bar {A}} - \boldsymbol {m} _ {\bar {A}} \right] _ {i} = \left[ \gamma_ {\bar {A}} \right] _ {i} \Rightarrow \lambda_ {a} = \min  _ {> \lambda_ {k}} \left(\frac {\boldsymbol {c} _ {\bar {A}} - \left[ \boldsymbol {H} ^ {\top} \boldsymbol {H} \tilde {\boldsymbol {c}} \right] _ {\bar {A}}}{\boldsymbol {m} _ {\bar {A}} - \left[ \boldsymbol {H} ^ {\top} \boldsymbol {H} \tilde {\boldsymbol {m}} \right] _ {\bar {A}}}\right), \tag {11}
$$

In practice, at each step of the path, we compute both  $\lambda_r$  and  $\lambda_a$ , set  $\lambda_{k+1} = \min \{\lambda_r, \lambda_a\}$  and update the active set accordingly.

Numerical computation of the entire path. Eq. (9) involves the computation of the matrix  $(\pmb{H}_{\mathcal{A}}^{\top}\pmb{H}_{\mathcal{A}})^{-1}$ , which is of size  $|\mathcal{A}|\times |\mathcal{A}|$ . As only one index leaves or enters the active set at each iteration, we can use the Schur complement of the matrix to compute its value from  $(\pmb{H}_{\mathcal{A}_k}^\top \pmb{H}_{\mathcal{A}_k})^{-1}$ ,

Algorithm 1 Regularization path of  $\ell_2$ -penalized UOT  
Require:  $a, b, C, \lambda_0 = 0, t_0 = 0, A = A_0 = \emptyset, k = 1$ $\lambda_1 = \min \frac{c_{\bar{A}}}{m_{\bar{A}}}, A = A_1 = \arg \min \frac{c_{\bar{A}}}{m_{\bar{A}}}, H_{A}^{\top}H_{A} = 2$ $t_{A_1}^{\lambda_1} = \frac{m_A}{2} - \frac{1}{\lambda_1} \frac{c_A}{2}$   
while  $(Ht^{\lambda_k} - y)^{\top}(Ht^{\lambda_k} - y) \neq 0$  do  
 $\lambda_r, \lambda_a \gets$  Compute as in Eq. (10) and Eq. (11)  
 $\lambda_{k+1} \gets \min(\lambda_r, \lambda_a)$ $t_{A}^{\lambda_{k+1}} \gets (H_{A}^{\top}H_{A})^{-1}m_{A} - \frac{1}{\lambda_{k+1}}(H_{A}^{\top}H_{A})^{-1}c_{A}$ $\mathcal{A} = \mathcal{A}_{k+1} \gets$  Update active set for next iteration.  
 $(H_{A}^{\top}H_{A})^{-1} \gets$  Update from  $(H_{A_k}^{\top}H_{A_k})^{-1}$  with Schur complement (see supplementary A.3)  
 $k \gets k + 1$   
end while  
return  $(\lambda_k, t^{\lambda_k})_k$

alleviating the computational burden of the algorithm as it only involves matrix-vector computations (see Section A.3 of supplementary). Algorithm II sums up the different steps of the full path computation. At each iteration, we compute  $\lambda_{a},\lambda_{r}$  , update the inverse matrix  $(H_{A_k}^\top H_{A_k})^{-1}$  and estimate the solution  $t^{\lambda_{k + 1}}$  with a complexity of  $O(nm)$  
Regularization path of the semi-relaxed  $\ell_2$ -penalized UOT. As a side result, let us consider the semi-relaxed OT problem  $\mathrm{SROT}^{\lambda}(\pmb{a},\pmb{b}) = \min_{\pmb{T}\geq 0,\pmb{T}^{\top}\mathbb{1}_{n}} = \pmb{b}\langle \pmb{C},\pmb{T}\rangle +\lambda \| \pmb{T}\mathbb{1}_{m} - \pmb{a}\|^{2}$ . The main difference with UOT is that the equality constraint  $\pmb{T}^{\top}\mathbb{1}_n = \pmb{b}$  (equivalent to  $\pmb{H}_c\pmb {t} = \pmb {b}$ ) must always be met. This leads to the following Lagrangian:

$$
L _ {\lambda} (\boldsymbol {t}, \gamma , \boldsymbol {u}) = \frac {1}{\lambda} \boldsymbol {c} ^ {\top} \boldsymbol {t} + \frac {1}{2} \left(\boldsymbol {H} _ {r} \boldsymbol {t} - \boldsymbol {a}\right) ^ {\top} \left(\boldsymbol {H} _ {r} \boldsymbol {t} - \boldsymbol {a}\right) + \left(\boldsymbol {H} _ {c} \boldsymbol {t} - \boldsymbol {b}\right) ^ {\top} \boldsymbol {u} - \gamma^ {\top} \boldsymbol {t}, \tag {12}
$$

where  $\pmb{u} \in \mathbb{R}^{m}$  contains the Lagrange parameters associated to the  $m$  equality constraints. The KKT optimality conditions now dictate that i)  $\nabla_{t}L_{\lambda}(t,\gamma,\pmb{u}) = \frac{1}{\lambda}\pmb{c} + \pmb{H}_{r}^{\top}\pmb{H}_{r}\pmb{t} - \pmb{H}_{r}^{\top}\pmb{a} + \pmb{H}_{c}^{\top}\pmb{u} - \gamma = 0$ , ii)  $\gamma \odot t = 0$ , iii)  $\gamma \geq 0$  and  $\pmb{H}_{c}\pmb{t} - \pmb{b} = \pmb{0}$ . We can use the same reasoning than previously to compute the entire path. Details are provided in Section A.4 of the supplementary. The main difference lies in solving, at each iteration, a linear system of size  $(m + |\mathcal{A}|)$  to comply with the marginal equality constraint. The path is initialized as follows: the  $j^{th}$  column of  $\pmb{T}^{0}$  for  $\lambda_0 = 0$  is set to the weighted canonical vector  $b_{i^*}\mathbf{e}_{i^*}$ , where  $i^* = \mathrm{argmin}\{C_{i,j}\}_i$ .

# 4 Numerical experiments

In this section, we first show the solutions obtained with our solvers on simple and interpretable examples. We then evaluate the computational complexity of the different algorithms and finally we show an application where the regularization path can be used on a domain adaptation problem.

Illustration of the algorithms. We first illustrate the regularization path for  $\ell_2$ -penized UOT on a simple example between two distributions containing 3 points each, with different masses and a cost matrix  $C$  given in Fig. (left). We can see on Fig. (right) that, starting from  $\lambda_0 = 0$  and  $T = 0$ , we successively add or remove components in the active set  $\mathcal{A}$  when increasing the  $\lambda$  values. When  $\lambda = \infty$ , we recover the balanced OT solution. Recall that the path is linear in  $1 / \lambda$  (and not  $\lambda$ ). We then illustrate the path for both  $\ell_2$ -penized UOT and semi-relaxed UOT on two 2D distributions with  $n = m = 100$  samples. We can see in Fig. the difference between the two regularization paths for specific values of  $\lambda$ . UOT starts with an empty plan for  $\lambda = 0$  and then activates samples from both source and target from the closest to the farthest ones until convergence to the balanced OT plan. Semi-relaxed UOT starts with all target samples active due to marginal constraints and progressively activates the source samples.

![](images/13dc57135c4a2b67878d4b15f3fc882588a2072f5033c88fe8f31bba4c582e18.jpg)  
Figure 1: (Left) cost matrix  $C$  (the higher the cost, the darker the color); (middle) OT plan whose cells are color-coded with respect to the  $\lambda$  values at which they are activated. The blank cells never enter the active set as the corresponding cost it too high; (right) evolution of  $T_{i,j}$  when  $\lambda$  increases. Note that the  $x$ -axis is in log scale and is discontinued between  $\lambda_7$  and  $\infty$ .

![](images/a3223d1593b3e06f12347aa7eb9fb8e1a870a1826761fe55057d92263179a929.jpg)  
Figure 2: Regularization paths for 2D empirical distributions for  $\ell_2$ -penalized UOT (top) and semi-relaxed UOT (bottom). The OT plan is shown as green lines between the source and target samples when  $T_{i,j} > 0$  and the resulting marginals are shown as filled circles.

Comparison of the performances of the algorithms. We now provide an empirical evaluation of the running times of the proposed algorithms, using 2 sets of 10-dimensional points with  $n = m$  and drawn according to IID Gaussian distributions. The cost matrix  $C$  is computed using a squared  $\ell_2$  norm. We first study the running times of the regularization path algorithm, for  $n = m$  ranging from 100 to 1000, averaging the results over 5 runs, see Fig. 3(left). We empirically observe that log-log plot is near-linear, with an empirical complexity  $O(n^{3.27})$  in this example.

Using  $n = m = 500$ , we compare the running times of the current state-of-the-art BFGS algorithm (Blondel et al., 2018) using Scipy (Virtanen et al., 2020) and those of our algorithms: the  $\ell_2$ -penalized UOT formulated as a Lasso problem (with both the Celer algorithm (Massias et al., 2018) and the coordinate descent solvers from Scikit-learn (Pedregosa et al., 2011)), the multiplicative algorithm for both the  $\ell_2$  and the KL penalties and the regularization path algorithm (see Section A.6 of the supplemental material for more details about the solvers and their parameters). Figure 3 (middle and right) shows the average running time for all algorithms. For  $\ell_2$ -penalized UOT, we observe that, for large  $\lambda$  values, the Lasso solvers are the fastest and that, whatever the value  $\lambda$ , BFGS is the slowest. We also notice that, for large  $\lambda$ , the running times for computing the path remain constant: when the last active set is found, computing the OT plan only involves a weighted sum. As for KL-penalized UOT, the BFGS algorithm is more efficient when large values of  $\lambda$  are considered. One can also notice that, similarly to Sinkhorn which is fast for large regularization values, the multiplicative algorithms for both penalties are also fast for high  $1 / \lambda$  values.

Regularization path for unbalanced domain adaptation. We demonstrate the interest of having the entire regularization path in a classification context where some of the data collection may be polluted by outliers. We consider a setup similar to Mukherjee et al. (2020). Let the source  $X$  be a set of 400 MNIST digits sampled from the digits 0, 1, 2, 3 (100 points per class) and let the target  $Y$  be a

![](images/9ef6ef017270d8526ad40a6e5bb1809a8b744e77f33cbd3b978c8541c45e86cb.jpg)  
Figure 3: (Left) Running times of Alg. 1 w.r.t. the number of points; (middle) comparison of  $\ell_2$ -penalized UOT with  $m = n = 500$  (right) likewise for KL-penalized UOT. Dark curves (resp. shaded regions) represent average (resp. variance) values over 5 runs.

![](images/1aa9ae56b4c192c9e5db975b92f55eb2216310798a2bc603c57f7a5ae15285ed.jpg)

![](images/cf76962f2d6076fc3eba44820d99f5ff28b18434e833339dedb29c07b8f658b8.jpg)

set of digits 0, 1 of MNIST (LeCun et al., 2010) and of digits 8, 9 from Fashion MNIST (Xiao et al., 2017). Our setting is simple classification: we classify a sample of the target dataset by propagating

the label of the source sample it is the most transported to, provided that the transported mass of the target point is greater than  $0.25b_{j}$ . Note that similarly to Mukherjee et al. (2020) a validation set can be used here to select the best  $\lambda$ . Figure 4 shows the overall accuracy, defined as the number of samples that are correctly classified divided by the total number of points, and the current accuracy, which is the proportion of well-classified points among the points that are classified, i.e., that are receiving mass. One can notice that, as the number of classified points increases (with  $\lambda$ ), the overall accuracy increases as more and more points are well classified while the current accuracy remains stable until outliers are included in the labeled set. This suggests that UOT can be used not only for classification but also as an automated outlier detection method.

![](images/01872677dc5c0b9643f6fa53f339fc09e65968d2f743732ffd19ec850355dab5.jpg)  
Figure 4: Evolution of the classification accuracy for the domain adaptation problem w.r.t. the number of classified points.

# 5 Discussion and perspectives

We showed that UOT can be recast as a non-negative penalized linear regression problem, encouraging us to dig into this well-established field of research in order to adapt existing algorithmic solutions to the structure of the UOT problem. In this section, we discuss the relation between the proposed algorithms and classical solvers used in OT, and also investigate some research directions that can widen the scope of proposed methods.

Multiplicative algorithms for UOT. As discussed in Section 3.1, the multiplicative updates for the KL divergence obtained from MM resemble the Sinkhorn algorithm from Chizat et al. (2018), except for the joint scaling and the weighting matrix  $\exp(-C/2)$ . Interestingly, this scaling matrix also appears in the Inexact Proximal Point OT (IPOT) algorithm of Xie et al. (2020) to solve balanced OT. As a matter of fact, we show in Section A.5 of the supplementary that IPOT is a MM algorithm. The idea is to re-write the OT objective as  $[\langle C, T \rangle + \lambda D_{\varphi}(T, ab^{\top})] - \lambda D_{\varphi}(T, ab^{\top})$  and upper bound the concave term by its tangent. This further supports the interest of MM for OT and UOT, and highlight an important feature of one of our contributions: designing the first Sinkhorn-like multiplicative algorithm for UOT that can be applied when the OT plan is not entropy-regularized.

More efficient solvers. Despite the positive experimental results of Section 4, multiplicative and regularization path algorithms can be slow, especially for large values of  $\lambda$ . Various accelerations can be envisaged. Regarding path algorithms, the approach of Mairal and Yu (2012) can compute a regularization path with precision  $\epsilon$  in  $o(1 / \epsilon)$  iterations. This would lead in our setting to a full complexity of  $O(mn / \epsilon)$  that is even interesting to approximate balanced OT. Another way to speed up computations is to use screening. In sparse regression, this consists of eliminating during optimization components that will not belong to the support of the solutions thanks to safe screening tests. Methods such as (El Ghaoui et al., 2012; Wang et al., 2015; Dantas et al., 2021) can readily

be adapted to our  $\ell_2$  or KL-penalized UOT algorithms. Finally, an other line of improvement is to consider stochastic optimization methods such as (Defazio et al., 2014). Given the particular structure of  $H$ , the complexity of stochastic updates shall be small and can lead to very efficient implementations (Nesterov, 2014).  
General case and entropy-regularized UOT. Following (Frogner et al., 2015; Chizat et al., 2018; Séjourne et al., 2019), general regularized UOT can be expressed as:

$$
\operatorname {R U O T} ^ {\lambda} (\boldsymbol {a}, \boldsymbol {b}) = \min  _ {\boldsymbol {T} \geq 0} \langle \boldsymbol {C}, \boldsymbol {T} \rangle + \lambda_ {1} D _ {\varphi} (\boldsymbol {T} \mathbb {1} _ {m}, \boldsymbol {a}) + \lambda_ {2} D _ {\varphi} (\boldsymbol {T} ^ {\top} \mathbb {1} _ {n}, \boldsymbol {b}) + \lambda_ {\text {r e g}} D _ {\varphi} (\boldsymbol {T}, \boldsymbol {a} \boldsymbol {b} ^ {\top}). \tag {13}
$$

As it turns out, this general problem involving different regularization weights  $(\lambda_{1},\lambda_{2},\lambda_{\mathrm{reg}})$  can easily be addressed in our framework as well using two simple tricks. The first one consists of absorbing the regularization weights into the divergences. Indeed, many divergences are homogeneous, i.e., satisfy a relation of the form  $\lambda D_{\varphi}(\mathbf{x}|\mathbf{y}) = D_{\varphi}(\lambda^{\alpha}\mathbf{x}|\lambda^{\alpha}\mathbf{y})$  where  $\alpha$  is divergence-specific. This holds in particular for the KL divergence  $(\alpha = 1)$  and the squared  $\ell_2$  norm  $(\alpha = 1 / 2)$ . The second one consists of complementing  $\pmb{H}$  and  $\pmb{y}$  with suitable terms to account for the regularization term. In the end, we may re-write Eq. (13) into Eq. (3) with  $\lambda = 1$ ,  $\pmb {H} = [\lambda_1^\alpha \pmb {H}_r^\top ,\lambda_2^\alpha \pmb {H}_c^\top ,\lambda_{\mathrm{reg}}^\alpha \pmb {I}]^\top$  and  $\pmb{y}^{\top} = [\lambda_{1}^{\alpha}\pmb{a}^{\top},\lambda_{2}^{\alpha}\pmb{b}^{\top},\lambda_{\mathrm{reg}}^{\alpha}\mathrm{vec}(\pmb{ab}^{\top})^{\top}]$ . In particular, we obtain the following multiplicative update in the case of entropy-regularized KL-penalized UOT:

$$
\boldsymbol {T} ^ {(k + 1)} = \operatorname {d i a g} \left(\frac {\boldsymbol {a}}{\boldsymbol {T} ^ {(k)} \mathbb {1} _ {m}}\right) ^ {\frac {\lambda_ {1}}{\lambda_ {\text {a l l}}}} \left(\left(\boldsymbol {T} ^ {(k)}\right) ^ {\frac {\lambda_ {1} + \lambda_ {2}}{\lambda_ {\text {a l l}}}} \odot \boldsymbol {K}\right) \operatorname {d i a g} \left(\frac {\boldsymbol {b}}{\boldsymbol {T} ^ {(k) \top}} \mathbb {1} _ {n}\right) ^ {\frac {\lambda_ {2}}{\lambda_ {\text {a l l}}}} \tag {14}
$$

where  $\pmb{K} = \left(\pmb{ab}^{\top}\right)^{\frac{\lambda_{\mathrm{reg}}}{\lambda_{\mathrm{all}}}} \odot \exp\left(-\frac{1}{\lambda_{\mathrm{all}}} \pmb{C}\right)$  and  $\lambda_{\mathrm{all}} = \lambda_1 + \lambda_2 + \lambda_{\mathrm{reg}}$ . This multiplicative update is slightly more complex than the Sinkhorn algorithms of Frogner et al. (2015); Chizat et al. (2018) and as such, it might have limited practical interest but is conceptually interesting and novel. Note that balanced UOT as of Eq. (2) is simply obtained with  $\lambda_{\mathrm{reg}} = 0$ .

Non-linear UOT. Finally, we discuss how our proposed reformulation of UOT can accommodate non-linear variants in which the linear term  $\langle C, T \rangle$  is replaced by a sparsity/robustness-promoting term, leading to problems of the form

$$
\operatorname {N L U O T} ^ {\boldsymbol {\lambda}} (\boldsymbol {a}, \boldsymbol {b}) = \min  _ {\boldsymbol {T} \geq 0} \sum_ {i, j} g \left(C _ {i, j} T _ {i, j}\right) + \lambda_ {1} D _ {\varphi} \left(\boldsymbol {T} \mathbb {1} _ {m}, \boldsymbol {a}\right) + \lambda_ {2} D _ {\varphi} \left(\boldsymbol {T} ^ {\top} \mathbb {1} _ {n}, \boldsymbol {b}\right) \tag {15}
$$

where  $g(\cdot)$  is a usually concave function, see, e.g., (Candes et al. 2008; Gasso et al. 2009). Our MM setting can readily accommodate such a formulation by majorizing the concave terms by their tangent. The non-linearity may improve robustness w.r.t outliers and better model realistic OT problems. For instance, in real life, the costs of transporting some goods between two places can be nonlinear due to economies of scale.

Broad and potential negative societal impact. The contributions in this paper are methodological and focus on a reformulation of a fundamental OT problem and adapting existing algorithms to solve it. In this sense, we bring more efficient solvers that run on GPU but this computational advantage can be counterbalanced by the possibility that it brings to be applied on larger datasets. The application of OT in domain adaptation has shown that it can be used to infer labels on samples/individuals when no labels are available, suggesting a capacity for violating user privacy. A potential application of UOT is the case where two datasets of users acquired by different methods contain some shared users. UOT can be used here to find correspondences between the users in the two datasets and also identify unique users in each dataset (those that do not receive mass).

# 6 Conclusion

In this paper, we reformulate the UOT problem as a non-negative penalized linear regression, allowing us to propose two new classes of algorithms. We first derive multiplicative algorithms for both KL and  $\ell_2$ -penalized UOT, providing numerical solutions that are fast and easy to implement. For the specific case of  $\ell_2$ -penalized UOT, we provide the first regularization path algorithm that computes the whole set of solutions for all the regularization parameter values. We finally build on the extensive literature in inverse problem and NMF to draw some fruitful perspectives on even more efficient algorithmic solutions or the definition of new OT problems.

# References

Arjovsky, M., S. Chintala, and L. Bottou (2017). Wasserstein Generative Adversarial Networks. In International Conference on Machine Learning, Volume 70, pp. 214-223.  
Benamou, J.-D. (2003). Numerical resolution of an "unbalanced" mass transport problem. *ESAIM: Mathematical Modelling and Numerical Analysis* 37(5), 851-868.  
Blondel, M., V. Seguy, and A. Rolet (2018). Smooth and Sparse Optimal Transport. In International Conference on Artificial Intelligence and Statistics, pp. 880-889.  
Bonneel, N. and D. Coeurjolly (2019). SPOT: Sliced Partial Optimal Transport. ACM Transactions on Graphics (SIGGRAPH) 38(4).  
Byrd, R. H., P. Lu, J. Nocedal, and C. Zhu (1995). A Limited Memory Algorithm for Bound-Constrained Optimization. SIAM Journal on Scientific Computing 16(5), 1190-1208.  
Caffarelli, L. A. and R. J. McCann (2010). Free boundaries in Optimal Transport and Monge-Ampère obstacle problems. Annals of Mathematics 171(2), 673-730.  
Candes, E. J., M. B. Wakin, and S. P. Boyd (2008). Enhancing Sparsity by Reweighted  $\ell_1$  Minimization. Journal of Fourier analysis and applications 14(5-6), 877-905.  
Chizat, L., G. Peyre, B. Schmitzer, and F.-X. Vialard (2018). Scaling algorithms for Unbalanced Optimal Transport problems. Mathematics of Computation 87(314), 2563-2609.  
Courty, N., R. Flamary, D. Tuia, and A. Rakotomamonjy (2017). Optimal Transport for Domain Adaptation. IEEE Transactions on Pattern Analysis and Machine Intelligence 39(9), 1853-1865.  
Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of Optimal Transport. Advances in Neural Information Processing Systems 26, 2292-2300.  
Dantas, C. F., E. Soubies, and C. Févotte (2021). Safe Screening for Sparse Regression with the Kullback-Leibler Divergence. In IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 5544-5548.  
De Pierro, A. R. (1993). On the relation between the ISRA and the EM algorithm for positron emission tomography. IEEE transactions on Medical Imaging 12(2), 328-333.  
Defazio, A., F. Bach, and S. Lacoste-Julien (2014). SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives. In Advances in Neural Information Processing Systems.  
Dhillon, I. S. and S. Sra (2005). Generalized nonnegative matrix approximations with Bregman divergences. In Advances in Neural Information Processing Systems, Volume 18.  
Efron, B., T. Hastie, I. Johnstone, and R. Tibshirani (2004). Least Angle Regression. Annals of statistics 32(2), 407-499.  
El Ghaoui, L., V. Viallon, and T. Rabbani (2012). Safe Feature Elimination for the LASSO and Sparse Supervised Learning Problems. Pacific Journal of Optimization 8(667-698).  
Févotte, C. and J. Idier (2011). Algorithms for nonnegative matrix factorization with the  $\beta$ -divergence. Neural computation 23(9), 2421-2456.  
Figalli, A. (2010). The Optimal Partial Transport Problem. Archive for Rational Mechanics and Analysis 195(2), 533-560.  
Frogner, C., C. Zhang, H. Mobahi, M. Araya, and T. A. Poggio (2015). Learning with a Wasserstein Loss. In Advances in Neural Information Processing System, pp. 2053-2061.  
Gasso, G., A. Rakotomamonjy, and S. Canu (2009). Recovering sparse signals with a certain family of nonconvex penalties and DC programming. IEEE Transactions on Signal Processing 57(12), 4686-4698.

Hastie, T., S. Rosset, R. Tibshirani, and J. Zhu (2004). The entire regularization path for the Support Vector Machine. Journal of Machine Learning Research 5, 1391-1415.  
Ho, N., X. L. Nguyen, M. Yurochkin, H. H. Bui, V. Huynh, and D. Phung (2017). Multilevel Clustering via Wasserstein Means. In International Conference on Machine Learning, Volume 70, pp. 1501-1509.  
Hoyer, P. O. (2002). Non-negative sparse coding. In IEEE Workshop on Neural Networks for Signal Processing, pp. 557-565.  
Hunter, D. R. and K. Lange (2004). A tutorial on MM algorithms. The American Statistician 58(1), 30-37.  
Kantorovich, L. (1942). On the transfer of masses (in Russian). Doklady Akademii Nauk 2, 227-229.  
Kusner, M., Y. Sun, N. Kolkin, and K. Weinberger (2015). From word embeddings to document distances. In International Conference on Machine Learning, pp. 957-966.  
LeCun, Y., C. Cortes, and C. Burges (2010). MNIST handwritten digit database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist.  
Lee, D. and H. Seung (1999). Learning the parts of objects by non-negative matrix factorization. Nature 401(6755), 788-791.  
Lee, D. and H. Seung (2001). Algorithms for Non-negative Matrix Factorization. In Advances in Neural Information Processing Sytems, Volume 13.  
Liero, M., A. Mielke, and G. Savaré (2018). Optimal entropy-transport problems and a new Hellinger-Kantorovich distance between positive measures. Inventiones mathematicae 211(3), 969–1117.  
Mairal, J. and B. Yu (2012). Complexity Analysis of the Lasso Regularization Path. In International Conference on Machine Learning, pp. 1835-1842.  
Maretic, H. P., M. E. Gheche, G. Chierchia, and P. Frossard (2019). GOT: An Optimal Transport framework for Graph comparison. In Advances In Neural Information Processing Systems, Volume 32.  
Massias, M., A. Gramfort, and J. Salmon (2018). Celer: a Fast Solver for the Lasso with Dual Extrapolation. In International Conference on Machine Learning, Volume 80, pp. 3321-3330.  
Mukherjee, D., A. Guha, J. Solomon, Y. Sun, and M. Yurochkin (2020). Outlier-Robust Optimal Transport. Technical report, arXiv preprint arXiv:2012.07363.  
Nesterov, Y. (2014). Subgradient methods for huge-scale optimization problems. Mathematical Programming 146(1), 275-297.  
Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12, 2825-2830.  
Rabin, J., S. Ferradans, and N. Papadakis (2014). Adaptive color transfer with relaxed optimal transport. In IEEE International Conference on Image Processing, pp. 4852-4856.  
Richardson, W. H. (1972). Bayesian-based iterative method of image restoration. *JoSA* 62(1), 55-59.  
Sato, R., M. Yamada, and H. Kashima (2020). Fast Unbalanced Optimal Transport on a Tree. In Advances in Neural Information Processing Systems, Volume 33.  
Sejourné, T., J. Feydy, F.-X. Vialard, A. Trouve, and G. Peyré (2019). Sinkhorn divergences for Unbalanced Optimal Transport. arXiv preprint arXiv:1910.12958.  
Sinkhorn, R. and P. Knopp (1967). Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics 21(2), 343-348.  
Sun, Y., P. Babu, and D. P. Palomar (2017). Majorization-minimization algorithms in signal processing, communications, and machine learning. IEEE Transactions on Signal Processing 65(3), 794-816.

Thibault, A., L. Chizat, C. Dossal, and N. Papadakis (2021). Overrelaxed Sinkhorn-Knopp Algorithm for Regularized Optimal Transport. Algorithms 14(5), 143.  
Vayer, T., L. Chapel, R. Flamary, R. Tavenard, and N. Courty (2019). Optimal transport for structured data with application on graphs. In International Conference on Machine Learning, pp. 6275-6284.  
Villani, C. (2009). Optimal Transport: Old and New, Volume 338. Springer Berlin Heidelberg.  
Virtanen, P., R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, S. J. van der Walt, M. Brett, J. Wilson, K. J. Millman, N. Mayorov, A. R. J. Nelson, E. Jones, R. Kern, E. Larson, C. J. Carey, I. Polat, Y. Feng, E. W. Moore, J. VanderPlas, D. Laxalde, J. Perktold, R. Cirmrnan, I. Henriksen, E. A. Quintero, C. R. Harris, A. M. Archibald, A. H. Ribeiro, F. Pedregosa, P. van Mulbregt, and SciPy 1.0 Contributors (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods 17, 261-272.  
Wang, J., P. Wonka, and J. Ye (2015). Lasso screening rules via dual polytope projection. Journal of Machine Learning Research 16(1), 1063-1101.  
Xiao, H., K. Rasul, and R. Vollgraf (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.  
Xie, Y., X. Wang, R. Wang, and H. Zha (2020). A fast proximal point method for computing exact Wasserstein distance. In Uncertainty in Artificial Intelligence, pp. 433-453.  
Yang, Z. and E. Oja (2011). Unified Development of Multiplicative Algorithms for Linear and Quadratic Nonnegative Matrix Factorization. IEEE Transactions on Neural Networks 22, 1878 - 1891.
