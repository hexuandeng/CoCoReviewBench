# FEATURE-ROBUST OPTIMAL TRANSPORT FOR HIGH-DIMENSIONAL DATA

Anonymous authors

Paper under double-blind review

# ABSTRACT

Optimal transport is a machine learning problem with applications including distribution comparison, feature selection, and generative adversarial networks. In this paper, we propose feature-robust optimal transport (FROT) for high-dimensional data, which solves high-dimensional OT problems using feature selection to avoid the curse of dimensionality. Specifically, we find a transport plan with discriminative features. To this end, we formulate the FROT problem as a min-max optimization problem. We then propose a convex formulation of the FROT problem and solve it using a Frank-Wolfe-based optimization algorithm, whereby the subproblem can be efficiently solved using the Sinkhorn algorithm. Since FROT finds the transport plan from selected features, it is robust to noise features. To show the effectiveness of FROT, we propose using the FROT algorithm for the layer selection problem in deep neural networks for semantic correspondence. By conducting synthetic and benchmark experiments, we demonstrate that the proposed method can find a strong correspondence by determining important layers. We show that the FROT algorithm achieves state-of-the-art performance in real-world semantic correspondence datasets.

# 1 INTRODUCTION

Optimal transport (OT) is a machine learning problem with several applications in the computer vision and natural language processing communities. The applications include Wasserstein distance estimation (Peyre et al., 2019), domain adaptation (Yan et al., 2018), multitask learning (Janati et al., 2019), barycenter estimation (Cuturi & Doucet, 2014), semantic correspondence (Liu et al., 2020), feature matching (Sarlin et al., 2019), and photo album summarization (Liu et al., 2019). The OT problem is extensively studied in the computer vision community as the earth mover's distance (EMD) (Rubner et al., 2000). However, the computational cost of EMD is cubic and highly expensive. Recently, the entropic regularized EMD problem was proposed; this problem can be solved using the Sinkhorn algorithm with a quadratic cost (Cuturi, 2013). Owing to the development of the Sinkhorn algorithm, researchers have replaced the EMD computation with its regularized counterparts. However, the optimal transport problem for high-dimensional data has remained unsolved for many years.

Recently, a robust variant of the OT was proposed for high-dimensional OT problems and used for divergence estimation (Paty & Cuturi, 2019; 2020). In the robust OT framework, the transport plan is computed with the discriminative subspace of the two data matrices  $\mathbf{X} \in \mathbb{R}^{d \times n}$  and  $\mathbf{Y} \in \mathbb{R}^{d \times m}$ . The subspace can be obtained using dimensionality reduction. An advantage of the subspace robust approach is that it does not require prior information about the subspace. However, given prior information such as feature groups, we can consider a computationally efficient formulation. The computation of the subspace can be expensive if the dimensionality of data is high, for example,  $10^4$ .

One of the most common prior information items is a feature group. The use of group features is popular in feature selection problems in the biomedical domain and has been extensively studied in Group Lasso (Yuan & Lin, 2006). The key idea of Group Lasso is to prespecify the group variables and select the set of group variables using the group norm (also known as the sum of  $\ell_2$  norms). For example, if we use a pretrained neural network as a feature extractor and compute OT using the features, then we require careful selection of important layers to compute OT. Specifically, each

![](images/1e89fe69e7682c8147fcfbbd265a44fc53eb59a3e52dc0a5828bc64f9ddee33f.jpg)  
(a) OT on clean data.

![](images/4151ec1be6dbc8e1276b7f76f80d8525370f38f4fe923080bf85f0053badd700.jpg)  
(b) OT on noisy data.  
Figure 1: transport plans between two synthetic distributions with 10-dimensional vectors  $\widetilde{\pmb{x}} = (\pmb{x}^{\top},\pmb{z}_{x}^{\top})$ ,  $\widetilde{\pmb{y}} = (\pmb{y}^{\top},\pmb{z}_{y}^{\top})$ , where two-dimensional vectors  $\pmb{x}\sim N(\pmb{\mu}_x,\pmb{\Sigma}_x)$  and  $\pmb{y}\sim N(\pmb{\mu}_y,\pmb{\Sigma}_y)$  are true features; and  $\pmb{z}_{x}\sim N(\mathbf{0}_{8},\mathbf{I}_{8})$  and  $\pmb{z}_{y}\sim N(\mathbf{0}_{8},\mathbf{I}_{8})$  are noisy features. (a) OT between distribution  $\pmb{x}$  and  $\pmb{y}$  is a reference. (b) OT between distribution  $\widetilde{\pmb{x}}$  and  $\widetilde{\pmb{y}}$ . (c) FROT transport plan between distribution  $\widetilde{\pmb{x}}$  and  $\widetilde{\pmb{y}}$  where true features and noisy features are grouped, respectively.

![](images/bd60fa374a8026e13cb20d48ff1c6c31944f11e8ae2cc927160ab8aaef930b14.jpg)  
(c) FROT on noisy data  $(\eta = 1)$ .

layer output is regarded as a grouped input. Therefore, using a feature group as prior information is a natural setup and is important for considering OT for deep neural networks (DNNs).

In this paper, we propose a high-dimensional optimal transport method by utilizing prior information in the form of grouped features. Specifically, we propose a feature-robust optimal transport (FROT) problem, for which we select distinct group feature sets to estimate a transport plan instead of determining its distinct subsets, as proposed in (Paty & Cuturi, 2019; 2020). To this end, we formulate the FROT problem as a min-max optimization problem and transform it into a convex optimization problem, which can be accurately solved using the Frank-Wolfe algorithm (Frank & Wolfe, 1956; Jaggi, 2013). The FROT's subproblem can be efficiently solved using the Sinkhorn algorithm (Cuturi, 2013). An advantage of FROT is that it can yield a transport plan from high-dimensional data using feature selection, using which the significance of the features is obtained without any additional cost. Therefore, the FROT formulation is highly suited for high-dimensional OT problems. Through synthetic experiments, we initially demonstrate that the proposed FROT is robust to noise dimensions (See Figure 1). Furthermore, we apply FROT to a semantic correspondence problem (Liu et al., 2020) and show that the proposed algorithm achieves SOTA performance.

# 2 BACKGROUND

In this section, we briefly introduce the OT problem.

Optimal transport (OT): The following are given: independent and identically distributed (i.i.d.) samples  $\mathbf{X} = \{\pmb{x}_i\}_{i=1}^n \in \mathbb{R}^{d \times n}$  from a  $d$ -dimensional distribution  $p$ , and i.i.d. samples  $\mathbf{Y} = \{\pmb{y}_j\}_{j=1}^m \in \mathbb{R}^{d \times m}$  from the  $d$ -dimensional distribution  $q$ . In the Kantorovich relaxation of OT, admissible couplings are defined by the set of the transport plan:

$$
\boldsymbol {U} (\mu , \nu) = \left\{\Pi \in \mathbb {R} _ {+} ^ {n \times m}: \Pi \mathbf {1} _ {m} = \boldsymbol {a}, \Pi^ {\top} \mathbf {1} _ {n} = \boldsymbol {b} \right\},
$$

where  $\Pi \in \mathbb{R}_+^{n\times m}$  is called the transport plan,  $\mathbf{1}_n$  is the  $n$ -dimensional vector whose elements are ones, and  $\pmb{a} = (a_{1},a_{2},\dots,a_{n})^{\top}\in \mathbb{R}_{+}^{n}$  and  $\pmb {b} = (b_{1},b_{2},\dots,b_{m})^{\top}\in \mathbb{R}_{+}^{m}$  are the weights. The OT problem between two discrete measures  $\mu = \sum_{i = 1}^{n}a_{i}\delta_{\boldsymbol{x}_{i}}$  and  $\nu = \sum_{j = 1}^{m}b_{j}\delta_{\boldsymbol{y}_{j}}$  determines the optimal transport plan of the following problem:

$$
\min  _ {\boldsymbol {\Pi} \in \boldsymbol {U} (\mu , \nu)} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \pi_ {i j} c \left(\boldsymbol {x} _ {i}, \boldsymbol {y} _ {j}\right), \tag {1}
$$

where  $c(\pmb{x}, \pmb{y})$  is a cost function. For example, the squared Euclidean distance is used, that is,  $c(\pmb{x}, \pmb{y}) = \| \pmb{x} - \pmb{y} \|_2^2$ . To solve the OT problem, Eq. (1) (also known as the earth mover's distance) using linear programming requires  $O(n^3)$ , ( $n = m$ ) computation, which is computationally expensive. To address this, an entropic-regularized optimal transport is used (Cuturi, 2013).

$$
\min _ {\boldsymbol {\Pi} \in \boldsymbol {U} (\mu , \nu)} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \pi_ {i j} c (\boldsymbol {x} _ {i}, \boldsymbol {y} _ {j}) + \epsilon H (\boldsymbol {\Pi}),
$$

where  $\epsilon \geq 0$  is the regularization parameter, and  $H(\Pi) = \sum_{i=1}^{n} \sum_{j=1}^{m} \pi_{ij} (\log(\pi_{ij}) - 1)$  is the entropic regularization. If  $\epsilon = 0$ , then the regularized OT problem reduces to the EMD problem. Owing to entropic regularization, the entropic regularized OT problem can be accurately solved using Sinkhorn iteration (Cuturi, 2013) with a  $O(nm)$  computational cost (See Algorithm 1).

Wasserstein distance: If the cost function is defined as  $c(\pmb{x}, \pmb{y}) = d(\pmb{x}, \pmb{y})$  with  $d(\pmb{x}, \pmb{y})$  as a distance function and  $p \geq 1$ , then we define the  $p$ -Wasserstein distance of two discrete measures  $\mu = \sum_{i=1}^{n} a_i \delta_{\pmb{x}_i}$  and  $\nu = \sum_{j=1}^{m} b_j \delta_{\pmb{y}_j}$  as

$$
W _ {p} (\mu , \nu) = \left(\min  _ {\boldsymbol {\Pi} \in \boldsymbol {U} (\mu , \nu)} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \pi_ {i j} d \left(\boldsymbol {x} _ {i}, \boldsymbol {y} _ {j}\right) ^ {p}\right) ^ {1 / p}.
$$

Recently, a robust variant of the Wasserstein distance, called the subspace robust Wasserstein distance (SRW), was proposed (Paty & Cuturi, 2019). The SRW computes the OT problem in the discriminative subspace. This can be determined by solving dimensionality-reduction problems. Owing to the robustness, it can compute the Wasserstein from noisy data. The SRW is given as

$$
\operatorname {S R W} (\mu , \nu) = \left(\min  _ {\boldsymbol {\Pi} \in \boldsymbol {U} (\mu , \nu)} \max  _ {\boldsymbol {U} \in \mathbb {R} ^ {d \times k}, \boldsymbol {U} ^ {\top} \boldsymbol {U} = \boldsymbol {I} _ {k}} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \pi_ {i j} \| \boldsymbol {U} ^ {\top} \boldsymbol {x} _ {i} - \boldsymbol {U} ^ {\top} \boldsymbol {y} _ {j} \| _ {2} ^ {2}\right) ^ {\frac {1}{2}}, \tag {2}
$$

where  $\pmb{U}$  is the projection matrix with  $k\leq d$  , and  $\pmb {I}_k\in \mathbb{R}^{k\times k}$  is the identity matrix. The SRW or its relaxed problem can be efficiently estimated using either eigenvalue decomposition or the Frank-Wolfe algorithm.

# 3 PROPOSED METHOD

This paper proposes FROT. We assume that the vectors are grouped as  $\pmb{x} = (x^{(1)}^{\top},\dots,x^{(L)}^{\top})^{\top}$  and  $\pmb{y} = (\pmb{y}^{(1)}^{\top},\dots,\pmb{y}^{(L)}^{\top})^{\top}$ . Here,  $\pmb{x}^{(\ell)}\in \mathbb{R}^{d_{\ell}}$  and  $\pmb{y}^{(\ell)}\in \mathbb{R}^{d_{\ell}}$  are the  $d_{\ell}$  dimensional vectors, where  $\sum_{\ell = 1}^{L}d_{\ell} = d$ . This setting is useful if we know the explicit group structure for the feature vectors a priori. In an application in  $L$ -layer neural networks, we consider  $\pmb{x}^{(\ell)}$  and  $\pmb{y}^{(\ell)}$  as outputs of the  $\ell$ th layer of the network. If we do not have a priori information, we can consider each feature independently (i.e.,  $d_{1} = d_{2} = \ldots = d_{L} = 1$  and  $L = d$ ). All proofs in this section are provided in the Appendix.

# 3.1 FEATURE-ROBUST OPTIMAL TRANSPORT (FROT)

The FROT formulation is given by

$$
\min  _ {\boldsymbol {\Pi} \in \boldsymbol {U} (\mu , \nu)} \max  _ {\boldsymbol {\alpha} \in \boldsymbol {\Sigma} ^ {L}} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \pi_ {i j} \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} c \left(\boldsymbol {x} _ {i} ^ {(\ell)}, \boldsymbol {y} _ {j} ^ {(\ell)}\right), \tag {3}
$$

where  $\boldsymbol{\Sigma}^{L} = \{\pmb{\alpha} \in \mathbb{R}_{+}^{L} : \pmb{\alpha}^{\top}\mathbf{1}_{L} = 1\}$  is the probability simplex. The underlying concept of FROT is to estimate the transport plan  $\Pi$  using distinct groups with large distances between  $\{\pmb{x}_i^{(\ell)}\}_{i=1}^n$  and  $\{\pmb{y}_j^{(\ell)}\}_{j=1}^m$ . We note that determining the transport plan in nondistinct groups is difficult because the data samples in  $\{\pmb{x}_i^{(\ell)}\}_{i=1}^n$  and  $\{\pmb{y}_j^{(\ell)}\}_{j=1}^m$  overlap. By contrast, in distinct groups,  $\{\pmb{x}_i^{(\ell)}\}_{i=1}^n$  and  $\{\pmb{y}_j^{(\ell)}\}_{j=1}^m$  are different, and this aids in determining an optimal transport plan. This is an intrinsically similar idea to the subspace robust Wasserstein distance (Paty & Cuturi, 2019), which estimates the transport plan in the discriminative subspace, while our approach selects important groups. Therefore, FROT can be regarded as a feature selection variant of the vanilla OT problem in Eq. (1), whereas the subspace robust version uses dimensionality-reduction counterparts.

# Algorithm 1 Sinkhorn algorithm.

1: Input:  $a, b, C, \epsilon, t_{max}$  
2: Initialize  $\mathbf{K} = e^{-\mathbf{C} / \epsilon}, \mathbf{u} = \mathbf{1}_n, \mathbf{v} = \mathbf{1}_m, t = 0$  
3: while  $t \leq t_{max}$  and not converge do  
4:  $\pmb{u} = \pmb{a} / (\pmb{K}\pmb{v})$  
5:  $\pmb {v} = \pmb {b} / (\pmb{K}^{\top}\pmb{u})$  
6:  $t = t + 1$  
7: end while  
8: return  $\Pi = \mathrm{diag}(\pmb {u})\pmb{K}\mathrm{diag}(\pmb {v})$

# Algorithm 2 FROT with the Frank-Wolfe.

1: Input:  $\{\pmb{x}_i\}_{i=1}^n, \{\pmb{y}_j\}_{j=1}^m, \eta,$  and  $\epsilon$ .  
2: Initialize  $\Pi$ , compute  $\{C_{\ell}\}_{\ell = 1}^{L}$ .  
3: for  $t = 0 \dots T$  do  
4:  $\widehat{\Pi} = \operatorname*{argmin}_{\mathbf{\Pi}\in \mathbf{U}(\mu ,\nu)}\langle \mathbf{\Pi},M_{\mathbf{\Pi}^{(t)}}\rangle +$ $\epsilon H(\Pi)$  
5:  $\pmb{\Pi}^{(t + 1)} = (1 - \gamma)\pmb{\Pi}^{(t)} + \gamma \widehat{\pmb{\Pi}}$  
6: with  $\gamma = \frac{2}{2 + t}$ .  
7: end for  
8: return  $\Pi^{(T)}$

Using FROT, we can define a  $p$ -feature robust Wasserstein distance ( $p$ -FRWD).

Proposition 1 For the distance function  $d(\pmb{x},\pmb{y})$

$$
\operatorname {F R W D} _ {p} (\mu , \nu) = \left(\min  _ {\boldsymbol {\Pi} \in U (\mu , \nu)} \max  _ {\boldsymbol {\alpha} \in \boldsymbol {\Sigma} ^ {L}} \left. \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \pi_ {i j} \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} d \left(\boldsymbol {x} _ {i} ^ {(\ell)}, \boldsymbol {y} _ {j} ^ {(\ell)}\right) ^ {p}\right) ^ {1 / p}, \right. \tag {4}
$$

is a distance for  $p \geq 1$ .

Note that we can show that 2-FRWD is a special case of SRW with  $d(\pmb{x}, \pmb{y}) = \| \pmb{x} - \pmb{y} \|_2$  (See Appendix). The key difference between SRW and FRWD is that FRWD can use any distance, while SRW can only use  $d(\pmb{x}, \pmb{y}) = \| \pmb{x} - \pmb{y} \|_2$ .

# 3.2 FROT OPTIMIZATION

Here, we propose two FROT algorithms based on the Frank-Wolfe algorithm and linear programming.

Frank-Wolfe: We propose a continuous variant of the FROT algorithm using the Frank-Wolfe algorithm, which can be fully differentiable. To this end, we introduce entropic regularization for  $\alpha$  and rewrite the FROT as a function of  $\Pi$ . Therefore, we solve the following problem for  $\alpha$ :

$$
\min _ {\boldsymbol {\Pi} \in \boldsymbol {U} (\mu , \nu)} \max _ {\boldsymbol {\alpha} \in \boldsymbol {\Sigma} ^ {L}} J _ {\eta} (\boldsymbol {\Pi}, \boldsymbol {\alpha}), \text {w i t h} J _ {\eta} (\boldsymbol {\Pi}, \boldsymbol {\alpha}) = \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \pi_ {i j} \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} c (\boldsymbol {x} _ {i} ^ {(\ell)}, \boldsymbol {y} _ {j} ^ {(\ell)}) - \eta H (\boldsymbol {\alpha}),
$$

where  $\eta \geq 0$  is the regularization parameter, and  $H(\alpha) = \sum_{\ell=1}^{L} \alpha_{\ell} (\log(\alpha_{\ell}) - 1)$  is the entropic regularization for  $\alpha$ . An advantage of entropic regularization is that the nonnegative constraint is naturally satisfied, and the entropic regularizer is a strong convex function.

Lemma 2 The optimal solution of the optimization problem

$$
\boldsymbol {\alpha} ^ {*} = \operatorname * {a r g m a x} _ {\boldsymbol {\alpha} \in \boldsymbol {\Sigma} ^ {L}} J _ {\eta} (\boldsymbol {\Pi}, \boldsymbol {\alpha}), \text {w i t h} J _ {\eta} (\boldsymbol {\Pi}, \boldsymbol {\alpha}) = \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} \phi_ {\ell} - \eta H (\boldsymbol {\alpha})
$$

with a fixed admissible transport plan  $\Pi \in U(\mu, \nu)$ , is given by

$$
\alpha_ {\ell} ^ {*} = \frac {\exp \left(\frac {1}{\eta} \phi_ {\ell}\right)}{\sum_ {\ell^ {\prime} = 1} ^ {L} \exp \left(\frac {1}{\eta} \phi_ {\ell^ {\prime}}\right)} \text {w i t h} J _ {\eta} (\boldsymbol {\Pi}, \boldsymbol {\alpha} ^ {*}) = \eta \log \left(\sum_ {\ell = 1} ^ {L} \exp \left(\frac {1}{\eta} \phi_ {\ell}\right)\right) + \eta .
$$

Using Lemma 2 (or Lemma 4 in Nesterov (2005)) together with the setting  $\phi_{\ell} = \sum_{i=1}^{n} \sum_{j=1}^{m} \pi_{ij} c(\pmb{x}_i^{(\ell)}, \pmb{y}_i^{(\ell)}) = \langle \pmb{\Pi}, \pmb{C}_{\ell} \rangle$ ,  $[C_{\ell}]_{ij} = c(\pmb{x}_i^{(\ell)}, \pmb{y}_i^{(\ell)})$ , the global problem is equivalent to

$$
\min  _ {\boldsymbol {\Pi} \in \boldsymbol {U} (\mu , \nu)} G _ {\eta} (\boldsymbol {\Pi}), \text {w i t h} G _ {\eta} (\boldsymbol {\Pi}) = \eta \log \left(\sum_ {\ell = 1} ^ {L} \exp \left(\frac {1}{\eta} \langle \boldsymbol {\Pi}, \boldsymbol {C} _ {\ell} \rangle\right)\right). \tag {5}
$$

Note that this is known as a smoothed max-operator (Nesterov, 2005; Blondel et al., 2018). Specifically, regularization parameter  $\eta$  controls the "smoothness" of the maximum.

Proposition 3  $G_{\eta}(\Pi)$  is a convex function relative to  $\Pi$

The derived optimization problem of FROT is convex. Therefore, we can determine globally optimal solutions. Note that the SRW optimization problem is not jointly convex (Paty & Cuturi, 2019) for the projection matrix and the transport plan. In this study, we employ the Frank-Wolfe algorithm (Frank & Wolfe, 1956; Jaggi, 2013), using which we approximate  $G_{\eta}(\Pi)$  with linear functions at  $\Pi^{(t)}$  and move  $\Pi$  toward the optimal solution in the convex set (See Algorithm 2).

The derivative of the loss function  $G_{\eta}(\Pi)$  at  $\Pi^{(t)}$  is given by

$$
\frac {\partial G _ {\eta} (\boldsymbol {\Pi})}{\partial \boldsymbol {\Pi}} \Bigg | _ {\boldsymbol {\Pi} = \boldsymbol {\Pi} ^ {(t)}} = \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} ^ {(t)} \boldsymbol {C} _ {\ell} = \boldsymbol {M} _ {\boldsymbol {\Pi} ^ {(t)}} \mathrm {w i t h} \alpha_ {\ell} ^ {(t)} = \frac {\exp \left(\frac {1}{\eta} \langle \boldsymbol {\Pi} ^ {(t)} , \boldsymbol {C} _ {\ell} \rangle\right)}{\sum_ {\ell^ {\prime} = 1} ^ {L} \exp \left(\frac {1}{\eta} \langle \boldsymbol {\Pi} ^ {(t)} , \boldsymbol {C} _ {\ell^ {\prime}} \rangle\right)}.
$$

Then, we update the transport plan by solving the EMD problem:

$$
\boldsymbol {\Pi} ^ {(t + 1)} = (1 - \gamma) \boldsymbol {\Pi} ^ {(t)} + \gamma \widehat {\boldsymbol {\Pi}} \text {w i t h} \widehat {\boldsymbol {\Pi}} = \operatorname * {a r g m i n} _ {\boldsymbol {\Pi} \in \boldsymbol {U} (\mu , \nu)} \langle \boldsymbol {\Pi}, \boldsymbol {M} _ {\boldsymbol {\Pi} ^ {(t)}} \rangle ,
$$

where  $\gamma = 2 / (2 + k)$ . Note that  $M_{\Pi^{(t)}}$  is given by the weighted sum of the cost matrices. Thus, we can utilize multiple features to estimate the transport plan  $\Pi$  for the relaxed problem in Eq. (5).

Using the Frank-Wolfe algorithm, we can obtain the optimal solution. However, solving the EMD problem requires a cubic computational cost that can be expensive if  $n$  and  $m$  are large. To address this, we can solve the regularized OT problem, which requires  $O(nm)$ . We denote the Frank-Wolfe algorithm with EMD as FW-EMD and the Frank-Wolfe algorithm with Sinkhorn as FW-Sinkhorn.

Computational complexity: The proposed method depends on the Sinkhorn algorithm, which requires an  $O(nm)$  operation. The computation of the cost matrix in each subproblem needs an  $O(Lnm)$  operation, where  $L$  is the number of groups. Therefore, the entire complexity is  $O(TLnm)$ , where  $T$  is the number of Frank-Wolfe iterations (in general,  $T = 10$  is sufficient).

Proposition 4 For each  $t \geq 1$ , the iteration  $\Pi^{(t)}$  of Algorithm 2 satisfies

$$
G _ {\eta} \left(\boldsymbol {\Pi} ^ {(t)}\right) - G _ {\eta} \left(\boldsymbol {\Pi} ^ {*}\right) \leq \frac {4 \sigma_ {m a x} \left(\boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi}\right)}{\eta (t + 2)} (1 + \delta),
$$

where  $\sigma_{max}(\Phi^{\top}\Phi)$  is the largest eigenvalue of the matrix  $\Phi^{\top}\Phi$  and  $\Phi = (\operatorname{vec}(C_1),\operatorname{vec}(C_2),\ldots,\operatorname{vec}(C_L))^{\top}$ ; and  $\delta \geq 0$  is the accuracy to which internal linear subproblems are solved.

Based on Proposition 4, the number of iterations depends on  $\eta$ ,  $\epsilon$ , and the number of groups. If we set a small  $\eta$ , convergence requires more time. In addition, if we use entropic regularization with a large  $\epsilon$ , the  $\delta$  in Proposition 4 can be large. Finally, if we use more groups, the largest eigenvalue of the matrix  $\Phi^{\top}\Phi$  can be larger. Note that the constant term of the upper bound is large; however, the Frank-Wolfe algorithm converges quickly in practice.

Linear Programming: Because  $\lim_{\eta \to 0^{+}}G_{\eta}(\Pi) = \max_{\ell \in \{1,2,\dots,L\}}\sum_{i = 1}^{n}\sum_{j = 1}^{m}\pi_{ij}c(\pmb{x}_i^{(\ell)},\pmb{y}_j^{(\ell)})$  the FROT problem can also be written as

$$
\min  _ {\boldsymbol {\Pi} \in \boldsymbol {U} (\mu , \nu)} \max  _ {\ell \in \{1, 2, \dots , L \}} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \pi_ {i j} c \left(\boldsymbol {x} _ {i} ^ {(\ell)}, \boldsymbol {y} _ {j} ^ {(\ell)}\right). \tag {6}
$$

Because the objective is the max of linear functions, it is convex with respect to  $\Pi$ . We can solve the problem via linear programming:

$$
\min  _ {\boldsymbol {\Pi} \in \boldsymbol {U} (\mu , \nu), t} t, \quad \text {s . t .} \quad \langle \boldsymbol {\Pi}, \boldsymbol {C} _ {\ell} \rangle \leq t, \ell = 1, 2, \dots , L. \tag {7}
$$

This optimization can be easily solved using an off-the-shelf LP package. However, the computational cost of this LP problem is high in general (i.e.,  $O(n^{3}), n = m$ ).

# 3.3 APPLICATION: SEMANTIC CORRESPONDENCE

We applied our proposed FROT algorithm to semantic correspondence. The semantic correspondence is a problem that determines the matching of objects in two images. That is, given input image pairs  $(A,B)$ , with common objects, we formulated the semantic correspondence problem to estimate the transport plan from the key points in  $A$  to those in  $B$ ; this framework was proposed in (Liu et al., 2020). In Figure 2, we show an overview of our proposed framework.

Cost matrix computation  $C_{\ell}$ : In our framework, we employed a pretrained convolutional neural network to extract dense feature maps for each convolutional layer. The dense feature map of the  $\ell$ th layer output of the  $s$ th image is given by

$$
\boldsymbol {f} _ {s, q + (r - 1) h _ {s}} ^ {(\ell , s)} \in \mathbb {R} ^ {d _ {\ell}}, q = 1, 2, \dots , h _ {s}, r = 1, 2, \dots , w _ {s}, \ell = 1, 2, \dots , L,
$$

where  $w_{s}$  and  $h_s$  are the width and height of the  $s$ th image, respectively, and  $d_{\ell}$  is the dimension of the  $\ell$ th layer's feature map. Note that because the dimension of the dense feature map is different for each layer, we sample feature maps to the size of the 1st layer's feature map size (i.e.,  $h_s\times w_s$ ).

The  $\ell$ th layer's cost matrix for images  $s$  and  $s'$  is given by

$$
[ \pmb {C} _ {\ell} ] _ {i j} = \| \pmb {f} _ {i} ^ {(\ell , s)} - \pmb {f} _ {j} ^ {(\ell , s ^ {\prime})} \| _ {2} ^ {2}, i = 1, 2, \dots , w _ {s} h _ {s}, j = 1, 2, \dots , w _ {s ^ {\prime}} h _ {s ^ {\prime}}.
$$

A potential problem with FROT is that the estimation depends significantly on the magnitude of the cost of each layer (also known as a group). Hence, normalizing each cost matrix is important. Therefore, we normalized each feature vector by  $\pmb{f}_i^{(\ell ,s)}\gets \pmb {f}_i^{(\ell ,s)} / \| \pmb {f}_i^{(\ell ,s)}\| _2$  Consequently, the cost matrix is given by  $[C_{\ell}]_{ij} = 2 - 2f_i^{(\ell ,s)^{\top}}f_j^{(\ell ,s')}$  We can use distances such as the  $L1$  distance.

![](images/d1de1a0816516bc5059e3cd230ce098a3c6945ee27195e69c302ab83f83f25b9.jpg)  
Figure 2: Semantic correspondence framework based on FROT.

staircase re-weighting: For semantic correspondence, setting  $\pmb{a} \in \mathbb{R}^{h_s w_s}$  and  $\pmb{b} \in \mathbb{R}^{h_{s'} w_{s'}}$  is important because semantic correspondence can be affected by background clutter. Therefore, we generated the class activation maps (Zhou et al., 2016) for the source and target images and used them as  $\pmb{a}$  and  $\pmb{b}$ , respectively. For CAM, we chose the class with the highest classification probability and normalized it to the range [0, 1].

# 4 RELATED WORK

OT algorithms: The Wasserstein distance can be determined by solving the OT problem. An advantage of the Wasserstein distance is its robustness to noise; moreover, we can obtain the transport plan, which is useful for many machine learning applications. To reduce the computation cost for the Wasserstein distance, the sliced Wasserstein distance is useful (Kolouri et al., 2016). Recently, a tree variant of the Wasserstein distance was proposed (Evans & Matsen, 2012; Le et al., 2019; Sato et al., 2020); the sliced Wasserstein distance is a special case of this algorithm.

In addition to accelerating the computation, structured optimal transport incorporates structural information directly into OT problems (Alvarez-Melis et al., 2018). Specifically, they formulate the submodular optimal transport problem and solve the problem using a saddle-point mirror prox algorithm. Recently, more complex structured information was introduced in the OT problem, including the hierarchical structure (Alvarez-Melis et al., 2020; Yurochkin et al., 2019). These approaches successfully incorporate structured information into OT problems with respect to data samples. By contrast, FROT incorporates the structured information into features.

The approach most closely related to FROT is a robust variant of the Wasserstein distance, called the subspace robust Wasserstein distance (SRW) (Paty & Cuturi, 2019). SRW computes the OT problem

![](images/b628ea939809837c2d88a6346f1858f10da32eb1236c3a7b225c8fb3b5b4165b.jpg)  
(a) Objective score.

![](images/6c48fed3100814fff03f828b84d3e347328499fdfa724fc2964b10eb4bfd51ea.jpg)  
(b) MSE  $(\eta)$ .  
Figure 3: (a) Objective scores for LP, FW-EMD, and FW-Sinkhorn. (b) MSE between transport plan of LP and FW-EMD and that with LP and FW-Sinkhorn with different  $\eta$ . (c) MSE between transport plan of LP and FW-Sinkhorn with different  $\epsilon$ .

![](images/d964faaf13052962adaa3f6a5126a1484bd873abef03735a68f5c51004120748.jpg)  
(c) MSE  $(\epsilon)$ .

in a discriminative subspace; this is possible by solving dimensionality-reduction problems. Owing to the robustness, SRW can successfully compute the Wasserstein distance from noisy data. The max-sliced Wasserstein distance (Deshpande et al., 2019) and its generalized counterpart (Kolouri et al., 2019) can also be regarded as subspace-robust Wasserstein methods. Note that SRW (Paty & Cuturi, 2019) is a min-max based approach, while the max-sliced Wasserstein distances (Deshpande et al., 2019; Kolouri et al., 2019) are max-min approaches. The FROT is a feature selection variant of the Wasserstein distance, whereas the subspace approaches are used for dimensionality reduction.

As a parallel work, a general minimax optimal transport problem called the robust Kantorovich problem (RKP) was recently proposed (Dhouib et al., 2020). RKP involves using a cutting-set method for a general minmax optimal transport problem that includes the FROT problem as a special case. The approaches are technically similar; however, our problem and that of Dhouib et al. (2020) are intrinsically different. Specifically, we aim to solve a high-dimensional OT problem using feature selection and apply it to semantic correspondence problems, while the RKP approach focuses on providing a general framework and uses it for color transformation problems. As a technical difference, the cutting-set method may not converge to an optimal solution if we use the regularized OT (Dhouib et al., 2020). By contrast, because we use a Frank-Wolfe algorithm, our algorithm converges to a true objective function with regularized OT solvers. The multiobjective optimal transport (MOT) is an approach (Scetbon et al., 2020) parallel to ours. The key difference between FROT and MOT is that MOT tries to use the weighted sum of cost functions, while FROT considers the worst case. Moreover, as applications, we focus on the cost matrices computed from subsets of features, while MOT considers cost matrices with different distance functions.

# 5 EXPERIMENTS

# 5.1 SYNTHETIC DATA

We compare FROT with a standard OT using synthetic datasets. In these experiments, we initially generate two-dimensional vectors  $\pmb{x} \sim N(\pmb{\mu}_x, \pmb{\Sigma}_x)$  and  $\pmb{y} \sim N(\pmb{\mu}_y, \pmb{\Sigma}_y)$ . Here, we set  $\pmb{\mu}_x = (-5,0)^{\top}$ ,  $\pmb{\mu}_y = (5,0)^{\top}$ ,  $\pmb{\Sigma}_x = \pmb{\Sigma}_y = ((5,1)^{\top}, (4,1)^{\top})$ . Then, we concatenate  $\pmb{z}_x \sim N(\pmb{0}_8, \pmb{I}_8)$  and  $\pmb{z}_y \sim N(\pmb{0}_8, \pmb{I}_8)$  to  $\pmb{x}$  and  $\pmb{y}$ , respectively, to give  $\widetilde{\pmb{x}} = (\pmb{x}^{\top}, \pmb{z}_x^{\top})$ ,  $\widetilde{\pmb{y}} = (\pmb{y}^{\top}, \pmb{z}_y^{\top})$ .

For FROT, we set  $\eta = 1.0$  and the number of iterations of the Frank-Wolfe algorithm as  $T = 10$ . The regularization parameter is set to  $\epsilon = 0.02$  for all methods. To show the proof-of-concept, we set the true features as a group and the remaining noise features as another group.

Fig. 1a shows the correspondence from  $x$  and  $y$  with the vanilla OT algorithm. Figs. 1b and 1c show the correspondence of FROT and OT with  $\widetilde{x}$  and  $\widetilde{y}$ , respectively. Although FROT can identify a suitable matching, the OT fails to obtain a significant correspondence. We observed that the  $\alpha$  parameter corresponding to a true group is  $\alpha_{1} = 0.9999$ . Moreover, we compared the objective scores of the FROT with LP, FW-EMD, and FW-Sinkhorn  $(\epsilon = 0.1)$ . Figure 3a shows the objective scores of FROTs with the different solvers, and both FW-EMD and FW-Sinkhorn can achieve almost the same objective score with a relatively small  $\eta$ . Moreover, Figure 3b shows the mean squared error between the LP method and the FW counterparts. Similar to the objective score cases, it can yield a similar transport plan with a relatively small  $\eta$ . Finally, we evaluated the FW-Sinkhorn by changing

Table 1: Per-class PCK  $(\alpha_{bbox} = 0.1)$  results using SPair-71k. All models use ResNet101. The numbers in the bracket of SRW are the input layer indices.  

<table><tr><td colspan="2">Methods</td><td>aero</td><td>bike</td><td>bird</td><td>boat</td><td>bottle</td><td>bus</td><td>car</td><td>cat</td><td>chair</td><td>cow</td><td>dog</td><td>horse</td><td>motor</td><td>person</td><td>plant</td><td>sheep</td><td>train</td><td>tv</td><td>all</td></tr><tr><td rowspan="4">SPair-71k finetuned models</td><td>CNNGeo (Rocco et al., 2017)</td><td>23.4</td><td>16.7</td><td>40.2</td><td>14.3</td><td>36.4</td><td>27.7</td><td>26.0</td><td>32.7</td><td>12.7</td><td>27.4</td><td>22.8</td><td>13.7</td><td>20.9</td><td>21.0</td><td>17.5</td><td>10.2</td><td>30.8</td><td>34.1</td><td>20.6</td></tr><tr><td>A2Net (Hongsuck Seo et al., 2018)</td><td>22.6</td><td>18.5</td><td>42.0</td><td>16.4</td><td>37.9</td><td>30.8</td><td>26.5</td><td>35.6</td><td>13.3</td><td>29.6</td><td>24.3</td><td>16.0</td><td>21.6</td><td>22.8</td><td>20.5</td><td>13.5</td><td>31.4</td><td>36.5</td><td>22.3</td></tr><tr><td>WeakAlign (Rocco et al., 2018a)</td><td>22.2</td><td>17.6</td><td>41.9</td><td>15.1</td><td>38.1</td><td>27.4</td><td>27.2</td><td>31.8</td><td>12.8</td><td>26.8</td><td>22.6</td><td>14.2</td><td>20.0</td><td>22.2</td><td>17.9</td><td>10.4</td><td>32.2</td><td>35.1</td><td>20.9</td></tr><tr><td>NC-Net (Rocco et al., 2018b)</td><td>17.9</td><td>12.2</td><td>32.1</td><td>11.7</td><td>29.0</td><td>19.9</td><td>16.1</td><td>39.2</td><td>9.9</td><td>23.9</td><td>18.8</td><td>15.7</td><td>17.4</td><td>15.9</td><td>14.8</td><td>9.6</td><td>24.2</td><td>31.1</td><td>20.1</td></tr><tr><td rowspan="3">SPair-71k validation</td><td>HPF (Min et al., 2019a)</td><td>25.2</td><td>18.9</td><td>52.1</td><td>15.7</td><td>38.0</td><td>22.8</td><td>19.1</td><td>52.9</td><td>17.9</td><td>33.0</td><td>32.8</td><td>20.6</td><td>24.4</td><td>27.9</td><td>21.1</td><td>15.9</td><td>31.5</td><td>35.6</td><td>28.2</td></tr><tr><td>OT-HPF (Liu et al., 2020)</td><td>32.6</td><td>18.9</td><td>62.5</td><td>20.7</td><td>42.0</td><td>26.1</td><td>20.4</td><td>61.4</td><td>19.7</td><td>41.3</td><td>41.7</td><td>29.8</td><td>29.6</td><td>31.8</td><td>25.0</td><td>23.5</td><td>44.7</td><td>37.0</td><td>33.9</td></tr><tr><td>FROT(η=0.2,ε=0.4)</td><td>35.1</td><td>20.3</td><td>59.8</td><td>21.1</td><td>42.9</td><td>27.7</td><td>21.2</td><td>63.5</td><td>18.8</td><td>39.7</td><td>37.9</td><td>29.2</td><td>28.8</td><td>29.9</td><td>28.2</td><td>24.3</td><td>52.1</td><td>39.5</td><td>34.7</td></tr><tr><td rowspan="7">Without SPair-71k validation</td><td>OT</td><td>30.1</td><td>16.5</td><td>50.4</td><td>17.3</td><td>38.0</td><td>22.9</td><td>19.7</td><td>54.3</td><td>17.0</td><td>28.4</td><td>31.3</td><td>22.1</td><td>28.0</td><td>19.5</td><td>21.0</td><td>17.8</td><td>42.6</td><td>28.8</td><td>28.3</td></tr><tr><td>FROT(η=0.3)</td><td>35.0</td><td>20.9</td><td>56.3</td><td>23.4</td><td>40.7</td><td>27.2</td><td>21.9</td><td>62.0</td><td>17.5</td><td>38.8</td><td>36.2</td><td>27.9</td><td>28.0</td><td>30.4</td><td>26.9</td><td>23.1</td><td>49.7</td><td>38.4</td><td>33.7</td></tr><tr><td>FROT(η=0.5)</td><td>34.1</td><td>18.8</td><td>56.9</td><td>19.9</td><td>40.0</td><td>25.6</td><td>19.2</td><td>61.9</td><td>17.4</td><td>38.7</td><td>36.5</td><td>25.6</td><td>26.9</td><td>27.2</td><td>26.3</td><td>22.1</td><td>50.3</td><td>38.6</td><td>32.8</td></tr><tr><td>FROT(η=0.7)</td><td>33.4</td><td>19.4</td><td>56.6</td><td>20.0</td><td>39.6</td><td>26.1</td><td>19.1</td><td>62.4</td><td>17.9</td><td>38.0</td><td>36.5</td><td>26.0</td><td>27.5</td><td>26.5</td><td>25.5</td><td>21.6</td><td>49.7</td><td>38.9</td><td>32.7</td></tr><tr><td>SRW (layers = {1, 32–34})</td><td>29.4</td><td>14.0</td><td>43.7</td><td>15.6</td><td>33.8</td><td>21.0</td><td>17.6</td><td>48.0</td><td>12.9</td><td>23.3</td><td>26.5</td><td>19.8</td><td>25.5</td><td>17.6</td><td>16.7</td><td>15.2</td><td>37.1</td><td>20.5</td><td>24.5</td></tr><tr><td>SRW (layers = {1, 31–34})</td><td>29.7</td><td>14.3</td><td>44.3</td><td>15.7</td><td>34.2</td><td>21.3</td><td>17.8</td><td>48.5</td><td>13.1</td><td>23.6</td><td>27.1</td><td>20.0</td><td>25.8</td><td>18.1</td><td>16.9</td><td>15.2</td><td>37.3</td><td>21.0</td><td>24.8</td></tr><tr><td>SRW (layers = {1, 30–34})</td><td>29.8</td><td>14.7</td><td>45.6</td><td>15.9</td><td>34.8</td><td>21.5</td><td>18.0</td><td>49.3</td><td>13.3</td><td>24.0</td><td>27.7</td><td>20.6</td><td>25.7</td><td>18.7</td><td>17.2</td><td>15.3</td><td>37.7</td><td>21.5</td><td>25.2</td></tr></table>

the regularization parameter  $\eta$ . In this experiment, we set  $\eta = 1$  and varied the  $\epsilon$  values. The result shows that we can obtain an accurate transport plan with a relatively small  $\epsilon$ .

# 5.2 SEMANTIC CORRESPONDENCE

We evaluated our FROT algorithm for semantic correspondence. In this study, we used the SPair-71k (Min et al., 2019b). The SPair-71k dataset consists of 70,958 image pairs with variations in viewpoint and scale. For evaluation, we employed a percentage of accurate key points (PCK), which counts the number of accurately predicted key points given a fixed threshold (Min et al., 2019b). All semantic correspondence experiments were run on a Linux server with NVIDIA P100.

For the optimal transport based frameworks, we employed ResNet101 (He et al., 2016) pretrained on ImageNet (Deng et al., 2009) for feature and activation map extraction. The ResNet101 consists of 34 convolutional layers and the entire number of features is  $d = 32,576$ . Note that we did not fine-tune the network. We compared the proposed method with several baselines (Min et al., 2019b) and the SRW<sup>1</sup>. Owing to the computational cost and the required memory size for SRW, we used the first and the last few convolutional layers of ResNet101 as the input of SRW. In our experiments, we empirically set  $T = 3$  and  $\epsilon = 0.1$  for FROT and SRW, respectively. For SRW, we set the number of latent dimension as  $k = 50$  for all experiments. HPF (Min et al., 2019a) and OT-HPF (Liu et al., 2020) are state-of-the-art methods for semantic correspondence. HPF and OT-HPF required the validation dataset to select important layers, whereas SRW and FROT did not require the validation dataset. OT is a simple optimal transport-based method that does not select layers.

Table 1 lists the per-class PCK results obtained using the SPair-71k dataset. FROT  $(\eta = 0.3)$  outperforms most existing baselines, including HPF and OT. Moreover, FROT  $(\eta = 0.3)$  is consistent with OT-HPF (Liu et al., 2020), which requires the validation dataset to select important layers. In this experiment, setting  $\eta < 1$  results in favorable performance (See Table 3 in the Appendix). The computational costs of FROT is 0.29, while SRWs are 8.73, 11.73, 15.76, respectively. Surprisingly, FROT outperformed SRWs. However, this is mainly due to the used input layers. Therefore, scaling up SRW would be an interesting future work.

We further evaluated FROT by tuning hyperparameters  $\eta$  and  $\epsilon$  using validation sets, where the maximum search ranges for  $\eta$  and  $\epsilon$  are set to 0.2 to 2.0 and 0.1 to 0.6 with intervals of 0.1, respectively. Figure 6 in Appendix shows the average PCK scores for  $(\eta, \epsilon)$  pairs on the validation split of SPair-71k. By using hyperparameter search, we selected  $(\eta = 0.2, \epsilon = 0.4)$  as an optimal parameter. The FROT with optimal parameters outperforms the state-of-the-art method (Liu et al., 2020).

# 6 CONCLUSION

In this paper, we proposed FROT for high-dimensional data. This approach jointly solves feature selection and OT problems. An advantage of FROT is that it is a convex optimization problem and can determine an accurate globally optimal solution using the Frank-Wolfe algorithm. We used FROT for high-dimensional feature selection and semantic correspondence problems. Through extensive experiments, we demonstrated that the proposed algorithm is consistent with state-of-the-art algorithms in both feature selection and semantic correspondence.

# REFERENCES

David Alvarez-Melis, Tommi Jaakkola, and Stefanie Jegelka. Structured optimal transport. In AISTATS, 2018.  
David Alvarez-Melis, Youssef Mroueh, and Tommi S Jaakkola. Unsupervised hierarchy matching with optimal transport over hyperbolic spaces. AISTATS, 2020.  
Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In ICML, 2017.  
Mathieu Blondel, Vivien Seguy, and Antoine Rolet. Smooth and sparse optimal transport. In AISTATS, 2018.  
Charlotte Bunne, David Alvarez-Melis, Andreas Krause, and Stefanie Jegelka. Learning generative models across incomparable spaces. In ICML, 2019.  
Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In NIPS, 2013.  
Marco Cuturi and Arnaud Doucet. Fast computation of Wasserstein barycenters. ICML, 2014.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.  
Ishan Deshpande, Yuan-Ting Hu, Ruoyu Sun, Ayis Pyrros, Nasir Siddiqui, Sanmi Koyejo, Zhizhen Zhao, David Forsyth, and Alexander G Schwing. Max-sliced Wasserstein distance and its use for GANs. In CVPR, 2019.  
Sofien Dhouib, Ievgen Redko, Tanguy Kerdoncuff, Rémi Emonet, and Marc Sebban. A swiss army knife for minimax optimal transport. In ICML, 2020.  
Steven N Evans and Frederick A Matsen. The phylogenetic kantorovich-rubinstein metric for environmental sequence samples. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 74(3):569-592, 2012.  
Marguerite Frank and Philip Wolfe. An algorithm for quadratic programming. Naval research logistics quarterly, 3(1-2):95-110, 1956.  
Bolin Gao and Lacra Pavel. On the properties of the softmax function with application in game theory and reinforcement learning. arXiv preprint arXiv:1704.00805, 2017.  
Arthur. Gretton, Kenji. Fukumizu, C. Hui. Teo, Le. Song, Bernhard. Schölkopf, and Alex Smola. A kernel statistical test of independence. In NIPS, 2007.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.  
Paul Hongsuck Seo, Jongmin Lee, Deunsol Jung, Bohyung Han, and Minsu Cho. Attentive semantic alignment with offset-aware correlation kernels. In ECCV, 2018.  
Martin Jaggi. Revisiting frank-wolfe: Projection-free sparse convex optimization. In ICML, 2013.  
Hicham Janati, Marco Cuturi, and Alexandre Gramfort. Wasserstein regularization for sparse multitask regression. In AISTATS, 2019.  
Soheil Kolouri, Yang Zou, and Gustavo K Rohde. Sliced wasserstein kernels for probability distributions. In CVPR, 2016.  
Soheil Kolouri, Kimia Nadjahi, Umut Simsekli, Roland Badeau, and Gustavo Rohde. Generalized sliced Wasserstein distances. In NeurIPS, 2019.  
Tam Le, Makoto Yamada, Kenji Fukumizu, and Marco Cuturi. Tree-sliced approximation of wasserstein distances. NeurIPS, 2019.

Yanbin Liu, Makoto Yamada, Yao-Hung Hubert Tsai, Tam Le, Ruslan Salakhutdinov, and Yi Yang. Lsmi-sinkhorn: Semi-supervised squared-loss mutual information estimation with optimal transport. arXiv preprint arXiv:1909.02373, 2019.  
Yanbin Liu, Linchao Zhu, Makoto Yamada, and Yi Yang. Semantic correspondence as an optimal transport problem. In CVPR, 2020.  
Juhong Min, Jongmin Lee, Jean Ponce, and Minsu Cho. Hyperpixel flow: Semantic correspondence with multi-layer neural features. In ICCV, 2019a.  
Juhong Min, Jongmin Lee, Jean Ponce, and Minsu Cho. Spair-71k: A large-scale benchmark for semantic correspondence. arXiv preprint arXiv:1908.10543, 2019b.  
Yu Nesterov. Smooth minimization of non-smooth functions. Mathematical programming, 103(1): 127-152, 2005.  
François-Pierre Paty and Marco Cuturei. Subspace robust Wasserstein distances. In ICML, 2019.  
François-Pierre Paty and Marco Cuturi. Regularized optimal transport is ground cost adversarial. ICML, 2020.  
Gabriel Peyré, Marco Cuturi, et al. Computational optimal transport. Foundations and Trends in Machine Learning, 11(5-6):355-607, 2019.  
Ignacio Rocco, Relja Arandjelovic, and Josef Sivic. Convolutional neural network architecture for geometric matching. In CVPR, 2017.  
Ignacio Rocco, Relja Arandjelović, and Josef Sivic. End-to-end weakly-supervised semantic alignment. In CVPR, 2018a.  
Ignacio Rocco, Mircea Cimpoi, Relja Arandjelović, Akihiko Torii, Tomas Pajdla, and Josef Sivic. Neighbourhood consensus networks. In NeurIPS, 2018b.  
Yossi Rubner, Carlo Tomasi, and Leonidas J Guibas. The earth mover's distance as a metric for image retrieval. International journal of computer vision, 40(2):99-121, 2000.  
Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. SuperGlue: Learning feature matching with graph neural networks. arXiv preprint arXiv:1911.11763, 2019.  
Ryoma Sato, Makoto Yamada, and Hisashi Kashima. Fast unbalanced optimal transport on tree. In NeurIPS, 2020.  
Meyer Sctbon, Laurent Meunier, Jamal Atif, and Marco Cuturi. Handling multiple costs in optimal transport: Strong duality and efficient computation. arXiv preprint arXiv:2006.07260, 2020.  
Hongteng Xu, Dixin Luo, and Lawrence Carin. Scalable gromov-wasserstein learning for graph partitioning and matching. arXiv preprint arXiv:1905.07645, 2019a.  
Hongteng Xu, Dixin Luo, Hongyuan Zha, and Lawrence Carin Duke. Gromov-wasserstein learning for graph matching and node embedding. In ICML, 2019b.  
Yuguang Yan, Wen Li, Hanrui Wu, Huaqing Min, Mingkui Tan, and Qingyao Wu. Semi-supervised optimal transport for heterogeneous domain adaptation. In IJCAI, 2018.  
Ming Yuan and Yi Lin. Model selection and estimation in regression with grouped variables. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 68(1):49-67, 2006.  
Mikhail Yurochkin, Sebastian Claici, Edward Chien, Farzaneh Mirzazadeh, and Justin M Solomon. Hierarchical optimal transport for document representation. In NeurIPS, 2019.  
Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. Learning deep features for discriminative localization. In CVPR, 2016.
