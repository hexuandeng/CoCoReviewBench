# Distributed Distributionally Robust Optimization with Non-Convex Objectives

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Distributionally Robust Optimization (DRO), which aims to find an optimal decision that minimizes the worst case cost over the ambiguity set of probability distribution, has been widely applied in diverse applications, e.g., network behavior analysis, risk management, etc. However, existing DRO techniques face three key challenges: 1) how to deal with the asynchronous updating in a distributed environment; 2) how to leverage the prior distribution effectively; 3) how to properly adjust the degree of robustness according to different scenarios. To this end, we propose an asynchronous distributed algorithm, named Asynchronous Single-looP alternIve gRadian project (ASPIRE) algorithm with the itErative Active SEt method (EASE) to tackle the distributed distributionally robust optimization (DDRO) problem. Furthermore, a new uncertainty set, i.e., constrained  $D$ -norm uncertainty set, is developed to effectively leverage the prior distribution and flexibly control the degree of robustness. Finally, our theoretical analysis elucidates that the proposed algorithm is guaranteed to converge and the iteration complexity is also analyzed. Extensive empirical studies on real-world datasets demonstrate that the proposed method can not only achieve fast convergence, and remain robust against data heterogeneity as well as malicious attacks, but also tradeoff robustness with performance.

# 1 Introduction

The past decade has witnessed the proliferation of smartphones and Internet of Things (IoT) devices, which generate a plethora of data everyday. Centralized machine learning requires gathering the data to a particular server to train models which incurs high communication overhead [41] and suffers privacy risks [38]. As a remedy, distributed machine learning methods have been proposed. Considering a distributed system composed of  $N$  devices (workers), we denote the dataset of these workers as  $\{D_1,\dots ,D_N\}$ . For the  $j^{\mathrm{th}}$ $(1\leq j\leq N)$  worker, the labeled dataset is given as  $D_{j} = \{\mathbf{x}_{j}^{i},y_{j}^{i}\}$ , where  $\mathbf{x}_j^i\in \mathbb{R}^d$  and  $y_{j}^{i}\in \{1,\dots ,c\}$  denote the  $i^{\mathrm{th}}$  data sample and the corresponding label, respectively. The distributed learning tasks can be formulated as the following optimization problem,

$$
\min  _ {\boldsymbol {w} \in \boldsymbol {\mathcal {W}}} F (\boldsymbol {w}) \quad \text {w i t h} \quad F (\boldsymbol {w}) := \sum_ {j} f _ {j} (\boldsymbol {w}), \tag {1}
$$

where  $\pmb{w} \in \mathbb{R}^p$  is the model parameter to be learned and  $\mathcal{W} \subseteq \mathbb{R}^p$  is a nonempty closed convex set,  $f_j(\cdot)$  is the empirical risk over the  $j^{\mathrm{th}}$  worker involving only the local data:

$$
f _ {j} (\boldsymbol {w}) = \sum_ {i: \mathbf {x} _ {j} ^ {i} \in D _ {j}} \frac {1}{| D _ {j} |} \mathcal {L} _ {j} \left(\boldsymbol {x} _ {j} ^ {i}, y _ {j} ^ {i}; \boldsymbol {w}\right), \tag {2}
$$

where  $\mathcal{L}_j$  is the local objective function over the  $j^{\mathrm{th}}$  worker. Problem in Eq. (1) arises in numerous areas, such as distributed signal processing [18], multi-agent optimization [32], etc. However, such

Submitted to 36th Conference on Neural Information Processing Systems (NeurIPS 2022). Do not distribute.

problem does not consider the data heterogeneity [49, 35, 34, 27] among different workers (i.e., data distribution of workers could be substantially different from each other [39]). Indeed, it has been shown that traditional federated approaches, such as FedAvg [29], built for independent and identically distributed (IID) data may perform poorly when applied to Non-IID data [24]. This issue can be mitigated via learning a robust model that aims to achieve uniformly good performance over all workers by solving the following distributionally robust optimization (DRO) problem in a distributed manner:

$$
\min  _ {\boldsymbol {w} \in \mathcal {W}} \max  _ {\mathbf {p} \in \Omega \subseteq \Delta_ {N}} F (\boldsymbol {w}, \mathbf {p}) := \sum_ {j} p _ {j} f _ {j} (\boldsymbol {w}), \tag {3}
$$

where  $\mathbf{p} = [p_1, \dots, p_N] \in \mathbb{R}^N$  is the adversarial distribution in  $N$  workers, the  $j^{\mathrm{th}}$  entry in this vector, i.e.,  $p_j$  represents the adversarial distribution value for the  $j^{\mathrm{th}}$  worker.  $\Delta_N = \{\mathbf{p} \in \mathbb{R}_+^N : \mathbf{1}^\top \mathbf{p} = 1\}$  and  $\Omega$  is a subset of  $\Delta_N$ . Agnostic federated learning (AFL) [31] firstly introduces the distributionally robust (agnostic) loss in federated learning and provides the convergence rate for (strongly) convex functions. However, AFL does not discuss the setting of  $\Omega$ . DRFA-Prox [15] considers  $\Omega = \Delta_N$  and imposes a regularizer on adversarial distribution to leverage the prior distribution. Nevertheless, three key challenges have not yet been addressed by prior works. First, whether it is possible to construct an uncertainty framework that can not only flexibly maintain the trade-off between the model robustness and performance but also effectively leverage the prior distribution? Second, how to design asynchronous algorithms with guaranteed convergence? Compared to synchronous algorithms, the master in asynchronous algorithms can update its parameters after receiving updates from only a small subset of workers [50, 10]. Asynchronous algorithms are particularly desirable in practice since they can relax strict data dependencies and ensure convergence even in the presence of device failures [50]. Finally, whether it is possible to flexibly adjust the degree of robustness? Moreover, it is necessary to provide convergence guarantee when the objectives (i.e.,  $f_j(w_j), \forall j$ ) are non-convex.

To this end, we propose ASPIRE-EASE to effectively address the aforementioned challenges. Firstly, different from existing works, the prior distribution is incorporated within the constraint in our formulation, which can not only leverage the prior distribution more effectively but also achieve guaranteed feasibility for any adversarial distribution within the uncertainty set. The prior distribution can be obtained from side information or uniform distribution [36], which is necessary to construct the uncertainty (ambiguity) set and obtain a more robust model [15]. Specifically, we formulate the prior distribution informed distributionally robust optimization (PD-DRO) problem as:

$$
\min  _ {\boldsymbol {z} \in \boldsymbol {\mathcal {Z}}, \left\{\boldsymbol {w} _ {j} \in \boldsymbol {\mathcal {W}} \right\}} \max  _ {\boldsymbol {\mathbf {p}} \in \boldsymbol {\mathcal {P}}} \sum_ {j} p _ {j} f _ {j} (\boldsymbol {w} _ {j}) \tag {4}
$$

$$
\begin{array}{l} \text {s . t .} \quad \boldsymbol {z} = \boldsymbol {w} _ {j}, \forall 1 \leq j \leq N, \end{array}
$$

$$
\begin{array}{l l} \text {v a r i a b l e s} & \boldsymbol {z}, \boldsymbol {w} _ {1}, \boldsymbol {w} _ {2}, \dots , \boldsymbol {w} _ {N}, \end{array}
$$

where  $\pmb{z} \in \mathbb{R}^p$  is the global consensus variable,  $\pmb{w}_j \in \mathbb{R}^p$  is the local variable (local model parameter) of  $j^{\mathrm{th}}$  worker and  $\mathcal{Z} \subseteq \mathbb{R}^p$  is a nonempty closed convex set.  $\mathcal{P} \subseteq \mathbb{R}_+^N$  is the uncertainty (ambiguity) set of adversarial distribution  $\mathbf{p}$ , which is set based on the prior distribution. To solve the PD-DRO problem in an asynchronous manner, we first propose Asynchronous Single-looP alternatIve gRadiant projEction (ASPIRE), which employs simple gradient projection steps for the update of primal and dual variables at every iteration, thus is computationally efficient. Next, the itErative Active SET method (EASE) is employed to replace the traditional cutting plane method to improve the computational efficiency and speed up the convergence. We further provide the convergence guarantee for the proposed algorithm. Furthermore, a new uncertainty set, i.e., constrained  $D$ -norm ( $CD$ -norm), is proposed in this paper and its advantages include: 1) it can flexibly control the degree of robustness; 2) the resulting subproblem is computationally simple; 3) it can effectively leverage the prior distribution and flexibly set the bounds for every  $p_j$ .

Contributions. Our contributions can be summarized as follows:

1. We formulate a PD-DRO problem with  $CD$ -norm uncertainty set. PD-DRO incorporates the prior distribution as constraints and thus can guarantee robustness. In addition,  $CD$ -norm is developed to model the ambiguity set around the prior distribution and it provides a flexible way to control the trade-off between model robustness and performance.  
2. We develop a single-loop asynchronous algorithm, namely ASPIRE-EASE, to optimize PD-DRO in an asynchronous manner. ASPIRE employs simple gradient projection steps to update the variables at every iteration, which is computationally efficient. And EASE is proposed to replace cutting plane

method to enhance the computational efficiency and speed up the convergence. We demonstrate that even if the objectives  $f_{j}(\boldsymbol{w}_{j}), \forall j$  are non-convex, the proposed algorithm is guaranteed to converge. We also theoretically derive the iteration complexity of ASPIRE-EASE.  
3. Extensive empirical studies on four different real world datasets demonstrate the superior performance of the proposed algorithm. It is seen that ASPIRE-EASE can not only ensure the model's robustness against data heterogeneity but also mitigate malicious attacks.

# 2 Preliminaries

# 2.1 Distributionally Robust Optimization

Optimization problems often contain uncertain parameters. A small perturbation of the parameters could render the optimal solution of the original optimization problem infeasible or completely meaningless [4]. Distributionally robust optimization (DRO) [25, 16, 6] assumes that the probability distributions of uncertain parameters are unknown but remain in an ambiguity (uncertainty) set and aims to find a decision that minimizes the worst case expected cost over the ambiguity set, whose general form can be expressed as,

$$
\min  _ {\boldsymbol {x} \in \boldsymbol {X}} \max  _ {P \in \boldsymbol {\mathcal {P}}} \mathbb {E} _ {P} [ r (\boldsymbol {x}, \boldsymbol {\xi}) ], \tag {5}
$$

where  $\pmb{x} \in \pmb{\mathcal{X}}$  represents the decision variable,  $\mathcal{P}$  is the ambiguity set of probability distributions  $P$  of uncertain parameters  $\xi$ . Existing methods for solving DRO can be broadly grouped into two widely-used categories [37]: 1) Dual methods [14, 44, 17] reformulate the primal DRO problems as deterministic optimization problems through duality theory. Ben-Tal et al. [2] reformulate the robust linear optimization (RLO) problem with an ellipsoidal uncertainty set as a second-order cone optimization problem (SOCP). 2) Cutting plane methods [30, 5] (also called adversarial approaches [20]) continuously solve an approximate problem with a finite number of constraints of the primal DRO problem, and subsequently check whether new constraints are needed to refine the feasible set. Recently, several new methods [36, 26, 21] have been developed to solve DRO, which need to solve the inner maximization problem at every iteration.

# 2.2 Cutting Plane Method for PD-DRO

In this section, we introduce the cutting plane method for PD-DRO in Eq. (4). We first reformulate PD-DRO by introducing an additional variable  $h \in \mathcal{H}$  ( $\mathcal{H} \subseteq \mathbb{R}^1$  is a nonempty closed convex set) and protection function  $g(\{\pmb{w}_j\})$  [48]. In this case, Eq. (4) can be reformulated as the form with uncertainty in the constraints:

$$
\min  _ {\boldsymbol {z} \in \boldsymbol {\mathcal {Z}}, \left\{\boldsymbol {w} _ {j} \in \boldsymbol {\mathcal {W}} \right\}, h \in \boldsymbol {\mathcal {H}}} h \tag {6}
$$

$$
\begin{array}{l} \begin{array}{l} \text {s . t .} \sum_ {j} \bar {p} f _ {j} (\boldsymbol {w} _ {j}) + g (\{\boldsymbol {w} _ {j} \}) - h \leq 0, \end{array} \\ \boldsymbol {z} = \boldsymbol {w} _ {j}, \forall 1 \leq j \leq N, \\ \end{array}
$$

$$
\begin{array}{l l} \text {v a r i a b l e s} & z, w _ {1}, w _ {2}, \dots , w _ {N}, h. \end{array}
$$

where  $\overline{p}$  is the nominal value of the adversarial distribution for every worker and  $g(\{\boldsymbol{w}_j\}) = \max_{\mathbf{p} \in \mathcal{P}} \sum_{j} (p_j - \overline{p}) f_j(\boldsymbol{w}_j)$  is the protection function. Eq. (6) is a semi-infinite program (SIP) which contains infinite constraints and cannot be solved directly [37]. Denoting the cutting plane set in  $(t + 1)^{\mathrm{th}}$  iteration as  $\mathbf{A}^t \subseteq \mathbb{R}^N$ , we can utilize the following function to approximate  $g(\{\boldsymbol{w}_j\})$ :

$$
\bar {g} \left(\left\{\boldsymbol {w} _ {j} \right\}\right) = \max  _ {\boldsymbol {a} _ {l} \in \mathbf {A} ^ {t}} \boldsymbol {a} _ {l} ^ {\top} \mathbf {f} (\boldsymbol {w}) = \max  _ {\boldsymbol {a} _ {l} \in \mathbf {A} ^ {t}} \sum_ {j} a _ {l, j} f _ {j} \left(\boldsymbol {w} _ {j}\right), \tag {7}
$$

where  $\mathbf{a}_l = [a_{l,1},\dots ,a_{l,N}]\in \mathbb{R}^N$  denotes the  $l^{\mathrm{th}}$  cutting plane in cutting plane set  $\mathbf{A}^t$  and  $\mathbf{f}(\boldsymbol {w}) = [f_1(\boldsymbol {w}_1),\dots ,f_N(\boldsymbol {w}_N)]\in \mathbb{R}^N$ . Substituting the protection function  $g(\{\pmb {w}_j\})$  with  $\overline{g} (\{\pmb {w}_j\})$ , we can obtain the following approximate problem:

$$
\min  _ {\boldsymbol {z} \in \boldsymbol {\mathcal {Z}}, \left\{\boldsymbol {w} _ {j} \in \boldsymbol {\mathcal {W}} \right\}, h \in \boldsymbol {\mathcal {H}}} h \tag {8}
$$

$$
\begin{array}{l} \mathrm {s . t .} \sum_ {j} (\overline {{p}} + a _ {l, j}) f _ {j} (\boldsymbol {w} _ {j}) - h \leq 0, \forall \boldsymbol {a} _ {l} \in \mathbf {A} ^ {t}, \\ \boldsymbol {z} = \boldsymbol {w} _ {j}, \forall 1 \leq j \leq N, \\ \end{array}
$$

$$
\begin{array}{l l} \text {v a r i a b l e s} & z, w _ {1}, w _ {2}, \dots , w _ {N}, h. \end{array}
$$

# 3 ASPIRE

Distributed optimization is an attractive approach for large-scale learning tasks [7, 8] since it does not require data aggregation, which protects data privacy while also reducing bandwidth requirements [40]. When the neural network models (i.e.,  $f_{j}(\boldsymbol{w}_{j}), \forall j$  are non-convex functions) are used, solving Eq. (8) in a distributed manner facing two challenges. 1) Computing the optimal solution to a non-convex subproblem requires a large number of iterations and therefore is highly computationally intensive if not impossible. Thus, the traditional Alternating Direction Method of Multipliers (ADMM) is ineffective. 2) the communication delays of workers may differ significantly [11], thus, asynchronous algorithms are strongly preferred.

To this end, we propose the Asynchronous Single-looP alternatIve gRadiant projEction (ASPIRE). The advantages of the proposed algorithm include: 1) ASPIRE uses simple gradient projection steps to update primal variables in each iteration and therefore it is computationally more efficient than the traditional ADMM method, which seeks to find the optimal solution in non-convex (for  $\boldsymbol{w}_j, \forall j$ ) and convex (for  $z$  and  $h$ ) optimization subproblems every iteration, 2) the proposed asynchronous algorithm does not need strict synchronization among different workers. Therefore, ASPIRE remains resilient against communication delays and potential hardware failures from workers. Details of the algorithm are given below. Firstly, we define the node as master which is responsible for updating the global variable  $z$ , and we define the node which is responsible for updating the local variable  $w_j$  as worker  $j$ . In each iteration, the master updates its variables once it receives updates from at least  $S$  workers, e.g., active workers, satisfying  $1 \leq S \leq N$ .  $\mathbf{Q}^{t+1}$  denotes the index subset of workers from which the master receives updates during  $(t+1)^{\text{th}}$  iteration. We also assume the master will receive updated variables from every worker at least once for each  $\tau$  iterations. Then a batch of instances with size  $m$  are randomly sampled from each worker during each iteration. The loss function of these instances from  $j^{\text{th}}$  worker is given by  $\hat{f}_j(\boldsymbol{w}) = \sum_{i=1}^{m} \frac{1}{m} \mathcal{L}_j(\mathbf{x}_j^i, y_j^i; \boldsymbol{w})$ . The use of mini-batch loss avoids enumerating the whole dataset, which is more efficient [36]. It is evident that  $\mathbb{E}[\hat{f}_j(\boldsymbol{w})] = f_j(\boldsymbol{w})$  and  $\mathbb{E}[\nabla \hat{f}_j(\boldsymbol{w})] = \nabla f_j(\boldsymbol{w})$ . Thus, the augmented Lagrangian function of Eq. (8) can be written as:

$$
L _ {p} = h + \sum_ {l} \lambda_ {l} \left(\sum_ {j} (\bar {p} + a _ {l, j}) \hat {f} _ {j} \left(\boldsymbol {w} _ {j}\right) - h\right) + \sum_ {j} \phi_ {j} ^ {\top} \left(\boldsymbol {z} - \boldsymbol {w} _ {j}\right) + \sum_ {j} \frac {\kappa_ {1}}{2} | | \boldsymbol {z} - \boldsymbol {w} _ {j} | | ^ {2}, \tag {9}
$$

where  $L_{p} = L_{p}(\{\boldsymbol{w}_{j}\}, \boldsymbol{z}, h, \{\lambda_{l}\}, \{\phi_{j}\})$ ,  $\lambda_{l} \in \Lambda, \forall l$  and  $\phi_{j} \in \Phi, \forall j$  represent the dual variables of inequality and equality constraints in Eq. (8), respectively.  $\Lambda \subseteq \mathbb{R}^{1}$  and  $\Phi \subseteq \mathbb{R}^{p}$  are nonempty closed convex sets, constant  $\kappa_{1} > 0$  is a penalty parameter. Note that Eq. (9) does not consider the second-order penalty term for inequality constraint since it will invalidate the distributed optimization. Following [46], the regularized version of Eq. (9) is employed to update all variables as follows,

$$
\widetilde {L} _ {p} \left(\left\{\boldsymbol {w} _ {j} \right\}, \boldsymbol {z}, h, \left\{\lambda_ {l} \right\}, \left\{\phi_ {j} \right\}\right) = L _ {p} - \sum_ {l} \frac {c _ {1} ^ {t}}{2} \left| \left| \lambda_ {l} \right| \right| ^ {2} - \sum_ {j} \frac {c _ {2} ^ {t}}{2} \left| \left| \phi_ {j} \right| \right| ^ {2}, \tag {10}
$$

where  $c_1^t$  and  $c_2^t$  denote the regularization terms in  $(t + 1)^{\mathrm{th}}$  iteration. In  $(t + 1)^{\mathrm{th}}$  iteration, the proposed algorithm proceeds as follows.

1) Each worker updates the local variables  $\pmb{w}_j$  as follows,

$$
\boldsymbol {w} _ {j} ^ {t + 1} = \left\{ \begin{array}{l} \mathcal {P} _ {\boldsymbol {W}} \left(\boldsymbol {w} _ {j} ^ {t} - \eta_ {\boldsymbol {w}} \nabla_ {\boldsymbol {w} _ {j}} \widetilde {L} _ {p} \left(\left\{\boldsymbol {w} _ {j} ^ {\widetilde {t} _ {j}} \right\}, \boldsymbol {z} ^ {\widetilde {t} _ {j}}, h ^ {\widetilde {t} _ {j}}, \left\{\lambda_ {l} ^ {\widetilde {t} _ {j}} \right\}, \left\{\phi_ {j} ^ {\widetilde {t} _ {j}} \right\}\right)\right), \forall j \in \mathbf {Q} ^ {t + 1}, \\ \boldsymbol {w} _ {j} ^ {t}, \forall j \notin \mathbf {Q} ^ {t + 1}, \end{array} \right. \tag {11}
$$

where  $\eta_{\mathbf{w}}$  represents the step-size and  $P_{\mathcal{W}}$  represents the projection onto the closed convex set  $\mathcal{W}$  and we set  $\mathcal{W} = \{\pmb{w}_j||\pmb{w}_j||_\infty \leq \alpha_1\}$ ,  $\alpha_{1}$  is a positive constant. And  $\widetilde{t_j}$  is the last iteration during which worker  $j$  was active. It is seen that  $\pmb{w}_j^t = \pmb{w}_j^{\widetilde{t}_j}$  and  $\phi_j^t = \phi_j^{\widetilde{t}_j},\forall j\in \mathbf{Q}^{t + 1}$ . Then, the active workers transmit their local model parameters  $\pmb{w}_j^{t + 1}$  and loss  $\hat{f}_j(\pmb {w}_j)$  to the master.

2) After receiving the updates from active workers, the master updates the global consensus variable  $z$ , additional variable  $h$  and dual variables  $\lambda_{l}$  as follows,

$$
\boldsymbol {z} ^ {t + 1} = \mathcal {P} _ {\boldsymbol {Z}} \left(\boldsymbol {z} ^ {t} - \eta_ {\boldsymbol {z}} \nabla_ {\boldsymbol {z}} \widetilde {L} _ {p} \left(\left\{\boldsymbol {w} _ {j} ^ {t + 1} \right\}, \boldsymbol {z} ^ {t}, h ^ {t}, \left\{\lambda_ {l} ^ {t} \right\}, \left\{\phi_ {j} ^ {t} \right\}\right)\right), \tag {12}
$$

$$
h ^ {t + 1} = \mathcal {P} _ {\boldsymbol {\mathcal {H}}} \left(h ^ {t} - \eta_ {h} \nabla_ {h} \widetilde {L} _ {p} \left(\left\{\boldsymbol {w} _ {j} ^ {t + 1} \right\}, \boldsymbol {z} ^ {t + 1}, h ^ {t}, \left\{\lambda_ {l} ^ {t} \right\}, \left\{\phi_ {j} ^ {t} \right\}\right)\right), \tag {13}
$$

160

$$
\lambda_ {l} ^ {t + 1} = \mathcal {P} _ {\Lambda} \left(\lambda_ {l} ^ {t} + \rho_ {1} \nabla_ {\lambda_ {l}} \widetilde {L} _ {p} \left(\left\{\boldsymbol {w} _ {j} ^ {t + 1} \right\}, \boldsymbol {z} ^ {t + 1}, h ^ {t + 1}, \left\{\lambda_ {l} ^ {t} \right\}, \left\{\phi_ {j} ^ {t} \right\}\right)\right), \forall 1 \leq l \leq | \mathbf {A} ^ {t} |, \tag {14}
$$

where  $\eta_z, \eta_h$  and  $\rho_1$  represent the step-size.  $P_{\mathcal{Z}}$ ,  $P_{\mathcal{H}}$  and  $P_{\Lambda}$  respectively represent the projection onto the closed convex sets  $\mathcal{Z}$ ,  $\mathcal{H}$  and  $\Phi$ . We set  $\mathcal{Z} = \{\pmb{z} | ||\pmb{z}||_{\infty} \leq \alpha_1\}$ ,  $\mathcal{H} = \{h | 0 \leq h \leq \alpha_2\}$  and  $\Lambda = \{\lambda_l | 0 \leq \lambda_l \leq \alpha_3\}$ , where  $\alpha_2$  and  $\alpha_3$  are positive constants. Then, master broadcasts  $\pmb{z}^{t+1}$ ,  $h^{t+1}$ ,  $\lambda_l^{t+1}$ ,  $\forall 1 \leq l \leq |\mathbf{A}^t|$  to the active workers.

165 3) Each worker updates the dual variables  $\phi_j$  as follows,

$$
\phi_ {j} ^ {t + 1} = \left\{ \begin{array}{l} \mathcal {P} _ {\Phi} \left(\phi_ {j} ^ {t} + \rho_ {2} \nabla_ {\phi_ {j}} \widetilde {L} _ {p} \left(\left\{\boldsymbol {w} _ {j} ^ {t + 1} \right\}, \boldsymbol {z} ^ {t + 1}, h ^ {t + 1}, \left\{\lambda_ {l} ^ {t + 1} \right\}, \left\{\phi_ {j} ^ {t} \right\}\right)\right), \forall j \in \mathbf {Q} ^ {t + 1}, \\ \phi_ {j} ^ {t}, \forall j \notin \mathbf {Q} ^ {t + 1}, \end{array} \right. \tag {15}
$$

where  $\rho_{2}$  represents the step-size and  $P_{\Phi}$  represents the projection onto the closed convex set  $\Phi$  and we set  $\Phi = \{\phi_j||\phi_j||_{\infty}\leq \alpha_4\}$ ,  $\alpha_{4}$  is a positive constant. It is seen that the projection operation in each step is computationally simple since the closed convex sets have simple structures [3].

# 169 4 Iterative Active Set Method

Cutting plane methods may give rise to numerous linear constraints and lots of extra message passing [48]. Moreover, more iterations are required to obtain the  $\varepsilon$ -stationary point when the size of a set containing cutting planes increases (which corresponds to a larger  $M$ ), which can be seen in Theorem 1. To improve the computational efficiency and speed up the convergence, we consider removing the inactive cutting planes. The proposed itErative Active SEt method (EASE) can be divided into the two steps: 1) solving the cutting plane generation subproblem to generate cutting plane, and 2) removing the inactive cutting plane every  $k$  iterations, where  $k > 0$  is a pre-set constant. The setting of  $k$  can be controlled flexibly and we can set  $k$  as a relatively large number.

The cutting planes are generated according to the uncertainty set. For example, if we employ ellipsoid uncertainty set, the cutting plane is generated via solving a SOCP. In this paper, we propose  $CD$ -norm uncertainty set, which can be expressed as follows,

$$
\mathcal {P} = \left\{\mathbf {p}: - \widetilde {p} _ {j} \leq p _ {j} - q _ {j} \leq \widetilde {p} _ {j}, \sum_ {j} \right| \frac {p _ {j} - q _ {j}}{\widetilde {p} _ {j}} | \leq \Gamma , \mathbf {1} ^ {\top} \mathbf {p} = 1 \}, \tag {16}
$$

where  $\Gamma \in \mathbb{R}^1$  can flexibly control the level of robustness,  $\mathbf{q} = [q_1,\dots ,q_N]\in \mathbb{R}^N$  represents the prior distribution,  $-\widetilde{p}_j$  and  $\widetilde{p}_j$  ( $\widetilde{p}_j\geq 0$ ) represent the lower and upper bounds for  $p_j - q_j$ , respectively. The setting of  $\mathbf{q}$  and  $\widetilde{p}_j,\forall j$  are based on the prior knowledge.  $D$ -norm is a classical uncertainty set (which is also called as budget uncertainty set) [4]. We call Eq. (16) as  $CD$ -norm uncertainty set since  $\mathbf{p}$  in our problem also needs to satisfy the constraint  $\mathbf{1}^{\top}\mathbf{p} = 1$ . We claim that  $L_{1}$ -norm (or twice total variation distance) uncertainty set is closely related to  $CD$ -norm uncertainty set. Nevertheless, there are two differences: 1)  $CD$ -norm uncertainty set could be regarded as a weighted  $L_{1}$ -norm with additional constraints. 2)  $CD$ -norm uncertainty set can flexibly set the lower and upper bounds for every  $p_j$  (i.e.,  $q_j - \widetilde{p}_j\leq p_j\leq p_j + \widetilde{p}_j$ ), while  $0\leq p_j\leq 1,\forall j$  in  $L_{1}$ -norm uncertainty set. Based on the  $CD$ -norm uncertainty set, the cutting plane can be derived as follows,

1) Solve the following problem,

$$
\mathbf {p} ^ {t + 1} = \underset {p _ {1}, \dots , p _ {N}} {\arg \max } \sum_ {j} (p _ {j} - \bar {p}) \hat {f} _ {j} (\boldsymbol {w} _ {j})
$$

$$
\text {s . t .} \sum_ {j} \left| \frac {p _ {j} - q _ {j}}{\widetilde {p} _ {j}} \right| \leq \Gamma , - \widetilde {p} _ {j} \leq p _ {j} - q _ {j} \leq \widetilde {p} _ {j}, \forall j, \sum_ {j} p _ {j} = 1 \tag {17}
$$

$$
\text {v a r i a b l e} \quad p _ {1}, \dots , p _ {N}
$$

where  $\mathbf{p}^{t + 1} = [p_1^{t + 1},\dots ,p_N^{t + 1}]\in \mathbb{R}^N$  . We denote  $\widetilde{\mathbf{a}}^{t + 1} = \mathbf{p}^{t + 1} - \overline{\mathbf{p}}$  where  $\overline{\mathbf{p}} = [\overline{p},\dots ,\overline{p} ]\in \mathbb{R}^N$  This first step aims to obtain the distribution  $\tilde{\mathbf{a}}^{t + 1}$  by solving problem in Eq. (17). This problem can be effectively solved through combining merge sort [12] (for sorting  $\widetilde{p_j}\hat{f}_j(\boldsymbol {w}_j),j = 1,\ldots ,N)$  with few basic arithmetic operations (for obtaining  $p_j^{t + 1},j = 1,\dots ,N$  . Since  $N$  is relatively large in distributed system, the arithmetic complexity of solving problem in Eq. (17) is dominated by merge sort, which can be regarded as  $\mathcal{O}(N\log (N))$

Algorithm 1 ASPIRE-EASE  
Initialization: iteration  $t = 0$  variables  $\{\pmb{w}_j^0\} ,\pmb {z}^0,h^0,\lambda_1^0,\{\phi_j^0\}$  and cutting plane set  $\mathbf{A}^0$  repeat for each worker do updates  $\pmb{w}_{j}^{t + 1}$  according to Eq. (11); end for active workers transmit variables and loss to master; master receives updates from active workers do updates  $\pmb{z}^{t + 1},h^{t + 1}$  and  $\lambda_l^{t + 1}$  according to Eq. (12), (13) and (14); master broadcasts variables to active workers; for each worker do updates  $\phi_j^{t + 1}$  according to Eq. (15); end for if  $t$  mod  $k = = 0$  then master updates  $\mathbf{A}^{t + 1}$  according to Eq. (19) and (20), then broadcasting  $\mathbf{A}^{t + 1}$  to all workers; end if  $t = t + 1$  until convergence

198 2) Let  $\mathbf{f}(\pmb {w}) = [f_1(\pmb {w}_1),\dots ,f_N(\pmb {w}_N)]\in \mathbb{R}^N$  , check the feasibility of the following constraints:

$$
\widetilde {\mathbf {a}} ^ {t + 1} \mathbf {\hat {f}} (\boldsymbol {w}) \leq \max  _ {\boldsymbol {a} _ {l} \in \mathbf {A} ^ {t}} \boldsymbol {a} _ {l} ^ {\top} \mathbf {\hat {f}} (\boldsymbol {w}), \tag {18}
$$

3) If Eq. (18) is violated, new cutting plane  $\widetilde{\mathbf{a}}^{t + 1}$  will be added into the cutting plane set:

$$
\mathbf {A} ^ {t + 1} = \left\{ \begin{array}{l} \mathbf {A} ^ {t} \cup \{\widetilde {\mathbf {a}} ^ {t + 1} \}, \text {i f E q . (1 8) i s v i o l a t e d ,} \\ \mathbf {A} ^ {t}, \text {o t h e r w i s e ,} \end{array} \right. \tag {19}
$$

when a new cutting plane is added, its corresponding dual variable  $\lambda_{|\mathbf{A}^t| + 1}^{t + 1}$  will be generated. After the cutting plane subproblem is solved, the inactive cutting plane will be removed, that is:

$$
\mathbf {A} ^ {t + 1} = \left\{ \begin{array}{l} \complement_ {\mathbf {A} ^ {t}} \left\{\boldsymbol {a} _ {l} \right\}, \text {i f} \lambda_ {l} = 0, \\ \mathbf {A} ^ {t}, \text {o t h e r w i s e}, \end{array} \right. \tag {20}
$$

where  $\mathbb{C}_{\mathbf{A}^t}\{\pmb {a}_l\}$  is the complement of  $\{\pmb {a}_l\}$  in  $\mathbf{A}^t$  . Then the master broadcasts  $\mathbf{A}^{t + 1}$  to all workers. Details of the algorithm are summarized in Algorithm 1.

# 5 Convergence Analysis

Definition 1 (Stationarity gap) Following [46, 28, 47], the stationarity gap of our problem at  $t^{\text{th}}$  iteration is defined as:

$$
\nabla G ^ {t} = \left[ \right.\begin{array}{l}\left\{\frac {1}{\eta_ {w}} \left(\boldsymbol {w} _ {j} ^ {t} - \mathcal {P} _ {\mathcal {W}} \left(\boldsymbol {w} _ {j} ^ {t} - \eta_ {w} \nabla_ {\boldsymbol {w} _ {j}} L _ {p} \left(\left\{\boldsymbol {w} _ {j} ^ {t} \right\}, \boldsymbol {z} ^ {t}, h ^ {t}, \left\{\lambda_ {l} ^ {t} \right\}, \left\{\phi_ {j} ^ {t} \right\}\right)\right)\right) \right\}\\\frac {1}{\eta_ {z}} \left(\boldsymbol {z} ^ {t} - \mathcal {P} _ {\boldsymbol {\mathcal {Z}}} \left(\boldsymbol {z} ^ {t} - \eta_ {z} \nabla_ {\boldsymbol {z}} L _ {p} \left(\left\{\boldsymbol {w} _ {j} ^ {t} \right\}, \boldsymbol {z} ^ {t}, h ^ {t}, \left\{\lambda_ {l} ^ {t} \right\}, \left\{\phi_ {j} ^ {t} \right\}\right)\right)\right)\\\frac {1}{\eta_ {h}} \left(h ^ {t} - \mathcal {P} _ {\mathcal {H}} \left(h ^ {t} - \eta_ {h} \nabla_ {h} L _ {p} \left(\left\{\boldsymbol {w} _ {j} ^ {t} \right\}, \boldsymbol {z} ^ {t}, h ^ {t}, \left\{\lambda_ {l} ^ {t} \right\}, \left\{\phi_ {j} ^ {t} \right\}\right)\right)\right)\\\left\{\frac {1}{\rho_ {1}} \left(\lambda_ {l} ^ {t} - \mathcal {P} _ {\Lambda} \left(\lambda_ {l} ^ {t} + \rho_ {1} \nabla_ {\lambda_ {l}} L _ {p} \left(\left\{\boldsymbol {w} _ {j} ^ {t} \right\}, \boldsymbol {z} ^ {t}, h ^ {t}, \left\{\lambda_ {l} ^ {t} \right\}, \left\{\phi_ {j} ^ {t} \right\}\right)\right)\right) \right.\\\left. \left. \left. \left\{\frac {1}{\rho_ {2}} \left(\phi_ {j} ^ {t} - \mathcal {P} _ {\Phi} \left(\phi_ {j} ^ {t} + \rho_ {2} \nabla_ {\phi_ {j}} L _ {p} \left(\left\{\boldsymbol {w} _ {j} ^ {t} \right\}, \boldsymbol {z} ^ {t}, h ^ {t}, \left\{\lambda_ {l} ^ {t} \right\}, \left\{\phi_ {j} ^ {t} \right\}\right)\right)\right) \right. \right. \right. \right]\end{array}, \tag {21}
$$

where  $\nabla G^t$  is the simplified form of  $\nabla G(\{\pmb{w}_j^t\}, \pmb{z}^t, h^t, \{\lambda_l^t\}, \{\phi_j^t\})$ .

Definition 2 ( $\varepsilon$ -stationary point)  $(\{\pmb{w}_j^t\}, \pmb{z}^t, h^t, \{\lambda_l^t\}, \{\phi_j^t\})$  is an  $\varepsilon$ -stationary point ( $\varepsilon \geq 0$ ) of a differentiable function  $L_p$ , if  $||\nabla G^t|| \leq \varepsilon$ .  $T(\varepsilon)$  is the first iteration index such that  $||\nabla G^t|| \leq \varepsilon$ , i.e.,  $T(\varepsilon) = \min \{t \mid ||\nabla G^t|| \leq \varepsilon\}$ .

Assumption 1 (Smoothness/Gradient Lipschitz)  $L_{p}$  has Lipschitz continuous gradients. We assume that there exists  $L > 0$  satisfying

$$
\begin{array}{l} \left| \left| \right| \nabla_ {\theta} L _ {p} (\{\boldsymbol {w} _ {j} \}, \boldsymbol {z}, h, \{\lambda_ {l} \}, \{\phi_ {j} \}) - \nabla_ {\theta} L _ {p} (\{\hat {\boldsymbol {w}} _ {j} \}, \hat {\boldsymbol {z}}, \hat {h}, \{\hat {\lambda} _ {l} \}, \{\hat {\phi} _ {j} \}) \right| | \\ \leq L | | [ \boldsymbol {w} _ {\mathrm {c a t}} - \hat {\boldsymbol {w}} _ {\mathrm {c a t}}; \boldsymbol {z} - \hat {\boldsymbol {z}}; h - \hat {h}; \boldsymbol {\lambda} _ {\mathrm {c a t}} - \hat {\boldsymbol {\lambda}} _ {\mathrm {c a t}}; \phi_ {\mathrm {c a t}} - \hat {\phi} _ {\mathrm {c a t}} ] | |, \\ \end{array}
$$

where  $\theta \in \{\{\pmb{w}_j\}, \pmb{z}, h, \{\lambda_l\}, \{\phi_j\}\}$  and  $[\cdot]$  represents the concatenation.  $\pmb{w}_{\mathrm{cat}} - \hat{\pmb{w}}_{\mathrm{cat}} = [\pmb{w}_1 - \hat{\pmb{w}}_1; \dots; \pmb{w}_N - \hat{\pmb{w}}_N] \in \mathbb{R}^{pN}$ ,  $\pmb{\lambda}_{\mathrm{cat}} - \hat{\pmb{\lambda}}_{\mathrm{cat}} = [\lambda_1 - \hat{\lambda}_1; \dots; \lambda_{|\mathbf{A}^t|} - \hat{\lambda}_{|\mathbf{A}^t|}] \in \mathbb{R}^{|\mathbf{A}^t|}$ ,  $\phi_{\mathrm{cat}} - \hat{\phi}_{\mathrm{cat}} = [\phi_1 - \hat{\phi}_1; \dots; \phi_N - \hat{\phi}_N] \in \mathbb{R}^{pN}$ .

Assumption 2 (Limited size of  $|\mathbf{A}^t|) |\mathbf{A}^t| \leq M, \forall t$ , i.e., the size of cutting plane set is limited.

Assumption 3 (Setting of  $c_1^t, c_2^t$ )  $c_1^t = \frac{1}{\rho_1(t + 1)^{\frac{1}{8}}} \geq \underline{c}_1$  and  $c_2^t = \frac{1}{\rho_2(t + 1)^{\frac{1}{8}}} \geq \underline{c}_2$  are nonnegative non-increasing sequences, where  $\underline{c}_1$  and  $\underline{c}_2$  are positive constants and meet  $M\underline{c}_1^2 + N\underline{c}_2^2 \leq \frac{\varepsilon^2}{4}$ .

Assumption 4 (Bounded change of variables) The change of the variables in master is upper bounded within  $\tau$  iterations:

$$
| | \boldsymbol {z} ^ {t} - \boldsymbol {z} ^ {t - k} | | ^ {2} \leq \tau k _ {1} k _ {z}, | | h ^ {t} - h ^ {t - k} | | ^ {2} \leq \tau k _ {2} k _ {h}, | | \lambda_ {l} ^ {t} - \lambda_ {l} ^ {t - k} | | ^ {2} \leq \tau k _ {3} k _ {\lambda}, \forall 1 \leq k \leq \tau ,
$$

where  $k_{1} > 0$ ,  $k_{2} > 0$  and  $k_{3} > 0$  are constants and  $k_{z} = \min \{|||\pmb{z}^{t + 1} - \pmb{z}^{t}||^{2}, ||\pmb{z}^{t - k + 1} - \pmb{z}^{t - k}||^{2}\}$ ,  $k_{h} = ||h^{t + 1} - h^{t}||^{2}$  and  $k_{\lambda} = ||\lambda_l^{t + 1} - \lambda_l^t ||^2$ .

Theorem 1 (Iteration complexity) Suppose the Assumption 1 to 4 hold, if we set the step-sizes of primal variables as  $\eta_{\mathbf{w}} = \eta_{\mathbf{z}} = \eta_{h} = \frac{2}{L + \rho_{1}|\mathbf{A}^{t}|L^{2} + \rho_{2}NL^{2} + 8(\frac{|\mathbf{A}^{t}|\gamma L^{2}}{\rho_{1}(c_{1}^{t})^{2}} + \frac{N\gamma L^{2}}{\rho_{2}(c_{2}^{t})^{2}})}$ . And we set the step-sizes of dual variables as  $\rho_{1}\leq \min \left\{\frac{2}{L + 2c_{1}^{0}},\frac{1}{5\tau k_{3}NL^{2}}\right\}$  and  $\rho_{2}\leq \frac{2}{L + 2c_{2}^{0}}$ , respectively. For a given  $\varepsilon$ , we have:

$$
T (\varepsilon) \sim \mathcal {O} \left(\max  \left\{\left. \frac {4 M \sigma_ {1} {} ^ {2}}{\rho_ {1} {} ^ {2}} + \frac {4 N \sigma_ {2} {} ^ {2}}{\rho_ {2} {} ^ {2}}\right) ^ {4} \frac {1}{\varepsilon^ {8}}, \left(\frac {4 (4 (\gamma - 2) L ^ {2} (M \rho_ {1} + N \rho_ {2}) + \frac {\rho_ {2} (N - S) L ^ {2}}{2}) ^ {3} (\bar {d} + k _ {d} (\tau - 1)) d _ {5}}{\varepsilon^ {2}} + 2\right) ^ {4} \right\}\right), \tag {22}
$$

where  $\sigma_1, \sigma_2, \gamma, k_d, \overline{d}$  and  $d_5$  are constants. The detailed proof is given in Appendix A.

There exists a wide array of works regarding the convergence analysis of various algorithms for nonconvex/convex optimization problems involved in machine learning [22, 47]. Our analysis, however, differs from existing works in two aspects. First, we solve the non-convex PD-DRO in an asynchronous distributed manner. To our best knowledge, there are few works focusing on solving the DRO in a distributed manner. Compared to solving the non-convex PD-DRO in a centralized manner, solving it in an asynchronous distributed manner poses significant challenges in algorithm design and convergence analysis. Secondly, we do not assume the inner problem can be solved nearly optimally for each outer iteration, which is numerically difficult to achieve in practice [3]. Instead, ASPIRE-EASE is single loop and involves a simple gradient projection operation at each step.

# 6 Experiment

In this section, we conduct experiments on four real-world datasets to assess the performance of the proposed method. Specifically, we evaluate the robustness against data heterogeneity, robustness against malicious attacks and efficiency of the proposed method. Ablation study is also carried out to demonstrate the excellent performance of ASPIRE-EASE.

# 6.1 Datasets and Baseline Methods

We compare our proposed ASPIRE-EASE with baseline methods based on SHL [19], Person Activity [23], Single Chest-Mounted Accelerometer (SM-AC) [9] and Fashion MNIST [45] datasets. The baseline methods include  $\mathrm{Ind}_j$  (learning the model from an individual worker  $j$ ),  $\mathrm{Mix}_{\mathrm{Even}}$  (learning the model from all workers with even weights using ASPIRE), FedAvg [29], AFL [31] and DRFA-Prox [15]. The detailed descriptions of datasets and baselines are given in Appendix C.

In our empirical studies, since the downstream tasks are multi-class classification, the cross entropy loss is used on each worker (i.e.,  $\mathcal{L}_j(\cdot),\forall j$ ). For SHL, Person Activity, and SM-AC, we adopt the deep multilayer perceptron [43] as the base model. And we use the same logistic regression model as in [31, 15] for Fashion MNIST dataset. The base models are trained with SGD. More details are given in Appendix C. Following related works in this direction [36, 31, 15], worst case performance are reported for the comparison of robustness. Specifically, we use  $\mathbf{A}\mathbf{c}\mathbf{c}_{w} = \min_{N}[\mathrm{Acc}_{1},\dots ,\mathrm{Acc}_{N}]$  and  $\mathbf{Loss}_w = \max_N[f_1,\dots ,f_N]$  to denote the worst case test accuracy and the training loss, respectively,

Table 1: Performance comparisons based on  $\mathbf{Acc}_w$  (\%) ↑,  $\mathbf{Loss}_w$  ↓ and  $\mathbf{Std} \downarrow$  (↑ and ↓ respectively) denote higher scores represent better performance and lower scores represent better performance). The boldfaced digits represent the best results, “-” represents not available.  

<table><tr><td rowspan="2">Model</td><td colspan="3">SHL</td><td colspan="3">Person Activity</td><td colspan="3">SC-MA</td><td colspan="3">Fashion MNIST</td></tr><tr><td>Accw↑</td><td>Lossw↓</td><td>Std↓</td><td>Accw↑</td><td>Lossw↓</td><td>Std↓</td><td>Accw↑</td><td>Lossw↓</td><td>Std↓</td><td>Accw↑</td><td>Lossw↓</td><td>Std↓</td></tr><tr><td>max{Indj}</td><td>19.06±0.65</td><td>-</td><td>29.1</td><td>49.38±0.08</td><td>-</td><td>8.32</td><td>22.56±0.78</td><td>-</td><td>17.5</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MixEven</td><td>69.87±3.10</td><td>0.806±0.018</td><td>4.81</td><td>56.31±0.69</td><td>1.165±0.017</td><td>3.00</td><td>49.81±0.21</td><td>1.424±0.024</td><td>6.99</td><td>66.80±0.18</td><td>0.784±0.003</td><td>10.1</td></tr><tr><td>FedAvg [29]</td><td>69.96±3.07</td><td>0.802±0.023</td><td>5.21</td><td>56.28±0.63</td><td>1.154±0.019</td><td>3.13</td><td>49.53±0.96</td><td>1.441±0.015</td><td>7.17</td><td>66.58±0.39</td><td>0.781±0.002</td><td>10.2</td></tr><tr><td>AFL [31]</td><td>78.11±1.99</td><td>0.582±0.021</td><td>1.87</td><td>58.39±0.37</td><td>1.081±0.014</td><td>0.99</td><td>54.56±0.79</td><td>1.172±0.018</td><td>3.50</td><td>77.32±0.15</td><td>0.703±0.001</td><td>1.86</td></tr><tr><td>DRFA-Prox [15]</td><td>78.34±1.46</td><td>0.532±0.034</td><td>1.85</td><td>58.62±0.16</td><td>1.096±0.037</td><td>1.26</td><td>54.61±0.76</td><td>1.151±0.039</td><td>4.69</td><td>77.95±0.51</td><td>0.702±0.007</td><td>1.34</td></tr><tr><td>ASPIRE-EASE</td><td>79.16±1.13</td><td>0.515±0.019</td><td>1.02</td><td>59.43±0.44</td><td>1.053±0.010</td><td>0.82</td><td>56.31±0.29</td><td>1.127±0.021</td><td>3.16</td><td>78.82±0.07</td><td>0.696±0.004</td><td>1.01</td></tr></table>

where  $N$  is the number of workers. We also report the standard deviation  $\mathbf{Std}$  of  $[\mathrm{Acc}_1,\dots ,\mathrm{Acc_N}]$ . In the experiment,  $S$  is set as 1, that means the master will make an update once it receives a message. Each experiment is repeated 10 times, both mean and standard deviations are reported. We implement our model with PyTorch and conduct all the experiments on a server with two TITAN V GPUs.

# 6.2 Results

Robustness against Data Heterogeneity. We first assess the robustness of the proposed ASPIRE-EASE by comparing it with baseline methods when data are heterogeneously distributed across different workers. Specifically, we compare the  $\mathbf{Acc}_w$ ,  $\mathbf{Loss}_w$  and  $\mathbf{Std}$  of different methods on all datasets. The performance comparison results are shown in Table 1. In this table, we can observe that  $\max\{\mathrm{Ind}_j\}$ , which represents the best performance of individual training over all workers, exhibits the worst robustness on SHL, Person Activity, and SC-MA. This is because individual training ( $\max\{\mathrm{Ind}_j\}$ ) only learns from the data in its local worker and cannot generalize to other workers due to different data distributions. Note that  $\max\{\mathrm{Ind}_j\}$  is unavailable for Fashion MNIST since each worker only contains one class of data and cross-entropy loss cannot be used in this case.  $\max\{\mathrm{Ind}_j\}$  also does not have  $\mathbf{Loss}_w$ , since  $\mathrm{Ind}_j$  is trained only on individual worker  $j$ . The FedAvg and  $\mathrm{Mix}_{\mathrm{Even}}$  exhibit better performance than  $\max\{\mathrm{Ind}_j\}$  since they consider the data from all workers. Nevertheless, FedAvg and  $\mathrm{Mix}_{\mathrm{Even}}$  only assign the fixed weight for each worker. AFL is more robust than FedAvg and  $\mathrm{Mix}_{\mathrm{Even}}$  since it not only utilizes the data from all workers but also considers optimizing the weight of each worker. DRFA-Prox outperforms AFL since it also considers the prior distribution and regards it as a regularizer in the objective function. Finally, we can observe that the proposed ASPIRE-EASE shows the best robustness, which can be attributed to two factors: 1) ASPIRE-EASE considers data from all workers and can optimize the weight of each worker; 2) compared with DRFA-Prox which uses prior distribution as a regularizer, the prior distribution is incorporated within the constraint in our formulation (4), which can be leveraged more effectively.

Within ASPIRE-EASE, the level of robustness can be controlled by adjusting  $\Gamma$ . Specially, when  $\Gamma = 0$ , we obtain a nominal optimization problem in which no adversarial distribution is considered. The size of the uncertainty set will increase with  $\Gamma$  (when  $\Gamma \leq N$ ), which enhances the adversarial robustness of the model. As shown in Figure 1, the robustness of ASPIRE-EASE can be gradually enhanced when  $\Gamma$  increases. More results are available in Figure C2 of Appendix C.

Robustness against Malicious Attacks. To assess the model robustness against malicious attacks, malicious workers with backdoor attacks [1, 42], which attempt to mislead the model training process, are added to the distributed system. Following [13], we report the success attack rate of backdoor attacks for comparison. It can be calculated by checking how many instances in the backdoor dataset can be misled and categorized into the target labels. Lower success attack rates indicate more robustness against backdoor attacks. The comparison results are summarized in Table 2 and more detailed settings of backdoor attacks are available in Appendix C. In Table 2, we observe that AFL can be attacked easily since it could assign higher weights to malicious workers. Compared to AFL, FedAvg and Mix_Even achieve relatively lower success attack rates since they assign equal weights to the malicious workers and other workers. DRFA-Prox can achieve even lower success attack rates since it can leverage the prior distribution to assign lower weights for malicious workers. The proposed ASPIRE-EASE achieves the lowest success attack rates since it can leverage the prior distribution more effectively. Specifically, it will assign lower weights to malicious workers with tight theoretical guarantees.

![](images/6ff361899d4577377097cbe0270b707c8cea4964140ce11e2b39b0934879abde.jpg)  
(a) Person Activity

![](images/48314ecc6d64c1ca5893a098ca2ca792699f306126a908c58b8483dc83c3d4da.jpg)  
(b) SC-MA

![](images/3f072ec88683f0a4afad24b99a2b1dc6a3fbb359d4559fcae7cc0dbf53263d35.jpg)  
(a) Person Activity

![](images/ba55f9c77306ada297395356831f7247f5c90ad8242001fe6d81a72b5f95495b.jpg)  
(b) SC-MA

![](images/fbf10672e9f489e7f957e8aeef068effac236cea4246b11c9b61aef6a106592a.jpg)  
Figure 1:  $\Gamma$  control the degree of robustness (worst case performance in the problem) on (a) Person Activity, (b) SC-MA datasets.  
(a) Person Activity  
Figure 3: Comparison of ASPIRE-CP and ASPIRE-EASE regarding the number of cutting planes on (a) Person Activity, (b) SC-MA datasets.

![](images/f35f700e5e45dfcfac2e79b04c3d0732d07325719be5eced1396f4124f1d2616.jpg)  
(b) SC-MA  
Figure 2: Comparison of the convergence time on worst case worker on (a) Person Activity, (b) SC-MA datasets.

Table 2: Performance comparisons about the success attack rate  $(\%)\downarrow$  . The boldfaced digits represent the best results.  

<table><tr><td>Model</td><td>SHL</td><td>Person Activity</td><td>SC-MA</td><td>Fashion MNIST</td></tr><tr><td>MixEven</td><td>36.21±2.23</td><td>34.32±2.18</td><td>52.14±2.89</td><td>83.18±2.07</td></tr><tr><td>FedAvg [29]</td><td>38.15±3.02</td><td>33.25±2.49</td><td>55.39±3.13</td><td>82.04±1.84</td></tr><tr><td>AFL [31]</td><td>68.63±4.24</td><td>43.66±3.87</td><td>75.81±4.03</td><td>90.04±2.52</td></tr><tr><td>DRFA-Prox [15]</td><td>21.23±3.63</td><td>27.27±3.31</td><td>30.79±3.65</td><td>63.24±2.47</td></tr><tr><td>ASPIRE-EASE</td><td>9.17±1.65</td><td>22.36±2.33</td><td>14.51±3.21</td><td>45.10±1.64</td></tr></table>

Efficiency. In Figure 2, we compare the convergence speed of the proposed ASPIRE-EASE with AFL and DRFA-Prox by considering different communication and computation delays for each worker. The proposed ASPIRE-EASE has two variants, ASPIRE-CP (ASPIRE with cutting plane method), ASPIRE-EASE(-)(ASPIRE-EASE without asynchronous setting). More results are available in Figure C3 of Appendix C. Based on the comparison, we can observe that the proposed ASPIRE-EASE generally converges faster than baseline methods and its two variants. This is because 1) compared with AFL, DRFA-Prox, and ASPIRE-EASE(-), ASPIRE-EASE is an asynchronous algorithm in which the master updates its parameters only after receiving the updates from active workers instead of all workers; 2) unlike DRFA-Prox, the master in ASPIRE-EASE only needs to communicate with active workers once per iteration; 3) compared with ASPIRE-CP, ASPIRE-EASE utilizes active set method instead of cutting plane method, which is more efficient. Furthermore, we demonstrate that the minimum time for our asynchronous algorithm to obtain the  $\varepsilon$ -stationary point could be  $\frac{S \times d}{N \times d}$  of the time for synchronous algorithm, where  $\overline{d}$  and  $\underline{d}$  are the maximum and minimum (computation + communication) delay of all workers, respectively. The detailed proof is provided in Theorem 2, Appendix B.

Ablation Study. For ASPIRE, compared with cutting plane method, EASE is more efficient since it considers removing the inactive cutting planes. To demonstrate the efficiency of EASE, we firstly compare ASPIRE-EASE with ASPIRE-CP concerning the number of cutting planes used during the training. In Figure 3, we can observe that ASPIRE-EASE uses fewer cutting planes than ASPIRE-CP, thus is more efficient. The convergence speed of ASPIRE-EASE and ASPIRE-CP in Figure 2 also suggests that ASPIRE-EASE converges much faster than ASPIRE-CP. More results are available in Figure C3 and C4, Appendix C.

# 7 Conclusion

In this paper, we present ASPIRE-EASE method to effectively solve the distributed distributionally robust optimization problem with non-convex objectives. In addition,  $CD$ -norm uncertainty set has been proposed to effectively incorporate the prior distribution into the problem formulation, which allows for flexible adjustment of the degree of robustness of DRO. Theoretical analysis has also been conducted to analyze the convergence properties and the iteration complexity of ASPIRE-EASE. ASPIRE-EASE exhibits strong empirical performance on multiple real-world datasets and is effective in tackling DRO problems in a fully distributed and asynchronous manner. In the future work, more uncertainty sets could be designed for our framework and more update rule for variables in ASPIRE could be considered.

# References

[1] E. Bagdasaryan, A. Veit, Y. Hua, D. Estrin, and V. Shmatikov. How to backdoor federated learning. In International Conference on Artificial Intelligence and Statistics, pages 2938-2948. PMLR, 2020.  
[2] A. Ben-Tal and A. Nemirovski. Robust solutions of uncertain linear programs. Operations research letters, 25(1):1-13, 1999.  
[3] D. P. Bertsekas. Nonlinear programming. Journal of the Operational Research Society, 48(3): 334-334, 1997.  
[4] D. Bertsimas and M. Sim. The price of robustness. Operations research, 52(1):35-53, 2004.  
[5] D. Bertsimas, I. Dunning, and M. Lubin. Reformulation versus cutting-planes for robust optimization. Computational Management Science, 13(2):195–217, 2016.  
[6] J. Blanchet and K. Murthy. Quantifying distributional model risk via optimal transport. Mathematics of Operations Research, 44(2):565-600, 2019.  
[7] L. Bottou. Large-scale machine learning with stochastic gradient descent. In Proceedings of COMPSTAT'2010, pages 177-186. Springer, 2010.  
[8] L. Bottou, F. E. Curtis, and J. Nocedal. Optimization methods for large-scale machine learning. Siam Review, 60(2):223-311, 2018.  
[9] P. Casale, O. Pujol, and P. Radeva. Personalization and user verification in wearable systems using biometric walking patterns. Personal and Ubiquitous Computing, 16(5):563-580, 2012.  
[10] T.-H. Chang, M. Hong, W.-C. Liao, and X. Wang. Asynchronous distributed ADMM for large-scale optimization—Part I: Algorithm and convergence analysis. IEEE Transactions on Signal Processing, 64(12):3118-3130, 2016.  
[11] Y. Chen, Y. Ning, M. Slawski, and H. Rangwala. Asynchronous online federated learning for edge devices with Non-IID data. In 2020 IEEE International Conference on Big Data (Big Data), pages 15-24. IEEE, 2020.  
[12] R. Cole. Parallel merge sort. SIAM Journal on Computing, 17(4):770-785, 1988.  
[13] J. Dai, C. Chen, and Y. Li. A backdoor attack against LSTM-based text classification systems. IEEE Access, 7:138872-138878, 2019.  
[14] E. Delage and Y. Ye. Distributionally robust optimization under moment uncertainty with application to data-driven problems. Operations research, 58(3):595-612, 2010.  
[15] Y. Deng, M. M. Kamani, and M. Mahdavi. Distributionally robust federated averaging. arXiv preprint arXiv:2102.12660, 2021.  
[16] J. C. Duchi and H. Namkoong. Learning models with uniform performance via distributionally robust optimization. The Annals of Statistics, 49(3):1378-1406, 2021.  
[17] R. Gao and A. J. Kleywegt. Distributionally robust stochastic optimization with Wasserstein distance. arXiv preprint arXiv:1604.02199, 2016.  
[18] G. Geraci, M. Wildemeersch, and T. Q. Quek. Energy efficiency of distributed signal processing in wireless networks: A cross-layer analysis. IEEE Transactions on Signal Processing, 64(4): 1034-1047, 2015.  
[19] H. Gjoreski, M. Ciliberto, L. Wang, F. J. O. Morales, S. Mekki, S. Valentin, and D. Roggen. The university of sussex-huawei locomotion and transportation dataset for multimodal analytics with mobile devices. IEEE Access, 6:42592-42604, 2018.  
[20] B. L. Gorissen, I. Yanikoglu, and D. den Hertog. A practical guide to robust optimization. Omega, 53:124-137, 2015.

[21] Y. Hu, X. Chen, and N. He. On the bias-variance-cost tradeoff of stochastic optimization. Advances in Neural Information Processing Systems, 34, 2021.  
[22] C. Jin, P. Netrapalli, and M. Jordan. What is local optimality in nonconvex-nonconcave minimax optimization? In International Conference on Machine Learning, pages 4880-4889. PMLR, 2020.  
[23] B. Kaluza, V. Mirchevska, E. Dovgan, M. Luštrek, and M. Gams. An agent-based approach to care in independent living. In International joint conference on ambient intelligence, pages 177-186. Springer, 2010.  
[24] S. P. Karimireddy, S. Kale, M. Mohri, S. J. Reddi, S. U. Stich, and A. T. Suresh. SCAFFOLD: Stochastic Controlled Averaging for On-Device Federated Learning. 2019.  
[25] D. Kuhn, P. M. Esfahani, V. A. Nguyen, and S. Shafieezadeh-Abadeh. Wasserstein distributionally robust optimization: Theory and applications in machine learning. In Operations Research & Management Science in the Age of Analytics, pages 130–166. INFORMS, 2019.  
[26] D. Levy, Y. Carmon, J. C. Duchi, and A. Sidford. Large-scale methods for distributionally robust optimization. Advances in Neural Information Processing Systems, 33:8847-8860, 2020.  
[27] W.-H. Liao and Y.-T. Huang. Investigation of DNN model robustness using heterogeneous datasets. In 2020 25th International Conference on Pattern Recognition (ICPR), pages 4393-4397. IEEE, 2021.  
[28] S. Lu, I. Tsakakis, M. Hong, and Y. Chen. Hybrid block successive approximation for one-sided non-convex min-max problems: algorithms and applications. IEEE Transactions on Signal Processing, 68:3676-3691, 2020.  
[29] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pages 1273-1282. PMLR, 2017.  
[30] S. Mehrotra and D. Papp. A cutting surface algorithm for semi-infinite convex programming with an application to moment robust optimization. SIAM Journal on Optimization, 24(4): 1670-1697, 2014.  
[31] M. Mohri, G. Sivek, and A. T. Suresh. Agnostic federated learning. In International Conference on Machine Learning, pages 4615-4625. PMLR, 2019.  
[32] A. Nedic and A. Ozdaglar. Distributed subgradient methods for multi-agent optimization. IEEE Transactions on Automatic Control, 54(1):48-61, 2009.  
[33] Y. Nesterov. Introductory lectures on convex optimization: A basic course, volume 87. Springer Science & Business Media, 2003.  
[34] J. Qian, X. Fafoutis, and L. K. Hansen. Towards federated learning: Robustness analytics to data heterogeneity. arXiv preprint arXiv:2002.05038, 2020.  
[35] J. Qian, L. K. Hansen, X. Fafoutis, P. Tiwari, and H. M. Pandey. Robustness analytics to data heterogeneity in edge computing. Computer Communications, 164:229-239, 2020.  
[36] Q. Qian, S. Zhu, J. Tang, R. Jin, B. Sun, and H. Li. Robust optimization over multiple domains. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 4739-4746, 2019.  
[37] H. Rahimian and S. Mehrotra. Distributionally robust optimization: A review. arXiv preprint arXiv:1908.05659, 2019.  
[38] S. Sicari, A. Rizzardi, L. A. Grieco, and A. Coen-Porisini. Security, privacy and trust in Internet of Things: The road ahead. Computer networks, 76:146-164, 2015.  
[39] K. Singhal, H. Sidahmed, Z. Garrett, S. Wu, J. Rush, and S. Prakash. Federated reconstruction: Partially local federated learning. Advances in Neural Information Processing Systems, 34, 2021.

[40] T. Subramanya and R. Riggio. Centralized and federated learning for predictive VNF autoscaling in multi-domain 5G networks and beyond. IEEE Transactions on Network and Service Management, 18(1):63-78, 2021.  
[41] J. Sun, T. Chen, G. B. Giannakis, and Z. Yang. Communication-efficient distributed learning via lazily aggregated quantized gradients. arXiv preprint arXiv:1909.07588, 2019.  
[42] B. Wang, Y. Yao, S. Shan, H. Li, B. Viswanath, H. Zheng, and B. Y. Zhao. Neural cleansse: Identifying and mitigating backdoor attacks in neural networks. In 2019 IEEE Symposium on Security and Privacy (SP), pages 707-723. IEEE, 2019.  
[43] Z. Wang, W. Yan, and T. Oates. Time series classification from scratch with deep neural networks: A strong baseline. In 2017 International joint conference on neural networks (IJCNN), pages 1578-1585. IEEE, 2017.  
[44] W. Wiesemann, D. Kuhn, and B. Rustem. Robust Markov decision processes. Mathematics of Operations Research, 38(1):153-183, 2013.  
[45] H. Xiao, K. Rasul, and R. Vollgraf. Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.  
[46] Z. Xu, H. Zhang, Y. Xu, and G. Lan. A unified single-loop alternating gradient projection algorithm for nonconvex-concave and convex-nonconcave minimax problems. arXiv preprint arXiv:2006.02032, 2020.  
[47] Z. Xu, J. Shen, Z. Wang, and Y. Dai. Zeroth-order alternating randomized gradient projection algorithms for general nonconvex-concave minimax problems. arXiv preprint arXiv:2108.00473, 2021.  
[48] K. Yang, J. Huang, Y. Wu, X. Wang, and M. Chiang. Distributed robust optimization (DRO), part I: Framework and example. Optimization and Engineering, 15(1):35-67, 2014.  
[49] S. Zawad, A. Ali, P.-Y. Chen, A. Anwar, Y. Zhou, N. Baracaldo, Y. Tian, and F. Yan. Curse or redemption? how data heterogeneity affects the robustness of federated learning. arXiv preprint arXiv:2102.00655, 2021.  
[50] R. Zhang and J. Kwok. Asynchronous distributed ADMM for consensus optimization. In International conference on machine learning, pages 1701-1709. PMLR, 2014.
