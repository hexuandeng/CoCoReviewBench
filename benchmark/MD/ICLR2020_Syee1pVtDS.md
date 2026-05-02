# DISTRIBUTED ONLINE OPTIMIZATION WITH LONG-TERM CONSTRAINTS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We consider distributed online convex optimization problems, where the distributed system consists of various computing units connected through a time-varying communication graph. In each time step, each computing unit selects a constrained vector, experiences a loss equal to an arbitrary convex function evaluated at this vector, and may communicate to its neighbors in the graph. The objective is to minimize the system-wide loss accumulated over time. We propose a decentralized algorithm with regret and cumulative constraint violation in  $\mathcal{O}(T^{\max\{c,1 - c\}})$  and  $\mathcal{O}(T^{1 - c/2})$ , respectively, for any  $c \in (0,1)$ , where  $T$  is the time horizon. When the loss functions are strongly convex, we establish improved regret and constraint violation upper bounds in  $\mathcal{O}(\log(T))$  and  $\mathcal{O}(\sqrt{T\log(T)})$ . These regret scalings match those obtained by state-of-the-art algorithms and fundamental limits in the corresponding centralized online optimization problem (for both convex and strongly convex loss functions). In the case of bandit feedback, the proposed algorithms achieve a regret and constraint violation in  $\mathcal{O}(T^{\max\{c,1 - c/3\}})$  and  $\mathcal{O}(T^{1 - c/2})$  for any  $c \in (0,1)$ . We numerically illustrate the performance of our algorithms for the particular case of distributed online regularized linear regression problems.

# 1 INTRODUCTION

The Online Convex Optimization (OCO) paradigm Hazan (2016) has recently become prominent in various areas of machine learning where the environment sequentially generating data is too complex to be efficiently modeled. OCO portrays optimization as a process, and applies a robust and sequential optimization approach where one learns from experiences as time evolves. Specifically, under the OCO framework, at each time-step the learner commits to a decision and suffers from a loss, a convex function of the decision. The successive loss functions are unknown beforehand and may vary arbitrarily over time. At the end of each step, the loss function may be revealed (a scenario referred to as full information). Alternatively, the experienced loss only might be available (bandit feedback). The objective of the decision maker is to minimize the loss accumulated over time. The performance of an algorithm in OCO is assessed through the notion of regret, comparing the accumulated loss under the algorithm and that achieved by an Oracle always selecting the best fixed decision. In case of full information feedback, it is known that the best possible regret scales in  $\mathcal{O}(\sqrt{T})$  (resp.  $\mathcal{O}(\log T)$ ) for convex (resp. strongly convex) loss functions Zinkevich (2003); Hazan et al. (2007); Abernethy et al. (2009).

This paper extends the OCO framework to a distributed setting where (different) data is collected and processed at  $N$  computing units in a network. More precisely, we consider scenarios where in each time-step, each unit  $i$  commits to a decision  $\mathbf{x}_i(t)$  and then experiences a local loss equal to  $\ell_{i,t}(\mathbf{x}_i(t))$ . Units update their decision based on previously observed local losses and messages received from neighboring units with the objective of identifying the decision  $\mathbf{x}^{\star} = \arg \min_{\mathbf{x}}\sum_{t = 1}^{T}\sum_{i = 1}^{N}\ell_{i,t}(\mathbf{x})$  minimizing the accumulated system-wide loss. Many traditional applications of the centralized OCO framework Hazan (2016) naturally extend to this distributed setting. As a motivating example, consider the following distributed online spam filtering task (refer to Hazan (2016) for a description of the spam filter design problem in a centralized setting). In each time-step, each unit  $i$  (here an email server) receives an email characterized by a vector  $\mathbf{a}_{i,t}\in \mathbb{R}^d$  (according to the "bag-of-words" representation). Unit  $i$  applies for this email a filter represented

by a vector  $\mathbf{x}_i(t) \in \mathcal{X}$  where  $\mathcal{X}$  is convex compact subset of  $\mathbb{R}^d$ , returns a label  $f(\mathbf{a}_{i,t}^\top \mathbf{x}_i(t))$ , and experiences a loss equal to  $\ell_{i,t}(\mathbf{x}_i(t)) = (f(\mathbf{a}_{i,t}^\top \mathbf{x}_i(t)) - y_{i,t})^2$  where  $y_{i,t}$  is the true email label (-1 for spam or 1 for valid). Note that the sequences of loss functions are inherently different at various units because the latter receive different emails. Nevertheless, each unit would ideally wish to identify and apply as fast as possible the filter minimizing the system-wide loss, i.e., a filter that exploits the knowledge extracted from all emails, including those received at other units. By leveraging this knowledge, each unit would adapt faster to an adversary also modifying in an online manner spam emails. More generally the distributed OCO framework can be applied to networks of learning agents, where each agent wishes to take advantage of what other agents have learnt to speed up and robustify its own learning process.

# 1.1 THE DOCO (DISTRIBUTED ONLINE CONVEX OPTIMIZATION) FRAMEWORK

We describe here our distributed optimization problem in more detail. We consider a network of  $N$  computing units described by a sequence of directed graphs  $\mathcal{G}_t = \{\mathcal{V},\mathcal{E}_t\}$  with node set  $\mathcal{V} = \{1,\dots ,N\}$  and edge set  $\mathcal{E}_t$  at time  $t$ .  $\mathcal{G}_t$  represents the communication constraints at the end of time-step  $t$ : each unit is allowed to send its decision at time  $t$  to its neighbors in  $\mathcal{G}_t$ . Each unit  $i\in \mathcal{V}$  is associated with a sequence of convex loss functions  $\{\ell_{i,t}\}_{t = 1}^{T}$ , where  $\ell_{i,t}:\mathbb{R}^d\to \mathbb{R}$ .

**Optimization process.** In each time-step  $t$ , each unit  $i \in \mathcal{V}$  selects  $\mathbf{x}_i(t) \in \mathbb{R}^d$ . Then, in case of full information feedback, the loss function  $\ell_{i,t}$  is revealed to unit  $i$ , whereas in case of bandit feedback, the loss  $\ell_{i,t}(\mathbf{x}_i(t))$  is revealed only. Unit  $i$  finally receives vectors, functions of decisions selected by its neighbors in  $\mathcal{G}_t$ , i.e.,  $\mathbf{x}_j(t)$  for  $j$  such that  $(j,i) \in \mathcal{E}_t$ , and updates its decision for the next time-step.

Decision constraints. The decisions should be selected in  $\mathcal{X}$  a convex subset of  $\mathbb{R}^d$  characterized by a family of inequalities:  $\mathcal{X} = \{\mathbf{x} \in \mathbb{R}^d \mid c_s(\mathbf{x}) \leq 0, s = 1, \dots, p\}$ . Imposing such constrained decisions implies that each unit should be able in each time-step to perform a projection onto  $\mathcal{X}$ , which can be extremely computationally expensive. To circumvent this difficulty, we adopt the notion of long-term constraints introduced in Mahdavi et al. (2012). Specifically, we only impose that the constraints are satisfied in a long run rather than in each time-step, i.e., that  $\sum_{t=1}^{T} \sum_{i=1}^{N} \sum_{s=1}^{p} c_s(\mathbf{x}_i(t)) \leq 0$ . This relaxation allows units to violate the constraints by projecting onto a simpler set that contains  $\mathcal{X}$ . Our results can be modified to account for the actual constraints (but using projection steps).

Regrets and cumulative absolute constraint violation. The objective is to design distributed sequential decision selection algorithms so that each unit identifies the decision minimizing the accumulated system-wide loss. The performance of such an algorithm is hence captured by the regrets at the various units. The regret at unit  $i$  is:

$$
\operatorname {R e g} (i, T) := \sum_ {t = 1} ^ {T} \sum_ {j = 1} ^ {N} \ell_ {j, t} \left(\mathbf {x} _ {i} (t)\right) - \sum_ {t = 1} ^ {T} \sum_ {j = 1} ^ {N} \ell_ {j, t} \left(\mathbf {x} ^ {\star}\right), \tag {1}
$$

where  $\mathbf{x}^{\star} = \arg \min_{\mathbf{x}\in \mathcal{X}}\sum_{t = 1}^{T}\sum_{j = 1}^{N}\ell_{j,t}(\mathbf{x})$ . The system-level regret is defined as the worst possible regret at all units:  $\mathsf{SReg}(T)\triangleq \max_{i = 1,\dots ,N}\mathsf{Reg}(i,T)$ . Now since we allow units to select decisions outside  $\mathcal{X}$ , the performance of an algorithm is further characterized by the so-called cumulative absolute constraint violation defined by: (here  $[a]_+ = \max \{0,a\}$ )

$$
\operatorname {C A C V} (T) := \sum_ {t = 1} ^ {T} \sum_ {i = 1} ^ {N} \sum_ {s = 1} ^ {p} \left[ c _ {s} \left(\mathbf {x} _ {i} (t)\right) \right] _ {+}. \tag {2}
$$

# 1.2 MAIN RESULTS

We propose simple distributed algorithms where in each time-step, each unit combines information received from its neighbors to update its decision and its local dual variable. Our algorithms enjoy the following performance guarantees:

Full Information feedback. In the case of full information feedback, the proposed algorithms achieve a system-level regret and a cumulative constraint violation in  $\mathcal{O}(T^{\max\{1 - c, c\}})$  and

$\mathcal{O}(T^{1 - c / 2})$ , respectively and for any  $c \in (0,1)$  ( $c$  expresses the trade-off between regret and cumulative constraint violation). Theses bounds match those of centralized online optimization algorithms in Mahdavi et al. (2012); Jenatton et al. (2016); Yuan & Lamperski (2018). When  $c = 1 / 2$ , we get a regret scaling in  $\mathcal{O}(\sqrt{T})$ , which corresponds to the fundamental regret limits for centralized online problems Abernethy et al. (2009), which is rather surprising in view of the dynamically changing environment, the decentralized structure of the algorithm, and the presence of the constraints. When the loss functions are strongly convex, we establish improved upper bounds on the regret and cumulative constraint violation in  $\mathcal{O}(\log(T))$  and  $\mathcal{O}(\sqrt{T \log(T)})$ . These bounds generalize to our distributed setting those derived in Yuan & Lamperski (2018) for centralized problems.

Bandit feedback. In the case of bandit feedback, the proposed algorithms achieve a system-level regret and a cumulative constraint violation in  $\mathcal{O}(d^2 T^{\max\{1 - c/3, c\}})$  and  $\mathcal{O}(dT^{1 - c/2})$ , respectively, for any  $c \in (0, 1)$ . For example, when  $c = \frac{3}{4}$ , the proposed algorithm attains a regret bound in  $\mathcal{O}(d^2 T^{3/4})$ . The performance guarantees can be improved to  $\mathcal{O}(d^2 T^{2/3} \log(T))$  and  $\mathcal{O}(d\sqrt{T \log(T)})$  in the case of strongly convex losses.

# 1.3 RELATED WORK

Early work on online convex optimization in a centralized setting include Zinkevich (2003); Flaxman et al. (2005). Today we know that a regret in  $\mathcal{O}(\sqrt{T})$  is achievable in both full information and bandit feedback, see e.g. Bubeck et al. (2017). Projection-free algorithms have been also developed Mahdavi et al. (2012); Jenatton et al. (2016); Yuan & Lamperski (2018) with regret and cumulative constraint violation in  $\mathcal{O}(T^{\max\{c,1-c\}})$  and  $\mathcal{O}(T^{1-c/2})$  ( $c \in (0,1)$ ) in case of full information feedback (Yuan & Lamperski (2018) uses the cumulative squared constraint violation). Our algorithms achieve the same guarantees in a distributed setting.

It is worth zooming into the rich literature on centralized online convex optimization with bandit feedback. In the seminal work Flaxman et al. (2005), the authors designed an algorithm with one-point bandit feedback and regret in  $\mathcal{O}(d^2 T^{3/4})$ . The work Agarwal et al. (2010) extended this algorithm to multi-point bandit feedback setting, where multiple points around the decision can be queried for the loss function; they established  $\mathcal{O}(d^2 \sqrt{T})$  and  $\mathcal{O}(d^2 \log(T))$  regret bounds for general convex and strongly convex loss functions, respectively. The work Mahdavi et al. (2012) studied the online bandit optimization with long-term constraints under two-point bandit feedback for domain. They established  $\mathcal{O}(\sqrt{T})$  and  $\mathcal{O}(d^2 T^{3/4})$  bounds on the regret and the cumulative constraint violations, respectively. In this paper, we design distributed algorithms with one-point bandit feedback only, and with the same regret guarantees as the centralized algorithm in Flaxman et al. (2005).

Over the last few years, there have been a rising interest for the Distributed OCO framework. Particularly, Shahrampour & Jabbabaie (2017); Lee et al. (2017) propose distributed algorithms with  $\mathcal{O}(\sqrt{T})$  regret, but require an exact projection onto the decision set in each time-step. Zhang et al. (2017) presents a distributed online conditional gradient algorithm, replacing the projection steps by a much simpler linear optimization steps, but at the expense of worse and sub-optimal regret guarantees, scaling in  $\mathcal{O}(T^{3/4})$ . The other approach to avoid projections is to allow the algorithm to violate the constraints, and has been studied in Yuan et al. (2018). The problem studied in Yuan et al. (2018) is a special case of our problem (where only one inequality constraint is considered), and the regret and cumulative constraint violation guarantees obtained there are much worse than ours. The authors of Li et al. (2018); Yi et al. (2019) also use the long-term constraints approach to avoid projections, but analyze a very different optimization problem where units have different decision variables, and no consensus among units is required. Finally it is worth mentioning that all the aforementioned papers are restricted to full information feedback.

Notation and Terminology. Let  $\| \mathbf{x}\|$  and  $[\mathbf{x}]_i$  to denote the Euclidean norm and the  $i$ th component of a vector  $\mathbf{x} \in \mathbb{R}^d$ , respectively. Let  $\Pi_{\mathcal{X}}[\mathbf{x}]$  be the Euclidean projection of a vector  $\mathbf{x}$  onto the set  $\mathcal{X}$ . Let  $\mathbb{R}_+^p$  be the nonnegative orthant in  $\mathbb{R}^p$ :  $\mathbb{R}_+^p = \{\mathbf{x} \in \mathbb{R}^p \mid [\mathbf{x}]_i \geq 0, i = 1, \dots, p\}$ . Denote the  $(i,j)$ -th element of a matrix  $\mathbf{A}$  by  $[\mathbf{A}]_{ij}$ . For a convex function  $f$ , a subgradient (resp. gradient when  $f$  is differentiable) at a point  $\mathbf{x}$  is denoted by  $\partial f(\mathbf{x})$  (resp.  $\nabla f(\mathbf{x})$ ). Given two positive sequences  $\{a_t\}_{t=1}^\infty$  and  $\{b_t\}_{t=1}^\infty$ , we write  $a_t = \mathcal{O}(b_t)$  if  $\lim_{t \to \infty} a_t / b_t < \infty$ .

# 2 FULL-INFORMATION FEEDBACK

In this section, we focus on the case of full-information feedback, where at the end of each time-step, the entire loss function  $\ell_{i,t}$  is revealed to unit  $i$ . More precisely, unit  $i$  has access to the gradient of the loss function  $\ell_{i,t}$  at any query point. We make the following assumptions, which are standard in the literature e.g., Mahdavi et al. (2012); Jenatton et al. (2016); Yuan & Lamperski (2018); Nedic & Ozdaglar (2009); Yan et al. (2013); Duchi et al. (2012); Hosseini et al. (2013); Nedic et al. (2010).

Assumption 1  $\mathcal{X} \subseteq \mathcal{B} := \{\mathbf{x} \in \mathbb{R}^d \mid \| \mathbf{x} \| \leq R_{\mathcal{X}}\}$  with  $R_{\mathcal{X}} > 0$ .

Assumption 2 The functions  $\ell_{i,t}$  and  $c_{s}$  are convex with bounded gradients:

$$
\max  _ {i = 1, \dots , N} \max  _ {t = 1, \dots , T} \max  _ {\mathbf {x} \in \mathcal {B}} \| \nabla \ell_ {i, t} (\mathbf {x}) \| \leq G _ {\ell}, \quad \max  _ {s = 1, \dots , p} \max  _ {\mathbf {x} \in \mathcal {B}} \| \nabla c _ {s} (\mathbf {x}) \| \leq G _ {c}.
$$

We let  $G = \max \{G_{\ell}, G_{c}\}$ .

Assumption 3 There exists an integer  $B \geq 1$  such that the union graph  $(\mathcal{V}, \mathcal{E}_{kB+1} \cup \dots \cup \mathcal{E}_{(k+1)B})$  is strongly connected for all  $k \geq 0$ .

Assumption 4 Associated with  $\mathcal{G}_t$  there is the weight matrix  $\mathbf{A}(t)$  which satisfies for all  $t\geq 1$ : (i)  $\mathbf{A}(t)$  is doubly stochastic for all  $t\geq 1$ , i.e.,  $\sum_{j = 1}^{N}[\mathbf{A}(t)]_{ij} = 1$  and  $\sum_{i = 1}^{N}[\mathbf{A}(t)]_{ij} = 1$ ,  $\forall i,j\in \mathcal{V}$ ; (ii) There exists a scalar  $\zeta >0$  such that  $[\mathbf{A}(t)]_{ii}\geq \zeta$  for all  $i$  and  $t\geq 1$ , and  $[\mathbf{A}(t)]_{ij}\geq \zeta$  if  $(j,i)\in \mathcal{E}_t$  and  $[\mathbf{A}(t)]_{ij} = 0$  for all  $j$  otherwise.

# Algorithm 1 DOCO-LTC with full-information feedback

Input: Step sizes  $\{\beta_t\}_{t=1}^T$ , regularization parameters  $\{\eta_t\}_{t=1}^T$

Initialize:  $\mathbf{x}_i(1) = \mathbf{0} \in \mathbb{R}^d, \pmb{\lambda}_i(1) = \mathbf{0} \in \mathbb{R}^p, \forall i = 1, \dots, N$

1: for  $t = 1$  to  $T$  do  
2: Unit  $i$  commits to a decision  $\mathbf{x}_i(t)$ , and then after receiving  $\ell_{i,t}$ , compute

$$
\mathbf {y} _ {i} (t) = \mathbf {x} _ {i} (t) - \beta_ {t} \left[ \nabla \ell_ {i, t} (\mathbf {x} _ {i} (t)) + \sum_ {s = 1} ^ {p} [ \boldsymbol {\lambda} _ {i} (t) ] _ {s} \partial [ c _ {s} (\mathbf {x} _ {i} (t)) ] _ {+} \right]
$$

3: Unit  $i$  communicates  $\mathbf{y}_i(t)$  to its neighbors and updates its decision as

$$
\mathbf {x} _ {i} (t + 1) = \Pi_ {\mathcal {B}} (\mathbf {p} _ {i} (t)), \quad \text {w h e r e} \quad \mathbf {p} _ {i} (t) = \sum_ {j = 1} ^ {N} [ \mathbf {A} (t) ] _ {i j} \mathbf {y} _ {j} (t)
$$

4: Unit  $i$  updates its dual variable:  $\lambda_{i}(t + 1) = \arg \max_{\lambda \in \mathbb{R}_{+}^{d}}\mathsf{L}_{i,t}((\mathbf{x}_{i}(t + 1),\lambda)$

5: end for

The pseudo-code of our algorithm, DOCO-LTC (LTC stands for Long-Term Constraints), is presented in Algorithm 1. It generalizes the algorithm in Yuan & Lamperski (2018) to our distributed setting. In contrast to the literature on (online) distributed optimization with inequality constraints Yuan et al. (2018); Li et al. (2018); Khuzani & Li (2016), the algorithm does not need to maintain an iterative dual update process for every unit, which can be computed locally and explicitly. Moreover, no consensus updates on the dual variables are necessary, reducing the communication complexity.

The design and convergence analysis of DOCO-LTC rely on the following online augmented Lagrangian function associated with unit  $i \in \mathcal{V}$ : for  $t \geq 1$

$$
\mathrm {L} _ {i, t} (\mathbf {x}, \boldsymbol {\lambda}) \triangleq \ell_ {i, t} (\mathbf {x}) + \sum_ {s = 1} ^ {p} [ \boldsymbol {\lambda} ] _ {s} \left[ c _ {s} (\mathbf {x}) \right] _ {+} - \frac {\eta_ {t}}{2} \| \boldsymbol {\lambda} \| ^ {2}, \tag {3}
$$

where  $\pmb{\lambda} = [[\pmb{\lambda}]_1, \dots, [\pmb{\lambda}]_p]^\top \in \mathbb{R}_+^p$  is the vector of Lagrangian multipliers with  $[\pmb{\lambda}]_s$  being associated with the  $s$ th inequality constraint  $c_s(\mathbf{x}) \leq 0$  and  $\eta_t$  is the regularization parameter. We note that:

$$
\nabla_ {\mathbf {x}} \mathsf {L} _ {i, t} (\mathbf {x} _ {i} (t), \boldsymbol {\lambda} _ {i} (t)) = \nabla \ell_ {i, t} (\mathbf {x} _ {i} (t)) + \sum_ {s = 1} ^ {p} [ \boldsymbol {\lambda} _ {i} (t) ] _ {s} \partial [ c _ {s} (\mathbf {x} _ {i} (t)) ] _ {+},
$$

where  $\partial [c_s(\mathbf{x}_i(t))]_+$  can be calculated as follows for  $s = 1,\dots ,p$

$$
\partial [ c _ {s} (\mathbf {x} _ {i} (t)) ] _ {+} = \left\{ \begin{array}{l l} \nabla c _ {s} (\mathbf {x} _ {i} (t)), & \text {i f} c _ {s} (\mathbf {x} _ {i} (t)) > 0 \\ 0, & \text {o t h e r w i s e .} \end{array} \right.
$$

Moreover, the dual update  $\lambda_{i}(t + 1)$  in DOCO-LTC can be calculated explicitly as follows:

$$
\left[ \boldsymbol {\lambda} _ {i} (t + 1) \right] _ {s} = \frac {\left[ c _ {s} \left(\mathbf {x} _ {i} (t + 1)\right) \right] _ {+}}{\eta_ {t}}, \quad s = 1, \dots , p. \tag {4}
$$

Theorem 1 (Convex loss functions and full-information feedback) Under Assumptions 1-4, the regret and cumulative constraint violation of DOCO-LTC with parameters  $\eta_t = \frac{1}{T^c}$  and  $\beta_t = \frac{1}{apG^2T^c}$  for some  $c \in (0,1)$ ,  $a > 1$ , and all  $t \geq 1$ , satisfy: for all  $T \geq 1$

$$
\operatorname {S R e g} (T) \leq \tilde {C} T ^ {\max  \{1 - c, c \}} \text {a n d} \operatorname {C A C V} (T) \leq \bar {C} T ^ {1 - c / 2},
$$

where  $\tilde{C} = \frac{1}{2} apNG^2 R_{\mathcal{X}}^2 +\frac{1}{ap} N(1 + \hat{C}) + \frac{NC^2}{4a(a - 1)p}$  with  $\hat{C} = 2N\left(\frac{3N}{\psi^{2 + 1 / B}(1 - \psi^{1 / B})} +4\right)$  and  $\psi = \left(1 - \frac{\zeta}{4N^2}\right)^{-2},$  and  $\bar{C} = \sqrt{\frac{N^2}{a - 1}\left(1 + 2apGR_{\mathcal{X}} + \frac{1}{2}a^2p^2G^2R_{\mathcal{X}}^2\right)}.$

Theorem 1 shows that DOCO-LTC has the same guarantees as those of the centralized algorithms in Mahdavi et al. (2012); Jenatton et al. (2016); Yuan & Lamperski (2018). The user-defined parameter  $c$  tunes the trade-off between SReg and CACV (for  $c = 1/2$ , we get a regret and constraint violation in  $\mathcal{O}(\sqrt{T})$  and  $\mathcal{O}(T^{3/4})$ ).

Communication cost vs. regret. The communication cost, i.e., the number of vectors transmitted per round in the network, is simply equal to the number of edges in the network. Taking the case of  $B = 1$  (i.e., the graph is fixed and connected) as an example, we can establish that the regret bound in Theorem 1 scales as  $\mathcal{O}\left(\frac{N^4}{(1 - \sigma_2(\mathbf{A}))^2} T^{\max\{c,1-c\}}\right)$ , where  $\sigma_2(\mathbf{A})$  is the second largest singular value of the weight matrix  $\mathbf{A}$ . If we choose the weight matrix as the maximum-degree weights (see, e.g., Yuan et al. (2019)), we have the following conclusions: i) Random geometric graph: the regret bound scales as  $\frac{N^6}{\log^2(N)} T^{\max\{c,1-c\}}$  and at most  $2\log^{1+\epsilon}(N)N$  vectors are transmitted per round; ii)  $k$ -regular expander graph:  $\sigma_2(\mathbf{A})$  is constant, the regret bounds scales as  $N^4 T^{\max\{c,1-c\}}$  and  $2kN$  vectors are transmitted per round; and iii) complete graph:  $\sigma_2(\mathbf{A}) = 0$  and  $N(N-1)$  vectors are transmitted per round.

Next we improve DOCO-LTC performance guarantees when the loss functions are strongly convex.

Assumption 5 The loss function  $\ell_{i,t}$  is  $\sigma$ -strongly convex over  $\mathcal{B}$ , that is, for any  $\mathbf{x}, \mathbf{y} \in \mathcal{B}$ ,

$$
\ell_ {i, t} (\mathbf {x}) \geq \ell_ {i, t} (\mathbf {y}) + \nabla \ell_ {i, t} (\mathbf {y}) ^ {\top} (\mathbf {x} - \mathbf {y}) + \frac {\sigma}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.
$$

Theorem 2 (Strongly convex loss functions and full-information feedback) Under Assumptions 1-5, the regret and cumulative constraint violation of DOCO-LTC with parameters  $\eta_t = \frac{2pG^2}{\sigma t}$  and  $\beta_t = \frac{1}{\sigma t}$  for all  $t \geq 1$ , satisfy: for all  $T \geq 3$

$$
\mathsf {S R e g} (T) \leq \tilde {C} _ {\mathrm {s c}} \log (T), a n d \mathsf {C A C V} (T) \leq \bar {C} _ {\mathrm {s c}} \sqrt {T \log (T)},
$$

where  $\tilde{C}_{\mathrm{sc}} = \frac{NG^2}{2\sigma} (4 + 4\hat{C} +\hat{C}^2)$  ( $\hat{C}$  is shown in Theorem 1) and  $\bar{C}_{\mathrm{sc}} = \frac{4pNG^{3 / 2}}{\sqrt{\sigma}}\left(\sqrt{R_{\mathcal{X}}} +\sqrt{\frac{G}{\sigma}}\right)$ .

In the case of strongly convex loss functions, the regret and constraint violation guarantees of DOCO-LTC also match those obtained by the centralized algorithm in Yuan & Lamperski (2018). Note that one cannot actually get a better regret scaling, even in the centralized setting Abernethy et al. (2009).

# 3 ONE-POINT BANDIT FEEDBACK

This section is devoted to the case of bandit feedback, where at the end of each time-step, unit  $i$  can observe the value of the loss function  $\ell_{i,t}$  at only one point around  $\mathbf{x}_i(t)$ . The pseudo-code of our algorithm adapted to this feedback is presented in Algorithm 2.

# Algorithm 2 DOCO-LTC with one-point bandit feedback

Input: Step sizes  $\{\beta_t\}_{t=1}^T$ , regularization parameters  $\{\eta_t\}_{t=1}^T$ , exploration parameters  $\{\varepsilon_t\}_{t=1}^T$ , and shrinkage parameter  $\pi$

Initialize:  $\mathbf{x}_i(1) = \mathbf{0} \in \mathbb{R}^d, \pmb{\lambda}_i(1) = \mathbf{0} \in \mathbb{R}^p, \forall i = 1, \dots, N$

1: for  $t = 1$  to  $T$  do  
2: Unit  $i$  commits to a decision  $\mathbf{x}_i(t)$ , and then observes the loss  $\ell_{i,t}(\mathbf{x}_i(t) + \varepsilon_t\mathbf{u}_i(t))$  where  $\mathbf{u}_i(t)$  is randomly chosen on the unit sphere  $(\| \mathbf{u}_i(t)\| = 1)$  
3: Unit  $i$  builds the following one-point gradient estimator:

$$
\tilde {\nabla} \ell_ {i, t} (\mathbf {x} _ {i} (t)) = \frac {d}{\varepsilon_ {t}} \ell_ {i, t} (\mathbf {x} _ {i} (t) + \varepsilon_ {t} \mathbf {u} _ {i} (t)) \mathbf {u} _ {i} (t)
$$

and computes

$$
\mathbf {y} _ {i} (t) = \mathbf {x} _ {i} (t) - \beta_ {t} \left[ \tilde {\nabla} \ell_ {i, t} (\mathbf {x} _ {i} (t)) + \sum_ {s = 1} ^ {p} [ \boldsymbol {\lambda} _ {i} (t) ] _ {s} \partial [ c _ {s} (\mathbf {x} _ {i} (t)) ] _ {+} \right]
$$

4: Unit  $i$  updates its decision using  $\mathbf{y}_j(t)$  received from its neighbors as

$$
\mathbf {x} _ {i} (t + 1) = \Pi_ {\mathcal {B}} (\mathbf {p} _ {i} (t)), \quad \text {w h e r e} \quad \mathbf {p} _ {i} (t) = \sum_ {j = 1} ^ {N} [ \mathbf {A} (t) ] _ {i j} \mathbf {y} _ {j} (t)
$$

5: Node  $i$  updates its dual variable  $\lambda_{i}(t + 1) = \arg \max_{\pmb{\lambda}\in \mathbb{R}_{+}^{d}}\tilde{\mathsf{L}}_{i,t}((\mathbf{x}_{i}(t + 1),\pmb {\lambda})$

6: end for

The design and convergence analysis of our algorithm here rely on the smoothed version  $\tilde{\mathsf{L}}_{i,t}(\mathbf{x},\boldsymbol {\lambda})$  of the online augmented Lagrangian function (3), i.e.,  $\tilde{\mathsf{L}}_{i,t}(\mathbf{x},\boldsymbol {\lambda})$  : for  $t\geq 1$

$$
\tilde {\mathcal {L}} _ {i, t} (\mathbf {x}, \boldsymbol {\lambda}) \triangleq \tilde {\ell} _ {i, t} (\mathbf {x}; \varepsilon) + \sum_ {s = 1} ^ {p} [ \boldsymbol {\lambda} ] _ {s} [ c _ {s} (\mathbf {x}) ] _ {+} - \frac {\eta_ {t}}{2} \| \boldsymbol {\lambda} \| ^ {2}, \tag {5}
$$

where  $\tilde{\ell}_{i,t}(\mathbf{x};\varepsilon) = \mathbb{E}_{\mathbf{v}}[\ell_{i,t}(\mathbf{x} + \varepsilon \mathbf{v})]$  is the smoothed loss function, and  $\mathbf{v}$  is a vector uniformly distributed over the unit sphere. As in the case of full information feedback, the dual update  $\lambda_{i}(t + 1)$  can be calculated explicitly according to (4).

In the case of bandit feedback, we need to introduce the shrinkage parameter  $\pi$  to ensure that the random query point  $\mathbf{x}_i(t) + \varepsilon_t\mathbf{u}_i(t)$  belongs to the set  $\mathcal{B}$ . Indeed, we have:

$$
\left\| \mathbf {x} _ {i} (t) + \varepsilon_ {t} \mathbf {u} _ {i} (t) \right\| \leq \left\| \mathbf {x} _ {i} (t) \right\| + \varepsilon_ {t} \left\| \mathbf {u} _ {i} (t) \right\| \leq (1 - \pi) R _ {\mathcal {X}} + \varepsilon_ {t} \leq R _ {\mathcal {X}}
$$

where the second inequality follows from the fact that  $\mathbf{x}_i(t) \in (1 - \pi)\mathcal{B}$  and  $\|\mathbf{u}_i(t)\| = 1$  and the last inequality holds when  $\varepsilon_t \leq \pi R_{\mathcal{X}}$ .

To establish upper bounds on the regret and cumulative constraint violation of our algorithm, we make the following standard assumption on the loss functions  $\ell_{i,t}(\mathbf{x})$  (commonly adopted even in centralized online bandit optimization Flaxman et al. (2005)).

Assumption 6 The loss functions  $\ell_{i,t}(\mathbf{x})$  are uniformly bounded over  $\mathcal{B}$ :

$$
\sup_{\mathbf{x}\in \mathcal{B}}\max_{i = 1,\ldots ,N}\max_{t = 1,\ldots ,T}|\ell_{i,t}(\mathbf{x})|\leq C.
$$

Since algorithms for bandit feedback are inherently randomized, we investigate averaged versions of the regret and the cumulative constraint violation:  $\mathsf{E}\text{-}\mathsf{SReg}(T)\coloneqq \max_{i = 1,\ldots ,N}\mathbb{E}[\mathsf{Reg}(i,T)]$  and  $\mathsf{E}\text{-}\mathsf{CACV}(T)\coloneqq \sum_{t = 1}^{T}\sum_{i = 1}^{N}\sum_{s = 1}^{p}\mathbb{E}[[c_s(\mathbf{x}_i(t))]_+]$

Theorem 3 (Convex functions with bandit feedback) Under Assumptions 1-4 and 6, the regret and cumulative constraint violation of DOCO-LTC with parameters

$$
\eta_ {t} = \frac {1}{T ^ {c}}, \quad \beta_ {t} = \frac {1}{a p G ^ {2} T ^ {c}}, \quad \varepsilon_ {t} = \frac {1}{T ^ {b}}, \quad \pi = \frac {1}{R _ {\mathcal {X}} T ^ {b}}
$$

for some  $c \in (0,1)$ ,  $b = c / 3$  and all  $t \geq 1$ , satisfy: for all  $T \geq 1$ ,

$$
E - S R e g (T) \leq \bar {C} ^ {\S} T ^ {\max  \{1 - c / 3, c \}} a n d E - C A C V (T) \leq \bar {C} ^ {\S} T ^ {1 - c / 2},
$$

where  $\tilde{C}^{\S} = 3NG + \frac{NC\hat{C}d}{apG} +\frac{NC^{2}d^{2}}{apG^{2}} +\frac{1}{2} apNG^{2}R_{\chi}^{2} + \frac{NC^{\prime}c^{2}}{4a(a - 1)p}$  ( $\hat{C}$  is shown in Theorem 1) and  $\bar{C}^{\S} = \sqrt{\frac{N^{2}}{a - 1}\left(\frac{C^{2}d^{2}}{G^{2}} + 2apGR_{\chi} + \frac{1}{2} a^{2}p^{2}G^{2}R_{\chi}^{2}\right)}.$

Note that DOCO-LTC achieves a regret scaling as  $T^{3/4}$  when  $c = \frac{3}{4}$ , which is identical to that of centralized online bandit optimization Flaxman et al. (2005). This is rather remarkable considering the decentralized nature of the algorithm. Again, we can improve our bounds in the case of strongly convex loss functions.

Theorem 4 (Strongly convex functions with bandit feedback) Under Assumptions 1-6, the regret and cumulative constraint violation of DOCO-LTC with parameters

$$
\eta_ {t} = \frac {2 p G ^ {2}}{\sigma t}, \quad \beta_ {t} = \frac {1}{\sigma t}, \quad \varepsilon_ {t} = \frac {1}{T ^ {b}}, \quad \pi = \frac {1}{R _ {\mathcal {X}} T ^ {b}}
$$

for  $b = \frac{1}{3}$ , and all  $t \geq 1$ , satisfy: for all  $T \geq 3$

$$
\mathsf {E} - \mathsf {S R e g} (T) \leq \tilde {C} _ {\mathrm {s c}} ^ {\S} T ^ {2 / 3} \log (T) \quad a n d \quad \mathsf {E} - \mathsf {C A C V} (T) \leq \bar {C} _ {\mathrm {s c}} ^ {\S} \sqrt {T \log (T)},
$$

where  $\tilde{C}_{\mathrm{sc}}^{\S} = 3NG + \frac{N}{2\sigma}\left(4C\hat{C}Gd + 4C^{2}d^{2} + \hat{C}^{2}G^{2}\right)$  ( $\hat{C}$  is shown in Theorem 1) and  $\bar{C}_{\mathrm{sc}}^{\S} = \frac{4pNG}{\sqrt{\sigma}}\left(\sqrt{GR_{\mathcal{X}}} +\frac{Cd}{\sqrt{\sigma}}\right)$ .

# 4 NUMERICAL EXPERIMENT

We illustrate the performance of the proposed algorithms using a simple experiment. Specifically, we consider distributed online regularized linear regression problem over a network, formulated as follows:

minimize  $\sum_{t=1}^{T} \sum_{i=1}^{N} \frac{1}{2}\left(\mathbf{a}_{i}(t)^{\top} \mathbf{x} - b_{i}(t)\right)^{2} + \rho\|\mathbf{x}\|^{2}$

subject to  $c_{m}(\bar{\mathbf{x}}) = L - [\bar{\mathbf{x}}]_{m}\leq 0,\quad m = 1,\ldots ,d$  (6)

$c_{d + m}(\mathbf{x}) = [\mathbf{x}]_m - U\leq 0,\quad m = 1,\ldots ,d$

where  $\rho \geq 0$  denotes the regularization parameter. The data  $(\mathbf{a}_i(t), b_i(t)) \in \mathbb{R}^d \times \mathbb{R}$  is revealed only to unit  $i$  at time  $t$ . Every entry of  $\mathbf{a}_i(t)$  is generated uniformly at random within the interval  $[-1, 1]$  and  $b_i(t)$  is generated according to

$$
b _ {i} (t) = \mathbf {a} _ {i} (t) ^ {\top} \bar {\mathbf {x}} + \epsilon_ {i} (t)
$$

where  $[\bar{\mathbf{x}}]_i = 1$ , for all  $1 \leq i \leq \lfloor d / 2 \rfloor$  and 0 otherwise, and the noise  $\epsilon_i(t) \sim \mathcal{N}(0,1)$ . Throughout the experiments, we implement our algorithms over a time-varying directed network depicted in Fig. 1: the network is not connected in every time-step, but the union graph of any two consecutive time instances is strongly connected, that is, we have  $B = 2$  in Assumption 3. The weight matrices associated with the networks in Fig. 1 are generated according to the maximum-degree weights (see, e.g., Yuan et al. (2019)). We set the parameters as follows:  $N = 6$ ,  $d = 4$ ,  $L = -0.15$ ,  $U = 0.15$ , and  $R_{\mathcal{X}} = U\sqrt{d}$ . The performance of DOCO-LTC is averaged over 10 runs.

![](images/5e0dea23d3658175d04170a2138d0cf8382d02af6cd4c0b524db40ee4032fb18.jpg)

![](images/7168b76ab279843b8327602b96f4472a053ab8ce69509ef468dfa316e5d8e49d.jpg)  
(a)

![](images/fbfb0953f69bc9c416508a045950510b61018fa7586b3ba1d812ae6f06cf28c6.jpg)

![](images/aa8568dd3166a88619fa052d56ec24517b450d3b44a18d72ee6d2835e6f812ae.jpg)  
(b)  
Figure 1: The network switches sequentially in a round robin manner between (a), (b), (c), and (d).

![](images/7836a7b3c8655af79fed682c1ab5d009ffe70ea313c02baca5d19e7aaa11451f.jpg)

![](images/45c64b9807c027716b287dece9d2aa1a1a53c5b1c188dd6aecdf6f649a3244ce.jpg)  
(c)

![](images/b785463df022e2a2998f74434cd6d5da8ea1822b997e03f3b5e9533a53a4283d.jpg)  
(d)

To get (not strongly) convex loss functions, we set  $\rho = 0$ . We run Algorithm 1 and Algorithm 2 with  $c = 1/2$  and  $c = 3/4$  and plot the maximum regret  $\max_{i \in \mathcal{V}} \operatorname{Reg}(i, T)$ , and  $\mathsf{CACV}(T)$  as a function of the time horizon  $T$  in Fig. 2(a) and Fig. 2(b), respectively. It can be seen from Fig. 2(a) that in the case of full-information feedback, the regret is smaller for  $c = 1/2$ , while in bandit feedback setting, the regret is smaller for  $c = 3/4$ . This is because  $c = 1/2$  and  $c = 3/4$  correspond to a balanced regret in the full-information setting and bandit feedback setting. By balanced, we mean that  $1 - c = c$  in  $T^{\max\{1 - c, c\}}$  in Theorem 1 and  $1 - c/3 = c$  in  $T^{\max\{1 - c/3, c\}}$  in Theorem 3, respectively. From Fig. 2(b) we also observe that for both feedback models,  $\mathsf{CACV}$  is smaller for a larger value of  $c$ , i.e.,  $c = 3/4$ . This is in compliance with the results established in Theorems 1 and 3. Finally, the performance is really degraded when going from full information to bandit feedback. This was also expected.

![](images/6a017ef30332b92f33fc560165adc00f3ee0ff70947c37e522182d49883e1ae6.jpg)  
Figure 2: SReg and CACV vs. time for convex costs.

![](images/4ea8a290dce5dbfd60612dc777eeda58f1aba2e9a08d62588ce6740538b93952.jpg)

In the case of strongly convex losses, we run Algorithm 1 and Algorithm 2 with  $\rho >0$ , namely  $\rho = 1$  and  $\rho = 2$ . We plot the performance of the algorithms as a function the time horizon in Fig. 3(a) and Fig. 3(b), respectively. From Fig. 3, we confirm that the cost of bandit feedback is rather high. We also observe the regret and the violation constraint are smaller and flatter than those achieved of non-strongly convex loss functions  $(\rho = 0)$ , for both feedback models. All these observations comply with the results established in Theorems 2 and 4.

![](images/0bfa263ec1148068e05cf526cec39b5a97393bf8cacf98a7190b2252fb086b96.jpg)  
(a)

![](images/bfdacbf8bdbba996bc1e6025ca2e59ef584164bbd53e71ef7b7f8c15e5640b04.jpg)  
(b)  
Figure 3: SReg and CACV vs. time for strongly convex costs.

# 5 CONCLUSIONS

In this paper, we consider the distributed online convex optimization problem with long-term constraints under full-information and bandit feedback. By introducing and exploiting the notion of online augmented Lagrangian function, we develop distributed algorithms that are based on con

sensus algorithms. For the case of full-information feedback, we establish sub-linear regret and cumulative absolute constraint violations that match those of centralized online optimization in the literature. Moreover, we also establish sub-linear regret and constraint violation in the case of bandit feedback, where the loss function can be locally evaluated at one point in each time-step.

# REFERENCES

Jacob D. Abernethy, Alekh Agarwal, Peter L. Bartlett, and Alexander Rakhlin. A stochastic view of optimal regret through minimax duality. In  $COLT$ , 2009. URL http://dblp.uni-trier.de/db/conf/colt/colt2009.html#AbernethyABR09.  
Alekh Agarwal, Ofer Dekel, and Lin Xiao. Optimal algorithms for online convex optimization with multi-point bandit feedback. In  $COLT$ , pp. 28-40. CiteSeer, 2010.  
Sebastien Bubeck, Yin Tat Lee, and Ronen Eldan. Kernel-based methods for bandit convex optimization. In Proceedings of the 49th Annual ACM SIGACT Symposium on Theory of Computing, STOC 2017, Montreal, QC, Canada, June 19-23, 2017, pp. 72-85, 2017. doi: 10.1145/3055399.3055403. URL https://doi.org/10.1145/3055399.3055403.  
John C Duchi, Alekh Agarwal, and Martin J Wainwright. Dual averaging for distributed optimization: Convergence analysis and network scaling. IEEE Transactions on Automatic control, 57(3): 592-606, 2012.  
Abraham D Flaxman, Adam Tauman Kalai, and H Brendan McMahan. Online convex optimization in the bandit setting: gradient descent without a gradient. In Proceedings of the sixteenth annual ACM-SIAM symposium on Discrete algorithms, pp. 385-394. Society for Industrial and Applied Mathematics, 2005.  
Elad Hazan. Introduction to online convex optimization. Found. Trends Optim., 2(3-4):157-325, August 2016. ISSN 2167-3888. doi: 10.1561/2400000013. URL https://doi.org/10.1561/2400000013.  
Elad Hazan, Amit Agarwal, and Satyen Kale. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2-3):169-192, 2007.  
Saghar Hosseini, Airlie Chapman, and Mehran Mesbahi. Online distributed optimization via dual averaging. In 52nd IEEE Conference on Decision and Control, pp. 1484-1489. IEEE, 2013.  
Rodolphe Jenatton, Jim C Huang, and Cedric Archambeau. Adaptive algorithms for online convex optimization with long-term constraints. In Proceedings of the 33rd International Conference on International Conference on Machine Learning-Volume 48, pp. 402-411. JMLR.org, 2016.  
Masoud Badiei Khuzani and Na Li. Distributed regularized primal-dual method: Convergence analysis and trade-offs. arXiv preprint arXiv:1609.08262, 2016.  
Soomin Lee, Angela Nedic, and Maxim Raginsky. Stochastic dual averaging for decentralized online optimization on time-varying communication graphs. IEEE Transactions on Automatic Control, 62(12):6407-6414, 2017.  
Xiuxian Li, Xinlei Yi, and Lihua Xie. Distributed online optimization for multi-agent networks with coupled inequality constraints. arXiv preprint arXiv:1805.05573, 2018.  
Mehrdad Mahdavi, Rong Jin, and Tianbao Yang. Trading regret for efficiency: online convex optimization with long term constraints. Journal of Machine Learning Research, 13(Sep):2503-2528, 2012.  
Angelia Nedic and Asuman Ozdaglar. Distributed subgradient methods for multi-agent optimization. IEEE Transactions on Automatic Control, 54(1):48, 2009.  
Angelia Nedic, Alex Olshevsky, Asuman Ozdaglar, and John N Tsitsiklis. Distributed subgradient methods and quantization effects. In 2008 47th IEEE Conference on Decision and Control, pp. 4177-4184. IEEE, 2008.  
Angelia Nedic, Asuman Ozdaglar, and Pablo A Parrilo. Constrained consensus and optimization in multi-agent networks. IEEE Transactions on Automatic Control, 55(4):922-938, 2010.  
Shahin Shahrampour and Ali Jabbabaie. Distributed online optimization in dynamic environments using mirror descent. IEEE Transactions on Automatic Control, 63(3):714-725, 2017.

Feng Yan, Shreyas Sundaram, SVN Vishwanathan, and Yuan Qi. Distributed autonomous online learning: Regrets and intrinsic privacy-preserving properties. IEEE Transactions on Knowledge and Data Engineering, 25(11):2483-2493, 2013.  
Xinlei Yi, Xiuxian Li, Lihua Xie, and Karl H Johansson. Distributed online convex optimization with time-varying coupled inequality constraints. arXiv preprint arXiv:1903.04277, 2019.  
Deming Yuan, Daniel WC Ho, and Guo-Ping Jiang. An adaptive primal-dual subgradient algorithm for online distributed constrained optimization. IEEE Transactions on Cybernetics, 48(11):3045-3055, 2018.  
Deming Yuan, Alexandre Proutiere, and Guodong Shi. Distributed online linear regression. arXiv preprint arXiv:1902.04774, 2019.  
Jianjun Yuan and Andrew Lamperski. Online convex optimization for cumulative constraints. In Advances in Neural Information Processing Systems, pp. 6140-6149, 2018.  
Wenpeng Zhang, Peilin Zhao, Wenwu Zhu, Steven CH Hoi, and Tong Zhang. Projection-free distributed online learning in networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 4054-4062. JMLR.org, 2017.  
Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the Twentieth International Conference on International Conference on Machine Learning, ICML'03, pp. 928-935. AAAI Press, 2003. ISBN 1-57735-189-4. URL http://dl.acm.org/citation.cfm?id=3041838.3041955.
