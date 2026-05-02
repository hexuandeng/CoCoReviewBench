# DELTA: DIVERSE CLIENT SAMPLING FOR FASTING FEDERATED LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Partial client participation has been widely adopted in Federated Learning (FL) to efficiently reduce the communication burden. However, an improper client sampling scheme will select unrepresentative subsets, which will cause a large variance in the model update and slows down the convergence. Existing sampling methods are either biased or can be further improved to accelerate the convergence. In this paper, we propose an unbiased sampling scheme, termed DELTA, to alleviate this problem. In particular, DELTA characterizes the impact of client diversity and local variance and samples the representative clients who carry valuable information for global model updates. Moreover, DELTA is a provably optimal unbiased sampling scheme that minimizes the variance caused by partial client participation and achieves better convergence than other unbiased sampling schemes. We corroborate our results with experiments on both synthetic and real data sets.

# 1 INTRODUCTION

Federated Learning (FL) has recently emerged as a critical distributed learning paradigm where a number of clients collaborate with a central server to train a model. Edge clients finish the update locally without any data sharing, thus preserving client privacy. Communication can become the primary bottleneck of FL since edge devices have limited bandwidth and connection availability (Wang et al., 2021). In order to reduce the communication burden, only a portion of clients will be chosen for training in practice. However, an improper client sampling strategy, such as uniform client sampling adopted in FedAvg (McMahan et al., 2017), might exacerbate the issues of data heterogeneity in FL, as the randomly-selected unrepresentative subsets can increase the variance introduced by client sampling and directly slow down the convergence.

Existing sampling strategies can usually be categorized into two classes: biased and unbiased. Considering the crucial unbiased client sampling that may preserve the optimization objective, only a few strategies are proposed, e.g., in terms of multinomial distribution (MD) sampling and cluster sampling, including clustering based on sample size and clustering based on similarity methods. However, these sampling methods usually suffer from a slow convergence with large variance and computation overhead problems (Balakrishnan et al., 2021; Fraboni et al., 2021b).

To accelerate the convergence of FL with partial client participation, Importance Sampling (IS), another unbiased sampling strategy, is proposed in recent literature (Chen et al., 2020; Rizk et al., 2020). IS will select clients with the large gradient norm, as shown in Fig 1(a). As for another sampling method in Figure 1(a), cluster-based IS will first cluster the clients according to the gradient norm and then use IS to select the clients with a large gradient norm within each cluster.

Though IS, and cluster-based IS have their advantages, 1) IS suffers from learning inefficiency due to the transmission of excessive important yet similar updates from clients to the server. This problem has been pointed out in recent works (Fraboni et al., 2021a; Shen et al., 2022), and some efforts are being conducted to solve this problem. One of them is cluster-based IS, which avoids redundant sampling of clients by first clustering similar clients into groups. Though clustering operation can somewhat alleviate this problem, 2) cluster-based IS suffers from a slow convergence since it keeps sampling clients from small gradient clusters. The additional clustering operation will lead to computation and memory overhead. Figure 2 illustrates these limitations and trade-offs at different learning stages. At 100 rounds, the cluster-based IS achieves better accuracy than IS since the clustering can ease the similar gradient problem. In contrast, this similar gradient will reduce the training efficiency of IS and leads to redundant sampling. When the model approaches

![](images/71e9e25c527da7c0a03b2d7da175417ddb34962f90feb26d76f77dda302b445d.jpg)  
(a)

![](images/876e6775fd0efa3e14a5a1adab78df5c55e1c7b2038158a636082af5e332069f.jpg)  
(b)

![](images/4e35f0184ecd6593e9c5086a7af104ea836dea0b67f78d55ff98eb03fd206ef4.jpg)  
Figure 1: Difference between IS, cluster-based IS, and our sampling scheme DELTA.  
Figure 2: We use a logistic regression model to show the performance of different methods on non-uid MNIST. We sample 10 out of 200 clients and run 500 communication rounds. We report the average of the best 10 accuracies under 100, 300, and 500 rounds, which shows the accuracy performance from the initial training state to convergence.

convergence in 500 rounds, cluster-based IS performs worse than IS because the gradients in small gradient norm clusters slow down convergence.

To address the above two challenges, namely similar gradients and slow convergence caused by small gradient norms, we propose a novel sampling method for Federated Learning, termed DivErse cLienT sAmpling (DELTA). To simplify the notion, in this paper, we term FL with IS as FedIS. Compared with FedIS and cluster-based IS methods, we show in Figure 1(b) that DELTA tends to select clients with diverse gradient w.r.t global gradient. In this way, DELTA not only utilizes the advantages of a large gradient norm for convergence acceleration but also overcomes the gradient similarity issue in the initial stage of training.

# 1.1 CONTRIBUTIONS

In this paper, we propose an efficient unbiased sampling scheme based on gradient diversity and local variance, in the sense that (i) it can effectively solve the excessive similar gradient problem without additional clustering operation, while taking advantage of the accelerated convergence of gradient-norm-based IS and (ii) is provable better than uniform sampling or gradient norm based sampling. The sampling scheme is completely generic and can be easily compatible with other variance reduction methods, like Fedprox (Li et al., 2018) and momentum (Karimireddy et al., 2020a).

As our key contributions,

- we present an unbiased sampling scheme for FL based on gradient diversity and local variance, a.k.a. DELTA. It can take advantage of the clients who select a large gradient norm and solve the problem of over-selection of clients with similar gradients at the beginning of training when that gradient of the global model is relatively large. Compared with the SOTA rate of FedAvg, its convergence rate removes the term  $\mathcal{O}(1 / T^{2 / 3})$  as well as a  $\sigma_G^2$ -related term in the numerator of  $\mathcal{O}(1 / T^{1 / 2})$ .  
- We provide theoretical proof of convergence for nonconvex FedIS. Unlike existing work, our analysis is based on a more relaxed assumption and yields no worse results than the existing convergence rates. Its rate removes the term  $\mathcal{O}(^{1} / T^{2 / 3})$  from that of FedAvg.

# 2 RELATED WORK

FedAvg is proposed by McMahan et al. (2017) as a de facto algorithm of FL, in which multiple local SGD steps are executed on the available clients to alleviate the communication bottleneck. While communication efficient, heterogeneity, such as system heterogeneity (Li et al., 2018; Wang et al., 2020; Mitra et al., 2021; Dao et al., 2020), and statistical/objective heterogeneity (Lin et al., 2020; Karimireddy et al., 2020b; Li et al., 2018; Wang et al., 2020; Guo et al., 2021), results in inconsistent optimization objectives and drifted client models, impeding federated optimization considerably.

Objective inconsistency in FL. Objective inconsistency is not rare in FL due to the heterogeneity of clients' data and the difference in computing ability. For instance, Wang et al. (2020) first identify an objective inconsistency caused by heterogeneous local updates. There also exist several works that encounter the difficulty from the objective inconsistency caused by partial client participation (Li et al., 2019; Cho et al., 2020; Balakrishnan et al., 2021). Li et al. (2019); Cho et al. (2020) use local-global gap  $f^{*} - \frac{1}{m}\sum_{i = 1}^{m}F_{i}^{*}$  to measure the distance between global optimum and average of all local personal optimum, where the local-global gap results from objective inconsistency at the final optimal point. In fact, objective inconsistency occurs in each training round, not only at the final optimal point. Balakrishnan et al. (2021) also encounter objective inconsistency caused by partial client participation. However, they use  $\| \frac{1}{n}\sum_{i = 1}^{n}\nabla F_{i}(x_{t}) - \nabla f(x_{t})\| \leq \epsilon$  as an assumption to describe such update inconsistency caused by objective inconsistency without any analysis on it. So far, the objective inconsistency caused by partial client participation has not been analyzed though it is prevalent in FL, even in homogeneous local updates. Our work gives the fundamental convergence analysis on the influence of the objective inconsistency of partial client participation.

Client selection in FL. In general, the sampling method can be divided into biased and unbiased sampling. Note that unbiased sampling guarantees the same expected value of the client aggregation as the global deterministic aggregation with all clients' participation. In contrast, biased sampling will lead to converging to sub-optimal. The most famous unbiased sampling strategy in FL is multinomial sampling (MD), that samples according to client data ratio (Wang et al., 2020; Fraboni et al., 2021a). Besides, IS, an unbiased sampling method, is recently used in FL to reduce the convergence variance. Chen et al. (2020) uses update norm as importance to sampling clients, Rizk et al. (2020) samples clients based on data variability and Mohammed et al. (2021) uses test accuracy as an estimation of importance. Meanwhile, many biased sampling strategies have been proposed for accelerating training, such as sampling clients with higher loss (Cho et al., 2020), sampling clients as many as possible under the limitation of threshold (Qu et al., 2021), sampling clients with larger updates (Ribero & Vikalo, 2020) and greedy sampling according to gradient diversity (Balakrishnan et al., 2021). However, all these biased sampling methods can exacerbate the negative effects of objective inconsistency and promise to converge to only a neighbor of optimum. Recently, cluster-based client selection has drawn some attention in FL (Fraboni et al., 2021a; Xu et al., 2021; Muhammad et al., 2020; Shen et al., 2022). Though cluster operation needs additional clustering operation, and causes computation and memory overhead, Fraboni et al. (2021a); Shen et al. (2022) show clustering is helpful for sampling diverse clients and benefits for reducing variance. The proposed DELTA in Algorithm 1 can be viewed as a muted version of the diverse client clustering algorithm while promising to be unbiased.

# 3 THEORETICAL ANALYSIS AND AN IMPROVED FL SAMPLING STRATEGY

In FL, the objective of the global model is a sum-structured optimization problem:

$$
f ^ {*} = \min  _ {x \in R ^ {d}} [ f (x) := \sum_ {i = 1} ^ {m} w _ {i} F _ {i} (x) ], \tag {1}
$$

where  $F_{i}(x) = \mathbb{E}_{\xi_{i} \sim D_{i}}[F_{i}(x, \xi_{i})]$  represents the local objective function of client  $i$  over data distribution  $D_{i}$ , and  $\xi_{i}$  means the sampled data of client  $i$ .  $m$  is the total number of clients and  $w_{i}$  represents the weight of client  $i$ . With partial client participation, FedAvg (McMahan et al., 2017) randomly selects  $|\tilde{S}_t| = n$  clients ( $n \leq m$ ) to communicate and update model. Then the loss function of actual participating users in each round can be expressed as:

$$
f \left(x _ {t}\right) = \frac {1}{n} \sum_ {i \in S _ {t}} F _ {i} \left(x _ {t}\right). \tag {2}
$$

To ease the theoretical analysis of our work, we use the following widely used assumptions.

# 3.1 ASSUMPTIONS

Assumption 1 (L-Smooth). The client's local objective function is Lipschitz smooth, i.e., there is a constant  $L > 0$ , such that  $\| \nabla F_i(x) - \nabla F_i(y) \| \leq L \| x - y \|$ ,  $\forall x, y \in \mathbb{R}^d$ , and  $i = 1, 2, \ldots, m$ .

Assumption 2 (Unbiased Local Gradient Estimator and Local Variance). let  $\xi_t^i$  be a random local data sample in the round  $t$  at client  $i$ :  $\mathbb{E}\left[\nabla F_i(x_t,\xi_t^i)\right] = \nabla F_i(x_t), \forall i \in [m]$ , where the expectation is over the local datasets sample. The function  $F_i(x_t,\xi_t^i)$  has  $\sigma_{L,i} > 0$  bounded local variance, i.e.,  $\mathbb{E}\left[\left\| \nabla F_i(x_t,\xi_t^i) - \nabla F_i(x_t)\right\|^2\right] = \sigma_{L,i}^2 \leq \sigma_L^2$ .

Table 1: Number of communication rounds required to reach  $\epsilon$  or  $\epsilon +\varphi$  ( $\epsilon$  for unbiased sampling and  $\epsilon +\varphi$  for biased sampling, where  $\varphi$  is a non-convergent constant term) accuracy for FL.  $\sigma_L$  is local variance bound, and  $\sigma_G$  bound is  $E\| \nabla F_i(x) - \nabla f(x)\|^2 \leq \sigma_G^2$ .  $\Gamma$  is the distance of global optimum and the average of local optimum(Heterogeneity bound),  $\mu$  corresponds to  $\mu$  strongly convex.  $G$  is the client's gradient bound, and  $\zeta_G$  means the gradient diversity.  

<table><tr><td>Algorithm</td><td>Convexity</td><td>Partial Worker</td><td>Unbiased Sampling</td><td>Convergence rate</td><td>Assumption</td></tr><tr><td>SGD</td><td>Strongly/Nonconvex</td><td>✓</td><td>✓</td><td>σL2μmKε+(1/μ)/mKε2+1/ε</td><td>σL bound</td></tr><tr><td>DELTA</td><td>Nonconvex</td><td>✓</td><td>✓</td><td>σL2nKε2+M2/Kε</td><td>Assumption 3</td></tr><tr><td>FedIS (ours)</td><td>Nonconvex</td><td>✓</td><td>✓</td><td>σL2+KG2ηKε2+M2/Kε</td><td>Assumption 3 and 4</td></tr><tr><td>FedIS (others) (Chen et al., 2020)</td><td>Nonconvex</td><td>✓</td><td>✓</td><td>M2nKε2+A2+1/ε+σG/ε3/2</td><td>Assumption 3 and ρ bound</td></tr><tr><td>Yang et al. (2021)</td><td>Nonconvex</td><td>✓</td><td>✓</td><td>σL2nKε2+4Kσ2ηKε2+M2/Kε+K1/3M2n1/3ε2/3</td><td>σG bound</td></tr><tr><td>Karimireddy et al. (2020b)</td><td>Nonconvex</td><td>✓</td><td>✓</td><td>M2nKε2+A2+1/ε+σG/ε3/2</td><td>Assumption 3</td></tr><tr><td>Balakrishnan et al. (2021)</td><td>Strongly convex</td><td>✓</td><td>×</td><td>1/ε+1/φ</td><td>Heterogeneity Gap</td></tr><tr><td>Cho et al. (2020)</td><td>Strongly convex</td><td>✓</td><td>×</td><td>σL2+G2/ε+φ</td><td>Heterogeneity Gap</td></tr><tr><td>Yang et al. (2021)</td><td>Nonconvex</td><td>×</td><td>✓</td><td>σL2mKε2+σL2/(4K)+σG/ε</td><td>σG bound</td></tr><tr><td>Karimireddy et al. (2020b)</td><td>Strongly Convex</td><td>×</td><td>✓</td><td>σL2+σG2/μmKε+σL2+σG/μ√ε+ m(A2+1)/μ</td><td>Assumption 3</td></tr></table>

[ M = \sigma_L^2 + 4K\sigma_G^2, \hat{M}^2 = \sigma_L^2 + K(1 - n / m)\sigma_G^2, \hat{M}^2 = \sigma_L^2 + 6K\sigma_G^2, \hat{M}^2 = \sigma_L^2 + 4K\zeta_G^2. ]  
$\rho$  assumption: A bound of the similarity among local gradients in Chen et al. (2020) Another FedIS(others) (Chen et al., 2020) has the same convergence rate as Karimireddy et al. (2020b) under the  $\rho$  assumption. While FedIS(ours) uses a looser Assumption 4 and achieves a faster rate than Chen et al. (2020).

Assumption 3 (Bound Dissimilarity). There exists constant  $\sigma_G \geq 0$  and  $A \geq 0$  s.t.  $\mathbb{E} \| \nabla F_i(x) \|^2 \leq (A^2 + 1) \| \nabla f(x) \|^2 + \sigma_G^2$ . When all local loss functions are identical,  $A^2 = 0$  and  $\sigma_G^2 = 0$ .

The above assumptions are commonly used in both non-convex optimization and FL literature, see e.g. Karimireddy et al. (2020b); Yang et al. (2021); Koloskova et al. (2020); Wang et al. (2020); Cho et al. (2020); Li et al. (2019). For Assumption 3, if all local loss functions are identical, then we have  $A = 0$  and  $\sigma_G = 0$ .

Assumption 4 (Stochastic Gradient bound). The stochastic gradient's norm is uniformly bounded, i.e.,  $\mathbb{E}\left[\|\nabla F_i(x_{t,k},\xi_{k,t})\|\right]^2 \leq G^2$  for all  $i$ .

Assumption 4 is widely used in IS community (Stich et al., 2017; Katharopoulos & Fleuret, 2017) and also used in FL works (Reddi et al., 2020; Yang et al., 2021; Li et al., 2019) to capture the bound of stochastic gradient.

# 3.2 CONVERGENCE RATE OF FEDIS

As discussed in the introduction, IS has an excessive gradient similarity problem, which may cause redundant sampling resulting in training inefficiency. As discussed in the introduction, IS has the issue of high gradient similarity, requiring us to design a new diversity sampling method. Before going to the details of our new sampling strategy, we first provide the convergence rate of FL under standard IS analysis in this section; the analysis itself is not well explored, especially for the nonconvex setting.

Theorem 3.1 (Convergence rate of FedIS). Under Assumptions 1-4, and sampling strategy FedIS  $p_i^t = \frac{\|\hat{g}_i^t\|}{\sum_{j=1}^m \|\hat{g}_j^t\|}$ , where  $\hat{g}_i^t = \sum_{k=0}^{K-1} g_i^t = \sum_{k=0}^{K-1} \nabla F_i(x_{k,t}^i, \xi_{k,t}^i)$  is the sum of the gradient updates of multiple local updates. Let constant local and global learning rates  $\eta_L$  and  $\eta$  be chosen as such that  $\eta_L \leq \frac{1}{8LK}$ ,  $\eta_L L \leq 1$  and  $\frac{1}{2} - 10L^2 \frac{1}{m} \sum_{i=1}^{m} K^2 \eta_L^2 (A^2 + 1) > 0$ , the expected gradient norm will be bounded as follows:

$$
\min  _ {t \in [ T ]} E \| \nabla f (x _ {t}) \| ^ {2} \leq \mathcal {O} \left(\frac {f ^ {0} - f ^ {*}}{\sqrt {n K T}}\right) + \underbrace {\mathcal {O} \left(\frac {\sigma_ {L} ^ {2}}{\sqrt {n K T}}\right) + \mathcal {O} \left(\frac {M ^ {2}}{T}\right) + \mathcal {O} \left(\frac {K G ^ {2}}{\sqrt {n K T}}\right)} _ {\text {o r d e r o f} \Phi}. \tag {3}
$$

where  $f^0 = f(x_0)$ ,  $f^* = f(x_*)$ ,  $c$  is a constant that satisfies  $\frac{1}{2} - 10L^2\frac{1}{m}\sum_{i=1}^{m}K^2\eta_L^2(A^2 + 1) > c > 0$ , and the expectation is over the local dataset samples among workers.

The FedIS sampling probability  $p_i^t = \frac{\|\hat{g}_i^t\|}{\sum_{j=1}^{m} \|\hat{g_j^t}\|}$  is derived from minimizing the variance of convergence w.r.t.  $p_i^t$ . The variance is

$$
\Phi = \frac {5 \eta_ {L} ^ {2} K L ^ {2}}{2} M ^ {2} + \frac {\eta \eta_ {L} L}{2 m} \sigma_ {L} ^ {2} + \frac {L \eta \eta_ {L}}{2 n K} \operatorname {V a r} \left(\frac {1}{m p _ {i} ^ {t}} \hat {g} _ {i} ^ {t}\right), \tag {4}
$$

where  $M = \sigma_L^2 + 4K\sigma_G^2$  and  $\mathrm{Var}\big(1 / (mp_i^t)\tilde{g}_i^t\big)$  is called update variance. The proof details of Theorem 3.1 and derivation of sampling probability FedIS are detailed in Appendix C and Appendix E.1.

Remark 3.2. It is worth mentioning that although a few works provide the convergence upper bound for FedIS, several limitations exist in these analyses and results.

1) Rizk et al. (2020); Luo et al. (2022) applied IS in FL to solve a convex/strongly convex problem, while we solved a nonconvex problem.  
2) In Rizk et al. (2020), their analysis result and sampling probability rely on the assumption of knowing the optimum  $x_{*}$ , which is not feasible in practice.  
3) Our analysis uses the common Assumption 1-4, while Chen et al. (2020) provides the convergence rate of nonconvex  $FL$  under the additional assumption of gradient similarity bound. Compared with Chen et al. (2020), we prove a tighter convergence upper bound for FedIS. Specifically, our convergence rate for FedIS improves from  $\mathcal{O}\left(\frac{1}{\sqrt{nKT}} + \frac{1}{T} + \frac{1}{T^{2/3}}\right)$  to  $\mathcal{O}\left(\frac{1}{\sqrt{nKT}} + \frac{1}{T}\right)$  (c.f. Table 1).

Despite the success of FedIS in reducing the variance term in the convergence rate, it is far from optimal, due to the issue of high gradient similarity and the improvement space of further minimizing the variance term (i.e., global variance  $\sigma_G$  and local variance  $\sigma_L$  in  $\Phi$ ). We will discuss how to address this challenging variance term in the next section.

# 3.3 AN IMPROVED CONVERGENCE ANALYSIS

To ease the understanding of the theoretical difference between FedIS and DELTA, as well as a better illustration of our design choice, we include an analysis flowchart in Figure 3. The detailed analysis can be found in Appendix D.

The limitations of FedIS. As identified by the Theorem 3.1 discussed above, IS suffers from excessive similar gradient selection. The variance  $\Phi$  in (4) shows that the standard IS strategy can only control the update variance  $\mathrm{Var}\left(1 / (mp_i^t)\hat{g}_i^t\right)$ , while leaving other terms in  $\Phi$  untouched, i.e.,  $\sigma_L$  and  $\sigma_G$ . Thus, the standard IS fails to handle the excessive similar gradient selection problem, and it motivates us to give a new sampling strategy below to address the issue of  $\sigma_L$  and  $\sigma_G$ .

The decomposition of the global objective. As inspired by the proof of Theorem 3.1 as well as the corresponding Lemma B.1 (stated in Appendix) proposed for unbiased sampling, the global objective can be decomposed into surrogate objective and update gap,

$$
\mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} = \mathbb {E} \left\| \nabla \tilde {f} (x _ {t}) \right\| ^ {2} + \chi_ {t} ^ {2}, \tag {5}
$$

where  $\chi_t = \mathbb{E}\left\| \nabla \tilde{f}(x_t) - \nabla f(x_t)\right\|$  is the update gap.

Intuitively, the surrogate objective is the practical objective of the participating clients in each round, while the update gap  $\chi_t$  means the update distance between partial client participation and full client participation. The convergence behavior of the update gap  $\chi_t^2$  corresponds to the update variance in  $\Phi$ , and the convergence of surrogate objective  $\mathbb{E}\left\| \nabla \tilde{f}(x_t)\right\|^2$  is dependent on the other variance terms in  $\Phi$ , i.e., local variance and global variance.

Minimizing the surrogate objective allows us further to reduce the variance term in the convergence rate, and we focus on the convergence analysis of the surrogate objective below. For the purpose of analysis, we use IS property to formulate the surrogate objective with an arbitrary unbiased sampling probability.

Surrogate objective formulation. The expression of the surrogate objective relies on the property of IS. In detail, IS aims to substitute the original sampling distribution  $p(z)$  with another arbitrary sampling distribution  $q(z)$  while keeping the expectation unchanged:  $\mathbb{E}_{q(z)}[F_i(z)] = \mathbb{E}_{p(z)}[q_i(z) / p_i(z)F_i(z)]$ . According to the Monte Carlo method, when  $q(z)$  follows the uniform distribution, we can estimate  $\mathbb{E}_{q(z)}[F_i(z)]$  by  $1/m\sum_{i=1}^{m} F_i(z)$  and  $\mathbb{E}_{p(z)}[q_i(z) / p_i(z)F_i(z)]$  by  $1/n\sum_{i\in S_t}1/mp_iF_i(z)$ , respectively, where  $m$  and  $|S_t| = n$  are sample sizes.

Based on the IS property, we formulate the surrogate objective as below:

$$
\tilde {f} \left(x _ {t}\right) = \frac {1}{n} \sum_ {i \in S _ {t}} \frac {1}{m p _ {i} ^ {t}} F _ {i} \left(x _ {t}\right), \tag {6}
$$

where  $m$  is the total number of clients,  $|S_{t}| = n$  is the number of participating clients in each round, and  $p_t^i$  is the probability that client  $i$  is selected at round  $t$ .

![](images/84534afd4cf313e356cddb9dc50d9fcc26417d4a9f30c7b5d2aa5771c430a483.jpg)  
Figure 3: Sketch of theoretical analysis flow (Compared with FedIS). The left side represents the analysis flow of FedIS, while the analysis of DELTA is shown on the right. The sampling probability difference comes from the difference in variance.

An improved rate for the global objective. Following the fact (c.f. Lemma B.2 in appendix) that:

$$
\min  _ {t \in [ T ]} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} = \min  _ {t \in [ T ]} \mathbb {E} \| \nabla \tilde {f} (x _ {t}) \| ^ {2} + \mathbb {E} \| \chi_ {t} ^ {2} \| \leq \min  _ {t \in [ T ]} 2 \mathbb {E} \| \nabla \tilde {f} (x _ {t}) \| ^ {2}, \tag {7}
$$

the convergence rate of the global objective can be formulated as follows:

Theorem 3.3 (Convergence rate). Under Assumption 1-3 and let local and global learning rates  $\eta$  and  $\eta_L$  satisfy  $\eta_L < \frac{1}{\sqrt{20K}L}\sqrt{\frac{1}{n}\sum_{l=1}^{m}\frac{1}{mp_l^t}}$  and  $\eta \eta_L \leq \frac{1}{KL}$ , the minimal gradient norm will be bounded as below:

$$
\min  _ {t \in [ T ]} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} \leq \frac {f ^ {0} - f ^ {*}}{c \eta \eta_ {L} K T} + \frac {\tilde {\Phi}}{c}, \tag {8}
$$

where  $f^0 = f(x_0)$ ,  $f^* = f(x_*)$ ,  $c$  is a constant, and the expectation is over the local dataset samples among all workers. The combination of variance  $\tilde{\Phi}$  represents combinations of local variance and client gradient diversity.

We derive the convergence rates for both sampling with replacement and sampling without replacement. For sampling without replacement:

$$
\tilde {\Phi} = \frac {5 L ^ {2} K \eta_ {L} ^ {2}}{2 m n} \sum_ {i = 1} ^ {m} \frac {1}{p _ {i} ^ {t}} \left(\sigma_ {L, i} ^ {2} + 4 K \zeta_ {G, i} ^ {2}\right) + \frac {L \eta_ {L} \eta}{2 n} \sum_ {i = 1} ^ {m} \frac {1}{m ^ {2} p _ {i} ^ {t}} \sigma_ {L, i} ^ {2}, \tag {9}
$$

For sampling with replacement,

$$
\tilde {\Phi} = \frac {5 L ^ {2} K \eta_ {L} ^ {2}}{2 m ^ {2}} \sum_ {i = 1} ^ {m} \frac {1}{p _ {i} ^ {t}} \left(\sigma_ {L, i} ^ {2} + 4 K \zeta_ {G, i} ^ {2}\right) + \frac {L \eta_ {L} \eta}{2 n} \sum_ {i = 1} ^ {m} \frac {1}{m ^ {2} p _ {i} ^ {t}} \sigma_ {L, i} ^ {2} \tag {10}
$$

where  $\zeta_{G,i} = \| \nabla F_i(x_t) - \nabla f(x_t)\|$ . The proof details of Theorem 3.3 can be found in Appendix D.

# 3.4 OUR PROPOSED SAMPLING STRATEGY: DELTA

The update difference between the surrogate objective and the global objective can be defined as objective inconsistency. As demonstrated in Figure 4, different sampling methods lead to different degrees of objective inconsistency, and such inconsistency can be alleviated by choosing clients with a small updating gap. Figure 4(a) uses a toy example of square functions to illustrate the objective inconsistency when two out of three clients are selected for training, where DELTA would sample diverse clients, leading to a small update gap. Figure 4(b) shows the one single round update process of different sampling schemes: IS tends to select client 2 and client 3 whose gradient norm is large, while diversity sampling DELTA tends to select client 1 and client 3. Therefore, compared with IS, the sampled clients of DELTA have a smaller bias from the global objective, illustrating a better sampling scheme of DELTA.

To derive our sampling strategy DELTA, it is equivalent to solving an optimization problem that minimizes the variance  $\tilde{\Phi}$  w.r.t the proposed sampling probability  $p_i^t$ :

$$
\min _ {p _ {i} ^ {t}} \tilde {\Phi} \quad \mathrm {s . t .} \quad \sum_ {i = 1} ^ {m} p _ {i} ^ {t} = 1,
$$

where  $\tilde{\Phi}$  is a linear combination of local variance  $\sigma_{L,i}$  and gradient diversity  $\zeta_{G,i}$  (cf. Theorem 3.3).

![](images/f88ac748a9d5701bc27d2300ecdc47b477630379f081a1f42603aeef5ef2b7f5.jpg)  
(a) Objective inconsistency and update gap.

![](images/3307538db2e60302c566c607990815ecb2de7c22a3f541aa53aaa0994c1da983.jpg)  
Figure 4: (a): Overview of objective inconsistency and update gap. Here is three square functions with expression  $y = 10x^{2}$  and  $y = 3(x\pm 8)^{2}$ , and gradient is calculated at  $x = -2$ . The detail enlargement shows the objective inconsistency. (b): Illustration of the different sampling methods. The client's update is shown by the grey arrow and the ideal global update is the black arrow. It shows our DELTA is better than FedIS and FedAvg.  
(b) Illustration of different sampling methods.

# Algorithm 1 DELTA

Require: initial weights  $x_0$ , global learning rate  $\eta$ , local learning rate  $\eta_l$ , number of training rounds  $T$

Ensure: trained weights  $x_{T}$

1: for round  $t = 1, \dots, T$  do  
2: Select a subset of clients according to the proposed sampling probability of DELTA (11)  
3: for each worker  $i\in S_t$  ,in parallel do  
4:  $\left| x_{t,0}^{i} = x_{t} \right|$  
5: for  $k = 0,\dots ,K - 1$  do  
6: compute  $g_{t,k}^{i} = \nabla F_{i}(x_{t,k}^{i},\xi_{t,k}^{i})$  
7: Local update:  $x_{t,k+1}^i = x_{t,k}^i - \eta_L g_{t,k}^i$  
8: Let  $\Delta_t^i = x_{t,K}^i -x_{t,0}^i = -\eta_L\sum_{k = 0}^{K - 1}g_{t,k}^i$  
9: Send gradient to server  
10: At Server:  
11: Receive  $\Delta_t^i, i \in S_t$  
12: let  $\Delta_t = \frac{1}{|S_t|}\sum_{i\in S_t}\frac{n_i}{np_i^2}\Delta_t^i$  
13: Server update:  $x_{t + 1} = x_t + \eta \Delta_t$  
14: Broadcast  $x_{t+1}$  to clients

Corollary 3.4 (Optimal sampling probability for DELTA). By solving the above optimization problem, the optimal sampling probability can be formulated as:

$$
p _ {i} ^ {t} = \frac {\sqrt {\alpha_ {1} \left\| \nabla F _ {i} (x) - \nabla f (x) \right\| ^ {2} + \alpha_ {2} \sigma_ {L , i} ^ {2}}}{\sum_ {j = 1} ^ {m} \sqrt {\alpha_ {1} \left\| \nabla F _ {j} (x) - \nabla f (x) \right\| ^ {2} + \alpha_ {2} \sigma_ {L , j} ^ {2}}}, \tag {11}
$$

where  $\alpha_{1}$  and  $\alpha_{2}$  are constants defined as  $\alpha_{1} = 20K^{2}L\eta_{L}$  and  $\alpha_{2} = 5KL\eta_{L} + \frac{\eta}{n}$ .

Let  $\eta_L = \mathcal{O}\left(\frac{1}{\sqrt{T}KL}\right)$ ,  $\eta = \mathcal{O}\left(\sqrt{Kn}\right)$  and substitute the optimal sampling probability (11) back to  $\tilde{\Phi}$ . Then for sufficiently large  $T$ , the iterates of Theorem 3.3 satisfy:

$$
\min  _ {t \in [ T ]} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} \leq \mathcal {O} \left(\frac {f ^ {0} - f ^ {*}}{\sqrt {n K T}}\right) + \underbrace {\mathcal {O} \left(\frac {\sigma_ {L} ^ {2}}{\sqrt {n K T}}\right) + \mathcal {O} \left(\frac {\sigma_ {L} ^ {2} + 4 K \zeta_ {G , i} ^ {2}}{K T}\right)} _ {\text {o r d e r o f} \bar {\Phi}}. \tag {12}
$$

# 3.5 DISCUSSIONS

Difference between DELTA and FedIS. The difference between DELTA and FedIS comes mainly from the difference between  $\tilde{\Phi}$  and  $\Phi$ . FedIS aims to reduce the update variance term  $\mathrm{Var}\left(1 / (m p_i^t)\hat{g}_i^t\right)$  in  $\Phi$ , while DELTA aims to reduce the whole  $\tilde{\Phi}$  which is composed of the gradient diversity and the local variance. Minimizing  $\tilde{\Phi}$  corresponds to further minimizing the terms of  $\Phi$  that can not be minimized by FedIS. Solving different optimization problems leads to different sampling probability expressions. As shown in Figure 4, DELTA selects the more diverse Client 1 and Client 3 for

participation, while FedIS tends to select Client 2 and Client 3 which have large gradient norms. It can be seen that the selection of DELTA leads to a smaller bias than FedIS. Moreover, as shown in Table 1, based on our convergence rate results, DELTA achieves a better convergence rate with  $\mathcal{O}(G^2/\epsilon^2)$  higher than other unbiased sampling algorithms.

Compare DELTA with uniform sampling. According to the Cauchy-Schwarz inequality, DELTA is at least better than uniform sampling by reducing variance:  $\frac{\tilde{\Phi}_{\text{uniform}}}{\tilde{\Phi}_{\text{DELTA}}} = \frac{m \sum_{i=1}^{m} \left( \sqrt{\alpha_1 \sigma_L^2 + \alpha_2 \zeta_{G,i}^2} \right)^2}{\left( \sum_{i=1}^{m} \sqrt{\alpha_1 \sigma_L^2 + \alpha_2 \zeta_{G,i}^2} \right)^2} \geq 1$ . This implies that DELTA does reduce the variance, especially when  $\frac{\left( \sum_{i=1}^{m} \sqrt{\alpha_1 \sigma_L^2 + \alpha_2 \zeta_{G,i}^2} \right)^2}{\sum_{i=1}^{m} \left( \sqrt{\alpha_1 \sigma_L^2 + \alpha_2 \zeta_{G,i}^2} \right)^2} \ll m$ .

Remark 3.5. DELTA ensures the convergence of FL with partial client participation to a stationary point without any gap. Our results can be considered as a theoretical explanation for the heuristic of gradient diversity sampling algorithm in FL, and DELTA encourages the global model to acquire more knowledge in each round. Specifically, the server will give more weight to the clients with larger gradient diversity and local variance. These clients are representative, and sampling these clients can accelerate training given the more diverse and informative data to reflect the global data distribution. However, DELTA may fail to identify the attacked clients and even tends to select them when it comes to user attack scenarios. We will leave the solution for this scenario in our future work.

# 4 PRACTICAL IMPLEMENTATION FOR DELTA AND FEDIS

Gradient-norm-based sampling method requires the computation of the full gradient in each iteration (Elvira & Martino, 2021; Zhao & Zhang, 2015). However, obtaining each client's gradient in advance is generally inadmissible in FL. For practical purposes, a series of IS algorithms estimate the current round's gradient by the historical gradient (Cho et al., 2020; Katharopoulos & Fleuret, 2017). Similarly, we utilize the gradient from the previous training iteration to estimate the gradient of the current round, where the previous iteration refers to the one in which the client participates. By using this approximation method, we can save computing resources (Rizk et al., 2020).

In particular, at iteration 0, all probabilities are set to  $1 / m$ , then during the  $i_{th}$  iteration, after the participating clients  $i \in S_t$  send the server their updated gradients, the sampling probabilities are updated as:  $p_{i,t + 1}^* = \frac{\|g_{\hat{i},t}\|}{\sum_{i \in S_t} \|g_{\hat{i},t}\|} (1 - \sum_{i \in S_t^c} p_{i,t}^*)$ , where the multiplicative factor follows from ensuring all the probabilities sum to 1. Specifically, we use the average of the latest participated clients' gradients to approximate the true gradient of the global model for DELTA. In this way, it is not necessary to obtain all clients' gradients in each round.

# 5 EXPERIMENTS

In this section, we use both synthetic dataset and split FEMNIST to demonstrate our theoretical results. To show the validity of the practical algorithm, we run experiments on FEMNIST and CIFAR-10, and show that DELTA converges faster and achieve higher accuracy than other baselines.

Synthetic datasets. We first examine our theoretical results through logistic regression on synthetic datasets. In details, we randomly generate  $(x,y)$  by  $y = \log ((^{Ax - b})^2 /2)$  with given  $A_{i}$  and  $b_{i}$  as training data for clients, and each client's local dataset contains 1000 samples. In each round, 10 out of 20 clients are selected to participate in training (we also provide the results of 10 out of 200 clients in Appendix F). To simulate the gradient noise, in each training step, we calculate the gradient of client  $i$  by  $g_{i} = \nabla f_{i}(A_{i},b_{i},D_{i}) + \nu_{i}$ , where  $A_{i}$  and  $b_{i}$  are model parameters,  $D_{i}$  is the local dataset of client  $i$ , and  $\nu_{i}$  is a zero-mean random variable which controls the heterogeneity of client  $i$ . The larger the  $\mathbb{E}\| \nu_i\|^2$ , the larger the heterogeneity of client  $i$ .

Figure 5 demonstrates that these empirical results align with our theoretical analysis. Additional experiments of different functions and different settings can be found in Appendix F. In detail,

- DELTA and FedIS outperform other biased and unbiased methods in convergence speed. We can see both DELTA and FedIS converge faster than both FedAvg and Power-of-choice sampling. The larger the noise (variance), the more obvious the convergence speed advantage of DELTA and FedIS. For  $\nu = 30$ , FedIS can achieve near twice faster than FedAvg, and for  $\nu = 40$ , DELTA can achieve nearly  $4\times$  times faster than FedAvg.

![](images/3439d34ed8970573b61e65b0140fc357bc014be3b913f5a9cef16009fd96f080.jpg)  
(a)  $\nu = 20$

![](images/70780331b02bde697bff70462c7d4e0c2ae87ab31204a09cb817b4949d74481e.jpg)  
(b)  $\nu = 30$

![](images/024391a84fd2899888fa531b454670c256141d7114d8824c364755f606511b3b.jpg)  
Figure 5: Performance of different algorithms on the regression model. The loss is calculated by  $f(x, y) = \left\| y - \log \left( (A_i x - b_i)^2 / 2 \right) \right\|^2$ ,  $A = 10$ ,  $b = 1$ . We report the logarithm of global loss with different degrees of gradient noise  $\nu$ . All methods are well-tuned, and we report the best result of each algorithm under each setting.  
(c)  $\nu = 40$

Table 2: Performance of algorithms. We run 500 communication rounds on FEMNIST and CIFAR10 for each algorithm. We report the mean of maximum 5 accuracies for test datasets and the average number of communication rounds to reach the threshold accuracy.  

<table><tr><td rowspan="2">Algorithm</td><td colspan="2">FEMNIST α = 0.1</td><td colspan="2">CIFAR10 α = 0.5</td></tr><tr><td>Acc (%)</td><td>Rounds for 70%</td><td>Acc (%)</td><td>Rounds for 54%</td></tr><tr><td>FedAvg (w/ uniform sampling)</td><td>70.35± 0.51</td><td>426</td><td>54.28± 0.29</td><td>338</td></tr><tr><td>FedIS</td><td>71.69± 0.43</td><td>404</td><td>55.05± 0.27</td><td>313</td></tr><tr><td>DELTA</td><td>72.10± 0.49</td><td>322</td><td>55.20± 0.26</td><td>303</td></tr></table>

- DELTA outperforms FedIS. In experiments, DELTA converges about twice faster as FedIS in Figure 5(a). As all results show, DELTA can reduce more variance than FedIS and thus converge a smaller loss.

Split FEMNIST In this section, we consider the split FEMNIST. We let  $10\%$  clients own  $90\%$  data and the detailed split algorithm is provided in Appendix F. Figure 6 shows that when the data distribution is highly heterogeneous, Our DELTA algorithm converges faster than other baselines.

FEMNIST and CIFAR-10. We also verify our practical algorithm on FEMNIST and CIFAR-10. We summarize our numerical results in Table 2: DELTA has better accuracy than FedIS, while DELTA and FedIS both outperform FedAvg with the same communication round. This demonstrates the practicality of our method.

We also test different choices of the number of participated clients  $n$  and test on different heterogeneity  $\alpha$ , and observe the consistent improvement of DELTA. The detailed setting and additional experiments are in Appendix F.

![](images/a65c4a8e87db31d147d3ced6c0338e5d00260b1fd515b5950ede68269311df6c.jpg)  
Figure 6: Performance of different sampling methods on Split FEMNIST dataset

# 6 CONCLUSION AND FUTURE WORK

In this work, we studied the optimal client

sampling strategy that addresses the data heterogeneity to fast the convergence speed of FL. We obtain a new tractable convergence rate for nonconvex FL algorithms with arbitrary client sampling probabilities. Based on the bound, we solve an optimization problem with respect to sampling probability and thus develop a novel unbiased sampling scheme that characterizes the impact of client diversity and local variance on the sampling design. Experimental results validated the superiority of our theoretical and practical algorithms compared to several baselines.

As we point out, when user attacks occur, DELTA requires some changes to be able to identify and avoid selecting users from these attacks. There is still much potential to study the adaptive implementation of our theory, which is also an open challenge in the IS community for giving an adaptive practical algorithm, even if our work has provided a novel sampling method and gives an effective practical version of our algorithm. We will leave these in our future work.

# REFERENCES

Ravikumar Balakrishnan, Tian Li, Tianyi Zhou, Mageen Himayat, Virginia Smith, and Jeff Bilmes. Diverse client selection for federated learning: Submodularity and convergence analysis. In ICML 2021 International Workshop on Federated Learning for User Privacy and Data Confidentiality, Virtual, July 2021.  
Wenlin Chen, Samuel Horvath, and Peter Richtarik. Optimal client sampling for federated learning. arXiv preprint arXiv:2010.13723, 2020.  
Yae Jee Cho, Jianyu Wang, and Gauri Joshi. Client selection in federated learning: Convergence analysis and power-of-choice selection strategies. arXiv preprint arXiv:2010.01243, 2020.  
Enmao Diao, Jie Ding, and Vahid Tarokh. Heterofl: Computation and communication efficient federated learning for heterogeneous clients. arXiv preprint arXiv:2010.01264, 2020.  
Víctor Elvira and Luca Martino. Advances in importance sampling. arXiv preprint arXiv:2102.05407, 2021.  
Yann Fraboni, Richard Vidal, Laetitia Kameni, and Marco Lorenzi. Clustered sampling: Low-variance and improved representativity for clients selection in federated learning, 2021a.  
Yann Fraboni, Richard Vidal, Laetitia Kameni, and Marco Lorenzi. A general theory for client sampling in federated learning, 2021b. URL https://arxiv.org/abs/2107.12211.  
Yongxin Guo, Tao Lin, and Xiaoying Tang. Towards federated learning on time-evolving heterogeneous data. arXiv preprint arXiv:2112.13246, 2021.  
Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335, 2019.  
Divyansh J Hunjhunwala, PRANAY SHARMA, Aushim Nagarkatti, and Gauri Joshi. Fedvarp: Tackling the variance due to partial client participation in federated learning. In The 38th Conference on Uncertainty in Artificial Intelligence, 2022.  
Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank J Reddi, Sebastian U Stich, and Ananda Theertha Suresh. Mime: Mimicking centralized stochastic algorithms in federated learning. arXiv preprint arXiv:2008.03606, 2020a.  
Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In International Conference on Machine Learning, pp. 5132-5143. PMLR, 2020b.  
Angelos Katharopoulos and François Fleuret. Biased importance sampling for deep neural network training. arXiv preprint arXiv:1706.00043, 2017.  
Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian Stich. A unified theory of decentralized sgd with changing topology and local updates. In International Conference on Machine Learning, pp. 5381-5393. PMLR, 2020.  
Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. arXiv preprint arXiv:1812.06127, 2018.  
Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of fedavg on non-iid data. arXiv preprint arXiv:1907.02189, 2019.  
Tao Lin, Lingjing Kong, Sebastian U Stich, and Martin Jaggi. Ensemble distillation for robust model fusion in federated learning. arXiv preprint arXiv:2006.07242, 2020.  
Bing Luo, Wenli Xiao, Shiqiang Wang, Jianwei Huang, and Leandros Tassiulas. Tackling system and statistical heterogeneity for federated learning with adaptive client sampling. In IEEE INFOCOM 2022-IEEE Conference on Computer Communications, pp. 1739-1748. IEEE, 2022.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273-1282. PMLR, 2017.  
Aritra Mitra, Rayana Jaafar, George J Pappas, and Hamed Hassani. Achieving linear convergence in federated learning under objective and systems heterogeneity. arXiv preprint arXiv:2102.07053, 2021.  
Ihab Mohammed, Shadha Tabatabai, Ala Al-Fuqaha, Faissal El Bouanani, Junaid Qadir, Basheer Qolomany, and Mohsen Guizani. Budgeted online selection of candidate iot clients to participate in federated learning. IEEE Internet of Things Journal, 8(7):5938-5952, 2021. doi: 10.1109/JIOT.2020.3036157.  
Khalil Muhammad, Qinqin Wang, Diarmuid O'Reilly-Morgan, Elias Tragos, Barry Smyth, Neil Hurley, James Geraci, and Aonghus Lawlor. Fedfast: Going beyond average for faster training of federated recommender systems. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1234-1242, 2020.  
Zhe Qu, Rui Duan, Lixing Chen, Jie Xu, Zhuo Lu, and Yao Liu. Context-aware online client selection for hierarchical federated learning, 2021.  
Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konečný, Sanjiv Kumar, and H Brendan McMahan. Adaptive federated optimization. arXiv preprint arXiv:2003.00295, 2020.  
Monica Ribero and Haris Vikalo. Communication-efficient federated learning via optimal client sampling. arXiv preprint arXiv:2007.15197, 2020.  
Elsa Rizk, Stefan Vlaski, and Ali H Sayed. Federated learning under importance sampling. arXiv preprint arXiv:2012.07383, 2020.  
Guangyuan Shen, Dehong Gao, DuanXiao Song, Xukai Zhou, Shirui Pan, Wei Lou, Fang Zhou, et al. Fast heterogeneous federated learning with hybrid client selection. arXiv preprint arXiv:2208.05135, 2022.  
Sebastian U Stich, Anant Raj, and Martin Jaggi. Safe adaptive importance sampling. arXiv preprint arXiv:1711.02637, 2017.  
Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H Vincent Poor. Tackling the objective inconsistency problem in heterogeneous federated optimization. arXiv preprint arXiv:2007.07481, 2020.  
Jianyu Wang, Zachary Charles, Zheng Xu, Gauri Joshi, H Brendan McMahan, Maruan Al-Shedivat, Galen Andrew, Salman Avestimehr, Katharine Daly, Deepesh Data, et al. A field guide to federated optimization. arXiv preprint arXiv:2107.06917, 2021.  
Xiaohui Xu, Sijing Duan, Jinrui Zhang, Yunzhen Luo, and Deyu Zhang. Optimizing federated learning on device heterogeneity with a sampling strategy. In 2021 IEEE/ACM 29th International Symposium on Quality of Service (IWQOS), pp. 1-10, 2021. doi: 10.1109/IWQOS52092.2021.9521361.  
Haibo Yang, Minghong Fang, and Jia Liu. Achieving linear speedup with partial worker participation in non-iid federated learning. arXiv preprint arXiv:2101.11203, 2021.  
Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel restarted sgd with faster convergence and less communication: Demystifying why model averaging works for deep learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 5693-5700, 2019.  
Peilin Zhao and Tong Zhang. Stochastic optimization with importance sampling for regularized loss minimization. In international conference on machine learning, pp. 1-9. PMLR, 2015.
