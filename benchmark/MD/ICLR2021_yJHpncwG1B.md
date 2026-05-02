# FEDERATED LEARNING'S BLESSING: FEDAVG HAS LINEAR SPEEDDUP

Anonymous authors

Paper under double-blind review

# ABSTRACT

Federated learning (FL) learns a model jointly from a set of participating devices without sharing each other's privately held data. The characteristics of non-i.i.d. data across the network, low device participation, high communication costs, and the mandate that data remain private bring challenges in understanding the convergence of FL algorithms, particularly in regards to how convergence scales with the number of participating devices. In this paper, we focus on Federated Averaging (FedAvg)—arguably the most popular and effective FL algorithm class in use today—and provide a unified and comprehensive study of its convergence rate. Although FedAvg has recently been studied by an emerging line of literature, it remains open as to how FedAvg's convergence scales with the number of participating devices in the fully heterogeneous FL setting—a crucial question whose answer would shed light on the performance of FedAvg in large FL systems. We fill this gap by providing a unified analysis that establishes convergence guarantees for FedAvg under three classes of problems: strongly convex smooth, convex smooth, and overparameterized strongly convex smooth problems. We show that FedAvg enjoys linear speedup in each case, although with different convergence rates and communication efficiencies. While there have been linear speedup results from distributed optimization that assumes full participation, ours are the first to establish linear speedup for FedAvg under both statistical and system heterogeneity. For strongly convex and convex problems, we also characterize the corresponding convergence rates for the Nesterov accelerated FedAvg algorithm, which are the first linear speedup guarantees for momentum variants of FedAvg in the convex setting. To provably accelerate FedAvg, we design a new momentum-based FL algorithm that further improves the convergence rate in overparameterized linear regression problems. Empirical studies of the algorithms in various settings have supported our theoretical results.

# 1 INTRODUCTION

Federated learning (FL) is a machine learning paradigm where many clients (e.g., mobile devices or organizations) collaboratively train a model under the orchestration of a central server (e.g., service provider), while keeping the training data decentralized (Smith et al. (2017); Kairouz et al. (2019)). In recent years, FL has swiftly emerged as an important learning paradigm (McMahan et al. (2017); Li et al. (2020a))—one that enjoys widespread success in applications such as personalized recommendation (Chen et al. (2018)), virtual assistant (Lam et al. (2019)), and keyboard prediction (Hard et al. (2018)), to name a few—for at least three reasons: First, the rapid proliferation of smart devices that are equipped with both computing power and data-capturing capabilities provided the infrastructure core for FL. Second, the rising awareness of privacy and the explosive growth of computational power in mobile devices have made it increasingly attractive to push the computation to the edge. Third, the empirical success of communication-efficient FL algorithms has enabled increasingly larger-scale parallel computing and learning with less communication overhead.

Despite its promise and broad applicability in our current era, the potential value FL delivers is coupled with the unique challenges it brings forth. In particular, when FL learns a single statistical model using data from across all the devices while keeping each individual device's data isolated (Kairouz et al. (2019)), it faces two challenges that are absent in centralized optimization and distributed (stochastic) optimization (Zhou & Cong (2018); Stich (2019); Khaled et al. (2019); Liang et al. (2019); Wang & Joshi (2018); Woodworth et al. (2018); Wang et al. (2019); Jiang & Agrawal (2018); Yu et al. (2019b;a); Khaled et al. (2020); Koloskova et al. (2020)):

1) Data (statistical) heterogeneity: data distributions in devices are different (and data cannot be shared);

<table><tr><td>Objective function Participation</td><td>Strongly Convex</td><td>Convex</td><td>Overparameterized general case</td><td>Overparameterized linear regression</td></tr><tr><td>Full</td><td>O(1/NT + E2/T2)</td><td>O(1/√NT + NE2/T)</td><td>O(exp(-NT/Eκ1))</td><td>O(exp(-NT/Eκ))†</td></tr><tr><td>Partial</td><td>O(E2/KT + E2/T2)</td><td>O(E2/√KT + KE2/T)</td><td>O(exp(-KT/Eκ1))</td><td>O(exp(-KT/Eκ))†</td></tr></table>

Table 1: Our convergence results for FedAvg and accelerated FedAvg in this paper. Throughout the paper,  $N$  is the total number of local devices, and  $K \leq N$  is the maximal number of devices that are accessible to the central server.  $T$  is the total number of stochastic updates performed by each local device,  $E$  is the local steps between two consecutive server communications (and hence  $T / E$  is the number of communications). In the linear regression setting, we have  $\kappa = \kappa_{1}$  for FedAvg and  $\kappa = \sqrt{\kappa_{1} \widetilde{\kappa}}$  for momentum accelerated FedAvg (FedMaSS), where  $\kappa_{1}$  and  $\sqrt{\kappa_{1} \widetilde{\kappa}}$  are condition numbers defined in Section G. Since  $\kappa_{1} \geq \widetilde{\kappa}$ , this implies a speedup factor of  $\sqrt{\frac{\kappa_{1}}{\widetilde{\kappa}}}^{\frac{K}{\kappa}}$  for accelerated FedAvg.

2) System heterogeneity: only a subset of devices may access the central server at each time both because the communications bandwidth profiles vary across devices and because there is no central server that has control over when a device is active (the presence of "stragglers").

To address these challenges, Federated Averaging (FedAvg) McMahan et al. (2017) was proposed as a particularly effective heuristic, which has enjoyed great empirical success. This success has since motivated a growing line of research efforts into understanding its theoretical convergence guarantees in various settings. For instance, Haddadpour & Mahdavi (2019) analyzed FedAvg (for non-convex smooth problems satisfying PL conditions) under the assumption that each local device's minimizer is the same as the minimizer of the joint problem (if all devices' data is aggregated together), an overly restrictive assumption that restricts the extent of data heterogeneity. Very recently, Li et al. (2020b) furthered the progress and established an  $\mathcal{O}\left(\frac{1}{T}\right)$  convergence rate for FedAvg for strongly convex smooth Federated learning problems with both data and system heterogeneity. A similar result in the same setting Karimireddy et al. (2019) also established an  $\mathcal{O}\left(\frac{1}{T}\right)$  result that allows for a linear speedup when the number of participating devices is large. At the same time, Huo et al. (2020) studied the Nesterov accelerated FedAvg for non-convex smooth problems and established an  $\mathcal{O}\left(\frac{1}{\sqrt{T}}\right)$  convergence rate to stationary points.

However, despite these very recent fruitful pioneering efforts into understanding the theoretical convergence properties of FedAvg, it remains open as to how the number of devices—particularly the number of devices that participate in the computation— affects the convergence speed. In particular, is linear speedup of FedAvg a universal phenomenon across different settings and for any number of devices? What about when FedAvg is accelerated with momentum updates? Does the presence of both data and system heterogeneity in FL imply different communication complexities and require technical novelties over results in distributed and decentralized optimization? These aspects are currently unexplored or underexplored in FL. We fill in the gaps here by providing affirmative answers.

Our Contributions We provide a comprehensive and unified convergence analysis of FedAvg and its accelerated variants in the presence of both data and system heterogeneity. Our contributions are threefold.

First, we establish an  $\mathcal{O}(1 / KT)$  convergence rate under FedAvg for strongly convex and smooth problems and an  $\mathcal{O}(1 / \sqrt{KT})$  convergence rate for convex and smooth problems (where  $K$  is the number of participating devices), thereby establishing that FedAvg enjoys the desirable linear speedup property in the FL setup. Prior to our work here, the best and the most related convergence analysis is given by Li et al. (2020b) and Karimireddy et al. (2019), which established an  $\mathcal{O}\left(\frac{1}{T}\right)$  convergence rate for strongly convex smooth problems under FedAvg. Our rate matches the same (and optimal) dependence on  $T$ , but also completes the picture by establishing the linear dependence on  $K$ , for any  $K \leq N$ , where  $N$  is the total number of devices, whereas Li et al. (2020b) does not have linear speedup analysis, and Karimireddy et al. (2019) only allows linear speedup close to full participation ( $K = \mathcal{O}(N)$ ). As for convex and smooth problems, there was no prior work that established the  $\mathcal{O}\left(\frac{1}{\sqrt{T}}\right)$  rate under both system and data heterogeneity. Our unified analysis highlights the common elements and distinctions between the strongly and convex settings.

Second, we establish the same convergence rates  $\mathcal{O}(1 / K T)$  for strongly convex and smooth problems and  $\mathcal{O}(1 / \sqrt{K T})$  for convex and smooth problems—for Nesterov accelerated FedAvg. We analyze the accelerated version of FedAvg here because empirically it tends to perform better; yet, its theoretical

convergence guarantee is unknown. To the best of our knowledge, these are the first results that provide a linear speedup characterization of Nesterov accelerated FedAvg in those two problem classes (that FedAvg and Nesterov accelerated FedAvg share the same convergence rate is to be expected: this is the case even for centralized stochastic optimization). Prior to our results here, the most relevant results Yu et al. (2019a); Li et al. (2020a); Huo et al. (2020) only concern the non-convex setting, where convergence is measured with respect to stationary points (vanishing of gradient norms, rather than optimality gaps). Our unified analysis of Nesterov FedAvg also illustrates the technical similarities and distinctions compared to the original FedAvg algorithm, whereas prior works (in the non-convex setting) were scattered and used different notations.

Third, we study a subclass of strongly convex smooth problems where the objective is overparameterized and establish a faster  $\mathcal{O}(\exp(-\frac{KT}{\kappa}))$  convergence rate for FedAvg, in contrast to the  $\mathcal{O}(\exp(-\frac{T}{\kappa}))$  rate for individual solvers Ma et al. (2018). Within this class, we further consider the linear regression problem and establish an even sharper rate under FedAvg. In addition, we propose a new variant of accelerated FedAvg based on a momentum update of Liu & Belkin (2020)-MaSS accelerated FedAvg-and establish a faster convergence rate (compared to if no acceleration is used). This stands in contrast to generic (strongly) convex stochastic problems where theoretically no rate improvement is obtained when one accelerates FedAvg. The detailed convergence results are summarized in Table 1.

Connections with Distributed and Decentralized Optimization Federated learning is closely related to distributed and decentralized optimization, and as such it is important to discuss connections and distinctions between our work and related results from that literature. First, when there is neither system heterogeneity, i.e. all devices participate in parameter averaging during a communication round, nor statistical heterogeneity, i.e. all devices have access to a common set of stochastic gradients, FedAvg coincides with the "Local SGD" of Stich (2019), which showed the linear speedup rate  $\mathcal{O}(1/NT)$  for strongly convex and smooth functions. When there is only data heterogeneity, some works have continued to use the term Local SGD to refer to FedAvg, while others subsume it in more general frameworks that include decentralized model averaging based on a network topology or a mixing matrix. They have provided linear speedup analyses for strongly convex and convex problems, e.g. Khaled et al. (2020); Koloskova et al. (2020) as well as non-convex problems, e.g. Jiang & Agrawal (2018); Yu et al. (2019b); Wang & Joshi (2018). However, these results do not consider system heterogeneity, i.e. the presence of stragglers in the device network. Even with decentralized model averaging, the assumptions usually imply that model averages over all devices is the same as decentralized model averages based on network topology (e.g. Koloskova et al. (2020) Proposition 1), which precludes system heterogeneity as defined in this paper and prevalent in FL problems. For momentum accelerated FedAvg, Yu et al. (2019a) provided linear speedup analysis for non-convex problems, while results for strongly convex and convex settings are entirely lacking, even without system heterogeneity. Karimireddy et al. (2019) considers both types of heterogeneities, but their rate implies a linear speedup only when the number of stragglers is negligible. In contrast, our linear speedup analyses consider both types of heterogeneity present in the full federated learning setting, and are valid for any number of participating devices. We also highlight a distinction in communication efficiency when system heterogeneity is present. Moreover, our results for Nesterov accelerated FedAvg completes the picture for strongly convex and convex problems. For a detailed comparison with related works, please refer to Table 2 in Appendix Section B.

# 2 SETUP

In this paper, we study the following federated learning problem:

$$
\min  _ {\mathbf {w}} \left\{F (\mathbf {w}) \triangleq \sum_ {k = 1} ^ {N} p _ {k} F _ {k} (\mathbf {w}) \right\}, \tag {1}
$$

where  $N$  is the number of local devices (users/nodes/workers) and  $p_k$  is the  $k$ -th device's weight satisfying  $p_k \geq 0$  and  $\sum_{k=1}^{N} p_k = 1$ . In the  $k$ -th local device, there are  $n_k$  data points:  $\mathbf{x}_k^1, \mathbf{x}_k^2, \ldots, \mathbf{x}_k^{n_k}$ . The local objective  $F_k(\cdot)$  is defined as:  $F_k(\mathbf{w}) \triangleq \frac{1}{n_k} \sum_{j=1}^{n_k} \ell(\mathbf{w}; \mathbf{x}_k^j)$ , where  $\ell$  denotes a user-specified loss function. Each device only has access to its local data, which gives rise to its own local objective  $F_k$ . Note that we do not make any assumptions on the data distributions of each local

device. The local minimum  $F_{k}^{*} = \min_{\mathbf{w}} F_{k}(\mathbf{w})$  can be far from the global minimum of Eq (1) (data heterogeneity).

# 2.1 THE FEDERATED AVERAGING (FEDAVG) ALGORITHM

We first introduce the standard Federated Averaging (FedAvg) algorithm which was first proposed by McMahan et al. (2017). FedAvg updates the model in each device by local Stochastic Gradient Descent (SGD) and sends the latest model to the central server every  $E$  steps. The central server conducts a weighted average over the model parameters received from active devices and broadcasts the latest averaged model to all devices. Formally, the updates of FedAvg at round  $t$  is described as follows:

$$
\mathbf {v} _ {t + 1} ^ {k} = \mathbf {w} _ {t} ^ {k} - \alpha_ {t} \mathbf {g} _ {t, k}, \quad \mathbf {w} _ {t + 1} ^ {k} = \left\{ \begin{array}{l l} \mathbf {v} _ {t + 1} ^ {k} & \text {i f} t + 1 \notin \mathcal {I} _ {E}, \\ \sum_ {k \in \mathcal {S} _ {t + 1}} q _ {k} \mathbf {v} _ {t + 1} ^ {k} & \text {i f} t + 1 \in \mathcal {I} _ {E}, \end{array} \right. \tag {2}
$$

where  $\mathbf{w}_t^k$  is the local model parameter maintained in the  $k$ -th device at the  $t$ -th iteration and  $\mathbf{g}_{t,k} := \nabla F_k(\mathbf{w}_t^k, \xi_t^k)$  is the stochastic gradient based on  $\xi_t^k$ , the data point sampled from  $k$ -th device's local data uniformly at random.  $\mathcal{I}_E = \{E, 2E, \ldots\}$  is the set of global communication steps, when local parameters from a set of active devices are averaged and broadcast to all devices. We use  $\mathcal{S}_{t+1}$  to represent the (random) set of active devices at  $t+1$ .  $q_k$  is a set of averaging weights that are specific to the sampling procedure used to obtain the set of active devices  $\mathcal{S}_{t+1}$ .

Since federated learning usually involves an enormous amount of local devices, it is often more realistic to assume only a subset of local devices is active at each communication round (system heterogeneity). In this work, we consider both the case of full participation where the model is averaged over all devices at each communication round, in which case  $q_{k} = p_{k}$  for all  $k$  and  $\mathbf{w}_{t + 1}^{k} = \sum_{k = 1}^{N}p_{k}\mathbf{v}_{t + 1}^{k}$  if  $t + 1\in \mathcal{I}_E$ , and the case of partial participation where  $|S_{t + 1}| < N$ .

With partial participation, we follow Li et al. (2020a); Karimireddy et al. (2019); Li et al. (2020b) and assume that  $S_{t + 1}$  is obtained by one of two types of sampling schemes to simulate practical scenarios. One scheme establishes  $S_{t + 1}$  by i.i.d. sampling the devices with probability  $p_k$  with replacement, and uses  $q_{k} = \frac{1}{K}$ , where  $K = |\mathcal{S}_{t + 1}|$ , while the other scheme samples  $S_{t + 1}$  uniformly i.i.d. from all devices without replacement, and uses  $q_{k} = p_{k}\frac{N}{K}$ . Both schemes guarantee that gradient updates in FedAvg are unbiased stochastic versions of updates in FedAvg with full participation, which is important in the theoretical analysis of convergence. Because the original sampling scheme and weights proposed by McMahan et al. (2017) lacks this nice property, it is not considered in this paper. For more details on the notations and setup as well as properties of the two sampling schemes, please refer to Section A in the appendix.

# 2.2 ASSUMPTIONS

We make the following standard assumptions on the objective function  $F_{1},\ldots ,F_{N}$ . Assumptions 1 and 2 are commonly satisfied by a range of popular objective functions, such as  $\ell^2$ -regularized logistic regression and cross-entropy loss functions.

Assumption 1 (L-smooth).  $F_{1},\dots ,F_{N}$  are all  $L$  -smooth: for all  $\mathbf{v}$  and  $\mathbf{w}$ ,  $F_{k}(\mathbf{v})\leq F_{k}(\mathbf{w}) + (\mathbf{v}-\mathbf{w})^{T}\nabla F_{k}(\mathbf{w}) + \frac{L}{2}\| \mathbf{v} - \mathbf{w}\|_{2}^{2}$ .

Assumption 2 (Strongly-convex).  $F_{1},\dots ,F_{N}$  are all  $\mu$  -strongly convex: for all  $\nu$  and  $\mathbf{w}$ ,  $F_{k}(\mathbf{v})\geq F_{k}(\mathbf{w}) + (\mathbf{v} - \mathbf{w})^{T}\nabla F_{k}(\mathbf{w}) + \frac{\mu}{2}\| \mathbf{v} - \mathbf{w}\|_{2}^{2}$

Assumption 3 (Bounded local variance). Let  $\xi_t^k$  be sampled from the  $k$ -th device's local data uniformly at random. The variance of stochastic gradients in each device is bounded:  $\mathbb{E}\left\| \nabla F_k(\mathbf{w}_t^k,\xi_t^k) - \nabla F_k(\mathbf{w}_t^k)\right\|^2 \leq \sigma_k^2$ , for  $k = 1, \dots, N$  and any  $\mathbf{w}_t^k$ . Let  $\sigma^2 \coloneqq \sum_{k=1}^{N} p_k \sigma_k^2$ .

Assumption 4 (Bounded local gradient). The expected squared norm of stochastic gradients is uniformly bounded. i.e.,  $\mathbb{E}\left\| \nabla F_k(\mathbf{w}_t^k,\xi_t^k)\right\|^2\leq G^2$  for all  $k = 1,\dots,N$  and  $t = 0,\ldots ,T - 1$

Assumptions 3 and 4 have been made in many previous works in federated learning, e.g. Yu et al. (2019b); Li et al. (2020b); Stich (2019). We provide further justification for their generality. As model average parameters become closer to  $\mathbf{w}^*$ , the  $L$ -smoothness property implies that  $\mathbb{E}\| \nabla F_k(\mathbf{w}_t^k,\xi_t^k)\|^2$  and  $\mathbb{E}\| \nabla F_k(\mathbf{w}_t^k,\xi_t^k) - \nabla F_k(\mathbf{w}_t^k)\|^2$  approach  $\mathbb{E}\| \nabla F_k(\mathbf{w}^*,\xi_t^k)\|^2$  and

$\mathbb{E}\| \nabla F_k(\mathbf{w}^*,\xi_t^k) - \nabla F_k(\mathbf{w}^*)\|^2$ . Therefore, there is no substantial difference between these assumptions and assuming the bounds at  $\mathbf{w}^*$  only Koloskova et al. (2020). Furthermore, compared to assuming bounded gradient diversity as in related work Haddadpour & Mahdavi (2019); Li et al. (2020a), Assumption 4 is much less restrictive. When the optimality gap converges to zero, bounded gradient diversity restricts local objectives to have the same minimizer as the global objective, contradicting the heterogeneous data setting. For detailed discussions of our assumptions, please refer to Appendix Section B.

# 3 LINEAR SPEEDUP ANALYSIS OF FEDAVG

In this section, we provide convergence analyses of FedAvg for convex objectives in the general setting with both heterogeneous data (statistical heterogeneity) and partial participation (system heterogeneity). We show that for strongly convex and smooth objectives, the convergence of the optimality gap of averaged parameters across devices is  $\mathcal{O}(1 / KT)$ , while for convex and smooth objectives, the rate is  $\mathcal{O}(1 / \sqrt{KT})$ . Our results improve upon Li et al. (2020b); Karimireddy et al. (2019) by showing linear speedup for any number of participating devices, and upon Khaled et al. (2020); Koloskova et al. (2020) by allowing system heterogeneity. The proofs also highlight similarities and distinctions between the strongly convex and convex settings. Detailed proofs are deferred to Appendix Section E.

# 3.1 STRONGLY CONVEX AND SMOOTH OBJECTIVES

We first show that FedAvg has an  $\mathcal{O}(1 / KT)$  convergence rate for  $\mu$ -strongly convex and  $L$ -smooth objectives. The result relies on a technical improvement over the analysis in Li et al. (2020b). Moreover, it implies a distinction in communication efficiency that guarantees this linear speedup for FedAvg with full and partial device participation. With full participation,  $E$  can be chosen as large as  $\mathcal{O}(\sqrt{T / N})$  without degrading the linear speedup in the number of workers. On the other hand, with partial participation,  $E$  must be  $\mathcal{O}(1)$  to guarantee  $\mathcal{O}(1 / KT)$  convergence.

Theorem 1. Let  $\overline{\mathbf{w}}_T = \sum_{k=1}^{N} p_k \mathbf{w}_T^k$  in FedAvg,  $\nu_{\max} = \max_k N p_k$ , and set decaying learning rates  $\alpha_t = \frac{4}{\mu(\gamma + t)}$  with  $\gamma = \max\{32\kappa, E\}$  and  $\kappa = \frac{L}{\mu}$ . Then under Assumptions 1 to 4 with full device participation,

$$
\mathbb {E} F (\overline {{\mathbf {w}}} _ {T}) - F ^ {*} = \mathcal {O} \left(\frac {\kappa \nu_ {\mathrm {m a x}} ^ {2} \sigma^ {2} / \mu}{N T} + \frac {\kappa^ {2} E ^ {2} G ^ {2} / \mu}{T ^ {2}}\right),
$$

and with partial device participation with at most  $K$  sampled devices at each communication round,

$$
\mathbb {E} F (\overline {{\mathbf {w}}} _ {T}) - F ^ {*} = \mathcal {O} \left(\frac {\kappa E ^ {2} G ^ {2} / \mu}{K T} + \frac {\kappa \nu_ {\max} ^ {2} \sigma^ {2} / \mu}{N T} + \frac {\kappa^ {2} E ^ {2} G ^ {2} / \mu}{T ^ {2}}\right).
$$

Proof sketch. Because our unified analyses of results in the main text follow the same framework with variations in technical details, we first give an outline of proof for Theorem 1 to illustrate the main ideas. For full participation, the main ingredient is a recursive contraction bound

$$
\mathbb {E} \| \overline {{\mathbf {w}}} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} \leq (1 - \mu \alpha_ {t}) \mathbb {E} \| \overline {{\mathbf {w}}} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \alpha_ {t} ^ {2} \frac {1}{N} \nu_ {m a x} ^ {2} \sigma^ {2} + 6 \alpha_ {t} ^ {3} L E ^ {2} G ^ {2}
$$

where the  $\mathcal{O}(\alpha_t^3 E^2 G^2)$  term is the key improvement over the bound in Li et al. (2020b), which has  $\mathcal{O}(\alpha_t^2 E^2 G^2)$  instead. We then use induction to obtain a non-recursive bound on  $\mathbb{E}\| \overline{\mathbf{w}}_T - \mathbf{w}^*\|^2$ , which is converted to a bound on  $\mathbb{E}F(\overline{\mathbf{w}}_T) - F^*$  using  $L$ -smoothness. For partial participation, an additional sampling variance term  $\mathcal{O}\left(\frac{1}{K}\alpha_t^2 E^2 G^2\right)$  of leading order is added to the contraction bound. To facilitate the understanding of our analysis, please refer to a high-level summary in Appendix C.

Linear speedup. We compare our bound with that in Li et al. (2020b), which is  $\mathcal{O}\left(\frac{1}{NT} +\frac{E^2}{KT} +\frac{E^2G^2}{T}\right)$ . Because the term  $\frac{E^2G^2}{T}$  is also  $\mathcal{O}(1 / T)$  without a dependence on  $N$ , for any choice of  $E$  their bound cannot achieve linear speedup. The improvement of our bound comes from the term  $\frac{\kappa^2E^2G^2 / \mu}{T^2}$ , which now is  $\mathcal{O}(E^2 /T^2)$ . As a result, all leading terms scale with  $1 / N$  in the full device participation

setting, and with  $1 / K$  in the partial participation setting. This implies that in both settings, there is a linear speedup in the number of active workers during a communication round. We also emphasize that the reason one cannot recover the full participation bound by setting  $K = N$  in the partial participation bound is due to the variance generated by sampling.

Communication Complexity. Our bound implies a distinction in the choice of  $E$  between the full and partial participation settings. With full participation there is linear speedup  $\mathcal{O}(1 / NT)$  as long as  $E = \mathcal{O}(\sqrt{T / N})$  since then  $\mathcal{O}(E^2 / T^2) = \mathcal{O}(1 / NT)$  matches the leading term. This corresponds to a communication complexity of  $T / E = \mathcal{O}(\sqrt{NT})$ . In contrast, the bound in Li et al. (2020b) does not allow  $E$  to scale with  $\sqrt{T}$  to preserve  $\mathcal{O}(1 / T)$  rate, even for full participation. On the other hand, with partial participation,  $\frac{\kappa E^2 G^2 / \mu}{KT}$  is also a leading term, and so  $E$  must be  $\mathcal{O}(1)$ . In this case, our bound still yields a linear speedup in  $K$ , which is also confirmed by experiments. The requirement  $E = \mathcal{O}(1)$  in partial participation cannot be removed for our sampling schemes, as the sampling variance is  $\Omega(E^2 / T^2)$  and the dependence on  $E$  is tight (see Proposition 1 in Section E of the appendix).

Comparison with related works. To better understand the significance of the obtained bound, we compare our rates to the best-known results in related settings. Haddadpour & Mahdavi (2019) proves a linear speedup  $\mathcal{O}(1 / KT)$  result for strongly convex and smooth objectives<sup>1</sup>, with  $\mathcal{O}(K^{1/3}T^{2/3})$  communication complexity with non-i.i.d. data and partial participation. However, their results build on the bounded gradient diversity assumption, which implies the existence of  $\mathbf{w}^*$  that minimizes all local objectives (see discussions in Section 2.2 and Appendix B), effectively removing statistical heterogeneity. The bound in Koloskova et al. (2020) matches our bound in the full participation case, but their framework excludes partial participation (Koloskova et al., 2020, Proposition 1). The result of Karimireddy et al. (2019) applies to the full FL setting, but only has linear speedup when  $K = \mathcal{O}(N)$ , i.e. close to full participation, whereas our result has linear speedup for any number of participating devices.

# 3.2 CONVEX SMOOTH OBJECTIVES

Next we provide linear speedup analysis of FedAvg with convex and smooth objectives and show that the optimality gap is  $\mathcal{O}(1 / \sqrt{KT})$ . This result complements the strongly convex case in the previous part, as well as the non-convex smooth setting in Jiang & Agrawal (2018); Yu et al. (2019b); Haddadpour & Mahdavi (2019), where  $\mathcal{O}(1 / \sqrt{KT})$  results are given in terms of averaged gradient norm, and it also extends the result in Khaled et al. (2020), which has linear speedup in the convex setting, but only for full participation.

Theorem 2. Under Assumptions 1,3,4 and constant learning rate  $\alpha_{t} = \mathcal{O}\left(\sqrt{\frac{N}{T}}\right)$ , FedAvg satisfies

$$
\min  _ {t \leq T} F (\overline {{\mathbf {w}}} _ {t}) - F (\mathbf {w} ^ {*}) = \mathcal {O} \left(\frac {\nu_ {\max } ^ {2} \sigma^ {2}}{\sqrt {N T}} + \frac {N E ^ {2} L G ^ {2}}{T}\right)
$$

with full participation, and with partial device participation with  $K$  sampled devices at each communication round and learning rate  $\alpha_{t} = \mathcal{O}\left(\sqrt{\frac{K}{T}}\right)$ ,

$$
\min _ {t \leq T} F (\overline {{\mathbf {w}}} _ {t}) - F (\mathbf {w} ^ {*}) = \mathcal {O} \left(\frac {\nu_ {\max} ^ {2} \sigma^ {2}}{\sqrt {K T}} + \frac {E ^ {2} G ^ {2}}{\sqrt {K T}} + \frac {K E ^ {2} L G ^ {2}}{T}\right).
$$

The analysis again relies on a recursive bound, but without contraction:

$$
\| \overline {{\mathbf {w}}} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} + \alpha_ {t} \big (F (\overline {{\mathbf {w}}} _ {t}) - F (\mathbf {w} ^ {*}) \big) \leq \| \overline {{\mathbf {w}}} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \alpha_ {t} ^ {2} \frac {1}{N} \nu_ {\max } ^ {2} \sigma^ {2} + 6 \alpha_ {t} ^ {3} E ^ {2} L G ^ {2}
$$

which is then summed over time steps to give the desired bound, with  $\alpha_{t} = \mathcal{O}\left(\sqrt{\frac{N}{T}}\right)$ .

Choice of  $E$  and linear speedup. With full participation, as long as  $E = \mathcal{O}(T^{1/4} / N^{3/4})$ , the convergence rate is  $\mathcal{O}(1 / \sqrt{NT})$  with  $\mathcal{O}(N^{3/4} T^{3/4})$  communication rounds. In the partial participation setting,  $E$  must be  $O(1)$  in order to achieve linear speedup of  $\mathcal{O}(1 / \sqrt{KT})$ . Our result again demonstrates the difference in communication complexities between full and partial participation, and is to our knowledge the first result on linear speedup in the general federated learning setting with both heterogeneous data and partial participation for convex objectives.

# 4 LINEAR SPEEDDUP ANALYSIS OF NESTEROV ACCELERATED FEDAVG

A natural extension of the FedAvg algorithm is to use momentum-based local updates instead of local SGD updates. To our knowledge, the only convergence analyses of FedAvg with momentum-based stochastic updates focus on the non-convex smooth case Huo et al. (2020); Yu et al. (2019a); Li et al. (2020a). In this section, we complete the picture by providing the first  $\mathcal{O}(1 / KT)$  and  $\mathcal{O}(1 / \sqrt{KT})$  convergence results for Nesterov-accelerated FedAvg for convex objectives that match the rates from the previous section. Detailed proofs of convergence results in this section are deferred to Appendix Section F.

# 4.1 STRONGLY CONVEX AND SMOOTH OBJECTIVES

The Nesterov Accelerated FedAvg algorithm follows the updates:

$$
\mathbf {v} _ {t + 1} ^ {k} = \mathbf {w} _ {t} ^ {k} - \alpha_ {t} \mathbf {g} _ {t, k}, \quad \mathbf {w} _ {t + 1} ^ {k} = \left\{ \begin{array}{l l} \mathbf {v} _ {t + 1} ^ {k} + \beta_ {t} (\mathbf {v} _ {t + 1} ^ {k} - \mathbf {v} _ {t} ^ {k}) & \text {i f t + 1 \notin \mathcal {I} _ {E}}, \\ \sum_ {k \in \mathcal {S} _ {t + 1}} q _ {k} \left[ \mathbf {v} _ {t + 1} ^ {k} + \beta_ {t} (\mathbf {v} _ {t + 1} ^ {k} - \mathbf {v} _ {t} ^ {k}) \right] & \text {i f t + 1 \in \mathcal {I} _ {E}}, \end{array} \right.
$$

where  $\mathbf{g}_{t,k} \coloneqq \nabla F_k(\mathbf{w}_t^k, \xi_t^k)$  is the stochastic gradient sampled on the  $k$ -th device at time  $t$ , and  $q_k$  again depends on participation and sampling schemes.

Theorem 3. Let  $\overline{\mathbf{v}}_T = \sum_{k=1}^{N} p_k \mathbf{v}_T^k$  in Nesterov accelerated FedAvg, and set learning rates  $\alpha_t = \frac{6}{\mu} \frac{1}{t + \gamma}$ ,  $\beta_{t-1} = \frac{3}{14(t + \gamma)(1 - \frac{6}{t + \gamma}) \max\{\mu, 1\}}$ . Then under Assumptions 1,2,3,4 with full device participation,

$$
\mathbb {E} F (\overline {{\mathbf {v}}} _ {T}) - F ^ {*} = \mathcal {O} \left(\frac {\kappa \nu_ {\max} ^ {2} \sigma^ {2} / \mu}{N T} + \frac {\kappa^ {2} E ^ {2} G ^ {2} / \mu}{T ^ {2}}\right),
$$

and with partial device participation with  $K$  sampled devices at each communication round,

$$
\mathbb {E} F (\overline {{\mathbf {v}}} _ {T}) - F ^ {*} = \mathcal {O} \left(\frac {\kappa \nu_ {\max} ^ {2} \sigma^ {2} / \mu}{N T} + \frac {\kappa E ^ {2} G ^ {2} / \mu}{K T} + \frac {\kappa^ {2} E ^ {2} G ^ {2} / \mu}{T ^ {2}}\right).
$$

Similar to FedAvg, the key step in the proof of this result is a recursive contraction bound, but different in that it involves three time steps, due to the update format of Nesterov SGD (see Lemma 7 in Appendix F.1). Then we can again use induction and  $L$ -smoothness to obtain the desired bound. To our knowledge, this is the first convergence result for Nesterov accelerated FedAvg in the strongly convex and smooth setting. The same discussion about linear speedup of FedAvg applies to the Nesterov accelerated variant. In particular, to achieve  $\mathcal{O}(1 / NT)$  linear speedup,  $T$  iterations of the algorithm require only  $\mathcal{O}(\sqrt{NT})$  communication rounds with full participation.

# 4.2 CONVEX SMOOTH OBJECTIVES

We now show that the optimality gap of Nesterov Accelerated FedAvg has  $\mathcal{O}(1 / \sqrt{KT})$  rate for convex and smooth objectives. This result complements the strongly convex case in the previous part, as well as the non-convex smooth setting in Huo et al. (2020); Yu et al. (2019a); Li et al. (2020a), where a similar  $\mathcal{O}(1 / \sqrt{KT})$  rate is given in terms of averaged gradient norm.

Theorem 4. Set learning rates  $\alpha_{t} = \beta_{t} = \mathcal{O}\left(\sqrt{\frac{N}{T}}\right)$ . Then under Assumptions 1,3,4 Nesterov accelerated FedAvg with full device participation has rate

$$
\min  _ {t \leq T} F (\overline {{\mathbf {v}}} _ {t}) - F ^ {*} = \mathcal {O} \left(\frac {\nu_ {\max } ^ {2} \sigma^ {2}}{\sqrt {N T}} + \frac {N E ^ {2} L G ^ {2}}{T}\right),
$$

and with partial device participation with  $K$  sampled devices at each communication round,

$$
\min  _ {t \leq T} F (\overline {{\mathbf {v}}} _ {t}) - F ^ {*} = \mathcal {O} \left(\frac {\nu_ {\max } ^ {2} \sigma^ {2}}{\sqrt {K T}} + \frac {E ^ {2} G ^ {2}}{\sqrt {K T}} + \frac {K E ^ {2} L G ^ {2}}{T}\right).
$$

It is possible to extend the results in this section to accelerated FedAvg algorithms with other momentum-based updates. However, as know from stochastic optimization, Nesterov and other momentum updates may fail to accelerate over SGD (Liu & Belkin (2020); Kidambi et al. (2018); Liu et al. (2018); Yuan et al. (2016)). For this reason, we will instead turn to the overparameterized setting Ma et al. (2018); Liu & Belkin (2020); Canziani et al. (2016) in Section G of the appendix where we show that FedAvg enjoys geometric convergence and it is possible to improve its convergence rate with a new momentum-based FedAvg variant, which we term "FedMaSS".

![](images/4ffde142916946b34c79943f24867a321b72cc15336c67cec7e1deb1d5d8a718.jpg)

![](images/a213db9d7f454765cbf7d1bd220f81b6fe85a705946c1a67bb3f27a791553719.jpg)

![](images/cf4a4d23a47a90dbd6ef2fb61b81395dbcd24f13c08f6d6fa78a2332894d9a4e.jpg)

![](images/f389d37e24702b30d50566b40ce191d5d116724c8f63b4c28e0d9ef13479bb6d.jpg)

![](images/05e98014f9480c5b16a8f06a8fd43f0afd2147dab03b7ec6bd9f46d9ca4a8604.jpg)

![](images/d38a37320239aab4c54dcc68435a6c854d53c795068b066b54354a99e39ba109.jpg)

![](images/acaad87c43ca405cbb1a23d6463470638b603f7f3e706bdefe7a23fbd010ca8c.jpg)  
(a) Strongly convex objective

![](images/1639eac6f4692ec8caeb91cae282e02b051c0612686c9f99ebffcd8ca2b7195b.jpg)  
(b) Convex smooth objective

![](images/0b06351e30759f3d98774ede65c57c51da2fde2f942ebc604049cff212699c66.jpg)  
(c) Linear regression  
Figure 1: The linear speedup of FedAvg in full participation, partial participation, and the linear speedup of Nesterov accelerated FedAvg, respectively.

# 5 NUMERICAL EXPERIMENTS

In this section, we empirically examine the linear speedup convergence of FedAvg and Nesterov accelerated FedAvg in various settings, including strongly convex function, convex smooth function, and overparameterized objectives, as analyzed in previous sections.

Setup. Following the experimental setting in Stich (2019), we conduct experiments on both synthetic datasets and real-world dataset w8a Platt (1998) ( $d = 300$ ,  $n = 49749$ ). We consider the distributed objectives  $F(\mathbf{w}) = \sum_{k=1}^{N} p_k F_k(\mathbf{w})$ , and the objective function on the  $k$ -th local device includes three cases: 1) Strongly convex objective: the regularized binary logistic regression problem,  $F_k(\mathbf{w}) = \frac{1}{N_k} \sum_{i=1}^{N_k} \log(1 + \exp(-y_i^k \mathbf{w}^T \mathbf{x}_i^k) + \frac{\lambda}{2} \| \mathbf{w} \|^2$ . The regularization parameter is set to  $\lambda = 1/n \approx 2e - 5$ . 2) Convex smooth objective: the binary logistic regression problem without regularization. 3) Overparameterized setting: the linear regression problem without adding noise to the label,  $F_k(\mathbf{w}) = \frac{1}{N_k} \sum_{i=1}^{N_k} (\mathbf{w}^T \mathbf{x}_i^k + b - y_i^k)^2$ .

Linear speedup of FedAvg and Nesterov accelerated FedAvg. To verify the linear speedup convergence as shown in Theorems 1 2 3 4, we evaluate the number of iterations needed to reach  $\epsilon$ -accuracy in three objectives. We initialize all runs with  $\mathbf{w}_0 = \mathbf{0}_d$  and measure the number of iterations to reach the target accuracy  $\epsilon$ . For each configuration  $(E, K)$ , we extensively search the learning rate from  $\min(\eta_0, \frac{nc}{1+t})$ , where  $\eta_0 \in \{0.1, 0.12, 1, 32\}$  according to different problems and  $c$  can take the values  $c = 2^i \forall i \in \mathbb{Z}$ . As the results shown in Figure 1, the number of iterations decreases as the number of (active) workers increasing, which is consistent for FedAvg and Nesterov accelerated FedAvg across all scenarios. For additional experiments on the impact of  $E$ , detailed experimental setup, and hyperparameter setting, please refer to the Appendix Section I.

# REFERENCES

Zeyuan Allen-Zhu, Yanzhi Li, and Zhao Song. A convergence theory for deep learning via overparameterization. arXiv preprint arXiv:1811.03962, 2018.  
Alfredo Canziani, Adam Paszke, and Eugenio Culurciello. An analysis of deep neural network models for practical applications. arXiv preprint arXiv:1605.07678, 2016.  
Fei Chen, Zhenhua Dong, Zhenguo Li, and Xiuqiang He. Federated meta-learning for recommendation. arXiv preprint arXiv:1802.07876, 2018.  
Farzin Haddadpour and Mehrdad Mahdavi. On the convergence of local descent methods in federated learning. arXiv preprint arXiv:1910.14425, 2019.  
Andrew Hard, Chloe M Kiddon, Daniel Ramage, Francoise Beaufays, Hubert Eichner, Kanishka Rao, Rajiv Mathews, and Sean Augenstein. Federated learning for mobile keyboard prediction, 2018. URL https://arxiv.org/abs/1811.03604.  
Zhouyuan Huo, Qian Yang, Bin Gu, Lawrence Carin Huang, et al. Faster on-device training using new federated momentum algorithm. arXiv preprint arXiv:2002.02090, 2020.  
Prateek Jain, Sham M Kakade, Rahul Kidambi, Praneeth Netrapalli, and Aaron Sidford. Accelerating stochastic gradient descent. In Proc. STAT, volume 1050, pp. 26, 2017.  
Peng Jiang and Gagan Agrawal. A linear speedup analysis of distributed deep learning with sparse and quantized communication. In Advances in Neural Information Processing Systems, pp. 2525-2536, 2018.  
Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Keith Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. arXiv preprint arXiv:1912.04977, 2019.  
Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J Reddi, Sebastian U Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for on-device federated learning. arXiv preprint arXiv:1910.06378, 2019.  
A Khaled, K Mishchenko, and P Richtárik. Tighter theory for local sgd on identical and heterogeneous data. In The 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020), 2020.  
Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. First analysis of local gd on heterogeneous data. *NeurIPS Workshop on Federated Learning for Data Privacy and Confidentiality*, 2019.  
Rahul Kidambi, Praneeth Netrapalli, Prateek Jain, and Sham Kakade. On the insufficiency of existing momentum schemes for stochastic optimization. In 2018 Information Theory and Applications Workshop (ITA), pp. 1-9. IEEE, 2018.  
Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian U Stich. A unified theory of decentralized sgd with changing topology and local updates. arXiv preprint arXiv:2003.10422, 2020.  
Monica S Lam, Giovanni Campagna, Silei Xu, Michael Fischer, and Mehrad Moradshahi. Protecting privacy and open competition with almond: an open-source virtual assistant. XRDS: Crossroads, The ACM Magazine for Students, 26(1):40-44, 2019.  
Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks.  $MLSys$ , 2020a.  
Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of fedavg on non-iid data. ICLR, 2020b.  
Xianfeng Liang, Shuheng Shen, Jingchang Liu, Zhen Pan, Enhong Chen, and Yifei Cheng. Variance reduced local sgd with lower communication complexity. arXiv preprint arXiv:1912.12844, 2019.

Chaoyue Liu and Mikhail Belkin. Accelerating sgd with momentum for over-parameterized learning. *ICLR*, 2020.  
Tianyi Liu, Zhehui Chen, Enlu Zhou, and Tuo Zhao. Toward deeper understanding of nonconvex stochastic optimization with momentum using diffusion approximations. arXiv preprint arXiv:1802.05155, 2018.  
Wei Liu, Li Chen, Yunfei Chen, and Wenyi Zhang. Accelerating federated learning via momentum gradient descent. IEEE Transactions on Parallel and Distributed Systems, 2020.  
Siyuan Ma, Raef Bassily, and Mikhail Belkin. The power of interpolation: Understanding the effectiveness of sgd in modern over-parametrized learning. ICML, 2018.  
H Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, et al. Communication-efficient learning of deep networks from decentralized data. Proceedings of the 20 th International Conference on Artificial Intelligence and Statistics (AISTATS), 2017.  
Eric Moulines and Francis R Bach. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. In Advances in Neural Information Processing Systems, pp. 451-459, 2011.  
Deanna Needell, Rachel Ward, and Nati Srebro. Stochastic gradient descent, weighted sampling, and the randomized kaczmarz algorithm. In Advances in neural information processing systems, pp. 1017-1025, 2014.  
J Platt. Fast training of support vector machines using sequential minimal optimization, in, b. scholkopf, c. burges, a. smola,(eds.): Advances in kernel methods-support vector learning, 1998.  
R Tyrrell Rockafellar. Convex analysis. Number 28. Princeton university press, 1970.  
Mark Schmidt and Nicolas Le Roux. Fast convergence of stochastic gradient descent under a strong growth condition. arXiv preprint arXiv:1308.6370, 2013.  
Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, and Ameet S Talwalkar. Federated multi-task learning. In Advances in Neural Information Processing Systems, pp. 4424-4434, 2017.  
Sebastian U Stich. Local sgd converges fast and communicates little. *ICLR*, 2019.  
Thomas Strohmer and Roman Vershynin. A randomized kaczmarz algorithm with exponential convergence. Journal of Fourier Analysis and Applications, 15(2):262, 2009.  
Jianyu Wang and Gauri Joshi. Cooperative sgd: A unified framework for the design and analysis of communication-efficient sgd algorithms. arXiv preprint arXiv:1808.07576, 2018.  
Shiqiang Wang, Tiffany Tuor, Theodoros Salonidis, Kin K Leung, Christian Makaya, Ting He, and Kevin Chan. Adaptive federated learning in resource constrained edge computing systems. IEEE Journal on Selected Areas in Communications, 37(6):1205-1221, 2019.  
Blake E Woodworth, Jialei Wang, Adam Smith, Brendan McMahan, and Nati Srebro. Graph oracle models, lower bounds, and gaps for parallel stochastic optimization. In Advances in neural information processing systems, pp. 8496-8506, 2018.  
Hao Yu, Rong Jin, and Sen Yang. On the linear speedup analysis of communication efficient momentum sgd for distributed non-convex optimization. ICML, 2019a.  
Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel restarted sgd with faster convergence and less communication: Demystifying why model averaging works for deep learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 5693-5700, 2019b.  
Kun Yuan, Bicheng Ying, and Ali H Sayed. On the influence of momentum acceleration on online learning. The Journal of Machine Learning Research, 17(1):6602-6667, 2016.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530, 2016.  
Fan Zhou and Guojing Cong. On the convergence properties of a  $k$ -step averaging stochastic gradient descent algorithm for nonconvex optimization. *IJCAI*, 2018.
