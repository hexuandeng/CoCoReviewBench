# A GENERAL DIFFERENTIALLY PRIVATE LEARNING FRAMEWORK FOR DECENTRALIZED DATA

Anonymous authors

Paper under double-blind review

# ABSTRACT

Decentralized consensus learning has been hugely successful, which minimizes a finite sum of expected objective functions over a network of nodes. However, the local communication across neighbouring nodes in the network may lead to the leakage of private information. To address this challenge, we propose a general differentially private (DP) learning framework for decentralized data that applies to many non-smooth learning problems. We show that the proposed algorithm retains the performance guarantee in terms of stability, generalization, and finite sample performance. We investigate the impact of local privacy-preserving computation on the global DP guarantee. Further, we extend the discussion by adopting a new class of noise-adding DP mechanisms based on generalized Gaussian distributions to improve the utility-privacy trade-offs. Our numerical results demonstrate the effectiveness of our algorithm and its better performance over the state-of-the-art baseline methods in various decentralized settings.

# 1 INTRODUCTION

Decentralized consensus computation is a form of distributed learning to achieve a consensus target that relies on communication between nodes through a network diagram without a centralized server. On large-scale optimization tasks involving distributed training samples, decentralized algorithms consider a low-communication-overhead alternative network structure to learn a global optimal solution in a parallel and distributed fashion, while, allowing multiple agents to keep their own datasets unexposed. Existing approaches to decentralized learning problems mainly consist of subgradient-based algorithms (Nedic et al., 2008), Alternating Direction Method of Multipliers (ADMM) based algorithms (Boyd et al., 2011; Shi et al., 2014), and the composite of subgradient descent and ADMM (Bianchi et al., 2014), which have shown superior performance on many application domains, such as distributed sensing in wireless sensor networks, healthcare, and multiagent robotic systems (Rabbat & Nowak, 2004; Bullo et al., 2009; Sayed et al., 2014; Yu et al., 2021a). However, the lack of adaptation of these existing algorithms has limited their application: most of the existing works require the objective functions to be smooth or strongly convex. To fill the gap, we provide a general framework of decentralized algorithms, stochastic Decentralized Krasnosel'skii-Mann (D-KM) iteration, that comprises diverse decentralized algorithms with straightforward, high-performance implementation. Depending on various iterative operations, D-KM offers intuitive interfaces to implement a spectrum of decentralized algorithms, which is adaptive to objectives with different functional properties (e.g., non-smoothness).

Although the decentralized algorithm obtains the consensus solution through local computation, namely, each agent solves its own problems independently and locally with only intermediate parameters that need to be shared. Past experience has demonstrated the possibility of privacy leakage in such an iterative process: the attacker can recover the sensitive information from shared parameters as pointed in Shokri et al. (2017), Fredrikson et al. (2015). One defence procedure is to adopt a private variant of the learning algorithm using Differential Privacy (DP) to secure the iterative process. While the developed DP-based methods mainly focus on standalone systems or centralized learning (Song et al., 2013; Li et al., 2022), how to ensure data privacy in decentralized systems is still an open question due to their complex communication typologies and distinct learning characteristics. Moreover, most of the existing variants of learning algorithms consider simply adding Gaussian noise to iterates to ensure differential privacy where the noise could be extremely large (Farokhi, 2022). However, enforcing and optimizing DP protection with unbounded noise ad

dition could severely affect the learning efficiency and degrade the performance of the trained model under differential privacy guarantee.

To mitigate privacy risks in decentralized settings, in this paper, we propose a general decentralized privacy-preserving algorithm, DP-KM. Specifically, the proposed algorithm enforces differential privacy protection of communication, and it can be applicable to much of the existing work on optimization and consensus computation with mild assumptions on objectives. We validate the privacy and effectiveness of the proposed approach through a rigorous theoretical analysis, which shows the performance guarantees of the proposed method. To enable the D-KM to be noise-resilient, we further propose a class of truncated generalized Gaussian noise-adding mechanisms. The key feature of the DP-KM with the mechanisms is to leverage the D-KM iteration with truncated generalized Gaussian noise in the iteration procedure to reduce the noise scale and achieve higher utility under differential privacy guarantee for general objective functions. Empirically, we conduct comprehensive experiments to demonstrate that our approach outperforms prior works in various decentralized settings. The main contributions of this paper are summarized as follows.

1. Leveraging the decentralized learning procedure to train models under privacy constraints, we propose a general differentially private decentralized algorithm, DP-KM.  
2. We provide a rigorous sensitivity analysis of D-KM and establish the generalization and optimization error bound of DP-KM under mild conditions. Additionally, we investigate the impact of the local privacy-preserving computation on global differential privacy.  
3. We introduce a new class of truncated generalized Gaussian mechanism that can achieve higher utility under differential privacy guarantee for general objective functions.  
4. We conduct numerical analysis to validate the effectiveness of DP-KM with truncated Laplace and Gaussian noise in decentralized learning settings comparing with the state-of-the-art methods.

The rest of this paper is organized as follows. Section 2 introduces problem setting, formal definitions of decentralized systems, and differential privacy. The stability of D-KM is presented in Section 3. Section 4 establishes the generalization bound of the DP-KM in a decentralized setting, and the effect of local privacy-preserving computation on global differential privacy. Section 5 illustrates a new class of noise-adding mechanisms, especially, two truncated mechanisms: Truncated Laplacian Mechanism and Truncated Gaussian Mechanism, for synchronous training models. Section 6 presents our privacy analysis and the performance of our approach in various settings. We conclude this paper in Section 7.

# 2 PROBLEM STATEMENT

In this section, we first start with the problem setting. We then present preliminaries about decentralized learning schemes as well as differential privacy.

Problem setting: Consider a network with  $M$  agents, each of which holds a dataset  $\Xi_{m} = \{\xi_{i(m)}\}_{i=1}^{N_{i}}$ , for  $m = 1, \dots, M$ , where  $N_{i}$  is the number of training samples in the dataset  $\Xi_{m}$ ,  $\xi_{i(m)} \in \mathbb{R}^{p}$  is the  $i$ -th sample stored in the  $m$ -th agent. For ease of presentation, we assume that data are evenly collected and each agent has a sample size of  $N$ .

In this paper, we consider solving a stochastic decentralized optimization approximated by its corresponding empirical risk minimization problem,

$$
\widehat {L} (x) = \min _ {x \in \mathbb {R} ^ {p}} \frac {1}{M N} \sum_ {m = 1} ^ {M} \sum_ {i = 1} ^ {N} \ell \left(x, \xi_ {i (m)}\right), \quad \widehat {x} = \arg \min _ {x \in \mathbb {R} ^ {p}} \frac {1}{M N} \sum_ {m = 1} ^ {M} \sum_ {i = 1} ^ {N} \ell \left(x, \xi_ {i (m)}\right),
$$

where  $x \in \mathbb{R}^p$  is the target parameter, and  $\ell(\cdot)$  is the loss function used to measure the quality of the trained model. Throughout, we assume the loss function is convex, closed, and proper but not necessarily differentiable. The goal of our problem is to learn a consensus parameter  $\bar{x} = \frac{1}{M} \sum_{m=1}^{M} x(m)$  on  $M$  agents across a network diagram, where  $x(m)$  is the solution of the local parameter on the  $m$ -th agent. We stress that, in practice, each agent operates independently and the average is only taken in the last iteration.

The decentralized optimization is associated with a given network topology that can be formulated mathematically by a mixing matrix (Alghunaim et al., 2019; Ying et al., 2021). The properties of the graph required to learn a global parameter can be summarized in Definition 1. The agent's communication with their neighbours is responsible for message passing and aggregation accordingly. To formalize the definition, we define the connected network by,  $\mathcal{G} = (\mathcal{V},\mathcal{E})$  with vertex set  $\mathcal{V} = \{1,\dots ,M\}$  and edge set  $\mathcal{E}\subseteq \mathcal{V}\times \mathcal{V}$ . Edge  $(m,l)\in \mathcal{E}$  represents the interconnection between agent  $m$  and its neighbors  $l\in \mathcal{N}(m)$ .

Definition 1 (Mixing Matrix) For any given graph  $\mathcal{G} = (\mathcal{V},\mathcal{E})$ , the mixing matrix  $\mathbf{W} = [w_{m,l}] \in \mathbb{R}^{M \times M}$  is defined on the edge set  $\mathcal{V}$  that satisfies: (1) if  $m \neq l$  and  $(m,l) \notin \mathcal{E}$ , then  $w_{m,l} = 0$ ; otherwise,  $w_{m,l} > 0$ ; (2)  $\mathbf{W} = \mathbf{W}^{\top}$ ; (3)  $\mathrm{null}\{\mathbb{I} - \mathbf{W}\} = \mathrm{span}\{\mathbf{1}\}$ ; (4)  $\mathbb{I} \succeq \mathbf{W} \succ -\mathbb{I}$ .

We remark that the mixing matrix  $\mathbf{W}$  is a doubly stochastic gossip matrix that characterizes the underlying topology and the matrix is non-unique for a given graph (Ying et al., 2021; Sun et al., 2021). An essential constant that characterizes the connectivity of gossip communications is measured by the spectral gap as  $1 - \lambda$  (Zhu et al., 2022), where  $\lambda \coloneqq \max \left\{\left|\lambda_2\right|, \left|\lambda_M\right|\right\}$ , and  $\lambda_i$  denotes the  $i$ th largest eigenvalue of  $\mathbf{W} \in \mathbb{R}^{M \times M}$ . The definition of the mixing matrix implies that  $0 \leq \lambda < 1$ . We now give a general decentralized learning algorithm on the  $m$ -th agent.

Definition 2 (Stochastic Decentralized Krasnosel'skii-Mann (D-KM)) Suppose the training sample set  $\Xi := \bigcup_{m=1}^{M} \Xi_m$  is distributed-stored in  $M$  agents with total sample size  $NM$ , where  $\Xi_m$  is a training dataset located in the  $m$ -th agent for  $m = 1, \dots, M$ . We assume that  $\xi_{i(m)} \sim \mathbb{P}$  with  $\xi_{i(m)} \in \Xi_m$  for any  $m, i$ . For each agent, given a nonexpansive operator  $T$ , the iterative formula of the stochastic  $D$ -KM algorithm,  $\mathcal{A}$ , is defined as,

$$
x ^ {k + 1} (m) = \mathcal {A} \left(x ^ {k} (m); \Xi\right) = \sum_ {l \in \mathcal {N} (m)} w _ {m l} x ^ {k} (l) + \alpha_ {k} \left(T \left(x ^ {k} (m); \xi_ {i _ {k} (m)}\right) - x ^ {k} (m)\right), \tag {1}
$$

where  $w_{ml}$  is the element of a given matrix  $\mathbf{W}$  satisfying Definition 1,  $\alpha_{k} \in (0,1]$ .  $i_{k}$  is i.i.d. variable drawn from the uniform distribution over  $\{1,\dots ,N\}$  at the  $k$ -th iteration.

The form of  $T$  in Definition 2 depends on the specific algorithm we adopt. Let  $I$  be an identity operator,  $\partial f, \nabla f$  be the sub-gradient and gradient of function  $f$ , respectively. Denote  $\mathrm{prox}_f$ ,  $\mathrm{relf}_f$  as the proximal and reflection operator of  $f$  (Davis & Yin, 2016). The forms of  $T$  in Definition 2 for Stochastic Gradient Descent (SGD), Stochastic Proximal Gradient Descent (SPGD) and ADMM algorithms are summarized in Table 1. Further, let  $\mathbf{X} = [x(1),\dots ,x(M)]^\top \in \mathbb{R}^{M\times p}$  that stores all local parameters across the network,  $\mathbf{T}(\mathbf{X};\Xi) = [T(x(1);\Xi_1),\dots ,T(x(M);\Xi_M)]^\top \in \mathbb{R}^{M\times p}$  stacking all local updating w.r.t. the first argument. In this way, the matrix form of Equation equation 1 is

$$
\mathbf {X} ^ {k + 1} = \mathbf {W X} ^ {k} + \alpha_ {k} \left(\mathbf {T} \left(\mathbf {X} ^ {k}; \boldsymbol {\Xi}\right) - \mathbf {X} ^ {k}\right).
$$

Table 1: Overview of several first-order algorithms  

<table><tr><td>Algorithm</td><td>Operator identity (T)</td><td>Subgradient identity</td></tr><tr><td>SGD</td><td>I - γ∇f</td><td>xk+1 = Wxk - γk∇f(xk)</td></tr><tr><td>SPGD</td><td>(I + γ∂g)^{-1}(I - γ∇f)</td><td>xk+1 = Wxk + (proxγkg(I - γk∇f(xk)) - xk)</td></tr><tr><td>ADMM</td><td>(I + γ∂f)^{-1}[(I + γ∂g)^{-1}(I - γ∂f) + γ∂f]</td><td>xk+1 = Wxk + 1/2 reflγk∂f ∘ reflγk∂g(xk)</td></tr></table>

Privacy Concern: In the stochastic D-KM algorithm, despite each agent communicating with its neighbours by sending parameters instead of directly exchanging raw data, the risk of leaking information still exists: the attacker can recover the sensitive information of data from shared parameters as pointed in Shokri et al. (2017), Fredrikson et al. (2015). This motivates us to consider privacy preserving iteration procedure with efficient communication while it retains a performance guarantee. Differential privacy, introduced by Dwork et al. (2006), is a widely adopted definition due to its important advantages over other privacy techniques. It quantifies to what extent individual privacy in a data set is preserved while releasing aggregated information.

Definition 3  $((\varepsilon, \delta)$ -Differential Privacy Dwork et al. (2006)) A stochastic algorithm  $\mathcal{A}$  is called  $(\varepsilon, \delta)$ -differential privacy if for any subset  $\mathbb{R}_0 \subset \mathbb{R}^p$  and any neighbouring sample set pair  $\Xi$  and  $\Xi'$  which differs by only one sample, we have,

$$
\log \left[ \frac {\mathbb {P} _ {\mathcal {A} (\Xi)} \left(\mathcal {A} (\Xi) \in \mathbb {R} _ {0}\right) - \delta}{\mathbb {P} _ {\mathcal {A} (\Xi^ {\prime})} \left(\mathcal {A} (\Xi^ {\prime}) \in \mathbb {R} _ {0}\right)} \right] \leq \varepsilon .
$$

Differential privacy provides privacy guarantees by requiring the capability of indistinguishable in identifying whether an individual is in the data set or not based on the released information. The parameters  $\varepsilon$  and  $\delta$  are privacy budgets indicating the strength of privacy protection from the algorithm. The classic differential privacy is called  $\varepsilon$ -differential privacy with  $\delta = 0$ , which imposes an upper bound  $e^{\varepsilon}$  on the multiplicative distance of probability distributions of randomized query outputs for any two neighboring data sets (Dong et al., 2019). The common interpretation of  $(\varepsilon, \delta)$ -differential privacy is that it is  $\varepsilon$ -differential privacy except with probability  $\delta$  (Mironov, 2017).

# 3 SENSITIVITY OF THE STOCHASTIC D-KM ALGORITHM

In this section, we estimate the  $l_{2}$  norm sensitivity of the stochastic D-KM, laying the foundation for noise addition in the generalized Gaussian mechanisms in Section 5. Note that the derivation of sensitivity of our proposed D-KM algorithm does not require the assumption of smoothness and strong convexity of objective functions. Before formalizing the result, we introduce the definition of sensitivity of algorithms in a decentralized learning setting, and assumptions.

Definition 4 (Sensitivity) For a specific algorithm  $\mathcal{A}$  acting on training samples,  $\Xi^{\prime},\Xi^{\prime \prime}$  which are two adjacent datasets that differ by one data point. Until iteration  $K$  define the  $\Delta_K$ -sensitivity of algorithm  $\mathcal{A}$ ,

$$
\Delta_ {K} := \sup  _ {\Xi^ {\prime}, \Xi^ {\prime \prime}} \| \mathcal {A} \left(\Xi^ {\prime}\right) - \mathcal {A} \left(\Xi^ {\prime \prime}\right) \|.
$$

Assumption 1 The loss function is convex, closed, proper and sub-differentiable with respect to  $x$ , and the fixed-point iteration is bounded by the finite constant  $B$ , i.e.,  $\max_{x,\xi}\| T(x;\xi) - x\| \leq B$ .

$\| T(x;\xi) - x\|$  is defined as a fixed point residual in the literature which typically relates to the gradient of the objective (Davis & Yin, 2016). And it has shown that Assumption 1 is a weaker and common assumption in the optimization literature (Sun et al., 2021). We are now establishing the  $\Delta_K$ -sensitivity of the D-KM algorithm. That is, we, through Theorem 1, provide the boundedness on  $\Delta_K$  due to the only one different point for any two adjacent datasets.

Theorem 1 ( $\Delta_K$ -Sensitivity) Let  $x^K = \frac{\sum_{m=1}^M x^K(m)}{M}$ ,  $y^K = \frac{\sum_{m=1}^M y^K(m)}{M}$ . Denote  $x^K$  and  $y^K$  as the corresponding outputs of the D-KM algorithm applied to two sets  $\Xi'$ ,  $\Xi''$  of size  $NM$  which differ at only one point. Assume the initial value  $\mathbf{X} = \mathbf{0}$ . With Assumption 1 satisfied, given relaxed parameter,  $\{\alpha_k\}_{k=0}^K \in (0,1]$ , the  $\Delta_K$ -sensitivity of the D-KM algorithm has the upper bound,

$$
\mathbb {E} \Delta_ {K} \leq \frac {2 B \sum_ {k = 0} ^ {K - 1} \alpha_ {k}}{N M} + 4 B \sum_ {k = 0} ^ {K - 1} (1 + 2 \alpha_ {k}) \sum_ {j = 0} ^ {k - 1} \alpha_ {j} \lambda^ {k - 1 - j}.
$$

Theorem 1 quantifies the accumulated deviation bound between two trajectories of iterates based on two datasets which differ at only one point. We note that Sun et al. (2021) shows similar results for the stochastic gradient descent under a convex setting. With a fixed iteration number  $K$ , as the data size,  $M, N$  increases and  $\lambda$  decreases,  $\Delta_K$  gets smaller for both diminishing and constant learning rates. However, it fails to have the sensitivity under control when  $K$  increases, which also suggests the risk of privacy that, with the higher iterative step, it will be easier to identify the specific sample. Moreover, different topologies affect the bound of  $\Delta_K$  as the sensitivity decreases with decreasing  $\lambda$ , see Table 2 for details.

Note that compared to the centralized optimization setting (Hardt et al., 2016), there is an additional term of  $4B\sum_{k=0}^{K-1}(1 + 2\alpha_k)\sum_{j=0}^{k-1}\alpha_j\lambda^{k-1-j}$  in Theorem 1, which quantifies the impact of the network topology on the learning procedure. We emphasize that, although decentralization has certain advantages compared with centralization, especially in the reduction of the expense of communication on the central server, it sacrifices data privacy and learning stability.

Table 2:  ${\Delta }_{K}$  -Sensitivity under Different Graph Topology  

<table><tr><td>Graph topology</td><td>Spectral gap (1-λ)</td><td>Sensitivity (αk=α)</td><td>Sensitivity (αk=(1/(k+1)))</td></tr><tr><td>Ring</td><td>O(1/M2)</td><td>O(αK/MN + M2αK)</td><td>O(ln(K)/MN + M2ln(K))</td></tr><tr><td>Grid</td><td>O(1/M log(M))</td><td>O(αK/MN + M log(M)αK)</td><td>O(ln(K)/MN + M log(M) ln(K))</td></tr><tr><td>Star</td><td>O(1/M)</td><td>O(αK/MN + MαK)</td><td>O(ln(K)/MN + M ln(K))</td></tr><tr><td>Exponential</td><td>O(1/log(M))</td><td>O(αK/MN + log(M)αK)</td><td>O(ln(K)/MN + log(M) ln(K))</td></tr><tr><td>Full connected</td><td>1</td><td>O(αK/MN + αK)</td><td>O(ln(K)/MN + ln(K))</td></tr></table>

# 4 PERFORMANCE OF DECENTRALIZED LEARNING ALGORITHMS WITH DIFFERENTIAL PRIVACY

Existing DP schemes in decentralized learning setting typically rely on the perturbation of objective functions, gradients, but are limited to iterates (Huang et al., 2019; Yu et al., 2021b; Asi et al., 2021). Such methods usually introduce extra noise that has privacy preservation but it is hard to examine how their algorithm and analysis can be adapted to the privacy multi-agent setting (McGregor et al., 2010), especially, the privacy and performance trade-off in the generalization of DP algorithms (He et al., 2021). In this paper, we consider iterate independent noise addition mechanisms (Definition 5) to preserve DP: a random noise is added to the iterate to reduce leakage information. Before introducing noise addition mechanisms used in the paper that satisfy differential privacy, in this section, we do a rigorous analysis by establishing a generalization error bound, as well as a finite sample guarantee of decentralized learning algorithms when these algorithms satisfy differential privacy. These results illustrate the effectiveness of using D-KM with the proposed noise addition mechanism in applications. We next proceed by quantifying the bound introduced by noise addition, and computing the end-to-end differential privacy guarantee across  $M$  agents over a network system.

Definition 5 (Noise-adding Mechanisms Geng et al. (2018)) Given a data set  $\Xi$ , a query-output independent noise-adding mechanism  $\tilde{\mathcal{A}}$  will release the query output  $\tilde{x}^k = \tilde{\mathcal{A}}(x^k; \Xi)$  corrupted by an additive random noise  $d$ ,

$$
\tilde {x} ^ {k} = x ^ {k} + d.
$$

Let  $L(x) = \mathbb{E}_{\xi \sim \mathbb{P}}[\ell (x,\xi)]$  and  $x^{\star}$  be its optimal solution. Note that, for a specific stochastic algorithm  $\mathcal{B}$  on  $\Xi$  with sample size  $NM$  with output  $\mathcal{B}(\Xi)$ , the excess generalization error of  $\mathcal{B}$  defined as,  $\mathbb{E}_{\Xi ,\mathcal{B}}\left[L(\mathcal{B}(\Xi)) - L(x^{\star})\right]$ , can be decomposed into three terms

$$
\underbrace {\mathbb {E} _ {\Xi , \mathcal {B}} \left[ L (\mathcal {B} (\Xi)) - \widehat {L} (\mathcal {B} (\Xi)) \right]} _ {\text {g e n e r a l i z a t i o n e r r o r}} + \underbrace {\mathbb {E} _ {\Xi , \mathcal {B}} \left[ \widehat {L} (\mathcal {B} (\Xi)) - \widehat {L} (\widehat {x}) \right]} _ {\text {o p t i m i z a t i o n e r r o r}} + \underbrace {\mathbb {E} _ {\Xi , \mathcal {B}} \left[ \widehat {L} (\widehat {x}) - \widehat {L} \left(x ^ {\star}\right) \right]} _ {\text {t e s t e r r o r}}. \tag {2}
$$

Here, the expectation is taken over the algorithm and the data. We establish the boundedness of generalization error in Theorem 2 that reflects joint effects caused by the data  $\Xi$  and the algorithm  $\mathcal{B}$ .

Assumption 2 The loss function  $\ell(x, \xi)$  is nonnegative and Lipschitz continuous with respect to  $x$ , i.e.,  $\| \ell(x, \xi) - \ell(y, \xi) \| \leq Lip \| x - y \|$  for all  $x, y \in \mathbb{R}^p$ ,  $\xi \sim \mathbb{P}$ .

Theorem 2 (Generalization Bound) Assume that the decentralized learning algorithm  $\mathcal{B}:\Xi \mapsto \mathbb{R}^p\times \{1,\dots ,M\}$  is  $(\varepsilon ,\delta)$ -differentially private, and the loss function  $\| \ell \|_{\infty}\leq R$  on the domain. Under Assumption 2, we have that,

$$
\left| _ {\Xi \sim \mathbb {P} ^ {M N}, \mathcal {B} (\Xi)} \mathbb {E} \left[ L (\mathcal {B} (\Xi)) - \widehat {L} (\mathcal {B} (\Xi)) \right] \right| \leq (1 - e ^ {- \varepsilon}) R + e ^ {- \varepsilon} M \delta .
$$

Theorem 3 (Finite Sample Guarantee) Under the Assumption of Theorem 2, for any  $\epsilon >0$ , we have,

$$
\mathbb {P} (L (\mathcal {B} (\Xi)) \leq \widehat {L} (\mathcal {B} (\Xi)) + \epsilon) \geq \frac {\epsilon - (1 - e ^ {- \varepsilon}) R - e ^ {- \varepsilon} M \delta}{\epsilon + R}.
$$

These two theorems represent the gap between the empirical loss based on finite samples and its expectation. It demonstrates the impact of differential privacy on in-sample and out-of-sample performance. Although ensuring data privacy sacrifices the generalization, these results show that a good privacy-preserving mechanism still retains a certain level of generalization as well as a finite sample guarantee. We further examine the optimization error bound in formula (2) that caused by adding noise to the query output.

Let  $\widetilde{\mathbf{X}} = [\tilde{x}(1),\dots,\tilde{x}(M)]^{\top}$  be the released iterates corrupted by an additive random noise for each agent,  $\mathbf{X}^{\star} = [x^{\star},\dots,x^{\star}]^{\top} \in \mathbb{R}^{M \times M}$ . With Assumption 2, the error bound can be controlled by the difference between the iterates and its true parameter,

$$
\| \mathbf {W} \widetilde {\mathbf {X}} ^ {k} + \alpha_ {k} \left(\mathbf {T} \left(\mathbf {X} ^ {k}; \boldsymbol {\Xi}\right) - \mathbf {X} ^ {k}\right) - \mathbf {X} ^ {\star} \| \leq \| \mathbf {X} ^ {k + 1} - \mathbf {X} ^ {\star} \| + \| \mathbf {W} \left(\mathbf {X} ^ {k} - \widetilde {\mathbf {X}} ^ {k}\right) \|,
$$

where the first term is the same as in the non-privacy setting, which depends on the convergence properties of a given algorithm (Wotao, 2019). The second indicates the deviation by using the privacy mechanisms.

Theorem 4 (Boundedness of local iterates based on Gaussian additive noise) Given a data set  $\Xi$ , assume an iterative independently noise-adding mechanism  $\tilde{\mathcal{A}}$  releases the output  $\tilde{\mathcal{A}}(x^k; \Xi) := x^k + d$  corrupted by an additive random noise  $d$ , where  $d$  follows a Gaussian distribution with mean  $\mu$  and variance  $\sigma^2$ . The error bound caused by the additive noise is

$$
\mathbb {E} \left[ \left\| \mathbf {W} \left(\mathbf {X} ^ {k} - \widetilde {\mathbf {X}} ^ {k}\right) \right\| _ {F} ^ {2} \leq p \left(\sigma^ {2} + \mu^ {2}\right) [ (M - 1) \lambda^ {2} + 1 ], \right.
$$

where  $\| \cdot \| _F$  is the Frobenius norm of.

Generalization of local computation In real applications, each agent acts independently, where there is less likely to reach an agreement on a consistent  $(\varepsilon, \delta)$ -DP across all agents (Bellet et al., 2018). Therefore, we proceed by investigating how the local computation would affect global differential privacy as a composition theorem. A similar result is also established in Kairouz et al. (2015).

Theorem 5 (Composition Theorem) Denote iterates generated by the specific stochastic algorithm with  $K$  steps as  $\{x^k\}_{k=1}^K$ . For the  $m$ -th agent, denote  $\tilde{\mathcal{A}}_m: \Xi \mapsto \{\tilde{x}^k(m)\}_{k=1}^K$ , where  $\tilde{x}^k(m)$  is the iterates corrupted by noise. Let  $\{\tilde{X}(m)\}_{m=1}^M = (\tilde{X}(1), \dots, \tilde{X}(M))$  with  $\tilde{X}(1) = (\tilde{x}^1(m), \dots, \tilde{x}^K(m))$ . For any fixed  $m$ , if  $\tilde{\mathcal{A}}_m$  is  $(\varepsilon_m, \delta_m)$ -differential private, then  $\left\{\tilde{X}(m)\right\}_{m=1}^M$  is  $(\varepsilon', \delta')$ -differential private, where,

$$
\delta^ {\prime} = 1 - \left\{\prod_ {m = 1} ^ {M} \left(1 - e ^ {a _ {m}} \frac {\delta_ {m}}{1 + e ^ {\varepsilon_ {m}}}\right) \right\} + \left\{1 - \prod_ {m = 1} ^ {M} \left(1 - \frac {\delta_ {m}}{1 + e ^ {\varepsilon_ {m}}}\right) \right\},
$$

$$
\varepsilon^ {\prime} = \min  \left\{\varepsilon_ {1}, \varepsilon_ {2}, \varepsilon_ {3} \right\},
$$

$$
\varepsilon_ {1} = \sum_ {m = 1} ^ {M} \varepsilon_ {m}, \quad \varepsilon_ {2} = \sum_ {m = 1} ^ {M} C _ {K L} (m) + \sqrt {2 l o g (\frac {1}{\delta^ {\prime}}) (\sum_ {m = 1} ^ {M} \varepsilon_ {m} ^ {2})},
$$

$$
\varepsilon_ {3} = \sum_ {m = 1} ^ {M} \frac {\left(e ^ {\varepsilon_ {m}} - 1\right) \varepsilon_ {m}}{e ^ {\varepsilon_ {m}} + 1} + \sqrt {\sum_ {m = 1} ^ {M} 2 \varepsilon_ {m} ^ {2} \log \left(e + \frac {\sqrt {\sum_ {m = 1} ^ {M} \varepsilon_ {m} ^ {2}}}{\tilde {\delta}}\right)},
$$

for some  $0 < a_{m} \leq \varepsilon_{m}$ ,  $\sum_{m=1}^{M} a_{m} = \varepsilon'$ , and a real constant  $\tilde{\delta}$ .

# 5 DIFFERENTIAL PRIVACY VIA TRUNCATED GENERALIZED GAUSSIAN MECHANISMS

While the commonly adopted Gaussian noise-adding mechanism for a single iterate can guarantee DP (Croft et al., 2022; Ghosh et al., 2012; Cormode et al., 2019), the extremely large noise

Algorithm 1 DP-KM iteration  
1: Initialize:  $\mathbf{X}^0$ , weight matrix  $\mathbf{W}$ ,  $\alpha_{k} \in (0,1]$ , number of iterations  $K$ , variance of noise  $\sigma$ , scale parameter  $b$   
2: while  $k \leq K$  do  
3: for  $m \in V$  ( $m \in [1,M]$ ) do  
4: Let  $\mathbf{X}^k = \tilde{\mathbf{X}}^{k-1}$   
5: 1. Local computation  
6:  $x^k(m) = \mathbf{W}\mathbf{X}^k(m) + \alpha_k(T(x^k(x(m))) - x^k(m))$   
7: end for  
8: for  $m \in V$  do  
9: 2. Add noise for privacy guarantee,  $\varepsilon_m^k \sim GG(0,\sigma,b)$   
10:  $\tilde{x}^k(m) = x^k(m) + \varepsilon_m^k$   
11: Broadcast  $\tilde{x}^k(m)$  to all neighbours  $j \in \mathcal{N}(m)$   
12: end for  
13: end while  
14: Output:  $\mathbf{X}^K = (x^K(1),\dots,x^K(M))$  and  $\bar{x}^K = \frac{1}{M}\sum_{m=1}^{M}x^K(m)$

will severely affect a learning process and degrade the performance of the trained model under differential privacy guarantee. Beyond satisfying DP, once the iterate is posed, especially for constrained optimization that requests an output fall into a close-set  $\mathcal{R}$ , i.e., it also require,  $\forall k, \mathbb{P}_d(x^k + d \in \mathcal{R} \mid x^k) \geq 1 - \epsilon$ , for some  $\epsilon > 0$ . To rectify this and therefore improve the utility, in this section, we consider a new noise-adding mechanism by adding controlled noise to iterates. The proposed mechanism unifies the Laplace and the Gaussian mechanism in a general family. [Liu (2018)] considers a similar mechanism for the generalized Gaussian mechanism by truncating it to the valid range.

Truncated generalized Gaussian mechanism Different from the common mechanisms, we truncate the probability density function used for the generation of noise with a careful determination of an appropriate bounding parameter. In the reminder, we consider truncated Generalized Gaussian (GG) distribution  $\mathcal{P}_d\coloneqq \mathrm{GG}(0,\sigma ,b)$  with location parameter 0, scale parameter  $\sigma >0$ , shape parameter  $b > 0$ . Its probability density function is,

$$
p (z \mid 0, \sigma , b) = C _ {g g} \exp \left\{- \left(\frac {| z |}{\sigma}\right) ^ {b} \right\}, \text {w h e r e} z \in [ - A, A ],
$$

where  $C_{gg}$  is a constant to guarantee  $\int_{-A}^{A} p(z \mid 0, \sigma, b) dz = 1$ . The differentially private DKM with truncated GG noise is shown in Algorithm 1. In applications, we consider the case of  $b = 1, 2$ , which represents the truncated Laplace distribution and truncated normal distribution, see Definition 6-7. We address that it can be extended to the case  $b \geq 3$  to preserve a differential privacy guarantee.

Definition 6 (Truncated Laplacian Distribution Geng et al. (2018)) Given the privacy parameters  $0 < \delta < \frac{1}{2}, \varepsilon > 0$  and iterates sensitivity  $\Delta > 0$ , the probability density function of the truncated Laplacian distribution is defined as,

$$
p _ {L a p} \left(z\right) := \left\{ \begin{array}{l l} C _ {L a p} e ^ {- \frac {| z |}{\lambda}}, & f o r   z \in [ - A, A ], \\ 0, & o t h e r w i s e, \end{array} \right.
$$

where  $\lambda := \frac{\Delta}{\varepsilon}$ ,  $A := \frac{\Delta}{\varepsilon} \log \left(1 + \frac{e^{\varepsilon} - 1}{2\delta}\right)$ ,  $C_{Lap} := \frac{1}{2\lambda \left(1 - e^{-\frac{A}{\lambda}}\right)} = \frac{1}{2\frac{\Delta}{\varepsilon} \left(1 - \frac{1}{1 + \frac{e^{\varepsilon} - 1}{2\delta}}\right)}$ .

Geng et al. (2018) shows the optimality of the truncated Laplacian mechanism for minimizing the noise amplitude and noise power under  $(\varepsilon, \delta)$ -differential privacy guarantee. In the following, we establish that truncated Gaussian noise also preserves  $(\varepsilon, \delta)$ -differential privacy.

Definition 7 (Truncated Gaussian Distribution) The probability density function of the truncated Gaussian distribution is defined as,

$$
p _ {n o r} (z) = C _ {n o r} \exp \left\{- \left(\frac {| z |}{\sigma}\right) ^ {2} \right\}, \quad f o r z \in [ - A, A ],
$$

where  $C_{nor}$  is a constant to guarantee  $\int_{-A}^{A} p_{nor}(z) dz = 1$ ,  $\sigma^2 \geq \varepsilon^{-1} \Delta^2$ . In practice, the  $C_{nor}$  and  $A$  are determined by the equation,

$$
\left\{ \begin{array}{l} C _ {n o r} \cdot \sum_ {l = 0} ^ {\infty} (- 1) ^ {l} \cdot \frac {A ^ {2 l + 1}}{\sigma^ {2 l} l ! (2 l + 1)} = \frac {1}{2}, \\ C _ {n o r} \cdot \sum_ {l = 0} ^ {\infty} (- 1) ^ {l} \frac {A ^ {2 l + 1} - (A - \Delta) ^ {2 l + 1}}{\sigma^ {2 l} l ! (2 l + 1)} = \delta . \end{array} \right.
$$

Theorem 6 The truncated Gaussian mechanism preserves  $(\varepsilon, \delta)$ -differential privacy.

An important property of the truncated GG mechanism is that the range of addition noises are bounded to  $[-A, A]$  while the DP still holds. We emphasize that, however, it cannot be applied as a way to achieve adherence to the valid range. Given constraints on the range of output, to ensure the utility, designing appropriate mechanisms are needed to be adaptive to the various constraint configurations, which will be one of the core aspects of our future work.

# 6 NUMERICAL EXPERIMENT

In this section, we evaluate the performance of DP-KM by considering  $\ell_1$  regularized least square regression,

$$
\min  _ {x \in \mathbb {R} ^ {p}} \frac {1}{M N} \sum_ {m = 1} ^ {M} \sum_ {i = 1} ^ {N} (A _ {m i} x - b _ {m i}) ^ {2}, \quad \min  _ {x \in \mathbb {R} ^ {p}} \frac {1}{M N} \sum_ {m = 1} ^ {M} \sum_ {i = 1} ^ {N} (A _ {m i} x - b _ {m i}) ^ {2} + \lambda \| x \|.
$$

We consider using SGD, SPGD and ADMM algorithms and evaluate the performance of its privacy variants on several decentralized settings: ring, star and full connected graph. We compare our proposed DP-KM algorithm with three baseline algorithms: (a) non-private decentralized approach; (b) private decentralized approach with Laplace noise; (c) private decentralized approach with Gaussian noise. Bias and average root mean squared errors (RMSEs) are used to quantify estimation performance and prediction accuracy. The result are summarized in Figure 1-4.

![](images/9ed600f247007a8e8b6c8d452b39a7b8d1cfcb1923cd7c6e98fbdef31bd43f10.jpg)

![](images/1194c5ca3563b8fd6c2c45a2b5641cdea628820c806bd541b2dcf089a4556ffb.jpg)

![](images/8ade07295d483319f843ab99c64960834dba99d66ba4ad8267c0fffcc62c0d31.jpg)

![](images/4d0949bf961baeb8d478d0c4d4cf9d31251c1f8244d10e37e761521fb6ba73a4.jpg)

![](images/5a203a7bdc357d34508cb23b3c818e56c86de9185e58824b08af41341bff15d2.jpg)  
Figure 1: SGD with a fully connected graph.

![](images/032cd48b38ff58e88c1c9b26e1eefe3af9eeec67fb228cc163d401abb4769922.jpg)

![](images/f53909faec5ae7bf5a1003b5faa44978d428634e69514280c91b8c4924c4cd87.jpg)

![](images/df7ed4a6af3c5aaec2abdf28ab07ac109f6cb031c4b2079ba2722e364a2e7d3a.jpg)

Figure 1 shows that a truncated mechanism with larger  $\varepsilon$  and larger  $\delta$  has better convergence. And the proposed mechanism has smaller estimation errors and prediction errors compared with Laplace and Gaussian mechanisms. In addition, our results demonstrate privacy-utility trade-offs of our approach. When privacy leakage increases, our truncated Laplace approach achieves better utility. We, then, fix the privacy budget and discuss the performance of the truncated approach under the settings that have different numbers of distributed data sources and different typologies, as shown in Figure 2, 3, 4. We demonstrate the effects of the structure of connected graphs. Compared with the ring graph, these results show a better convergence with the star and full-connected graphs.

![](images/c921aa730f63fd69893b707c9ee8883ecc8768fb93bbed45ae5cd8bb6183153a.jpg)

![](images/ec3a0d0deaad3bcd6174c378196535c15f37604b409e9cd3cbabc5c4de8fb106.jpg)  
Figure 2: SGD with three graphs.

![](images/9f63334ae8ac1f66866902844fb4781179c133b094b538d4c9fc944b36d54770.jpg)

![](images/2cbda83bf5a9fefcf34f3e6181c6d03ab55d8ea566952ae848f0b47837ebffd3.jpg)

![](images/8dd220d6da91c4be0068d2cb5e95b085800efd5b6ca7fe82bae5c7cf205f67fb.jpg)

![](images/03979352b50e5750eaf5896c510e8c15de414811f7719213bc876f1ce35a7b52.jpg)

![](images/813572f01fe8e1d63f8350b02e1117991b29e0ce6636edd6b5d012e370664f7d.jpg)

![](images/45915ac9d96603428b3ab3eb9640c4526bedeb510def78bc8d834597e58712b6.jpg)

![](images/f41d2ec0e3fa4383bbe744d449b49cd48a8bbacb0ba8bc5c750ae4f3bd4c44f6.jpg)

![](images/80e6e9f99b61a88bf6a70c1838b2d2bf5827d4c2930f793768e34b68dd42cd15.jpg)  
Figure 3:SPGD with three graphs.

![](images/536372fdae133a6ac854d0e92db158bdbebb0cb58fe685a58aa2e6089e6479e2.jpg)

![](images/db5fde9cce3b25bac95e8eed22539573d1551407674a0daca8e061f405c49fe7.jpg)

![](images/050750c17152adbb72e4b24529b5a2edb37d93a3111beb9bb33bc06071be3401.jpg)

![](images/a7dac6827831feaa595d0d7e2a2b182264ba22cae15aeabfadbd5901510bc78e.jpg)

![](images/69fd92b4495ecb10a29b2d9a8808900c6521094aecc7dc645117804a5533d31e.jpg)

![](images/74b5c330516945b866afbc64c0ab894746a6162dadc6206b2d573360c11bcffc.jpg)

![](images/279d82ee5cfb2520c868b3fa27a5d13a3f2449df1d40dabc42ffb4982c6c47bd.jpg)

![](images/10c3b41732ac2090278ec4ee7101c2d05b6d7c16d8988530a93013716fa0ef4a.jpg)  
Figure 4: ADMM with three graphs.

![](images/e62710ecefc152431fa27e40db16dcdb37e76dc9f68f4bdb22c089aa448ee19f.jpg)

![](images/d2f42ba8b2e225dd9c13f315369bdbea16794c3626ad39ef254f44de3b581fe8.jpg)

![](images/f615365ba8d36d0d69088784e0f8263c108fa52f45c553c2aec4815de4808f0a.jpg)

![](images/e6a77e22a0e7b09bf04d01ec9faf278f82df8f01c1d7779cab4c92f4b803f937.jpg)

![](images/56dcaefc66e4bb6921b54c0ef8ce6e261adbe4b7e3839a2e2935444d89c84e27.jpg)

![](images/086b2cb4f5e98f29be468ba926da8fc9c4244eeaba28c7ae002bcbe66baeae91.jpg)

# 7 CONCLUSION

Leveraging the decentralized learning procedure to train models under privacy constraints, in this paper, we have proposed a general framework of privacy-preserving algorithm, DP-KM, that is applicable to much of the existing work on optimization and consensus computation and, show that the proposed algorithm retains the performance guarantee regarding stability, generalization, and finite sample performance. The effect of local privacy-preserving computation on global differential privacy is also derived. To avoid extremely large addition noise added to the shared information that will severely affect and degrade the performance of learning process, we have introduced truncated generalized Gaussian mechanism, in which we demonstrate privacy and utility trade-offs under differential privacy guarantee. Experiments have demonstrated that our algorithm is highly effective in decentralized settings and that it performs better than state-of-the-art baseline algorithms.

# REFERENCES

Sulaiman Alghunaim, Kun Yuan, and Ali H Sayed. A linearly convergent proximal gradient algorithm for decentralized optimization. Advances in Neural Information Processing Systems, 32, 2019.  
Hilal Asi, John Duchi, Alireza Fallah, Omid Javidbakht, and Kunal Talwar. Private adaptive gradient methods for convex optimization. In International Conference on Machine Learning, pp. 383-392. PMLR, 2021.  
Aurelien Bellet, Rachid Guerraoui, Mahsa Taziki, and Marc Tommasi. Personalized and private peer-to-peer machine learning. In International Conference on Artificial Intelligence and Statistics, pp. 473-481. PMLR, 2018.  
Pascal Bianchi, Walid Hachem, and Franck Iutzeler. A stochastic primal-dual algorithm for distributed asynchronous composite optimization. In IEEE Global Conference on Signal and Information Processing (GlobalSIP), pp. 732-736. IEEE, 2014.  
Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends® in Machine learning, 3(1):1-122, 2011.  
Francesco Bullo, Jorge Cortés, and Sonia Martinez. Distributed control of robotic networks: a mathematical approach to motion coordination algorithms, volume 27. Princeton University Press, 2009.  
Graham Cormode, Tejas Kulkarni, and Divesh Srivastava. Constrained private mechanisms for count data. IEEE Transactions on Knowledge and Data Engineering, 33(2):415-430, 2019.  
William Croft, Jörg-Rüdiger Sack, and Wei Shi. Differential privacy via a truncated and normalized laplace mechanism. Journal of Computer Science and Technology, 37(2):369-388, 2022.  
Damek Davis and Wotao Yin. Convergence rate analysis of several splitting schemes. In *Splitting Methods in Communication, Imaging, Science, and Engineering*, pp. 115-163. Springer, 2016.  
Jinshuo Dong, Aaron Roth, and Weijie J Su. Gaussian differential privacy. arXiv preprint arXiv:1905.02383, 2019.  
John Duchi and Ryan Rogers. Lower bounds for locally private estimation via communication complexity. In Conference on Learning Theory, pp. 1161-1191. PMLR, 2019.  
Cynthia Dwork and Guy N Rothblum. Concentrated differential privacy. arXiv preprint arXiv:1603.01887, 2016.  
Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography Conference, pp. 265-284. Springer, 2006.  
Farhad Farokhi. Distributionally-robust machine learning using locally differentially-private data. Optimization Letters, 16(4):1167-1179, 2022.  
Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security, pp. 1322-1333, 2015.  
Quan Geng, Wei Ding, Ruiqi Guo, and Sanjiv Kumar. Privacy and utility tradeoff in approximate differential privacy. arXiv preprint arXiv:1810.00877, 2018.  
Arpita Ghosh, Tim Roughgarden, and Mukund Sundararajan. Universally utility-maximizing privacy mechanisms. SIAM Journal on Computing, 41(6):1673-1693, 2012.  
Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In International Conference on Machine Learning, pp. 1225-1234. PMLR, 2016.

Fengxiang He, Bohan Wang, and Dacheng Tao. Tighter generalization bounds for iterative differentially private learning algorithms. In Uncertainty in Artificial Intelligence, pp. 802-812. PMLR, 2021.  
Zonghao Huang, Rui Hu, Yuanxiong Guo, Eric Chan-Tin, and Yanmin Gong. Dp-admm: Admm-based distributed learning with differential privacy. IEEE Transactions on Information Forensics and Security, 15:1002-1012, 2019.  
Peter Kairouz, Sewoong Oh, and Pramod Viswanath. The composition theorem for differential privacy. In International Conference on Machine Learning, pp. 1376-1385. PMLR, 2015.  
Yiwei Li, Shuai Wang, Tsung-Hui Chang, and Chong-Yung Chi. Federated stochastic primal-dual learning with differential privacy. arXiv preprint arXiv:2204.12284, 2022.  
Fang Liu. Generalized gaussian mechanism for differential privacy. IEEE Transactions on Knowledge and Data Engineering, 31(4):747-756, 2018.  
Andrew McGregor, Ilya Mironov, Toniann Pitassi, Omer Reingold, Kunal Talwar, and Salil Vadhan. The limits of two-party differential privacy. In IEEE 51st Annual Symposium on Foundations of Computer Science, pp. 81-90. IEEE, 2010.  
Ilya Mironov. Rényi differential privacy. In IEEE 30th Computer Security Foundations Symposium (CSF), pp. 263-275. IEEE, 2017.  
Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of machine learning. MIT press, 2018.  
Angelia Nedic, Alex Olshevsky, Asuman Ozdaglar, and John N Tsitsiklis. Distributed subgradient methods and quantization effects. In The 47th IEEE Conference on Decision and Control, pp. 4177-4184. IEEE, 2008.  
Michael Rabbat and Robert Nowak. Distributed optimization in sensor networks. In Proceedings of the 3rd International Symposium on Information Processing in Sensor Networks, pp. 20-27, 2004.  
Ali H Sayed et al. Adaptation, learning, and optimization over networks. Foundations and Trends® in Machine Learning, 7(4-5):311-801, 2014.  
Wei Shi, Qing Ling, Kun Yuan, Gang Wu, and Wotao Yin. On the linear convergence of the admm in decentralized consensus optimization. IEEE Transactions on Signal Processing, 62(7):1750-1761, 2014.  
Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In IEEE Symposium on Security and Privacy (SP), pp. 3-18. IEEE, 2017.  
Shuang Song, Kamalika Chaudhuri, and Anand D Sarwate. Stochastic gradient descent with differentially private updates. In 2013 IEEE Global Conference on Signal and Information Processing, pp. 245-248. IEEE, 2013.  
Tao Sun, Dongsheng Li, and Bao Wang. Stability and generalization of decentralized stochastic gradient descent. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 9756-9764, 2021.  
Yin Wotao. Operator splitting methods for decentralized optimization. Mathematica Numerica Sinica, 41(3):225, 2019.  
Bicheng Ying, Kun Yuan, Hanbin Hu, Yiming Chen, and Wotao Yin. Bluefog: Make decentralized algorithms practical for optimization and deep learning. arXiv preprint arXiv:2111.04287, 2021.  
Dongxiao Yu, Zongrui Zou, Shuzhen Chen, Youming Tao, Bing Tian, Weifeng Lv, and Xiuzhen Cheng. Decentralized parallel sgd with privacy preservation in vehicular networks. IEEE Transactions on Vehicular Technology, 70(6):5211-5220, 2021a. doi: 10.1109/TVT.2021.3064877.

Dongxiao Yu, Zongrui Zou, Shuzhen Chen, Youming Tao, Bing Tian, Weifeng Lv, and Xiuzhen Cheng. Decentralized parallel sgd with privacy preservation in vehicular networks. IEEE Transactions on Vehicular Technology, 70(6):5211-5220, 2021b.  
Tongtian Zhu, Fengxiang He, Lan Zhang, Zhengyang Niu, Mingli Song, and Dacheng Tao. Topology-aware generalization of decentralized sgd. In International Conference on Machine Learning, pp. 27479-27503. PMLR, 2022.
