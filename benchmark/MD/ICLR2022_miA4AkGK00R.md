# EF21 WITH BELLs & WHISTLES: PRACTICAL ALG- RITHMIC EXTENSIONS OF MODERN ERROR FEEDBACK

Anonymous authors

Paper under double-blind review

# ABSTRACT

First proposed by Seide et al. (2014) as a heuristic, error feedback (EF) is a very popular mechanism for enforcing convergence of distributed gradient-based optimization methods enhanced with communication compression strategies based on the application of contractive compression operators. However, existing theory of EF relies on very strong assumptions (e.g., bounded gradients), and provides pessimistic convergence rates (e.g., while the best known rate for EF in the smooth nonconvex regime, and when full gradients are compressed, is  $O(1 / T^{2 / 3})$ , the rate of gradient descent in the same regime is  $O(1 / T)$ ). Recently, Richtárik et al. (2021) (2021) proposed a new error feedback mechanism, EF21, based on the construction of a Markov compressor induced by a contractive compressor. EF21 removes the aforementioned theoretical deficiencies of EF and at the same time works better in practice. In this work we propose six practical extensions of EF21, all supported by strong convergence theory: partial participation, stochastic approximation, variance reduction, proximal setting, momentum and bidirectional compression. Several of these techniques were never analyzed in conjunction with EF before, and in cases where they were (e.g., bidirectional compression), our rates are vastly superior.

# 1 INTRODUCTION

In this paper, we consider the nonconvex distributed/federated optimization problem of the form

$$
\min  _ {x \in \mathbb {R} ^ {d}} \left\{f (x) \stackrel {\text {d e f}} {=} \frac {1}{n} \sum_ {i = 1} ^ {n} f _ {i} (x) \right\}, \tag {1}
$$

where  $n$  denotes the number of clients/workers/devices/nodes connected with a server/master and client  $i$  has an access to the local loss function  $f_{i}$  only. The local loss of each client is allowed to have the online/expectation form

$$
f _ {i} (x) = \mathbb {E} _ {\xi_ {i} \sim \mathcal {D} _ {i}} \left[ f _ {\xi_ {i}} (x) \right], \tag {2}
$$

or the finite-sum form

$$
f _ {i} (x) = \frac {1}{m} \sum_ {j = 1} ^ {m} f _ {i j} (x). \tag {3}
$$

Problems of this structure appear in federated learning (Konečný et al., 2016; Kairouz, 2019), where training is performed directly on the clients' devices. In a quest for state-of-the-art performance, machine learning practitioners develop elaborate model architectures and train their models on enormous data sets. Naturally, for training at this scale to be possible, one needs to rely on distributed computing (Goyal et al., 2017; You et al., 2020). Since in recent years remarkable empirical successes were obtained with massively over-parameterized models (Arora et al., 2018), which puts an extra strain on the communication links during training, recent research activity and practice focuses on developing distributed optimization methods and systems capitalizing on (deterministic or randomized) lossy communication compression techniques to reduce the amount of communication traffic.

A compression mechanism is typically formalized as an operator  $\mathcal{C}:\mathbb{R}^d\mapsto \mathbb{R}^d$  mapping hard-to-communicate (e.g., dense) input messages into easy-to-communicate (e.g., sparse) output messages. The operator is allowed to be randomized, and typically operates on models Khaled & Richtarik

(2019) or on gradients Alistarh et al. (2017); Beznosikov et al. (2020), both of which can be described as vectors in  $\mathbb{R}^d$ . Besides sparsification (Alistarh et al., 2018), typical examples of useful compression mechanisms include quantization (Alistarh et al., 2017; Horváth et al., 2019a) and low-rank approximation (Vogels et al., 2019; Safaryan et al., 2021).

There are two large classes of compression operators often studied in the literature: i) unbiased compression operators  $\mathcal{C}$ , meaning that there exists  $\omega \geq 0$  such that

$$
\mathbb {E} [ \mathcal {C} (x) ] = x, \quad \mathbb {E} \left[ \| \mathcal {C} (x) - x \| ^ {2} \right] \leq \omega \| x \| ^ {2}, \quad \forall x \in \mathbb {R} ^ {d}; \tag {4}
$$

and ii) biased compression operators  $\mathcal{C}$ , meaning that there exists  $0 < \alpha \leq 1$  such that

$$
\mathbb {E} \left[ \| \mathcal {C} (x) - x \| ^ {2} \right] \leq (1 - \alpha) \| x \| ^ {2}, \quad \forall x \in \mathbb {R} ^ {d}. \tag {5}
$$

Note that the latter "biased" class contains the former one, i.e., if  $\mathcal{C}$  satisfies (4) with  $\omega$ , then a scaled version  $(1 + \omega)^{-1}\mathcal{C}$  satisfies (5) with  $\alpha = 1 / (1 + \omega)$ . While distributed optimization methods with unbiased compressors (4) are well understood (Alistarh et al., 2017; Khirirat et al., 2018; Mishchenko et al., 2019; Horvath et al., 2019b; Li et al., 2020; Li & Richtárik, 2021a; Li & Richtárik, 2020; Islamov et al., 2021; Gorbunov et al., 2021), biased compressors (5) are significantly harder to analyze. One of the main reasons behind this is rooted in the observation that when deployed within distributed gradient descent in a naive way, biased compressors may lead to (even exponential) divergence (Karimireddy et al., 2019; Beznosikov et al., 2020). Error Feedback (EF) (or Error Compensation (EC))—a technique originally proposed by Seide et al. (2014)—emerged as an empirical fix of this problem. However, this technique remained poorly understood until very recently.

Although several theoretical results were obtained supporting the EF framework in recent years (Stich et al., 2018; Alistarh et al., 2018; Beznosikov et al., 2020; Gorbunov et al., 2020; Qian et al., 2020; Tang et al., 2020; Koloskova et al., 2020), they use strong assumptions (e.g., convexity, bounded gradients, bounded dissimilarity), and do not get  $\mathcal{O}(1 / \alpha T)$  convergence rates in the smooth nonconvex regime. Very recently, Richtárik et al. (2021) proposed a new EF mechanism called EF21, which uses standard smoothness assumptions only, and also enjoys the desirable  $O(1 / \alpha T)$  convergence rate for the nonconvex case (in terms of number of communication rounds  $T$  this matches the best-known rate  $\mathcal{O}((1 + \omega /\sqrt{n}) / T)$  obtained by Gorbunov et al. (2021) using unbiased compressors), improving the previous  $O(1 / (\alpha T)^{2 / 3})$  rate of the standard EF mechanism (Koloskova et al., 2020).

# 2 OUR CONTRIBUTIONS

While Richtárik et al. (2021) provide a new theoretical SOTA for error feedback based methods, the authors only study their EF21 mechanism in a pure form, without any additional "bells and whistles" which are of importance in practice. In this paper, we aim to push the EF21 framework beyond its pure form by extending it in several directions of high theoretical and practical importance. In particular, we further enhance the EF21 mechanism with the following six useful and practical algorithmic extensions: stochastic approximation, variance reduction, partial participation, bidirectional compression, momentum, and proximal (regularization). We do not stop at merely proposing these algorithmic enhancements: we derive strong convergence results for all of these extensions. Several of these techniques were never analyzed in conjunction with the original EF mechanism before, and in cases where they were, our new results with EF21 are vastly superior. See Table 1 for an overview of our results. In summary, our results constitute the new algorithmic and theoretical state-of-the-art in the area of error feedback.

We now briefly comment on each extension proposed in this paper:

$\diamond$  Stochastic approximation. The vanilla EF21 method requires all clients to compute the exact/full gradient in each round. While Richtárik et al. (2021) do consider a stochastic extension of EF21, they do not formalize their result, and only consider the simplistic scenario of uniformly bounded variance, which does not in general hold for stochasticity coming from subsampling (Khaled & Richtárik, 2020). However, exact gradients are not available in the stochastic/online setting (2), and in the finite-sum setting (3) it is more efficient in practice to use subsampling and work with stochastic gradients instead. In our paper, we extend EF21 to a more general stochastic approximation framework than the simplistic framework considered in the original paper. Our method is called EF21-SGD (Algorithm 2); see Appendix D for more details.

<table><tr><td>Setup</td><td>Method</td><td>Citation</td><td>Compl. (NC)</td><td>Compl. (PL)</td><td>Comment</td></tr><tr><td>Full grads</td><td>EF21</td><td>Richtárik et al. (2021)</td><td>1/αε2</td><td>1/αμ</td><td></td></tr><tr><td rowspan="4">Stoch. grads</td><td>Choco-SGD</td><td>Koloskova et al. (2020)</td><td>1/ε2 + G/αε3 + σ2/nε4</td><td>N/A</td><td>||∇f(x)|| ≤ G</td></tr><tr><td>EF21-SGD</td><td>Richtárik et al. (2021)</td><td>1/αε2 + σ2/α3ε4</td><td>1/αμ + σ2/μ2α3ε</td><td>UBV (Ex. 1)</td></tr><tr><td>EF21-SGD</td><td>NEW</td><td>1/αε2 + 1+Δinf/α3ε4</td><td>1/αμ + 1+Δinf/μ2α3ε</td><td>IS (Ex. 2)</td></tr><tr><td>EF21-PAGE</td><td>NEW</td><td>√m+1/α/ε2 + m</td><td>√m+1/α/μ + m</td><td>fi(x) = 1/m ∑j=1m fij(x)</td></tr><tr><td>PP</td><td>EF21-PP</td><td>NEW</td><td>1/pαε2(1) + 1/αε2</td><td>1/pαμ(1) + 1/αμ</td><td>Full grads</td></tr><tr><td rowspan="2">BC</td><td>DoubleSqueeze</td><td>Tang et al. (2020)</td><td>1/ε2 + Δ/ε3 + σ2/nε4</td><td>N/A</td><td>E [||C(x) - x||] ≤ Δ</td></tr><tr><td>EF21-BC</td><td>NEW</td><td>1/αw/αM/ε2</td><td>1/αw/αM/μ</td><td>Full grads</td></tr><tr><td rowspan="2">Mom.</td><td>M-CSER</td><td>Xie et al. (2020)(2)</td><td>1/ε2 + G/(1-η)αε3</td><td>N/A</td><td>||∇f(x)|| ≤ G</td></tr><tr><td>EF21-HB</td><td>NEW</td><td>1/ε2(1/1-η + 1/α)</td><td>N/A</td><td>Full grads</td></tr><tr><td>Prox</td><td>EF21-Prox</td><td>NEW</td><td>1/αε2</td><td>1/αμ(3)</td><td>Full grads</td></tr></table>

(1) Red term = number of communication rounds, blue term = expected number of gradient computations per client.  
(2) Xie et al. (2020) consider Nesterov's momentum. Moreover, they analyzed the version with stochastic gradients, bidirectional compression and local steps. However, the derived result is not better than state-of-the-art ones with either stochastic gradients or bidirectional compression. Therefore, to maintain the table compact, we do not include the results of Xie et al. (2020) in the other parts of the table.  
(3) This result is obtained under the generalized PL-condition for composite optimization problems (see Assumption 5 from Appendix I.2).

Table 1: Summary of the state-of-the-art complexity results for finding an  $\varepsilon$ -stationary point, i.e., such a point  $\hat{x}$  that  $\mathbb{E}\left[\|\nabla f(\hat{x})\|^2\right] \leq \varepsilon^2$ , for generally non-convex functions and an  $\varepsilon$ -solution, i.e., such a point  $\hat{x}$  that  $\mathbb{E}\left[f(\hat{x}) - f(x^*)\right] \leq \varepsilon$ , for functions satisfying PL-condition using error-feedback type methods. By (computation) complexity we mean the average number of (stochastic) first-order oracle calls needed to find an  $\varepsilon$ -stationary point ("Compl. (NC)") or  $\varepsilon$ -solution ("Compl. (PL)"). Removing the terms colored in blue from the complexity bounds shown in the table, one can get communication complexity bounds, i.e., the total number of communication rounds needed to find an  $\varepsilon$ -stationary point ("Compl. (NC)") or  $\varepsilon$ -solution ("Compl. (PL)"). Dependences on the numerical constants, "quality" of the starting point, and smoothness constants are omitted in the complexity bounds. Moreover, dependencies on  $\log(1/\varepsilon)$  are also omitted in the column "Compl. (PL)". Abbreviations: "BC" = bidirectional compression, "PP" = partial participation; "Mom." = momentum;  $T$  = the number of communications rounds needed to find an  $\varepsilon$ -stationary point;  $\# \text{grads} =$  the number of (stochastic) first-order oracle calls needed to find an  $\varepsilon$ -stationary point. Notation:  $\alpha =$  the compression parameter,  $\alpha_w$  and  $\alpha_M =$  the compression parameters of worker and master nodes respectively for EF21-BC,  $\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} \sigma_i^2$  (see Example 1),  $\Delta^{\inf} = f^{\inf} - \frac{1}{n} \sum_{i=1}^{n} \frac{1}{m_i} \sum_{j=1}^{m_i} f_{ij}^{\inf}$  (see Example 2),  $p =$  probability of sampling the client in EF21-PP,  $\eta =$  momentum parameter. To the best of our knowledge, combinations of error feedback with partial participation (EF21-PP) and proximal versions of error feedback (EF21-Prox) were never analyzed in the literature.

$\diamond$  Variance reduction. As mentioned above, EF21 relies on full gradient computations at all clients. This incurs a high or unaffordable computation cost, especially when local clients hold large training sets, i.e., if  $m$  is very large in (3). In the finite-sum setting (3), we enhance EF21 with a variance reduction technique to reduce the computational complexity. In particular, we adopt the simple and efficient variance-reduced method PAGE (Li et al., 2021; Li, 2021b) (which is optimal for solving problems (3)) into EF21, and call the resulting method EF21-PAGE (Algorithm 3). See Appendix E for more details.  
$\diamond$  Partial participation. The EF21 method proposed by Richtárik et al. (2021) requires full participation of clients for solving problem (1), i.e., in each round, the server needs to communicate with all  $n$  clients. However, full participation is usually impractical or very hard to achieve in massively distributed (e.g., federated) learning problems (Konečný et al., 2016; Cho et al., 2020; Kairouz, 2019; Li & Richtárik, 2021b; Zhao et al., 2021). To remedy this situation, we propose a partial participation (PP) variant of EF21, which we call EF21-PP (Algorithm 4). See Appendix F for more details.  
$\diamond$  Bidirectional compression. The vanilla EF21 method only considers upstream compression of the messages sent by the clients to the server. However, in some situations, downstream communication is also costly (Horváth et al., 2019a; Tang et al., 2020; Philippenko & Dieuleveut, 2020). In order to cater to these situations, we modify EF21 so that the server can also optionally compresses messages before communication. Our master compression is intelligent in that it employs the Markov compressor proposed in EF21 to be used at the devices. The proposed method, based on bidirectional compression, is EF21-BC (Algorithm 5). See Appendix G for more details.

Table 2: Description of the methods developed and analyzed in the paper. For the ease of comparison, we also provide a description of EF21. In all methods only compressed vectors  $c_i^t$  are transmitted from workers to the master and the master broadcasts non-compressed iterates  $x^{t + 1}$  (except EF21-BC, where the master broadcasts compressed vector  $b^{t + 1}$ ). Initialization of  $g_{i}^{0}, i = 1,\dots ,n$  can be arbitrary (possibly randomized). One possible choice is  $g_{i}^{0} = \mathcal{C}(\nabla f_{i}(x^{0}))$ . The pseudocodes for each method are given in the appendix.  

<table><tr><td>Update</td><td>Method</td><td>Alg. #</td><td>c^t_i</td><td>Comment</td></tr><tr><td rowspan="4">xt+1 = xt - γgt, g^t = 1/n ∑i=1n gi^t, gi^{t+1} = gi^t + ci^t</td><td>EF21</td><td>Alg. 1</td><td>C(∇fi(xt+1) - gt)</td><td></td></tr><tr><td>EF21-SGD</td><td>Alg. 2</td><td>C(hg(i)(xt+1) - gt)</td><td>h^t_i(xt+1) satisfies As. 2</td></tr><tr><td>EF21-PAGE</td><td>Alg. 3</td><td>C(v^t_{i+1} - gi^t)</td><td>b^t_i \sim Be(p), v^t_{i+1} = ∇fi(xt+1), if b^t_i = 1, v^t_{i+1} = vi^t + 1/τ_i ∑j∈Itv^t (xt+1) - 1/τ_i ∑j∈Itv^t (xt), if b^t_i = 0, I^t_i is a minibatch, |I^t_i| = τ_i</td></tr><tr><td>EF21-PP</td><td>Alg. 4</td><td>C(∇fi(xt+1) - gt)</td><td>if i ∈ St, if i∉St</td></tr><tr><td>xt+1 = xt - γgt, g^{t+1} = gt + bt+1, b^{t+1} = CM(θ^{t+1} - gt), θ^{t+1} = 1/n ∑i=1n gi^{t+1}, θ^{t+1}_{i} = θ^{t}_{i} + ci^t</td><td>EF21-BC</td><td>Alg. 5</td><td>Cw(∇fi(xt+1) - g^t_i)</td><td>Master broadcasts b^{t+1}; Cw is used on the workers&#x27; side, C_M is used on the master&#x27;s side</td></tr><tr><td>xt+1 = xt - γvt, vt+1 = ηvt + gt+1, g^{t+1} = 1/n ∑i=1n gi^{t+1}, g^{t+1}_{i} = g^{t}_{i} + ci^t</td><td>EF21-HB</td><td>Alg. 6</td><td>C(∇fi(xt+1) - gt)</td><td>η ∈ [0,1) - momentum parameter</td></tr><tr><td>xt+1 = proxγr(xt - γgt), g^{t+1} = 1/n ∑i=1n gi^{t+1}, g^{t+1}_{i} = gi^t + ci^t</td><td>EF21-Prox</td><td>Alg. 7</td><td>C(∇fi(xt+1) - gt)</td><td>For problem (6); proxγr(x) is defined in (91)</td></tr></table>

$\diamond$  Momentum. A very successful and popular technique for enhancing both optimization and generalization is momentum/acceleration (Polyak, 1964; Nesterov, 1983; Lan & Zhou, 2015; Allen-Zhu, 2017; Lan et al., 2019; Li, 2021a). For instance, momentum is a key building block behind the widely-used Adam method (Kingma & Ba, 2014). In this paper, we add the well-known (Polyak) heavy ball momentum (Polyak, 1964; Loizou & Richtárik, 2020) to EF21, and call the resulting method EF21-HB (Algorithm 6). See Appendix H for more details.  
$\diamond$  Proximal setting. It is common practice to solve regularized versions of empirical risk minimization problems instead of their vanilla variants (Shalev-Shwartz & Ben-David, 2014). We thus consider the composite/regularized/proximal problem

$$
\min  _ {x \in \mathbb {R} ^ {d}} \left\{\Phi (x) \stackrel {\text {d e f}} {=} \frac {1}{n} \sum_ {i = 1} ^ {n} f _ {i} (x) + r (x) \right\}, \tag {6}
$$

where  $r(x):\mathbb{R}^d\to \mathbb{R}\cup \{+\infty \}$  is a regularizer, e.g.,  $\ell_1$  regularizer  $\| x\| _1$  or  $\ell_{2}$  regularizer  $\| x\| _2^2$  . To broaden the applicability of EF21 to such problems, we propose a proximal variant of EF21 to solve the more general composite problems (6). We call this new method EF21-Prox (Algorithm 7). See Appendix I for more details.

Our theoretical complexity results are summarized in Table 1. In addition, we also analyze EF21-SGD, EF21-PAGE, EF21-PP, EF21-BC under Polyak-Lojasiewicz (PL) condition (Polyak, 1963; Lojasiewicz, 1963) and EF21-Prox under the generalized PL-condition (Li & Li, 2018) for composite optimization problems. Due to space limitations, we defer all the details about the analysis under the PL-condition to the appendix and provide only simplified rates in Table 1. We comment on some preliminary experimental results in Section 5. More experiments including deep learning experiments are presented in Appendix A.

# 3 METHODS

Since our methods are modifications of EF21, they share many features, and are presented in a unified way in Table 2. At each iteration of the proposed methods, worker  $i$  computes the compressed vector  $c_{i}^{t}$  and sends it to the master. The methods differ in the way of computing  $c_{i}^{t}$  but have similar (in case

of EF21-SGD, EF21-PAGE, EF21-PP – exactly the same) update rules to the one of EF21:

$$
x ^ {t + 1} = x ^ {t} - \gamma g ^ {t}, \quad g _ {i} ^ {t + 1} = g _ {i} ^ {t} + c _ {i} ^ {t}, \quad g ^ {t + 1} = \frac {1}{n} \sum_ {i = 1} ^ {n} g _ {i} ^ {t + 1} = g ^ {t} + \frac {1}{n} \sum_ {i = 1} ^ {n} c _ {i} ^ {t}. \tag {7}
$$

The pseudocodes of the methods are given in the appendix. Below we briefly describe each method.

$\diamond$  EF21-SGD: Error feedback and SGD. EF21-SGD is essentially EF21 but instead of the full gradients  $\nabla f_{i}(x^{t + 1})$ , workers compute the stochastic gradients  $\hat{g}_i(x^{t + 1})$ , and use them to compute  $c_{i}^{t} = \mathcal{C}(\hat{g}_{i}(x^{t + 1}) - g_{i}^{t})$ . Despite the seeming simplicity of this extension, it is highly important for various applications of machine learning and statistics where exact gradients are either unavailable or prohibitively expensive to compute.

$\diamond$  EF21-PAGE: Error feedback and variance reduction. In the finite-sum regime (3), variance reduced methods usually perform better than vanilla SGD in many situations (Gower et al., 2020). Therefore, for this setup we modify EF21 and combine it with variance reduction. In particular, this time we replace  $\nabla f_{i}(x^{t + 1})$  in the formula for  $c_i^t$  with the PAGE estimator (Li et al., 2021)  $v_{i}^{t + 1}$ . With (typically small) probability  $p$  this estimator equals the full gradient  $v_{i}^{t + 1} = \nabla f_{i}(x^{t + 1})$ , and with probability  $1 - p$  it is set to

$$
v _ {i} ^ {t + 1} = v _ {i} ^ {t} + \frac {1}{\tau_ {i}} \sum_ {j \in I _ {i} ^ {t}} \left(\nabla f _ {i j} (x ^ {t + 1}) - \nabla f _ {i j} (x ^ {t})\right),
$$

where  $I_{i}^{t}$  is a minibatch of size  $\tau_{i}$ . Typically, the number of data points  $m$  owned by each client is large, and  $p \leq 1 / m$  when  $\tau_{i} \equiv 1$ . As a result, computation of full gradients rarely happens during the optimization procedure: on average, once in every  $m$  iterations only. Although it is possible to use other variance-reduced estimators like in SVRG or SAGA, we use the PAGE-estimator: unlike SVRG or SAGA, PAGE is optimal for smooth nonconvex optimization, and therefore gives the best theoretical guarantees (we have obtained results for both SVRG and SAGA and indeed, they are worse, and hence we do not include them).

Notice that unlike VR-MARINA (Gorbunov et al., 2021), which is a state-of-the-art distributed optimization method designed specifically for unbiased compressors and which also uses the PAGE-estimator, EF21-PAGE does not require the communication of full (non-compressed) vectors at all. This is an important property of the algorithm since, in some distributed networks, and especially when  $d$  is very large, as is the case in modern over-parameterized deep learning, full vector communication is prohibitive. However, unlike the rate of VR-MARINA, the rate of EF21-PAGE does not improve with increasing  $n$ . This is not a flaw of our method, but rather an inevitable drawback of distributed methods that rely on biased compressors such as Top- $k$ .

$\diamond$  EF21-PP: Error feedback and partial participation. The extension of EF21 to the case of partial participation of the clients is mathematically identical to EF21 up to the following change:  $c_{i}^{t} = 0$  for all clients  $i \notin S_{t} \subseteq \{1, \dots, n\}$  that are not selected for communication at iteration  $t$ . In practice,  $c_{i}^{t} = 0$  means that client  $i$  does not take part in the  $t$ -th communication round. Here the set  $S_{t} \subseteq \{1, \dots, n\}$  is formed randomly such that  $\mathbf{Prob}(i \in S_{t}) = p_{i} > 0$  for all  $i = 1, \dots, n$ .  
$\diamond$  EF21-BC: Error feedback and bidirectional compression. The simplicity of the EF21 mechanism allows us to naturally extend it to the case when it is desirable to have efficient/compressed communication between the clients and the server in both directions. At each iteration of EF21-BC, clients compute and send to the master node  $c_{i}^{t} = \mathcal{C}_{w}(\nabla f_{i}(x^{t + 1}) - \widetilde{g}_{i}^{t})$  and update  $\widetilde{g}_i^{t + 1} = \widetilde{g}_i^t +c_i^t$  in the usual way, i.e., workers apply the EF21 mechanism. The key difference between EF21 and EF21-BC is that the master node in EF21-BC also uses this mechanism: it computes and broadcasts to the workers the compressed vector  $b^{t + 1} = \mathcal{C}_M(\widetilde{g}^{t + 1} - g^t)$  and updates  $g^{t + 1} = g^t +b^{t + 1}$ , where  $\widetilde{g}^{t + 1} = \frac{1}{n}\sum_{i = 1}^{n}\widetilde{g}_{i}^{t + 1}$ . Vector  $g^{t}$  is maintained by the master and workers. Therefore, the clients are able to update it via using  $g^{t + 1} = g^{t} + b^{t + 1}$  and compute  $x^{t + 1} = x^{t} - \gamma g^{t}$  once they receive  $b^{t + 1}$ .  
$\diamond$  EF21-HB: Error feedback with momentum. We consider classical Heavy-ball method (Polyak, 1964) with EF21 estimator  $g^t$ :

$$
x ^ {t + 1} = x ^ {t} - \gamma v ^ {t}, v ^ {t + 1} = \eta v ^ {t} + g ^ {t + 1}, g _ {i} ^ {t + 1} = g _ {i} ^ {t} + c _ {i} ^ {t}, g ^ {t + 1} = \frac {1}{n} \sum_ {i = 1} ^ {n} g _ {i} ^ {t + 1} = g ^ {t} + \frac {1}{n} \sum_ {i = 1} ^ {n} c _ {i} ^ {t}.
$$

The resulting method is not better than EF21 in terms of the complexity of finding  $\varepsilon$ -stationary point, i.e., momentum does not improve the theoretical convergence rate. Unfortunately, this is common

issue for a wide range of results for momentum methods Loizou & Richtárik (2020). However, it is important to theoretically analyze momentum-extensions such as EF21-HB due to their importance in practice and generalization behaviour.

$\diamond$  EF21-Prox: Error feedback for composite problems. Finally, we make EF21 applicable to the composite optimization problems (6) by simply taking the prox-operator from the right-hand side of the  $x^{t + 1}$  update rule (7):  $x^{t + 1} = \mathrm{prox}_{\gamma r}(x^t -\gamma g^t) = \arg \min_{x\in \mathbb{R}^d}\{\gamma r(x) + \| x - x^t +\gamma g^t\| ^2 /2\}$ . This trick is simple, but, surprisingly, EF21-Prox is the first distributed method with error-feedback that provably converges for composite problems (6).

# 4 THEORETICAL CONVERGENCE RESULTS

In this section, we formulate a single corollary derived from the main convergence theorems for our six enhancements of EF21, and formulate the assumptions that we use in the analysis. The complete statements of the theorems and their proofs are provided in the appendices. In Table 1 we compare our new results with existing results.

# 4.1 ASSUMPTIONS

In this subsection, we list and discuss the assumptions that we use in the analysis.

# 4.1.1 GENERAL ASSUMPTIONS

To derive our convergence results, we invoke the following standard smoothness assumption.

Assumption 1 (Smoothness and lower boundedness). Every  $f_{i}$  has  $L_{i}$ -Lipschitz gradient, i.e.,  $\| \nabla f_{i}(x) - \nabla f_{i}(y) \| \leq L_{i} \| x - y \|$  for all  $i \in [n], x, y \in \mathbb{R}^{d}$ , and  $f^{\inf} \stackrel{\text{def}}{=} \inf_{x \in \mathbb{R}^{d}} f(x) > -\infty$ .

We also assume that the compression operators used by all algorithms satisfy the following property.

Definition 1 (Contractive compressors). We say that a (possibly randomized) map  $\mathcal{C}:\mathbb{R}^d\to \mathbb{R}^d$  is a contractive compression operator, or simply contractive compressor, if there exists a constant  $0 < \alpha \leq 1$  such that

$$
\mathbb {E} \left[ \| \mathcal {C} (x) - x \| ^ {2} \right] \leq (1 - \alpha) \| x \| ^ {2}, \quad \forall x \in \mathbb {R} ^ {d}. \tag {8}
$$

We emphasize that we do not assume  $\mathcal{C}$  to be unbiased. Hence, our theory works with the Top- $k$  (Alistarh et al., 2018) and the Rank- $r$  (Safaryan et al., 2021) compressors, for example.

# 4.1.2 ADDITIONAL ASSUMPTIONS FOR EF21-SGD

We analyze EF21-SGD under the assumption that local stochastic gradients  $\nabla f_{\xi_{ij}^t}(x^t)$  satisfy the following inequality (see Assumption 2 of Khaled & Richtárik (2020)).

Assumption 2 (General assumption for stochastic gradients). We assume that for all  $i = 1, \dots, n$  there exist parameters  $A_i, C_i \geq 0, B_i \geq 1$  such that

$$
\mathbb {E} \left[ \| \nabla f _ {\xi_ {i j} ^ {t}} (x ^ {t}) \| ^ {2} \mid x ^ {t} \right] \leq 2 A _ {i} \left(f _ {i} (x ^ {t}) - f _ {i} ^ {\inf }\right) + B _ {i} \| \nabla f _ {i} (x ^ {t}) \| ^ {2} + C _ {i}, \tag {9}
$$

where  $f_{i}^{\inf} = \inf_{x\in \mathbb{R}^{d}}f_{i}(x) > - \infty$

Below we provide two examples of stochastic gradients fitting this assumption (for more detail, see (Khaled & Richtárik, 2020)).

Example 1. Consider  $\nabla f_{\xi_{ij}^t}(x^t)$  such that

$$
\mathbb {E} \left[ \nabla f _ {\xi_ {i j} ^ {t}} (x ^ {t}) \mid x ^ {t} \right] = \nabla f _ {i} (x ^ {t}) \quad a n d \quad \mathbb {E} \left[ \left\| \nabla f _ {\xi_ {i j} ^ {t}} (x ^ {t}) - \nabla f _ {i} (x ^ {t}) \right\| ^ {2} \mid x ^ {t} \right] \leq \sigma_ {i} ^ {2}
$$

for some  $\sigma_{i}\geq 0$ . Then, due to variance decomposition,(9) holds with  $A_{i} = 0$ ,  $B_{i} = 0$ ,  $C_i = \sigma_i^2$ .

Example 2. Let  $f_{i}(x) = \frac{1}{m_{i}}\sum_{j = 1}^{m_{i}}f_{ij}(x)$ ,  $f_{ij}$  be  $L_{ij}$ -smooth and  $f_{ij}^{\mathrm{inf}} = \inf_{x\in \mathbb{R}^d}f_{ij}(x) > -\infty$ . Following Gower et al. (2019), we consider a stochastic reformulation

$$
f _ {i} (x) = \mathbb {E} _ {v _ {i} \sim \mathcal {D} _ {i}} \left[ f _ {v _ {i}} (x) \right] = \mathbb {E} _ {v _ {i} \sim \mathcal {D} _ {i}} \left[ \frac {1}{m _ {i}} \sum_ {j = 1} ^ {m _ {i}} f _ {v _ {i j}} (x) \right], \tag {10}
$$

where  $\mathbb{E}_{v_i\sim \mathcal{D}_i}\left[v_{ij}\right] = 1$ . One can show (see Proposition 2 of Khaled & Richtárik (2020)) that under the assumption that  $\mathbb{E}_{v_i\sim \mathcal{D}_i}\left[v_{ij}^2\right]$  is finite for all  $j$  stochastic gradient  $\nabla f_{\xi_{ij}^t}(x^t) = \nabla f_{v_i^t}(x^t)$  with  $v_{i}^{t}$  sampled from  $\mathcal{D}_i$  satisfies (9) with  $A_{i} = \max_{j}L_{ij}\mathbb{E}_{v_{i}\sim \mathcal{D}_{i}}\left[v_{ij}^{2}\right]$ ,  $B_{i} = 1$ ,  $C_i = 2A_i\Delta_i^{\inf}$ , where  $\Delta_i^{\inf} = \frac{1}{m_i}\sum_{j = 1}^{m_i}(f_i^{\inf} - f_{ij}^{\inf})$ . In particular, if  $\mathbf{Prob}(\nabla f_{\xi_{ij}^t}(x^t) = \nabla f_{ij}(x^t)) = \frac{L_j}{\sum_{l = 1}^{m_l}L_{il}}$ , then  $A_{i} = \overline{L}_{i} = \frac{1}{m_{i}}\sum_{j = 1}^{m_{i}}L_{ij}$ ,  $B_{i} = 1$ , and  $C_i = 2A_i\Delta_i^{\inf}$ .

Stochastic gradient  $\hat{g}_i(x^t)$  is computed using a mini-batch of  $\tau_{i}$  independent samples satisfying (9):

$$
\hat {g} _ {i} (x ^ {t}) \stackrel {\mathrm {d e f}} {=} \frac {1}{\tau_ {i}} \sum_ {j = 1} ^ {\tau_ {i}} \nabla f _ {\xi_ {i j} ^ {t}} (x ^ {t}).
$$

# 4.1.3 ADDITIONAL ASSUMPTIONS FOR EF21-PAGE

In the analysis of EF21-PAGE, we rely on the following assumption.

Assumption 3 (Average  $\mathcal{L}$ -smoothness). Let every  $f_{i}$  have the form (3). Assume that for all  $t\geq 0$ ,  $i = 1,\ldots ,n$ , and batch  $I_i^t$  (of size  $\tau_{i}$ ), the minibatch stochastic gradients difference  $\widetilde{\Delta}_i^t\stackrel {\text{def}}{=}\frac{1}{\tau_i}\sum_{j\in I_i^t}(\nabla f_{ij}(x^{t + 1}) - \nabla f_{ij}(x^t))$  computed on the node  $i$ , satisfies  $\mathbb{E}\left[\widetilde{\Delta}_i^t\mid x^t,x^{t + 1}\right] = \Delta_i^t$  and

$$
\mathbb {E} \left[ \left\| \widetilde {\Delta} _ {i} ^ {t} - \Delta_ {i} ^ {t} \right\| ^ {2} \mid x ^ {t}, x ^ {t + 1} \right] \leq \frac {\mathcal {L} _ {i} ^ {2}}{\tau_ {i}} \| x ^ {t + 1} - x ^ {t} \| ^ {2} \tag {11}
$$

with some  $\mathcal{L}_i\geq 0$  , where  $\Delta_i^t\stackrel {def}{=}\nabla f_i(x^{t + 1}) - \nabla f_i(x^t)$  . We also define  $\widetilde{\mathcal{L}}\stackrel {def}{=}\frac{1}{n}\sum_{i = 1}^{n}\frac{(1 - p_i)\mathcal{L}_i^2}{\tau_i}$

This assumption is satisfied for many standard/popular sampling strategies. For example, if  $I_{i}^{t}$  is a full batch, then  $\mathcal{L}_i = 0$ . Another example is uniform sampling on  $\{1,\dots ,m\}$ , and each  $f_{ij}$  is  $L_{ij}$ -smooth. In this regime, one may verify that  $\mathcal{L}_i\leq \max_{1\leq j\leq m}L_{ij}$ .

# 4.2 MAIN RESULTS

Below we formulate the corollary establishing the complexities for each method. The complete version of this result is formulated and rigorously derived for each method in the appendix.

Corollary 1. Suppose that Assumption 1 holds. Then, there exist appropriate choices of parameters for EF21-PP, EF21-BC, EF21-HB, EF21-Prox such that the number of communication rounds  $T$  and the (expected) number of gradient computations at each node #grad for these methods to find an  $\varepsilon$ -stationary point, i.e., a point  $\hat{x}^T$  such that  $\mathbb{E}[\| \nabla f(\hat{x}^T)\|^2] \leq \varepsilon^2$  for EF21-PP, EF21-BC, EF21-HB and  $\mathbb{E}[\| \mathcal{G}_{\gamma}(\hat{x}^T)\|^2] \leq \varepsilon^2$  for EF21-Prox, where  $\mathcal{G}_{\gamma}(x) = 1 / \gamma$  ( $x - \mathrm{prox}_{\gamma r}(x - \gamma \nabla f(x))$ ), are

EF21-PP:  $T = \mathcal{O}\left(\frac{\widetilde{L}\delta^0}{p\alpha\varepsilon^2}\right)$  #grad = O (

EF21-BC:  $T = \# \operatorname{grad} = \mathcal{O}\left(\frac{\widetilde{L}\delta^{0}}{\alpha_{w}\alpha_{M}\varepsilon^{2}}\right)$

EF21-HB:  $T = \# \operatorname{grad} = \mathcal{O}\left(\frac{\widetilde{L}\delta^{0}}{\varepsilon^{2}}\left(\frac{1}{\alpha} + \frac{1}{1 - \eta}\right)\right)$

EF21-Prox:  $T = \# grad = \mathcal{O}\left(\frac{\widetilde{L}\delta^{0}}{\alpha\varepsilon^{2}}\right),$

where  $\widetilde{L} \stackrel{\text{def}}{=} \sqrt{\frac{1}{n} \sum_{i=1}^{n} L_i^2}$ ,  $\delta_0 \stackrel{\text{def}}{=} f(x^0) - f^{\inf}$  (for EF21-Prox  $\delta^0 = \Phi(x^0) - \Phi^{inf}$ ),  $p$  is the probability of sampling the client in EF21-PP,  $\alpha_w$  and  $\alpha_M$  are contraction factors for compressors applied on the workers' and the master's sides respectively in EF21-BC, and  $\eta \in [0,1)$  is the momentum parameter in EF21-HB.

If Assumptions 1 and 2 in the setup from Example 1 hold, then there exist appropriate choices of parameters for EF21-SGD such that the corresponding  $T$  and the averaged number of gradient computations at each node #grad are

$$
\mathrm {E F 2 1 - S G D :} \qquad T = \mathcal {O} \left(\frac {\widetilde {L} \delta^ {0}}{\alpha \varepsilon^ {2}}\right), \quad \overline {{\# g r a d}} = \mathcal {O} \left(\frac {\widetilde {L} \delta^ {0}}{\alpha \varepsilon^ {2}} + \frac {\widetilde {L} \delta^ {0} \sigma^ {2}}{\alpha^ {3} \varepsilon^ {4}}\right),
$$

where  $\sigma = \frac{1}{n}\sum_{i = 1}^{n}\sigma_{i}^{2}$

If Assumptions 1 and 3 hold, then there exist appropriate choices of parameters for EF21-PAGE such that the corresponding  $T$  and  $\# \mathrm{grad}$  are

$$
\text {E F 2 1 - P A G E :} \qquad T = \mathcal {O} \left(\frac {(\widetilde {L} + \widetilde {\mathcal {L}}) \delta^ {0}}{\alpha \varepsilon^ {2}} + \frac {\sqrt {m} \widetilde {\mathcal {L}} \delta^ {0}}{\varepsilon^ {2}}\right), \quad \overline {{\# g r a d}} = \mathcal {O} \left(m + \frac {(\widetilde {L} + \widetilde {\mathcal {L}}) \delta^ {0}}{\alpha \varepsilon^ {2}} + \frac {\sqrt {m} \widetilde {\mathcal {L}} \delta^ {0}}{\varepsilon^ {2}}\right),
$$

where  $\widetilde{\mathcal{L}} = \sqrt{\frac{1 - p}{n}\sum_{i = 1}^{n}\mathcal{L}_i^2},\tau_i\equiv \tau = 1.$

Remark: We highlight some points for our results in Corollary 1 as follows:

- For EF21-PP and EF21-Prox, none of previous error feedback methods work on these two settings (partial participation and proximal/composite case). Thus, we provide the first convergence results for them. Moreover, we show that the gradient (computation) complexity for both EF21-PP and EF21-Prox is  $\mathcal{O}(^{1} / \alpha \varepsilon)$ , matching the original vanilla EF21. It means that we extend EF21 to both settings for free.  
- For EF21-BC, we show  $\mathcal{O}\left(1 / \alpha_w\alpha_M\varepsilon^2\right)$  complexity result. In particular, if one uses constant ratio of compression (e.g.,  $10\%$ ), then  $\alpha \approx 0.1$ . Then the result will be  $\mathcal{O}\left(1 / \varepsilon^2\right)$ . However, previous result of DoubleSqueeze is  $\mathcal{O}\left(\Delta /\varepsilon^3\right)$  and it also uses more strict assumption for the compressors  $(\mathbb{E}\left[\| \mathcal{C}(x) - x\| \right]\leq \Delta)$ . Even if we ignore this, our results for EF21-BC is better than the one for DoubleSqueeze by a large factor  $1 / \varepsilon$ .  
- Similarly, our result for EF21-HB is roughly  $\mathcal{O}(1 / \varepsilon^2)$  (note that the momentum parameter  $\eta$  is usually constant such as 0.2, 0.4, 0.9 used in our experiments). However, previous results of M-CSER are roughly  $\mathcal{O}(G / \varepsilon^3)$  and it is proven under an additional bounded gradient assumption. Similarly, our EF21-HB is better by a large factor  $1 / \varepsilon$ .  
- For EF21-SGD and EF21-PAGE, we want to reduce the gradient complexity by using (variance-reduced) stochastic gradients instead of full gradient in the vanilla EF21. Note that  $\sigma^2$  and  $\Delta^{\mathrm{inf}}$  in EF21-SGD could be much smaller than  $G$  in Choco-SGD since  $G$  always depends on the dimension (and can be even infinite), while  $\sigma^2$  and  $\Delta^{\mathrm{inf}}$  are mostly dimension-free parameters (particularly, they are very small if the functions/data samples are similar/close). Thus, for high dimensional problems (e.g., deep neural networks), EF21-SGD can be better than Choco-SGD. Besides, in the finite-sum case (3), especially if the number of data samples  $m$  on each client is not very large, then EF21-PAGE is much better since its complexity is roughly  $\mathcal{O}(\sqrt{m} /\varepsilon^2)$  while EF21-SGD ones is roughly  $\mathcal{O}(\sigma^2 /\varepsilon^4)$ .

# 5 EXPERIMENTS

In this section, we consider a logistic regression problem with a non-convex regularizer

$$
\min  _ {x \in \mathbb {R} ^ {d}} \left\{f (x) = \frac {1}{N} \sum_ {i = 1} ^ {N} \log \left(1 + \exp \left(- b _ {i} a _ {i} ^ {\top} x\right)\right) + \lambda \sum_ {j = 1} ^ {d} \frac {x _ {j} ^ {2}}{1 + x _ {j} ^ {2}} \right\}, \tag {12}
$$

where  $a_{i}\in \mathbb{R}^{d},b_{i}\in \{-1,1\}$  are the training data, and  $\lambda >0$  is the regularization parameter, which is set to  $\lambda = 0.1$  in all experiments. For all methods the stepsizes are initially chosen as the largest stepsize predicted by theory for EF21 (see Theorem 1), then they are tuned individually for each parameter setting. We provide more details on the datasets, hardware, experimental setups, and additional experiments, including deep learning experiments in Appendix A.

Experiment 1: Fast convergence with variance reduction. In our first experiment, we showcase the computation and communication superiority of EF21-PAGE (Alg. 3) over EF21-SGD.

Figure 1 illustrates that, in all cases, EF21-PAGE perfectly reduces the accumulated variance and converges to the desired tolerance, whereas EF21-SGD is stuck at some accuracy level. Moreover,

EF21-PAGE turns out to be surprisingly efficient with small bathsizes (eg,  $1.5\%$  of the local data) both in terms of the number of epochs and the # bits sent to the server per client. Interestingly, for most datasets, a further increase of bathsize does not considerably improve the convergence.

![](images/40e1b47fe74f8c81de45df4f82aea15d084ff3d1238737592886b4b0f1947ffc.jpg)

![](images/928321ab92a79888545e203ce7450960892f64ea2c48fc09e247e36ef55822cb.jpg)

![](images/98eff6ace78692a4e0ed532f3415af24f1c75027e64630ca0e53325e80f65754.jpg)

![](images/0adfb8cddf6eeebdc798c57ee27ded0c77f17538c586cc6b64f57288c3354ad4.jpg)

![](images/cf713d314f1e714c4602c6ec5448b6bdd051a34edc0e19b5fead098e6df390ad.jpg)  
(b) Convergence in terms of total number of bits sent from Clients to the Server divided by  $n$ .

![](images/3302276986bdb550f553d6b41f7a90711fcdc46293021c335e3c698c7d480e9b.jpg)  
(a) Convergence in epochs.

![](images/f8be3aa3180a25009ee82556869a541fe6f60f1d72fa658189c55d05191c58a8.jpg)

![](images/273061df4ce2b06f3ff0f864ec9426e42be045c922dcf4d617fb3fe73553af0c.jpg)

Figure 1: Comparison of EF21-PAGE and EF21-SGD with tuned parameters. By  $1 \times, 2 \times, 4 \times$  (and so on) we indicate that the stepsize was set to a multiple of the largest stepsize predicted by theory for EF21. By  $25\%$ ,  $12.5\%$  and  $1.5\%$  we refer to batchesizes equal  $\lfloor 0.25N_i \rfloor$ ,  $\lfloor 0.125N_i \rfloor$  and  $\lfloor 0.015N_i \rfloor$  for all clients  $i = 1, \dots, n$ , where  $N_i$  denotes the size of local dataset.

Experiment 2: On the effect of partial participation of clients. This experiment shows that EF21-PP (Alg. 4) can reduce communication costs and can be more practical than EF21. For this comparison, we consider  $n = 100$  and, therefore, apply a different data partitioning, see Table 5 from Appendix A for more details.

It is predicted by our theory (Corollary 1) that, in terms of the number of iterations/communication rounds, partial participation slows down the convergence of EF21 by a fraction of participating clients. We observe this behavior in practice as well (see Figure 2a). However, since for EF21-PP the communications are considerably cheaper it outperforms EF21 in terms of # number of bits sent to the server per client on average (see Figure 2).

![](images/118584ad2cff8ffb0d9fe707d33869f9a8397dc7888fde0203a81347860dd33d.jpg)

![](images/2011cd0fde420ab49f94ab8a373ad87e41309afc6274f7b214c27ae740c88d4b.jpg)

![](images/1ca0350f412730a781f23380b9c69b150b8f21c5ea26fff74795e5684542243a.jpg)

![](images/0888b2f7b34a67172b25018c9b895345417010810e5a9aa1498dbfad5bafcdb9.jpg)

![](images/e457d52724519575db9e7750959bfe465c477867e708e06fb340160bf22c81fb.jpg)  
(b) Convergence in terms of total number of bits sent from Clients to the Server divided by  $n$ .

![](images/e58afa6b6429728e8466de6fb7c52d6c82ad629fe6204029642ad9e21607a372.jpg)  
(a) Convergence in communication rounds.

![](images/b9c3e0ab850b2884740bdaed6f6b493a29f65006f3e65d58d861017033be351c.jpg)

![](images/34113712109eccab5faef8d806b64c57628f2c3e7134f1c4eea87326fc4ee226.jpg)

Figure 2: Comparison of EF21-PP and EF21 with tuned parameters. By  $1 \times, 2 \times, 4 \times$  (and so on) we indicate that the stepsize was set to a multiple of the largest stepsize predicted by theory for EF21. By  $50\%$ ,  $25\%$ ,  $12.5\%$ , and  $6.5\%$  we refer to a number of participating clients equal to  $\lfloor 0.5n \rfloor$ ,  $\lfloor 0.25n \rfloor$ ,  $\lfloor 0.125n \rfloor$  and  $\lfloor 0.065n \rfloor$ .

# REFERENCES

Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. QSGD: Communication-efficient SGD via gradient quantization and encoding. In Advances in Neural Information Processing Systems (NIPS), pp. 1709-1720, 2017.  
Dan Alistarh, Torsten Hoefler, Mikael Johansson, Sarit Khirirat, Nikola Konstantinov, and Cédric Renggli. The convergence of sparsified gradient methods. In Advances in Neural Information Processing Systems (NeurIPS), 2018.  
Zeyuan Allen-Zhu. Katyusha: The first direct acceleration of stochastic gradient methods. In Proceedings of the 49th Annual ACM SIGACT Symposium on Theory of Computing, pp. 1200-1205. ACM, 2017.  
Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Nathan Srebro, and Blake Woodworth. Lower bounds for non-convex stochastic optimization. arXiv preprint arXiv:1912.02365, 2019.  
Sanjeev Arora, Nadav Cohen, and Elad Hazan. On the optimization of deep networks: Implicit acceleration by overparameterization. In Proceedings of the 35th International Conference on Machine Learning (ICML), 2018.  
Amir Beck. First-Order Methods in Optimization. Society for Industrial and Applied Mathematics, 2017.  
Aleksandr Beznosikov, Samuel Horváth, Peter Richtárik, and Mher Safaryan. On biased compression for distributed learning. arXiv preprint arXiv:2002.12410, 2020.  
Léon Bottou. Curiously fast convergence of some stochastic gradient descent algorithms. In Proceedings of the symposium on learning and data science, Paris, volume 8, pp. 2624-2633, 2009.  
Léon Bottou. Stochastic gradient descent tricks. In Neural networks: Tricks of the trade, pp. 421-436. Springer, 2012.  
Chih-Chung Chang and Chih-Jen Lin. LIBSVM: a library for support vector machines. ACM Transactions on Intelligent Systems and Technology (TIST), 2(3):1-27, 2011.  
Yae Jee Cho, Jianyu Wang, and Gauri Joshi. Client selection in federated learning: Convergence analysis and power-of-choice selection strategies. arXiv preprint arXiv:2010.01243v1, 2020.  
Eduard Gorbunov, Dmitry Kovalev, Dmitry Makarenko, and Peter Richtárik. Linearly converging error compensated SGD. In 34th Conference on Neural Information Processing Systems (NeurIPS), 2020.  
Eduard Gorbunov, Konstantin Burlachenko, Zhize Li, and Peter Richtárik. MARINA: Faster nonconvex distributed learning with compression. In International Conference on Machine Learning, pp. 3788-3798. PMLR, 2021. arXiv:2102.07845.  
Robert M Gower, Mark Schmidt, Francis Bach, and Peter Richtárik. Variance-reduced methods for machine learning. Proceedings of the IEEE, 108(11):1968-1983, 2020.  
Robert Mansel Gower, Nicolas Loizou, Xun Qian, Alibek Sailanbayev, Egor Shulgin, and Peter Richtárik. SGD: General analysis and improved rates. In International Conference on Machine Learning, pp. 5200-5209. PMLR, 2019.  
Priya Goyal, Piotr Dólar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.

Samuel Horváth and Peter Richtárik. A better alternative to error feedback for communication-efficient distributed learning. In 9th International Conference on Learning Representations (ICLR), 2021.  
Samuel Horváth, Chen-Yu Ho, L'udovít Horváth, Atal Narayan Sahu, Marco Canini, and Peter Richtárik. Natural compression for distributed deep learning. arXiv preprint arXiv:1905.10988, 2019a.  
Samuel Horváth, Dmitry Kovalev, Konstantin Mishchenko, Sebastian Stich, and Peter Richtárik. Stochastic distributed learning with gradient quantization and variance reduction. arXiv preprint arXiv:1904.05115, 2019b.  
Rustem Islamov, Xun Qian, and Peter Richtárik. Distributed second order methods with fast rates and compressed communication. arXiv preprint arXiv:2102.07158, 2021.  
Peter et al Kairouz. Advances and open problems in federated learning. arXiv preprint arXiv:1912.04977, 2019.  
Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Stich, and Martin Jaggi. Error feedback fixes SignSGD and other gradient compression schemes. In 36th International Conference on Machine Learning (ICML), 2019.  
Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. SCAFFOLD: Stochastic controlled averaging for federated learning. In Proceedings of the 37th International Conference on Machine Learning, 2020.  
Ahmed Khaled and Peter Richtárik. Gradient descent with compressed iterates. In NeurIPS Workshop on Federated Learning for Data Privacy and Confidentiality, 2019.  
Ahmed Khaled and Peter Richtárik. Better theory for SGD in the nonconvex world. arXiv preprint arXiv:2002.03329, 2020.  
Sarit Khirirat, Hamid Reza Feyzmahdavian, and Mikael Johansson. Distributed learning with compressed gradients. arXiv preprint arXiv:1806.06573, 2018.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Anastasia Koloskova, Tao Lin, S. Stich, and Martin Jaggi. Decentralized deep learning with arbitrary communication compression. In International Conference on Learning Representations (ICLR), 2020.  
Jakub Konečný, H. Brendan McMahan, Felix Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave Bacon. Federated learning: strategies for improving communication efficiency. In NIPS Private Multi-Party Machine Learning Workshop, 2016.  
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical report, University of Toronto, Toronto, 2009.  
Guanghui Lan and Yi Zhou. An optimal randomized incremental gradient method. arXiv preprint arXiv:1507.02000, 2015.  
Guanghui Lan, Zhize Li, and Yi Zhou. A unified variance-reduced accelerated gradient method for convex optimization. In Advances in Neural Information Processing Systems, pp. 10462-10472, 2019.  
Zhize Li. ANITA: An optimal loopless accelerated variance-reduced gradient method. arXiv preprint arXiv:2103.11333, 2021a.  
Zhize Li. A short note of page: Optimal convergence rates for nonconvex optimization. arXiv preprint arXiv:2106.09663, 2021b.  
Zhize Li and Jian Li. A simple proximal stochastic gradient method for nonsmooth nonconvex optimization. In Advances in Neural Information Processing Systems (NeurIPS), pp. 5569-5579, 2018.

Zhize Li and Peter Richtárik. A unified analysis of stochastic gradient methods for nonconvex federated optimization. arXiv preprint arXiv:2006.07013, 2020.  
Zhize Li and Peter Richtárik. CANITA: Faster rates for distributed convex optimization with communication compression. arXiv preprint arXiv:2107.09461, 2021a.  
Zhize Li and Peter Rictarik. ZeroSARAH: Efficient nonconvex finite-sum optimization with zero full gradient computation. arXiv preprint arXiv:2103.01447, 2021b.  
Zhize Li, Dmitry Kovalev, Xun Qian, and Peter Richtárik. Acceleration for compressed gradient descent in distributed and federated optimization. In International Conference on Machine Learning (ICML), pp. 5895-5904. PMLR, 2020.  
Zhize Li, Hongyan Bao, Xiangliang Zhang, and Peter Richtárik. PAGE: A simple and optimal probabilistic gradient estimator for nonconvex optimization. In International Conference on Machine Learning (ICML), pp. 6286-6295. PMLR, 2021. arXiv:2008.10898.  
Nicolas Loizou and Peter Richtárik. Momentum and stochastic momentum for stochastic gradient, Newton, proximal point and subspace descent methods. Computational Optimization and Applications, 77:653-710, 2020.  
Stanislaw Lojasiewicz. A topological property of real analytic subsets. Coll. du CNRS, Les équations aux dérivées partielles, 117(87-89):2, 1963.  
Konstantin Mishchenko, Eduard Gorbunov, Martin Takáč, and Peter Richtárik. Distributed learning with compressed gradient differences. arXiv preprint arXiv:1901.09269, 2019.  
Konstantin Mishchenko, Ahmed Khaled, and Peter Richtarik. Random reshuffling: Simple analysis with vast improvements. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 17309-17320. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/c8cc6e90ccbff44c9cee23611711cdc4-[]Paper.pdf.  
Yurii Nesterov. A method for unconstrained convex minimization problem with the rate of convergence o  $(1 / \mathbf{k}^{\wedge}2)$ . In Doklady AN USSR, volume 269, pp. 543-547, 1983.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (NeurIPS), 2019.  
Constantin Philippenko and Aymeric Dieuleveut. Bidirectional compression in heterogeneous settings for distributed or federated learning with partial participation: tight convergence guarantees. arXiv preprint arXiv:2006.14591, 2020.  
Boris T Polyak. Gradient methods for the minimisation of functionals. USSR Computational Mathematics and Mathematical Physics, 3(4):864-878, 1963.  
Boris T Polyak. Some methods of speeding up the convergence of iteration methods. Ussr computational mathematics and mathematical physics, 4(5):1-17, 1964.  
Xun Qian, Peter Richtárik, and Tong Zhang. Error compensated distributed SGD can be accelerated. arXiv preprint arXiv:2010.00091, 2020.  
Zheng Qu and Peter Richtárik. Coordinate descent with arbitrary sampling ii: Expected separable overapproximation. arXiv preprint arXiv:1412.8063, 2014.  
Peter Richtárik, Igor Sokolov, and Ilyas Fatkhullin. EF21: A new, simpler, theoretically better, and practically faster error feedback. arXiv preprint arXiv:2106.05203, 2021.  
Mher Safaryan, Rustem Islamov, Xun Qian, and Peter Richtárik. FedNL: Making Newton-type methods applicable to federated learning. arXiv preprint arXiv:2106.02969, 2021.

Frank Seide, Hao Fu, Jasha Droppo, Gang Li, and Dong Yu. 1-bit stochastic gradient descent and its application to data-parallel distributed training of speech DNNs. In Fifteenth Annual Conference of the International Speech Communication Association, 2014.  
Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: from theory to algorithms. Cambridge University Press, 2014.  
Sebastian U. Stich, J.-B. Cordonnier, and Martin Jaggi. Sparsified SGD with memory. In Advances in Neural Information Processing Systems (NeurIPS), 2018.  
Hanlin Tang, Xiangru Lian, Chen Yu, Tong Zhang, and Ji Liu. DoubleSqueeze: Parallel stochastic gradient descent with double-pass error-compensated compression. In Proceedings of the 36th International Conference on Machine Learning (ICML), 2020.  
Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. PowerSGD: Practical low-rank gradient compression for distributed optimization. In Neural Information Processing Systems, 2019.  
Zhe Wang, Kaiyi Ji, Yi Zhou, Yingbin Liang, and Vahid Tarokh. Spiderboost and momentum: Faster stochastic variance reduction algorithms. arXiv preprint arXiv:1810.10690, 2018.  
Cong Xie, Shuai Zheng, Oluwasanmi Koyejo, Indranil Gupta, Mu Li, and Haibin Lin. CSER: Communication-efficient SGD with error reset. In Advances in Neural Information Processing Systems (NeurIPS), pp. 12593-12603, 2020.  
Haibo Yang, Minghong Fang, and Jia Liu. Achieving linear speedup with partial worker participation in non-iid federated learning. arXiv preprint arXiv:2101.11203v3, 2021.  
Tianbao Yang, Qihang Lin, and Zhe Li. Unified convergence analysis of stochastic momentum methods for convex and non-convex optimization. arXiv preprint arXiv:1604.03257, 2016.  
Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=Syx4wnEtvH.  
Haoyu Zhao, Zhize Li, and Peter Richtárik. FedPAGE: A fast local stochastic gradient method for communication-efficient federated learning. arXiv preprint arXiv:2108.04755, 2021.
