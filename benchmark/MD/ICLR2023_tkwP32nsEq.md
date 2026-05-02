# VARIANCE-AWARE SPARSE LINEAR BANDITS

Anonymous authors

Paper under double-blind review

# ABSTRACT

It is well-known that for sparse linear bandits, when ignoring the dependency on sparsity which is much smaller than the ambient dimension, the worst-case minimax regret is  $\widetilde{\Theta}\left(\sqrt{dT}\right)$  where  $d$  is the ambient dimension and  $T$  is the number of rounds. On the other hand, in the benign setting where there is no noise and the action set is the unit sphere, one can use divide-and-conquer to achieve  $\widetilde{\mathcal{O}}(1)$  regret, which is (nearly) independent of  $d$  and  $T$ . In this paper, we present the first variance-aware regret guarantee for sparse linear bandits:  $\widetilde{\mathcal{O}}\left(\sqrt{d\sum_{t=1}^{T}\sigma_t^2} + 1\right)$ ,

where  $\sigma_t^2$  is the variance of the noise at the  $t$ -th round. This bound naturally interpolates the regret bounds for the worst-case constant-variance regime (i.e.,  $\sigma_t \equiv \Omega(1)$ ) and the benign deterministic regimes (i.e.,  $\sigma_t \equiv 0$ ). To achieve this variance-aware regret guarantee, we develop a general framework that converts any variance-aware linear bandit algorithm to a variance-aware algorithm for sparse linear bandits in a "black-box" manner. Specifically, we take two recent algorithms as black boxes to illustrate that the claimed bounds indeed hold, where the first algorithm can handle unknown-variance cases and the second one is more efficient.

# 1 INTRODUCTION

This paper studies the sparse linear stochastic bandit problem, which is a special case of linear stochastic bandits. In linear bandits (Dani et al., 2008), the agent is facing a sequential decision-making problem lasting for  $T$  rounds. For the  $t$ -th round, the agent chooses an action  $x_{t} \in \mathcal{X} \subseteq \mathbb{R}^{d}$  where  $\mathcal{X}$  is an action set, and receives a noisy reward  $r_{t} = \langle \theta^{*}, x_{t} \rangle + \eta_{t}$  where  $\theta^{*} \in \mathcal{X}$  is the (hidden) parameter of the game and  $\eta_{t}$  is random zero-mean noise. The goal of the agent is to minimize her regret  $\mathcal{R}_{T}$ , that is, the difference between her cumulative reward  $\sum_{t=1}^{T} \langle \theta^{*}, x_{t} \rangle$  and  $\max_{x \in \mathcal{X}} \sum_{t=1}^{T} \langle \theta^{*}, x \rangle$  (check Eq. (1) for a definition). Dani et al. (2008) proved that the minimax optimal regret for linear bandits is  $\widetilde{\Theta}(d\sqrt{T})$  when the noises are independent Gaussian random variables with means 0 and variances 1 and both  $\theta^{*}$  and the actions  $x_{t}$  lie in the unit sphere in  $\mathbb{R}^{d}$ .<sup>1</sup>

In real-world applications such as recommendation systems, only a few features may be relevant despite a large candidate feature space. In other words, the high-dimensional linear regime may actually allow a low-dimensional structure. As a result, if we still use the linear bandit model, we will always suffer  $\Omega(d\sqrt{T})$  regret no matter how many features are useful. Motivated by this, the sparse linear stochastic bandit problem was introduced (Abbasi-Yadkori et al., 2012; Carpentier & Munos, 2012). This problem has an additional constraint that the hidden parameter,  $\theta^{*}$ , is sparse, i.e.,  $\| \theta^{*} \|_{0} \leq s$  for some  $s \ll d$ . However, the agent has no prior knowledge about  $s$  and thus the interaction protocol is exactly the same as that of linear bandits. The minimax optimal regret for sparse linear bandits is  $\widetilde{\Theta}(\sqrt{sdT})$  (Abbasi-Yadkori et al., 2012; Antos & Szepesvári, 2009). This bound bypasses the  $\Omega(d\sqrt{T})$  lower bound for linear bandits as we always have  $s = \| \theta^{*} \|_{0} \leq d$  and the agent does not have access to  $s$  either (though a few previous works assumed a known  $s$ ).

However, both the  $\widetilde{\mathcal{O}}(d\sqrt{T})$  and the  $\widetilde{\mathcal{O}}(\sqrt{sdT})$  bounds are the worst-case regret bounds and sometime are too pessimistic especially when  $d$  is large. On the other hand, many problems with delicate structures permit a regret bound much smaller than the worst-case bound. The structure this paper focuses on is the magnitude of the noise. Consider the following motivating example.

Motivating Example (Deterministic Sparse Linear Bandits). Consider the case where the action set is the unit sphere  $\mathcal{X} = \mathbb{S}^{d - 1}$ , and there is no noise, i.e., the feedback is  $r_t = \langle \theta^*, x_t \rangle$  for each round  $t \in [T]$ . In this case, one can identify all non-zero entries of  $\theta^*$  coordinates in  $\mathcal{O}(s \log d)$  steps with high probability via a divide-and-conquer algorithm, and thus yield a dimension-free regret  $\tilde{\mathcal{O}}(s)$  (see Appendix B for more details about this).<sup>3</sup> However, this divide-and-conquer algorithm is specific for deterministic sparse linear bandit problems and does not work for noisy models. Henceforth, we study the following natural question:

Can we design an algorithm whose regret adapts to the noise level such that the regret interpolates the  $\sqrt{dT}$ -type bound in the worst case and the dimension-free bound in the deterministic case?

Before introducing our results, we would like to mention that there are recent works that studied the noise-adaptivity in linear bandits (Zhou et al., 2021; Zhang et al., 2021; Kim et al., 2021). They gave variance-aware regret bounds of the form  $\widetilde{\mathcal{O}}\left(\mathrm{poly}(d)\sqrt{\sum_{t=1}^{T}\sigma_t^2} + \mathrm{poly}(d)\right)$  where  $\sigma_t^2$  is the (conditional) variance of the noise  $\eta_t$ . This bound reduces to the standard  $\widetilde{\mathcal{O}}(\mathrm{poly}(d)\sqrt{T})$  bound in the worst-case when  $\sigma_t = \Omega(1)$ , and to a constant-type regret  $\widetilde{\mathcal{O}}(\mathrm{poly}(d))$  that is independent of  $T$ . However, compared with the linear bandits setting, the variance-aware bound for sparse linear bandits is more significant because it reduces to a dimension-free bound in the noiseless setting. Despite this, to our knowledge, no variance-aware regret bounds exist for sparse linear bandits.

# 1.1 OUR CONTRIBUTIONS

This paper gives the first set of variance-aware regret bounds for sparse linear bandits. We design a general framework, VASLB, to reduce variance-aware sparse linear bandits to variance-aware linear bandits with little overhead in regret. For ease of presentation, we define the following notation to characterize the variance-awareness of a sparse linear bandit algorithm:

Definition 1. A variance-aware sparse linear bandit algorithm  $\mathcal{F}$  is  $(f(s,d),g(s,d))$ -variance-aware, if for any given failure probability  $\delta >0$ , with probability  $1 - \delta$ ,  $\mathcal{F}$  ensures

$$
\mathcal {R} _ {T} ^ {\mathcal {F}} \leq \widetilde {\mathcal {O}} \left(f (s, d) \sqrt {\sum_ {t = 1} ^ {T} \sigma_ {t} ^ {2}} \operatorname {p o l y l o g} \frac {1}{\delta} + g (s, d) \operatorname {p o l y l o g} \frac {1}{\delta}\right),
$$

where  $\mathcal{R}_T^{\mathcal{F}}$  is the regret of  $\mathcal{F}$  in  $T$  rounds,  $d$  is the ambient dimension and  $s$  is the maximum number of non-zero coordinates. Specifically, for linear bandits,  $f, g$  are functions only of  $d$ .

Hence, an  $(f,g)$ -variance-aware algorithm will achieve  $\widetilde{\mathcal{O}}(f(s,d)\sqrt{T}\mathrm{polylog}\frac{1}{\delta})$  worst-case regret and  $\widetilde{\mathcal{O}}(g(s,d)\mathrm{polylog}\frac{1}{\delta})$  deterministic-case regret. Ideally, we would like  $g(s,d)$  being independent of  $d$ , making the bound dimension-free in deterministic cases, as the divide-and-conquer approach.

In this paper, we provide a general framework that can convert any linear bandit algorithm  $\mathcal{F}$  to a corresponding sparse linear bandit algorithm  $\mathcal{G}$  in a black-box manner. Moreover, it is variance-aware-preserving, in the sense that, if  $\mathcal{F}$  enjoys the variance-aware property, so does  $\mathcal{G}$ . Generally speaking, if the plug-in linear bandit algorithm  $\mathcal{F}$  is  $(f(d), g(d))$ -variance-aware, then our framework directly gives an  $(s(f(s) + \sqrt{d}), s(g(s) + 1))$ -variance-aware algorithm  $\mathcal{G}$  for sparse linear bandits.

Besides presenting our framework, we also illustrate its usefulness by plugging in two existing variance-aware linear bandit algorithms, where the first one is variance-aware (i.e., works in unknown-variance cases) but computationally inefficient. In contrast, the second one is efficient but requires the variance  $\sigma_t^2$  to be delivered together with feedback  $r_t$ . Their regret guarantees are stated as follows.

Table 1: An overview of the proposed algorithms/results and comparisons with related works.  

<table><tr><td>Algorithm</td><td>Setting</td><td>Worst-case Regreta</td><td>Deterministic-case Regretb</td><td>Efficiency</td><td>Variances</td></tr><tr><td>ConfidenceBall2 (Dani et al., 2008)</td><td>LinBandit</td><td>O(d√T)</td><td rowspan="4">N/A</td><td>✓</td><td rowspan="4">N/A</td></tr><tr><td>OFUL (Abbasi-Yadkori et al., 2012)</td><td>Sparse LinBandit</td><td>O(√sdT)</td><td>✓</td></tr><tr><td>SL-UCB (Carpentier &amp; Munos, 2012)</td><td>Sparse LinBandit</td><td>O(s√T)c</td><td>✓</td></tr><tr><td>Lattimore et al. (2015, Algorithm 4)</td><td>Sparse LinBandit</td><td>O(s√T)d</td><td>✓</td></tr><tr><td>Weighted OFUL (Zhou et al., 2021)</td><td>LinBandit</td><td>O(d√T)</td><td>O(√dT)</td><td>✓</td><td>Known</td></tr><tr><td>VOFUL2 (Kim et al., 2021)</td><td>LinBandit</td><td>O(d1.5√T)</td><td>O(d2)</td><td>✗</td><td>Unknown</td></tr><tr><td rowspan="2">VASLB (This work)</td><td>Sparse LinBandit</td><td>O(s2√T + s√dT)</td><td>O(s1.5√T)</td><td>✓</td><td>Known</td></tr><tr><td>Sparse LinBandit</td><td>O(s2.5√T + s√dT)</td><td>O(s3)</td><td>✗</td><td>Unknown</td></tr><tr><td>Lower Bound (Antos &amp; Szeptsvári, 2009)</td><td>Sparse LinBandit</td><td>Ω(√dT)e</td><td>N/A</td><td>N/A</td><td>N/A</td></tr></table>

${}^{a}$  "Worst-case" means the variances  ${\sigma }_{t}^{2}$  are all 1 . Here,  $d$  is the ambient dimension,  $T$  is the number of rounds, and  $s$  is the sparsity parameter (only applicable to sparse linear bandits).  
${}^{b}$  "Deterministic-case" means the variances  ${\sigma }_{t}^{2}$  are all 0. Only applicable to variance-aware algorithms.  
cWith a different feedback model; see Appendix A for more comparison.  
$^d$ With a different action set and an different assumption on  $\theta^*$ ; see Appendix A for more comparison.  
${}^{e}$  This bound holds even if  $s = 1$  and the action set is fixed to be the unit sphere.

1. The first variance-aware linear bandit algorithm we plug in is VOFUL, which was proposed by Zhang et al. (2021) and improved by Kim et al. (2021). This algorithm is computationally inefficient but deals with unknown variances. Using this VOFUL, our framework generates a  $(s^{2.5} + s\sqrt{d}, s^3)$ -variance-aware algorithm for sparse linear bandits. Compared to the  $\Omega(\sqrt{sdT})$  regret lower-bound for sparse linear bandits (Lattimore & Szepesvári, 2020, §24.3), our worst-case regret bound is near-optimal up to a factor  $\sqrt{s}$ . Moreover, our bound is independent of  $d$  and  $T$  in the deterministic case, nearly matching the bound of divide-and-conquer algorithm dedicated to the deterministic setting up to  $\mathrm{poly}(s)$  factors.  
2. The second algorithm we plug in is Weighted OFUL (Zhou et al., 2021), which requires known variances but is computationally efficient. We obtain an  $(s^2 + s\sqrt{d}, s^{1.5}\sqrt{T})$ -variance-aware efficient algorithm. In the deterministic case, this algorithm can only achieve a  $\sqrt{T}$ -type regret bound (albeit still independent of  $d$ ). We note that this is not due to our framework but due to Weighted OFUL which itself cannot give constant regret bound in the deterministic setting.

Finally, we would like to remark that the state-of-the-art variance-aware linear bandit algorithm (Kim et al., 2021) does not match the regret lower bound by a gap of  $\sqrt{d}$ . If better variance-aware linear bandit algorithms are derived in the future, it will also give a better bound for sparse ones by VASLB.

# 1.2 RELATED WORK

Linear Bandits. This problem was first introduced by Dani et al. (2008), where an algorithm with regret  $\mathcal{O}(d\sqrt{T} (\log T)^{3 / 2})$  and a near-matching regret lower-bound  $\Omega (d\sqrt{T})$  were given. After that, an improved upper bound  $\mathcal{O}(d\sqrt{T}\log T)$  (Abbasi-Yadkori et al., 2011) together with an improved lower bound  $\Omega (d\sqrt{T}\log T)$  (Li et al., 2019) were derived. An extension of it, namely linear contextual bandits, where the action set allowed for each step can vary with time (Chu et al., 2011; Kannan et al., 2018; Li et al., 2019; 2021), is receiving more and more attention. The best-arm identification problem where the goal of the agent is to approximate  $\theta^{*}$  with as few samples as possible (Soare et al., 2014; Degenne et al., 2019; Jedra & Proutiere, 2020; Alieva et al., 2021) is also of great interest.

Sparse Linear Bandits. Abbasi-Yadkori et al. (2011) and Carpentier & Munos (2012) concurrently considered the sparse linear bandit problem, where the former work assumed a noise model of  $r_t = \langle x_t, \theta^* \rangle + \eta_t$  such that  $\eta_t$  is  $R$ -sub-Gaussian and achieved  $\widetilde{\mathcal{O}}(R\sqrt{sdT})$  regret, while the latter one considered the noise model of  $r_t = \langle x_t + \eta_t, \theta^* \rangle$  such that  $\|\eta_t\|_2 \leq \sigma$  and  $\|\theta^*\|_2 \leq \theta$ , achieving  $\widetilde{\mathcal{O}}((\sigma + \theta)^2 s\sqrt{T})$  regret. Lattimore et al. (2015) assumed an hypercube (i.e.,  $\mathcal{X} = [-1, 1]^d$ ) action set and a  $\|\theta^*\|_1 \leq 1$  ground-truth, yielding  $\widetilde{\mathcal{O}}(s\sqrt{T})$  regret. Antos & Szepesvári (2009) proved a  $\Omega(\sqrt{dT})$  lower-bound when  $s = 1$  with the unit sphere as  $\mathcal{X}$ . Some recent works considered data-poor regimes where  $d \gg T$  (Hao et al., 2020; 2021a,b; Wang et al., 2020), which is beyond the

scope of this paper. Another work worth mentioning is the recent work by Dong et al. (2021), which studies bandits or MDPs with deterministic rewards. Their result implies an  $\widetilde{\mathcal{O}}(T^{15/16}s^{1/16})$  bound for deterministic sparse linear bandits, which is independent of  $d$ . They also provided an ad-hoc divide-and-conquer algorithm, which achieves  $\mathcal{O}(s\log d)$  regret only for deterministic cases.

Variance-Aware Online Learning. For tabular MDPs, the variance information is widely used in both discounted settings (Lattimore & Hutter, 2012) and episodic settings (Azar et al., 2017; Jin et al., 2018), where Zanette & Brunskill (2019) used variance information to derive problem-dependent regret bounds for tabular MDPs. For bandits, Audibert et al. (2009) made use of variance information in multi-armed bandits, giving an algorithm outperforming existing ones when the variances for suboptimal arms are relatively small. For bandits with high-dimensional structures, Faury et al. (2020) studied variance adaptation for logistic bandits, Zhou et al. (2021) considered linear bandits and linear mixture MDPs where the variance information is revealed to the agent, giving an  $\widetilde{\mathcal{O}}(d\sqrt{\sum_{t=1}^{T}\sigma_t^2} + \sqrt{dT})$  guarantee for linear bandits, and Zhang et al. (2021) proposed another algorithm for linear bandits and linear mixture MDPs, which does not require any variance information, whose regret can be improved to be  $\widetilde{\mathcal{O}}(d^{1.5}\sqrt{\sum_{t=1}^{T}\sigma_t^2} + d^2)$  as shown by Kim et al. (2021). The recent work by Hou et al. (2022) considered variance-constrained best arm identification, where the feedback noise only depends on the action by the agent (whereas ours can depend on time, which is more general than theirs). Another recent work (Zhao et al., 2022) studied variance-aware regret bounds for bandits with general function approximation in the known variance case.

Stochastic Contextual Sparse Linear Bandits. In the setting, the action set for each round  $t$  is i.i.d. sampled (called the "context"). It is known that  $\widetilde{\mathcal{O}}(\sqrt{sT})$  regret is achievable in this setting (Kim & Paik, 2019; Ren & Zhou, 2020; Oh et al., 2021; Ariu et al., 2022). However, in our setting where both action set  $\mathcal{X}$  and ground-truth  $\theta^{*}$  are fixed, a polynomial dependency on  $d$  is in general unavoidable because it is impossible to learn more than one parameter per arm (Bastani & Bayati, 2020), agreeing with the  $\Omega(\sqrt{dT})$  lower bound when  $s = 1$  (Antos & Szepesvári, 2009; Abbasi-Yadkori et al., 2012).

# 2 PROBLEM SETUP

Notations. We use  $[N]$  to denote the set  $\{1,2,\ldots ,N\}$  where  $N\in \mathbb{N}$ . For a vector  $x\in \mathbb{R}^d$ , we use  $\| x\| _p$  to its  $L_{p}$ -norm, namely  $\| x\| _p\triangleq (\sum_{i = 1}^d x_i^p)^{1 / p}$ . We use  $\mathbb{S}^{d - 1}$  to denote the  $(d - 1)$ -dimensional unit sphere, i.e.,  $\mathbb{S}^{d - 1}\triangleq \{x\in \mathbb{R}^d\mid \| x\| _2 = 1\}$ . We use  $\widetilde{\mathcal{O}} (\cdot)$  and  $\widetilde{\Theta} (\cdot)$  to hide all logarithmic factors in  $T,s,d$  and  $\log \frac{1}{\delta}$  (see Footnote 1). For a random event  $\mathcal{E}$ , we denote its indicator by  $\mathbb{1}[\mathcal{E}]$ .

We assume the action space and the ground-truth space are both the  $(d - 1)$ -dimensional unit sphere, denoted by  $\mathcal{X} \triangleq \mathbb{S}^{d - 1}$ . Denote the ground-truth by  $\theta^{*} \in \mathcal{X}$ . There will be  $T \geq 1$  rounds for the agent to make decisions sequentially. At the beginning of round  $t \in [T]$ , the agent has to choose an action  $x_{t} \in \mathcal{X}$ . At the end of step  $t$ , the agent receives a noisy feedback  $r_{t} = \langle x_{t}, \theta^{*} \rangle + \eta_{t}, \forall t \in [T]$ , where  $\eta_{t}$  is an independent zero-mean Gaussian random variable. Denote by  $\sigma_{t}^{2} = \mathrm{Var}(\eta_{t})$  the variance of  $\eta_{t}$ . For a fair comparison with non-variance-aware algorithms, we assume that  $\sigma_{t}^{2} \leq 1$ . The agent then receives a (deterministic and unrevealed) reward of magnitude  $\langle x_{t}, \theta^{*} \rangle$  for this round.

The agent is allowed to make the decision  $x_{t}$  based on all historical actions  $x_{1},\ldots ,x_{t - 1}$ , all historical feedback  $r_1,\dots ,r_{t - 1}$ , and any amount of private randomness. The agent's goal is to minimize the regret, defined as follows.

Definition 2 (Regret). The following random variable is the regret of a linear bandit algorithm:

$$
\mathcal {R} _ {T} = \max  _ {x \in \mathcal {X}} \sum_ {t = 1} ^ {T} \left\langle x, \theta^ {*} \right\rangle - \sum_ {t = 1} ^ {T} \left\langle x _ {t}, \theta^ {*} \right\rangle = \sum_ {t = 1} ^ {T} \left\langle \theta^ {*} - x _ {t}, \theta^ {*} \right\rangle , \tag {1}
$$

where the second equality is due to our assumption that  $\mathcal{X} = \mathbb{S}^{d - 1}$

For the sparse linear bandit problem, we have an additional restriction that  $\| \theta^{*}\|_{0}\leq s$ , i.e., there are at most  $s$  coordinates of  $\theta^{*}$  is non-zero. However, as mentioned in the introduction, the agent does not know anything about  $s$  - she only knows that she is facing a (probably sparse) linear environment.

Algorithm 1 Variance-Aware Sparse Linear Bandits (VASLB) Framework  
Input: Number of dimensions  $d$ , linear bandit algorithm  $\mathcal{F}$  and its regret estimator  $\overline{\mathcal{R}_n^{\mathcal{F}}}$   
1: Initialize gap threshold  $\Delta \gets \frac{1}{4}$ , estimated "good" coordinates  $S \gets \emptyset$ , current round  $t \gets 0$ .  
2: while  $t < T$  do  
3: if  $S \neq \emptyset$  then  
4: Initialize a new linear bandit instance  $\mathcal{F}$  on coordinates  $S$ .  
5: Execute  $\mathcal{F}$  for  $n_{\Delta}^{a} \geq 1$  steps & maintain pessimistic estimation  $\overline{\mathcal{R}_{n_{\Delta}}^{\mathcal{F}}}$ , until  $\frac{1}{n_{\Delta}^{a}} \overline{\mathcal{R}_{n_{\Delta}}^{\mathcal{F}}} < \Delta^{2}$ .  
6: Suppose that  $\mathcal{F}$  plays  $x_{1}, x_{2}, \ldots, x_{n_{\Delta}^{a}}$ . Set  $\widehat{\theta} = \frac{1}{n_{\Delta}^{a}} \sum_{i=1}^{n_{\Delta}^{a}} x_{i}$  as the estimate for  $\{\theta_{i}^{*}\}_{i \in S}$ .  
7: if  $\sum_{i \in S} \widehat{\theta}_{i}^{2} \leq 1 - \Delta^{2}$  then  
8: Let  $R \gets \sqrt{1 - \sum_{i \in S} \widehat{\theta}_{i}^{2}}$ ,  $K = d - |S|$ .  
9: Perform  $n_{\Delta}^{b} \geq 1$  calls to RANDOMPROJECTION $(K, R, S, \widehat{\theta})$  in Algorithm 2, until  
2√2∑n b k=1 (rk,i - r i)² ln 4/δ < n b · Δ/4, ∀1 ≤ i ≤ K, (2) where rk is the k-th return vector of RANDOMPROJECTION and  $\bar{r} \triangleq \frac{1}{n_{\Delta}^{b}} \sum_{k=1}^{n_{\Delta}^{b}} r_{k}$ .  
10: for i = 1, 2, ..., K do  
11: if  $|\bar{r}_{i}| > \Delta$  where  $\bar{r} = \frac{1}{n_{\Delta}^{b}} \sum_{k=1}^{n_{\Delta}^{b}} r_{k}$  then add the i-th element that is not in S to S.

Algorithm 2 The RANDOMPROJECTION Subroutine  
1: function RANDOMPROJECTION(K,R,S,θ)  
2: Generate K i.i.d. samples  $y_{1},y_{2},\ldots ,y_{K}$ , each with equal probability being  $\pm \frac{R}{\sqrt{K}}$ .  
3: Play  $x\in \mathcal{X}$  constructed as  $x_{i} = \left\{ \begin{array}{ll}\widehat{\theta}_{i}, & i\in S\\ y_{j}, & i \text{is the } j\text{-th element that is not in } S \end{array} \right.$   
4: return  $\frac{K}{R^2} ((r - \sum_{i\in S}\widehat{\theta}_i^2)y)$  where  $r = \langle x,\theta^{*}\rangle +\eta$  is the (noisy) feedback.

# 3 FRAMEWORK AND ANALYSIS

Our framework VASLB is presented in Algorithm 1. We explain its design in Section 3.1 and sketch its analysis in Section 3.2. Then we give two applications using VOFUL2 (Kim et al., 2021) and Weighted VOFUL (Zhou et al., 2021) as  $\mathcal{F}$ , whose analyses are sketched in Sections 4.1 and 4.2.

# 3.1 MAIN DIFFICULTIES AND TECHNICAL OVERVIEW

At a high level, our framework follows the spirit of the classic "explore-then-commit" approach (which is directly adopted by Carpentier & Munos (2012)), where the agent first identifies those "huge" entries of  $\theta^{*}$  and then performs a linear bandit algorithm on them. However, it is hard to incorporate variances into this vanilla idea to make it variance-aware – the desired regret depends on variances and is thus unknown to the agent. Thus it is difficult to determine a "gap threshold"  $\Delta$  (that is, the agent stops to "commit" after identifying all  $\theta_{i}^{*} \geq \Delta$ ) within a few rounds. For example, in the deterministic case, the agent must identify all non-zero entries to make the regret independent of  $T$ ; on the other hand, in the worst case where  $\sigma_{t} \equiv 1$ , the agent only needs to identify all entries with magnitude at least  $T^{-1/4}$  to yield  $\sqrt{T}$ -style regret bounds. At the same time, the actual setting might be mixture of them (e.g.,  $\sigma_{t} \equiv 0$  for  $t \leq t_{0}$  and  $\sigma_{t} \equiv 1$  for  $t > t_{0}$  where  $t_{0} \in [T]$ ). As a result, such an idea cannot always succeed in determining the correct threshold  $\Delta$  and getting the desired regret.

In our proposed framework, we tackle this issue by "explore-then-commit" multiple times. We reduce the uncertainty gently and alternate between "explore" and "commit" modes. We decrease a "gap threshold"  $\Delta$  in a halving manner and, at the same time, maintain a set  $S$  of coordinates that we

believe to have a magnitude larger than  $\Delta$ . For each  $\Delta$ , we "explore" (estimating  $\theta_i^*$  and adding those greater than  $\Delta$  into  $S$ ) and "commit" (performing linear bandit algorithms on coordinates in  $S$ ).

However, as we "explore" again after "committing", we face a unique challenge: Suppose we have already identified an entry  $i \in [d]$  to be at least  $2\Delta$ . During the next "explore" phase, we cannot do pure exploration – otherwise, coordinate  $i$  will incur  $4\Delta^2$  regret for each round. Fortunately, we can get an estimation  $\widehat{\theta}_i$  of  $\theta_i^*$  during the previous "commit" phase thanks to the regret-to-sample-complexity conversion (Eq. (3)). Guarded with this estimation, we can reserve  $\widehat{\theta}_i$  mass for arm  $i$  and subtract  $\widehat{\theta}_i^2$  from the feedback in subsequent "explore" phases. More precisely, we do the following.

1. In the "commit" phase where we apply the black-box  $\mathcal{F}$ , we estimate  $\{\theta_i^*\}_{i\in S}$  by the regret-to-sample-complexity conversion: Suppose  $\mathcal{F}$  plays  $x_{1},x_{2},\ldots ,x_{n}$  and achieves regret  $\mathcal{R}_n^{\mathcal{F}}$ , then

$$
\langle \theta^ {*} - \widehat {\theta}, \theta^ {*} \rangle \leq \frac {\mathcal {R} _ {n} ^ {\mathcal {F}}}{n}, \text {w h e r e} \widehat {\theta} \triangleq \frac {1}{n} \sum_ {i = 1} ^ {n} x _ {i}. \tag {3}
$$

Hence, if we take  $\{\widehat{\theta}_i\}_{i\in S}$  as an estimate of  $\{\theta_i^*\}_{i\in S}$ , the estimation error shrinks as  $\mathcal{R}_n^{\mathcal{F}}$  is sublinear and the LHS of Eq. (3) is non-negative. Moreover, as we can show that  $\widehat{\theta}$  is not away from  $\mathcal{X}$  by a lot (Lemma 18), we can safely use  $\{\widehat{\theta}_i\}_{i\in S}$  to estimate  $\{\theta_i^*\}_{i\in S}$  in subsequent phases.

More importantly, if we are granted access to  $\mathcal{R}_n^{\mathcal{F}}$ , we know how close the estimate is; we can proceed to the next stage once it becomes satisfactory. But it is unrevealed. Fortunately, we know the regret guarantee of  $\mathcal{F}$ , namely  $\overline{\mathcal{R}_n^{\mathcal{F}}}$ , which can serve as a pessimistic estimation of  $\mathcal{R}_n^{\mathcal{F}}$ . Hence, terminating when  $\frac{1}{n} \overline{\mathcal{R}_n^{\mathcal{F}}} < \Delta^2$  can ensure  $\langle \theta^* - \widehat{\theta}, \theta^* \rangle < \Delta^2$  to hold with high probability.

2. In the "exploration" phase, as mentioned before, we can keep the regret incurred by the coordinates identified in  $S$  small by putting mass  $\widehat{\theta}_i$  for each  $i\in S$ . For the remaining ones, we use random projection, an idea borrowed from compressed sensing literature (Blumensath & Davies, 2009; Carpentier & Munos, 2012), to find those with large magnitudes to add them to  $S$ .

One may notice that putting mass  $\widehat{\theta}_i$  for all  $i\in S$  will induce bias to our estimation as  $\sum_{i\in S}\widehat{\theta}_i^2\neq \sum_{i\in S}\widehat{\theta}_i\theta_i^*$ . However, as  $\widehat{\theta}_i$  is close to  $\theta_i^*$ , this bias will be bounded by  $\mathcal{O}(\Delta^2)$  and become dominated by  $\frac{\Delta}{4}$  as  $\Delta$  decreases. Hence, if we omit this bias, we can overestimate the estimation error due to standard concentration inequalities like Empirical Bernstein (Maurer & Pontil, 2009; Zhang et al., 2021). Once it becomes small enough, we alternate to the "commit" phase again.

Therefore, with high probability, we can ensure all coordinates not in  $S$  have magnitudes no more than  $\mathcal{O}(\Delta)$  and all coordinates in  $S$  will together contribute regret bounded by  $\mathcal{O}(\Delta^2)$ . Hence, the regret in each step is (roughly) bounded by  $\mathcal{O}(s\Delta^2)$ . Upper bounding the number of steps needed for each stage and exploiting the regret guarantees of the chosen  $\mathcal{F}$  then gives well-bounded regret.

# 3.2 ANALYSIS OF THE FRAMEWORK

Notations. For each  $\Delta$ , define  $\mathcal{T}_{\Delta}$  as the set of rounds associated with gap threshold  $\Delta$ . By the design of the algorithm, each  $\mathcal{T}_{\Delta}$  should be an interval. Moreover,  $\{\mathcal{T}_{\Delta}\}_{\Delta}$  forms a partition of  $[T]$ . Define  $\mathcal{T}_{\Delta}^{a}$  as all the rounds in the "commit" phase when the gap threshold is  $\Delta$  (where  $\mathcal{F}$  is executed), and  $\mathcal{T}_{\Delta}^{b}$  as the "explore" phase (i.e., those executing RANDOMPROJECTION). Let  $\widetilde{\mathcal{T}}_{\Delta}^{a}$  and  $\widetilde{\mathcal{T}}_{\Delta}^{b}$  be the steps that the agent decided not to proceed in  $\mathcal{T}_{\Delta}^{a}$  and  $\mathcal{T}_{\Delta}^{b}$ , respectively, which are formally defined as  $\widetilde{\mathcal{T}}_{\Delta}^{i} = \{t \in \mathcal{T}_{\Delta}^{i} \mid t \neq \max_{t' \in \mathcal{T}_{\Delta}^{i}} t'\}$ ,  $i = a, b$ . Define the final value of  $\Delta$  as  $\Delta_f$ . Denote  $n_{\Delta}^{a} = |\mathcal{T}_{\Delta}^{a}|$  and  $n_{\Delta}^{b} = |\mathcal{T}_{\Delta}^{b}|$  (both are stopping times). We then have  $\sum_{\Delta=2^{-2}, \ldots, \Delta_f}(n_{\Delta}^a + n_{\Delta}^b) = T$ .

With these notations, we can decompose our regret  $\mathcal{R}_T$  into two parts,  $\mathcal{R}_T^a$  and  $\mathcal{R}_T^b$ , each defined as

$$
\mathcal {R} _ {T} ^ {a} = \sum_ {\Delta = 2 ^ {- 2}, \ldots , \Delta_ {f}} \sum_ {t \in \mathcal {T} _ {\Delta} ^ {a}} \langle \theta^ {*} - x _ {t}, \theta^ {*} \rangle , \quad \mathcal {R} _ {T} ^ {b} = \sum_ {\Delta = 2 ^ {- 2}, \ldots , \Delta_ {f}} \sum_ {t \in \mathcal {T} _ {\Delta} ^ {b}} \langle \theta^ {*} - x _ {t}, \theta^ {*} \rangle ,
$$

where  $\mathcal{R}_T^a$  may depend on the choice of  $\mathcal{F}$  and  $\mathcal{R}_T^b$  only depends on the framework (Algorithm 1) itself. We now show that, as long as the regret estimation  $\overline{\mathcal{R}_n^{\mathcal{F}}}$  is indeed an overestimation of  $\mathcal{R}_n^{\mathcal{F}}$  with high probability, we can get a good upper bound of  $\mathcal{R}_T^b$ , which is formally stated as Theorem 3. The full proof of Theorem 3 will be presented in Appendix E and is only sketched here.

Theorem 3. Suppose that for any execution of  $\mathcal{F}$  that last for  $n$  steps,  $\overline{\mathcal{R}_n^{\mathcal{F}}} \geq \mathcal{R}_n^{\mathcal{F}}$  holds with probability  $1 - \delta$ , i.e.,  $\overline{\mathcal{R}_n^{\mathcal{F}}}$  is pessimistic. Then the total regret incurred by the second phase satisfies

$$
\mathcal {R} _ {T} ^ {b} = \widetilde {\mathcal {O}} \left(s \sqrt {d} \sqrt {\sum_ {t = 1} ^ {T} \sigma_ {t} ^ {2}} \log \frac {1}{\delta} + s \log \frac {1}{\delta}\right) \quad w i t h p r o b a b i l i t y 1 - \delta .
$$

Remark. This theorem indicates that our framework itself will only induce an  $(s\sqrt{d}, s)$ -variance-awareness to the resulting algorithm. As noticed by Abbasi-Yadtori et al. (2011), when  $\sigma_t \equiv 1$ ,  $\Omega(\sqrt{sdT})$  regret is unavoidable, which means that it is only sub-optimal by a factor no more than  $\sqrt{s}$ . Moreover, for deterministic cases, the  $\tilde{\mathcal{O}}(s)$  regret also matches the aforementioned divide-and-conquer algorithm, which is specially designed and can only work for deterministic cases.

Proof Sketch of Theorem 3. We define two good events with high probability for a given gap threshold  $\Delta$ :  $\mathcal{G}_{\Delta}$  and  $\mathcal{H}_{\Delta}$ . Informally,  $\mathcal{G}_{\Delta}$  means  $\sum_{i\in S}\theta_i^* (\theta_i^* -\widehat{\theta}_i) < \Delta^2$  (i.e.,  $\widehat{\theta}$  is close to  $\theta^{*}$  after "commit") and  $\mathcal{H}_{\Delta}$  stands for  $|\theta_i^*\| \geq \Omega (\Delta)$  if and only if  $i\in S$  (i.e., we "explore" correctly). Check Eq. (10) in the appendix for formal definitions. For  $\mathcal{G}_{\Delta}$ , from Eq. (3), we know that it happens as long as  $\overline{\mathcal{R}_n^\mathcal{F}}\geq \mathcal{R}_n^\mathcal{F}$ . It remains to argue that  $\operatorname *{Pr}\{\mathcal{H}_{\Delta}\mid \mathcal{G}_{\Delta},\mathcal{H}_{2\Delta}\} \geq 1 - s\delta$ .

By Algorithm 2, the  $i$ -th coordinate of each  $r_k$  ( $1 \leq k \leq n_{\Delta}^b$ ) is an independent sample of

$$
\frac {K}{R ^ {2}} \left(y _ {i}\right) ^ {2} \theta_ {i} ^ {*} + \sum_ {j \in S} \widehat {\theta} _ {j} \left(\theta_ {j} ^ {*} - \widehat {\theta} _ {j}\right) + \sum_ {j \notin S, j \neq i} \left(\frac {K}{R ^ {2}} y _ {i} y _ {j}\right) \theta_ {j} ^ {*} + \left(\frac {K}{R ^ {2}} y _ {i}\right) \eta_ {n}, \tag {4}
$$

where  $\frac{\sqrt{K}}{R} y_{i}$  is an independent Rademacher random variable. After conditioning on  $\mathcal{G}_{\Delta}$  and  $\mathcal{H}_{2\Delta}$ ,  $\sum_{i\in S}\widehat{\theta}_i^2$  and  $\sum_{i\in S}\widehat{\theta}_i\theta_i^*$  will be close. Therefore, the first term is exactly  $\theta_{i}^{*}$  (the magnitude we want to estimate), the second term is a small bias bounded by  $\mathcal{O}(\Delta^2)$  and the last two terms are zero-mean noises, which are bounded by  $\frac{\Delta}{4}$  according to Empirical Bernstein Inequality (Theorem 10) and our choice of  $n_{\Delta}^{b}$  (Eq. (2)). Hence,  $\operatorname*{Pr}\{\mathcal{H}_{\Delta}\mid \mathcal{G}_{\Delta},\mathcal{H}_{2\Delta}\} \geq 1 - s\delta$ .

Let us focus on an arm  $i^{*}$  never identified into  $S$  in Algorithm 1. By definition of  $n_{\Delta}^{b}$  (Eq. (2)),

$$
(n _ {\Delta} ^ {b} - 1) \frac {\Delta}{4} <   2 \sqrt {2 \sum_ {t \in \widetilde {\mathcal {T}} _ {\Delta} ^ {b}} (r _ {t , i ^ {*}} - \overline {{r}} _ {i ^ {*}}) ^ {2} \ln \frac {4}{\delta}} \leq 2 \sqrt {2 \sum_ {t \in \widetilde {\mathcal {T}} _ {\Delta} ^ {b}} (r _ {t , i ^ {*}} - \mathbb {E} [ r _ {t , i ^ {*}} ]) ^ {2} \ln \frac {4}{\delta}},
$$

where the second inequality is due to properties of sample variances. By  $\mathcal{G}_{\Delta}$ , those coordinates in  $S$  will incur regret of  $\sum_{i\in S}(\theta_i^* -x_{t,i})\theta_i^* = \sum_{i\in S}(\theta_i^* -\widehat{\theta}_i)\theta_i^* < \Delta^2$  for all  $t\in T_{\Delta}^{b}$ . Moreover, by  $\mathcal{H}_{2\Delta}$ , each arm outside  $S$  will roughly incur  $n_{\Delta}^{b}(\theta_{i}^{*})^{2} < \mathcal{O}(n_{\Delta}^{b}\Delta^{2})$  regret, as  $y_{i}$ 's are independent and zero-mean. As there are at most  $s$  non-zero coordinates, the total regret for  $\mathcal{T}_{\Delta}^{b}$  will be roughly bounded by  $\mathcal{O}(n_{\Delta}^{b}\cdot s\Delta^{2})$  (there exists another term due to randomized  $y_{i}$ 's, which is dominated and omitted here; check Lemma 21 for more details). Hence, the total regret is bounded by

$$
\mathcal {R} _ {T} ^ {b} \lesssim \sum_ {\Delta} \mathcal {O} (s n _ {\Delta} ^ {b} \Delta^ {2}) = s \cdot \widetilde {\mathcal {O}} \left(\sum_ {\Delta} \Delta \sqrt {\sum_ {t \in \mathcal {T} _ {\Delta} ^ {b}} (r _ {t , i ^ {*}} - \mathbb {E} [ r _ {t , i ^ {*}} ]) ^ {2} \ln \frac {4}{\delta}}\right) + \mathcal {O} (s).
$$

To avoid undesired poly  $(T)$  factors, we cannot directly apply Cauchy-Schwartz inequality to the sum of square roots (as there are a lot of  $\Delta$ 's). Instead, again by definition of  $n_{\Delta}^{b}$  (Eq. (2)), we observe the following lower bound of  $n_{\Delta}^{b}$ , which holds for all  $\Delta$  's except for  $\Delta_f$ :  $n_{\Delta}^{b} \geq \mathcal{O}\left(\frac{1}{\Delta}\sqrt{\sum_{t \in T_{\Delta}^{b}}(r_{t,i^*} - \mathbb{E}[r_{t,i^*}])^2\ln\frac{1}{\delta}}\right)$ . As  $\sum_{\Delta} n_{\Delta}^{b} \leq T$ , some arithmetic calculation gives (intuitively, by thresholding, we manage to "move" the summation over  $\Delta$  into the square root, though suffering an extra logarithmic factor; see Eq. (15) in the appendix for more details)

$$
\sum_ {\Delta \neq \Delta_ {f}} \Delta \sqrt {\sum_ {t \in \mathcal {T} _ {\Delta} ^ {b}} (r _ {t , i ^ {*}} - \mathbb {E} [ r _ {t , i ^ {*}} ]) ^ {2}} = \widetilde {\mathcal {O}} \left(\sqrt {\sum_ {\Delta \neq \Delta_ {f}} \Delta^ {2} \sum_ {t \in \mathcal {T} _ {\Delta} ^ {b}} (r _ {t , i ^ {*}} - \mathbb {E} [ r _ {t , i ^ {*}} ]) ^ {2}}\right).
$$

For a given  $\Delta$  and any  $1\leq k\leq n_{\Delta}^{b}$ , the expectation of  $(r_{k,i^{*}} - \mathbb{E}[r_{k,i^{*}}])^{2}$  is bounded by  $\left(1 + \frac{K}{R^2}\sigma_k^2\right)$  (Eq. (19) in the appendix), which is no more than  $\left(1 + \frac{4d}{\Delta^2}\sigma_k^2\right)$ . By concentration properties in the sample variances (Theorem 14 in the appendix), one can write (omitting all  $\log \frac{1}{\delta}$  terms)

$$
\mathcal {R} _ {T} ^ {b} = \mathcal {O} \left(\sum_ {\Delta} s n _ {\Delta} ^ {b} \Delta^ {2}\right) = \widetilde {\mathcal {O}} \left(\sqrt {\sum_ {\Delta} n _ {\Delta} ^ {b} \Delta^ {2}} + \sqrt {d \sum_ {t = 1} ^ {T} \sigma_ {t} ^ {2}}\right).
$$

Notice that  $\sum_{\Delta} n_{\Delta}^{b} \Delta^{2}$  also appears in the LHS. By some "self-bounding" property (Efroni et al., 2020, Lemma 38), we can conclude  $\mathcal{R}_T^b = \mathcal{O}(\sum_{\Delta} s n_{\Delta}^{b} \Delta^{2}) = \widetilde{\mathcal{O}}\left(s \sqrt{d \sum_{t=1}^{T} \sigma_t^2} + s\right)$ , as claimed.

# 4 APPLICATIONS OF THE PROPOSED FRAMEWORK

After showing Theorem 16, it only remains to bound  $\mathcal{R}_T^a$ , which depends on the choice of the plug-in algorithm  $\mathcal{F}$ . In this section, we give two specific choices of  $\mathcal{F}$ , VOFUL2 (Kim et al., 2021) and Weighted OFUL (Zhou et al., 2021). The former algorithm does not require the information of  $\sigma_t$ 's (i.e., it works in unknown-variance cases), albeit computationally inefficient. In contrast, the latter is computationally efficient but requires  $\sigma_t^2$  to be revealed with the feedback  $r_t$  at round  $t$ .

# 4.1 COMPUTATIONALLY INEFFICIENT ALGORITHM FOR UNKNOWN VARIANCES

We first use the VOFUL2 algorithm from Kim et al. (2021) as the plug-in algorithm  $\mathcal{F}$ , which has the following regret guarantee. Note that this is slightly stronger than the original bound: We derive a strengthened "self-bounding" version of it (the first inequality), which is critical to our analysis.

Proposition 4 (Kim et al. (2021, Variant of Theorem 2)). VOFUL2 executed for  $n$  rounds on  $d$  dimensions guarantees, w.p. at least  $1 - \delta$ , there exists a constant  $C = \widetilde{\mathcal{O}}(1)$  such that  $\mathcal{R}_n^{\mathcal{F}} \leq C\left(d^{1.5}\sqrt{\sum_{k=1}^{n}\eta_k^2\ln\frac{1}{\delta}} + d^2\ln\frac{4}{\delta}\right) = \widetilde{\mathcal{O}}\left(d^{1.5}\sqrt{\sum_{k=1}^{n}\sigma_k^2}\log\frac{1}{\delta} + d^2\log\frac{4}{\delta}\right)$ , where  $n$  is a stopping time finite a.s. and  $\sigma_1^2,\sigma_2^2,\ldots,\sigma_n^2$  are the variances of the independent Gaussians  $\eta_1,\eta_2,\ldots,\eta_n$ .

We now construct the regret over-estimation  $\overline{\mathcal{R}_n^{\mathcal{F}}}$ . Due to unknown variances, it is not straightforward. Our rescue is to use ridge linear regression  $\widehat{\beta} \triangleq \operatorname*{argmin}_{\beta \in \mathbb{R}^d} \left( \sum_{k=1}^{n} (r_k - \langle x_k, \beta \rangle)^2 + \lambda \| \beta \|_2 \right)$  for samples  $\{(x_k, r_k)\}_{k=1}^n$ , which ensures that the empirical variance estimation  $\sum_{k=1}^{n} (r_k - \langle x_k, \widehat{\beta} \rangle)^2$  differs from the true sample variance  $\sum_{k=1}^{n} \eta_k^2 = \sum_{k=1}^{n} (r_k - \langle x_k, \beta^* \rangle)^2$  by no more than  $\widetilde{\mathcal{O}}(s \log \frac{1}{\delta})$  (check Appendix D for a formal version). Accordingly, from Proposition 4, we can see that

$$
\mathcal {R} _ {n} ^ {\mathcal {F}} \leq \overline {{\mathcal {R} _ {n} ^ {\mathcal {F}}}} \triangleq C \left(s ^ {1. 5} \sqrt {\sum_ {k = 1} ^ {n} (r _ {k} - \langle x _ {k} , \widehat {\beta} \rangle) ^ {2} \ln \frac {1}{\delta}} + s ^ {2} \sqrt {2 \ln \frac {n}{s \delta^ {2}} \ln \frac {1}{\delta}} + s ^ {1. 5} \sqrt {2 \ln \frac {1}{\delta}} + s ^ {2} \ln \frac {1}{\delta}\right). \tag {5}
$$

Moreover, one can observe that the total sample variance  $\sum_{k=1}^{n} \eta_k^2$  is bounded by (a constant multiple of) the total variance  $\sum_{k=1}^{n} \sigma_k^2$  (which is formally stated as Theorem 13 in the appendix). Therefore, with Eq. (5) as our pessimistic regret estimation  $\overline{\mathcal{R}_n^\mathcal{F}}$ , we have the following regret guarantee.

Theorem 5 (Regret of Algorithm 1 with VOFUL2). Algorithm 1 with VOFUL2 as  $\mathcal{F}$  and  $\overline{\mathcal{R}_n^{\mathcal{F}}}$  defined in Eq. (5) ensures that  $\mathcal{R}_T = \widetilde{\mathcal{O}}\big((s^{2.5} + s\sqrt{d})\sqrt{\sum_{t=1}^{T}\sigma_t^2}\log \frac{1}{\delta} + s^3\log \frac{1}{\delta}\big)$  with probability  $1 - \delta$ .

Due to space limitations, we defer the full proof to Appendix F.1 and only sketch it here.

Proof Sketch of Theorem 5. To bound  $\mathcal{R}_T^a$ , we consider the regret from the coordinates in and outside  $S$  separately. For the former, the total regret in a single phase with gap threshold  $\Delta$  is simply controlled by  $\widetilde{\mathcal{O}}\left(s^{1.5}\sqrt{\sum_{t\in\mathcal{T}_{\Delta}^a}\eta_t^2\log\frac{1}{\delta}} + s^2\log\frac{1}{\delta}\right)$  (thanks to Proposition 4). For the latter, each non-zero coordinate outside  $S$  can at most incur  $\mathcal{O}(\Delta^2)$  regret for each  $t\in \mathcal{T}_{\Delta}^a$ . By definition of  $n_{\Delta}^a$  (Line 5),

we have  $n_{\Delta}^{a} = |\mathcal{T}_{\Delta}^{a}| = \mathcal{O}\left(\frac{s^{1.5}}{\Delta^{2}}\sqrt{\sum_{t\in\widetilde{\mathcal{T}}_{\Delta}^{a}}\eta_{t}^{2}\ln\frac{1}{\delta}} +\frac{s^{2}}{\Delta^{2}}\ln\frac{1}{\delta}\right)$ , just like the proof of Theorem 3. As the regret from the second part is bounded by  $\mathcal{O}(s\Delta^2\cdot n_{\Delta}^{a})$ , these two parts together sum to

$$
\mathcal {R} _ {T} ^ {a} \leq \sum_ {\Delta} \mathcal {O} \left(s ^ {2. 5} \sqrt {\sum_ {t \in \mathcal {T} _ {\Delta} ^ {a}} \eta_ {t} ^ {2} \log \frac {1}{\delta}} + s ^ {3} \log \frac {1}{\delta} + s \Delta^ {2}\right).
$$

As in Theorem 3, we notice that  $n_{\Delta}^{a} = \Omega \left( \frac{s^{1.5}}{\Delta^{2}} \sqrt{\sum_{t \in \widetilde{\mathcal{T}_{\Delta}^{a}}} \eta_{t}^{2} \ln \frac{1}{\delta}} + \frac{s^{2}}{\Delta^{2}} \ln \frac{1}{\delta} \right)$  for all  $\Delta \neq \Delta_{f}$  again by definition of  $n_{\Delta}^{a}$ . This will move the summation over  $\Delta$  into the square root. Moreover, by the fact that  $\eta_{t}^{2} = \mathcal{O}(\sigma_{t}^{2} \log \frac{1}{\delta})$  (Theorem 14 in the appendix), we have  $\mathcal{R}_T^a = \widetilde{\mathcal{O}}\left(s^{2.5} \sqrt{\sum_{t=1}^T \sigma_t^2} \log \frac{1}{\delta} + s^3 \log \frac{1}{\delta}\right)$ . Combining this with the bound of  $\mathcal{R}_T^b$  provided by Theorem 3 concludes the proof.

# 4.2 COMPUTATIONALLY EFFICIENT ALGORITHM FOR KNOWN VARIANCES

In this section, we consider a computational efficient algorithm Weighted OFUL (Zhou et al., 2021), which itself requires  $\sigma_t^2$  to be presented at the end of round  $t$ . Their algorithm guarantees:

Proposition 6 (Zhou et al. (2021, Corollary 4.3)). With probability at least  $1 - \delta$ , Weighted OFUL executed for  $n$  steps on  $d$  dimensions guarantees  $\mathcal{R}_T^{\mathcal{F}} \leq C \left( \sqrt{dn \log \frac{1}{\delta}} + d \sqrt{\sum_{k=1}^{n} \sigma_k^2 \log \frac{1}{\delta}} \right)$ , where  $C = \widetilde{\mathcal{O}}(1)$ ,  $n$  is a stopping time finite a.s., and  $\sigma_1^2, \sigma_2^2, \ldots, \sigma_n^2$  are the variances of  $\eta_1, \eta_2, \ldots, \eta_n$ .

Taking  $\mathcal{F}$  as Weighted OFUL, we will have the following regret guarantee for sparse linear bandits: Theorem 7 (Regret of Algorithm 1 with Weighted OFUL). Algorithm 1 with Weighted OFUL as  $\mathcal{F}$  and  $\overline{\mathcal{R}_n^{\mathcal{F}}}$  defined as  $C\big(\sqrt{sn\ln\frac{1}{\delta}} +s\sqrt{\sum_{k = 1}^{n}\sigma_k^2\ln\frac{1}{\delta}}\big)$  guarantees  $\mathcal{R}_T = \widetilde{\mathcal{O}}\left((s^2 + s\sqrt{d})\sqrt{\sum_{t = 1}^{T}\sigma_t^2}\log \frac{1}{\delta} +s^{1.5}\sqrt{T}\log \frac{1}{\delta}\right)$  with probability  $1 - \delta$

The proof is similar to that of Theorem 5, i.e., bounding  $n_{\Delta}^{a}$  by Line 5 of Algorithm 1 and then using summation techniques to move the summation over  $\Delta$  into the square root. The only difference is that we will need to bound  $\mathcal{O}\left(\sum_{\Delta} \Delta^{-2}\right)$ , which seems to be as large as  $T$  if we follow the analysis of Theorem 5. However, as we included an additive factor  $\sqrt{sn \ln \frac{1}{\delta}}$  in the regret over-estimation  $\overline{\mathcal{R}_n^{\mathcal{F}}}$ , we have  $n_{\Delta}^{a} \geq \Delta^{-2} \sqrt{sn_{\Delta}^{a} \ln \frac{1}{\delta}}$ , which means  $n_{\Delta}^{a} = \Omega(s \Delta^{-4})$ . From  $\sum_{\Delta} n_{\Delta}^{a} \leq T$ , we can consequently bound  $\sum_{\Delta} \Delta^{-2}$  as  $\mathcal{O}\left(\sqrt{\frac{T}{s}}\right)$ . The remaining part is just an analog of Theorem 5. Therefore, the proof is omitted in the main text and postponed to Appendix G.

# 5 CONCLUSION

We considered the sparse linear bandit problem with heteroscedastic noises and provided a general framework to reduce any variance-aware linear bandit algorithm  $\mathcal{F}$  to an algorithm  $\mathcal{G}$  for sparse linear bandits that is also variance-aware. As an specific application, we first applied the computationally inefficient algorithm VOFUL from Zhang et al. (2021) and Kim et al. (2021). The result algorithm works for the unknown-variance case and gets  $\widetilde{\mathcal{O}}((s^{2.5} + s\sqrt{d})\sqrt{\sum_{t=1}^{T}\sigma_t^2}\log\frac{1}{\delta} + s^3\log\frac{1}{\delta})$  regret, which, when regarding the sparsity factor  $s \ll d$  as a constant, not only is worst-case optimal but also enjoys constant regret in the deterministic case. We also applied the efficient algorithm Weighted OFUL by Zhou et al. (2021) which, requires known variance, and got  $\widetilde{\mathcal{O}}((s^2 + s\sqrt{d})\sqrt{\sum_{t=1}^{T}\sigma_t^2}\log\frac{1}{\delta} + (\sqrt{sT} + s)\log\frac{1}{\delta})$  regret, still independent of  $d$  in the deterministic case.

However, there is still a gap in the worst-case regret in terms of  $s$ , as the lower bound for sparse linear bandits is  $\Omega(\sqrt{sdT})$  instead of our  $\widetilde{\mathcal{O}}(s\sqrt{dT})$  when  $\sigma_t \equiv 1$ . Closing this gap is an interesting future work. Improving dependencies on  $s$  can also be important.

# REFERENCES

Yasin Abbasi-Yadkori, David Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems, 24, 2011.  
Yasin Abbasi-Yadkori, David Pal, and Csaba Szepesvari. Online-to-confidence-set conversions and application to sparse stochastic bandits. In Artificial Intelligence and Statistics, pp. 1-9. PMLR, 2012.  
Ayya Alieva, Ashok Cutkosky, and Abhimanyu Das. Robust pure exploration in linear bandits with limited budget. In International Conference on Machine Learning, pp. 187-195. PMLR, 2021.  
András Antos and Csaba Szepesvári. Stochastic bandits with large action sets revisited. Personal communication, 2009.  
Kaito Ariu, Kenshi Abe, and Alexandre Proutiere. Thresholded lasso bandit. In International Conference on Machine Learning, pp. 878-928. PMLR, 2022.  
Jean-Yves Audibert, Rémi Munos, and Csaba Szepesvári. Exploration-exploitation tradeoff using variance estimates in multi-armed bandits. Theoretical Computer Science, 410(19):1876-1902, 2009.  
Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning, pp. 263-272. PMLR, 2017.  
Hamsa Bastani and Mohsen Bayati. Online decision making with high-dimensional covariates. Operations Research, 68(1):276-294, 2020.  
Thomas Blumensath and Mike E Davies. Iterative hard thresholding for compressed sensing. Applied and computational harmonic analysis, 27(3):265-274, 2009.  
Alexandra Carpentier and Rémi Munos. Bandit theory meets compressed sensing for high dimensional stochastic linear bandit. In Artificial Intelligence and Statistics, pp. 190-198. PMLR, 2012.  
Wei Chu, Lihong Li, Lev Reyzin, and Robert Schapire. Contextual bandits with linear payoff functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 208-214. JMLR Workshop and Conference Proceedings, 2011.  
Varsha Dani, Thomas P Hayes, and Sham M Kakade. Stochastic linear optimization under bandit feedback. In 21st Annual Conference on Learning Theory, pp. 355-366. Omnipress, 2008.  
Rémy Degenne, Wouter M Koolen, and Pierre Ménard. Non-asymptotic pure exploration by solving games. Advances in Neural Information Processing Systems, 32, 2019.  
Kefan Dong, Jiaqi Yang, and Tengyu Ma. Provable model-based nonlinear bandit and reinforcement learning: Shelve optimism, embrace virtual curvature. Advances in Neural Information Processing Systems, 34, 2021.  
Yonathan Efroni, Shie Mannor, and Matteo Pirotta. Exploration-exploitation in constrained mdps. arXiv preprint arXiv:2003.02189, 2020.  
Xiequan Fan, Ion Grama, and Quansheng Liu. Exponential inequalities for martingales with applications. Electronic Journal of Probability, 20:1-22, 2015.  
Louis Faury, Marc Abeille, Clément Calauzènes, and Olivier Fercoq. Improved optimistic algorithms for logistic bandits. In International Conference on Machine Learning, pp. 3052-3060. PMLR, 2020.  
David A Freedman. On tail probabilities for martingales. the Annals of Probability, pp. 100-118, 1975.  
Botao Hao, Tor Lattimore, and Mengdi Wang. High-dimensional sparse linear bandits. Advances in Neural Information Processing Systems, 33:10753-10763, 2020.

Botao Hao, Tor Lattimore, and Wei Deng. Information directed sampling for sparse linear bandits. Advances in Neural Information Processing Systems, 34, 2021a.  
Botao Hao, Tor Lattimore, Csaba Szepesvári, and Mengdi Wang. Online sparse reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pp. 316-324. PMLR, 2021b.  
Jean Honorio and Tommi Jaakkola. Tight bounds for the expected risk of linear classifiers and pac-bayes finite-sample guarantees. In Artificial Intelligence and Statistics, pp. 384-392. PMLR, 2014.  
Yunlong Hou, Vincent YF Tan, and Zixin Zhong. Almost optimal variance-constrained best arm identification. arXiv preprint arXiv:2201.10142, 2022.  
Yassir Jedra and Alexandre Proutiere. Optimal best-arm identification in linear bandits. Advances in Neural Information Processing Systems, 33:10007-10017, 2020.  
Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient? Advances in neural information processing systems, 31, 2018.  
Sampath Kannan, Jamie H Morgenstern, Aaron Roth, Bo Waggoner, and Zhiwei Steven Wu. A smoothed analysis of the greedy algorithm for the linear contextual bandit problem. Advances in neural information processing systems, 31, 2018.  
Gi-Soo Kim and Myunghee Cho Paik. Doubly-robust lasso bandit. Advances in Neural Information Processing Systems, 32, 2019.  
Yeoneung Kim, Insoon Yang, and Kwang-Sung Jun. Improved regret analysis for variance-adaptive linear bandits and horizon-free linear mixture mdps. arXiv preprint arXiv:2111.03289, 2021.  
Johannes Kirschner and Andreas Krause. Information directed sampling and bandits with heteroscedastic noise. In Conference On Learning Theory, pp. 358-384. PMLR, 2018.  
Tor Lattimore and Marcus Hutter. Pac bounds for discounted mdps. In International Conference on Algorithmic Learning Theory, pp. 320-334. Springer, 2012.  
Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University Press, 2020.  
Tor Lattimore, Koby Crammer, and Csaba Szepesvári. Linear multi-resource allocation with semibandit feedback. Advances in Neural Information Processing Systems, 28, 2015.  
Yingkai Li, Yining Wang, and Yuan Zhou. Nearly minimax-optimal regret for linearly parameterized bandits. In Conference on Learning Theory, pp. 2173-2174. PMLR, 2019.  
Yingkai Li, Yining Wang, Xi Chen, and Yuan Zhou. Tight regret bounds for infinite-armed linear contextual bandits. In International Conference on Artificial Intelligence and Statistics, pp. 370-378. PMLR, 2021.  
Andreas Maurer and Massimiliano Pontil. Empirical bernstein bounds and sample-variance penalization. In COLT 2009 - The 22nd Conference on Learning Theory, 2009.  
Min-hwan Oh, Garud Iyengar, and Assaf Zeevi. Sparsity-agnostic lasso bandit. In International Conference on Machine Learning, pp. 8271-8280. PMLR, 2021.  
Zhimei Ren and Zhengyuan Zhou. Dynamic batch learning in high-dimensional sparse linear contextual bandits. arXiv preprint arXiv:2008.11918, 2020.  
Marta Soare, Alessandro Lazaric, and Rémi Munos. Best-arm identification in linear bandits. Advances in Neural Information Processing Systems, 27, 2014.  
Yining Wang, Yi Chen, Ethan X Fang, Zhaoran Wang, and Runze Li. Nearly dimension-independent sparse linear bandit over small action spaces via best subset selection. arXiv preprint arXiv:2009.02003, 2020.

Andrea Zanette and Emma Brunskill. Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. In International Conference on Machine Learning, pp. 7304-7312. PMLR, 2019.  
Zihan Zhang, Jiaqi Yang, Xiangyang Ji, and Simon S Du. Improved variance-aware confidence sets for linear bandits and linear mixture mdp. Advances in Neural Information Processing Systems, 34, 2021.  
Heyang Zhao, Dongruo Zhou, Jiafan He, and Quanquan Gu. Bandit learning with general function classes: Heteroscedastic noise and variance-dependent regret bounds. arXiv preprint arXiv:2202.13603, 2022.  
Dongruo Zhou, Quanquan Gu, and Csaba Szepesvari. Nearly minimax optimal reinforcement learning for linear mixture markov decision processes. In Conference on Learning Theory, pp. 4532-4576. PMLR, 2021.
