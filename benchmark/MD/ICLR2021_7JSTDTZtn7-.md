# BYZANTINE-ROBUST LEARNING ON HETEROGENEOUS DATASES VIA RESAMPLING

Anonymous authors

Paper under double-blind review

# ABSTRACT

In Byzantine robust distributed optimization, a central server wants to train a machine learning model over data distributed across multiple workers. However, a fraction of these workers may deviate from the prescribed algorithm and send arbitrary messages to the server. While this problem has received significant attention recently, most current defenses assume that the workers have identical data. For realistic cases when the data across workers are heterogeneous (non-iid), we design new attacks which circumvent these defenses leading to significant loss of performance. We then propose a simple resampling scheme that adapts existing robust algorithms to heterogeneous datasets at a negligible computational cost. We theoretically and experimentally validate our approach, showing that combining resampling with existing robust algorithms is effective against challenging attacks.

# 1 INTRODUCTION

Distributed or federated machine learning, where the data is distributed across multiple workers, has become an increasingly important learning paradigm both due to growing sizes of datasets, as well as privacy and security concerns. In such a setting, the workers collaborate to train a single model without transmitting their data directly over the networks (McMahan et al., 2016; Bonawitz et al., 2019; Kairouz et al., 2019). Due to the presence of either actively malicious agents in the network, or simply due to system and network failures, some workers may disobey the protocols and send arbitrary messages; such workers are also known as Byzantine workers (Lamport et al., 2019). Byzantine robust optimization algorithms combine the gradients received by all workers using robust aggregation rules, to ensure that the training is not impacted by the malicious workers.

While this problem has received significant recent attention, (Alistarh et al., 2018; Blanchard et al., 2017; Yin et al., 2018a), most of the current approaches assume that the data present on each different worker has identical distribution. In this work, we show that existing Byzantine-robust methods catastrophically fail in the realistic setting when the data is distributed heterogeneously across the workers. We then propose a simple resampling scheme which can be readily combined with existing aggregation rules to allow robust training on heterogeneous data.

Contribution. Concretely, our contributions in this work are

- We show that when the data across workers is heterogeneous, existing robust rules might not converge, even without any Byzantine adversaries.  
- We propose two new attacks, normalized gradient and mimic, which take advantage of data heterogeneity and circumvent median and sign-based defenses (Blanchard et al., 2017; Pillutla et al., 2019; Li et al., 2019).  
- We propose a simple new resampling step which can be used before any existing robust aggregation rule. We instantiate our scheme with KRUM and theoretically prove that the resampling generalizes it to the setting of heterogeneous data.  
- Our experiments evaluate the proposed resampling scheme against known and new attacks and show that it drastically improves the performance of 3 existing schemes on realistic heterogeneously distributed datasets.

Setup and notations. We study the general distributed optimization problem

$$
\mathcal {L} ^ {\star} = \min  _ {\boldsymbol {x} \in \mathbb {R} ^ {d}} \left\{\mathcal {L} (\boldsymbol {x}) := \frac {1}{n} \sum_ {i = 1} ^ {n} \mathcal {L} _ {i} (\boldsymbol {x}) \right\} \tag {1}
$$

where  $\mathcal{L}_i:\mathbb{R}^d\to \mathbb{R}$  are the individual loss functions distributed among  $n$  workers, each having its own (heterogeneous) data distribution  $\{\mathcal{D}_i\}_{i = 1}^n$ . The case of empirical risk minimization with  $m_{i}$  datapoints  $\pmb {\xi}_i\sim \mathcal{D}_i$  on worker  $i$  is obtained when using  $\mathcal{L}_i(\pmb {x})\coloneqq \frac{1}{m_i}\sum_{j = 1}^{m_i}\mathcal{L}_i(\pmb {x},\pmb {\xi}_i^j)$ . The (stochastic) gradient computed by a good node  $i$  with sample  $j$  is given as  $\pmb {g}_i(\pmb {x})\coloneqq \nabla \mathcal{L}_i(\pmb {x},\pmb {\xi}_i^j)$  with mean  $\mu_{i}$  and variance  $\sigma_i^2$ . We also assume that the heterogeneity (variance across good workers) is bounded i.e.

$$
\mathbb {E} _ {i} \| \nabla \mathcal {L} _ {i} (\boldsymbol {x}) - \nabla \mathcal {L} (\boldsymbol {x}) \| ^ {2} \leq \bar {\sigma} ^ {2}, \forall \boldsymbol {x}.
$$

We write  $\pmb{g}_i$  instead of  $g_i(\pmb{x}^t)$  when there is no ambiguity. A distributed training step using an aggregation rule is given as

$$
\boldsymbol {x} ^ {t + 1} := \boldsymbol {x} ^ {t} - \gamma^ {t} \operatorname {A g g r} \left(\left\{\boldsymbol {g} _ {i} \left(\boldsymbol {x} ^ {t}\right): i \in [ n ] \right\}\right) \tag {2}
$$

If the aggregation rule is the arithmetic mean, then (2) recovers standard minibatch SGD.

Byzantine attack model. In each iteration, there is a set  $\mathbf{Byz}$  of at most  $f$  Byzantine workers. The remaining workers are good, thus follow the described protocol. A Byzantine worker  $j \in \mathbf{Byz}$  can deviate from protocol and send an arbitrary vector to the server. Besides, we also allow that Byzantine workers can collude with each other and know every state of the system. Unlike martingale-based approaches like (Alistarh et al., 2018), we allow the set  $\mathbf{Byz}$  to change over time (Blanchard et al., 2017; Chen et al., 2017; Mhamdi et al., 2018).

# 2 RELATED WORK

There has been significant recent work of the case when the workers have identical data distributions (Blanchard et al., 2017; Chen et al., 2017; Mhamdi et al., 2018; Alistarh et al., 2018; Mhamdi et al., 2018; Yin et al., 2018a;b; Su & Xu, 2018; Damaskinos et al., 2019). We discuss the most pertinent of these methods next. Blanchard et al. (2017) formalize the Byzantine robust setup and propose a distance-based approach Krum which selects a worker whose gradient is very close to at least half the other workers. A different approach involves using the median and its variants (Blanchard et al., 2017; Pillutla et al., 2019; Yin et al., 2018a). Yin et al. (2018a) propose to use and analyze the coordinate-wise median method (CM). Pillutla et al. (2019) use a smoothed version of Weiszfeld's algorithm to iteratively compute an approximate geometric median of the input gradients. In a third approach, (Bernstein et al., 2018) propose to use the signs of the gradients and then aggregate them by majority vote, however, (Karimireddy et al., 2019) show that it may not always converge. Finally, Alistarh et al. (2018) use a martingale-based aggregation rule which gives a sample complexity optimal algorithm for iid data. The distance-based approach of Krum was later extended in Mhamdi et al. (2018) who propose Bulyan to overcome the dimensional leeway attack. This is the so called strong Byzantine resilience and is orthogonal to the question of non-iid-ness we study here. Recently, (Peng & Ling, 2020; Yang & Bajwa, 2019a;b) studied Byzantine-resilient algorithms in the decentralized setting where there is no central server available. Extending our techniques to the decentralized setting is an important direction for future work.

In a different line of work, (Lai et al., 2016; Diakonikolas et al., 2019) develop sophisticated spectral techniques to robust estimate the mean of a high dimensional multi-variate standard Gaussian distribution where samples are evenly distributed in all directions and the attackers are concentrated in one direction. Very recent work (Data & Diggavi, 2020) extend the theoretical analysis to non-convex, strongly-convex and non-i.i.d setup under a gradient dissimilarity assumption and propose a gradient compression scheme on top of it. Our resampling trick can be combined with it to further reduce gradient dissimilarity.

Many attacks have been devised for distributed training. For the iid setting, the state-of-the-art attacks are (Baruch et al., 2019; Xie et al., 2019b). The latter attack is very strong when the fraction of adversaries is large (nearly half), but in this work we focus on settings when this fraction is quite small (e.g.  $\leq 0.2$ ). Further our normalized mean attack Section 3.2 is inspired by (Xie et al., 2019b). The former work focuses on attacks which are coordinated across time steps. Developing strong practical defenses even in the iid case against such time-coordinated attacks remains an open problem. In this work, we sidestep this issue by restricting ourselves to new attacks made possible by non-iid data and studying how to overcome them. We focus on schemes which work in the iid setting, but fail with non-iid data. Once a new method which can defend against (Baruch et al., 2019) is developed, our proposed scheme shows how to adapt such a method to the important non-iid case. For the non-iid setting, backdoor attacks are designed to take advantage of heavy-tailed data and manipulate

model inference on specific subtask, rather than lower the overall accuracies of training (Bagdasaryan et al., 2018; Bhagoji et al., 2018). In contrast, this paper is not intended to address aforementioned challenges but rather to defend the attacks that lower the training accuracies in the non-iid setting.

As far as we are aware, only (Li et al., 2019; Ghosh et al., 2019; Sattler et al., 2020) explicitly investigate Byzantine robustness with non-iid workers. Li et al. (2019) proposes an SGD variant (RSA) which modifies the original objective by adding an  $\ell_1$  penalty. Ghosh et al. (2019); Sattler et al. (2020) assume that all workers belong to an apriori fixed number of clusters and use an outlier-robust clustering method to recover these clusters. If we assume that the server has the entire training dataset and can control the distribution of samples to good workers, Xie et al. (2019a); Chen et al. (2018); Rajput et al. (2019) show that non-iid-ness can be overcome. Typical examples of this is distributed training of neural networks on public cloud, or volunteer computing Meeds et al. (2015); Miura & Harada (2015). However, none of these methods are applicable in the standard federated learning setup we consider here. We aim to minimize the original loss function over workers while respecting the non-iid data locality, i.e. the partition of the given heterogeneous dataset over the workers, without data transfer.

# 3 ATTACKS AGAINST EXISTING AGGREGATION SCHEMES

In this section we show that when the data across the workers is heterogeneous (non-iid), then we can design new attacks which take advantage of the heterogeneity, leading to the failure of existing aggregation schemes. We study three classes of robust aggregation schemes: i) schemes which select a representative worker in each round (e.g. KRUM (Blanchard et al., 2017)), ii) schemes which use normalized means (e.g. RSA (Li et al., 2019)), and iii) those which use the median (e.g. RFA (Pillutla et al., 2019)). We show realistic settings under which each of these classes would fail when faced with heterogeneous data.

# 3.1 FAILURE OF REPRESENTATIVE WORKER SCHEMES ON NON-IID DATA

Algorithms like KRUM select workers who are representative of a majority of the workers, by relying on statistics such as pairwise differences between the various worker updates. Let  $(g_{1},\ldots ,g_{n})$  be the gradients by the workers,  $f$  of which are Byzantine (e.g.  $n\geq 2f + 3$  for KRUM). For  $i\neq j$ , let  $i\rightarrow j$  denote that  $g_{j}$  belongs to the  $n - f - 2$  closest vectors to  $g_{i}$ . Then KRUM is defined as follows

$$
\operatorname {K R U M} \left(\boldsymbol {g} _ {1}, \dots , \boldsymbol {g} _ {n}\right) := \arg \min  _ {i} \sum_ {i \rightarrow j} \| \boldsymbol {g} _ {i} - \boldsymbol {g} _ {j} \| ^ {2} \tag {3}
$$

However, when the data across the workers is heterogeneous, there is no 'representative' worker. This is because each worker computes their local gradient over vastly different local data. Hence, for convergence it is important to not only select a good (non-Byzantine) worker, but also ensure that each of the good workers is selected with roughly equal frequency. Hence KRUM suffers a significant loss in performance with heterogeneous data, even when there are no Byzantine workers.

For example, when KRUM is used for iid datasets without adversary ( $f = 0$ , see left of Figure 1a), the test accuracy is close to simple average and the gap can be filled by MULTI-KRUM (Blanchard et al., 2017). The right plot of Figure 1a also shows that KRUM's selection of gradients is biased towards certain nodes. When KRUM is applied to non-iid datasets (the middle of Figure 1a), KRUM performs poorly even without any attack. This is because KRUM mostly selects gradients from a few nodes whose distribution is closer to others (the right of Figure 1a). This is an example of how robust aggregation rules may fail on realistic non-iid datasets.

# 3.2 ATTACKS ON NORMALIZED AGGREGATION SCHEMES

Instead of simply averaging the gradients, some methods first normalize them and then average. This limits the influence of the Byzantine workers since they cannot output extremely large gradients, and hence is more robust. For example RFA (Pillutla et al., 2019) with  $T = 1$  uses following aggregation rule:

$$
\mathrm {N M} \left(\boldsymbol {g} _ {1}, \dots , \boldsymbol {g} _ {n}\right) = \sum_ {i = 1} ^ {n} \frac {\boldsymbol {g} _ {i}}{\| \boldsymbol {g} _ {i} \| _ {2}} \tag {4}
$$

Other methods such as RSA (Li et al., 2019) or signum (Bernstein et al., 2018) normalize entries coordinate-wise before taking a majority vote i.e. update the server model  $\pmb{x}_0$  on server using local model  $\pmb{x}_i$  from node  $i$  (not gradient) using

$$
\operatorname {R S A} \left(\boldsymbol {x} _ {0}; \boldsymbol {x} _ {1}, \dots , \boldsymbol {x} _ {n}\right) := \nabla f _ {0} \left(\boldsymbol {x} _ {0}\right) + \lambda \sum_ {i = 1} ^ {n} \operatorname {s i g n} \left(\boldsymbol {x} _ {0} - \boldsymbol {x} _ {i}\right) \tag {5}
$$

![](images/3613ee2486d4369b4ae7bd44f78c0fb0377dd15693ac55d83f98c11b2a1914b7.jpg)

![](images/46c8d3d0dfc7ee84023f9cf5f6a8a373cb31e49680c5257d1433c4ade476ad29.jpg)  
(b) Comparing normalized mean (RFA with  $\mathrm{T} = 1$ ) under the normalized mean attack with  $f = 0, 1, 2$  attackers.

![](images/123b06507fa9157cc4d0e9652e0438fce02b1b06f936035054d603b71efbffd2.jpg)  
(a) Left & middle: Comparing arithmetic mean with Krum on iid and non-iid datasets, without any Byzantine workers. Right: Histogram of selected gradients.  
Figure 1: Failures of existing aggregation rules on the non-uid MNIST dataset. In all experiments, there are 8 good and  $f$  Byzantine workers.  
(c) Comparing coordinate-wise median (CM) and geometric median (RFA with  $T = 8$ ) under the mimic2 attack on iid and non-iid datasets.

where  $f_0$  is a strongly convex penalty term and  $\lambda > 0$  is a relaxation parameter.

However, a Byzantine worker can still craft an "omniscient" attack to foil robust aggregations, using an approach similar to the negative sum for the arithmetic mean (Blanchard et al., 2017; Li et al., 2019):

$$
\boldsymbol {v} := - \sum_ {i \in \mathbf {g o o d}} \frac {\boldsymbol {g} _ {i}}{\| \boldsymbol {g} _ {i} \| _ {2}} \tag {6}
$$

On the right side of Figure 1b, we can see that this attack lowers the accuracy of RFA-T1 significantly, as the number of Byzantine workers increases. Comparing to its iid counterpart, the normalized mean attack is even more impactful in the non-iid setting.

# 3.3 ATTACKS ON MEDIAN-BASED SCHEMES

Geometric median and its variants are popular in robust learning research (Blanchard et al., 2017; Chen et al., 2017; Pillutla et al., 2019; Yin et al., 2018a; Mhamdi et al., 2018). Given gradients  $\{\pmb{g}_1, \dots, \pmb{g}_n\}$ , we use the estimator

$$
\operatorname {G M} \left(\boldsymbol {g} _ {1}, \dots , \boldsymbol {g} _ {n}\right) := \operatorname {a r g m i n} _ {\boldsymbol {v}} \sum_ {i = 1} ^ {n} \| \boldsymbol {v} - \boldsymbol {g} _ {i} \|. \tag {7}
$$

If the vectors  $\{\pmb{g}_1, \dots, \pmb{g}_n\}$  are drawn independently from the same distribution, intuitively most of them would concentrate around their mean. Then, even if there are some Byzantine outputs, the median would ignore those as outliers and output a 'central' point close to the mean.

However, when  $\{g_1,\dots ,g_n\}$  are gradients over heterogeneous data, they may be vastly different from each other and do not concentrate around the mean. In such a scenario, the median such as (7) can be even less robust than simply taking the mean. Suppose that worker 0 is Byzantine and the remaining workers  $\{1,\ldots ,2n\}$  are good, with a total of  $2n + 1$  workers. Now suppose that  $g_{i} = (-1)^{i}$  for all the workers, with half the good workers having  $-1$  and the other half  $+1$ . This means that the true mean is 0, however, the median estimator (7) will output 1.

Mimic attack. This motivates our mimic attack in which all Byzantine workers collude and agree to always send gradients from the same worker. We define a specialized attack, called mimic2, where half of the good workers have same datasets and send  $g_{1}$  while the rest good workers send  $g_{2}$ ; then all Byzantine workers send  $v = g_{1}$  such that the geometric median of the gradients received by the server is always  $g_{1}$ . Therefore, this attack breaks geometric-median-based robust aggregation rules, by leading them to wrong solutions. The left plot of Figure 1c shows the impact of the mimic2 attack. Test accuracies of CM and RFA both drop drastically to around  $50\%$ .

# Algorithm 1 Robust Learning with Resampling

Setup:  $n$  workers,  $f$  of which are Byzantine; resampling  $T$  times, each time samples  $s$  gradients. A robust learning algorithm AGGR on iid datasets;  $\gamma$  is the learning rate.

# Workers:

1. Each good worker  $i$  randomly samples a datapoint  $j$  and computes a stochastic gradient  $\pmb{g}_i \coloneqq \nabla F_i(\pmb{x}, \pmb{\xi}_i^j)$  where  $\pmb{\xi}_i^j \sim \mathcal{D}_i$ ; each Byzantine worker  $i$  sends arbitrary vector  $\pmb{g}_i$ .  
2. Send  $g_{i}$  to server.

# Servers:

1. Receive  $\{\pmb{g}_i\}_{i=1}^n$  from all workers.  
2.  $S, \mathcal{I}_S = \mathrm{Resampling}(\{\pmb{g}_i : i \in [n]\}, f, T, s)$ ; See Algorithm 2.  
3. Compute  $\pmb{x}^{\prime} \coloneqq \pmb{x} - \gamma \mathrm{AGGR}(\mathcal{S})$  
4. Broadcast  $x'$  to all workers.

# Algorithm 2 Resampling with  $s$ -replacement

Input:  $\{\pmb{g}_i:i\in [n]\} ,T:= n,s,\{c[i]:=0:i\in [n]\}$

for  $t\coloneqq 1,\ldots ,T$  do

for  $i\coloneqq 1,\ldots ,s$  do

while Select  $j_{i}\sim \mathrm{Uniform}([n])$  do

if  $c[j_i] < s$  then

$c[j_i] + = 1$

If  $c[j_i] == s$  Break;

Compute average  $\bar{g}_t\coloneqq \frac{1}{s}\sum_{i = 1}^{s}g_{j_i}$

Return  $\{\bar{\pmb{g}}_t:t\in [T]\}$ $\{j_i^t:t\in [T],i\in [s]\}$

# 4 ROBUST AGGREGATION ON NON-IID DATA

In Section 3 we have demonstrated how existing robust aggregation rules can fail in realistic non-iid scenarios, with and without attackers (Sections 3.2 and 3.3 and Section 3.1 respectively). To overcome this problem, we propose a simple new resampling-based aggregation rule for training, shown in Algorithm 1. More specifically, we choose  $s$ -resampling without replacement in Algorithm 2 where each gradient can be sampled at most  $s$  times. The key property of our rule is that after resampling, the resulting set of averaged gradients  $\{\bar{g}_t : t \in [T]\}$  are much more homogeneous (lower variance). Then these averaged gradients are fed to existing Byzantine robust aggregation schemes, such as KRUM, see Section 5. Given an existing aggregation rule AGGR, we denote by AGGR  $\circ$  Resampling the resulting new robust aggregation rule for non-iid input gradients.

In the following proposition, we list the desired properties of Algorithm 2

Proposition I. Given a population  $\{\pmb{g}_i : i \in [n]\} \subset \mathbb{R}^d$  of mean  $\pmb{\mu} \coloneqq \frac{1}{n} \sum_{i=1}^{n} \pmb{g}_i$  and variance  $\sigma^2 \coloneqq \frac{1}{n} \sum_{i=1}^{n} \| \pmb{g}_i - \pmb{\mu} \|^2$ , let  $\{\bar{\pmb{g}}_t : t \in [T]\}$  be the output of Algorithm 2 on  $\{\pmb{g}_i : i \in [n]\}$ . Then

- If there are no Byzantine workers, then  $\{\bar{\mathbf{g}}_t : t \in [T]\}$  are identically distributed

$$
\mathbb {E} \left[ \bar {\boldsymbol {g}} _ {t} \right] = \boldsymbol {\mu}, \operatorname {v a r} \left(\bar {\boldsymbol {g}} _ {t}\right) = \frac {n - 1}{s n - 1} \sigma^ {2} \quad \forall t \in [ T ] \tag {8}
$$

- If  $f$  of the  $n$  inputs are Byzantine, then at least  $T - sf$  gradients in  $\{\bar{g}_t : t \in [T]\}$  are good; that is, a good  $\bar{g}_t$  is the average of gradients  $\{g_{j_i^t} : i \in [s]\} \subset \mathbf{good} \subset [n]$ . Then such good  $\{\bar{g}_t\}$  are identically distributed with

$$
\mathbb {E} \left[ \bar {\boldsymbol {g}} _ {t} \right] = \tilde {\boldsymbol {\mu}}, \operatorname {v a r} \left(\bar {\boldsymbol {g}} _ {t}\right) = \frac {n - 1}{s n - 1} \tilde {\sigma} ^ {2} \tag {9}
$$

where  $\tilde{\mu} := \frac{1}{|good|} \sum_{i \in good} \pmb{g}_i$ , and  $\tilde{\sigma}^2 := \frac{1}{|good|} \sum_{i \in good} \| \pmb{g}_i - \mathbb{E}[\bar{\pmb{g}}_t] \|^2$ .

Proof. Since Algorithm 2 resamples  $s$  gradients to estimate a population of  $sn$  samples, we can use sampling theory (Middleton, 1988, Ch. Survey Sampling) to compute the sample mean

$$
\mathbb {E} \left[ \mathbf {R S} \left(\boldsymbol {g} _ {1}, \dots , \boldsymbol {g} _ {n}\right) \right] = \boldsymbol {\mu} \tag {10}
$$

and the sample variance

$$
\mathbb {E} \left[ \left(\mathbf {R S} \left(\boldsymbol {g} _ {1}, \dots , \boldsymbol {g} _ {n}\right) - \boldsymbol {\mu}\right) ^ {2} \right] = \frac {1}{s} \left(1 - \frac {s - 1}{s n - 1}\right) \sigma^ {2} = \frac {n - 1}{s n - 1} \sigma^ {2}. \tag {11}
$$

Since the gradients are sampled at most  $s$  times, at most  $sf$  out of the  $T$  gradients are affected by a Byzantine worker. Its mean and variance can be calculated in the same way shown above.

Remark 1. For  $s = 1$ , resampling simply becomes shuffling of the input elements, and  $var(\bar{g}_t) = \sigma^2$  is unchanged. For  $s > 1$ , the resampling scheme reduces the heterogeneity (variance) by approximately  $1 / s$ . Thus, increasing  $s$  leads to the resulting resampled gradients being a better estimator of the population mean, thus improving training convergence speed. On the other hand, increasing  $s$  also increases the number of resampled gradients which can be affected by a Byzantine worker. In particular, if  $f$  workers are Byzantine, then up to  $f$ s resampled gradients can be incorrect, which has to be taken into account by the employed robust aggregation rule. In practice, we found that using a small value  $s = 2$  was already sufficient to overcome heterogeneity.

Remark 2. A natural question to ask is what happens if we resample with replacement but do not limit on the number of replacements. We discuss this additional algorithm variant in Appendix C.

Note that the  $\{\bar{g}_t:t\in [T]\}$  are identically distributed but not independent. This does not directly fit into the original assumptions of Byzantine robust algorithms like KRUM and hence the robustness has to be reproved for our more general setting.

# 5 CONVERGENCE ANALYSIS WITH KRUM

In this section, we analyze the convergence of SGD with robust aggregation on non-iid data. Since the definition of robustness and other conditions vary from paper to paper, it is not possible to give a uniform proof perfectly fit for all methods. For example, (Yin et al., 2018a) assumes the gradients have bounded variance and skewness whereas others like KRUM, RFA, BULYAN does not. Thus we only analyze KRUM for its simplicity and popularity, and show that analysis is only slightly different from the original version. For other algorithms, we show by experiments that resampling helps them achieve better performance on heterogeneous data, see Section 6.

Definition A generalizes the Byzantine resilience of (Blanchard et al., 2017, Definition 1) to the cases where we have non-iid data. Let  $G$  be an estimator of the good gradients.

Definition A  $((\alpha, f)$ -Byzantine Resilience.). Let  $0 \leq \alpha < \pi/2$  be any angular value, and any integer  $0 \leq f \leq n$ . Let  $\mathcal{B} = \{j_1, \ldots, j_f : j_1 \leq j_1 < \dots < j_f \leq n\}$  be the indices of Byzantine workers. Let  $\{V_i \in \mathcal{D}_i : i \in [n] \backslash \mathcal{B}\}$  be independent random vectors in  $\mathbb{R}^d$ . Let  $G = G(\pmb{\xi})$  be an independent random variable which randomly selects a good worker  $i$  and samples a vector from  $\mathcal{D}_i$  and  $\mathbb{E}G = g$ . Let  $B_1, \ldots, B_f$  be any Byzantine vectors in  $\mathbb{R}^d$ , possibly dependent on the  $V_i$ 's. An aggregation rule  $F$  is said to be  $(\alpha, f)$ -Byzantine resilient if

$$
F = F (V _ {1}, \ldots , \underbrace {B _ {1}} _ {j _ {1}}, \ldots , \underbrace {B _ {f}} _ {j _ {f}}, \ldots V _ {n})
$$

satisfies (i)  $\langle \mathbb{E}F,\pmb {g}\rangle \geq (1 - \sin \alpha)\cdot \| \pmb {g}\| ^2 >0$  and (ii) for  $r = 2,3,4,\mathbb{E}\| F\| ^r\leq \mathbb{E}\| G\| ^r$

Then we can conclude the almost sure convergence similar to (Blanchard et al., 2017, Proposition 2)

Theorem II (Resampling Krum). We assume that (i) the cost function  $\mathcal{L}$  is three times differentiable with continuous derivatives and non-negative  $\mathcal{L}(\pmb{x}) \geq 0$ ; (ii) the learning rates satisfy  $\sum_{t} \gamma_{t} = \infty$  and  $\sum_{t} \gamma_{t}^{2} < \infty$ . Let the good workers have stochastic gradients  $G_{i}(\pmb{x}, \pmb{\xi})$  for  $i \in \mathbf{good} \subset [n]$ . We assume that for a uniformly chosen  $j \in \mathbf{good}$ , the following is true (iii)  $\mathbb{E}_{j, \pmb{\xi}}[G_{j}(\pmb{x}, \pmb{\xi})] = \nabla \mathcal{L}(\pmb{x})$  and  $\forall r \in \{2, 3, 4\}$ ,  $\mathbb{E}_{j, \pmb{\xi}} \| G_{j}(\pmb{x}, \pmb{\xi}) \|^{r} \leq A_{r} + B_{r} \| \pmb{x} \|^{r}$  for some constants  $A_{r}$ ,  $B_{r}$ ; (iv) there exists a constant  $0 \leq \alpha < \pi/2$  such that for all  $\pmb{x}$  we have  $\eta(T, sf) \cdot \sqrt{d} \cdot \sigma(\pmb{x}) \leq \| \nabla \mathcal{L}(\pmb{x}) \| \cdot \sin \alpha$  where  $\sigma^{2}(\pmb{x}) := \frac{n-1}{sn-1} \tilde{\sigma}^{2}(\pmb{x})$ ; (v) finally, beyond a certain horizon,  $\| \pmb{x} \|^{2} \geq D$ , there exist  $\varepsilon > 0$  and  $0 \leq \beta < \pi/2 - \alpha$  such that  $\| \nabla \mathcal{L}(\pmb{x}) \| \geq \varepsilon > 0$  and  $\langle \pmb{x}, \nabla \mathcal{L}(\pmb{x}) \rangle \geq \cos \beta \| \pmb{x} \| \cdot \| \nabla \mathcal{L}(\pmb{x}) \|$ . If  $s > 1$  and  $2sf + 3 \leq n$ , then

- KRUM oResampling is  $(\alpha, sf)$ -Byzantine resilient where  $0 \leq \alpha < \pi/2$  is defined by

$$
\sin \alpha = \frac {\eta (T , s f) \cdot \sqrt {d} \cdot \sigma}{\| \nabla \mathcal {L} (\boldsymbol {x}) \|}, \eta (n, f) := \sqrt {2 \left(n - f + \frac {f \cdot (n - f - 2) + f ^ {2} \cdot (n - f - 1)}{n - 2 f - 2}\right)} \tag {12}
$$

- the sequence of gradients  $\nabla \mathcal{L}(\pmb{x}_t)$  converges almost surely to zero.

We defer the proof to Appendix A. The above convergence result for heterogeneous data is nearly identical to (Blanchard et al., 2017, Proposition 2) for iid data, except for the slightly stronger restriction on the number of Byzantine workers  $2sf + 3 \leq n$ .

![](images/2d59c27d660a21d9e342f1c21ad7daa7d42a51f1484de80635ab55f2a19030a7.jpg)

![](images/4b01dd2a259a1a721daface81aa99cf5eb39318f455ddc8baa8c8eeef17a2cc5.jpg)  
(b) Comparing normalized mean (RFA with  $\mathrm{T} = 1$ ) under the normalized mean attack with  $f = 0, 1, 2$  attackers.

![](images/ea3776520d84628a91cc3803abbd883798354b253c7a8b0cabdbd9ba2fcbe728.jpg)  
(a) Left & middle: Comparing arithmetic mean with Krum on iid and non-iid datasets, without any Byzantine workers. Right: Histogram of selected gradients.  
Figure 2: Combining resampling with existing aggregation rules on non-id MNIST dataset. In all experiments, there are 8 good and  $f$  Byzantine workers. For each aggregation we resample and average  $s$  gradients for  $T = n$  times.  
(c) Comparing coordinate-wise median (CM) and geometric median (RFA with  $T = 8$ ) under mimic2 attack on iid and non-iid datasets.

# 6 EXPERIMENTS

In this section, we demonstrate the effect of resampling on datasets distributed in a non-iid fashion. Throughout the section, we illustrate the challenge, attacks, and defense by an example of training an MLP on the MNIST dataset (LeCun et al., 1998). In Appendix D, we present the results of similar experiments on Fashion-MNIST (Xiao et al., 2017) and CIFAR-10 (Krizhevsky et al., 2009). The dataset is sorted by labels and sequentially divided into equal parts among good workers; Byzantine workers have access to the dataset on all good workers. Implementations are based on PyTorch (Paszke et al., 2019) and will be made publicly available.

# 6.1 RESAMPLING AGAINST THE ATTACKS ON NON-IID DATA

In Section 3 we have presented how heterogeneous data can lead to failure of existing robust aggregation rules. Here we apply our proposed resampling with  $T = n$ ,  $s = 2$  to the same aggregation rules, showing that resampling overcomes the described failures. Results are presented in Figure 2. In Figure 2a, we show that using resampling helps KRUM to achieve better test accuracy on non-iid data. Since resampling KRUM with  $s = 2$  actually averages 2 gradients, we compare it with MULTIKRUM with  $m = 2$ . The middle of Figure 2a shows that MULTIKRUM with  $m = 2$  performs better than KRUM, but KRUM with resampling is even better which suggests the resampling step improves the performance on non-iid data. The selection histogram on the rightmost part of Figure 2a shows that after resampling, KRUM's selection is much more evenly distributed between the good workers. In Figure 2b, we show that resampling fixes RFA with  $T = 1$  and allows it to defend against the normalized mean attack. The resampling-based aggregation can almost reach same accuracy for both iid and non-iid setup. In Figure 2c, while mimic attack does not work for median-based rules in the iid setting, resampling still slightly improves the performance due to variance reduction. In the non-iid setting, resampling drastically improves the accuracy to the same level as the iid setting.

# 6.2 RESAMPLING AGAINST GENERAL BYZANTINE ATTACKS

In Figure 3, we present thorough experiments on non-iid data over 12 workers with 2 Byzantine workers. In each subfigure, we compare an aggregation rule with its variant with resampling. Three aggregation rules are compared: KRUM, CM, RFA. In particular, we compare to RFA with both  $T = 1$  (normalized mean) and  $T = 8$  (geometric median).

![](images/28021419aeb0c1d8bb4231c2e666d183561094b876be164c7e133e8b0cc46628.jpg)  
Figure 3: Test accuracies of KRUM, CM, RFA under 5 kinds of attacks (and without attack) on non-iid datasets. There are 12 workers and 2 of them are Byzantine according to each attack row. Columns show each aggregation rule applied without (red) and with resampling (blue). Dotted lines for comparison are showing the same method without any Byzantine workers ( $f = 0$ ). For RFA, T1, T8 refers to the number of inner iterations of Weiszfeld's algorithm.

Attacks. 5 different kinds of attacks are applied (one per row in the figure): bitflipping, labelflipping, gaussian attack, as well as the mimic and mimic2 attacks.

- Bitflipping: A Byzantine worker flips the sign bits and sends  $-\nabla f(\pmb{x})$  instead of  $\nabla f(\pmb{x})$  because of problems like hardware failures etc.  
- Labelflipping: The dataset on workers have corrupted labels. For the MNIST dataset, we let Byzantine workers transform labels by  $\mathcal{T}(y) \coloneqq 9 - y$ .  
- Gaussian: A Byzantine worker sends a Gaussian random vector of 0 mean and isotropic covariance matrix with standard deviation 200 (Xie et al., 2018).  
- mimic & mimic2: Explained in Section 3.3.

From Figure 3 we can see that resampling improves the accuracy on most of the tasks. The final accuracies achieved vary with the aggregation rules we use. Notice that RFA-T1 is more robust to the mimic attack than RFA-T8 in Figure 3 because more inner iterations lead to better approximate geometric median and less robust to normalized mean attacks. The normalized mean attack has been addressed in Section 3.2.

# 7 CONCLUSION

In this paper, we initiated a study of robust distributed learning problem under realistic heterogeneous data. We showed that many existing Byzantine-robust aggregation rules fail under simple new attacks, or sometimes even without any Byzantine workers. As a solution, we propose a resampling scheme which effectively adapts existing robust algorithms to heterogeneous datasets at a negligible computational cost. We believe robustness under heterogeneous conditions has been an overlooked direction of research thus far and hope to inspire more work on this topic. Extending to the decentralized setting, stronger Byzantine adversaries, as well as obtaining optimal algorithms are other challenging directions for future work.

# REFERENCES

Dan Alistarh, Zeyuan Allen-Zhu, and Jerry Li. Byzantine stochastic gradient descent. In NeurIPS - Advances in Neural Information Processing Systems, pp. 4613-4623, 2018.  
Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, and Vitaly Shmatikov. How to backdoor federated learning, 2018.  
Moran Baruch, Gilad Baruch, and Yoav Goldberg. A little is enough: Circumventing defenses for distributed learning. arXiv preprint arXiv:1902.06156, 2019.  
Jeremy Bernstein, Jiawei Zhao, Kamyar Azizzadenesheli, and Anima Anandkumar. signSGD with majority vote is communication efficient and fault tolerant. arXiv preprint arXiv:1810.05291, 2018.  
Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal, and Seraphin Calo. Analyzing federated learning through an adversarial lens, 2018.  
Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. In NeurIPS - Advances in Neural Information Processing Systems 30, pp. 119-129, 2017.  
Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konecny, Stefano Mazzocchi, H Brendan McMahan, et al. Towards federated learning at scale: System design. In SysML - Proceedings of the 2nd SysML Conference, Palo Alto, CA, USA, 2019.  
Lingjiao Chen, Hongyi Wang, Zachary Charles, and Dimitris Papailiopoulos. Draco: Byzantine-resilient distributed training via redundant gradients. arXiv preprint arXiv:1803.09877, 2018.  
Yudong Chen, Lili Su, and Jiaming Xu. Distributed statistical machine learning in adversarial settings. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 1(2):1-25, Dec 2017. ISSN 2476-1249. doi: 10.1145/3154503. URL http://dx.doi.org/10.1145/3154503.  
Georgios Damaskinos, El Mahdi El Mhamdi, Rachid Guerraoui, Arsany Hany Abdelmessih Guirguis, and Sébastien Louis Alexandre Rouault. Aggregathor: Byzantine machine learning via robust gradient aggregation. Conference on Systems and Machine Learning (SysML) 2019, Stanford, CA, USA, pp. 19, 2019. URL http://infoscience.epfl.ch/record/265684.  
Deepesh Data and Suhas Diggavi. Byzantine-resilient sgd in high dimensions on heterogeneous data. arXiv preprint arXiv:2005.07866, 2020.  
Ilias Diakonikolas, Gautam Kamath, Daniel Kane, Jerry Li, Ankur Moitra, and Alistair Stewart. Robust estimators in high-dimensions without the computational intractability. SIAM Journal on Computing, 48(2):742-864, 2019.  
Avishek Ghosh, Justin Hong, Dong Yin, and Kannan Ramchandran. Robust federated learning in a heterogeneous environment. arXiv preprint arXiv:1906.06629, 2019.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015.  
Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Keith Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, Rafael G. L. D'Oliveira, Salim El Rouayheb, David Evans, Josh Gardner, Zachary Garrett, Adrià Gascon, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaid Harchaoui, Chaoyang He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara Javidi, Gauri Joshi, Mikhail Khodak, Jakub Konecný, Aleksandra Korolova, Farinaz Koushanfar, Sanmi Koyejo, Tancrede Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock, Ayfer Özgür, Rasmus Pagh, Mariana Raykova, Hang Qi, Daniel Ramage, Ramesh Raskar, Dawn Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tramér, Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu, and Sen Zhao. Advances and open problems in federated learning. arXiv preprint arXiv:1912.04977, 2019.

Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian U Stich, and Martin Jaggi. Error Feedback Fixes SignSGD and other Gradient Compression Schemes. In ICML 2019 - Proceedings of the 36th International Conference on Machine Learning, 2019.  
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
Kevin A Lai, Anup B Rao, and Santosh Vempala. Agnostic estimation of mean and covariance. In 2016 IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS), pp. 665-674. IEEE, 2016.  
Leslie Lamport, Robert Shostak, and Marshall Pease. The byzantine generals problem. In Concurrency: the Works of Leslie Lamport, pp. 203-226. 2019.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Liping Li, Wei Xu, Tianyi Chen, Georgios B Giannakis, and Qing Ling. RSA: Byzantine-robust stochastic aggregation methods for distributed learning from heterogeneous datasets. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 1544–1551, 2019.  
H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Areas. Communication-efficient learning of deep networks from decentralized data. arXiv preprint arXiv:1602.05629, 2016.  
Edward Meeds, Remco Hendriks, Said Al Faraby, Magiel Bruntink, and Max Welling. Mlith: machine learning in the browser. PeerJ Computer Science, 1:e11, Jul 2015. ISSN 2376-5992. doi: 10.7717/peerj-cs.11. URL http://dx.doi.org/10.7717/peerj-cs.11.  
El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault. The hidden vulnerability of distributed learning in byzantium. arXiv preprint arXiv:1802.07927, 2018.  
D Middleton. Mathematical statistics and data analysis, by john a. rice. pp 595.1988. ISBN 0-534-08247-5 (wadsworth & brooks/cole). The Mathematical Gazette, 72(462):330-331, 1988.  
Ken Miura and Tatsuya Harada. Implementation of a practical distributed calculation system with browsers and javascript, and application to distributed deep learning. arXiv preprint arXiv:1503.05743, 2015.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, pp. 8024-8035, 2019.  
Jie Peng and Qing Ling. Byzantine-robust decentralized stochastic optimization. In ICASSP 2020 - IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 5935-5939. IEEE, 2020.  
Krishna Pillutla, Sham M. Kakade, and Zaid Harchaoui. Robust Aggregation for Federated Learning. arXiv preprint arXiv:1912.13445, 2019.  
Shashank Rajput, Hongyi Wang, Zachary Charles, and Dimitris Papailiopoulos. Detox: A redundancy-based framework for faster and more robust gradient aggregation. arXiv preprint arXiv:1907.12205, 2019.  
F. Sattler, K. Müller, T. Wiegand, and W. Samek. On the byzantine robustness of clustered federated learning. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 8861-8865, 2020.  
Lili Su and Jiaming Xu. Securing distributed gradient descent in high dimensional statistical learning. arXiv preprint arXiv:1804.10140, 2018.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017.

Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Generalized Byzantine-tolerant SGD. arXiv preprint arXiv:1802.10116, 2018.  
Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Zeno: Distributed stochastic gradient descent with suspicion-based fault-tolerance. In ICML 2019 - 35th International Conference on Machine Learning, 2019a.  
Cong Xie, Sanmi Koyejo, and Indranil Gupta. Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation. arXiv preprint arXiv:1903.03936, 2019b.  
Zhixiong Yang and Waheed U Bajwa. Bridge: Byzantine-resilient decentralized gradient descent. arXiv preprint arXiv:1908.08098, 2019a.  
Zhixiong Yang and Waheed U Bajwa. ByRDiE: Byzantine-resilient distributed coordinate descent for decentralized learning. IEEE Transactions on Signal and Information Processing over Networks, 2019b.  
Dong Yin, Yudong Chen, Kannan Ramchandran, and Peter Bartlett. Byzantine-robust distributed learning: Towards optimal statistical rates. arXiv preprint arXiv:1803.01498, 2018a.  
Dong Yin, Yudong Chen, Kannan Ramchandran, and Peter Bartlett. Defending against saddle point attack in byzantine-robust distributed learning. arXiv preprint arXiv:1806.05358, 2018b.
