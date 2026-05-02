# BYZANTINE-RESILIENT NON-CONVEX STOCHASTIC GRADIENT DESCENT

# ABSTRACT

We study adversary-resilient stochastic distributed optimization, in which  $m$  machines can independently compute stochastic gradients, and cooperate to jointly optimize over their local objective functions. However, an  $\alpha$ -fraction of the machines are Byzantine, in that they may behave in arbitrary, adversarial ways. We consider a variant of this procedure in the challenging non-convex case. Our main result is a new algorithm SafeguardSGD which can provably escape saddle points and find approximate local minima of the non-convex objective. The algorithm is based on a new concentration filtering technique, and its sample and time complexity bounds match the best known theoretical bounds in the stochastic, distributed setting when no Byzantine machines are present. Our algorithm is practical: it improves upon the performance of prior methods when training deep neural networks, it is relatively lightweight, and it is the first method to withstand two recently-proposed Byzantine attacks.

# 1 INTRODUCTION

Motivated by the pervasiveness of large-scale distributed machine learning, there has recently been significant interest in providing distributed optimization algorithms with strong fault-tolerance guarantees. In this context, the strongest, most stringent fault model is that of Byzantine faults (Lamport et al., 1982): given  $m$  machines, each having access to private data, at most an  $\alpha$  fraction of the machines can behave in arbitrary, possibly adversarial ways, with the goal of breaking or at least slowing down the algorithm. Although extremely harsh, this fault model is the "gold standard" in distributed computing (Lynch, 1996; Lamport et al., 1982; Castro et al., 1999), as algorithms proven to be correct in this setting are guaranteed to converge under arbitrary system behaviour.

A setting of particular interest in this context has been that of distributed stochastic optimization. Here, the task is to minimize some stochastic function  $f(x) = \mathbb{E}_{s\sim \mathcal{D}}[f_s(x)]$  over a distribution  $\mathcal{D}$ , where  $f_{s}(\cdot)$  can be viewed as the loss function for sample  $s\sim \mathcal{D}$ . We assume there are  $m$  machines (workers) and an honest master, and  $\alpha < 1 / 2$  fraction of the workers may be Byzantine. In each iteration  $t$ , each worker has access to a version of the global iterate  $x_{t}$ , which is maintained by the master. The worker can independently sample  $s\sim \mathcal{D}$ , compute  $\nabla f_{s}(x_{t})$ , and then synchronously send this stochastic gradient to the master. The master aggregates the workers' messages, and sends an updated iterate  $x_{t + 1}$  to all the workers. Eventually, the master has to output an approximate minimizer of  $f$ . Clearly, the above description only applies to honest workers; Byzantine workers may deviate arbitrarily and return adversarial "gradient" vectors to the master in every iteration.

This distributed framework is quite general and well studied. One of the first references in this setting studied distributed PCA and regression (Feng et al., 2014). Other early approaches (Blanchard et al., 2017; Chen et al., 2017; Su and Vaidya, 2016a;b; Xie et al., 2018a) relied on defining generalizations of the geometric median. These approaches can withstand up to half of the nodes being malicious, but can have relatively high local computational cost  $\Omega(m^2 d)$  (Blanchard et al., 2017; Chen et al., 2017), and usually have sub-optimal sample and iteration complexity.

Follow-up work resolved this last issue when the objective  $f(\cdot)$  is convex, leading to tight sample complexity bounds. Specifically, Yin et al. (2018) provided bounds for gradient descent-type algorithms, and showed that the bounds are tight when the dimension is constant. Alistarh et al. (2018) provided a stochastic gradient descent (SGD) type algorithm and showed that its sample and time complexities are asymptotically optimal even when the dimension is large.

Non-convex Byzantine-resilient stochastic optimization. In this paper, we focus on the more challenging non-convex setting, and shoot for the strong goal of finding approximate local minima (a.k.a. second-order critical points). In a nutshell, our main result is the following. Assume the objective  $f: \mathbb{R}^d \to \mathbb{R}$  is Lipschitz smooth and second-order smooth. We have  $m$  worker machines, each having access to unbiased, bounded estimators of the gradient of  $f$ . Our SafeguardSGD algo

rithm ensures that, even if at most  $\alpha < 1/2$  fraction of the machines are Byzantine, after

$$
T = \widetilde {O} \left(\left(\alpha^ {2} + \frac {1}{m}\right) \frac {d (f (x _ {0}) - \min  f (x))}{\varepsilon^ {4}}\right) \quad \text {p a r a l l e t i r a t i o n s},
$$

for at least a constant fraction of the indices  $t \in [T]$ , the following hold:

$$
\| \nabla f (x _ {t}) \| \leq \varepsilon \quad \text {a n d} \quad \nabla^ {2} f (x _ {t}) \succeq - \sqrt {\varepsilon} \mathbf {I}.
$$

(If the goal is simply  $\| \nabla f(x_{t})\| \leq \varepsilon$ , then  $T = \widetilde{O}\big(\big(\alpha^{2} + \frac{1}{m}\big)\frac{(f(x_{0}) - \min f(x))}{\varepsilon^{4}}\big)$  iterations suffice.)

- When  $\alpha < 1/\sqrt{m}$ , our sample complexity ( $= mT$ ) matches the best known result in the non-Byzantine case (Jin et al., 2019) without additional assumptions, and enjoys linear parallel speed-up: with  $m$  workers of which  $< \sqrt{m}$  are Byzantine, the parallel speedup is  $\widetilde{\Omega}(m)$ .  
- For  $\alpha \in [1/\sqrt{m}, 1/2)$ , our parallel time complexity is  $\widetilde{O}(\alpha^2)$  times that needed when no parallelism is used. This still gives parallel speedup. This  $\alpha^2$  factor appears in convex Byzantine distributed optimization, where it is tight (Yin et al., 2018; Alistarh et al., 2018).  
- The Lipschitz smoothness and second-order smoothness assumptions are the minimal assumptions needed to derive convergence rates for finding second-order critical points (Jin et al., 2019).

Comparison with prior bounds. The closest known bounds are by Yin et al. (2019), who derived three gradient descent-type of algorithms (based on median, mean, and iterative filtering) to find a weaker type of approximate local minima. Since it relies on full gradients, their algorithm is arguably less practical, and their time complexities are generally higher than ours (see Section 2.1).

Other prior works consider a weaker goal: to find approximate stationary points  $\| \nabla f(x)\| \leq \varepsilon$  only: Bulusu et al. (2020) additionally assumed there is a guaranteed good (i.e. non-Byzantine) worker known by the master, Xie et al. (2018b) gave a practical algorithm when the Byzantine attackers have no information about the loss function or its gradient, Yang et al. (2019); Xie et al. (2018a); Blanchard et al. (2017) derived eventual convergence without an explicit complexity bound, and the non-convex result obtained in Yin et al. (2018) is subsumed by Yin et al. (2019), discussed above.

Our algorithm and techniques. The structure of our algorithm is deceptively simple. The master node keeps track of the sum of gradients produced by each worker across time. It labels (allegedly) good workers as those whose sum of gradients "concentrate" well with respect to a surrogate of the median vector, and labels bad workers otherwise. Once a worker is labelled bad, it is removed from consideration forever. The master then performs the vanilla SGD, by moving in the negative direction of the average gradients produced by those workers currently labelled as good.

We call our algorithm SafeguardSGD, since it behaves like having a safe guard to filter away bad workers. Its processing overhead at the master is  $O(md)$ , negligible compared to standard SGD.

As the astute reader may have guessed, the key non-trivial technical ingredient is to identify the right quantity to check for concentration, and make it compatible with the task of non-convex optimization. In particular, we manage to construct such quantities so that (1) good non-Byzantine workers never get mislabelled as bad ones; (2) Byzantine workers may be labelled as good ones (which is inevitable) but when they do, the convergence rates are not impacted significantly; and (3) the notion does not require additional assumptions or running time overhead. To achieve these goals, we leverage a new concentration lemma that may be of independent interest (see Section 3).

The idea of using concentration (for each worker across time) to filter out Byzantine machines traces back to the convex setting (Alistarh et al., 2018). However, the quantities used in (Alistarh et al., 2018) to check for concentration are necessarily different from this paper, and our analysis is completely new, as deriving non-convex rates is known to be much more delicate and challenging. Recently, Bulusu et al. (2020) used similar concentration filters to Alistarh et al. (2018) in the non-convex setting, but under stronger assumptions, and for the simpler task of finding stationary points.

Many other algorithms do not rely on concentration filters. In each iteration, they ask each worker to compute a batch of stochastic gradients, and then use coordinate-wise median or mean over the batch average (e.g. Yin et al. (2018; 2019); Yang et al. (2019)) or iterative filtering (e.g. Su and Xu (2018); Yin et al. (2019)) by the master to derive a "robust mean." These works fundamentally rely on each iteration to calculate an almost precise full gradient, so that they can apply a surrogate of full gradient descent. Such algorithms can introduce higher sample and time complexities (see Section 2), are less practical than stochastic gradient schemes, require additional restrictions on the resilience factor  $\alpha$ , e.g.  $\alpha < 1/4$  (Su and Xu, 2018), and, critically, have been shown to be vulnerable to recent attacks (Baruch et al., 2019; Xie et al., 2020).

Attack resilience and experimental validation. There is a growing literature on customized attacks against Byzantine-resilient algorithms, showing that many defenses can be entirely circumvented in real-world scenarios (Baruch et al., 2019; Xie et al., 2020). Our algorithm is provably correct against these attacks, a fact we also validate experimentally. Specifically, we implemented SafeguardSGD to examine its practical performance against a range of prior works (Xie et al., 2018b; Blanchard et al., 2017; Chen et al., 2017; Yin et al., 2018; 2019), and against recent attacks on the distributed task of training deep neural networks. Our experiments show that SafeguardSGD generally outperforms previous methods in convergence speed and final accuracy, sometimes by a wide accuracy margin. This is true not only against known Byzantine attacks, but also against attack variants we fine-crafted to specifically slow down our algorithm. SafeguardSGD provides two desirable properties: (1) SafeguardSGD quickly detects nodes which significantly misbehave; (2) the impact of the remaining Byzantine workers (with undetectable deviations) is limited in practice.

# 2 STATEMENT OF OUR THEORETICAL RESULT

We denote by  $\| \cdot \|$  the Euclidean norm and  $[n] \coloneqq \{1,2,\dots ,n\}$ . Given symmetric matrices  $\mathbf{A}, \mathbf{B}$ , we let  $\| \mathbf{A}\|_2$  denote the spectral norm of  $\mathbf{A}$ . We use  $\succeq$  to denote Lower ordering, i.e.  $\mathbf{A} \succeq \mathbf{B}$  if  $\mathbf{A} - \mathbf{B}$  is positive semi-definite. We denote by  $\lambda_{\min}(\mathbf{A})$  the minimum eigenvalue of matrix  $\mathbf{A}$ .

We consider arbitrary non-convex functions  $f\colon \mathbb{R}^d\to \mathbb{R}$  satisfying the following:

-  $f(x)$  is  $L$ -Lipschitz smooth: meaning  $\| \nabla f(x) - \nabla f(y) \| \leq L \| x - y \|$  for any  $x, y \in \mathbb{R}^d$ ;  
-  $f(x)$  is  $L_{2}$ -second-order smooth:  $\| \nabla^2 f(x) - \nabla^2 f(y) \|_2 \leq L_2 \cdot \| x - y \|$  for any  $x, y \in \mathbb{R}^d$ ;

For notational simplicity of the proofs, we assume  $L = L_{2} = \mathcal{V} = 1$ . Note that we have also assumed the domain of  $f$  is the entire space  $\mathbb{R}^d$ . If instead there is a compact domain  $\mathcal{X}\subset \mathbb{R}^d$ , then one can use projected SGD and re-derive similar results of this paper. We choose to present our result in the simplest setting to convey our main ideas.

Byzantine non-convex stochastic distributed optimization. We let  $m$  be the number of worker machines and assume at most an  $\alpha$  fraction of them are Byzantine for  $\alpha \in \left[0, \frac{1}{2}\right)$ . We denote by good  $\subseteq [m]$  the set of good (i.e. non-Byzantine) machines, and the algorithm does not know good.

Assumption 2.1. In each iteration  $t$ , the algorithm (on the master) is allowed to specify a point  $x_{t}$  and query  $m$  machines. Each machine  $i \in [m]$  gives back a vector  $\nabla_{t,i} \in \mathbb{R}^d$  satisfying

- If  $i \in \mathrm{good}$ , the stochastic gradient  $\nabla_{t,i}$  satisfies  $\mathbb{E}[\nabla_{t,i}] = \nabla f(x_t)$  and  $\| \nabla f(x_t) - \nabla_{t,i} \| \leq \mathcal{V}$ .  
- If  $i \in [m] \backslash \mathrm{good}$ , then  $\nabla_{t,i}$  can be arbitrary (w.l.o.g. we assume  $\| \nabla f(x_t) - \nabla_{t,i} \| \leq \mathcal{V}$ ).<sup>2</sup>

Remark 2.2. For each  $t$  and  $i \notin \mathbb{G}$ , the vector  $\nabla_{t,i}$  can be adversarially chosen and may depend on  $\{\nabla_{t',i}\}_{t' \leq t, i \in [m]}$ . In particular, the Byzantine machines can even collude during an iteration.

# 2.1 OUR ALGORITHM AND THEOREM

Our algorithm is based on arguably the simplest possible method for achieving this goal, (perturbed) stochastic gradient descent (SGD) (Ge et al., 2015). Our techniques more broadly apply to more complicated methods (e.g. at least to Allen-Zhu (2018a;b)), but we choose to analyze the simplest SGD since it is the most widely applied method in modern non-convex machine learning.

As illustrated in Algorithm 1, in each iteration  $t = 0, 1, \dots, T - 1$ , we maintain a set of (allegedly) good machines  $\mathrm{good}_t \subseteq [m]$ . We begin with  $\mathrm{good}_0 = [m]$  and start to detect malicious machines and remove them from the set. We choose a learning rate  $\eta > 0$ , and perform the SGD update

$$
x _ {t + 1} = x _ {t} + \xi_ {t} - \eta \frac {1}{\left| \mathbf {g o o d} _ {t} \right|} \sum_ {i \in \mathbf {g o o d} _ {t}} \nabla_ {t, i}
$$

where  $\xi_t \sim \mathcal{N}(0, \nu^2\mathbf{I})$  is a random Gaussian perturbation that we introduce.

For each machine  $i \in [m]$ , we keep track of the history of its stochastic gradients up to two windows. Namely,  $A_i \gets \sum_{k=last1}^{t} \frac{\nabla_{k,i}}{|\mathbf{good}_k|}$  and  $B_i \gets \sum_{k=last0}^{t} \frac{\nabla_{k,i}}{|\mathbf{good}_k|}$ , for windows sizes  $T_0 \leq T_1 \leq T$ . We

Algorithm 1 SafeguardSGD: perturbed SGD with double safe guard  
Input: point  $x_0\in \mathbb{R}^d$  , rate  $\eta >0$  , lengths  $T\geq T_{1}\geq T_{0}\geq 1$  , threshold  $\mathfrak{T}_1 > \mathfrak{T}_0 > 0$  .   
1: good  $\leftarrow [m]$  .   
2: for  $t\gets 0$  to  $T - 1$  do   
3: last1  $\leftarrow$  max{t1∈[t]: t1is a multiple of  $T_{1}\} ;$    
4: lasto  $\leftarrow$  max{t0∈[t]: t0is a multiple of  $T_{0}\}$    
5: for each i e good do   
6: receive  $\nabla_{t,i}\in \mathbb{R}^d$  from machine i;   
7:  $A_{i}\gets \sum_{k = last_{1}}^{t}\frac{\nabla_{k,i}}{|good_{k}|}$  and  $B_{i}\gets \sum_{k = last_{0}}^{t}\frac{\nabla_{k,i}}{|good_{k}|};$    
8:  $A_{med}\gets A_{i}$  where i e good is any machine s.t. {j e good: ||Aj-Ai||≤T} > m/2.   
9:  $B_{med}\leftarrow B_{i}$  where i e good is any machine s.t. {j e good: ||Bj-Bi||≤T} > m/2.   
10: good+1  $\leftarrow \{i\in \mathrm{good}_t:\| A_i - A_\mathrm{med}\| \leq 2\mathfrak{T}_1\wedge \| B_i - B_\mathrm{med}\| \leq 2\mathfrak{T}_0\}$  .   
11:  $x_{t + 1} = x_t - \eta (\xi_t + \frac{1}{|\mathrm{good}_t|}\sum_{i\in \mathrm{good}_t}\nabla_{t,i})$  . Gaussian noise  $\xi_t\sim \mathcal{N}(0,\nu^2\mathbf{I})$

compare among remaining machines in  $\mathrm{good}_t$ , and kick out those ones whose  $A_i$  or  $B_i$  deviate "more than usual" to construct  $\mathrm{good}_{t+1}$ . Conceptually, we view these two as safe guards.

Our theory makes sure that, when the "window sizes" and the thresholds for "more than usual" are defined properly, then  $\mathrm{good}_t$  shall always include good, and the algorithm shall proceed to find approximate local minima. Formally, we have (letting the  $\widetilde{O}$  notion to hide polylogarithmic factors)

Theorem 2.3. Let  $C_3 = \alpha^2 +\frac{1}{m}$ . Suppose we choose  $\nu^{2} = \widetilde{\Theta} (C_{3})$ ,  $\eta = \widetilde{\Theta} (\frac{\varepsilon^2}{dC_3})$ ,  $T_{0} = \widetilde{\Theta} (\frac{1}{\eta})$ ,  $T_{1} = \widetilde{\Theta} (\frac{1}{\eta\sqrt{\varepsilon}})$ ,  $\mathfrak{T}_0 = \widetilde{\Theta} (\sqrt{T_0})$ , and  $\mathfrak{T}_1 = \widetilde{\Theta} (\sqrt{T_1})$ , then after

$$
T = \widetilde {O} \left(\frac {(f (x _ {0}) - \min f (x)) d}{\varepsilon^ {4}} (\alpha^ {2} + \frac {1}{m})\right)
$$

iterations, with high probability, for at least constant fraction of the indices  $t \in [T]$ , they satisfy

$$
\| \nabla f (x _ {t}) \| \leq \varepsilon \quad a n d \quad \nabla^ {2} f (x _ {t}) \succeq - \sqrt {\varepsilon} \mathbf {I}.
$$

Remark 2.4. If one only wishes to achieve a significantly simpler goal — finding first-order critical points  $\| \nabla f(x_{t})\| \leq \varepsilon$  — the analysis becomes much easier (see Section 4). In particular, having one safe guard without perturbation (i.e.  $\nu = 0$ ) suffices, and the iteration complexity reduces to  $T = \widetilde{O}\left(\frac{f(x_0) - \min f(x)}{\varepsilon^4} (\alpha^2 +\frac{1}{m})\right)$ . Bulusu et al. (2020) achieves this easier goal but requires an additional assumption: there is one guaranteed good worker known by the master.

Our contribution. We reiterate our theoretical contributions from three perspectives. 1) When  $\alpha < 1/\sqrt{m}$ , our algorithm requires  $mT = \widetilde{O}\left(\frac{(f(x_0) - \min f(x))d}{\varepsilon^4}\right)$  stochastic gradient computations. This matches the best known result (Jin et al., 2019) under our minimal assumptions of the non-convex objective. (There exist other works in the stochastic setting that break the  $\varepsilon^{-4}$  barrier and get rid of the dimension dependence  $d$  under stronger assumptions.)<sup>3</sup>. 2) When  $\alpha < 1/\sqrt{m}$ , our algorithm enjoys linear parallel speed-up: the parallel time complexity reduces by a factor of  $\Theta(m)$ . When  $\alpha \in [1/\sqrt{m}, 1/2)$ , our parallel time complexity is  $\widetilde{O}(\alpha^2)$  times that needed when no parallelism is used, still giving noticeable speedup. The  $\alpha^2$  factor also appeared in convex Byzantine distributed optimization (and is known to be tight there) (Yin et al., 2018; Alistarh et al., 2018).

Comparison to Yin et al. (2019). Yin et al. (2019) derived three gradient descent-type algorithms to find points with a weaker (and less standard) guarantee:  $\| \nabla f(x)\| \leq \varepsilon$  and  $\nabla^2 f(x)\succeq -(\varepsilon^2 d)^{1 / 5}\mathbf{I}$ . Despite practical differences (namely, gradient descent may be less favorable comparing to stochastic gradient descent especially in deep learning applications), the parallel time complexities derived from their result are also generally larger than ours.

Their paper focuses on bounding the number of sampled stochastic functions, as opposed to the number of stochastic gradient evaluations like we do in this paper. When translated to our language, each of the workers in their setting needs to evaluate  $T$  stochastic gradients, where (1)  $T = \widetilde{O}\left(\frac{\alpha^2d}{\varepsilon^4} +\right.$

$\frac{d^2}{\varepsilon^4m} + \frac{\sqrt{d}}{\varepsilon^3}$  if using coordinate-wise median, (2)  $T = \widetilde{O}\left(\frac{\alpha^2d^2}{\varepsilon^4} + \frac{d^2}{\varepsilon^4m}\right)$  if using trimmed mean, and (3)  $T = \widetilde{O}\left(\frac{\alpha}{\varepsilon^4} + \frac{d}{\varepsilon^4m}\right)$  if using iterative filtering. The complexities (1) and (2) are larger than ours (also with a weaker guarantee); the complexity (3) seems incomparable to ours, but when translating to the more standard  $(\varepsilon, \sqrt{\varepsilon})$  guarantee, becomes  $T = \widetilde{O}\left(\frac{\alpha d^2}{\varepsilon^5} + \frac{d^3}{\varepsilon^5m}\right)$  so is also larger than ours. It is worth noting that (3) requires  $\alpha < 1/4$  so cannot withstand half of the machines being Byzantine.

Resilience against practical attacks. Recall that our algorithm's filtering technique is based upon tracking  $B_{i}$ , the stochastic gradients of each machine  $i$  averaged over a window of  $T_{0}$  iterations. This is a departure from previous defenses, most of which are history-less, and enables us to be provably Byzantine-resilient even against recent attacks (Baruch et al., 2019; Xie et al., 2020).

Specifically, in Baruch et al. (2019), Byzantine workers collude to shift the gradient mean by a factor  $\beta$  times the standard deviation of the (true stochastic) gradient, while staying within population variance. They noticed  $\beta$  can be quite large in practice, since stochastic gradients tend to have large variance, especially in neural network training. Their attack can circumvent existing defenses because those defense algorithms are "historyless", while their attack is statistically indistinguishable from an honest execution in a single iteration. However, our algorithm can provably defend against this attack since it has memory: Byzantine workers following their strategy will progressively diverge from the (honest) "median"  $B_{\mathrm{med}}$  (by an amount proportional to  $\Omega(T)$  in  $T$  iterations), and thus be marked as malicious by our algorithm (since our safeguard threshold is proportional to  $O(\sqrt{T})$ ). Alternatively, if the Byzantine workers attempt to disrupt the mean while staying within our algorithm's thresholds, then we prove their influence on convergence must be negligible. In Xie et al. (2020), Byzantine workers collude to deviate in the negative direction of the gradient. Similarly to the previous attack, to avoid being caught by our algorithm, the maximum "magnitude" of this attack again has to stay within our thresholds. We implemented both attacks and showed our algorithm's robustness experimentally.

# 3 CONCENTRATION OF MEASURE FOR SUMS OF PARTIAL SUMS

We present a novel concentration which is critical to our analysis and may be of independent interest.

Lemma 3.1. Suppose  $\xi_0, \ldots, \xi_{T-1} \sim \mathcal{N}(0,1)$  are i.i.d. and  $\Delta_1, \ldots, \Delta_{T-1} \in \mathbb{R}$ . Here, each  $\Delta_t$  can depend on  $\xi_0, \ldots, \xi_{t-1}$  but not on  $\xi_t, \ldots, \xi_{T-1}$ . Suppose  $\Delta$  satisfies  $|\Delta_1 + \dots + \Delta_t|^2 \leq \mathfrak{T}$  for every  $t = 1, \ldots, T-1$ . Then, with probability at least  $1 - p$ ,

$$
\left| \sum_ {t = 1} ^ {T - 1} \left(\xi_ {0} + \dots + \xi_ {t - 1}\right) \cdot \Delta_ {t} \right| \leq O \left(\sqrt {T \mathfrak {T} \log (T / p)}\right).
$$

(In our final convergence analysis of SafeguardSGD, we shall apply Lemma 3.1 by substituting each  $\xi_{t}$  with the Gaussian noise and each  $\Delta_t$  with the Byzantine attack in iteration  $t$ .)

The main difficulty in proving Lemma 3.1 comes from the "sum of partial sums." It is substantially harder than bounding  $\sum_{t} (\xi_0 + \dots + \xi_{t-1}) \cdot \Delta_T$  or  $(\sum_{t} \xi_t)(\sum_{t} \Delta_t)$ . In particular, if one decomposes the final quantity into the sum of  $O(\mathfrak{T})$  terms each bounded by  $O(\sqrt{T})$ , or the sum of  $O(T)$  terms each bounded by  $O(\sqrt{\mathfrak{T}})$ , then this cannot give the desired  $\widetilde{O}(\sqrt{T\mathfrak{T}})$  concentration.

Our proof instead writes this "sum of partial sums" as a certain weighted graph, and then decomposes the graph into essentially  $O(\sqrt{\mathfrak{T}})$  paths. We argue that each path can be bounded by  $O(\sqrt{T})$  with high probability, so this leads to the desired  $\widetilde{O} (\sqrt{T\mathfrak{T}})$  concentration. Details are in Appendix A.

We also present a similar concentration lemma for inner products over  $d$ -dimensional vectors.

Lemma 3.2. Suppose  $\xi_0,\ldots ,\xi_{T - 1}\in \mathbb{R}^d$  are i.i.d. drawn from  $\mathcal{N}(0,\mathbf{I})$  are  $\Delta_1,\dots ,\Delta_{T - 1}\in \mathbb{R}^d$  Here, each  $\Delta_t$  can depend on  $\xi_0,\ldots ,\xi_{t - 1}$  but not on  $\xi_t,\ldots ,\xi_{T - 1}$  . Suppose  $\Delta$  satisfies  $\| \Delta_1 + \dots +$ $\Delta_t\| ^2\leq \mathfrak{T}$  for every  $t = 1,\dots ,T - 1$  . Then, with probability at least  $1 - p$

$$
\left| \sum_ {t = 1} ^ {T - 1} \left\langle \xi_ {0} + \dots + \xi_ {t - 1}, \Delta_ {t} \right\rangle \right| \leq O \left(\sqrt {d T \mathfrak {T} \log (T / p)}\right).
$$

# 4 WARMUP: SINGLE SAFE GUARD

As a warmup, let us first analyze the behavior of perturbed SGD with a single safe guard. Consider Algorithm 2, where we start with a point  $w_0$ , a set  $\mathrm{good}_0 \supseteq$  good, and perform  $T$  steps of perturbed SGD. (We use the  $w_t$  sequence instead of the  $x_t$  sequence to emphasize that we are in Algorithm 2.)

Algorithm 2 Perturbed SGD with single safe guard (for analysis purpose only)  
Input: point  $w_0 \in \mathbb{R}^d$ , set  $\mathrm{good}_0 \supseteq$  good, rate  $\eta > 0$ , length  $T \geq 1$ , threshold  $\mathfrak{T} > 0$ ;  
1: for  $t \gets 0$  to  $T - 1$  do  
2: for each  $i \in \mathrm{good}_t$  do  
3: receive  $\nabla_{t,i} \in \mathbb{R}^d$  from machine  $i$ ;  
4:  $B_i \gets \sum_{k=0}^{t} \frac{\nabla_{k,i}}{|\mathrm{good}_k|}$ ;  
5:  $B_{\mathrm{med}} \gets B_i$  where  $i \in \mathrm{good}_t$  is any machine s.t.  $\left|\{j \in \mathrm{good}_t : \|B_j - B_i\| \leq \mathfrak{T}\}\right| > m/2$ .  
6:  $\mathrm{good}_{t+1} \gets \left\{i \in \mathrm{good}_t : \|B_i - B_{\mathrm{med}}\| \leq 2\mathfrak{T}\right\}$ ;  
7:  $w_{t+1} = w_t - \eta\left(\xi_t + \frac{1}{|\mathrm{good}_t|} \sum_{i \in \mathrm{good}_t} \nabla_{t,i}\right)$ ;  
 $\diamond$  Gaussian noise  $\xi_t \sim \mathcal{N}(0, \nu^2\mathbf{I})$

Definition 4.1. We make the following definition to simplify notations: let  $\Xi_t \coloneqq \sigma_t + \Delta_t$  where

$\sigma_{t}:= \frac{1}{|\mathbf{good}_{t}|}\sum_{i\in \mathbf{good}}\left(\nabla_{t,i} - \nabla f(w_{t})\right)$  
$\Delta_t \coloneqq \frac{1}{|\mathbf{good}_t|} \sum_{i \in \mathbf{good}_t \backslash \mathbf{good}} (\nabla_{t,i} - \nabla f(w_t))$

Therefore, we can re-write the SGD update as  $w_{t + 1} = w_t - \eta (\nabla f(w_t) + \xi_t + \Xi_t)$ .

The following lemma is almost trivial to prove:

Lemma 4.2 (single safe guard). In Algorithm 2, suppose we choose  $\mathfrak{T} = 8\sqrt{T\log(16mT / p)}$ . Then, with probability at least  $1 - p / 4$ , for every  $t = 0,\dots ,T - 1$

- good  $t \supseteq$  good.  
$\| \sigma_t\| ^2\leq O(\frac{\log(T / p)}{m})$  and  $\| \sigma_0 + \dots +\sigma_{t - 1}\| ^2\leq O(\frac{T\log(T / p)}{m})$  
$\| \Delta_t\| ^2\leq \alpha^2$  and  $\| \Delta_0 + \dots +\Delta_{t - 1}\| ^2\leq O(\alpha^2 T\log (mT / p))$  
-  $\left| \left\langle  {{\nabla f}\left( {w}_{t}\right) ,{\xi }_{t}}\right\rangle   \right|  \leq  \parallel {\nabla f}\left( {w}_{t}\right) \parallel  \cdot  O\left( {\nu \sqrt{\log \left( {T/p}\right) }}\right)$  ,  
-  $\| \xi_t\|^2 \leq O(\nu^2 d\log (T / p)), \| \xi_0 + \dots + \xi_{t - 1}\|^2 \leq O(\nu^2 dT\log (T / p))$

We call this probabilistic event  $\text{Event}_T^{\text{single}}(w_0)$  and  $\Pr[\text{Event}_T^{\text{single}}(w_0)] \geq 1 - p/4$ .

(The third property above is ensured by our choice of  $\mathfrak{T}$  and the use of safe guard, and the rest of the properties follow from simple martingale concentration arguments. Details are in Appendix B.1.)

Core Technical Lemma 1: Objective Decrease. Our first main technical lemma is the following:

Lemma 4.3. Suppose we choose  $\mathfrak{T}$  as in Lemma 4.2. Denote by  $C_1 = \log (T / p)$  and  $C_2 = \alpha^2\log \frac{mT}{p} +\frac{\log(T / p)}{m}$ . Suppose  $\eta \leq 0.01\min \{1,\frac{1}{C_2}\}$ ,  $T = \frac{1}{100\eta(1 + \sqrt{C_2})}$  and we start from  $w_{0}$  and apply Algorithm 2. Under event  $\mathrm{Event}_T^{\mathrm{single}}(w_0)$ , it satisfies

$$
f (w _ {0}) - f (w _ {T}) \geq 0. 7 \eta \sum_ {t = 0} ^ {T - 1} \left(\| \nabla f (w _ {t}) \| ^ {2} - \eta \cdot O (C _ {2} + (C _ {2}) ^ {1. 5}) - O (C _ {1} \nu^ {2} \eta (d + \sqrt {C _ {2}}))\right)
$$

Lemma 4.3 says after  $T \approx \frac{1}{\eta}$  steps of perturbed SGD, the objective value decreases by, up to some small additive error and up to logarithmic factors,  $f(w_0) - f(w_T) \geq 0.7\eta \sum_{t=0}^{T-1} (\|\nabla f(w_t)\|^2 - \eta C_2)$ . This immediately implies, if we choose  $\eta \approx \frac{\varepsilon^2}{C_2}$ , then by repeating this analysis for  $O\left(\frac{C_2}{\varepsilon^4}\right) = O\left(\frac{\alpha^2 + 1/m}{\varepsilon^4}\right)$  iterations, we can find approximate critical point  $x$  with  $\| \nabla f(x) \| \leq \varepsilon$ .

Proof sketch of Lemma 4.3. The full proof is in Appendix B.2 but we illustrate the main idea and difficulties below. After simple manipulations, it is not hard to derive that

$$
f (w _ {0}) - f (w _ {T}) \gtrsim 0. 9 \eta \sum_ {t = 0} ^ {T - 1} \left(\| \nabla f (w _ {t}) \| ^ {2} - \eta\right) + \underbrace {\eta \sum_ {t = 0} ^ {T - 1} \langle \nabla f (w _ {t}) , \Xi_ {t} \rangle} _ {\text {r e m a i n d e r t e r m s}}
$$

where recall that  $\Xi_t = \sigma_t + \Delta_t$ . When there are no Byzantine machines, we have  $\mathbb{E}[\Xi_t] = \mathbb{E}[\sigma_t] = 0$  so the remainder terms must be small by martingale concentration. Therefore, the main technical difficulty arises to deal with those Byzantine machines, who can adversely design their  $\nabla_t$  (even by collusion) so as to negatively correlate with  $\nabla f(w_t)$  to "maximally destroy" the above inequality. Our main idea is to use second-order smoothness to write  $\nabla f(w_t) \approx \nabla f(w_0) + \nabla^2 f(w_0) \cdot (w_t - w_0)$ . To illustrate our idea, let us ignore the constant vector and assume that the Hessian is the identity:

that is, imagine as if  $\nabla f(w_{t})\approx w_{t} - w_{0}$ . Using  $w_{t} - w_{0} = -\sum_{k < t}\Xi_{t} + \xi_{t}$ , we immediately have

$$
- \left\langle \nabla f \left(w _ {t}\right), \Xi_ {t} \right\rangle \approx - \left\langle w _ {t} - w _ {0}, \Xi_ {t} \right\rangle = \sum_ {k <   t} \left\langle \Xi_ {k}, \Xi_ {t} \right\rangle + \sum_ {k <   t} \left\langle \xi_ {k}, \Xi_ {t} \right\rangle \tag {4.1}
$$

For the first partial sum  $\langle \sum_{k < t} \Xi_k, \Xi_t \rangle$  in (4.1), it is easy to bound its magnitude using our safeguard. Indeed, we have  $\left| \sum_{t} \langle \sum_{k < t} \Xi_k, \Xi_t \rangle \right| \leq \left\| \sum_{t} \Xi_t \right\|^2 + \sum_{t} \left\| \Xi_t \right\|^2$  so we can apply Lemma 4.2. For the second partial sum  $\sum_{t} \sum_{k < t} \langle \xi_k, \Xi_t \rangle$ , we can apply Lemma 3.2.

Core Technical Lemma 2: Randomness Coupling. Our next technical lemma studies that, if run Algorithm 2 from a point  $w_0$  so that the Hessian  $\nabla^2 f(w_0)$  has an eigenvalue which is less than  $-\delta$  (think of  $w_0$  as a saddle point), then with good probability, after sufficiently many iterations, the sequence  $w_1, w_2, \ldots, w_T$  shall escape from  $w_0$  to distance at least  $R$  for some parameter  $R \approx \delta$ . To prove this, motivated by Jin et al. (2017), we study two executions of Algorithm 2 where their randomness are coupled. We then argue that at least one of them has to escape from  $w_0$ . For any vector  $v$ , let  $[v]_i$  denote the  $i$ -th coordinate of  $v$ .

Lemma 4.4. Suppose we choose  $\mathfrak{T}$  as in Lemma 4.2 and  $C_1, C_2$  as in Lemma 4.3. Suppose  $w_0 \in \mathbb{R}^d$  satisfies  $\lambda_{\min}(\nabla^2 f(w_0)) = -\delta$  for some  $\delta \geq 0$ . Without loss of generality let  $\mathbf{e}_1$  be the eigenvector of  $\nabla^2 f(w_0)$  with smallest eigenvalue. Consider now two executions of Algorithm 2, both starting from  $w_0^{\mathrm{a}} = w_0^{\mathrm{b}} = w_0$ , and suppose their randomness  $\{\xi_t^{\mathrm{a}}\}_t$  and  $\{\xi_t^{\mathrm{b}}\}_t$  are coupled so that  $[\xi_t^{\mathrm{a}}]_1 = -[\xi_t^{\mathrm{b}}]_1$  but  $[\xi_t^{\mathrm{a}}]_i = [\xi_t^{\mathrm{b}}]_i$  for  $i > 1$ . In words, the randomness is the same orthogonal to  $\mathbf{e}_1$ , but along  $\mathbf{e}_1$ , the two have opposite signs. Now, suppose we perform  $T = \Theta\left(\frac{1}{\eta\delta}\log \frac{R^2\delta}{\eta\nu^2}\right)$  steps of perturbed SGD from  $w_0^{\mathrm{a}}, w_0^{\mathrm{b}}$  respectively using Algorithm 2. Suppose

$$
R \leq O \left(\frac {\delta}{\sqrt {C _ {1}} \log \left(R ^ {2} \delta / \eta \nu^ {2}\right)}\right) \quad a n d \quad \nu^ {2} \geq \Omega \left(C _ {2} \log \frac {R ^ {2} \delta}{\eta \nu}\right).
$$

Then, under events  $\mathsf{Event}_T^{\mathrm{single}}(w_0^{\mathsf{a}})$  and  $\mathsf{Event}_T^{\mathrm{single}}(w_0^{\mathsf{b}})$ , with probability at least 0.98, either  $\| w_t^{\mathsf{a}} - w_0 \| > R$  or  $\| w_t^{\mathsf{b}} - w_0 \| > R$  for some  $t \in [T]$ .

Proof details in Appendix B.3. The main proof difficulty is to analyze a noisy version of the power method, where the noise comes from (1) Gaussian perturbation (which is the good noise), (2) stochastic gradients (which has zero mean), and (3) Byzantine workers (which can be adversarial).

From Warmup to Final Theorem with Double Safe Guards. At a high level, Lemma 4.3 ensures that if we keep encountering points with large gradient  $\| \nabla f(w_t)\|$ , then the objective value should sufficiently decrease; in contrast, Lemma 4.4 says that if we keep encountering points with negative Hessian directions (i.e.,  $\lambda_{\mathrm{min}}(\nabla^2 f(w_t)) < -\delta$ ), then the points must move a lot (i.e., by more than  $R$  in  $T$  iterations, which can also lead to sufficient objective decrease, see Lemma C.4). Therefore, at a high level, when the two lemmas are combined, they should tell that we must not encounter points with  $\| \nabla f(x)\|$  being large, or  $\lambda_{\mathrm{min}}(\nabla^2 f(x))$  being very negative, for too many iterations. Therefore, the algorithm can find approximate local minima. The reason we eventually need two safe guards, is because the number of rounds  $T$  for Lemma 4.3 and Lemma 4.4 differ by a factor. Thus, we need two safe guards with different window sizes to ensure that the two lemmas simultaneously hold. We encourage the reader to examine the full analysis in Appendix C.

# 5 EXPERIMENTAL VALIDATION

We evaluate the convergence of SafeguardSGD to examine its practical performance against prior works (Xie et al., 2018b; Blanchard et al., 2017; Chen et al., 2017; Yin et al., 2018; 2019). We perform the non-convex task of training a deep residual network (ResNet-20) (He et al., 2016) on the CIFAR-10 dataset (Krizhevsky et al., 2014). A full experimental report is given in the Appendix.

We instantiate  $m = 10$  workers and one master executing data-parallel SGD for 200 passes (i.e. epochs) over the training dataset. The ideal baseline is gradient mean without attacks, and we compare against (Naive) Mean, Geometric Median (Chen et al., 2017), Coordinate-wise Median (Yin et al., 2018; 2019), Krum (Blanchard et al., 2017), and Zeno (Xie et al., 2018b) with attacks. We also implemented Yang et al. (2019), but found it very sensitive to hyper-parameter values and were not able to make it converge across all attacks even after heavy tuning of its  $\gamma$  parameter. Overall, our experimental setup is very similar to Zeno (Xie et al., 2018b) but with additional attacks. The numbers provided are averaged over three repetitions.

To make the comparison stronger, when implementing SafeguardSGD, we have chosen fixed window sizes  $T_0 = 1$  epoch and  $T_1 = 6$  epochs across all experiments, and adopted an automated

![](images/d8b1c889524ec93aab5f214d72b56aeb182b43775ae8ce635f205a78fc1dff0a.jpg)

![](images/7494f4e2067758f1f0aed172f7135a657ba288b82e961c92f4306fb75fa2138d.jpg)

![](images/ea8b35c1a80b383638f43b63f20a9d0c659f61f34b8630d0496249c86d11c95f.jpg)  
Figure 1: Convergence comparison (CIFAR-10 test accuracy) under different attacks. Figures for the Safeguard and Variance attacks are deferred to the Appendix due to space limitations.

![](images/c8ec88e3c2899ad46542a3781cf0b4781b8dd072a739d55002b17511a8bd013e.jpg)

Table 1: Test accuracy comparison under different attacks. For full results see Table 2 in the Appendix.  

<table><tr><td rowspan="2">Method</td><td colspan="6">Test Accuracy / Attack Type</td></tr><tr><td>Delayed-gradient</td><td>Label-flipping</td><td>Sign-flipping</td><td>Zero-gradient</td><td>Safeguard Attack (rescaling-factor = 0.4)</td><td>Variance Attack (stdev factor = 0.3)</td></tr><tr><td>Ideal Baseline</td><td>90.8</td><td>90.8</td><td>90.8</td><td>90.8</td><td>90.8</td><td>90.8</td></tr><tr><td>Double Safe-guard</td><td>91.7</td><td>90.6</td><td>90.5</td><td>90.3</td><td>88.3</td><td>90.8</td></tr><tr><td>Single Safe-guard</td><td>91.1</td><td>85.5</td><td>90.9</td><td>90.6</td><td>87.5</td><td>91.0</td></tr><tr><td>Best Other Method</td><td>79.6 (Zeno)</td><td>85.3 (Zeno)</td><td>74.4 (Zeno)</td><td>88.6 (Zeno)</td><td>87.7 (Zeno)</td><td>29.0 (Zeno)</td></tr></table>

process to select  $\mathfrak{T}_0, \mathfrak{T}_1$  (by pre-running the experiment for 20 epochs). We have also implemented a single safeguard variant of SafeguardSGD with window size  $T = 3$  epochs.

Attackers. We set  $\alpha = 0.4$  so there are 4 Byzantine workers. (This exceeds the fault-tolerance of Krum, and so we also tested Krum with only 3 Byzantine workers.)

- SIGN-FLIPPING ATTACK: each Byzantine worker sends the negative gradient to the master.  
- ZERO-GRADIENT ATTACK: each Byzantine worker sends zero gradient to the master.  
- LABEL-FLIPPING ATTACK: each Byzantine worker computes its gradient based on the cross-entropy loss with flipped labels: for CIFAR-10, label  $\ell \in \{0,\dots ,9\}$  is flipped to  $9 - \ell$  
- DELAYED-GRADIENT ATTACK: each Byzantine worker sends an old gradient to master. In our experiments, the delay is of  $D = 1000$  iterations.  
- SAFEGUARD ATTACK: Byzantine workers send negative but re-scaled gradient to the master. The re-scale factor 0.4 is chosen to avoid triggering the safe-guard conditions at the master (we also show results for re-scale factor 0.5 in the Appendix). This attack is an instantiation of the inner-product attack (Xie et al., 2020), customized specifically to maximally affect our algorithm.  
- VARIANCE ATTACK (Baruch et al., 2019): Byzantine workers measure the mean and the standard-deviation of gradients at each round, and collude to move the mean by the largest value which still operates within population variance. (For our parameter settings, this is 0.3 times the standard deviation. We discuss results for additional parameter values in the Appendix.)

Results. Figure 1 compares the convergence curves, while Table 1 compares the best test accuracy. SafeguardSGD generally outperforms the previous methods in test accuracy and convergence, and closely tracks the performance of the ideal baseline, across all attacks. The test accuracy difference can be  $>10\%$  between our algorithm and the best prior work (see delayed gradient, sign-flipping, and variance attacks). SafeguardSGD slightly outperforms all other algorithms even for the customized safeguard attacks, which were designed to maximally impact its performance. In most cases, the single-safeguard algorithm is close to double-safeguard, except for the label-flipping attack. We conclude that SafeguardSGD can be practical, and outperforms previous approaches.

# REFERENCES

Dan Alistarh, Zeyuan Allen-Zhu, and Jerry Li. Byzantine stochastic gradient descent. In Advances in Neural Information Processing Systems, pages 4613-4623, 2018.  
Zeyuan Allen-Zhu. Natasha 2: Faster Non-Convex Optimization Than SGD. In NeurIPS, 2018a. Full version available at http://arxiv.org/abs/1708.08694.  
Zeyuan Allen-Zhu. How To Make the Gradients Small Stochastically. In NeurIPS, 2018b. Full version available at http://arxiv.org/abs/1801.02982.  
Gilad Baruch, Moran Baruch, and Yoav Goldberg. A little is enough: Circumventing defenses for distributed learning. In Advances in Neural Information Processing Systems, pages 8635-8645, 2019.  
Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. Machine learning with adversaries: Byzantine tolerant gradient descent. In NIPS, pages 118-128, 2017.  
Saikiran Bulusu, Prashant Khanduri, Pranay Sharma, and Pramod K Varshney. On distributed stochastic gradient descent for nonconvex functions in the presence of byzantines. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3137-3141. IEEE, 2020.  
Miguel Castro, Barbara Liskov, et al. Practical byzantine fault tolerance. In OSDI, 1999.  
Yudong Chen, Lili Su, and Jiaming Xu. Distributed statistical machine learning in adversarial settings: Byzantine gradient descent. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 1(2): 1-25, 2017.  
Cong Fang, Chris Junchi Li, Zhouchen Lin, and Tong Zhang. Spider: Near-optimal non-convex optimization via stochastic path-integrated differential estimator. In Advances in Neural Information Processing Systems, pages 689-699, 2018.  
Jiashi Feng, Huan Xu, and Shie Mannor. Distributed robust learning. arXiv preprint arXiv:1409.5937, 2014.  
Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points—online stochastic gradient for tensor decomposition. In Proceedings of the 28th Annual Conference on Learning Theory, COLT 2015, 2015.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770-778, 2016.  
Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M Kakade, and Michael I Jordan. How to escape saddle points efficiently. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 1724-1732. JMLR.org, 2017.  
Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M Kakade, and Michael I. Jordan. On nonconvex optimization for machine learning: Gradients, stochasticity, and saddle points. arXiv preprint arXiv:1902.04811, 2019.  
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The CIFAR-10 dataset. online: http://www.cs.toronto.edu/kriz/cifar.html, 55, 2014.  
Leslie Lamport, Robert Shostak, and Marshall Pease. The byzantine generals problem. ACM Transactions on Programming Languages and Systems (TOPLAS), 4(3):382-401, 1982.  
Lihua Lei, Cheng Ju, Jianbo Chen, and Michael I Jordan. Nonconvex Finite-Sum Optimization Via SCSG Methods. In NIPS, 2017.  
Nancy A Lynch. Distributed algorithms. Elsevier, 1996.  
Lam M Nguyen, Jie Liu, Katya Scheinberg, and Martin Takáč. Sarah: A novel method for machine learning problems using stochastic recursive gradient. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 2613–2621. JMLR.org, 2017.  
Iosif Pinelis. Optimum bounds for the distributions of martingales in banach spaces. The Annals of Probability, pages 1679-1706, 1994.  
Lili Su and Nitin H Vaidya. Fault-tolerant multi-agent optimization: optimal iterative distributed algorithms. In PODC, pages 425-434. ACM, 2016a.  
Lili Su and Nitin H Vaidya. Defending non-bayesian learning against adversarial attacks. ISDC, 2016b.  
Lili Su and Jiaming Xu. Securing distributed machine learning in high dimensions. arXiv preprint arXiv:1804.10140, 2018.  
Nilesh Tripuraneni, Mitchell Stern, Chi Jin, Jeffrey Regier, and Michael I Jordan. Stochastic Cubic Regularization for Fast Nonconvex Optimization. ArXiv e-prints, abs/1711.02838, November 2017.  
Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Generalized Byzantine-tolerant SGD. arXiv preprint arXiv:1802.10116, 2018a.  
Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Zeno: Byzantine-suspicious stochastic gradient descent. arXiv preprint arXiv:1805.10032, 2018b.  
Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Fall of empires: Breaking byzantine-tolerant sgd by inner

product manipulation. volume 115 of Proceedings of Machine Learning Research, pages 261-270, Tel Aviv, Israel, 22-25 Jul 2020. PMLR. URL http://proceedings.mlr.press/v115/xie20a.html.  
Haibo Yang, Xin Zhang, Minghong Fang, and Jia Liu. Byzantine-resilient stochastic gradient descent for distributed learning: A lipschitz-inspired coordinate-wise median approach. arXiv preprint arXiv:1909.04532, 2019.  
Dong Yin, Yudong Chen, Kanna Ramchandran, and Peter Bartlett. Byzantine-robust distributed learning: Towards optimal statistical rates. arXiv preprint arXiv:1803.01498, 2018.  
Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Defending against saddle point attack in byzantine-robust distributed learning. In International Conference on Machine Learning, pages 7074-7084, 2019.
