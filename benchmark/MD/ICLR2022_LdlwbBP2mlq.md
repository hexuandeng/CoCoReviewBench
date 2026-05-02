# MINIBATCH VS LOCAL SGD WITH SHUFFLING: TIGHT CONVERGENCE BOUNDS AND BEYOND

Anonymous authors

Paper under double-blind review

# ABSTRACT

In distributed learning, local SGD (also known as federated averaging) and its simple baseline minibatch SGD are widely studied optimization methods. Most existing analyses of these methods assume independent and unbiased gradient estimates obtained via with-replacement sampling. In contrast, we study shuffling-based variants: minibatch and local Random Reshuffling, which draw stochastic gradients without replacement and are thus closer to practice. For smooth functions satisfying the Polyak-Lojasiewicz condition, we obtain convergence bounds (in the large epoch regime) which show that these shuffling-based variants converge faster than their with-replacement counterparts. Moreover, we prove matching lower bounds showing that our convergence analysis is tight. Finally, we propose an algorithmic modification called synchronized shuffling that leads to convergence rates faster than our lower bounds in near-homogeneous settings.

# 1 INTRODUCTION

Distributed learning within the framework of federated learning (Konečný et al., 2016; McMahan et al., 2017) has witnessed increasing interest recently. A key property of this framework is that models are trained locally using only private data on devices/machines distributed across a network, while parameter updates are aggregated and synchronized at a server. Communication is often the key bottleneck for federated learning, which drives the search for algorithms that can train fast while requiring less communication—see Li et al. (2020a); Kairouz et al. (2021) for recent surveys.

A basic algorithm for federated learning is local stochastic gradient descent (SGD), also known as federated averaging. The goal is to minimize the global objective that is an average of the local objectives. In local SGD, we have  $M$  machines and a server. After each round of communication, each of the  $M$  machines locally runs  $B$  steps of SGD on its local objective. Every  $B$  iterations, the server aggregates the updated local iterates from the machines, averages them, and then synchronizes the machines with the average. Convergence analysis of local SGD and its variants has drawn great interest recently (Dieuleveut & Patel, 2019; Haddadpour et al., 2019; Haddadpour & Mahdavi, 2019; Stich, 2019; Yu et al., 2019; Li et al., 2020b;c; Koloskova et al., 2020; Khaled et al., 2020; Spiridonoff et al., 2020; Karimireddy et al., 2020; Stich & Karimireddy, 2020; Qu et al., 2020).

Of the many, the biggest motivation for our paper comes from the line of work by Woodworth et al. (2020a;b; 2021). In (Woodworth et al., 2020a;b), minibatch SGD is studied as a simple yet powerful baseline for this intermittent communication setting. Instead of locally updating the iterates  $B$  times, minibatch SGD aggregates  $B$  gradients (evaluated at the last synced iterate) from each of the  $M$  machines, forms a minibatch of size  $MB$ , and then updates the shared iterate. Given the same  $M$  and  $B$ , local SGD and minibatch SGD have the same number of gradient computations per round of communication, so it is worthwhile to understand which converges faster. Woodworth et al. (2020a;b) point out that many existing analyses on local SGD show inferior convergence rate compared to minibatch SGD. Through their new upper and lower bounds, they identify regimes where local SGD can be faster than minibatch SGD.

While the theory of local and minibatch SGD has seen recent progress, there is still a gap between what is analyzed versus what is actually used. Most theoretical results assume independent and

unbiased gradient estimates obtained via with-replacement sampling of stochastic gradients (i.e., choosing training data indices uniformly at random). In contrast, most practitioners use without-replacement sampling, where they shuffle indices randomly and access them sequentially.

Convergence analysis of without-replacement methods is challenging because gradients sampled within an epoch lack independence. As a result, the standard theory based on independent gradient estimates does not apply to shuffling-based methods. While shuffling-based methods are believed to be faster in practice (Bottou, 2009), broad theoretical understanding of such methods remains elusive, except for noteworthy recent progress mainly focusing on the analysis of SGD (Gürbüzbalaban et al., 2019; Haochen & Sra, 2019; Nagaraj et al., 2019; Nguyen et al., 2020; Safran & Shamir, 2020; 2021; Rajput et al., 2020; 2021; Ahn et al., 2020; Mishchenko et al., 2020; 2021; Tran et al., 2021). These results indicate that in the large-epoch regime (where the number of epochs is greater than some threshold), without-replacement SGD converges faster than with-replacement SGD.

# 1.1 OUR CONTRIBUTIONS

We analyze convergence rates of without-replacement versions of local and minibatch SGD, where local component functions are reshuffled at every epoch. We call the respective algorithms local RR (Algorithm 1) and minibatch RR (Algorithm 2), and their with-replacement counterparts local SGD and minibatch SGD. Our key contributions are as follows:

- In Section 3, we present convergence bounds on minibatch and local RR for smooth functions satisfying the Polyak-Lojasiewicz condition (Theorems 1 & 2). Our theorems give high-probability bounds, a departure from the common in expectation bounds in the literature. We show that minibatch and local RR converge faster than minibatch and local SGD when the number of epochs is sufficiently large. We also identify a regime where local RR converges as fast as minibatch RR: when synchronization happens frequently enough and local objectives are not too heterogeneous. See also Appendix A for a detailed comparison with existing upper bounds.  
- In Section 4, we prove that the upper bounds obtained in Section 3 are tight. We present Theorems 3 & 4 and Proposition 5 which show matching lower bounds for both minibatch and local RR with constant step-size. Our lower bound on local RR indicates that if the synchronization interval  $B$  is too large, then local RR has no advantage over single-machine RR.  
- In Section 5, we propose a simple modification called synchronized shuffling that allows us to bypass the lower bounds in Section 4, at the cost of a slight increase in communication. By having the server broadcast random permutations to local machines, we show that in near-homogeneous settings, the modified algorithms converge faster than the lower bounds (Theorems 6 & 7).

# 2 PROBLEM SETUP

Notation. For a natural number  $a \in \mathbb{N}$ , let  $[a] := \{1, 2, \dots, a\}$ . Let  $S_a$  be the set of all permutations of  $[a]$ . Since our indices start from 1, we redefine the modulo operation between  $a \in \mathbb{Z}$  and  $b \in \mathbb{N}$  as  $a \mod b := a - \lfloor \frac{a - 1}{b} \rfloor b$ , to make  $a \mod b \in [b]$ .

**Optimization task.** Consider  $M$  machines, each with its objective  $F^{m}(\pmb{x}) \coloneqq \frac{1}{N}\sum_{i=1}^{N}f_{i}^{m}(\pmb{x})$ , for  $m \in [M]$ . The  $m$ -th machine has access only to the gradients of its own  $N$  local components  $f_{1}^{m}(\pmb{x}), \ldots, f_{N}^{m}(\pmb{x})$ . In this setting, we wish to minimize the global objective function which is an average of the local objectives:  $F(\pmb{x}) \coloneqq \frac{1}{M}\sum_{m=1}^{M}F^{m}(\pmb{x}) = \frac{1}{MN}\sum_{m=1}^{M}\sum_{i=1}^{N}f_{i}^{m}(\pmb{x})$ .

Further, we assume that each individual component function  $f_{i}^{m}$  is  $L$ -smooth, so that

$$
f _ {i} ^ {m} (\boldsymbol {y}) \leq f _ {i} ^ {m} (\boldsymbol {x}) + \left\langle \nabla f _ {i} ^ {m} (\boldsymbol {x}), \boldsymbol {y} - \boldsymbol {x} \right\rangle + \frac {L}{2} \| \boldsymbol {y} - \boldsymbol {x} \| ^ {2}, \text {f o r a l l} \boldsymbol {x}, \boldsymbol {y} \in \mathbb {R} ^ {d}, \tag {1}
$$

and that the global objective  $F$  satisfies the  $\mu$ -Polyak-Lojasiewicz (PL) condition.2

$$
\frac {1}{2} \left\| \nabla F (\boldsymbol {x}) \right\| ^ {2} \geq \mu \left(F (\boldsymbol {x}) - F ^ {*}\right) \text {f o r a l l} \boldsymbol {x} \in \mathbb {R} ^ {d}, \quad \text {w h e r e} \mu > 0. \tag {2}
$$

Algorithms. Under the above setting, we analyze local RR (Algorithm 1) and minibatch RR (Algorithm 2) and characterize their worst-case convergence rates. The algorithms are run over  $K$  epochs,

Algorithm 1 Local RR (with and without SYNCSHUF)  
Input: Initialization  $y_0$ , step-size  $\eta$ , # machines  $M$ , # components  $N$ , # epochs  $K$ , sync interval  $B$ .  
1: Initialize  $x_{1,0}^m := y_0$  for all  $m \in [M]$ .  
2: for  $k \in [K]$  do  
3: if SYNCSHUF = TRUE then ▷ Local RR with SYNCSHUF  
4: Sample  $\sigma \sim \mathrm{Unif}(S_N)$ ,  $\pi \sim \mathrm{Unif}(S_M)$ .  
5: Set  $\sigma_k^m (i) := \sigma ((i + \frac{N}{M}\pi (m)) \bmod N)$  for all  $m \in [M], i \in [N]$ .  
6: else ▷ Local RR  
7: Sample  $\sigma_k^m \sim \mathrm{Unif}(S_N)$  independently and locally, for all  $m \in [M]$ .  
8: end if  
9: for  $i \in [N]$  do  
10: for  $m \in [M]$  do locally  
11: Update  $x_{k,i}^{m} := x_{k,i-1}^{m} - \eta\nabla f_{\sigma_k^m(i)}^{m}(x_{k,i-1}^{m})$ .  
12: end for  
13: if  $B$  divides  $i$  then  
14: Aggregate and average  $y_{k,\frac{i}{B}} := \frac{1}{M}\sum_{m=1}^{M}x_{k,i}^{m}$ .  
15: Synchronize  $x_{k,i}^{m} := y_{k,\frac{i}{B}}$ , for all  $m \in [M]$ .  
16: end if  
17: end for  
18:  $x_{k+1,0}^{m} := y_{k,\frac{N}{B}}$ , for all  $m \in [M]$ .  
19: end for  
20: return the last iterate  $y_{K,\frac{N}{B}}$ .

Algorithm 2 Minibatch RR (with and without SYNCSHUF)  
Input: Initialization  $\pmb{x}_0$  step-size  $\eta$  #machines  $M$  #components  $N$  #epochs  $K$  ,sync interval  $B$  1: Initialize  $\pmb{x}_{1,0}\coloneqq \pmb{x}_0$    
2: for  $k\in [K]$  do   
3: if SYNCSHUF  $\equiv$  TRUE then ▷Minibatch RR with SYNCSHUF   
4: Sample  $\sigma \sim$  Unif  $(S_N)$ $\pi \sim$  Unif  $(S_M)$    
5: Set  $\sigma_k^m (i)\coloneqq \sigma ((i + \frac{N}{M}\pi (m))\mathrm{mod}N)$  for all  $m\in [M],i\in [N].$    
6: else ▷Minibatch RR   
7: Sample  $\sigma_k^m\sim$  Unif  $(S_N)$  independently and locally, for all  $m\in [M]$    
8: end if   
9: for  $i\in [\frac{N}{B} ]$  do   
10: Update  $\pmb {x}_{k,i}\coloneqq \pmb {x}_{k,i - 1} - \frac{\eta}{M}\sum_{m = 1}^{M}\underbrace{\frac{1}{B}\sum_{j = (i - 1)B + 1}^{iB}\nabla f_{\sigma_k^m (j)}^m(\pmb {x}_{k,i - 1})}_{\text{averaging done locally}}.$    
11: end for   
12:  $\pmb {x}_{k + 1,0}\coloneqq \pmb {x}_k,\frac{N}{B}$  .

i.e.,  $K$  passes over the entire component functions. At the beginning of epoch  $k$ , each machine  $m$  shuffles its local component functions  $\{f_i^m\}_{i=1}^N$  using a random permutation  $\sigma_k^m \sim \mathrm{Unif}(S_N)$ . In local RR, each machine makes  $B$  local RR updates to its iterate by sequentially accessing its shuffled component functions, before the server aggregates iterates from all the machines and then synchronizes the machines with the average iterate. In minibatch RR, instead of making  $B$  local updates, each machine collects  $B$  gradients evaluated at the last iterate, and the server aggregates them to make an update using these  $MB$  gradients. Since these two algorithms use the same amount of communication and local gradients, minibatch RR is a simple yet powerful baseline for local RR.

Below, we collect our assumptions on the algorithm parameters used throughout the paper.

Assumption 1 (Algorithm parameters). We assume  $M \geq 1$ ,  $N \geq 2$ , and  $K \geq 1$ . Also, assume that  $B$  divides  $N$ . We restrict  $1 \leq B \leq \frac{N}{2}$  for minibatch RR because  $B = N$  makes the algorithm equal to  $GD$ . We also assume  $2 \leq B \leq N$  for local RR because  $B = 1$  makes the two algorithms the same. We choose a constant step-size scheme, i.e.,  $\eta > 0$  is kept constant over all updates.

We next state assumptions on intra- and inter-machine deviations used in this paper.

Assumption 2 (Intra-machine deviation). There exists  $\nu \geq 0$  such that for all  $m\in [M]$  and  $i\in [N]$

$$
\left\| \nabla f _ {i} ^ {m} (\boldsymbol {x}) - \nabla F ^ {m} (\boldsymbol {x}) \right\| \leq \nu , \text {f o r a l l} \boldsymbol {x} \in \mathbb {R} ^ {d}.
$$

Assumption 2 requires that the difference between the gradient of each local component function  $f_{i}^{m}(\pmb{x})$  and its corresponding local objective function  $F^{m}(\pmb{x})$  is uniformly bounded. It models the variance of local component functions  $f_{i}^{m}$  within each machine.

The next two assumptions capture the deviation across different machines, i.e., the degree of heterogeneity, in two different levels of granularity: objective-wise and component-wise.

Assumption 3 (Objective-wise inter-machine deviation). There exist  $\tau \geq 0$  and  $\rho \geq 1$  such that

$$
\frac {1}{M} \sum_ {m = 1} ^ {M} \| \nabla F ^ {m} (\boldsymbol {x}) \| \leq \tau + \rho \| \nabla F (\boldsymbol {x}) \|, f o r a l l \boldsymbol {x} \in \mathbb {R} ^ {d}.
$$

Assumption 3 models the heterogeneity by bounding the mean of  $\| \nabla F^m\|$  by a constant plus a multiplicative factor times  $\| \nabla F\|$ . The assumption includes the homogeneous case (i.e.,  $F^{1} = \dots = F^{M} = F$ ) by  $\tau = 0$  and  $\rho = 1$ . Assumption 3 is weaker than many other heterogeneity assumptions in the literature (e.g., Karimireddy et al. (2020)); see Appendix A for detailed comparisons.

Assumption 3 measures heterogeneity by only considering the local objectives  $F^{m}$ , not the local components  $f_{i}^{m}$ . We consider a more fine-grained notion of heterogeneity in Assumption 4:

Assumption 4 (Component-wise inter-machine deviation). For all  $i \in [N]$ , let  $\bar{f}_i \coloneqq \frac{1}{M} \sum_{m=1}^{M} f_i^m$ . There exist  $\lambda \geq 0$  such that for all  $m \in [M]$  and  $i \in [N]$ ,

$$
\left\| \nabla f _ {i} ^ {m} (\boldsymbol {x}) - \nabla \bar {f} _ {i} (\boldsymbol {x}) \right\| \leq \lambda , f o r a l l \boldsymbol {x} \in \mathbb {R} ^ {d}.
$$

Assumption 4 states that the gradients of the  $i$ -th components of local machines are "close" to each other. The assumption subsumes the component-wise homogeneous setting, i.e.,  $f_{i}^{1} = f_{i}^{2} = \dots = f_{i}^{M}$ , by  $\lambda = 0$ . In distributed learning, this choice corresponds to the setting where each machine has the same training dataset. Assumption 4 with  $\lambda > 0$  is also relevant to the case where each device has a slightly perturbed (e.g., by data augmentation techniques) version of a certain dataset. It is straightforward to check that Assumption 4 implies Assumption 3 with  $\tau = \lambda$  and  $\rho = 1$ .

We conclude this section by defining the function classes we study in this paper.

Definition 1 (Function classes). We consider two classes of global objective functions  $F$ , also taking into account their local objectives  $F^m$  and local components  $f_i^m$ . We assume throughout that  $f_i^m$  are differentiable and  $F$  is bounded from below.

$\mathcal{F}_{\mathrm{obj}}(L,\mu ,\nu ,\tau ,\rho)\coloneqq \bigl \{F\mid F$  is  $\mu$  -PL;  $f_{i}^{m}$  are L-smooth;  $F,F^{m},f_{i}^{m}$  satisfy Assumptions 2 & 3},  $\mathcal{F}_{\mathrm{cmp}}(L,\mu ,\nu ,\lambda)\coloneqq \bigl \{F\mid F$  is  $\mu$  -PL;  $f_{i}^{m}$  are L-smooth;  $F,F^{m},f_{i}^{m}$  satisfy Assumptions 2 & 4}.

Notice that  $\mathcal{F}_{\mathrm{obj}}(L,\mu ,\nu ,\tau ,\rho)\supset \mathcal{F}_{\mathrm{cmp}}(L,\mu ,\nu ,\tau)$  for any  $\rho \geq 1$ . We only make the PL assumption on the global objective  $F$ , not on the local objectives  $F^{m}$  nor on the local components  $f_{i}^{m}$ . Using  $L$  and  $\mu$ , we define the condition number  $\kappa \coloneqq L / \mu \geq 1$ .

# 3 CONVERGENCE ANALYSIS OF MINIBATCH AND LOCAL RR

# 3.1 UPPERBOUND FOR MINIBATCH RR

We first begin with the convergence result for minibatch RR on  $\mathcal{F}_{\mathrm{obj}}(L,\mu ,\nu ,\tau ,\rho)$ , which exhibits a faster large-epoch rate compared to the single-machine setting. For upper bounds, we use  $\tilde{\mathcal{O}} (\cdot)$  to hide universal constants and logarithmic factors of  $\frac{1}{\delta}$ ,  $M$ ,  $N$ ,  $K$ , and  $B$ .

Theorem 1 (Upper bound for minibatch RR). Suppose that minibatch RR has parameters satisfying Assumption 1. For any  $F \in \mathcal{F}_{\mathrm{obj}}(L, \mu, \nu, \tau, \rho)$ , consider running the algorithm using step-size  $\eta = \frac{B \log(MNK^2)}{\mu NK}$  for epochs  $K \geq 6\kappa \log(MNK^2)$ . Then, with probability at least  $1 - \delta$ ,

$$
F \left(\boldsymbol {x} _ {K, \frac {N}{B}}\right) - F ^ {*} \leq \frac {F \left(\boldsymbol {x} _ {0}\right) - F ^ {*}}{M N K ^ {2}} + \tilde {\mathcal {O}} \left(\frac {L ^ {2}}{\mu^ {3}} \frac {\nu^ {2}}{M N K ^ {2}}\right). \tag {3}
$$

Proof. The proof is in Appendix B.2. The key challenge in the convergence analysis of our shuffling-based method stems from the indices sampled within an epoch being dependent on each other. For example, if  $f_1^m$  is accessed already, then the index  $i = 1$  will not be used in later iterations of the

epoch; this dependence significantly complicates the analysis. Our approach starts with realizing that for any permutation  $\sigma$ ,  $\sum_{i=1}^{N} f_{\sigma(i)}^{m} = NF^{m}$ . We decompose gradients  $\nabla f_{\sigma_k^m(j)}^m(\boldsymbol{x}_{k,i-1})$  (see Line 10 of Algorithm 2) into  $\nabla f_{\sigma_k^m(j)}^m(\boldsymbol{x}_{k,0})$  plus noise, then aggregate all updates over an epoch to get "one step of GD plus noise":  $\boldsymbol{x}_{k+1,0} = \boldsymbol{x}_{k,0} - \eta N\nabla F(\boldsymbol{x}_{k,0}) + \eta^2 \boldsymbol{r}_k$ . We bound the noise  $\boldsymbol{r}_k$  using Lemma 8 (Appendix B.6), which is our extension of the Hoeffding-Serfling inequality to the mean of  $M$  independent without-replacement sums of vectors; the lemma might be of independent interest too. Lemma 8 shows that averaging accumulated gradients over  $M$  machines reduces variance by  $M$ , which leads to the reduction by a factor of  $M$  in the bound (3).

Theorem 1 shows that for large enough epochs  $K \gtrsim \kappa$ , minibatch RR converges at a rate of  $\tilde{\mathcal{O}}\left(\frac{L^2\nu^2}{\mu^3MNK^2}\right)$ , with high probability. Compared to the large-epoch rate  $\tilde{\mathcal{O}}\left(\frac{L^2\nu^2}{\mu^3NK^2}\right)$  of single-machine RR (e.g., Ahn et al. (2020)), we see an additional factor  $M$  in the denominator, which highlights the advantage of multiple machines. If we compare against the with-replacement counterpart, it is known that for strongly convex and smooth  $F$ , the optimal convergence rate of minibatch SGD is  $\Theta\left(\frac{\nu^2}{\mu MNK}\right)^4$  which is worse than our bound (3) if  $K \gtrsim \kappa^2$ . Also notable is that the convergence rate does not depend on the heterogeneity constants (i.e.,  $\tau$  and  $\rho$  from Assumption 3) of the local objective functions. This observation that minibatch RR is "immune" to heterogeneity is consistent with minibatch SGD in the with-replacement setting (Woodworth et al., 2020b).

Epoch vs communication complexity. One might wonder why (3) does not have the batch size  $B$ . In (3), we wrote convergence rates in terms of epochs  $K$ , which captures the gradient computation complexity because the same number of gradients are evaluated in a single epoch regardless of  $B$ . If we are interested in communication complexity instead, we can write (3) in terms of the number of communication rounds  $R := \frac{NK}{B}$  and get a rate of  $\tilde{\mathcal{O}}\left(\frac{L^2\nu^2N}{\mu^3MB^2R^2}\right)$ . From these, we can also discuss the overall cost of the algorithm. If the cost of a communication round is  $c_{c}$ , and the cost of local gradient computations over an epoch is  $c_{e}$ , then the total cost to obtain an  $\epsilon$ -accurate solution is

$$
C _ {\text {m i n i b a t c h}} (\epsilon) = \tilde {\mathcal {O}} \left(\frac {c _ {c} \nu \sqrt {N}}{B \sqrt {M \epsilon}} + \frac {c _ {e} \nu}{\sqrt {M N \epsilon}}\right), \tag {4}
$$

omitting  $L$  and  $\mu$  for simplicity. The total cost shows that there is essentially no harm increasing the batch size  $B$  in minibatch RR, as we can get more accurate estimates of true gradients as  $B$  becomes larger. In the next subsection, we will see that this is not the case in local RR.

What about  $K \lesssim \kappa$ ? We remark that all upper bounds in this paper hold only for the "large-epoch" regime, where  $K \gtrsim \kappa$ . Such requirements are common in the literature of without-replacement SGD (Haochen & Sra, 2019; Nagaraj et al., 2019; Rajput et al., 2020; Ahn et al., 2020), and there is a recent result (Safran & Shamir, 2021) suggesting that faster convergence of without-replacement SGD may not be possible in the  $K \lesssim \kappa$  regime. We defer a more detailed discussion on this regime to Section 4, after Theorem 3.

# 3.2 UPPERBOUND FOR LOCAL RR

Next, we are interested in how fast local RR can converge, what is the optimal batch size  $B$ , and whether local RR can be as fast as minibatch RR.

Theorem 2 (Upper bound for local RR). Suppose that local RR has parameters satisfying Assumption 1. For any  $F \in \mathcal{F}_{\mathrm{obj}}(L, \mu, \nu, \tau, \rho)$ , consider running the algorithm using step-size  $\eta = \frac{\log(MNK^2)}{\mu NK}$  for epochs  $K \geq 7\rho \kappa \log(MNK^2)$ . Then, with probability at least  $1 - \delta$ ,

$$
F \left(\boldsymbol {y} _ {K, \frac {N}{B}}\right) - F ^ {*} \leq \frac {F \left(\boldsymbol {y} _ {0}\right) - F ^ {*}}{M N K ^ {2}} + \tilde {\mathcal {O}} \left(\frac {L ^ {2}}{\mu^ {3}} \left(\frac {\nu^ {2}}{M N K ^ {2}} + \frac {\nu^ {2} B}{N ^ {2} K ^ {2}} + \frac {\tau^ {2} B ^ {2}}{N ^ {2} K ^ {2}}\right)\right). \tag {5}
$$

Proof. The proof is in Appendix B.3. We take the same "GD step plus noise" approach as in Theorem 1; however, due to local updates, bounding the noise is much more involved. In the proof,

we obtain the epoch update  $\pmb{y}_{k + 1,0} = \pmb{y}_{k,0} - \eta N\nabla F(\pmb{x}_{k,0}) + \eta^2\pmb{r}_{k,1} + \eta^2\pmb{r}_{k,2} - \eta^3\pmb{r}_{k,3}$ , where  $\pmb{r}_{k,1}$  and  $\pmb{r}_{k,3}$  contain errors introduced by local updates. Noise from local updates accumulates over  $B$  iterations, which cannot be remedied by averaging over  $M$  machines. They result in two additional terms in the rate (5), one from intra-machine variance and the other from heterogeneity.

# 3.2.1 DISCUSSION OF THEOREM 2

Let us compare our high-probability bound (5) with existing in-expectation bounds. For strongly convex  $F$ , the corresponding last-iterate bound of local SGD is  $\tilde{\mathcal{O}}\left(\frac{L\nu^2}{\mu^2MNK} + \frac{L^2\nu^2B}{\mu^3N^2K^2} + \frac{L^2\tau^2B^2}{\mu^3N^2K^2}\right)^5$  (Khaled et al., 2020; Spiridonoff et al., 2020; Qu et al., 2020). Notice that (5) is better than this with-replacement bound when  $K \gtrsim \kappa$ . For average iterates, there are known bounds  $\tilde{\mathcal{O}}\left(\frac{\nu^2}{\mu MNK} + \frac{L\nu^2B}{\mu^2N^2K^2} + \frac{L\tau^2B^2}{\mu^2N^2K^2}\right)^5$  (Koloskova et al., 2020; Woodworth et al., 2020b) which are smaller than the last-iterate bound by a factor of  $\kappa$ . It is unclear if averaging iterates could improve our rate, because most such analyses exploit Jensen's inequality, which we cannot use for nonconvex  $F$ .

Dependence on  $\tau$  and  $\rho$ . Out of the two heterogeneity constants  $\tau$  and  $\rho$  (Assumption 3),  $\rho$  does not appear in (5), and it only affects the epoch requirement  $K \gtrsim \rho \kappa$ . Consider the case  $\tau = 0$  and  $\rho > 1$ , which is heterogeneous but in the "interpolation regime," because  $\nabla F^m(\boldsymbol{x}) = \mathbf{0}$  whenever  $\nabla F(\boldsymbol{x}) = \mathbf{0}$ . In such a case, the rate (5) is equal to the homogeneous case.

Using  $B = \Theta(N)$  is no better than single-machine. A close look at Theorem 2 reveals a rather surprising fact. Even in the homogeneous case ( $\tau = 0$ ), if we choose  $B = \Theta(N)$ , then local RR converges at the rate of  $\tilde{\mathcal{O}}\left(\frac{1}{NK^2}\right)$ : the same rate as the single-machine RR! In Section 4, we show that this observation is not due to a suboptimal analysis; the rate  $\tilde{\mathcal{O}}\left(\frac{1}{NK^2}\right)$  is tight for  $B = \Theta(N)$ .

Trade-off in the choice of  $B$ . As done for Theorem 1, we can compute from (5) that the total cost of local RR for  $\epsilon$ -accuracy is (omitting  $L$  and  $\mu$  for simplicity)

$$
C _ {\text {l o c a l}} (\epsilon) = \tilde {\mathcal {O}} \left(c _ {c} \left(\frac {\nu \sqrt {N}}{B \sqrt {M} \epsilon} + \frac {\nu}{\sqrt {B} \epsilon} + \frac {\tau}{\sqrt {\epsilon}}\right) + c _ {e} \left(\frac {\nu}{\sqrt {M N} \epsilon} + \frac {\nu \sqrt {B}}{N \sqrt {\epsilon}} + \frac {\tau B}{N \sqrt {\epsilon}}\right)\right). \tag {6}
$$

Note that for local RR, there exists a trade-off between communication and epoch complexity in the choice of  $B$ . If  $B$  is too small, this reduces the number of epochs required but increases communication costs. On the other hand, if  $B$  is too large, this reduces communication rounds but errors that accumulate in local updates get severer, resulting in the need for more epochs. Hence, the optimal choice of  $B$  must balance the two complexity measures. The existence of this trade-off is indeed different from minibatch RR where larger  $B$  always reduces the total cost  $C_{\mathrm{minbatch}}(\epsilon)$ .

When can local RR match minibatch RR? Comparing the convergence rates (3) and (5), we can identify some regimes in which local RR converges as fast as minibatch RR. In a nutshell, if machines are not too heterogeneous and communication happens frequently, then local RR can have the same upper bound as minibatch RR. For example, if  $B$  is chosen to be a constant,  $M \lesssim N$ , and  $\tau \lesssim \nu \sqrt{N / M}$ , then the  $\tilde{\mathcal{O}}\left(\frac{L^2\nu^2}{\mu^3MNK^2}\right)$  term in (5) becomes the dominating factor and hence matches (3). Another example of such a regime is when  $B \lesssim \frac{N}{M}$  and  $\tau \lesssim \nu \sqrt{M / N}$ . Note that this comparison assumes that the same values of  $B$  are chosen for both algorithms. Also, such "frequent communication" regimes are favorable if the communication cost  $c_{c}$  is small.

Can local RR ever beat minibatch RR? The upper bounds (3) and (5) indicate that local RR is always no better than minibatch RR, at least for the function class  $\mathcal{F}_{\mathrm{obj}}(L,\mu ,\nu ,\tau ,\rho)$ . This is in fact consistent with Woodworth et al. (2020a;b), because the authors identify a regime where local SGD performs better than minibatch SGD for convex objective functions, but fail to do so for strongly convex functions. However, as was also pointed out in Woodworth et al. (2020a), there is a simple extreme scenario in which local RR can be faster: when  $\nu \approx \tau \approx 0$  and  $\rho \approx 1$ . In this case, we have  $f_{i}^{m}\approx F$  for all  $m$  and  $i$ , so local RR corresponds to  $NK$  steps of GD, whereas minibatch RR corresponds to  $\frac{NK}{B}$  steps of GD. Clearly, local RR will converge faster, exploiting the advantage of more updates. Of course, this extreme example is not so relevant to practice, so we do not pursue formally proving this; however, it demonstrates the existence of a regime where local RR can be faster than minibatch RR. Finding out other such regimes is an important future direction.

# 4 MATCHING LOWER BOUNDS

In Section 3, we presented large-epoch upper bounds (i.e., for  $K \gtrsim \kappa$ ) for constant step-size minibatch and local RR. In this section, we prove matching lower bounds to show that the upper bounds are tight, in all factors except  $L$  and  $\mu$ . We use  $\Omega(\cdot)$  to hide universal constants in lower bounds.

# 4.1 LOWERBOUNDFORMINIBATCHRR

Theorem 3 (Lower bound for minibatch RR). Suppose that minibatch RR has parameters satisfying Assumption 1. Additionally, assume that  $N$  is a multiple of 2. Then, there exist large enough constants  $c_{1}, c_{2} > 0$  such that the following holds: For  $L$  and  $\mu$  satisfying  $\kappa = \frac{L}{\mu} \geq c_{1}$ , there exists a function  $F \in \mathcal{F}_{\mathrm{cmp}}(L, \mu, \nu, 0)$  such that for any constant step-size  $\eta$ ,

$$
\mathbb {E} \left[ F \left(\boldsymbol {x} _ {K, \frac {N}{K}}\right) - F ^ {*} \right] = \left\{ \begin{array}{l l} \Omega \left(\frac {\nu^ {2}}{\mu M N K}\right) & \text {i f} K <   c _ {2} \kappa , \\ \Omega \left(\frac {\nu^ {2}}{\mu M N K ^ {2}}\right) & \text {i f} K \geq c _ {2} \kappa . \end{array} \right. \tag {7}
$$

Proof. We prove Theorem 3 in Appendix C. The proof is an extension of Rajput et al. (2020); Safran & Shamir (2020; 2021) to minibatch RR. We will sketch some key intuitions after Theorem 4.  $\square$

First notice that the function  $F$  is from  $\mathcal{F}_{\mathrm{cmp}}(L,\mu ,\nu ,0)$ , where all the machines are component-wise homogeneous. As seen in Definition 1,  $\mathcal{F}_{\mathrm{cmp}}(L,\mu ,\nu ,0)\subset \mathcal{F}_{\mathrm{obj}}(L,\mu ,\nu ,\tau ,\rho)$  for any  $\tau \geq 0$  and  $\rho \geq 1$ , so Theorem 3 provides a lower bound for  $\mathcal{F}_{\mathrm{cmp}}(\cdot)$  and  $\mathcal{F}_{\mathrm{obj}}(\cdot)$ , with arbitrary heterogeneity constants. We assume that  $N$  is even because we construct functions  $g_{1}$  and  $g_{2}$  such that  $f_{i}^{m}\coloneqq g_{1}$  if  $i\leq \frac{N}{2}$ , and  $f_{i}^{m}\coloneqq g_{2}$  if  $i > \frac{N}{2}$ . One can remove this assumption by using a zero function when  $N$  is odd (see e.g., Safran & Shamir (2020)). It is rather unsatisfactory that our theorem requires large enough constants  $c_{1}$  and  $c_{2}$ ; we believe a tighter analysis can relax this restriction.

Theorem 3 proves lower bounds for two different regimes:  $K \gtrsim \kappa$  and  $K \lesssim \kappa$ . In the large-epoch regime ( $K \gtrsim \kappa$ ), we can observe that the lower bound  $\Omega\left(\frac{\nu^2}{\mu MNK^2}\right)$  matches the upper bound (3) in Theorem 1, modulo a factor of  $\kappa^2$ . Tightening the  $\kappa^2$  gap between upper and lower bounds is left for future work. In the small-epoch regime ( $K \lesssim \kappa$ ), we observe that the lower bound  $\Omega\left(\frac{\nu^2}{\mu MNK}\right)$  exactly matches the convergence rate of (with-replacement) minibatch SGD; hence, the lower bound implies that minibatch RR has no hope for faster convergence than minibatch SGD, at least in the constant step-size and small-epoch regime. This observation is in line with Safran & Shamir (2021).

Upper bounds for  $K \lesssim \kappa$ ? Even for single-machine RR ( $M = 1$ ), proving an upper bound that matches the small-epoch lower bound  $\Omega\left(\frac{\nu^2}{\mu NK}\right)$  still remains a challenge. Nagaraj et al. (2019, Theorem 2) prove an upper bound for non-quadratic strongly convex functions that matches  $\Omega\left(\frac{\nu^2}{\mu NK}\right)$  if  $NK \gtrsim \kappa^2$ ; however, they use suffix averaging, so it is not directly comparable to Theorem 3 which considers last iterates. Safran & Shamir (2021) prove upper bounds for quadratic strongly convex functions, but assume that their Hessian matrices commute. For noncommutative cases, proving a small-epoch upper bound seems to require some form of matrix AM-GM inequalities, whose availability is an open problem (Recht & Ré, 2012; Lai & Lim, 2020; De Sa, 2020; Yun et al., 2021).

Remark 1 (Strong convexity in construction). We note that all lower bounds in this paper are constructed with strongly convex functions, a stronger assumption than PL functions (2). Thus, our lower bounds are also applicable to strong convexity counterparts of  $\mathcal{F}_{\mathrm{obj}}(\cdot)$  and  $\mathcal{F}_{\mathrm{cmp}}(\cdot)$ .

# 4.2 LOWER BOUNDS FOR LOCAL RR

In this subsection, we present lower bounds for local RR. We prove two bounds that correspond to homogeneous and heterogeneous cases. By combining the two bounds, we get a lower bound that matches our upper bound (5) in Theorem 2.

Theorem 4 (Lower bound for local RR: homogeneous case). Suppose that local RR has parameters satisfying Assumption 1. Additionally, assume that  $B$  is a multiple of 4 and  $MB \leq NK$ . Then, there exist large enough constants  $c_{3}, c_{4} > 0$  such that the following holds: For  $L$  and  $\mu$  satisfying  $\kappa = \frac{L}{\mu} \geq c_{3}$ , there exists a function  $F \in \mathcal{F}_{\mathrm{cmp}}(L, \mu, \nu, 0)$  such that for any constant step-size  $\eta$ ,

$$
\mathbb {E} \left[ F \left(\boldsymbol {y} _ {K, \frac {N}{K}}\right) - F ^ {*} \right] = \left\{ \begin{array}{l l} \Omega \left(\frac {\nu^ {2}}{\mu M N K}\right) & \text {i f} K <   c _ {4} \kappa , \\ \Omega \left(\frac {\nu^ {2}}{\mu M N K ^ {2}} + \frac {\nu^ {2} B}{\mu N ^ {2} K ^ {2}}\right) & \text {i f} K \geq c _ {4} \kappa . \end{array} \right. \tag {8}
$$

Proof. The proof is in Appendix E. For the large-epoch lower bounds in Theorems 3 and 4, we use "skewed" quadratics  $f_{i}^{m}(x) = (L1_{x\leq 0} + \mu 1_{x > 0})\frac{x^{2}}{2} +z_{i}\nu x$ , where  $z_{i} = +1$  if  $i\leq \frac{N}{2}$  and  $z_{i} = -1$  otherwise. For  $x\approx 0$ , the imbalance results in a "drift" towards positive  $x$ , whose strength is approximately proportional to the absolute value of partial sums of random permutations over  $\frac{N}{2} +1$ 's and  $\frac{N}{2} -1$ 's. By averaging the sums over  $M$  machines (minibatch RR), their absolute values shrink by  $\frac{1}{\sqrt{M}}$ ; in contrast, if each machine makes local updates (local RR), the magnitude of the drift cannot be reduced with  $M$ , because we average after local iterates already have taken  $B$  "big" steps. The proof uses techniques from Rajput et al. (2020).

Proposition 5 (Lower bound for local RR: heterogeneous case). Suppose that local RR has parameters satisfying Assumption 1. Additionally, assume that  $B$  is a multiple of 2 and  $\kappa = \frac{L}{\mu} \geq 2$ . Then, there exists a function  $F \in \mathcal{F}_{\mathrm{obj}}(L, \mu, 0, \tau, 1)$  such that for any constant step-size  $\eta$ ,

$$
\mathbb {E} \left[ F \left(\boldsymbol {y} _ {K, \frac {N}{K}}\right) - F ^ {*} \right] = \Omega \left(\frac {\tau^ {2} B ^ {2}}{\mu N ^ {2} K ^ {2}}\right). \tag {9}
$$

Proof. We note that Proposition 5 is almost identical to Theorem II of Karimireddy et al. (2020); however, we provide a proof specific to our algorithm in Appendix G.  $\square$

Theorem 4 constructs a component-wise homogeneous function from  $\mathcal{F}_{\mathrm{cmp}}(L,\mu ,\nu ,0)$  and Proposition 5 constructs a heterogeneous function from  $\mathcal{F}_{\mathrm{obj}}(L,\mu ,0,\tau ,1)$ . Since  $\mathcal{F}_{\mathrm{cmp}}(L,\mu ,\nu ,0)\cup$ $\mathcal{F}_{\mathrm{obj}}(L,\mu ,0,\tau ,1)\subset \mathcal{F}_{\mathrm{obj}}(L,\mu ,\nu ,\tau ,\rho)$  for any  $\rho \geq 1$ , combining (8) and (9) for the  $K\gtrsim \kappa$  case gives a lower bound  $\Omega (\max \{\frac{\nu^2}{\mu MNK^2} +\frac{\nu^2B}{\mu N^2K^2},\frac{\tau^2B^2}{\mu N^2K^2}\})$  that matches the large-epoch upper bound (5) in Theorem 2, up to a factor of  $\kappa^2$ . Tightening the  $\kappa^2$  gap as well as removing some additional assumptions such as  $\kappa \geq c_3$  and  $MB\leq NK$  are left for future work.

Using  $B = \Theta(N)$  does not help, indeed. In Section 3.2.1, we observed that if  $B = \Theta(N)$ , then even in the homogeneous case ( $\tau = 0$ ), local RR converges at the rate of  $\tilde{\mathcal{O}}\left(\frac{1}{NK^2}\right)$ . This is the same rate as single-machine RR, meaning the efforts by  $M - 1$  machines become meaningless. Our lower bound (8) shows that  $\tilde{\mathcal{O}}\left(\frac{1}{NK^2}\right)$  is in fact the best we can hope for (treating  $L$  and  $\mu$  as constants). In order to make the best use of  $M$  machines,  $B$  should be smaller than  $\Theta(N)$ , as suggested in Section 3.2.1. In an existing work, Mishchenko et al. (2021) consider local RR with  $B = N$  as a special case of a proximal algorithm. In Theorem 8 of Mishchenko et al. (2021), the authors claim "the convergence bound improves with the number of devices involved" because the bound has a factor of  $M$  in the denominator. However, at least under our assumption, this is not the case; if we apply our Assumption 3 to upper-bound their  $\sigma_*$ , the term " $N\sigma_*^2$ " in the numerator grows linearly with  $M$ . Hence, our bounds do not contradict Mishchenko et al. (2021); see Appendix A for details.

Remark 2 (Small-epoch bound is likely loose). We note that while we focused on deriving a matching large-epoch lower bound, we did not try hard to tighten the small-epoch lower bound. Our small-epoch lower bound in (8) misses a term (such as  $\frac{\nu^2 B}{\mu N^2 K^2}$ ) that corresponds to the error from local updates. We leave investigations on small-epoch lower and upper bounds for future work.

# 5 SYNCHRONIZED SHUFFLING: HOW TO BYPASS LOWER BOUNDS

Recall from the total complexity of minibatch RR (4) that the total cost shrinks with a factor of  $\frac{1}{\sqrt{M}}$ . Using  $M$  machines, we are only getting a  $\sqrt{M}$ -factor speedup. Ideally, we hope to see a linear speedup, i.e., cost inverse proportional to  $M$ . Hence, Theorem 1 falls short of achieving this goal, and our lower bound in Theorem 3 confirms that linear speedup is indeed impossible.

In this section, we show that the desired linear speedup is possible, at least in some special cases. We consider the component-wise near-homogeneous case (i.e., Assumption 4 with small  $\lambda$ ) and discuss how a simple modification to minibatch and local RR can let us "break" the lower bounds and achieve linear speedup. This comes at a cost of broadcasting permutations: at the beginning of the  $k$ -th epoch, the server samples  $\sigma \sim \mathrm{Unif}(S_N)$  and  $\pi \sim \mathrm{Unif}(S_M)$ , and broadcasts them to the machines. Then, local machines choose their permutations  $\sigma_k^m$  to be shifted versions of  $\sigma$ , i.e.,  $\sigma_k^m(i) \coloneqq \sigma\left((i + \frac{N}{M}\pi(m)) \bmod N\right)$ . We call this trick synchronized shuffling, denoted as

SYNCSHUF. Please revisit Algorithms 1 and 2 for the precise descriptions of the modified algorithms local RR with SYNCSHUF and minibatch RR with SYNCSHUF, respectively.

The intuition why this should help the homogeneous case is simple. In the proof of RR, we aggregate the component gradients over an epoch (i.e.,  $N$  iterations) to write it as a full gradient plus noise. If we are in the component-wise homogeneous setting and permutations are synchronized, then instead of aggregating  $N$  component gradients on a single machine, we can aggregate  $\frac{N}{M}$  component gradients on  $M$  machines to get a full gradient. This allows us to reduce the "noise" from without-replacement sampling. We emphasize here that we do not necessarily set  $B = \frac{N}{M}$  to get a full gradient every time; our analysis works for arbitrary  $B$  and  $M$ , as long as both divide  $N$ .

The idea of synchronized shuffling is in fact similar to approaches in distributed learning that shuffle and partition datasets and distribute them to local machines (see e.g., Lee et al. (2017); Meng et al. (2017)). In contrast, we do not communicate data, but communicate how to permute datasets stored in local machines. Meng et al. (2017, Theorem 3.3) provide an analysis for a distributed method similar to minibatch RR, but fail to show convergence to global minima in strongly convex cases.

# 5.1 UPPER BOUNDS FOR MINIBATCH AND LOCAL RR WITH SYNCSHUF

With SYNCSHUF, we can show that the  $M$  's appearing in the convergence rates ((3) and (5)) in Theorems 1 and 2 can be replaced with  $M^2$ , for a more stringent function class  $\mathcal{F}_{\mathrm{cmp}}(\cdot)$  that requires bounded component-wise inter-machine deviation (Assumption 4).

Theorem 6 (Upper bound for minibatch RR with SYNCSHUF). Suppose that minibatch RR with SYNCSHUF has parameters satisfying Assumption 1. Additionally assume that  $M$  divides  $N$ . For any  $F \in \mathcal{F}_{\mathrm{cmp}}(L, \mu, \nu, \lambda)$ , consider running the algorithm using step-size  $\eta = \frac{B \log(M^2 NK^2)}{\mu NK}$  for epochs  $K \geq 6\kappa \log(M^2 NK^2)$ . Then, with probability at least  $1 - \delta$ ,

$$
F \left(\boldsymbol {x} _ {K, \frac {N}{B}}\right) - F ^ {*} \leq \frac {F \left(\boldsymbol {x} _ {0}\right) - F ^ {*}}{M ^ {2} N K ^ {2}} + \tilde {\mathcal {O}} \left(\frac {L ^ {2}}{\mu^ {3}} \left(\frac {\nu^ {2}}{M ^ {2} N K ^ {2}} + \frac {\lambda^ {2}}{M K ^ {2}}\right)\right). \tag {10}
$$

The proof of Theorem 6 is presented in Appendix B.4. One can check that if the component-wise deviation constant  $\lambda$  satisfies  $\lambda \lesssim \frac{\nu}{\sqrt{MN}}$  (i.e., near-homogeneous), then the rate (10) becomes  $\tilde{\mathcal{O}}\left(\frac{1}{M^2NK^2}\right)$ . It is then easy to confirm that  $M$  machines reduce total costs by  $\frac{1}{M}$ -a linear speedup.

A similar speedup can be shown for local RR. In Appendix B.5, we prove that

Theorem 7 (Upper bound for local RR with SYNCSHUF). Suppose that local RR with SYNCSHUF has parameters satisfying Assumption 1. Additionally assume that  $M$  divides  $N$ . For any  $F \in \mathcal{F}_{\mathrm{cmp}}(L, \mu, \nu, \lambda)$ , consider running the algorithm with step-size  $\eta = \frac{\log(M^2NK^2)}{\mu NK}$  for epochs  $K \geq 7\kappa \log(M^2NK^2)$ . Then, with probability at least  $1 - \delta$ ,

$$
F \left(\boldsymbol {y} _ {K, \frac {N}{B}}\right) - F ^ {*} \leq \frac {F \left(\boldsymbol {y} _ {0}\right) - F ^ {*}}{M ^ {2} N K ^ {2}} + \tilde {\mathcal {O}} \left(\frac {L ^ {2}}{\mu^ {3}} \left(\frac {\nu^ {2}}{M ^ {2} N K ^ {2}} + \frac {\nu^ {2} B}{N ^ {2} K ^ {2}} + \frac {\lambda^ {2} B ^ {2}}{N ^ {2} K ^ {2}} + \frac {\lambda^ {2}}{M K ^ {2}}\right)\right). \tag {11}
$$

We can similarly check that if  $B \lesssim \frac{N}{M^2}$  and  $\lambda \lesssim \frac{\nu}{\sqrt{MN}}$ , i.e., frequent communication and nearhomogeneity, then the  $\tilde{\mathcal{O}}\left(\frac{1}{M^2NK^2}\right)$  term dominates in (11), and hence gives a linear speedup that matches the best rate of minibatch RR with SYNCSHUF (10). Nevertheless, we note again that for local RR, such a small  $B$  is favorable only when the communication cost  $c_c$  is small (recall (6)).

# 6 CONCLUSION

We studied convergence bounds for local RR and minibatch RR, which are the practical without-replacement versions of local and minibatch SGD studied in the theory literature. For smooth functions satisfying the Polyak-Lojasiewicz condition, we showed large-epoch convergence bounds for minibatch and local RR that are faster than their with-replacement counterparts. We also proved matching lower bounds showing that our convergence analysis is tight. We also proposed a simple modification called synchronized shuffling that leads to convergence rates faster than our lower bounds in near-homogeneous settings, at a cost of a slight increase in communication. Immediate future research directions include extension to small-epoch regimes, as well as to general convex and nonconvex functions.

# ETHICS STATEMENT

This paper develops theoretical guarantees for popular distributed stochastic optimization algorithms. Therefore, the authors do not see any particular concerns related to its ethical aspects or future societal consequences.

# REPRODUCIBILITY STATEMENT

This paper is a theoretical work, without any experimental results. Definitions and assumptions are provided in Section 2. Our theoretical contributions as well as some additionally required assumptions are clearly stated in Sections 3, 4, and 5. Complete proofs of all the theorems are provided in the appendix.

# REFERENCES

Kwangjun Ahn, Chulhee Yun, and Suvrit Sra. SGD with shuffling: optimal rates without component convexity and large epoch requirements. In Advances in Neural Information Processing Systems, 2020.  
Léon Bottou. Curiously fast convergence of some stochastic gradient descent algorithms. In Proceedings of the symposium on learning and data science, Paris, 2009.  
Christopher M De Sa. Random reshuffling is not always better. Advances in Neural Information Processing Systems, 33, 2020.  
Aymeric Dieuleveut and Kumar Kshitij Patel. Communication trade-offs for local-sgd with large step size. Advances in Neural Information Processing Systems, 32:13601-13612, 2019.  
Mert Gürbüzbalaban, Asu Ozdaglar, and Pablo Parrilo. Why random reshuffling beats stochastic gradient descent. Mathematical Programming, pp. 1-36, 2019.  
Farzin Haddadpour and Mehrdad Mahdavi. On the convergence of local descent methods in federated learning. arXiv preprint arXiv:1910.14425, 2019.  
Farzin Haddadpour, Mohammad Mahdi Kamani, Mehrdad Mahdavi, and Viveck Cadambe. Local sgd with periodic averaging: Tighter analysis and adaptive synchronization. Advances in Neural Information Processing Systems, 32:11082-11094, 2019.  
Jeff Haochen and Suvrit Sra. Random shuffling beats SGD after finite epochs. In International Conference on Machine Learning, pp. 2624-2633, 2019.  
Prateek Jain, Dheeraj Nagaraj, and Praneeth Netrapalli. Making the last iterate of sgd information theoretically optimal. In Conference on Learning Theory, pp. 1752-1755. PMLR, 2019.  
Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, Rafael G. L. D'Oliveira, Hubert Eichner, Salim El Rouayheb, David Evans, Josh Gardner, Zachary Garrett, Adrià Gascon, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaid Harchaoui, Chaoyang He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara Javidi, Gauri Joshi, Mikhail Khodak, Jakub Konecný, Aleksandra Korolova, Farinaz Koushanfar, Sanmi Koyejo, Tancrede Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock, Ayfer Özgür, Rasmus Pagh, Hang Qi, Daniel Ramage, Ramesh Raskar, Mariana Raykova, Dawn Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tramèr, Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu, and Sen Zhao. Advances and open problems in federated learning. Foundations and Trends® in Machine Learning, 14(1-2):1-210, 2021. ISSN 1935-8237. doi: 10.1561/2200000083. URL http://dx.doi.org/10.1561/2200000083.  
Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In International Conference on Machine Learning, pp. 5132-5143. PMLR, 2020.

Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. Tighter theory for local sgd on identical and heterogeneous data. In International Conference on Artificial Intelligence and Statistics, pp. 4519-4529. PMLR, 2020.  
Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian Stich. A unified theory of decentralized sgd with changing topology and local updates. In International Conference on Machine Learning, pp. 5381-5393. PMLR, 2020.  
Jakub Konečný, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave Bacon. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492, 2016.  
Zehua Lai and Lek-Heng Lim. Recht-Ré noncommutative arithmetic-geometric mean conjecture is false. In International Conference on Machine Learning, 2020.  
Kangwook Lee, Maximilian Lam, Ramtin Pedarsani, Dimitris Papailiopoulos, and Kannan Ramchandran. Speeding up distributed machine learning using codes. IEEE Transactions on Information Theory, 64(3):1514-1529, 2017.  
Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. Federated learning: Challenges, methods, and future directions. IEEE Signal Processing Magazine, 37(3):50-60, 2020a.  
Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. In I. Dhillon, D. Papailiopoulos, and V. Sze (eds.), Proceedings of Machine Learning and Systems, volume 2, pp. 429-450, 2020b. URL https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf.  
Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of fedavg on non-iid data. In International Conference on Learning Representations, 2020c.  
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273-1282. PMLR, 2017.  
Qi Meng, Wei Chen, Yue Wang, Zhi-Ming Ma, and Tie-Yan Liu. Convergence analysis of distributed stochastic gradient descent with shuffling. arXiv preprint arXiv:1709.10432, 2017.  
Konstantin Mishchenko, Ahmed Khaled, and Peter Richtárik. Random reshuffling: Simple analysis with vast improvements. arXiv preprint arXiv:2006.05988, 2020.  
Konstantin Mishchenko, Ahmed Khaled, and Peter Richtárik. Proximal and federated random reshuffling. arXiv preprint arXiv:2102.06704, 2021.  
Dheeraj Nagaraj, Prateek Jain, and Praneeth Netrapalli. SGD without replacement: Sharper rates for general smooth convex functions. In International Conference on Machine Learning, pp. 4703-4711, 2019.  
Lam M. Nguyen, Quoc Tran-Dinh, Dzung T. Phan, Phuong Ha Nguyen, and Marten van Dijk. A unified convergence analysis for shuffling-type gradient methods. arXiv preprint arXiv:2002.08246, 2020.  
Iosif Pinelis. An approach to inequalities for the distributions of infinite-dimensional martingales. In Probability in Banach Spaces, 8: Proceedings of the Eighth International Conference, pp. 128-134. Springer, 1992.  
Iosif Pinelis. Optimum bounds for the distributions of martingales in banach spaces. The Annals of Probability, pp. 1679-1706, 1994.  
Zhaonan Qu, Kaixiang Lin, Jayant Kalagnanam, Zhaojian Li, Jiayu Zhou, and Zhengyuan Zhou. Federated learning's blessing: Fedavg has linear speedup. arXiv preprint arXiv:2007.05690, 2020.

Shashank Rajput, Anant Gupta, and Dimitris Papailiopoulos. Closing the convergence gap of SGD without replacement. In International Conference on Machine Learning, 2020.  
Shashank Rajput, Kangwook Lee, and Dimitris Papailiopoulos. Permutation-based sgd: Is random optimal? arXiv preprint arXiv:2102.09718, 2021.  
Alexander Rakhlin, Ohad Shamir, and Karthik Sridharan. Making gradient descent optimal for strongly convex stochastic optimization. In Proceedings of the 29th International Coference on International Conference on Machine Learning, pp. 1571-1578, 2012.  
Benjamin Recht and Christopher Ré. Toward a noncommutative arithmetic-geometric mean inequality: conjectures, case-studies, and consequences. In Conference on Learning Theory, pp. 11-1, 2012.  
Itay Safran and Ohad Shamir. How good is SGD with random shuffling? In Conference on Learning Theory, pp. 3250-3284. PMLR, 2020.  
Itay Safran and Ohad Shamir. Random shuffling beats SGD only after many epochs on ill-conditioned problems. arXiv preprint arXiv:2106.06880, 2021.  
Markus Schneider. Probability inequalities for kernel embeddings in sampling without replacement. In Artificial Intelligence and Statistics, pp. 66-74, 2016.  
Robert J Serfling. Probability inequalities for the sum in sampling without replacement. The Annals of Statistics, pp. 39-48, 1974.  
Artin Spiridonoff, Alex Olshovsky, and Ioannis Ch Paschalidis. Local sgd with a communication overhead depending only on the number of workers. arXiv preprint arXiv:2006.02582, 2020.  
Sebastian U Stich. Local sgd converges fast and communicates little. In International Conference on Learning Representations, 2019.  
Sebastian U Stich and Sai Praneeth Karimireddy. The error-feedback framework: Better rates for sgd with delayed gradients and compressed updates. Journal of Machine Learning Research, 21: 1-36, 2020.  
Trang H Tran, Lam M Nguyen, and Quoc Tran-Dinh. Smg: A shuffling gradient-based method with momentum. In International Conference on Machine Learning, pp. 10379-10389. PMLR, 2021.  
Blake Woodworth, Kumar Kshitij Patel, Sebastian Stich, Zhen Dai, Brian Bullins, Brendan Mcmahan, Ohad Shamir, and Nathan Srebro. Is local SGD better than minibatch SGD? In International Conference on Machine Learning, pp. 10334-10343. PMLR, 2020a.  
Blake Woodworth, Brian Bullins, Ohad Shamir, and Nathan Srebro. The min-max complexity of distributed stochastic convex optimization with intermittent communication. arXiv preprint arXiv:2102.01583, 2021.  
Blake E Woodworth, Kumar Kshitij Patel, and Nati Srebro. Minibatch vs local SGD for heterogeneous distributed learning. Advances in Neural Information Processing Systems, 33:6281-6292, 2020b.  
Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel restarted sgd with faster convergence and less communication: Demystifying why model averaging works for deep learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 5693-5700, 2019.  
Chulhee Yun, Suvrit Sra, and Ali Jabbabaie. Open problem: Can single-shuffle SGD be better than reshuffling SGD and GD? In Conference on Learning Theory, pp. 4653-4658. PMLR, 2021.
