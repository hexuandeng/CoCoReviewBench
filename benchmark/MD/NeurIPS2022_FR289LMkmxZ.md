# On-Demand Sampling: Learning Optimally from Multiple Distributions

Anonymous Author(s) Affiliation Address email

# Abstract

Social and real-world considerations such as robustness, fairness, social welfare and multi-agent tradeoffs have given rise to multi-distribution learning paradigms, such as collaborative [4], group distributionally robust [22], and fair federated learning [15]. In each of these settings, a learner seeks to minimize its worst-case loss over a set of  $n$  predefined distributions, while using as few samples as possible. In this paper, we establish the optimal sample complexity of these learning paradigms and give algorithms that meet this sample complexity. Importantly, our sample complexity bounds exceed that of the sample complexity of learning a single distribution only by an additive factor of  $\frac{n\log(n)}{\varepsilon^2}$ . These improve upon the best known sample complexity of agnostic federated learning by Mohri et al. [15] by a multiplicative factor of  $n$ , the sample complexity of collaborative learning by Nguyen and Zakynthinou [17] by a multiplicative factor  $\frac{\log n}{\varepsilon^3}$ , and give the first sample complexity bounds for the group DRO objective of Sagawa et al. [22]. To achieve optimal sample complexity, our algorithms learn to sample and learn from distributions on demand. Our algorithm design and analysis is enabled by our extensions of stochastic optimization techniques for solving stochastic zero-sum games. In particular, we contribute variants of Stochastic Mirror Descent that can trade off between players' access to cheap one-off samples or more expensive reusable ones.

# 1 Introduction

Pervasive needs for robustness, fairness, and multi-agent collaboration in learning have given rise to multi-distribution learning paradigms (e.g., [4, 22, 15, 9]). In these settings, we seek to learn a model that performs well on any distribution in a pre-defined set of interest. For fairness considerations, these distributions may represent heterogeneous populations of different protected or socio-economic attributes; in robustness applications, they may capture a learner's uncertainty regarding the true underlying task; and in multi-agent collaborative or federated applications, they may represent agent-specific learning tasks. In these applications, the performance and optimality of a model is measured by its worst test-time performance on a distribution in the set. We are concerned with this fundamental problem of designing sample-efficient multi-distribution learning algorithms.

The sample complexity of multi-distribution learning differs from that of learning a single distribution in several ways. On one hand, learning tasks of varying difficulty require different numbers of samples. On the other hand, similarity or overlap among learning tasks may obviate the need to sample from some distributions. This makes the use of a fixed per-distribution sample budget highly inefficient and suggests that optimal multi-distribution learning algorithms should sample on demand. That is, algorithms should take additional samples whenever they need them and from whichever distribution they want them. On-demand sampling is especially appropriate when

<table><tr><td>Problem</td><td>Sample Complexity</td><td>Thm</td><td>Best Previous Result</td></tr><tr><td>Collab. Learning UB</td><td>ε-2(log |H| + n log(n/δ))</td><td>[4.1]</td><td>ε-5 log (1/ε) log(n/δ) (log |H| + n) [17]</td></tr><tr><td>Collab. Learning LB</td><td>ε-2(log |H| + n log(k/δ))</td><td>[4.3]</td><td>ε-1n log(k/δ) [4]</td></tr><tr><td>GDRO/AFL UB</td><td>ε-2(log |H| + n log(n/δ))</td><td>[4.1]</td><td>ε-2(n log |H| + n log(n/δ)) [15]</td></tr><tr><td>GDRO/AFL UB</td><td>ε-2(DH + n log(n/δ))</td><td>[5.1]</td><td>N/A</td></tr><tr><td>(Training error convg.)</td><td>ε-2(DH + n log(n/δ))</td><td>[5.2]</td><td>ε-2DH (expected convergence only) [22]</td></tr></table>

Table 1: This table gives upper (UB) and lower bounds (LB) on the sample complexity of learning model class  $H$  on  $n$  distributions. For the collaborative learning and AFL settings, the sample complexity upper bounds refer to the problem of learning a randomized model of worst-case error OPT +  $\varepsilon$  or a deterministic classifier of worst-case error 2OPT +  $\varepsilon$ . For the GDRO setting, sample complexity refers to learning a deterministic model with worst-case error of R-OPT +  $\varepsilon$ , where R-OPT is the best worst-case error attainable in a convex compact model space  $H$ .  $D_{\mathcal{H}}$  denotes the Bregman radius of  $H$ , and  $k = \min \{n, \log |\mathcal{H}|\}$ .

some population data may be scarce to start with (as in fairness mechanisms in which samples are amended [19]); when the designer can actively perturb datasets towards rare or atypical instances (such as in robustness applications [13, 25]); or when sample sets represent agents' contributions to an interactive multi-agent system [15, 5]. Blum et al. [4] demonstrated the benefit of on-demand sampling in the collaborative learning setting, where all data distributions are realizable with respect to the same target classifier. This line of work established that learning  $n$  distributions on-demand takes  $\widetilde{O}(\log(n))$  times the sample complexity of learning a single realizable distribution [4, 6, 17], whereas relying on batched uniform convergence takes  $\widetilde{\Omega}(n)$  times that of learning a single distribution [4]. However, beyond the realizable setting, the best known multi-distribution learning results fall short of this promise: existing on-demand sample complexity bounds for agnostic collaborative learning have highly suboptimal dependence on  $\varepsilon$ , requiring  $\widetilde{O}\left(\log(n)/\varepsilon^{3}\right)$  times the sample complexity of agnostically learning a single distribution [17]. On the other hand, agnostic federated learning bounds [15] have been studied only on algorithms that sample in one large batch and thus require  $\widetilde{\Omega}(n)$  times the sample complexity of learning a single task. Moreover, the test-time performance of some key multi-distribution methods, such as group distributionally robust optimization [22], have not been studied from a theoretical perspective before.

In this paper, we give a general framework for obtaining optimal and on-demand sample complexity for three multi-distribution learning settings. Table 1 summarizes our results. All three settings consider a set  $\mathcal{D}$  of  $n$  distributions and a model class  $\mathcal{H}$ . They evaluate the performance of a model  $h$  (or a distribution over models) by its worst-case performance,  $\max_{D\in \mathcal{D}}\log_D(h)$ . As a benchmark, they consider the worst-case loss of the best model, i.e.,  $\mathrm{OPT} = \min_{h^* \in \mathcal{H}}\max_{D\in \mathcal{D}}\log_D(h^*)$ . Importantly, all of our sample complexity upper bounds demonstrate only an additive increase of  $\varepsilon^{-2}n\log (n / \delta)$  over the sample complexity of learning a single task, compared to the multiplicative factor increase required by existing works.

- Collaborative learning of Blum et al. [4]: For agnostic collaborative learning, our Theorem 4.1 gives a randomized and a deterministic model that achieve performance guarantees of  $\mathrm{OPT} + \varepsilon$  and  $2\mathrm{OPT} + \varepsilon$ , respectively. Our algorithms have an optimal sample complexity of  $O\left(\frac{1}{\varepsilon^2} (\log (|H|) + n\log (\frac{n}{\delta}))\right)$ . This improves upon the work of Nguyen and Zakynthinou [17] in two ways. First, it provides error bounds of  $\mathrm{OPT} + \varepsilon$  for randomized classifiers, where only  $2\mathrm{OPT} + \varepsilon$  was previously established. Second, it improves the upper bound of Nguyen and Zakynthinou [17] by a multiplicative factor of  $\log (n) / \varepsilon^3$ . In Theorem 4.3, we give a matching lower bound on this sample complexity, thereby establishing the optimality of our algorithms.  
- Group distributionally robust learning (group DRO) of Sagawa et al. [22]: For group DRO, we consider a convex and compact model space  $\mathcal{H}$ . Our Theorem 5.1 studies a model that achieves an OPT + ε guarantee on the worst-case test-time performance of the model with an on-demand sample complexity of  $\mathcal{O}\left(\frac{1}{\varepsilon^2} (D_H + n\log \left(\frac{n}{\delta}\right)\right)$ . Our results also imply a high-probability bound

for the convergence of group DRO training error that improves upon the (expected) convergence guarantees of Sagawa et al. [22] by a factor of  $n$ .

- Agnostic federated learning of [15]: For agnostic federated learning, we consider a finite class of hypotheses. Our Theorems 4.1 and 5.1 show that on-demand sampling can accelerate the generalization of agnostic federated learning by a factor of  $n$  compared to batch results established by Mohri et al. [15]. Our results also imply matching high-probability bounds to Mohri et al. [15] on the convergence of the training error in the batched setting.

To achieve these results, we contribute new insights and techniques for solving stochastic zero-sum games with sources of randomization that differ in both cost and quality. We frame the multi-distribution learning problems as a stochastic zero-sum game with uncertain payoffs and utilize stochastic mirror descent and a variational perspective to solve the game. In this case, the maximizing player can be interpreted as a weight vector for distributions  $\mathcal{D}$ , specifying from which distributions future on-demand samples should be taken. These on-demand samples form a stochastic gradient for the players. However, the quality of these estimators, the number of samples needed for them, and whether they can be reused later on, differs between the two players. We extend the Stochastic Mirror Descent framework to optimally trade off these asymmetric needs for samples. In Section 3 we give an overview of this approach and its technical challenges and contributions.

# 1.1 Related Work

Learning models. Three independent lines of work study multi-distribution learning, with different motivating applications. Collaborative learning interprets multiple distributions as players that each seek to learn a model with low error on their data distributions [4, 17, 6]. Agnostic federated learning interprets these distributions as clients in a federated learning system [15]. Group distributionally robust optimization interprets these distributions as data attributes or sources that a learner should avoid linking spuriously to labels [12, 22, 23]. Formally, these learning objectives are all equivalent but have been studied from different points of view and with different technical tools.

Existing work on group DRO has assumed that data is pre-collected and has studied the convergence of multi-distribution training error. The agnostic federated learning literature has studied a single-batch approach and derived data-dependent generalization bounds that suggest how much of the batch should be collected from each distribution. Finally, the collaborative learning literature has studied an on-demand framework for collecting data from each distribution. This approach also relates to a line of work on multi-source learning and domain adaptation [3, 14].

Stochastic game equilibria. Our approach relates to a line of research on using online algorithms to find min-max equilibria by playing no-regret algorithms against one another [21, 10, 18, 7, 8]. One such method, online mirror descent (OMD), can also approximate minima of convex functions with high probability using noisy first-order information [20, 16, 2]. This allows OMD to efficiently find min-max equilibria even in stochastic convex-concave zero-sum games [11]. We bring these online learning tools to bear on the problem of finding equilibria in robust optimization formulations.

# 2 Preliminaries

Let  $\mathcal{X}$  be an instance space,  $\mathcal{Y}$  a label space, and  $\mathcal{Z} = \mathcal{X} \times \mathcal{Y}$  a space of datapoints. A data distribution  $D$  is a joint probability distribution over  $\mathcal{Z}$ . We consider a hypothesis class  $\mathcal{H}$  of a subset of functions mapping  $\mathcal{X}$  to  $\mathcal{Y}$ . We work with loss functions  $\ell: \mathcal{H} \times \mathcal{Z} \to [0,1]$  that measure the loss of hypothesis  $h$  on data point  $z \in \mathcal{Z}$ . When  $\mathcal{Y} = \{0,1\}$ ,  $\ell$  is the misclassification error. We denote the expected loss, i.e. risk, of a hypothesis  $h \in \mathcal{H}$  under a data distribution  $D \in \mathcal{D}$  by:

$$
\operatorname {l o s s} _ {D} (h) := \underset {(x, y) \sim D} {\mathbb {E}} \left[ \ell \left(h, (x, y)\right) \right].
$$

For a distribution over the hypothesis class,  $p \in \Delta \mathcal{H}$ , and a distribution over data distributions,  $q \in \Delta \mathcal{D}$ , we refer to their expected loss by  $\mathrm{loss}_q(p) \coloneqq \mathbb{E}_{D \sim q}[\mathbb{E}_{h \sim p}[\mathrm{loss}_D(h)]]$ .

Collaborative Learning. We will use the collaborative PAC learning model of Blum et al. [4] and its agnostic extensions by Nguyen and Zakynthinou [17]. The overall goal of this setting is to guarantee low loss for every distribution in a collection of distributions. Formally, we consider a set

of data distributions  $\mathcal{D} \coloneqq \{D_1, \ldots, D_n\}$ . The goal of the learner is to learn a hypothesis  $h$  such that, with probability  $1 - \delta$ ,

$$
\max  _ {D \in \mathcal {D}} \operatorname {l o s s} _ {D} (h) \leq \mathrm {O P T} + \varepsilon , \text {w h e r e} \mathrm {O P T} := \min  _ {h \in \mathcal {H}} \max  _ {D \in \mathcal {D}} \operatorname {l o s s} _ {D} (h). \tag {1}
$$

Group Distribution Robustness. We will also study the closely related setting of group distributionally robust optimization (Group DRO) of Sagawa et al. [22]. Formally, the group DRO setting considers a model set  $\Theta$  that is a convex compact subset of the Euclidean space and a convex loss function  $\ell : \Theta \times \mathcal{Z} \to [0,1]$  that is assumed to be differentiable over  $\Theta$ . Given a set of data distributions  $\mathcal{D} := \{D_1, \ldots, D_n\}$ , the learner seeks a model  $\theta \in \Theta$ , such that, with probability  $1 - \delta$ ,

$$
\max  _ {D \in \mathcal {D}} \underset {(x, y) \sim D} {\mathbb {E}} [ \ell (\theta , (x, y)) ] \leq \mathrm {R} - \mathrm {O P T} + \varepsilon , \text {w h e r e R - O P T} := \min  _ {\theta \in \Theta} \max  _ {D \in \mathcal {D}} \underset {(x, y) \sim D} {\mathbb {E}} [ \ell (\theta , (x, y)) ]. \tag {2}
$$

There is a close relationship between the Group DRO setting and collaborative learning. In particular, when  $\Theta = \Delta (\mathcal{H})$  and  $\mathcal{H}$  is finite, the two goals are analogous but with two exceptions: first, the Group DRO could return a distribution over functions while collaborative learning requires the solution to be a deterministic function, and second, allowing for randomized hypothesis leads to R-OPT being potentially more competitive than OPT. We note that the group DRO setting is equivalent to the agnostic federated learning framework of [15], thus our results for DRO extend to that setting as well.

Sample complexity. We are interested in the design of algorithms that achieve the above goals while using smallest number of samples from distributions  $D_{1},\ldots ,D_{n}$ . We formalize the sample complexity by the total number of calls made to example oracles  $\mathrm{EX}(D_i)$ . Each call  $\mathrm{EX}(D)$  produces an i.i.d. sample from  $D$ . We note that these example oracles also allow us to sample from any mixture distribution  $q\in \Delta \mathcal{D}$ , e.g., by first selecting a  $D_{i}$  according to the mixture and then calling  $\mathrm{EX}(D_i)$ .

# 2.1 Technical Background

We will use tools and definitions from the literature on zero-sum games and no-regret learning throughout the paper. This section provides a brief overview of these concepts.

Zero-Sum Games. A finite two-player zero-sum game is described by the tuple  $(A_{-},A_{+},\phi)$  where  $A_{-} = \{1,\dots ,n\}$  and  $A_{+} = \{1,\dots ,m\}$  are finite sets of actions and where  $\phi :A_{-}\times A_{+}\to [0,C]$ . In this game, the players choose mixed strategies over actions sets. These are distributions that are denoted by a vector of probabilities  $p\in \Delta A_{-}$  and  $q\in \Delta A_{+}$ . The expected payoff of mixed strategies is denoted by  $\phi (p,q) = \mathbb{E}_{i\sim p,j\sim q}[\phi (i,j)]$ . The goal of the minimizing player is to minimize this expected payoff and the maximizer seeks to maximize the expected payoff; that is, to solve

$$
\min  _ {p \in \Delta A _ {-}} \max  _ {q \in \Delta A _ {+}} \phi (p, q).
$$

A pair  $(p,q)$  that solves this optimization problem is called a min-max equilibrium. Similarly, a solution is called an  $\varepsilon$ -min-max equilibrium if neither player can unilaterally improve their objective by more than  $\varepsilon$ . Formally,  $(p,q)$  is an  $\varepsilon$ -min-max equilibrium if both players' regrets are at most  $\varepsilon$ , i.e.,  $\operatorname{Reg-Min}(p,q) := \phi(p,q) - \min_{i^* \in A_+} \phi(i^*,q) \leq \varepsilon$  and  $\operatorname{Reg-Max}(p,q) := \max_{j^* \in A_+} \phi(p,j^*) - \phi(p,q) \leq \varepsilon$ . We will next describe methods that find  $\varepsilon$ -min-max equilibria by finding solutions  $(p,q)$  for which  $\operatorname{Reg-Min}(p,q) + \operatorname{Reg-Max}(p,q)$  is at most  $\varepsilon$ . We describe a more general formulation for convex-concave zero-sum games in Appendix A.1 which we will use for the Group DRO problem.

No-Regret Learning. We consider an online setting where an arbitrary set of operators,  $g^{(1)}, \ldots, g^{(T)} \in \mathcal{E}^*$ , is revealed sequentially to a learner who must choose a matching sequence of actions,  $w^{(1)}, \ldots, w^{(T)}$ , from a convex compact set  $Z \subseteq \mathcal{E}$ . Here,  $\mathcal{E}$  and  $\mathcal{E}^*$  respectively refer to an arbitrary Euclidean space and its dual. We focus on a setting where an online learner commits to action  $w^{(t)} \in Z$  before seeing  $g^{(t)}, g^{(t+1)}, \ldots$  and aims to achieve vanishing variational error  $\mathrm{Err}_{\mathrm{V}}(w^{(1:T)})$  defined by

$$
\operatorname {E r r} _ {\mathbf {V}} \left(w ^ {(1: T)}\right) := \max  _ {w ^ {*} \in Z} \frac {1}{T} \sum_ {t = 1} ^ {T} \left\langle g ^ {(t)}, w ^ {(t)} - w ^ {*} \right\rangle . \tag {3}
$$

We will denote no-regret algorithms by their update rule  $\mathcal{Q}:\{Z\times \mathcal{E}^*\} \to Z$ , where  $\{Z\times \mathcal{E}^*\}$  denotes the space of arbitrary length sequences of action-operator pairs. Given a history sequence  $w^{(1)},\ldots ,w^{(t)}\in Z$  and operator sequence  $g^{(1)},\ldots ,g^{(t)}\in \mathcal{E}^*$ , the algorithm returns  $w^{(t + 1)} = \mathcal{Q}\left(\left\{w^{(1)},g^{(1)}\right\} ,\dots ,\left\{w^{(t)},g^{(t)}\right\}\right)$ . When the history is clear from context, we write  $w^{(t + 1)} = \mathcal{Q}\left(w^{(t)},g^{(t)}\right)$  as shorthand. For the particular case where  $Z = \Delta^n$  is a probability simplex, one such algorithm is Exponential Gradient Descent (also known as Hedge):

$$
\mathcal {Q} _ {\text {h e d g e}} \left(\left\{w ^ {(1)}, g ^ {(1)} \right\}, \dots , \left\{w ^ {(t)}, g ^ {(t)} \right\}\right) := \frac {\widetilde {w}}{\| \widetilde {w} \| _ {1}} \text {w h e r e} \widetilde {w} _ {i} := w _ {i} ^ {(t)} \exp \left\{- \eta g _ {i} ^ {(t)} \right\}, \tag {4}
$$

where  $\eta$  is a user-defined step size, and  $w_{1}$  is a user-defined initial iterate. By default, we take  $w_{1} = \left[\frac{1}{n}\right]^{n}$ .

The following lemma is a classical result on the variational error of exponential gradient descent.

Lemma 2.1 ([24]). Let  $g^{(1)}, \ldots, g^{(T)} \in \mathbb{R}^n$  and  $Z = \Delta^n$ . Further assume  $\| g^{(t)} \|_{\infty} \leq C$  for all timesteps  $t = 1, \ldots, T$ . Choosing  $\eta = \log n / \sqrt{T}$ , after  $T$  iterations of exponential gradient descent, the output  $\{w\}_{t=1}^T$  satisfies,

$$
\operatorname {E r r} _ {\mathbf {V}} (w ^ {(1: T)}) \leq \frac {3 C}{2} \sqrt {\frac {K L (w ^ {(T)} | | w ^ {(1)})}{T}}.
$$

# 3 Technical Overview of Our Approach

In this section, we provide an overview of our technical approach for addressing the sample complexity of collaborative learning and group DRO problems. In later sections, we will refer to the approach outlined in this section to sketch proofs and design algorithms. We will focus our exposition on collaborative learning and briefly indicate how the same approach applies to the group DRO setting.

At a high level, we first frame collaborative learning as a zero-sum game with uncertain payoffs and aim to use a variational perspective to learn its minmax equilibrium. We specifically choose the variational perspective (instead of an arbitrary online learning approach), since it allows us to linearize the effect of uncertain payoffs on the resulting error. We then use stochastic gradients to solve the variational problem. Our stochastic gradients will rely on i.i.d. samples from the distributions to estimate gradients both with respect to distributions over  $\mathcal{H}$  and mixtures over  $\mathcal{D}$  but with an asymmetric bound on the bias and variance of the estimates. Along the way, we develop tools and formalisms that handle the asymmetric cost of stochastic gradients and obtain optimal sample complexity results. We now address these steps in more detail.

Collaborative Learning as Zero-Sum Games. When the hypothesis class  $\mathcal{H}$  is finite, the collaborative learning problem with distribution set  $\mathcal{D}$  corresponds to a zero-sum game  $(A_{-}, A_{+}, \phi)$  with  $A_{-} = \mathcal{H}$ ,  $A_{+} = \mathcal{D}$ ,  $\phi(i,j) = \mathrm{loss}_j(i)$ , such that the value of the min-max solution is equivalent to R-OPT. It is not hard to see that any  $\varepsilon$ -min-max equilibrium  $(p,q)$  of this game corresponds to a  $2\varepsilon$  collaborative learning solution, i.e.,

$$
\underset {h \sim p} {\mathbb {E}} \left[ \max  _ {D \in \mathcal {D}} \operatorname {l o s s} _ {D} (h) \right] \leq \mathrm {O P T} + 2 \varepsilon . \tag {5}
$$

This enables us to use tools that have been developed for solving zero-sum games in order to address collaborative learning and group DRO settings. We will use a similar construction when hypothesis class  $\mathcal{H}$  has finite VC dimension, where  $A_{-}$  will instead refer to an appropriate  $\varepsilon$ -cover of  $\mathcal{H}$ .

Using VI to deal with Payoff Uncertainty. A sufficient condition for minimizing regret, and thus finding  $\varepsilon$ -min-max equilibrium, is minimizing the variational error (Equation 3). In particular, for any finite zero-sum game  $(A_{-},A_{+},\phi)$ , defining  $Z = [\Delta A_{-},\Delta A_{+}]$  and operators

$$
g ^ {(t)} = \left[ \left\{\partial_ {p _ {i}} \phi \left(p ^ {(t)}, q ^ {(t)}\right) \right\} _ {i \in A _ {-}}, \left\{- \partial_ {q _ {j}} \phi \left(p ^ {(t)}, q ^ {(t)}\right) \right\} _ {j \in A _ {+}} \right], \tag {6}
$$

ensures that variational error provides an upper bound on regret:  $\mathrm{Err}_{\mathbf{V}}(w^{(1:T)}) \geq \mathrm{Reg - Min}(p,q) + \mathrm{Reg - Max}(p,q)$ , where  $w = (p,q)$  (see Lemma B.1). In collaborative learning, when  $p^{(t)}$  is the

min-player's distribution over hypotheses and  $q^{(t)}$  is max-player's distribution over the mixtures, the gradient vectors refer to

$$
g ^ {(t)} = \left[ g _ {-} ^ {(t)}, g _ {+} ^ {(t)} \right], \quad g _ {-} ^ {(t)} = \left\{\operatorname {l o s s} _ {q ^ {(t)}} (h) \right\} _ {h \in \mathcal {H}}, \quad g _ {+} ^ {(t)} = \left\{\operatorname {l o s s} _ {D} (p ^ {(t)}) \right\} _ {D \in \mathcal {D}}. \tag {7}
$$

In the collaborative learning setting, we can only create noisy estimates  $\widehat{g}$  for these gradients from samples. This is where no-regret algorithms that minimize variational error become advantageous. By linearizing the effect of noise,  $\varepsilon^{(t)}\coloneqq g^{(t)} - \widehat{g}^{(t)}$ , they decompose the variational error into the training and generalization error as follows

$$
\operatorname {E r r} _ {\mathbf {V}} \left(w ^ {(1: T)}\right) \leq \max  _ {w ^ {*} \in \Delta^ {n}} \frac {1}{T} \sum_ {t = 1} ^ {T} \left\langle \widehat {g} ^ {(t)}, w ^ {(t)} - w ^ {*} \right\rangle + \max  _ {w ^ {*} \in \Delta^ {n}} \frac {1}{T} \sum_ {t = 1} ^ {T} \left\langle \varepsilon^ {(t)}, w ^ {(t)} - w ^ {*} \right\rangle . \tag {8}
$$

In contrast, generic no-regret algorithms that do not solve the variational inequality (e.g., when one player plays Hedge and another plays clairvoyant best-response as used in existing work in collaborative learning due to Blum et al. [4], Nguyen and Zakynthinou [17], Chen et al. [6]) nest the generalization and training errors which leads to a multiplicative increase in sample complexity.

Leveraging Noisy Stochastic Gradients. We will work with stochastic estimators of  $g$ . These are functions  $\widehat{g} : \xi \times \Delta A_{-} \times \Delta A_{+}$  of some external source of randomness,  $\xi \in \xi$ , and a strategy profile of interest. For collaborative learning, the randomness source  $\xi$  is an i.i.d.-sampled data point from an appropriate mixture of distributions and the estimator  $\widehat{g}$  is then the empirical loss on this sample, which is an unbiased and bounded estimator in the range of the loss function, i.e., [0, 1].

Interestingly, estimators of these stochastic gradients have an asymmetric need for data. As seen in Equation 7, the min-player's gradient  $g_{-}(p,q)$  includes the loss of every hypothesis  $h\in \mathcal{H}$  for the same data distribution  $q$ . Therefore, an unbiased estimator  $\widehat{g}_{-}(p,q)$  can be constructed from a single call to an example oracle  $\mathrm{EX}(q)$ . We call this source of randomness  $\xi^q$  and say that its cost is  $r_{-} = 1$ . While  $\xi^q$  costs 1 unit, the randomness it provides is specialized to the point of inquiry, that is, it cannot be used for estimating other  $\widehat{g}_{-}(p,q')$ . We call this source of randomness and its associated unbiased estimation a locally unbiased estimator.

On the other hand, the max-player's gradient  $g_{+}(p,q)$  includes the loss of the same hypothesis  $p$  on every distribution  $D \in \mathcal{D}$ . Therefore, an unbiased estimator  $\widehat{g}_{+}(p,q)$  requires  $n$  samples, i.e., a call to every example oracle  $\mathrm{EX}(D_i)$ . We call this source of randomness producing  $n$  samples  $\xi^p$  and say that its cost is  $r_+ = n$ . Importantly, while  $\xi^p$  costs  $n$  units, the randomness it provides can be reused for estimating other gradients, that is, it can provide an unbiased and bounded estimators for all  $\widehat{g}_{+}(p',q')$ . We call this source of randomness and its associated unbiased estimator a globally unbiased estimator. To emphasize the fact that this source of randomness is agnostic to  $(p,q)$  we refer to it by  $\xi^{\perp}$  hereafter. We refer the reader to Appendix A.2 for a more formal definition and description of these asymmetries.

Minimizing Regret with Asymmetric Cost. With the goal of minimizing sample complexity in mind, it is essential that we reuse randomness  $\xi^{\perp}$  across  $n$  time steps of variational algorithms. To do this, we introduce a stochastic variational approach in Algorithm 1 that accommodates different sampling frequencies for the minimizing and maximizing players. This will decouple the sample complexity of the minimizing agent (who requires a time horizon of at least  $\log(A_{-}) \approx \log(\mathcal{H})$ ) and the maximizing agent. This decoupling will lead to additive  $n + \log(\mathcal{H})$  sample complexity instead of the multiplicative  $n \log(\mathcal{H})$ .

Algorithm 1 uses the same randomness  $\xi^{\perp (a)}$  of cost  $r$  for estimating  $g_{+}(p^{t},q^{t})$  for all  $t\in [ar + 1,\ldots ,a(r + 1)]$ . On the other hand, the algorithm uses fresh randomness  $\xi^{(t)}$  of cost 1 to estimate  $g_{-}(p^{t},q^{t})$  for every time step  $t$ . We note that the total randomness cost of this algorithm is  $2t$  because iteration of the outer loop incurs  $2r$  cost.

Lemma 3.1. Let  $(A_{-}, A_{+}, \phi)$  be a finite zero-sum game. Assume there exists  $\xi^{q^{(t)}}$  of cost 1 providing locally unbiased estimates  $\widehat{g}_{-}(\cdot)$  and there exists  $\xi^{\perp(a)}$  of cost  $r$  providing globally unbiased estimates  $\widehat{g}_{+}(\cdot)$ . With probability  $1 - \delta$ , Algorithm 1 returns an  $\varepsilon$ -min-max equilibrium of the game, so long as

$$
T \geq \frac {4}{\varepsilon^ {2}} \left(\max  \left\{\frac {9 \log | A _ {-} |}{4}, 8 \log \left(\frac {r + 1}{\delta}\right) \right\} + \max  \left\{\frac {9 \log | A _ {+} |}{4}, \frac {8 r ^ {2}}{r + 1} \log \left(\frac {r + 1}{\delta}\right) \right\}\right). \tag {9}
$$

Moreover, the total cost of randomness incurred by the algorithm is at most  $2t$ .

Algorithm 1 Finding Equilibria in Finite Zero-Sum Games with Asymmetric Costs.  
Output: Mixed strategy profile  $(p,q)\in \Delta A_{-}\times \Delta A_{+}$    
Input: Action sets  $A_{-},A_{+}$  , cost  $r\in \mathbb{Z}_+$  , timesteps  $T$  , iterates  $p^{(1)},q^{(1)}$  , gradient estimators  $\widehat{g}_{-},\widehat{g}_{+}$  .   
for  $a = 1,2,\ldots ,[T / r]$  do Realize  $\xi^{\perp (a)}$  at cost  $r$  // Sample datapoints from every distribution. for  $t = ar + 1 - r,\dots ,ar$  do Realize  $\xi^{q^{(t)}}$  at cost 1; // Sample from adversary-selected distribution. Estimate gradients:  $\widehat{g}_{+}^{(t)} = \widehat{g}_{+}\left(\xi^{\perp^{(a)}},p^{(t)},q^{(t)}\right),\quad \widehat{g}_{-}^{(t)} = \widehat{g}_{-}\left(\xi^{q^{(t)}},p^{(t)},q^{(t)}\right);$  Run Hedge updates:  $p^{(t + 1)} = \mathcal{Q}_{\mathrm{hedge}}\left(p^{(t)},\widehat{g}_{+}^{(t)}\right),q^{(t + 1)} = \mathcal{Q}_{\mathrm{hedge}}\left(q^{(t)},\widehat{g}_{+}^{(t)}\right);$  end for   
end for   
Return the uniformly mixed strategies  $\overline{p} = \frac{1}{T}\sum_{t = 1}^{T}p^{(t)}$  and  $\overline{q} = \frac{1}{T}\sum_{t = 1}^{T}q^{(t)}$

Proof sketch. Our approach uses Equation 8 to decompose the variational error into training error and generalization error. Since exponential gradient descent is known to bound the training error (as shown in Lemma B.5), it only remains to bound the generalization error (the second term in Equation 3). We note that in expectation each summand  $\langle \varepsilon^{(t)}, w^{(t)} - w^* \rangle$  is zero. This is because  $\varepsilon^{(t)} = g^{(t)} - \widehat{g}^{(t)}$  and  $\widehat{g}^{(t)}$  are unbiased estimators. Therefore, the sum of these terms has an intuitive martingale interpretation and could be bounded by the Azuma-Hoeffding inequality.

There is a subtlety here, however. When we reuse the maximizing player's randomness over  $r$  rounds, we create correlations between these terms in the generalization error that cannot be directly accommodated by a martingale. The trick here is to note that these correlations are entirely contained in  $r$ -length periods. So, we can partition our sequence to  $r$  martingales and bound each one. This completes the proof. See Appendix B.1 for detailed proof of this lemma.

Derandomization. The  $\varepsilon$ -min-max equilibria  $(\overline{p}, \overline{q})$  returned by Exponentiated Gradient Descent gives a probability distribution  $\overline{p}$  over the hypothesis class that achieves the collaborative learning bound. To obtain a deterministic hypothesis, we can instead work with  $h_p^{Maj}$  whose predictions are  $p$ -weighted majority votes over the hypotheses. As stated below, the error of this deterministic classifier is approximately bounded by the expected error of  $\overline{p}$ .

Lemma 3.2. For any  $p \in \Delta \mathcal{H}$ ,  $\max_{D \in \mathcal{D}} \mathrm{loss}_D(h_p^{Maj}) \leq 2 \max_{D \in \mathcal{D}} \mathrm{loss}_D(p)$ .

This lemma in particular implies that for any  $\varepsilon$ -min-max equilibria  $(\overline{p},\overline{q})$ , we have

$$
\max  _ {D \in \mathcal {D}} \operatorname {l o s s} _ {D} \left(h _ {\overline {{p}}} ^ {M a j}\right) \leq 2 \mathrm {R - O P T} + 4 \varepsilon \leq 2 \mathrm {O P T} + 4 \varepsilon .
$$

# 4 Collaborative Learning Bounds

In this section, we characterize the sample complexity of collaborative learning by providing tight upper and lower bounds for this problem.

# 4.1 Sample Complexity Upper Bounds

We are now prepared to describe our collaborative learning algorithm and guarantees, using the tools we developed in Section 3. Algorithm 2 is a direct application of Algorithm 1 to a zero-sum game with action sets  $A_{-} = \mathcal{H}$ ,  $A_{+} = \mathcal{D}$  and payoff  $\phi(h, D) = \mathrm{loss}_D(h)$ . Here,  $\xi^{q^{(t)}}$  makes one call to  $\mathrm{EX}(q^{(t)})$  and  $\xi^{\perp(a)}$  makes one call to  $\mathrm{EX}(D)$  for each  $D \in \mathcal{D}$ . In other words, Algorithm 2 constructs distributions  $p^{(t)} \in \Delta \mathcal{H}$  and  $q^{(t)} \in \Delta \mathcal{D}$  by running the Hedge update. The gradient estimators used by Hedge are the empirical losses on a set of independent random variables. In particular, the minimizing player uses gradients  $\ell(h, z^{(t)})$  for all  $h \in \mathcal{H}$  for a single sample  $z^{(t)} \sim \mathrm{EX}(q^{(t)})$  and the maximizing player uses gradients  $\ell(p^{(t)}, z_D^a)$  for all distributions  $D \in \mathcal{D}$  where a single sample  $z_D^a \sim \mathrm{EX}(D)$  is drawn per distribution and is reused for all time steps  $t \in [(a - 1)n + 1, \dots, an]$ .

Our main result in this section bounds the sample complexity of Algorithm 2.

Algorithm 2 On-Demand Agnostic Collaborative Learning.  
Input: Hypothesis class  $\mathcal{H}$ , distribution set  $\mathcal{D}$  with  $n := |\mathcal{D}|$ ;  
Initialize:  $p^{(1)} = [1 / |\mathcal{H}|]^{\lvert\mathcal{H}\rvert},q^{(1)} = [1 / n]^{n}$ , and iterations  $T = \frac{16}{\varepsilon^2}$  ( $2\log (|\mathcal{H}|) + 9n\log (n / \delta)$ );  
for  $a = 1,2,\ldots ,[T / n]$  do  
    For all  $D\in \mathcal{D}$ , sample datapoint  $z_D^a$  from  $\mathrm{EX}(D)$ .  
    for  $t = an + 1 - n,\dots ,an$  do  
        Sample  $z^{(t)}$  from  $\mathrm{EX}(q^{(t)})$  and estimate  $\widehat{g}_{-}^{(t)} = [\ell (h,z^{(t)})]_{h\in \mathcal{H}},\widehat{g}_{+}^{(t)} = [\ell (p^{(t)},z_{D}^{a})]_{D\in \mathcal{D}}$ ;  
        Run Hedge updates:  $p^{(t + 1)} = \mathcal{Q}_{\mathrm{hedge}}\left(p^{(t)},\widehat{g}_{+}^{(t)}\right),q^{(t + 1)} = \mathcal{Q}_{\mathrm{hedge}}\left(q^{(t)},\widehat{g}_{+}^{(t)}\right)$ ;  
    end for  
end for  
Return: probability distribution over  $\mathcal{H}$  given by the uniform mixture  $\frac{1}{T}\sum_{t = 1}^{T}p^{(t)}$ .

Theorem 4.1. For any finite hypothesis class  $\mathcal{H}$  and unknown set of distributions  $\mathcal{D}$ , with probability  $1 - \delta$ , Algorithm 2 returns a distribution  $\overline{p} \in \Delta \mathcal{H}$  such that

$$
\underset {h \sim \bar {p}} {\mathbb {E}} \left[ \max  _ {D \in \mathcal {D}} (h) \right] \leq O P T + \varepsilon \text {a n d} \max  _ {D \in \mathcal {D}} \left(h _ {\bar {p}} ^ {M a j}\right) \leq 2 O P T + \varepsilon ,
$$

using a number of samples that is  $\mathcal{O}\left(\frac{\log|\mathcal{H}| + n\log(n / \delta)}{\varepsilon^2}\right)$ .

Proof sketch. By construction, Lemma 3.1 guarantees that with probability at least  $1 - \delta$ , the pair  $(\overline{p}, \overline{q})$  is an  $\varepsilon/2$ -min-max equilibrium for the corresponding zero-sum game. As shown by Equation 5,  $\overline{p}$  is a randomized classifier that meets the collaborative learning objective, i.e., its expected worst-case error is  $\mathrm{OPT} + \varepsilon$ . By Lemma 3.2, the corresponding deterministic classifier  $h_{\overline{p}}^{Maj}$  has worst-case error of  $2\mathrm{OPT} + \varepsilon$ . This bounds the error of the resulting classifier.

To bound the sample complexity, Lemma 3.1 shows that the randomness cost of Algorithm 1 is at most  $2t$ . Since the cost of randomness is exactly the total number of samples we take from our example oracles, the total sample complexity of Algorithm 2 is  $2t \in \mathcal{O}\left(\frac{\log|\mathcal{H}| + n\log(n / \delta)}{\varepsilon^2}\right)$ .

A similar result holds for the case of infinite hypothesis classes of bounded VC dimension. In this case, one can instead run Algorithm 2 with a hypothesis class  $\mathcal{H}'$  that is an  $\varepsilon$ -net with respect to every distribution in  $\mathcal{D}$ . We note that such  $\varepsilon$ -nets of size  $n\varepsilon^{-2\mathrm{VCD}}(\mathcal{H})$  necessarily exist (see, e.g., [1]); for example, the union of  $\varepsilon$ -nets with respect to each distribution  $D \in \mathcal{D}$ . When such  $\mathcal{H}'$  is known in advance, we may run Algorithm 2 with  $\mathcal{H}'$  and incur a sample complexity that now replaces  $\log(|\mathcal{H}'|) = O(d\log(1/\varepsilon))$  in the sample complexity of Theorem 4.1.

We remark that it is not strictly necessary to know an  $\varepsilon$ -net in advance. Instead, one can compute a net from samples or from other information about distributions in  $\mathcal{D}$ . In Appendix D, we explore a range of assumptions that allow us to compute such an  $\varepsilon$ -net from samples, without incurring a significant increase in the sample complexity of Theorem 4.1. As an example, here we mention two such assumptions. Assumption 1: we know the marginal distribution for all  $D \in \mathcal{D}$ , or a weaker Assumption 2: we have access to  $n$  marginal distributions  $P_{1}, \ldots, P_{n}$  such that for all  $x \in \mathcal{X}$ ,  $d_{i}(A) \leq p_{i}(A)\mathrm{poly}(1 / \varepsilon, \mathrm{VCD}(\mathcal{H}), n)$  for all  $A \subseteq \mathcal{X}$ , where  $p_{i}$  and  $d_{i}$  are the densities of  $P_{i}$  and  $D_{i}$ , respectively. These assumptions allow one to construct  $\varepsilon$ -nets of small size, e.g., by projecting  $\mathcal{H}$  on a sufficiently large set of random feature vectors generated from distributions  $P_{i}$ . We refer the reader to Appendix D for more detail on how these assumptions can be used to construct  $\varepsilon$ -nets.

Theorem 4.2. For any  $\mathcal{H}$  of VC dimension  $d$  and unknown set of distributions  $\mathcal{D}$  for which Assumption 1 or 2 is met, there is an algorithm that, with probability  $1 - \delta$ , returns a distribution  $\overline{p} \in \Delta \mathcal{H}$  with

$$
\underset {h \sim \bar {p}} {\mathbb {E}} \left[ \max  _ {D \in \mathcal {D}} (h) \right] \leq O P T + \varepsilon \text {a n d} \max  _ {D \in \mathcal {D}} \left(h _ {\bar {p}} ^ {M a j}\right) \leq 2 O P T + \varepsilon ,
$$

using a number of samples that is  $\mathcal{O}\left(\frac{d\log(dn / \varepsilon) + n\log(n / \delta)}{\varepsilon^2}\right)$ .

We end this subsection with two remarks about our sample complexity upper bound.

Remark 4.1. Theorem 4.1 improves over the best-known sample complexity for agnostic collaborative learning by Nguyen and Zakynthinou [17] in two ways. First, it provides  $OPT + \varepsilon$  for randomized classifiers whereas Nguyen and Zakynthinou [17] gave a  $2OPT + \varepsilon$  bound. Second, it improves their sample complexity of  $\mathcal{O}\left(\frac{1}{\varepsilon^5} (\log (n)\log (|\mathcal{H}|))\log \left(\frac{1}{\varepsilon}\right) + n\log \left(\frac{n}{\delta}\right)\right)$  by a multiplicative factor of  $\frac{1}{\varepsilon^3}\log (n)\log \left(\frac{1}{\varepsilon}\right)$ .  
Remark 4.2. For constants  $\varepsilon$  and  $\delta$ , our sample complexity of  $\mathcal{O}\left(\log (|\mathcal{H}|) + n\log n\right)$  appears to violate the lower bound of  $\Omega$  ( $\log (|\mathcal{H}|)\log n + n\log \log |\mathcal{H}|$ ) due to Chen, Zhang, and Zhou [6]. This discrepancy is due to a small error in the proof of that lower bound, which we have verified in private communications with the authors. In the next subsection, we give lower bounds on the sample complexity of collaborative learning that match our upper bounds.

# 4.2 Sample Complexity Lower Bound

We now provide matching lower bounds for agnostic collaborative learning. Our lower bounds hold for collaborative learning algorithms obtaining error of R-OPT  $+\varepsilon$ , using a randomized or deterministic hypothesis. We call an algorithm an  $(\varepsilon, \delta)$ -collaborative learning algorithm if for any collaborative instances it attains an error of R-OPT  $+\varepsilon$  with probability at least  $1 - \delta$ .

Theorem 4.3. Take any  $n, d \in Z_{+}, \varepsilon, \delta \in (0,1/8)$ , and  $(\varepsilon, \delta)$ -collaborative learning algorithm  $A$ . There exists a collaborative learning problem  $(\mathcal{H},\mathcal{D})$  with  $|\mathcal{D}| = n$  and  $|h| = 2^d$ , on which  $A$  takes at least  $\Omega\left(\frac{1}{\varepsilon^2}\left(\log |\mathcal{H}| + |\mathcal{D}|\log (\min \{| \mathcal{D}|,\log |\mathcal{H}|\} /\delta)\right)\right)$  samples in expectation.

Proof sketch. We defer the formal proof of this theorem to Appendix B.2 and sketch the main ideas here. We use  $\mathcal{X} = \{1, \ldots, d\}$ ,  $\mathcal{Y} = \{+, -\}$ , and let  $\mathcal{H}$  be the set of all functions  $\mathcal{X} \to \mathcal{Y}$ . Our construction combines two types of hard distributions. We describe the ideas for the case of  $n = d$ . First, we use a hard construction for agnostic learning of hypothesis classes with VC dimension  $d$  as the distribution of one of the agents. This gives us the  $\Omega\left(\log(|\mathcal{H}|)/\varepsilon^{2}\right)$  part of the lower bound. Second, we construct  $n$  hard instances each of VC dimension 1 on  $n$  independent points. Since the learning algorithms has to solve each problem it has to incur a loss of  $n\log(n/\delta)/\varepsilon^{2}$ .

# 5 Group DRO and Agnostic Federated Learning

The results we describe in the collaborative learning setting can be generalized to the group DRO setting, and equivalently, agnostic federated learning.

Theorem 5.1. Consider any group distributionally robust problem  $(\Theta, \mathcal{D})$  with a convex compact parameter space  $\Theta$  of Bregman radius  $D_{\Theta} = \min_{x \in \Theta} \max_{y \in \Theta} D(y||x)$ , and convex loss function  $\ell : \Theta \times \mathcal{Z} \to [0, C]$ . A variant of Algorithm 2 (in particular Algorithm 5 in Appendix B.3), returns  $\overline{\theta} \in \Theta$  such that  $\max_{D \in \mathcal{D}} \mathbb{E}_{z \sim D}[\ell(\theta, z)] \leq R-OPT + \varepsilon$ , using a number of samples that is  $\mathcal{O}\left(\frac{D_{\Theta}C^2 + nC^2\log(n/\delta)}{\varepsilon^2}\right)$ .

The proof of this lemma is deferred to Appendix B.3 and is similar to the proof of Theorem 4.1 except that it uses a generalization of Lemma 3.1 for general convex-concave games. This theorem establishes a generalization bound for the problem of group distributionally robust optimization [22] and improves, by a factor of  $n$ , existing sample complexity bounds for agnostic federated learning [15]. This improvement is attained by sampling data on-demand, whereas [15] only chooses a fixed distribution over groups/clients to sample from; this highlights the importance of adapting one's sampling strategy on-the-fly when learning robust models.

Another important question is how fast the training error of stochastic gradient descent converges for the group DRO/AFL settings and was considered by Sagawa et al. [22]. We can transfer our generalization guarantees for on-demand settings into batch settings and achieve the following corollary, which improves on the convergence guarantees of Sagawa et al. [22] by a factor of  $n$ .

Corollary 5.2. Under the same assumptions of Theorem 5.1, we give a procedure (see Appendix B.3) that minimizes GDRO/AFL training error within  $\varepsilon$  of  $R$ -OPT with probability at least  $1 - \delta$  in fewer samples than  $\mathcal{O}\left(\frac{D\Theta C^2 + nC^2\log(n / \delta)}{\varepsilon^2}\right)$ .

# References

[1] M. Anthony and P. L. Bartlett. Neural Network Learning: Theoretical Foundations. Cambridge University Press, Cambridge, 1999. ISBN 978-0-521-57353-5. doi: 10.1017/CBO9780511624216. URL https://www.cambridge.org/core/books/neural-network-learning/665C8C7EB5E2ABC5367A55ADB04E2866.  
[2] A. Beck and M. Teboulle. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters, 31(3):167-175, May 2003. ISSN 0167-6377. doi: 10.1016/S0167-6377(02)00231-6. URL https://www.sciencedirect.com/science/article/pii/S0167637702002316.  
[3] S. Ben-David and R. Schuller. Exploiting task relatedness for multiple task learning. In Learning theory and kernel machines, pages 567-580. Springer, 2003.  
[4] A. Blum, N. Haghtalab, A. D. Procaccia, and M. Qiao. Collaborative PAC Learning. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://papers.nips.cc/paper/2017/bitstream/186a157b2992e7daed3677ce8e9fe40f-Abstract.html.  
[5] A. Blum, N. Haghtalab, R. L. Phillips, and H. Shao. One for One, or All for All: Equilibria and Optimality of Collaboration in Federated Learning. arXiv:2103.03228 [cs], Mar. 2021. URL http://arxiv.org/abs/2103.03228. arXiv: 2103.03228.  
[6] J. Chen, Q. Zhang, and Y. Zhou. Tight Bounds for Collaborative PAC Learning via Multiplicative Weights. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper/2018/bitstreamed519dacc89b2bead3f453b0b05a4a8b-Abstract.html.  
[7] C. Daskalakis, A. Deckelbaum, and A. Kim. Near-Optimal No-Regret Algorithms for Zero-Sum Games. In Proceedings of the 2011 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), Proceedings, pages 235-254. Society for Industrial and Applied Mathematics, Jan. 2011. ISBN 978-0-89871-993-2. doi: 10.1137/1.9781611973082.21. URL https://epubs.siam.org/doi/abs/10.1137/1.9781611973082.21.  
[8] C. Daskalakis, M. Fishelson, and N. Golowich. Near-Optimal No-Regret Learning in General Games. arXiv:2108.06924 [cs], Aug. 2021. URL http://arxiv.org/abs/2108.06924.arXiv:2108.06924.  
[9] J. Duchi and H. Namkoong. Learning Models with Uniform Performance via Distributionally Robust Optimization. arXiv:1810.08750 [cs, stat], July 2020. URL http://arxiv.org/abs/1810.08750. arXiv:1810.08750.  
[10] Y. Freund and R. E. Schapire. A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. Journal of Computer and System Sciences, 55(1):119-139, Aug. 1997. ISSN 0022-0000. doi: 10.1006/jcss.1997.1504. URL https://www.sciencedirect.com/science/article/pii/S002200009791504X.  
[11] S. Hart and A. Mas-Colell. A Simple Adaptive Procedure Leading to Correlated Equilibrium. Econometrica, 68(5):1127-1150, 2000. ISSN 1468-0262. doi: 10.1111/1468-0262.00153. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00153. _eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/1468-0262.00153.  
[12] T. Hashimoto, M. Srivastava, H. Namkoong, and P. Liang. Fairness without demographics in repeated loss minimization. In International Conference on Machine Learning, pages 1929-1938. PMLR, 2018.  
[13] A. Kar, A. Prakash, M.-Y. Liu, E. Cameracci, J. Yuan, M. Rusiniak, D. Acuna, A. Torralba, and S. Fidler. Meta-Sim: Learning to Generate Synthetic Datasets. Technical Report arXiv:1904.11621, arXiv, Apr. 2019. URL http://arxiv.org/abs/1904.11621.arXiv:1904.11621 [cs] type: article.  
[14] Y. Mansour, M. Mohri, and A. Rostamizadeh. Domain adaptation with multiple sources. Advances in neural information processing systems, 21, 2008.

[15] M. Mohri, G. Sivek, and A. T. Suresh. Agnostic Federated Learning. Technical Report arXiv:1902.00146, arXiv, Jan. 2019. URL http://arxiv.org/abs/1902.00146.arXiv:1902.00146 [cs, stat] type: article.  
[16] A. S. Nemirovskij and D. B. Yudin. Problem complexity and method efficiency in optimization. Wiley-Interscience, 1983. Publisher: Wiley-Interscience.  
[17] H. L. Nguyen and L. Zakynthinou. Improved Algorithms for Collaborative PAC Learning. arXiv:1805.08356 [cs, stat], Oct. 2018. URL http://arxiv.org/abs/1805.08356. arXiv: 1805.08356.  
[18] S. Rakhlin and K. Sridharan. Optimization, learning, and games with predictable sequences. Advances in Neural Information Processing Systems, 26, 2013.  
[19] V. V. Ramaswamy, S. S. Y. Kim, and O. Russakovsky. Fair Attribute Classification through Latent Space De-biasing. Technical Report arXiv:2012.01469, arXiv, Apr. 2021. URL http://arxiv.org/abs/2012.01469.arXiv:2012.01469 [cs] type: article.  
[20] H. Robbins and S. Monro. A stochastic approximation method. The annals of mathematical statistics, pages 400-407, 1951. Publisher: JSTOR.  
[21] J. Robinson. An Iterative Method of Solving a Game. Annals of Mathematics, 54(2):296-301, 1951. ISSN 0003-486X. doi: 10.2307/1969530. URL https://www.jstor.org/stable/1969530. Publisher: Annals of Mathematics.  
[22] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang. Distributionally robust neural networks. In International Conference on Learning Representations, 2019.  
[23] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang. Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. arXiv:1911.08731 [cs, stat], Apr. 2020. URL http://arxiv.org/abs/1911.08731. arXiv: 1911.08731.  
[24] N. K. Vishnoi. Algorithms for Convex Optimization. Cambridge University Press, 2021. doi: 10.1017/9781108699211.  
[25] S. Zakharov, W. Kehl, and S. Ilic. DeceptionNet: Network-Driven Domain Randomization. Technical Report arXiv:1904.02750, arXiv, Aug. 2019. URL http://arxiv.org/abs/1904.02750. arXiv:1904.02750 [cs] type: article.
