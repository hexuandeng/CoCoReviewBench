# AUGMENTED LAGRANGIAN IS ENOUGH FOR OPTIMAL OFFLINE RL WITH GENERAL FUNCTION APPROXIMATION AND PARTIAL COVERAGE

Anonymous authors

Paper under double-blind review

# ABSTRACT

Offline reinforcement learning (RL), which refers to decision-making from a previously-collected dataset of interactions, has received significant attention over the past years. Much effort has focused on improving offline RL practicality by addressing the prevalent issue of partial data coverage through various forms of conservative policy learning. While the majority of algorithms do not have finite-sample guarantees, several provable conservative offline RL algorithms are designed and analyzed within the single-policy concentrability framework that handles partial coverage. Yet, in the nonlinear function approximation setting where confidence intervals are difficult to obtain, existing provable algorithms suffer from computational intractability, prohibitively strong assumptions, and suboptimal statistical rates. In this paper, we leverage the marginalized importance sampling (MIS) formulation of RL and present the first set of offline RL algorithms that are statistically optimal and practical under general function approximation and single-policy concentrability, bypassing the need for uncertainty quantification. We identify that the key for successfully solving the sample-based approximation of the MIS problem is ensuring that certain state occupancy validity constraints are nearly satisfied. We enforce these constraints by a novel application of the augmented Lagrangian method and prove the following result: with MIS formulation, augmented Lagrangian is enough for statistically optimal offline RL. In stark contrast to prior algorithms that induce additional conservatism through methods such as behavior regularization, our approach provably eliminates this need and reinterprets regularizers as "enforcers of state occupancy validity" than "promoters of conservatism."

# 1 INTRODUCTION

The goal of offline RL is to design agents that learn to achieve competence in a task using only a previously-collected dataset of interactions (Lange et al., 2012). Offline RL is a promising tool for many critical applications, from healthcare to autonomous driving to scientific discovery, where the online mode of learning by interacting with the environment is dangerous, impractical, costly, or even impossible (Levine et al., 2020). Despite this, offline RL has not yet been truly successful in practice (Fujimoto et al., 2019; Levine et al., 2020) and impressive RL performance has been limited to settings with known environments (Silver et al., 2017; Moravčík et al., 2017), access to accurate simulators (Mnih et al., 2015; Degrave et al., 2022), or expert demonstrations (Vinyals et al., 2017).

One of the central challenges in offline RL is the lack of uniform coverage in real datasets and the distribution shift between the occupancy of candidate policies and offline data distribution, which pose difficulties in accurately evaluating the candidate policies. Over the past years, a body of literature has focused on addressing this challenge through developing conservative algorithms, which aim at picking a policy among those well-covered in the data. On the practical front, various forms of conservatism are proposed such as behavior regularization through policy constraints (Kumar et al., 2019; Fujimoto et al., 2019; Nachum & Dai, 2020), learning conservative values (Kumar et al., 2020; Liu et al., 2020; Agarwal et al., 2020), or learning pessimistic models (Kidambi et al., 2020; Yu et al., 2020; 2021); see Appendix A for further discussion on related work.

From a theoretical standpoint, partial data coverage has recently been studied within variants of the single-policy concentrability framework (Rashidinejad et al., 2021; Xie et al., 2021; Uehara & Sun, 2021), which characterizes the distribution shift between offline data and occupancy of a target (often optimal) policy, in contrast to all-policy concentrability commonly used in earlier works (Scherrer, 2014; Chen & Jiang, 2019; Liao et al., 2020; Zhang et al., 2020a; Xie & Jiang, 2021). Within this framework and in the tabular and linear function approximation settings, pessimistic algorithms that leverage uncertainty quantifiers to construct lower confidence bounds (Jin et al., 2021; Rashidinejad et al., 2021; Yin et al., 2021; Shi et al., 2022; Li et al., 2022) enjoy optimal statistical rate. In the general function approximation setting, pessimistic algorithms largely assume oracle access to uncertainty quantification, either for constructing penalties that are subtracted from rewards (Jin et al., 2021; Jiang & Huang, 2020) or selecting the most pessimistic option among those that fall within the confidence region implied by the offline data (Uehara & Sun, 2021; Xie et al., 2021; Chen & Jiang, 2022). However, uncertainty quantifiers are difficult to obtain in non-linear function approximation and existing heuristics are empirically observed to be unreliable (Rashid et al., 2019; Tennenholtz et al., 2021). Recent works by Cheng et al. (2022) and Zhan et al. (2022) propose provable alternatives to uncertainty-based methods, but leave achieving optimal statistical rate of  $1 / \sqrt{N}$ , where  $N$  is the dataset size, as an open problem.

Among all, the marginal importance sampling (MIS) methods, which aim at learning weights  $w$  that estimate the distribution shift between induced policy occupancy  $d_w$  and data distribution  $\mu$ , lend themselves well to the single-policy concentrability framework. Though more popular in off-policy evaluation (Liu et al., 2018; Xie et al., 2019; Uehara et al., 2020; Zhang et al., 2020b), MIS has also been used for conservative offline RL such as AlgaeDICE (Nachum et al., 2019b) and OptiDICE (Lee et al., 2021), both of which incorporate behavior regularization. Recently, Zhan et al. (2022) theoretically studied a variant of OptiDICE, showing that MIS with behavior regularization enjoys finite-sample guarantees (though achieving a suboptimal  $1 / N^{1/6}$  rate) and circumvents certain fundamental difficulties observed in value-based offline RL with function approximation (Du et al., 2019; Wang et al., 2020; 2021; Weisz et al., 2021; Zanette, 2021; Foster et al., 2021).

# 1.1 CONTRIBUTIONS AND RESULTS

Motivated by the benefits offered by MIS, we study designing statistically optimal offline learning algorithms under this formulation in the general function approximation and single-policy concentrability setting. We conduct theoretical investigations and design algorithms starting from multi-armed bandits (MABs), going forward to contextual bandits (CBs), and finally Markov decision processes (MDPs). In the rest of this section, we present a preview of our contributions and results.

Multi-armed bandits. Empirical MIS algorithms often incorporate behavior regularization, whose role is justified as promoting conservatism by keeping the occupancies of learned and behavior policies close (Nachum et al., 2019b; Lee et al., 2021). Yet, whether and why these regularizers are necessary from a theoretical perspective remain unclear. Zhan et al. (2022) motivates behavior regularization as a way of introducing curvature in an otherwise linear optimization problem. We extensively investigate the effect of regularization, starting from the simplest setting of MABs with function approximation, as existing algorithms when specialized to offline MABs, are either intractable, have suboptimal finite-sample guarantees, or require access to uncertainty quantifiers.

We state our results on unregularized MIS and MIS with behavior regularization (PRO-MAB Algorithm 1), which is a special case of PRO-RL (Zhan et al., 2022), in the informal theorem below.

Theorem (informal) (I) Unregularized MIS fails to achieve a decaying suboptimality in certain offline MAB instances. (II) MIS with behavior regularization (PRO-MAB Algorithm 1) achieves suboptimality  $\widetilde{O}(1/\sqrt{N})$  in offline MABs. (III) If one searches only over the space of weights that induce valid occupancies ( $d_w = 1$ ), then unregularized MIS achieves  $O(1/\sqrt{N})$  suboptimality.

In this theorem, we prove that unregularized MIS fails even in bandits and provide a tight analysis of PRO-MAB improving over the original  $1 / N^{1 / 6}$  rate shown by Zhan et al. (2022). In our analysis of PRO-MAB, we find that the key to the success of the MIS algorithm is near-validity of the learned occupancy  $d_w$ . In the MAB setting, the validity constraint simply requires the learned occupancy to be a probability distribution:  $d_w = \sum_a w(a)\mu (a) = 1$ . With a proper choice of hyperparameter, we show that behavior regularization enforces learned occupancy to be nearly valid:  $d_w = \Omega (1 / \log N)$ . We further prove that regularization is not required if validity is otherwise satisfied.

Given that occupancy validity is the constraint of the optimization problem solved by MIS (see e.g. (1)), we ask whether there are any methods for solving empirical optimization problems that find solutions that adhere more to the constraints, compared to the Lagrange multiplier adopted in prior works (Lee et al., 2021; Zhan et al., 2022). The augmented Lagrangian method (ALM), which adds a quadratic loss on the constraints  $(d_w - 1)^2$ , is a natural choice for our purpose. The ALM term can be easily estimated from offline data and forms Algorithm 1. We show that ALM results in  $d_w = \Omega(1)$ , ensuring near-validity of estimated occupancy and leading to the following guarantee.

Theorem (informal) Th policy returned by conservative offline MAB with augmented Lagrangian (Algorithm 1) achieves  $O(1 / \sqrt{N})$  suboptimality.

Our algorithm offers several benefits over PRO-MAB such as improving the rate by  $\log N$  and only requiring single-policy concentrability instead of the two-policy requirement of PRO-MAB, which can be strong (see Section 5.3). Additionally, behavior regularization introduces bias in the solution even with infinite data (Chen & Jiang, 2022) and the bias-variance tradeoff must be carefully handled. However, ALM merely enforces the optimization constraints and leads to provably unbiased solutions (Lemma 13). More importantly, as we discuss shortly, going beyond the single-state MAB setting, behavior regularization is suboptimal while ALM maintains the optimal rate.

Contextual bandits. In offline CBs, we analyze two approaches: MIS with behavior regularization, and an extension of ALM. We state our results in the following informal theorem.

Theorem (informal) (I) MIS with behavior regularization (PRO-CB Algorithm 6) is statistically suboptimal for certain CB instances. (II) Policy returned by conservative offline CB with augmented Lagrangian (Algorithm 2) achieves suboptimality of  $O(1 / \sqrt{N})$ .

Intuitively, the failure of PRO-CB to achieve the optimal rate is because the regularization parameter has to be small to control bias, but such small regularization is not strong enough to ensure the validity of learned occupancy in most states. Therefore, one must choose larger regularization, leading to an overall suboptimal rate. Prior works Chen & Jiang (2022); Cheng et al. (2022) also allude to this phenomenon, explaining that regularizers appear to be the culprit behind suboptimal rates. In CB, the occupancy validity constraints require conditional occupancy to be a valid probability distribution in every state. In Algorithm 2, we incorporate ALM in offline CBs by adding a weighted sum of quadratic losses describing the validity constraint in each state, where the weights are set to the state occupancies to capture their relative importance. Enforcement of the constraints by ALM yields the guarantee stated above.

MDPs. Validity constraints in MDPs ensure that the learned state occupancy  $d_w(s) = \sum_{a} w(s, a) \mu(s, a)$  is close to the actual state occupancy  $d^{\pi_w}(s)$ , where  $\pi_w$  is the policy computed from weights  $w$ . Directly enforcing this constraint results in an ALM term that cannot easily be estimated from offline data. We address this difficulty by expressing the ALM term in the variational form. From there, we derive two variants, one model-based and one model-free, of our conservative offline RL with augmented Lagrangian (CORAL) algorithm, that enjoys the following guarantee.

Theorem (informal) Model-based and model-free CORAL both achieve  $O(1 / \sqrt{N})$  suboptimality.

This marks CORAL as the first practical and statistically optimal offline RL algorithm that operates in the general function approximation and partial data coverage setting, while avoiding uncertainty quantification and additional regularizers. Conservatism of CORAL is baked into the MIS formulation and supported by the ALM: bounded MIS weights prevent learned occupancy to deviate significantly from data distribution, and ALM ensures closeness of the learned and actual occupancies. When combined, CORAL learns a policy whose actual occupancy is close to the data distribution.

We thus proved that ALM improves sample complexity compared to alternatives such as behavior regularization. This is in addition to the benefits on optimization stability that are likely to be offered by the ALM, as it improves over the ill-posed Lagrange multiplier objective (Ben-Tal & Nemirovski, 2022). Our theoretical findings can explain the empirical observations of Yang et al. (2020), who find MIS with behavior regularization unstable, propose regularizers in "the spirit of ALM" that gain superior performance, and attribute performance gain to improved optimization. In this work, we present a theoretically-founded way of introducing ALM in offline RL and our analysis shows that ALM also leads to improved sample complexity.

# 2 BACKGROUND

Markov decision process. An infinite-horizon discounted MDP is described by a tuple  $M = (S, \mathcal{A}, P, R, \rho, \gamma)$ , where  $\mathcal{S}$  is the state space,  $\mathcal{A}$  is the action space,  $P: S \times \mathcal{A} \mapsto \Delta(S)$  is the transition kernel,  $R: S \times \mathcal{A} \mapsto \Delta([0,1])$  encodes a family of reward distributions with  $r: S \times \mathcal{A} \mapsto [0,1]$  as the expected reward function,  $\rho: S \mapsto \Delta(S)$  is the initial state distribution, and  $\gamma \in [0,1)$  is the discount factor. We assume  $\mathcal{S}$  and  $\mathcal{A}$  are finite however, our results do not depend on their cardinalities and can be naturally extended to infinite sets. A stationary (stochastic) policy  $\pi: S \mapsto \Delta(\mathcal{A})$  specifies a distribution over actions in each state. Each policy  $\pi$  induces an occupancy density over state-action pairs  $d^{\pi}: S \times \mathcal{A} \mapsto [0,1]$  defined as  $d^{\pi}(s,a) := (1 - \gamma)\sum_{t=0}^{\infty}\gamma^{t}P_{t}(s_{t} = s,a_{t} = a;\pi)$ , where  $P_{t}(s_{t} = s,a_{t} = a;\pi)$  denotes  $(s,a)$  visitation probability at step  $t$ , starting at  $s_{0} \sim \rho(\cdot)$  and following  $\pi$ . We abuse notation and also write  $d^{\pi}(s) = \sum_{a\in\mathcal{A}}d^{\pi}(s,a)$  to denote the discounted state occupancy. Additionally, operator  $\mathbb{P}^{\pi}$  is applied to any function  $u: S \times \mathcal{A} \to \mathbb{R}$  and is defined as  $(\mathbb{P}^{\pi}u)(s,a) := \sum_{s',a'}P(s'|s,a)\pi(a'|s')u(s',a')$ .

An important quantity is the value a policy  $\pi$ , which is the discounted sum of rewards  $V^{\pi}(s) \coloneqq \mathbb{E}[\sum_{t=0}^{\infty} \gamma^{t} r_{t} \mid s_{0} = s, a_{t} \sim \pi(\cdot \mid s_{t}), \forall t \geq 0]$  starting at state  $s \in S$ . We use the notation  $J(\pi) \coloneqq (1 - \gamma) \mathbb{E}_{s \sim \rho}[V^{\pi}(s)] = \mathbb{E}_{s, a \sim d^{\pi}}[r(s, a)]$  to represent a scalar summary of the performance of a policy  $\pi$ . We denote by  $\pi^{\star}$  the optimal policy that maximizes the above objective and use the shorthand  $V^{\star} \coloneqq V^{\pi^{\star}}$  to denote the optimal value function.

Offline reinforcement learning. We focus on the offline RL, where the agent is only provided with a previously-collected offline dataset  $\mathcal{D} = \{(s_i,a_i,r_i,s_i')\}_{i=1}^N$ . Here,  $r_i \sim R(s_i,a_i)$ ,  $s_i' \sim P(\cdot \mid s_i,a_i)$ , and we assume  $s_i,a_i$  pairs are generated i.i.d. according to a data distribution  $\mu \in \Delta(\mathcal{S} \times \mathcal{A})$ . To streamline the analysis, we assume that the conditional distribution  $\mu(a|s)$  is known. The goal of offline RL is to learn a policy  $\hat{\pi}$  based on the offline dataset so as to minimize the sub-optimality with respect to the optimal policy  $\pi^\star$ , i.e.  $J(\pi^\star) - J(\hat{\pi})$  with high probability. In this paper, we consider marginal importance sampling (MIS) formulation that aims at learning weights  $w(s,a)$  to represent policy occupancy when multiplied by data distribution:  $d_w(s,a) = w(s,a)\mu(s,a)$ . Also denote  $d_w(s) = \sum_{a \in \mathcal{A}} d_w(s,a)$ . We define the policy induced by  $w$  as  $\pi_w(a|s) = d_w(s,a) / d_w(s)$  for  $d_w(s) > 0$  and  $\pi_w(a|s) = 1 / |\mathcal{A}|$  for  $d_w(s) = 0$ .

Offline data coverage assumption. We design and analyze our algorithms within the single-policy concentrability framework (Rashidinejad et al., 2021), stated below.

Definition 1 (Single-policy concentrability) Given a policy  $\pi$ , define  $C^{\pi}$  to be the smallest constant that satisfies  $\frac{d^{\pi}(s,a)}{\mu(s,a)} \leq C^{\pi}$  for all  $s \in S$  and  $a \in \mathcal{A}$ .

$C^{\pi^{\star}} = C^{\star}$  captures coverage of  $\pi^{\star}$  in the offline data and is much weaker than the widely used all-policy concentrability that assumes bounded  $\max_{\pi} C^{\pi}$ ; see Appendix A for further discussion.

Notation. Given a set  $S$ , we write  $|\mathcal{S}|$  to represent its cardinality and  $\Delta(S)$  to denote the probability simplex over  $S$ . For a function class  $\mathcal{W}$ , we write  $|\mathcal{W}|$  to denote its cardinality (discrete) or covering number (continuous). We use the notation  $x \lesssim y$  when there exists constant  $c > 0$  such that  $x \leq cy$  and  $x \asymp y$  if constants  $c_1, c_2 > 0$  exist such that  $c_1|x| \leq |y| \leq c_2|x|$ . We write  $f(x) = O(g(x))$  if  $M > 0, x_0$  exist such that  $|f(x)| \leq Mg(x)$  for all  $x \geq x_0$  and use  $\widetilde{O}(\cdot)$  to be the big- $O$  notation ignoring logarithmic factors. Define  $\operatorname{clip}(x, a, b) \triangleq \max\{a, \min\{x, b\}\}$  for  $x, a, b \in \mathbb{R}$ .

# 3 MULTI-ARMED BANDITS

We start by considering the offline learning problem in the multi-armed bandit (MAB) setting, which is a special case of MDP with  $\gamma = 0$ ,  $|\mathcal{S}| = 1$ , and  $\mathcal{D} = \{(a_i,r_i)\}_{i=1}^N$ , where  $a_i \sim \mu(\cdot), r_i \sim R(a_i)$ . The goal of offline learning in MABs can be described as the following constrained optimization problem, where  $d$  represents occupancy

$$
\max  _ {d \geq 0} \mathbb {E} _ {a \sim d} [ r (a) ] \quad \text {s . t .} \quad \sum_ {a} d (a) = 1. \tag {1}
$$

# 3.1 PRIMAL-DUAL REGULARIZED OFFLINE BANDITS

To solve (1), the MIS approach with behavior regularization defines importance weights  $w(a) = d(a) / \mu(a)$  and converts the problem (1) to its dual form by introducing the Lagrange multiplier  $v$ :

$$
\max  _ {w \geq 0} \min  _ {v} L _ {\alpha} ^ {\mathrm {M A B}} (w, v) := \mathbb {E} _ {a \sim \mu} [ w (a) r (a) ] - v (\mathbb {E} _ {a \sim \mu} [ w (a) ] - 1) - \alpha \mathbb {E} _ {a \sim \mu} [ f (w (a)) ]. \tag {2}
$$

The last term in (2) is the behavior regularizer that characterizes the  $f$ -divergence between the learned occupancy  $d$  and data distribution  $\mu$ , with  $\alpha$  determining the strength of regularization. This term was originally proposed to induce conservatism by keeping the learned policy close to behavior policy (Nachum et al., 2019b; Lee et al., 2021). Approximating  $w$  via a function class  $\mathcal{W} \subseteq \mathbb{R}^{|\mathcal{A}|}$  and solving the empirical version of (2) yields Algorithm 5, which we call primal-dual regularized offline MAB (PRO-MAB) as it is a special case of PRO-RL algorithm of Zhan et al. (2022).

One might wonder whether the unregularized algorithm  $(\alpha = 0)$  is sufficient for solving the offline learning problem in MABs, particularly under the natural and common assumption that elements of the function class  $\mathcal{W}$  are bounded:  $w(a) = d(a) / \mu (a)\leq B_w$ . In the following proposition, we show that the answer is negative and there exist MAB instances in which the unregularized algorithm finds a policy that suffers from a constant suboptimality. The proof is provided in Appendix B.2.

Proposition 1 (Unregularized algorithm fails in MAB) Let  $\hat{\pi}$  be the policy returned by Algorithm 5 with  $\alpha = 0$ . There exists a 2-armed bandit problem in which policy  $\hat{\pi}$  satisfies  $J(\pi^{\star}) - J(\hat{\pi}) = 1/6$  with a constant probability.

We note that Zhan et al. (2022) also argues the failure of the unregularized algorithm by giving a counterexample in the MDP setting. We discuss this example in detail in Section 5.3. Proposition 1 reveals additional insights: the objective (15) with  $\alpha = 0$  fails not just in MDPs but also in bandits, even when the optimal policy is unique and data are collected by running a behavior policy.

Given the failure of the unregularized algorithm, we conduct a tight analysis of PRO-MAB with  $\alpha > 0$ . In the next theorem, we prove that under similar assumptions as Zhan et al. (2022) and with a proper choice of  $\alpha$ , PRO-MAB returns a policy that enjoys near-optimal sample complexity.

Theorem 1 (Suboptimality of PRO-MAB) Let  $f: \mathbb{R} \mapsto \mathbb{R}$  be  $M_{f}$ -strongly convex, non-negative  $f(x) \geq 0$ , and bounded  $|f(x)| \leq B_{f}$ . Assume  $0 \leq w(a) \leq B_{w}$  for any  $w \in \mathcal{W}$ . Set  $\alpha \asymp M_{f}(B_{w}(B_{v} + 1) + B_{f})\log(N)\sqrt{\log(|\mathcal{V}||\mathcal{W}| / \delta) / N}$ . Assume realizability of  $w_{\alpha}^{\star} \in \mathcal{W}$  and  $v_{\alpha}^{\star} \in \mathcal{V}$  the optimal solutions to (2). Let  $\pi_{\alpha}^{\star} = \pi_{w_{\alpha}^{\star}}$  and suppose concentrability of  $\pi^{\star}$  and  $\pi_{\alpha}^{\star}$  (Definition 1). For any fixed  $\delta \geq 0$ , policy  $\hat{\pi}$  returned by Algorithm 5 achieves

$$
J (\pi^ {\star}) - J (\hat {\pi}) \lesssim \alpha (B _ {f} + f ^ {\prime} (C ^ {\star}) B _ {w}) = \widetilde {O} \left(\frac {1}{\sqrt {N}}\right).
$$

To our knowledge, this is the first statistically optimal guarantee for a practical offline MAB algorithm with function approximation and partial coverage and improves over the  $1 / N^{1 / 6}$  guarantee given by Zhan et al. (2022). We now briefly explain the differences between the analysis methods; a complete proof is deferred to Appendix B.3. Zhan et al. (2022) bounds policy suboptimality by  $\alpha +1 / (\alpha^{1 / 2}N^{1 / 4})$ , where the first term comes from the bias caused by the regularizer and the second term stems from bounding the difference between  $\hat{w}$  and  $w_{\alpha}^{\star}$  relying on the strong convexity of  $L_{\alpha}$ . Optimizing the bound over  $\alpha$  gives the final  $1 / N^{1 / 6}$  guarantee. In contrast, our analysis connects suboptimality to occupancy validity. In particular, we show that suboptimality is bounded by  $\alpha +1 / (d_{\hat{w}}\sqrt{N})$ , where  $d_{\hat{w}} = \sum_{a}\hat{w} (a)\mu (a)$ . We then show that setting  $\alpha = \tilde{O} (1 / \sqrt{N})$  is sufficient to ensure near-validity of occupancy  $d_{\hat{w}} = \Omega (1 / \log (N))$ , which proves the claim.

We observe a similar phenomenon in Proposition 1 that small  $d_w$  for certain  $w \in \mathcal{W}$  can cause the unregularized algorithm to fail. In the following section, we investigate this phenomenon further, leading to a new offline learning algorithm.

# 3.2 AUGMENTED LAGRANGIAN REPLACES BEHAVIOR REGULARIZATION

The next proposition further cements the importance of policy validity and shows that if the occupancy is valid, such as by searching only over the weights that induce valid occupancies, then the unregularized algorithm enjoys an optimal rate. Proof of this result can be found in Appendix B.4.

# Algorithm 1 Conservative Offline MAB with Augmented Lagrangian

1: Inputs: Dataset  $\mathcal{D} = \{(a_i,r_i)\}_{i = 1}^N$ , classes  $\mathcal{W}$  and  $\mathcal{V} = [-B_v,B_v]$ .  
2: Find a solution  $\hat{w},\hat{v}$  to the following problem

$$
\max  _ {w \in \mathcal {W}} \min  _ {v \in \mathcal {V}} \hat {L} _ {A L} ^ {\mathrm {M A B}} (w, v) := \frac {1}{N} \sum_ {i = 1} ^ {N} w \left(a _ {i}\right) r _ {i} - v \left(w \left(a _ {i}\right) - 1\right) - \left(\frac {1}{N} \sum_ {i = 1} ^ {N} w \left(a _ {i}\right) - 1\right) ^ {2}. \tag {3}
$$

3: Return:  $\hat{\pi} = \pi_{\hat{w}}$

Proposition 2 (Constraint satisfaction is sufficient in MAB) Assume as in Theorem 1. Let  $\hat{\pi}$  be the output of Algorithm 5 with  $\alpha = 0$  and assume that  $\sum_{a}\mu(a)\hat{w}(a) = 1$ . Then, for any fixed  $\delta > 0$

$$
J (\pi^ {\star}) - J (\hat {\pi}) \lesssim (B _ {w} (B _ {v} + 1) + \alpha B _ {f}) \sqrt {\frac {\log | \mathcal {V} | | \mathcal {W} | / \delta}{N}}.
$$

Motivated by the discussion above, we take a step back and ask: are there any other methods for solving constrained optimization problems that find more constraint-satisfying solutions when applied to the empirical approximation of the original problem? A promising candidate is the augmented Lagrangian method (ALM) which adds a quadratic loss on the constraints to the objective. Applied to (1), ALM forms the following objective, whose empirical version leads to Algorithm 1.

$$
\max  _ {w \geq 0} \min  _ {v} L _ {\mathrm {A L}} ^ {\mathrm {M A B}} (w, v) := \mathbb {E} _ {a \sim \mu} [ w (a) r (a) ] - v (\mathbb {E} _ {a \sim \mu} [ w (a) ] - 1) - (\mathbb {E} _ {a \sim \mu} [ w (a) ] - 1) ^ {2}. \tag {4}
$$

The following theorem establishes an upper bound on the suboptimality of the policy returned by Algorithm 1, whose proof can be found in Appendix B.5.

Theorem 2 (Suboptimality of Algorithm 1) Suppose  $\pi^{\star}$ -concentrability (Definition 1). Assume  $w^{\star} \in \mathcal{W}$ , where  $w^{\star}(a) = d^{\pi^{\star}}(a) / \mu(a)$ , and  $v^{\star} \in \mathcal{V}$ , where  $v^{\star} = J(\pi^{\star})$ . Moreover, assume that  $0 \leq w(a) \leq B_w$  for  $w \in \mathcal{W}$ . For any fixed  $\delta \geq 0$ , policy  $\hat{\pi}$  returned by Algorithm 1 achieves

$$
J (\pi^ {\star}) - J (\hat {\pi}) \lesssim (B _ {w} + 1) ^ {2} (B _ {v} + 1) \sqrt {\frac {\log (| \mathcal {W} | | \mathcal {V} | / \delta)}{N}}.
$$

In the proof, we show that ALM results in near-validity of  $\hat{w}$  by ensuring that  $d_{\hat{w}} = \Omega(1)$ , leading to the optimal rate. Note that Algorithm 1 does not include any explicit form of conservatism through regularizers or uncertainty quantifiers. Colloquially, the MIS formulation and boundedness of  $\mathcal{W}$  elements ensure that  $d_{\hat{w}}(a) / \mu(a) = \hat{w}(a) \leq B_w$  and ALM ensures that  $d_{\hat{w}}$  is close to the actual occupancy. Thus, Algorithm 1 seeks a policy whose actual occupancy is within data distribution. Algorithm 1 offers several benefits compared to PRO-MAB: it only requires  $\pi^{\star}$ -concentrability as instead of  $\pi^{\star}, \pi_{\alpha}^{\star}$ -concentrability in PRO-MAB, improves the rate by  $\log N$ , removes the need to design regularizer  $f$  and adjust  $\alpha$ , and does not introduce any bias in the objective. The main advantage of ALM, however, becomes more evident as we move beyond MAB, where the behavior regularization provably fails to achieve optimal statistical rate while ALM maintains optimality.

# 4 CONTEXTUAL BANDITS

The problem offline contextual bandits (CB) is a special case of offline RL with  $\gamma = 0$  and offline dataset  $\mathcal{D} = \{(s_i, a_i, r_i)\}_{i=1}^N$ , where  $s_i \sim \mu(\cdot) = \rho(\cdot)$ ,  $a_i \sim \mu(\cdot \mid a_i)$ , and  $r_i \sim R(s_i, a_i)$ . The linear programming constrained optimization problem for CB is given by

$$
\max  _ {d \geq 0} \mathbb {E} _ {s, a \sim d} [ r (s, a) ] \quad \text {s . t .} \quad \sum_ {a} d (s, a) = \rho (s) \quad \forall s \in \mathcal {S}. \tag {5}
$$

# 4.1 PRIMAL-DUAL REGULARIZED OFFLINE CONTEXTUAL BANDITS

In the following proposition, we prove a performance lower bound on the primal-dual regularized offline CB (PRO-CB) presented in Algorithm 6, which includes behavior regularization.

Proposition 3 (PRO-CB is suboptimal) Let  $f: \mathbb{R} \mapsto \mathbb{R}$  be  $M_f$ -strongly convex, non-negative, and bounded. Assume  $0 \leq w(s, a) \leq B_w$  for  $w \in \mathcal{W}$  and realizability of the optimal solutions to (23):  $w_\alpha^\star \in \mathcal{W}$  and  $v_\alpha^\star \in \mathcal{V}$ . Let  $\pi_\alpha^\star = \pi_{w_\alpha^\star}$  and assume  $\pi_0^\star, \pi_\alpha^\star$ -concentrability. Let  $\hat{\pi}$  be the output of Algorithm 6. Then, for any  $\alpha \geq 0$  there exists a CB instance such that with probability  $\Omega(1)$ ,  $J(\pi^\star) - J(\hat{\pi}) = \Omega(N^\beta)$ , where  $\beta > -1/2$ .

# Algorithm 2 Conservative Offline CB with Augmented Lagrangian

1: Inputs: Dataset  $\mathcal{D} = \{(s_i, a_i, r_i)\}_{i=1}^N$ , function classes  $\mathcal{W}, \mathcal{V}$  
2: Find a solution  $\hat{w},\hat{v}$  to the following problem

$$
\max  _ {w \in \mathcal {W}} \min  _ {v \in \mathcal {V}} \hat {L} _ {\mathrm {A L}} ^ {\mathrm {C B}} (w, v) := \frac {1}{N} \sum_ {i = 1} ^ {N} w \left(s _ {i}, a _ {i}\right) \left(r _ {i} - v \left(s _ {i}\right)\right) + v \left(s _ {i}\right) - \left(\sum_ {a \in \mathcal {A}} w \left(s _ {i}, a\right) \mu \left(a \mid s _ {i}\right) - 1\right) ^ {2} \tag {7}
$$

3: Return:  $\hat{\pi} = \pi_{\hat{w}}$

Proposition 3 shows that behavior regularization is statistically suboptimal regardless of  $\alpha$ . The proof is presented in Appendix C.2. The main takeaway is that ensuring occupancy validity  $\sum_{a}\hat{w}(s,a)\mu(a|s) = \Omega(1)$  for nearly all states appears to be critical in achieving the optimal rate. Yet, without introducing a large bias, behavior regularization is insufficient for such a guarantee.

# 4.2 OFFLINE CONTEXTUAL BANDITS WITH AUGMENTED LAGRANGIAN

To encourage occupancy validity, we extend ALM to CBs and propose the following objective:

$$
\begin{array}{l} \max  _ {w \geq 0} \min  _ {v} L _ {\mathrm {A L}} ^ {\mathrm {C B}} (w, v) \tag {6} \\ := \mathbb {E} _ {\mu} [ w (s, a) r (s, a) ] - \mathbb {E} _ {\mu} [ v (s) (w (s, a) - 1) ] - \mathbb {E} _ {s \sim \mu} [ (\mathbb {E} _ {a \sim \mu (\cdot | s)} [ w (s, a) ] - 1) ^ {2} ] \\ \end{array}
$$

Notice that when  $|S| = 1$ , (6) simplifies to the ALM objective (2) in the MAB setting. The ALM term can be understood in the following way. Each element encourages the validity of occupancy in each state  $\sum_{a} w(s, a) \mu(s, a) \approx 1$  and the elements are weighted according to the true state distribution: validity is more important in states that are actually more likely to be visited. We analyze the suboptimality of Algorithm 2 and present the following theorem, showing that the ALM achieves optimal rate without any need for behavior regularization. The proof can be found in Appendix C.3.

Theorem 3 (Suboptimality of Algorithm 2) Suppose concentrability of an optimal policy  $\pi^{\star}$  (Definition 1). Assume realizability of  $w^{\star} \in \mathcal{W}$  where  $w^{\star}(s, a) = d^{\pi^{\star}}(s, a) / \mu(s, a)$  and  $v^{\star} \in \mathcal{V}$ , where  $v^{\star} = J(\pi^{\star})$ . Moreover, assume that  $|v(s)| \leq B_v$  for  $v \in \mathcal{V}$  and  $0 \leq w(s, a) \leq B_w$ . For any fixed  $\delta \geq 0$ , policy  $\hat{\pi}$  returned by Algorithm 2 achieves

$$
J (\pi^ {\star}) - J (\hat {\pi}) \lesssim (B _ {w} + 1) ^ {2} (B _ {v} + 1) \sqrt {\frac {\log (| \mathcal {W} | | \mathcal {V} | / \delta)}{N}}.
$$

# 5 MARKOV DECISION PROCESSES

We now turn to offline RL. In addition to the offline dataset, we assume access to a dataset  $\mathcal{D}_0 = \{s_i\}_{i=1}^N$  with i.i.d. samples from the initial distribution  $\rho$ , similar to prior works (Lee et al., 2021; Zhan et al., 2022). The linear programming formulation of RL (Puterman, 2014) solves

$$
\max  _ {d \geq 0} \mathbb {E} _ {s, a \sim d} [ r (s, a) ] \quad \text {s . t .} \quad d (s) = (1 - \gamma) \rho (s) + \gamma \sum_ {s ^ {\prime}, a ^ {\prime}} P (s | s ^ {\prime}, a ^ {\prime}) d \left(s ^ {\prime}, a ^ {\prime}\right) \quad \forall s \in \mathcal {S}. \tag {8}
$$

The constraints are known as Bellman flow equations and restrict the search to the space of valid occupancy distributions  $d^{\pi}$  that can be induced in the MDP by running a policy  $\pi$ .

# 5.1 CONSERVATIVE OFFLINE RL WITH AUGMENTED LAGRANGIAN

Motivated by the success of ALM in bandits, we propose the following extension to offline RL:

$$
\max  _ {w \geq 0} \min  _ {v} L _ {\mathrm {A L}} ^ {\mathrm {M D P}} (w, v) := (1 - \gamma) \mathbb {E} _ {\rho} [ v (s) ] + \mathbb {E} _ {\mu} [ w (s, a) e _ {v} (s, a) ] - \mathbb {E} _ {d ^ {\pi_ {w}}} \left(\frac {d _ {w} (s)}{d ^ {\pi_ {w}} (s)} - 1\right) ^ {2}, \tag {9}
$$

where  $e_v(s, a) \coloneqq r(s, a) + \gamma \sum_{s'} P(s'|s, a)v(s') - v(s)$ . One can check that the first two terms are the Lagrange dual of (8) and the last term is a generalization of the ALM terms in bandits. The ALM elements encourage occupancy  $d_w(s)$  to be close in ratio to the actual occupancy  $d^{\pi_w}(s)$  in each state and as before, the ALM elements are weighted according to their actual visitation  $d^{\pi_w}(s)$ . Our particular ALM construction can be intuitively understood as follows. The MIS formulation learns bounded weights  $\hat{w}(s, a) = d_{\hat{w}}(s, a) / \mu(s, a) \leq B_w$ . The ALM term ensures that the ratio  $d_{\hat{w}}(s) / d^{\pi_{\hat{w}}}(s) = \Omega(1)$  which roughly translates to  $d^{\pi_{\hat{w}}}(s, a) / \mu(s, a) \lesssim B_w$ .

The ALM term in (9) is difficult to estimate as it involves the expectation over unknown occupancy  $d^{\pi_w}$  and the computation of the ratio  $d_w(s) / d^{\pi_w}(s)$ . We resolve this difficulty in the next sections.

# Algorithm 3 Conservative Offline RL with Augmented Lagrangian (CORAL) — Model-based

1: Inputs: Datasets  $\mathcal{D},\mathcal{D}_0,\mathcal{D}_m$  , function classes  $\mathcal{W},\mathcal{V},\mathcal{U},\mathcal{P}$ $f_{*}^{-1}(x) = 2\sqrt{x + 1} -2.$  
2: Estimate transitions via maximum likelihood:  $\hat{P} = \operatorname*{argmax}_{P\in \mathcal{P}}\sum_{i = 1}^{N_m}\ln P(s_i'|s_i,a_i)$  
3: Find a solution  $\hat{w},\hat{v},\hat{u}$  to the following problem

$$
\begin{array}{l} \max  _ {w \in \mathcal {W}} \min  _ {v \in \mathcal {V}} \min  _ {u \in \mathcal {U}} \hat {L} _ {A L} ^ {\text {m o d e l - b a s e d}} (w, v) := \frac {(1 - \gamma)}{N _ {0}} \sum_ {i = 1} ^ {N _ {0}} \left(v \left(s _ {i}\right) + \sum_ {a} u \left(s _ {i}, a\right) \pi_ {w} (a \mid s _ {i})\right) \tag {13} \\ + \frac {1}{N} \sum_ {i = 1} ^ {N} w \left(s _ {i}, a _ {i}\right) \left[ r _ {i} + \gamma v \left(s _ {i} ^ {\prime}\right) - v \left(s _ {i}\right) - f _ {*} ^ {- 1} \left(u \left(s _ {i}, a _ {i}\right) - \gamma \left(\hat {\mathbb {P}} ^ {\pi_ {w}} u\right) \left(s _ {i}, a _ {i}\right)\right) \right] \\ \end{array}
$$

4: Return:  $\hat{\pi} = \pi_{\hat{w}}$

# 5.2 ESTIMATING THE ALM TERM AND CORAL ALGORITHM

We view the ALM term as the negative  $f$ -divergence between  $d_w$  and  $d^{\pi_w}$  with  $f(x) := (x - 1)^2$  and express it in the variational form (Nguyen et al., 2010):

$$
- \mathbb {E} _ {d ^ {\pi_ {w}}} \left(\frac {d _ {w} (s)}{d ^ {\pi_ {w}} (s)} - 1\right) ^ {2} = - D _ {f} \left(d _ {w} \| d ^ {\pi_ {w}}\right) = \min  _ {x} \mathbb {E} _ {d ^ {\pi_ {w}}} \left[ f _ {*} (x (s, a)) \right] - \mathbb {E} _ {d _ {w}} [ x (s, a) ]. \tag {10}
$$

Here,  $f_{*}$  is the convex conjugate of  $f$  and we used the fact that  $d_w(s,a) / d^{\pi_w}(s,a) = d_w(s) / d^{\pi_w}(s)$ . Notice that  $\mathbb{E}_{d\pi_w}[f_*(x(s,a))]$  is the value of  $\pi_w$  in the same MDP but with rewards  $f_{*}(x(s,a))$ . Define  $u$  as the fixed point of the following Bellman equation

$$
u (s, a) := f _ {*} (x (s, a)) + \gamma \left(\mathbb {P} ^ {\pi_ {w}} u\right) (s, a). \tag {11}
$$

Since  $u(s, a)$  is the Q-function of  $\pi_w$  with rewards  $f_*(x(s, a))$ , we can rewrite (10) as

$$
(1 0) = \min  _ {u} (1 - \gamma) \mathbb {E} _ {s \sim \rho , a \sim \pi_ {w}} [ u (s, a) ] - \mathbb {E} _ {\mu} \left[ w (s, a) f _ {*} ^ {- 1} (u (s, a) - \gamma (\mathbb {P} ^ {\pi_ {w}} u) (s, a)) \right]. \tag {12}
$$

(12) involves expectations over  $\rho$  and  $\mu$ , which can be estimated empirically. Below, we discuss model-free and model-based methods for estimating the term involving the transition operator  $\mathbb{P}^{\pi_w}$ . We include some details on practical implementations in Appendix D.1.

Model-based CORAL. For the model-based route, we assume access to a realizable function class  $\mathcal{P}$  that contains the true transitions and an additional dataset  $\mathcal{D}_m = \{(s_i,a_i,s_i')\}_{i=1}^{N_m}$ , where  $s_i,a_i \sim \mu$  and  $s_i' \sim P(\cdot \mid s_i,a_i)$ . Given  $\mathcal{D}_m$ , we obtain a maximum likelihood estimate of transitions and then approximate the expectations using  $\mathcal{D}_0$  and  $\mathcal{D}$ . This leads to Algorithm 3, which we name model-based conservative offline RL with augmented Lagrangian (CORAL).

Model-free CORAL. As an alternative, we consider developing a model-free that uses a single-sample estimate of  $f_{*}^{-1}(u(s,a) - \gamma (P^{\pi_w}u)(s,a))$ . This, however, roughly leads to the infamous double sampling problem (Baird, 1995). To circumvent this difficulty, in Appendix D.4 we use the dual embedding trick in Nachum et al. (2019a), to derive model-free CORAL (Algorithm 4).

Theorem 4 shows that both variants of CORAL enjoy optimal rates; see Appendix D.5 for the proof.

Theorem 4 (CORAL Suboptimality) Suppose concentrability of an optimal policy  $\pi^{\star}$  and assume  $w^{\star}(s,a) = d^{\pi^{\star}}(s,a) / \mu (s,a)\in \mathcal{W}$  and  $v^{\star}(s) = V^{\star}(s)\in \mathcal{V}$ . Moreover, assume  $0\leq w(s,a)\leq B_w$  for  $w\in \mathcal{W}$  and  $|v(s)|\leq B_v$  for  $v\in \mathcal{V}$ . Let  $\tilde{x}_w(s,a) = \mathrm{clip}(x_w^\star (s,a), - 1,1)$ , where  $x_{w}^{\star}$  is a solution to (10), and define  $u_{w}^{\star}$  as the fixed-point solution to (11) when  $x = \tilde{x}_w$ . Assume  $u_{w}^{\star}\in \mathcal{U}$  for any  $w\in \mathcal{W}$  and  $\| u\|_{\infty}\leq B_u$  where  $B_{u}\geq 5 / (4(1 - \gamma))$ . For any fixed  $\delta \geq 0$ , the following claims hold.

(I) Assume  $N = N_0 = N_m$  for simplicity. If  $P^{\star} \in \mathcal{P}$ , then  $\hat{\pi}$  returned by Algorithm 3 achieves

$$
J (\pi^ {\star}) - J (\hat {\pi}) \lesssim \frac {B _ {v} + B _ {u} + (1 + B _ {v}) B _ {w}}{1 - \gamma} \sqrt {\frac {B _ {u} \log (| \mathcal {P} | | \mathcal {U} | | \mathcal {W} | | \mathcal {V} | / \delta)}{N}}.
$$

(II) Assume  $N = N_{0}$  for simplicity. Let  $\zeta_{w,u}^{\star} = \operatorname{argmax}_{\zeta < 0} L_{AL}^{model - free}(w,v,u,\zeta)$  defined in (43). Assume  $\zeta_{w^{\star},u}^{\star} \in \mathcal{Z}$  for  $u \in \mathcal{U}$  and  $B_{\zeta,L} \leq |\zeta(s,a)| \leq B_{\zeta,U}$  for  $\zeta \in \mathcal{Z}$ , where  $B_{\zeta,L} \in (0, \sqrt{2}/3)$  and  $B_{\zeta,U} \geq \sqrt{2}$ . Let  $B_{\zeta} = \max\{B_{\zeta,U}, B_{\zeta,L}^{-1}\}$ . Then,  $\hat{\pi}$  returned by Algorithm 4 achieves

$$
J (\pi^ {\star}) - J (\hat {\pi}) \lesssim \frac {B _ {v} + B _ {u} + (1 + B _ {v} + B _ {\zeta} (B _ {u} + 1)) B _ {w}}{1 - \gamma} \sqrt {\frac {\log (| \mathcal {U} | | \mathcal {W} | | \mathcal {V} | | \mathcal {Z} | / \delta)}{N}}.
$$

# Algorithm 4 Conservative Offline RL with Augmented Lagrangian (CORAL) — Model-free

1: Inputs: Datasets  $\mathcal{D}$ ,  $\mathcal{D}_0$ , function classes  $\mathcal{W}, \mathcal{V}, \mathcal{U}, \mathcal{Z}$ ,  $g_*(x) = x + 2 + \frac{1}{x}$ .  
2: Find a solution  $\hat{w},\hat{v},\hat{u},\hat{\zeta}$  to  $\max_{w\in \mathcal{W}}\min_{v\in \mathcal{V}}\min_{u\in \mathcal{U}}\max_{\zeta \in Z}\hat{L}_{\mathrm{AL}}^{\mathrm{model - free}}(w,v,u,\zeta)$  defined as

$$
\frac {(1 - \gamma)}{N _ {0}} \sum_ {i = 1} ^ {N _ {0}} v \left(s _ {i}\right) + \sum_ {a} u \left(s _ {i}, a\right) \pi_ {w} \left(a \mid s _ {i}\right) + \frac {1}{N} \sum_ {i = 1} ^ {N} w \left(s _ {i}, a _ {i}\right) \left[ r _ {i} + \gamma v \left(s _ {i} ^ {\prime}\right) - v \left(s _ {i}\right) \right. \tag {14}
$$

$$
\left. + \zeta \left(s _ {i}, a _ {i}\right) \left(u \left(s _ {i}, a _ {i}\right) - \gamma \sum_ {i} u \left(s _ {i} ^ {\prime}, a ^ {\prime}\right) \pi_ {w} \left(a ^ {\prime} \mid s _ {i} ^ {\prime}\right)\right) + g _ {*} \left(\zeta \left(s _ {i}, a _ {i}\right)\right) \right]
$$

3: Return:  $\hat{\pi} = \pi_{\hat{w}}$

In Theorem 4, we make realizability assumptions on  $u_{w}^{\star}$  for  $w \in \mathcal{W}$  and  $\zeta_{w^{\star},u}^{\star}$  for  $u \in \mathcal{U}$ . Such assumptions are common in theory of RL with function approximation (Munos & Szepesvári, 2008; Xie et al., 2021; Jiang & Huang, 2020) and removing them can be difficult or impossible (Foster et al., 2021). Recently, Zhan et al. (2022); Chen & Jiang (2022) propose algorithms that only require optimal solution realizability, however, these algorithms are either intractable or suboptimal.

# 5.3 EXAMPLE: BEHAVIOR REGULARIZATION VS. AUGMENTED LAGRANGIAN

We examine the MDP example in Figure 1 presented by Zhan et al. (2022). Assume  $\mathcal{V} = \{v^{\star}\}$  and  $\mathcal{W} = \{w_1, w_2\}$ , where  $w_1$  always selects  $L$  from  $A$  and  $w_2$  always selects  $R$  from  $A$ . One can check  $w_1(A, L) = 2$ ,  $w_1(A, R) = 0$  and  $w_2(A, L) = 0$ ,  $w_2(A, R) = 1$ .

Unregularized algorithm. As Zhan et al. (2022) state, the unregularized algorithm fails to distinguish between  $w_{1}$  and  $w_{2}$  even with infinite data as the objectives at  $w_{1}$  and  $w_{2}$  are exactly equal.

Behavior regularization. Consider an instantiation of PRO-RL with regularizer  $-\alpha \mathbb{E}_{\mu}[w^{2}(s,a)]$ . Since in this example  $\mathbb{E}_{\mu}[w_1^2 (s,a)] > \mathbb{E}_{\mu}[w_2^2 (s,a)]$ , PRO-RL picks the wrong weight  $w_{2}$ , suffering constant suboptimality. Note, however, that PRO-RL guarantees assume  $\pi_{\alpha}^{\star}$ -concentrability. Intuitively, behavior regularization causes  $\pi_{\alpha}^{\star}$  to be more stochastic and thus requiring  $\mu (s,a) > 0$  for more states and actions. Here, since  $\mu$  covers both  $(A,L)$  and  $(A,R)$ , behavior regularization causes  $\pi_{\alpha}^{\star}(R|A) > 0$  and thus  $d^{\pi_{\alpha}^{\star}}(C) > 0$ . To handle the MDP in Figure 1, PRO-RL additionally requires  $\mu (C) > 0$  to satisfy  $\pi_{\alpha}^{\star}$ -concentrability.

ALM. In this example, ALM successfully picks the optimal  $w_{1}$ , as  $1/4, \mu(A, R) = 0$  it avoids a mismatch between the actual and learned occupancies.  $1/4, \mu(C) = 0$ . This is because in (9) the ALM term is zero at  $w_{1}$  due to realizability  $\pi_{w_{1}}$ -conce  $\pi_{w_{2}}$  whereas at  $w_{2}$ , it has a lower bound  $\mathbb{E}_{s \sim d^{\pi_{2}}} (d_{w_{2}}(C) / d^{\pi_{2}}(C) - 1)^{2} \geq d^{\pi_{2}}(C) > 0$ .

![](images/9988a4fd8e3b857ce1ba295a4549cde95e1d832f59f06e9487bfdfb8cc0bd726.jpg)  
Figure 1: The agent starts from  $A$ . Action  $L$  leads to  $B$ , from where the agent collects +1 reward. Action  $R$  leads to  $C$ , from where only one action leads to a +1 reward. Nature decides which MDP is presented to the learner. Data distribution is  $\mu(A, L) = 1/4$ ,  $\mu(A, R) = 1/2$ ,  $\mu(B) = 1/4$ ,  $\mu(C) = 0$ , which satisfies  $\pi_{w_1}$ -concentrability.

# 6 DISCUSSION

We present a set of practical and statistically optimal algorithms for offline MAB, CB, and RL, under general function approximation and single-policy concentrability. Our algorithms are designed within the MIS formulation combined with a novel application of augmented Lagrangian method. Importantly, our optimality guarantees hold under MIS combined with ALM alone, without any additional form of conservatism such as via regularization or uncertainty quantification. Furthermore, we investigate the role of regularizers in MIS algorithms. Although the empirical benefits of such regularizers are often attributed to conservatism, our analysis suggests that conservatism stems from the MIS formulation while the role of regularizers is to ensure the validity of learned occupancy. Interesting future directions include conducting empirical evaluations of ALM, examining the possibility of removing strong realizability assumptions, and investigating practical and optimal offline RL algorithms whose guarantees hold under milder variants of single-policy concentrability more suited to function approximation.

# REFERENCES

Rishabh Agarwal, Dale Schuurmans, and Mohammad Norouzi. An optimistic perspective on offline reinforcement learning. In International Conference on Machine Learning, pp. 104-114. PMLR, 2020.  
Andras Antos, Rémi Munos, and Csaba Szepesvari. Fitted Q-iteration in continuous action-space mdps. In Neural Information Processing Systems, 2007.  
András Antos, Csaba Szepesvári, and Rémi Munos. Learning near-optimal policies with Bellman-residual minimization based fitted policy iteration and a single sample path. Machine Learning, 71(1):89-129, 2008.  
Leemon Baird. *Residual algorithms: Reinforcement learning with function approximation.* In *Machine Learning Proceedings* 1995, pp. 30-37. Elsevier, 1995.  
Aharon Ben-Tal and Arkadi Nemirovski. Lecture notes optimization III: Convex analysis, Non-linear programming theory, Non-linear programming algorithms, 2022.  
Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. arXiv preprint arXiv:1905.00360, 2019.  
Jinglin Chen and Nan Jiang. Offline reinforcement learning under value and density-ratio realizability: The power of gaps. In *The 38th Conference on Uncertainty in Artificial Intelligence*, 2022.  
Ching-An Cheng, Tengyang Xie, Nan Jiang, and Alekh Agarwal. Adversarily trained actor critic for offline reinforcement learning. In Proceedings of the 39th International Conference on Machine Learning, volume 162, 2022.  
Bo Dai, Niao He, Yunpeng Pan, Byron Boots, and Le Song. Learning from conditional distributions via dual embeddings. In Artificial Intelligence and Statistics, pp. 1458-1467. PMLR, 2017.  
Jonas Degrave, Federico Felici, Jonas Buchli, Michael Neunert, Brendan Tracey, Francesco Carpanese, Timo Ewalds, Roland Hafner, Abbas Abdelmaleki, Diego de Las Casas, et al. Magnetic control of tokamak plasmas through deep reinforcement learning. Nature, 602(7897):414-419, 2022.  
Simon S Du, Sham M Kakade, Ruosong Wang, and Lin F Yang. Is a good representation sufficient for sample efficient reinforcement learning? In International Conference on Learning Representations, 2019.  
Amir Massoud Farahmand, Rémi Munos, and Csaba Szepesvári. Error propagation for approximate policy and value iteration. In Advances in Neural Information Processing Systems, 2010.  
Yihao Feng, Lihong Li, and Qiang Liu. A kernel loss for solving the Bellman equation. arXiv preprint arXiv:1905.10506, 2019.  
Dylan J Foster, Akshay Krishnamurthy, David Simchi-Levi, and Yunzong Xu. Offline reinforcement learning: Fundamental barriers for value function approximation. arXiv preprint arXiv:2111.10919, 2021.  
Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning. Advances in neural information processing systems, 34:20132-20145, 2021.  
Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In International Conference on Machine Learning, pp. 2052-2062. PMLR, 2019.  
Seyed Kamyar Seyed Ghasemipour, Dale Schuurmans, and Shixiang Shane Gu. EMaQ: Expectedmax Q-learning operator for simple yet effective offline and online RL. arXiv preprint arXiv:2007.11091, 2020.  
L Jeff Hong, Weiwei Fan, and Jun Luo. Review on ranking and selection: A new perspective. Frontiers of Engineering Management, 8(3):321-343, 2021.

Natasha Jaques, Asma Ghandeharioun, Judy Hanwen Shen, Craig Ferguson, Agata Lapedriza, Noah Jones, Shixiang Gu, and Rosalind Picard. Way off-policy batch deep reinforcement learning of implicit human preferences in dialog. arXiv preprint arXiv:1907.00456, 2019.  
Nan Jiang. On value functions and the agent-environment boundary. arXiv preprint arXiv:1905.13341, 2019.  
Nan Jiang and Jiawei Huang. Minimax value interval for off-policy evaluation and policy optimization. Advances in Neural Information Processing Systems, 33, 2020.  
Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably efficient for offline RL? In International Conference on Machine Learning, pp. 5084-5096. PMLR, 2021.  
Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In ICML, volume 2, pp. 267-274, 2002.  
Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. MOREL: Model-based offline reinforcement learning. arXiv preprint arXiv:2005.05951, 2020.  
Ilya Kostrikov, Rob Fergus, Jonathan Tompson, and Ofir Nachum. Offline reinforcement learning with Fisher divergence critic regularization. In International Conference on Machine Learning, pp. 5774-5783. PMLR, 2021.  
Aviral Kumar, Justin Fu, George Tucker, and Sergey Levine. Stabilizing off-policy Q-learning via bootstrapping error reduction. arXiv preprint arXiv:1906.00949, 2019.  
Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative Q-learning for offline reinforcement learning. arXiv preprint arXiv:2006.04779, 2020.  
Aviral Kumar, Joey Hong, Anikait Singh, and Sergey Levine. Should i run offline reinforcement learning or behavioral cloning? In International Conference on Learning Representations, 2021.  
Sascha Lange, Thomas Gabel, and Martin Riedmiller. Batch reinforcement learning. In Reinforcement learning, pp. 45-73. Springer, 2012.  
Jongmin Lee, Wonseok Jeon, Byungjun Lee, Joelle Pineau, and Kee-Eung Kim. OptiDICE: Offline policy optimization via stationary distribution correction estimation. In International Conference on Machine Learning, pp. 6120-6130. PMLR, 2021.  
Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.  
Gen Li, Laixi Shi, Yuxin Chen, Yuejie Chi, and Yuting Wei. Settling the sample complexity of model-based offline reinforcement learning. arXiv preprint arXiv:2204.05275, 2022.  
Peng Liao, Zhengling Qi, and Susan Murphy. Batch policy learning in average reward Markov decision processes. arXiv preprint arXiv:2007.11771, 2020.  
Boyi Liu, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural trust region/proximal policy optimization attains globally optimal policy. In *Neural Information Processing Systems*, 2019a.  
Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou. Breaking the curse of horizon: Infinite-horizon off-policy estimation. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 5361-5371, 2018.  
Yao Liu, Adith Swaminathan, Alekh Agarwal, and Emma Brunskill. Off-policy policy gradient with state distribution correction. arXiv preprint arXiv:1904.08473, 2019b.  
Yao Liu, Adith Swaminathan, Alekh Agarwal, and Emma Brunskill. Provably good batch reinforcement learning without great exploration. arXiv preprint arXiv:2007.08202, 2020.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529-533, 2015.

Matej Moravčík, Martin Schmid, Neil Burch, Viliam Lisý, Dustin Morrill, Nolan Bard, Trevor Davis, Kevin Waugh, Michael Johanson, and Michael Bowling. Deepstack: Expert-level artificial intelligence in heads-up no-limit poker. Science, 356(6337):508-513, 2017.  
Rémi Munos. Performance bounds in  $\ell_p$ -norm for approximate value iteration. SIAM journal on control and optimization, 46(2):541-561, 2007.  
Rémi Munos and Csaba Szepesvári. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research, 9(5), 2008.  
Ofir Nachum and Bo Dai. Reinforcement learning via Fenchel-Rockafeller duality. arXiv preprint arXiv:2001.01866, 2020.  
Ofir Nachum, Yinlam Chow, Bo Dai, and Lihong Li. DualDICE: Behavior-agnostic estimation of discounted stationary distribution corrections. In Advances in Neural Information Processing Systems, pp. 2315-2325, 2019a.  
Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and Dale Schuurmans. AlgaeDICE: Policy gradient from arbitrary experience. arXiv preprint arXiv:1912.02074, 2019b.  
Ashvin Nair, Murtaza Dalal, Abhishek Gupta, and Sergey Levine. Accelerating online reinforcement learning with offline datasets. arXiv preprint arXiv:2006.09359, 2020.  
XuanLong Nguyen, Martin J Wainwright, and Michael I Jordan. Estimating divergence functionals and the likelihood ratio by convex risk minimization. IEEE Transactions on Information Theory, 56(11):5847-5861, 2010.  
Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.  
Martin L Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons, 2014.  
Tabish Rashid, Bei Peng, Wendelin Boehmer, and Shimon Whiteson. Optimistic exploration even with a pessimistic initialisation. In International Conference on Learning Representations, 2019.  
Paria Rashidinejad, Banghua Zhu, Cong Ma, Jiantao Jiao, and Stuart Russell. Bridging offline reinforcement learning and imitation learning: A tale of pessimism. Advances in Neural Information Processing Systems, 34:11702-11716, 2021.  
Shideh RezaEIFar, Robert Dadashi, Nino Vieillard, L'ONard Hussenot, Olivier Bachem, Olivier Pietquin, and Matthieu Geist. Offline reinforcement learning as anti-exploration. In Proceedings of the AAAI Conference on Artificial Intelligence, number 7, pp. 8106-8114, 2022.  
R Tyrrell Rockafellar and Roger J-B Wets. Variational analysis, volume 317. Springer Science & Business Media, 2009.  
Stephane Ross and J Andrew Bagnell. Reinforcement and imitation learning via interactive no-regret learning. arXiv preprint arXiv:1406.5979, 2014.  
Bruno Scherrer. Approximate policy iteration schemes: A comparison. In International Conference on Machine Learning, pp. 1314-1322, 2014.  
Alexander Shapiro, Darinka Dentcheva, and Andrzej Ruszczynski. Lectures on Stochastic Programming: Modeling and Theory. SIAM, 2021.  
Laixi Shi and Yuejie Chi. Distributionally robust model-based offline reinforcement learning with near-optimal sample complexity. arXiv preprint arXiv:2208.05767, 2022.  
Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, and Yuejie Chi. Pessimistic Q-learning for offline reinforcement learning: Towards optimal sample complexity. arXiv preprint arXiv:2202.13890, 2022.

Noah Y Siegel, Jost Tobias Springenberg, Felix Berkenkamp, Abbas Abdolmaleki, Michael Neunert, Thomas Lampe, Roland Hafner, and Martin Riedmiller. Keep doing what worked: Behavioral modelling priors for offline reinforcement learning. arXiv preprint arXiv:2002.08396, 2020.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of Go without human knowledge. nature, 550(7676):354-359, 2017.  
Adith Swaminathan and Thorsten Joachims. Batch learning from logged bandit feedback through counterfactual risk minimization. The Journal of Machine Learning Research, 16(1):1731-1755, 2015.  
Csaba Szepesvári and Rémi Munos. Finite time bounds for sampling based fitted value iteration. In Proceedings of the 22nd international conference on Machine learning, pp. 880-887, 2005.  
Guy Tennenholtz, Nir Baram, and Shie Mannor. Latent geodesics of model dynamics for offline reinforcement learning. In Deep RL Workshop NeurIPS 2021, 2021.  
Masatoshi Uehara and Wen Sun. Pessimistic model-based offline reinforcement learning under partial coverage. In International Conference on Learning Representations, 2021.  
Masatoshi Uehara, Jiawei Huang, and Nan Jiang. Minimax weight and Q-function learning for off-policy evaluation. In International Conference on Machine Learning, pp. 9659-9668. PMLR, 2020.  
Masatoshi Uehara, Xuezhou Zhang, and Wen Sun. Representation learning for online and offline RL in low-rank MDPs. arXiv preprint arXiv:2110.04652, 2021.  
Sara Van de Geer. Empirical Processes in M-estimation, volume 6. Cambridge university press, 2000.  
Oriol Vinyals, Timo Ewalds, Sergey Bartunov, Petko Georgiev, Alexander Sasha Vezhnevets, Michelle Yeo, Alireza Makhzani, Heinrich Kuttler, John Agapiou, Julian Schrittwieser, et al. Starcraft II: A new challenge for reinforcement learning. arXiv preprint arXiv:1708.04782, 2017.  
Lingxiao Wang, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural policy gradient methods: Global optimality and rates of convergence. In International Conference on Learning Representations, 2019.  
Ruosong Wang, Dean P Foster, and Sham M Kakade. What are the statistical limits of offline rl with linear function approximation? arXiv preprint arXiv:2010.11895, 2020.  
Xinqi Wang, Qiwen Cui, and Simon S Du. On gap-dependent bounds for offline reinforcement learning. arXiv preprint arXiv:2206.00177, 2022.  
Yuanhao Wang, Ruosong Wang, and Sham Kakade. An exponential lower bound for linearly realizable MDP with constant suboptimality gap. Advances in Neural Information Processing Systems, 34:9521-9533, 2021.  
Gellert Weisz, Philip Amortila, and Csaba Szepesvári. Exponential lower bounds for planning in MDPs with linearly-realizable optimal action-value functions. In Algorithmic Learning Theory, pp. 1237-1264. PMLR, 2021.  
Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized offline reinforcement learning. arXiv preprint arXiv:1911.11361, 2019.  
Tengyang Xie and Nan Jiang. Batch value-function approximation with only realizability. In International Conference on Machine Learning, pp. 11404-11413. PMLR, 2021.  
Tengyang Xie, Yifei Ma, and Yu-Xiang Wang. Towards optimal off-policy evaluation for reinforcement learning with marginalized importance sampling. Advances in Neural Information Processing Systems, 32, 2019.

Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, and Alekh Agarwal. Bellman-consistent pessimism for offline reinforcement learning. Advances in neural information processing systems, 34:6683-6694, 2021.  
Yuling Yan, Gen Li, Yuxin Chen, and Jianqing Fan. The efficacy of pessimism in asynchronous Q-learning. arXiv preprint arXiv:2203.07368, 2022.  
Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans. Off-policy evaluation via the regularized Lagrangian. Advances in Neural Information Processing Systems, 33:6551-6561, 2020.  
Ming Yin and Yu-Xiang Wang. Towards instance-optimal offline reinforcement learning with pessimism. Advances in neural information processing systems, 34:4065-4078, 2021.  
Ming Yin, Yu Bai, and Yu-Xiang Wang. Near-optimal offline reinforcement learning via double variance reduction. arXiv preprint arXiv:2102.01748, 2021.  
Ming Yin, Yaqi Duan, Mengdi Wang, and Yu-Xiang Wang. Near-optimal offline reinforcement learning with linear representation: Leveraging variance information with pessimism. arXiv preprint arXiv:2203.05804, 2022.  
Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Zou, Sergey Levine, Chelsea Finn, and Tengyu Ma. MOPO: Model-based offline policy optimization. arXiv preprint arXiv:2005.13239, 2020.  
Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, and Chelsea Finn. COMBO: Conservative offline model-based policy optimization. arXiv preprint arXiv:2102.08363, 2021.  
Andrea Zanette. Exponential lower bounds for batch reinforcement learning: Batch RL can be exponentially harder than online RL. In International Conference on Machine Learning, pp. 12287-12297. PMLR, 2021.  
Andrea Zanette, Martin J Wainwright, and Emma Brunskill. Provable benefits of actor-critic methods for offline reinforcement learning. Advances in neural information processing systems, 34: 13626-13640, 2021.  
Wenhao Zhan, Baihe Huang, Audrey Huang, Nan Jiang, and Jason Lee. Offline reinforcement learning with realizability and single-policy concentrability. In Conference on Learning Theory, pp. 2730-2775. PMLR, 2022.  
Junyu Zhang, Alec Koppel, Amrit Singh Bedi, Csaba Szepesvari, and Mengdi Wang. Variational policy gradient method for reinforcement learning with general utilities. arXiv preprint arXiv:2007.02151, 2020a.  
Ruiyi Zhang, Bo Dai, Lihong Li, and Dale Schuurmans. GenDICE: Generalized offline estimation of stationary values. In International Conference on Learning Representations, 2020b.  
Shantong Zhang, Bo Liu, and Shimon Whiteson. GradientDICE: Rethinking generalized offline estimation of stationary values. arXiv preprint arXiv:2001.11113, 2020c.  
Xuezhou Zhang, Yiding Chen, Xiaojin Zhu, and Wen Sun. Corruption-robust offline reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pp. 5757-5773. PMLR, 2022.
