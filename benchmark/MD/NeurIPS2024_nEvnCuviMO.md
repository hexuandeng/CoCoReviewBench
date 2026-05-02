# Metric Distortion Under Probabilistic Voting

Anonymous Author(s)

# Abstract

Metric distortion in social choice provides a framework for assessing how well voting rules minimize social cost in scenarios where voters and candidates exist in a shared metric space, with voters submitting rankings and the rule outputting a single winner. We expand this framework to include probabilistic voting. Our extension encompasses a broad range of probability functions, including widely studied models like Plackett-Luce (PL) and Bradley-Terry, and a novel "pairwise quantal voting" model inspired by quantal response theory. We demonstrate that distortion results under probabilistic voting better correspond with conventional intuitions regarding popular voting rules such as Plurality, Copeland, and Random Dictator (RD) than those under deterministic voting. For example, in the PL model with candidate strength inversely proportional to the square of their metric distance, we show that Copeland's distortion is at most 2, whereas that of RD is  $\Omega(\sqrt{m})$  in large elections, where  $m$  is the number of candidates. This contrasts sharply with the classical model, where RD beats Copeland with a distortion of 3 versus 5 [1].

# 1 Introduction

Societies must make decisions collectively; different agents often have conflicting interests, and the choice of the mechanism used for combining everyone's opinions often makes a big difference to the outcome. The machine learning community has applied social choice principles for AI alignment [2, 3], algorithmic fairness [4, 5], and preference modelling [6, 7]. Over the last century, there has been increasing interest in using computational tools to analyse and design voting rules [8-11]. One prominent framework for evaluating voting rules is that of distortion [12], where the voting rule has access to only the ordinal preferences of the voters. However, the figure of merit is the sum of all voters' cardinal utilities (or costs). The distortion of a voting rule is the worst-case ratio of the cost of the alternative it selects and the cost of the truly optimal alternative.

An additional assumption is imposed in metric distortion [1] – that the voters and candidates all lie in a shared (unknown) metric space, and costs are given by distances (thus satisfying non-negativity and triangular inequality). This model is a generalization of a commonly studied spatial model of voting in the Economics literature [13, 14], and has a natural interpretation of voters liking candidates with a similar ideological position across many dimensions. While metric distortion is a powerful framework and has led to the discovery and re-discovery of interesting voting rules (e.g. Plurality Veto [15] and the study of Maximal Lotteries [16] for metric distortion by Charikar et al. [17]), its outcomes sometimes do not correspond with traditional wisdom around popular voting rules. For example, the overly simple Random Dictator (RD) rule (where the winner is the top choice of a uniform randomly selected voter) beats the Copeland rule (which satisfies the Condorcet Criterion [10] and other desirable properties) with a metric distortion of 3 versus 5 [1].

While not yet adopted in the metric distortion framework, there is a mature line of work on Probabilistic voting (PV) [18-20]. Here, the focus is on the behavioural modelling of voters and accounting for the randomness of their votes. Two sources of this randomness often cited in the literature are the boundedness of the voters' rationality and the noise in their estimates of candidates' positions. A popular model for this behaviour is based on the Quantal Response Theory [20]. Another closely related line of work is on Random Utility Models (RUMs) [21-23] in social choice where

the hypothesis is that the candidates have ground-truth strengths. Voters make noisy observations of these strengths and vote accordingly. We adopt these models of voting behaviour and study it within the metric distortion framework. The questions we ask are:

Given a model of probabilistic voting, what is the metric distortion of popular voting rules?

How does this differ (qualitatively and quantitatively) from the deterministic model?

# 1.1 Preliminaries and Notation

Let  $\mathcal{N}$  be a set of  $n$  voters and  $\mathcal{A}$  be the set of  $m$  candidates. Let  $S$  be the set of total orders on  $\mathcal{A}$ . Each voter  $i\in \mathcal{N}$  has a preference ranking  $\sigma_{i}\in S$ . A vote profile is a set of preference rankings  $\sigma_{\mathcal{N}} = (\sigma_1,\dots,\sigma_n)\in S^n$  for all voters. The tuple  $(\mathcal{N},\mathcal{A},\sigma_{\mathcal{N}})$  defines an instance of an election. Let  $\Delta (\mathcal{A})$  denote the set of all probability distributions over the set of candidates.

Definition 1 (Voting Rule). A voting rule  $f: S^n \to \Delta(\mathcal{A})$  takes a vote profile  $\sigma_N$  and outputs a probability distribution  $p$  over the alternatives.

For deterministic voting rules, we overload notation by saying that the rule's output is a candidate and not a distribution. We now define some voting rules [10]. Let  $\mathbb{I}$  denote the indicator function.

Random Dictator Rule: Select a voter uniformly at random and output their top choice, i.e.,  $\mathrm{RD}(\sigma_{\mathcal{N}}) = p$  such that  $p_j = \frac{1}{n}\sum_{i\in \mathcal{N}}\mathbb{I}(\sigma_{i,1} = j)$ .

Plurality Rule: Choose the candidate who is the top choice of the most voters, i.e.,  $\mathrm{PLU}(\sigma_{\mathcal{N}}) = \arg \max_{j\in \mathcal{A}}\sum_{i\in \mathcal{N}}\mathbb{I}(\sigma_{i,1} = j)$ . Ties are broken arbitrarily.

Copeland Rule: Choose the candidate who wins the most pairwise comparisons, i.e.,  $\mathrm{COP}(\sigma_{\mathcal{N}}) = \arg \max_{j\in \mathcal{A}}\sum_{j'\in \mathcal{A}\setminus \{j\}}\mathbb{I}\left(\sum_{i\in \mathcal{N}}\mathbb{I}(j\succ_{\sigma_i}j') > \frac{n}{2}\right)$ . Ties are broken arbitrarily.

Distance function  $d: (\mathcal{N} \cup \mathcal{A})^2 \to \mathbb{R}_{\geq 0}$  satisfies triangular inequality  $(d(x, y) \leq d(x, z) + d(z, y))$  and symmetry  $(d(x, y) = d(y, x))$ . The distance between voter  $i \in \mathcal{N}$  and candidate  $j \in \mathcal{A}$  is also referred to as the cost of  $j$  for  $i$ . We consider the most commonly studied social cost function, which is the sum of the costs of all voters.  $SC(j, d) := \sum_{i \in \mathcal{N}} d(i, j)$ .

In deterministic voting, the preference ranking  $\sigma_{i}$  of voter  $i$  is consistent with the distances. That is,  $d(i,j) > d(i,j') \implies j' \succ_{\sigma_{i}} j$  for all voters  $i \in \mathcal{N}$  and candidates  $j, j' \in \mathcal{A}$ . Let  $\rho(\sigma_{\mathcal{N}})$  be the set of distance functions  $d$  consistent with vote profile  $\sigma_{\mathcal{N}}$ . The metric distortion of a voting rule is:

Definition 2 (Metric Distortion).  $\mathrm{DIST}(f) = \sup_{\mathcal{N},\mathcal{A},\sigma_{\mathcal{N}}}\sup_{d\in \rho (\sigma_{\mathcal{N}})}\frac{\mathbb{E}[SC(f(\sigma_{\mathcal{N}}),d)]}{\min_{j\in\mathcal{A}}SC(j,d)}.$

# 1.2 Our Contributions

We extend the study of metric distortion to probabilistic voting (Definition 4). This extension is useful since voters, in practice, have been shown to vote randomly [20]. We define axiomatic properties of models of probabilistic voting which are suitable for studying metric distortion. These are scalefreeness with distances (Axiom 1), pairwise order probabilities being independent of other candidates (Axiom 2), and strict monotonicity of pairwise order probabilities in distances (Axiom 3).

All our results apply to a broad class of models of probabilistic models, as explained in § 2. We provide distortion bounds for all  $n \geq 3$  and  $m \geq 2$ , which are most salient in the limit  $n \to \infty$ . For large elections ( $m$  fixed,  $n \to \infty$ ), we provide matching upper and lower bounds on the distortion of Plurality, an upper bound for Copeland, and a lower bound for RD. The distortion of plurality grows linearly in  $m$ . The distortion upper bound of Copeland is constant. The distortion lower bound for RD increases sublinearly in  $m$  where this rate depends on the probabilistic model. Crucially, our results match those in deterministic voting in the limit where the randomness goes to zero.

The technique is as follows. For the problem of maximizing the distortion, we establish a critical threshold of the expected fraction of votes on pairwise comparisons on all edges on a directed path from a winner to the "true optimal" candidate for Copeland and Plurality. This path is one or two hops for Copeland and one for Plurality. We then formulate a linear-fractional program which incorporates this critical threshold. We linearize this program via the sub-level sets technique [24], and find a feasible solution of the dual problem. Concentration inequalities on this solution provide an upper bound on the distortion. We find a matching lower bound for Plurality by construction.

# 1.3 Related Work

Metric distortion Anshelevich et al. [1] initiated the study of metric distortion and showed that any deterministic voting rule has a distortion of at least 3 and that Copeland has a distortion of 5. The Plurality Veto Rule attains the optimal distortion of 3 [15]. Charikar and Ramakrishnan [25] showed that any randomized voting rule has a distortion of at least 2.112. Charikar et al. [17] gave a randomized voting rule with a distortion of at most 2.753. Anshelevich et al. [26] gave a useful survey on distortion in social choice.

Distortion with Additional Information Abramowitz et al. [27] showed that deterministic voting rules achieve a distortion of 2 when voters provide preference strengths as ratios of distances. Amanatidis et al. [28] demonstrated that even a few queries from each voter can significantly improve distortion in non-metric settings. Anshelevich et al. [29] examined threshold approval voting, where voters approve candidates with utilities above a threshold. Our work relates to these studies since in probabilistic voting, the likelihood of a voter switching the order of two candidates depends on the relative strength of their preference, often resulting in lower distortion than deterministic methods.

Probabilisitc voting and random utility models (RUMs) Hinich [30] showed that the celebrated Median Voter Theorem of [31] does not hold under probabilistic voting. Classical work has focused on studying the equilibrium positions of voters and/or candidates in game-theoretic models of probabilistic voting [20, 32-35]. McKelvey and Patty [20] adopt the quantal response model, a popular way to model agents' bounded rationality.

RUMs have mostly been studied in social choice [21, 23, 36] with the hypothesis that candidates have universal ground-truth strengths, which voters make noisy observations of. Our model is the same as RUM regarding the voters' behaviour; however, voters have independent costs from candidates. The Plackett-Luce (PL) model [37, 38] has been widely studied in social choice [39-41]. For probabilities on pairwise orders, PL reduces to the Bradley-Terry (BT) model [42]. These probabilities are proportional to candidates' strengths (which we define as the inverse of powers of costs).

The widely studied Mallows model [43], based on Condorcet [44], flips the order of each candidate pair (relative to a ground truth ranking) with a constant probability  $p \in (0, \frac{1}{2})$  [45, 46]. The process is repeated if a linear order is not attained. In the context of metric distortion, a limitation of this model is that it doesn't account for the relative distance of candidates to the voter. For a comprehensive review of RUM models, see Marden [47]. Critchlow et al. [48] does an axiomatic study of RUM models; our axioms are grounded in metric distortion and are distinct from theirs.

Recently, there has been significant interest in smoothed analysis [49] of social choice. Here a small amount of randomness is added to problem instances and its effect is studied on the satisfiability of axioms [50-53] and the computational complexity of voting rules [54-56]. Baumeister et al. [50] term this model as being 'towards reality,' highlighting the need to study the randomness in the election instance generation processes. Unlike smoothed analysis where the voter and candidate positions are randomized, we consider these positions fixed, but the submitted votes are random given these positions. The technical difference appears in the benchmark (the "optimal" outcome in the denominator of the distortion is unchanged in our framework and changes in smoothed analysis).

# 2 Axioms and Model

Under probabilistic voting, the submitted preferences may no longer be consistent with the underlying distances. For a distribution  $\mathcal{P}(d)$  over  $\sigma_{\mathcal{N}}$ , let  $q^{\mathcal{P}(d)}(i,j,j')$  denote the induced marginal probability that voter  $i$  ranks candidate  $j$  higher than  $j'$ . We focus on these marginal probabilities on pairwise orders and provide axioms for classifying which  $q^{\mathcal{P}(d)}(\cdot)$  are suitable for studying distortion.

Axiom 1 (Scale-Freeness (SF)). The probability  $q^{\mathcal{P}(d)}(\cdot)$  must be invariant to scaling of  $d$ . That is, for any tuple  $(i,j,j')$  and any constant  $\kappa > 0$ , we must have  $q^{\mathcal{P}(d)}(i,j,j') = q^{\mathcal{P}(\kappa d)}(i,j,j')$ .

Note that the metric distortion (Definition 2) for deterministic voting is scale-free. We want to retain the same property in the probabilistic model as well. Conceptually, one may think of the voter's preferences as being a function of the relative (and not absolute) distances to the candidates.

Axiom 2 (Independence of Other Candidates (IOC)). The probability  $q^{\mathcal{P}(d)}(i,j,j')$  must be independent of the distance of voter  $i$  to all 'other' candidates, i.e., those in  $\mathcal{A} \setminus \{j,j'\}$ .

Table 1: Axioms satisfied by commonly studied models of probabilistic voting  

<table><tr><td></td><td>Axiom 1: SF</td><td>Axiom 2: IOC</td><td>Axiom 3: Strict Monotonicity</td></tr><tr><td>Mallows</td><td>✓</td><td>✕</td><td>✕</td></tr><tr><td>PL/BT with exponential in d</td><td>✕</td><td>✓</td><td>✓</td></tr><tr><td>PL/BT with powers of d</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>PQV</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

This axiom extends Luce's choice axioms [38], defined for selecting the top choice, to entire rankings. IOC is reminiscent of the independence of irrelevant alternatives axiom for voting rules.

Axiom 3 (Strict Monotonicity (SM)). For every tuple  $(i,j,j')$ , for fixed distance  $d(i,j) > 0$ , the probability  $q^{\mathcal{P}(d)}(i,j,j')$  must be strictly increasing in  $d(i,j')$  at all but at most finitely many points.

The monotonicity in  $d(i,j)$  follows since  $q^{\mathcal{P}(d)}(i,j',j) = 1 - q^{\mathcal{P}(d)}(i,j,j')$ . This axiom is natural.

In the Mallows model [43],  $q^{\mathcal{P}(d)}(\cdot)$  was derived by Busa-Fekete et al. [57] and is as follows:

$$
\text {M a l l o w s :} \quad q ^ {\mathcal {P} (d)} (i, j, j ^ {\prime}) = h \left(r _ {j ^ {\prime}} - r _ {j} + 1, \phi\right) - h \left(r _ {j ^ {\prime}} - r _ {j}, \phi\right). \tag {1}
$$

Here  $h(k,\phi) = \frac{k}{(1 - \phi^k)}$ . Whereas  $r_j$  and  $r_{j'}$  are the positions of  $j$  and  $j'$  in the ground-truth (noiseless) ranking, and the constant  $\phi$  is a dispersion parameter. Observe that this model fails Axiom 2 since it depends on the number of candidates between  $j$  and  $j'$  in the noiseless ranking. It also fails Axiom 3 since it does not depend on the exact distances but only on the order of the distances.

Plackett-Luce Model: The PL model [37, 38] is 'sequential' in the following way. For each voter  $i \in \mathcal{N}$ , each candidate  $j \in \mathcal{A}$  has a 'strength'  $s_{i,j}$ . In most of the literature on RUMs, a common assumption is that  $s_{i,j}$  is the same for all voters  $i$ . However, we choose this more general model to make it useful in the context of metric distortion. The voter chooses their top choice with probability proportional to the strengths. Similarly, for every subsequent rank, they choose a candidate from among the remaining ones with probabilities proportional to their strengths. In terms of the pairwise order probabilities, the PL model reduces to the Bradley-Terry (BT) model [42], that is:

$$
\text {P L / B T :} \quad q ^ {\mathcal {P} (d)} \left(i, j, j ^ {\prime}\right) = \frac {s _ {i , j}}{s _ {i , j} + s _ {i , j ^ {\prime}}} \tag {2}
$$

Prima facie, in the metric distortion framework, any decreasing function of distance  $d(i,j)$  would be a natural choice for  $s_{i,j}$ . However, not all such functions satisfy Axiom 1. The exponential function is a popular choice in the literature employing BT or PL models. However, in general,  $\frac{e^{-d(i,j)}}{e^{-d(i,j)} + e^{-d(i,j')} \neq \frac{e^{-2d(i,j)}}{e^{-2d(i,j)} + e^{-2d(i,j')} \}}$ , thus failing the Scale-Freeness Axiom 1.

On the other hand, observe that all functions  $s = d^{-\theta}$  for any  $\theta \in (0,\infty)$  satisfy our axioms. We use the regime  $\theta \in (1,\infty)$  for technical simplicity in this work.

We also define the following class of functions "PQV" for  $q^{\mathcal{P}(d)}(\cdot)$  motivated by Quantal Response Theory [58] and its use in probabilistic voting [20]. Observe that PQV satisfies all our axioms.

Definition 3 (Pairwise Quantal Voting (PQV)). Let the relative preference  $r(i,j,j')$  be the ratio of distances,  $\frac{d(i,j')}{d(i,j)}$ . For constant  $\lambda > 0$ , PQV is as follows:  $q^{\mathcal{P}(d)}(i,j,j') = \frac{e^{-\lambda / r(i,j,j')}}{e^{-\lambda r(i,j,j')} + e^{-\lambda / r(i,j,j')}}$ .

We now define a general class of functions for pairwise order probabilities in terms of the relative preference (ratio of distances)  $r$ . Let  $\mathbf{G}$  be a class of functions such that any  $\mathbf{G} \ni g:[0,\infty) \cup \{\infty\} \to [0,1]$  has the following properties.

1.  $g$  is continuous and twice-differentiable.  
2.  $g(0) = 0$ . Further,  $g'(r) > 0 \forall r \in (0,\infty)$  i.e.  $g(r)$  is strictly increasing in  $[0,\infty)$ .  
3. Define  $\frac{1}{r}$  as  $+\infty$  when  $r = 0$ . Then we must have  $g(r) + g(\frac{1}{r}) = 1 \forall r \geq 0$ .  
4. There  $\exists c\in [0,\infty)$  s.t.  $g^{\prime \prime}(r) > 0$ $\forall r\in (0,c)$  i.e.  $g$  is convex in the open interval  $(0,c)$

Observe that PL (with  $g(r) = \frac{r^{\theta}}{1 + r^{\theta}}, \theta > 1$ ) and PQV (with  $g(r) = \frac{e^{-\lambda / r}}{e^{-\lambda r} + e^{-\lambda / r}}, \lambda > 0$ ) are in G. Construction of distributions (if any exists) on rankings  $\sigma_{\mathcal{N}}$  which generate pairwise order

![](images/88d16e6ea94275acec64ad94ba089033b5acba13237ba974636b6911665e7a8d.jpg)  
Figure 1: A 1-d Euclidean example of voting probabilities. There are two candidates at 0 and 1. The figure on the left shows the voter position between 0 and 1. In the right figure, the voter is in positions to the left of 0. As the distance grows, both candidates look similar to the voter in the probabilistic model but not in deterministic voting. The case of voter positions to the right of 1 is symmetric.

![](images/8d4e61381ca7a2043edfdb8915c4cc0c8533231aed068b031321bd41098390bc.jpg)

probabilities  $q^{\mathcal{P}(d)}(i,j,j') = g(\frac{d(i,j')}{d(i,j)})$  according to PQV is left for future work. We do not need it for our technical derivations. For PL, these distributions are known from prior work [40].

We assume  $g \in \mathbf{G}$  in the rest of the paper. Let  $\mathcal{M}(\mathcal{N} \cup \mathcal{A})$  denote the set of valid distance functions on  $(\mathcal{N}, \mathcal{A})$ . For any  $g$  and  $d \in \mathcal{M}(\mathcal{N} \cup \mathcal{A})$  let  $\hat{\mathcal{P}}^{(g)}(d)$  denote the set of probability distributions on  $\sigma_{\mathcal{N}}$  for which the marginal pairwise order probabilities are  $g\left(\frac{d(i,j')}{d(i,j)}\right)$ . That is,

$$
\forall \mathcal {P} \in \hat {\mathcal {P}} ^ {(g)} (d), \sigma_ {\mathcal {N}} \sim \mathcal {P} \Rightarrow \mathbb {P} [ A \succ_ {i} B ] = g \left(\frac {d (i , B)}{d (i , A)}\right). \tag {3}
$$

We assume that all voters vote independently of each other. We now define metric distortion under probabilistic voting as a function of  $g$  for a given  $m$  and  $n$ .

Definition 4 (Metric Distortion under Probabilistic Voting).

$$
\operatorname {D i s t} ^ {(g)} (f, n, m) := \sup  _ {\substack {\mathcal {N}: | \mathcal {N} | = n \\ \mathcal {A}: | \mathcal {A} | = m}} \sup  _ {d \in \mathcal {M} (\mathcal {N} \cup \mathcal {A})} \sup  _ {\mathcal {P} \in \hat {\mathcal {P}} ^ {(g)} (d)} \frac {\mathbb {E} _ {\sigma_ {\mathcal {N}} \sim \mathcal {P}} [ S C (f (\sigma_ {N}) , d) ]}{\min  _ {A \in \mathcal {A}} S C (A , d)}. \tag{4}
$$

$\mathrm{DIST}^{(g)}(f) = \sup_{n,m}\mathrm{DIST}^{(g)}(f,n,m)$  by supremizing over all possible  $n$  and  $m$ .

The expectation is both over the randomness in the votes and the voting rule  $f$ .

Observe that the distortion is a supremum over all distributions in  $\hat{\mathcal{P}}^{(g)}(d)$ . Since we focus on large elections (with large  $n$  and relatively small  $m$ ), we define  $\mathrm{DIST}^{(g)}$  as a function of  $m$  and  $n$ .

As in Fig. 1, consider the 1-d Euclidean space with candidate  $X$  at the origin and  $Y$  at 1. Observe that  $g\left(\frac{x}{1 - x}\right)$  and  $g\left(\frac{x}{1 + x}\right)$  denote the probability that a voter located at a distance  $x$  from  $X$  votes for  $Y$  when the voter is to the left and right of  $X$  respectively. Interestingly, this 1-d intuition extends well for general metric spaces. Towards this, we define the following functions.

$$
g _ {\mathrm {M I D}} (x) := g \left(\frac {x}{1 - x}\right) \forall x \in (0, 1) \text {a n d} g _ {\mathrm {O U T}} (x) := g \left(\frac {x}{1 + x}\right) \forall x \in [ 0, \infty). \tag {5}
$$

Lemma 1.  $\frac{g_{\mathrm{MID}}(x)}{x}$  and  $\frac{g_{\mathrm{OUT}}(x)}{x}$  have unique local maxima in  $(0,1)$  and  $(0,\infty)$  respectively.

Denote the unique maximisers of  $\frac{g_{\mathrm{MID}}(x)}{x}$  and  $\frac{g_{\mathrm{OUT}}(x)}{x}$  by  $x_{\mathrm{MID}}^*$  and  $x_{\mathrm{OUT}}^*$  respectively.

For simplifying notation, in the rest of the work, we use  $\hat{g}_{\mathrm{MID}}$  for  $\frac{g_{\mathrm{MID}}(x_{\mathrm{MID}}^*)}{x_{\mathrm{MID}}^*}$  and  $\hat{g}_{\mathrm{OUT}}$  for  $\frac{g_{\mathrm{OUT}}(x_{\mathrm{OUT}}^*)}{x_{\mathrm{OUT}}^*}$ .

In the analysis in the rest of the paper, we will see  $\hat{g}_{\mathrm{MID}}$  and  $\hat{g}_{\mathrm{OUT}}$  appear many times, so we note these quantities for the PL and PQV models here. For the PL model with  $\theta = 2$ ,  $\hat{g}_{\mathrm{MID}} = \frac{\sqrt{2} + 1}{2} \approx 1.21$  and  $\hat{g}_{\mathrm{OUT}} = \frac{\sqrt{2} - 1}{2} \approx 0.21$ . When  $\theta = 4$ ,  $\hat{g}_{\mathrm{MID}} \approx 1.42$  and  $\hat{g}_{\mathrm{OUT}} \approx 0.06$ . When  $\theta \to \infty$ ,  $\hat{g}_{\mathrm{MID}} \to 2$  and  $\hat{g}_{\mathrm{OUT}} \to 0$ . This limit is where PL resembles deterministic voting.

For PQV with  $\lambda = 1$ ,  $\hat{g}_{\mathrm{MID}} \approx 1.25$  and  $\hat{g}_{\mathrm{OUT}} = 0.18$ . When  $\lambda \to \infty$ ,  $\hat{g}_{\mathrm{MID}} \to 2$  and  $\hat{g}_{\mathrm{OUT}} \to 0$ .

# 3 Distortion of Plurality Rule Under Probabilistic Voting

In this section, we give upper and lower bounds on the distortion of the Plurality rule [59] (PLU). In the limit the number of voters  $n \to \infty$  ("large election"), our upper and lower bounds match and are linear in the number of candidates  $m$ . Let  $B$  represent the candidate that minimizes the social cost (referred to as 'best'), and let  $\{A_j\}_{j \in [m-1]}$  denote the set of other candidates.

# 3.1 Upper bound on the distortion of Plurarity(PLU)

Theorem 1. For every  $\epsilon >0$  and  $m\geq 2$  and  $n\geq m^2$  we have

$$
\begin{array}{l} \operatorname {D I S T} ^ {(g)} (\mathrm {P L U}, n, m) \leq m (m - 1) \left(\hat {g} _ {\mathrm {M I D}} + \hat {g} _ {\mathrm {O U T}}\right) \exp \left(\frac {- n ^ {\left(\frac {1}{2} + \epsilon\right)} + 2 m}{\left(2 n ^ {\left(\frac {1}{2} - \epsilon\right)} - 1\right) m}\right) \tag {6} \\ + \max  \left(\frac {m \hat {g} _ {\mathrm {M I D}}}{\left(1 - n ^ {- \left(\frac {1}{2} - \epsilon\right)}\right)} - 1, \frac {m \hat {g} _ {\mathrm {O U T}}}{\left(1 - n ^ {- \left(\frac {1}{2} - \epsilon\right)}\right)} + 1\right). \\ \end{array}
$$

Further,  $\lim_{n\to \infty}\mathrm{DIST}^{(g)}(\mathrm{PLU},n,m)\leq \max \left(m\hat{g}_{\mathrm{MID}} - 1,m\hat{g}_{\mathrm{OUT}} + 1\right)$

To prove this theorem, we first give a lemma which upper bounds  $\frac{SC(W,d)}{SC(B,d)}$  under the constraint that the expected number of voters that rank candidate  $W$  over  $B$  is given by  $\alpha$ . This ratio will be useful to bound the contribution of non-optimal candidate  $W$  to the distortion of PLU. We state an optimization problem (7) below, which would be required to bound the ratio as a function of  $\alpha$ .

$$
\mathcal {E} _ {\alpha} = \left\{ \begin{array}{c c} \min  _ {\mathbf {b}, \mathbf {w} \in \mathbb {R} _ {\geq 0} ^ {n}} \frac {\sum_ {i = 1} ^ {n} b _ {i}}{\sum_ {i = 1} ^ {n} w _ {i}} & \\ \text {s . t .} \quad \sum_ {i = 1} ^ {n} g \left(\frac {b _ {i}}{w _ {i}}\right) \geq \alpha & \forall \alpha \geq 0 \\ \max  _ {i} | w _ {i} - b _ {i} | \leq \min  _ {i} (w _ {i} + b _ {i}) & \end{array} \right. \tag {7}
$$

Lemma 2. For any two candidates  $W, B \in \mathcal{A}$  which satisfy  $\sum_{i=1}^{n} \mathbb{P}[W \succ_i B] = \alpha$ , we have

$$
\frac {S C (W , d)}{S C (B , d)} \leq \frac {1}{o p t \left(\mathcal {E} _ {\alpha}\right)} \leq \max  \left(\frac {n}{\alpha} \hat {g} _ {\mathrm {M I D}} - 1, \frac {n}{\alpha} \hat {g} _ {\mathrm {O U T}} + 1\right). \tag {8}
$$

Our proof is via Lemmas 3 and 4. Lemma 3 shows that we can bound the ratio of social costs by the inverse of the optimum value of  $\mathcal{E}_{\alpha}$  and Lemma 4 gives a lower bound on the optimum value of  $\mathcal{E}_{\alpha}$ .

Lemma 3. For any two candidates  $W, B \in \mathcal{A}$  satisfying  $\sum_{i=1}^{n} \mathbb{P}[W \succ_i B] = \alpha$ , we have

$$
\frac {S C (W , d)}{S C (B , d)} \leq \frac {1}{o p t \left(\mathcal {E} _ {\alpha}\right)}. \tag {9}
$$

Proof.  $b_{i}$  and  $w_{i}$  in (7) represent the distances  $d(i, B)$  and  $d(i, W)$ . The last constraint is the triangle inequality i.e.  $|d(i, B) - d(i, W)| \leq d(B, W) \leq |d(i, B) + d(i, W)|$  for every voter  $i \in \mathcal{N}$ .  
Consider the following linearized version of (7).

$$
\mathcal {E} _ {\mu , \alpha} = \left\{ \begin{array}{l l} \min  _ {\mathbf {w}, \mathbf {b} \in \mathbb {R} _ {\geq 0} ^ {n}} \left(\sum_ {i = 1} ^ {n} b _ {i}\right) - \mu \left(\sum_ {i = 1} ^ {n} w _ {i}\right) \\ \text {s . t .} \quad \sum_ {i = 1} ^ {n} g \left(\frac {b _ {i}}{w _ {i}}\right) \geq \alpha & \forall 0 \leq \mu \leq 1, \alpha \geq 0. \\ \left| b _ {i} - w _ {i} \right| \leq 1 \forall i \in [ n ] \\ b _ {i} + w _ {i} \geq 1 \forall i \in [ n ] \end{array} \right. \tag {10}
$$

Lemma 4.  $opt(\mathcal{E}_{\alpha}) \geq \min \left( \left( \frac{n}{\alpha} \hat{g}_{\mathrm{MID}} - 1 \right)^{-1}, \left( \frac{n}{\alpha} \hat{g}_{\mathrm{OUT}} + 1 \right)^{-1} \right)$ .

Our proof uses Lemma 5 and is by solving a linearized version of (7) in (10). This is done by introducing an extra non-negative parameter  $\mu \leq 1$ . Note that it is sufficient to consider  $\mu \leq 1$  since  $\mathrm{opt}(\mathcal{E}_{\alpha}) \leq 1$  because  $B$  minimises the social cost by definition. We find the smallest  $\mu \in (0,1)$  such that its objective is non-negative.

Lemma 5. If  $\text{opt}(\mathcal{E}_{\mu, \alpha}) \geq 0$ , then  $\text{opt}(\mathcal{E}_{\alpha}) \geq \mu$ .

Further,  $opt(\mathcal{E}_{\mu,\alpha}) \geq 0$  if  $\mu = \min \left( \left( \frac{n}{\alpha} \hat{g}_{\mathrm{MID}} - 1 \right)^{-1}, \left( \frac{n}{\alpha} \hat{g}_{\mathrm{OUT}} + 1 \right)^{-1} \right)$ .

The first part follows since scaling each term by a constant  $r$  satisfies the constraints and also yields the same objective. And thus we may replace the constraints by  $\max_i |w_i - b_i| \leq 1$  and  $\min_i (w_i + b_i) \geq 1$  in equation (10). Further, the objective function is linearized as  $(\sum_{i=1}^n b_i) - \mu(\sum_{i=1}^n w_i)$ .

The proof of the second part is technical and has been moved to Appendix B. It involves introducing a Lagrangian multiplier  $\lambda$  and demonstrating that the objective function is non-negative for a suitably chosen  $\lambda$ . To establish this, we show that minimising the Lagrangian over the boundaries of the constraint set given by  $|b_{i} - w_{i}| = 1$  and  $b_{i} + w_{i} = 1$  is sufficient. This requires a careful analysis.

The main technique used in proving Theorem 1 involves considering two cases for every non-optimal candidate  $A_{j}$ : one where the expected number of voters ranking candidate  $A_{j}$  above  $B$  (call it  $\alpha_{j}$ ) exceeds a threshold of  $\frac{n}{m} - \frac{n^{\epsilon + 1 / 2}}{m}$  and one where it does not. In the first case, the ratio of social costs of  $A_{j}$  and  $B$  is bounded using Lemma 2 that naturally gives a bound on contribution of candidate  $A_{j}$  to the distortion. In the later case, we use Chernoff bound to bound the probability of  $A_{j}$  being the winner and multiply it with the ratio of social costs of  $A_{j}$  and  $B$  to bound the distortion. The proof of Theorem 1 is in Appendix C.

# 3.2 Lower bound on the distortion of Plurality

We now present a lower bound on the distortion of PLU for any  $m$  in the limit  $n$  tends to infinity. This lower bound matches the upper bound of Theorem 1 in the limit. A full proof is in Appendix D. Note that the proof has an adversarially chosen distribution over the rankings subject to the marginals on pairwise relationships satisfying  $g$  (as in the definition of distortion under probabilistic voting 4). This lower bound does not apply to the PL model, which has a specific distribution over rankings.

Theorem 2. For every  $m \geq 2$ ,  $\lim_{n \to \infty} \mathrm{DIST}^{(g)}(\mathrm{PLU}, n, m) \geq \max(m \hat{g}_{\mathrm{MID}} - 1, m \hat{g}_{\mathrm{OUT}} + 1)$ .

Proof Sketch. The proof is by an example in an Euclidean metric space in  $\mathbb{R}^3$ . One candidate "C" is at  $(1,0,0)$ . The other  $m - 1$  candidates are "good" and are equidistantly placed on a circle of radius  $\epsilon$  on the  $y - z$  plane centred at  $(0,0,0)$ . We call them  $\mathcal{G} \coloneqq \{G_1, G_2, \ldots, G_{m-1}\}$ .

We present sketches of two constructions below for every  $\epsilon, \zeta > 0$ .

Construction 1: Let  $q_{\mathrm{MID}} \coloneqq g\left(\frac{\sqrt{(x_{\mathrm{MID}}^*)^2 + \epsilon^2}}{1 - x_{\mathrm{MID}}^*}\right)$  and  $a_{\mathrm{MID}} \coloneqq \frac{1}{m-1}\left(1 - \frac{1 + \zeta}{mq_{\mathrm{MID}}}\right)$ . Each of the  $m-1$  candidates in  $\mathcal{G}$  has  $\lfloor a_{\mathrm{MID}}n \rfloor$  voters overlapping with it. The remaining voters (we call them "ambivalent") are placed at  $(x_{\mathrm{MID}}^*,0,0)$ . Clearly, each voter overlapping with a candidate votes for it as the most preferred candidate with probability one. Each of the ambivalent voters votes as follows.

- With probability  $q_{\mathrm{MID}}$ , vote for candidate  $C$  as the top choice and uniformly randomly permute the other candidates in the rest of the vote.

- With probability  $1 - q_{\mathrm{MID}}$ , vote for candidate  $C$  as the last choice and uniformly randomly permute the other candidates in the rest of the vote.

We show that the probability that  $C$  wins tends to 1 as  $n\to \infty$  and the distortion is  $m\hat{g}_{\mathrm{MID}} - 1$

Construction 2: We give a construction where the locations of the candidates are identical as in Construction 1, and some voters are located with the "good" candidates. The ambivalent voters are at  $(-x_{\mathrm{OUT}}^*,0,0)$ . We show that  $\mathbb{P}[C$  wins] tends to 1 as  $n\to \infty$  and the distortion is  $m\hat{g}_{\mathrm{OUT}} + 1$ .

This result establishes that the distortion of Plurality is bound to increase linearly with  $m$  even under probabilistic voting, and is therefore not a good choice when  $m$  is even moderately large.

# 4 Distortion of Copeland Rule Under Probabilistic Voting

We now bound the distortion of the Copeland voting rule. We say that candidate  $W$  defeats candidate  $Y$  if more than half of the voters rank  $W$  above  $Y$ .

Theorem 3. For every  $\epsilon >0,m\geq 2$  and  $n\geq 4$  we have

$$
\begin{array}{l} \operatorname {D I S T} ^ {(g)} (\operatorname {C O P}, n, m) \leq 4 m (m - 1) \exp \left(\frac {- n ^ {\left(\frac {1}{2} + \epsilon\right)} + 8}{2 \left(2 n ^ {\left(\frac {1}{2} - \epsilon\right)} - 1\right)}\right) \left(\hat {g} _ {\mathrm {M I D}} + \hat {g} _ {\mathrm {O U T}}\right) ^ {2} \\ + \max \Bigl (\Bigl (\frac {2 \hat {g} _ {\mathrm {M I D}}}{1 - n ^ {- (\frac {1}{2} - \epsilon)}} - 1 \Bigr) ^ {2}, \Bigl (\frac {2 \hat {g} _ {\mathrm {O U T}}}{1 - n ^ {- (\frac {1}{2} - \epsilon)}} + 1 \Bigr) ^ {2} \Bigr). \\ \end{array}
$$

For every  $m \geq 2$ , we have  $\lim_{n \to \infty} \mathrm{DIST}^{(g)}(\mathrm{COP}, n, m) \leq \max \left( (2\hat{g}_{\mathrm{MID}} - 1)^2, (2\hat{g}_{\mathrm{OUT}} + 1)^2 \right)$ .

Proof Sketch. A Copeland winner belongs to the uncovered set in the tournament graph, as demonstrated in [1, Theorem 15]. Recall that  $B$  denotes the candidate with the least social cost. For a Copeland winner  $W$ , either  $W$  defeats  $B$  or it defeats a candidate  $Y$  who defeats  $B$ .

We now consider two exhaustive cases on candidate  $A_{j}$  and define event  $E_{j}$  for every  $j \in [m - 1]$  by computing the expected fraction of votes on pairwise comparisons. The event  $E_{j}$  denotes the existence of an at-most two hop directed path from a candidate  $A_{j}$  to candidate  $B$  for Copeland such that the expected fraction of votes on all edges along that path exceeds  $\frac{n}{2} - \frac{n^{(1/2 + \epsilon)}}{2}$ .

If  $E_{j}$  holds true, we upper bound the ratio of social cost of candidate  $A_{j}$  and social cost of candidate  $B$  using Lemma 2 which in-turn would give a bound on the distortion. Otherwise, we use union bound and Chernoff's bound to upper bound the probability of  $A_{j}$  being the winner. Multiplying the probability bound with the ratio of social costs (one obtained from Lemma 2) leads to a bound on the distortion. A detailed proof is in Appendix E.

# 5 Distortion of Random Dictator Rule Under Probabilistic Voting

We first give an upper bound on the distortion of RD; the proof is in Appendix F.

Theorem 4.  $\mathrm{DIST}^{(g)}(RD,m,n)\leq (m - 1)\hat{g}_{\mathrm{MID}} + 1$

We now give a lower bound on the distortion of RD. We do this by constructing an example.

Theorem 5. For  $m \geq 3$  and  $n \geq 2$ ,  $\mathrm{DIST}^{(g)}(RD, m, n) \geq 2 + \frac{1}{g^{-1}\left(\frac{1}{m - 1}\right)} - \frac{2}{n}$ .

Proof. We have a 1-D Euclidean construction. Let  $B$  be at 0 and all other candidates  $\mathcal{A} \setminus \{B\}$  be at 1.  $m - 1$  voters are at 0 and one voter  $V$  is at  $\tilde{x} = g^{-1}\left(\frac{1}{m - 1}\right) / (1 + g^{-1}\left(\frac{1}{m - 1}\right))$ .

The ranking for  $V$  is generated as follows: pick a candidate from  $\mathcal{A} \setminus \{B\}$  as the top rank uniformly at random. Keep  $B$  on the second rank. Permute the remaining candidates uniformly at random for the remaining ranks. Observe that the marginal pairwise order probabilities are consistent with the distance of  $V$  from  $B$  and each candidate in  $\mathcal{A} \setminus \{B\}$ . In particular  $g\left(\frac{\tilde{x}}{1 - \tilde{x}}\right) = \frac{1}{m - 1}$ . The distortion for this instance is  $\mathbb{P}[B \text{ wins} ] \cdot 1 + \mathbb{P}[B \text{ loses} ] \cdot \frac{n - \tilde{x}}{\tilde{x}} = \frac{n - 1}{n} + \frac{1}{n} \frac{n - \tilde{x}}{\tilde{x}} = 1 + \frac{1}{\tilde{x}} - \frac{2}{n} = 2 + \frac{1}{g^{-1}\left(\frac{1}{m - 1}\right)} - \frac{2}{n}$ .

For  $g(r) = \frac{r^{\theta}}{1 + r^{\theta}}$ , we have  $g^{-1}(t) = \left(\frac{t}{1 - t}\right)^{\frac{1}{\theta}}$ . Then  $g^{-1}\left(\frac{1}{m - 1}\right) = (m - 2)^{-\frac{1}{\theta}}$ , and the distortion lower bound is  $\mathrm{DIST}^{(g)}(\mathrm{RD}, m, n) \geq 2 + (m - 2)^{\frac{1}{\theta}} - \frac{2}{n}$ , and  $\lim_{n \to \infty} \mathrm{DIST}^{(g)}(\mathrm{RD}, m, n) \geq 2 + (m - 2)^{\frac{1}{\theta}}$ .

However, note that this result does not apply to the PL model! This is because the PL model has a specific distribution on the rankings. In contrast, the above result is obtained by choosing an adversarial distribution on rankings subject to the constraint that its marginals on pairwise relations are given by  $g$ . In the PL model,  $\mathbb{P}[A_j$  is top-ranked in  $\sigma_i] = \frac{d(i,A_j)^{-\theta}}{\sum_{A_k \in \mathcal{A}} d(i,A_k)^{-\theta}}$  [45]. We have the following result for the PL model. A proof via a similar construction as Theorem 5 is in Appendix G.

Theorem 6. Let  $\mathrm{DIST}_{PL}^{\theta}(RD, m, n)$  denote the distortion when the voters' rankings are generated per the  $PL$  model with parameter  $\theta$ . We have  $\lim_{n \to \infty} \mathrm{DIST}_{PL}^{\theta}(RD, m, n) \geq 1 + \frac{(m - 1)^{1 / \theta}}{2}$ .

![](images/c923c8b0184b4f9bbc58f7234e00f209036849fa3d3cf2301e80a6a8c392ba86.jpg)  
Figure 2: Here, we illustrate how the distortion bounds on different voting rules vary with  $m$  and with the randomness parameters of the two models, PL and PQV, in the limit  $n \to \infty$ . Both the x and y axes are on the log scale. We plot the upper bound for Copeland (Theorem 3), the lower bound for RD (Theorem 5), and the matching bounds for Plurality (Theorem 1).

![](images/7676945c7a2b97139c5fb61902e6c0353f29a2424f7209a6044f66f4bfdadd1d.jpg)

Recall that higher values of  $\theta$  and  $\lambda$  correspond to lower randomness. From Figure 2, we observe that under sufficient randomness, the more intricate voting rule Copeland outshines the simpler rule RD, which only looks at a voter's top choice. Moreover, its distortion is independent of  $m$  in the limit  $n\to \infty$ . This is in sharp contrast to RD, where the distortion is  $\Omega (m^{1 / \theta})$  in the PL model, a sharp rate of increase in  $m$  for low values of  $\theta$ . The distortion of Plurality increases linearly in  $m$ .

An important observation is regarding the asymptotics when  $\theta$  or  $\lambda$  increases. The distortion of RD converges to its value under deterministic voting, i.e., 3. The distortion of Plurality also converges to  $2m - 1$ , the same as in deterministic voting. Since our bound on Copeland is not tight, it converges to 9 rather than 5. So far, in the study of metric distortion, the social choice community has looked only at these asymptotic; here, we present insights available from looking at the 'complete' picture. Interestingly, the distortion of RD increases with randomness, whereas that of Copeland decreases up to a certain point and then increases again. The reason for the increases in the high randomness regime is that the votes become too noisy to reveal the best candidate any more.

Since these plots have no abrupt transitions, this figure hints that smoothened analysis [52] (typically done with small amounts of noise) is unlikely to give any new insights regarding metric distortion.

# 7 Discussion and Future Work

We extend the metric distortion framework in social choice in an important way – by capturing the bounded rationality and randomness in voters' behaviour. Consideration of this randomness shows that, in general, the original metric distortion framework is too pessimistic on important voting rules, most notably on Copeland. On the other hand, the simplistic voting rule Random Dictator, which attains a distortion of 3 (at least as good as any deterministic rule [1]), is not so good when we look at the full picture – its distortion increases with the number of candidates in our model. Our framework opens up opportunities to revisit the metric distortion problem with a closer-to-reality view of voters. It may hopefully lead to the development of new voting rules that consider the randomness of voters' behaviour. For example, Liu and Moitra [46] take a learning theory approach to design voting rules under the assumption of random voting per the Mallows model. However, technical analysis in our framework may be challenging because of the interplay of the geometric structure of voters' positions and the probabilistic nature of their votes.

Future Work An interesting extension would be to other tournament graph-based voting rules (weighted or unweighted). Our techniques are well-suited for this class of rules since it is based on the expected weights of the edges of the tournament graph. Closing the gap for the distortion of Copeland would be useful for getting deeper insights. Another open problem is the characterization of the set of distributions on rankings that induce the pairwise probabilities per PQV.

# References

[1] Elliot Anshelevich, Onkar Bhardwaj, and John Postl. Approximating optimal social choice under metric preferences. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, pages 777-783, 2015.  
[2] Jessica Dai and Eve Fleisig. Mapping social choice theory to RLHF. arXiv preprint arXiv:2404.13038, 2024.  
[3] Vincent Conitzer, Rachel Freedman, Jobst Heitzig, Wesley H Holliday, Bob M Jacobs, Nathan Lambert, Milan Mossé, Eric Pacuit, Stuart Russell, Hailey Schoelkopf, et al. Social choice for AI alignment: Dealing with diverse human feedback. arXiv preprint arXiv:2404.10271, 2024.  
[4] Seth D Baum. Social choice ethics in artificial intelligence. AI & Society, 35(1):165-176, 2020.  
[5] Jessie Finocchiaro, Roland Maio, Faidra Monachou, Gourab K Patro, Manish Raghavan, Ana-Andreea Stoica, and Stratis Tsirtsis. Bridging machine learning and mechanism design towards algorithmic fairness. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 489–503, 2021.  
[6] Francesca Rossi, Kristen Brent Venable, and Toby Walsh. A Short Introduction to Preferences: Between AI and Social Choice. Morgan & Claypool Publishers, 2011.  
[7] Meltem Öztürk, Alexis Tsoukias, and Philippe Vincke. Preference modelling. Multiple criteria decision analysis: State of the art surveys, 78:27-59, 2005.  
[8] Kenneth J Arrow. A difficulty in the concept of social welfare. Journal of political economy, 58 (4):328-346, 1950.  
[9] Amartya Sen. Social choice theory. Handbook of mathematical economics, 3:1073-1181, 1986.  
[10] Kenneth J Arrow, Amartya Sen, and Kotaro Suzumura. Handbook of social choice and welfare, volume 2. Elsevier, 2010.  
[11] Felix Brandt, Vincent Conitzer, Ulle Endriss, Jérôme Lang, and Ariel D Procaccia. Handbook of computational social choice. Cambridge University Press, 2016.  
[12] Ariel D Procaccia and Jeffrey S Rosenschein. The distortion of cardinal preferences in voting. In International Workshop on Cooperative Information Agents, pages 317-331. Springer, 2006.  
[13] James M Enelow and Melvin J Hinich. The spatial theory of voting: An introduction. CUP Archive, 1984.  
[14] Samuel Merrill and Bernard Grofman. A unified theory of voting: Directional and proximity spatial models. Cambridge University Press, 1999.  
[15] Fatih Erdem Kizilkaya and David Kempe. Plurality veto: A simple voting rule achieving optimal metric distortion. Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI), pages 349-355, 2022.  
[16] Germain Kreweras. Aggregation of preference orderings. In Mathematics and Social Sciences I: Proceedings of the seminars of Menthon-Saint-Bernard, France (1-27 July 1960) and of Gösing, Austria (3-27 July 1962), pages 73-79, 1965.  
[17] Moses Charikar, Prasanna Ramakrishnan, Kangning Wang, and Hongxun Wu. Breaking the metric voting distortion barrier. In Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1621-1640. SIAM, 2024.  
[18] Peter J Coughlin. Probabilistic voting theory. Cambridge University Press, 1992.  
[19] Kevin M Quinn, Andrew D Martin, and Andrew B Whitford. Voter choice in multi-party democracies: a test of competing theories and models. American Journal of Political Science, pages 1231-1247, 1999.  
[20] Richard D McKelvey and John W Patty. A theory of voting in large elections. Games and Economic Behavior, 57(1):155-180, 2006.

[21] Thomas Pfeiffer, Xi Gao, Yiling Chen, Andrew Mao, and David Rand. Adaptive polling for information aggregation. In Proceedings of the AAAI conference on artificial intelligence, volume 26, pages 122-128, 2012.  
[22] David C Parkes, Houssein Azari Soufiani, and Lirong Xia. Random utility theory for social choice. In Proceedings of the 25th Annual Conference on Neural Information Processing Systems. Curran Associates, Inc., 2012.  
[23] Hossein Azari Soufiani, David C Parkes, and Lirong Xia. Preference elicitation for general random utility models. In Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence, pages 596-605, 2013.  
[24] Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.  
[25] Moses Charikar and Prasanna Ramakrishnan. Metric distortion bounds for randomized social choice. In Proceedings of the 2022 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 2986-3004. SIAM, 2022.  
[26] Elliot Anshelevich, Aris Filos-Ratsikas, Nisarg Shah, and Alexandros A Voudouris. Distortion in social choice problems: The first 15 years and beyond. In 30th International Joint Conference on Artificial Intelligence, pages 4294-4301, 2021.  
[27] Ben Abramowitz, Elliot Anshelevich, and Wennan Zhu. Awareness of voter passion greatly improves the distortion of metric social choice. In International Conference on Web and Internet Economics, pages 3-16. Springer, 2019.  
[28] Georgios Amanatidis, Georgios Birmpas, Aris Filos-Ratsikas, and Alexandros A Voudouris. Peeking behind the ordinal curtain: Improving distortion via cardinal queries. Artificial Intelligence, 296:103488, 2021.  
[29] Elliot Anshelevich, Aris Filos-Ratsikas, Christopher Jerrett, and Alexandros A Voudouris. Improved metric distortion via threshold approvals. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 9460-9468, 2024.  
[30] Melvin J Hinich. Equilibrium in spatial voting: The median voter result is an artifact. Journal of Economic Theory, 16(2):208-219, 1977.  
[31] Duncan Black. On the rationale of group decision-making. Journal of political economy, 56(1): 23-34, 1948.  
[32] Jeffrey S Banks and John Duggan. Probabilistic voting in the spatial model of elections: The theory of office-motivated candidates. In Social Choice and Strategic Decisions: Essays in Honor of Jeffrey S. Banks, pages 15-56. Springer, 2005.  
[33] John Wiggs Patty. Local equilibrium equivalence in probabilistic voting models. Games and Economic Behavior, 51(2):523-536, 2005.  
[34] Peter Coughlin and Shmuel Nitzan. Electoral outcomes with probabilistic voting and nash social welfare maxima. Journal of Public Economics, 15(1):113-121, 1981.  
[35] Peter Coughlin and Shmuel Nitzan. Directional and local electoral equilibria with probabilistic voting. Journal of Economic Theory, 24(2):226-239, 1981.  
[36] Lirong Xia. Designing social choice mechanisms using machine learning. In Proceedings of the international conference on Autonomous agents and multi-agent systems, pages 471-474, 2013.  
[37] Robin L Plackett. The analysis of permutations. Journal of the Royal Statistical Society Series C: Applied Statistics, 24(2):193-202, 1975.  
[38] R Duncan Luce. Individual choice behavior: A theoretical analysis. Courier Corporation, 2005.  
[39] Isobel Claire Gormley and Thomas Brendan Murphy. Analysis of Irish third-level college applications data. Journal of the Royal Statistical Society Series A: Statistics in Society, 169(2): 361-379, 2006.

[40] Hossein Azari, David Parks, and Lirong Xia. Random utility theory for social choice. Advances in Neural Information Processing Systems, 25, 2012.  
[41] Isobel Claire Gormley and Thomas Brendan Murphy. A grade of membership model for rank data. Bayesian Analysis, 1(1):1-32, 2004.  
[42] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika, 39(3/4):324-345, 1952.  
[43] Colin L Mallows. Non-null ranking models. i. Biometrika, 44(1/2):114-130, 1957.  
[44] Marquis de Condorcet. Essay on the application of analysis to the probability of majority decisions. Paris: Imprimerie Royale, page 1785, 1785.  
[45] Ioannis Caragiannis, Ariel D Procaccia, and Nisarg Shah. When do noisy votes reveal the truth? ACM Transactions on Economics and Computation (TEAC), 4(3):1-30, 2016.  
[46] Allen Liu and Ankur Moitra. Robust voting rules from algorithmic robust statistics. In Proceedings of the Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 3471-3512. SIAM, 2023.  
[47] John I Marden. Analyzing and modeling rank data. CRC Press, 1996.  
[48] Douglas E Critchlow, Michael A Fligner, and Joseph S Verducci. Probability models on rankings. Journal of mathematical psychology, 35(3):294-318, 1991.  
[49] Daniel A Spielman and Shang-Hua Teng. Smoothed analysis of algorithms: Why the simplex algorithm usually takes polynomial time. Journal of the ACM (JACM), 51(3):385-463, 2004.  
[50] Dorothea Baumeister, Tobias Hogrebe, and Jörg Rothe. Towards reality: smoothed analysis in computational social choice. In Proceedings of the 19th International Conference on Autonomous Agents and Multiagent Systems, pages 1691–1695, 2020.  
[51] Bailey Flanigan, Daniel Halpern, and Alexandros Psomas. Smoothed analysis of social choice revisited. In International Conference on Web and Internet Economics, pages 290–309. Springer, 2023.  
[52] Lirong Xia. The smoothed possibility of social choice. Advances in Neural Information Processing Systems, 33:11044-11055, 2020.  
[53] Lirong Xia. Semi-random impossibilities of condorcet criterion. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 5867-5875, 2023.  
[54] Ao Liu and Lirong Xia. The semi-random likelihood of doctrinal paradoxes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 5124-5132, 2022.  
[55] Lirong Xia and Weiqiang Zheng. The smoothed complexity of computing kemeny and slater rankings. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 5742-5750, 2021.  
[56] Lirong Xia and Weiqiang Zheng. Beyond the worst case: Semi-random complexity analysis of winner determination. In International Conference on Web and Internet Economics, pages 330-347. Springer, 2022.  
[57] Róbert Busa-Fekete, Eyke Hüllermeier, and Balázs Szö rényi. Preference-based rank elicitation using statistical models: The case of mallows. In International conference on machine learning, pages 1071-1079. PMLR, 2014.  
[58] Richard D McKelvey and Thomas R Palfrey. Quantal response equilibria for normal form games. Games and economic behavior, 10(1):6-38, 1995.  
[59] Kenneth J. Arrow. Social Choice and Individual Values. Yale University Press, New Haven, 2 edition, 1963.  
[60] John Canny. Chernoff bounds. URL https://people.eecs.berkeley.edu/~jfc/cs174/ lecs/lec10/lec10.pdf.
