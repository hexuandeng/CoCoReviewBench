# Uncoupled Learning Dynamics with  $O(\log T)$  Swap Regret in Multiplayer Games

Anonymous Author(s)

Affiliation

Address

email

# Abstract

In this paper we establish efficient and uncoupled learning dynamics so that, when employed by all players in a general-sum multiplayer game, the swap regret of each player after  $T$  repetitions of the game is bounded by  $O(\log T)$ , improving over the prior best bounds of  $O(\log^4 (T))$ . At the same time, we guarantee optimal  $O(\sqrt{T})$  swap regret in the adversarial regime as well. To obtain these results, our primary contribution is to show that when all players follow our dynamics with a time-invariant learning rate, the second-order path lengths of the dynamics up to time  $T$  are bounded by  $O(\log T)$ , a fundamental property which could have further implications beyond near-optimally bounding the (swap) regret. Our proposed learning dynamics combine in a novel way optimistic regularized learning with the use of self-concordant barriers. Further, our analysis is remarkably simple, bypassing the cumbersome framework of higher-order smoothness recently developed by Daskalakis, Fishelson, and Golowich (NeurIPS'21).

# 1 Introduction

Online learning and game theory share an intricately connected history tracing back to the inception of the modern no-regret framework with Robinson's analysis of fictitious play [Robinson, 1951] and Blackwell's approachability theorem [Blackwell, 1956]. Indeed, the no-regret framework addresses the fundamental question of how independent and decentralized agents can "learn" with only limited feedback from their environment, and has led to celebrated connections with game-theoretic equilibrium concepts [Hart and Mas-Colell, 2000, Foster and Vohra, 1997]. One of the remarkable features of these results is that the learning dynamics are fully uncoupled [Hart and Mas-Colell, 2000]: each player is completely agnostic to the utilities of the other players. Thus, there is no communication between the players or any centralized authority dictating behavior throughout the game. Instead, the only "coordination device" is the common history of play. An additional desideratum, which is fundamentally tied to the no-regret framework, is what Daskalakis et al. [2011] refer to as strong uncoupledness: players have no information whatsoever about the game (even their own utilities), and they only make decisions based on the utilities received as feedback throughout the repeated game.

In this context, it is well-known that there are broad families of no-regret learning algorithms that, after  $T$  repetitions, guarantee regret bounded by  $O(\sqrt{T})$ , and this bound is known to be insuperable in adversarial environments [Cesa-Bianchi and Lugosi, 2006]. However, this begs the question: What if the player is not facing adversarial utilities, but instead is competing with other learning agents in a repeated game? This question was first formulated and addressed by Daskalakis et al.

[2011], who devised strongly uncoupled dynamics converging with a near-optimal rate of  $O\left(\frac{\log T}{T}\right)$  in zero-sum games, a substantial improvement over the  $O(1 / \sqrt{T})$  rate obtained via traditional approaches within the no-regret framework. Thereafter, there has been a considerable amount of effort in strengthening their result, leading to extensions along several important lines [Rakhlin and Sridharan, 2013, Syrgkanis et al., 2015, Chen and Peng, 2020, Farina et al., 2019, Daskalakis et al., 2021, Anagnostides et al., 2021, Wei and Luo, 2018, Foster et al., 2016]. In particular, in a recent breakthrough result, Daskalakis et al. [2021] showed that when all players in a general game employ an optimistic variant of multiplicative weights update (MWU) (henceforth OMWU), the external regret of each player grows as  $O(\log^4 (T))$ . That result was also subsequently extended to the substantially more challenging performance measure of swap regret [Anagnostides et al., 2021]. Perhaps the main drawback of the latter results is the complexity of the analysis, relying on establishing a refined property for the dynamics they refer to as higher-order smoothness. Our primary contribution in this paper is to develop a novel and much simpler framework, which furthermore improves the prior state of the art  $O(\log^4 (T))$  regret bounds to  $O(\log T)$  in general multiplayer games.

# 1.1 Overview of Our Contributions

Before we state our main result, let us first introduce some basic notation. We assume that each player  $i \in [[n]]$  selects at every iteration  $t$  of the repeated game a probability distribution (mixed strategy) over the set of available actions  $\boldsymbol{x}_i^{(t)} \in \Delta(\mathcal{A}_i)$  (see Section 2 for further details). The following theorem is the primary contribution of our work. $^2$

Theorem 1.1 (Precise Statement in Theorem 4.4). There exist strongly uncoupled no-swap-regret learning dynamics so that when employed by all players with learning rate  $\eta = O(1)$ , the second-order path lengths of the dynamics up to any time  $T \in \mathbb{N}$  are bounded by  $O(\log T)$ ; that is,

$$
\sum_ {t = 1} ^ {T} \sum_ {i = 1} ^ {n} \| \pmb {x} _ {i} ^ {(t)} - \pmb {x} _ {i} ^ {(t - 1)} \| _ {1} ^ {2} = O (\log T).
$$

We are not aware of even an  $o(T)$  bound for the second-order path lengths—under a time-invariant learning rate—prior to our work, except for very restricted classes of games such as zero-sum games. The dynamics of Theorem 1.1 combine: (i) the celebrated no-swap-regret template of Blum and Mansour [2007]; (ii) the optimistic follow the regularizer leader (OFTRL) algorithm of Syrgkanis et al. [2015]; and (iii) using a self-concordant barrier as a regularizer. The latter was introduced in online learning in the seminal work of Abernethy et al. [2008], where the authors obtained the first near-optimal and efficient online learning algorithm for linear bandit optimization; the way we leverage the log-barrier in the setting of no-regret learning in games is novel, and crucially leverages the local norm induced by the regularizer. The dynamics of Theorem 1.1 are also efficiently implementable (see Remark 4.7).

The implication of Theorem 1.1 is perhaps surprising in view of the inherent cycling aspect of no-regret learning in general games. Indeed, it is by now well-understood that any no-regret dynamics will fail to converge—at least for certain games (e.g., see [Milionis et al., 2022]). Nevertheless, Theorem 1.1 implies that players will change their strategies arbitrarily slowly as the game progresses. As such, players will observe utilities that exhibit very small variation over time, immediately implying near-optimal swap regret.

Corollary 1.2 (Precise Statement in Corollaries 4.5 and 4.6). There exist strongly uncoupled no-swap-regret learning dynamics so that when employed by all players, the individual swap regret of each player is bounded by  $O(\log T)$ . At the same time, when faced against adversarial utilities each player guarantees  $O(\sqrt{T})$  swap regret.

Corollary 1.2 improves over the prior best bounds of  $O(\log^4 (T))$  [Daskalakis et al., 2021, Anagnostides et al., 2021]; a comparison with prior works regarding the algorithm of Blum and Mansour [2007] is given in Table 1. In fact, Corollary 1.2 yields, to our knowledge, the first no-regret guarantee in general games for uncoupled methods when players use a time-invariant learning rate, a feature that has been extensively motivated in prior works (see, e.g., the discussion in [Bailey and

Table 1: Prior results regarding the no-swap-regret algorithm of Blum and Mansour [2007] (BM). The second column indicates the algorithm internally employed by the "master" BM algorithm; our construction uses OFTRL with log-barrier regularization (Section 3). Further,  $m$  is the maximum number of actions available to each player. We point out that in the adversarial swap regret bound we have suppressed lower order factors in terms of  $T$ .  

<table><tr><td>Reference</td><td>Algorithm</td><td>Swap Regret in Games</td><td>Adversarial Swap Regret</td></tr><tr><td>Blum and Mansour [2007]</td><td>E.g., BM-MWU</td><td>—</td><td>O(√m log mT)</td></tr><tr><td>Chen and Peng [2020]</td><td>BM-OMWU</td><td>O(√n(m log m)3/4)T1/4)</td><td>O(√mT)</td></tr><tr><td>Anagnostides et al. [2021]</td><td>BM-OMWU</td><td>O(nm4 log(m) log4(T))</td><td>—</td></tr><tr><td>This paper</td><td>BM-OFTRL-LogBar</td><td>O(nm5/2 log T)</td><td>O(√m log mT)</td></tr></table>

[Piliouras, 2019]). Corollary 1.2 also establishes near-optimality in the adversarial regime as well, a crucial desideratum in this line of work. Finally, swap regret is a powerful notion of hindsight rationality, trivially subsuming external regret. In particular, in light of well-established connections (see Theorem 2.3), we obtain the best known rate of convergence of  $O\left(\frac{\log T}{T}\right)$  to correlated equilibria in general games.

Corollary 1.3. There exist strongly uncoupled learning dynamics so that, when employed by all players, the average correlated distribution of play after  $T$  repetitions of the game is an  $O\left(\frac{\log T}{T}\right)$ -approximate correlated equilibrium.

From a technical standpoint, our approach is conceptually remarkably simple and direct. Specifically, Theorem 1.1 is shown by first establishing the RVU bound—a fundamental property first identified in [Syrgkanis et al., 2015, Definition 3]—for swap regret in Theorem 4.3; the key ingredient is Lemma 4.2, which crucially leverages the local norm induced by the log-barrier regularizer over the simplex. Next, Theorem 1.1 follows directly by making a seemingly trivial observation: swap regret is always nonnegative. A related approach was recently employed in [Anagnostides et al., 2022] for external regret, but only works for very restricted classes of games such as zero-sum. As such, we bypasses the cumbersome framework of higher-order smoothness recently introduced by Daskalakis et al. [2021].

# 1.2 Further Related Work

The first accelerated dynamics in general games were established by Syrgkanis et al. [2015]. In particular, they identified a broad class of no-regret learning dynamics—satisfying the so-called RVU property—for which the sum of the players' regrets is  $O(1)$ . On the other hand, they only obtained an  $O(T^{1/4})$  bound for the individual external regret of each player. This is crucial given that the rate of convergence to coarse correlated equilibria is driven by the maximum of the external regrets. It is important to note that a bound for the sum of the external regrets does not necessarily translate to a bound for the maximum since external regrets can be negative. This is in stark contrast to swap regret (Observation 2.1), a property crucially leveraged in our work. Furthermore, the  $O(T^{1/4})$  bounds for the individual external regret in [Syrgkanis et al., 2015] were only recently extended to swap regret by Chen and Peng [2020]. The main challenge with swap regret—which is also the main focus of our paper—is that the underlying dynamics are much more complex, maintaining and aggregating over multiple independent external regret minimizers. In addition, the dynamics involve a fixed point operation—namely, the stationary distribution of a Markov chain—posing new challenges compared to the analysis of no-external-regret algorithms [Chen and Peng, 2020]. Finally, a very intriguing approach for obtaining near-optimal no-external-regret dynamics was recently introduced by Piliouras et al. [2021]. The main caveat of that result is that the dynamics they propose are not uncoupled, which has been a central desideratum in the line of work on no-regret learning in games. For this reason, the result in [Piliouras et al., 2021] is not directly comparable with the previous approaches.

# 2 Preliminaries

In this section we introduce the basic background on online optimization and learning in games. For a comprehensive treatment on the subject we refer the interested reader to the excellent book of Cesa-Bianchi and Lugosi [2006].

Conventions We denote by  $\mathbb{N} = \{1,2,\dots \}$  the set of natural numbers. We use the shorthand notation  $\llbracket n\rrbracket \coloneqq \{1,2,\ldots ,n\}$ . Subscripts are typically used to indicate the player, or a parameter uniquely associated with a player (such as an action available to the player). On the other hand, superscripts are reserved almost exclusively for the (discrete) time index, which is represented via the variable  $t$ . Also, the  $r$ -th coordinate of a  $d$ -dimensional vector  $\pmb{x}\in \mathbb{R}^d$  is denoted by  $\pmb{x}[r]$ . Finally, we let  $\log (\cdot)$  be the natural logarithm.

# 2.1 Online Learning and Phi-Regret

Let  $\mathcal{X} \subseteq \mathbb{R}^d$  be a nonempty convex and compact set of strategies, for some  $d \in \mathbb{N}$ . In the online learning framework the learner has to select at every iteration  $t \in \mathbb{N}$  a strategy  $\boldsymbol{x}^{(t)} \in \mathcal{X}$ . Then, the environment—be it the "nature" or some "adversary"—returns a (linear) utility function  $u^{(t)}: \mathcal{X} \ni \boldsymbol{x} \mapsto \langle \boldsymbol{x}, \boldsymbol{u}^{(t)} \rangle$ , for some utility vector  $\boldsymbol{u}^{(t)} \in \mathbb{R}^d$ , so that the learner receives a utility of  $\langle \boldsymbol{x}^{(t)}, \boldsymbol{u}^{(t)} \rangle$  at time  $t$ . In the full information model the learner receives as feedback the entire utility function, represented by  $\boldsymbol{u}^{(t)}$ . The canonical measure of performance in online learning is based on the notion of regret, or more generally, on Phi-regret [Greenwald and Jafari, 2003, Stoltz and Lugosi, 2007, Gordon et al., 2008]. Formally, for a set of transformations  $\Phi: \mathcal{X} \to \mathcal{X}$ , the  $\Phi$ -regret of a regret minimization algorithm  $\Re$  up to a time horizon  $T \in \mathbb{N}$  is defined as

$$
\operatorname {R e g} _ {\Phi} ^ {T} := \max  _ {\phi^ {*} \in \Phi} \left\{\sum_ {t = 1} ^ {T} \langle \phi^ {*} (\boldsymbol {x} ^ {(t)}), \boldsymbol {u} ^ {(t)} \rangle \right\} - \sum_ {t = 1} ^ {T} \langle \boldsymbol {x} ^ {(t)}, \boldsymbol {u} ^ {(t)} \rangle . \tag {1}
$$

Naturally, a broader collection of transformations leads to a stronger notion of hindsight rationality; canonical instantiations of Phi-regret include:

(i) External regret (denoted by Reg):  $\Phi$  includes only constant transformations;  
(ii) Swap regret (denoted by SwapReg):  $\Phi$  includes all possible linear transformations.

As such, swap regret induces the more powerful notion of hindsight rationality. We point out that our main focus in this paper (Section 4) will be for the special case where  $\mathcal{X}$  is the probability simplex. A crucial property of swap regret is that  $\mathrm{SwapReg} \geq 0$ , as formalized below.

Observation 2.1. Fix any time horizon  $T \in \mathbb{N}$ . For any sequence of utilities  $\pmb{u}^{(1)}, \dots, \pmb{u}^{(T)}$  and any sequence of strategies  $\pmb{x}^{(1)}, \dots, \pmb{x}^{(T)}$  it holds that  $\mathrm{SwapReg}^T \geq 0$ .

In proof, just consider the identity transformation  $\Phi \ni \phi : \pmb{x} \mapsto \pmb{x}$  in (1). In contrast, this property does not necessarily hold for external regret.

Moreover, it will be convenient to model a regret minimization algorithm  $\Re$  as a black box which interacts with its environment via the following two subroutines.

(i)  $\Re .\mathrm{NEXTSTRATEGY}()\colon \Re$  returns the next strategy of the learner;  
(ii)  $\Re$ .OBSERVEUTILITY( $\mathbf{u}$ ):  $\Re$  receives as feedback from the environment a utility vector  $\mathbf{u}$ , and may adapt its internal state accordingly.

# 2.2 No-Regret Learning and Correlated Equilibria

A fundamental connection ensures that as long as all players employ no-swap-regret learning dynamics (in the sense that  $\mathrm{SwapReg}^T = o(T)$ ), the average correlated distribution of play converges to the set of correlated equilibria [Hart and Mas-Colell, 2000, Foster and Vohra, 1997, Blum and Mansour, 2007]. Before we formalize this connection, let us first introduce some basic background on games.

Finite Games Let  $\llbracket n\rrbracket \coloneqq \{1,2,\ldots ,n\}$  be the set of players, with  $n\geq 2$  . In a (finite) game, represented in normal form, each player  $i\in \llbracket n\rrbracket$  has a finite set of actions  $\mathcal{A}_i$  ; for notational simplicity, we will let  $m_{i}\coloneqq |\mathcal{A}_{i}|\geq 2$  . For a given joint action profile  $\pmb {a} = (a_{1},\dots ,a_{n})\in$ $\times_{i = 1}^{n}\mathcal{A}_{i}$  , the (normalized) utility received by player  $i$  is given by some arbitrary function  $u_{i}:$ $\times_{i = 1}^{n}\mathcal{A}_{i}\to [-1,1]$  . Players are allowed to randomize by selecting a (mixed) strategy  $\pmb {x}_i\in$ $\Delta (\mathcal{A}_i)\coloneqq \left\{\pmb {x}\in \mathbb{R}_{\geq 0}^{|A_i|}:\sum_{a_i\in \mathcal{A}_i}\pmb {x}[a_i] = 1\right\}$  ; that is, a probability distribution over the available actions. For a joint strategy profile  $\pmb {x} = (x_{1},\dots ,x_{n})$  , player  $i$  receives an expected utility of  $\mathbb{E}_{\pmb{a}\sim \pmb{x}}[u_i(\pmb {a})] = \sum_{\pmb {a}\in \mathcal{A}}u_i(\pmb {a})\prod_{j\in \llbracket n\rrbracket}\pmb{x}_j[a_j].$

In the problem of no-regret learning in games, every player receives as feedback at time  $t \in \mathbb{N}$  a utility vector  $\pmb{u}_i^{(t)} \in \mathbb{R}^{|A_i|}$ , so that  $\pmb{u}_i^{(t)}[a_i] := u_i(a_i; \pmb{x}_{-i}^{(t)}) := \mathbb{E}_{\pmb{a}_{-i} \sim \pmb{x}_{-i}}[u_i(a_i, \pmb{a}_{-i})]$ , for any  $a_i \in A_i$ ; here, we used the notation  $\pmb{a}_{-i}$  to represent the joint action profile excluding  $i$ 's component, and analogously for the notation  $\pmb{x}_{-i}$ . No other information is available to the player. We are now ready to introduce the concept of a correlated equilibrium due to Aumann [1974].

Definition 2.2 (Correlated Equilibrium [Aumann, 1974]). A probability distribution  $\pmb{\mu}$  over  $\times_{i=1}^{n} \mathcal{A}_{i}$  is an  $\epsilon$ -approximate correlated equilibrium, for  $\epsilon \geq 0$ , if for any player  $i \in [[n]]$  and any swap function  $\phi_{i}: \mathcal{A}_{i} \to \mathcal{A}_{i}$ ,

$$
\mathbb {E} _ {\boldsymbol {a} \sim \boldsymbol {\mu}} \left[ u _ {i} (\boldsymbol {a}) \right] \geq \mathbb {E} _ {\boldsymbol {a} \sim \boldsymbol {\mu}} \left[ u _ {i} \left(\phi_ {i} \left(a _ {i}\right), \boldsymbol {a} _ {- i}\right) \right] - \epsilon .
$$

Theorem 2.3 (Folklore). Suppose that each player  $i \in [[n]]$  employs a no-swap-regret algorithm such that the cumulative swap regret up to time  $T \in \mathbb{N}$  is upper bounded by  $\mathrm{SwapReg}_i^T$ . Further, let  $\pmb{\mu}^{(t)} \coloneqq \pmb{x}_1^{(t)} \otimes \pmb{x}_2^{(t)} \otimes \dots \otimes \pmb{x}_n^{(t)}$  be the product distribution at time  $t \in [[T]]$ , and  $\bar{\pmb{\mu}} \coloneqq \frac{1}{T} \sum_{t=1}^{T} \pmb{\mu}^{(t)}$  be the average correlated distribution of play up to time  $T$ . Then,  $\bar{\pmb{\mu}}$  is a  $\max_{i=1}^{n} \{\mathrm{SwapReg}_i^T / T\}$ -approximate correlated equilibrium.

Consequently, a central challenge for correlated equilibria is that the rate of convergence is driven by the maximum of the swap regrets; this is in contrast to, for example, the rate of convergence of the (utilitarian) social welfare in smooth games, which is driven by the sum of the players' external regrets [Syrgkanis et al., 2015, Roughgarden, 2015].

# 3 Optimistic Learning with Self-Concordant Barriers

Optimistic follow the regularizer leader (OFTRL) [Syrgkanis et al., 2015] is a predictive variant of the standard FTRL paradigm. Specifically, OFTRL maintains an internal prediction vector  $\pmb{m}^{(t)}\in \mathbb{R}^d$  and can be expressed with the following update rule for  $t\in \mathbb{N}$ .

$$
\boldsymbol {x} ^ {(t)} := \underset {\boldsymbol {x} \in \mathcal {X}} {\arg \max } \left\{\Phi^ {(t)} (\boldsymbol {x}) := \eta \left\langle \boldsymbol {x}, \boldsymbol {m} ^ {(t)} + \sum_ {\tau = 1} ^ {t - 1} \boldsymbol {u} ^ {(\tau)} \right\rangle - \mathcal {R} (\boldsymbol {x}) \right\}; \quad \text {(O F T R L)}
$$

here,  $\eta > 0$  serves as the learning rate, and  $\mathcal{R}$  is the regularizer. For convenience, we also define  $\pmb{x}^{(0)} \coloneqq \arg \min_{\pmb{x} \in \mathcal{X}} \mathcal{R}(\pmb{x})$ . Unless specified otherwise, (OFTRL) will be instantiated with  $\pmb{m}^{(t)} \coloneqq \pmb{u}^{(t-1)}$ , for  $t \in \mathbb{N}$ . (For convenience in the analysis, and without any loss, we assume that players initially obtain the utilities corresponding to the other players' strategies at time  $t = 0$ .)

In [Syrgkanis et al., 2015] the regularizer  $\mathcal{R}$  was assumed to be 1-strongly convex with respect to some (static) norm  $\| \cdot \|$  on  $\mathbb{R}^d$ . On the other hand, we are introducing an important twist:  $\mathcal{R}$  will be a self-concordant barrier function over  $\mathcal{X}$ . In this context, we first extend (in Appendix B) the so-called RVU bound established in [Syrgkanis et al., 2015] under self-concordant regularization. More precisely, we assume that  $\mathcal{X}$  has nonempty interior  $\operatorname{int}(\mathcal{X})$ . Further, for  $\boldsymbol{u} \in \mathbb{R}^d$  the primal local norm with respect to  $\boldsymbol{x} \in \operatorname{int}(\mathcal{X})$  is defined as  $\| \boldsymbol{u} \|_{\boldsymbol{x}} \coloneqq \sqrt{\boldsymbol{u}^\top \nabla^2 \mathcal{R}(\boldsymbol{x}) \boldsymbol{u}}$ , while the dual norm is defined as  $\| \boldsymbol{u} \|_{*,\boldsymbol{x}} \coloneqq \sqrt{\boldsymbol{u}^\top (\nabla^2 \mathcal{R}(\boldsymbol{x}))^{-1} \boldsymbol{u}}$ , assuming that  $\mathcal{R}$  nondegenerate—in the sense that its Hessian is positive definite. Finally, for the purpose of the analysis, we let  $\boldsymbol{g}^{(t)}$  denote the be the leader sequence (see (BTL) in Appendix B); no attempt was made to optimize universal constants.

Theorem 3.1 (RVU for Self-Concordant Regularizers). Suppose that  $\mathcal{R}$  is a nondegenerate self-concordant function for  $\operatorname{int}(\mathcal{X})$ . Moreover, let  $\eta > 0$  be such that  $\eta \| \pmb{u}^{(t)} - \pmb{m}^{(t)} \|_{*, \pmb{x}^{(t)}} \leq \frac{1}{2}$  and  $\eta \| \pmb{m}^{(t)} \|_{*, \pmb{g}^{(t-1)}} \leq \frac{1}{2}$  for all  $t \in [T]$ . Then, the regret  $\mathrm{Reg}^T(\pmb{x}^*)$  of (OFTRL) with respect to any comparator  $\pmb{x}^* \in \operatorname{int}(\mathcal{X})$  under any sequence of utilities  $\pmb{u}^{(1)}, \dots, \pmb{u}^{(T)}$  can be bounded by

$$
\frac {\mathcal {R} \left(\boldsymbol {x} ^ {*}\right)}{\eta} + 2 \eta \sum_ {t = 1} ^ {T} \left\| \boldsymbol {u} ^ {(t)} - \boldsymbol {m} ^ {(t)} \right\| _ {*, \boldsymbol {x} ^ {(t)}} ^ {2} - \frac {1}{4 \eta} \sum_ {t = 1} ^ {T} \left(\left\| \boldsymbol {x} ^ {(t)} - \boldsymbol {g} ^ {(t)} \right\| _ {\boldsymbol {x} ^ {(t)}} ^ {2} + \left\| \boldsymbol {x} ^ {(t)} - \boldsymbol {g} ^ {(t - 1)} \right\| _ {\boldsymbol {g} ^ {(t - 1)}} ^ {2}\right).
$$

Here, we also used the standard notation  $\mathrm{Reg}^T(\pmb{x}^*) := \sum_{t=1}^T \langle \pmb{x}^* - \pmb{x}^{(t)}, \pmb{u}^{(t)} \rangle$ . Next, we instantiate Theorem 3.1 using the log-barrier on the (probability) simplex:  $\mathcal{R}(\pmb{x}) = -\sum_{r=1}^d \log(\pmb{x}[r])$ . While the probability simplex has empty interior, there is a simple transformation on the relative interior  $\mathrm{relint}(\Delta^d)$  that addresses that issue (see Appendix B).

Corollary 3.2 (RVU for Log-Barrier on the Simplex). Suppose that  $\mathcal{R}$  is the log-barrier on the simplex and  $\eta \leq \frac{1}{16}$ . Then, the regret of (OFTRL) under any sequence of utilities  $\pmb{u}^{(1)},\dots,\pmb{u}^{(T)}$  can be bounded as

$$
\operatorname {R e g} ^ {T} \left(\boldsymbol {x} ^ {*}\right) \leq \frac {\mathcal {R} \left(\boldsymbol {x} ^ {*}\right)}{\eta} + 2 \eta \sum_ {t = 1} ^ {T} \left\| \boldsymbol {u} ^ {(t)} - \boldsymbol {u} ^ {(t - 1)} \right\| _ {*, \boldsymbol {x} ^ {(t)}} ^ {2} - \frac {1}{1 6 \eta} \sum_ {t = 1} ^ {T} \left\| \boldsymbol {x} ^ {(t)} - \boldsymbol {x} ^ {(t - 1)} \right\| _ {\boldsymbol {x} ^ {(t - 1)}} ^ {2},
$$

for any  $\pmb{x}^{*}\in \mathrm{relint}(\Delta^{d})$ , where  $\| \pmb{x}^{(t)} - \pmb{x}^{(t - 1)}\|_{\pmb{x}^{(t - 1)}}^2 \coloneqq \sum_{r = 1}^{d}\left(\frac{\pmb{x}^{(t)}[r] - \pmb{x}^{(t - 1)}[r]}{\pmb{x}^{(t - 1)}[r]}\right)^2$ .

We remark that a similar regret bound for optimistic mirror descent [Rakhlin and Sridharan, 2013] under log-barrier regularization was shown by [Wei and Luo, 2018, Theorem 7].

# 4 Main Result

In this section we sketch the proof of our main result, namely Theorem 1.1, leading to Corollaries 1.2 and 1.3; detailed proofs are deferred to Appendix C. In this context, we first employ the general template of Blum and Mansour [2007] for constructing a no-swap-regret minimizer  $\Re_{swap}$  over the simplex. We proceed with a brief overview of their construction (summarized in Algorithm 1). In the sequel, we first perform the analysis from the perspective of a single player, without explicitly indicating so in our notation.

The Algorithm of Blum and Mansour Blum and Mansour [2007] construct a "master" regret minimization algorithm  $\mathfrak{R}_{swap}$  by maintaining a separate and independent external regret minimizer  $\mathfrak{R}_a$  for every action  $a\in \mathcal{A}$ . To compute the next strategy,  $\mathfrak{R}_{swap}$  first obtains the strategy  $\pmb{x}_a^{(t)}\in \Delta (\mathcal{A})$  of  $\mathfrak{R}_a$ , for every  $a\in \mathcal{A}$ . Then, a (row) stochastic matrix  $\mathbf{Q}^{(t)}\in \mathbb{S}^{|A|}$  is constructed, so that the row associated with action  $a\in \mathcal{A}$  is equal to the distribution  $\pmb{x}_a^{(t)}$ , while  $\mathfrak{R}_{swap}$  outputs as the next strategy  $\pmb{x}^{(t)}\in \Delta (\mathcal{A})$  any stationary distribution of  $\mathbf{Q}^{(t)}$ ; that is,  $(\mathbf{Q}^{(t)})^\top \pmb{x}^{(t)} = \pmb{x}^{(t)}$ . Next, upon observing a utility  $\pmb{u}^{(t)}\in \mathbb{R}^{|A|}$ ,  $\mathfrak{R}_{swap}$  forwards to each individual regret minimizer  $\mathfrak{R}_a$  the utility  $\pmb{u}_a^{(t)}\coloneqq \pmb{u}^{(t)}\pmb{x}^{(t)}[a]\in \mathbb{R}^{|A|}$ . This construction is summarized in Algorithm 1.

# Algorithm 1: Blum and Mansour [2007]

Input: A set of external regret minimizers  $\{\Re_a\}_{a\in \mathcal{A}}$  , each for the simplex  $\Delta (\mathcal{A})$

```latex
1 function NEXTSTRATEGY()  
2  $\mathbf{Q}^{(t)}\gets \mathbf{0}\in \mathbb{R}^{|A|\times |A|}$   
3 for  $a\in \mathcal{A}$  do  
4  $\mid \mathbf{Q}^{(t)}[a,\cdot ]\gets \Re_{a}.\mathrm{NEXTSTRATEGY}()$   
5  $\pmb{x}^{(t)}\gets \mathrm{STATIONARYDISTRIBUTION}(\mathbf{Q}^{(t)})$   
6 return  $\pmb{x}^{(t)}$
```

```txt
7 function OBSERVEUTILITY  $(\pmb{u}^{(t)})$   
8 for  $a \in \mathcal{A}$  do  
9 |  $\Re_{a}$ . OBSERVEUTILITY  $(\pmb{x}^{(t)}[a]\pmb{u}^{(t)})$
```

Blum and Mansour [2007] showed that this algorithm guarantees no-swap-regret as long as each individual regret minimizer has sublinear external regret; this is formalized in the theorem below.

Theorem 4.1 (From External to Swap Regret [Blum and Mansour, 2007]). Let  $\mathrm{SwapReg}^T$  be the swap regret of  $\Re_{swap}$  and  $\mathrm{Reg}_a^T$  be the external regret of  $\Re_a$ , for each  $a \in \mathcal{A}$ , up to time  $T \in \mathbb{N}$ . Then,

$$
\operatorname {S w a p R e g} ^ {T} = \sum_ {a \in \mathcal {A}} \operatorname {R e g} _ {a} ^ {T}.
$$

In this context, we will instantiate each individual regret minimizer  $\Re_{a}$  with (OFTRL) under log-barrier regularization—and the same learning rate  $\eta > 0$ . We will refer to the resulting algorithm as BM-OFTRL-LogBar. A central ingredient in our proof of Theorem 1.1 is to establish that the resulting no-swap-regret algorithm  $\Re_{swap}$  will enjoy an RVU bound, as stated in Theorem 4.3. To this end, we first apply Corollary 3.2 for each individual regret minimizer  $\Re_{a}$ , implying that  $\mathrm{SwapReg}^T = \sum_{a \in \mathcal{A}} \mathrm{Reg}_a^T$  (by Theorem 4.1) is upper bounded as

$$
\begin{array}{l} \operatorname {S w a p R e g} ^ {T} \leq \frac {2 m ^ {2} \log T}{\eta} + 2 \eta \sum_ {a \in \mathcal {A}} \sum_ {t = 1} ^ {T} \| \boldsymbol {u} ^ {(t)} \boldsymbol {x} ^ {(t)} [ a ] - \boldsymbol {u} ^ {(t - 1)} \boldsymbol {x} ^ {(t - 1)} [ a ] \| _ {*}, \\ - \frac {1}{1 6 \eta} \sum_ {a \in \mathcal {A}} \sum_ {t = 1} ^ {T} \| \boldsymbol {x} _ {a} ^ {(t)} - \boldsymbol {x} _ {a} ^ {(t - 1)} \| _ {\boldsymbol {x} _ {a} ^ {(t - 1)}} ^ {2}. \tag {2} \\ \end{array}
$$

The  $\log T$  factor derives from the diameter of the log-barrier regularizer (see Theorem A.9), and appears to be unavoidable using our approach. Now the crux in establishing an RVU bound for  $\Re_{swap}$  is to upper bound the last term in (2) in terms of the "movement" of the stationary distribution. This is exactly where the local norm induced by the log-barrier turns out to be crucial, leading to the following key technical ingredient.

Lemma 4.2. Suppose that each regret minimizer  $\Re_{a}$  employs (OFTRL) with log-barrier regularization and  $\eta \leq \frac{1}{16}$ . Then, for any  $t \in \mathbb{N}$ ,

$$
\| \boldsymbol {x} ^ {(t)} - \boldsymbol {x} ^ {(t - 1)} \| _ {1} ^ {2} \leq 6 4 | \mathcal {A} | \sum_ {a \in \mathcal {A}} \| \boldsymbol {x} _ {a} ^ {(t)} - \boldsymbol {x} _ {a} ^ {(t - 1)} \| _ {\boldsymbol {x} _ {a} ^ {(t - 1)}} ^ {2}.
$$

Intuitively, this lemma ensures that the "movement" of the stationary distribution is smooth in terms of the "movement" of each row of the transition matrix  $\mathbf{Q}^{(t)}$ . To show this, we use the Markov chain tree theorem (Theorem C.3), which provides a closed-form combinatorial formula for the stationary distribution of an ergodic Markov chain, along with the fact that the log-barrier regularizer guarantees "multiplicative stability" of the iterates (Corollary C.1). While similar in spirit results have been documented in the literature for dynamics akin to MWU [Candogan et al., 2013, Chen and Peng, 2020], our proof of Lemma 4.2 crucially hinges on the local norm induced by the log-barrier regularizer. Thus, we are now ready to derive an RVU bound for swap regret.

Theorem 4.3 (RVU Bound for Swap Regret). Suppose that each  $\Re_{a}$  employs (OFTRL) with log-barrier regularization and  $\eta \leq \frac{1}{128\sqrt{m}}$ . Then, for  $T \geq 2$ , the swap regret of  $\Re_{swap}$  is bounded as

$$
\operatorname {S w a p R e g} ^ {T} \leq \frac {2 m ^ {2} \log T}{\eta} + 4 \eta \sum_ {t = 1} ^ {T} \| \boldsymbol {u} ^ {(t)} - \boldsymbol {u} ^ {(t - 1)} \| _ {\infty} ^ {2} - \frac {1}{2 0 4 8 m \eta} \sum_ {t = 1} ^ {T} \| \boldsymbol {x} ^ {(t)} - \boldsymbol {x} ^ {(t - 1)} \| _ {1} ^ {2}.
$$

This theorem follows directly from (2) and Lemma 4.2. So far we have focused on bounding the swap regret of each player when faced against arbitrary utilities. Next, we use Theorem 4.3 to establish a new fundamental property when all players employ the dynamics. Our proof crucially relies on the seemingly insignificant fact that  $\mathrm{SwapReg}_i^T\geq 0$  (recall Observation 2.1).

Theorem 4.4 (Log-Bounded Second-Order Path Lengths). Suppose that each player  $i \in [[n]]$  employs BM-OFTRL-LogBar with  $\eta_i = \frac{1}{128(n - 1)\sqrt{m_i}}$ . Then, for  $T \geq 2$ ,

$$
\sum_ {i = 1} ^ {n} \sum_ {t = 1} ^ {T} \| \pmb {x} _ {i} ^ {(t)} - \pmb {x} _ {i} ^ {(t - 1)} \| _ {1} ^ {2} \leq 8 1 9 2 \max _ {i \in [ [ n ] ]} \{\sqrt {m _ {i}} \} \sum_ {i = 1} ^ {n} m _ {i} ^ {5 / 2} \log T.
$$

Proof. Consider any player  $i \in [[n]]$ . Given that  $|u_i(\pmb{a})| \leq 1$ , for any  $\pmb{a} \in \mathcal{A}$  (by the normalization assumption), we have that for any  $t \in [[T]]$ ,

$$
\| \boldsymbol {u} _ {i} ^ {(t)} - \boldsymbol {u} _ {i} ^ {(t - 1)} \| _ {\infty} \leq \sum_ {\boldsymbol {a} _ {- i} \in \mathcal {A} _ {- i}} \left| \prod_ {j \neq i} \boldsymbol {x} _ {j} ^ {(t)} [ a _ {j} ] - \prod_ {j \neq i} \boldsymbol {x} _ {j} ^ {(t - 1)} [ a _ {j} ] \right| \leq \sum_ {j \neq i} \| \boldsymbol {x} _ {j} ^ {(t)} - \boldsymbol {x} _ {j} ^ {(t - 1)} \| _ {1},
$$

where we used that the total variation distance between two product distributions is bounded by the sum of the total variations of each individual marginal distribution [Hoeffding and Wolfowitz, 1958]. Thus,

$$
\left(\| \boldsymbol {u} _ {i} ^ {(t)} - \boldsymbol {u} _ {i} ^ {(t - 1)} \| _ {\infty}\right) ^ {2} \leq \left(\sum_ {j \neq i} \| \boldsymbol {x} _ {j} ^ {(t)} - \boldsymbol {x} _ {j} ^ {(t - 1)} \| _ {1}\right) ^ {2} \leq (n - 1) \sum_ {j \neq i} \| \boldsymbol {x} _ {j} ^ {(t)} - \boldsymbol {x} _ {j} ^ {(t - 1)} \| _ {1} ^ {2}.
$$

As a result, using Theorem 4.3 we conclude that  $\sum_{i=1}^{n} \operatorname{SwapReg}_i^T$  can be upper bounded by

$$
\begin{array}{l} 2 \log T \sum_ {i = 1} ^ {n} \frac {m _ {i} ^ {2}}{\eta_ {i}} + 4 (n - 1) \sum_ {i = 1} ^ {n} \eta_ {i} \sum_ {j \neq i} \sum_ {t = 1} ^ {T} \| \boldsymbol {x} _ {j} ^ {(t)} - \boldsymbol {x} _ {j} ^ {(t - 1)} \| _ {1} ^ {2} - \sum_ {i = 1} ^ {n} \frac {1}{2 0 4 8 m _ {i} \eta_ {i}} \sum_ {t = 1} ^ {T} \| \boldsymbol {x} _ {i} ^ {(t)} - \boldsymbol {x} _ {i} ^ {(t - 1)} \| _ {1} ^ {2} \\ = 2 \log T \sum_ {i = 1} ^ {n} \frac {m _ {i} ^ {2}}{\eta_ {i}} + \sum_ {i = 1} ^ {n} \left(4 \eta_ {i} (n - 1) ^ {2} - \frac {1}{2 0 4 8 m _ {i} \eta_ {i}}\right) \sum_ {t = 1} ^ {T} \| \boldsymbol {x} _ {i} ^ {(t)} - \boldsymbol {x} _ {i} ^ {(t - 1)} \| _ {1} ^ {2} \\ \leq 2 \log T \sum_ {i = 1} ^ {n} \frac {m _ {i} ^ {2}}{\eta_ {i}} - \frac {1}{4 0 9 6} \sum_ {i = 1} ^ {n} \frac {1}{m _ {i} \eta_ {i}} \sum_ {t = 1} ^ {T} \| \pmb {x} _ {i} ^ {(t)} - \pmb {x} _ {i} ^ {(t - 1)} \| _ {1} ^ {2}, \\ \end{array}
$$

since  $\eta_{i} = \frac{1}{128(n - 1)\sqrt{m_{i}}}$  for all  $i\in \llbracket n\rrbracket$ . But, given that  $0\leq \sum_{i = 1}^{n}\mathrm{SwapReg}_{i}^{T}$ , we conclude that

$$
\frac {1}{\max _ {i \in [ [ n ] ]} \{\sqrt {m _ {i}} \}} \sum_ {i = 1} ^ {n} \sum_ {t = 1} ^ {T} \| \boldsymbol {x} _ {i} ^ {(t)} - \boldsymbol {x} _ {i} ^ {(t - 1)} \| _ {1} ^ {2} \leq 8 1 9 2 \sum_ {i = 1} ^ {n} m _ {i} ^ {5 / 2} \log T.
$$

We are not aware of even  $o(T)$  bounds for the second-order path lengths in prior works (using a time-invariant learning rate), except in very restricted classes of games such as zero-sum and potential games [Anagnostides et al., 2022]. An example of the implication of Theorem 4.4 in a variant of Shapley's game [Shapley, 1964, Daskalakis et al., 2010] is illustrated in Figure 1. Although the dynamics appear to cycle, and the Nash gap—the maximum of the best response gaps—is always large, the players are changing their (mixed) strategies with gradually diminishing speed; further discussion and experiments are included in Appendix D.

![](images/b20b29031abd55b9257f165587d72e8ac5a380e4278a72b6a186b1766a8185c2.jpg)  
Figure 1: The trajectories of the BM-OFTRL-LogBar algorithm.

![](images/f78182bdbbf309ebb0d97d3d158f016c367c6151655e71ebcb102badab9144f7.jpg)

As an immediate consequence, combining Theorem 4.4 with Theorem 4.3 implies near-optimal individual swap regret.

Corollary 4.5 (Near-Optimal Individual Swap Regret). Suppose that all players use BM-OFTRL-LogBar with  $\eta_{i} = \frac{1}{128(n - 1)\sqrt{m_{i}}}$ . Then, the individual swap regret  $\mathrm{SwapReg}_i^T$  up to time  $T\geq 2$  of each player  $i\in [[n]]$  can be bounded as

$$
\operatorname {S w a p R e g} _ {i} ^ {T} \leq 2 5 6 \left((n - 1) m _ {i} ^ {5 / 2} + \frac {\max  _ {j \in [ n ]} \{\sqrt {m _ {j}} \}}{\sqrt {m _ {i}}} \sum_ {j = 1} ^ {n} m _ {j} ^ {5 / 2}\right) \log T.
$$

We point out that our distributed protocol makes the very mild assumption that each player knows an upper bound on the total number of players in order to appropriately tune the learning rate. Further, as is the case with the result in [Daskalakis et al., 2021], the individual regret of each player predicted by Corollary 4.5 grows linearly with the number of players. This can be unsatisfactory in games with a large number of players—i.e.,  $n \gg 1$ . For this reason, in Theorem C.4 we refine and improve the guarantee of Corollary 4.5 in games where the utility of each player depends only on a small number of other players, and each player's actions only affect a small number of others players; no other constraint is imposed on the game. Understanding whether the linear dependence on  $n$  is necessary to obtain near-optimal (swap) regret is left as an interesting question for future work.

Finally, we adapt the learning dynamics so that each player enjoys at the same time near-optimal swap regret in the adversarial regime as well.

Corollary 4.6 (Adversarial Robustness). There exist dynamics such that when all players follow them the individual swap regret of each player grows as in Corollary 4.5. Moreover, when faced against adversarial utilities, such that  $\| \pmb{u}_i^{(t)}\|_{\infty}\leq 1$  for all  $t\in [[T]]$ , the algorithm guarantees that

$$
\operatorname {S w a p R e g} _ {i} ^ {T} \leq 2 5 6 \left((n - 1) m _ {i} ^ {5 / 2} + \frac {\max  _ {j \in [   [ n ]   ]} \{\sqrt {m _ {j}} \}}{\sqrt {m _ {i}}} \sum_ {j = 1} ^ {n} m _ {j} ^ {5 / 2}\right) \log T + 2 \sqrt {m _ {i} \log m _ {i} T} + 2.
$$

Our adaptation is particularly natural: If all players follow the prescribed protocol, Theorem 4.4 implies that the observed utilities of each player  $i$  will be such that  $\sum_{\tau=1}^{t} \| \pmb{u}_i^{(\tau)} - \pmb{u}_i^{(\tau-1)} \|_{\infty} = O(\log t)$ . So, if at any time the player identifies that the previous condition was violated, it suffices to switch to a no-swap-regret minimizer (such as BM-MWU) tuned to face adversarial losses—in which case it is crucial to use a vanishing learning rate  $\eta = O(1 / \sqrt{T})$ .

Remark 4.7 (Numerical Precision). As is standard, we assumed that the iterates of (OFTRL) were computed exactly, without taking into account issues relating to numerical precision. To justify this, one can use Damped Newton's method in order to determine an  $\epsilon$ -nearby point to the optimal in  $O(\log \log (1 / \epsilon))$  iterations [Nemirovski and Todd, 2008]. This would extend all the regret bounds with up to an  $O(\epsilon T)$  error. So, with only  $O(\log \log T)$  repetitions of Damped Newton's method (per iteration) the error in the regret bounds becomes  $O(1)$ , and all of our guarantees immediately extend.

# 5 Discussion

Our main contribution in this paper was to establish a fundamental new property characterizing the trajectories of certain uncoupled no-regret learning dynamics, summarized in Theorem 1.1. This property directly guarantees the best known and near-optimal bound of  $O(\log T)$  for the swap regret incurred by each player in a general multiplayer game. Investigating further consequences of Theorem 1.1 is an interesting direction for the future. We also believe that our framework could have new implications for learning in games with partial information; e.g., see [Wei and Luo, 2018]. Another interesting avenue is to extend our scope to more general and combinatorial sets beyond the probability simplex, in order to (efficiently) encompass, for example, games in extensive form.

Further, our no-swap-regret learning dynamics have external regret trivially bounded according to Corollary 4.5. Consequently, our construction yields no-external-regret learning dynamics with a more favorable dependence on  $T$  compared to [Daskalakis et al., 2021] ( $\log T$  compared to the  $\log^4 (T)$  of the latter), but with a worse dependence on the number of actions (polynomial rather than logarithmic). Our method also has higher per-iteration complexity. For these reasons, extending the scope of our framework beyond self-concordant regularization is an important direction of future research.

# References

Julia Robinson. An iterative method of solving a game. Annals of Mathematics, 54:296-301, 1951.  
David Blackwell. An analog of the minmax theorem for vector payoffs. Pacific Journal of Mathematics, 6:1-8, 1956.  
Sergiu Hart and Andreu Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. *Econometrica*, 68:1127-1150, 2000.  
Dean Foster and Rakesh Vohra. Calibrated learning and correlated equilibrium. Games and Economic Behavior, 21:40-55, 1997.  
Constantinos Daskalakis, Alan Deckelbaum, and Anthony Kim. Near-optimal no-regret algorithms for zero-sum games. In Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), 2011.  
Nicolo Cesa-Bianchi and Gabor Lugosi. Prediction, learning, and games. Cambridge University Press, 2006.  
Alexander Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. In Advances in Neural Information Processing Systems, pages 3066-3074, 2013.  
Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E Schapire. Fast convergence of regularized learning in games. In Advances in Neural Information Processing Systems, pages 2989-2997, 2015.  
Xi Chen and Binghui Peng. Hedging in games: Faster convergence of external and swap regrets. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2020.  
Gabriele Farina, Christian Kroer, Noam Brown, and Tuomas Sandholm. Stable-predictive optimistic counterfactual regret minimization. In International Conference on Machine Learning (ICML), 2019.  
Constantinos Daskalakis, Maxwell Fishelson, and Noah Golowich. Near-optimal no-regret learning in general games. CoRR, abs/2108.06924, 2021.  
Ioannis Anagnostides, Constantinos Daskalakis, Gabriele Farina, Maxwell Fishelson, Noah Golowich, and Tuomas Sandholm. Near-optimal no-regret learning for correlated equilibria in multi-player general-sum games. CoRR, abs/2111.06008, 2021.  
Chen-Yu Wei and Haipeng Luo. More adaptive algorithms for adversarial bandits. In Conference On Learning Theory, COLT 2018, volume 75 of Proceedings of Machine Learning Research, pages 1263-1291. PMLR, 2018.  
Dylan J. Foster, Zhiyuan Li, Thodoris Lykouris, Karthik Sridharan, and Éva Tardos. Learning in games: Robustness of fast convergence. In Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, pages 4727-4735, 2016.  
Avrim Blum and Yishay Mansour. From external to internal regret. J. Mach. Learn. Res., 8: 1307-1324, 2007.  
Jacob Abernethy, Elad Hazan, and Alexander Rakhlin. Competing in the dark: An efficient algorithm for bandit linear optimization. In In Proceedings of the 21st Annual Conference on Learning Theory (COLT), 2008.  
Jason Millionis, Christos Papadimitriou, Georgios Piliouras, and Kelly Spendlove. Nash, conley, and computation: Impossibility and incompleteness in game dynamics, 2022.  
James P. Bailey and Georgios Piliouras. Fast and furious learning in zero-sum games: Vanishing regret with non-vanishing step sizes. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, pages 12977-12987, 2019.

Ioannis Anagnostides, Ioannis Panageas, Gabriele Farina, and Tuomas Sandholm. On last-iterate convergence beyond zero-sum games. arXiv preprint arXiv:2203.12056, 2022.  
Georgios Piliouras, Ryann Sim, and Stratis Skoulakis. Optimal no-regret learning in general games: Bounded regret with unbounded step-sizes via clairvoyant mwu. arXiv preprint arXiv:2111.14737, 2021.  
Amy Greenwald and Amir Jafari. A general class of no-regret learning algorithms and game-theoretic equilibria. In Conference on Learning Theory (COLT), Washington, D.C., 2003.  
Gilles Stoltz and Gábor Lugosi. Learning correlated equilibria in games with compact sets of strategies. Games Econ. Behav., 59(1):187-208, 2007.  
Geoffrey J Gordon, Amy Greenwald, and Casey Marks. No-regret learning in convex games. In Proceedings of the  $25^{th}$  international conference on Machine learning, pages 360-367. ACM, 2008.  
Robert Aumann. Subjectivity and correlation in randomized strategies. Journal of Mathematical Economics, 1:67-96, 1974.  
Tim Roughgarden. Intrinsic robustness of the price of anarchy. J. ACM, 62(5):32:1-32:42, 2015.  
Ozan Candogan, Asuman E. Ozdaglar, and Pablo A. Parrilo. Dynamics in near-potential games. Games Econ. Behav., 82:66-90, 2013.  
Wassily Hoeffding and J. Wolfowitz. Distinguishability of sets of distributions. The Annals of Mathematical Statistics, 29(3):700-718, 1958.  
Lloyd S Shapley. Some topics in two-person games. In M. Drescher, L. S. Shapley, and A. W. Tucker, editors, Advances in Game Theory. Princeton University Press, 1964.  
Constantinos Daskalakis, Rafael M. Frongillo, Christos H. Papadimitriou, George Pierrakos, and Gregory Valiant. On learning algorithms for nash equilibria. In Algorithmic Game Theory - Third International Symposium, SAGT 2010, volume 6386 of Lecture Notes in Computer Science, pages 114-125. Springer, 2010.  
Arkadi S Nemirovski and Michael J Todd. Interior-point methods for optimization. Acta Numerica, 17:191-234, 2008.  
Yurii Nesterov. Introductory Lectures on Convex Optimization: A Basic Course. Kluwer Academic Publishers, 2004.  
Arkadi Nemirovski. Interior point polynomial time methods in convex programming. Lecture notes, 42(16):3215-3224, 2004.  
V. Anantharam and P. Tsoucas. A proof of the markov chain tree theorem. Statistics & Probability Letters, 8(2):189-192, 1989.  
David Avis, Gabriel D. Rosenberg, Rahul Savani, and Bernhard von Stengel. Enumeration of nash equilibria for two-player games. *Economic Theory*, 42(1):9-37, 2010.
