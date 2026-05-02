# AN ONLINE LEARNING APPROACH TO GENERATIVE ADVERSARIAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We consider the problem of training generative models with a Generative Adversarial Network (GAN). Although GANs can accurately model complex distributions, they are known to be difficult to train due to instabilities caused by a difficult minimax optimization problem. In this paper, we view the problem of training GANs as finding a mixed strategy in a zero-sum game. Building on ideas from online learning we propose a novel training method named CHEKHOV GAN<sup>1</sup>. On the theory side, we show that our method provably converges to an equilibrium for semi-shallow GAN architectures, i.e. architectures where the discriminator is a one-layer network and the generator is arbitrary. On the practical side, we develop an efficient heuristic guided by our theoretical results, which we apply to commonly used deep GAN architectures. On several real-world tasks our approach exhibits improved stability and performance compared to standard GAN training.

# 1 INTRODUCTION

A recent trend in generative models is to use a deep neural network as a generator. Two notable approaches are variational auto-encoders (VAE) (Kingma & Welling, 2013; Rezende et al., 2014) as well as Generative Adversarial Networks (GAN) (Goodfellow et al., 2014). Unlike VAEs, the GAN approach offers a way to circumvent log-likelihood-based estimation and it also typically produces visually sharper samples (Goodfellow et al., 2014). The goal of the generator network is to generate samples that are indistinguishable from real samples, where indistinguishability is measured by an additional discriminative model. This creates an adversarial game setting where one pits a generator against a discriminator.

Let us denote the data distribution by  $p_{\mathrm{data}}(\mathbf{x})$  and the model distribution by  $p_{\mathbf{u}}(\mathbf{x})$ . A probabilistic discriminator is denoted by  $h_{\mathbf{v}}: \mathbf{x} \to [0;1]$  and a generator by  $G_{\mathbf{u}}: \mathbf{z} \to \mathbf{x}$ . The GAN objective is:

$$
\min  _ {\mathbf {u}} \max  _ {\mathbf {v}} M (\mathbf {u}, \mathbf {v}) = \frac {1}{2} \mathbb {E} _ {\mathbf {x} \sim p _ {\mathrm {d a t a}}} \log h _ {\mathbf {v}} (\mathbf {x}) + \frac {1}{2} \mathbb {E} _ {\mathbf {z} \sim p _ {\mathbf {z}}} \log \left(1 - h _ {\mathbf {v}} \left(G _ {\mathbf {u}} (\mathbf {z})\right)\right). \tag {1}
$$

Each of the two players (generator/discriminator) tries to optimize their own objective, which is exactly balanced by the loss of the other player, thus yielding a two-player zero-sum minimax game. Standard GAN approaches aim at finding a pure Nash Equilibrium by using traditional gradient-based techniques to minimize each player's cost in an alternating fashion. However, an update made by one player can repeatedly undo the progress made by the other one, without ever converging.

In general, alternating gradient descent fails to converge even for very simple games as shown by Salimans et al. (2016). In the setting of GANs, one of the central open issues is this nonconvergence problem, which in practice leads to oscillations between different kinds of generated samples (Metz et al., 2016).

While standard GAN methods seek to find pure minimax strategies, we propose to consider mixed strategies, which allows us to leverage online learning algorithms for mixed strategies in large games. Building on the approach of Freund & Schapire (1999), we propose a novel training algorithm for GANs that we call CHEKHOV GAN.

![](images/fd024157ef741ed3803d919a492b784bbfdc72ca820012997984a22f3088054b.jpg)  
(a)

![](images/b7e8ca1a90cc5814c07f50aa402e6dd9a12d50fa78e50291c6d50ca636bb1a54.jpg)  
(b)  
Figure 1: Three types of GAN architectures. Left: shallow. Middle: semi-shallow. Right: deep.

![](images/abfc261dbf79ba8aa31f3a55ad0632961c2f810d8006fc9f5a1403c1141fe51b.jpg)  
(c)

The standard GAN training method is not guaranteed to converge for general GAN architectures. Nevertheless, it does converge for shallow ones  $^2$ , i.e. for GAN architectures which consist of a single layer network as a discriminator, and a generator with one hidden layer (see Fig. 1(a)). Unfortunately, shallow GANs are very different from the deep GANs (Fig. 1(c)) which are used in practice, and ideally one would hope to understand models which are more similar to deep architectures.

In this paper we make a step forward by considering semi-shallow GANs, an intermediate architecture where the generator is any arbitrary network and the discriminator consists of a single layer (Fig. 1(b)). Our contributions are:

(1) We show that finding a Mixed Nash Equilibrium (MNE or simply equilibrium) gives rise to useful generators and discriminators.  
(2) We provide an algorithm that provably converges to an equilibrium for semi-shallow architectures.  
(3) Guided by our theoretical results we devise a new GAN training algorithm that is applicable to standard deep GAN architectures.

We first discuss the benefits of pursuing a mixed equilibrium. Based on results from game theory we show that by reaching an equilibrium we obtain useful (mixed) generator and discriminator. In the context of GANs, "usefulness" means that the mixed generator provided by the equilibrium solution will "fool" any adversary at least as good as any single generator (similar results hold for the discriminator).

On the theory side, we show that GANs with semi-shallow architectures induce semi-concave games, i.e., games which are concave with respect to the max player, but need not have a special structure with respect to the min player. Then we show that in such games players may efficiently invoke regret minimization procedures in order to find an equilibrium; this in turn gives rise to a way of finding an equilibrium in semi-shallow GANs. To the best of our knowledge, this result is novel in the context of GANs and might also find uses in other scenarios where such structure may arise. We would like to emphasize that this is a significant step from a theoretical point of view as the standard approach to training GANs is only known to be theoretically sound for convex-concave games, which correspond to shallow-networks (though it is still used heuristically to train deep GAN architectures).

On the practical side, we develop an efficient heuristic guided by our theoretical results, which we apply to commonly used deep GAN architectures. We provide experimental results demonstrating that our approach exhibits better empirical stability compared to the vanilla GAN and generates more diverse samples, while retaining the same level of visual quality.

In Section 2, we briefly review necessary notions from online learning and zero-sum games. We then present our approach and its theoretical guarantees in Section 3, and our practical algorithm is presented in Section 4. Lastly, we present empirical results on standard benchmarks in Section 5.

# 2 BACKGROUND & RELATED WORK

# 2.1 GANs

GAN Objectives: The classical way to learn a generative model consists of minimizing a divergence function between a parametrized model distribution  $p_{\mathbf{u}}(\mathbf{x})$  and the true data distribution  $p_{\mathrm{data}}(\mathbf{x})$ . The original GAN approach by Goodfellow et al. (2014) is for example known to be related to optimizing the Jensen-Shannon divergence. This was later generalized by Nowozin et al. (2016) who described a broader family of GAN objectives stemming from  $f$ -divergences. A different popular type of GAN objectives is the family of Integral Probability Metrics (Müller, 1997), such as the kernel MMD (Gretton et al., 2012; Li et al., 2015) or the Wasserstein metric (Arjovsky & Bottou, 2017). All of these divergence measures yield a minimax objective.

Training methods for GANs: In order to solve the minimax objective in Eq. 1, Goodfellow et al. (2014) suggested an approach that alternatively minimizes over  $\mathbf{u}$  and  $\mathbf{v}$  using mini-batch stochastic gradient descent. This approach can be shown to converge only when the updates are made in function space. In practice, this condition is not met - since this procedure works in the parameter space - and many issues arise during training (Arjovsky & Bottou, 2017; Radford et al., 2015), thus requiring careful initialization and proper regularization as well as other tricks (Metz et al., 2016; Pfau & Vinyals, 2016; Radford et al., 2015; Salimans et al., 2016). Even so, several problems are still commonly observed including a phenomena where the generator oscillates, without ever converging to a fixed point, or mode collapse when the generator maps many latent codes  $z$  to the same point, thus failing to produce diverse samples.

The closest work related to our approach is that of Arora et al. (2017) who showed the existence of an approximate mixed equilibrium with certain generalization properties; yet without providing a constructive way to find such equilibria. Instead, they advocate the use of mixed strategies, and suggest to do so by using the exponentiated gradient algorithm of Kivinen & Warmuth (1997). The work of Tolstikhin et al. (2017) also uses a similar mixture approach based on boosting. Other works have studied the problem of equilibrium and stabilization of GANs, often relying on the use of an auto-encoder as discriminator (Berthelot et al., 2017) or jointly with the GAN models (Che et al., 2016). In this work, we focus on providing convergence guarantees to a mixed equilibrium (definition in Section 3.2) using a technique from online optimization that relies on the players' past actions.

# 2.2 ONLINE LEARNING

Online learning is a sequential decision making framework in which a player aims at minimizing a cumulative loss function revealed to her sequentially. The source of the loss functions may be arbitrary or even adversarial, and the player seeks to provide worst case guarantees on her performance. Formally, this framework can be described as a repeated game of  $T$  rounds between a player  $\mathcal{P}_1$  and an adversary  $\mathcal{P}_2$ . At each round  $t \in [T]$ :

1. The player  $(\mathcal{P}_1)$  chooses a point  $\mathbf{u}_t \in \mathcal{K}$  according to some algorithm  $\mathcal{A}$  
2. The adversary  $(\mathcal{P}_2)$  chooses a loss function  $f_{t}\in \mathcal{F}$  
3. The player  $(\mathcal{P}_1)$  suffers a loss  $f_{t}(\mathbf{u}_{t})$ , and the loss function  $f_{t}(\cdot)$  is revealed to her.

The adversary is usually limited to choosing losses from a structured class of objectives  $\mathcal{F}$ , most commonly linear/convex losses. Also, the decision set  $\mathcal{K}$  is often assumed to be convex. The performance of the player's strategy is measured by the regret, defined as,

$$
\operatorname {R e g r e t} _ {T} ^ {\mathcal {A}} \left(f _ {1}, \dots , f _ {T}\right) = \sum_ {t = 1} ^ {T} f _ {t} \left(\mathbf {u} _ {t}\right) - \min  _ {\mathbf {u} ^ {*} \in \mathcal {K}} \sum_ {t = 1} ^ {T} f _ {t} \left(\mathbf {u} ^ {*}\right). \tag {2}
$$

Thus, the regret measures the cumulative loss of the player compared to the loss of the best fixed decision in hindsight. A player aims at minimizing her regret, and we are interested in no-regret strategies for which players ensure regret which is sublinear in  $T$  for any loss sequence<sup>3</sup>.

While there are several no-regret strategies, many of them may be seen as instantiations of the Follow-the-Regularized-Leader (FTRL) algorithm where

$$
\mathbf {u} _ {t} = \arg \min  _ {\mathbf {u} \in \mathcal {K}} \sum_ {\tau = 1} ^ {t - 1} f _ {\tau} (\mathbf {u}) + \eta_ {t} ^ {- 1} R (\mathbf {u}) \quad (\mathbf {F T R L}) \tag {3}
$$

FTRL takes the accumulated loss observed up to time  $t$  and then chooses the point in  $\mathcal{K}$  that minimizes the accumulated loss plus a regularization term  $\eta_t^{-1} R(\mathbf{u})$ . The regularization term prevents the player from abruptly changing her decisions between consecutive rounds<sup>4</sup>. This property is often crucial to obtaining no-regret guarantees. Note that FTRL is not always guaranteed to yield no-regret, and is mainly known to provide such guarantees in the setting where losses are linear/convex (Hazan et al., 2016; Shalev-Shwartz et al., 2012).

# 2.3 ZERO-SUM GAMES

Consider two players,  $\mathcal{P}_1, \mathcal{P}_2$ , which may choose pure decisions among the sets  $\mathcal{K}_1$  and  $\mathcal{K}_2$ , respectively. A zero-sum game is defined by a function  $M: \mathcal{K}_1 \times \mathcal{K}_2 \mapsto \mathbb{R}$  which sets the utilities of the players. Concretely, upon choosing a pure strategy  $(\mathbf{u}, \mathbf{v}) \in \mathcal{K}_1 \times \mathcal{K}_2$  the utility of  $\mathcal{P}_1$  is  $-M(\mathbf{u}, \mathbf{v})$ , while the utility of  $\mathcal{P}_2$  is  $M(\mathbf{u}, \mathbf{v})$ . The goal of either  $\mathcal{P}_1 / \mathcal{P}_2$  is to maximize their worst case utilities; thus,

$$
\min  _ {\mathbf {u} \in \mathcal {K} _ {1}} \max  _ {\mathbf {v} \in \mathcal {K} _ {2}} M (\mathbf {u}, \mathbf {v}) \quad (\mathbf {G o a l} \mathbf {o f} \mathcal {P} _ {1}), \quad \& \quad \max  _ {\mathbf {v} \in \mathcal {K} _ {2}} \min  _ {\mathbf {u} \in \mathcal {K} _ {1}} M (\mathbf {u}, \mathbf {v}) \quad (\mathbf {G o a l} \mathbf {o f} \mathcal {P} _ {2}) \tag {4}
$$

This definition of a game makes sense if there exists a point  $(\mathbf{u}^{*},\mathbf{v}^{*})$ , such that neither  $\mathcal{P}_1$  nor  $\mathcal{P}_2$  may increase their utility by unilateral deviation. Such a point  $(\mathbf{u}^{*},\mathbf{v}^{*})$  is called a Pure Nash Equilibrium, which is formally defined as a point which satisfies the following conditions:

$$
M (\mathbf {u} ^ {*}, \mathbf {v} ^ {*}) \leq \min  _ {\mathbf {u} \in \mathcal {K} _ {1}} M (\mathbf {u}, \mathbf {v} ^ {*}), \& M (\mathbf {u} ^ {*}, \mathbf {v} ^ {*}) \geq \max  _ {\mathbf {v} \in \mathcal {K} _ {2}} M (\mathbf {u} ^ {*}, \mathbf {v}).
$$

While a pure Nash equilibrium does not always exist, the pioneering work of Nash et al. (1950) established that there always exists a Mixed Nash Equilibrium (MNE or simply equilibrium), i.e., there always exist two distributions  $\mathcal{D}_1, \mathcal{D}_2$  such that,

$$
\mathbb {E} _ {(\mathbf {u}, \mathbf {v}) \sim \mathcal {D} _ {1} \times \mathcal {D} _ {2}} [ M (\mathbf {u}, \mathbf {v}) ] \leq \min  _ {\mathbf {u} \in \mathcal {K} _ {1}} \mathbb {E} _ {\mathbf {v} \sim \mathcal {D} _ {2}} [ M (\mathbf {u}, \mathbf {v}) ], \& \mathbb {E} _ {(\mathbf {u}, \mathbf {v}) \sim \mathcal {D} _ {1} \times \mathcal {D} _ {2}} [ M (\mathbf {u}, \mathbf {v}) ] \geq \max  _ {\mathbf {v} \in \mathcal {K} _ {2}} \mathbb {E} _ {\mathbf {u} \sim \mathcal {D} _ {1}} [ M (\mathbf {u}, \mathbf {v}) ].
$$

Finding an exact MNE might be computationally hard, and we are usually satisfied with finding an approximate MNE. This is defined below,

Definition 1. Let  $\varepsilon >0$ . Two distributions  $\mathcal{D}_1,\mathcal{D}_2$  are called  $\varepsilon$ -MNE if the following holds,

$$
\mathbb {E} _ {(\mathbf {u}, \mathbf {v}) \sim \mathcal {D} _ {1} \times \mathcal {D} _ {2}} [ M (\mathbf {u}, \mathbf {v}) ] \leq \min  _ {\mathbf {u} \in \mathcal {K} _ {1}} \mathbb {E} _ {\mathbf {v} \sim \mathcal {D} _ {2}} [ M (\mathbf {u}, \mathbf {v}) ] + \varepsilon ,
$$

$$
\mathbb {E} _ {(\mathbf {u}, \mathbf {v}) \sim \mathcal {D} _ {1} \times \mathcal {D} _ {2}} [ M (\mathbf {u}, \mathbf {v}) ] \geq \max _ {\mathbf {v} \in \mathcal {K} _ {2}} \mathbb {E} _ {\mathbf {u} \sim \mathcal {D} _ {1}} [ M (\mathbf {u}, \mathbf {v}) ] - \varepsilon .
$$

Terminology: In the sequel when we discuss zero-sum games, we shall sometimes use the GAN terminology, relating the min player  $\mathcal{P}_1$  as the generator, and the max player  $\mathcal{P}_2$ , as the discriminator.

No-Regret & Zero-sum Games: In zero-sum games, no-regret algorithms may be used to find an approximate MNE. Unfortunately, computationally tractable no-regret algorithms do not always exist. An exception is the setting when  $M$  is convex-concave. In this case, the players may invoke the powerful no-regret methods from online convex optimization to (approximately) solve the game. This seminal idea was introduced in Freund & Schapire (1999), where it was demonstrated how to invoke no-regret algorithms during  $T$  rounds to obtain an approximation guarantee of  $\varepsilon = O(1 / \sqrt{T})$  in zero-sum matrix games. This was later improved by Daskalakis et al. (2015), and Rakhlin & Sridharan (2013), demonstrating a guarantee of  $\varepsilon = O(\log T / T)$ . The result that we are about to present builds on the scheme of Freund & Schapire (1999).

# 3 FINDING EQUILIBRIUM IN GANS

Why Mixed Equilibrium? In this work, our ultimate goal is to efficiently find an approximate MNE for the game. However, in GANs, we are usually interested in designing good generators, and one might ask whether finding an equilibrium serves this cause better than solving the minimax problem, i.e., finding  $\mathbf{u} \in \operatorname{argmin}_{\mathbf{u} \in \mathcal{K}_1} \max_{\mathbf{v} \in \mathcal{K}_2} M(\mathbf{u}, \mathbf{v})$ . Interestingly, the minimax value of the equilibrium generator is always smaller than the minimax value of any pure strategy. Actually, the equilibrium strategy of the generator might be much better. This benefit of finding an equilibrium can be demonstrated on the following simple zero-sum game. Consider the following paper-rock-scissors game, i.e., a zero-sum game with the minimax objective

$$
\min  _ {i \in \{1, 2, 3 \}} \max  _ {j \in \{1, 2, 3 \}} M (i, j); \text {w h e r e} M = \left[ \begin{array}{c c c} 0 & - 1 & 1 \\ 1 & 0 & - 1 \\ - 1 & 1 & 0 \end{array} \right].
$$

Solving for the minimax objective yields a pure strategy with a minimax value of 1; conversely, the equilibrium strategy of the min player is a uniform distribution over actions; and its minimax value is 0. Thus, finding an equilibrium by allowing mixed strategies implies a smaller minimax value, and as we show in the Section 3.3 this is true in general. In the context of GANs, this result means that the mixed generator provided by the equilibrium solution will "fool" any adversary at least as good as any single generator. Similarly, the mixed discriminator provided by the equilibrium solution will discern any generator at least as good as any single discriminator.

The rest of this section presents a method that efficiently finds an equilibrium for semi-shallow GANs (see Fig. 1(b)). Such architectures do not induce a convex-concave game, and therefore the result of Freund & Schapire (1999) does not directly apply. Nevertheless, we show that semi-shallow GANs imply a game structure which gives rise to an efficient procedure for finding an equilibrium. In Sec. 3.1 we show that semi-shallow GANs define games with a property that we denote as semiconcave. Later, Sec. 3.2 provides an algorithm with provable guarantees for such games. Finally, in Section 3.3 we show that the minimax objective of the generator's equilibrium strategy is optimal with respect to the minimax objective.

# 3.1 SEMI-SHALLOW GANS

Semi-shallow GANs do not lead to a convex-concave game. Nonetheless, here we show that for an appropriate choice of the activation function they induce a game that is concave with respect to the discriminator. Later, in Sec. 3.2, we show that this property allows to efficiently find an equilibrium.

Proposition 1. Consider the GAN objective in Eq. (1) and assume that the discriminator is a single-layer network with a sigmoid activation function, meaning  $h_{\mathbf{v}}(\mathbf{x}) = 1 / (1 + \exp (-\mathbf{v}^{\top}\mathbf{x}))$ , where  $\mathbf{v} \in \mathbb{R}^n$ . Then the GAN objective is concave in  $\mathbf{v}$ .

Note that the above is not restricted to the sigmoid activation function, and it also holds for other choices of the activation function.

# 3.2 SEMI-CONCAVE ZERO-SUM GAMES

Here we discuss the setting of zero-sum games (see Eq. (4)) which are semi-concave. Formally a game,  $M$ , is semi-concave if for any fixed  $\mathbf{u}_0 \in \mathcal{K}_1$  the function  $g(\mathbf{v}) \coloneqq M(\mathbf{u}_0, \mathbf{v})$  is concave in  $\mathbf{v}$ . Algorithm 1 presents our method for semi-concave games. This algorithm is an instantiation of the scheme derived by Freund & Schapire (1999), with specific choices of the online algorithms  $\mathcal{A}_1, \mathcal{A}_2$ , used by the players. Note that both  $\mathcal{A}_1, \mathcal{A}_2$  are two different instances of the FTRL approach presented in Eq. (3).

Let us discuss Algorithm 1 and then present its guarantees. First note that each player calculates a sequence of  $T$  points based on an online algorithm  $\mathcal{A}_1 / \mathcal{A}_2$ . Interestingly, the sequence of (loss/reward) functions given to the online algorithm is based on the game objective  $M$ , and

Algorithm 1 CHEKHOV GAN

Input: #steps  $T$ , Game objective  $M(\cdot, \cdot)$

for  $t = 1\ldots T$  do

Calculate:

$$
\left(\operatorname {A l g}. \mathcal {A} _ {1}\right) \quad \mathbf {u} _ {t} \leftarrow \underset {\mathbf {u} \in \mathcal {K} _ {1}} {\operatorname {a r g m i n}} \sum_ {\tau = 0} ^ {t - 1} f _ {\tau} (\mathbf {u}) \quad \& \quad \left(\operatorname {A l g}. \mathcal {A} _ {2}\right) \quad \mathbf {v} _ {t} \leftarrow \underset {\mathbf {v} \in \mathcal {K} _ {2}} {\operatorname {a r g m a x}} \sum_ {\tau = 0} ^ {t - 1} \nabla g _ {\tau} \left(\mathbf {v} _ {\tau}\right) ^ {\top} \mathbf {v} - \frac {\sqrt {T}}{2 \eta_ {0}} \| \mathbf {v} \| ^ {2}
$$

Update:  $f_{t}(\cdot) = M(\cdot ,\mathbf{v}_{t})$  &  $g_{t}(\cdot) = M(\mathbf{u}_{t},\cdot)$

end for

Output mixed strategies:  $\mathcal{D}_1\sim \mathrm{Uni}\{\mathbf{u}_1,\ldots ,\mathbf{u}_T\} ,\mathcal{D}_2\sim \mathrm{Uni}\{\mathbf{v}_1,\ldots ,\mathbf{v}_T\} .$

also on the decisions made by the other player. For example, the loss sequence that  $\mathcal{P}_1$  receives is  $\{f_t(\mathbf{u})\coloneqq M(\mathbf{u},\mathbf{v}_t)\}_{t\in [T]}$ . After  $T$  rounds we end up with two mixed strategies  $\mathcal{D}_1,\mathcal{D}_2$ , each being a uniform distribution over the respective online decisions  $\{\mathbf{u}_t\}_{t\in [T]},\{\mathbf{v}_t\}_{t\in [T]}$ . Note that the first decision points  $\mathbf{u}_1,\mathbf{v}_1$  are set by  $\mathcal{A}_1,\mathcal{A}_2$  before encountering any (loss/reward) function, and the dummy functions  $f_0(\mathbf{u}) = 0,g_0(\mathbf{v}) = 0$  are only introduced in order to simplify the exposition. Since  $\mathcal{P}_1$  's goal is to minimize, it is natural to think of the  $f_{t}$ 's as loss functions, and measure the guarantees of  $\mathcal{A}_1$  according to the regret as defined in Equation (2). Analogously, since  $\mathcal{P}_2$  's goal is to maximize, it is natural to think of the  $g_{t}$ 's as reward functions, and measure the guarantees of  $\mathcal{A}_2$  according to the following appropriate definition of regret,  $\mathrm{Regret}_T^{\mathcal{A}_2} = \max_{\mathbf{v}^* \in \mathcal{K}_2}\sum_{t = 1}^{T}g_t(\mathbf{v}^*) - \sum_{t = 1}^{T}g_t(\mathbf{v}_t)$ .

The following theorem presents our guarantees for semi-concave games:

Theorem 1. Let  $\mathcal{K}_2$  be a convex set. Also, let  $M$  be a semi-concave zero-sum game, and assume  $M$  is  $L$ -Lipschitz continuous. Then upon invoking Alg. 1 for  $T$  steps it outputs mixed strategies  $(\mathcal{D}_1, \mathcal{D}_2)$  that are  $\varepsilon$ -MNE, where  $\varepsilon = O(1 / \sqrt{T})$ .

The most important point to note is that the accuracy of the approximation  $\varepsilon$  improves as the number of iterations  $T$  grows. This lets us obtain an arbitrarily good approximation for a large enough  $T$ . As mentioned before, both  $\mathcal{A}_1, \mathcal{A}_2$  are two different instances of the FTRL approach presented in Eq. (3). Concretely, Alg.  $\mathcal{A}_1$  is in fact follow-the-leader (FTL), i.e., FTRL without regularization. Alg.  $\mathcal{A}_2$  also uses the FTRL scheme. Yet, instead of the original reward functions,  $g_t(\cdot)$ , it utilizes linear approximations  $\tilde{g}_t(\mathbf{v}) = \nabla g_t(\mathbf{v}_t)^\top \mathbf{v}$ . Also note the use of the (minus) square  $\ell_2$  norm as regularization<sup>6</sup>. The  $\eta_0$  parameter depends on the Lipschitz constant of  $M$  as well as on the diameter of  $\mathcal{K}_2$  defined as,  $d_2 := \max_{\mathbf{v}_1, \mathbf{v}_2 \in \mathcal{K}_2} \| \mathbf{v}_1 - \mathbf{v}_2 \|$ . Concretely,  $\eta_0 = d_2 / \sqrt{2} L$ .

Next we provide a short proof sketch for Thm. 1. The full proof appears in Appendix A.

Proof sketch. The proof makes use of a theorem due to Freund & Schapire (1999) which shows that if both  $\mathcal{A}_1$  and  $\mathcal{A}_2$  ensure no-regret then it implies convergence to an approximate MNE. Since the game is concave with respect to  $\mathcal{P}_2$ , it is well known that the FTRL version  $\mathcal{A}_2$  appearing in Thm. 1 is a no-regret strategy (see e.g. Hazan et al. (2016)). The challenge is therefore to show that  $\mathcal{A}_1$  is also a no-regret strategy. This is non-trivial, especially for semi-concave games that do not necessarily have any special structure with respect to the generator<sup>7</sup>. However, the loss sequence received by the generator is not arbitrary, but rather it follows a special sequence based on the choices of the discriminator,  $\{f_t(\cdot) = M(\cdot, \mathbf{v}_t)\}_{t}$ . In the case of semi-concave games, the sequence of discriminator decisions,  $\{\mathbf{v}_t\}_{t}$  has a special property which "stabilizes" the loss sequence  $\{f_t\}_{t}$ , which in turn enables us to establish no-regret for  $\mathcal{A}_1$ .

Remark: Note that Alg.  $\mathcal{A}_1$  in Thm. 1 assumes the availability of an oracle that can efficiently find a global minimum for the FTL objective,  $\sum_{\tau=0}^{t-1} f_{\tau}(\mathbf{u})$ . This involves a minimization over a sum of generative networks. Therefore, our result may be seen as a reduction from the problem of finding

an equilibrium to an offline optimization problem. This reduction is not trivial, especially in light of the negative results of Hazan & Koren (2016), which imply that in the general case finding an equilibrium is hard, even with such an efficient offline optimization oracle at hand. Thus, our result enables to take advantage of progress made in supervised deep learning in order to efficiently find an equilibrium for GANs.

# 3.3 MINIMAX VALUE OF EQUILIBRIUM STRATEGY

In GANs we are mainly interested in ensuring the performance of the generator (resp. discriminator) with respect to the minimax (resp. maximin) objective. Let  $(\mathcal{D}_1, \mathcal{D}_2)$  be the pair of mixed strategies that Algorithm 1 outputs. Note that the minimax value of  $\mathcal{D}_1$  might be considerably smaller than the pure minimax value, as is shown in the example regarding the paper-rock-scissors game (see Sec. 3). The next lemma shows that the mixed strategy  $\mathcal{D}_1$  is always (approximately) better with respect to the pure minimax value (see proof in appendix B.2)

Lemma 1. The mixed strategy  $\mathcal{D}_1$  that Algorithm 1 outputs is  $\varepsilon$ -optimal with respect to the minimax value, i.e.,

$$
\max  _ {\mathbf {v} \in \mathcal {K} _ {2}} \mathbb {E} _ {\mathbf {u} \sim \mathcal {D} _ {1}} [ M (\mathbf {u}, \mathbf {v}) ] \leq \min  _ {\mathbf {u} \in \mathcal {K} _ {1}} \max  _ {\mathbf {v} \in \mathcal {K} _ {2}} M (\mathbf {u}, \mathbf {v}) + \varepsilon
$$

where  $\varepsilon$  here is equal to the one defined in Thm. 2.

Analogous result hold for  $\mathcal{D}_2$  with respect to the pure maximin objective.

# 4 PRACTICAL CHEKHOV GAN ALGORITHM FOR DEEP ARCHITECTURES

# Algorithm 2 Practical CHEKHOV GAN

Input: #steps  $T$ , Game objective  $M(\cdot, \cdot)$ , number of past states  $K$ , spacing  $m$

Initialize: Set loss/reward  $f_{0}(\cdot) = 0$ ,  $g_{0}(\cdot) = 0$ , initialize queues  $\mathcal{Q}_1$ .insert  $(f_0)$ ,  $\mathcal{Q}_2$ .insert  $(g_0)$

for  $t = 1\ldots T$  do

Update generator and discriminator based on a mini-batch of noise samples and data samples:

$$
\mathbf {u} _ {t + 1} \leftarrow \mathbf {u} _ {t} - \eta_ {t} \cdot \nabla_ {\mathbf {u} _ {t}} \left(\frac {1}{| \mathcal {Q} _ {1} |} \sum_ {f \in \mathcal {Q} _ {1}} f (\mathbf {u}) + \frac {C}{\sqrt {t}} \| \mathbf {u} \| ^ {2}\right) \& \mathbf {v} _ {t + 1} \leftarrow \mathbf {v} _ {t} - \eta_ {t} \cdot \nabla_ {\mathbf {v} _ {t}} \left(\frac {1}{| \mathcal {Q} _ {2} |} \sum_ {g \in \mathcal {Q} _ {2}} g (\mathbf {v}) - \frac {C}{\sqrt {t}} \| \mathbf {v} \| ^ {2}\right)
$$

Calculate:  $f_{t}(\cdot) = M(\cdot, \mathbf{v}_{t}) \& g_{t}(\cdot) = M(\mathbf{u}_{t}, \cdot)$

Update  $\mathcal{Q}_1$  and  $\mathcal{Q}_2$  (see main text or Algorithm 3 in the appendix)

end for

Output mixed strategies:  $\mathcal{D}_1\sim \mathrm{Uni}\{\mathbf{u}_1,\ldots ,\mathbf{u}_K\in \mathcal{Q}_1\}$ $\mathcal{D}_2\sim \mathrm{Uni}\{\mathbf{v}_1,\dots ,\mathbf{v}_K\in \mathcal{Q}_2\}$

In Section 3 we described a method (Alg. 1) which provably reaches an equilibrium for semi-shallow GANs. This method considers the whole history of generators and discriminators in making a decision at each round, which contrasts with the standard GAN training method that only considers the last generator and discriminator. Another difference is that our method outputs a mixed model (i.e., generator and discriminator) rather than a single model.

Building on the theoretical approach introduced in Section 3, we now present a practical method (Alg. 2) which can be efficiently applied to train common deep GAN architectures. Algorithm 2 combines the ideas of (a) considering the history of generators and discriminators at each update, and (b) outputting a mixed strategy, while only requiring an access to gradient information which can be efficiently obtained by running back-propagation. Next we discuss Alg. 2 in more details and highlight the differences compared to the theoretical approach:

(i) We use the FTRL objective (Eq. (3)) for both players. Note that Alg.  $\mathcal{A}_1$  appearing in Thm 1 uses FTRL with linear approximations, which is only appropriate for semi-concave games.  
(ii) As calculating the global minimizer of the FTRL objective is impractical, we instead update the weights based on the gradients of the FTRL objective. This can be done by using traditional optimization techniques such as SGD or Adam. Thus the update at each round depends on the

gradients of the past generators and discriminators. This differs from the standard GAN training which only employs the gradient of the last generator and discriminator.

(iii) The full FTRL algorithm requires saving the entire history of past generators/discriminators, which is computationally intractable. We find it sufficient to maintain a summary of the history using a small number of representative models. In order to capture a diverse subset of the history, we keep a queue  $\mathcal{Q}$  containing  $K \coloneqq |\mathcal{Q}|$  states (models). The spacing between consecutive models is determined by the following heuristic: every  $m$  update steps we remove the oldest model in the queue and add the current one. The number of steps between switches,  $m$ , can be set as a constant, but our experiments revealed it is more effective to keep  $m$  small at the beginning and increase its value as the number of rounds increases. We hypothesize that as the training progresses and the individual models become more discriminative, we should switch the models at a lower rate, keeping them more spaced out. The pseudo-code and a detailed description of the algorithm appears in the Appendix.

Intuition. In practice GANs commonly exhibit a non-convergent behavior. As a consequence, the generator oscillates between generating different modes from the target distribution. This is hypothesized to be due to the differences of the minimax and maximin solutions of the game (Goodfellow, 2016). If the order of the min and max operations switch, the minimization with respect to the generator's parameters is performed in the inner loop. This causes the generator to map every latent code to one or very few points for which the discriminator believes are likely. As simultaneous gradient descent updates do not clearly prioritize any specific ordering of minimax or maximin, in practice we often obtain results that resemble the latter.

In contrast, CHEKHOV GAN takes advantage of the history of the player's actions which yields better gradient information. Intuitively, the generator is updated such that it fools the past discriminators. In order to do so, the generator has to spread its mass more fairly according to the true data distribution. The discriminator can no longer simply learn to put low probability on the few modes of generated samples, which causes oscillations.

The mode collapse problems of GANs is also closely related to the phenomenon of catastrophic forgetting (Seff et al., 2017). When GANs are trained sequentially on samples coming from different modes, the discriminator tends to forget the previous modes it has learned about. This leads to having generated samples that focus only on the last or most prominent modes. By introducing a history of samples from previous generators, the discriminator is less likely to forget the part of the space that it has already learned.

Fig. 2 illustrates the mode collapse problem. The data consists of a mixture of 7 Gaussians with different sampling probabilities whose centers are aligned in a circle. As two modes have higher probabilities and are seen more frequently, they attract the gradients towards them and cause mode collapse and forgetting. Chekhov GAN manages to recover the true data distribution in this case as well, unlike vanilla GANs.

![](images/8f9b4ba0e41908ecb832e355ee963e6b4c82529e0e86465760ab109377017cb1.jpg)  
GAN  
CHEKHOV GAN  
Figure 2: Mode Collapse on a Gaussian Mixture. We show heat maps of the generator distribution over time, as well as the target data distribution in the last column. Standard GAN updates (top row) cause mode collapse, whereas CHEKHOV GAN using  $K = 10$  past steps (bottom row) spreads its mass over all the modes of the target distribution.

![](images/938d2877fdea4266f9001c92cb7c35a58f5100cc54ad5dc6624c19edf5e37dab.jpg)

![](images/d05f009a7e295b126f4bf81064ff16d821d203323c0c1a3a47f18274ad2fb31b.jpg)

# 5 EXPERIMENTAL RESULTS

We now demonstrate that CHEKHOV GAN yields improved stability and sample diversity. To do so, we use a comparable number of datasets and baselines as standard GAN approaches, e.g. Metz et al.

(2016); Arjovsky & Bottou (2017). We test our method on models where the traditional GAN training has difficulties converging and engages in a behavior of mode collapse. We also perform experiments on harder tasks using the DCGAN architecture (Radford et al., 2015) which is commonly used in the literature. Note that the DCGAN architecture, when trained using standard techniques, still suffers from instabilities and mode collapse (Nagarajan & Kolter, 2017; Roth et al., 2017). We here demonstrate that CHEKHOV GAN reduces mode dropping while retaining high visual sample quality. For all of the experiments, we generate from the newest generator only. Experimental details and comparisons to additional baselines, as well as a set of recommended hyperparameters are available in Appendix D and Appendix C, respectively.

# 5.1 NON-CONVERGENCE AND MODE DROPPING

# 5.1.1 AUGMENTED MNIST

We first evaluate the ability of our approach to avoid mode collapse on real image data coming from an augmented version of the MNIST dataset. Similarly to (Metz et al., 2016; Che et al., 2016), we combine three randomly selected MNIST digits to form 3-channel images, resulting in a dataset with 1000 different classes, one for each of the possible combinations of the ten MNIST digits.

We train a simplified DCGAN architecture (see details in Appendix D) with both GAN and CHEKHOV GAN with a different number of saved past states. The evaluation of each model is done as follows. We generate a fixed amount of samples (25,600) from each model and classify them using a pretrained MNIST classifier with an accuracy of  $99.99\%$ . The models that exhibit less mode collapse are expected to generate samples from most of the 1000 modes.

We report two different evaluation metrics in Table 1: i) the number of classes for which a model generated at least one sample, and ii) the reverse KL divergence. The reverse KL divergence between the model and the target data distribution is computed by considering that the data distribution is a uniform distribution over all classes.

<table><tr><td>Models</td><td>0 states (GAN)</td><td>5 states</td><td>10 states</td></tr><tr><td>Generated Classes</td><td>629 ± 121.08</td><td>743 ± 64.31</td><td>795 ± 37</td></tr><tr><td>Reverse KL</td><td>1.96 ± 0.64</td><td>1.40 ± 0.21</td><td>1.24 ± 0.17</td></tr></table>

Table 1: Stacked MNIST: Number of generated classes out of 1000 possible combinations, and the reverse KL divergence score. The results are averaged over 10 runs.

# 5.2 IMAGE MODELING

We turn to the evaluation of our model for the task of generating rich image data for which the modes of the data distribution are unknown. In the following, we perform experiments that indirectly measure mode coverage through metrics based on the sample diversity and quality.

# 5.2.1 INFERENCE VIA OPTIMIZATION ON CIFAR10

We train a DCGAN architecture on CIFAR10 (Krizhevsky & Hinton, 2009) and evaluate the performance of each model using the inference via optimization technique introduced in (Metz et al., 2016) and explained in Appendix D.3.3.

The average MSE over 10 rounds using different seeds is reported in Table 2. Using CHEKHOV GAN with as few as 5 past states results in a significant gain which can be further improved by increasing the number of past states to 10 and 25. In addition, the training procedure becomes more stable as indicated by the decrease in the standard deviation. The percentage of mini-batches that achieve the lowest reconstruction loss with the different models is given in Table 2. This can also be visualized by comparing the closest images  $x_{\text{closest}}$  from each model to real target images  $x_{\text{target}}$  as shown in Figure 3. The images are randomly selected images from the batch which has the largest absolute difference in MSE between GAN and CHEKHOV GAN with 25 states. The samples obtained by the original GAN are often blurry while samples from CHEKHOV GAN are both sharper and exhibit more variety, suggesting a better coverage of the true data distribution.

<table><tr><td>Target</td><td>Past States</td><td>0 (GAN)</td><td>5 states</td><td>10 states</td><td>25 states</td></tr><tr><td rowspan="2">Train Set</td><td>MSE</td><td>61.13 ± 3.99</td><td>58.84 ± 3.67</td><td>56.99 ± 3.49</td><td>48.42 ± 2.99</td></tr><tr><td>Best Rank (%)</td><td>0 %</td><td>0 %</td><td>18.66 %</td><td>81.33 %</td></tr><tr><td rowspan="2">Test Set</td><td>MSE</td><td>59.5 ± 3.65</td><td>56.66 ± 3.60</td><td>53.75 ± 3.47</td><td>46.82 ± 2.96</td></tr><tr><td>Best Rank (%)</td><td>0 %</td><td>0 %</td><td>17.57 %</td><td>82.43 %</td></tr></table>

Table 2: CIFAR10: MSE between target images from the train and test set and the best rank which consists of the percentage of minibatches containing target images that can be reconstructed with the lowest loss across various models. We use 20 different minibatches, each containing 64 target images. Increasing the number of past states for CHEKHOV GAN allows the model to better match the real images.  

<table><tr><td>Real Image</td><td>0 states</td><td>10 states</td><td>25 states</td></tr><tr><td></td><td>4.14</td><td>2.37</td><td>2.19</td></tr><tr><td></td><td>4.17</td><td>2.97</td><td>2.58</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td>Real Image</td><td>0 states</td><td>10 states</td><td>25 states</td></tr><tr><td></td><td>3.06</td><td>2.51</td><td>2.35</td></tr><tr><td></td><td>1.12</td><td>0.30</td><td>0.39</td></tr></table>

Note that the numbers quoted in our paper can directly be compared to the ones reported in unrolled GAN (Metz et al., 2016) since we have used the same architecture and choice of hyper-parameters. We include a comparison in the appendix.

# 5.2.2 ESTIMATION OF MISSING MODES ON CELEBA

We estimate the number of missing modes on the CelebA dataset (Liu et al., 2015) by using an auxiliary discriminator as performed in (Che et al., 2016). The experiment consists of two phases. In the first phase we train GAN and CHEKHOV GAN models and generate a fixed number of images. In the second phase we independently train a noisy discriminator using the DCGAN architecture where the training data is the previously generated data from each of the models, respectively. The noisy discriminator is then used as a mode estimator. Test images from CelebA are provided to the mode estimator and the number of images that are classified as fake can be viewed as images on a missing mode. Table 4 showcases number of missed modes for the two models. Generated samples from each model are given in the Appendix.

Table 3: CIFAR10: Target images from the test set are shown on the left. The images from each model that best resemble the target image are shown for different number of past states: 0 (GAN), 10 and 25 (CHEKHOV GAN). The reconstruction MSE loss is indicated above each image.  

<table><tr><td>σ</td><td>0 states (GAN)</td><td>5 states (CHEKHOV GAN )</td></tr><tr><td>0.25</td><td>3004 ± 4154</td><td>1407 ± 1848</td></tr><tr><td>0.5</td><td>2568.25 ± 4148</td><td>1007 ± 1805</td></tr></table>

Table 4: CelebA: Number of images from the test set that the auxiliary discriminator classifies as not real. Gaussian noise with variance  $\sigma^2$  is added to the input of the auxiliary discriminator, with the standard deviation shown in the first row. The test set consists of 50,000 images.

Interestingly, even with small number of past states  $(\mathrm{K} = 5)$ , CHEKHOV GAN manages to stabilize the training and generate more diverse samples on all the datasets. In terms of computational complexity, our algorithm scales linearly with  $K$ . However, all the elements in the sum are independent and can be computed efficiently in a parallel manner.

# 6 CONCLUSION

We have presented a principled approach to training GANs, which is guaranteed to reach convergence to a mixed equilibrium for semi-shallow architectures. Empirically, our approach presents several advantages when applied to commonly used GAN architectures, such as improved stability or

reduction in mode dropping. Our results open an avenue for the use of online-learning and game-theoretic techniques in the context of training GANs. One question that remains open is whether the theoretical guarantees can be extended to more complex architectures.

# REFERENCES

Martin Arjovsky and Léon Bottou. Towards principled methods for training generative adversarial networks. In NIPS 2016 Workshop on Adversarial Training. In review for ICLR, volume 2016, 2017.  
Sanjeev Arora, Rong Ge, Yingyu Liang, Tengyu Ma, and Yi Zhang. Generalization and equilibrium in generative adversarial nets (gans). arXiv preprint arXiv:1703.00573, 2017.  
David Berthelot, Tom Schumm, and Luke Metz. Began: Boundary equilibrium generative adversarial networks. arXiv preprint arXiv:1703.10717, 2017.  
Tong Che, Yanran Li, Athul Paul Jacob, Yoshua Bengio, and Wenjie Li. Mode regularized generative adversarial networks. arXiv preprint arXiv:1612.02136, 2016.  
Constantinos Daskalakis, Alan Deckelbaum, and Anthony Kim. Near-optimal no-regret algorithms for zero-sum games. Games and Economic Behavior, 92:327-348, 2015.  
Yoav Freund and Robert E Schapire. Adaptive game playing using multiplicative weights. Games and Economic Behavior, 29(1-2):79-103, 1999.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Aistats, volume 9, pp. 249-256, 2010.  
Ian Goodfellow. Nips 2016 tutorial: Generative adversarial networks. arXiv preprint arXiv:1701.00160, 2016.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Nets. pp. 2672-2680, 2014.  
Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 13(Mar):723-773, 2012.  
Elad Hazan and Tomer Koren. The computational power of optimization in online learning. In Proc. STOC, pp. 128-141. ACM, 2016.  
Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends® in Optimization, 2(3-4):157-325, 2016.  
Adam Kalai and Santosh Vempala. Efficient algorithms for online decision problems. Journal of Computer and System Sciences, 71(3):291-307, 2005.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Diederik P Kingma and Max Welling. Auto-Encoding Variational Bayes. arXiv.org, December 2013.  
Jyrki Kivinen and Manfred K Warmuth. Exponentiated gradient versus gradient descent for linear predictors. Information and Computation, 132(1):1-63, 1997.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.  
Yujia Li, Kevin Swersky, and Richard S Zemel. Generative moment matching networks. In ICML, pp. 1718-1727, 2015.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision, pp. 3730-3738, 2015.  
Peter McCullagh and James A Nelder. Generalized linear models, no. 37 in monograph on statistics and applied probability, 1989.

Luke Metz, Ben Poole, David Pfau, and Jascha Sohl-Dickstein. Unrolled generative adversarial networks. arXiv preprint arXiv:1611.02163, 2016.  
Alfred Müller. Integral probability metrics and their generating classes of functions. Advances in Applied Probability, 29(02):429-443, 1997.  
Vaishnavh Nagarajan and J Zico Kolter. Gradient descent gan optimization is locally stable. arXiv preprint arXiv:1706.04156, 2017.  
John F Nash et al. Equilibrium points in n-person games. Proceedings of the national academy of sciences, 36(1):48-49, 1950.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-gan: Training generative neural samplers using variational divergence minimization. In Advances in Neural Information Processing Systems, pp. 271-279, 2016.  
David Pfau and Oriol Vinyals. Connecting generative adversarial networks and actor-critic methods. arXiv preprint arXiv:1610.01945, 2016.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
Sasha Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. In Advances in Neural Information Processing Systems, pp. 3066-3074, 2013.  
D J Rezende, S Mohamed, and D Wierstra. Stochastic backpropagation and approximate inference in deep generative models. arXiv.org, 2014.  
Kevin Roth, Aurelien Lucchi, Sebastian Nowozin, and Thomas Hofmann. Stabilizing training of generative adversarial networks through regularization. arXiv preprint arXiv:1705.09367, 2017.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, pp. 2226-2234, 2016.  
Ari Seff, Alex Beatson, Daniel Suo, and Han Liu. Continual learning in generative adversarial nets. arXiv preprint arXiv:1705.08395, 2017.  
Shai Shalev-Shwartz et al. Online learning and online convex optimization. Foundations and Trends® in Machine Learning, 4(2):107-194, 2012.  
Ilya Tolstikhin, Sylvain Gelly, Olivier Bousquet, Carl-Johann Simon-Gabriel, and Bernhard Scholkopf. Adagan: Boosting generative models. arXiv preprint arXiv:1701.02386, 2017.
