# DISTRIBUTIONAL BELLMAN OPERATORS OVER MEAN EMBEDDINGS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose a novel algorithmic framework for distributional reinforcement learning, based on learning finite-dimensional mean embeddings of return distributions. We derive several new algorithms for dynamic programming and temporal-difference learning based on this framework, provide asymptotic convergence theory, and examine the empirical performance of the algorithms on a suite of tabular tasks. Further, we show that this approach can be straightforwardly combined with deep reinforcement learning, and obtain a new deep RL agent that improves over baseline distributional approaches on the Arcade Learning Environment.

# 1 INTRODUCTION

In distributional approaches to reinforcement learning (RL), the aim is to learn the full probability distribution of future returns (Morimura et al., 2010a; Bellemare et al., 2017; 2023), rather than just their expected value, as is typically the case in value-based reinforcement learning (Sutton & Barto, 2018). Distributional RL was proposed in the setting of deep reinforcement learning by Bellemare et al. (2017), with a variety of precursor work stretching back almost as far as Markov decision processes themselves (Jaquette, 1973; Sobel, 1982; Chung & Sobel, 1987; Morimura et al., 2010a;b). Beginning with the work in Bellemare et al. (2017), the distributional approach to reinforcement learning has been central across a variety of applications of deep RL in simulation and in the real world (Bodnar et al., 2020; Bellemare et al., 2020; Wurman et al., 2022; Fawzi et al., 2022).

Typically, predictions of return distributions are represented directly as approximate probability distributions, such as categorical distributions (Bellemare et al., 2017). Rowland et al. (2019) proposed an alternative framework where return distributions are represented via the values of statistical functionals, called a sketch by Bellemare et al. (2023). This provided a new space of distributional reinforcement learning algorithms, leading to improvements in deep RL agents, and hypotheses regarding distributional RL in the brain (Dabney et al., 2020; Lowet et al., 2020). On the other hand, a potential drawback of this approach is that each distributional Bellman update to the representation, these values must be "decoded" back into an approximate distribution via an imputation strategy. In practice, this can introduce significant computational overhead to Bellman updates, and is unlikely to be biologically plausible for distributional learning in the brain (Tano et al., 2020).

Here, we focus on a notable instance of the sketch called the mean embedding sketch. In short, the mean embedding is the expectation of nonlinear functions under the distribution represented (Smola et al., 2007; Striperumbudur et al., 2010; Berlinet & Thomas-Agnan, 2011), and is related to frames in signal processing (Mallat, 1999) and distributed distributional code in neuroscience (Sahani & Dayan, 2003; Vertes & Sahani, 2018). The core contributions of this paper are to revisit the approach to distributional reinforcement learning based on sketches (Rowland et al., 2019), and to propose the sketch Bellman operator that updates the implicit distributional representation as a simple linear operation, obviating the need for the expensive imputation strategies converting between sketches and distributions. This provides a rich new space of distributional RL algorithms that operate entirely in the space of sketches. We provide theoretical convergence analysis to accompany the framework, investigate the practical behaviour of various instantiations of the proposed algorithms in tabular domains, and demonstrate the effectiveness of the sketch framework in deep reinforcement learning, showing that our approach is robust enough to serve as the basis for a new variety of deep distributional reinforcement learning algorithms.

# 2 BACKGROUND

We consider a Markov decision process (MDP) with state space  $\mathcal{X}$ , action space  $\mathcal{A}$ , transition probabilities  $P: \mathcal{X} \times \mathcal{A} \to \mathcal{P}(\mathcal{X})$ , reward distribution function  $P_R: \mathcal{X} \times \mathcal{A} \to \mathcal{P}(\mathbb{R})$ , and discount factor  $\gamma \in [0,1)$ . Given a policy  $\pi: \mathcal{X} \to \mathcal{P}(\mathcal{A})$  and initial state  $x \in \mathcal{X}$ , a random trajectory  $(X_t, A_t, R_t)_{t \geq 0}$  is the sequence of random states, actions, and rewards encountered when using the policy  $\pi$  to select actions in this MDP. More precisely, we have  $X_0 = x$ ,  $A_t \sim \pi(\cdot | X_t)$ ,  $R_t \sim P_R(X_t, A_t)$ ,  $X_{t+1} \sim P(\cdot | X_t, A_t)$  for all  $t \geq 0$ . We write  $\mathbb{P}_x^\pi$  and  $\mathbb{E}_x^\pi$  for probabilities and expectations with respect to this distribution, respectively. The performance along the trajectory is measured by the discounted return, defined by

$$
\sum_ {t = 0} ^ {\infty} \gamma^ {t} R _ {t}. \tag {1}
$$

In typical value-based reinforcement learning, during policy evaluation, the agent learns the expectation of the return for each possible initial state  $x \in \mathcal{X}$ , which is encoded by the value function  $V^{\pi}: \mathcal{X} \to \mathbb{R}$ , given by  $V^{\pi}(x) = \mathbb{E}_x^{\pi}[\sum_{t=0}^{\infty} \gamma^t R_t]$ .

# 2.1 DISTRIBUTIONAL RL AND THE DISTRIBUTIONAL BELLMAN EQUATION

In distributional reinforcement learning, the problem of policy evaluation is to learn the probability distribution of return in Equation (1) for each possible initial state  $x \in \mathcal{X}$ . This is encoded by the return-distribution function  $\eta^{\pi}: \mathcal{X} \to \mathcal{P}(\mathbb{R})$ , which maps each initial state  $x \in \mathcal{X}$  to the corresponding distribution of the random return, e.g.  $\eta^{\pi}(x)$  is the return distribution of state  $x$ . A central result in distributional reinforcement learning is the distributional Bellman equation, which relates the distribution of the random return under different combinations of initial states and actions.

To build the random variable formulation of the returns, we let  $(G^{\pi}(x): x \in \mathcal{X})$  be a collection of random variables with the property that  $G^{\pi}(x)$  is equal to Equation (1) in distribution, conditioned on the initial state  $X_0 = x$ . This formulation implies that the random variable  $G^{\pi}(x)$  is distributed as  $\eta^{\pi}(x)$ , introduced above, for all  $x \in \mathcal{X}$ . Consider a random transition  $(x, R, X')$  generated by  $\pi$ , independent of the  $G^{\pi}$  random variables. Then, the (random variable) distributional Bellman equation states that for each initial state  $x$ ,

$$
G ^ {\pi} (x) \stackrel {{\mathcal {D}}} {=} R + \gamma G ^ {\pi} \left(X ^ {\prime}\right) \quad | X = x.
$$

Here, we use the slight abuse of the conditioning bar to set the distribution of  $X$  in the random transition. It is also useful to introduce the distributional Bellman operator  $\mathcal{T}^{\pi}:\mathcal{P}(\mathbb{R})^{\mathcal{X}}\to \mathcal{P}(\mathbb{R})^{\mathcal{X}}$  to describe the transformation that occurs on the right-hans side (Morimura et al., 2010a; Bellemare et al., 2017). If  $\eta \in \mathcal{P}(\mathbb{R})^{\mathcal{X}}$  is a collection of probability distributions, and  $(G(x):x\in \mathcal{X})$  is a collection of random variables such that  $G(x)\sim \eta (x)$  for all  $x$ , and  $(X,R,X^{\prime})$  is random transition generated by  $\pi$ , independent of  $(G(x):x\in \mathcal{X})$ , then  $(\mathcal{T}^{\pi}\eta)(x) = \mathrm{Dist}(R + \gamma G(X^{\prime})|X = x)$ .

To implement algorithms of distributional RL, one needs to approximate the infinite-dimensional return-distribution function  $\eta^{\pi}$  with finite-dimensional representations. This is typically done via direct approximations in the space of distributions; see e.g. Bellemare et al. (Chapter 5; 2023).

# 2.2 STATISTICAL FUNCTIONALS AND SKETCHES

Rather than using approximations in the space of distributions, Rowland et al. (2019) proposed to represent return distributions indirectly via functionals of the return distribution, called sketches by Bellemare et al. (2023). In this work we consider a specific class of sketches, defined below.

Definition 2.1 (Mean embedding sketches). A mean embedding sketch  $\psi$  is specified by a function  $\phi : \mathbb{R} \to \mathbb{R}^m$ , and defined by

$$
\psi (\nu) := \mathbb {E} _ {Z \sim \nu} [ \phi (Z) ]. \tag {2}
$$

For a given distribution  $\nu$ , the embedding  $\psi(\nu)$  can therefore be thought of as providing a lossy summary of the distribution. The name here is motivated by the kernel literature, in which Equation (2) can be viewed as embedding the distribution  $\nu$  into  $\mathbb{R}^m$  based on the mean value of  $\phi$  under this distribution (Smola et al., 2007; Sriperumbudur et al., 2010; Berlinet & Thomas-Agnan, 2011).

Statistical functional dynamic programming and temporal-difference learning (SFDP/SFTD; Rowland et al. (2019), see also Bellemare et al. (2023)) is an approach to distributional RL in which sketch values, rather than approximate distributions, are the primary object learned. Given a sketch  $\psi$  and estimated sketch values  $U: \mathcal{X} \to \mathbb{R}^m$ , these approaches proceed by first defining an imputation strategy  $\iota: \mathbb{R}^m \to \mathcal{P}(\mathbb{R})$  mapping sketch values back to distributions, with the aim that  $\psi(\iota(U)) \approx U$ , so that  $\iota$  behaves as an approximate pseudo-inverse to  $\psi$ . The usual Bellman backup is then applied to this imputed distribution, and the sketch value extracted from this updated distribution. Thus, if  $U: \mathcal{X} \to \mathbb{R}^m$  represents approximations to sketch values, a typical update in SFDP takes the form  $U \gets \psi((\mathcal{T}^\pi \iota(U))(x))$  (Bellemare et al., 2023); see Figure 1.

![](images/51600d14ff09f87c3ec2777e6e037519421c4644022ebad3cdcdef4c1b334950.jpg)  
Figure 1: The statistical functional framework proposed by Rowland et al. (2019) (top), and the framework proposed in this paper, directly updating sketches, avoiding the imputation step (bottom).

This approach led to expectile-regression DQN, a deep RL agent that aims to learn the sketch values associated with certain expectiles (Newey & Powell, 1987) of the return, and influenced a distributional model of dopamine signalling in the brain (Dabney et al., 2020). An important consideration is that computation of the imputation strategy is often costly in machine learning applications, and considered biologically implausible in neuroscience models (Tano et al., 2020).

# 3 THE BELLMAN SKETCH FRAMEWORK

Our goal is to derive a framework for approximate computation of the sketch  $\psi$  (with corresponding feature function  $\phi$ ) of the return distributions corresponding to a policy  $\pi$ , without needing to design, implement, and compute an imputation strategy as in the case of SFDP/TD; see Figure 1 for a visual comparison of the two approaches. That is, we aim to compute the object  $U^{\pi}:\mathcal{X}\to \mathbb{R}^{m}$ , given by

$$
U ^ {\pi} (x) := \psi (\eta^ {\pi} (x)) = \mathbb {E} _ {x} ^ {\pi} [ \phi (\sum_ {t = 0} ^ {\infty} \gamma^ {t} R _ {t}) ].
$$

We begin by considering environments with a finite set of possible rewards  $\mathcal{R} \subseteq \mathbb{R}$ ; we discuss generalisations later. To motivate our method, we first consider a special case; suppose that for each possible return  $g \in \mathbb{R}$ , and each possible immediate reward  $r \in \mathcal{R}$ , there exists a matrix  $B_r$  such that

$$
\phi (r + \gamma g) = B _ {r} \phi (g); \tag {3}
$$

note that  $B_r$  does not depend on  $g$ , and  $\gamma$  is a constant. In words, this says that the feature function  $\phi$  evaluated at the bootstrap return  $r + \gamma g$  is expressible as a linear transformation of the feature function evaluated at  $g$  itself. If such a relationship holds, then we have

$$
U ^ {\pi} (x) \stackrel {(a)} {=} \mathbb {E} _ {x} ^ {\pi} [ \phi (R + \gamma G ^ {\pi} (X ^ {\prime})) ] \stackrel {(b)} {=} \mathbb {E} _ {x} ^ {\pi} \left[ B _ {R} \phi \left(G ^ {\pi} \left(X ^ {\prime}\right)\right) \right] \stackrel {(c)} {=} \mathbb {E} _ {x} ^ {\pi} \left[ B _ {R} U ^ {\pi} \left(X ^ {\prime}\right) \right], \tag {4}
$$

where (a) follows from the distributional Bellman equation, (b) follows from Equation (3), and (c) from exchanging the linear map  $B_r$  and the conditional expectation given  $(R, X')$ , crucially relying on the linearity of the approximation in Equation (3). Note that for example with  $\phi(g) = (1, g)^{\top}$  we have  $B_r = \begin{pmatrix} 1 & 0 \\ r & \gamma \end{pmatrix}$ , and Equation (4) reduces to the classical Bellman equation for  $V^{\pi}$ , with  $U^{\pi}(x) = (1, V^{\pi}(x))^{\top}$ .

Thus,  $U^{\pi}(x)$  satisfies its own linear Bellman equation, which motivates algorithms that work directly in the space of sketches, without recourse to imputation strategies. In particular, a natural dynamic programming algorithm to consider is based on the recursion

$$
U (x) \leftarrow \mathbb {E} _ {x} ^ {\pi} \left[ B _ {R} U \left(X ^ {\prime}\right) \right]. \tag {Sketch-DP}
$$

As this is an update applied directly to sketch values themselves, we introduce the sketch Bellman operator  $\mathcal{T}_{\phi}^{\pi}:(\mathbb{R}^{m})^{\mathcal{X}}\to (\mathbb{R}^{m})^{\mathcal{X}}$ , with  $(\mathcal{T}_{\phi}^{\pi}U)(x)$  defined according to the right-hand side of

Equation (Sketch-DP). Note that  $\mathcal{T}_{\phi}^{\pi}$  is a linear operator, in contrast to the standard expected-value Bellman operator, which is affine. We recover the affine case by taking one component of  $\phi$  to be constant, e.g.  $\phi_1(g)\equiv 1$ , and enforcing  $U_{1}(x)\equiv 1$ .

The right-hand side of Equation (Sketch-DP) can be unbiasedly approximated with a sample transition  $(x, r, x')$ . Stochastic approximation theory (Kushner & Yin, 1997; Bertsekas & Tsitsiklis, 1996) then naturally suggests the following temporal-difference learning update, given a learning rate  $\alpha$ :

$$
U (x) \leftarrow (1 - \alpha) U (x) + \alpha B _ {r} V \left(x ^ {\prime}\right), \tag {Sketch-TD}
$$

Rowland et al. (2019) introduced the term Bellman closed for sketches for which an exact dynamic programming algorithm is available, and provided a characterisation of Bellman closed mean embedding sketches. The notion of Bellman closedness is closely related to the relationship in Equation (3), and from Rowland et al. (Theorem 4.3; 2019), we can deduce that the only mean embedding sketches that satisfy Equation (3) are invertible linear combinations of first- $m$  moments.

Thus, our discussion above serves as a way of re-deriving known algorithms for computing moments of the return (Sobel, 1982; Lattimore & Hutter, 2014), but is insufficient to yield algorithms for computing other sketches. Additionally, since moments of the return distribution are naturally of widely differing magnitudes, it is difficult to learn a high-dimensional mean embedding based on moments; see Appendix D.3 for further details. To go further, we must weaken the assumption made in Equation (3).

# 3.1 GENERAL SKETCHES

To extend our framework to a much more general family of sketches, we relax our assumption of the exact predictability of  $\phi (r + \gamma g)$  from  $\phi (g)$  in Equation (3), by defining a matrix of Bellman coefficients  $B_{r}$  for each possible reward  $r\in \mathcal{R}$  as the solution of the linear regression problem:

$$
B _ {r} := \underset {B} {\arg \min } \mathbb {E} _ {G \sim \mu} \left[ \| \phi (r + \gamma G) - B \phi (G) \| _ {2} ^ {2} \right], \tag {5}
$$

so that, informally, we have  $\phi (r + \gamma g)\approx B_r\phi (g)$  for each  $g$ . Here,  $\mu$  is a distribution to be specified that weights the returns  $G$ . Using the same motivation as in the previous section, we therefore obtain

$$
U ^ {\pi} (x) \stackrel {(a)} {=} \mathbb {E} _ {x} ^ {\pi} [ \phi (R + \gamma G ^ {\pi} (X ^ {\prime})) ] \approx \mathbb {E} _ {x} ^ {\pi} \left[ B _ {R} \phi \left(G ^ {\pi} \left(X ^ {\prime}\right)\right) \right] \stackrel {(c)} {=} \mathbb {E} _ {x} ^ {\pi} \left[ B _ {R} U ^ {\pi} \left(X ^ {\prime}\right) \right], \tag {6}
$$

noting that informally we have approximate equality in the middle of this line. This still motivates the approaches expressed in Equations (Sketch-DP) and (Sketch-TD), though we have lost the property that the exact sketch values  $U^{\pi}$  are a fixed point of the dynamic programming procedure.

Computing Bellman coefficients. Under mild conditions (invertibility of  $C$  as follows) the matrix of Bellman coefficients  $B_r$  defined in Equation (5) can be expressed as  $B_r = C_r C^{-1}$ , where  $C, C_r \in \mathbb{R}^{m \times m}$  are defined by

$$
C := \mathbb {E} _ {G \sim \mu} [ \phi (G) \phi (G) ^ {\top} ], \tag {7}
$$

$$
C _ {r} := \mathbb {E} _ {G \sim \mu} [ \phi (r + \gamma G) \phi (G) ^ {\top} ].
$$

The elements of these matrices are expressible as integrals over the real line, and hence several possibilities are available for (approximate) computation: if  $\mu$  is finitely-supported, direct summation is possible; in certain cases the integrals may be analytically available, and otherwise numerical integration can be performed. Additionally, for certain feature maps  $\phi$ , the Bellman coefficients  $B_{r}$  have particular structure that can be exploited computationally; see Appendix B.3 for further discussion. Detailed properties of  $B_{r}$  are studied in Appendix B.5.

# Algorithm 1 Sketch-DP/Sketch-TD

Precompute Bellman coefficients

Compute  $C$  as in Equation (7)

for  $r\in \mathcal{R}$  do

Compute  $C_r$  as in Equation (7)

Set  $\vec{B}_r = C_rC^{-1}$

end for

Initialise  $U:\mathcal{X}\to \mathbb{R}^m$

Main loop

if DP then

for  $k = 1,2,\ldots$  do

$$
U (x) \leftarrow \sum_ {r, x ^ {\prime}, a} P (r, x ^ {\prime} | x, a) \pi (a | x) B _ {r} U (x ^ {\prime}) \forall x
$$

end for

else if TD then

for  $k = 1,2,\ldots$  do

Observe transition  $(x_{k},a_{k},r_{k},x_{k}^{\prime})$

$$
U \left(x _ {k}\right) \leftarrow \left(1 - \alpha_ {k}\right) U \left(x _ {k}\right) + \alpha_ {k} B _ {r _ {k}} U \left(x _ {k} ^ {\prime}\right)
$$

end for

end if

Algorithms. We summarise the two core algorithmic contributions, sketch dynamic programming (Sketch-DP) and sketch temporal-difference learning (Sketch-TD), that arise from our proposed framework in Algorithm 1. Pausing to take stock, we have proposed an algorithm framework for computing approximations of lossy mean embeddings for a wide variety of feature functions  $\phi$ . Further, these algorithms operate directly within the space of sketch values.

Selecting feature maps. A natural question is what effects the choice of feature map  $\phi$  has on the performance of the algorithm. There are several competing concerns. First, the richer the map  $\phi$ , the more information about the return distribution can be captured by the corresponding mean embedding. However, the computational costs (both in time and memory) of our proposed algorithms scale in the worst case cubically with  $m$ , the dimensionality of the mean embedding. In addition, the accuracy of the algorithm in approximating the mean embeddings of the true return distributions relies on having a low approximation error in Equation (6), which in turn relies on a low regression error in Equation (5) (see Proposition 4.1 below). Selecting an appropriate feature map is therefore somewhat nuanced, and involves trading off a variety of computational and approximation concerns.

A collection of feature maps we will use throughout the paper that offer the potential for trade-offs along the dimensions identified above is given by the translation family

$$
\phi_ {i} (z) := \kappa \left(s \left(z - z _ {i}\right)\right), \forall i \in \{1, \dots , m \}, \tag {8}
$$

where  $\kappa : \mathbb{R} \to \mathbb{R}$  is a base feature function,  $s \in \mathbb{R}^+$  is the slope, and the set  $\{z_1, \ldots, z_m\} \subseteq \mathbb{R}$  is the anchors of the feature map. We will often take  $\kappa$  to be commonly used bounded and smooth nonlinear functions, such as the Gaussian or the sigmoid functions, and spread the anchor points over the return range. We emphasise that in principle there are no restrictions on the feature maps that can be considered in the framework; see Appendix B.2 for other possible choices.

Remark 3.1 (Invariance). Given the  $m$ -dimensional function space obtained from the span of the coordinate functions  $\phi_1, \ldots, \phi_m$ , the algorithms proposed above are essentially independent of the choice of basis for this space. For any invertible matrix  $M \in \mathbb{R}^{m \times m}$ , replacing  $\phi$  by  $M^{-1}\phi$ , and also  $\| \cdot \|_2$  by  $\| \cdot \|_{M^\top M}$  in Equation (5) gives an equivalent algorithm.

Remark 3.2 (The need for linear regression). It is tempting to try and obtain a more general framework by allowing non-linear regression of  $\phi(r + \gamma g)$  on  $\phi(g)$  in Equation (5), to obtain a more accurate fit, for example fitting a function  $H: \mathbb{R} \times \mathbb{R}^m \to \mathbb{R}^m$  so that  $\phi(r + \gamma g) \approx H(r, \phi(g))$ . The issue is that if  $H$  is not linear in the second argument, then generally  $\mathbb{E}[H(r, \phi(G(X')))] \neq H(r, \mathbb{E}[\phi(G(X'))])$ , and so step (c) in Equation (6) is not valid. However, there may be settings where it is desirable to learn such a function  $H$ , to avoid online computation of Bellman coefficients every time a new reward is encountered in TD learning.

# 3.2 SKETCH-DP AT WORK

To provide more intuition for the Bellman sketch framework, we provide a walk-through of using Algorithm 1 to estimate the return distributions for the environment in Figure 2A; full details for replication are given in Appendix C. We take a feature map  $\phi$  of the form given in Equation (8), taking  $\kappa$  to be the sigmoid function, and  $m = 13$  anchors evenly spaced between  $-4.5$  and 4.5 (Figure 2B). The Bellman regression problem in Equation (5) is set with  $\mu = \mathrm{Uniform}([-4,4])$ , based on the typical returns observed in the environment. The anchors and the choice of  $\mu$  for regression are important but can be set following simple heuristics; see Appendices B.2 and B.3. We then run the Sketch-DP algorithm with the initial estimates  $U(x)$  set to  $\phi(0)$  for all  $x \in \mathcal{X}$ .

We compare the estimates produced by Sketch-DP against ground-truth by estimating the true mean embeddings from a large number of Monte Carlo samples of the returns from each state. Figure 2C (top) shows illustrates the ground-truth embeddings for each state, and Figure 2D (top) compares these ground-truth embeddings with those computed by Sketch-DP as the algorithm progresses; by 30 iterations, the mean embeddings are very close to the ground-truth.

To aid interpretation of these results, we also include a comparison in which we "decode" the mean embeddings back into probability distributions (via an imputation strategy (Rowland et al., 2019)), and compare with the ground-truth return distributions, projected onto the anchor locations of the features (Rowland et al., 2018). Full details of the imputation strategy are in Appendix B.1. These results are shown in the bottom panels of Figure 2C & D. Initially, the imputed distributions of the Sketch-DP mean embedding estimates reflect the initialisation to the mean embedding of  $\delta_0$ ,

![](images/67cb5c2e0aaf28e8e8b4f6438960a9c3675a4456a11b81f129af68bb3b30a120.jpg)  
A

![](images/16c5347e45f8746f2dbf9ad2ca45b9d3d7f4afb3cb527a666d2a533c47e3d7b9.jpg)

![](images/450b85215e6c4802bdc146f13de412c249d8b2ca8d5ec9f0f774d5f6e81c8bdc.jpg)  
B  
Figure 2: A: State transitions and rewards in the environment. B: The feature functions  $\phi$  for the sketch-DP. Dotted lines indicate anchors. The regression Equation (5) is performed under a densely spaced grid over the light region  $[-4, 4]$ . C: The ground-truth mean embeddings under the sigmoid features in B, and the categorical projection of the ground-truth distribution onto the anchors of  $\phi$  in B. D: The evolution of the estimated mean embeddings (bright blue lines) and imputed distributions (bright red lines) during Sketch-DP. The stems are the respective ground-truth from panel C.

![](images/f10af08ca192ca5910357aceb2eff2d273e2bc0a853e5b5445fb34e3958ce441.jpg)

![](images/d8af32378ccaeaf2b2c2c0b21481382fa0ce3b2ff7a19119d51a7e4b72e41922.jpg)  
D

![](images/01095ef65665e6cbfd20aad0703959cdd0a95c2955ad1321e56ea3e7df1aa825.jpg)

![](images/d3ee6d3ef2475376c8036d9731b035017cd2a12f838d1e6215dd5b7d03a6cc12.jpg)

![](images/2009bd6c026f37ca4a35766ab252c5ada1d3d3630202a80f87c8ea1b03df90ac.jpg)

![](images/23c45572cf7f156c513a505af5a817837bb1d0783b5609e6c6c4867296a1901a.jpg)

![](images/de2d2811599a4d1158548271c8422a3445c2674f27e8daf5d206635fb9c7160c.jpg)

![](images/a1a18a8c90ad5e96b4f70506b90c9142a3a5a7acdd54a20141bfe5a079eac853.jpg)

![](images/8b1449a0abf97fe2ec5fdff5ccf6ed7ec57ac17a98273be2e127811baf268093.jpg)

![](images/6769e4ef74af32a524bf37af746cea84332a4b7a858adf73170010fa46747b3c.jpg)

![](images/d1dccddd7b1e1566cab83477c36d15cbb6dfcebc7073ab267264501f9bc1e95f.jpg)

though as more iterations of Sketch-DP are applied, the imputed distributions become close to the ground-truth. This indicates that, in this example, not only does Sketch-DP compute accurate mean embeddings of the return, but that this embedding is rich enough to recover a lot of information regarding the return distributions themselves.

Concluding the introduction of the Sketch-DP algorithmic framework, there are several natural questions that arise. Can we quantify how accurately Sketch-DP algorithms can approximate mean embeddings of return distributions? What effects do choices such as the feature map  $\phi$  have on the algorithms in practice? The next sections are devoted to answering these questions in turn.

# 4 CONVERGENCE ANALYSIS

We analyse the Sketch-DP procedure described in Algorithm 1, which can be mathematically described in the following succinct manner. We let  $U_0: \mathcal{X} \to \mathbb{R}^m$  denote the initial sketch value estimates, and then note from Algorithm 1 that the collection of estimates after each DP update form a sequence  $(U_k)_{k=0}^\infty$ , with  $U_{k+1} = \mathcal{T}_{\phi}^{\pi} U_k$ . Our convergence analysis therefore focuses on the asymptotic behaviour of this sequence. We introduce the notation  $\Phi: \mathcal{P}(\mathbb{R}) \to \mathbb{R}^m$  for the sketch associated with the feature function  $\phi$ , so that  $\Phi \mu = \mathbb{E}_{Z \sim \mu}[\phi(Z)]$ , and define  $\Phi$  for return-distribution functions by specifying for  $\eta \in \mathcal{P}(\mathbb{R})^{\mathcal{X}}$  that  $(\Phi \eta)(x) = \Phi(\eta(x))$ . Ideally, we would

like these iterates to approach  $U^{\pi} : \mathcal{X} \to \mathbb{R}^{m}$ , the sketch values of the true return distributions, given by  $U^{\pi}(x) = \mathbb{E}_{x}^{\pi}[\phi(\sum_{t=0}^{\infty} \gamma^{t} R_{t})]$ . As already described, typically this is not possible when the sketch  $\Phi$  is not Bellman closed, and so we can only expect to approximate  $U^{\pi}$ . Mathematically, this is because in general we have  $\Phi T^{\pi} \neq T_{\phi}^{\pi} \Phi$  when  $\phi$  is not Bellman closed.

![](images/202a90490acfb074adacf868485b074fb0454dcc4ed6bee304783f96dd26d5e2.jpg)  
Figure 3: The objects and structure used to analyse the Sketch-DP algorithm.

The first step is to bound the error incurred in a single step of dynamic programming due to using  $\mathcal{T}_{\phi}^{\pi}$  directly on the sketch values, rather taking sketch values after applying the true distributional Bellman operator to the underlying distributions; this corresponds to the foreground of Figure 3.

Proposition 4.1. (Regression error to Bellman approximation.) Let  $\| \cdot \|$  be a norm on  $\mathbb{R}^m$ . Then for any return-distribution function  $\eta \in \mathcal{P}([G_{\min}, G_{\max}])^\chi$ , we have

$$
\max  _ {x \in \mathcal {X}} \| \Phi (\mathcal {T} ^ {\pi} \eta) (x) - (\mathcal {T} _ {\phi} ^ {\pi} \Phi \eta) (x) \| \leq \sup  _ {g \in [ G _ {\min }, G _ {\max } ]} \max  _ {r \in \mathcal {R}} \| \phi (r + \gamma g) - B _ {r} \phi (g) \|. \tag {9}
$$

The second step of the analysis is to chain together the errors that are incurred at each step of dynamic programming, so as to obtain a bound on the asymptotic distance of the sequence  $(U_k)_{k=0}^{\infty}$

from  $U^{\pi}$ , motivated by error propagation analysis in the case of function approximation (Bertsekas & Tsitsiklis (1996); Munos (2003); see also Wu et al. (2023) in the distributional setting). The next proposition provides the technical tools required for this; the notation is chosen to match the illustration in Figure 3.

Proposition 4.2. (Error propagation.) Consider a norm  $\| \cdot \|$  on  $\mathbb{R}^m$ , and let  $\| \cdot \|_{\infty}$  be the norm on  $(\mathbb{R}^m)^\mathcal{X}$  defined by  $\| U \|_{\infty} = \max_{x \in \mathcal{X}} \| U(x) \|$ . Let  $d$  be a metric on return-distribution functions (RDFs) such that  $\mathcal{T}^{\pi}$  is a  $\gamma^{c}$ -contraction with respect to  $d$ . Suppose the following bounds hold.

- (Bellman approximation bound.) For any  $\eta \in \mathcal{P}([G_{\min}, G_{\max}])^{\chi}$ ,

$$
\max _ {x \in \mathcal {X}} \| \Phi (\mathcal {T} ^ {\pi} \eta) (x) - (\mathcal {T} _ {\phi} ^ {\pi} \Phi \eta) (x) \| \leq \varepsilon_ {\mathrm {B}}.
$$

- (Reconstruction error bound.) For any  $\eta, \bar{\eta} \in \mathcal{P}([G_{\min}, G_{\max}])^{\mathcal{X}}$  with sketches  $U, \bar{U}$ , we have  $d(\eta, \bar{\eta}) \leq \| U - \bar{U} \|_{\infty} + \varepsilon_{\mathrm{R}}$ .  
- (Embedding error bound.) For any  $\eta', \bar{\eta}' \in \mathcal{P}([G_{\min}, G_{\max}])^{\mathcal{X}}$  with sketches  $U'$ ,  $\bar{U}'$ , we have  $\| U' - \bar{U}' \|_{\infty} \leq d(\eta', \bar{\eta}') + \varepsilon_{\mathrm{E}}$ .

Then for any two return-distribution functions  $\eta, \bar{\eta} \in \mathcal{P}([G_{\min}, G_{\max}])^{\mathcal{X}}$  with sketches  $U, \bar{U}$  satisfying  $\| U - \bar{U} \| \leq \delta$ , we have

$$
\left\| \Phi \mathcal {T} ^ {\pi} \eta - \mathcal {T} _ {\phi} ^ {\pi} \bar {U} \right\| _ {\infty} \leq \gamma^ {c} (\delta + \varepsilon_ {\mathrm {R}}) + \varepsilon_ {\mathrm {R}} + \varepsilon_ {\mathrm {E}}.
$$

A formal proof is given in Appendix A; Figure 3 (bottom) shows the intuition, propagating bounds through different intermediate stages of the analysis of the update. We now state the main error bound result, which combines the two earlier results.

Proposition 4.3. Suppose the assumptions of Proposition 4.2 hold, that  $\mathcal{T}^{\pi}$  maps  $\mathcal{P}([G_{\min}, G_{\max}])^{\mathcal{X}}$  to itself, and suppose  $\mathcal{T}_{\phi}^{\pi}$  maps  $\{\Phi \nu : \nu \in \mathcal{P}([G_{\min}, G_{\max}])^{\mathcal{X}}\}$  to itself. Then for a sequence of sketches  $(U_k)_{k=0}^{\infty}$  defined iteratively via  $U_{k+1} = \mathcal{T}_{\phi}^{\pi} U_k$ , we have

$$
\operatorname * {l i m s u p} _ {k \to \infty} \| U _ {k} - U ^ {\pi} \| \leq \frac {1}{1 - \gamma^ {c}} (\gamma^ {c} \varepsilon_ {\mathrm {R}} + \varepsilon_ {\mathrm {B}} + \varepsilon_ {\mathrm {E}}).
$$

Proof. For each  $U_{k}$ , let  $\eta_{k}$  be an RDF with the property  $\Phi \eta_{k} = U_{k}$ . Applying Proposition 4.2 to sketches  $U^{\pi}$  and  $U_{k}$ , we obtain  $\| U_{k + 1} - U^{\pi}\|_{\infty}\leq \gamma^{c}\| U_{k} - U^{\pi}\|_{\infty} + \gamma^{c}\varepsilon_{\mathrm{R}} + \varepsilon_{\mathrm{B}} + \varepsilon_{\mathrm{E}}$ . Taking a limsup on both sides over  $k$  and rearranging yields the result.

# 4.1 CONCRETE EXAMPLE

The analysis presented above is abstract; it provides a generic template for conducting error propagation analysis to show that Sketch-DP converges to a neighbourhood of the true values, and moreover illustrates the dependence of this error on the "richness" of the sketch, and accuracy of the Bellman coefficients. To apply this abstract result to a concrete algorithm, we are required to establish the three error bounds that appear in the statement of Proposition 4.2. The result below shows how this can lead to a concrete result for a novel class of sketches; in particular, proving that computed mean embeddings under these features become arbitrarily accurate as the number of features increases.

Proposition 4.4. Consider a sketch  $\phi$  whose coordinates are feature functions of the form  $\phi_i(z) = \mathbb{1}\{z_1\leq z < z_{i + 1}\}$ $(i = 1,\dots ,m - 1)$ , and  $\phi_m(z) = \mathbb{1}\{z_1\leq z\leq z_{m + 1}\}$ , where  $z_{1},\ldots ,z_{m + 1}$  is an equally-spaced grid over  $[G_{\min},G_{\max}]$ , with  $G_{\min} = \min \mathcal{R} / (1 - \gamma)$ ,  $G_{\max} = \max \mathcal{R} / (1 - \gamma)$ . Let  $\mathcal{T}_{\phi}^{\pi}$  be the corresponding Sketch-DP operator given by solving Equation (5) with  $\mu = \mathrm{Unif}([G_{\min},G_{\max}])$ , and define a sequence  $(U_k)_{k = 0}^\infty$  by taking  $U_{0}(x)$  to be the sketch of some initial distribution in  $\mathcal{P}([G_{\min},G_{\max}])$ , and  $U_{k + 1} = \mathcal{T}_{\phi}^{\pi}U_{k}$  for all  $k\geq 0$ . Let  $U^{\pi}\in (\mathbb{R}^{m})^{\mathcal{X}}$  be the mean embeddings of the true return distributions. Finally, let  $\| \cdot \|$  be the norm on  $\mathbb{R}^m$  defined by  $\| u\| = \frac{G_{\max} - G_{\min}}{m}\sum_{i = 1}^{m}|u_i|$ . Then we have

$$
\limsup_{k\to \infty}\| U_{k} - U^{\pi}\|_{\infty}\leq \frac{(G_{\max} - G_{\min})(3 + 2\gamma)}{(1 - \gamma)m}.
$$

# 5 EXPERIMENTS

We first conduct a broad empirical investigation into the effects of three key factors in Equation (8): the base feature  $\kappa$ , the number of features  $m$ , and the slope  $s$ , using three tabular MRPs (details in Appendix C.1, extended results in Appendix D.1). As in the example in Section 3.2, we compare the mean embeddings estimated by Sketch-DP with ground-truth mean embeddings, reporting their squared  $L^2$  distance (mean embedding squared error), and also compare the Cramér distance  $\max_{x \in \mathcal{X}} \ell_2^2(\hat{\eta}(x), \eta^\pi(x))$  (see e.g. Rowland et al. (2018)) between the distribution  $\hat{\eta}(x)$  imputed from the Sketch-DP estimate, and the ground-truth return distribution  $\eta^\pi(x)$ . To aid interpretation of the Cramér distance results, we also report the Cramér distance between the ground truth  $\eta^\pi(x)$  and two baselines. First, the Dirac delta  $\delta_{V^\pi(X)}$  at the mean return; we expect Sketch-DP to outperform this naive baseline by better capturing properties of the return distribution beyond the mean. Second, the return distribution estimate computed by categorical DP (Rowland et al., 2018; Bellemare et al., 2023), a well-understood approach to distributional RL based on categorical distributions.

The results for sweeps over feature count  $m$  and slope  $s$  are shown in Figure 4. By sweeping over  $m$ , we see that the estimated mean embedding goes towards the ground-truth as we use more features. Further, the Cramér distance also decreases as  $m$  increases, suggesting that the distribution represented also approaches the ground-truth. To highlight differences between various Sketch-DP algorithms, we also compute the excess Cramér: the Cramér distance  $\max_{x \in X} \ell_2^2(\hat{\eta}(x), \eta^\pi(x))$  as above, minus the corresponding distance between the categorical projection of  $\eta^\pi$  (c.f. the red stems in Figure 2) and  $\eta^\pi$  itself. All distributional methods perform well on these tasks, and significantly outperform the Dirac estimator in stochastic environments; we note that all methods have tunable hyperparameters (bin locations for CDRL, feature parameters for Sketch-DP), which should inform the interpretation of these results, and in particular direct comparison between methods. The results of the sweep on the slope parameter  $s$  show different trends depending on the metric. For smoother  $\phi$ , generally we can obtain smaller errors on the mean embeddings, but the Cramér distances are only small for intermediate range of slope values. This result is expected: when the features are too smooth or too sharp, there exists regions within the return range where the feature values do not vary meaningfully. This results in a more lossy encoding of the return distribution, indicating the importance of tuning the slope parameter of the translation family (Equation (8)).

# 5.1 DEEP REINFORCEMENT LEARNING

We also verify that the Bellman sketch framework is robust enough to apply in combination with deep reinforcement learning. To do so, we aim to learn neural-network predictions  $U_{\theta}(x,a)$  of sketch values for each state-action pair  $(x,a)$  in the environment. To be able to define greedy policy improvements based on estimated sketch values, we precompute value-readout coefficients  $\beta \in \mathbb{R}^m$  by solving  $\arg \min_{\beta} \mathbb{E}_{G \sim \mu}[(G - \langle \beta, \phi(G) \rangle)^2]$ , so that we can predict expected returns from the sketch value as  $\langle \beta, U_{\theta}(x,a) \rangle$ . This allows us to define a greedy policy, and therefore a Q-learning-style update rule, which given an observed transition  $(x,a,r,x')$ , first computes  $a' =$

![](images/1ea542d9bdc81392ab09b23e46529d771f1600f9b57b15b93939421743b20b8f.jpg)  
Figure 4: Results of running Algorithm 1 on tabular environments.

![](images/acac4b3d6d0487ab4fc0a9b1230b945d43d8ea6381e111679983096d5d53d55d.jpg)

![](images/efbc8018d0b598e29204c175dd7dc207c545742be6c04b36067e28d314a0b89e.jpg)  
Figure 5: Median (left) and mean (right) human-normalised scores on the Atari 57 suite.

![](images/3695e7f3e3c879bce8d70ddd96d0142f4bbc43a9fb4cd5f3d772bfd7e9f04ffb.jpg)

arg  $\max_{\tilde{a}}\langle \beta ,U_{\bar{\theta}}(x^{\prime},\tilde{a})\rangle$  , and then the gradient:  $\nabla_{\theta}\| U_{\theta}(x,a) - B_rU_{\bar{\theta}}(x',a')\| _2^2$  , where  $\bar{\theta}$  are the target network parameters. In our experiments, we parametrise  $U_{\theta}$  according to the architecture of QR-DQN (Dabney et al., 2018b), so that the  $m$  outputs of the network predict the values of the  $m$  coordinates of the corresponding sketch value. We use the sigmoid function as the base feature  $\kappa$  Full experimental details for replication are in Appendix C.2; further results are in Appendix D.2.

Figure 5 shows the mean and median human-normalised performance on the Atari suite of environments (Bellemare et al., 2013) across 200M training frames, and includes comparisons against DQN (Mnih et al., 2015), as well as the distributional agents C51 (Bellemare et al., 2017), QR-DQN (Dabney et al., 2018b), and IQN (Dabney et al., 2018a). Sketch-DQN attains higher performance on both metrics relative to the comparator agents C51 and QR-DQN, and approaches the performance of IQN, which uses a more complex prediction network to make non-parametric predictions of the quantile function of the return. These results indicate that the sketch framework can be reliably applied to deep RL, and we believe further investigation of the combination of this framework and deep RL agents is a promising direction for future work.

# 6 RELATED WORK

Typical approaches to distributional RL focus on learning approximate distributions directly (see, e.g., Bellemare et al. (2017); Dabney et al. (2018b); Yang et al. (2019); Nguyen-Tang et al. (2021); Wu et al. (2023)). Much prior work has considered statistical functionals of the random return, at varying levels of generality with regard to the underlying Markov decision process model. See for example Mandl (1971); Farahmand (2019) for work on characteristic functions, Chung & Sobel (1987) for the Laplace transform, Tamar et al. (2013; 2016) for variance, and Sobel (1982) for higher moments. Our use of finite-dimensional mean embeddings is inspired by distributed distributional codes (DDCs) from theoretical neuroscience (Sahani & Dayan, 2003; Vértes & Sahani, 2018; Wenliang & Sahani, 2019), which can be regarded as neural activities encoding return distributions. DDCs were previously used to model transition dynamics and successor features in partially observable MDPs (Vértes & Sahani, 2019). Tano et al. (2020) consider applying non-linearities to rewards themselves, rather than the return, and learning with a variety of discount factors, to encode the distribution of rewards at each timestep. The sketches in this paper are in fact mean embeddings into finite-dimensional reproducing kernel Hilbert spaces (RKHSs; the kernel corresponding to the feature function  $\phi$  is  $K(z,z^{\prime}) = \langle \phi (z),\phi (z^{\prime})\rangle$ ). Kernel mean embeddings have previously been used in RL for representing state-transition distributions (Grünewälder et al., 2012; Boots et al., 2013; Lever et al., 2016; Chowdhury & Oliveira, 2023), and maximum mean discrepancies in RKHSs (Gretton et al., 2012) have been used to define losses in distributional RL by Nguyen-Tang et al. (2021).

# 7 CONCLUSION

We have proposed a framework for distributional reinforcement learning based on Bellman updates that take place entirely within the sketch domain. This has yielded new dynamic programming and temporal-difference learning algorithms as well as novel error propagation analysis, and we have provided further empirical analysis in the context of a suite of tabular MRPs, as well as demonstrating that the approach can be successfully applied at scale as a variant of the DQN architecture. We expect that there will be benefits from further exploration of algorithmic possibilities opened up by this framework, as well as potential consequences for value representations in the nervous system.

# REFERENCES

Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 2013.  
Marc G. Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning. In Proceedings of the International Conference on Machine Learning, 2017.  
Marc G. Bellemare, Salvatore Candido, Pablo Samuel Castro, Jun Gong, Marlos C. Machado, Subhodeep Moitra, Sameera S. Ponda, and Ziyu Wang. Autonomous navigation of stratospheric balloons using reinforcement learning. Nature, 588(7836):77-82, 2020.  
Marc G. Bellemare, Will Dabney, and Mark Rowland. Distributional Reinforcement Learning. MIT Press, 2023. http://www.distributional-rl.org.  
Alain Berlinet and Christine Thomas-Agnan. Reproducing kernel Hilbert spaces in probability and statistics. Springer Science & Business Media, 2011.  
Dimitri Bertsekas and John N. Tsitsiklis. Neuro-dynamic programming. Athena Scientific, 1996.  
Cristian Bodnar, Adrian Li, Karol Hausman, Peter Pastor, and Mrinal Kalakrishnan. Quantile QT-Opt for risk-aware vision-based robotic grasping. In Robotics: Science and Systems, 2020.  
Giulio Bondanelli and Srdjan Ostojic. Coding with transient trajectories in recurrent neural networks. PLoS computational biology, 16(2):e1007655, 2020.  
Byron Boots, Arthur Gretton, and Geoffrey J. Gordon. Hilbert space embeddings of predictive state representations. In Proceedings of the Conference on Uncertainty in Artificial Intelligence, 2013.  
Sayak Ray Chowdhury and Rafael Oliveira. Value function approximations via kernel embeddings for no-regret reinforcement learning. In Proceedings of The Asian Conference on Machine Learning, 2023.  
Kun-Jen Chung and Matthew J. Sobel. Discounted MDPs: Distribution functions and exponential utility maximization. SIAM Journal on Control and Optimization, 25(1):49-62, 1987.  
Will Dabney, Georg Ostrovski, David Silver, and Rémi Munos. Implicit quantile networks for distributional reinforcement learning. In Proceedings of the International Conference on Machine Learning, 2018a.  
Will Dabney, Mark Rowland, Marc G. Bellemare, and Rémi Munos. Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018b.  
Will Dabney, Zeb Kurth-Nelson, Naoshige Uchida, Clara Kwon Starkweather, Demis Hassabis, Rémi Munos, and Matthew Botvinick. A distributional code for value in dopamine-based reinforcement learning. Nature, 577(7792):671-675, 2020.  
Thang Doan, Bogdan Mazoure, and Clare Lyle. GAN Q-learning. arXiv preprint arXiv:1805.04874, 2018.  
Amir-massoud Farahmand. Value function in frequency domain and the characteristic value iteration algorithm. In Advances in Neural Information Processing Systems, 2019.  
Alhussein Fawzi, Matej Balog, Aja Huang, Thomas Hubert, Bernardino Romera-Paredes, Mohammadin Barekatain, Alexander Novikov, Francisco J. R. Ruiz, Julian Schrittwieser, Grzegorz Swirszcz, David Silver, Demis Hassabis, and Pushmeet Kohli. Discovering faster matrix multiplication algorithms with reinforcement learning. Nature, 610(7930):47-53, 2022.  
Dror Freirich, Tzahi Shimkin, Ron Meir, and Aviv Tamar. Distributional multivariate policy evaluation and exploration with the Bellman GAN. In Proceedings of the International Conference on Machine Learning, 2019.

Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander Smola. A kernel two-sample test. The Journal of Machine Learning Research, 13(1):723-773, 2012.  
Steffen Grünewäder, Guy Lever, Luca Baldassarre, Massi Pontil, and Arthur Gretton. Modelling transition dynamics in MDPs with RKHS embeddings. In Proceedings of the International Conference on Machine Learning, 2012.  
Guillaume Hennequin, Tim P Vogels, and Wulfram Gerstner. Non-normal amplification in random balanced neuronal networks. Physical Review E, 86(1):011909, 2012.  
Stratton C. Jaquette. Markov decision processes with a new optimality criterion: Discrete time. The Annals of Statistics, 1(3):496-505, 1973.  
Harold J. Kushner and George Yin. Stochastic approximation and recursive algorithm and applications. Springer, 1997.  
Tor Lattimore and Marcus Hutter. Near-optimal PAC bounds for discounted MDPs. Theoretical Computer Science, 558:125-143, 2014.  
Guy Lever, John Shawe-Taylor, Ronnie Stafford, and Csaba Szepesvari. Compressed conditional mean embeddings for model-based reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, 2016.  
Adam S. Lowet, Qiao Zheng, Sara Matias, Jan Drugowitsch, and Naoshige Uchida. Distributional reinforcement learning in the brain. Trends in neurosciences, 43(12):980-997, 2020.  
Stéphane Mallat. A wavelet tour of signal processing. Elsevier, 1999.  
Petr Mandl. On the variance in controlled Markov chains. Kybernetika, 7(1):1-12, 1971.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. Nature, 2015.  
Tetsuro Morimura, Masashi Sugiyama, Hisashi Kashima, Hirotaka Hachiya, and Toshiyuki Tanaka. Nonparametric return distribution approximation for reinforcement learning. In Proceedings of the International Conference on Machine Learning, 2010a.  
Tetsuro Morimura, Masashi Sugiyama, Hisashi Kashima, Hirotaka Hachiya, and Toshiyuki Tanaka. Parametric return density estimation for reinforcement learning. In Proceedings of the Conference on Uncertainty in Artificial Intelligence, 2010b.  
Rémi Munos. Error bounds for approximate policy iteration. In Proceedings of the International Conference on Machine Learning, 2003.  
Whitney K. Newey and James L. Powell. Asymmetric least squares estimation and testing. *Econometrica: Journal of the Econometric Society*, pp. 819-847, 1987.  
Thanh Nguyen-Tang, Sunil Gupta, and Svetha Venkatesh. Distributional reinforcement learning via moment matching. In Proceedings of the AAAI Conference on Artificial Intelligence, 2021.  
Mark Rowland, Marc Bellemare, Will Dabney, Rémi Munos, and Yee Whye Teh. An analysis of categorical distributional reinforcement learning. In Proceedings of the International Conference on Artificial Intelligence and Statistics, 2018.  
Mark Rowland, Robert Dadashi, Saurabh Kumar, Rémi Munos, Marc G. Bellemare, and Will Dabney. Statistics and samples in distributional reinforcement learning. In Proceedings of the International Conference on Machine Learning, 2019.  
Mark Rowland, Rémi Munos, Mohammad Gheshlaghi Azar, Yunhao Tang, Georg Ostrovski, Anna Harutyunyan, Karl Tuyls, Marc G Bellemare, and Will Dabney. An analysis of quantile temporal-difference learning. arXiv preprint arXiv:2301.04462, 2023.

Maneesh Sahani and Peter Dayan. Doubly distributional population codes: Simultaneous representation of uncertainty and multiplicity. Neural Computation, 2003.  
Alex Smola, Arthur Gretton, Le Song, and Bernhard Schölkopf. A Hilbert space embedding for distributions. In Proceedings of the International Conference on Algorithmic Learning Theory, 2007.  
Matthew J. Sobel. The variance of discounted Markov decision processes. Journal of Applied Probability, 19(4):794-802, 1982.  
Le Song, Xinhua Zhang, Alex Smola, Arthur Gretton, and Bernhard Scholkopf. Tailoring density estimation via reproducing kernel moment matching. In Proceedings of the International Conference on Machine Learning, 2008.  
Bharath K. Striperumbudur, Arthur Gretton, Kenji Fukumizu, Bernhard Scholkopf, and Gert R. G. Lanckriet. Hilbert space embeddings and metrics on probability measures. Journal of Machine Learning Research, 11:1517-1561, 2010.  
Ke Sun, Yingnan Zhao, Yi Liu, Wulong Liu, Bei Jiang, and Linglong Kong. Distributional reinforcement learning via sinkhorn iterations. arXiv preprint arXiv:2202.00769, 2022.  
Richard S. Sutton and Andrew G. Barto. Reinforcement learning: An introduction. MIT Press, 2nd edition, 2018.  
Aviv Tamar, Dotan Di Castro, and Shie Mannor. Temporal difference methods for the variance of the reward to go. In Proceedings of the International Conference on Machine Learning, 2013.  
Aviv Tamar, Dotan Di Castro, and Shie Mannor. Learning the variance of the reward-to-go. The Journal of Machine Learning Research, 17(1):361-396, 2016.  
Pablo Tano, Peter Dayan, and Alexandre Pouget. A local temporal difference code for distributional reinforcement learning. In Advances in Neural Information Processing Systems, 2020.  
Eszter Vértes and Maneesh Sahani. Flexible and accurate inference and learning for deep generative models. Advances in Neural Information Processing Systems, 31, 2018.  
Eszter Vértes and Maneesh Sahani. A neurally plausible model learns successor representations in partially observable environments. Advances in Neural Information Processing Systems, 32, 2019.  
Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, PEARU Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cirmrnan, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antonio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261-272, 2020.  
Li Kevin Wenliang and Maneesh Sahani. A neurally plausible model for online recognition and postdiction in a dynamical environment. Advances in Neural Information Processing Systems, 2019.  
Runzhe Wu, Masatoshi Uehara, and Wen Sun. Distributional offline policy evaluation with predictive error guarantees. In Proceedings of the International Conference on Machine Learning, 2023.  
Peter R. Wurman, Samuel Barrett, Kenta Kawamoto, James MacGlashan, Kaushik Subramanian, Thomas J. Walsh, Roberto Capobianco, Alisa Devlic, Franziska Eckert, Florian Fuchs, Leilani Gilpin, Piyush Khandelwal, Varun Kompella, HaoChih Lin, Patrick MacAlpine, Declan Oller, Takuma Seno, Craig Sherstan, Michael D. Thomure, Houmehr Aghabozorgi, Leon Barrett, Rory Douglas, Dion Whitehead, Peter Durr, Peter Stone, Michael Spranger, and Hiroaki Kitano. Outracing champion Gran Turismo drivers with deep reinforcement learning. Nature, 602(7896): 223-228, 2022.

Derek Yang, Li Zhao, Zichuan Lin, Tao Qin, Jiang Bian, and Tie-Yan Liu. Fully parameterized quantile function for distributional reinforcement learning. In Advances in Neural Information Processing Systems, 2019.  
Pushi Zhang, Xiaoyu Chen, Li Zhao, Wei Xiong, Tao Qin, and Tie-Yan Liu. Distributional reinforcement learning for multi-dimensional reward functions. Advances in Neural Information Processing Systems, 2021.
