# On the difficulty of learning chaotic dynamics with RNNs

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Recurrent neural networks (RNNs) are wide-spread machine learning tools for modeling sequential and time series data. They are notoriously hard to train because their loss gradients backpropagated in time tend to saturate or diverge during training. This is known as the exploding and vanishing gradient problem. Previous solutions to this issue either built on rather complicated, purpose-engineered architectures with gated memory buffers, or - more recently - imposed constraints that ensure convergence to a fixed point or restrict (the eigenspectrum of) the recurrence matrix. Such constraints, however, convey severe limitations on the expressivity of the RNN. Essential intrinsic dynamics such as multistability or chaos are disabled. This is inherently at disaccord with the chaotic nature of many, if not most, time series encountered in nature and society. It is particularly problematic in scientific applications where one aims to reconstruct the underlying dynamical system. Here we offer a comprehensive theoretical treatment of this problem by relating the loss gradients during RNN training to the Lyapunov spectrum of RNN-generated orbits. We mathematically prove that RNNs producing stable equilibrium or cyclic behavior have bounded gradients, whereas the gradients of RNNs with chaotic dynamics always diverge. Based on these analyses and insights we suggest ways of how to optimize the training process on chaotic data according to the system's Lyapunov spectrum, regardless of the employed RNN architecture.

# 1 Introduction

Recurrent neural networks (RNNs) are widely used across various fields in engineering and science for learning sequential tasks or modeling and predicting time series [53]. Yet, they struggle when long-term temporal dependencies, very slow, or hugely varying time scales are involved [6, 33, 51, 70, 77]. Time series or sequential data with such properties are, however, very common in fields like climate physics [81], neuroscience [21, 72], ecology [84], or language processing [10]. Training RNNs on such data is hard because the loss gradients backpropagated in time easily saturate or diverge in this process. This is commonly referred to as the exploding and vanishing gradient problem (EVGP) [6, 33, 65].

One solution to the EVGP is based on specifically designed RNN architectures with gating mechanisms, such as long short-term memory (LSTM) [34] or gated recurrent units (GRU) [11]. These architectures allow states at earlier time steps to more easily influence activity much later through a kind of protected memory buffer, thus alleviating the EVGP by structural design. In practice, such models need to be backed up by further techniques like gradient clipping to keep the gradients in check [65]. The relatively complex architectural design of these networks impedes their mathematical analysis and requires reverse engineering after training [55, 61, 62, 77]. Partly to forego these complications, a variety of other solutions has been proposed recently, imposing restrictions on the recurrence matrix to bound the gradients [4, 9], or enforcing global stability by design or

regularization [18, 48]. Often these procedures dramatically curtail the expressivity of the RNN [44, 64, 77]; in particular, they rule out chaotic dynamics (for reasons discussed further below).

This is at odds with the plethora of chaotic phenomena in nature, engineering, and society. Chaotic dynamics are commonplace, almost default in any complex physical or biological system. This includes scientific areas as diverse as neuroscience [16, 90], physiology [45], geophysics [78], climate systems [85], astrophysics [50], ecology [13], chemical reactions [20], cell [63] or population [57] biology. Chaotic phenomena are also crucial for the understanding of societal and epidemiological processes, such as the spread of diseases [56], or in economics [19]. They are further relevant in purely technical contexts such as electrical engineering [40, 80] or laser optics [42]. They have even been suggested to play an up to now largely neglected, but potentially very significant role in speech recognition [74] and natural language processing [35]. Hence, in almost any practical setting, chaotic phenomena abound. They cannot, in general, be ignored when devising RNN training algorithms.

Here we offer a comprehensive theoretical treatment of the relation between RNN dynamics and the behavior of the loss gradients during training. We find a close connection between an RNN's loss gradients and the largest Lyapunov exponent of its freely generated orbits. We mathematically prove that RNNs producing stable fixed point or cyclic behavior have bounded gradients. Crucially, however, the loss gradients of RNNs producing chaotic dynamics always diverge. Hence, the chaotic nature of many time series data induces a principle problem, and, despite significant efforts in the past to solve the EVGP, training RNNs on such data remains an open issue. We illustrate the implications of our theory for RNN training on several simulated and empirical chaotic time series, and adapt the old idea of sparsely forced BPTT as a simple yet effective remedy that enables to learn the underlying dynamics despite exploding gradients.

# 2 Related works

Exploding and vanishing gradients. While 'classical' remedies of the EVGP [6, 33, 65] rest on purpose-tailored architectures with gating mechanisms, which safeguard information flow across longer temporal distances [11, 34], the focus has recently shifted to simpler RNNs that address the EVGP by restricting the recurrence matrix to be orthogonal [30, 31, 36], unitary [4], or antisymmetric [9], or by ensuring globally stable fixed point solutions [37, 39], for example through co-trained Lyapunov functions [48]. However, all these approaches impose strong limitations on the dynamical repertoire of the RNN, enforcing global convergence to fixed points or simple cycles. In doing so, they drastically reduce the expressiveness of these models [44, 64]. To address this problem, Erichson et al. [18] somewhat relaxed the constraints on the recurrence matrix by introducing a skew-symmetric decomposition combined with a Lipschitz condition on the activation function. Another recent approach discretizes oscillator ODEs to arrive at a stable system of coupled [70] or independent [71] oscillators which increase the RNN's expressiveness while bounding its gradients. By design (and as acknowledged by the authors), neither of these architectures is capable of producing chaotic dynamics, however, as the underlying ODEs do not allow for exponential divergence of close-by trajectories (a prerequisite for chaos). Given these often principle limitations of parametrically or dynamically strongly constrained models, a fruitful direction may be to modify the training process itself, e.g. through modified or auxiliary loss functions [77, 82], or special procedures for parameter updating [38] or loss truncation [59, 92]. Our empirical evaluation will follow up on such ideas, but also highlight that simple loss truncation, windowing, or architectural solutions like LSTMs are not sufficient.

Learning dynamical systems. Surprisingly disconnected from the work on the EVGP and learning long-term dependencies, a huge and long-standing literature deals with training RNNs on nonlinear dynamical systems (DS) [67, 83, 87], including chaotic systems like the famous Lorenz equations [54] or chaotic turbulence in fluid dynamics [52]. Teacher forcing (TF; [12, 67, 92], see also [27]) is one of the earliest techniques introduced to keep RNN trajectories on track while training. The idea behind TF is to simply replace RNN states by observations when available, thereby also effectively cutting off the gradients. TF essentially derives from ideas in dynamical control theory, and adaptive schemes that increasingly hand over control to the RNN throughout training have been devised [1, 2, 5]. A related technique from the control theory literature is "multiple shooting" [89]: Here the whole observed time series is chopped into chunks, and for each chunk of trajectory a new initial condition is estimated. Explicit constraints ensure continuity between the separate trajectory

bits during optimization. State space models and the Expectation-Maximization algorithm became popular particularly in the 90es for uncovering the latent dynamics underlying a set of time series observations [22], and remain an important tool until today [15, 49]. Most recently, approaches based on variational inference and the reparameterization trick [47], like sequential variational autoencoders (SVAE), gained in popularity for DS approximation [8, 32]. "Deterministic" RNNs (i.e., with latent states not treated as random variables), like conventional LSTMs [86], remain top choices for DS reconstruction, however.

Although connections between DS ideas and loss gradients have been drawn early on [6], so far only particular scenarios (like fixed point attractors) have been considered. Closest to our work is recent work by Schmidt et al. [77], where non-divergence of loss gradients is established when RNNs converge to fixed points or cycles. However, this was done only for the particular class of piecewise-linear RNNs (PLRNNs), more restrictive conditions for cycles were imposed than assumed here, and - most importantly - the chaotic case on which we focus here was not considered. Recent studies [17, 88] also point out the general connections between Lyapunov exponents and loss gradients that we develop in sect. 3.1, but do not provide any in-depth theoretical treatment, proofs, or empirical evaluation of methods to alleviate exploding gradients in training, as we do here. Thus, a systematic theoretical framework that relates RNN dynamics more generally, and across a range of different RNN architectures, to the behavior of its training gradients, is still lacking so far.

# 3 Theoretical analysis: Relation between RNN dynamics and loss gradients

In our analysis, we will cover all major types of system dynamics (fixed points, cycles, chaos, and quasi-periodicity), and mathematically investigate their implications for the loss gradients. We will do this for all major classes of RNNs, including standard RNNs with largely arbitrary activation function, LSTMs, GRUs, and PLRNNs. The next section will first develop and illustrate the basic intuition behind the relations between RNN dynamics and loss gradients.

# 3.1 Preliminaries: RNN dynamics and loss gradients

Formally, all popular RNN architectures, including LSTMs, GRUs, or PLRNNs, are discrete time DS, defined by a (first-order-Markovian) recursive prescription for the temporal evolution of the latent states  $\mathbf{z}_t \in \mathbb{R}^M$  of the form

$$
\boldsymbol {z} _ {t} = F _ {\boldsymbol {\theta}} \left(\boldsymbol {z} _ {t - 1}, \boldsymbol {s} _ {t}\right), \tag {1}
$$

where  $s_t \in \mathbb{R}^N$  is the input at time  $t$  and  $\theta$  are RNN parameters. For instance, for standard RNNs we have  $F_{\theta}(z_{t-1}, s_t) = f(Wz_{t-1} + Bs_t + h)$ , where  $f$  is an element-wise activation function like tanh or a rectified linear unit (ReLU).

Assuming we start at some initial value  $z_{1} \in \mathbb{R}^{M}$ , and given a sequence of external inputs  $S = \{s_t\}$  we can recursively rewrite eq. (1) as

$$
\boldsymbol {z} _ {T} = F _ {\boldsymbol {\theta}} \left(F _ {\boldsymbol {\theta}} \left(F _ {\boldsymbol {\theta}} \left(\dots F _ {\boldsymbol {\theta}} \left(\boldsymbol {z} _ {1}, \boldsymbol {s} _ {2}\right) \dots\right)\right)\right) =: F _ {\boldsymbol {\theta}} ^ {T - 1} \left(\boldsymbol {z} _ {1}, \boldsymbol {s} _ {2}\right). \tag {2}
$$

In DS theory, we characterize the long-term behavior of such sequences by its spectrum of Lyapunov exponents. The Lyapunov exponents estimate the exponential growth rates in different local directions of the system's state space, and the largest Lyapunov exponent gives the dominant exponential behavior. Let us denote the system's Jacobian at time  $t$  by

$$
\boldsymbol {J} _ {t} := \frac {\partial F _ {\boldsymbol {\theta}} \left(\boldsymbol {z} _ {t - 1} , \boldsymbol {s} _ {t}\right)}{\partial \boldsymbol {z} _ {t - 1}} = \frac {\partial \boldsymbol {z} _ {t}}{\partial \boldsymbol {z} _ {t - 1}}. \tag {3}
$$

Then, the maximal Lyapunov exponent along an RNN trajectory  $\{z_1, z_2, \dots, z_T, \dots\}$  is defined as

$$
\lambda_ {m a x} := \lim  _ {T \rightarrow \infty} \frac {1}{T} \log \left\| \prod_ {r = 0} ^ {T - 2} J _ {T - r} \right\|, \tag {4}
$$

where  $\| \cdot \|$  denotes the spectral norm (or any subordinate norm) of a matrix. If  $\lambda_{max} < 0$  nearby trajectories will ultimately converge to a fixed point or cycle, while for  $\lambda_{max} > 0$  (a necessary condition for chaos) initially nearby trajectories will exponentially separate, i.e. we will have

divergence along one (or more) directions in state space. This accounts for the sensitive dependence on initial conditions in chaotic systems.

Now let  $\mathcal{L}(\boldsymbol{W},\boldsymbol{B},\boldsymbol{h})$  be some loss function employed for RNN training that decomposes in time as  $\mathcal{L} = \sum_{t=1}^{T}\mathcal{L}_{t}$ . Suppose we fancy BPTT as our training algorithm (similar derivations could be performed for RTRL), we recursively develop the loss gradients w.r.t. some RNN parameter  $\theta$  in time (i.e., across layers of the RNN unrolled in time) as

$$
\frac {\partial \mathcal {L}}{\partial \theta} = \sum_ {t = 1} ^ {T} \frac {\partial \mathcal {L} _ {t}}{\partial \theta} \quad \text {w i t h} \quad \frac {\partial \mathcal {L} _ {t}}{\partial \theta} = \sum_ {r = 1} ^ {t} \frac {\partial \mathcal {L} _ {t}}{\partial z _ {t}} \frac {\partial z _ {t}}{\partial z _ {r}} \frac {\partial^ {+} z _ {r}}{\partial \theta}, \tag {5}
$$

and

$$
\begin{array}{l} \frac {\partial \boldsymbol {z} _ {t}}{\partial \boldsymbol {z} _ {r}} = \frac {\partial \boldsymbol {z} _ {t}}{\partial \boldsymbol {z} _ {t - 1}} \frac {\partial \boldsymbol {z} _ {t - 1}}{\partial \boldsymbol {z} _ {t - 2}} \dots \frac {\partial \boldsymbol {z} _ {r + 1}}{\partial \boldsymbol {z} _ {r}} \\ = \prod_ {k = 0} ^ {t - r - 1} \frac {\partial \boldsymbol {z} _ {t - k}}{\partial \boldsymbol {z} _ {t - k - 1}} = \prod_ {k = 0} ^ {t - r - 1} \boldsymbol {J} _ {t - k}, \tag {6} \\ \end{array}
$$

where  $\partial^{+}$  denotes the immediate derivative. Now observe that the behavior of the loss gradients crucially depends on the product series of Jacobians in eqn. (6): If the maximum absolute eigenvalues of the Jacobians  $J_{t}$  will, in the geometric mean, be larger than 1 (i.e.,  $\left\| \prod_{r = 0}^{T - 2}J_{T - r}\right\|^{1 / T} > 1$ ), gradients will explode as  $T\to \infty$ , while they will saturate if  $\left\| \prod_{r = 0}^{T - 2}J_{T - r}\right\|^{1 / T} < 1$ . Thus, the key point to note is that the same terms that occur in the definition of the Lyapunov spectrum, eqn. (4), resurface in the loss gradients, eqn. (5) & (6). This accounts for the tight links between system dynamics and gradients.

# 3.2 Fixed points and cyclic dynamics

Let us start by considering the simplest types of dynamics that can occur in RNNs (or any discrete-time DS): fixed points and cycles. In fact, by far most of the literature on global stability in RNNs and on loss gradients focused on just fixed points [9, 18, 48], with only few authors who recently started to also connect cyclic behavior to loss gradients [70, 77]. Recall that a fixed point of a recursive map  $z_{t} = F(z_{t - 1})$  is defined as a point  $z^{*}$  for which we have  $z^{*} = F(z^{*})$ .2 Likewise, a  $k$ -cycle ( $k > 1$ ) is a set of temporally consecutive periodic points  $P_{k} := \{z_{t_{1}}, z_{t_{2}}, \ldots, z_{t_{k}}\} = \{z_{t_{1}}, F(z_{t_{1}}), \ldots, F^{k - 1}(z_{t_{1}})\}$  that we obtain from recursive application of the map such that each of the cyclic points  $z_{t_{r}} \in P_{k}$  is a fixed point of the  $k$  times iterated map  $F^{k}$  (with  $k$  being the smallest positive integer for which this holds). To simplify the subsequent treatment, we will collectively refer to fixed points and cycles as  $k$ -cycles ( $k \geq 1$ ). Further recall that a fixed point or  $k$ -cycle is called stable if the maximum absolute eigenvalue of the Jacobian evaluated at that point is smaller than 1, neutrally stable if exactly 1, and unstable otherwise. Although the results we develop in this and the following sections will hold more widely, we will restrict our attention to recursive maps  $F_{\theta}$  from the class of RNNs  $\mathcal{R} = \{\text{standardRNN}, \text{LSTM}, \text{GRU}, \text{PLRNN}\}$  (see Appx. A.1 for details).

Based on the observations made in the previous sections we can state the following theorem that links RNN dynamics and loss gradients:

Theorem 1. Consider an RNN  $F_{\theta} \in \mathcal{R}$  parameterized by  $\pmb{\theta}$ , and assume that it converges to a stable fixed point or  $k$ -cycle  $\Gamma_k$  ( $k \geq 1$ ) with  $\mathcal{B}_{\Gamma_k}$  as its basin of attraction. Then for every  $z_1 \in \mathcal{B}_{\Gamma_k}$  (i) the Jacobian  $\frac{\partial z_T}{\partial z_1}$  exponentially vanishes as  $T \to \infty$ ; (ii) for  $\Gamma_k$  the tangent vectors  $\frac{\partial z_T}{\partial \theta}$  and thus the gradient of the loss function,  $\frac{\partial \mathcal{L}_T}{\partial \theta}$ , will be bounded from above, i.e. will not diverge for  $T \to \infty$ ; and (iii) for the PLRNN (27) both  $\left\| \frac{\partial z_T}{\partial \theta} \right\|$  and  $\left\| \frac{\partial \mathcal{L}_T}{\partial \theta} \right\|$  will remain bounded for every  $z_1 \in \mathcal{B}_{\Gamma_k}$  as  $T \to \infty$ .

Proof. (i) Assume that  $\Gamma_{k}$  is a stable  $k$ -cycle ( $k \geq 1$ ) denoted by

$$
\Gamma_ {k} = \left\{z _ {1}, z _ {2}, \dots , z _ {T}, \dots \right\} = \left\{z _ {t ^ {* k}}, z _ {t ^ {* k} - 1}, \dots , \right.
$$

$$
\left. \boldsymbol {z} _ {t ^ {* k} - (k - 1)}, \boldsymbol {z} _ {t ^ {* k}}, \boldsymbol {z} _ {t ^ {* k} - 1}, \dots , \boldsymbol {z} _ {t ^ {* k} - (k - 1)}, \dots \right\}. \tag {7}
$$

171

Then, the largest Lyapunov exponent of  $\Gamma_{k}$  is given by

$$
\begin{array}{l} \lambda_ {\Gamma_ {k}} = \lim _ {t \rightarrow \infty} \frac {1}{t} \ln \left\| J _ {t} ^ {*} J _ {t - 1} ^ {*} \dots J _ {2} ^ {*} \right\| \\ = \lim  _ {j \rightarrow \infty} \frac {1}{j k} \ln \left\|\left(\prod_ {s = 0} ^ {k - 1} J _ {t ^ {* k} - s}\right) ^ {j} \right\|. \tag {8} \\ \end{array}
$$

$$
\lim  _ {t \rightarrow \infty} J _ {t} ^ {*} J _ {t - 1} ^ {*} \dots J _ {2} ^ {*} = \lim  _ {j \rightarrow \infty} \left(\prod_ {s = 0} ^ {k - 1} J _ {t ^ {* k} - s}\right) ^ {j} = 0. \tag {9}
$$

$$
\lambda_ {\mathcal {O} _ {z _ {1}}} = \lim  _ {T \rightarrow \infty} \frac {1}{T} \ln \| J _ {T} J _ {T - 1} \dots J _ {2} \| = \lambda_ {\Gamma_ {k}} <   0, \tag {10}
$$

$$
\lim  _ {T \rightarrow \infty} \left\| \frac {\partial \boldsymbol {z} _ {T}}{\partial \boldsymbol {z} _ {1}} \right\| = \lim  _ {T \rightarrow \infty} \| J _ {T} J _ {T - 1} \dots J _ {2} \| = 0. \tag {11}
$$

# 3.3 Chaotic dynamics

172  
By assumption of stability of  $\Gamma_{k}$  we have  $\lambda_{\Gamma_k} < 0$  and also  $\rho \left(\prod_{s = 0}^{k - 1}J_{t^{*k} - s}\right) < 1$ , which implies  
Now suppose that  $\mathcal{O}_{\pmb{z}_1}$  is an orbit of (1) converging to  $\Gamma_k$ , i.e.  $\pmb{z}_1 \in \mathcal{B}_{\Gamma_k}$ . Since  $\mathcal{O}_{\pmb{z}_1}$  and  $\Gamma_k$  have the same largest Lyapunov exponent, we have  
and hence for  $z_{1}\in \mathcal{B}_{\Gamma_{k}}$  
(ii) & (iii) See Appx. A.2.1.  
Remark 1. The result of Theorem 1 part (i) will be generally true for any first-order-Markovian recursive map (1), but the conclusions in part (ii) may hinge on its specific definition.  
Remark 2. None of the results above and throughout sect. 3 require the dynamics to be autonomous, the theory applies whether there is external input or not. In fact, mathematically, non-autonomous (externally forced) systems can always be rewritten as autonomous dynamical systems [3, 68, 94], see Appx. A.1.1 for details.  
The results above ensure that loss gradients will not diverge (explode) as  $T \to \infty$  in RNNs that are "well-behaved" in the sense that they converge to a fixed point or cycle. This is a generalization of the results given in Theorem 1 in Schmidt et al. [77], where this was shown only a) for the specific class of PLRNNs and b) for specific constraints imposed on the eigenvalue spectrum of the RNN's Jacobians which were relaxed in our theorem above.  
While our treatment above is centered on the "exploding-gradients" case, various architectural modifications or regularization techniques can ensure that gradients do not vanish either, i.e. remain bounded from below as well. This was established, for instance, in Schmidt et al. [77] for PLRNNs using 'manifold attractor regularization'. In Appx. A.2.1 we show that the results from Theorem 2 from Schmidt et al. [77] on doubly bounded (from below and above) loss gradients can indeed be extended to the more general case covered by Theorem 1 above.  
We will now consider the all-important chaotic case. Let  $F$  be a recursive map and  $\mathcal{O}_{z_1} = \{z_1, z_2, z_3, \dots\}$  be an orbit of  $F$ . The orbit is chaotic if (i) it is not asymptotically periodic and (ii) has at least one positive Lyapunov exponent [25, 58]. If the system's invariant set is bounded, condition (ii) is considered a standard signature of chaos, as in this case two nearby orbits separate exponentially fast, but at the same time their mutual separation cannot go to infinity so that there are also folds. The following theorem states the sufficient condition for exploding gradients:  
Theorem 2. Suppose that an RNN  $F_{\theta} \in \mathcal{R}$  (parameterized by  $\theta$ ) has a chaotic attractor  $\Gamma^{*}$  with  $\mathcal{B}_{\Gamma^{*}}$  as its basin of attraction. Then, for almost every orbit with  $z_{1} \in \mathcal{B}_{\Gamma^{*}}$ , (i) the Jacobians connecting temporally distal states  $z_{T}$  and  $z_{t}$  ( $T \gg t$ ),  $\frac{\partial z_{T}}{\partial z_{t}}$ , will exponentially explode for  $T \to \infty$ , and (ii) the tangent vector  $\frac{\partial z_{T}}{\partial \theta}$  and so the gradients of the loss function,  $\frac{\partial \mathcal{L}_T}{\partial \theta}$ , will diverge as  $T \to \infty$ .

Proof. Let the RNN  $F_{\theta} \in \mathcal{R}$  have a chaotic orbit denoted by  $\Gamma^{*} = \{z_{1}^{*}, z_{2}^{*}, \dots, z_{T}^{*}, \dots\}$ . Then, denoting by  $J_{T}^{*}$  the Jacobian of (1) at  $z_{T}^{*} \in \Gamma^{*}$ , the largest Lyapunov exponent of  $\Gamma^{*}$  is given by

$$
\lambda = \lim  _ {T \rightarrow \infty} \frac {1}{T} \ln \left\| J _ {T} ^ {*} J _ {T - 1} ^ {*} \dots J _ {2} ^ {*} \right\|. \tag {12}
$$

Since  $\Gamma^{*}$  is chaotic, so  $\lambda > 0$ . Hence, from (12), it is concluded that

$$
\lim  _ {T \rightarrow \infty} \left\| J _ {T} ^ {*} J _ {T - 1} ^ {*} \dots J _ {2} ^ {*} \right\| = \lim  _ {T \rightarrow \infty} \left\| \frac {\partial \boldsymbol {z} _ {T} ^ {*}}{\partial \boldsymbol {z} _ {t} ^ {*}} \right\| = \infty , T \gg t. \tag {13}
$$

Now, according to Oseledec's multiplicative ergodic Theorem, almost all the points in the basin of attraction of  $\Gamma^{*}$  have the same largest Lyapunov exponent  $\lambda$ . Thus, (13) holds for almost every  $z_{1} \in \mathcal{B}_{\Gamma^{*}}$ .

(ii) See Appx. A.2.2.

Remark 3. The first part of Theorem 2 holds for all first-order-Markovian recursive maps (1). Note that for LSTMs,  $\frac{\partial\boldsymbol{z}_T}{\partial\boldsymbol{z}_t}$  ( $\boldsymbol {z} \coloneqq (\boldsymbol {h},\boldsymbol {c})^{\mathrm{T}}$ ) denotes the full Jacobian of both hidden and cell states.

We collect some further mathematical results and remarks related to Theorem 2 in Appx. A.3.1.

Hence, the essential result is that for all popular RNNs  $\mathcal{R}$  and activation functions, loss gradients will inevitably diverge if the RNN latent states converge to a chaotic attractor.

# 3.4 Quasi-periodicity

Quasi-periodicity is a long-term behavior which occurs on a torus and, superficially, bears some similarity to chaos in the sense that, strictly speaking, orbits are also aperiodic. That is, as  $T \to \infty$ , trajectories will never close up with themselves. Moreover, every trajectory becomes arbitrarily close to any point on the torus, that is, it is dense. One important difference between quasi-periodic and chaotic systems is, however, that in a quasi-periodic system, as time passes, two close initial conditions are linearly diverging, while in a chaotic system the divergence is exponential.

Theorem 3. Assume that an RNN  $F_{\theta} \in \mathcal{R}$  (parameterized by  $\theta$ ) has a quasi-periodic attractor  $\Gamma$  with  $\mathcal{B}_{\Gamma}$  as its basin of attraction. Then, for every  $z_1 \in \mathcal{B}_{\Gamma}$

$$
\forall 0 <   \epsilon <   1 \exists T _ {0} > 1 s. t. \forall T \geq T _ {0} \Longrightarrow
$$

$$
(1 - \epsilon) ^ {T - 1} <   \left\| \frac {\partial z _ {T}}{\partial z _ {1}} \right\| <   (1 + \epsilon) ^ {T - 1}. \tag {14}
$$

Proof. See Appx. A.2.3.

According to Theorem 3, for every orbit converging to a quasi-periodic attractor, the Jacobians  $\frac{\partial z_T}{\partial z_t}$  may diverge or vanish as  $T \to \infty$ , but this will not occur exponentially fast as  $T \to \infty$ . Thus, even for bounded non-chaotic RNNs we may sometimes stumble into the problem of diverging gradients. Although this may be a less common scenario, we point out it may occur if we train RNNs on real data from oscillatory systems with incommensurate frequencies, as for instance encountered in electronic engineering.

In Appx. A.3.2 we have collected further mathematical results on the connection between RNN dynamics and loss gradients that hold regardless of the RNN's limiting behavior.

# 4 Empirical evaluation

Our theoretical results imply that chaotic time series pose a principle challenge for RNN training that cannot easily be circumvented through specifically designed architectures, constraints, or regularization criteria. If the underlying DS we aim to capture is chaotic, loss gradients propagated back in time will inevitably explode. Hence we need to curtail gradients in an ideal way. The issue arises especially in scientific ML where time series from chaotic systems are ubiquitous and the aim is to reconstruct the generating DS with its limiting behavior. Thus, our exposition will focus on this area.

# 4.1 Training on systems with exploding gradients by sparse teacher forcing

To illustrate the connections between theory and RNN training, we revive the old idea of TF [92] as a mechanism for truncating error gradients while training. However, we would like to do this such that important information about the system dynamics does not get lost, for which Lyapunov theory offers some guidance. Specifically, we should not force the system back onto the true trajectory all or most of the time (as in "classical TF"), but should effectively "re-calibrate" it only at certain time points chosen wisely according to the system's local divergence rates. This procedure will be referred to as sparsely forced BPTT in the following. Assume we want to train an RNN with hidden states  $\boldsymbol{z}_t \in \mathbb{R}^M$  and linear (or affine) output layer on a time-series  $\{\boldsymbol{x}_1, \boldsymbol{x}_2, \dots, \boldsymbol{x}_T\}$  generated by a chaotic system. The linear output layer  $\hat{\boldsymbol{x}}_t = \boldsymbol{B}\boldsymbol{z}_t$ ,  $\boldsymbol{B} \in \mathbb{R}^{N \times M}$ , maps the RNN hidden states into the observation space. This allows us to modify the original TF procedure by constructing a control series  $\{\tilde{z}_1, \tilde{z}_2, \dots, \tilde{z}_T\}$  from the observations by "inverting" the linear output mapping

$$
\tilde {z} _ {t} = \left(\boldsymbol {B} ^ {\mathsf {T}} \boldsymbol {B}\right) ^ {- 1} \boldsymbol {B} ^ {\mathsf {T}} \boldsymbol {x} _ {t}. \tag {15}
$$

The idea is to supply this control signal only sparsely, separated by the learning interval  $\tau$  between consecutive forcings. Hence, defining  $\mathcal{T} = \{n\tau +1\}_{n\in \mathbb{N}_0}$  as the set of all time points at which we force the RNN onto the 'true' values, the RNN updates can be written as

$$
\boldsymbol {z} _ {t + 1} = \left\{ \begin{array}{l l} R N N \left(\tilde {\boldsymbol {z}} _ {t}\right) & \text {i f} t \in \mathcal {T} \\ R N N \left(\boldsymbol {z} _ {t}\right) & \text {e l s e} \end{array} . \right. \tag {16}
$$

This forcing is applied after calculation of the loss, such that  $\mathcal{L}_t = \| \pmb{x}_t - \pmb{B}\pmb{z}_t\|_2^2$  irrespective of whether  $t$  is in  $\mathcal{T}$  or not (and of course it is applied only during training, not at test time!). Replacing hidden states  $\pmb{z}_t$  with their teacher-forced signals  $\tilde{\pmb{z}}_t$  simply breaks divergence between true and predicted trajectories at time points  $t \in \mathcal{T}$ , and also cuts off the Jacobians by breaking the temporal contingency (for details see Appx. A.7). The learning interval  $\tau$  hence controls how many time steps are included in the gradient calculation and has to be chosen with care such as to balance the effects of exploding gradients vs. those of losing relevant time scales and long-term dependencies. While it is general wisdom that an optimal batch size will facilitate training, the point here is thus much more specific: Ideally  $\tau$  should be chosen in accordance with the system's Lyapunov spectrum, for instance based on the predictability time [7]

$$
\tau_ {\text {p r e d}} = \frac {\ln 2}{\lambda_ {\max }}. \tag {17}
$$

We emphasize that such a simple recipe for addressing the exploding gradient problem is based on modifying the training routine, and is thus in principle applicable to any model architecture.

# 4.2 Example 1: Lorenz system and externally forced Duffing oscillator in chaotic regime

Let us illustrate these ideas on two classical textbook examples of chaotic DS, the chaotic Lorenz attractor as an autonomous system, and the chaotically forced Duffing oscillator as an example with explicit external input (see Appx. A.4 for details). Trajectories were repeatedly drawn from these systems, on which we trained a PLRNN, a vanilla RNN with tanh activation function, and a LSTM by stochastic gradient descent (SGD) to minimize the MSE loss between predicted and actual observations. As optimizer we used Adam [46] from PyTorch [66] with a learning rate of 0.001. For all models, training proceeded solely by sparsely forced BPTT and did not employ gradient clipping or any other technique that may interfere with optimal loss truncation.

In nonlinear DS reconstruction, we are mainly interested in reproducing invariant properties of the underlying system such as the attractor geometry (or topology; [75, 79]) or the frequency composition (i.e., time-averaged properties), while measures like ahead-prediction errors are less meaningful especially on chaotic time series [49, 93]. Thus, in evaluating training performance, here we follow Koppe et al. [49] in using a Kullback-Leibler divergence  $D_{stsp}$  to quantify the agreement between observed and generated probability distributions across state-space to assess the overlap in attractor geometry (Appx. A.5). Moreover, we calculate the dimension-wise Hellinger distance  $D_H$  between

![](images/0f45b820aabe4ef2d843bf8427d4216642350b0c2380b46153d920a864cd378b.jpg)

![](images/e0fbdcee41078bf43b574bbfff5720274c6e5b40417f5dfade93b8aad4db37b7.jpg)  
power spectra to quantify the temporal agreement of the observed and generated time-series (Appx. A.5).

![](images/ffe23f3cb58aa2c0029e16d90dd8d49f947b041c5a512ee623b25baf591c256a.jpg)  
Figure 1: Overlap in attractor geometry ( $D_{stsp}$ , lower = better) and dimension-wise comparison of power-spectra ( $D_H$ , lower = better) against learning interval  $\tau$  for (a) the Lorenz and (b) the chaotically forced Duffing oscillator. Continuous lines = sparsely forced BPTT. Dashed lines = classical BPTT with gradient clipping. Prediction time indicated vertically in black.

![](images/6ee24c6eb2827b43f671ed4861de71d71380d06252bdfda08719ac4cd2fc2253.jpg)

![](images/dedaccb50ab0c7d88aaadb3aa264e3b8f2819beed15eaff56a707bd07698f26f.jpg)  
Figure 2: Lorenz attractor (blue) and example reconstructions by a LSTM (orange) trained with a learning interval (a) chosen too small ( $\tau = 5$ ), (b) chosen optimally ( $\tau = 30$ ), and (c) chosen too large ( $\tau = 200$ ).

![](images/4fa94e0b72afe961c913ff59b48588ad6befff33326963aba2445596cc8246da.jpg)  
Fig. 1 shows the dependence of the reconstruction quality on the learning interval  $\tau$  for all RNN architectures on (a) the Lorenz and (b) the externally forced Duffing system. Fig. 2 provides particular examples of reconstructions for  $\tau$  chosen too small, too large, or about right.

![](images/52799b32fd02ba5f43aa411b60dc60c042699fd99e75f2a2adff91fa0640a3b7.jpg)

For all models we find a system-dependent range for the optimal learning interval that agrees well with the predictability time defined in eqn. (17), where estimates for the maximal Lyapunov exponent were taken from the literature [23, 69]. As a reference, dashed lines represent the reconstruction performance for all architectures when trained with classical BPTT and gradient clipping. The training procedure was the same as for sparsely forced BPTT, except that instead of supplying a control-signal, gradients were normalized to 1 prior to each parameter update. As evidenced by the much worse performance, gradient clipping does not effectively address the EVGP, even for LSTMs. As further shown in Fig. 9 in Appx. A.6.4, using the optimal window length  $\tau$  but resetting the initial condition to zero (instead of its control value  $\tilde{z}_t$ ) for each chunk equally destroys performance. This suggests that neither mere gradient normalization nor simple windowing are sufficient, but will wipe out essential information about the dynamics.

In Appx. A.6 we collect further results on the chaotic Rössler attractor (Fig. 5), high-dimensional Mackey-Glass equations (Fig. 7), and the Lorenz attractor with partial observations (Fig. 8).

# 4.3 Example 2: Chaotic weather data

As for an empirical example, we trained all RNNs (vanilla RNN, PLRNN, LSTM) on a temperature time series recorded at the Weather Station at the Max Planck Institute for Biogeochemistry in Jena,

![](images/017b81e17285abb7c83c23cd3013ef0797df96fe6c49dd66a7878f01601ab2db.jpg)  
(a)

![](images/7fb1c4b7b7ad81a3af8a06e0784ec4c556acf8c4f166ff3714ccad956c92de44.jpg)  
Figure 3: (a) The maximal Lyapunov exponent was determined as the slope of the average log-divergence of nearest neighbors in embedding space ( $m =$  embedding dimension). (b) Reconstruction quality assessed by attractor overlap (lower  $=$  better) and dimension-wise comparison of power-spectra ( $D_H$ , lower  $=$  better). Black vertical lines  $= \tau_{\mathrm{pred}}$ .  
(b)

![](images/a79302099c353c902f7cd6b643b4c73e5287f3f463e79a0a896652d70a421bcd.jpg)

Germany. To expose the chaotic behavior and obtain a robust estimate of the maximal Lyapunov exponent, trends and yearly cycles were removed, and nonlinear noise-reduction was performed ([42]; Appx. A.4). The maximal Lyapunov exponent was determined with the TISEAN package [29], as shown in Figure 3 (a). The value obtained is in close agreement with the literature [60].

Figure 3 shows that also for these empirical data the optimal training interval  $\tau$  agrees well with the predictability time, eqn. (17), for all trained RNNs. Furthermore, as was the case for the DS benchmarks, gradient clipping was not able to satisfactorily tackle the EVGP, even when paired with architectures like LSTMs explicitly designed for alleviating this problem. Similar results are reported for another real-world dataset, electroencephalogram (EEG) recordings, in Appx. A.11.

# 5 Discussion and conclusions

In this paper we proved that RNN dynamics and loss gradients are intimately related for all major types of RNNs and activation functions. If the RNN is "well behaved" in the sense that its dynamics converges to a fixed point or cycle, loss gradients will remain bounded, and established remedies [34, 77] can be used to refrain them from vanishing. However, if the dynamics are chaotic, gradients will always explode. This constitutes a principle problem in RNN training that cannot easily be mastered through architectural design or gradient clipping. It is furthermore a practically highly relevant one, as most time series we encounter in nature, and many from man-made systems as well, are inherently chaotic. While we do not offer a full solution to this problem here, we suggest it might be tackled in training by taking a system's local divergence rates as measured through the Lyapunov spectrum into account. Hence, rather than conquering the EVGP by structural design or specific constraints or regularization terms, we recommend to put the focus more on the training process itself. We illustrated this point empirically using sparsely forced BPTT, a training technique that pulls trajectories back on track at times determined by the maximal Lyapunov exponent. Doing so leads to optimal reconstruction results for a variety of simulated and real-world benchmarks, regardless of the specific RNN architecture employed in training.

We stress that our goal here above all was to provide a mathematically grounded perspective on the problem, with the empirical section focused on elucidating the practical implications of the theoretical results. We believe that a more thorough theoretical understanding is important and needed for guiding future research into more powerful training procedures that avoid exploding gradients without compromising expressiveness. In our application examples, we developed the case from the perspective of scientific machine learning, which by now is a broad area in its own right with huge societal relevance (e.g., climate or epidemiological time series), and where the reconstruction of geometrical or topological (invariant) properties is important, beyond mere prediction. Nevertheless, we believe that our theoretical results will also have implications for other domains, like NLP [35]. While scientific time series problems traditionally have been extensively considered from a DS perspective (e.g., [41]), much more groundwork is needed, however, in areas like NLP, where, for instance, it may not even be immediately clear how to best define a Lyapunov spectrum.

# References

[1] H. D. I. Abarbanel. Predicting the Future: Completing Models of Observed Complex Systems. en. Understanding Complex Systems. New York: Springer-Verlag, 2013. DOI: 10.1007/978-1-4614-7218-6. URL: https://www.springer.com/gp/book/9781461472179 (visited on 09/13/2021).  
[2] H. D. I. Abarbanel, P. J. Rozdeba, and S. Shirman. "Machine Learning: Deepest Learning as Statistical Data Assimilation Problems". eng. In: Neural Computation 30.8 (Aug. 2018), pp. 2025-2055. DOI: 10.1162/neco_a_01094.  
[3] K. T. Alligood, T. D. Sauer, and J. A. Yorke. Chaos: An Introduction to Dynamical Systems. Springer, New York, NY, 1996.  
[4] M. Arjovsky, A. Shah, and Y. Bengio. "Unitary Evolution Recurrent Neural Networks". In: Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48. ICML'16. New York, NY, USA: JMLR.org, 2016, pp. 1120-1128.  
[5] S. Bengio et al. "Scheduled sampling for sequence prediction with recurrent Neural networks". In: Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1. NIPS'15. Cambridge, MA, USA: MIT Press, Dec. 2015, pp. 1171-1179. (Visited on 09/07/2021).  
[6] Y. Bengio, P. Simard, and P. Frasconi. "Learning long-term dependencies with gradient descent is difficult". In: IEEE Transactions on Neural Networks 5.2 (1994), pp. 157-166. DOI: 10.1109/72.279181.  
[7] B. P. Bezruchko and D. A. Smirnov. Extracting Knowledge From Time Series: An Introduction to Nonlinear Empirical Modeling. en. Springer Series in Synergetics. Berlin Heidelberg: Springer-Verlag, 2010. DOI: 10.1007/978-3-642-12601-7. URL: https://www.springer.com/gp/book/9783642126000.  
[8] P. L. Bommer et al. Identifying nonlinear dynamical systems from multi-modal time series data. 2021. arXiv: 2111.02922 [cs.LG].  
[9] B. Chang et al. "AntisymmetricRNN: A Dynamical System View on Recurrent Neural Networks". In: International Conference on Learning Representations. 2019. URL: https://openreview.net/forum?id=ryxepo0cFX.  
[10] K. Cho et al. "Learning phrase representations using RNN encoder-decoder for statistical machine translation". English (US). In: Conference on Empirical Methods in Natural Language Processing (EMNLP 2014). 2014.  
[11] K. Cho et al. "On the Properties of Neural Machine Translation: Encoder-Decoder Approaches". In: Proceedings of SSST-8, Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation. Doha, Qatar: Association for Computational Linguistics, Oct. 2014, pp. 103-111. DOI: 10.3115/v1/W14-4012. URL: https://aclanthology.org/W14-4012.  
[12] K. Doya. “Bifurcations in the learning of recurrent neural networks”. In: [Proceedings] 1992 IEEE International Symposium on Circuits and Systems. Vol. 6. May 1992, 2777–2780 vol.6. DOI: 10.1109/ISCAS.1992.230622.  
[13] J. Duarte et al. “Quantifying chaos for ecological stoichiometry”. In: Chaos: An Interdisciplinary Journal of Nonlinear Science 20.3 (Sept. 2010). Publisher: American Institute of Physics, p. 033105. DOI: 10.1063/1.3464327. URL: https://aip.scitation.org/doi/full/10.1063/1.3464327.  
[14] G. Duffing. Erzwungene Schwingungen bei Veränderlicher Eigenfrequenz. 1918.  
[15] D. Durstewitz. Advanced Data Analysis in Neuroscience: Integrating Statistical and Computational Models. en. Bernstein Series in Computational Neuroscience. Springer International Publishing, 2017. DOI: 10.1007/978-3-319-59976-2. URL: https://www.springer.com/de/book/9783319599748 (visited on 09/10/2021).  
[16] D. Durstewitz and T. Gabriel. "Dynamical Basis of Irregular Spiking in NMDA-Driven Prefrontal Cortex Neurons". en. In: Cerebral Cortex 17.4 (Apr. 2007), pp. 894–908. DOI: 10.1093/cercor/bhk044. URL: https://academic.oup.com/cercor/article-lookup/doi/10.1093/cercor/bhk044.  
[17] R. Engelken, F. Wolf, and L. F. Abbott. "Lyapunov spectra of chaotic recurrent neural networks". In: arXiv:2006.02427 [nlin, q-bio] (June 2020). arXiv: 2006.02427. URL: http://arxiv.org/abs/2006.02427 (visited on 09/26/2021).

[18] N. B. Erichson et al. "Lipschitz Recurrent Neural Networks". In: International Conference on Learning Representations. 2021. URL: https://openreview.net/forum?id=-N7PBXqOUJZ.  
[19] M. Faggini. "Chaotic time series analysis in economics: Balance and perspectives". In: Chaos: An Interdisciplinary Journal of Nonlinear Science 24.4 (Dec. 2014). Publisher: American Institute of Physics, p. 042101. DOI: 10.1063/1.4903797. URL: https://aip.scitation.org/doi/full/10.1063/1.4903797.  
[20] R. J. Field and L. Györgyi. Chaos in Chemistry and Biochemistry. WORLD SCIENTIFIC, 1993. DOI: 10.1142/1706. eprint: https://www.worldscientific.com/doi/pdf/10.1142/1706. URL: https://www.worldscientific.com/doi/abs/10.1142/1706.  
[21] S. Fusi et al. "A neural circuit model of flexible sensorimotor mapping: learning and forgetting on multiple timescales". eng. In: Neuron 54.2 (Apr. 2007), pp. 319-333. DOI: 10.1016/j.neuron.2007.03.017.  
[22] Z. Ghahramani and S. Roweis. "Learning Nonlinear Dynamical Systems Using an EM Algorithm". In: Advances in Neural Information Processing Systems. Ed. by M. Kearns, S. Solla, and D. Cohn. Vol. 11. MIT Press, 1999. URL: https://proceedings.neurips.cc/paper/1998/file/0ebcc77dc72360d0eb8e9504c78d38bd-Paper.pdf.  
[23] W. Gilpin. “Chaos as an interpretable benchmark for forecasting and data-driven modelling”. In: Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). 2021. URL: https://openreview.net/forum?id=enYjtbjYJrf.  
[24] L. Glass and M. C. Mackey. "Pathological conditions resulting from instabilities in physiological control systems". eng. In: Annals of the New York Academy of Sciences 316 (1979), pp. 214-235. DOI: 10.1111/j.1749-6632.1979.tb29471.x.  
[25] P. A. Glendinning and D. J. W. Simpson. "A constructive approach to robust chaos using invariant manifolds and expanding cones". In: Discrete & Continuous Dynamical Systems 41.7 (2021), pp. 3367-3387.  
[26] A. L. Goldberger et al. "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals". In: Circulation 101.23 (2000). Circulation Electronic Pages: http://circ.ahajournals.org/content/101/23/e215.full PMID:1085218; doi: 10.1161/01.CIR.101.23.e215, e215-e220.  
[27] I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. http : / / www . deeplearningbook.org. MIT Press, 2016.  
[28] A. Graves et al. "Hybrid computing using a neural network with dynamic external memory". In: Nature 538 (2016), 471-476. DOI: 10.1038/nature20101.  
[29] R. Hegger, H. Kantz, and T. Schreiber. "Practical implementation of nonlinear time series methods: The TISEAN package". In: Chaos: An Interdisciplinary Journal of Nonlinear Science 9.2 (June 1999). Publisher: American Institute of Physics, pp. 413-435. DOI: 10.1063/1.166424. URL: https://aip.scitation.org/doi/citedby/10.1063/1.166424 (visited on 09/07/2021).  
[30] K. Helfrich, D. Willmott, and Q. Ye. "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform". en. In: International Conference on Machine Learning. ISSN: 2640-3498. PMLR, July 2018, pp. 1969-1978. URL: http://proceedings.mlr.press/v80/helfrich18a.html.  
[31] M. Henaff, A. Szlam, and Y. LeCun. "Recurrent Orthogonal Networks and Long-Memory Tasks". en. In: International Conference on Machine Learning. ISSN: 1938-7228. PMLR, June 2016, pp. 2034-2042. URL: http://proceedings.mlr.press/v48/henaff16.html (visited on 07/21/2021).  
[32] D. Hernandez et al. "Nonlinear Evolution via Spatially-Dependent Linear Dynamics for Electrophysiology and Calcium Data". en. In: arXiv preprint arXiv:1811.02459 (2020). URL: http://arxiv.org/abs/1811.02459 (visited on 07/23/2020).  
[33] S. Hochreiter. Untersuchungen zu dynamischen neuronalen Netzen. 1991.  
[34] S. Hochreiter and J. Schmidhuber. “Long Short-Term Memory”. In: Neural Computation 9.8 (Nov. 1997), pp. 1735–1780. DOI: 10.1162/neco.1997.9.8.1735. URL: https://doi.org/10.1162/neco.1997.9.8.1735 (visited on 07/05/2021).  
[35] K. Inoue et al. "Transient Chaos in BERT". In: arXiv:2106.03181 [nlin] (June 2021). arXiv: 2106.03181. URL: http://arxiv.org/abs/2106.03181.

[36] L. Jing et al. “Gated Orthogonal Recurrent Units: On Learning to Forget”. eng. In: Neural Computation 31.4 (Apr. 2019), pp. 765–783. DOI: 10.1162/neco_a_01174.  
[37] A. Kag and V. Saligrama. "Time Adaptive Recurrent Neural Network". en. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Nashville, TN, USA: IEEE, June 2021, pp. 15144-15153. DOI: 10.1109/CVPR46437.2021.01490. URL: https://ieeexplore.ieee.org/document/9578651/ (visited on 11/14/2021).  
[38] A. Kag and V. Saligrama. "Training Recurrent Neural Networks via Forward Propagation Through Time". In: Proceedings of the 38th International Conference on Machine Learning. Ed. by M. Meila and T. Zhang. Vol. 139. Proceedings of Machine Learning Research. PMLR, 2021, pp. 5189-5200. URL: https://proceedings.mlr.press/v139/kag21a.html.  
[39] A. Kag, Z. Zhang, and V. Saligrama. "RNNs Incrementally Evolving on an Equilibrium Manifold: A Panacea for Vanishing and Exploding Gradients?" In: International Conference on Learning Representations. 2020. URL: https://openreview.net/forum?id=HylpqA4FwS.  
[40] L. Kamdjeu Kengne, J. R. Mboupda Pone, and H. B. Fotsin. "On the dynamics of chaotic circuits based on memristive diode-bridge with variable symmetry: A case study". en. In: Chaos, Solitons & Fractals 145 (Apr. 2021), p. 110795. DOI: 10.1016/j.chaos.2021.110795. URL: https://www.sciencedirect.com/science/article/pii/S0960077921001478.  
[41] H. Kantz and T. Schreiber. Nonlinear Time Series Analysis. 2nd ed. Cambridge University Press, 2003. DOI: 10.1017/CB09780511755798.  
[42] H. Kantz et al. "Nonlinear noise reduction: A case study on experimental data". In: Physical Review E 48.2 (Aug. 1993). Publisher: American Physical Society, pp. 1529-1538. DOI: 10.1103/PhysRevE.48.1529. URL: https://link.aps.org/doi/10.1103/PhysRevE.48.1529 (visited on 09/10/2021).  
[43] M. B. Kennel, R. Brown, and H. D. I. Abarbanel. "Determining embedding dimension for phase-space reconstruction using a geometrical construction". In: Phys. Rev. A 45 (6 1992), pp. 3403-3411. DOI: 10.1103/PhysRevA.45.3403. URL: https://link.aps.org/doi/10.1103/PhysRevA.45.3403.  
[44] G. Kerg et al. "Non-normal Recurrent Neural Network (nnRNN): learning long time dependencies while improving expressivity with transient dynamics". In: Advances in Neural Information Processing Systems. Ed. by H. Wallach et al. Vol. 32. Curran Associates, Inc., 2019. URL: https://proceedings.neurips.cc/paper/2019/file/9d7099d87947faa8d07a272dd6954b80-Paper.pdf.  
[45] M. Kesmia, S. Boughaba, and S. Jacquir. "Control of continuous dynamical systems modeling physiological states". en. In: Chaos, Solitons & Fractals 136 (July 2020), p. 109805. DOI: 10.1016/j.chaos.2020.109805. URL: https://www.sciencedirect.com/science/article/pii/S096007792030206X.  
[46] D. P. Kingma and J. Ba. “Adam: A Method for Stochastic Optimization”. In: 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings. Ed. by Y. Bengio and Y. LeCun. 2015. URL: http://arxiv.org/abs/1412.6980.  
[47] D. P. Kingma and M. Welling. "Auto-Encoding Variational Bayes". en. In: Proceedings of the 2nd International Conference on Learning Representations. 2014. URL: http://arxiv.org/abs/1312.6114 (visited on 01/29/2020).  
[48] J. Z. Kolter and G. Manek. "Learning Stable Deep Dynamics Models". In: Advances in Neural Information Processing Systems. Ed. by H. Wallach et al. Vol. 32. Curran Associates, Inc., 2019. URL: https://proceedings.neurips.cc/paper/2019/file/0a4bbceda17a6253386bc9eb45240e25-Paper.pdf.  
[49] G. Koppe et al. "Identifying nonlinear dynamical systems via generative recurrent neural networks with applications to fMRI". In: PLOS Computational Biology 15.8 (Aug. 2019), pp. 1-35. DOI: 10.1371/journal.pcbi.1007263. URL: https://doi.org/10.1371/journal.pcbi.1007263.  
[50] J. Laskar and P. Robutel. "The chaotic obliquity of the planets". en. In: Nature 361.6413 (Feb. 1993), pp. 608-612. DOI: 10.1038/361608a0. URL: https://www.nature.com/articles/361608a0 (visited on 09/10/2021).

[51] S. Li et al. "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN". In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2018, pp. 5457-5466. DOI: 10.1109/CVPR.2018.00572.  
[52] Z. Li et al. “Fourier Neural Operator for Parametric Partial Differential Equations”. In: International Conference on Learning Representations. 2021. URL: https://openreview.net/forum?id=c8P9NQVtmn0.  
[53] Z. C. Lipton, J. Berkowitz, and C. Elkan. “A Critical Review of Recurrent Neural Networks for Sequence Learning”. In: arXiv:1506.00019 [cs] (Oct. 2015). arXiv: 1506.00019. URL: http://arxiv.org/abs/1506.00019 (visited on 09/12/2021).  
[54] E. N. Lorenz. "Deterministic Nonperiodic Flow". EN. In: Journal of the Atmospheric Sciences 20.2 (Mar. 1963). Publisher: American Meteorological Society Section: Journal of the Atmospheric Sciences, pp. 130-141. DOI: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2. URL: https://journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml.  
[55] N. Maheswaranathan et al. “Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics”. In: Advances in Neural Information Processing Systems. Ed. by H. Wallach et al. Vol. 32. Curran Associates, Inc., 2019. URL: https://proceedings.neurips.cc/paper/2019/file/d921c3c762b1522c475ac8fc0811bb0f-Paper.pdf.  
[56] S. Mangiarotti et al. “Chaos theory applied to the outbreak of COVID-19: an ancillary approach to decision making in pandemic context”. In: Epidemiology and Infection 148 (May 2020), e95. DOI: 10.1017/S0950268820000990. URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7231667/.  
[57] R. M. May. “Chaos and the Dynamics of Biological Populations”. In: Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences 413.1844 (1987), pp. 27-44. URL: http://www.jstor.org/stable/2398225.  
[58] J. D. Meiss. Differential Dynamical Systems. Society for Industrial and Applied Mathematics, 2007.  
[59] J. Menick et al. "Practical Real Time Recurrent Learning with a Sparse Approximation". In: International Conference on Learning Representations. 2021. URL: https://openreview.net/forum?id=q3KSThy2GwB.  
[60] H. Millán, B. Ghanbarian-Alavijeh, and I. García-Fornaris. "Nonlinear dynamics of mean daily temperature and dewpoint time series at Babolsar, Iran, 1961-2005". en. In: Atmospheric Research. Clouds, Aerosols and Radiation 98.1 (Oct. 2010), pp. 89-101. DOI: 10.1016/j.atmosres.2010.06.001. URL: https://www.sciencedirect.com/science/article/pii/S0169809510001419 (visited on 06/12/2021).  
[61] Z. Monfared and D. Durstewitz. "Existence of n-cycles and border-collision bifurcations in piecewise-linear continuous maps with applications to recurrent neural networks". en. In: Nonlinear Dynamics 101.2 (July 2020), pp. 1037-1052. DOI: 10.1007/s11071-020-05841-x. URL: http://link.springer.com/10.1007/s11071-020-05841-x (visited on 07/15/2021).  
[62] Z. Monfared and D. Durstewitz. "Transformation of ReLU-based recurrent neural networks from discrete-time to continuous-time". en. In: International Conference on Machine Learning. ISSN: 2640-3498. PMLR, Nov. 2020, pp. 6999-7009. URL: http://proceedings.mlr. press/v119/monfared20a.html (visited on 07/15/2021).  
[63] L. F. Olsen and H. Degn. “Chaos in an enzyme reaction”. en. In: Nature 267.5607 (May 1977), pp. 177–178. DOI: 10.1038/267177a0. URL: https://www.nature.com/articles/267177a0 (visited on 09/12/2021).  
[64] E. Orhan and X. Pitkow. "Improved memory in recurrent neural networks with sequential non-normal dynamics". In: International Conference on Learning Representations. 2020. URL: https://openreview.net/forum?id=ryx1wRNFvB.  
[65] R. Pascanu, T. Mikolov, and Y. Bengio. "On the Difficulty of Training Recurrent Neural Networks". In: Proceedings of the 30th International Conference on International Conference on Machine Learning - Volume 28. ICML'13. Atlanta, GA, USA: JMLR.org, 2013, III-1310-III-1318.  
[66] A. Paszke et al. "Automatic differentiation in PyTorch". In: 2017. URL: https : // openreview.net/forum?id=BBJsrmfCZ.

[67] B. Pearlmutter. Dynamic recurrent neural networks. 1990. DOI: 10.1184/R1/6605018.v1. URL: https://kilthub.cmu.edu/articles/journal_contribution/Dynamic_recurrent_neural_networks/6605018/1.  
[68] L. Perko. Differential Equations and Dynamical Systems. Vol. 7. Springer, New York, NY, 2001.  
[69] M. T. Rosenstein, J. J. Collins, and C. J. De Luca. "A practical method for calculating largest Lyapunov exponents from small data sets". en. In: Physica D: Nonlinear Phenomena 65.1 (May 1993), pp. 117-134. DOI: 10.1016/0167-2789(93)90009-P. URL: https://www.sciencedirect.com/science/article/pii/016727899390009P (visited on 06/11/2021).  
[70] T. K. Rusch and S. Mishra. “Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies”. In: International Conference on Learning Representations. 2021. URL: https://openreview.net/forum?id=F3s69XzW0ia.  
[71] T. K. Rusch and S. Mishra. "UnICORNN: A recurrent model for learning very long time dependencies". In: Proceedings of the 38th International Conference on Machine Learning. Ed. by M. Meila and T. Zhang. Vol. 139. Proceedings of Machine Learning Research. PMLR, 2021, pp. 9168-9178. URL: https://proceedings.mlr.press/v139/rusch21a.html.  
[72] E. Russo and D. Durstewitz. "Cell assemblies at multiple time scales with arbitrary lag constellations". In: eLife 6 (Jan. 2017). Ed. by M. Howard. Publisher: eLife Sciences Publications, Ltd, e19428. DOI: 10.7554/eLife.19428. URL: https://doi.org/10.7554/eLife.19428 (visited on 09/16/2021).  
[73] O. E. Rössler. “An equation for continuous chaos”. en. In: Physics Letters A 57.5 (July 1976), pp. 397–398. DOI: 10.1016/0375-9601(76)90101-8. URL: https://www.sciencedirect.com/science/article/pii/0375960176901018 (visited on 09/14/2021).  
[74] S. Sabanal and M. Nakagawa. "The fractal properties of vocal sounds and their application in the speech recognition model". en. In: Chaos, Solitons & Fractals 7.11 (Nov. 1996), pp. 1825-1843. DOI: 10.1016/S0960-0779(96)00043-4. URL: https://www.sciencedirect.com/science/article/pii/S0960077996000434.  
[75] T. Sauer, J. A. Yorke, and M. Casdagli. "Embedology". en. In: Journal of Statistical Physics 65.3 (Nov. 1991), pp. 579-616. DOI: 10.1007/BF01053745. URL: https://doi.org/10.1007/BF01053745 (visited on 07/14/2021).  
[76] G. Schalk et al. "BCI2000: a general-purpose brain-computer interface (BCI) system". eng. In: IEEE transactions on bio-medical engineering 51.6 (June 2004), pp. 1034-1043. DOI: 10.1109/ TBME.2004.827072.  
[77] D. Schmidt et al. "Identifying nonlinear dynamical systems with multiple time scales and long-range dependencies". In: International Conference on Learning Representations. 2021. URL: https://openreview.net/forum?id=XYzwxPIQu6.  
[78] B. Sivakumar. “Chaos theory in geophysics: past, present and future”. en. In: Chaos, Solitons & Fractals. Fractals in Geophysics 19.2 (Jan. 2004), pp. 441-462. DOI: 10.1016/S0960-0779(03)00055-9. URL: https://www.sciencedirect.com/science/article/pii/S0960077903000559.  
[79] F. Takens. "Detecting strange attractors in turbulence". In: Dynamical Systems and Turbulence, Warwick 1980. Ed. by D. Rand and L.-S. Young. Berlin, Heidelberg: Springer Berlin Heidelberg, 1981, pp. 366-381.  
[80] R. Tchitnga et al. “A novel hyperchaotic three-component oscillator operating at high frequency”. en. In: Chaos, Solitons & Fractals 118 (Jan. 2019), pp. 166-180. DOI: 10.1016/j.chaos.2018.11.015. URL: https://www.sciencedirect.com/science/article/pii/S0960077918303047.  
[81] D. J. Thomson. “Time series analysis of Holocene climate data”. In: Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences 330.1615 (Apr. 1990). Publisher: Royal Society, pp. 601–616. DOI: 10.1098/rsta.1990.0041. URL: https://royalsocietypublishing.org/doi/abs/10.1098/rsta.1990.0041 (visited on 09/12/2021).  
[82] T. H. Trinh et al. Learning Longer-term Dependencies in RNNs with Auxiliary Losses. 2018. URL: https://openreview.net/forum?id=Hy9xDwyPM.

[83] A. P. Trischler and G. M. D'Eulerio. "Synthesis of recurrent neural networks for dynamical system simulation". en. In: Neural Networks 80 (2016), pp. 67-78. DOI: 10.1016/j.neunet.2016.04.001. URL: https://linkinghub.elsevier.com/retrieve/pii/S0893608016300314 (visited on 03/08/2019).  
[84] P. Turchin and A. D. Taylor. “Complex Dynamics in ecological Time Series”. en. In: Ecology 73.1 (1992), pp. 289–305. DOI: 10.2307/1938740. (Visited on 09/12/2021).  
[85] E. Tziperman et al. “Controlling Spatiotemporal Chaos in a Realistic El Niño Prediction Model”. In: Physical Review Letters 79.6 (Aug. 1997). Publisher: American Physical Society, pp. 1034–1037. DOI: 10.1103/PhysRevLett.79.1034. URL: https://link.aps.org/doi/10.1103/PhysRevLett.79.1034.  
[86] P. R. Vlachas et al. "Data-driven forecasting of high-dimensional chaotic systems with long short-term memory networks". en. In: Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 474.2213 (2018), p. 20170844. DOI: 10.1098/rspa.2017.0844. URL: https://royalsocietypublishing.org/doi/10.1098/rspa.2017.0844 (visited on 11/17/2020).  
[87] P. R. Vlachas et al. "Learning the Effective Dynamics of Complex Multiscale Systems". In: arXiv:2006.13431 [nlin, physics:physics] (July 2020). arXiv: 2006.13431. URL: http://arxiv.org/abs/2006.13431 (visited on 07/19/2021).  
[88] R. Vogt et al. "On Lyapunov Exponents for RNNs: Understanding Information Propagation Using Dynamical Systems Tools". In: Frontiers in Applied Mathematics and Statistics 8 (2022). DOI: 10.3389/fams.2022.818799. URL: https://www.frontiersin.org/article/10.3389/fams.2022.818799.  
[89] H. U. Voss, J. Timmer, and J. Kurths. "Nonlinear dynamical system identification from uncertain and indirect measurements". en. In: International Journal of Bifurcation and Chaos 14.06 (June 2004), pp. 1905-1933. DOI: 10.1142/S0218127404010345. URL: https://www.worldscientific.com/doi/abs/10.1142/S0218127404010345.  
[90] C. van Vreeswijk and H. Sompolinsky. “Chaos in Neuronal Networks with Balanced Excitatory and Inhibitory Activity”. en. In: Science 274.5293 (Dec. 1996). Publisher: American Association for the Advancement of Science Section: Reports, pp. 1724–1726. DOI: 10.1126/science.274.5293.1724. URL: https://science.sciencemag.org/content/274/5293/1724.  
[91] J. H. M. Wedderburn. Lectures on Matrices. New York: American mathematical society, New York : Dover Publications, 1964.  
[92] R. J. Williams and D. Zipser. "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks". en. In: Neural Computation 1.2 (June 1989), pp. 270-280. DOI: 10.1162/neco.1989.1.2.270. URL: https://direct.mit.edu/neco/article/1/2/270-280/5490.  
[93] S. N. Wood. "Statistical inference for noisy nonlinear ecological dynamic systems". en. In: Nature 466.7310 (Aug. 2010). DOI: 10.1038/nature09319. URL: https://www.nature.com/articles/nature09319 (visited on 07/14/2021).  
[94] H. Zhang, D. Liu, and Z. Wang. Controlling Chaos. Springer, London, 2009.

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes]  
(b) Did you describe the limitations of your work? [Yes] See Sec. 5.  
(c) Did you discuss any potential negative societal impacts of your work? [N/A]  
(d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes] We provide comprehensive proofs of all presented theorems in the Appendix.  
(b) Did you include complete proofs of all theoretical results? [Yes]

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] The code will be made public after publication.  
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes]  
(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes]  
(d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [N/A]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]  
(b) Did you mention the license of the assets? [Yes]  
(c) Did you include any new assets either in the supplemental material or as a URL? [No]  
(d) Did you discuss whether and how consent was obtained from people whose data you're using/curating? [N/A]  
(e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]  
(b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]  
(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]