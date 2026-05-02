# Local Identifiability of Deep ReLU Neural Networks: the Theory

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Is a sample rich enough to determine, at least locally, the parameters of a neural network? To answer this question, we introduce a new local parameterization of a given deep ReLU neural network by fixing the values of some of its weights. This allows us to define local lifting operators whose inverses are charts of a smooth manifold of a high dimensional space. The function implemented by the deep ReLU neural network composes the local lifting with a linear operator which depends on the sample. We derive from this convenient representation a geometrical necessary and sufficient condition of local identifiability. Looking at tangent spaces, the geometrical condition provides: 1/ a sharp and testable necessary condition of identifiability and 2/ a sharp and testable sufficient condition of local identifiability. The validity of the conditions can be tested numerically using backpropagation and matrix rank computations.

# 1 Introduction

# 1.1 Context and motivations

Neural networks are famous for their capacity to perform complex tasks in a wide variety of domains such as image classification [18], object recognition [31, 32], speech recognition [15, 34, 14], natural language processing [25, 24, 17], anomaly detection [30] or climate sciences [1].

A question that has recently drawn attention is the question of the identifiability of the parameters of neural networks. This question can be described as follows: for a given architecture and some given inputs  $x^{i}$ , do the responses  $f_{\theta}(x^{i})$  of the network to these inputs uniquely characterize the weight and bias parameters  $\theta$ , up to neuron permutation and positive rescaling of the weights and biases? It is indeed well known, for ReLU networks, that the latter operations on  $\theta$  do not change the function  $f_{\theta}$  [29]. It is therefore impossible, knowing only the  $f_{\theta}(x^{i})$ , to distinguish the elements within the equivalence class of  $\theta$  modulo these operations. Questions that are naturally related to identifiability, and that we do not address in this article, are inverse stability -is the characterization stable to small perturbation?- and stable recovery -are we able to stably recover  $\theta$  (up to equivalence) in practice?

Identifiability is important for different reasons. In the first place, model extraction attacks for neural networks have been a growing topic over the last years. Indeed, some algorithms are able to recover in practice the parameters of a neural network from queries [7, 33]. This can be a concern since neural network providers may wish to keep these parameters secret, for security [19], for privacy [11, 6], or for intellectual property [40].

A way of preventing such a recovery can be by guaranteeing that identifiability does not hold, that is, for a list of requests  $X$ , guaranteeing that  $\theta$  is not uniquely characterized by the answers  $f_{\theta}(X)$  to these requests. To do so, one needs to check that a necessary condition of identifiability is not met.

On the opposite side, guaranteeing that identifiability holds is interesting in the position of an attacker. If the attacker has access to  $X$ , to  $f_{\theta}(X)$ , and is able to compute a  $\tilde{\theta}$  such that  $f_{\tilde{\theta}}(X) = f_{\theta}(X)$ , the question then becomes: does this guarantee that  $\tilde{\theta} \sim \theta$  or shall the attacker expand  $X$  with new queries? The attacker needs a sufficient condition of identifiability.

In these examples  $X$  represents requests to an already trained network. We can think of another context in which  $X$  represents a training database or a test set. In the former case, optimizing the empirical risk may be difficult in general and the model may generalize poorly. However, if successful, that is if the trained network matches the samples, a sufficient condition of identifiability can guarantee that the function implemented by the network only depends on its values at these samples. In particular, it does not depend on the choice of the optimizer, on its initialization or on stochastic parameters.

# 1.2 Existing work on identifiability, inverse stability and stable recovery

Even though it has regained interest recently, the question of identifiability for neural networks is not new. Indeed, in the 1990s, interesting results on identifiability of networks with smooth activation functions (tanh, logistic sigmoid, Gaussian...) have been established [38, 2, 20, 16, 10].

When it comes to shallow [28, 36] as well as deep [29, 4] ReLU neural networks, some results have been recently established. They show that under some conditions, the function implemented by the network uniquely characterizes its parameters, up to neuron permutation and rescaling operations.

All these results assume the function implemented by the network to be known on the whole input space, or at least on an open subset of it. As far as we know, there exists only one identifiability result for deep ReLU networks assuming the knowledge of this function on a finite sample. Stock and Gribonval [37] give a theoretical condition for the existence of a finite set which locally identifies the parameters of a deep neural network. The construction in [37] shares similarities with previous works on deep structured matrix factorization [22]. The present article lies in this line of research.

Closely related to identifiability are the topics of inverse stability and stable recovery of the parameters of a network. Some negative [27] as well as positive [9, 21, 22, 23] results of inverse stability exist for ReLU and identity activation functions. Several stable recovery algorithms have also been proposed, for shallow networks in a first place, for smooth activation function [12], as well as ReLU in the fully-connected case [13, 42, 43, 44] or in the convolutional case [5, 41]. These references provide a large sample complexity under which minimizing the empirical risk allows to recover the parameters of the network.

For deep networks, some stable recovery algorithms exist, for instance for Heavyside activation function [3], or for the first layer with sparsity assumptions [35] in the ReLU case. For deep ReLU networks, when one has full access to the function implemented by the network, a practical algorithm [33] sequentially constructs a sample and approximately recovers the architecture and the parameters modulo permutation and rescaling. Similarly, formulating the problem as a cryptanalytic problem, [7] reconstructs a functionally equivalent network with fewer requests.

# 1.3 Contributions

1/ We establish a necessary and sufficient geometrical condition of local identifiability from a finite sample  $X$  for deep fully-connected ReLU networks. The condition is that the intersection between a smooth manifold and an affine space is reduced to a single point. 2/ Considering tangent spaces, we then provide a computable necessary condition of local identifiability which, since global identifiability implies local identifiability, is also a computable necessary condition of identifiability.  
3/ We also establish a computable sufficient condition of local identifiability, which is close to the necessary condition. To the best of our knowledge, these are the first testable conditions of local identifiability for any finite input sample. In particular, [37] provides a theoretical condition equivalent to the existence of a finite sample for which local identifiability holds. The existence of the sample is proved in a constructive manner. The authors do not characterize local identifiability for any given sample.  
4/ To prove these results, we develop geometrical tools which can be of independent interest for theoretically understanding deep ReLU networks as well as for possible applications. Namely, we introduce local reparameterizations  $\rho_{\theta}$  of the network by fixing some weight values as constants.

Building on these local parameterizations, we introduce local lifting operators  $\psi^{\theta}$  and we decompose the function implemented by the network  $f_{\theta}(x)$  as a composition of  $\psi^{\theta}$ , which only depends on the parameters, and a piecewise constant operator  $\alpha$  which depends on  $\theta$  and the inputs  $x^{i}$ . For almost any parameterization  $\theta$ , the operator  $\alpha$  is constant in a neighborhood of  $\theta$  and consists in applying a linear function to  $\psi^{\theta}$ . We show that in fact, the operators  $\psi^{\theta}$  are the inverses of coordinate charts of a smooth manifold  $\Sigma_1^*$ , contained in a high dimensional space. We find  $\Sigma_1^*$  to be of particular interest in representing geometrically some properties of the network parameters (in particular to establish 1/2/ and 3/ above).

# 1.4 Overview of the article

This work is structured as follows. We start by introducing basic tools and already known results in Section 2. We then introduce the local parameterizations  $\rho_{\theta}$  and the set  $\Sigma_1^*$ , and we show that it is a smooth manifold in Section 3. This allows us to state our main results in Section 4, that is the geometrical and the numerically testable conditions of local identifiability. Finally we discuss in Section 5 the numerical computations needed to test the latter conditions. All the proofs are provided in the supplementary material.

# 2 ReLU networks, lifting operator and rescaling of the parameters

# 2.1 ReLU networks

Let us introduce our notations for deep fully-connected ReLU networks. In this paper, a network is a graph  $(E,V)$  of the following form.

-  $V$  is a set of neurons, which is divided in  $L + 1$  layers, with  $L \geq 2$ :  $V = (V_l)_{l \in [0, L]}$ .

$V_{0}$  is the input layer,  $V_{L}$  the output layer and the layers  $V_{l}$  with  $1 \leq l \leq L - 1$  are the hidden layers. Using the notation  $|C|$  for the cardinal of a finite set  $C$ , we denote, for all  $l \in [0, L]$ ,  $N_{l} = |V_{l}|$  the size of the layer  $V_{l}$ .

-  $E$  is the set of all oriented edges  $v \to v'$  between neurons in consecutive layers, that is

$$
E = \{v \rightarrow v ^ {\prime}, v \in V _ {l}, v ^ {\prime} \in V _ {l + 1}, \text {f o r} l \in [   [ 0, L - 1 ]   ] \}.
$$

A network is parameterized by weights and biases, gathered in its parameterization  $\theta$ , with

$$
\theta = \left(\left(w _ {v \rightarrow v ^ {\prime}}\right) _ {v \rightarrow v ^ {\prime} \in E}, \left(b _ {v}\right) _ {v \in B}\right) \quad \in \mathbb {R} ^ {E} \times \mathbb {R} ^ {B},
$$

where  $B = \bigcup_{l=1}^{L} V_{l}$ . It is also convenient to consider the weights and biases in matrix/vector form: for a given  $\theta$ , we denote, for  $l \in [[1, L]]$ ,

$$
W _ {l} = \left(w _ {v \rightarrow v ^ {\prime}}\right) _ {v ^ {\prime} \in V _ {l}, v \in V _ {l - 1}} \in \mathbb {R} ^ {N _ {l} \times N _ {l - 1}} \quad \text {a n d} \quad b _ {l} = \left(b _ {v}\right) _ {v \in V _ {l}} \in \mathbb {R} ^ {N _ {l}}.
$$

When dealing with two parameterizations  $\theta$  and  $\widetilde{\theta} \in \mathbb{R}^{E} \times \mathbb{R}^{B}$ , we take as a convention that  $w_{v \to v'}$  and  $b_{v}$  as well as  $W_{l}$  and  $b_{l}$  denote the weights and biases associated to  $\theta$ , and  $\tilde{w}_{v \to v'}$  and  $\tilde{b}_{v}$  as well as  $\widetilde{W_{l}}$  and  $\tilde{b}_{l}$  denote those associated to  $\widetilde{\theta}$ .

The activation function, denoted  $\sigma$ , is always  $\mathrm{ReLU}$ : for any  $p \in \mathbb{N}^*$  and any vector  $x = (x_1, \ldots, x_p)^T \in \mathbb{R}^p$ , it is defined as  $\sigma(x) = (\max(x_1, 0), \ldots, \max(x_p, 0))^T$ .

For a given  $\theta$ , we define recursively  $f_{l}:\mathbb{R}^{V_{0}}\to \mathbb{R}^{V_{l}}$  (we omit the dependency in  $\theta$  in the notation for simplicity), for  $l\in [[0,L]]$ , by

-  $\forall x\in \mathbb{R}^{V_0}$ $f_{0}(x) = x$  
-  $\forall l\in [[1,L - 1]]$ $\forall x\in \mathbb{R}^{V_0}$ $f_{l}(x) = \sigma (W_{l}f_{l - 1}(x) + b_{l})$  
-  $\forall x\in \mathbb{R}^{V_0}$ $f_{L}(x) = W_{L}f_{L - 1}(x) + b_{L}$

We define  $f_{\theta} : \mathbb{R}^{V_0} \to \mathbb{R}^{V_L}$  as  $f_{\theta} = f_{L}$  and we refer to it as the function implemented by the network of parameter  $\theta$ .

# 2.2 The lifting operator  $\phi$

For all  $l \in [[1, L - 1]]$ , for all  $v \in V_l$ , we denote, for all  $\theta \in \mathbb{R}^E \times \mathbb{R}^B$  and  $x \in \mathbb{R}^{V_0}$

$$
a _ {v} (x, \theta) = \left\{ \begin{array}{l l} 1 & \text {i f} f _ {l} (x) _ {v} \geq 0 \\ 0 & \text {o t h e r w i s e ,} \end{array} \right.
$$

which is the activation indicator of neuron  $v$ . For all  $l \in [0, L - 1]$ , we define

$$
\mathcal {P} _ {l} = V _ {l} \times \dots \times V _ {L - 1},
$$

which is the set of all paths in the network starting from layer  $l$  and ending in layer  $L - 1$ . We consider an additional element  $\beta$  which can be interpreted as an empty path and whose role will be clear once  $\phi$  has been defined and Proposition 1 stated. We define

$$
\mathcal {P} = \left(\bigcup_ {l = 0} ^ {L - 1} \mathcal {P} _ {l}\right) \cup \{\beta \}.
$$

In a similar way to [37], we define a 'lifting operator'

$$
\begin{array}{r c l} \phi : & \mathbb {R} ^ {E} \times \mathbb {R} ^ {B} & \longrightarrow & \mathbb {R} ^ {\mathcal {P} \times V _ {L}} \\ & \theta & \longmapsto & \left(\phi_ {p, v} (\theta)\right) _ {p \in \mathcal {P}, v \in V _ {L}} \end{array} \tag {1}
$$

by:

- for all  $l \in [[0, L - 1]]$  and all  $p = (v_{l}, \dots, v_{L - 1}) \in \mathcal{P}_{l}$ , and for all  $v_{L} \in V_{L}$ ,

$$
\phi_ {p, v _ {L}} (\theta) = \left\{ \begin{array}{l l} \prod_ {l ^ {\prime} = 0} ^ {L - 1} w _ {v _ {l ^ {\prime}} \to v _ {l ^ {\prime} + 1}} & \text {i f} l = 0 \\ b _ {v _ {l}} \prod_ {l ^ {\prime} = l} ^ {L - 1} w _ {v _ {l ^ {\prime}} \to v _ {l ^ {\prime} + 1}} & \text {i f} l \geq 1; \end{array} \right.
$$

for  $p = \beta$  and  $v_{L}\in V_{L}$ $\phi_{\beta ,v_L}(\theta) = b_{v_L}$

We now define the 'activation operator'

$$
\begin{array}{r c l} \alpha : & \mathbb {R} ^ {V _ {0}} \times \left(\mathbb {R} ^ {E} \times \mathbb {R} ^ {B}\right) & \longrightarrow & \mathbb {R} ^ {1 \times \mathcal {P}} \\ & (x, \theta) & \longmapsto & (\alpha_ {p} (x, \theta)) _ {p \in \mathcal {P}} \end{array} \tag {2}
$$

by:

- for all  $l \in [[0, L - 1]]$  and all  $p = (v_{l}, \dots, v_{L - 1}) \in \mathcal{P}_{l}$ :

$$
\alpha_ {p} (x, \theta) = \left\{ \begin{array}{l l} x _ {v _ {0}} \prod_ {l ^ {\prime} = 1} ^ {L - 1} a _ {v _ {l ^ {\prime}}} (x, \theta) & \text {i f} l = 0 \\ \prod_ {l ^ {\prime} = l} ^ {L - 1} a _ {v _ {l ^ {\prime}}} (x, \theta) & \text {i f} l \geq 1; \end{array} \right.
$$

for  $p = \beta$ $\alpha_{\beta}(x,\theta) = 1$

We have the following decomposition of the function  $f_{\theta}$  implemented by the network.

Proposition 1. For all  $\theta \in \mathbb{R}^E\times \mathbb{R}^B$  and all  $x\in \mathbb{R}^{V_0}$

$$
f _ {\theta} (x) ^ {T} = \alpha (x, \theta) \phi (\theta).
$$

This result, which is proven in the supplement, is also stated in [37, Sec. 4] with slightly different notations. To describe the interest of Proposition 1, let us anticipate on Proposition 2, which states that  $\theta \mapsto \alpha(x,\theta)$  is piecewise constant. When  $x$  is fixed, on a piece where  $\alpha(x,\theta)$  is constant and therefore independent of  $\theta$ , Proposition 1 expresses the map  $\theta \mapsto f_{\theta}(x)$  as the composition of a fixed linear operator and a polynomial lifting operator. This allows to decompose the complex map  $\theta \mapsto f_{\theta}(x)$  in two simpler 'bricks'.

Let us reformulate Proposition 1 with several inputs. We consider, for some  $n \in \mathbb{N}^*$ , some given inputs  $x^i \in \mathbb{R}^{V_0}$ , with  $i \in [1, n]$ . We denote by  $X \in \mathbb{R}^{n \times V_0}$  the matrix whose lines are the transpose  $(x^i)^T$  of the inputs. For all  $\theta \in \mathbb{R}^E \times \mathbb{R}^B$ , we denote by  $f_\theta(X) \in \mathbb{R}^{n \times V_L}$  the matrix whose lines are the transpose  $f_\theta(x^i)^T$  of the outputs, for all  $i \in [1, n]$ . We also denote by  $\alpha(X, \theta) \in \mathbb{R}^{n \times P}$  the matrix whose lines are the line vectors  $\alpha(x^i, \theta)$ , for all  $i \in [1, n]$ . Using Proposition 1 for all the  $x^i$ , we have the relation

$$
f _ {\theta} (X) = \alpha (X, \theta) \phi (\theta). \tag {3}
$$

We prove in the supplement the following proposition.

Proposition 2. For all  $n \in \mathbb{N}^*$ , for all  $X \in \mathbb{R}^{n \times V_0}$ , the mapping

$$
\begin{array}{c c c c} \alpha_ {X}: & \mathbb {R} ^ {E} \times \mathbb {R} ^ {B} & \longrightarrow & \mathbb {R} ^ {n \times \mathcal {P}} \\ & \theta & \longmapsto & \alpha (X, \theta) \end{array}
$$

is piecewise-constant, with a finite number of pieces. Furthermore, the boundary of each piece has Lebesgue measure zero. We call  $\Delta_X$  the union of all these boundaries. The set  $\Delta_X$  is closed and has Lebesgue measure zero.

As discussed before, for a given  $X \in \mathbb{R}^{n \times V_0}$ , when studying the function  $\theta \mapsto f_{\theta}(X)$ , Proposition 2 alongside (3) shows that on a piece over which  $\alpha_X$  is constant,  $f_{\theta}(X)$  depends linearly on  $\phi(\theta)$ . Since  $\Delta_X$  is closed with measure zero, for almost all  $\theta \in \mathbb{R}^E \times \mathbb{R}^B$ , there exists a neighborhood of  $\theta$  over which  $\alpha_X$  is constant. As noted for instance by Stock and Gribonval [37, Sec. 2], for any  $\tilde{\theta}$  in such a neighborhood, we thus have

$$
f _ {\theta} (X) - f _ {\tilde {\theta}} (X) = \alpha (X, \theta) (\phi (\theta) - \phi (\tilde {\theta})). \tag {4}
$$

Hence, studying  $\phi$  will allow us to understand better how  $f_{\theta}(X)$  locally depends on  $\theta$ .

# 2.3 Invariant rescaling operations on  $\theta$

Some well-known rescaling operations on the parameters  $\theta$  do not affect the value of  $\phi(\theta)$ . Before detailing them, let us define, for all  $t \in \mathbb{R}$ , the sign indicator  $\mathrm{sign}(t)$  as  $1, 0$  or  $-1$  depending on whether  $t > 0$ ,  $t = 0$  or  $t < 0$  respectively. For any  $\theta \in \mathbb{R}^E \times \mathbb{R}^B$ , we then define

$$
\operatorname {s i g n} (\theta) = \left( \right.\left(\operatorname {s i g n} \left(w _ {v \rightarrow v ^ {\prime}}\right) _ {v \rightarrow v ^ {\prime} \in E}, \left(\operatorname {s i g n} \left(b _ {v}\right)\right) _ {v \in B}\right) \in \{- 1, 0, 1 \} ^ {E} \times \{- 1, 0, 1 \} ^ {B}.
$$

We can now describe the rescaling operations.

Definition 3. Let  $\theta \in \mathbb{R}^E\times \mathbb{R}^B$  and  $\tilde{\theta}\in \mathbb{R}^{E}\times \mathbb{R}^{B}$

- We say that  $\theta$  is equivalent to  $\tilde{\theta}$  modulo rescaling, and we write  $\theta \stackrel{R}{\sim} \tilde{\theta}$  iff there exists a family of vectors  $(\lambda^0, \dots, \lambda^L) \in (\mathbb{R}^*)^{V_0} \times \dots \times (\mathbb{R}^*)^{V_L}$ , with  $\lambda^0 = \mathbb{1}_{V_0}$  and  $\lambda^L = \mathbb{1}_{V_L}$ , such that, for all  $l \in [1, L]$ ,

$$
\left\{ \begin{array}{l} W _ {l} = \operatorname {D i a g} \left(\lambda^ {l}\right) \widetilde {W _ {l}} \operatorname {D i a g} \left(\lambda^ {l - 1}\right) ^ {- 1} \\ b _ {l} = \operatorname {D i a g} \left(\lambda^ {l}\right) \tilde {b} _ {l}. \end{array} \right. \tag {5}
$$

- We say that  $\theta$  is equivalent to  $\tilde{\theta}$  modulo positive rescaling, and we write  $\theta \sim \tilde{\theta}$  iff

$$
\theta \stackrel {{R}} {{\sim}} \tilde {\theta} \quad \text {a n d} \quad \operatorname {s i g n} (\theta) = \operatorname {s i g n} (\tilde {\theta}).
$$

For all  $l \in [[1, L]]$ , to satisfy (5) is equivalent to satisfy, for all  $(v_{l-1}, v_l) \in V_{l-1} \times V_l$ ,

$$
\left\{\begin{array}{l}w _ {v _ {l - 1} \rightarrow v _ {l}} = \frac {\lambda_ {v _ {l}} ^ {l}}{\lambda_ {v _ {l - 1}} ^ {l - 1}} \tilde {w} _ {v _ {l - 1} \rightarrow v _ {l}}.\\b _ {v _ {l}} = \lambda_ {v _ {l}} ^ {l} \tilde {b} _ {v _ {l}}\end{array}\right. \tag {6}
$$

The relations  $\stackrel{R}{\sim}$  and  $\sim$  are equivalence relations on the set of parameters  $\mathbb{R}^E \times \mathbb{R}^B$ . The equivalence modulo positive rescaling  $\sim$  is a well-known invariant for ReLU networks [36, 37, 4, 26, 39]. We have indeed the following property: if  $\theta \sim \tilde{\theta}$ , for all  $x \in \mathbb{R}^{V_0}$ ,

$$
f _ {\theta} (x) = f _ {\bar {\theta}} (x). \tag {7}
$$

One of the interests of the operator  $\phi$  is that it captures this invariant, as described by Stock and Gribonval [37, Sec. 2.4]. Propositions 4 and 5 are similar to their results and are restated here and proven in the supplement for completeness. Indeed, combining the definition of  $\phi$  with (6), we have the following property.

Proposition 4. For all  $\theta, \tilde{\theta} \in \mathbb{R}^E \times \mathbb{R}^B$ , we have

$$
\theta \stackrel {R} {\sim} \tilde {\theta} \quad \Longrightarrow \quad \phi (\theta) = \phi (\tilde {\theta}),
$$

and thus in particular

$$
\theta \sim \tilde {\theta} \quad \Longrightarrow \quad \phi (\theta) = \phi (\tilde {\theta}).
$$

![](images/123eaca0195628919afce7058fab1b6c06eabc92b8785a76c3c369d47b5b85dc.jpg)  
Figure 1: Left: The outward edges of a hidden neuron  $v$  and their weights. In this example,  $v_{1} = s_{\max}^{\theta}(v)$ , so the weight of the edge in red,  $v \rightarrow v_{1}$ , has its value fixed as  $w_{v \rightarrow v_{1}}$ . The weights of the remaining edges,  $\tau_{v \rightarrow v_{2}}$  and  $\tau_{v \rightarrow v_{3}}$ , are free to vary. Right: In red, all the edges whose weights are fixed. The remaining edges, in black, constitute the set  $F_{\theta}$ .

![](images/c85da850e28b12a3486e043617276252bbd33737523dd50360641d28e6b0f800.jpg)

The reciprocal of Proposition 4 holds provided we exclude some degenerate cases. Let us denote, for any  $l \in [[1, L - 1]]$  and any  $v \in V_l$ , by  $w_{\bullet \to v}$  the vector  $(w_{v' \to v})_{v' \in V_{l-1}} \in \mathbb{R}^{V_{l-1}}$  and by  $w_{v \to \bullet}$  the vector  $(w_{v \to v'})_{v' \in V_{l+1}} \in \mathbb{R}^{V_{l+1}}$ . We define the following set, which is close to the notion of 'non admissible parameter' in [37].

$$
S = \{\theta \in \mathbb {R} ^ {E} \times \mathbb {R} ^ {B}, \exists v \in V _ {1} \cup \dots \cup V _ {L - 1}, w _ {v \rightarrow \bullet} = 0 \mathrm {o r} (w _ {\bullet \rightarrow v}, b _ {v}) = (0, 0) \}.
$$

A parameterization  $\theta$  belongs to  $S$  iff there exists a hidden neuron  $v\in V_1\cup \dots \cup V_{L - 1}$  such that  $(w_{\bullet \rightarrow v},b_v) = (0,0)$  or  $w_{v\to \bullet} = 0$ . In the first case, all the inward weights and the bias of  $v$  are zero, so for any input the information flowing through neuron  $v$  is always zero. In the second case, all the outward weights of  $v$  are zero. In both cases, neuron  $v$  does not contribute to the output and could be removed from the network without changing the function  $f_{\theta}$ .

Since it is composed of a finite union of linear subspaces of codimension larger than 1, defined by linear equations, the set  $S$  is closed and has Lebesgue measure zero, so we can exclude the degenerate cases in  $S$  without loss of generality. Proposition 5 states that the reciprocal of Proposition 4 holds over  $\left(\mathbb{R}^E \times \mathbb{R}^B\right) \backslash S$ .

Proposition 5. For all  $\theta \in (\mathbb{R}^E\times \mathbb{R}^B)\backslash S$  , for all  $\tilde{\theta}\in \mathbb{R}^{E}\times \mathbb{R}^{B}$

$$
\phi (\theta) = \phi (\tilde {\theta}) \quad \Longrightarrow \quad \theta \stackrel {R} {\sim} \tilde {\theta}.
$$

# 3 The smooth manifold  $\Sigma_1^*$

We explained in the previous section that studying  $\phi$  allows to better understand how the output  $f_{\theta}(X)$  locally depends on  $\theta$ . The image of  $\phi$  is of particular interest in this study and is the subject of this section. We define

$$
\Sigma_ {1} ^ {*} = \{\phi (\theta), \theta \in \left(\mathbb {R} ^ {E} \times \mathbb {R} ^ {B}\right) \backslash S \}.
$$

The main result of this section, Theorem 6, states that  $\Sigma_1^*$  is a smooth manifold. This is a key element of the proofs of Theorems 7, 8 and 9.

As explained in Section 2.3, the positive rescaling operations on  $\theta$  described by the relation  $\sim$  do not affect the value of  $\phi(\theta)$  or of  $f_{\theta}(x)$  for any input  $x$ . This creates some degrees of freedom in the parameterization of a network without changing its output. To suppress these degrees of freedom, we propose to reduce the number of parameters by fixing locally the values of some weights as constants. More precisely, for a given  $\theta$ , for each neuron  $v$  in a hidden layer, we choose the outward edge  $v \to v'$  whose weight  $w_{v \to v'}$  has largest (absolute) value, and we consider its value to be fixed from now on (if there are several such edges, we choose one arbitrarily). We denote by  $s_{\max}^{\theta}(v)$  such a neuron  $v'$ . For each neuron  $v$  in a hidden layer  $V_l$ , there is exactly one neuron  $s_{\max}^{\theta}(v)$  in the layer  $V_{l+1}$ , and one corresponding edge  $v \to s_{\max}^{\theta}(v)$  whose weight is fixed. See Figure 1 for an illustration.

214 We denote by  $F_{\theta} \subset E$  the set of remaining edges, which is formally defined as<sup>1</sup>

$$
F _ {\theta} = E \backslash \left(\bigcup_ {l = 1} ^ {L - 1} \left\{(v, s _ {\max } ^ {\theta} (v)), v \in V _ {l} \right\}\right), \tag {8}
$$

and we take as new parameters the weights in  $F_{\theta}$  and the biases  $B$ . This is formalized by the following application, for  $\theta \in (\mathbb{R}^E\times \mathbb{R}^B)\backslash S$

$$
\rho_ {\theta}: \mathbb {R} ^ {F _ {\theta}} \times \mathbb {R} ^ {B} \longrightarrow \mathbb {R} ^ {E} \times \mathbb {R} ^ {B}
$$

$$
\tau \quad \longmapsto \quad \tilde {\theta} \quad \text {s u c h t h a t} \left\{\begin{array}{l}\forall (v, v ^ {\prime}) \in F _ {\theta}, \quad \tilde {w} _ {v \rightarrow v ^ {\prime}} = \tau_ {v \rightarrow v ^ {\prime}}\\\forall (v, v ^ {\prime}) \in E \backslash F _ {\theta}, \quad \tilde {w} _ {v \rightarrow v ^ {\prime}} = w _ {v \rightarrow v ^ {\prime}}\\\forall v \in B, \quad \tilde {b} _ {v} = \tau_ {v}.\end{array}\right. \tag {9}
$$

In particular, if we define  $\tau_{\theta} \in \mathbb{R}^{F_{\theta}} \times \mathbb{R}^{B}$  by  $(\tau_{\theta})_{v \to v'} = w_{v \to v'}$  and  $(\tau_{\theta})_v = b_v$ , we have  $\rho_{\theta}(\tau_{\theta}) = \theta$ . The function  $\rho_{\theta}$  is affine and injective. We define

$$
U _ {\theta} = \rho_ {\theta} ^ {- 1} \left(\left(\mathbb {R} ^ {E} \times \mathbb {R} ^ {B}\right) \backslash S\right), \tag {10}
$$

which is an open set of  $\mathbb{R}^{F_{\theta}}\times \mathbb{R}^{B}$ . We define, for all  $\theta \in \left(\mathbb{R}^{E}\times \mathbb{R}^{B}\right)\backslash S$ , the local lifting operator

$$
\begin{array}{r c l} \psi^ {\theta}: & U _ {\theta} & \longrightarrow & \mathbb {R} ^ {\mathcal {P} \times V _ {L}} \\ & \tau & \longmapsto & \phi \circ \rho_ {\theta} (\tau). \end{array} \tag {11}
$$

One can show that  $\psi^{\theta}$  is  $C^\infty$  and that it is a homeomorphism from  $U_{\theta}$  onto its image (see the proofs in the supplement), which we denote  $V_{\theta}$  and is thus an open subset of  $\Sigma_1^*$  (with the topology induced on  $\Sigma_1^*$  by the standard topology on  $\mathbb{R}^{\mathcal{P}\times V_L}$ ). In particular, since  $\rho_{\theta}(\tau_{\theta}) = \theta$ , we have  $\phi (\theta) = \psi^{\theta}(\tau_{\theta})\in V_{\theta}$ . We have the following fundamental result.  
224 Theorem 6.  $\Sigma_1^*$  is a smooth manifold of  $\mathbb{R}^{\mathcal{P}\times V_L}$  of dimension

$$
\left| F _ {\theta} \right| + \left| B \right| = N _ {0} N _ {1} + N _ {1} N _ {2} + \dots + N _ {L - 1} N _ {L} + N _ {L},
$$

225 and the family  $(V_{\theta},(\psi^{\theta})^{-1})_{\theta \in (\mathbb{R}^{E}\times \mathbb{R}^{B})\setminus S}$  is an atlas.

Theorem 6 is proven in the supplement. Besides being key in Section 4, Theorem 6 (both the smooth manifold nature of  $\Sigma_1^*$  and the explicit atlas  $(V_{\theta},(\psi^{\theta})^{-1})_{\theta \in (\mathbb{R}^{E}\times \mathbb{R}^{B})\setminus S})$  may also be considered of more general independent interest. To our knowledge, such a result has not been established elsewhere in the literature.

# 230 4 Main results: necessary and sufficient conditions for local identifiability

The main results of this paper rely on the decomposition (4) introduced in Section 2. To reformulate (4), let us introduce the linear operator  $A(X,\theta)$ , which simply corresponds to the matrix product with  $\alpha(X,\theta)$ :

$$
\begin{array}{r c l} A (X, \theta): & \mathbb {R} ^ {\mathcal {P} \times V _ {L}} & \longrightarrow & \mathbb {R} ^ {n \times V _ {L}} \\ & \eta & \longmapsto & \alpha (X, \theta) \eta , \end{array}
$$

where  $\alpha (X,\theta)\eta$  is the matrix product between  $\alpha (X,\theta)\in \mathbb{R}^{n\times \mathcal{P}}$  and  $\eta \in \mathbb{R}^{\mathcal{P}\times V_L}$ . The operator  $A(X,\theta)$  inherits the properties of  $\alpha (X,\theta)$ , in particular those stated in Proposition 2. Using  $A(X,\theta)$ , the relation (4) satisfies by  $\tilde{\theta}$  in the neighborhood of  $\theta$  becomes

$$
f _ {\theta} (X) - f _ {\tilde {\theta}} (X) = A (X, \theta) \cdot (\phi (\theta) - \phi (\tilde {\theta})). \tag {12}
$$

237 Let us also define the affine space

$$
N (X, \theta) = \phi (\theta) + \operatorname {K e r} A (X, \theta).
$$

If a parameterization  $\tilde{\theta} \in \mathbb{R}^E \times \mathbb{R}^B$  is such that  $f_{\tilde{\theta}}(X) = f_{\theta}(X)$  and (12) holds, then  $\phi(\theta) - \phi(\tilde{\theta}) \in \mathrm{Ker} A(X, \theta)$ , so by definition  $\phi(\tilde{\theta}) \in N(X, \theta)$ . Since for  $\tilde{\theta}$  in the neighborhood of  $\theta$ , the image  $\phi(\tilde{\theta})$

![](images/2592217735662bb747720dd0f39cff433868e58568f19cc8cc2b385005c63842.jpg)  
Figure 2: The local intersection between  $N(X, \theta)$  (in green) and  $\Sigma_1^*$  (color gradient). We also represent in red the tangent space to  $\Sigma_1^*$  at  $\phi(\theta)$ . Left: The identifiable case. The intersection is reduced to  $\{\phi(\theta)\}$ . Right: The non-identifiable case. The intersection, represented with a dashed white line, is not reduced to  $\{\phi(\theta)\}$ .

![](images/b8118d2499e77fb519b7ccba13651ed324d510380a7fae341cdf8fa0ea0b7b46.jpg)

belongs to  $\Sigma_1^*$ , this shows that local identifiability is closely related to the nature of the intersection between the smooth manifold  $\Sigma_1^*$  and the affine subspace  $N(X,\theta)$ .  
Indeed, let us denote by  $B_{\infty}(\phi (\theta),\epsilon^{\prime}) = \{\eta \in \mathbb{R}^{\mathcal{P}\times V_L},\| \phi (\theta) - \eta \|_{\infty} <   \epsilon '\}$  the ball of center  $\phi (\theta)$  and of radius  $\epsilon^{\prime} > 0$  . We have the following geometrical necessary and sufficient condition of local identifiability, which states that local identifiability of  $\theta$  holds if and only if the intersection between  $\Sigma_1^*$  and  $N(X,\theta)$  is locally reduced to the single point  $\{\phi (\theta)\}$  
Theorem 7. For any  $\theta \in (\mathbb{R}^E\times \mathbb{R}^B)\backslash (S\cup \Delta_X)$ , the two following statements are equivalent.

i) There exists  $\epsilon > 0$  such that for any  $\tilde{\theta} \in \mathbb{R}^E \times \mathbb{R}^B$ , if  $\| \theta - \tilde{\theta} \|_{\infty} < \epsilon$ , then

$$
f _ {\theta} (X) = f _ {\tilde {\theta}} (X) \quad \Longrightarrow \quad \theta \sim \tilde {\theta}.
$$

ii) There exists  $\epsilon' > 0$  such that

$$
B _ {\infty} \left(\phi (\theta), \epsilon^ {\prime}\right) \cap \Sigma_ {1} ^ {*} \cap N (X, \theta) = \{\phi (\theta) \}.
$$

Theorem 7 is proven in the supplement, and is illustrated in Figure 2. This geometrical condition is crucial for showing the next two results which give testable conditions of identifiability. Theorems 8 and 9 rely on the rank of  $A(X,\theta)$  and of another linear operator  $\Gamma (X,\theta)$ , which we now define. Since, as we said, the function  $\psi^{\theta}$  is  $C^\infty$ , let us denote by  $D\psi^{\theta}(\tau):\mathbb{R}^{F_{\theta}}\times \mathbb{R}^{B}\to \mathbb{R}^{\mathcal{P}\times V_{L}}$  its differential at the point  $\tau$ , for any  $\tau \in U_{\theta}$ . We define the linear operator  $\Gamma (X,\theta):\mathbb{R}^{F_{\theta}}\times \mathbb{R}^{B}\to \mathbb{R}^{n\times V_{L}}$  by

$$
\Gamma (X, \theta) = A (X, \theta) \circ D \psi^ {\theta} (\tau_ {\theta}). \tag {13}
$$

We denote  $R_{A} = \mathrm{rank}(A(X,\theta))$  and  $R_{\Gamma} = \mathrm{rank}(\Gamma (X,\theta))$ . Since  $\Gamma (X,\theta)$  is defined on  $\mathbb{R}^{F_{\theta}}\times \mathbb{R}^{B}$ , we have

$$
0 \leq R _ {\Gamma} \leq | F _ {\theta} | + | B |,
$$

256 and the expression (13) shows that we also have

$$
0 \leq R _ {\Gamma} \leq R _ {A} \leq | \mathcal {P} | N _ {L}.
$$

257 We can now define the two following conditions.

258 Condition  $C_N$ . Condition  $C_N$  is satisfied by  $(\theta, X)$  if  $R_{\Gamma} < R_A$  or  $R_{\Gamma} = |F_{\theta}| + |B|$ .

259 Condition  $C_S$ . Condition  $C_S$  is satisfied by  $(\theta, X)$  if  $R_{\Gamma} = |F_{\theta}| + |B|$ .

The following result states that  $C_N$  is a necessary condition for local and therefore global identifiability.

Theorem 8 (Necessary condition of identifiability). Let  $X \in \mathbb{R}^{n \times V_0}$  and  $\theta \in \left(\mathbb{R}^E \times \mathbb{R}^B\right) \setminus (S \cup \Delta_X)$ . If  $C_N$  is not satisfied, then  $\theta$  is not locally identifiable, that is, for all  $\epsilon > 0$  there exists  $\tilde{\theta} \in \left(\mathbb{R}^E \times \mathbb{R}^B\right) \setminus (S \cup \Delta_X)$  such that  $\|\theta - \tilde{\theta}\|_{\infty} < \epsilon$  and  $\theta \not\sim \tilde{\theta}$  but

$$
f _ {\theta} (X) = f _ {\bar {\theta}} (X).
$$

265 Thus, in particular, if  $C_N$  is not satisfied, then  $\theta$  is not identifiable.

The following result states that  $C_S$  is a sufficient condition of local identifiability.

Theorem 9 (Sufficient condition of local identifiability). Let  $X \in \mathbb{R}^{n \times V_0}$  and  $\theta \in (\mathbb{R}^E \times \mathbb{R}^B) \setminus (S \cup \Delta_X)$ . If  $C_S$  is satisfied, then  $\theta$  is locally identifiable, that is there exists  $\epsilon > 0$  such that for all  $\tilde{\theta} \in \mathbb{R}^E \times \mathbb{R}^B$ , if  $\| \theta - \tilde{\theta} \|_{\infty} < \epsilon$ ,

$$
f _ {\theta} (X) = f _ {\tilde {\theta}} (X) \quad \Longrightarrow \quad \theta \sim \tilde {\theta}.
$$

Both theorems are proven in the supplement. To discuss these two results, let us first point out that  $C_N$  and  $C_S$  are close from one another. We argue in fact that they are sharp in the sense that the case separating them,  $R_{\Gamma} < R_A$ , corresponds to an existing alignment between the image of  $D\psi^{\theta}(\tau_{\theta})$  and  $\operatorname{Ker} A(X,\theta)$ , which should be unlikely as only  $\operatorname{Ker} A(X,\theta)$  is sample dependent. Second, in order to have  $R_{\Gamma} = |F_{\theta}| + |B|$ , on needs to have  $nN_L \geq |F_{\theta}| + |B|$ . This means that the number of scalar measurements (number of samples  $n$  times output dimension  $N_L$ ) is larger than or equal to the number of parameters (up to local reparameterization).

# 5 Checking the conditions numerically

The key benefit of the conditions  $C_N$  and  $C_S$ , compared to the existing literature, is that they can be numerically tested for any fixed finite sample. They need the computation of the rank of two linear operators, namely  $\Gamma(X, \theta)$  and  $A(X, \theta)$ . The operator  $\Gamma(X, \theta)$  satisfies the following:

Proposition 10. Let  $X \in \mathbb{R}^{n \times V_0}$  and  $\theta \in \left(\mathbb{R}^E \times \mathbb{R}^B\right) \setminus (S \cup \Delta_X)$ . The function

$$
\begin{array}{r c l} U _ {\theta} & \longrightarrow & \mathbb {R} ^ {n \times V _ {L}} \\ \tau & \longmapsto & f _ {\rho_ {\theta} (\tau)} (X) \end{array}
$$

is differentiable in a neighborhood of  $\tau_{\theta}$ , and we denote by  $D_{\tau}f_{\theta}(X)$  its differential at  $\tau_{\theta}$ . We have

$$
D _ {\tau} f _ {\theta} (X) = \Gamma (X, \theta). \tag {14}
$$

The proof of Proposition 10 is in the supplement. Since the reparameterization with  $\rho_{\theta}$  simply consists in fixing the weights of the edges  $v \to s_{\max}^{\theta}(v)$  to the value  $w_{v \to s_{\max}^{\theta}(v)}$ , (14) shows that the coefficients of  $\Gamma(X, \theta)$  can be computed by a classic backpropagation algorithm  $N_L$  times for each input  $x^i$ , simply omitting the derivatives with respect to the edges of the form  $v \to s_{\max}^{\theta}(v)$ . An explicit expression of the coefficients of  $\Gamma(X, \theta)$  is given in the supplement.

To be satisfied,  $C_S$  needs the dimensions of  $\Gamma(X, \theta)$  to satisfy  $nN_L \geq |F_\theta| + |B|$ . One then needs to compute the rank  $R_{\Gamma}$  of  $\Gamma(X, \theta)$ , which means computing the rank of a  $nN_L \times (|F_\theta| + |B|)$  matrix. Existing algorithms allow to do this with a complexity  $O(nN_L(|F_\theta| + |B|)^{\omega - 1})$  (up to polylog terms), where  $\omega$  is the matrix multiplication exponent and satisfies  $\omega < 2.38$  [8].

When it comes to  $C_N$ , one needs in addition to know the rank  $R_A$  of  $A(X, \theta)$ , which, as Proposition 11 states, requires to compute the rank of  $\alpha(X, \theta)$ .

Proposition 11. Let  $X\in \mathbb{R}^{n\times V_0}$  and  $\theta \in \mathbb{R}^E\times \mathbb{R}^B$  . We have

$$
R _ {A} = N _ {L} \operatorname {r a n k} (\alpha (X, \theta)).
$$

The dimensions of  $\alpha(X, \theta)$  are sensibly larger, with  $|\mathcal{P}|$  columns and  $n$  lines, and typically  $|\mathcal{P}| >> n$ . However it may have some sparsity properties, as its entries consist in products of activation indicators (with possibly one input  $x_{v_0}^i$ ), any one of them being zero causing many entries to vanish. The question of the efficient computation of  $R_A$  still needs to be explored and is left as open for future work.

# 6 Conclusion

This paper is the first to characterize local identifiability for deep ReLU networks for any given finite sample, with testable conditions. The practical use of these conditions deserves follow-up research, and so does an extension of our approach to inverse stability.

# References

[1] Rilwan A Adewoyin, Peter Dueben, Peter Watson, Yulan He, and Ritabrata Dutta. Tru-net: a deep learning approach to high resolution prediction of rainfall. Machine Learning, 110(8): 2035-2062, 2021.  
[2] Francesca Albertini, Eduardo D Sontag, and Vincent Maillot. Uniqueness of weights for neural networks. Artificial Neural Networks for Speech and Vision, pages 115-125, 1993.  
[3] Sanjeev Arora, Aditya Bhaskara, Rong Ge, and Tengyu Ma. Provable bounds for learning some deep representations. In Eric P. Xing and Tony Jebara, editors, Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 584-592, Beijing, China, 22-24 Jun 2014. PMLR.  
[4] Joachim Bona-Pellissier, François Bachoc, and François Malgouyres. Parameter identifiability of a deep feedforward ReLU neural network. arXiv preprint arXiv:2112.12982, 2021.  
[5] Alon Brutzkus and Amir Globerson. Globally optimal gradient descent for a ConvNet with Gaussian inputs. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 605–614, 2017.  
[6] Nicholas Carlini, Chang Liu, Ülfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer: Evaluating and testing unintended memorization in neural networks. In 28th {USENIX} Security Symposium (\{USENIX\} Security 19), pages 267-284, 2019.  
[7] Nicholas Carlini, Matthew Jagielski, and Ilya Mironov. Cryptanalytic extraction of neural network models. In Annual International Cryptology Conference, pages 189-218. Springer, 2020.  
[8] Ho Yee Cheung, Tsz Chiu Kwok, and Lap Chi Lau. Fast matrix rank algorithms and applications. Journal of the ACM (JACM), 60(5):1-25, 2013.  
[9] Dennis Maximilian Elbrächter, Julius Berner, and Philipp Grohs. How degenerate is the parametrization of neural networks with the ReLU activation function? In Advances in Neural Information Processing Systems, volume 32, 2019.  
[10] Charles Fefferman. Reconstructing a neural net from its output. Revista Matemática Iberoamericana, 10(3):507-555, 1994.  
[11] Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security, pages 1322-1333, 2015.  
[12] Haoyu Fu, Yuejie Chi, and Yingbin Liang. Guaranteed recovery of one-hidden-layer neural networks via cross entropy. IEEE Transactions on Signal Processing, 68:3225-3235, 2020.  
[13] Rong Ge, Jason D Lee, and Tengyu Ma. Learning one-hidden-layer neural networks with landscape design. In 6th International Conference on Learning Representations, ICLR 2018, 2018.  
[14] Awni Hannun, Carl Case, Jared Casper, Bryan Catanzaro, Greg Diamos, Erich Elsen, Ryan Prenger, Sanjeev Satheesh, Shubho Sengupta, Adam Coates, et al. Deep speech: Scaling up end-to-end speech recognition. arXiv preprint arXiv:1412.5567, 2014.  
[15] Geoffrey Hinton, Li Deng, Dong Yu, George E. Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N. Sainath, and Brian Kingsbury. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. IEEE Signal Processing Magazine, 29(6):82–97, 2012.  
[16] Paul C Kainen, Věra Kürková, Vladik Kreinovich, and Ongard Sirisaengtaksin. Uniqueness of network parametrization and faster learning. Neural, Parallel & Scientific Computations, 2(4): 459-466, 1994.

[17] Nal Kalchbrenner and Phil Blunsom. Recurrent continuous translation models. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1700-1709, 2013.  
[18] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25:1097-1105, 2012.  
[19] Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial examples in the physical world. International Conference on Learning Representations, 2017.  
[20] Věra Kürková and Paul C Kainen. Functionally equivalent feedforward neural networks. Neural Computation, 6(3):543-558, 1994.  
[21] François Malgouyres and Joseph Landsberg. On the identifiability and stable recovery of deep/multi-layer structured matrix factorization. In IEEE, Info. Theory Workshop, Sept. 2016.  
[22] François Malgouyres and Joseph Landsberg. Multilinear compressive sensing and an application to convolutional linear networks. SIAM Journal on Mathematics of Data Science, 1(3):446-475, 2019.  
[23] Francois Malgouyres. On the stable recovery of deep structured linear networks under sparsity constraints. In Mathematical and Scientific Machine Learning, pages 107-127. PMLR, 2020.  
[24] Tomas Mikolov, Martin Karafiát, Lukas Burget, Jan Cernocký, and Sanjeev Khudanpur. Recurrent neural network based language model. In Interspeech, volume 2, pages 1045–1048, 2010.  
[25] Tomás Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. In Yoshua Bengio and Yann LeCun, editors, 1st International Conference on Learning Representations, ICLR 2013, Scottsdale, Arizona, USA, May 2-4, 2013, Workshop Track Proceedings, 2013.  
[26] Behnam Neyshabur, Russ R Salakhutdinov, and Nati Srebro. Path-SGD: Path-normalized optimization in deep neural networks. Advances in neural information processing systems, 28, 2015.  
[27] Philipp Petersen, Mones Raslan, and Felix Voigtlaender. Topological properties of the set of functions generated by neural networks of fixed size. Foundations of Computational Mathematics, 21:375-444, 2021.  
[28] Henning Petzka, Martin Trimmel, and Cristian Sminchisescu. Notes on the symmetries of 2-layer ReLU-networks. In Proceedings of the Northern Lights Deep Learning Workshop, volume 1, pages 6-6, 2020.  
[29] Mary Phuong and Christoph H. Lampert. Functional vs. parametric equivalence of ReLU networks. In International Conference on Learning Representations, 2020.  
[30] José Pedro Pinto, André Pimenta, and Paulo Novais. Deep learning and multivariate time series for cheat detection in video games. Machine Learning, 110(11):3037-3057, 2021.  
[31] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779-788, 2016.  
[32] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28:91-99, 2015.  
[33] David Rolnick and Konrad Kording. Reverse-engineering deep ReLU networks. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 8178-8187, 13-18 Jul 2020.

[34] Hasim Sak, Andrew Senior, and Françoise Beaufays. Long short-term memory recurrent neural network architectures for large scale acoustic modeling. In Fifteenth Annual Conference of the International Speech Communication Association, 2014.  
[35] Hanie Sedghi and Anima Anandkumar. Provable methods for training neural networks with sparse connectivity. In Deep Learning and representation learning workshop: NIPS, 2014.  
[36] Pierre Stock. Efficiency and Redundancy in Deep Learning Models: Theoretical Considerations and Practical Applications. PhD thesis, Université de Lyon, April 2021. URL https://tel.archives-ouvertes.fr/tel-03208517.  
[37] Pierre Stock and Rémi Gribonval. An Embedding of ReLU Networks and an Analysis of their Identifiability. Constructive Approximation, 2022. URL https://hal.archives-ouvertes.fr/hal-03292203.  
[38] Héctor J Sussmann. Uniqueness of the weights for minimal feedforward nets with a given input-output map. Neural networks, 5(4):589-593, 1992.  
[39] Mingyang Yi, Qi Meng, Wei Chen, Zhi-ming Ma, and Tie-Yan Liu. Positively scale-invariant flatness of ReLU neural networks. arXiv preprint arXiv:1903.02237, 2019.  
[40] Jialong Zhang, Zhongshu Gu, Jiyong Jang, Hui Wu, Marc Ph Stoecklin, Heqing Huang, and Ian Molloy. Protecting intellectual property of deep neural networks with watermarking. In Proceedings of the 2018 on Asia Conference on Computer and Communications Security, pages 159–172, 2018.  
[41] Shuai Zhang, Meng Wang, Jinjun Xiong, Sijia Liu, and Pin-Yu Chen. Improved linear convergence of training CNNs with generalizability guarantees: A one-hidden-layer case. IEEE Transactions on Neural Networks and Learning Systems, 32(6):2622-2635, 2020.  
[42] Xiao Zhang, Yaodong Yu, Lingxiao Wang, and Quanquan Gu. Learning one-hidden-layer ReLU networks via gradient descent. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 1524-1534. PMLR, 2019.  
[43] Kai Zhong, Zhao Song, Prateek Jain, Peter L Bartlett, and Inderjit S Dhillon. Recovery guarantees for one-hidden-layer neural networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 4140–4149, 2017.  
[44] Mo Zhou, Rong Ge, and Chi Jin. A local convergence theory for mildly over-parameterized two-layer neural network. arXiv preprint arXiv:2102.02410, 2021.
