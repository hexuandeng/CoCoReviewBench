# GLOBAL MINIMA, RECOVERABILITY THRESHOLDS, AND HIGHER-ORDER STRUCTURE IN GNNS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We analyze the performance of graph neural network (GNN) architectures from the perspective of random graph theory. Our approach promises to complement existing lenses on GNN analysis, such as combinatorial expressive power and worst-case adversarial analysis, by connecting the performance of GNNs to typical-case properties of the training data. First, we theoretically characterize the accuracy of one- and two-layer GCNs relative to the contextual stochastic block model (cSBM) and related models. We additionally prove that GCNs cannot beat linear models under certain circumstances. Second, we numerically map the recoverability thresholds, in terms of accuracy, of four diverse GNN architectures (GCN, GAT, SAGE, and Graph Transformer) under a variety of assumptions about the data. Sample results of this second analysis include: heavy-tailed degree distributions enhance GNN performance, GNNs can work well on strongly heterophilous graphs, and SAGE and Graph Transformer can perform well on arbitrarily noisy edge data, but no architecture handled sufficiently noisy feature data well. Finally, we show how both specific higher-order structures in synthetic data and the mix of empirical structures in real data have dramatic effects (usually negative) on GNN performance.

# 1 INTRODUCTION

Graph neural networks (GNNs) have achieved impressive success across many domains, including natural language processing (Wu et al., 2023a), image representation learning (Adnan et al., 2020), and perhaps most impressively in protein folding prediction (Jumper et al., 2021). GNNs' success across these fields is due to their ability to harness non-Euclidean graph topology in the learning process (Xu et al., 2019). Despite the growing use of GNN architectures, we still grapple with a significant knowledge gap concerning the intricate relationship between the statistical structure of graph data and the nuanced behavior of these models. By aligning GNN designs with data distributions, we can not only unveil the underlying mechanics and behaviors of these models but also pave the way for architectures that intuitively resonate with inherent data patterns.

While significant focus has been directed towards homophily in the context of GNN performance (Maurya et al., 2021; Halcrow et al., 2020; Zhu et al., 2020), other critical properties of graph data have remained relatively underexplored. Features such as degree distribution and mesoscale structure offer important insights into the behavior of networks. Similarly, despite the depth of theoretical advancements in graph modularity, including works such as the one by Abbe (2018), there remains a sizable gap in their integration and applicability within the GNN domain. We seek to explore such properties to bridge this gap.

In particular, our results imply that commonly studied properties such as homophily, Gaussian feature separation, and high dimensionality aren't enough to explain and justify the use of certain nonlinear GNNs, as we show that their performance is matched by linear GNN models in the cSBM setting. This motivates research into which significant features of the data should be incorporated into the data generation models commonly used to study GNNs.

In this paper we:

NEW

NEW

- fully characterize the accuracy of one- and two-layer GNNs satisfying certain assumptions, as well as proving that the accuracy of certain nonlinear GNNs is bounded above by the accuracy of a linear GNN when the graph is drawn from a broad family,  
- report extensive numerical studies that map the degree to which edge and feature information contribute to overall performance across diverse models in a variety of random graph contexts, and  
- demonstrate that the presence of higher order structures in graphs causes a dramatic (and usually negative) change in GNN accuracy.

# 2 PREVIOUS WORK

As part of this work we lay out theoretical bounds for GNN architectures. Some foundational work in our topic is as follows. Fountoulakis et al. (2023) investigated regimes in which the attention module in the Graph (GAT) (Veličković et al., 2018) makes a meaningful difference in performance. Following this, Baranwal et al. (2023) proved theoretically that using graph convolutions expands the range where a vanilla neural network can correctly classify nodes. Baranwal et al. (2021) discovered that linear classifiers on GNN embeddings generalize well to out of distribution data in stochastic block models. Lu (2022) characterized how well a GNN can separate communities on a two-class stochastic block model. Recently, Ma et al. (2022) rigorously identified noise regimes where GNNs perform well on heterophilous graphs and Chien et al. (2021) propose an architecture that adapts to the modularity of a graph. Lastly, N.T. & Maehara (2019) found that a Graph Convolutional Network (GCN) performs low pass filtering on the feature vectors and doesn't learn non-linear manifolds.

While many have attempted to understand models through the lens of specialized data, our approach offers a unique and deeper perspective on the subject. The monograph Abbe (2018) lays out the key mathematical findings related to SBMs as they relate to community detection. Karrer & Newman (2011) developed the degree-corrected SBM, which allows for heavy-tailed degree distributions. Gao et al. (2018) derived asymptotic minimax risks for misclassification in degree-corrected SBMs and Mehta et al. (2019) propose a variational autoencoder for SBMs. Deshpande et al. (2018) proposed a contextual SBM (cSBM) that generates feature data alongside the graph data. This was originally proposed to analyze specific properties of belief propagation (Bickson, 2009). Finally, Wu et al. (2023b) not only explore the characteristics of oversmoothing in GNNs through cSBMs but also characterize how graph convolutions function both as denoising and feature-mixing mechanisms, detailing the extent and manner in which these processes occur.

Our investigation presents a novel angle that bridges interplay between edge data and feature data. Binkiewicz et al. (2017) explored how to use features to aid spectral clustering. Yang et al. (2022) and Arroyo et al. (2021) used edges and features that contain orthogonal information to better understand the relationship between the two. While the influence of motifs or higher-order structures on GNN performance remains a hot area of exploration, our approach delves deeper into this pressing topic. Works such as Tu et al. (2020) have proposed using graphlets to aid in learning representations. Others have utilized hypergraphs to make better predictions (Huang & Yang, 2021). Much of the work quantifying the expressive power of GNNs is achieved by relating GNNs to the classical Weisfeiler-Leman (WL) heuristic for graph isomorphism (Li & Leskovec, 2022; Huang & Villar, 2021). These have inspired corresponding GNN architectures that have increased distinguishing capabilities (Hamilton, 2020)

# 3 BACKGROUND

In this work, we first theoretically determine nodewise accuracy for certain one- and two-layer GNNs and identify cases where nonlinear GNNs cannot outperform linear GNN models. We map the performance of the Graph Convolutional Network (Kipf & Welling, 2017), Graph SAGE (Hamilton et al., 2017), the Graph Attention Network (Veličković et al., 2018), and the Structure-Aware Transformer (Chen et al., 2022) on several related random graph models related to the cSBM. We will also inject and remove higher-order structure in various contexts to see how GNN performance is affected. We now describe some of the random graph models and GNN architectures on which our analysis relies. Note, when referring to data generation methods we use the term generative models while model will refer to a trained GNN.

# 3.1 STOCHASTIC BLOCK MODELS

The stochastic block model is a random graph model that encodes node clusters ("classes") in the graph topology. The presence or absence of each edge is determined by an independent Bernoulli draw with probability determined by the class identities of the nodes. We restrict attention to SBMs where all classes have the same size and uniform inter-class and intra-class probabilities. The parameters for such an SBM are: the total number of nodes  $n$ , the number of equally sized classes  $k$ , the intra-class edge probability  $p_{\mathrm{in}}$ , and the inter-class edge probability  $p_{\mathrm{out}}$ . While SBMs generate realistic clustering patterns, without further modification they exhibit a binomial degree distribution. To more closely model many realistic classes of data, Karrer & Newman (2011) proposed the degree-corrected SBM, which can exhibit any degree distribution, notably heavy-tailed distributions.

In this paper, we represent edge similarity using an edge information parameter,  $\lambda$ , which has the following relationship to  $p_{\mathrm{in}}$  and  $p_{\mathrm{out}}$ :

$$
p _ {\mathrm {i n}} = \frac {d + \lambda \sqrt {d}}{n}, \quad p _ {\mathrm {o u t}} = \frac {d - \lambda \sqrt {d}}{n},
$$

where  $d$  is the expected average node degree. Setting  $\lambda = 0$  yields identical inter- and intra-class edge probabilities, meaning the topology of the graph encodes no information about class labels. A positive  $\lambda$  indicates that nodes of the same class are more likely to connect than nodes of different classes (homophily), while a negative  $\lambda$  indicates the reverse relationship (heterophily).

To generate node attributes, Deshpande et al. (2018) proposed the contextual SBM (cSBM), where features are drawn from Gaussian point clouds with mean at a specified distance  $\mu$  from the origin. Features,  $X$ , are thus defined as  $X(i) = \mu m_{v_i} + z_i$ , where  $z_i$  a standard normally distributed random variable,  $v_i$  is the ground-truth class label of node  $i$ , and  $m_{v_i}$  is the mean for class  $v_i$ . The means are chosen to be an orthogonal set. We can then vary the level of feature separability (feature information) by modifying  $\mu$ . Setting  $\mu = 0$  makes node features indistinguishable across classes, while a large value of  $\mu$  indicates high distinguishability. We thus refer to  $\mu$  as the feature information parameter.

# 3.2 GRAPH NEURAL NETWORKS

As stated before, we analyze the performance of four diverse and influential architectures: GCN Kipf & Welling (2017), SAGE Hamilton et al. (2017), GAT Velicković et al. (2018), and GraphTransformer Chen et al. (2022). In our numerical work, we also assess the performance of a standard feedforward neural network and spectral clustering (von Luxburg, 2007), which are useful points of comparison as they are agnostic to the graph and feature structures, respectively. Lastly we also use graph-tool (Peixoto, 2014) to evaluate feature-agnostic performance on heterophilous graphs.

# 4 THEORETICAL RESULTS

We now derive analytically the performance of GNN architectures when the data-generating process is known. Section 4.1 covers the one-layer case for a GCN architecture and cSBM-generated data, and section 4.2 handles the two-layer case in for a more general class of GNN architecture as well as a broader class of generating processes. We introduce the following notation first: for a given node  $i$ ,  $n_{\mathrm{in}}(i)$  is the number of neighbors in the same class as  $i$ , and  $n_{\mathrm{out}}(i)$  is the number of neighbors in other classes.  $\mathcal{N}(i)$  is the one-hop neighborhood of  $i$ .  $v_{i}$  is the ground-truth class label of  $i$ . erf is the Gaussian error function. Both subsections assume a binary classification setting. The results in 4.1 are at least partly known in other literature (e.g. Lemma 1 from Wu et al. (2023b)), but they are included here for completeness.

# 4.1 ACCURACY ESTIMATES FOR SINGLE-LAYER GCNS

In the one-layer case, we assume the GNN is of the simple form  $y[X] = \mathrm{sign}(AXW)$ , that the final embedding is into  $\mathbb{R}$ , and that  $A$  and  $X$  are generated by a cSBM, with no self-loops (but see remark 2). We also make a slight modification to the cSBM setup so that the means are diametrically

opposed rather than orthogonal. That is,

$$
X (i) = \left\{ \begin{array}{l l} \mu m + z _ {i} & \text {i f n o d e i i s i n c l a s s 1} \\ - \mu m + z _ {i} & \text {i f n o d e i i s i n c l a s s 2}. \end{array} \right.
$$

This requires no loss of generality, since all choices of two means may be translated to fit this assumption. We then have the following proposition:

Proposition 1. Under the preceding assumptions, we have

1. For each  $i$ ,  $(AXW)_i$  has the distribution

FIX

FIX

$$
\underbrace{\mu(n_{\mathrm{in}}(i) - n_{\mathrm{out}}(i))m\cdot W}_{\text{neighborhood signal}} + \underbrace{\left(\sum_{j\in\mathcal{N}(i)}z_{j}\right)\cdot W}_{\text{noise}}.
$$

2. If  $W \neq 0$ , the generalization accuracy, conditioned on the graph structure is,

FIX

$$
P (y [ X ] (i) = 1 \mid n _ {\mathrm {i n}} (i), n _ {\mathrm {o u t}} (i), v _ {i} = 1) = \frac {1}{2} \left(\operatorname {e r f} \left(\frac {\mu \left(n _ {\mathrm {i n}} (i) - n _ {\mathrm {o u t}} (i)\right)}{\sqrt {2 \left(n _ {\mathrm {i n}} (i) + n _ {\mathrm {o u t}} (i)\right)}} \cos \theta\right) + 1\right),
$$

where  $\theta$  is the angle between  $W$  and  $m$ .

FIX

3. The maximum expected accuracy for an arbitrary node in the homophilous regime is achieved when  $\theta = \pi$ . In the heterophilous regime,  $\theta = 0$  is the maximizer.

Proof. See appendix A.

![](images/538a1ed93b8593d542803cad4b71544ab9f5a471b8ba471b01c5ae8cb0598b42.jpg)

Remark 1. Part three of this proposition shows that, in the one-layer case, optimal performance is achieved simply by aligning the learned parameters with the axis separating the means of the distributions. The proof consists largely of manipulations of the probability densities, together with calculus. A similar alignment result applies in the two-layer case, but in that case, the fastest way forward is to rely on the symmetries of the distribution and GNN, as shown below.

Remark 2. The analysis with self-loops is nearly identical, with the exception that it is possible that the maximizing parameters may possibly differ in the extremely dense, slightly heterophilous case, but this is not the regime in which GNNs are typically used. Extremely dense refers to the case where almost all edges are present. See the proof for full details.

NEW

# 4.2 ANALYSIS OF TWO-LAYER GCNS

NEW

In this section, we make two claims about the effectiveness of a class of GNNs given certain symmetries in the model space. These symmetry assumptions are satisfied by the cSBM, and both results make precise that the effectiveness of nonlinear GNNs cannot be explained by cSBM-type data models. First, given a certain symmetry about the origin, we claim the cost of the model  $y$  is no smaller than the cost of a linear model. Second, given an additional symmetry about any subspace  $S$  of the feature space, we claim the cost of the linear model is no smaller than the cost of a projection of the linear model.

# 4.2.1 SET-UP

FIX

We define a 2-class attributed random graph model to be a probability space  $(\Omega, P)$  of tuples  $(G, i, v, X)$  where  $G$  is a graph,  $i$  is a node in  $G$ , and  $v$  and  $X$  are functions mapping each node in the graph to its class and its feature vector, respectively. That is,

$$
v: G \to \{- 1, 1 \}
$$

$$
X: G \to \mathbb {R} ^ {m _ {\mathrm {f e a t}}}
$$

As a notational convenience, let  $v(x)$  denote the class of the node corresponding to a tuple  $x \in \Omega$ .

FIX

A model  $y$  on a 2-class attributed random graph model assigns to each  $x \in \Omega$  a real number  $y(x) \in \mathbb{R}$  that corresponds to the estimated probability that the node corresponding to  $x$  is of class 1. More concretely, the predicted probability is given by

$$
P (v (x): y (x)) = \left\{ \begin{array}{l l} \sigma_ {s} (y (x)) & v (x) = 1 \\ 1 - \sigma_ {s} (y (x)) & v (x) = - 1. \end{array} \right.
$$

where  $\sigma_s: \mathbb{R} \to (0,1)$  is the logistic sigmoid  $\sigma_s(z) = (1 + e^{-z})^{-1}$ . According to maximum likelihood learning, the cost function of the model  $y$  is

$$
C (y) = \mathbb {E} _ {x \sim \Omega} [ - \log P (v (x): y (x)) ].
$$

In this section, we will focus on 2-layer models consisting of linear aggregators interspersed by the non-linear ReLU function  $\sigma$ . More concretely, a graph aggregator maps a graph and its features to a new set of features on the graph:

$$
\phi : (G, X) \to X ^ {\prime}
$$

where  $X^{\prime}:G\to \mathbb{R}^{l}$  for some  $l$ . We write  $\phi_G = \phi (G,\cdot)$ . A linear aggregator (without bias) satisfies

$$
\phi_ {G} \left(X _ {1} + X _ {2}\right) = \phi_ {G} \left(X _ {1}\right) + \phi_ {G} \left(X _ {2}\right)
$$

for all graphs  $G$  and features  $X_{1}, X_{2}$ . Linear aggregators include the standard sum and mean aggregators, but they also include more general aggregators such as applying the sum aggregator after adding self-loops with a custom weight. A generalized 2-layer graph convolutional network (GCN) without bias is then given by

$$
y (x) = (\phi_ {G} ^ {\prime} \circ \sigma \circ \phi_ {G}) [ X ] (i)
$$

where  $\phi$  and  $\phi^{\prime}$  are linear aggregators,  $\phi^\prime$  maps into  $\mathbb{R}$ , and  $\sigma$  is the ReLU function.

# 4.2.2 PRINCIPAL CLAIMS

FIX

In this section, we make two claims on the effectiveness of these generalized GCNs given certain symmetries in the model space  $\Omega$ . First, given a certain symmetry about the origin, we claim the cost of the model  $y$  is no smaller than the cost of the linear model  $L[y]$ :

$$
L [ y ] (x) = \frac {1}{2} \left(\phi_ {G} ^ {\prime} \circ \phi_ {G}\right) [ X ] (i)
$$

Second, given an additional symmetry about any subspace  $S$  of the feature space, we claim the cost of the linear model  $L[y]$  is no smaller than the cost of the projection of the linear model  $P_{S}[L[y]]$ :

$$
P _ {S} [ L [ y ] ] (x) = \frac {1}{2} \left(\phi_ {G} ^ {\prime} \circ \phi_ {G} \circ P _ {S}\right) [ X ] (i)
$$

where  $P_{S}$  is simply the projection on the subspace  $S$ . For example, if  $\phi'$  and  $\phi$  are both simply the classical right-multiplication by a weight matrix followed by summing the features of neighbors, then model  $y$  becomes

$$
y (x) = \sum_ {j \in \mathcal {N} (x)} \sigma \left(\sum_ {k \in \mathcal {N} (j)} X (k) W\right) \cdot c
$$

for some weight matrix  $W$  and weight vector  $c$ . If the symmetries mentioned above hold for the subspace  $S = \operatorname{span}\{\vec{m}\}$  for some vector  $\vec{m}$  (as is the case with a cSBM), then the above claims assert the cost of the model  $y$  is no smaller than the cost of the model,

$$
P _ {S} [ L [ y ] ] (x) = \frac {1}{2} \sum_ {j \in \mathcal {N} (x)} \sum_ {k \in \mathcal {N} (j)} P _ {S} (X (k)) W \cdot c = K \sum_ {j \in \mathcal {N} (x)} \sum_ {k \in \mathcal {N} (j)} X (k) \cdot \vec {m}
$$

for some  $K\in \mathbb{R}$

FIX

The first symmetry is defined using the negation of element of  $\Omega$ . If  $x = (G,i,v,X)\in \Omega$ , we define the negation of  $x$  to be the tuple  $-x = (G,i, - v, - X)$ . In other words,  $x$  has the same graph with all of the classes and features negated. We similarly define the negation of a subset  $F\subset \Omega$  by  $-F = \{-x:x\in F\}$ . We say a 2-class attributed random graph model is class-symmetric about the origin if  $P(F) = P(-F)$  for all measurable  $F\subset \Omega$ . Heuristically, this means that in the distribution of graphs, the nodes of the two classes have the same topological distribution (which still allows for homophily/heterophily) and that the feature distribution of of class -1 is equal to the feature distribution of class 1 reflected across the origin. A cSBM with an equal number of nodes in both classes satisfies

this symmetry, but this property is also held by graph models having non-Gaussian noise so long as there is symmetry across the origin.

The second symmetry concerns the feature distribution alone. If  $S$  is a subspace of the feature space and  $R_{S}$  is the reflection across  $S$ , then the reflection of  $x \in \Omega$  is defined by  $R_{S}(x) = (G, i, v, R_{S} \circ X)$ . In other words,  $x$  has the same graph with all the features reflected across  $S$ . We similarly define the reflection of a subset  $F \subset \Omega$  by  $R_{S}(F) = \{R_{S}(x) : x \in F\}$ . We say a 2-class attributed random graph model is symmetric about  $S$  if  $P(F) = P(R_{S}(F))$  for all measurable  $F \subset \Omega$ . Heuristically, this means the feature distribution is symmetric about the subspace  $S$ .

Theorem 1. Let  $\Omega$  be a 2-class attributed random graph model and let  $y$  be any two-layer generalized GCN without bias on  $\Omega$ . If  $\Omega$  is class-symmetric about the origin then,

$$
C (L [ y ]) \leq C [ y ].
$$

Furthermore, if  $\Omega$  is symmetric about  $S$  then,

$$
C (P _ {S} [ L [ y ] ]) \leq C (L [ y ])
$$

Proof. See appendix B. The main idea is to use the symmetries of the space together with the convexity of the objective to invoke Jensen's inequality.  $\square$

FIX

NEW

We note that similarity between the previous theorem and ideas from Wu et al. (2023b). Our work focuses on models with a stacked non-linearity, while the latter deals primarily with linear models.

In light of the preceding theorem, linear GCNs are optimal over the binary cSBM. Carefully analyzing the linear case, we obtain an explicit formula for the optimal accuracy of any GCN over cSBM data. Although difficult to analyze theoretically, the accuracy can be calculated empirically using the following formula (see the remark afterward for an intuitive explanation):

NEW

Theorem 2. In the large node limit of a cSBM, the linear model

$$
y (x) = K \sum_ {j \in \mathcal {N} (x)} \sum_ {k \in \mathcal {N} (j)} X (k) \cdot m
$$

has accuracy

$$
\sum_ {n _ {\mathrm {i n}}, n _ {\mathrm {o u t}}, n _ {2 - \mathrm {i n}}, n _ {2 - \mathrm {o u t}} = 0} ^ {\infty} P (n _ {\mathrm {i n}}, n _ {\mathrm {o u t}}, n _ {2 - \mathrm {i n}}, n _ {2 - \mathrm {o u t}}) \Phi \Bigg (\psi \bigg (\operatorname {s g n} (K) \frac {\mu}{\sigma}, n _ {\mathrm {i n}}, n _ {\mathrm {o u t}}, n _ {2 - \mathrm {i n}}, n _ {2 - \mathrm {o u t}} \bigg) \Bigg)
$$

where  $\Phi$  is the cdf of the standard normal distribution and the following definitions apply:

$$
\begin{array}{l} P (n _ {\mathrm {i n}}, n _ {\mathrm {o u t}}, n _ {2 - \mathrm {i n}}, n _ {2 - \mathrm {o u t}}) \\ = p (n _ {\mathrm {i n}}, d _ {\mathrm {i n}}) \cdot p (n _ {\mathrm {o u t}}, d _ {\mathrm {o u t}}) \cdot p (n _ {2 - \mathrm {i n}}, d _ {\mathrm {i n}} n _ {\mathrm {i n}} + d _ {\mathrm {o u t}} n _ {\mathrm {o u t}}) \cdot p (n _ {2 - \mathrm {o u t}}, d _ {\mathrm {o u t}} n _ {\mathrm {i n}} + d _ {\mathrm {i n}} n _ {\mathrm {o u t}}), \\ \end{array}
$$

$$
p (k, \lambda) = \frac {\lambda^ {k} e ^ {- \lambda}}{k !}, a n d
$$

$$
\psi (c, n _ {\mathrm {i n}}, n _ {\mathrm {o u t}}, n _ {2 - \mathrm {i n}}, n _ {2 - \mathrm {o u t}}) = c \frac {1 + 3 n _ {\mathrm {i n}} - n _ {\mathrm {o u t}} + n _ {2 - \mathrm {i n}} - n _ {2 - \mathrm {o u t}}}{\sqrt {(n _ {\mathrm {i n}} + n _ {\mathrm {o u t}} + 1) ^ {2} + 4 (n _ {\mathrm {i n}} + n _ {\mathrm {o u t}}) + (n _ {2 - \mathrm {i n}} + n _ {2 - \mathrm {o u t}})}}.
$$

Proof. See appendix B.

![](images/e9e10a040c04a8292fdbf299a3e9c97862e46605e3d511197b4995cf1bbe6875.jpg)

Remark 3. In the theorem, the indices  $n_{\mathrm{in}}$ ,  $n_{\mathrm{out}}$ ,  $n_{2 - \mathrm{in}}$ , and  $n_{2 - \mathrm{out}}$  refer to the number of distance 1 and 2 nodes with the same and the opposite class of the base node. The function  $P$  represents the probability of the graph structure having such characteristics, while the function  $\Phi \circ \psi$  is the accuracy at the base node given such characteristics. The function  $p$  is the p.m.f. of the Poisson distribution.

FIX

# 5 EMPIRICAL EXPLORATION OF DATA REGIMES

In section 5.1 and section 5.2, we present results from our simplest set of experiments in detail to illustrate the interplay between edges and features. Then, in section 5.3 we compare performance across each of the four architectures. Finally, we contrast how GNNs performed on degree-corrected and non-degree-corrected graphs in section 5.4. See also our full code online to extend this work to other architectures and parameter ranges:

# 5.1 EXPERIMENTAL DESIGN

To better understand how GNN architectures harness information embedded in the features or edges, we evaluated them across a variety of graphs. Each of our architectures was comprised of one input layer, a hidden layer of size 16 (with ReLU activation functions), and an output layer (with softmax). As baselines, we trained a feedforward neural network, with one hidden layer of size 16, on the feature data. Our exploration also encompassed a variety of methods for feature-agnostic methods such as graph-tool (Peixoto, 2014), Leidenalg (python package), Louvian (python package), and Spectral clustering (Pedregosa et al., 2011). In doing so we found that spectral clustering worked the best for assortative graphs (edge information from [0,3]) and graphtool performed the best on dissasorptive graphs (edge information from [-3,0)).

We generated graph data using a cSBM with average degree  $d = 10$ ; the number of nodes  $n = 1,000$ ; the number of features  $m_{\mathrm{feat}} = 10$ ; the number of classes  $c = 2$ ; and standard deviation of the Gaussian clouds .2. These hyperparameters were selected to be representative of a large variety of datasets without being too computationally expensive (specifically when using transformers). We observed that 1,000 nodes was large enough to get statistical regularity and that using larger graphs (up to 40,000 nodes) didn't introduce major deviations. With these hyperparameters, we vary  $\lambda$  (edge separation in cSBMs) between -3 and 3 and vary feature separation (cloud distance from origin) from 0 to 2 to obtain  $121 \times 200$  (how finely we discretized the interval) possible sets of graph data. This data ranges from being highly disassortative to highly assortative.

To train each architecture, we used an Adam optimizer (PyTorch) with a learning rate of 0.01 for 400 epochs (typically where the model ceased improving). We evaluated the final accuracy on a separate graph, with the same graph parameters to prevent overfitting.

In addition to the class count of two, we ran the architectures across class counts of three, five, and seven each with both a degree-corrected case and a binomial case. As each test was averaged/maxed over 10 trials, the number of tests totals 320 different tests with 15,488,000 accuracy scores generated (more than .25 petaflops used in total). We note that we used two hidden layers and Gaussian distributions for simplicity, but more complex distributions and additional layers merit future research.

NEW

# 5.2 EXAMPLE: BINARY NODE CLASSIFICATION WITH GRAPH TRANSFORMER

![](images/90aad3844e3aff7159d0b7f27f045c4abeb16253a4b5cf6ca707cb4d6c66a8be.jpg)  
Figure 1: (Left) Transformer's performance on a five-class non-degree-corrected cSBM, with color gradients indicating accuracy levels. To the right and below, performance curves for the feedforward neural network (graph-blind) and graph-based (feature-blind) methodologies are displayed respectively. (Right) A comparison of the top-performing model among the Graph Transformer, feedforward neural network, and graph-based clustering. White space indicates where one model was not consistently better than the others. The Transformer predominantly excels when edge and feature information were moderately noisy. The graph based method is able to surpass the transformer if we have a combination of high feature noise and low edge noise.

Our experiments with the Transformer architecture elucidate its robustness across a wide parameter space (see fig. 1). Remarkably, the Transformer consistently delivers superior performance across

most scenarios, with exceptions only in cases where both the feature and edge information are heavily compromised by noise. An intriguing capability of the Transformer is its potential to achieve flawless accuracy even when presented with solely noisy edge information. This implies an innate adaptability within the Transformer to sift through the noise, selectively emphasizing pertinent features over less informative edges. Message-passing GNNs seem to struggle with this (Bechler-Speicher et al., 2023) as seen in fig. 2.

The Transformer performs well on heterophilous graphs as well, most clearly seen in fig. 2. Such proficiency makes the Transformer an excellent candidate for tasks demanding the assimilation of diverse or opposing sets of information. A marked limitation is observed in the Transformer's ability to process noisy feature scenarios, where spectral clustering performs better. The Transformer's somewhat dependent relationship with feature information, even when suboptimal, necessitates further investigation.

NEW

# 5.3 PERFORMANCE OF GCN, GAT, SAGE, AND TRANSFORMER ARCHITECTURES

![](images/cf878f12e19960be5df2139876d5d2b08bcfc9c3b8b9f55e9e2575a8ed48bd7e.jpg)  
Figure 2: Comparison performance on non-degree-corrected and degree-corrected SBMs for GCN, GAT, SAGE and Transformer architectures. Notice the GCN and GAT consistently perform worse when the edge information is roughly zero, but the other two models achieve perfect accuracy given enough feature information. This could be due to SAGE and Transformer learning a more global context for each node. In this regime we see that almost all of the models did better on the heavy tailed graphs. GCN achieved higher accuracy on such graphs when the edges were just noise. The accuracy of the GAT improved as well in the regime of very noisy edges and features. All values are the best of 10 trials, with a  $5 \times 5$  convolutional filter applied for visual clarity.

We now juxtapose the performances of four distinct architectures, particularly considering the influence of heavy-tailed degree distributions. Refer to fig. 2 for insights on the three-class scenario, while an exhaustive analysis is cataloged in appendix C.1 and appendix C.2. Generally, both GraphTransformer and SAGE stand out for their resistance to edge and feature noise, demonstrating their robustness in noisy regimes. In a three-class, non-degree-corrected cSBM setting, SAGE and GraphTransformer consistently outperform the other two models, GAT and GCN. This is shown by their strong resistance to feature noise and their ability to classify accurately even without edge information. Such performance highlights SAGE's use of global information from random walks and graph embeddings, while the Transformer simply ignores the graph embedding.

Each architecture performs differently, as shown by their varying weak areas (seen as blue areas in fig. 2) and how they compare to neural network and spectral clustering benchmarks (detailed in appendix C.2). The GAT and GCNs weak area is especially prominent with no edge information, showing it relies heavily on clear features. Interestingly, both Transformer and GAT perform better with degree correction, especially in heterophilous settings. For a more in depth comparison of different models see appendix C.2

FIX

NEW

NEW

# 5.4 DEGREE-CORRECTED SBMS

We found that all models performed better on scale-free graphs. We believe this occurs due to a filtering out of bad neighbors. Most nodes in the heavy-tailed data have relatively few neighbors, this allows for fewer confusing neighbors to contribute misleading information in the aggregation step than in the binomial degree distribution. This is similar to ideas from Albert et al. (2000).

The scale free graphs affected the models in different ways, for example the performance of SAGE only improved in the higher signal edge regimes (right and left sides of the fig. 2). The performance of GAT increased dramatically in the case of very noisy edges and features. This is likely because degree correction gave it more information on what edges to prune. Interestingly, the attention based models, the Transformer and GAT, saw a stark increase in performance in the heterophilous clustering, suggesting that self-attention allows for a better interpretation of such graphs.

# 6 EFFECT OF HIGHER-ORDER STRUCTURE IN REAL WORLD DATASETS

![](images/04523d27e81ee7541144fd5c578e12d5593ac1736c97d2962b8933eee04e8dc9.jpg)  
Figure 3: Comparison of model accuracies on real data compared to performance on matched synthetic data. The accuracy tends to improve when we erase higher-order structure in the data. The datasets from left to right are: Flickr, DeezerEurope, Citeseer, LastFMA, DBLP, FacebookPagePage, Pubmed, GitHub, Cora, Amazon Computers, and Amazon Photos. The figure depicts cases where we transform only the edges, only the features, and both. The transformer was not run due to memory requirements.

The experiments to be described in this section support the claim that higher-order structure, such as clustering or motifs, influence the performance of GNN architectures. We found that the models generally performed better on matched synthetic data than on real data, suggesting that the higher-order structure that was erased is an impediment to GNN learning (see fig. 3).

To make the synthetic data for each data set, we transformed the edge and feature data as if each dataset were already a degree-corrected cSBM. We used a variety of datasets from pytorch geometric (Fey & Lenssen, 2019). In particular, the edge data was randomized by rewiring every edge to preserve degree distribution and modularity similar to ideas in Fosdick et al. (2018). In some experiments, the node features were also transformed by sampling from the estimated normal distribution. Thus, the synthetic data lacks nontrivial structure except the structure implied by the degree distribution, intra/inter-class linkage frequency, and feature means and standard deviations match the corresponding empirical network.

We see a positive impact on the accuracy of the GCN when removing the higher-order structure (see fig. 3) specifically with edge structure. The fact that the GNNs do better on this semirandomized data suggests that they may perform optimally on SBM-like data, but are negatively impacted by the additional structure present in real data. Uncovering why such structure can be detrimental to these GNNs is a significant opportunity for future work.

To further verify that we are not confusing higher-order structure with label noise, we verified these results on synthetic data with controlled structure. Such results indicate that GNNs perform worse on datasets with spatial structure, but are unaffected by local motifs such as triadic closure. Results on graphs with planted hierarchical structure were mixed but largely favored SBM data. A more detailed analysis can be found in appendix D.

# 7 REPRODUCIBILITY STATEMENT

For further explanation of various proofs explored in section 4, see appendix A and appendix B. For code implementations of our studies in section 5.3 and section 6, see our GitHub or the supplementary material. For the exact implementation of section 5.3, view the hyperparameters discussed in section 5.1. In regards to our findings in section 6, view appendix D for a more in-depth explanation.

# REFERENCES

Emmanuel Abbe. Community detection and stochastic block models: Recent developments. Journal of Machine Learning Research, 18(177):1-86, 2018.  
Mohammed Adnan, Shivam Kalra, and Hamid R. Tizhoosh. Representation learning of histopathology images using graph neural networks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2020.  
Réka Albert, Hawoong Jeong, and Albert-László Barabási. Error and attack tolerance of complex networks. Nature, 406(6794):378-382, 2000.  
Jesús Arroyo, Avanti Athreya, Joshua Cape, Guodong Chen, Carey E. Priebe, and Joshua T. Vogelstein. Inference for multiple heterogeneous networks with a common invariant subspace. Journal of Machine Learning Research, 22(142):1-49, 2021.  
Aseem Baranwal, Kimon Fountoulakis, and Aukosh Jagannath. Graph convolution for semi-supervised classification: Improved linear separability and out-of-distribution generalization. In The 38th International Conference on Machine Learning, 2021.  
Aseem Baranwal, Kimon Fountoulakis, and Aukosh Jagannath. Effects of graph convolutions in multi-layer networks. In The Eleventh International Conference on Learning Representations, 2023.  
Maya Bechler-Speicher, Ido Amos, Ran Gilad-Bachrach, and Amir Globerson. Graph neural networks use graphs when they shouldn't, 2023. arXiv:2309.04332.  
Danny Bickson. *Gaussian belief propagation: Theory and application*. PhD thesis, Hebrew University of Jerusalem, 2009.  
Norbert Binkiewicz, Joshua T. Vogelstein, and Karl Rohe. Covariate-assisted spectral clustering. Biometrika, 104(2):361-377, 2017.  
Dexiong Chen, Leslie O'Bray, and Karsten Borgwardt. Structure-aware transformer for graph representation learning. In International Conference for Machine Learning, 2022.  
Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. Adaptive universal generalized pagerank graph neural network. In International Conference on Learning Representations, 2021.  
Yash Deshpande, Andrea Montanari, Elchanan Mossel, and Subhabrata Sen. Contextual stochastic block models. In Advances in Neural Information Processing Systems, 2018.  
Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.  
Bailey K. Fosdick, Daniel B. Larremore, Joel Nishimura, and Johan Ugander. Configuring random graph models with fixed degree sequences. SIAM Review, 60(2):315-355, 2018.  
Kimon Fountoulakis, Amit Levi, Shenghao Yang, Aseem Baranwal, and Aukosh Jagannath. Graph attention retrospective. Journal of Machine Learning Research, 24(246):1-52, 2023.  
Chao Gao, Zongming Ma, Anderson Y. Zhang, and Harrison H. Zhou. Community detection in degree-corrected block models. The Annals of Statistics, 46(5), 2018.  
Jonathan Halcrow, Alexandru Mosoi, Sam Ruth, and Bryan Perozzi. Grale: Designing networks for graph learning. In the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2020.

William L. Hamilton. Theoretical motivations. In Graph Representation Learning, pp. 77-103. Springer, 2020.  
William L. Hamilton, Rex Ying, and Jure Lescovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing, 2017.  
Jing Huang and Jie Yang. UniGNN: a unified framework for graph and hypergraph neural networks. In International Joint Conferences on Artificial Intelligence Organization, 2021.  
Ningyuan Huang and Soledad Villar. A short tutorial on the Weisfeiler-Lehman test and its variants. In 2021-2021 IEEE International Conference on Acoustics, 2021.  
John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie, Bernardino Romero-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis Hassabis. Highly accurate protein structure prediction with AlphaFold. Nature, 596(7873):583-589, 2021.  
Brian Karrer and M. E. J. Newman. Stochastic blockmodels and community structure in networks. Phys. Rev. E, 83:016107, 2011.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In Internation Conference on Learning Representations, 2017.  
Pan Li and Jure Leskovec. The expressive power of graph neural networks. In Graph Neural Networks: Foundations, Frontiers, and Applications, pp. 63-98. Springer, 2022.  
Wei Lu. Learning guarantees for graph convolutional networks on the stochastic block model. In International Conference on Learning Representations, 2022.  
Yao Ma, Xiaorui Liu, Neil Shah, and Jiliang Tang. Is homophily a necessity for graph neural networks? In International Conference on Learning Representations, 2022.  
Sunil Kumar Maurya, Xin Liu, and Tsuyoshi Murata. Improving graph neural networks with simple architecture design, 2021. arXiv:2105.07634.  
Nikhil Mehta, Lawrence Carin Duke, and Piyush Rai. Stochastic blockmodels meet graph neural networks. In 36th International Conference on Machine Learning, 2019.  
Hoang N.T. and Takanori Maehara. Revisiting graph neural networks: All we have is low-pass filters, 2019. arXiv:1905.09550.  
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournaepau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Proceedings of Machine Learning Research, 12:2825-2830, 2011.  
Tiago P. Peixoto. The graph-tool python library. figshare, 2014.  
Kun Tu, Jian Li, Don Towsley, Dave Braines, and Liam D. Turner. gl2vec: Learning feature representation using graphlets for directed networks. In 2019 IEEE/ACM International Conference on Advances in Social Networkss Analysis and Mining, 2020.  
Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. In International Conference for Learning Representations, 2018.  
Ulrike von Luxburg. A tutorial on spectral clustering. Statistics and Computing, 17(4):395-416, 2007.

Lingfei Wu, Yu Chen, Kai Shen, Xiaojie Guo, Hanning Gao, Shucheng Li, Jian Pei, and Bo Long. Graph neural networks for natural language processing: A survey. Foundations and Trends in Machine Learning, 16(2):119-328, 2023a.  
Xinyi Wu, Zhengdao Chen, William Wei Wang, and Ali Jabbabaie. A non-asymptotic analysis of oversmoothing in graph neural networks. In The Eleventh International Conference on Learning Representations, 2023b.  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In *Internation Conference for Learning Representations*, 2019.  
Liang Yang, Wenmiao Zhou, Weihang Peng, Bingxin Niu, Junhua Gu, Chuan Wang, Xiaochun Cao, and Dongxiao He. Graph neural networks beyond compromise between attribute and topology. In the ACM Web Conference, 2022.  
Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai Koutra. Beyond homophily in graph neural networks: Current limitations and effective designs. In Advances in Neural Information Processing Systems, 2020.
