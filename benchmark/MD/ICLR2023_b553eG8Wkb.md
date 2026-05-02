# (LA)YER-NEIGH(BOR) SAMPLING: DEFUSING NEIGHBORHOOD EXPLOSION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Graph Neural Networks have recently received a significant attention, however, training them at a large scale still remains as a challenge. Minibatch training coupled with sampling is used to alleviate this challenge. However existing approaches either suffer from the neighborhood explosion phenomenon or does not have good performance. To deal with these issues, we propose a new sampling algorithm called LAYER-neighBOR sampling (LABOR). It is designed to be a direct replacement for Neighborhood Sampling with the same fanout hyperparameter while sampling much fewer vertices, without sacrificing quality. By design, the variance of the estimator of each vertex matches Neighbor Sampling from the point of view from a single vertex. In our experiments, we demonstrate the superiority of our approach when it comes to model convergence behaviour against Neighbor Sampling and also the other Layer Sampling approaches under the same limited vertex sampling budget constraints.

# 1 INTRODUCTION

Graph Neural Networks (GNN) Hamilton et al. (2017); Kipf & Welling (2017) have become de facto models for representation learning on graph structured data. Hence they have started being deployed in production systems Ying et al. (2018); Niu et al. (2020). These models iteratively update the node embeddings by passing messages along the direction of the edges in the given graph with nonlinearities in between different layers. With  $l$  layers, the computed node embedding contain information from the  $l$ -hop neighborhood of the seed vertex.

In the production setting, the GNN models need to be trained on billion-scale graphs (Ching et al., 2015; Ying et al., 2018). The training of these models takes hours to days even on distributed systems Zheng et al. (2022b;a). As in general Deep Neural Networks (DNN), it is more efficient to use minibatch training (Bertsekas, 1994) on GNNs, even though it is a bit trickier in this case. The node embeddings in GNNs depend recursively on their set of neighbors' embeddings, so when there are  $l$  layers, this dependency spans the  $l$ -hop neighborhood of the node. Real world graphs usually have a very small diameter and if  $l$  is large, the  $l$ -hop neighborhood may very well span the entire graph, also known as the Neighborhood Explosion Phenomenon (NEP) (Zeng et al., 2020).

To solve these issues, researchers proposed sampling a subgraph of the  $l$ -hop neighborhood of the nodes in the batch. There are mainly three different approaches: Node-based, Layer-based and Subgraph-based methods. Node-based sampling methods (Hamilton et al., 2017; Chen et al., 2018; Liu et al., 2020; Zhang et al., 2021) sample independently and recursively for each node. It was noticed that node-based methods sample subgraphs that are too shallow, i.e., with a low ratio of number of edges to nodes. Thus layer-based sampling methods were proposed (Chen et al., 2018; Zou et al., 2019; Huang et al., 2018; Dong et al., 2021), where the sampling for the whole layer is done collectively. On the other hand subgraph sampling methods (Chiang et al., 2019; Zeng et al., 2020; Hu et al., 2020) do not use the recursive layer by layer sampling scheme used in the node- and layer-based sampling methods and instead tend to use the same subgraph for all of the layers. Some of these sampling methods take the magnitudes of embeddings into account (Liu et al., 2020; Zhang et al., 2021; Huang et al., 2018), while others, such as Chen et al. (2018); Cong et al. (2021), cache the historical embeddings to reduce the variance of the computed approximate embeddings. There are methods sampling from a vertex cache Dong et al. (2021) filled with popular vertices. Most

of these approaches are orthogonal to each other and they can be incorporated into other sampling algorithms.

Node-based sampling methods suffer most from the NEP but they guarantee a good approximation for each embedding by ensuring each vertex gets  $k$  neighbors which is the only hyperparameter of the sampling algorithm. Layer-based sampling methods do not suffer as much from the NEP because number of vertices sampled is a hyperparameter but they can not guarantee that each vertex approximation is good enough and also their hyperparameters are hard to reason with, number of nodes to sample at each layer depends highly on the graph structure (as the numbers in Table 2 show). Subgraph sampling methods usually have more bias than their node- and layer-based counterparts. Hence, in this paper, we focus on the node- and layer-based sampling methods and combine their advantages. The major contributions of this work can be listed as follows:

- We propose a new sampling algorithm called LABOR, combining advantages of neighbor and layer sampling approaches using Poisson Sampling. LABOR correlates the sampling procedures of the given set of seed nodes so that the sampled vertices from different seeds have a lot of overlap, resulting into a big reduction in computation, memory and communication. Furthermore, LABOR has the same hyperparameters as neighbor sampling to use as a drop-in replacement.  
- We experimentally verify our findings, show that our proposed sampling algorithm LABOR outperforms both neighbor sampling and layer sampling approaches.

# 2 BACKGROUND

Graph Neural Networks: Given a directed graph  $\mathcal{G} = (V, E)$ , where  $V$  and  $E \subset V \times V$  are vertex and edge sets respectively,  $(t \rightarrow s) \in E$  denotes an edge from a source vertex  $t \in V$  to a destination vertex  $s \in V$ , and  $A_{ts}$  denotes the corresponding edge weight if provided. If we have a batch of seed vertices  $S \subset V$ , let us define  $l$ -hop neighborhood  $N^l(S)$  for the incoming edges as follows:

$$
N (s) = \{t | (t \rightarrow s) \in E \}, N ^ {1} (S) = N (S) = \cup_ {s \in S} N (s), N ^ {l} (S) = N \left(N ^ {l - 1} (S)\right) \tag {1}
$$

Let us also define the degree  $d_{s}$  of vertex  $s$  as  $d_{s} = |N(s)|$ . To simplify the discussion, let's assume uniform edge weights,  $A_{ts} = 1, \forall (t \to s) \in E$ . Then, our goal is to estimate the following for each vertex  $s \in S$ , where  $H_{t}^{(l-1)}$  is defined as the embedding of the vertex  $t$  at layer  $l-1$ , and  $W^{(l-1)}$  is the trainable weight matrix at layer  $l-1$ , and  $\sigma$  is the nonlinear activation function (Hamilton et al., 2017):

$$
Z _ {s} ^ {(l)} = \frac {1}{d _ {s}} \sum_ {t \rightarrow s} H _ {t} ^ {(l - 1)} W ^ {(l - 1)}, H _ {s} ^ {(l)} = \sigma \left(Z _ {s} ^ {(l)}\right) \tag {2}
$$

Exact Stochastic Gradient Descent: If we have a node prediction task and  $V_{t} \subseteq V$  is the set of training vertices,  $y_{s}, s \in V_{t}$  are the labels of the prediction task, and  $\ell$  is the loss function for the prediction task, then our goal is to minimize the following loss function:  $\frac{1}{|V_t|} \sum_{s \in V_t} \ell(y_s, Z_s^l)$ . Replacing  $V_{t}$  in the loss function with  $S \subset V_{t}$  for each iteration of gradient descent, we get stochastic gradient descent for GNNs. However with  $l$  layers, the computation dependency is on  $N^{l}(S)$ , which reaches large portions of the real world graphs, i.e.  $|N^{l}(S)| \approx |V|$ , making each iteration costly both in terms of computation and memory.

Neighbor Sampling: Neighbor sampling approach was proposed by Hamilton et al. (2017) to approximate  $Z_{s}^{(l)}$  for each  $s \in S$  with a subset of  $N^l(S)$ . Given a fanout hyperparameter  $k$ , this subset is computed recursively by randomly picking  $k$  neighbors for each  $s \in S$  from  $N(s)$  to form the next layer  $S^1$ , that is a subset of  $N^1(S)$ . If  $d_{s} \leq k$ , then the exact neighborhood  $N(s)$  is used. For the next layer,  $S^1$  is treated as the new set of seed vertices and this procedure is applied recursively.

Revisiting LADIES, Dependent Layer-based Sampling From now on, we will drop the layer notation and focus on a single layer and also ignore the nonlinearities. Let us define  $M_t = H_t W$  as a shorthand notation. Then our goal is to approximate:

$$
H _ {s} = \frac {1}{d _ {s}} \sum_ {t \rightarrow s} M _ {t} \tag {3}
$$

If we assign probabilities  $\pi_t > 0, \forall t \in N(S)$  and normalize it so that  $\sum_{t \in N(S)} \pi_t = 1$ , then use sampling with replacement to sample  $T \subset N(S)$  with  $|T| = n$ , where  $n$  is the number of vertices to sample given as input to the LADIES algorithm and  $T$  is a multiset possibly with multiple copies of the same vertices, and let  $\tilde{d}_s = |T \cap N(s)|$  which is the number of sampled vertices for a given vertex  $s$ , we get the following two possible estimators for each vertex  $s \in S$ :

$$
H _ {s} ^ {\prime} = \frac {1}{n d _ {s}} \sum_ {t \in T \cap N (s)} \frac {M _ {t}}{\pi_ {t}} \tag {4}
$$

$$
H _ {s} ^ {\prime \prime} = \frac {\sum_ {t \in T \cap N (s)} \frac {M _ {t}}{\pi_ {t}}}{\sum_ {t \in T \cap N (s)} \frac {1}{\pi_ {t}}} \tag {5}
$$

Note that  $H_{s}^{\prime}$  in Eq. 4 is the Thompson-Horvitz estimator and the  $H_{s}^{\prime \prime}$  in Eq. 5 is the Hajek estimator. For a comparison between the two and how to get an even better estimator by combining them, see Khan & Ugander (2021). The formulation in the LADIES paper uses  $H_{s}^{\prime}$ , but it proposes to row-normalize the sampled adjacency matrix, meaning they use  $H_{s}^{\prime \prime}$  in their implementation. However, analysing the variance of the Thompson-Horvitz estimator is simpler and its variance serves as an upper bound for the variance of the Hajek estimator when  $|M_{t}|$  and  $\pi_{t}$  are uncorrelated Khan & Ugander (2021); Dorfman (1997), which we assume to be true in our case.

$$
\operatorname {V a r} \left(H _ {s} ^ {\prime \prime}\right) \leq \operatorname {V a r} \left(H _ {s} ^ {\prime}\right) = \frac {1}{\tilde {d} _ {s} d _ {s} ^ {2}} \sum_ {t \rightarrow s} \pi_ {t} \sum_ {t ^ {\prime} \rightarrow s} \frac {\operatorname {V a r} \left(M _ {t ^ {\prime}}\right)}{\pi_ {t ^ {\prime}}} \tag {6}
$$

Since we assume that we do not have access to the computed embeddings and to simplify the analysis, we assume that  $\operatorname{Var}(M_t) = 1$  from now on. One can see that  $\operatorname{Var}(H_s')$  is minimized when  $\pi_t = p, \forall t \to s$  under the constraint  $\sum_{t \to s} \pi_t \leq pd_s$  for some constant  $p \in [0,1]$ , hence any deviation from uniformity increases the variance. The variance is also smaller the larger  $\tilde{d}_s$  is. However, in theory and in practice, there is no guarantee that each vertex  $s \in S$  will get any neighbors in  $T$ , not to mention equal numbers of neighbors. Some vertices will have pretty good estimators with thousands of samples and very low variances, while others might not even get a single neighbor sampled. For this reason, we designed LABOR so that every vertex in  $S$  will sample enough neighbors in expectation.

While LADIES is optimal from an approximate matrix multiplication perspective Chen et al. (2022), it is far from optimal in the case of nonlinearities and multiple layers. If there was a single layer, then the loss is computed by summing up the embeddings of the seed vertices, and it is optimal in this case. However, the existence of nonlinearities in-between layers and the fact that there are multiple layers necessitates that each vertex gets a good enough estimator with low enough variance. Also, LADIES gives a formulation using sampling with replacement instead of without replacement and that is sub-optimal from the variance perspective while its implementation uses sampling without replacement without taking care of the bias created thereby. In the next section, we will show how all of these problems are addressed by our newly proposed poisson sampling framework and LABOR sampling.

# 3 LABOR: LAYER NEIGHBOR SAMPLING

As mentioned previously, node-based sampling methods suffer from sampling too shallow subgraphs leading to NEP in just a few hops (e.g., see Table 2). Layer sampling methods Zou et al. (2019) attempt to fix this by sampling a fixed number of vertices in each layer, however they can not ensure that the estimators for the vertices are of high quality, and it is hard to reason how to choose the number of vertices to sample in each layer. The original paper of LADIES Zou et al. (2019) proposes using the same number of vertices in each layer while papers evaluating it found it is better to sample an increasing number of vertices in each layer Liu et al. (2020); Chen et al. (2022). There

is no systematic way to choose how many vertices to sample in each layer for the LADIES method, and since each graph has different density and connectivity structure, this choice highly depends on the graph in question. Therefore, due to its simplicity and high quality results, Neighbor Sampling currently seems to be the most popular of all of the sampling approaches and there exists high quality implementations on both CPUs and GPUs in the popular GNN frameworks Wang et al. (2019); Fey & Lenssen (2019).

We propose a new approach that combines the advantages of layer and neighbor sampling approaches using a vertex-centric variance based framework, reducing the number of sampled vertices drastically while ensuring the training quality does not suffer and matches the quality of neighbor sampling. Another advantage of our method is that the user only needs to choose the batch size and the fanout hyperparameters as in the Neighbor Sampling approach, the algorithm itself then samples the minimum number of vertices in the later layers in an unbiased way while ensuring each vertex gets enough neighbors and a good approximation.

# 3.1 LABOR SAMPLING

The design philosophy of LABOR Sampling is to create a direct alternative to Neighbor Sampling while incorporating the advantages of layer sampling. In layer sampling, the main idea can be summarized as individual vertices making correlated decisions while sampling their neighbors, because in the end if a vertex  $t$  is sampled, all edges  $t \to s$ , into the seed vertices  $S$  ( $s \in S$ ) are added to the sampled subgraph. This can be interpreted as vertices in  $S$  making a collective decision on whether to sample  $t$ .

The other thing to keep in mind is that, the existing layer sampling methods use sampling with replacement when doing importance sampling with unequal probabilities, because it is computationally intractable to compute the inclusion probabilities in the without replacement case. The Hajek estimator in the without replacement case with equal probabilities becomes:

$$
H _ {s} ^ {\prime \prime} = \frac {\sum_ {t \in T \cap N (s)} \frac {M _ {t}}{\bar {\pi} _ {t}}}{\sum_ {t \in T \cap N (s)} \frac {1}{\bar {\pi} _ {t}}} = \frac {\sum_ {t \in T \cap N (s)} M _ {t} | N (S) |}{\sum_ {t \in T \cap N (s)} | N (S) |} = \frac {1}{\tilde {d} _ {s}} \sum_ {t \in T \cap N (s)} M _ {t} \tag {7}
$$

and it has the variance:

$$
\operatorname {V a r} \left(H _ {s} ^ {\prime \prime}\right) = \frac {d _ {s} - \tilde {d} _ {s}}{d _ {s} - 1} \frac {1}{\tilde {d} _ {s}} \tag {8}
$$

Keeping these two points in mind, we use Poisson Sampling and design LABOR sampling around it. First, let us show how one can do layer sampling using Poisson sampling (PLADIES). Given probabilities  $\pi_t \in [0,1], \forall t \in N(S)$  so that  $\sum_{t \in N(S)} \pi_t = n$ , we include  $t \in N(S)$  in our sample  $T$  with probability  $\pi_t$  by flipping a coin for it, i.e., we sample  $r_t \sim U(0,1)$  and include  $t \in T$  if  $r_t \leq \pi_t$ . In the end,  $E[|T|] = n$  and we can still use the Hajek estimator  $H_s''$  or the Horvitz Thomson estimator  $H_s'$  to estimate  $H_s$ . This way of doing layer sampling is unbiased by construction and achieves the same goal in linear time in constraint to the quadratic time debiasing approach explained in Chen et al. (2022). In this case, the variance becomes (Williams et al., 1998):

$$
\operatorname {V a r} \left(H _ {s} ^ {\prime \prime}\right) \leq \operatorname {V a r} \left(H _ {s} ^ {\prime}\right) = \frac {1}{d _ {s} ^ {2}} \sum_ {t \rightarrow s} \frac {1}{\pi_ {t}} - \frac {1}{d _ {s}} \tag {9}
$$

One can notice the existence of the minus term  $\frac{1}{d_s}$ , and it enables the variance to converge to 0 if all  $\pi_t = 1$  and we get the exact result. However, in the sampling with replacement case, the variance goes to 0 only as the sample size goes to infinity.

This type of mimicking Layer Sampling with Poisson Sampling still has the disadvantage that  $\tilde{d}_s$  varies wildly for different  $s$ . To overcome this and mimic Neighbor Sampling where  $E[\tilde{d}_s] = \min(d_s, k)$ , where  $k$  is a given fanout hyperparameter. For given  $\pi_t \geq 0, \forall t \in N(S)$  denoting unnormalized probabilities, for a given  $s$ , let us define  $c_s$  as the quantity satisfying the following equality if  $k < d_s$ , otherwise  $c_s = \max_{t \to s} \frac{1}{\pi_t}$ :

$$
\frac {1}{d _ {s} ^ {2}} \sum_ {t \rightarrow s} \frac {1}{\min  \left(1 , c _ {s} \pi_ {t}\right)} - \frac {1}{d _ {s}} = \frac {1}{k} - \frac {1}{d _ {s}} \tag {10}
$$

Note that  $\frac{1}{k} -\frac{1}{d_s}$  is the variance when  $\pi_t = \frac{k}{d_s},\forall t\in N(s)$  so that  $E[\tilde{d}_s] = k$  . Also note that:

$$
\frac {1}{k} - \frac {1}{d _ {s}} - \frac {d _ {s} - k}{d _ {s} - 1} \frac {1}{k} = \frac {d _ {s} - k}{k d _ {s}} - \frac {d _ {s} - k}{d _ {s} - 1} \frac {1}{k} = \frac {d _ {s} - k}{k} \left(\frac {1}{d _ {s}} - \frac {1}{d _ {s} - 1}\right) <   0 \tag {11}
$$

meaning that the variance target we set through Eq. 10 is strictly better than Neighbor Sampling's variance in Eq. 8 and it will result in  $E[\tilde{d}_s] \geq k$  with strict equality in the uniform probability case. Then each vertex  $s \in S$  samples  $t \to s$  with probability  $c_s \pi_t$ . To keep the collective decision making, we sample  $r_t \sim U(0,1), \forall t \in N(S)$  and vertex  $s$  samples vertex  $t$  if and only if  $r_t \leq c_s \pi_t$ . Note that if we use a uniform random variable for each edge  $r_{ts}$  instead of each vertex  $r_t$ , and if  $\pi$  is uniformly initialized, then we get the same behaviour as Neighbor Sampling.

# 3.2 IMPORTANCE SAMPLING

Given the sampling procedure above, one wonders how different choices of  $\pi \geq 0$  will affect  $|T|$ , the total number of unique vertices sampled. In our case, it is extremely easy to compute:

$$
E [ | T | ] = \sum_ {t \in N (S)} \mathbb {P} (t \in T) = \sum_ {t \in N (S)} \min  \left(1, \pi_ {t} \max  _ {t \rightarrow s} c _ {s}\right) \tag {12}
$$

In particular, we need to find  $\pi^{*}\geq 0$  minimizing  $E[|T|]$

$$
\pi^ {*} = \underset {\pi \geq 0} {\arg \min } \sum_ {t \in N (S)} \min  \left(1, \pi_ {t} \max  _ {t \rightarrow s} c _ {s}\right) \tag {13}
$$

Note that for any given  $\pi \geq 0$ ,  $E[|T|]$  is the same for any vector multiple  $x\pi, x \in \mathbb{R}^+$ , meaning that the objective function is homogenous of degree 0.

# 3.3 COMPUTING  $c$  AND  $\pi^{*}$

Note that  $c_{s}$  was defined to be the scalar satisfying the following equality involving the variance of the estimator of  $H_{s}$ :

$$
\frac {1}{d _ {s} ^ {2}} \sum_ {t \rightarrow s} \frac {1}{\min  \left(1 , c _ {s} \pi_ {t}\right)} - \frac {1}{d _ {s}} = \frac {1}{k} - \frac {1}{d _ {s}} \tag {14}
$$

If we rearrange the terms, we get:

$$
\sum_ {t \rightarrow s} \frac {1}{\min  \left(1 , c _ {s} \pi_ {t}\right)} = \frac {d _ {s} ^ {2}}{k} \tag {15}
$$

One can see that the left hand side of the equality is monotonically decreasing with respect to  $c_{s} \geq 0$ . Thus one can use binary search to find the  $c_{s}$  satisfying the above equality to any precision needed. But we opt to use the following iterative algorithm to compute it:

$$
v _ {s} ^ {(0)} = 0, c _ {s} ^ {(0)} = \frac {k}{d _ {s} ^ {2}} \sum_ {t \rightarrow s} \frac {1}{\pi_ {t}} \tag {16}
$$

$$
c _ {s} ^ {(i + 1)} = \frac {c _ {s} ^ {(i)}}{\frac {d _ {s} ^ {2}}{k} - v _ {s} ^ {(i)}} \left(- v _ {s} ^ {(i)} + \sum_ {t \rightarrow s} \frac {1}{\min \left(1 , c _ {s} ^ {(i)} \pi_ {t}\right)}\right), v _ {s} ^ {(i + 1)} = \sum_ {t \rightarrow s} \mathbb {1} \left[ c _ {s} ^ {(i + 1)} \pi_ {t} \geq 1 \right] \tag {17}
$$

This iterative algorithm converges in at most  $d_{s}$  steps and the convergence is exact and monotonic from below. One can also implement it in linear time  $\mathcal{O}(d_s)$  if  $\{\pi_t\mid t\to s\}$  is sorted and making use of precomputed prefix sum arrays. Note that  $c = c(\pi)$ , meaning that  $c$  is a function of the given probability vector  $\pi$ . To compute  $\pi^*$ , we use a similar fixed point iteration as follows:

$$
\pi^ {(0)} = 1, \forall t \in N (S): \pi_ {t} ^ {(i + 1)} = \pi_ {t} ^ {(i)} \max  _ {t \rightarrow s} c _ {s} (\pi^ {(i)}) \tag {18}
$$

Thus, we alternate between computing  $c = c(\pi)$ , meaning  $c$  is computed with the current  $\pi$ , and updating the  $\pi$  with the computed  $c$  values. Each step of this iteration is guaranteed to lower the objective function value in Eq. 13 until convergence to a fixed point, see the Appendix A.1. Modified method for a given nonuniform weight matrix  $A_{ts}$  is explained in the Appendix A.3.

Table 1: Datasets used in experiments, numbers of vertices, edges, avg. degree, features, sampling budget used, training, validation and test vertex split.  

<table><tr><td>Dataset</td><td>|V|</td><td>|E|</td><td>|E|/V|</td><td># feats.</td><td>|V3| budget</td><td>train - val - test (%)</td></tr><tr><td>products</td><td>2.45M</td><td>61.9M</td><td>25.26</td><td>100</td><td>60k</td><td>8 - 2 - 90</td></tr><tr><td>reddit</td><td>233K</td><td>115M</td><td>493.56</td><td>602</td><td>400k</td><td>66 - 10 - 24</td></tr><tr><td>yelp</td><td>717K</td><td>14.0M</td><td>19.52</td><td>300</td><td>200k</td><td>75 - 10 - 15</td></tr><tr><td>flickr</td><td>89.2K</td><td>900K</td><td>10.09</td><td>500</td><td>70k</td><td>50 - 25 - 25</td></tr></table>

# 3.4 CHOOSING HOW MANY NEIGHBORS TO SAMPLE

The variance of Poisson Sampling when  $\pi_t = \frac{k}{d_s}$  is  $\frac{1}{k} - \frac{1}{d_s}$ . One might question why we are trying to match the variance of Neighbor Sampling and choose to use a fixed fanout for all the seed vertices. In the uniform probability case, if we have already sampled some set of edges for all vertices in  $S$ , and want to sample one more edge, the question becomes which vertex in  $S$  should we sample the new edge for? Our answer to this question is the vertex  $s$ , whose variance would improve the most. If currently vertex  $s$  has  $\tilde{d}_s$  edges sampled, then sampling one more edge for it would improve its variance from  $\frac{1}{\tilde{d}_s} - \frac{1}{d_s}$  to  $\frac{1}{1 + \tilde{d}_s} - \frac{1}{d_s}$ . Since the derivative of the variance with respect to  $\tilde{d}_s$  is monotonic, we are allowed to reason about the marginal improvements by comparing their derivatives, which is:

$$
\frac {\partial \left(\frac {1}{\tilde {d} _ {s}} - \frac {1}{d _ {s}}\right)}{\partial \tilde {d} _ {s}} = - \frac {1}{\tilde {d} _ {s} ^ {2}} \tag {19}
$$

Notice that the derivative does not depend on the degree  $d_{s}$  of the vertex  $s$  at all, and the greater the magnitude of the derivative, the more improvement the variance of a vertex gets by sampling one more edge. Thus, choosing any vertex  $s$  with least number of edges sampled would work for us, that is:  $s = \arg \min_{s' \in S} \tilde{d}_{s'}$ . In the light of this observation, one can see that it is optimal to sample an equal number of edges for each vertex in  $S$ . This is one of the reasons LADIES is not efficient with respect to the number of edges it samples. On graphs with skewed degree distributions, it samples thousands of edges for some seed vertices, which contribute very small amounts to the variance of the estimator since it is already very close to 0.

# 4 EXPERIMENTS

In this section, we empirically evaluate the performance of each method in the node-prediction setting on the following datasets: reddit (Hamilton et al., 2017), products (Hu et al., 2020), yelp, flickr (Zeng et al., 2020). More details about these datasets are given in Table 1. We compare LABOR variants LABOR-0, LABOR-1 and LABOR- $*$ , where  $0, 1, *$  stand for the number of fixed point iterations applied to optimize 13 respectively, against NS (Neighbor Sampling), LADIES and PLADIES sampling methods, where PLADIES is the unbiased Poisson Sampling variant of LADIES introduced in Section 3.1. We do not include Fast-GCN in our comparisons as it is superseeded by the LADIES paper. The works of Liu et al. (2020); Zhang et al. (2021); Huang et al. (2018); Cong et al. (2021); Dong et al. (2021) are not included in the comparisons because they either take into account additional information such as historical embeddings or their magnitudes or they have a different sampling structure such a vertex cache to sample from. Also the techniques in these papers are mostly orthogonal to the sampling problem and algorithms discussed in this paper. We evaluate all the baselines on the GCN model in Eq. 2 with 3 layers, with 256 hidden dimension and residual skip connections enabled. We use the Adam optimizer (Kingma & Ba, 2014) with 0.001 learning rate. We carried out our experiments using the DGL framework (Wang et al., 2019) with the Pytorch backend (Paszke et al., 2019) $^{1}$ . Experiments were repeated 100 times and averages are presented.

We will first show that despite the different number of sampled vertices, LABOR and NS training loss curves are almost the same in Section 4.1 with the same fanout and batch size hyperparam-

![](images/94aeea1fcb528de44446a32b0b8150c9a6240c00280531e76969af7b3373504e.jpg)  
Figure 1: The loss curve on different datasets with same batch size. The soft edges represent the confidence interval. Number of sampled vertices and edges can be found in Table 2.

![](images/8a1115ca983ec689d6a2293ecf3b0ca8f75b8d4ddc78068a390d639f5db2819b.jpg)

![](images/5b376992876b543a8985e8f08df6c443b39d7cecd1130a13ba69020edbe4fb4e.jpg)

![](images/cae260757016259c41a82ff2c2a8f7977dd104af099bcca2fc56f1ef12d24a00.jpg)

Table 2: Average number of vertices and edges sampled in different layers (All the numbers are in thousands, lower is better). Last column shows iterations(minibatches) per second (it/s) (higher is better). The hyperparameters of LADIES and PLADIES were picked to roughly match the number of vertices sampled by the LABOR-* to get a fair comparison. The convergence curves can be found in Figure 1. The timing information was measured on an NVIDIA T4 GPU. Green stands for best, red stands for worst results, with a  $5\%$  cutoff.  

<table><tr><td>Dataset</td><td>Algo.</td><td>|V3|</td><td>|E3|</td><td>|V2|</td><td>|E2|</td><td>|V1|</td><td>|E1|</td><td>|V0|</td><td>it/s</td></tr><tr><td rowspan="6">reddit</td><td>PLADIES</td><td>24.0</td><td>3450</td><td>14.1</td><td>1010</td><td>5.97</td><td>35.2</td><td>1</td><td>1.7</td></tr><tr><td>LADIES</td><td>25.3</td><td>3520</td><td>14.5</td><td>1010</td><td>5.97</td><td>34.5</td><td>1</td><td>1.8</td></tr><tr><td>LABOR-*</td><td>24.1</td><td>1070</td><td>13.7</td><td>435</td><td>6.03</td><td>26.9</td><td>1</td><td>4.1</td></tr><tr><td>LABOR-1</td><td>26.6</td><td>261</td><td>14.4</td><td>116</td><td>6.12</td><td>16.7</td><td>1</td><td>24.8</td></tr><tr><td>LABOR-0</td><td>35.8</td><td>177</td><td>17.8</td><td>67.1</td><td>6.77</td><td>9.64</td><td>1</td><td>37.6</td></tr><tr><td>NS</td><td>167</td><td>682</td><td>68.3</td><td>100</td><td>10.1</td><td>9.65</td><td>1</td><td>14.2</td></tr><tr><td rowspan="6">products</td><td>PLADIES</td><td>160</td><td>2620</td><td>51.2</td><td>320</td><td>9.68</td><td>12.0</td><td>1</td><td>4.1</td></tr><tr><td>LADIES</td><td>165</td><td>2560</td><td>51.8</td><td>302</td><td>9.68</td><td>11.8</td><td>1</td><td>4.2</td></tr><tr><td>LABOR-*</td><td>166</td><td>1250</td><td>51.8</td><td>167</td><td>9.78</td><td>10.6</td><td>1</td><td>6.2</td></tr><tr><td>LABOR-1</td><td>178</td><td>799</td><td>53.4</td><td>136</td><td>9.78</td><td>10.5</td><td>1</td><td>21.3</td></tr><tr><td>LABOR-0</td><td>237</td><td>615</td><td>62.4</td><td>99.6</td><td>10.1</td><td>9.89</td><td>1</td><td>32.5</td></tr><tr><td>NS</td><td>513</td><td>944</td><td>95.4</td><td>106</td><td>10.6</td><td>9.89</td><td>1</td><td>24.6</td></tr><tr><td rowspan="6">yelp</td><td>PLADIES</td><td>100</td><td>1370</td><td>29.5</td><td>187</td><td>6.16</td><td>7.07</td><td>1</td><td>5.1</td></tr><tr><td>LADIES</td><td>102</td><td>1390</td><td>29.7</td><td>191</td><td>6.17</td><td>7.08</td><td>1</td><td>5.3</td></tr><tr><td>LABOR-*</td><td>105</td><td>991</td><td>30.7</td><td>158</td><td>6.15</td><td>6.83</td><td>1</td><td>13.3</td></tr><tr><td>LABOR-1</td><td>109</td><td>447</td><td>31.0</td><td>96.1</td><td>6.18</td><td>6.76</td><td>1</td><td>27.3</td></tr><tr><td>LABOR-0</td><td>138</td><td>318</td><td>35.1</td><td>53.9</td><td>6.25</td><td>6.29</td><td>1</td><td>27.2</td></tr><tr><td>NS</td><td>188</td><td>392</td><td>42.5</td><td>54.8</td><td>6.32</td><td>6.27</td><td>1</td><td>23.0</td></tr><tr><td rowspan="6">flickr</td><td>PLADIES</td><td>55.2</td><td>321</td><td>24.9</td><td>86.0</td><td>6.23</td><td>6.93</td><td>1</td><td>10.2</td></tr><tr><td>LADIES</td><td>55.9</td><td>323</td><td>25.1</td><td>86.2</td><td>6.23</td><td>6.93</td><td>1</td><td>10.5</td></tr><tr><td>LABOR-*</td><td>56.6</td><td>308</td><td>25.6</td><td>84.7</td><td>6.29</td><td>6.95</td><td>1</td><td>20.3</td></tr><tr><td>LABOR-1</td><td>57.7</td><td>242</td><td>25.9</td><td>73.4</td><td>6.29</td><td>6.93</td><td>1</td><td>32.7</td></tr><tr><td>LABOR-0</td><td>65.9</td><td>219</td><td>29.1</td><td>51.9</td><td>6.37</td><td>6.71</td><td>1</td><td>33.3</td></tr><tr><td>NS</td><td>73.3</td><td>244</td><td>32.8</td><td>51.9</td><td>6.37</td><td>6.72</td><td>1</td><td>31.7</td></tr></table>

eters. We will match the hyperparameters of LADIES with the number of vertices sampled on LABOR and see whose batches have better quality. Then, we will show what happens when different sampling algorithms are given the same budget and compare their vertex sampling efficiency in Section 4.2. Section 4.3 shows the reduction in the number of vertices sampled with each fixed point iteration. PLADIES is the poisson sampling version of LADIES the ladies algorithm as described in Section 3.1.

# 4.1 COMPARISON AGAINST NEIGHBOR SAMPLING AND LADIES

In this experiment, we set the batch size to 1,000 and the fanout  $k = 10$  for LABOR and NS methods to see the difference in the sizes of the sampled subgraphs and also whether convergence behaviour is the same. In Figure 1, we can see that the convergence curves of both NS and LABOR variants are almost the same showing that sampling smaller subgraphs does not really affect the batch quality. Table 2 shows the difference of the sampled subgraph sizes in each layer. One can see that on reddit, LABOR-* samples  $6.9 \times$  fewer vertices in the 3rd layer while keeping the same convergence behaviour. On the flickr dataset however, LABOR-* samples only  $1.3 \times$  fewer vertices. The amount of difference depends on two factors. The first is the amount of overlap of neighbors among the vertices in  $S$ . If the neighbors of vertices in  $S$  did not overlap at all, then one obviously can not do better than NS. The second is the average degree of the graph. With a fanout of 10, both Neighbor Sampling and LABOR has to copy the whole neighborhood of a vertex  $s$  with degree  $d_s \leq 10$ . Thus for such graphs, it is expected that there is a small difference because for a lot of the vertices, their whole neighborhood is copied. If we look at Table 1, the average degree of the flickr graph is 10.09, and thus there is only a small difference between LABOR and NS.

In Table 2, the number of sampled edges is another important metric to look at. We can see that LABOR-0 reduces both the number of vertices and edges sampled. On the other hand, when importance sampling is enabled, the number of vertices sampled goes down while number of edges sampled goes up. This is because when importance sampling is used, inclusion probabilities become nonuniform and it takes more edges per seed vertex to get a good approximation (see Eq. 10).

The hyperparameters of LADIES and PLADIES were picked to match LABOR-\* so that all methods have the same sampling budget in each layer (see Table 2). Figure 1 shows that, in terms of loss curve, LADIES and PLADIES perform almost the same on all but the flickr dataset, in which case there is a big difference between the two in favor of PLADIES. We also see that LABOR variants either match the quality of PLADIES or outperform it, as can be seen on the products dataset.

Looking at Table 2, we can see that LABOR-0 has the best runtime performance across all datasets. This is both due to lack of the overhead of performing the fixed point iterations and also it sampled the fewest edges, compared to the other LABOR variants. By design, all LABOR variants should have the same convergence curves, as seen in Figure 1. Then, the decision of which variant to use depends on one factor: feature access speed. If vertex features were stored on a slow storage medium (such as, on host memory accessed over PCI-E), then minimizing number of sampled vertices would become the highest priority, in which case, one should pick LABOR-\*. Depending on the relative vertex feature access performance and the performance of the training processor, one can choose to use LABOR- $j$ , the faster feature access, the lower the  $j$ .

# 4.2 EVALUATION OF VERTEX SAMPLING EFFICIENCY

In this experiment, we set a limit on the number of sampled vertices and after each epoch, we adjust the batch size to match the given vertex budget starting with an initial batch size of 1,000. The budgets used were picked around the same magnitude with numbers in the Table 2 in the  $|V_{3}|$  column and can be found in Table 1. Figure 2 displays the result of this experiment. We can see that the more vertex efficient the sampling method is, the larger batch size it can use during training. Number of vertices sampled is not a function of the batch size for the LADIES algorithm so we do not include it in this comparison. All of the experiments were repeated 100 times and their averages were plotted, that is why our convergence plots are smooth and differences are clear. The most striking result in this experiment is that there is a two order magnitude difference in batch sizes of LABOR-* and NS algorithms on the reddit dataset.

# 4.3 IMPORTANCE SAMPLING NUMBER OF FIXED POINT ITERATIONS

In this section, we look at the convergence behaviour of the fixed point iterations described in Section 3.3. Table 3 shows the number of sampled vertices in the last layer with respect to the number of fixed point iterations applied. In this table, the  $\infty$  stands for applying the fixed point iterations until convergence, and convergence occurs in at most 15 iterations in practice before the relative change in the objective function is less than  $10^{-4}$ . One can see that most of the reduction in the objective function 13 occurs after the first iteration, and the remaining iterations have diminishing

![](images/9cb7683e7f920c8f09ed17c4081ab7c991359c8486049da6b7ee9c11c4e5c405.jpg)  
Figure 2: Vertex sampling efficiency under the same sampling budget. A starting batch size of 1k is used and the batch size is adjusted at the end of each epoch to better match the vertex budget. The first row of plots is for the batch size and the second row of plots shows the number of vertices sampled in the last layer,  $|V_3|$  of running averages. The last row shows the training loss curves.

returns. Applying the fixed point iterations until convergence can save from  $14\% - 33\%$  depending on the dataset. The monotonically decreasing numbers provide empirical evidence for proof in the Appendix A.1.

Table 3: Number of vertices in 3rd layer w.r.t the number of fixed point iterations (its).  $\infty$  stands for applying the fixed point iterations until convergence, i.e., LABOR- $*$ , 1 its stands for LABOR-1 etc. Numbers are in thousands.  

<table><tr><td>Dataset</td><td>NS</td><td>0 its</td><td>1 its</td><td>2 its</td><td>3 its</td><td>∞ its</td></tr><tr><td>reddit</td><td>167</td><td>35.8</td><td>26.5</td><td>25.0</td><td>24.7</td><td>24.2</td></tr><tr><td>products</td><td>513</td><td>237</td><td>178</td><td>170</td><td>169</td><td>166</td></tr><tr><td>yelp</td><td>188</td><td>138</td><td>109</td><td>106</td><td>105</td><td>105</td></tr><tr><td>flickr</td><td>73.3</td><td>65.9</td><td>57.6</td><td>56.8</td><td>56.6</td><td>56.5</td></tr></table>

# 5 CONCLUSIONS

In this paper, we introduced LABOR sampling, a novel way to combine layer and neighbor sampling approaches using a vertex-variance centric framework. We then transform the sampling problem into an optimization problem where the constraint is to match neighbor sampling variance for each vertex while sampling the fewest number of vertices. We show how to minimize this new objective function via fixed-point iterations. On dense datasets like Reddit, we show that our approach can sample a subgraph with  $7 \times$  fewer vertices without degrading the batch quality. We also show that compared to LADIES, LABOR converges faster with same sampling budget.

# REFERENCES

D.P. Bertsekas. Incremental least squares methods and the extended kalman filter. In Proceedings of 1994 33rd IEEE Conference on Decision and Control, volume 2, pp. 1211-1214 vol.2, 1994. doi: 10.1109/CDC.1994.411166.  
Jianfei Chen, Jun Zhu, and Le Song. Stochastic training of graph convolutional networks with variance reduction. 35th International Conference on Machine Learning, ICML 2018, 3:1503-1532, 2018.  
Yifan Chen, Tianning Xu, Dilek Hakkani-Tur, Di Jin, Yun Yang, and Ruoqing Zhu. Calibrate and Debias Layer-wise Sampling for Graph Convolutional Networks. jun 2022. URL http://arxiv.org/abs/2206.00583.  
Wei Lin Chiang, Yang Li, Xuanqing Liu, Samy Bengio, Si Si, and Cho Jui Hsieh. Cluster-GCN: An efficient algorithm for training deep and large graph convolutional networks. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 257-266. Association for Computing Machinery, jul 2019. doi: 10.1145/3292500.3330925.  
Avery Ching, Sergey Edunov, Maja Kabiljo, Dionysios Logothetis, and Sambavi Muthukrishnan. One trillion edges: Graph processing at facebook-scale. Proc. VLDB Endow., 8(12): 1804-1815, aug 2015. doi: 10.14778/2824032.2824077. URL https://doi.org/10. 14778/2824032.2824077.  
Weilin Cong, Morteza Ramezani, and Mehrdad Mahdavi. On the Importance of Sampling in Training GCNs: Tighter Analysis and Variance Reduction. 2021. URL http://arxiv.org/abs/2103.02696.  
Jialin Dong, Da Zheng, Lin F. Yang, and George Karypis. Global Neighbor Sampling for Mixed CPU-GPU Training on Giant Graphs. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 289-299, 2021. doi: 10.1145/3447548.3467437.  
Alan H Dorfman. The Hajek Estimator Revisited. (4):760-765, 1997. URL http://www.asasrms.org/Proceedings/papers/1997_130.pdf.  
Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric, 2019. URL https://arxiv.org/abs/1903.02428.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf.  
C. A. R. Hoare. Algorithm 65: Find. Commun. ACM, 4(7):321-322, jul 1961. doi: 10.1145/366622.366647. URL https://doi.org/10.1145/366622.366647.  
Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. Advances in Neural Information Processing Systems, 2020-Decem(NeurIPS):1-34, 2020.  
Wenbing Huang, Tong Zhang, Yu Rong, and Junzhou Huang. Adaptive sampling towards fast graph representation learning. Advances in Neural Information Processing Systems, 2018-Decem(Nips): 4558-4567, 2018.  
Samir Khan and Johan Ugander. Adaptive normalization for IPW estimation. pp. 1-31, 2021. URL http://arxiv.org/abs/2106.07695.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2014. URL https://arxiv.org/abs/1412.6980.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. 5th International Conference on Learning Representations, ICLR 2017 - Conference Track Proceedings, pp. 1-14, 2017.

Ziqi Liu, Zhengwei Wu, Zhiqiang Zhang, Jun Zhou, Shuang Yang, Le Song, and Yuan Qi. Bandit samplers for training graph neural networks. Advances in Neural Information Processing Systems, 2020-Decem, 2020.  
Xichuan Niu, Bofang Li, Chenliang Li, Rong Xiao, Haochuan Sun, Hongbo Deng, and Zhenzhong Chen. A dual heterogeneous graph attention network to improve long-tail performance for shop search in e-commerce. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '20, pp. 3405-3415, 2020. doi: 10.1145/3394486.3403393.  
E Ohlsson. Sequentialoisson sampling. Journal of Official Statistics-Stockholm-, 14(2):149-162, 1998.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019.  
Minjie Wang, Da Zheng, Zihao Ye, Quan Gan, Mufei Li, Xiang Song, Jinjing Zhou, Chao Ma, Lingfan Yu, Yu Gai, Tianjun Xiao, Tong He, George Karypis, Jinyang Li, and Zheng Zhang. Deep graph library: A graph-centric, highly-performant package for graph neural networks, 2019. URL https://arxiv.org/abs/1909.01315.  
Michael S Williams, Hans T Schreuder, and Gerardo H Terrazas. Poisson Sampling – The Adjusted and Unadjusted Estimator Revisited. pp. 12, 1998.  
Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, and Jure Leskovec. Graph convolutional neural networks for web-scale recommender systems. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 974-983, 2018. doi: 10.1145/3219819.3219890.  
Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor Prasanna. GraphSAINT: Graph sampling based inductive learning method. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=BJe8pkHFwS.  
Qingru Zhang, David Wipf, Quan Gan, and Le Song. A Biased Graph Neural Network Sampler with Near-Optimal Regret. (NeurIPS):1-25, 2021. URL http://arxiv.org/abs/2103.01089.  
Che Zheng, Hongzhi Chen, Yuxuan Cheng, Zhezheng Song, Yifan Wu, Changji, Li, James Cheng, Han Yang, and Shuai Zhang. Bytegnn: Efficient graph neural network training at large scale. Proc. VLDB Endow., 15:1228-1242, 2022a.  
Da Zheng, Xiang Song, Chengru Yang, Dominique LaSalle, and George Karypis. Distributed hybrid cpu andgpu training for graph neural networks on billion-scale heterogeneous graphs. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 4582-4591, 2022b.  
Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, and Quanquan Gu. Layer-dependent importance sampling for training deep and large graph convolutional networks. Advances in Neural Information Processing Systems, 32(NeurIPS), 2019.
