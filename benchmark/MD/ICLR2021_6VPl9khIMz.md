# ADAPTIVE STACKED GRAPH FILTER

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study Graph Convolutional Networks (GCN) from the graph signal processing viewpoint by addressing a difference between learning graph filters with fully-connected weights versus trainable polynomial coefficients. We find that by stacking graph filters with learnable polynomial parameters, we can build a highly adaptive and robust vertex classification model. Our treatment here relaxes the low-frequency (or equivalently, high homophily) assumptions in existing vertex classification models, resulting a more ubiquitous solution in terms of spectral properties. Empirically, by using only one hyper-parameter setting, our model achieves strong results on most benchmark datasets across the frequency spectrum.

# 1 INTRODUCTION

The semi-supervised vertex classification problem (Weston et al., 2012; Yang et al., 2016) in attributed graphs has become one of the most fundamental machine learning problems in recent years. This problem is often associated with its most popular recent solution, namely Graph Convolutional Networks (Kipf & Welling, 2017). Since the GCN proposal, there has been a vast amount of research to improve its scalability (Hamilton et al., 2017; Chen et al., 2018; Wu et al., 2019) as well as performance (Liao et al., 2019; Li et al., 2019; Pei et al., 2020).

Existing vertex classification models often (implicitly) assume that the graph has large vertex homophily (Pei et al., 2020), or equivalently, low-frequency property (Li et al., 2019; Wu et al., 2019); see Section 2.1 for graph frequency. However, this assumption is not true in general. For instance, let us take the Wisconsin dataset (Table 1), which captures a network of students, faculty, staff, courses, and projects. These categories naturally exhibit different frequency patterns<sup>1</sup>. Connections between people are often low-frequency, while connections between topics and projects are often midrange. This problem becomes apparent as GCN-like models show low accuracies on this dataset; for example, see (Pei et al., 2020; Chen et al., 2020b; Liu et al., 2020).

This paper aims at establishing a GCN model for the vertex classification problem (Definition 1) that does not rely on any frequency assumption. Such a model can be applied to ubiquitous datasets without any hyper-parameter tuning for the graph structure.

Contributions. By observing the relation between label frequency and performance of existing GCN-like models, we propose to learn the graph filters coefficients directly rather than learning the MLP part of a GCN-like layer. We use filter stacking to implement a trainable graph filter, which is capable of learning any filter function. Our stacked filter construction with novel learnable filter parameters is easy to implement, sufficiently expressive, and less sensitive to the filters' degree. By using only one hyper-parameter setting, we show that our model is more adaptive than existing work on a wide range of benchmark datasets.

The rest of our paper is organized as follows. Section 2 introduces notations and analytical tools. Section 3 provides insights into the vertex classification problem and motivations to our model's design. Section 4 presents an implementation of our model. Section 5 summarizes related literature with a focus on graph filters and state-of-the-art models. Section 6 compares our model and other existing methods empirically. We also provide additional experimental results in Appendix A.

# 2 PRELIMINARIES

We consider a simple undirected graph  $G = (V, G)$ , where  $V = \{1, \dots, n\}$  is a set of  $n$  vertices and  $E \subseteq V \times V$  is a set of edges. A graph  $G$  is called an attributed graph, denoted by  $G(X)$ , when it is associated with a vertex feature mapping  $X: V \mapsto \mathbb{R}^d$ , where  $d$  is the dimension of the features. We define the following vertex classification problem, also known in the literature as the semi-supervised vertex classification problem (Yang et al., 2016).

Definition 1 (Vertex Classification Problem). We are given an attributed graph  $G(X)$ , a set of training vertices  $V_{\mathrm{tr}} \subset V$ , training labels  $Y_{\mathrm{tr}}: V_{\mathrm{tr}} \to \mathcal{C}$ , and label set  $\mathcal{C}$ . The task is to find a model  $h: V \to \mathcal{C}$  using the training data  $(V_{\mathrm{tr}}, Y_{\mathrm{tr}})$  that approximates the true labeling function  $Y: V \to \mathcal{C}$ .

Let  $A$  be the adjacency matrix of the graph  $G$ , i.e.,  $A_{i,j} = 1$  if  $(i,j) \in E$  and 0 otherwise. Let  $d_i = \sum_j A_{ij}$  be the degree of vertex  $i \in V$ , and let  $D = \mathrm{diag}(d_1, \ldots, d_n)$  be the  $n \times n$  diagonal matrix of degrees. Let  $L = D - A$  be the combinatorial graph Laplacian. Let  $\mathcal{L} = D^{-1/2}LD^{-1/2}$  be the symmetric normalized graph Laplacian. We mainly focus on the symmetric normalized graph Laplacian due to its interesting spectral properties: (1) its eigenvalues range from 0 to 2; and (2) the spectral properties can be compared between different graphs (Chung & Graham, 1997). In recent literature, the normalized adjacency matrix with added self-loops,  $\tilde{A} = I - \mathcal{L} + c$ , is often used as the propagation matrix, where  $c$  is some diagonal matrix.

# 2.1 GRAPH FREQUENCY

Graph signal processing (Shuman et al., 2012) extends "frequency" concepts in the classical signal processing to graphs using the graph Laplacian. Let  $\mathcal{L} = U\Lambda U^{\top}$  be the eigendecomposition of the Laplacian, where  $U\in \mathbb{R}^{n\times n}$  is the orthogonal matrix consists of the orthonormal eigenvectors of  $\mathcal{L}$  and  $\Lambda$  is the diagonal matrix of eigenvalues. Then, we can regard each eigenvector  $u_{k}$  as a "oscillation pattern" and its eigenvalue  $\lambda_{k}$  as the "frequency" of the oscillation. This intuition is supported by the Rayleigh quotient as follows.

$$
r (\mathcal {L}, x) \triangleq \frac {x ^ {\top} \mathcal {L} x}{x ^ {\top} x} = \frac {\sum_ {u \sim v} \mathcal {L} _ {u , v} (x (u) - x (v)) ^ {2}}{\sum_ {u \in V} x (u) ^ {2}}. \tag {1}
$$

where  $\sum_{u\sim v}$  sums over all unordered pairs for which  $u$  and  $v$  are adjacent,  $x(u)$  denotes the entry of vector  $x$  corresponding to vertex  $u$ , and  $\mathcal{L}_{u,v}$  is the  $(u,v)$ -entry of  $\mathcal{L}$ . From the definition we see that  $r(x)$  is non-negative and  $\mathcal{L}$  is positive semi-definite.  $r(x)$  is also known as a variational characterization of eigenvalues of  $\mathcal{L}$  (Horn & Johnson, 2012, Chapter 4), hence  $0\leq r(x)\leq 2$  for any non-zero real vector  $x$ . We use the notation  $r(x)$  to denote the Rayleigh quotient when the normalized graph Laplacian is clear from context. The Rayleigh quotient  $r(x)$  measures how the data  $x$  is oscillating. Hence, in this study, we use the term "frequency" and the "Rayleigh quotient" interchangeably. By the definition, the eigenvector  $u_{i}$  has the frequency of  $\lambda_{i}$ .

The labeling  $y$  of the vertices is low-frequency if the adjacent vertices are more likely to have the same label. This is a common assumption made by the spectral clustering algorithms (Shi & Malik, 2000; Ng et al., 2002; Shaham et al., 2018). Commonly used terms, homophily and heterophily, used in network science, correspond to low-frequency and high-frequency, respectively.

# 2.2 GRAPH FILTERING

In classical signal processing, a given signal is processed by filters in order to remove unwanted interference. Here, we first design a frequency response  $f(\lambda)$  of the filter, and then apply the filter to the signal in the sense that each frequency component  $\hat{x}(\lambda)$  of the data is modulated as  $f(\lambda)\hat{x}(\lambda)$ . Graph signal processing extends this concept as follows. Same as in classical signal processing, we design a filter  $f(\lambda)$ . Then, we represent a given graph signal  $x \in \mathbb{R}^{|V|}$  as a linear combination of the eigenvectors as  $x = \sum_{i}x_{i}u_{i}$ . Then, we modulate each frequency component by  $f(\lambda)$  as  $x = \sum_{i}f(\lambda_{i})x_{i}u_{i}$ . An important fact is that this can be done without performing the eigendecomposition explicitly. Let  $f(\mathcal{L})$  be the matrix function induced from  $f(\lambda)$ . Then, the filter is represented by  $f(\mathcal{L})x$ .

As an extension of signal processing, graph signal processing deals with signals defined on graphs. In definition 1, each column of the feature matrix  $X \in \mathbb{R}^{N \times d}$  is a "graph signal". Let  $\mathcal{L} = U \Lambda U^\top$

be the eigendecomposition where  $U \in \mathbb{R}^{N \times N}$  consists of orthonormal eigenvectors. Signal  $X$  is filtered by function  $f$  of the eigenvalues as follows.

$$
\bar {X} = U f (\Lambda) U ^ {\top} X = f (\mathcal {L}) X \tag {2}
$$

In general, different implementations of  $f(\mathcal{L})$  lead to different graph convolution models. For instance, GCN and SGC (Wu et al., 2019) are implemented by  $f(\mathcal{L}) = (I - \mathcal{L} + \mathrm{const.})^k$ , where the constant term stems from the fact that self-loops are added to vertices and  $k$  is the filter order. Generally, the underlying principle is to learn or construct the appropriate filter function  $f$  such that it transforms  $X$  into a more expressive representation. Note that the filter in GCN is called a low-pass filter because it amplifies low-frequency components (Li et al., 2018; NT & Maehara, 2019).

# 3 SPECTRAL PROPERTIES OF FILTERS

Towards building a ubiquitous solution, we take an intermediate step to study the vertex classification problem. Similar to the unsupervised clustering problem, an (implicit) low-frequency assumption is commonly made. However, the semi-supervised vertex classification problem is more involved because vertex labels can have complicated non-local patterns. Table 1 shows three groups of datasets, each with different label frequency ranges. Notably, WebKB datasets (Wisconsin, Cornell, Texas) have mixed label frequencies; some labels have low frequencies while others have midrange frequencies. Therefore, in order to relax the frequency assumptions, we need to learn the filtering function  $f(\lambda)$ .

The filtering function  $f(\lambda)$  is often approximated using a polynomial of the graph Laplacian as

$$
f (\mathcal {L}) \approx \operatorname {p o l y} (\mathcal {L}) = \sum_ {i = 0} ^ {K} \theta_ {i} \mathcal {L} ^ {i}. \tag {3}
$$

Because polynomials can uniformly approximate any real continuous function on a compact interval (see, e.g., (Brosowski & Deutsch, 1981)), such approximation scheme is well-justified.

Kipf & Welling (2017) derived their GCN formulation as follows. In their equation 5, they approximated a graph filter  $g_{\theta}$  by Chebyshev polynomials  $T_{k}$  as

$$
g _ {\theta} * x \approx \sum_ {k = 0} ^ {K} \theta_ {k} T _ {k} \left(D ^ {- 1 / 2} A D ^ {- 1 / 2}\right) x. \tag {4}
$$

Then, they took the first two terms and shared the parameters as  $\theta_0 = -\theta_1$  to obtain their equation 7:

$$
g _ {\theta} * x \approx \theta \left(I _ {N} + D ^ {- 1 / 2} A D ^ {- 1 / 2}\right) x \approx \theta (2 I _ {N} - \mathcal {L}) \tag {5}
$$

Finally, they extended a scalar  $\theta$  to a matrix  $\Theta$  to accommodate multiple feature dimensions as

$$
Z = \tilde {D} ^ {- 1 / 2} \tilde {A} \tilde {D} ^ {- 1 / 2} X \Theta \tag {6}
$$

Kipf & Welling (2017) claimed that the weight matrix  $\Theta$  can learn different filters, and subsequent works (e.g., (Veličković et al., 2018; Spinelli et al., 2020; Chen et al., 2020b)) also learned filters by  $\Theta$ . However, neither in theory nor practice it is the case (Oono & Suzuki, 2020). As the construction suggests, a GCN layer only represents a filter of the form  $f(\lambda) \approx 2 - \lambda$ . To properly learn different graph filters, we should learn the multiplying parameters  $\theta_0, \theta_1, \ldots, \theta_K$  in equation 3. In the next section, we propose a learning model which directly learns these multiplying parameters.

# 4 MODEL DESCRIPTION

The previous discussion provided several insights: (1) Vertex classification model's frequency is decided by its filter, (2) a mechanism to match the frequencies of data is necessary, and (3) directly learning the polynomial filter's coefficients is more desirable if we do not want to make any frequency assumption. Based on these observations, we implemented an adaptive Stacked Graph Filter (SGF) model. Figure 1 visually describes SGF.

![](images/f794ebf18494d5b9955fa5efab506689fe04d54b9bd8a1b9534d9377c6ae1455.jpg)  
Figure 1: Block description of SGF.  $\tilde{A} / \mathcal{L}$  means we can plug either the augmented normalized adjacency matrix or the symmetric normalized Laplacian into this model. In each filter layer, the scalar  $\alpha_{\ell}$  controls the filter's tangent and the scalar  $\beta_{\ell}$  controls the filter's vertical translation.

Design decisions. The novelty of our model is the stacked filter, and we directly learn the filtering function by filter coefficients  $\alpha$  and  $\beta$ , which makes SGF work well universally without frequency hyper-parameters. The deep filter module consists of filters stacked on top of each other with skip-connections to implement the ideas in Proposition 2. Each filter layer has two learnable scalars:  $\alpha_{\ell}$  and  $\beta_{\ell}$  which control the shape of the linear filter (Figure 1). Two learnable linear layers  $W_{\mathrm{in}}$  and  $W_{\mathrm{out}}$  with a non-linear activation serve as a non-linear classifier (NT & Maehara, 2019).

The input part of our architecture resembles APPNP (Klicpera et al., 2019) in the sense that the input signals (vertex features) are passed through a learning weight, then fed into filtering. The output part of our architecture resembles SGC (Wu et al., 2019) where we learn the vertex labels with filtered signals. This combination naturally takes advantages of both bottom-up (APPNP) and top-down (SGC) approaches. Compared to APPNP and SGC, besides the novel learning of filter functions, our model performs filtering (propagation) on the latent representation and classifies the filtered representation, whereas APPNP propagates the predicted features and SGC classifies the filtered features.

Given an instance of Problem 1, let  $\sigma$  be an activation function (e.g., ReLU),  $\tilde{A} = I - \mathcal{L} + c$  be the augmented adjacency matrix,  $\alpha_{\ell}$  and  $\beta_{\ell}$  be the filter parameters at layer  $\ell$ , a  $K$ -layer SGF is given by:

SGF: Input  $\tilde{A}$

$$
H _ {0} = \sigma (X W _ {\mathrm {i n}})
$$

$$
H _ {\ell} = \alpha_ {\ell} \tilde {A} H _ {\ell - 1} + \beta_ {\ell} H _ {0}, \ell = 1 \dots K
$$

$$
\hat {y} = H _ {K} W _ {\mathrm {o u t}}
$$

SGF: Input  $\mathcal{L}$

$$
H _ {0} = \sigma (X W _ {\mathrm {i n}})
$$

$$
H _ {\ell} = \alpha_ {\ell} \mathcal {L} H _ {\ell - 1} + \beta_ {\ell} H _ {0}, \ell = 1 \dots K
$$

$$
\hat {y} = H _ {K} W _ {\mathrm {o u t}}
$$

SGF can be trained with conventional objectives (e.g., negative log-likelihood) to obtain a solution to Problem 1. We present our models using the augmented adjacency matrix to show its similarity to existing literature. However, as noted in Figure 1, we can replace  $\tilde{A}$  with  $\mathcal{L}$ .

The stacked filter is easy to implement. Moreover, it can learn any polynomial of order-  $K$  as follows. The closed-form of the stacked filter (Figure 1) is given by

$$
\beta_ {K} I + \sum_ {i = 1} ^ {K} \left(\prod_ {j = i} ^ {K} \alpha_ {j}\right) \beta_ {i - 1} \mathcal {L} ^ {K - i + 1} \tag {7}
$$

where  $\beta_0 = 1$ . Because each term of equation 7 contains a unique parameter, we obtain the following.

Proposition 2. Any polynomial poly  $(\mathcal{L})$  of order  $K$  can be represented by the form equation 7.

Note that the same result holds if we replace  $\mathcal{L}$  in equation 7 by  $\tilde{A}$ . In practice, we typically set the initial values of  $\alpha_{i} = 0.5$  and update them via the back-propagation. The learned  $\alpha_{i}$  is then likely to satisfy  $|\alpha_{i}| < 1$ , which yields a further property of the stacked filter: it prefers a low-degree filter, because the coefficients of the higher-order terms are higher-order in  $\alpha_{i}$  which vanishes exponentially

faster. This advantage is relevant when we compare with a trivial implementation of the polynomial filter that learns  $\theta_{i}$  directly. (This approach corresponds to horizontal stacking.) In Appendix A.1, we compare these two implementations and confirm that the stacked filter is more robust in terms of filter degree than the trivial implementation.

# 5 RELATED WORK

GCN-like models cover a subset of an increasingly large literature on graph-structured data learning with graph neural networks (Gori et al., 2005; Scarselli et al., 2008). In general, vertex classification and graph classification are the two main benchmark problems. The principles for representation learning behind modern graph learning models can also be split into two views: graph propagation/diffusion and graph signal filtering. In this section, we briefly summarize recent advances in the vertex classification problem with a focus on propagation and filtering methods. For a more comprehensive view, readers can refer to review articles by Wu et al. (2020), Grohe (2020), and also recent workshops on graph representation learning<sup>2</sup>.

Feature Propagation. Feature propagation message-passing and graph signal filtering are two equivalent views on graph representation learning (Defferrard et al., 2016; Kipf & Welling, 2017). From the viewpoint of feature propagation (Scarselli et al., 2008; Gilmer et al., 2017), researchers focus on novel ways to propagate and aggregate vertex features to their neighbors. Klicpera et al. (2019) proposed PPNP and APPNP models, which propagate the hidden representation of vertices. More importantly, they pioneered in the decoupling of the graph part (propagation) and the classifier part (prediction). Abu-El-Haija et al. (2019) also proposed to use skip-connections to distinguish between 1-hop and 2-hop neighbors. Zeng et al. (2020) later proposed GraphSAINT to aggregate features from random subgraphs to further improve their model's expressivity. Pei et al. (2020) proposed a more involved geometric aggregation scheme named Geom-GCN to address weaknesses of GCN-like models. Most notably, they discussed the relation between network homophily and GCN's performance, which is similar to label frequency  $r(Y)$  in Table 1. Spinelli et al. (2020) introduced an adaptive model named AP-GCN, in which each vertex can learn the number of "hops" to propagate its feature via a trainable halting probability. Similar to our discussion in Section 3, they still use a fully-connected layer to implement the halting criteria, which controls feature propagation. AP-GCN's architecture resembles horizontal stacking of graph filters where they learn coefficients  $\theta$  directly. However their construction only allows for binary coefficients<sup>3</sup>. We later show that full horizontal stacking models (more expressive than AP-GCN) is less stable in terms of polynomial order than our approach (Appendix A.1). More recently, Liu et al. (2020) continued to address the difficulty of low homophily datasets and proposed a non-local aggregation based on 1D convolution and the attention mechanism, which has a "reconnecting" effect to increase homophily.

Graph Filtering. GCN-like models can also be viewed as graph signal filters where vertex feature vectors are signals and graph structure defines graph Fourier bases (Shuman et al., 2012; Defferrard et al., 2016; Li et al., 2018; Wu et al., 2019). This graph signal processing view addresses label efficiency (Li et al., 2019) and provides an analogue for understanding graph signal processing using traditional signal processing techniques. For example, the Lanczos algorithm is applied in learning graph filters by Liao et al. (2019). Bianchi et al. (2019) applies the ARMA filter to graph neural networks. Similar to (Klicpera et al., 2019), Wu et al. (2019) and NT & Maehara (2019) also follow the decoupling principle but in a reversed way (filter-then-classify). (Chen et al., 2020b) built a deep GCN named GCNII which holds the current best results for original splits of Cora, CiteSeer, and Pubmed. They further showed that their model can estimate any filter function with an assumption that the fully-connected layers can learn filter coefficients (Chen et al., 2020b, Proof of Theorem 2).

# 6 EXPERIMENTAL RESULTS

We conduct experiments on benchmark and synthetic data to empirically evaluate our proposed models. First, we compare our models with several existing models in terms of average classification accuracy. Our experimental results show that our single model can perform well across all frequency

ranges. Second, we plot the learned filter functions of our model to show that our model can learn the frequency range from the data — such visualization is difficult in existing works as the models' filters are fixed before the training process.

# 6.1 DATASETS

We use three groups of datasets corresponding to three types of label frequency (low, midrange, high). The first group is low-frequency labeled data, which consists of citation networks: Cora, Citeseer, Pubmed (Sen et al., 2008); and co-purchase networks Amazon-Photo, Amazon-Computer (Shchur et al., 2018). The second group is network datasets with midrange label frequency (close to 1): Wisconsin, Cornell, Texas (Pei et al., 2020); and Chameleon (Rozemberczki et al., 2019). The last group consists of a synthetic dataset with high label frequency (close to 2). For the Biparite dataset, we generate a connected bipartite graph on 2,000 vertices (1,000 on each part) with an edge density of 0.025. We then use the bipartite parts as binary vertex labels. Table 1 gives an overview of these datasets; see Appendix B.3 for more detail.

Table 1: Overview of graph datasets, divided to three frequency groups  

<table><tr><td>DATASETS</td><td>|V|</td><td>|E|</td><td>d</td><td>|C|</td><td>r(Y)</td><td>r(X)</td><td>Type</td></tr><tr><td>Cora</td><td>2,708</td><td>5,278</td><td>1,433</td><td>7</td><td>0.23 ± 0.04</td><td>0.91 ± 0.10</td><td>Citation</td></tr><tr><td>Citeseer</td><td>3,327</td><td>4,676</td><td>3,703</td><td>6</td><td>0.27 ± 0.03</td><td>0.81 ± 0.19</td><td>Citation</td></tr><tr><td>Pubmed</td><td>19,717</td><td>44,327</td><td>500</td><td>3</td><td>0.55 ± 0.02</td><td>0.87 ± 0.07</td><td>Citation</td></tr><tr><td>Amz-Photo</td><td>7,487</td><td>119,043</td><td>745</td><td>8</td><td>0.25 ± 0.04</td><td>0.82 ± 0.04</td><td>Co-purchase</td></tr><tr><td>Amz-Computer</td><td>13,381</td><td>245,778</td><td>767</td><td>10</td><td>0.27 ± 0.05</td><td>0.83 ± 0.04</td><td>Co-purchase</td></tr><tr><td>Wisconsin</td><td>251</td><td>450</td><td>1703</td><td>5</td><td>0.87 ± 0.08</td><td>0.89 ± 0.23</td><td>Web</td></tr><tr><td>Cornell</td><td>183</td><td>277</td><td>1703</td><td>5</td><td>0.86 ± 0.11</td><td>0.86 ± 0.32</td><td>Web</td></tr><tr><td>Texas</td><td>183</td><td>279</td><td>1703</td><td>5</td><td>0.98 ± 0.03</td><td>0.84 ± 0.32</td><td>Web</td></tr><tr><td>Chameleon</td><td>2,277</td><td>31,371</td><td>2325</td><td>5</td><td>0.81 ± 0.05</td><td>0.99 ± 0.01</td><td>Wikipedia</td></tr><tr><td>Bipartite</td><td>2,000</td><td>50,182</td><td>50</td><td>2</td><td>2.0 ± 0.00</td><td>1.0 ± 0.00</td><td>Synthetic</td></tr></table>

# 6.2 VERTEX CLASSIFICATION

We compare our method with some of the best models in the current literature. Two layers MLP (our model without graph filters), GCN (Kipf & Welling, 2017), SGC (Wu et al., 2019), and APPNP (Klicpera et al., 2019) are used as a baseline. Geom-GCN-(I,P,S) (Pei et al., 2020), JKNet+DE (Xu et al., 2018; Rong et al., 2019), and GCNII (Chen et al., 2020a) are currently among the best models. The Literature section of Table 2 and 3 shows the best results found in the literature where these models are set at the recommended hyper-parameters and recommended variants for each dataset. In our experiment, we fix the graph-related hyper-parameters of each model and report the classification results. Our model contains 16 layers of stacked filters  $(\hat{A})$  and has 64 hidden dimensions. Learning rate is set at 0.01, weight decay is  $5e\times 10^{-4}$ , and dropout rate for linear layers is 0.7. From an intuition that the filter should discover the required frequency pattern before the linear layers, we set the learning rate of linear layers to be one-fourth of the main learning rate. This experimental setup shows that SGF can adapt to the label frequency without setting specific hyper-parameters. In Table 2, SGF performs comparably with the current state-of-the-art. On the other hand, in Table 3, SGF is not only better than others in our experiments but also surpassing the best results in the literature. Note that we also the exact same SGF model across all experiments.

Results in Table 3 also suggest that the ability to adapt of the state of the art model GCNII is sensitive to its parameters  $\alpha$  and  $\theta$ . In our experiment, we fix the  $\theta$  parameter to 0.5 for all datasets, while in their manuscript the recommended values are around 1.5 depending on the dataset. With the recommended hyper-parameters, GCNII can achieve the average accuracy of  $81.57\%$  on Wisconsin data. However, its performance dropped around  $3\sim 10\%$  with different  $\theta$  values. This comparison highlights our model's ability to adapt to a wider range of datasets without any graph-related hyper-parameters.

Table 2: Vertex classification accuracy for low-frequency datasets  

<table><tr><td rowspan="2">METHODS</td><td colspan="5">DATASETS</td></tr><tr><td>Cora</td><td>Citeseer</td><td>Pubmed</td><td>Photo</td><td>Computer</td></tr><tr><td colspan="6">Our experiments (Average over 10 runs of stratified 0.6/0.2/0.2 splits)</td></tr><tr><td>MLP</td><td>75.01 ± 1.33</td><td>73.24 ± 1.28</td><td>83.56 ± 0.44</td><td>85.05 ± 1.62</td><td>80.42 ± 0.73</td></tr><tr><td>SGC (k=2)</td><td>87.15 ± 1.57</td><td>75.00 ± 0.93</td><td>87.97 ± 0.35</td><td>93.67 ± 0.68</td><td>90.87 ± 0.43</td></tr><tr><td>APPNP (α=0.2)</td><td>88.07 ± 1.32</td><td>76.71 ± 0.88</td><td>88.21 ± 0.37</td><td>94.70 ± 0.50</td><td>91.16 ± 0.44</td></tr><tr><td>GCNII (0.5,0.5)</td><td>86.21 ± 1.40</td><td>76.86 ± 1.29</td><td>89.77 ± 0.52</td><td>92.57 ± 0.61</td><td>88.71 ± 0.55</td></tr><tr><td>SGF</td><td>88.97 ± 1.21</td><td>77.58 ± 1.11</td><td>90.12 ± 0.40</td><td>95.58 ± 0.55</td><td>92.15 ± 0.41</td></tr><tr><td colspan="6">Literature (Best result among their variants)</td></tr><tr><td>GCN</td><td>85.77</td><td>73.68</td><td>88.13</td><td>(not avail.)</td><td>(not avail.)</td></tr><tr><td>GAT</td><td>86.37</td><td>74.32</td><td>87.62</td><td>(not avail.)</td><td>(not avail.)</td></tr><tr><td>Geom-GCN</td><td>85.27</td><td>77.99</td><td>90.05</td><td>(not avail.)</td><td>(not avail.)</td></tr><tr><td>APPNP</td><td>87.87</td><td>76.53</td><td>89.40</td><td>(not avail.)</td><td>(not avail.)</td></tr><tr><td>JKNet+DE</td><td>87.46</td><td>75.96</td><td>89.45</td><td>(not avail.)</td><td>(not avail.)</td></tr><tr><td>GCNII</td><td>88.49</td><td>77.13</td><td>90.30</td><td>(not avail.)</td><td>(not avail.)</td></tr></table>

Table 3: Vertex classification accuracy for midrange and high frequency datasets  

<table><tr><td rowspan="2">METHODS</td><td colspan="5">DATASETS</td></tr><tr><td>Wisconsin</td><td>Cornell</td><td>Texas</td><td>Chameleon</td><td>Bipartite</td></tr><tr><td colspan="6">Our experiments (Average over 10 runs of stratified 0.6/0.2/0.2 splits)</td></tr><tr><td>MLP</td><td>83.72 ± 3.40</td><td>80.13 ± 4.59</td><td>80.30 ± 5.55</td><td>45.63 ± 1.88</td><td>48.34 ± 1.67</td></tr><tr><td>SGC (k=2)</td><td>56.27 ± 6.79</td><td>53.37 ± 5.41</td><td>51.49 ± 6.75</td><td>66.51 ± 2.44</td><td>48.07 ± 1.47</td></tr><tr><td>APPNP (α=0.2)</td><td>71.02 ± 5.98</td><td>74.55 ± 4.49</td><td>66.95 ± 6.02</td><td>54.58 ± 1.67</td><td>50.89 ± 1.08</td></tr><tr><td>GCNII (0.5,0.5)</td><td>71.57 ± 5.13</td><td>74.47 ± 5.42</td><td>73.78 ± 6.72</td><td>55.81 ± 1.55</td><td>49.70 ± 1.75</td></tr><tr><td>SGF</td><td>87.06 ± 4.66</td><td>82.45 ± 6.19</td><td>80.56 ± 5.63</td><td>58.77 ± 1.90</td><td>100.0 ± 0.00</td></tr><tr><td colspan="6">Literature (Best results among their variants)</td></tr><tr><td>GCN</td><td>45.88</td><td>52.70</td><td>52.16</td><td>28.18</td><td>(not avail.)</td></tr><tr><td>GAT</td><td>49.41</td><td>54.32</td><td>58.38</td><td>42.93</td><td>(not avail.)</td></tr><tr><td>Geom-GCN</td><td>64.12</td><td>60.81</td><td>60.90</td><td>(not avail.)</td><td>(not avail.)</td></tr><tr><td>APPNP</td><td>69.02</td><td>73.51</td><td>65.41</td><td>54.30</td><td>(not avail.)</td></tr><tr><td>JKNet+DE</td><td>50.59</td><td>61.08</td><td>57.30</td><td>62.08</td><td>(not avail.)</td></tr><tr><td>GCNII</td><td>81.57</td><td>76.49</td><td>77.84</td><td>62.48</td><td>(not avail.)</td></tr></table>

# 6.3 FILTER VISUALIZATION

Another advantage of our model is the ability to visualize the filter function using an inversion of Proposition 2. The first row of Figure 2 shows the filtering functions at initialization and after training when input is the normalized augmented adjacency matrix. The second row shows the results when the input is the normalized Laplacian matrix. These two cases can be interpreted as starting with a low-pass filter  $(\tilde{A})$  or starting with a high-pass filter  $(\mathcal{L})$ . Figure 2 clearly shows that our method can learn the suitable filtering shapes from data regardless of the initialization. We expect the visualization here can be used as an effective exploratory tool and baseline method for future graph data.

# 6.4 ADAPTIVITY TO STRUCTURAL NOISE

Recently, Fox & Rajamanickam (2019) raised a problem regarding structural robustness of a graph neural network for graph classification. Zügner et al. (2018) posed a similar problem related to adversarial attack on graphs by perturbations of vertex feature or graph structure for the vertex classification setting (Dai et al., 2018; Bojchevski & Gunnemann, 2019; Zügner & Gunnemann, 2019). Here, we evaluate the robustness of the models against the structural noise, where we perturb

![](images/8855b6ca4f849c30483166ebd76b5e3d08b3c8e65391515f19076fa8059c0d93.jpg)

![](images/b1cffd7710cd6551c20691c64fe984c30d8344185ea9ec003a6944fac5816b58.jpg)

![](images/a941215c0ed85a550e24efa13f3268404380915cc44f940c323b25e9c8de3ed8.jpg)

![](images/454652b9bff70c50eeffa4313e18d76ba681186ee24e3c66b542f0c5dc9abd9e.jpg)  
Figure 2: Learned filtering functions  $f(\lambda)$  on three datasets corresponding to three frequency ranges. Each row shows the learning results for each initialization. Lightened lines represent the learned filtering functions of 10 different runs. The average accuracy is shown on the top right corner.

![](images/7c3b03cced917ff32980fe8f377b59dcdf3c416ca546742a57318523915406c1.jpg)  
Figure 3: Vertex classification accuracy for each amount of edge perturbation. Since GCNII has similar performance as our model in this setting, we only plot the results for SGF.

![](images/487dacc6f9c5458acf5e8187b9314b5c28b9bb3ec3e7ea449c1eac522d4f57c6.jpg)

![](images/35c17f9477978186d280dfa6750c3acaefd2c9222b3fea40a6e097e865181e8d.jpg)

a fraction of edges while preserving the degree sequence<sup>4</sup>. This structural noise collapses the relation between the features and the graph structure; hence, it makes the dataset to have the midrange frequency. This experimental setting shows that adaptive models like ours and GCNII are more robust to structural noise. In the worst-case scenario (90% edges are swapped), the adaptive models are at least as good as an MLP on vertex features. Figure 3 shows vertex classification results at each amount of edge perturbation: from 10% to 90%. APPNP with  $\alpha = 0.2$  and SGC with  $k = 2$  have similar behavior under structural noise since these models give more weights to filtered features. On the other hand, APPNP with  $\alpha = 0.8$  is much more robust to structural noise as it depends more on the vertex features. This result suggests that adaptive models like ours and GCNII can be a good baseline for future graph adversarial attack studies (SGF's advantage here is being much simpler).

Additional Experiments. We provide several other experimental results in Appendix A. Section A.1 discusses the advantages of vertical stacking (SGF) versus a naive horizontal stacking (learning  $\theta$  in equation 3 directly). Section A.2 discusses the difficulty of estimating the frequency range (Rayleigh quotient) of vertex labels when the training set is small.

# 7 CONCLUSION

We show that simply by learning the polynomial coefficients rather than the linear layers in the formulation of GCN can lead to a highly adaptive vertex classification model. Our experiment shows that by using only one setting, SGF is comparable with all current state-of-the-art methods. Furthermore, SGF can also adapt to structural noise extremely well, promising a robust model in practice. Since our objective is to relax the frequency assumption, one could expect our model will perform weakly when number of training data is limited. Because the estimation of label frequency becomes difficult with a small number of data (Appendix A.2), designing a learning model that is both adaptive and data-efficient is an exciting challenge. We believe an unbiased estimation (Proposition 4) with a more involved filter learning scheme is needed to address this problem in the future.

# REFERENCES

Sami Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Kristina Lerman, Hrayr Harutyunyan, Greg Ver Steeg, and Aram Galstyan. Mixhop: Higher-order graph convolutional architectures via sparsified neighborhood mixing. In International Conference on Machine Learning, pp. 21-29, 2019.  
Filippo Maria Bianchi, Daniele Grattarola, Cesare Alippi, and Lorenzo Livi. Graph neural networks with convolutionalarma filters.arXiv preprint arXiv:1901.01343,2019.  
Aleksandar Bojchevski and Stephan Gunnemann. Adversarial attacks on node embeddings via graph poisoning. In International Conference on Machine Learning, pp. 695-704. PMLR, 2019.  
Bruno Brosowski and Frank Deutsch. An elementary proof of the stone-weierstrass theorem. Proceedings of the American Mathematical Society, pp. 89-92, 1981.  
Jie Chen, Tengfei Ma, and Cao Xiao. Fastgcn: Fast learning with graph convolutional networks via importance sampling. 2018.  
Lei Chen, Le Wu, Richang Hong, Kun Zhang, and Meng Wang. Revisiting graph based collaborative filtering: A linear residual graph convolutional network approach. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 27-34, 2020a.  
Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and deep graph convolutional networks. In Proceedings of the 37th International Conference on Machine Learning, 2020b.  
Fan RK Chung and Fan Chung Graham. Spectral graph theory. Number 92 in CBMS Workshop on Spectral Graph Theory. American Mathematical Society, 1997.  
Hanjun Dai, Hui Li, Tian Tian, Xin Huang, Lin Wang, Jun Zhu, and Le Song. Adversarial attack on graph structured data. arXiv preprint arXiv:1806.02371, 2018.  
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in neural information processing systems, pp. 3844-3852, 2016.  
James Fox and Sivasankaran Rajamanickam. How robust are graph neural networks to structural noise? arXiv preprint arXiv:1912.10206, 2019.  
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. arXiv preprint arXiv:1704.01212, 2017.  
Marco Gori, Gabriele Monfardini, and Franco Scarselli. A new model for learning in graph domains. In Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2005., volume 2, pp. 729-734. IEEE, 2005.  
Martin Grohe. word2vec, node2vec, graph2vec, x2vec: Towards a theory of vector embeddings of structured data. In Proceedings of the 39th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems, pp. 1-16, 2020.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, pp. 1024-1034, 2017.  
Roger A Horn and Charles R Johnson. Matrix analysis. Cambridge university press, 2012.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. 2017.  
Johannes Klicpera, Aleksandar Bojchevski, and Stephan Gunnemann. Predict then propagate: Graph neural networks meet personalized pagerank. In International Conference on Learning Representations, 2019.  
Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-supervised learning. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

Qimai Li, Xiao-Ming Wu, Han Liu, Xiaotong Zhang, and Zhichao Guan. Label efficient semi-supervised learning via graph filtering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9582-9591, 2019.  
Renjie Liao, Zhizhen Zhao, Raquel Urtasun, and Richard S Zemel. Lanczosnet: Multi-scale deep graph convolutional networks. 2019.  
Meng Liu, Zhengyang Wang, and Shuiwang Ji. Non-local graph neural networks. arXiv preprint arXiv:2005.14612, 2020.  
Andrew Y Ng, Michael I Jordan, and Yair Weiss. On spectral clustering: Analysis and an algorithm. In Advances in neural information processing systems, pp. 849-856, 2002.  
Hoang NT and Takanori Maehara. Revisiting graph neural networks: All we have is low-pass filters. arXiv preprint arXiv:1905.09550, 2019.  
Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node classification. International Conference on Representation Learning, 2020.  
Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, and Bo Yang. Geom-gcn: Geometric graph convolutional networks. arXiv preprint arXiv:2002.05287, 2020.  
Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. Droppede: Towards deep graph convolutional networks on node classification. 2019.  
Benedek Rozemberczki, Carl Allen, and Rik Sarkar. Multi-scale attributed node embedding. arXiv preprint arXiv:1909.13021, 2019.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61-80, 2008.  
Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Collective classification in network data. AI magazine, 29(3):93-93, 2008.  
Uri Shaham, Kelly Stanton, Henry Li, Boaz Nadler, Ronen Basri, and Yuval Kluger. Spectralnet: Spectral clustering using deep neural networks. International Conference on Learning Representations, 2018.  
Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan Gunnemann. Pitfalls of graph neural network evaluation. arXiv preprint arXiv:1811.05868, 2018.  
Jianbo Shi and Jitendra Malik. Normalized cuts and image segmentation. IEEE Transactions on pattern analysis and machine intelligence, 22(8):888-905, 2000.  
David I. Shuman, Sunil K. Narang, Pascal Frossard, Antonio Ortega, and Pierre Vandergheynst. The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains. arXiv preprint arXiv:1211.0053, 2012.  
Indro Spinelli, Simone Scardapane, and Uncini Aurelio. Adaptive propagation graph convolutional network. arXiv preprint arXiv:2002.10306, 2020.  
Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. International Conference on Representation Learning, 2018.  
Jason Weston, Frédéric Ratle, Hossein Mobahi, and Ronan Collobert. Deep learning via semi-supervised embedding. In Neural networks: Tricks of the trade, pp. 639-655. Springer, 2012.  
Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, and Kilian Q. Weinberger. Simplifying graph convolutional networks. In Proceedings of the 36th International Conference on Machine Learning, volume 97. JMLR, 2019.  
Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu Philip. A comprehensive survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems, 2020.

Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. arXiv preprint arXiv:1806.03536, 2018.  
Zhilin Yang, William Cohen, and Ruslan Salakhudinov. Revisiting semi-supervised learning with graph embeddings. In Maria Florina Balcan and Kilian Q. Weinberger (eds.), Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pp. 40-48, New York, New York, USA, 20-22 Jun 2016. PMLR.  
Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor Prasanna. Graph-saint: Graph sampling based inductive learning method. In International Conference on Learning Representations, 2020.  
Daniel Zügner and Stephan Gunnemann. Adversarial attacks on graph neural networks via meta learning. arXiv preprint arXiv:1902.08412, 2019.  
Daniel Züigner, Amir Akbarnejad, and Stephan Gümnmann. Adversarial attacks on neural networks for graph data. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 2847-2856, 2018.
