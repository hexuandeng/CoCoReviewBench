# GRAPH CONVOLUTIONAL NETWORKS VIA ADAPTIVE FILTER BANKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Graph convolutional networks have been a powerful tool in representation learning of networked data. However, most architectures of message passing graph convolutional networks (MPGCNs) are limited as they employ a single message passing strategy and typically focus on low-frequency information, especially when graph features or signals are heterogeneous in different dimensions. Then, existing spectral graph convolutional operators lack a proper sharing scheme between filters, which may result in overfitting problems with numerous parameters. In this paper, we present a novel graph convolution operator, termed BankGCN, which extends the capabilities of MPGCNs beyond single 'low-pass' features and simplifies spectral methods with a carefully designed sharing scheme between filters. BankGCN decomposes multi-channel signals on arbitrary graphs into subspaces and shares adaptive filters to represent information in each subspace. The filters of all subspaces differ in frequency response and together form a filter bank. The filter bank and the signal decomposition permit to adaptively capture diverse spectral characteristics of graph data for target applications with a compact architecture. We finally show through extensive experiments that BankGCN achieves excellent performance on a collection of benchmark graph datasets.

# 1 INTRODUCTION

In many application domains, structured data is supported by graphs or networks. To deal with such data, a collection of graph convolutional networks (GCNs), including MPGCNs and spectral GCNs, have been proposed by generalizing architectures used for data in the Euclidean domain, to the irregular graphs. Unfortunately, both types of architectures have implicit limitations when it comes to efficiently representing graph data information in terms of frequency filtering.

MPGCNs (Gilmer et al., 2017; Velikovi et al., 2018; Hamilton et al., 2017; Xu et al., 2019) built on diverse message passing (MP) schemes are much prevalent with their flexible and intuitive formulations of graph convolution. The message passing schemes in MPGCNs, however, mostly focus on the low frequency characteristics of the data. For example, GCN (Kipf & Welling, 2017) performs as Laplacian smoothing (Li et al., 2018), and the mean aggregator adopted in GraphSage (Hamilton et al., 2017) has naturally low-pass properties. However, in addition to low-frequency components, graph data may also carry rich information in the middle- and high-frequency ranges. The importance of different frequency components may further vary with the target tasks, it is beneficial to be able to adapt the representation accordingly.

Spectral graph convolutional networks (Defferrard et al., 2016; Levie et al., 2018; Bianchi et al., 2021) are designed in the graph frequency domain directly and are much more powerful alternatives to process diverse spectral characteristics of graphs. The lack of proper sharing schemes between filters, however, renders these models redundant and even prone to overfitting.

We go beyond the above limitations and propose to build effective adaptive representations for graph data with diverse spectral properties. We design a novel graph convolution operator termed BankGCN that utilizes an adaptive filter bank to process graph data in the frequency domain, as presented in Fig. 1. BankGCN provides an effective implementation to simplify spectral methods with a proper sharing scheme between filters with the help of signal decomposition. Firstly, we decompose multi-channel graph signals into a collection of subspaces through projection, in order to adaptively separate input data according to signal characteristics. Components in a subspace will

then share a learnable filter that captures their particular frequency properties. Notably, representations are built on finite impulse response (FIR) filters with a universal design (Tremblay et al., 2018) that correspond to local message passing schemes in the spatial domain. Filters of all the subspaces together form a filter bank, and they are simultaneously learned from data together with subspace decomposition. The filter bank representation is regularized to favour diversity in the frequency responses, in order to properly capture various spectral components in the graph signals. The proposed convolution operator is stackable, and can be optimized together with other modules in the GCNs like graph pooling. We validate the proposed architecture through extensive experiments on various graph classification tasks. BankGCN is more powerful than most MPGCNs, such as GCN (Kipf & Welling, 2017), GraphSage (Hamilton et al., 2017), and GIN (Xu et al., 2019), in that it is able to exploit more diverse spectral characteristics than 'low-pass' features in the data and adapts to its heterogeneous properties with a group of learnable multi-hop message passing strategies. Furthermore, it outperforms its counterpart ChebNets (Defferrard et al., 2016) with a more compact architecture and achieves better generalization, and is superior to the most recent spectral method ARMA (Bianchi et al., 2021). It would be interesting to extend BankGCN to tasks like link prediction and applications into non-Euclidean data like 3-D point clouds.

# 2 RELATED WORK

We briefly overview the main architectures for graph representation learning. We describe first existing message passing schemes, and then discuss several spectral methods.

Message Passing Graph Convolutional Networks. MPGCNs design graph convolution with a variety of message passing schemes in the spatial domain. For instance, messages are aggregated with the node-wise mean or max operation in a localized neighborhood in GraphSage (Hamilton et al., 2017), or based on attention scores in GAT (Velikovi et al., 2018). A more expressive scheme, GIN, is further proposed with summing features in a neighborhood followed by a multi-layer perceptrons (MLP) to approximate any injective function on multiset (Xu et al., 2019). However, these methods are typically constrained to a single one-hop message passing strategy that mostly captures 'low-pass' characteristics in the graph data. Klicpera et al. (2019) expand the MP range to multi-hop neighborhoods with Personalized PageRanks, and GPR-GNN (Chien et al., 2021) further improves it in an adaptive way to alleviate the over-smoothing issue. Similarly, Scattering GCN (Min et al., 2020) resorts to geometric scattering transform (Gao et al., 2019) to complement the 'low-pass' features of GCN (Kipf & Welling, 2017) with 'band-pass' features; a one-order high-pass filter is adopted as the complement (Bo et al., 2021); MixHop (Abu-El-Haija et al., 2019) explores linear mixing of neighborhood information to further realize difference operators, which can capture high-frequency information. In contrary to these methods, we employ a learnable filter bank with various frequency responses to adaptively capture diverse spectral characteristics of signals rather than merely additional 'high-pass' features or specific components via predefined graph wavelets.

Filtering on Graphs and Spectral GCNs. Filtering and Graph Fourier Transform have been generalized to graph data via spectral graph theory (Shuman et al., 2013). Correspondingly, several spectral GCNs are derived from filtering in the graph frequency domain directly, as pioneered by Bruna et al. (2014). To avoid expensive eigendecomposition of the graph Laplacian, similarly to dictionary learning methods (Thanou et al., 2014) in graph signal processing (GSP), the filters are built as polynomial or rational functions of eigenvalues of the graph Laplacian, such as Chebyshev polynomials in ChebNets (Defferrard et al., 2016) and TIGraNet (Khasanova & Frossard, 2017), Cayley polynomials in CayleyNets (Levie et al., 2018), and auto-regressive moving average (ARMA) filters (Bianchi et al., 2021). In contrast with these spectral convolution methods that focus on the implementation of filters with desirable properties, like localization and narrowband specialization, this paper focuses on the design of the filter bank in order to construct a simplified architecture with improved generalization.

Finally, there are several papers in graph signal processing literature about the design of graph filter banks for graph signal decomposition (Narang & Ortega, 2012; Tanaka & Sakiyama, 2014) and multiscale analysis (Hammond et al., 2011; Narang & Ortega, 2013). However, they usually work under a rigid constraint, e.g., perfect reconstruction, and thereby their produced (sparse) representations may not be flexible enough to adapt to diverse tasks in machine learning. In contrast, our

![](images/4e9d2008f42d72fe0936cf35831f5d25c4d89c3ecd487b2efe9b7dd002b7b7f8.jpg)  
(a) Framework.

![](images/6eda36f315367f8eccce45f524a2c29bca2c72829872dfaa6ab6387e6f5ed448.jpg)  
(b) Filter bank.

![](images/98207f72bf7b7241ac262f38172c3ea3339ae2b3b4af60819d6ed204feb0e673.jpg)  
Figure 1: Illustration of the BankGCN operator. In each layer, the graph signal  $\pmb{x}(v_m)$  is decomposed into a collection of subspaces and processed by an adaptive filter bank to capture distinct frequency properties. A filter in the filter bank corresponds to an adapted  $K$ -hop neighborhood message passing scheme in the spatial domain, as illustrated in (b) and (c).  
(c) Message passing.

method relaxes the perfect reconstruction condition and rather regularizes filters in a filter bank to be sufficiently different in the graph frequency domain.

# 3 PRELIMINARIES AND PROBLEMS

In this paper, we consider data that is represented on undirected graphs. Each vertex of graphs is attributed a  $d$ -channel signal or feature. The node signal is typically multi-channel, i.e.,  $d > 1$ , and with different spectral statistics along different dimensions, which we refer to heterogeneous signals in this paper. Notations and preliminaries on Graph Fourier Transform and filtering are introduced in Appendix A.

We first show how filtering can be implemented by messaging passing, which is used in many state-of-the-art graph representation learning methods. Filtering in the graph frequency domain and MP in the spatial domain is closely related (Shuman et al., 2013). Taking the most popular GCN (Kipf & Welling, 2017) as an example, according to its initial formation, the  $j$ -th channel of a filtered signal is

$$
\boldsymbol {h} _ {j} = \sum_ {i = 1} ^ {d} \left(I + D ^ {- \frac {1}{2}} A D ^ {- \frac {1}{2}}\right) \Theta \boldsymbol {x} _ {i} = \sum_ {i = 1} ^ {d} (2 I - L) \Theta_ {i, j} \boldsymbol {x} _ {i} = \sum_ {i = 1} ^ {d} \Theta_ {i, j} U \hat {g} (\Lambda) U ^ {*} \boldsymbol {x} _ {i} \tag {1}
$$

where  $\Theta$  denotes learnable parameters,  $d$  indicates the dimension of signals, and  $\hat{g} (\Lambda) = 2I - \Lambda$  is a low-pass filter (NT & Maehara, 2019; Min et al., 2020). The renormalized version of GCN and the GIN can be formulated similarly. Please refer to (Balcilar et al., 2021) for their specific equivalent supports of  $\hat{g} (\Lambda)$ . Besides, GraphSage uses the mean or max operator to aggregate information in neighborhoods. Although it is hard to explicitly formulate its corresponding frequency response, the element-wise mean operation is a low-pass operation by nature and the element-wise max operator performs as an envelope extraction that suppresses the high-frequency components.

Another line of works, spectral GCNs, are directly derived from spectral filtering. For the representative ChebNets and CayleyNets, the filters are defined as  $\hat{g} (\lambda) = \sum_{k = 0}^{K}\theta^{(k)}T_{k}(\lambda)$  in the graph frequency domain with the Chebyshev polynomial or Cayley polynomial basis  $\{T_k(\cdot)\}$  and correspondingly

$$
\boldsymbol {h} _ {j} = \sum_ {i = 1} ^ {d} U \sum_ {k = 0} ^ {K} \Theta_ {i, j} ^ {(k)} T _ {k} (\Lambda) U ^ {*} \boldsymbol {x} _ {i} = \sum_ {i = 1} ^ {d} U \hat {g} _ {i j} (\Lambda) U ^ {*} \boldsymbol {x} _ {i}, \tag {2}
$$

where  $K$  is the order of the polynomial filters and  $\{\Theta^{(k)}\}$  are learnable parameters.

Notably, the above MPGCNs employ just a single kind of filter to handle all the channels of signals, and then takes different (linear) combinations of the filtered signals to obtain different channels of the output signals. Thereby, most MPGCNs are restricted in the number and types of filters and have a limited capacity in frequency-domain filtering. On the contrary, ChebNets and CayleyNets employ a different filter for each mapping from an input channel to an output channel and adopt

a total of  $d \times d'$ $K$ -order polynomial filters ( $d$  and  $d'$  denote the respective number of channels of input and output signals), as presented in Eq. 2. This naive design neglects the potential relationship between filters and fails to reduce redundancy. Consequently, they explore such a large number of learnable filters to process diverse frequency components of graph data at the expense of numerous free parameters, which sometimes leads to overfitting.

# 4 THE BANKGCN ALGORITHM

In this section, we first outline the framework of BankGCN, then present its main elements, and finally discuss connections with spectral methods.

# 4.1 GRAPH CONVOLUTION WITH ADAPTIVE FILTER BANKS

Signals (features) supported on nodes of a graph are usually high-dimensional and composed of multiple spectral patterns. In other words, different signal channels vary differently over nodes, leading to diverse spectral characteristics. To handle the heterogeneous signals, we employ a filter bank composed of a set of filters with different frequency responses in the design of BankGCN. In order to further facilitate information processing, signal decomposition is adopted to explore latent relationships in the data.

For a multi-channel input node signal  $\pmb{x}(v_m)$ , we first adaptively decompose it into different subspaces and then employ different filters to deal with the signal components in each subspace separately. The decomposition aims to map the components of signals with similar characteristics into the same subspace in order to facilitate the subsequent adaptive filtering. This is implemented with learnable subspace projections. Mathematically, the input signal  $\pmb{x}(v_m)$  is projected into  $s$  subspaces with a group of projection functions denoted by  $\{f_{[p]}(\cdot)\}$ ,

$$
\boldsymbol {r} _ {[ p ]} (v _ {m}) = f _ {[ p ]} (\boldsymbol {x} (v _ {m})), \quad p = 1, 2, \dots , s, \forall v _ {m} \in \mathcal {V}. \tag {3}
$$

Here, the subscript  $[p]$  indicates the terms belonging to the  $p$ -th subspace and  $\boldsymbol{r}_{[p]}(v_m)$  is the projected signal. The choices for projection functions will be introduced in Section 4.2. Subsequently, an adaptive filter  $g_{[p]}(\cdot)$  is designed within each subspace to represent the spectral characteristics of the corresponding signal components,

$$
\boldsymbol {h} _ {[ p ]} \left(v _ {m}\right) = \left(g _ {[ p ]} * \boldsymbol {r} _ {[ p ]}\right) \left(v _ {m}\right), \tag {4}
$$

where we reuse  $*$  to denote the convolution between each channel of signal  $r_{[p]}$  and the filter  $g_{[p]}$ .

The filtered signals  $h_{[p]}(v_m), p = 1,2,\dots ,s$ , of all the subspaces are concatenated to produce the output features. A Rectified Linear Unit (ReLU) is then used as a non-linear activation function,

$$
\boldsymbol {h} \left(v _ {l, m}\right) = \operatorname {C o n c a t} \left(\boldsymbol {h} _ {[ 1 ]} \left(v _ {l, m}\right), \boldsymbol {h} _ {[ 2 ]} \left(v _ {l, m}\right), \dots , \boldsymbol {h} _ {[ s ]} \left(v _ {l, m}\right)\right), \tag {5}
$$

$$
\boldsymbol {x} \left(v _ {l + 1, m}\right) = \operatorname {R e L U} \left(\boldsymbol {h} \left(v _ {l, m}\right), \right. \tag {6}
$$

where the subscript  $l$  indicates variables or parameters in the  $l$ -th layer to describe the forward propagation between different layers in a hierarchical architecture.

Specially, considering that some filters are only supported on medium-to-high frequency components that usually mean large signal variations in the spatial domain, we further introduce a shortcut in each subspace that corresponds to full-pass in the graph frequency domain in order to make the filtered signals stable. Correspondingly, the filtered signal  $h_{[p]}(v_m)$  in each subspace becomes

$$
\boldsymbol {h} _ {[ p ]} \left(v _ {m}\right) = \left(g _ {[ p ]} * \boldsymbol {r} _ {[ p ]}\right) \left(v _ {m}\right) + \boldsymbol {r} _ {[ p ]} \left(v _ {m}\right). \tag {7}
$$

The shortcut can further facilitate back propagation of gradients, as for CNNs (He et al., 2016).

The proposed graph convolution operator is stackable, and the subspace mapping function in the following layers will further combine features from different subspaces in the preceding layers to enable information interaction between different channels.

# 4.2 SUBSPACE PROJECTION

For the projection function  $f_{[p]}(\cdot)$ , there is a variety of design choices. Here, we take the linear mapping as an example, since it is simple yet able to decompose different channels of signals.

Specifically, the  $d$ -channel  ${}^{1}$  signal  $\pmb{x}(v_m)$  is projected into  $s$  different subspaces with learnable matrices  $\{W_{[p]}\}_{p=1}^s$ . For the sake of simplicity, all the subspaces have the same dimension in this paper. To increase flexibility, we further introduce a learnable bias  $\pmb{b}_{[p]}$  for each subspace.

$$
\boldsymbol {r} _ {[ p ]} (v _ {m}) = f _ {[ p ]} (x (v _ {m})) = W _ {[ p ]} ^ {T} \boldsymbol {x} (v _ {m}) + \boldsymbol {b} _ {[ p ]}, \quad p = 1, 2, \dots , s, \tag {8}
$$

The introduction of subspace projection brings in two advantages: (i) it simplifies the learning process. Since the multi-channel graph signals have been decomposed, the filter in each subspace just needs to learn to capture spectral characteristics of the corresponding signal components; (ii) with low-dimensional subspaces, it limits the dimension of features output by the filter bank, and thereby reduces the number of free parameters and computation in the following layers.

# 4.3 FILTERS

In order to handle graphs with arbitrary topologies and diverse signals, we adopt universal and adaptive filters to construct the filter bank.

For any function  $t:\mathcal{D} = [0,2]\to \mathbb{R}$ , we can obtain a corresponding filter whose frequency response is  $\hat{g}_{[p]}(\lambda) = t(\cdot)$ , and its spatial construction is computed through IGFT:

$$
g _ {[ p ]} (v _ {m}) = \sum_ {i = 1} ^ {n} \hat {g} _ {[ p ]} (\lambda_ {i}) u _ {i} (v _ {m}), \tag {9}
$$

with  $u_{i}, i = 1, \dots, n$ , the eigenvectors of the symmetric normalized graph Laplacian of a graph. Notably, we directly design the frequency response of the filter for the continuous range  $\mathcal{D} = [0,2]$  in which the spectrum of an arbitrary graph locates as introduced in Section A. In other words, given the discrete spectrum  $\{\lambda_0, \lambda_1, \ldots, \lambda_n\}$  of an arbitrary graph, we have  $\lambda_i \in \mathcal{D}$  and obtain the corresponding filter value on these specific discrete values  $\hat{g}_{[p]}(\lambda_i) = t(\lambda_i)$ , for  $i = 1, 2, \ldots, n$ . Thereby, the filter  $g_{[p]}(\cdot)$  is adaptable to any graph even with different topologies, i.e., it is a universal form (Tremblay et al., 2018; Levie et al., 2019).

Furthermore, we constrain the filter to the  $K$ -order polynomial function space in order to avoid the computation-intensive eigendecomposition of the graph Laplacian, similarly to parametric dictionary learning (Thanou et al., 2014) in GSP. It corresponds to an FIR filter. Mathematically, the frequency response of a filter can be represented as

$$
\hat {g} _ {[ p ]} (\lambda) = \sum_ {k = 0} ^ {K} \alpha_ {[ p ]} ^ {(k)} T _ {k} (\lambda), \tag {10}
$$

where  $\{T_k\}$  denotes a specific polynomial basis such as Chebyshev polynomials, and  $\{\alpha_{[p]}^{(k)}\}$  indicates the corresponding coefficients. With  $\{\alpha_{[p]}^{(k)}\}$  learnable, we obtain an adaptive filter whose frequency response adapts to the data and to the target task. Correspondingly, for the signal projected to the  $p$ -th subspace  $\mathbf{r}_{[p]}(v_m)$  and  $R_{[p]} = [\mathbf{r}_{[p]}(v_1), \mathbf{r}_{[p]}(v_2), \dots, \mathbf{r}_{[p]}(v_n)]^T$ , the filtered signal is calculated as

$$
\boldsymbol {h} _ {[ p ]} (v _ {m}) = \left(\boldsymbol {r} _ {[ p ]} * g _ {[ p ]}\right) (v _ {m}) = \sum_ {k = 0} ^ {K} \alpha_ {[ p ]} ^ {(k)} \left(T _ {k} (L) R _ {[ p ]}\right) _ {m} ^ {T}. \tag {11}
$$

Equivalently, according to the relationship between frequency filtering and localized linear transforms in the spatial domain (Shuman et al., 2013), the filtering strategy corresponds to a message passing scheme within a  $K$ -hop neighborhood

$$
\boldsymbol {h} _ {[ p ]} \left(v _ {m}\right) = c _ {m m} \boldsymbol {r} _ {[ p ]} \left(v _ {m}\right) + \sum_ {v _ {o} \in N ^ {K} \left(v _ {m}\right)} c _ {m o} \boldsymbol {r} _ {[ p ]} \left(v _ {o}\right), \tag {12}
$$

with

$$
c _ {m o} = \sum_ {k = 0} ^ {K} \alpha_ {[ p ]} ^ {(k)} \left(T _ {k} (L)\right) _ {m, o} \quad \forall m, o \in \{1, 2, \dots , n \}. \tag {13}
$$

Specially, the induced  $K$ -hop message passing scheme is learned from the data and exploits the multi-hop topological information of graphs through polynomials of the graph Laplacian. Signal information is also taken into consideration through the learnable parameters  $\{\alpha_{[p]}^{(k)}\}$  as well as in the subspace projection step. More importantly, it permits to represent features that do not only have "low-pass" properties and explores the frequency components in a data-driven manner.

# 4.4 DIVERSITY REGULARIZATION FOR FILTER BANKS

The filters constituting a filter bank should ideally have diverse frequency responses so that the signal is decomposed through filtering into a series of signals with different frequency characteristics. In GSP, the filters are usually band-pass and divide the spectrum in different bands. Considering that strict band-pass filters are difficult to fit through polynomial functions, we relax this strict bandpass requirement and rather target filters with diverse frequency responses, which we call "diversity condition". With the filter  $\hat{g}_{[p]}(\lambda)$  given as a  $K$ -order polynomial function, the regularization on the filter is imposed on the respective polynomial coefficients  $\{\alpha_{[p]}^{(k)}\}_{p,k}$  in Eq. 10. To achieve the diversity condition, we regularize the polynomial coefficients to be well distributed in the parameter space. Considering that the distances of the coefficient vectors of two scaled filters may still be large in terms of the Euclidean distance, we thereby take the cosine distance to measure the distance between the polynomial coefficients of filters. Specifically, the regularization term is:

$$
\Omega (\alpha) = \max  _ {p \neq q} \frac {\left| <   \boldsymbol {\alpha} _ {[ p ]} , \boldsymbol {\alpha} _ {[ q ]} > \right|}{\left\| \boldsymbol {\alpha} _ {[ p ]} \right\| _ {2} \left\| \boldsymbol {\alpha} _ {[ q ]} \right\| _ {2}}, \tag {14}
$$

where  $\alpha_{[p]} = [\alpha_{[p]}^{(0)},\alpha_{[p]}^{(1)},\dots,\alpha_{[p]}^{(K)}]^T$ . The max function reflects the maximum similarity between the polynomial coefficients of any pair of filters in the filter bank. Through minimizing Eq. 14, the most similar filters will have different orientations in the parameter space. Thus, all the pairs of filters tend to be different. When  $\{T_k(L)\}_{k}$  is an orthogonal basis such as Chebyshev polynomials, the diversity of polynomial coefficients  $\{\alpha_{[p]}\}_{p}$  implies that filters defined as Eq. 10 are different in the graph frequency domain. More intuitively, the message passing schemes in the spatial domain induced by the filters are different with diverse  $\{\alpha_{[p]}\}_{p}$ , as presented in Eq. 12 and Eq. 13.

We can note that, if the filter is defined on the basis composed of rectangular pulse functions $^2$ , i.e.,

$$
T _ {k} (\lambda) = \left\{ \begin{array}{l l} 1 & \frac {2 k}{K + 1} \leq \lambda <   \frac {2 (k + 1)}{K + 1} \\ 0 & \text {o t h e r s} \end{array} , \right. \tag {15}
$$

the ideal subband filter banks, whose filters have different passbands in GSP, just corresponds to the optimal solution to the regularization with  $\Omega (\alpha) = 0$  in Eq. 14, when  $K\geq s$ .

With  $T_{\Theta}(\mathcal{G}, Y)$  generally representing a target function, the overall objective function is then formulated as

$$
\min  _ {\Theta} T _ {\Theta} (\mathcal {G}, Y) + \gamma \Omega (\alpha), \tag {16}
$$

where  $Y$  indicates ground truth labels,  $\Theta$  denotes the parameter set including  $\{\alpha_{[p]}, W_{[p]}, b_{[p]}\}_{p=1}^{s}$ , and  $\gamma$  is a hyperparameter to adjust the contribution of regularization term. Like most popular graph convolution operators, BankGCN can be optimized via gradient-based methods together with modules such as graph pooling operators in GCNs. It achieves linear computational complexity with  $O(K|\mathcal{E}|d)$  and constant learning complexity, similarly to most existing MPGCNs.

# 4.5 DISCUSSION

BankGCN actually provides an effective simplification of existing polynomial-based spectral methods with parameter decomposition. With  $\Theta = [\Theta_0; \Theta_1; \ldots; \Theta_K] \in \mathbb{R}^{d \times d' \times (K + 1)}$  representing the

free parameters of the spectral methods adopting corresponding polynomial filters, BankGCN with a linear mapping as projection function actually provides a decomposition based approximation of  $\Theta$  with

$$
[ W _ {[ 1 ]} \otimes \alpha_ {[ 1 ]}, W _ {[ 2 ]} \otimes \alpha_ {[ 2 ]}, \dots , W _ {[ s ]} \otimes \alpha_ {[ s ]} ], \tag {17}
$$

where  $\otimes$  indicates outer product. This decomposition strategy allows for filter sharing to reduce potential redundancy of existing spectral methods. Multi-channel signals are first adaptively decomposed into various subspaces with  $\{W_{[p]}\}_{p}$ , and signal components within a subspace share a filter defined with  $\alpha_{[p]}$  to process corresponding characteristics. In addition, it permits to further control the relationship between filters with the diversity regularization as introduced in Section 4.4.

We further study the capacity of BankGCN in handling the spectrum of signals with a compact architecture.

Proposition 1. BankGCN can capture and preserve diverse frequency components of input graph signals, like its counterpart introduced in Eq. 2.

Proof. Please refer to Appendix B.

![](images/e0524f5fcf0ff777f1cb56fb5a19ca40d8c271c3f8da86f84c41ce2bcdf3c4f4.jpg)

As demonstrated in Prop. 1 and its proof, a single filter  $g_{[p]}$  in BankGCN together with signal projection is capable to explore and represent diverse frequency components of graph signals, as done in its counterpart with a group of filters  $\{g_{ij}\}_i$ . This theoretically validates the effectiveness of BankGCN in reducing the number of filters and thereby potential redundancy in existing polynomial-based spectral architectures, while maintaining the capability of representing diverse spectral information.

# 5 EXPERIMENTS

In this section, in order to evaluate models in learning representation of diverse spectral information of graph data, we resort to graph classification tasks instead of semi-supervised node classification tasks that are mainly based on "cluster assumption" and dependent on low-frequency information (Li et al., 2018). The proposed BankGCN is compared with a collection of popular GCNs on several TU benchmark datasets (Kersting et al., 2016), CIFAR-10 (Dwivedi et al., 2020) and Ogbg-molhiv (Hu et al., 2020).

The network in the experiment consists of four convolution layers, one graph-level readout module and a linear classifier. For a fair and complete comparison, we consider two cases, (i) the same number of features, i.e., all of models with the same number of feature maps per hidden layer, and (ii) the same number of parameters, i.e., models with nearly the same number of free parameters per hidden layer, in section 5.1 and 5.2 respectively. More details about network architecture and experimental settings are presented in Appendix C.

Baselines. We compare BankGCN with several state-of-the-art graph convolution methods. For the MPGCNs, we consider GCN (Kipf & Welling, 2017), GraphSage (Hamilton et al., 2017) with mean aggregation, GAT (8-heads) (Velikovi et al., 2018), and GIN using SUM-MLP (2 layers) that achieves the best performance (Xu et al., 2019). Regarding spectral GCNs, we compare with its counterpart ChebNets (Defferrard et al., 2016) that also adopts Chebyshev polynomial filters, and the most recent spectral method ARMA (1-stack) (Bianchi et al., 2021). For a fair comparison, the results of baseline models are obtained with the same configurations as BankGCN using the public versions provided in the pytorch-geometric package (Fey & Lenssen, 2019).

Ablation Study. Furthermore, we consider two simplifications of the proposed model for ablation study. First, BankGCN-Diff adopts a predefined filter bank. The filter bank consists of a low-pass filter, a high-pass filter, and band-pass filters from graph diffusion wavelets as used by Gama et al. (2019); Min et al. (2020). The specific frequency responses of these filters are presented in Appendix C. Second, BankGCN-NR removes the diversity regularization from BankGCN with  $\gamma = 0$ .

# 5.1 RESULTS AND ANALYSIS ON TU-BENCHMARKS

As presented in Table 1, BankGCN outperforms all the MP baselines with nearly the same or even fewer number of parameters. It further achieves better performance than its counterpart ChebNets

Table 1: Results on graph classification with 20 runs for different datasets (#P/L denotes the number of free parameters per hidden layer).  

<table><tr><td></td><td>ENZY</td><td>DD</td><td>NCII1</td><td>PROT</td><td>NCI109</td><td>MUTA</td><td>FRAN</td><td>~#P/L</td></tr><tr><td>GCN</td><td>62.75 ± 5.83</td><td>77.75 ± 3.55</td><td>79.00 ± 1.93</td><td>74.87 ± 4.08</td><td>78.90 ± 1.52</td><td>81.34 ± 1.61</td><td>62.21 ± 2.41</td><td>4.2k</td></tr><tr><td>GraphSage</td><td>66.75 ± 6.31</td><td>75.21 ± 2.72</td><td>80.97 ± 1.87</td><td>75.13 ± 4.04</td><td>79.54 ± 2.24</td><td>82.30 ± 1.48</td><td>63.91 ± 1.96</td><td>8.3k</td></tr><tr><td>GIN</td><td>61.08 ± 4.92</td><td>75.42 ± 3.31</td><td>81.19 ± 2.27</td><td>74.91 ± 3.88</td><td>80.71 ± 2.38</td><td>81.66 ± 2.48</td><td>68.11 ± 2.09</td><td>8.3k</td></tr><tr><td>GAT</td><td>62.67 ± 7.52</td><td>77.50 ± 2.14</td><td>79.43 ± 2.38</td><td>75.09 ± 4.05</td><td>79.16 ± 1.85</td><td>81.28 ± 2.20</td><td>63.89 ± 1.53</td><td>4.3k</td></tr><tr><td>ChebNets (K=2)</td><td>66.75 ± 4.79</td><td>77.67 ± 2.91</td><td>81.80 ± 2.35</td><td>74.64 ± 4.75</td><td>81.27 ± 1.89</td><td>82.50 ± 1.58</td><td>68.35 ± 2.65</td><td>12.4k</td></tr><tr><td>ARMA(K=2)</td><td>63.33 ± 6.32</td><td>78.81 ± 3.04</td><td>80.86 ± 2.58</td><td>74.91 ± 5.36</td><td>80.12 ± 1.89</td><td>81.97 ± 1.81</td><td>67.65 ± 2.35</td><td>12.4k</td></tr><tr><td>BankGCN-Diff (s=8)</td><td>66.92 ± 5.71</td><td>77.88 ± 2.81</td><td>80.07 ± 2.03</td><td>74.87 ± 4.21</td><td>79.23 ± 2.29</td><td>82.27 ± 2.00</td><td>64.63 ± 1.96</td><td>4.2k</td></tr><tr><td>BankGCN-NR (K=2,s=8)</td><td>65.83 ± 6.66</td><td>77.03 ± 4.08</td><td>81.89 ± 1.95</td><td>75.36 ± 4.68</td><td>81.03 ± 1.95</td><td>82.44 ± 1.69</td><td>67.82 ± 2.30</td><td>4.2k</td></tr><tr><td>BankGCN (K=2,s=8)</td><td>68.00 ± 5.23</td><td>78.14 ± 2.81</td><td>82.06 ± 1.75</td><td>75.67 ± 4.19</td><td>81.54 ± 2.13</td><td>82.89 ± 1.61</td><td>67.82 ± 2.30</td><td>4.2k</td></tr><tr><td>BankGCN (K=2,s=16)</td><td>66.83 ± 5.19</td><td>77.42 ± 3.50</td><td>81.93 ± 2.15</td><td>76.12 ± 5.08</td><td>81.62 ± 1.87</td><td>82.57 ± 1.61</td><td>68.43 ± 1.98</td><td>4.2k</td></tr></table>

Table 2: Study on the order  $K$  of filters and the number of subspaces  $s$  per layer.  

<table><tr><td colspan="2"></td><td>ENZY</td><td>DD</td><td>NCI1</td><td>PROT</td><td>NCI109</td><td>MUTA</td><td>FRAN</td></tr><tr><td rowspan="4">K=2</td><td>s=1</td><td>63.58 ± 6.31</td><td>76.40 ± 2.34</td><td>80.46 ± 2.34</td><td>74.38 ± 4.80</td><td>79.23 ± 2.29</td><td>82.09 ± 1.51</td><td>65.52 ± 2.44</td></tr><tr><td>s=4</td><td>66.75 ± 5.61</td><td>78.14 ± 2.81</td><td>81.62 ± 1.84</td><td>75.67 ± 4.61</td><td>81.19 ± 2.08</td><td>82.70 ± 1.63</td><td>67.48 ± 2.09</td></tr><tr><td>s=8</td><td>68.00 ± 5.23</td><td>78.14 ± 2.81</td><td>82.06 ± 1.75</td><td>75.67 ± 4.19</td><td>81.54 ± 2.13</td><td>82.89 ± 1.61</td><td>67.82 ± 2.30</td></tr><tr><td>s=16</td><td>66.83 ± 5.19</td><td>77.42 ± 3.50</td><td>81.93 ± 2.15</td><td>76.12 ± 5.08</td><td>81.62 ± 1.87</td><td>82.57 ± 1.61</td><td>68.43 ± 1.98</td></tr><tr><td>K=1</td><td rowspan="4">s=8</td><td>67.17 ± 5.68</td><td>76.99 ± 2.99</td><td>81.02 ± 1.88</td><td>75.89 ± 5.07</td><td>80.92 ± 1.66</td><td>82.40 ± 1.89</td><td>66.95 ± 1.91</td></tr><tr><td>K=2</td><td>68.00 ± 5.23</td><td>78.14 ± 2.81</td><td>82.06 ± 1.75</td><td>75.67 ± 4.19</td><td>81.54 ± 2.13</td><td>82.89 ± 1.61</td><td>67.82 ± 2.30</td></tr><tr><td>K=3</td><td>65.75 ± 5.54</td><td>77.75 ± 2.66</td><td>81.85 ± 1.92</td><td>74.96 ± 5.77</td><td>80.82 ± 1.84</td><td>82.53 ± 1.56</td><td>68.35 ± 2.13</td></tr><tr><td>K=4</td><td>65.17 ± 6.62</td><td>77.75 ± 3.01</td><td>82.46 ± 1.98</td><td>75.09 ± 4.94</td><td>81.17 ± 2.11</td><td>82.26 ± 1.71</td><td>68.35 ± 1.92</td></tr></table>

with much fewer free parameters, i.e., about  $1 / (K + 1)$ , and is also superior to the most recent spectral method ARMA on most datasets. As presented in Fig. 2, the learned filters have different frequency responses rather than only low-pass, some of them being high-pass and some focusing on middle-frequencies. With such a bank of filters, BankGCN handles the multi-channel signals flexibly and thereby achieves the best performance on almost all the datasets.

We then go one step further to evaluate the adaptive filtering capabilities. As presented in Fig. 2, the learned filters have different frequency responses on various datasets as they are adapted to the data characteristics. As listed in Tables 1 and 2, the BankGCN ( $s = 1$ ) employing one single adaptive filter still outperforms GCN with 'low-pass' filtering on most datasets; Furthermore, BankGCN is superior to its variant BankGCN-Diff that uses predefined filter banks on all the datasets. These validate the benefits of adaptive filtering to flexibly capture the spectral characteristics of data.

Study on the number of filters. Furthermore, we evaluate the adoption of filter banks rather than a single filter, and study the impact of the number of filters in the filter bank (equivalently  $s$ , the number of subspaces) on the classification performance. With a group of filters, the ability of convolution operators to handle information is enhanced. As presented in Table 1, BankGCN-Diff outperforms GCN with additional band-pass and high-pass filters on almost all the datasets, and BankGCN further improves the performance with adaptive filters. Furthermore, as  $s$  increases from 1 to 8, the performance of BankGCN is improved on most datasets, as listed in Table 2. These validate the benefits of using more than one filter. With  $s$  further increased into 16, the performance is degraded on several datasets. Given that the total dimension of all the subspaces is fixed, the dimension of each space decreases and the representation capacity of each subspace probably declines with the further growth of  $s$ .

Study on the order of filters. The order of polynomials determines the function space of filters. In the graph frequency domain, as demonstrated in Fig. 2, it can better realize the bandpass property of filters using a larger  $K$  but at the cost of a greater risk of overfitting the spectrum of training data. In the spatial domain, the value of  $K$  corresponds to the message passing range, and a large  $K$  will affect the locality of signals. Thereby, a tradeoff is needed. As shown in Table 2, BankGCN with  $K = 2$  achieves the best performance on most datasets. We notice that for the cases with complex node signals, like FRANKENSTEIN with 780-channel node attributes, a relatively large  $K$  is needed to exploit their various spectral characteristics; and the small  $K$  is preferred on simple datasets, such as the PROTEINS dataset with 3-channel node category features.

![](images/dcfd8c79420ad6b24dddd177088815cd63c4a60cab12a9ef4b04748592e6a0aa.jpg)  
(a)  $K = 2, \gamma = 0$ .

![](images/eaea5ff729a10c89a4a085d5be64fcf09bd9bce5b4126783c83fcf20f815eb03.jpg)  
(b)  $K = 2,\gamma = 0.1$

![](images/1d2c0be85e6eadb96d5b96b205c1701cf8980948a14afe1dfc26a18c607099c2.jpg)  
(c)  $K = 2, \gamma = 10$ .

![](images/08b630c312677b3f996ea50a109c09f824581c0a6a02faa5b230057bcefcfbe4.jpg)  
Figure 2: Illustrations of the frequency responses of the learned filters of BankGCN ( $s = 8$ ) in the first layer of networks. (a)  $\sim$  (d) are on NCI109 and (e) on FRANKENSTEIN.  
(d)  $K = 3, \gamma = 10$ .

![](images/90e3822f0b463467640e98e0d4226380e6eaa9661f45b87b5233e3044327926e.jpg)  
(e)  $K = 2, \gamma = 0$ .

Study on the regularization. We further evaluate the effect of the proposed diversity regularization. The comparison of BankGCN-NR and BankGCN in Table 1 shows that the regularization improves the classification performance on almost all the datasets. On the FRANKENSTEIN dataset whose signal is composed of 780-channel attributes, the regularization is not helpful. We infer that the information in such high-channel signal is complex enough to induce different filters, as presented in Fig. 2(e). Fig. 2(a)-(c) show that the learned filters in a filter bank with regularization present better diversity in frequency response, than those without regularization. For example, the filters denoted by blue and red in Fig. 2(a) are with similar frequency responses, while they are more diverse in Fig. 2(b) and Fig. 2(c). This is further verified by the maximum similarity scores of the polynomial coefficients that define the filters  $\Omega (\alpha) = 0.997$ , 0.744, and 0.649 (computed as Eq. 14) for Fig. 2(a)-(c), respectively. More results and analysis are presented in Appendix D.

# 5.2 RESULTS AND ANALYSIS ON CIFAR-10 AND OGBG-MOLHIV

BankGCN still achieves the best performance on both CIFAR-10 and Ogbg-molhiv with all the models having a similar number of free parameters per hidden layer, as presented in Table 3. Furthermore, to evaluate the generalization of models, we construct a reduced CIFAR-10 (1000) dataset by taking 100 graphs per category to form the training set, while maintaining the validation and testing sets. BankGCN still performs best on the reduced CIFAR

Table 3: Classification accuracy on CIFAR-10 and Ogbg-molhiv (no edge attributes) datasets.  

<table><tr><td>Method</td><td>CIFAR-10 Acc</td><td>CIFAR-10 (1000) Acc</td><td>Decrease</td><td>Ogbg-molhiv ROC-AUC</td></tr><tr><td>GCN</td><td>55.64 ± 0.11</td><td>36.47 ± 0.31</td><td>-34.5%</td><td>75.18 ± 1.85</td></tr><tr><td>GraphSage</td><td>63.51 ± 0.40</td><td>40.03 ± 0.56</td><td>-37.0%</td><td>75.39 ± 1.64</td></tr><tr><td>GIN</td><td>50.04 ± 0.06</td><td>31.97 ± 0.20</td><td>-36.1%</td><td>71.52 ± 1.45</td></tr><tr><td>GAT</td><td>60.34 ± 0.19</td><td>36.08 ± 0.04</td><td>-40.2%</td><td>75.08 ± 0.39</td></tr><tr><td>ChebNets (K=2)</td><td>64.33 ± 0.14</td><td>39.46 ± 0.75</td><td>-38.7%</td><td>74.69 ± 2.08</td></tr><tr><td>ChebNets (K=3)</td><td>63.62 ± 0.23</td><td>37.91 ± 0.40</td><td>-40.4%</td><td>73.17 ± 1.57</td></tr><tr><td>ARMA (K=2)</td><td>61.66 ± 0.35</td><td>32.66 ± 0.09</td><td>-47.0%</td><td>75.73 ± 1.15</td></tr><tr><td>ARMA (K=3)</td><td>61.79 ± 0.28</td><td>32.47 ± 0.32</td><td>-47.5%</td><td>74.61 ± 0.98</td></tr><tr><td>BankGCN(K=2,s=16)</td><td>66.17 ± 0.34</td><td>42.82 ± 0.33</td><td>-35.3%</td><td>77.95 ± 1.96</td></tr><tr><td>BankGCN(K=3,s=16)</td><td>66.00 ± 0.51</td><td>42.95 ± 0.49</td><td>-34.9%</td><td>75.72 ± 1.45</td></tr></table>

10 dataset and is among the models with the least performance loss compared with the full dataset. Together with ROC-AUC being a measure of the generalization ability of a model, BankGCN performs well in the sense of generalization, especially when compared with its counterpart ChebNets. These further validate the effect of the proposed sharing scheme between filters for model simplification on improving generalization.

# 6 CONCLUSION

In this paper, we propose a novel graph convolution operator, termed BankGCN, constructed on an adaptive filter bank for graph representation learning. The filter bank is equivalent to a group of learnable message passing schemes in  $K$ -hop neighborhoods. Together with subspace decomposition, BankGCN explores a sharing scheme between filters to adaptively handle information of diverse spectral characteristics with significantly fewer parameters than its competitors, and achieves excellent performance on graph classification tasks. An interesting direction for future research resides in discussing the capacity of the proposed graph convolution operator in terms of graph isomorphism test. It may also be promising to employ BankGCN in a variety of tasks on non-Euclidean data like 3-D point cloud classification and segmentation.

# REFERENCES

Sami Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Kristina Lerman, Hrayr Harutyunyan, Greg Ver Steeg, and Aram Galstyan. Mixhop: Higher-order graph convolutional architectures via sparsified neighborhood mixing. In Proc. 36th Int. Conf. Mach. Learn., pp. 21-29, Long Beach, California, USA, 2019.  
Muhammet Balcilar, Renton Guillaume, Pierre Héroux, Benoit Gaüzère, Sébastien Adam, and Paul Honeine. Analyzing the expressive power of graph neural networks in a spectral perspective. In 9th Int. Conf. Learn. Rep., 2021.  
Filippo Maria Bianchi, Daniele Grattarola, Lorenzo Livi, and Cesare Alippi. Graph neural networks with convolutionalarma filters.IEEE Trans.Pattern Anal.and Mach.Intell.,2021.  
Deyu Bo, Xiao Wang, Chuan Shi, and Huawei Shen. Beyond low-frequency information in graph convolutional networks. In Proc. 35th AAAI Conf. Artif. Intell., pp. 3950-3957, Virtual Event, 2021.  
Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally connected networks on graphs. In 2nd Int. Conf. Learn. Rep., Banff, AB, Canada, April 2014.  
Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. Adaptive universal generalized pagerank graph neural network. In 9th Int. Conf. Learn. Rep., 2021.  
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Adv. Neural Inf. Process. Syst. 29, pp. 3844-3852, Barcelona, Spain, December 2016.  
Vijay Prakash Dwivedi, Chaitanya K Joshi, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.  
Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.  
Fernando Gama, Alejandro Ribeiro, and Joan Bruna. Diffusion scattering transforms on graphs. In 7th Int. Conf. Learn. Rep., 2019.  
Feng Gao, Guy Wolf, and Matthew Hirn. Geometric scattering for graph data analysis. In Proc. 36th Int. Conf. Mach. Learn., Long Beach, California, USA, 2019.  
Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for quantum chemistry. In Proc. 34th Int. Conf. Mach. Learn., pp. 1263-1272, Sydney, NSW, Australia, August 2017.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Adv. Neural Inf. Process. Syst. 30, pp. 1024-1034, Long Beach, CA, USA, December 2017.  
David K Hammond, Pierre Vandergheynst, and Rémi Gribonval. Wavelets on graphs via spectral graph theory. Appl. Comput. Harmon. Anal., 30(2):129-150, 2011.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 770-778, 2016.  
Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In Adv. Neural Inf. Process. Syst. 33, 2020.  
Kristian Kersting, Nils M. Kriege, Christopher Morris, Petra Mutzel, and Marion Neumann. Benchmark data sets for graph kernels, 2016. URL http://graphkernels.cs.tu-dortmund.de.  
Renata Khasanova and Pascal Frossard. Graph-based isometry invariant representation learning. In Proc. 34th Int. Conf. Mach. Learn., pp. 1847-1856, Sydney, NSW, Australia, August 2017.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In 3rd Int. Conf. Learn. Rep., San Diego, CA, USA, May 2015.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In 5th Int. Conf. Learn. Rep., Toulouse, France, April 2017.  
Johannes Klicpera, Stefan Weißenberger, and Stephan Gunnemann. Diffusion improves graph learning. In Adv. Neural Inf. Process. Syst. 32, pp. 13354-13366, 2019.  
Junhyun Lee, Inyeop Lee, and Jaewoo Kang. Self-attention graph pooling. In Proc. 36th Int. Conf. Mach. Learn., pp. 3734–3743, Long Beach, CA, USA, June 2019.  
Ron Levie, Federico Monti, Xavier Bresson, and Michael M. Bronstein. CayleyNets: Graph convolutional neural networks with complex rational spectral filters. IEEE Trans. Signal Process., 67 (1):97-109, January 2018.  
Ron Levie, Wei Huang, Lorenzo Bucci, Michael M Bronstein, and Gitta Kutyniok. Transferability of spectral graph convolutional neural networks. arXiv preprint arXiv:1907.12972, 2019.  
Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-supervised learning. In Proc. 32nd AAAI Conf. Artif. Intell., pp. 3538-3545, New Orleans, LA, USA, February 2018.  
Yimeng Min, Frederik Wenkel, and Guy Wolf. Scattering GCN: overcoming oversmoothness in graph convolutional networks. In Adv. Neural Inf. Process. Syst. 33, Virtual Event, 2020.  
Sunil K Narang and Antonio Ortega. Perfect reconstruction two-channel wavelet filter banks for graph structured data. IEEE Trans. Signal Process., 60(6):2786-2799, 2012.  
Sunil K Narang and Antonio Ortega. Compact support biorthogonal wavelet filterbanks for arbitrary undirected graphs. IEEE Trans. Signal Process., 61(19):4673-4685, 2013.  
Hoang NT and Takanori Maehara. Revisiting graph neural networks: All we have is low-pass filters. arXiv preprint arXiv:1905.09550, 2019.  
Adam Paszke et al. Automatic differentiation in PyTorch. In Adv. Neural Inf. Process. Syst., Autodiff Workshop, Long Beach, CA, USA, December 2017.  
David I. Shuman, Sunil K. Narang, Pascal Frossard, Antonio Ortega, and Pierre Vandergheynst. The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains. IEEE Signal Process. Mag., 30(3):83-98, May 2013.  
Yuichi Tanaka and Akie Sakiyama.  $m$ -channel oversampled graph filter banks. IEEE Trans. Signal Process., 62(14):3578-3590, 2014.  
Dorina Thanou, David I Shuman, and Pascal Frossard. Learning parametric dictionaries for signals on graphs. IEEE Trans. Signal Process., 62(15):3849-3862, 2014.  
Nicolas Tremblay, Paulo Gonçalves, and Pierre Borgnat. Design of graph filters and filterbanks. In Cooperative and Graph Signal Processing, pp. 299-324. Elsevier, 2018.  
Petar Velikovi et al. Graph attention networks. In 6th Int. Conf. Learn. Rep., Vancouver, BC, Canada, May 2018.  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In 7th Int. Conf. Learn. Rep., New Orleans, LA, USA, May 2019.  
Keyulu Xu et al. Representation learning on graphs with jumping knowledge networks. In Proc. 35th Int. Conf. Mach. Learn., pp. 5449-5458, Stockholm, Sweden, July 2018.
