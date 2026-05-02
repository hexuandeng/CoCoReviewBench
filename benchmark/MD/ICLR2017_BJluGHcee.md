# TENSORIAL MIXTURE MODELS

Or Sharir, Ronen Tamari, Nadav Cohen & Amnon Shashua

The Hebrew University of Jerusalem

{or.sharir,ronent,cohennadav,shashua}@cs.huji.ac.il

# ABSTRACT

We introduce a generative model, we call Tensorial Mixture Models (TMMs) based on mixtures of basic component distributions over local structures (e.g. patches in an image) where the dependencies between the local-structures are represented by a "priors tensor" holding the prior probabilities of assigning a component distribution to each local-structure.

In their general form, TMMs are intractable as the priors tensor is typically of exponential size. However, when the priors tensor is decomposed it gives rise to an arithmetic circuit which in turn transforms the TMM into a Convolutional Arithmetic Circuit (ConvAC). A ConvAC corresponds to a shallow (single hidden layer) network when the priors tensor is decomposed by a CP (sum of rank-1) approach and corresponds to a deep network when the decomposition follows the Hierarchical Tucker (HT) model.

The ConvAC representation of a TMM possesses several attractive properties. First, the inference is tractable and is implemented by a forward pass through a deep network. Second, the architectural design of the model follows the deep networks community design, i.e., the structure of TMMs is determined by just two easily understood factors: size of pooling windows and number of channels. Finally, we demonstrate the effectiveness of our model when tackling the problem of classification with missing data, leveraging TMMs unique ability of tractable marginalization which leads to optimal classifiers regardless of the missingness distribution.

# 1 INTRODUCTION

Generative models have played a crucial part in the early development of the field of Machine Learning. However, in recent years they were mostly cast aside in favor of discriminative models, lead by the rise of ConvNets (LeCun et al., 2015), which were found to perform equally well or better than classical generative counter-parts on almost any task. Despite the increased interest in unsupervised learning, many of the recent studies on generative models choose to focus solely on the generation capabilities of these models (Goodfellow et al., 2014; Gregor et al., 2015; van den Oord et al., 2016; Dinh et al., 2016; Tran et al., 2016; Chen et al., 2016; Kingma et al., 2016; Kim and Bengio, 2016). There is much less emphasis on leveraging generative models to solve actual tasks, e.g. semi-supervised learning (Kingma et al., 2014; Springenberg, 2016; Maaloe et al., 2016; Forster et al., 2015; Salimans et al., 2016), image restoration (Dinh et al., 2014; Bengio et al., 2014; van den Oord et al., 2016; Zoran and Weiss, 2011; Rosenbaum and Weiss, 2015; Sohl-Dickstein et al., 2015; Theis and Bethge, 2015) or unsupervised feature representation (Radford et al., 2016; Coates et al., 2011). Nevertheless, work on generative models for solving actual problems are yet to show a meaningful advantage over competing discriminative models.

On the most fundamental level, the difference between a generative model and a discriminative one is simply the difference between learning  $P(X,Y)$  and learning  $P(Y|X)$ , respectively. While it is always possible to infer  $P(Y|X)$  given  $P(X,Y)$ , it might not be immediately apparent why the generative objective is preferred over the discriminative one. In Ng and Jordan (2002), this question was studied w.r.t. the sample complexity, proving that under some cases it can be significantly lesser in favor of the generative classifier. However, their analysis was limited only to specific pairs of discriminative and generative classifiers, and they did not present a general case where the generative method is undeniably preferred. We wish to highlight one such case, where learning

$P(X,Y)$  is provenly better regardless of the models in question, by examining the problem of classification with missing data. Despite the artificially well-behave nature of the typical classification benchmarks presented in current publications, real-world data is usually riddled with noise and missing values – instead of observing  $X$  we only have a partial observation  $\hat{X}$  – a situation that tends to be ignored in modern research. Discriminative models have no natural mechanisms to handle missing data and instead must rely on data imputation, i.e. filling missing data by a preprocessing step prior to prediction. Unlike the discriminative approaches, generative models are naturally fitted to handle missing data by simply marginalizing over the unknown values in  $P(X,Y)$ , from which we can attain  $P(Y|\hat{X})$  by an application of Bayes Rule. Moreover, under mild assumptions which apply to many real-world settings, this method is proven to be optimal regardless of the process by which values become missing (see sec. 5 for a more detailed discussion).

While almost all generative models can represent  $P(X,Y)$ , only few can actually infer its exact value efficiently. Models which possess this property are said to have tractable inference. Many studies specifically address the hard problem of learning generative models that do not have this property. Notable amongst those are works based on Variational Inference (Kingma and Welling, 2014; Kingma et al., 2014; Blei et al., 2003; Wang and Grimson, 2007; Makhzani et al., 2015; Kingma et al., 2016), which only provide approximated inference, and ones based on Generative Adversarial Networks (Goodfellow et al., 2014; Radford et al., 2016; Springenberg, 2016; Chen et al., 2016; Salimans et al., 2016; Makhzani et al., 2015), which completely circumvent the inference problem by restructuring the learning problem as a two-player game of discriminative objectives – both of these approaches are incapable of tractable inference.

There are several advantages to models with tractable inference (e.g. they could be simpler to train), and as we have shown above, this property is also a requirement for proper handling of missing data in the form of marginalization. In practice, to marginalize over  $P(X,Y)$  means to perform integration on it, thus, even if it is tractable to compute  $P(X,Y)$ , it still might not be tractable to compute every possible marginalization. Models which are capable of this are said to have tractable marginalization. Mixture Models (e.g. Gaussian Mixture Models) are the classical example of a generative model with tractable inference, as well as tractable marginalization. Though they are simple to understand, easy to train and even known to be universal - can approximate any distribution given sufficient capacity - they do not scale well to high-dimensional data. The Gaussian Mixture Model is an example of a shallow model - containing just a single latent variable - with limited expressive efficiency. More generally, Graphical Models are deep and exponentially more expressive, capable of representing intricate relations between many latent variables. While not all kinds of Graphical Models are tractable, many are, e.g. Latent Tree Models (Zhang, 2004; Mourad et al., 2013) and Sum-Product Networks (Poon and Domingos, 2011). The main issue with generic graphical models is that by virtue of being too general they lack the inductive bias needed to efficiently model unstructured data, e.g. images or text. Despite the success of structure learning algorithms (Huang et al., 2015; Gens and Domingos, 2013; Adel et al., 2015) on structured datasets, such as discovering a hierarchy among diseases in patients health records, there are no similar results on unstructured datasets. Indeed some recent works on the subject have failed to solve even simple handwritten digit classification tasks (Adel et al., 2015). Thus deploying graphical models on such cases requires experts to manually design the model. Other attempts which harness neural networks blocks (Dinh et al., 2014; 2016) offer tractable inference, but not tractable marginalization.

To summarize, most generative models do not have tractable inference, and of the few models which do, they all possess one or more of the following shortcomings: (i) they do not possess the expressive capacity to model high-dimensional data (e.g. images), (ii) they require explicitly designing all the dependencies of the data, or (iii) they do not have tractable marginalization.

We present in this paper a family of generative models we call Tensorial Mixture Models (TMMs), which aim to address the above shortcomings of alternative models. Under TMMs, we assume that the data generated by our model is composed of a sequence of local-structures (e.g. patches in an image), where each local-structure is generated from a small set of simple component distributions (e.g. Gaussian), and the dependencies between the local-structures are represented by a prior tensor holding the prior probabilities of assigning a component distribution to each local-structure. In their general form, TMMs are intractable as the prior tensor is typically of exponential size. However, by decomposing the prior tensor, inference of TMMs becomes realizable by Convolutional Arithmetic Circuits (ConvACs) – a recently proposed (Cohen et al., 2016a) ConvNet architecture based on two

![](images/47068b5cb7847625e2270d559a9314a60d333113d74c6694f8d0aae872f0e221.jpg)  
Figure 1: The decoding algorithm of an arbitrary tensor decomposition represented by a ConvAC.

operations, weighted sum and product pooling – which enables both tractable inference as well as tractable marginalization. While Graphical Models are typically hard to design, ConvACs follow the same design conventions of modern ConvNets, which reduces the task of designing a model to simply choosing the number of channels at each layer, and size of pooling windows. ConvACs were also the subject of several theoretical studies on its expressive capacity (Cohen et al., 2016a; Cohen and Shashua, 2016b) and comparing them to ConvNets (Cohen and Shashua, 2016a), showing they are especially suitable for high-dimensional natural data (images, audio, etc.) with a non-negligible advantage over standard ConvNets. Sum-Product Networks are another kind of Graphical Model realizable by Arithmetic Circuits, but they do not possess the same theoretical guarantees, nor do they provide a simple method to design efficient and expressive models.

The rest of the article is organized as follows. In sec. 2 we briefly review mathematical background on tensors required in order to follow our work. This is followed by sec. 3 which presents our generative model and its theoretical properties. How our model is trained is covered in sec. 4, and a thorough discussion on the importance of marginalization and its implications on our model is given in sec. 5. We conclude the article by presenting our experiments on classification with missing data in sec. 6, and revisit the main points of the article and future research in sec. 7.

# 2 PRELIMINARIES

We begin by establishing the minimal background in the field of tensor analysis required for following our work (see app. A for a more detailed review of the subject). A tensor is best thought of as a multi-dimensional array  $\mathcal{A}_{d_1,\ldots ,d_N}\in \mathbb{R}$ , where  $\forall i\in [N],d_i\in [M_i]$  and  $N$  is referred to as the order of the tensor. For our purposes we typically assume that  $M_{1} = \dots = M_{N} = M$ , and denote it as  $\mathcal{A}\in (\mathbb{R}^M)^{\otimes N}$ . It is immediately apparent that performing operations with tensors, or simply storing them, quickly becomes intractable due to their exponential size of  $M^{N}$ . That is one of the primary motivations behind tensor decomposition, which can be seen as a generalization of low-rank matrix factorization.

The relationship between tensor decomposition and networks arises from the simple observation, that through decomposition one can tradeoff storage complexity with computation, where the type of computation consists of sums and products. Specifically, the decompositions could be described by a compact representation coupled with a decoding algorithm of polynomial complexity to retrieve the entries of the tensor. Most tensor decompositions have a decoding algorithm representable via computation graphs of products and weighted sums, also known as Arithmetic Circuits (Shpilka and Yehudayoff, 2010) or Sum-Product Networks (Poon and Domingos, 2011). More specifically, these circuits take as input  $N$  indicator vectors  $\delta_1,\ldots ,\delta_N$ , representing the coordinates  $(d_{1},\dots,d_{N})$  where  $\delta_i = \mathbf{1}_{[j = d_i]}$ , and output the value of  $\mathcal{A}_{d_1,\dots,d_N}$ , where the weights of these circuits form the compact representation of tensors.

Applying this perspective to two of the most common decomposition formats, CANDE-COMP/PARFAC (CP) and Hierarchical Tucker (HT), give rise to a shared framework for representing their decoding circuits by convolutional networks as illustrated in fig. 1, where a shallow network with one hidden layer corresponds to the CP decomposition, and a deep network with  $\log_2(N)$  hidden layers corresponds to the HT decomposition. The networks consist of just product pooling and  $1\times 1$  conv layers. Having no point-wise activations between the layers, the non-linearity of the models stems from the product pooling operation itself. The pooling layers also control the depth of the network by the choice of the size and the shape of pooling windows. The conv operator is not unlike the standard convolutional layer of ConvNets, with the sole difference being that it may operate without coefficient sharing, i.e. the filters that generate feature maps by sliding across the

previous layer may have different coefficients at different spatial locations. This is often referred to in the deep learning community as a locally-connected operator (Taigman et al., 2014).

Arithmetic Circuits constructed from the above conv and product pooling layers are called Convolutional Arithmetic Circuits, or ConvACs for short, first suggested by Cohen et al. (2016a) as a theoretical framework for studying standard convolutional networks, sharing many of the defining traits of the latter, most noteworthy, the locality, sharing and pooling properties of ConvNets. Unlike general circuits, the structure of the network is determined solely by two parameters, the number of channels of each conv layer and the size of pooling windows, which indirectly controls the depth of the network. Any decomposition that corresponds to a ConvAC can represent any tensor, given sufficient number of channels, though deeper circuits result in more efficient representations (Cohen et al., 2016a).

Finally, since we are dealing with generative models, the tensors we study are non-negative and sum to one, i.e. the vectorization of  $\mathcal{A}$  (rearranging its entries to the shape of a vector), denoted by  $\operatorname{vec}(\mathcal{A})$ , is constrained to lie in the multi-dimensional simplex, denoted by:

$$
\triangle^ {k} := \left\{\mathbf {x} \in \mathbb {R} ^ {k + 1} \mid \sum_ {i = 1} ^ {k + 1} x _ {i} = 1, \forall i \in [ k + 1 ]: x _ {i} \geq 0 \right\} \tag {1}
$$

# 3 TENSORIAL MIXTURE MODELS

We represent the input signal  $X$  by a sequence of low-dimensional local structures

$$
X = \left(\mathbf {x} _ {1}, \dots , \mathbf {x} _ {N}\right) \in \left(\mathbb {R} ^ {s}\right) ^ {N}
$$

This representation is quite natural for many high-dimensional input domains such as images - where the local structures represent patches consisting of  $s$  pixels - voice through spectrograms, and text through words.

A well-known observation, which has been verified in several empirical studies (e.g. by Zoran and Weiss (2011)), is that the distributions of local structures typically found in natural data could be sufficiently modeled by a mixture model consisting of only few components (on the order of 100) of simple distributions (e.g. Gaussian). Assuming the above holds for  $X \in (\mathbb{R}^s)^N$  and let  $\{P(\mathbf{x}|d;\theta_d)\}_{d=1}^M$  be the mixing components, parameterized by  $\theta_1, \ldots, \theta_M$ , from which local structures are generated, i.e. for all  $i \in [N]$  there exist  $d_i \in [M]$  such that  $\mathbf{x}_i \sim P(\mathbf{x}|d_i; \theta_{d_i})$ , where  $d_i$  is a hidden variable specifying the matching component for the  $i$ -th local structure, then the probability density of sampling  $X$  is fully described by:

$$
P (X) = \sum_ {d _ {1}, \dots , d _ {N} = 1} ^ {M} P \left(d _ {1}, \dots , d _ {N}\right) \prod_ {i = 1} ^ {N} P \left(\mathbf {x} _ {i} \mid d _ {i}; \theta_ {d _ {i}}\right) \tag {2}
$$

where  $P(d_{1},\ldots ,d_{N})$  represents the prior probability of assigning components  $d_{1},\ldots ,d_{N}$  to their respective local structures  $\mathbf{x}_1,\dots ,\mathbf{x}_N$ . Even though we had to make an assumption on  $X$  to derive eq. 2, it is important to note that if we allow  $M$  to become unbounded, then any distribution with support in  $(\mathbb{R}^s)^N$  could be approximated by this equation. The argument follows from the universality property of the common parametric families of distributions (Gaussian, Laplacian, etc.), where any distribution can be approximated given sufficient number of components from these families, and thus the assumption always holds to some degree (see app. B for the complete proof).

The prior probabilities  $P(d_{1},\ldots ,d_{N})$  can also be represented by a tensor  $\mathcal{A}\in (\mathbb{R}^{M})^{\otimes N}$  of order  $N$ , given that the vectorization of  $\mathcal{A}$  is constrained to the simplex, i.e.  $\mathrm{vec}(\mathcal{A})\in \triangle^{(M^N -1)}$  (see eq. 1). Thus, we refer to eq. 2 as a Tensorial Mixture Model (TMM) with priors tensor  $\mathcal{A}$  and mixing components  $P(\mathbf{x}|d_1;\theta_1),\dots,P(\mathbf{x}|d_N;\theta_N)$ . Notice that if  $N = 1$  then we obtain the standard mixture model, whereas for a general  $N$  it is equivalent to a mixture model with tensorised mixing weights and conditionally independent mixing components.

Unlike standard mixture models, we cannot perform inference directly from eq. 2, nor can we even store the priors tensor directly given its exponential size of  $M^N$  entries. Therefore the TMM as presented by eq. 2 is not tractable. The way to make the TMM tractable is to replace the tensor  $\mathcal{A}_{d_1,\dots,d_N}$  by a tensor decomposition and, as described in the previous section, this gives rise to arithmetic circuits. But before we present our approach for tractable TMMs through tensor decompositions, it is worth examining some of the TMM special cases and how they relate to other known generative models.

![](images/0c9907c8ab33b59332a4b7c1ed80f62f254c8724319ce2d78200a9d4ff570e8e.jpg)  
Figure 2: Inference of a TMM carried out by a ConvAC.

# 3.1 SPECIAL CASES

We have already shown that TMMs can be thought of as a special case of mixture models, but it is important to also note that diagonal Gaussian Mixture Models (GMMs), probably the most common type of mixture models, are a strict subset of TMMs. Assume  $M = N \cdot K$ , as well as:

$$
P (d _ {1}, \ldots , d _ {N}) = \left\{ \begin{array}{l l} w _ {k} & \forall i \in [ N ], d _ {i} = N \cdot (k - 1) + i \\ 0 & \text {O t h e r w i s e} \end{array} \right.
$$

$$
P (\mathbf {x} \mid d; \theta_ {d}) = \mathcal {N} (\mathbf {x}; \boldsymbol {\mu} _ {k i}, \operatorname {d i a g} \left(\boldsymbol {\sigma} _ {k i} ^ {2}\right)), d = N \cdot (k - 1) + i
$$

then eq. 2 reduces to:

$$
P (X) = \sum_ {k = 1} ^ {K} w _ {k} \prod_ {i = 1} ^ {N} \mathcal {N} (\mathbf {x}; \boldsymbol {\mu} _ {k i}, \operatorname {d i a g} \left(\boldsymbol {\sigma} _ {k i} ^ {2}\right)) = \sum_ {k = 1} ^ {K} w _ {k} \mathcal {N} (\mathbf {x}; \tilde {\boldsymbol {\mu}} _ {k}, \operatorname {d i a g} \left(\tilde {\boldsymbol {\sigma}} _ {k} ^ {2}\right))
$$

$$
\tilde {\boldsymbol {\mu}} _ {k} = \left(\boldsymbol {\mu} _ {k 1} ^ {T}, \dots , \boldsymbol {\mu} _ {k N} ^ {T}\right) ^ {T} \quad \tilde {\boldsymbol {\sigma}} _ {k} ^ {2} = \left(\left(\boldsymbol {\sigma} _ {k 1} ^ {2}\right) ^ {T}, \dots , \left(\boldsymbol {\sigma} _ {k N} ^ {2}\right) ^ {T}\right) ^ {T}
$$

which is equivalent to a diagonal GMM with mixing weights  $\mathbf{w} \in \triangle^{K-1}$  and Gaussian mixture components with means  $\{\tilde{\mu}_k\}_{k=1}^K$  and covariances  $\{\mathrm{diag}(\tilde{\sigma}_k^2)\}_{k=1}^K$ .

While the previous example highlights another connection between TMMs and mixture models, it does not take full advantage of the priors tensor, setting most of its entries to zero. Perhaps the simplest assumption we could make about the priors tensor, without it becoming degenerate, would be to assume that the hidden variables  $d_{1}, \ldots, d_{N}$  are statistically independent, i.e.  $P(d_{1}, \ldots, d_{N}) = \prod_{i=1}^{N} P(d_{i})$ . Then rearranging eq. 2 will result in a product of mixture models:

$$
P (X) = \prod_ {i = 1} ^ {N} \sum_ {d = 1} ^ {M} P (d _ {i} = d) P (\mathbf {x} _ {i} | d _ {i} = d; \theta_ {d})
$$

If we also assume that the priors are identical in addition to being independent, i.e.  $P(d_{1} = d) = \ldots = P(d_{N} = d)$ , then this model becomes a bag-of-words model, where the components  $\{P(\mathbf{x}|d;\theta_d)\}_{d = 1}^{M}$  define a soft dictionary for translating local-structures into "words", as is often done when applying bag-of-words models to images. Despite this familiar setting, had we subscribed to only using independent priors, we would lose the universality property of the general TMM model – it would not be capable of modeling dependencies between the local-structures.

# 3.2 DECOMPOSING THE PRIORS TENSOR

We have just seen that TMMs could be made tractable through constraints on the priors tensor, but it was at the expense of either not taking advantage of its tensor structure, or losing its universality property. Our approach for tractable TMMs is to apply tensor decompositions to the priors tensor, which is the conventional method for tackling the exponential size of high-order tensors.

We have already mentioned in sec. 2 that any decomposition representable by ConvACs, including the well-known CP and HT decompositions, can represent any tensor, and thus applying them would not limit the expressivity of our model. Fixing a ConvAC representing the priors tensor, i.e.  $\Phi_{\Theta}(\delta_1,\ldots ,\delta_N) = \mathcal{A}_{d_1,\ldots ,d_N}$  where  $\Theta$  are the parameters of the ConvAC and  $\{\delta_i\}_{i = 1}^N$  are the indicators representation of  $\{d_i\}_{i = 1}^N$ , and simply rearranging the terms of eq. 2 after substituting the entries of the priors tensor with the sums and products expression of  $\Phi_{\Theta}(\delta_1,\dots ,\delta_N)$  results in:

$$
P (X) = \Phi_ {\Theta} \left(\mathbf {q} ^ {1}, \dots , \mathbf {q} ^ {N}\right) \quad \forall i \in [ N ] \forall d \in [ M ], q _ {d} ^ {i} = P \left(\mathbf {x} _ {i} \mid d _ {i} = d\right) \tag {3}
$$

which is nearly equivalent to how the ConvAC is used for computing the entries of the priors tensor, differing only in the way the input vectors are defined. Namely, eq. 3 is a result of

replacing indicator vectors  $\delta_{i}$  with probability vectors  $\mathbf{q}^i$ , which could be interpreted as a soft variant of indicator vectors. Viewed as a network, it begins with a representation layer, mapping the local structures to the likelihood probabilities of belonging to each mixing component, i.e.  $\{\mathbf{x}_i\}_{i=1}^N \rightarrow \{P(\mathbf{x}_i | d_i = d; \theta_d)\}_{i=1,d=1}^{N,M}$ . Following the representation layer is the same ConvAC described by  $\Phi_\Theta(\cdot, \dots, \cdot)$ . The complete network is illustrated by fig. 2.

Unlike general tensors, for a TMM to represent a valid distribution, the priors tensor is constrained to the simplex and thus not every choice of parameters for the decomposition would result in a tensor holding this constraint. By restricting ourselves to non-negative decomposition parameters, i.e. use positive weights in the  $1 \times 1$  conv layers, it guarantees the resulting tensors would be non-negative as well. Additionally, normalizing the non-negative tensor is equivalent to requiring the parameters to be restricted to the simplex, i.e. for every layer  $l$  and spatial position  $j$  the weight vector  $\mathbf{w}^{l,j} \in \triangle^{r_{l-1}-1}$  of the respective  $1 \times 1$  conv kernel is normalized to sum to one. Under these constraints we refer to it as a generative decomposition. Notice that restricting ourselves to generative decompositions does not limit the expressivity of our model, as we can still represent any non-negative tensor and thus any distribution that the original TMM could represent. In discussing the above, it helps to distinguish between the two extreme cases of generative decompositions representable by ConvACs, namely, the shallow Generative CP decomposition referred to as the GCP-model, and the deep Generative HT decomposition referred to as the GHT-model.

Non-negative matrix and tensor decompositions have a long history together with the development of corresponding generative models, e.g., pLSA (Hofmann, 1999) which uses non-negative matrix decompositions for text analysis, which was later extended for images with the help of "visual words" (Li and Perona, 2005). The non-negative variant of the CP decomposition presented above is related to the more general Latent Class Models (Zhang, 2004), which could be seen as a multi-dimensional pLSA. Likewise, the non-negative HT decomposition is related to the Latent Tree Model (Zhang, 2004; Mourad et al., 2013) with the structure of a complete binary tree. Thus both the GCP and GHT models can be represented as a two-level graphical model, where the top level is either an LCM or an LTM, and the bottom level represent the local structures which are conditionally sampled from the mixing components of the TMM.

To conclude, the application of ConvACs to decompose the priors tensor leads to tractable TMMs with inference implemented by convolutional networks, has deep roots to classical use of nonnegative factorizations of generative models, and given sufficient resources does not limit expressivity. However, practical considerations raise the question on the extent of the expressive capacity of our models when the size of the ConvAC is polynomial with respect to the number of local structures and mixing components. This question was thoroughly studied in a series of works analyzing the importance of depth (Cohen et al., 2016a), compared them to the expressive capacity of ConvNets (Cohen and Shashua, 2016a), showing the latter is less capable than ConvACs, and the ability of ConvACs to model the dependency structure typically found in natural data (Cohen and Shashua, 2016b). We prove in app. D that their main results are not hindered by the introduction of simplex constraints to ConvACs as we did above. Together these results give us a detailed understanding of how the number of channels and size of pooling windows control the expressivity of the model. A more in depth overview of their results and its application to our models can be found in app. C.

# 3.3 COMPARISON TO SUM-PRODUCT NETWORKS

Sum-Product Networks (SPNs) are a related class of generative models which are also realized by Arithmetic Circuits, though not strictly convolutional circuits as defined above. While SPNs can realize any ConvAC and thus are universal and posses tractable inference, their lack of structure puts them at a disadvantage.

Picking the right SPN structure from the infinite possible combinations of sum and product nodes could be perplexing even for experts in the field. Indeed Poon and Domingos (2011); Gens and Domingos (2012) had to hand-engineer complex structures for each dataset guided by prior knowledge and heuristics, and while their results were impressive for their time, they are poor by current measures. This lead to many works studying the task of learning the structure directly from the data itself (Peharz et al., 2013; Gens and Domingos, 2013; Adel et al., 2015; Rooshenas and Lowd, 2014), which indeed improved upon manually designed SPNs on some tasks. Nevertheless, when

![](images/33c272023148c6718a46d9bac2a6fa12b45f893579a97507ceb946343db6079d.jpg)  
Figure 3: Classifier variant of TMM carried out by a ConvAC.

compared in absolute terms compared to other models, and not just average log-likelihood, they do not perform well even on simple handwritten digit classification datasets (Adel et al., 2015).

As opposed to SPNs, TMMs implemented with ConvACs have an easily designed architecture with only two set of parameters, size of pooling windows and number of channels, both of which can be directly related to the expressivity of the model as detailed in app. C. Additionally, while SPNs are typically trained using special EM-type algorithms, TMMs are trained using the stochastic gradient descent type algorithms as is common in training neural networks (see sec. 4 for details), thereby benefiting from the shared experience of a large and growing community.

# 4 CLASSIFICATION AND LEARNING WITH TMMS

Until this point we presented the TMM as a generative model for high-dimensional data, which is universal, and whose structure is tightly coupled to that of convolutional networks. We have yet to incorporate classification and learning into our framework. This is the purpose of the current section.

The common way to introduce object classes into a generative framework is to consider a class variable  $Y$ , and the distributions  $P(X|Y)$  of the instance  $X$  conditioned on  $Y$ . Under our model this is equivalent to having shared mixing components, but different priors tensors  $P(d_1, \ldots, d_N | Y = y)$  for each class. Though it is possible to decompose each priors tensor separately, it is much more efficient to employ the concept of joint tensor decomposition, and use a shared ConvAC instead. This results in a single ConvAC computing inference, where instead of a single scalar output, multiple outputs are driven by the network – one for each class – as illustrated through the network in fig. 3.

Heading on to predicting the class of a given instance, we note that in practice, naive implementation of ConvACs is not numerically stable, the reason being that high degree polynomials (as computed by such networks) are easily susceptible to numerical underflow or overflow. The conventional method for tackling this issue is to perform all computations in log-space. This transforms ConvACs into SimNets, a recently introduced deep learning architecture (Cohen and Shashua, 2014; Cohen et al., 2016b). Finally, prediction is carried by returning the most likely class, which in the common setting of uniform class priors  $(P_{\Theta}(Y = y)\equiv 1 / K)$ , translates to simply predicting the class for which the corresponding network output is maximal, in accordance with standard neural network practice:

$$
\hat {Y} (X) = \operatorname {a r g m a x} _ {y} P (Y = y | X) = \operatorname {a r g m a x} _ {y} \log P (X | Y = y)
$$

Suppose now that we are given a training set  $S = \{(X^{(i)}\in (\mathbb{R}^s)^N,Y^{(i)}\in [K])\}_{i = 1}^{|S|}$  of instances and labels, and would like to fit the parameters  $\Theta$  of multi-class TMM according to the Maximum Likelihood method. Equivalently, we minimize the Negative Log-Likelihood (NLL) loss function:  $\mathcal{L}(\Theta) = \mathbf{E}[-\log P_{\Theta}(X,Y)]$ , which can be factorized into two separate loss functions:

$$
\mathcal {L} (\Theta) = \mathbf {E} \left[ - \log P _ {\Theta} (Y | X) \right] + \mathbf {E} \left[ - \log P _ {\Theta} (X) \right]
$$

where  $\mathbf{E}[-\log P_{\Theta}(Y|X)]$  is commonly known as the cross-entropy loss, which we refer to as the discriminative loss, while  $\mathbf{E}[-\log P_{\Theta}(X)]$  corresponds to maximizing the prior likelihood  $P(X)$  and has no analogy in standard discriminative neural networks. It is this term that captures the generative nature of our model, and we accordingly refer to it as the generative loss. Now, let  $N_{\Theta}(X^{(i)};y)\coloneqq \log P_{\Theta}(X^{(i)}|Y = y)$  stand for the  $y$  th output of the SimNet (ConvAC in log-space) realizing the TMM with parameters  $\Theta$ , then in the case of uniform class priors, the empirical estimation of  $\mathcal{L}(\Theta)$  may be written as:

$$
\mathcal {L} (\Theta ; S) = - \frac {1}{| S |} \sum_ {i = 1} ^ {| S |} \log \frac {e ^ {N _ {\Theta} (X ^ {(i)} ; Y ^ {(i)})}}{\sum_ {y = 1} ^ {K} e ^ {N _ {\Theta} (X ^ {(i)} ; y)}} - \frac {1}{| S |} \sum_ {i = 1} ^ {| S |} \log \sum_ {y = 1} ^ {K} e ^ {N _ {\Theta} (X ^ {(i)}; y)} \tag {4}
$$

Maximum likelihood training of generative models is oftentimes based on dedicated algorithms such as Expectation-Maximization, which are typically difficult to apply at scale. We leverage the resemblance between our objective (eq. 4) and that of standard neural networks, and apply the same optimization procedures used for the latter, which have proven to be extremely effective for training classifiers at scale. Whereas other works have used tensor decompositions for the optimization of probabilistic models (Song et al., 2013; Anandkumar et al., 2014), we employ them strictly for modeling and instead make use of conventional methods. In particular, our implementation of TMMs is based on the SimNets extension of Caffe toolbox (Cohen et al., 2016b; Jia et al., 2014), and uses standard Stochastic Gradient Descent-type methods for optimization (see sec. 6 for more details).

# 5 CLASSIFICATION WITH MISSING DATA THROUGH MARGINALIZATION

A major advantage of generative models over discriminative ones lies in the ability to cope with missing data, specifically in the context of classification. By and large, discriminative methods either attempt to complete missing parts of the data before classification, known as data imputation, or learn directly to classify data with missing values (Little and Rubin, 2002). The first of these approaches relies on the quality of data completion, a much more difficult task than the original one of classification with missing data. Even if the completion was optimal, the resulting classifier is known to be sub-optimal (see app. E). The second approach does not make this assumption, but nonetheless assumes that the distribution of missing values at train and test times are similar, a condition which often does not hold in practice. Indeed, Globerson and Roweis (2006) coined the term "nightmare at test time" to refer to the common situation where a classifier must cope with missing data whose distribution is different from that encountered in training.

As opposed to discriminative methods, generative models are endowed with a natural mechanism for classification with missing data. Namely, a generative model can simply marginalize over missing values, effectively classifying under all possible completions, weighing each completion according to its probability. This, however, requires tractable inference and marginalization. We have already shown in sec. 3 that TMM support the former, and will show in sec. 5.1 bring forth marginalization which is just as efficient. Beforehand, we lay out the formulation of classification with missing data.

Let  $\mathcal{X}$  be a random vector in  $\mathbb{R}^s$  representing an object, and  $\mathcal{Y}$  be a random variable in  $[K] := \{1, \dots, K\}$  representing its label. Denote by  $\mathcal{D}(\mathcal{X}, \mathcal{Y})$  the joint distribution of  $(\mathcal{X}, \mathcal{Y})$ , and by  $(\mathbf{x} \in \mathbb{R}^s, y \in [K])$  specific realizations thereof. Assume that after sampling a specific instance  $(\mathbf{x}, y)$ , a random binary vector  $\mathcal{M}$  is drawn conditioned on  $\mathcal{X} = \mathbf{x}$ . More concretely, we sample a binary mask  $\mathbf{m} \in \{0, 1\}^s$  (realization of  $\mathcal{M}$ ) according to a distribution  $\mathcal{Q}(\cdot | \mathcal{X} = \mathbf{x})$ .  $x_i$  is considered missing if  $m_i$  is equal to zero, and observed otherwise. Formally, we consider the vector  $\mathbf{x} \odot \mathbf{m}$ , whose  $i$ 'th coordinate is defined to hold  $x_i$  if  $m_i = 1$ , and the wildcard * if  $m_i = 0$ . The classification task is then to predict  $y$  given access solely to  $\mathbf{x} \odot \mathbf{m}$ .

Following the works of Rubin (1976); Little and Rubin (2002), we consider three cases for the missingness distribution  $\mathcal{Q}(\mathcal{M} = \mathbf{m}|\mathcal{X} = \mathbf{x})$ : missing completely at random (MCAR), where  $\mathcal{M}$  is independent of  $\mathcal{X}$ , i.e.  $\mathcal{Q}(\mathcal{M} = \mathbf{m}|\mathcal{X} = \mathbf{x})$  is a function of  $\mathbf{m}$  but not of  $\mathbf{x}$ ; missing at random (MAR), where  $\mathcal{M}$  is independent of the missing values in  $\mathcal{X}$ , i.e.  $\mathcal{Q}(\mathcal{M} = \mathbf{m}|\mathcal{X} = \mathbf{x})$  is a function of both  $\mathbf{m}$  and  $\mathbf{x}$ , but is not affected by changes in  $x_{i}$  if  $m_{i} = 0$ ; and missing not at random (MNAR), covering the rest of the distributions for which  $\mathcal{M}$  depends on missing values in  $\mathcal{X}$ , i.e.  $\mathcal{Q}(\mathcal{M} = \mathbf{m}|\mathcal{X} = \mathbf{x})$  is a function of both  $\mathbf{m}$  and  $\mathbf{x}$ , which at least sometimes is sensitive to changes in  $x_{i}$  when  $m_{i} = 0$ .

Let  $\mathcal{P}$  be the joint distribution of the object  $\mathcal{X}$ , label  $\mathcal{Y}$ , and missingness mask  $\mathcal{M}$ :

$$
\mathcal {P} (\mathcal {X} = \mathbf {x}, \mathcal {Y} = y, \mathcal {M} = \mathbf {m}) = \mathcal {D} (\mathcal {X} = \mathbf {x}, \mathcal {Y} = y) \cdot \mathcal {Q} (\mathcal {M} = \mathbf {m} | \mathcal {X} = \mathbf {x})
$$

For given  $x \in \mathbb{R}^s$  and  $\mathbf{m} \in \{0,1\}^s$ , denote by  $o(\mathbf{x},\mathbf{m})$  the event where the random vector  $\mathcal{X}$  coincides with  $\mathbf{x}$  on the coordinates  $i$  for which  $m_i = 1$ . For example, if  $\mathbf{m}$  is an all-zero vector  $o(\mathbf{x},\mathbf{m})$  covers the entire probability space, and if  $\mathbf{m}$  is an all-one vector  $o(\mathbf{x},\mathbf{m})$  corresponds to the event  $\mathcal{X} = \mathbf{x}$ . With these notations in hand, we are now in a position to characterize the optimal predictor in the presence of missing data:

Claim 1. For any data distribution  $\mathcal{D}$  and missingness distribution  $\mathcal{Q}$ , the optimal classification rule in terms of 0-1 loss is given by:

$$
h ^ {*} (\mathbf {x} \odot \mathbf {m}) = \operatorname {a r g m a x} _ {y} \mathcal {P} (\mathcal {Y} = y | o (\mathbf {x}, \mathbf {m})) \mathcal {P} (\mathcal {M} = \mathbf {m} | o (\mathbf {x}, \mathbf {m}), \mathcal {Y} = y)
$$

Proof. See app. E.

![](images/70c45992d37c0dddc82e8ff9f607781632f8491a3ad5fdcbf0c3688e20e398c5.jpg)

When the distribution  $\mathcal{Q}$  is MAR (or MCAR), the classifier admits a simpler form, referred to as the marginalized Bayes predictor:

Corollary 1. Under the conditions of claim 1, if the distribution  $\mathcal{Q}$  is MAR (or MCAR), the optimal classification rule may be written as:

$$
h ^ {*} (\mathbf {x} \odot \mathbf {m}) = \operatorname {a r g m a x} _ {y} \mathcal {P} (\mathcal {Y} = y | o (\mathbf {x}, \mathbf {m})) \tag {5}
$$

Proof. See app. E.

![](images/e12aebc80f06de513c61e733fc2373a4aad6b764f49c375ae8b31baf335293a3.jpg)

Corollary 1 indicates that in the MAR setting, which is frequently encountered in practice, optimal classification does not require prior knowledge regarding the missingness distribution  $\mathcal{Q}$ . As long as one is able to realize the marginalized Bayes predictor (eq. 5), or equivalently, to compute the likelihoods of observed values conditioned on labels  $(\mathcal{P}(o(\mathbf{x},\mathbf{m})|Y = y))$ , classification with missing data is guaranteed to be optimal, regardless of the corruption process taking place. This is in stark contrast to discriminative methods, which require access to the missingness distribution during training, and thus are not able to cope with unknown conditions at test time.

Most of this section dealt with the task of prediction given an input with missing data, where we assumed we had access to a complete and uncorrupted training set, and only faced missingness during prediction. However, many times we wish to tackle the reverse problem, where the training set itself is riddled with missing data. Generative methods can once again leverage their natural ability to handle missing data in the form of marginalization during the learning stage. Generative models are typically learned through the Maximum Likelihood principle. When it comes to learning from missing data, the marginalized likelihood objective is used instead. Under the MAR assumption, this method results in an unbiased classifier (Little and Rubin, 2002).

# 5.1 EFFICIENT MARGINALIZATION WITH TMMS

As discussed above, with generative models optimal classification with missing data (in the MAR setting) is oblivious to the specific missingness distribution. However, it requires tractable computation of the likelihood of observed values conditioned on labels, i.e. tractable marginalization over missing values. The plurality of generative models that have recently gained attention in the deep learning community (Goodfellow et al., 2014; Kingma and Welling, 2014; Dinh et al., 2014; 2016) do not meet this requirement, and thus are not suitable for classification with missing data. TMMs on the other hand bring forth extremely efficient marginalization, requiring only a single forward pass through the corresponding network. Details follow.

Recall from sec. 3 and 4 that a multi-class TMM realizes the following form:

$$
P \left(\mathbf {x} _ {1}, \dots , \mathbf {x} _ {N} \mid Y = y\right) = \sum_ {d _ {1}, \dots , d _ {N}} ^ {M} P \left(d _ {1}, \dots , d _ {N} \mid Y = y\right) \prod_ {i = 1} ^ {N} P \left(\mathbf {x} _ {i} \mid d _ {i}; \theta_ {d _ {i}}\right) \tag {6}
$$

Suppose now that only the local structures  $\mathbf{x}_{i_1}\ldots \mathbf{x}_{i_V}$  are observed, and we would like to marginalize over the rest. Integrating eq. 6 gives:

$$
P (\mathbf {x} _ {i _ {1}}, \ldots , \mathbf {x} _ {i _ {V}} | Y = y) = \sum_ {d _ {1}, \ldots , d _ {N}} ^ {M} P (d _ {1}, \ldots , d _ {N} | Y = y) \prod_ {v = 1} ^ {V} P (\mathbf {x} _ {i _ {v}} | d _ {i _ {v}}; \theta_ {d _ {i _ {v}}})
$$

from which it is evident that the same ConvAC used to compute  $P(\mathbf{x}_1, \ldots, \mathbf{x}_N | Y = y)$ , can be used to compute  $P(\mathbf{x}_{i_1}, \ldots, \mathbf{x}_{i_V} | Y = y)$  – all it requires is a slight adaptation of the representation layer. Namely, the latter would represent observed values through the usual likelihoods, whereas missing (marginalized) values would now be represented via constant ones:

$$
\operatorname {r e p} (i, d) = \left\{ \begin{array}{l l} 1 & , \mathbf {x} _ {i} \text {i s m i s s i n g (m a r g i n a l i z e d)} \\ P (\mathbf {x} _ {i} | d; \Theta) & , \mathbf {x} _ {i} \text {i s v i s i b l e (n o t m a r g i n a l i z e d)} \end{array} \right.
$$

To conclude, with TMMs marginalizing over missing values is just as efficient as plain inference - requires only a single pass through the corresponding ConvAC. Accordingly, the marginalized Bayes predictor (eq. 5) is realized efficiently, and classification with missing data (in the MAR setting) is optimal, regardless of the missingness distribution. This capability is not provided by discriminative methods, which rely on the distribution of missing values being know at training, and by contemporary generative models, which do not bring forth tractable marginalization.

<table><tr><td></td><td>N=0</td><td>25</td><td>50</td><td>75</td><td>100</td><td>125</td><td>150</td></tr><tr><td>LP-Based</td><td>97.9</td><td>97.5</td><td>96.4</td><td>94.1</td><td>89.2</td><td>80.9</td><td>70.2</td></tr><tr><td>GHT-model</td><td>98.5</td><td>98.2</td><td>97.8</td><td>96.5</td><td>93.9</td><td>87.1</td><td>76.3</td></tr></table>

Table 1: Blind classification with missing data on the binary MNIST dataset with feature deletion noise according to Globerson and Roweis (2006), averaged over all pairs of digits.

# 6 EXPERIMENTS

We demonstrate the properties of our models through both qualitative and quantitative experiments. In subsec. 6.1 we present our state-of-the-art results on image classification with missing data, with robustness to various missingness distributions. In app. G we show visualizations produced by our models, which gives us insight into its inner workings. Our experiments were conducted on the MNIST digit classification dataset, consisting of 60000 grayscale images of single digit numbers, as well as the small NORB 3D object recognition dataset, consisting of 48600 grayscale stereo images of toys belonging to 5 categories: four-legged animals, human figures, airplanes, trucks, and cars

In all our experiments we use either the GCP or GHT model with Gaussian mixing components. The weights of the conv layers are partially shared as described in sec 3.2, and are represented in log-space. For the case of the GHT model, we use  $2 \times 2$  pooling windows for all pooling layers. We train our model according to the loss described in sec. 4, using the Adam (Kingma and Ba, 2015) variant of SGD and decaying learning rates. We apply  $L^2$ -regularization to the weights while taking into account they are stored in log-space. Additionally, we also adapt a probabilistic interpretation of dropout (?) by introducing random marginalization layers, that randomly select spatial locations in the input and marginalize over them. We provide a complete and detailed description of our experiments in app. F.

Our implementation, which is based on Caffe (Jia et al., 2014) and MAPS (Ben-Nun et al., 2015), as well as other code for reproducing our experiments, is available through our Github repository: https://github.com/HUJI-Deep/TMM.

# 6.1 IMAGE CLASSIFICATION WITH MISSING DATA

We demonstrate the effectiveness of our method for classification with missing data of unknown missingness distribution (see sec. 5), by conducting three kinds of experiments on the MNIST dataset, and an additional experiment on the NORB dataset. We begin by following the protocol of Globerson and Roweis (2006) – the binary classification problem of digit pairs with feature deletion noise – where we compare our method to the best known result on that benchmark (Dekel and Shamir, 2008). For our main experiment, we move to the harder multi-class digit classification under two different MAR missingness distributions, comparing against other methods which do not assume a specific missingness distribution. We repeat this experiment on the NORB dataset as well. Finally, our last experiment demonstrates the failure of purely discriminative methods to adapt to previously unseen missingness distributions, underlining the importance of the generative approach to missing data. We do wish to emphasize that missing data is not typically found in most image data, nevertheless, experiments on images with missing data are very common, for both classification and inpainting tasks. Additionally, there is nothing about our method, nor the methods we compare it against, that is very specific to the image domain, and thus any conclusion drawn should not be limited to the chosen datasets, but be taken in the broader context of the missing data problem.

The problem of learning classifiers which are robust to unforeseen missingness distributions at test time was first proposed by Globerson and Roweis (2006). They suggested missing values could be denoted by values which were deleted, i.e. their values were changed to zero, and a robust classifier would have to assume that any of its zero-value inputs could be the result of such a deletion process, and must be treated as missing. Their solution was to train a linear classifier and formulate the optimization as a quadric program under the constraint that  $N$  of its features could be deleted. In Dekel and Shamir (2008), this solution was improved upon and generalized to other kinds of corruption beyond deletion as well as to an adversarial setting.

We follow the central experiment of these articles, conducted on binary classification of digits pairs from the MNIST dataset, where  $N$  non-zero pixels are deleted with uniform probability over the set of  $N$  non-zero pixel locations of the given image. We compare our method, using the deep GHT-

![](images/265595888b73a6c212f315af1a31e530141953bcbd85cfd417ccac69e56c45fc.jpg)

![](images/d23f8887560c8529fa20582d0dbcd833c1a314d64951106b330667ef5efb93c8.jpg)  
(b) MNIST with missing rectangles.

![](images/8890e7b930e53c8a03c25eb075c6095cf675c46c8b49526311fa93bc778c3825.jpg)  
(a) MNIST with i.i.d. corruption.  
(c) NORB with i.i.d. corruption.  
Figure 4: Blind classification with missing data. (a,c) Testing i.i.d. corruption with probability  $p$  for each pixel. (b,d) Testing missing rectangles corruption with  $N$  missing rectangles, each of width and height equal to  $W$ . (*) Accuracies are estimated from the plot of Goodfellow et al. (2013). ( $\dagger$ ) Data imputation algorithms followed by a ConvNet. Raw results can be found in app. H.

![](images/6a0ec26f61cfdf171c94ace87e1122786b498d36e87a5029444b664bb4b49eeb.jpg)  
(d) NORB with missing rectangles.

Model, solely against the LP-based algorithm of Dekel and Shamir (2008), which is the previous state-of-the-art on this task. Due to the limited computational resources at the time, the original experiments were limited to training sets of just 50 images per digit. We have repeated their experiment, using the implementation kindly supplied to us by the authors, and increased the limit to 300 images per digit, which is the maximal amount possible with our current computational resources. Though it is possible to train our own models using much larger training sets, we have trained them under the same limitations. Despite the fact that missingness distribution of this experiment is of the MNAR type, which our method was not guaranteed to be optimal under, the test results (see table 1) clearly show the large gap between our method and theirs. Additionally, whereas our method uses a single model trained once and with no prior knowledge on the missingness distribution, their method requires training special classifiers for each value of  $N$ , chosen through a cross-validation process, disqualifying it from being truly blind to the missingness distribution.

We continue to our main experiments on multi-class blind classification with missing data, where the missingness distribution is completely unknown during test time, and a single classifier must handle all possible distributions. We simulate two kinds of MAR missingness distributions: (i) an i.i.d. mask with a fixed probability  $p \in [0,1]$  of missing each pixel, and (ii) a mask composed of the union of  $N$  possibly overlapping rectangles of width and height equal to  $W$ , each with a randomly assigned position in the image, distributed uniformly. We evaluate both our shallow GCP-Model as well as the deep GHT-Model against the most widely used methods for blind classification with missing data. We repeat these experiments on the MNIST and NORB datasets, the results of which are presented in fig. 4.

As a baseline for our results, we use K-Nearest Neighbors (KNN) to vote on the most likely class of a given example. We extend KNN to missing data by comparing distances using only the observed entries, i.e. for a corrupted instance  $\mathbf{x} \odot \mathbf{m}$ , and a clean image from the training set  $\tilde{\mathbf{x}}$ , we compute:  $d(\tilde{\mathbf{x}}, \mathbf{x} \odot \mathbf{m}) = \sum_{m_{ij} = 1} (\tilde{x}_{ij} - x_{ij})^2$ . Though it scores better than the majority of modern methods we have compared, in practice KNN is very inefficient, even more so for missing data, which prevents most common memory and runtime optimizations typically employed to reduce its inefficiency. Additionally, KNN does not generalize well for more complex datasets, as is evident by its poor performance on the clean test set of the NORB dataset.

<table><tr><td>\( p_{\text{train}} \)</td><td>0.25</td><td>0.50</td><td>0.75</td><td>0.90</td><td>0.95</td><td>0.99</td></tr><tr><td>0.25</td><td>98.9</td><td>97.8</td><td>78.9</td><td>32.4</td><td>17.6</td><td>11.0</td></tr><tr><td>0.50</td><td>99.1</td><td>98.6</td><td>94.6</td><td>68.1</td><td>37.9</td><td>12.9</td></tr><tr><td>0.75</td><td>98.9</td><td>98.7</td><td>97.2</td><td>83.9</td><td>56.4</td><td>16.7</td></tr><tr><td>0.90</td><td>97.6</td><td>97.5</td><td>96.7</td><td>89.0</td><td>71.0</td><td>21.3</td></tr><tr><td>0.95</td><td>95.7</td><td>95.6</td><td>94.8</td><td>88.3</td><td>74.0</td><td>30.5</td></tr><tr><td>0.99</td><td>87.3</td><td>86.7</td><td>85.0</td><td>78.2</td><td>66.2</td><td>31.3</td></tr><tr><td>i.i.d. (rand)</td><td>98.7</td><td>98.4</td><td>97.0</td><td>87.6</td><td>70.6</td><td>29.6</td></tr><tr><td>rects (rand)</td><td>98.2</td><td>95.7</td><td>83.2</td><td>54.7</td><td>35.8</td><td>17.5</td></tr></table>

(a) MNIST with i.i.d. corruption

![](images/bb47d4ec1f5a81fb60a17d25c68908be4e58a61cf6ff10fd93fba35005c44fcf.jpg)  
Figure 5: We compare ConvNets trained on one distribution while tested on others. Training on randomly (rand) chosen distributions were also examined. (a) Trained on i.i.d. corruption with probability  $p_{\mathrm{train}}$ , while tested on i.i.d. corruption with probability  $p_{\mathrm{test}}$ . (b) Train and tested on the same (fixed) missing rectangles distribution, against ones trained on randomly chosen distributions.

(b) MNIST with missing rectangles.

As discusses in sec. 5, data-imputation is the most common method to handle missing data of unknown missingness distributions. Despite the popularity of this method, high quality data imputations are very hard to produce, amplified by the fact that classification algorithms are known to be highly sensitive to even a small noise applied to their inputs (?). Even if we assume the data-imputation step was done optimally, it would still not give optimal performance under all MAR missingness distributions, and under some settings could produce results which are only half as good as our method (see app. E for such a case). In our experiments, we have applied several data-imputations methods to complete the missing data, followed by classifying its outputs using a standard ConvNet fitted to the fully-observed training set. We first tested naive heuristics, filling missing values with zeros or the mean pixel value computed over all the images in the dataset. We then tested three generative models: GSN (Bengio et al., 2014), NICE (Dinh et al., 2014) and DPM (Sohl-Dickstein et al., 2015), which are known to work well for inpainting. GSN was omitted from the NORB experiments as we have not manage to properly train it on that dataset. Though the data-imputation methods are competitive when only few of the pixels are missing, they all fall far behind our models above a certain threshold, with more than 50 percentage points separating our GHT-model from the best data-imputation method under some of the cases. Additionally, all the generative models require very long runtimes, which prevents from using them in most real-world applications. While we tried to be as comprehensive as possible when choosing which inpainting methods to use, some of the most recent studies on the subject, e.g. the works of van den Oord et al. (2016) and Pathak et al. (2016), have either not yet published their code or only partially published it. We have also ruled out inpainting algorithms which are made specifically for images, as we did not want to limit the implications of these experiments solely to images.

We have also compared ourselves to the published results of the MPDBM model (Goodfellow et al., 2013). Unlike the previous generative models we tested, MPDBM is a generative classifier similar to our method. However, unlike our model, MPDBM does not possess the tractable marginalization nor the tractable inference properties, and uses approximations instead. Its lesser performance underlines the importance of these properties for achieving optimality under missing data. An additional factor might also be their training method, which includes randomly picking a subset of variables to act as missing, which might have introduced a bias to the specific missingness distribution used during their training.

In order to demonstrate the ineffectiveness of purely discriminative models, we trained ConvNets directly on randomly corrupted instances according to pre-selected missingness distributions on the MNIST dataset. Unlike the previous experiments, we do allow prior knowledge about the missingness distribution during training time. We found that the best results are achieved when replacing missing values with zeros, and adding as an extra input channel the mask of missing values (known as flag data-imputation). The results (see fig. 5) unequivocally show the effectiveness of this method when tested on the same distribution it was trained on, achieving a high accuracy even when only  $10\%$  of the pixels are visible. However, when tested on different distributions, whether on a completely different kind or even on the same kind but with different parameters, the accuracy drops by a large factor, at times by more than 35 percentage points. This illustrates the disadvantage of the discriminative method, as it necessarily incorporates bias towards the corruption process it had seen during training, which makes it fail on other distributions. One might wonder whether it is

possible for a single network to be robust on more than a single distribution. We found out that the latter is true, and if we train a network on multiple different missingness distributions<sup>1</sup>, then the network will achieve good performance on all such distributions, though at some cases not reaching the optimal performance. However, though it is possible to train a network to be robust on more than one distribution, the type of missingness distributions are rarely known in advance, and there is no known method to train a neural network against all possible distributions, limiting the effectivity of this method in practice.

Unlike all the above methods, our GHT-model, which is trained only once on the clean dataset, match or sometimes even surpass the performance of ConvNets that are trained and tested on the same distribution, showing it is achieving near optimal performance – as much as possible on any given distribution. Additionally, note that similar to ConvNets and according to the theory in app. C, the deep GHT-model is decidedly superior to the shallow GCP-model. Experimenting on more complex datasets is left for further research. Progress on optimization and regularization of networks based on product pooling (even in log-space) is required, and ways to incorporate larger  $b \times b$  convolutional operations with overlaps would be useful before we venture into larger and complex datasets. Nevertheless, our preliminary results demonstrate an overwhelming advantage of our TMM models compared to competing methods, both in terms of robustness to different types of missing data, as well as in terms of raw performance, with very wide gaps in absolute accuracy than the next best method, at times as large as 50 percentage points more than the next best method.

# 7 SUMMARY

We have introduced a new family of probabilistic models, which we call Tensorial Mixture Models. TMMs are based on a simple assumption on the data, which stems from known empirical results on natural images, that gives rise to mixture models with tensorial structure represented by the priors tensor. When the priors tensor is decomposed it gives rise to an arithmetic circuit which in turn transforms the TMM into a Convolutional Arithmetic Circuit (ConvAC). A ConvAC corresponds to a shallow (single hidden layer) network when the priors tensor is decomposed by a CP (sum of rank-1) approach and corresponds to a deep network when the decomposition follows the Hierarchical Tucker (HT) model.

The ConvAC representation of a TMM possesses several attractive properties. First, the inference is tractable and is implemented by a forward pass through a deep network. Second, the architectural design of the model follows the deep networks community design, i.e., the structure of TMMs is determined by just two easily understood factors: size of pooling windows and number of channels. Finally, we have demonstrated the effectiveness of our model when tackling the problem of classification with missing data, leveraging TMMs unique ability of tractable marginalization which leads to optimal classifiers regardless of the missingness distribution.

There are several avenues for future research on TMMs which we are currently looking at, including other problems which TMMs could solve (e.g. semi-supervised learning), experimenting with other ConvACs architectures (e.g. through different decompositions), and further progress on optimization and regularization of networks with product pooling.

# REFERENCES

Tameem Adel, David Balduzzi, and Ali Ghodsi. Learning the Structure of Sum-Product Networks via an SVD-based Algorithm. UAI, 2015.  
Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. Journal of Machine Learning Research (), 15(1):2773-2832, 2014.  
Tal Ben-Nun, Ely Levy, Amnon Barak, and Eri Rubin. Memory Access Patterns: The Missing Piece of the Multi-GPU Puzzle. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pages 19:1-19:12. ACM, 2015.

Yoshua Bengio, Éric Thibodeau-Laufer, Guillaume Alain, and Jason Yosinski. Deep Generative Stochastic Networks Trainable by Backprop. In International Conference on Machine Learning, 2014.  
David M Blei, Andrew Y Ng, and Michael I Jordan. Latent dirichlet allocation. the Journal of machine Learning research, 3:993-1022, March 2003.  
Richard Caron and Tim Traynor. The Zero Set of a Polynomial. WSMR Report 05-02, 2005.  
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets. arXiv.org, June 2016.  
Adam Coates, Andrew Y Ng, and Honglak Lee. An Analysis of Single-Layer Networks in Unsupervised Feature Learning. International Conference on Artificial Intelligence and Statistics, pages 215-223, 2011.  
Nadav Cohen and Amnon Shashua. SimNets: A Generalization of Convolutional Networks. In Advances in Neural Information Processing Systems NIPS, Deep Learning Workshop, 2014.  
Nadav Cohen and Amnon Shashua. Convolutional Rectifier Networks as Generalized Tensor Decompositions. In International Conference on Machine Learning, May 2016a.  
Nadav Cohen and Amnon Shashua. Inductive Bias of Deep Convolutional Networks through Pooling Geometry. arXiv.org, May 2016b.  
Nadav Cohen, Or Sharir, and Amnon Shashua. On the Expressive Power of Deep Learning: A Tensor Analysis. In Conference on Learning Theory COLT, May 2016a.  
Nadav Cohen, Or Sharir, and Amnon Shashua. Deep SimNets. In Computer Vision and Pattern Recognition CVPR, May 2016b.  
Ofer Dekel and Ohad Shamir. Learning to classify with missing and corrupted features. In International Conference on Machine Learning. ACM, 2008.  
Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear Independent Components Estimation. arXiv.org, October 2014.  
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP. arXiv.org, May 2016.  
Dennis Forster, Abdul-Saboor Sheikh, and Jörg Lücke. Neural Simpletrons - Minimalistic Probabilistic Networks for Learning With Few Labels. arXiv.org, June 2015.  
R Gens and P M Domingos. Learning the Structure of Sum-Product Networks. Internation Conference on Machine Learning, 2013.  
Robert Gens and Pedro M Domingos. Discriminative Learning of Sum-Product Networks. Advances in Neural Information Processing Systems, 2012.  
Amir Globerson and Sam Roweis. Nightmare at test time: robust learning by feature deletion. In International Conference on Machine Learning. ACM, 2006.  
Ian Goodfellow, Mehdi Mirza, Aaron Courville, and Yoshua Bengio. Multi-Prediction Deep Boltzmann Machines. Advances in Neural Information Processing Systems, 2013.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Nets. Advances in Neural Information Processing Systems, 2014.  
Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, and Daan Wierstra. DRAW: A Recurrent Neural Network For Image Generation. In International Conference on Machine Learning ICML, 2015.  
W Hackbusch and S Kuhn. A New Scheme for the Tensor Representation. Journal of Fourier Analysis and Applications, 15(5):706-722, 2009.  
Thomas Hofmann. Probabilistic latent semantic analysis. Morgan Kaufmann Publishers Inc., July 1999.  
Furong Huang, Niranjan U N, Ioakeim Perros, Robert Chen, Jimeng Sun, and Anima Anandkumar. Scalable Latent Tree Model and its Application to Health Analytics. In NIPS Machine Learning for Healthcare Workshop, 2015.

Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross B Girshick, Sergio Guadarrama, and Trevor Darrell. Caffe: Convolutional Architecture for Fast Feature Embedding. CoRR abs/1202.2745, cs.CV, 2014.  
Taesup Kim and Yoshua Bengio. Deep Directed Generative Models with Energy-Based Probability Estimation. arXiv.org, June 2016.  
Diederik Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In International Conference on Learning Representations, 2015.  
Diederik P Kingma and Max Welling. Auto-Encoding Variational Bayes. In International Conference on Learning Representations, 2014.  
Diederik P Kingma, Danilo J Rezende, Shakir Mohamed, and Max Welling. Semi-Supervised Learning with Deep Generative Models. In Advances in Neural Information Processing Systems, 2014.  
Diederik P Kingma, Tim Salimans, and Max Welling. Improving Variational Inference with Inverse Autoregressive Flow. In Advances in Neural Information Processing Systems, June 2016.  
Yan LeCun, Leon Bottou, Joshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436-444, May 2015.  
Fei-Fei Li and Pietro Perona. A Bayesian Hierarchical Model for Learning Natural Scene Categories. Computer Vision and Pattern Recognition, 2:524-531, 2005.  
Roderick J A Little and Donald B Rubin. Statistical analysis with missing data (2nd edition). John Wiley & Sons, Inc., September 2002.  
Lars Maaløe, Casper Kaae Sønderby, Søren Kaae Sønderby, and Ole Winther. Auxiliary Deep Generative Models. In International Conference on Machine Learning ICML, May 2016.  
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan Frey. Adversarial Autoencoders. arXiv.org, November 2015.  
Raphaël Mourad, Christine Sinoquet, Nevin Lianwen Zhang, Tengfei Liu, and Philippe Leray. A Survey on Latent Tree Models and Applications. J. Artif. Intell. Res. (), cs.LG:157-203, 2013.  
Andrew Y Ng and Michael I Jordan. On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes. In Advances in Neural Information Processing Systems NIPS, Deep Learning Workshop, 2002.  
Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context Encoders: Feature Learning by Inpainting. In Computer Vision and Pattern Recognition CVPR, May 2016.  
F Pedregosa, G Varoquaux, A Gramfort, V Michel, B Thirion, O Grisel, M Blondel, P Prettenhofer, R Weiss, V Dubourg, J Vanderplas, A Passos, D Cournaepau, M Brucher, M Perrot, and E Duchesnay. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research (), 12:2825-2830, 2011.  
Robert Peharz, Bernhard C Geiger, and Franz Pernkopf. Greedy Part-Wise Learning of Sum-Product Networks. In Machine Learning and Knowledge Discovery in Databases, pages 612-627. Springer Berlin Heidelberg, Berlin, Heidelberg, September 2013.  
Hoifung Poon and Pedro Domingos. Sum-Product Networks: A New Deep Architecture. In Uncertainty in Artificial Intelligence, 2011.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. In International Conference on Learning Representations ICLR, 2016.  
Amirmohammad Rooshenas and Daniel Lowd. Learning Sum-Product Networks with Direct and Indirect Variable Interactions. ICML, 2014.  
Dan Rosenbaum and Yair Weiss. The return of the gating network: combining generative models and discriminative training in natural image priors. In Advances in Neural Information Processing Systems. Hebrew University of Jerusalem, MIT Press, 2015.  
Donald B Rubin. Inference and missing data. Biometrika, 63(3):581-592, December 1976.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved Techniques for Training GANs. In Advances in Neural Information Processing Systems, 2016.  
Amir Shpilka and Amir Yehudayoff. Arithmetic Circuits: A survey of recent results and open questions. Foundations and Trends in Theoretical Computer Science, 5(3-4):207-388, March 2010.  
Jascha Sohl-Dickstein, Eric A Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. Internation Conference on Machine Learning, 2015.  
Le Song, Mariya Ishteva, Ankur P Parikh, Eric P Xing, and Haesun Park. Hierarchical Tensor Decomposition of Latent Tree Graphical Models. ICML, pages 334-342, 2013.  
Jost Tobias Springenberg. Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks. In International Conference on Learning Representations, 2016.  
Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, and Lior Wolf. DeepFace: Closing the Gap to Human-Level Performance in Face Verification. In Computer Vision and Pattern Recognition CVPR. IEEE Computer Society, June 2014.  
Lucas Theis and Matthias Bethge. Generative Image Modeling Using Spatial LSTMs. In Advances in Neural Information Processing Systems, 2015.  
Dustin Tran, Rajesh Ranganath, and David M Blei. The Variational Gaussian Process. In International Conference on Learning Representations ICLR, 2016.  
Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel Recurrent Neural Networks. In International Conference on Machine Learning, 2016.  
Xiaogang Wang and Eric Grimson. Spatial Latent Dirichlet Allocation. Advances in Neural Information Processing Systems, 2007.  
Matthew D Zeiler and Rob Fergus. Visualizing and Understanding Convolutional Networks. In European Conference on Computer Vision. Springer International Publishing, 2014.  
Nevin Lianwen Zhang. Hierarchical Latent Class Models for Cluster Analysis. Journal of Machine Learning Research (), pages 697-723, 2004.  
Daniel Zoran and Yair Weiss. From learning models of natural image patches to whole image restoration. ICCV, pages 479-486, 2011.

![](images/9af3f7bd135d047ae8651f93002b0e21e87ee3f88eb3ef2f9a3382d4011bf3e8.jpg)  
Figure 6: The decoding algorithm of the CP decomposition represented by an Arithmetic Circuit.
