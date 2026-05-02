# HIERARCHICAL COMPOSITIONAL FEATURE LEARNING

Miguel Lázaro-Gredilla, Yi Liu, D. Scott Phoenix, Dileep George

Vicarious

San Francisco, CA, USA

{miguel,yiliu,scott,dileep}@vicarious.com

# ABSTRACT

We introduce the hierarchical compositional network (HCN), a directed generative model able to discover and disentangle, without supervision, the building blocks of a set of binary images. The building blocks are binary features defined hierarchically as a composition of some of the features in the layer immediately below, arranged in a particular manner. At a high level, HCN is similar to a sigmoid belief network with pooling. Inference and learning in HCN are very challenging and existing variational approximations do not work satisfactorily. A main contribution of this work is to show that both can be addressed using max-product message passing (MPMP) with a particular schedule (no EM required). Also, using MPMP as an inference engine for HCN makes new tasks simple: adding supervision information, classifying images, or performing inpainting all correspond to clamping some variables of the model to their known values and running MPMP on the rest. When used for classification, fast inference with HCN has exactly the same functional form as a convolutional neural network (CNN) with linear activations and binary weights. However, HCN's features are qualitatively very different.

# 1 INTRODUCTION

Deep neural networks coupled with the availability of vast amounts of data have proved very successful over the last few years at visual discrimination (Goodfellow et al., 2014; Kingma & Welling, 2013; LeCun et al., 1998; Mnih & Gregor, 2014). A basic desire of deep architectures is to discover the blocks – or features – that compose an image (or in general, a sensory input) at different levels of abstraction. Tasks that require some degree of image understanding can be performed more easily when using representations based on these building blocks.

It would make intuitive sense that if we were to train one of the above models (particularly, those that are generative, such as variational autoencoders or generative adversarial networks) on images containing, e.g. text, the learned features would be individual letters, since those are the building blocks of the provided images. In addition to matching our intuition, one can argue that a model that realizes (from noisy raw pixels) that the building blocks of text are letters, and is able to extract a representation based on those, has "understood" the nature of the observations and can prove it by being able to efficiently compress text images.

![](images/d429bfb538478800a9c7fd12651f42a0698653d8a8ca0354b72b252449078528.jpg)  
#

![](images/a674fa621525a8f9b64d0c4e05bc1ae5a99990a1fa3525be6b30349c92525cee.jpg)  
Figure 1: Features extracted by HCN. Left: from multiple images. Right: from a single image.

![](images/e56b03f9df304828f5f9c642f0c75ee13f15c14c93a9d4c2ad2ab06388a3a0fd.jpg)  
#

![](images/2139d6b9a5e2991bac7df831c0b1e952d4c67805be8f1b688c0b7ef6eea7e665.jpg)

However, this is not the case with existing incarnations of the above models<sup>1</sup>. We can see in Fig. 1 the features recovered by the hierarchical compositional network (HCN) from a single image with no supervision. They appear to be reasonable building blocks and are easy to find for a human. Yet we are not aware of any model that can perform such apparently simple recovery with no supervision.

The HCN is a multilayer generative model with features defined at each layer. A feature (at a given position) is defined as the composition of features of the layer immediately below (by specifying their relative positions). To increase flexibility, the positions of the composing features can be perturbed slightly with respect to their default values (pooling). This results in a latent variable model, with some of the latent variables (the features) being shared for all images while others (the pool states) are specific for each image.

Comparing HCN with other generative models for images, we note that existing models tend to have at least one of the following limitations: a) priors are not rich enough; typically, the sources of variation are not distributed among the layers of the network, and instead the generative model is expressed as  $X = f(Y) + \varepsilon$  where  $Y$  and  $\varepsilon$  are two set of random variables,  $X$  is the generated image and  $f(\cdot)$  is the network, i.e., the entire network behaves as a sophisticated deterministic function, b) the inference method (usually a separate recognition network) considers all the latent variables as independent and does not solve explaining away, which leads to c) the learned features being not directly interpretable as reusable parts of the learned images.

Although directed models enjoy important advantages such as the ability to represent causal semantics and easy sampling mechanics, it is known that the "explaining away" phenomenon makes inference difficult in these models (Hinton et al., 2006). For this reason, representation learning efforts have largely focused on undirected models (Salakhutdinov & Hinton, 2009), or have tried to avoid the problem of explaining away by using complementary priors (Hinton et al., 2006).

An important contribution of this work is to show that approximate inference using max-product message passing (MPMP) can learn features that are composable, interpretable and causally meaningful. It is also noteworthy that unlike previous works, we consider the weights (a.k.a. features) to be latent variables and not parameters. Thus, we do not use separate expectation-maximization (EM) stages. Instead, we perform feature learning and pool state inference jointly as part of the same message passing loop.

When augmented with supervision information, HCN can be used for classification, with inference and learning still being taken care of by a largely unmodified MPMP procedure. After training, discrimination can be achieved via a fast forward pass which turns out to have the same functional form as a convolutional neural network (CNN).

The rest of the paper is organized as follows: we describe the HCN model in Section 2; Section 3 describes learning and inference in the single layer and multilayer HCNs; Section 4 tests the HCN experimentally and we conclude with a brief discussion in Section 5.

# 2 THE HIERARCHICAL COMPOSITIONAL NETWORK

The HCN model is a discrete latent variable model that generates binary images by composing parts with different levels of abstraction. These parts are shared across all images. Training the model involves learning such parts from data as well as how to combine them to create each concrete image. The HCN model can be expressed as a factor graph consisting only of three types of factors: AND, OR and POOL. These perform the obvious binary operations and will be defined more precisely later in this section. The flexibility of the model allows training in supervised, semisupervised and unsupervised settings, including missing image data. Once trained, the HCN can be used for classification, missing value completion (pixel inference), sparsification, denoising, etc. See Fig. 2 for a factor graph of the complete model. Additional details of each layer type are given in Fig. 4.

At a high level, the HCN consists of a class layer at the top followed by alternating convolutional layers and pooling layers. Inside each layer there is a sparsification, a representation and weights

![](images/86da076224e28ed58cf8e9c0f41b4a9ca71b1888a3aaaf86db86b536d7f97bb2.jpg)  
Figure 2: Factor graph of the HCN model when connected to multiple images  $X_{n}$ . The weights are the only variables that entangle multiple images. The top variables are clamped to 1 and the bottom variables are clamped to  $X_{n}$ . Additional details of each layer type are given in Fig. 4.

(a.k.a. features), each of which is a multidimensional array of latent variables. The class layer selects a category, and within it, which template is going to be used, producing the top-level sparsification. A sparsification is simply an encoding of the representation. A sparsification encodes a representation by specifying which features compose it and where they should be placed. The features are in turn stored in the form of weights. Convolutional layers deterministically combine the sparsification and the weights of a layer to create its representation. Pooling layers randomly perturb the position of the active elements (within a local neighborhood), introducing small variations in the process.

# 2.1 BINARY CONVOLUTIONAL FEATURE LAYER (SINGLE-LAYER HCN)

This layer can perform non-trivial feature learning on its own. We refer to it as a single-layer HCN. See Section 4.1 for the corresponding experiments.

In this case, since there is no additional top-down structure, a binary image is created by placing features at random locations of an image. Wherever two features overlap, they are ORed, i.e., if a pixel of the binary image is activated due to two features, it is simply kept active. We will call  $W$  to the features,  $S$  to the sparsification of the image (locations at which features are placed in that image) and  $X$  to the image. All of these variables are multidimensional binary arrays.

The values of each of the involved arrays for a concrete example with a single-channel image is given in Fig. 3 (to display  $S$  we maximize over  $f$ ). The corresponding diagram is shown in Fig. 4.

In practice, each image  $X$  is possibly multichannel, so it will have size  $\mathrm{F}_X\times \mathrm{H}_X\times \mathrm{W}_X$ , where the first dimension is the number of channels in the image and the other two are its height and width.  $S$  has size  $\mathrm{F}_S\times \mathrm{H}_S\times \mathrm{W}_S$ , where the first dimension is the number of features and the other two are its height and width. We refer to an entry of  $S_{n}$  as  $S_{frc}$ . Setting an entry  $S_{frc} = 1$  corresponds to placing feature  $f$  at position  $(r,c)$  in the final image  $X$ . The features themselves are stored in  $W$ , which has size  $\mathrm{F}_W^{\mathrm{below}}\times \mathrm{F}_W\times \mathrm{H}_W\times \mathrm{W}_W$ , where  $\mathrm{F}_W = \mathrm{F}_S$  and  $F_{W}^{\mathrm{below}} = F_{X}$ . I.e., each feature is a

![](images/95d708cb9ef51a115cc15a5e3157a5b57d28aefcc920ee6ae8fba569c8efd7c5.jpg)  
(a) Image  $X$

![](images/63e6f308e82c58539f8500fda97626f82f20cdfc40f5f2593abc2790d8a33c85.jpg)  
(b) Sparsification  $S$  
Figure 3: Unsupervised analysis of image  $X$  by a standalone convolutional feature layer of HCN.

![](images/aab98917b61752be61c2dbc2ad4bf7bcbf73e0abeb1e9d191125791604eaf5fe.jpg)  
(c) Features  $W$

![](images/3c7eefa8d6aeeb0b59abc028b49d92465a51c9393fd41ef9391ccf9270044d62.jpg)  
(d) Reconstruction  $R$

small 3D array containing one of the building blocks of the image. Those are placed in the positions specified by  $S$ , and the same block can be used many times at different positions, hence calling this layer convolutional<sup>2</sup>.

We can fully specify a probabilistic model for a binary images by adding independent priors over the entries of  $S$  and  $W$  and connecting those to  $X$  through a binary convolution and a noisy channel. The complete model is

$$
p (S) = \prod_ {f r c} p \left(S _ {f r c}\right) = \prod_ {f r c} p _ {S} ^ {S _ {f r c}} \left(1 - p _ {S}\right) ^ {1 - S _ {f r c}}
$$

$$
p (W) = \prod_ {a f r c} p \left(W _ {a f r c}\right) = \prod_ {a f r c} p _ {W} ^ {W _ {a f r c}} \left(1 - p _ {W}\right) ^ {1 - W _ {a f r c}} \tag {1}
$$

$$
p (X | R) = \prod_ {a r c} p _ {\text {n o i s y}} \left(X _ {a r c} \mid R _ {a r c}\right) \text {w i t h} R = \operatorname {b c o n v} (S, W) \text {a n d} p _ {\text {n o i s y}} (1 | 0) = p _ {1 0}, p _ {\text {n o i s y}} (0 | 1) = p _ {0 1},
$$

which depends on four scalar parameters  $p_S, p_W, p_{01}, p_{10}$ , controlling the density of features in the image, of pixels in each feature, and the noise of the channel, respectively. The indexes  $a, f, r, c$  run over channels, features, rows and columns, respectively.

We have used the binary convolution operator  $R = \mathrm{bconv}(S, W)$ . A binary convolution performs the same operation as a normal convolution, but operates on binary inputs and truncates outputs above 1. Our latent variables are arranged as three- and four-dimensional arrays, so we define  $R = \mathrm{bconv}(S, W)$  to mean  $R_{a,\dots} = \min(1, \sum_{f} \mathrm{conv2D}(S_{f,\dots}, W_{a,f,\dots}))$  where  $\mathrm{conv2D}(\cdot, \cdot)$  is the usual 2D convolution operator,  $R$  and  $S$  are binary 3D arrays and  $W$  is a binary 4D arrays. The operator  $\min(1, \cdot)$  truncates values above 1 to 1, performing the ORing of two overlapping features previously mentioned.

The binary convolution (and hence model (1)) can be expressed as a factor graph, as seen in Fig. 4. The AND factor can be written as  $\mathrm{AND}(b|t_1,t_2)$  and takes value 0 when the bottom variable  $b$  is the logical AND of the two top variables  $t_1$  and  $t_2$ . It takes value  $-\infty$  in any other case. The OR factor,  $\mathrm{OR}(b|t_1\ldots ,t_M)$  takes value 0 when the bottom variable  $b$  is the logical OR of the  $M$  top variables  $t_1\dots ,t_M$ . It takes value  $-\infty$  in any other case.

When this layer is not used in standalone mode, but inside a multilayer HCN, the variables  $R$  are connected to the pooling layer immediately below (instead of being connected to the image  $X$  through the noisy channel) and the variables  $S$  are connected to the pooling layer immediately above (instead of being connected to the prior).

# 2.2 THE CLASS LAYER

We assume for now that a single class is present in each image. We can then write

$$
\log p \left(c _ {1}, \dots , c _ {K}\right) = \operatorname {P O O L} \left(c _ {1}, \dots , c _ {K} \mid 1\right)
$$

where  $c_{k}$  are mutually exclusive binary variables representing which of the  $K$  categories is present.

In general, we define  $\mathrm{POOL}(b_1,\ldots ,b_M|t = 1) = -\log M$  when exactly one of the bottom variables  $b_{1},\dots,b_{m}$  takes value 1 (we say that the pool is active), and  $\mathrm{POOL}(b_1,\dots,b_M|t = 0) = 0$  when  $b_{m} = 0\forall_{m}$  (the pool is off). It takes value  $-\infty$  in any other case.

![](images/f43aa756f1c70d9ddce8c865c5d642c21f861c7bce1487f56f8a8350f9f956c0.jpg)  
(a) Binary convolution

![](images/88b1eb79885476481408058db8dce389bc3712da859aa9ebef24b7af59d564ef.jpg)  
(b) Feature layer  
Figure 4: Diagrams of binary convolution and factor graph connectivity for 1D image.

![](images/772c8b36136ca2cd5625e81b06085972082d4b76fc87188dc1f825d4f92eabd6.jpg)  
(c) Pooling layer

Within each category, we might have multiple templates. Each template corresponds to a different visual expression of the same conceptual category. For instance, if one category is furniture, we could have a template for chair and another template for table. Each category has binary variables representing each of the  $J$  templates,  $s_{jk}$  with  $j \in [1 \dots J]$ . If a category is active, exactly one of its templates will be active. The joint probability of the templates is then

$$
\log p \left(S ^ {L} \mid c _ {1}, \dots , c _ {K}\right) = \sum_ {k} \log p \left(s _ {1 k}, \dots , s _ {J k} \mid c _ {k}\right) = \sum_ {k} \operatorname {P O O L} \left(s _ {1 k}, \dots , s _ {J k} \mid c _ {k}\right)
$$

where these  $JK$  variables are arranged as a 3D array of size  $1 \times 1 \times JK$  called  $S^L$  which forms the top-level sparsification of the template. A sample from  $S^L$  will always have exactly one element set to 1 and the rest set to 0. Superscript  $L$  is used to identify the layer to which a variable belongs. Since there are  $L$  layers,  $S^L$  is the top layer sparsification.

# 2.3 THE POOLING LAYER

In a multilayer HCN, feature layers and pooling layers appear in pairs. Inside layer  $\ell$ , the pooling layer  $\ell$  is placed below the feature layer  $\ell$ .

Since the convolutional feature layer is deterministic, any variation in the generated image must come from the pooling layers (and the final noisy channel). Each pooling layer shifts the position of the active units in  $R^{\ell}$  to produce the sparsification  $S^{\ell - 1}$  in the layer below. This shifting is local, constrained to a region of size $^3$ $\mathrm{H}_P \times \mathrm{W}_P \times 1$ , the pooling window. When two or more active units in  $R^{\ell}$  are shifted towards the same position in  $S^{\ell - 1}$ , they result in a single activation, so the number of active units in  $S^{\ell - 1}$  is equal or smaller than the number of activations in  $R^{\ell}$ .

The above description should be enough to know how to sample  $S^{\ell - 1}$  from  $R^\ell$ , but to provide a rigorous probabilistic description, we need to introduce the intermediate binary variables  $U_{\Delta r, \Delta c, f, r, c}$ , which are associated to a shift  $\Delta r, \Delta c$  of the element  $R_{frc}^\ell$ . The  $\mathrm{H}_P\mathrm{W}_P$  intermediate variables associated to the same element  $R_{frc}^\ell$  are noted as  $U_{\therefore , frc}^\ell$ . Since an element can be shifted to a single position per realization and only when it is active, the elements in  $U_{\therefore , frc}^\ell$  are grouped into a pool

$$
\log p \left(U ^ {\ell} \mid R ^ {\ell}\right) = \sum_ {f r c} \log p \left(U _ {:,: f r c} ^ {\ell} \mid R _ {f r c} ^ {\ell}\right) = \sum_ {f r c} \operatorname {P O O L} \left(U _ {:,: f r c} ^ {\ell} \mid R _ {f r c} ^ {\ell}\right)
$$

and then  $S^{\ell - 1}$  can be obtained deterministically from  $U^\ell$  by ORing the  $\mathrm{H}_P\mathrm{W}_P$  variables of  $U$  that can potentially turn it on,  $\log p(S^{\ell - 1}|U^\ell) = \sum_{fr'c'} \log p(S_{fr'c'}^{\ell - 1}|U^\ell) = \sum_{fr'c'} \mathrm{OR}(S_{fr'c'}^{\ell - 1}|\{U_{\Delta r,\Delta c,f,r,c}\}_{r':r + \Delta r,c':c + \Delta c})$ . i.e., the above expression evaluates to 0 if the above OR relations are satisfied and to  $-\infty$  if they are not.

# 2.4 JOINT PROBABILITY WITH MULTIPLE IMAGES

The observed binary image  $X$  corresponds to the bottommost sparsification<sup>4</sup>  $S^0$  after it has traversed, element by element, a noisy channel with bit flip probabilities  $p(X_{frc} = 1|S_{frc}^0 = 0) = p_{10} < 0.5$  and  $p(X_{frc} = 0|S_{frc}^0 = 1) = p_{01} < 0.5$ . This defines  $p(X|S^0)$ .

Finally, if we consider the weight variables to be independent Bernoulli variables with a fixed per-layer sparse prior  $p_W^\ell$  that are drawn once and shared for the generation of all images, we can write the joint probability of multiple images, latent variables and weights as

$$
\begin{array}{l} \log p (\{X _ {n}, H _ {n}, C _ {n} \} _ {n = 1} ^ {N}, \{W ^ {\ell} \} _ {\ell = 1} ^ {L}) = \sum_ {\ell = 1} ^ {L} \log p (W ^ {\ell}) + \sum_ {n = 1} ^ {N} \log p (X _ {n} | S _ {n} ^ {0}) + \log p (S _ {n} ^ {L} | C _ {n}) + \log p (C _ {n}) \\ + \sum_ {n = 1} ^ {N} \sum_ {\ell = 1} ^ {L} \log p (S _ {n} ^ {\ell - 1} | U _ {n} ^ {\ell}) + \log p (U _ {n} ^ {\ell} | R _ {n} ^ {\ell}) + \log p (R _ {n} ^ {\ell} | S _ {n} ^ {\ell}, W ^ {\ell}) \\ \end{array}
$$

where we have collected all the category variables  $\{c_k\}$  of each image in  $C_n$  and the remaining latent variables in  $H_{n}$  and for convenience. Each image uses its own copy of the latent variables, but the weights are shared across all images, which is the only coupling between the latent variables.

The above expression shows how, in addition to factorizing over observations (conditionally on the weights), there is a factorization across layers. Furthermore, the previous description of each of these layers implies that the entire model can be further reduced to small factors of type AND, OR and POOL, involving only a few local variables each.

Since we are interested in a point estimate of the features, given the images  $\{X_{n}\}_{n = 1}^{N}$  and a (possibly empty)  $^5$  subset of the labels  $\{C_n\}_{n = 1}^N$ , we will attempt to recover the maximum a posteriori6 (MAP) configuration over features, sparsifications, and unknown labels. Note that for classification, selecting  $\{W^{\ell}\}_{\ell = 1}^{L}$  by maximizing the joint probability is very different from selecting it by maximizing a discriminative loss of the type log  $p(\{C_n\}_{n = 1}^N |\{X_n\}_{n = 1}^N,\{W^\ell \}_{\ell = 1}^L)$ , since in this case, all the prior information  $p(X)$  about the structure of the images is lost. This results in more samples being required to achieve the same performance, and less invariance to new test data.

Once learning is complete, we can fix  $\{W^{\ell}\}_{\ell = 1}^{L}$ , thus decoupling the model for every image, and use approximate MAP inference to classify new test images, or to complete them if they include missing data (while benefiting from the class label if it is available).

Even though we only consider the single-class-per-image setting, the compositional property of this model means that we can train it on single-class images and then, without retraining, change the class layer to make it generate (and therefore, recognize) combinations of classes in the same image.

# 3 LEARNING AND INFERENCE

We will consider first the simpler case of a single-layer HCN, as described in Section 2.1. Then we will tackle inference in the multilayer HCN.

# 3.1 LEARNING IN SINGLE-LAYER HCN

In this case, for model (1), we want to find

$$
S ^ {*}, W ^ {*} = \arg \max  _ {S, W} p (X | S, W) p (S) p (W). \tag {2}
$$

This is a challenging problem even in simple cases. In fact, it can be easily shown that boolean matrix factorization (BMF), a.k.a. boolean factor analysis, arises as a particular case of (2) in which the

heights and widths of all the involved arrays are set to one. BMF is a decades-old problem proved to be NP-complete in (Stockmeyer, 1975) and with applications in machine learning, communications and combinatorial optimization. Another related problem is non-negative matrix factorization (NMF) (Lee & Seung, 1999), but NMF is additive instead of ORing the contributions of multiple features, which is not desired here.

One of the best-known heuristics to address BMF is the Asso (Miettinen et al., 2006). Unfortunately, it is not clear how to extend it to solve (2) because it relies on assumptions that no longer hold in the present case. The variational bound of (Jaakkola & Jordan, 1999) addresses inference in the presence of a noisy-OR gate and was successfully used in by (Singliar & Hauskrecht, 2006) to obtain the noisy-OR component analysis (NOCA) algorithm. NOCA addresses a very similar problem to (2), the two differences being that a) the weight values are continuous between 0 and 1 (instead of binary) and b) there is no convolutional weight sharing among the features. NOCA can be modified to include the convolutional weight sharing, but it is not an entirely satisfactory solution to the feature learning problem as we will show. We observed that the obtained local maxima, even after significant tweaking of parameters and learning schedule, are poor for problems of small to moderate size.

We are not aware of other existing algorithms that can solve (2) for medium image sizes. The model (1) is directly amenable to mean-field inference without requiring the additional lower-bounding used in NOCA, but we experimented with several optimization strategies (both based in mean field updates and gradient-based) and the obtained local maxima were consistently worse than those of NOCA.

In (Ravanbakhsh et al., 2015) it is shown that max-product message passing (MPMP) produces state-of-the-art results for the BMF problem, improving even on the performance of the Asso heuristic. We also address problem (2) using MPMP. Even though MPMP is not guaranteed to converge, we found that with the right schedule, even with very slight or no damping, good solutions are found consistently.

Model (1) can be expressed both as a directed Bayesian network or as a factor graph using only AND and OR factors, each involving a small number of local binary variables. Finding features and sparsifications can be cast as MAP inference<sup>7</sup> in this factor graph.

MPMP is a local message passing technique to perform MAP inference in factor graphs. MPMP is exact on factor graphs without loops (trees). In loopy models, such as (1), it is an approximation with no convergence guarantees<sup>8</sup>, although convergence can be often attained by using some damping  $0 < \alpha \leq 1$ . See Appendix A for a quick review on MPMP and Appendix B for the message update equations required for the factors used in this work. Unlike Ravanbakhsh et al. (2015) which uses parallel updates and damping, we update each AND-OR factor<sup>9</sup> in turn, following a random in a sequential schedule. This results in faster convergence with less or no damping.

# 3.2 LEARNING IN MULTILAYER HCN (UNSUPERVISED, SEMISUPERVISED, SUPERVISED)

Despite its loopiness, we can also apply MPMP inference to the full, multilayer model and obtain good results. The learning procedure iterates forward and backward passes (a precise description can be found in Appendix C, Algorithm 1). In a forward pass, we proceed updating the bottom-up messages to variables, starting from the bottom of the hierarchy (closer to the image) and going up to the class layer. In a backward pass, we update the top-down messages visiting the variables in top-down order. Messages to the weight variables are updated only in the forward pass. We use damping only in the update of the bottom-up messages from a pooling layer during the forward pass. The AND-OR factors in the binary convolutional layer form trees, so we treat each of these trees as a single factor, since closed form message updates for them can be obtained. Those factors are updated once in random order inside each layer, i.e., sequentially. The pools at the class layer also from a tree, so we also treat them as a single factor. The message updates for AND, OR and POOL factors follow trivially from their definition and are provided in Appendix B.

After enough iterations, weights are set to 1 if their max-marginal difference is positive and to 0 otherwise. This hard assignment converts some of the AND factors into a pass-through and the rest in disconnections. Thus the weight assignments define the connectivity between  $S^{\ell}$  and  $R^{\ell}$  on a new graph without ANDs. This is the learned model, that we can use to perform inference with with on new test images.

# 3.3 INFERENCE IN MULTILAYER HCN

Typical inference tasks are classification and missing value imputation. For classification, we find that a single forward pass seems good enough and further forward and backward passes are not needed. For missing value imputation a single forward and backward pass is enough. These forward and backward passes follow the general pattern described in Algorithm 1 (Appendix C) except for step 5) in the backward pass. In order to achieve higher quality explaining-away $^{10}$  with a single backward pass, we replace step 5) with multiple alternating executions of steps 5) and 2).

Interestingly, the functional form of the forward pass of an HCN is the same as that of a standard CNN, see Appendix D, and therefore, an actual CNN can be used to perform a fast forward pass.

# 4 EXPERIMENTS

In the following, we experimentally characterize both the single-layer and multilayer HCN.

# 4.1 SINGLE-LAYER HCN

We create several synthetic (both noisy and noiseless) images in which the building blocks –or features– are obvious to a human observer and check the ability of HCN to recover the them. The task is deceptively simple, and the existing the state of the art at this task, NOCA, is unable to solve several of our examples. Since the number of free parameters of the model is so small (3 in the case of a symmetric noisy channel), these can be easily explored using grid search and selected using maximum likelihood. The sensitivity of the results to these parameters is small.

HCN only requires straightforward MPMP with random order over the factors. For NOCA, initializing the variational posterior over the latent sources and choosing how to interleave the updates of this posterior with the update of the additional variational parameters (Singliar & Hauskrecht, 2006) is tricky. For best results, during each E step we repeated the following 10 times: update the variational parameters for 20 iterations and then update the variational posterior (which is a single closed form update). The M update also required an inner loop of variational parameter updating.

The performance of HCN and NOCA can be assessed visually in Fig. 5. Column (a) shows each input image (these are single-image datasets) and the remaining columns show the features and reconstructions obtained by HCN and NOCA. In some of the input images we have added noise that flips pixels with  $3\%$  probability. For HCN (respectively NOCA), we binarize all the beliefs (respectively, variational posteriors) from the  $[0,1]$  range by thresholding at 0.5 and then perform a binary convolution to obtain the reconstruction. Because noise is not included in this reconstruction, a cleaner image may be obtained, resulting in unsupervised denoising (rows 1 and 4 of Fig. 5).

For a quantitative comparison, refer to Tab. 1. One algorithm-independent way to measure performance in the feature learning problem is to measure compression. It is known that to transmit a long sequence of  $N$  bits which are 1 with probability  $p$ , we only need to transmit  $NH(p)$  bits with an optimal encoding, where  $H$  is the entropy. Thus sparse sequences compress well. In order to transmit these images without loss, we need to transmit either one sequence of bits (encoding the image itself) or three sequences of bits, one encoding the features, another encoding the sparsification and a last one encoding the errors between the reconstruction and the original image. Ideally, the second method is more efficient, because the features are only sent once and the sparsification and errors sequences are much sparser than the original image. The ratio between the two is shown together with running time on a single CPU. Unused features are discarded prior to computing compression.

![](images/5ef63540ebde75db93bbf8bf8647f0b0daf254ab5b38a2a2180c8d411611f40c.jpg)

![](images/e23ded2b1a70b899cf22553b0a8fae774dc17afa05278255ef48b6f8db407dd0.jpg)

![](images/f8062c8361ce12fc716fca948c8a4ce82c0d4bd943a4166f4233d3ee94ed164d.jpg)

![](images/6897d5a3202ec2f521d8b9a2fc4ef793b00b1d03a78614424a93e5826bd504d1.jpg)

![](images/6fa0bb16cf897c1154a823525f48769051ef8850ded0eeb5c298332734a565f8.jpg)

![](images/eab2f4dfb4ebfffeb4da169a12bb644a430cfac2fe5168160ef4fab8f2e22d4c.jpg)  
(a) Input image  $X$

![](images/3b29d9d2f6f6d8f6015ea6a3449bad03f876f0b6d7536be3fe029bc092134c43.jpg)

![](images/fbdebdc96fbd87e297cc92d5534dadaa4fdf65576865e628995c85dc59447377.jpg)

![](images/7b42a147e560672c7b76293c0619ee3faaf881a0b0e72d61c43e19ffd6f96137.jpg)

![](images/18a9a9234852f2b89f7badc3772bef27b429d8f05d30eca2aae42146ae1ce032.jpg)

![](images/36e181e548e059638a8428dd4ae54a15155ae488ab8e1e3bcedf4e5acb313ee9.jpg)  
(b) HCN  $W$  
(c) HCN  $R$

![](images/bd25c447b2210a48a88bf828e0404df621f8184545c9f3df3c126787728f62fb.jpg)

![](images/332d991525e96b646034ae23919f42072744a8ce349ca808802bc9e2dd107aa1.jpg)

![](images/64c848463ca90efb0830cb7e9e7ae891356a06ed5e964dcd2aa589444e825646.jpg)

![](images/2e83b0ac5af4cc3b5e11b8c0727867c9616865eb279b9875b8d3741745211698.jpg)

![](images/9d9512603e01719907cd77a22c286f4a187a8e000fef79fb0d76282e643ce1ae.jpg)  
(d) NOCA  $W^{\prime}$  
Figure 5: Features extracted by HCN and NOCA and image reconstructions for several datasets. Best viewed on screen with zoom.

![](images/bb2b41b7cddee55ce034dbabc5d522cbc25934057f6548716551fd015a02fbee.jpg)

![](images/e0256114871417f8fc13e2ffdcc88d765758d3705664b485f17bdea1b8ef9e8a.jpg)

![](images/09a6f188b87bb1fada02b5d41375fdd09c94c6a0b531eb99d869f7b270c3e969.jpg)

![](images/7a0215d0514b02fbe162c0046b227b0b5c4035823b384147bd8cc844fffb0a0b.jpg)

![](images/24027a59851baaf277bb5e6558fa69ed4f462565db57e9d8db2dce28bf990b9b.jpg)  
(e) NOCA  $R$

![](images/2c8f67dbbdcd9e95a40e3e1761ac068bbe5cee81001995db3761702331113fea.jpg)  
(a) Image  $X_{1}$  
Figure 6: Online learning. (a) and (b) show two sample input images; (c) and (d) show the features learned by batch and online HCN using 30 input images and 100 epochs; (e) shows the features learned by online HCN using 3000 input images and 1 epoch.

![](images/bc5066c4f74ad903b2b943cbdddc6d351e8aad637fb5ae1d19475d2f11b322b9.jpg)  
(b) Image  $X_{2}$

![](images/fbb00e6112517e03070535096f4104aaf5dcb55ecec9432bd9b9fcd10c1011cf.jpg)  
(c) Batch HCN  $W$

![](images/e992eebba68f183e8556f649503f1a9bc214ad567cbc790682197c86dfa431e4.jpg)  
(d) Online HCN  $W$

![](images/b1297b06e7cc7760dac496c63718df460e9e26ea2abd9febb092c7f444ceb09c.jpg)  
(e) Online HCN  $W$

<table><tr><td></td><td colspan="2">Two bars</td><td colspan="2">Symbols</td><td colspan="2">Clean letters</td><td colspan="2">Noisy letters</td><td colspan="2">Text</td></tr><tr><td></td><td>comp.</td><td>time</td><td>comp.</td><td>time</td><td>comp.</td><td>time</td><td>comp.</td><td>time</td><td>comp.</td><td>time</td></tr><tr><td>NOCA</td><td>84%</td><td>0.67 m</td><td>85%</td><td>92 m</td><td>98%</td><td>662 m</td><td>102%</td><td>716 m</td><td>84%</td><td>1222 m</td></tr><tr><td>HCN</td><td>83%</td><td>0.07 m</td><td>11%</td><td>0.42 m</td><td>38%</td><td>25 m</td><td>73%</td><td>24 m</td><td>28%</td><td>31 m</td></tr></table>

Table 1: Comp.:  $E(X) / (E(S) + E(W) + E(X - R))$ , where  $E$  is the encoding cost. Time: minutes.

# 4.2 ONLINE LEARNING

The above experiments use a batch formulation, i.e., consider simultaneously all the available training data  $\{X_{n}\}_{1}^{N}$ . Since the amount of memory required to store the messages for MPMP scales linearly with the training data, this imposes a practical limit in the number of images that can be processed. In order to overcome this limit, we also consider a particular message update schedule in which the messages outgoing from factors connected to each image and sparsification  $X_{n}, S_{n}$  are updated only once and therefore, after an image has been processed, can be discarded, since they are never reused. This effectively allows for online processing of images without memory scaling issues. Two modifications are needed in practice for this to work well: first, instead of processing only one image at a time, better results are obtained if the factors of multiple images (forming a minibatch) are processed in random order. Second, a forgetting mechanism must be introduced to avoid accumulating an unbounded amount of evidence from the processed minibatches.

In detail, the beliefs of the variables  $W$  are initialized uniformly at random in the interval  $(0.9p_{W}, p_{W})$  (we call these initial beliefs  $b_{\mathrm{prior}}^{(0)}(W_{a_{frc}})$ ) and the beliefs of the variables  $\{S_{n}\}_{1}^{N}$  are initialized to  $p_{S}$ . The initial outgoing messages from all the AND-OR factors are set to 0. Since each factor is only processed once, this allows implementing MPMP without ever having to store messages and only requiring to store beliefs. After processing the first minibatch using MPMP (with no damping), we call the resulting belief over each of the weights  $b_{\mathrm{post}}^{(0)}(W_{a_{frc}})$  (as it standard for MPMP of binary variables, beliefs are represented using max-marginal differences in log space). Instead of processing the second minibatch using  $b_{\mathrm{post}}^{(0)}(W_{a_{frc}})$  as the initial belief, we use  $b_{\mathrm{prior}}^{(1)}(W_{a_{frc}}) = \lambda b_{\mathrm{post}}^{(0)}(W_{a_{frc}}) + (1 - \lambda)b_{\mathrm{prior}}^{(0)}(W_{a_{frc}})$ , i.e., we "forget" part of the observed evidence, substituting it with the prior. This introduces an exponential weighing in the contribution of each minibatch. The forgetting factor is  $\lambda \in (0, 1]$  specifies the amount of forgetting. When  $\lambda = 1$  this reduces to normal MPMP (no forgetting), when  $\lambda = 0$ , we completely forget the previous minibatch and process the new one from scratch.

Fig. 6 illustrates online learning. HCN is shown 30 small images containing 5 randomly chosen and randomly placed characters with  $3\%$  flipping noise (see Fig. 5.(a) and (b) for two examples). They are learned in different manners. Fig. 5.(c): as a single batch with damping  $\alpha = 0.8$  and using 100 epochs (each factor is updated 100 times); Fig. 6.(d): with minibatches of 5 images, no damping,  $\lambda = 0.95$  and using 100 epochs; Fig. 6.(e): with minibatches of 5 images, no damping,  $\lambda = 0.95$ , using a single epoch, but using 3000 images, so that running time is the same.

# 4.3 MULTI-LAYER HCN: SYNTHETIC DATA

We create a dataset by combining two traits: a) either a square (with four holes) or a circle and b) either a forward or a backward diagonal line. This results in four patterns, which we group in two categories, see Fig. 7.(a). Categories are chosen such that we cannot decide the label of an image based only on one of the traits. The position of the traits is jittered within a  $3 \times 3$  window, and after combining them, the position of the individual pixels is also jittered by the same amount. Finally, each pixel is flipped with probability  $10^{-3}$ . This sampling procedure corresponds to a 2-layer HCN sampling for some parameterization. We generate 100 training samples and 10000 test samples.

# 4.3.1 UNSUPERVISED LEARNING

We train the HCN as described in Section A on the 100 training data samples, not using any label information. We do set the architecture of the network to match the architecture generating the data. There are four hyperparameters in this model,  $p_{01}, p_{10}, p_W^1, p_W^2$ . Their selection is not critical. We

will choose them to match the generation process. MAP inference does discover and disentangle almost perfectly the compositional parts at the first and second layers of the hierarchy, see Figs. 7.(b) and 8.(a). In 8.(a), rows correspond with templates and columns correspond to each of the features of the first layer. We can see that the model has "understood" the data and can be used to generate more samples from it. Performing inference on this model is very challenging. We are not aware of any previous method that can learn the features of this simple dataset with so few samples. In other experiments we verified that, using local message passing as opposed to gradient descent was critical to successfully minimize our objective function. Results with the quality of Figs. 7.(b) and 8.(a) were obtained in every run of the algorithm. Running time is  $7\mathrm{min}$  on a single CPU.

We can now clamp the discovered weights on both layers and use the fast forward pass to classify each training image as belonging to one of the four discovered templates (i.e., cluster them). We can even classify the test images as belonging to one of the four templates. When doing this, all the images in the training set get assigned to the right template and only 60 out of 10000 images in the test set do not get classified in the right cluster. This means that if we had just 4 labeled images, one from each cluster, we could perform 4-class minimally-supervised classification with just  $0.6\%$  error.

Finally, we run a single forward-backward pass of the inference algorithm on a test image with missing pixels. We show the inferred missing pixels in Fig. 7.(c). See also footnote 10.

# 4.3.2 SUPERVISED LEARNING

Now we retrain the model using label information. This results in the same weights being found, but this time the templates are properly grouped in two classes, as shown in Fig. 8.(a). Classification error on the test set is very low,  $0.07\%$ . We now compare the HCN classification performance with that of a CNN with the same functional form but trained discriminatively and with a standard CNN with ReLU activations, a densely connected layer and softmax activation. We minimize the crossentropy loss function with  $L_{2}$  regularization on the weights. The test errors are respectively  $0.5\%$  and  $2.5\%$ , much larger than those of HCN. We then consider versions of our training set with different levels of pixel-flipping noise. The evolution of the test error is shown in Fig. 8.(c). For the competing methods we needed many random restarts to obtain good results. Their regularization parameter was chosen based on the test set performance.

# 4.4 MULTI-LAYER HCN: MNIST DATA

We turn now to a problem with real data, the MNIST database (LeCun et al., 1998), which contains 60000/10000 training/testing images of size  $28 \times 28$ . We want to generalize from very few samples, so we only use the first 40 digits of each category to train. We pre-process each image with a fixed set of 16 oriented filters, so that the inputs are a 16-channel image. We use a 2-layer HCN with 32 templates per class and 64 lower level features of size  $26 \times 26$  and two layers of  $3 \times 3$  pooling,  $p_W^1 = 0.001$ ,  $p_W^2 = 0.05$ . These values are set a priori, not optimized. Then we test on both the regular MNIST training set and different corrupted versions<sup>11</sup> of it (same preprocessing

![](images/a06e3bb516c0017b97dd00d31039342d67f10ec3a1222cef43e1bd27fa9efb02.jpg)  
(a) 16 training samples and labels

![](images/b372ee08e47dd3134145dd6dd21a87a4fc775ba84d7d7a391cd84382664ad74a.jpg)  
(b)  $W^1$  , no supervision

![](images/c763dda3ca58786504cce89414ed6d4923510065f62dddaa40954acf5f18891b.jpg)  
(c) Missing value imputation  
Figure 7: Samples from synthetic data and results from unsupervised learning tasks.

![](images/6b593a543d32f2ebdb2578248b7dbaab3ff27fc532c76a0917afc5a4987a444d.jpg)  
(a) Supervised, unsupervised (top, bottom)  $W^2$

![](images/63596e7549974cccde81baaa1fb7762e1d750b7558ba11838d9a379d1919075c.jpg)  
(b)  $W^1$ , discriminative training

![](images/dd13e17938a042614df8b058b4c78165b36faea9f91867228b743c53e756ec48.jpg)  
(c) Effect of increased noise level

![](images/b94beeb7ea0572840be08940e75434c7fffa1cd92708fbb5d853a03e99151e69.jpg)  
Figure 8: Discriminative vs. generative training and supervised vs. unsupervised generative training.  
(a) Learned  $W^{1}$  by HCN  
Figure 9: First layer of weights learned by HCN and CNN on the preprocessed MNIST dataset.

![](images/17604d6bf69b7a1c0c38d18b1ce9d54263fbdc2af9e7e268e3715c897b6efdaf.jpg)  
(b) Learned  $W^2$  by HCN

<table><tr><td>Corruption</td><td>HCN</td><td>CNN</td></tr><tr><td>None</td><td>11.15%</td><td>9.53%</td></tr><tr><td>Noise</td><td>20.69%</td><td>39.28%</td></tr><tr><td>Border</td><td>16.97%</td><td>17.78%</td></tr><tr><td>Patches</td><td>14.52%</td><td>16.27%</td></tr><tr><td>Grid</td><td>68.52%</td><td>82.69%</td></tr><tr><td>Line clutter</td><td>37.22%</td><td>55.77%</td></tr><tr><td>Deletion</td><td>22.03%</td><td>25.05%</td></tr></table>

(c) Test error with different corruptions

and no retraining). We follow the same preprocessing and procedure using a regular CNN with discriminative training and explore different regularizations, architectures and activation types, only fixing the pooling sizes and number of layers to match the HCN. We select the parameterization that minimizes the error on the clean test set. This CNN uses 96 low level features. Results for all test sets are reported on Fig. 9.(c). It can be seen that HCN generalizes better. The weights of the first layer of the HCN after training are shown in Fig. 9.(a). Notice how HCN is able to discover reusable parts of digits.

The training time of HCN scales exactly as that of a CNN. It is linear in each of its architectural parameters: Number of images, number of pixels per image, features at each layer, size of those features, etc. However, the forward and backward passes of an HCN are more complex and optimized code for them is not readily available as it is for a CNN, so a significant constant factor separates the running times of both. Training time for MNIST is around 17 hours on a single CPU. The RAM required to store all the messages for 400 training images in MNIST goes up to around 150GB. To scale to bigger training sets, an online extension (see Section 4.2) needs to be used.

# 5 CONCLUSIONS AND FUTURE WORK

We have described the HCN, a hierarchical feature model with a rich prior and provided a novel method to solve the challenging learning problem it poses. The model effectively learns convolutional features and is interpretable and flexible. The learned weights are binary, which is advantageous for storage and computation purposes (Courbariaux et al., 2015; Han et al., 2015). Future work entails adding more structure to the prior, leveraging more refined MAP inference techniques, exploring other update schedules and further exploiting the generalization-without-retraining capabilities of this model.

# REFERENCES

Matthew James Beal. Variational algorithms for approximate Bayesian inference. University of London London, 2003.  
Matthieu Courbariaux, Yoshua Bengio, and Jean-Pierre David. Binaryconnect: Training deep neural networks with binary weights during propagations. In Advances in Neural Information Processing Systems, pp. 3105-3113, 2015.  
Amir Globerson and Tommi S Jaakkola. Fixing max-product: Convergent message passing algorithms for MAP LP-relaxations. In Advances in Neural Information Processing Systems, pp. 553-560, 2008.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pp. 2672-2680, 2014.  
Ian J Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, and Yoshua Bengio. Maxout networks. arXiv preprint arXiv:1302.4389, 2013.  
Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. arXiv preprint arXiv:1510.00149, 2015.  
Tom Heskes. Stable fixed points of loopy belief propagation are local minima of the bethe free energy. In Advances in neural information processing systems, pp. 343-350, 2002.  
Geoffrey E Hinton, Simon Osindero, and Yee-Whye Teh. A fast learning algorithm for deep belief nets. Neural computation, 18(7):1527-1554, 2006.  
Tommi S Jaakkola and Michael I Jordan. Variational probabilistic inference and the qmr-dt network. Journal of artificial intelligence research, 10:291-322, 1999.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Daphne Koller and Nir Friedman. Probabilistic graphical models: principles and techniques. MIT press, 2009.  
Vladimir Kolmogorov. Convergent tree-reweighted message passing for energy minimization. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 28(10):1568-1583, 2006.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Daniel D Lee and H Sebastian Seung. Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755):788-791, 1999.  
Talya Meltzer, Amir Globerson, and Yair Weiss. Convergent message passing algorithms - a unifying view. In Jeff A. Bilmes and Andrew Y. Ng (eds.), UAI, pp. 393-401, 2009.  
Pauli Miettinen, Taneli Mielikainen, Aristides Gionis, Gautam Das, and Heikki Mannila. The discrete basis problem. In European Conference on Principles of Data Mining and Knowledge Discovery, pp. 335-346. Springer, 2006.  
Tom Minka et al. Divergence measures and message passing. Technical report, 2005.  
Andriy Mnih and Karol Gregor. Neural variational inference and learning in belief networks. arXiv preprint arXiv:1402.0030, 2014.  
Judea Pearl. Probabilistic reasoning in intelligent systems: networks of plausible inference. 1988.  
Siamak Ravanbakhsh, Barnabás Póczos, and Russell Greiner. Boolean matrix factorization and noisy completion via message passing. 2015.  
Ruslan Salakhutdinov and Geoffrey E Hinton. Deep boltzmann machines. In AISTATS, volume 1, pp. 3, 2009.

Shimony. Finding MAPs for belief networks is NP-hard. AIJ: Artificial Intelligence, 68, 1994.  
Tomáš Singliar and Miloš Hauskrecht. Noisy-or component analysis and its application to link analysis. Journal of Machine Learning Research, 7(Oct):2189-2213, 2006.  
Larry J. Stockmeyer. The set basis problem is NP-complete. IBM Thomas J. Watson Research Division, 1975.  
Huayan Wang and Koller Daphne. Subproblem-tree calibration: A unified approach to max-product message passing. In Proceedings of the 30th International Conference on Machine Learning (ICML-13), pp. 190–198, 2013.  
Tomás Werner. A linear programming approach to max-sum problem: A review. IEEE Trans. Pattern Analysis and Machine Intelligence, 29(7):1165-1179, July 2007.
