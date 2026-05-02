# ENERGY-BASED SPHERICAL SPARSE CODING

Bailey Kong and Charless C. Fowlkes

Department of Computer Science

University of California, Irvine

Irvine, CA 92697 USA

{bhkong,fowlkes}@ics.uci.edu

# ABSTRACT

In this paper, we explore an efficient variant of convolutional sparse coding with unit norm code vectors and reconstructions are evaluated using an inner product (cosine distance). To use these codes for discriminative classification, we describe a model we term Energy-Based Spherical Sparse Coding (EB-SSC) in which the hypothesized class label introduces a learned linear bias into the coding step. We evaluate and visualize performance of stacking this encoder to make a deep layered model for image classification.

# 1 INTRODUCTION

Sparse coding has been widely studied as a representation for images, audio and other vectorial data. This has been a highly successful method that has found its way into many applications, from signal compression and denoising (Donoho, 2006; Elad & Aharon, 2006) to image classification (Wright et al., 2009), and as a model for understanding neuronal receptive fields in visual cortex (Olshausen & Field, 1997). Since its introduction, subsequent works have brought sparse coding into the supervised learning setting—introducing classification terms to the original formulation to encode features that are not only discriminative, but are still able to reconstruct the original signal (Jiang et al., 2011; Yang et al., 2010; Zeiler et al., 2010; Ji et al., 2011; Zhou et al., 2012; Zhang et al., 2013).

While supervised sparse coding methods have shown to find more discriminative features leading to improved classification performance over their unsupervised counterparts, they have received much less attention in recent years and have been eclipsed by simpler feed-forward architectures.

There are two reasons for this: (1) Sparse coding is computationally expensive. Sparse coding is traditionally formulated as a least-squares (LSQ) minimization with a convex sparse inducing regularizer. Because there is no closed-form solution to this formulation, iterative optimization is necessary to find a solution (Zeiler et al., 2010; Bristow et al., 2013; Yang et al., 2013; Heide et al., 2015). This computational overhead becomes quite significant when training discriminative models due to the demand of processing many training examples necessary for good performance, and so sparse coding has fallen out of favor by not being able to keep up with simpler non-iterative coding methods.

(2) Euclidean distance is not a good metric for many features in computer vision applications (Yan et al., 2007; Wu & Rehg, 2009; Choi et al., 2014). This problem is frequently attributed to that of the curse of dimensionality, where the proportional difference between the distance of the furthest-points and the closest-points vanish as the dimensionality increases (Beyer et al., 1999). Contrary to popular thinking, Houle et al. (2010) suggest the actual problem to be that of irrelevant features. Since Euclidean distance weighs all dimensions equally, irrelevant features are emphasized just as much as relevant features are.

In this paper we show these issues can be addressed by changing distance functions. We introduce a novel supervised sparse coding formulation using cosine distance called energy-based spherical sparse coding (EB-SSC). This bi-directional coding method incorporates both top-down and bottom-up information using an energy-based model where encoding the features depends on both a hypothesized class label and a input signal. Like Cao et al. (2015) our motivation for bi-directional coding comes from the "Biased Competition Theory" from cognitive science, which suggests that

visual processing can be biased by other mental processes (e.g., bottom-up and top-down systems) to prioritize certain features that are most relevant to current task. A key advantage of our formulation is that we can perform coding in a feed-forward manner without an iterative algorithm. Fig. 1 illustrates the flow of computation used by our SSC and EB-SSC building blocks compared to a standard feed-forward layer.

Our energy based approach for combining top-down and bottom-up information is closely tied to ideas of Larochelle & Bengio (2008); Ji et al. (2011); Zhang et al. (2013); Li & Guo (2014)—although the model details are substantially different (e.g., Ji et al. (2011) and Zhang et al. (2013) use sigmoid non-linearities while Li & Guo (2014) use separate representations for top-down and bottom-up information). The energy function of Larochelle & Bengio (2008) is also similar but includes an extra classification term and is trained as a restricted Boltzmann machine.

![](images/e1440026fd559936264f5c0f731eb757447dc6a80de2533d46d5796cb2026888.jpg)  
(a) CReLU

![](images/ee36605aa19516c3d9444ffb1c2329ebee7955eb8152a9f564cd6b78f49691cf.jpg)  
(b) SSC

![](images/79a7159baa02f074083b089bd476923b6d9db1deb2a0de41e0e2d0e732f839ec.jpg)  
(c) EB-SSC  
Figure 1: Building blocks for networks in our paper. The baseline feed-forward model uses (a) concatenated ReLU (CReLU) blocks that preserve both positive and negative activations. (b) A spherical sparse coding layer that has a similar structure but with an extra normalization step. Our proposed model uses (c) energy-based spherical sparse coding (EB-SSC) blocks that produces sparse activations which are not only positive and negative, but are class-specific. These blocks can be stacked to build deeper architectures.

# 1.1 NOTATION

Matrices are denoted as uppercase bold (e.g., A), vectors are lowercase bold (e.g., a), and scalars are lowercase (e.g., a). We denote the transpose operator with  $\mathsf{T}$ , the element-wise multiplication operator with  $\odot$ , the convolution operator with  $*$ , and the cross-correlation operator with  $\star$ . For vectors where we dropped the subscript  $k$  (e.g.,  $\mathbf{d}$  and  $\mathbf{z}$ ), we refer to a super vector with  $K$  components stacked together (e.g.,  $\mathbf{z} = [\mathbf{z}_1^{\mathsf{T}}, \ldots, \mathbf{z}_K^{\mathsf{T}}]^{\mathsf{T}}$ ).

# 2 ENERGY-BASED SPHERICAL SPARSE CODING

Energy-based models capture dependencies between variables using an energy function that measure the compatibility of the configuration of variables (LeCun et al., 2006). To measure the compatibility between the top-down and bottom-up information, we define the energy function of EB-SSC to be the sum of bottom-up coding term and a top-down classification term:

$$
E (\mathbf {x}, y, \mathbf {z}) = E _ {\text {c o d e}} (\mathbf {x}, \mathbf {z}) + E _ {\text {c l a s s}} (y, \mathbf {z}). \tag {1}
$$

The bottom-up information (input signal  $\mathbf{x}$ ) and the top-down information (class label  $y$ ) are tied together by a latent feature map  $\mathbf{z}$ .

# 2.1 BOTTOM-UP RECONSTRUCTION

To measure the compatibility between the input signal  $\mathbf{x}$  and the latent feature maps  $\mathbf{z}$ , we introduce a novel variant of sparse coding that is amenable to efficient feed-forward optimization. While the

idea behind this variant can be applied to either patch-based or convolutional sparse coding, we specifically use the convolutional variant that shares the burden of coding an image among nearby overlapping dictionary elements. Using such a shift-invariant approach avoids the need to learn dictionary elements which are simply translated copies of each other, freeing up resources to discover more diverse and specific filters (see Kavukcuoglu et al. (2010)).

Convolutional sparse coding (CSC) attempts to find a set of dictionary elements  $\{\mathbf{d}_1, \dots, \mathbf{d}_K\}$  and corresponding sparse codes  $\{\mathbf{z}_1, \dots, \mathbf{z}_K\}$  so that the resulting reconstruction,  $\mathbf{r} = \sum_{k=1}^{K} \mathbf{d}_k * \mathbf{z}_k$  accurately represents the input signal  $\mathbf{x}$ . This is traditionally framed as a least-squares minimization with a sparsity inducing prior on  $\mathbf{z}$ :

$$
\underset {\mathbf {z}} {\arg \min } \| \mathbf {x} - \sum_ {k = 1} ^ {K} \mathbf {d} _ {k} * \mathbf {z} _ {k} \| _ {2} ^ {2} + \beta \| \mathbf {z} \| _ {1}. \tag {2}
$$

Unlike standard feed-forward CNN models that convolve the input signal  $\mathbf{x}$  with the filters, this energy function corresponds to a generative model where the latent feature maps  $\{\mathbf{z}_1,\dots ,\mathbf{z}_K\}$  are convolved with the filters and compared to the input signal (Bristow et al., 2013; Heide et al., 2015; Zeiler et al., 2010).

To motivate our novel variant of CSC, consider expanding the squared reconstruction error  $\| \mathbf{x} - \mathbf{r}\| _2^2 = \| \mathbf{x}\| _2^2 -2\mathbf{x}^\top \mathbf{r} + \| \mathbf{r}\| _2^2$ . If we constrain the reconstruction  $\mathbf{r}$  to have unit norm, the reconstruction error depends entirely on the inner product between  $\mathbf{x}$  and  $\mathbf{r}$  and is equivalent to the cosine similarity (up to additive and multiplicative constants). This suggests the closely related unit-length reconstruction problem:

$$
\underset {\mathbf {z}} {\arg \max } \mathbf {x} ^ {\top} \left(\sum_ {k = 1} ^ {K} \mathbf {d} _ {k} * \mathbf {z} _ {k}\right) - \beta \| \mathbf {z} \| _ {1} \tag {3}
$$

$$
\mathrm {s . t .} \left. \left\| \sum_ {k = 1} ^ {K} \mathbf {d} _ {k} * \mathbf {z} _ {k} \right\| _ {2} \leq 1 \right.
$$

In Appendix A we show that, given an optimal unit length reconstruction  $\bar{\mathbf{r}}^*$  with corresponding codes  $\bar{\mathbf{z}}^*$ , the solution to the least squares reconstruction problem (Eq. 2) can be computed by a simple scaling  $\mathbf{r}^* = (\mathbf{x}^\top \bar{\mathbf{r}}^* - \frac{\beta}{2} \| \bar{\mathbf{z}}^* \|_1) \bar{\mathbf{r}}^*$ .

This problem is no easier than the original optimization due to the constraint on the reconstruction, but leads us to consider a simplified constraint on  $\mathbf{z}$  which we refer to as spherical sparse coding (SSC):

$$
\underset {\| \mathbf {z} _ {k} \| _ {2} \leq 1} {\arg \max } E _ {\text {c o d e}} (\mathbf {x}, \mathbf {z}) = \underset {\| \mathbf {z} _ {k} \| _ {2} \leq 1} {\arg \max } \mathbf {x} ^ {\mathsf {T}} \left(\sum_ {k = 1} ^ {K} \mathbf {d} _ {k} * \mathbf {z} _ {k}\right) - \beta \| \mathbf {z} \| _ {1}. \tag {4}
$$

In 2.3 below, we show that the solution to this problem can be found very efficiently without requiring iterative optimization.

This problem is a relaxation of convolutional sparse coding since the code norm constraint upper bounds the reconstruction length. Assuming unit norm dictionary elements, we have by the triangle and Young's inequality:

$$
\left| \left| \sum_ {k} \mathbf {d} _ {k} * \mathbf {z} _ {k} \right| \right| _ {2} \leq \sum_ {k} \| \mathbf {d} _ {k} * \mathbf {z} _ {k} \| _ {2} \leq \sum_ {k} \| \mathbf {d} _ {k} \| _ {1} \| \mathbf {z} _ {k} \| _ {1} \leq D \sum_ {k} \| \mathbf {z} _ {k} \| _ {2} \tag {5}
$$

where  $D$  is the dimension of  $\mathbf{z}_k$ , so that:

$$
\max  _ {\| \sum_ {k} \mathbf {d} _ {k} * \mathbf {z} _ {k} \| _ {2} \leq 1} E _ {\text {c o d e}} (\mathbf {x}, \mathbf {z}) \leq \max  _ {\| \mathbf {z} _ {k} \| _ {2} \leq 1} D ^ {- 1} \cdot E _ {\text {c o d e}} (\mathbf {x}, \mathbf {z}) \tag {6}
$$

However, this relaxation is very loose, primarily due to the triangle inequality. Except in special cases (e.g., if the dictionary elements have disjoint spectra) the SSC codes will be quite different from the standard least-squares reconstruction.

We note that this formulation is also closely related to the dynamical model suggested by Rozell et al. (2008), but lacks dictionary-dependent lateral inhibition between feature maps. Lateral inhibition can solve the unit-length reconstruction formulation of standard sparse coding but requires iterative optimization.

# 2.2 TOP-DOWN CLASSIFICATION

To measure the compatibility between the class label  $y$  and the latent feature maps  $\mathbf{z}$ , we use a set of one-vs-all linear classifiers. To provide more flexibility, we generalize this by splitting the code vector into positive and negative components:

$$
\mathbf {z} _ {k} = \mathbf {z} _ {k} ^ {+} + \mathbf {z} _ {k} ^ {-} \quad \mathbf {z} _ {k} ^ {+} \geq 0 \quad \mathbf {z} _ {k} ^ {-} \leq 0
$$

and allow the linear classifier to operate on each component separately. We express the classifier score for a hypothesized class label  $y$  by:

$$
E _ {\text {c l a s s}} (y, \mathbf {z}) = \sum_ {k = 1} ^ {K} \mathbf {w} _ {y} ^ {+ \intercal} \mathbf {z} _ {k} ^ {+} + \sum_ {k = 1} ^ {K} \mathbf {w} _ {y} ^ {- \intercal} \mathbf {z} _ {k} ^ {-}. \tag {7}
$$

The classifier thus is parameterized by a pair of weight vectors  $(\mathbf{w}_{yk}^{+}$  and  $\mathbf{w}_{yk}^{-})$  for each class label  $y$  and  $k$ -th channel of the latent feature map.

This splitting, sometimes referred to as full-wave rectification, is useful since a dictionary element and its negative do not necessarily have opposite visual semantics. This splitting also allows the classifier the flexibility to assign distinct meanings or alternately be completely invariant to contrast reversal depending on the problem domain. For example, Shang et al. (2016) found CNN models with ReLU non-linearities which discard the negative activations tend to learn pairs of filters which are related by negation. Keeping both positive and negative responses allowed them to halve the number of dictionary elements.

We note that it is also straightforward to introduce spatial average pooling prior to classification by introducing a fixed linear operator  $\mathbf{P}$  used to pool the codes (e.g.,  $\mathbf{w}_y^+ \mathbf{\Psi}^T \mathbf{P} \mathbf{z}_k^+$ ). This is motivated by a variety of hand-engineered feature extractors and sparse coding models, such as Ren & Ramanan (2013), which use spatially pooled histograms of sparse codes for classification. This fixed pooling can be viewed as a form of regularization on the linear classifier which enforces shared weights over spatial blocks of the latent feature map. Splitting is also quite important to prevent information loss when performing additive pooling since positive and negative components of  $\mathbf{z}_k$  can cancel each other out.

# 2.3 CODING

Bottom-up reconstruction and top-down classification each provide half of the story, coupled by the latent feature maps. For a given input  $\mathbf{x}$  and hypothesized class  $y$ , we would like to find the optimal activations  $\mathbf{z}$  that maximize the joint energy function  $E(\mathbf{x}, y, \mathbf{z})$ . This requires solving the following optimization:

$$
\underset {\| \mathbf {z} \| _ {2} \leq 1} {\arg \max } \mathbf {x} ^ {\top} \left(\sum_ {k = 1} ^ {K} \mathbf {d} _ {k} * \mathbf {z} _ {k}\right) - \beta \| \mathbf {z} \| _ {1} + \sum_ {k = 1} ^ {K} \mathbf {w} _ {y k} ^ {+ \top} \mathbf {z} _ {k} ^ {+} + \sum_ {k = 1} ^ {K} \mathbf {w} _ {y k} ^ {- \top} \mathbf {z} _ {k} ^ {-}, \tag {8}
$$

where  $\mathbf{x} \in \mathbb{R}^D$  is an image and  $y \in \mathcal{V}$  is a class hypothesis.  $\mathbf{z}_k \in \mathbb{R}^F$  is the  $k$ -th component latent variable being inferred;  $\mathbf{z}_k^+$  and  $\mathbf{z}_k^-$  are the positive and negative coefficients of  $\mathbf{z}_k$ , such that  $\mathbf{z}_k = \mathbf{z}_k^+ + \mathbf{z}_k^-$ . The parameters  $\mathbf{d}_k \in \mathbb{R}^M$ ,  $\mathbf{w}_{yk}^+ \in \mathbb{R}^F$ , and  $\mathbf{w}_{yk}^- \in \mathbb{R}^F$  are the dictionary filter, positive coefficient classifier, and negative coefficient classifier for the  $k$ -th component respectively. A key aspect of our formulation is that the optimal codes can be found very efficiently in closed-form—in a feed-forward manner (see Appendix B for a detailed argument).

# 2.3.1 ASYMMETRIC SHRINKAGE

To describe the coding processes, let us first define a generalized version of the shrinkage function commonly used in sparse coding. Our asymmetric shrinkage is parameterized by upper and lower thresholds  $-\beta^{-} \leq \beta^{+}$

$$
\operatorname {s h r i n k} _ {\left(\beta^ {+}, \beta^ {-}\right)} (v) = \left\{ \begin{array}{c l} v - \beta^ {+} & \text {i f} v - \beta^ {+} > 0 \\ 0 & \text {o t h e r w i s e} \\ v + \beta^ {-} & \text {i f} v + \beta^ {-} <   0 \end{array} \right. \tag {9}
$$

![](images/3eff5ae2d011d8372a7367dbae7d646770fac421a34ca559c2a717d50ee80970.jpg)  
(a)  $-\beta^{-} \leq 0 \leq \beta^{+}$

![](images/05c0fcb9d9c6a350f06d843f980fc8a0a1d833b064c6280d6775481c62f98fd6.jpg)  
(b)  $0\leq -\beta^{-}\leq \beta^{+}$

![](images/eb8badd2457f4ec83243c2747c377a6601501b4de25df60d4d904046d04a0888.jpg)  
(c)  $-\beta^{-} \leq \beta^{+} \leq 0$  
Figure 2: Comparing the behavior of asymmetric shrinkage for different settings of  $\beta^{+}$  and  $\beta^{-}$ . (a)-(c) satisfy the condition that  $-\beta^{-} \leq \beta^{+}$  while (d) does not.

![](images/8142dbe247fb217521ede9e4656db8395ee4d94da7b059a1d1ca151f881c5d0a.jpg)  
(d)  $\beta^{-}\leq 0\leq -\beta^{+}$

Fig. 2 shows a visualization of this function which generalizes the standard shrinkage proximal operator by allowing for the positive and negative thresholds. In particular, it corresponds to the proximal operator for a version of the  $\ell_1$ -norm that penalizes the positive and negative components with different weights  $|\mathbf{v}|_{asym} = \beta^{+}\| \mathbf{v}^{+}\|_{1} + \beta^{-}\| \mathbf{v}^{-}\|_{1}$ . The standard shrink operator corresponds to  $\mathrm{shrink}_{(\beta , - \beta)}(v)$  while the rectified linear unit common in CNNs is given by a limiting case  $\mathrm{shrink}_{(0, - \infty)}(v)$ . We note that  $-\beta^{-}\leq \beta^{+}$  is required for  $\mathrm{shrink}_{(\beta^{+},\beta^{-})}$  to be a proper function (see Fig. 2).

# 2.3.2 FEED-FORWARD CODING

We now describe how codes can be computed in a simple feed-forward pass. Let

$$
\beta_ {y k} ^ {+} = \beta - \mathbf {w} _ {y k} ^ {+}, \quad \beta_ {y k} ^ {-} = \beta - \mathbf {w} _ {y k} ^ {-} \tag {10}
$$

be vectors of positive and negative biases whose entries are associated with a spatial location in the feature map  $k$  for class  $y$ . The optimal code  $\mathbf{z}$  can be computed in three sequential steps:

1. Cross-correlate the data with the filterbank  $\mathbf{d}_k\star \mathbf{x}$  
2. Apply an asymmetric version of the standard shrinkage operator

$$
\tilde {\mathbf {z}} _ {k} = \operatorname {s h r i n k} _ {\left(\beta_ {y k} ^ {+}, \beta_ {y k} ^ {-}\right)} \left(\mathbf {d} _ {k} \star \mathbf {x}\right) \tag {11}
$$

where, with abuse of notation, we allow the shrinkage function (Eq. 9) to apply entries in the vectors of threshold parameter pairs  $\beta_{yk}^{+},\beta_{yk}^{-}$  to the corresponding elements of the argument.

3. Project onto the feasible set of unit length codes

$$
\mathbf {z} ^ {*} = \frac {\tilde {\mathbf {z}}}{\| \tilde {\mathbf {z}} \| _ {2}}. \tag {12}
$$

# 2.3.3 RELATIONSHIP TO CNNS:

We note that this formulation of coding has a close connection to single layer convolutional neural network (CNN). A typical CNN layer consists of convolution with a filterbank followed by a nonlinear activation such as a rectified linear unit (ReLU). ReLUs can be viewed as another way of inducing sparsity, but rather than coring the values around zero like the shrink function, ReLU truncates negative values. On the other hand, the asymmetric shrink function can be viewed as the sum of two ReLUs applied to appropriately biased inputs:

$$
\operatorname {s h r i n k} _ {(\beta^ {+}, \beta^ {-})} (x) = \operatorname {R e L U} (x - \beta^ {+}) - \operatorname {R e L U} (- (x + \beta^ {-})),
$$

SSC coding can thus be seen as a CNN in which the ReLU activation has been replaced with shrinkage followed by a global normalization.

# 3 LEARNING

We formulate supervised learning using the softmax log-loss that maximizes the energy for the true class label  $y_{i}$  and squashes all other label  $\bar{y}$ .

$$
\begin{array}{l} \operatorname *{arg  min}_{\mathbf{d},\mathbf{w}^{+},\mathbf{w}^{-},\beta \geq 0}\frac{\alpha}{2} (\| \mathbf{w}^{+}\|_{2}^{2} + \| \mathbf{w}^{-}\|_{2}^{2} + \| \mathbf{d}\|_{2}^{2}) \\ + \frac {1}{N} \sum_ {i = 1} ^ {N} \left[ - \max  _ {\| \mathbf {z} \| _ {2} \leq 1} E \left(\mathbf {x} _ {i}, y _ {i}, \mathbf {z}\right) + \log \sum_ {\bar {y} \in \mathcal {Y}} \max  _ {\| \bar {\mathbf {z}} \| _ {2} \leq 1} e ^ {E \left(\mathbf {x} _ {i}, \bar {y}, \bar {\mathbf {z}}\right)} \right], \tag {13} \\ \mathrm {s . t .} - \left(\beta - \mathbf {w} _ {y k} ^ {-}\right) \leq \left(\beta - \mathbf {w} _ {y k} ^ {+}\right) \forall y, k \\ \end{array}
$$

where  $\alpha$  is the hyperparameter regularizing  $\mathbf{w}_y^+$ ,  $\mathbf{w}_y^-$ , and  $\mathbf{d}$ . We constrain the relationship between  $\beta$  and the entries of  $\mathbf{w}_y^+$  and  $\mathbf{w}_y^-$  in order for the asymmetric shrinkage to be a proper function (see Sec. 2.3.1 and Appendix B for details).

In classical sparse coding, it is typical to constrain the  $\ell_2$ -norm of each dictionary filter to unit length. Our spherical coding objective behaves similarly. For any optimal code  $\mathbf{z}^*$ , there is a 1-dimensional subspace of parameters for which  $\mathbf{z}^*$  is optimal given by scaling  $\mathbf{d}$  inversely to  $\mathbf{w}$ ,  $\beta$ . For simplicity of the implementation, we opt to regularize  $\mathbf{d}$  to assure a unique solution. However, as Tygert et al. (2015) point out, it may be advantageous from the perspective of optimization to explicitly constrain the norm of the filter bank.

Note that unlike classical sparse coding, where  $\beta$  is a hyperparameter that is usually set using cross-validation, we treat it as a parameter of the model that is learned to maximize performance.

# 3.1 OPTIMIZATION

In order to solve Eq. 13, we explicitly formulate our model as a directed-acyclic-graph (DAG) neural network with shared weights, where the forward-pass computes the sparse code vectors and the backward-pass updates the parameter weights. We optimize the objective using stochastic gradient descent (SGD).

As mentioned in Sec. 2.3 the amount the shrinkage function applies is either  $\beta_{yk}^{+}$  or  $\beta_{yk}^{-}$  as defined in Eq. 10. To simplify the constraint, we reparameterize the these thresholds relative to a central offset. This means we can use variable substitution and redefine the energy function (Eq. 1) as

$$
E ^ {\prime} (\mathbf {x}, y, \mathbf {z}) = \mathbf {x} ^ {\intercal} \left(\sum_ {k = 1} ^ {K} \mathbf {d} _ {k} * \mathbf {z} _ {k}\right) + b _ {k} \mathbf {1} ^ {\intercal} \mathbf {z} _ {k} - \sum_ {k = 1} ^ {K} \hat {\mathbf {w}} _ {y k} ^ {+ \intercal} \mathbf {z} _ {k} ^ {+} + \sum_ {k = 1} ^ {K} \hat {\mathbf {w}} _ {y k} ^ {- \intercal} \mathbf {z} _ {k} ^ {-} \tag {14}
$$

where  $\mathbf{b}$  is constant offset for each code channel. The modified linear "classification" terms now take on a dual role of inducing sparsity and measuring the compatibility between  $\mathbf{z}$  and  $y$ . Adding in this additional offset term allows us to convert the inequality constraint into simple positivity constraints on the classifier parameters.

This yields a modified learning objective that can easily be solved with existing implementations for learning convolutional neural nets:

$$
\begin{array}{l} \underset {\mathbf {d}, \hat {\mathbf {w}} ^ {+}, \hat {\mathbf {w}} ^ {-}, \mathbf {b}} {\arg \min } \frac {\alpha}{2} \left(\| \hat {\mathbf {w}} ^ {+} \| _ {2} ^ {2} + \| \hat {\mathbf {w}} ^ {-} \| _ {2} ^ {2} + \| \mathbf {d} \| _ {2} ^ {2}\right) \\ + \frac {1}{N} \sum_ {i = 1} ^ {N} \left[ - \max  _ {\| \mathbf {z} \| _ {2} \leq 1} E ^ {\prime} \left(\mathbf {x} _ {i}, y _ {i}, \mathbf {z}\right) + \log \sum_ {\bar {y} \in \mathcal {Y}} \max  _ {\| \bar {\mathbf {z}} \| _ {2} \leq 1} e ^ {E ^ {\prime} \left(\mathbf {x} _ {i}, \bar {y}, \mathbf {z}\right)} \right], \tag {15} \\ \mathrm {s . t .} \hat {\mathbf {w}} _ {y k} ^ {+}, \hat {\mathbf {w}} _ {y k} ^ {-} \succeq 0 \quad \forall y \\ \end{array}
$$

where  $\hat{\mathbf{w}}^{+}$  and  $\hat{\mathbf{w}}^{-}$  are the new sparsity inducing classifiers, and  $\mathbf{b}$  are the arbitrary origin points. In particular, adding the  $K$  origin points allows us to enforce the constraint by simply projecting  $\hat{\mathbf{w}}^{+}$  and  $\hat{\mathbf{w}}^{-}$  onto the positive orthant during SGD.

<table><tr><td colspan="3">Base Network</td></tr><tr><td>block</td><td>kernel, stride, padding</td><td>activation</td></tr><tr><td>conv1</td><td>3 × 3 × 3 × 96, 1, 1</td><td>CReLU</td></tr><tr><td>conv2</td><td>3 × 3 × 192 × 96, 1, 1</td><td>CReLU</td></tr><tr><td>pool1</td><td>3 × 3, 2, 1</td><td></td></tr><tr><td>conv3</td><td>3 × 3 × 192 × 192, 1, 1</td><td>CReLU</td></tr><tr><td>conv4</td><td>3 × 3 × 384 × 192, 1, 1</td><td>CReLU</td></tr><tr><td>conv5</td><td>3 × 3 × 384 × 192, 1, 1</td><td>CReLU</td></tr><tr><td>pool2</td><td>3 × 3, 2, 1</td><td></td></tr><tr><td>conv6</td><td>3 × 3 × 384 × 192, 1, 1</td><td>CReLU</td></tr><tr><td>conv7</td><td>1 × 1 × 384 × 192, 1, 1</td><td>CReLU</td></tr></table>

Table 1: Underlying block architecture common across all models we evaluated. SSC networks add an extra normalization layer after the non-linearity. And EB-SSC networks insert class-specific bias layers between the convolution layer and the non-linearity. Concatenated ReLU (CReLU) splits positive and negative activations into two separate channels rather than discarding the negative component as in the standard ReLU.

# 4 EXPERIMENTS

We evaluate the benefits of combining top-down and bottom-up information to produce class-specific features on the CIFAR-10 (Krizhevsky & Hinton, 2009) dataset using a deep version of our EB-SSC. All experiments were performed using MatConvNet (Vedaldi & Lenc, 2015) framework with the ADAM optimizer (Kingma & Ba, 2014). The data was preprocessed and augmented following the procedure in Goodfellow et al. (2013). Specifically, the data was subtracted by the mean image and was whitened; the augmentation strategy adopted was horizontal flip (with a 0.5 probability) and random cropping. No weight decay was used, but we used a dropout rate of 0.3 before every convolution layer except for the first. Our full energy-based model requires roughly one week to train from scratch on an NVIDIA Titan X.

# 4.1 CLASSIFICATION

We compare our proposed EB-SSC model to that of Shang et al. (2016), which uses concatenated rectified linear units (CReLU) as its non-linearity. This model can be viewed as a basic feed-forward version of our proposed model which we take as a baseline. We also consider variants of the baseline model which take attributes from the proposed model to understand how subtle design changes of the network architecture affects performance.

We describe the model architecture in terms of the feature extractor and classifier. Table 1 shows the overall network architecture of feature extractors, which consist of seven convolution blocks and two pooling layers. We consider max-pooling and average-pooling for feature extraction and indicate which operation in the subscript of the feature extractor (e.g.,  $\mathrm{SSC}_{avg}$  is used to denote average-pooling with spherical sparse coding features). We look at two possible classifiers: a simple linear classifier (LC) and a shallow energy-based classifier (EBC), and use softmax-loss for all models. A numerical superscript indicates which of the seven conv block of the feature extractor is used for classification (e.g.,  $\mathrm{LC}_7$  indicates the activations out of the last conv block is fed into the linear classifier).

The results shown in Table 2 compare our proposed model to the baseline  $\mathrm{CReLU}_{max} + \mathrm{LC}_7$  (Shang et al., 2016), and to intermediate variants. The baseline models all perform very similarly with some small reductions in error rates over the baseline  $\mathrm{CReLU}_{max} + \mathrm{LC}_7$ . Our full energy based model performs substantially worse compared to the baseline model,  $15.62\%$  versus  $10.68\%$ . There are two factors that may be contributing to this worse performance. The first is that we did no optimization of learning hyperparameters (learning rates and dropout rates). Evaluation was done using the same parameters across the board and were obtained from Shang et al. (2016). Given that the full  $EBC_{1-7}$  model has substantially higher capacity (10.1M parameters versus 2.6M parameters), we expect that the model tested here is under-fit (as suggested by the training error). The second factor is that, as can be seen in the table, average-pooling generally performs worse than max pooling in

<table><tr><td>Model</td><td>Train Err. (%)</td><td>Test Err. (%)</td><td># params</td></tr><tr><td>CReLUavg+LC7</td><td>1.16</td><td>11.01</td><td>2.6M</td></tr><tr><td>CReLU max + LC7</td><td>0.98</td><td>10.68</td><td>2.6M</td></tr><tr><td>CReLU max + EBC7</td><td>0.63</td><td>10.49</td><td>2.9M</td></tr><tr><td>SSCavg+LC7</td><td>1.52</td><td>11.84</td><td>2.6M</td></tr><tr><td>SSCmax+LC7</td><td>0.32</td><td>10.37</td><td>2.6M</td></tr><tr><td>SSCavg+ EBC7</td><td>3.91</td><td>13.64</td><td>2.9M</td></tr><tr><td>SSCavg+ EBC1-7</td><td>3.24</td><td>15.62</td><td>10.1M</td></tr></table>

Table 2: Comparison of the baseline  $\mathrm{CReLU}_{\text{max}} + \mathrm{LC}_7$  model, its derivative models, and our proposed model on CIFAR-10.

other models. We expect the same to be true here. As of the submission deadline the max-pooled variants were still training.

# 4.2 DECODING CLASS-SPECIFIC CODES

A unique aspect of our model is that it is generative in the sense that each layer is explicitly trying to encode the activation pattern in the prior layer. Similar to the work on deconvolutional networks built on least-squares sparse coding (Zeiler et al., 2010), we can synthesize input images from activations in our spherical coding network by performing repeated deconvolutions (transposed convolutions) back through the network. Since our model is energy based, we can further examine how the top-down information of a hypothesized class effects the intermediate activations.

The first column in Fig. 3 visualizes reconstructions of a given input image based on activations from different layers of the model by convolution transpose. In this case we put in 0 class biases (no top-down) and are able to recover high fidelity reconstructions of the input. In the remaining columns, we use the same deconvolution pass to construct input space representations of the learned classifier biases. At low levels of the feature hierarchy, these biases are spatially smooth since the receptive fields are small and there is little spatial invariance capture in the activations. At higher levels these class-conditional bias fields become more tightly localized.

Finally, in Fig. 4 we show decodings from the conv2 and conv5 layer of the EB-SSC model for a given input under different class hypotheses. Here we subtract out the contribution of the top-down bias term in order to isolate the effect of the class conditioning on the encoding of input features. As visible in the figure, the modulation of the activations focused around particular regions of the image and the differences across class hypotheses become more pronounced at higher layers of the network.

![](images/5b0c9b4b03ede1e78ba7f1cc0bf308aae34889c15304a6186bc4ce36e567aaac.jpg)  
Figure 3: The reconstruction of an airplane image from different levels of the network (rows) across different hypothesized class labels (columns). The first column is pure reconstruction, i.e., unbiased by a hypothesized class label, the remaining columns are one of ten possible CIFAR-10 class labels. (Best viewed in color.)

![](images/7f3c068dc36ee7d3d5e8d5374eb066bc4acdef36187cf6bbc4b90c1332a5fa36.jpg)  
Figure 4: We visualize the reconstruction of different input images (rows) for each of 10 different class hypotheses (cols) from the 2nd and 5th block activations. (Best viewed in color.)

# 5 CONCLUSION

We presented an energy-based sparse coding method that efficiently combines cosine similarity convolutional sparse coding and linear classification. Our model shows a clear mathematical connection between the activation functions used in CNNs to introduce sparsity and our cosine similarity convolutional sparse coding formulation. Although our proposed model did not outperform the baseline models, it provides an interesting framework to probe the effects of class-specific coding.

# REFERENCES

Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, and Uri Shaft. When is "nearest neighbor" meaningful? In International conference on database theory (ICDT), 1999.  
Hilton Bristow, Anders Eriksson, and Simon Lucey. Fast convolutional sparse coding. In Computer Vision and Pattern Recognition (CVPR), 2013.  
Chunshui Cao, Xianming Liu, Yi Yang, Yinan Yu, Jiang Wang, Zilei Wang, Yongzhen Huang, Liang Wang, Chang Huang, Wei Xu, et al. Look and think twice: Capturing top-down visual attention with feedback convolutional neural networks. In International Conference on Computer Vision (ICCV), 2015.  
Jonghyun Choi, Hyunjong Cho, Jungsuk Kwac, and Larry S Davis. Toward sparse coding on cosine distance. In International Conference on Pattern Recognition (ICPR), 2014.  
David L Donoho. Compressed sensing. IEEE Transactions on information theory, 2006.  
Michael Elad and Michal Aharon. Image denoising via sparse and redundant representations over learned dictionaries. IEEE Transactions on Image processing, 2006.  
Ian J Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron C Courville, and Yoshua Bengio. Maxout networks. In International conference on Machine learning (ICML), 2013.  
Felix Heide, Wolfgang Heidrich, and Gordon Wetzstein. Fast and flexible convolutional sparse coding. In Computer Vision and Pattern Recognition (CVPR), 2015.  
Michael E Houle, Hans-Peter Kriegel, Peer Kröger, Erich Schubert, and Arthur Zimek. Can shared-neighbor distances defeat the curse of dimensionality? In International Conference on Scientific and Statistical Database Management (ICSSDM), 2010.

Zhengping Ji, Wentao Huang, G. Kenyon, and L.M.A. Bettencourt. Hierarchical discriminative sparse coding via bidirectional connections. In International Joint Convergence on Neural Networks (IJCNN), 2011.  
Zhuolin Jiang, Zhe Lin, and Larry S Davis. Learning a discriminative dictionary for sparse coding via label consistent K-SVD. In Computer Vision and Pattern Recognition (CVPR), 2011.  
Koray Kavukcuoglu, Pierre Sermanet, Y-Lan Boureau, Karol Gregor, Michael Mathieu, and Yann L Cun. Learning convolutional feature hierarchies for visual recognition. In Advances in neural information processing systems (NIPS), 2010.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.  
Hugo Larochelle and Yoshua Bengio. Classification using discriminative restricted boltzmann machines. In International conference on Machine learning (ICML), 2008.  
Yann LeCun, Sumit Chopra, Raia Hadsell, M Ranzato, and F Huang. A tutorial on energy-based learning. Predicting structured data, 2006.  
Xin Li and Yuhong Guo. Bi-directional representation learning for multi-label classification. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML KDD). 2014.  
Bruno A Olshausen and David J Field. Sparse coding with an overcomplete basis set: A strategy employed by v1? Vision research, 1997.  
Xiaofeng Ren and Deva Ramanan. Histograms of sparse codes for object detection. In Computer Vision and Pattern Recognition (CVPR), 2013.  
Christopher J Rozell, Don H Johnson, Richard G Baraniuk, and Bruno A Olshausen. Sparse coding via thresholding and local competition in neural circuits. Neural computation, 2008.  
Wenling Shang, Kihyuk Sohn, Diogo Almeida, and Honglak Lee. Understanding and improving convolutional neural networks via concatenated rectified linear units. In International conference on Machine learning (ICML), 2016.  
Mark Tygert, Arthur Szlam, Soumith Chintala, Marc'Aurelio Ranzato, Yuandong Tian, and Wojciech Zaremba. Convolutional networks and learning invariant to homogeneous multiplicative scalings. arXiv preprint arXiv:1506.08230, 2015.  
A. Vedaldi and K. Lenc. Matconvnet - convolutional neural networks for matlab. In ACM International Conference on Multimedia, 2015.  
John Wright, Allen Y Yang, Arvind Ganesh, S Shankar Sastry, and Yi Ma. Robust face recognition via sparse representation. IEEE transactions on pattern analysis and machine intelligence (TPAMI), 2009.  
Jianxin Wu and James M Rehg. Beyond the euclidean distance: Creating effective visual codebooks using the histogram intersection kernel. In International Conference on Computer Vision (ICCV), 2009.  
Shuicheng Yan, Huan Wang, Xiaou Tang, and Thomas Huang. Exploring feature descriptors for face recognition. In International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2007.  
Allen Y Yang, Zihan Zhou, Arvind Ganesh Balasubramanian, S Shankar Sastry, and Yi Ma. Fast-minimization algorithms for robust face recognition. IEEE Transactions on Image Processing, 2013.  
Jianchao Yang, Kai Yu, and Thomas Huang. Supervised translation-invariant sparse coding. In Computer Vision and Pattern Recognition (CVPR), 2010.

Matthew D. Zeiler, Dilip Krishnan, Graham W. Taylor, and Robert Fergus. Deconvolutional networks. In Computer Vision and Pattern Recognition (CVPR), 2010.  
Yangmuzi Zhang, Zhuolin Jiang, and Larry S Davis. Discriminative tensor sparse coding for image classification. In *British Machine Vision Conference (BMVC)*, 2013.  
Ning Zhou, Yi Shen, Jinye Peng, and Jianping Fan. Learning inter-related visual dictionary for object recognition. In Computer Vision and Pattern Recognition (CVPR), 2012.
