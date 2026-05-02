# RETHINKING ATTENTION WITH PERFORMERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We introduce Performers, Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness. To approximate softmax attention-kernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+), which may be of independent interest for scalable kernel methods. FAVOR+ can also be used to efficiently model kernelizable attention mechanisms beyond softmax. This representational power is crucial to accurately compare softmax with other kernels for the first time on large-scale tasks, beyond the reach of regular Transformers, and investigate optimal attention-kernels. Performers are linear architectures fully compatible with regular Transformers and with strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and low estimation variance. We tested Performers on a rich set of tasks stretching from pixel-prediction through text models to protein sequence modeling. We demonstrate competitive results with other examined efficient sparse and dense attention methods, showcasing effectiveness of the novel attention-learning paradigm leveraged by Performers.

# 1 INTRODUCTION AND RELATED WORK

Transformers (Vaswani et al., 2017; Dehghani et al., 2019) are powerful neural network architectures that have become SOTA in several areas of machine learning including natural language processing (NLP) (e.g. speech recognition (Luo et al., 2020)), neural machine translation (NMT) (Chen et al., 2018), document generation/summarization, time series prediction, generative modeling (e.g. image generation (Parmar et al., 2018)), music generation (Huang et al., 2019), and bioinformatics (Rives et al., 2019; Madani et al., 2020; Ingraham et al., 2019; Elnaggar et al., 2019; Du et al., 2020).

Transformers rely on a trainable attention mechanism that identifies complex dependencies between the elements of each input sequence. Unfortunately, the regular Transformer scales quadratically with the number of tokens  $L$  in the input sequence, which is prohibitively expensive for large  $L$  and precludes its usage in settings with limited computational resources even for moderate values of  $L$ . Several solutions have been proposed to address this issue (Beltagy et al., 2020; Gulati et al., 2020; Chan et al., 2020; Child et al., 2019; Bello et al., 2019). Most approaches restrict the attention mechanism to attend to local neighborhoods (Parmar et al., 2018) or incorporate structural priors on attention such as sparsity (Child et al., 2019), pooling-based compression (Rae et al., 2020) clustering/binning/convolution techniques (e.g. (Roy et al., 2020) which applies  $k$ -means clustering to learn dynamic sparse attention regions, or (Kitaev et al., 2020), where locality sensitive hashing is used to group together tokens of similar embeddings), sliding windows (Beltagy et al., 2020), or truncated targeting (Chelba et al., 2020). There is also a long line of research on using dense attention matrices, but defined by low-rank kernels substituting softmax (Katharopoulos et al., 2020; Shen et al., 2018). Those methods critically rely on kernels admitting explicit representations as dot-products of finite positive-feature vectors.

The approaches above do not aim to approximate regular attention, but rather propose simpler and more tractable attention mechanisms, often by incorporating additional constraints (e.g. identical query and key sets as in (Kitaev et al., 2020)), or by trading regular with sparse attention using more layers (Child et al., 2019). Unfortunately, there is a lack of rigorous guarantees for the representation power produced by such methods, and sometimes the validity of sparsity patterns can only be verified empirically through trial and error by constructing special GPU operations (e.g. either writing  $\mathrm{C + + }$  CUDA kernels (Child et al., 2019) or using TVMs (Beltagy et al., 2020)). Other techniques which

aim to reduce Transformers' space complexity include reversible residual layers allowing one-time activation storage in training (Kitaev et al., 2020) and shared attention weights (Xiao et al., 2019). These constraints may impede application to long-sequence problems, where approximations of the attention mechanism are not sufficient. Approximations based on truncated back-propagation (Dai et al., 2019) are also unable to capture long-distance correlations since the gradients are only propagated inside a localized window. Other methods propose biased estimation of the regular attention but only in the non-causal setting and of large mean squared error (Wang et al., 2020).

In response, we introduce the first Transformer architectures, Performers, capable of provably accurate and practical estimation of regular (softmax) full-rank attention, but of only linear space and time complexity and not relying on any priors such as sparsity or low-rankness. Performers use the Fast Attention Via positive Orthogonal Random features (FAVOR+) mechanism, leveraging new methods for approximating softmax and Gaussian kernels, which we propose. We believe these methods are of independent interest, contributing to the theory of scalable kernel methods. Consequently, Performers are the first linear architectures fully compatible (via small amounts of fine-tuning) with regular Transformers, providing strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and lower variance of the approximation.

FAVOR+ can be also applied to efficiently model other kernelizable attention mechanisms beyond softmax. This representational power is crucial to accurately compare softmax with other kernels for the first time on large-scale tasks, that are beyond the reach of regular Transformers, and find for them optimal attention-kernels. FAVOR+ can be also applied beyond the Transformer scope as a more scalable replacement for regular attention, which itself has a wide variety of uses in computer vision (Fu et al., 2019), reinforcement learning (Zambaldi et al., 2019), training with softmax cross entropy loss, and even combinatorial optimization (Vinyals et al., 2015).

We test Performers on a rich set of tasks ranging from pixel-prediction through text models to protein sequence modeling. We demonstrate competitive results with other examined efficient sparse and dense attention methods, showcasing the effectiveness of the novel attention-learning paradigm leveraged by Performers. We emphasize that in principle, FAVOR+ can also be combined with other techniques, such as reversible layers (Kitaev et al., 2020) or cluster-based attention (Roy et al., 2020).

# 2 FAVOR+ MECHANISM & POSITIVE ORTHOGONAL RANDOM FEATURES

Below we describe in detail the FAVOR+ mechanism - the backbone of the Performer's architecture. We introduce a new method for estimating softmax (and Gaussian) kernels with positive orthogonal random features which FAVOR+ leverages for the robust and unbiased estimation of regular (softmax) attention and show how FAVOR+ can be applied for other attention-kernels.

# 2.1 PRELIMINARIES - REGULAR ATTENTION MECHANISM

Let  $L$  be the size of an input sequence of tokens. Then regular dot-product attention (Vaswani et al., 2017) is a mapping which accepts matrices  $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{L \times d}$  as input where  $d$  is the hidden dimension (dimension of the latent representation). Matrices  $\mathbf{Q}, \mathbf{K}, \mathbf{V}$  are intermediate representations of the input and their rows can be interpreted as queries, keys and values of the continuous dictionary data structure respectively. Bidirectional (or non-directional (Devlin et al., 2018)) dot-product attention has the following form, where  $\mathbf{A} \in \mathbb{R}^{L \times L}$  is the so-called attention matrix:

$$
\operatorname {A t t} _ {\leftrightarrow} (\mathbf {Q}, \mathbf {K}, \mathbf {V}) = \mathbf {D} ^ {- 1} \mathbf {A V}, \quad \mathbf {A} = \exp (\mathbf {Q} \mathbf {K} ^ {\top} / \sqrt {d}), \quad \mathbf {D} = \operatorname {d i a g} (\mathbf {A} \mathbf {1} _ {L}). \tag {1}
$$

Here  $\exp (\cdot)$  is applied elementwise,  $\mathbf{1}_L$  is the all-ones vector of length  $L$ , and  $\mathrm{diag}(\cdot)$  is a diagonal matrix with the input vector as the diagonal. Time and space complexity of computing (1) are  $O(L^2 d)$  and  $O(L^{2} + Ld)$  respectively, because  $\mathbf{A}$  has to be stored explicitly. Hence, in principle, dot-product attention of type (1) is incompatible with end-to-end processing of long sequences. Bidirectional attention is applied in encoder self-attention and encoder-decoder attention in Seq2Seq architectures.

Another important type of attention is unidirectional dot-product attention which has the form:

$$
\operatorname {A t t} _ {\rightarrow} (\mathbf {Q}, \mathbf {K}, \mathbf {V}) = \widetilde {\mathbf {D}} ^ {- 1} \widetilde {\mathbf {A}} \mathbf {V}, \quad \widetilde {\mathbf {A}} = \operatorname {t r i l} (\mathbf {A}), \quad \widetilde {\mathbf {D}} = \operatorname {d i a g} (\widetilde {\mathbf {A}} \mathbf {1} _ {L}), \tag {2}
$$

where  $\mathrm{tril}(\cdot)$  returns the lower-triangular part of the argument matrix including the diagonal. As discussed in (Vaswani et al., 2017), unidirectional attention is used for autoregressive generative

modelling, e.g. as self-attention in generative Transformers as well as the decoder part of Seq2Seq Transformers.

We will show that attention matrix  $\mathbf{A}$  can be approximated up to any precision in time  $O(Ld^{2}\log (d))$ . For comparison, popular methods leveraging sparsity via Locality-Sensitive Hashing (LSH) techniques (Kitaev et al., 2020) have  $O(Ld^{2}\log L)$  time complexity. In the main body of the paper we will describe FAVOR+ for bidirectional attention. Completely analogous results can be obtained for the unidirectional variant via the mechanism of prefix-sums (all details in the Appendix B.1).

# 2.2 GENERALIZED KERNELIZABLE ATTENTION

FAVOR+ works for attention blocks using matrices  $\mathbf{A} \in \mathbb{R}^{L \times L}$  of the form  $\mathbf{A}(i,j) = \mathrm{K}(\mathbf{q}_i^\top, \mathbf{k}_j^\top)$ , with  $\mathbf{q}_i / \mathbf{k}_j$  standing for the  $i^{th} / j^{th}$  query/key row-vector in  $\mathbf{Q} / \mathbf{K}$  and kernel  $\mathrm{K}: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_+$  defined for the (usually randomized) mapping:  $\phi: \mathbb{R}^d \to \mathbb{R}_+^r$  (for some  $r > 0$ ) as:

$$
\mathrm {K} (\mathbf {x}, \mathbf {y}) = \mathbb {E} [ \phi (\mathbf {x}) ^ {\top} \phi (\mathbf {y}) ]. \tag {3}
$$

We call  $\phi(\mathbf{u})$  a random feature map for  $\mathbf{u} \in \mathbb{R}^d$ . For  $\mathbf{Q}'$ ,  $\mathbf{K}' \in \mathbb{R}^{L \times r}$  with rows given as  $\phi(\mathbf{q}_i^\top)^\top$  and  $\phi(\mathbf{k}_i^\top)^\top$  respectively, Equation 3 leads directly to the efficient attention mechanism of the form:

$$
\widehat {\mathrm {A t t} _ {\leftrightarrow}} (\mathbf {Q}, \mathbf {K}, \mathbf {V}) = \widehat {\mathbf {D}} ^ {- 1} (\mathbf {Q} ^ {\prime} ((\mathbf {K} ^ {\prime}) ^ {\top} \mathbf {V})), \qquad \widehat {\mathbf {D}} = \operatorname {d i a g} (\mathbf {Q} ^ {\prime} ((\mathbf {K} ^ {\prime}) ^ {\top} \mathbf {1} _ {L})). \qquad (4)
$$

Here  $\widehat{\mathrm{Att}}_{\leftrightarrow}$  stands for the approximate attention and brackets indicate the order of computations. It is easy to see that such a mechanism is characterized by space complexity  $O(Lr + Ld + rd)$  and time complexity  $O(Lrd)$  as opposed to  $O(L^2 +Ld)$  and  $O(L^{2}d)$  of the regular attention (see also Fig. 1).

![](images/b7732e346beefc39f0c50e2be846322394014a2807e96c4d20f9143316b25d90.jpg)  
Figure 1: Approximation of the regular attention mechanism AV (before  $\mathbf{D}^{-1}$ -renormalization) via (random) feature maps. Dashed-blocks indicate order of computation with corresponding time complexities attached.

![](images/eb9c92bf7426b260274280d60c2ab7e4160048f87ed44fc9dc3d849067e0d8e6.jpg)

The above scheme constitutes the FA-part of the FAVOR+ mechanism. The remaining OR+ part answers the following questions: (1) How expressive is the attention model defined in Equation 3, and in particular, can we use it in principle to approximate regular softmax attention? (2) How do we implement it robustly in practice, and in particular, can we choose  $r \ll L$  for  $L \gg d$  to obtain desired space and time complexity gains? We answer these questions in the next sections.

# 2.3 HOW TO AND HOW NOT TO APPROXIMATE SOFTMAX-KERNS FOR ATTENTION

It turns out that by taking  $\phi$  of the following form for functions  $f_{1},\ldots ,f_{l}:\mathbb{R}\to \mathbb{R}$ , function  $g:\mathbb{R}^d\to \mathbb{R}$  and deterministic vectors  $\omega_{i}$  or  $\omega_{1},\dots,\omega_{m}\stackrel{\mathrm{iid}}{\sim}\mathcal{D}$  for some distribution  $\mathcal{D}\in \mathcal{P}(\mathbb{R})^d$ :

$$
\phi (\mathbf {x}) = \frac {h (\mathbf {x})}{\sqrt {m}} \left(f _ {1} \left(\omega_ {1} ^ {\top} \mathbf {x}\right), \dots , f _ {1} \left(\omega_ {m} ^ {\top} \mathbf {x}\right), \dots , f _ {l} \left(\omega_ {1} ^ {\top} \mathbf {x}\right), \dots , f _ {l} \left(\omega_ {m} ^ {\top} \mathbf {x}\right)\right), \tag {5}
$$

we can model most kernels used in practice. Furthermore, in most cases  $\mathcal{D}$  is isotropic (i.e. with pdf function constant on a sphere), usually Gaussian. For example, by taking  $h(\mathbf{x}) = 1$ ,  $l = 1$  and  $\mathcal{D} = \mathcal{N}(0,\mathbf{I}_d)$  we obtain estimators of the so-called PNG-kernels (Choromanski et al., 2017) (e.g.  $f_{1} = \mathrm{sgn}$  corresponds to the angular kernel). Configurations:  $h(\mathbf{x}) = 1$ ,  $l = 2$ ,  $f_{1} = \sin$ ,  $f_{2} = \cos$  correspond to shift-invariant kernels, in particular  $\mathcal{D} = \mathcal{N}(0,\mathbf{I}_d)$  leads to the Gaussian kernel  $\mathrm{K}_{\mathrm{gauss}}$  (Rahimi & Recht, 2007). The softmax-kernel which defines regular attention matrix  $\mathbf{A}$  is given as:

$$
\operatorname {S M} (\mathbf {x}, \mathbf {y}) \stackrel {\text {d e f}} {=} \exp \left(\mathbf {x} ^ {\top} \mathbf {y}\right). \tag {6}
$$

In the above, without loss of generality, we omit  $\sqrt{d}$ -renormalization since we can equivalently renormalize input keys and queries. Since:  $\mathrm{SM}(\mathbf{x},\mathbf{y}) = \exp (\frac{\|\mathbf{x}\|^2}{2})\mathrm{K}_{\mathrm{gauss}}(\mathbf{x},\mathbf{y})\exp (\frac{\|\mathbf{y}\|^2}{2})$  based on what we have said, we obtain random feature map unbiased approximation of  $\mathrm{SM}(\mathbf{x},\mathbf{y})$  using trigonometric functions with:  $h(\mathbf{x}) = \exp (\frac{\|\mathbf{x}\|^2}{2}),l = 2,f_1 = \sin ,f_2 = \cos .$  We call it  $\widehat{\mathrm{SM}}_{m}^{\mathrm{trig}}(\mathbf{x},\mathbf{y})$

There is however a caveat there. The attention module from (1) constructs for each token, a convex combination of value-vectors with coefficients given as corresponding renormalized kernel scores. That is why kernels producing non-negative scores are used. Applying random feature maps with potentially negative dimension-values  $(\sin / \cos)$  leads to unstable behaviours, especially when kernel scores close to 0 (which is the case for lots of entries of  $\mathbf{A}$  corresponding to low relevance tokens) are approximated by estimators with large variance in such regions. This results in abnormal behaviours, e.g. negative-diagonal-values renormalizers  $\mathbf{D}^{-1}$ , and consequently either completely prevents training or leads to sub-optimal models. We demonstrate empirically that this is what happens for  $\widehat{\mathrm{SM}}_m^{\mathrm{trig}}$  and provide detailed theoretical explanations showing that the variance of  $\widehat{\mathrm{SM}}_m^{\mathrm{trig}}$  is large as approximated values tend to 0 (see: Section 3). This is one of the main reasons why the robust random feature map mechanism for approximating regular softmax attention was never proposed.

We propose a robust mechanism in this paper. Furthermore, the variance of our new unbiased positive random feature map estimator tends to 0 as approximated values tend to 0 (see: Section 3).

Lemma 1 (Positive Random Features (PRFs) for Softmax). For  $\mathbf{x},\mathbf{y}\in \mathbb{R}^d$ $\mathbf{z} = \mathbf{x} + \mathbf{y}$  we have:

$$
\operatorname {S M} (\mathbf {x}, \mathbf {y}) = \mathbb {E} _ {\omega \sim \mathcal {N} (0, \mathbf {I} _ {d})} [ \exp (\omega^ {\top} \mathbf {x} - \frac {\| \mathbf {x} \| ^ {2}}{2}) \exp (\omega^ {\top} \mathbf {y} - \frac {\| \mathbf {y} \| ^ {2}}{2}) ] = \Lambda \mathbb {E} _ {\omega \sim \mathcal {N} (0, \mathbf {I} _ {d})} \cosh (\omega^ {\top} \mathbf {z}), \tag {7}
$$

where  $\Lambda = \exp(-\frac{\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2}{2})$  and  $\cosh$  is a hyperbolic cosine. Consequently, softmax-kernel admits a positive random feature map unbiased approximation with  $h(\mathbf{x}) = \exp(-\frac{\|\mathbf{x}\|^2}{2})$ ,  $l = 1$ ,  $f_1 = \exp$  and  $\mathcal{D} = \mathcal{N}(0, \mathbf{I}_d)$  or:  $h(\mathbf{x}) = \frac{1}{\sqrt{2}} \exp(-\frac{\|\mathbf{x}\|^2}{2})$ ,  $l = 2$ ,  $f_1(u) = \exp(u)$ ,  $f_2(u) = \exp(-u)$  and the same  $\mathcal{D}$  (the latter for further variance reduction). We call related estimators:  $\widehat{\mathrm{SM}}_m^+$  and  $\widehat{\mathrm{SM}}_m^{\mathrm{hyp}+}$ .

![](images/654915a49d488dee5e39f5967ca820c0867154299029323458097738878f6952.jpg)  
Figure 2: Left: Symmetrized (around origin) utility function  $r$  (defined as a ratio of the mean squared errors (MSEs) of estimators built on: trigonometric and positive random features) as a function of the angle  $\phi$  (in radians) between input feature vectors and their lengths  $l$ . Larger values indicate regions of  $(\phi, l)$ -space with better performance of positive random features. We see that for critical regions with  $\phi$  large enough (small enough softmax-kernel values) our method is arbitrarily more accurate than trigonometric random features. Plot presented for domain  $[-\pi, \pi] \times [-2, 2]$ . Right: The slice of function  $r$  for fixed  $l = 1$  and varying angle  $\phi$ . Right Upper Corner: Comparison of the MSEs of both the estimators in a low softmax-kernel value region.

In Fig. 2 we visualize the advantages of positive versus standard trigonometric random features. In critical regions, where kernel values are small and need careful approximation, our method outperforms its counterpart. In Section 4 we further confirm our method's advantages empirically, using positive features to efficiently train softmax-based linear Transformers. If we replace in (7)  $\omega$  with  $\sqrt{d}\frac{\omega}{\|\omega\|}$ , we obtain the so-called regularized softmax-kernel SMREG which we can approximate in a similar manner, simply changing  $\mathcal{D} = \mathcal{N}(0,\mathbf{I}_d)$  to  $\mathcal{D} = \mathrm{Unif}(\sqrt{d}\mathcal{S}^{d - 1})$ , a distribution corresponding to Haar measure on the sphere of radius  $\sqrt{d}$  in  $\mathbb{R}^d$ , obtaining estimator  $\widehat{\mathrm{SMREG}}_m^+$ . As we show in Section 3, such random features can be also used to accurately approximate regular softmax-kernel.

# 2.4 ORTHOGONAL RANDOM FEATURES (ORFs)

The above constitutes the  $\mathbb{R}+$  part of the FAVOR+ method. It remains to explain the O-part. To further reduce the variance of the estimator (so that we can use even smaller number of random features  $r$ ), we entangle different random samples  $\omega_{1}, \ldots, \omega_{m}$  to be exactly orthogonal. This can be done while maintaining unbiasedness whenever isotropic distributions  $\mathcal{D}$  are used (i.e. in particular in all kernels we considered so far) by standard Gram-Schmidt renormalization procedure (see: (Choromanski et al., 2017) for details). ORFs is a well-known method, yet it turns out that it works particularly well with our introduced PRFs for softmax. This leads to first theoretical results showing that ORFs can be applied to reduce the variance of softmax/Gaussian kernel estimators for any dimensionality  $d$  rather than just asymptotically for large enough  $d$  (as is the case for previous methods, see: next section) and leads to the first exponentially small bounds on large deviations probabilities that are strictly smaller than for non-orthogonal methods. Positivity of random features plays a key role in these bounds. The ORF mechanism requires  $m \leq d$ , but this will be the case in all our experiments. The pseudocode of the entire FAVOR+ algorithm is given in Appendix B.

Our theoretical results are tightly aligned with experiments. We show in Section 4 that PRFs+ORFs drastically improve accuracy of the approximation of the attention matrix and enable us to reduce  $r$  which results in an accurate as well as space and time efficient mechanism which we call FAVOR+.

# 3 THEORETICAL RESULTS

We present here the theory of positive orthogonal random features for softmax-kernel estimation. All these results can be applied also to the Gaussian kernel, since as explained in the previous section, one can be obtained from the other by renormalization (see: Section 2.3). All proofs and additional more general theoretical results with a discussion are given in the Appendix.

Lemma 2 (positive (hyperbolic) versus trigonometric random features). The following is true:

$$
\begin{array}{l} \mathrm {M S E} (\widehat {\mathrm {S M}} _ {m} ^ {\mathrm {t r i g}} (\mathbf {x}, \mathbf {y})) = \frac {1}{2 m} \exp (\| \mathbf {x} + \mathbf {y} \| ^ {2}) \mathrm {S M} ^ {- 2} (\mathbf {x}, \mathbf {y}) (1 - \exp (- \| \mathbf {x} - \mathbf {y} \| ^ {2})) ^ {2}, \\ \operatorname {M S E} \left(\widehat {\mathrm {S M}} _ {m} ^ {+} (\mathbf {x}, \mathbf {y})\right) = \frac {1}{m} \exp \left(\| \mathbf {x} + \mathbf {y} \| ^ {2}\right) \mathrm {S M} ^ {2} (\mathbf {x}, \mathbf {y}) \left(1 - \exp \left(- \| \mathbf {x} + \mathbf {y} \| ^ {2}\right)\right), \tag {8} \\ \mathrm {M S E} (\widehat {\mathrm {S M}} _ {m} ^ {\mathrm {h y p +}} (\mathbf {x}, \mathbf {y})) = \frac {1}{2} (1 - \exp (- \| \mathbf {x} + \mathbf {y} \| ^ {2})) \mathrm {M S E} (\widehat {\mathrm {S M}} _ {m} ^ {+} (\mathbf {x}, \mathbf {y})). \\ \end{array}
$$

for independent random samples  $\omega_{i}$  and where MSE stands for the mean squared error.

Thus, for  $\mathrm{SM}(\mathbf{x},\mathbf{y})\to 0$  we have:  $\mathrm{MSE}(\widehat{\mathrm{SM}}_m^{\mathrm{trig}}(\mathbf{x},\mathbf{y}))\to \infty$  and  $\mathrm{MSE}(\widehat{\mathrm{SM}}_m^+ (\mathbf{x},\mathbf{y}))\to 0$ . Furthermore, the hyperbolic estimator provides additional accuracy improvements that are strictly better than those from  $\widehat{\mathrm{SM}}_{2m}^{+}(\mathbf{x},\mathbf{y}))$  with twice as many random features. The next result shows that the regularized softmax-kernel is in practice an accurate proxy of the softmax-kernel in attention.

Theorem 1 (regularized versus softmax-kernel). Assume that the  $L_{\infty}$ -norm of the attention matrix for the softmax-kernel satisfies:  $\| \mathbf{A}\|_{\infty} \leq C$  for some constant  $C \geq 1$ . Denote by  $\mathbf{A}^{\mathrm{reg}}$  the corresponding attention matrix for the regularized softmax-kernel. The following holds:

$$
\inf  _ {i, j} \frac {\mathbf {A} ^ {\operatorname* {r e g}} (i , j)}{\mathbf {A} (i , j)} \geq 1 - \frac {2}{d ^ {\frac {1}{3}}} + o \left(\frac {1}{d ^ {\frac {1}{3}}}\right), a n d \sup  _ {i, j} \frac {\mathbf {A} ^ {\operatorname* {r e g}} (i , j)}{\mathbf {A} (i , j)} \leq 1. \tag {9}
$$

Furthermore, the latter holds for  $d \geq 2$  even if  $L_{\infty}$ -norm condition is not satisfied, i.e. the regularized softmax-kernel is a universal lower bound for the softmax-kernel.

Consequently, positive random features for SMREG can be used to approximate the softmax-kernel. Our next result shows that orthogonality provably reduces mean squared error of the estimation with positive random features for any dimensionality  $d > 0$  and we explicitly provide the gap.

Theorem 2. If  $\widehat{\mathrm{SM}}_m^{\mathrm{ort} + }(\mathbf{x},\mathbf{y})$  stands for the modification of  $\widehat{\mathrm{SM}}_m^+ (\mathbf{x},\mathbf{y})$  with orthogonal random features (and thus for  $m\leq d$ ), then the following holds for any  $d > 0$ :

$$
\operatorname {M S E} \left(\widehat {\operatorname {S M}} _ {m} ^ {\operatorname {o r t} +} (\mathbf {x}, \mathbf {y})\right) \leq \operatorname {M S E} \left(\widehat {\operatorname {S M}} _ {m} ^ {+} (\mathbf {x}, \mathbf {y})\right) - \left(1 - \frac {1}{m}\right) \frac {2}{d + 2} \operatorname {S M} ^ {2} (\mathbf {x}, \mathbf {y}). \tag {10}
$$

Furthermore, completely analogous result holds for the regularized softmax-kernel SMREG.

For the regularized softmax-kernel, orthogonal features provide additional concentration results - the first exponentially small bounds for probabilities of estimators' tails that are strictly better than for non-orthogonal variants for every  $d > 0$ . Our next result enables us to explicitly estimate the gap.

Theorem 3. Let  $\mathbf{x},\mathbf{y}\in \mathbb{R}^d$ . The following holds for any  $a > \mathrm{SMREG}(\mathbf{x},\mathbf{y})$  and  $m\leq d$ :

$$
\mathbb {P} [ \widehat {\mathrm {S M R E G}} _ {m} ^ {+} (\mathbf {x}, \mathbf {y}) > a ] \leq \exp (- m \mathcal {L} _ {X} (a)), \mathbb {P} [ \widehat {\mathrm {S M R E G}} _ {m} ^ {\mathrm {o r t} +} (\mathbf {x}, \mathbf {y}) > a ] \leq \frac {d}{d + 2} \exp (- m \mathcal {L} _ {X} (a))
$$

where  $\widehat{\mathrm{SMREG}}_m^{\mathrm{ort + }}(\mathbf{x},\mathbf{y})$  stands for the modification of  $\widehat{\mathrm{SMREG}}_m^+ (\mathbf{x},\mathbf{y})$  with ORFs,  $X = \Lambda \exp (\sqrt{d}\frac{\omega^\top}{\|\omega\|_2} (\mathbf{x} + \mathbf{y}))$ ,  $\omega \sim \mathcal{N}(0,\mathbf{I}_d)$ ,  $\Lambda$  is as in Lemma 1 and  $\mathcal{L}_Z$  is a Legendre Transform of  $Z$  defined as:  $\mathcal{L}_Z(a) = \sup_{\theta >0}\log \left(\frac{e^{\theta a}}{M_Z(\theta)}\right)$  for the moment generating function  $M_Z$  of  $Z$ .

We see that ORFs provide exponentially small and sharper bounds for critical regions where softmax-kernel is small. Below we show that even for the  $\mathrm{SM}^{\mathrm{trig}}$  mechanism with ORFs, it suffices to take  $m = \Theta (d\log (d))$  random projections to accurately approximate the attention matrix (thus if not attention renormalization, PRFs would not be needed). In general,  $m$  depends on the dimensionality  $d$  of the embeddings, radius  $R$  of the ball where all queries/keys live and precision parameter  $\epsilon$  (see: Appendix F.6 for additional discussion), but does not depend on input sequence length  $L$ .

Theorem 4 (uniform convergence for attention approximation). Take  $h(\mathbf{x}) = \exp \left(\frac{\|\mathbf{x}\|^2}{2}\right)$ . Assume that  $L_{2}$ -norms of queries/keys are upper-bounded by  $R > 0$ . Define  $l = Rd^{-\frac{1}{4}}$  and take  $h^* = \max_{\mathbf{x} \in B(l)} |h(\mathbf{x})|$ , where  $B(l)$  is a ball of radius  $l$  and centered at 0. Then for any  $\epsilon > 0$ ,  $\delta = \frac{\epsilon}{(h^*)^2}$  and the number of random projections  $m = \Theta \left( \frac{d}{\delta^2} \log \left( \frac{4d^{\frac{3}{4}}R}{\delta} \right) \right)$  the following holds for the attention approximation mechanism leveraging estimators  $\widehat{\mathrm{SM}}^{\mathrm{trig}}$  with ORFs:  $\| \widehat{\mathbf{A}} - \mathbf{A} \|_1 \leq \epsilon$  with any constant probability, where  $\widehat{\mathbf{A}}$  is the approximation of the attention matrix  $\mathbf{A}$ .

# 4 EXPERIMENTS

We implemented our setup on top of pre-existing Transformer training code in Jax (Frostig et al., 2018) optimized with just-in-time (jax.jit) compilation, and complement our theory with empirical evidence to demonstrate the practicality of FAVOR+ in multiple settings. Unless explicitly stated, a Performer replaces only the attention component with our method, while all other components are exactly the same as for the regular Transformer. For shorthand notation, we denote unidirectional/causal modelling as (U) and bidirectional/masked language modelling as (B).

In terms of baselines, we use other Transformer models for comparison, although some of them are restricted to only one case - e.g. Reformer (Kitaev et al., 2020) is only (U), and Linformer (Wang et al., 2020) is only (B). Furthermore, we use PG-19 (Rae et al., 2020) as an alternative (B) pretraining benchmark, as it is made for long-length sequence training compared to the (now publicly unavailable) BookCorpus (Zhu et al., 2015) + Wikipedia dataset used in BERT (Devlin et al., 2018) and Linformer. All model and tokenization hyperparameters are shown in Appendix A.

![](images/4a33564baaf1dc0bc8c15f0988287a6230d42a5955aa905668aacb047af7bf2c.jpg)  
Figure 3: Comparison of Transformer and Performer in terms of forward and backward pass speed and maximum  $L$  allowed. "X" (OPT) denotes the maximum possible speedup achievable, when attention simply returns the V-matrix. Plots shown up to when a model produces an out of memory error on a V100 GPU with 16GB. Vocabulary size used was 256. Best in color.

# 4.1 COMPUTATIONAL COSTS

We compared speed-wise the backward pass of the Transformer and the Performer in (B) setting, as it is one of the main computational bottlenecks during training, when using the regular default size  $(n_{\text{heads}}, n_{\text{layers}}, d_{ff}, d) = (8, 6, 2048, 512)$ , where  $d_{ff}$  denotes the width of the MLP layers.

We observed (Fig. 3) that in terms of  $L$ , the Performer reaches nearly linear time and sub-quadratic memory consumption (since the explicit  $O(L^2)$  attention matrix is not stored). In fact, the Performer achieves nearly optimal speedup and memory efficiency possible, depicted by the "X"-line when attention is replaced with the "identity function" simply returning the V-matrix. The combination of both memory and backward pass efficiencies for large  $L$  allows respectively, large batch training and lower wall clock time per gradient step. Extensive additional results are demonstrated in Appendix E by varying layers, raw attention, and architecture sizes.

# 4.2 SOFTMAX ATTENTION APPROXIMATION ERROR

We further examined the approximation error via FAVOR+ in Fig. 4. We demonstrate that 1. Orthogonal features produce lower error than unstructured (IID) features, 2. Positive features produce lower error than trigonometric sin/cos features. These two empirically validate the PORF mechanism.

![](images/f2d79c148489e8ea79a0611b9ea54827bde503a733e44d1de0af3439dce91fc4.jpg)  
Figure 4: MSE of the approximation output when comparing Orthogonal vs IID features and trigonometric sin/cos vs positive features. We took  $L = 4096$ ,  $d = 16$ , and varied the number of random samples  $m$ . Standard deviations shown across 15 samples of appropriately normalized random matrix input data.

![](images/ab4a2f189311332c0c9d2bc5c1f09d22be3da146f453b76f13e4c4812c178289.jpg)

To further improve overall approximation of attention blocks across multiple iterations which further improves training, random samples should be periodically redrawn (Fig. 5, right). This is a cheap procedure, but can be further optimized (Appendix B.2).

# 4.3 SOFTMAX APPROXIMATION ON TRANSFORMERS

Even if the approximation of the attention mechanism is tight, small errors can easily propagate throughout multiple Transformer layers (e.g. MLPs, multiple heads), as we show in Fig. 14 (Appendix). In other words, the model's Lipschitz constant can easily scale up small attention approximation error, which means that very tight approximations may sometimes be needed. Thus, when applying FAVOR  $(+)$  's softmax approximations on a Transformer model (i.e. "Performer-X- SOFTMAX"), we demonstrate that:

1. Backwards compatibility with pretrained models is available as a benefit from softmax approximation, via small finetuning (required due to error propagation) even for trigonometric features (Fig. 5, left) on the LM1B dataset (Chelba et al., 2014). However, when on larger dataset PG-19, 2. Positive (POS) softmax features (with redrawing) become crucial for achieving performance matching regular Transformers (Fig. 5, right).

![](images/1383ebf918b962faf5ccd8e0dfc166d3e9838575a45965bf54ae1bfb3570ca7e.jpg)  
Figure 5: We transferred the original pretrained Transformer's weights into the Performer, which produces an initial non-zero 0.07 accuracy (dotted orange line), but quickly recovers accuracy in a small fraction of the original number of gradient steps. However on PG-19, Trigonometric (TRIG) softmax approximation becomes highly unstable (full curve in Appendix D.2), while positive features (POS) (without redrawing) and Linformer (which also approximates softmax) even with redrawn projections, plateau at the same perplexity. Positive softmax with feature redrawing is necessary to match the Transformer, with SMREG (regularization from Sec. 3) allowing faster convergence. Additional ablation studies over many attention kernels, showing also that trigonometric random features lead even to NaN values in training are given in Appendix D.3.

![](images/010ca0a61ee50e9f695e0b30596b15ac55013016350b2fe6dbad71a2d246493d.jpg)

# 4.4 MULTIPLE LAYER TRAINING FOR PROTEINS

We further benchmark the Performer on both (U) and (B) cases by training a 36-layer model using protein sequences from the Jan. 2019 release of TrEMBL (Consortium, 2019), similar to (Madani et al., 2020). In Fig. 6, the Reformer and Linformer significantly drop in accuracy on the protein dataset. Furthermore, the usefulness of generalized attention is evidenced by Performer-RELU (taking  $f = \mathrm{ReLU}$  in Equation 5) achieving the highest accuracy in both (U) and (B) cases. Our proposed softmax approximation is also shown to be tight, achieving the same accuracy as the exact-softmax Transformer and confirming our theoretical claims from Section 3.

![](images/3b9f291e0b30bc2098b523095859c67ac225e05a72d38d55728b81f859da6bbd.jpg)  
Figure 6: Train = Dashed, Validation = Solid. For TrEMBL, we used the exact same model parameters  $(n_{\text{heads}}, n_{\text{layers}}, d_{ff}, d) = (8, 36, 1024, 512)$  from (Madani et al., 2020) for all runs. For fairness, all TrEMBL experiments used 16x16 TPU-v2's. Batch sizes were maximized for each separate run given the compute constraints. Hyperparameters can be found in Appendix A. Extended results including dataset statistics, out of distribution evaluations, and visualizations, can be found in Appendix C.

![](images/3ee19f304f537c89d62efbf208ba56c5e7723e68976645f799c77d218a1c7613.jpg)

# 4.5 LARGE LENGTH TRAINING - COMMON DATASETS

On the standard (U) ImageNet64 benchmark from (Parmar et al., 2018) with  $L = 12288$  which is unfeasible for regular Transformers, we set all models to use the same ( $n_{\text{heads}}, d_{ff}, d$ ) but varying  $n_{\text{layers}}$ . Performer/6-layers matches the Reformer/12-layers, while the Performer/12-layers matches the Reformer/24-layers (Fig. 7: left). Depending on hardware (TPU or GPU), we also found that the Performer can be 2x faster than the Reformer via Jax optimizations for the (U) setting.

For a proof of principle study, we also create an initial protein benchmark for predicting interactions among groups of proteins by concatenating protein sequences to length  $L = 8192$  from TrEMBL, long enough to model protein interaction networks without the large sequence alignments required by existing methods (Cong et al., 2019). In this setting, a regular Transformer overloads memory even at a batch size of 1 per chip, by a wide margin. Thus as a baseline, we were forced to use a significantly smaller variant, reducing to  $(n_{\text{heads}}, n_{\text{layers}}, d_{ff}, d) = (8, \{1, 2, 3\}, 256, 256)$ . Meanwhile, the Performer trains efficiently at a batch size of 8 per chip using the standard (8, 6, 2048, 512) architecture. We see in Fig. 7 (right subfigure) that the smaller Transformer  $(n_{\text{layer}} = 3)$  is quickly bounded at  $\approx 19\%$ , while the Performer is able to train continuously to  $\approx 24\%$ .

![](images/c305e56a8c85d1cb8d7a54c1200ce4db53dbd545291b7fd264099ee5bf7cbb5e.jpg)  
Figure 7: Train = Dashed, Validation = Solid. For ImageNet64, all models used the standard  $(n_{\text{heads}}, d_{ff}, d) = (8, 2048, 512)$ . We further show that our positive softmax approximation achieves the same performance as ReLU in Appendix D.2. For concatenated TrEMBL, we varied  $n_{\text{layers}} \in \{1, 2, 3\}$  for the smaller Transformer. Hyperparameters can be found in Appendix A.

![](images/f9c612ebebb9a41406018faff707b6410021a88d4594cc5b1b5828dbe0d01176.jpg)

# 5 CONCLUSION

We presented Performer, a new type of Transformer, relying on our Fast Attention Via positive Orthogonal Random features (FAVOR+) mechanism to significantly improve space and time complexity of regular Transformers. Our mechanism provides to our knowledge the first effective unbiased estimation of the original softmax-based Transformer with linear space and time complexity and opens new avenues in the research on Transformers and the role of non-sparsifying attention mechanisms.

# REFERENCES

Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, and Quoc V. Le. Attention augmented convolutional networks. CoRR, abs/1904.09925, 2019. URL http://arxiv.org/abs/1904.09925.  
Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. CoRR, abs/2004.05150, 2020. URL https://arxiv.org/abs/2004.05150.  
William Chan, Chitwan Sahara, Geoffrey E. Hinton, Mohammad Norouzi, and Navdeep Jaitly. Imputer: Sequence modelling via imputation and dynamic programming. CoRR, abs/2002.08926, 2020. URL https://arxiv.org/abs/2002.08926.  
Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Philipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling. In *INTERSPEECH* 2014, 15th Annual Conference of the International Speech Communication Association, Singapore, September 14-18, 2014, pp. 2635-2639, 2014.  
Ciprian Chelba, Mia Xu Chen, Ankur Bapna, and Noam Shazeer. Faster transformer decoding: N-gram masked self-attention. CoRR, abs/2001.04589, 2020. URL https://arxiv.org/abs/2001.04589.  
Mia Xu Chen, Orhan First, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George F. Foster, Llion Jones, Mike Schuster, Noam Shazeer, Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Zhifeng Chen, Yonghui Wu, and Macduff Hughes. The best of both worlds: Combining recent advances in neural machine translation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 1: Long Papers, pp. 76-86. Association for Computational Linguistics, 2018. doi: 10.18653/v1/P18-1008. URL https://www.aclweb.org/anthology/P18-1008/.  
Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. CoRR, abs/1904.10509, 2019. URL http://arxiv.org/abs/1904.10509.  
Krzysztof Choromanski, Carlton Downey, and Byron Boots. Initialization matters: Orthogonal predictive state recurrent neural networks. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018a. URL https://openreview.net/forum?id=HJJ23bW0b.  
Krzysztof Choromanski, Mark Rowland, Tamás Sarlós, Vikas Sindhwani, Richard E. Turner, and Adrian Weller. The geometry of random features. In International Conference on Artificial Intelligence and Statistics, AISTATS 2018, 9-11 April 2018, Playa Blanca, Lanzarote, Canary Islands, Spain, volume 84 of Proceedings of Machine Learning Research, pp. 1-9. PMLR, 2018b. URL http://proceedings.mlr.press/v84/choromanski18a.html.  
Krzysztof Choromanski, Aldo Pacchiano, Jeffrey Pennington, and Yunhao Tang. KAMA-NNs: Low-dimensional rotation based neural networks. In The 22nd International Conference on Artificial Intelligence and Statistics, AISTATS 2019, 16-18 April 2019, Naha, Okinawa, Japan, volume 89 of Proceedings of Machine Learning Research, pp. 236-245. PMLR, 2019a. URL http://proceedings.mlr.press/v89/choromanski19a.html.  
Krzysztof Choromanski, Mark Rowland, Wenyu Chen, and Adrian Weller. Unifying orthogonal Monte Carlo methods. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pp. 1203-1212. PMLR, 2019b. URL http://proceedings.mlr.press/v97/choromanski19a.html.  
Krzysztof Marcin Choromanski, Mark Rowland, and Adrian Weller. The unreasonable effectiveness of structured random orthogonal embeddings. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, 4-9 December 2017, Long Beach, CA, USA, pp. 219-228, 2017.  
Qian Cong, Ivan Anishchenko, Sergey Ovchinnikov, and David Baker. Protein interaction networks revealed by proteome coevolution. Science, 365(6449):185-189, 2019.

UniProt Consortium. Uniprot: a worldwide hub of protein knowledge. *Nucleic acids research*, 47 (D1):D506–D515, 2019.  
Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. Introduction to Algorithms, 3rd Edition. MIT Press, 2009. ISBN 978-0-262-03384-8. URL http://mitpress.mit.edu/books/introduction-algorithms.  
Zihang Dai, Zhilin Yang, Yiming Yang, William W. Cohen, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformer-XL: Language modeling with longer-term dependency, 2019. URL https://openreview.net/forum?id=HJePnoOcYm.  
Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=HyzdRiR9Y7.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805, 2018. URL http://arxiv.org/abs/1810.04805.  
Yilun Du, Joshua Meier, Jerry Ma, Rob Fergus, and Alexander Rives. Energy-based models for atomic-resolution protein conformations. arXiv preprint arXiv:2004.13167, 2020.  
Ahmed Elnaggar, Michael Heinzinger, Christian Dallago, and Burkhard Rost. End-to-end multitask learning, from protein language to protein features without alignments. bioRxiv, pp. 864405, 2019.  
Roy Frostig, Matthew Johnson, and Chris Leary. Compiling machine learning programs via high-level tracing. In Conference on Machine Learning and Systems 2018, 2018. URL http://www.sysml.cc/doc/2018/146.pdf.  
Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang, and Hanqing Lu. Dual attention network for scene segmentation. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pp. 3146-3154, 2019.  
Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, and Ruoming Pang. Conformer: Convolution-augmented transformer for speech recognition, 2020.  
Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Ian Simon, Curtis Hawthorne, Noam Shazeer, Andrew M. Dai, Matthew D. Hoffman, Monica Dinculescu, and Douglas Eck. Music transformer: Generating music with long-term structure. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=rJe4ShAcF7.  
John Ingraham, Vikas Garg, Regina Barzilay, and Tommi Jaakkola. Generative models for graph-based protein design. In Advances in Neural Information Processing Systems, pp. 15794-15805, 2019.  
Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. CoRR, abs/2006.16236, 2020. URL https://arxiv.org/abs/2006.16236.  
Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id= rkgNKkHtvB.  
Olga Kovaleva, Alexey Romanov, Anna Rogers, and Anna Rumshisky. Revealing the dark secrets of bert. arXiv preprint arXiv:1908.08593, 2019.  
Taku Kudo and John Richardson. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. CoRR, abs/1808.06226, 2018. URL http://arxiv.org/abs/1808.06226.

Richard E. Ladner and Michael J. Fischer. Parallel prefix computation. J. ACM, 27(4):831-838, October 1980. ISSN 0004-5411. doi: 10.1145/322217.322232. URL https://doi.org/10.1145/322217.322232.  
Han Lin, Haoxian Chen, Tianyi Zhang, Clément Laroche, and Krzysztof Choromanski. Demystifying orthogonal Monte Carlo and beyond. CoRR, abs/2005.13590, 2020.  
Haoneng Luo, Shiliang Zhang, Ming Lei, and Lei Xie. Simplified self-attention for transformer-based end-to-end speech recognition. CoRR, abs/2005.10463, 2020. URL https://arxiv.org/abs/2005.10463.  
Ali Madani, Bryan McCann, Nikhil Naik, Nitish Shirish Keskar, Namrata Anand, Raphael R. Eguchi, Po-Ssu Huang, and Richard Socher. Progen: Language modeling for protein generation. CoRR, abs/2004.03497, 2020. URL https://arxiv.org/abs/2004.03497.  
Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pp. 4052-4061. PMLR, 2018. URL http://proceedings.mlr.press/v80/parmar18a.html.  
Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SylKikSYDH.  
Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems 20, Proceedings of the Twenty-First Annual Conference on Neural Information Processing Systems, Vancouver, British Columbia, Canada, December 3-6, 2007, pp. 1177-1184. Curran Associates, Inc., 2007. URL http://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.  
Alexander Rives, Siddharth Goyal, Joshua Meier, Demi Guo, Myle Ott, C. Zitnick, Jerry Ma, and Rob Fergus. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. bioArxiv, 04 2019. doi: 10.1101/622803.  
Mark Rowland, Jiri Hron, Yunhao Tang, Krzysztof Choromanski, Tamás Sarlós, and Adrian Weller. Orthogonal estimation of Wasserstein distances. In The 22nd International Conference on Artificial Intelligence and Statistics, AISTATS 2019, 16-18 April 2019, Naha, Okinawa, Japan, volume 89 of Proceedings of Machine Learning Research, pp. 186-195. PMLR, 2019. URL http://proceedings.mlr.press/v89/rowland19a.html.  
Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. CoRR, abs/2003.05997, 2020. URL https://arxiv.org/abs/2003.05997.  
Zhuoran Shen, Mingyuan Zhang, Shuai Yi, Junjie Yan, and Haiyu Zhao. Factorized attention: Self-attention with linear complexities. CoRR, abs/1812.01243, 2018. URL http://arxiv.org/abs/1812.01243.  
Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, and Ruslan Salakhutdinov. Transformer dissection: An unified understanding for transformer's attention via the lens of kernel. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4335-4344, 2019.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30, pp. 5998-6008. Curran Associates, Inc., 2017. URL http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf.  
Jesse Vig. A multiscale visualization of attention in the transformer model. arXiv preprint arXiv:1906.05714, 2019.

Jesse Vig and Yonatan Belinkov. Analyzing the structure of attention in a transformer language model. CoRR, abs/1906.04284, 2019. URL http://arxiv.org/abs/1906.04284.  
Jesse Vig, Ali Madani, Lav R. Varshney, Caiming Xiong, Richard Socher, and Nazneen Fatema Rajani. Bertology meets biology: Interpreting attention in protein language models. CoRR, abs/2006.15222, 2020. URL https://arxiv.org/abs/2006.15222.  
Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, pp. 2692-2700, 2015.  
Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. CoRR, abs/2006.04768, 2020. URL https://arxiv.org/abs/2006.04768.  
Tong Xiao, Yinqiao Li, Jingbo Zhu, Zhengtao Yu, and Tongran Liu. Sharing attention weights for fast transformer. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019, pp. 5292-5298. ijcai.org, 2019. doi: 10.24963/ijcai.2019/735. URL https://doi.org/10.24963/ijcai.2019/735.  
Felix X. Yu, Ananda Theertha Suresh, Krzysztof Marcin Choromanski, Daniel N. Holtmann-Rice, and Sanjiv Kumar. Orthogonal random features. In Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pp. 1975-1983, 2016.  
Vinicius Flores Zambaldi, David Raposo, Adam Santoro, Victor Bapst, Yujia Li, Igor Babuschkin, Karl Tuyls, David P. Reichert, Timothy P. Lillicrap, Edward Lockhart, Murray Shanahan, Victoria Langston, Razvan Pascanu, Matthew Botvinick, Oriol Vinyals, and Peter W. Battaglia. Deep reinforcement learning with relational inductive biases. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019, 2019.  
Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pp. 19-27, 2015. doi: 10.1109/ICCV.2015.11. URL https://doi.org/10.1109/ICCV.2015.11.
