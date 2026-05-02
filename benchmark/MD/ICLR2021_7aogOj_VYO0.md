# DO NOT LET PRIVACY OVERBILL UTILITY: GRADIENT EMBEDDING PERTURBATION FOR PRIVATE LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

The privacy leakage of the model about the training data can be bounded in the differential privacy mechanism. However, for meaningful privacy parameters, a differentially private model degrades the utility drastically when the model comprises a large number of trainable parameters. In this paper, we propose an algorithm Gradient Embedding Perturbation (GEP) towards training differentially private deep models with decent accuracy. Specifically, in each gradient descent step, GEP first projects individual private gradient into a non-sensitive anchor subspace, producing a low-dimensional gradient embedding and a small-norm residual gradient. Then, GEP perturbs the low-dimensional embedding and the residual gradient separately according to the privacy budget. Such a decomposition permits a small perturbation variance, which greatly helps to break the dimensional barrier of private learning. With GEP, we achieve decent accuracy with low computation cost and modest privacy guarantee for deep models. Especially, with privacy bound  $\epsilon = 8$ , we achieve  $74.9\%$  test accuracy on CIFAR10 and  $95.1\%$  test accuracy on SVHN, significantly improving over existing results.

# 1 INTRODUCTION

Recent works have shown that the trained model may leak/memorize the information of its training set (Fredrikson et al., 2015; Wu et al., 2016; Shokri et al., 2017; Hitaj et al., 2017), which raises privacy issue when the model are trained with sensitive data. Differential privacy (DP) mechanism provides a way to quantitatively measure and upper bound such information leakage. It theoretically ensures that the influence of any individual sample is negligible with the DP parameter  $\epsilon$  or  $(\epsilon, \delta)$ . Moreover, it has been observed that differentially private models can also resist model inversion attack (Carlini et al., 2019), membership inference attack (Rahman et al., 2018; Bernau et al., 2019), gradient matching attack (Zhu et al., 2019), and data poisoning attack (Ma et al., 2019).

One popular way to achieve differentially private machine learning is to perturb the training process with noise (Song et al., 2013; Bassily et al., 2014; Shokri & Shmatikov, 2015; Wu et al., 2017; Fukuchi et al., 2017; Iyengar et al., 2019; Phan et al., 2020). Specifically, gradient perturbation perturbs the gradient at each iteration of (stochastic) gradient descent algorithm and guarantees the privacy of the final model via composition property of DP. It is worthy to note that gradient perturbation does not assume (strongly) convex objective and hence is applicable to various settings (Abadi et al., 2016; Wang et al., 2017; Lee & Kifer, 2018; Jayaraman et al., 2018; Wang & Gu, 2019). Specifically, for given gradient sensitivity  $S$ , a general form of gradient perturbation is to add an isotropic Gaussian noise  $z$  to the gradient  $g \in \mathbb{R}^p$  independently for each step,

$$
\tilde {\boldsymbol {g}} = \boldsymbol {g} + \boldsymbol {z}, \text {w h e r e} \boldsymbol {z} \sim \mathcal {N} \left(0, \sigma^ {2} S ^ {2} \boldsymbol {I} _ {p \times p}\right). \tag {1}
$$

One can set proper variance  $\sigma^2$  to make each update differentially private with parameter  $(\epsilon, \delta)$ . It is easy to see that the intensity of the added noise  $\mathbb{E}[\|z\|^2]$  scales linearly with the model dimension  $p$ . This indicates that as the model becomes larger, the useful signal, i.e., gradient, would be submerged in the added noise (see Figure 1). This dimensional barrier restricts the utility of deep learning models trained with gradient perturbation.

The dimensional barrier is attributed to the fact that the added noise is isotropic while the gradients live on a very low dimensional manifold, which has been observed in (Gur-Ari et al., 2018; Vogels

![](images/76f423bb19c9d3bf76f5b4b419d4264e502096a50f5468bd4c455ebcb4ff4b79.jpg)  
Figure 1: Noise norm vs gradient norm of ResNet20 at initialization. The noise variance is chosen such that SGD satisfies  $(5,10^{-5})$ -DP after 90 epochs in Abadi et al. (2016).

![](images/a6111a4b5e83917dc03811e15b5b54b8dc5b00496a89983b31b60f3ee1311e7e.jpg)  
Figure 2: Stable rank  $\| \cdot \| _F / \| \cdot \|$  of batch gradient matrix of given groups (with  $p$  parameters). The setting is ResNet20 on CIFAR-10. The stable rank is small throughout training.

et al., 2019; Gooneratne et al., 2020; Li et al., 2020) and is also verified in Figure 2 for the gradients of a 20-layer ResNet (He et al., 2016). Hence to limit the noise energy, it is natural to think

"Can we reduce the dimension of gradients first and then add the isotropic noise onto a low-dimensional gradient embedding?"

The answer is affirmative. We propose a new algorithm Gradient Embedding Perturbation (GEP), illustrated in Figure 3. Specifically, we first compute anchor gradients on some non-sensitive auxiliary data, and identify an anchor subspace that is spanned by several top principal components of the anchor gradient matrix. Then we project the private gradients into the anchor subspace and obtain low-dimensional gradient embeddings and small-norm residual gradients. Finally, we perturb the gradient embedding and residual gradient separately according to the sensitivities and privacy budget.

We intuitively argue why GEP could reduce the perturbation variance and achieve good utility for large models. First, because the gradient embedding has a very low dimension, the added isotropic noise on embedding has small energy that scales linearly only with the subspace dimension. Second, if the anchor subspace can cover most of the gradient information, the residual gradient, though high dimensional, should have small magnitude, which permits smaller added noise to guarantee the same level privacy because of the reduced sensitivity. Overall, we can use a much lower perturbation compared with the original gradient perturbation to guarantee the same level of privacy.

We emphasize several properties of GEP. First, the non-sensitive auxiliary data assumption is weak. In fact, GEP only requires a small number of non-sensitive unlabeled data following a similar feature distribution as the private data, which often exist even for learning on sensitive data. In our experiments, we use a few unlabeled samples from ImageNet to serve as auxiliary data for MNIST, SVHN, and CIFAR-10. This assumption is much weaker than the public data assumption in previous works (Papernot et al., 2017; 2018; Alon et al., 2019; Wang & Zhou, 2020), where the public data should follow exactly the same distribution as the private data. Second, GEP produces an unbiased estimator of the target gradient because of releasing both the perturbed gradient embedding and the perturbed residual gradient, which turns out to be critical for good utility. Third, the additional computational cost of GEP is low. We use power method to estimate the principal components of anchor gradients, achievable with a few matrix multiplications. The fact that GEP is not sensitive to the choices of subspace dimension further allows a very efficient implementation.

Compared with existing works of differentially private machine learning, our contribution can be summarized as follows: (1) we propose a novel algorithm GEP that achieves good utility for large models with modest differential privacy guarantee; (2) we show that GEP returns an unbiased estimator of target private gradient with much lower perturbation variance than original gradient perturbation; (3) we demonstrate that GEP achieves state-of-the-art utility in differentially private learning with three benchmark datasets. Specifically, for  $\epsilon = 8$ , GEP achieves  $74.9\%$  test accuracy on CIFAR-10 with a ResNet20 model. To the best of our knowledge, GEP is the first algorithm that can achieve such utility with training deep models from scratch for a "single-digit" privacy budget<sup>1</sup>.

![](images/92c40af95e47b9a0393e550654902ad81a488793822c597ace7a4c851078fe28.jpg)  
Figure 3: Overview of the proposed GEP approach. 1) We estimate an anchor subspace on some non-sensitive data; 2) We project the private gradients into the anchor subspace, producing low-dimensional embeddings and residual gradients; 3) We perturb the gradient embedding and residual gradient separately to guarantee differential privacy. The auxiliary data are only required to share similar features as the private data. In our experiments, we use 2000 images from ImageNet as auxiliary data for MNIST, SVHN, and CIFAR-10 datasets.

# 1.1 RELATED WORK

Existing works studying differentially private machine learning in high-dimensional setting can be roughly categorized into two sets. One is treating the optimization of the machine learning objective as a whole mechanism and adding noise into this process. The other one is based on the knowledge transfer of machine learning models, which trains a differentially private publishable student model with private signals from teacher models. Nextly, we review them one by one.

Differentially private convex optimization in high-dimensional setting has been studied extensively over the years (Kifer et al., 2012; Thakurta & Smith, 2013; Talwar et al., 2015; Wang & Xu, 2019; Wang & Gu, 2019). Although these methods demonstrate good utility on some convex settings, their analyses can not be directly applied to non-convex setting. Right before the submission, we note two independent and concurrent works (Zhou et al., 2020; Kairouz et al., 2020) that also leverage the gradient redundancy to reduce the added noise. Specifically, Kairouz et al. (2020) track historical gradients to do dimension reduction for private AdaGrad. Zhou et al. (2020) requires gradients on some public data and then project the noisy gradients into a public subspace at each update. One core difference between these two works and GEP is that we introduce residual gradient perturbation and GEP produces an unbiased estimator of the private gradients, which is essential for achieving the superior utility. Moreover, we weaken the auxiliary data assumption and introduce several designs that significantly boost the efficiency and applicability of GEP.

One recent progress towards training arbitrary models with differential privacy is Private Aggregation of Teacher Ensembles (PATE) (Papernot et al., 2017; 2018; Jordon et al., 2019). PATE first trains independent teacher models on disjoint shards of private data. Then it trains a student model with privacy guarantee by distilling noisy predictions of teacher models on some public samples. In comparison, GEP only requires some non-sensitive data that have similar natural features as the private data while PATE requires the public data follow exactly the same distribution as the private data and in practice it uses a portion of the test data to serve as public data. Moreover, GEP demonstrates better performance than PATE especially for complex datasets, e.g., CIFAR-10, because GEP can train the model with the whole private data rather than a small shard of data.

# 2 PRELIMINARIES

We introduce some notations and definitions. We use bold lowercase letters, e.g.,  $\pmb{v}$ , and bold capital letters, e.g.,  $M$ , to denote vectors and matrices, respectively. The  $L^2$  norm of a vector  $\pmb{v}$  is denoted by  $\| \pmb{v} \|$ . The spectral norm and the Frobenius norm of a matrix  $M$  are denoted by  $\| M \|$  and  $\| M \|_F$ , respectively. A sample  $d = (x, y)$  consists of feature  $\pmb{x}$  and label  $y$ . A dataset  $\mathbb{D}$  is a collection of individual samples. A dataset  $\mathbb{D}'$  is said to be a neighboring dataset of  $\mathbb{D}$  if they differ in a single sample, denoted as  $\mathbb{D} \sim \mathbb{D}'$ . Differential privacy ensures that the outputs of an algorithm on neighboring datasets have approximately indistinguishable distributions.

Definition 1  $((\epsilon, \delta)$ -DP (Dwork et al., 2006a,b)). A randomized mechanism  $\mathcal{M}$  guarantees  $(\epsilon, \delta)$ -differential privacy if for any two neighboring input datasets  $\mathbb{D} \sim \mathbb{D}'$  and for any subset of outputs  $S$  it holds that  $Pr[\mathcal{M}(\mathbb{D}) \in S] \leq e^{\epsilon} Pr[\mathcal{M}(\mathbb{D}') \in S] + \delta$ .

By its definition,  $(\epsilon, \delta)$ -DP controls the maximum influence that any individual sample can produce. One can adjust the privacy parameters to trade off between privacy and utility. Differential privacy is immune to post-processing (Dwork et al., 2014), i.e., any function applied on the output of a differentially private algorithm would not increase the privacy loss as long as it does not have new interaction with the private dataset. Differential privacy also allows composition, i.e., the composition of a series of differentially private mechanisms is also differentially private but with different parameters. Several variants of  $(\epsilon, \delta)$ -DP have been proposed (Bun & Steinke, 2016; Dong et al., 2019) to address certain weakness of  $(\epsilon, \delta)$ -DP, e.g., they achieve better composition property. In this work, we use Rényi differential privacy (Mironov, 2017) to track the privacy loss and then convert it to  $(\epsilon, \delta)$ -DP.

Suppose that there is a private dataset  $\mathbb{D} = \{(\pmb{x}_i, y_i)\}_{i=1}^n$  with  $n$  samples. We want to train a model  $f$  to learn the mapping in  $\mathbb{D}$ . Specifically,  $f$  takes  $\pmb{x}$  as input and outputs a label  $y$ , and  $f$  has parameter  $\theta \in \mathbb{R}^p$ . The training objective is to minimize an empirical risk  $\frac{1}{n} \sum_{i=1}^{n} \ell(f(\pmb{x}_i), y_i)$ , where  $\ell(\cdot, \cdot)$  is a loss function. We further assume that there is an auxiliary dataset  $\mathbb{D}_a = \{(\tilde{\pmb{x}}_j, \tilde{y}_j)\}_{j=1}^m$  that  $\tilde{\pmb{x}}$  shares similar features as  $\pmb{x}$  in  $\mathbb{D}$  while  $\tilde{y}$  could be random.

# 3 GRADIENT EMBEDDING PERTURBATION

An overview of GEP is given in Figure 3. GEP has three major ingredients: 1) first, estimate an anchor subspace that contains the principal components of some non-sensitive anchor gradients via power method; 2) then, project private gradients into the anchor subspace and produce low-dimensional embeddings of private gradients and residual gradients; 3) finally, perturb gradient embedding and residual gradient separately to establish differential privacy guarantee. In Section 3.1, we present the GEP algorithm in detail. In Section 3.2, we show that GEP outputs an unbiased estimator of target private gradient and permits much lower perturbation variance than the original gradient perturbation.

# 3.1 THE GEP ALGORITHM AND ITS PRIVACY ANALYSIS

The pseudocode of GEP is presented in Algorithm 1. For convenience, we write a set of gradients and a set of basis vectors as matrices with each row being one gradient/basis vector.

The anchor subspace is constructed as follows. We first compute the gradients of the model on an auxiliary dataset  $\mathbb{D}_a$  with  $m$  samples, which is referred to as the anchor gradients  $G_{a}\in \mathbb{R}^{m\times p}$ . We then use the power method to estimate the principal components of  $G_{a}$  to construct a subspace basis  $B\in \mathbb{R}^{k\times p}$ , which is referred to as the anchor subspace. All these matrices are publishable because  $\mathbb{D}_a$  is non-sensitive. We expect that the anchor subspace  $B$  can cover most energy of private gradients when the auxiliary data are not far from private data and  $m,k$  are reasonably large.

Suppose that the private gradients are  $\pmb{G} \in \mathbb{R}^{n \times p}$ . Then, we project the private gradients into the anchor subspace  $\pmb{B}$ . The projection produces low-dimensional embeddings  $\pmb{W} = \pmb{G}\pmb{B}^T$  and residual gradients  $\pmb{R} = \pmb{G} - \pmb{G}\pmb{B}^T\pmb{B}$ . The magnitude of residual gradients is usually much smaller than original gradient even when  $k$  is small because of the gradient redundancy.

Then, we aggregate the gradient embeddings and the residual gradients, respectively. We perturb the aggregated embedding and the aggregated residual gradient respectively to guarantee certain differential privacy. Finally, we release the perturbed embedding and the perturbed residual gradient and construct an unbiased estimator of the private gradient:  $\tilde{\boldsymbol{v}}\coloneqq (\tilde{\boldsymbol{w}}^T\boldsymbol {B} + \tilde{\boldsymbol{r}}) / n$ . This construction process does not resulting in additional privacy loss because of DP's post-processing property. The privacy analysis of the whole process of GEP is given in Theorem 3.1.

Theorem 3.1. Let  $S_{1}$  and  $S_{2}$  be the sensitivity of  $\mathbf{w}$  and  $\mathbf{r}$ , respectively, the output of Algorithm 1 satisfies  $(\epsilon, \delta)$ -DP for any  $\delta \in (0,1)$  and  $\epsilon \leq 2\log(1/\delta)$  if we choose  $\sigma_{1} \geq 2S_{1}\sqrt{2\log(1/\delta)}/\epsilon$  and  $\sigma_{2} \geq 2S_{2}\sqrt{2\log(1/\delta)}/\epsilon$ .

A common practice to control sensitivity is to clip the individual gradients with a pre-defined threshold. The privacy loss of GEP consists of two parts: the privacy loss incurred by releasing the perturbed

Algorithm 1: Gradient embedding perturbation  
1: Input: anchor gradients  $G_{a} \in \mathbb{R}^{m \times p}$ ; number of basis vectors  $k$ ; private gradients  $G \in \mathbb{R}^{n \times p}$ ; standard deviation of perturbation noise  $\sigma_{1}, \sigma_{2}$ ; number of power iterations  $t$ .  
2: //First stage: Compute an orthonormal basis for the anchor subspace.  
3: Initialize  $B \in \mathbb{R}^{k \times p}$  randomly.  
4: for  $i = 1$  to  $t$  do  
5: Compute  $A = G_{a} B^{T}$  and  $B = A^{T} G_{a}$ .  
6: Orthogonalize  $B$  and normalize row vectors.  
7: end for  
8: Delete  $G_{a}$  to free memory.  
9: //Second stage: project the private gradients  $G$  into anchor subspace  $B$   
10: Compute gradient embeddings  $W = G B^{T}$  and residual gradients  $R = G - W B$ .  
11: //Third stage: perturb gradient embedding and residual gradient separately  
12: Perturb gradient embedding with noise  $z_{1} \sim \mathcal{N}(0, \sigma_{1}^{2} I_{k \times k})$ :  $w := \sum_{i} W_{i,:}$ ,  $\tilde{w} := w + z_{1}$ .  
13: Perturb residual gradient with noise  $z_{2} \sim \mathcal{N}(0, \sigma_{2}^{2} I_{p \times p})$ :  $r := \sum_{i} R_{i,:}$ ,  $\tilde{r} := r + z_{2}$ .  
14: Return  $\tilde{v} := (\tilde{w}^{T} B + \tilde{r}) / n$ .

embedding and the privacy loss incurred by releasing the perturbed residual gradient. We compose these two parts via the Rényi differential privacy and convert it to  $(\epsilon, \delta)$ -DP.

We highlight two ingredients of Algorithm 1 that make GEP widely applicable and implementable with small computational cost. Firstly, auxiliary non-sensitive data do not have to be the same source as the private data and the auxiliary data can be randomly labeled. This non-sensitive data assumption is very weak and easy to satisfy in practical scenarios. To understand why random label works, a quick example is that for the least squares regression problem the individual gradient is aligned with the feature vector while the label only scales the length but does not change the direction. This auxiliary data assumption avoids conducting principal component analysis (PCA) on private gradients, which requires releasing private high-dimensional basis vectors and hence introduces large privacy loss. Secondly, we use power method (Panju, 2011; Vogels et al., 2019) to approximately estimate the principal components. The new operation we introduce is standard matrix multiplication that enjoys efficient implementation on GPU. The computational complexity of each power iteration is  $2mkp$ , where  $p$  is the number of model parameters,  $m$  is the number of anchor gradients and  $k$  is the number of subspace basis vectors.

Curious readers may wonder if we can use random projection to reduce the dimensionality as Johnson-Lindenstrauss Lemma (Dasgupta & Gupta, 2003) guarantees that one can preserve the pairwise distance between any two points after projecting into a random subspace of much lower dimension. However, preserving the pairwise distance is not sufficient for high quality gradient reconstruction, which is verified by empirical observation.

# 3.2 AN ANALYSIS ON THE PERTURBATION VARIANCE OF GEP

Let  $\pmb{g} \coloneqq \frac{1}{n}\sum_{i}\pmb{G}_{i,}$  be the target private gradient and  $\tilde{\pmb{v}}$  be the output of GEP. For convenience and without loss of generality, we assume the loss function is 1-Lipschitz continuous. For a given anchor subspace  $B$ , the residual gradients are defined as  $\pmb{R} \coloneqq \pmb{G} - \pmb{GB}^T\pmb{B}$ . We analyze the expected distance between  $\tilde{\pmb{v}}$  and  $\pmb{g}$  in Theorem 3.2.

Theorem 3.2. For given privacy parameters  $\epsilon$  and  $\delta$ , the output of  $GEP\tilde{v}$  satisfies

$$
\mathbb {E} [ \tilde {\boldsymbol {v}} ] = \boldsymbol {g}, \quad \mathbb {E} [ \| \tilde {\boldsymbol {v}} - \boldsymbol {g} \| ^ {2} ] \leq k \sigma^ {2} / n ^ {2} + p \sigma^ {2} S ^ {2} / n ^ {2},
$$

where  $\sigma = 2\sqrt{2\log(1 / \delta)} /\epsilon$  and  $S = \max_{i}\| \pmb{R}_{i,:}\|$  is the sensitivity of residual gradient.

Let the output of original gradient perturbation be  $\tilde{\pmb{g}}$ . The perturbation variance of  $\tilde{\pmb{g}}$  is  $\mathbb{E}[\| \tilde{\pmb{g}} - \pmb{g}\|^2] = p\sigma^2 / n^2$ . Although  $\tilde{\pmb{g}}$  and  $\tilde{\pmb{v}}$  are both unbiased estimators of  $\pmb{g}$ , their perturbation variances are different.

![](images/6298a3afffd31c86cba5c5f0e83e5153dfa0e6b2b7a67e8afb17f36892173524.jpg)  
Figure 4: Relative projection error  $\left(\left\| \frac{1}{n} r \right\| / \| g\|\right)$  of the second stage in ResNet20. The number of anchor gradients is 2000. The dimension of anchor subspace is  $k$ . The learning rate is decayed by 10 at epoch 30. The left plot uses random samples from ImageNet. The right plot uses random samples from test data. The benefit of increasing  $k$  becomes smaller when  $k$  is larger.

![](images/1ae41f94a9eaaa391545925625a3eaac81186027617cba2a543ef0ed97a6069a.jpg)  
Figure 5: Stable rank of the residual gradient matrix versus original gradient matrix. The gradients are computed on full batch data for the first stage in ResNet20. The dimension of anchor subspace is  $k = 1000$ .

![](images/e91d49cfd5053db36519dce7b76f65ae545838fb1b4a79850287d3d3f2f5594c.jpg)

The ratio of their perturbation variances is

$$
\frac {\mathbb {E} [ \| \tilde {\boldsymbol {v}} - \boldsymbol {g} \| ^ {2} ]}{\mathbb {E} [ \| \tilde {\boldsymbol {g}} - \boldsymbol {g} \| ^ {2} ]} = \frac {k}{p} + S ^ {2} = \frac {k}{p} + \max _ {i} \| \boldsymbol {R} _ {i,:} \| ^ {2}. \tag {2}
$$

Since  $k \ll p$ , the  $k / p$  term in Eq (2) is small. The  $\max_i \| R_{i,:} \|$  term, the maximal norm of residual gradients, decides how small the ratio is. A natural upper bound of  $\max_i \| R_{i,:} \|$  is  $\| \pmb{R} \|$ . Let  $R := \| \pmb{G} \|_F^2 / \| \pmb{G} \|^2$  denote the stable rank of  $\pmb{G}$ . Lemma 3.1 gives an upper bound on  $\| \pmb{R} \|$ .

Lemma 3.1 ((Rudelson & Vershynin, 2007)). Suppose the anchor gradients  $G_{a}$  are sampled uniformly from  $G$ ,  $B$  consists of the first  $k$  principal components of  $G_{a}$  and the number of anchor gradients  $m$  satisfies  $m = \Omega (R\log R / \alpha^4)$ , then with high probability we have  $\| R\| \leq \lambda_{k + 1} + \alpha \lambda_1$ , where  $\lambda_{i}$  is the  $i_{th}$  largest eigenvalue of  $G$  and  $\alpha \in (0,1)$  is constant.

From Lemma 3.1, we can see the larger the dimension of the anchor subspace  $k$ , the smaller the residual gradients. The benefit of increasing  $k$  is limited when  $\lambda_{k+1} \ll \alpha \| G \|_2$ . We can choose  $m, k$  properly such that  $\lambda_{k+1} \ll \lambda_1$  and  $\alpha \ll 1$ . Then the output of GEP  $\tilde{v}$  enjoys a much smaller perturbation variance than the original gradient perturbation while still being an unbiased estimator.

We next empirically examine the projection error  $\boldsymbol{r} = \sum_{i} \boldsymbol{R}_{i}$ ; by training a 20-layer ResNet on CIFAR10 dataset. We try two different types of auxiliary data to compute the anchor gradients: 1) samples from the same source as private data with correct labels, i.e., 2000 random samples from the test data; 2) samples from different sources with random labels, i.e., 2000 random samples from ImageNet. The relation between the dimension of anchor subspace  $k$  and the projection error rate  $\left( \left\| \frac{1}{n} \boldsymbol{r} \right\| / \| \boldsymbol{g} \| \right)$  is presented in Figure 4. We can see that the project error is small and decreases with  $k$ , and the benefit of increasing  $k$  diminishes when  $k$  is large, as predicted in the theory. In practice one can only use small or moderate  $k$  because of the memory constraint. GEP needs to store at least  $k$  individual gradients and each individual gradient consumes the same amount of memory as the model itself. Moreover, we can see that the projection into anchor subspace of random labeled auxiliary data yields comparable projection error, corroborating our argument that unlabeled auxiliary data are sufficient for finding the anchor subspace.

We also verify that the redundancy of residual gradients is small, by plotting the stable rank of residual gradient matrix in Figure 5. We can see that the stable rank of residual gradient matrix is an order of magnitude higher than the stable rank of original gradient matrix. This implies that it could be hard to further approximate  $\pmb{R}$  with low-dimensional embeddings.

One can also simply discard the residual gradients and output only the perturbed gradient embedding,  $\hat{\pmb{g}}\coloneqq \tilde{\pmb{w}}^T B / n$ . We have the following remark characterizing its property.

Remark 1. Let  $\hat{\pmb{g}}\coloneqq \tilde{\pmb{w}}^T\pmb {B} / n$  be the reconstructed gradient, we have

$$
\mathbb {E} [ \hat {\boldsymbol {g}} ] = \boldsymbol {g} - \boldsymbol {r} / n, \quad \mathbb {E} [ \| \hat {\boldsymbol {g}} - \boldsymbol {g} \| ^ {2} ] \leq k \sigma^ {2} / n ^ {2} + \| \boldsymbol {r} \| ^ {2} / n ^ {2}.
$$

where  $\boldsymbol{r} = \sum_{i}\boldsymbol{R}_{i}$  is the aggregated residual gradients and  $\tilde{\boldsymbol{w}},\boldsymbol{B}$  are given in Algorithm 1.

It is hard to directly compare  $\mathbb{E}[\| \hat{\pmb{g}} -\pmb {g}\| ^2 ]$  with  $\mathbb{E}[\| \tilde{\pmb{v}} -\pmb {g}\| ^2 ]$ . However, we note that  $\hat{\pmb{g}}$  contains a systematic error that makes  $\hat{\pmb{g}}$  a biased estimator of  $\pmb{g}$ . This systematic error is the projection error, which is plotted in Figure 4. We refer to the algorithm releasing  $\hat{\pmb{g}}$  directly as Biased-  $GEP$  or  $B - GEP$  for short. In our experiments, we find that Biased-GEP can outperform standard gradient perturbation when  $k$  is large while the performance of Biased-GEP is always inferior to GEP.

# 3.3 PRIVATE LEARNING WITH GRADIENT EMBEDDING PERTURBATION

GEP (Algorithm 1) describes how to release one-step gradient with privacy guarantee. In this section, we compose the privacy losses at each step to establish the privacy guarantee for the whole learning process. The differentially private learning process with GEP is given in Algorithm 2 and the privacy analysis is presented in Theorem 3.3.

Algorithm 2: Differentially private gradient descent with GEP.  
```latex
1: Input: private dataset  $\mathbb{D}$ ; auxiliary dataset  $\mathbb{D}_a$ ; noise multiplier  $\sigma_1, \sigma_2$ ; number of updates  $T$ ; learning rate  $\eta$ ; number of power iterations  $t$ ; the dimension of anchor subspace  $k$ .  
2: Output: Differentially private model  $\theta^{(T+1)}$ .  
3: for  $i = 1$  to  $T$  do  
4: Compute the private gradients  $G^{(i)}$  and anchor gradients  $G_a^{(i)}$  of  $\theta^{(i)}$ .  
5: Call GEP with  $G^{(i)}, G_a^{(i)}$  and given configuration to get  $\tilde{\boldsymbol{v}}^{(i)}$ .  
6: Update model  $\theta^{(i+1)} = \theta^{(i)} - \eta \tilde{\boldsymbol{v}}^{(i)}$ .  
7: end for
```

Theorem 3.3. For any  $\epsilon < 2\log(1/\delta)$  and  $\delta \in (0,1)$ , the output of Algorithm 2 satisfies  $(\epsilon, \delta)$ -DP if we set  $\sigma \geq 2\sqrt{2T\log(1/\delta)}/\epsilon$ .

If the private gradients are randomly sampled from the full batch gradients, the privacy guarantee can be strengthened via the privacy amplification by subsampling theorem of DP (Balle et al., 2018; Wang et al., 2019; Zhu & Wang, 2019; Mironov et al., 2019).

# 4 EXPERIMENTS

We conduct experiments on MNIST, extended SVHN and CIFAR-10 datasets. The model for MNIST has two convolutional layers with max-pooling and one fully connected layer. The model for SVHN and CIFAR-10 is ResNet20 in He et al. (2016). We replace all batch normalization (Ioffe & Szegedy, 2015) layers with group normalization (Wu & He, 2018) layers because batch normalization mixes the representations of different samples and makes the privacy loss cannot be analyzed accurately. The non-private accuracy for MNIST, SVHN, and CIFAR-10 is  $99.1\%$ ,  $95.9\%$ , and  $90.4\%$ , respectively.

Evaluated algorithms We use the algorithm in Abadi et al. (2016) as benchmark gradient perturbation approach, referred to as "GP". We also compare GEP with PATE (Papernot et al., 2017). The privacy parameter  $\epsilon$  of PATE is data-dependent and hence cannot be released directly (see Section 3.3 in Papernot et al. (2017)). Nonetheless, we report the results of PATE anyway. Before training on CIFAR-10 with privacy guarantee, Abadi et al. (2016) use CIFAR-100 as public data to pre-train their model. This pipeline may not be feasible in practice because CIFAR-10 and CIFAR-100 share the same source and data collection procedure. In our experiments, we train the models from scratch for both GP and GEP.

Implementation details We use SGD with momentum 0.9 as the optimizer and bound privacy loss using the numerical tool in Mironov et al. (2019). Each call of GEP has two privacy losses: (1) the loss of releasing noisy embedding, (2) the loss of releasing noisy residual gradient. Biased-GEP only has the first privacy loss. For given privacy budget and sampling probability,  $\sigma$  is set to be the smallest value such that the privacy budget is allowable to run desired epochs. All experiments are run on a single Tesla V100 GPU with 16G memory. For ResNet20, the parameters are divided into five groups: input layer, output layer, and three intermediate stages. For a given quota of basis vectors, we allocate it to each group according to the square root of the number of parameters in each group. We compute an orthonormal subspace basis on each group separately. Then we concatenate

Table 1: Test performance (in %) with DP guarantee. For PATE, we use the numbers reported in Papernot et al. (2018). Symbol  $\Delta$  denotes the improvement over GP baseline.  

<table><tr><td>Dataset</td><td>Algorithm</td><td>ε = 1.99</td><td>Δ</td><td>ε = 4.97</td><td>Δ</td><td>ε = 7.98</td><td>Δ</td></tr><tr><td rowspan="4">MNIST</td><td>GP</td><td>94.7</td><td>+0.0</td><td>96.8</td><td>+0.0</td><td>97.2</td><td>+0.0</td></tr><tr><td>PATE</td><td>98.5</td><td>+3.8</td><td>/</td><td>/</td><td>/</td><td>/</td></tr><tr><td>B-GEP</td><td>93.1</td><td>-1.6</td><td>94.5</td><td>-2.3</td><td>95.9</td><td>-1.3</td></tr><tr><td>GEP</td><td>96.3</td><td>+1.6</td><td>97.9</td><td>+1.1</td><td>98.4</td><td>+1.2</td></tr><tr><td rowspan="4">SVHN</td><td>GP</td><td>87.1</td><td>+0.0</td><td>91.3</td><td>+0.0</td><td>91.6</td><td>+0.0</td></tr><tr><td>PATE</td><td>/</td><td>/</td><td>91.6</td><td>+0.3</td><td>/</td><td>/</td></tr><tr><td>B-GEP</td><td>88.5</td><td>+1.4</td><td>91.8</td><td>+0.5</td><td>92.3</td><td>+0.7</td></tr><tr><td>GEP</td><td>92.3</td><td>+5.2</td><td>94.7</td><td>+3.4</td><td>95.1</td><td>+3.5</td></tr><tr><td rowspan="3">CIFAR-10</td><td>GP</td><td>43.6</td><td>+0.0</td><td>52.2</td><td>+0.0</td><td>56.4</td><td>+0.0</td></tr><tr><td>B-GEP</td><td>50.3</td><td>+6.7</td><td>59.5</td><td>+7.3</td><td>63.0</td><td>+6.6</td></tr><tr><td>GEP</td><td>59.7</td><td>+16.1</td><td>70.1</td><td>+17.9</td><td>74.9</td><td>+18.5</td></tr></table>

![](images/831abdd1b49d541132f9f1271ccd2416df15fade47180af980414d73950ed8e9.jpg)  
Figure 6: Test accuracy when varying the dimension of anchor subspace. GEP significantly outperforms Biased-GEP for all  $k$ . Moreover, the performance of GEP is not that sensitive to  $k$ .

![](images/eb346f5550edaeaf035887b4d877d58f4c5ff2529846f3d8f4cca53f0dccc6a2.jpg)

![](images/f332b2f2d7bc539c9726dd7fcf22627e26d75f7bbfb15a2e685349d9f2983016.jpg)

![](images/a337395bf1f2c820dc511a3bc305448fd1ddff244d787f521963ee39a4ca0ad2.jpg)

the projections of all groups to construct gradient embeddings. The number of power iterations  $t$  is set as 1 as more iterations do not improve the performance for both GEP and Biased-GEP.

For all datasets, the anchor gradients are computed on 2000 random samples from ImageNet. The selected images are downsampled into size of  $32 \times 32$  ( $28 \times 28$  for MNIST) and we label them randomly at each update. For SVHN and CIFAR-10,  $k$  is chosen from [500, 1000, 1500, 2000]. For MNIST, we halve the size of  $k$ . Initial learning rate and batchsize are 0.1 and 1000, respectively. The learning rate is divided by 10 at middle of training. Weight decay is set as  $1 \times 10^{-4}$ . The clipping threshold for is 10 for original gradients and 2 for residual gradients. The number of training epochs for CIFAR-10 and MNIST is 50, 100, 200 for privacy parameter  $\epsilon = 2, 5, 8$ , respectively. The number of training epochs for SVHN is 5, 10, 20 for privacy parameter  $\epsilon = 2, 5, 8$ , respectively. Privacy parameter  $\delta$  is  $10^{-6}$  for SVHN and  $10^{-5}$  for CIFAR-10 and MNIST.

Results The best accuracy with given  $\epsilon$  is in Table 4. For all datasets, GEP achieves considerable improvement over GP in Abadi et al. (2016). Specifically, GEP achieves  $74.9\%$  test accuracy on CIFAR-10 with  $(8,10^{-5})$ -DP, outperforming GP by  $18.5\%$ . PATE achieves best accuracy on MNIST but its performance drops as the dataset becomes more complex.

We also plot the relation between accuracy and  $k$  in Figure 6. GEP is less sensitive to the choice of  $k$  and outperforms Biased-GEP for all choices of  $k$ . The improvement of increasing  $k$  becomes smaller as  $k$  becomes larger. We note that the memory cost of choosing large  $k$  is high because we need to store at least  $k$  individual gradients to compute anchor subspace.

# 5 CONCLUSION

In this paper, we propose Gradient Embedding Perturbation (GEP) for learning with differential privacy. GEP leverages the gradient redundancy to reduce the added noise and outputs an unbiased estimator of target gradient. The several key designs of GEP significantly boost the applicability of GEP. Extensive experiments on real world datasets demonstrate the superior utility of GEP. In the future, it is exciting to explore the usage of GEP in other kinds of tasks under the constraint of differential privacy, e.g., language modeling (McMahan et al., 2018) and GAN (Xie et al., 2018).

# REFERENCES

Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In ACM SIGSAC Conference on Computer and Communications Security, 2016.  
Noga Alon, Raef Bassily, and Shay Moran. Limits of private learning with access to public data. In Advances in Neural Information Processing Systems, 2019.  
Borja Balle, Gilles Barthe, and Marco Gaboardi. Privacy amplification by subsampling: Tight analyses via couplings and divergences. In Advances in Neural Information Processing Systems, 2018.  
Raef Bassily, Adam Smith, and Abhradeep Thakurta. Differentially private empirical risk minimization: Efficient algorithms and tight error bounds. Annual Symposium on Foundations of Computer Science, 2014.  
Daniel Bernau, Philip-William Grassal, Jonas Robl, and Florian Kerschbaum. Assessing differentially private deep learning with membership inference. arXiv preprint arXiv:1912.11328, 2019.  
Mark Bun and Thomas Steinke. Concentrated differential privacy: Simplifications, extensions, and lower bounds. In Theory of Cryptography Conference, 2016.  
Nicholas Carlini, Chang Liu, Ülfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer: Evaluating and testing unintended memorization in neural networks. In USENIX Security Symposium, 2019.  
Sanjoy Dasgupta and Anupam Gupta. An elementary proof of a theorem of johnson and lindenstrauss. _Random Structures & Algorithms_, 2003.  
Jinshuo Dong, Aaron Roth, and Weijie J Su. Gaussian differential privacy. arXiv preprint arXiv:1905.02383, 2019.  
Cynthia Dwork, Krishnamaram Kenthapadi, Frank McSherry, Ilya Mironov, and Moni Naor. Our data, ourselves: Privacy via distributed noise generation. In Annual International Conference on the Theory and Applications of Cryptographic Techniques, 2006a.  
Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, 2006b.  
Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer Science, 2014.  
Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In ACM SIGSAC Conference on Computer and Communications Security, 2015.  
Kazuto Fukuchi, Quang Khai Tran, and Jun Sakuma. Differentially private empirical risk minimization with input perturbation. In International Conference on Discovery Science, 2017.  
Mary Gooneratne, Khe Chai Sim, Petr Zadrazil, Andreas Kabel, Françoise Beaufays, and Giovanni Motta. Low-rank gradient approximation for memory-efficient on-device training of deep neural network. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.  
Guy Gur-Ari, Daniel A Roberts, and Ethan Dyer. Gradient descent happens in a tiny subspace. arXiv preprint arXiv:1812.04754, 2018.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.  
Briland Hitaj, Giuseppe Ateniese, and Fernando Pérez-Cruz. Deep models under the gan: information leakage from collaborative deep learning. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, 2017.

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, 2015.  
Roger Iyengar, Joseph P Near, Dawn Song, Om Thakkar, Abhradeep Thakurta, and Lun Wang. Towards practical differentially private convex optimization. In IEEE Symposium on Security and Privacy, 2019.  
Bargav Jayaraman, Lingxiao Wang, David Evans, and Quanquan Gu. Distributed learning without distress: Privacy-preserving empirical risk minimization. In Advances in Neural Information Processing Systems, 2018.  
James Jordon, Jinsung Yoon, and Mihaela van der Schaar. Pate-gan: Generating synthetic data with differential privacy guarantees. In International Conference on Learning Representations, 2019.  
Peter Kairouz, Mónica Ribero, Keith Rush, and Abhradeep Thakurta. Dimension independence in unconstrained private erm via adaptive preconditioning. arXiv preprint arXiv:2008.06570, 2020.  
Daniel Kifer, Adam Smith, and Abhradeep Thakurta. Private convex empirical risk minimization and high-dimensional regression. In Conference on Learning Theory, 2012.  
Jaewoo Lee and Daniel Kifer. Concentrated differentially private gradient descent with adaptive per-iteration privacy budget. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018.  
Xinyan Li, Qilong Gu, Yingxue Zhou, Tiancong Chen, and Arindam Banerjee. Hessian based analysis of sgd for deep nets: Dynamics and generalization. In SIAM International Conference on Data Mining, 2020.  
Yuzhe Ma, Xiaojin Zhu, and Justin Hsu. Data poisoning against differentially-private learners: attacks and defenses. In Proceedings of the 28th International Joint Conference on Artificial Intelligence, pp. 4732-4738. AAAI Press, 2019.  
H Brendan McMahan, Daniel Ramage, Kunal Talwar, and Li Zhang. Learning differentially private recurrent language models. In International Conference on Learning Representations, 2018.  
Ilya Mironov. Rényi differential privacy. In IEEE Computer Security Foundations Symposium, 2017.  
Ilya Mironov, Kunal Talwar, and Li Zhang. Rényi differential privacy of the sampled gaussian mechanism. arXiv, 2019.  
Maysum Panju. Iterative methods for computing eigenvalues and eigenvectors. arXiv preprint arXiv:1105.1185, 2011.  
Nicolas Papernot, Martin Abadi, Ulfar Erlingsson, Ian Goodfellow, and Kunal Talwar. Semi-supervised knowledge transfer for deep learning from private training data. 2017.  
Nicolas Papernot, Shuang Song, Ilya Mironov, Ananth Raghunathan, Kunal Talwar, and Ülfar Erlingsson. Scalable private learning with pate. 2018.  
NhatHai Phan, My T Thai, Han Hu, Ruoming Jin, Tong Sun, and Dejing Dou. Scalable differential privacy with certified robustness in adversarial learning. International Conference on Machine Learning, 2020.  
Md Atiqur Rahman, Tanzila Rahman, Robert Laganiere, Noman Mohammed, and Yang Wang. Membership inference attack against differentially private deep learning model. Transactions on Data Privacy, 2018.  
Mark Rudelson and Roman Vershynin. Sampling from large matrices: An approach through geometric functional analysis. Journal of the ACM, 2007.  
Reza Shokri and Vitaly Shmatikov. Privacy-preserving deep learning. In ACM SIGSAC conference on computer and communications security, 2015.  
Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In IEEE Symposium on Security and Privacy (SP), 2017.

Shuang Song, Kamalika Chaudhuri, and Anand D Sarwate. Stochastic gradient descent with differentially private updates. In Global Conference on Signal and Information Processing (GlobalSIP), 2013.  
Kunal Talwar, Abhradeep Guha Thakurta, and Li Zhang. Nearly optimal private lasso. In Advances in Neural Information Processing Systems, 2015.  
Abhradeep Guha Thakurta and Adam Smith. Differentially private feature selection via stability arguments, and the robustness of the lasso. In Conference on Learning Theory, 2013.  
Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. Powersgd: Practical low-rank gradient compression for distributed optimization. In Advances in Neural Information Processing Systems, 2019.  
Di Wang and Jinhui Xu. On sparse linear regression in the local differential privacy model. In International Conference on Machine Learning, 2019.  
Di Wang, Minwei Ye, and Jinhui Xu. Differentially private empirical risk minimization revisited: Faster and more general. In Advances in Neural Information Processing Systems, 2017.  
Jun Wang and Zhi-Hua Zhou. Differentially private learning with small public data. In AAAI, 2020.  
Lingxiao Wang and Quanquan Gu. Differentially private iterative gradient hard thresholding for sparse learning. In International Joint Conference on Artificial Intelligence, 2019.  
Yu-Xiang Wang, Borja Balle, and Shiva Prasad Kasiviswanathan. Subsampled rényi differential privacy and analytical moments accountant. In International Conference on Artificial Intelligence and Statistics, 2019.  
Xi Wu, Matthew Fredrikson, Somesh Jha, and Jeffrey F Naughton. A methodology for formalizing model-inversion attacks. In IEEE Computer Security Foundations Symposium, 2016.  
Xi Wu, Fengan Li, Arun Kumar, Kamalika Chaudhuri, Somesh Jha, and Jeffrey Naughton. Bolt-on differential privacy for scalable stochastic gradient descent-based analytics. In ACM International Conference on Management of Data, 2017.  
Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European conference on computer vision (ECCV), 2018.  
Liyang Xie, Kaixiang Lin, Shu Wang, Fei Wang, and Jiayu Zhou. Differentially private generative adversarial network. arXiv preprint arXiv:1802.06739, 2018.  
Yingxue Zhou, Zhiwei Steven Wu, and Arindam Banerjee. Bypassing the ambient dimension: Private sgd with gradient subspace identification. arXiv preprint arXiv:2007.03813, 2020.  
Ligeng Zhu, Zhijian Liu, and Song Han. Deep leakage from gradients. In Advances in Neural Information Processing Systems, 2019.  
Yuqing Zhu and Yu-Xiang Wang. Poisson subsampled rényi differential privacy. In International Conference on Machine Learning, 2019.
