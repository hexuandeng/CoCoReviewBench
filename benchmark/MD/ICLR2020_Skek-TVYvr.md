# A UNIFORM GENERALIZATION BOUND FOR GENERATIVE ADVERSARIAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

This paper focuses on the theoretical investigation of unsupervised generalization theory of generative adversarial networks (GANs). We first formulate a more reasonable definition of general error and generalization bounds for GANs. On top of that, we establish a bound for generalization error with a fixed generator in a general weight normalization context. Then, we obtain a width-independent bound by applying  $\ell_{p,q}$  and spectral norm weight normalization. To better understand the unsupervised model, GANs, we establish the generalization bound, which uniformly holds with respect to the choice of generators. Hence, we can explain how the complexity of discriminators and generators contribute to generalization error. For  $\ell_{p,q}$  and spectral weight normalization, we provide explicit guidance on how to design parameters to train robust generators. Our numerical simulations also verify that our generalization bound is reasonable.

# 1 INTRODUCTION

The generative adversarial network (GAN) (Goodfellow et al., 2014) is one of the most powerful generative models for modeling complex high-dimensional tasks, such as image generation, dialogue generation, and image inpainting. Many variants of GANs (Ho & Ermon, 2016; Abadi & Andersen, 2016; Goodfellow et al., 2014; Li et al., 2017; Yu et al., 2018) have also been introduced to reinforce the stability of training processes to obtain more realistic models.

A GAN consists of two neural networks: a discriminator and a generator. Literally, the generator generates simulated data, while the discriminator tries to discriminate between simulated data and real data. The training process of GANs is tantamount to a two-player game between a generator and a discriminator. The main goal is to obtain a good generator, which is able to successfully approximate the distribution of real data. We denote the distribution of real data and the generator-induced distribution by  $\mathcal{D}_{real}$  and  $\mathcal{D}_g$ , respectively. Our goal is to find a generator such that  $\mathcal{D}_{real} = \mathcal{D}_g$ . We revise the goal as  $d(\mathcal{D}_{real},\mathcal{D}_g) = 0$ , with a distribution distance  $d(\cdot ,\cdot)$ . The Jensen-Shannon (JS) divergence is implicitly used in Vanillar GANs (Goodfellow et al., 2014), and the 1-Wasserstein distance is employed in WGANs (Arjovsky et al., 2017). Empirical experiments suggest that the Wasserstein distance is a more sensible measure for differentiating probability measures supported in low-dimensional manifolds.

The generalization properties of GANs are less explored in the literature, and some exceptions are these works (Jiang et al., 2019; Arora et al., 2017; Bartlett et al., 2017; Zhang et al., 2017). Motivated by the supervised learning context, where we say training to be generalized if the gap between the training loss and the test loss is small, we can define the generalization for GANs in a similar way. Concretely, generalization for GANs means that, the population distance between  $\mathcal{D}_{real}$  and  $\mathcal{D}_g$  is closed to the empirical distance between the empirical distributions of  $\mathcal{D}_{real}$  and  $\mathcal{D}_g$ . Hence, we define the gap between the population and empirical distance as the generalization error. Though our ultimate goal is to minimize the former distance, the latter one is what we minimize in practice. Given that our training process provides a small distance between empirical distributions, a small generalization error indicates that the population distance is also small. In other words, a small generalization error guarantees that the generated distribution is close to the real data distribution.

In fact, the training process of GANs is sample-dependent. In other words, the generator depends on the training data sets, which are random samples from  $\mathcal{D}_{real}$ . The training process minimizes  $d(\hat{\mathcal{D}}_{real},\mathcal{D}_g)$ , where  $\hat{\mathcal{D}}_{real}$  denotes the empirical distribution over samples. The deviation

between  $\hat{\mathcal{D}}_{real}$  and  $\mathcal{D}_{real}$  leads to the generalization error, i.e., the gap between  $d(\mathcal{D}_{real},\mathcal{D}_g)$  and  $d(\hat{\mathcal{D}}_{real},\mathcal{D}_g)$ . This motivates us to establish a bound for generalization error, that is, the generalization bound. A tight generalization bound guarantees that the generalization error is small. The highlights and main contributions of this article are summarized as follows:

- We formulate new definitions for both generalization error and generalization bound, which are more reasonable than the definitions in previous work (Arora et al., 2017; Jiang et al., 2019).  
- We establish the generalization error bound in a general version, with a fixed generator. By applying  $\ell_{p,q}$  weight normalization, we obtain a tighter bound.  
- We establish the generalization error bound, which uniformly holds over any choice of generator. Hence, we can explain how the complexity of the generator class and discriminator class contribute to the generalization error.  
- Numerical experiments on Gaussian Mixture models verify that the theory of generalization error bound is consistent with the numerical performance.

# 1.1 RELATED WORK

Some previous works provide theoretical investigations of the generalization of GANs. Arora et al. (2017) introduces a new metric for distributions called  $\mathcal{F}$ -distance, and defines the generalization for GANs based on this distance. On top of that, the paper shows that generalization does happen with a moderate number of training examples (i.e., when the generator wins, the two distributions must be close in  $\mathcal{F}$ -distance). However, they analyze the generalization with a fixed generator. Hence, the result is not guaranteed to hold uniformly across all generators. In this paper, we establish generalization error bounds for both scenarios. When a generator is fixed, we provide a tighter bound for generalization error than that in Arora et al. (2017).

Jiang et al. (2019) also established a bound for generalization error using spectral normalization of GANs with a fixed generator. They show the advantages of spectrum control for generalization by constraining discriminator class. By adopting the Rademacher Complexity, Bartlett et al. (2017) yielded a bound of order  $O(\sqrt{d^3k / m})$ , where  $d, k, m$  stand for the largest discriminator width, the discriminator depth, and the training data size, respectively. Jiang et al. (2019) derived a bound of order  $O(\sqrt{d^2k / m})$ . In our work, we establish a bound of order  $O(d_1^{\frac{1}{p^*}}\sqrt{k / m})$  with  $1 / p^{*} \leqslant 1$ ,  $d_{1} \leqslant d$ , which is tighter than those from previous works. Moreover, we provide a more general version of the bound for generalization error with a fixed generator, and the result in Jiang et al. (2019) is a special case under the spectral weight normalization. To the best of our knowledge, we are the first to establish a generalization bound for GANs that holds uniformly across all generators.

# 2 PRELIMINARIES

We first introduce the formulation of the generalization bound for GANs. We use  $\mathcal{D}_{real}$  for the real data distribution over  $\mathbb{R}^{d_0}$ , and  $\mathcal{N}_d(\mathbf{0},\mathbf{I})$  for a  $d$ -dimensional standard Gaussian distribution. We define the sample set as  $\mathcal{S} \triangleq \{\mathbf{x}_i\}_{i=1}^m$ , where  $\{\mathbf{x}_i\}_{i=1}^m$  are i.i.d samples from  $\mathcal{D}_{real}$  and denote  $\mathbf{z} \sim \mathcal{N}_{l_0}(\mathbf{0},\mathbf{I})$  as Gaussian noise. We denote the  $\ell_{p,q}$ -norm of a matrix  $\mathbf{A}$  as  $\| \mathbf{A}\|_{p,q} \triangleq (\sum_j (\sum_i \mathbf{A}_{i,j}^p)^{\frac{q}{p}})^{\frac{1}{q}}$  and the spectral norm as  $\| \mathbf{A}\|_2$ . We denote the conjugate of  $p$  as  $p*$ , with  $1/p^* + 1/p = 1$ . For an arbitrary distribution  $\mu$ ,  $\hat{\mu}$  denotes the empirical distribution over a random sample of size  $m$  from  $\mu$ .

We let  $\mathcal{F} = \{f\mid f:\mathbb{R}^{d_0}\to [-1,1]\}$  denote the function class of discriminators, and  $\mathcal{G} = \{g\mid g:$ $\mathbb{R}^{l_0}\rightarrow \mathbb{R}^{d_0}\}$  denote the function class of generators. Every generator  $g\in \mathcal{G}$  induces a distribution  $\mathcal{D}_g$  by applying  $g$  to a random sample  $\mathbf{z}\sim \mathcal{N}_{l_0}(\mathbf{0},\mathbf{I})$  , then we generate a sample  $g(\mathbf{z})$  from  $\mathcal{D}_g$  . In the context of GANs, both  $\mathcal{F}$  and  $\mathcal{G}$  are neural network function classes. Specifically,  $\forall f\in \mathcal{F}$ $\mathbf{x}\in$ $\mathbb{R}^{d_0}$ $f(\mathbf{x}) = T_{f,k_1 + 1}\circ \sigma \circ T_{f,k_1}\circ \dots \circ \sigma \circ T_{f,1}\circ \mathbf{x}$  , where  $\forall 1\leq i\leq k_1 + 1,T_{f,i}(\mathbf{u})\triangleq \mathbf{W}_{f,i}^{\top}\mathbf{u} + \mathbf{b}_{f,i}$  and  $\sigma (\cdot)$  is a  $\rho$  -Lipschitz active function. Note that,  $\mathbf{b}_{f,i}\in \mathbb{R}^{d_i\times 1},\mathbf{W}_{f,i}\in \mathbb{R}^{d_{i - 1}\times d_i}$  , where  $d_{i}$  is the width of the ith layer of  $f$  , and  $k_{1} + 1$  is the depth of  $f$  . For convenience, we introduce  $\mathbf{M}_{f,i}\triangleq (\mathbf{b}_{f,i},\mathbf{W}_{f,i}^{\top})^{\top}$  . Similarly, we define the generator class as  $\mathcal{G}\triangleq \{g\mid g = T_{g,k_2 + 1}\circ \sigma \circ$

$T_{g,k_2} \circ \dots \circ \sigma \circ T_{g,1} \circ \mathbf{z}, \mathbf{z} \in \mathbb{R}^{l_0}$ . Here,  $T_{g,i}(\mathbf{u}) \triangleq \mathbf{W}_{g,i}^{\top} \mathbf{u} + \mathbf{b}_{g,i}$ ,  $1 \leq i \leq k_2 + 1$ . Note that,  $\mathbf{b}_{g,i} \in \mathbb{R}^{l_i \times 1}$ ,  $\mathbf{W}_{g,i} \in \mathbb{R}^{l_i - 1 \times l_i}$ , where  $l_i$  is the width of the  $i$ th layer of  $f$ , and  $k_2 + 1$  is the depth of  $g$ . For neural network functions  $f \in \mathcal{F}$  and  $g \in \mathcal{G}$ , we parameterize them as  $f_{\mathbf{w}}, g_{\mathbf{v}}$ , respectively, where  $\mathbf{w}, \mathbf{v}$  are the weight parameters. We denote  $\mathcal{W}$  and  $\mathcal{V}$  as the parameter space of  $\mathcal{F}, \mathcal{G}$ , respectively. We denote the Lipschitz constant (with respect to the input  $\mathbf{x} \in \mathbb{R}^{d_f,0}$ ) of  $f$  as  $L$ .

Weight normalization. Weight normalization is an efficient regularization method for training robust models. We introduce the  $\ell_{p,q}$  and spectral weight normalizations, and establish the generalization theory for such weight normalized neural networks.

Assume that,  $\mathcal{F}$  is a neural network function class, which is parameterized as  $\mathcal{F} = \{f_{\mathbf{w}}|\mathbf{w} = (\mathbf{M}_{f,k_1 + 1},\dots ,\mathbf{M}_{f,1}),\mathbf{w}\in \mathcal{W}\}$ . We define  $\mathcal{F}$  with  $\ell_{p,q}$  weight normalization as  $\| \mathbf{M}_{f,i}\|_{p,q}\leqslant c_{f,i},i = 1,\ldots ,k_1 + 1$ . In this context, we define the parameter norm as  $\| \mathbf{w} - \mathbf{w}'\|_{p,q}\triangleq \sum_{i = 1}^{k_1 + 1}\| \mathbf{M}_{f,i} - \mathbf{M}_{f,i}'\|_{p,q} / c_{f,i}$ .

For spectral weight normalization, we repeat the definition in Jiang et al. (2019)'s work. Assume that,  $\mathcal{F}$  is a neural network function class without bias terms. It is parameterized as  $\mathcal{F} = \{f_{\mathbf{w}}|\mathbf{w} = (\mathbf{W}_{f,k_1 + 1},\dots ,\mathbf{W}_{f,1}),\mathbf{w}\in \mathcal{W}\}$ . We define  $\mathcal{F}$  with spectral weight normalization as  $\| \mathbf{W}_{f,i}\| _2\leqslant B_{f,i},i = 1,\ldots ,k_1 + 1$ . In this context, we define the parameter norm as  $\| \mathbf{w} - \mathbf{w}'\| _2\triangleq \sum_{i = 1}^{k_1 + 1}\| \mathbf{W}_{f,i} - \mathbf{W}_{f,i}'\| _2 / B_{f,i}$ .

For  $\mathcal{F}$  and  $\mathcal{G}$ , the parameter norm  $\| \cdot \|$  induces the metric parameter space  $(\mathcal{W}, \| \cdot \|)$  and  $(\mathcal{V}, \| \cdot \|$ $L_{f}$  and  $L_{g}$ , respectively. Hence, we define the Lipschitz constants of  $f \in \mathcal{F}$  and  $g \in \mathcal{G}$  (with respect to  $\| \cdot \|$ ) as

GANs. According to the training process of GANs, we formulate the objective functions as:

$$
\min _ {g \in \mathcal {G}} \max _ {f \in \mathcal {F}} \underset {\mathbf {x} \sim \mathcal {D} _ {r e a l}} {\mathbb {E}} \left[ \phi (f (\mathbf {x}) \right] - \underset {\mathbf {x} \sim \mathcal {D} _ {g}} {\mathbb {E}} \left[ \phi (f (\mathbf {x})) \right],
$$

where  $\phi(\cdot): [-1,1] \to \mathbb{R}$  is a monotone  $L_{\phi}$ -Lipschitz continuous function. The objective function shows that, the discriminator  $f$  should give high values to  $\mathbf{x} \sim \mathcal{D}_{real}$  and low values to  $\mathbf{x} \sim \mathcal{D}_g$ . When  $\mathcal{D}_{real}, \mathcal{D}_g$  are the same,  $f$  is expected to output 0.

The training process for GAN is tantamount to minimizing a specific distance, between  $\mathcal{D}_g$  and  $\mathcal{D}_{real}$ . To measure the distance between distributions, we consider a general distance. Let  $\mu, \nu$  be two distributions supported on  $\mathbb{R}^{d_1}$ :

$$
d _ {\phi} (\mu , \nu) \triangleq \sup  _ {f \in \mathcal {F}} \mathbb {E} _ {\mathbf {x} \sim \mu} [ \phi (f (\mathbf {x})) ] - \mathbb {E} _ {\mathbf {x} \sim \nu} [ \phi (f (\mathbf {x})) ].
$$

The objective function is equivalent to  $\min_{g\in \mathcal{G}}d_{\phi}(\mathcal{D}_{real},\mathcal{D}_g)$ . Without a loss of generality, we take  $\phi (x)\triangleq x$  in our work. Therefore, our distribution distance can be revised as  $d_{\phi}(\mu ,\nu) = \sup_{f\in \mathcal{F}}\mathbb{E}\left[f(\mathbf{x})\right] - \mathbb{E}\left[f(\mathbf{x})\right]$ . Though the distribution distance is similar to Wasserstein Distance (Arjovsky et al., 2017), our discriminator function class  $\mathcal{F}$  is not forced to be an 1-Lipschitz function class. According to Arora et al. (2017)'s work, if we constrain the range of  $f$  to [0, 1] and utilize the  $\mathcal{F}$ -distance (Arora et al., 2017), the representation can be reduced to the original GAN and WGAN by specific  $\phi (\cdot)$ . Thus, it can unify the JS divergence and the Wasserstein distance. Since our generalization theory also holds in these cases, we omit repetitive discussions of  $\mathcal{F}$ -distance.

Rademacher Complexity. The complexity or capacity of a network function class has a direct effect on the generalization properties of a network. Since a GAN model consists of two network structures, the complexities of  $\mathcal{F}$  and  $\mathcal{G}$  are the keys to further investigation of the generalization properties of GANs. The definition of the Rademacher complexity is given as follows. If we assume that  $\mathcal{F}$  is a class of real value functions, and  $\epsilon_{i}$  is the Rademacher Random Variable, then the empirical and expected Rademacher complexities are defined accordingly,

$$
\hat {\mathfrak {R}} _ {S} (\mathcal {F}) \triangleq \mathbb {E} _ {\epsilon} \left[ \sup  _ {f \in \mathcal {N}} \frac {1}{n} \sum_ {i = 1} ^ {n} \epsilon_ {i} f (\mathbf {z} _ {i}) \right], \quad \mathfrak {R} _ {n, \mathcal {D}} (\mathcal {F}) \triangleq \mathbb {E} _ {S \sim \mathcal {D} ^ {n}} \left[ \hat {\mathfrak {R}} _ {S} (\mathcal {F}) \right],
$$

where  $\epsilon_1, \ldots, \epsilon_n$  are independent Rademacher random variables, i.e.,  $\mathbb{P}(\epsilon_i = 1) = \mathbb{P}(\epsilon_i = -1) = 1/2$ .

# 3 GENERALIZATION ERROR BOUND WITH A FIXED  $g$

We first introduce the definition of generalization error for GANs. In supervised learning, generalization error refers to the gap between the training error and the test error. However, in the context of GANs, neither the training error nor the test error is well defined. This is because the discriminator  $f$ , which is the counterpart of the loss function, varies throughout the training process. In the following, we provide a reasonable measure for the counterparts of the training error and the test error for GANs using the distribution distance  $d_{\phi}(\cdot ,\cdot)$ . For a GAN model with the discriminator  $f\in \mathcal{F}$  and the generator  $g\in \mathcal{G}$ , we define the training error as  $d_{\phi}(\hat{\mathcal{D}}_{real},\mathcal{D}_g)$  and the real error as  $d_{\phi}(\mathcal{D}_{real},\mathcal{D}_g)$ , where  $\hat{\mathcal{D}}_{real}$  is the empirical distribution over the sample set  $S$ . The generalization error bound for GANs is defined as the difference between these two errors.

We compare our definition of the generalization error with that in Arora et al. (2017); Jiang et al. (2019); Zhang et al. (2017). Arora et al. (2017) defined the training error as  $d_{\phi}(\hat{\mathcal{D}}_{real}, \hat{\mathcal{D}}_g)$ , which is related to the empirical distribution of  $\mathcal{D}_g$ . In other words, noise samples are regarded as a training set for GANs, while  $\mathcal{D}_g$  is regarded as an unknown distribution. However, such a consideration does not make much sense. Instead of being reinput in every iteration like the training set  $S$ , a new noise sample set of size  $m$  is generated in every epoch. Thus, noise sample sets are not equivalent to the training set  $S$ , since every noise sample sets is utilized only once.

The definition of the generalization bound in Jiang et al. (2019); Zhang et al. (2017) is slightly different from our intuition. In these works,  $\bar{g}$  is defined as the generator, which is obtained by the training process, and  $g^{*}$  is defined as the optimal generator for  $\inf_{g\in \mathcal{G}}d_{\phi}(\mathcal{D}_{real},\mathcal{D}_g)$ . Then, they define the generalization bound as  $d_{\phi}(\mathcal{D}_{real},\mathcal{D}_{\bar{g}}) - d_{\phi}(\mathcal{D}_{real},\mathcal{D}_{g^*})$ . However, in practise, exact values for  $g^{*}$  and  $\mathcal{D}_{real}$  are not accessible. Thus, the first term fails to represent the training error, and the second term fails to represent the testing error.

It is more reasonable to assume that  $\mathcal{D}_g$  is a known distribution for a fixed  $g$ . The process of generating new noise samples in every epoch, is an empirical approximation of  $\mathcal{D}_g$ , rather than a simple "training noise" data collection process. From this perspective, our definition provides a more reasonable theoretical measure of the generalization error for GANs.

Definition 3.1. For a GAN model with the discriminator  $f \in \mathcal{F}$  and the generator  $g \in \mathcal{G}$ , the generalization error is defined as  $|d_{\phi}(\mathcal{D}_{real}, \mathcal{D}_g) - d_{\phi}(\hat{\mathcal{D}}_{real}, \mathcal{D}_g)|$ . We say  $\epsilon$  is a generalization error bound if the following holds:

$$
\sup _ {g \in \mathcal {G}} | d _ {\phi} (\mathcal {D} _ {r e a l}, \mathcal {D} _ {g}) - d _ {\phi} (\hat {\mathcal {D}} _ {r e a l}, \mathcal {D} _ {g}) | \leqslant \epsilon ,
$$

where  $\epsilon$  only relies on the parameter settings of  $\mathcal{F}$  and  $\mathcal{G}$ .

Intuitively, a low generalization error bound guarantees that the generalization error is low. Hence, the discriminator successfully discriminates between real data and unseen data. In these cases, the generator generates a distribution close to  $\mathcal{D}_{real}$ . An explicit generalization error bound provides us with guidance to design a GAN for designing a GAN to adequately fit real data. The following theorem provides an upper bound for the generalization error, with a fixed  $g$ .

Theorem 3.2. For any fixed  $g \in \mathcal{G}$ , with a probability of at least  $1 - \delta$  over the choice of samples  $S$ :

$$
\left| d _ {\phi} \left(\mathcal {D} _ {\text {r e a l}}, \mathcal {D} _ {g}\right) - d _ {\phi} \left(\hat {\mathcal {D}} _ {\text {r e a l}}, \mathcal {D} _ {g}\right) \right| \leqslant 2 \Re_ {m, \mathcal {D} _ {\text {r e a l}}} (\mathcal {F}) + \sqrt {\frac {\log (1 / \delta)}{m}}. \tag {1}
$$

Though the result of Theorem 3.2 is seems similar to the Theorem 3.1 in Zhang et al. (2017), our definition of generalization error is different. Theorem 3.2 shows that, for a fixed  $g \in \mathcal{G}$ , the bound for the generalization error mainly depends on the complexity of  $\mathcal{F}$ . Although the discriminator class should be complex enough to discriminate between  $\mathcal{D}_{real}$  and  $\mathcal{D}_g$ , a grossly complicated discriminator class generates extra generalization errors. Theorem 3.2 also shows the relationship between  $\Re_{m,\mathcal{D}_{real}}(\mathcal{F})$  and the generalization error bound. For some specific neural network function classes, such as the class of  $\ell_{p,q}$  weight normalized neural networks, we can compute the Rademacher complexity to obtain an explicit upper bound for the generalization error.

Corollary 3.3. Assume  $\| \mathbf{x}\|_{p^*}\leqslant 1,\forall \mathbf{x}\in S$ . For any fixed  $g\in \mathcal{G}$ , with a probability of at least  $1 - \delta$  over the choice of samples  $\mathcal{S}$ , with  $\ell_{p,q}$  weight normalization:

$$
\begin{array}{l} \left| d _ {\phi} (\mathcal {D} _ {r e a l}, \mathcal {D} _ {g}) - d _ {\phi} (\hat {\mathcal {D}} _ {r e a l}, \mathcal {D} _ {g}) \right| \\ \leqslant 2 \left(s _ {k _ {1} + 1} \sqrt {\frac {(2 k _ {1} + 4) \log 2}{m}} + \prod_ {i = 1} ^ {k _ {1} + 1} c _ {f, i} \rho d _ {f, i} ^ {\left[ \frac {1}{p ^ {*}} - \frac {1}{q} \right]} + d _ {f, 0} ^ {\frac {1}{p ^ {*}}} \sqrt {\frac {C (p)}{m}}\right) + \sqrt {\frac {\log (1 / \delta)}{m}}, \\ \end{array}
$$

where

$$
s _ {k + 1} \triangleq \sum_ {i = 1} ^ {k _ {1} + 1} \left(\prod_ {l = i} ^ {k _ {1} + 1} c _ {f, l} \rho d _ {f, l} ^ {\left[ \frac {1}{p ^ {*}} - \frac {1}{q} \right] +}\right) + d _ {f, 0} ^ {\frac {1}{p ^ {*}}} \prod_ {l = 1} ^ {k _ {1} + 1} c _ {f, l} \rho d _ {f, l} ^ {\left[ \frac {1}{p ^ {*}} - \frac {1}{q} \right] +} a n d
$$

$$
C (p) \triangleq \left\{ \begin{array}{l l} 2 \log (2 d _ {f, 0}) & p \in \{1 \} \cup (2, \infty), \\ \min  (p ^ {*} - 1, 2 \log (2 d _ {f, 0})) & p \in (1, 2 ]  . \end{array} \right.
$$

By utilizing the big  $O$  notation, Theorem 3.2 provides a bound of order  $O(d_{f,0}^{\frac{1}{p^*}}\sqrt{k_1 / m})$ , where  $1 / p^{*}\leqslant 1$  usually holds. Notice that, some previous works (Bartlett et al., 2017; Jiang et al., 2019) provide bounds of order  $O(\sqrt{d^3k_1 / m})$  and  $O(d\sqrt{k_1 / m})$ , where  $d = \max \{d_{f,i}\}_{i = 1}^{k_1}$ . Hence, our bound is tighter than previous works.

Remark 3.4. To make a fair comparison between our bound and the bound in Arora et al. (2017), we revise the expression of our result. We denote  $P_{\mathcal{F}}$  as the number of parameters of  $f \in \mathcal{F}$ , and  $L_{f}$  as the Lipschitz constant with respect to the parameters of  $f$ . According to Arora et al. (2017), if  $m \geqslant 3P_{\mathcal{F}}\log (L_fP_{\mathcal{F}} / \epsilon) / \epsilon^2$ , we have a probability of at least  $1 - \exp (-P_{\mathcal{F}})$  over the choice of  $S$ ,  $|d_{\phi}(\mathcal{D}_{real},\mathcal{D}_g) - d_{\phi}(\hat{\mathcal{D}}_{real},\mathcal{D}_g)| \leqslant \epsilon$ . We convert our result into a similar fashion by utilizing  $1 / m \leqslant \epsilon^{2} / (3P_{\mathcal{F}}\log (L_fP_{\mathcal{F}} / \epsilon))$  and  $\delta = \exp (-P_{\mathcal{F}})$ . We obtain

$$
\begin{array}{l} \left| d _ {\phi} \left(\mathcal {D} _ {\text {r e a l}}, \mathcal {D} _ {g}\right) - d _ {\phi} \left(\hat {\mathcal {D}} _ {\text {r e a l}}, \mathcal {D} _ {g}\right) \right| \\ \leqslant \left(2 \left(s _ {k _ {1} + 1} \sqrt {\left(2 k _ {1} + 4\right) \log 2} + \prod_ {i = 1} ^ {k _ {1} + 1} c _ {f, i} \rho d _ {f, i} ^ {\left[ \frac {1}{p ^ {*}} - \frac {1}{q} \right] +} d _ {f, 0} ^ {\frac {1}{p ^ {*}}} \sqrt {C (p)}\right) + \sqrt {p _ {\mathcal {F}}}\right) \cdot \frac {\epsilon}{\sqrt {3 P _ {\mathcal {F}} \log \left(L _ {f} P _ {\mathcal {F}} / \epsilon\right)}}. \\ \end{array}
$$

Since  $\{c_{f,i}\}_{i = 1}^{k_1}$  can be constrained to small values by applying weight normalization, we can force the following holds:

$$
\begin{array}{l} 2 \left(s _ {k _ {1} + 1} \sqrt {\left(2 k _ {1} + 4\right) \log 2} + \prod_ {i = 1} ^ {k _ {1} + 1} c _ {f, i} \rho d _ {f, i} ^ {\left[ \frac {1}{p ^ {*}} - \frac {1}{q} \right] +} d _ {f, 0} ^ {\frac {1}{p ^ {*}}} \sqrt {C (p)}\right) \\ \leqslant \sqrt {P _ {\mathcal {F}}} \cdot (\sqrt {3 P _ {\mathcal {F}} \log (\sum_ {i = 1} ^ {k _ {1} + 1} \rho^ {k _ {1} + 1 - i} \prod_ {j = i} ^ {l} d _ {f , l} ^ {[ \frac {1}{p ^ {*}} - \frac {1}{q} ] +} c _ {f , j} P _ {\mathcal {F}} / \epsilon)} - 1). \\ \end{array}
$$

$\log (L_fP_{\mathcal{F}} / \epsilon)\approx \log (P_{\mathcal{F}}\sqrt{m})$  and  $P_{\mathcal{F}}$  are large numbers, which only depend on the structure of  $\mathcal{F}$ . Hence, there exists  $\{c_{f,i}\}_{i = 1}^{k_1}$  that is small enough to satisfy the inequality above. In these cases, our bound is tighter than Arora et al. (2017)'s bound. In fact, the weight normalization contracts the range of network parameters, so the complexity of  $\mathcal{F}$  is reduced, thereby leading to a tighter generalization error bound.

Inspired by the probabilistic inequality in corollary 3.3, we formulate a hypothesis testing process to judge whether a generator produces data with the same distribution as that of the real data. The appendix contains the theory and experiments related to this novel hypothesis test on a toy dataset.

If we adopt ReLU (a homogeneous function) as an active function and apply the  $\ell_{p,q}$  weight normalization, the bound for the generalization error can be further reduced to become width-independent.

Corollary 3.5. Under the same settings as Corollary 3.3, if  $1 / p + 1 / q \geqslant 1$ , we adopt the ReLU function as active function. Then, for any fixed  $g \in \mathcal{G}$ , with a probability of at least  $1 - \delta$  over the choice of samples  $\mathcal{S}$ :

$$
\begin{array}{l} \left| d _ {\phi} (\mathcal {D} _ {r e a l}, \mathcal {D} _ {g}) - d _ {\phi} (\hat {\mathcal {D}} _ {r e a l}, \mathcal {D} _ {g}) \right| \\ \leqslant 2 \left(\left(1 + d _ {f, 0} ^ {\frac {1}{p ^ {*}}}\right) \prod_ {l = 1} ^ {k _ {1} + 1} c _ {f, l} \sqrt {\frac {(2 k _ {1} + 4) \log 2}{m}} + \prod_ {i = 1} ^ {k _ {1} + 1} c _ {f, i} d _ {f, 0} ^ {\frac {1}{p ^ {*}}} \sqrt {\frac {C (p)}{m}}\right) + \sqrt {\frac {\log (1 / \delta)}{m}}. \\ \end{array}
$$

We can easily extend our results to cases with spectral weight normalization (Jiang et al., 2019).

Corollary 3.6. Let  $d = \max \{d_{f,i}\}_{i=1}^{k_1 + 1}$ . Under the same settings as Corollary 3.3, for any fixed  $g \in \mathcal{G}$ , with a probability of at least  $1 - \delta$  over the choice of samples  $S$  with spectral weight normalization:

$$
\begin{array}{l} \left| d _ {\phi} (\mathcal {D} _ {r e a l}, \mathcal {D} _ {g}) - d _ {\phi} (\hat {\mathcal {D}} _ {r e a l}, \mathcal {D} _ {g}) \right| \\ \leqslant \frac {2 4 \left(\prod_ {i = 1} ^ {k _ {1} + 1} B _ {f , i}\right) d \sqrt {k _ {1} \log \left(2 \sqrt {d m} k _ {1} \prod_ {i = 1} ^ {k _ {1} + 1} B _ {f , i}\right)}}{\sqrt {m}} + \frac {8}{\sqrt {m}} + \sqrt {\frac {\log (1 / \delta)}{m}}. \\ \end{array}
$$

Corollary 3.6 shows the advantage of applying spectral weight normalization. We constrain  $\prod_{i=1}^{k_1+1} B_{f,i}$  such that it is small enough to force the first term to be small. If we set  $\prod_{i=1}^{k_1+1} B_{f,i} = 1$ , the conclusion is reduced to  $O\{\sqrt{d^2 k_1 / m}\}$ . Experiments in Miyato et al. (2018) also show that, spectral weight normalization render the discriminator more powerful for distinguishing between generated data and real data. Hence, we can suffer less from model collapse. Since  $d$  depends on the largest width of the network, the bound in spectral normalization is not actually width-independent. For this reason, we prefer to utilize the  $\ell_{p,q}$  weight normalization to obtain a width-independent bound.

# 4 UNIFORM GENERALIZATION ERROR BOUND

We have established the generalization error upper bound with a fixed generator  $g$ , that is,  $g$  is independent of the training process and the choice of sample set  $S$ . However, such a bound is not a uniform bound for  $\forall g \in G$ . For any fixed  $g$ , let  $S(g)$  denote the set of samples where the inequality (1) in Theorem 3.2 holds, i.e.,  $S(g) \triangleq \{S \mid S \stackrel{i.i.d}{\sim} D_{real}^{m}\}$ , bound (1) holds with  $S$ . We emphasize the dependence of  $S(g)$  on  $g$  because this set varies as  $g$  changes. In other words, different  $S(g)$  values lead to different probability levels, at which the bound holds. Hence, the probability that (1) holds with  $S \in \cap_{g \in G} S(g)$  is not guaranteed to be greater than  $1 - \delta$ . An upper bound for the generalization error with a fixed  $g$  is tantamount to the generalization error bound for neural networks in supervised learning. To further understand GANs, it is necessary for us to establish a uniform bound for generalization error with varying  $g$ . The following theorem provides a generalization bound for GANs.

Theorem 4.1. With a probability of at least  $1 - 2|\mathcal{X}|\cdot \exp (-m\epsilon^2 /4)$  over the choice of samples  $S$  ..

$$
\sup _ {g \in \mathcal {G}} \left| d _ {\phi} (\mathcal {D} _ {r e a l}, \mathcal {D} _ {g}) - d _ {\phi} (\hat {\mathcal {D}} _ {r e a l}, \mathcal {D} _ {g}) \right| \leqslant 2 \Re_ {m, \mathcal {D} _ {r e a l}} (\mathcal {F}) + \epsilon ,
$$

where  $\mathcal{X}$  is a  $\frac{\epsilon}{2LL_g}$ -net of  $\mathcal{V}$ , and  $\mathcal{V}$  is the parameter space of  $\mathcal{G}$ .

Theorem 4.1 depicts the error that is contributed by  $\mathcal{G}$ . Note that, if  $\mathcal{G}$  is complicated, then  $|\mathcal{X}|$  can be a large number. In other words, if we reduce the complexity of  $\mathcal{G}$  by applying weight normalization, the generalization error bound will consequently decrease. In fact, weight normalization is an approach for reducing the complexity of a function class and provides generalization error bound control. For the  $\ell_{p,q}$  weight normalization, the next theorem provides an explicit expression of the generalization error bound.

Corollary 4.2. With a probability of at least  $1 - \delta$  over the choice of samples  $S$ :

$$
\begin{array}{l} \sup  _ {g \in \mathcal {G}} \left| d _ {\phi} \left(\mathcal {D} _ {\text {r e a l}}, \mathcal {D} _ {g}\right) - d _ {\phi} \left(\hat {\mathcal {D}} _ {\text {r e a l}}, \mathcal {D} _ {g}\right) \right| \\ \leqslant 2 \Bigg (s _ {k _ {1} + 1} \sqrt {\frac {(2 k _ {1} + 4) \log 2}{m}} + \prod_ {i = 1} ^ {k _ {1} + 1} c _ {f, i} \rho d _ {f, i} ^ {[ \frac {1}{p ^ {*}} - \frac {1}{q} ] _ {+}} d _ {f, 0} ^ {\frac {1}{p ^ {*}}} \sqrt {\frac {C (p)}{m}} \Bigg) + c _ {\delta} \sqrt {\frac {2 P _ {\mathcal {G}}}{m} \log (6 k _ {2} L L _ {g})}, \\ \end{array}
$$

where  $c_{\delta}$  satisfies

$$
\log (1 / \delta) \leqslant (\frac {c _ {\delta} ^ {2}}{2} - 1) P _ {\mathcal {G}} \log (6 k _ {2} L L _ {g}) + P _ {\mathcal {G}} \log (c _ {\delta} \sqrt {\frac {2 P _ {\mathcal {G}}}{m} \log (6 k _ {2} L L _ {g})},
$$

$s_{k_1 + 1}$  and  $C(p)$  are given in theorem 3.3, and  $P_{\mathcal{G}}$  denotes the number of parameters in  $g \in \mathcal{G}$ .

Notice that with the  $\ell_{p,q}$  weight normalization, we obtain an upper bound for  $L, L_g$ , in a  $\ell_{p,q}$  norm version:  $L \leqslant \prod_{i=1}^{k_1+1} c_{f,i} \rho d_{f,i}^{\left[\frac{1}{p^*}-\frac{1}{q}\right] + }$ ,  $L_g \leqslant \sum_{i=1}^{k_2+1} \rho^{k_2+1-i} \prod_{j=i}^{k_2+1} d_{g,l}^{\left[\frac{1}{p^*}-\frac{1}{q}\right] + }c_{g,j}$ . Hence, for an arbitrary weight normalized GAN, we calculate the generalization bound with  $P_{\mathcal{G}}, \{c_{f,i}, c_{g,j}\}_{i,j=1}^{m}$ . For every  $\delta$ , we calculate the right hand side of the constraint and pick a feasible and small  $c_\delta$ . According to theorem 4.2, the choice of  $g$  contributes to the second term. Since the bounds  $\{c_{f,i}\}_{i=1}^{k_1}$  for weight normalization can be artificially constrained to a small range, the first two terms can be constrained to a small value. Since  $P_{\mathcal{G}}$  is determined by the network structure of  $g \in \mathcal{G}$  and is usually much larger than  $k_2^2$ , the third term is dominate. Similarly, we obtain a width-independent generalization bound by adopting ReLU functions and applying specific  $\ell_{p,q}$  weight normalization, with  $1/p + 1/q \geqslant 1$ .

Our result can be extended to cases with spectral weight normalization.

Corollary 4.3. Assume  $\| \mathbf{x}\| _2\leqslant 1$  for  $\mathbf{x}\in S$ . With a probability of at least  $1 - \delta$  over the choice of samples  $\mathcal{S}$  with spectral weight normalization:

$$
\begin{array}{l} \sup  _ {g \in \mathcal {G}} | d _ {\phi} (\mathcal {D} _ {r e a l}, \mathcal {D} _ {g}) - d _ {\phi} (\hat {\mathcal {D}} _ {r e a l}, \mathcal {D} _ {g}) | \leqslant 2 4 \prod_ {i = 1} ^ {k _ {1} + 1} B _ {f, i} d \sqrt {\frac {k _ {1}}{m} \log (2 \sqrt {d m} k _ {1} \prod_ {i = 1} ^ {k _ {1} + 1} B _ {f , i})} \\ + \frac {8}{\sqrt {m}} + c _ {\delta} \sqrt {\frac {P _ {\mathcal {G}}}{m} \log (6 k _ {2} \prod_ {i = 1} ^ {k _ {1} + 1} B _ {f , i} \prod_ {j = i} ^ {k _ {2} + 1} B _ {g , j})}, \\ \end{array}
$$

where  $c_{\delta}$  satisfies

$$
\begin{array}{l} \log (1 / \delta) \leqslant \left(\frac {c _ {\delta} ^ {2}}{2} - 1\right) P _ {\mathcal {G}} \log \left(6 k _ {2} \prod_ {i = 1} ^ {k _ {1} + 1} B _ {f, i} \prod_ {j = i} ^ {k _ {2} + 1} B _ {g, j}\right) \\ + P _ {\mathcal {G}} \log (c _ {\delta} \sqrt {\frac {2 P _ {\mathcal {G}}}{m} \log (6 k _ {2} \prod_ {i = 1} ^ {k _ {1} + 1} B _ {f , i} \prod_ {j = i} ^ {k _ {2} + 1} B _ {g , j})}). \\ \end{array}
$$

# 5 NUMERICAL EXPERIMENTS.

In this section, we illustrate some numerical experiments to verify that our generalization error bound is consistent with numerical studies. We train Wasserstein generative adversarial networks (WGANs) to learn a three-Gaussian Mixture distribution. The structure of the discriminator is a three-layer  $(2\times 50\mathrm{FC})$  -ReLU-  $(50\times 50\mathrm{FC})$  -ReLU-  $(50\times 50\mathrm{FC})$  -ReLU-  $(50\times 1\mathrm{FC})$  network, where FC denotes a fully connected layer. The generator is a three-layer  $(2\times d\mathrm{FC})$  -ReLU-  $(d\times 50\mathrm{FC})$  ReLU-  $(50\times 50\mathrm{FC})$  -ReLU-  $(50\times 2\mathrm{FC})$  network, where  $d$  takes value in  $\{50,70,90,110,130\}$ . After the training process, we calculate the generalization error and generalization error bound. The details of the experimental settings are included in the appendix.

In order to compute the generalization error with an empirical approach, we generate two data sets,  $S^{train}$  and  $S^{test}$ , while the sample size of  $S^{test}$  is much larger than that of  $S^{train}$ . We regard the empirical distribution over  $S^{test}$  as an approximation of  $\mathcal{D}_{real}$ . The noise is generated from  $\mathbf{z} \sim N_2(\mathbf{0},\mathbf{I})$ , in the training process.

We compare the generalization error and generalization bound of the WGAN model. We adopt an empirical approach to calculate  $d_{\phi}(\mathcal{D}_{real}, \mathcal{D}_g)$  and  $d_{\phi}(\hat{\mathcal{D}}_{real}, \mathcal{D}_g)$ . Since theorem 4.2 provides a computable generalization bound with  $\ell_{2,2}$  weight normalization, for a trained WGAN, we can calculate the generalization error and generalization bound. For each  $d \in \{50, 70, 90, 110, 130\}$ , we repeat the following process 50 times to compute the generalization error and the average generalization error. At the same time, we calculate the generalization error bound for each  $d$  and the average value. Figure 1 displays two visualizations of the simulated results. The left panel shows a visibly good generator and the right panel shows a visibly bad generator.

According to the previous discussions, generalization error is defined as  $|d_{\phi}(\mathcal{D}_{real}, \mathcal{D}_g) - d_{\phi}(\hat{\mathcal{D}}_{real}, \mathcal{D}_g)|$ . To compute  $d_{\phi}(\widehat{\mathcal{D}_{real}}, \mathcal{D}_g), d_{\phi}(\widehat{\mathcal{D}_{real}}, \mathcal{D}_g)$ , we approximate the distribution distance  $d_{\phi}(\mathcal{D}_{real}, \mathcal{D}_g), d_{\phi}(\hat{\mathcal{D}}_{real}, \mathcal{D}_g)$  by substituting  $\mathcal{D}_{real}$ ,  $\hat{\mathcal{D}}_{real}$  with the uniform distribution over

![](images/998c55519ce900c9a778aa3c9c2d68c5d0fee121707e8375adfbf2ba280f8ba9.jpg)  
(a) A visibly good generator.

![](images/4cb754430727173833fec1bf392120af0157faf69684d6693c61cb3d39e62aee.jpg)  
(b) A visibly bad generator.

![](images/cd710476c9040463bf16527cd01ebf4753d1e1bf46609b87435e81f72f66fa78.jpg)  
Figure 1: The blue point cloud represents  $S^{train}$ , a sample from a Gaussian Mixture distribution. The red points are  $\{g(\mathbf{z}_i)\}_{i=1}^{m_{train}}$ , while  $\{\mathbf{z}_i\}_{i=1}^{m_{train}} \stackrel{i.i.d.}{\sim} N_2(\mathbf{0}, \mathbf{I})$ .  
Figure 2: Each dot on the graph represents the average generalization error and the generalization bound of WGAN with  $d \in \{50, 70, 90, 110, 130\}$ .

$\mathcal{S}_{test}, \mathcal{S}_{train}$  (denoted as  $\hat{\mathcal{D}}_{test}, \hat{\mathcal{D}}_{train}$ ), respectively. The detailed process is included in the appendix.

According to figure 2, there is a positive correlation between the generalization error and generalization bound. As the number of parameters in the generator increases, the generalization bound increases. In other words, the experiment verifies that, our bound provides generalization error control. A low generalization bound guarantees that the generalization error is low, whereas a generator with a large number of parameters introduces more error.

In fact, by applying  $\ell_{p,q}$  weight normalization to  $f,g$  with small  $\{c_{f,i},c_{g,i}\}_{i = 1}^{k_1}$ , we control the generalization bound so that it remains a small value. The number of parameters in the generator should not be extremely large. So far, the experiment and our theory have explained how  $\ell_{p,q}$  weight normalization and the parameter settings of  $f,g$  affect the generalization of GANs. Our generalization bound provides explicit guidance on parameters designing to train GANs with small generalization error. Thus, we can obtain robust generators.

# 6 CONCLUSION

In this paper, we establish the generalization theory for GANs and provide a more reasonable definition of generalization error. We first establish a general bound for generalization error, with a fixed generator. To further understand GANs, we establish the generalization error bound, which uniformly holds over any choice of generators. Our numerical experiments on Gaussian Mixture models verify that, our theory is consistent with the numerical studies. In the Appendix, we also formulate a novel hypothesis testing procedure to judge whether the generated distribution equals the distribution of observed data. Notice that, in these high dimension cases, the ordinary statistical approaches do not work well. Our hypothesis test is capable of discriminating between good and bad generators. One interesting future research topic is to develop generalization error bounds for autoencoder GANs with an additional encoder network.

# REFERENCES

Martín Abadi and David G. Andersen. Learning to protect communications with adversarial neural cryptography. CoRR, abs/1610.06918, 2016. URL http://arxiv.org/abs/1610.06918.  
Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein gan. arXiv preprint arXiv:1701.07875, 2017.  
Sanjeev Arora, Rong Ge, Yingyu Liang, Tengyu Ma, and Yi Zhang. Generalization and equilibrium in generative adversarial nets (gans). In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 224-232. JMLR.org, 2017.  
Peter L. Bartlett, Dylan J. Foster, and Matus Telgarsky. Spectrally-normalized margin bounds for neural networks. CoRR, abs/1706.08498, 2017. URL http://arxiv.org/abs/1706.08498.  
Hao Chen, Zhanfeng Mo, Zhouwang Yang, and Xiao Wang. Theoretical investigation of generalization bound for residual networks. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19, pp. 2081-2087. International Joint Conferences on Artificial Intelligence Organization, 7 2019. doi: 10.24963/ijcai.2019/288. URL https://doi.org/10.24963/ijcai.2019/288.  
Ilya Dumer. Covering an ellipsoid with equal balls. J. Comb. Theory, Ser. A, 113(8):1667-1676, 2006. doi: 10.1016/j.jcta.2006.03.021. URL https://doi.org/10.1016/j.jcta. 2006.03.021.  
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2, NIPS '14, pp. 2672-2680, Cambridge, MA, USA, 2014. MIT Press. URL http://dl.acm.org/citation.cfm?id=2969033.2969125.  
Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. CoRR, abs/1606.03476, 2016. URL http://arxiv.org/abs/1606.03476.  
Haoming Jiang, Zhehui Chen, Minshuo Chen, Feng Liu, Dingding Wang, and Tuo Zhao. On computation and generalization of generative adversarial networks under spectrum control. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=rJNH6sAqY7.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Jiwei Li, Will Monroe, Tianlin Shi, Sébastien Jean, Alan Ritter, and Dan Jurafsky. Adversarial learning for neural dialogue generation. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 2157-2169, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1230. URL https://www.aclweb.org/anthology/D17-1230.  
Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. CoRR, abs/1802.05957, 2018. URL http://arxiv.org/abs/1802.05957.  
Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S. Huang. Generative image inpainting with contextual attention. CoRR, abs/1801.07892, 2018. URL http://arxiv.org/abs/1801.07892.  
Pengchuan Zhang, Qiang Liu, Dengyong Zhou, Tao Xu, and Xiaodong He. On the discrimination-generalization tradeoff in gans. CoRR, abs/1711.02771, 2017. URL http://arxiv.org/abs/1711.02771.
