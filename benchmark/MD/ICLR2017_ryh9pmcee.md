# ENERGY-BASED GENERATIVE ADVERSARIAL NETWORKS

Junbo Zhao, Michael Mathieu and Yann LeCun  
Department of Computer Science, New York University  
Facebook Artificial Intelligence Research  
{jakezhao, mathieu, yann}@cs.nyu.edu

# ABSTRACT

We introduce the "Energy-based Generative Adversarial Network" model (EBGAN) which views the discriminator as an energy function that attributes low energies to the regions near the data manifold and higher energies to other regions. Similar to the probabilistic GANs, a generator is seen as being trained to produce contrastive samples with minimal energies, while the discriminator is trained to assign high energies to these generated samples. Viewing the discriminator as an energy function allows us to use a wide variety of architectures and loss functionals in addition to the usual binary classifier with logistic output. Among them, we show one instantiation of EBGAN framework as using an auto-encoder architecture, with the energy being the reconstruction error, in place of the discriminator. We show that this form of EBGAN exhibits more stable behavior than regular GANs during training. We also show that a single-scale architecture can be trained to generate high-resolution images.

# 1 INTRODUCTION

# 1.1 ENERGY-BASED MODEL

The essence of the energy-based model (LeCun et al., 2006) is to build a function that maps each point of an input space to a single scalar, which is called "energy". The learning phase is a data-driven process that shapes the energy surface in such a way that the desired configuration gets assigned low energies, while the incorrect ones are given high energies. Supervised learning falls into this framework: for each  $X$  in the training set, the energy of the pair  $(X,Y)$  takes low values when  $Y$  is the correct label and higher values for incorrect  $Y$ 's. Similarly, when modeling  $X$  alone within an unsupervised learning setting, lower energy is attributed to the data manifold. The term contrastive sample is often used to refer to a data point causing an energy pull-up, such as the incorrect  $Y$ 's in supervised learning and points from low data density regions in unsupervised learning.

# 1.2 GENERATIVE ADVERSARIAL NETWORKS

Generative Adversarial Networks (GAN) (Goodfellow et al., 2014) have led to significant improvements in image generation (Denton et al., 2015; Radford et al., 2015; Im et al., 2016; Salimans et al., 2016), video prediction (Mathieu et al., 2015) and a number of other domains. The basic idea of GAN is to simultaneously train a discriminator and a generator. The discriminator is trained to distinguish real samples of a dataset from fake samples produced by the generator. The generator uses input from an easy-to-sample random source, and is trained to produce fake samples that the discriminator cannot distinguish from real data samples. During training, the generator receives the gradient of the output of the discriminator with respect to the fake sample. In the original formulation of GAN in Goodfellow et al. (2014), the discriminator produces a probability and, under certain conditions, convergence occurs when the distribution produced by the generator matches the data distribution. From a game theory point of view, the convergence of a GAN is reached when the generator and the discriminator reach a Nash equilibrium.

# 1.3 ENERGY-BASED GENERATIVE ADVERSARIAL NETWORKS

In this work, we propose to view the discriminator as an energy function (or a contrast function) without explicit probabilistic interpretation. The energy function computed by the discriminator can be viewed as a trainable cost function for the generator. The discriminator is trained to assign low energy values to regions of high data density, and higher energy values outside the regions of high data density. Conversely, the generator can be viewed as a trainable parameterized function that produces samples in regions of the space to which the generator assigns low energy. While it is often possible to convert energies into probabilities through a Gibbs distribution (LeCun et al., 2006), the absence of normalization in this energy-based form of GAN provides greater flexibility in the choice of architectures of the discriminator and the training procedures.

The probabilistic binary discriminator in the original formulation of GAN can be seen as one way among many to define the contrast function and loss functional, as described in LeCun et al. (2006) for the supervised and weakly supervised settings, and Ranzato et al. (2007) for the unsupervised setting. We experimentally demonstrate this concept, in the setting where the discriminator is an auto-encoder architecture, and the energy is the reconstruction error. More details of the formulation of EBGAN are provided in the appendix B.

Our main contributions are summarized as follows:

- An energy-based formulation for generative adversarial training.  
- A proof that under a simple hinge loss, when the system reaches convergence, the generator of EBGAN produces points that follow the true dataset distribution.  
- An EBGAN framework with the discriminator using an auto-encoder architecture in which the energy is the reconstruction error.  
- A set of systematic experiments to explore the set of hyper-parameters and architectural choices that produce good results for EBGANs and conventional GANs. These experiments demonstrate EBGAN framework to be more robust with respect to the choice of hyperparameter and architecture.  
- A demonstration that EBGANs can generate reasonable-looking high-resolution images from the ImageNet dataset at  $256 \times 256$  pixel resolution, without a multi-scale approach.

# 2 THE EBGAN MODEL

Let  $p_{data}$  be the underlying probability density of the distribution that produces the dataset. The generator  $G$  is trained to produce a sample  $G(z)$ , for instance an image, from a random vector  $z$  which is sampled from a known distribution  $p_z$ , for instance  $\mathcal{N}(0,1)$ . The discriminator  $D$  takes either real or generated images, and estimates the energy value  $E \in \mathbb{R}$  accordingly, as explained later. For simplicity, we assume that  $D$  produces non-negative values, but the analysis would hold as long as the values are bounded below.

# 2.1 OBJECTIVE FUNCTIONAL

The output of the discriminator goes through an objective functional in order to shape the energy function, attributing low energy to the real data samples and higher energy to the generated ("fake") ones. In this work, we use a margin loss, but many other choices are possible in LeCun et al. (2006). Similarly to what has been done with the "classical" GAN (Goodfellow et al., 2014), in order to get better quality gradients when the generator is far from convergence, we use a two different losses, one to train  $D$  and the other to train  $G$ .

Given a positive margin  $m$ , a data sample  $x$  and a generated sample  $G(z)$ , the discriminator loss  $\mathcal{L}_D$  and the generator loss  $\mathcal{L}_G$  are formally defined:

$$
\mathcal {L} _ {D} (x, z) = D (x) + [ m - D (G (z)) ] ^ {+} \tag {1}
$$

$$
\mathcal {L} _ {G} (z) = D \bigl (G (z) \bigr) \tag {2}
$$

where  $[\cdot ]^{+} = max(0,\cdot)$ . Minimizing  $\mathcal{L}_G$  with respect to the parameters of  $G$  is similar to maximizing the second term of  $\mathcal{L}_D$ . It has the same minimum but non-zero gradients when  $D(G(z))\geq m$ .

# 2.2 OPTIMALITY OF THE SOLUTION

In this section, we present a theoretical analysis of the system presented in section 2.1. We show that if the system reaches a Nash equilibrium, then the generator  $G$  produces samples that are indistinguishable from the distribution of the dataset. This section is done in a non-parametric setting, i.e. we assume that  $D$  and  $G$  have infinite capacity.

Given a generator  $G$ , let  $p_G$  be the density distribution of  $G(z)$  where  $z \sim p_z$ . In other words,  $p_G$  is the density distribution of the samples generated by  $G$ .

We define  $V(G, D) = \int_{x,z} \mathcal{L}_D(x,z) p_{data}(x) p_z(z) \, \mathrm{d}x \, \mathrm{d}z$  and  $U(G, D) = \int_{z} \mathcal{L}_G(z) p_z(z) \, \mathrm{d}z$ . We train the discriminator  $D$  to minimize the quantity  $V$  and the generator  $G$  to minimize the quantity  $U$ . A Nash equilibrium of the system is a pair  $(G^*, D^*)$  that satisfies:

$$
V \left(G ^ {*}, D ^ {*}\right) \leq V \left(G ^ {*}, D\right) \quad \forall D \tag {3}
$$

$$
U \left(G ^ {*}, D ^ {*}\right) \leq U \left(G, D ^ {*}\right) \quad \forall G \tag {4}
$$

Theorem 1. If  $(D^{*},G^{*})$  is a Nash equilibrium of the system, then  $p_{G^*} = p_{data}$  almost everywhere, and  $V(D^{*},G^{*}) = m$ .

Proof. First we observe that

$$
\begin{array}{l} V \left(G ^ {*}, D\right) = \int_ {x} p _ {\text {d a t a}} (x) D (x) \mathrm {d} x + \int_ {z} p _ {z} (z) [ m - D \left(G ^ {*} (z)\right) ] ^ {+} \mathrm {d} z (5) \\ = \int_ {x} \left(p _ {\text {d a t a}} (x) D (x) + p _ {G ^ {*}} (x) [ m - D (x) ] ^ {+}\right) \mathrm {d} x. (6) \\ \end{array}
$$

The analysis of the function  $\varphi(y) = ay + b(m - y)^+$  (see lemma 1 in appendix A for details) shows: (a)  $D^{*}(x) \leq m$  almost everywhere. To verify it, let us assume that there exists a set of measure non-zero such that  $D^{*}(x) > m$ . Let  $D^{\prime}(x) = D^{*}(x)$  if  $D^{*}(x) \leq m$  and  $D^{\prime}(x) = m$  otherwise. Then  $V(G^{*}, D^{\prime}) < V(G^{*}, D^{*})$  which violates equation 3.

(b) The function  $\varphi$  reaches its minimum in  $m$  if  $a < b$  and in 0 otherwise. So  $V(G^{*},D)$  reaches its minimum when we replace  $D^{*}(x)$  by these values. We obtain

$$
\begin{array}{l} V \left(G ^ {*}, D ^ {*}\right) = m \int_ {x} \mathbb {1} _ {p _ {d a t a} (x) <   p _ {G ^ {*}} (x)} p _ {d a t a} (x) d x + m \int_ {x} \mathbb {1} _ {p _ {d a t a} (x) \geq p _ {G ^ {*}} (x)} p _ {G ^ {*}} (x) d x (7) \\ = m \int_ {x} \left(\mathbb {1} _ {p _ {d a t a} (x) <   p _ {G ^ {*}} (x)} p _ {d a t a} (x) + \left(1 - \mathbb {1} _ {p _ {d a t a} (x) <   p _ {G ^ {*}} (x)}\right) p _ {G ^ {*}} (x)\right) d x (8) \\ = m \int_ {x} p _ {G ^ {*}} (x) \mathrm {d} x + m \int_ {x} \mathbb {1} _ {p _ {d a t a} (x) <   p _ {G ^ {*}} (x)} \left(p _ {d a t a} (x) - p _ {G ^ {*}} (x)\right) \mathrm {d} x (9) \\ = m + m \int_ {x} \mathbb {1} _ {p _ {d a t a} (x) <   p _ {G ^ {*}} (x)} \left(p _ {d a t a} (x) - p _ {G ^ {*}} (x)\right) \mathrm {d} x. (10) \\ \end{array}
$$

The second term in equation 10 is non-positive, so  $V(G^{*},D^{*})\leq m$

By putting the ideal generator that generates  $p_{data}$  into the right side of equation 4, we get

$$
\int_ {x} p _ {G ^ {*}} (x) D ^ {*} (x) \mathrm {d} x \leq \int_ {x} p _ {\text {d a t a}} (x) D ^ {*} (x) \mathrm {d} x. \tag {11}
$$

$$
\text {T h u s} (6), \quad \int_ {x} p _ {G ^ {*}} (x) D ^ {*} (x) \mathrm {d} x + \int_ {x} p _ {G ^ {*}} (x) [ m - D ^ {*} (x) ] ^ {+} \mathrm {d} x \leq V \left(G ^ {*}, D ^ {*}\right) \tag {12}
$$

and since  $D^{*}(x) \leq m$ , we get  $m \leq V(G^{*}, D^{*})$ .

Thus,  $m \leq V(G^{*}, D^{*}) \leq m$  i.e.  $V(G^{*}, D^{*}) = m$ . Using equation 10, we see that can only happen if  $\int_{x} \mathbb{1}_{p_{data}(x) < p_{G}(x)} \mathrm{d}x = 0$ , which is true if and only if  $p_{G} = p_{data}$  almost everywhere (this is because  $p_{data}$  and  $p_{G}$  are probabilities densities, see lemma 2 in the appendix A for details).

Theorem 2. Nash equilibrium of this system exists and is characterized by (a)  $p_{G^*} = p_{data}$  (almost everywhere) and (b) there exists a constant  $\gamma \in [0, m]$  such that  $D^*(x) = \gamma$  (almost everywhere).<sup>1</sup>

# Proof. See appendix A.

# 2.3 USING AUTO-ENCODERS

In our experiments, the discriminator  $D$  is structured as an auto-encoder:

$$
D (x) = \left\| D e c (E n c (x)) - x \right\|. \tag {13}
$$

![](images/4decac332c04ab5c4302d9d966d909b2d09b1d96690ae50eb2133f5e708260ed.jpg)  
Figure 1: EBGAN architecture with an auto-encoder discriminator.

The diagram of the EBGAN model with an auto-encoder discriminator is depicted in figure 1. The choice of the auto-encoders for  $D$  may seem arbitrary at the first glance, yet we postulate that it is conceptually more attractive than a binary logistic network:

- Rather than using a single bit of target information to train the model, the reconstruction-based output offers a diverse targets for the discriminator. With the binary logistic loss, only two targets are possible, so within a minibatch, the gradients corresponding to different samples are most likely far from orthogonal. This leads to inefficient training, and reducing the minibatch sizes is often not an option on current hardware. On the other hand, the reconstruction loss will likely produce very different gradient directions within the minibatch, allowing for larger minibatch size without loss of efficiency.  
- Auto-encoders have traditionally been used to represent energy-based model and arise naturally. Given some regularization (see section 2.3.1), auto-encoders have the ability to learn an energy manifold without supervision or negative examples. This mean that even when an EBGAN auto-encoding model is trained to reconstruct a real sample, the discriminator contributes to discovering the data manifold by itself. To the contrary, without the presence of negative examples from the generator, a discriminator trained with binary logistic loss becomes pointless.

# 2.3.1 CONNECTION TO THE REGULARIZED AUTO-ENCODERS

One common issue in training auto-encoders is that the model may learn little more than an identity function. From an energy-based perspective, this means attributing low energy to the whole space. In order to avoid this problem, the model must be pushed to give higher energy to points outside of the data manifold. Theoretical and experimental results have addressed this issue by regularizing the latent representations (Vincent et al., 2010; Rifai et al., 2011; MarcAurelio Ranzato & Chopra, 2007; Kavukcuoglu et al., 2010). Such regularizers aim at restricting the reconstructing power of the auto-encoder so that it can only attribute low energy to a smaller portion of the input points.

We argue that the energy function (the discriminator) in the EBGAN framework can also be seen as "regularized" by having a generator producing the contrastive samples, to which the discriminator ought to give high reconstruction energies. We further argue that the EBGAN framework allows more flexibility from this perspective, because: (i)-the regularizer (generator) is fully trainable instead of being handcrafted; (ii)-the adversarial training paradigm enables a direct interaction between the processes of producing contrastive samples and learning the energy function.

Furthermore, recent work such as Larsen et al. (2015) addresses the insufficient capacity of the  $\ell_2$  loss function; the authors show that training a variational auto-encoder with the element-wise  $\ell_2$  loss failed to capture the fine details in its reconstruction. However, we argue that the EBGAN framework is established from an orthogonal angle where the  $\ell_2$  loss function (or any element-wise loss we may choose) merely serves to produce an energy. It is not a problem if the discriminator of an EBGAN does not reconstructs perfectly, as long as the it is able to tell apart real and fake images by the energies. It is the generator which is producing the final samples.

# 2.4 REPELLING REGULARIZER

We propose a "repelling regularizer" which fits well into the EBGAN auto-encoder model, to keep the model from producing samples that are clustered in one or a few modes of  $p_{data}$ . Another technique "minibatch discrimination" was developed in Salimans et al. (2016) from the same philosophy.

Implementing repelling regularizer has a pulling-away (PT) effect at a representation level. Formally, let  $S \in \mathbb{R}^{s \times N}$  denotes a batch of sample representations taken from the encoder output layer. The PT term is defined as:

$$
f _ {P T} (S) = \frac {1}{N (N - 1)} \sum_ {i} \sum_ {j \neq i} \left(\frac {S _ {i} ^ {\mathrm {T}} S _ {j}}{\| S _ {i} \| \| S _ {j} \|}\right) ^ {2}. \tag {14}
$$

The PT term is intended to decrease the magnitude of cosine similarity between pairwise sample representations, and thus making them as orthogonal as possible. Prior work showed that the output layer of encoder carries representational powerful information for various tasks (Rasmus et al., 2015; Zhao et al., 2015). The rationale for choosing the cosine similarity instead of Euclidean distance is to make the term bounded below and invariant to scale. We use the notation "EBGAN-PT" to refer to the EBGAN auto-encoder model trained with this PT term. Note the PT term is used in the generator loss but not in the discriminator loss, where a weight of 0.1 is associated with it when being added to the loss.

# 3 RELATED WORK

Our work primarily casts GANs into an energy-based model scope. Besides the various type of regularized auto-encoders that the EBGAN framework is connected to (see section 2.3.1), the approaches producing contrastive samples are also highly relevant, such as the use of noisy samples (Vincent et al., 2010) and noisy gradient descent methods including contrastive divergence (Carreira-Perpinan & Hinton, 2005). Several papers was presented with helpful techniques for stabilizing GAN training, (Salimans et al., 2016; Denton et al., 2015; Radford et al., 2015; Im et al., 2016; Mathieu et al., 2015). To our knowledge, our approach is novel and being developed on a model level.

# 4 EXPERIMENTS

# 4.1 EXHAUSTIVE GRID SEARCH ON MNIST

In this section we demonstrate the better training stability of EBGANs over GANs on the simple task of MNIST digit generation with fully-connected networks. We run an exhaustive grid search over a set of architectural choices and hyper-parameters for both frameworks. The convolutional architectures applied on larger scale and more complex datasets are exhibited in later sections.

Formally, we specify the search grid in table 1. We impose the following restrictions on EBGANs: (i)-using learning rate 0.001 and Adam (Kingma & Ba, 2014) for both  $G$  and  $D$ ; (ii)-nLayerD represents the total number of layers of Enc and Dec put together. Also for simplicity, only the number of layers of Enc is varied, Dec is always a single layer; (iii)-the margin is set to 10 throughout all experiments of EBGANs. To analyze the results of the large grid search, we use the inception score (Salimans et al., 2016) as a numerical means for assessing generation quality. We further tweak it and define it as  $I' = E_xKL(p(y)||p(y|\mathbf{x}))^2$  to plot more compact histograms (more details see appendix C). A higher inception score corresponds to better quality of the generated images.

Histograms We plot the histogram of  $I'$  scores in figure 2. We further separated out the optimization related setting from GAN's grid (optimD, optimG and lr) and plot the histogram of each sub-grid individually, together with the EBGAN scores as a reference, in figure 3. The number of experiments for GANs and EBGANs are both 512 in every subplot. The histograms are intended to show that EBGANs are generally more reliably trained than GANs. GANs conversely may demand exquisitely tuned architectural setting and hyper-parameter.

Digits generated with the configurations presenting the best inception score are shown in figure 4.

Table 1: Grid search specs  

<table><tr><td>Settings</td><td>Description</td><td>EBGANs</td><td>GANs</td></tr><tr><td>nLayerG</td><td>number of layers in G</td><td>[2, 3, 4, 5]</td><td>[2, 3, 4, 5]</td></tr><tr><td>nLayerD</td><td>number of layers in D</td><td>[2, 3, 4, 5]</td><td>[2, 3, 4, 5]</td></tr><tr><td>sizeG</td><td>number of neurons in G</td><td>[400, 800, 1600, 3200]</td><td>[400, 800, 1600, 3200]</td></tr><tr><td>sizeD</td><td>number of neurons in D</td><td>[128, 256, 512, 1024]</td><td>[128, 256, 512, 1024]</td></tr><tr><td>dropoutD</td><td>if to use dropout in D</td><td>[true, false]</td><td>[true, false]</td></tr><tr><td>optimD</td><td>to use Adam or SGD for D</td><td>adam</td><td>[adam, sgd]</td></tr><tr><td>optimG</td><td>to use Adam or SGD for G</td><td>adam</td><td>[adam, sgd]</td></tr><tr><td>lr</td><td>learning rate</td><td>0.001</td><td>[0.01, 0.001, 0.0001]</td></tr><tr><td>#experiments:</td><td>-</td><td>512</td><td>6144</td></tr></table>

![](images/83b5bc99b3e725de2755356f4b12c7a21435767783d8de4bf669aeea2b3956ac.jpg)  
Figure 2: (Zooming in on pdf file is recommended.) Histogram of the inception scores from the grid search. The x-axis carries the inception score  $I$  and y-axis informs the portion of the models (in percentage) falling into certain bins. Left (a): general comparison of EBGANs against GANs; Middle (b): EBGANs and GANs both constrained by nLayer [GD] <= 4; Right (c): EBGANs and GANs both constrained by nLayer [GD] <= 3.

![](images/266b1d9da12b4f80631b63d81efad7ca573ccf5cd66357ca3f21cb5b23412da0.jpg)

![](images/703b4a769b97265c24982acb274ac8b95b8f1457d10e8857602cefe16d226a75.jpg)

# 4.2 SEMI-SUPERVISED LEARNING ON MNIST

We examine the possibility of using the EBGAN framework for semi-supervised learning on permutation-invariant MNIST, respectively with 100, 200 and 1000 labels. We utilized a bottom-layer-cost ladder network (LN) (Rasmus et al., 2015) with the EGBAN framework. Ladder Networks can be categorized as an energy-based model that is built with both feedforward and feedback hierarchies with powerful lateral connections coupling two pathways. From table 2, it shows that positioning a bottom-layer-cost LN into an EBGAN framework profitably improves the performance of the LN itself. Albeit slightly lower than the state-of-the-art result by Salimans et al. (2016), our result does shed some light on the postulation that within the scope of the EBGAN framework, iteratively feeding the adversarial contrastive samples produced by the generator to the energy function acts to be an effective regularizer. We notice there was a discrepancy between the reported results between Rasmus et al. (2015) and Pezeshki et al. (2015), so we report both results along with our own implementation of the Ladder Network, which is obtained by running the same setting. The specific experimental setting and analysis is available in appendix D.

Table 2: The comparison of LN bottom-layer-cost model and its EBGAN extension on PI-MNIST semi-supervised task. Note the results are error rate (in %) and they were averaged over 15 different random seeds.  

<table><tr><td>model</td><td>100</td><td>200</td><td>1000</td></tr><tr><td>LN bottom-layer-cost, reported in Pezeshki et al. (2015)</td><td>1.69±0.18</td><td>-</td><td>1.05±0.02</td></tr><tr><td>LN bottom-layer-cost, reported in Rasmus et al. (2015)</td><td>1.09±0.32</td><td>-</td><td>0.90±0.05</td></tr><tr><td>LN bottom-layer-cost, reproduced in this work (see appendix D)</td><td>1.36±0.21</td><td>1.24±0.09</td><td>1.04±0.06</td></tr><tr><td>LN bottom-layer-cost within EBGAN framework</td><td>1.04±0.12</td><td>0.99±0.12</td><td>0.89±0.04</td></tr><tr><td>Relative percentage improvement</td><td>23.5%</td><td>20.2%</td><td>14.4%</td></tr></table>

![](images/517de90129b96c3445c2f747c932cd692c84ef540a3c3d6620b1ad10efb93d54.jpg)

![](images/9c3692b03bcc8b977811785e996a0af4cb3666b0d0c3dcac0684764af8b5c26b.jpg)

![](images/c59dbaa888b09ccafa610b8cb21eb06105dc404a62a0e82b93b4ba6f36cf8292.jpg)

![](images/e89e14d710f9e177be4adc6194d20bc273b449a0c637ba538376f266c8587ef9.jpg)

![](images/cb494d49d2ba41fdd3102b5334476f26ab788d749d7ec27a5e1ff20ca218f105.jpg)

![](images/1843140e1537ec5b1099abef7c0a683fb016c15182b53119d3419efa1777cb87.jpg)

![](images/1635486c89764dcff9022e96debd9f44e871083a4c1250ec8afb2b1481a83f2b.jpg)

![](images/1488f2e03bbb21e32893a4173ab6e56b8ffd69d43f1546d8fc52dc7c0a4558de.jpg)

![](images/a45feeab015da7aedbf95abab6f34a3ced38f4a2caf9e86fa6e4801f78f04007.jpg)  
Figure 3: (Zooming in on pdf file is recommended.) Histogram of the inception scores grouped by different optimization combinations, drawn from optimD, optimG and lr (See text).

![](images/c66f085ed3b01fc919dd3fbec4e958cb6ca83ce111529b42de67d29ae52566d6.jpg)

![](images/ec1c43fa6ae428fef3dab164193759afd71f28fcb10068ff18421bd9f7d82879.jpg)

![](images/ab12b280d829f9b0aae2b5a41379767f46ddd218a96174b78d8de722dcd19516.jpg)

![](images/4d232f82620d4fa7fbe932dda6cd9c4c6e85e5c21ff57280cf3336133b9a1530.jpg)  
Figure 4: Generation from the grid search on MNIST. Left(a): Best GAN model; Middle(b): Best EBGAN model. Right(c): Best EBGAN-PT model.

![](images/320167232f3c749fee8867fa07d2d9d5f9447cb09fd0568a9eab3f430ccd55a8.jpg)

![](images/efdc9b9a278388423314e55d2c1d414df1e1875646368d8d51c4286f14af3f70.jpg)

# 4.3 LSUN & CELEBA

We apply the EBGAN framework with deep convolutional architecture to generate  $64 \times 64$  RGB images, a more realistic task, using the LSUN bedroom dataset (Yu et al., 2015) and the large-scale face dataset CelebA under alignment (Liu et al., 2015). To compare EBGANs with DCGANs (Radford et al., 2015), we train a DCGAN model under the same configuration and show the generations side-by-side with the EBGAN model, in figures 5 and 6. The specific settings are listed in appendix C.

# 4.4 IMAGENET

Finally, we trained EBGANs to generate high-resolution images on ImageNet (Russakovsky et al., 2015). Compared with the datasets we have experimented so far, ImageNet presents an extensively larger and wider space, so modeling the data distribution in a generative model becomes very challenging. We devised an experiment to generate  $128 \times 128$  images, trained on the full ImageNet-1k dataset, which contains roughly 1.3 million images from 1000 different categories. We also trained a network to generate images of size  $256 \times 256$ , on a dog-breed subset of ImageNet, using the wordNet IDs provided by Vinyals et al. (2016). The results are shown in figures 7 and 8. Despite the difficulty of generating images on a high-resolution level, we observe that EBGANs are able to learn about the fact that objects appear in the foreground, together with various background components resembling grass texture, sea under the horizon, mirrored mountain in the water, buildings, etc. In addition, our  $256 \times 256$  dog-breed generations, although far from realistic, show knowledge about the appearances of dogs such as their body, furs and eye.

![](images/10018886ac297623b94b94478dc295f07dbb4898d2e7d94af668f4ff9d7c024c.jpg)  
Figure 5: Generation from LSUN bedroom full-images. Left(a): DCGAN generation. Right(b): EBGAN-PT generation.

![](images/a478664a321321a0c603c3066614dded3b114093399d1989bd947e9772e2a5ca.jpg)  
Figure 6: Generation from CelebA face dataset. Left(a): DCGAN generation. Right(b): EBGAN-PT generation.

# 5 OUTLOOK

We bridge two classes of unsupervised learning methods – GANs and auto-encoders – and revisit the GAN framework from an alternative energy-based perspective. EBGANs show better convergence pattern and scalability to generate high-resolution images. A family of energy-based loss functionals presented in LeCun et al. (2006) can easily be incorporated into the EBGAN framework. For the future work, the conditional setting (Denton et al., 2015; Mathieu et al., 2015) is a promising setup to explore. We hope the future research will raise more attention on a broader view of GANs from the energy-based perspective.

# ACKNOWLEDGMENT

We thank Emily Denton, Soumith Chitala, Arthur Szlam, Marc'Aurelio Ranzato, Pablo Sprechmann, Ross Goroshin and Ruoyu Sun for fruitful discussions. We also thank Emily Denton and Tian Jiang for their help with the manuscript.

![](images/f763b87e16e924273d9dbc134c4a31502711560f2c7c8a7e8bd5a046c9485aff.jpg)  
Figure 7: ImageNet  $128 \times 128$  generations using an EBGAN-PT.

![](images/d3ad49ca8c5c33d2eb1d08e059289bb4f584297efe561e277473248a1671f9ff.jpg)  
Figure 8: ImageNet  $256 \times 256$  generations using an EBGAN-PT.

# REFERENCES

Carreira-Perpinan, Miguel A and Hinton, Geoffrey. On contrastive divergence learning. In AISTATS, volume 10, pp. 33-40. Citeseer, 2005.  
Denton, Emily L, Chintala, Soumith, Fergus, Rob, et al. Deep generative image models using a laplacian pyramid of adversarial networks. In Advances in neural information processing systems, pp. 1486-1494, 2015.  
Goodfellow, Ian, Pouget-Abadie, Jean, Mirza, Mehdi, Xu, Bing, Warde-Farley, David, Ozair, Sherjil, Courville, Aaron, and Bengio, Yoshua. Generative adversarial nets. In Advances in Neural Information Processing Systems, pp. 2672-2680, 2014.  
Im, Daniel Jiwoong, Kim, Chris Dongjoo, Jiang, Hui, and Memisevic, Roland. Generating images with recurrent adversarial networks. arXiv preprint arXiv:1602.05110, 2016.  
Ioffe, Sergey and Szegedy, Christian. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Kavukcuoglu, Koray, Sermanet, Pierre, Boureau, Y-Lan, Gregor, Karol, Mathieu, Michael, and Cun, Yann L. Learning convolutional feature hierarchies for visual recognition. In Advances in neural information processing systems, pp. 1090-1098, 2010.

Kingma, Diederik and Ba, Jimmy. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Larsen, Anders Boesen Lindbo, Sønderby, Søren Kaae, and Winther, Ole. Autoencoding beyond pixels using a learned similarity metric. arXiv preprint arXiv:1512.09300, 2015.  
LeCun, Yann, Chopra, Sumit, and Hadsell, Raia. A tutorial on energy-based learning. 2006.  
Liu, Ziwei, Luo, Ping, Wang, Xiaogang, and Tang, Xiaou. Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision, pp. 3730-3738, 2015.  
MarcAurelio Ranzato, Christopher Poultney and Chopra, Sumit. Efficient learning of sparse representations with an energy-based model. 2007.  
Mathieu, Michael, Couprie, Camille, and LeCun, Yann. Deep multi-scale video prediction beyond mean square error. arXiv preprint arXiv:1511.05440, 2015.  
Pezeshki, Mohammad, Fan, Linxi, Brakel, Philemon, Courville, Aaron, and Bengio, Yoshua. Deconstructing the ladder network architecture. arXiv preprint arXiv:1511.06430, 2015.  
Radford, Alec, Metz, Luke, and Chintala, Soumith. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
Ranzato, Marc'Aurelio, Boureau, Y-Lan, Chopra, Sumit, and LeCun, Yann. A unified energy-based framework for unsupervised learning. In Proc. Conference on AI and Statistics (AI-Stats), 2007.  
Rasmus, Antti, Berglund, Mathias, Honkala, Mikko, Valpola, Harri, and Raiko, Tapani. Semi-supervised learning with ladder networks. In Advances in Neural Information Processing Systems, pp. 3546-3554, 2015.  
Rifai, Salah, Vincent, Pascal, Muller, Xavier, Glorot, Xavier, and Bengio, Yoshua. Contractive auto-encoders: Explicit invariance during feature extraction. In Proceedings of the 28th international conference on machine learning (ICML-11), pp. 833-840, 2011.  
Russakovsky, Olga, Deng, Jia, Su, Hao, Krause, Jonathan, Satheesh, Sanjeev, Ma, Sean, Huang, Zhiheng, Karpathy, Andrej, Khosla, Aditya, Bernstein, Michael, Berg, Alexander C., and Fei-Fei, Li. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211-252, 2015. doi: 10.1007/s11263-015-0816-y.  
Salimans, Tim, Goodfellow, Ian, Zaremba, Wojciech, Cheung, Vicki, Radford, Alec, and Chen, Xi. Improved techniques for training gans. arXiv preprint arXiv:1606.03498, 2016.  
Vincent, Pascal, Larochelle, Hugo, Lajoie, Isabelle, Bengio, Yoshua, and Manzagol, Pierre-Antoine. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of Machine Learning Research, 11(Dec):3371-3408, 2010.  
Vinyals, Oriol, Blundell, Charles, Lillicrap, Timothy, Kavukcuoglu, Koray, and Wierstra, Daan. Matching networks for one shot learning. arXiv preprint arXiv:1606.04080, 2016.  
Yu, Fisher, Seff, Ari, Zhang, Yinda, Song, Shuran, Funkhouser, Thomas, and Xiao, Jianxiong. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.  
Zhao, Junbo, Mathieu, Michael, Goroshin, Ross, and Lecun, Yann. Stacked what-where auto-encoders. arXiv preprint arXiv:1506.02351, 2015.
