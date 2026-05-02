# Distilling Representations from GAN Generator via Squeeze and Span

Anonymous Author(s)

Affiliation

Address

email

# Abstract

In recent years, generative adversarial networks (GANs) have been an actively studied topic and shown to successfully produce high-quality realistic images in various domains. The controllable synthesis ability of GAN generators suggests that they maintain informative, disentangled, and explainable image representations, but leveraging and transferring their representations to downstream tasks is largely unexplored. In this paper, we propose to distill knowledge from GAN generators by squeezing and spanning their representations. We squeeze the generator features into representations that are invariant to semantic-preserving transformations through a network before they are distilled into the student network. We span the distilled representation of the synthetic domain to the real domain by also using real training data to remedy the mode collapse of GANs and boost the student network performance in a real domain. Experiments justify the efficacy of our method and reveal its great significance in self-supervised representation learning. Code will be made public.

# 1 Introduction

Generative adversarial networks (GANs) [22] continue to achieve impressive image synthesis results thanks to the large datasets and the recent advances in network architecture design [4, 35, 36, 33]. GANs synthesize not only realistic images but also steerable ones towards specific content or styles [21, 51, 48, 32, 56, 52, 31]. These properties motivate a rich body of work to adopt powerful pretrained GANs for various computer vision tasks, including part segmentation [66, 55, 59], 3D reconstruction [65], image alignment [47, 44], showing the strengths of GANs in the few-label regime.

GANs typically produce fine-grained, disentangled, and explainable representations, which allow for higher data efficiency and better generalization [41, 66, 55, 59, 65, 47]. Prior works on GAN-based representation learning focus on learned features from either a discriminator network [49] or an encoder network mapping images back into the latent space [19, 17, 18]. However, there is still inadequate exploration about how to leverage or transfer the learned representations in generators. Inspired by the recent success of [66, 55, 59], we hypothesize that representations produced in generator networks are rich and informative for downstream discriminative tasks. Hence, this paper proposes to distill representations from feature maps of a pretrained generator network into a student network (see Fig. 1).

In this paper, we present a novel "squeeze-and-span" technique to distill knowledge from a generator into a representation network<sup>1</sup> that is transferred to a downstream task. Unlike transferring discrimini

![](images/c9e134d92924721a9afd77aaf29b96d3ad2c7e68de9be0e5545b3e80323a935c.jpg)  
(a) Discrimination of real and fake

![](images/0c8d52a488b9b99919639e4af38bed0206f5e127e0123adf8c2027dd17358891.jpg)  
(b) Encoding into latent space

![](images/301391d5eaba565da5a0484fff095665edf37abb82dc960591982e9c320e2d00.jpg)  
Figure 1: A comparison of different ways to transfer representations in GANs. (a) Transferring representations in discriminator  $(D)$  which is tasked to distinguish real or fake images. (b) Transferring representations in encoder  $(E)$  which projects an image into latent space. (c) Transferring representations in student  $(S)$  which predicts the generator features.  
(c) Distillation of generator feature

nator network, generator network is not directly transferable to downstream image recognition tasks, as it cannot ingest image input but a latent vector. Hence, we distill generator network representations into a representation network that can be further transferred to the target task. When fed in a synthesized image, the representation network is optimized to produce similar representations to the generator network's. However, the generator representations are very high-dimensional and not all of them are informative for the downstream task. Thus, we propose a squeeze module that purifies generator representations to be invariant to semantic-preserving transformations through an MLP and an augmentation strategy. As the joint optimization of the squeeze module and representation network can lead to a trivial solution (e.g. mapping representations to zero vector), we employ variance-covariance regularization in [3] while maximizing the agreement between the two networks. Finally, to address the potential domain gap between synthetic and real images, we span the learned representation of synthetic images by training the representation network additionally on real images. We evaluate our distilled representations on CIFAR10, CIFAR100 and STL10 with linear classification tasks as commonly done in representation learning. Experimental results show that squeezing and spanning generator representations outperforms methods that build on discriminator and encoding images into latent space. Moreover, our method achieves better results than discriminative SSL algorithms, including SimSiam [10] and VICReg [3] on CIFAR10 and CIFAR100, and competitive results on STL10, showing significant potential for transferable representation learning.

Our contributions can be summarized as follows: We (1) provide a new taxonomy of representation and transfer learning in generative adversarial networks based on the location of the representations, (2) propose a novel "squeeze-and-span" technique to distill representations in the GAN generator and transfer them for downstream tasks, (3) empirically show the promise of utilizing generator features to benefit self-supervised representation learning.

# 2 Related Work

GANs for Representation Learning. Significant progress has been made upon the interpretability, manipulability, and versatility of the latent space and representation of GANs [35, 36, 33, 34]. It inspires a broad spectrum of GAN-based applications, such as semantic segmentation [66, 55, 59], visual alignment [47, 44], and 3D reconstruction [65], where GAN representations are leveraged to synthesize supervision signals efficiently. As GAN can be trained unsupervised, its representations are transferred to downstream tasks. DCGAN [49] proposes a convolutional GAN and uses the pre-trained discriminator for image classification. BiGAN [17] adopts an inverse mapping strategy to transfer the real domain knowledge for representation learning. While ALI [19] improves this idea with a stochastic network instead of a deterministic one, BigBiGAN [18] extends BiGAN with BigGAN [4] for large scale representation learning. These works leverage or transfer representations from either discriminators or encoders. In contrast, our method reveals that the generator of a pre-trained GAN is typically more suitable for representation transfer with a proper distillation strategy.

![](images/677987abafd787fe9d69996dbddd154ac1b58ab332668cd46f514823ca8ab62a.jpg)  
(a) Discriminator feature

![](images/0d9849c134dd11b4bc031e96c7b1a87a2101104588e4b38a31f1595b5a06cffc.jpg)  
(b) Latent variable

![](images/7f76c9bf3e983d0bcfc2174b7ebe3e44aeb04ca07a073856477c16629cc4c2ad.jpg)  
(c) Generator feature

![](images/1789870ee5956419c3aa0d143a53293a434b221c12adccccce5e2583d93a526d.jpg)  
Figure 2: Visualization of three types of GAN representations: (a) discriminator feature, (b) latent variable, and (c) generator feature. An unconditional StyleGAN2-ADA model pre-trained on CIFAR10 is employed. Colors indicate different classes. Generator features are naturally clustered and consistent with classes.

Knowledge Distillation (KD) aims at training a small network, termed student network, under the supervision of a relatively large network, termed teacher network [30]. According to the knowledge source, it can be roughly divided into logit-based KD and feature-based KD. Logit-based KD methods [40, 58, 12] optimize the divergence loss between the predicted class distributions, usually called logits or soft labels, of the teacher and student network. Feature-based KD methods [37, 2, 53] adopt the teacher model's intermediate layers as supervisory signals for the student. FitNet [50] introduces the output of hidden layers of the teacher network as supervision. AT [61] proposes to match attention maps between the teacher and student. FSP [60] calculates flow between layers as guidance for distillation. Likewise, our method distills knowledge from intermediate layers from a pre-trained GAN generator.

Self-Supervised Representation Learning (SSL) aims at learning general transferable representations from unlabelled data. To produce informative self-supervision signals, the design of handcrafted pretext tasks has flourished for a long time, including jigsaw puzzle completion [45], relative position prediction [15, 16], rotation perception [20], inpainting [46], colorization [39, 63], masked image modeling [26, 57], etc. Instead of performing intra-instance prediction, contrastive learning-based SSL methods explore inter-instance relation. Applying the InfoNCE loss or its variants [25], they typically partition informative positive/negative data subsets and attempt to attract positive pairs while repelling negative ones. MoCo series [27, 7, 11] introduce an offline memory bank to store large negative samples for contrast and a momentum encoder to make them consistent. SimCLR [6] adopts an end-to-end manner to provide negatives in a mini-batch and introduce substantial data augmentation and a projection head to improve the performance significantly. Surprisingly, without negative pairs, BYOL [24] proposes a simple asymmetry SSL framework with the momentum branch applying the stop gradient to avoid model collapse. It inspires a series of in-deep explorations, such as SimSiam [9], Barlow Twins [62], VICReg [3], etc. In this paper, despite the same end goal of obtaining transferable representations and the use of techniques from VICReg [3], we study the transferability of generator representations in pretrained GANs to discriminative tasks, use asymmetric instead of siamese networks, and design effective distillation strategies.

# 3 Rethinking GAN Representations

Let  $G: \mathcal{W} \to \mathcal{X}$  denote a generator network that maps a latent variable in  $\mathcal{W}$  to an image in  $\mathcal{X}$ . An unconditional GAN trains  $G$  adversarially against a discriminator network  $D: \mathcal{X} \to [0,1]$  that estimates the realness of the given images,

$$
\max  _ {G} \min  _ {D} \mathbb {E} \log (1 - D (G (\mathbf {w}))) + \log D (\mathbf {x}). \tag {1}
$$

This adversarial learning is not supervised by any human label and therefore provides a source of unsupervised representation. This characteristic motivates us to visualize and observe different GAN representations (see Fig. 2), which we explain in the following paragraphs. In particular, we use UMAP [43] to embed representation vectors in 2-dimensional space. As unconditional GANs do not associate class labels with generated images, a pre-trained classifier ( $\sim 95\%$  top-1 acc on CIFAR10 validation set) is employed to infer class labels.

Discriminator Feature One way is to employ discriminator network  $D$  [49]. As the discriminator is tasked to distinguish real and fake images, it possibly learns representations that can transfer to other recognition tasks from this pretext task. Formally, let  $D = d^{(L)} \circ d^{(L-1)} \circ \dots \circ d^{(1)}$  denote the decomposition of a discriminator into  $L$  consecutive layers. As shown in Fig. 1(a), given an image  $\mathbf{x}$ , the discriminator representation can be extracted by concatenating the features averagely pooled from each discriminator block output,

$$
\mathbf {h} ^ {d} = \left[ \mu \left(\mathbf {h} _ {1} ^ {d}\right), \dots , \mu \left(\mathbf {h} _ {L} ^ {d}\right) \right], \quad \text {w h e r e} \mathbf {h} _ {i} ^ {d} = d ^ {(i)} \circ \dots \circ d ^ {(1)} (\mathbf {x}), \tag {2}
$$

where  $\mu$  denotes the average pooling operator. However, Fig. 2(a) shows that the cluster of discriminator features is not significantly correlated with class information, probably due to the ineffectiveness of realness discrimination as a representation learning pretext task.

Latent Variable Another choice of GAN representation is the latent variable [19, 17, 18]. Latent variables are assumed to observe simple statistic distribution, e.g. normal distribution, and can be transformed with a given generator into an image. This suggests that latent variables represent and depict statistical characteristics of images. Moreover, works on steerable GANs show that specific dimensions in latent variables correspond to interpretable data variation, suggesting latent variable is an informative representation source. To extract a latent variable representation of an image, one can invert the generator with an encoder. While some works integrate training encoder into adversarial learning [19, 17, 18], we consider training a post hoc encoder [5] given a fixed pre-trained generator  $G$  for comparison in a unified setting,

$$
E ^ {*} = \arg \min  _ {E} \mathbb {E} _ {\mathbf {w} \sim P (\mathbf {w}), \mathbf {x} = G (\mathbf {w})} \left[ \| G (E (\mathbf {x})) - \mathbf {x} \| _ {1} + \mathcal {L} _ {\text {p e r c e p}} (G (E (\mathbf {x})), \mathbf {x}) + \lambda \| E (\mathbf {x}) - \mathbf {w} \| _ {2} ^ {2} \right], \tag {3}
$$

where  $\mathcal{L}_{\mathrm{percep}}$  denotes the LPIPS loss [64] and  $\lambda = 1.0$  is used to balance different loss terms. Fig. 2(b) visualizes the embedding of latent variables<sup>2</sup>. It shows the class information in latent variables is entangled despite they contain information on image synthesis factors.

Generator Feature An overlooked practice is to utilize generator features. Typically, GAN generators transform a low-resolution (e.g.  $4 \times 4$ ) feature map to a higher-resolution one (e.g.  $256 \times 256$ ) and further synthesize images from the final feature map [17, 35] or multi-scale feature maps [36]. The image synthesis is performed hierarchically: feature map from low to high resolution encodes the low-frequency to high-frequency component for composing an image signal [34]. This understanding is also evidenced by image edition works which show that interfering with low-resolution feature maps leads to a structural and high-level change of an image, and altering high-resolution feature maps only induces subtle appearance changes. Therefore, generator features contain valuable hierarchical knowledge about an image. Formally, Let  $G = g^{(L)} \circ g^{(L-1)} \circ \dots \circ g^{(1)}$  denote the decomposition of a discriminator into  $L$  consecutive layers. Given a latent variable  $\mathbf{w} \sim P(\mathbf{w})$  drawn from a prior distribution, we consider the concatenated features average pooled from each generator block output,

$$
\mathbf {h} ^ {g} = \left[ \mu \left(\mathbf {h} _ {1} ^ {g}\right), \dots , \mu \left(\mathbf {h} _ {L} ^ {g}\right) \right], \text {w h e r e} \mathbf {h} _ {i} ^ {g} = g ^ {(i)} \circ \dots \circ g ^ {(1)} (\mathbf {w}). \tag {4}
$$

As Fig. 2(c) shows, generator features within the same class are naturally clustered. This result suggests that generators contain identifiable representations that can be transferred for downstream tasks. However, as GANs do not initially provide a reverse model for the accurate recovery of generator features, it is still inconvenient to extract generator features for any given image. This limitation motivates us to distill the valuable features from GAN generators.

# 4 Squeeze-and-Span Representations from GAN Generator

This section introduces the "Squeeze-and-Span" technique to distill representation from GANs into a student network, which can then be readily transferred for downstream tasks, e.g. image classification. Let  $S_{\theta}:\mathcal{X}\to \mathcal{H}$  denote a student network that maps a given image into representation space. A

![](images/50dc8444c3d2326ab3d8f5a9b066c978de4748d5f19511c9ff66bd00f0b9b944.jpg)  
Figure 3: Squeeze and span representation from the GAN generator. Left: pretrained generator  $G$  and squeeze module  $T_{\phi}$  constitute teacher network to produce squeezed representations which are further distilled into a student network  $S_{\theta}$  (Squeeze part). The student network is also trained on real data (Span part). Right: the generator structure and our squeeze module. The StyleGAN2 generator for synthesizing images of  $32 \times 32$  resolution is illustrated. We average pool (denoted as  $\mu$ ) the feature maps from each synthesis block and transform them with a linear layer plus an MLP, termed squeeze module.

![](images/8766e2203586462c40bc774cf600dd0b13eca99f8b7b9b963b526a996137fdd2.jpg)

naive way of representation learning can be achieved by tasking the student network to predict the teacher representation, which can be formulated as the following optimization problem:

$$
\min  _ {\theta} \mathbb {E} _ {\mathbf {w} \sim P (\mathbf {w})} \| S _ {\theta} (G (\mathbf {w})) - \mathbf {h} ^ {g} (\mathbf {w}) \| _ {2} ^ {2}, \tag {5}
$$

where we use mean squared error to measure the prediction loss and  $\mathbf{h}^g (\mathbf{w})$  to denote the dependence of  $\mathbf{h}^g$  on  $\mathbf{w}$ . However, this formulation has two problems. First, representations extracted through multiple layers of the generator are likely to contain significantly redundant information for downstream tasks but necessary for image synthesis. Second, as the student network is only optimized on synthetic images, it is likely to perform poorly in extracting features from real images in the downstream task due to the potential domain gap between real and synthetic images. To mitigate these issues, we propose the "Squeeze and Span" technique as illustrated in Fig. 3.

# 4.1 Squeezing Informative Representations

To alleviate the first issue that generator representation may contain too much irrelevant information for downstream tasks, we introduce a squeeze (or bottleneck) module  $T_{\phi}$  (Fig. 3) that squeezes informative representations out of the generator representation. In addition, we transform the generated image via a semantic-preserving image transformation  $a$  (e.g. color jittering and cropping) before feeding it to the student work. Equ. 5 can be rewritten as

$$
\min  _ {\theta , \phi} \mathcal {L} _ {\mathrm {R D}} = \mathbb {E} _ {\mathbf {w} \sim P (\mathbf {w}), a \sim \mathcal {A}} \| S _ {\theta} (a [ G (\mathbf {w}) ]) - T _ {\phi} \left(\mathbf {h} ^ {g} (\mathbf {w})\right) \| _ {2} ^ {2}, \tag {6}
$$

where image transformation  $a$  is randomly sampled from  $\mathcal{A}$ . In words, we seek to distill compact representations from the generator among the ones that are invariant to data augmentation  $\mathcal{A}$ , inspired from the success of recent self-supervised methods [6, 10]. where image transformation  $a$  is randomly sampled from  $\mathcal{A}$ . In words, we seek to distill compact representations from the generator among the ones that are invariant to data augmentation  $\mathcal{A}$ , inspired by the success of recent self-supervised methods [6, 10]. An informal interpretation is that, similar to Chen & He [10], considering one of the alternate subproblems that fix  $\theta$  and solve  $\phi$ , the optimal solution would result in the effect of  $T_{\phi^*}(\mathbf{h}^g (\mathbf{w}))\approx \mathbb{E}_{a\sim \mathcal{A}}S_\theta (a[G(\mathbf{w})])$ , which implies Equ 6 encourages  $T_{\phi}$  to squeeze out transformation-invariant representation. However, similar to the siamese network in SSL [10], there exists a trivial solution to Equ. 6: both the squeeze module and the student network degenerate to output constant for any input.

Therefore, we consult the techniques from SSL methods and add regularization terms to the distillation loss. In particular, we employ variance-covariance [3] to explicitly regularize representations to be significantly uncorrelated and varied in each dimension. Formally, in a mini-batch of  $N$  samples, we

denote the squeezed generator representations and student representations with

$$
Z _ {g} = \left[ T _ {\phi} \left(\mathbf {h} ^ {g} \left(\mathbf {w} _ {1}\right)\right), T _ {\phi} \left(\mathbf {h} ^ {g} \left(\mathbf {w} _ {2}\right)\right), \dots , T _ {\phi} \left(\mathbf {h} ^ {g} \left(\mathbf {w} _ {N}\right)\right) \right] \in \mathbb {R} ^ {M \times N}, \tag {7}
$$

$$
Z _ {s} = \left[ S _ {\theta} \left(a _ {1} [ G (\mathbf {w} _ {1}) ]\right), S _ {\theta} \left(a _ {2} [ G (\mathbf {w} _ {2}) ]\right), \dots , S _ {\theta} \left(a _ {N} [ G (\mathbf {w} _ {N}) ]\right) \right] \in \mathbb {R} ^ {M \times N}, \tag {8}
$$

where  $\mathbf{w}_i\sim P(\mathbf{w})$  and  $a_{i}\sim \mathcal{A}$  denote random sample of latent variable and data augmentation operator. The variance loss is introduced to encourage the standard deviation of each representation dimension to be greater than 1,

$$
\mathcal {L} _ {\operatorname {v a r}} (Z) = \frac {1}{M} \sum_ {j = 1} ^ {M} \max  \left(0, 1 - \sqrt {\operatorname {V a r} \left(z ^ {j}\right) + \epsilon}\right), \tag {9}
$$

where  $z^j$  represents the  $j$ -th dimension in representation  $\mathbf{z}$ . The covariance loss is introduced to encourage the correlation of any pair of dimensions to be uncorrelated,

$$
\mathcal {L} _ {\mathrm {c o v}} (Z) = \frac {1}{M} \sum_ {i \neq j} [ C (Z) ] _ {i j} ^ {2}, \tag {10}
$$

$$
\text {w h e r e} C (Z) = \frac {1}{N - 1} \sum_ {i = 1} ^ {N} (\mathbf {z} _ {i} - \bar {\mathbf {z}}) (\mathbf {z} _ {i} - \bar {\mathbf {z}}) ^ {\top}, \bar {\mathbf {z}} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbf {z} _ {i}.
$$

To this end, the loss function of squeezing representations from the generator into the student network can be summarized as

$$
\mathcal {L} _ {\text {s q u e e z e}} = \lambda \mathcal {L} _ {\mathrm {R D}} + \mu \left[ \mathcal {L} _ {\text {v a r}} \left(Z _ {f}\right) + \mathcal {L} _ {\text {v a r}} \left(Z _ {g}\right) \right] + \nu \left[ \mathcal {L} _ {\text {c o v}} \left(Z _ {f}\right) + \mathcal {L} _ {\text {c o v}} \left(Z _ {g}\right) \right]. \tag {11}
$$

Discussion Our work differs from multi-view representation learning methods [3, 10] in the following aspects. (1) Our work studies the transfer of the generative model that does not originally favor representation extraction, whereas most multi-view representation learning learns representation with discriminative pretext tasks. (2) Unlike Siamese networks widely considered in most multi-view representation learning, the two networks in our work are asymmetric: one takes in noise and outputs an image and the other works in the reverse fashion. (3) While most multi-view representation learning methods learn representation networks from scratch, our work distills representations from a pre-trained model. We also see the following connection from a multiview representation learning perspective. While most SSL methods create multiview representations by transforming the input images in multiple ways, we instead pursue different representation views from a well-trained data generation model.

# 4.2 Spanning Representations from Synthetic to Real Domain

Here we address the second problem, the domain between synthetic and real domains, due to two factors. First, the synthesized images may be of low quality. This aspect has been improved a lot with recent GAN modelling [36, 34] and is out of our concern. Second, more importantly, GAN is notorious for the mode collapse issue, suggesting the synthetic data can only cover partial modes of real data distribution. In other words, the synthetic dataset appears to be a subset of the real dataset.

To undermine the harm of mode collapse, we include the real data in the training data of the student network. In particular, in each training step, synthetic data and real data consist of a mini-batch of training data. For synthetic data, the aforementioned squeeze loss is employed. For real data, we employ the original VICReg to compute loss. Specifically, given a mini-batch of real data  $\{\mathbf{x}_i^r\}_{i=1}^N$ , each image  $\mathbf{x}_i^r$  is transformed twice with random data augmentation to obtain two views  $a_i(\mathbf{x}_i^r)$  and  $a_i'(\mathbf{x}_i^r)$ , where  $a_i, a_i' \sim \mathcal{A}$ . The corresponding representations  $Z_r$  and  $Z_r'$  are obtained by feeding the transformed images into  $S_\theta$  similarly to Eq. 8. Then the loss on real data is computed as

$$
\mathcal {L} _ {\text {s p a n}} = \lambda \mathcal {L} _ {\mathrm {R D}} ^ {\prime} + \mu \left[ \mathcal {L} _ {\text {v a r}} \left(Z _ {r}\right) + \mathcal {L} _ {\text {v a r}} \left(Z _ {r} ^ {\prime}\right) \right] + \nu \left[ \mathcal {L} _ {\text {c o v}} \left(Z _ {r}\right) + \mathcal {L} _ {\text {c o v}} \left(Z _ {r} ^ {\prime}\right) \right], \tag {12}
$$

where  $\mathcal{L}_{\mathrm{RD}}^{\prime}$  denotes a self-distillation by measuring the distance of two-view representations on real images. The overall loss is computed by simply combine the generated data loss and real data loss as  $\mathcal{L}_{\mathrm{total}} = \alpha \mathcal{L}_{\mathrm{squeeze}} + (1 - \alpha)\mathcal{L}_{\mathrm{span}}$ , where  $\alpha = 0.5$  denotes the proportion of synthetic data in a mini-batch of training samples.

From a technical perspective, spanning representation seems to be a combination of representation distillation and self-supervised representation learning using VICReg [3]. We interpret this combination as spanning representation from the synthetic domain to the real domain. The representation

Table 1: Representation transfer from different teachers. Top-1 accuracy of linear classification on CIFAR10 and CIFAR100 validation sets are reported and compared.  

<table><tr><td>Knowledge Source</td><td>Transfer Method</td><td>Domain</td><td>CIFAR10</td><td>CIFAR100</td></tr><tr><td rowspan="2">Discriminator</td><td>Direct use (single feature)</td><td>Syn. &amp; Real</td><td>63.81</td><td>30.11</td></tr><tr><td>Direct use (multi-feature)</td><td>Syn. &amp; Real</td><td>77.58</td><td>51.63</td></tr><tr><td rowspan="4">Latent variable</td><td>Encoding</td><td>Syn.</td><td>57.15</td><td>32.19</td></tr><tr><td>Encoding</td><td>Syn. &amp; Real</td><td>50.27</td><td>28.43</td></tr><tr><td>Vanilla distillation (w/ aug)</td><td>Syn.</td><td>84.84</td><td>53.26</td></tr><tr><td>Squeeze</td><td>Syn.</td><td>86.99</td><td>58.56</td></tr><tr><td rowspan="3">Generator feature</td><td>Vanilla distillation (w/ aug)</td><td>Syn.</td><td>84.48</td><td>52.77</td></tr><tr><td>Squeeze</td><td>Syn.</td><td>87.67</td><td>57.35</td></tr><tr><td>Squeeze and span</td><td>Syn. &amp; Real</td><td>92.54</td><td>67.87</td></tr></table>

is dominantly learned in the synthetic domain and generalized to the real domain. The student network learns to fuse representation spaces of two domains into a consistent one in the spanning process. Our experimental evaluation shows that "squeeze and span" can outperform VICReg on real data, suggesting that the squeezed representations do have a nontrivial contribution to the learned representation.

# 5 Experiments

# 5.1 Setup

Dataset and pre-trained GAN Our methods are evaluated on three datasets that do not contain any personally identifiable information or offensive content: CIFAR10, CIFAR100, and STL10. CIFAR10 and CIFAR100 [38] are two image datasets containing small images at  $32 \times 32$  resolution with 10 and 100 classes, respectively. Both datasets are split into 50,000 images as the training set and 10,000 as the validation set. STL-10 [13], which is derived from the ImageNet [14], includes images at  $96 \times 96$  resolution over 10 classes. STL-10 contains 500 labeled images per class (i.e. 5K in total) with an additional 100K unlabeled images for training and 800 labeled images for testing. We adopt StyleGAN2-ADA $^3$  as GAN for representation distillation since it has good stability and high performance. GANs are all pre-trained on training split. Details of the pre-trained GAN checkpoints are available in the supplementary material.

Implementation details The squeeze module uses linear layers to transform the generator features into vectors with 2048 dimensions, which are then summed up and fed into a three-layer MLP to get a 2048-d teacher representation. On CIFAR10 and CIFAR100, we use ResNet18 [29] of the CIFAR variant as the backbone and ResNet18 as the backbone on STL10. On top of the backbone network, a five-layer MLP is added for producing representation. During the evaluation period, only the backbone network is kept. For data augmentation, we use ones proposed in MoCov2 [8]. We use SGD optimizer with cosine learning rate decay [42] scheduler to optimize our models. The actual learning rate is linearly scaled according to the ratio of batch size to 256, i.e. base_lr  $\times$  batch_size/256 [23]. More details of training hyperparameters are available in the supplementary material.

Evaluation We follow the common practice in SSL [6, 54, 28] to evaluate the distilled representation with linear classification task. On top of transferred representation, a linear classifier is trained on training split for 90 epochs using SGD with a base learning rate of 30.0, momentum 0.9, weight decay 0., and batch size 256. Top-1 accuracy is reported as the performance of learned representations.

# 5.2 Transferring GAN Representation

Compared methods In this section, we justify the advantage of distilling generator representations by comparing the performance of different ways of transferring GAN representation. In particular, we consider the following competitors:

Table 2: Comparison to SSL on CIFAR10, CIFAR100, and STL10. Top-1 accuracy on validation set is reported. The biggest number is bolded and the second biggest number is underlined.  

<table><tr><td>Pretrain Data</td><td>Methods</td><td>CIFAR10</td><td>CIFAR100</td><td>STL10</td></tr><tr><td rowspan="2">Real</td><td>SimSiam [10]</td><td>90.94</td><td>62.44</td><td>58.52</td></tr><tr><td>VICReg [3]</td><td>89.20</td><td>63.31</td><td>72.30</td></tr><tr><td rowspan="3">Syn</td><td>SimSiam [10]</td><td>85.11</td><td>47.89</td><td>59.85</td></tr><tr><td>VICReg [3]</td><td>84.68</td><td>52.84</td><td>65.42</td></tr><tr><td>Squeeze (Ours)</td><td>87.67</td><td>57.35</td><td>65.73</td></tr><tr><td rowspan="3">Real &amp; Syn</td><td>SimSiam [10]</td><td>90.88</td><td>62.68</td><td>55.87</td></tr><tr><td>VICReg [3]</td><td>90.46</td><td>65.22</td><td>69.32</td></tr><tr><td>Squeeze &amp; Span (Ours)</td><td>92.54</td><td>67.87</td><td>69.60</td></tr></table>

- Discriminator. As the discriminator network receives image as input and is ready for representation extraction, we directly extract features, single penultimate features, or multiple features (Equ 2), using a pre-trained discriminator and train a linear classifier on top of them.  
- Encoding. We train a post hoc encoder with or without real images involved in the training process as in Equ 3.  
- Distilling latent variable. We employ the vanilla distillation or squeeze method on latent variables with data augmentation engaged.  
- Distilling generator feature. Our method as described in Section 4.

Results Table 1 presents the comparison results, from which we can draw the following conclusions. (1) Representation distillation, whether from the latent variable or generator feature, significantly outperforms discriminator and encoding. We think this is because image reconstruction and realness discrimination are not suitable pretext tasks for representation learning. (2) Distillation from latent variable achieve comparable performance to distillation from generator feature, despite that the former one show entangled class information (Fig. 2). This result can be attributed to a projection head in the student network. (3) Our method works significantly better than vanilla distillation which does not employ a squeeze module. This result suggests that our method squeeze more informative representation that can help to improve the student performance.

# 5.3 Comparison to SSL

We further compare our methods to SSL algorithms such as SimSiam [10] and VICReg [3] in different training data domains: real, synthetic, and a mixture of real and synthetic. Table 2 presents the results from which we want to highlight the following points. (1) Both SimSiam and VICReg perform worse when pre-trained on only synthetic data than only real data, indicating the existence of a domain gap between synthetic data and real data. (2) Our methods outperform SimSiam and VICReg in synthetic and mixture domains, suggesting distillation of generator feature contributes extra improvement SSL. (3) our "Squeeze and Span" is the best among all the competitors on CIFAR10 and CIFAR100 but fails to outperform VICReg pre-trained on real data of STL10. Meanwhile, we also observe the supplement of synthetic data to training data hurts the performance of the SSL algorithm on STL10. This result implies that the GAN we employed is not powerful enough, which can be why our method fails to beat VICReg on STL10.

# 5.4 Ablation Study

Effect of squeeze and span The effect of our method is studied by adding modules to the vanilla version of representation distillation (a) one by one. (a)  $\rightarrow$  (b): After added data augmentation, significant improvement can be observed, suggesting that invariant representation to data augmentation is crucial for linear classification performance. This result inspires us to make teacher representation more invariant. (b)  $\rightarrow$  (c): the learnable  $T_{\phi}$  is introduced to squeeze out invariant representation as teacher. However, trivial performance (10% top-1 accuracy, no better than random guess) is obtained, implying models learn trivial solutions, probably constant output. (c)  $\rightarrow$  (d) & (e): regularization terms are added, and the student network now achieves meaningful performance, which indicates

Table 3: Ablation study on CIFAR10.  $T_{\phi}$  and  $\mathcal{A}$  denote whether to introduce the learnable squeeze module and data augmentation, respectively.  $\mathcal{L}_{\mathrm{RD}}$ ,  $\mathcal{L}_{\mathrm{var}}$ , and  $\mathcal{L}_{\mathrm{cov}}$  represent whether to enable the corresponding losses.  

<table><tr><td></td><td>\( \mathcal{L}_{\text{RD}} \)</td><td>\( \mathcal{A} \)</td><td>\( T_{\phi} \)</td><td>\( \mathcal{L}_{\text{var}} \)</td><td>\( \mathcal{L}_{\text{cov}} \)</td><td>Span</td><td>Top-1 Acc</td></tr><tr><td>a</td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td>74.20</td></tr><tr><td>b</td><td>✓</td><td>✓</td><td></td><td></td><td></td><td></td><td>84.48</td></tr><tr><td>c</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td><td></td><td>10.00</td></tr><tr><td>d</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td><td>79.10</td></tr><tr><td>e</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>87.67</td></tr><tr><td>f</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>92.54</td></tr></table>

![](images/44230a9999801688b8316c95607570e96da16ebee2920b64b0faef059ad8dde7.jpg)  
Figure 4: Representation performance (top-1 accuracy) versus generator quality (FID) on CIFAR100. A better GAN has a lower FID.

the trivial solution is prevented. Moreover, using both regularizations achieves the best performance, which outperforms (b) without "squeeze". (e)  $\rightarrow$  (f): training data is supplemented with real data, i.e. adding "span", the performance is further improved.

Impact of generator performance We further compare the performance of our method when we use GAN checkpoints of different quality. Fig. 4 shows the top-1 accuracy with respect to FID, which indicates the quality of GAN. It is not surprising that GAN quality significantly impacts our method. The higher the quality of generator we utilize, the higher performance of learned representation our method can attain. It is noteworthy that a moderately trained GAN (FID < 11.03) is already able to contribute additional performance improvement on CIFAR100 when compared to VICReg trained on a mixture of synthetic and real data.

# 6 Conclusions

This paper proposes to "squeeze and span" representation from the GAN generator to extract transferable representation for downstream tasks like image classification. The key techniques, "squeeze" and "span", aim to mitigate issues that the GAN generator contains the information necessary for image synthesis but unnecessary for downstream tasks and the domain gap between synthetic and real data. Experimental results justify the effectiveness of our method and show its great promise in self-supervised representation learning. We hope more attention can be drawn to studying GAN for representation learning.

Limitation and future work The current form of our work still maintains several limitations that need to be studied in the future. (1) Since we distill representation from GANs, the performance of learned representation relies on the quality of pre-trained GANs and thus is limited by the progress of GAN techniques. Therefore, whether a pre-maturely trained GAN can also contribute to self-supervised representation learning and how to effectively distill them will be an interesting problem in the future. (2) In this paper, the squeeze module sets the transformation-invariant as a learning objective and only concerns classification as downstream tasks. How to squeeze representations for different downstream tasks selectively would be an interesting problem. (3) As it is a great challenge to train generative models on large-scale datasets like ImageNet [17, 18], it is unknown whether distillation of GAN representation is also helpful under the large-scale setting.

# References

[1] Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan: How to embed images into the stylegan latent space? In ICCV, 2019. 4  
[2] Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, and Zhenwen Dai. Variational information distillation for knowledge transfer. In CVPR, 2019. 3  
[3] Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906, 2021. 2, 3, 5, 6, 8  
[4] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In ICLR, 2018. 1, 2

[5] Lucy Chai, Jun-Yan Zhu, Eli Shechtman, Phillip Isola, and Richard Zhang. Ensembling with deep generative views. In CVPR, 2021. 4  
[6] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020. 3, 5, 7  
[7] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020. 3  
[8] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020. 7  
[9] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR, 2020. 3  
[10] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR, 2021. 2, 5, 6, 8  
[11] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In ICCV, 2021. 3  
[12] Jang Hyun Cho and Bharath Hariharan. On the efficacy of knowledge distillation. In ICCV, 2019. 3  
[13] Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In AISTATS, 2011. 7  
[14] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 7  
[15] Carl Doersch, Abhinav Gupta, and Alexei A. Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015. 3  
[16] Carl Doersch and Andrew Zisserman. Multi-task self-supervised visual learning. In ICCV, 2017. 3  
[17] Jeff Donahue, Philipp Krahenbuhl, and Trevor Darrell. Adversarial feature learning. In ICLR, 2017. 1, 2, 4, 9  
[18] Jeff Donahue and Karen Simonyan. Large scale adversarial representation learning. In NeurIPS, 2019. 1, 2, 4, 9  
[19] Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Olivier Mastropietro, Alex Lamb, Martin Arjovsky, and Aaron Courville. Adversarily learned inference. In ICLR, 2017. 1, 2, 4  
[20] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018. 3  
[21] Lore Goetschalckx, Alex Andonian, Aude Oliva, and Phillip Isola. Ganalyze: Toward visual definitions of cognitive image properties. In ICCV, 2019. 1  
[22] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. NeurIPS, 2014. 1  
[23] Priya Goyal, Piotr Dollar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: TrainingImagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017. 7  
[24] Jean-Bastien Grill, Florian Strub, Florent Alché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. In NeuRIPS, 2020. 3  
[25] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In CVPR, 2006. 3  
[26] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dólár, and Ross Girshick. Masked autoencoders are scalable vision learners. arXiv preprint arXiv:2111.06377, 2021. 3  
[27] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020. 3  
[28] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020. 7  
[29] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 7  
[30] Geoffrey E. Hinton, Oriol Vinyls, and Jeffrey Dean. Distilling the knowledge in a neural network. In NeurIPS Workshop, 2014. 3  
[31] Erik Härkönen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. Ganspace: Discovering interpretable gan controls. In NeurIPS, 2020. 1  
[32] Ali Jahanian, Lucy Chai, and Phillip Isola. On the"steerability" of generative adversarial networks. In ICLR, 2020. 1  
[33] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. In NeurIPS, 2020. 1, 2  
[34] Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. NeurIPS, 2021. 2, 4, 6  
[35] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019. 1, 2, 4

[36] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, 2020. 1, 2, 4, 6  
[37] Jangho Kim, Seonguk Park, and Nojun Kwak. Paraphrasing complex network: Network compression via factor transfer. In NeuRIPS, 2018. 3  
[38] Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009. 7  
[39] Gustav Larsson, Michael Maire, and Gregory Shakhnarovich. Learning representations for automatic colorization. In ECCV, 2016. 3  
[40] Yuncheng Li, Jianchao Yang, Yale Song, Liangliang Cao, Jiebo Luo, and Li-Jia Li. Learning from noisy labels with distillation. In ICCV, 2017. 3  
[41] Francesco Locatello, Ben Poole, Gunnar Ratsch, Bernhard Scholkopf, Olivier Bachem, and Michael Tschannen. Weakly-supervised disentanglement without compromises. In ICML, 2020. 1  
[42] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. In ICLR, 2016. 7  
[43] Leland McInnes, John Healy, Nathaniel Saul, and Lukas Grossberger. Umap: Uniform manifold approximation and projection. JOSS, 2018. 3  
[44] Jiteng Mu, Shalini De Mello, Zhiding Yu, Nuno Vasconcelos, Xiaolong Wang, Jan Kautz, Sifei Liu, and Uc Diego. Coordgan: Self-supervised dense correspondences emerge from gans. In CVPR, 2022. 1, 2  
[45] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV, 2016. 3  
[46] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. Context encoders: Feature learning by inpainting. In CVPR, 2016. 3  
[47] William Peebles, Jun-Yan Zhu, Richard Zhang, Antonio Torralba, Alexei Efros, and Eli Shechtman. Gan-supervised dense visual alignment. In CVPR, 2022. 1, 2  
[48] Antoine Plumerault, Hervé Le Borgne, and Céline Hudelot. Controlling generative models with continuous factors of variations. In ICLR, 2020. 1  
[49] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In ICLR, 2016. 1, 2, 4  
[50] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. In ICLR, 2014. 3  
[51] Yujun Shen, Jinjin Gu, Xiaou Tang, and Bolei Zhou. Interpreting the latent space of gans for semantic face editing. In CVPR, 2020. 1  
[52] Nurit Spingarn-Eliezer, Ron Banner, and Tomer Michaeli. Gan steerability without optimization. *ICLR*, 2021. 1  
[53] Suraj Srinivas and François Fleuret. Knowledge transfer with jacobian matching. In ICML, 2018. 3  
[54] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In ECCV, 2020. 7  
[55] Nontawat Tritrong, Pitchaporn Rewatbowornwong, and Supasorn Suwajanakorn. Repurposing gans for one-shot semantic part segmentation. In CVPR, 2021. 1, 2  
[56] Andrey Voynov and Artem Babenko. Unsupervised discovery of interpretable directions in the gan latent space. In ICML, 2020. 1  
[57] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. arXiv preprint arXiv:2111.09886, 2021. 3  
[58] Chenglin Yang, Lingxi Xie, Siyuan Qiao, and Alan L. Yuille. Training deep neural networks in generations: A more tolerant teacher educates better students. In AAAI, 2019. 3  
[59] Yu Yang, Xiaotian Cheng, Hakan Bilen, and Xiangyang Ji. Learning to annotate part segmentation with gradient matching. In ICLR, 2022. 1, 2  
[60] Junho Yim, Donggyu Joo, Ji-Hoon Bae, and Junmo Kim. A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. In CVPR, 2017. 3  
[61] Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: improving the performance of convolutional neural networks via attention transfer. In ICLR, 2016. 3  
[62] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stephane Deny. Barlow twins: Self-supervised learning via redundancy reduction. In ICML, 2021. 3  
[63] Richard Zhang, Phillip Isola, and Alexei A. Efros. Colorful image colorization. In ECCV, 2016. 3  
[64] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 4  
[65] Yuxuan Zhang, Wenzheng Chen, Huan Ling, Jun Gao, Yinan Zhang, Antonio Torralba, and Sanja Fidler. Image gans meet differentiable rendering for inverse graphics and interpretable 3d neural rendering. In ICLR, 2021. 1, 2  
[66] Yuxuan Zhang, Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche, Adela Barriuso, Antonio Torralba, and Sanja Fidler. Datasetgan: Efficient labeled data factory with minimal human effort. In CVPR, 2021. 1, 2
