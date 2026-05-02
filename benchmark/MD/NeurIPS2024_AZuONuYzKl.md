# What Do You See in Common? Learning Hierarchical Prototypes over Tree-of-Life to Discover Evolutionary Traits

Anonymous Author(s)

Affiliation

Address

email

Phylogenetic Tree

![](images/a37054b627051c598d81976b538be08f2726f7dc642a1988e5fe14e91634fc8b.jpg)  
Black Billed Cuckoo

![](images/8ac58226063d0322219a4589ede5b9d700c18484d5485a2cca9e25695ea410b2.jpg)  
Figure 1: Sample images of bird species with zoomed-in views of learned prototypes along with their associated score maps. We consider the problem of finding evolutionary traits common to a group of species derived from the same ancestor (blue) that are absent in other species from a different ancestor (red). We can infer that descendants of the blue node share a common trait: long tail, absent from descendants of the red node.

![](images/91fb6dec7987094babb7246062559eac2525f4fb0be911a4e79b3b168b573f26.jpg)  
Yellow Billed Cuckoo

![](images/4c884098fb7bf3e3afc306ce6d0af7cd28cd2a91df9fdeeff2efca21bba907a9.jpg)

![](images/2326c1e0dd4d1c5f87b8f3cc87dc955070e35755b452b44005ff3a639cf44482.jpg)  
Mangrove Cuckoo

![](images/ff94e8153c7213ff529f2fe8b9c5cad9e5ea7adcf0311ff47865b46acf2c4f50.jpg)

![](images/c2afe502fa59a34f5420b86aeb3ff90de5079d0c683331e3deb21356b946b65d.jpg)  
Groove Billed Ani

![](images/338d2c312ef8267a9dfa73d3298da96a4976ae469868ab35bc07820fbc605754.jpg)

![](images/54e903fb29fc49afe3851e4b01e6fe5dc643d826d5d1f57741e7ecb9a3203657.jpg)  
Northern Fulmar

![](images/d346c1691c16d56c3123a2bc2677fbe02cf012e8373fdda01f11511cea383100.jpg)

![](images/9ae4a4c34c7fa0f18fd956c3509191c5d21d223dfa2d0ef5f9470befafae3550.jpg)  
Sooty Albatross

![](images/ab361e8d80ce64cff34a5d353be280ff266e4cb087d75df9aa03a5bdaa6fd2f7.jpg)

![](images/9cd8a0b5e6e2fa3af9f334099c5d8bb57e9293c33a1569be09b837fefceba36d.jpg)  
Laysan Albatross

![](images/1dea829e20415f695af69cefe9af161aefa20ecd383f7258f75a64b1180edd46.jpg)

![](images/d4d80191fb8346b03b21d50c56e03ec90aad250fe188dd799a1c110a402ca2d9.jpg)  
Black Footed Albatross

![](images/bc1caec6b812c7326be6ce9873f97a7247ab0e3a19acfdf2d658ba10acc8a0fd.jpg)

# Abstract

A grand challenge in biology is to discover evolutionary traits—features of organisms common to a group of species with a shared ancestor in the tree of life (also referred to as phylogenetic tree). With the growing availability of image repositories in biology, there is a tremendous opportunity to discover evolutionary traits directly from images in the form of a hierarchy of prototypes. However, current prototype-based methods are mostly designed to operate over a flat structure of classes and face several challenges in discovering hierarchical prototypes, including the issue of learning over-specific features at internal nodes. To overcome these challenges, we introduce the framework of Hierarchy aligned Commonality through Prototypical Networks (HComP-Net). We empirically show that HComP-Net learns prototypes that are accurate, semantically consistent, and generalizable to unseen species in comparison to baselines on birds, butterflies, and fishes datasets.

# 1 Introduction

A central goal in biology is to discover the observable characteristics of organisms, or traits (e.g., beak color, stripe pattern, and fin curvature), that help in discriminating between species and understanding

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.

![](images/78df304517259b95ab9a155468b8919e3f4af33dd3c53d423b18be304cc94145.jpg)  
Figure 2: Examples to illustrate the problem of learning "over-specific" prototypes at internal nodes, which only cover one descendant species of the node instead of learning prototypes common to all descendants.

how organisms evolve and adapt to their environment [1]. For example, discovering traits inherited by a group of species that share a common ancestor on the tree of life (also referred to as the phylogenetic tree, see Figure 1) is of great interest to biologists to understand how organisms diversify and evolve [2]. The measurement of such traits with evolutionary signals, termed evolutionary traits, is not straightforward and often relies on subjective and labor-intensive human expertise and definitions [3, 4], hindering rapid scientific advancement [5].

With the growing availability of large-scale image repositories in biology containing millions of images of organisms [6, 7, 8], there is an opportunity for machine learning (ML) methods to discover evolutionary traits automatically from images [5, 9]. This is especially true in light of recent advances in the field of explainable ML, such as the seminal work of ProtoPNet [10] and its variants [11, 12, 13] which find representative patches in training images (termed prototypes) capturing discriminatory features for every class. We can thus cast the problem of discovering evolutionary traits into asking the following question: what image features or prototypes are common across a group of species with a shared ancestor in the tree of life that are absent in species with a different shared ancestor?

For example, in Figure 1, we can see that the four species of birds on the left descending from the blue node show the common feature of having "long tails," unlike any of the descendant species of the red node. Learning such common features at every internal node as a hierarchy of prototypes can help biologists generate novel hypotheses of species diversification (e.g., the splitting of blue and red nodes) and accumulation of evolutionary trait changes.

Despite the success of ProtoPNet [10] and its variants in learning prototypes over a flat structure of classes, applying them to discover a hierarchy of prototypes is challenging for three main reasons. First, existing methods that learn multiple prototypes for every class are prone to learning "overspecific" prototypes at internal nodes of a tree, which cover only one (or a few) of its descendant species. Figure 2 shows a few examples to illustrate the concept of over-specific prototypes. Consider the problem of learning prototypes common to descendant species of the Felidae family: Lion and Bobcat. If we learn one prototype focusing on the feature of the mane (specific only to Lion) and another prototype focusing on the feature of spotted back (specific only to Bobcat), then these two prototypes taken together can classify all images from the Felidae family. However, they do not represent common features shared between Lion and Bobcat and hence are not useful for discovering evolutionary traits. Such over-specific prototypes should be instead pushed down to be learned at lower levels of the tree (e.g., the species leaf nodes of Lion and Bobcat).

Second, while existing methods such as ProtoPShare [11], ProtoPool [12], and ProtoTree [13] allow prototypes to be shared across classes for re-usability and sparsity, in the problem of discovering evolutionary traits, we want to learn prototypes at an internal node  $n$  that are not just shared across all its descendant species but are also absent in the contrasting set of species (i.e., species descending from sibling nodes of  $n$  representing alternate paths of diversification). Third, at higher levels of the tree, finding features that are common across a large number of diverse species is challenging [14, 15]. In such cases, we should be able to abstain from finding common prototypes without hampering accuracy at the leaf nodes—a feature missing in existing methods.

To address these challenges, we present Hierarchy aligned Commonality through Prototypical Networks (HComP-Net), a framework to learn hierarchical prototypes over the tree of life for discovering evolutionary traits. Here are the main contributions of our work:

1. HComP-Net learns common traits shared by all descendant species of an internal node and avoids the learning of over-specific prototypes in contrast to baseline methods using a novel overspecificity loss.  
2. HComP-Net uses a novel discriminative loss to ensure that the prototypes learned at an internal node are absent in the contrasting set of species with different ancestry.  
3. HComP-Net includes a novel masking module to allow for the exclusion of over-specific prototypes at higher levels of the tree without hampering classification performance.  
4. We empirically show that HComP-Net learns prototypes that are accurate, semantically consistent, and generalizable to unseen species compared to baselines on data from 190 species of birds (CUB-200-2011 dataset) [8], 38 species of fishes [9], and 30 species of butterflies [16]. We show the ability of HComP-Net to generate novel hypotheses about evolutionary traits at different levels of the phylogenetic tree of organisms.

# 2 Related Works

One of the seminal lines of work in the field of prototype-based interpretability methods is the framework of ProtoPNet [10] that learns a set of "prototypical patches" from training images of every class to enable case-based reasoning. Following this work, several variants have been developed such as ProtoPShare [11], ProtoPool [12], ProtoTree [13], and HPnet [17] suited to different interpretability requirements. Among all these approaches, our work is closely related to HPnet [17], the hierarchical extension of ProtoPNet that learns a prototype layer for every parent node in the tree. Despite sharing a similar motivation as our work, HPnet is not designed to avoid the learning of over-specific prototypes or to abstain from learning common prototypes at higher levels of the tree.

Another related line of work is the framework of PIPNet [18], which uses self-supervised learning methods to reduce the "semantic gap" [19, 20] between the latent space of prototypes and the space of images, such that the prototypes in latent space correspond to the same visual concept in the image space. In HComP-Net, we build upon the idea of self-supervised learning introduced in PIPNet to learn semantically consistent hierarchy of prototypes. Our work is also related to ProtoTree [13], which structures the prototypes as nodes in a decision tree to offer more granular interpretability. However, ProtoTree differs from our work in that it learns the tree-based structure of prototypes automatically from data and cannot handle a known hierarchy. Moreover, the prototypes learned in ProtoTree are purely discriminative and allow for negative reasoning, which is not aligned with our objective of finding common traits of descendant species.

Other related works that focus on finding shared features are ProtoPShare [11] and ProtoPool [12]. Both approaches aim to find common features among classes, but their primary goal is to reduce the prototype count by exploiting similarities among classes, leading to a sparser network. This is different from our goal of finding a hierarchy of prototypes to find evolutionary traits common to a group of species (that are absent from other species).

Outside the realm of prototype-based methods, the framework of Phylogeny-guided Neural Networks (PhyloNN) [9] shares a similar motivation as our work to discover evolutionary traits by representing biological images in feature spaces structured by tree-based knowledge (i.e., phylogeny). However, PhyloNN primarily focuses on the tasks of image generation and translation rather than interpretability. Additionally, PhyloNN can only work with discretized trees with fixed number of ancestor levels per leaf node, unlike our work that does not require any discretization of the tree.

# 3 Proposed Methodology

# 3.1 HComP-Net Model Architecture

Given a phylogenetic tree with  $N$  internal nodes, the goal of HComP-Net is to jointly learn a set of prototype vectors  $\mathbf{P}_{\mathbf{n}}$  for every internal node  $n\in \{1,\dots ,N\}$ . Our architecture as shown in Figure 3 begins with a CNN that acts as a common feature extractor  $f(x;\theta)$  for all nodes, where  $\theta$  represents the learnable parameters of  $f$ .  $f$  converts an image  $x$  into a latent representation  $Z\in \mathbb{R}^{H\times W\times C}$ , where each "patch" at location  $(h,w)$  is,  $\mathbf{z}_{\mathrm{h,w}}\in \mathbb{R}^C$ . Following the feature extractor, for every node  $n$ , we initialize a set of  $K_{n}$  prototype vectors  $\mathbf{P}_{\mathbf{n}} = \{\mathbf{p}_{\mathbf{i}}\}_{i = 1}^{K_n}$ , where  $\mathbf{p}_{\mathbf{i}}\in \mathbb{R}^{C}$ . Here, the number of

![](images/9d69428f9da42ba0a9d30cd9120d54c2f047ed18f10ca36752b64f527e78d558.jpg)  
Figure 3: Schematic illustration of HComP-Net model architecture.

108 prototypes  $K_{n}$  learned at node  $n$  varies in proportion to the number of children of node  $n$ , with  $\beta$  as the proportionality constant, i.e., at each node  $n$  we assign  $\beta$  prototypes for every child node. To simplify notations, we drop the subscript  $n$  in  $\mathbf{P}_{\mathbf{n}}$  and  $K_{n}$  while discussing the operations occurring in node  $n$ .

We consider the following sequence of operations at every node  $n$ . We first compute the similarity score between every prototype in  $\mathbf{P}$  and every patch in  $Z$ . This results in a matrix  $\hat{Z} \in \mathbb{R}^{H \times W \times K}$ , where every element represents a similarity score between image patches and prototype vectors. We apply a softmax operation across the  $K$  channels of  $\hat{Z}$  such that the vector  $\hat{\mathbf{z}}_{\mathrm{h,w}} \in \mathbb{R}^K$  at spatial location  $(h, w)$  in  $\hat{Z}$  represents the probability that the corresponding patch  $\mathbf{z}_{\mathrm{h,w}}$  is similar to the  $K$  prototypes. Furthermore, the  $i^{th}$  channel of  $\hat{Z}$  serves as a prototype score map for the prototype vector  $\mathbf{p_i}$ , indicating the presence of  $\mathbf{p_i}$  in the image. We perform global max-pooling across the spatial dimensions  $H \times W$  of  $\hat{Z}$  to obtain a vector  $\mathbf{g} \in \mathbb{R}^K$ , where the  $i^{th}$  element represents the highest similarity score of the prototype vector  $\mathbf{p_i}$  across the entire image.  $\mathbf{g}$  is then fed to a linear classification layer with weights  $\phi$  to produce the final classification scores for every child node of node  $n$ . We restrict the connections in the classification layer so that every child node  $n_c$  is connected to a distinct set of  $\beta$  prototypes, to ensure that every prototype uniquely maps to a child node.  $\phi$  is restricted to be non-negative to ensure that the classification is done solely through positive reasoning, similar to the approach used in PIP-Net [18]. We borrow the regularization scheme of PIP-Net to induce sparsity in  $\phi$  by computing the logit of child node  $n_c$  as  $\log((\mathbf{g}\phi)^2 + 1)$ .  $\mathbf{g}$  and  $\phi$  here are again unique to each node.

# 128 3.2 Loss Functions Used to Train HComP-Net

129 Contrastive Losses for Learning Hierarchical Prototypes: PIP-Net [18] introduced the idea of using self-supervised contrastive learning to learn semantically meaningful prototypes. We build upon this idea in our work to learn semantically meaningful hierarchical prototypes at every node in the tree as follows. For every input image  $\mathbf{x}$ , we pass in two augmentations of the image,  $\mathbf{x}'$  and  $\mathbf{x}''$  to our framework. The prototype score maps for the two augmentations,  $\hat{Z}'$  and  $\hat{Z}''$ , are then considered as positive pairs. Since  $\hat{\mathbf{z}}_{\mathrm{h,w}} \in \mathbb{R}^K$  represents the probabilities of patch  $\mathbf{z}_{\mathrm{h,w}}$  being similar to the prototypes from  $\mathbf{P}$ , we align the probabilities from the two augmentations  $\hat{\mathbf{z}}_{\mathrm{h,w}}'$  and  $\hat{\mathbf{z}}_{\mathrm{h,w}}''$  to be similar using the following alignment loss:

$$
\mathcal {L} _ {A} = - \frac {1}{H W} \sum_ {(h, w) \in H \times W} \log \left(\hat {\mathbf {z}} _ {\mathbf {h}, \mathbf {w}} ^ {\prime} \cdot \hat {\mathbf {z}} _ {\mathbf {h}, \mathbf {w}} ^ {\prime \prime}\right) \tag {1}
$$

Since  $\sum_{i=1}^{K} \hat{\mathbf{z}}_{\mathbf{h},\mathbf{w},\mathbf{i}} = 1$  due to softmax operation,  $\mathcal{L}_A$  is minimum (i.e.,  $\mathcal{L}_A = 0$ ) when both  $\hat{\mathbf{z}}_{\mathbf{h},\mathbf{w}}'$  and  $\hat{\mathbf{z}}_{\mathbf{h},\mathbf{w}}''$  are identical one-hot encoded vectors. A trivial solution that minimizes  $\mathcal{L}_A$  is when all

patches across all images are similar to the same prototype. To avoid such representation collapse, we use the following tanh-loss  $\mathcal{L}_T$  of PIP-Net [18], which serves the same purpose as uniformity losses in [21] and [22]:

$$
\mathcal {L} _ {T} = - \frac {1}{K} \sum_ {i = 1} ^ {K} \log (\tanh  (\sum_ {b = 1} ^ {B} \mathbf {g} _ {\mathbf {b}, \mathbf {i}})), \tag {2}
$$

where  $\mathbf{g}_{\mathbf{b},\mathbf{i}}$  is the prototype score for prototype  $i$  with respect to image  $b$  of mini-batch.  $\mathcal{L}_T$  encourages each prototype  $\mathbf{p_i}$  to be activated at least once in a given mini-batch of  $B$  images, thereby helping to avoid the possibility of representation collapse. The use of tanh ensures that only the presence of a prototype is taken into account and not its frequency.

Over-specificity Loss: To achieve the goal of learning prototypes common to all descendant species of an internal node, we introduce a novel loss, termed over-specificity loss  $\mathcal{L}_{ovsp}$  that avoids learning over-specific prototypes at any node  $n$ .  $\mathcal{L}_{ovsp}$  is formulated as a modification of the tanh-loss such that prototype  $\mathbf{p_i}$  is encouraged to be activated at least once in every one of the descendant species  $d\in \{1,\dots ,D_{i}\}$  of its corresponding child node in the mini-batch of images fed to the model, as follows:

$$
\mathcal {L} _ {o v s p} = - \frac {1}{K} \sum_ {i = 1} ^ {K} \sum_ {d = 1} ^ {D _ {i}} \log (\tanh  (\sum_ {b \in B _ {d}} \mathbf {g} _ {\mathbf {b}, \mathbf {i}})), \tag {3}
$$

where  $B_{d}$  is the subset of images in the mini-batch that belong to species  $d$ .

Discriminative loss: In order to ensure that a learned prototype for a child node  $n_c$  is not activated by any of its contrasting set of species (i.e., species that are descendants of child nodes of  $n$  other than  $n_c$ ), we introduce another novel loss function,  $\mathcal{L}_{disc}$ , defined as follows:

$$
\mathcal {L} _ {d i s c} = \frac {1}{K} \sum_ {i = 1} ^ {K} \sum_ {d \in \widetilde {D} _ {i}} \max  _ {b \in B _ {d}} (\mathbf {g} _ {\mathbf {b}, \mathbf {i}}), \tag {4}
$$

where  $\widetilde{D_i}$  is the contrasting set of all descendant species of child nodes of  $n$  other than  $n_c$ . This is similar to the separation loss used in other prototype-based methods such as [10], [13], and [23].

Orthogonality loss: We also apply kernel orthogonality as introduced in [24] to the prototype vectors at every node  $n$ , so that the learned prototypes are orthogonal and capture diverse features:

$$
\mathcal {L} _ {\text {o r t h}} = \left\| \hat {\mathbf {P}} \hat {\mathbf {P}} ^ {\top} - I \right\| _ {F} ^ {2} \tag {5}
$$

where  $\hat{\mathbf{P}}$  is the matrix of normalized prototype vectors of size  $C\times K$ ,  $I$  is an identity matrix, and  $\| \cdot \| _F^2$  is the Frobenius norm. Each prototype  $\hat{\mathbf{p}}_{\mathrm{i}}$  in  $\hat{\mathbf{P}}$  is normalized as,  $\hat{\mathbf{p}}_{\mathrm{i}} = \frac{\mathbf{p}_{\mathrm{i}}}{\|\mathbf{p}_{\mathrm{i}}\|}$ .

Classification loss: Finally, we apply cross entropy loss for classification at each internal node as follows:

$$
\mathcal {L} _ {C E} = - \sum_ {b} ^ {B} y _ {b} \log \left(\hat {y} _ {b}\right) \tag {6}
$$

where  $y$  is ground truth label and  $\hat{y}$  is the prediction at every node of the tree.

# 3.3 Masking Module to Identify Over-specific Prototypes

We employ an additional masking module at every node  $n$  to identify over-specific prototypes without hampering their training. The learned mask for prototype  $\mathbf{p_i}$  simply serves as an indicator of whether  $\mathbf{p_i}$  is over-specific or not, enabling our approach to abstain from finding common prototypes if there are none, especially at higher levels of the tree. To obtain the mask values, we first calculate the over-specificity score for prototype  $\mathbf{p_i}$  as the product of the maximum prototype scores obtained across all images in the mini-batch belonging to every descendant species  $d$  as:

$$
\mathcal {O} _ {i} = - \prod_ {d = 1} ^ {D _ {i}} \max  _ {(b \in B _ {d})} (\mathbf {g} _ {\mathbf {b}, \mathbf {i}}) \tag {7}
$$

where  $\mathbf{g}_{\mathbf{b},\mathbf{i}}$  is the prototype score for prototype  $\mathbf{p_i}$  with respect to image  $b$  of mini-batch and  $B_{d}$  is the subset of images in the mini-batch that belong to descendant species  $d$ . Since  $\mathbf{g}_{\mathbf{b},\mathbf{i}}$  takes a value between 0 to 1 due to the softmax operation,  $\mathcal{O}_i$  ranges from -1 to 0, where -1 denotes least

over-specificity and 0 denotes the most over-specificity. The multiplication of the prototype scores ensures that even when the score is less with respect to only one descendant species, the prototype will be assigned a high over-specificity score (close to 0).

As shown in Figure 3,  $\mathcal{O}_i$  is then fed into the masking module, which includes a learned mask value  $M_{i}$  for every prototype  $\mathbf{p_i}$ . We generate  $M_{i}$  from a Gumbel-softmax distribution [25] so that the values are skewed to be very close to either 0 or 1, i.e.,  $M_{i} = \mathrm{Gumbel - Softmax}(\gamma_{i},\tau)$ , where  $\gamma_{i}$  are the learnable parameters of the distribution and  $\tau$  is temperature. We then compute the masking loss,  $\mathcal{L}_{mask}$ , as:

$$
\mathcal {L} _ {\text {m a s k}} = \sum_ {i = 1} ^ {K} \left(\lambda_ {\text {m a s k}} M _ {i} \circ \operatorname {s t o p g r a d} \left(\mathcal {O} _ {i}\right) + \lambda_ {L _ {1}} \| M _ {i} \| _ {1}\right) \tag {8}
$$

where  $\lambda_{mask}$  and  $\lambda_{L_1}$  are trade-off coefficients,  $\| .\| _1$  is the  $L_{1}$  norm added to induce sparsity in the masks, and stopgrad represents the stop gradient operation applied over  $\mathcal{O}_i$  to ensure that the gradient of  $\mathcal{L}_{mask}$  does not flow back to the learning of prototype vectors and impact their training. Note that the learned masks are not used for pruning the prototypes during training, they are only used during inference to determine which of the learned prototypes are over-specific and likely to not represent evolutionary traits. Therefore, even if all the prototypes are identified as over-specific by the masking module at an internal node, it will not affect the classification performance at that node.

# 3.4 Training HComP-Net

We first pre-train the prototypes at every internal node in a self-supervised learning manner using alignment and tanh-losses as  $\mathcal{L}_{SS} = \lambda_A\mathcal{L}_A + \lambda_T\mathcal{L}_T$ . We then fine-tune the model using the following combined loss:  $(\lambda_{CE}\mathcal{L}_{CE} + \mathcal{L}_{SS} + \lambda_{ovsp}\mathcal{L}_{ovsp} + \lambda_{disc}\mathcal{L}_{disc} + \lambda_{orth}\mathcal{L}_{orth} + \mathcal{L}_{mask})$ , where  $\lambda$ 's are trade-off parameters. Note that the loss is applied over every node in the tree. We show an ablation of key loss terms in our framework in Table 6 in the Supplementary Section.

# 4 Experimental Setup

Dataset: In our experiments, we primarily focus on the 190 species of birds (Bird) from the CUB-200-2011 [8] dataset for which the phylogenetic relationship [26] is known. The tree is quite large with a total of 184 internal nodes. We removed the background from the images to avoid the possibility of learning prototypes corresponding to background information such as the bird's habitat as we are only interested in the traits corresponding to the body of the organism. We also apply our method on a fish dataset with 38 species (Fish) [9] along with its associated phylogeny [9] and 30 subspecies of Heliconius butterflies (Butterfly) from the Jiggins Heliconius Collection dataset [16] collected from various sources  ${}^{1}$  along with its phylogeny [52, 53]. The qualitative results of Butterfly and Fish datasets are provided in the supplementary materials. The complete details of hyper-parameter settings and training strategy are also provided in the Supplementary Section E.

Baselines: We compare HComP-Net to ResNet-50 [54], INTR (Interpretable Transformer) [55] and HPnet [17]. For HPnet, we used the same hyperparameter settings and training strategy as used by ProtoPNet for CUB-200-2011 dataset. For a fair comparison, we also set the number of prototypes for each child in HPnet to be equal to 10 similar to our implementation. We follow the same training strategy as provided by ProtoPNet for CUB-200-2011 dataset.

# 5 Results

# 5.1 Fine-grained Accuracy

Similar to HPnet [17], we calculate the fine-grained accuracy for each leaf node by calculating the path probability over every image. During inference, the final probability for leaf class  $Y$  given an image  $X$  is calculated as,  $P(Y|X) = P(Y^{(1)},Y^{(2)},\dots,Y^{(L)}|X) = \prod_{l = 1}^{L}P(Y^{(l)}|X)$ , where  $P(Y^{(l)}|X)$  is the probability of assigning image  $X$  to a node at level  $l$ , and  $L$  is the depth of the leaf node. Every image is assigned to the leaf class with maximum path probability, which is used to compute the fine-grained accuracy. The comparison of the fine-grained accuracy calculated for

![](images/606b8f5154df58284cce0306bd8a6935559a3d2551d590e72972a0cd895c4c5a.jpg)  
Figure 4: Comparing the part consistency of HPnet and HComP-Net for their prototype learned at an internal node in the bird dataset that corresponds to 3 descendant species (names shown on the rows). For every species, we are visualizing the top-3 images with highest prototype score for both HPnet and HComP-Net, shown as the four columns with zoomed in views of their discovered prototypes. We can see that HPnet highlights varying parts of the bird across the 3 species and across multiple images of the same species, making it difficult to associate a consistent semantic meaning to its learned prototype. In contrast, HComP-Net consistently highlights the head region of the bird across all four species and their images.

![](images/4f133a9a1693cd085289fe4fb3a9200be694fe6ff9e2c46f195e4898d37cddf2.jpg)

HComP-Net and the baselines are given in Table 1. We can see that HComP-Net performs better than the other interpretable methods, such as INTR and HPNet, and is also able to nearly match the performance of non-interpretable models, such as ResNet-50, even outperforming it for the Fish and Butterfly dataset. This shows the ability of our proposed framework to achieve competitive classification accuracy along with serving the goal of discovering evolutionary traits.

Table 1: % Accuracy  

<table><tr><td>Model</td><td>Hierarchy</td><td>Bird</td><td>Butterfly</td><td>Fish</td></tr><tr><td>ResNet-50</td><td rowspan="2">No</td><td>74.18</td><td>95.76</td><td>86.63</td></tr><tr><td>INTR</td><td>69.22</td><td>95.53</td><td>86.73</td></tr><tr><td>HPnet</td><td rowspan="2">Yes</td><td>36.18</td><td>94.69</td><td>77.51</td></tr><tr><td>HComP-Net</td><td>70.01</td><td>97.35</td><td>90.80</td></tr></table>

Table 2: % Accuracy (on unseen species)  

<table><tr><td>Species Name</td><td>HComP-Net</td><td>HPnet</td></tr><tr><td>Fish Crow</td><td>53.33</td><td>10.55</td></tr><tr><td>Rock Wren</td><td>53.33</td><td>10.22</td></tr><tr><td>Indigo Bunting</td><td>96.67</td><td>49.2</td></tr><tr><td>Bohemian Waxwing</td><td>70.00</td><td>44.9</td></tr></table>

# 5.2 Generalizing to Unseen Species in the Phylogeny

We analyze the performance of HComP-Net in generalizing to unseen species that the model hasn't seen during training. The biological motivation for this experiment is to evaluate if HComP-Net can situate newly discovered species at its appropriate position in the phylogeny by identifying its common ancestors shared with the known species. An added advantage of our work is that along with identifying the ancestor of an unseen species, we can also identify the common traits shared by the novel species with known species in the phylogeny. Since unseen species cannot be classified to the finest levels (i.e., up to the leaf node corresponding to the unseen species), we analyze the ability of HComP-Net to classify unseen species accurately up to one level above the leaf level in the hierarchy. With this consideration, the final probability of an unseen species for a given image is calculated as,  $P(Y|X_{\text{unseen}}) = P(Y^{(1)}, Y^{(2)}, \dots, Y^{(L-1)}|X) = \prod_{l=1}^{L-1} P(Y^{(l)}|X)$ . Note that we leave out the class probability at the  $L^{th}$  level, since we do not take into account the class probability of the leaf level. We leave four species from the Bird training set and calculate their accuracy during inference in Table 2. We can see that HComP-Net is able to generalize better than HPnet for all four species.

# 5.3 Analyzing the Semantic Quality of Prototypes

Following the method introduced in PIPNet [18], we assess the semantic quality of our learned prototypes by evaluating their part purity. A prototype with high part purity (close to 1) is one that consistently highlights the same image region in the score maps (corresponding to consistent local features such as the eye or wing of a bird) across images belonging to the same class. The part

purity is calculated using the part locations of 15 parts that are provided in the CUB dataset. For each prototype, we take the top-10 images from each leaf descendant. We consider the  $32 \times 32$  image patch that is centered around the max activation location of the prototype from the top-10 images. With these top-10 image patches, we calculate for each part how frequently the part is present inside

the image patch. For example, a part that is found inside the image patch 8 out of 10 times is given a score of 0.8. In PIP-Net, the highest value among the values calculated for each part is given as the part purity of the prototype. In our approach, since we are dealing with a hierarchy and taking the top-10 from each leaf descendant, a particular part, let's say the eye, might have a score of 0.5 for one leaf descendant and 0.7 for a different leaf descendant. Since we want the prototype to represent the same part for all the leaf descendants, we take the lowest score (the weakest link) among all the leaf descendants as the score of the part. By following this method, for a given prototype we can arrive at a value for each part and finally take the maximum among the values as the purity of the prototype. We take the mean of the part purity across all the prototypes and report the results in Table 3 for different ablations of HComP-Net and also HPnet, which is the only baseline method that can learn hierarchical prototypes.

We can see that HComP-Net, even without the use of over-specificity loss performs much better than HPnet due to the contrastive learning approach we have adopted from PIPNet [18]. The addition of over-specificity loss improves the part purity because over-specific prototypes tend to have poor part purity for some of the leaf descendants which will affect their overall part purity score. Further, for both ablations with and without over-specificity loss, we apply the masking module and remove masked (over-specific) prototypes during the calculation of part purity. We see that the part purity goes higher by applying the masking module, demonstrating its effectiveness in identifying over-specific prototypes. We further compute the purity of masked-out prototypes and notice that the masked-out prototypes have drastically lower part purity  $(0.29 \pm 0.17)$  compared to non-masked prototypes  $(0.77 \pm 0.16)$ . An alternative approach to learning the masking module is to identify over-specific prototypes using a fixed global threshold over  $\mathcal{O}_i$ . We show in Table 9 of Supplementary Section F, that given the right choice of such a threshold, we can identify over-specific prototypes. However, selecting the ideal threshold can be non-trivial. On the other hand, our masking module learns the appropriate threshold dynamically as part of the training process.

Figure 4 visualizes the part consistency of prototypes discovered by HComP-Net in comparison to HPnet for the bird dataset. We can see that HComP-Net is finding a consistent region in the image (corresponding to the head region) across all three descendant species and all images of a species, in contrast to HPnet. Furthermore, thanks to the alignment loss, every patch  $\hat{\mathbf{z}}_{\mathrm{h,w}}$  is encoded as nearly a one-hot encoding with respect to the  $K$  prototypes which causes the prototype score maps to be highly localized. The concise and focused nature of the prototype score maps makes the interpretation much more effective compared to baselines.

# 5.4 Analyzing Evolutionary Traits Discovered by HComP-Net

We now qualitatively analyze some of the hypothesized evolutionary traits discovered in the hierarchy of prototypes learned by HComP-Net. Figure 5 shows the hierarchy of prototypes discovered over a small subtree of the phylogeny from Bird (four species) and Fish (three species) dataset. In the visualization of bird prototypes, we can see that the two Pelican species share a consistent region in the learned Prototype labeled 2, which corresponds to the head region of the birds. We can hypothesize this prototype to be capturing the white colored crown common to the two species. On the other hand, Prototype 1 finds the shared trait of similar beak morphology (e.g., sharpness of beaks) across the two Cormorant species. We can see that HComP-Net avoids the learning of over-specific prototypes at internal nodes, which are pushed down to individual leaf nodes, as shown in visualizations of Prototype 3, 4, 5, and 6. Similarly, in the visualization of the fish prototypes, we can see that Prototype 1 is highlighting a specific fin (dorsal fin) of the Carassius auratus and Notropis hudsonius species, possibly representing their pigmentation and structure, which is noticeably different compared to the contrasting species of Alosa chrysochloris. Note that while HComP-Net identifies the common

![](images/38ca831924dc8e6ddca0ba1ea4d1b38c87146d2adcca3b714a05f3559b043865.jpg)  
Figure 5: Visualizing the hierarchy of prototypes discovered by HComP-Net for birds and fishes. *Note that the textual descriptions of the hypothesized traits shown for every prototype are based on human interpretation.

![](images/6880af3ac3856d2e5db48e9b4d44eb98a566f24422b4de4982b056e2c8affac0.jpg)  
Figure 6: We trace the prototypes learned for Western Grebe at three different levels in the phylogenetic tree (corresponding to different periods of time in evolution). Text in blue is the interpretation of common traits of descendants found by HComP-Net at every ancestor node of Western Grebe.

regions corresponding to each prototype (shown as heatmaps), the textual descriptions of the traits provided in Figure 5 are based on human interpretation.

Figure 6 shows another visualization of the sequence of prototypes learned by HComP-Net for the Western Grebe species at different levels of the phylogeny. We can see that at level 0, we are capturing features closer to the neck region, indicating the likely difference between the length of necks between Grebe species and other species (Cuckoo, Albatross, and Fulmar) that diversify at an earlier time in the process of evolution. At level 1, the prototype is focusing on the eye region, potentially indicating to difference in the color of red and black patterns around the eyes. At level 2, we are differentiating Western Grebe from Horned Grebe based on the feature of bills. We also validate our prototypes by comparing them with the multi-head cross-attention maps learned by INTR [55]. We can see that some of the prototypes discovered by HComP-Net can be mapped to equivalent attention heads of INTR. However, while INTR is designed to produce a flat structure of attention maps, we are able to place these maps on the tree of life. This shows the power of HComP-Net in generating novel hypotheses about how trait changes may have evolved and accumulated across different branches of the phylogeny. Additional visualizations of discovered evolutionary traits for butterfly species and fish species are provided in the supplementary section in Figures 7 to 16.

# 6 Conclusion

We introduce a novel approach for learning hierarchy-aligned prototypes while avoiding the learning of over-specific features at internal nodes of the phylogenetic tree, enabling the discovery of novel evolutionary traits. Our empirical analysis on birds, fishes, and butterflies, demonstrates the efficacy of HComP-Net over baseline methods. Furthermore, HComP-Net demonstrates a unique ability to generate novel hypotheses about evolutionary traits, showcasing its potential in advancing our understanding of evolution. We discuss the limitations of our work in Supplementary Section I. While we focus on the biological problem of discovering evolutionary traits, our work can be applied in general to domains involving a hierarchy of classes, which can be explored in future research.

# References

[1] David Houle and Daniela M Rossoni. Complexity, evolvability, and the process of adaptation. Annual Review of Ecology, Evolution, and Systematics, 53, 2022.  
[2] Maureen A O'Leary and Seth Kaufman. Morphobank: phylophenomics in the "cloud". *Cladistics*, 27(5):529-537, 2011.  
[3] Tiago R Simões, Michael W Caldwell, Alessandro Palci, and Randall L Nydam. Giant taxon-character matrices: quality of character constructions remains critical regardless of size. *Cladistics*, 33(2):198–219, 2017.  
[4] Paul C Sereno. Logical basis for morphological characters in phylogenetics. *Cladistics*, 23(6):565-587, 2007.  
[5] Moritz D Lurig, Seth Donoughe, Erik I Svensson, Arthur Porto, and Masahito Tsuboi. Computer vision, machine learning, and the promise of phenomics in ecology and evolutionary biology. Frontiers in Ecology and Evolution, 9:642774, 2021.  
[6] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8769-8778, 2018.  
[7] Randal A Singer, Kevin J Love, and Lawrence M Page. A survey of digitized data from us fish collections in the idigbio data aggregator. *PloS one*, 13(12):e0207636, 2018.  
[8] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.  
[9] Mohannad Elhamod, Mridul Khurana, Harish Babu Manogaran, Josef C Uyeda, Meghan A Balk, Wasila Dahdul, Yasin Bakis, Henry L Bart Jr, Paula M Mabee, Hilmar Lapp, et al. Discovering novel biological traits from images using phylogeny-guided neural networks. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3966-3978, 2023.  
[10] Chaofan Chen, Oscar Li, Daniel Tao, Alina Barnett, Cynthia Rudin, and Jonathan K Su. This looks like that: deep learning for interpretable image recognition. Advances in neural information processing systems, 32, 2019.  
[11] Dawid Rymarczyk, Łukasz Struski, Jacek Tabor, and Bartosz Zielinski. Protopshare: Prototypical parts sharing for similarity discovery in interpretable image classification. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 1420-1430, 2021.  
[12] Dawid Rymarczyk, Łukasz Struski, Michal Górszczak, Koryna Lewandowska, Jacek Tabor, and Bartosz Zielinski. Interpretable image classification with differentiable prototypes assignment. In European Conference on Computer Vision, pages 351-368. Springer, 2022.  
[13] Meike Nauta, Ron Van Bree, and Christin Seifert. Neural prototype trees for interpretable fine-grained image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14933-14943, 2021.  
[14] Luke J Harmon, Jonathan B Losos, T Jonathan Davies, Rosemary G Gillespie, John L Gittleman, W Bryan Jennings, Kenneth H Kozak, Mark A McPeek, Franck Moreno-Roark, Thomas J Near, et al. Early bursts of body size and shape evolution are rare in comparative data. Evolution, 64(8):2385-2396, 2010.  
[15] Matthew W Pennell, Richard G FitzJohn, William K Cornwell, and Luke J Harmon. Model adequacy and the macroevolution of angiosperm functional traits. The American Naturalist, 186(2):E33-E50, 2015.  
[16] Christopher Lawrence and Elizabeth G. Campolongo. Heliconius collection (cambridge butterfly), 2024.

[17] Peter Hase, Chaofan Chen, Oscar Li, and Cynthia Rudin. Interpretable image recognition with hierarchical prototypes. In Proceedings of the AAAI Conference on Human Computation and Crowdsourcing, volume 7, pages 32-40, 2019.  
[18] Meike Nauta, Jörg Schlötterer, Maurice van Keulen, and Christin Seifert. Pip-net: Patch-based intuitive prototypes for interpretable image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2744-2753, 2023.  
[19] Adrian Hoffmann, Claudio Fanconi, Rahul Rade, and Jonas Kohler. This looks like that... does it? shortcomings of latent space prototype interpretability in deep networks. arXiv preprint arXiv:2105.02968, 2021.  
[20] Sunnie SY Kim, Nicole Meister, Vikram V Ramaswamy, Ruth Fong, and Olga Russakovsky. Hive: Evaluating the human interpretability of visual explanations. In European Conference on Computer Vision, pages 280–298. Springer, 2022.  
[21] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International Conference on Machine Learning, pages 9929-9939. PMLR, 2020.  
[22] Thalles Silva and Adín Ramírez Rivera. Representation learning via consistent assignment of views to clusters. In Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing, pages 987-994, 2022.  
[23] Jiaqi Wang, Huafeng Liu, Xinyue Wang, and Liping Jing. Interpretable image recognition by constructing transparent embedding space. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 895–904, 2021.  
[24] Jiayun Wang, Yubei Chen, Rudrasis Chakraborty, and Stella X Yu. Orthogonal convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11505-11515, 2020.  
[25] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.  
[26] W. Jetz, G. H. Thomas, J. B. Joy, K. Hartmann, and A. O. Mooers. The global diversity of birds in space and time. Nature, 491:444-448, 2012.  
[27] Gabriela Montejo-Kovacevich, Eva van der Heijden, Nicola Nadeau, and Chris Jiggins. Cambridge butterfly wing collection batch 10, November 2020.  
[28] Patricio A. Salazar, Nicola Nadeau, Gabriela Montejo-Kovacevich, and Chris Jiggins. Sheffield butterfly wing collection - Patricio Salazar, Nicola Nadeau, Ikiam broods batch 1 and 2, November 2020.  
[29] Gabriela Montejo-Kovacevich, Chris Jiggins, and Ian Warren. Cambridge butterfly wing collection batch 2, May 2019.  
[30] Chris Jiggins, Gabriela Montejo-Kovacevich, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 3, May 2019.  
[31] Gabriela Montejo-Kovacevich, Chris Jiggins, and Ian Warren. Cambridge butterfly wing collection batch 4, May 2019.  
[32] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 5, May 2019.  
[33] Ian Warren and Chris Jiggins. Miscellaneous Heliconius wing photographs (2001-2019) Part 1, February 2019.  
[34] Ian Warren and Chris Jiggins. Miscellaneous Heliconius wing photographs (2001-2019) Part 3, February 2019.  
[35] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 6, May 2019.

[36] Chris Jiggins and Ian Warren. Cambridge butterfly wing collection - Chris Jiggins 2001/2 broods batch 1, January 2019.  
[37] Chris Jiggins and Ian Warren. Cambridge butterfly wing collection - Chris Jiggins 2001/2 broods batch 2, January 2019.  
[38] Joana I. Meier, Patricio Salazar, Gabriela Montejo-Kovacevich, Ian Warren, and Chris Jggs. Cambridge butterfly wing collection - Patricio Salazar PhD wild specimens batch 3, October 2020.  
[39] Gabriela Montejo-Kovacevich, Chris Jiggins, and Ian Warren. Cambridge butterfly wing collection batch 1- version 2, May 2019.  
[40] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, Camilo Salazar, Marianne Elias, Imogen Gavins, Eva Wiltshire, Stephen Montgomery, and Owen McMillan. Cambridge and collaborators butterfly wing collection batch 10, May 2019.  
[41] Patricio Salazar, Gabriela Montejo-Kovacevich, Ian Warren, and Chris Jiggins. Cambridge butterfly wing collection - Patricio Salazar PhD wild and bred specimens batch 1, December 2018.  
[42] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 7, May 2019.  
[43] Patricio Salazar, Gabriela Montejo-Kovacevich, Ian Warren, and Chris Jiggins. Cambridge butterfly wing collection - Patricio Salazar PhD wild and bred specimens batch 2, January 2019.  
[44] Erika Pinheiro de Castro, Christopher Jiggins, Karina Lucas da Silva-Brand00e3o, Andre Victor Lucci Freitas, Marcio Zikan Cardoso, Eva Van Der Heijden, Joana Meier, and Ian Warren. Brazilian Butterflies Collected December 2020 to January 2021, February 2022.  
[45] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 8, May 2019.  
[46] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, Eva Wiltshire, and Imogen Gavins. Cambridge butterfly wing collection batch 9, May 2019.  
[47] Gabriela Montejo-Kovacevich, Eva van der Heijden, and Chris Jiggins. Cambridge butterfly collection - GMK Broods Ikiam 2018, November 2020.  
[48] Gabriela Montejo-Kovacevich, Quentin Paynter, and Amin Ghane. Heliconius erato cyrbia, Cook Islands (New Zealand) 2016, 2019, 2021, September 2021.  
[49] Ian Warren and Chris Jiggins. Miscellaneous Heliconius wing photographs (2001-2019) Part 2, February 2019.  
[50] Camilo Salazar, Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Imogen Gavins. Camilo Salazar and Cambridge butterfly wing collection batch 1, May 2019.  
[51] Anniina Mattila, Chris Jiggins, and Ian Warren. University of Helsinki butterfly collection - Anniina Mattila bred specimens, February 2019.  
[52] OpenTreeOfLife, Benjamin Redelings, Luna Luisa Sanchez Reyes, Karen A. Cranston, Jim Allman, Mark T. Holder, and Emily Jane McTavish. Open tree of life synthetic tree, 2019.  
[53] Francois Michonneau, Joseph W. Brown, and David J. Winter. rotl: an r package to interact with the open tree of life data. Methods in Ecology and Evolution, 7(12):1476-1481, 2016.  
[54] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770-778, 2016.  
[55] Dipanjyoti Paul, Arpita Chowdhury, Xinqi Xiong, Feng-Ju Chang, David Carlyn, Samuel Stevens, Kaiya Provost, Anuj Karpatne, Bryan Carstens, Daniel Rubenstein, et al. A simple interpretable transformer for fine-grained image classification and analysis. arXiv preprint arXiv:2311.04157, 2023.

[56] Abien Fred Agarap. Deep learning using rectified linear units (relu). arXiv preprint arXiv:1803.08375, 2018.  
[57] Samuel G Müller and Frank Hutter. Trivialaugment: Tuning-free yet state-of-the-art data augmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 774-782, 2021.  
[58] R. Farrell. Cub-200-2011 segmentations (1.0) [data set], 2024.
