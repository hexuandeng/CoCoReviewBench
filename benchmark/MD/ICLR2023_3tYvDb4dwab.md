# UNDERSTANDING SELF-SUPERVISED PRETRAINING WITH PART-AWARE REPRESENTATION LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

In this paper, we are interested in understanding self-supervised pretraining through studying the capability that self-supervised representation pretraining methods learn part-aware representations. The study is mainly motivated by that random views, used in contrastive learning, and random masked (visible) patches, used in masked image modeling, are often about object parts.

We explain that masked image modeling is a part-to-part task: the masked patches of the object are hallucinated from the visible patches, and that contrastive learning is a part-to-whole task: the projection layer hallucinates the whole object representation from the object part representation learned from the encoder. The explanation suggests that the self-supervised pretrained encoder is required to understand the object part. We empirically compare the off-the-shelf encoders pretrained with several representative methods on object-level recognition and part-level recognition. The results show that the fully-supervised model outperforms self-supervised models for object-level recognition, and most self-supervised contrastive learning and masked image modeling methods outperform the fully-supervised method for part-level recognition. It is observed that the combination of contrastive learning and masked image modeling further improves the performance.

# 1 INTRODUCTION

Self-supervised representation pretraining has been attracting a lot of research efforts recently. The goal is to train an encoder that maps an image to a representation from visual contents without the necessity of human annotation, expecting that the encoder benefits the downstream tasks, e.g., segmentation and detection.

There are two main frameworks: contrastive learning<sup>1</sup> and masked image modeling. Contrastive learning aims to maximize the agreement of the embeddings of random augmented views from the same image. Masked image modeling partitions an image into masked patches and visible patches, and makes predictions for masked patches from visible patches. Figure 1 gives examples of random views for contrastive learning and masked and visible patches for masked image modeling.

We observe that a random view and a set of masked (visible) patches usually contain a portion of an object. It is also reported in self-supervised learning methods, e.g., DINO (Caron et al., 2021) and iBOT (Zhou et al., 2021), that different attention heads in ViTs can attend to different semantic regions or parts of an object. In light of this, we attempt to understand self-supervised pretraining by studying the capability that the pretrained encoder learns part representations.

We present a part-to-whole explanation for typical contrastive learning methods (e.g., SimCLR (Chen et al., 2020), MoCo (Chen et al., 2021), and BYOL (Grill et al., 2020)): the embedding of the whole object is hallucinated from the embedding of the part of the object contained in the random crop through a projection layer. In this way, embeddings of random crops from the same image naturally agree with each other. Masked image modeling is a part-to-part process: the embeddings of the masked patches of the object (a part of the object), are hallucinated from the visible patches (the other part of the object).

![](images/13a12399e3592a5ddcbad8ba9c86feb8ff2819b43c7a3a89a2caeedd04f669f7.jpg)  
(a)

![](images/31319638037f577cd5be76ca2c74f5debd2a3fff1b7d06ecf4380e4910698366.jpg)  
(b)

![](images/a1f2a98be25a21c9cef9f711c34b56ad7e46cbedda895bd4a3ac90edddc61d13.jpg)  
(c)

![](images/97ec38d611a31d43ec99e10f2d7617334722f230d568e994d8c2310c173046a1.jpg)  
(d)

![](images/9daeb7d5ba9aeaba5ff23b5d96edd386403ec251c18425454995886da46b8720.jpg)  
(e)

![](images/16d7ec947af05632bbbb1f6e33959970fe5b376789d8c20154c5ef531a9cd323.jpg)  
Figure 1: (a) original image, (b-c) two random crops, and (d-e) masked and visible patches.  
Figure 2: Top-24 patch retrieval results with three frozen encoders of DeiT, MoCo v3, and CAE, by taking the patch in the red box as the query. It can be seen that the retrieved results from CAE and MoCo v3 are about the object part (wing and dog mouth) and more precise than DeiT (about the whole object) implying that self-supervised pretraining methods, CAE and MoCo v3 are stronger at learning part-aware representations than the fully-supervised method DeiT.

We empirically compare the supervised model DeiT (Touvron et al., 2020) and typical self-supervised representation pretraining methods, including MoCo v3 (Chen et al., 2021), DINO (Caron et al., 2021), CAE (Chen et al., 2022a), MAE (He et al., 2021), BEiT (Bao et al., 2021), and iBOT (Zhou et al., 2021), on object-level recognition (image classification and object segmentation) and part-level recognition (patch retrieval, patch classification, and part segmentation). Figure 2 presents patch retrieval results using the encoders learned through CAE, MoCo v3, and DeiT, implying that the encoders pretrained by CAE and MoCo v3 are able to learn part-aware representations.

Through extensive studies and comparisons, we make the following observations. 1) DeiT outperforms contrastive learning and MIM methods except iBOT in object-level recognition tasks, which may benefit from its explicit object-level supervision. 2) In contrast, self-supervised methods learn better part-aware representations than DeiT. For example, while DeiT is superior to DINO and CAE by  $0.4\%$  and  $2.3\%$  on ADE20K object segmentation, DINO and CAE outperform DeiT by  $1.6\%$  and  $1.1\%$  on ADE20K part segmentation, respectively. 3) In contrastive learning, the encoder can learn part-aware information, while the projected representation tends to be more about the whole object. The evidence could be found in part retrieval experiments on MoCo v3, DINO, and iBOT. 4) The MIM method CAE shows good potential in part-aware representation learning. Interestingly, the method combines contrastive learning and MIM is promising, e.g., iBOT learns better representations at both object and part levels.

To summarize, this paper presents the following contributions:

- We study the capability of learning part-aware representations as a way of understanding self-supervised representation pretraining.  
- We explain masked image modeling as a part-to-part task and contrastive learning as a part-to-whole task, and speculate that self-supervised pretraining has the potential for learning part-aware representations.

- We empirically compare several pretrained models on object-level and part-level recognition tasks, showing interesting findings with supporting evidence of the capability of part-aware representation learning for self-supervised learning.

# 2 RELATED WORK

Contrastive learning. Contrastive pretraining has been an intense academic field in the CNN era. In this work, we use it to refer to methods for comparing random views (Caron et al., 2020; Chen et al., 2020; Zbontar et al., 2021; Xie et al., 2021a; Chen et al., 2021; Caron et al., 2021), including some instance discrimination work such as (Grill et al., 2020; Chen & He, 2021; Bardes et al., 2021). As one of the representative works, SimCLR (Chen et al., 2020) learns representations through maximizing agreement between different views of the same image in the latent space. BYOL (Grill et al., 2020) uses two asymmetrical networks to bootstrap latent representation without negative samples involved during the interaction. As vision transformer (ViT) (Dosovitskiy et al., 2021) shows excellent performance via supervised learning, it is adopted subsequently in contrastive pertaining, and numerous outstanding works are proposed. For example, MoCo v3 (Chen et al., 2021) observes the hidden instability while training self-supervised ViT and solves it by using a fixed random patch projection. DINO (Caron et al., 2021) explores new properties derived from self-supervised ViT and accordingly designs a learning strategy interpreted as a form of self-distillation with no labels.

Masked image modeling (MIM). Masked image modeling is another self-supervised pretraining paradigm that attracts much attention recently. BEiT (Bao et al., 2021) follows masked language modeling in the natural language process (NLP) area and predicts tokens via mapping image patches by d-VAE (Ramesh et al., 2021). PeCo (Dong et al., 2021) boosts BEiT by taking into consideration more semantic information in visual tokens. MAE (He et al., 2021) learns rich hidden information by directly performing masked image reconstruction in RGB color space using ViT while SimMIM (Xie et al., 2021b) uses Swin-transformer (Liu et al., 2021). CAE (Chen et al., 2022a) adds a regressor between encoder and decoder, which is designed to align unmasked patches with masked ones, leading to a pure context encoder. Recently, a trend that combines MIM with siamese frameworks has surfaced and showed encouraging results including MST (Li et al., 2021), SplitMask (El-Nouby et al., 2021), iBOT (Zhou et al., 2021), dBOT (Liu et al., 2022), and SIM (Tao et al., 2022).

Understanding self-supervised contrastive pretraining. The studies on understanding contrastive pretraining (Saunshi et al., 2022; Chen et al., 2022b; Zhong et al., 2022; Wei et al., 2022) mainly focus on random augmentations (views), contrastive loss function and its variants under the assumption that: the augmentations of inputs from the same class have significant overlap in the representation space, but there is little overlap for inputs from different classes. Our work is complementary to these studies. Inspired by the observation that random views usually contain a portion of an object, and methods (Caron et al., 2021; Zhou et al., 2021) show that different attention heads in ViTs can attend to different semantic regions of an object, we investigate what the encoder and the projector do in typical self-supervised contrastive pretraining. We speculate that the pretraining task is a part-to-whole problem, predicting the representation of the whole object through the projector from the representation (obtained from the encoder) of the part of an object. We use empirical results to verify our analysis.

Understanding self-supervised masked image modeling. The comparison of attention in different layers between the pretrained models from MIM and the supervised approach is conducted: MIM pretraining brings locality to the trained model with sufficient diversity on the attention heads (Xie et al., 2022a). Consistent with the analysis in NLP, empirical studies are conducted in Xie et al. (2022b) to verify that MIM benefits from larger models, more data, and longer training. CAE (Chen et al., 2022a) gives the comparison between contrastive and MIM and shows MIM cares about all patches and thus achieves better results for fine-tuning. Cao et al. (2022) provides a mathematical understanding of MIM. Kong & Zhang (2022) points out that the learned occlusion invariant feature contributes to the success of MIM. In this work, we speculate that masked image modeling is a part-to-part process: the embeddings of the masked part of the object are hallucinated from the visible part using the position information of the masked patches, leading to better part-aware representation than the supervised model DeiT (Touvron et al., 2020).

![](images/eab40dd07ab173e6bb2ddc80ce8caccf55e723d1f7e06599cfd8fafdcce75f5e.jpg)  
Figure 3: The pipeline of a typical contrastive learning approach. Two augmented views, red box and blue box, are generated from the original image. The augmented view in red is fed into the encoder and the projector, and then the predictor (which does not appear in earlier works like MoCo (Chen et al., 2021) and SimCLR (Chen et al., 2020)), and the view in blue is fed into the encoder and the projector. The two outputs are expected to be aligned. The gradient is stopped for the bottom stream.

# 3 UNDERSTANDING CONTRASTIVE LEARNING AND MASK IMAGE MODELING

# 3.1 CONTRASTIVE LEARNING

Contrastive learning aims to learn the encoder through maximizing the agreement between differently augmented views of the same image in the representation space. An example pipeline is depicted in Figure 3. Given an image  $\mathbf{l}$ , the augmentations, e.g., random cropping, random color distortion, and random Gaussian blur, are applied to generate a set of  $N$  augmented views,  $\{\mathsf{V}_1,\mathsf{V}_2,\dots ,\mathsf{V}_N\}$ . An augmented view  $\mathsf{V}_n$  is fed into an encoder Encoder, generating the encoded representation  $\mathbf{x}_n$ , and followed by a projector, generating the projection  $\mathbf{z}_n$ . The basic goal is to maximize the agreement between the projections  $\{\mathbf{z}_1,\mathbf{z}_2,\dots ,\mathbf{z}_N\}$ , i.e., minimize the loss

$$
\mathcal {L} _ {\mathrm {C P T}} = \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {N} \ell (\text {P r o j e c t o r (E n c o d e r} (\mathrm {V} _ {i})) \text {, P r o j e c t o r (E n c o d e r} (\mathrm {V} _ {j}))). \tag {1}
$$

In the formulation with a contrastive loss, the agreement between the projections of random augmentations from different images is minimized.

Part-to-whole prediction explanation. Let us consider two crops randomly sampled from the original image (see the examples given in Figure 1(b-c)). The encoded representation of the first crop is expected to describe a part of the object dog; the encoded representation of the second crop is expected to describe another part of the object  $\mathrm{dog}^2$ . The two representations are related but different. Contrastive learning methods project the two encoded representations into two projected representations that are expected to agree. We hypothesize that the projection process maps the encoded part representation to the representation of the whole object<sup>3</sup>. Through this way, the projected representations will agree to different views from the same image. It is assumed that the part-to-whole projection is more reliable if the encoded representation is semantically richer and is able to describe the part information. The part-to-whole process suggests that the encoder pretrained by contrastive learning methods is potentially capable of learning part-aware representations.

Figure 4 provides patch search results of a representative contrastive learning method MoCo v3 (Chen et al., 2021) based on the encoded representations before and after the projections. One can see that the results through the encoded representations are mainly about the local part, and the results through the projections tend to include the other parts of the same object. In other words, the projections tend to be about the whole object. Similar observations are also shown in Chen et al. (2022b). The search results verify the part-to-whole hypothesis.

# 3.2 MASKED IMAGE MODELING

Mask image modeling is the task of predicting some parts of an image from the remaining parts. An augmented view of an image is partitioned into patches,  $\mathcal{R} = \{\mathsf{R}_1,\mathsf{R}_2,\dots ,\mathsf{R}_M\}$ . The task is

![](images/e1082815bdc6c769b806f461d261939f9a27aa1f4ef06a42b434078a405ad0c8.jpg)

![](images/9414c9dcf2064fdb8517ef658a79e4af8c7704f91307b34eaedfa8045c5390d5.jpg)  
Encoded representation

![](images/2bbd0c7f2d3908325a9b638e51ff1e05dc9d00d4cb882f4daf127a438b1e0e60.jpg)  
Projected representation

![](images/2e0a7844856e68ae9e7fd6dc5e6f9d7fb11dfa42b34143654182916529f0e3d9.jpg)  
Figure 4: Illustration of patch search results using encoded representations and projections (pretrained with MoCo v3 as). Left: patch search results with encoded representations. Right: patch search results with projections. In each result, the small patch encircled by the red box is taken as the query. It can be seen that for encoded representations, the returned patches are about the same part, and for projections, the result patches are about the same object, verifying the part-to-whole hypothesis.

![](images/556a8f1d79f290d76fe226f1c9c76fba02c1fef3f0994c69e3aef03bceadc152.jpg)

![](images/e6a6ba97ae7f2cfff59a79dc7a762bcbdb39382b5f75e85d18be69e08391b682.jpg)

to predict a subset of patches  $\mathcal{R}_m$ , named masked patches, from the remaining patches  $\mathcal{R}_v$ , named visible patches. Considering contrastive learning that explicitly compares representations of random views, we take context autoencoder (CAE) (Chen et al., 2022a) as an example that explicitly predicts the encoded representations of the masked patches from the encoded representations of the visible patches<sup>4</sup>.

One goal of CAE (illustrated in Figure 5), which we call masked representation modeling (MRM), is to maximize the agreement between the predictions of the representations of masked patches (through a regressor) and the representation of masked patches computed from the encoder by minimizing the loss

$$
\ell_ {\mathrm {M R M}} (\operatorname {R e g u s s o r} (\operatorname {E n c o d e r} \left(\mathcal {P} _ {v}\right)), \operatorname {E n c o d e r} \left(\mathcal {P} _ {m}\right)). \tag {2}
$$

Here, we do not include the positional embeddings of masked and visible patches for clarity. It is noted that MRM differs from contrastive learning: MRM does not compare multiple random views, but compares the regressed representations for masked patches and the encoded representations of masked patches. In addition, there is another loss for target prediction (reconstruction) for the masked patches, which is commonly used in masked image modeling (MIM) methods:

$$
\ell_ {\mathrm {M I M}} (\operatorname {D e c o d e r} (\operatorname {R e g u s s e r} (\operatorname {E n c o d e r} (\mathcal {P} _ {v}))), \operatorname {T a r g e t} (\mathcal {P} _ {m})), \tag {3}
$$

where  $\mathrm{Target}(\mathcal{P}_m)$  is a function to map the masked patches to the targets, e.g., d-VAE (Ramesh et al., 2021) token used in CAE and BeiT (Bao et al., 2021), or normalized RGB values used in MAE (He et al., 2021).

Part-to-part prediction explanation. The masked image modeling approaches, including CAE, MAE, and BEiT, make use of the positions of masked patches for making predictions for masked patches from visible patches. The visible patches and masked patches often contain different parts of an object. In other words, MIM aims to predict the masked part of an object from the visible part. We name this a part-to-part process. There are two part-to-part tasks: one is to reconstruct the part targets from the visible part representations (MAE and CAE) or from the visible part raw pixels (BEiT), and the other one is to regress the masked part representations (CAE). The part-to-part process suggests that the encoder pretrained by MIM methods is potentially capable of learning part-aware representations. Figure 2 illustrates the capability with the patch retrieval results.

![](images/d089df07f2cec425bb5eb8f0f2d038c38698af2390e420c3f37f6be79706aeb1.jpg)  
Figure 5: The pipeline of an MIM approach, context autoencoder (CAE). An augmented view (in blue) of the image is partitioned into visible and masked patches. The CAE approach feeds visible patches into the encoder and extracts their representations  $\mathbf{Z}_v$  and then completes the pretext task by predicting the representations  $\mathbf{Z}_m$  of the masked patches from the visible patches in the encoded representation space with latent contextual regressor and alignment constraint, and mapping predicted representations  $\mathbf{Z}_m$  of masked patches to the targets. The pretrained encoder in (a) is applied to downstream tasks by simply replacing the pretext task part (b, c) with the downstream task completion part.

Table 1: Top-1 accuracy with linear probing, and attentive probing (Chen et al., 2022a), on the ImageNet classification benchmark (Deng et al., 2009).  

<table><tr><td>Method</td><td>Linear</td><td>Attentive</td></tr><tr><td colspan="3">Supervised Model:</td></tr><tr><td>DeiT</td><td>81.8</td><td>81.8</td></tr><tr><td colspan="3">Contrastive Learning:</td></tr><tr><td>MoCo v3</td><td>76.2</td><td>77.0</td></tr><tr><td>DINO</td><td>77.3</td><td>77.8</td></tr><tr><td colspan="3">Masked Image Modeling (MIM):</td></tr><tr><td>BEiT</td><td>41.8</td><td>51.9</td></tr><tr><td>MAE</td><td>67.8</td><td>74.2</td></tr><tr><td>CAE</td><td>70.4</td><td>77.1</td></tr><tr><td colspan="3">Contrastive Learning + MIM:</td></tr><tr><td>iBOT</td><td>79.5</td><td>79.8</td></tr></table>

Table 2: Linear evaluation of ADE20K (Zhou et al., 2019) object-level semantic segmentation (150 classes) using  $4 \times$  upsampling and a single  $1 \times 1$  convolutional layer on frozen backbones.  

<table><tr><td>Method</td><td>mIoU</td><td>mAcc</td><td>aAcc</td></tr><tr><td colspan="4">Supervised Model:</td></tr><tr><td>DeiT</td><td>34.9</td><td>44.2</td><td>75.4</td></tr><tr><td colspan="4">Contrastive Learning:</td></tr><tr><td>MoCo v3</td><td>34.7</td><td>43.9</td><td>75.9</td></tr><tr><td>DINO</td><td>34.5</td><td>43.5</td><td>76.1</td></tr><tr><td colspan="4">Masked Image Modeling (MIM):</td></tr><tr><td>BEiT</td><td>17.8</td><td>23.7</td><td>64.9</td></tr><tr><td>MAE</td><td>27.1</td><td>34.8</td><td>71.6</td></tr><tr><td>CAE</td><td>32.6</td><td>42.2</td><td>75.2</td></tr><tr><td colspan="4">Contrastive Learning + MIM:</td></tr><tr><td>iBOT</td><td>38.3</td><td>47.4</td><td>78.1</td></tr></table>

# 4 EXPERIMENTS

We study seven representative methods with the same ViT-B encoder, including a supervised method DeiT (Touvron et al., 2020); contrastive learning methods MoCo v3 (Chen et al., 2021), DINO (Caron et al., 2021); masked image modeling (MIM) methods BEiT (Bao et al., 2021), MAE (He et al., 2021), and CAE (Chen et al., 2022a); and iBOT (Zhou et al., 2021) that combines contrastive learning and MIM. We take the training epochs specified in each work to ensure that all compared models are properly trained: 300 for DeiT, 300  $(600^{5})$  for MoCo v3, 400  $(1600^{5})$  for DINO and iBOT, 800 for BEiT, and 1600 for MAE and CAE. Frozen encoders are used in all experiments to understand what these different representation pretraining methods learn. More details can be found in Appendix A.1.

# 4.1 OBJECT-LEVEL RECOGNITION

We benchmark two widely-studied object-level recognition, i.e., image classification and semantic segmentation to show the capability that the pretrained encoder learns object-level representations.

Table 3: Part retrieval (AP, %) and classification (accuracy, %) results on the cropped part patches of CUB-200-2011 and COCO. The "Encoded" and "Projected" refer to the encoded and projected representations. "Linear" and "Attentive" columns denote the linear probing and attentive probing accuracy, respectively.  

<table><tr><td rowspan="3">Methods</td><td colspan="4">Part Retrieval</td><td colspan="4">Part Classification</td></tr><tr><td colspan="2">CUB-200-2011</td><td colspan="2">COCO</td><td colspan="2">CUB-200-2011</td><td colspan="2">COCO</td></tr><tr><td>Encoded</td><td>Projected</td><td>Encoded</td><td>Projected</td><td>Linear</td><td>Attentive</td><td>Linear</td><td>Attentive</td></tr><tr><td colspan="9">Supervised Model:</td></tr><tr><td>DeiT</td><td>35.0</td><td>-</td><td>44.1</td><td>-</td><td>90.9</td><td>92.9</td><td>88.5</td><td>91.4</td></tr><tr><td colspan="9">Contrastive Learning:</td></tr><tr><td>MoCo v3</td><td>50.8</td><td>28.4</td><td>52.3</td><td>36.8</td><td>93.8</td><td>96.0</td><td>92.4</td><td>95.3</td></tr><tr><td>DINO</td><td>48.9</td><td>31.7</td><td>51.8</td><td>41.2</td><td>93.2</td><td>95.2</td><td>91.7</td><td>94.5</td></tr><tr><td colspan="9">Masked Image Modeling (MIM):</td></tr><tr><td>BEiT</td><td>27.9</td><td>-</td><td>35.3</td><td>-</td><td>55.4</td><td>86.5</td><td>69.3</td><td>86.5</td></tr><tr><td>MAE</td><td>28.5</td><td>-</td><td>37.1</td><td>-</td><td>86.9</td><td>92.8</td><td>88.0</td><td>93.9</td></tr><tr><td>CAE</td><td>58.0</td><td>-</td><td>57.0</td><td>-</td><td>89.5</td><td>95.8</td><td>91.1</td><td>95.5</td></tr><tr><td colspan="9">Contrastive Learning + MIM:</td></tr><tr><td>iBOT</td><td>49.3</td><td>31.2</td><td>59.2</td><td>41.5</td><td>93.8</td><td>95.8</td><td>92.1</td><td>95.1</td></tr></table>

Image classification. We report the linear probing, and attentive probing results of the selected models on ImageNet (Deng et al., 2009). For attentive probing, we follow the protocol in CAE (Chen et al., 2022a) that append a cross-attention layer together with a batch normalization layer and a linear classifier.

We have the following observations from Table 1. 1) The supervised model, DeiT performs better than self-supervised models at object-level recognition. 2) The models that leverage contrastive learning, i.e., MoCo, DINO, and iBOT, show superior linear probing performance than MIM-based models, demonstrating they contain more object-aware high-level semantics. 3) MIM-based models, e.g., CAE, show inferior results in linear probing while competitive results with contrastive-based methods in attentive probing. The reason might be that MIM is capable of attending to all the regions, including non-object regions in an image, thus needs a spatial feature selection step to attend to the object part, which is pointed out in Chen et al. (2022a). BEiT and MAE perform inferior, implying that the two methods are less capable of learning semantics.

Object-level semantic segmentation. We perform linear evaluation on ADE20K (Zhou et al., 2019) to show the object-level semantic capabilities of the pretrained models. A  $4 \times$  bilinear interpolation and a single  $1 \times 1$  convolutional layer for pixel labeling are attached to the frozen encoder.

We can see from Table 2 that the supervised model DeiT outperforms all self-supervised models except iBOT, including contrastive learning and MIM methods on ADE20K object-level segmentation. This implies that in general the self-supervised models are not strong at object-level understanding, which is consistent with the observations for image classification. iBOT (Zhou et al., 2021), as a combination of contrastive learning and MIM, shows surprisingly better performance than the supervised model DeiT on ADE20K, implying the power of combining contrastive learning and masked image modeling for downstream tasks.

# 4.2 PART-LEVEL RECOGNITION

Self-supervised methods like iBOT (Zhou et al., 2021) and DINO (Caron et al., 2021) qualitatively show that different attention heads in ViTs can attend to different semantic regions of an object. We conduct the quantitative evaluation for part-aware representation obtained by pretrained models that is not well explored before, through three part-level recognition tasks, part retrieval, part classification, and part segmentation.

Part retrieval. We conduct part retrieval experiments on two datasets, CUB-200-2011 (Wah et al., 2011) and COCO (Lin et al., 2014). We build the part patch databases by cropping the patches centered at the keypoint. We consider four and three keypoints from the two datasets, respectively. For each keypoint, we find the minimum L2 distance  $(d)$  from the distances between it and all the

Table 4: Part-level linear semantic segmentation results on ADE20K-Part, Pascal-Part, and LIP datasets.  

<table><tr><td rowspan="2">Methods</td><td colspan="3">ADE20K-Part
209 Part Classes</td><td colspan="3">Pascal-Part
193 Part Classes</td><td colspan="3">LIP
19 Part Classes</td></tr><tr><td>mIoU</td><td>mAcc</td><td>aAcc</td><td>mIoU</td><td>mAcc</td><td>aAcc</td><td>mIoU</td><td>mAcc</td><td>aAcc</td></tr><tr><td colspan="10">Supervised Model:</td></tr><tr><td>DeiT</td><td>27.3</td><td>34.7</td><td>69.2</td><td>27.4</td><td>36.1</td><td>65.8</td><td>41.4</td><td>52.6</td><td>73.5</td></tr><tr><td colspan="10">Contrastive Learning:</td></tr><tr><td>MoCo v3</td><td>27.1</td><td>34.7</td><td>70.1</td><td>27.1</td><td>35.8</td><td>66.0</td><td>41.9</td><td>53.0</td><td>74.5</td></tr><tr><td>DINO</td><td>28.9</td><td>36.8</td><td>70.3</td><td>27.8</td><td>36.5</td><td>66.4</td><td>41.0</td><td>51.9</td><td>74.0</td></tr><tr><td colspan="10">Masked Image Modeling (MIM):</td></tr><tr><td>BEiT</td><td>18.6</td><td>25.8</td><td>58.2</td><td>14.8</td><td>21.4</td><td>47.0</td><td>27.2</td><td>36.5</td><td>60.1</td></tr><tr><td>MAE</td><td>26.3</td><td>35.0</td><td>67.3</td><td>24.3</td><td>32.9</td><td>61.5</td><td>38.2</td><td>48.7</td><td>71.3</td></tr><tr><td>CAE</td><td>28.4</td><td>36.9</td><td>71.1</td><td>27.8</td><td>37.0</td><td>66.3</td><td>43.7</td><td>55.1</td><td>75.9</td></tr><tr><td colspan="10">Contrastive Learning + MIM:</td></tr><tr><td>iBOT</td><td>32.2</td><td>40.0</td><td>73.4</td><td>30.7</td><td>40.0</td><td>69.7</td><td>44.6</td><td>55.7</td><td>76.6</td></tr></table>

other keypoints in the same image, then crop a  $d \times d$  patch centered at this keypoint and resize it to  $224 \times 224$ . We use the cosine distance as the patch distance and evaluate the retrieval performance using average precision (AP) as the retrieval metric.

The results are provided in Table 3. We have the following observations. 1) Self-supervised models except BEiT and MAE outperform the supervised model DeiT, indicating the capability that contrastive learning and CAE learn part-aware representations. BEiT and MAE perform inferior, consistent to the observations in ImageNet classification in Table 1. 2) iBOT performs the best, and the reason might be that the capability of learning part-aware representations is boosted by making use of both contrastive learning and masked image modeling.

We also report the part retrieval performance of the projected representations of contrastive learning methods in Table 3. The performance is much lower than the encoded representations. This provides an extra evidence for the part-to-whole hypothesis of contrastive learning: the projected representations are more about the whole object.

Part classification. We further conduct part classification experiments on the datasets used for part retrieval. We consider two kinds of extra learnable layers, linear probing and attentive probing, for classification. The results in Table 3 show that: 1) While DeiT performs the best in the image classification task (see Table 1), for part classification, contrastive-based methods like MoCo v3, DINO, and iBOT outperform DeiT by more than  $2\%$  under both linear and attentive probing settings. 2) Though MIM-based models CAE and MAE are inferior to DeiT in object-level classification (e.g., more than  $10\%$  and  $4\%$  lower in linear and attentive probing), they show competitive performance in linear probing and higher results than DeiT in attentive probing, demonstrating they learn better part-aware representations. 3) BEiT is inferior to other works, and iBOT has good performance, implying that the probing quality of pretrained encoders is a good indicator for downstream performance.

Part segmentation. We perform part-level linear semantic segmentation to study the finer-grained part representation modeling capability of different pretraining paradigms on three widely used datasets: ADE20K-Part (Zhou et al., 2019) containing 209 parts from the ADE20K dataset (Zhou et al., 2019), Pascal-Part (Chen et al., 2014) including 193 part categories, and LIP (Gong et al., 2017) consisting of 19 semantic human part labels. Similar to the object-level semantic segmentation experiments, linear evaluation is employed here. We maintain the same training protocols for all methods for fair comparisons. See Appendix A.2 and A.4 for dataset and training details.

The results are reported in Table 4 with the following observations. 1) Contrastive learning models, i.e., MoCo v3 and DINO, achieve competitive performance with the supervised model DeiT: DINO outperforms DeiT on ADE20K-Part and Pascal-Part, and MoCo v3 outperforms DeiT on LIP. 2) The MIM model CAE, outperforms DeiT by large margins on all three datasets, e.g.,  $1.1\%$  on ADE20K-Part and  $2.3\%$  on LIP, indicating CAE learns good part-aware representations. Similar to part retrieval, possibly due to pretraining quality in representation encoding, BEiT and MAE perform

![](images/dcfdfc9d63796b70416be20e1e3b2608da572e01d5a72dd656e5c7d85fc44f25.jpg)  
Figure 6: Comparisons between object-level and part-level semantic segmentation on ADE20K and Pascal-Part datasets. Though the supervised DeiT is superior over self-supervised models (i.e., MoCo v3, DINO, MAE, CAE) on object-level segmentation, it is generally inferior to self-supervised models on part segmentation, demonstrating self-supervised methods learn good part-aware representations. iBOT enjoys the benefits of contrastive learning and MIM. See Appendix A.3 for detailed results.

inferior. 3) Compared with object-level segmentation results in Table 2, DeiT learns better object-level semantics by explicit supervision than both contrastive learning and MIM, however, it is generally inferior to self-supervised models on part segmentation. 4) The model iBOT, which leverages both contrastive learning and MIM, outperforms all other works on three datasets, demonstrating its powerful capability in learning finer part-level semantics. Combining the two self-supervised learning techniques is thus a promising direction.

In summary, we show that self-supervised methods are potentially capable of learning part-aware representations. Among them, CAE is a representative MIM work, showing good performance by explicitly predicting the encoded representations of the masked patches in the encoding space; contrastive learning methods MoCo v3 and DINO outperform BEiT and MAE; and iBOT performs the best by combining contrastive learning and MIM. The observations are evidenced by three part-based segmentation benchmarks consistently.

# 4.3 OBSERVATION SUMMARY BETWEEN OBJECT-LEVEL AND PART-LEVEL SEGMENTATION

We conduct both object-level and part-level linear semantic segmentation on different hierarchies of the same dataset. Considering that the 209 classes in ADE20K-Part are basically chosen from 59 object classes, we denote the 59-object dataset as ADE20K-Object. Similarly, Pascal-Object consists of 16 object categories, corresponding to the 193 part categories in Pascal-Part.

The results in Figure 6 show that: although the supervised DeiT is superior over contrastive learning and masked image modeling methods on ADE20K-Object and Pascal-Object except iBOT, it is generally inferior to self-supervised models on ADE20K-Part and Pascal-Part, demonstrating self-supervised methods can learn good part-aware representations. Similar observations could be found from the object classification in Table 1 and part classification in Table 3.

In comparison to contrastive learning, CAE shows a stronger capability of learning part-aware representations, and a weaker capability of learning object-level semantics. The superiority of iBOT, a combination of contrastive learning and masked image modeling, demonstrates that it enjoys the benefits of contrastive learning and masked image modeling.

# 5 CONCLUSION

We attempt to study the capability of learning part-aware representations of self-supervised representation pretraining methods. We provide speculations for contrastive learning and masked image modeling: part-to-whole and part-to-part, with empirical results justifying the speculations. Our study presents an aspect to understand what self-supervised representation pretraining methods learn.

Future work. The strong capability of part-aware representation learning is one of the properties of self-supervised pretraining. There should be other characteristics that are leaving as the future work.

# REFERENCES

Hangbo Bao, Li Dong, and Furu Wei. BEiT: BERT pre-training of image transformers. arXiv:2106.08254, 2021.  
Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906, 2021.  
Suzanna Becker and Geoffrey E. Hinton. A self-organizing neural network that discovers surfaces in random-dot stereograms. Nature, 355(6356):161-163, 1992.  
Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In CVPR, pp. 1209-1218, 2018.  
Shuhao Cao, Peng Xu, and David A Clifton. How to understand masked autoencoders. arXiv preprint arXiv:2202.03670, 2022.  
Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. arXiv preprint arXiv:2006.09882, 2020.  
Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.  
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In ICML, volume 119 of Proceedings of Machine Learning Research, pp. 1597-1607. PMLR, 2020.  
Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, Raquel Urtasun, and Alan Yuille. Detect what you can: Detecting and representing objects using holistic models and body parts. In CVPR, pp. 1971-1978, 2014.  
Xiaokang Chen, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo, Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, and Jingdong Wang. Context autoencoder for self-supervised representation learning. arXiv preprint arXiv:2202.03026, 2022a.  
Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR, pp. 15750-15758, 2021.  
Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In ICCV, pp. 9640-9649, 2021.  
Yubei Chen, Adrien Bardes, Zengyi Li, and Yann LeCun. Intra-instance vicreg: Bag of self-supervised image patch embedding. arXiv preprint arXiv:2206.08954, 2022b.  
MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. https://github.com/open-mmlab/mmsegmentation, 2020.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pp. 248-255. IEEE, 2009.  
Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, and Nenghai Yu. Peco: Perceptual codebook for bert pre-training of vision transformers. arXiv preprint arXiv:2111.12710, 2021.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR. OpenReview.net, 2021.  
Alaaeldin El-Nouby, Gautier Izacard, Hugo Touvron, Ivan Laptev, Herve Jegou, and Edouard Grave. Are large-scale datasets necessary for self-supervised pre-training? arXiv preprint arXiv:2112.10740, 2021.

M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2010 (VOC2010) Results. http://www.pascalnetwork.org/challenges/VOC/voc2010/workshop/index.html, 2010.  
Ke Gong, Xiaodan Liang, Dongyu Zhang, Xiaohui Shen, and Liang Lin. Look into person: Self-supervised structure-sensitive learning and a new benchmark for human parsing. In CVPR, July 2017.  
Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent: A new approach to self-supervised learning. arXiv preprint arXiv:2006.07733, 2020.  
Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dólár, and Ross Girshick. Masked autoencoders are scalable vision learners. arXiv preprint arXiv:2111.06377, 2021.  
Xiangwen Kong and Xiangyu Zhang. Understanding masked image modeling via learning occlusion invariant feature. arXiv preprint arXiv:2208.04164, 2022.  
Zhaowen Li, Zhiyang Chen, Fan Yang, Wei Li, Yousong Zhu, Chaoyang Zhao, Rui Deng, Liwei Wu, Rui Zhao, Ming Tang, et al. Mst: Masked self-supervised transformer for visual representation. Advances in Neural Information Processing Systems, 34:13165-13176, 2021.  
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dólar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, pp. 740-755. Springer, 2014.  
Xingbin Liu, Jinghao Zhou, Tao Kong, Xianming Lin, and Rongrong Ji. Exploring target representations for masked autoencoders. arXiv preprint arXiv:2209.03917, 2022.  
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, pp. 10012-10022, 2021.  
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, pp. 8821-8831. PMLR, 2021.  
Nikunj Saunshi, Jordan T. Ash, Surbhi Goel, Dipendra Misra, Cyril Zhang, Sanjeev Arora, Sham M. Kakade, and Akshay Krishnamurthy. Understanding contrastive learning requires incorporating inductive biases. CoRR, abs/2202.14037, 2022. URL https://arxiv.org/abs/2202.14037.  
Chenxin Tao, Xizhou Zhu, Gao Huang, Yu Qiao, Xiaogang Wang, and Jifeng Dai. Siamese image modeling for self-supervised vision representation learning. arXiv preprint arXiv:2206.01204, 2022.  
Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. arXiv preprint arXiv:2012.12877, 2020.  
Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.  
Yixuan Wei, Han Hu, Zhenda Xie, Zheng Zhang, Yue Cao, Jianmin Bao, Dong Chen, and Baining Guo. Contrastive learning rivals masked image modeling in fine-tuning via feature distillation. arXiv preprint arXiv:2205.14141, 2022.  
Zhenda Xie, Yutong Lin, Zhuliang Yao, Zheng Zhang, Qi Dai, Yue Cao, and Han Hu. Self-supervised learning with swim transformers. arXiv preprint arXiv:2105.04553, 2021a.  
Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. arXiv preprint arXiv:2111.09886, 2021b.

Zhenda Xie, Zigang Geng, Jingcheng Hu, Zheng Zhang, Han Hu, and Yue Cao. Revealing the dark secrets of masked image modeling. CoRR, abs/2205.13543, 2022a. doi: 10.48550/arXiv.2205.13543. URL https://doi.org/10.48550/arXiv.2205.13543.  
Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Yixuan Wei, Qi Dai, and Han Hu. On data scaling in masked image modeling. CoRR, abs/2206.04664, 2022b. doi: 10.48550/arXiv.2206.04664. URL https://doi.org/10.48550/arXiv.2206.04664.  
Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised learning via redundancy reduction. arXiv preprint arXiv:2103.03230, 2021.  
Yuanyi Zhong, Haoran Tang, Junkun Chen, Jian Peng, and Yu-Xiong Wang. Is self-supervised learning more robust than supervised learning? arXiv preprint arXiv:2206.05259, 2022.  
Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. IJCV, 127(3):302-321, 2019.  
Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. Ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832, 2021.
