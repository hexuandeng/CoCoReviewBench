# DEFECT TRANSFER GAN: DIVERSE DEFECT SYNTHESIS FOR DATA AUGMENTATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Large amounts of data are a common requirement for many deep learning approaches. However, data is not always equally available at large scale for all classes. For example, on highly optimized production lines, defective samples are hardly acquired while non-defective samples come almost for free. The defects however often seem to resemble each other, e.g., scratches on different products may only differ in few characteristics. In this work, we propose to make use of the shared characteristics by transferring a stylized defect-specific content from one type of background product to another. Moreover, the stochastic variations of the shared characteristics are captured, which also allows generating novel defects from random noise. These synthetic defective samples enlarge the dataset and increase the diversity of defects on the target product. Experiments demonstrate that our model is able to disentangle the defect-specific content from the background of an image without pixel-level labels. We present convincing results on images from real industrial production lines. Also, we show consistent gains of using our method to enlarge training sets in classification tasks.

# 1 INTRODUCTION

Automated Visual Inspection (AVI) is vital for quality control in modern production lines. Despite the fact that AVI has been studied for decades, it remains a challenging task with many open research questions await to be answered. One of the main challenges in data-driven AVI is the acquisition of suitable training data. This is for two reasons: First, collecting a vast amount of labelled data is usually labor-intensive and time-consuming. In many cases, even experts are required to identify where and what to look for. However, the acquired label information is task-specific and cannot be reused or transferred to a new task in most cases. Thus, the tedious labelling process must be repeated for each new product, even if its defect is similar to other products in people's eyes. Second, in real-world scenarios such as highly optimized production lines, a more severe problem emerges: data imbalance. Only very few defective parts are produced by design. Moreover, the acquired anomaly images from a single product are lacking diversity and may not capture the full defect distribution. Training a robust deep neural network model in such conditions is very challenging.

Since collecting sufficient real-world defective samples is impractical, algorithms to synthesize required images became a focus in research. Image synthesis through Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) has shown promising performance in recent years. But it also requires large amounts of balanced data which are not available in most industrial use cases, in particular for irregular defect patterns and large variation. Therefore, GANs tend to overfit to the training examples when trained with little data (Karras et al., 2020a).

In this work, we tackle these issues by exploiting cross-domain information: we first define two sets of domains—foreground domains and background domains. The foreground domain describes a set of images that contains a specific foreground content to be grouped into a distinctive category, and each content has a different style. The background domain instead is considered as a group of images that shares similar structural appearance over the whole image. For example, we can set foreground domains as defect types and background domains as product types while the styles of defects indicate their artistic looks such as light or heavy strokes. Building upon StarGAN v2 (Choi et al., 2020), the concept underlying this work is to transfer and generate foreground contents with a variety of styles across different background domains, as illustrated in Figure 1.

![](images/15a7b24e19cabbf15faf1e5c69dd1a27f822f2dbd030de367072fc62c6245253.jpg)  
Figure 1: The underlying concept of DT-GAN is to transfer and generate foreground contents with a variety of artistic styles (e.g., light / heavy strokes) across different background domains.

The contributions of this work are three-fold: First, we introduce Defect Transfer GAN (DT-GAN), a model that learns transferring existing foreground content and generating novel contents onto different backgrounds at the same time. In the real-world scenario, it allows defect inspection networks to learn from a variety of synthetic defective images by composing the foreground defects together with various non-defective images from different products. Second, DT-GAN is able to disentangle the foreground defect-specific content and the defect-irrelevant background in a weakly-supervised manner. Third, extensive experiments show that our method can generate diverse and real-looking defective samples even for products with only 20 real defective images. These defective images generated by DT-GAN boost the performance in defect inspection networks significantly.

# 2 RELATED WORK

GANs have shown their power in many computer vision tasks such as image synthesis (Lučić et al., 2019), style translation (Johnson et al., 2016), super-resolution (Ledig et al., 2017), image inpainting (Pathak et al., 2016) and many other applications. To quantify the performance of GANs, visual quality and the diversity of generated images are considered as two of the most important criteria. Recent models address these requirements either by dedicated loss functions (Mao et al., 2019b; Yang et al., 2019) or architectural design (Brock et al., 2019). StyleGAN v2 (Karras et al., 2020b), the latest state-of-the-art model in image synthesis, introduces stochastic variation in image generating process by adding per-pixel noise after each convolution. However, it is non-trivial to adapt the model to transform given input images due to the design of the generator.

In contrast, image-to-image translation methods (Isola et al., 2017) provide a way to recover the connection between inputs and the generated images while encouraging diversity. For example, Zhu et al. (2017b) and Huang et al. (2018) impose consistent mappings in latent space to achieve the goal. Some approaches (Ma et al., 2019; Park et al., 2019) use reference images as guidance to generate diverse outputs. Mokady et al. (2020) further extends the translation task from styles to contents. It learns to identify a specific content in a given input (e.g., a specific pair of glasses) and transfer it to the target image. However, aforementioned methods only consider the translation between two domains and their extension to multiple domains is non-trivial.

Surface defect detection is one of the important tasks in real-world industrial manufacturing. It aims at identifying and classifying defects with the help of machine vision. Traditional methods (Ngan et al., 2011) build models upon hand-crafted feature extractors, which are unstable and outperformed by deep learning based models. However, the performance and generalization ability of deep learning approaches are restricted due to limited number of defective samples in real-world scenarios. Data augmentation aims to enrich the training dataset by introducing different kinds of invariance for the model to capture. Several recent works (Niu et al., 2020; Zhang et al., 2021) have proposed to adopt GANs as a data augmentation method to generate realistic defective samples. Among them, Defect-GAN (Zhang et al., 2021) tries to capture the stochastic variation within defects by mimicking the defacement and restoration processes. However, it still learns a deterministic mapping between inputs and outputs while DT-GAN achieves multi-modality by varying styles. Moreover, our method can generate realistic defects with sophisticated patterns copied from real-world defective samples.

![](images/da826b5a6a4f4bf6408e016b7511860e206366749d620c1c08185ebcddfba356.jpg)  
(a) Generator

![](images/6a9da929b12440b654b4791d603aaac50b510782fd8ecb696d476d48600cd692.jpg)  
(b) Mapping Network (c) Style-Content Encoder

![](images/5c530be980ce7897d3d988827f58b1b85272a0a9e2bb01837dd535a9723eb47c.jpg)  
Figure 2: Overview of all modules in DT-GAN.

![](images/3efbbe8d98104133f53bcf2521ab19c7d8b037279714ccc4706b2e44dbe15247.jpg)  
(d) Discriminator

# 3 METHODOLOGY

Our primary aim is to perform unpaired image-to-image translation across multiple foreground domains within a single model. In our use case, the foreground domains refer to the defect types, which means we want to achieve translations between different types of defects while the background remains unaffected. We assume that there is always an adequate amount of normal samples (e.g., non-defective) available, while anomaly samples are rare and hard to acquire.

# 3.1 PROPOSED FRAMEWORK

Our framework builds on StarGAN v2, a multimodal image-to-image translation model. Given an input image  $\mathbf{x} \in \mathcal{X}$  and an arbitrary domain  $y \in \mathcal{Y}$ , StarGAN v2 generates a domain specific style code in a learned style space and outputs an image that stylized to fit the domain of  $y$ . Its network architecture consists of four modules: a generator, a mapping network, a style encoder and a discriminator. We modify and extend all four modules (see Figure 2) to transfer not only styles but also foreground specific contents. The details are described below.

Generator (Figure 2(a)). The generator  $G$  translates an input image  $\mathbf{x}$  into an output image  $G(\mathbf{x},\widetilde{\mathbf{s}},\widetilde{\mathbf{c}})$  according to given domain specific style code  $\widetilde{\mathbf{s}}$  and content  $\widetilde{\mathbf{c}}$ , which are provided either by the mapping network  $M$  when generating from random noise or by the style-content encoder  $E$  when transferring an existing content from a reference image. As shown in Figure 2(a), we detach the original content  $\hat{\mathbf{c}}$  from the input image and take in the transferred content  $\widetilde{\mathbf{c}}$  from the target domain. The adaptive instance normalization (AdaIN) (Huang & Belongie, 2017) is then used to inject  $\widetilde{\mathbf{s}}$  into  $\widetilde{\mathbf{c}}$  during the decoding process while the background  $BG_{G}(\mathbf{x})$  is decoded separately. Finally,  $BG_{G}(\mathbf{x})$  and  $\widetilde{\mathbf{c}}$  are concatenated together and then fused before output.

Mapping network (Figure 2(b)). Given a latent code  $\mathbf{z}$  and a domain  $y$ , the mapping network  $M$  generates a style code  $\mathbf{s} = M_y(\mathbf{z})$  and a domain specific content  $\mathbf{c} = M_y(\mathbf{z})$ .  $M_y$  here denotes an output of  $M$  corresponding to the domain  $y$ .  $M$  is composed of an multi-layer perceptron with multiple output branches to provide style codes and domain specific contents for all available foreground domains. By randomly sampling  $\mathbf{z}$  from a standard normal distribution and  $y$  from all available foreground domains,  $M$  is able to produce diverse style codes and domain specific contents.

Style-content encoder (Figure 2(c)). Given an image  $\mathbf{x}$  and its domain  $y$ , the encoder  $E$  extracts the style code  $\mathbf{s} = E_y(\mathbf{x})$  and the domain specific content  $\mathbf{c} = E_y(\mathbf{x})$  from  $\mathbf{x}$ . Similar to the mapping network  $M$ , the style-content encoder  $E$  produces diverse style codes and domain specific contents that reflect the characteristics of reference images instead of randomly sampled noise.

Discriminator (Figure 2(d)). The discriminator  $D$  is a multi-task discriminator with two auxiliary classifiers: a foreground domain classifier and a background domain classifier. The input image  $\mathbf{x}$  needs to contain a domain specific content that can be recognized by the foreground domain classifier. Later, each branch  $D_y$  in the multi-task discriminator  $D$  is trained to determine if an image  $\mathbf{x}$  is a real image of its foreground domain or a fake image  $G(\mathbf{x},\mathbf{s},\mathbf{c})$  generated by  $G$ . Apart from that, one extra branch  $BG_{\mathrm{cls}}$  is attached to decide whether the background information of the input images is well preserved.

# 3.2 CONTENT TRANSFER

Inspired by Mokady et al. (2020), our model learns to discriminate the domain specific content in a weakly-supervised manner. That is, given unpaired samples from two randomly selected foreground domains  $A$  and  $B$ , we aim to identify the domain specific parts  $\mathbf{c}_a$  in image  $\mathbf{a} \in A$  and  $\mathbf{c}_b$  in image  $\mathbf{b} \in B$ , then perform content swapping. To achieve this, we manipulate the three-dimensional feature map (i.e.,  $H \times W \times C$ ) at the bottle neck of  $G$  by encouraging the model to encode the domain specific content into the latter channels and replacing it with new content from the target domain. Note that we treat domain Normal as a special case ('anchor domain') because a normal image does not have domain specific content in our definition. Thus, all the domain specific content of normal images produced by subnetworks will be replaced by zero. Together with the foreground content classifier in  $D$ , we observe that this design allows  $G$  to disentangle the domain specific content from an image without pixel-level label information.

Compared to StarGAN v2, our method now not only models style codes and contents separately but also disentangles the foreground and background of an image in a weakly-supervised manner. These features allow explicit control over output images by combining desired style codes and contents from one of the subnetworks with the input images. Therefore, it leads to higher variance regarding the location, structural pattern and artistic style of defects in the synthetic images of DT-GAN.

# 3.3 TRAINING OBJECTIVES

Given an image  $\mathbf{x} \in \mathcal{X}$ , its original foreground domain  $y \in \mathcal{Y}$  and its background domain  $p \in \mathcal{P}$ , the following objectives are used to train our framework.

Adversarial loss. In the training phase, a noise vector  $\mathbf{z} \in \mathcal{Z}$  and a target foreground domain  $\widetilde{y} \in \mathcal{Y}$  are sampled randomly. Both of them are fed to  $M$ , producing a target style code  $\widetilde{\mathbf{s}}$  and a target content  $\widetilde{\mathbf{c}}$  as follows:  $\widetilde{\mathbf{s}}, \widetilde{\mathbf{c}} = M_{\widetilde{y}}(\mathbf{z})$ . Goal of the training is to ensure that  $\widetilde{\mathbf{s}}$  and  $\widetilde{\mathbf{c}}$  are sampled from the distribution over styles and contents of the target domain  $\widetilde{y}$ . The generator  $G$  then combines an image  $\mathbf{x}$  with  $\widetilde{\mathbf{s}}$  and  $\widetilde{\mathbf{c}}$  and learns to generate an output image  $G(\mathbf{x}, \widetilde{\mathbf{s}}, \widetilde{\mathbf{c}})$  that is indistinguishable from real images in the target domain  $\widetilde{y}$ . We encourage this behavior by using an adversarial loss same as in Choi et al. (2020)

$$
\mathcal {L} _ {\mathrm {a d v}} = \mathbb {E} _ {\mathbf {x}, y} [ \log D _ {y} (\mathbf {x}) ] + \mathbb {E} _ {\mathbf {x}, \widetilde {y}, \mathbf {z}} [ \log (1 - D _ {\widetilde {y}} (G (\mathbf {x}, \widetilde {\mathbf {s}}, \widetilde {\mathbf {c}}))) ], \tag {1}
$$

where  $D_y$  and  $D_{\widetilde{y}}$  are the output branches of  $D$  that correspond to the source domain  $y$  and the target domain  $\widetilde{y}$ , respectively.

Style-content reconstruction loss. Similar to StarGAN v2, to enforce the generator  $G$  takes the style code  $\widetilde{\mathbf{s}}$  and the domain specific content  $\widetilde{\mathbf{c}}$  into consideration during the generation process, we employ a style-content reconstruction loss

$$
\mathcal {L} _ {\mathrm {s t y} \cdot \text {c o n}} = \mathbb {E} _ {\mathbf {x}, \widetilde {y}, \mathbf {z}} \left[ \| \widetilde {\mathbf {s}} - S _ {E} (G (\mathbf {x}, \widetilde {\mathbf {s}}, \widetilde {\mathbf {c}})) \| _ {1} \right] + \mathbb {E} _ {\mathbf {x}, \widetilde {y}, \mathbf {z}} \left[ \| \widetilde {\mathbf {c}} - C _ {E} (G (\mathbf {x}, \widetilde {\mathbf {s}}, \widetilde {\mathbf {c}})) \| _ {1} \right]. \tag {2}
$$

This objective urges the style-content encoder  $E$  to recover  $\widetilde{\mathbf{s}}$  and  $\widetilde{\mathbf{c}}$  from  $G(\mathbf{x},\widetilde{\mathbf{s}},\widetilde{\mathbf{c}})$ . Here, the style-content encoder  $E$  learns a mapping from an image to its style and content domains, which allows  $G$  to synthesize an image with given  $\mathbf{s}$  and  $\mathbf{c}$  from reference images at test time.

Diversity loss. In order to further boost the diversity of output images from  $G$ , we introduce a loss that encourages diversity as follows: for a pair of random latent codes  $\mathbf{z}_1$  and  $\mathbf{z}_2$  we compute  $\widetilde{\mathbf{s}}_i, \widetilde{\mathbf{c}}_i = M_{\widetilde{y}}(\mathbf{z}_i)$  for  $i \in \{1, 2\}$  and enforce a different outcome of the generator  $G$  for differently mixed style and content input pairs:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm {d s}} = \mathbb {E} _ {\mathbf {x}, \widetilde {y}, \mathbf {z} _ {1}, \mathbf {z} _ {2}} \left[ \left\| G (\mathbf {x}, \widetilde {\mathbf {s}} _ {1}, \widetilde {\mathbf {c}} _ {2}) - G (\mathbf {x}, \widetilde {\mathbf {s}} _ {2}, \widetilde {\mathbf {c}} _ {1}) \right\| _ {1} \right] \\ + \mathbb {E} _ {\mathbf {x}, \widetilde {y}, \mathbf {z} _ {1}, \mathbf {z} _ {2}} \left[ \| G (\mathbf {x}, \widetilde {\mathbf {s}} _ {1}, \widetilde {\mathbf {c}} _ {1}) - G (\mathbf {x}, \widetilde {\mathbf {s}} _ {2}, \widetilde {\mathbf {c}} _ {2}) \| _ {1} \right] \tag {3} \\ \left. \right. + \Sigma_ {m, n, o} \left[ \mathbb {E} _ {\mathbf {x}, \widetilde {y}, \mathbf {z} _ {1}, \mathbf {z} _ {2}} \left[\left\| G (\mathbf {x}, \widetilde {\mathbf {s}} _ {m}, \widetilde {\mathbf {c}} _ {n}) - G (\mathbf {x}, \widetilde {\mathbf {s}} _ {o}, \widetilde {\mathbf {c}} _ {o}) \right\| _ {1} \right]\right], \\ \end{array}
$$

where  $m, n \in \{1, 2 | m \neq n\}$  and  $o \in \{1, 2\}$ . Driven by this term, the generator  $G$  is forced to discover meaningful style features and contents that eventually lead to diversity in generated images. We ignore the denominator  $\|\mathbf{z}_1 - \mathbf{z}_2\|_1$  of the original diversity loss (Mao et al., 2019a) for stable training as in StarGAN v2.

Cycle consistency loss. To ensure that the generated image  $G(\mathbf{x}, \widetilde{\mathbf{s}}, \widetilde{\mathbf{c}})$  preserves the domain-invariant properties of its input image  $\mathbf{x}$ , we impose the cycle consistency loss (Zhu et al., 2017a)

$$
\mathcal {L} _ {\mathrm {c y c}} = \mathbb {E} _ {\mathbf {x}, y, \widetilde {y}, \mathbf {z}} \left[ \left| \left| \mathbf {x} - G \left(G (\mathbf {x}, \widetilde {\mathbf {s}}, \widetilde {\mathbf {c}}), \hat {\mathbf {s}}, \hat {\mathbf {c}}\right) \right| \right| _ {1} \right], \tag {4}
$$

where  $\hat{\mathbf{s}},\hat{\mathbf{c}} = E_y(\mathbf{x})$  is the extracted style code and domain specific content of the input image  $\mathbf{x}$  and  $y$  is the original domain of  $\mathbf{x}$ . By learning to reconstruct the input image  $\mathbf{x}$  with given style code  $\hat{\mathbf{s}}$  and content  $\hat{\mathbf{c}}$ , the generator  $G$  is then further encouraged to disentangle the background, the domain specific content and the style code.

Content consistency loss. Besides the cycle consistency loss, we apply another constraint to enforce that the detached domain specific content from  $G$  is consistent with the one retrieved from  $E$  according to

$$
\mathcal {L} _ {\text {c o n - c y c}} = \mathbb {E} _ {\mathbf {x}, y, \widetilde {y}, \mathbf {z}} \left[ \| F G _ {G} (\mathbf {x}) - \hat {\mathbf {c}} \| _ {1} \right] + \mathbb {E} _ {\mathbf {x}, y, \widetilde {y}, \mathbf {z}} \left[ \| F G _ {G} (G (\mathbf {x}, \widetilde {\mathbf {s}}, \widetilde {\mathbf {c}})) - \widetilde {\mathbf {c}} \| _ {1} \right], \tag {5}
$$

where  $\hat{\mathbf{c}} = E_y(\mathbf{x})$ ,  $\widetilde{\mathbf{c}} = E_{\widetilde{y}}(\mathbf{x})$ ,  $FG_{G}(\mathbf{x})$  and  $FG_{G}(G(\mathbf{x},\widetilde{\mathbf{s}},\widetilde{\mathbf{c}}))$  are the pop-out domain specific content from input image  $\mathbf{x}$  and generated image  $G(\mathbf{x},\widetilde{\mathbf{s}},\widetilde{\mathbf{c}})$ , respectively.

Classification losses. We employ two classification losses to strengthen the separation of the background and the foreground domain specific content of an image. The first one is the foreground content classification loss

$$
\mathcal {L} _ {\mathrm {F G . c l s}} = \mathbb {E} _ {\mathbf {x} _ {\text {r e a l}}, y} \left[ - \log D _ {\mathrm {F G . c l s}} (y | \mathbf {x} _ {\text {r e a l}}) \right] + \mathbb {E} _ {\mathbf {x} _ {\text {f a k e}}, \widetilde {y}} \left[ - \log D _ {\mathrm {F G . c l s}} (\widetilde {y} | \mathbf {x} _ {\text {f a k e}}) \right], \tag {6}
$$

which aims to ensure that the domain specific content is properly encoded and carries enough information from the target domain. The second one is the background classification loss

$$
\mathcal {L} _ {\mathrm {B G . c l s}} = \mathbb {E} _ {\mathbf {x} _ {\text {r e a l}}, p} \left[ - \log D _ {\mathrm {B G . c l s}} (p | \mathbf {x} _ {\text {r e a l}}) \right] + \mathbb {E} _ {\mathbf {x} _ {\text {f a k e}}, p} \left[ - \log D _ {\mathrm {B G . c l s}} (p | \mathbf {x} _ {\text {f a k e}}) \right], \tag {7}
$$

where  $p$  is the corresponding background type of  $\mathbf{x}_{\mathrm{real}}$  and  $\mathbf{x}_{\mathrm{fake}}$ . With the help of this objective, the generator  $G$  learns to preserve the domain-invariant characteristics of its input image  $\mathbf{x}$  while dissociating the foreground domain specific part.

Full objective. Our full objective functions can be summarized as

$$
\min  _ {G, F, E} \max  _ {D} \quad \mathcal {L} _ {\mathrm {a d v}} + \lambda_ {\mathrm {s t y} - \mathrm {c o n}} \mathcal {L} _ {\mathrm {s t y} - \mathrm {c o n}} - \lambda_ {\mathrm {d s}} \mathcal {L} _ {\mathrm {d s}} + \lambda_ {\mathrm {c y c}} \mathcal {L} _ {\mathrm {c y c}} + \tag {8}
$$

$$
\lambda_ {\text {c o n} \cdot \text {c y c}} \mathcal {L} _ {\text {c o n} \cdot \text {c y c}} + \lambda_ {\text {F G} \cdot \text {c l s}} \mathcal {L} _ {\text {F G} \cdot \text {c l s}} + \lambda_ {\text {B G} \cdot \text {c l s}} \mathcal {L} _ {\text {B G} \cdot \text {c l s}},
$$

where  $\lambda_{\mathrm{sty}}$ ,  $\lambda_{\mathrm{ds}}$ ,  $\lambda_{\mathrm{cyc}}$ ,  $\lambda_{\mathrm{con\_cyc}}$ ,  $\lambda_{\mathrm{FG\_cls}}$  and  $\lambda_{\mathrm{BG\_cls}}$  are the hyperparameters for each term.

# 4 EXPERIMENTS

We evaluated the images generated by DT-GAN through a series of experiments both quantitatively and qualitatively. Finally, we demonstrate the benefits of our generated images when being used as data augmentation for a defect classification task on limited data.

Dataset. All experiments were performed on a real industrial dataset: a Surface Defect Inspection (SDI) dataset that contains three different kinds of products from production lines and samples from each product are classified into three mutually exclusive classes: Normal, Scratch and Spot. All of the images are grayscale. Detailed statistics of the dataset are summarized in Appendix A. Note that only the training set was used in GAN training, the test set was left untouched for final evaluation in classifier training. For a fair comparison, all images were resized to  $128 \times 128$  resolution for both GAN training and classifier training, which was also the highest resolution used in the baselines for image generation.

# 4.1 DEFECT GENERATION

Baselines. As discussed in Section 3, DT-GAN can either use the mapping network to randomly generate styles and defects, or it can use the style-content encoder to extract both from reference images. We refer to these cases as 'latent-guided' and 'reference-guided', respectively.

Since the two ways of guidance are fundamentally different, we evaluated them against two sets of baselines: Our reference-guided image generation was compared to Mokady et al. (2020) and StarGAN v2, because both of them can perform a reference-guided translation. Note that Mokady et al. (2020) can only translate between two domains while StarGAN v2 and DT-GAN can achieve multi-domain translation within a single model. Images generated through the latent-guided part of DT-GAN were compared to state-of-the-art GANs in image synthesis: BigGAN (Brock et al., 2019) and StyleGAN v2 (Karras et al., 2020b). We set BigGAN to condition on defect types during training while StyleGAN v2 was trained unconditionally. All baselines were trained with the public implementations provided by the authors<sup>1</sup>.

# 4.1.1 QUANTITATIVE EVALUATION

Metrics. We employed the commonly used frechet inception distance (FID) (Heusel et al., 2017) to evaluate both the visual quality and the diversity of the generated images. We also report the kernel inception distance (KID) (Binkowski et al., 2018) which is a more stable metric for small sets of images like our SDI dataset. Lower FID and KID scores indicate better performance.

Both scores are shown in Table 1. We observe that methods like BigGAN and StyleGAN v2, which perform defect synthesis purely based on latent codes, generally provide unsatisfactory results on the SDI dataset, presumably due to the small number of defective samples that were available. These methods then struggle to capture the complex and irregular patterns of defects. We also experimented with augmentation methods for GAN training (Karras et al., 2020a; Zhao et al., 2020) but did not find a consistent improvement (see Appendix E). We thus only report the best scores.

Reference-guided synthesis methods like Mokady et al. (2020) and StarGAN v2 seem to generate more realistic images. The scores of StarGAN v2 on a single product are omitted here because generating images with specified background is not possible due to its network design—the product type changes in output images, which we refer to as 'identity-shift'. As seen in Table 1, our method achieves better scores in all cases. We believe this is due to the fact that our method allows free combination of foreground defects and backgrounds, making the generated images more diverse even with a small number of training samples.

Table 1: Quantitative comparison of DT-GAN with baseline image synthesis methods using FID and KID. Note that the reported values are not comparable between columns, because they were calculated on different training sets.  

<table><tr><td rowspan="2">Method</td><td colspan="4">FID↓</td><td colspan="4">KID↓</td></tr><tr><td>A</td><td>B</td><td>C</td><td>All</td><td>A</td><td>B</td><td>C</td><td>All</td></tr><tr><td>Mokady (2020)</td><td>68.69</td><td>66.90</td><td>36.21</td><td>58.63</td><td>0.050</td><td>0.036</td><td>0.030</td><td>0.036</td></tr><tr><td>StarGAN v2</td><td>-</td><td>-</td><td>-</td><td>37.70</td><td>-</td><td>-</td><td>-</td><td>0.013</td></tr><tr><td>StyleGAN v2</td><td>90.10</td><td>52.95</td><td>138.09</td><td>35.34</td><td>0.072</td><td>0.027</td><td>0.186</td><td>0.013</td></tr><tr><td>BigGAN + DiffAug</td><td>218.74</td><td>134.41</td><td>270.89</td><td>155.88</td><td>0.220</td><td>0.121</td><td>0.378</td><td>0.099</td></tr><tr><td>Ours</td><td>58.43</td><td>36.44</td><td>22.68</td><td>29.73</td><td>0.025</td><td>0.013</td><td>0.012</td><td>0.009</td></tr></table>

# 4.1.2 QUALITATIVE EVALUATION

We present a qualitative comparison with the baseline methods in latent-guided image synthesis in Figure 3. To make a fair comparison, we trained StyleGAN v2 and BigGAN on each product separately to have control on background products. Note however, that images from DT-GAN were always obtained from a single model. We can see that some generated samples from StyleGAN v2 do not contain clear defects, and some samples from BigGAN do not appear natural. Both methods do not take images as inputs, they generate synthetic images according to given latent codes. Therefore it is not possible to explicitly model foreground contents. On the other hand, StarGAN v2 performs translation based on input images but then fails to preserve the background, which results in artifacts or identity-shift in its outputs. Our network architecture that disentangles foreground and background seems to mitigate these issues.

![](images/7f5f89a443f668c5e2525f1fa777d00b4d89c5168da5253837564861358ce882.jpg)  
(a) Normal-to-Scratches

![](images/fd3dd7cac271af5808a2cc39cdd60403d388fe642f41e9002f27e6b70d2b6c09.jpg)

![](images/5704514e96c7fdd1682ee61ba8e4c299bafdc7093dd62b28a2339f3ac1c42260.jpg)  
(b) Normal-to-Spots

![](images/4fa046de9759344acfdfa09094659ddd8c582fbe555f77c943c242b91cd7007a.jpg)

![](images/d058a705aecf6ea909aeddd3b40a1ca5533f13277d559b20772d84ba4dd8589f.jpg)

![](images/387af98ff029c9cb9f3e975a5451e9c2278c37afd5009ee991973532f2a5eee4.jpg)  
Figure 3: Qualitative comparison of latent-guided image synthesis results. In each subfigure: on the left, defective images are fully generated from random noise. On the right, random defects are synthesized onto given normal samples. Note that BigGAN* denotes it was trained with DiffAug.

![](images/ddc890996eb016ee9ee6ac5b4e36de9e425eacc6f0e9dd618fb450fe0642888f.jpg)  
(a) Normal-to-Scratches

![](images/7b177bc07c43e0c2c40290314d71718ea1d8bdd20b7f3d9b42b10ffc1d9e853f.jpg)  
(b) Normal-to-Spots

![](images/a17197cd3ebc4e4aab65d91f433c159d433283c757290b32a2efd547a0010aaf.jpg)  
Figure 4: Qualitative comparison of reference-guided image synthesis results on the SDI dataset. Each method transforms the given images into target foreground domains with the styles and contents extracted from the reference images.

Also for reference-guided image synthesis, where we used different background and foreground reference images as illustrated in Figure 4, only our method produces high quality images with preserved background from the source and transferred foreground defect from the reference.

Ablation study. We visually demonstrate the effect of each component we added to DT-GAN compared to StarGAN v2 in Figure 5, using the examples of both latent- and reference-guided image synthesis from Normal to Scratches.

Column (a) corresponds to StarGAN v2 and highlights the identity-shift in the background again. We first tackle this problem by modeling the style code and foreground content explicitly and feeding them separately to the generator. This leads to a better preservation of the background structure in column (b) for the reference-guided subnetwork, but not for the latent-guided synthesis on the bottom of Figure 5. Thus, we add a foreground classifier in the discriminator in (c) to ensure the output image contains the desired foreground content (scratch). Similarly, we introduce a background classifier to the discriminator in column (d). Note that the additional product type labels can be acquired automatically from production lines.

For column (e), we add the separate decoders for foreground and background in the generator which are fused only in the end. This enhances the preservation of background characteristics like lighting even more. Imposing an additional penalty for foreground content extracted from a normal sample as described in Section 3.2 leads to another visual improvement of the foreground edges for reference-guided synthesis in column (f). Finally, inspired by StyleGAN, we incorporate adaptive noise injection to the mapping network, which significantly boosts the performance of our latent-guided image synthesis as shown in column (g).

Styling. We visually demonstrate the effect of style codes in our method by randomly sampling those and combining them with fixed reference background and foreground images in Figure 6, where a variety of artistic styles can be seen on the output columns.

![](images/70422c5fcbcac9f78b5d1fd5293841e72bd7ea374416c1e4d79f17b2b0390712.jpg)  
Figure 5: Ablation study. (a) The baseline StarGAN v2. (b) + Style-Content branches. (c) + Foreground classifier. (d) + Background classifier. (e) + Separately decoding foreground and background in  $G$ . (f) + Anchor foreground domain (e.g. Normal). (g) + Noise injection in Mapping Network.

![](images/26cc2ecc9364ec75934521d3990517598d92e3dd3efb1e3e3f3861e91999b683.jpg)  
(a) Normal-to-Scratches

![](images/0f57cfad8ac8fff192191039dd2ca9bb16c37f616a86f8dbe99c6e91dbb90697.jpg)  
Figure 6: Visual effect of randomly sampled style codes on fixed pairs of reference background and foreground images.  
(b) Normal-to-Spots

# 4.2 DT-GAN FOR DATA AUGMENTATION

We also evaluated our method as a data augmentation method for defect classification on the SDI dataset. We defined one task 'general', where the classifier was trained on images from all products at once, while task 'single product' only used the subset of images for one product.

Besides, we incrementally varied the amount of real Normal data available for classifier training: 4500, 6600, 12000 and 18600. In the case of defective images, all of them were always used due to the small amount unless otherwise specified. As backbone we used a ResNet-50 (He et al., 2016a) with ImageNet pretrained weights. For experiments with synthetic data, we attached an auxiliary domain classifier to the network through a Gradient Reversal Layer (Ganin & Lempitsky, 2015).

Table 2: Quantitative comparison of the baseline methods on defect classification task at the scale of 12000 images/class. The reported values are the achieved error rates (\%) over five runs.  

<table><tr><td>Method</td><td>ResNet-50</td><td>EfficientNet-b4</td></tr><tr><td>No-Aug</td><td>21.64±1.24</td><td>12.06±0.64</td></tr><tr><td>Trad-Aug</td><td>12.58±0.81</td><td>9.33±0.73</td></tr><tr><td>Mokady (2020)</td><td>11.11±1.19</td><td>13.26±1.13</td></tr><tr><td>StarGAN v2</td><td>13.07±1.30</td><td>12.25±0.79</td></tr><tr><td>StyleGAN v2</td><td>11.55±1.79</td><td>11.68±0.76</td></tr><tr><td>BigGAN+DiffAug</td><td>11.45±0.61</td><td>12.06±0.50</td></tr><tr><td>Ours</td><td>9.9±0.69</td><td>9.14±1.02</td></tr></table>

Since the SDI dataset is highly imbalanced, we oversampled the minority classes (Ling et al., 1998) unless the data was balanced through synthetic images. Additionally, we always applied traditional data augmentation techniques like random horizontal flips, jittering and lighting (Shorten & Khoshgoftaar, 2019) except where noted. All following results were evaluated by the achieved error rates over five runs with different random seeds.

Effectiveness of synthetic data. We first compare classifier performance for no augmentation (No-Aug), traditional data augmentation (Trad-Aug), and a combination of traditional augmentation with synthetic images for GAN methods including DT-GAN. We also introduce a stronger backbone, EfficientNet-b4 (Tan & Le, 2019), to demonstrate that our results are not confined to a specific network. Table 2 shows that our method is the only one that improves performance for both backbones, presumably due to the combination of high visual image quality and diversity in our samples.

Table 3: Experimental results on using different amount of synthetic images generated by DT-GAN to train classifiers. The left-most column stands for number of samples per class to be classified. The training set of the baselines is balanced by oversampling while ours is by synthetic images.  

<table><tr><td rowspan="2">Dataset Size</td><td colspan="2">20A</td><td colspan="2">All</td></tr><tr><td>Trad-Aug</td><td>Ours</td><td>Trad-Aug</td><td>Ours</td></tr><tr><td>4500</td><td>15.55±0.63</td><td>14.28±1.25</td><td>12.75±0.61</td><td>11.04±0.76</td></tr><tr><td>6600</td><td>16.69±0.76</td><td>14.41±3.12</td><td>13.07±1.57</td><td>10.60±0.48</td></tr><tr><td>12000</td><td>16.95±1.02</td><td>14.22±1.53</td><td>12.05±0.81</td><td>9.90±0.69</td></tr><tr><td>18600</td><td>16.12±2.19</td><td>15.36±0.86</td><td>12.37±0.32</td><td>10.21±0.96</td></tr></table>

Impact of dataset size. Motivated by the limited availability of data in real-world production scenarios, we therefore evaluated DT-GAN for data augmentation on a subset of the full SDI dataset (All), which only contains 20 defective samples in product A for each defect type (20A). In this case, DT-GAN was also trained on the reduced subset. As shown in Table 3, there is a clear improvement when synthetic images from DT-GAN are used as data augmentation, even for the extremely limited data subset. Further results on single product classifiers can be found in Appendix E.

Table 4: Cross-domain effect on single product classifiers trained with reference-guided synthetic images at the scale of 12000 images/class. Note that A, B and C stand for the three products in the SDI dataset while vA, vB, vC and vABC indicate the defects are copied from which reference set.  

<table><tr><td></td><td>Trad-Aug</td><td>vA</td><td>vB</td><td>vC</td><td>vABC</td></tr><tr><td>A</td><td>13.81±2.36</td><td>11.81±2.65</td><td>12.72±2.87</td><td>11.99±1.63</td><td>11.09±3.49</td></tr><tr><td>B</td><td>6.80±1.64</td><td>6.40±1.34</td><td>6.60±1.52</td><td>6.59±1.34</td><td>5.60±1.34</td></tr><tr><td>C</td><td>16.57±3.20</td><td>13.14±2.81</td><td>11.23±0.80</td><td>14.85±1.73</td><td>11.42±0.96</td></tr></table>

Cross-domain effect. We hypothesized that limited data can be counteracted by transferring defects across multiple background products. We tested this approach by comparing the performance of classifiers trained on synthetic images with defects from a specific source (vA, vB, vC) to classifiers trained on images with defects from all products (vABC). As we can see in Table 4, the best performances are reached by the models that take over defects from other products. We interpret this as support for our hypothesis and its practical usefulness. Further results are in Appendix E.

# 5 CONCLUSION

We propose a novel method, DT-GAN, which allows diverse defect synthesis both by generating from randomly sampled noise and by following the guidance of given reference images. It extends the StarGAN v2 architecture by six elements which leads to higher image fidelity, better variance in defects, disentanglement and full control over background and foreground. We demonstrated the feasibility and benefits of DT-GAN on a real industrial defect classification task. The extensive experimental results show that our method provides consistent gains even with limited data and can boost the performance of classifiers compared to state-of-the-art image synthesis methods. For future investigation, we aim to represent defects more explicitly to enhance the model transferability to unseen products and also improve the explainability of the model.

# REPRODUCIBILITY STATEMENT

We aim for full reproducibility by publishing the source code and dataset with the final version of the paper. Besides, we provide descriptions of the training details in Appendix B, the evaluation setup in Appendix C and the network architecture in Appendix D.

# REFERENCES

Mikolaj Binkowski, Danica J. Sutherland, Michael Arbel, and A. Gretton. Demystifying MMD GANs. *ArXiv*, abs/1801.01401, 2018.  
Andrew Brock, Jeff Donahue, and K. Simonyan. Large scale gan training for high fidelity natural image synthesis. ArXiv, abs/1809.11096, 2019.  
Yuhua Chen, Wen Li, Christos Sakaridis, Dengxin Dai, and Luc Van Gool. Domain adaptive faster r-cnn for object detection in the wild. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3339-3348, 2018.  
Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248-255. IEEE, 2009.  
Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In Francis Bach and David Blei (eds.), Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pp. 1180-1189, Lille, France, 07-09 Jul 2015. PMLR. URL https://proceedings.mlr.press/v37/ganin15.html.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. In NeurIPS, 2014.  
Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016a.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks, 2016b. URL http://arxiv.org/abs/1603.05027. cite arxiv:1603.05027Comment: ECCV 2016 camera-ready.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NIPS, 2017.  
Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017.  
Xun Huang, Ming-Yu Liu, Serge J. Belongie, and J. Kautz. Multimodal unsupervised image-to-image translation. ArXiv, abs/1804.04732, 2018.  
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5967-5976, 2017.  
Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution, 2016.  
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data, 2020a.

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020b.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical Report 0, University of Toronto, Toronto, Ontario, 2009.  
Christian Ledig, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photo-realistic single image super-resolution using a generative adversarial network. In CVPR, 2017.  
Charles Ling, , Charles X. Ling, and Chenghui Li. Data mining for direct marketing: Problems and solutions. In In Proceedings of the Fourth International Conference on Knowledge Discovery and Data Mining (KDD-98, pp. 73-79. AAAI Press, 1998.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.  
Mario Lucic, Michael Tschannen, Marvin Ritter, Xiaohua Zhai, Olivier Bachem, and Sylvain Gelly. High-fidelity image generation with fewer labels. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 4183-4192. PMLR, 09-15 Jun 2019. URL https://proceedings.mlr.press/v97/lucic19a.html.  
Liqian Ma, Xu Jia, Stamatios Georgoulis, Tinne Tuytelaars, and Luc Van Gool. Exemplar guided unsupervised image-to-image translation with semantic consistency. In ICLR, 2019.  
Qi Mao, Hsin-Ying Lee, Hung-Yu Tseng, Siwei Ma, and Ming-Hsuan Yang. Mode seeking generative adversarial networks for diverse image synthesis. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1429-1437, 2019a.  
Qi Mao, Hsin-Ying Lee, Hung-Yu Tseng, Siwei Ma, and Ming-Hsuan Yang. Mode seeking generative adversarial networks for diverse image synthesis. In CVPR, 2019b.  
Ron Mokady, Sagie Benaim, Lior Wolf, and Amit Bermano. Masked based unsupervised content transfer. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=BJe-91BtvH.  
Henry Y. T. Ngan, Grantham K. H. Pang, and Nelson H. C. Yung. Review article: Automated fabric defect detection-a review. Image Vision Comput., 29(7):442-458, June 2011. ISSN 0262-8856. doi: 10.1016/j.imavis.2011.02.002. URL https://doi.org/10.1016/j.imavis.2011.02.002.  
Shuanlong Niu, Bin Li, Xinggang Wang, and Hui Lin. Defect image sample generation with gan for improving defect recognition. IEEE Transactions on Automation Science and Engineering, 17(3):1611-1622, 2020. doi: 10.1109/TASE.2020.2967415.  
Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. Semantic image synthesis with spatially-adaptive normalization. In CVPR, 2019.  
Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zach DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.  
Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. Context encoders: Feature learning by inpainting. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2536-2544, 2016.  
Sebastian Ruder. An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747, 2016.  
Connor Shorten and T. Khoshgoftaar. A survey on image data augmentation for deep learning. Journal of Big Data, 6:1-48, 2019.

Mingxing Tan and Quoc Le. EfficientNet: Rethinking model scaling for convolutional neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 6105-6114. PMLR, 09-15 Jun 2019. URL https://proceedings.mlrpress/v97/tan19a.html.  
Dingdong Yang, Seunghoon Hong, Yunseok Jang, Tiangchen Zhao, and Honglak Lee. Diversity-sensitive conditional generative adversarial networks. In ICLR, 2019.  
Gongjie Zhang, Kaiwen Cui, Tzu-Yi Hung, and Shijian Lu. Defect-gan: High-fidelity defect synthesis for automated defect inspection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 2524-2534, January 2021.  
Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han. Differentiable augmentation for data-efficient gan training, 2020.  
Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017a.  
Jun-Yan Zhu, Richard Zhang, Deepak Pathak, Trevor Darrell, Alexei A. Efros, O. Wang, and E. Shechtman. Toward multimodal image-to-image translation. In NIPS, 2017b.
