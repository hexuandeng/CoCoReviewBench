# CoAtNet: Marrying Convolution and Attention for All Data Sizes

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Transformers have attracted increasing interests in computer vision, but they still fall behind state-of-the-art convolutional networks. In this work, we show that while Transformers tend to have larger model capacity, their generalization can be worse than convolutional networks due to the lack of the right inductive bias. To effectively combine the strengths from both architectures, we present CoAtNets (pronounced "coat nets"), a family of hybrid models built from two key insights: (1) depthwise Convolution and self-Attention can be naturally unified via simple relative attention; (2) vertically stacking convolution layers and attention layers in a principled way is surprisingly effective in improving generalization, capacity and efficiency. Experiments show that our CoAtNets achieve state-of-the-art performance under different resource constraints across various datasets. For example, CoAtNet achieves  $86.0\%$  ImageNet top-1 accuracy without extra data, and  $89.77\%$  with extra JFT data, outperforming prior arts of both convolutional networks and Transformers. Notably, when pretrained with 13M images from ImageNet-21K, our CoAtNet achieves  $88.56\%$  top-1 accuracy, matching ViT-huge pertained with 300M images from JFT while using 23x less data.

# 1 Introduction

Since the breakthrough of AlexNet [1], Convolutional Neural Networks (ConvNets) have been the dominating model architecture for computer vision [2, 3, 4, 5]. Meanwhile, with the success of self-attention models like Transformers [6] in natural language processing [7, 8], many previous works have attempted to bring in the power of attention into computer vision [9, 10, 11, 12]. More recently, Vision Transformer (ViT) [13] has shown that with almost only vanilla Transformer layers, one could obtain reasonable performance on ImageNet-1K [14] alone. More importantly, when pre-trained on large-scale weakly labeled JFT-300M dataset [15], ViT achieves comparable results to state-of-the-art (SOTA) ConvNets, indicating that Transformer models potentially have higher capacity at scale than ConvNets.

While ViT has shown impressive results with enormous JFT 300M training images, its performance still falls behind ConvNets in the low data regime. For example, without extra JFT-300M pretraining, the ImageNet accuracy of ViT is still significantly lower than ConvNets with comparable model size [5] (see Table 11). Subsequent works use special regularization and stronger data augmentation to improve the vanilla ViT [16, 17, 18], yet none of these ViT variants could outperform the SOTA convolution-only models on ImageNet classification given the same amount of data and computation [19, 20]. This suggests that vanilla transformer layers may lack certain desirable inductive biases possessed by ConvNets, and thus require significant amount of data and computational resource to compensate. Not surprisingly, many recent works have been trying to incorporate the

inductive biases of ConvNets into transformer models, by imposing local receptive fields for attention layers [21, 22] or augmenting the attention and FFN layers with implicit or explicit convolutional operations [23, 24, 25]. However, these approaches are either ad-hoc or focused on injecting a particular property, lacking a systematic understanding of the respective roles of convolution and attention when combined.

In this work, we systematically study the problem of hybridizing convolution and attention from two fundamental aspects in machine learning – generalization and model capacity. Our study shows that convolutional layers tend to have better generalization with faster converging speed thanks to their strong prior of inductive bias, while attention layers have higher model capacity that can benefit from larger datasets. Combining convolutional and attention layers can achieve better generalization and capacity; however, a key challenge here is how to effectively combine them to achieve better trade-offs between accuracy and efficiency. In this paper, we investigate two key insights: First, we observe that the commonly used depthwise convolution can be effectively merged into attention layers with simple relative attention; Second, simply stacking convolutional and attention layers, in a proper way, could be surprisingly effective to achieve better generalization and capacity. Based on these insights, we propose a simple yet effective network architecture named CoAtNet, which enjoys the strengths from both ConvNets and Transformers.

Our CoAtNet achieves SOTA performances under comparable resource constraints across different data sizes. Specifically, under the low-data regime, CoAtNet inherits the great generalization property of ConvNets thanks to the favorable inductive biases. Moreover, given abundant data, CoAtNet not only enjoys the superior scalability of Transformer models, but also achieves faster convergence and thus improved efficiency. When only ImageNet-1K is used for training, CoAtNet achieves  $86.0\%$  top-1 accuracy, matching the best public record set by a ConvNet variant NFNet. Further, when pretrained on ImageNet-21K with about 10M images, CoAtNet reaches  $88.56\%$  top-1 accuracy when finetuned on ImageNet-1K, matching the ViT-Huge pretrained on JFT-300M, a  $23\times$  larger dataset. Finally, when JFT is used for pre-training, CoAtNet exhibits better efficiency compared to ViT, pushes the ImageNet-1K top-1 accuracy to  $89.77\%$  with less amount of computation.

# 2 Related Work

Convolutional network building blocks: Convolutional Networks (ConvNets) have been the dominating neural architectures for many computer vision tasks. Traditionally, regular convolutions, such as ResNet blocks [3], are popular in large-scale ConvNets; in contrast, depthwise convolutions [26] are popular in mobile platforms due to its lower computational cost and smaller parameter size [27]. Recent works show that an improved inverted residual bottlenecks (MBConv [27, 28]), which is built upon depthwise convolutions, can achieve both high accuracy and better efficiency [5, 19]. This paper will mostly employs MBConv as convolutional building blocks, and interestingly we will show the strong connection between MBConv and transformer blocks.

Self-attention and transformers: With the key ingredients of self-attention, transformers have been widely adopted for neural language processing and speech understanding. An early work that shows self-attention alone can work for vision tasks is the stand-alone self-attention network [29]. Recently, ViT [13] applies a vanilla transformer to ImageNet classification, and achieves impressive results by pretraining on a large-scale JFT dataset, but they still largely lag behind state-of-the-art ConvNets when training data is limited. Following to that, many recent works have been focused on improving vision transformers for data efficiency and model efficiency. For a more comprehensive review of vision transformers, we refer readers to the dedicated surveys [30, 31].

Combining convolution and self-attention. Previous works have also tried to combine convolution and self-attention for vision recognition. A common approach is to augment the ConvNet backbone with explicit self-attention or non-local modules [9, 10, 11, 12], or replace certain convolution layers with standard self-attention [11] or a more delicate mix of linear attention and convolution [32]. While self-attention usually improves the accuracy, they come with significant extra computational cost and hence are often regarded as an add-on to the ConvNets, similar to squeeze-and-excitation [33] module. In comparison, after the success of ViT and ResNet-ViT [13], another popular line of research starts with a Transformer backbone and tries to incorporate explicit convolution or some desirable properties of convolution into the Transformer backbone [25, 24, 23, 22, 21, 34, 35].

While our work also belongs to this category, we show that our relative attention instantiation is a natural mixture of depthwise convolution and content-based attention with minimum additional cost. More importantly, starting from the perspectives of generalization and capacity, we take a systematic approach to the vertical layout design and show how and why different network stages prefer different types of layers.

# 3 Model

In the section, we focus on the question of how to "optimally" combine the convolution and transformer. Roughly speaking, we decompose the question into two parts:

1. How to combine the convolution and self-attention within one basic computational block?  
2. How to vertically stack different types of computational blocks together to form a complete network?

The rationale of the decomposition will become clearer as we gradually reveal our design choices.

# 3.1 Merging Convolution and Self-Attention

For convolution, we mainly focus on the MBConv block [27] which employs depthwise convolution [26] to capture the spatial interaction. A key reason of this choice is that both the FFN module in Transformer and MBConv employ the design of "inverted bottleneck", which first expands the channel size of the input by  $4\mathrm{x}$  and later project the the  $4\mathrm{x}$ -wide hidden state back to the original channel size to enable residual connection.

Besides the similarity of inverted bottleneck, we also notice that both depthwise convolution and self-attention can be expressed as a weighted sum of values in a pre-defined receptive field. Specifically, convolution relies on a fixed kernel to gather information from a local receptive field

$$
y _ {i} = \sum_ {j \in \mathcal {L} (i)} w _ {i - j} \odot x _ {j} \quad (\text {d e p t h w i s e c o n v o l u t i o n}), \tag {1}
$$

where  $x_{i}, y_{i} \in \mathbb{R}^{D}$  are the input and output at position  $i$  respectively, and  $\mathcal{L}(i)$  denotes a local neighborhood of  $i$ , e.g., a 3x3 grid centered at  $i$  in image processing.

In comparison, self-attention allows the receptive field to be the entire spatial locations and computes the weights based on the re-normalized pairwise similarity between the pair  $(x_{i}, x_{j})$ :<sup>2</sup>

$$
y _ {i} = \sum_ {j \in \mathcal {G}} \underbrace {\frac {\exp \left(x _ {i} ^ {\top} x _ {j}\right)}{\sum_ {k \in \mathcal {G}} \exp \left(x _ {i} ^ {\top} x _ {k}\right)}} _ {A _ {i, j}} x _ {j} \quad (\text {s e l f - a t t e n t i o n}), \tag {2}
$$

where  $\mathcal{G}$  indicates the global spatial space. Before getting into the question of how to best combine them, it is worthwhile to compare their relative strengths and weaknesses, which helps to figure out the good properties we hope to retain.

- First of all, the depthwise convolution kernel  $w_{i-j}$  is an input-independent parameter of static value, while the attention weight  $A_{i,j}$  dynamically depends on the representation of the input. Hence, it is much easier for the self-attention to capture complicated relational interactions between different spatial positions, a property that we desire most when processing high-level concepts. However, the flexibility comes with a risk of easier overfitting, especially when data is limited.  
- Secondly, notice that given any position pair  $(i,j)$ , the corresponding convolution weight  $w_{i-j}$  only cares about the relative shift between them, i.e.  $i - j$ , rather than the specific values of  $i$  or  $j$ . This property is often referred to translation equivalence, which has been found to improve generalization under datasets of limited size [36]. Due to the usage of absolution positional embeddings, standard Transformer (ViT) lacks this property. This partially explains why ConvNets are usually better than Transformers when the dataset is not enormously large.  
- Finally, the size of the receptive field is one of the most crucial differences between self-attention and convolution. Generally speaking, a larger receptive field provides more contextual information,

which could lead to higher model capacity. Hence, the global receptive field has been a key motivation to employ self-attention in vision. However, a large receptive field requires significantly more computation. In the case of global attention, the complexity is quadratic w.r.t. spatial size, which has been a fundamental trade-off in applying self-attention models.

Table 1: Desirable properties found in convolution or self-attention.  

<table><tr><td>Properties</td><td>Convolution</td><td>Self-Attention</td></tr><tr><td>Translation Equivariance</td><td>✓</td><td></td></tr><tr><td>Input-adaptive Weighting</td><td></td><td>✓</td></tr><tr><td>Global Receptive Field</td><td></td><td>✓</td></tr></table>

Given the comparison above, an ideal model should be able to combine the 3 desirable properties in Table 1. With the similar form of depthwise convolution in Eqn. (1) and self-attention in Eqn. (2), a straightforward idea that could achieve this is simply to sum a global static convolution kernel with the adaptive attention matrix, either after or before the Softmax normalization, i.e.,

$$
y _ {i} ^ {\text {p o s t}} = \sum_ {j \in \mathcal {G}} \left(\frac {\exp \left(x _ {i} ^ {\top} x _ {j}\right)}{\sum_ {k \in \mathcal {G}} \exp \left(x _ {i} ^ {\top} x _ {k}\right)} + w _ {i - j}\right) x _ {j} \text {o r} y _ {i} ^ {\text {p r e}} = \sum_ {j \in \mathcal {G}} \frac {\exp \left(x _ {i} ^ {\top} x _ {j} + w _ {i - j}\right)}{\sum_ {k \in \mathcal {G}} \exp \left(x _ {i} ^ {\top} x _ {k} + w _ {i - k}\right)} x _ {j}. \tag {3}
$$

Interestingly, while the idea seems overly simplified, the pre-normalization version  $y^{\mathrm{pre}}$  corresponds to a particular variant of relative self-attention [37, 38]. In this case, the attention weight  $A_{i,j}$  is decided jointly by the  $w_{i-j}$  of translation equivariance and the input-adaptive  $x_i^\top x_j$ , which can enjoy both effects depending on their relative magnitudes. Importantly, note that in order to enable the global convolution kernel without blowing up the number of parameters, we have reloaded the notation of  $w_{i-j}$  as a scalar (i.e.,  $w \in \mathbb{R}^{O(|\mathcal{G}|)}$ ) rather than a vector in Eqn. (1). Another advantage of the scalar formulation of  $w$  is that retrieving  $w_{i-j}$  for all  $(i,j)$  is clearly subsumed by computing the pairwise dot-product attention, hence resulting in minimum additional cost (see Appendix A.1). Given the benefits, we will use the Transformer block with the pre-normalization relative attention variant in Eqn. (3) as the key component of the proposed CoAtNet model.

# 3.2 Vertical Layout Design

After figuring out a neat way to combine convolution and attention, we next consider how to utilize it to stack an entire network.

As we have discuss above, the global context has a quadratic complexity w.r.t. the spatial size. Hence, if we directly apply the relative attention in Eqn. (3) to the raw image input, the computation will be excessively slow due to the large number of pixels in any image of common sizes. Hence, to construct a network that is feasible in practice, we have mainly three options:

(A) Perform some down-sampling to reduce the spatial size and employ the global relative attention after the feature map reaches manageable level.  
(B) Enforce local attention, which restricts the global receptive field  $\mathcal{G}$  in attention to a local field  $\mathcal{L}$  just like in convolution [22, 21].  
(C) Replace the quadratic Softmax attention with certain linear attention variant which only has a linear complexity w.r.t. the spatial size [12, 39, 40].

We briefly experimented with option (C) without getting a reasonably good result. For option (B), we found that implementing local attention involves many non-trivial shape formatting operations that requires intensive memory access. On our accelerator of choice (TPU), such operation turns out to be extremely slow [29], which not only defeats the original purpose of speeding up global attention, but also hurts the model capacity. Hence, as some recent work has studied this variant [22, 21], we will focus on option (A) and compare our results with theirs in our empirical study (Section 4).

For option (A), the down-sampling can be achieved by either (1) a convolution stem with aggressive stride (e.g., stride 16x16) as in ViT or (2) a multi-stage network with gradual pooling as in ConvNets. With these choices, we derive a search space of 5 variants and compare them in controlled experiments.

- When the ViT Stem is used, we directly stack  $L$  Transformer blocks with relative attention, which we denote as  $\mathrm{VIT}_{\mathrm{REL}}$ .

- When the multi-stage layout is used, we mimic ConvNets to construct a network of 5 stages (S0, S1, S2, S3 & S4), with spatial resolution gradually decreased from S0 to S4. At the beginning of each stage, we always reduce the spatial size by  $2\mathrm{x}$  and increase the number of channels. The first stage S0 is a simple 2-layer convolutional Stem and S1 always employs MBCnv blocks with squeeze-excitation (SE), as the spatial size is too large for global attention. Starting from S2 through S4, we consider either the MBCnv or the Transformer block, with a constraint that convolution stages must appear before Transformer stages. The constraint is based on the prior that convolution is better at processing local patterns that are more common in early stages. This leads to 4 variants with increasingly more Transformer stages, C-C-C-C, C-C-C-T, C-C-T-T and C-T-T-T, where C and T denote Convolution and Transformer respectively.

To systematically study the design choices, we consider two fundamental aspects generalization capability and model capacity: For generalization, we are interested in the gap between the training loss and the evaluation accuracy. If two models have the same training loss, then the model with higher evaluation accuracy has better generalization capability, since it can generalize better to unseen evaluation dataset. Generalization capability is particularly important to data efficiency when training data size is limited. For model capacity, we measure the ability to fit large training datasets. When training data is abundant and overfitting is not an issue, the model with higher capacity will achieve better final performance after reasonable training steps. Note that, since simply increasing the model size can lead to higher model capacity, to perform a meaningful comparison, we make sure the model sizes of the 5 variants are comparable.

To compare the generalization and model capacity, we train different variants of hybrid models on ImageNet-1K (1.3M) and JFT ( $>300\mathrm{M}$ ) dataset for 300 and 3 epochs respectively, both without any regularization or augmentation. The training loss and evaluation accuracy on both datasets are summarized in Figure 1.

![](images/bc1e74e53d35b30675b943b06fc4eba0fbf86dfa7a265598523b94fc693c8611.jpg)

![](images/4e219a30612fb8ad4378f4b946d6af8761727f52aca7e5445b23c8069cb35968.jpg)  
(a) ImageNet-1K

![](images/fea1235da2d9ad616485e0229a010b9c03901990b2cac779b702f7bf147248d8.jpg)

![](images/2ea089607a75d3fe51532b9c2a513b6ffd3de35d59232f27dd4a9701ea78e868.jpg)  
Figure 1: Comparison for model generalization and capacity under different data size. For fair comparison, all models have similar parameter size and computational cost.  
(b) JFT

- From the ImageNet-1K results, a key observation is that, in terms of generalization capability (i.e., gap between train and evaluation metrics), we have

$$
C - C - C - C \approx C - C - C - T \geq C - C - T - T > C - T - T - T \gg V I T _ {R E L}.
$$

Particularly,  $\mathrm{VIT}_{\mathrm{REL}}$  is significantly worse other variants by a large margin, which we conjecture is related to the lack of proper low-level information processing in its aggressive down-sampling Stem. Among the multi-stage variants, the overall trend is that the more convolution stages the model has, the smaller the generalization gap is.  
- As for model capacity, from the JFT comparison, both the train and evaluation metrics at the end of the training suggest the following ranking:

$$
C - C - T - T \approx C - T - T - T > V I T _ {R E L} > C - C - C - T > C - C - C - C.
$$

Importantly, this suggests that simply having more Transformer blocks does NOT necessarily mean higher capacity for visual processing. On one hand, while initially worse,  $\mathrm{ViT}_{\mathrm{REL}}$  ultimately catch up with the two variants with more MBConv stages, indicating the capacity advantage of Transformer blocks. On the other hand, both C-C-T-T and C-T-T-T clearly outperforming  $\mathrm{ViT}_{\mathrm{REL}}$  suggest that the ViT stem with an aggressive stride may have lost too much information and hence limit the model capacity. More interestingly, the fact that C-C-T-T ≈ C-T-T-T indicates the for

processing low-level information, static local operations like convolution could be as capable as adaptive global attention mechanism, while saving computation and memory usage substantially.

Finally, to decide between C-C-T-T and C-T-T-T, we conduct another transferability test<sup>3</sup> - we finetune the two JFT pretrained models above on ImageNet-1K for 30 epochs and compare their transfer performances. From Table 2, it turns out that C-C-T-T achieves a clearly better transfer accuracy than C-T-T-T, despite the same pre-training performance.

Table 2: Transferability test results.  

<table><tr><td>Metric</td><td>C-C-T-T</td><td>C-T-T-T</td></tr><tr><td>Pre-training Precision@1 (JFT)</td><td>34.40</td><td>34.36</td></tr><tr><td>Transfer Accuracy 224x224</td><td>82.39</td><td>81.78</td></tr><tr><td>Transfer Accuracy 384x384</td><td>84.23</td><td>84.02</td></tr></table>

Taking generalization, model capacity, transferability and efficiency into consideration, we adapt the C-C-T-T multi-stage layout for CoAtNet. More model details are included in Appendix A.1.

# 4 Experiments

In this section, we compare CoAtNet with previous results under comparable settings. For completeness, all the hyper-parameters not mentioned here are included in Appendix A.2.

# 4.1 Experiment Setting

CoAtNet model family. To compare with existing models of different sizes, we also design a family of CoAtNet models as summarized in Table 3. Overall, we always double the number of channels from S1 to S4, while ensuring the width of the Stem S0 to be smaller or equal to that of S1. Also, for simplicity, when increasing the depth of the network, we only scale the number of blocks in S2 and S3.

Table 3: L denotes the number of blocks and D denotes the hidden dimension (#channels). For all Conv and MBConv blocks, we always use the kernel size 3. For all Transformer blocks, we set the size of each attention head to 32, following [22]. The expansion rate for the inverted bottleneck is always 4 and the expansion (shrink) rate for the SE is always 0.25.  

<table><tr><td>Stages</td><td>Size</td><td colspan="2">CoAtNet-0</td><td colspan="2">CoAtNet-1</td><td colspan="2">CoAtNet-2</td><td colspan="2">CoAtNet-3</td><td colspan="2">CoAtNet-4</td></tr><tr><td>S0-Conv</td><td>1/2</td><td>L=2</td><td>D=64</td><td>L=2</td><td>D=64</td><td>L=2</td><td>D=128</td><td>L=2</td><td>D=192</td><td>L=2</td><td>D=192</td></tr><tr><td>S1-MbConv</td><td>1/4</td><td>L=2</td><td>D=96</td><td>L=2</td><td>D=96</td><td>L=2</td><td>D=128</td><td>L=2</td><td>D=192</td><td>L=2</td><td>D=192</td></tr><tr><td>S2-MBConv</td><td>1/8</td><td>L=3</td><td>D=192</td><td>L=6</td><td>D=192</td><td>L=6</td><td>D=256</td><td>L=6</td><td>D=384</td><td>L=12</td><td>D=384</td></tr><tr><td>S3-TFMRel</td><td>1/16</td><td>L=5</td><td>D=384</td><td>L=14</td><td>D=384</td><td>L=14</td><td>D=512</td><td>L=14</td><td>D=768</td><td>L=28</td><td>D=768</td></tr><tr><td>S4-TFMRel</td><td>1/32</td><td>L=2</td><td>D=768</td><td>L=2</td><td>D=768</td><td>L=2</td><td>D=1024</td><td>L=2</td><td>D=1536</td><td>L=2</td><td>D=1536</td></tr></table>

Evaluation Protocol. Our experiments focus on image classification. To evaluate the performance of the model across different data sizes, we utilize three datasets of increasingly larger sizes, namely ImageNet-1K (1.28M images), ImageNet-21K (12.7M images) and JFT (300M images). Following previous works, we first pretrain our models on each of the three datasets at resolution 224 for 300, 90 and 14 epochs respectively. Then, we finetune the pretrained models on ImageNet-1K at the desired resolutions for 30 epochs and obtain the corresponding evaluation accuracy. One exception is the ImageNet-1K performance at resolution 224, which can be directly obtained at the end of pre-training. Note that similar to other models utilizing Transformer blocks, directly evaluating models pre-trained on ImageNet-1K at a larger resolution without finetuning usually leads to performance drop. Hence, finetuning is always employed whenever input resolution changes.

Data Augmentation & Regularization. In this work, we only consider two widely used data augmentations, namely RandAugment [41] and MixUp [42], and three common techniques, including stochastic depth [43], label smoothing [44] and weight decay [45], to regularize the model. Intuitively,

the specific hyper-parameters of the augmentation and regularization methods depend on model size and data scale, where strong regularization is usually applied for larger models and smaller dataset.

Under the general principle, a complication under the current paradigm is how to adjust the regularization for pretraining and finetuning as data size can change. Specifically, we have an interesting observation that if a certain type of augmentation is entirely disabled during pre-training, simply turning it on during fine-tuning would most likely harm the performance rather than improving. We conjecture this could be related to data distribution shift. As a result, for certain runs of the proposed model, we deliberately apply RandAugment and stochastic depth of a small degree when pre-training on the two larger datasets, ImageNet21-K and JFT. Although such regularization can harm the pre-training metrics, this allows more versatile regularization and augmentation during finetuning, leading to improved down-stream performances.

# 4.2 Main Results

![](images/032d21d6779561a5e3cdeb8f6799cb7af61c74e7714e2100d5f99977ccdda2c8.jpg)  
Figure 2: Accuracy-to-FLOPs scaling curve under ImageNet-1K only setting at  $224\mathrm{x}224$

![](images/c314f94e19eda2e483ffdd3f85af11c51f4d2ce95901f372a34dc410893bd1d3.jpg)  
Figure 3: Accuracy-to-Params scaling curve under ImageNet-21K  $\Rightarrow$  ImageNet-1K setting.

ImageNet-1K The experiment results with only the ImageNet-1K dataset are shown in Table 4. Under similar conditions, the proposed CoAtNet models not only outperform ViT variants, but also match the best convolution-only architectures, i.e., EfficientNet-V2 and NFNets. Additionally, we also visualize the all results at resolution  $224 \times 224$  in Fig. 2. As we can see, CoAtNet scales much better than previous model with attention modules.

ImageNet-21K As we can see from Table 4 and Fig. 3, when ImageNet-21K is used for pretraining, the advantage of CoAtNet becomes more obvious, substantially outperforming all previous models. Notably, the best CoAtNet variant achieves a top-1 accuracy of  $88.56\%$ , matching the ViT-H/14 performance of  $88.55\%$ , which requires pre-training the  $2.3\mathrm{x}$  larger ViT model on a  $23\mathrm{x}$  larger proprietary weakly labeled dataset (JFT) for  $2.2\mathrm{x}$  more steps. This marks a dramatic improvement in both data efficiency and computation efficiency.

JFT Finally, in Table 5, we further evaluate CoAtNet under the large-scale data regime with JFT. Encouragingly, our CoAtNet-4 can almost match the best previous performance with JFT set by NFNet-F4+, while being 2x more efficient in terms of both TPU training time and parameter count. When we scale up the model to consume similar training resource as NFNet-F4+, CoAtNet reaches  $89.77\%$  on top-1 accuracy, outperforming previous results under comparable settings.

# 4.3 Ablation Studies

In this section, we will ablate our design choices for CoAtNet.

Firstly, we study the importance of the relative attention from combining convolution and attention into a single computation unit. Specifically, we compare two models, one with the relative attention and the other without, under both the ImageNet-1K alone and ImageNet-21K transfer setting. As we can see from Table 6, when only the ImageNet-1K is used, relative attention clearly outperforms the

Table 4: Model performance on ImageNet. 1K only denotes training on ImageNet-1K only;  $21\mathrm{K} + 1\mathrm{K}$  denotes pretraining on ImageNet-21K and finetuning on ImageNet-1K; PT-RA denotes applying RandAugment during 21K pre-training, and E150 means 150 epochs of 21K pre-training, which is longer than the standard 90 epochs. More results are in Appendix A.3.  

<table><tr><td colspan="2">Models</td><td>Eval Size</td><td>#Params</td><td>#FLOPs</td><td colspan="2">ImageNet Top-1 Accuracy</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>1K only</td><td>21K+1K</td></tr><tr><td rowspan="3">Conv Only</td><td>ENetV2-L</td><td>4802</td><td>121M</td><td>53B</td><td>85.7</td><td>86.8</td></tr><tr><td>NFNet-F3</td><td>4162</td><td>255M</td><td>114.8B</td><td>85.7</td><td>-</td></tr><tr><td>NFNet-F5</td><td>5442</td><td>377M</td><td>289.8B</td><td>86.0</td><td>-</td></tr><tr><td rowspan="4">ViT-Stem TFM</td><td>DeiT-B</td><td>3842</td><td>86M</td><td>55.4B</td><td>83.1</td><td>-</td></tr><tr><td>ViT-L/16</td><td>3842</td><td>304M</td><td>190.7B</td><td>-</td><td>85.3</td></tr><tr><td>CaiT-S-36</td><td>3842</td><td>68M</td><td>48.0B</td><td>85.0</td><td>-</td></tr><tr><td>DeepViT-L</td><td>2242</td><td>55M</td><td>12.5B</td><td>83.1</td><td>-</td></tr><tr><td rowspan="2">Multi-stage TFM</td><td>Swin-B</td><td>3842</td><td>88M</td><td>47.0B</td><td>84.2</td><td>86.0</td></tr><tr><td>Swin-L</td><td>3842</td><td>197M</td><td>103.9B</td><td>-</td><td>86.4</td></tr><tr><td rowspan="5">Conv+TFM</td><td>BotNet-T7</td><td>3842</td><td>75.1M</td><td>45.8B</td><td>84.7</td><td>-</td></tr><tr><td>LambdaResNet-420</td><td>3202</td><td>-</td><td>-</td><td>84.8</td><td>-</td></tr><tr><td>T2T-ViT-24</td><td>2242</td><td>64.1M</td><td>15.0B</td><td>82.6</td><td>-</td></tr><tr><td>CvT-21</td><td>3842</td><td>32M</td><td>24.9B</td><td>83.3</td><td>-</td></tr><tr><td>CvT-W24</td><td>3842</td><td>277M</td><td>193.2B</td><td>-</td><td>87.7</td></tr><tr><td rowspan="16">Conv+TFM (ours)</td><td>CoAtNet-0</td><td>2242</td><td>25M</td><td>4.2B</td><td>81.6</td><td>-</td></tr><tr><td>CoAtNet-1</td><td>2242</td><td>42M</td><td>8.4B</td><td>83.3</td><td>-</td></tr><tr><td>CoAtNet-2</td><td>2242</td><td>75M</td><td>15.7B</td><td>84.1</td><td>87.1</td></tr><tr><td>CoAtNet-3</td><td>2242</td><td>167M</td><td>34.7B</td><td>84.5</td><td>87.6</td></tr><tr><td>CoAtNet-0</td><td>3842</td><td>25M</td><td>13.4B</td><td>83.9</td><td>-</td></tr><tr><td>CoAtNet-1</td><td>3842</td><td>42M</td><td>27.4B</td><td>85.1</td><td>-</td></tr><tr><td>CoAtNet-2</td><td>3842</td><td>75M</td><td>49.8B</td><td>85.7</td><td>87.1</td></tr><tr><td>CoAtNet-3</td><td>3842</td><td>167M</td><td>107.4B</td><td>85.8</td><td>87.6</td></tr><tr><td>CoAtNet-4</td><td>3842</td><td>275M</td><td>189.5B</td><td>-</td><td>87.9</td></tr><tr><td>+ PT-RA</td><td>3842</td><td>275M</td><td>189.5B</td><td>-</td><td>88.3</td></tr><tr><td>+ PT-RA-E150</td><td>3842</td><td>275M</td><td>189.5B</td><td>-</td><td>88.4</td></tr><tr><td>CoAtNet-2</td><td>5122</td><td>75M</td><td>96.7B</td><td>85.9</td><td>87.3</td></tr><tr><td>CoAtNet-3</td><td>5122</td><td>167M</td><td>203.1B</td><td>86.0</td><td>87.9</td></tr><tr><td>CoAtNet-4</td><td>5122</td><td>275M</td><td>360.9B</td><td>-</td><td>88.1</td></tr><tr><td>+ PT-RA</td><td>5122</td><td>275M</td><td>360.9B</td><td>-</td><td>88.4</td></tr><tr><td>+ PT-RA-E150</td><td>5122</td><td>275M</td><td>360.9B</td><td>-</td><td>88.56</td></tr></table>

Table 5: Performance Comparison on large-scale JFT dataset. TPUv3-core-days denotes the pretraining time, Top-1 Accuracy denotes the finetuned accuracy on ImageNet. See Appendix A.2 for the size details of CoAtNet-5.  

<table><tr><td>Models</td><td>Eval Size</td><td>#Params</td><td>#FLOPs</td><td>TPUv3-core-days</td><td>Top-1 Accuracy</td></tr><tr><td>ViT-L/16</td><td>5122</td><td>307M</td><td>364B</td><td>0.68k</td><td>87.76</td></tr><tr><td>ViT-H/14</td><td>5182</td><td>632M</td><td>1021B</td><td>2.5k</td><td>88.55</td></tr><tr><td>NFNet-F4+</td><td>5122</td><td>527M</td><td>367B</td><td>1.86k</td><td>89.2</td></tr><tr><td>CoAtNet-4</td><td>3842</td><td>275M</td><td>189.5B</td><td>0.95k</td><td>88.91</td></tr><tr><td>CoAtNet-4</td><td>5122</td><td>275M</td><td>361B</td><td>0.95k</td><td>89.11</td></tr><tr><td>CoAtNet-5</td><td>5122</td><td>688M</td><td>812B</td><td>1.82k</td><td>89.77</td></tr></table>

standard attention, indicating a better generalization. In addition, under the ImageNet-21K transfer setting, the relative attention variant achieves a substantially better transfer accuracy, despite their very close pre-training performances. This suggests the main advantage of relative attention in visual processing is not in higher capacity but in better generalization.  
Secondly, as S2 with MBConv blocks and S3 with relative Transformer blocks occupy most of the computation of the CoAtNet, a question to ask is how to split the computation between S2 (MBConv)

Table 6: Ablation on relative attention.  

<table><tr><td>Setting</td><td>Metric</td><td>With Rel-Attn</td><td>Without Rel-Attn</td></tr><tr><td rowspan="2">ImageNet-1K</td><td>Accuracy (2242)</td><td>84.1</td><td>83.8</td></tr><tr><td>Accuracy (3842)</td><td>85.7</td><td>85.3</td></tr><tr><td>ImageNet-21K</td><td>Pre-train Precision@1 (2242)</td><td>53.0</td><td>52.8</td></tr><tr><td>⇒ ImageNet-1K</td><td>Finetune Accuracy (3842)</td><td>87.9</td><td>87.4</td></tr></table>

Table 7: Ablation on architecture layout.  

<table><tr><td>Setting</td><td>Models</td><td>Layout</td><td>Top-1 Accuracy</td></tr><tr><td rowspan="3">ImageNet-1K</td><td>V0: CoAtNet-2</td><td>[2, 2, 6, 14, 2]</td><td>84.1</td></tr><tr><td>V1: S2 ⇌ S3</td><td>[2, 2, 2, 18, 2]</td><td>83.4</td></tr><tr><td>V2: S2 ⇒ S3</td><td>[2, 2, 8, 12, 2]</td><td>84.0</td></tr><tr><td>ImageNet-21K</td><td>V0: CoAtNet-3</td><td>[2, 2, 6, 14, 2]</td><td>53.0 → 87.6</td></tr><tr><td>⇒ ImageNet-1K</td><td>V1: S2 ⇌ S3</td><td>[2, 2, 2, 18, 2]</td><td>53.0 → 87.4</td></tr></table>

Table 8: Ablation on head size and normalization type.  

<table><tr><td>Setting</td><td>Models</td><td>Image Size</td><td>Top-1 Accuracy</td></tr><tr><td rowspan="3">ImageNet-1K</td><td>CoAtNet-2</td><td>2242</td><td>84.1</td></tr><tr><td>Head size: 32 → 64</td><td>2242</td><td>83.9</td></tr><tr><td>Norm type: BN → LN</td><td>2242</td><td>84.1</td></tr><tr><td>ImageNet-21K</td><td>CoAtNet-3</td><td>3842</td><td>87.9</td></tr><tr><td>⇒ ImageNet-1K</td><td>Norm type: BN → LN</td><td>3842</td><td>87.8</td></tr></table>

and S3 (Transformer) to achieve a good performance. In practice, it boils down to deciding the number of blocks to have in each stage, which we will refer to as "layout" design. For this purpose, we compare a few different layouts that we experimented with in Table 7.

- If we keep the total number of blocks in S2 and S3 fixed and vary the number in each stage, we observe that V0 is a sweet spot between V1 and V2. Basically, having more Transformer blocks in S3 generally leads to better performance until the number of MBConv blocks in S2 is too small to generalize well.  
- To further evaluate whether the sweet spot also holds in the transfer setting, where a higher capacity is often regarded more important, we further compare V0 and V1 under the ImageNet-21K transferring to ImageNet-1K setup. Interestingly, despite that V1 achieves a slightly better performance during ImageNet-21K pre-training than V0 does, the transfer accuracy of V1 clearly falls behind V0. Again, this suggests the importance of convolution in achieving good generalization.

Lastly, we study two choices of model details, namely the dimension of each attention (default to 32) head as well as the type of normalization (default to BatchNorm) used in MBConv blocks. From Table 8, we can see increasing head size from 32 to 64 can slightly hurt performance, though it actually improves the TPU speed by a significant amount. In practice, this will be a quality-speed trade-off one can make. On the other hand, BatchNorm and LayerNorm have almost the same performance, while BatchNorm is  $10 - 20\%$  faster on TPU depending on the per-core batch size.

# 5 Conclusion

In this paper, we systematically study the properties of convolutions and transformers, which leads to a principled way to combine them into a new family of models named CoAtNet. Extensive experiments show that CoAtNet enjoys both good generalization like ConvNets and superior model capacity like Transformers, achieving state-of-the-art performances under different data sizes and computation budgets.

Note that this paper currently focuses on ImageNet classification for model development. However, we believe our approach is applicable to broader applications like object detection and semantic segmentation. We will leave them for future work.

# References

[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, pages 1097-1105, 2012.  
[2] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.  
[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.  
[4] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1-9, 2015.  
[5] Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. ICML, 2019.  
[6] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017.  
[7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.  
[8] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.  
[9] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7794-7803, 2018.  
[10] Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, and Quoc V Le. Attention augmented convolutional networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3286-3295, 2019.  
[11] Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, Jonathon Shlens, Pieter Abbeel, and Ashish Vaswani. Bottleneck transformers for visual recognition. arXiv preprint arXiv:2101.11605, 2021.  
[12] Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, and Hongsheng Li. Efficient attention: Attention with linear complexities. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3531-3539, 2021.  
[13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.  
[14] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. IEEE, 2009.  
[15] Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In Proceedings of the IEEE international conference on computer vision, pages 843-852, 2017.  
[16] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. arXiv preprint arXiv:2012.12877, 2020.

[17] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou. Going deeper with image transformers. arXiv preprint arXiv:2103.17239, 2021.  
[18] Daquan Zhou, Bingyi Kang, Xiaojie Jin, Linjie Yang, Xiaochen Lian, Qibin Hou, and Jiashi Feng. Deepvit: Towards deeper vision transformer. arXiv preprint arXiv:2103.11886, 2021.  
[19] Mingxing Tan and Quoc V Le. Efficientnetv2: Smaller models and faster training. ICML, 2021.  
[20] Andrew Brock, Soham De, Samuel L Smith, and Karen Simonyan. High-performance large-scale image recognition without normalization. arXiv preprint arXiv:2102.06171, 2021.  
[21] Ashish Vaswani, Prajit Ramachandran, Aravind Srinivas, Niki Parmar, Blake Hechtman, and Jonathon Shlens. Scaling local self-attention for parameter efficient visual backbones. arXiv preprint arXiv:2103.12731, 2021.  
[22] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. arXiv preprint arXiv:2103.14030, 2021.  
[23] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. Cvt: Introducing convolutions to vision transformers. arXiv preprint arXiv:2103.15808, 2021.  
[24] Ben Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Herve Jégou, and Matthijs Douze. Levit: a vision transformer in convnet's clothing for faster inference. arXiv preprint arXiv:2104.01136, 2021.  
[25] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch onImagenet. arXiv preprint arXiv:2101.11986, 2021.  
[26] Laurent Sifre. Rigid-motion scattering for image classification. Ph.D. thesis section 6.2, 2014.  
[27] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4510-4520, 2018.  
[28] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V Le. Mnasnet: Platform-aware neural architecture search for mobile. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2820-2828, 2019.  
[29] Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jonathon Shlens. Stand-alone self-attention in vision models. arXiv preprint arXiv:1906.05909, 2019.  
[30] Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, et al. A survey on visual transformer. arXiv preprint arXiv:2012.12556, 2020.  
[31] Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. arXiv preprint arXiv:2101.01169, 2021.  
[32] Irwan Bello. Lambda networks: Modeling long-range interactions without attention. arXiv preprint arXiv:2102.08602, 2021.  
[33] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7132-7141, 2018.  
[34] Kun Yuan, Shaopeng Guo, Ziwei Liu, Aojun Zhou, Fengwei Yu, and Wei Wu. Incorporating convolution designs into visual transformers. arXiv preprint arXiv:2103.11816, 2021.  
[35] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. arXiv preprint arXiv:2102.12122, 2021.

[36] Mirgahney Mohamed, Gabriele Cesa, Taco S Cohen, and Max Welling. A data and compute efficient design for limited-resources deep learning. arXiv preprint arXiv:2004.09691, 2020.  
[37] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. arXiv preprint arXiv:1803.02155, 2018.  
[38] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.  
[39] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, pages 5156-5165. PMLR, 2020.  
[40] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.  
[41] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 702-703, 2020.  
[42] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017.  
[43] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In European conference on computer vision, pages 646-661. Springer, 2016.  
[44] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818-2826, 2016.  
[45] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.  
[46] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In European conference on computer vision, pages 630-645. Springer, 2016.  
[47] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.  
[48] Zihang Dai, Guokun Lai, Yiming Yang, and Quoc V Le. Funnel-transformer: Filtering out sequential redundancy for efficient language processing. arXiv preprint arXiv:2006.03236, 2020.
