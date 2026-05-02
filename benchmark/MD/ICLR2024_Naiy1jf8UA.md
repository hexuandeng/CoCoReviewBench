# MGDC-UNET: MULTI-GROUP DEFORMABLE CONVOLUTION FOR MEDICAL IMAGE SEGMENTATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recently, there has been growing interest in developing Vision Transformer (ViT) or Convolutional Neural Network (CNN) methods for 3D medical image segmentation, which necessitates both large receptive fields and adaptations to varying spatial geometries. Previous works in both CNNs and ViTs demonstrated limitations in capturing the complex spatial and semantic structure of 3D medical images. In this paper, we introduce MGDC-UNet, a multi-group deformable convolution network for 3D volumetric medical image segmentation. Our MGDC-UNet employs deformable convolution operators with learnable spatial offsets to improve attention on semantically important regions. Our approach leverages stable spatial distribution across subjects to enhance semantic learning. We also incorporate transformer components to augment feature learning and reduce inductive biases inherent in traditional CNNs. MGDC-UNet demonstrated superior performance accuracy on three challenging segmentation tasks using public datasets: 1). brain tumor segmentation (BraTS21), 2). CT multi-organ segmentation (FLARE21) and 3). cross-modality MR/CT segmentation (AMOS22). Our network also compared favorably with existing methods in terms of computational efficiency.

# 1 INTRODUCTION

Volumetric medical image segmentation plays an important role in the identification and delineation of specific regions, such as tumors or organs, within 3D medical images. In diagnostic and therapeutic applications, this technique aids clinicians in precisely determining the location and scale of pathological changes, which consequently enhances treatment planning and improves patients' quality of life. However, the task of volumetric medical image segmentation is challenging. The complexity of anatomical structures, such as the congestion or even the invasion among tissues, organs, and systems in the limited human body space, may complicate the segmentation process. Additionally, the large volume of 3D image data often demands substantial resources and efficiency.

Previous learning-based approaches have shown remarkable performance in medical image analysis tasks, particularly the U-Net architecture in volumetric image segmentation (Ronneberger et al., 2015). However, existing methods demonstrate limitations in effective receptive fields (ERFs) when dealing with the complicated structure and semantics of volumetric medical image segmentation. Our analysis, illustrated in Fig. 1, shows the ERF distributions of previous network designs lack specificity towards pertinent anatomical structures. Conventional CNN is constrained by its uniform convolution strategy. Since plain convolution kernel samples evenly on the feature map, it underperforms in regions requiring more attention and overcompensates in regions requiring less focus. Small kernel CNN  $(3\times 3\times 3)$  is hindered by a constrained ERF thus offering limited attention and fine-grained analysis capabilities (Fig 1.a). Large kernel CNN  $(7\times 7\times 7)$  improves ERF and local segmentation accuracy but still lacks long-range dependencies (Fig 1.b). Vision Transformers have better attention mechanisms than CNNs, but fall short in capturing semantic correlations due to simple feature correlation design and the complexity of the input volumetric image structures (Fig 1.c). Furthermore, self-attention in ViTs might not inherently focus on the most semantically relevant features of the images, thereby increasing the risk of overfitting.

To address this issue, we propose a novel 3D volumetric feature extraction network designed to explicitly attract more attention to regions with relevant semantics. We observe that, despite their

complexity, medical images often possess strong location-semantics correlations. That is, the position distribution of each organ tends to remain consistent across different subjects. Inspired by Deformable Neural Networks (Zhu et al., 2019b; Wang et al., 2023), we have developed a deformable convolution approach used for 3D volumetric images. Through convolution kernels with learnable position distribution, our network can gather more attention to semantically important regions. Due to the strong correlation between semantics and spatial distribution in 3D volumetric medical images, the learned positional information tends to be more stable, leading to a more efficient and robust network that extracts more semantically accurate features. Our result in Fig. 1d shows a noticeable semantic-related spatial distribution in feature attention.

Specifically, our 3D volumetric medical image segmentation network is named MGDC-UNet. First, we design a learnable spatial offset for each deformable convolution operator which can be applied to 3D volumetric data. The network can adaptively adjust the offset of sampled locations, concentrating its attention on semantically relevant organ positions. This design leverages the stable positional prior of organs to capture robust semantic features. Furthermore, we dynamically adjust offsets and modulation scalars to mitigate the inductive biases inherent in traditional CNNs, achieving transformer-like spatial aggregation. Finally, we designed MGDC blocks with a hybrid deformable convolution and multi-layer perceptron (MLP) structure for effective channel scaling and enhanced feature learning.

Our contributions are as follows:

- We proposed a core operator named MGDC, which capitalizes on the correlation between location and semantics in medical images. Our operator achieves more accurate semantic learning through adaptive attentional positional offsets.  
- We augment our core MGDC operator with transformer components in the MGDC block to boost feature learning and attain optimal performance.  
- We evaluate our proposed architecture on three large publicly available datasets, demonstrating superior performance in terms of segmentation metrics, inference time, and model parameters.

![](images/51341a109a123e5a5ff16b8c94ab3c88b4f7c753eb3062132c9ecc28283eb5e0.jpg)

![](images/61c2dcfcdbced4d8f37a1009efa56cc0e67319dd73bc16ac892fe092d4989a9f.jpg)  
Ground Truth

![](images/fa5a7fbdcc75b8db972b4747e871eca060a1d6caed70bf31ca91b094d05a80b9.jpg)

![](images/98956d8997a7d48e03af1e73685b67f6d5e60b5d39ddbb57819df6350035f066.jpg)  
a). Small kernel conv

![](images/e6a011588c52ce270435ade327cc093d877c40575e62867b9228a01483b03def.jpg)

![](images/91a84d94de1d277b416d394911753e8074f14c8fa77bb55f4f783001f77293c1.jpg)  
b). Large kernel conv

![](images/245685c9006d202e632beb2b16ce470f7e32d7b058fab2dae1b04ae24c81b00b.jpg)

![](images/f020caf9995ff3a436bc18fd00a626bd5ac4475c6eea8b6ec803fa0bbab3748f.jpg)  
Figure 1: We compare the ERFs on segmented regions from different operations and their effects on multi-organ segmentation. Top row: ERFs from the bottleneck layer of every method. Bottom row: the segmentation results on an example CT (white arrow indicating improvements). (a) Small-kernel convolutions often segment regions without accounting for the anatomical correlation with adjacent structures. (b) Large-kernel convolutions enhance anatomical context but remain confined to considering only nearby structures. (c) Global self-attention extends the ERF but still falls short in capturing semantic relationships among correlated organs. (d) Multi-group deformable convolutions successfully expand the ERF while adapting to task-specific geometry through learnable offsets, thereby focusing on semantically relevant regions.  
c). Self-attention

![](images/a4f5828e83b2a27c3003f28d5f10a3c8535fcdbb137aed623708349f273c7076.jpg)

![](images/d9fdb0c2168d758a7f9dd03707c0bd076310ff6e605b958804f8bce46870d0f9.jpg)  
d). Multi-group deformable conv

# 2 RELATED WORKS

# 2.1 3D VOLUMETRIC MEDICAL IMAGE SEGMENTATION

Due to the strong local inductive bias and parameter efficiency, CNN-based methods have long dominated medical image segmentation. The spatial parameter sharing of CNN enables compact designs suitable for medical image analysis. (Ronneberger et al., 2015) introduced U-Net, a CNN architecture with symmetric expansive and contractive paths enabling precise localization, making it a standard choice for many segmentation tasks in medical imaging. However, the limited receptive field of CNNs can significantly hinder their performance on medical segmentation tasks, where the objects are often irregular or distorted. A standard convolutional layer with a small kernel size can only capture local spatial patterns. Even with pooling or striding, the inherent design of CNNs forces them to accumulate global context through many layers, potentially losing or diluting important long-range information. To address the locality of CNN, variants of U-Net have been proposed by leveraging novel breakthroughs from various vision tasks. Attention UNet utilized attention-gates to select important features to improve segmentation performance (Oktay et al., 2018). (Zhang et al., 2017) introduced dilated convolution and pyramid pooling to U-Net to enlarge the receptive field. Self-attention has also been applied to address the locality of convolution operation (Sinha & Dolz, 2020). Different from previous works on this task, our proposed method can leverage the correlation of spatial prior and semantics in 3D volumetric image segmentation tasks, which gives better attention to semantic relevant regions.

# 2.2 DEFORMABLE CONVOLUTION NEURAL NETWORKS

Deformable convolution has emerged as a powerful technique for addressing the limitations of traditional CNNs in tasks requiring adaptive receptive fields, such as image segmentation. Initial contributions, such as Deformable ConvNet by (Dai et al., 2017) laid the foundation by introducing dynamic offsets to adapt receptive fields. Subsequent advancements, such as DCNv2 and DCNv3, incorporated learnable modulation scalars and multi-group spatial aggregation for greater flexibility and efficiency (Zhu et al., 2019b; Wang et al., 2023). While deformable convolution has been effectively applied in 2D medical image segmentation and 3D CT multi-organ segmentation tasks (Jin et al., 2019; Heinrich et al., 2019), its full potential in combination with transformer-like architectures for 3D medical image segmentation remains underexplored. Our hypothesis is that deformable convolution can significantly augment transformer-like architectures, offering benefits in handling long-range dependencies and providing computational efficiency compared to traditional CNNs and Vision Transformers. Different from previous works, we further equipped deformable convolution with multi-group spatial aggregation and transformer-like components for 3D medical image segmentation, while still improving computational efficiency.

# 3 METHOD

In this section, we first present our MGDC module and block design. To design a large-scale deformable CNN for medical image segmentation, we start by improving the original deformable convolution with multi-group mechanisms to improve feature encoding capabilities. We then design the basic block of MGDC by incorporating transformer components to stronger modeling capacity.

# 3.1 MULTI-GROUP DEFORMABLE CONVOLUTION

While traditional CNNs typically use small convolution kernels that result in limited effective receptive fields, deformable convolution enhances the conventional convolutional process by allowing for adaptive sampling positions within the convolutional grid. Unlike traditional convolution, which operates on uniformly spaced grid points, deformable convolution modifies these positions based on learnable offsets, thus enabling the model to learn more flexible representations of the input. Accordingly, we first take a 3D a dynamic deformable convolution network (3D DCN) (Zhu et al., 2019b) with adaptive sampling offsets and modulation masks to enhance the targeted segmentation tasks. Given an input  $x \in R^{H \times W \times D}$  and a current voxel  $v_{0}$ , our proposed 3D DCN layer can be

formulated in the following:

$$
y \left(v _ {0}\right) = \sum_ {s = 1} ^ {S} w _ {s} m _ {s} x \left(v _ {0} + v _ {s} + \Delta_ {v _ {s}}\right) \tag {1}
$$

where  $s$  enumerates the sampling points with a total of  $S$  points.  $v_{s}$  represents the  $s$ -th location of the pre-defined grid sampling  $\{(-1, -1, -1), (-1, -1, 0), \dots, (1, 1, 0), (1, 1, 1)\}$  as in regular  $3 \times 3 \times 3$  convolutions.  $\Delta v_{s}$  is the offset corresponding to the  $s$ -th sampling location,  $w_{s}$  denotes the projection weights of the  $s$ -th sampling point, and  $m_{s}$  is the modulation scalar of the  $s$ -th sampling point normalized by sigmoid function. From equation (1), we can see that the sampling offset  $\Delta v_{s}$  is conditioned based on inputs and is able to achieve both short and long-range dependencies. Furthermore, the modulation scalar  $m_{s}$  is also learnable and dynamically adjusted based on inputs. Therefore, the 3D DCN layer already shares similar properties with MHSA. Nonetheless, the proposed 3D DCN layer faces challenges in medical image segmentation. First, the design leads to linear memory complexity and computational demands, raising the risk of overfitting in data-limited medical settings. Second, unlike transformers or group convolutions, DCN lack a multi-group mechanism to capture diverse features, limiting their representational power.

To address these limitations, we introduce MGDC, a specialized deformable convolution operator. To remedy the computation complexity, we propose to use depth-wise convolution and detach the regular convolution  $w_{s}$  into depth-wise and point-wise parts. The depth-wise part is responsible for the location-aware modulation scalar  $m_{k}$  and the point-wise part is the shared projection weights  $w_{g}$  among sampling points. We also introduce multi-group spatial aggregation to effectively learn richer information from different representation subspaces at different locations. Similar to the concept of grouped convolution, we split the spatial aggregation process into  $G$  groups, each of which has individual sampling offsets  $\Delta p_{gs}$  and modulation scalar  $m_{gs}$  and hence different groups on a single convolution layer can have different spatial aggregation patterns, resulting in stronger features for downstream tasks.  $m_{gs}$  and  $\Delta v_{gs}$  are obtained via two linear layers applied over input. Given an input  $x$ , our proposed MGDC can be formulated as the following:

$$
x _ {1} = D W C (x) \tag {2}
$$

$$
\Delta v _ {g s} = \operatorname {l i n e a r} \left(x _ {1}\right) \tag {3}
$$

$$
m _ {g s} = \operatorname {s o f t m a x} \left(\operatorname {l i n e a r} \left(x _ {1}\right), S\right) \tag {4}
$$

$$
y \left(v _ {0}\right) = \sum_ {g = 1} ^ {G} \sum_ {s = 1} ^ {S} w _ {g} m _ {g s} x \left(v _ {0} + v _ {s} + \Delta v _ {g s}\right) \tag {5}
$$

where DWC stands for depth-wise convolution and linear stands for linear transformation.  $S$  stands for the total number of sampled points.  $G$  denotes the total number of aggregation groups. For the  $g$ -th group,  $w_{g}$  denotes the location-irrelevant projection weights of the group,  $w_{g} \in \mathbb{R}^{C_{g} \times C_{g}}$  where  $C_{g} = C / G$  represents the group channel dimension.  $m_{gs}$  denotes the modulation scalar of the  $s$ -th sampling point in the  $g$ -th group, normalized by the softmax function along dimension  $S$ .  $x_{g} \in \mathbb{R}^{C_{g} \times H \times W \times D}$  represents the  $g$ -th grouped input feature map.  $\Delta v_{gs}$  is the offset corresponding to the grid sampling location  $p_{s}$  in the  $g$ -th group. Since  $v_{0} + v_{s} + \Delta v_{gs}$  might be fractional, trilinear interpolation is used to convert fractions to integers.

# 3.2 MGDC-UNET

The overall pipeline of our proposed method is illustrated in Figure 2. Following the encoder-decoder design of Hatamizadeh et al. (2022), our MGDC-UNet consists of four stages in encoder, decoder, and four residual connections. For an input volume with a size of  $H \times W \times D$ , MGDC-UNet first leverages two convolution embedding layers to obtain downsampled feature maps of  $\frac{H}{4} \times \frac{W}{4} \times \frac{D}{4} \times C$ , where we set  $C$  empirically to 48. Next, each stage of encoding starts with MGDC blocks to extract spatial representations and ends with a downsample block (except for the last stage) to produce hierarchical features and double the channel dimension. After hierarchical encoding, the output from each stage in the encoder is fed to a CNN-based decoder with skip connections. Inside the decoder, a transposed convolutional layer is used for upsampling input and concatenating with

multi-scale features. On the final layer, we concatenated the transformed input with the upsampled features to produce the final segmentation map Below we show detailed design of the MGDC blocks.

1). MGDC Block: We present the MGDC Block, a new architecture that includes a reverse bottleneck design similar to MobileNetV2 (Sandler et al., 2018), but augmented with transformer components. While traditional inverted bottleneck design utilized depthwise convolution, our MGDC block leverages two MLP layers for channel expansion and reduction and LayerNorm for normalization, a design further inspired by Vision Transformers. This approach enables the network to capture more complex and richer features. Given input into the MLP layer  $m_{\mathrm{in}}$ , we define MLP function as:

$$
\mathrm {M L P} = \mathrm {L N} (\text {L i n e a r (G E L U (L i n e a r (m _ {i n}))))}) \tag {6}
$$

The overall block is formulated by the MLP layer with the GELU activation and post-normalization strategy as:

$$
x ^ {\prime} = x + \operatorname {G E L U} (\ln (\operatorname {M G D C} (x))) \tag {7}
$$

$$
x _ {\text {o u t}} = x ^ {\prime} + \ln (\operatorname {M L P} \left(x ^ {\prime}\right)) \tag {8}
$$

2). Stem block & downsample block: Hierarchical design downsamples the input to varying resolutions to extract multi-scale features and is commonly used in image segmentation. To obtain hierarchical feature maps, our stem block first reduces the input resolution by a factor of 4. We stack two plain convolution layers with a stride of 2, two Layer Normalization layers, and one GELU activation layer. The downsample block only reduces the input feature by a factor of 2. It consists of one plain convolution with a stride of 2, followed by one Layer Normalization layer.  
3). Upsample block & final block: To upsample the processed feature maps, we utilize transposed convolution with a stride of 2 (except for the last upsample block which uses a stride of 4), followed by Instance Normalization. An additional plain convolution layer is used to further extract semantic information from the decoded feature maps. In the final block, we swap the transposed convolution with plain convolution and output the segmentation maps.

![](images/c9e5d3d7180802f2608f1d371babe8a16c66588214b677c9830177dbb2c7f5eb.jpg)  
Figure 2: Illustration of Proposed MGDC-UNet Architecture. The complete encoder-decoder architecture is displayed on the left. Structures of MGDC block, stem block, downsample block and upsample block are revealed on the right.

# 4 RESULTS

# 4.1 IMPLEMENTATION DETAILS AND DATASET

To evaluate the proposed MGDC-UNet, we trained and evaluated the network in BraTS21, FLARE 2021, and AMOS 2022 dataset on an NVIDIA A6000. A comprehensive overview of datasets and

evaluation strategies can be found in Appendix A.1. The BraTS21 dataset for glioma segmentation includes 1,251 multi-parametric MRI scans with four modalities and evaluates using Dice score (DSC) and  $95\%$  Hausdorff distance (HD95). Annotations target three sub-regions: Gd-enhancing tumor (ET), peritumoral tissue (ED), and necrotic core (NCR). FLARE 2021 dataset for abdominal organ segmentation consists of 361 multi-contrast CT scans from two major medical centers and involves verification from five radiologists. For AMOS 2022, we focused on cross-modality CT-MRI segmentation using 300 CT and 60 MRI scans. Annotations were performed for 15 abdominal organs by multiple groups of radiologists. Both the Dice score and surface Dice score were computed for FLARE 2021 and AMOS 2022. A comprehensive overview of our training procedure can be found in Appendix A.2. In all experiments, the networks are optimized by the AdamW optimizer with a linear warmup and cosine annealing strategy. For the BraTS21 dataset, we opted for an input size of (128, 128, 128) following the methodology established by (Wang et al., 2021). On the other hand, for the AMOS and FLARE datasets, an input size of (96, 96, 96) was employed, as suggested by Lee et al. (2022). Several techniques including random rotation, random flipping, random cropping, random intensity shifts, and random affine transformations were deployed. Additionally, to fully demonstrate the capability of the DCN layer in handling large kernels for performance enhancement, we conducted experiments using various convolution kernel sizes (3, 5, and 7) to maximize MGDC's performance.

# 4.2 COMPARISONS WITH STATE-OF-THE-ART METHODS

To demonstrate the effectiveness of our proposed method, we compare it against state-of-the-art CNNs, transformers, and ConvNext methods on volumetric segmentation tasks. Our comparative methods include ResUNET (Zhang et al., 2018), SegResNet (Myronenko, 2019), Swin UNETR (Tang et al., 2022), TransBTS (Wang et al., 2021) and UXNET (Lee et al., 2022). We reimplemented the above methods according to the publicly released codes. To ensure the fairness of the comparison, we utilized the same optimization tool, data augmentation strategies, and data split for each method. We conducted five-fold cross-validation on each dataset respectively, and paired student's t-test was used to evaluate statistical significance.

1). Experiment results on BRaTS21 dataset: Table 1 presents a comparative analysis of MGDC-UNet with state-of-the-art segmentation techniques on the BraTS21 dataset. Notably, MGDC-UNet outperformed all competing methods, registering remarkable improvements in both the DSC and HD95. For a kernel size of 3, the MGDC-UNet achieved a DSC score of  $90.6\%$  and an HD95 value of  $4.816 \mathrm{~mm}$ , surpassing Swin UNETR by  $0.9\%$  and  $0.849 \mathrm{~mm}$ , respectively. Further investigation revealed consistent performance gains when incrementing the kernel size from 3 to 5 and ultimately to 7, corroborating our theory that larger receptive fields improve segmentation performance. A paired t-test provided additional statistical validation for the observed enhancements when increasing the kernel size from 3 to 7. For a deeper visual understanding, we refer the reader to Figure 3. As depicted in the first and second rows, our MGDC-UNet effectively minimizes false positive NCR (red) and ET (yellow) regions when segmenting brain tumors compared to competing methods. The third row also clearly illustrates MGDC-UNet's exceptional accuracy in outlining various tumor boundaries. Our observations further revealed that even with a small kernel size ( $k = 3$ ), our MGDC-UNet still excelled over the large-kernel ConvNext method, UXNET, by  $0.9\%$  in DSC. This indicates that deformable convolutions are capable of capturing long-range dependencies efficiently. Statistical validation reinforced the superior performance of MGDC-UNet over the best SOTA methods.

Furthermore, we provide the time efficiency and the memory usage of MGDC-UNet and comparison methods. For CNN methods, although ResUNet and SegResNet demonstrated fast training and inference time, their segmentation performances were much worse than our MGDC-UNet. For transformer methods, Swin UNETR outperformed TransBTS in segmentation accuracy but demonstrated lower training and inference speed. Compared to MGDC-UNet, both methods still have relatively high memory consumption. For the ConvNext method, UXNET demonstrated a good balance between performance and training speed. However, our MGDC-UNet  $k = 3$  is  $38\%$  faster and has  $19\%$  less memory consumption than UXNET while still improving DSC by  $1.3\%$ . Therefore, our model achieves the best balance between segmentation performance and time-resource efficiency.

2). Experiment results on FLARE21 dataset: As shown in Table 2, our MGDC-UNet outperformed all comparable methods in terms of DSC and SDC. Notably, MGDC-UNet outperformed

Table 1: Quantitative comparison with SOTA methods in BraTS21 dataset with Avg (average) results. The best result from SOTA methods is underlined. T-test is performed between the best result from SOTA models and our models. Bold means p-value  $\mathrm{p} < {0.05}$  . Efficiency analysis was also performed in terms of time (training or inference on each sample) and memory consumption for various models.  

<table><tr><td rowspan="2">Methods</td><td colspan="4">DSC</td><td colspan="4">HD95 (mm)</td><td colspan="2">Time (s)</td><td rowspan="2">Memory (G)</td></tr><tr><td>TC</td><td>WT</td><td>ET</td><td>Avg</td><td>TC</td><td>WT</td><td>ET</td><td>Avg</td><td>Train</td><td>Inference</td></tr><tr><td>ResUNET</td><td>0.875</td><td>0.912</td><td>0.858</td><td>0.881</td><td>7.740</td><td>12.446</td><td>6.542</td><td>8.912</td><td>0.25</td><td>0.37</td><td>2.5</td></tr><tr><td>SegResNet</td><td>0.901</td><td>0.917</td><td>0.867</td><td>0.895</td><td>6.481</td><td>10.421</td><td>5.478</td><td>7.460</td><td>0.24</td><td>0.78</td><td>3.3</td></tr><tr><td>UXNET</td><td>0.890</td><td>0.916</td><td>0.873</td><td>0.893</td><td>7.442</td><td>9.583</td><td>5.053</td><td>7.357</td><td>0.63</td><td>2.7</td><td>10.3</td></tr><tr><td>Swin UNETR</td><td>0.898</td><td>0.921</td><td>0.872</td><td>0.897</td><td>5.091</td><td>7.770</td><td>4.135</td><td>5.665</td><td>0.55</td><td>2.56</td><td>11.4</td></tr><tr><td>TransBTS</td><td>0.864</td><td>0.907</td><td>0.838</td><td>0.869</td><td>8.651</td><td>10.972</td><td>7.385</td><td>9.003</td><td>0.36</td><td>1.69</td><td>9.6</td></tr><tr><td>MGDC-UNet (k=3)</td><td>0.908</td><td>0.928</td><td>0.881</td><td>0.906</td><td>4.774</td><td>6.024</td><td>3.951</td><td>4.816</td><td>0.39</td><td>1.67</td><td>8.3</td></tr><tr><td>MGDC-UNet (k=5)</td><td>0.911</td><td>0.933</td><td>0.885</td><td>0.910</td><td>4.083</td><td>5.880</td><td>3.787</td><td>4.583</td><td>0.45</td><td>1.89</td><td>8.6</td></tr><tr><td>MGDC-UNet (k=7)</td><td>0.917</td><td>0.936</td><td>0.888</td><td>0.914</td><td>3.818</td><td>5.504</td><td>3.605</td><td>4.309</td><td>0.51</td><td>2.08</td><td>9.4</td></tr></table>

![](images/198f24310fb3a75d6af8ff14581ecdfddb830d08299737e8db37a7f9eb37e836.jpg)  
Figure 3: Visualization of segmentation results on BraTS21 dataset. Green, yellow and red regions indicate ED, ET and NCR.

UXNET (the previous state-of-the-art on Flare 21) by  $0.8\%$  in DSC and  $0.4\%$  in SDC. Experiments on enlarging the kernel size showed that MGDC-UNet achieved the best performance when  $k = 7$ , achieving  $94.4\%$  DSC and  $94.1\%$  SDC. We also generated visualization results in Figure 4. MGDC-UNet demonstrated the best segmentation performance for kidneys (row 2) and reduced false negative regions for liver segmentation (row 3).

3). Experiment results on AMOS22 dataset: Table 3 summarizes results on the AMOS 22 dataset. Our MGDC-UNet  $(k = 3)$  outperformed all comparable methods on CT segmentation tasks in both DSC and SDC. For MRI segmentation, both SegResNet and UXNET demonstrated similar performance to MGDC-UNet  $(k = 3)$  in terms of DSC. However, after switching kernel size to 7, MGDC-UNET outperformed both methods by  $0.3\%$  in DSC. For SDC, all MGDC-UNet models demonstrated superior performance, leading comparison methods by  $0.7\%$  to  $4.6\%$ . While cross-modal multi-organ segmentation still remained a challenge, our MGDC-UNet still achieved satisfactory performance for most organs (Figure 5, row two). In row one, we found that MGDC-UNet provided finer segmentation details of the stomach than other methods.

# 4.3 ABLATION STUDY

1). Effectiveness of MGDC operator: We started by investigating the effectiveness of our proposed MGDC operator. As shown in Table 4 (row 1 and 2), introducing a shared weight mechanism to MGDC decreased  $22\%$  parameters. Our MGDC introduced shared weights to alleviate the high computational costs and reduce memory consumption by  $33\%$ . We also observed a small performance boost after switching from 3D DCN to MGDC. Next, we compared the MGDC with and

Table 2: Quantitative comparison with SOTA methods in FLARE21 dataset with Avg (average) results. The best result from SOTA methods is underlined. T-test is performed between the best result from SOTA models and our models. Bold means p-value  $p < 0.05$ .  

<table><tr><td rowspan="2">Methods</td><td colspan="5">DSC</td><td colspan="5">SDC</td></tr><tr><td>Spleen</td><td>Kidney</td><td>Liver</td><td>Pancreas</td><td>Avg</td><td>Spleen</td><td>Kidney</td><td>Liver</td><td>Pancreas</td><td>Avg</td></tr><tr><td>ResUNET</td><td>0.976</td><td>0.955</td><td>0.968</td><td>0.774</td><td>0.918</td><td>0.957</td><td>0.958</td><td>0.986</td><td>0.726</td><td>0.907</td></tr><tr><td>SegResNet</td><td>0.976</td><td>0.956</td><td>0.969</td><td>0.816</td><td>0.929</td><td>0.966</td><td>0.965</td><td>0.992</td><td>0.799</td><td>0.930</td></tr><tr><td>UXNET</td><td>0.977</td><td>0.959</td><td>0.973</td><td>0.819</td><td>0.932</td><td>0.966</td><td>0.967</td><td>0.994</td><td>0.810</td><td>0.934</td></tr><tr><td>Swin UNETR</td><td>0.978</td><td>0.959</td><td>0.971</td><td>0.803</td><td>0.928</td><td>0.965</td><td>0.963</td><td>0.986</td><td>0.782</td><td>0.924</td></tr><tr><td>TransBTS</td><td>0.978</td><td>0.959</td><td>0.971</td><td>0.764</td><td>0.918</td><td>0.968</td><td>0.966</td><td>0.991</td><td>0.719</td><td>0.911</td></tr><tr><td>MGDC-UNet (k=3)</td><td>0.982</td><td>0.963</td><td>0.972</td><td>0.842</td><td>0.940</td><td>0.973</td><td>0.967</td><td>0.992</td><td>0.819</td><td>0.938</td></tr><tr><td>MGDC-UNet (k=5)</td><td>0.992</td><td>0.965</td><td>0.967</td><td>0.840</td><td>0.941</td><td>0.973</td><td>0.968</td><td>0.994</td><td>0.823</td><td>0.940</td></tr><tr><td>MGDC-UNet (k=7)</td><td>0.995</td><td>0.968</td><td>0.971</td><td>0.843</td><td>0.944</td><td>0.975</td><td>0.969</td><td>0.994</td><td>0.825</td><td>0.941</td></tr></table>

![](images/7aba7688a8ce6e802b47fe37ebbbd38d94e5e3af112fda5008de2c256dd5d651.jpg)

![](images/bc89285ab819434ad251f0f78a23a049fcff69625388f21604043bd0a334738d.jpg)

![](images/60957ea1ae36e753760cbcaa9aa2256c4ba3d0eee309ae54947ce603b00bfb0d.jpg)  
a). Ground Truth

![](images/270513e152ff792c414b05be676254b85605eb29194822666dec568f1ffc4aa2.jpg)

![](images/6aa537b841ccecc85652d81e57f38b181e0b70a76e60fde579cf154cac4fc2fe.jpg)

![](images/f629a7cf86f3d42e956967dd0a63cb875b39bdc11982e68a903927a62124ea59.jpg)  
b).ResUNet

![](images/8d9eef91a2c0045fed1a2f95f8a412eff00bccddb44eab60b6457730cdadaf6a.jpg)

![](images/fc013d677778263950705a7fd47d81a5b7ce92035b9150093749ff03c2415eea.jpg)

![](images/77b8e7c9f04e86a4f48c18c0c98db1c51d385436c891a3cdac33f769285c832b.jpg)  
c).SegResNet

![](images/d524c8b5bbce8a8924a0dbf7e5619334f4ea078d7ce715c7c7df683ed00a9233.jpg)

![](images/adf16145e34d045f1145a641756d3d0cd094866bedb467ce4ba8b5caee0a8100.jpg)

![](images/f5c4952443ba0adf0a486efc126417e50710da33418e0d6105556243801b61e3.jpg)  
d). Swin UNETR

![](images/4243896a2b411f98694e6ca1fb30b72e0bfce676fab77b9bdc6a77bfad987f36.jpg)

![](images/79d273cd96f312b77cd204155c5df7c0106f3c809d7ddac0bf2df1a7b497a13f.jpg)

![](images/1cb39670e36a9d176a6137fdff4fb6942aebf02f00aa74eb696bb113a1a12a30.jpg)  
e). TransBTS

![](images/91ad6d8fd180e516a2701aee2356ff4dbb007ba00bad4d674be9463ddf0d6983.jpg)

![](images/c7254a3815feee89a641d5663bf33a12bb13085f29bddd962e678d8d83fedccf.jpg)

![](images/d90b5f3cbc5e13b6d1eb858d29bbb343518d400f86b6926f8cfb666c3db4593a.jpg)  
f).UXNET

![](images/1faf886f419284e8765dc1a9113e4d5e0a3bf9cd5dd14bd82a3b6fa338fa3b26.jpg)

![](images/2aceffcd68532e9f201b5972ee398ac6030a743f36084b965c77b35f859286d1.jpg)

![](images/9f5beb2940c7129ce9cf5addd1a55040f2df0aa4a0a482732fb379cfcecc456e.jpg)  
g). MGDC-UNet (ours)

![](images/d24cd27f7f04249ea9b791f1eff522e82514f2e377ddc05d05f75a2b7868573b.jpg)  
Figure 4: Visualization of segmentation results on FLARE21 dataset. White arrow indicates superior regions of our results compared with other models

![](images/0d5d4a2439cbe11f8a7642d6ce0fed2a851cc5e80fe040ccdc48e095d0e55c25.jpg)

![](images/aade2107fde319aa4a2f9aee23c82a53909f5b10d3e2beac7cc2ad0f87638b25.jpg)  
a). Ground Truth

![](images/2bca2de9efaebb73e29cd68bc53869ef23dacf40d8dfcd8d08f5d3958feb9412.jpg)

![](images/742700670a2b4bd6ea7876355381cc84b0ca643c3098c2baf663dc6d53bca4ab.jpg)

![](images/b6158941f0930825ec15b85b35afa236f26073e7779cbeffd8d8d9490c245114.jpg)  
b). ResUNet

![](images/179989c3fe3bfea71f3aa0bab885bfcd50eb3a9efb458826843a14ac11025aff.jpg)

![](images/21f5b029c984fc4f6e79de5269c5ba0e0b2cc23d243807e9f6edd83a0a4da126.jpg)

![](images/880d044d7d96fac87c9604c3bdfee778cf15acab0d99959ec7ddc56fd98813fc.jpg)  
c).SegResNet

![](images/45fa34e4695d6f7137bb39f485361dc0dff9725d024765c17e966bba37c627d9.jpg)

![](images/e586dd99da804d9a7b2f65a3d12971150180394e2f4cfe5524fb622ca679f064.jpg)

![](images/a021af6cf0f0303f4a9c7d933fe4d829725c6e523f0b0be73ab181865dd84428.jpg)  
d). Swin UNETR

![](images/c5a4212b86daac24764e410c4f69fda741c2f861cfc69f0e578791a2a1f91e5c.jpg)

![](images/226ef296620e1b037c14825f5913e8d831065f2e1122335f47c9e64476ed444a.jpg)

![](images/8859b69a8838b08283c8f219fc854362025f0002c3db2dd19b00a7ed8cd16261.jpg)  
Figure 5: Visualization of segmentation results on AMOS22 dataset. White arrow indicates superior regions of our results compared with other models  
e). TransBTS

![](images/129120b967327a9243b22f79effcff737f4628da3f253c0b5d5d04c694466c19.jpg)

![](images/691aaf7179634cf3c67ba1078d4030e4ab3f42d732c53836db8be95c34e61eb7.jpg)

![](images/7de24bd994df69c2ccfd7b34f37994f1b613a697c491df8bdb633a272872e39e.jpg)  
f).UXNET

![](images/bb53130e8f7975cd0cb73b388ac1c92d83195a1a57ca84bf1c03790f5cd9a88e.jpg)

![](images/c7e86af75f5da937f9b9feb77e766955bd6a5d8e8e1af939cdbb0158fa43eab7.jpg)

![](images/707704466a3e106e01521a2ecd48a895b53dccf7b2a3e1fda194b75f7a4e79f2.jpg)  
g). MGDC-UNet (ours)

without the multi-group spatial aggregation. As shown in row 2 and row 3, introducing a multi-group mechanism into deformable convolution improved DSC by  $0.5\%$  for brain tumor segmentation and  $0.4\%$  for multi-organ segmentation. We suspected that a larger training sample size would further improve the performance gains of the larger kernel convolution method. In this section, we study how the different components in our designed MGDC-UNet contribute to gains in segmenta

Table 3: Quantitative comparison with SOTA methods in AMOS22 dataset with Avg (average) results. The best result from SOTA methods is underlined. T-test is performed between the best result from SOTA models and our models. Bold means p-value  $p < 0.05$ .  

<table><tr><td rowspan="2">Methods</td><td colspan="3">DSC</td><td colspan="3">SDC</td></tr><tr><td>CT</td><td>MRI</td><td>Avg</td><td>CT</td><td>MRI</td><td>Avg</td></tr><tr><td>ResUNET</td><td>0.825</td><td>0.706</td><td>0.805</td><td>0.840</td><td>0.823</td><td>0.846</td></tr><tr><td>SegResNet</td><td>0.854</td><td>0.720</td><td>0.830</td><td>0.888</td><td>0.867</td><td>0.885</td></tr><tr><td>UXNET</td><td>0.856</td><td>0.720</td><td>0.833</td><td>0.886</td><td>0.860</td><td>0.882</td></tr><tr><td>Swin UNETR</td><td>0.851</td><td>0.712</td><td>0.828</td><td>0.876</td><td>0.862</td><td>0.874</td></tr><tr><td>TransBTS</td><td>0.847</td><td>0.717</td><td>0.826</td><td>0.877</td><td>0.858</td><td>0.873</td></tr><tr><td>MGDC-UNet (k=3)</td><td>0.865</td><td>0.720</td><td>0.840</td><td>0.893</td><td>0.884</td><td>0.891</td></tr><tr><td>MGDC-UNet (k=5)</td><td>0.865</td><td>0.721</td><td>0.841</td><td>0.894</td><td>0.886</td><td>0.892</td></tr><tr><td>MGDC-UNet (k=7)</td><td>0.866</td><td>0.723</td><td>0.841</td><td>0.894</td><td>0.885</td><td>0.892</td></tr></table>

Table 4: Ablation on component of MGDC-UNet. Network parameters and DSC from BraTS21 and FLARE21 were reported.  

<table><tr><td>Operator</td><td>Multi-group</td><td>MLP</td><td>Params (M)</td><td>BraTS21</td><td>FLARE21</td></tr><tr><td>3D DCN</td><td>×</td><td>×</td><td>71.5</td><td>0.894</td><td>0.929</td></tr><tr><td>MGDC</td><td>×</td><td>×</td><td>58.7</td><td>0.896</td><td>0.930</td></tr><tr><td>MGDC</td><td>✓</td><td>×</td><td>58.1</td><td>0.901</td><td>0.932</td></tr><tr><td>MGDC</td><td>✓</td><td>✓</td><td>61.2</td><td>0.906</td><td>0.940</td></tr></table>

tion performance. We conducted ablation studies on BraTS21 and Flare21 datasets due to their large sample sizes. All ablation studies on MGDC-UNet were performed with kernel size set to 3.

2). Effectiveness of MGDC Block: The core design of our MGDC Block is introducing a multi-layer perceptron as a feed-forward network. As shown in Table 4 (row 3 and 4), introducing MLP layers to the network successfully scaled up the model and further improved segmentation performance by  $0.5\%$  and  $0.8\%$  in DSC on BraTS21 and FLARE21 datasets. This also confirmed our hypothesis that transformer-like components can also enhance medical image segmentation.

# 5 CONCLUSION AND DISCUSSION

In this paper, we introduce MGDC-UNet, the first 3D multi-group deformable convolution network for medical image segmentation. Our architecture integrates multi-group spatial aggregation into deformable convolutions, inspired by the multi-head mechanism found in ViTs. Additionally, we incorporate transformer-specific elements such as MLP and LayerNorm to emulate the inverted-bottleneck design featured in ViT blocks. To further enhance performance, we explore the use of large deformable convolutional kernels, which further improve the network's capability for capturing long-range dependencies—crucial for achieving high-quality segmentation results. Our rigorous evaluation clearly demonstrates MGDC-UNet's advantages through both quantitative and statistical metrics, establishing its superiority over existing methods. MGDC-UNet excels in capturing long-range dependencies, a feat attributed mainly to its flexible offsets and modulation scalars. This distinctive feature sets our model apart from traditional CNNs, which frequently struggle with global attention, a limitation we overcome as demonstrated in Figure 1. When compared to transformer-based architectures, MGDC-UNet offers dual benefits: it not only learns more robust representations but also achieves this with fewer model parameters. Thus, MGDC-UNet emerges as a resilient solution, less prone to overfitting while maintaining higher computational efficiency. In summary, MGDC-UNet surpasses the state-of-the-art transformer models in performance with less memory usage and better performance speed across three challenging public datasets. We believe that MGDC-UNet holds significant potential as a tool for fast organ delineation in clinical applications.

# REFERENCES

Lisa C Adams, Marcus R Makowski, Gunther Engel, Maximilian Rattunde, Felix Busch, Patrick Asbach, Stefan M Niehues, Shankeeth Vinayahalingam, Bram van Ginneken, Geert Litjens, et al. Prostate158-an expert-annotated 3t mri dataset and algorithm for prostate cancer detection. Computers in Biology and Medicine, 148:105817, 2022.  
Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In Proceedings of the IEEE international conference on computer vision, pp. 764-773, 2017.  
Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath, Dong Yang, Andriy Myronenko, Bennett Landman, Holger R Roth, and Daguang Xu. Unetr: Transformers for 3d medical image segmentation. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 574-584, 2022.  
Mattias P Heinrich, Ozan Oktay, and Nassim Bouteldja. Obelisk-net: Fewer layers to solve 3d multi-organ segmentation with sparse deformable convolutions. Medical image analysis, 54:1-9, 2019.  
Qiangguo Jin, Zhaopeng Meng, Tuan D Pham, Qi Chen, Leyi Wei, and Ran Su. Dunet: A deformable network for retinal vessel segmentation. Knowledge-Based Systems, 178:149-162, 2019.  
Ho Hin Lee, Shunxing Bao, Yuankai Huo, and Bennett A Landman. 3d ux-net: A large kernel volumetric convnet modernizing hierarchical transformer for medical image segmentation. arXiv preprint arXiv:2209.15076, 2022.  
Andriy Myronenko. 3d mri brain tumor segmentation using autoencoder regularization. In *Brain-lesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries: 4th International Workshop*, *BrainLes* 2018, Held in Conjunction with MICCAI* 2018, Granada, Spain, September 16, 2018, Revised Selected Papers, Part II 4, pp. 311–320. Springer, 2019.  
Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori, Steven McDonagh, Nils Y Hammerla, Bernhard Kainz, et al. Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999, 2018.  
Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention-MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pp. 234-241. Springer, 2015.  
Anindo Saha, Matin Hosseinzadeh, and Henkjan Huisman. End-to-end prostate cancer detection in bpmri via 3d cnns: effects of attention mechanisms, clinical priori and decoupled false positive reduction. Medical image analysis, 73:102155, 2021.  
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. *Mobilenetv2: Inverted residuals and linear bottlenecks*. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 4510-4520, 2018.  
Ashish Sinha and Jose Dolz. Multi-scale self-guided attention for medical image segmentation. IEEE journal of biomedical and health informatics, 25(1):121-130, 2020.  
Yucheng Tang, Dong Yang, Wenqi Li, Holger R Roth, Bennett Landman, Daguang Xu, Vishwesh Nath, and Ali Hatamizadeh. Self-supervised pre-training of swin transformers for 3d medical image analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20730-20740, 2022.  
Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, et al. Internimage: Exploring large-scale vision foundation models with deformable convolutions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14408-14419, 2023.

Wenxuan Wang, Chen Chen, Meng Ding, Hong Yu, Sen Zha, and Jiangyun Li. Transbts: Multimodal brain tumor segmentation using transformer. In Medical Image Computing and Computer Assisted Intervention-MICCAI 2021: 24th International Conference, Strasbourg, France, September 27-October 1, 2021, Proceedings, Part I 24, pp. 109-119. Springer, 2021.  
Qiao Zhang, Zhipeng Cui, Xiaoguang Niu, Shijie Geng, and Yu Qiao. Image segmentation with pyramid dilated convolution based on resnet and u-net. In Neural Information Processing: 24th International Conference, ICONIP 2017, Guangzhou, China, November 14-18, 2017, Proceedings, Part II 24, pp. 364-372. Springer, 2017.  
Zhengxin Zhang, Qingjie Liu, and Yunhong Wang. Road extraction by deep residual u-net. IEEE Geoscience and Remote Sensing Letters, 15(5):749-753, 2018.  
Qikui Zhu, Bo Du, and Pingkun Yan. Boundary-weighted domain adaptive neural network for prostate mr image segmentation. IEEE transactions on medical imaging, 39(3):753-763, 2019a.  
Xizhou Zhu, Han Hu, Stephen Lin, and Jifeng Dai. Deformable convnets v2: More deformable, better results. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9308-9316, 2019b.
