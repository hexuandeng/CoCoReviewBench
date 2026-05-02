# DH-Fusion: Depth-Aware Hybrid Feature Fusion for Multimodal 3D Object Detection

Anonymous Author(s)

Affiliation

Address

email

# Abstract

State-of-the-art LiDAR-camera 3D object detectors usually focus on feature fusion. However, they neglect the factor of depth while designing the fusion strategy. In this work, we for the first time point out that different modalities play different roles as depth varies via statistical analysis and visualization. Based on this finding, we propose a Depth-Aware Hybrid Feature Fusion (DH-Fusion) strategy that guides the weights of point cloud and RGB image modalities by introducing depth encoding at both global and local levels. Specifically, the Depth-Aware Global Feature Fusion (DGF) module adaptively adjusts the weights of image Bird's-Eye-View (BEV) features in multi-modal global features via depth encoding. Furthermore, to compensate for the information lost when transferring raw features to the BEV space, we propose a Depth-Aware Local Feature Fusion (DLF) module, which adaptively adjusts the weights of original voxel features and multi-view image features in multi-modal local features via depth encoding. Extensive experiments on the nuScenes dataset demonstrate that our DH-Fusion method surpasses previous state-of-the-art methods w.r.t. NDS. Moreover, our DH-Fusion is more robust to various kinds of corruptions, outperforming previous methods on nuScenes-C w.r.t. both NDS and mAP.

# 1 Introduction

3D object detection has a wide range of applications in the fields of autonomous driving and robotics. A large number of previous works have successfully focused on using a single modality, such as point cloud or images, to design efficient 3D object detectors. However, the performance of these detectors reaches a bottleneck due to the limitations of modality characteristics. For instance, the point cloud modality can only provide rich geometric information while lacks detailed semantic information; the image modality can only provide rich texture information while lacks three-dimensional spatial information. To address the aforementioned issues, we are highly motivated to obtain comprehensive information that represents objects by designing a LiDAR-camera 3D object detector.

In recent years, LiDAR-camera 3D object detection develops rapidly. Some works [1, 4, 28, 33, 67] propose effective methods to integrate information from two modalities at the feature level. However, they all overlook an important factor of depth in their fusion strategies. To understand how point cloud and image information vary with depth, we first conduct statistical and visualization analysis on the nuScenes-mini dataset [3], and find that: (1) The number of points representing objects at near range is relatively large, which allows us to accurately determine the object's location, size, and category, even without the aid of images. As shown in Fig. 1a, there is an average of 163.7 points per object within 0-10 meters, which is a substantial number. We also visualize a car at 6.8 meters in Fig. 1b  $①$  and find it encompasses a considerable number of points, well representing the shape. In contrast, some background noise in the image may interfere with detection (Fig. 1b  $②$ ). (2) As the

![](images/d910a1b47010438b25255b69863a38bcf96eb73f0cdf0f55bd7929dc36825c74.jpg)  
(a) Statistical chart

![](images/5fbd775befbe5f89f948775eaec2a1adb18eb8b3620e3222d6f375016094f7fc.jpg)  
Figure 1: Statistical and visualization analysis on the nuScenes-mini dataset. (a) The average numbers of points and pixels for each object at different depths. (b) Examples of near-range and long-range objects in images and point cloud. Points within the bounding boxes are colored red for observation.  
(b) Visualization

depth increases, the number of points representing objects decreases rapidly. As shown in Fig. 1a, the number of points within 30-50 meters falls below one per object, meaning that many objects are even not represented by any points, such as the object at 42.1 meters in Fig. 1b  $③$ . In contrast, the complete objects may still be observed on the image, as in Fig. 1b  $④$ , where the image information becomes more important. To address the above problems, we propose a feature fusion strategy that adaptively adjusts the importance of the two modalities based on depth.

Specifically, we propose a novel method for multi-modal 3D object detection, namely Depth-Aware Hybrid Feature Fusion (DH-Fusion). The innovation lies in adaptively adjusting the weights of features by introducing depth encoding to hybrid feature fusion at both global and local levels. The fusion strategy consists of two crucial components: Depth-Aware Global Feature Fusion (DGF) module and Depth-Aware Local Feature Fusion (DLF) module. In DGF, we take point cloud Bird's-Eye-View (BEV) features and image BEV features as inputs, and dynamically adjust the weights of image BEV features based on depth during fusion by utilizing a global-fusion transformer encoder with a depth encoder. To compensate for the information lost when transforming raw features to BEV space, we enhance the fused BEV features at a lower cost by utilizing the original instance features. In DLF, we obtain 3D boxes by utilizing a Region Proposal Network (RPN). Then, the 3D boxes are projected into both LiDAR voxel features and multi-view image features to crop out corresponding local instance features with more detailed information. Afterward, we take these as inputs and dynamically adjust the weights of local multi-view image features and local LiDAR voxel features based on depth through the use of a local-fusion transformer encoder with the depth encoder. In the end, we update local features for each object on the global feature map to enhance the detailed instance information of multi-modal global features for detection.

Our contributions are summarized as follows.

1. We for the first time point out that depth is an important factor to consider while fusing LiDAR point cloud features and RGB image features for 3D object detection. From our statistical and visualization analysis, we can see that image features play different roles as depth varies.  
2. We propose a depth-aware hybrid feature fusion strategy that dynamically adjusts the weights of features during feature fusion by introducing depth encoding at both global and local levels. The above strategy can obtain high-quality features for detection, fully leveraging the advantages of different modalities at various depths.  
3. Our method is evaluated on the nuScenes [3] dataset and a more challenging nuScenes-C [13] dataset, outperforming previous multi-modal methods and being robust to various kinds of data corruptions.

# 2 Related Work

Since our method is based on conducting 3D object detection using data from multiple modalities, including point cloud and images, we briefly review recent works in the following fields: LiDAR-based 3D object detection, camera-based 3D object detection, and LiDAR-camera 3D object detection.

# 2.1 LiDAR-based 3D Object Detection

LiDAR-based 3D object detectors only take the point cloud as input. Based on their different data representations, they can be divided into point-based [44-46, 64, 65], voxel-based [12, 22, 61, 68, 71], and point-voxel-based [17, 42, 43] methods. The feature extraction networks of point-based methods typically extract features directly from the point cloud through a point-based backbone [40], such as PointRCNN [44]. The voxel-based methods first convert the point cloud into voxels and then extract voxel features through a 3D sparse convolution network [14], such as VoxelNet [71]. Point-voxel-based methods like PV-RCNN [42] combine the above two methods to extract and fuse point and voxel features. The purpose of these approaches is to capture the geometric spatial information of the point cloud. However, point cloud is sparse and incomplete, lacking detailed texture information, which greatly limits the detection performance.

# 2.2 Camera-based 3D Object Detection

Camera-based 3D object detectors only take images as inputs. Depending on the form of inputs, they can be divided into monocular [2, 24, 32, 41, 47, 55], stereo [6, 25, 30, 48, 70], and multi-view [19, 27, 56, 62] 3D object detectors. Early works like FCOS3D [55] input a monocular image and utilize 2D object detectors to directly predict 3D bounding boxes, but these approaches have limited capability in capturing spatial information. Subsequently, stereo and multi-view 3D object detectors are proposed to obtain more precise depth information by constructing spatial relationships among multiple images, such as Stereo RCNN [25] and BEVDet [19]. These methods successfully achieve purely visual 3D object detection, but they do not perform as well as LiDAR-based methods, because the spatial depth information provided by images is not as direct and precise as that provided by point cloud.

# 2.3 LiDAR-Camera 3D Object Detection

LiDAR-camera 3D object detectors take point cloud and images as inputs, and can be classified into early-fusion-based [50, 52, 57, 59, 69], intermediate-fusion-based [1, 4, 28, 33, 67], and late-fusion-based [37, 38] 3D object detectors based on the location of multi-modal information fusion [36].

Early-fusion-based methods perform at the point level, where the typical approach involves enhancing the raw point cloud with semantic information extracted from images. PointPainting [50] and FusionPainting [59] decorate the raw point cloud with semantic scores from 2D semantic segmentation. Similarly, PointAugmenting [52] enhances the raw point cloud using features extracted from a 2D semantic segmentation network. However, early-fusion-based methods are sensitive to alignment errors between the two modalities.

Intermediate-fusion-based methods perform at the feature level. Transfusion [1] first proposes to utilize the transformer for fine-grained fusion from LiDAR BEV features and multi-view image features. FUTR3D [5] encode each modality using deformable attention [73] in its own coordinate and concatenate them for fusion. BEVFusion [28, 33] projects both point cloud and images to BEV space for BEV feature fusion. SparseFusion [58] extracts instance-level features from both two modalities separately, and fuse them to perform detection. Similarly, ObjectFusion [4] utilizes 3D proposals from LiDAR modality to extract instance-level features for fusion. CMT [60] proposes the simultaneous interaction between the object queries and multi-modal features in the transformer encoder and decoder. IS-Fusion [67] proposes feature fusion at both the instance level and scene level. The intermediate-fusion-based methods gradually become a mainstream approach due to the diversity of fusion strategies.

Late-fusion-based methods perform at the bounding box level. Typically, CLOCs [37] obtains 2D and 3D bounding boxes by separately using 2D and 3D object detectors, and then combine them to achieve more accurate 3D bounding boxes. However, the interaction between modalities in late-fusion-based methods is very limited, which constrains model performance.

These multi-modal methods successfully outperform single-modal methods. However, their feature fusion methods do not take depth into account. In contrast, our approach introduces depth information to guide the hybrid feature fusion, boosting the performance of the detector.

![](images/a8955c566bb1c1691040c16a2fea3db8f2552fbd5dad5b502ac372b87bd5457d.jpg)  
Figure 2: Overview of our method. It introduces depth encoding in both global and local feature fusion to obtain depth-adaptive multi-modal representations for detection.  $\otimes$  is the multiplication operation, and  $\mathbb{M}$  is the merge operation.

# 3 Methodology

In this section, we first give an overview of our proposed multi-modal 3D object detector, and then provide a detailed introduction to our proposed feature fusion method.

# 3.1 Overview

We propose a multi-modal 3D object detection method via Depth-Aware Hybrid Feature Fusion (DH-Fusion). As illustrated in Fig. 2, our approach consists of two important feature fusion modules: Depth-Aware Global Feature Fusion (DGF) and Depth-Aware Local Feature Fusion (DLF). In the following, we briefly describe the detection pipeline.

Inputs. First, we take the point cloud  $P$  and multi-view images  $I$  as inputs, where point cloud consists of a set of points:  $P = \{P_{1}, P_{2}, \dots, P_{N_{l}}\}$ , and each point has four dimensions: X-axis, Y-axis, Z-axis, and intensity; the multi-view images comprise  $N_{c}$  images:  $I = \{I_{1}, I_{2}, \dots, I_{N_{c}}\}$ , each image captured by its corresponding camera.

Input Encoding. For the point cloud  $P$ , we use a 3D encoder to extract raw global voxel features  $\mathcal{V}_O^G$ ; for the multi-view images  $I$ , we use a 2D encoder to extract image features of all views  $\mathcal{T}_O^G$ .

Hybrid Feature Fusion. Then, for voxel features  $\mathcal{V}_O^G$ , we compress the height dimension to obtain point cloud BEV features  $\mathcal{V}_B^G$ ; for image features  $\mathcal{T}_O^G$ , we transform their perspective view to bird's eye view to obtain image BEV features  $\mathcal{T}_B^G$ . To fully leverage the features from two modalities, we design a DGF module that aims to dynamically adjust the weights of image BEV features based on depth values during feature fusion. Please refer to Sec. 3.2 for more details. To compensate for the information lost when transforming raw features to BEV space, we propose a DLF module that, based on depth, utilizes the raw features to enhance the detailed information of each object instance in global multi-modal features. It consists of three processes: local feature selection, local feature fusion, and merging local features into global features. First, we obtain the local multi-modal BEV features  $\mathcal{F}_B^L$ , local voxel features  $\mathcal{V}_O^L$ , and local multi-view image features  $\mathcal{T}_O^L$ , by cropping the corresponding global features based on the 3D boxes obtained from an RPN; then, it dynamically and individually adjusts the weights of each local feature of  $\mathcal{V}_O^L$  and  $\mathcal{T}_O^L$  based on depth values during feature fusion; finally, we update local features for each object on the global feature map. Please refer to Sec. 3.3 for more details. In this way, we obtain enhanced multi-modal global features for detection.

Decoding. Based on the enhanced multi-modal global features  $\hat{\mathcal{F}}_B^G$  that contain rich semantic and spatial information, we utilize a transformer decoder and a detection head to predict the object categories and 3D bounding boxes.

![](images/f5c850d3422b8ad73e3943463a03006a1b7706239a88dadcd21c3926792a0486.jpg)  
Figure 3: Illustration of the DGF. It consists of a global fusion transformer with the depth encoder.

![](images/fdb2f7748925323cc59e88dcc248018d023c53ad718dc444e84d82dc25e5acba.jpg)  
Figure 4: Illustration of the DLF. It consists of a local feature selection module and a local fusion transformer with the depth encoder.

# 3.2 Depth-Aware Global Feature Fusion

As shown in Fig. 3, the DGF module consists of a global-fusion transformer with a depth encoder. In the following, we provide a detailed explanation of each component.

# 3.2.1 Depth Encoder

We introduce depth encoding (DE) in feature fusion to dynamically adjust the weights of image BEV features during fusion. First, we build a depth matrix  $M$  to store the depth value of each position element  $p_k$  represented as:

$$
p _ {k} = \left\{\left(x _ {k}, y _ {k}\right): d _ {k} \right\}, k \in [ 1, n ], \tag {1}
$$

where  $(x_{k},y_{k})$  are the positional coordinates,  $d_{k}$  is the depth value, and  $n$  is the number of elements. Then, we use Euclidean distance to calculate the distance between every element's spatial location  $(x_{k},y_{k})$  and the ego coordinate element's location  $(x_{\frac{n}{2}},y_{\frac{n}{2}})$ :

$$
d _ {k} = E \left(\left(x _ {k}, y _ {k}\right), \left(x _ {\frac {n}{2}}, y _ {\frac {n}{2}}\right)\right), k \in [ 1, n ], \tag {2}
$$

where we denote  $E(\cdot)$  as the Euclidean distance calculation. The depth matrix  $M$  serves as a lookup table to avoid redundant computation of depth values. Since the size of the BEV features is large and the depth distribution is simple, to avoid introducing additional parameters, the depth encoding  $De$  is obtained by applying sine and cosine functions [49] to the depth matrix.

# 3.2.2 Global-Fusion Transformer

In the global-fusion transformer, we take the point cloud BEV features  $\mathcal{V}_B^G \in \mathbb{R}^{W \times H \times C}$  and image BEV features  $\mathcal{I}_B^G \in \mathbb{R}^{W \times H \times C}$  as inputs, and integrate the depth encoding obtained above by multiplying it with the point cloud BEV features, forming the query  $Q_{\mathcal{V}}^G = N(\mathcal{V}_B^G \times \text{Conv}(De))$ , where  $\text{Conv}(\cdot)$  is a convolution operation to align with the channels of  $\mathcal{V}_B^G$ , and  $N(\cdot)$  is a normalization layer. The image BEV features are queried as the corresponding key  $K_{\mathcal{I}}^G$  and value  $V_{\mathcal{I}}^G$ . We utilize the multi-head cross attention to achieve the interacted feature  $\hat{\mathcal{V}}_B^G$  based on depth:

$$
\hat {\mathcal {V}} _ {B} ^ {G} = C A \left(Q _ {\mathcal {V}} ^ {G}, K _ {\mathcal {I}} ^ {G}, V _ {\mathcal {I}} ^ {G}\right), \tag {3}
$$

where  $CA(\cdot)$  indicates the multi-head cross attention. Afterward, we aggregate the information from both modalities to obtain the fused features  $\mathcal{F}_B^G$ :

$$
\mathcal {F} _ {B} ^ {G} = N (F F N (N (\hat {\mathcal {V}} _ {B} ^ {G} + \mathcal {V} _ {B} ^ {G})) + N (\hat {\mathcal {V}} _ {B} ^ {G} + \mathcal {V} _ {B} ^ {G})), \tag {4}
$$

where  $N(\cdot)$  is a normalization layer;  $FFN(\cdot)$  specifies a feed-forward network containing two convolution operations. In this way, we obtain fused features in which the image features play different roles as the depth varies.

# 3.3 Depth-Aware Local Feature Fusion

As shown in Fig. 4, the DLF module consists of a local feature selection and a local-fusion transformer with the depth encoder. In the following, we provide a detailed explanation of each component.

# 3.3.1 Local Feature Selection

To compensate for the information lost when transforming point cloud features and image features to BEV space, we enhance the instance details of fused BEV features  $\mathcal{F}_B^G$  using instance features from raw voxel features  $\nu_{O}^{G}$  and multi-view image features  $\mathcal{I}_O^G$ . Specifically, we utilize an RPN to regress  $t$  3D boxes based on the BEV features  $\mathcal{F}_B^G$ . We directly crop the global fused BEV features  $\mathcal{F}_B^G$  based on the regressed 3D boxes to obtain the local fused BEV features  $\mathcal{F}_B^L \in \mathbb{R}^{c\times t}$ . On the other hand, we project the 3D boxes onto the raw voxel features and multi-view image features to obtain their corresponding local features before global fusion, preserving richer information for each object instance. Specifically, we utilize the voxel pooling operation [12], followed by a 3D convolution operation and a linear layer, to extract local voxel features  $\nu_{O}^{L} \in \mathbb{R}^{c\times t}$ ; we transform the 3D boxes from bird's eye view to perspective view, and utilize the RoI Align operation [15], followed by a linear layer, to extract instance image features  $\mathcal{I}_O^L \in \mathbb{R}^{c\times t}$ . By doing this, we obtain the hybrid (before & after global fusion) local features, which will be sent to the subsequent fusion module.

# 3.3.2 Local-Fusion Transformer

In the local-fusion transformer, the weights of each local raw feature are dynamically adjusted based on depth values during feature fusion, and we update local features for each object on the global feature map. Specifically, we take the local multi-modal BEV features  $\mathcal{F}_B^L$ , local voxel features  $\mathcal{V}_O^L$ , and local multi-view image features  $\mathcal{I}_O^L$  as inputs, and integrate the depth encoding by multiplying it with the local multi-modal BEV features, forming the query  $Q_{\mathcal{F}}^L$ . The local multi-view image features and local voxel features are respectively queried as the corresponding key  $K_{\mathcal{I}}^L$ ,  $K_{\mathcal{V}}^L$  and value  $V_{\mathcal{I}}^L$ ,  $V_{\mathcal{V}}^L$ . The two multi-head cross-attention modules are utilized to achieve the interacted features  $\hat{Q}_{\mathcal{F}}^L$ ,  $\hat{Q}_{\mathcal{F}}^{L'}$ . Note that the computation process of multi-head cross attention is similar to that described in Sec. 3.2.2 and is omitted here. Afterward, we aggregate the above features:

$$
\hat {\mathcal {F}} _ {B} ^ {L} = \operatorname {C o n v} \left(\operatorname {C a t} \left(\hat {Q} _ {\mathcal {F}} ^ {L} + \mathcal {F} _ {B} ^ {L}, \hat {Q} _ {\mathcal {F}} ^ {L ^ {\prime}} + \mathcal {F} _ {B} ^ {L ^ {\prime}}\right)\right), \tag {5}
$$

where  $Cat(\cdot)$  is the concatenation operation;  $Conv(\cdot)$  is used to align with the feature channels of global fused BEV features  $\mathcal{F}_B^G$ . As a result, we obtain enhanced local features by dynamically calling back rich information in raw modalities at various depths. Afterward, we update the global features  $\mathcal{F}_B^G$  by inserting the enhanced local features at corresponding locations.

# 4 Experiments

In this section, we will first introduce the dataset and evaluation metrics, followed by the implementation details. Then, we will compare our method with the state-of-the-art methods on nuScenes and also present results on a more challenging dataset of nuScenes-C with data corruptions. Finally, we will show the ablation studies and qualitative results. More experiments are provided in Appendix A.2.

# 4.1 Experimental Setup

Datasets and evaluation metrics. We evaluate our proposed DH-Fusion on the nuScenes benchmark [3] and a more challenging dataset of nuScenes-C [13] with data corruptions. nuScenes dataset provides 700 scene sequences for training, 150 scene sequences for validation, and 150 scene sequences for testing. Each sequence contains 40 frames of 32-beam LiDAR data, and each frame

has six corresponding images covering a 360-degree field of view. It offers calibration matrices that facilitate accurate projection of 3D points onto 2D pixels, and contains 10 object categories that are commonly encountered within autonomous driving. nuScenes-C dataset provides 27 corruptions with 5 severities on the nuScenes validation set, including corruptions at the weather, sensor, motion, object, and alignment level. We use the nuScenes detection scores (NDS) and mean Average Precision (mAP) to evaluate our detection results, where NDS is a comprehensive metric in nuScenes that combines object translation, scale, orientation, velocity, and attribute errors.

Implementation details. We implement the proposed DH-Fusion with PyTorch [39] under the open-source framework MMDetection3D [10]. Specifically, for the LiDAR branch, we use VoxelNet [71] with FPN [61] as the 3D encoder. The voxel size is set to  $[0.075\mathrm{m}, 0.075\mathrm{m}, 0.1\mathrm{m}]$ , and the range of point cloud is  $[-54\mathrm{m}, 54\mathrm{m}]$  along the X-axis,  $[-54\mathrm{m}, 54\mathrm{m}]$  along the Y-axis, and  $[-3\mathrm{m}, 5\mathrm{m}]$  along the Z-axis. For the image branch, we use the ResNet18 [16], ResNet50 [16], and SwinTiny [34] with FPN [29] as the 2D image encoder of DH-Fusion-light, -base, -large, respectively. Correspondingly, the resolution of input images is resized to  $256 \times 704$ ,  $320 \times 800$ , and  $384 \times 1056$ . Additionally, we utilize BEVPoolV2 [18] to obtain image BEV features. Following [33], the feature size  $W \times H$  is set to  $180 \times 180$ , the channel  $C$  is set to 128, and the channel  $c$  is also set to 128. The multi-head cross attention is implemented with 8 heads, and the FFN contains 2 MLP layers with a hidden dimension of 128. Following [58], the number of regressed 3D boxes  $t$  is set to 200. More implementation details are provided in Appendix A.1.

# 4.2 Comparison to the State of the Art

Aiming for a fair comparison, we categorize previous methods based on the types of 2D backbones into ResNet50-based, SwinTiny-based, and others, and provide three versions of our proposed method, named DH-Fusion-light, DH-Fusion-base, and DH-Fusion-large. The results are shown in Tab. 1. (1) Compared with the ResNet50-based methods, our DH-Fusion-base outperforms the top method FocalFormer3D [7] by up to 1 pp w.r.t. NDS under the same configuration. Specifically, we reach  $74.0\%$  w.r.t. NDS and  $71.2\%$  w.r.t. mAP on the validation set, and  $74.7\%$  w.r.t. NDS and  $71.7\%$  w.r.t. mAP on the test set, while maintaining comparable inference speed of 8.7 FPS on a 3090 GPU. (2) Compared with the SwinTiny-based methods and others, our DH-Fusion-large outperforms the top method IS-Fusion [67] under the same configuration, and runs 2x faster than it. Specifically, we reach  $74.4\%$  w.r.t. NDS on the validation set, and  $75.4\%$  w.r.t. NDS on the test set, while achieving a faster inference speed of 5.7 FPS on a 3090 GPU, indicating that our proposed method is both more effective and efficient. (3) Furthermore, our DH-Fusion-light surpasses the typical BEVFusion [33] by up to 1 pp w.r.t. all metrics using a lighter 2D backbone, and achieves a real-time inference speed of 13.8 FPS. Overall, our method achieves higher detection accuracy and faster inference speed.

# 4.3 Robustness to Corruptions

We further implement some experiments on the nuScenes-C [13] dataset to evaluate the model's robustness under various corruptions, including changes in weather, data loss or temporal-spatial misalignment in multi-modal inputs, etc. The results for different kinds of corruptions are shown in Tab. 2, and more detailed results for each fine-grained corruption are shown in Appendix A.2.3. We find that our DH-Fusion-light still achieves an average performance of  $68.67\%$  w.r.t. NDS and  $63.07\%$  w.r.t. mAP under various corruptions, which only decreases by 4.63 pp w.r.t. NDS and 6.68 pp w.r.t. mAP, compared to its performance without corruptions. Performance drop is smaller than that observed with previous methods including BEVFusion [28] across all kinds of corruptions, indicating that our DH-Fusion-light possesses superior robustness. Furthermore, we observe that our DH-Fusion-light is particularly robust against weather and object corruptions, where the performance drop is less than 3pp. The more stable performance indicates that our method is more friendly to practical applications, where data corruption may occur.

# 4.4 Ablation Studies

We conduct ablation studies to first demonstrate the effect of each component of DH-Fusion, then to demonstrate the effect of depth encoding in DGF and DLF, and finally to assess the impact of multiplying depth encoding. All method variants are implemented on the nuScenes validation dataset.

Table 1: Comparisons with the state of the art on the nuScenes validation and test sets. FPS is measured on a 3090 GPU by default, and * denotes the inference speed on an A100 GPU referred from the original paper. Note that all results are obtained without any model ensemble or test time augmentation.  

<table><tr><td>Methods</td><td>Present at</td><td>Image Size - 2D Backbone</td><td>FPS</td><td>Validation NDS mAP</td><td>Test NDS mAP</td></tr><tr><td colspan="6">Image Backbone: ResNet50[16]</td></tr><tr><td>Trainsfusion [1]</td><td>CVPR&#x27;22</td><td>320 × 800-ResNet50</td><td>6.5</td><td>71.3</td><td>67.5</td></tr><tr><td>DeepInteraction [66]</td><td>NeurIPS&#x27;22</td><td>448 × 800-ResNet50</td><td>1.9</td><td>72.4</td><td>69.9</td></tr><tr><td>MSMDFusion [21]</td><td>CVPR&#x27;23</td><td>448 × 800-ResNet50</td><td>2.1</td><td>72.1</td><td>69.7</td></tr><tr><td>FocalFormer3D [7]</td><td>ICCV&#x27;23</td><td>320 × 800-ResNet50</td><td>9.2*</td><td>73.1</td><td>70.1</td></tr><tr><td>DH-Fusion-base (Ours)</td><td>-</td><td>320 × 800-ResNet50</td><td>8.7</td><td>74.0</td><td>71.2</td></tr><tr><td colspan="6">Image Backbone: SwinTiny[31]</td></tr><tr><td>BEVFusion [28]</td><td>NeurIPS&#x27;22</td><td>448 × 800-SwinTiny</td><td>0.7*</td><td>71.0</td><td>67.9</td></tr><tr><td>BEVFusion [33]</td><td>ICRA&#x27;23</td><td>256 × 704-SwinTiny</td><td>9.6</td><td>71.4</td><td>68.5</td></tr><tr><td>ObjectFusion [4]</td><td>ICCV&#x27;23</td><td>256 × 704-SwinTiny</td><td>-</td><td>72.3</td><td>69.8</td></tr><tr><td>SparseFusion [58]</td><td>ICCV&#x27;23</td><td>256 × 704-SwinTiny</td><td>4.4</td><td>72.8</td><td>70.5</td></tr><tr><td>IS-Fusion [67]</td><td>CVPR&#x27;24</td><td>384 × 1056-SwinTiny</td><td>3.2*</td><td>74.0</td><td>72.8</td></tr><tr><td colspan="6">Image Backbone: Others</td></tr><tr><td>AutoAlignV2 [8]</td><td>ECCV&#x27;22</td><td>640 × 1280-CSPNet [51]</td><td>4.8*</td><td>71.2</td><td>67.1</td></tr><tr><td>UVTR [26]</td><td>NeurIPS&#x27;22</td><td>640 × 1280-ResNet101 [16]</td><td>1.8</td><td>70.2</td><td>65.4</td></tr><tr><td>FUTR3D [5]</td><td>CVPR&#x27;23</td><td>900 × 1600-VOVNet [23]</td><td>3.3*</td><td>68.0</td><td>64.2</td></tr><tr><td>UniTR [54]</td><td>ICCV&#x27;23</td><td>256 × 704-DSVT [53]</td><td>9.3*</td><td>73.3</td><td>70.5</td></tr><tr><td>CMT [60]</td><td>ICCV&#x27;23</td><td>640 × 1600-VOVNet</td><td>6.0*</td><td>72.9</td><td>70.3</td></tr><tr><td>UniPAD [63]</td><td>CVPR&#x27;24</td><td>900 × 1600-ConvNeXtS [34]</td><td>-</td><td>73.2</td><td>69.9</td></tr><tr><td>DH-Fusion-large (Ours)</td><td>-</td><td>384 × 1056-SwinTiny</td><td>5.7</td><td>74.4</td><td>72.3</td></tr><tr><td>DH-Fusion-light (Ours)</td><td>-</td><td>256 × 704-ResNet18</td><td>13.8</td><td>73.3</td><td>69.8</td></tr></table>

Table 2: Robustness experiments on nuScenes-C. Numbers are NDS / mAP.  

<table><tr><td rowspan="2">Methods</td><td colspan="7">Corruption</td><td rowspan="2">Average</td></tr><tr><td>None</td><td>Weather</td><td>Sensor</td><td>Motion</td><td>Object</td><td>Alignment</td><td></td></tr><tr><td>FUTR3D [5]</td><td>68.05 / 64.17</td><td>62.75 / 55.51</td><td>63.66 / 56.83</td><td>53.16 / 44.43</td><td>65.45 / 61.04</td><td>62.83 / 57.60</td><td>62.8215.23 / 56.9917.18</td><td></td></tr><tr><td>TransFusion [1]</td><td>69.82 / 66.38</td><td>65.42 / 59.37</td><td>66.17 / 59.82</td><td>51.52 / 41.47</td><td>68.28 / 64.38</td><td>61.98 / 54.94</td><td>63.7416.08 / 58.7317.65</td><td></td></tr><tr><td>BEVFusion [33]</td><td>71.40 / 68.45</td><td>67.54 / 61.87</td><td>67.59 / 61.80</td><td>55.19 / 47.30</td><td>68.01 / 65.14</td><td>63.94 / 58.71</td><td>66.0615.34 / 61.0317.42</td><td></td></tr><tr><td>DH-Fusion-light (Ours)</td><td>73.30 / 69.75</td><td>72.19 / 67.48</td><td>69.16 / 62.87</td><td>57.07 / 47.52</td><td>71.01 / 67.11</td><td>67.24 / 62.38</td><td>68.6714.63 / 63.0716.68</td><td></td></tr></table>

Effect of DGF and DLF. To demonstrate the effect of DGF and DLF, we conduct experiments by integrating the components one by one into the baseline, BEVFusion [33]. The results are shown in Tab. 3. We find that our DGF improves the baseline performance by  $1.0\mathrm{pp}$  w.r.t. NDS and  $0.9$  pp w.r.t. mAP. This demonstrates that dynamically adjusting the weights of the image BEV features during fusion is effective for 3D object detection. Additionally, our DLF improves the baseline performance by  $1.3\mathrm{pp}$  w.r.t. NDS and  $0.8\mathrm{pp}$  w.r.t. mAP, which indicates that dynamically adjusting the weights of the local raw instance features based on depth during fusion effectively compensates for the information loss caused by the transformation of global features into the BEV feature space. The results of integrating both components show an improvement of  $1.9\mathrm{pp}$  w.r.t. NDS and  $1.3\mathrm{pp}$  w.r.t. mAP, well verifying the benefits of dynamically fusing global and local hybrid features based on depth.

Effect of depth encoding in DGF and DLF. To evaluate the effectiveness of our depth encoding, we conduct experiments where the depth encoding is removed from the DGF and DLF modules, respectively. The results are shown in Tab. 4. When removing the depth encoding from Baseline+DGF, the performance drops by  $0.6\mathrm{pp}$  w.r.t. NDS and  $0.4\mathrm{pp}$  w.r.t. mAP. Similarly, when removing the depth encoding from Baseline+DLF, the performance also decreases by  $1.1\mathrm{pp}$  w.r.t. NDS and  $0.9\mathrm{pp}$  w.r.t. mAP. These results indicate that our depth encoding is effective. Furthermore, we observe that removing the depth encoding from the DLF module results in a larger performance drop, suggesting that depth encoding plays a more crucial role in local feature fusion.

Impact of different operations for depth encoding. We conduct experiments with different operations of depth encoding, including concatenation, summation, and multiplication. The results in Tab. 5, show that the multiplication operation consistently outperforms the summation and concatenation operations w.r.t. both metrics. The superior performance of multiplication can be attributed to its ability to more effectively modulate the feature maps based on depth information. Unlike summation, which simply shifts the feature values, or concatenation, which increases the dimensionality without direct interaction, multiplication allows for more interaction between the

Table 3: Ablation studies of each proposed module.  

<table><tr><td>Baseline</td><td>DGF</td><td>DLF</td><td>NDS</td><td>mAP</td></tr><tr><td>✓</td><td></td><td></td><td>71.4</td><td>68.5</td></tr><tr><td>✓</td><td>✓</td><td></td><td>72.4↑1.0</td><td>69.4↑0.9</td></tr><tr><td>✓</td><td></td><td>✓</td><td>72.7↑1.3</td><td>69.3↑0.8</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>73.3↑1.9</td><td>69.8↑1.3</td></tr></table>

Table 4: Ablation studies of depth encoding (DE) in DGF and DLF.  

<table><tr><td>Methods</td><td>NDS</td><td>mAP</td></tr><tr><td>Baseline + DGF</td><td>72.4</td><td>69.4</td></tr><tr><td>w/o DE</td><td>71.810.6</td><td>69.010.4</td></tr><tr><td>Baseline + DLF</td><td>72.7</td><td>69.3</td></tr><tr><td>w/o DE</td><td>71.611.1</td><td>68.410.9</td></tr></table>

Table 5: Ablation studies of different operations for depth encoding.  

<table><tr><td>Methods</td><td>NDS</td><td>mAP</td></tr><tr><td>Summation</td><td>72.8</td><td>69.2</td></tr><tr><td>Concatenation</td><td>72.5</td><td>68.7</td></tr><tr><td>Multiplication</td><td>73.3</td><td>69.8</td></tr></table>

![](images/e50c807d7e9ab00ef642f30c700bd3b8e551d435cf4cf8c82299478db53b759c.jpg)  
(a) Attention weights

![](images/c7ef52e90902bd6eb7f88d6e6491a96c9416d8c6b7c201c4e72ce1b191821fbd.jpg)  
(b) Average map

![](images/76c932e4e9aa7c1ad5d02e3f76c1dfefa8cf92d61f24db9618059cef624bda32.jpg)  
Figure 5: Attention weights applied on BEV image features in DGF vary with depth.  
Figure 6: Qualitative detection results and BEV features of BEVFusion and ours. We show the ground truth boxes in green, and the prediction boxes in blue.

depth encoding and features, leading to better feature representation and ultimately improving the detection performance.

# 4.5 Qualitative Results

To better understand how depth encoding affects the feature fusion, in Fig. 5, we plot a curve to observe how the attention weights applied on the image BEV features in our DGF module vary with depth, and visualize the average attention map. It is evident that the weights of the image BEV features stay low in near range, but go up significantly as depth increases when the depth is larger than 40 meters. This trend supports our hypothesis that the image modality would become more important as depth increases. In this way, our depth encoding allows the model to dynamically adjust the weights of image BEV features based on depth.

We also compare the detection results of our DH-Fusion method with the baseline BEVFusion [33] in Fig. 6, where we clearly find that our method better localizes those distant objects compared to BEVFusion. These results demonstrate that our proposed multi-modal fusion strategy based on depth is more effective for detection. Besides, we exhibit the corresponding BEV feature maps, where our method shows a stronger feature response for the foreground objects, especially for distant ones. That is why our feature fusion strategy can provide higher-quality detection results. More qualitative results can be found in Appendix A.3.

# 5 Conclusion

In this paper, we for the first time point out that different modalities play different roles as depth varies via statistical analysis and visualization. Based on this finding, we propose a feature fusion strategy for multi-modal 3D object detection, namely Depth-Aware Hybrid Feature Fusion (DH-Fusion), that dynamically adjusts the weights of features during feature fusion by introducing depth encoding at both global and local levels. Extensive experiments on the nuScenes dataset demonstrate that our DH-Fusion method surpasses previous state-of-the-art methods w.r.t. NDS. Moreover, our DH-Fusion is more robust to various kinds of corruptions, outperforming previous methods on the nuScenes-C dataset w.r.t. both NDS and mAP. Our method uses an attention-based approach to interact with the two modalities, making the detection results sensitive to modality loss. We plan to further explore feature fusion methods that are robust to modality loss. Although our method improves detection performance, emergency plans still need to be implemented in practical applications to ensure personnel safety.

# References

[1] Bai, X., Hu, Z., Zhu, X., Huang, Q., Chen, Y., Fu, H., Tai, C.L.: Transfusion: Robust lidar-camera fusion for 3d object detection with transformers. In: CVPR (2022)  
[2] Brazil, G., Liu, X.: M3d-rpn: Monocular 3d region proposal network for object detection. In: ICCV (2019)  
[3] Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., Beijbom, O.: nuscenes: A multimodal dataset for autonomous driving. In: CVPR (2020)  
[4] Cai, Q., Pan, Y., Yao, T., Ngo, C.W., Mei, T.: Objectfusion: Multi-modal 3d object detection with object-centric fusion. In: ICCV (2023)  
[5] Chen, X., Zhang, T., Wang, Y., Wang, Y., Zhao, H.: Futr3d: A unified sensor fusion framework for 3d detection. In: CVPR (2023)  
[6] Chen, Y., Liu, S., Shen, X., Jia, J.: Dsgn: Deep stereo geometry network for 3d object detection. In: CVPR (2020)  
[7] Chen, Y., Yu, Z., Chen, Y., Lan, S., Anandkumar, A., Jia, J., Alvarez, J.M.: Focalformer3d: focusing on hard instance for 3d object detection. In: ICCV (2023)  
[8] Chen, Z., Li, Z., Zhang, S., Fang, L., Jiang, Q., Zhao, F.: Deformable feature aggregation for dynamic multi-modal 3d object detection. In: ECCV (2022)  
[9] Chiu, H.k., Prioletti, A., Li, J., Bohg, J.: Probabilistic 3d multi-object tracking for autonomous driving. arxiv 2020. arXiv preprint arXiv:2001.05673 (2020)  
[10] Contributors, M.: MMDetection3D: OpenMMLab next-generation platform for general 3D object detection. https://github.com/open-mmlab/mmdetection3d (2020)  
[11] Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: CVPR (2009)  
[12] Deng, J., Shi, S., Li, P., Zhou, W., Zhang, Y., Li, H.: Voxel r-cnn: Towards high performance voxel-based 3d object detection. In: AAAI (2021)  
[13] Dong, Y., Kang, C., Zhang, J., Zhu, Z., Wang, Y., Yang, X., Su, H., Wei, X., Zhu, J.: Benchmarking robustness of 3d object detection to common corruptions. In: CVPR (2023)  
[14] Graham, B., Engelcke, M., Van Der Maaten, L.: 3d semantic segmentation with submanifold sparse convolutional networks. In: CVPR (2018)  
[15] He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: CVPR (2017)  
[16] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR (2016)  
[17] Hu, J.S., Kuai, T., Waslander, S.L.: Point density-aware voxels for lidar 3d object detection. In: CVPR (2022)  
[18] Huang, J., Huang, G.: Bevpoolv2: A cutting-edge implementation of bevdet toward deployment. arXiv:2211.17111 (2022)  
[19] Huang, J., Huang, G., Zhu, Z., Ye, Y., Du, D.: Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. arXiv:2112.11790 (2021)  
[20] Huang, J., Ye, Y., Liang, Z., Shan, Y., Du, D.: Detecting as labeling: Rethinking lidar-camera fusion in 3d object detection. arXiv arXiv:2311.07152 (2023)  
[21] Jiao, Y., Jie, Z., Chen, S., Chen, J., Ma, L., Jiang, Y.G.: Msmdfusion: Fusing lidar and camera at multiple scales with multi-depth seeds for 3d object detection. In: CVPR (2023)  
[22] Lang, A.H., Vora, S., Caesar, H., Zhou, L., Yang, J., Beijbom, O.: Pointpillars: Fast encoders for object detection from point clouds. In: CVPR (2019)

[23] Lee, Y., Hwang, J.w., Lee, S., Bae, Y., Park, J.: An energy andgpu-computation efficient backbone network for real-time object detection. In: CVPR workshops (2019)  
[24] Li, B., Ouyang, W., Sheng, L., Zeng, X., Wang, X.: Gs3d: An efficient 3d object detection framework for autonomous driving. In: CVPR (2019)  
[25] Li, P., Chen, X., Shen, S.: Stereo r-cnn based 3d object detection for autonomous driving. In: CVPR (2019)  
[26] Li, Y., Chen, Y., Qi, X., Li, Z., Sun, J., Jia, J.: Unifying voxel-based representation with transformer for 3d object detection. In: NeurIPS (2022)  
[27] Li, Z., Wang, W., Li, H., Xie, E., Sima, C., Lu, T., Qiao, Y., Dai, J.: Bevformer: Learning bird's-eye-view representation from multi-camera images via spatiotemporal transformers. In: ECCV (2022)  
[28] Liang, T., Xie, H., Yu, K., Xia, Z., Lin, Z., Wang, Y., Tang, T., Wang, B., Tang, Z.: Bevfusion: A simple and robust lidar-camera fusion framework. In: NeurIPS (2022)  
[29] Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: CVPR (2017)  
[30] Liu, Y., Wang, L., Liu, M.: YoIostereo3d: A step back to 2d for efficient stereo 3d detection. In: ICRA. IEEE (2021)  
[31] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: ICCV (2021)  
[32] Liu, Z., Wu, Z., Toth, R.: Smoke: Single-stage monocular 3d object detection via keypoint estimation. In: CVPR (2020)  
[33] Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D.L., Han, S.: Bevfusion: Multi-task multi-sensor fusion with unified bird's-eye view representation. In: ICRA (2023)  
[34] Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: CVPR (2022)  
[35] Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)  
[36] Mao, J., Shi, S., Wang, X., Li, H.: 3d object detection for autonomous driving: A comprehensive survey. IJCV (2023)  
[37] Pang, S., Morris, D., Radha, H.: Clocs: Camera-lidar object candidates fusion for 3d object detection. In: IROS (2020)  
[38] Pang, S., Morris, D., Radha, H.: Fast-clocs: Fast camera-lidar object candidates fusion for 3d object detection. In: WACV (2022)  
[39] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, high-performance deep learning library. In: NeurIPS (2019)  
[40] Qi, C.R., Yi, L., Su, H., Guibas, L.J.: Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In: NeurIPS (2017)  
[41] Qin, Z., Wang, J., Lu, Y.: Monogrnet: A geometric reasoning network for monocular 3d object localization. In: AAAI (2019)  
[42] Shi, S., Guo, C., Jiang, L., Wang, Z., Shi, J., Wang, X., Li, H.: Pv-rcnn: Point-voxel feature set abstraction for 3d object detection. In: CVPR (2020)  
[43] Shi, S., Jiang, L., Deng, J., Wang, Z., Guo, C., Shi, J., Wang, X., Li, H.: Pv-rcnn++: Point-voxel feature set abstraction with local vector representation for 3d object detection. IJCV (2022)

[44] Shi, S., Wang, X., Li, H.: Pointrecnn: 3d object proposal generation and detection from point cloud. In: CVPR (2019)  
[45] Shi, S., Wang, Z., Shi, J., Wang, X., Li, H.: From points to parts: 3d object detection from point cloud with part-aware and part-aggregation network. IEEE TPAMI (2020)  
[46] Shi, W., Rajkumar, R.: Point-gnn: Graph neural network for 3d object detection in a point cloud. In: CVPR (2020)  
[47] Shi, X., Ye, Q., Chen, X., Chen, C., Chen, Z., Kim, T.K.: Geometry-based distance decomposition for monocular 3d object detection. In: ICCV (2021)  
[48] Sun, J., Chen, L., Xie, Y., Zhang, S., Jiang, Q., Zhou, X., Bao, H.: Disp r-cnn: Stereo 3d object detection via shape prior guided instance disparity estimation. In: CVPR (2020)  
[49] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: NeurIPS (2017)  
[50] Vora, S., Lang, A.H., Helou, B., Beijbom, O.: Pointpainting: Sequential fusion for 3d object detection. In: CVPR (2020)  
[51] Wang, C.Y., Liao, H.Y.M., Wu, Y.H., Chen, P.Y., Hsieh, J.W., Yeh, I.H.: Cspnet: A new backbone that can enhance learning capability of cnn. In: CVPR workshops (2020)  
[52] Wang, C., Ma, C., Zhu, M., Yang, X.: Pointaugmenting: Cross-modal augmentation for 3d object detection. In: CVPR (2021)  
[53] Wang, H., Shi, C., Shi, S., Lei, M., Wang, S., He, D., Schiele, B., Wang, L.: Dsvt: Dynamic sparse voxel transformer with rotated sets. In: CVPR (2023)  
[54] Wang, H., Tang, H., Shi, S., Li, A., Li, Z., Schiele, B., Wang, L.: Unitr: A unified and efficient multi-modal transformer for bird's-eye-view representation. In: ICCV (2023)  
[55] Wang, T., Zhu, X., Pang, J., Lin, D.: Fcos3d: Fully convolutional one-stage monocular 3d object detection. In: ICCV (2021)  
[56] Wang, Y., Guizilini, V.C., Zhang, T., Wang, Y., Zhao, H., Solomon, J.: Detr3d: 3d object detection from multi-view images via 3d-to-2d queries. In: Robot Learning (2022)  
[57] Wu, H., Wen, C., Shi, S., Li, X., Wang, C.: Virtual sparse convolution for multimodal 3d object detection. In: CVPR (2023)  
[58] Xie, Y., Xu, C., Rakotosaona, M.J., Rim, P., Tombari, F., Keutzer, K., Tomizuka, M., Zhan, W.: Sparsefusion: Fusing multi-modal sparse representations for multi-sensor 3d object detection. In: ICCV (2023)  
[59] Xu, S., Zhou, D., Fang, J., Yin, J., Bin, Z., Zhang, L.: Fusionpainting: Multimodal fusion with adaptive attention for 3d object detection. In: ITSC (2021)  
[60] Yan, J., Liu, Y., Sun, J., Jia, F., Li, S., Wang, T., Zhang, X.: Cross modal transformer via coordinates encoding for 3d object detection. In: ICCV (2023)  
[61] Yan, Y., Mao, Y., Li, B.: Second: Sparsely embedded convolutional detection. Sensors (2018)  
[62] Yang, C., Chen, Y., Tian, H., Tao, C., Zhu, X., Zhang, Z., Huang, G., Li, H., Qiao, Y., Lu, L., et al.: Bevformer v2: Adapting modern image backbones to bird's-eye-view recognition via perspective supervision. In: CVPR (2023)  
[63] Yang, H., Zhang, S., Huang, D., Wu, X., Zhu, H., He, T., Tang, S., Zhao, H., Qiu, Q., Lin, B., He, X., Ouyang, W.: Unipad: A universal pre-training paradigm for autonomous driving. In: CVPR (2024)  
[64] Yang, Z., Sun, Y., Liu, S., Shen, X., Jia, J.: Ipod: Intensive point-based object detector for point cloud. arXiv:1812.05276 (2018)

[65] Yang, Z., Sun, Y., Liu, S., Shen, X., Jia, J.: Std: Sparse-to-dense 3d object detector for point cloud. In: ICCV (2019)  
[66] Yang, Z., Chen, J., Miao, Z., Li, W., Zhu, X., Zhang, L.: Deepinteraction: 3d object detection via modality interaction. In: NeurIPS (2022)  
[67] Yin, J., Shen, J., Chen, R., Li, W., Yang, R., Frossard, P., Wang, W.: Is-fusion: Instance-scene collaborative fusion for multimodal 3d object detection. In: CVPR (2024)  
[68] Yin, T., Zhou, X., Krahenbuhl, P.: Center-based 3d object detection and tracking. In: CVPR (2021)  
[69] Yin, T., Zhou, X., Krahenbuhl, P.: Multimodal virtual point 3d detection. In: NeurIPS (2021)  
[70] You, Y., Wang, Y., Chao, W.L., Garg, D., Pleiss, G., Hariharan, B., Campbell, M., Weinberger, K.Q.: Pseudo-lidar++: Accurate depth for 3d object detection in autonomous driving. arXiv:1906.06310 (2019)  
[71] Zhou, Y., Tuzel, O.: Voxelnet: End-to-end learning for point cloud based 3d object detection. In: CVPR (2018)  
[72] Zhu, B., Jiang, Z., Zhou, X., Li, Z., Yu, G.: Class-balanced grouping and sampling for point cloud 3d object detection. arXiv preprint arXiv:1908.09492 (2019)  
[73] Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159 (2020)
