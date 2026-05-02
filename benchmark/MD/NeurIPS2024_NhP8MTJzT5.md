# SyntheOcc: Synthesize Geometric-Controlled Street View Images through 3D Semantic MPIs

Anonymous Author(s)

Affiliation

Address

email

# Abstract

The advancement of autonomous driving is increasingly reliant on high-quality annotated datasets, especially in the task of 3D occupancy prediction, where the occupancy labels require dense 3D annotation with significant human effort. In this paper, we propose SyntheOcc, which denotes a diffusion model that Synthesize photorealistic and geometric-controlled images by conditioning Occupancy labels in driving scenarios. This yields an unlimited amount of diverse, annotated, and controllable datasets for applications like training perception models and simulation. SyntheOcc addresses the critical challenge of how to efficiently encode 3D geometric information as conditional input to a 2D diffusion model. Our approach innovatively incorporates 3D semantic multi-plane images (MPIs) to provide comprehensive and spatially aligned 3D scene descriptions for conditioning. As a result, SyntheOcc can generate photorealistic multi-view images and videos that faithfully align with the given geometric labels (semantics in 3D voxel space). Extensive qualitative and quantitative evaluations of SyntheOcc on the nuScenes dataset prove its effectiveness in generating controllable occupancy datasets that serve as an effective data augmentation to perception models.

# 1 Introduction

With the rapid development of generative models, they have shown realistic image synthesis and diverse controllability. This progress has opened up new avenues for dataset generation in autonomous driving [5, 12, 23, 30]. The task of dataset generation is usually modeled as controllable image generation, where the ground truth (e.g. 3D Box) is employed to control the generation of new datasets in downstream tasks (e.g. 3D detection). This approach helps to mitigate the data collection and annotation effort as it can generate labeled data for free. However, a novel task of vital importance, occupancy prediction [24, 27], poses new challenges for dataset generation compared with 3D detection. It requires finer and more nuanced geometry controllability, which refers to use the occupancy state and semantics of voxels in the whole 3D space to control the image generation. We argue that solving this problem not only allows us to synthesize occupancy datasets, but also empowers valuable applications such as editing geometry to generate rare data for corner case evaluation, as shown in Fig. 1. In the following, we first illustrate why prior work struggles to achieve the above objective, and then demonstrate how we address these challenges.

In the area of diffusion models, several representative works have displayed high-quality image synthesis; however, they are constrained by limited 3D controllability: they are incapable of editing 3D voxels for precise control. For example, BEVGen [23] generates street view images by conditioning BEV layouts using diffusion models. MagicDrive [5] extend BEVGen and additionally converts the 3D box parameters into text embedding through Fourier mapping that is similar to NeRF [19], and uses cross-attention to learn conditional generation. Although these methods achieve satisfactory results in image generation, their 3D controllability is inherently limited. These approaches are

![](images/27ce459714b8435e78cf3be5727cab4f3a3184088951b5c0fcbea47185addb62.jpg)  
Figure 1: A showcase of application of SytheOcc. We enable geometric-controlled generation that conveys the user editing in 3D voxel space to generate realistic street view images. In this case, we create a rare scene that traffic cones block the way. This advancement facilitates the evaluation of autonomous systems, such as the end-to-end planner VAD [9], in simulated corner case scenes.

![](images/371ccd8b006630164faf81bf298b7df95e31260aee78b1402213dee173dbfc08.jpg)

![](images/90581d37b6fc8f018b9bcfeddb28fe61dc31a0ee678f642ef24899ef4059dece.jpg)

restricted to manipulating the scene in types of 3D boxes and BEV layouts, and hardly adapt to finer geometry control such as editing the shape of objects and scenes. Meanwhile, they usually convert conditional input into 1D embedding that aligns with prompt embedding, which is less effective in 3D-aware generation due to lack of spatial alignment with the generated images. This limitation hinders their utility in downstream applications, such as occupancy prediction and editing scene geometry to create long-tailed scenes, where granular volumetric control is paramount in both tasks.

ControlNet [41] and GLIGEN [14] is another type of prominent method in the field of controllable image generation. These approaches exhibit several desirable attributes in terms of controllability. They leverage conditional images such as semantic masks for control, thereby offering a unified framework to manipulate both foreground and background. However, despite its precise spatial control, ControlNet does not align with our specific requirements. Their conditions of pixel-level images differ fundamentally from what we require in 3D contexts. Our experimental results also find that ControlNet struggles to handle overlapping objects with varying depths (see Fig. 6 (a)), as it only utilizes an ambiguous 2D semantic map as conditional input. As a result, it is non-trivial to extend the ControlNet framework and convey their desirable attributes for 3D conditioning.

To address the above challenges, we propose an innovative representation, 3D semantic multi-plane images (MPIs), which contribute to image generation with finer geometric control. In detail, we employ multi-plane images [43] to represent the occupancy, where each plane represents a slice of semantic label at a specific depth. Our 3D semantic MPIs not only preserve accurate and authentic 3D information, but also keep pixel-wise alignment with the generated images. We additionally introduce the MPI encoder to encode features, and the reweighing methods to ease the training with long-tailed cases. As a collection, our framework enables 3D geometry and semantic control for image generation and further facilitates corner case evaluation as depicted in Fig. 1. Finally, experimental results demonstrate that our synthetic data achieve better recognizability, and are effective in improving the perception model on occupancy prediction. In summary, our contributions include:

- We present SytheOcc, a novel image generation framework to attain finer and precise 3D geometric control, thereby unlocking a spectrum of applications such as 3D editing, dataset generation, and long-tailed scene generation.  
- Incorporating the proposed 3D semantic MPI, MPI encoder, and reweighing strategy, we deliver a substantial advancement in image quality and recognizability over prior works.  
- Our extensive experimental results demonstrate that our synthetic data yields an effective data augmentation in the realm of 3D occupancy prediction.

# 2 Related Work

# 2.1 3D Occupancy Prediction

The task of 3D occupancy prediction aims to predict the occupancy status of each voxel in 3D space, as well as its semantic label if occupied. Compared with previous perception methods like 3D object

detection, occupancy prediction offers a more detailed and nuanced understanding of the environment, as it provides finer geometric details, is capable of handling general, out-of-vocabulary objects, and finally, enriches the planning stack with comprehensive 3D information. Early methods exploited LiDAR as inputs to complete the 3D occupancy of the entire 3D scene [18, 33]. Recent methods began to explore the more challenging vision-based 3D occupancy prediction [24, 25, 27, 29]. By predicting the geometric and semantic properties of both dynamic and static elements, 3D occupancy prediction offers a more comprehensive understanding of the surrounding environment.

# 2.2 Diffusion-based Image Generation

Recent advancements in diffusion models (DMs) have achieved remarkable progress in image generation. In particular, Stable Diffusion (SD) [21] employs DMs within the latent space of autoencoders, striking a balance between computational efficiency and high image quality. Beyond text control, there is also the introduction of additional control signals. A noteworthy work is ControlNet [41], which incorporates a trainable copy of the SD encoder to extract the feature of conditional images and adds it to the UNet feature. It significantly enhances the controllability and unlocking pathways for advanced applications. We refer readers to recent survey [35] for more details.

# 2.3 Image Generation in Autonomous Driving

As training neural networks relies heavily on labeled data, numerous studies are delving into dataset generation to boost training. Lift3D [12] designs generative NeRF to synthesize labeled datasets for 3D detection for the first time. Several other works employ BEV layouts to synthesize image data, proving beneficial for perception models. For example, BEVGen [23] conditions BEV layouts to generate multi-view street images, while BEVControl [34] separately generates foregrounds and backgrounds from BEV layouts. MagicDrive [5] generates images with 3D geometry controls by independently encoding objects and maps through a text encoder or map encoder. Compared with MagicDrive, our geometry control is characterized by a more detailed and lossless representation of 3D scenes for control, which poses significant challenges than projected layout or box embedding.

Recently, DriveDreamer [26], DrivingDiffusion [13], Drive-WM [28] and Panacea [30] use a ControlNet framework, which involves projecting bounding boxes and road maps onto 2D FoV images as a conditioning input. This approach has proven to be effective for geometric control. However, it is limited in that it only achieves alignment at the 2D-pixel level. Consequently, this method falls short in capturing the depth hierarchy and fails to account for the occlusion relationships present in the 3D real world. Besides, adding a depth channel like Panacea [30] may address the limitations of depth order, but it discards the occluded part and only contains partial observation. UrbanGiraffe [37] train a generative NeRF to perform image generation. WoVoGen [17] creates a 4D world volume feature using occupancy to guide the generation, but seems to rely on object mask guidance.

As described above, most of the prior work is restricted by only modeling a projected primitive of 3D boxes and road maps as conditions. They suffer from ill-posed un-projection ambiguity. In contrast, we model 3D occupancy labels as conditions, as they provide finer geometric details and semantic information. However, designing an input representation of 3D occupancy labels into a 2D diffusion model is challenging. In this paper, we propose a novel representation: 3D semantic Multi-Plane Images (MPIs) as conditional inputs, which not only provide spatial alignment that improves visual consistency, but also encode comprehensive 3D geometric information including occluded parts.

# 3 Method

Overview The overview of our method is depicted in Fig. 2. Built upon the SD pipeline, we aim to perform geometry-controlled image generation by conditioning on 3D geometry labels with semantics (occupancy labels). One requirement is that the images should faithfully align with the given label. This task is more challenging than conditioned on 3D box due to the sparse and irregular nature of occupancy. We first discuss how to efficiently represent occupancy in Sec. 3.2, followed by our designed MPI encoder to enhance generation quality in Sec. 3.3, and reweighing strategy to handle the long-tailed depth and category in Sec. 3.5.

# 3.1 Representation of Condition: Local Control Aligns Better than Global Control

One of the key challenges is how to represent our conditional occupancy input. A straightforward method [3, 5] is to convert the 3D occupancy voxel to 1D global embedding that is similar to text embedding, and then use cross-attention to learn controllable generation. However, these global methods can be less effective when dealing with dense or irregular data due to the following reasons:

![](images/455f1739116f4bad58ce8a4d9d0d44e053432cec84ae1ef10ea385c8498678b3.jpg)  
Figure 2: The overall architecture of SytheOcc. We achieve 3D geometric control in image generation by utilizing our proposed 3D semantic multiplane images to encode scene occupancy. In our framework, we can edit the occupied state and semantics of every voxel in 3D space to control the image generation, thereby opening up a wide spectrum of applications as shown in the top right.

(i) They perform controllable generation through hard encoding the spatial relationship between 1D global embedding and 2D UNet features. (ii) Ignore the underlying geometry alignment between the conditional input and the generated image. In contrast, local methods like ControlNet, directly add spatial features to the UNet features, providing 2D local control with pixel-level spatial alignment. They are better than the global method (see Tab. 1), but suffer from 3D ambiguity (see Fig. 6 (a)). Consequently, this comparison motivates us to seek a more compact and efficient manner to encode and condition our 3D occupancy labels.

# 3.2 Represent Occupancy as 3D Semantic Multiplane Images

It is non-trivial to design a 3D representation for conditioning. To efficiently store both the semantic and geometric information of the irregular occupancy input, we propose to use multiplane images (MPIs) [43] as representation. An MPI is composed of a series of fronto-parallel RGBA layers within the frustum of the source camera with a specific viewpoint. These planes are arranged at varying depths, from  $d_{min}$  to  $d_{max}$ , starting from the nearest to the farthest. Each layer of these images contains both an RGB image and an alpha map, which collectively capture the visual and geometric details of the scene at the respective depth. In our work, instead of storing RGB value and alpha map in the original MPI, we store our 3D semantic labels. Each layer of MPI represents the semantic index at the corresponding depth. We display the colored MPI in the top row of Fig. 2 for visual clarity, but we actually use the integer index for learning. We obtain our 3D semantic MPI by:

$$
P _ {l} = \left(u \times d _ {l}, v \times d _ {l}, d _ {l}\right) ^ {T}, d _ {l} = d _ {\min } + \left(d _ {\max } - d _ {\min }\right) \times l / D, \tag {1}
$$

$$
\mathbf {M P I} _ {n, l} = \text {I n t e r p o l a t e} (\text {O c c u p a n c y}, \mathbf {T} _ {\mathbf {n}} \cdot \mathbf {K} _ {\mathbf {n}} ^ {- 1} \cdot P _ {l}), \tag {2}
$$

$$
\mathbf {M P I} = \text {C o n c a t e n a t e} \left(\mathbf {M P I} _ {i, j}\right), i \in (0, N), j \in (0, D), \tag {3}
$$

where  $(u,v)$  is a pixel coordinate in image space,  $d_{l}$  is depth value of the  $l^{th}$  layer,  $n$  denotes the  $n^{th}$  camera view. This equation implies we first back project points  $P$  in camera frustum space  $(u,v,d)$  to Euclid space  $(x,y,z)$  by multiplying inverse intrinsic  $\mathbf{K}^{-1}$ . Then we use transformation matrix  $\mathbf{T}$  to map points from camera coordinates to occupancy coordinates. We then use the point coordinates to interpolate the nearest semantic index from the dense occupancy voxel to form a slice of MPI. Finally, we concatenate all slices to form  $\mathsf{MPI} \in \mathbb{R}^{N \times D \times H \times W}$ , where  $D$  is the number of layers that is set at 256,  $N$  is the number of camera views in the case of batch size = 1.

By representing occupancy as 3D semantic MPI, every pixel in MPI contains geometry and semantic information with implicit depth, seamlessly integrating occluded elements, and ensuring a precise spatial alignment with the generated images.

# 3.3 3D Semantic MPI Encoder

To enable local control with spatially aligned conditions, we develop a simple but effective MPI encoder that aligns the 3D multi-plane feature to the latent space of the diffusion model. The purpose of the MPI encoder is to obtain features from multi-plane images to perform 3D-aware

![](images/75851608b54f9e37c6326467c36ea36310a258892de00983c549c2255c1bf312.jpg)  
(a) Raw generation  
Figure 3: Visualizations of geometric controlled generation. Top row: Fusion of 3D semantic MPI. Bottom row: our generation concatenated from neighboring views.

![](images/46e37c9acc57e8ba89a5b519116266e0fe9aad89a3c82c96e3ba69f634d86a60.jpg)  
(b) Editing: Object manipulation (copy object)

![](images/e31f0d8c2b40a856bbae2b265fe12d343d2948ce2c2cb476064c5ab7525edaa6.jpg)  
(c) Editing: Foreground removal

image synthesis. Unlike the original ControlNet which downsampling conditional input through  $3\times 3$  convolutions with padding, we design a  $1\times 1$  convolutional encoder without downsampling to encode features. In detail, the 3D multiplane features which have the sample resolution with latent features, are transformed by a  $1\times 1$  convolution layer and ReLU activation [1] in the MPI encoder.

After obtaining the multi-scale feature after the MPI encoder, we add the feature to the decoder of diffusion UNet to provide spatial features. Experimental results in Tab. 3 will show that our  $1 \times 1$  conv in MPI encoder is more effective than  $3 \times 3$  conv, as the  $1 \times 1$  conv with receptive field  $= 1$  provides a spatial align feature to the latent feature in the diffusion UNet. In contrast,  $3 \times 3$  conv is conducted in a camera frustum space rather than Euclid space, making an imprecise correspondence between 3D multiplane features and 2D image features. Moreover, using  $3 \times 3$  conv to process 3D semantic MRI will introduce a large computational burden as the channel number increases from 3 channels of RGB to 256 planes. We display our 3D geometry and semantic control property in Fig. 3.

In summary, we chose MPIs as the representation because they (i) Incorporate lossless 3D information, including scene geometry rather than 2.5D depth. (ii) Provide spatially aligned conditional features that naturally extend the ControlNet framework from image level to 3D level. (iii) Capable of representing geometry and semantics including occluded elements.

# 3.4 Cross-View and Cross-Frame Attention

The sensor arrangement in a self-driving car usually requires a full surround view of cameras to capture the entire 360-degree environment. To effectively simulate the multi-view and subsequent multi-frame generation, zero-initialized [41] cross-view and cross-frame attention are integrated into the diffusion model to maintain consistency between views and frames. Following prior work [5, 28, 30, 31], each cross-view attention allows the target view to access information from its neighboring left and right views, thus training cross-view attention using multi-view consistent images will enforce it to generate the same instance in the overlapping region of multi-view cameras.

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q K ^ {T}}{\sqrt {d}}\right) \cdot V, \tag {4}
$$

$$
h _ {o u t} = h _ {i n} + \sum_ {i \in \{l, r \}} \text {A t t e n t i o n} \left(Q _ {i n}, K _ {i}, V _ {i}\right), \tag {5}
$$

where  $l$ , and  $r$  is the camera view of left and right.  $Q_{in}$  and  $h_{in}$  denotes the query and the hidden state of input view. Similarly, we add cross-frame attention that attend previous frame and future frame to enable video generation. In this case, we use the same formulation while  $i \in \{f, h\}$ , where  $f$  and  $h$  is the camera view of future and history frames.

# 3.5 Importance Reweighing

To deal with the extreme imbalance problem between foreground, background, and object categories, and also to ease the training, we propose three types of reweighting methods to improve the generation quality of foreground objects.

Progressive Foreground Enhancement To mitigate the complexity of the learning task, we propose a progressive reweighting method that incrementally enhances the loss associated with the foreground regions (based on semantic class) as the training progresses. The detailed formulation is:

$$
w (x, m, n) = \frac {(m - 1)}{2} \cdot \left(1 + \cos \left(\frac {x}{n} \cdot \pi + \pi\right)\right) + 1, \tag {6}
$$

![](images/1888d15fe925cf9da6f828ae31698821b71a68d3345cc989878f8358434e67c4.jpg)  
Figure 4: Visualizations of the reweighing function in Eq. 6.

![](images/2ec09db3d77c1c3fb15a52293484798402cbaca574e199232398e6d8ae8c0053.jpg)  
(a) Top: Fusion of 3D semantic MPI. Bottom: GT

![](images/908e8fab934ea5c1c823abf1e26f2ab2ead9b119b0d2f58d664eac05ebc339a1.jpg)  
(b) Generation1: Ordinary scenes. Red rectangle denotes geometry alignment of trees

![](images/62956685f66f6804f1e780e2a1eeb47fc5b22476197687a2ae20a23daf7ecec0.jpg)  
(c) Generation2, Weather variation: Snow (top) and Sandstorm (bottom)

![](images/04ee51d020286b8eaf97fc6da42ac54707ea346b582c5b91688a22dbd0cfe8a4.jpg)  
Figure 5: Visualizations of generated multi-view images. The generation conditions (occupancy labels) are from nuScenes validation set. We highlight that (i) Geometry alignment of trees in red rectangle in (b). (ii) Use text prompt to control high-level appearance in (c,d).  
(d) Generation3, Style control: Minecraft style (top) and Diablo style (bottom)

where  $x$  is the current training step,  $m$  is the maximum value of weights that set at 2, and  $n$  is the total training steps. This approach is engineered to facilitate a learning trajectory that progresses from simplicity to complexity, thereby aiding in the convergence of the model. This curve can be interpreted as a cosine annealing but inverted to amplify the importance of the foreground region.

Depth-aware Foreground Reweighing In the meantime, we acknowledge the learning difficulty in different depth places in 3D scenes. Following GeoDiffusion [3], we perform depth reweighing to foreground objects by adaptively assigning higher weights to farther foreground areas. This enables the model to focus more thoroughly on hard examples with depth-aware importance reweighting. Instead of using their exponential function to increase weights, we use our designed cosine function Eq. 6 for stability. Here  $x$  is the input depth value, and  $n$  is the maximum depth that set at 50.

CBGS Sampling To deal with the class imbalance problem in driving scenarios, where certain object categories appear infrequently, we employ the Class-Balanced Grouping and Sampling (CBGS) [44] to better handle the long-tailed classes. CBGS addresses the challenge of class imbalance by grouping and re-sampling training data to ensure each group has a balanced distribution of sample frequency across different object categories. This method reduces the bias towards more frequent classes and enables better generalization to rare scenarios.

# 3.6 Model Training

To ease the training of the MPI encoder and added attention module, we use a two stage training pipeline. We first train MPI encoder and cross-view attention in a multi-view image generation setting. Then we train cross-frame attention and freeze other components in a video generation setting.

Table 1: Downstream evaluation on the nuScenes-Occupancy validation set. Based on the used train and val data, two types of settings are reported. The first is to use generated training set to augment the real training set, and evaluate on the real validation set, denoted as Aug. The second is to use pretrained models trained on the real training datasets to test on the generated validation set, denoted as Gen.  

<table><tr><td>Method</td><td>Train</td><td>Val</td><td>mIoU</td><td>barrier</td><td>bicycle</td><td>bus</td><td>car</td><td>cons. veh.</td><td>motor.</td><td>pedes.</td><td>traf. cone</td><td>trailer</td><td>truck</td><td>drive. suf.</td><td>other flat</td><td>sidewalk</td><td>terrain</td><td>manmade</td><td>vegetation</td></tr><tr><td>Oracle (FB-Occ [15])</td><td>Real</td><td>Real</td><td>39.3</td><td>45.4</td><td>28.2</td><td>44.1</td><td>49.4</td><td>25.9</td><td>28.8</td><td>28.0</td><td>27.7</td><td>32.4</td><td>37.3</td><td>80.4</td><td>42.2</td><td>49.9</td><td>55.2</td><td>42.0</td><td>37.7</td></tr><tr><td>SytheOcc-Aug</td><td>Real+Gen</td><td>Real</td><td>40.3</td><td>45.4</td><td>27.2</td><td>46.6</td><td>49.5</td><td>26.4</td><td>27.8</td><td>28.4</td><td>29.4</td><td>34.0</td><td>37.2</td><td>81.3</td><td>46.0</td><td>52.4</td><td>56.5</td><td>43.3</td><td>38.9</td></tr><tr><td>MagicDrive</td><td>Real</td><td>Gen</td><td>13.4</td><td>0.7</td><td>0.0</td><td>11.8</td><td>32.4</td><td>0.0</td><td>6.6</td><td>2.8</td><td>0.3</td><td>2.6</td><td>19.6</td><td>60.1</td><td>12.1</td><td>26.2</td><td>23.4</td><td>15.5</td><td>12.8</td></tr><tr><td>ControlNet</td><td>Real</td><td>Gen</td><td>17.3</td><td>17.7</td><td>0.2</td><td>13.6</td><td>21.0</td><td>0.6</td><td>0.8</td><td>8.6</td><td>10.4</td><td>6.9</td><td>11.9</td><td>67.4</td><td>18.8</td><td>36.4</td><td>36.9</td><td>20.8</td><td>22.4</td></tr><tr><td>ControlNet+depth</td><td>Real</td><td>Gen</td><td>17.5</td><td>19.3</td><td>0.3</td><td>14.0</td><td>23.7</td><td>1.0</td><td>0.6</td><td>9.2</td><td>9.2</td><td>5.7</td><td>12.1</td><td>68.8</td><td>19.2</td><td>36.0</td><td>35.3</td><td>19.8</td><td>22.8</td></tr><tr><td>SytheOcc-Gen</td><td>Real</td><td>Gen</td><td>25.5</td><td>32.6</td><td>13.8</td><td>27.7</td><td>33.4</td><td>7.5</td><td>6.5</td><td>15.7</td><td>16.5</td><td>16.5</td><td>25.6</td><td>74.3</td><td>24.5</td><td>39.4</td><td>40.5</td><td>28.6</td><td>28.8</td></tr></table>

Objective Function Our final objective function can be formulated as a standard denoising objective with reweighing:

$$
\mathcal {L} = \mathbb {E} _ {\mathcal {E} (x), \epsilon , t} \| \epsilon - \epsilon_ {\theta} \left(z _ {t}, t, \tau_ {\theta} (y)\right) \| ^ {2} \odot w, \tag {7}
$$

where  $w$  is the multiplication of progressive reweighing and depth-aware reweighing.

# 4 Experiments

# 4.1 Dataset and Setups

We conduct our experiments on the nuScenes dataset [2], which is collected using 6 surrounded-view cameras that cover the full  $360^{\circ}$  field of view around the ego-vehicle. It contains 700 scenes for training and 150 scenes for validation. We resize the original image from  $1600 \times 900$  to  $800 \times 448$  for training. In our work, we use the occupancy label with a resolution of  $0.2m$  from OpenOccupancy [27] as condition input, while the benchmark of occupancy prediction uses a resolution of  $0.4m$  from Occ3D [24] dataset for its popularity.

Networks We use Stable Diffusion [21] v2.1 checkpoint as initialization and only train occupancy encoder, cross-view attention. We additionally add cross-frame attention if in video experiments. We adopt FB-Occ [15] as the target model for occupancy prediction for its SOTA performance in this task. The pretrained checkpoint of the network is obtained from their official repository. Since FB-Occ predicts occupancy using only single frame images, we thus train SyntheOcc without cross-frame attention in related experiments. For video generation, we provide experimental results in appendix.

Metrics We use Frechet Inception Distance (FID) [6] to measure the perceptual quality of generated images, and use mIoU to measure the precision of occupancy prediction.

Hyperparameters We set  $D = 256$ ,  $d_{min} = 0$  and  $d_{max} = 50$ . The depth resolution of MPI is thus higher than occupancy voxel. We train our model in 6 epochs with batch size  $= 8$ . The learning rate is set at  $2e^{-5}$ . The training phase takes around 1 day using 8 NVIDIA A100 80G GPUs. We use UniPC scheduler [42] with the classifier-free guidance (CFG) [7] that is set as 7.0. During inference, we use 20 denoising steps for dataset generation.

Baselines We compare our method with prior methods in Tab. 1. ControlNet denotes we train a ControlNet using an RGB semantic mask as the condition. ControlNet+depth denotes we add a depth channel after the semantic mask to provide 2.5D depth information. The depth map rendered by occupancy is normalized to [0-255] to accommodate the RGB value. The ControlNet+depth can be regarded as a degradation of SytheOcc which is reduced to a single plane. Then we evaluate MagicDrive since it is the only open-sourced method in this area. MagicDrive separately encodes foreground and background using prompt and BEV layout. Furthermore, we evaluate the image quality (FID [6]) of our method in Tab. 2. Compared with prior methods, we use a unified 3D representation that seamlessly handles foreground and background, surpassing them by a large margin.

# 4.2 Qualitative Results

High-level Control using Prompt In Fig. 5 (c,d) and Fig. 6 (c), we demonstrate the capability to employ user-defined prompts to generate images with specific weather conditions and high-level

![](images/b5f6038ee07c9edc7d089e9433a8fa01e9c8b9ca8d5db085c758c6b94943a53d.jpg)  
(a0) Fusion of MPI

![](images/9bed588a88f7ecd11cb969642a8c8851ab1403971f3ed359a43aa9dff2817456.jpg)  
(a1) Occupancy in 3D space

![](images/0253e8374aa6952d628e035fd64187c5dd3c7eed8a4c9f636964037069c7cc46.jpg)  
(a2) ControlNet generation

![](images/50ca8489434bff0c9b815ec0215c0d590db99017a70b7980703f0328fa146d27.jpg)

![](images/7761000f3fb6e258df3c0ad03a7fa956caf7d2f2c95ff5d6c35aa58a439085a2.jpg)

![](images/580e368cb6fb53fef62907daaa7e59f21721e18d1f8a0ae7f40c3a7bea394bbe.jpg)  
(b) A scene with human

![](images/a1c970b956399882f14c4b4aaebe6c765ef7e6531f838a5ee7090b5487dbf671.jpg)

![](images/8a663ed8e6a618b9fad3d3d23f6a492c6e8e41c08b4b9424c50a5f4cc35ad5f7.jpg)  
(d) A scene with hinged-articulated trucks  
Figure 6: Top row: Comparison with ControlNet. We achieve a precise alignment between conditional labels and synthesized images, while ControlNet generates objects with incorrect pose due to ambiguous 2D condition. Mid and Bottom row: Visualizations of geometry-controlled image generation. We can faithfully generate objects with the desired topology in a specific 3D position.

![](images/3d6d9dc62f7a43303d2a8ba5cfc60b5faca3a8a28e9e74cd172d16edd791bf3b.jpg)  
(a3) Ours generation  
(e) A scene with excavator use geometry control

![](images/22ac902225af1390ea7cfd24a1b966bcbdbbc5a28daaa8d38aac08bee235c3a7.jpg)  
(a4) Ground truth

style. Although the nuScenes dataset doesn't contain rare weather images like snow and sandstorms, our method successfully conveys prior knowledge pretrained from stable diffusion to our scenes. Compared with visualization results in prior work like Fig. 8 of MagicDrive, our method shows better alignment with the text prompt, demonstrating the cross-domain generalization ability of our method.

3D Geometric Control Our flexible framework enables us to create novel scenes by manipulating voxels as displayed in Fig. 1 and Fig. 3. Basically, we can edit the occupied state and semantics of every voxel in our scenes for generation. We highlight that we can create a hinged-articulated truck and an excavator as shown in Fig. 6 (d,e). The generated excavator image exhibits a remarkable alignment with the input occupancy that is delineated by a black outline.

Long-tailed Scene Generation The flexibility of 3D semantic MPI has conferred significant advantages upon our approach. In the following, we create long-tail scenes that rarely occur in our real world for evaluation. In Fig. 1, we show that we manually add parallel traffic cones in front of the ego vehicle. This scene has never happened in the training dataset, but our geometric controllability provides us the capability to create such data. We then use the created scene to test autonomous driving systems such as end-to-end planner VAD [9] to validate its effectiveness. In this case, VAD successfully predicts correct waypoints with the high-level command 'turn left'. Moreover, in appendix Sec. B, we generate long-tailed scenes with extreme weather such as snow and sandstorms, and evaluate perception model on it to examine its generalizability of rare weather.

Comparison with Baselines In Fig. 6 (a), we visualize a comparison with ControlNet. We find that ControlNet struggles to distinguish the overlapping instances in 2D-pixel space. This leads to the two parked cars being merged into a single car with incorrect pose. In contrast, our 3D semantic MPs contain more than 2D semantic mask, but also account for complete scene geometry with occluded

Table 2: Comparison of FID with previous methods on the nuScenes dataset.  

<table><tr><td>Method</td><td>Condition Type</td><td>FID</td></tr><tr><td>BEVGen [23]</td><td>BEV map</td><td>25.54</td></tr><tr><td>BEVControl [34]</td><td>BEV map</td><td>24.85</td></tr><tr><td>DriveDreamer [26]</td><td>Box + FoV map</td><td>52.60</td></tr><tr><td>MagicDrive [5]</td><td>Box + BEV map</td><td>16.20</td></tr><tr><td>Panacea [30]</td><td>Box + FoV map</td><td>16.96</td></tr><tr><td>Ours</td><td>3D Semantic MPI</td><td>14.75</td></tr></table>

Table 3: Ablation of different designs of the MPI encoder and reweighing methods.  

<table><tr><td>MPI Encoder</td><td colspan="3">Reweighing Method</td><td>Metric</td></tr><tr><td>design</td><td>Progressive</td><td>Depth</td><td>CBGS</td><td>mIoU</td></tr><tr><td>3×3</td><td>-</td><td>-</td><td>-</td><td>21.96</td></tr><tr><td>1×1</td><td>-</td><td>-</td><td>-</td><td>23.05</td></tr><tr><td>1×1</td><td>✓</td><td>-</td><td>-</td><td>23.63</td></tr><tr><td>1×1</td><td>✓</td><td>✓</td><td>-</td><td>24.40</td></tr><tr><td>1×1</td><td>✓</td><td>✓</td><td>✓</td><td>25.50</td></tr></table>

parts. Together with our proposed MPI encoder and reweighing strategy, our framework yields a realistic image generation with high-quality label alignment. More comparison is provided in Sec. D.

# 4.3 Quantitative Results

Recognizability, Realism and Controllability Evaluation To evaluate whether our generated images aligned with given annotations, we provide Gen experiment in Tab. 1. Using the annotation of val set, we synthesize a copy of val set's images, then use perception model trained on real training set to perform evaluation. The performance will be more effective as it is close to the oracle performance. We find that local method (ControlNet) performs better than global method (MagicDrive). Furthermore, SytheOcc generalizes the locality for 3D conditioning to yield better performance.

Data Augmentation for 3D Occupancy Prediction Notably, we conduct experiments using our synthesized dataset to enhance the real training set in Tab. 1. We first use the occupancy labels from training set to create a synthetic training set. Then we modify the loading pipeline in perception model to randomly sample images from real dataset or synthetic dataset and train network from scratch. Therefore, our approach preserves the inherent training dynamics of the neural network by solely modifying the training images, without any alteration to the number of training iterations or epochs. As MagicDrive-Aug exhibits numerical overflow when training FB-Occ, which may attributed to unsatisfactory recognizability, we have to omit it and only provide MagicDrive-Gen experiments.

As shown in Tab. 1, where SytheOcc-Aug denotes the augmentation experiments using our generated dataset, shows a satisfactory improvement over the prior state of the art. We emphasize that surpassing the performance of the original dataset is not the primary objective of our work; rather, it is an ancillary benefit that emerges from our framework for geometry-controlled generation.

Ablations In Tab. 3, we present ablation studies across several design spaces of our model, analogous to the Gen experiment in Tab. 1. We find that our designed MPI encoder of  $1 \times 1$  conv have significant improvement when compared to the conventional  $3 \times 3$  conv approach. Besides, our proposed three types of reweighing methods demonstrate a consistent improvement over the baseline. As a result, the improved image quality and label alignment enable higher precision in downstream tasks.

# 5 Limitation and Broader Impacts

Layout Generation Our method is restricted in a conditional generation framework that should have a conditional input at first. Our condition signal is from the original dataset annotation. Thus most of the augmented data is generated using the same occupancy layout, or with minimal human editing. Future research can incorporate the recent research [10, 17, 32, 40] that generates occupancy descriptions of the scenes to synthesize images with novel occupancy layouts.

Closed-loop Simulation Given the underlying diverse and controllable image generation of our method, it would be advantageous and valuable to extend our work to a broader domain such as closed-loop simulation [16, 38], to enable high-fidelity autonomous systems testing. This line of work can be conducted by utilizing motion conditions to generate future frames as in world model [17, 28, 36], or by explicitly modeling scene graph as in the case of UniSim [20, 38] and NeuroNCAP [16].

Long-tailed Scene Generation In this paper, we only investigate a limited number of long-tailed scene generation and corner case evaluations such as rare layout in Fig. 1 and extreme weather in Sec. B. Future work can extend our framework to (i) Synthesize more samples for tail classes to boost performance. (ii) Generate or replicate large-scale databases of corner cases [11] for robust perception.

# 6 Conclusion

In this paper, we propose SytheOcc, an innovative image generation framework that is empowered with geometry-controlled capabilities using occupancy. We introduce a novel 3D representation, 3D semantic MPs, to address the critical challenge of how to efficiently encode occupancy. This representation not only preserves the authentic and complete 3D geometry details with semantics, but also provides a spatial-align feature representation for 2D diffusion models. With this property, our method enjoys photorealistic appearances and fine-grained 3D controllability, serves as a generative data engine to enable a broad range of applications. Extensive experiments demonstrate that our synthetic data facilitate the training for perception models on occupancy prediction, and provide valuable corner case evaluation in a simulated world.

# References

[1] Abien Fred Agarap. Deep learning using rectified linear units (relu). arXiv preprint arXiv:1803.08375, 2018.5  
[2] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In CVPR, 2020. 7  
[3] Kai Chen, Enze Xie, Zhe Chen, Lanqing Hong, Zhenguo Li, and Dit-Yan Yeung. Integrating geometric control into text-to-image diffusion models for high-quality detection data generation via text prompt. arXiv preprint arXiv:2306.04607, 2023. 3, 6  
[4] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384, 2023. 12  
[5] Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. In ICLR, 2024. 1, 3, 5, 8, 14  
[6] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 7  
[7] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint:2207.12598, 2022. 7  
[8] Lukas Hollein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting textured 3d meshes from 2d text-to-image models. In ICCV, 2023. 12  
[9] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In ICCV, 2023. 2, 8, 12, 13  
[10] Jumin Lee, Sebin Lee, Changho Jo, Woobin Im, Juhyeong Seon, and Sung-Eui Yoon. Semcity: Semantic scene generation with triplane diffusion. arXiv preprint arXiv:2403.07773, 2024. 9  
[11] Kaican Li, Kai Chen, Haoyu Wang, Lanqing Hong, Chaoqiang Ye, Jianhua Han, Yukuai Chen, Wei Zhang, Chunjing Xu, Dit-Yan Yeung, et al. Coda: A real-world road corner case dataset for object detection in autonomous driving. In ECCV, 2022. 9  
[12] Leheng Li, Qing Lian, Luozhou Wang, Ningning Ma, and Ying-Cong Chen. Lift3d: Synthesize 3d training data by lifting 2d gan to 3d generative radiance field. In CVPR, 2023. 1, 3  
[13] Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdiffusion: Layout-guided multi-view driving scene video generation with latent diffusion model. arXiv preprint arXiv:2310.07771, 2023. 3  
[14] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In CVPR, 2023. 2  
[15] Zhiqi Li, Zhiding Yu, David Austin, Mingsheng Fang, Shiyi Lan, Jan Kautz, and Jose M Alvarez. Fb-occ: 3d occupancy prediction based on forward-backward view transformation. arXiv preprint arXiv:2307.01492, 2023. 7  
[16] William Ljungbergh, Adam Tonderski, Joakim Johnander, Holger Caesar, Kalle Åström, Michael Felsberg, and Christoffer Petersson. Neuroncap: Photorealistic closed-loop safety testing for autonomous driving. arXiv preprint arXiv:2404.07762, 2024. 9  
[17] Jiachen Lu, Ze Huang, Jiahui Zhang, Zeyu Yang, and Li Zhang. Wovogen: World volume-aware diffusion for controllable multi-camera driving scene generation. arXiv preprint arXiv:2312.02934, 2023. 3, 9  
[18] Jianbiao Mei, Yu Yang, Mengmeng Wang, Tianxin Huang, Xuemeng Yang, and Yong Liu. Ssc-rs: Elevate lidar semantic scene completion with representation separation and bev fusion. In IROS, 2023. 3  
[19] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 1  
[20] Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, and Felix Heide. Neural scene graphs for dynamic scenes. In CVPR, 2021. 9  
[21] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3, 7  
[22] Liangchen Song, Liangliang Cao, Hongyu Xu, Kai Kang, Feng Tang, Junsong Yuan, and Yang Zhao. Roomdreamer: Text-driven 3d indoor scene synthesis with coherent geometry and texture. arXiv preprint arXiv:2305.11337, 2023. 12  
[23] Alexander Swerdlow, Runsheng Xu, and Bolei Zhou. Street-view image generation from a bird's-eye view layout. IEEE RAL, 2024. 1, 3, 8

[24] Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, and Hang Zhao. Occ3d: A large-scale 3d occupancy prediction benchmark for autonomous driving. NeurIPS, 2024. 1, 3, 7  
[25] Wenwen Tong, Chonghao Sima, Tai Wang, Li Chen, Silei Wu, Hanming Deng, Yi Gu, Lewei Lu, Ping Luo, Dahua Lin, et al. Scene as occupancy. In ICCV, 2023. 3  
[26] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, and Jiwen Lu. Drivedreamer: Towards real-world-driven world models for autonomous driving. arXiv preprint arXiv:2309.09777, 2023. 3, 8  
[27] Xiaofeng Wang, Zheng Zhu, Wenbo Xu, Yunpeng Zhang, Yi Wei, Xu Chi, Yun Ye, Dalong Du, Jiwen Lu, and Xingang Wang. Openoccupancy: A large scale benchmark for surrounding semantic occupancy perception. In ICCV, 2023. 1, 3, 7  
[28] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. arXiv preprint arXiv:2311.17918, 2023. 3, 5, 9  
[29] Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, and Jiwen Lu. Surroundocc: Multi-camera 3d occupancy prediction for autonomous driving. In ICCV, 2023. 3  
[30] Yuqing Wen, Yucheng Zhao, Yingfei Liu, Fan Jia, Yanhui Wang, Chong Luo, Chi Zhang, Tiancai Wang, Xiaoyan Sun, and Xiangyu Zhang. Panacea: Panoramic and controllable video generation for autonomous driving. arXiv preprint arXiv:2311.16813, 2023. 1, 3, 5, 8  
[31] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, 2023. 5, 14  
[32] Zhennan Wu, Yang Li, Han Yan, Taizhang Shang, Weixuan Sun, Senbo Wang, Ruikai Cui, Weizhe Liu, Hiroyuki Sato, Hongdong Li, et al. Blockfusion: Expandable 3d scene generation using latent tri-plane extrapolation. arXiv preprint arXiv:2401.17053, 2024. 9  
[33] Xu Yan, Jiantao Gao, Jie Li, Ruimao Zhang, Zhen Li, Rui Huang, and Shuguang Cui. Sparse single sweep lidar point cloud segmentation via learning contextual shape priors from scene completion. In AAAI, 2021. 3  
[34] Kairui Yang, Enhui Ma, Jibin Peng, Qing Guo, Di Lin, and Kaicheng Yu. Bevcontrol: Accurately controlling street-view elements with multi-perspective consistency via bev sketch layout. arXiv preprint arXiv:2308.01661, 2023. 3, 8  
[35] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 2023. 3  
[36] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Thompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023. 9  
[37] Yuanbo Yang, Yifei Yang, Hanlei Guo, Rong Xiong, Yue Wang, and Yiyi Liao. Urbangiraffe: Representing urban scenes as compositional generative neural feature fields. In ICCV, 2023. 3  
[38] Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, and Raquel Urtasun. Unisim: A neural closed-loop sensor simulator. In CVPR, 2023. 9  
[39] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. Wonderjourney: Going from anywhere to everywhere. arXiv preprint arXiv:2312.03884, 2023. 12  
[40] Junge Zhang, Qihang Zhang, Li Zhang, Ramana Rao Kompella, Gaowen Liu, and Bolei Zhou. Urban scene diffusion through semantic occupancy map. arXiv preprint arXiv:2403.11697, 2024. 9  
[41] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2, 3, 5  
[42] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. NeurIPS, 2023. 7  
[43] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018. 2, 4  
[44] Benjin Zhu, Zhengkai Jiang, Xiangxin Zhou, Zeming Li, and Gang Yu. Class-balanced grouping and sampling for point cloud 3d object detection. arXiv preprint arXiv:1908.09492, 2019. 6
