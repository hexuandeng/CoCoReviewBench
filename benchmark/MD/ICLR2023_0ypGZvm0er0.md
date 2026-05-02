# VIEW SYNTHESIS WITH SCULPTED NEURAL POINTS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We address the task of view synthesis, generating novel views of a scene given a set of images as input. In many recent works such as NeRF (17), the scene geometry is parameterized using neural implicit representations (MLPs). Implicit neural representations have achieved impressive visual quality but have drawbacks in computational efficiency. In this work, we propose a new approach that performs view synthesis using point clouds. It is the first point-based method that achieves better visual quality than NeRF while being  $100 \times$  faster in rendering speed. Our approach builds on existing works on differentiable point-based rendering but introduces a novel technique we call "Sculpted Neural Points (SNP)", which significantly improves the robustness to errors and holes in the reconstructed point cloud. We further propose to use view-dependent point features based on spherical harmonics to capture non-Lambertian surfaces, and new designs in the point-based rendering pipeline that further boost the performance. Finally, we show that our system supports fine-grained scene editing in a user-friendly way.

# 1 INTRODUCTION

We address the task of view synthesis: generating novel views of a scene given a set of images as input. It has important applications including augmented and virtual reality. View synthesis can be posed as the task of recovering from existing images a rendering function that maps an arbitrary viewpoint into an image. In many recent works, this rendering function is parameterized using neural implicit representations of scene geometry (17; 44; 19; 2; 20; 5; 18). In particular, NeRF (17) represents 3D geometry as a neural network that maps a 3D coordinate to a scalar indicating occupancy.

![](images/dcaaf39ec30692dd226024da883d5d624b4b4881798b84a08921002076fabfa2.jpg)  
Figure 1: The overall pipeline of the Sculpted Neural Points. We first use an MVS network to extract a point cloud. We then sculpt the point cloud by pruning (blue points) and adding (red points). The featurized point cloud finally passes through a differentiable rendering module to produce the image.

Implicit neural representations have achieved impressive visual quality but are typically computationally inefficient. To render a single pixel, NeRF needs to evaluate the neural network at hundreds of 3D points along the ray, which is wasteful because most of the 3D spaces are unoccupied. NeRF's implicit representation also makes it inflexible for scene editing operations such as deformation, which is important for downstream applications including augmented reality and video games. Several works enable NeRF to do scene editing (14; 13; 37; 22), but either the way of editing is highly constrained, or images captured under all desired object poses are required.

On the other hand, this limitation is easily overcome by explicit representations such as meshes or point clouds. To rasterize a mesh or a point cloud, no computation is wasted on unoccupied 3D spaces. Scene editing such as composition and deformation is also straightforward. Moreover, rasterizing meshes or point clouds is a mature technology already widely deployed in the industry for movies and video games, capable of producing real-time performance and high realism.

An intriguing question is whether we can achieve state-of-the-art visual quality by using explicit representations such as point clouds. The basic framework of point-based neural rendering is to represent the scene as a featurized point cloud, which is reconstructed through a multiview stereo (MVS) system. The features are learned by maximizing photoconsistency on the input images via differentiable rendering. Although this framework has been studied in several recent works (1; 34; 10), the overall rendering quality still lags behind NeRF, mainly due to the ghosting effects and blurriness caused by the errors in geometry.

Our approach adopts this basic framework but introduces a novel technique we call "Sculpted Neural Points (SNP)", which significantly improves the robustness to the errors and holes in the reconstructed point cloud. The idea is to "sculpt" the initial point cloud reconstructed by the MVS system. In particular, we remove existing points and add additional points to improve the photo-consistency of the renders against input images. These sculpting decisions are discrete in nature, but are tightly coupled with gradient-based optimization of the continuous per-point features.

We further propose a few novel designs in the point-based rendering pipeline that boost the performance. We use spherical harmonics (SH) in high-dimensional point feature space to capture non-Lambertian visual effects, which is faster and better than using MLPs. Inspired by Dropout (31), we propose a point dropout layer that significantly improves the generalization to novel views. Last but not least, we find that it is essential to not use any normalization layers in the U-Net.

Compared to previous works that use point cloud-based representation, ours is the first model that achieves better rendering quality than NeRF, while being  $100 \times$  faster in rendering, and reducing the training time by  $66\%$ . We evaluate our method on common benchmarks including DTU (7), LLFF (16), NeRF-Synthetic (17), and Tanks&Temples (9), and our method shows better or comparable performance against all baselines. We encourage the reader to watch the supplementary video to compare our methods against baselines and see the effectiveness of point sculpting.

Finally, we show that our model allows fine-grained scene editing in a user-friendly way. Compared to previous works that can only do object-level composition (14; 43; 38) or require a special user interface (13), our point-based representation inherently supports editing at a finer resolution, and users can use existing graphics toolbox to edit the point cloud without any custom changes.

The main contributions of this paper are three-fold: 1) We propose a novel point-based approach to view synthesis, "Sculpted Neural Points", a technique that is key to achieving high quality and view-consistent output; 2) We demonstrate, for the first time, that a point-based method can achieve better visual quality than NeRF while being  $100 \times$  faster in rendering. 3) We propose several improvements to the point-based rendering pipeline that significantly boost the visual quality.

# 2 RELATED WORK

Methods for view synthesis can be categorized based on how they represent the scene geometry.

View Synthesis with Implicit Representations NeRF (17) uses a neural network to map a 3D spatial location to volume density. To render a pixel, the neural network needs to be repeatedly evaluated along the ray, making rendering computationally expensive. Followup works on NeRF (44; 43; 19) focus on improving the speed or the generalization ability to new scenes or with a limited number of reference views. Our method does not use an implicit representation; instead, we

explicitly represent the scene geometry using a point cloud, which allows much faster rendering, as well as easy and flexible scene editing.

View Synthesis with Voxels NV (14) first combines neural volume rendering with voxel grids, and NSVF (11) proposes to construct a progressive pruning process to obtain fine-grained and sparse voxels. Yu et al. (42) further combines the voxel representations with spherical harmonics, achieving a very fast training speed. To apply neural rendering, voxel-based methods generally need to traverse the entire voxel space along each ray, whereas with our point-based representation we only rasterize the points on the surfaces. The complexity and memory cost grows cubically for voxels as resolution increases while only quadratically for our method.

View Synthesis with Point Clouds To use point clouds for view synthesis, existing approaches typically use an external multiview stereo system to reconstruct a point cloud from the input images, and then optimize the features and 3D positions of each point through differentiable rendering.

NPBG (1) is the first work to combine neural rendering with point clouds; it uses a featurized point cloud to represent the scene, and rasterizes with one-pixel point splats at multiple scales followed by a post-processing U-Net to fill the holes. SynSin (34) proposes a soft rasterization pipeline that allows better gradient flow and produces smoother results. Our method achieves significantly better visual quality compared to them. Pulsar (10) uses featurized spheres to represent the scene, and proposes a very efficient soft rasterizer that can rasterize millions of spheres in less than 100 milliseconds. Pulsar qualitatively shows that geometry reconstruction can be done through its differentiable rendering system, but has shown no quantitative results on view synthesis. We adopt Pulsar as our rendering backbone.

There are a few concurrent works using point cloud representations. NPBG++ (23) focuses on lifting the requirement of per-scene optimization. It does not revise the geometry and is thus more sensitive to the point cloud quality compared to ours. It also proposes to use view-dependent point features, but parameterized as MLP rather than SH as we do. Point-NeRF (35) uses a featurized point cloud to represent the scene geometry, but renders with volume rendering instead of point rasterization. ADOP (26) mainly focuses on unbounded outdoor scenes with large exposure changes among views.

To revise the initial point cloud provided by an MVS system, existing differential renderers compute gradients with respect to the 3D coordinates of each point. Pulsar (10) approximates the gradient by modeling points as spheres with a certain radius, with the color blending weights changing smoothly with respect to the distance of the camera ray to the sphere center. ADOP (26) instead computes the partial derivatives of the photometric loss with respect to the point positions by taking the finite difference in the pixel space.

Our method builds upon existing techniques of differentiable point-based rendering but differs substantially in how we revise the initial point cloud given by an MVS system. Although we find prior methods capable of local revision around the existing points by adjusting their locations using the gradients, such local operations, however, cannot fill large holes or add new points in empty spaces far away. In contrast, our sculpting technique is global. It does not use gradients and can add new points in locations arbitrarily far away from existing points.

# 3 APPROACH OVERVIEW

An overview of our approach is illustrated in Fig. 1. The input is a set of  $H \times W$  RGB images  $\{I_1, \ldots, I_m\}$  of  $m$  reference views and their corresponding intrinsics and extrinsics camera parameters,  $\{C_1, \ldots, C_m\}$ . We define the camera projection function  $\Pi$  and its inverse  $\Pi^{-1}$  as follows:

$$
\Pi (P, C) := \left[ K _ {C} \left(R _ {C} P + t _ {C}\right) \right] ^ {\downarrow}; \Pi^ {- 1} (p, d _ {p}, C) := R _ {c} ^ {- 1} \left(K _ {C} ^ {- 1} d _ {p} \left[ \begin{array}{l} p \\ 1 \end{array} \right] - t _ {c}\right) \tag {1}
$$

where  $K_{C}, R_{C}$  and  $t_{C}$  are the intrinsics, rotation, and translation of camera  $C$ .  $P \in \mathbb{R}^{3}$  is a 3D point and  $p \in \mathbb{R}^{2}$  is its 2D projection in the pixel space with depth  $d_{p}$ . Further,  $([X,Y,Z]^{T})^{\downarrow}$  is defined as  $([X / Z, Y / Z]^{T})$ . Our approach consists of three main components: point cloud reconstruction, point cloud sculpting, and differentiable rendering. In this section, we describe the backbone of our approach with only point cloud reconstruction and differentiable rendering, and leave point cloud sculpting to Sec. 4.

# 3.1 POINT CLOUD RECONSTRUCTION

We use an MVS network (15) to extract a dense depth map  $\{D_1, \ldots, D_m\} \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4}}$  for each of the reference views. Each depth map is un-projected into a set of 3D points by applying the inverse projection in Eqn. 1. The points from all depth maps are combined, without any filtering, to form a larger set of original 3D points  $P_o = \{p_1, \ldots, p_N\}$ . We associate point  $p_i \in \mathbb{R}^3$  with a learnable  $K$ -dimensional feature vector  $f_i \in \mathbb{R}^K$  and a learnable scalar  $o_i \in [0,1]$  representing its opacity.

# 3.2 DIFFERENTIABLE RENDERING

Given a featurized 3D point cloud and a target view, we use a differentiable rendering function with learnable parameters to map the point cloud into an RGB image. For each scene individually, we learn the parameters of this rendering function together with the point features through gradient descent to minimize photometric errors against the input images.

Spherical Harmonic Point Feature We use the spherical harmonics (SH) functions to model the view-dependent effects. Recently Yu et al. (43; 42) brings up the attention of using SH in neural rendering. Unlike previous works that use SH directly in RGB space, we propose to use SH in high-dimensional feature space, where we model each element of a vector with a set of SH coefficients.

Specifically, we use the SH basis up to degree 2, which yields 9 basis in total. For a 3D point  $p_i$ , the SH layer takes its feature vector  $f_i$  and a target view direction  $v$  as input, and outputs a feature vector  $s_i \in \mathbb{R}^{\frac{K}{9}}$ . Note that evaluating SH functions is cheap as it avoids matrix multiplication operations. We find that it leads to better performance and faster rendering speed compared to the MLP parameterization used in NeRF (17), as shown in Sec. 5.3.

Differentiable Soft Rasterization We use soft rasterization proposed in (12; 10) to convert the view-dependent features into a 2D feature map  $F$  given a target view. Soft rasterization blends the features of multiple points hit by a camera ray with weights depending on their depth and opacity values. We refer the readers to Pulsar (10) for details.

Note that in addition to updating  $f_{i}$ , we also compute the gradient of the photometric loss w.r.t. the point positions  $p_i$  and opacity  $o_i$ , and optimize them through gradient descent, following (10). Experiment results show that such gradient-based refinement helps improve fine geometric details.

Point Dropout Layer We find that the existing point rendering pipeline is prone to over-fitting, i.e., the image quality on test views is much worse than on training views. The reasons are two-fold: 1. The "training set" for view synthesis consists of only tens of images, and there are barely any data augmentation techniques that can be applied except random cropping. 2. Some points are covered by their neighbors in training views, but get visible in test views. The features for these points are not well-optimized. The blending weights are very low for these points when the rasterizer is "soft".

To resolve the above issue, we propose to use a "Point Dropout Layer" before rasterization. In each forward pass, we randomly select a subset of points to feed into the rasterizer, whose size depends on the dropout rate  $p_d$ . Note that at inference time, we cannot simply rasterize all points and multiply the output by  $p_d$  as in the neural network (NN) case (31), because the rasterization operation is nonlinear in contrast to the matrix multiplication in NN. Since it is impossible to traverse all subsets, we simply rasterize  $L$  multiple random subsets and average the output feature maps at inference time.

Intuitively, the point dropout layer allows us to train on an ensemble of point clouds, and give all points a chance to get optimized even if they are covered, and thus alleviating the over-fitting problem. As a side effect, we also gain a speed-up because fewer points get rasterized. Although the design is simple, this idea has not been explored in previous works, and our experiments show that it significantly improves the image quality on test views.

2-D Rendering without BatchNorm Given a target view, we convert the 2D feature map  $F$  into the RGB image  $I_{t}$  with a 2D ConvNet. We use a U-Net (25) with two downsampling layers and two upsampling layers, and optionally one more upsampling layer to produce high-resolution outputs.

Previous works (1; 23) directly use the original U-Net design with BatchNorm (6) layers, which we find unsuitable for view synthesis task for two reasons. First, the small training set size makes the estimation of the moving average in BatchNorm unstable. Second, the benefit of accelerated training is minimal since the network is shallow. Therefore, we use no normalization layer in our U-Net.

![](images/43d6cf4d4d416eb36411ae9bdbbc04e540f913bc66abf4dc3c9ebfe7cfc49fc1.jpg)  
(a)

![](images/2a3a93af96574438c22c4514f969c2b2d77fe191fa34b4c543af0c5933bd2e5a.jpg)  
(b)

![](images/fea1dc58b253d5cdaf153148af38ae8f7537747a21deb282e32c05868771f872.jpg)  
Figure 2: (a) The initial point cloud is incomplete and noisy. (b) The point cloud filtered with DCC (36) is accurate but incomplete. (c) The output of SNP. It removes most of the outliers and the points added (colored red) further fill the missing areas. (d) The closest training view, which shows what the actual geometry should look like. (e) The blue curve and dashed black curve represents the reconstructed surface and the actual surface, respectively. A set of candidates is generated along the ray from camera  $B$ .  $c$  is discarded because it occludes the existing surface in view  $A$ , and  $a$  is discarded because we only keep the closest  $M = 5$  non-occluding points. Only  $b$  is added.  
(c)

![](images/fdac94a957e7615fa47a78876231da47eb43fefc7c518c7e5e89852b64f76e88.jpg)  
(d)

![](images/83f4d5b4722d627a01145e5c7ca45e43ec07267207829282be92eaed4726aa71.jpg)  
(e)

# 4 POINT SCULPTING

The point clouds reconstructed from MVS usually contain many errors, even with the state-of-the-art MVS systems. The errors typically take the form of distorted or incomplete geometry. If we directly use such point clouds, the synthesized views will have poor visual quality with salient artifacts. To address this issue, we introduce a new technique we call "point sculpting". It has two steps, Point Pruning and Point Adding. The sculpting procedure and outputs are illustrated in Fig. 2.

# 4.1 POINT PRUNING

The MVS system we use produces a dense depth map for each input image. Like other depth-based systems (39; 36; 3), it adopts a fusion step that merges the depth maps from different views into a final point cloud. A geometry consistency check is often used to remove outlier points. Using the depth maps, the consistency check projects a pixel into another view, reprojects the corresponding point back, and sees if the original pixel is recovered up to a threshold. For example, COLMAP (28; 27) defines the consistency error  $\psi_{p}^{i,j}$  between view  $i$  and  $j$  for pixel  $p$  as:

$$
\boldsymbol {q} = \Pi \left(\Pi^ {- 1} \left(\boldsymbol {p}, d _ {\boldsymbol {p}} ^ {i}, C ^ {i}\right), C ^ {j}\right); \psi_ {\boldsymbol {p}} ^ {i, j} = \left\| \boldsymbol {p} - \Pi \left(\Pi^ {- 1} \left(\boldsymbol {q}, d _ {\boldsymbol {q}} ^ {j}, C ^ {j}\right), C ^ {i}\right) \right\| _ {2} \tag {2}
$$

where  $\pmb{q}$  is  $\pmb{p}$ 's corresponding point in view  $j$ . Yan et al. (36) further propose an improved version, Dynamic Consistency Checking (DCC), which achieved the state-of-the-art filtering results.

The main problem of this type of forward-backward consistency check is that it tends to be overaggressive in filtering out points, resulting highly incomplete geometry that is detrimental to view synthesis. In datasets such as DTU (7) and LLFF (16), many areas are only visible in a small number of views (or just a single view). Those areas can easily be filtered out by this check as no confident match could be found.

Therefore, we take the raw depth maps from the multiview stereo system and propose a new technique for our own consistency checking geared toward view synthesis. We check only the forward consistency to maximize completeness while still removing outliers. Formally, a pixel  $\pmb{p}$  in view  $i$  passes the check if and only if

$$
\bigcap_ {j = 1} ^ {m} \left[ D ^ {j} \left(\Pi^ {- 1} \left(\boldsymbol {p}, d _ {\boldsymbol {p}} ^ {i}, C ^ {i}\right)\right) \geq \delta_ {d} \cdot d _ {\boldsymbol {q}} ^ {j} \right] \tag {3}
$$

where  $\pmb{q}$  is  $\pmb{p}$ 's corresponding point in view  $j$  (same as in Eqn. 2),  $d_{\pmb{q}}^{j}$  is the predicted depth of  $q$  in view  $j$ ,  $D^{j}(\cdot)$  is the depth of a point in view  $j$  (the  $z$  value of a 3D point in camera  $j$ 's coordinate), and  $\delta_{d}$  is a hyperparameter controlling the relative tolerance.

Intuitively, our point pruning method keeps a point as long as it is not significantly closer than the original surface to any reference view camera. It filters out the points that are floating in the free space between the actual surface and the camera, which are likely to be outliers. It also keeps all points that are only visible in a small number of views. Although the position of such points may not be accurate, it is useful to keep them as candidates for further optimization.

# 4.2 POINT ADDING

As Fig. 2 shows, after pruning, the point cloud can have holes, either due to points being pruned or incorrect depth estimates (e.g. depth estimates that are close to infinity or zero). Previous works (10; 41) tackled this problem by performing gradient-based updates to the point locations. However, such updates are limited to local changes of existing points and are unable to recover large areas of missing geometry.

We thus introduce a technique to add new points to the pruned point cloud. The basic idea is to find a set of 3D points that, if added to the point cloud, could help minimize the photometric error after optimization of the point features. Note that these new points do not need to be perfect; they just need to be a superset of the ground truth geometry, because the extraneous points can get optimized through the subsequent gradient-based optimization. On the other hand, an excessive number of new points can lead to overfitting and slower rendering, so a good balance is needed. Our point adding algorithm consists of two steps:

- Optimizing with existing points: We optimize the features and opacity of the current points through gradient descent until convergence. For the  $i$ -th input image, we extract an error map between the rendered and ground-truth image:  $E_{i} = ||I_{i}^{gt} - I_{i}^{render}||_{1}$ . Note that we directly treat point features as RGB values during rasterization and use no U-Net in this step, because U-Net hallucination is harmful.

- Proposing new points: For a pixel  $(u, v)$  in an input view  $i$ , we check if its rendering error  $E_i(u, v)$  is bigger than a pre-defined threshold  $\delta_e$ . If so, we sample 3D points uniformly along the ray of the pixel within the bounds of the scene, and search for points that do not occlude any of the existing points in any of the input views. If multiple such points exist, we propose the closest  $M$  points, where  $M$  is a hyperparameter. We go through all pixels in all input views, collect all the proposed points, and add them to the existing point cloud.

The design of this algorithm builds upon the assumption that our rendering pipeline can approximate the radiance of each point arbitrarily well on the input images and that any high rendering error can only be caused by incomplete geometry, as those areas having no points covered can only take the default background color. Based on this assumption, we propose new points for pixels with high rendering errors, but exclude points that occlude the existing surface in other views. We propose up to  $M$  closest points and choose  $M = 5$  in our experiments to strike a good balance between geometry coverage and rendering speed.

We can alternate between gradient-based optimization and point adding for multiple rounds, but in practice we find one round of point adding to be sufficient for good results. We present the full details of the point adding algorithm in Appendix C.

# 5 EXPERIMENTS

# 5.1 EXPERIMENT DETAILS

Datasets We evaluate our method on DTU (7), LLFF (16; 17), NeRF's Realistic Synthetic  $360^{\circ}$  (17), and Tanks&Temples (9). The datasets we choose provide good coverage of both forward-facing and  $360^{\circ}$  scenes. We evaluate using PSNR, SSIM and LPIPS (47) metrics.

Baselines We compare our model with NeRF (17). On DTU and LLFF, we run the state-of-the-art point-based rendering methods NPBG (1) and SynSim (34) using the same external MVS system to reconstruct point clouds, with no pruning or adding. We additionally compare against other concurrent point-based methods including NPBG++ (23) and Point-NeRF (35).

Implementation Details We implement our method with PyTorch (21) and PyTorch3D (24). We experiment on a single RTX 3090 GPU. We optimize for 50,000 steps on each scene with a batch size of 1. We initialize all SH coefficients for each point as 0s and the point opacity as 1. We initialize the U-Net parameters randomly. We set the point radii to be a dataset-specific hyperparameter, which is the same for all points and fixed. The point dropout rate  $d_{p}$  is 0.5 for all experiments. See Appendix B for details of the MVS network and other implementations.

# 5.2 PRIMARY RESULTS

![](images/94b00b05d906508043a617e382e5b6426c307e9819c8095e36e41ba5b684a0cf.jpg)  
Figure 3: Qualitative comparison of our model v.s. baselines on the LLFF dataset.

Results on DTU The quantitative comparison is presented in Tab. 1. Results show that our model has better SSIM and LPIPS, and slightly worse PSNR compared to NeRF. We present the visualizations in Fig. 11, Appendix D. We claim to use LPIPS as the major quality metric, as we find that PSNR and SSIM may not reflect actual visual quality because they are highly sensitive to small pixel shifts (See Appendix A).

Table 1: Quantitative results on the DTU and the LLFF dataset. For all tables in this paper, we mark the best number in bold and the second-best number with underline.  

<table><tr><td rowspan="2">Method</td><td colspan="4">DTU</td><td colspan="4">LLFF</td></tr><tr><td>NPBG(1)</td><td>SynSin(34)</td><td>NeRF(17)</td><td>SNP (ours)</td><td>NPBG(1)</td><td>SynSin(34)</td><td>NeRF(17)</td><td>SNP (ours)</td></tr><tr><td>PSNR†</td><td>19.38</td><td>21.04</td><td>28.97</td><td>26.68</td><td>19.98</td><td>22.34</td><td>26.50</td><td>25.32</td></tr><tr><td>SSIM↑</td><td>0.652</td><td>0.714</td><td>0.846</td><td>0.884</td><td>0.624</td><td>0.705</td><td>0.811</td><td>0.817</td></tr><tr><td>LPIPS↓</td><td>0.412</td><td>0.337</td><td>0.266</td><td>0.156</td><td>0.454</td><td>0.351</td><td>0.250</td><td>0.229</td></tr></table>

Results on LLFF The quantitative results are shown in Tab. 1. Similar to DTU, our method achieves consistently better SSIM and LPIPS, while being slightly worse in PSNR. Qualitative comparisons are shown in Fig. 3. Compared to NeRF, our model can reconstruct very fine details. Our method also has significantly better visual quality compared to the two point-based baselines.

Results on Tanks&Temples We present the numbers in Tab. 2. All baselines numbers are copied from Point-NeRF (35). Our method achieves comparable quality as Point-NeRF, while being significantly better than other baselines. We present qualitative comparisons in Fig. 14, Appendix D.

Table 2: Quantitative results on Tanks&Temples.  

<table><tr><td rowspan="2">Method</td><td colspan="5">Tanks&amp;Temples</td></tr><tr><td>NV(14)</td><td>NeRF(17)</td><td>NSVF(11)</td><td>Point-NeRF(35)</td><td>SNP (ours)</td></tr><tr><td>PSNR↑</td><td>23.70</td><td>25.78</td><td>28.40</td><td>29.61</td><td>29.78</td></tr><tr><td>SSIM↑</td><td>0.848</td><td>0.864</td><td>0.900</td><td>0.954</td><td>0.942</td></tr><tr><td>LPIPSAlex↓</td><td>0.260</td><td>0.198</td><td>0.153</td><td>0.080</td><td>0.079</td></tr></table>

Results on NeRF-Synthetic Results are shown in Tab. 3. Our method achieves comparable performance to NeRF while being worse than Point-NeRF, which is also reflected in Fig. 4. Our method is better at capturing the reflective drum surfaces, while struggles with the microphone which has fine geometry. Our explanation is that our view-dependent point features are very expressive in modeling high-frequency textures, while our point cloud is not accurate enough in case of fine geometries.

Table 3: Quantitative results on the NeRF-Synthetic dataset. NPBG++ only presents results on the hotdog, ficus, and mic scenes. All other baseline numbers are copied from the Point-NeRF paper.  

<table><tr><td rowspan="2">Method</td><td colspan="4">NeRF-Synthetic (all 8 scenes)</td><td colspan="2">NeRF-Synthetic (3 scenes)</td></tr><tr><td>NPBG(1)</td><td>NeRF(17)</td><td>Point-NeRF(35)</td><td>SNP (ours)</td><td>NPBG++(23)</td><td>SNP (ours)</td></tr><tr><td>PSNR↑</td><td>24.56</td><td>31.01</td><td>33.31</td><td>27.47</td><td>28.67</td><td>29.16</td></tr><tr><td>SSIM↑</td><td>0.923</td><td>0.947</td><td>0.978</td><td>0.939</td><td>0.952</td><td>0.961</td></tr><tr><td>LPIPS↓</td><td>0.109</td><td>0.081</td><td>0.049</td><td>0.067</td><td>0.050</td><td>0.037</td></tr></table>

![](images/1d38705f2c825657a726d2b99db79f02366d4a17f3963132d9f1e8bb458b4c48.jpg)  
Figure 4: Qualitative comparison of our model v.s. baselines on the NeRF-Synthetic dataset.

# 5.3 ABLATION STUDIES

Table 4: Ablation studies on the DTU dataset.  

<table><tr><td></td><td>View-dependent Layer Latency</td><td>Num. Points</td><td>Dropout Rate</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Use DCC(36) Filtering</td><td>15ms</td><td>3.3M</td><td>50%</td><td>19.97</td><td>0.844</td><td>0.196</td></tr><tr><td>No Adding; No Pruning</td><td>15ms</td><td>4.2M</td><td>50%</td><td>25.06</td><td>0.836</td><td>0.201</td></tr><tr><td>No Adding</td><td>15ms</td><td>4.0M</td><td>50%</td><td>26.15</td><td>0.882</td><td>0.163</td></tr><tr><td>No Gradient-based Refine</td><td>15ms</td><td>4.4M</td><td>50%</td><td>26.52</td><td>0.880</td><td>0.157</td></tr><tr><td>No View Dependence</td><td>N/A</td><td>4.4M</td><td>50%</td><td>25.67</td><td>0.876</td><td>0.160</td></tr><tr><td>View Dependence w/ MLP</td><td>79ms</td><td>4.4M</td><td>50%</td><td>26.30</td><td>0.881</td><td>0.160</td></tr><tr><td>No Point Dropout</td><td>31ms</td><td>4.4M</td><td>0%</td><td>25.40</td><td>0.852</td><td>0.191</td></tr><tr><td>Low Dropout Rate</td><td>23ms</td><td>4.4M</td><td>25%</td><td>26.47</td><td>0.880</td><td>0.158</td></tr><tr><td>High Dropout Rate</td><td>8ms</td><td>4.4M</td><td>75%</td><td>26.46</td><td>0.883</td><td>0.157</td></tr><tr><td>BatchNorm(6) in UNet</td><td>15ms</td><td>4.4M</td><td>50%</td><td>25.19</td><td>0.857</td><td>0.171</td></tr><tr><td>InstanceNorm(32) in UNet</td><td>15ms</td><td>4.4M</td><td>50%</td><td>26.08</td><td>0.869</td><td>0.169</td></tr><tr><td>Complete Model</td><td>15ms</td><td>4.4M</td><td>50%</td><td>26.68</td><td>0.884</td><td>0.156</td></tr></table>

![](images/eabce3870aacd8115a84cc8ca884719b484fd9812df281183b1bd0ab477f1a1f.jpg)  
Figure 5: Qualitative comparison of point sculpting v.s. baselines on the DTU dataset.

We conduct ablation studies of our proposed designs. We show results in Tab. 4. In the  $1^{\text{st}}$  block, We compare with several baselines on point cloud refinement, including 1) the filtering algorithm DCC (36) which achieves SOTA performance on MVS, 2) using the raw MVS point cloud without any pruning or adding, 3) pruning with the proposed point pruning but no point adding, 4) using the same pointcloud as the complete model, but keep the point positions and opacity values fixed during gradient updates. Also see Fig. 5 for qualitative comparisons. Results show that all proposed geometry refinement components contribute to the final model. While point pruning contributes to sharper object boundaries near the head of the plush, point adding is especially helpful for filling large holes on the table in the rabbit scene. See Fig. 9 in Appendix C for visualizations of the point cloud generated by each method. We also encourage the readers to watch the supplementary video to see that our method achieves better view consistency compared to baselines.

The  $2^{\mathrm{nd}}$  block shows that using SH reduces the layer latency by  $82\%$  and improves the PSNR by 0.38dB compared to MLP. For the MLP baseline, we use a 2-layers MLP with 256 hidden units, which takes as input the concatenation of the point feature and the positional-encoded view direction, following NeRF. See also Fig. 12, Appendix D for visualization of the non-Lambertian effect learned

by our model. The  $3^{\mathrm{rd}}$  block shows that using dropout layer improves the PSNR by 1.28dB, and the model is not sensitive to the dropout rate. The  $4^{\mathrm{th}}$  block shows that not using any normalization layers improves the PSNR by 1.49dB compared to using BatchNorm, and by 0.60dB compared to using InstanceNorm.

# 5.4 SCENE EDITING

We show that our system supports, with high fidelity, scene editing operations such as scene composition, object deformation, and erasing. Results are shown in the inset figure on the right. Composition is achieved by first co-training two scenes with separate point features and a shared U-Net, then

![](images/4bd0c782438c4f4b28a76b421245eb0130eff35d7261097bb643248f997457db.jpg)  
Composition

![](images/bb0645bf0f55dae3e2a3bdd559b1f47d777205b3a8e9b6202ae1ae1b9d79cfb2.jpg)  
Deformation

![](images/bb0df3c1c6856beb37f2e6f0e8b40a3d7ebd66ebc9a4403d62ac03d301343d2e.jpg)  
Erasing

putting the points into the same scene at inference time. For deformation, we export the sculpted point cloud into MeshLab (4), where we manually select the moving part and its axis of rotation. For erasing, we filter out points based on their  $z$  coordinates.

Compared to existing neural rendering pipelines that support scene editing, our system has two main advantages: 1. Fine-grained editing: Previous works (14; 43; 38) use explicit representations like voxel grids, which are typically limited in resolution. Therefore, they can only achieve object-level operations such as composition. In comparison, we represent object surfaces densely with millions of points, so we can do fine-grained editing such as object deformation. 2. Ease of use: Previous works doing scene editing with NeRF either require a special interface to take user inputs (13), or a complex pipeline that uses meshes as an intermediate representation (45). In contrast, our point cloud representation is directly supported by nearly all graphics toolboxes such as MeshLab or Blender, which allows users to edit the scene intuitively without any specialized tool.

# 5.5 INFERENCE SPEED, TRAINING TIME, AND MODEL SIZE

We compare our model's inference speed, training time, and model size with a few baselines on the NeRF-Synthetic dataset, shown in Tab. 5. All speeds are benchmarked using an RTX 3090 GPU. Compared to NeRF, our model is more than  $100 \times$  faster in inference and requires only  $33\%$  training time. PlenOctrees (43) bakes the radiance field into a voxel-based cache, resulting in faster rendering speed but also significantly larger model size and longer training time. NPBG (1) achieves faster inference speed by their one-pixel point splats with the cost of worse visual quality. Finally, we are about  $25 \times$  faster than Point-NeRF (35) in rendering while other metrics are roughly the same.

Table 5: On NeRF-Synthetic Dataset, we comparison on model inference speed, training time, model size, and rendering quality (measured in LPIPS).  

<table><tr><td></td><td>NeRF(17)</td><td>PlenOctrees(43)</td><td>NPBG(1)</td><td>Point-NeRF(35)</td><td>SNP (ours)</td></tr><tr><td>Inference↑ (FPS)</td><td>0.053</td><td>127</td><td>20.3</td><td>0.192</td><td>5.06</td></tr><tr><td>Training↓ (Hours)</td><td>20</td><td>50</td><td>6.9</td><td>8.0</td><td>6.6</td></tr><tr><td>Model Size↓</td><td>14MB</td><td>1.9GB</td><td>31MB</td><td>106MB</td><td>290MB</td></tr><tr><td>LPIPS↓</td><td>0.081</td><td>0.053</td><td>0.109</td><td>0.049</td><td>0.067</td></tr></table>

# 6 DISCUSSIONS AND LIMITATIONS

There are a few limitations that need to be addressed in future work: 1) MVS dependency. Although the proposed point sculpting can partly solve this problem, the performance of the system still depends heavily on the MVS quality. That said, as MVS systems continue to improve, we do not see this as a fundamental limitation in the long run. 2) View Consistency. Our system has a 2D U-Net and is thus only approximately 3D consistent. Especially when viewed in videos, some background areas have flickering effects due to the hallucination of U-Net. Doing away with a 2D post-processing network is a future direction. 3) Far-away background. Our current system cannot deal with outdoor scenes with arbitrarily far-away objects (e.g. the sky or clouds). Using a spherical environment map as in (46; 26) could resolve this problem.

# Reproducibility Statement

We provide the full implementation details of the algorithms and datasets we use in Appendix. B and Appendix. C. We also include an anonymous version of our source code in the supplementary materials. We will release all data pre-processing code, the model checkpoints, and the training code for all experiments in this paper after the reviewing process.

# REFERENCES

[1] Aliev, K.A., Sevastopolsky, A., Kolos, M., Ulyanov, D., Lempitsky, V.: Neural point-based graphics. In: Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XXII 16. pp. 696-712. Springer (2020)  
[2] Chen, A., Xu, Z., Zhao, F., Zhang, X., Xiang, F., Yu, J., Su, H.: Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14124-14133 (2021)  
[3] Chen, R., Han, S., Xu, J., Su, H.: Visibility-aware point-based multi-view stereo network. IEEE transactions on pattern analysis and machine intelligence 43(10), 3695-3708 (2020)  
[4] Cignoni, P., Callieri, M., Corsini, M., Dellepiane, M., Ganovelli, F., Ranzuglia, G., et al.: Meshlab: an open-source mesh processing tool. In: Eurographics Italian chapter conference. vol. 2008, pp. 129-136. Salerno, Italy (2008)  
[5] Garbin, S.J., Kowalski, M., Johnson, M., Shotton, J., Valentin, J.: Fastnerf: High-fidelity neural rendering at 200fps. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14346-14355 (2021)  
[6] Ioffe, S., Szegedy, C.: Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: International conference on machine learning, pp. 448-456. PMLR (2015)  
[7] Jensen, R., Dahl, A., Vogiatzis, G., Tola, E., Aanæs, H.: Large scale multi-view stereopsis evaluation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 406-413 (2014)  
[8] Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)  
[9] Knapitsch, A., Park, J., Zhou, Q.Y., Koltun, V.: Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG) 36(4), 1-13 (2017)  
[10] Lassner, C., Zollhofer, M.: Pulsar: Efficient sphere-based neural rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1440-1449 (2021)  
[11] Liu, L., Gu, J., Zaw Lin, K., Chua, T.S., Theobalt, C.: Neural sparse voxel fields. Advances in Neural Information Processing Systems 33, 15651-15663 (2020)  
[12] Liu, S., Li, T., Chen, W., Li, H.: Soft rasterizer: A differentiable renderer for image-based 3d reasoning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7708-7717 (2019)  
[13] Liu, S., Zhang, X., Zhang, Z., Zhang, R., Zhu, J.Y., Russell, B.: Editing conditional radiance fields. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5773-5783 (2021)  
[14] Lombardi, S., Simon, T., Saragih, J., Schwartz, G., Lehrmann, A., Sheikh, Y.: Neural volumes: Learning dynamic renderable volumes from images. ACM Trans. Graph. 38(4), 65:1-65:14 (2019)  
[15] Ma, Z., Teed, Z., Deng, J.: Multiview stereo with cascaded epipolar raft. arXiv preprint arXiv:2205.04502 (2022)  
[16] Mildenhall, B., Srinivasan, P.P., Ortiz-Cayon, R., Kalantari, N.K., Ramamoorthi, R., Ng, R., Kar, A.: Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (TOG) 38(4), 1-14 (2019)  
[17] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: European conference on computer vision. pp. 405-421. Springer (2020)

[18] Niemeyer, M., Barron, J.T., Mildenhall, B., Sajjadi, M.S., Geiger, A., Radwan, N.: Regn-erf: Regularizing neural radiance fields for view synthesis from sparse inputs. arXiv preprint arXiv:2112.00724 (2021)  
[19] Park, K., Sinha, U., Barron, J.T., Bouaziz, S., Goldman, D.B., Seitz, S.M., Martin-Brualla, R.: Nerfies: Deformable neural radiance fields. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5865-5874 (2021)  
[20] Park, K., Sinha, U., Hedman, P., Barron, J.T., Bouaziz, S., Goldman, D.B., Martin-Brualla, R., Seitz, S.M.: Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228 (2021)  
[21] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S.: Pytorch: An imperative style, high-performance deep learning library. In: Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., Garnett, R. (eds.) Advances in Neural Information Processing Systems 32, pp. 8024-8035. Curran Associates, Inc. (2019)  
[22] Pumarola, A., Corona, E., Pons-Moll, G., Moreno-Noguer, F.: D-nerf: Neural radiance fields for dynamic scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10318–10327 (2021)  
[23] Rakhimov, R., Ardelean, A.T., Lempitsky, V., Burnaev, E.: Npgb++: Accelerating neural point-based graphics. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15969-15979 (2022)  
[24] Ravi, N., Reizenstein, J., Novotny, D., Gordon, T., Lo, W.Y., Johnson, J., Gkioxari, G.: Accelerating 3d deep learning with pytorch3d. arXiv:2007.08501 (2020)  
[25] Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234-241. Springer (2015)  
[26] Rückert, D., Franke, L., Stamminger, M.: Adop: Approximate differentiable one-pixel point rendering. ACM Transactions on Graphics (TOG) 41(4), 1-14 (2022)  
[27] Schonberger, J.L., Zheng, E., Frahm, J.M., Pollefeys, M.: Pixelwise view selection for unstructured multi-view stereo. In: European Conference on Computer Vision. pp. 501-518. Springer (2016)  
[28] Schonberger, J.L., Frahm, J.M.: Structure-from-Motion Revisited. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2016)  
[29] Schonberger, J.L., Zheng, E., Pollefeys, M., Frahm, J.M.: Pixelwise View Selection for Unstructured Multi-View Stereo. In: European Conference on Computer Vision (ECCV) (2016)  
[30] Smith, L.N., Topin, N.: Super-convergence: Very fast training of neural networks using large learning rates. In: Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications. vol. 11006, p. 1100612. International Society for Optics and Photonics (2019)  
[31] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.: Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research 15(1), 1929-1958 (2014)  
[32] Ulyanov, D., Vedaldi, A., Lempitsky, V.: Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022 (2016)  
[33] Wei, Y., Liu, S., Rao, Y., Zhao, W., Lu, J., Zhou, J.: Nerfingmvs: Guided optimization of neural radiance fields for indoor multi-view stereo. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5610-5619 (2021)

[34] Wiles, O., Gkioxari, G., Szeliski, R., Johnson, J.: Synsin: End-to-end view synthesis from a single image. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7467-7477 (2020)  
[35] Xu, Q., Xu, Z., Philip, J., Bi, S., Shu, Z., Sunkavalli, K., Neumann, U.: Point-nerf: Point-based neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5438–5448 (2022)  
[36] Yan, J., Wei, Z., Yi, H., Ding, M., Zhang, R., Chen, Y., Wang, G., Tai, Y.W.: Dense hybrid recurrent multi-view stereo net with dynamic consistency checking. In: European Conference on Computer Vision. pp. 674–689. Springer (2020)  
[37] Yang, B., Zhang, Y., Xu, Y., Li, Y., Zhou, H., Bao, H., Zhang, G., Cui, Z.: Learning object-compositional neural radiance field for editable scene rendering. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 13779-13788 (2021)  
[38] Yang, B., Zhang, Y., Xu, Y., Li, Y., Zhou, H., Bao, H., Zhang, G., Cui, Z.: Learning object-compositional neural radiance field for editable scene rendering. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 13779-13788 (2021)  
[39] Yao, Y., Luo, Z., Li, S., Fang, T., Quan, L.: Mvsnet: Depth inference for unstructured multiview stereo. In: Proceedings of the European Conference on Computer Vision (ECCV). pp. 767-783 (2018)  
[40] Yao, Y., Luo, Z., Li, S., Zhang, J., Ren, Y., Zhou, L., Fang, T., Quan, L.: Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1790-1799 (2020)  
[41] Yifan, W., Serena, F., Wu, S., Öztireli, C., Sorkine-Hornung, O.: Differentiable surface splatt-ting for point-based geometry processing. ACM Transactions on Graphics (TOG) 38(6), 1-14 (2019)  
[42] Yu, A., Fridovich-Keil, S., Tancik, M., Chen, Q., Recht, B., Kanazawa, A.: Plenoxels: Radiance fields without neural networks. arXiv preprint arXiv:2112.05131 (2021)  
[43] Yu, A., Li, R., Tancik, M., Li, H., Ng, R., Kanazawa, A.: Plenoctrees for real-time rendering of neural radiance fields. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5752-5761 (2021)  
[44] Yu, A., Ye, V., Tancik, M., Kanazawa, A.: pixelnerf: Neural radiance fields from one or few images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4578-4587 (2021)  
[45] Yuan, Y.J., Sun, Y.T., Lai, Y.K., Ma, Y., Jia, R., Gao, L.: Nerf-editing: geometry editing of neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18353–18364 (2022)  
[46] Zhang, K., Riegler, G., Snavely, N., Koltun, V.: Nerf++: Analyzing and improving neural radiance fields. arXiv preprint arXiv:2010.07492 (2020)  
[47] Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586-595 (2018)
