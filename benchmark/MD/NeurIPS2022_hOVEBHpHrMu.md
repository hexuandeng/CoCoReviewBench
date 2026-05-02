# MsSVT: Mixed-scale Sparse Voxel Transformer for 3D Object Detection on Point Clouds

Anonymous Author(s)

Affiliation

Address

email

# Abstract

3D object detection from the LiDAR point cloud is fundamental to autonomous driving. Large-scale outdoor scenes usually feature significant variance in instance scales, thus requiring features rich in long-range and fine-grained information to support accurate detection. Recent detectors leverage the power of window-based transformers to model long-range dependencies but, unfortunately, tend to blur out fine-grained details. To mitigate this gap, we present a novel Mixed-scale Sparse Voxel Transformer, named MsSVT, which can well capture both types of information simultaneously by the divide-and-conquer philosophy. Specifically, MsSVT explicitly divides attention heads into multiple groups, each in charge of attending to information within a particular range. The output from all the groups is merged to obtain the final mixed-scale features. Moreover, we provide a novel chessboard sampling strategy to reduce the computational complexity of applying a window-based transformer in 3D voxel space and implement the voxel sampling and gathering operations sparsely with a hash map to improve efficiency. Endowed by the powerful capability and high efficiency of modeling mixed-scale information, our single-stage detector built on top of MsSVT surprisingly outperforms state-of-the-art two-stage detectors on Waymo. Code will soon be available.

# 1 Introduction

3D object detection has received increasing attention due to its successful autonomous driving applications. Unlike 2D images with a regular structure of pixels, LiDAR point clouds are naturally irregular and unordered. Hence directly applying CNN-like operations [10, 11] to them can be difficult. To solve this, many researchers have rasterized point clouds into regular voxel grids [21] and employed 3D CNNs to extract high-dimensional voxel features. With the recent rise of vision transformer (ViT) [35] on 2D images, some attempts have been made to generalize global or more efficient window-based transformers to 3D voxels [20] or pillars [6]. These methods successfully seek long-range context by utilizing transformers' powerful abilities in modeling long-range information. However, they ignore that blindly increasing receptive fields would easily blur fine-grained details, especially in sparse 3D space, crucial to accurate object recognition and localization.

Standard window-based transformers update the features of queries in a local window by attending to keys from the same window. Hence simultaneously aggregating long-range context and fine-grained details require enlarging the window size to embrace local and distant voxels. Nevertheless, directly gathering all the voxels within the window as keys suffer a cubical growth in computational cost w.r.t. the window size. Some attempts [20] alleviate this by sampling only a certain number of key voxels. While a trivial sampling strategy quickly leads to sparse sampling of local voxels (Fig. 1 b), thus bias mainly on long-range context. To mitigate the above, we try to set up multiple key windows of varying sizes centered on a query window and sample the same number of local and distant key

![](images/af18b7f877784c2cd58b019097147065a9fbf30f7ba45bf34f9d73cc8490faa5.jpg)

![](images/74f7c35d5d5d5837251dfe59cd386e858ddd2b4725176412f34980bc9eb3a15b.jpg)

![](images/4a17bcf237b03bd505db7dcbabab51fcb94e6197417ac56f6738f6f5a81bcea8.jpg)

![](images/c81afd2af770c8df416ee6f9d1be42fad30a44af9411672182abdc33f06bdaf7.jpg)  
Figure 1: Top: In contrast to sampling key voxels from b) a single-scale 3D window in a) raw point clouds, our MsSVT samples key voxels from c) multi-scale windows, thus keeps finer granularity on the target object while covering large-scale neighborhood. Bottom: Different head groups accept keys sampled from windows of different scales, and are respectively responsible for obtaining d) fine-grained details and e) long-range context (reflected by higher attention weights), thus together contribute to accurate object detection collaboratively.

![](images/e85525b69b090640c48dc25e8fc7318420a9a5125c76ad26c93b61958c5433ef.jpg)

voxels from the smaller and larger windows separately. As a result, we can keep finer granularity in the local region to retain fine-grained details while collecting distant voxels roughly to enlarge the receptive field (Fig. 1 (c)).  
With the sampled voxels ready, the next question is how to attend to voxels from different windows effectively and capture long-range context and fine-grained details simultaneously. We argue that the divide-and-conquer philosophy can satisfactorily resolve this issue. Specifically, inspired by the recent findings [35, 48, 23] that transformers learn different levels of self-attention by different heads, we propose a novel Mixed-scale Sparse Voxel Transformer (MsSVT), which explicitly divides the transformer heads into multiple groups. Different head groups accept voxels sampled from windows of different sizes, such that they are each in charge of capturing information of a particular scale. By combining the outputs from all the head groups, mixed-scale information, i.e., long-range context, and fine-grained details can be well captured simultaneously. We also design a novel scale-aware relative position encoding strategy to adaptively adjust the position encoding used in each head group according to the range of keys. We provide some resulting attention maps by two different head groups (Fig. 1 (d), (e)). It is also worth mentioning that the mixed-scale attention meanwhile enables information exchange across local windows, which makes MsSVT more compact by saving additional shift window operation that is commonly required by window-based transformers [18, 6].  
Moreover, to improve the efficiency of applying transformers in 3D voxel space, we strive to reduce computational costs in two ways. First, we propose a novel chessboard sampling (CBS) strategy to reduce the number of query voxels that need to be sampled within the query window, to reduce computational costs without losing information. Specifically, we partition the query window into chess-like spaced, and termed as "×", "○", "△", "□" positions separately. During each attention layer, only one specific position of voxels is sampled and updated by serving as queries, and the updates of the other voxels can be obtained by interpolation. Four positions are selected in the circular. Thus, we can update all the voxels without introducing deviation. Second, we take advantage of non-empty voxels' sparsity by performing mixed-scale window-based attention solely on non-empty sites in 3D space and parallelizing the search and feature gathering for non-empty voxels using hash mapping for further acceleration.  
We build a 3D detector by replacing the original sparse 3D CNN backbone in SECOND [39] with our MsSVT and conduct extensive experiments on the large-scale Waymo open dataset [32]. Benefiting from the powerful capability of abstracting voxel features of mixed-scales, our single-stage detector based on MsSVT surprisingly outperforms state-of-the-art two-stage detectors. We summarize our contributions as follows:

- We present a novel Mixed-scale Sparse Voxel Transformer (MsSVT), which abstracts voxel features with both long-range context and fine-grained details simultaneously.  
- We design an efficient chessboard sampling strategy to vastly reduce the computational cost of applying a voxel-based transformer in 3D space and sparsely implement all operations to improve efficiency.  
- Our single-stage detector based on MsSVT even outperforms state-of-the-art two-stage detectors on Waymo.

# 2 Related work

Outdoor 3D detection on point cloud. The mainstream outdoor 3D detection models are based on voxel [2, 40, 41, 31, 15, 13, 49, 39] or pillar [14]. VoxelNet [49] utilizes a point net to aggregate the features within each voxel and is further processed by sparse 3D convolution and 2D convolution to generate detection results. SECOND [39] investigates an improved sparse convolution method, which dramatically improves the model's speed. Pointpillar [14] utilizes a pillar (voxel with infinite length in the vertical direction) to quantify the point cloud, so it can fully apply the mature 2D CNN, considering both efficiency and performance. Two-stage models [17, 29, 30, 26] aggregate the original point cloud or voxel features on the proposal of the single-stage model, which is used to adjust the bounding box. At present, they have state-of-the-art performance.

Vision transformer. Inspired by the great success of Transformer [35, 4] in NLP, some works employ it in the field of computer vision [5, 1, 18, 16, 50, 42, 36]. Swin-transformer [18] increases efficiency by limiting self-attention computation to non-overlapping local windows while allowing for cross-window connection. SSA [24] divides the attention head into multiple groups and aggregates image features with different granularity, respectively, which has achieved excellent performance. In the field of point cloud, Guo et al. [8] and Zhao et al. [47] introduce transformer paradigm for point cloud classification and segmentation. Recently, many methods [37, 46, 22, 9, 19] apply local self-attention mechanism on voxels to learn richer feature representations. Our model extends window attention to 3D voxels and flexibly combines different window sizes to capture multi-scale and multi-granularity features while shunting them to different attention heads in SSA.

Voxel transformer backbone in 3D detection. VoTr [20] introduces two kinds of sparse voxel attention, including local attention and extended attention. Each voxel serves as a query and attends with the neighbor voxels. We optimize the sparse operation and introduce window attention, which significantly improves efficiency and performance. The recent SST [6] adopts a single-stride design and swim-transformer structure, which performs well on small objects. However, SST is implemented based on a pillar. The single window size is not conducive to capturing multi-scale features, resulting in the low performance of vehicles when multiple categories are detected simultaneously. In contrast, our MsSVT can capture fine and global features and better performance on large and small objects.

# 3 Method

In this section, We first detail the pipeline of the Mixed-scale Sparse Voxel Transformer (MsSVT), then move to the novel chessboard sampling strategy and efficient sparse implementation, and build a complete 3D detector based on the above components.

# 3.1 Mixed-scale Sparse Voxel Transformer

Fig. 2 illustrates the overall framework of our proposed MsSVT, consisting of two core modules: a balanced multi-window sampling module and a scale-aware head attention module. We first get the query, key, and value voxels balanced sampled from multiple windows through the former and then feed them into our scale-aware head attention to simultaneously capture mixed-scale information. We also add the scale-aware relative position encoding to the attention mechanism so that even the same relative position information can play various roles in different scales.

![](images/29e94f845b5c61c2f4ff715020e1f502945ec621ec606cc695a7a0470ce6acb4.jpg)  
Figure 2: Top: overall architecture of our detection network. Bottom: details of our MsSVT block. In MsSVT block, we gather the voxels with the query window size (e.g.  $s_0 = (2,2,2)$ ) and the key window sizes (e.g.  $s_1 = (2,2,2)$  and  $s_2 = (4,4,4)$ ), respectively. Then, we apply CBS to obtain sampled queries, while employ FPS to get sampled keys and values of same number (e.g.  $N_K = 3$ ) among all key windows. Finally, we feed the same query and the different keys into different attention head groups, respectively.

# 3.1.1 Balanced Multi-window Sampling

Given a series of window sizes  $\{s_k|s_k\in \mathbb{Z}^3\}_{k = 0}^M$ , where  $s_0$  denotes the query window size and  $s_{1,\dots,M}$  denote  $M$  different key window sizes from small to large. Let  $\mathcal{V} = \{\pmb {v}_i|\pmb {v}_i = (\pmb {x}_i,\pmb {f}_i)\}_{i = 1}^{|\mathcal{V}|}$  be the input voxel set, where  $\pmb {x}_i\in \mathbb{Z}^3$  denotes voxel coordinates and  $\pmb {f}_i\in \mathbb{R}^C$  denotes voxel features. We first partition the voxel set into non-overlap 3D windows  $s_0$  by finding non-empty window centers  $\{c_i|c_i\in \mathbb{Z}^3\}_{i = 0}^L$ , where  $L$  is the number of non-empty windows. To get query voxels, we can simply gather all the non-empty voxels  $\mathcal{V}_{c_i,s_0}$  centered on  $c_{i}$  within query window  $s_0$  to guarantee that every voxel can serve as a query and be updated after one attention layer. We also present another novel chessboard sampling strategy for the query voxel sampling which will be discussed in Section 3.2. For easy understanding, readers can regard all the non-empty voxels in each query window as the queries.

As for the key voxels, instead of sampling in a single large window at once like previous methods [20], which inevitably bias on local or distant voxels, we simultaneously research the neighbors for each center  $c_i$  with different window size  $s_k$  and gather no more than  $m$  non-empty voxels  $\mathcal{V}_{c_i,s_k} = \{v_j| - s_k < x_j - c_i < s_k\}_{j=1}^m$ , where  $m$  is a pre-defined number for all key windows. Furthermore, to reduce the computational cost and keep balanced sampling, we adopt farthest point sampling (FPS) to uniformly sample  $N_K$  voxels for each window  $s_k$  to obtain the final key voxels  $\mathcal{V}_{c_i,s_k}^{fps}$ ,  $k = 1,\dots,M$  at different scales, where  $N_K$  is a predefined maximum number of sampled voxels. Benefit from the multi-window sampling combined with the uniform FPS operation, we can achieve more balanced key voxels sampling from various scales which is crucial to capturing mixed-scale information.

# 3.1.2 Scale-aware Head Attention

Once receiving the query voxels  $\mathcal{V}_{\pmb{c}_i,\pmb{s}_0} = (\pmb{X}_0,\pmb{F}_0)$  and sampled key voxels  $\mathcal{V}_{\pmb{c}_i,\pmb{s}_k}^{fps} = (\pmb{X}_k,\pmb{F}_k), k = 1,\dots,M$  at different scales, where  $\pmb{X}_0 \in \mathbb{Z}^{N_Q \times 3}$ ,  $\pmb{F}_0 \in \mathbb{R}^{N_Q \times C}$  denote query voxel coordinates and features,  $\pmb{X}_k \in \mathbb{Z}^{N_K \times 3}$ ,  $\pmb{F}_k \in \mathbb{R}^{N_K \times C}$  denote key voxel coordinates and features, we first get queries  $\pmb{Q}$ , keys  $\{\pmb{K}_k\}_{k=1}^M$  and values  $\{\pmb{V}_k\}_{k=1}^M$  as

$$
\boldsymbol {Q}, \boldsymbol {K} _ {k}, \boldsymbol {V} _ {k} = \boldsymbol {F} _ {0} \boldsymbol {W} ^ {Q}, \boldsymbol {F} _ {k} \boldsymbol {W} ^ {K}, \boldsymbol {F} _ {k} \boldsymbol {W} ^ {V}, \quad k = 1, \dots , M \tag {1}
$$

where  $W^{Q}, W^{K}, W^{V} \in \mathbb{R}^{C \times C}$  are linear projections. To make our attention module aware to different scales, we divide the attention heads into  $M$  multiple groups and assign keys from different windows to different head groups so that the queries  $Q$  should only attend  $K_{k}$  in the  $k$ -th head group. We also split the feature channel into  $M$  parts and explicitly make each part represent the information of a specific scale. We denote  $Q^{k} = Q[(:, (k - 1) \times C / M : k \times C / M]$ ,  $k = 1, \dots, M$  as the  $k$ -th scale features for queries and the same is true for  $K$  and  $V$ . So for the  $k$ -th head group, we can get the attended feature as:

$$
\tilde {\boldsymbol {Y}} ^ {k} = \mathbf {M H A} \left(\boldsymbol {Q} ^ {k}, \boldsymbol {K} ^ {k}, \boldsymbol {V} ^ {k}, \mathbf {R P E} \left(\boldsymbol {X} _ {0}, \boldsymbol {X} _ {k}\right)\right), \tag {2}
$$

where  $\mathbf{MHA}(\cdot)$  denotes multi-head-group attention and  $\mathbf{RPE}(\cdot)$  stands for our relative position encoding which will be discussed in Section 3.1.3. Each window size corresponds to a "head group" that includes one or more attention heads. Finally, we concatenate the output of all heads group $\{\tilde{\mathbf{Y}}^k\}_{k=1}^M$  to  $\tilde{\mathbf{Y}} \in \mathbb{R}^{L \times N_Q \times C}$ , and feed into FFN, where  $L$  denotes the total number of non-empty query windows. After that, we obtain the mixed-scale features  $\mathbf{Y}$ :

$$
\tilde {\boldsymbol {Y}} = \mathbf {C A T} \left(\tilde {\boldsymbol {Y}} ^ {1}, \dots , \tilde {\boldsymbol {Y}} ^ {M}\right), \tag {3}
$$

$$
\boldsymbol {Y} = \mathbf {M L P} (\mathbf {L N} (\tilde {\boldsymbol {Y}}))) + \tilde {\boldsymbol {Y}}, \tag {4}
$$

# 3.1.3 Scale-aware Relative Position Encoding

The 3D point cloud feature generally contains the original coordinates information, which voxels will inherit. However, the fine-grained location information may be blurred with the deepening of the network, so the relative position encoding is necessary. Furthermore, since MsSVT can extract multi-scale features, the position-coding should differ at different scales, even for the same relative position. In light of these, we adopt an adaptive, scale-aware relative position encoding strategy inspired by [25, 38, 44].

Specifically, according to the size of the maximum key window, we establish a learnable embedding table  $T^k \in \mathbb{R}^{D \times R}$  for  $k$ -th head group like [18], where  $R$  is the number of possible relative position pairs and  $D$  is the feature dimension. Then we can get the query relative position bias  $B_Q^k \in \mathbb{R}^{L \times N_Q \times N_K}$  as:

$$
\boldsymbol {B} _ {Q} ^ {k} = \mathcal {G} (\boldsymbol {Q} ^ {k} \boldsymbol {T} ^ {k}, \boldsymbol {I} ^ {k}), \tag {5}
$$

where  $\pmb{I}^{k} \in \mathbb{Z}^{L \times N_{Q} \times N_{K}}$  means the table indices corresponding to the actual relative position of the queries and keys,  $\mathcal{G}(\cdot)$  means the operation of gather features via indices. We get the  $B_{K}^{k}$  in the same way. Finally, the bias acts directly on the attention, and Eq. (2) can be rewritten as:

$$
\tilde {\boldsymbol {Y}} ^ {k} = \sigma \left(\frac {\boldsymbol {Q} ^ {k} \boldsymbol {K} ^ {k \top}}{\sqrt {D}} + \boldsymbol {B} _ {Q} ^ {k} + \boldsymbol {B} _ {K} ^ {k}\right) \boldsymbol {V} ^ {k}, \tag {6}
$$

where  $\sigma(\cdot)$  represents softmax. In this way, position embeddings can be adjusted adaptively according to different scales to guide our scale-ware head attention better.

# 3.2 Chessboard Sampling

In Section 3.1, the prerequisite for us to sample key voxels to a specific number is that we do not need to preserve all the key voxels and only need to select some representative ones. However, things are different for queries. On the one hand, we should maintain and update every query voxel after an attention layer; otherwise, the information could be irreversibly lost, and sampling queries seems impossible. On the other hand, the computational cost grows cubically with window size, and it would bring unacceptable memory costs if we can not reduce the number of queries.

However, since the non-empty voxel positions are not changed during a sequential of attention blocks, an effective way is to iteratively sample a subset of voxels from all the queries and then use the sampled voxels to update those unsampled. In the light of this, we present Chessboard Sampling (CBS) as shown in Fig. 3. In an attention module, we select one "×", "○", "△", "□" as the sampling location. The sampled non-empty voxels are sent to the subsequent network for updating. The updates of other non-empty voxels can be obtained by linear interpolation with their top k nearest neighbors (default by 3). Four positions will be selected in turn in stacked modules. In this way, we retain the original structure and cover all voxels as comprehensively as possible. Note that we can apply the interval to any axis to obtain the sampling rate of  $1/2$ ,  $1/4$ , or  $1/8$ . Generally, we apply it to the horizontal  $xy$  plane (like Fig. 3).

# 3.3 Sparse Implementation

We implement all our window center searching, window gathering, and balanced window sampling sparsely in CUDA operations to leverage the natural sparsity of point clouds and improve efficiency. These operations are mainly based on a hash map which establishes the mapping from coordinate space to voxel index as in [20]. For example, for window gathering, we query each possible position wrt the given center within the window and retrieve the feature if the position is a valid key in our pre

built hash map. More details are available in the supplementary materials.

![](images/3b4d58733894e53fa4ad6eb89d6200a0bac20d4926ab55f48aa064ab93d362d7.jpg)  
Figure 3: Diagram of chessboard sampling.

# 3.4 Detector Establishment

As shown in Fig 2, we build our 3D backbone by stacking Mixed-scale Sparse Voxel Transformer (MsSVT) blocks. It is worth noting that both the query and key window size in the last block of MsSVT are set to  $(1,1,\infty)$  so as to compressing the 3D voxels into 2D feature map, where the query is the average of all the voxel features within the pillar window. We replace the 3D backbone with our MsSVT in SECOND [39] and keep the other architecture the same except for abandoning the down-sampling process because MsSVT can already capture features at different scales. The input point cloud is first converted into regular voxels and fed into our MsSVT to get the mixed-scale voxel features. Next, features are compressed vertically and sent to the following 2D RPN and detection head, outputting detection results.

# 4 Experiments

In this section, we provide architectural details of MsSVT and compare our model with recent state-of-the-art detectors on Waymo Open Dataset [32] and KITTI [7]. Thorough ablation studies and in-depth analysis are further provided to validate our design choices.

# 4.1 Architectural Details

MsSVT includes four regular MsSVT attention blocks and an additional special one where the windows are set as  $1 \times 1$  pillars as mentioned in 3.4. The query window size of the normal MsSVT attention blocks is set to  $(3,3,5)$ , and the key windows are set to  $(3,3,5)$  and  $(7,7,7)$ . We divide total 8 attention heads into 2 groups in the scale-aware head attention layer. The queries are obtained via alternating chessboard sampling with 0.25 sampling ratio, and the keys are balanced sampled with a maximum number of 32 in each window. We use center head [45] as the detection head to generate a single-stage detection bounding box. In addition, we also provide a two-stage detector with CT3D [26]. Please refer to OpenPCDet [34] for more details since we conduct our experiments with this toolbox.

# 4.2 Results on Waymo

Setup. We first evaluate our model on large-scale Waymo Open Dataset [32]. The input is a single-frame point cloud with  $150\mathrm{m}\times 150\mathrm{m}$  detection range. We set the detection range in the horizontal and the vertical direction as  $[-75.2\mathrm{m}, 75.2\mathrm{m}]$  and  $[-2.0\mathrm{m}, 4.0\mathrm{m}]$ , respectively. The voxel size is  $(0.4\mathrm{m}, 0.4\mathrm{m}, 0.6\mathrm{m})$ . We follow the same training strategy as in [20]. Specifically, we train the model on Tesla-V100 using the Adam [12] optimizer for 80 epochs on  $20\%$  Waymo data. We apply the cyclic decay scheme [39], by which the learning rate is increased from  $1e-4$  to  $1e-3$  during the first  $40\%$  epochs and further decreased to 1e-5 in the remaining epochs. We also provide the results of 30 epochs training on  $100\%$  data with the same optimizer and scheme. The evaluation metric is the 3D mean Average Precision (mAP) for difficulty levels of LEVEL 1 and LEVEL 2.

Table 1: Results on WOD validation set (train with  $20\%$  Waymo data). SS: Single-stage model, TS: Two-stage model, SF: Single frame input. Note that some priors only publish results of single-class training, which is generally simpler than multi-class training.  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Reference</td><td colspan="2">Vel_L1</td><td colspan="2">Vel_L2</td><td colspan="2">Ped_L1</td><td colspan="2">Ped_L2</td><td colspan="2">Cyc_L1</td><td colspan="2">Cyc_L2</td></tr><tr><td>mAP</td><td>mAPH</td><td>mAP</td><td>mAPH</td><td>mAP</td><td>mAPH</td><td>mAP</td><td>mAPH</td><td>mAP</td><td>mAPH</td><td>mAP</td><td>mAPH</td></tr><tr><td colspan="14">Single-Stage Methods</td></tr><tr><td>SECOND [39]</td><td>Sensors 2018</td><td>70.96</td><td>70.34</td><td>62.58</td><td>62.02</td><td>65.23</td><td>54.24</td><td>57.22</td><td>47.49</td><td>57.13</td><td>55.62</td><td>54.97</td><td>53.53</td></tr><tr><td>PointPillar [14]</td><td>CVPR 2019</td><td>70.43</td><td>69.83</td><td>62.18</td><td>61.64</td><td>66.21</td><td>46.32</td><td>58.18</td><td>40.64</td><td>55.26</td><td>51.75</td><td>53.18</td><td>49.80</td></tr><tr><td>CenterPoint [45]</td><td>CVPR 2021</td><td>72.76</td><td>72.23</td><td>64.91</td><td>64.42</td><td>74.19</td><td>67.96</td><td>66.03</td><td>60.34</td><td>71.04</td><td>69.79</td><td>68.49</td><td>67.28</td></tr><tr><td>VOTR-SS [20]</td><td>ICCV 2021</td><td>68.99</td><td>68.39</td><td>60.22</td><td>59.69</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RSN-SF [33]</td><td>CVPR 2021</td><td>75.10</td><td>74.60</td><td>66.00</td><td>65.50</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MsSVT-SS (ours)</td><td>-</td><td>77.18</td><td>76.67</td><td>68.75</td><td>68.28</td><td>80.25</td><td>73.05</td><td>72.88</td><td>66.14</td><td>73.75</td><td>72.53</td><td>70.96</td><td>69.79</td></tr><tr><td colspan="14">Two-Stage Methods</td></tr><tr><td>Part-A2 [28]</td><td>TPAMI 2020</td><td>74.66</td><td>74.12</td><td>65.82</td><td>65.32</td><td>71.71</td><td>62.24</td><td>62.46</td><td>54.06</td><td>66.53</td><td>65.18</td><td>64.05</td><td>62.75</td></tr><tr><td>PV-RCNN [29]</td><td>CVPR 2020</td><td>75.95</td><td>75.43</td><td>68.02</td><td>67.54</td><td>75.94</td><td>69.40</td><td>67.66</td><td>61.62</td><td>70.18</td><td>68.98</td><td>67.73</td><td>66.57</td></tr><tr><td>Voxel-RCNN [3]</td><td>AAAI 2021</td><td>76.13</td><td>75.66</td><td>68.18</td><td>67.74</td><td>78.20</td><td>71.98</td><td>69.29</td><td>63.59</td><td>70.75</td><td>69.68</td><td>68.25</td><td>67.21</td></tr><tr><td>PV-RCNN++ [30]</td><td>ARXIV 2021</td><td>77.61</td><td>77.14</td><td>69.18</td><td>68.75</td><td>79.42</td><td>73.31</td><td>70.88</td><td>65.21</td><td>72.50</td><td>71.39</td><td>69.84</td><td>68.77</td></tr><tr><td>VOTR-TS [20]</td><td>ICCV 2021</td><td>74.95</td><td>74.25</td><td>65.91</td><td>65.29</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CT3D [26]</td><td>ICCV 2021</td><td>76.30</td><td>-</td><td>69.04</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MsSVT-TS (ours)</td><td>-</td><td>78.41</td><td>77.91</td><td>69.74</td><td>69.17</td><td>82.34</td><td>76.77</td><td>74.71</td><td>69.36</td><td>75.74</td><td>74.65</td><td>73.72</td><td>72.64</td></tr></table>

Table 2: Results on WOD validation set (train with  $100\%$  Waymo data).  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Reference</td><td colspan="2">Vel_L1</td><td colspan="2">Vel_L2</td><td colspan="2">Ped_L1</td><td colspan="2">Ped_L2</td><td colspan="2">Cyc_L1</td><td colspan="2">Cyc_L2</td></tr><tr><td>mAP</td><td>mAPH</td><td>mAP</td><td>mAPH</td><td>mAP</td><td>mAPH</td><td>mAP</td><td>mAPH</td><td>mAP</td><td>mAPH</td><td>mAP</td><td>mAPH</td></tr><tr><td colspan="14">Single-Stage Methods</td></tr><tr><td>SECOND [39]</td><td>Sensors 2018</td><td>72.27</td><td>71.69</td><td>63.85</td><td>63.33</td><td>68.70</td><td>58.18</td><td>60.72</td><td>51.31</td><td>60.62</td><td>59.28</td><td>58.34</td><td>57.05</td></tr><tr><td>PointPillar [14]</td><td>CVPR 2019</td><td>71.57</td><td>70.99</td><td>63.06</td><td>62.54</td><td>70.61</td><td>56.70</td><td>62.85</td><td>50.24</td><td>64.36</td><td>62.27</td><td>61.95</td><td>59.93</td></tr><tr><td>SST-SS-SF [6]</td><td>CVPR 2022</td><td>73.57</td><td>-</td><td>64.80</td><td>-</td><td>80.01</td><td>-</td><td>71.66</td><td>-</td><td>70.72</td><td>-</td><td>68.01</td><td>-</td></tr><tr><td>MsSVT-SS (ours)</td><td>-</td><td>77.83</td><td>77.32</td><td>69.53</td><td>69.06</td><td>80.39</td><td>73.61</td><td>73.00</td><td>66.65</td><td>75.17</td><td>73.99</td><td>72.37</td><td>71.24</td></tr><tr><td colspan="14">Two-Stage Methods</td></tr><tr><td>Part-A2 [28]</td><td>TPAMI 2020</td><td>77.05</td><td>76.51</td><td>68.47</td><td>67.97</td><td>75.24</td><td>66.87</td><td>66.18</td><td>58.62</td><td>68.60</td><td>67.36</td><td>66.13</td><td>64.93</td></tr><tr><td>PV-RCNN [29]</td><td>CVPR 2020</td><td>78.00</td><td>77.50</td><td>69.43</td><td>68.98</td><td>79.21</td><td>73.03</td><td>70.42</td><td>64.72</td><td>71.46</td><td>70.27</td><td>68.95</td><td>67.79</td></tr><tr><td>PV-RCNN++ [30]</td><td>ARXIV 2021</td><td>79.25</td><td>78.78</td><td>70.61</td><td>70.18</td><td>81.83</td><td>76.28</td><td>73.17</td><td>68.00</td><td>73.72</td><td>72.66</td><td>71.21</td><td>70.19</td></tr><tr><td>MsSVT-TS (ours)</td><td>-</td><td>79.35</td><td>78.86</td><td>70.65</td><td>70.23</td><td>82.41</td><td>77.04</td><td>74.74</td><td>69.57</td><td>77.12</td><td>76.01</td><td>74.98</td><td>74.07</td></tr></table>

Main results. Our main results are shown in Table 1 and Table 2. It is worth noting that our model detects three categories simultaneously, which is more challenging than detecting a single category. On  $20\%$  data, MsSVT-SS significantly outperforms other single-stage detectors, even though some methods are only trained on a single category. The single-stage results can almost match the state-of-the-art two-stage methods PV-RCNN++ [30] on the vehicle and even outperform 0.8-1.5 mAP on pedestrians and cyclists. MsSVT-TS achieves the best performance in all categories and has absolute superiority over pedestrians and cyclists.

On  $100\%$  data, our model also achieves state-of-the-art performance. Compared with the recent transformer work SST [6], which is adept at the small object detection, MsSVT not only takes into account all categories but also has better performance on small objects. This result proves MsSVT's superiority over general transformers in capturing multi-scale features.

Visualization. Fig. 4 shows some visualization of our results. As shown in Fig. 4 (a), in the range of  $>50m$  that the points are very sparse, the bounding boxes are still accurate, which means that MsSVT can rely on context information in the absence of fine-grained information. As shown in Fig. 4 (b), even in a complex scene with many different categories of objects at the same time, the model still perform well, which should be owed to the flexibility and robustness of MsSVT.

Analysis. We visualized some attention weights. As shown in Fig. 5, the smaller window model focuses on the local foreground information, while the larger window model tends to focus on the context information. The two windows can complement each other. This proves the ability of MsSVT to capture mixed-scale features.

Table 3: Results on KITTI validation set.  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Reference</td><td colspan="3">3D Car (IoU=0.7)</td><td colspan="3">3D Ped. (IoU=0.5)</td><td colspan="3">3D Cyc. (IoU=0.5)</td></tr><tr><td>Easy</td><td>Mod</td><td>Hard</td><td>Easy</td><td>Mod.</td><td>Hard</td><td>Easy</td><td>Mod.</td><td>Hard</td></tr><tr><td colspan="11">Single-Stage Methods</td></tr><tr><td>SECOND [39]</td><td>Sensors 2018</td><td>86.46</td><td>77.28</td><td>74.65</td><td>61.63</td><td>56.27</td><td>52.60</td><td>80.10</td><td>62.69</td><td>59.71</td></tr><tr><td>PointPillar [14]</td><td>CVPR 2019</td><td>88.61</td><td>78.62</td><td>77.22</td><td>56.55</td><td>52.98</td><td>47.73</td><td>80.59</td><td>67.16</td><td>63.11</td></tr><tr><td>3DSSD [43]</td><td>CVPR 2020</td><td>88.55</td><td>78.45</td><td>77.30</td><td>58.18</td><td>54.32</td><td>49.56</td><td>86.25</td><td>70.49</td><td>65.32</td></tr><tr><td>VoTr-SS [20]</td><td>ICCV 2021</td><td>87.86</td><td>78.27</td><td>76.93</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MsSVT-SS (ours)</td><td>-</td><td>89.08</td><td>78.75</td><td>77.35</td><td>63.59</td><td>57.33</td><td>53.12</td><td>88.57</td><td>71.70</td><td>66.29</td></tr><tr><td colspan="11">Two-Stage Methods</td></tr><tr><td>PointRCNN [27]</td><td>CVPR 2019</td><td>89.03</td><td>78.78</td><td>77.86</td><td>62.50</td><td>55.18</td><td>50.15</td><td>87.49</td><td>72.55</td><td>66.01</td></tr><tr><td>Part-A2 [28]</td><td>TPAMI 2020</td><td>88.48</td><td>78.96</td><td>78.36</td><td>70.73</td><td>64.13</td><td>57.45</td><td>88.18</td><td>73.35</td><td>70.75</td></tr><tr><td>PV-RCNN [29]</td><td>CVPR 2020</td><td>89.35</td><td>83.69</td><td>78.70</td><td>63.12</td><td>54.84</td><td>51.78</td><td>86.06</td><td>69.48</td><td>64.50</td></tr><tr><td>VoTr-TS [20]</td><td>ICCV 2021</td><td>89.04</td><td>84.04</td><td>78.68</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MsSVT-TS (ours)</td><td>-</td><td>89.32</td><td>84.66</td><td>78.94</td><td>66.11</td><td>58.94</td><td>53.86</td><td>92.49</td><td>73.60</td><td>69.34</td></tr></table>

# 4.3 Results on KITTI

Setup. We further evaluate our model on KITTI [7]. Given an input point cloud, we reserve the points within the range of  $[0\mathrm{m}, 70.4\mathrm{m}]$ ,  $[-40.0\mathrm{m}, 40.0\mathrm{m}]$  and  $[-3.0\mathrm{m}, 1.0\mathrm{m}]$  in the X, Y, and Z directions, respectively. The voxel size is set to  $(0.32\mathrm{m}, 0.32\mathrm{m}, 0.4\mathrm{m})$  and other settings are same as 4.2. We train the model for 100 epochs with the Adam optimizer. The learning rate is 0.003, decayed by the cyclic scheme [39]. The evaluation metric is 3D mAP for three difficulty levels (easy, moderate, and hard).

Main results. Table 3 depicts our MsSVT achieves competitive performance on all the three categories. Specifically, on Car, MsSVT-SS beats the superior VoTr [20] by  $0.5\mathrm{mAP}$ . Meanwhile, on Pedestrian and Cyclist, MsSVT-SS even outperforms some preeminent two-stage detectors. Moreover, our two-stage MsSVT-TS further increases the lead. The results on KITTI demonstrate that our MsSVT has a good generalization ability to adapt to different datasets.

# 4.4 Ablation Study

We perform ablation studies to validate our design choices. All models are trained for 12 epochs on  $20\%$  Waymo data. More implementation details and ablations on hyperparameters are available in supplementary materials.

# Balanced multi-window sampling.

We first validate our balanced multiwindow sampling strategy in Table 4. The base model, listed in the first row, employs 3D version of standard window-based attention [18] (with window size  $(3,3,5)$  and without shift window scheme), adopts dilated key sampling strategy [20], and gathers all non-empty voxels within the window as queries. We build a model variant by simply replacing the dilated key

Table 4: Ablations on different components of MsSVT. BMS: Balanced Multi-window Sampling, SHA: Scale-aware Head Attention, SRPE: Relative Position Encoding.

<table><tr><td>BMS</td><td>SHA</td><td>SRPE</td><td>Veh / Ped / Cyc</td></tr><tr><td></td><td></td><td></td><td>69.51/71.66/63.31</td></tr><tr><td>✓</td><td></td><td></td><td>71.24/73.44/66.88</td></tr><tr><td>✓</td><td>✓</td><td></td><td>71.96/75.08/67.16</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>72.37/75.99/67.90</td></tr></table>

sampling in the base model with our balanced multi-window sampling strategy. The results in the second row show sampling key voxels from windows of multiple sizes brings noticeable performance gains, i.e.,  $71.66 \rightarrow 73.44$  on Pedestrian and  $63.31 \rightarrow 66.88$  on Cyclist.

Scale-aware head attention. Table 4 also validates the effectiveness of our scale-aware head attention module. By comparing the second and the third rows, we find that making different attention head groups specialized in different scales helps to capture mixed-scale features and thus brings significant performance gains. We further visualize some attention maps by different head groups in

![](images/5098acd34954ecffd3df8b6f8cdc8fa0aa2daa752a7f5349b3610857a9e8d06d.jpg)  
Figure 4: Qualitative results on Waymo. The red and green boxes denote ground-truths and predictions, respectively. Our MsSVT performs impressively in scenes (a) beyond the range of  $50\mathrm{m}$ , and (b) in the presence of dense objects with large scale variations.

![](images/52805a5d9ae857f30cfc93645347028eb0986a83378fc7325fec92d35814ece5.jpg)

![](images/2d40c0fbe3eb1acfe80d9dc0d93452bcbd1a046fc7a89d00765ff4f0a2ce8bab.jpg)  
Figure 5: Visualization of attention maps. Pink dot denotes the query position. Positions with high and low attention weights are in red and blue, respectively.

![](images/e75c42e44bd7fe2c9999fb51788145fdbaaefe315580ca5c9ffae5a6cafbccf0.jpg)  
Fig. 5. One can observe that the query point pays more attention to fine-grained details in a small window and is more interested in long-range contextual information in a large window.

![](images/51fb738ca5880b86bae516faff2e46244d9e21ded2d49bb7951ac1602c651d60.jpg)

![](images/23aa041cdd46c1a221e93a9e43590eac45acb2c67f33a18472256a416c64b849.jpg)

![](images/7610ccf1efec71ab07fc107df712dc825d5e385f2b0409c43bdad74f1250e5f5.jpg)

Scale-aware relative position encoding. Table 4 shows compared with the model variant that adopts scale-agnostic position encoding (listed in the second row), our scale-aware relative position encoding gives better performance, which well supports our design motivation that the importance of positional information varies with different scales.

Chessboard sampling. Table 5 reports the performance of the model variants using chessboard sampling with different sampling ratios, and of the model without performing any sampling. One can see that our model is not sensitive to the variation of sampling ratio. The model variant with sampling ratio 1/4 performs even on par with the model without sampling, while bringing a significant reduction (up to 6.1G) in memory footprint.

Table 5: Ablations on sampling strategy.  

<table><tr><td>Strategy</td><td>Veh / Ped / Cyc</td><td>Mem (G)</td><td>Lat (ms)</td></tr><tr><td>w/o</td><td>72.58/75.74/68.24</td><td>18.3</td><td>167</td></tr><tr><td>1/2</td><td>72.44/76.03/67.81</td><td>14.5</td><td>138</td></tr><tr><td>1/4</td><td>72.37/75.99/67.90</td><td>12.2</td><td>121</td></tr><tr><td>1/8</td><td>72.01/75.54/67.43</td><td>11.4</td><td>113</td></tr></table>

# 5 Conclusion

In this paper, we propose MsSVT, a novel 3D sparse voxel transformer backbone for 3D detection to capture mixed-scale information simultaneously. We first utilize the novel chessboard sampling and balanced multi-window sampling to obtain queries and a set of keys with different scales. Then MsSVT explicitly divides attention heads into multiple scale-aware groups and each in charge of capturing information at a specific scale. Equipped with the scale-aware relative position encoding, MsSVT finally gets the mixed-scale features which is crucial for accurate detection.

Limitations. MsSVT considers both fine-grained details and long-range contextual information in different windows and achieves promising performance, but still needs to set hand-crafted window sizes, we will explore adaptive window version of MsSVT in the future.

# References

[1] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020.  
[2] Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li, and Tian Xia. Multi-view 3d object detection network for autonomous driving. In CVPR, pages 1907-1915, 2017.  
[3] Jiajun Deng, Shaoshuai Shi, Peiwei Li, Wengang Zhou, Yanyong Zhang, and Houqiang Li. Voxel r-cnn: Towards high performance voxel-based 3d object detection. arXiv preprint arXiv:2012.15712, 1(2):4, 2020.  
[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.  
[5] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.  
[6] Lue Fan, Ziqi Pang, Tianyuan Zhang, Yu-Xiong Wang, Hang Zhao, Feng Wang, Naiyan Wang, and Zhaoxiang Zhang. Embracing single stride 3d object detector with sparse transformer. arXiv preprint arXiv:2112.06375, 2021.  
[7] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The International Journal of Robotics Research, 32(11):1231-1237, 2013.  
[8] Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R Martin, and Shi-Min Hu. Pct: Point cloud transformer. Computational Visual Media, 7(2):187-199, 2021.  
[9] Chenhang He, Ruihuang Li, Shuai Li, and Lei Zhang. Voxel set transformer: A set-to-set approach to 3d object detection from point clouds. arXiv preprint arXiv:2203.10314, 2022.  
[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.  
[11] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In CVPR, 2017.  
[12] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
[13] Jason Ku, Melissa Mozifian, Jungwook Lee, Ali Harakeh, and Steven L Waslander. Joint 3d proposal generation and object detection from view aggregation. In IROS, pages 1-8. IEEE, 2018.  
[14] Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In CVPR, 2019.  
[15] Bo Li, Tianlei Zhang, and Tian Xia. Vehicle detection from 3d lidar using fully convolutional network. arXiv preprint arXiv:1608.07916, 2016.  
[16] Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Improved multiscale vision transformers for classification and detection. arXiv preprint arXiv:2112.01526, 2021.  
[17] Zhichao Li, Feng Wang, and Naiyan Wang. Lidar r-cnn: An efficient and universal 3d object detector. In CVPR, 2021.  
[18] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021.  
[19] Anas Mahmoud, Jordan SK Hu, and Steven L Waslander. Dense voxel fusion for 3d object detection. arXiv preprint arXiv:2203.00871, 2022.

[20] Jiageng Mao, Yujing Xue, Minzhe Niu, Haoyue Bai, Jiashi Feng, Xiaodan Liang, Hang Xu, and Chunjing Xu. Voxel transformer for 3d object detection. In ICCV, 2021.  
[21] Daniel Maturana and Sebastian Scherer. Voxnet: A 3d convolutional neural network for real-time object recognition. In 2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 922-928. IEEE, 2015.  
[22] Chunghyun Park, Yoonwoo Jeong, Minsu Cho, and Jaesik Park. Fast point transformer. arXiv preprint arXiv:2112.04702, 2021.  
[23] Namuk Park and Songkuk Kim. How do vision transformers work? arXiv preprint arXiv:2202.06709, 2022.  
[24] Sucheng Ren, Daquan Zhou, Shengfeng He, Jiashi Feng, and Xinchao Wang. Shunted self-attention via multi-scale token aggregation. arXiv preprint arXiv:2111.15193, 2021.  
[25] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. arXiv preprint arXiv:1803.02155, 2018.  
[26] Hualian Sheng, Sijia Cai, Yuan Liu, Bing Deng, Jianqiang Huang, Xian-Sheng Hua, and Min-Jian Zhao. Improving 3d object detection with channel-wise transformer. In ICCV, 2021.  
[27] Shaoshuai Shi, Xiaogang Wang, and Hongsheng Li. Pointcnn: 3d object proposal generation and detection from point cloud. In CVPR, 2019.  
[28] Shaoshuai Shi, Zhe Wang, Xiaogang Wang, and Hongsheng Li. Part-a^2 net: 3d part-aware and aggregation neural network for object detection from point cloud. arXiv preprint arXiv:1907.03670, 2(3), 2019.  
[29] Shaoshuai Shi, Chaoxu Guo, Li Jiang, Zhe Wang, Jianping Shi, Xiaogang Wang, and Hongsheng Li. Pv-rcnn: Point-voxel feature set abstraction for 3d object detection. In CVPR, 2020.  
[30] Shaoshuai Shi, Li Jiang, Jiajun Deng, Zhe Wang, Chaoxu Guo, Jianping Shi, Xiaogang Wang, and Hongsheng Li. Pv-rcnn++: Point-voxel feature set abstraction with local vector representation for 3d object detection. arXiv preprint arXiv:2102.00463, 2021.  
[31] Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller. Multi-view convolutional neural networks for 3d shape recognition. In ICCV, pages 945-953, 2015.  
[32] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In CVPR, 2020.  
[33] Pei Sun, Weiyue Wang, Yuning Chai, Gamaleldin Elsayed, Alex Bewley, Xiao Zhang, Cristian Sminchisescu, and Dragomir Anguelov. Rsn: Range sparse net for efficient, accurate lidar 3d object detection. In CVPR, 2021.  
[34] OpenPCDet Development Team. Openpcdet: An open-source toolbox for 3d object detection from point clouds. https://github.com/open-mmlab/OpenPCDet, 2020.  
[35] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017.  
[36] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In ICCV, pages 568-578, 2021.  
[37] Yue Wang, Vitor Campagnolo Guizilini, Tianyuan Zhang, Yilun Wang, Hang Zhao, and Justin Solomon. Detr3d: 3d object detection from multi-view images via 3d-to-2d queries. In Conference on Robot Learning, pages 180–191. PMLR, 2022.  
[38] Kan Wu, Houwen Peng, Minghao Chen, Jianlong Fu, and Hongyang Chao. Rethinking and improving relative position encoding for vision transformer. In ICCV, 2021.

[39] Yan Yan, Yuxing Mao, and Bo Li. Second: Sparsely embedded convolutional detection. Sensors, 18(10):3337, 2018.  
[40] Bin Yang, Ming Liang, and Raquel Urtasun. Hdnet: Exploiting hd maps for 3d object detection. In Conference on Robot Learning, pages 146-155. PMLR, 2018.  
[41] Bin Yang, Wenjie Luo, and Raquel Urtasun. Pixor: Real-time 3d object detection from point clouds. In CVPR, pages 7652-7660, 2018.  
[42] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, and Jianfeng Gao. Focal self-attention for local-global interactions in vision transformers. arXiv preprint arXiv:2107.00641, 2021.  
[43] Zetong Yang, Yanan Sun, Shu Liu, and Jiaya Jia. 3dssd: Point-based 3d single stage object detector. In CVPR, 2020.  
[44] Zetong Yang, Li Jiang, Yanan Sun, Bernt Schiele, and Jiaya Jia. A unified query-based paradigm for point cloud understanding. arXiv preprint arXiv:2203.01252, 2022.  
[45] Tianwei Yin, Xingyi Zhou, and Philipp Krahenbuhl. Center-based 3d object detection and tracking. In CVPR, 2021.  
[46] Cheng Zhang, Haocheng Wan, Shengqiang Liu, Xinyi Shen, and Zizhao Wu. Pvt: Point-voxel transformer for 3d deep learning. arXiv preprint arXiv:2108.06076, 2021.  
[47] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In ICCV, 2021.  
[48] Daquan Zhou, Zhiding Yu, Enze Xie, Chaowei Xiao, Anima Anandkumar, Jiashi Feng, and Jose M Alvarez. Understanding the robustness in vision transformers. arXiv preprint arXiv:2204.12451, 2022.  
[49] Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. In CVPR, 2018.  
[50] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.
