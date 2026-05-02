# COMPUTATION REALLOCATION FOR OBJECT DETECTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

The allocation of computation resources across different feature resolutions in the backbone is a crucial issue in object detection. However, classification allocation pattern is usually adopted directly to object detection, which is proved to be sub-optimal. In order to reallocate the engaged computation resources in a more efficient way, we present CR-NAS (Computation Reallocation Neural Architecture Search) that can learn computation reallocation strategies on the target detection dataset. A two-level reallocation space is proposed for both stage and spatial reallocation. A novel hierarchical search procedure is adopted to cope with the complex search space. We apply CR-NAS to multiple backbones and achieve consistent improvements. Our CR-ResNet50 and CR-MobileNetV2 outperforms the baseline by  $1.9\%$  and  $1.7\%$  COCO AP respectively without any additional computation budget. The models discovered by CR-NAS can be easily transferred to other dataset, e.g. PASCAL VOC, and other vision tasks, e.g. instance segmentation. Our CR-NAS can be used as a plugin to improve the performance of various networks, which is demanding.

# 1 INTRODUCTION

Object detection is one of the fundamental tasks in computer vision. The backbone feature extractor is usually take directly from classification literature (Girshick, 2015; Ren et al., 2015; Lin et al., 2017a; Lu et al., 2019). However, comparing with classification, object detection aims to know not only what but also where the object is. Directly taking the backbone of classification network for object detectors is sub-optimal, which has been observed in Li et al. (2018). To address this issue, there are many approaches either manually or automatically modify the backbone network. Chen et al. (2019) proposes a neural architecture search (NAS) framework for detection backbone to avoid expert efforts and design trails. However, previous works rely on the prior knowledge for classification task, either inheriting the backbone for classification, or designing search space similar to NAS on classification. This raises a natural question: How to design an effective backbone dedicated to detection tasks?

To answer this question, we first draw a link between the Effective Receptive Field (ERF) and the computation allocation of backbone. In many cases, the density of the connections between the output neuron and the input distributes as a Gaussian (Luo et al., 2016). The ERF of image classification task can be easily fulfilled, e.g. the input size is  $224 \times 224$  for the ImageNet data, while the ERF of object detection task need more capacities to handle scale variance across the instances, e.g. the input size is  $800 \times 1333$  and the sizes of objects vary from 32 to 800 for the COCO dataset. Lin et al. (2017a) allocates objects of different scales into different feature resolutions to capture the appropriate ERF in each stage. Here we conduct an experiment to study the differences between the ERF of several FPN features. As shown in Figure 1, we notice the allocation of computation across different resolution has a great impact on the ERFs. Furthermore, appropriate computation allocation across spacial position (Dai et al., 2017; Zhu et al., 2019) boost the performance of detector by affecting the ERFs.

Based on the above observation, in this paper, we aim to automatically design the computation allocation of backbone for object detectors. Different from existing detection NAS works (Ghiasi et al., 2019; Ning Wang & Shen, 2019) which achieve accuracy improvement by introducing higher

![](images/5458a3b8cc878f4b8046fb7f7989e8575031d8f3e145f5606e1de3d56617861f.jpg)  
Figure 1: Following the instructions in Luo et al. (2016), we draw the ERF of FPN in different resolution features. The size of base plate is  $512 \times 512$ , with respective anchor boxes (\{64, 128, 256\} for  $\{p_3, p_4, p_5\}$ ) drawn in. The classification CNNs ResNet50 tends to have redundant ERF for high resolution features  $p_3$  and limited ERF for low resolution features  $p_5$ . After stage reallocation, our SCR-ResNet50 has more balanced ERF across all resolutions which leads to a high performance.

![](images/97a415fee29b1edba0e5e0aeb2bf86c608dfadef66e2a0f03409f7f56b750736.jpg)

computation complexity, we reallocate the engaged computation cost in a more efficient way. We propose computation reallocation NAS (CR-NAS) to search the allocation strategy directly on the detection task. A two-level reallocation space is conducted to reallocate the computation across different resolution and spatial position. In stage level, we search for the best strategies to distribute the computation among different stages. In operation level, we reallocate the computation by introducing a powerful search space designed specially for object detection. The details about search space can be found in Sec. 3.2. We propose a hierarchical search algorithm to cope with the complex search space. Typically in stage reallocation, we exploit a reusable search space to reduce stage-level searching cost and adapt different computational requirements.

Extensive experiments show the effectiveness of our approach. Our CR-NAS offers improvements for both fast mobile model and accurate model, such as ResNet (He et al., 2016), MobileNetV2 (Sandler et al., 2018), ResNeXt (Xie et al., 2017). On the COCO dataset, our CR-ResNet50 and CR-MobileNetV2 can achieve a 38.3 and  $33.9\mathrm{mAP}$ , outperforming the baseline by  $1.9\%$  and  $1.7\%$  respectively without any additional computation budget. Furthermore, we transfer our CR-ResNet and CR-MobileNetV2 into the another ERFs-sensitive task, instance segmentation, by using the Mask RCNN (He et al., 2017) framework. Our CR-ResNet50 and CR-MobileNetV2 yields  $1.3\%$  and  $1.2\%$  COCO segmentation AP improvement over baseline.

To summarize, the contribution of our paper is three-fold:

- We propose computation reallocation NAS(CR-NAS) to reallocate engaged computation resources. To our knowledge, we are the first to dig inside the computation allocation across different resolution.  
- We develop a two-level reallocation space and hierarchical search paradigm to cope with the complex search space. Typically in stage reallocation, we exploit a reusable model to reduce stage-level searching cost and adapt different computational requirements.  
- Our CR-NAS offers significant improvements for various types of networks. The discovered models show great transferability over other datasets, e.g. PASCAL VOC (Everingham et al., 2015) and other vision tasks, e.g. instance segmentation (He et al., 2017).

# 2 RELATED WORK

**Objection Detection** Object detection algorithms using deep learning methods are categorized into two types, one-stage approaches and two-stage approaches. Two-stage approaches (Girshick, 2015; Ren et al., 2015; Lin et al., 2017a) generate RoIs in the first stage, then classify and refine the RoIs. One-stage detectors (Redmon et al., 2016; Liu et al., 2016; Lin et al., 2017b) infer object categories and anchor boxes directly without RoI generation. Lin et al. (2017a) use FPN to combine features of different resolutions. We adopt our CR-NAS on Faster R-CNN (Ren et al., 2015) with

FPN (Lin et al., 2017a). Li et al. (2019; 2018) show the effectiveness of dilated convolution in object detection.

Network Architecture Search(NAS) Network architecture search focus on automating the architecture design process which requires great expert knowledge and tremendous trails. Early reinforcement learning methods (Zoph & Le, 2016; Zoph et al., 2018; Guo et al., 2019a) are computationally expensive. Recently, weight sharing strategy (Cai et al., 2018b; Liu et al., 2018; Pham et al., 2018; Wu et al., 2019) is proposed to reduce computation cost. In this strategy a super-network is trained to produce suitable weights for each sub-model in the search space, i.e. the sub-networks share the weights in the supernet. One-shot method (Brock et al., 2017; Guo et al., 2019b; Chu et al., 2019) make use of shared weights to train the supernet and choose promising structures first, and then re-train the chosen structures from scratch to evaluate their performance. We adopt one-shot method to implement our CR-NAS.

NAS on detection. There are some work use NAS methods on detection task (Chen et al., 2019; Ning Wang & Shen, 2019; Ghiasi et al., 2019). Ghiasi et al. (2019) search for feature pyramid architectures and Ning Wang & Shen (2019) search for feature pyramid network and the prediction heads. These two works do not focus on the CNN backbone. Chen et al. (2019) search for channel split and shuffle operation in the CNN backbone but concern nothing on stage level allocation. Peng et al. (2019) search for dilated rate on channel level in the CNN backbone. These two approaches assume the fixed number of blocks in each resolution, while we search the number of blocks in each stage that is important for object detection and complementary to these approaches.

# 3 METHOD

In this section, we first introduce some basic settings. Then we describe our two-level architecture search space to reallocate engaged computation in stage and convolution level. Finally, we introduce our hierarchical search algorithm to cope with the complex search space.

# 3.1 BASIC SETTINGS

Our search method is based on the Faster RCNN (Ren et al., 2015) with FPN (Lin et al., 2017a) for its excellent performance. We only reallocate the computation within the backbone, while fix other components for fair comparison.

For more efficient search, we adopt the idea of one-shot NAS method (Brock et al., 2017; Bender et al., 2018; Guo et al., 2019b). In one-shot NAS, a directed acyclic graph  $\mathcal{G}$  (a.k.a. supernet) is built to subsume all architectures in the search space and is trained only once. Each architecture  $g$  is a subgraph of  $\mathcal{G}$  and can inherit weights from the trained supernet. For a specific subgraph  $g \in \mathcal{G}$ , its corresponding network can be denoted as  $\mathcal{N}(g, w)$  with network weights  $w$ .

# 3.2 TWO-LEVEL ARCHITECTURE SEARCH SPACE

We propose Computation Reallocation NAS (CR-NAS) to distribute the computation resources in two dimensions: stage allocation in different resolution, convolution allocation in spatial position.

# 3.2.1 STAGE REALLOCATION SPACE

The backbone aims to generate intermediate-level features  $C$  with increasing downsampling rates  $4 \times$ ,  $8 \times$ ,  $16 \times$ , and  $32 \times$ , which can be regarded as 4 stages. The blocks in the same stage share the same spatial resolution. Note that the FLOPs of a single block in two adjacent spatial resolutions remain the same because a downsampling/pooling layer doubles the number of channels. So given the number of total blocks of a backbone  $N$ , we can reallocate the number of blocks for each stage while keep the total FLOPs the same. Fig. 2 shows our stage reallocation space. In this search space, each stage contains several branches, and each branch has certain number of blocks. The numbers of blocks in different branches are different, corresponding to different computational budget for the stage. For example, there are 5 branches for the stage 1 in Fig. 2, the numbers of blocks for these 5 branches are, respectively, 1, 2, 3, 4, and 5. We consider the whole network as a supernet  $T = \{T_{1}, T_{2}, T_{3}, T_{4}\}$ , where  $T_{i}$  at the  $i$ th stage has  $K_{i}$  branches, i.e.  $T_{i} = \{t_{i}^{k} | k = 1 \dots K_{i}\}$ . Then an

![](images/7d29d59b38abe497a68c406ecf21ef2170f3c20e31b69d00e6432e80d494a2d0.jpg)  
Figure 2: Stage reallocation in different resolution. In supernet training, we random sample a choice in each stage and optimize corresponding weights. In reallocation searching, eligible strategies are evaluated according to the computation budget.

allocation strategy can be represented as  $\tau = [\tau_{1},\tau_{2},\tau_{3},\tau_{4}]$ , where  $\tau_{i}$  denote the number of blocks in the  $i$ th branch. All blocks in the same stage have the same structure.  $\sum_{i=1}^{4}\tau_{i} = N$  for a network with  $N$  blocks. For example, the original ResNet101 has  $\tau = [3,4,23,3]$  and  $N = 33$  residual blocks. We would like to find the best allocation strategy among the  $33^3$  possible choices. Since validating a single detection architecture requires hundreds of GPU-hours, it not realist to find the optimal architecture by human trails.

On the other hand, we would like to learn stage reallocation strategy for different computation budgets simultaneously. Different applications require CNNs of different numbers of layers for achieving different latency requirements. This is why we have ResNet18, ResNet50, ResNet101, etc. We build a search space to cover all the candidate instances in a certain series, e.g. ResNet series. After considering the trade off between granularity and range, we set the numbers of blocks for  $T_{1}$  and  $T_{2}$  as  $\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$ , and set the numbers of blocks for  $T_{3}$  as  $\{2, 3, 5, 6, 9, 11, 14, 17, 20, 23\}$ , for  $T_{4}$  as  $\{2, 3, 4, 6, 7, 9, 11, 13, 15, 17\}$  for the ResNet series. The stage reallocation space of MobileNetV2 (Sandler et al., 2018) and ResNeXt (Xie et al., 2017) can be found in Appendix. B.

# 3.2.2 CONVOLUTION REALLOCATION SPACE

To reallocate the computation across spatial position, we utilize dilated convolution Yu & Koltun (2015), Li et al. (2019), Li et al. (2018), Cai et al. (2018a). Dilated convolution effects the ERF by performing convolution at sparsely sampled locations. Another good feature of dilated convolution is that dilation introduces no extra parameter and computation. We define a choice block to be a basic unit which consists of multiple dilations and search for the best computation allocation. For ResNet BasicBlock, we only modify the second  $3 \times 3$  convolution. For ResNet Bottleneck, we only modify the center  $3 \times 3$  convolution. We have three candidates in our operation set  $\mathcal{O}$ :  $\{\text{dilated convolution } 3 \times 3 \text{ with dilation rate } i | i = 1, 2, 3\}$ . Across the entire ResNet50 search space, there are therefore  $3^{16} \approx 4 \times 10^7$  possible architectures.

# 3.3 HIERARCHICAL SEARCH FOR OBJECT DETECTION

We propose a hierarchical search procedure to cope with the complex reallocation space. Firstly, the stage space is explored to find the best computation allocation for different resolution. Then, the operation space is explored to further improve the architecture with better spatial allocation.

# 3.3.1 STAGE REALLOCATION SEARCH

To reduce the side effect of weights coupling, we adopt the uniform sampling in supernet training(a.k.a single-path one-shot) (Guo et al., 2019b). After the supernet training, we can validate the allocation strategies  $\tau \in T$  directly on the task detection task. Model accuracy(COCO AP) is defined as  $AP_{val}(\mathcal{N}(\tau ,w))$ . We set the block number constraint  $N$ . We can find the best allocation strategy in the following equation:

![](images/5c848daa13bf71cbca603c9469bed84f71f8f3328f4a17c9e32b3b255af2cb9b.jpg)  
Figure 3: Evaluation of a choice in block operation search approach. As shown in figure, we have partial architecture of block 1 and block 2, and now we need to evaluate the performance of convolution with dilated rate 3 in the third block. We uniform sample the operation of rest blocks to generate a temporary architecture and then evaluate the choice through several temporary architectures.

$$
\tau^ {*} = \arg \max  A P _ {v a l} (\mathcal {N} (\tau , w)). \tag {1}
$$

$$
\sum_ {i = 1} ^ {4} \tau_ {i} = N
$$

# 3.3.2 BLOCK OPERATION SEARCH

Algorithm 1: Greedy operation search algorithm  
Input: number of blocks  $B$ ; Possible operations set of each blocks  $O = \{O_{i} \mid i = 1,2,\dots,B\}$ ; supernet with trained weights  $\mathcal{N}(O,W^{*})$ ; dataset for validation  $D_{val}$ ; evaluation metric  $AP_{val}$ .

Output: Best architecture  $o^*$  
initialize top  $K$  partial architecture  $p = \emptyset$

for  $i = 1,2,\dots,B$  do  
end  
```latex
\[
\begin{array}{l}
p_{\text{extend}} = p \times O_i \quad \# \times \text{denotes Cartesian product} \\
\text{result} = \{(arch, AP) \mid arch \in p_{\text{extend}}, AP = \text{evaluate}(arch)\} \\
p = \text{choose\_topK(result)}
\end{array}
\]
```

Output: Best architecture  $o^{*} = \text{choose\_top1}(p)$ .

By introducing the operation allocation space as in Sec. 3.2.2, we can reallocate the computation across spatial position. Same as stage reallocation search, we train an operation supernet adopting random sampling in each choice block (Guo et al., 2019b). For architecture search process, previous one-shot works use random search (Brock et al., 2017; Bender et al., 2018) or evolutionary search (Guo et al., 2019b). In our approach, We propose a greedy algorithm to make sequential decisions to obtain the final result. We decode network architecture  $o$  as a sequential of choices  $[o_1, o_2, \dots, o_B]$ . In each choice step, the top  $K$  partial architectures are maintained to shrink the search space. We evaluate each candidate operation from the first choice block to the last. The greedy operation search algorithm is shown in Algorithm 1.

The hyper-parameter  $K$  is set equal to 3 in our experiment. We first extend the partial architecture in the first block choice which contains three partial architectures in  $p_{\text{extend}}$ . Then we expand the top 3 partial architectures into the whole length  $B$ , which means that there are  $3 \times 3 = 9$  partial architectures in other block choice. For a specific partial architecture  $\text{arch}$ , we sample the operation of the unselected blocks uniformly for  $c$  architectures where  $c$  denotes mini batch number of  $D_{\text{val}}$ . We validate each architecture on a mini batch and combine the results to generate  $\text{evaluate}(\text{arch})$ . We finally choose the best architecture to obtain  $o^*$ .

# 4 EXPERIMENTS AND RESULTS

# 4.1 DATASET AND IMPLEMENTATION DETAILS

Dataset We evaluate our method on the challenging MS COCO benchmark (Lin et al., 2014). We split the 135K training images trainval135 into 130K images archtrain and 5K images archival. First,

Table 1: Faster RCNN + FPN detection performance on COCO minival for different backbones using our computation reallocation (denoted by 'CR-x'). FLOPs are measured on the whole detector(w/o ROIAlign layer) using the input size  ${800} \times  {1088}$  ,which is the median of the input size on COCO.  

<table><tr><td>Backbone</td><td>FLOPs (G)</td><td>AP</td><td>\( AP_{50} \)</td><td>\( AP_{75} \)</td><td>\( AP_s \)</td><td>\( AP_m \)</td><td>\( AP_l \)</td></tr><tr><td>MobileNetV2</td><td>121.1</td><td>32.2</td><td>54.0</td><td>33.6</td><td>18.1</td><td>34.9</td><td>42.1</td></tr><tr><td>CR-MobileNetV2</td><td>121.4</td><td>33.9</td><td>56.2</td><td>35.6</td><td>19.7</td><td>36.8</td><td>44.8</td></tr><tr><td>ResNet18</td><td>147.7</td><td>32.1</td><td>53.5</td><td>33.7</td><td>17.4</td><td>34.6</td><td>41.9</td></tr><tr><td>CR-ResNet18</td><td>147.6</td><td>33.8</td><td>55.8</td><td>35.4</td><td>18.2</td><td>36.2</td><td>45.8</td></tr><tr><td>ResNet50</td><td>192.5</td><td>36.4</td><td>58.6</td><td>38.7</td><td>21.8</td><td>39.7</td><td>47.2</td></tr><tr><td>CR-ResNet50</td><td>192.7</td><td>38.3</td><td>61.1</td><td>40.9</td><td>21.8</td><td>41.6</td><td>50.7</td></tr><tr><td>ResNet101</td><td>257.3</td><td>38.6</td><td>60.7</td><td>41.7</td><td>22.8</td><td>42.8</td><td>49.6</td></tr><tr><td>CR-ResNet101</td><td>257.5</td><td>40.2</td><td>62.7</td><td>43.0</td><td>22.7</td><td>43.9</td><td>54.2</td></tr></table>

![](images/55efcb1977f609138e3707e00fa998ffb836de9a617d169c27cf92c9b1ceeed9.jpg)  
Figure 4: Architecture sketches. From top to bottom, they are baseline ResNet50, stage reallocation SCR-ResNet50 and final CR-ResNet50.

we train the supernet using archtrain and evaluate the architecture using archival. After the architecture is obtained, we follow other standard detectors (Ren et al., 2015; Lin et al., 2017a) on using ImageNet (Russakovsky et al., 2015) for pre-training the weights of this architecture. The final model is fine-tuned on the whole COCO trainval135 and validated on COCO minival. Another detection dataset VOC (Everingham et al., 2015) is also used. We use VOC trainval2007+trainval2012 as our training dataset and VOC test2007 as our validation dataset.

Implementation details The supernet training setting details can be found in Appendix A. For the training of our searched models, the input images are resized to have a short side of 800 pixels or a long side of 1333 pixels. We use stochastic gradient descent (SGD) as optimizer with 0.9 momentum and 0.0001 weight decay. For fair comparison, all our models are trained for 13 epochs, known as  $1 \times$  schedule (Girshick et al., 2018). We use multi-GPU training over 8 1080TI GPUs with total batch size 16. The initial learning rate is 0.00125 and is divided by 10 at 8 and 11 epochs. Warm-up and synchronized BatchNorm (SyncBN) (Peng et al., 2018) are adopted for both baselines and our searched models.

# 4.2 MAIN RESULTS

# 4.2.1 COMPUTATION REALLOCATION PERFORMANCE

We denote the architecture using our computation reallocation by prefix 'CR-', e.g. CR-ResNet50. Our final architectures have the almost the same FLOPs as the original network(the negligible difference in FLOPs is from the BatchNorm layer and activation layer). As shown in Table 1, our CR-ResNet50 and CR-ResNet101 outperforms the baseline by  $1.9\%$  and  $1.6\%$  respectively. Our CR-ResNet50 and CR-ResNet101 are especially effective for large objects  $(3.5\%, 4.8\%)$  improve

Table 2: Faster RCNN + FPN detection performance on VOC test2007. Our computation reallocation models are denoted by 'CR-x'  

<table><tr><td></td><td>ResNet50</td><td>CR-ResNet50</td><td>ResNet101</td><td>CR-ResNet101</td></tr><tr><td>AP50</td><td>84.1</td><td>85.1</td><td>85.8</td><td>86.5</td></tr></table>

Table 3: Mask RCNN detection and instance segmentation performance on COCO minival for different backbones using our computation reallocation (denoted by 'CR-x'). Box and Seg are the AP (\%) of the bounding box and segmentation results respectively.  

<table><tr><td>Backbone</td><td>FLOPs</td><td>Seg</td><td>\( Seg_s \)</td><td>\( Seg_m \)</td><td>\( Seg_l \)</td><td>Box</td><td>\( Box_s \)</td><td>\( Box_m \)</td><td>\( Box_l \)</td></tr><tr><td>MobileNetV2</td><td>189.5</td><td>30.6</td><td>15.3</td><td>33.2</td><td>44.1</td><td>33.1</td><td>18.8</td><td>35.8</td><td>43.3</td></tr><tr><td>CR-MobileNetV2</td><td>189.8</td><td>31.8</td><td>16.3</td><td>34.3</td><td>42.2</td><td>34.6</td><td>19.9</td><td>37.3</td><td>45.7</td></tr><tr><td>ResNet50</td><td>261.2</td><td>33.9</td><td>17.4</td><td>37.3</td><td>46.6</td><td>37.6</td><td>21.8</td><td>41.2</td><td>48.9</td></tr><tr><td>CR-ResNet50</td><td>261.0</td><td>35.2</td><td>17.6</td><td>38.5</td><td>49.4</td><td>39.1</td><td>22.2</td><td>42.3</td><td>52.3</td></tr><tr><td>ResNet101</td><td>325.9</td><td>35.6</td><td>18.6</td><td>39.2</td><td>49.5</td><td>39.7</td><td>23.4</td><td>43.9</td><td>51.7</td></tr><tr><td>CR-ResNet101</td><td>325.8</td><td>36.7</td><td>19.4</td><td>40.0</td><td>52.0</td><td>41.5</td><td>24.2</td><td>45.2</td><td>55.7</td></tr></table>

ment for  $\mathrm{AP}_l$ ). To understand these improvements, we depict the architecture sketches in Figure. 4. We can find in the stage-level, our Stage CR-ResNet50 reallocate more capacity in deep stage. It reveals the fact that the budget in shallow stage is redundant while the resources in deep stage is limited. This pattern is consistent with ERF as in Figure. 1. In operation-level, dilated convolution with large rates tends to appear in the deep stage. We explain the shallow stage needs more dense sampling to gather exact information while deep stage aims to recognize large object by more sparse sampling. For light backbone, our CR-ResNet18 and CR-MobileNetV2 all improves  $1.7\%$  AP over the baselines. It is worth mentioning that the light backbone achieve all scale improvements.

# 4.2.2 TRANSFERABILITY VERIFICATION

Different dataset We transfer our searched model to another object detection dataset VOC (Everingham et al., 2015). Training details can be found in Appendix. C. We denote the VOC metric  $\mathrm{mAP}@0.5$  as  $\mathrm{AP}_{50}$  for consistency. As shown in Table. 2, our CR-ResNet50 and CR-ResNet101 achieves  $\mathrm{AP}_{50}$  improvement  $1.0\%$  and  $0.7\%$  comparing with the already high baseline.

Different task Segmentation is another task that is highly sensitive to the ERFs (Chen et al., 2017; Hamaguchi et al., 2018; Wang et al., 2018). Therefore, we transfer our computation reallocation network into the instance segmentation task by using the Mask RCNN (He et al., 2017) framework. The experimental results on COCO are shown in Table. 3. The instance segmentation AP of our CR-MobileNetV2, CR-ResNet50 and CR-ResNet101 outperform the baseline respectively by  $1.2\%$ ,  $1.3\%$  and  $1.1\%$  absolute AP. We also achieve bounding box AP improvement by  $1.5\%$ ,  $1.5\%$  and  $1.8\%$  respectively.

# 4.3 ANALYSIS

# 4.3.1 EFFECT OF STAGE REALLOCATION

Our design includes two parts, stage reallocation search and block operation search. In this section, we analyse the effectiveness of stage reallocation search alone. Table. 4 shows the performance comparison between the baseline and the baseline with our stage reallocation search. From the fast MobileNetV2 model to the accurate ResNeXt101, our stage reallocation brings a solid average  $1.0\%$  AP improvement for ResNet (He et al., 2016), MobileNetV2 (Sandler et al., 2018), ResNeXt (Xie et al., 2017). Detailed results can be found in Appendix. B. Figure. 5 shows that our Stage-CR network series yield overall improvements over baselines with negligible difference in computation.

Table 4: COCO minival AP (%) evaluating stage reallocation performance for different networks. Res50 denotes ResNet50, similarly for Res101. ReX50 denotes ResNeXt50, similarly for ReXt101.  

<table><tr><td></td><td>MbileNetV2</td><td>Res18</td><td>Res50</td><td>Res101</td><td>ReX50-32×4d</td><td>ReX101-32×4d</td></tr><tr><td>Baseline AP</td><td>32.2</td><td>32.1</td><td>36.4</td><td>38.6</td><td>37.9</td><td>40.6</td></tr><tr><td>Stage-CR AP</td><td>33.5</td><td>33.4</td><td>37.4</td><td>39.5</td><td>38.9</td><td>41.5</td></tr></table>

![](images/1a34ca4664115b4168470750d393974adf7e102a4d97fb1518eb1a6d29fa2880.jpg)  
Figure 5: Detector FLOPs(G) versus AP on COCO minival. The bold lines and dotted lines are the baselines and our stage computation reallocation models(SCR-) respectively.

![](images/ea49ec6e2930798a1a2a00243c1a445244fcd089a1e024ae907f31a0d9cf9655.jpg)  
Figure 6: Top1 accuracy on ImageNet validation set versus AP on COCO minival. Each dot is a model which has equivalent FLOPs as the baseline.

# 4.3.2 CORRELATIONS BETWEEN CLS. AND DET. PERFORMANCE

Often, a large AP increase could be obtained by simply replacing backbone with stronger network, e.g. from ResNet50 to ResNet101 and then to ResNeXt101. The assumption is that strong network can perform well on both classification and detection tasks. We further explore the performance correlation between these two tasks by a lot of experiments. We draw ImageNet top1 accuracy versus COCO AP correlation in Figure. 6 for different architectures of the same FLOPS. Each dot is a single network architecture. We can easily find that the performance correlation between these two tasks is not strong. This study further shows the gap between these two tasks. And the results validate that a network good for classification is not necessarily good for detection.

# 5 CONCLUSION

In this paper, we present CR-NAS (Computation Reallocation Neural Architecture Search) that can learn computation reallocation strategies across different resolution and spatial position. A two-level reallocation space is proposed for the effective search. A novel hierarchical search procedure is adopted to cope with the complex search space. Extensive experiments show the effectiveness of our approach. Our CR-NAS offers improvements for both fast mobile model and accurate model. Our searched models have great transfer ability among different datasets and tasks. Our CR-NAS can be used as a plugin to other detection backbones to get consistent gain.

# REFERENCES

Gabriel Bender, Pieter-Jan Kindermans, Barret Zoph, Vijay Vasudevan, and Quoc Le. Understanding and simplifying one-shot architecture search. In International Conference on Machine Learning, pp. 549-558, 2018.  
Andrew Brock, Theodore Lim, James M Ritchie, and Nick Weston. Smash: one-shot model architecture search through hypernetworks. arXiv preprint arXiv:1708.05344, 2017.  
Han Cai, Tianyao Chen, Weinan Zhang, Yong Yu, and Jun Wang. Efficient architecture search by network transformation. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018a.  
Han Cai, Ligeng Zhu, and Song Han. Proxylessnas: Direct neural architecture search on target task and hardware. arXiv preprint arXiv:1812.00332, 2018b.  
Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence, 40(4): 834-848, 2017.  
Yukang Chen, Tong Yang, Xiangyu Zhang, Gaofeng Meng, Chunhong Pan, and Jian Sun. Detnas: Neural architecture search on object detection. arXiv preprint arXiv:1903.10979, 2019.  
Xiangxiang Chu, Bo Zhang, Ruijun Xu, and Jixiang Li. Fairnas: Rethinking evaluation fairness of weight sharing neural architecture search. arXiv preprint arXiv:1907.01845, 2019.  
Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In Proceedings of the IEEE international conference on computer vision, pp. 764-773, 2017.  
Mark Everingham, SM Ali Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The Pascal visual object classes challenge: A retrospective. International journal of computer vision, 111(1):98-136, 2015.  
Golnaz Ghiasi, Tsung-Yi Lin, and Quoc V Le. Nas-fpn: Learning scalable feature pyramid architecture for object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7036-7045, 2019.  
Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, pp. 1440-1448, 2015.  
Ross Girshick, Ilija Radosavovic, Georgia Gkioxari, Piotr Dollár, and Kaiming He. Detector, 2018.  
Minghao Guo, Zhao Zhong, Wei Wu, Dahua Lin, and Junjie Yan. Irlas: Inverse reinforcement learning for architecture search. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9021-9029, 2019a.  
Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei, and Jian Sun. Single path one-shot neural architecture search with uniform sampling. arXiv preprint arXiv:1904.00420, 2019b.  
Ryuhei Hamaguchi, Aito Fujita, Keisuke Nemoto, Tomoyuki Imaizumi, and Shuhei Hikosaka. Effective use of dilated convolutions for segmenting small object instances in remote sensing imagery. In 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 1442-1450. IEEE, 2018.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Kaiming He, Georgia Gkioxari, Piotr Dólar, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pp. 2961-2969, 2017.  
Yanghao Li, Yuntao Chen, Naiyan Wang, and Zhaoxiang Zhang. Scale-aware trident networks for object detection. arXiv preprint arXiv:1901.01892, 2019.

Zeming Li, Chao Peng, Gang Yu, Xiangyu Zhang, Yangdong Deng, and Jian Sun. Detnet: A backbone network for object detection. arXiv preprint arXiv:1804.06215, 2018.  
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dólar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740-755. Springer, 2014.  
Tsung-Yi Lin, Piotr Dólar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2117-2125, 2017a.  
Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dólar. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pp. 2980-2988, 2017b.  
Hanxiao Liu, Karen Simonyan, and Yiming Yang. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055, 2018.  
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C Berg. Ssd: Single shot multibox detector. In European conference on computer vision, pp. 21-37. Springer, 2016.  
Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.  
Xin Lu, Buyu Li, Yuxin Yue, Quanquan Li, and Junjie Yan. Grid r-cnn. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7363-7372, 2019.  
Wenjie Luo, Yujia Li, Raquel Urtasun, and Richard Zemel. Understanding the effective receptive field in deep convolutional neural networks. In Advances in neural information processing systems, pp. 4898-4906, 2016.  
Hao Chen Peng Wang Zhi Tian Ning Wang, Yang Gao and Chunhua Shen. Nas-fcos: Fast neural architecture search for object detection. arXiv preprint arXiv:1906.04423, 2019.  
Chao Peng, Tete Xiao, Zeming Li, Yuning Jiang, Xiangyu Zhang, Kai Jia, Gang Yu, and Jian Sun. Megdet: A large mini-batch object detector. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6181-6189, 2018.  
Junran Peng, Ming Sun, Zhaoxiang Zhang, Tieniu Tan, and Junjie Yan. Efficient neural architecture transformation searchin channel-level for object detection. arXiv preprint arXiv:1909.02293, 2019.  
Hieu Pham, Melody Y Guan, Barret Zoph, Quoc V Le, and Jeff Dean. Efficient neural architecture search via parameter sharing. arXiv preprint arXiv:1802.03268, 2018.  
Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 779-788, 2016.  
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems, pp. 91-99, 2015.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211-252, 2015.  
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. *Mobilenetv2: Inverted residuals and linear bottlenecks*. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 4510-4520, 2018.  
Panqu Wang, Pengfei Chen, Ye Yuan, Ding Liu, Zehua Huang, Xiaodi Hou, and Garrison Cottrell. Understanding convolution for semantic segmentation. In 2018 IEEE winter conference on applications of computer vision (WACV), pp. 1451-1460. IEEE, 2018.

Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, and Kurt Keutzer. Fbnet: Hardware-aware efficient convnet design via differentiable neural architecture search. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 10734-10742, 2019.  
Saining Xie, Ross Girshick, Piotr Dólar, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1492-1500, 2017.  
Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions. arXiv preprint arXiv:1511.07122, 2015.  
Xizhou Zhu, Han Hu, Stephen Lin, and Jifeng Dai. Deformable convnets v2: More deformable, better results. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9308-9316, 2019.  
Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. arXiv preprint arXiv:1611.01578, 2016.  
Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V Le. Learning transferable architectures for scalable image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8697-8710, 2018.
