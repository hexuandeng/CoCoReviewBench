# EFFICIENT SELF-SUPERVISED VISION TRANSFORMERS FOR REPRESENTATION LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

This paper investigates two techniques for developing efficient self-supervised vision transformers (EsViT) for visual representation learning. First, we show through a comprehensive empirical study that multi-stage architectures with sparse self-attention can significantly reduce modeling complexity but with a cost of losing the ability to capture fine-grained correspondences between image regions. Second, we propose a new pre-training task of region matching which allows the model to capture fine-grained region dependencies and as a result significantly improves the quality of the learned vision representations. Our results show that combining the two techniques, EsViT achieves  $81.3\%$  top-1 accuracy on the ImageNet linear probe evaluation, outperforming prior arts with around an order magnitude of higher throughput. When transferring to downstream linear classification tasks, EsViT outperforms its supervised counterpart on 17 out of 18 datasets. The code and models will be publicly available.

# 1 INTRODUCTION

Self-supervised learning (SSL) with Transformers (Vaswani et al., 2017) has become a de facto standard of model choice in natural language processing (NLP). The dominant approaches such as GPT (Radford et al., 2018) and BERT (Devlin et al., 2019) are pre-training on a large text corpus and then fine-tuning to various smaller task-specific datasets, showing superior performance. Larger Transformers pre-trained with larger-scale language datasets often lead to a stronger generalization ability, demonstrated by improved performance in downsteam tasks (with no sign of performance saturation yet), as exemplified in GPT-3 (Brown et al., 2020).

In computer vision (CV), however, self-supervised visual representation learning is still dominated by convolutional neural networks (CNNs). Sharing a similar goal/spirit with NLP, SSL in CV aims to learn general-purpose image features from raw pixels without relying on manual supervisions, and the learned networks are expected to serve as the backbone of various downstream tasks such as classification, detection and segmentation. Recently, impressive performance have been achieved by CNN-based SSL, outperforming state-of-the-art (SoTA) fully-supervised pre-training methods (He et al., 2020; Caron et al., 2020) on tasks with a limited number of labels. The key to success is view-level learning: maximizing agreement of learned representations between differently augmented views of the same example. Recent works, including SimCLR-v2 (Chen et al., 2020d), BYOL (Grill et al., 2020) and SwAV (Caron et al., 2020), have scaled up the CNN-based models to hundreds of millions of parameters. However, SSL has not enjoyed the same scaling success in CV as that in NLP.

Several attempts have been made to close the gap by combining SSL with Transformer and self-attention architectures. Early works include Selfie (Trinh et al., 2019), which generalizes the concept of masked language modeling of BERT for images. The idea has been recently revisited in Vision Transformer (ViT) (Dosovitskiy et al., 2021) via pre-training on a much larger scale dataset, e.g., JFT-300M. ImageGPT (iGPT) (Chen et al., 2020b) generalizes the concept of auto-regressive language modeling of GPT for images, showing encouraging ImageNet recognition accuracy with a large model size. Contrastive learning with ViT has also been studied very recently in DINO (Caron et al., 2021) and MoCo-v3 (Chen et al., 2021), where new SoTA result by linear probe evaluation on ImageNet-1K is achieved, by exhaustively consuming computation resource on full self-attention operators with long sequences of split image patches.

![](images/bb1dae6d697be6f6d189d8528d7817192aa081e658906e52bedd0e5225541969.jpg)  
Figure 1: Efficiency vs accuracy comparison under the linear classification protocol on ImageNet. Left: Throughput of all SoTA SSL vision systems, circle sizes indicates model parameter counts; Right: performance over varied parameter counts for models with moderate (throughout/#parameters) ratio. Please refer Section 4.1 for details.

![](images/d5d3fdf896e42383fdf2fca8b05f0a50ecdf6bd824bf47a5dd3ee9c58929c5c2.jpg)

Aiming to improve the efficiency of Transformer-based SSL, this paper presents Efficient self-supervised Vision Transformers (EsViT), by using a multi-stage architecture and a region-based pre-training task for self-supervised representation learning. Our main findings and contributions can be summarized as follows:

- An intriguing property of self-supervised monolithic Transformers is firstly reported in our paper: automatic discovery of semantic correspondence between local regions.  
- We present the first comprehensive empirical study to show the pros and cons of multi-stage vision Transformer architectures for SSL. Though greatly reducing compute complexity, we find that the multi-stage architecture causes the loss of this property.  
- A region matching pre-train task is proposed to alleviate this issue, and further improve the learned representations and attentions.  
- We validate the new EsViT, which combines the two techniques, on a range of tasks. It significantly reduces the cost in building SoTA SSL vision systems, as summarized in Figure 1, and shows better scaling performance on accuracy vs. throughput and model size. Under the linear evaluation protocol, EsViT achieves  $81.3\%$  top-1 accuracy, showing the best performance compared with all systems, and is  $3.5 \times$  parameter-efficient and has at least  $10 \times$  higher throughput than previous SoTA (81.0%, MoCo-v3 with ViT-BN-L/7 (Chen et al., 2021)). Compared with its supervised counterpart Swin Transformers (Liu et al., 2021), EsViT shows superior performance on 17 out 18 datasets, when transferring the learned representations to downstream linear classification tasks.

# 2 METHODS

Transformer-based SSL methods emerge very recently to lead the state-of-the-art performance on the ImageNet linear probe task (Chen et al., 2021; Caron et al., 2021). It inherits the successes from (1) monolithic Transformer architectures that dominate in NLP (Devlin et al., 2019; Radford et al., 2018), and (2) instance-level contrastive learning objectives that demonstrate arguably the best SSL performance in computer vision (Chen et al., 2020c). Though simple and effective, the existing Transformer-based SSL methods require a large amount of compute resources (e.g.,  $>1.7$  TPU years of training) to reach SoTA performance. We believe that the SSL system efficiency is highly related to two ingredients: the network architecture and the pre-train task. To strike for a better tradeoff between accuracy and efficiency, we present EsViT, showing better synergy of networks (a multi-stage Transformer architecture) and pre-train tasks (a non-contrastive region-matching task).

# 2.1 NETWORK ARCHITECTURES: FROM MONOLITHIC TO MULTI-STAGE VIT BACKBONE

Multi-stage ViT. This paper presents the first empirical study of multi-stage Transformer architectures (Vaswani et al., 2021; Wang et al., 2021; Liu et al., 2021; Zhang et al., 2021; Wu et al., 2021) for SSL. Each stage consists of a patch merging/embedding module, and a Transformer with sparse self-attention module. (i) The patch merging module plays a slightly different role in different

stages. In the first stage, it splits an input RGB image into non-overlapping patches. Each patch is treated as a "token", constructed as a concatenation of the raw pixel RGB values, which is further projected into a  $C$ -dimension feature. In the later stage, the patch merging module concatenates the features of each group of  $2 \times 2$  neighboring patches, and applies a linear layer on the  $4C$ -dimensional concatenated features. This reduces the number of tokens by a multiple of  $2 \times 2 = 4$ , and the output dimension is set to  $2C$ . (ii) A Transformer with sparse self-attention module are then employed to enable interactions among the merged features. The two modules above are repeated for multiple times, typically 4 times, resulting in a multi-stage ViT. As a result, a hierarchical representation is generated: the number of tokens is reduced and the feature dimension (and the number of heads in self-attention) of each token is increased, as the network gets deeper. An overview comparison of the monolithic and multi-stage Transformer architectures for SSL is illustrated in Figure 7 in Appendix.

An intriguing property of self-supervised monolithic ViT. Though straightforward in implementation, changing from monolithic to multi-stage architecture without careful treatments may lose some desirable properties of self-supervised Transformers, as discussed in the following. In our study, we first empirically note an intriguing property of self-supervised monolithic ViT(Caron et al., 2021): the pre-trained model exhibits a very strong ability to automatically discovers correspondences, even without a region-level matching objective specified in training. To illustrate this property, in Section C.7 in Appendix, we quantitatively show that a self-supervised monolithic ViT yields  $95\%$  accuracy in successfully identifying the correct region-to-region correspondences for 50K images on the ImageNet validation dataset. However, simply replacing the network with a multi-stage Transformer yields only  $66\%$  accuracy. This significant degradation (absolute  $29\%$  accuracy drop) reveals the loss of the correspondence learning property. We first raise this critical problem, and believe that it has a large impact on the pre-trained model's performance in various downstream tasks.

# 2.2 PRE-TRAINING TASKS: DELVING INTO VIEWS WITH REGIONS

We employ a non-contrastive learning framework to build our SSL method. Specifically, Self-distillation with no labels (DINO) (Caron et al., 2021) is considered. It leverages the knowledge distillation learning paradigm where a student network  $g_{\theta_s}$  is trained to match the output of a given teacher network  $g_{\theta_t}$ , parameterized by  $\theta_s$  and  $\theta_t$  respectively. The neural network  $g$  is composed of a backbone  $f$  (e.g., Transformers or ConvNets), and of a projection head  $h$ :  $g = h \circ f$ . The features used in downstream tasks are the output of backbone  $f$ . In SSL, different augmented views  $\tilde{\pmb{x}}$  of an image  $\pmb{x}$  are fed into backbone network to obtain feature maps  $z = f(\tilde{\pmb{x}})$ . Two MLP heads followed by softmax per network further convert the feature vectors  $z \in z$  into probability vectors  $p = h(z)$ ; one head for view-level and the other head for region-level, respectively.

More precisely, from a given image, we generate a set  $\mathcal{V}$  of different views $^1$  following (Caron et al., 2021). The resulting feature map at the top layer for each view is  $z = [z_1, \ldots, z_T]$ , where  $T$  is the sequence length, and  $z_i$  is a region-level representation for the local patch at position  $i$ . Average pooling is applied to obtain the view-level representation  $\bar{z} = \operatorname{avg-pool}(z)$ .

View-level task Given the augmented view set for student  $\nu$  and teacher  $\nu^{*}$ , a set of pairs  $\mathcal{P} = \{(s,t) | \tilde{\boldsymbol{x}}_s \in \nu, \tilde{\boldsymbol{x}}_t \in \nu^* \text{ and } s \neq t\}$  is constructed to perform cross-view prediction tasks. We consider the pre-training task at the view level proposed by (Caron et al., 2021):

$$
\mathcal {L} _ {V} = \frac {1}{| \mathcal {P} |} \sum_ {(s, t) \in \mathcal {P}} \mathcal {M} _ {V} (s, t), \text {w i t h} \mathcal {M} _ {V} (s, t) = - p _ {s} \log p _ {t}, \tag {1}
$$

where  $p_{s} = h(\bar{z}_{s})$  and  $p_{t} = h(\bar{z}_{t})$  are the probability output of an MLP head  $h$  over the view-level representations  $\bar{z}_{s}$  and  $\bar{z}_{t}$ , learned by student and teacher, respectively. In DINO, ViT/DeiT are considered, hence the view-level representation is the feature of the [CLS] token.

Region-level task In (Caron et al., 2021), the  $\mathcal{L}_V$  encourages "local-to-global" correspondences only at a coarse level: the large crop and the small crop are matched in the view level, leaving region-to-region correspondence unspecified. In monolithic Transformers, the drop paths and skip connections from low-level features to high-level features help the latter to remain discriminative,

thus maintain good region-matching performance. However, such a property gets diluted due to the merging operators in multi-stage Transformers. As shown in our experiments later, training a multi-stage network with  $\mathcal{L}_V$  only indeed results in sub-optimal representations, though network efficiency is greatly improved.

Further, it could be a waste of computation not to leverage region-level features  $z$  that are computed in the process of extracting view-level feature. Inspired by the success of masked language modeling task in BERT, we argue that it is important to have region-level pre-training task for computer vision, so that the model can (1) amortize the computation and fully leverage the extracted region-level features, and (2) take into account the co-occurrences/structures between local features. Unfortunately, directly performing masked patch prediction (MPP) for the multi-stage Transformer architecture is infeasible, as the one-to-one correspondences between the input visual tokens and output features get diluted due to the merging operation. Even for monolithic architectures, MPP has not been proved effective in computer vision, as empirically shown in (Dosovitskiy et al., 2021).

To address this problem, we propose a non-contrastive, region-matching method that directly works at the level of local features by taking into account their correspondences:

$$
\mathcal {L} _ {R} = \frac {1}{| \mathcal {P} |} \sum_ {(s, t) \in \mathcal {P}} \mathcal {M} _ {R} (s, t), \text {w i t h} \mathcal {M} _ {R} (s, t) = - \frac {1}{T} \sum_ {i = 1} ^ {T} p _ {j ^ {*}} \log p _ {i}, j ^ {*} = \arg \max  _ {j} \frac {z _ {i} ^ {T} z _ {j}}{\| z _ {i} \| \| z _ {j} \|}, \tag {2}
$$

where  $p_i = h'(z_i)$  and  $p_j = h'(z_j)$  are the probability outputs of a new MLP head  $h'$  over the local features of student  $z_i \in \mathbf{z}_s$  and teacher  $z_j \in \mathbf{z}_t$ , respectively.  $j^*$  is the index of the feature in  $z_t$  that best matches the  $i$ -th feature in  $z_s$ , in the sense of highest cosine similarity.

The overall pre-training objective of EsViT is  $\mathcal{L} = \mathcal{L}_R + \mathcal{L}_V$ , we learn to match the feature distributions at both the view and region levels by minimizing the cross-entropy loss w.r.t. the parameters of the student network  $g_{\theta_s}$ . A visual illustration is in Figure 2, and the full algorithm is in Appendix. We updates teacher/student network alternatively: (i) Given a fixed teacher network, the student network is updated by minimizing the full cross-entropy loss:  $\pmb{\theta}_{s}\gets \arg \min_{\pmb{\theta}_{s}}\mathcal{L}(s,t;\pmb{\theta}_{s})$ . (ii) The teacher model is updated as an exponential mov

ing average (EMA) of the student weights  $\pmb{\theta}_t\gets \lambda \pmb{\theta}_t + (1 - \lambda)\pmb{\theta}_s$  , with  $\lambda$  following a cosine schedule from 0.996 to 1 during training. By default, the full objective  $\mathcal{L}$  is used from the beginning. One can also load a checkpoint trained by  $\mathcal{L}_V$  only, and add  $\mathcal{L}_R$  for continual pre-training, which is shown effective in boosting performance in our experiments.

![](images/d9461e4f6a1cde20e81de1370318c245107e4591563ad11462ab9bf1ba3e30a4.jpg)  
Figure 2: Pre-training objectives, including view-level (left) and region-level (right) prediction.

Computational overhead Note that applying  $\mathcal{L}_R$  on the traditional monolithic Transformer architecture can be prohibitively computationally expensive, as it requires  $\mathcal{O}(T^2)$  to compute  $\mathcal{L}_R$ . For a typical image of resolution  $224 \times 224$ , the feature map length of ViT/DeiT (with patch size 16) at the top layer is  $T = 196$ , while the multi-stage architecture yields  $T = 49$ , which requires 3 times less compute in computing  $\mathcal{L}_R$ . To empirically illustrate this, we show in Appendix Section C.2 that  $\mathcal{L}_R$  adds acceptable extra memory and computational cost (around 1.2 and  $1.05 \times$ , respectively) for multi-stage Transformers, while it will quickly go out-of-memory for monolithic Transformers when the batch size is increased.

# 3 RELATED WORKS

Relation to mask prediction tasks We can consider the proposed  $\mathcal{L}_R$  as a proxy to mimic masked language modeling in BERT, where the "ground-truth" local token is a soft label provided by the teacher network, while the student network makes predictions to match that target, based on the context of regions in a different augmented view. Importantly, our  $\mathcal{L}_R$  considers softmax with cross-entropy in the objective, rather than MSE as in MPP. A very sharp teacher distribution is used by choosing small temperatures. This encourages the model to focus on the salient dimensions, rather than waste modeling capability on training short-range dependencies and high-frequency details (Ramesh et al., 2021).

Relation to DenseCL The proposed  $\mathcal{L}_R$  mostly related to DenseCL (Wang et al., 2020b) in that the region correspondences in both methods are determined as the two most similar grid features. One critical difference is that DenseCL is a contrastive region-matching task, while our  $\mathcal{L}_R$  is a non-contrastive region-matching task, where no negative samples/queue is needed. This technical difference has a significant impact on the downstream task performance. We find that  $\mathcal{L}_R$  is particularly effective in serving our goal to improve image classification performance and build efficient & affordable SoTA SSL system; In contrast, DenseCL degrades the classification performance.

Relation to other region-level tasks The ideas of leveraging local region-level pre-training tasks for visual representation learning have been explored for ConvNets (Misra & Maaten, 2020; Xiong et al., 2020; Wang et al., 2020b; Xie et al., 2021a; Yang et al., 2021; Xie et al., 2021c). We summarize the differences in three aspects: (i) Motivation. Our region-matching task  $\mathcal{L}_R$  aims to recover the lost property of automatic correspondence learning in self-supervised monolithic Transformers, while most existing region-level tasks aim to improve dense visual prediction tasks. (ii) Technical difference. Our  $\mathcal{L}_R$  is a non-contrastive region-matching task, while others are contrastive learning. (iii) Empirical performance. Most region-level tasks improve dense visual prediction tasks but sacrifice their image classification performance, while  $\mathcal{L}_R$  consistently improves classification performance. Among them, EsViT training method achieves the best ImageNet linear probe performance with minimum computational overhead. For detailed comparisons, please refer to Table 8 in Appendix.

Self-supervised vision Transformers. The research on Transformer-based self-supervised representation learning just scratches the tip of the iceberg, and only a few attempts are made on this topic. ImageGPT (Chen et al., 2020b) and MoCo-v3 (Chen et al., 2021) dedicate huge compute resource with large models to exploring the frontier. DINO (Caron et al., 2021) achieves comparable performance of large self-supervised ConvNets using small/medium-size Transformers. The proposed EsViT further pursues efficient and affordable solutions to self-supervised vision Transformers. For more general related works on Transformers for vision tasks and self-supervised ConvNets, please refer to Section B in Appendix.

# 4 EXPERIMENTAL RESULTS

We describe the experimental settings in Appendix Section C.3, and evaluate the proposed EsViT to answer three questions: Q1: How does EsViT perform on standard ImageNet benchmark compared to SoTA methods? Q2: How effective EsViT is when transferring to downstream tasks? Q3: What are the design choices and empirical contributions of  $\mathcal{L}_R$ ? Q4: When does the intriguing property of self-supervised Transformers exist, including learned correspondence and attentions?

# 4.1 COMPARISONS WITH PRIOR ART ON IMAGENET

We report top-1 linear probe and  $k$ -NN classification accuracy on the ImageNet validation set. Table 1 presents comparisons with all SoTA SSL vision systems across various network architectures. Please refer to Figure 1 for visual comparisons over scaling parameter counts and throughput. Our main findings are summarized below.

Comparisons with self-supervised Transformers. The DINO- and MoCo-based ViT has higher accuracy and smaller models than iGPT, under the same linear probing protocol and training data. At the similar level of model size and compute complexity, the proposed EsViT improve SoTA methods DINO/MoCo-v3 by a large margin: EsViT (Swin-B) outperforms DINO (ViT-B/16) by  $2.2\%$  linear probe accuracy and  $2.8\%$ $k$ -NN accuracy in absolute values. EsViT (Swin-B) even performs slightly better than DINO (ViT-B/8) ( $0.3\%$  higher linear probe accuracy and  $1.5\%$  higher  $k$ -NN accuracy), with  $4\times$  higher throughput. MoBY (Xie et al., 2021b) is a con-current work that investigates multi-stage ViT in SSL. With the same architecture Swin-T, our EsViT pre-training tasks significantly outperform MoBY, showing  $3\%$  higher accuracy. In EsViT, longer sequences in self-attention is implemented by increasing the window size. We experiment this by considering a window size of  $W = 14$ . Overall, the proposed EsViT (Swin-B/W=14) shows the best performance (top-1 accuracy  $81.3\%$ , top-5 accuracy  $95.5\%$ ,  $k$ -NN accuracy  $79.3\%$ ), compared with all systems, and is  $3.5\times$  parameter-efficient and has at least  $10\times$  higher throughput than previous SoTA MoCo-v3.

Comparisons with big ConvNets. We compare with the SoTA big ResNets reported by SimCLR-v2 (Chen et al., 2020d), BYOL (Grill et al., 2020) and SwAV (Caron et al., 2020). Among them, the best accuracy  $79.8\%$  under the linear probing protocol is reported by SimCLR-v2 with SK-ResNet,

Table 1: Comparison with SoTA across different architectures on ImageNet linear probing. ViT-BN is ViT that has BatchNorm (Frankle et al., 2020), and “/P” denotes a patch size of  $P \times P$ . “~” indicates through-puts estimated by comparing different papers, detailed in Appendix.† The mask patch prediction in (Dosovitskiy et al., 2021) is pre-trained on JFT-300M and end-to-end fine-tuned in ImageNet, which we append as a reference.  

<table><tr><td>Method</td><td>#Parameters ↓</td><td>Throughput (Image/s) ↑</td><td>Linear ↑</td><td>k-NN ↑</td></tr><tr><td colspan="5">SoTA SSL methods with Big ConvNets</td></tr><tr><td>SwAV, RN50w5 (Caron et al., 2020)</td><td>586</td><td>76</td><td>78.5</td><td>67.1</td></tr><tr><td>BYOL, RN200w2 (Grill et al., 2020)</td><td>250</td><td>123</td><td>79.6</td><td>73.9</td></tr><tr><td>SimCLR-v2, RN152w3+SK (Chen et al., 2020d)</td><td>794</td><td>46</td><td>79.8</td><td>73.1</td></tr><tr><td colspan="5">Skyline methods with excessively long sequences for self-attention</td></tr><tr><td>DINO, DeiT-S/8 (Caron et al., 2021)</td><td>21</td><td>180</td><td>79.7</td><td>78.3</td></tr><tr><td>DINO, ViT-B/8 (Caron et al., 2021)</td><td>85</td><td>63</td><td>80.1</td><td>77.4</td></tr><tr><td>MoCo-v3, ViT-B-BN/7 (Chen et al., 2021)</td><td>85</td><td>~63</td><td>79.5</td><td>-</td></tr><tr><td>MoCo-v3, ViT-L-BN/7 (Chen et al., 2021)</td><td>304</td><td>~17</td><td>81.0</td><td>-</td></tr><tr><td>iGPT, iGPT-XL (Chen et al., 2020b)</td><td>6801</td><td>-</td><td>72.0</td><td>-</td></tr><tr><td>EsViT, Swin-S/W = 14</td><td>49</td><td>383</td><td>80.8</td><td>79.1</td></tr><tr><td>EsViT, Swin-B/W = 14</td><td>87</td><td>254</td><td>81.3</td><td>79.3</td></tr><tr><td colspan="5">Transformer-based SSL, with moderate sequence length for self-attention</td></tr><tr><td>Masked Patch Pred., ViT-B/16 (Dosovitskiy et al., 2021)</td><td>85</td><td>312</td><td>79.9†</td><td>-</td></tr><tr><td>DINO, DeiT-S/16 (Caron et al., 2021)</td><td>21</td><td>1007</td><td>77.0</td><td>74.5</td></tr><tr><td>DINO, ViT-B/16 (Caron et al., 2021)</td><td>85</td><td>312</td><td>78.2</td><td>76.1</td></tr><tr><td>MoCo-v3, ViT-B/16 (Chen et al., 2021)</td><td>85</td><td>312</td><td>76.7</td><td>-</td></tr><tr><td>MoCo-v3, ViT-H-BN/16 (Chen et al., 2021)</td><td>632</td><td>~32</td><td>79.1</td><td>-</td></tr><tr><td>MoBY, Swin-T (Xie et al., 2021b)</td><td>28</td><td>808</td><td>75.1</td><td>-</td></tr><tr><td>EsViT, Swin-T</td><td>28</td><td>808</td><td>78.1</td><td>75.7</td></tr><tr><td>EsViT, Swin-S</td><td>49</td><td>467</td><td>79.5</td><td>77.7</td></tr><tr><td>EsViT, Swin-B</td><td>87</td><td>297</td><td>80.4</td><td>78.9</td></tr></table>

where Selective Kernel (SK) (Li et al., 2019c) is a form of attention to enhance CNNs. It is clear in Figure 1 (b) that all ConvNets-based SSL methods show an envelope in the regime of scaling up model sizes after passing 500M. EsViT achieves better accuracy than their highest envelope, with  $16 \times$  less model parameters and  $8 \times$  higher throughput.

# 4.2 TRANSFER LEARNING

We also conduct transfer learning in downstream tasks to evaluate the quality of learned representations. Two sets of tasks are considered:

- Classification on a suite of 18 small datasets. As exemplified in (Radford et al., 2021), it is a common and clean approach to evaluate a learned representation by fitting a linear classifier on the representation and measuring its performance across multiple datasets. We study 18 datasets used in (Radford et al., 2021). Automatic hyper-parameter tuning is considered to ensure fairness of comparison. Besides averaged scores, we report # wins as the number of datasets on which the model outperforms its supervised counterpart. Detailed dataset description and settings are in Appendix.  
- Detection and segmentation on COCO. Different from previous monolithic self-supervised ViT, the multi-stage architecture in EsViT can be readily used for dense visual tasks that require hierarchical feature representations.

Comparison with supervised counterparts. We compare with the supervised-learning Swin, whose checkpoints are downloaded from the official codebase<sup>2</sup>. Figure 3 shows the classification results of Swin-S, EsViT consistently outperforms its supervised variant, often by a large margin. Similar conclusions are drawn for other model sizes. On COCO detection and segmentation task, however, comparable results with the supervised counterpart are obtained, as shown in Table 2 for Swin-T. We hypothesize this is related to the non-constrastive nature of EsViT, as explained later.

Effects of larger, less-curated pre-train datasets. The performance of Transformer-based SSL research has thus far been limited to highly curated pre-train data such as ImageNet-1K. To push the frontier in leveraging large amounts of unlabeled data, we explore the effects of pre-training from larger, less-curated image datasets: WebVision-v1 (Li et al., 2017), OpenImages-v4 (Kuznetsova

![](images/a7fe2bed7aee3394bf3cdfa363fcbecb62aa12aff7e848fd66f063f6d031ed87.jpg)  
Figure 3: Transfer learning to a wide variety of datasets. Fitting a linear classifier on EsViT's features outperforms using its supervised counterpart on 17 out of 18 datasets.

Table 2: COCO Detection & Segmentation.  

<table><tr><td rowspan="3">Supervised EsViT</td><td>APbb</td><td>APbb50</td><td>APbb75</td></tr><tr><td>46.0</td><td>68.1</td><td>50.3</td></tr><tr><td>46.2</td><td>68.0</td><td>50.6</td></tr><tr><td rowspan="3">Supervised EsViT</td><td>APmb</td><td>APmb50</td><td>APmb75</td></tr><tr><td>41.6</td><td>65.1</td><td>44.9</td></tr><tr><td>41.6</td><td>64.9</td><td>44.8</td></tr></table>

Table 3: Impact of the pre-train datasets.  

<table><tr><td rowspan="2">Pre-train Data</td><td colspan="2">ImageNet-1K</td><td colspan="2">18 Datasets</td></tr><tr><td>Linear</td><td>k-NN</td><td>Scores</td><td># Wins</td></tr><tr><td>Supervised</td><td>-</td><td>-</td><td>77.29</td><td>-</td></tr><tr><td>ImageNet-1K</td><td>78.0 (77.1)</td><td>75.7 (73.7)</td><td>80.66</td><td>16</td></tr><tr><td>WebVision-v1</td><td>75.9 (75.4)</td><td>71.2 (69.4)</td><td>80.00</td><td>14</td></tr><tr><td>OpenImages-v4</td><td>70.6 (69.6)</td><td>62.0 (60.3)</td><td>77.97</td><td>10</td></tr><tr><td>ImageNet-22K</td><td>75.0 (73.5)</td><td>67.9 (66.1)</td><td>81.03</td><td>17</td></tr></table>

et al., 2020) and ImageNet-22K (Deng et al., 2009), described in Appendix. The pre-train epochs on different datasets are adjusted so that all models see a similar number of augmented views. We summarize the results in Table 3 and would like to emphasize the following findings. First,  $\mathcal{L}_R$  improves  $\mathcal{L}_V$  (shown in parentheses) on all datasets. Second, all EsViT pre-trained checkpoints outperform supervised checkpoint in downstream classification tasks, but performance varies a lot, with ImageNet-22K checkpoint showing the best transfer ability. Third, ImageNet-1K pre-trained model shows the best ImageNet-1K linear probe performance. We hypothesize that it is not only the size of pre-train dataset matters, but also the distribution of image classes matters: more diverse and well-balanced distribution results in a stronger generalization ability.

# 4.3 DISCUSSION ON THE NON-CONTRASTIVE REGION-MATCHING TASK

Compatibility with various network architectures. We investigate ResNet-50 and different efficient sparse Transformers in Table 4. DeiT is shown as a baseline reference. Batch size  $= 1024$  in this experiment. To ensure fair comparison, we modify all into a 4-stage architecture with the number of Transformer layers in each stage as 2-2-6-2. We see that  $\mathcal{L}_R$  improves all network architectures, including ResNet-50, Swin (Liu et al., 2021), ViL (Zhang et al., 2021), CvT (Wu et al., 2021) and PvT (Wang et al., 2021). Though directly adding  $\mathcal{L}_R$  to monolithic ViT is computationally infeasible, we uniformly sampled top-layer grid features of DeiT and then add  $\mathcal{L}_R$ , but did not observe performance improvement. This is partly because the monolithic

ViT itself already has a good corresponding ability, an extra region-matching task does not provide new learning signals. As compared in Appendix Table 13 with the ResNet-50 backbone, EsViT learning method shows the highest accuracy, compared with existing SSL methods.

Table 4: Different architectures with and without  $\mathcal{L}_R$ . DeiT and ResNet-50 are shown as references.  $^\dagger$  indicates numbers reported in (Caron et al., 2021).  

<table><tr><td>Method</td><td>#Param.</td><td>Im./s</td><td>Pre-train tasks</td><td>Linear</td><td>k-NN</td></tr><tr><td>DeiT</td><td>21</td><td>1007</td><td>LV</td><td>75.9</td><td>73.2</td></tr><tr><td rowspan="3">R-50</td><td rowspan="3">24</td><td rowspan="3">1237</td><td>LV</td><td>75.3†</td><td>67.5†</td></tr><tr><td>LV</td><td>75.0</td><td>69.3</td></tr><tr><td>LV+LR</td><td>75.7</td><td>71.2</td></tr><tr><td rowspan="2">Swin</td><td rowspan="2">28</td><td rowspan="2">808</td><td>LV</td><td>77.1</td><td>73.7</td></tr><tr><td>LV+LR</td><td>77.6</td><td>75.4</td></tr><tr><td rowspan="2">ViL</td><td rowspan="2">28</td><td rowspan="2">386</td><td>LV</td><td>77.3</td><td>73.9</td></tr><tr><td>LV+LR</td><td>77.5</td><td>74.5</td></tr><tr><td rowspan="2">CvT</td><td rowspan="2">29</td><td rowspan="2">848</td><td>LV</td><td>77.6</td><td>74.8</td></tr><tr><td>LV+LR</td><td>78.5</td><td>76.7</td></tr><tr><td rowspan="2">PvT</td><td rowspan="2">24</td><td rowspan="2">851</td><td>LV</td><td>75.4</td><td>72.0</td></tr><tr><td>LV+LR</td><td>76.3</td><td>72.9</td></tr></table>

Model scaling with  $\mathcal{L}_R$ . We compare the pre-training objective with and without  $\mathcal{L}_R$  in Table 5. Across different model scales and window sizes, the proposed region level  $\mathcal{L}_R$  can consistently improve the performance. The gains can be clearly seen by  $k$ -NN accuracy (around  $1 - 2\%$ ), where no additional tuning is needed as in linear probe. Figure 4 demonstrates that  $\mathcal{L}_R$  helps model convergence, and can be used as a drop-in to improve models trained with the view level task.

Table 5: Ablations of various configurations.  

<table><tr><td>Arch.</td><td>Objectives</td><td>Window S.</td><td>Linear</td><td>k-NN</td></tr><tr><td rowspan="4">Swin-T</td><td>LV</td><td>7</td><td>77.0</td><td>74.2</td></tr><tr><td>LV+LR</td><td>7</td><td>78.1</td><td>75.7</td></tr><tr><td>LV</td><td>14</td><td>77.9</td><td>75.5</td></tr><tr><td>LV+LR</td><td>14</td><td>78.7</td><td>77.0</td></tr><tr><td rowspan="4">Swin-S</td><td>LV</td><td>7</td><td>79.2</td><td>76.8</td></tr><tr><td>LV+LR</td><td>7</td><td>79.5</td><td>77.7</td></tr><tr><td>LV</td><td>14</td><td>79.4</td><td>77.3</td></tr><tr><td>LV+LR</td><td>14</td><td>80.8</td><td>79.1</td></tr><tr><td rowspan="4">Swin-B</td><td>LV</td><td>7</td><td>79.6</td><td>77.7</td></tr><tr><td>LV+LR</td><td>7</td><td>80.4</td><td>78.9</td></tr><tr><td>LV</td><td>14</td><td>80.5</td><td>78.3</td></tr><tr><td>LV+LR</td><td>14</td><td>81.3</td><td>79.3</td></tr></table>

![](images/18fd02dcbe4336b28d6e6dd37d223240fdc280a3392eea7e73281de0a4ada4b7.jpg)  
Figure 4: Learning curves of different pretraining tasks. For Base model,  $\mathcal{L}_{\mathrm{R}}$  is added from the 200th epoch.

Table 6: Comparison between contrastive and non-contrastive region-matching tasks.  

<table><tr><td rowspan="2">Pre-training Types</td><td colspan="3">ResNet50 in different settings</td><td colspan="2">ImageNet-1K</td><td colspan="2">COCO</td></tr><tr><td>Methods</td><td>#Epochs</td><td>#Views</td><td>Linear</td><td>k-NN</td><td>APbb</td><td>APmb</td></tr><tr><td colspan="4">Supervised</td><td>-</td><td>-</td><td>38.2</td><td>33.3</td></tr><tr><td rowspan="2">Contrastive</td><td>MoCo-v2</td><td>200</td><td>2</td><td>67.5</td><td>55.6</td><td>38.7</td><td>33.9</td></tr><tr><td>DenseCL</td><td>200</td><td>2</td><td>63.6 (-3.9)</td><td>48.6 (-7.0)</td><td>39.1 (+0.4)</td><td>34.2 (+0.3)</td></tr><tr><td rowspan="2">Non-Contrastive</td><td>LV</td><td>200</td><td>2</td><td>69.2</td><td>59.9</td><td>37.8</td><td>33.1</td></tr><tr><td>LV+LR</td><td>200</td><td>2</td><td>69.9 (+0.7)</td><td>61.7 (+1.8)</td><td>38.0 (+0.2)</td><td>33.2 (+0.1)</td></tr></table>

Contrastive vs Non-contrastive region-matching tasks. The proposed  $\mathcal{L}_R$  adds a non-contrastive region-matching task to the non-contrastive view-level task  $\mathcal{L}_V$ ; On the contrary, DenseCL adds a contrastive region-matching task to the contrastive view-level task MoCo-v2. In Table 6, we compare four methods in the same setting with ResNet-50. DenseCL improves dense visual prediction performance, but hurts classification performance.  $\mathcal{L}_R$  improves both tasks, especially the classification performance. One limitation is that the non-contrastive methods show lower performance in dense prediction tasks, this is consistent with the observations for BYOL in (Wang et al., 2020b). The simple  $\mathcal{L}_R$  shows the best ImageNet accuracy compared with all sophisticated region-level tasks in this 200-epoch setting in Appendix Table 8, and the best overall accuracy in Table 13. It indicates that  $\mathcal{L}_R$  well serves our goal in building efficient SoTA SSL systems.

Design choices of  $\mathcal{L}_R$ . We ablate a couple of choices in constructing  $\mathcal{L}_R$  in Eq. (2). (i) Softmax vs MSE. One alternative way to measure the distance between two projected vectors is MSE, as employed in the popular non-contrastive SSL algorithm BYOL (Grill et al., 2020). When adding region-matching tasks to BYOL and pre-training 50 epochs, Softmax and MSE yield  $k$ -NN accuracy of  $37.2\%$  and  $34.9\%$ , while the baseline BYOL yields  $33.1\%$ . We also replace the region-matching metric in EsViT as MSE, yielding  $k$ -NN accuracy  $72.6\%$ , which lower than the view-level task only  $(74.2\%)$ . These results show that Softmax is essential in  $\mathcal{L}_R$ . (ii) Optimal Transport (OT) vs Simple Argmax. To avoid heavy computational overhead, a simple feature-level argmax solution is considered in Eq. (2) to pair two local regions. To study the impact of high region-matching quality, we consider OT. Empirically, we observe OT yields slightly higher  $k$ -NN accuracy at the early stage, but the gain is diminished in the end. Considering the extra computational cost of solving OT with an inner loop in sinkhorn algorithm (Cuturei, 2013), we opt for simple argmax in our experiments.

# 4.4 QUALITATIVE STUDIES

Visualization of correspondences. Given two views of the same image, we use the pre-trained backbone to extract the top-layer features  $\mathbf{z}_1$  and  $\mathbf{z}_2$ . For each feature vector in  $\mathbf{z}_1$ , we find the feature vector in  $\mathbf{z}_2$  that best matches it in terms of highest cosine similarity, as defined in Equation (2). In Figure 5, we show the top-10 correspondences between two views for three methods. In Figure 5 (b), EsViT with  $\mathcal{L}_V$  tends to identify pairs in the background as the most matched ones (and in a wrong way in this example). This could be a valid solution to  $\mathcal{L}_V$ , as the invariance in the level of aggregated global features does not necessarily induce invariances in the local region level. This is significantly alleviated with  $\mathcal{L}_R$  (shown in Figure 5 (c)), a task that implicitly requires local matching.

![](images/c49f4f90e9a06efbbe9b2a299123eb8a34470bebb25eedc2e27cd278233b65b4.jpg)  
(a) DINO: DeiT-S

![](images/7baf04e905894ae307e902bb30706471771d23aa4cb916fa53bd1bf713beddcf.jpg)  
(b) EsViT:  $\mathcal{L}_V$

![](images/dcf0a961cbd6b926d94c6b757f9d5d5e2fd855119b37ec896eb274269b1ccac9.jpg)  
(c) EsViT:  $\mathcal{L}_V + \mathcal{L}_R$

![](images/dc55d3ce6ca78ce1d99caa9a02efa37fc2f79d5e486bc74bb95191ea6bd43d0b.jpg)  
Figure 5: The learned correspondences. Yellow lines are the top-10 correspondences between two views, where the numbers indicate the rankings of similarity scores, yellow dots with the same number are paired.

![](images/56f8b04419b618145b2bb0ee5b3f8d6bed490d98044e9fcd663244abbaa5ea04.jpg)  
(a) DINO: DeiT-S

![](images/6eb621c582d99a264f526ba6d1d965011933f3ce875c8c275834be8a16866ed3.jpg)

![](images/dd55b21c88a6f540de1883b08c77ae482a20789898166e576809796d8378976e.jpg)  
Figure 6: Visualization of the the learned attention map for different heads in the last layer. The query is the blue dot in the center of the images. We visualize masks (as red) obtained by thresholding the self-attention maps to keep  $60\%$  of the probability mass. Note that all 6 heads are visualized for DINO with DeiT-S, and 6 out of 24 heads in EsViT are chosen to visualize (ranked by entropy values). Please see enlarged pictures with all heads in Appendix.

![](images/428fb70ad4be1a41266641f651639ebbaed18895c0ad1aabf14583105f72f09c.jpg)

![](images/35a28ece7bda0a13b19545c2a79da9f778b18a6dd63ae32c7c06cbec35c5f96b.jpg)

![](images/d8731e5f8b5426dd593948638856ec9f7b6225ac2c24d4dc35dc6234e2097099.jpg)

![](images/aa0e182ec6656d11a6c4ec36048671c9f4e7500f6a5989e9b5c31838903758c3.jpg)  
(b) EsViT:  $\mathcal{L}_V$

![](images/c6e6d0adf379158f5d086d28d103e9afb9c80f1c6a224ebcbbeb046dddeba4ef.jpg)

![](images/4409d95824d4bcf85e5c3cb38d9aea585bedb094c17d7b53f8bf2c46e1917909.jpg)

![](images/7d193d947081035603207962da1f9c7f251c6b24fe9c69e8b21da3d41e000ef9.jpg)

![](images/2f3f73743866af3195e1c1e67fe9281caf3b9d12db2d42ed4002f52930a7946f.jpg)

![](images/eb062a4318976ebfc05e97dd4ce71168ba73b776a240da65fccf18813e34919c.jpg)

![](images/912ba5408368a7286e3662c1387604cbd38c1a57d4be6946f3f2186b1742d5d2.jpg)  
(c) EsViT:  $\mathcal{L}_V + \mathcal{L}_R$

![](images/8877f3352a74160ad47fa0b3bf660cbcc052c76cac0b3e58f437944a45ee4e20.jpg)

![](images/7fb7025e595c3e212838611c228cbc8b54bbfdaed0d35258efb8c38d186ee93f.jpg)

![](images/1c99ba6ce2bcfcae419680d83c16650ff2c2b29d3152e7e3e5e8609641f829c3.jpg)

![](images/2bea799c8f7a98731f851b57bc4c532ee6ed6d6b036a44e1bd992a91bfc0948c.jpg)

Surprisingly, DINO is able to learn good correspondences even without the region-level matching task. To the best of our knowledge, this is a previously unreported intriguing property of self-supervised Transformers with monolithic architectures: good semantic correspondences are automatically learned. We hypothesize that features at lower layers (image patch itself in the extreme case) can directly pass to higher layers, and the former regularizes the latter to remain discriminative. Nevertheless, the proposed  $\mathcal{L}_R$  can dramatically reduce the issue, and is good remedy to rescue the loss of semantic correspondence for the multi-stage architecture. In Appendix, we quantitatively measures the correspondence learning ability of these SSL methods on ImageNet validation dataset, the observations are consistent:  $\mathcal{L}_R$  improves the matching accuracy from  $66\%$  to  $91\%$ .

Visualization of attention maps. We look at the self-attention in the different heads of the last layer in Figure 6. A local region on the edge of the main object is employed as query, and the attended regions are highlighted in red for those the query's top  $60\%$  mass are assigned. In Appendix, we visualize more examples with different query positions. DINO tends to automatically learn class-specific attention maps leading to foreground object segmentation, regardless of its query located in foreground or background. This is probably because main objects remain as the major invariance factor in different augmented views. This property is lost when a multi-stage architecture is employed, as shown in EsViT with  $\mathcal{L}_V$ . These patterns are consistent for different heads. After introducing  $\mathcal{L}_R$  for EsViT, we note that the attention maps become more diverse in different heads, i.e., entropy values of attentions get more skewed, and attended regions are more different. This is perhaps because  $\mathcal{L}_R$  requires each region to consider many matching tasks to regions in different augmented views, each head automatically learns to distribute the tasks and complete a few of them.

# 5 CONCLUSIONS

In this paper, we first discover the automatic correspondence learning property of self-supervised monolithic Transformers. Inspired by this, we present efficient self-supervised vision Transformers (EsViT) to with two major insights: a multi-stage Transformer architecture with sparse self-attention, and a non-contrastive region-matching pre-training task. The synergy of both helps EsViT reach the SoTA performance of SSL vision systems with significantly less compute and smaller model size. Our study also reveals that exploration of effective solutions to learn from larger and less curated pre-training data in the wild is a key but less studied factor in paving the way toward the scaling success of SSL vision systems.

# ETHICS STATEMENT

Though self-supervised learning (SSL) has great potentials to learn powerful representation without human annotation, the existing techniques to build SoTA SSL vision systems tend to be Red AI (Schwartz et al., 2020): it could be environmentally unfriendly and the computational cost is extensively high. The required training resource is typically not accessible for a lab environment (thus raising barriers to participation in AI research). For example, the prior art MoCo-v3 has greatly pushes the performance limit of SSL system (Chen et al., 2021). The authors kindly reported that "it (MoCo-v3, ViT-H) takes 9.8 hours per 100 epochs using 512 TPUs. This is a gigantic scale of training: for the 300-epoch ViT-H, this amounts to  $\sim 625$  TPU days, or  $\sim 1.7$  TPU years of training." The SoTA model MoCo-v3 with ViT-BN-L/7 should have a higher cost than this. Even for a smaller model ViT-B, "it takes 24 hours in 128 GPUs (vs. 2.1 hours in 256 TPUs)". Hence, improving the efficiency of building SoTA SSL systems is of high value for the community and society to achieve Green AI (Schwartz et al., 2020).

To this end, we propose EsViT to provide more affordable and efficient solutions for the community to experiment and explore the directions of SoTA SSL in computer vision. Our EsViT model shows the best ImageNet linear probe performance compared with all existing SSL vision systems, and is  $3.5 \times$  parameter-efficient and has  $10 \times$  higher throughput than previous SoTA. This efficiency gain can significantly decrease its carbon footprint and increase its inclusivity, encouraging more researchers to participate the study of the SSL topic.

# REPRODUCIBILITY STATEMENT

Our paper provides comprehensive empirical studies on the EsViT algorithm. We provide PyTorch-style pseudo-code in Appendix. We also include an example code with instruction as supplementary material to ensure the reproducibility. For empirical results on both various network architecture and large-scale datasets, we provide detailed hyper-parameter specifications. We will release the pre-trained checkpoints and codebase for the research community for reproducible research.

# REFERENCES

Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In NeurIPS, 2019.  
Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.  
Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020.  
Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In ECCV, 2018.  
Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. arXiv preprint arXiv:2006.09882, 2020.  
Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. arXiv preprint arXiv:2104.14294, 2021.  
Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen Gao. Pre-trained image processing transformer. arXiv preprint arXiv:2012.00364, 2020a.  
Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, Prafulla Dhariwal, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020b.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. ICML, 2020c.  
Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. Big self-supervised models are strong semi-supervised learners. arXiv preprint arXiv:2006.10029, 2020d.  
Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020e.  
Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised visual transformers. arXiv preprint arXiv:2104.02057, 2021.  
Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Learning universal image-text representations. arXiv preprint arXiv:1909.11740, 2019.  
Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 2013.  
Zhigang Dai, Bolun Cai, Yugeng Lin, and Junying Chen. UP-DETR: Unsupervised pre-training for object detection with transformers. arXiv preprint arXiv:2011.09094, 2020.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*, 2019.  
Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015.  
Jeff Donahue and Karen Simonyan. Large scale adversarial representation learning. In NeurIPS, 2019.  
Alexey Dosovitskiy, Philipp Fischer, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with exemplar convolutional neural networks. T-PAMI, 2015.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.  
Jonathan Frankle, David J Schwab, and Ari S Morcos. Training Batchnorm and only Batchnorm: On the expressive power of random features in CNNs. arXiv preprint arXiv:2003.00152, 2020.  
Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. arXiv preprint arXiv:1803.07728, 2018.  
Priya Goyal, Piotr Dálár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
Priya Goyal, Mathilde Caron, Benjamin Lefaudeaux, Min Xu, Pengchao Wang, Vivek Pai, Mannat Singh, Vitaliy Liptchinsky, Ishan Misra, Armand Joulin, et al. Self-supervised pretraining of visual features in the wild. arXiv preprint arXiv:2103.01988, 2021.  
Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent: A new approach to self-supervised learning. arXiv preprint arXiv:2006.07733, 2020.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.  
R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. arXiv preprint arXiv:1808.06670, 2018.  
Xu Ji, João F Henriques, and Andrea Vedaldi. Invariant information clustering for unsupervised image classification and segmentation. In ICCV, 2019.  
Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4. International Journal of Computer Vision, 2020.  
Gustav Larsson, Michael Maire, and Gregory Shakhnarovich. Learning representations for automatic colorization. In ECCV, 2016.  
Chunyuan Li, Xiujun Li, Lei Zhang, Baolin Peng, Mingyuan Zhou, and Jianfeng Gao. Self-supervised pre-training with hard examples improves visual representations. arXiv preprint arXiv:2012.13493, 2020a.  
Gen Li, Nan Duan, Yuejian Fang, Daxin Jiang, and Ming Zhou. Unicoder-VL: A universal encoder for vision and language by cross-modal pre-training. arXiv preprint arXiv:1908.06066, 2019a.  
Junnan Li, Pan Zhou, Caiming Xiong, Richard Socher, and Steven CH Hoi. Prototypical contrastive learning of unsupervised representations. arXiv preprint arXiv:2005.04966, 2020b.  
Lianian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019b.  
Wen Li, Limin Wang, Wei Li, Eirikur Agustsson, and Luc Van Gool. Webvision database: Visual learning and understanding from web data. arXiv preprint arXiv:1708.02862, 2017.  
Xiang Li, Wenhai Wang, Xiaolin Hu, and Jian Yang. Selective kernel networks. In CVPR, 2019c.  
Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, and Jianfeng Gao. Oscar: Object-semantics aligned pretraining for vision-language tasks. In ECCV, 2020c.  
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. arXiv preprint arXiv:2103.14030, 2021.  
Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in Adam. arXiv preprint arXiv:1706.02677, 2018.  
Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. VilBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. NeurIPS, 2019.  
Ishan Misra and Laurens van der Maaten. Self-supervised learning of pretext-invariant representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.  
Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV, 2016.  
Aaron van den Oord, Yazhe Li, and Oriol Vinyls. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
Niki Parmar, Ashish Vavwani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International Conference on Machine Learning, 2018.  
Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In CVPR, 2016.

Yunchen Pu, Zhe Gan, Ricardo Henao, Xin Yuan, Chunyuan Li, Andrew Stevens, and Lawrence Carin. Variational autoencoder for deep learning of images, labels and captions. NIPS, 2016.  
Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. OpenAI Blog, 2018.  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021.  
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. arXiv preprint arXiv:2102.12092, 2021.  
Roy Schwartz, Jesse Dodge, Noah A Smith, and Oren Etzioni. Green AI. Communications of the ACM, 2020.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. VL-BERT: Pre-training of generic visual-linguistic representations. arXiv preprint arXiv:1908.08530, 2019.  
Hao Tan and Mohit Bansal. LXMERT: Learning cross-modality encoder representations from transformers. EMNLP, 2019.  
Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning. arXiv preprint arXiv:2005.10243, 2020.  
Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, et al. MLP-mixer: An all-MLP architecture for vision. arXiv preprint arXiv:2105.01601, 2021.  
Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. arXiv preprint arXiv:2012.12877, 2020.  
Trieu H Trinh, Minh-Thang Luong, and Quoc V Le. Selfie: Self-supervised pretraining for image embedding. arXiv preprint arXiv:1906.02940, 2019.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017.  
Ashish Vaswani, Prajit Ramachandran, Aravind Srinivas, Niki Parmar, Blake Hechtman, and Jonathon Shlens. Scaling local self-attention for parameter efficient visual backbones. CVPR, 2021.  
Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Max-deeplab: End-to-end panoptic segmentation with mask transformers. arXiv preprint arXiv:2012.00759, 2020a.  
Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. arXiv preprint arXiv:2102.12122, 2021.  
Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, and Lei Li. Dense contrastive learning for self-supervised visual pre-training. arXiv preprint arXiv:2011.09157, 2020b.  
Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen, Baoshan Cheng, Hao Shen, and Huaxia Xia. End-to-end video instance segmentation with transformers. arXiv preprint arXiv:2011.14503, 2020c.  
Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. Cvt: Introducing convolutions to vision transformers. arXiv preprint arXiv:2103.15808, 2021.  
Enze Xie, Jian Ding, Wenhai Wang, Xiaohang Zhan, Hang Xu, Zhenguo Li, and Ping Luo. Detco: Unsupervised contrastive learning for object detection. arXiv preprint arXiv:2102.04803, 2021a.

Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. In ICML, 2016.  
Zhenda Xie, Yutong Lin, Zhuliang Yao, Zheng Zhang, Qi Dai, Yue Cao, and Han Hu. Self-supervised learning with swin transformers. arXiv preprint arXiv:2105.04553, 2021b.  
Zhenda Xie, Yutong Lin, Zheng Zhang, Yue Cao, Stephen Lin, and Han Hu. Propagate yourself: Exploring pixel-level consistency for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021c.  
Yuwen Xiong, Mengye Ren, and Raquel Urtasun. Loco: Local contrastive representation learning. arXiv preprint arXiv:2008.01342, 2020.  
Ceyuan Yang, Zhirong Wu, Bolei Zhou, and Stephen Lin. Instance localization for self-supervised detection pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.  
Fuzhi Yang, Huan Yang, Jianlong Fu, Hongtao Lu, and Baining Guo. Learning texture transformer network for image super-resolution. In CVPR, 2020.  
Jianwei Yang, Devi Parikh, and Dhruv Batra. Joint unsupervised learning of deep representations and image clusters. In CVPR, 2016.  
Tian Yonglong, Olivier J. Henaff, and Aaron van den Oord. Divide and contrast: Self-supervised learning from uncurated data. arXiv preprint arXiv:2105.08054, 2021.  
Xiaohang Zhan, Jiahao Xie, Ziwei Liu, Yew-Soon Ong, and Chen Change Loy. Online deep clustering for unsupervised representation learning. In CVPR, pp. 6688-6697, 2020.  
Pengchuan Zhang, Xiyang Dai, Jianwei Yang, Bin Xiao, Lu Yuan, Lei Zhang, and Jianfeng Gao. Multi-scale vision longformer: A new vision transformer for high-resolution image encoding. arXiv preprint arXiv:2103.15358, 2021.  
Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In ECCV, 2016.  
Richard Zhang, Phillip Isola, and Alexei A Efros. Split-brain autoencoders: Unsupervised learning by cross-channel prediction. In CVPR, 2017.  
Minghang Zheng, Peng Gao, Xiaogang Wang, Hongsheng Li, and Hao Dong. End-to-end object detection with adaptive clustering transformer. arXiv preprint arXiv:2011.09315, 2020.  
Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J Corso, and Jianfeng Gao. Unified vision-language pre-training for image captioning and VQA. AAAI, 2020.  
Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.  
Chengxu Zhuang, Alex Lin Zhai, and Daniel Yamins. Local aggregation for unsupervised learning of visual embeddings. In CVPR, 2019.
