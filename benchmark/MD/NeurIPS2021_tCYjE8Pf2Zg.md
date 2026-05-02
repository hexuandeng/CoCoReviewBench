# Distilling Image Classifiers in Object Detectors

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Knowledge distillation constitutes a simple yet effective way to improve the performance of a compact student network by exploiting the knowledge of a more powerful teacher. Nevertheless, the knowledge distillation literature remains limited to the scenario where the student and the teacher tackle the same task. Here, we investigate the problem of transferring knowledge not only across architectures but also across tasks. To this end, we study the case of object detection and, instead of following the standard detector-to-detector distillation approach, introduce a classifier-to-detector knowledge transfer framework. In particular, we propose strategies to exploit the classification teacher to improve both the detector's recognition accuracy and localization performance. Our experiments on several detectors with different backbones demonstrate the effectiveness of our approach, allowing us to outperform the state-of-the-art detector-to-detector distillation methods.

# 1 Introduction

Object detection plays a critical role in many real-world applications, such as autonomous driving and video surveillance. While deep learning has achieved tremendous success in this task [26, 27, 32, 33, 39], the speed-accuracy trade-off of the resulting models remains a challenge. This is particularly important for real-time prediction on embedded platforms, whose limited memory and computation power impose strict constraints on the deep network architecture.

To address this, much progress has recently been made to obtain compact deep networks. Existing methods include pruning [1, 2, 13, 37, 23] and quantization [7, 43, 31], both of which aim to reduce the size of an initial deep architecture, as well as knowledge distillation, whose goal is to exploit a deep teacher network to improve the training of a given compact student one. In this paper, we introduce a knowledge distillation approach for object detection.

While early knowledge distillation techniques [18, 34, 35] focused on the task of image classification, several attempts have nonetheless been made for object detection. To this end, existing techniques [5, 12, 38] typically leverage the fact that object detection frameworks consist of three main stages depicted by Figure 1(a): A backbone to extract features; a neck to fuse the extracted features; and heads to predict classes and bounding boxes. Knowledge distillation is then achieved using a teacher with the same architecture as the student but a deeper and wider backbone, such as a Faster RCNN [33] with ResNet152 [14] teacher for a Faster RCNN with ResNet50 student, thus facilitating knowledge transfer at all three stages of the frameworks. To the best of our knowledge, [42] constitutes the only exception to this strategy, demonstrating distillation across different detection frameworks, such as from a RetinaNet [26] teacher to a RepPoints [39] student. This method, however, requires the teacher and the student to rely on a similar detection strategy, i.e., both must be either one-stage detectors or two-stage ones, and, more importantly, still follows a detector-to-detector approach to distillation. In other words, the study of knowledge distillation remains limited to transfer across architectures tackling the same task.

In this paper, we investigate the problem of transferring knowledge not only across architectures but also across tasks. In particular, we observed that the classification head of state-of-the-art object detectors still typically yields inferior performance compared to what can be expected from an image classifier. Thus, as depicted by Figure 1(b), we focus on the scenario where the teacher is an image classifier while the student is an object detector. We then develop distillation strategies to improve both the recognition accuracy and the localization ability of the student.

Our contributions can thus be summarized as follows:

- We introduce the idea of classifier-to-detector knowledge distillation to improve the performance of a student detector using a classification teacher.  
- We propose a distillation method to improve the student's classification accuracy, applicable when the student uses either a categorical cross-entropy loss or a binary cross-entropy one.  
- We develop a distillation strategy to improve the localization performance of the student by exploiting the feature maps from the classification teacher.

We demonstrate the effectiveness of our approach on the COCO2017 benchmark [24] using diverse detectors, including the relatively large two-stage Faster RCNN and single-stage RetinaNet used in previous knowledge distillation works, as well as more compact detectors, such as SSD300, SSD512 [27] and Faster RCNNs[33] with lightweight backbones. Our classifier-to-detector distillation approach outperforms the detector-to-detector distillation ones in the presence of compact students, and helps to further boost the performance of detector-to-detector distillation techniques for larger ones, such as Faster RCNN and RetinaNet with a ResNet50 backbone. We will make our code publicly available.

# 2 Related work

Object detection is one of the fundamental tasks in computer vision, aiming to localize the objects observed in an image and classify them. Recently, much progress has been made via the development of both one-stage [32, 27, 9, 22, 36] and two-stage [33, 4, 15, 25] deep object detection frameworks, significantly improving the mean average precision (mAP) on standard benchmarks [24, 10, 11]. However, the performance of these models typically increases with their size, and so does their inference runtime. This conflicts with their deployment on embedded platforms, such as mobile phones, drones, and autonomous vehicles, which involve computation and memory constraints. While some efforts have been made to design smaller detectors, such as SSD [27], YOLO [32] and detectors with lightweight backbones [19], the performance of these methods does not match that of deeper ones.

Knowledge distillation offers the promise to boost the performance of such compact networks by exploiting deeper teacher architectures. Early work in this space focused on the task of image classification. In particular, Hinton et al. [18] proposed to distill the teacher's class probability distribution into the student, and Romero et al. [34] encouraged the student's intermediate feature maps to mimic the teacher's ones. These initial works were followed by a rapid growth in the number of knowledge distillation strategies, including methods based on attention maps [41], on transferring feature flows defined by the inner product of features [40], and on contrastive learning to structure the knowledge distilled from teacher to the student [35]. Heo et al. [17] provides a comprehensive study of feature distillation.

Compared to image classification, object detection poses the challenge of involving both recognition and localization. As such, several works have introduced knowledge distillation methods specifically tailored to this task. This trend was initiated by Chen et al. [5], which proposed to distill knowledge from a teacher detector to a student detector in both the backbone and head stages. Then, Wang et al. [38] proposed to restrict the teacher-student feature imitation to regions around positive anchor boxes; Dai et al. [8] produced general instances based on both the teacher's and student's outputs, and distilled feature-based, relation-based and response-based knowledge in these general instances; Guo et al. [12] proposed to decouple the intermediate features and classification predictions of the positive and negative regions during knowledge distillation. All the aforementioned knowledge distillation methods require the student and the teacher to follow the same kind of detection framework, and thus typically transfer knowledge between models that only differ in terms of backbone, such as from a RetinaNet-ResNet152 to a RetinaNet-ResNet50. In [42], such a constraint was relaxed via a method

![](images/e5fd156214adeae1001c4d8c924d3d0341c8a7ebadf22258c8b429a10e1999fd.jpg)  
(a) Detector-to-Detector Distillation  
Figure 1: Overview of our classifier-to-detector distillation framework. (a) Existing methods perform distillation across corresponding stages in the teacher and student, which restricts their applicability to detector-to-detector distillation. (b) By contrast, we introduce strategies to transfer the knowledge from an image classification teacher to an object detection student, improving both its recognition and localization accuracy.

![](images/1db40aa3f8b374519f2bb395b35f94d182ed0aa9945887a39be23e3167b0a743.jpg)  
(b) Our Classifier-to-Detector Distillation

able to transfer knowledge across the feature maps of different frameworks. This allowed the authors to leverage the best one-stage, resp. two-stage, teacher model to perform distillation to any one-stage, resp. two-stage, student. This method, however, still assumes that the teacher is a detector.

In short, existing knowledge distillation methods for object detection all follow a detector-to-detector transfer strategy. In fact, to the best of our knowledge, distillation has only been studied across two architectures that tackle the same task, may it be image classification, object detection, or even semantic segmentation [16, 28]. In this paper, by contrast, we investigate the use of knowledge distillation across tasks and develop strategies to distill the knowledge of an image classification teacher to an object detection student.

# 3 Our Approach

Our goal is to investigate the transfer of knowledge from an image classifier to an object detector. As illustrated in Figure 1, this contrasts with existing knowledge distillation techniques for object detection, which typically assume that the teacher and the student both follow a similar three-stage detection pipeline. For our classifier-to-detector knowledge distillation to be effective, we nonetheless need the student and teacher to process the same data and use the same loss for classification. To this end, given a detection dataset  $\mathcal{D}_{det}$  depicting  $C$  foreground object categories, we construct a classification dataset  $\mathcal{D}_{cls}$  by extracting all objects from  $\mathcal{D}_{det}$  according to their ground-truth bounding boxes and labels. We then train our classification teacher  $\mathcal{F}_t$ , with parameters  $\theta^t$ , on  $\mathcal{D}_{cls}$  in a standard classification manner. In the remainder of this section, we introduce our strategies to exploit the resulting teacher to improve both the classification and localization accuracy of the student detector  $\mathcal{F}_s$ , with parameters  $\theta^s$ .

# 3.1  $\mathbf{KD}_{cls}$ : Knowledge Distillation for Classification

Our first approach to classifier-to-detector distillation focuses on the classification accuracy of the student network. To this end, we make use of the class-wise probability distributions obtained by the teacher and the student, softened by making use of a temperature parameter  $T$ . Below, we first derive our general formulation for distillation for classification, and then discuss in more detail how we obtain the teacher and student class distributions for the two types of classification losses commonly used by object detection frameworks.

Formally, given  $K$  positive anchor boxes or object proposals, which are assigned with one of the ground-truth labels and bounding boxes during training, let  $p_k^{s,T}$  denote the vector of softened class probabilities for box  $k$  from the student network, obtained at temperature  $T$ , and let  $p_k^{t,T}$  denote the corresponding softened probability vector from the teacher network. We express knowledge distillation for classification as a loss function measuring the Kullback-Leibler (KL) divergence between the teacher and student softened distributions. This can be written as

$$
\mathcal {L} _ {k d - c l s} = \frac {1}{K} \sum_ {k = 1} ^ {K} K L \left(p _ {k} ^ {t, T} \| p _ {k} ^ {s, T}\right). \tag {1}
$$

The specific way we define the probability vectors  $p_k^{s,T}$  and  $p_k^{t,T}$  then depends on the loss function that the student detector uses for classification. Indeed, existing detectors follow two main trends: some, such as Faster RCNN and SSD, exploit the categorical cross-entropy loss with a softmax, accounting for the  $C$  foreground classes and 1 background one; others, such as RetinaNet, employ a form of binary cross-entropy loss with a sigmoid<sup>1</sup>, focusing only on the  $C$  foreground classes. Let us now discuss these two cases in more detail.

Categorical cross-entropy. In this case, for each positive object bounding box  $k$ , the student detector outputs logits  $z_{k}^{s} \in (C + 1)$ . We then compute the corresponding softened probability for class  $c$  with temperature  $T$  as

$$
p _ {k} ^ {s, T} \left(c \mid \theta^ {s}\right) = \frac {e ^ {z _ {k , c} ^ {s} / T}}{\sum_ {j = 1} ^ {C + 1} e ^ {z _ {k , j} ^ {s} / T}}, \tag {2}
$$

where  $z_{k,c}^{s}$  denote the logit corresponding to class  $c$ . By contrast, as our teacher is a  $C$ -way classifier, it produces logits  $z_{k}^{t} \in C$ . We thus compute its softened probability for class  $c$  as

$$
\tilde {p} _ {k} ^ {t, T} \left(c \mid \theta^ {t}\right) = \frac {e ^ {z _ {k , c} ^ {t} / T}}{\sum_ {j = 1} ^ {C} e ^ {z _ {k , j} ^ {t} / T}}, \tag {3}
$$

and, assuming that all true objects should be classified as background with 0 probability, augment the resulting distribution to account for the background class as  $p^{t,T} = [\tilde{p}^{t,T},0]$ .

The KL-divergence between the teacher and student softened distributions for object  $k$  can then be written as

$$
K L \left(p _ {k} ^ {t, T} \| p _ {k} ^ {s, T}\right) = T ^ {2} \sum_ {c = 1} ^ {C + 1} p _ {k, c} ^ {t, T} \log p _ {k, c} ^ {t, T} - p _ {k, c} ^ {t, T} \log p _ {k, c} ^ {s, T}. \tag {4}
$$

Binary cross-entropy. The detectors that rely on the binary cross-entropy output a score between 0 and 1 for each of the  $C$  foreground classes, but, together, these scores do not form a valid distribution over the  $C$  classes as they do not sum to 1. To nonetheless use them in a KL-divergence measure between the teacher and student, we rely on the following strategy. Given the student and teacher  $C$ -dimensional logit vectors for an object  $k$ , we compute softened probabilities as

$$
\tilde {p} _ {k} ^ {s, T} (c \mid \theta^ {s}) = (1 + e ^ {- z _ {k, c} ^ {s} / T}) ^ {- 1},
$$

$$
\tilde {p} _ {k} ^ {t, T} (c \mid \theta^ {t}) = \left(1 + e ^ {- z _ {k, c} ^ {t} / T}\right) ^ {- 1}. \tag {5}
$$

We then build a 2-class probability distribution for each category according to the ground-truth label  $l$  of object  $k$ . Specifically, for each category  $c$ , we write

$$
p _ {k, c} ^ {s, T} = \left\{ \begin{array}{l} \left[ 1 - \tilde {p} _ {k, c} ^ {s, T}, \tilde {p} _ {k, c} ^ {s, T} \right], \text {i f} c = l \\ \left[ \tilde {p} _ {k, c} ^ {s, T}, 1 - \tilde {p} _ {k, c} ^ {s, T} \right], \text {o t h e r w i s e} \end{array} \right. \tag {6}
$$

for the student, and similarly for the teacher. This lets us express the KL-divergence for object  $k$  as

$$
K L \left(p _ {k} ^ {t, T} \| p _ {k} ^ {s, T}\right) = \frac {T ^ {2}}{C} \sum_ {c = 1} ^ {C} \sum_ {i = 0} ^ {1} p _ {k, c} ^ {t, T} (i) \log p _ {k, c} ^ {t, T} (i) - p _ {k, c} ^ {t, T} (i) \log p _ {k, c} ^ {s, T} (i), \tag {7}
$$

where  $p_{k,c}^{t,T}(i)$  indicates the  $i$ -th element of the 2-class distribution  $p_{k,c}^{t,T}$ .

# 3.2  $\mathbf{KD}_{loc}$ : Knowledge Distillation for Localization

While, as will be shown by our experiments, knowledge distillation for classification already helps the student detector, it does not aim to improve its localization performance. Nevertheless, localization, or bounding box regression, is critical for the success of a detector and is typically addressed by existing detector-to-detector distillation frameworks [5, 8]. To also tackle this in our classifier-to-detector approach, we develop a feature-level distillation strategy, exploiting the intuition that the intermediate features extracted by the classification teacher from a bounding box produced by the student should match those of the ground-truth bounding box.

Formally, given an input image  $I$  of size  $w \times h$ , let us denote by  $B_{k} = (x_{1}, y_{1}, x_{2}, y_{2})$  the bottom-left and top-right corners of the  $k$ -th bounding box produced by the student network. Typically, this is achieved by regressing the offset of an anchor box or object proposal. We then make use of a Spatial Transformer [20] to extract the image region corresponding to  $B_{k}$ . Specifically, we compute the transformer matrix

$$
A _ {k} = \left[ \begin{array}{c c c} \left(x _ {2} - x _ {1}\right) / w & 0 & - 1 + \left(x _ {1} + x _ {2}\right) / w \\ 0 & \left(y _ {2} - y _ {1}\right) / h & - 1 + \left(y _ {1} + y _ {2}\right) / h \end{array} \right], \tag {8}
$$

which allows us to extract the predicted object region  $O_{k}^{p}$  with a grid sampling size  $s$  as

$$
O _ {k} ^ {p} = f _ {S T} \left(A _ {k}, I, s\right), \tag {9}
$$

where  $f_{ST}$  denotes the spatial transformer function. As illustrated in the right portion of Figure 1(b), we then perform distillation by comparing the teacher's intermediate features within the predicted object region  $O_k^p$  to those within its assigned ground-truth one  $O_k^{gt}$ .

Specifically, for a given layer  $\ell$ , we seek to compare the features  $\mathcal{F}_t^\ell (O_k^p)$  and  $\mathcal{F}_t^\ell (O_k^{gt})$  of the positive box  $k$ . To relax the pixel-wise difference between the features, we make use of the adaptive pooling strategy of [29], which produces a feature map  $AP(\mathcal{F}_t^\ell (O))$  of a fixed size  $M\times W\times H$  from the features extracted within region  $O$ . We therefore write our localization distillation loss as

$$
\mathcal {L} _ {k d - l o c} = \frac {1}{K L M H W} \sum_ {k = 1} ^ {K} \sum_ {\ell = 1} ^ {L} \mathbb {1} _ {\ell} \| A P \left(\mathcal {F} _ {t} ^ {\ell} \left(O _ {k} ^ {p}\right)\right) - A P \left(\mathcal {F} _ {t} ^ {\ell} \left(O _ {k} ^ {g t}\right)\right) \| _ {1}, \tag {10}
$$

where  $K$  is the number of positive anchor boxes or proposals,  $L$  is the number of layers at which we perform distillation,  $\mathbb{1}_l$  is the indicator function to denote whether the layer  $\ell$  is used or not to distill knowledge, and  $\| \cdot \| _1$  denotes the  $L_{1}$  norm. As both the spatial transformer and the adaptive pooling operation are differentiable, this loss can be backpropagated through the student detector.

Note that, as a special case, our localization distillation strategy can be employed not only on intermediate feature maps but on the object region itself, encouraging the student to produce bounding boxes whose underlying image pixels match those of the ground-truth box. This translates to a loss function that does not exploit the teacher and can be expressed as

$$
\mathcal {L} _ {l o c} \left(O ^ {p}, O ^ {g t}\right) = \frac {1}{K M H W} \sum_ {k = 1} ^ {K} \| A P \left(O _ {k} ^ {p}\right) - A P \left(O _ {k} ^ {g t}\right) \| _ {1}. \tag {11}
$$

Depending on the output size of the adaptive pooling operation, this loss function encodes a more-or-less relaxed localization error. As will be shown by our experiments, it can serve as an attractive alternative to the standard bounding box regression loss of existing object detectors, whether using distillation or not.

# 3.3 Overall Training Loss

To train the student detector given the image classification teacher, we then seek to minimize the overall loss

$$
\mathcal {L} = \mathcal {L} _ {\text {d e t}} + \lambda_ {k c} \mathcal {L} _ {k d - c l s} + \lambda_ {k l} \mathcal {L} _ {k d - l o c} + \lambda_ {l} \mathcal {L} _ {l o c}, \tag {12}
$$

where  $\mathcal{L}_{det}$  encompasses the standard classification and localization losses used to train the student detector of interest.  $\lambda_{kc},\lambda_{kl}$  and  $\lambda_{l}$  are hyper-parameters setting the influence of each loss.

Table 1: Analysis of our classifier-to-detector distillation method with compact students on the COCO2017 validation set. R50 is ResNet50, MV2 is MobileNetV2, QR50 is quartered ResNet50.  

<table><tr><td>Method</td><td>mAP</td><td>\( AP_{50} \)</td><td>\( AP_{75} \)</td><td>\( AP_s \)</td><td>\( AP_m \)</td><td>\( AP_l \)</td><td>mAR</td><td>\( AR_s \)</td><td>\( AR_m \)</td><td>\( AR_l \)</td></tr><tr><td>SSD300-VGG16</td><td>25.6</td><td>43.8</td><td>26.3</td><td>6.8</td><td>27.8</td><td>42.2</td><td>37.6</td><td>12.5</td><td>41.7</td><td>58.6</td></tr><tr><td>+ \( KD_{cls} \)</td><td>26.3 (↑ 0.7)</td><td>45.2</td><td>27.2</td><td>7.3</td><td>28.5</td><td>43.6</td><td>38.4</td><td>12.8</td><td>42.6</td><td>59.1</td></tr><tr><td>+ loc</td><td>27.1 (↑ 1.5)</td><td>43.2</td><td>28.4</td><td>7.5</td><td>29.4</td><td>43.3</td><td>40.0</td><td>13.4</td><td>44.4</td><td>60.6</td></tr><tr><td>+ loc + \( KD_{loc} \)</td><td>27.2 (↑ 1.6)</td><td>43.3</td><td>28.5</td><td>7.5</td><td>29.5</td><td>43.5</td><td>40.2</td><td>13.2</td><td>44.7</td><td>61.5</td></tr><tr><td>+ \( KD_{cls} + loc + KD_{loc} \)</td><td>27.9 (↑ 2.3)</td><td>45.1</td><td>29.2</td><td>8.1</td><td>30.1</td><td>45.4</td><td>40.4</td><td>13.9</td><td>44.7</td><td>61.4</td></tr><tr><td>SSD512-VGG16</td><td>29.4</td><td>49.3</td><td>31.0</td><td>11.7</td><td>34.1</td><td>44.9</td><td>42.7</td><td>17.6</td><td>48.7</td><td>60.6</td></tr><tr><td>+ \( KD_{cls} \)</td><td>30.3 (↑ 0.9)</td><td>51.1</td><td>31.7</td><td>12.7</td><td>34.6</td><td>45.5</td><td>43.3</td><td>19.4</td><td>49.0</td><td>60.4</td></tr><tr><td>+ loc</td><td>30.8 (↑ 1.4)</td><td>48.8</td><td>32.9</td><td>12.8</td><td>35.8</td><td>46.2</td><td>44.7</td><td>18.8</td><td>51.1</td><td>63.4</td></tr><tr><td>+ loc + \( KD_{loc} \)</td><td>31.0 (↑ 1.6)</td><td>49.1</td><td>32.8</td><td>12.6</td><td>35.8</td><td>46.2</td><td>45.0</td><td>18.9</td><td>51.6</td><td>63.2</td></tr><tr><td>+ \( KD_{cls} + loc + KD_{loc} \)</td><td>32.1 (↑ 2.7)</td><td>51.0</td><td>34.0</td><td>13.3</td><td>36.6</td><td>47.9</td><td>45.3</td><td>20.1</td><td>51.2</td><td>63.1</td></tr><tr><td>Faster RCNN-QR50</td><td>23.3</td><td>40.7</td><td>23.9</td><td>13.1</td><td>25.0</td><td>30.7</td><td>40.2</td><td>22.7</td><td>42.8</td><td>51.8</td></tr><tr><td>+ \( KD_{cls} \)</td><td>25.9 (↑ 2.6)</td><td>45.5</td><td>26.2</td><td>15.3</td><td>27.9</td><td>34.0</td><td>42.8</td><td>25.5</td><td>46.0</td><td>54.9</td></tr><tr><td>+ loc</td><td>24.2 (↑ 0.9)</td><td>41.1</td><td>25.0</td><td>13.7</td><td>25.8</td><td>32.1</td><td>41.7</td><td>23.8</td><td>44.3</td><td>54.8</td></tr><tr><td>+ loc + \( KD_{loc} \)</td><td>24.3 (↑ 1.0)</td><td>41.0</td><td>25.1</td><td>13.0</td><td>25.9</td><td>32.5</td><td>41.6</td><td>22.7</td><td>44.6</td><td>54.7</td></tr><tr><td>+ \( KD_{cls} + loc + KD_{loc} \)</td><td>27.2 (↑ 3.9)</td><td>46.0</td><td>27.7</td><td>15.2</td><td>29.3</td><td>36.2</td><td>44.5</td><td>25.9</td><td>48.1</td><td>58.3</td></tr><tr><td>Faster RCNN-MV2</td><td>31.9</td><td>52.0</td><td>34.0</td><td>18.5</td><td>34.4</td><td>41.0</td><td>47.5</td><td>29.7</td><td>50.9</td><td>60.4</td></tr><tr><td>+ \( KD_{cls} \)</td><td>32.6 (↑ 0.7)</td><td>53.3</td><td>34.6</td><td>18.9</td><td>34.8</td><td>42.3</td><td>48.1</td><td>29.7</td><td>51.2</td><td>61.5</td></tr><tr><td>+ loc</td><td>32.2 (↑ 0.3)</td><td>51.9</td><td>34.2</td><td>18.3</td><td>34.4</td><td>41.8</td><td>47.9</td><td>29.0</td><td>50.8</td><td>61.5</td></tr><tr><td>+ loc + \( KD_{loc} \)</td><td>32.3 (↑ 0.4)</td><td>52.0</td><td>34.7</td><td>18.1</td><td>34.8</td><td>41.6</td><td>48.0</td><td>28.7</td><td>51.3</td><td>61.6</td></tr><tr><td>+ \( KD_{cls} + loc + KD_{loc} \)</td><td>32.7 (↑ 0.8)</td><td>52.9</td><td>35.0</td><td>19.0</td><td>35.0</td><td>42.9</td><td>48.4</td><td>29.9</td><td>51.8</td><td>61.9</td></tr></table>

# 4 Experiments

In this section, we first conduct a full study of our classification and localization distillation methods on several compact detectors, and then compare our classifier-to-detector approach to the state-of-the-art detector-to-detector ones. Finally, we perform an extensive ablation study of our method and analyze how it improves the class recognition and localization in object detection. All models are trained and evaluated on MS COCO2017 [24], which contains over 118k images for training and 5k images for validation (minimal) depicting 80 foreground object classes. Our implementation is based on mmdetection [6] with Pytorch [30]. Otherwise specified, we take the ResNet50 as the classification teacher. We will use the same teacher for all two-stage Faster RCNNs and one-stage RetinaNets in our classifier-to-detector distillation method. We consider this to be an advantage of our method, since it lets us use the same teacher for multiple detectors. To train this classification teacher, we use the losses from Faster RCNN and RetinaNet frameworks jointly. Since SSDs use different data augmentation, we train another ResNet50 classification teacher for them. Additional experimental details on how to train our classification teachers are provided in the supplementary material.

# 4.1 Classifier-to-Detector Distillation on Compact Students

We first demonstrate the effectiveness of our classifier-to-detector distillation method on compact detectors, namely, SSD300, SSD512 and the two-stage Faster RCNN detector with lightweight backbones, i.e., MobileNetV2 and Quartered-ResNet50 (QR50), obtained by dividing the number of channels by 4 in every layer of ResNet50, reaching a  $66.33\%$  top-1 accuracy on ImageNet [21].

Experimental setting. All object detectors are trained in their default settings on Tesla V100 GPUs. The SSDs follow the basic training recipe in mmdetection [6]. The lightweight Faster RCNNs are trained with a  $1 \times$  training schedule for 12 epochs. The details for the training settings of each model are provided in the supplementary material. We use a ResNet50 with input resolution  $112 \times 112$  as classification teacher for all student detectors. We report the mean average precision (mAP) and mean average recall (mAR) for intersection over unions (IoUs) in [0.5:0.95], the APs at IoU=0.5 and 0.75, and the APs and ARs for small, medium and large objects.

Results. The results are shown in Table 1. Our classification distillation yields improvements of at least  $0.7\mathrm{mAP}$  for all student detectors. It reaches a  $2.3\mathrm{mAP}$  improvement for Faster RCNN-QR50, which indicates that the classification in this model is much weaker. The classification distillation improves  $\mathrm{AP}_{50}$  more than  $\mathrm{AP}_{75}$ , while the localization distillation improves  $\mathrm{AP}_{75}$  more than  $\mathrm{AP}_{50}$ . As

increasing  $\mathrm{AP}_{75}$  requires more precise localization, these results indicate that each of our distillation losses plays its expected role. Note that the SSDs benefit more from the localization method than the Faster RCNNs. We conjecture this to be due to the denser, more accurate proposals of the Faster RCNNs compared to the generic anchors of the SSDs. Note also that a Faster RCNNs with a smaller backbone benefits more from our distillation than a larger one.

# 4.2 Comparison with Detector-to-detector Distillation

We then compare our classifier-to-detector distillation approach with the state-of-the-art detector-to-detector ones, such as KD [5], FGFI [38], GID [8] and FKD [42]. Here, in addition to the compact students used in Section 4.1, we also report results on the larger students that are commonly used in the literature, i.e., Faster RCNN and RetinaNet with deeper ResNet50 (R50) backbones.

Experimental setting. Following [42], the Faster RCNN-R50 and RetinaNet-R50 are trained with a  $2 \times$  schedule for 24 epochs. To illustrate the generality of our approach, we also report the results of our distillation strategy used in conjunction with FKD [42], one of the current best detector-to-detector distillation methods. Note that, while preparing this work, we also noticed the concurrent work of [12], whose DeFeat method also follows a detector-to-detector distillation approach, and thus could also be complemented with out strategy.

# Results. We report the results in Table 2.

For compact student detectors, such as Faster RCNN-QR50 and SSD512, our classifier-to-detector distillation surpasses the best detector-to-detector one by 1.1 and  $0.9\mathrm{mAP}$  points, respectively. For student detectors with deeper backbones, our methods can improve the baseline by 0.4, 0.5 and 0.8 points. Furthermore, using it in conjunction with the FKD detector-to-detector distillation method boosts the performance to 41.9, 40.7 and  $34.2\mathrm{mAP}$ .

# 4.3 Ablation Study

In this section, we investigate the influence of the hyper-parameters and of different classification teachers in our approach. To this end, we use the SSD300 student detector.

Ablation study of  $\mathbf{KD}_{cls}$ . We first study the effect of the loss weight  $\lambda_{kc}$  and the temperature  $T$  for classification distillation. As shown in Table 3a, these two hyper-parameters have a mild impact on the results, and we obtain the best results with  $\lambda_{kc} = 0.4$  and  $T = 2$ , which were used for all other experiments.

We then investigate the impact of different classification teacher networks. To this end, we

Table 2: Comparison to detector-to-detector distillation methods on the COCO2017 validation set.  

<table><tr><td>Method</td><td>mAP</td><td>\( AP_s \)</td><td>\( AP_m \)</td><td>\( AP_l \)</td></tr><tr><td>Faster RCNN-R50</td><td>38.4</td><td>21.5</td><td>42.1</td><td>50.3</td></tr><tr><td>+ KD [5]</td><td>38.7</td><td>22.0</td><td>41.9</td><td>51.0</td></tr><tr><td>+ FGFI [38]</td><td>39.1</td><td>22.2</td><td>42.9</td><td>51.1</td></tr><tr><td>+ GID [8]</td><td>40.2</td><td>22.7</td><td>44.0</td><td>53.2</td></tr><tr><td>+ FKD [42]</td><td>41.5</td><td>23.5</td><td>45.0</td><td>55.3</td></tr><tr><td>+ Ours</td><td>38.8</td><td>22.5</td><td>42.5</td><td>50.8</td></tr><tr><td>+ Ours + FKD</td><td>41.9</td><td>23.8</td><td>45.2</td><td>56.0</td></tr><tr><td>RetinaNet-R50</td><td>37.4</td><td>20.0</td><td>40.7</td><td>49.7</td></tr><tr><td>+ FGFI [38]</td><td>38.6</td><td>21.4</td><td>42.5</td><td>51.5</td></tr><tr><td>+ GID [8]</td><td>39.1</td><td>22.8</td><td>43.1</td><td>52.3</td></tr><tr><td>+ FKD [42]</td><td>39.6</td><td>22.7</td><td>43.3</td><td>52.5</td></tr><tr><td>+ Ours</td><td>37.9</td><td>20.5</td><td>41.3</td><td>50.5</td></tr><tr><td>+ Ours + FKD</td><td>40.7</td><td>23.1</td><td>44.7</td><td>53.8</td></tr><tr><td>Faster RCNN-MV2</td><td>31.9</td><td>18.5</td><td>34.4</td><td>41.0</td></tr><tr><td>+ FKD [42]</td><td>33.9</td><td>18.3</td><td>36.3</td><td>45.4</td></tr><tr><td>+ Ours</td><td>32.7</td><td>19.0</td><td>35.0</td><td>42.9</td></tr><tr><td>+ Ours + FKD</td><td>34.2</td><td>18.5</td><td>36.3</td><td>45.9</td></tr><tr><td>Faster RCNN-QR50</td><td>23.3</td><td>13.1</td><td>25.0</td><td>30.7</td></tr><tr><td>+ FKD [42]</td><td>26.1</td><td>14.6</td><td>27.3</td><td>35.0</td></tr><tr><td>+ Ours</td><td>27.2</td><td>15.2</td><td>29.3</td><td>36.2</td></tr><tr><td>SSD512-VGG16</td><td>29.4</td><td>11.7</td><td>34.1</td><td>44.9</td></tr><tr><td>+ FKD [42]</td><td>31.2</td><td>12.6</td><td>37.4</td><td>46.2</td></tr><tr><td>+ Ours</td><td>32.1</td><td>13.3</td><td>36.6</td><td>47.9</td></tr></table>

trained three teacher networks ranging from shallow to deep: ResNet18, ResNet50 and ResNext101-  $32 \times 8d$ . We further study the impact of the input size to these teachers on classification distillation, using the three sizes  $[56 \times 56, 112 \times 112, 224 \times 224]$ . As shown in Table 3b, even the shallow ResNet18 classification teacher can improve the performance of the student detector by 0.3 points, and the improvement increases by another 0.4 points with the deeper ResNet50 teacher. However, the performance drops with the ResNeXt101 teacher, which is the teacher with the highest top-1 accuracy. This indicates that a deeper teacher is not always helpful, as it might be overconfident to bring much additional information compared to the ground-truth labels. As for the input size, we observe only small variations across the different sizes, and thus use a size of 112 in all other experiments.

Ablation study of  $\mathbf{KD}_{loc}$ . We then evaluate the influence of the two main hyper-parameters of localization distillation, i.e., the grid sampling size of the spatial transformer and the adaptive pooling

Table 3: Ablation study of  $\mathbf{KD}_{cls}$ . We evaluate the impact of the hyper-parameters and of various classification teachers on our classification distillation.  

<table><tr><td colspan="5">(a) Varying λkc and T.</td></tr><tr><td>λkc</td><td>T</td><td>mAP</td><td>AP50</td><td>AP75</td></tr><tr><td>baseline</td><td>/</td><td>25.6</td><td>43.8</td><td>26.3</td></tr><tr><td>0.1</td><td>1</td><td>25.8</td><td>44.2</td><td>26.6</td></tr><tr><td>0.1</td><td>2</td><td>25.4</td><td>44.4</td><td>25.7</td></tr><tr><td>0.2</td><td>1</td><td>25.8</td><td>44.2</td><td>26.6</td></tr><tr><td>0.3</td><td>1</td><td>26.0</td><td>44.6</td><td>26.7</td></tr><tr><td>0.4</td><td>1</td><td>26.1</td><td>44.8</td><td>26.6</td></tr><tr><td>0.4</td><td>2</td><td>26.3</td><td>45.2</td><td>27.2</td></tr><tr><td>0.4</td><td>3</td><td>26.0</td><td>45.2</td><td>26.7</td></tr></table>

<table><tr><td colspan="5">(b) Varying the teacher network.</td></tr><tr><td>Teacher</td><td>Top-1</td><td>mAP</td><td>AP50</td><td>AP75</td></tr><tr><td>ResNet18</td><td>75.78</td><td>25.9</td><td>44.4</td><td>26.4</td></tr><tr><td>ResNet50</td><td>80.30</td><td>26.3</td><td>45.2</td><td>27.2</td></tr><tr><td>ResNeXt101</td><td>83.35</td><td>25.3</td><td>43.3</td><td>25.8</td></tr><tr><td>Input size</td><td>Top-1</td><td>mAP</td><td>AP50</td><td>AP75</td></tr><tr><td>56 × 56</td><td>76.26</td><td>26.2</td><td>44.8</td><td>26.9</td></tr><tr><td>112 × 112</td><td>80.30</td><td>26.3</td><td>45.2</td><td>27.2</td></tr><tr><td>224 × 224</td><td>80.41</td><td>26.2</td><td>44.9</td><td>26.9</td></tr></table>

Table 4: Ablation study of  $\mathbf{KD}_{loc}$ . We investigate the effect of the sampling size, the pooling size and the choice of distilled layers on our localization distillation.  
(a) Varying the sampling size.  

<table><tr><td>Sampling size</td><td>mAP</td><td>AP50</td><td>AP75</td></tr><tr><td>14 × 14</td><td>26.4</td><td>43.0</td><td>27.0</td></tr><tr><td>28 × 28</td><td>26.7</td><td>43.2</td><td>27.8</td></tr><tr><td>56 × 56</td><td>26.8</td><td>43.3</td><td>28.0</td></tr><tr><td>112 × 112</td><td>27.0</td><td>43.5</td><td>28.1</td></tr><tr><td>224 × 224</td><td>27.0</td><td>43.4</td><td>28.2</td></tr></table>

(b) Varying the pooling size.  

<table><tr><td>Pooling size</td><td>mAP</td><td>\( AP_{50} \)</td><td>\( AP_{75} \)</td></tr><tr><td>2 × 2</td><td>26.6</td><td>43.5</td><td>27.5</td></tr><tr><td>4 × 4</td><td>27.0</td><td>43.5</td><td>28.1</td></tr><tr><td>8 × 8</td><td>27.1</td><td>43.2</td><td>28.4</td></tr><tr><td>16 × 16</td><td>26.9</td><td>42.8</td><td>28.1</td></tr></table>

(c) Varying distilled layers.  

<table><tr><td>obj</td><td>l0</td><td>l1</td><td>mAP</td></tr><tr><td>✓</td><td></td><td></td><td>27.1</td></tr><tr><td></td><td>✓</td><td></td><td>26.8</td></tr><tr><td>✓</td><td>✓</td><td></td><td>27.2</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>26.9</td></tr></table>

size of the feature maps. To this end, we vary the sampling size in [14, 28, 56, 112, 224] and the pooling size in  $[2 \times 2, 4 \times 4, 8 \times 8, 16 \times 16]$ .

As shown in Table 4a, our localization distillation method benefits from a larger sampling size, although the improvement saturates after a size of 112. This lets us use the same classification teacher, with input size 112, for both classification and localization distillation. The adaptive pooling size has a milder effect on the performance, as shown in Table 4b, with a size of 8 yielding the best mAP. In our experiments, we adopt either 4 or 8, according to the best performance on the validation set.

We further study the layers to be distilled in our localization distillation. To this end, we extract features from the first convolutional layer  $\ell_0$ , and from the following bottleneck block  $\ell_1$  of the ResNet50 teacher. As shown in Table 4c, distilling the knowledge of only the object regions (obj) yields a better mAP than using the  $\ell_0$  features. However, combining the object regions with the feature maps from  $\ell_0$  improves the results. Adding more layers does not help, which we conjecture to be due to the fact that these layers extract higher-level features that are thus less localized.

# 4.4 Analysis

To further understand how our classifier-to-detector distillation method affects the quality of the classification and localization, in Table 5, we report the APs obtained with IoUs in [0.5, 0.95] with a step of 0.05. These results highlight that our classification and localization distillation strategies behave differently for different IoU thresholds. Specifically,  $\mathrm{KD}_{cls}$  yields larger improvements for smaller IoUs, whereas  $\mathrm{KD}_{loc}$  is more effective with IoUs larger than 0.75. This indicates that  $\mathrm{KD}_{loc}$  indeed focuses on precise

![](images/11f020d20d76ec4624f8596a1a6c25501533921f49924145f3987bb2bfd06dc6.jpg)  
(a) Classification error.  
Figure 3: Detection error analysis.

![](images/f86a58b34d7bd216c65da01f7218a6215bf2904fc8257610245a0c3f3f7e937f.jpg)  
(b) Localization error.

localization, while  $\mathrm{KD}_{cls}$  distills category information. The complementarity of both terms is further evidenced by the fact that all APs increase when using both of them jointly.

Detection error analysis. We analyze the different types of detection errors using the tool proposed by Bolya et al. [3] for the baseline SSD300 and the distilled models with our  $\mathrm{KD}_{cls} + loc + \mathrm{KD}_{loc}$ . We focus on the classification and localization errors, which are the main errors in object detection.

Table 5: APs for IoUs ranging from 0.5 to 0.95 on the COCO2017 validation set.  

<table><tr><td>Method</td><td>mAP</td><td>\( AP_{50} \)</td><td>\( AP_{55} \)</td><td>\( AP_{60} \)</td><td>\( AP_{65} \)</td><td>\( AP_{70} \)</td><td>\( AP_{75} \)</td><td>\( AP_{80} \)</td><td>\( AP_{85} \)</td><td>\( AP_{90} \)</td><td>\( AP_{95} \)</td></tr><tr><td>SSD300</td><td>25.6</td><td>43.8</td><td>41.3</td><td>38.4</td><td>35.1</td><td>31.2</td><td>26.3</td><td>20.3</td><td>13.0</td><td>5.2</td><td>0.5</td></tr><tr><td>+ \( KD_{cls} \)</td><td>26.3</td><td>45.2</td><td>42.6</td><td>39.9</td><td>36.1</td><td>31.6</td><td>27.2</td><td>21.0</td><td>13.5</td><td>5.1</td><td>0.5</td></tr><tr><td>+ \( loc + KD_{loc} \)</td><td>27.2</td><td>43.3</td><td>41.3</td><td>38.8</td><td>36.0</td><td>32.9</td><td>28.5</td><td>23.0</td><td>16.5</td><td>8.4</td><td>1.3</td></tr><tr><td>+ \( KD_{cls} + loc + KD_{loc} \)</td><td>27.9</td><td>45.1</td><td>42.8</td><td>40.2</td><td>37.0</td><td>34.0</td><td>29.2</td><td>23.9</td><td>17.0</td><td>8.8</td><td>1.2</td></tr></table>

![](images/33b48aaa581e563be7f75b751d67e4255b18c9dcf01b7feffe080a50b5ed1401.jpg)  
Figure 4: Qualitative analysis (better viewed in color). The ground-truth bounding boxes are in blue with their labels, and the predictions are in red with predicted labels and confidence.

The details of all error types are provided in the supplementary material. As shown in Figure 3a,  $\mathrm{KD}_{cls}$  decreases the classification error especially for IoUs smaller than 0.65. By contrast, as shown in Figure 3b, the effect of  $\mathrm{KD}_{loc}$  increases with the IoU. This again shows the complementary nature of these terms.

Qualitative analysis. Figure 4 compares the detection results of the baseline model and of our distilled model on a few images. We observe that (i) the bounding box predictions of the distilled model are more precise than those of the baseline; (ii) the distilled model generates higher confidence for the correct predictions and is thus able to detect objects that were missed by the baseline, such as the boat in Figure 4c and the giraffe in Figure 4d.

# 5 Conclusion

We have introduced a novel approach to knowledge distillation for object detection, replacing the standard detector-to-detector strategy with a classifier-to-detector one. To this end, we have developed a classification distillation loss function and a localization distillation one, allowing us to exploit the classification teacher in two complementary manners. Our approach outperforms the state-of-the-art detector-to-detector ones on compact student detectors. While the improvement decreases for larger student networks, our approach can nonetheless boost the performance of detector-to-detector distillation. We have further shown that the same classification teacher could be used for all student detectors if they employ the same data augmentation strategy, thus reducing the burden of training a separate teacher for every student detector. Ultimately, we believe that our work opens the door to a new approach to distillation beyond object detection: Knowledge should be transferred not only across architectures, but also across tasks.

# Broader impact

Knowledge distillation is a simple yet effective method to improve the performance of a compact neural network by exploiting the knowledge of a more powerful teacher model. Our work introduces a general approach to knowledge distillation for object detection to transfer knowledge across architectures and tasks. Our approach enables distilling knowledge from a single classification teacher into different student detectors. As such, our work reduces the need for a separate deep teacher detector for each student networks; therefore, we reduce training resources and memory footprint. As we focus on compact networks, our work could significantly impact applications in resource-constrained environments, such as mobile phones, drones, or autonomous vehicles. We do not foresee any obvious undesirable ethical/social impact at this moment.

# References

[1] J. M. Alvarez and M. Salzmann. Learning the number of neurons in deep networks. In Advances in Neural Information Processing Systems. 2016.  
[2] J. M. Alvarez and M. Salzmann. Compression-aware training of deep networks. In Advances in Neural Information Processing Systems. 2017.  
[3] D. Bolya, S. Foley, J. Hays, and J. Hoffman. Tide: A general toolbox for identifying object detection errors. In European Conference on Computer Vision, 2020.  
[4] Z. Cai and N. Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In Conference on Computer Vision and Pattern Recognition, 2018.  
[5] G. Chen, W. Choi, X. Yu, T. Han, and M. Chandraker. Learning efficient object detection models with knowledge distillation. In Advances in Neural Information Processing Systems, 2017.  
[6] K. Chen, J. Wang, J. Pang, Y. Cao, Y. Xiong, X. Li, S. Sun, W. Feng, Z. Liu, J. Xu, Z. Zhang, D. Cheng, C. Zhu, T. Cheng, Q. Zhao, B. Li, X. Lu, R. Zhu, Y. Wu, J. Dai, J. Wang, J. Shi, W. Ouyang, C. C. Loy, and D. Lin. MMDetection: Open mmlab detection toolbox and benchmark. arXiv Preprint, 2019.  
[7] M. Courbariaux, I. Hubara, D. Soudry, R. El-Yaniv, and Y. Bengio. Binarized neural networks: Training deep neural networks with weights and activations constrained to + 1 or-1. arXiv Preprint, 2016.  
[8] X. Dai, Z. Jiang, Z. Wu, Y. Bao, Z. Wang, S. Liu, and E. Zhou. General instance distillation for object detection. arXiv Preprint, 2021.  
[9] K. Duan, S. Bai, L. Xie, H. Qi, Q. Huang, and Q. Tian. Centernet: Keypoint triplets for object detection. In International Conference on Computer Vision, 2019.  
[10] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2007 (VOC2007) Results. http://www.pascalnetwork.org/challenges/VOC/voc2007/workshop/index.html, 2007.  
[11] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results. http://www.pascalnetwork.org/challenges/VOC/voc2012/workshop/index.html, 2012.  
[12] J. Guo, K. Han, Y. Wang, H. Wu, X. Chen, C. Xu, and C. Xu. Distilling object detectors via decoupled features. arXiv Preprint, 2021.  
[13] S. Han, H. Mao, and W. J. Dally. Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding. In International Conference on Learning Representations, 2016.  
[14] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Conference on Computer Vision and Pattern Recognition, 2016.  
[15] K. He, G. Gkioxari, P. Dollar, and R. Girshick. Mask r-cnn. In International Conference on Computer Vision, 2017.  
[16] T. He, C. Shen, Z. Tian, D. Gong, C. Sun, and Y. Yan. Knowledge adaptation for efficient semantic segmentation. In Conference on Computer Vision and Pattern Recognition, 2019.  
[17] B. Heo, J. Kim, S. Yun, H. Park, N. Kwak, and J. Y. Choi. A comprehensive overhaul of feature distillation. In International Conference on Computer Vision, 2019.  
[18] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. arXiv Preprint, 2015.  
[19] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv Preprint, 2017.

[20] M. Jaderberg, K. Simonyan, A. Zisserman, and k. kavukcuoglu. Spatial transformer networks. In Advances in Neural Information Processing Systems, 2015.  
[21] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, 2012.  
[22] H. Law and J. Deng. Cornernet: Detecting objects as paired keypoints. In European Conference on Computer Vision, 2018.  
[23] N. Lee, T. Ajanthan, and P. H. Torr. Snip: Single-shot network pruning based on connection sensitivity. In International Conference on Learning Representations, 2019.  
[24] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and L. Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, 2014.  
[25] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie. Feature pyramid networks for object detection. In Conference on Computer Vision and Pattern Recognition, 2017.  
[26] T.-Y. Lin, P. Goyal, R. B. Girshick, K. He, and P. Dollar. Focal loss for dense object detection. International Conference on Computer Vision, 2017.  
[27] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg. Ssd: Single shot multibox detector. In European Conference on Computer Vision, 2016.  
[28] Y. Liu, K. Chen, C. Liu, Z. Qin, Z. Luo, and J. Wang. Structured knowledge distillation for semantic segmentation. In Conference on Computer Vision and Pattern Recognition, 2019.  
[29] B. McFee, J. Salamon, and J. Bello. Adaptive pooling operators for weakly labeled sound event detection. IEEE/ACM Transactions on Speech and Language Processing, 26(11):2180-2193, 2018. ISSN 2329-9290.  
[30] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in pytorch. 2017.  
[31] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi. Xnor-net: Imagenet classification using binary convolutional neural networks. In European Conference on Computer Vision, 2016.  
[32] J. Redmon and A. Farhadi. Yolov3: An incremental improvement. arXiv Preprint, 2018.  
[33] S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in Neural Information Processing Systems, 2015.  
[34] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio. Fitnets: Hints for thin deep nets. arXiv Preprint, 2014.  
[35] Y. Tian, D. Krishnan, and P. Isola. Contrastive representation distillation. In International Conference on Learning Representations, 2020.  
[36] Z. Tian, C. Shen, H. Chen, and T. He. Fcos: Fully convolutional one-stage object detection. In International Conference on Computer Vision, October 2019.  
[37] K. Ullrich, E. Meeds, and M. Welling. Soft weight-sharing for neural network compression. In International Conference on Learning Representations, 2017.  
[38] T. Wang, L. Yuan, X. Zhang, and J. Feng. Distilling object detectors with fine-grained feature imitation. In Conference on Computer Vision and Pattern Recognition, 2019.  
[39] Z. Yang, S. Liu, H. Hu, L. Wang, and S. Lin. Repoints: Point set representation for object detection. In International Conference on Computer Vision, 2019.  
[40] J. Yim, D. Joo, J. Bae, and J. Kim. A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. In Conference on Computer Vision and Pattern Recognition, 2017.

[41] S. Zagoruyko and N. Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. In International Conference on Learning Representations, 2017.  
[42] L. Zhang and K. Ma. Improve object detection with feature-based knowledge distillation: Towards accurate and efficient detectors. 2021.  
[43] R. Zhao, Y. Hu, J. Dotzel, C. De Sa, and Z. Zhang. Improving neural network quantization without retraining using outlier channel splitting. In International Conference on Machine Learning, 2019.
