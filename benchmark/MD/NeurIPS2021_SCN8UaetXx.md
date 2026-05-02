# Efficient Training of Visual Transformers with Small-Size Datasets

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Visual Transformers (VTs) are emerging as an architectural paradigm alternative to Convolutional networks (CNNs). Differently from CNNs, VTs can capture global relations between image elements and they potentially have a larger representation capacity. However, the lack of the typical convolutional inductive bias makes these models more data-hungry than common CNNs. In fact, some local properties of the visual domain which are embedded in the CNN architectural design, in VTs should be learned from samples. In this paper, we empirically analyse different VTs, comparing their robustness in a small training-set regime, and we show that, despite having a comparable accuracy when trained on ImageNet, their performance on smaller datasets can be largely different. Moreover, we propose a self-supervised task which can extract additional information from images with only a negligible computational overhead. This task encourages the VTs to learn spatial relations within an image and makes the VT training much more robust when training data are scarce. Our task is used jointly with the standard (supervised) training and it does not depend on specific architectural choices, thus it can be easily plugged in the existing VTs. Using an extensive evaluation with different VTs and datasets, we show that our method can improve (sometimes dramatically) the final accuracy of the VTs. The code will be available upon acceptance.

# 1 Introduction

Visual Transformers (VTs) are progressively emerging architectures in computer vision as an alternative to standard Convolutional Neural Networks (CNNs), and they have already been applied to many tasks, such as image classification [16, 51, 59, 34, 56, 58, 33, 57], object detection [4, 61, 13], segmentation [48], tracking [35] and image generation [30, 28], to mention a few. These architectures are inspired by the well-known Transformer [53], which is the de-facto standard in Natural Language Processing (NLP) [14, 44], and one of their appealing properties is the possibility to develop a unified information-processing paradigm for both visual and textual domains. A pioneering work in this direction is ViT [16], in which an image is split using a grid of non-overlapping patches, and each patch is linearly projected in the input embedding space, so obtaining a "token". After that, all the tokens are processed by a series of multi-head attention and feed-forward layers, similarly to how (word) tokens are processed in NLP Transformers.

A clear advantage of VTs is the possibility for the network to use the attention layers to model global relations between tokens, and this is the main difference with respect to CNNs, where the receptive field of the convolutional kernels locally limits the type of relations which can be learned. However, the increased representation capacity of the VTs comes at a price, which is the lack of the typical CNN inductive biases, based on exploiting the locality, the translation invariance and the hierarchical structure of visual information [34, 56, 58]. As a result, VTs need a lot of data for training, usually more than what is necessary to standard CNNs [16]. For instance, ViT is trained with

JFT-300M [16], a (proprietary) huge dataset of 303 million (weakly) labeled high-resolution images, and performs worse than ResNets [24] with similar capacity when trained on ImageNet-1K ( $\sim$  1.3 million samples [46]). This is likely due to the fact that ViT needs to learn some local properties of the visual data using more samples than a CNN, while the latter embeds these properties in its architectural design.

In order to alleviate this problem, very recently a second generation of VTs has been independently proposed by different groups [59, 34, 56, 58, 57, 33, 28]. A common idea behind these works is to mix convolutional layers with attention layers, in such a way providing a local inductive bias to the VT. These hybrid architectures enjoy the advantages of both paradigms: attention layers model long-range dependencies, while convolutional operations can emphasise the local properties of the image content. The empirical results shown in most of these works demonstrate that this second-generation VTs can be trained on ImageNet outperforming similar-size ResNets on this dataset [59, 34, 56, 58, 57, 33]. However, it is still not clear what is the behaviour of these networks when trained on medium-small size datasets. In fact, from an application point of view, most of the vision tasks cannot rely on (supervised) datasets whose size is comparable with (or larger than) ImageNet.

In this paper, we compare to each other different second-generation VTs by either training them from scratch or fine-tuning them on medium-small size datasets, and we empirically show that, despite their ImageNet results are basically on par with each other, their classification accuracy with smaller datasets largely varies. Moreover, we propose to use an additional self-supervised pretext task and a corresponding loss function in order to accelerate training in a small training-set or few-epochs regime. Specifically, the proposed task is based on (unsupervised) learning the spatial relations between the output-token embeddings. Given an image, we densely sample random pairs from the final embedding grid, and, for each pair, we ask the network to guess their relative translation offsets. To solve this task, the network needs to encode both local and contextual information in each embedding. In fact, without local information, embeddings representing different input image patches cannot be distinguished the one from the others, while, without contextual information (aggregated using the attention layers), the task may be ambiguous.

Our task is inspired by ELECTRA [11], an NLP model in which the pretext task is densely defined for each output embedding (Sec. 2). Clark et al. [11] show that their task is more sample-efficient than commonly used NLP pretext tasks, and this gain is particularly strong with small-capacity models or relatively smaller training sets. Similarly, we exploit the fact that an image is represented by a VT using multiple token embeddings, and we use their relative distances to define a localization task over a subset of all the possible embedding pairs. In this way, for a single image forward, we can compare to each other many embedding pairs and average our localization loss over all of them. Thus, our task is drastically different from those multi-crop strategies proposed, for instance, in SwAV [6], which need to independently forward each input patch through the network.

Since our additional task is self-supervised, our dense relative localization loss  $(\mathcal{L}_{drloc})$  does not require additional annotation, and we use it jointly with the standard (supervised) cross-entropy as a regularization of the VT training.  $\mathcal{L}_{drloc}$  is very easy-to-be-reproduced and, despite this simplicity, it can largely boost the accuracy of the VTs, especially when the VT is either trained from scratch on a small-size dataset, or fine-tuned on a dataset with a large domain-shift with respect to the pretraining ImageNet dataset. In our empirical analysis, based on different training scenarios, a variable amount of training data and three different second-generation VTs,  $\mathcal{L}_{drloc}$  has always improved the results of the tested baselines, sometimes boosting the final accuracy of tens of points (and up to 45 points).

In summary, the contributions of this paper are the following:

- We empirically compare to each other different very recent VTs, and we show that their behaviour can largely differ when trained with small-size datasets or few training epochs.  
- We propose a new, straightforward self-supervised relative localization loss which is used as an additional task for VT training. Using an extensive empirical analysis, we show that our loss is beneficial to speed-up training and increase the generalization ability of different VTs, independently of their specific architectural design or application task.

# 2 Related work

In this section, we briefly review previous work related to both VTs and self-supervised learning.

Visual Transformers. Despite some previous work in which attention is used inside the convolutional layers of a CNN [55, 26], the first fully-transformer architectures for vision are iGPT [7] and ViT [16]. The former is trained using a "masked-pixel" self-supervised approach, similar in spirit to the common masked-word task used, for instance, in BERT [14] and in GPT [44] (see below). On the other hand, ViT is trained in a supervised way, using a special "class token" and a classification head attached to the final embedding of this token. Both methods are computationally expensive and, despite their very good results when trained on huge datasets, they underperform ResNet architectures when trained from scratch using only ImageNet-1K [16, 7]. VideoBERT [49] is conceptually similar to iGPT, but, rather than using pixels as tokens, each frame of a video is holistically represented by a feature vector, which is quantized using an off-the-shelf pretrained video classification model. DeiT [51] trains ViT using distillation information provided by a pretrained CNN.

The success of ViT has attracted a lot of interest in the vision community, and different variants of this architecture have been recently used in many tasks [51, 48, 30, 10]. However, as mentioned in Sec. 1, the lack of the typical CNN inductive biases in ViT, makes this model difficult to train without using (very) large-size datasets. For this reason, very recently, a second-generation of VTs has focused on hybrid architectures, in which convolutions are used jointly with long-range attention layers [59, 34, 56, 58, 57, 33, 28]. The common idea behind all these works is that the sequence of the individual token embeddings can be shaped/reshaped in a geometric grid, in which the position of each embedding vector corresponds to a fixed location in the input image. Given this geometric layout of the embeddings, convolutional layers can be applied to neighboring embeddings, so encouraging the network to focus on local properties of the image. The main difference among these works concerns where the convolutional operation is applied (e.g., only in the initial representations [59] or in all the layers [34, 56, 58, 57, 33], in the token to query/key/value projections [56] or in the forward-layers [58, 33, 28], etc.). In this paper we do not use ViT, being the original ViT architectures too big as number of parameters, and because we focus on training/fine-tuning on datasets whose size is smaller than ImageNet-1K. Conversely, we use three state-of-the-art second-generation VTs (T2T [59], Swin [34] and CvT [56]), for which there is a public implementation. For each of them, we select the model whose number of parameters is comparable with a ResNet-50 [24] (more details in Sec. 3). We do not modify the native architectures because the goal of this work is to propose a pretext task and a loss function which can be easily plugged in existing VTs.

Similarly to the original Transformer [53], in ViT, an (absolute) positional embedding is added to the representation of the input tokens. In Transformer networks, positional embedding is used to provide information about the token order, since both the attention and the (individual token based) feed-forward layers are permutation invariant. In [34, 57], relative positional embedding [47] is used, where the position of each token is represented relatively to the others. Generally speaking, positional embedding is a representation of the token position which is provided as input to the network. Conversely, our relative localization loss exploits the relative positions (of the final VT embeddings) as a pretext task to extract additional information without manual supervision.

Self-supervised learning. Reviewing the vast self-supervised learning literature is out of the scope of this paper. However, we briefly mention that self-supervised learning was first successfully applied in NLP, as a means to get supervision from text by replacing costly manual annotations with pretext tasks [36, 37]. A typical NLP pretext task consists in masking a word in an input sentence and asking the network to guess which is the masked token [36, 37, 14, 44]. ELECTRA [11] is a sample-efficient language model in which the masked-token pretext task is replaced by a discriminative task defined over all the tokens of the input sentence. Our work is inspired by this method, since we propose a pretext task which can be efficiently computed by densely sampling the final VT embeddings.

In vision, common pretext tasks with still images are based on extracting two different views from the same image (e.g., two different crops) and then considering these as a pair of positive images, likely sharing the same semantic content [8]. Current self-supervised vision approaches can be broadly categorised in contrastive learning [52, 25, 8, 23, 50, 54, 18], clustering methods [3, 62, 29, 5, 1, 20, 6], asymmetric networks [22, 9] and feature-decorrelation methods [19, 60, 2, 27]. While the aforementioned approaches are all based on ResNets, very recently, Chen et al. [10] have empirically tested some of these methods with a ViT architecture [16].

One important difference of our proposal with respect to previous work, is that we do not propose a fully-self-supervised method, but we rather use self-supervision jointly with standard supervision (i.e., image labels) in order to regularize VT training, hence our framework is a multi-task learning

approach [12]. Moreover, our dense relative localization loss is not based on positive pairs, and we do not use multiple views of the same image in the current batch, thus our method can be used with standard (supervised) data-augmentation techniques. Specifically, our pretext task is based on predicting the relative positions of pairs of tokens extracted from the same image.

Previous work using localization for self-supervision is based on predicting the input image rotation [21] or the relative position of adjacent patches extracted from the same image [15, 41, 42, 38]. For instance, in [41], the network should predict the correct permutation of a grid of  $3 \times 3$  patches (in NLP, a similar, permutation based pretext task, is *deshuffling* [45]). In contrast, we do not need to extract multiple patches from the same input image, since we can efficiently use the final token embeddings (thus, we need a single forward and backward pass per image). Moreover, differently from previous work based on localization pretext tasks, our loss is densely computed between many random pairs of (non necessarily adjacent) token embeddings. Note that one of the reasons for which we can use non-adjacent image positions, is that the attention layers of the VT include contextual information in each token representation, thus making the prediction task easier. Finally, in [13], the position of a random query patch is used for self-supervised training a transformer-based object detector [4]. However, the localization loss used in [13] is specific for the final task (object localization) and the specific DETR architecture [4], while our loss is generic and can be plugged in any VT.

![](images/9b4abbf52080c5abd517d6ec3688a14ffa0c7afd7a46c1ecb2f94780fe7ee652.jpg)  
Figure 1: A schematic representation of the VT architecture. (a) A typical second-generation VT. (b) Our localization MLP which takes as input (concatenated) pairs of final token embeddings.

# 3 Preliminaries

As mentioned in Sec. 1-2, in this paper we focus on second-generation discriminative VTs [59, 34, 56, 58, 57, 33] which are hybrid architectures mixing Transformer-like multi-attention layers with convolutional operations. Without loss of generality, these networks take as input an image which is split in a grid of (possibly overlapping)  $K \times K$  patches. Each patch is projected in the input embedding space, obtaining a set of  $K \times K$  input tokens, fed to the VT. The latter is based on the typical Transformer multi-attention layers [53], which model pairwise relations over the token intermediate representations. Importantly, the attention layers gradually introduce contextual information in each token representation. Differently from a pure Transformer [53], hybrid architectures usually shape or re-shape the sequence of these token embeddings in a spatial grid, which makes it possible to apply convolutional operations over a small set of neighboring token embeddings. Using convolutions with a stride greater than 1 and/or pooling operations, the resolution of the initial  $K \times K$  token grid can possibly be reduced, thus simulating the hierarchical structure of a CNN. We assume that the final embedding grid has a resolution of  $k \times k$  (where, usually,  $k \leq K$ ), see Fig. 1 (a).

The final  $k \times k$  grid of embeddings represents the input image and it is used for the discriminative task. For instance, some methods include an additional "class token" which collects contextual information over the whole grid [59, 56, 58, 57, 33], while others [34] apply an average global pooling over the final grid to get a compact representation of the whole image. Finally, a standard, small MLP head takes as input the whole image representation and it outputs a posterior distribution over the set of the target classes (Fig. 1 (a)). The VT is trained using a standard cross-entropy loss  $(\mathcal{L}_{ce})$ , computed using these posteriors and the image ground-truth labels.

When we plug our relative localization loss (Sec. 4) in an existing VT, we always use the native VT architecture of each tested method, without any change apart from the dedicated localization MLP (see Sec. 4). For instance, we use the class token when available, or the average pooling layer when

it is not, and on top of these we use the cross-entropy loss. We also keep the positional embedding (Sec. 2) for those VTs which add this information to the tokens (see Sec. 4.1 for a discussion about this choice). The only architectural change we do is to downsample the final embedding grid of T2T [59] and CvT [56] to make them of the same size as that used in Swin [6]. Specifically, in Swin, the final grid has a resolution of  $7 \times 7$  ( $k = 7$ ), while, in T2T and in CvT, it is  $14 \times 14$ . Thus, in T2T and in CvT, we use a  $2 \times 2$  average pooling (without learnable parameters) and we get a final  $7 \times 7$  grid for all the three tested architectures. This pooling operation is motivated in Sec. 4.1, and it is used only together with our localization task (it does not affect the posterior computed by the classification MLP). Finally, note that T2T uses convolutional operations only in the input stage, and it outputs a sequence of  $14 \times 14 = 196$  embeddings, corresponding to its  $14 \times 14$  input grid. In this case, we first reshape the sequence and then we use pooling.

# 4 Dense Relative Localization task

The goal of our regularization loss is to encourage the VT to learn spatial information without using additional manual annotations. We achieve this by densely sampling multiple embedding pairs for each image and asking the network to guess their relative positions. In more detail, given an image  $x$ , we denote its corresponding  $k \times k$  grid of final embeddings (Sec. 3), as  $G_{x} = \{\mathbf{e}_{i,j}\}_{1 \leq i,j \leq k}$ , where  $\mathbf{e}_{i,j} \in \mathbb{R}^{D}$ , and  $D$  is the dimension of the embedding space. For each  $G_{x}$ , we randomly sample multiple pairs of embeddings and, for each pair  $(\mathbf{e}_{i,j}, \mathbf{e}_{p,h})$ , we compute the 2D normalized target translation offset  $(t_{u}, t_{v})^{T}$ , where:

$$
t _ {u} = \frac {| i - p |}{k}, \quad t _ {v} = \frac {| j - h |}{k}, \quad \left(t _ {u}, t _ {v}\right) ^ {T} \in [ 0, 1 ] ^ {2}. \tag {1}
$$

The selected embedding vectors  $\mathbf{e}_{i,j}$  and  $\mathbf{e}_{p,h}$  are concatenated and input to a small MLP  $(f)$ , with only one hidden layer and two output neurons, one per spatial dimension (Fig. 1 (b)), which predicts the relative distance between position  $(i,j)$  and position  $(p,h)$  on the grid. Let  $(d_u,d_v)^T = f(\mathbf{e}_{i,j},\mathbf{e}_{p,h})^T$ . Given a mini-batch  $B$  of  $n$  images, our dense relative localization loss is:

$$
\mathcal {L} _ {d r l o c} = \sum_ {x \in B} \mathbb {E} _ {\left(\mathbf {e} _ {i, j}, \mathbf {e} _ {p, h}\right) \sim G _ {x}} [ \left| \left(t _ {u}, t _ {v}\right) ^ {T} - \left(d _ {u}, d _ {v}\right) ^ {T} \right| _ {1} ]. \tag {2}
$$

In Eq. (2), for each image  $x$ , the expectation is computed by sampling uniformly at random  $m$  embedding pairs  $(\mathbf{e}_{i,j}, \mathbf{e}_{p,h})$  in  $G_{x}$ , and averaging the  $L_{1}$  loss between the corresponding  $(t_{u}, t_{v})^{T}$  and  $(d_{u}, d_{v})^{T}$ .  $\mathcal{L}_{drloc}$  is added to the standard cross-entropy loss  $(\mathcal{L}_{ce})$  of each native VT (Sec. 3). The final loss is:  $\mathcal{L}_{tot} = \mathcal{L}_{ce} + \lambda \mathcal{L}_{drloc}$ . We use  $\lambda = 0.1$  in all the experiments with both T2T and CvT, and  $\lambda = 0.5$  in case of Swin.

# 4.1 Discussion

Intuitively,  $\mathcal{L}_{drloc}$  transforms the relative positional embedding (Sec. 2), used, for instance, in Swin [34], in a pretext task, asking the network to guess which is the relative distance of a random subset of all the possible token pairs. Thus a question may arise: is the relative positional embedding used in some VTs sufficient for the localization MLP  $(f)$  to solve the localization task? The experiments presented in Sec. 5.2-5.3 show that, when we plug  $\mathcal{L}_{drloc}$  on CvT, in which no kind of positional embedding is used [56], the relative accuracy boost is usually smaller than in case of Swin, confirming that the relative positional embedding, used in the latter, is not sufficient to make our task trivial.

In Sec. 3, we mentioned that, in case of Swin, the final embedding grid has a  $7 \times 7$  resolution, while for the other two VTs we consider here (T2T and CvT), we average-pool their  $14 \times 14$  grids and we obtain a final  $7 \times 7$  grid  $G_{x}$ . In fact, in preliminary experiments with both T2T and CvT at their original  $14 \times 14$  resolution, we observed a very slow convergence of  $\mathcal{L}_{drloc}$ . We presume this is due to the fact that, with a finer grid, the localization task is harder. This makes more difficult the convergence of  $f$ , and it likely generates noisy gradients which are backpropagated through the whole VT (see also Sec. 5.1). We leave this for future investigation and, in the rest of this article, we always assume that our pretext task is computed with a  $7 \times 7$  grid  $G_{x}$ .

# 4.2 Loss variants

In this section, we present different variants of the relative localization loss which will be empirically analyzed in Sec. 5.1.

The first variant consists in including negative target offsets:

$$
t _ {u} ^ {\prime} = \frac {i - p}{k}, \quad t _ {v} ^ {\prime} = \frac {j - h}{k}, \quad \left(t _ {u} ^ {\prime}, t _ {v} ^ {\prime}\right) ^ {T} \in [ - 1, 1 ] ^ {2}. \tag {3}
$$

Replacing  $(t_u,t_v)^T$  in Eq. (2) with  $(t_u',t_v')^T$  computed as in Eq. (3), and keeping all the rest unchanged, we obtain the first variant, which we call  $\mathcal{L}_{drloc}^{*}$ .

In the second variant, we transform the regression task in Eq. (2) in a classification task, and we replace the  $L_{1}$  loss with the cross-entropy loss. In more detail, we use as target offsets:

$$
c _ {u} = i - p, \quad c _ {v} = j - h, \quad \left(c _ {u}, c _ {v}\right) ^ {T} \in \{- k, \dots , k \} ^ {2}, \tag {4}
$$

and we associate each of the  $2k + 1$  discrete elements in  $C = \{-k,\dots,k\}$  with a "class". Accordingly, the localization MLP  $f$  is modified by replacing the 2 output neurons with 2 different sets of neurons, one per spatial dimension ( $u$  and  $v$ ). Each set of neurons represents a discrete offset prediction over the  $2k + 1$  "classes" in  $C$ . Softmax is applied separately to each set of  $2k + 1$  neurons, and the output of  $f$  is composed of two posterior distributions over  $C$ :  $(\mathbf{p}_u,\mathbf{p}_v)^T = f(\mathbf{e}_{i,j},\mathbf{e}_{p,h})^T$ , where  $\mathbf{p}_u,\mathbf{p}_v\in [0,1]^{2k + 1}$ . Eq. (2) is then replaced by:

$$
\mathcal {L} _ {d r l o c} ^ {c e} = - \sum_ {x \in B} \mathbb {E} _ {\left(\mathbf {e} _ {i, j}, \mathbf {e} _ {p, h}\right) \sim G _ {x}} [ \log \left(\mathbf {p} _ {u} [ c _ {u} ] + \log \left(\mathbf {p} _ {v} [ c _ {v} ] \right], \right. \tag {5}
$$

where  $\mathbf{p}_u[c_u]$  indicates the  $c_{u}$ -th element of  $\mathbf{p}_u$  (and similarly for  $\mathbf{p}_v[c_v]$ ).

Note that, using the cross-entropy loss in Eq. (5), corresponds to considering  $C$  an unordered set of "categories". This implies that prediction errors in  $\mathbf{p}_u$  (and  $\mathbf{p}_v$ ) are independent of the "distance" with respect to the ground-truth  $c_u$  (respectively,  $c_v$ ). In order to alleviate this problem, and inspired by [17], the third variant we propose imposes a Gaussian prior on  $\mathbf{p}_u$  and  $\mathbf{p}_v$ , and minimizes the normalized squared distance between the expectation of  $\mathbf{p}_u$  and the ground-truth  $c_u$  (respectively,  $\mathbf{p}_v$  and  $c_v$ ). In more detail, let  $\mu_u = \sum_{c\in C}\mathbf{p}_u[c]*c$  and  $\sigma_u^2 = \sum_{c\in C}\mathbf{p}_u[c]*(c - \mu_u)^2$  (and similarly for  $\mu_v$  and  $\sigma_v^2$ ). Then, Eq. (5) is replaced by:

$$
\mathcal {L} _ {d r l o c} ^ {r e g} = \sum_ {x \in B} \mathbb {E} _ {\left(\mathbf {e} _ {i, j}, \mathbf {e} _ {p, h}\right) \sim G _ {x}} \left[ \frac {\left(c _ {u} - \mu_ {u}\right) ^ {2}}{\sigma_ {u} ^ {2}} + \alpha l o g \left(\sigma_ {u}\right) + \frac {\left(c _ {v} - \mu_ {v}\right) ^ {2}}{\sigma_ {v} ^ {2}} + \alpha l o g \left(\sigma_ {v}\right) \right], \tag {6}
$$

where the terms  $\log(\sigma_u)$  and  $\log(\sigma_v)$  are used for variance regularization [17].

The last variant we propose is based on a "very-dense" localization loss, where  $\mathcal{L}_{drloc}$  is computed for every transformer block of VT. Specifically, let  $G_{x}^{l}$  be the  $k_{l} \times k_{l}$  grid of token embeddings output by the  $l$ -th block of VT, and let  $L$  be the total number of these blocks. Then, Eq. (2) is replaced by:

$$
\mathcal {L} _ {d r l o c} ^ {a l l} = \sum_ {x \in B} \sum_ {l = 1} ^ {L} \mathbb {E} _ {\left(\mathbf {e} _ {i, j}, \mathbf {e} _ {p, h}\right) \sim G _ {x} ^ {l}} [ \left| \left(t _ {u} ^ {l}, t _ {v} ^ {l}\right) ^ {T} - \left(d _ {u} ^ {l}, d _ {v} ^ {l}\right) ^ {T} \right| _ {1} ], \tag {7}
$$

where  $(t_u^l,t_v^l)^T$  and  $(d_u^l,d_v^l)^T$  are, respectively, the target and the prediction offsets computed at block  $l$  using the randomly sampled pair  $(\mathbf{e}_{i,j},\mathbf{e}_{p,h})\in G_x^l$ . For each block, we use a block-specific MLP  $f^{l}$  to compute  $(d_u^l,d_v^l)^T$ . Note that, using Eq. (7), the initial layers of VT receive more "signal", because each block  $l$  accumulates the gradients produced by all the blocks  $l'\geq l$ .

All the proposed variants but the last  $(\mathcal{L}_{drloc}^{all})$  are very computationally efficient, because they involve only one forward and one backward pass per image, and  $m$  forward passes through  $f$ . In Sec. 5.1 we use the final VT accuracy to empirically compare to each other the proposed localization losses.

# 5 Experiments

All the experiments presented in this section are based on image classification tasks, while in the Supplementary Material we also show object detection, instance segmentation and semantic segmentation tasks. We use 11 different datasets: ImageNet-100 (IN-100) [50, 54], which is a subset of 100 classes of ImageNet; CIFAR-10 [31], CIFAR-100 [31], Oxford Flowers102 [40], and SVHN [39], which are four widely used vision datasets; and the six datasets of DomainNet [43], a benchmark commonly used for domain adaptation tasks. We chose the latter because of the large

domain-shift between some of its datasets and ImageNet, which mekes the fine-tuning experiments non-trivial. Tab. 1 (a) shows the number of samples for each of these 11 datasets.

We used, when available, the official VT code (for T2T [59] and Swin [34]) and a publicly available implementation of CvT  $[56]^1$ . In the fine-tuning experiments (Sec. 5.3), we use only T2T and Swin because of the lack of publicly available ImageNet pre-trained CvT networks. For each of the three baselines, we chose a model of comparable size to ResNet-50 (25M parameters): see Tab. 2 (b) for more details. When we plug our loss on one of these baselines, we follow Sec. 4, keeping unchanged the VT architecture apart from our localization MLP  $(f)$ . Moreover, in all the experiments, we train the baselines, both with and without our localization loss, using the same data-augmentation protocol for all the models, and we use the VT-specific hyper-parameter configuration suggested by the authors of each VT. We train each model using 8 Nvidia V100 32GB GPUs.

# 5.1 Ablation study

In Tab. 1 (b) we compare the loss variants presented in Sec. 4.2. For these experiments, we use IN-100, we train all the models for 100 epochs, and we show the top-1 classification accuracy on the test set. For all the variants, the baseline model is Swin [34] (row (A) of Tab. 1 (b)).

When we plug  $\mathcal{L}_{drloc}$  on top of Swin (Sec. 4), the final accuracy increases by 1.26 points (B). All the other dense localization loss variants underperform  $\mathcal{L}_{drloc}$  (C-F). A bit surprisingly, the very-dense localization loss  $\mathcal{L}_{drloc}^{all}$  is significantly outperformed by the much simpler (and computationally more efficient)  $\mathcal{L}_{drloc}$ . Moreover,  $\mathcal{L}_{drloc}^{all}$  is the only variant which underperforms the baseline. We presume that this is due to the fact that most of the Swin intermediate blocks have resolution grids  $G_{x}^{l}$  finer than the last grid  $G_{x}^{L}$  ( $l < L$ ,  $k_{l} > k_{L}$ , Sec. 4.2), and this makes the localization task harder, slowing down the convergence of  $f^{l}$ , and likely providing noisy gradients to the VT (see Sec. 4.1). In the rest of this paper and in all the other experiments, we always use  $\mathcal{L}_{drloc}$  as the relative localization loss.

Finally, we analyze the impact of different values of  $m$  (the total number of embedding pairs used per image, see Sec. 4). Since we use the same grid resolution for all the VTs (i.e.,  $7 \times 7$ , Sec. 3), also the number of embeddings per image is the same for all the VTs ( $k^2 = 49$ ). Hence, following the results of Tab. 2 (a), obtained with CIFAR-100 and Swin, we use  $m = 64$  for all the VTs and all the datasets.

Table 1: (a) The size of the datasets used in our empirical analysis. (b) IN-100, 100 epoch training: a comparison between the different variants of our proposed loss.

(a)  

<table><tr><td colspan="2">Dataset</td><td>Train size</td><td>Test size</td><td>Classes</td></tr><tr><td colspan="2">ImageNet-100 [50]</td><td>126,689</td><td>5,000</td><td>100</td></tr><tr><td colspan="2">CIFAR-10 [32]</td><td>50,000</td><td>10,000</td><td>10</td></tr><tr><td colspan="2">CIFAR-100 [32]</td><td>50,000</td><td>10,000</td><td>100</td></tr><tr><td colspan="2">Oxford Flowers102 [40]</td><td>2,040</td><td>6,149</td><td>102</td></tr><tr><td colspan="2">SVHN [39]</td><td>73,257</td><td>26,032</td><td>10</td></tr><tr><td rowspan="6">DomainNet [43]</td><td>ClipArt</td><td>33,525</td><td>14,604</td><td></td></tr><tr><td>Infograph</td><td>36,023</td><td>15,582</td><td></td></tr><tr><td>Painting</td><td>50,416</td><td>21,850</td><td rowspan="2">345</td></tr><tr><td>Quickdraw</td><td>120,750</td><td>51,750</td></tr><tr><td>Real</td><td>120,906</td><td>52,041</td><td></td></tr><tr><td>Sketch</td><td>48,212</td><td>20,916</td><td></td></tr></table>

(b)  

<table><tr><td></td><td>Model</td><td>Top-1 Acc.</td></tr><tr><td>A:</td><td>Swin-T [34]</td><td>82.76</td></tr><tr><td>B:</td><td>A + Ldrloc</td><td>84.02 (+1.26)</td></tr><tr><td>C:</td><td>A + Ldrloc*</td><td>83.14 (+0.38)</td></tr><tr><td>D:</td><td>A + Ldrloc</td><td>83.86 (+1.10)</td></tr><tr><td>E:</td><td>A + Lreg</td><td>83.24 (+0.48)</td></tr><tr><td>F:</td><td>A + Lall</td><td>81.88 (-0.88)</td></tr></table>

# 5.2 Training from scratch

In this section, we analyze the performance of both the VT baselines and our regularization loss using small-medium size datasets and different number of training epochs. The goal is to simulate a scenario with limited computational resources and/or limited training data.

We start by analyzing the impact of the number of training epochs on IN-100. Tab. 2 (b) shows that, using  $\mathcal{L}_{drloc}$ , all the tested VTs show an accuracy improvement, and this boost is larger with fewer

Table 2: (a) CIFAR-100, 100 training epochs: an analysis of the impact of the number of pair samples  $(m)$  in  $L_{drloc}$ . (b) Accuracy results on IN-100 with different number of training epochs.  
(a)  

<table><tr><td>Model</td><td>Top-1 Acc.</td></tr><tr><td>A: Swin-T [34]</td><td>53.28</td></tr><tr><td>B: A + Ldrloc, m=32</td><td>63.70</td></tr><tr><td>C: A + Ldrloc, m=64</td><td>66.23</td></tr><tr><td>D: A + Ldrloc, m=128</td><td>65.16</td></tr><tr><td>E: A + Ldrloc, m=256</td><td>64.87</td></tr></table>

(b)  

<table><tr><td></td><td>Model</td><td>#Param</td><td colspan="2">Top-1 Acc.</td></tr><tr><td></td><td></td><td>(M)</td><td>100 epochs</td><td>300 epochs</td></tr><tr><td rowspan="2">CvT</td><td>CvT-13</td><td>20</td><td>85.62</td><td>90.16</td></tr><tr><td>CvT-13+Ldrloc</td><td>20</td><td>85.98 (+0.36)</td><td>90.28 (+0.12)</td></tr><tr><td rowspan="2">Swin</td><td>Swin-T</td><td>29</td><td>82.76</td><td>89.68</td></tr><tr><td>Swin-T+Ldrloc</td><td>29</td><td>84.02 (+1.26)</td><td>90.32 (+0.64)</td></tr><tr><td rowspan="2">T2T</td><td>T2T-ViT-14</td><td>22</td><td>82.74</td><td>87.76</td></tr><tr><td>T2T-ViT-14+Ldrloc</td><td>22</td><td>83.90 (+1.16)</td><td>88.16 (+0.4)</td></tr></table>

epochs. As expected, our loss acts as a regularizer, whose effects are more pronounced in a shorter training regime. We believe this result is particularly significant considering the larger computational times which are necessary to train typical VTs with respect to ResNets.

In Tab. 3, we use all the other datasets and we train from scratch with 100 epochs. First, we note that the accuracy of the VT baselines varies a lot depending on the dataset (which is expected), but also depending on the specific VT architecture. As a reference, when these VTs are trained on ImageNet-1K (for 300 epochs), the differences of their respective top-1 accuracy is much smaller: Swin-T, 81.3 [34]; T2T-ViT-14, 81.5 [59]; CvT-13, 81.6 [56]. Conversely, Tab. 3 shows that, for instance, the accuracy difference between CvT and Swin is about 45-46 points in Quickdraw and Sketch, 30 points on CIFAR-10, and about 20 points on many other datasets. Analogously, the difference between CvT and T2T is between 20 and 25 points in Sketch, Painting, Flowers102, and quite significant in the other datasets. This comparison shows that CvT is usually much more robust in a small training-set regime with respect to the other two VTs. We believe that these results are interesting especially for those tasks in which fine-tuning a model pre-trained on large datasets is not possible. This is the case, for instance, when there is a large domain-shift with respect to the application dataset (e.g., medical images, etc.) or when the VT architecture should be drastically modified and adapted to the specific task (e.g., processing 3D data, etc.). In these scenarios, choosing an architecture which can quickly learn from small-medium size datasets may be crucial.

In Tab. 3, we also show the accuracy of these three VTs when training is done using  $\mathcal{L}_{drloc}$  as a regularizer. Similarly to the IN-100 results, also in this case our loss improves the accuracy of all the tested VTs in all the datasets. Most of the time, this improvement is quite significant (e.g., almost 4 points on SVHN with CvT), and sometimes dramatic (e.g., more than 45 points on Quickdraw with Swin). These results show that a self-supervised side task can provide a significant "signal" to the VT when the training set is limited, and, specifically, that our loss can be very effective in boosting the accuracy of a VT trained from scratch in this scenario.

Table 3: Top-1 accuracy of the VTs, trained from scratch on different datasets (100 epochs).  

<table><tr><td colspan="2"></td><td>CIFAR-10</td><td>CIFAR-100</td><td>Flowers102</td><td>SVHN</td><td>ClipArt</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch</td></tr><tr><td rowspan="2">CvT</td><td>CvT-13</td><td>89.02</td><td>73.50</td><td>54.29</td><td>91.47</td><td>60.34</td><td>19.39</td><td>54.79</td><td>70.10</td><td>76.33</td><td>56.98</td></tr><tr><td>CvT-13+Ldrloc</td><td>90.30(+1.28)</td><td>74.51(+1.01)</td><td>56.29(+2.00)</td><td>95.36(+3.89)</td><td>60.64(+0.30)</td><td>20.05(+0.67)</td><td>55.26(+0.47)</td><td>70.36(+0.26)</td><td>77.05(+0.68)</td><td>57.56(+0.58)</td></tr><tr><td rowspan="2">Swin</td><td>Swin-T</td><td>59.47</td><td>53.28</td><td>34.51</td><td>71.60</td><td>38.05</td><td>8.20</td><td>35.92</td><td>24.08</td><td>73.47</td><td>11.97</td></tr><tr><td>Swin-T+Ldrloc</td><td>83.89(+24.42)</td><td>66.23(+12.95)</td><td>39.37(+4.86)</td><td>94.23(+22.63)</td><td>47.47(+9.42)</td><td>10.16(+1.96)</td><td>41.86(+5.94)</td><td>69.41(+45.33)</td><td>75.59(+2.12)</td><td>38.55(+26.58)</td></tr><tr><td rowspan="2">T2T</td><td>T2T-ViT-14</td><td>84.19</td><td>65.16</td><td>31.73</td><td>95.36</td><td>43.55</td><td>6.89</td><td>34.24</td><td>69.83</td><td>73.93</td><td>31.51</td></tr><tr><td>T2T-ViT-14+Ldrloc</td><td>87.56(+3.37)</td><td>68.03(+2.87)</td><td>34.35(+2.62)</td><td>96.49(+1.13)</td><td>52.36(+8.81)</td><td>9.51(+2.62)</td><td>42.78(+8.54)</td><td>70.16(+0.33)</td><td>74.63(+0.70)</td><td>51.95(+20.44)</td></tr></table>

Table 4: VTs pre-trained on ImageNet-1K and then fine-tuned (top-1 accuracy, 100 epochs).  

<table><tr><td colspan="2"></td><td>CIFAR-10</td><td>CIFAR-100</td><td>Flowers102</td><td>SVHN</td><td>ClipArt</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch</td></tr><tr><td rowspan="2">Swin</td><td>Swin-T</td><td>97.95</td><td>88.22</td><td>98.03</td><td>96.10</td><td>73.51</td><td>41.07</td><td>72.99</td><td>75.81</td><td>85.48</td><td>72.37</td></tr><tr><td>Swin-T+Ldrloc</td><td>98.37(+0.42)</td><td>88.40(+0.18)</td><td>98.21(+0.18)</td><td>97.87(+1.77)</td><td>79.51(+6.00)</td><td>46.10(+5.03)</td><td>73.28(+0.29)</td><td>76.01(+0.20)</td><td>85.61(+0.13)</td><td>72.86(+0.49)</td></tr><tr><td rowspan="2">T2T</td><td>T2T-ViT-14</td><td>98.37</td><td>87.33</td><td>97.98</td><td>97.03</td><td>74.59</td><td>38.53</td><td>72.29</td><td>74.16</td><td>84.56</td><td>72.18</td></tr><tr><td>T2T-ViT-14+Ldrloc</td><td>98.52(+0.15)</td><td>87.65(+0.32)</td><td>98.08(+0.20)</td><td>98.20(+1.17)</td><td>78.22(+0.37)</td><td>45.69(+7.16)</td><td>72.42(+0.13)</td><td>74.27(+0.11)</td><td>84.57(+0.01)</td><td>72.29(+0.11)</td></tr></table>

# 5.3 Fine-tuning

In this section, we analyze a typical fine-tuning scenario, in which a model is pre-trained on a big dataset (e.g., ImageNet), and then fine-tuned on the target domain. Specifically, in all the experiments, we use VT models pre-trained by the corresponding VT authors on ImageNet-1K without our localization loss. The difference between the baselines and ours concerns only the fine-tuning stage, which is done in the standard way for the former and using our  $\mathcal{L}_{drloc}$  regularizer for the latter. Starting from standard pre-trained models and using our loss only in the fine-tuning stage, emphasises the easy to use of our proposal in practical scenarios, in which fine-tuning can be done without re-training the model on ImageNet. As mentioned in Sec. 5, in this analysis we do not include CvT because of the lack of publicly available ImageNet-1K pre-trained models for this architecture.

The results are presented in Tab. 4. Differently from the results shown in Sec. 5.2, the accuracy difference between T2T and Swin is much less pronounced, and the latter outperforms the former in most of the datasets. Moreover, analogously to all the other experiments, also in this case, using  $\mathcal{L}_{drloc}$  leads to an accuracy improvement with all the tested VTs and in all the datasets. For instance, on Infograph, Swin with  $\mathcal{L}_{drloc}$  improves of more than 5 points, and T2T more than 7 points.

# 6 Conclusion

In this paper, we have empirically analyzed different VTs, showing that their performance largely varies when trained from scratch with small-medium size datasets, and that CvT is usually much more effective in generalizing with less data. Moreover, we proposed a self-supervised side-task to regularize VT training. Our localization task, inspired by [11], is densely defined for a random subset of final-token embedding pairs, and it encourages the VT to learn spatial information.

In our extensive empirical analysis, with 11 datasets, different training scenarios and three VTs, our dense localization loss has always improved the corresponding baseline accuracy, usually by a significant margin, and sometimes dramatically (up to +45 points). We believe that this shows that our proposal is an easy-to-reproduce, yet very effective tool to boost the performance of VTs, especially in training regimes with a limited amount of data/training time. It also paves the way to investigating other forms of self-supervised/multi-task learning which are specific for VTs, and can help VT training without resorting to the use of huge annotated datasets.

Limitations. A deeper analysis on why fine-grained embedding grids have a negative impact on our localization loss (Sec. 4.1 and 5.1) was left as a future work. Moreover, in our analysis, we focused on VT models of approximately the same size as a ResNet-50. We have not considered bigger Swin/T2T/CvT models because training and fine-tuning very large networks on 11 datasets is too computationally demanding. For the same reason, we have not tested ViT. However, since the goal of this paper is investigating the VT behaviour with medium-small size datasets, most likely these big models are not the best choice in a training scenario with scarcity of data, as witnessed by the fact that ViT underperforms similar-capacity CNNs when trained on ImageNet-1K [16].

# References

[1] Yuki Markus Asano, Christian Rupprecht, and Andrea Vedaldi. Self-labelling via simultaneous clustering and representation learning. In ICLR, 2020.  
[2] Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. arXiv:2105.04906, 2021.  
[3] Miguel A Bautista, Artsiom Sanakoyeu, Ekaterina Tikhoncheva, and Bjorn Ommer. CliqueCNN: deep unsupervised exemplar learning. In NeurIPS, 2016.  
[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. arXiv:2005.12872, 2020.  
[5] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In ECCV, 2018.  
[6] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, 2020.  
[7] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020.  
[8] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.  
[9] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. arXiv:2011.10566, 2020.  
[10] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. arXiv:2104.02057, 2021.  
[11] Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. Electra: Pretraining text encoders as discriminators rather than generators. In ICLR, 2020.  
[12] Michael Crawshaw. Multi-task learning with deep neural networks: A survey. arXiv:2009.09796, 2020.  
[13] Zhigang Dai, Bolun Cai, Yugeng Lin, and Junying Chen. UP-DETR: unsupervised pre-training for object detection with transformers. arXiv:2011.09094, 2020.  
[14] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL., 2019.  
[15] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015.  
[16] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.  
[17] Debidatta Dwibedi, Yusuf Aytar, Jonathan Thompson, Pierre Sermanet, and Andrew Zisserman. Temporal cycle-consistency learning. In CVPR, 2019.  
[18] Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. With a little help from my friends: Nearest-neighbor contrastive learning of visual representations. arXiv:2104.14548, 2021.  
[19] Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto, and Nicu Sebe. Whitening for self-supervised representation learning. In ICML, 2021.  
[20] Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, and Luc Van Gool. SCAN: learning to classify images without labels. In ECCV, 2020.

[21] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018.  
[22] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. arXiv:2006.07733, 2020.  
[23] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. CVPR, 2020.  
[24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.  
[25] R. Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Philip Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. In ICLR, 2019.  
[26] Han Hu, Zheng Zhang, Zhenda Xie, and Stephen Lin. Local relation networks for image recognition. In ICCV, 2019.  
[27] Tianyu Hua, Wenxiao Wang, Zihui Xue, Yue Wang, Sucheng Ren, and Hang Zhao. On feature decorrelation in self-supervised learning. arXiv:2105.00470, 2021.  
[28] Drew A. Hudson and C. Lawrence Zitnick. Generative Adversarial Transformers. arXiv:2103.01209, 2021.  
[29] Xu Ji, João F. Henriques, and Andrea Vedaldi. Invariant information clustering for unsupervised image classification and segmentation. In ICCV, 2019.  
[30] Yifan Jiang, Shiyu Chang, and Zhangyang Wang. TransGAN: Two transformers can make one strong GAN. arXiv:2102.07074, 2021.  
[31] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
[32] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
[33] Yawei Li, Kai Zhang, Jiezhang Cao, Radu Timofte, and Luc Van Gool. LocalViT: Bringing locality to vision transformers. arXiv:2104.05707, 2021.  
[34] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. arXiv:2103.14030, 2021.  
[35] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. Track-Former: Multi-object tracking with transformers. arXiv:2101.02702, 2021.  
[36] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv:1301.3781, 2013.  
[37] Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. Distributed representations of words and phrases and their compositionality. In MIPS, 2013.  
[38] Ishan Misra and Laurens van der Maaten. Self-supervised learning of pretext-invariant representations. In CVPR, 2020.  
[39] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. In NeurIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.  
[40] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics & Image Processing, 2008.

[41] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV, 2016.  
[42] Mehdi Noroozi, Hamed Pirsiavash, and Paolo Favaro. Representation learning by learning to count. In ICCV, 2017.  
[43] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In CVPR, pages 1406-1415, 2019.  
[44] Alec Radford and Karthik Narasimhan. Improving language understanding by generative pre-training. 2018.  
[45] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21:140:1-140:67, 2020.  
[46] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211-252, 2015.  
[47] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. In NAACL-HLT, 2018.  
[48] Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia Schmid. Segmenter: Transformer for semantic segmentation. arXiv:2105.05633, 2021.  
[49] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. VideoBERT: A joint model for video and language representation learning. In ICCV, 2019.  
[50] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In ECCV, 2020.  
[51] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. arXiv:2012.12877, 2020.  
[52] Aäron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv:1807.03748, 2018.  
[53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017.  
[54] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In ICML, 2020.  
[55] Xiaolong Wang, Ross B. Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In CVPR, 2018.  
[56] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. CvT: Introducing convolutions to vision transformers. arXiv:2103.15808, 2021.  
[57] Weijian Xu, Yifan Xu, Tyler Chang, and Zhuowen Tu. Co-scale conv-attentional image transformers. arXiv:2104.06399, 2021.  
[58] Kun Yuan, Shaopeng Guo, Ziwei Liu, Aojun Zhou, Fengwei Yu, and Wei Wu. Incorporating convolution designs into visual transformers. arXiv:2103.11816, 2021.  
[59] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zihang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token ViT: Training vision transformers from scratch on ImageNet. arXiv:2101.11986, 2021.  
[60] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised learning via redundancy reduction. arXiv:2103.03230, 2021.

[61] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: Deformable transformers for end-to-end object detection. In ICLR, 2021.  
[62] Chengxu Zhuang, Alex Lin Zhai, and Daniel Yamins. Local aggregation for unsupervised learning of visual embeddings. In ICCV, 2019.
