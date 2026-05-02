# ADAPTIVE FEATURE ABSTRACTION FOR TRANSLATING VIDEO TO LANGUAGE

Yunchen Pu *

Department of Electrical and Computer Engineering  
Duke University

yunchen.pu@duke.edu

Martin Renqiang Min

Machine Learning Group

NEC Laboratories America

renqiang@nec-labs.com

Zhe Gan

Department of Electrical and Computer Engineering

Duke University

zhe.gan@duke.edu

Lawrence Carin

Department of Electrical and Computer Engineering

Duke University

lcarin@duke.edu

# ABSTRACT

Previous models for video captioning often use the output from a specific layer of a Convolutional Neural Network (CNN) as video representations, preventing them from modeling rich, varying context-dependent semantics in video descriptions. In this paper, we propose a new approach to generating adaptive spatiotemporal representations of videos for a captioning task. For this purpose, novel attention mechanisms with spatiotemporal alignment is employed to adaptively and sequentially focus on different layers of CNN features (levels of feature "abstraction"), as well as local spatiotemporal regions of the feature maps at each layer. Our approach is evaluated on three benchmark datasets: YouTube2Text, M-VAD and MSR-VTT. Along with visualizing the results and how the model works, these experiments quantitatively demonstrate the effectiveness of the proposed adaptive spatiotemporal feature abstraction for translating videos to sentences with rich semantics.

# 1 INTRODUCTION

Videos represent among the most widely used forms of data, and their accurate characterization poses an important challenge for computer vision and machine learning. Generating a natural-language description of a video, termed video captioning, is an important component of video analysis. Inspired by the successful encoder-decoder framework used in machine translation (Cho et al., 2014; Bahdanau et al., 2015; Sutskever et al., 2014) and image caption generation (Kiros et al., 2014; Vinyals et al., 2015; Karpathy & Li, 2015; Mao et al., 2015), most recent work on video captioning (Donahue et al., 2015; Venugopalan et al., 2015a;b; Yao et al., 2015; Pan et al., 2016b; Yu et al., 2016) employs a two-dimensional (2D) or three-dimensional (3D) Convolutional Neural Network (CNN) as an encoder, mapping an input video to a compact feature-vector representation. A Recurrent Neural Network (RNN) is typically employed as a decoder, unrolling the feature vector to generate a sequence of words of arbitrary length.

Despite achieving encouraging success in video captioning, previous models suffer important limitations. Often the rich video content is mapped to a single feature vector for caption generation; this approach is prone to miss detailed and localized spatiotemporal information. To mitigate this, one may employ methods to focus attention on local regions of the feature map, but typically this

is done with features from a selected (usually top) CNN layer. By employing features from a fixed CNN layer, the algorithm is limited in its ability to model rich, context-aware semantics that requires focusing on different feature abstraction levels. As investigated in Mahendran & Vedaldi (2015); Zeiler & Fergus (2014), the feature characteristics/abstraction is correlated with the CNN layer: features from layers at or near the top of a CNN tend to focus on global (extended) visual percepts, while features from lower CNN layers provide more local, fine-grained information. It is desirable to select/weight features from different CNN layers adaptively when decoding a caption, selecting different levels of feature abstraction by sequentially emphasizing features from different CNN layers. In addition to focusing on features from different CNN layers, it is also desirable to emphasize local spatiotemporal regions in feature maps at particular layers.

To realize these desiderata, our proposed decoding process for generating a sequence of words dynamically emphasizes different levels (CNN layers) of 3D convolutional features, to model important coarse or fine-grained spatiotemporal structure. Additionally, the model adaptively attends to different locations within the feature maps at particular layers. While some previous models use 2D CNN features to generate video representations, our model adopts features from a deep 3D convolutional neural network (C3D). Such features have been shown to be effective for video representations, action recognition and scene understanding (Tran et al., 2015), by learning the spatiotemporal features that can provide better appearance and motion information. In addition, the proposed model is inspired by the recent success of attention-based models that mimic human perception (Mnih et al., 2014; Xu et al., 2015).

Our proposed model, adaptive spatiotemporal feature representation with dynamic feature abstraction, involves comparing and evaluating different levels of 3D convolutional feature maps. However, this has three challenges to overcome to directly compare features between layers: (i) the features from different C3D levels have distinct dimensions, undermining the use of a multi-layer perceptron (MLP) (Xu et al., 2015; Bahdanau et al., 2015) based attention model; (ii) the features represented in each layer are not spatiotemporally aligned, undermining our ability to quantify the value of features at a specific spatiotemporal location, based on information from all CNN layers; and (iii) the semantic meaning of feature vectors from the convolutional filters of C3D varies across layers, implying that the layer-dependent features are in different semantic spaces and making it difficult to feed the features into the same RNN decoder.

To address these issues, one may use either pooling or MLPs to map different levels of features to the same semantic-space dimension. However, these approaches either lose a lot of feature abstraction information or have too many parameters to generalize well. In our approach, we employ convolution operations to elegantly achieve spatiotemporal alignment among C3D features from different levels and attention mechanisms to dynamically select context-dependent feature abstraction information.

The principal contributions of this paper are as follows: (i) A new video-caption-generation model is developed by dynamically modeling context-dependent feature abstractions; (ii) New attention mechanisms to adaptively and sequentially emphasize different levels of feature abstraction (CNN layers), while also imposing attention within local spatiotemporal regions of the feature maps at each layer are employed; (iii) 3D convolutional transformations are introduced to achieve spatiotemporal and semantic feature consistency across different layers; (iv) We demonstrate that the proposed model outperforms other multi-level feature based methods such as hypercolumns (Hariharan et al., 2015) and achieves state-of-the-art performance on several benchmark datasets. We call the proposed algorithm Adaptive SpatioTemporal representation with dynAmic abstReaction (ASTAR).

# 2 RELATED WORK

Early work on video captioning used a two-step approach, employing role-word detection (e.g., subject, verb and object) and rules for language grammar. In such work, the sentence for video description is first split into several parts, each of which is aligned with visual content. For example, Rohrbach et al. (2013; 2014) learn a Conditional Random Field (CRF) to infer high-level concepts such as object and action; in Guadarrama et al. (2013) semantic hierarchies are used to choose an appropriate level of sentence fragments. Statistical language models, learned from large text corpora, are used to translate the semantic representation to a grammatically correct sentence.

Recent work often develops a probabilistic model of the caption, conditioned on a video. Donahue et al. (2015); Venugopalan et al. (2015a,b); Yu et al. (2016); Pan et al. (2016a) applied a 2D CNN pretrained on ImageNet to video frames, with the top-layer output of the CNN used as features. Given the sequence of features extracted from video frames, the video representation is then obtained by a CRF (Donahue et al., 2015), mean pooling (Venugopalan et al., 2015b), weighted mean pooling with attention (Yu et al., 2016), or via the last hidden state of an RNN encoder (Venugopalan et al., 2015a). Yao et al. (2015) replace the 2D CNN features with a 3D CNN to model the short temporal dynamics. These works were followed by Pan et al. (2016b), which jointly embedded the 2D CNN features and spatiotemporal features extracted from a 3D CNN (Tran et al., 2015). However, all of these previous models utilize features extracted from the top layer of the CNN.

There are also some work that combines mult-level features of CNN. Sermanet et al. (2013) employed a combination of intermediate layers with the top layer for pedestrian detection while Hariharan et al. (2015) utilized hypercolumn representation for object segmentation and localization. Our proposed model is mostly related to Ballas et al. (2016), but distinct in important ways. The intermediate convolutional feature maps are leveraged, like Ballas et al. (2016), but an attention model is developed instead of the "stack" RNN in Ballas et al. (2016). In addition, a powerful decoder enhanced with two attention mechanisms is constructed for generating captions, while a simple RNN decoder is employed in Ballas et al. (2016). Finally, we use features extracted from C3D instead of a 2D CNN.

# 3 METHOD

Consider  $N$  training videos, the  $n$ th of which is denoted  $\mathbf{X}^{(n)}$ , with associated caption  $\mathbf{Y}^{(n)}$ . The length- $T_{n}$  caption is represented  $\mathbf{Y}^{(n)} = (\pmb{y}_1^{(n)}, \dots, \pmb{y}_{T_n}^{(n)})$ , with  $\pmb{y}_t^{(n)}$  a 1-of-  $V$  ("one hot") encoding vector, with  $V$  the size of the vocabulary.

For each video, the C3D feature extractor (Tran et al., 2015) produces a set of features  $\mathbf{A}^{(n)} = \{\pmb{a}_1^{(n)},\dots,\pmb{a}_L^{(n)},\pmb{a}_{L + 1}^{(n)}\}$ , where  $\{\pmb{a}_1^{(n)},\dots,\pmb{a}_L^{(n)}\}$  are feature maps extracted from  $L$  convolutional layers, and  $\pmb{a}_{L + 1}^{(n)}$  is a vector obtained from the top fully-connected layer.

The convolutional-layer features,  $\{\pmb{a}_1^{(n)},\dots,\pmb{a}_L^{(n)}\}$ , are extracted by feeding the entire video into C3D, and hence the dimensions of  $\{\pmb{a}_1^{(n)},\dots,\pmb{a}_L^{(n)}\}$  are dependent on the video length (number of frames). As discussed below, we employ spatiotemporal attention at each layer (and between layers), and therefore it is not required that the sizes of  $\{\pmb{a}_1^{(n)},\dots,\pmb{a}_L^{(n)}\}$  be the same for all videos. However, the fully connected layer at the top, responsible for  $\pmb{a}_{L + 1}^{(n)}$ , assumes that  $\pmb{a}_L^{(n)}$  is of the same size for all videos (like the 16-frame-length videos in (Tran et al., 2015)). To account for variable-length videos, and maintain the same fully-connected layer at the top, we employ mean pooling to  $\pmb{a}_L^{(n)}$ , based on a window of length 16 (as in (Tran et al., 2015)) with an overlap of 8 frames. The particular form of pooling used here (one could also use max pooling) is less important than the need to make the dimension of the top-layer features the same for feeding into the final fully-connected layer.

# 3.1 CAPTION MODEL

For notational simplicity, henceforth we omit superscript  $n$ . The  $t$ -th word in a caption,  $\mathbf{y}_t$ , is mapped to an  $M$ -dimensional vector  $\mathbf{w}_t = \mathbf{W}_e\mathbf{y}_t$ , where  $\mathbf{W}_e \in \mathbb{R}^{M \times V}$  is a learned word-embedding matrix, i.e.,  $\mathbf{w}_t$  is a column of  $\mathbf{W}_e$  chosen by the one-hot  $\mathbf{y}_t$ . The probability of caption  $\mathbf{Y} = \{\mathbf{y}_t\}_{t=1,T}$  is defined as

$$
p (\mathbf {Y} | \mathbf {A}) = p (\boldsymbol {y} _ {1} | \mathbf {A}) \prod_ {t = 2} ^ {T} p (\boldsymbol {y} _ {t} | \boldsymbol {y} _ {<   t}, \mathbf {A}). \tag {1}
$$

Specifically, the first word  $\mathbf{y}_1$  is drawn from  $p(\mathbf{y}_1|\mathbf{A}) = \mathrm{softmax}(\mathbf{V}\mathbf{h}_1)$ , where  $\mathbf{h}_1 = \tanh(\mathbf{C}\mathbf{a}_{L+1})$ . Bias terms are omitted for simplicity throughout the paper. All the other words in the caption are then sequentially generated using an RNN, until the end-sentence symbol is generated. Conditional distribution  $p(\mathbf{y}_t|\mathbf{y}_{<t},\mathbf{A})$  is specified as  $\mathrm{softmax}(\mathbf{V}\mathbf{h}_t)$ , where  $\mathbf{h}_t$  is recursively updated as  $\mathbf{h}_t = \mathcal{H}(\mathbf{w}_{t-1},\mathbf{h}_{t-1},\mathbf{z}_t)$ .  $\mathbf{V}$  is a matrix connecting the RNN hidden state to a softmax, for computing a distribution over words.  $\mathbf{z}_t = \phi(\mathbf{h}_{t-1},\mathbf{a}_1,\dots,\mathbf{a}_L)$  is the context vector used in

![](images/b3c40e5ffe7424044ac362d68a1ba9c2ac8081d89a8dcfe4423169439cb0ca4e.jpg)

![](images/00698028053bb153c300cf102ff2d4d270f222a551f55c236c494dfd5f775b90.jpg)  
Figure 1: Illustration of our proposed caption-generation model. The model leverages a fully-connected map from the top layer as well as convolutional maps from different mid-level layers of a pretrained 3D convolutional neural network (C3D). The context vector  $\mathbf{z}_t$  is generated from previous hidden unit  $\mathbf{h}_{t-1}$  and the convolutional maps  $\{\mathbf{a}_1, \dots, \mathbf{a}_L\}$  (the red frame) which is detailed in Fig.2  
Figure 2: Illustration of our attention mechanism. The video features are extracted by C3D, followed by a 3D convolutional transformation. At each time step, the spatiotemporal attention takes these features and previous hidden units to generate  $L$  feature vectors which are fed to the abstraction attention, manifesting the context vector.

the attention mechanism, capturing the relevant visual features associated with the spatiotemporal attention (also weighting level of feature abstraction), as detailed in Sec. 3.2.

Note that the output of the fully-connected-layer,  $\pmb{a}_{L + 1}$ , is only used to generate the first word (encapsulating overall-video features). We found that only using  $\pmb{a}_{L + 1}$  there works better in practice than using it at each time step of the RNN, as also found in Vinyals et al. (2015); Venugopalan et al. (2015b).

The transition function  $\mathcal{H}(\cdot)$  is implemented with Long Short-Term Memory (LSTM) (Hochreiter & Schmidhuber, 1997). At time  $t$ , the LSTM unit consists of a memory cell  $c_{t}$  and three gates: the input  $i_{t}$ , forget  $f_{t}$ , and output  $o_{t}$  gates. The memory cell transmits the information from the previous step to the current step, while the gates control reading or writing the memory unit through sigmoid functions. Specifically, the hidden units  $h_t$  are updated as follows:

$$
\boldsymbol {i} _ {t} = \sigma \left(\mathbf {W} _ {i w} \boldsymbol {w} _ {t - 1} + \mathbf {W} _ {i h} \boldsymbol {h} _ {t - 1} + \mathbf {W} _ {i z} \boldsymbol {z} _ {t}\right), \quad \boldsymbol {f} _ {t} = \sigma \left(\mathbf {W} _ {f w} \boldsymbol {w} _ {t - 1} + \mathbf {W} _ {f h} \boldsymbol {h} _ {t - 1} + \mathbf {W} _ {f z} \boldsymbol {z} _ {t}\right),
$$

$$
\boldsymbol {o} _ {t} = \sigma (\mathbf {W} _ {o w} \boldsymbol {w} _ {t - 1} + \mathbf {W} _ {o h} \boldsymbol {h} _ {t - 1} + \mathbf {W} _ {o z} \boldsymbol {z} _ {t}), \quad \tilde {\boldsymbol {c}} _ {t} = \tanh  (\mathbf {W} _ {c w} \boldsymbol {w} _ {t - 1} + \mathbf {W} _ {c h} \boldsymbol {h} _ {t - 1} + \mathbf {W} _ {c z} \boldsymbol {z} _ {t}),
$$

$$
\boldsymbol {c} _ {t} = \boldsymbol {f} _ {t} \odot \boldsymbol {c} _ {t - 1} + \boldsymbol {i} _ {t} \odot \tilde {\boldsymbol {c}} _ {t}, \quad \boldsymbol {h} _ {t} = \boldsymbol {o} _ {t} \odot \tanh  (\boldsymbol {c} _ {t}), \tag {2}
$$

where  $\sigma (\cdot)$  and  $\odot$  denote the sigmoid function and the element-wise multiply operator, respectively. Matrices  $\mathbf{W}_{\{i,f,o,c\}}$ ,  $\mathbf{V}$  and  $\mathbf{C}$  represent the set of LSTM parameters that will be learned (plus associated biases).

Given the video  $\mathbf{X}$  (with features  $\mathbf{A}$ ) and associated caption  $\mathbf{Y}$ , the objective function is the sum of the log-likelihood of the caption conditioned on the video representation:

$$
\log p (\mathbf {Y} | \mathbf {A}) = \log p (\boldsymbol {y} _ {1} | \mathbf {A}) + \sum_ {t = 2} ^ {T} \log p (\boldsymbol {y} _ {t} | \boldsymbol {y} _ {<   t}, \mathbf {A}), \tag {3}
$$

Equation (3) is a function of all model parameters to be learned; they are not explicitly depicted in (3) for notational simplicity. Further, (3) corresponds to a single video-caption pair, and when training we sum over all such training pairs. Our proposed model is illustrated in Figure 1.

# 3.2 ATTENTION MECHANISM

We first define notation needed to describe attention mechanism  $\phi(h_{t-1}, a_1, \ldots, a_L)$ . Each feature map,  $a_l$ , is a 4D tensor, with elements corresponding to two spatial coordinates (i.e., vertical and horizontal dimensions in a given frame), third tensor index in the frame-index dimension, and a fourth dimension associated with the filter index (for the convolutional filters). To be explicit, at CNN layer  $l$ , the number of dimensions of this tensor are denoted  $n_x^l \times n_y^l \times n_z^l \times n_F^l$ , with respective dimensions corresponding to vertical, horizontal, frame, and filter (e.g.,  $n_F^l$  convolutional filters at layer  $l$ ). Note that dimensions  $n_x^l$ ,  $n_y^l$ , and  $n_z^l$  vary with layer level  $l$  (getting smaller with increasing  $l$ , due to pooling).

We define  $\pmb{a}_{i,l}$  as a  $n_F^l$ -dimensional vector, corresponding to a fixed coordinate in three of the tensor dimensions, i.e.,  $i \in [1,\dots ,n_x^l ] \times [1,\dots ,n_y^l ] \times [1,\dots ,n_z^l ]$ , while sweeping across all  $n_F^l$  feature/filter dimensions. Further, define  $\pmb{a}_l(k)$  as a 3D tensor, associated with 4D tensor  $\pmb{a}_l$ . Specifically,  $\pmb{a}_l(k)$  corresponds to the 3D tensor manifested from  $\pmb{a}_l$ , with  $k \in \{1,\dots ,n_F^l\}$  a fixed coordinate in the dimension of the filter index. Hence,  $\pmb{a}_l(k)$  corresponds to the 3D feature map at layer  $l$ , due to the  $k$ th filter at that layer.

We introduce two attention mechanisms when predicting word  $\mathbf{y}_t$ : (i) spatiotemporal-localization attention, and (ii) abstraction-level attention; these, respectively, measure the relative importance of a particular spatiotemporal location and a particular CNN layer (feature abstraction) for producing  $\mathbf{y}_t$ , based on the word-history information  $\mathbf{y}_{< t}$ .

To achieve this, we seek to map  $\pmb{a}_l \rightarrow \hat{\pmb{a}}_l$ , where 4D tensors  $\hat{\pmb{a}}_l$  all have the same dimensions, are embedded into same semantic spaces, and are aligned spatialtemporally. Specifically,  $\hat{\pmb{a}}_l$ ,  $l = 1, \dots, L - 1$  are aligned in the above ways with  $\pmb{a}_L$ . To achieve this, we filter each  $\pmb{a}_l$ ,  $l = 1, \dots, L - 1$ , and then apply max-pooling; the filters seek semantic alignment of the features (including feature dimension), and the pooling is used to spatiotemporally align the features with  $\pmb{a}_L$ . Specifically, consider

$$
\hat {\boldsymbol {a}} _ {l} = f \left(\sum_ {k = 1} ^ {n _ {F} ^ {l}} \boldsymbol {a} _ {l} (k) * \mathbf {U} _ {k, l}\right), \tag {4}
$$

for  $l = 1,\dots ,L - 1$ , and with  $\hat{\pmb{a}}_L = \pmb{a}_L$ . As discussed above,  $\pmb{a}_{l}(k)$  is the 3D feature map (tensor) for dictionary  $k\in \{1,\ldots ,n_F^l\}$  at layer  $l$ , and  $\mathbf{U}_{k,l}$  is a 4D tensor. The convolution  $*$  in (4) operates in the three shift dimensions, and  $\pmb{a}_{l}(k)*\mathbf{U}_{k,l}$  manifests a 4D tensor. Specifically, each of the  $n_F^L$  3D "slices" of  $\mathbf{U}_{k,l}$  are spatiotemporally convolved (3D convolution) with  $\pmb{a}_{k,l}$ , and after summing over the  $n_F^l$  convolutional filters, followed by  $f(\cdot)$ , this manifests each of the  $n_F^L$  3D slices of  $\hat{\pmb{a}}_l$ . Function  $f(\cdot)$  is an element-wise nonlinear activation function, followed by max pooling, with the pooling dimensions meant to realize final dimensions consistent with  $\pmb{a}_L$ , i.e., dimension  $n_x^L\times n_y^L\times n_z^L\times n_F^L$ . Consequently,  $\hat{\pmb{a}}_{i,l}\in \mathbb{R}^{n_F^L}$  is a feature vector with  $i\in [1,\dots ,n_x^L ]\times [1,\dots ,n_y^L ]\times [1,\dots ,n_z^L ]$ . Note that it may only require one 3D tensor  $\mathbf{U}_l$  applied on each 3D slices  $\pmb{a}_{l}(k)$  for each layer to achieve spatiotemporal alignment of the layer-dependent features. However, the features from two distinct layers will not be in the same "semantic" space, making it difficult to assess the value of the layer-dependent features. The multiple tensors in set  $\{\mathbf{U}_{k,l}\}$  provide the desired semantic alignment between layers, allowing analysis of the value of features from different layers via a single MLP, in the following (5) and (6).

With  $\{\hat{a}_l\}_{l = 1,L}$  semantically and spatiotemporally aligned, we now seek to jointly quantify the value of a particular spatiotemporal region and a particular feature layer ("abstraction") for prediction of the next word. For each  $\hat{a}_{i,l}$ , the attention mechanism generates two positive weights,  $\alpha_{ti}$  and  $\beta_{tl}$ , which measure the relative importance of location  $i$  and layer  $l$  for producing  $\mathbf{y}_t$  based  $\mathbf{y}_{< t}$ . Attention weights  $\alpha_{ti}$  and  $\beta_{tl}$  and context vector  $\mathbf{z}_t$  are computed as

$$
e _ {t i} = \boldsymbol {w} _ {\alpha} ^ {T} \tanh  \left(\mathbf {W} _ {a \alpha} \hat {\boldsymbol {a}} _ {i} + \mathbf {W} _ {h \alpha} \boldsymbol {h} _ {t - 1}\right), \quad \alpha_ {t i} = \operatorname {s o f t m a x} \left(\left\{e _ {t i} \right\}\right), \quad \boldsymbol {s} _ {t} = \psi \left(\left\{\hat {\boldsymbol {a}} _ {i} \right\}, \left\{\alpha_ {t i} \right\}\right), \tag {5}
$$

$$
b _ {t l} = \boldsymbol {w} _ {\beta} ^ {T} \operatorname {t a n h} \left(\mathbf {W} _ {s \beta} \boldsymbol {s} _ {t l} + \mathbf {W} _ {h \beta} \boldsymbol {h} _ {t - 1}\right), \quad \beta_ {t l} = \operatorname {s o f t m a x} \left(\left\{b _ {t l} \right\}\right), \quad \boldsymbol {z} _ {t} = \psi \left(\left\{\boldsymbol {s} _ {t l} \right\}, \left\{\beta_ {t l} \right\}\right), \tag {6}
$$

where  $\hat{\pmb{a}}_i$  is a vector composed by stacking  $\{\hat{a}_{i,l}\}_{l = 1,L}$  (all features at position  $i$ ).  $e_{ti}$  and  $b_{tl}$  are scalars reflecting the importance of spatiotemporal region  $i$  and layer  $t$  to predicting  $\mathbf{y}_t$ , while  $\alpha_{ti}$  and  $\beta_{tl}$  are relative weights of this importance, reflected by the softmax output.  $\psi (\cdot)$  is a function that returns a single feature vector when given a set of feature vectors, and their corresponding weights across all  $i$  or  $l$ . Vector  $\pmb{s}_{tl}$  reflects the sub-portion of  $\pmb{s}_t$  associated with layer  $l$ .

In (5) we provide attention in the spatiotemporal dimensions, with that spatiotemporal attention shared across all  $L$  (now aligned) CNN layers. In (6) the attention is further refined, focusing attention in the layer dimension.

To make the following discussion concrete, we describe the attention function within the context of  $\mathbf{z}_t = \psi(\{\mathbf{s}_{tl}\}, \{\beta_{tl}\})$ . This function setup is applied in the same way to  $\mathbf{s}_t = \psi(\{\hat{\mathbf{a}}_i\}, \{\alpha_{ti}\})$ .

Soft attention Following Bahdanau et al. (2015), we formulate the soft attention model by computing a weighted sum of the input features

$$
\boldsymbol {z} _ {t} = \psi \left(\left\{\boldsymbol {s} _ {t l} \right\}, \left\{\beta_ {t l} \right\}\right) = \sum_ {l = 1} ^ {L} \beta_ {t l} \boldsymbol {s} _ {t l}. \tag {7}
$$

The model is differentiable for all parameters and can be learned end-to-end using standard backpropagation.

**Hard attention** Let  $\boldsymbol{m}_t \in \{0,1\}^L$  be a vector of all zeros, and a single one, and the location of the non-zero element of  $\boldsymbol{m}_t$  identifies the location to extract features for generating the next word. We impose

$$
\boldsymbol {m} _ {t} \sim \operatorname {M u l t} \left(1, \left\{\beta_ {t l} \right\}\right), \quad \boldsymbol {z} _ {t} = \sum_ {l = 1} ^ {L} m _ {t l} \boldsymbol {s} _ {t l}. \tag {8}
$$

In this case, optimizing the objective function in (3) is intractable. However, the marginal log-likelihood can be lower-bounded as

$$
\log p (\mathbf {Y} | \mathbf {A}) = \log \sum_ {\boldsymbol {m}} p (\boldsymbol {m} | \mathbf {A}) p (\mathbf {Y} | \boldsymbol {m}, \mathbf {A}) \geq \sum_ {\boldsymbol {m}} p (\boldsymbol {m} | \mathbf {A}) \log p (\mathbf {Y} | \boldsymbol {m}, \mathbf {A}), \tag {9}
$$

where  $\pmb{m} = \{\pmb{m}_t\}_{t=1,\dots,T}$ . Inspired by importance sampling, the multi-sample stochastic lower bound has been recently used for latent variable models (Ba et al., 2015; Burda et al., 2016), defined as

$$
\mathcal {L} ^ {K} (\mathbf {Y}) = \sum_ {\boldsymbol {m} ^ {1: K}} p \left(\boldsymbol {m} ^ {1: K} \mid \mathbf {A}\right) \left[ \log \frac {1}{K} \sum_ {k = 1} ^ {K} p \left(\mathbf {Y} \mid \boldsymbol {m} ^ {k}, \mathbf {A}\right) \right], \tag {10}
$$

where  $m^1, \ldots, m^K$  are independent samples. This lower bound is guaranteed to be tighter with the increase of the number of samples  $K$  (Burda et al., 2016), thus providing a better approximation of the objective function than (9). As shown in Mnih & Rezende (2016), the gradient of  $\mathcal{L}^K(\mathbf{Y})$  with respect to the model parameters is

$$
\nabla \mathcal {L} ^ {K} (\mathbf {Y}) = \sum_ {\boldsymbol {m} ^ {1: K}} p \left(\boldsymbol {m} ^ {1: K} \mid \mathbf {A}\right) \sum_ {k = 1} ^ {K} \left[ L \left(\boldsymbol {m} ^ {1: K}\right) \nabla \log p \left(\boldsymbol {m} ^ {k} \mid \mathbf {A}\right) + \omega_ {k} \nabla p \left(\mathbf {Y} \mid \boldsymbol {m} ^ {k}, \mathbf {A}\right) \right], \tag {11}
$$

where  $L(\pmb{m}^{1:K}) = \log \frac{1}{K}\sum_{k=1}^{K} p(\mathbf{Y}|\pmb{m}^k, \mathbf{A})$  and  $\omega_k = \frac{p(\mathbf{Y}|\pmb{m}^k, \mathbf{A})}{\sum_j p(\mathbf{Y}|\pmb{m}^j, \mathbf{A})}$ . A variance reduction technique is introduced in Mnih & Rezende (2016) by replacing the above gradient with an unbiased estimator

$$
\nabla \mathcal {L} ^ {K} (\mathbf {Y}) \approx \sum_ {k = 1} ^ {K} \left[ \hat {L} \left(\boldsymbol {m} ^ {k} \mid \boldsymbol {m} ^ {- k}\right) \nabla \log p \left(\boldsymbol {m} ^ {k} \mid \mathbf {A}\right) + \omega_ {k} \nabla p \left(\mathbf {Y} \mid \boldsymbol {m} ^ {k}, \mathbf {A}\right) \right], \tag {12}
$$

where

$$
\hat {L} \left(\boldsymbol {m} ^ {k} \mid \boldsymbol {m} ^ {- k}\right) = L \left(\boldsymbol {m} ^ {1: K}\right) - \log \frac {1}{K} \left(\sum_ {j \neq k} p (\mathbf {Y} \mid \boldsymbol {m} ^ {j}, \mathbf {A}) + f (\mathbf {Y}, \boldsymbol {m} ^ {- k}, \mathbf {A}))\right), \tag {13}
$$

$$
f (\mathbf {Y}, \boldsymbol {m} ^ {- k}, \mathbf {A}) = \exp \left(\frac {1}{K - 1} \sum_ {j \neq k} \log p (\mathbf {Y} | \boldsymbol {m} ^ {j}, \mathbf {A}). \right. \tag {14}
$$

When learning the model parameters, the lower bound (10) is optimized via the gradient approximation in (12).

As an alternative method, one may first produce abstraction-level attention weights  $\beta_{l}$ , followed by generating spatiotemporal attention weights  $\alpha_{i}$ , i.e., switching the order of (5) and (6). However, we empirically found that this method provides slightly worse performance (we implemented and tested both) in our experiments, partially due to the increase of the number of parameters.

# 4 EXPERIMENTS

# 4.1 DATASETS

We present results on three benchmark datasets: Microsoft Research Video Description Corpus (YouTube2Text) (Chen & Dolan, 2011), Montreal Video Annotation Dataset (M-VAD) (Torabi et al., 2015), and Microsoft Research Video to Text (MSR-VTT) (Xu et al., 2016).

The Youtube2Text contains 1970 Youtube clips, and each video is annotated with around 40 sentences. For fair comparison, we used the same splits as provided in Venugopalan et al. (2015b), with 1200 videos for training, 100 videos for validation, and 670 videos for testing.

The M-VAD is a large-scale movie description dataset, which is composed of 46587 movie snippets annotated with 56752 sentences. We follow the setting in Torabi et al. (2015), taking 36920 videos for training, 4950 videos for validation, and 4717 videos for testing.

The MSR-VTT is a newly collected large-scale video dataset, consisting of 20 video categories. The dataset was split into 6513, 2990 and 497 clips in the training, testing and validation sets. Each video has about 20-sentence descriptions. The ground-truth captions in the testing set are not available now. Thus, we split the original training dataset into a training set of 5513 clips and a testing set of 1000 clips.

We convert all captions to lower case and remove the punctuation, yielding vocabulary sizes  $V = 12594$  for Youtube2Text,  $V = 13276$  for M-VAD, and  $V = 8071$  for MSR-VTT.

# 4.2 TRAINING PROCEDURE

We consider the RGB frames of videos as input, and all videos are resized to  $112 \times 112$  spatially, with 2 frames per second; note that the temporal sample rate of the videos are consequently consistent, but not necessarily the total number of frames. The C3D (Tran et al., 2015) is pretrained on Sports-1M dataset Karpathy et al. (2014), consisting of 1.1 million sports videos belonging to 487 categories. We extract the features from four convolutional layers and one fully connected layer, named as pool2, pool3, pool4, pool5 and fc-7 in the C3D (Tran et al., 2015), respectively. The model architecture of C3D is provided in Appendix B. The kernel sizes of the convolutional transformation in (4) are  $7 \times 7 \times 7$ ,  $5 \times 5 \times 5$  and  $3 \times 3 \times 3$  for layer pool2, pool3 and pool4 with  $3 \times 3 \times 3$ ,  $2 \times 2 \times 2$  and  $1 \times 1 \times 1$  zero padding, respectively.  $f(\cdot)$  is implemented by ReLU (Nair & Hinton, 2010), followed by 3D max-pooling with  $8 \times 8 \times 8$ ,  $4 \times 4 \times 4$  and  $2 \times 2 \times 2$  ratios. More details for achieving spatiotemporal alignment are provided in Appendix C.

All recurrent matrices in the LSTM are initialized with orthogonal initialization (Saxe et al., 2014). We initialize non-recurrent weights from a uniform distribution in  $[-0.01, 0.01]$  and all the bias terms are initialized to zero. Word embedding vectors are initialized with the publicly available word2vec vectors that were trained on 100 billion words from Google News, which have dimensionality 300, and were trained using a continuous bag-of-words architecture (Mikolov et al., 2013). The embedding vectors of words not present in the pre-trained set are initialized randomly. The number of hidden units in the LSTM is set as 512 and we use mini-batches of size 32. Gradients are clipped if the norm of the parameter vector exceeds 5 (Sutskever et al., 2014). The number of samples for multi-sample stochastic lower bound is set to 10. We do not perform any dataset-specific tuning and regularization other than dropout (Srivastava et al., 2014) and early stopping on validation sets. The Adam algorithm Kingma & Ba (2014) with learning rate 0.0002 is utilized for optimization. All experiments are implemented in Torch (Collobert et al., 2011).

# 4.3 EVALUATION

The widely used BLEU (Papineni et al., 2002), METEOR (Banerjee & Lavie, 2005) and CIDER (Vedantam et al., 2015) metrics are employed to quantitatively evaluate the performance of our video caption generation model, and other models in the literature. For each dataset, we show three types of results, using part of or all of our model to illustrate the role of each component:

1.  $C3D fc7 + LSTM$ : The LSTM caption decoder is employed, only using features extracted from the top fully-connected layer. No context vector  $\mathbf{z}_t$  is generated from intermediate convolutional layer features.  
2. Spatiotemporal attention + LSTM: The context vector  $\mathbf{z}_t$  is included, but only features extracted from a certain convolutional layer are employed, i.e.,  $\mathbf{z}_t$  is equal to  $s_t$  in (5). The spatiotemporal attention is implemented with the soft attention in (7). For example, "C3D fc7 + pool2" means we leverage the features from layer pool2 to generate attention weights and context vector.

Table 1: Results on BLEU-4, METEOR and CIDEr metrics compared to other state-of-the-art results and baselines on Youtube2Text and M-VAD datasets. [1,2,3,4] represent Venugopalan et al. (2015a); Pan et al. (2016b); Ballas et al. (2016); Yu et al. (2016), respectively.  

<table><tr><td rowspan="2">Methods</td><td colspan="3">Youtube2Text</td><td colspan="3">M-VAD</td></tr><tr><td>BLEU-4</td><td>METEOR</td><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>CIDEr</td></tr><tr><td>S2VT (VGG) [1]</td><td>-</td><td>29.2</td><td>-</td><td>-</td><td>5.6</td><td>-</td></tr><tr><td>LSTM-E (VGG + C3D) [2]</td><td>45.3</td><td>31.0</td><td>-</td><td>-</td><td>6.7</td><td>-</td></tr><tr><td>GRU-RCN [3]</td><td>47.90</td><td>31.14</td><td>67.82</td><td>-</td><td>-</td><td>-</td></tr><tr><td>h-RNN (C3D+VGG) [4]</td><td>49.9</td><td>32.6</td><td>65.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="7">Baselines</td></tr><tr><td>ResNet + LSTM</td><td>44.08</td><td>30.99</td><td>66.88</td><td>0.81</td><td>6.22</td><td>5.54</td></tr><tr><td>C3D fc7 + LSTM</td><td>45.34</td><td>31.21</td><td>66.12</td><td>0.83</td><td>6.31</td><td>5.96</td></tr><tr><td colspan="7">Spatiotemporal attention + LSTM</td></tr><tr><td>C3D fc7 + pool2</td><td>45.46</td><td>31.53</td><td>67.38</td><td>0.98</td><td>6.42</td><td>6.01</td></tr><tr><td>C3D fc7 + pool3</td><td>48.07</td><td>33.52</td><td>69.97</td><td>1.12</td><td>6.71</td><td>6.97</td></tr><tr><td>C3D fc7 + pool4</td><td>48.18</td><td>34.47</td><td>70.97</td><td>1.24</td><td>6.89</td><td>7.12</td></tr><tr><td>C3D fc7 + pool5</td><td>47.75</td><td>33.35</td><td>69.71</td><td>1.02</td><td>6.49</td><td>6.48</td></tr><tr><td colspan="7">Baselines of multi-level features with attention</td></tr><tr><td>MLP + soft attention</td><td>35.99</td><td>23.56</td><td>44.21</td><td></td><td></td><td></td></tr><tr><td>Max-pooling + soft attention</td><td>45.35</td><td>31.50</td><td>62.99</td><td></td><td></td><td></td></tr><tr><td>Average-pooling + soft attention</td><td>48.53</td><td>33.59</td><td>65.07</td><td></td><td></td><td></td></tr><tr><td>Hypercolumn + soft attention</td><td>44.92</td><td>30.18</td><td>63.18</td><td></td><td></td><td></td></tr><tr><td colspan="7">ASTAR</td></tr><tr><td>Soft Attention</td><td>51.74</td><td>36.39</td><td>72.18</td><td>1.94</td><td>7.72</td><td>7.98</td></tr><tr><td>Hard Attention</td><td>51.64</td><td>34.99</td><td>72.08</td><td>1.82</td><td>7.12</td><td>8.12</td></tr></table>

3. ASTAR: This is our proposed model developed in Sec. 3.2. We present results based on both soft attention and hard attention.

To compare our method with other multi-level feature methods, we show several baseline results on Youtube2Text dataset:

1. MLP: Each  $\mathbf{a}_l$  is feed to a different MLP to achieve the same dimension of  $\mathbf{a}_L$ , i.e.  $\hat{\mathbf{a}}_l = \mathrm{MLP}(\mathrm{vec}(\mathbf{a}_l))$  for  $l = 1, \dots, L - 1$ , where  $\mathrm{vec}(\cdot)$  denotes vectorization. The context vector  $\mathbf{z}_t$  is obtained by abstraction-level and spatiotemporal attention.  
2. Max/Average-pooling: A max or average pooling operation is utilized to achieve saptiotemporal alignment, and an MLP is then employed to embed the feature vectors into the same semantic space, i.e. for each  $\hat{\pmb{a}}_l$  with  $l = 1,\dots ,L$ :

$$
\tilde {\boldsymbol {a}} _ {i, l} (k) = \max  _ {j \in \mathcal {N} _ {i, l}} \boldsymbol {a} _ {j, l} (k) \text {o r} \frac {1}{| \mathcal {N} _ {i , l} |} \sum_ {j \in \mathcal {N} _ {i, l}} \boldsymbol {a} _ {i, l} (k) \quad \text {f o r} k = 1, \dots , n _ {F} ^ {l} \tag {15}
$$

$$
\hat {\boldsymbol {a}} _ {i, l} = \operatorname {M L P} \left(\tilde {\boldsymbol {a}} _ {i, l}\right) \tag {16}
$$

where  $\mathcal{N}_{i,l}$  is the receptive field (see Appendix C for the definition) of  $a_{i,L}$  in the  $l$ -th layer. Similarly, the context vector  $\pmb{z}_t$  is computed by abstraction-level and spatiotemporal attention.

3. Hypercolumn: This method is similar with max/average-pooling but replace the pooling operation with hypercolumn representation. All the features are concatenated into one long vector for every location and only a septiotemporal attention is employed to generate the context vector.

In these methods, the attention weights are produced by the soft attention in (7).

To verify the effectiveness of our video caption generation model and C3D features, we also implement a strong baseline method based on the LSTM encoder-decoder network (Cho et al., 2014), where ResNet He et al. (2016) is employed as the feature extractor on each frame. We denote results using this method as  $ResNet + LSTM$ .

# 4.4 QUANTITATIVE RESULTS

Results are summarized in Tables 1 and 2, and we obtain state-of-the-art results on both Youtube2Text and M-VAD datasets. On the MSR-VTT dataset, our results also significantly outperform the strong baselines, demonstrating the effectiveness of the proposed model. Note that no comparative results exist in the literature for MSR-VTT. As a reference, with similar settings, Xu et al. (2016) using  $C3D$  fc7 + LSTM achieved 39.9 on BLEU-4 and 29.3 on METEOR using the standard testing set, while state-of-the-art results are 40.8 on BLEU-4, 28.2 on METEOR, and 44.8 on CIDEr, achieved by team "v2t责任感" from RUC and CMU

Table 2: Results on BLEU-4, METEOR and CIDEr metrics compared to baselines on MSR-VTT.  

<table><tr><td>Method</td><td>BLEU-4</td><td>METEOR</td><td>CIDEr</td></tr><tr><td colspan="4">Baselines</td></tr><tr><td>ResNet + LSTM</td><td>39.54</td><td>26.59</td><td>45.22</td></tr><tr><td>C3D fc7 + LSTM</td><td>40.17</td><td>26.86</td><td>45.95</td></tr><tr><td colspan="4">Spatiotemporal attention + LSTM</td></tr><tr><td>C3D fc7 + pool2</td><td>40.43</td><td>26.93</td><td>47.15</td></tr><tr><td>C3D fc7 + pool3</td><td>42.04</td><td>27.18</td><td>48.93</td></tr><tr><td>C3D fc7 + pool4</td><td>41.98</td><td>27.42</td><td>48.21</td></tr><tr><td>C3D fc7 + pool5</td><td>40.83</td><td>27.01</td><td>47.86</td></tr><tr><td colspan="4">ASTAR</td></tr><tr><td>soft attention</td><td>43.72</td><td>29.67</td><td>50.21</td></tr><tr><td>hard attention</td><td>43.89</td><td>28.71</td><td>50.29</td></tr></table>

It is worth noting that our model consistently yields significant improvements over models with only spatiotemporal attention, which further achieve better performance than only taking the C3D top fully-connected layer features; this demonstrates the importance of leveraging intermediate convolutional layer features. In addition, our model outperforms all the baseline results of multi-level feature based method, demonstrating the effectiveness of our 3D convolutional transformation operation. It is partly because the sparse connectivity of convolutional operation, which indicates fewer parameters are required. Furthermore, We achieve these results using a single model, without averaging over an ensemble of such models.

# 4.5 QUALITATIVE RESULTS

Following Xu et al. (2015), we visualize the attention components learned by our model on Youtube2Text in Figure 3. As can be seen from Figure 3, the spatiotemporal attention aligns the objects in the video well with the corresponding words. In addition, the abstraction-level attention tends to focus on low level features when the model will generate a noun and high level features when an article or a preposition is to be generated. More results are provided in Appendix A.1.

Examples of generated captions from unseen videos on Youtube2Text are shown in Figure 4. We find the results with abstraction-layer attention (indicated as "soft attention" or "hard attention") is generally equal to or better than the best results, compared to those only taking a certain convolutional-layer feature (indicated as "Pool2" etc.). It demonstrates the effectiveness of our abstraction layer attention. More results are provided in Appendix A.2 and http://www.cs.toronto.edu/pub/cuty/videocaption/.

# 5 CONCLUSION AND FUTURE WORK

We have proposed a novel video captioning model, that adaptively selects/weights the feature abstraction (CNN layer), as well as the location within a layer-dependent feature map. We have implemented the attention using both "hard" and "soft" mechanisms, with the latter typically delivering better performance. Our model achieves excellent video caption generation performance, and has the capacity to provide interpretable alignments seemingly analogous to human perception.

We have focused on analysis of videos and associated captions. Similar ideas may be applied in the future to image captioning. Additionally, the CNN parameters were learned separately as a first step, prior to analysis of the captions. It is also of interest to consider CNN-parameter refinement conditioned on observed training captions.

![](images/af2d7b215a6076a71a7c5ae701220f52398246cd232955d58da1b5db87d2e27b.jpg)  
Figure 3: Visualization of attention weights aligned with input video and generated caption on Youtube2Text: (top) Sampled frames of input video, (middle) soft attention, (bottom) hard attention. We show the frame with the strongest attention weights. The bar plot above each frame corresponds layer attention weights  $\beta_{tl}$  when the corresponding word (under the frame) is generated.

![](images/ab0bc4b24703735bdf603c48f8fe7665700773baecacc0f5698e2b8387620dfe.jpg)  
Pool2:  
Pool3:  
Pool4:

![](images/d5c30cb10c50c76831f37f9ce6047dca81f7f133e23a7629c2212ab895e8cf64.jpg)  
Pool5:  
Soft attention:  
Hard attention:  
Pool2:  
Pool3:  
Pool4:  
Pool5:  
Soft attention:  
Hard attention:  
Figure 4: Examples of generated captions with sampled frames of input video on YouTube2Text.

![](images/27e0ff215bbc6133fb6347721ec6027e69e601375a30e7b98b923b8aff63823f.jpg)  
a baby is walking  
a bear is walking  
a bear is walking  
king

![](images/6bb2bb3c2b4b45c13f1538eef68dc8f8e8f671abc794578313ec400047f56461.jpg)  
a badger is walking  
a turtle is walking  
a turtle is walking  
a group of men are playing  
a group of men are playing  
a girl is doing gymnastics  
a gymnast falls down  
a woman is doing gymnastics  
a girl is doing gymnastics

![](images/37f5640ea4ea42149cccb45549e0f7580cf683f1fec26523de7292297d5fc1c4.jpg)

![](images/d9a69ca06186a37da0764061db7cafaad84d05b1bfca96f1118b45b6f50c3cdc.jpg)

king

king

alking

king

king

![](images/47064d354f0545316ec2584964fb4cbbd5ef19341977b633b1a010714fdc715d.jpg)

![](images/fce5693d292a691a6258d18a318799c262a33fdbb559f1cbc165f2fe7b74fb1a.jpg)  
Pool2:  
Pool3:  
Pool4:

![](images/bfe61bb538f74e2fa7f4bf39049b2bf731d5755dc992b7a74281820bd2990a71.jpg)  
a girl is smiling  
a girl is talking  
a woman is licking a face

![](images/3cc36c6a98bfa560362fb7f29e1c73c22e5e70c274856e1a8d116955bebd48f2.jpg)

![](images/06378694ae0a04f84324f9bb85e453c5ee26e7f8c5b7d1c42b14c7d177b4d9d0.jpg)  
Pool5:  
Soft attention:  
Hard attention:  
Pool2:  
Pool3:  
Pool4:  
Pool5:  
Soft attention:  
Hard attention:

![](images/78ff1dbde31469d8d234eb32b87bda442e0a540f4d43083396c61e6bd890722b.jpg)  
a girl is putting stickers on her face  
a woman is putting stickers on her face  
a woman is putting stickers on her face  
a woman is peeling a toad  
a woman is applying eye makeup  
a woman is licking a brush  
a woman is plucking her face  
a woman is applying eye shadow  
a woman is applying eye makeup

![](images/f4f6e022c5376459b6ee7dc3b4ed3353f4e8539b712ee7332cf940437d0d09f5.jpg)

# REFERENCES

J. Ba, R. Grosse, R. Salakhutdinov, and B. Frey. Learning wake-sleep recurrent attention models. In NIPS, 2015.  
D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. In ICLR, 2015.  
N. Ballas, L. Yao, C. Pal, and A. Courville. Delving deeper into convolutional networks for learning video representations. In ICLR, 2016.  
S. Banerjee and A. Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In ACL workshop, 2005.  
Y. Burda, R. Grosse, and R. Salakhutdinov. Importance weighted autoencoders. In ICLR, 2016.  
D. Chen and W. B. Dolan. Collecting highly parallel data for paraphrase evaluation. In ACL, 2011.  
K. Cho, B. V. Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. In EMNLP, 2014.  
R. Collobert, K. Kavukcuoglu, and C. Farabet. Torch7: A matlab-like environment for machine learning. In *BigLearn, NIPS Workshop*, 2011.  
J. Donahue, L. Anne Hendricks, S. Guadarrama, M. Rohrbach, S. Venugopalan, K. Saenko, and T. Darrell. Long-term recurrent convolutional networks for visual recognition and description. In CVPR, 2015.  
S. Guadarrama, N. Krishnamoorthy, G. Malkarnenkar, S. Venugopalan, R. Mooney, T. Darrell, and K. Saenko. Youtube2text: Recognizing and describing arbitrary activities using semantic hierarchies and zero-shot recognition. In ICCV, 2013.  
B. Hariharan, . Arbeláez, R. Girshick, and J. Malik. Hypercolumns for object segmentation and fine-grained localization. In CVPR, 2015.  
K. He, X. Zhang, S. Ren, and Sun J. Deep residual learning for image recognition. In CVPR, 2016.  
S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 1997.  
A. Karpathy and F. Li. Deep visual-semantic alignments for generating image descriptions. In CVPR, 2015.  
A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and Li Fei-Fei. Large-scale video classification with convolutional neural networks. In CVPR, 2014.  
D. Kingma and J. Ba. Adam: A method for stochastic optimization. In ICLR, 2014.  
R. Kiros, R. Salakhutdinov, and R. Zemel. Multimodal neural language models. In ICML, 2014.  
A. Mahendran and A. Vedaldi. Understanding deep image representations by inverting them. In CVPR, 2015.  
J. Mao, W. Xu, Y. Yang, J. Wang, and A. Yuille. Deep captioning with multimodal recurrent neural networks (m-RNN). In ICLR, 2015.  
T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In NIPS, 2013.  
A. Mnih and D. J. Rezende. Variational inference for monte carlo objectives. In ICML, 2016.  
V. Mnih, N. Heess, A. Graves, and K. Kavukcuoglu. Recurrent models of visual attention. In NIPS, 2014.  
V. Nair and G. E. Hinton. Rectied linear units improve restricted boltzmann machines. In ICML, 2010.

P. Pan, Z. Xu, Y. Yang, F. Wu, and Y. Zhuang. Hierarchical recurrent neural encoder for video representation with application to captioning. In CVPR, 2016a.  
Y. Pan, T. Mei, T. Yao, H. Li, and Y. Rui. Jointly modeling embedding and translation to bridge video and language. In CVPR, 2016b.  
K. Papineni, S. Roukos, T. Ward, and W. Zhu. Bleu: a method for automatic evaluation of machine translation. Transactions of the Association for Computational Linguistics, 2002.  
A. Rohrbach, M. Rohrbach, W. Qiu, A. Friedrich, M. Pinkal, and B. Schiele. Coherent multisentence video description with variable level of detail. In  $GCPR$ . 2014.  
M. Rohrbach, W. Qiu, I. Titov, S. Thater, M. Pinkal, and B. Schiele. Translating video content to natural language descriptions. In ICCV, 2013.  
A. M. Saxe, J. L. McClelland, and S. Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. In ICLR, 2014.  
P. Sermanet, K. Kavukcuoglu, and Y. Chintala, S.and LeCun. Pedestrian detection with unsupervised multi-stage feature learning. In CVOR, 2013.  
N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. JMLR, 2014.  
I. Sutskever, O. Vinyals, and Q. V. Le. Sequence to sequence learning with neural networks. In NIPS, 2014.  
A Torabi, H Larochelle C Pal, and A Courville. Using descriptive video services to create a large data source for video annotation research. In arXiv preprint arXiv:1503.01070, 2015.  
D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri. Learning spatiotemporal features with 3d convolutional networks. In ICCV, 2015.  
R. Vedantam, Z. C. Lawrence, and D. Parikh. Cider: Consensus-based image description evaluation. In CVPR, 2015.  
S. Venugopalan, M. Rohrbach, J. Donahue, R. Mooney, T. Darrell, and K. Saenko. Sequence to sequence-video to text. In ICCV, 2015a.  
S. Venugopalan, H. Xu, J. Donahue, M. Rohrbach, R. Mooney, and K. Saenko. Translating videos to natural language using deep recurrent neural networks. In NAACL, 2015b.  
O. Vinyals, A. Toshev, S. Bengio, and D. Erhan. Show and tell: A neural image caption generator. In CVPR, 2015.  
J. Xu, T. Mei, T. Yao, and Y. Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, 2016.  
K. Xu, J. L. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R. S. Zemel, and Y. Bengio. Show, attend and tell: Neural image caption generation with visual attention. In ICML, 2015.  
L Yao, A. Torabi, K. Cho, N. Ballas, C. Pal, H. Larochelle, and A. Courville. Describing videos by exploiting temporal structure. In ICCV, 2015.  
H. Yu, J. Wang, Z. Huang, Y. Yang, and W. Xu. Video paragraph captioning using hierarchical recurrent neural networks. In CVPR, 2016.  
M. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. In ECCV, 2014.
