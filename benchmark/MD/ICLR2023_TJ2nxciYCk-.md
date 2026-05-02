# ON EMERGENCE OF ACTIVATION SPARSITY IN TRAINED TRANSFORMERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

This paper reveals a curious observation that modern large-scale machine learning models with Transformer architectures have sparse activation maps. By activation map we refer to the intermediate output of the multi-layer perceptrons (MLPs) after a ReLU activation function, and by "sparse" we mean that on average very few entries (e.g.,  $3.0\%$  for T5-Base and  $6.3\%$  for ViT-B16) are nonzero for each input to MLP. Moreover, larger Transformers with more layers and wider MLP hidden dimensions are sparser as measured by the percentage of nonzero entries. Through extensive experiments we demonstrate that the emergence of sparsity is a prevalent phenomenon that occurs for both natural language processing and vision tasks, on both training and evaluation data, for Transformers of various configurations, at layers of all depth levels. We discuss how sparsity immediately implies a way to significantly reduce the FLOP count and improve efficiency for Transformers. Moreover, we demonstrate perhaps surprisingly that enforcing an even sparser activation via Top- $k$  thresholding with a small value of  $k$  brings a collection of desired but missing properties for Transformers, namely less sensitivity to noisy training data, more robustness to input corruptions, and better calibration for their prediction confidence.

# 1 INTRODUCTION

The great success of modern machine learning for tasks in computer vision, natural language processing, game playing and beyond is driven primarily by the computational model known as deep neural networks (DNNs) (LeCun et al., 2015). With inspirations drawn from biological intelligent systems, DNNs are massive systems of distributed computational nodes (a.k.a. neurons) with learned inter-connections, which possess the capacity of accomplishing complex real-world tasks.

Although originally motivated from biological brains, there are differences at very fundamental levels on how DNNs work compared to biological neural networks. One of such differences is in the sparsity of neural activities. Evidence from neuroscience suggests that neural activity in biological brains is sparse, namely, only a small percentage of all neurons fire at each time (Ahmed et al., 2020; Barth & Poulet, 2012; Kerr et al., 2005; Poo & Isaacson, 2009). Sparse firing suggests that despite having billions of neurons, only a small fraction of the brain participates in computation at each time, which may explain why brains can sustain at a very low energy cost. In contrast, learning and inference with DNNs rely primarily on dense computations where all neurons are involved for any input. In fact, modern computational hardware for deep neural networks, such as GPUs and TPUs, are designed to facilitate massive scale dense computations. Even with such dedicated hardware, DNNs are still notoriously resource-demanding to train and deploy. Aside from computation efficiency, DNNs also lag far behind biological brains in terms of robustness to input perturbation, error correction for erroneous training labels, confidence calibration for the predictions, etc.

1.1 AN INTRIGUING OBSERVATION: ACTIVATIONS ARE SPARSE IN TRAINED TRANSFORMERS This paper reveals a surprising observation that despite performing dense computations, DNNs produce very sparse activation in their intermediate layers once trained<sup>1</sup>. Specifically, we study

![](images/91661e1fc93c303167e5ba02a03bebd92aaa75b1c95a132ae4974e09a0073315.jpg)  
(a) T5 Encoder

![](images/b69ec83f7137297f189dcbc1108d5b7eb4d8a100ca834218bb68457ef2a07dc6.jpg)

![](images/978885e6c811c5338fa48aa0d1477da1b49eb3407659dc7d02342bf78c106196.jpg)  
Figure 1: Percentage of nonzero entries (y-axis, log scale) in the activation map as a function of number of training steps (x-axis) for a T5-Base model trained with the span corruption objective on the C4 dataset. Left: layers (from shallow to deep) of the encoder. Right: layers of the decoder.  
(b) T5 Decoder

Transformer (Vaswani et al., 2017), a DNN model architecture that has become a workhorse for modern applications. Transformers are constructed by interleaving a self-attention module and a multi-layer perceptrons (MLPs) of depth 2, and the focus of this paper is on the activation map of the first MLP layer. Figure 1 shows the sparsity of the activation maps, measured by the percentage of nonzeros, in all MLP layers of a T5-Base model (Raffel et al., 2020). We see that the percentage of nonzero entries is around  $50\%$  at initialization, which is expected: randomly initialized weights produce roughly equal numbers of positive and negative entries in the pre-activation map, resulting in about  $50\%$  non-zeros after the ReLU. However, at the end of training the percentage of nonzero entries reduces drastically: the average value across all encoder-decoder layers is  $2.7\%$  with the largest one being  $12.0\%$  and the smallest one being only  $1.1\%$ . The emergence of sparse activation in Transformers bears a similarity to the sparsity of neural activities in biological brains, revealing an interesting connection between artificial and biological networks. Moreover, unlike classical sparse methods where such a connection is established via explicit sparse regularization (Olshausen & Field, 1996), the sparsity observed in Transformers is emergent without any explicit design.

# 1.2 PREVALENCE, BENEFITS, AND CAUSES OF SPARSITY

This paper studies the aforementioned phenomenon of sparse activation in trained Transformers, with a focus on the following two questions. First, is the phenomenon shown in Figure 1 a corner case or does it occur broadly? Second, why should we care about the sparsity in DNNs, other than the appeal of its similarity to biological brains? Our main results along these two lines are summarized below.

1. Sparsity is a prevalent phenomenon. We show in Section 2 that the emergence of sparse activation reported in Figure 1 is not an isolated and cherry-picked case. Rather, sparsity is prevalent, and occurs broadly in Transformer models: it emerges in all layers of a Transformer, for Transformers trained on both vision and natural language data, for Transformers of various configurations, and for activation maps computed on both train and test data, etc. Moreover, through controlled experiments on the width and depth of Transformers, we reveal that larger models are sparser, as measured by percentage of nonzero entries. We also show in the Appendix B.1 that sparsity emerges in MLP-Mixers, vanilla MLPs, and to some extent Residual Networks.  
2. Sparsity improves efficiency. Sparsity of activation map implies that a lot of the computations being carried out in Transformer training and inference are wasted: they are doing nothing but multiplying values by zero. This immediately suggests that FLOPs can be reduced by avoiding all such computations, which we discuss in Section 3.1. Importantly, our method does not affect model output or their performance, in contrast to existing work that uses sparsity to improve efficiency which is built upon trial-and-error without ensuring their correctness (Hoeffler et al., 2021). To explicitly control the sparsity level, we further introduce Top- $k$  Transformer in Section 3.2, a simple modification of Transformers where a Top- $k$  thresholding is applied to the activation maps<sup>2</sup>. We provide experimental evidence that reduction in FLOPs via sparsity brings wall time benefits for unbatched decoding on TPUv4 with Top- $k$  T5 models with 11B parameters.

3. Sparsity improves robustness and calibration. We further show in Section 3.3 that enforcing explicit sparsity via Top- $k$ ' Transformers improves model performance in terms of less sensitivity to noisy training data, less sensitivity to input corruptions, and better confidence calibration.

In addition, we provide a study on the causes of sparsity in the Appendix D, showing that sparsity is likely not an artifact of the training data, and may be attributed to the training dynamics in the optimization process.

# 1.3 EXPERIMENTAL SETUP

We study the sparsity in activation maps of Transformers with two commonly used Transformer models, namely Text-to-Text Transfer Transformer (i.e., T5) and Vision Transformer (i.e., ViT).

- T5 is an encoder-decoder model for natural language processing tasks (Raffel et al., 2020). We train T5 on the Colossal Clean Crawled Corpus (C4) using the span corruption task.  
- ViT is an encoder model for vision tasks (Dosovitskiy et al., 2021). Unless specified otherwise, we train ViT on ImageNet-21k (Deng et al., 2009), an image classification dataset with 14M images and 21k classes. For certain cases we also use ImageNet-1k which is a subset of ImageNet-21k with 1.3M images and 1k classes.

We measure the sparsity level at the intermediate output of the two-layer MLPs in a Transformer. Recall that an MLP performs the following mapping

$$
f (\boldsymbol {x}; \boldsymbol {K}, \boldsymbol {V}) \doteq \sum_ {i = 1} ^ {d _ {\mathrm {f f}}} \left(\sigma \left(\left\langle \boldsymbol {k} _ {i}, \boldsymbol {x} \right\rangle\right) \cdot \boldsymbol {v} _ {i}\right), \text {o r e q u i v a l e n t l y ,} f (\boldsymbol {x}; \boldsymbol {K}, \boldsymbol {V}) \doteq \boldsymbol {V} \sigma \left(\boldsymbol {K} ^ {\top} \boldsymbol {x}\right), \tag {1}
$$

where  $\pmb{x} \in \mathbb{R}^{d_{\mathrm{model}}}$  is the input,  $\pmb{K} = [\pmb{k}_1, \dots, \pmb{k}_{d_{\mathrm{ff}}}] \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}$  and  $\pmb{V} = [\pmb{v}_1, \dots, \pmb{v}_{d_{\mathrm{ff}}}] \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}$  are learnable layer parameters, and  $\sigma(\cdot)$  is a nonlinear activation function. We use ReLU as the activation function  $\sigma(\cdot)$  for both T5 and ViT<sup>3</sup>. A two-layer MLP may be regarded as having  $d_{\mathrm{ff}}$  neurons in the hidden layer, where the  $i$ -th neuron performs the computation  $\sigma(\langle \pmb{k}_i, \pmb{x} \rangle) \cdot \pmb{v}_i$  and the final layer output is the sum of the output of all neurons. Each neuron is called activated if  $\sigma(\langle \pmb{k}_i, \pmb{x} \rangle)$  is strictly positive. Hence, the sparsity of neuron activation can be measured by the number of nonzero entries in the feature map

$$
\boldsymbol {a} \doteq \sigma (\boldsymbol {K} ^ {\top} \boldsymbol {x}) \in \mathbb {R} ^ {d _ {\mathrm {f f}}}. \tag {2}
$$

Both T5 and ViT come with several configurations for  $d_{\mathrm{model}}$ ,  $d_{\mathrm{ff}}$ , number of layers, etc. Unless specified otherwise, we will use the Base models (i.e., T5-Base and ViT-B/16) which have  $d_{\mathrm{model}} = 768$ ,  $d_{\mathrm{ff}} = 3072$ , and 12 layers (for ViT) and 12 encoder layers +12 decoder layers (for T5). Training details of T5 and ViT are provided in Appendix A.

# 2 PREVALENCE OF SPARSITY IN LEARNED TRANSFORMERS

This section shows thorough experiments on commonly used Transformers that sparsity in activation maps is a prevalent phenomenon. We also show through some controlled experiments that deeper and wider Transformers tend to be sparser measured by percentage of nonzero entries in activation maps.

# 2.1 SPARSITY IS A UBIQUITOUS PHENOMONON

We start by providing experimental evidence that the emergence of sparse activation in trained Transformers is a ubiquitous phenomenon. To this end, we plot the percentage of nonzero entries of activation maps in different Transformers, and present the results in Figure 2. These results demonstrate the following.

- Sparsity emerges for both Vision and NLP tasks. Figure 2a shows the percentage of nonzero entries of trained T5 and ViT models evaluated on their respective training datasets. We see that both encoder and decoder of T5, as well as the ViT, all exhibit sparsity.  
- Sparsity emerges on both training and evaluation data. Figure 2b shows the percentage of nonzero entries in a trained T5 model measured on both the training data and the evaluation data. We see

![](images/17f63ccd379ad81697e975d7b521f96e909c584d9b46d6ebd92d256b2e27b5d4.jpg)  
(a) T5 vs ViT

![](images/66bae0bda720e5ccd34bbd3cc7cc967c89cbf66406597376f539497dcd834919.jpg)  
(b) Train vs evaluation data

![](images/d7c7c0a9d90338113c603f8a4a04c5f9777fdc08bc1ca7883961a78c34ec423f.jpg)

![](images/45114272d25af3bb2fdd6be2bd5b21c87e324ddb800c221b75d156013896d4af.jpg)  
(d) Varying configuration (ViT)

![](images/415217a9a7c83e5e65d2b1b74b4ab46ff2bf063fb29f4653199c52f4a9ac40e8.jpg)  
(e) Varying config. (T5 Encoder)

![](images/34084ed95ca4bd905d319bc098afc2014333c8c6e01f2654472c5d01a288badb.jpg)  
Figure 2: Percentage of nonzero entries across different layers of trained Transformers (a) for both language data with T5 and vision data with ViT, (b) on both train and evaluation data, (c) for ViT trained on ImageNet of 21k vs 1k classes, (d) on ViT of varying configurations, and (e, f) on T5 of varying configurations. Note that the y-axis is in log scale. Sparsity emerges in all cases.  
(c) Different training data size  
(f) Varying config. (T5 Decoder)

that the property of sparsity generalizes very well to evaluation data as the curves for training and evaluation data align very closely with each other.

- Sparsity emerges on datasets of varying scale. Figure 2c shows the percentage of nonzero entries in ViT trained on both ImageNet-21k and ImageNet-1k, where the former is a superset of the later with approximately  $10 \times$  more images and  $21 \times$  more classes. We see that the scale of data does not affect much of the sparsity level.  
- Sparsity emerges on Transformers of varying configurations. Figure 2d shows the percentage of nonzero entries for ViT of varying configurations in model size. Figure 2e and 2f show the percentage of nonzero entries for encoder and decoder, respectively, of T5 with varying configurations in model size. We see that sparsity persists for all cases.  
- Sparsity emerges across all layers of a Transformer. Finally, all plots in Figure 2 show that sparsity emerges in all layers of a Transformer. Moreover, in all cases the first few and last few layers tend to be denser than intermediate layers.

The presence of sparsity in activation maps does not rule out the possibility that a small percentage of the neurons are always activated for all inputs, whereas the rest of the neurons are never activated. To illustrate that this is not the case, we experiment with a pretrained T5 base model<sup>4</sup> to plot the percentage of layer inputs for which each of the  $d_{\mathrm{ff}}$  neurons is activated when evaluated on 800 examples taken from C4 dataset with span corruption task. Note that there are  $800 \times 512 = 409600$  samples as MLP activation is computed per token. The results are presented in Figure 3 with x-axis being indices of neurons in the first encoder layer of T5 sorted in descending order according to percentage of layer inputs on which they are activated. It can be seen that while a few neurons are activated for around  $50\%$  of the

time, the vast majority of neurons (around  $93.5\%$ ) are activated less than  $10\%$  of the time. Moreover, there are no dead neurons that are never activated, and the least activated neuron is activated for around  $0.001\%$  of the time, and  $99\%$  of neurons are activated over  $1\%$  of the time. Finally, while the

![](images/f9bae37384b658dbe5d5057098113ead235186f9961310b085025b840f3f5327.jpg)  
Figure 3: Percentage of times that each neuron in the first MLP layer of a trained T5 is activated on C4 dataset.

![](images/1d97a0351a922f614be33966f7f312e9f141e8d43e055195ab6d51434333da62.jpg)  
(a) Sparsity vs. depth

![](images/d6179d9d644e74ae8eb6b675abe428603c8fc3aeb0e0f6e75efc02bc1001df2a.jpg)  
Figure 4: Activation sparsity across different encoder layers of trained T5 Transformers of (a) varying depth and (b, c) varying width. Since with varying width the dimension of activation maps also changes, we evaluate sparsity both in term of the percentage (as in (b)) and the count (as in (c)) of nonzeros. Deeper and wider models are sparser in terms of percentage of activated neurons.

![](images/741aef1c4f95d4a8944510c4c7aed7e2b23bdb2e6c101a117afe88c4c7fc7b0e.jpg)  
(b) Sparsity (percentage) vs. width  
(c) Sparsity (count) vs. width

results here are for neurons in the first MLP layer of a pretrained T5 base encoder, all other MLP layers show qualitatively similar behavior.

# 2.2 THE LARGER, THE SPARSER

We next examine the effect of model size on the sparsity level of activation maps. Note that Figure 2e and Figure 2f provide evidence with T5 of varying configuration that larger models tend to be sparser. Here we perform controlled experiments to examine the effect of model depth, measured by the number of Transformer layers, as well as the effect of model width, measured by the dimension of activation map of MLPs (i.e.,  $d_{\mathrm{ff}}$ ), separately. Towards that, we take a standard T5 model and vary the depth and width, respectively while keeping the rest of the configuration fixed, and examine their sparsity level after training. The results are presented in Figure 4 for the encoder, whereas we omit the results for the decoder as they are qualitatively the same as those for encoder.

It can be seen from Figure 4a that deeper Transformers are arguably sparser. For example, many of the middle layers of the 32-layer model have less than  $1\%$  nonzero entries while all shallower models have more than  $1\%$  nonzero entries across all layers. For comparing networks of different widths, we measure the sparsity with the percentage and the count of nonzero entries in Figure 4b and Figure 4c, respectively. It can be seen that wider models have a lower percentage of nonzero entries, though a higher count of nonzero entries.

# 3 EFFICIENT, ROBUST, AND CALIBRATED: SPARSITY IS ALL YOU NEED?

Exploiting sparsity is one of the most common strategies for improving the computational efficiency of Transformers. To introduce activation sparsity, previous work has tried using, e.g., an extra mixture-of-experts module (Fedus et al., 2022b), often with some nontrivial engineering design. The emergence of sparse activation and its prevalence discussed in Section 2 reveals that sparsity actually comes for free without the need of extra engineering for inducing sparsity at all.

In Section 3.1 we discuss how the free sparsity in Transformers brings us free computation efficiency in terms of FLOPs count. In Section 3.2 we introduce Top- $k$  Transformers, a simple modification of Transformers where a top- $k$  thresholding operation is applied to the activation maps in all MLPs. Top- $k$  Transformer allows us to obtain sparsity for all input to MLPs and throughout training, hence enables the benefit of sparsity to be realized during training and for all data. Finally, in Section 3.3 we show that enforcing sparser activation with smaller values of  $k$  in Top- $k$  Transformer (without any other hacks, tweaks and hyperparameter tuning) bestows Transformers several desired properties, namely, robustness of training with erroneous annotations, less sensitivity to input noise/perturbation, and better confidence calibration of the predictions.

# 3.1 EFFICIENCY FOR FREE

Given an embedding dimension  $d_{\mathrm{model}}$  and an MLP intermediate dimension  $d_{\mathrm{ff}}$ , the computational complexity of a Transformer for an input sequence of length  $N$  is  $\mathcal{O}(Nd_{\mathrm{model}}^2 + N^2 d_{\mathrm{model}} + Nd_{\mathrm{model}}d_{\mathrm{ff}})$ , where the first term comes from computing the key, query, and value matrices, the second term

comes from computing the self-attention matrix, and the third term comes from the MLP. For a fixed sequence length  $N$ , and considering the fact that  $d_{\mathrm{ff}}$  is often much larger than  $d_{\mathrm{model}}$ , it is arguable that MLP poses the computational bottleneck in large Transformers. In the following, we explain how sparsity in activation map of MLP can be leveraged to significantly reduce its computational cost, without affecting the model performance.

Efficiency for the Second MLP Layer. The sparse activation immediately suggests that a lot of the computation for inference with Transformers is not needed at all. That is, while doing dense matrix-matrix multiplications, much of it is about multiplying a vector by a value of zero, which can be avoided to save computation.

Specifically, we consider the second layer of the MLP in (1) which performs the computation

$$
V a, \tag {3}
$$

where  $\pmb{a} \in \mathbb{R}^{d_{\mathrm{ff}}}$  is the intermediate activation map of MLP (see (2)) and  $\pmb{V} \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}$  is the layer parameter. Eq. (3) involves a simple matrix-vector multiplication which has a FLOP count of  $2d_{\mathrm{model}} \times d_{\mathrm{ff}}$ . However, if  $\pmb{a}$  is sparse with, say  $s$  nonzero entries, then the FLOP count for (3) reduces to  $2d_{\mathrm{model}} \times s$ . Hence,

$$
F L O P \text {i n} 1 - \frac {s}{d _ {\mathrm {f f}}}.
$$

Note that  $\frac{s}{d_{\mathrm{ff}}}$  is exactly the percentage of nonzeros plotted in the y-axis of e.g. Figure 1, which is  $2.7\%$  averaged across all layers. Hence, the computational cost of the second MLP layer can be reduced by a significant amount. More excitingly, the reduction factor  $1 - \frac{s}{d_{\mathrm{ff}}}$  is likely to be even bigger for larger Transformer models (see Figures 4a and 4b), pointing to a greater reduction in computation.

Efficiency for the First MLP Layer. The sparsity in the intermediate activation map of MLP does not immediately suggest a reduction in computation for the first MLP layer. Nonetheless, it is possible to significantly reduce the computation in the first MLP layer by leveraging approximate nearest neighbor search, which we explain next.

Recall from (1) that the computation in the first MLP layer is given by

$$
\sigma \left(\boldsymbol {K} ^ {\top} \boldsymbol {x}\right), \tag {4}
$$

with  $\pmb{K} = [k_{1},\dots ,k_{d_{\mathrm{ff}}}] \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}$  being the layer parameter and  $\pmb{x}$  being the layer input. If the output is sparse with  $k$  nonzero entries, then the calculation in (4) may be formulated as finding  $k$  points from the set  $\{\pmb{k}_i\}_{i=1}^{d_{\mathrm{ff}}}$  that are "closest" to the input  $\pmb{x}$  measured by values of inner product. Such a problem is well-known as the nearest neighbor search (NNS) problem or the maximum inner product search problem. While naive solution of the NNS problem has linear complexity in  $d_{\mathrm{ff}}$ , there exists approximate algorithms (Guo et al., 2020; Johnson et al., 2019; Shrivastava & Li, 2014) that are of sublinear complexity, and using them in Transformers means that

$$
F L O P \text {i n t h e f i r s t M L P l a y m a r y} d _ {\mathrm {f f}}.
$$

There are of course the questions of whether such approximate NNS algorithms could hurt Transformer performance, which we leave for future study.

Finally, while the gain in efficiency discussed above is measured by FLOPs, the methods therein require computation with unstructured sparsity and data-dependent sparsity patterns, which are usually not well supported on computation hardwares such as TPUs and GPUs. As a result, they do not necessarily translate to wall time reductions with naive implementations. In the next section, we present a Top- $k$  Transformer with sparse computation realized by the implementation in Chern et al. (2022), which we show to bring wall time reduction for certain cases. We leave a study of more general implementation of sparse computation for obtaining wall time reduction to future work.

# 3.2 CONTROLLING SPARSITY WITH TOP- $k$  TRANSFORMERS

The benefit of efficiency from sparsity in Section 3.1 comes with caveats. First, while the activation maps are sparse on average, there is the possibility that some of the activation maps for certain inputs are denser hence cannot benefit from sparse computation. Second, sparsity occurs only in trained Transformers while the computation is dense during and particularly at the beginning of training.

Here we present Top- $k$  Transformer, a simple modification to Transformer architecture that allows us to control sparsity level for all model inputs, and throughout training. Top- $k$  Transformer is built

![](images/19df24e91a2ddd3e274d1ff695ca3060fc6e08bcac023b6b36998ee14f30fa80.jpg)  
(a) T5

![](images/f1d866012c5df52db974efd6ec67bc35a1b08ea2d7cb98bc28b7d56cc0964133.jpg)  
Figure 5: Training and evaluation accuracy of Top-  $k$  T5 for three different sizes: base, large and 3B (left) and Top-  $k$  ViT (right) with varying  $k$ . Top-  $k$  Transformer is on par with regular Transformer for a large enough  $k$ . e.g. for T5 3B with  $k = 128$ , and ViT with  $k = 256$ , the drop is around  $0.3\%$ .  
(b) ViT

upon a regular Transformer with the only modification being the MLP layers, where at the output of the activation function  $\sigma(\cdot)$  (see (1)) we add a Top- $k$  thresholding operator. That is, the MLPs of Top- $k$  Transformers perform the following computation

$$
f (\boldsymbol {x}; \boldsymbol {K}, \boldsymbol {V}) = \boldsymbol {V} \cdot \operatorname {T o p} _ {k} \left(\sigma \left(\boldsymbol {K} ^ {T} \boldsymbol {x}\right)\right), \tag {5}
$$

where  $\mathrm{Top}_k(\cdot)$  performs a thresholding that all entries other than those of the largest  $k$  values are set to zero with  $k$  being a hyper-parameter subject to design choices. Note that  $\mathrm{Top} - k$  Transformer reduces to a regular Transformer if we set  $k = d_{\mathrm{ff}}$ . By using a small value of  $k$ , the benefit of efficiency in terms of reduction in FLOP as discussed in Section 3.1 applies to Transformer training as well.

The immediate question for Top- $k$  Transformer is whether it offers controllable sparsity at the cost of a reduced performance. Here we conduct experiments with Top- $k$  T5 and Top- $k$  ViT, and evaluate their performance measured by prediction accuracy for C4 span corruption and ImageNet-21k classification tasks, respectively. The results are provided in Figure 5. We see that with the Top- $k$  T5-{Base, Large, 3B} (resp., Top- $k$  ViT) Transformer, taking  $k$  to be 128 (resp., 256) is sufficient for closely matching the test performance of the vanilla T5-{Base, Large, 3B} (resp., ViT). Note that this is achieved without any other hyper-parameter tuning for the Top- $k$  Transformers upon those used for a regular Transformer, and other hyper-parameter choices may further improve the performance of Top- $k$  Transformers.

Finally, we evaluate the benefit of Top- $k$  in terms of reducing inference time latency. In our experiment, we add a Top- $k$  thresholding to T5X (Roberts et al., 2022) with the implementation of jax.lax.approx_max_k (Chern et al., 2022) using a recall target of 0.95. Then, we evaluate the decoder per-token wall time for unbatched greedy decoding during inference on a single TPUv4 chip. The results with varying model sizes and varying values of  $k$  are presented in Figure 6. We observe that larger models have more wall time reduction, due to the fact that they have larger  $d_{\mathrm{ff}}$  hence more FLOPs reduction by our discussion in Section 3.1. In particular, for T5-11B we observe around  $10\%$  wall time reduction with  $k \leq 128$ , though this amount becomes smaller with a larger  $k = 256$ .

![](images/524b99e8c15245ea8e9fddf8e58384aff17bfd0970c2668600cd4c750272fdc0.jpg)  
Figure 6: Latency reduction for unbatched greedy decoding in decoder of Top- $k$  Transformers on TPUv4.

# 3.3 BONUS! IMPROVED ROBUSTNESS AND CALIBRATION

Despite not being explicitly designed for such purposes, inducing sparse activation via Top- $k$  Transformer has the benefits of improving model robustness<sup>5</sup> and confidence calibration. We demonstrate this using the image classification task with the ImageNet-1k dataset, and present the results in Table 1. All results for Top- $k$  ViT are obtained without any model and training hyper-parameter tuning upon those for ViT. Contexts and details are presented below. More results are presented in Appendix C.

Table 1: Evaluation of Top-128 ViT for ImageNet-1k classification in terms of 1) natural accuracy with ImageNet-1k evaluation set, 2) robust accuracy with  $\{40\%, 80\% \}$  corrupted training labels, 3) robust accuracy under input perturbation with additive {Gaussian, Impulse, Shot} noise on evaluation images, and 4) calibration error on evaluation data measured by ECE. Top-128 ViT is on par with ViT for natural accuracy while is significantly better for model robustness and calibration.  

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Natural Accuracy</td><td colspan="2">Accuracy w/ Train Label Noise</td><td colspan="3">Accuracy under Input Perturbation</td><td rowspan="2">Expected Calibration Error (ECE)</td></tr><tr><td>40%</td><td>80%</td><td>Gaussian</td><td>Impulse</td><td>Shot</td></tr><tr><td>ViT</td><td>74.85%</td><td>59.44%</td><td>25.35%</td><td>39.54%</td><td>37.37%</td><td>38.56%</td><td>8.42%</td></tr><tr><td>Top-128 ViT</td><td>74.83%</td><td>62.13%</td><td>30.80%</td><td>42.29%</td><td>40.07%</td><td>40.68%</td><td>7.48%</td></tr></table>

Robustness to Label Noise. An important challenge for DNNs is that they are highly susceptible to label noise, the problem where a certain percentage of training labels are corrupted or erroneously generated. This may be attributed to the fact that DNNs are often over-parameterized, hence too "capable" that they tend to overfit, or "memorize" the noisy labels without generalizing to test data. While many dedicated techniques exist (see e.g., Algan & Ulusoy (2021); Song et al. (2022) for a review), here we show that a simple Top- $k$  Transformer can effectively address the label noise issue.

We conduct experiments using the ImageNet-1k dataset for which we replace  $p\%$  of the labels in the training set with a random label drawn uniformly from the set of all possible labels. The evaluation performance under  $p \in \{40\%, 80\% \}$  label noise is presented in Table 1. It shows that Top- $k$  offers a consistent performance gain with label noise.

Confidence Calibration. Aside from label noise, another symptom of over-parameterization of DNNs is that they tend to be overly confident in their predictions. In the context of classification problems, they tend to assign a high (i.e., close to 1) probability to the class of its prediction, while it is more desirable that they produce a probability that is commensurate with its confidence level (Guo et al., 2017). A commonly used metric for confidence calibration is the expected calibration error (ECE) (Naeini et al., 2015), which is the discrepancy between the probability to the class of a model's prediction and the probability that its prediction is actually correct.

Here we measure the calibration of Top-  $k$  ViT via ECE and report the results in Table 1. It shows that Top-  $k$  enables the Transformer to be more calibrated when compared to a vanilla Transformer.

Robustness to Input Perturbation. Another important challenge with DNNs is that their outputs tend to be sensitive to naturally occurring image corruptions, which limits their application to mission critical tasks (Bhojanapalli et al., 2021). Here we evaluate the robustness of Top- $k$  ViT to three types of additive noises, namely Gaussian noise, impulse noise, and shot noise. For that purpose, we train Top- $k$  ViT on standard ImageNet-1k training data and report their classification accuracy on ImageNet-C (Hendrycks & Dietterich, 2019), a benchmark that contains algorithmically generated Gaussian, impulse, and shot noise (among many others types) applied to the ImageNet-1k test dataset. For each noise type, there are five severity levels. We report the averaged performance over all severity levels of each corruption type in Table 1.

# 4 RELATED WORK

Prior efforts on introducing sparsity in deep neural networks abound, though often with diverse motivations and objectives. Here we provide a brief overview of several popular lines of work.

Sparsity for Efficiency. Sparsity in either model weights or activation maps is often used for improving training and inference efficiency (see e.g. Hoefer et al. (2021) for a review). For activation sparsity in particular, sparsity for efficiency is explored perhaps first in ConvNets (Georgiadis, 2019; Kurtz et al., 2020; Rhu et al., 2018) before subsequently becoming a key design component in many of the largest Transformer based language and vision models (Du et al., 2022; Fedus et al., 2022a;b; Rajbhandari et al., 2022). The Top- $k$  thresholding that we use in Top- $k$  Transformer has also been previously used in Gupta et al. (2021) to improve memory efficiency of Transformers. However, it has been unclear a priori whether sparsity hurts model performance, hence the practice often relies on wishful design, trial-and-error, and post-hot justification (Baykal et al., 2022). Our discovery that

Transformers naturally produce sparse activation maps, and that larger models are even sparser, may provide principled perspectives towards efficiently training future large models.

Sparsity for Robustness. Many work find that smaller and sparser networks obtained by model compression are more robust to adversarial perturbation (Chen et al., 2022; Guo et al., 2018; Jordao & Pedrini, 2021) and label noise (Xue et al., 2022). Another line of work that uses sparsity for robustness leverages the property that practical data corruption is often sparse (Ghosh et al., 2017; Liu et al., 2022; You et al., 2020). None of the work mentioned above is based on sparsity in activation maps. More closely related to ours is the work of Ahmad & Scheinkman (2019) where sparsity in activation map of convolutional DNNs is shown to improve robustness to input perturbation, and Muthukumar & Sulam (2022) that leverages sparse activation to derive robust generalization error bounds.

Sparsity for Explainability. Work on leveraging sparsity for interpreting deep learning models long exist but often in a post-hoc fashion for examining the semantic meanings encoded by a neuron of a trained model (Dalvi et al., 2019). For Transformers, evidence suggests that the learned knowledge is encoded mainly in its MLPs with individual neurons expressing specific factual knowledge (Dai et al., 2022). Moreover, enforcing neuron activation sparsity in MLPs helps to improve the percentage of neurons that are interpretable (Elhage et al., 2022). Hence, our discovery may point to new directions towards developing more interpretable DNNs (Cuadros et al., 2022; Sajjad et al., 2021).

Sparsity for Data Modeling. Following the seminal work of Olshausen & Field (1996), there are a lot of interests in sparsity as an effective modeling of natural signals (Mairal et al., 2014). With the close resemblance of the computational structure of ReLU networks and sparse encoding algorithms (Gregor & LeCun, 2010), it became natural to study a DNN as a multi-layer sparse modeling of the data (Papyan et al., 2018). Along with substantial theoretical understanding of such a modeling are obtained (Papyan et al., 2017; Sulam et al., 2018), there are also experimental results on their practical benefits (Sun et al., 2018) though less often on modern large-scale data.

Sparsity for Theory of Over-parameterized Models. Because of its simplicity and well-developed theory in classical machine learning (Candes & Wakin, 2008; Vidal et al., 2015; Wright & Ma, 2022), sparse modeling is often used to provide theoretical understanding of modern large and overparameterized models. This include works on implicit regularization (Chou et al., 2021; Nacson et al., 2022; Vaskevicius et al., 2019; Woodworth et al., 2020; Zhao et al., 2019), nonconvex optimization (Buhai et al., 2020; Sulam et al., 2022), noise interpolators (Chinot et al., 2022; Donhauser et al., 2022; Koehler et al., 2021), etc. However, the aforementioned work uses sparsity as a testbed or toy model to gain insights, without implication of existence of sparsity in DNNs.

# 5 DISCUSSION

This work demonstrates the natural emergence of sparse activation in commonly used Transformer models (Section 2). The notion of sparsity pertains to the law of parsimony, a.k.a. Occam's razor, where among all possible explanations of observed data, the simplest ones are preferred. It is a fundamental scientific principle broadly used in various scientific and engineering subjects (Domingos, 1999; Epstein, 1984), including classical machine learning (Tibshirani, 1996). Hence, our discovery may be suggesting that the law of parsimony is playing a role in Transformers even though they are not explicitly designed so, resonating with recent view on the role of sparsity for intelligence systems (LeCun, 2022; Ma et al., 2022; Roberts, 2021; Vasudevan et al., 2021). More importantly, we back such a perspective by providing evidence of improved robustness and calibration via enforcing sparsity using Top- $k$  thresholding (Section 3), which indicates that sparsity is indeed a pertinent prior for good generalization. We hope that our work may motivate future effort on introducing sparsity in deep learning models in a more principled way for obtaining more efficient, robust, and calibrated models. Finally, while our motivation of studying sparse activation in Transformers comes (partly) from study of biological brains, establishing such a connection may reciprocally benefit efforts on applying artificial intelligence to the study of biology and neuroscience (Richards et al., 2022).

# REFERENCES

Subutai Ahmad and Luiz Scheinkman. How can we be so dense? the benefits of using highly sparse representations. arXiv preprint arXiv:1903.11257, 2019.  
Mohsin S Ahmed, James B Priestley, Angel Castro, Fabio Stefanini, Ana Sofia Solis Canales, Elizabeth M Balough, Erin Lavoie, Luca Mazzucato, Stefano Fusi, and Attila Losonczy. Hippocampal network reorganization underlies the formation of a temporal association memory. Neuron, 107(2): 283-291, 2020.  
Görkem Algan and Ilkay Ulusoy. Image classification with deep learning in the presence of noisy labels: A survey. Knowledge-Based Systems, 215:106771, 2021.  
Alison L Barth and James FA Poulet. Experimental evidence for sparse firing in the neocortex. Trends in neurosciences, 35(6):345-355, 2012.  
Cenk Baykal, Nishanth Dikkala, Rina Panigrahy, Cyrus Rashtchian, and Xin Wang. A theoretical view on sparsely activated networks. arXiv preprint arXiv:2208.04461, 2022.  
Srinadh Bhojanapalli, Ayan Chakrabarti, Daniel Glasner, Daliang Li, Thomas Unterthiner, and Andreas Veit. Understanding robustness of transformers for image classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10231-10241, 2021.  
Rares-Darius Buhai, Yoni Halpern, Yoon Kim, Andrej Risteski, and David Sontag. Empirical study of the benefits of overparameterization in learning latent variable models. In International Conference on Machine Learning, pp. 1211-1219. PMLR, 2020.  
Emmanuel J Candès and Michael B Wakin. An introduction to compressive sampling. IEEE signal processing magazine, 25(2):21-30, 2008.  
Tianlong Chen, Zhenyu Zhang, Santosh Balachandra, Haoyu Ma, Zehao Wang, Zhangyang Wang, et al. Sparsity winning twice: Better robust generalization from more efficient training. In International Conference on Learning Representations, 2022.  
Felix Chern, Blake Hechtman, Andy Davis, Ruiqi Guo, David Majnemer, and Sanjiv Kumar. Tpu-knn: K nearest neighbor search at peak flop/s. arXiv preprint arXiv:2206.14286, 2022.  
Geoffrey Chinot, Matthias Löffler, and Sara van de Geer. On the robustness of minimum norm interpolators and regularized empirical risk minimizers. The Annals of Statistics, 50(4):2306-2333, 2022.  
Hung-Hsu Chou, Johannes Maly, and Holger Rauhut. More is less: Inducing sparsity via overparameterization. arXiv preprint arXiv:2112.11027, 2021.  
Xavier Suau Cuadros, Luca Zappella, and Nicholas Apostoloff. Self-conditioning pre-trained language models. In International Conference on Machine Learning, pp. 4455-4473. PMLR, 2022.  
Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8493-8502, 2022.  
Fahim Dalvi, Nadir Durrani, Hassan Sajjad, Yonatan Belinkov, Anthony Bau, and James Glass. What is one grain of sand in the desert? analyzing individual neurons in deep nlp models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 6309-6317, 2019.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248-255. Ieee, 2009.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pp. 4171-4186, 2019.

Pedro Domingos. The role of occam's razor in knowledge discovery. Data mining and knowledge discovery, 3(4):409-425, 1999.  
Konstantin Donhauser, Nicolo Ruggeri, Stefan Stojanovic, and Fanny Yang. Fast rates for noisy interpolation require rethinking the effects of inductive bias. arXiv preprint arXiv:2203.03597, 2022.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.  
Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning, pp. 5547-5569. PMLR, 2022.  
Nelson Elhage, Tristan Hume, Catherine Olsson, Neel Nanda, Tom Henighan, Scott Johnston, Sheer ElShowk, Nicholas Joseph, Nova DasSarma, Ben Mann, Danny Hernandez, Amanda Askell, Kamal Ndousse, Jones, , Dawn Drain, Anna Chen, Yuntao Bai, Deep Ganguli, Liane Lovitt, Zac Hatfield-Dodds, Jackson Kernion, Tom Conerly, Shauna Kravec, Stanislav Fort, Saurav Kadavath, Josh Jacobson, Eli Tran-Johnson, Jared Kaplan, Jack Clark, Tom Brown, Sam McCandlish, Dario Amodei, and Christopher Olah. Softmax linear units. Transformer Circuits Thread, 2022. https://transformer-circuits.pub/2022/solu/index.html.  
Robert Epstein. The principle of parsimony and some applications in psychology. The Journal of Mind and Behavior, pp. 119-130, 1984.  
William Fedus, Jeff Dean, and Barret Zoph. A review of sparse expert models in deep learning. arXiv preprint arXiv:2209.01667, 2022a.  
William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1-39, 2022b.  
Georgios Georgiadis. Accelerating convolutional neural networks via activation map compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7085-7095, 2019.  
Aritra Ghosh, Himanshu Kumar, and PS Sastry. Robust loss functions under label noise for deep neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 31, 2017.  
Karol Gregor and Yann LeCun. Learning fast approximations of sparse coding. In Proceedings of the 27th international conference on international conference on machine learning, pp. 399-406, 2010.  
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pp. 1321-1330. PMLR, 2017.  
Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. Accelerating large-scale inference with anisotropic vector quantization. In International Conference on Machine Learning, pp. 3887-3896. PMLR, 2020.  
Yiwen Guo, Chao Zhang, Changshui Zhang, and Yurong Chen. Sparse dnns with improved adversarial robustness. Advances in neural information processing systems, 31, 2018.  
Ankit Gupta, Guy Dar, Shaya Goodman, David Ciprut, and Jonathan Berant. Memory-efficient transformers via top-k attention. In Proceedings of the Second Workshop on Simple and Efficient Natural Language Processing, pp. 39-52, 2021.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations, 2019.  
Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. J. Mach. Learn. Res., 22(241):1-124, 2021.  
Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535-547, 2019.  
Artur Jordao and Helio Pedrini. On the effect of pruning on adversarial robustness. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1-11, 2021.  
Jason ND Kerr, David Greenberg, and Fritjof Helmchen. Imaging input and output of neocortical networks in vivo. Proceedings of the National Academy of Sciences, 102(39):14063-14068, 2005.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.  
Frederic Koehler, Lijia Zhou, Danica J Sutherland, and Nathan Srebro. Uniform convergence of interpolators: Gaussian width, norm bounds and benign overfitting. In Advances in Neural Information Processing Systems, 2021.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Communications of the ACM, 60(6):84-90, 2017.  
Mark Kurtz, Justin Kopinsky, Rati Gelashvili, Alexander Matveev, John Carr, Michael Goin, William Leiserson, Sage Moore, Nir Shavit, and Dan Alistarh. Inducing and exploiting activation sparsity for fast inference on deep neural networks. In International Conference on Machine Learning, pp. 5533-5543. PMLR, 2020.  
Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. 2022.  
Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436-444, 2015.  
Sheng Liu, Zhihui Zhu, Qing Qu, and Chong You. Robust training under label noise by overparameterization. 2022.  
Yi Ma, Doris Tsao, and Heung-Yeung Shum. On the principles of parsimony and self-consistency for the emergence of intelligence. Frontiers of Information Technology & Electronic Engineering, pp. 1-26, 2022.  
Julien Mairal, Francis Bach, Jean Ponce, et al. Sparse modeling for image and vision processing. Foundations and Trends® in Computer Graphics and Vision, 8(2-3):85-283, 2014.  
Ramchandran Muthukumar and Jeremias Sulam. Adversarial robustness of sparse local lipschitz predictors. arXiv preprint arXiv:2202.13216, 2022.  
Mor Shpigel Nacson, Kavya Ravichandran, Nathan Srebro, and Daniel Soudry. Implicit bias of the step size in linear diagonal neural networks. In International Conference on Machine Learning, pp. 16270-16295. PMLR, 2022.  
Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.  
Bruno A Olshausen and David J Field. Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583):607-609, 1996.  
Vardan Papyan, Yaniv Romano, and Michael Elad. Convolutional neural networks analyzed via convolutional sparse coding. The Journal of Machine Learning Research, 18(1):2887-2938, 2017.

Vardan Papyan, Yaniv Romano, Jeremias Sulam, and Michael Elad. Theoretical foundations of deep learning via sparse representations: A multilayer sparse model and its connection to convolutional neural networks. IEEE Signal Processing Magazine, 35(4):72-89, 2018.  
Cindy Poo and Jeffry S Isaacson. Odor representations in olfactory cortex: "sparse" coding, global inhibition, and oscillations. Neuron, 62(6):850-861, 2009.  
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(140):1-67, 2020.  
Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale. arXiv preprint arXiv:2201.05596, 2022.  
Minsoo Rhu, Mike O'Connor, Niladrish Chatterjee, Jeff Pool, Youngeun Kwon, and Stephen W Keckler. Compressing dma engine: Leveraging activation sparsity for training deep neural networks. In 2018 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 78-91. IEEE, 2018.  
Blake Richards, Doris Tsao, and Anthony Zador. The application of artificial intelligence to biology and neuroscience. Cell, 185(15):2640-2643, 2022.  
Adam Roberts, Hyung Won Chung, Anselm Levskaya, Gaurav Mishra, James Bradbury, Daniel Andor, Sharan Narang, Brian Lester, Colin Gaffney, Afroz Mohiuddin, Curtis Hawthorne, Aitor Lewkowycz, Alex Salcianu, Marc van Zee, Jacob Austin, Sebastian Goodman, Livio Baldini Soares, Haitang Hu, Sasha Tsvyashchenko, Aakanksha Chowdhery, Jasmijn Bastings, Jannis Bulian, Xavier Garcia, Jianmo Ni, Andrew Chen, Kathleen Kenealy, Jonathan H. Clark, Stephan Lee, Dan Garrette, James Lee-Thorp, Colin Raffel, Noam Shazeer, Marvin Ritter, Maarten Bosma, Alexandre Passos, Jeremy Maitin-Shepard, Noah Fiedel, Mark Omernick, Brennan Saeta, Ryan Sepassi, Alexander Spiridonov, Joshua Newlan, and Andrea Gesmundo. Scaling up models and data with t5x and seqio. arXiv preprint arXiv:2203.17189, 2022. URL https://arxiv.org/abs/2203.17189.  
Daniel A. Roberts. Why is ai hard and physics simple?, 2021.  
Hassan Sajjad, Nadir Durrani, and Fahim Dalvi. Neuron-level interpretation of deep nlp models: A survey. arXiv preprint arXiv:2108.13138, 2021.  
Anshumali Shrivastava and Ping Li. Asymmetric lsh (alsh) for sublinear time maximum inner product search (mips). Advances in neural information processing systems, 27, 2014.  
Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from noisy labels with deep neural networks: A survey. IEEE Transactions on Neural Networks and Learning Systems, 2022.  
Jeremias Sulam, Vardan Papyan, Yaniv Romano, and Michael Elad. Multilayer convolutional sparse modeling: Pursuit and dictionary learning. IEEE Transactions on Signal Processing, 66(15): 4090-4104, 2018.  
Jeremias Sulam, Chong You, and Zhihui Zhu. Recovery and generalization in over-realized dictionary learning. Journal of Machine Learning Research, 23(135):1-23, 2022.  
Xiaoxia Sun, Nasser M Nasrabadi, and Trac D Tran. Supervised deep sparse coding networks. In 2018 25th IEEE International Conference on Image Processing (ICIP), pp. 346-350. IEEE, 2018.  
Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1):267-288, 1996.  
Ilya O Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, et al. Mlp-mixer: An all-mlp architecture for vision. Advances in Neural Information Processing Systems, 34: 24261-24272, 2021.

Tomas Vaskevicius, Varun Kanade, and Patrick Rebeschini. Implicit regularization for optimal sparse recovery. Advances in Neural Information Processing Systems, 32, 2019.  
Rama K Vasudevan, Maxim Ziatdinov, Lukas Vlcek, and Sergei V Kalinin. Off-the-shelf deep learning is not enough, and requires parsimony, bayesianity, and causality. npj Computational Materials, 7(1):1-6, 2021.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.  
Rene Vidal, Yi Ma, and Shankar Sastry. Generalized principal component analysis. *Interdisciplinary Applied Mathematics*, 43:22-23, 2015.  
Blake Woodworth, Suriya Gunasekar, Jason D Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models. In Conference on Learning Theory, pp. 3635-3673. PMLR, 2020.  
John Wright and Yi Ma. High-Dimensional Data Analysis with Low-Dimensional Models: Principles, Computation, and Applications. Cambridge University Press, 2022.  
Yihao Xue, Kyle Whitecross, and Baharan Mirzasoleiman. Superior generalization of smaller models in the presence of significant label noise. arXiv preprint arXiv:2208.08003, 2022.  
Chong You, Zhihui Zhu, Qing Qu, and Yi Ma. Robust recovery via implicit bias of discrepant learning rates for double over-parameterization. Advances in Neural Information Processing Systems, 33: 17733-17744, 2020.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM, 64(3):107-115, 2021.  
Peng Zhao, Yun Yang, and Qiao-Chu He. Implicit regularization via hadamard product overparametrization in high-dimensional linear regression. arXiv preprint arXiv:1903.09367, 2019.
