# PRUNING CONVOLUTIONAL NEURAL NETWORKS FOR RESOURCE EFFICIENT INFERENCE

Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, Jan Kautz  
NVIDIA

{pmolchanov, styree, tkarras, taila, jkautz}@nvidia.com

# ABSTRACT

We propose a new formulation for pruning convolutional kernels in neural networks to enable efficient inference. We interleave greedy criteria-based pruning with fine-tuning by backpropagation—a computationally efficient procedure that maintains good generalization in the pruned network. We propose a new criterion based on Taylor expansion that approximates the change in the cost function induced by pruning network parameters. We particularly focus on transfer learning, where large pretrained networks are adapted to specialized tasks. The proposed criterion demonstrates superior performance compared to other criteria, such as the norm of kernel weights or average feature map activation, for large 2D-CNNs on datasets for fine-grained classification (Birds-200 and Flowers-102). We also show results for the large-scale ImageNet dataset, emphasizing the flexibility of our approach.

# 1 INTRODUCTION

Convolutional neural networks (CNN) are used extensively in computer vision applications, including object classification and localization, pedestrian and car detection, and video classification. Many problems like these focus on specialized domains for which there are only small amounts of carefully curated training data. In these cases, accuracy may be improved by fine-tuning an existing deep network previously trained on a much larger labeled vision dataset, such as images from ImageNet (Russakovsky et al., 2015) or videos from Sports-1M (Karpathy et al., 2014). While this transfer learning supports state of the art accuracy, inference is expensive due to the time, power, and memory demanded by the heavy-weight architecture of the fine-tuned network.

While modern deep CNNs are composed of a variety of layer types, runtime during prediction is dominated by the evaluation of convolutional layers. With the goal of speeding up inference, we prune entire feature maps so the resulting networks may be run efficiently even on embedded devices. We interleave greedy criteria-based pruning with fine-tuning by backpropagation, a computationally efficient procedure that maintains good generalization in the pruned network. This is in line with existing work in structured pruning (Anwar et al., 2015), while largely orthogonal to fine-grained pruning of fully-connected layers (Han et al., 2015) and speedup by reduced precision (Gupta et al., 2015) or tensor decomposition (Kim et al., 2015).

Neural network pruning was pioneered in the early development of neural networks. Optimal Brain Damage (LeCun et al., 1990) leverages a second-order Taylor expansion to select parameters for deletion, using pruning as regularization to improve training and generalization. This method requires computation of the Hessian matrix, which can be memory bounded for modern networks.

More recently, Han et al. (2015) introduce a simpler approach by fine-tuning with a strong  $\ell_2$  regularization term and dropping parameters with values below a predefined threshold. Unstructured pruning is very effective for network compression, and this approach demonstrates good performance for intra-kernel pruning. But compression may not translate directly to faster inference, since modern hardware exploits regularities in computation for high throughput. So specialized hardware may be needed for efficient inference of a network with intra-kernel sparsity. This approach also requires long fine-tuning times that may exceed the original network training by a factor of 10.

Anwar et al. (2015) describe structured pruning in convolutional layers at the level of feature maps and kernels, as well as strided sparsity to prune with regularity within kernels. Pruning is accomplished

by particle filtering wherein configurations are weighted by misclassification rate. Other methods include reduced precision (Rastegari et al., 2016) and tensor decomposition (Kim et al., 2015). These approaches usually require a separate training procedure and significant fine-tuning. Our work is orthogonal to these approaches and can potentially be combined with each for additional speedups.

# 2 METHOD

The proposed method for pruning consists of the following steps: 1) Fine-tune the network until convergence on the target task; 2) Alternate iterations of pruning and further fine-tuning; 3) Stop pruning when the required trade-off between accuracy and pruning objective (e.g. FLOPs or memory) is reached.

We denote a set of image feature maps by  $\mathbf{z}_{\ell} \in \mathbb{R}^{H_{\ell} \times W_{\ell} \times C_{\ell}}$  with dimensionality  $H_{\ell} \times W_{\ell}$  and  $C_{\ell}$  individual maps (or channels). The feature maps can either be the input to the network,  $\mathbf{z}_0$ , or the output from a convolutional layer  $\mathbf{z}_{\ell}$  with  $\ell \in [1,2,\dots,L]$ . Individual feature maps are denoted  $\mathbf{z}_{\ell}^{(k)}$  for  $k \in [1,2,\dots,C_{\ell}]$ . A convolutional layer  $\ell$  applies the convolution operation  $(*)$  to a set of input feature maps  $\mathbf{z}_{\ell-1}$  with kernels parameterized by  $\mathbf{w}_{\ell}^{(k)} \in \mathbb{R}^{C_{\ell-1} \times p \times p}$ :

$$
\mathbf {z} _ {\ell} ^ {(k)} = g _ {\ell} ^ {(k)} \mathcal {R} \left(\mathbf {z} _ {\ell - 1} * \mathbf {w} _ {\ell} ^ {(k)} + b _ {\ell} ^ {(k)}\right), \tag {1}
$$

where  $\mathbf{z}_{\ell}^{(k)} \in \mathbb{R}^{H_{\ell} \times W_{\ell}}$  is the result of convolving each of  $C_{\ell-1}$  kernels of size  $p \times p$  with its respective input feature map and adding bias  $b_{\ell}^{(k)}; g_{\ell}^{(k)} \in \{0,1\}^{H_{\ell} \times W_{\ell}}$  is the pruning gate. The pruning gate determines if the output of a particular neuron is used during feedforward propagation or not. While we only provide notation for 2D convolutions, our methods are readily applicable to 3D convolutions and fully connected layers.

Assume a set of training examples is given as  $\mathcal{D} = \{\mathcal{X} = \{\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_N\}, \mathcal{Y} = \{y_0, y_1, \dots, y_N\}\}$ , where  $\mathbf{x}$  and  $y$  represent an input and an output, respectively. The network's parameters  $\mathcal{W} = \{(\mathbf{w}_1^1, b_1^1), (\mathbf{w}_1^2, b_1^2), \dots, (\mathbf{w}_L^{C_\ell}, b_L^{C_\ell})\}$  are optimized to minimize a cost value  $\mathcal{C}(\mathcal{D}|\mathcal{W})$ . The most common choice for a cost function  $\mathcal{C}(\cdot)$  is a negative log-likelihood function. A cost function is selected independently of pruning and depends only on the task to be solved by the original network. In the case of transfer learning, we adapt a large network initialized with parameters  $\mathcal{W}_0$  pretrained on a related but distinct dataset.

During pruning, we refine a subset of parameters, such that  $\mathcal{W}' = \mathcal{W}g$ , which preserves the accuracy of the adapted network,  $\mathcal{C}(\mathcal{D}|\mathcal{W}') \approx \mathcal{C}(\mathcal{D}|\mathcal{W})$ . This corresponds to a combinatorial optimization:

$$
\min  _ {\mathcal {W} ^ {\prime}} \left| \mathcal {C} \left(\mathcal {D} \mid \mathcal {W} ^ {\prime}\right) - \mathcal {C} \left(\mathcal {D} \mid \mathcal {W}\right) \right| \quad \text {s . t .} \quad \| \mathcal {W} ^ {\prime} \| _ {0} \leq B, \tag {2}
$$

where the  $\ell_0$  norm in  $\| \mathcal{W}'\| _0$  bounds the number of non-zero parameters  $B$  in  $W^{\prime}$

Intuitively if  $\mathcal{W}' = \mathcal{W}$  we reach the global minimum of the error function, however  $||\mathcal{W}'||_0$  will have it's maximum.

Finding a best subset of neurons that reproduce a cost value as close as possible is a combinatorial problem. It will require  $2^{|W|}$  evaluations of the cost function for a selected subset of data. For current networks it would be impossible to compute (for example VGG-16 has  $|\mathcal{W}| = 4224$  convolutional neurons). While it is impossible to solve this optimization exactly for networks of any reasonable size, in this work we investigate a class of greedy methods. Starting with a full set of parameters  $\mathcal{W}$ , we iteratively identify and remove the least important parameters, as illustrated in Figure 1. By removing parameters at each iteration, we ensure the eventual satisfaction of the  $\ell_0$  bound on  $\mathcal{W}'$ .

![](images/681b1d38035ab6606781fd547744fce98ef4dd35b4580619dd59a9016f52d761.jpg)  
Figure 1: Network pruning as a backward filter.

# 2.1 ORACLE PRUNING

Minimizing the difference in accuracy between the full and pruned models depends on the criterion for identifying the "least important" parameters at each step. The best criterion would be an exact empirical evaluation of each parameter, called oracle, accomplished by ablating each remaining parameter  $w \in \mathcal{W}'$  in turn and recording the difference in cost. This can be implemented as setting the pruning gate to 0 for each neuron in turn and estimating  $\mathcal{C}(\mathcal{D}|\mathcal{W})$ .

We differentiate two ways of using this Oracle estimation of importance: 1) Oracle-loss is a difference of losses, in this case the importance is estimated as  $\mathcal{C}(\mathcal{D}|\mathcal{W}') - \mathcal{C}(\mathcal{D}|\mathcal{W})$  and 2) Oracle-abs is the absolute difference of losses  $|\mathcal{C}(\mathcal{D}|\mathcal{W}') - \mathcal{C}(\mathcal{D}|\mathcal{W})|$ . Oracle-loss will give lower importance to neurons removing which is going to decrease loss. Oracle-abs will give lower importance to neurons removing which doesn't affect cost function.

While optimal for this greedy procedure, such an oracle is prohibitively costly to compute as it requires  $||W'||_0$  evaluations on a dataset set. Hence, the key part in such a pruning approach is how the importance of each neuron is estimated. We evaluate different criteria in terms of performance and estimation costs.

# 2.2 CRITERIA FOR PRUNING

There are many heuristic criteria which are much more computationally efficient than the oracle, and we evaluate several here. For the specific case of evaluating the importance of a feature map (and implicitly the set of convolutional kernels from which it is computed), reasonable criteria include: the combined  $\ell_2$ -norm of the kernel weights, the mean or standard deviation of the feature map's activation, and mutual information between activations and predictions. We introduce these criteria in the following and further propose a new criterion that estimates the cost of pruning based on the Taylor expansion.

Minimum weight. Pruning by magnitude of kernel weights is perhaps the simplest possible criterion, and it does not require any additional computation during the fine-tuning process. In case of using the weight's norm for pruning, a criterion is evaluated as:  $\Theta_{MW}:\mathbb{R}^{C_{\ell -1}\times p\times p}\to \mathbb{R}$  ..  $\Theta_{MW}(a) = \frac{1}{I}\sum_{i}a_{i}^{2}$ , where  $I$  is dimensionality of the weight after vectorization. The motivation to apply this type of pruning is that a convolutional kernel with low  $\ell_2$  norm detects less important features than those with a high norm. This can be aided during training by applying  $\ell_1$  and/or  $\ell_2$  regularization, which will push unimportant kernels to have smaller values.

Activation. One of the reasons of ReLU's popularity is that convolutional layers with this activation act as feature detectors. Therefore it is reasonable to assume that if the activation value (the output of the neuron) is small then this feature detector is not important for prediction of the output of the network  $\Theta_{MA}:\mathbb{R}^{H_l\times W_\ell \times C_\ell}\to \mathbb{R}$  ..  $\Theta_{MA}(a) = \frac{1}{I}\sum_{i}a_{i}$

Similarly, the standard deviation of the activation can indicate importance of the feature detector as well, i.e.,  $\Theta_{MA\_std}(a) = \sqrt{\frac{1}{I}\sum_i(a_i - \mu)^2}$ .

Mutual information. Mutual information (MI) is a measure of how much information is present in one variable about another variable. It captures linear and non-linear correlations between two variables. We apply MI as a criterion for pruning:  $\Theta_{MI}:\mathbb{R}^{H_l\times W_\ell \times C_\ell}\to \mathbb{R},\Theta_{MI}(z_\ell^{(k)}) = MI(z_\ell^{(k)},y)$ , where  $y$  is the target of neural network. MI is defined for continuous variables, and to simplify computations we exchange it with information gain (IG), which is defined for quantized variables  $IG(y|x) = H(x) + H(y) - H(x,y)$ , where  $H(x)$  is the entropy of variable x. We accumulate statistics on activations and ground truth for a number of updates, then they are quantized and we compute IG.

Taylor expansion. We phrase pruning as an optimization problem, trying to find a subset  $\mathcal{W}'$  of features that minimize  $|\Delta C(h_i)| = |\mathcal{C}(\mathcal{D}|\mathcal{W}') - \mathcal{C}(\mathcal{D}|\mathcal{W})|$ . With this approach, we will approximate change in the loss function dropping a particular neuron. Let  $h_i$  be the output of the neuron  $i$ , such that  $h = \{z_0^{(1)}, z_0^{(2)}, \dots, z_L^{(C_\ell)}\}$ . Also allow  $C(D|h_i) = C(D|(w, b)_i)$ . Assuming inter-independence

of neurons, we have:

$$
\left| \Delta C \left(h _ {i}\right) \right| = \left| C \left(D, h _ {i} = 0\right) - C \left(D, h _ {i}\right) \right|, \tag {3}
$$

where  $C(D, h_i = 0)$  is a cost value if neuron  $h_i$  is pruned, and  $C(D, h_i)$  is a cost value if it is not pruned. Note that the assumption of inter-independence of neurons is already made in the SGD training of DNNs. During training, we perform the following optimization:  $\min_h C(D, h) \forall h \in \mathcal{W}$ .

To approximate  $\Delta C(h_i)$  we will use the first-degree Taylor polynomial at point  $x = a$ :

$$
f (x) = \sum_ {p = 0} ^ {P} \frac {f ^ {(p)} (a)}{p !} (x - a) ^ {p} + R _ {p} (x), \tag {4}
$$

where  $f^{(p)}(a)$  is the  $p$ -th derivative of  $f$  at point  $a$ , and  $R_{p}(x)$  is a  $p$ -th order remainder. Approximating  $C(D, h_i = 0)$  with a first-order Taylor polynomial near  $x = 0$ , we have:

$$
C (D, h _ {i} = 0) = C (D, h _ {i}) - \frac {\delta C}{\delta h _ {i}} h _ {i} + R _ {1} (h _ {i} = 0). \tag {5}
$$

The remainder  $R_{1}(0)$  can be calculated through the Lagrange form:

$$
R _ {1} \left(h _ {i} = 0\right) = \frac {\delta^ {2} C}{\delta \left(h _ {i} ^ {2} = \xi\right)} \frac {h _ {i} ^ {2}}{2}, \tag {6}
$$

where  $\xi$  is a real number between 0 and  $h_i$ . We neglect the first-order remainder, largely due to the significant calculation required, but also in part because the widely-used ReLU activation function encourages a smaller second order term. Finally, by substituting Eq. (??) into Eq. (3) and ignoring the remainder, we get:

$$
\left| \Delta C \left(h _ {i}\right) \right| = \left| C (D, h _ {i}) - \frac {\delta C}{\delta h _ {i}} h _ {i} - C (D, h _ {i}) \right| = \left| \frac {\delta C}{\delta h _ {i}} h _ {i} \right| = \Theta_ {T E} \left(h _ {i}\right). \tag {7}
$$

For pruning we will use the gated feature map previously defined in Eq. (1), with the only exception that the pruning criterion will be  $\Theta_{TE}:\mathbb{R}^{H_l\times W_l\times C_l}\to \mathbb{R}^+$ , as defined in Eq. (7). Intuitively, this criterion prunes neurons that have an almost flat influence on the cost function. This approach requires accumulation of the product of the activation and the gradient wrt. the cost function which is precomputed for back-propagation during training.

$\Theta_{TE}$  is estimated for a neuron's output as:

$$
\Theta_ {T E} \left(z _ {l} ^ {(k)}\right) = \left| \frac {1}{M} \sum_ {m} \frac {\delta C}{\delta z _ {l , m} ^ {(k)}} z _ {l, m} ^ {(k)} \right|, \tag {8}
$$

where  $M$  is length of vectorized feature map. For a minibatch with  $T > 1$  examples, the criterion is computed for each example separately and averaged over  $T$ .

# 2.3 NORMALIZATION

Some criteria return "raw" values, whose scale vary with the depth of the parameter's layer in the network. A simple layer-wise  $\ell_2$ -normalization can achieve adequate scaling across layers:

$$
\hat {\Theta} (\mathbf {z} _ {l} ^ {(k)}) = \frac {\Theta (\mathbf {z} _ {l} ^ {(k)})}{\sqrt {\sum_ {j} \left(\Theta (\mathbf {z} _ {l} ^ {(j)})\right) ^ {2}}}.
$$

# 2.4 FLOPS REGULARIZED PRUNING

One of the main reasons to apply pruning is to reduce number of operations in the network. Neurons from different layers require different amount of computation due the sizes of input feature maps and kernels. To take this into account we introduce FLOPs regularization:

$$
\Theta \left(z _ {l} ^ {(k)}\right) = \Theta \left(z _ {l} ^ {(k)}\right) - \lambda \Theta_ {l} ^ {\text {f l o p s}}, \tag {9}
$$

where  $\lambda$  controls the regularization. For our experiments we select  $\lambda = 10^{-3}$ .  $\Theta^{flops}$  is computed under the assumption that convolution is implemented as a sliding window (see Appendix). Other regularization conditions can be applied here, such as storage size, kernel sizes, memory footprint.

![](images/616690beb5c1646df503fbf91246b282ceeba5c85a850c0c8013263be86f71f7.jpg)  
Figure 2: Statistics of neuron rank per layer.

![](images/276016328010dea130ada4f523652ca37532d3d54a6892c6f9bd9d70b7964478.jpg)  
Figure 3: Pruning conv kernels w/o fine-tuning.

# 3 RESULTS

# 3.1 ORACLE ANALYSIS

First we are going to evaluate the oracle for the visual transfer learning problem. We fine-tune the VGG-16 Simonyan & Zisserman (2014) network for classification of bird species using the Caltech-UCSD Birds 200-2011 dataset Wah et al. (2011). The dataset consists of nearly 6000 training images and 5700 testing images, covering 200 species. We fine-tune VGG-16 for 60 epochs with a learning rate of 0.0001 to achieve a test accuracy of  $72.2\%$ . Change in the loss caused by removing a particular neuron (computing oracle) from the VGG16 network fine-tuned on Birds-200 dataset is analyzed in Appendix 5.3. All results are obtained with Theano (Theano Development Team, 2016).

We further analyze the empirical contribution of neurons and rank them wrt. all others. Each convolutional neuron gets its rank (from 1 to 4224), where rank 1 means the most important neuron (removing it results in dramatic the highest increase of the loss) and rank 4224 means the least important. Statistics of neuron's ranks per layer is shown in Fig. 2. This analysis helps us to conclude the following:1) Each layer has very important and less valuable neurons. It means pruning should scale across all layers and not focus on particular layers. 2) Median demonstrates that the importance of layers decreases with depth. 3) Layers with max-pooling are more important than those without (VGG-16 has pooling after 2nd, 4th, 7th, 10th and 13th layer).

Next, we iteratively prune the VGG-16 network with oracle's ranking. At each iteration we take the least important neuron from the remaining subset of parameters and remove it. During this procedure we do not update the parameters of the network. Results (on training dataset) of pruning with the oracle are illustrated in Fig. 3. We observe that pruning with Oracle-abs scheme yields better results: within the same number of iterations it shows higher accuracy. These results justify the choice of the absolute difference of costs in the optimization problem for pruning in Eq. 2.

# 3.2 EVALUATING THE CRITERIA

Spearman's rank correlation estimates how well two estimators provide monotonically related outputs, even if their relationship is not linear. We use the Spearman's correlation to evaluate how well a criterion approximates the oracle:

$$
\mathcal {S} = 1 - \frac {6 \sum d _ {i} ^ {2}}{N \left(N ^ {2} - 1\right)}, \tag {10}
$$

where  $N$  is the total number of neurons (and the highest rank),  $d_{i} = \text{rank}(\text{oracle}_{i}) - \text{rank}(\text{criterion}_{i})$  is the rank difference. This correlation coefficient takes values between  $-1$  to 1, with  $-1$  meaning full negative correlation, 0 is no correlation, and 1 means full positive correlation.

Using Spearman's correlation, we compare the oracle-abs ranking to rankings by mutual information, mean weight magnitude, mean and standard deviation of feature map activation, and Taylor expansion. All values are obtained on training set only. As a sanity check, we also evaluated random ranking and observed 0.0 correlation coefficient across all layers.

<table><tr><td rowspan="2"></td><td rowspan="2">Mutual Information</td><td rowspan="2">Weight</td><td colspan="2">Activation</td><td rowspan="2">Taylor</td></tr><tr><td>Mean</td><td>S.d.</td></tr><tr><td>Per layer</td><td>0.28</td><td>0.27</td><td>0.56</td><td>0.57</td><td>0.73</td></tr><tr><td rowspan="2">All layers (w/ l2 norm)</td><td>0.35</td><td>0.34</td><td>0.35</td><td>0.30</td><td>0.14</td></tr><tr><td>0.47</td><td>0.33</td><td>0.64</td><td>0.66</td><td>0.73</td></tr></table>

Table 1: Spearman's rank correlation of criteria vs. oracle-abs for convolutional feature maps of VGG-16 fine-tuned on Birds-200.

![](images/8be8c0bd97b6ee9ec0f75588358bc5a54a02d46bd6629eea1f1a5857a82ca986.jpg)  
Figure 4: Pruning of VGG-16 fine-tuned on the Birds-200 dataset, with additional fine-tuning updates of 30 minibatches after each pruning iteration. Only convolutional kernels are pruned.

![](images/4f0af2fc6fc70a446fc6b73dfdc1c324dece4830c91dc649e398862ce723c003.jpg)

Results are shown in Table 1. "Per layer" analysis focuses on ranking within each convolutional layer, while "All layers" describes how well the criteria rank feature maps across layers. Results indicate that the Taylor expansion method exhibits superior performance within layers. Averaged correlation within layers is 0.73. Activation, Activation-standard and Taylor do not scale across layers without normalization. However, normalization across layers solves this, with  $\ell_2$  showing the best scores. Overall Taylor  $+\ell_{2}$  shows the best performance. We analyze this further in the appendix.

# 3.3 PRUNING 2D-CNN NETWORKS FINE-TUNED FROM IMAGENET NETWORKS

We now detail the performance of the full iterative pruning/fine-tuning procedure. We focus on reducing the number of convolutional parameters and floating point operations (FLOPs), assuming a sliding window implementation of the convolutions Lavin (2015) (see appendix).

Results of pruning VGG-16 after fine-tuning on the Birds-200 dataset are shown in Figure 4. Between pruning iterations we perform 30 minibatch SGD fine-tuning updates with batch-size 32, momentum 0.9, learning rate  $10^{-4}$ , and weight decay  $10^{-4}$ . We observe that "Taylor" shows the highest accuracy for any number of convolutional filters pruned. "Taylor with flops reg" demonstrates the best performance when the objective is to minimize FLOPs.

Next, we adapt the CaffeNet implementation of AlexNet (Krizhevsky et al., 2012) to the Oxford Flowers dataset (Nilsback & Zisserman, 2008), a collection with 2040 training and 6129 test images from 102 species of flowers found in the UK. We initially fine-tune the network for 20 epochs using a learning rate of 0.001 for a test accuracy of  $80.11\%$ .

Pruning of Alexnet after fine-tuning on Flowers-102 is shown in Figure 5. We keep the same fine-tuning parameters, except we reduce the number of mini-batch updates between pruning iterations to 10. We observe the superior performance of Taylor in pruning for both #parameters and FLOPs. Figure 6 shows pruning results computed with our Taylor technique and different numbers of updates between pruning iterations. We notice that increasing number of updates results in higher accuracy, while increasing the runtime of the pruning procedure.

![](images/1057c9b5df1213da07142996f6385122162613a138abd0bc16363090e3f7b443.jpg)  
Figure 5: Pruning of AlexNet on Flowers-102.

![](images/65aa07e1d56eeb5f49bfe81098f1db1d8671ec39d49b2a42d5c820466272545c.jpg)

![](images/a5c71a433e07561b9333b647247f1f14130d7050c6e207db1795ac28a138016a.jpg)  
Figure 6: Varying the number of minibatch updates between pruning iterations with AlexNet/Flowers-102.

![](images/cb56a1412ab798ca276b530e7611cb3e0e9a0f278417697c36de2d5cbc4b96b5.jpg)  
Figure 7: Pruning of recurrent 3D-CNN for dynamic hand gesture recognition, Molchanov et al. (2016).

# 3.4 RECURRENT 3D-CNN ON HAND GESTURES

Molchanov et al. (2016) learn to recognize 25 dynamic hand gestures in streaming video with a large recurrent neural network. The network is constructed by adding recurrent connections to a 3D-CNN network pretrained on Sports-1M (Karpathy et al., 2014) and fine tuning on a gesture dataset. The full network achieves an accuracy of  $80.7\%$  when trained on the depth modality, but a single inference requires an estimated 37.8 GFLOPs, too much for deployment on an embedded GPU. After several iterations of pruning with the Taylor criterion (with learning rate 0.0003, momentum 0.9, FLOPs regularization  $10^{-3}$ ), we reduce inference to 3.0 GFLOPs, as shown in Figure 7. While initial pruning increases classification error by nearly  $6\%$ , subsequent fine tuning restores much of the lost accuracy, yielding a final pruned network with a  $12.6\times$  reduction in GFLOPs with only  $2.5\%$  loss in accuracy.

# 3.4.1 PRUNING NETWORKS FOR IMAGENET

We also test our pruning scheme on a large-scale classification task, such as ImageNet. In the first experiment, we took a pretrained AlexNet (CaffeNet implementation) which gave us  $79.18\%$  top-5 accuracy on the validation set. We applied pruning with the following fine-tuning parameters: learning rate  $10^{-4}$ , momentum 0.9, weight decay  $10^{-4}$ , batch size 32, drop-out  $50\%$ . We took a small subset of the training examples (5k images) and computed oracle-abs and Spearman's rank correlation with the criteria in Table 2. Results of pruning are illustrated in Fig. 8.

We observed the following: 1) Taylor performs better than random or min weight pruning when 100 updates are used between pruning iterations. When results are displayed wrt. GFLOPs, the difference with random pruning is  $0\% - 4\%$ . When displayed according to iterations the difference is higher:  $1\% - 10\%$ . 2) Increasing the number of updates from 100 to 1000 improves performance of pruning significantly for TE and random pruning. Unsurprisingly, it also reduces the difference to random pruning.

Table 2: AlexNet on Imagenet: Spearman's rank correlation of criteria vs. oracle-abs.  

<table><tr><td></td><td>Weight</td><td>Activation</td><td>Taylor</td></tr><tr><td>Per layer</td><td>0.57</td><td>0.11</td><td>0.58</td></tr><tr><td>All layers – No regularization</td><td>0.67</td><td>0.0</td><td>-</td></tr><tr><td>All layers – ℓ2</td><td>0.44</td><td>0.1</td><td>0.55</td></tr></table>

![](images/bc3505bad21168715fa868abe17ba8842820ee28502b5c11f0458a73f5f58122.jpg)  
(a) Validation set, top-5 acc, iterations, 100 updates.

![](images/86cba6b6e71eb68e21284cc3fa495dc0a733b059bd5ac1a13889d395a90b2dd2.jpg)  
(b) Validation set, top-5 acc, GFLOPs, 100 updates.  
Figure 8: Pruning of AlexNet onImagenet with different number of updates between pruning iterations.

Second experiment is carried out for VGG-16 network on ImageNet dataset. We used the same parameters as before, except enabling flops regularization. We stop pruning at 11.5 and 8 GFLOPs and fine-tune models for additional 5 epochs with learning rate  $\lambda = 0.0001$ . We found that fine-tuning after pruning significantly improves results, such network pruned till  $11.5\%$  improves accuracy from  $83\%$  to  $87\%$ , and the network pruned to 8 GFLOPs improves from  $77.8\%$  to  $84.5\%$ .

# 4 CONCLUSIONS

We propose a new scheme for iterative pruning entire neurons from neural networks. We find: 1) CNNs may be successfully pruned by iteratively removing the least important neurons according to heuristics; 2) a Taylor expansion-based heuristic criterion demonstrates significant improvement over other criteria; 3) per-layer normalization of the criterion is important to obtain global scaling.

![](images/d30a4c4b59ffd0de99024d1b2747829980f19da2ac62883d3a875cec64d16d0d.jpg)  
Figure 9: Pruning VGG-16 on ImageNet dataset with following fine-tuning. Accuracy is shown on test set for top-5 prediction.

# REFERENCES

Sajid Anwar, Kyuyeon Hwang, and Wonyong Sung. Structured pruning of deep convolutional neural networks. arXiv preprint arXiv:1512.08571, 2015. URL http://arxiv.org/abs/1512.08571.  
Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. Deep learning with limited numerical precision. CoRR, abs/1502.02551, 392, 2015.  
Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In Advances in Neural Information Processing Systems, pp. 1135-1143, 2015.  
Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar, and Li Fei-Fei. Large-scale video classification with convolutional neural networks. In CVPR, 2014.  
Yong-Deok Kim, Eunhyeok Park, Sungjoo Yoo, Taelim Choi, Lu Yang, and Dongjun Shin. Compression of deep convolutional neural networks for fast and low power mobile applications. In Proceedings of the International Conference on Learning Representations (ICLR), 2015. URL http://arxiv.org/abs/1511.06530.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Andrew Lavin. maxdnn: An efficient convolution kernel for deep learning with maxwell gpus. CoRR, abs/1501.06633, 2015. URL http://arxiv.org/abs/1501.06633.  
Yann LeCun, J. S. Denker, S. Solla, R. E. Howard, and L. D. Jackel. Optimal brain damage. In Advances in Neural Information Processing Systems (NIPS), 1990.  
Pavlo Molchanov, Xiaodong Yang, Shalini Gupta, Kihwan Kim, Stephen Tyree, and Jan Kautz. Online detection and classification of dynamic hand gestures with recurrent 3d convolutional neural network. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.  
M-E. Nilsback and A. Zisserman. Automated flower classification over a large number of classes. In Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing, Dec 2008.  
Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. Xnor-net: Imagenet classification using binary convolutional neural networks. CoRR, abs/1603.05279, 2016. URL http://arxiv.org/abs/1603.05279.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115 (3):211-252, 2015. doi: 10.1007/s11263-015-0816-y.  
K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014.  
Theano Development Team. Theano: A Python framework for fast computation of mathematical expressions. arXiv e-prints, abs/1605.02688, May 2016. URL http://arxiv.org/abs/1605.02688.  
Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.
