# PRUNING FILTERS FOR EFFICIENT CONVNETS

Hao Li*

University of Maryland

haoli@cs.umd.edu

Asim Kadav

NEC Labs America

asim@nec-labs.com

Igor Durdanovic

NEC Labs America

igord@nec-labs.com

Hanan Samet

University of Maryland

hjs@cs.umd.edu

Hans Peter Graf

NEC Labs America

hpg@nec-labs.com

# ABSTRACT

The success of CNNs in various applications is accompanied by a significant increase in the computation and parameter storage costs. Recent efforts towards reducing these overheads involve pruning and compressing the weights of various layers without hurting the accuracy. However, using weights pruning to generate sparse CNNs mostly reduces parameters from the fully connected layers and may not significantly reduce the computation costs with irregular sparsity.

We present an acceleration method for CNNs, where we prune filters from CNNs that are identified as having a small effect on the output accuracy. By removing whole filters in the network, together with their connecting feature maps, the computational costs are reduced significantly. In contrast to weights pruning, this approach does not result in sparse connectivity patterns. Hence, it does not need the support of sparse convolution libraries and can work with existing efficient BLAS libraries for dense matrix multiplications. We show that even simple filter pruning techniques can reduce inference costs for VGG-16 by up to  $34\%$  and ResNet-110 by upto  $38\%$  while regaining close to the original accuracy by retraining the networks.

# 1 INTRODUCTION

The ImageNet challenge has led to significant advancements in exploring various architectural choices in CNNs (Russakovsky et al. (2015); Krizhevsky et al. (2012); Simonyan & Zisserman (2015); Szegedy et al. (2015a); He et al. (2015)). The general trend since the past few years has been that the networks have grown deeper, with an overall increase in the number of parameters and convolution operations. These high capacity networks have significant inference costs especially when used with embedded sensors or mobile devices where computational and power resources can be limited. For these applications, in addition to accuracy, computational efficiency and small network sizes are crucial enabling factors (Szegedy et al. (2015b)). In addition, for web services like image search and image classification APIs that operate on a time budget, often serving hundreds of thousands of images per second, benefit significantly from lower inference time.

There has been a significant amount of work on reducing the storage and computational cost by model compression (Le Cun et al. (1989); Hassibi & Stork (1993); Srinivas & Babu (2015); Han et al. (2015); Mariet & Sra (2016)). Recently Han et al. (2015; 2016b) report impressive compression rates on AlexNet (Krizhevsky et al. (2012)) and VGGNet (Simonyan & Zisserman (2015)) by pruning weights with small magnitudes, followed by retraining without hurting the accuracy. However, pruning parameters does not necessarily reduce the computation since majority of the parameters removed are from the fully connected layers where the computation cost is low, e.g., the fully connected layers of VGG-16 occupies  $90\%$  of the total parameters but only contribute less than  $1\%$  of the overall compute floating point operations (FLOP). They also demonstrate that the convolutional layers can be compressed and accelerated ( Iandola et al. (2016)), but additionally require sparse BLAS libraries or even specialized hardware (Han et al. (2016a)). Modern libraries that provide speedup using sparse operations over CNNs are often limited (Szegedy et al. (2015a); Liu et al.

(2015)) and maintaining sparse data structures also creates an additional storage overhead which can be significant for low-precision weights.

Recent work on CNNs have developed deep architectures with more efficient design (Szegedy et al. (2015a;b); He & Sun (2015); He et al. (2015)), in which the fully connected layers are replaced with average pooling layers (Lin et al. (2013); He et al. (2015)), which reduces the number of parameters significantly. The computation cost is also reduced by downsampling the image at an early stage to reduce the size of feature maps (He & Sun (2015)). Nevertheless, as the networks continue to become deeper, the computational costs of convolutional layers continue to dominate.

CNNs with large capacity usually have significant redundancy between different filters and feature channels. In this work, we focus on reducing the computation cost of well-trained CNNs by pruning filters. Compared to pruning weights across the network, filter pruning is a naturally structured way of weights pruning without introducing sparsity and therefore does not require using sparse libraries or any specialized hardware. The number of pruned filters correlates directly with acceleration by reducing the number of matrix multiplications, which is easy to tune for a target speedup. In addition, instead of layer-wise iterative fine-tuning (retraining), we adopt a one-shot pruning and retraining strategy to save retraining time for pruning filters across multiple layers, which is critical for pruning very deep networks. Finally, we observe that even for ResNets, which have significantly fewer parameters and inference costs than AlexNet or VGGNet, still have about  $30\%$  of FLOP reduction without scarifying accuracy. We conduct sensitivity analysis for convolutional layers in ResNets, which provide hints for further understanding and improvement of ResNets.

# 2 RELATED WORK

The early work by Le Cun et al. (1989) introduces Optimal Brain Damage, which prunes weights with a theoretically justified saliency measure. Later, Hassibi & Stork (1993) propose Optimal Brain Surgeon to remove unimportant weights determined by the second-order derivative information. Mariet & Sra (2016) reduce the network redundancy by identifying a subset of diverse neurons that does not require retraining. However, this method only operates on the fully-connected layers and introduce sparse connections.

To reduce the computation costs of the convolutional layer, some work have proposed to approximate convolutional operations by representing the weight matrix as a low rank product of two smaller matrices without changing the original number of filters (Denil et al. (2013); Jaderberg et al. (2014); Zhang et al. (2015b;a); Tai et al. (2016); Ioannou et al. (2015)). Other approaches to reduce the convolutional overheads including using FFT based convolutions (Mathieu et al. (2013)) and fast convolution using the Winograd algorithm (Lavin & Gray (2016)). Additionally, quantization (Han et al. (2016b)) and binarization (Rastegari et al. (2016); Courbariaux & Bengio (2016)) can be used to reduce the model size and lower the computation overheads. Our method can be used in addition to these techniques to reduce computation costs without incurring additional overheads.

Several work have studied removing redundant feature maps from a well trained network (Anwar et al. (2015); Polyak & Wolf (2015)). Anwar et al. (2015) introduce a three-level pruning of the weights and locate the pruning candidates using particle filtering, which selects the best combination from a number of random generated masks. Polyak & Wolf (2015) detect the less frequently activated feature maps with sample input data for face detection applications. We choose to analyze the filter weights and prune filters with its corresponding feature maps using a simple magnitude based measure, without examining possible combinations. We also introducing network wide holistic approaches to prune filters for simple as well as more complex convolutional network architectures.

Concurrently with our work, there is also a growing interest in learning compact CNNs with sparse constraints (Lebedev & Lempitsky (2016); Zhou et al. (2016); Wen et al. (2016)). Lebedev & Lempitsky (2016) leverage group-sparsity on the convolutional filters to achieve structured brain damage, i.e., prune the entries of the convolution kernel in a groupwise fashion. Zhou et al. (2016) add group-sparse regularization on neurons during training to learn compact CNNs with reduced filters. Wen et al. (2016) add structured sparsity regularizer on each layer to reduce unimportant filters, channels or even layers. In the filter-level pruning, they all adopt  $\ell_{2,1}$  norm as regularizer. Similar with above work, we use a  $\ell_1$ -norm to select unimportant filters and physically prune them. The fine-tuning process is the same as the conventional training procedure, without introducing

additional regularization. Our approach does not introduce extra layer-wise meta-parameters for the regularizer except the percentage of filters to be pruned, which is directly related to the desired speedup. By employing stage-wise pruning, we can set one pruning rate for all layers in one stage.

# 3 PRUNING FILTERS AND FEATURE MAPS

Let  $n_i$  denote the number of input channels for the  $i$ th convolutional layer and  $h_i / w_i$  be the height/width of the input feature maps. The convolutional layer transforms the input feature maps  $\mathbf{x}_i \in \mathbb{R}^{n_i \times h_i \times w_i}$  into the output feature maps  $\mathbf{x}_{i+1} \in \mathbb{R}^{n_{i+1} \times h_{i+1} \times w_{i+1}}$ , which is used as input feature maps for the next convolutional layer. This is achieved by applying  $n_{i+1}$  3D filters  $\mathcal{F}_{i,j} \in \mathbb{R}^{n_i \times k \times k}$  on the  $n_i$  input channels, in which one filter generates one feature map. Each filter is composed by  $n_i$  2D kernels  $\mathcal{K} \in \mathbb{R}^{k \times k}$  (e.g.,  $3 \times 3$ ). All the filters, together, constitute the kernel matrix  $\mathcal{F}_i \in \mathbb{R}^{n_i \times n_{i+1} \times k \times k}$ . The operations of the convolutional layer are  $n_{i+1} n_i k^2 h_{i+1} w_{i+1}$ .

As shown in Figure 1, when a filter  $\mathcal{F}_{i,j}$  is pruned, its corresponding feature map  $\mathbf{x}_{i+1,j}$  is removed, which reduces  $n_i k^2 h_{i+1} w_{i+1}$  operations. The kernels that apply on the removed feature maps from the filters of the next convolutional layer are also removed, which saves an additional  $n_{i+2} k^2 h_{i+2} w_{i+2}$  operations. Pruning  $m$  filters of layer  $i$  will reduce  $m / n_{i+1}$  of the computation cost for both layer  $i$  that  $i + 1$ .

![](images/1dc8fd464dc36aa2c8ea158f923a0d48a0e186ecc23f52849683c36f06272f50.jpg)  
Figure 1: Pruning a filter results in removal of its corresponding feature map and related kernels in the next layer.

# 3.1 DETERMINING WHICH FILTERS TO PRUNE

Our method prunes the less useful filters from a well-trained model for computational efficiency while minimizing the accuracy drop. We measure the relative importance of a filter in each layer by calculating its absolute weight sum  $\sum |\mathcal{F}_{i,j}|$ , i.e., its  $\ell_1$ -norm  $\| \mathcal{F}_{i,j}\| _1$ . Since the number of input channels,  $n_i$ , is the same across filters,  $\sum |\mathcal{F}_{i,j}|$  also represents the average magnitude of its kernel weights. This value gives an expectation of the magnitude of the output feature map. Filters with smaller kernel weights tend to produce feature maps with weak activations as compared to the other filters in that layer. Pruning smallest filters also works best in comparison with pruning the same number of largest or random filters. Figure 2(a) illustrates the distribution of this filter weight sum for each layer in a VGG-16 network trained on the CIFAR-10 dataset.

The procedure of pruning  $m$  filters from the  $i$ th convolutional layer is as follows:

1. For each filter  $\mathcal{F}_{i,j}$ , calculate the absolute sum of its kernel weights  $s_j = \sum_{l=1}^{n_i} \sum |\mathcal{K}_l|$ .  
2. Sort the filters by  $s_j$ .  
3. Prune  $m$  filters with smallest sum values and their corresponding feature maps. The kernels in the next convolutional layer corresponding to the pruned feature maps are also removed.  
4. A new kernel matrix is created for both  $i$ th and  $i + 1$ th layer, and the remaining kernel weights are copied to the new model.

Relationship to pruning weights Pruning filters with low absolute weight sum values is similar to pruning low magnitude weights (Han et al. (2015)). Magnitude based weights pruning may also prune away whole filters when all the kernel weights of a filter are lower than a threshold. But it requires a careful tuning of the threshold and makes it difficult to predict the exact number of filters that will be eventually pruned. Furthermore, it generates sparse convolutional kernels which can be hard to accelerate given the lack of efficient sparse libraries, especially for the case of low-sparsity.

![](images/0b455a27f569149c9d84dd2f7f332af7bdb48c21666942176edaddbe55716823.jpg)  
(a) Filters are ranked by  $s_j$  
Figure 2: (a) Sorted filter weight sum for each layer of VGG-16 on CIFAR-10. The x-axis is the filter index divided by the total number of filters; The y-axis is the filter weight sum divided by the max sum value among filters in that layer. (b) Pruning filters with lowest absolute weights sum and their corresponding test accuracy on CIFAR-10. (c) Prune and retrain for each single layer of VGG-16 on CIFAR-10. Some layers are sensitive and it can be harder to recover accuracy after pruning them.

![](images/558699a533a92403719edda1c422935964d6300d9eb06cd0ab9ee6657fbd6e7a.jpg)  
(b) Prune smallest filters of each layer

![](images/08c82b8f6eacbcdfb37d8394b8adcb39f07f57496b79f2f0b1791a05c3c58c14.jpg)  
(c) Retraining result

Relationship to group-sparse regularization on filters Recent works (Zhou et al. (2016); Wen et al. (2016)) apply group-sparse regularization  $(\sum_{j=1}^{n_i} \| \mathcal{F}_{i,j} \|_2$  or  $\ell_{2,1}$ -norm) on convolutional filters, which also favors zero-out filters with small  $l_2$ -norms, i.e.  $\mathcal{F}_{i,j} = \mathbf{0}$ . In practice, we do not observe noticeable difference between  $\ell_2$ -norm or  $\ell_1$ -norm for filter selection, as the important filters tend to have large values in both measures. Zeroing out weights of a group filters during training actually has similar effect as pruning filters with the strategy of iterative prune and retrain as introduced in 3.4.

# 3.2 DETERMINING SINGLE LAYER'S SENSITIVITY TO PRUNING

To understand the sensitivity of each layer, we prune each layer independently and evaluate the resulting pruned network's accuracy on the validation set. Figure 2(b) shows the layers that maintain their accuracy as the filters are pruned away and they correspond to filters with larger slopes in Figure 2(a). On the contrary, layers with relatively flat slopes are more sensitive to pruning. We empirically determine the number of filters to prune for each layer based on their sensitivity to pruning. For deep networks like VGG-16 or ResNets, we observe that layers in the same stage (with the same feature map size) have similar sensitivity to pruning. To avoid introducing layer-wise meta-parameters, we use the same pruning ratio for all layers in the same stage. For layers that are very sensitive to pruning, we prune a smaller percentage of these layers or completely skip pruning them.

# 3.3 PRUNING FILTERS ACROSS MULTIPLE LAYERS

We now discuss how to prune filters across the network. Previous work prunes the weights on a layer by layer basis, followed by iteratively retraining and compensating for any loss of accuracy (Han et al. (2015)). However, understanding how to prune filters of multiple layers at once can be useful: 1) For deep networks, pruning and retraining on a layer by layer basis can be extremely time-consuming 2) Pruning layers across the network gives a holistic view of the robustness of the network resulting in a smaller network 3) For complex networks, a holistic approach may be necessary. For example, for the ResNet, pruning the identity feature maps or the second layer of each residual block results in additional pruning of other layers.

To prune filters across multiple layers, we consider two strategies for layer-wise filter selection:

- Independent pruning determines which filters should be pruned at each layer independent of other layers.  
- Greedy pruning accounts for the filters that have been removed in the previous layers. This strategy does not consider the kernels for the previously pruned feature maps while calculating the weight sum.

![](images/d7bf22eb5a9f952c56f8d81af30acd2a61241b4692e230651ee9273c7ad777b9.jpg)  
Figure 3 illustrates the difference between two approaches in weight sum calculation. The greedy approach, though not globally optimal, is more holistic and results in pruned networks with higher accuracy especially when large number of filters are pruned.

![](images/79a85d7f1931a4f81a328a839d9a8d9a17530a636b31bb8a4c42659c8133adc9.jpg)  
Figure 3: Pruning filters across consecutive layers. The independent pruning strategy calculates the filter sum (columns marked in green) without considering feature maps removed in previous layer (shown in blue), so the kernel weights marked in yellow is still included. The greedy pruning strategy does not count kernels for the already pruned feature maps (shown in yellow). Both approaches result in a  $(n_{i+1} - 1) \times (n_{i+2} - 1)$  kernel matrix.  
Figure 4: Pruning residual blocks with projection shortcut. The filters to be pruned for the second layer of the residual blocks (marked as green) is determined by the pruning result of the shortcut projection. The first layer of the residual blocks can be pruned without restrictions.

For simpler CNNs like VGGNet or AlexNet, we can easily prune any of the filters in any convolutional layer. However, for more complex network architectures such as Residual networks He et al. (2015), pruning filters may not be straightforward. The architecture of ResNet imposes several restrictions and the filters need to be pruned carefully. We show the filter pruning for residual blocks with projection mapping in Figure 4. Here, the filters of the first layer in the residual block can be arbitrarily pruned, as it does not change the number of output feature maps of the block. However, the correspondence between the output feature maps of the second convolutional layer and the identity feature maps making it difficult to prune. Hence, to prune the second convolutional layer of the residual block, the corresponding projected feature maps must also be pruned. Since the identical feature maps are more important than the added residual maps, the feature map to be pruned should be determined by the pruning results of the shortcut layer. To determine which identity feature maps are to be pruned, we use the same selection criterion based on the filters of the shortcut convolutional layers (with  $1 \times 1$  kernels). The second layer of the residual block is pruned with the same filter index as selected by the pruning of shortcut layer.

# 3.4 RETRAINED NETWORKS TO REGAIN ACCURACY

After pruning the filters, the performance degradation should be compensated by retraining the network. There are two strategies to prune the filters across multiple layers:

1. Prune once and retrain: Prune filters of multiple layers at once and retrain them until the original accuracy is restored.  
2. Prune and retrain iteratively: Prune filters layer by layer or filter by filter and then retrain iteratively. The model is retrained before pruning the next layer for the weights to adapt to the changes from the pruning process.

We find that for the layers that are resilient to pruning, the pruning and retrain once strategy can be used to prune away significant portions of the network and any loss in accuracy can be regained by retraining for a short period of time (less than the original training times). However, when some sensitive layers are pruned away or a very large portions of the networks are pruned away, it may not be possible to recover the original accuracy. Iterative pruning and retraining may yield better results, but the iterative process requires many more epochs especially for very deep networks.

# 4 EXPERIMENTS

We prune two types of networks: simple CNNs (VGG-16 on CIFAR-10) and Residual networks (ResNet-56/110 on CIFAR-10 and ResNet-34 on ImageNet). Unlike AlexNet that is often used to demonstrate efficiency speedups, both VGG and Residual networks have fewer parameters in the fully connected layers. Hence, pruning a large percentage of parameters from these networks is challenging. We implement our filter pruning methods in Torch. When filters are pruned, a new model with fewer filters is created and the remaining parameters of the modified layers as well as unaffected layers are copied into a new model. Furthermore, if a convolutional layer is pruned, the weights of the subsequent batch normalization layer are also removed. To get the baseline accuracies for each network, we train each model from scratch and follow the same pre-processing and hyper-parameters as ResNet. For retraining, we use a constant learning rate 0.001 and retrain 40 epochs for CIFAR-10 and 20 epochs for ImageNet, which represents one-fourth of the original training epochs. Past work has reported up to 3X original training times to retrain pruned networks Han et al. (2015).

Table 1: Overall results. The best test/validation accuracy during the retraining process is reported. Training a pruned model from scratch performs worse than retraining a pruned model, which may indicate the difficulty of training a network with a small capacity.  

<table><tr><td>Model</td><td>Error(%)</td><td>FLOP</td><td>Pruned %</td><td>Parameters</td><td>Pruned %</td></tr><tr><td>VGG-16</td><td>6.75</td><td>3.13 × 108</td><td></td><td>1.5 × 107</td><td></td></tr><tr><td>VGG-16-pruned-A</td><td>6.60</td><td>2.06 × 108</td><td>34.2%</td><td>5.4 × 106</td><td>64.0%</td></tr><tr><td>VGG-16-pruned-A scratch-train</td><td>6.88</td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet-56</td><td>6.96</td><td>1.25 × 108</td><td></td><td>8.5 × 105</td><td></td></tr><tr><td>ResNet-56-pruned-A</td><td>6.90</td><td>1.12 × 108</td><td>10.4%</td><td>7.7 × 105</td><td>9.4%</td></tr><tr><td>ResNet-56-pruned-B</td><td>6.94</td><td>9.09 × 107</td><td>27.6%</td><td>7.3 × 105</td><td>13.7%</td></tr><tr><td>ResNet-56-pruned-B scratch-train</td><td>8.69</td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet-110</td><td>6.47</td><td>2.53 × 108</td><td></td><td>1.72 × 106</td><td></td></tr><tr><td>ResNet-110-pruned-A</td><td>6.45</td><td>2.13 × 108</td><td>15.9%</td><td>1.68 × 106</td><td>2.3%</td></tr><tr><td>ResNet-110-pruned-B</td><td>6.70</td><td>1.55 × 108</td><td>38.6%</td><td>1.16 × 106</td><td>32.4%</td></tr><tr><td>ResNet-110-pruned-B scratch-train</td><td>7.06</td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet-34</td><td>26.77</td><td>3.64 × 109</td><td></td><td>2.16 × 107</td><td></td></tr><tr><td>ResNet-34-pruned-A</td><td>27.44</td><td>3.08 × 109</td><td>15.5%</td><td>1.99 × 107</td><td>7.6%</td></tr><tr><td>ResNet-34-pruned-B</td><td>27.83</td><td>2.76 × 109</td><td>24.2%</td><td>1.93 × 107</td><td>10.8%</td></tr><tr><td>ResNet-34-pruned-C</td><td>27.52</td><td>3.37 × 109</td><td>7.5%</td><td>2.01 × 107</td><td>7.2%</td></tr></table>

# 4.1 VGG-16 ON CIFAR-10

VGG-16 is a large capacity network originally designed for the ImageNet dataset by Simonyan & Zisserman (2015). Recent work Zagoruyko (2015) has found that a slightly modified version produces state of the art results on the CIFAR-10 dataset. As shown in Table 2, VGG-16 on CIFAR-10 consists of 13 convolutional layers and 2 fully connected layers, in which the fully connected layers do not occupy large portions of parameters due to the small input size and less hidden units. We use the model described in Zagoruyko (2015) but add Batch Normalization Ioffe & Szegedy (2015) after each convolutional layer and the first linear layer, without using Dropout Srivastava et al. (2014). Note that when the last convolutional layer is pruned, the input to the linear layer is changed and the connections are also removed.

As seen in Figure 2, each of the convolutional layers with 512 feature maps can drop at least  $60\%$  filters without affecting the accuracy. Figure 2(c) shows that with retraining, almost  $90\%$  of the filters of these layers can be safely removed. One explanation is that these filters operate on  $4 \times 4$  or  $2 \times 2$

Table 2: VGG-16 on CIFAR-10 and the pruned model. The last two columns show the number of feature maps and the reduced percentage of FLOP from the pruned model.  

<table><tr><td>layer type</td><td>wi × hi</td><td>#Maps</td><td>FLOP</td><td>#Params</td><td>#Maps</td><td>FLOP%</td></tr><tr><td>Conv_1</td><td>32 × 32</td><td>64</td><td>1.8E+06</td><td>1.7E+03</td><td>32</td><td>50%</td></tr><tr><td>Conv_2</td><td>32 × 32</td><td>64</td><td>3.8E+07</td><td>3.7E+04</td><td>64</td><td>50%</td></tr><tr><td>Conv_3</td><td>16 × 16</td><td>128</td><td>1.9E+07</td><td>7.4E+04</td><td>128</td><td>0%</td></tr><tr><td>Conv_4</td><td>16 × 16</td><td>128</td><td>3.8E+07</td><td>1.5E+05</td><td>128</td><td>0%</td></tr><tr><td>Conv_5</td><td>8 × 8</td><td>256</td><td>1.9E+07</td><td>2.9E+05</td><td>256</td><td>0%</td></tr><tr><td>Conv_6</td><td>8 × 8</td><td>256</td><td>3.8E+07</td><td>5.9E+05</td><td>256</td><td>0%</td></tr><tr><td>Conv_7</td><td>8 × 8</td><td>256</td><td>3.8E+07</td><td>5.9E+05</td><td>256</td><td>0%</td></tr><tr><td>Conv_8</td><td>4 × 4</td><td>512</td><td>1.9E+07</td><td>1.2E+06</td><td>256</td><td>50%</td></tr><tr><td>Conv_9</td><td>4 × 4</td><td>512</td><td>3.8E+07</td><td>2.4E+06</td><td>256</td><td>75%</td></tr><tr><td>Conv_10</td><td>4 × 4</td><td>512</td><td>3.8E+07</td><td>2.4E+06</td><td>256</td><td>75%</td></tr><tr><td>Conv_11</td><td>2 × 2</td><td>512</td><td>9.4E+06</td><td>2.4E+06</td><td>256</td><td>75%</td></tr><tr><td>Conv_12</td><td>2 × 2</td><td>512</td><td>9.4E+06</td><td>2.4E+06</td><td>256</td><td>75%</td></tr><tr><td>Conv_13</td><td>2 × 2</td><td>512</td><td>9.4E+06</td><td>2.4E+06</td><td>256</td><td>75%</td></tr><tr><td>Linear</td><td>1</td><td>512</td><td>2.6E+05</td><td>2.6E+05</td><td>512</td><td>50%</td></tr><tr><td>Linear</td><td>1</td><td>10</td><td>5.1E+03</td><td>5.1E+03</td><td>10</td><td>0%</td></tr><tr><td>Total</td><td></td><td></td><td>3.1E+08</td><td>1.5E+07</td><td></td><td>34%</td></tr></table>

feature maps, which may have no meaningful spatial connections in such small dimensions. Even ResNets, over CIFAR-10 do not perform any additional convolutions for smaller feature maps below  $8 \times 8$  dimensions. Unlike previous work Zeiler & Fergus (2014); Han et al. (2015), we find that the first layer is quite robust to pruning as compared to the next few layers. This is possible because even when  $80\%$  of the filters of the first layer are pruned, there are still 12 filters remaining. This number is larger than the number of raw input channels. However, when removing  $80\%$  filters of the second layer, the layer input corresponds to a 64 to 12 mapping, which may lose significant information from previous layers hurting the accuracy. With  $50\%$  of the filters being pruned in layer 1 and from 8 to 13, we achieve  $34\%$  FLOP reduction for the same original accuracy.

# 4.2 RESNET-56/110 ON CIFAR-10

ResNets for CIFAR-10 have three stages of residual blocks for feature maps with the sizes of  $32 \times 32$ ,  $16 \times 16$  and  $8 \times 8$ . Each stage has the same number of residual blocks. When the feature maps increase, the shortcut layer provides an identity mapping with an additional zero padding for the increased dimensions. As there is no projection mapping for choosing the identity feature maps, we only consider pruning the first layer of the residual block. As shown in Figure 5, most of the layers are robust to pruning. For ResNet-110, pruning some single layers without retraining even improves the performance. In addition, layers that are sensitive to pruning (layer 20, 38 and 54 for ResNet-56, layer 36, 38 and 74 for ResNet-110) lie at the residual blocks close to the layers where the number of feature maps change, e.g., the first and the last residual blocks for each stage. We believe this happens because the precise residual errors are necessary for the new added empty feature maps.

The retraining performance can be improved by skipping these sensitive layers. As shown in Table 1, ResNet-56-pruned-A improves the performance by pruning  $10\%$  filters while skipping the sensitive layers 16, 20, 38 and 54. In addition, we find that the deeper layers are more sensitive to pruning than ones in the earlier stages of the network. Hence, we use a different pruning rate for each stage. We use  $p_i$  to denote the pruning rate for layers in the  $i$ th stage. ResNet-56-pruned-B skips more layers (16, 18, 20, 34, 38, 54) and prunes layers with  $p_1 = 60\%$ ,  $p_2 = 30\%$  and  $p_3 = 10\%$ . For ResNet-110, the first pruned model gets a slightly better result with  $p_1 = 50$  and layer 36 skipped. ResNet-110-pruned-B skips layer 36, 38, 74 and prunes with  $p_1 = 50\%$ ,  $p_2 = 40\%$  and  $p_3 = 30$ . When there are more than two residual blocks at each stage, the middle residual blocks may be redundant and be easily pruned. This might explain why ResNet-110 is easier to prune than ResNet-56.

# 4.3 RESNET-34 ON ILSVRC2012

ResNets for ImageNet have four stages of residual blocks for feature maps with sizes of  $56 \times 56$ ,  $28 \times 28$ ,  $14 \times 14$  and  $7 \times 7$ . ResNet-34 on ImageNet uses the projection shortcut when the feature maps are down-sampled. We first prune the first layer of each residual block. Figure 6 shows the

![](images/102d0d143a48aa635099a262a599f20b26459aa4f370687786317dafb51b0fc8.jpg)

![](images/79b30f03d743cdcf1182320cd1a473e61c07fb9de06c6a6d1579890d2f10620e.jpg)

![](images/c303c84026094655ea973758e05ec24e8f4e75548e8342d79aa6f87417e7564c.jpg)

![](images/8bf8e24775a43c41d595ea3aef23eb8579e5102ce137ee9419ba8850fcc0b77f.jpg)  
Figure 5: Sensitivity to pruning for the first layer of each residual block of ResNet-56/110.

![](images/94901e80f9078a13bbea36f820c873bc5874c2aa4acd0187d2fcde14c0b60a83.jpg)

![](images/51526835bdc697b9539601ac90276d34665d79df76fc7704b3f02bac57103340.jpg)

sensitivity of the first layer of each residual block. Similar to ResNet-56/110, the first and the last residual blocks of each stage are more sensitive to pruning than the intermediate blocks (i.e., layer 2, 8, 14, 16, 26, 28, 30, 32). We skip those layers and prune the remaining layers at each stage equally. In Table 1 we compare two configurations of pruning percentages for the first three stages: (A)  $p_1 = 30\%$ ,  $p_2 = 30\%$ ,  $p_3 = 30\%$ ; (B)  $p_1 = 50\%$ ,  $p_2 = 60\%$ ,  $p_3 = 40\%$ . Option-B provides  $24\%$  FLOP reduction with about  $1\%$  loss in accuracy. As seen in ResNet-50/110 results with CIFAR-10, we can predict that ResNet-34 is relatively more difficult to prune as compared to the deeper ResNet-50/110.

We also prune the identity shortcuts and the second convolutional layer of the residual blocks. As these layers have the same number of filters, they are pruned equally. As shown in Figure 6(b), these layers are more sensitive to pruning than the first layers. With retraining, ResNet-34-pruned-C prunes the third stage with  $p_3 = 20\%$  and can only reduce  $7.5\%$  FLOP with  $0.75\%$  loss in accuracy. Therefore, pruning the first layer of the residual block is more effective at reducing the overall FLOP. This finding also correlates with the bottleneck block design for deeper ResNets, which first reduce the dimension of input feature maps for the residual layer and then increase the dimension to match the identity mapping.

![](images/e245e3408b255e6000de5241ebe47f42e9036b336ea521f7dd01a6ab3be348ca.jpg)  
(a) Pruning the first layer of each residual blocks  
Figure 6: Sensitivity to pruning for the residual blocks of ResNet-34.

![](images/91192541ba65ef46328e08efb5f3316913174f2bf1e668fdfde426e6c78248b7.jpg)  
(b) Pruning the second layers of residual blocks

# 5 CONCLUSIONS

Modern CNNs are often over-capacity with large training and inference costs. In this paper we present a method to prune filters with relatively low weight magnitudes to produce CNNs with reduced computation costs without introducing irregular sparsity. It achieves about  $30\%$  reduction in FLOP for VGGNet and deep ResNet without significant loss in the original accuracy. Instead of aggressive pruning with highly specific layer-wise parameters and long-time of iterative retraining, we introduce one-shot pruning and retrain strategy for simplicity and ease of implementation, which is critical for pruning very deep networks. By performing lesion studies on very deep CNNs, we identify layers that are robust or sensitive to pruning, which can be useful for further understanding and improving the architectures.

# REFERENCES

Sajid Anwar, Kyuyeon Hwang, and Wonyong Sung. Structured Pruning of Deep Convolutional Neural Networks. arXiv preprint arXiv:1512.08571, 2015.  
Matthieu Courbariaux and Yoshua Bengio. Binarynet: Training deep neural networks with weights and activations constrained to + 1 or -1. arXiv preprint arXiv:1602.02830, 2016.  
Misha Denil, Babak Shakibi, Laurent Dinh, Nando de Freitas, et al. Predicting parameters in deep learning. In NIPS, 2013.  
Song Han, Jeff Pool, John Tran, and William Dally. Learning both Weights and Connections for Efficient Neural Network. In NIPS, 2015.  
Song Han, Xingyu Liu, Huizi Mao, Jing Pu, Ardavan Pedram, Mark A Horowitz, and William J Dally. EIE: Efficient Inference Engine on Compressed Deep Neural Network. arXiv preprint arXiv:1602.01528, 2016a.  
Song Han, Huizi Mao, and William J Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. *ICLR*, 2016b.  
Babak Hassibi and David G Stork. Second Order Derivatives for Network Pruning: Optimal Brain Surgeon. In NIPS, 1993.  
Kaiming He and Jian Sun. Convolutional Neural Networks at Constrained Time Cost. In CVPR, 2015.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. arXiv preprint arXiv:1512.03385, 2015.  
Forrest Iandola, Matthew Moskewicz, Khalidand Ashraf, Song Han, William Dally, and Keutzer Kurt. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and ; 1MB model size. arXiv preprint arXiv:1602.07360, 2016.  
Yani Ioannou, Duncan Robertson, Jamie Shotton, Roberto Cipolla, and Antonio Criminisi. Training cnns with low-rank filters for efficient image classification. arXiv preprint arXiv:1511.06744, 2015.  
Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv preprint arXiv:1502.03167, 2015.  
Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman. Speeding up convolutional neural networks with low rank expansions. In BMVC, 2014.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet Classification with Deep Convolutional Neural Networks. In NIPS, 2012.  
Andrew Lavin and Scott Gray. Fast Algorithms for Convolutional Neural Networks. CVPR, 2016.  
Yann Le Cun, John S Denker, and Sara A Solla. Optimal brain damage. In NIPS, 1989.

Vadim Lebedev and Victor Lempitsky. Fast Convnets Using Group-wise Brain Damage. CVPR, 2016.  
Min Lin, Qiang Chen, and Shuicheng Yan. Network in Network. arXiv preprint arXiv:1312.4400, 2013.  
Baoyuan Liu, Min Wang, Hassan Foroosh, Marshall Tappen, and Marianna Pensky. Sparse convolutional neural networks. In CVPR, 2015.  
Zelda Mariet and Suvrit Sra. Diversity Networks. In ICLR, 2016.  
Michael Mathieu, Mikael Henaff, and Yann LeCun. Fast Training of Convolutional Networks through FFTs. arXiv preprint arXiv:1312.5851, 2013.  
Adam Polyak and Lior Wolf. Channel-Level Acceleration of Deep Face Representations. IEEE Access, 2015.  
Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks. arXiv preprint arXiv:1603.05279, 2016.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.  
Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*, 2015.  
Suraj Srinivas and R Venkatesh Babu. Data-free parameter pruning for deep neural networks. BMVC, 2015.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 2014.  
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going Deeper with Convolutions. In CVPR, 2015a.  
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the Inception Architecture for Computer Vision. arXiv preprint arXiv:1512.00567, 2015b.  
Cheng Tai, Tong Xiao, Xiaogang Wang, et al. Convolutional neural networks with low-rank regularization. *ICLR*, 2016.  
Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Learning Structured Sparsity in Deep Learning. In NIPS, 2016.  
Sergey Zagoruyko.  $92.45\%$  on CIFAR-10 in Torch. http://torch.ch/blog/2015/07/30/cifar.html, 2015.  
Matthew D Zeiler and Rob Fergus. Visualizing and Understanding Convolutional Networks. In ECCV, 2014.  
X Zhang, J Zou, K He, and J Sun. Accelerating very deep convolutional networks for classification and detection. IEEE transactions on pattern analysis and machine intelligence, 2015a.  
Xiangyu Zhang, Jianhua Zou, Xiang Ming, Kaiming He, and Jian Sun. Efficient and accurate approximations of nonlinear convolutional networks. In CVPR, 2015b.  
Hao Zhou, Jose Alvarez, and Fatih Porikli. Less Is More: Towards Compact CNNs. In ECCV, 2016.