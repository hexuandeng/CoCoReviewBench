# ABS: AUTOMATIC BIT SHARING FOR MODEL COMPRESSION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present Automatic Bit Sharing (ABS) to automatically search for optimal model compression configurations (e.g., pruning ratio and bitwidth). Unlike previous works that consider model pruning and quantization separately, we seek to optimize them jointly. To deal with the resultant large designing space, we propose a novel super-bit model, a single-path method, to encode all candidate compression configurations, rather than maintaining separate paths for each configuration. Specifically, we first propose a novel decomposition of quantization that encapsulates all the candidate bitwidths in the search space. Starting from a low bitwidth, we sequentially consider higher bitwidths by recursively adding re-assignment offsets. We then introduce learnable binary gates to encode the choice of bitwidth, including 0-bit for pruning. By jointly training the binary gates in conjunction with network parameters, the compression configurations of each layer can be automatically determined. Our ABS brings two benefits for model compression: 1) It avoids the combinatorially large design space, with a reduced number of trainable parameters and search costs. 2) It also averts directly fitting an extremely low bit quantizer to the data, hence greatly reducing the optimization difficulty due to the non-differentiable quantization. Experiments on CIFAR-100 and ImageNet show that our methods achieve significant computational cost reduction while preserving promising performance.

# 1 INTRODUCTION

Deep neural networks (DNNs) have achieved great success in many challenging computer vision tasks, including image classification (Krizhevsky et al., 2012; He et al., 2016) and object detection (Lin et al., 2017a,b). However, a deep model usually has a large number of parameters and consumes huge amounts of computational resources, which remains great obstacles for many applications, especially on resource-limited devices with limited memory and computational resources, such as smartphones. To reduce the number of parameters and computational overhead, many methods (He et al., 2019; Zhou et al., 2016) have been proposed to conduct model compression by removing the redundancy while maintaining the performance.

In the last decades, we have witnessed a lot of model compression methods, such as network pruning (He et al., 2017; 2019) and quantization (Zhou et al., 2016; Hubara et al., 2016). Specifically, network pruning reduces the model size and computational costs by removing redundant modules while network quantization maps the full-precision values to low-precision ones. It has been shown that sequentially perform network pruning and quantization is able to get a compressed network with small model size and lower computational overhead (Han et al., 2016). However, performing pruning and quantization in a separate step may lead to sub-optimal results. For example, the best quantization strategy for the uncompressed network is not necessarily the optimal one after network pruning. Therefore, we need to consider performing pruning and quantization simultaneously.

Recently, many attempts have been made to automatically determine the compression configurations of each layer (i.e., pruning ratios, and/or bitwidths), either based on reinforcement learning (RL) (Wang et al., 2019), evolutionary search (ES) (Wang et al., 2020), Bayesian optimization (BO) (Tung & Mori, 2018) or differentiable methods (Wu et al., 2018; Dong & Yang, 2019). In particular, previous differentiable methods formulate model compression as a differentiable searching problem to explore the search space using gradient-based optimization. As shown in Figure 1(a), each candi-

![](images/24456b5bcd58ab9696f24929291904b3ce38748168b2aea0ebb44ba94a19a5e1.jpg)  
(a) Multi-path scheme (Wu et al., 2018)

$$
\hat {z} = \alpha_ {2} ^ {q} z _ {2} + \dots + \alpha_ {3 2} ^ {q} z _ {3 2}
$$

![](images/a382a8d25bdb2d497510ee2dfea00575d7c85868dca40ccc0734325a70d7d01f.jpg)  
(b) Single-path scheme (Ours)  
Figure 1: Multi-path v.s. single-path compression scheme. (a) Multi-path search scheme (Wu et al., 2018): each candidate operation is represented as a separate path, which gives rise to huge numbers of trainable parameters and high computational overhead. Here,  $\alpha_{k}^{q}$  is the architecture parameters corresponding to the path of  $k$  bit quantization. (b) Single-path search scheme (Ours): different candidate operations share the same super-bit, which greatly reduces the computational costs and optimization difficulty from the discontinuity of quantization. Here,  $g_{k}^{q}$  is a binary gate that controls the decision of bitwidth, and  $\epsilon_{k}$  is the quantized residual error.

$$
\hat {z} = z _ {2} + g _ {4} ^ {q} \epsilon_ {4} + \dots + g _ {3 2} ^ {q} \epsilon_ {3 2}
$$

date operation is maintained as a separate path, which leads to a huge number of trainable parameters and high computational overhead when the search space becomes combinatorially large. Moreover, due to the non-differentiable quantizer and pruning process, the optimization of heavily compressed candidate networks can be more challenging than that in the conventional search problem.

In this paper, we propose a simple yet effective model compression method named Automatic Bit Sharing (ABS) to reduce the search cost and ease the optimization for the compressed candidates. Inspired by recent single-path neural architecture search (NAS) methods (Stamoulis et al., 2019; Guo et al., 2020), the proposed ABS introduces a novel single-path super-bit to encode all effective bitwidths in the search space instead of formulating each candidate operation as a separate path, as shown in Figure 1(b). Specifically, we build upon the observation that the quantized values of a high bitwidth can share the ones of low bitwidths under some conditions. Therefore, we are able to decompose the quantized representation into the sum of the lowest bit quantization and a series of re-assignment offsets. We then introduce learnable binary gates to encode the choice of bitwidth, including 0-bit for pruning. By jointly training the binary gates and network parameters, the compression ratio of each layer can be automatically determined. The proposed scheme has several advantages. First, we only need to solve the search problem as finding which subset of the super-bit to use for each layer's weights and activations rather than selecting from different paths. Second, we enforce the candidate bitwidths to share the quantized values. Hence, we are able to optimize them jointly instead of separately, which greatly reduces the optimization difficulty from the discontinuity of discretization.

Our main contributions are summarized as follows:

- We devise a novel super-bit scheme that encapsulates multiple compression configurations in a unified single-path framework. Relying on the super-bit scheme, we further introduce learnable binary gates to determine the optimal bitwidths (including 0-bit) of each layer's weights and activations. The proposed ABS casts the search problem as subset selection problem, hence significantly reducing the search cost.  
- We formulate the quantized representation as a gated combination of the lowest bitwidth quantization and a series of re-assignment offsets, in which we explicitly share the quantized values between different bitwidths. In this way, we enable the candidate operations to learn jointly rather than separately, hence greatly easing the optimization, especially in the non-differentiable quantization scenario.  
- We evaluate our ABS on CIFAR-100 and ImageNet over various network architectures. Extensive experiments show that the proposed method achieves the state-of-the-art perfor

mance. For example, on ImageNet, our ABS compressed MobileNetV2 achieves  $28.5 \times$  Bit-Operation (BOP) reduction with only  $0.2\%$  performance drop on the Top-1 accuracy.

# 2 RELATED WORK

Network quantization. Network quantization represents the weights, activations and even gradients in low-precision to yield compact DNNs. With low-precision integers or power-of-two representations, the heavy matrix multiplications can be replaced by efficient bitwise operations, leading to much faster test-time inference and lower power consumption. To improve the quantization performance, current methods either focus on designing accurate quantizers by fitting the quantizer to the data (Jung et al., 2019; Zhang et al., 2018; Choi et al., 2018; Cai et al., 2017), or seek to approximate the gradients due to the non-differentiable discretization (Ding et al., 2019; Louizos et al., 2019; Zhuang et al., 2020). Moreover, most previous works assign the same bitwidth for all layers (Zhou et al., 2016; Zhuang et al., 2018a; 2019; Jung et al., 2019; Li et al., 2020; Esser et al., 2020). Though attractive for simplicity, setting a uniform precision places no guarantee on optimizing network performance since different layers have different redundancy and arithmetic intensity. Therefore, several studies proposed mixed-precision quantization (Wang et al., 2019; Dong et al., 2019; Wu et al., 2018; Uhlich et al., 2020) to set different bitwidths according to the redundancy of each layer. In this paper, based on the proposed quantization decomposition, we devise an approach that can effectively learn appropriate bitwidths for each layer through gradient-based optimization.

NAS and pruning. Neural architecture search (NAS) aims to automatically design efficient architectures with low model size and computational costs, either based on reinforcement learning (Pham et al., 2018; Guo et al., 2019), evolutionary search (Real et al., 2019) or gradient-based methods (Liu et al., 2019a). In particular, gradient-based NAS has gained increased popularity, where the search space can be divided into the multi-path design (Liu et al., 2019a; Cai et al., 2019) and single-path formulation (Stamoulis et al., 2019; Guo et al., 2020), depending on whether adding each operation as a separate path or not. While prevailing NAS methods optimize the network topology, the focus of this paper is to search optimal compression ratios for a given architecture. Moreover, network pruning can be treated as fine-grained NAS, which aims at removing redundant modules to accelerate the run-time inference speed, giving rise to methods based on unstructured weight pruning (Han et al., 2016; Guo et al., 2016) or structured channel pruning (He et al., 2017; Zhuang et al., 2018b; Luo et al., 2017). Based on channel pruning, our paper further takes quantization into consideration to generate more compact networks.

AutoML for model compression. Recently, much effort has been put into automatically determining either the optimal pruning rate (Tung & Mori, 2018; Dong & Yang, 2019; He et al., 2018), or the bitwidth (Lou et al., 2019; Cai & Vasconcelos, 2020) of each layer via hyper-parameter search, without relying on heuristics. In particular, HAQ (Wang et al., 2019) employs reinforcement learning to search bitwidth strategies with the hardware accelerator's feedback. Meta-pruning (Liu et al., 2019b) uses meta-learning to generate the weight parameters of the pruned networks and then adopts an evolutionary search algorithm to find the layer-wise sparsity for channel pruning. More recently, several studies (Wu et al., 2018; Cai & Vasconcelos, 2020) have focused on using differentiable schemes via gradient-based optimization.

Closely related methods. To further improve the compression ratio, several methods propose to jointly optimize pruning and quantization strategies. In particular, some works only support weight quantization (Tung & Mori, 2018; Ye et al., 2019) or use fine-grained pruning (Yang et al., 2020). However, the resultant networks cannot be implemented efficiently on edge devices. Recently, several methods (Wu et al., 2018; Wang et al., 2020; Ying et al., 2020) have been proposed to consider filter pruning, weight quantization, and activation quantization jointly. In contrast to these methods, we carefully design the compression search space by sharing the candidate configurations, which significantly reduces the search cost and eases the optimization. Compared with those methods that share the similarities of using quantized residual errors (Chen et al., 2010; Gong et al., 2014; Li et al., 2017b; van Baalen et al., 2020), our proposed method recursively uses quantized residual errors to decompose a quantized representation as a set of candidate bitwidths and parameterize the selection of optimal bitwidth via binary gates.

# 3 PROPOSED METHOD

# 3.1 PRELIMINARY: NORMALIZATION AND QUANTIZATION FUNCTION

Without loss of generality, given a convolutional layer, let  $x$  and  $w$  be the activations of the last layer and its weight parameters, respectively. First, for convenience, following (Choi et al., 2018; Bai et al., 2019), we can normalize  $x$  and  $w$  into scale [0, 1] by  $T_{x}$  and  $T_{w}$ , respectively:

$$
z _ {x} = T _ {x} (x) = \operatorname {c l i p} \left(\frac {x}{v _ {x}}, 0, 1\right), \tag {1}
$$

$$
z _ {w} = T _ {w} (w) = \frac {1}{2} \left(\operatorname {c l i p} \left(\frac {w}{v _ {w}}, - 1, 1\right) + 1\right), \tag {2}
$$

where the function  $\operatorname{clip}(v, v_{\mathrm{low}}, v_{\mathrm{up}}) = \min(\max(v, v_{\mathrm{low}}), v_{\mathrm{up}})$  clips any number  $v$  into the range  $[v_{\mathrm{low}}, v_{\mathrm{up}}]$ , and  $v_x$  and  $v_w$  are trainable quantization intervals which indicate the range of weights and activations to be quantized. Then, we can apply the following function to quantize the normalized activations and parameters, namely  $z_x \in [0, 1]$  and  $z_w \in [0, 1]$ , to discretized ones:

$$
D (z, s) = s \cdot \operatorname {r o u n d} \left(\frac {z}{s}\right), \tag {3}
$$

where round  $(\cdot)$  returns the nearest integer of a given value and  $s$  denotes the normalized step size. Typically, for  $k$ -bit quantization, the normalized step size  $s$  can be computed by

$$
s = \frac {1}{2 ^ {k} - 1}. \tag {4}
$$

After doing the  $k$ -bit quantization, we shall have  $2^k - 1$  quantized values. Specifically, we obtain the quantization  $Q(w)$  and  $Q(x)$  by

$$
Q (w) = T _ {w} ^ {- 1} \left(D \left(z _ {w}, s\right)\right) = v _ {w} \cdot \left(2 \cdot D \left(z _ {w}, s\right) - 1\right), \tag {5}
$$

$$
Q (x) = T _ {x} ^ {- 1} \left(D \left(z _ {x}, s\right)\right) = v _ {x} \cdot D \left(z _ {x}, s\right), \tag {6}
$$

where  $T_w^{-1}$  and  $T_x^{-1}$  denote the inverse functions of  $T_w$  and  $T_x$ , respectively.

# 3.2 BIT SHARING DECOMPOSITION

Previous methods consider different compression configurations as different paths and reformulate model compression as a path selection problem, which gives rise to a huge number of trainable parameters and high computational costs. In this paper, we seek to conduct filter pruning and quantization simultaneously by solving the following problem:

$$
\min  _ {\mathbf {W}, \alpha^ {p}, \alpha^ {q}} \mathcal {L} \left(\mathbf {W}, \alpha^ {p}, \alpha^ {q}\right), \tag {7}
$$

where  $\mathcal{L}(\cdot)$  denotes some losses, and  $\mathbf{W}$  is the parameters of the network.  $\alpha^p$  and  $\alpha^q$  are the pruning and quantization configurations, respectively. As shown in Eq. (7), we propose to encode all compression configurations in a single-path super-bit model (See Figure 1(b)). In the following, we first introduce the bit sharing decomposition and then describe how to learn for compression.

To illustrate the bit sharing decomposition, we begin with an example of 2-bit quantization for  $z \in \{z_x, z_w\}$ . Specifically, we consider using the following equation to quantize  $z$  to 2-bit:

$$
z _ {2} = D (z, s _ {2}), \quad s _ {2} = \frac {1}{2 ^ {2} - 1}, \tag {8}
$$

where  $z_{2}$  and  $s_{2}$  are the quantized value and the step size of 2-bit quantization, respectively. Due to the large step size, the residual error  $z - z_{2} \in [-s_{2}/2, s_{2}/2]$  may be big and result in a significant performance drop. To reduce the residual error, an intuitive way is to use a smaller step size, which indicates that we quantize  $z$  to a higher bitwidth. Since the step size  $s_{4} = 1/(2^{4}-1)$  in 4-bit quantization is a divisor of the step size  $s_{2}$  in 2-bit quantization, the quantized values of 2-bit quantization are among the ones of 4-bit quantization. In fact, based on 2-bit quantization, the 4-bit counterpart introduces additional unshared quantized values. In particular, if  $z_{2}$  has zero residual error, then 4-bit quantization maps  $z$  to the shared quantized values (i.e.,  $z_{2}$ ). In contrast, if  $z_{2}$  is with non-zero

residual error, 4-bit quantization is likely to map  $z$  to the unshared quantized values. In this case, 4-bit quantization can be regarded as performing quantized value re-assignment based on  $z_{2}$ . Such a re-assignment process can be formulated as follows:

$$
z _ {4} = z _ {2} + \epsilon_ {4}, \tag {9}
$$

where  $z_{4}$  is the 4-bit quantized value and  $\epsilon_4$  is the re-assignment offset based on  $z_{2}$ . To ensure that the results of re-assignment fall into the unshared quantized values, the re-assignment offset  $\epsilon_4$  must be an integer multiplying of the 4-bit step size  $s_4$ . Formally,  $\epsilon_4$  can be computed by performing 4-bit quantization on the residual error of  $z_{2}$ :

$$
\epsilon_ {4} = D \left(z - z _ {2}, s _ {4}\right), \quad s _ {4} = \frac {s _ {2}}{2 ^ {2} + 1} = \frac {1}{2 ^ {4} - 1}. \tag {10}
$$

Therefore, according to Eq. (9), a 4-bit quantized value can be decomposed into the 2-bit representation and its re-assignment offset. Similarly, an 8-bit quantized value can also be decomposed into the 4-bit representation and its corresponding re-assignment offset. In this way, we can generalize the idea of decomposition to arbitrary effective bitwidths as follows.

Definition 1 (Quantization decomposition) Let  $z \in [0,1]$  be a normalized full-precision input,  $\{b_1, \dots, b_K\}$  be a sequence of candidate bitwidths, and  $b_1 < b_2, \dots, < b_{K-1} < b_K$ . We use the following quantized  $\widehat{z}$  to approximate  $z$ :

$$
\widehat {z} = z _ {b _ {1}} + \sum_ {j = 2} ^ {K} \epsilon_ {b _ {j}}, \quad \text {w h e r e} \epsilon_ {b _ {j}} = D \left(z - z _ {b _ {j - 1}}, s _ {b _ {j}}\right), \quad s _ {b _ {j}} = \frac {s _ {b _ {j - 1}}}{2 ^ {b _ {j - 1}} + 1} = \frac {1}{2 ^ {b _ {j}} - 1}. \tag {11}
$$

In other words, the quantized approximation  $\widehat{z}$  can be decomposed into the sum of the lowest bit quantization and a series of recursive re-assignment offsets. In Proposition (1), to enable quantized value re-assignment, we need to constrain that  $s_{b_{j - 1}}$  is divisible by  $s_{b_j}$ , which requires the bitwidths  $b_{j}(j > 1)$  to satisfy the following relation:

$$
b _ {j} = 2 ^ {j - 1} \cdot b _ {1}. \tag {12}
$$

In fact, the bitwidth  $b_{1}$  can be set to arbitrary appropriate integer values (e.g., 1, 2, 3, etc.). To get a hardware-friendly compressed network<sup>1</sup>, we set  $b_{1}$  to 2, which ensures that all the decomposition bitwidths are power-of-two. Moreover, since 8-bit quantization achieves lossless performance compared with the full-precision counterpart (Zhou et al., 2016), we only consider those candidate bitwidths that are not greater than 8-bit. In other words, we constrain the value of  $j$  to [1, 3].

Remark 1 The proposed bit sharing decomposition has several advantages. First, the proposed method only needs to maintain a small number of trainable parameters, which greatly reduces the computational costs during search. Second, we are able to directly extract a low-precision representation from its higher precision, which allows optimizing different bitwidths jointly and ease the discontinuous optimization due to quantization.

# 3.3 LEARNING FOR COMPRESSION

Note that different layers have different levels of redundancy, which indicates that different layers may choose different subsets of the quantized values. To learn the quantized approximation for each layer, we introduce a layer-wise binary quantization gate  $g_{b_j}^q \in \{0,1\}$  on each of the re-assignment offsets in Eq. (11) to encode the choice of the quantization bitwidth, which can be formulated as

$$
g _ {b _ {j}} ^ {q} = \mathbb {1} \left(\left| \left| z - z _ {b _ {j} - 1} \right| \right| > \alpha_ {b _ {j}} ^ {q}\right), \tag {13}
$$

$$
\widehat {z} = z _ {b _ {1}} + g _ {b _ {2}} ^ {q} \big (\epsilon_ {b _ {2}} + \dots + g _ {b _ {K - 1}} ^ {q} \big (\epsilon_ {b _ {K - 1}} + g _ {b _ {K}} ^ {q} \epsilon_ {b _ {K}} \big) \big),
$$

where  $\alpha_{b_j}^q$  is a layer-wise threshold that controls the choice of bitwidth, and  $\mathbb{1}(\cdot)$  is the indicator function. Specifically, if the quantization error  $\| z - z_{b_{j-1}}\|$  is greater than the threshold  $\alpha_{b_j}^q$ , we activate the corresponding quantization gate to increase the bitwidth so that the residual error can be reduced, and vice versa.

Note that from Eq. (13), we can consider the filter pruning as 0-bit quantization. To avoid the prohibitively large filter-wise search space, we propose to divide the filters into groups based on indexes and consider the group-wise space instead. To be specific, we introduce a binary gate  $g_{c}^{p}$  for each group to encode the choice of pruning, which can be formulated as follows:

$$
g _ {c} ^ {p} = \mathbb {1} \left(\left| \left| w _ {c} \right| \right| > \alpha^ {p}\right),
$$

$$
\widehat {z} _ {c} = g _ {c} ^ {p} \cdot \left(z _ {c, b _ {1}} + g _ {b _ {2}} ^ {q} \left(\epsilon_ {c, b _ {2}} + \dots + g _ {b _ {K - 1}} ^ {q} \left(\epsilon_ {c, b _ {K - 1}} + g _ {b _ {K}} ^ {q} \epsilon_ {c, b _ {K}}\right)\right)\right), \tag {14}
$$

where  $\widehat{z}_c$  is the  $c$ -th group of quantized filters and  $\epsilon_{c,b_j}$  is the corresponding re-assignment offset by quantizing the residual error  $z_c - z_{c,b_{j-1}}$ . Here,  $\alpha^p$  is a layer-wise threshold for filter pruning. Following PFEC (Li et al., 2017a), we use  $\ell_1$ -norm to evaluate the importance of the filter. Specifically, if a group of filters are important, the corresponding pruning gate will be activated and vice versa.

Note that both quantization and pruning have their corresponding thresholds. Instead of manually setting the thresholds, we propose to learn them via gradient descent. However, the indicator function in Eq. (13) is non-differentiable. To address this, we propose to approximate the gradient of the indicator function using the gradient of the sigmoid function  $\sigma(\cdot)$ , which can be formulated as:

$$
\frac {\partial g}{\partial \alpha} \approx \frac {\partial \sigma (A - \alpha)}{\partial \alpha} = - \sigma (A - \alpha) (1 - \sigma (A - \alpha)), \tag {15}
$$

where  $g$  is the output of a binary gate,  $\alpha \in \{\alpha^p, \alpha^q\}$  is the corresponding threshold and  $A$  denotes some specific metrics (i.e.,  $\ell_1$ -norm of the filter or the quantization error). By jointly training the binary gates and the network parameters, the pruning ratio and bitwidth of each layer can be automatically determined. However, the gradient approximation of the binary gate inevitably introduces noisy signals, which can be even more severe when we quantize both weights and activations. Thus, we propose to train the binary gates of weights and activations in an alternative manner. Specifically, when training the binary gates of weights, we fix the binary gates of activations, and vice versa.

Search Space for Model Compression. Given an uncompressed network with  $L$  layers, we use  $C_l$  to denote the number of filters at the  $l$ -th layer. To obtain the compressed model, we first divide the filters of each layer into groups and then search for their optimal bitwidths separately. Let  $B$  be the number of filters in a group. For any layer  $l$ , there would be  $\left\lfloor \frac{C_l}{B} \right\rfloor$  groups in total. Since we quantize both weights and activations, given  $K$  candidate bitwidths, there are  $K^2$  different quantization configurations for each filter group. Thus, for the whole network with  $L$  layers, the size of the search space  $\Omega$  can be computed by

$$
| \Omega | = \prod_ {l = 1} ^ {L} \left(K ^ {2} \times \left\lfloor \frac {C _ {l}}{B} \right\rfloor\right). \tag {16}
$$

Eq. (16) indicates that the search space is large enough to cover the potentially good configurations.

Training Objective. To design a hardware-efficient network, the objective function in Eq. (7) should reflect both the accuracy of the compressed network and its computational costs. Following (Cai et al., 2019), we train the network and architecture by minimizing following loss function:

$$
\mathcal {L} (\mathbf {W}, \alpha^ {p}, \alpha^ {q}) = \mathcal {L} _ {c e} (\mathbf {W}, \alpha^ {p}, \alpha^ {q}) + \lambda \log R (\mathbf {W}, \alpha^ {p}, \alpha^ {q}), \tag {17}
$$

where  $\mathcal{L}_{ce}(\cdot)$  is the cross-entropy loss,  $R(\cdot)$  is the computational costs of the network and  $\lambda$  is a balancing hyper-parameter. Following single-path NAS (Stamoulis et al., 2019), we use a similar formulation of computational costs to preserve the differentiability of the objective function. The details of the differentiable computational loss can be found in Appendix B. Once the training is finished, we can obtain the compressed network by selecting those filters and bitwidths with activated binary gates. Then, we fine-tune the compressed network to compensate the accuracy loss.

# 4 EXPERIMENTS

Compared methods. To investigate the effectiveness of the proposed method, we consider the following methods for comparisons: ABS: our proposed method with joint pruning and quantization; ABS-Q: ABS with quantization only; ABS-P: ABS with pruning only; and several state-of-the-art model compression methods including HAQ (Wang et al., 2019), DQ (Uhlich et al., 2020), DJPQ (Ying et al., 2020) and DNAS (Wu et al., 2018). We measure the performance of different

Table 1: Comparisons of different methods on CIFAR-100.  

<table><tr><td>Network</td><td>Method</td><td>Top-1 Acc. (%)</td><td>Top-5 Acc. (%)</td><td>BOPs (M)</td><td>BOP comp. ratio</td></tr><tr><td rowspan="8">ResNet-20</td><td>Full-precision</td><td>67.5</td><td>90.8</td><td>41798.6</td><td>1.0</td></tr><tr><td>4-bit precision</td><td>67.8±0.3</td><td>90.4±0.2</td><td>674.6</td><td>62.0</td></tr><tr><td>DQ</td><td>67.7±0.6</td><td>90.4±0.5</td><td>1180.0</td><td>35.4</td></tr><tr><td>HAQ</td><td>67.7±0.1</td><td>90.4±0.3</td><td>653.4</td><td>64.0</td></tr><tr><td>DNAS</td><td>67.8±0.3</td><td>90.4±0.2</td><td>664.2</td><td>62.9</td></tr><tr><td>ABS-P (Ours)</td><td>67.9±0.1</td><td>90.7±0.2</td><td>28586.5</td><td>1.5</td></tr><tr><td>ABS-Q (Ours)</td><td>68.1±0.1</td><td>90.5±0.0</td><td>649.5</td><td>64.4</td></tr><tr><td>ABS (Ours)</td><td>68.1±0.3</td><td>90.6±0.2</td><td>630.6</td><td>66.3</td></tr><tr><td rowspan="8">ResNet-56</td><td>Full-precision</td><td>71.7</td><td>92.2</td><td>128771.7</td><td>1.0</td></tr><tr><td>4-bit precision</td><td>70.9±0.3</td><td>91.2±0.4</td><td>2033.6</td><td>63.3</td></tr><tr><td>DQ</td><td>70.7±0.2</td><td>91.4±0.4</td><td>2222.9</td><td>57.9</td></tr><tr><td>HAQ</td><td>71.2±0.1</td><td>91.1±0.2</td><td>2014.9</td><td>63.9</td></tr><tr><td>DNAS</td><td>71.2±0.2</td><td>91.3±0.3</td><td>1996.9</td><td>65.3</td></tr><tr><td>ABS-P (Ours)</td><td>71.5±0.1</td><td>91.8±0.2</td><td>87021.6</td><td>1.5</td></tr><tr><td>ABS-Q (Ours)</td><td>71.5±0.2</td><td>91.5±0.2</td><td>1970.7</td><td>65.3</td></tr><tr><td>ABS (Ours)</td><td>71.6±0.1</td><td>91.8±0.4</td><td>1918.8</td><td>67.1</td></tr></table>

![](images/4148d3a53f69ced05c82667cfd03a3a009ae86a919e00d8297cd9f2a424c2fed.jpg)  
Figure 2: Top-1 accuracy results of different compressed networks with different BOPs.

methods in terms of the Top-1 and Top-5 accuracy. Following (Guo et al., 2020; Ying et al., 2020), we measure the computational costs by the Bit-Operation (BOP) count. The BOP compression ratio is defined as the ratio between the total BOPs of the uncompressed and compressed models.

Implementation details. Following HAQ (Wang et al., 2019), we quantize all the layers, in which the first and the last layers are quantized to 8-bit. Following ThiNet (Luo et al., 2017), we only conduct filter pruning for the first layer in the residual block. For ResNet-20 and ResNet-56 on CIFAR-100 (Krizhevsky et al., 2009), we set  $B$  to 4. For ResNet-18 and MobileNetV2 on ImageNet (Russakovsky et al., 2015),  $B$  is set to 16 and 8, respectively. We first train the full-precision models and then use the pretrained weights to initialize the compressed models. Following (Li et al., 2020; Esser et al., 2020), we introduce weight normalization during training. We use SGD with nesterov (Nesterov, 1983) for optimization, with a momentum of 0.9. For CIFAR-100, we use the same data augmentation as in (He et al., 2016), including translation and horizontal flipping. For ImageNet, images are resized to  $256 \times 256$ , and then a  $224 \times 224$  patch is randomly cropped from an image or its horizontal flip for training. For testing, a  $224 \times 224$  center cropped is chosen. We first train the uncompressed network for 30 epochs on CIFAR-100 and 10 epochs on ImageNet. The learning rate is set to 0.001. We then fine-tune the searched compressed network to recover the performance drop. On CIFAR-100, we train the searched network for 200 epochs with a mini-batch size of 128. The learning rate is initialized to 0.1 and is divided by 10 at 80-th and 120-th epochs. Experiments on CIFAR-100 are repeated for 5 times and we report the mean and standard deviation. For ResNet-18 on ImageNet, we finetune the searched network for 90 epochs with a mini-batch size of 256. For MobileNetV2 on ImageNet, we fine-tune for 150 epochs. For all models on ImageNet, the learning rate starts at 0.01 and decays with cosine annealing (Loshchilov & Hutter, 2017).

# 4.1 MAIN RESULTS

We apply the proposed methods to compress ResNet-20, ResNet-56 on CIFAR-100 and ResNet-18, MobileNetV2 on ImageNet. We compare the performance of different methods in Table 1 and Table 2. We also show the Top-1 accuracy results of the compressed ResNet-56 with different BOPs in Figure 2. From the results, we can see that 4-bit quantized networks achieve lossless performance. Also, 6-bit MobileNetV2 only leads to a  $0.1\%$  performance drop on the Top-1 Accuracy. Compared with fixed-precision quantization, mixed-precision methods are able to reduce the BOPs

Table 2: Comparisons on ImageNet. " -" denotes that the results are not reported.  

<table><tr><td>Network</td><td>Method</td><td>Top-1 Acc. (%)</td><td>Top-5 Acc. (%)</td><td>BOPs (G)</td><td>BOP comp. ratio</td></tr><tr><td rowspan="7">ResNet-18</td><td>Full-precision</td><td>70.7</td><td>89.8</td><td>1857.6</td><td>1.0</td></tr><tr><td>4-bit precision</td><td>71.0</td><td>89.8</td><td>34.7</td><td>53.5</td></tr><tr><td>DQ</td><td>68.5</td><td>-</td><td>40.7</td><td>40.6</td></tr><tr><td>DJQ</td><td>69.1</td><td>-</td><td>35.5</td><td>52.3</td></tr><tr><td>HAQ</td><td>70.2</td><td>89.5</td><td>34.7</td><td>53.5</td></tr><tr><td>ABS-Q (Ours)</td><td>70.9</td><td>89.7</td><td>33.1</td><td>56.1</td></tr><tr><td>ABS (Ours)</td><td>70.8</td><td>89.6</td><td>32.3</td><td>57.5</td></tr><tr><td rowspan="6">MobileNetV2</td><td>Full-precision</td><td>71.9</td><td>90.3</td><td>308.0</td><td>1.0</td></tr><tr><td>6-bit precision</td><td>71.8</td><td>90.3</td><td>11.2</td><td>27.5</td></tr><tr><td>DQ</td><td>70.4</td><td>89.7</td><td>158.5</td><td>1.9</td></tr><tr><td>HAQ</td><td>71.2</td><td>90.0</td><td>10.8</td><td>28.5</td></tr><tr><td>ABS-Q (Ours)</td><td>71.8</td><td>90.4</td><td>10.9</td><td>28.3</td></tr><tr><td>ABS (Ours)</td><td>71.7</td><td>90.3</td><td>10.8</td><td>28.5</td></tr></table>

while preserving the performance. Critically, our proposed ABS-Q outperforms the state-of-the-arts baselines with less computational costs. Specifically, ABS-Q compressed ResNet-18 outperforms the one compressed by HAQ with more BOPs reduction. Moreover, by combing pruning and quantization, ABS achieves nearly lossless performance while further reducing the computational costs of ABS-Q. For example, ABS compressed ResNet-18 reduces the BOPs by  $57.5 \times$  while still outperforming the full-precision network by  $0.1\%$  in terms of the Top-1 accuracy on ImageNet.

# 4.2 ABLATION STUDIES

Effect of the bit-sharing scheme. To investigate the effect of the bit-sharing scheme, we apply our methods to quantize ResNet-20 and ResNet-56 with and without the bit sharing scheme on CIFAR100. We report the testing accuracy and BOPs in Table 3. We also present the search costs and consumed GPU memory measured on a GPU device (NVIDIA TITAN Xp). It can be seen from the results that the method with the bit sharing scheme consistently outperforms the ones without the bit sharing scheme while significantly reducing the search costs and GPU memory.

Table 3: Effect of the bit-sharing scheme. We report the testing accuracy, BOPs, and search costs on CIFAR-100. The search costs are measured on a GPU device (NVIDIA TITAN Xp).  

<table><tr><td>Network</td><td>Method</td><td>Top-1 Acc.</td><td>Top-5 Acc.</td><td>BOPs (M)</td><td>Search Cost (GPU hours)</td><td>GPU Memory (GB)</td></tr><tr><td rowspan="2">ResNet-20</td><td>w/o bit sharing</td><td>67.8±0.1</td><td>90.5±0.2</td><td>664.2</td><td>2.8</td><td>4.4</td></tr><tr><td>w/ bit sharing</td><td>68.1±0.1</td><td>90.5±0.0</td><td>649.5</td><td>0.8</td><td>1.5</td></tr><tr><td rowspan="2">ResNet-56</td><td>w/o bit sharing</td><td>71.3±0.3</td><td>91.4±0.4</td><td>2001.1</td><td>8.7</td><td>10.9</td></tr><tr><td>w/ bit sharing</td><td>71.5±0.2</td><td>91.5±0.2</td><td>1970.7</td><td>1.9</td><td>3.0</td></tr></table>

Effect of the one-stage compression. To investigate the effect of the one-stage compression scheme (perform pruning and quantization jointly), we extend ABS to two-stage optimization, where we sequentially do filter pruning and quantization, denoted as  $\mathrm{ABS - P\rightarrow ABS - Q}$ . The results are shown in Table 4. Compared with the two-stage counterpart, ABS achieves better performance with less computational costs, which shows the superiority of the one-stage optimization.

Table 4: Effect of the one-stage compression. We report the results of ResNet-56 on CIFAR-100.  

<table><tr><td>Network</td><td>Method</td><td>Top-1 Acc.</td><td>Top-5 Acc.</td><td>BOPs (M)</td></tr><tr><td rowspan="2">ResNet-56</td><td>ABS-P → ABS-Q</td><td>70.4 ±0.1</td><td>90.8±0.2</td><td>1077.7</td></tr><tr><td>ABS</td><td>70.8±0.4</td><td>91.2±0.1</td><td>1042.5</td></tr></table>

# 5 CONCLUSION

In this paper, we have proposed a novel model compression method called Automatically Bit Sharing (ABS). Specifically, our ABS is based on the observation that quantized values of a high bitwidth share the ones of lower bitwidths under some constraints. We therefore have proposed the decomposition of quantization that encapsulates all candidate bitwidths. Starting from a low bitwidth in the search space, we sequentially increase the effective bitwidth by recursively adding re-assignment offsets. Based on this, we have further introduced learnable binary gates to encode the choice of different compression policies. By training the binary gates, the optimal compression ratio of each layer can be automatically determined. Experiments on CIFAR-100 and ImageNet have shown that our methods are able to achieve significant cost reduction while preserving the performance.

# REFERENCES

Yu Bai, Yu-Xiang Wang, and Edo Liberty. Proxquant: Quantized neural networks via proximal operators. In Proc. Int. Conf. Learn. Repren., 2019.  
Han Cai, Ligeng Zhu, and Song Han. ProxylessNAS: Direct neural architecture search on target task and hardware. In Proc. Int. Conf. Learn. Repren., 2019.  
Zhaowei Cai and Nuno Vasconcelos. Rethinking differentiable search for mixed-precision neural networks. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2020.  
Zhaowei Cai, Xiaodong He, Jian Sun, and Nuno Vasconcelos. Deep learning with low precision by half-wave gaussian quantization. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2017.  
Yongjian Chen, Tao Guan, and Cheng Wang. Approximate nearest neighbor search by residual vector quantization. Sensors, 10(12):11259-11273, 2010.  
Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. Pact: Parameterized clipping activation for quantized neural networks. arXiv preprint arXiv:1805.06085, 2018.  
Ruizhou Ding, Ting-Wu Chin, Zeye Liu, and Diana Marculescu. Regularizing activation distribution for training binarized deep networks. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2019.  
Xuanyi Dong and Yi Yang. Network pruning via transformable architecture search. In Proc. Adv. Neural Inf. Process. Syst., 2019.  
Zhen Dong, Zhewei Yao, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. Hawq: Hessian aware quantization of neural networks with mixed-precision. In Proc. IEEE Int. Conf. Comp. Vis., 2019.  
Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S. Modha. Learned step size quantization. In Proc. Int. Conf. Learn. Repren., 2020.  
Yunchao Gong, Liu Liu, Ming Yang, and Lubomir Bourdev. Compressing deep convolutional networks using vector quantization. arXiv preprint arXiv:1412.6115, 2014.  
Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. In Proc. Adv. Neural Inf. Process. Syst., 2016.  
Yong Guo, Yin Zheng, Mingkui Tan, Qi Chen, Jian Chen, Peilin Zhao, and Junzhou Huang. Nat: Neural architecture transformer for accurate and compact architectures. In Proc. Adv. Neural Inf. Process. Syst., 2019.  
Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei, and Jian Sun. Single path one-shot neural architecture search with uniform sampling. In Proc. Eur. Conf. Comp. Vis., 2020.  
Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In Proc. Int. Conf. Learn. Repren., 2016.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2016.  
Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, and Yi Yang. Filter pruning via geometric median for deep convolutional neural networks acceleration. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2019.  
Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural networks. In Proc. IEEE Int. Conf. Comp. Vis., 2017.  
Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, and Song Han. Amc: Automl for model compression and acceleration on mobile devices. In Proc. Eur. Conf. Comp. Vis., 2018.

Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized neural networks. In Proc. Adv. Neural Inf. Process. Syst., 2016.  
Sangil Jung, Changyong Son, Seohyung Lee, Jinwoo Son, Jae-Joon Han, Youngjun Kwak, Sung Ju Hwang, and Changkyu Choi. Learning to quantize deep networks by optimizing quantization intervals with task loss. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2019.  
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Proc. Adv. Neural Inf. Process. Syst., 2012.  
Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. In Proc. Int. Conf. Learn. Repren., 2017a.  
Yuhang Li, Xin Dong, and Wei Wang. Additive powers-of-two quantization: An efficient non-uniform discretization for neural networks. In Proc. Int. Conf. Learn. Repren., 2020.  
Zefan Li, Bingbing Ni, Wenjun Zhang, Xiaokang Yang, and Wen Gao. Performance guaranteed network acceleration via high-order residual quantization. In Proc. IEEE Int. Conf. Comp. Vis., 2017b.  
Tsung-Yi Lin, Piotr Dólar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2017a.  
Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dólar. Focal loss for dense object detection. In Proc. IEEE Int. Conf. Comp. Vis., 2017b.  
Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: Differentiable architecture search. In Proc. Int. Conf. Learn. Repren., 2019a.  
Zechun Liu, Haoyuan Mu, Xiangyu Zhang, Zichao Guo, Xin Yang, Kwang-Ting Cheng, and Jian Sun. Metapruning: Meta learning for automatic neural network channel pruning. In Proc. IEEE Int. Conf. Comp. Vis., 2019b.  
Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts. In Proc. Int. Conf. Learn. Repren., 2017.  
Qian Lou, Lantao Liu, Minje Kim, and Lei Jiang. Autoqb: Automl for network quantization and binarization on mobile devices. arXiv preprint arXiv:1902.05690, 2019.  
Christos Louizos, Matthias Reisser, Tijmen Blankevoort, Efstratios Gavves, and Max Welling. Relaxed quantization for discretized neural networks. In Proc. Int. Conf. Learn. Repren., 2019.  
Jian-Hao Luo, Jianxin Wu, and Weiyao Lin. Thinet: A filter level pruning method for deep neural network compression. In Proc. IEEE Int. Conf. Comp. Vis., 2017.  
Yurii E Nesterov. A method for solving the convex programming problem with convergence rate of  $(1 / \mathrm{k}^{\hat{}}2)$ . In Proceedings of the USSR Academy of Sciences, volume 269, pp. 543-547, 1983.  
Hieu Pham, Melody Guan, Barret Zoph, Quoc Le, and Jeff Dean. Efficient neural architecture search via parameter sharing. In Proc. Int. Conf. Mach. Learn., 2018.  
Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V Le. Regularized evolution for image classifier architecture search. In Proc. AAAI Conf. on Arti. Intel., 2019.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. Int. J. Comp. Vis., 115(3):211-252, 2015.  
Dimitrios Stamoulis, Ruizhou Ding, Di Wang, Dimitrios Lymberopoulos, Bodhi Priyantha, Jie Liu, and Diana Marculescu. Single-path nas: Designing hardware-efficient convnets in less than 4 hours. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 481-497, 2019.

Frederick Tung and Greg Mori. Clip-q: Deep network compression learning by in-parallel pruning-quantization. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2018.  
Stefan Uhlich, Lukas Mauch, Fabien Cardinaux, Kazuki Yoshiyama, Javier Alonso Garcia, Stephen Tiedemann, Thomas Kemp, and Akira Nakamura. Mixed precision dnns: All you need is a good parametrization. In Proc. Int. Conf. Learn. Repren., 2020.  
Mart van Baalen, Christos Louizos, Markus Nagel, Rana Ali Amjad, Ying Wang, Tijmen Blankevoort, and Max Welling. Bayesian bits: Unifying quantization and pruning. arXiv preprint arXiv:2005.07093, 2020.  
Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin, and Song Han. Haq: Hardware-aware automated quantization with mixed precision. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2019.  
Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Hanrui Wang, Yujun Lin, and Song Han. Apq: Joint search for network architecture, pruning and quantization policy. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2020.  
Bichen Wu, Yanghan Wang, Peizhao Zhang, Yuandong Tian, Peter Vajda, and Kurt Keutzer. Mixed precision quantization of convnets via differentiable neural architecture search. arXiv preprint arXiv:1812.00090, 2018.  
Haichuan Yang, Shupeng Gui, Yuhao Zhu, and Ji Liu. Automatic neural network compression by sparsity-quantization joint learning: A constrained optimization-based approach. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2020.  
Shaokai Ye, Tianyun Zhang, Kaiqi Zhang, Jiayu Li, Jiaming Xie, Yun Liang, Sijia Liu, Xue Lin, and Yanzhi Wang. A unified framework of dnn weight pruning and weight clustering/quantization using admm. In Proc. AAAI Conf. on Arti. Intel., 2019.  
Wang Ying, Lu Yadong, and Blankevoort Tijmen. Differentiable joint pruning and quantization for hardware efficiency. In Proc. Eur. Conf. Comp. Vis., 2020.  
Dongqing Zhang, Jiaolong Yang, Dongqiangzi Ye, and Gang Hua. Lq-nets: Learned quantization for highly accurate and compact deep neural networks. In Proc. Eur. Conf. Comp. Vis., 2018.  
Shuchang Zhou, Yuxin Wu, Zekun Ni, Xinyu Zhou, He Wen, and Yuheng Zou. Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients. arXiv preprint arXiv:1606.06160, 2016.  
Bohan Zhuang, Chunhua Shen, Mingkui Tan, Lingqiao Liu, and Ian Reid. Towards effective low-bitwidth convolutional neural networks. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2018a.  
Bohan Zhuang, Chunhua Shen, Mingkui Tan, Lingqiao Liu, and Ian Reid. Structured binary neural networks for accurate image classification and semantic segmentation. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2019.  
Bohan Zhuang, Lingqiao Liu, Mingkui Tan, Chunhua Shen, and Ian Reid. Training quantized neural networks with a full-precision auxiliary module. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2020.  
Zhuangwei Zhuang, Mingkui Tan, Bohan Zhuang, Jing Liu, Yong Guo, Qingyao Wu, Junzhou Huang, and Jinhui Zhu. Discrimination-aware channel pruning for deep neural networks. In Proc. Adv. Neural Inf. Process. Syst., 2018b.
