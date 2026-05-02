# TOWARDS THE LIMIT OF NETWORK QUANTIZATION

Yoojin Choi, Mostafa El-Khamy, and Jungwon Lee

Samsung US R&D Center, San Diego, CA 92121, USA

{yoojin.c,mostafa.e,jungwon2.lee}@samsung.com

# ABSTRACT

Network quantization is one of network compression techniques to reduce the redundancy of deep neural networks. It reduces the number of distinct network parameter values by quantization in order to save the storage for them. In this paper, we design network quantization schemes that minimize the performance loss due to quantization given a compression ratio constraint. We analyze the quantitative relation of quantization errors to the neural network loss function and identify that the Hessian-weighted distortion measure is locally the right objective function for the optimization of network quantization. As a result, Hessian-weighted k-means clustering is proposed for clustering network parameters to quantize. When optimal variable-length binary codes, e.g., Huffman codes, are employed for further compression, we derive that the network quantization problem can be related to the entropy-constrained scalar quantization (ECSQ) problem in information theory and consequently propose two solutions of ECSQ for network quantization, i.e., uniform quantization and an iterative algorithm similar to Lloyd's algorithm. Finally, using the simple uniform quantization followed by Huffman coding, our experiment results show that the compression ratios of 51.25, 22.17 and 40.65 are achievable for LeNet, 32-layer ResNet and AlexNet, respectively.

# 1 INTRODUCTION

Deep neural networks have emerged to be the state-of-the-art in the field of machine learning for image classification, object detection, speech recognition, natural language processing, and machine translation (LeCun et al., 2015). The substantial progress of neural networks however comes with high cost of computations and hardware resources resulting from a large number of parameters. For example, Krizhevsky et al. (2012) came up with a deep convolutional neural network consisting of 61 million parameters and won the ImageNet competition in 2012. It is followed by deeper neural networks with even larger numbers of parameters, e.g., Simonyan & Zisserman (2014).

The large sizes of deep neural networks make it difficult to deploy them on resource-limited devices, e.g., mobile or embedded devices, and network compression is of great interest in recent years to reduce computational cost and memory requirements for deep neural networks. Our interest in this paper is mainly on curtailing the size of the storage (memory) for network parameters (weights and biases). In particular, we focus on the network size compression by reducing the number of distinct network parameters by quantization.

The most related work to our investigation can be found in Gong et al. (2014); Han et al. (2015a), where a conventional quantization method using k-means clustering is employed for network quantization. This conventional approach however is proposed with little consideration for the impact of quantization errors on the neural network performance loss and no effort to optimize the quantization procedure for a given compression ratio constraint. In this paper, we reveal the suboptimality of this conventional method and newly design quantization schemes for neural networks. In particular, we formulate an optimization problem to minimize the network performance loss due to quantization given a compression ratio constraint and find efficient quantization methods for neural networks.

The main contribution of the paper can be summarized as follows:

- It is derived that the performance loss due to quantization in neural networks can be quantified approximately by the Hessian-weighted distortion measure. Then, Hessian-weighted k-means clustering is proposed for network quantization to minimize the performance loss.

- It is identified that the optimization problem for network quantization provided a compression ratio constraint can be reduced to an entropy-constrained scalar quantization (ECSQ) problem when optimal variable-length binary coding is employed after quantization. Two efficient heuristic solutions for ECSQ are proposed for network quantization, i.e., uniform quantization and an iterative algorithm similar to Lloyd's algorithm.  
- As an alternative of Hessian, it is proposed to utilize some function (e.g., square root) of the second moments of gradients when the Adam (Kingma & Ba, 2014) stochastic gradient decent (SGD) optimizer is used in training. The advantage of using this alternative is that it is computed while training and can be obtained at the end of training at no additional cost.  
- It is shown how the proposed network quantization schemes can be applied for quantizing network parameters of all layers together at once, rather than layer-by-layer network quantization in Gong et al. (2014); Han et al. (2015a). This follows from our investigation that Hessian-weighting can handle the different impact of quantization errors properly not only within layers but also across layers. Moreover, quantizing network parameters of all layers together, one can avoid layer-by-layer compression rate optimization.

The rest of the paper is organized as follows. Section 2 describes a general neural network model. In Section 3, we define the network quantization problem and review the conventional quantization method using k-means clustering. Section 4 discusses Hessian-weighted network quantization. Our entropy-constrained network quantization schemes follow in Section 5. Finally, experiment results and conclusion can be found in Section 6 and Section 7, respectively.

# 2 NETWORK MODEL

We consider a general non-linear neural network that yields output  $\mathbf{y}$  from input  $\mathbf{x}$  according to

$$
\mathbf {y} = f (\mathbf {x}; \mathbf {w}),
$$

where the function  $f$  is determined by the structure of the neural network while  $\mathbf{w} = [w_{1}\dots w_{N}]^{T}$  is the vector consisting of all trainable network parameters in the network;  $N$  is the total number of trainable parameters in the network. A loss function  $\mathrm{loss}(\mathbf{y},\hat{\mathbf{y}})$  is defined as the objective function that we aim to minimize in average:

$$
\operatorname {l o s s} (\mathbf {y}, \hat {\mathbf {y}}) = \operatorname {l o s s} (f (\mathbf {x}; \mathbf {w}), \hat {\mathbf {y}} (\mathbf {x})).
$$

Observe that  $\mathbf{y} = f(\mathbf{x};\mathbf{w})$  is the predicted output from the network for input  $\mathbf{x}$  and  $\hat{\mathbf{y}} = \hat{\mathbf{y}} (\mathbf{x})$  is the expected (ground-truth) output for input  $\mathbf{x}$ . Cross entropy or mean square error are typical examples of a loss function. We define the average loss function for any input data set  $\mathcal{X}$  as follows:

$$
L (\mathcal {X}; \mathbf {w}) = \frac {1}{| \mathcal {X} |} \sum_ {\mathbf {x} \in \mathcal {X}} \operatorname {l o s s} (f (\mathbf {x}; \mathbf {w}), \hat {\mathbf {y}} (\mathbf {x})).
$$

Given a training data set  $\mathcal{X}_{\mathrm{train}}$ , we optimize network parameters by solving the following problem, e.g., approximately by using a stochastic gradient descent (SGD) method with mini-batches:

$$
\hat {\mathbf {w}} = \underset {\mathbf {w}} {\operatorname {a r g m i n}} L (\mathcal {X} _ {\mathrm {t r a i n}}; \mathbf {w}).
$$

# 3 NETWORK QUANTIZATION

We consider a neural network that is already trained, pruned if employed and fine-tuned before quantization. If no network pruning is employed, all parameters in a network are subject to quantization. For pruned networks, our focus is on quantization of unpruned parameters.

The goal of network quantization is to quantize (unpruned) network parameters in order to reduce the size of the storage for them while minimizing the performance degradation due to quantization. For network quantization, network parameters are grouped into clusters. Parameters in the same cluster share their quantized value, which is the representative value (i.e., cluster center) of the cluster they belong to. After quantization, lossless binary coding follows to encode quantized parameters into binary codewords to store instead of actual parameter values. Either fixed-length binary coding or variable-length binary coding, e.g., Huffman coding, can be employed to this end.

# 3.1 COMPRESSION RATIO

Before quantization, each network parameter is assumed to be of  $b$  bits. Suppose that we partition the network parameters into  $k$  clusters. For  $1 \leq i \leq k$ , let  $C_i$  be the set of network parameters in cluster  $i$  and let  $b_i$  be the number of bits of the codeword assigned to the network parameters in cluster  $i$ . For a lookup table to decode quantized values from their binary encoded codewords, we store  $k$  binary codewords ( $b_i$  bits for  $1 \leq i \leq k$ ) and corresponding quantized values ( $b$  bits for each). The compression ratio is then given by

$$
\text {C o m p r e s s i o n} = \frac {N b}{\sum_ {i = 1} ^ {k} \left(\left| \mathcal {C} _ {i} \right| + 1\right) b _ {i} + k b}. \tag {1}
$$

Observe in (1) that the compression ratio depends not only on the number of clusters but also on the sizes of the clusters and the lengths of the binary codewords assigned to them, in particular, when a variable-length code is used for encoding quantized values. However, for fixed-length codes, where all codewords are of the same length, i.e.,  $b_{i} = \lceil \log_{2}k\rceil$  for all  $1\leq i\leq k$ , it reduces to

$$
\text {C o m p r e s s i o n} = \frac {N b}{N \left[ \log_ {2} k \right] + k b}, \tag {2}
$$

which is only a function of the number of clusters, i.e.,  $k$ , assuming that  $N$  and  $b$  are given; here, we note that it is not necessary to store  $k$  binary codewords in a lookup table for fixed-length codes since they can be implicitly known.

# 3.2 K-MEANS CLUSTERING

Provided network parameters  $\{w_{i}\}_{i = 1}^{N}$  to quantize, k-means clustering partitions them into  $k$  disjoint sets (clusters), denoted by  $\mathcal{C}_1,\mathcal{C}_2,\ldots ,\mathcal{C}_k$ , while minimizing the mean square quantization error (MSQE) as follows:

$$
\underset {\mathcal {C} _ {1}, \mathcal {C} _ {2}, \dots , \mathcal {C} _ {k}} {\operatorname {a r g m i n}} \sum_ {i = 1} ^ {k} \sum_ {w \in \mathcal {C} _ {i}} | w - c _ {i} | ^ {2}, \quad \text {w h e r e} \quad c _ {i} = \frac {1}{| \mathcal {C} _ {i} |} \sum_ {w \in \mathcal {C} _ {i}} w. \tag {3}
$$

We observe two issues with employing k-means clustering for network quantization.

- First, although k-means clustering minimizes the MSQE, it does not imply that k-means clustering minimizes the performance loss due to quantization as well in neural networks. K-means clustering treats quantization errors from all network parameters with equal importance. However, quantization errors from some network parameters may degrade the performance more significantly than the others. Thus, for minimizing the loss due to quantization in neural networks, one needs to take this dissimilarity into account.  
- Second, k-means clustering does not consider any compression ratio constraint. It simply minimizes its distortion measure for a given number of clusters, i.e., for  $k$  clusters. This is however suboptimal when variable-length binary coding follows since the compression ratio depends not only on the number of clusters but also on the sizes of the clusters and assigned codeword lengths to them, which are determined by the binary coding scheme employed after clustering. Thus, for the optimization of network quantization given a compression ratio constraint, one need to take the impact of binary coding into account, i.e., we need to solve the quantization problem under the actual compression ratio constraint imposed by the specific binary coding scheme employed after clustering.

# 4 HESSIAN-WEIGHTED NETWORK QUANTIZATION

In this section, we analyze the impact of quantization errors on the neural network loss function and derive that the Hessian-weighted distortion measure is a relevant objective function for network quantization in order to minimize the quantization loss locally. Moreover, from this analysis, we propose Hessian-weighted k-means clustering for network quantization to minimize the performance loss due to quantization in neural networks.

# 4.1 HESSIAN-WEIGHTED QUANTIZATION ERROR

The average loss function  $L(\mathcal{X}; \mathbf{w})$  can be expanded by Taylor series with respect to  $\mathbf{w}$  as follows:

$$
\delta L (\mathcal {X}; \mathbf {w}) = \mathbf {g} (\mathbf {w}) ^ {T} \delta \mathbf {w} + \frac {1}{2} \delta \mathbf {w} ^ {T} \mathbf {H} (\mathbf {w}) \delta \mathbf {w} + O \left(\| \delta \mathbf {w} \| ^ {3}\right), \tag {4}
$$

where  $\mathbf{w} = [w_1\cdot \cdot \cdot w_N]^T$  and

$$
\mathbf {g} (\mathbf {w}) = \frac {\partial L (\mathcal {X} ; \mathbf {w})}{\partial \mathbf {w}}, \quad \mathbf {H} (\mathbf {w}) = \frac {\partial^ {2} L (\mathcal {X} ; \mathbf {w})}{\partial \mathbf {w} ^ {2}};
$$

the square matrix  $\mathbf{H}(\mathbf{w})$  consisting of the second-order partial derivatives is called as Hessian matrix or Hessian. Assume that the loss function has reached to one of its local minima, at  $\mathbf{w} = \hat{\mathbf{w}}$ , after training. At local minima, gradients are all zero, i.e.,  $\mathbf{g}(\hat{\mathbf{w}}) = \mathbf{0}$ , and thus the first term in the right-hand side of (4) can be neglected. The third term in right-hand side of (4) is also ignored under the assumption that the average loss function is approximately quadratic at the local minimum  $\mathbf{w} = \hat{\mathbf{w}}$ . Finally, for simplicity, we approximate the Hessian matrix as a diagonal matrix by setting its off-diagonal terms to be zero. Then, it follows from (4) that

$$
\delta L (\mathcal {X}; \hat {\mathbf {w}}) \approx \frac {1}{2} \sum_ {i = 1} ^ {N} h _ {i i} (\hat {\mathbf {w}}) | \delta \hat {w} _ {i} | ^ {2}, \tag {5}
$$

where  $h_{ii}(\hat{\mathbf{w}})$  is the second-order partial derivative of the average loss function with respect to  $w_i$  evaluated at  $\mathbf{w} = \hat{\mathbf{w}}$ , which is the  $i$ -th diagonal element of the Hessian matrix  $\mathbf{H}(\hat{\mathbf{w}})$ .

Remark 1. The diagonal approximation for Hessian simplifies the optimization problem as well as its solution for network quantization. This simplification however comes with some performance loss. We conjecture that the loss due to this approximation is small. The reason is that the contributions from off-diagonal terms are not always additive and their summation may end up with a small value. However, diagonal terms are all non-negative and therefore their contributions are always additive.

Now, we connect (5) with the problem of network quantization by treating  $\delta \hat{w}_i$  as the quantization error of network parameter  $w_i$  at its local optimum  $w_i = \hat{w}_i$ , i.e.,

$$
\delta \hat {w} _ {i} = \bar {w} _ {i} - \hat {w} _ {i}, \tag {6}
$$

where  $\bar{w}_i$  is a quantized value of  $\hat{w}_i$ . Finally, combining (5) and (6), we derive that the local impact of quantization on the average loss function at  $\mathbf{w} = \hat{\mathbf{w}}$  can be quantified approximately as follows:

$$
\delta L (\mathcal {X}; \hat {\mathbf {w}}) \approx \frac {1}{2} \sum_ {i = 1} ^ {N} h _ {i i} (\hat {\mathbf {w}}) | \hat {w} _ {i} - \bar {w} _ {i} | ^ {2}. \tag {7}
$$

At a local minimum, the diagonal elements of Hessian, i.e.,  $h_{ii}(\hat{\mathbf{w}})$ 's, are all non-negative and thus the summation in (7) is always additive, implying that the average loss function either increases or stays the same. Therefore, the performance degradation due to quantization of a neural network can be measured approximately by the Hessian-weighted distortion as shown in (7).

Remark 2. We note that we do not consider the interactions between quantization and retraining in our formulation. We analyze the expected loss due to quantization assuming no further retraining and focus on finding optimal network quantization schemes that minimize the performance loss. In our experiments, however, we further fine-tune the quantized values (cluster centers) so that we can recover the loss due to quantization and improve the performance.

Remark 3. Observe that the relation of the Hessian-weighted distortion measure to the quantization loss holds for any model for which the objective function can be approximated as a quadratic function with respect to the parameters to quantize. Hence, the quantization methods proposed in this paper to minimize the Hessian-weighted distortion measure are not specific to neural networks but are generally applicable to quantization of parameters of any model whose objective function is locally quadratic with respect to its parameters approximately.

# 4.2 HESSIAN-WEIGHTED K-MEANS CLUSTERING

For notational simplicity, we use  $w_{i}\equiv \hat{w}_{i}$  and  $h_{ii}\equiv h_{ii}(\hat{\mathbf{w}})$  from now on. The optimal clustering that minimizes the Hessian-weighted distortion measure is given by

$$
\underset {\mathcal {C} _ {1}, \mathcal {C} _ {2}, \dots , \mathcal {C} _ {k}} {\operatorname {a r g m i n}} \sum_ {j = 1} ^ {k} \sum_ {w _ {i} \in \mathcal {C} _ {j}} h _ {i i} \left| w _ {i} - c _ {j} \right| ^ {2}, \quad \text {w h e r e} \quad c _ {j} = \frac {\sum_ {w _ {i} \in \mathcal {C} _ {j}} h _ {i i} w _ {i}}{\sum_ {w _ {i} \in \mathcal {C} _ {j}} h _ {i i}}, \tag {8}
$$

We call this as Hessian-weighted k-means clustering. Observe in (8) that we give a larger penalty for a network parameter in defining the distortion measure for clustering when its second-order partial derivative is larger, in order to avoid a large deviation from its original value, since the impact on the loss function due to quantization is expected to be larger for that parameter. Hessian-weighted k-means clustering is locally optimal in minimizing the quantization loss when fixed-length binary coding follows, where the compression ratio solely depends on the number of clusters as shown in Section 3.1.

# 4.3 HESSIAN COMPUTATION

For obtaining Hessian-weights, one needs to evaluate the second-order partial derivative of the average loss function with respect to each of network parameters, i.e., we need to calculate

$$
h _ {i i} (\hat {\mathbf {w}}) = \frac {\partial^ {2} L (\mathcal {X} ; \mathbf {w})}{\partial w _ {i} ^ {2}} \Bigg | _ {\mathbf {w} = \hat {\mathbf {w}}} = \frac {1}{| \mathcal {X} |} \frac {\partial^ {2}}{\partial w _ {i} ^ {2}} \sum_ {\mathbf {x} \in \mathcal {X}} \left. \operatorname {l o s s} (f (\mathbf {x}; \mathbf {w}), \hat {\mathbf {y}} (\mathbf {x})) \right| _ {\mathbf {w} = \hat {\mathbf {w}}}. \tag {9}
$$

Recall that we are interested in only the diagonal of Hessian. An efficient way of computing the diagonal of Hessian is presented in Le Cun (1987); Becker & Le Cun (1988) and it is based on the back propagation method that is similar to the back propagation algorithm used for computing first-order partial derivatives (gradients). That is, computing the diagonal of Hessian is of the same order of complexity as computing gradients.

Hessian computation and our network quantization are performed after completing network training. For the data set  $\mathcal{X}$  to compute Hessian in (9), we can either reuse a training data set or use some other data set, e.g., validation data set. We observed from our experiments that even using a small subset of the training or validation data set is sufficient to yield good approximation of Hessian for network quantization.

# 4.4 ALTERNATIVE OF HESSIAN

Although there is an efficient way to obtain the diagonal of Hessian as discussed in the previous subsection, Hessian-weight computation is not free. In order to avoid this additional Hessian computation, we propose to use an alternative metric instead of Hessian-weight. In particular, we consider neural networks trained with the Adam SGD optimizer (Kingma & Ba, 2014) and propose to use some function (e.g., square root) of the second moment estimates of gradients as an alternative of Hessian.

The advantage of using the second moment estimates from the Adam method is that they are computed while training and we can obtain them at the end of training at no additional cost. It makes Hessian-weighting more feasible for deep neural networks, which have millions of parameters. We note that similar quantities can be found and used for other SGD optimization methods using adaptive learning rates, e.g., AdaGrad (Duchi et al., 2011), Adadelta (Zeiler, 2012) and RMSProp (Tieleman & Hinton, 2012).

# 4.5 QUANTIZATION OF ALL LAYERS

We propose quantizing the network parameters of all layers in a neural network together at once by taking Hessian-weight into account. Layer-by-layer quantization was examined in the previous work (Gong et al., 2014; Han et al., 2015a). However, e.g., in Han et al. (2015a), a larger number of bits (a larger number of clusters) are assigned to convolutional layers than fully-connected layers, which implies that they heuristically treat convolutional layers more importantly. This follows from the fact that the impact of quantization errors on the performance varies significantly across layers; some layers, e.g., convolutional layers, may be more important than the others. This concern is exactly what we can address by using Hessian-weight.

Hessian-weighting properly handles the different impact of quantization errors not only within layers but also across layers, and so we propose performing quantization all layers together with our quantization schemes using Hessian-weight. We note that Hessian-weighting can still provide gain even for layer-by-layer quantization since it can address the different impact of the quantization errors of network parameters within each layer as well.

Finally, we note that recent neural networks are getting deeper, e.g., see Szegedy et al. (2015a;b); He et al. (2015). In such deep neural networks, quantizing network parameters of all layers together is more efficient since we can avoid layer-by-layer compression rate optimization. Optimizing compression ratios jointly across all layers (to maximize the overall compression ratio for all layers) requires exponential time complexity with respect to the number of layers. This is because the total number of possible combinations of compression ratios for individual layers increases exponentially as the number of layers increases.

# 5 ENTROPY-CONSTRAINED NETWORK QUANTIZATION

In this section, we investigate how to solve the network quantization problem under a constraint on the compression ratio. In designing network quantization schemes, we not only want to minimize the performance loss but also want to maximize the compression ratio. In Section 4, we explored how to quantify and minimize the loss due to quantization. In this section, we investigate how to take the compression ratio into account properly in the optimization of network quantization.

# 5.1 ENTROPY CODING

After clustering network parameters, lossless data compression with a variable-length binary code can be followed for compressing quantized values by assigning short binary codewords to the most frequent symbols (i.e., quantized values) and necessarily longer binary codewords to the less frequent symbols. There is a set of optimal codes that achieve the minimum average codeword length for a given source. Entropy is the theoretical limit of the average codeword length per symbol that we can achieve by lossless data compression, proved by Shannon (see, e.g., Cover & Thomas (2012, Section 5.3)). It is known that optimal codes achieve this limit with some overhead less than 1 bit when only integer-length codewords are allowed. So optimal coding is also called as entropy coding. Huffman coding is one of entropy coding schemes commonly used when the distribution of a source is provided (see, e.g., Cover & Thomas (2012, Section 5.6)), or can be estimated.

# 5.2 ENTROPY-CONSTRAINED SCALAR QUANTIZATION (ECSQ)

Considering the impact of variable-length binary coding employed for lossless data compression of quantized network parameters, we need to solve the optimization problem in (3) or the problem with Hessian-weight in (8) under the compression ratio constraint given by

$$
\text {C o m p r e s s i o n} = \frac {b}{\bar {b} + \left(\sum_ {i = 1} ^ {k} b _ {i} + k b\right) / N} > C, \quad \text {w h e r e} \quad \bar {b} = \frac {1}{N} \sum_ {i = 1} ^ {k} \left| \mathcal {C} _ {i} \right| b _ {i}, \tag {10}
$$

which follows from (1). Solving the optimization in (3) or (8) with a constraint on the compression ratio for any arbitrary variable-length binary code is too complex in general since the average codeword length can be arbitrary depending on the clustering output. However, we identify that it can be simplified if optimal codes, e.g., Huffman codes, are assumed to be used. In particular, since optimal coding closely achieves the lower limit of the average source code length, i.e., entropy, we approximately have

$$
\bar {b} \approx H = - \sum_ {i = 1} ^ {k} p _ {i} \log_ {2} p _ {i}, \tag {11}
$$

where  $H$  is the entropy of quantized network parameters after clustering (i.e., source), given that  $p_i = |\mathcal{C}_i| / N$  is the ratio of the number of network parameters in cluster  $\mathcal{C}_i$  to the number of all network parameters (i.e., source distribution). Moreover, assuming that  $N \gg k$ , we have

$$
\frac {1}{N} \left(\sum_ {i = 1} ^ {k} b _ {i} + k b\right) \approx 0, \tag {12}
$$

in (10). From (11) and (12), the constraint in (10) can be altered to an entropy constraint given by

$$
H = - \sum_ {i = 1} ^ {k} p _ {i} \log_ {2} p _ {i} <   R.
$$

where  $R \approx b / C$ . In summary, assuming that optimal coding is employed after clustering, one can approximately replace a compression ratio constraint with an entropy constraint for the clustering output. The network quantization problem is then translated into a quantization problem with an entropy constraint, which is called as entropy-constrained scalar quantization (ECSQ) in information theory. Two efficient heuristic solutions for ECSQ are proposed for network quantization in the following subsections, i.e., uniform quantization and an iterative algorithm similar to Lloyd's algorithm for k-means clustering.

# 5.3 UNIFORM QUANTIZATION

It is shown in Gish & Pierce (1968) that the uniform quantizer is the optimal high-resolution entropy-constrained scalar quantizer regardless of the source distribution for the mean square error criterion, implying that it is asymptotically optimal in minimizing the mean square quantization error for any random source with a reasonably smooth density function as the resolution becomes infinite, i.e., the number of clusters  $k \to \infty$ . This asymptotic result leads us to come up with a very simple but efficient network quantization scheme as follows:

1. We first set uniformly spaced thresholds and divide network parameters into clusters.  
2. After determining clusters, their quantized values (cluster centers) are obtained by taking the mean of network parameters in each cluster.

Note that one can use Hessian-weighted mean instead of non-weighted mean in computing cluster centers in the second step above in order to take the benefit of Hessian-weight. A performance comparison of uniform quantization with non-weighted mean and uniform quantization with Hessian-weighted mean can be found in Appendix A.1.

Remark 4. Although uniform quantization is a straightforward method, it has never been shown before in the literature that it is actually one of the most efficient quantization schemes for neural networks when optimal variable-length binary coding, e.g., Huffman coding, follows. Recall that k-means clustering followed by Huffman coding is proposed in Han et al. (2015a). We identified in this paper that simple uniform quantization outperforms k-means clustering when Huffman coding follows. We note that uniform quantization is not always good; it is not efficient for fixed-length coding, which is also first shown in this paper.

# 5.4 ITERATIVE ALGORITHM TO SOLVE ECSQ

Another scheme proposed to solve the ECSQ problem for network quantization is an iterative algorithm, which is similar to Lloyd's algorithm for k-means clustering. Although this iterative algorithm is more complicated than the uniform quantization in Section 5.3, it finds a local optimum for a given discrete source. An iterative algorithm to solve a general ECSQ problem is provided in Chou et al. (1989). We derive a similar iterative algorithm to solve the ECSQ problem for network quantization. The main difference from the method in Chou et al. (1989) is that we minimize the Hessian-weighted distortion measure instead of the non-weighted regular distortion measure for optimal quantization. The detailed algorithm and further discussion can be found in Appendix A.2.

# 6 EXPERIMENTS

This section presents our experiment results for the proposed network quantization schemes in three exemplary convolutional neural networks: (a) LeNet (LeCun et al., 1998) for the MNIST data set, (b) ResNet (He et al., 2015) for the CIFAR-10 data set, and (c) AlexNet (Krizhevsky et al., 2012) for the ImageNet ILSVRC-2012 data set. Our experiments can be summarized as follows:

- We employ the proposed network quantization methods to quantize all of network parameters in a network together at once, as discussed in 4.5. In particular, we include 32-layer ResNet (He et al., 2015) in our experiments in order to see the gain of our methods using Hessian-weight for very deep convolution neural networks.  
- We evaluate the performance of the proposed network quantization methods with and without network pruning. For a pruned model, we need to store not only the values of unpruned

parameters but also their respective indexes (locations) in the original model. For the index information, we compute the index differences between unpruned parameters in the original model and further compress them by Huffman coding as in Han et al. (2015a).

- We experiment our network quantization methods with fixed-length coding as well as with Huffman coding. It is straightforward that Huffman coding yields more compression than fixed-length coding since Huffman coding is optimal for a given source distribution. However, we evaluate both since fixed-length coding could be advantageous in practice due to its simplicity.  
- We also compare the quantization results before and after fine-tuning of quantized values (cluster centers) in order to show the impact of fine-tuning after quantization.  
- Finally, we evaluate the performance of our network quantization schemes using Hessian-weight when its alternative is used instead, as discussed in Section 4.4. To this end, we retrain the considered neural networks with the Adam SGD optimizer and obtain the second moment estimates of gradients at the end of training. Then, we use the square roots of the second moment estimates instead of Hessian-weights and evaluate the performance.

# 6.1 EXPERIMENT MODELS

First, we evaluate our network quantization schemes for the MNIST data set with a simplified version of LeNet5 (LeCun et al., 1998), consisting of two convolutional layers and two fully-connected layers followed by a soft-max layer. It has total 431,080 parameters and achieves  $99.25\%$  accuracy. For a pruned model, we keep only  $8.55\%$  of the original network parameters and prune the rest. For Hessian computation, 50,000 samples of the training set are reused. We also evaluate the performance when Hessian is computed with 1,000 samples only.

Second, we experiment our network quantization schemes for the CIFAR-10 data set (Krizhevsky, 2009) with a pre-trained 32-layer ResNet (He et al., 2015). The 32-layer ResNet consists of 464,154 parameters in total and achieves  $92.58\%$  accuracy. For a pruned model, we keep only  $20\%$  of the original network parameters and prune the rest. Similar to LeNet, for Hessian computation, we reuse 50,000 training images and also evaluate the performance when Hessian is computed with only 1,000 training images.

Third, we evaluate our network quantization schemes for the ImageNet ILSVRC-2012 data set (Russakovsky et al., 2015) with AlexNet (Krizhevsky et al., 2012). We obtain a pre-trained AlexNet Caffe model, which achieves  $57.16\%$  top-1 accuracy. For a pruned model, we prune  $89\%$  parameters and fine-tune the rest. In fine-tuning, the Adam SGD optimizer is used in order to avoid the computation of Hessian by utilizing its alternative (see Section 4.4). However, the pruned model does not recover the original accuracy after fine-tuning with the Adam method; the top-1 accuracy recovered after pruning and fine-tuning is  $56.00\%$ . We note that we are able to find a pruned model achieving the original accuracy by iterative pruning and retraining (Han et al., 2015b), which is however not used in our experiment.

# 6.2 EXPERIMENT RESULTS

We present the quantization results for unpruned models first. In particular, we show the results for 32-layer ResNet in Figure 1. Recall that we employ our network quantization schemes in order to quantize network parameters in all layers together and evaluate the performance. The impact of quantization errors could vary more substantially across layers than within layers. Thus, Hessian-weighting can have more benefit in deeper neural networks.

In Figure 1, the accuracy of 32-layer ResNet is plotted against the average codeword length per network parameter after quantization. When fixed-length coding is employed, the proposed Hessian-weighted k-means clustering method performs the best, as expected. Observe that Hessian-weighted k-means clustering provides better accuracy than others even after fine-tuning. On the other hand, when Huffman coding is employed, uniform quantization and the iterative algorithm for ECSQ outperform Hessian-weighted k-means clustering and k-means clustering. However, these two ECSQ solutions underperform Hessian-weighted k-means clustering and even k-means clustering when fixed-length coding is employed since they are optimized for optimal variable-length coding.

![](images/ab71826ca5e0ab7126fed24b71780088e6aa479d9afa77eea5ab4062bc372579.jpg)  
(a) Fixed-length coding

![](images/da7a1677cd1b2620b58f1443cebad5616073aaddee718294a0ecb62c574d2f39.jpg)  
(b) Fixed-length coding + fine-tuning

![](images/ca4f3248041426dcb15a8772b03a9611dcac3215a55f51b7c10ec98bd54c925b.jpg)  
(c) Huffman coding

![](images/748de65a931c6af4488820ef6f377eb11186b2f0d6eae05bd7daa039a35ff558.jpg)  
(d) Huffman coding + fine-tuning

![](images/83cfaaf2772c705bcb1c0180295df9b2ec7d8a17af2982bb654f9e9101646765.jpg)  
(a) LeNet  
Figure 2: Accuracy versus average codeword length per network parameter after network quantization, Huffman coding and fine-tuning for LeNet and 32-layer ResNet when Hessian is computed with 50,000 or 1,000 samples and when the square roots of second moments of gradients are used instead of Hessian as an alternative.

![](images/bd6b9d53e932b2cdaf8666cdf2abefa06b0ea53c77b11ce7879737dde413aa5b.jpg)  
Figure 1: Accuracy versus average codeword length per network parameter after network quantization for 32-layer ResNet.  
(b) ResNet

Figure 2 shows the performance of Hessian-weighted k-means clustering when Hessian is computed with a small number of samples (1,000 samples). Observe that even using Hessian computed with a small number of samples yields almost the same performance. Furthermore, we show the performance of Hessian-weighted k-means clustering when an alternative of Hessian is used instead of Hessian as explained in Section 4.4. In particular, the square roots of the second moments of gradients are used instead of Hessian-weight, and using this alternative provides similar performance to using Hessian.

Finally, in Table 1, we summarize the compression ratios that we can achieve with different network quantization methods for pruned models. The original network parameters are 32-bit float numbers.

Table 1: Summary of network quantization results with Huffman coding for pruned models.  

<table><tr><td colspan="3"></td><td>Accuracy %</td><td>Compression ratio</td></tr><tr><td rowspan="7">LeNet</td><td colspan="2">Original model</td><td>99.25</td><td>-</td></tr><tr><td colspan="2">Pruned model</td><td>99.27</td><td>10.13</td></tr><tr><td rowspan="4">Pruning + Quantization all layers + Huffman coding</td><td>k-means</td><td>99.27</td><td>44.58</td></tr><tr><td>Hessian-weighted k-means</td><td>99.27</td><td>47.16</td></tr><tr><td>Uniform quantization</td><td>99.28</td><td>51.25</td></tr><tr><td>Iterative ECSQ</td><td>99.27</td><td>49.01</td></tr><tr><td colspan="2">Deep compression (Han et al., 2015a)</td><td>99.26</td><td>39.00</td></tr><tr><td rowspan="7">ResNet</td><td colspan="2">Original model</td><td>92.58</td><td>-</td></tr><tr><td colspan="2">Pruned model</td><td>92.58</td><td>4.52</td></tr><tr><td rowspan="4">Pruning + Quantization all layers + Huffman coding</td><td>k-means</td><td>92.64</td><td>18.25</td></tr><tr><td>Hessian-weighted k-means</td><td>92.67</td><td>20.51</td></tr><tr><td>Uniform quantization</td><td>92.68</td><td>22.17</td></tr><tr><td>Iterative ECSQ</td><td>92.73</td><td>21.01</td></tr><tr><td colspan="2">Deep compression (Han et al., 2015a)</td><td>N/A</td><td>N/A</td></tr><tr><td rowspan="6">AlexNet</td><td colspan="2">Original model</td><td>57.16</td><td>-</td></tr><tr><td colspan="2">Pruned model</td><td>56.00</td><td>7.91</td></tr><tr><td rowspan="3">Pruning + Quantization all layers + Huffman coding</td><td>k-means</td><td>56.12</td><td>30.53</td></tr><tr><td>Alt-Hessian-weighted k-means</td><td>56.04</td><td>33.71</td></tr><tr><td>Uniform quantization</td><td>56.20</td><td>40.65</td></tr><tr><td colspan="2">Deep compression (Han et al., 2015a)</td><td>57.22</td><td>35.00</td></tr></table>

Using the simple uniform quantization followed by Huffman coding, we achieve the compression ratios of 51.25, 22.17 and 40.65 (i.e., the sizes of the compressed models are  $1.95\%$ ,  $4.51\%$  and  $2.46\%$  of the original model sizes) for LeNet, ResNet and AlexNet, respectively, at no or marginal performance loss. Note that the loss in the compressed AlexNet is mainly due to pruning. Moreover, we note that layer-by-layer quantization with k-means clustering is evaluated in Han et al. (2015a) while our k-means clustering is employed to quantize network parameters of all layers together at once (see Section 4.5).

# 7 CONCLUSION

This paper investigates the quantization problem of network parameters in deep neural networks. We identify the suboptimality of the conventional quantization method using k-means clustering and newly design network quantization schemes so that they can minimize the performance loss due to quantization given a compression ratio constraint. In particular, we analytically show that Hessian-weight can be employed as a measure of the importance of network parameters and propose to minimize Hessian-weighted quantization errors in average for clustering network parameters to quantize. Hessian-weighting is beneficial in quantizing all of the network parameters together at once since it can handle the different impact of quantization errors properly not only within layers but also across layers. Furthermore, we make connection from the network quantization problem to the entropy-constrained data compression problem in information theory and push the compression ratio to the limit that information theory provides. Two efficient heuristic solutions are presented to this end, i.e., uniform quantization and an iterative solution for ECSQ. Finally, our experiment results show that the proposed network quantization schemes provide considerable gain over the conventional method using k-means clustering, in particular for deeper and larger neural networks.

# REFERENCES

Sue Becker and Yann Le Cun. Improving the convergence of back-propagation learning with second order methods. In Proceedings of the Connectionist Models Summer School, pp. 29-37. San Matteo, CA: Morgan Kaufmann, 1988.

Philip A Chou, Tom Lookabaugh, and Robert M Gray. Entropy-constrained vector quantization. IEEE Transactions on Acoustics, Speech, and Signal Processing, 37(1):31-42, 1989.  
Thomas M Cover and Joy A Thomas. Elements of information theory. John Wiley & Sons, 2012.  
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul):2121-2159, 2011.  
Herbert Gish and John Pierce. Asymptotically efficient quantizing. IEEE Transactions on Information Theory, 14(5):676-683, 1968.  
Yunchao Gong, Liu Liu, Ming Yang, and Lubomir Bourdev. Compressing deep convolutional networks using vector quantization. arXiv preprint arXiv:1412.6115, 2014.  
Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. arXiv preprint arXiv:1510.00149, 2015a.  
Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In Advances in Neural Information Processing Systems, pp. 1135-1143, 2015b.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, pp. 1097-1105, 2012.  
Yann Le Cun. Modèles connexionnistes de l'apprentissage. PhD thesis, Paris 6, 1987.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436-444, 2015.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211-252, 2015.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1-9, 2015a.  
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. arXiv preprint arXiv:1512.00567, 2015b.  
Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 4(2), 2012.  
Matthew D Zeiler. Adadelta: an adaptive learning rate method. arXiv preprint arXiv:1212.5701, 2012.
