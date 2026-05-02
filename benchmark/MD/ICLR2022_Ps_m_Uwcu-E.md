# LAYER-WISE ADAPTIVE MODEL AGGREGATION FOR SCALABLE FEDERATED LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

In Federated Learning, a common approach for aggregating local models across clients is periodic averaging of the full model parameters. It is, however, known that different layers of neural networks can have a different degree of model discrepancy across the clients. The conventional full aggregation scheme does not consider such a difference and synchronizes the whole model parameters at once, resulting in inefficient network bandwidth consumption. Aggregating the parameters that are similar across the clients does not make meaningful training progress while increasing the communication cost. We propose FedLAMA, a layer-wise model aggregation scheme for scalable Federated Learning. FedLAMA adaptively adjusts the aggregation interval in a layer-wise manner, jointly considering the model discrepancy and the communication cost. The layer-wise aggregation method enables to finely control the aggregation interval to relax the aggregation frequency without a significant impact on the model accuracy. Our empirical study shows that FedLAMA reduces the communication cost by up to  $60\%$  for IID data and  $70\%$  for non-IID data while achieving a comparable accuracy to FedAvg.

# 1 INTRODUCTION

In Federated Learning, periodic full model aggregation is the most common approach for aggregating local models across clients. Many Federated Learning algorithms, such as FedAvg (McMahan et al. (2017)), FedProx (Li et al. (2018)), FedNova (Wang et al. (2020)), and SCAFFOLD (Karimireddy et al. (2020)), assume the underlying periodic full aggregation scheme. However, it has been observed that the magnitude of gradients can be significantly different across the layers of neural networks (You et al. (2019)). That is, all the layers can have a different degree of model discrepancy. The periodic full aggregation scheme does not consider such a difference and synchronizes the entire model parameters at once. Aggregating the parameters that are similar across all the clients does not make meaningful training progress while increasing the communication cost. Considering the limited network bandwidth in usual Federated Learning environments, such an inefficient network bandwidth consumption can significantly harm the scalability of Federated Learning applications.

Many researchers have put much effort into addressing the expensive communication issue. Adaptive model aggregation methods adjust the aggregation interval to reduce the total communication cost (Wang & Joshi (2018); Haddadpour et al. (2019)). Gradient (model) compression (Alistarh et al. (2018); Albasyoni et al. (2020)), sparsification (Wangni et al. (2017); Wang et al. (2018)), low-rank approximation (Vogels et al. (2020); Wang et al. (2021)), and quantization (Alistarh et al. (2017); Wen et al. (2017); Reisizadeh et al. (2020)) techniques directly reduce the local data size. Employing heterogeneous model architectures across clients is also a communication-efficient approach (Diao et al. (2020)). While all these works effectively tackle the expensive communication issue from different angles, they commonly assume the underlying periodic full model aggregation.

To break such a convention of periodic full model aggregation, we propose FedLAMA, a novel layer-wise adaptive model aggregation scheme for scalable and accurate Federated Learning. FedLAMA first prioritizes all the layers based on their contributions to the total model discrepancy. We present a metric for estimating the layer-wise degree of model discrepancy at run-time. The aggregation intervals are adjusted based on the layer-wise model discrepancy such that the layers with a smaller degree of model discrepancy is assigned with a longer aggregation interval than the other layers. The above steps are repeatedly performed once the entire model is synchronized once.

Our focus is on how to relax the model aggregation frequency at each layer, jointly considering the communication efficiency and the impact on the convergence properties of federated optimization. By adjusting the aggregation interval based on the layer-wise model discrepancy, the local models can be effectively synchronized while reducing the number of communications at each layer. The model accuracy is marginally affected since the intervals are increased only at the layers that have the smallest contribution to the total model discrepancy. Our empirical study demonstrates that FedLAMA automatically finds the interval settings that make a practical trade-off between the communication cost and the model accuracy. We also provide a theoretical convergence analysis of FedLAMA for smooth and non-convex problems under non-IID data settings.

We evaluate the performance of FedLAMA across three representative image classification benchmark datasets: CIFAR-10 (Krizhevsky et al. (2009)), CIFAR-100, and Federated Extended MNIST (Cohen et al. (2017)). Our experimental results deliver novel insights on how to aggregate the local models efficiently consuming the network bandwidth. Given a fixed number of training iterations, as the aggregation interval increases, FedLAMA reduces the communication cost by up to  $60\%$  under IID data settings and  $70\%$  under non-IID data settings, while having only a marginal accuracy drop.

# 2 RELATED WORKS

Compression Methods - The communication-efficient global model update methods can be categorized into two groups: structured update and sketched update (Konečný et al. (2016)). The structured update indicates the methods that enforce a pre-defined fixed structure of the local updates, such as low-rank approximation and random mask methods. The sketched update indicates the methods that post-process the local updates via compression, sparsification, or quantization. Both directions are well studied and have shown successful results (Alistarh et al. (2018); Albasyoni et al. (2020); Wangni et al. (2017); Wang et al. (2018); Vogels et al. (2020); Wang et al. (2021); Alistarh et al. (2017); Wen et al. (2017); Reisizadeh et al. (2020)). The common principle behind these methods is that the local updates can be replaced with a different data representation with a smaller size.

These compression methods can be independently applied to our layer-wise aggregation scheme such that the each layer's local update is compressed before being aggregated. Since our focus is on adjusting the aggregation frequency rather than changing the data representation, we do not directly compare the performance between these two approaches. We leave harmonizing the layer-wise aggregation scheme and a variety of compression methods as a promising future work.

Similarity Scores – Canonical Correlation Analysis (CCA) methods are proposed to estimate the representational similarity across different models (Raghu et al. (2017); Marcos et al. (2018)). Centered Kernel Alignment (CKA) is an improved extension of CCA (Kornblith et al. (2019)). While these methods effectively quantify the degree of similarity, they commonly require expensive computations. For example, SVCCA performs singular vector decomposition of the model and CKA computes Hilbert-Schmidt Independence Criterion multiple times (Gretton et al. (2005)). In addition, the representational similarity does not deliver any information regarding the gradient difference that is strongly related to the convergence property. We will propose a practical metric for estimating the layer-wise model discrepancy, which is cheap enough to be used at run-time.

Layer-wise Model Freezing - Layer freezing (dropping) is the representative layer-wise technique for neural network training (Brock et al. (2017); Kumar et al. (2019); Zhang & He (2020); Goutam et al. (2020)). All these methods commonly stop updating the parameters of the layers in a bottom-up direction. These empirical techniques are supported by the analysis presented in (Raghu et al. (2017)). Since the layers converge from the input-side sequentially, the layer-wise freezing can reduce the training time without strongly affecting the accuracy. These previous works clearly demonstrate the advantages of processing individual layers separately.

# 3 BACKGROUND

Federated Optimization - We consider federated optimization problems as follows.

$$
\min  _ {\mathbf {x} \in \mathbb {R} ^ {d}} \left[ F (\mathbf {x}) := \sum_ {i = 1} ^ {m} p _ {i} F _ {i} (\mathbf {x}) \right], \tag {1}
$$

Algorithm 1: FedLAMA: Federated Layer-wise Adaptive Model Aggregation.  
Input:  $\tau^{\prime}$  : base aggregation interval,  $\phi$  : interval increasing factor,  $p_i,i\in \{1,\dots ,m\}$  1  $\tau_{l}\gets \tau^{\prime},\forall l\in \{1,\dots ,L\} ;$    
2 for  $k = 1$  to  $K$  do   
3 SGD step:  $\mathbf{x}_k^i = \mathbf{x}_{k - 1}^i -\eta \nabla f(w_{k - 1}^i,\xi_k)$  .   
4 for  $l = 1$  to  $L$  do   
5 if  $k$  mod  $\tau_{l}$  is O then   
6 Synchronize layer  $l$  ..  $\mathbf{u}_{(l,k)}\gets \sum_{i = 1}^{m}p_{i}\mathbf{x}_{(l,k)}^{i};$    
7  $d_{l}\gets \| \mathbf{u}_{(l,k)} - \mathbf{x}_{(l,k)}^{i}\|^{2} / (\tau_{l}(\dim (\mathbf{u}_{(l,k)}));$    
8 if  $k$  mod  $\phi \tau^{\prime}$  is O then   
9 Adjust aggregation interval at all  $L$  layers (Algorithm 2).;   
10 Output:  $\mathbf{u}_K$

where  $p_i = n_i / n$  is the ratio of local data to the total dataset, and  $F_{i}(\mathbf{x}) = \frac{1}{n_{i}}\sum_{\xi \in \mathcal{D}}f_{i}(\mathbf{x},\xi)$  is the local objective function of client  $i$ .  $n$  is the global dataset size and  $n_i$  is the local dataset size.

FedAvg is a basic algorithm that solves the above minimization problem. As the degree of data heterogeneity increases, FedAvg converges more slowly. Several variants of FedAvg, such as FedProx, FedNova, and SCAFFOLD, tackle the data heterogeneity issue. All these algorithms commonly aggregate the local solutions using the periodic full aggregation scheme.

Model Discrepancy – All local SGD-based algorithms allow the clients to independently train their local models within each communication round. The variance of stochastic gradients and heterogeneous data distribution can lead the local models to different directions on parameter space during the local update steps. We formally define such a discrepancy among the models as follows.

$$
\sum_ {i = 1} ^ {m} p _ {i} \| \mathbf {u} - \mathbf {x} ^ {i} \| ^ {2},
$$

where  $m$  is the number of clients,  $\mathbf{u}$  is the synchronized model, and  $\mathbf{x}^i$  is client  $i$ 's local model. This quantity bounds the difference between the local gradients and the global gradients under a smoothness assumption on objective functions.

# 4 LAYER-WISE ADAPTIVE MODEL AGGREGATION

Layer Prioritization – In theoretical analysis, it is common to assume the smoothness of objective functions such that the difference between local gradients and global gradients is bounded by a scaled difference of the corresponding sets of parameters. Motivated by this convention, we define 'layer-wise unit model discrepancy', a useful metric for prioritizing the layers as follows.

$$
d _ {l} = \frac {\left\| \mathbf {u} _ {l} - \mathbf {x} _ {l} ^ {i} \right\| ^ {2}}{\tau_ {l} (\dim (\mathbf {u} _ {l}))}, \quad l \in \{1, \dots , L \} \tag {2}
$$

where  $L$  is the number of layers,  $l$  is the layer index,  $\mathbf{u}$  is the global parameters,  $\mathbf{x}^i$  is the client  $i$ 's local parameters,  $\tau$  is the aggregation interval, and  $\dim(\cdot)$  is the number of parameters.

This metric quantifies how much each parameter contributes to the model discrepancy at each iteration. The communication cost is proportional to the number of parameters. Thus,  $\| \mathbf{u}_l - \mathbf{x}_l^i\| ^2 /\dim (\mathbf{u}_l)$  shows how much model discrepancy can be eliminated by synchronizing the layer at a unit communication cost. This metric allows prioritizing the layers such that the layers with a smaller  $d_{l}$  value has a lower priority than the others.

Adaptive Model Aggregation Algorithm - We propose FedLAMA, a layer-wise adaptive model aggregation scheme. Algorithm 1 shows FedLAMA algorithm. There are two input parameters:  $\tau'$  is the base aggregation interval and  $\phi$  is the interval increase factor. First, the parameters at layer  $l$  are synchronized across the clients after every  $\tau_l$  iterations (line 6). Then, the proposed metric

Algorithm 2: Layer-wise Adaptive Interval Adjustment.  
Input: d: the observed model discrepancy at all  $L$  layers,  $\tau'$ : the base aggregation interval,  $\phi$ : the interval increasing factor.  
1 Sorted model discrepancy:  $\hat{\mathbf{d}} \gets \text{sort}(\mathbf{d})$ ;  
2 Sorted index of the layers:  $\hat{\mathbf{i}} \gets \text{argsort}(\mathbf{d})$ ;  
3 Total model size:  $\lambda \gets \sum_{l=1}^{L} \dim(\mathbf{u}_l)$ ;  
4 Total model discrepancy:  $\delta \gets \sum_{l=1}^{L} d_l * \dim(\mathbf{u}_l)$ ;  
5 for  $l = 1$  to  $L$  do  
6  $\delta_l \gets (\sum_{i=1}^{l} \hat{d}_i * \dim(\mathbf{u}_i)) / \delta$ ;  
7  $\lambda_l \gets (\sum_{i=1}^{l} \dim(\mathbf{u}_i)) / \lambda$ ;  
8 Find the layer index:  $i \gets i_l$ ;  
9 if  $\delta_l < \lambda_l$  then  
10  $\tau_i \gets \phi \tau'$ ;  
11 else  
12  $\tau_i \gets \tau'$ ;  
13 Output:  $\tau$ : the adjusted aggregation intervals at all  $L$  layers;

$d_{l}$  is calculated using the synchronized parameters  $\mathbf{u}_l$  (line 7). At the end of every  $\phi \tau'$  iterations, FedLAMA adjusts the model aggregation interval at all the  $L$  layers. (line 9).

Algorithm 2 finds the layers that can be less frequently aggregated making a minimal impact on the total model discrepancy. First, the layer-wise degree of model discrepancy is estimated as follows.

$$
\delta_ {l} = \frac {\sum_ {i = 1} ^ {l} \hat {d} _ {i} * \dim (\mathbf {u} _ {i})}{\sum_ {i = 1} ^ {L} \hat {d} _ {i} * \dim (\mathbf {u} _ {i})}, \tag {3}
$$

where  $\hat{d}_i$  is the  $i^{th}$  smallest element in the sorted list of the proposed metric  $d$ . Given  $l$  layers with the smallest  $d_l$  values,  $\delta_l$  quantifies their contribution to the total model discrepancy. Second, the communication cost impact is estimated as follows.

$$
\lambda_ {l} = \frac {\sum_ {i = 1} ^ {l} \dim (\mathbf {u} _ {i})}{\sum_ {i = 1} ^ {L} \dim (\mathbf {u} _ {i})} \tag {4}
$$

$\lambda_{l}$  is the ratio of the parameters at the  $l$  layers with the smallest  $d_{l}$  values. Thus,  $1 - \lambda_{l}$  estimates the number of parameters that will be more frequently synchronized than the others. As  $l$  increases,  $\delta_{l}$  increases while  $1 - \lambda_{l}$  decreases monotonically. Algorithm 2 loops over the  $L$  layers finding the  $l$  value that makes  $\delta_{l}$  and  $1 - \lambda_{l}$  similar. In this way, it finds the aggregation interval setting that slightly sacrifices the model discrepancy while remarkably reducing the communication cost.

Figure 1 shows the  $\delta_l$  and  $1 - \lambda_{l}$  curves collected from a) CIFAR-10 (ResNet20) training and b) CIFAR-100 (Wide-ResNet28-10) training. The x-axis is the number of layers to increase the aggregation interval and the y-axis is the  $\delta_l$  and  $1 - \lambda_{l}$  values. The cross point of the two curves is much lower than 0.5 on y-axis in both charts. For instance, in Figure 1.a), the two curves are crossed when  $x$  value is 9, and the corresponding  $y$  value is near 0.2. That is, when the aggregation interval is increased at those 9 layers,  $20\%$  of the total model discrepancy will increase by a factor of  $\phi$  while  $80\%$  of the total communication cost will decrease by the same factor. Note that the cross points are below 0.5 since the  $\delta_l$  and  $1 - \lambda_{l}$  are calculated using the  $d_{l}$  values sorted in an increasing order.

It is worth noting that FedLAMA can be easily extended to improve the convergence rate at the cost of having minor extra communications. In this work, we do not consider finding such interval settings because it can increase the latency cost, which is not desired in Federated Learning. However, in the environments where the latency cost can be ignored, such as high-performance computing platforms, FedLAMA can accelerate the convergence by adjusting the intervals based on the cross point of  $1 - \delta_{l}$  and  $\lambda_{l}$  calculated using the list of  $d_{l}$  values sorted in a decreasing order.

Impact of Aggregation Interval Increasing Factor  $\phi$  - In Federated Learning, the communication latency cost is usually not negligible, and the total number of communications strongly affects the

![](images/6e99553fb449df0a7ababf9a5028d48aac333de73788d5b6a6f0b786b5fa83b9.jpg)  
a) CIFAR-10 (ResNet20)

![](images/2511d4f1ac659f7ed0ab37c269bbdf8e3111822a82b661fca513f4b1a6e36b81.jpg)  
Figure 1: The comparison between the model discrepancy increase factor  $\delta_{l}$  and the communication cost decrease factor  $1 - \lambda_{l}$  for a) CIFAR-10 and b) CIFAR-100 training.  
b) CIFAR-100 (WRN28-10)

scalability. When increasing the aggregation interval, Algorithm 2 multiplies a pre-defined small constant  $\phi$  to the fixed base interval  $\tau'$  (line 10). This approach ensures that the communication latency cost is not increased while the network bandwidth consumption is reduced by a factor of  $\phi$ .

FedAvg can be considered as a special case of FedLAMA where  $\phi$  is set to 1. When  $\phi > 1$ , FedLAMA less frequently synchronize a subset of layers, and it results in reducing their communication costs. When increasing the aggregation interval, FedLAMA multiplies  $\phi$  to the base interval  $\tau'$ . So, it is guaranteed that the whole model parameters are fully synchronized after  $\phi \tau'$  iterations. Because of the layers with the base aggregation interval  $\tau'$ , the total model discrepancy of FedLAMA after  $\phi \tau'$  iterations is always smaller than that of FedAvg with an aggregation interval of  $\phi \tau'$ .

# 5 CONVERGENCE ANALYSIS

# 5.1 PRELIMINARIES

Notations - All vectors in this paper are column vectors.  $\mathbf{x} \in \mathbb{R}^d$  denotes the parameters of one local model and  $m$  is the number of clients. The stochastic gradient computed from a single training data point  $\pmb{\xi}$  is denoted by  $g(\mathbf{x},\pmb{\xi})$ . For convenience, we use  $g(\mathbf{x})$  instead. The full batch gradient is denoted by  $\nabla F(\mathbf{x})$ . We use  $\|\cdot\|$  and  $\|\cdot\|_{op}$  to denote  $l2$  norm and matrix operator norm, respectively.

Assumptions - We analyze the convergence rate of FedLAMA under the following assumptions.

1. (Smoothness). Each local objective function is L-smooth, that is,  $\| \nabla F_i(\mathbf{x}) - \nabla F_i(\mathbf{y})\| \leq L\| \mathbf{x} - \mathbf{y}\|$ ,  $\forall i \in \{1,\dots,m\}$ .  
2. (Unbiased Gradient). The stochastic gradient at each client is an unbiased estimator of the local full-batch gradient:  $\mathbb{E}_{\xi}[g_i(\mathbf{x},\xi)] = \nabla F_i(\mathbf{x})$  
3. (Bounded Variance). The stochastic gradient at each client has bounded variance:  $\mathbb{E}_{\xi}[\| g_i(\mathbf{x},\xi) - \nabla F_i(\mathbf{x})\|^2 \leq \sigma^2], \forall i \in \{1,\dots,m\}, \sigma^2 \geq 0$ .  
4. (Bounded Dissimilarity). For any sets of weights  $\{p_i\geq 0\}_{i = 1}^m,\sum_{i = 1}^m p_i = 1$  , there exist constants  $\beta^2\geq 1$  and  $\kappa^2\geq 0$  such that  $\begin{array}{r}\sum_{i = 1}^{m}p_{i}\| \nabla F_{i}(\mathbf{x})\|^{2}\leq \beta^{2}\| \sum_{i = 1}^{m}p_{i}\nabla F_{i}(\mathbf{x})\|^{2} + \kappa^{2}. \end{array}$  If local objective functions are identical to each other,  $\beta^2 = 1$  and  $\kappa^2 = 0$

# 5.2 ANALYSIS

We begin with showing two key lemmas. All the proofs can be found in Appendix.

Lemma 5.1. (Framework) Under Assumption  $1 \sim 3$ , if the learning rate satisfies  $\eta \leq \frac{1}{2L}$ , FedLAMA ensures

$$
\begin{array}{l} \frac {1}{K} \sum_ {k = 1} ^ {K} \mathbb {E} \left[ \| \nabla F (\mathbf {u} _ {k}) \| ^ {2} \right] \leq \frac {2}{\eta K} \mathbb {E} \left[ F \left(\mathbf {u} _ {1}\right) - F \left(\mathbf {u} _ {*}\right) \right] + 2 \eta L \sigma^ {2} \sum_ {i = 1} ^ {m} (p _ {i}) ^ {2} \tag {5} \\ + \frac {L ^ {2}}{K} \sum_ {k = 1} ^ {K} \sum_ {i = 1} ^ {m} p _ {i} \mathbb {E} \left[ \left\| \mathbf {u} _ {k} - \mathbf {x} _ {k} ^ {i} \right\| ^ {2} \right]. \\ \end{array}
$$

Lemma 5.2. (Model Discrepancy) Under Assumption  $1 \sim 4$ , FedLAMA ensures

$$
\begin{array}{l} \frac {1}{K} \sum_ {k = 1} ^ {K} \sum_ {i = 1} ^ {m} p _ {i} \mathbb {E} \left[ \left\| \mathbf {u} _ {k} - \mathbf {x} _ {k} ^ {i} \right\| ^ {2} \right] \leq \frac {2 \eta^ {2} \left(\tau_ {\max } - 1\right) \sigma^ {2}}{1 - A} + \frac {A \kappa^ {2}}{L ^ {2} (1 - A)} \tag {6} \\ + \frac {A \beta^ {2}}{K L ^ {2} (1 - A)} \sum_ {k = 1} ^ {K} \mathbb {E} \left[ \| \nabla F (\mathbf {u} _ {k}) \| ^ {2} \right], \\ \end{array}
$$

where  $A = 4\eta^{2}(\tau_{max} - 1)^{2}L^{2}$  and  $\tau_{max}$  is the largest averaging interval across all the layers.

Based on Lemma 5.1 and 5.2, we analyze the convergence rate of FedLAMA as follows.

Theorem 5.3. Suppose all  $m$  local models are initialized to the same point  $\mathbf{u}_1$ . Under Assumption  $1 \sim 4$ , if FedLAMA runs for  $K$  iterations and the learning rate satisfies  $\eta \leq \min \left\{\frac{1}{2L}, \frac{1}{L\sqrt{2\tau_{max}(\tau_{max} - 1)(2\beta^2 + 1)}}\right\}$ , the average-squared gradient norm of  $\mathbf{u}_k$  is bounded as follows

$$
\begin{array}{l} \mathbb {E} \left[ \frac {1}{K} \sum_ {i = 1} ^ {K} \| \nabla F (\mathbf {u} _ {k}) \| ^ {2} \right] \leq \frac {4}{\eta K} \left(\mathbb {E} \left[ F (\mathbf {u} _ {1}) - F (\mathbf {u} _ {*}) \right]\right) + 4 \eta \sum_ {i = 1} ^ {m} p _ {i} ^ {2} L \sigma^ {2} \tag {7} \\ + 3 \eta^ {2} (\tau_ {m a x} - 1) L ^ {2} \sigma^ {2} + 6 \eta^ {2} \tau_ {m a x} (\tau_ {m a x} - 1) L ^ {2} \kappa^ {2}, \\ \end{array}
$$

where  $\mathbf{u}_{*}$  indicates a local minimum and  $\tau_{max}$  is the largest averaging interval across all the layers.

Remark 1. (Linear Speedup) With a sufficiently small diminishing learning rate and a large number of training iterations, FedLAMA achieves linear speedup. If the learning rate is  $\eta = \frac{\sqrt{m}}{\sqrt{K}}$ , we have

$$
\mathbb {E} \left[ \frac {1}{K} \sum_ {i = 1} ^ {K} \| \nabla F (\mathbf {u} _ {k}) \| ^ {2} \right] \leq \mathcal {O} \left(\frac {1}{\sqrt {m K}}\right) + \mathcal {O} \left(\frac {\sqrt {m}}{\sqrt {K}}\right) + \mathcal {O} \left(\frac {m}{K}\right) \tag {8}
$$

If  $K > m^3$ , the first term on the right-hand side becomes dominant and it achieves linear speedup.

Remark 2. (Impact of Interval Increase Factor  $\phi$ ) The worst-case model discrepancy depends on the largest averaging interval across all the layers,  $\tau_{max} = \phi \tau'$ . The larger the interval increase factor  $\phi$ , the larger the model discrepancy terms in (7). In the meantime, as  $\phi$  increases, the communication frequency at the selected layers is proportionally reduced. So,  $\phi$  should be appropriately tuned to effectively reduce the communication cost while not much increasing the model discrepancy.

# 6 EXPERIMENTS

Experimental Settings - We evaluate FedLAMA using three representative benchmark datasets: CIFAR-10 (ResNet20 (He et al. (2016))), CIFAR-100 (WideResNet28-10 (Zagoruyko & Komodakis (2016))), and Federated Extended MNIST (CNN (Caldas et al. (2018))). We use TensorFlow 2.4.3 for local training and MPI for model aggregation. All our experiments are conducted on 4 compute nodes each of which has 2 NVIDIA v100 GPUs.

Due to the limited compute resources, we simulate Federated Learning such that each process sequentially trains multiple models and then the models are aggregated across all the processes at once. While it provides the same classification results as the actual Federated Learning, the training time is serialized within each process. Thus, instead of wall-clock time, we consider the total communication cost calculated as follows.

$$
\mathcal {C} = \sum_ {l = 1} ^ {L} \mathcal {C} _ {l} = \sum_ {l = 1} ^ {L} \dim (\mathbf {u} _ {l}) * \kappa_ {l}, \tag {9}
$$

where  $\kappa_{l}$  is the total number of communications at layer  $l$  during the training.

Hyper-Parameter Settings – We use 128 clients in our experiments. The local batch size is set to 32 and the learning rate is tuned based on a grid search. For CIFAR-10 and CIFAR-100, we artificially generate heterogeneous data distributions using Dirichlet's distribution. When using Non-IID data, we also consider partial device participation such that randomly chosen  $25\%$  of the clients participate in training at every  $\phi \tau'$  iterations. We report the average accuracy across at least three separate runs.

Table 1: (IID data) CIFAR-10 classification results. The number of workers is 128 and the local batch size is 32 in all the experiments. The epoch budget is 300.  

<table><tr><td>LR</td><td>Base aggregation interval: τ&#x27;</td><td>Interval increase factor: φ</td><td>Validation acc.</td><td>Comm. cost</td></tr><tr><td>0.8</td><td>6</td><td>1 (FedAvg)</td><td>88.37 ± 0.02%</td><td>100%</td></tr><tr><td>0.8</td><td>12</td><td>1 (FedAvg)</td><td>84.74 ± 0.05%</td><td>50%</td></tr><tr><td>0.4</td><td>6</td><td>2 (FedLAMA)</td><td>88.41 ± 0.01%</td><td>62.33%</td></tr><tr><td>0.6</td><td>24</td><td>1 (FedAvg)</td><td>80.34 ± 0.3%</td><td>25%</td></tr><tr><td>0.6</td><td>6</td><td>4 (FedLAMA)</td><td>86.21 ± 0.1%</td><td>42.17%</td></tr></table>

Table 2: (IID data) CIFAR-100 classification results. The number of workers is 128 and the local batch size is 32 in all the experiments. The epoch budget is 250.  

<table><tr><td>LR</td><td>Base aggregation interval: τ&#x27;</td><td>Interval increase factor: φ</td><td>Validation acc.</td><td>Comm. cost</td></tr><tr><td>0.6</td><td>6</td><td>1 (FedAvg)</td><td>76.50 ± 0.02%</td><td>100%</td></tr><tr><td>0.6</td><td>12</td><td>1 (FedAvg)</td><td>66.97 ± 0.9%</td><td>50%</td></tr><tr><td>0.5</td><td>6</td><td>2 (FedLAMA)</td><td>76.02 ± 0.01%</td><td>66.01%</td></tr><tr><td>0.6</td><td>24</td><td>1 (FedAvg)</td><td>45.01 ± 1.1%</td><td>25%</td></tr><tr><td>0.5</td><td>6</td><td>4 (FedLAMA)</td><td>76.17 ± 0.02%</td><td>39.91%</td></tr></table>

# 6.1 CLASSIFICATION PERFORMANCE ANALYSIS

We compare the performance across three different model aggregation settings as follows.

- Periodic full aggregation with an interval of  $\tau'$  
Periodic full aggregation with an interval of  $\phi \tau^{\prime}$  
- Layer-wise adaptive aggregation with intervals of  $\tau'$  and  $\phi$

The first setting provides the baseline communication cost, and we compare it to the other settings' communication costs. The third setting is FedLAMA with the base aggregation interval  $\tau'$  and the interval increase factor  $\phi$ . Due to the limited space, we present a part of experimental results that deliver the key insights. More results can be found in Appendix.

Experimental Results with IID Data - We first present CIFAR-10 and CIFAR-100 classification results under IID data settings. Table 1 and 2 show the CIFAR-10 and CIFAR-100 results, respectively. Note that the learning rate is individually tuned for each setting using a grid search, and we report the best settings. In both tables, the first row shows the performance of FedAvg with a short interval  $\tau' = 6$ . As the interval increases, FedAvg significantly loses the accuracy while the communication cost is proportionally reduced. FedLAMA achieves a comparable accuracy to FedAvg with  $\tau' = 6$  while its communication cost is similar to that of FedAvg with  $\phi \tau'$ . These results demonstrate that Algorithm 2 effectively finds the layer-wise interval settings that maximize the communication cost reduction while minimizing the model discrepancy increase.

Experimental Results with Non-IID Data - We now evaluate the performance of FedLAMA using non-IID data. FEMNIST is inherently heterogeneous such that it contains the hand-written digit pictures collected from 3,550 different writers. We use random  $10\%$  of the writers' training samples in our experiments. Table 3 shows the FEMNIST classification results. The base interval  $\tau^{\prime}$  is set to 10. FedAvg  $(\phi = 1)$  significantly loses the accuracy as the aggregation interval increases. For example, when the interval increases from 10 to 40, the accuracy is dropped by  $2.1\% \sim 2.7\%$ . In contrast, FedLAMA maintains the accuracy when  $\phi$  increases, while the communication cost is remarkably reduced. This result demonstrates that FedLAMA effectively finds the best interval setting that reduces the communication cost while maintaining the accuracy.

Table 4 and 5 show the non-IID CIFAR-10 and CIFAR-100 experimental results. We use Dirichlet's distribution to generate heterogeneous data across all the clients. The base aggregation interval  $\tau'$  is set to 6. The interval increase factor  $\phi$  is set to 2 for FedLAMA. Likely to the IID data experiments, we observe that the periodic full averaging significantly loses the accuracy as the model aggregation interval increases, while it has a proportionally reduced communication cost. FedLAMA achieves the comparable classification performance to the periodic full averaging, regardless of the device activation ratio and the degree of data heterogeneity. For CIFAR-100, FedLAMA has a minor accuracy drop compared to the periodic full averaging with  $\tau' = 6$ , however, the accuracy is still much

Table 3: (Non-IID data) FEMNIST classification results. The number of workers is 128 and the local batch size is 32 in all the experiments. The number of training iterations is 2,000.  

<table><tr><td>LR</td><td>Base aggregation interval: τ&#x27;</td><td>Interval increase factor: φ</td><td>active ratio</td><td>Validation acc.</td><td>Comm. cost</td></tr><tr><td rowspan="5">0.04</td><td>10</td><td>1 (FedAvg)</td><td rowspan="5">25%</td><td>86.04 ± 0.01%</td><td>100%</td></tr><tr><td>20</td><td>1 (FedAvg)</td><td>85.38 ± 0.02%</td><td>50%</td></tr><tr><td>10</td><td>2 (FedLAMA)</td><td>86.01 ± 0.01%</td><td>52.83%</td></tr><tr><td>40</td><td>1 (FedAvg)</td><td>83.97 ± 0.02%</td><td>25%</td></tr><tr><td>10</td><td>4 (FedLAMA)</td><td>85.61 ± 0.02%</td><td>29.97%</td></tr><tr><td rowspan="5">0.04</td><td>10</td><td>1 (FedAvg)</td><td rowspan="5">50%</td><td>86.59 ± 0.01%</td><td>100%</td></tr><tr><td>20</td><td>1 (FedAvg)</td><td>85.50 ± 0.02%</td><td>50%</td></tr><tr><td>10</td><td>2 (FedLAMA)</td><td>86.07 ± 0.02%</td><td>53.32%</td></tr><tr><td>40</td><td>1 (FedAvg)</td><td>83.92 ± 0.02%</td><td>25%</td></tr><tr><td>10</td><td>4 (FedLAMA)</td><td>85.77 ± 0.02%</td><td>29.98%</td></tr><tr><td rowspan="5">0.04</td><td>10</td><td>1 (FedAvg)</td><td rowspan="5">100%</td><td>85.74 ± 0.03%</td><td>100%</td></tr><tr><td>20</td><td>1 (FedAvg)</td><td>85.08 ± 0.01%</td><td>50%</td></tr><tr><td>10</td><td>2 (FedLAMA)</td><td>85.40 ± 0.02%</td><td>51.86%</td></tr><tr><td>40</td><td>1 (FedAvg)</td><td>83.62 ± 0.02%</td><td>25%</td></tr><tr><td>10</td><td>4 (FedLAMA)</td><td>84.67 ± 0.02%</td><td>29.98%</td></tr></table>

Table 4: (Non-IID data) CIFAR-10 classification results. The number of workers is 128 and the local batch size is 32 in all the experiments. The number of training iterations is 6,000.  

<table><tr><td>LR</td><td>Base aggregation interval: τ&#x27;</td><td>Interval increase factor: φ</td><td>active ratio</td><td>Dirichlet&#x27;s coeff.</td><td>Validation acc.</td><td>Comm. cost</td></tr><tr><td rowspan="3">0.6</td><td>6</td><td>1 (FedAvg)</td><td rowspan="3">25%</td><td rowspan="3">0.1</td><td>84.02 ± 0.1%</td><td>100%</td></tr><tr><td>24</td><td>1 (FedAvg)</td><td>76.27 ± 0.08%</td><td>25%</td></tr><tr><td>6</td><td>4 (FedLAMA)</td><td>80.58 ± 0.1%</td><td>39.52%</td></tr><tr><td rowspan="3">0.8</td><td>6</td><td>1 (FedAvg)</td><td rowspan="3">25%</td><td rowspan="3">0.5</td><td>87.59 ± 0.2%</td><td>100%</td></tr><tr><td>24</td><td>1 (FedAvg)</td><td>84.03 ± 0.4%</td><td>25%</td></tr><tr><td>6</td><td>4 (FedLAMA)</td><td>86.07 ± 0.1%</td><td>42.40%</td></tr><tr><td rowspan="3">0.8</td><td>6</td><td>1 (FedAvg)</td><td rowspan="3">100%</td><td rowspan="3">0.1</td><td>89.52 ± 0.05%</td><td>100%</td></tr><tr><td>24</td><td>1 (FedAvg)</td><td>84.82 ± 0.06%</td><td>25%</td></tr><tr><td>6</td><td>4 (FedLAMA)</td><td>87.47 ± 0.1%</td><td>42.49%</td></tr><tr><td rowspan="3">0.8</td><td>6</td><td>1 (FedAvg)</td><td rowspan="3">100%</td><td rowspan="3">0.5</td><td>90.53 ± 0.08%</td><td>100%</td></tr><tr><td>24</td><td>1 (FedAvg)</td><td>85.68 ± 0.1%</td><td>25%</td></tr><tr><td>6</td><td>4 (FedLAMA)</td><td>87.45 ± 0.05%</td><td>42.73%</td></tr></table>

Table 5: (Non-IID data) CIFAR-100 classification results. The number of workers is 128 and the local batch size is 32 in all the experiments. The number of training iterations is 6,000.  

<table><tr><td>LR</td><td>Base aggregation interval: τ&#x27;</td><td>Interval increase factor: φ</td><td>active ratio</td><td>Dirichlet&#x27;s coeff.</td><td>Validation acc.</td><td>Comm. cost</td></tr><tr><td rowspan="3">0.4</td><td>6</td><td>1 (FedAvg)</td><td rowspan="3">25%</td><td rowspan="3">0.1</td><td>79.15 ± 0.02%</td><td>100%</td></tr><tr><td>12</td><td>1 (FedAvg)</td><td>76.16 ± 0.05%</td><td>50%</td></tr><tr><td>6</td><td>2 (FedLAMA)</td><td>77.84 ± 0.03%</td><td>63.14%</td></tr><tr><td rowspan="3">0.4</td><td>6</td><td>1 (FedAvg)</td><td rowspan="3">25%</td><td rowspan="3">0.5</td><td>78.81 ± 0.1%</td><td>100%</td></tr><tr><td>12</td><td>1 (FedAvg)</td><td>76.11 ± 0.05%</td><td>50%</td></tr><tr><td>6</td><td>2 (FedLAMA)</td><td>77.78 ± 0.04%</td><td>63.20%</td></tr><tr><td rowspan="3">0.4</td><td>6</td><td>1 (FedAvg)</td><td rowspan="3">100%</td><td rowspan="3">0.1</td><td>79.77 ± 0.04%</td><td>100%</td></tr><tr><td>12</td><td>1 (FedAvg)</td><td>77.71 ± 0.08%</td><td>50%</td></tr><tr><td>6</td><td>2 (FedLAMA)</td><td>79.07 ± 0.1%</td><td>60.48%</td></tr><tr><td rowspan="3">0.4</td><td>6</td><td>1 (FedAvg)</td><td rowspan="3">100%</td><td rowspan="3">0.5</td><td>80.19 ± 0.05%</td><td>100%</td></tr><tr><td>12</td><td>1 (FedAvg)</td><td>77.40 ± 0.06%</td><td>50%</td></tr><tr><td>6</td><td>2 (FedLAMA)</td><td>78.88 ± 0.05%</td><td>61.73%</td></tr></table>

higher than the periodic full averaging with  $\tau' = 12$ . For both datasets, FedLAMA has a remarkably reduced communication cost compared to the periodic full averaging with  $\tau' = 6$ .

# 6.2 COMMUNICATION EFFICIENCY ANALYSIS

We analyze the total number of communications and the accumulated data size to evaluate the communication efficiency of FedLAMA. Figure 2 shows the total number of communications at the individual layers. The  $\tau'$  is set to 6 and  $\phi$  is 2 for FedLAMA. The key insight is that FedLAMA increases the aggregation interval mostly at the output-side large layers. This means the  $d_{l}$  value shown in Equation (2) at the these layers are smaller than the others. Since these large layers take up most of the total model parameters, the communication cost is remarkably reduced when their aggregation intervals are increased. Figure 3 shows the layer-wise local data size shown in Equation 9. FedLAMA shows the significantly smaller total data size than FedAvg. The extra computational

![](images/938baec991fe29280595b8ea5c0327b8735449a25fc363836bf84e53ee14caed.jpg)  
Figure 2: The number of communications at the individual layers. The communications are counted during the whole training (non-IID data).

![](images/deef3ef2f73bf8921994d0736fc31030b14bb804ff648b6b1c384a6925002324.jpg)  
Figure 3: The total data size (communication cost) that correspond to Figure 2. The data size comparison clearly shows where the performance gain of FedLAMA comes from.

cost of FedLAMA is almost negligible since it calculates  $d_{l}$  after each communication round only. Therefore, given the virtually same computational cost, FedLAMA aggregates the local models at a cheaper communication cost, and thus it improves the scalability of Federated Learning.

We found that the amount of the reduced communication cost was not strongly affected by the degree of data heterogeneity. As shown in Table 4 and 5, the reduced communication cost is similar across different Dirichlet's coefficients and device participation ratios. That is, FedLAMA can be considered as an effective model aggregation scheme regardless of the degree of data heterogeneity.

# 7 CONCLUSION

We proposed a layer-wise model aggregation scheme that adaptively adjusts the model aggregation interval at run-time. Breaking the convention of aggregating the whole model parameters at once, this novel model aggregation scheme introduces a flexible communication strategy for scalable Federated Learning. Furthermore, we provide a solid convergence guarantee of FedLAMA under the assumptions on the non-convex objective functions and the non-IID data distribution. Our empirical study also demonstrates the efficacy of FedLAMA for scalable and accurate Federated Learning. Harmonizing FedLAMA with other advanced optimizers, gradient compression, and low-rank approximation methods is a promising future work.

# 8 CODE OF ETHICS

Our work does not deliver potentially harmful insights or conflicts of interests. We also do not find any potential inappropriate application or privacy/security issues. The datasets we used in our study are all public benchmark datasets, and our source code will be opened once the paper is accepted.

# 9 REPRODUCIBILITY STATEMENT

The software versions, implementation details, hyper-parameter settings can be found in the first two paragraphs of Section 6. The entire source code used in our experiments will be published as an open source once the paper is accepted. We believe one can exactly reproduce our experimental results following the provided descriptions.

# REFERENCES

Alyazed Albasyoni, Mher Safaryan, Laurent Condat, and Peter Rictarik. Optimal gradient compression for distributed and federated learning. arXiv preprint arXiv:2010.03246, 2020.  
Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. Qsgd: Communication-efficient sgd via gradient quantization and encoding. Advances in Neural Information Processing Systems, 30:1709-1720, 2017.  
Dan Alistarh, Torsten Hoefler, Mikael Johansson, Sarit Khirirat, Nikola Konstantinov, and Cedric Renggli. The convergence of sparsified gradient methods. arXiv preprint arXiv:1809.10505, 2018.  
Andrew Brock, Theodore Lim, James M Ritchie, and Nick Weston. Freezeout: Accelerate training by progressively freezing layers. arXiv preprint arXiv:1706.04983, 2017.  
Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Konečný, H Brendan McMahan, Virginia Smith, and Ameet Talwalkar. Leaf: A benchmark for federated settings. arXiv preprint arXiv:1812.01097, 2018.  
Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre Van Schaik. Emmist: Extending mnist to handwritten letters. In 2017 International Joint Conference on Neural Networks (IJCNN), pp. 2921-2926. IEEE, 2017.  
Enmao Diao, Jie Ding, and Vahid Tarokh. Heterofl: Computation and communication efficient federated learning for heterogeneous clients. arXiv preprint arXiv:2010.01264, 2020.  
Kelam Goutam, S Balasubramanian, Darshan Gera, and R Raghunatha Sarma. Layerout: Freezing layers in deep neural networks. SN Computer Science, 1(5):1-9, 2020.  
Priya Goyal, Piotr Dólar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
Arthur Gretton, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf. Measuring statistical dependence with hilbert-schmidt norms. In International conference on algorithmic learning theory, pp. 63-77. Springer, 2005.  
Farzin Haddadpour, Mohammad Mahdi Kamani, Mehrdad Mahdavi, and Viveck R Cadambe. Local sgd with periodic averaging: Tighter analysis and adaptive synchronization. arXiv preprint arXiv:1910.13598, 2019.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In International Conference on Machine Learning, pp. 5132-5143. PMLR, 2020.

Jakub Konečný, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave Bacon. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492, 2016.  
Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International Conference on Machine Learning, pp. 3519-3529. PMLR, 2019.  
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
Adarsh Kumar, Arjun Balasubramanian, Shivaram Venkataraman, and Aditya Akella. Accelerating deep learning inference via freezing. In 11th {USENIX} Workshop on Hot Topics in Cloud Computing (HotCloud 19), 2019.  
Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. arXiv preprint arXiv:1812.06127, 2018.  
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273-1282. PMLR, 2017.  
Ari S Morcos, Maithra Raghu, and Samy Bengio. Insights on representational similarity in neural networks with canonical correlation. arXiv preprint arXiv:1806.05759, 2018.  
Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. arXiv preprint arXiv:1706.05806, 2017.  
Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jabbabaie, and Ramtin Pedarsani. Fedpaq: A communication-efficient federated learning method with periodic averaging and quantization. In International Conference on Artificial Intelligence and Statistics, pp. 2021-2031. PMLR, 2020.  
Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. Practical low-rank communication compression in decentralized deep learning. In NeurIPS, 2020.  
Hongyi Wang, Scott Sievert, Zachary Charles, Shengchao Liu, Stephen Wright, and Dimitris Papailopoulos. Atomo: Communication-efficient learning via atomic sparsification. arXiv preprint arXiv:1806.04090, 2018.  
Hongyi Wang, Saurabh Agarwal, and Dimitris Papailiopoulos. Pufferfish: Communication-efficient models at no extra cost. arXiv preprint arXiv:2103.03936, 2021.  
Jianyu Wang and Gauri Joshi. Adaptive communication strategies to achieve the best error-routine trade-off in local-update sgd. arXiv preprint arXiv:1810.08313, 2018.  
Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H Vincent Poor. Tackling the objective inconsistency problem in heterogeneous federated optimization. arXiv preprint arXiv:2007.07481, 2020.  
Jianqiao Wangni, Jialei Wang, Ji Liu, and Tong Zhang. Gradient sparsification for communication-efficient distributed optimization. arXiv preprint arXiv:1710.09854, 2017.  
Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. arXiv preprint arXiv:1705.07878, 2017.  
Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes. arXiv preprint arXiv:1904.00962, 2019.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.  
Minjia Zhang and Yuxiong He. Accelerating training of transformer-based language models with progressive layer dropping. arXiv preprint arXiv:2010.13369, 2020.
