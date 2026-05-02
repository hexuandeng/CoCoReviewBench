# SCALABLE DEEP NEURAL NETWORKS VIA LOW-RANK MATRIX FACTORIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Compressing deep neural networks (DNNs) is important for real-world applications operating on resource-constrained devices. However, it is difficult to change the model size once the training is completed, which needs re-training to configure models suitable for different devices. In this paper, we propose a novel method that enables DNNs to flexibly change their size after training. We factorize the weight matrices of the DNNs via singular value decomposition (SVD) and change their ranks according to the target size. In contrast with existing methods, we introduce simple criteria that characterize the importance of each basis and layer, which enables to effectively compress the error and complexity of models as little as possible. In experiments on multiple image-classification tasks, our method exhibits favorable performance compared with other methods.

# 1 INTRODUCTION

As part of the great progress made in deep learning, deep neural network (DNN) models with higher performance have been proposed for various machine-learning tasks (LeCun et al., 2015). However, these performance improvements require a higher number of parameters and greater computational complexity. Therefore, it is important to compress them without sacrificing the performance for running the models on resource-constrained devices.

Han et al. (2016) reduced the memory requirement for devices by pruning and quantizing weight coefficients after training the models. Howard et al. (2017); Sandler et al. (2018); Howard et al. (2019) used factorized operations called depth-wise and point-wise convolutions in a proposal for light-weight models suited to mobile devices. However, these methods require pre-defined network structures and pruning the model weights after training. Recently, automated frameworks, such as the so-called neural architecture search (NAS) (Zoph & Le, 2017), have been proposed. Tan et al. (2019) proposed a NAS method to accelerate the inference speed on smartphones by incorporating resource-related constraints into the objective function. Stamoulis et al. (2019) significantly reduced the search costs for NAS by applying a gradient-based search scheme with a superkernel that shares weights for multiple convolutional kernels.

However, the models trained by these methods are dedicated to specific devices, and thus do not possess the ability to be reconfigured for use on different devices. In order to change the model size, it is necessary to re-train them according to the resources of the target devices. For example, it has been reported that the inference speed when operating the same model on different devices differs according to the computing performance and memory capacity of the hardware accelerator (Ignatov et al., 2018). Therefore, it is desirable that the model size can be flexibly changed according to the resources of the target devices without re-training the model, which we refer to as scalability in this paper.

To this end, Yu et al. (2019) introduced switchable batch normalization (BN) (Ioffe & Szegedy, 2015), which switches BN layers according to pre-defined widths, and proposed "slimmable" networks whose width can be changed after training. Moreover, Yu & Huang (2019) proposed universally slimmable networks (US-Nets) that extend slimmable networks to arbitrary widths. However, since these methods directly reduce the width (i.e., dimensionality) in each layer, the principal components are not taken into account. In addition, they reduce the width uniformly across all layers, which ignores differences in the importance of different layers.

![](images/3d0c6731e56c8a86193038b0d02a5c5b0bf07254bc3adf23d0fb46b7485ba22c.jpg)  
Figure 1: An illustration of our scalable neural networks. Each weight matrix in the neural network is factorized into two matrices of rank  $R$  via SVD, which leads to having sub-layers of width  $R$  between the original layers. Since the number of parameters in each layer is changed from  $mn$  to  $(m + n)R$  by this factorization, we can compress the network to an arbitrary size while changing the rank  $r$  ( $\leq R$ ) within the range  $1 \leq r \leq mn / (m + n)$ .

In this paper, we propose a novel method that enables DNNs to flexibly change their size after training. We factorize a weight matrix in each layer into two low-rank matrices after training the DNNs via singular value decomposition (SVD). By changing the rank in each layer, our method can scale the model to an arbitrary size (Figure 1). Our contributions are as follows.

- We do not directly reduce the width but instead reduce the redundant basis in the column space of the weight matrix, which prevents the feature map in each layer from losing important features.  
- We introduce simple criteria that characterize the importance of each basis and layer, namely, the error- and complexity-based criteria. These enable to effectively compress the error and complexity of the models as little as possible.  
- We facilitate the performance of low rank networks with the following methods: a learning procedure that simultaneously minimizes losses for both the full and low rank networks, and the mean & variance correction for each BN layer according to the given rank.

In the experiments on image-classification tasks of the CIFAR-10/100 (Krizhevsky, 2009) datasets using deep convolutional neural networks (CNNs), our method exhibits better performance for up to approximately  $75\%$  compressed models than slimmable networks and US-Nets. In the following, we first describe the details of our method (Section 2) and briefly review related works (Section 3). Then, we give some experimental results (Section 4) and conclude the paper (Section 5).

# 2 METHODS

In this section, we first give an overview then describe the details of the inference and learning methods.

# 2.1 OVERVIEW

For a layer in the network, let  $\pmb{y} = \pmb{W}^{\top}\pmb{x} \in \mathbb{R}^{n}$  be an output vector given by linear transformation of an input vector  $\pmb{x} \in \mathbb{R}^{m}$  with a weight matrix  $\pmb{W} \in \mathbb{R}^{m \times n}$ , where  $m$  and  $n$  are the numbers of input and output nodes, respectively. Let  $R$  be the rank of the weight matrix, with  $1 \leq R \leq \min(m,n)$ . Given  $\pmb{U} = (\pmb{u}_1, \dots, \pmb{u}_R) \in \mathbb{R}^{m \times R}$  and  $\pmb{V} = (\pmb{v}_1, \dots, \pmb{v}_R) \in \mathbb{R}^{n \times R}$  as matrices that have left and right singular vectors (i.e., bases) in columns, and  $\pmb{S} = \mathrm{diag}(\sigma_1, \dots, \sigma_R) \in \mathbb{R}^{R \times R}$  as a matrix composed of singular values in diagonal components, we can formulate the truncated-SVD as  $\pmb{W} = U\pmb{S}\pmb{V}^{\top}$ .

An example of our scalable neural networks with fully connected layers is shown in Figure 1. After the training, each weight matrix in the network is factorized into two matrices of rank  $R$  via SVD, and we control this value to change the model size. This can be viewed as inserting a sub-layer between the original layers and changing its width  $R$ . For the convolutional tensor  $\mathbf{W} \in \mathbb{R}^{K_w \times K_h \times C_{in} \times C_{out}}$  of kernel width  $K_w$ , kernel height  $K_h$ , input channels  $C_{in}$ , and output channels  $C_{out}$ , we first transform it to the matrix form  $\mathbf{W} \in \mathbb{R}^{K_w K_h C_{in} \times C_{out}}$  and apply SVD as in Zhang et al. (2016); Wen et al. (2017). This yields two layers with a tensor  $\mathbf{W}_1 \in \mathbb{R}^{K_w \times K_h \times C_{in} \times R}$  and a tensor  $\mathbf{W}_2 \in \mathbb{R}^{1 \times 1 \times R \times C_{out}}$ . The number of parameters in each layer becomes  $(m + n)R$  by this factorization. Thus, we can compress the network to an arbitrary size by changing the rank  $r (\leq R)$  within the range  $1 \leq r \leq mn / (m + n)$ .

Associated with changing the rank, the monotonicity of approximation error holds for each layer.

Proposition 2.1. Let  $\pmb{W}_r = \pmb{U}_r\pmb{S}_r\pmb{V}_r^\top$  be a rank-  $r$  approximation using the truncated-SVD for  $\pmb{W}$  and let  $\pmb{y}_r = \pmb{W}_r^\top \pmb{x}$ . Then,  $\| \pmb{y} - \pmb{y}_1\|^2 \geq \dots \geq \| \pmb{y} - \pmb{y}_r\|^2 \geq \| \pmb{y} - \pmb{y}_{r + 1}\|^2 \geq \dots \geq \| \pmb{y} - \pmb{y}_R\|^2 \geq 0$ .

The proof is given in Appendix A. According to the above, errors between an original output  $\mathbf{y}$  and its approximation  $\mathbf{y}_r$  monotonically decrease as the rank increases. Hence, it can be expected the performance of the entire network will scale with the model size, which is controlled by the rank in our method.

# 2.2 INFERENCE

# 2.2.1 RANK SELECTION

Given a target size for a model, we select the rank of each layer by reference to the following criteria.

Error-based criterion. According to Eq. (5) in Appendix A, the error associated with a rank-1 decrease is given by  $|\pmb{v}^{\top}\pmb{y}| = \sigma |\pmb{u}^{\top}\pmb{x}| = \sigma \| \pmb{x}\| |\cos \theta |$ . This implies that the error depends on the singular value  $\sigma$  and the cosine similarity between an input vector  $\pmb{x}$  and the corresponding left singular vector  $\pmb{u}$ . Based on this, we consider how to compress the model with as little error as possible by reducing the bases that induce lower errors. It has been reported that networks with BN layers and ReLUs (rectified linear units) (Glorot et al., 2011) possess the scale-invariance property (Arora et al., 2019). Thus, the error  $|\pmb{v}^{\top}\pmb{y}|$  should be normalized by the scale of  $\pmb{y}$  in each layer. Exploiting the fact that  $\| \pmb{y}\| \leq \| \pmb{W}\| _2\| x\|$ , we normalize it as  $|\pmb{v}^{\top}\pmb{y}| / \| \pmb{W}\| _2\| x\| = \sigma |\cos \theta | / \| \pmb{W}\| _2\in [0,1]$ , where  $\| W\| _2$  is the spectrum norm of  $W$  (i.e., the maximum singular value).

Computing the cosine similarities is costly because it requires the whole input  $x$  over the dataset in each layer. Therefore, we omit it and simply use the following criterion for selecting the rank:

$$
C _ {1} (\ell , k) = \sigma_ {k} ^ {(\ell)} / \| \boldsymbol {W} ^ {(\ell)} \| _ {2}, \tag {1}
$$

where  $\ell$  is a layer index. This is equivalent to keeping  $\| W^{(\ell)} - W_r^{(\ell)}\|_F^2$  small in each layer. We consider this is a simple but effective criterion for the following reasons. First, Arora et al. (2018) have reported that the subspace spanned by each layer's weight vectors and the subspace spanned by their input vectors both become implicitly low rank and correlated after training. In other words, there should be many small singular values in each layer's weight matrix. Second, the principal directions of the weights are correlated with those of the inputs. Thus, by reducing the bases that correspond to smaller singular values, we can reduce by a large number of ranks without significantly increasing the errors. Moreover, the cosine similarities are expected to be higher for large singular values, meaning that our method can reflect the principal directions of the data distribution even if we only use the singular values of the weight matrices as the criterion.

Complexity-based criterion. We achieve a high compression rate by reducing the rank in layers that have a large number of parameters and multiplications (MACs). For convolutional layers, the numbers of parameters, excluding biases and the MACs, are given by  $P = K_wK_hC_{in}C_{out}$  and  $M = PHW$  for a feature map of height  $H$  and width  $W$ , respectively. We use the following as a complexity-based criterion.

$$
C _ {2} (\ell) = \left(1 - P ^ {(\ell)} / \sum_ {i} P ^ {(i)}\right) \left(1 - M ^ {(\ell)} / \sum_ {i} M ^ {(i)}\right) \tag {2}
$$

By coupling the above two criteria, we reduce the bases with lower values of  $C(\ell, k) = C_1(\ell, k) C_2(\ell)$  across the entire network. In practice, we compute the criterion for all bases after training. Then, we sort them in ascending order and store as a list. The only necessary step for selection is to reduce the first  $d$  bases in the list, where  $d$  is determined by the target model size. The algorithm is given in Appendix B.

# 2.2.2 BN CORRECTION

As pointed out by Yu et al. (2019), the means and variances of the BN layers should be corrected when the model size is changed. Suppose that a BN layer is inserted right after the convolutional layer, and that the mean and variance of  $\pmb{y}$  ( $= \pmb{W}^{\top} \pmb{x}$ ) are normalized in the BN layer. Then, we should correct those values according to the rank-  $r$  approximation of  $\pmb{y}$  (i.e.,  $\pmb{y}_r$ ). Because  $\pmb{y}_r = V_r S_r U_r^\top \pmb{x}$ ,  $\pmb{y}_r$  lies in the rank-  $r$  subspace spanned by the columns of  $V_r$ . Hence, letting  $\pmb{\mu}$  and  $\pmb{\Sigma}$  be, respectively, the population mean and covariance matrix for  $\pmb{y}$ , we can exactly compute their projection onto the subspace as  $\pmb{\mu}_r = V_r V_r^\top \pmb{\mu}$  and  $\pmb{\Sigma}_r = V_r V_r^\top \pmb{\Sigma} V_r V_r^\top$  (note that diagonal components are extracted from  $\pmb{\Sigma}_r$  for the BN layer). For practical reasons, we compute  $\pmb{\mu}$  and  $\pmb{\Sigma}$  for each layer after training (Ioffe & Szegedy, 2015). Because  $\pmb{\Sigma}$  has  $n(n + 1)/2$  extra parameters to store, we keep  $V_R^\top \pmb{\Sigma} V_R$  instead, which reduces the number of parameters to  $R(R + 1)/2$ . At the time of inference, we can correct the mean and variance according to the ranks in each layer. On the other hand, if a list of candidate model sizes is available in advance, we can retain the means and variances for those models as Yu & Huang (2019). We compare both methods in Section 4.

# 2.3 LEARNING

Although our scalable neural networks can operate regardless of learning methods, we propose a method to gain better performance. We simultaneously minimize losses for both the full-rank and the low-rank networks as follows.

$$
\min  _ {\boldsymbol {\Theta}} \frac {1}{B} \sum_ {b = 1} ^ {B} \left\{\left(1 - \lambda\right) \mathcal {L} \left(\mathcal {D} _ {b}, \mathcal {W}, \Theta\right) + \lambda \mathcal {L} \left(\mathcal {D} _ {b}, \widetilde {\mathcal {W}}, \Theta\right) \right\} \tag {3}
$$

Here,  $\mathcal{L}(\cdot)$  is a loss function,  $\mathcal{D}_b$  is a set of training samples in a mini-batch,  $B$  is the batch size, and  $\lambda \in [0,1]$  is a hyperparameter for balancing between the two losses. For this,  $\mathcal{W} = \{\pmb{W}^{(\ell)}\}_{\ell=1}^{L}$ ,  $\widetilde{\mathcal{W}}$ , and  $\Theta$  are sets of  $L$  weight matrices, their low-rank approximations, and other trainable parameters (e.g., biases), respectively. We additionally propagate each mini batch to a low-rank network in which the weights are generated by SVD. Because  $\pmb{W}_r^{(\ell)} = \pmb{U}_r^{(\ell)}\pmb{U}_r^{(\ell)\top}\pmb{W}^{(\ell)}$ , the gradient with respect to  $\pmb{W}^{(\ell)}$  can be computed as follows:

$$
(1 - \lambda) \frac {\partial \mathcal {L} (\mathcal {D} _ {b} , \mathcal {W} , \Theta)}{\partial \boldsymbol {W} ^ {(\ell)}} + \lambda \boldsymbol {U} _ {r} ^ {(\ell)} \boldsymbol {U} _ {r} ^ {(\ell) \top} \frac {\partial \mathcal {L} (\mathcal {D} _ {b} , \widetilde {\mathcal {W}} , \Theta)}{\partial \boldsymbol {W} _ {r} ^ {(\ell)}}. \tag {4}
$$

$\Theta$  is shared between the full- and low-rank networks, so the gradients are simply computed from the  $\lambda$ -weighted average for those of both networks. At each iteration step, we randomly select the model size for the low-rank network by sampling the rate of rank reduction  $Z$  from a uniform distribution  $\mathcal{U}(\alpha_l,\alpha_u)$  with  $0 < \alpha_{l} < \alpha_{u}\leq 1$ . Then, letting  $R^{(\ell)}$  be the rank of  $\pmb{W}^{(\ell)}$ , we reduce  $(1 - Z)\sum_{\ell = 1}^{L}R^{(\ell)}$  bases across the entire network using the criterion mentioned in subsection 2.2.1. In a later section, we experimentally investigate the effects of the parameters  $\lambda$ ,  $\alpha_{l}$ , and  $\alpha_{u}$  in the experiment.

Arora et al. (2018); Suzuki (2019) derived the generalization error bound for DNNs under a condition that the trained network has near low-rank weight matrices. They proved that the condition contributes not only to yield a better generalization error bound for the non-compressed network but also to compress the network efficiently. That motivates our approaches: a learning which aims to facilitate the performance of the low-rank networks as well as that of the full-rank network, and a compression scheme which reduces the redundant basis obtained via SVD.

![](images/73548ce1c5524c2fa272d9081eaa414ceb7657bba24af32e358154cd0bfb64fe.jpg)  
Figure 2: Training results for VGG-15 on CIFAR-10. (Left) Full-rank validation accuracy by changing  $\alpha_{l}$  with  $\alpha_{u} = 1.0$ . (Center) Full-rank validation accuracy by changing  $\alpha_{u}$  with  $\alpha_{l} = 0.01$ . (Right) Maximum singular value for each basis index in a full-rank model. "base" indicates normal learning as our baseline.

![](images/2bfb67af74a4f4bc6b76817efafce786bd147c2df5ebcb5224409d40cb9ba9ce.jpg)

![](images/b44dd6e486cae6f9c8cd58034ad186386f06a615481225cdd4bd1bfbc6a0cfa3.jpg)

# 3 RELATED WORK

Low-rank approximation & regularization. Compression methods based on low-rank approximation have been proposed in the literature. Denil et al. (2013); Tai et al. (2016); Ioannou et al. (2016) trained networks after factorizing the weight matrix into a low-rank form. Ioannou et al. (2016) achieved a high compression rate by factorizing a convolutional kernel of  $K_w \times K_h$  into  $K_w \times 1$  and  $1 \times K_h$ . Denton et al. (2014); Lebedev et al. (2015); Kim et al. (2016) proposed methods that use tensor factorization without rearranging the convolutional tensor into the matrix form. Yu et al. (2017) further improved the compression rate by incorporating sparseness into the low-rank constraint. Zhang et al. (2016); Li & Shi (2018) took resource-related constraints into account to automatically select an appropriate rank. Each of these methods trains a network with pre-defined ranks or compress redundant networks by applying complicated optimizations under a given target size for the model. That is, those methods would require re-training to reconfigure the models for different devices.

Kliegl et al. (2017) utilized trace-norm regularization as a low-rank constraint in learning the network. Wen et al. (2017) proposed a novel method called force regularization to obtain the low-rank weights. The performance of these methods depends on a hyperparameter to adjust strength of regularization. It is difficult to decide on an appropriate range for the hyperparameter in advance, meaning that selection requires trial and error to achieve a particular model size.

Scalable neural networks. Chen et al. (2018) represented the data flow in ResNet-type structures (He et al., 2016) as ordinary differential equations (ODEs), and proposed a Neural-ODEs, which can be used to arbitrarily control the computational cost in the depth direction. Zhang et al. (2019) also obtained scalability in the depth direction by allowing pre-defined intermediate layers to be bypassed.

Yu et al. (2019); Yu & Huang (2019) proposed slimmable networks and US-Nets, which are scalable in the width direction. Their works are closely related to ours, but there are differences in some aspects. First, since their methods directly and uniformly reduce the width for every layer, the principal components are not taken into account, and the relative importance of each layer is not considered. Second, for US-Nets in particular, they introduced a "sandwich rule" to keep the performance for an arbitrary width. However, this rule does not guarantee monotonicity of the error with increasing layer width. In the next section, we compare our method with them.

# 4 EXPERIMENTS

We evaluate our methods on the image-classification tasks of CIFAR-10/100 (Krizhevsky, 2009) datasets using deep CNNs. The CIFAR-10/100 datasets contain  $32 \times 32$  images for object recognition including 10 and 100 classes, respectively. Each dataset contains 50K images for training and 10K images for validation. We implement our method with TensorFlow (Abadi et al., 2015).

![](images/2c2da61e6127b121f67cc5e3f70f93451ec035af754bbf43e7ec6121a95ca968.jpg)  
Figure 3: Inference results for VGG-15 on CIFAR-10. (a) # of parameters vs. validation accuracy. (b) # of MACs vs. validation accuracy. “+learn” indicates results with our learning method. “+bn” and “+bnRe” indicate results with our BN correction and those with recomputation, respectively. “uni”, “c1”, “c2”, and “c1c2” in the bracket indicate rank selection by a uniform method, by Eq. (1), by Eq. (2), and by both, respectively. We do not apply “c2” to learning because it slightly decreases the full-rank accuracy.

![](images/2e9a46b322a4292b6bc7022c33cee726a10158247fff8d0f432dd451a8d82fe5.jpg)

![](images/d7a06695b257eddc01e19c5235eea6fff7dc7dde6b830a61d1dcf4eabb789175.jpg)  
Figure 4: Results of rank selection for VGG-15 on CIFAR-10. (Left) Layer-wise importance with complexity-based criterion (normalized to sum 1). (Right) Remaining rank ratio per layer by different selection methods. "uni", "c1", and "c1c2" indicate selection results by a uniform method, by Eq. (1), and by Eq. (1 & 2), respectively. We do not reduce parameters for the last fully connected layer for the uniform method because it significantly decreases accuracy.

![](images/570633fb33a89c83de3bdd0b0b26c2196f029e795451f6516178fbbdd285b8db.jpg)

# 4.1 ABLATION STUDY

We test each component in our method on the CIFAR-10 dataset. We use the same baseline setup as in Zagoruyko & Komodakis (2016), which is summarized in Table 1 in Appendix C. Unless otherwise specified, we report the average result from 5 trials with different random seeds. We adopt a VGG-like network with 15 layers (Zagoruyko, 2015; Liu et al., 2017)  $^{2}$ , which we refer to as VGG-15 below.

Firstly, we evaluate our learning method for various values of the parameters  $\alpha_{l}$  and  $\alpha_{u}$ , fixing  $\lambda = 0.5$ . Our method requires SVD at each iteration step, which makes it costly. To address this, we suppose that the weight subspaces are not drastically changed at each step and recompute the SVD after every two steps, reusing the results to speed up the training. We illustrate the validation accuracy of a full-rank model for different values of  $\alpha_{l}$  (resp.,  $\alpha_{u}$ ) with  $\alpha_{u} = 1.0$  (resp.,  $\alpha_{l} = 0.01$ ) fixed, on the left (resp., center) of Figure 2. It can be observed that smaller values of  $\alpha_{l}$  and larger values of  $\alpha_{u}$  are better. This can be interpreted as indicating that it is better for a full-rank model to learn with various low-rank models than to learn with models biased to a specific range of ranks. Thus, we set  $\alpha_{l} = 0.01$  and  $\alpha_{u} = 1.0$  for the other experiments described below. On the right side of Figure 2, we show the maximum singular value for each basis index in a full-rank model<sup>3</sup>. We can see that our learning method obtains smaller singular values than the baseline. This implies that our learning method has an effect similar to trace-norm regularization (Kliegl et al., 2017), suggesting that we can suppress the errors produced by reducing the bases.

![](images/cef0699e1204d04371d55fec358a1827c49eccd84d0df0720e4932cd6a30bf91.jpg)  
(a) # of parameters vs. val. acc. (US-Nets)

![](images/4cc65995091aa18dd29d9d18904f7ce6a20c12007e02d601c09d630546fe4dc8.jpg)  
(b) # of parameters vs. val.acc. (Ours)  
Figure 5: Scalability for ResNet-34 on CIFAR-100 dataset when using (a) US-Nets and (b) our method. We show the results for each of 5 trials with different random seeds in this figure.

Next, we evaluated the performance of our inference method for various model sizes. In Figure 3, we illustrate the inference results for validation data with various number of parameters and MACs. In the figure, "infer (uni)" indicate the results obtained by uniformly reducing the basis in each layer. Concretely, with a global rank ratio  $G \in (0,1]$ , we reduce  $(1 - G)R^{(\ell)}$  bases in order from the one corresponding to the smallest singular value. Despite the method being simple, the accuracy changes almost smoothly, and it can be confirmed that there are accuracy scales against changes in the model size. This can be considered as due to the monotonicity of errors, which is formalized in Proposition 2.1. Additionally, the performance is also improved with our learning method by applying uniform rank selection and by using our BN correction. Furthermore, the performance with respect to the parameters is improved when we apply the error- and complexity-based criteria for rank selection to both inference and learning (in the figure, "c1" and "c1c2"). However, the performance with respect to the MACs is dropped by changing the rank selection from uniform ("uni") to error-based ("c1"). As shown on the left side of Figure 4, it is more effective for decreasing MACs to reduce the parameters in shallower layers, which have large feature maps. However, the error-based criterion tends to reduce the parameters in deeper layers because those tend to be low rank. When both criteria are applied (in the figure, "c1c2"), the performance is also improved for the MACs. We show the rank-selection results for different criteria on the right side of Figure 4. It can be confirmed that the ranks are decreased for 4, 6, 7, 9, and 10 layers which have large MACs in the case with both criteria ("c1c2") relative to the case with only the error-based criterion ("c1"). For the BN correction, our method is effective, but it is better with a method that recomputes the means and variances for given ranks ("bnRe"). Because our method is layer-by-layer correction, this probably occurs because our method cannot fully correct for the inter-layer gap, with the statistics of the deep layer changing due to the reduction of rank in the shallow layer.

Additionally, we investigate the effect of a parameter  $\lambda$ . We evaluate the validation accuracy with respect to the number of paramters for  $\lambda \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$  with VGG-15 and ResNet-34 on the CIFAR-10/100 datasets. The results are shown in Figure 7 in Appendix D. We consider that there is a trade-off between the performance of full- and low-rank models, which depends on  $\lambda$ .

# 4.2 COMPARISON WITH SLIMMABLE NETWORKS

We compare our method with slimmable networks (Yu et al., 2019) and US-Nets (Yu & Huang, 2019) in terms of performance on the CIFAR-10/100 datasets. We adopt VGG-15 and ResNet34 (He et al., 2016). We implement the models based on the Yu's code, available at https://github.com/JiahuiYu/slimmable_networks (written in PyTorch (Paszke et al., 2017)). US-Nets is trained with 2 random widths between the lower and upper width and in-place distillation (Yu & Huang, 2019), then BN-calibration (Yu & Huang, 2019) is applied to each of the slimmable networks and US-Nets after training. For our method, we incorporate all components into the comparisons and adopt BN correction with recomputation. We train the models using  $\lambda = 0.4$  and the same setup as in the previous subsection. In the following, we report the results for models after the last iteration in training.

First, we compare the scalability of ResNet-34 on the CIFAR-100 dataset. We illustrate the inference results over various numbers of parameters for 5 models trained with different random seeds in Figure 5. The results in the figure show that US-Nets are unstable, which is a problem for practical

![](images/fedde05316bba1d85d4286aa14b6dbc67b310c640f20c70e7c5db1acb992b58d.jpg)

![](images/18e33f8fe009083903a2e3a1a1da48450a8b11d9b3a27e6e6fc750461d780377.jpg)

![](images/ddfd55313ed6d45de0823b91dbccb3a25d3e884bd187d782b2c5b946641b0532.jpg)  
Figure 6: Comparisons with slimmable networks (Yu et al., 2019) and US-Nets (Yu & Huang, 2019) for VGG-15 and ResNet-34 on CIFAR-10/100 datasets. (Left) # of parameters vs. validation accuracy. (Right) # of MACs vs. validation accuracy.

![](images/6affe3120566333cb58e187c8b8eb5876caf9bba51946c7a9d21288056cc13a4.jpg)

use. This instability is because US-Nets do not have monotonic error changes in each layer, a property that our method ensures. Next, we show the results for comparison of VGG-15 on CIFAR-10 and ResNet-34 on CIFAR-100 in Figure 6. Additional results are shown in Figure 8 in Appendix D. The notations "base (Yu's code)" and "base (our code)" indicate the baseline results obtained by the Yu's code and our code with the same setup. Our baseline is slightly better than the Yu's baseline. We consider this to be due to differences in the framework. Comparing the results with those for VGG-15 on CIFAR-10, our method tends to be more accurate in terms of the number of parameters than in terms of the number of MACs. Since deep layers have more parameters than shallow layers, the rank in deep layers tends to be lower than in shallow layers, resulting in more paramters reduced in deep layers by our method. In contrast, US-Nets reduce the width uniformly across layers, which may contribute to reducing the number of MACs. However, reducing the number of MACs does not necessarily lead to cut the inference cost dominantly, depending on the target device (Yang et al., 2018). Although we only consider the number of parameters and MACs as the complexity metrics in this paper, other metrics such as memory footprint, memory access cost, and runtime latency should be taken into account for validating the effectiveness in practical use case (Tan et al., 2019; Sandler et al., 2018; Dai et al., 2019).

We can see that the accuracy of our method is lower than that of US-Nets when the compression rate is extremely high. Our method uses SVD and reduces the bases, which means it does not change the number of inputs and outputs (i.e., the in and out dimensionalities). Because the number of parameters in each layer is  $(m + n)r$ , it decreases linearly with respect to the rank. US-Nets reduce both input and output dimensionality, meaning that the number of parameters is decreased at a quadratic rate. This makes it easier for US-Nets to achieve extremely high compression. However, our method is better in larger regimes. In particular, for ResNet-34 on CIFAR-100, the performance of slimmable networks and US-Nets on the full-size model are decreased, while our method does not decrease performance much.

# 5 CONCLUSIONS

We proposed a novel method that enables DNNs to flexibly change their size after training. We used to factorize the weight matrix for each layer into two low-rank matrices after training the DNNs. By changing the rank in each layer, our method can scale the model to an arbitrary size. We introduced simple criteria for characterizing the importance of each basis and layer; these are the error- and complexity-based criteria. Those criteria enabled effectively compressing models without introducing much error. In experiments on multiple image-classification tasks using deep CNNs, our method exhibited good performance relative to that of other methods.

# REFERENCES

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. URL http://tensorflow.org/. Software available from tensorflow.org.  
Sanjeev Arora, Rong Ge, Behnam Neyshabur, and Yi Zhang. Stronger generalization bounds for deep nets via a compression approach. In International Conference on Machine Learning (ICML), pp. 254-263, 2018.  
Sanjeev Arora, Zhiyuan Li, and Kaifeng Lyu. Theoretical analysis of auto rate-tuning by batch normalization. In International Conference on Learning Representations (ICLR), 2019.  
Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems (NeurIPS), pp. 6571-6583. 2018.  
Xiaoliang Dai, Peizhao Zhang, Bichen Wu, Hongxu Yin, Fei Sun, Yanghan Wang, Marat Dukhan, Yunqing Hu, Yiming Wu, Yangqing Jia, Peter Vajda, Matt Uytendaele, and Niraj K. Jha. ChamNet: Towards efficient network design through platform-aware model adaptation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11398-11407, 2019.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009.  
Misha Denil, Babak Shakibi, Laurent Dinh, Marc'Aurelio Ranzato, and Nando de Freitas. Predicting parameters in deep learning. In Advances in Neural Information Processing Systems (NeurIPS), pp. 2148-2156. 2013.  
Emily L Denton, Wojciech Zaremba, Joan Bruna, Yann LeCun, and Rob Fergus. Exploiting linear structure within convolutional networks for efficient evaluation. In Advances in Neural Information Processing Systems (NeurIPS), pp. 1269-1277. 2014.  
Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In International Conference on Artificial Intelligence and Statistics (AISTATS), pp. 315-323, 2011.  
Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural network with , trained quantization and huffman coding. In International Conference on Learning Representations (ICLR), 2016.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In IEEE International Conference on Computer Vision (ICCV), pp. 1026-1034, 2015.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.  
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, and Hartwig Adam. Searching for MobileNetV3. IEEE International Conference on Computer Vision (ICCV), 2019.  
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

Andrey Ignatov, Radu Timofte, William Chou, Ke Wang, Max Wu, Tim Hartley, and Luc Van Gool. AI benchmark: Running deep neural networks on android smartphones. arXiv preprint arXiv:1810.01109, 2018.  
Yani Ioannou, Duncan Robertson, Jamie Shotton, Roberto Cipolla, and Antonio Criminisi. Training CNNs with low-rank filters for efficient image classification. In International Conference on Learning Representations (ICLR), 2016.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning (ICML), pp. 448-456, 2015.  
Yong-Deok Kim, Eunhyeok Park, Sungjoo Yoo, Taelim Choi, Lu Yang, and Dongjun Shin. Compression of deep convolutional neural networks for fast and low power mobile applications. In International Conference on Learning Representations (ICLR), 2016.  
Markus Kliegl, Siddharth Goyal, Kexin Zhao, Kavya Srinet, and Mohammad Shoeybi. Trace norm regularization and faster inference for embedded speech recognition RNNs. arXiv preprint arXiv:1710.09026, 2017.  
A. Krizhevsky. Learning multiple layers of features from tiny images. Master's thesis, Department of Computer Science, University of Toronto, 2009.  
Vadim Lebedev, Yaroslav Ganin, Maksim Rakhuba, Ivan Oseledets, and Victor Lempitsky. Speeding-up convolutional neural networks using fine-tuned cp-decomposition. In International Conference on Learning Representations (ICLR), 2015.  
Yann LeCun, Joshua Bengio, and Geoffrey E. Hinton. Deep learning. Nature, 521:436-444, 2015.  
Chong Li and C. J. Richard Shi. Constrained optimization based low-rank approximation of deep neural networks. In European Conference on Computer Vision (ECCV), pp. 746-761, 2018.  
Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning efficient convolutional networks through network slimming. In IEEE International Conference on Computer Vision (ICCV), pp. 2755-2763, 2017.  
Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in PyTorch. In NIPS Autodiff Workshop, 2017.  
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. MobileNetV2: Inverted residuals and linear bottlenecks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510-4520, 2018.  
Dimitrios Stamoulis, Ruizhou Ding, Di Wang, Dimitrios Lymberopoulos, Bodhi Priyantha, Jie Liu, and Diana Marculescu. Single-Path NAS: Designing hardware-efficient convnets in less than 4 hours. arXiv preprint arXiv:1904.02877, 2019.  
Taiji Suzuki. Compression based bound for non-compressed network: unified generalization error analysis of large compressible deep neural network. arXiv preprint, 2019.  
Cheng Tai, Tong Xiao, Yi Zhang, XiaogangWang, and Weinan E. Convolutional neural networks with lowrank regularization. In International Conference on Learning Representations (ICLR), 2016.  
Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V. Le. MnasNet: Platform-aware neural architecture search for mobile. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.  
Wei Wen, Cong Xu, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Coordinating filters for faster deep neural networks. In IEEE International Conference on Computer Vision (ICCV), pp. 658-666, 2017.

Tien-Ju Yang, Andrew Howard, Bo Chen, Xiao Zhang, Alec Go, Mark Sandler, Vivienne Sze, and Hartwig Adam. NetAdapt: Platform-aware neural network adaptation for mobile applications. In European Conference on Computer Vision (ECCV), pp. 289-304, 2018.  
Jiahui Yu and Thomas Huang. Universally slimmable networks and improved training techniques. IEEE International Conference on Computer Vision (ICCV), 2019.  
Jiahui Yu, Linjie Yang, Ning Xu, Jianchao Yang, and Thomas Huang. Slimmable neural networks. In International Conference on Learning Representations (ICLR), 2019.  
Xiyu Yu, Tongliang Liu, Xinchao Wang, and Dacheng Tao. On compressing deep models by low rank and sparse decomposition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 67-76, 2017.  
Sergey Zagoruyko.  $92.45\%$  on cifar-10 in torch, 2015. URL http://torch.ch/blog/2015/07/30/cifar.html.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In the British Machine Vision Conference (BMVC), 2016.  
Linfeng Zhang, Zhanhong Tan, Jiebo Song, Jingwei Chen, Chenglong Bao, and Kaisheng Ma. SCAN: A scalable neural networks framework towards compact and efficient models. Advances in Neural Information Processing Systems (NeurIPS), 2019.  
Xiangyu Zhang, Jianhua Zou, Kaiming He, and Jian Sun. Accelerating very deep convolutional networks for classification and detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(10):1943-1955, 2016.  
Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. In International Conference on Learning Representations (ICLR), 2017.
