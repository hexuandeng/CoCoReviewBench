# NORMALIZATION GRADIENTS ARE LEAST-SQUARES RESIDUALS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Batch Normalization (BN) and its variants have seen widespread adoption in the deep learning community because they improve the training of deep neural networks. Discussions of why this normalization works so well remain unsettled. We make explicit the relationship between ordinary least squares and partial derivatives computed when back-propagating through BN. We recast the backpropagation of BN as a least squares fit, which zero-centers and decorrelates partial derivatives from normalized activations. This view, which we term gradient-least-squares, is an extensible and arithmetically accurate description of BN. Our view offers a unified interpretation of BN and related work; we motivate, from a regression perspective, two improvements upon BN, and evaluate on CIFAR-10.

# 1 INTRODUCTION

Training deep neural networks has become central to many machine learning tasks in computer vision, speech, and many other application areas. Ioffe & Szegedy (2015) showed empirically that Batch Normalization (BN) enables deep networks to attain faster convergence and lower loss. Reasons for the effectiveness of BN remain an open question (Lipton & Steinhardt, 2018). Existing work towards explaining this have focused on covariate shift; Santurkar et al. (2018) described how BN makes the loss function smoother. In our work, the back-propagation of BN is recast into a least squares fit. This gradient regression zero-centers and decorrelates partial derivatives from the normalized activations; it passes on a scaled residual during back-propagation. Our view provides novel insight into the effectiveness of BN, and unifies several existing alternative normalization approaches in the literature. This enables us to leverage principles from least squares to design better normalizations.

# 1.1 CONTRIBUTIONS

Foremost, we draw an unexpected connection between least squares and the gradient computation of BN. This motivates a novel view that complements earlier investigations into why it is so effective. Other popular normalization strategies can be unified under this view. Our view is consistent with recent empirical surprises regarding ordering of layers within ResNet residual maps (He et al., 2016b) and within shake-shake regularization branches (Huang & Narayanan, 2018). Finally, to demonstrate the extensibility of our view, we motivate and evaluate two variants of BN from the perspective of gradient-least-squares. In the first variant, a least squares explanation motivates the serial chaining of BN and Layer Normalization (LN) (Ba et al., 2016). In the second variant, regularization of the least-squares leads to a version of BN that performs better on batch size two. In both variants, we provide empirical support on CIFAR-10.

In summary, our work presents a view, which we term gradient-least-squares, through which the back-propagation of BN and related work in a neural network can be recast as least squares regression. This regression decomposes gradients into an explained portion and a residual portion; BN back-propagation will be shown to remove the explained portion. Hopefully, gradient-least-squares will be broadly useful in the future design and understanding of neural network components. Figure 1 reviews normalization with batch statistics, and illustrates our main theorem.

![](images/64ede05b2bc6983702488ac6f3f84258e9b76410cf451b5004eb97c3e94d9fe2.jpg)  
Figure 1: The left figure reviews, for a single channel at a particular layer within a single batch, notable quantities computed during the forward pass and during back-propagation of BN. Let  $\{x_{i}\}_{i = 1\dots N}$  be activations. Let  $\mu = \sum_{i = 1}^{N}\frac{x_i}{N}$  and  $\sigma^2 = \sum_{i = 1}^{N}\frac{(x_i - \mu)^2}{N}$ . Let  $L$  be a function dependent on the normalized activations  $z_{i}$  defined for each  $j$  by  $z_{j} = \frac{(x_{j} - \mu)}{\sigma}$ . This, along with partial derivatives, are shown in the left figure. Our work establishes a novel identity on the quantities shown in hexagons. The right figure illustrates our main result in a scatter plot, in which each pair  $\left(z_{i},\frac{\partial L}{\partial z_{i}}\right)$  is shown as a data point in the regression.

![](images/a79161936e38edefb8341e637d4ef110b56960576e0b4936b4cd7ebdc172ee5b.jpg)

# 2 NORMALIZATION GRADIENTS ARE LEAST-SQUARES RESIDUALS

Consider any particular channel within which  $\{x_{i}\}$  are activations to be normalized in BN moment calculations. Ioffe & Szegedy (2015) defined BN as

$$
B N (\boldsymbol {x}) = \frac {(\boldsymbol {x} - \mu)}{\sigma} \cdot c + b \tag {1}
$$

where  $\sigma, \mu$  are batch moments, but  $b$  and  $c$  are learned per-channel parameters persistent across batches. In BN, the batch dimension and spatial dimensions are marginalized out in the computation of batch moments. For clarity, we consider a simplified version of BN. We ignore the variables  $b$  and  $c$  in equation 1 responsible for a downstream channel-wise affine transformation. We examine back-propagation of partial derivatives through this normalization, where  $\mu$  and  $\sigma$  are viewed as functions of  $\boldsymbol{x}$ . We also ignore a numerical stability hyperparameter  $\epsilon$ . We write the normalized output as

$$
z = \frac {(\boldsymbol {x} - \mu)}{\sigma} \tag {2}
$$

Note that  $\mu$  and  $\sigma$  are functions of each  $x_{i}$

We review ordinary least squares of a single variable with intercept (Friedman et al., 2001).

Let  $g_{j} = \alpha + \beta z_{j} + \epsilon_{j}$  where  $\alpha$  and  $\beta$  are parameters,  $\mathbf{z}$  and  $\mathbf{g}$  are observations.  $z_{j}$  and  $g_{j}$  are entries in  $\mathbf{z}$  and  $\mathbf{g}$  respectively.  $\epsilon_{j}$  are i.i.d. Gaussian residuals. We wish to fit  $\alpha$  and  $\beta$

$$
\hat {\alpha}, \hat {\beta} = \underset {\alpha , \beta} {\arg \min } \mathbb {E} _ {j} \left(\left\| \boldsymbol {g} - \alpha - \beta \boldsymbol {z} \right\| ^ {2}\right) \tag {3}
$$

The least-squares problem in equation 3 is satisfied by  $\hat{\beta} = \frac{\operatorname{Cov}(\boldsymbol{z},\boldsymbol{g})}{\operatorname{Var}(\boldsymbol{z})}$  and  $\hat{\alpha} = \mathbb{E}(\pmb {g}) - \hat{\beta}\mathbb{E}(\pmb {z})$

When  $z$  are normalized activations and  $g$  are partial derivatives, then  $\mathbb{E}z = 0$  and  $\operatorname{Var}(z) = 1$ . In this special case, the solution simplifies into

$$
\hat {\beta} = \operatorname {C o v} (\boldsymbol {z}, \boldsymbol {g}) \tag {4}
$$

$$
\hat {\alpha} = \mathbb {E} (\boldsymbol {g}) \tag {5}
$$

Theorem 1 (Normalization gradients are least-squares residuals). Let  $i \in \{1 \dots N\}$  be indices over some set of activations  $\{x_i\}$ . Then the moment statistics are defined by  $\mu = \sum_{i=1}^{N} \frac{x_i}{N}$  and  $\sigma^2 = \sum_{i=1}^{N} \frac{(x_i - \mu)^2}{N}$ . Let  $L$  be a function dependent on the normalized activations  $z_i$  defined for each  $j$  by  $z_j = \frac{(x_j - \mu)}{\sigma}$ . Then, the gradients of  $L$  satisfy, for all  $j \in \{1, \dots, N\}$ , the following:

$$
\sigma \frac {\partial L}{\partial x _ {j}} = \frac {\partial L}{\partial z _ {j}} - \widehat {\frac {\partial L}{\partial z _ {j}}} \tag {6}
$$

where

$$
\frac {\widehat {\partial L}}{\partial z _ {j}} = \hat {\alpha} + \hat {\beta} z _ {j} \tag {7}
$$

$$
\hat {\alpha}, \hat {\beta} = \arg \min  _ {\alpha , \beta} \sum_ {i = 1} ^ {N} \left(\frac {\partial L}{\partial z _ {i}} - \alpha - \beta z _ {i}\right) ^ {2} \tag {8}
$$

Proof: Normalization gradients are least-squares residuals. The proof involves a derivation of partial derivatives by repeated applications of the chain rule and rules of total derivative. Because  $\{z_i\}$  normalized over  $i$  has mean 0 and variance 1, the partial derivatives can be rearranged to satisfy the single variable ordinary least squares framework.

Fix  $j$ . We expand  $\frac{\partial L}{\partial x_j}$  as a linear combination of  $\left\{\frac{\partial L}{\partial z_i}\right\}_{i=1\dots N}$

$$
\frac {\partial L}{\partial x _ {j}} = \frac {\partial L}{\partial z _ {j}} \frac {\partial z _ {j}}{\partial x _ {j}} + \sum_ {i \neq j} ^ {N} \frac {\partial L}{\partial z _ {i}} \frac {\partial z _ {i}}{\partial x _ {j}} \tag {9}
$$

We state  $\frac{\partial z_i}{\partial x_j}$  directly. Steps are in Appendix A under Lemma 1.

$$
\frac {\partial z _ {i}}{\partial x _ {j}} = \left\{ \begin{array}{l} \frac {- 1 - z _ {j} z _ {i}}{\sigma N} \text {i f} i \neq j \\ \frac {N - 1 - z _ {j} ^ {2}}{\sigma N} \text {i f} i = j \end{array} \right. \tag {10}
$$

Through substitution of equations 10 into 9, we get

$$
\frac {\partial L}{\partial x _ {j}} = \frac {\partial L}{\partial z _ {j}} \frac {N - 1 - z _ {j} ^ {2}}{\sigma N} + \sum_ {i \neq j} ^ {N} \left[ \frac {\partial L}{\partial z _ {i}} \cdot \frac {- 1 - z _ {j} z _ {i}}{\sigma N} \right] \tag {11}
$$

$$
\sigma \frac {\partial L}{\partial x _ {j}} = \frac {\partial L}{\partial z _ {j}} + \frac {1}{N} \sum_ {i = 1} ^ {N} \left[ (- 1 - z _ {i} z _ {j}) \frac {\partial L}{\partial z _ {i}} \right] \tag {12}
$$

$$
\sigma \frac {\partial L}{\partial x _ {j}} = \frac {\partial L}{\partial z _ {j}} - \left(\frac {1}{N} \sum_ {i = 1} ^ {N} \frac {\partial L}{\partial z _ {i}}\right) - \frac {z _ {j}}{N} \sum_ {i = 1} ^ {N} z _ {i} \frac {\partial L}{\partial z _ {i}} \tag {13}
$$

Noting that  $\{z_i\}$  normalized over  $i$  has mean 0 and variance 1, we recover  $\hat{\beta}$  and  $\hat{\alpha}$ , in the sense of equations 4 and 5, from equation 13.

$$
\frac {1}{N} \sum_ {i = 1} ^ {N} z _ {i} \frac {\partial L}{\partial z _ {i}} = \operatorname {C o v} _ {i} \left(z _ {i}, \frac {\partial L}{\partial z _ {i}}\right) = \hat {\beta} \tag {14}
$$

$$
\frac {1}{N} \sum_ {i = 1} ^ {N} \frac {\partial L}{\partial z _ {i}} = \mathbb {E} _ {i} \left[ \frac {\partial L}{\partial z _ {i}} \right] - \hat {\beta} \cdot 0 = \hat {\alpha} \tag {15}
$$

Finally, we rearrange equations 15 and 14 into 13 to conclude, as desired,

$$
\sigma \frac {\partial L}{\partial x _ {j}} = \frac {\partial L}{\partial z _ {j}} - \hat {\alpha} - \hat {\beta} z _ {j} = \frac {\partial L}{\partial z _ {j}} - \widehat {\frac {\partial L}{\partial z _ {j}}} \tag {16}
$$

![](images/1432d319b87b839ba83a205d0009af47aa5c2e57bca956d45b8fd05886aff5da.jpg)

During back-propagation of a single batch, the normalization function takes in partial derivatives  $\frac{\partial L}{\partial z_{(\cdot)}}$ , and removes that which can be explained by least squares of  $\frac{\partial L}{\partial z_{(\cdot)}}$  against  $z_{(\cdot)}$ . As illustrated in Figure 1, during back-propagation, the residual then divides away  $\sigma$  to become  $\frac{\partial L}{\partial x_{(\cdot)}}$ , the gradient for the unnormalized activations.

# 3 RELATED DEEP LEARNING COMPONENTS VIEWED AS GRADIENT CALCULATIONS

BN aims to control its output to have mean near 0 and variance near 1, normalized over the dataset; this is related to the original explanation termed internal covariate shift (Ioffe & Szegedy, 2015). Most existing work that improve or repurpose BN have focused on describing the distribution of activations.

Definition 1. In the context of normalization layers inside a neural network, activations are split into partitions, within which means and variances are computed. We refer to these partitions as normalization partitions.

Definition 2. Within the context of a normalization partition, we refer to the moments calculated on the partitions as partition statistics.

Theorem 1 shows that BN has least squares fitting built into the gradient computation. Gradients of the activations being normalized in each batch moment calculation are fit with a single-variable with-Intercept least squares model, and only a rescaled residual is kept during back-propagation. We emphasize that the data on which the regression is trained and applied is a subset of empirical activations within a batch, corresponding to the normalization partitions of BN.

To show extensibility, we recast several popular normalization techniques into the gradient-least-squares view. We refer to activations arising from a single member of a particular batch as an item.  $BHWC$  refers to dimensions corresponding to items, height, width, and channels respectively. In non-image applications or fully connected layers,  $H$  and  $W$  are 1. BN marginalizes out the items and spatial dimensions, but statistics for each channel are kept separate.

In the subsequent sections, we unify several normalization methods from the perspective of the gradient. Figure 2 reviews the normalization partitions of these methods, and places our main theorem about gradient-least-squares into context.

# 3.1 LAYER NORMALIZATION,INSTANCE NORMALIZATION,GROUP NORMALIZATION

Ba et al. (2016) introduced Layer Normalization (LN) in the context of large LSTM models and recurrent networks. Only the  $(H,W,C)$  dimensions are marginalized in LN, whereas BN marginalizes out the  $(B,H,W)$  dimensions. In our regression framework, the distinction can be understood as changing the data point partitions in which least squares are fit during back-propagation. LN marginalizes out the channels, but computes separate statistics for each batch item. To summarize, the regression setup in the back-propagation of LN is performed against other channels, rather than against other batch items.

Huang & Belongie (2017) introduced Instance Normalization (IN) in the context of transferring styles across images. IN is is closely related to contrast normalization, an older technique used in image processing. IN emphasizes end-to-end training with derivatives passing through the moments. Only the  $(H,W)$  dimensions are marginalized in IN, whereas BN marginalizes  $(B,H,W)$  dimensions. In our framework, this can be understood as using fewer data points and a finer binning to fit

![](images/d1fa053ae3c26ab4d3492ce2ea296d5efc49f4e589d3840348b116cd252a9c69.jpg)  
Figure 2: We review the normalization partitions of BN, LN, GN, and IN. Each normalization partition contains a separate set of data points on which the gradient regression is performed. One partition for each method is illustrated in blue. This figure also shows the correspondence between a single activation and a gradient regression data point for BN.

the least squares during back-propagation, as each batch item now falls into its own normalization partition.

Wu & He (2018) introduced Group Normalization (GN) to improve performance on image-related tasks when memory constrains the batch size. Similar to LN, GN also marginalizes out the  $(H,W,C)$  dimensions in the moment computations. The partitions of GN are finer: the channels are grouped into disjoint sub-partitions, and the moments are computed for each sub-partition. When the number of groups is one, GN reduces to LN.

In future normalization methods that involve normalizing with respect to different normalization partitions; such methods can pattern match with BN, LN, IN, or GN; the backpropagation can be formulated as a least-squares fit, in which the partial derivatives at normalized activations  $\frac{\partial L}{\partial z_{(\cdot)}}$  are fitted against the normalized  $z_{(\cdot)}$ , and then the residual of the fit is rescaled to become  $\frac{\partial L}{\partial x_{(\cdot)}}$ .

Figure 2 summarize the normalization partitions for BN, LN, IN, and GN; the figure visualizes, as an example, a one-to-one correspondence between an activation in BN, and a data point in the gradient regression.

Theorem 1 is agnostic to the precise nature of how activations are partitioned before being normalized; thus, equation 9 applies directly to any method that partitions activations and performs Gaussian normalization on each partition. The partitioning of BN, LN, IN, and GN are performed in different respective manners, and each partition is individually subject to Gaussian normalization. Thus, the gradients of BN, LN, IN, and GN are residuals of regressions in the sense of Theorem 1.

# 3.2 WEIGHT NORMALIZATION

Salimans & Kingma (2016) introduced Weight Normalization (WN) in LSTMs, and noted improvements in the condition number of deep networks; WN divides each weight tensor by their respective vector 2-norms. In the view of gradient-least-squares, WN has a single-variable intercept-0 regression interpretation in back-propagation, analogous to BN. A raw weight vector  $v$ , is normalized and

scaled before being used as coefficient weights  $\pmb{w} = \frac{c}{\|\pmb{v}\|}\pmb{v}$ , where  $c$  is a learned downstream linear scaling parameter.

In this regression setup, the length normalized weights of WN are analogous to the Gaussian normalized activations in BN. We write that  $z = \frac{\boldsymbol{v}}{\|\boldsymbol{v}\|} = \frac{\boldsymbol{w}}{c}$ , and state directly an analogous relationship between each  $\| \boldsymbol{v} \| \frac{\partial L}{\partial v_j}$  and the regression on  $\left\{\left(z_i, \frac{\partial L}{\partial z_i}\right)\right\}_{i=1\dots N}$ . See Appendix B Lemma 2 for steps that derive the following identity: for loss  $L$  and for each component  $j$ , we have

$$
\left\| \boldsymbol {v} \right\| \frac {\partial L}{\partial v _ {j}} = \frac {\partial L}{\partial z _ {j}} - \hat {\beta} z _ {j} \tag {17}
$$

where

$$
\hat {\beta} = \underset {\beta} {\arg \min } \| \nabla_ {z} L - \beta z \| ^ {2} = (\nabla_ {z} L) ^ {T} z \tag {18}
$$

The L2 normalization of weights in WN appears distinct from the Gaussian normalization of activations in BN; nevertheless, WN can also be recast as a least squares regression.

# 3.3 IDENTITY MAPPINGS IN RESNET, AND SHAKE-SHAKE RESNEXT REGULARIZATION

![](images/564392ad00e729edf8f817ada742b008cd1f351de260968ca739a227fb48b0f5.jpg)  
Figure 3: This figure illustrates the original (He et al., 2016a) and improved (He et al., 2016b) residual mappings in ResNets. Arrows point in the direction of the forward pass. Dotted lines indicate that gradients are zero-centered and decorrelated with respect to downstream activations in the residual mapping. The improved ordering has BN coming first, and thus constrains that gradients of the residual map must be decorrelated with respect to some normalized activations inside the residual mapping.

An update to the popular ResNet architecture showed that the network's residual mappings can be dramatically improved with a new ordering (He et al., 2016b). The improvement moved BN operations into early positions and surprised the authors; we support the change from the perspective of gradient-least-squares. Figure 3 reviews the precise ordering in the two versions. Huang & Narayanan (2018) provides independent empirical support for the BN-early order, in shake-shake regularization (Gastaldi, 2017) architectures. We believe that the surprise arises from a perspective that views BN only as a way to control the distribution of activations; one would place BN after a sequence of convolution layers. In the gradient-least-squares perspective, the first layer of each residual mapping is also the final calculation for these gradients before they are added back into the main trunk. The improved residual branch constrains the gradients returning from the residual mappings to be zero-centered and decorrelated with respect to some activations inside the branch. We illustrate this idea in Figure 3.

# 4 NORMALIZATION APPROACHES MOTIVATED BY LEAST SQUARES

Gradient-least-squares views back-propagation in deep neural networks as a solution to a regression problem. Thus, formulations and ideas from a regression perspective would motivate improvements and alternatives to BN. We pursue and evaluate two of these ideas.

Table 1: BN plus LN final validation performance (ResNet-34-v2, batch size 128)  

<table><tr><td>Normalization</td><td>CIFAR-10 Accuracy</td><td>CIFAR-10 Cross Entropy</td></tr><tr><td>BN, LN</td><td>0.9259</td><td>0.3087</td></tr><tr><td>LN, BN</td><td>0.9245</td><td>0.3389</td></tr><tr><td>BN (Ioffe &amp; Szegedy, 2015)</td><td>0.9209</td><td>0.3969</td></tr><tr><td>LN (Ba et al., 2016)</td><td>0.9102</td><td>0.3548</td></tr></table>

# 4.1 BN AND LN AS TWO-STEP GRADIENT REGRESSION

BN and LN are similar to each other, but they normalize over different partitioning of the activations; in back-propagation, the regressions occur respectively with respect to different partitions of the activations. Suppose that a BN and a LN layer are chained serially in either order. This results in a two-step regression during back-propagation; in reversed order, the residual from the first regression is further explained by a second regression on a different partitioning. In principle, whether this helps would depend on the empirical characteristics of the gradients encountered during training. The second regression could further decorrelate partial gradients from activations. Empirically, we show improvement in a reference ResNet-34-v2 implementation on CIFAR-10 relative to BN with batch size 128. In all cases, only a single per-channel downstream affine transformation is applied, after both normalization layers, for consistency in the number of parameters. See table 1 for CIFAR-10 validation performances. We kept all default hyperparameters from the reference implementation: learning schedules, batch sizes, and optimizer settings.

# 4.2 ADDRESSING SMALL BATCHES WITH LEAST-SQUARES REGULARIZATION

BN performs less well on small batches (Ioffe, 2017). Gradient-least-squares interprets this as gradient regressions failing on correlated data, an issue typically addressed by regularization. We pursue this idea to recover some performance on small batches by use of regularization. Our regularization uses streaming estimates of past gradients to create virtual data in the regression. This performed better than standard BN on the same batch size, but we did not recover the performance of large batches; this is consistent with the idea that regularization could not in general compensate for having much less data. See Appendix C for CIFAR-10 validation performances.

# 5 LIMITATIONS AND RELATED WORK

# 5.1 SWITCH NORMALIZATION

Luo et al. (2018a) introduced Switch Normalization (SwN), a hybrid strategy for combining moment calculations from LN, BN, and IN. SwN uses learnable scalar logits  $\lambda_{k}$  for  $k\in \Omega = \{BN,IN,LN\}$  with corresponding softmax weighting activations  $w_{k} = \frac{\exp(\lambda_{k})}{\sum_{k^{\prime}}\exp(\lambda_{k^{\prime}})}$  to rescale the contributions to the batch mean for each normalization scheme. It uses an analogous set of parameters  $\lambda_k^\prime$  and activations  $w_{k}^{\prime}$  for variances. We sketch the back-propagation of a simplified version of SN in the perspective of gradient-least-squares. We ignore both the division  $\epsilon$  and downstream affine  $z\rightarrow c\cdot z + b$ . The normalization calculation inside SwN can be written as:

$$
z _ {b h w c} = \frac {x _ {b h w c} - \sum_ {k \in \Omega} w _ {k} \mu_ {b h w c , k}}{\sqrt {\sum_ {k \in \Omega} w _ {k} ^ {\prime} \sigma_ {b h w c , k} ^ {2}}} \tag {19}
$$

where  $\Omega = \{BN, LN, IN\}$ . There is potentially a unique mean and variance used for each activation. Equation 19 bears similarities to the setup in Theorem 1, but we leave unresolved whether there is a gradient-least-squares regression interpretation for SN.

# 5.2 DECORRELATED BATCH NORMALIZATION AND SPECTRAL NORMALIZATION

Decorrelated Batch Normalization (DBN) (Huang et al., 2018) is a generalization of BN that performs Mahalanobis ZCA whitening to decorrelate the channels, using differentiable operations. On some leve, the matrix gradient equation resemble the least squares formulation in Theorem 1.

Spectral Normalization (SpN) (Miyato et al., 2018) is an approximate spectral generalization of WN. For DBN and SpN, the regression interpretations remain unresolved.

# 5.3 RELATED WORK

BN has been instrumental in the training of deeper networks (Ioffe & Szegedy, 2015). Subsequent work resulted in Batch Renormalization (Ioffe, 2017), and further emphasized the importance of passing gradients through the minibatch moments, instead of a gradient-free exponential running average. In gradient-least-squares, use of running accumulators in the training forward pass would stop the gradients from flowing through them during training, and there would be no least-squares. He et al. (2016b) demonstrate empirically the unexpected advantages of placing BN early in residual mappings of ResNet.

Santurkar et al. (2018) showed that BN makes the loss landscape smoother, and gradients more predictable across stochastic gradient descent steps. Balduzzi et al. (2017) found evidence that spatial correlation of gradients explains why ResNet outperforms earlier designs of deep neural networks. Kohler et al. (2018) proved that BN accelerates convergence on least squares loss, but did not consider back-propagation of BN as a least squares residual. Luo et al. (2018b) has recast BN as a stochastic process, resulting in a novel treatment of regularization.

# 6 DISCUSSION, AND FUTURE WORK

This work makes explicit how BN back-propagation regresses partial derivatives against the normalized activations and keeps the residual. This view, in conjunction with the empirical success of BN, suggests an interpretation of BN as a gradient regression calculation. BN and its variants decorrelate and zero-center the gradients with respect to the normalized activations. Subjectively, this can be viewed as removing systematic errors from the gradients. Our view offers a unified interpretation of normalization variants already in literature. Our view also supports empirical results in literature preferring early BN placement within neural network branches.

Leveraging gradient-least-squares considerations, we ran two sets of normalization experiments, applicable to large batch and small batch settings. Placing a LN layer either before or after BN can be viewed as two-step regression that better explains the residual. We show empirically that BN and LN together are better than either individually. In a second set of experiments, we address BN's performance degradation with small batch size. We regularize the gradient regression with streaming gradient statistics, which empirically recovers some performance on CIFAR-10 relative to basic BN, on batch size two.

Why do empirical improvements in neural networks with BN keep the gradient-least-squares residuals and drop the explained portion? We propose two open approaches for investigating this in future work. A first approach focuses on how changes to the gradient regression result in different formulations; the two empirical experiments in our work contribute to this. A second approach examines the empirical relationships between gradients of activations evaluated on the same parameter values; we can search for a shared noisy component arising from gradients in the same normalization partition. Suppose that the gradient noise correlates with the activations – this is plausible because the population of internal activations arise from using shared weights – then normalizations could be viewed as a layer that removes systematic noise during back-propagation.

In conclusion, we have presented a novel view that reorganizes the back-propagation of BN as a least squares residual calculation. This view generates novel descriptions of normalization techniques related to BN, and comments on the ordering of layers inside the residual mappings of ResNet. This view is extensible and will motivate novel designs of neural network components in future work.

# REFERENCES

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.  
David Balduzzi, Marcus Frean, Lennox Leary, J. P. Lewis, Kurt Wan-Duo Ma, and Brian McWilliams. The shattered gradients problem: If resnets are the answer, then what is the question? In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, pp. 342-350, 2017.  
Jerome Friedman, Trevor Hastie, and Robert Tibshirani. The elements of statistical learning, volume 1. Springer series in statistics New York, NY, USA.: 2001.  
Xavier Gastaldi. Shake-shake regularization. arXiv preprint arXiv:1705.07485, 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016a.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In European conference on computer vision, pp. 630-645. Springer, 2016b.  
Che-Wei Huang and Shrikanth S Narayanan. Normalization before shaking toward learning symmetrically distributed representation without margin in speech emotion recognition. arXiv preprint arXiv:1808.00876, 2018.  
Lei Huang, Dawei Yang, Bo Lang, and Jia Deng. Decorated batch normalization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 791-800, 2018.  
Xun Huang and Serge J Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV, pp. 1510-1519, 2017.  
Sergey Ioffe. Batch renormalization: Towards reducing minibatch dependence in batch-normalized models. In Advances in Neural Information Processing Systems, pp. 1945-1953, 2017.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Jonas Kohler, Hadi Daneshmand, Aurelien Lucchi, Ming Zhou, Klaus Neymeyr, and Thomas Hofmann. Towards a theoretical understanding of batch normalization. arXiv preprint arXiv:1805.10694, 2018.  
Zachary C Lipton and Jacob Steinhardt. Troubling trends in machine learning scholarship. arXiv preprint arXiv:1807.03341, 2018.  
Ping Luo, Jiamin Ren, and Zhanglin Peng. Differentiable learning-to-normalize via switchable normalization. arXiv preprint arXiv:1806.10779, 2018a.  
Ping Luo, Xinjiang Wang, Wenqi Shao, and Zhanglin Peng. Understanding regularization in batch normalization. arXiv preprint arXiv:1809.00846, 2018b.  
Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. arXiv preprint arXiv:1802.05957, 2018.  
Tim Salimans and Diederik P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in Neural Information Processing Systems, pp. 901-909, 2016.  
Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How does batch normalization help optimization? (no, it is not about internal covariate shift). arXiv preprint arXiv:1805.11604, 2018.  
Yuxin Wu and Kaiming He. Group normalization. In The European Conference on Computer Vision (ECCV), September 2018.
