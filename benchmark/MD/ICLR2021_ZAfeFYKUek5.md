# OPTIMIZATION VARIANCE: EXPLORING GENERALIZATION PROPERTIES OF DNNS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Unlike the conventional wisdom in statistical learning theory, the test error of a deep neural network (DNN) often demonstrates double descent: as the model complexity increases, it first follows a classical U-shaped curve and then shows a second descent. Through bias-variance decomposition, recent studies revealed that the bell-shaped variance is the major cause of model-wise double descent (when the DNN is widened gradually). This paper investigates epoch-wise double descent, i.e., the test error of a DNN also shows double descent as the number of training epochs increases. Specifically, we extend the bias-variance analysis to epoch-wise double descent, and reveal that the variance also contributes the most to the zero-one loss, as in model-wise double descent. Inspired by this result, we propose a novel metric, optimization variance (OV), to measure the diversity of model updates caused by the stochastic gradients of random training batches drawn in the same iteration. OV can be estimated using samples from the training set only but correlates well with the (unknown) test error. It can be used to predict the generalization ability of a DNN when the zero-one loss is used in test, and hence early stopping may be achieved without using a validation set.

# 1 INTRODUCTION

Deep Neural Networks (DNNs) usually have large model capacity, but also generalize well. This violates the conventional VC dimension (Vapnik, 1999) or Rademacher complexity theory (Shalev-Shwartz & Ben-David, 2014), inspiring new designs of network architectures (Krizhevsky et al., 2012; Simonyan & Zisserman, 2015; He et al., 2016; Zagoruyko & Komodakis, 2016) and reconsideration of their optimization and generalization (Zhang et al., 2017; Arpit et al., 2017; Wang et al., 2018; Kalimeris et al., 2019; Rahaman et al., 2019; Allen-Zhu et al., 2019).

Model-wise double descent, i.e., as a DNN's model complexity increases, its test error first shows a classical U-shaped curve and then enters a second descent, has been observed on many machine learning models (Advani & Saxe, 2017; Belkin et al., 2019a; Geiger et al., 2019; Maddox et al., 2020; Nakkiran et al., 2020). Multiple studies provided theoretical evidence of this phenomenon in some tractable settings (Mitra, 2019; Hastie et al., 2019; Belkin et al., 2019b; Yang et al., 2020; Bartlett et al., 2020; Muthukumar et al., 2020). Specifically, Neal et al. (2018) and Yang et al. (2020) performed bias-variance decomposition for mean squared error (MSE) and the cross-entropy (CE) loss, and empirically revealed that the bell-shaped curve of the variance is the major cause of modelwise double descent. Maddox et al. (2020) proposed to measure the effective dimensionality of the parameter space, which can be further used to explain model-wise double descent.

Recently, a new double descent phenomenon, epoch-wise double descent, was observed, when increasing the number of training epochs instead of the model complexity $^{1}$  (Nakkiran et al., 2020). Compared with model-wise double descent, epoch-wise double descent is relatively less explored. Zhang & Wu (2020) discovered that the energy ratio of the high-frequency components of a DNN's prediction landscape, which can reflect the model capacity, switches from increase to decrease at a certain training epoch, leading to the second descent of the test error. However, this metric fails to provide further information on generalization, such as the early stopping point, or how the size of a DNN influences its performance.

This paper utilizes bias-variance decomposition of the zero-one (ZO) loss (CE loss is still used in training) to further investigate epoch-wise double descent. By monitoring the behaviors of the bias and the variance, we find that the variance plays an important role in epoch-wise double descent, which dominates and highly correlates with the variation of the test error.

Though the variance correlates well with the test error, estimating its value requires training models on multiple different training sets drawn from the same data distribution, whereas in practice usually only one training set is available<sup>2</sup>. Inspired by the fact that the source of variance comes from the random-sampled training sets, we propose a novel metric, optimization variance (OV), to measure the diversity of model updates caused by the stochastic gradients of random training batches drawn in the same iteration. This metric can be estimated from a single model using samples drawn from the training set only. More importantly, it correlates well with the test error, and thus can be used to determine the early stopping point in DNN training, without using any validation set.

Some complexity measures have been proposed to illustrate the generalization ability of DNNs, such as sharpness (Keskar et al., 2017) and norm-based measures (Neyshabur et al., 2015). However, their values rely heavily on the model parameters, making comparisons across different models very difficult. Dinh et al. (2017) shows that by re-parameterizing a DNN, one can alter the sharpness of its searched local minima without affecting the function it represents; Neyshabur et al. (2018) shows that these measures cannot explain the generalization behaviors when the size of a DNN increases. Our proposed metric, which only requires the logit outputs of a DNN, is less dependent on model parameters, and hence can explain many generalization behaviors, e.g., the test error decreases as the network size increases.

To summarize, our contributions are:

- We perform bias-variance decomposition on the test error to explore epoch-wise double descent. We show that the variance dominates the variation of the test classification error.  
- We propose a novel metric, OV, which is calculated from the training set only and correlates well with the test classification error.  
- Based on the OV, we propose an approach to search for the early stopping point without using a validation set, when the zero-one loss is used in test. Experiments verified its effectiveness.

The remainder of this paper is organized as follows: Section 2 introduces the details of tracing bias and variance over training epochs. Section 3 proposes the OV and demonstrates its ability to indicate the test behaviors. Section 4 draws conclusions and points out some future research directions.

# 2 BIAS AND VARIANCE IN EPOCH-WISE DOUBLE DESCENT

This section presents the details of tracing the bias and the variance during training. We show that the variance dominates the epoch-wise double descent of the test error.

# 2.1 A UNIFIED BIAS-VARIANCE DECOMPOSITION

Bias-variance decomposition is widely used to analyze the generalization properties of machine learning algorithms (Geman et al., 1992; Friedman et al., 2001). It was originally proposed for the MSE loss and later extended to other loss functions, e.g., CE and ZO losses (Kong & Dietterich, 1995; Tibshirani, 1996; Kohavi et al., 1996; Heskes, 1998). Our study utilizes a unified bias-variance decomposition that was proposed by Domingos (2000) and applicable to arbitrary loss functions.

Let  $(\pmb{x},\pmb{t})$  be a sample drawn from the data distribution  $\mathcal{D}$ , where  $\pmb{x} \in \mathbb{R}^d$  denotes the  $d$ -dimensional input, and  $\pmb{t} \in \mathbb{R}^c$  the one-hot encoding of the label in  $c$  classes. The training set  $\mathcal{T} = \{(\pmb{x}_i,\pmb{t}_i)\}_{i=1}^n \sim \mathcal{D}^n$  is utilized to train the model  $f: \mathbb{R}^d \to \mathbb{R}^c$ . Let  $\pmb{y} = f(\pmb{x};\mathcal{T}) \in \mathbb{R}^c$  be

the probability output of the model  $f$  trained on  $\mathcal{T}$ , and  $\mathcal{L}(t, y)$  the loss function. The expected loss  $\mathbb{E}_{\mathcal{T}}[\mathcal{L}(t, y)]$  should be small to ensure good generalization performance.

According to Domingos (2000), a unified bias-variance decomposition<sup>3</sup> of  $\mathbb{E}_{\mathcal{T}}[\mathcal{L}(\boldsymbol{y},t)]$  is:

$$
\mathbb {E} _ {\mathcal {T}} [ \mathcal {L} (\boldsymbol {t}, \boldsymbol {y}) ] = \underbrace {\mathcal {L} (\boldsymbol {t} , \bar {\boldsymbol {y}})} _ {\text {B i a s}} + \beta \underbrace {\mathbb {E} _ {\mathcal {T}} [ \mathcal {L} (\bar {\boldsymbol {y}} , \boldsymbol {y}) ]} _ {\text {V a r i a n c e}}, \tag {1}
$$

where  $\beta$  takes different values for different loss functions, and  $\bar{y}$  is the expected output:

$$
\bar {\boldsymbol {y}} = \quad \arg \min  \quad \mathbb {E} _ {\mathcal {T}} [ \mathcal {L} (\boldsymbol {y} ^ {*}, \boldsymbol {y}) ]. \tag {2}
$$

$$
\boldsymbol {y} ^ {*} \in \mathbb {R} ^ {c} \left| \sum_ {k = 1} ^ {c} \boldsymbol {y} _ {k} ^ {*} = 1, \boldsymbol {y} _ {k} ^ {*} \geq 0 \right.
$$

$\overline{y}$  minimizes the variance term in (1), which can be regarded as the "center" or "ensemble" of  $\pmb{y}$  w.r.t. different  $\mathcal{T}$ .

Table 1 shows specific forms of  $\mathcal{L}$ ,  $\bar{\pmb{y}}$ , and  $\beta$  for different loss functions (the detailed derivations can be found in Appendix A). This paper focuses on the bias-variance decomposition of the ZO loss, because epoch-wise double descent of the test error is more obvious when the ZO loss is used (see Appendix C). To capture the overall bias and variance, we choose to analyze  $\mathbb{E}_{x,t} \mathbb{E}_{\mathcal{T}}[\mathcal{L}(t,y)]$ , i.e., the expectation of  $\mathbb{E}_{\mathcal{T}}[\mathcal{L}(t,y)]$  over the distribution  $\mathcal{D}$ .

Table 1: Bias-variance decomposition for different loss functions. The CE loss herein is the complete form of the commonly used one, originated from the Kullback-Leibler divergence.  $Z = \sum_{k=1}^{c} \exp \{ \mathbb{E}_{\mathcal{T}}[\log y_k] \}$  is a normalization constant independent of  $k$ .  $H(\cdot)$  is the hard-max which sets the maximal element to 1 and others to 0.  $\mathbf{1}_{\mathrm{con}}\{\cdot\}$  is an indicator function which equals 1 if its argument is true, and 0 otherwise. log and exp are element-wise operators.

<table><tr><td>Loss</td><td>L(t,y)</td><td>y̅</td><td>β</td></tr><tr><td>MSE</td><td>||t - y||2</td><td>ETy</td><td>1</td></tr><tr><td>CE</td><td>∑k=1ctk log tk/yk</td><td>1/Z exp{ET[log y]}</td><td>1</td></tr><tr><td>ZO</td><td>1con{H(t) ≠ H(y)}</td><td>H(ET[H(y)])</td><td>1 if y̅ = t, otherwise -PT(H(y) = t|y̅ ≠ H(y))</td></tr></table>

# 2.2 TRACE THE BIAS AND VARIANCE TERMS OVER TRAINING EPOCHS

To trace the bias term  $\mathbb{E}_{\boldsymbol{x},t}[\mathcal{L}(\boldsymbol{t},\bar{\boldsymbol{y}})]$  and the variance term  $\mathbb{E}_{\boldsymbol{x},t}\mathbb{E}_{\mathcal{T}}[\mathcal{L}(\bar{\boldsymbol{y}},\boldsymbol{y})]$  w.r.t. the training epoch, we need to sample several training sets and train models on them respectively, so that the bias and variance terms can be estimated from them.

Concretely, let  $\mathcal{T}^*$  denote the test set,  $f(x; \mathcal{T}_j, q)$  the model  $f$  trained on  $\mathcal{T}_j \sim \mathcal{D}^n$  ( $j = 1, 2, \dots, K$ ) for  $q$  epochs. Then, the estimated bias and variance terms at the  $q$ -th epoch, denoted as  $B(q)$  and  $V(q)$ , respectively, can be written as:

$$
B (q) = \mathbb {E} _ {(\boldsymbol {x}, t) \in \mathcal {T} ^ {*}} \left[ \mathcal {L} \left(\boldsymbol {t}, \bar {f} (\boldsymbol {x}; q)\right) \right], \tag {3}
$$

$$
V (q) = \mathbb {E} _ {(\boldsymbol {x}, t) \in \mathcal {T} ^ {*}} \left[ \frac {1}{K} \sum_ {j = 1} ^ {K} \mathcal {L} (\bar {f} (\boldsymbol {x}; q), f (\boldsymbol {x}; \mathcal {T} _ {j}, q)) \right], \tag {4}
$$

where

$$
\bar {f} (\boldsymbol {x}; q) = \mathrm {H} \left(\sum_ {j = 1} ^ {K} \mathrm {H} (f (\boldsymbol {x}; \mathcal {T} _ {j}, q))\right), \tag {5}
$$

is the voting result of  $\{f(\pmb {x};\mathcal{T}_j,q)\}_{j = 1}^K$

We should emphasize that, in real-world situations,  $\mathcal{D}$  cannot be obtained, hence  $\mathcal{T}_j$  in our experiments was randomly sampled from the training set (we sampled  $50\%$  training data for each  $\mathcal{T}_j$ ). As a result, despite of showing the cause of epoch-wise double descent, the behaviors of bias and variance may be different when the whole training set is used.

We considered ResNet (He et al., 2016) and VGG (Simonyan & Zisserman, 2015) models trained on SVHN (Netzer et al., 2011), CIFAR10 (Krizhevsky, 2009), and CIFAR100 (Krizhevsky, 2009). SGD and Adam (Kingma & Ba, 2014) optimizers with different learning rates were used. The batchsize was set to 128, and all models were trained for 250 epochs with data augmentation. Prior to sampling  $\{\mathcal{T}_j\}_{j=1}^K$  ( $K = 5$ ) from the training set,  $20\%$  labels of the training data were randomly shuffled to strengthen epoch-wise double descent.

Figure 1 shows the expected ZO loss and its bias and variance. The bias descends rapidly at first and then generally converges to a low value, whereas the variance behaves almost exactly the same as the test error, mimicking even small fluctuations of the test error. To stabilize that, we performed additional experiments with different optimizers, learning rates, and levels of label noise (see Appendices E and G). All experimental results demonstrated that it is mainly the variance that contributes to epoch-wise double descent.

![](images/36ba3d6628d853d49b47610f9adec25b7aec4d2a369387cd51d96482c0821a40.jpg)

![](images/90f47f3d3a28a5f7d4c7f437a269696948d6840b521f2d4ed36eb3425cccf3b3.jpg)

![](images/446e8245c18a5765e32a78b316fcedce12310ec39569120585b2410a96644dac.jpg)

![](images/37cdd71259274fb02b481780d5b4dcb83dbd19c0cc47964b491ad49b3e36e654.jpg)  
(a) SVHN

![](images/d8473fc04f9afb776ce308f15deea505ce2832922c497f3e7e6bf7a91018b4ec.jpg)  
(b) CIFAR10

![](images/4abe4205b02c56b4afe0e825741f7e260053207ce5756f8c15aa4340b89e0bfd.jpg)  
Figure 1: The expected test ZO loss and its bias and variance. The models were trained with  $20\%$  label noise. Adam optimizer with learning rate 0.0001 was used.  
(c) CIFAR100

# 2.3 DISCUSSION

Contradicting to the traditional view that the variance keeps increasing because of overfitting, our experimental results show a more complex behavior: the variance starts high and then decreases rapidly, followed by a bell-shaped curve. The difference at the beginning (when the number of epochs is small) is mainly due to the choice of loss functions (see experimental results of bias-variance decomposition for MSE and CE losses in Appendix F). CE and MSE losses, analyzed in the traditional learning theory, can reflect the degree of difference of probabilities, whereas the ZO loss only the labels. At the early stage of training, the output probabilities are close to random guesses, and hence a small difference in probabilities may lead to completely different labels, resulting in the distinct variance for different loss functions. However, the reason why the variance begins to diminish at the late phase of training is still unclear. We will explore this problem in our future research.

# 3 OPTIMIZATION VARIANCE (OV)

This section proposes a new metric, OV, to measure the diversity of model updates introduced by random training batches during optimization. This metric can indicate test behaviors without any validation set.

# 3.1 NOTATION AND DEFINITION

Section 2 verified the synchronization between the test error and the variance, but its application is limited because estimating the variance requires: 1) a test set, and, 2) models trained on different training sets drawn from the same data distribution. It'd be desirable to capture the test behavior of a DNN using a single training set only, without a test set.

According to the definition in (1), the variance measures the model diversity caused by different training samples drawn from the same distribution, i.e., the outputs of DNN change according to the sampled training set. As the gradients are usually the only information transferred from training sets to models during the optimization of DNN, we should explore how gradients calculated from different training batches influence the DNN model.

Mathematically, for a sample  $(\pmb{x},t)\sim \mathcal{D}$ , let  $f(\pmb{x};\pmb{\theta})$  be the logit output of a DNN with parameter  $\pmb{\theta}$ . Let  $\mathcal{T}_B\sim \mathcal{D}^m$  be a training batch with  $m$  samples,  $g:\mathcal{T}_B\to \mathbb{R}^{|\pmb{\theta}|}$  the optimizer outputting the update of  $\pmb{\theta}$  based on  $\mathcal{T}_B$ . Then, we can get the function distribution  $F_{x}(\mathcal{T}_{B})$  over a training batch  $\mathcal{T}_B$ , i.e.,  $f(\pmb {x};\pmb {\theta} + g(\mathcal{T}_B))\sim F_{\pmb{x}}(\mathcal{T}_B)$ . The variance of  $F_{x}(\mathcal{T}_{B})$  reflects the model diversity caused by different training batches. The formal definition of OV is given below.

Definition 1 (Optimization Variance (OV)) Given an input  $\mathbf{x}$  and model parameters  $\theta_{q}$  at the  $q$ -th training epoch, the OV on  $\mathbf{x}$  at the  $q$ -th epoch is defined as

$$
O V _ {q} (\boldsymbol {x}) \triangleq \frac {\mathbb {E} _ {\mathcal {T} _ {B}} \left[ \| f (\boldsymbol {x} ; \boldsymbol {\theta} _ {q} + g (\mathcal {T} _ {B})) - \mathbb {E} _ {\mathcal {T} _ {B}} f (\boldsymbol {x} ; \boldsymbol {\theta} _ {q} + g (\mathcal {T} _ {B})) \| _ {2} ^ {2} \right]}{\mathbb {E} _ {\mathcal {T} _ {B}} \left[ \| f (\boldsymbol {x} ; \boldsymbol {\theta} _ {q} + g (\mathcal {T} _ {B})) \| _ {2} ^ {2} \right]}. \tag {6}
$$

Note that  $OV_{q}(\pmb{x})$  measures the relative variance, because the denominator in (6) eliminates the influence of the logit's norm. In this way,  $OV_{q}(\pmb{x})$  at different training phases can be compared.

Intuitively, the OV represents the inconsistency of gradients' influence on the model. If  $OV_{q}(\pmb{x})$  is very large, then the models trained with different sampled  $\mathcal{T}_B$  may have distinct outputs for the same input, leading to high model diversity.

Here we emphasize the inconsistency of model updates rather than the gradients themselves. The latter can be measured by the gradient variance. The gradient variance and the OV are different, because sometimes diverse gradients may lead to similar changes of the function represented by DNN, and hence small OV. More details on the relationship between the two variances can be found in Appendix B.

# 3.2 EXPERIMENTAL RESULTS

We calculated the expectation of the OV over  $\pmb{x}$ , i.e.,  $\mathbb{E}_{\pmb{x}}[OV_{q}(\pmb{x})]$ , which was estimated from 1,000 random training samples. The test set was not involved at all.

Figure 2 shows how the test accuracy (solid curves) and  $\mathbb{E}_{\pmb{x}}[OV_{q}(\pmb{x})]$  (dashed curves) change with the number of training epochs. Though sometimes the OV may not exhibit clear epoch-wise double descent, e.g., VGG16 in Figure 2(c), the symmetry between the solid and dashed curves generally exist, suggesting that the OV, which is calculated from the training set only, is capable of predicting the variation of the test accuracy. Similar results can also be observed using different optimizers and learning rates (see Appendix H).

$OV_{q}(\pmb{x})$  in Figure 2 was estimated on all training batches; however, this may not be necessary: a small number of training batches are usually enough. To demonstrate this, we trained ResNet and VGG on several datasets using Adam optimizer with learning rate 0.0001, and estimated  $OV_{q}(\pmb{x})$  from different numbers of training batches. The results in Figure 3 show that we can well estimate the OV using as few as 10 training batches.

![](images/19a9d41e7caa0d38aec909ac34464b52b187d07162ea824df6669087b2ecf4b2.jpg)

![](images/68251a655c125deca0eab5096e5fc670994310f2965c1e6197d7715c162d7d9a.jpg)

![](images/2073cabcec12eaf45442c8661ea6eb910c021658fc20d9dc60258f6f8e61635a.jpg)

![](images/8df962d78a0d8cf9203544ccf007c2d8dd645de42b9ee2248d6ba0142d748e5c.jpg)  
(a) SVHN

![](images/4d41e2270eb9a973f327495aef02c9fad04f9411549928cd2f884b7599b3556e.jpg)  
(b) CIFAR10

![](images/95c215dde6787555d1930f63e5746a37af7e9ea40000b26ca2e290faf66a1f75.jpg)  
(c) CIFAR100

![](images/51f00215013f7c52a88708ac4825f6ab579da9bedc88da92c3e5a5085a1b62a8.jpg)  
Figure 2: Test accuracy and OV. The models were trained with Adam optimizer (learning rate 0.0001). The number in each legend indicates its percentage of label noise.

![](images/825872ab01ee5a7781c9a70cfe03be167fd9173f8f56243beda30810de4ea89e.jpg)

![](images/c518c58a2688561f2de68c3044f54817cf2ea8a795fc11d7201db5eb8508659e.jpg)

![](images/46d1dd266eba02d0acd584f0527c051fe1f8c360376aa1bbd45cc28020c6e179.jpg)  
(a) SVHN

![](images/f73165816bee43f41d693baca44c0f99f78de922a863db9dd2a203a75514e508.jpg)  
(b) CIFAR10

![](images/540e26912d346ee0586222e4ab93d58fd7b77c54380e1acf44787ad388d72caf.jpg)  
Figure 3: OV estimated from different number of training batches. The models were trained with  $20\%$  label noise. Adam optimizer with learning rate 0.0001 was used.  
(c) CIFAR100

Another intriguing finding is that even unstable variations of the test accuracy can be reflected by the OV. This correspondence is clearer on simpler datasets, e.g., MNIST (LeCun et al., 1998) and FashionMNIST (Xiao et al., 2017). Figure 4 shows the test accuracy and OV for LeNet-5 (LeCun et al., 1998) trained on MNIST and FashionMNIST without label noise. Spikes of the OV and the test accuracy happen simultaneously at the same epoch.

Our experimental results demonstrate that the generalization ability of a DNN can be indicated by the OV during stochastic training, without using a validation set. This phenomenon can be used to determine the early stopping point, and beyond.

![](images/d53c773f4164326cd53cda4962cc7cb2985ccb7f9525103ab11eaebf16cae500.jpg)  
(a) MNIST

![](images/a53258aeab8b64f14218100868838bad5cfea3101a1036d35388dcb2309aefa7.jpg)  
Figure 4: Test accuracy and OV. The model was LeNet-5 trained on MNIST and FashionMNIST with Adam optimizer (learning rate 0.0001).  
(b) FashionMNIST

# 3.3 EARLY STOPPING WITHOUT A VALIDATION SET

The common process to train a DNN involves three steps: 1) partition the dataset into a training set and a validation set; 2) use the training set to optimize the DNN parameters, and the validation set to determine when to stop training, i.e., early stopping, and record the early stopping point; 3) train the DNN on the entire dataset (combination of training and validation sets) for the same number of epochs. However, there is no guarantee that the early stopping point on the training set is the same as the one on the entire dataset. So, an interesting questions is: is it possible to directly perform early stopping on the entire dataset, without a validation set?

The OV can be used for this purpose. For more robust performance, instead of using the OV directly, we may need to smooth it to alleviate random fluctuations.

As an example, we smoothed the OV by a moving average filter of 10 epochs, and then performed early stopping on the smoothed OV with a patience of 10 epochs. As a reference, early stopping with the same patience was also performed directly on the test accuracy to get the groundtruth. However, it should be noted that the latter is unknown in real-world applications. It is provided for verification purpose only.

We trained different DNN models on several datasets (SVHN: VGG11 and ResNet18; CIFAR10: VGG13 and ResNet18; CIFAR100: VGG16 and ResNet34) with different levels of label noise (10% and 20%) and optimizers (Adam with learning rate 0.001 and 0.0001, SGD with momentum 0.9 and learning rate 0.01 and 0.001). Then, we compared the groundtruth early stopping point and the test accuracy with those found by performing early stopping on the OV<sup>5</sup>. The results are shown in Figure 5. The true early stopping points and those found from the OV curve were generally close, though there were some exceptions, e.g., the point near (40, 100) in Figure 5(a). However, the test errors, which are what a model designer really cares about, were always close.

# 3.4 NETWORK SIZE

In addition to indicating the early stopping point, the OV can also explain some other generalization behaviors, such as the influence of the network size. To verify that, we trained ResNet18 with different network sizes on CIFAR10 for 100 epochs with no label noise, using Adam optimizer with learning rate 0.0001. For each convolutional layer, we set the number of filters  $k / 4$  ( $k = 1,2,\dots,8$ ) times the number of filters in the original model. We then examined the OV of ResNet18 with different network sizes to validate its correlation with the test accuracy. Note that we used SGD optimizer with learning rate 0.001 and no momentum to calculate the OV, so that the cumulative influence during training can be removed to make the comparison fairer.

The results are shown in Figure 6. As  $k$  increases, the OV gradually decreases, i.e., the diversity of model updates introduced by different training batches decreases when widening ResNet18,

![](images/83f7656acc3eb8d68bf80a3631278c9cae631f3c27c93d1c3a7d1e05f8c7130d.jpg)  
(a) Early stopping point

![](images/33e64d1ba19618fcfc35d1237abb4416442c38ec8a51e34f324f1c1ca9e02641.jpg)  
(b) Test error

suggesting that increasing the network size can improve the model's resilience to sampling noise, which leads to better generalization performance. The Pearson correlation coefficient between the OV and the test accuracy reached  $-0.94$  ( $p = 0.0006$ ).

Lastly, we need to point out that we did not observe a strong correlation between the OV and the test accuracy when using significantly different model architectures, e.g., VGG and ResNet. Our future research will look for a more universal metric of the generalization ability.

![](images/c86744fc28734b81c7a3b3c4f03c0117920bea2e33b59d16803805ecc5c2856c.jpg)  
Figure 5: Early stopping based on the test error (True) and the OV (Found).  
Figure 6: Test accuracy and OV w.r.t. the network size.

# 4 CONCLUSIONS

This paper has shown that the variance dominates the epoch-wise double descent, and highly correlates with the test error. Inspired by this finding, we propose a novel metric called optimization variance, which is calculated from the training set only but powerful enough to predict how the test error changes during training. Based on this metric, we further propose an approach to perform early stopping without any validation set. Remarkably, we demonstrate that the training set itself may be enough to predict the generalization ability of a DNN, without a dedicated validation set.

Our future work will: 1) apply the OV to other tasks, such as regression problems, unsupervised learning, and so on; 2) figure out the cause of the second descent of the OV; and, 3) design regularization approaches to penalize the OV for better generalization performance.

# REFERENCES

Madhu S Advani and Andrew M Saxe. High-dimensional dynamics of generalization error in neural networks. CoRR, abs/1710.03667, 2017.  
Zeyuan Allen-Zhu, Yanzhi Li, and Zhao Song. A convergence theory for deep learning via overparameterization. In Proc. 36th Int'l Conf. on Machine Learning, pp. 242-252, Long Beach, CA, May 2019.  
Devansh Arpit, Stanisław Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, and Simon Lacoste-Julien. A closer look at memorization in deep networks. In Proc. 34th Int'l Conf. on Machine Learning, volume 70, pp. 233-242, Sydney, Australia, August 2017.  
Peter L Bartlett, Philip M Long, Gábor Lugosi, and Alexander Tsigler. Benign overfitting in linear regression. Proceedings of the National Academy of Sciences, 2020. In press.

Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machine learning practice and the classical bias-variance trade-off. Proceedings of the National Academy of Sciences, 116(32):15849-15854, 2019a.  
Mikhail Belkin, Daniel Hsu, and Ji Xu. Two models of double descent for weak features. CoRR, abs/1903.07571, 2019b.  
Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, 2004.  
Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep nets. In Proc. 34th Int'l Conf. on Machine Learning, volume 70, pp. 1019-1028, Sydney, Australia, August 2017.  
Pedro Domingos. A unified bias-variance decomposition for zero-one and squared loss. In Proc. of the 17th National Conf. on Artificial Intelligence, pp. 564-569, Austin, TX, July 2000.  
Jerome Friedman, Trevor Hastie, and Robert Tibshirani. The Elements of Statistical Learning, volume 1. Springer series in statistics New York, second edition, 2001.  
Mario Geiger, Arthur Jacot, Stefano Spigler, Franck Gabriel, Levent Sagun, Stéphane d'Ascoli, Giulio Biroli, Clément Hongler, and Matthieu Wyart. Scaling description of generalization with number of parameters in deep learning. CoRR, abs/1901.01608, 2019.  
Stuart Geman, Elie Bienenstock, and René Doursat. Neural networks and the bias/variance dilemma. Neural Computation, 4(1):1-58, 1992.  
Trevor Hastie, Andrea Montanari, Saharon Rosset, and Ryan J Tibshirani. Surprises in high-dimensional ridgeless least squares interpolation. CoRR, abs/1903.08560, 2019.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition, pp. 770-778, Las Vegas, NV, June 2016.  
Tom Heskes. Bias/variance decompositions for likelihood-based estimators. Neural Computation, 10(6):1425-1433, 1998.  
Dimitris Kalimeris, Gal Kaplun, Preetum Nakkiran, Benjamin Edelman, Tristan Yang, Boaz Barak, and Haofeng Zhang. SGD on neural networks learns functions of increasing complexity. In Proc. Advances in Neural Information Processing Systems, pp. 3491-3501, Vancouver, Canada, December 2019.  
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. In Proc. Int'l Conf. on Learning Representations, Toulon, France, April 2017.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proc. Int'l Conf. on Learning Representations, Banff, Canada, April 2014.  
Ron Kohavi, David H Wolpert, et al. Bias plus variance decomposition for zero-one loss functions. In Proc. 13th Int'l Conf. on Machine Learning, volume 96, pp. 275-283, Bari, Italy, July 1996.  
Eun Bae Kong and Thomas G Dietterich. Error-correcting output coding corrects bias and variance. In Proc. 12th Int'l Conf. on Machine Learning, pp. 313-321, Tahoe City, CA, July 1995.  
Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009. URL https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. ImageNet classification with deep convolutional neural networks. In Proc. Advances in Neural Information Processing Systems, pp. 1097-1105, Lake Tahoe, NE, December 2012.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.

Wesley J Maddox, Gregory Benton, and Andrew Gordon Wilson. Rethinking parameter counting in deep models: Effective dimensionality revisited. CoRR, abs/2003.02139, 2020.  
Partha P Mitra. Understanding overfitting peaks in generalization error: Analytical risk curves for  $l_{2}$  and  $l_{1}$  penalized interpolation. CoRR, abs/1906.03667, 2019.  
Vidya Muthukumar, Kailas Vodrahalli, Vignesh Subramanian, and Anant Sahai. Harmless interpolation of noisy data in regression. IEEE Journal on Selected Areas in Information Theory, 1(1): 67-83, 2020.  
Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever. Deep double descent: Where bigger models and more data hurt. In Proc. Int'l Conf. on Learning Representations, Addis Ababa, Ethiopia, April 2020.  
Brady Neal, Sarthak Mittal, Aristide Baratin, Vinayak Tantia, Matthew Scicluna, Simon Lacoste-Julien, and Ioannis Mitliagkas. A modern take on the bias-variance tradeoff in neural networks. CoRR, abs/1810.08591, 2018.  
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. Reading digits in natural images with unsupervised feature learning. In Proc. NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011, Granada, Spain, December 2011.  
Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. Norm-based capacity control in neural networks. In Proc. of the 28th Conf. on Learning Theory, pp. 1376-1401, Paris, France, July 2015.  
Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nati Srebro. Exploring generalization in deep learning. In Proc. Advances in Neural Information Processing Systems, pp. 5947-5956, Long Beach, CA, January 2018.  
Nasim Rahman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In Proc. 36th Int'l Conf. on Machine Learning, pp. 5301-5310, Long Beach, CA, May 2019.  
Shai Shalev-Shwartz and Shai Ben-David. Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press, 2014.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Proc. Int'l Conf. on Learning Representations, San Diego, CA, May 2015.  
Robert Tibshirani. Bias, variance and prediction error for classification rules. Technical report, Department of Preventive Medicine and Biostatistics and Department of Statistics, University of Toronto, Toronto, Canada, 1996.  
Vladimir N Vapnik. An overview of statistical learning theory. IEEE Trans. on Neural Networks, 10(5):988-999, 1999.  
Huan Wang, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. Identifying generalization properties in neural networks. CoRR, abs/1809.07402, 2018.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. CoRR, abs/1708.07747, 2017.  
Zitong Yang, Yaodong Yu, Chong You, Jacob Steinhardt, and Yi Ma. Rethinking bias-variance trade-off for generalization of neural networks. CoRR, abs/2002.11328, 2020.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. CoRR, abs/1605.07146, 2016.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In Proc. Int'l Conf. on Learning Representations, Toulouse, France, April 2017.  
Xiao Zhang and Dongrui Wu. Rethink the connections among generalization, memorization and the spectral bias of DNNs. CoRR, abs/2004.13954, 2020.
