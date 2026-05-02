# THE NEEDLE IN THE HAYSTACK: OUT-DISTRIBUTION AWARE SELF-TRAINING IN AN OPEN-WORLD SETTING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Traditional semi-supervised learning (SSL) has focused on the closed world assumption where all unlabeled samples are task-related. In practice, this assumption is often violated when leveraging data from very large image databases that contain mostly non-task-relevant samples. While standard self-training and other established methods fail in this open-world setting, we demonstrate that our out-distribution-aware self-learning (ODST) with a careful sample selection strategy can leverage unlabeled datasets with millions of samples, more than 1600 times larger than the labeled datasets, and which contain only about  $2\%$  task-relevant inputs. Standard and open world SSL techniques degrade in performance when the ratio of task-relevant sample decreases and show a significant distribution shift which is problematic regarding AI safety while ODST outperforms them with respect to test performance, corruption robustness and out-of-distribution detection.

# 1 INTRODUCTION

In past years we have seen tremendous progress in image recognition based on deep learning (Krizhevsky et al., 2012; He et al., 2016). However, this success required large labeled datasets that are expensive to generate. On the other hand, large amounts of unlabeled data are broadly available, in particular in image recognition. The promise of semi-supervised learning (Chapelle et al., 2006) is to leverage unlabeled data in order to improve prediction performance. However, the underlying assumption of traditional and modern (Berthelot et al., 2019; Sohn et al., 2020) SSL algorithms is that the unlabeled data comes from the same distribution or at least contains data from the same classes (closed world assumption). Oliver et al. (2018) criticized this as being unrealistic as the assumption is hard to check when retrieving large amounts of unlabeled data from the web. SSL in an open world setting, where the unlabeled data contains task-relevant but also non-related out-of-distribution (OOD) images, has recently attracted attention as a more realistic approach to SSL (Guo et al., 2020; Chen et al., 2020; Yu et al., 2020). While these methods could outperform standard SSL techniques when the unlabeled data contains out-of-distribution samples, they have mostly been evaluated in settings where the unlabeled data contains relatively few non-task-related samples. It is thus an open question whether existing open world SSL methods can scale to large unlabeled data bases where the ratio of task-relevant to non-related samples is much smaller.

In this paper, we demonstrate that existing methods suffer from severe performance degradations when the ratio of non-related images in the unlabeled dataset increases. Moreover, even if they achieve high accuracy, they are not OOD aware, that is they systematically assign high confidence to non-related inputs, e.g. a CIFAR10 model classifies images containing humans as dogs or horses. Our contributions are: i) we propose our scalable, iterative out-distribution-aware self-training (ODST) which enforces low-confidence predictions on non-task-related samples and selects unlabeled samples for the in-distribution task via a novel class-adaptive selection scheme. This confidence based selection scheme allows us to deal with strongly unbalanced in-distribution classes in the unlabeled dataset, ii) we outperform state-of-the-art SSL techniques (Berthelot et al., 2019; Sohn et al., 2020), as well as recent open world SSL techniques (Guo et al., 2020; Yu et al., 2020) in terms of prediction performance and out-of-distribution detection and show that all other approaches suffer from distribution shifts, iii) we show strong performance gains when using ODST with the full training sets of CIFAR10 and CIFAR100 as labeled data together with the 80 Million Tiny Image (80MTI) dataset as unlabeled dataset. Thus we show that even with large amounts of labeled data, ODST can still leverage unlabeled data to improve prediction and OOD performance.

![](images/33f0f67d7442b2108cae08fb3a365fff4d9e30202affbdd69c38ceb913fd4164.jpg)

![](images/bb65229aacc687f2c569e076b45046f9a9d5a32f75307d85b6c6e23ce5bb4fa8.jpg)  
(a) ODST

![](images/1110ac4dd43b45bbeb69f9a290573f6f8ac6a94cb924eb20a9cadbc9133e91f0.jpg)  
Figure 1: Random unlabeled samples considered to be task-relevant by the SSL method (confidence above in-distribution thresholds for ODST, above  $95\%$  confidence for Fixmatch and above the Otsu threshold for MTCF) are shown together with their confidence and predicted label (mistakes are marked red). MTCF and Fixmatch show severe distribution shift and only our ODST is able to select the correct samples. All methods are trained on CIFAR10 with 4k labeled images and an unlabeled set consisting of 41k CIFAR10 training images and 1M images from 80MTI.

![](images/c0c133fc019df884a19d947dfcd1cbd9725f7a99ba6f05f3962572c9a6d91de8.jpg)

![](images/ac3942493a7401dfe289ded9f69cbc6c48d88f13e679532e521810d7b030fbeb.jpg)

![](images/3dcf2fca7ad2a40ac704f21e4d4c7a43e6e8cbf41d52897162dc33d6e57c8f70.jpg)  
(b) MTCF

![](images/cbf2559a3985ce885066ff4ad7d8521f7dc37a1bb8163088f414556f6eab2a12.jpg)

![](images/352d9ac5814b6675eb2aca6c3d4b9081780a57dceaa90854361537896a4399d3.jpg)

![](images/6648ee8dd6692e4e1ad705cc3f559de151f69543c2456ff24458f656dec614c7.jpg)

![](images/2d3af2f3735092768b400d38397ea9f40a262f6409b33938e2ddc4edcfca84d8.jpg)

![](images/fa3f82f416d7dabd524bb83a9aaf52a55676c74b6f1947b2d372077798370b5a.jpg)  
(c) FixMatch

![](images/87402171700c906f43630f8b53e8db7c4f5bf12bdc5edec640965c0069e497e6.jpg)

![](images/330b5ba780c1ddc418cda5577845cd9f8a9bbddd2f8a41aa7d97330a28a99e92.jpg)

![](images/8d1c4a1070f9140eed7cdf7dcd2785ce119bc46d12007430ccf3c9cc0b059a5f.jpg)

![](images/b647a95a84f1aa2720b19dd4d2ff64a4a3ce01b97ba106aa1abca17bdac6298f.jpg)

# 2 RELATED WORK

Semi-supervised learning (SSL) is an established technique (Zhu, 2005; Zhu & Goldberg, 2009) for leveraging information from unlabeled data to improve predictive performance. In self-training (Riloff, 1996; Riloff & Wiebe, 2003; Scudder, 1965), a teacher model is trained in a fully-supervised fashion on a labeled dataset. The teacher model is then used to label a set of unlabeled examples, typically drawn from the original data distribution (closed world assumption), which is then used in combination with the labeled samples to train a new student model. Various extensions of this protocol have been proposed, including the use of an ensemble of teacher models (Zhou et al., 2018) and Co-training (Blum & Mitchell, 1998).

Recently, Xie et al. (2020) and Yalniz et al. (2019) used self-training to improve performance on ImageNet (Russakovsky et al., 2015) by using large image databases consisting of millions of task-relevant and out-of-distribution samples. On CIFAR10, Carmon et al. (2019) were able to significantly improve model robustness to adversarial perturbations by adding unlabeled samples from 80 million tiny images (80MTI) (Torralba et al., 2008) using self-training. The distinctive feature of self-training in comparison to other SSL methods is that the training of the teacher model is separated from the labeling process. In contrast, in pseudo-labeling (Lee, 2013; Iscen et al., 2019; Shi et al., 2018) labels are generated during training by the model itself. Similarly, consistency-based SSL-methods like II-models (Laine & Aila, 2016; Sajjadi et al., 2016), mean-teacher (Tarvainen & Valpola, 2017) and virtual adversarial training (Miyato et al., 2018) enforce an invariance of the model's output on the unlabeled data under a specific set of perturbations. Methods like MixMatch (Berthelot et al., 2019) and FixMatch (Sohn et al., 2020) combine consistency regularization with strong augmentation e.g. RandAugment (Cubuk et al., 2020). A related technique is entropy-minimization (Grandvalet & Bengio, 2005), which penalizes low-confidence predictions on unlabeled samples during training. Oliver et al. (2018) found that SSL can improve the model's performance in the traditional SSL setting where the unlabeled data is sampled from the same distribution as the training data (closed world assumption) but can degrade the performance when the unlabeled data contains non-task-related samples (open world setting).

The open world SSL setting has been explored only recently, by combining elements of consistency regularization with online OOD detection and sample filtering (Yu et al., 2020; Chen et al., 2020) or soft per-sample weighting (Guo et al., 2020). While they demonstrate performance improvements when the unlabeled data contains non-task-related samples, their evaluation is restricted to settings where the unlabeled data contains mostly task-related samples. We later demonstrate that in more challenging settings, where the ratio of task-related samples in the unlabeled data is small, these methods show severe performance degradations.

Out-of-distribution detection: Deep Neural networks (DNN) have empirically and theoretically been shown to produce overconfident predictions for inputs not related to the task, e.g. noise or other classes not contained in the labeled dataset (Nguyen et al., 2015; Hendrycks & Gimpel, 2017; Hein et al., 2019), i.e. the confidence of a DNN is not reliable for the detection of out-of-distribution (OOD) samples. Approaches for OOD detection include ODIN (Liang et al., 2018) or using the Mahalanobis distance of higher-order features (Lee et al., 2018). Hendrycks et al. (2019) introduced Outlier exposure (OE), see Hein et al. (2019) for the related CEDA, and show that the confidence

can be used as a reliable OOD detector when enforcing low confidence on an OOD training set, even when tested on other OOD test datasets. OOD detection is also related to open set recognition (Boult et al., 2019) which is beyond the scope of this paper.

# 3 METHOD

In contrast to previous works which focused on problems where the ratio of non-task-related to task-related images is small, our goal is to show that our ODST in combination with a careful sample selection strategy yields a self-training scheme that can leverage large unlabeled datasets to improve performance on the CIFAR10 and CIFAR100 test set over a fully-supervised baseline trained on the entire training set and additionally has excellent OOD detection performance.

First, we introduce self-training as in the noisy student self-training (NSST) of Xie et al. (2020) which serves a baseline. Then we introduce our out-distribution aware self-training ODST.

In the following  $\mathbf{T} = (x_i, y_i)_{i=1}^n$  denotes the set of labeled examples, where  $x_i \in \mathbb{R}^d$  and  $y_i \in \{1, \dots, K\}$  and  $\mathbf{U} = (z_i)_{i=1}^m$  is a collection of unlabeled samples. The traditional SSL literature makes the assumption that the unlabeled samples  $\mathbf{U}$  are drawn from the same distribution as the labeled examples  $\mathbf{T}$ , or at least belong to the same set of classes which we explicitly do not do in this paper. Given the logits of a neural network  $f: \mathbb{R}^d \to \mathbb{R}$  the predicted probability distribution for a point  $x$  is computed via the softmax as:  $\hat{p}_f(s|x) = e^{f_s(x)} / \sum_{l=1}^{K} e^{f_l(x)}$ . The confidence in the decision for  $x$  is then given by  $\max_{s=1, \dots, K} \hat{p}_f(s|x)$ , and the cross-entropy loss between (soft)-labels  $p \in \mathbb{R}^K$  ( $\sum_{i} p_i = 1$ ,  $p_i \geq 0$ ) and prediction  $\hat{p}$  is defined as:  $L(p, \hat{p}) = -\sum_{i=1}^{K} p_i \log \hat{p}_i$ .

# 3.1 NOISY STUDENT SELF-TRAINING (NSST)

In NSST (Xie et al., 2020) the first teacher is a base model  $f^{(0)}$  obtained by minimizing the cross-entropy loss on the labeled set only. The iterative scheme of NSST starting at  $t = 0$  is:

1) pseudo-label all unlabeled samples in  $\mathbf{U}$  with current teacher  $f^{(t)}$  
2) select a subset  $\mathbf{I} \subset \mathbf{U}$  of the pseudo-labeled examples by choosing per class the top- $k$  with highest confidence and which are above a fixed threshold  
3) train new model  $f^{(t + 1)}$  by minimizing the loss on labeled and pseudo-labeled samples in I with AutoAugment (Cubuk et al., 2019) as data augmentation:

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} L \left(y _ {i}, \hat {p} _ {f ^ {(t + 1)}} (x _ {i})\right) + \frac {1}{| \mathbf {I} |} \sum_ {z \in \mathbf {I}} L \left(\hat {p} _ {f ^ {(t)}} (z), \hat {p} _ {f ^ {(t + 1)}} (z)\right)
$$

4)  $t \gets t + 1$  and go back to step 1

Xie et al. (2020) call their self-training noisy due to the very strong data augmentation which avoids overfitting via noise injection. The main difficulty in self-training in a closed world setting is the propagation of labeling mistakes which leads to a degradation of prediction performance which is taken care of by step 2). While Xie et al. (2020) chose a fixed threshold, we choose it according to the false positive rate on an in-distribution validation set. However, in an open-world setting, an equally severe problem is that a large fraction of the unlabeled samples is not task-relevant such that including them leads to a distribution shift which happens for NSST (see Figure 2) and other SSL methods (see Figure 1). However, more importantly, the classifier is highly confident on unrelated classes which is a problem for AI safety and this undesired behavior is even hard to notice as predictive performance might appear to improve when evaluated only on the test set.

# 3.2 OUT-DISTRIBUTION-aware SELF-TRAINING (ODST)

We first provide an overview over our algorithmic scheme and then explain the individual steps in more detail. The base ODST model is initialized with an out-distribution aware base teacher model  $f^{(0)}$ , trained by minimizing:

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} L \left(y _ {i}, \hat {p} _ {f ^ {(0)}} \left(x _ {i}\right)\right) + \frac {1}{| \mathbf {U} |} \sum_ {z \in \mathbf {U}} L \left(\frac {1}{K} \mathbf {1}, \hat {p} _ {f ^ {(0)}} (z)\right). \tag {1}
$$

We then iterate the following steps starting from  $t = 0$ :

A) calibrate  $f^{(t)}$  on the in-distribution validation set  
B) pseudo-label all unlabeled samples in  $\mathbf{U}$  with current teacher  $f^{(t)}$  
C) for each class  $c$ : select the top- $k$  unlabeled instances with highest confidence classified as  $c$  that lie above the threshold. The selected samples for all classes are denoted as  $\mathbf{I}$  
D) determine new pseudo-labels for the unlabeled instances. We use  $q(z) = \hat{p}_{f^{(t)}}(z)$  for  $z \in \mathbf{I}$  (selected samples in step C)) and

$$
v (z) = \frac {1}{2} \left(\frac {1}{K} + \hat {p} _ {f ^ {(t)}} (z)\right), \text {f o r} z \in \mathbf {U} \backslash \mathbf {I}. \tag {2}
$$

E) train a new model  $f^{(t + 1)}$  by minimizing the loss on labeled and pseudo-labeled samples with AutoAugment (Cubuk et al., 2019) as strong data augmentation:

$$
\begin{array}{l} \frac {1}{n + | \mathbf {I} |} \left[ \sum_ {i = 1} ^ {n} L \left(y _ {i}, \hat {p} _ {f ^ {(t + 1)}} (x _ {i})\right) + \sum_ {z \in \mathbf {I}} L \left(q (z), \hat {p} _ {f ^ {(t + 1)}} (z)\right) \right] \\ + \frac {1}{| \mathbf {U} \backslash \mathbf {I} |} \sum_ {z \in \mathbf {U} \backslash \mathbf {I}} L (v (z), \hat {p} _ {f ^ {(t + 1)}} (z)) \tag {3} \\ \end{array}
$$

F)  $t\gets t + 1$  and go to step A)

The Base classifier is essentially an Outlier Exposure (OE) model (Hendrycks et al., 2019) (see also (Hein et al., 2019; Papadopoulos et al., 2019) for related losses) where the unlabeled set  $\mathbf{U}$  is our training out-distribution on which we enforce uniform confidence. OE is known to be one of the best methods for out-of-distribution detection. As in our case, a crucial assumption is that the unlabeled samples are partially task-related, it might appear odd to enforce uniform confidence on all of  $\mathbf{U}$ . However, we show in Section 3.3 that this asymptotically only leads to a down-weighting of the confidence for task-related samples but preserves the Bayes optimal decision and, in particular, enforces close-to-uniform confidence for all unrelated samples.

A) Calibration: while normal neural networks are known to be overconfident on in-(Guo et al., 2017) and out-distribution (Nguyen et al., 2015; Hein et al., 2019), the models resulting from enforcing low confidence on unlabeled points (such as OE) tend to be underconfident on the in-distribution. As we use the predictions of the teacher  $f^{(t)}$  as new pseudo-labels for the unlabeled data, we calibrate  $f^{(t)}$  by minimizing the expected calibration error using temperature rescaling Guo et al. (2017). Thus the teacher model assigns the correct uncertainty score to its predictions on in-distribution samples which improves pseudo-label quality and and stabilizes the training procedure.

C) Sample Selection: The most important problem in self-training is to integrate the right samples into the pseudo-labeled set  $\mathbf{I}$ . While our out-distribution aware teacher is better at discriminating between the in- and out-distribution based on confidence, there are still many samples with highly confident predictions due to the sheer size of the unlabeled dataset ( $\geq 10^6$ ). Note that we select at most the top- $k$  samples (where  $k = 5N(t + 1) / K$ ), but this might still be too much if not sufficiently many task-related examples of a class exist in the unlabeled dataset. We thus need to determine class-specific confidence thresholds to limit the selection. Class-specific selection has been neglected in the literature but is particularly important in practice as the number of task-related examples in the unlabeled dataset typically varies widely between the different classes.

Using the in-distribution validation set, we define the in-distribution threshold for class  $c$  as the smallest predicted probability for class  $c$  such that the precision for all images which are above this threshold is greater than or equal to  $\alpha$  (binary classification problem: class  $c$  versus all other classes). An in-distribution precision threshold is especially important if the classification task contains similar classes, as learning from wrongly labeled in-distribution images (a dog that is classified as a cat in CIFAR10) is likely to hurt predictive performance on the in-distribution task even more than the inclusion of an unrelated out-distribution image.

If there exist less than  $k$  samples above the threshold for a particular class, we randomly repeat the accepted samples above the threshold to maintain a class-balanced training scheme. Note that it is

much easier and also more interpretable to fix a precision value rather than the choice of a confidence threshold (in particular if the model is not calibrated) as done in Xie et al. (2020).

D) Pseudo Labels: for the original labeled dataset we always use one-hot labels. For unlabeled data points that have been selected in  $\mathbf{I}$ , we determine pseudo-labels  $q$  according to the predicted probability distribution over the classes by the calibrated teacher model. Due to the calibration, this should reflect the "correct" uncertainty about these labels. For all remaining images in our unlabeled dataset  $\mathbf{U} \backslash \mathbf{I}$ , we use a weak form of knowledge distillation by defining pseudo-labels  $v$  as the average of the predicted probability distribution of the teacher model and the uniform distribution, given in (2). This has two reasons: i) a purely uniform distribution on  $\mathbf{U} \backslash \mathbf{I}$ , which in the first iterations might still contain a lot of task-relevant images, leads to a bias as it does not distinguish between task-relevant and irrelevant images, ii) only using soft-labels from the teacher model leads to overconfident predictions as one can observe in the non-out-distribution aware NSST method. Thus a trade-off between these opposing goals is their average which leads to heavy damping of the confidence (note that the pseudo-labels have a maximal confidence of  $\frac{1}{2} + \frac{1}{K}$  on  $\mathbf{U} \backslash \mathbf{I}$ ).

E) Training: For the final objective in (3), the selected pseudo-labeled samples in  $\mathbf{I}$  and the original samples in  $\mathbf{T}$  are assigned the same weight. This is quite aggressive as for iteration  $t$  we add up to  $5t$  as much pseudo-labeled data as labeled training data. However, this also enables larger performance gains given that the sample selection process is successful. Note that the losses on  $\mathbf{I} \cup \mathbf{T}$  and on  $\mathbf{U} \backslash \mathbf{I}$  have equal weight as the damping of confidences on  $\mathbf{U} \backslash \mathbf{I}$  is crucial for the sample selection process. As pseudo-labels are computed on non-augmented images and we use heavy augmentation during training, our scheme can be regarded as offline consistency learning, where the model is encouraged to replicate the teacher's output independent of the randomly selected augmentation. This allows us to limit the number of passes through the entire unlabeled pool to the number of self-training iterations, do model calibration and results in more stable targets. The entire scheme is repeated multiple times until the validation accuracy starts to degrade or a fixed maximum number of iterations is reached.

ODST+: In addition to ODST, we provide the variant ODST+ for scenarios where AI safety is critical. It differs from ODST only in step C) where we calculate the final threshold as maximum over ODST's in-distribution threshold and an additional out-distribution threshold. This class-specific out-distribution threshold controls the number of task-irrelevant samples that are falsely added into our pseudo-labeled sample pool I. This is done using an extra out-distribution validation set, i.e. a set of natural images that does not contain any class relevant images (we discuss this choice in Section 4). For each class  $c$ , we compute the  $\alpha$ -quantile of the predicted probabilities for class  $c$  on the out-distribution images which we define as the out-distribution threshold for class  $c$ .

# 3.3 BAYESIAN DECISION THEORY OF SELF-TRAINING

In this section, we analyze our iterations in the framework of Bayesian decision theory. We show that the base classifier that enforces uniform confidence on the unlabeled points still leads to optimal decisions on the in-distribution. Moreover, we show that the iterative scheme with soft-labels ultimately reaches the optimal classifier which is Bayes optimal on the in-distribution task and maximally uncertain elsewhere. Proofs can be found in the Appendix A.

In this section we assume that the labeled examples  $(x_{i},y_{i})_{i = 1}^{n}$  are drawn i.i.d. from  $\mathrm{pin}(x,y)$  (joint distribution on  $\mathbb{R}^d\times \{1,\dots ,K\}$ ) and the unlabeled data  $(z_{i})_{i = 1}^{m}$  are drawn i.i.d. from  $\mathrm{p_{all}}$  on  $\mathbb{R}^d$ . In the open-world setting we think of  $\mathrm{p_{all}}$  as the marginal distribution of a mixture of a very large number of classes (much larger than  $K$ ), including the in-distribution ones (see Appendix) in which case it naturally holds for the marginal distribution  $\mathrm{pin}(x)$  that  $\mathrm{pin}(x) > 0$  implies  $\mathrm{p_{all}}(x) > 0$ .

In expectation, the ODST base classifier (Eq. (1)) optimizes (we omit the index 0 in  $f^{(0)}$ ):

$$
\mathbb {E} _ {(X, Y) \sim \mathrm {p} _ {\text {i n}}} \left[ L (Y, f (X)) \right] + \mathbb {E} _ {Z \sim \mathrm {p} _ {\text {a l l}}} \left[ L \left(\frac {1}{K} \mathbf {1}, f (Z)\right) \right]. \tag {4}
$$

Lemma 3.1 Let  $\hat{p}(k|x) = \frac{e^{f_k(x)}}{\sum_{l=1}^{K} e^{fl(x)}}$  then the Bayes optimal prediction for the loss (4) is given for any  $x$  with  $\mathrm{p}_{all}(x) + \mathrm{p}_{in}(x) > 0$  as

$$
\hat {p} (k | x) = \frac {\mathrm {p} _ {i n} (k | x) \mathrm {p} _ {i n} (x) + \frac {1}{K} \mathrm {p} _ {a l l} (x)}{\mathrm {p} _ {i n} (x) + \mathrm {p} _ {a l l} (x)}, \quad k = 1, \ldots , K.
$$

We have directly provided the optimal predictive probability distribution instead of expressing it in terms of the classifier  $f$ . Note that  $\hat{p}(k|x)$  is a monotonic transformation of  $\mathrm{p}_{\mathrm{in}}(k|x)$  and thus preserves for each point the ranking of the classes according to  $\mathrm{p}_{\mathrm{in}}(k|x)$  and thus the optimal decision does not change. However, the ordering of the confidence  $\max_k \mathrm{p}_{\mathrm{in}}(k|x)$  across different inputs  $x$  is influenced significantly by the ratio of  $\mathrm{p}_{\mathrm{in}}(x)$  to  $\mathrm{p}_{\mathrm{all}}(x)$ . Non-task relevant instances where  $\mathrm{p}_{\mathrm{all}}(x)$  is larger than  $\mathrm{p}_{\mathrm{in}}(x)$  are significantly down-weighted and thus will not be selected, whereas if  $\mathrm{p}_{\mathrm{in}}(x)$  is much larger than  $\mathrm{p}_{\mathrm{all}}(x)$  the confidence  $\max_k \hat{p}(k|x)$  is almost equal to  $\max_k \mathrm{p}_{\mathrm{in}}(k|x)$ . Note that the latter case is in particular true for task-relevant images  $(\mathrm{p}_{\mathrm{in}}(x)$  large) as  $\mathrm{p}_{\mathrm{all}}$  is a much more spread out distribution and thus the density value  $\mathrm{p}_{\mathrm{all}}(x)$  will be small. This justifies our OOD aware base teacher and also our post-training calibration step A) as  $\hat{p}$  is under-confident on the in-distribution.

The mathematical treatment of our sample selection strategy is difficult, but it is instructive to check the case where at each iteration  $t + 1$  we impose on all unlabeled points soft-labels,  $\hat{p}_t(k|x)$  defined by the Bayes optimal teacher  $f^{(t)}$  at iteration  $t$ . Then we get the total expected loss at iteration  $t + 1$ :

$$
\mathbb {E} _ {(X, Y) \sim \mathrm {p} _ {\mathrm {i n}}} [ L (Y, f ^ {(t + 1)} (X)) ] + \mathbb {E} _ {Z \sim \mathrm {p} _ {\mathrm {a l l}}} [ L (\hat {p} _ {t} (Z), f ^ {(t + 1)} (Z)) ]. \tag {5}
$$

Lemma 3.2 The Bayes optimal prediction for (5) at iteration  $t$  for  $t \geq 0$  is given for any  $x$  with  $\mathrm{p}_{all}(x) + \mathrm{p}_{in}(x) > 0$  and  $k = 1, \ldots, K$  as

$$
\hat {p} _ {t} (k | x) = \mathrm {p} _ {i n} (k | x) + \left(\frac {\mathrm {p} _ {a l l} (x)}{\mathrm {p} _ {i n} (x) + \mathrm {p} _ {a l l} (x)}\right) ^ {t + 1} \big (\frac {1}{K} - \mathrm {p} _ {i n} (k | x) \big).
$$

In particular, for any  $x$  with  $\mathrm{p_{in}}(x) + \mathrm{p_{all}}(x) > 0$  we get:

$$
\lim  _ {t \to \infty} \hat {p} _ {t} (k | x) = \left\{ \begin{array}{l l} p _ {\text {i n}} (k | x) & \text {i f} p _ {\text {i n}} (x) > 0 \\ \frac {1}{K} & \text {i f} p _ {\text {i n}} (x) = 0. \end{array} \right..
$$

This is the perfect out-distribution aware classifier: Bayes optimal for the in-distribution and maximal uncertain on non-task-related regions  $(\mathrm{p}_{\mathrm{in}}(x) = 0)$ . This justifies our approach from a decision-theoretic perspective. In the finite sample case, neural networks are overconfident on far away regions (Hein et al., 2019) and thus we damp the pseudo-labels on the unlabeled part in step D).

# 4 EVALUATION

First, we evaluate ODST/ODST+ on CIFAR10 in the standard SSL setting with 4k labeled images but now using open world unlabeled data with up to 10M unlabeled images and compare it to existing (open-world) SSL methods. Then in our main evaluation, we use the full training sets of CIFAR10/100 together with the full 80 million unlabeled images from 80MTI and show that we can improve performance in this challenging setting. Moreover, we identify that existing (open-world) SSL methods show a strong distribution shift, which is problematic when using these methods for safety-critical applications.

Training of ODST and NSST and (open world) SSL baselines: For the small scale experiments, we use a WideResNet28x2 for all methods and for the full 80M experiments a standard ResNet50 He et al. (2016) and also the larger PyramidNet272 (Han et al., 2017) with ShakeDrop regularization (Yamada et al., 2019). In each self-training iteration, due to computational restrictions, we use finetuning to train the PyramidNets and train all other architectures from scratch. We perform three iterations for the large-scale experiments and five iterations for CIFAR10-4k and always report the iteration with the best in-distribution validation error.

In the sample selection step of iteration  $t$ , we select the top- $k$  predictions on the full unlabeled dataset per class as potential candidates for the labeled set, where  $k = \frac{5tN}{K}$  ( $N$  is the size of the labeled set,  $K$  the number of classes). Among these points, we select the ones which are above the in-distribution threshold (NSST and ODST) resp. in- and out-distribution threshold (NSST+ and ODST+). This difference in sample selection is the only difference to the plus version and we note that NSST and ODST do not need access to any additional data compared to existing approaches.

As standard SSL baselines, we use the state-of-the-art MixMatch (Berthelot et al., 2019) and FixMatch (Sohn et al., 2020). For open world SSL we use the recent DS3L (Guo et al., 2020) and

Table 1: CIFAR10-4k: Results for 4k labeled images and an unlabeled dataset containing 41k CIFAR10 training and 1 million resp. 10 million unlabeled images from 80MTI or 1M LSUN images. MixMatch (MM) and DS3L perform worse than the "plain" baseline (shown in red). ODST+ and ODST outperform all methods in terms of accuracy. For out-of-distribution detection, we report the average false positive rate (FPR) at  $95\%$  TPR. ODST+ has a FPR more than  $30\%$  better than the closest competitors FixMatch (FM) and MTCF.  

<table><tr><td rowspan="2" colspan="2"></td><td colspan="2">Labeled only</td><td colspan="3">SSL</td><td colspan="5">Open World SSL</td></tr><tr><td>plain</td><td>OE</td><td>MM</td><td>FM</td><td>NSST</td><td>NSST+</td><td>MTCF</td><td>DS3L</td><td>ODST</td><td>ODST+</td></tr><tr><td>4k L</td><td>Acc. ↑</td><td>86.62</td><td>84.91</td><td>81.50</td><td>89.03</td><td>88.70</td><td>87.75</td><td>91.86</td><td>78.40</td><td>93.89</td><td>93.41</td></tr><tr><td>1M TI</td><td>FPR ↓</td><td>78.48</td><td>31.17</td><td>84.16</td><td>66.71</td><td>82.20</td><td>80.60</td><td>62.17</td><td>85.51</td><td>10.51</td><td>16.71</td></tr><tr><td>4k L</td><td>Acc. ↑</td><td>86.62</td><td>86.57</td><td>79.92</td><td>85.43</td><td>88.15</td><td>87.88</td><td>86.76</td><td>-</td><td>92.14</td><td>92.21</td></tr><tr><td>10M TI</td><td>FPR ↓</td><td>78.48</td><td>27.13</td><td>82.28</td><td>82.03</td><td>75.23</td><td>77.48</td><td>73.43</td><td>-</td><td>19.33</td><td>13.78</td></tr><tr><td>4k L</td><td>Acc. ↑</td><td>86.62</td><td>86.30</td><td>81.26</td><td>89.53</td><td>88.30</td><td>87.82</td><td>90.39</td><td>-</td><td>94.31</td><td>94.31</td></tr><tr><td>1M LSUN</td><td>FPR ↓</td><td>78.48</td><td>43.98</td><td>85.27</td><td>75.50</td><td>77.47</td><td>75.73</td><td>54.08</td><td>-</td><td>23.26</td><td>25.61</td></tr></table>

MTCF (Yu et al., 2020). We only compare to existing approaches in the small scale setting as DS3L and MTCF do not scale to larger unlabeled datasets due to excessive memory consumption (several terabytes) and MTCF's domain training requiring 5 days for 10M datapoints. Self-training methods are superior in this regard as the number of full passes through the unlabeled data is limited by the number of training iterations and labeling can be parallelized arbitrarily to scale to large datasets.

Unlabeled dataset: We use 80 million tiny images Torralba et al. (2008) (80MTI) as unlabeled dataset, which has been created by querying 53,464 different nouns from the wordnet hierarchy. Note that CIFAR10 and CIFAR100 are subsets of 80MTI (Krizhevsky & Hinton, 2009), and we remove (near)-duplicates from 80MTI, see Appendix B.5 for details. For the small scale experiments we use randomly selected subsets of 1M resp. 10M images and additionally a 1M LSUN subset.

Evaluation metrics: In an open-world setting, it is important to not suffer from a distribution shift. Thus, in addition to test accuracy, we evaluate the out-of-distribution (OOD) detection performance by reporting average false positive rate over several OOD datasets: test set of CIFAR100 or CIFAR10, SVHN, LSUN-CR(Yu et al., 2015) with Flowers(Nilsback & Zisserman, 2008) plus Food-101(Bossard et al., 2014) for CIFAR10 and FGVC-Aircraft(Maji et al., 2013) for CIFAR100.

# 4.1 CIFAR10-4K

A standard setting in SSL is to use 4k training images of CIFAR10 together with 5k as validation set (Berthelot et al., 2019; Yu et al., 2020). The remaining 41k training images are used as unlabeled data, which in our challenging open world setting are further mixed with different out-of-distribution datasets: i) 1M resp. ii) 10M unlabeled images from 80MTI or iii) 1M LSUN images. Note that 80MTI contains further task relevant samples for CIFAR10.

The class-specific in-distribution threshold is set to  $\alpha = 0.98$  (meaning that we accept maximally  $2\%$  false positives per class on the in-distribution validation set) and for  $\mathrm{ODST + }$  and  $\mathrm{NSST + }$  we use the same value as out-distribution threshold. For the creation of an out-distribution validation set for  $\mathrm{ODST + }$  and  $\mathrm{NSST + }$ , there exist two strategies. The first one is to use an existing dataset with a sufficient variety and remove any potentially task-related samples. For this, we use a subset of  $2\mathrm{k}$  CIFAR100 images as validation set for the 80MTI experiments and remove the classes "bus" and "pickup-truck" as they can be confused with "car" and "truck" from CIFAR10. The second approach is to manually label a sufficient number of unlabeled samples as out-of-distribution. We simulate this by using a subset of  $2\mathrm{k}$  unseen LSUN images in the 1M LSUN experiment.

The results in Table 1 show that the considered open world setting is challenging due to the low number of task-relevant samples (lower bounded by  $4.1\%$  resp.  $0.41\%$ ). MixMatch but also the open world SSL method DS3L perform worse (shown in red) than training on the labeled data only (plain). Our ODST and  $\mathrm{ODST + }$  outperform all other (open world) SSL methods in terms of test accuracy and in both cases improve significantly over the base teacher model OE (Hendrycks et al., 2019) trained only on the labeled data. While for 1M unlabeled data points from 80MTI or LSUN, MTCF and FixMatch perform well, they show severe performance degradation for 10M data points. We believe that the non-adaptive confidence threshold of FixMatch resp. the Otsu

Table 2: CIFAR10-50k: We show test accuracy and FPR@95TPR as out-of-distribution detection performance. ODST+ has the best improvement (1.26%) and final test error (1.93%) for Resnet50 and is the only method which improves for the Pyramid272 architecture by 0.29% with 1.43% test error whereas NSST and NSST+ degrade in test performance.  

<table><tr><td rowspan="2"></td><td colspan="6">ResNet50</td><td colspan="5">PyramidNet</td></tr><tr><td>plain</td><td>OE</td><td>NSST</td><td>NSST+</td><td>ODST</td><td>ODST+</td><td>plain</td><td>OE</td><td>NSST</td><td>NSST+</td><td>ODST+</td></tr><tr><td>Acc.↑</td><td>96.11</td><td>96.81</td><td>96.86</td><td>96.93</td><td>97.98</td><td>98.07</td><td>98.49</td><td>98.28</td><td>98.24</td><td>98.13</td><td>98.57</td></tr><tr><td>FPR↓</td><td>45.54</td><td>4.40</td><td>49.72</td><td>51.76</td><td>4.30</td><td>3.84</td><td>22.76</td><td>2.56</td><td>33.51</td><td>34.71</td><td>2.40</td></tr></table>

threshold of MTCF are not sufficient for the accurate selection of task-related images. Even though the prediction performance is reasonable for 1M samples, the classifiers show a strong distribution shift as illustrated in Fig. 1 and Fig. 6, where we visualize unlabeled samples which are considered by the different methods to be task-relevant. As the failure of MTCF and FixMatch is not noticeable from test accuracy, we strongly suggest that open world SSL papers should visualize high confidence samples and evaluate OOD detection performance. In Table 1 we therefore report the average FPR over different OOD datasets. ODST and  $\mathrm{ODST + }$  have a  $30\%$  better FPR than any other SSL method. While  $\mathrm{ODST + }$  and ODST show similar empirical performance we demonstrate potential advantages of  $\mathrm{ODST + }$  regarding AI safety in Appendix B.6. In Appendix C.1 we provide details over the full run of  $\mathrm{ODST + }$  and ODST and report corruption robustness on CIFAR10-C.

Figure 2: CIFAR10-50k: Plot of randomly chosen, exclusively selected samples from 80MTI for NSST (top) and ODST (bottom) over all three iterations. False positives are marked red.  

<table><tr><td></td><td colspan="5">1st Iteration (25k)</td><td colspan="5">2nd Iteration (50k)</td><td colspan="5">3rd Iteration (75k)</td></tr><tr><td rowspan="2">NSST</td><td>bird - 1.00</td><td>horse - 1.00</td><td>deer - 1.00</td><td>car - 1.00</td><td>cat - 1.00</td><td>truck - 1.00</td><td>cat - 1.00</td><td>horse - 1.00</td><td>bird - 1.00</td><td>car - 1.00</td><td>cat - 0.99</td><td>deer - 0.98</td><td>car - 0.98</td><td>dog - 0.99</td><td>truck - 0.99</td></tr><tr><td>horse - 1.00</td><td>dog - 1.00</td><td>plane - 1.00</td><td>bird - 1.00</td><td>ship - 1.00</td><td>car - 1.00</td><td>horse - 1.00</td><td>car - 1.00</td><td>car - 1.00</td><td>plane - 1.00</td><td>truck - 0.99</td><td>cat - 0.99</td><td>truck - 0.99</td><td>plane - 0.99</td><td>horse - 0.98</td></tr><tr><td rowspan="2">ODST</td><td>bird - 1.00</td><td>car - 1.00</td><td>dog - 1.00</td><td>plane - 1.00</td><td>car - 1.00</td><td>deer - 1.00</td><td>car - 1.00</td><td>bird - 1.00</td><td>deer - 1.00</td><td>bird - 1.00</td><td>cat - 1.00</td><td>car - 1.00</td><td>dog - 1.00</td><td>car - 1.00</td><td>bird - 1.00</td></tr><tr><td>frog - 1.00</td><td>cat - 1.00</td><td>frog - 1.00</td><td>car - 1.00</td><td>bird - 1.00</td><td>frog - 1.00</td><td>horse - 1.00</td><td>plane - 1.00</td><td>cat - 1.00</td><td>car - 1.00</td><td>ship - 1.00</td><td>dog - 1.00</td><td>horse - 1.00</td><td>frog - 1.00</td><td>ship - 1.00</td></tr></table>

# 4.2 CIFAR10-50k

Up to our knowledge, no SSL method could yet show performance improvements in an open world setting when using the full training set of CIFAR10 as labeled set. We show that this indeed possible when using 80MTI as unlabeled dataset. While DS3L and MTCF do not scale to this setting, both noisy student variants outperform any consistency-based (open world) method in the 10M setting and act as a strong baseline for our ODST and ODST+ models. For the in-distribution validation set, we use the recent CIFAR10.1 dataset (Recht et al., 2018) designed to assess the generalization of classifiers trained on CIFAR10. As out-distribution validation set for ODST+, we use 2k CIFAR100 samples without conflicting classes (See Appendix B.6 for details). As threshold parameter  $\alpha$  we use  $99.8\%$  which is conservative but justified by the high accuracy of the base CIFAR10 classifiers. In Table 2, we report the results for both architectures.

ResNet50: ODST+ improves test accuracy by  $1.26\%$  from  $96.81\%$  to  $98.07\%$  and outperforms NSST and  $\mathrm{NSST + }$  by at least  $1.14\%$ . We are not aware that such a high test accuracy has been reported before for a ResNet50 on CIFAR10<sup>1</sup>. Even though NSST and  $\mathrm{NSST + }$  improve slightly upon the baseline, Figure 2 highlights that even with the OD threshold,  $\mathrm{NSST + }$  suffers from a distribution shift and identifies completely unrelated samples as task-relevant and thus self-training degrades very early e.g. almost all images containing humans are classified as "horse" or "dog". ODST+

Table 3: CIFAR100-45k: ODST (+) have the best accuracy for both model architectures. SSLresults in red are worse than using labeled data only.  

<table><tr><td rowspan="2"></td><td colspan="6">ResNet50</td><td colspan="5">PyramidNet</td></tr><tr><td>plain</td><td>OE</td><td>NSST</td><td>NSST+</td><td>ODST</td><td>ODST+</td><td>plain</td><td>OE</td><td>NSST</td><td>NSST+</td><td>ODST+</td></tr><tr><td>Acc.↑</td><td>80.69</td><td>79.98</td><td>82.44</td><td>82.03</td><td>83.54</td><td>84.09</td><td>88.07</td><td>87.60</td><td>87.71</td><td>87.53</td><td>88.66</td></tr><tr><td>FPR↓</td><td>72.05</td><td>43.88</td><td>73.47</td><td>82.54</td><td>35.64</td><td>38.72</td><td>60.20</td><td>27.97</td><td>74.90</td><td>73.91</td><td>29.22</td></tr></table>

<table><tr><td>palm tree - 0.99dinosaur - 0.99</td><td>fox - 0.99</td><td>woman - 1.00</td><td>tiger - 0.96</td><td>plate - 0.99</td></tr><tr><td>crab - 0.99</td><td>shrew - 0.98</td><td>sunflower - 0.98</td><td>baby - 0.99</td><td>wolf - 0.98</td></tr><tr><td colspan="5">NSST</td></tr></table>

Figure 3: CIFAR100-45k: Random selection of 14 samples in the third iteration of self-training for the ResNet50 architecture. False positives are marked in red.

slightly outperforms ODST, most likely due to a better sample selection. But even without the additional OOD-validation set ODST improves the OOD performance from  $4.40\%$  FPR to  $4.30\%$ , which is further improved by  $\mathrm{ODST + }$  to  $3.89\%$ . We highlight that all other methods show worse FPR than the base model, which is likely caused by the distribution shift which is concerning regarding AI safety. This again highlights that just relying on test accuracy can be misleading.

PyramidNet272: As the test error of the baseline OE is already below  $2\%$ , further improvements are much harder to realize and can only be obtained by succeeding in the challenging task of selecting high-quality, task-relevant samples from the large pool of unlabeled samples. ODST+ achieves this and improves to  $1.31\%$  accuracy which up to our knowledge is the best test accuracy achieved with this architecture (previously  $1.36\%$ , see Harris et al. (2020)). We had to skip ODST due to computational constraints. In contrast, due to their poor sample selection NSST and  $\mathrm{NSST + }$  degrade from the first iteration on and thus can not profit from unlabeled data.

# 4.3 CIFAR100-45k

For CIFAR100 we randomly select 50 out of the 500 training samples per class as in-distribution validation set. We use 2k CIFAR10 samples without classes "car" and "truck", as they are ambiguous wrt to "pickup-truck", as out-distribution set validation for the Plus methods. In- and out-distribution thresholds are set to  $98\%$ , due to the lower base accuracy on CIFAR100. The results are in Table 3, where we see that on the ResNet ODST+ improves the test accuracy by  $4.11\%$  to  $84.09\%$  compared to the OE model and to  $83.54\%$  for ODST. While NSST and NSST+ improve by  $1.75\%$  to  $82.44\%$  resp. 1.34 to  $82.03\%$ , we again notice a distribution shift which can be observed from a random selection of samples that pass the threshold and are thus considered task-relevant (Figure 3). This is further verified by the very high FPR of  $73.47\%$  resp.  $74.19\%$  for OOD detection for the ResNet architecture, whereas both ODST variants improve OOD-performance over the OE baseline. For the PyramidNet, both NSST variants are worse than the plain baseline in accuracy and FPR whereas ODST+ clearly improves test accuracy with a small degradation in FPR.

# 5 Conclusion

We show that ODST can leverage large unlabeled open-world datasets with only a tiny fraction of task-related samples and consistently improves over the baseline and other SSL methods. The resulting classifiers are more accurate, robust and show better out-distribution detection performance. Moreover, we observe that all competing methods suffer from a distribution shift, which is problematic regarding AI safety, whereas ODST and in particular ODST+ shows almost no such degradation.

# Reproducibility Statement

We include our entire codebase in the supplementary material of our submission which allows to easily reproduce our results. This includes indices to any train/validation splits and indices of near-duplicates in 80MTI such that it is possible to run the experiments on the same data that was used in the paper. Due to the stochastic nature of neural network training, results can vary slightly.

# Ethics Statement

There are no conflicts of interest in this work. A potential danger of semi-supervised learning methods, in particular self-training, is that they suffer from a distribution shift. This danger is even more severe in an open-world scenario. As we show in this paper, heavy distribution shifts happen for other SSL methods but our ODST is very robust to it. Nevertheless, we have introduced ODST+ where an extra out-distribution validation set is used to prevent distribution shifts. While in the considered classification problems ODST was already very robust, this might seem unnecessary. But in our opinion this is a too much benchmark-focused point of view. In order to build trustworthy ML systems using open-world SSL, we thus encourage subsequent work to i) check out-of-distribution detection performance, ii) check the selected unlabeled samples, and iii) consider an out-distribution validation set for the construction of the models.

# References

David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin Raffel. Mixmatch: A holistic approach to semi-supervised learning. In NeurIPS, 2019.  
Avrim Blum and Tom Mitchell. Combining labeled and unlabeled data with co-training. In  $COLT$ , 1998.  
Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 - mining discriminative components with random forests. In ECCV, 2014.  
T. E. Boult, S. Cruz, A.R. Dhamija, M. Gunther, J. Henrydoss, and W.J. Scheirer. Learning and the unknown: Surveying steps toward openworld recognition. In AAAI, 2019.  
S. Boyd and L. Vandenberghe. Convex Optimization. Cambridge University Press, 2004.  
Yair Carmon, Aditi Raghunathan, Ludwig Schmidt, John C Duchi, and Percy S Liang. Unlabeled data improves adversarial robustness. In NeurIPS, 2019.  
O. Chapelle, B. Scholkopf, and A. Zien. Semi-Supervised Learning. MIT Press, 2006.  
Yanbei Chen, Xiatian Zhu, Wei Li, and Shaogang Gong. Semi-supervised learning under class distribution mismatch. In AAAI, 2020.  
Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le. Autoaugment: Learning augmentation strategies from data. In CVPR, 2019.  
Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In CVPR Workshop, 2020.  
Terrance DeVries and Graham W Taylor. Improved regularization of convolutional neural networks with cutout. arXiv preprint, 2017.  
Yves Grandvalet and Yoshua Bengio. Semi-supervised learning by entropy minimization. In NeurIPS, 2005.  
C. Guo, G. Pleiss, Y. Sun, and K. Weinberger. On calibration of modern neural networks. In ICML, 2017.  
Lan-Zhe Guo, Zhen-Yu Zhang, Yuan Jiang, Yu-Feng Li, and Zhi-Hua Zhou. Safe deep semi-supervised learning for unseen-class unlabeled data. In ICML, 2020.

Dongyoon Han, Jiwhan Kim, and Junmo Kim. Deep pyramidal residual networks. In CVPR, 2017.  
Ethan Harris, Antonia Marcu, Matthew Painter, Mahesan Niranjan, and Adam Prügel-Bennett Jonathon Hare. Fmix: Enhancing mixed sample data augmentation. arXiv preprint, 2020.  
K. He, X. Zhang, , S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.  
M. Hein, M. Andriushchenko, and J. Bitterwolf. Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. In CVPR, 2019.  
D. Hendrycks, M. Mazeika, and T. Dietterich. Deep anomaly detection with outlier exposure. In ICLR, 2019.  
Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In ICLR, 2019.  
Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In ICLR, 2017.  
Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, and Ondrej Chum. Label propagation for deep semi-supervised learning. In CVPR, 2019.  
M. Stinchcombe K. Hornik and H. White. Multilayer feedforward networks are universal approximators. Neural Networks, 2:359-366, 1989.  
A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.  
Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learning. arXiv preprint, 2016.  
Dong-Hyun Lee. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In Workshop on challenges in representation learning, ICML, 2013.  
K. Lee, H. Lee, K. Lee, and J. Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. In NeurIPS, 2018.  
S. Liang, Y. Li, and R. Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. In ICLR, 2018.  
S. Maji, J. Kannala, E. Rahtu, M. Blaschko, and A. Vedaldi. Fine-grained visual classification of aircraft. Technical report, 2013.  
Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training: a regularization method for supervised and semi-supervised learning. IEEE PAMI, 41(8):1979-1993, 2018.  
A. Nguyen, J. Yosinski, and J. Clune. Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. In CVPR, 2015.  
Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In ICVGIP, 2008.  
Avital Oliver, Augustus Odena, Colin A Raffel, Ekin Dogus Cubuk, and Ian Goodfellow. Realistic evaluation of deep semi-supervised learning algorithms. In NeurIPS, 2018.  
Aristotelis-Angelos Papadopoulos, Mohammad Reza Rajati, Nazim Shaikh, and Jiamian Wang. Outlier exposure with confidence control for out-of-distribution detection. arXiv preprint, 2019.

Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do cifar-10 classifiers generalize to CIFar-10? In arXiv preprint, 2018.  
Ellen Riloff. Automatically generating extraction patterns from untagged text. In AAAI, 1996.  
Ellen Riloff and Janyce Wiebe. Learning extraction patterns for subjective expressions. In EMNLP, 2003.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. IJCV, 115(3):211-252, 2015.  
Mehdi Sajjadi, Mehran Javanmardi, and Tolga Tasdizen. Regularization with stochastic transformations and perturbations for deep semi-supervised learning. In NeurIPS, 2016.  
H Scudder. Probability of error of some adaptive pattern-recognition machines. IEEE Transactions on Information Theory, 11(3):363-371, 1965.  
Weiwei Shi, Yihong Gong, Chris Ding, Zhiheng MaXiaoyu Tao, and Nanning Zheng. Transductive semi-supervised deep learning using min-max features. In ECCV, 2018.  
Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In NeurIPS, 2020.  
Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In NeurIPS, 2017.  
Antonio Torralba, Rob Fergus, and William T Freeman. 80 million tiny images: A large data set for nonparametric object and scene recognition. IEEE PAMI, 30(11):1958-1970, 2008.  
Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600-612, 2004.  
Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves imagenet classification. In CVPR, pp. 10687-10698, 2020.  
I Zeki Yalniz, Herve Jégou, Kan Chen, Manohar Paluri, and Dhruv Mahajan. Billion-scale semi-supervised learning for image classification. arXiv preprint, 2019.  
Yoshihiro Yamada, Masakazu Iwamura, Takuya Akiba, and Koichi Kise. Shakedrop regularization for deep residual learning. IEEE Access, 2019.  
F. Yu, A. Seff, Y. Zhang, S. Song, T. Funkhouser, and J. Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint, 2015.  
Qing Yu, Daiki Ikami, Go Irie, and Kiyoharu Aizawa. Multi-task curriculum framework for open-set semi-supervised learning. In ECCV, 2020.  
S. Zagoruyko and N. Komodakis. Wide residual networks. In BMVC, 2016.  
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586-595, 2018.  
Giulio Zhou, Subramanya Dulloor, David G Andersen, and Michael Kaminsky. Edf: ensemble, distill, and fuse for easy video labeling. arXiv preprint, 2018.  
Xiaojin Zhu and Andrew B Goldberg. Introduction to semi-supervised learning. Synthesis lectures on artificial intelligence and machine learning, 3(1):1-130, 2009.  
Xiaojin Jerry Zhu. Semi-supervised learning literature survey. Technical report, University of Wisconsin-Madison Department of Computer Sciences, 2005.

First, we give an overview over the content of the appendix:

- in Section A we give the missing proofs from the paper.  
- Section B contains a more detailed description and hyperparameters for all methods and datasets. More specifically, we discuss our duplicate removal for 80MTI and the choice of the out-distribution validation set  
- In Section C, we give a more detailed breakthrough of the main results from the paper.  
- Section D contains ablations studies in the large scale setting with the full 80 tiny images setting.
