# LABEL CLEANING WITH A LIKELIHOOD RATIO TEST

Anonymous authors

Paper under double-blind review

# ABSTRACT

To collect large scale annotated data, it is inevitable to introduce label noise, i.e., incorrect class labels. A major challenge is to develop robust deep learning models that achieve high test performance despite training set label noise. We introduce a novel approach that directly cleans labels in order to train a high quality model. Our method leverages statistical principles to correct data labels and has a theoretical guarantee of the correctness. In particular, we use a likelihood ratio test (LRT) to flip the labels of training data. We prove that our LRT label correction algorithm is guaranteed to flip the label so it is consistent with the true Bayesian optimal classifier with high probability. We incorporate our label correction algorithm into the training of deep neural networks and train models that achieve superior testing performance on multiple public datasets.

# 1 INTRODUCTION

Label noise is ubiquitous in real world data. It may be caused by unintentional mistakes of manual or automatic annotators (Yan et al., 2014; Veit et al., 2017). It may also be introduced by malicious attackers (Steinhardt et al., 2017). Noisy labels impair the performance of a model (Smyth et al., 1994; Brodley & Friedl, 1999), especially a deep neural network, which tends to have strong memorization power (Frnay & Verleysen, 2014; Zhang et al., 2017). Improving the robustness of a model to label noise is a crucial yet challenging task in many applications (Mnih & Hinton, 2012; Wu et al., 2018). Existing methods mainly follow two directions, probabilistic reasoning and data selecting.

Probabilistic methods explicitly model a noise transition matrix, namely, the probability of one label being corrupted into another (Goldberger & Ben-Reuven, 2017; Patrini et al., 2017). The transition matrix is often estimated from the data, and is used to re-calibrate the training loss or to correct the prediction. Explicit estimation of the transition matrix can be problematic due to the large variation of noise patterns, e.g., uniform noise, asymmetric noise, or mixtures. Furthermore, the transition matrix size is quadratic to the number of classes, making the estimation task prohibitive when the data has hundreds or even thousands of classes.

Data-selecting methods are agnostic of the underlying noise pattern. These methods gradually collect clean data whose labels are trustworthy (Malach & Shalev-Shwartz, 2017; Jiang et al., 2018; Han et al., 2018). As more clean data are collected, the quality of the trained models improves. The major issue of these methods is the lack of a quantitative control of the quality of the collected clean data. Without a principled guideline, it is hard to find the correct data collection pace. An aggressive selection can unknowingly accumulate irreversible errors. On the other hand, an overly-conservative strategy can be very slow in training, or stops with insufficient clean data and mediocre models.

We propose a novel method with the benefit from both the probabilistic and the data-selecting approaches. Similar to data-selecting methods, our method continuously improves the purity of the data labels by correcting the noise-corrupted ones. Meanwhile, we improve the classifier using the updated labels. Our label correction algorithm is based on statistical principles and is theoretically guaranteed to deliver a high quality label set. Instead of explicitly estimating the transition matrix, the correction algorithm only depends on the prediction of the current model, denoted as  $f$ . Using an  $f$ -based likelihood ratio test, we determine whether the current label of each data should be corrected. Our main theorem proves that the label correction algorithm will clean a majority of noisy labels with high probability.

In practice, we incorporate the label correction algorithm into the training of deep neural networks. Our method iteratively updates the labels of the data while continuously training a deep neural

network. To ensure the deep neural network does not overfit with noise labels that are yet to be corrected, we introduce a new retroactive loss term that regulates the model by enforcing its consistency with models in previous epochs. The rationale is that the model in an earlier training stage tends to fit the true signal rather than noise, although its overall performance is sub-optimal. Through experiments on various datasets with various noise patterns and levels, we show that our method produces robust neural network models with superior performance.

To the best of our knowledge, our method is the first to correct labels with theoretical guarantees. It is has advantages over both probabilistic methods and data-selecting methods. Compared with other data-selecting methods, it has a better quantitative control of the label quality and thus is less brittle when generalizing to different datasets and different noise patterns. Also note that we are not selecting clean data. Instead, we correct labels and always use the whole training set to train. This brings an additional advantage of fully leveraging the data. Compared with other probabilistic methods, our correction algorithm assumes a rather general family of underlying noise patterns and avoids an explicit estimation of the transition matrix.

# 1.1 RELATED WORK

Recent works could be classified into three categories. One is to model and employ noise transition matrix to correct the loss. For example, Patrini et al. (2017) proposes to correct the loss function with estimated noise pattern. The resulting loss is an unbiased estimator of the ground truth loss, and enables the trained model to achieve better performance. However, such an estimator relies on strong assumptions and could be inaccurate in certain scenarios. Reed et al. (2014) considers modeling the noise pattern with a hidden layer. The learning of this hidden layer is regularized with a feature reconstruction loss, yet without a guarantee that the true label distribution is learned. Another method mentioned in their work is to minimize the entropy of neural network output; however, this method tends to predict a single class. To address this weakness, Hendrycks et al. (2019) proposes to utilize a small number of kosher data to pre-train a network and estimate the noise pattern. However, such clean data may not always be available in practice.

Another strategy to handle noisy label problem is to design models that are intrinsically robust to noisy data. Crammer et al. (2009) introduces a regularized confidence weighting learning algorithm (AROW), where parameters of classifiers are assumed normally distributed and the mean and covariance of this distribution is updated during training. The idea here is to preserve the weight distribution as much as possible while requiring the model to maintain predictive ability. In the follow-up work (Crammer & Lee 2010) proposes to improve this algorithm by herding the updating direction via specific velocity field (NHERD), achieving better performance. Both of these works impose parametric constraint on parameters, which could prevent classifiers from adapting to complex data set. Arpit et al. (2017a) shows that deep neural networks tend to learn meaningful patterns before they over-fit to noisy ones. Based on this observation, they propose to add Gaussian or adversarial noise to input when training with noisy labels, and empirically show that such data perturbation is able to make the resulting model more robust. Other commonly adopted techniques, such as weight decay and dropout, are also shown to be effective in increasing the robustness of trained classifier (Arpit et al. 2017a; Zhang et al. 2017). However, the intrinsic reasons for this phenomenon still remains unclear and overfitting to noisy label is still inevitable.

Apart from the above mentioned strategies, one recent work proposes to correct the corrupted labels during training. In particular, Tanaka et al. (2018) propose to jointly train the deep network and estimate the underlying true labels. While achieving improved performance, their method largely relies on the prior distribution and is difficult to deploy under cases where there is a large number of classes.

Finally, beyond deep learning framework, there are several theoretic works that demonstrate the robustness of a variety of losses to label noise (Long & Servedio 2010; Natarajan et al. 2013; Ghosh et al. 2015; van Rooyen et al. 2015). Following the work of (Wang & Chaudhuri 2018), Gao et al. (2016) proposes an algorithm that can converge to the Bayesian optimal classifier under different noisy settings. Moreover, they discuss the performance of k-nearest neighbor (KNN) classifiers however, the problem with KNN is that it is computationally intensive and thus difficult to be incorporated into a learning context. Within the framework of deep learning, there are more efforts that need to be made to bridge theory and practice.

# 2 METHOD

Our method has two synchronized modules, the training module and the label correction module. The training module continues to learn a classifier based on the current labels. Meanwhile, the label correction module uses the prediction of the classifier to correct labels.

We start with some preliminaries necessary for the exposition (Section 2.1). In Section 2.2, we explain our correction algorithm. It uses the prediction of the classifier (trained on noisy labels) for a likelihood ratio test. Based on the test result, it decides whether to correct the label of a data. Theoretically, we prove that under certain assumptions of the prediction,  $f$ , the algorithm will change the labels to the correct ones, i.e., the ones consistent with the Bayes optimal classifier (Theorem 1).

In Section 2.3, we present the overall training method. We incrementally train a deep neural network based on the corrected labels,  $\widetilde{y}$ . Meanwhile, the network's prediction is used for label correction. To improve the quality of the prediction, we introduce a new loss term, called the

![](images/ba5d017f4e35f04bd5828821e89491c2612484952845cd6dc74ed484d6c598e6.jpg)  
Figure 1: The overview of our method.

retroactive loss. The goal is to regulate the model using models trained in earlier epochs, as they may be less overfitting with corrupted labels.

For the label correction module, we focus on a binary classifier. But the algorithm and the theoretical results can easily be generalized to the multiclass setting (Corollary 1).

# 2.1 PRELIMINARIES

Let  $\mathcal{X}$  be the feature space,  $\mathcal{Y} = \{0,1\}$  be the label space, and  $D$  be an unknown distribution on  $\mathcal{X} \times \mathcal{Y}$ . The joint probability can be factored as  $D(\boldsymbol{x},y) = \operatorname*{Pr}(y|\boldsymbol{x})\operatorname*{Pr}(\boldsymbol{x})$ . We denote by  $\eta(\boldsymbol{x}) = \operatorname*{Pr}(y = 1|\boldsymbol{x})$  the true conditional probability. The Bayes risk of a classifier  $h: \mathcal{X} \to \mathcal{Y}$  is  $R(D,h) = \operatorname*{Pr}_{(\boldsymbol{x},y) \sim D}(h(\boldsymbol{x}) \neq y)$ . A Bayes optimal classifier is the minimizer of the Bayes risk, i.e.,  $h^* = \arg \min_h R(D,h)$ . It can be calculated using the true conditional probability,  $\eta$ ,

$$
h ^ {*} (\boldsymbol {x}) = \mathbf {1} _ {\left\{\eta (\boldsymbol {x}) \geq \frac {1}{2} \right\}} (\mathbf {x}) = \left\{ \begin{array}{l l} 1 & \text {i f} \quad \eta (\boldsymbol {x}) > = \frac {1}{2} \\ 0 & \text {o t h e r w i s e} \end{array} \right. \tag {1}
$$

We assume the true conditional probability,  $\eta$ , satisfies the Tsybakov condition (also called the TNC condition) (Tsybakov et al., 2004). This condition stipulates that the uncertainty of  $\eta$  and thus the Bayes optimal classifier is bounded. This assumption in general helps to bound the margin of Bayesian decision rule such that the proposed classifier's risk can be bounded accordingly.

Definition 1 (Tsybakov Condition). There exist  $C > 0, \lambda > 0$ , and  $t_0 \in (0, \frac{1}{2}]$ , such that for all  $t \leq t_0$ ,

$$
\Pr \left[ \left| \eta (\boldsymbol {x}) - \frac {1}{2} \right| <   t \right] \leq C t ^ {\lambda}.
$$

The noisy label setting. Instead of samples from  $D$ , we are given a sample set with noisy labels  $S = \{(\pmb{x},\widetilde{y})\}$  where  $\widetilde{y}$  is the possibly corrupted label based on the true label  $y$ . We assume a transition probability  $\tau_{i\rightarrow j} = \operatorname*{Pr}(\widetilde{y} = j|y = i)$ , i.e., the chance a ground truth label  $y$  is flipped from class  $i$  to class  $j$ . For simplicity, we denote  $\tau_{ij} = \tau_{i\rightarrow j}$ . The transition probabilities  $\tau_{01}$  and  $\tau_{10}$  are independent of the true joint distribution  $D$  and the feature  $x$ . We denote the conditional probability of the noisy labels as  $\widetilde{\eta} (\pmb {x}) = \operatorname *{Pr}(\widetilde{y} = 1|\pmb {x})$ . In short, we call  $\widetilde{\eta}$  the noisy conditional probability. It is linearly to the true conditional probability,  $\eta$ :

$$
\widetilde {\eta} (\boldsymbol {x}) = (1 - \tau_ {1 0}) \eta (\boldsymbol {x}) + \tau_ {0 1} [ 1 - \eta (\boldsymbol {x}) ] = (1 - \tau_ {0 1} - \tau_ {1 0}) \eta (\boldsymbol {x}) + \tau_ {0 1}. \tag {2}
$$

# 2.2 THE LABEL CORRECTION ALGORITHM AND ITS THEORETICAL GUARANTEES

Our label correction algorithm takes in a current neural network prediction  $f: \mathcal{X} \to [0,1]$  (i.e., an estimation of  $\eta$  based on the noisy labels). For all training data and their current noisy label  $(\pmb{x},\widetilde{\pmb{y}})$ ,

![](images/b70e08abfc15ba7e2c83af4c350c2e45b581331be6b94b14fd769a2cc9da3547.jpg)  
(a) Noisy labels and  $f$ .

![](images/3dfa5810de9ac07acebde97273be22b82b0255d424a7bc2166f22e2ce025b5fe.jpg)  
(b) Corrected labels and  $\eta$

![](images/b128695d47fa17ef7ad0bbd15357cbe7e43355bf3025ef72e023d6bdf77d3234.jpg)  
(c) LR for  $\widetilde{y} = 1$

![](images/8be04ed808b69a71a2273a35a9a21cba2b98b55b4c6ccc0db511b2568baf11ba.jpg)  
(d) LR for  $\widetilde{y} = 0$  
Figure 2: An illustration of the label correction algorithm.  $\Delta$  is set to 1. (a): a corrupted sample and its corresponding classifier prediction  $f$ . (b): after correction, the labels are consistent with the true conditional probability,  $\eta$ . (c): likelihood ratio for  $\widetilde{y} = 1$ . Data with  $x < 0$  are corrected to  $\widetilde{\eta}_{new} = 0$  as  $LR(\boldsymbol{x})$  are below  $\Delta = 1$ . (d): likelihood ratio for  $\widetilde{y} = 0$ . Data with  $x > 0$  are corrected to  $\widetilde{\eta}_{new} = 1$  as  $LR(\boldsymbol{x})$  are below  $\Delta = 1$ .

the correction algorithm uses  $f$  to run a likelihood ratio test and to decide whether to flip the label according to the result. The goal of the likelihood test is to decide whether the null hypothesis,  $H_0: \widetilde{y} = y$ , is true. If yes,  $\widetilde{y}$  is accepted as it is. Otherwise, we flip  $\widetilde{y}$ , so that hopefully it becomes  $y$ . Formally, the likelihood ratio is defined as

$$
\operatorname {L R} (f, \boldsymbol {x}, \widetilde {y}) = \frac {f (\boldsymbol {x}) ^ {\widetilde {y}} [ 1 - f (\boldsymbol {x}) ] ^ {1 - \widetilde {y}}}{f (\boldsymbol {x}) ^ {1 - \widetilde {y}} [ 1 - f (\boldsymbol {x}) ] ^ {\widetilde {y}}} \tag {3}
$$

We compare this likelihood ratio with a predetermined value  $\Delta$ . If  $\mathrm{LR}(f, x, y) \leq \Delta$ , we reject the null hypothesis and flip the label  $\widetilde{y}_{new} = 1 - \widetilde{y}$ . Otherwise, the label remains unchanged,  $\widetilde{y}_{new} = \widetilde{y}$ . See Figure 2 for an illustration of the algorithm.

Note that the constant  $\Delta$  depends on the underlying noise pattern,  $f$  and  $\widetilde{y}$ . Below we show that if we choose  $\Delta$  carefully, the label correction algorithm is guaranteed to make proper correction and clean most of the corrupted labels. However, in practice,  $\Delta$  is unknown and needs to be tuned.

Intuition. In the likelihood ratio (Eq. (3)), the numerator is the likelihood that the prediction  $f$  is consistent with the noisy label  $\widetilde{y}$ . The denominator is the likelihood of the opposite case. When this ratio is smaller than 1, we know that the prediction of  $f$  is more likely to be inconsistent with  $\widetilde{y}$ . But whether  $f$  agrees with  $\widetilde{y}$  is not the hypothesis to test. To test the intended null hypothesis  $(\widetilde{y} = y)$ , we need to check whether  $\widetilde{y}$  is consistent with the true conditional distribution  $\eta$ , namely, the Bayes optimal classifier prediction  $h^{*}(\boldsymbol{x})$ . To this end, we assume  $f$  is a close enough approximate of  $\widetilde{\eta}$  as it is trained on the noisy labels. This way, testing whether  $f$  agrees with  $\widetilde{y}$  is close to testing whether  $\widetilde{\eta}$  agrees with  $\widetilde{y}$ , except that the threshold  $\Delta$  needs to be carefully chosen. Another issue we need to consider is that the  $\Delta$  is unknown. Our main theorem will bound the chance of failed correction by how close  $f$  approximates  $\widetilde{\eta}$  and how close we can set  $\Delta$  to the perfect one.

Remark 1. Our likelihood testing is the uniformly most powerful one for the intended hypothesis, based on the Neyman-Pearson Lemma. In other words, this test has the strongest statistical power in rejecting a false null hypothesis.

# 2.2.1 FORMAL STATEMENT OF THE ALGORITHM AND THE THEOREM

We start by assuming  $f$  depends linearly on  $\eta$ , i.e.,  $f(\pmb{x}) = a\eta(\pmb{x}) + b$ , in which  $a, b > 0$  are known constants. But this constraint will be relaxed later (Remark 2). Consider three different conditions based on the noise patterns:  $\tau_{10} < \tau_{01}$ ,  $\tau_{10} = \tau_{01}$ , or  $\tau_{10} > \tau_{01}$ . Let  $u$  be  $-1, 0$  or  $+1$  corresponding

to which one of these three conditions holds. Based on different  $u$ ,  $\widetilde{y}$  and  $f(\boldsymbol{x})$ , we choose different  $\Delta$  as in Table 1. The label correction algorithm is given in Procedure 1. It checks the likelihood ratio with regard to the chosen  $\Delta$ . If LR is no greater than  $\Delta$ , we flip the label. Otherwise,  $\widetilde{y}_{\text{new}}$  is the same as  $\widetilde{y}$ . In practice,  $\Delta$  is unknown and needs to be decided empirically.

# Procedure 1 LRT-Correction

Input:  $f, (\pmb{x}, \widetilde{y}), u, a, b$ .

Output:  $\widetilde{y}_{\text {new }}$

1: if  $\mathrm{LR}(f, \pmb{x}, \widetilde{y}) \leq \Delta(f, \widetilde{y}, u, a, b)$ , where  $\Delta$  is as in Table 1 then

2:  $\widetilde{y}_{new} = 1 - \widetilde{y}$  
3: else  
4:  $\widetilde{y}_{new} = \widetilde{y}$  
5: end if

Table 1: Values of  $\Delta$  

<table><tr><td>y</td><td>f &lt; b + a/2</td><td>f &gt; b + a/2</td></tr><tr><td>y = 1</td><td>a+2b/2-a-2b</td><td>0</td></tr><tr><td>y = 0</td><td>0</td><td>2-a-2b/a+2b</td></tr></table>

Our main theorem states that suppose the classifier prediction,  $f$ , is a close approximation of the noisy conditional probability,  $\widetilde{\eta}$ . And suppose we can find a good enough  $\Delta'$  that is close enough to the ideal  $\Delta$ . Then there is a very good chance that our algorithm corrects most labels to the correct ones, i.e., the same as the Bayes optimal classifier prediction. Please note that here "proper correction" means that the new label,  $\widetilde{y}_{new}$ , is the same as the Bayes optimal classifier prediction,  $h^*(x)$ , instead of  $y$ . This is well justified as it means that the correction will give us a classifier as good as the Bayes optimal one.

Theorem 1. Assume  $\eta$  satisfies the Tsybakov condition with constants  $C > 0$  and  $\lambda > 0$ . Recall  $h^*$  denote the Bayes optimal classifier. The noisy conditional probability  $\widetilde{\eta}(\boldsymbol{x}) = (1 - \tau_{01} - \tau_{10})\eta(\boldsymbol{x}) + \tau_{01}$ . Assume  $f(\boldsymbol{x}) = a\eta(\boldsymbol{x}) + b$  with  $a$  and  $b$  unknown, such that  $f(\boldsymbol{x}) \in [\widetilde{\eta}(\boldsymbol{x}) - \epsilon, \widetilde{\eta}(\boldsymbol{x}) + \epsilon]$  for some  $\epsilon > 0$ . Let  $\Delta(\widetilde{y}, f(x), u, a, b)$  be as in Table 1. Let  $\Delta' > 0$  be a constant such that  $\Delta' \in [\Delta - \epsilon, \Delta + \epsilon]$ . If  $\widetilde{y}_{new}$  denotes the output of the LRT-Correction with  $\widetilde{y}$ ,  $\boldsymbol{x}$ ,  $f$ , and the give  $\Delta'$ , then

$$
\operatorname * {P r} _ {(x, y) \sim D} (\widetilde {y} _ {n e w} \neq h ^ {*}) \leq 8 C \left(\left| \frac {\tau_ {1 0} - \tau_ {0 1}}{2 (1 - \tau_ {1 0} - \tau_ {0 1})} \right| + O (\epsilon)\right) ^ {\lambda}.
$$

If  $\tau_{01} = \tau_{10}$ , then  $\operatorname{Pr}_{(x,y) \sim D}(\widetilde{y}_{\text{new}} \neq h^*) = 8C(O(\epsilon))^{\lambda}$ .

Remark 2. The condition that  $f$  is linear to  $\eta$  is not necessary for the theorem to hold. We only require  $f$  to be pointwise close to  $\widetilde{\eta}$ .

Intuition of the proof. We will prove two lemmas. The first lemma shows that when strictly assuming  $f$  is linear to  $\eta$ , with known coefficients  $a, b$ , and we set  $\Delta$  according to Table 1, then the correction algorithm can be correct everywhere. The second lemma proves a more relaxed version. If  $f$  is exactly  $\widetilde{\eta}$  and if we only know the difference between the transition probabilities,  $\tau_{01} - \tau_{10}$ , then we can bound the chance of mistakes of the correction algorithm. Finally, based on these two lemmas, careful case analysis, and the Tsybakov condition of the true conditional probability,  $\eta$ , we can prove the theorem. The complete proof can be found in Appendix A.

So far, all the description and theoretical results are based on a binary classification setting. However, the results can be generalized to a multiclass setting without any technical difficulties. Informally, we state the following corollary (proof omitted).

Corollary 1. LRT-Correction can be generalized to multiclass classification tasks, by flipping  $\widetilde{y}$  to the best prediction of  $f$  when the null hypothesis is rejected. Theorem 1 can be generalized to multiclass classification tasks, by considering all pairs of class values.

# 2.3 TRAINING DEEP NEURAL NETWORKS WITH LRT-LABEL-CORRECTION

Our training algorithm continuously trains a deep neural network while correcting the noisy labels. Procedure 2 is the pseudocode of the training method, called AdaCorr. It trains a neural network model iteratively. Each iteration includes both label correction and model training steps. In label correction step, the prediction of the current neural network,  $f$ , is used to run LRT test on all training data, and to correct their labels according to the test result. Since  $f$  is used to approximate the conditional probability  $\widetilde{\eta}$ , we use the softmax layer output of the neural network as  $f$ . After the labels of all training data are updated, we use them to train the neural network incrementally. We continue this iterative procedure until the whole training converges.

# Procedure 2 AdaCorr

Input:  $S = \{x,\widetilde{y}\} ,\Delta ,m,T$    
1: for epoch  $= 1$  to m do   
2: Train neural network with  $L_{CE}$    
3: end for   
4:  $f^{\prime} =$  current model prediction   
5: for epoch  $= m + 1$  to  $T$  do   
6: if epoch  $\geq m + 10$  then   
7:  $f =$  current model prediction   
8: for all  $(x,\widetilde{y})\in S$  do   
9:  $\widetilde{y}_{new} = \mathrm{LRT - Correction}(f,(x,\widetilde{y}),\Delta)$    
10:  $\widetilde{y} = \widetilde{y}_{new}$    
11: end for   
12: end if   
13: Train using  $L_{retro} + L_{CE}$  ,with  $f^{\prime}$  and  $\widetilde{y}$    
14: end for

We also have a burn-in stage in which we

train the network using the original noisy labels for  $m$  epochs. During the burn-in stage, we use the original cross-entropy loss,  $L_{CE}$ . Afterwards, we add an additional retroactive loss which will be explained below.

Training with retroactive loss. After the burn-in stage, we want to avoid overfitting of the neural network, so that its output better approximates  $\widetilde{\eta}$ . To achieve this goal, we introduce a retroactive loss term  $L_{\text{retro}}(f(\boldsymbol{x}), \widetilde{y})$ . The idea is to enforce the consistency between  $f$  and the prediction of the model at a previous epoch,  $f'$ . It has been observed that a neural network at earlier training stage tends to learn the true pattern rather than to overfit the noise (Arpit et al., 2017a). Formally, the loss can be written as  $\sum_{c=1}^{N_c} f_c'(\boldsymbol{x}) \log f_c(\boldsymbol{x})$ , in which  $N_c$  is the number of possible label classes. The training loss is the sum of the retroactive loss and the cross-entropy loss

$$
L (f (\pmb {x}), \widetilde {y}) = L _ {r e t r o} (f (\pmb {x}), \widetilde {y}) + L _ {C E} (f (\pmb {x}), \widetilde {y}) = \sum_ {c = 1} ^ {N _ {c}} f _ {c} ^ {\prime} (\pmb {x}) \log f _ {c} (\pmb {x}) + \sum_ {c = 1} ^ {N _ {c}} \widetilde {y} _ {c} \log f _ {c} (\pmb {x}).
$$

In practice, we set  $f'$  to be the prediction of the model at the  $m$ -th epoch. In other words, once the burn-in stage is finished, the training switches from  $L_{CE}$  to  $L_{CE} + L_{retro}$ . And the model at the end of the burn-in stage is used for the retroactive loss. We also set the label correction to start slightly after the burn-in stage, say  $m + 10$ . The key hyperparameter is the starting epoch  $m$ . Another hyperparameter is  $\Delta$ . We select both  $m$  and  $\Delta$  empirically. Ablation study in Section 4 shows that our method is robust to these hyperparameters.

# 3 EXPERIMENTS

In this section we empirically evaluate our proposed method with several datasets, where noisy labels are injected according to specified noise transition matrices.

Datasets. We use the following datasets: MNIST (LeCun & Cortes 2010), CIFAR10 (Krizhevsky et al. a), CIFAR100 (Krizhevsky et al. b) and ModelNet40 (Z. Wu & Xiao 2015). MNIST consists of  $28 \times 28$  grayscale images with 10 categories. It contains 60,000 images, and we use 45,000 for training, 5,000 for validation and 10,000 for testing. CIFAR10 and CIFAR100 share the same 60,000  $32 \times 32 \times 3$  image data, with CIFAR10 having 10 categories while CIFAR100 having 100 categories. Similar to MNIST, we split  $90\%$  and  $10\%$  data from the official training set for the training and validation respectively, and use the official test set for testing. ModelNet40 contains 12,311 CAD models from 40 categories, where 8,859 are used for training, 984 for validation and the remaining 2,468 for testing. We follow the protocol of (Qi et al. (2017)) to convert the CAD models into point clouds by uniformly sampling 1,024 points from the triangular mesh and normalizing them within a

unit ball. In all experiments, we use early stopping on validation set to tune hyperparameters and to report the performance on test set.

Noise patterns. Following (Reed et al. 2014; Patrini et al. 2017), we corrupt our labels artificially using a noise transition matrix  $T$ , where  $T_{ij} = \tau_{ij} = \operatorname*{Pr}(\widetilde{y} = j|y = i)$  is the probability that category  $i$  is flipped to category  $j$ . In our work we focus on two types of  $T$ : (1) uniform, where the true label  $i$  is flipped to other classes with equal probabilities, i.e.,  $T_{ij} = p / (N_c - 1)$  for  $i \neq j$  and  $T_{ii} = 1 - p$ , where  $p$  is the noise level and  $N_{c}$  is the class number; (2) pair flipping, where the true label  $i$  is flipped to  $j$  with  $T_{ij} = p$  for  $i \neq j$  and  $T_{ii} = 1 - p$ . Examples of these two types of transition matrices are as follows:

$$
T _ {1} = \left( \begin{array}{l l l l} 0. 7 & 0. 1 & 0. 1 & 0. 1 \\ 0. 1 & 0. 7 & 0. 1 & 0. 1 \\ 0. 1 & 0. 1 & 0. 7 & 0. 1 \\ 0. 1 & 0. 1 & 0. 1 & 0. 7 \end{array} \right) \qquad \qquad T _ {2} = \left( \begin{array}{l l l l} 0. 7 & 0. 3 & 0. 0 & 0. 0 \\ 0. 0 & 0. 7 & 0. 3 & 0. 0 \\ 0. 0 & 0. 0 & 0. 7 & 0. 3 \\ 0. 3 & 0. 0 & 0. 0 & 0. 7 \end{array} \right)
$$

in which  $T_{1}$  is for uniform noise pattern with noise level 0.3 and  $T_{2}$  is for pair flipping with noise level 0.3.

Baselines. We compare the proposed method with the following methods: (1) Standard, which trains the network in a standard manner, without any label resistance technique; (2) Forward correction (Patrini et al. 2017), which explicitly estimates the noise transition matrix to correct the training loss; (3) Decoupling (Malach & Shalev-Shwartz 2017), which trains two networks simultaneously and updates the parameters on selected data whose labels are possibly clean; (4) Coteaching (Han et al. 2018), which also trains two networks but exchanges their error information for network update; (5) MentorNet (Jiang et al. 2018), which learns a curriculum to filter out noisy data; (6) Forgetting (Arpit et al., 2017b), which uses dropout to help deep models resist label noise.

Experimental Setup. For the classification of MNIST, CIFAR10 and CIFAR100, we use preactive resnet34 (He et al. 2016) as the backbone for all the methods. On ModelNet40, we use PointNet (Qi et al. 2017). We train the models for 180 epochs to ensure that all the methods have converged. We utilize RAdam (Liu et al. 2019) for the network optimization, and adopt batch size 128 for all the datasets. The experimental results are listed in Table 2. As is shown, our method outperforms the competing methods across the datasets under different noise settings.

Table 2: The classification accuracies of different methods.  

<table><tr><td rowspan="2">Data Set</td><td rowspan="2">Method</td><td colspan="4">Noise Level of Uniform Flipping</td><td colspan="3">Noise Level of Pair Flipping</td></tr><tr><td>0.2</td><td>0.4</td><td>0.6</td><td>0.8</td><td>0.2</td><td>0.3</td><td>0.4</td></tr><tr><td rowspan="7">MINIST</td><td>Standard</td><td>99.0 ± 0.2</td><td>98.7 ± 0.4</td><td>98.1 ± 0.3</td><td>91.3 ± 0.9</td><td>99.3 ± 0.1</td><td>99.2 ± 0.1</td><td>98.8 ± 0.1</td></tr><tr><td>Forget</td><td>99.0 ± 0.1</td><td>98.8 ± 0.1</td><td>97.7 ± 0.2</td><td>62.6 ± 8.9</td><td>99.3 ± 0.1</td><td>96.5 ± 2.0</td><td>89.7 ± 1.9</td></tr><tr><td>Forward</td><td>99.1 ± 0.1</td><td>98.7 ± 0.2</td><td>98.0 ± 0.4</td><td>89.6 ± 4.8</td><td>99.4 ± 0.0</td><td>99.2 ± 0.2</td><td>96.5 ± 4.4</td></tr><tr><td>Decouple</td><td>99.3 ± 0.1</td><td>99.0 ± 0.1</td><td>98.5 ± 0.2</td><td>94.6 ± 0.2</td><td>99.4 ± 0.0</td><td>99.3 ± 0.1</td><td>99.1 ± 0.2</td></tr><tr><td>MentorNet</td><td>99.2 ± 0.2</td><td>98.7 ± 0.1</td><td>98.1 ± 0.1</td><td>87.5 ± 5.2</td><td>98.6 ± 0.4</td><td>99.1 ± 0.1</td><td>98.9 ± 0.1</td></tr><tr><td>Coteach</td><td>99.1 ± 0.2</td><td>98.7 ± 0.3</td><td>98.2 ± 0.3</td><td>95.7 ± 0.7</td><td>99.1 ± 0.1</td><td>99.0 ± 0.2</td><td>98.9 ± 0.2</td></tr><tr><td>AdaCorr</td><td>99.5 ± 0.0</td><td>99.4 ± 0.0</td><td>99.1 ± 0.0</td><td>97.7 ± 0.2</td><td>99.5 ± 0.0</td><td>99.6 ± 0.0</td><td>99.4 ± 0.0</td></tr><tr><td rowspan="7">CIFAR10</td><td>Standard</td><td>87.5 ± 0.2</td><td>83.1 ± 0.4</td><td>76.4 ± 0.4</td><td>47.6 ± 2.0</td><td>88.8 ± 0.2</td><td>88.4 ± 0.3</td><td>84.5 ± 0.3</td></tr><tr><td>Forget</td><td>87.1 ± 0.2</td><td>83.4 ± 0.2</td><td>76.5 ± 0.7</td><td>33.0 ± 1.6</td><td>89.6 ± 0.1</td><td>83.7 ± 0.1</td><td>86.4 ± 0.5</td></tr><tr><td>Forward</td><td>87.4 ± 0.8</td><td>83.1 ± 0.8</td><td>74.7 ± 1.7</td><td>38.3 ± 3.0</td><td>89.0 ± 0.5</td><td>87.4 ± 1.1</td><td>84.7 ± 0.5</td></tr><tr><td>Decouple</td><td>87.6 ± 0.4</td><td>84.2 ± 0.5</td><td>77.6 ± 0.1</td><td>48.5 ± 0.9</td><td>90.6 ± 0.3</td><td>89.1 ± 0.3</td><td>86.3 ± 0.5</td></tr><tr><td>MentorNet</td><td>90.3 ± 0.3</td><td>83.2 ± 0.5</td><td>75.5 ± 0.7</td><td>34.1 ± 2.5</td><td>90.4 ± 0.2</td><td>88.9 ± 0.1</td><td>83.3 ± 1.0</td></tr><tr><td>Coteach</td><td>90.1 ± 0.4</td><td>87.3 ± 0.5</td><td>80.9 ± 0.5</td><td>25.0 ± 3.6</td><td>91.8 ± 0.1</td><td>89.9 ± 0.2</td><td>80.1 ± 0.7</td></tr><tr><td>AdaCorr</td><td>91.0 ± 0.3</td><td>88.7 ± 0.5</td><td>81.2 ± 0.4</td><td>49.2 ± 2.4</td><td>92.2 ± 0.1</td><td>91.3 ± 0.3</td><td>89.2 ± 0.4</td></tr><tr><td rowspan="7">CIFAR100</td><td>Standard</td><td>58.9 ± 0.8</td><td>52.1 ± 1.0</td><td>42.1 ± 0.7</td><td>20.8 ± 1.0</td><td>59.5 ± 0.4</td><td>52.9 ± 0.6</td><td>44.7 ± 1.3</td></tr><tr><td>Forget</td><td>59.3 ± 0.8</td><td>53.0 ± 0.2</td><td>40.9 ± 0.5</td><td>7.7 ± 1.1</td><td>61.4 ± 0.9</td><td>54.6 ± 0.6</td><td>37.7 ± 4.6</td></tr><tr><td>Forward</td><td>58.4 ± 0.5</td><td>52.2 ± 0.3</td><td>41.1 ± 0.5</td><td>20.6 ± 0.6</td><td>58.3 ± 0.7</td><td>53.2 ± 0.6</td><td>44.4 ± 2.8</td></tr><tr><td>Decouple</td><td>59.0 ± 0.7</td><td>52.2 ± 0.7</td><td>40.2 ± 0.4</td><td>18.5 ± 0.8</td><td>60.8 ± 0.7</td><td>56.1 ± 0.7</td><td>48.4 ± 1.0</td></tr><tr><td>MentorNet</td><td>63.6 ± 0.5</td><td>51.4 ± 1.4</td><td>38.7 ± 0.8</td><td>17.4 ± 0.9</td><td>64.7 ± 0.2</td><td>57.4 ± 0.8</td><td>47.4 ± 1.7</td></tr><tr><td>Coteach</td><td>66.1 ± 0.5</td><td>60.0 ± 0.6</td><td>48.3 ± 0.1</td><td>16.1 ± 1.1</td><td>63.4 ± 0.9</td><td>57.6 ± 0.3</td><td>49.2 ± 0.3</td></tr><tr><td>AdaCorr</td><td>67.8 ± 0.1</td><td>60.2 ± 0.8</td><td>46.5 ± 1.2</td><td>24.6 ± 1.1</td><td>68.3 ± 0.2</td><td>61.1 ± 0.5</td><td>49.8 ± 0.7</td></tr><tr><td rowspan="7">ModelNet40</td><td>Standard</td><td>79.1 ± 2.6</td><td>75.3 ± 3.3</td><td>70.0 ± 3.0</td><td>57.9 ± 2.3</td><td>84.4 ± 1.2</td><td>82.3 ± 1.3</td><td>78.9 ± 0.7</td></tr><tr><td>Forget</td><td>80.1 ± 1.8</td><td>73.9 ± 0.6</td><td>69.0 ± 0.7</td><td>26.2 ± 4.8</td><td>83.3 ± 1.1</td><td>62.0 ± 3.0</td><td>59.5 ± 2.9</td></tr><tr><td>Forward</td><td>52.3 ± 5.1</td><td>49.4 ± 6.8</td><td>43.5 ± 5.2</td><td>28.2 ± 5.5</td><td>48.1 ± 6.8</td><td>48.0 ± 3.7</td><td>49.1 ± 4.4</td></tr><tr><td>Decouple</td><td>82.5 ± 2.2</td><td>80.7 ± 0.7</td><td>72.9 ± 1.0</td><td>55.4 ± 2.7</td><td>85.7 ± 1.4</td><td>84.3 ± 1.0</td><td>80.5 ± 2.4</td></tr><tr><td>MentorNet</td><td>86.5 ± 0.5</td><td>75.4 ± 1.8</td><td>70.9 ± 1.9</td><td>52.7 ± 3.1</td><td>83.7 ± 1.8</td><td>81.0 ± 1.5</td><td>79.3 ± 2.1</td></tr><tr><td>Coteach</td><td>85.6 ± 0.9</td><td>84.2 ± 0.8</td><td>81.8 ± 1.1</td><td>68.9 ± 2.8</td><td>85.7 ± 0.8</td><td>79.1 ± 3.0</td><td>69.1 ± 2.4</td></tr><tr><td>AdaCorr</td><td>86.9 ± 0.3</td><td>85.1 ± 0.6</td><td>78.6 ± 1.4</td><td>72.1 ± 1.1</td><td>87.6 ± 0.4</td><td>84.6 ± 0.5</td><td>83.7 ± 0.5</td></tr></table>

# 4 ABLATION STUDY AND DISCUSSION

We conduct ablation study to see the significance of our contributions. We compare our method  $(\mathrm{LRT} + L_{ce} + L_{retro})$  with two baselines: our method without the retroactive loss  $(\mathrm{LRT} + L_{ce})$  and using cross-entropy loss only without LRT-Correction  $(L_{ce}$  Only). We report the test accuracy on CIFAR10 in Table 3. We observe that adding LRT Label Correction to  $L_{ce}$  alone helps improve the performance significantly. The numbers inside the parenthesis are the percentage of correct labels after the training and label correction. We observe that the LRT Correction corrected a large portion of noise labels. Also we observe adding the retroactive loss improves the method further in both test accuracy and label correction rates.

Table 3: Effect of LRT Correction and  ${L}_{\text{retro }}$  . The experiments are performed on CIFAR10 (accuracy in %). The number in parenthesis denotes the rate of correct labels after training.  

<table><tr><td>Method</td><td>uniform 0.2</td><td>uniform 0.4</td><td>uniform 0.6</td><td>uniform 0.8</td><td>pair 0.4</td></tr><tr><td>LceOnly</td><td>87.5(80.0)</td><td>83.1(60.0)</td><td>76.4(40.0)</td><td>47.6(20.0)</td><td>84.5(60.0)</td></tr><tr><td>LRT+Lce</td><td>91.1(95.7)</td><td>87.9(91.2)</td><td>80.7(81.7)</td><td>47.3(47.1)</td><td>87.5(91.9)</td></tr><tr><td>LRT+Lce+Lretro</td><td>91.0(95.7)</td><td>88.7(90.5)</td><td>81.2(82.5)</td><td>49.2(49.0)</td><td>89.2(91.3)</td></tr></table>

We evaluate how different hyperparameters affect the performance of our method. We compare our method with different  $m$ , the length of the burn-in stage. We start introducing the retroactive loss after  $m$  epochs, and start label corrections after  $m + 10$  epochs. The final testing accuracies are shown in Table 4. We observe the performance of our method is rather robust w.r.t. different  $m$ 's. We choose  $m = 20$  in this data set (CIFAR10) and similarly in other datasets.

Table 4: Effect of different  $m$  ’s. The experiments are performed on CIFAR10 (accuracy in %). The number in parenthesis denotes the rate of correct labels after training.  

<table><tr><td>Noisy Type</td><td>Epoch 15</td><td>Epoch 20</td><td>Epoch 25</td><td>Epoch 30</td><td>Epoch 35</td><td>Epoch 40</td></tr><tr><td>uniform 0.4</td><td>87.6(90.1)</td><td>88.7(90.5)</td><td>87.4(90.7)</td><td>86.7(90.6)</td><td>84.8(88.7)</td><td>84.1(87.2)</td></tr><tr><td>uniform 0.6</td><td>79.4(81.0)</td><td>81.2(82.5)</td><td>80.9(81.9)</td><td>79.3(81.9)</td><td>79.1(81.8)</td><td>78.1(81.7)</td></tr><tr><td>pair 0.4</td><td>75.8(80.0)</td><td>89.2(80.1)</td><td>90.8(87.0)</td><td>89.2(89.3)</td><td>88.2(90.1)</td><td>86.7(90.1)</td></tr></table>

We also evaluate the performance on different  $\Delta$ ’s.  $\Delta$  is the unknown value for our likelihood ratio test. It controls how aggressive we would like to correct the labels. From Table 5, we observe a bigger  $\Delta$  tends to give better results (as it is less aggressive in correcting labels). We observe  $1/1.2$  is the best one for CIFAR10. Similar values of optimal  $\Delta$  are found in other data sets.

Table 5: Effect of  $\Delta$  . The experiments are performed on CIFAR10, and the number in parenthesis denotes the rate of correct labels after flipping.  

<table><tr><td>Noisy Type</td><td>1/1.0</td><td>1/1.2</td><td>1/1.5</td><td>1/2.0</td><td>1/2.5</td><td>1/3.0</td></tr><tr><td>uniform 0.4</td><td>88.3(91.7)</td><td>88.7(90.5)</td><td>83.0(89.3)</td><td>77.1(86.7)</td><td>75.2(85.1)</td><td>75.5(84.0)</td></tr><tr><td>uniform 0.6</td><td>79.9(81.0)</td><td>81.2(82.5)</td><td>80.9(81.9)</td><td>79.3(81.9)</td><td>79.1(81.8)</td><td>78.1(81.7)</td></tr><tr><td>pair 0.4</td><td>88.2(92.2)</td><td>89.2(80.1)</td><td>84.4(89.8)</td><td>77.0(86.8)</td><td>76.4(85.1)</td><td>77.0(84.1)</td></tr></table>

In general, we observe our hyperparameters are rather consistent across different datasets. This reveals a better generalization power of our method over other datasets and noise patterns. We believe this is due to the principled approach we take in label cleaning.

# 5 CONCLUSION

We propose a label correction algorithm to combat label noise. We perform a likelihood ratio test for each input label such that if it is rejected, this label is flipped to the class that has the highest likelihood. Theoretically, we prove that our method corrects noisy labels with high probability. Experiments on various datasets show that our method outperforms state-of-the-arts and is robust to hyperparameters.

# REFERENCES

Devansh Arpit, Stanisław Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxin-der S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, and Simon Lacoste-Julien. A closer look at memorization in deep networks. In ICML, pp. 233–242, 2017a.  
Devansh Arpit, Stanislaw K. Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron C. Courville, Yoshua Bengio, and Simon Lacoste-Julien. A closer look at memorization in deep networks. In ICML, pp. 233-242, 2017b.  
Carla E. Brodley and Mark A. Friedl. Identifying mislabeled training data. J. Artif. Int. Res., 11(1): 131-167, July 1999. ISSN 1076-9757.  
Koby Crammer and Daniel D. Lee. Learning via gaussian herding. In NIPS, pp. 451-459. 2010.  
Koby Crammer, Alex Kulesza, and Mark Dredze. Adaptive regularization of weight vectors. In NIPS, 2009.  
Benot Frnay and Michel Verleysen. Classification in the presence of label noise: A survey. Neural Networks and Learning Systems, IEEE Transactions on, 25:845-869, 05 2014.  
Wei Gao, Bin bin Yang, and Zuo cheng Zhou. On the resistance of nearest neighbor to random noisy labels. 2016.  
Aritra Ghosh, Naresh Manwani, and P.S. Sastry. Making risk minimization tolerant to label noise. Neurocomput., 160:93-107, July 2015.  
Jacob Goldberger and Ehud Ben-Reuven. Training deep neural-networks using a noise adaptation layer. In ICLR, 2017.  
Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor W. Tsang, and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In NeurIPS, pp. 8536-8546, 2018.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In European conference on computer vision, pp. 630-645, 2016.  
Dan Hendrycks, Kimin Lee, and Mantas Mazeika. Using pre-training can improve model robustness and uncertainty. In ICML, pp. 2712-2721, 2019.  
Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In ICML, 2018.  
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10 (canadian institute for advanced research). a. URL http://www.cs.toronto.edu/~kriz/cifar.html.  
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-100 (canadian institute for advanced research). b. URL http://www.cs.toronto.edu/~kriz/cifar.html.  
Yann LeCun and Corinna Cortes. MNIST handwritten digit database. 2010. URL http://yann.lecun.com/exdb/mnist/.  
Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. On the variance of the adaptive learning rate and beyond. arXiv preprint arXiv:1908.03265, 2019.  
Philip M. Long and Rocco A. Servedio. Random classification noise defeats all convex potential boosters. Machine Learning, 78(3):287-304, Mar 2010.  
Eran Malach and Shai Shalev-Shwartz. Decoupling "when to update" from "how to update". In NeurIPS, 2017.  
Volodymyr Mnih and Geoffrey E. Hinton. Learning to label aerial images from noisy data. In ICML, 2012.

Nagarajan Natarajan, Ambuj Tewari, Inderjit S. Dhillon, and Pradeep Ravikumar. Learning with noisy labels. In NeurIPS, 2013.  
Giorgio Patrini, Alessandro Rozza, Aditya Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: A loss correction approach. In CVPR, pp. 2233-2241, 2017.  
Charles Ruizhongtai Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In CVPR, pp. 77-85, 2017.  
Scott E. Reed, Honglak Lee, Dragomir Anguelov, Christian Szegedy, Dumitru Erhan, and Andrew Rabinovich. Training deep neural networks on noisy labels with bootstrapping. CoRR, abs/1412.6596, 2014.  
Padhraic Smyth, Usama Fayyad, Michael Burl, Pietro Perona, and Pierre Baldi. Inferring ground truth from subjective labelling of venus images. pp. 1085-1092, 1994.  
Jacob Steinhardt, Pang Wei Koh, and Percy S. Liang. Certified defenses for data poisoning attacks. In NIPS, pp. 3520-3532, 2017.  
Daiki Tanaka, Daiki Ikami, Toshihiko Yamasaki, and Kiyoharu Aizawa. Joint optimization framework for learning with noisy labels. In CVPR, 2018.  
Alexander B Tsybakov et al. Optimal aggregation of classifiers in statistical learning. The Annals of Statistics, 32(1):135-166, 2004.  
Brendan van Rooyen, Aditya Menon, and Robert C Williamson. Learning with symmetric label noise: The importance of being unhinged. In NeurIPS, pp. 10-18. 2015.  
Andreas Veit, Neil Alldrin, Gal Chechik, Ivan Krasin, Abhinav Gupta, and Serge J. Belongie. Learning from noisy large-scale datasets with minimal supervision. In CVPR, pp. 6575-6583, 2017.  
Jha Wang and Chaudhuri. Analyzing the robustness of nearest neighbors to adversarial examples. In ICML, pp. 5133-5142, 2018.  
Xiang Wu, Ran He, Zhenan Sun, and Tieniu Tan. A light CNN for deep face representation with noisy labels. IEEE Trans. Information Forensics and Security, 13(11):2884-2896, 2018.  
Yan Yan, Rómer Rosales, Glenn Fung, Ramanathan Subramanian, and Jennifer Dy. Learning from multiple annotators with varying expertise. *Machine learning*, 95(3):291–327, 2014.  
A. Khosla F. Yu L. Zhang X. Tang Z. Wu, S. Song and J. Xiao. 3d shapenets: A deep representation for volumetric shape modeling. In CVPR, pp. 1912-1920, 2015.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In ICLR, 2017.
