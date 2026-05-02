# A Geometric Perspective towards Neural Calibration via Sensitivity Decomposition

Anonymous Author(s)

Affiliation

Address

email

# Abstract

It is well known that vision classification models suffer from poor calibration in the face of data distribution shifts. In this paper, we take a geometric approach to this problem. We propose Geometric Sensitivity Decomposition (GSD) which decomposes the norm of a sample feature embedding and the angular similarity to a target classifier into an instance-dependent and an instance-independent component. The instance-dependent component captures the sensitive information about changes in the input while the instance-independent component represents the insensitive information serving solely to minimize the loss on the training dataset. Inspired by the decomposition, we analytically derive a simple extension to current softmax-linear models, which learns to disentangle the two components during training. On several common vision models, the disentangled model outperforms other calibration methods on standard calibration metrics in the face of out-of-distribution (OOD) data and corruption with significantly less complexity. Specifically, we surpass the current state of the art by  $30.8\%$  relative improvement on corrupted CIFAR100 in Expected Calibration Error.

# 1 Introduction

During development, deep learning models are trained and validated on data from the same distribution. However, in the real world sensors degrade and weather conditions change. Similarly, subtle changes in image acquisition and processing can also lead to distribution shift of the input data. This is often known as covariate shift, and will typically decrease the performance (e.g. classification accuracy). However, it has been empirically found that the model's confidence remains high even when accuracy has degraded [1]. The process of aligning confidence to empirical accuracy is called model calibration. Calibrated probability provides valuable uncertainty information for decision making. For example, knowing when a decision cannot be trusted and more data is needed is important for safety and efficiency in real world applications such as self-driving [2] and active learning [3].

A comprehensive comparison of calibration methods has been studied for in-distribution (IND) data [4], However, these methods lead to unsatisfactory performance under distribution shift [5]. To resolve the problem, high-quality uncertainty estimation [6, 5] is required. Principled Bayesian methods [7] model uncertainty directly but are computationally heavy. Recent deterministic methods [8, 9] propose to improve a model's sensitivity to input changes by regularizing the model's intermediate layers. In this context, sensitivity is defined as preserving distance between two different input samples through layers of the model. We would like to utilize the improved sensitivity to better detect Out-of-Distribution (OOD) data. However, these methods introduce added architecture changes and large combinatorics of hyperparameters.

Unlike existing works, we propose to study sensitivity from a geometric perspective. The last linear layer in a softmax-linear model can be decomposed into the multiplication of a norm and a cosine

similarity term [10, 11, 12, 13]. Geometrically, the angular similarity dictates the membership of an input and the norm only affects the confidence in a softmax-linear model. Counter-intuitively, the norm of a sample's feature embedding exhibits little correlation to the hardness of the input [11]. Based on this observation, we explore two questions: 1) why is a model's confidence insensitive to distribution shift? 2) how do we improve model sensitivity and calibration?

We hypothesize that in part an insensitive norm is responsible for bad calibration especially on shifted data. We observe that the sensitivity of the angular similarity increases with training whereas the sensitivity of the norm remains low. More importantly, calibration worsens during the period when the norm increases while the angular similarity changes slowly. This shows a concrete example of the inability of the norm to adapt when accuracy has dropped. Intuitively, training on clean datasets encourages neural networks to always output increasingly large feature norm to continuously minimize the training loss. Because the probability of the prevalent class of an input is proportional to its norm, larger norms lead to smaller training loss when most training data have been classified correctly (See Sec. 3.1). This renders the norm insensitive to input differences because the model is trained to always output features with large norm on clean data. While we have put forth that the norm is poorly calibrated, we must emphasize that it can still play an important role in model calibration (See Sec. 4.1).

To encourage sensitivity, we propose to decompose the norm of a sample's feature embedding and the angular similarity into two components: instance-dependent and instance-independent. The instance-dependent component captures the sensitive information about the input while the instance-independent component represents the insensitive information serving solely to minimize the loss on the training dataset. Inspired by the decomposition, we analytically derive a simple extension to the current softmax-linear model, which learns to disentangle the two components during training. We show that our model outperforms other deterministic methods (despite their significant complexity) and is comparable or better to multi-pass methods with fewer training hyperparameters in Sec. 4.1.

In summary, our contributions are four fold:

- We study the problem of calibration geometrically and identify that the insensitive norm is responsible for bad calibration under distribution shift.  
- We derive a principled but simple geometric decomposition that decomposes the norm into an instance-dependent and instance-independent component.  
- Based on the decomposition, we propose a simple training and inference scheme to encourage the norm to reflect distribution changes.  
- We achieve state of the art results in calibration metrics in the face of corruptions while having arguably the simplest calibration method to implement.

# 2 Related Work

Methods dedicated to strengthening calibration can be divided into two camps: multi-pass models and single-pass deterministic models. The current state-of-the-art multi-pass models are: Bayesian Monte Carlo Drop Out (MCDO) [7] and Deep Ensembles [14]. Bayesian methods are the most principled way to model uncertainty. Instead of optimizing max likelihood for a single set of parameters, Bayesian methods obtain a posterior distribution over possible parameters given a prior distribution over parameters and calculated data likelihood assuming some process noise. The posterior distribution over parameters captures epistemic uncertainty or uncertainty due to the limits of what the model knows. The final predictive distribution is obtained by marginalizing out model parameters. While Bayesian methods are theoretically sound, they are intractable in practice. Deep Ensembles leverage multiple models trained using different random initialization of weights so models learn different classification functions, and these variations then ensemble by averaging their predicted probabilities.

A recent trend is to use a single-pass deterministic non-Bayesian model to improve uncertainty estimation. Two recent works DUQ [8] and SNGP [9] propose to improve uncertainty-awarenesss of deterministic networks by improving the networks' sensitivity to input changes. Intuitively, a sensitive model should map samples further from the training data as they become more out-of-distribution. This can be achieved at two levels: feature level and output level. At the feature level,

both methods require the feature extractors (CNNs) to be regularized to prevent feature collapse, which is the mapping of two different data points to the same embedded vector. This is ensured by having input distance awareness, which is equivalent to ensuring bi-Lipschitz continuity over layers of the model [15]. In order to achieve this, DUQ [8] uses a two-sided gradient penalty [16] and SNGP [9] uses bounded spectral normalization [15]. The output level needs to reflect the changes in feature space. This can be done by adopting distance-aware classifiers. DUQ [8] uses a RBF networks with learned centroids for each class and SNGP [9] uses an approximate Gaussian Process layer. We were inspired by temperature scaling [4], which is another method for bettering calibration, but fails under distribution shift [5]. Our method does not require input distance awareness and instead leverages the geometric intuitions about the output layer, specifically properties of the norm of the input embedding, in order to strengthen calibration.

# 3 Method

Following our hypothesise that the insensitivity of the norm is responsible for bad calibration on distribution shifted data, we propose geometric sensitivity decomposition (GSD) for the norm. We first introduce the geometric perspective of the last linear layer in Sec. 3.1 and then derive GSD in Sec. 3.2. To improve sensitivity of the norm and model calibration on shifted data, we propose a GSD-inspired training and inference procedure in Sec. 3.3 and Sec. 3.4.

# 3.1 Norm and Similarity

The output layer of a neural network can be written as a dot-product  $< \mathbf{x}, \mathbf{w}_{\mathbf{y}} >$ , where  $\mathbf{x}$  is the embedded input and  $\mathbf{w}_{\mathbf{y}}$  is the weight vector associated with class  $y$ . Though seemingly simple there are strong geometric and calibration related intuitions drawn from this. Several prior works [10, 12, 11] have studied the effects decomposition of the last linear layer in a softmax model can have on classification. The output layer can be decomposed into angular similarity  $\cos \phi_{y}$  and norm  $\| \mathbf{x} \|_2$ .

$$
P (y | x) = \frac {\exp l _ {y}}{\sum_ {j = 1} ^ {c} \exp l _ {j}} = \frac {\exp \left(\| \mathbf {w} _ {\mathbf {y}} \| _ {2} \| \mathbf {x} \| _ {2} \cos \phi_ {y}\right)}{\sum_ {j = 1} ^ {c} \exp \left(\| \mathbf {w} _ {\mathbf {j}} \| _ {2} \| \mathbf {x} \| _ {2} \cos \phi_ {j}\right)} \tag {1}
$$

where  $\| \mathbf{w}_{\mathbf{y}}\| _2$  is the norm of a specific classifier in the linear layer. We'll use this geometric view of the linear layer instead of the dot-product representation.

Based on this perspective, we base the foundation of our work on the following observations from prior works [10, 12, 11]: 1) The probability/confidence of the prevalent class of an input is proportional to its norm [12]. 2) While the norm of a feature strongly scales the predictive probability, due to it's unregularized nature the norm is not sensitive to the hardness of the input [11]. In other words, the norm could be the reason for bad sensitivity of the confidence to input distribution shift. Consequently, the insensitive norm can be causally related to bad calibration. We will examine a strong correlation between the quality of calibration and the magnitude of norm in Sec. 4.2.

# 3.2 Geometric Sensitivity Decomposition of Norm and Angular Similarity

To motivate the subsequent geometric decomposition, we can revisit the softmax model,  $P(y|x) \propto \exp \left( \| \mathbf{w_y} \|_2 \| \mathbf{x} \|_2 \cos \phi_y \right)$ . There are three terms contributing to the magnitude of the exponential function,  $\| \mathbf{w_y} \|_2$ ,  $\| \mathbf{x} \|_2$  and  $\cos \phi_y$ . Due to weight regularizations,  $\| \mathbf{w_y} \|_2$  is most likely very small, while  $\cos \phi_y \in [-1, 1]$ . Therefore, the only way to obtain a high probability/confidence on training data and minimize cross-entropy loss is to 1) push the norm  $\| \mathbf{x} \|_2$  to a large value and 2) keep  $\cos |\phi_y|$  of the ground truth class close to one, i.e.,  $|\phi_y|$  close to zero. This is further supported by [17], where it was shown that logits of the ground truth class must diverge to infinity in order to minimize cross-entropy loss under gradient descent. In this process, models tend towards large norms and small angles for all training samples.

Therefore, we propose to decompose the norms of features into two components: an instance-independent scalar offset and an instance-dependent variance factor, which we define in Eq. 2. The role of the instance-independent offset  $\mathcal{C}_x$  is to minimize the loss on the entire training set and the instance-dependent component  $\Delta x$  accounts for differences in samples. Therefore, if we can disentangle the instance-independent component from the instance-dependent component, we can

obtain a norm that is sensitive to the hardness of data. Following this logic, we decompose the norm into two components.

$$
\left\| \mathbf {x} \right\| _ {2} = \left\| \Delta x \right\| _ {2} + \mathcal {C} _ {x} \tag {2}
$$

Similarly, we relax the angles such that the predicted angular similarity does not need to be close to one on the training data, i.e., making the angles larger. To achieve this, we introduce an instance-independent relaxation angle  $\mathcal{C}_{\phi}$  and an instance-dependent angle  $\Delta \phi_y$ . Analogous to the norm decomposition, the scalar  $\mathcal{C}_{\phi}$  serves solely to minimize the training loss while the instance-dependent  $\Delta \phi_y$  accounts for differences in samples. Because we need to account for the sign of the angle, we put an absolute value on it.

$$
\left| \phi_ {y} \right| = \left| \Delta \phi_ {y} \right| - \left| \mathcal {C} _ {\phi} \right| \tag {3}
$$

The  $\| \Delta \mathbf{x} \|_2$ ,  $|\Delta \phi_y|$  are the instance-dependent components and  $\mathcal{C}_x$ ,  $|\mathcal{C}_{\phi}|$  are the instance-independent components. We can rewrite the pre-softmax logits in Eq. 1 with the decomposed norm and angular similarity. (Detailed derivation in Sec. A.1 in the Appendix.)

$$
\begin{array}{l} \| \mathbf {x} \| _ {2} \cos \phi_ {y} = \| \mathbf {x} \| _ {2} \cos | \phi_ {y} | = (\| \Delta \mathbf {x} \| _ {2} + \mathcal {C} _ {x}) \cos (| \Delta \phi_ {y} | - | \mathcal {C} _ {\phi} |) \tag {4} \\ = \left(\| \Delta \mathbf {x} \| _ {2} + \mathcal {C} _ {x}\right) \frac {1}{\cos | \mathcal {C} _ {\phi} |} \cos | \Delta \phi_ {y} | \left(1 - \sin | \mathcal {C} _ {\phi} | ^ {2} \left(1 - \frac {\cos | \mathcal {C} _ {\phi} | \sin | \Delta \phi_ {y} |}{\sin | \mathcal {C} _ {\phi} | \cos | \Delta \phi_ {y} |}\right)\right) \\ \end{array}
$$

We can simplify the equation by assuming  $\cos |\phi_y|$  is close to one, which means  $|\phi_y|$  is small. This is due to the fact that  $|\phi_y|$  is the angle between the correct class weight and  $x$ , which means as training ensues, the angle converges to 0 and thus the cosine similarity converges to 1.

$$
\frac {\cos \left| \mathcal {C} _ {\phi} \right| \sin \left| \Delta \phi_ {y} \right|}{\sin \left| \mathcal {C} _ {\phi} \right| \cos \left| \Delta \phi_ {y} \right|} = \frac {\sin \left(\left| \Delta \phi_ {y} \right| + \left| \mathcal {C} _ {\phi} \right|\right) + \sin \left| \phi_ {y} \right|}{\sin \left(\left| \Delta \phi_ {y} \right| + \left| \mathcal {C} _ {\phi} \right|\right) - \sin \left| \phi_ {y} \right|} \approx 1 \tag {5}
$$

Therefore, Eq. 4, omitting the absolute value on angles because  $\cos$  is an even function, can be written as:

$$
\begin{array}{l} \left\| \mathbf {x} \right\| _ {2} \cos \phi_ {y} \approx \left(\left\| \Delta \mathbf {x} \right\| _ {2} + \mathcal {C} _ {x}\right) \frac {1}{\cos \mathcal {C} _ {\phi}} \cos \Delta \phi_ {y} \tag {6} \\ = \left(\frac {1}{\cos \mathcal {C} _ {\phi}} \| \Delta \mathbf {x} \| _ {2} + \frac {1}{\cos \mathcal {C} _ {\phi}} \mathcal {C} _ {x}\right) \cos \Delta \phi_ {y} \\ = (\alpha \| \Delta \mathbf {x} \| _ {2} + \beta) \cos \Delta \phi_ {y} \\ \end{array}
$$

Because  $\frac{1}{\cos\mathcal{C}_{\phi}}$  and  $\frac{1}{\cos\mathcal{C}_{\phi}}\mathcal{C}_x$  are instance-independent, we denote them as  $\alpha$  and  $\beta$  respectively. This geometric decomposition of norm and cosine similarity inspires us to include  $\alpha$  and  $\beta$  as free trainable parameters in a new network and the network can learn to predict the more input-sensitive  $\| \Delta \mathbf{x}\| _2$  and  $\Delta \phi_y$  instead of the original  $\| \mathbf{x}\| _2$  and  $\phi_y$ . While both the angle and norm can be decomposed we direct the focus of our paper to the norm as the angle is calibrated to accuracy [11].

# 3.3 Disentangled Training

Following the derivation in Eq 6, we replace the norm,  $\| \mathbf{x}\| _2$ , in Eq. 1 by  $(\alpha \| \Delta \mathbf{x}\| _2 + \beta)$  and  $\phi_y$  by  $\Delta \phi_j$ .  $\| \Delta \mathbf{x}\| _2$  and  $\Delta \phi_y$  are now learned outputs from a new network instead as shown in Eq. 6:

$$
P (y | x) = \frac {\exp l _ {y}}{\sum_ {j = 1} ^ {c} \exp l _ {j}} = \frac {\exp \left(\| \mathbf {w} _ {\mathbf {y}} \| _ {2} (\alpha \| \Delta \mathbf {x} \| _ {2} + \beta) \cos \Delta \phi_ {y}\right)}{\sum_ {j = 1} ^ {c} \exp \left(\| \mathbf {w} _ {\mathbf {j}} \| _ {2} (\alpha \| \Delta \mathbf {x} \| _ {2} + \beta) \cos \Delta \phi_ {j}\right)} \tag {7}
$$

The new model can be trained using the same training procedures as the vanilla network without additional hyperparameter tuning, changing the architecture or extended training time. Even though the outputs of the new network,  $\| \Delta \mathbf{x} \|_2$  and  $\Delta \phi_y$ , only approximate the original geometric relationships with Eq. 6, the effect of  $\alpha$  and  $\beta$  reflects the decomposition in Eq. 3 and Eq. 2.

![](images/0bcdaf3546071dba16266c9c6ad34a2c69128447401935be7a356c570975c5c2.jpg)  
(a) Temperature Scaling

![](images/ff516069dfd8a3f543dcfcaf9d4c17d7b13269566815da3de153ad5d2d20e520.jpg)  
Figure 1: Calibration Procedure (a): Temperature Scaling [4] changes the slope of the effective norm based on in-distribution (IND) data (See A.9 in Appendix). (b) Step 1: calibrate the offset  $\beta$  on a validation set. (c): Step 2: use a non-linear function to map OOD data further away from IND data.  
(b) Ours: calibration Step 1

![](images/3b89800ee9725faffe6b6a7c2eca2386254f6d4ca3b60ae6c4d7d702f9333ee0.jpg)  
(c) Ours: Calibration Step 2

-  $\beta$  encodes an instance-independent scalar  $C_x$  of the norm. A larger  $\beta$  corresponds to a smaller instance-dependent component  $\|\Delta \mathbf{x}\|_2$ .  
-  $\alpha$  encodes the inverse of the cosine of a relaxation angle  $\mathcal{C}_{\phi}$ . A larger  $\alpha$  corresponds to a larger  $\mathcal{C}_{\phi}$  and therefore a larger  $\Delta \phi_{j}$ .

Because  $\beta$  encodes the independent component, the new feature norm  $\| \Delta \mathbf{x}\| _2$  becomes sensitive to input changes and maps OOD data to lower norms than IND data as we can see in Fig. 3a, 3b. We regularize  $\alpha$  such that the instance-independent component  $\mathcal{C}_{\phi}$  is small. Specifically, we penalize  $\| \alpha -1\| _2^2$  because  $\alpha = 1 / \cos \mathcal{C}_{\phi}$ , i.e., if  $\alpha \approx 1$ ,  $\mathcal{C}_{\phi}\approx 0$ . We empirically found that a larger relaxation angle  $\mathcal{C}_{\phi}$  deteriorates performance because the angular similarity already correlates well with difficulty of data [11] and we do not need to encourage a large relaxation. Sec. 4.3 will empirically verify this argument.

# 3.4 Disentangled Inference

The decomposition theory in Sec. 3.2 provides a geometric perspective on the sensitivity of the norm and the angular similarity to input changes and inspires a disentangled model in Sec. 3.3. The new model uses a learnable affine transformation on the norm  $\| \Delta \mathbf{x}\| _2$ . Let's denote the affine transformed norm as the effective norm  $\mathcal{N}(\Delta \mathbf{x})\doteq \alpha \| \Delta \mathbf{x}\| _2 + \beta$ . However, the training only separates the sensitive components of the norm and angular similarity, the model can still be overconfident due to the existence of insensitive components. Therefore, we can improve calibration by modifying insensitive components, e.g.,  $\beta$  in our case. We propose a two-step calibration procedure that combines in-distribution calibration (Fig. 1b) and out-of-distribution detection (Fig. 1c) based on two observations: 1) overconfident IND data can be easily calibrated on a validation set, similar to temperature scaling [4]. 2) for OOD data, without access to a calibration set for OOD data, the best strategy is to map them far away from the IND data given that the model clearly distinguishes them.

The first step is calibrating the model on IND validation set (note our method does not rely on OOD validation data), similar to temperature calibration [4]. However, instead of tuning a temperature parameter as shown in Fig. 1a, we simply tune the offset parameter  $\beta$  on the validation set in one of two ways: 1) grid-search based on minimizing Expected Calibration Error (see Sec. 4) 2) SGD optimization based on Negative Log Likelihood [4]. Because these are post-training procedures, both methods are very efficient. We denote the new parameter as  $\beta'$ . As shown in Fig. 1b, by changing the offset, we decrease the magnitude of the norms after the affine transformation. Formally,

$$
\mathcal {N} (\Delta \mathbf {x}) = \alpha \| \Delta \mathbf {x} \| _ {2} + \beta \rightarrow \mathcal {N} (\Delta \mathbf {x}) = \alpha \| \Delta \mathbf {x} \| _ {2} + \beta^ {\prime} \tag {8}
$$

The second step approximates the calibrated affine mapping in Eq. 8 by a non-linear function which covers a wider range of the effective norm as shown in Eq. 9 and maps OOD data further away from IND data. Intuitively, when a sample is more likely IND, the non-linear function maps it closer to the calibrated transformation. When a sample is OOD, the non-linear function maps it more aggressively to a smaller magnitude, exponentially away from the IND samples.

$$
\mathcal {N} (\Delta \mathbf {x}) = \alpha \| \Delta \mathbf {x} \| _ {2} + \beta^ {\prime} (1 - e ^ {- c \| \Delta \mathbf {x} \| _ {2}}) \tag {9}
$$

where  $c$  is a hyperparameter which can be calculated as in Eq. 10. . The non-linear function grows exponentially close to the calibrated affine mapping in Eq. 8 dictated by  $1 - e^{-c\|\Delta \mathbf{x}\|_2}$  as shown in 1c. Therefore,  $e^{-c\|\Delta \mathbf{x}\|_2}$  can be viewed as an error term that quantifies how close the non-linear function is to the calibrated affine function in Eq. 8. Let  $\mu_x$  and  $\sigma_x$  denote the mean and standard deviation of the distribution of the norm of IND sample embedding calculated on the validation set. We use the heuristic that when evaluated at one standard deviation below the mean,  $\|\Delta \mathbf{x}\|_2 = \mu_x - \sigma_x$ , the approximation error  $e^{-c(\mu_x - \sigma_x)} = 0.1$ . Even though the error threshold is a hyperparameter, using an error of 0.1 leads to state-of-the-art results across all models applied.

$$
c = \frac {- l n (1 - e r r o r)}{\mu_ {x} - \sigma_ {x}} = \frac {- l n (0 . 9)}{\mu_ {x} - \sigma_ {x}} \tag {10}
$$

In summary, the sensitive norm  $\| \Delta \mathbf{x}\| _2$  is used both as a soft threshold for OOD detection and as a criterion for calibration.

While similar post-processing calibration procedure exists, such as temperature scaling [4] (illustrated in Fig. 1a and further introduced in A.9) it only provides good calibration on IND data and does not provide any mechanism to improve calibration on shifted data [5]. Our calibration procedure can improve calibration on both IND and OOD data, without access to OOD data, because the training method extracts the sensitive component in a principled manner. Just as temperature scaling, the non-linear mapping needs only to be calculated once and adds no computation at inference.

# 4 Experiments

# 4.1 Experiments on Calibration

Table 1: ResNet-28-10 on CIFAR10 averaged over 10 seed. † denotes results from [9]. Our method outperforms other single-pass methods and is comparable to Deep Ensemble [14] on corrupted data.  

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">Accuracy ↑</td><td colspan="2">ECE ↓</td><td colspan="2">NLL ↓</td></tr><tr><td>Clean</td><td>Corrupted</td><td>Clean</td><td>Corrupted</td><td>Clean</td><td>Corrupted</td></tr><tr><td>Vanilla</td><td>Wide ResNet†</td><td>96.0±0.01</td><td>72.9±0.01</td><td>0.023±0.002</td><td>0.153±0.011</td><td>0.158±0.01</td><td>1.059±0.02</td></tr><tr><td rowspan="2">Multi-Pass</td><td>Deep Ensembles†</td><td>96.6±0.01</td><td>77.9±0.01</td><td>0.010±0.001</td><td>0.087±0.004</td><td>0.114±0.01</td><td>0.815±0.01</td></tr><tr><td>MC Dropout†</td><td>96.0±0.01</td><td>70.0±0.02</td><td>0.021±0.002</td><td>0.116±0.009</td><td>0.173±0.001</td><td>1.152±0.01</td></tr><tr><td rowspan="2">Single-Pass</td><td>DUQ†</td><td>94.7±0.02</td><td>71.6±0.02</td><td>0.034±0.002</td><td>0.183±0.011</td><td>0.239±0.02</td><td>1.348±0.01</td></tr><tr><td>SNGP†</td><td>95.9±0.01</td><td>74.6±0.01</td><td>0.018±0.001</td><td>0.090±0.012</td><td>0.138±0.01</td><td>0.935±0.01</td></tr><tr><td rowspan="2">Ours</td><td>β&#x27; Grid-Founded</td><td>95.9±0.01</td><td>74.9±0.05</td><td>0.018±0.003</td><td>0.067±0.01</td><td>0.148±0.003</td><td>0.826±0.033</td></tr><tr><td>β&#x27; Optimized</td><td>95.9±0.01</td><td>74.9±0.05</td><td>0.008±0.002</td><td>0.085±0.012</td><td>0.14±0.004</td><td>0.853±0.039</td></tr></table>

Table 2: ResNet-28-10 on CIFAR100 averaged over 10 seeds. † denotes results from [9]. Our method outperforms other single-pass methods and Deep Ensemble [14] on corrupted data.  

<table><tr><td rowspan="2"></td><td rowspan="2">Method†</td><td colspan="2">Accuracy↑</td><td colspan="2">ECE ↓</td><td colspan="2">NLL ↓</td></tr><tr><td>Clean</td><td>Corrupted</td><td>Clean</td><td>Corrupted</td><td>Clean</td><td>Corrupted</td></tr><tr><td>Vanilla</td><td>Wide ResNet†</td><td>79.8±0.02</td><td>50.5±0.04</td><td>0.085±0.004</td><td>0.239±0.020</td><td>0.872±0.01</td><td>2.756±0.03</td></tr><tr><td rowspan="2">Multi-Pass</td><td>Deep Ensembles†</td><td>80.2±0.01</td><td>54.1±0.04</td><td>0.021±0.004</td><td>0.138±0.013</td><td>0.666±0.02</td><td>2.281±0.03</td></tr><tr><td>MC Dropout†</td><td>79.6±0.02</td><td>42.6±0.08</td><td>0.050±0.003</td><td>0.202±0.010</td><td>0.825±0.01</td><td>2.881±0.01</td></tr><tr><td rowspan="2">Single-Pass</td><td>DUQ†</td><td>78.5±0.02</td><td>50.4±0.02</td><td>0.119±0.001</td><td>0.281±0.012</td><td>0.980±0.02</td><td>2.841±0.01</td></tr><tr><td>SNGP†</td><td>79.9±0.03</td><td>49.0±0.02</td><td>0.025±0.012</td><td>0.117±0.014</td><td>0.847±0.01</td><td>2.626±0.01</td></tr><tr><td rowspan="2">Ours</td><td>β&#x27; Grid-Searcheds</td><td>79.8±0.03</td><td>49.8 ± 0.003</td><td>0.027±0.003</td><td>0.081 ± 0.007</td><td>0.787±0.009</td><td>2.23±0.02</td></tr><tr><td>β&#x27; Optimized</td><td>79.8±0.03</td><td>49.8±0.03</td><td>0.027±0.003</td><td>0.088±0.007</td><td>0.784±0.011</td><td>2.236±0.021</td></tr></table>

The ultimate goal of the paper is to improve model calibration under distribution shift by improving sensitivity. Popular metrics for measuring calibration include: Negative Log-Likelihood (NLL [18]), Brier [19] and Expected Calibration Error (ECE [20]). Our goal is for our model is to produce values close to 0 in these metrics, which maximizes calibration. Please refer to Sec. A.2 (Appendix) for more detailed discussion on these metrics. Following prior works [9, 8, 5], we will use CIFAR10 and CIFAR100 as the in-distribution training and testing dataset, and apply the image corruption library provided by [1] to benchmark calibration performance under distribution shift. The library provides 16 types of noises with 5 severity scales. In this section, we show that our model outperforms

Table 3: Generalizability Experiments We benchmark our method against the vanilla models using 12 different backbones and 4 different datasets.  

<table><tr><td></td><td></td><td colspan="4">Clean</td><td colspan="4">Corrupt/Rotate</td></tr><tr><td>model</td><td>dataset</td><td>accuracy↑</td><td>ECE↓</td><td>NLL↓</td><td>Brier↓</td><td>accuracy↑</td><td>ECE↓</td><td>NLL↓</td><td>Brier↓</td></tr><tr><td>LeNet5</td><td>Mnist</td><td>96.16%</td><td>0.01</td><td>0.132</td><td>0.006</td><td>33.95%</td><td>0.43</td><td>4.533</td><td>0.104</td></tr><tr><td>GSD LeNet5</td><td>Mnist</td><td>96.86%</td><td>0.005</td><td>0.103</td><td>0.005</td><td>35.73%</td><td>0.42</td><td>4.405</td><td>0.101</td></tr><tr><td>DenseNet</td><td>SVHN</td><td>41.72%</td><td>0.051</td><td>1.71</td><td>0.072</td><td>14.31%</td><td>0.301</td><td>3.844</td><td>0.107</td></tr><tr><td>GSD DenseNet</td><td>SVHN</td><td>41.7%</td><td>0.027</td><td>1.62</td><td>0.069</td><td>14.41%</td><td>0.287</td><td>3.134</td><td>0.106</td></tr><tr><td>ResNet34</td><td>CIFAR10</td><td>95.9%</td><td>0.007</td><td>0.149</td><td>0.006</td><td>76.54%</td><td>0.178</td><td>1.28</td><td>0.603</td></tr><tr><td>GSD ResNet34</td><td>CIFAR10</td><td>95.9%</td><td>0.005</td><td>0.148</td><td>0.006</td><td>76.54%</td><td>0.088</td><td>0.882</td><td>0.037</td></tr><tr><td>ResNet50</td><td>CIFAR10</td><td>95.32%</td><td>0.03</td><td>0.203</td><td>0.008</td><td>76.32%</td><td>0.17</td><td>1.23</td><td>0.039</td></tr><tr><td>GSD ResNet50</td><td>CIFAR10</td><td>95.82%</td><td>0.008</td><td>0.147</td><td>0.007</td><td>76.23%</td><td>0.057</td><td>0.766</td><td>0.033</td></tr><tr><td>ResNet101</td><td>CIFAR10</td><td>95.61%</td><td>0.028</td><td>0.197</td><td>0.007</td><td>77.59%</td><td>0.154</td><td>1.118</td><td>0.037</td></tr><tr><td>GSD ResNet101</td><td>CIFAR10</td><td>95.62%</td><td>0.007</td><td>0.158</td><td>0.007</td><td>77.21%</td><td>0.075</td><td>0.852</td><td>0.036</td></tr><tr><td>ResNet152</td><td>CIFAR10</td><td>95.7%</td><td>0.028</td><td>0.196</td><td>0.007</td><td>75.2%</td><td>0.179</td><td>1.337</td><td>0.041</td></tr><tr><td>GSD ResNet152</td><td>CIFAR10</td><td>95.63%</td><td>0.007</td><td>0.151</td><td>0.007</td><td>76.58%</td><td>0.058</td><td>0.765</td><td>0.033</td></tr><tr><td>ResNet34</td><td>CIFAR100</td><td>78.81%</td><td>0.071</td><td>0.868</td><td>0.003</td><td>51.16%</td><td>0.19</td><td>2.387</td><td>0.007</td></tr><tr><td>GSD ResNet34</td><td>CIFAR100</td><td>78.02%</td><td>0.037</td><td>0.938</td><td>0.003</td><td>49.27%</td><td>0.098</td><td>2.361</td><td>0.007</td></tr><tr><td>ResNet50</td><td>CIFAR100</td><td>79.28%</td><td>0.0746</td><td>0.861</td><td>0.003</td><td>49.71%</td><td>0.213</td><td>2.477</td><td>0.007</td></tr><tr><td>GSD ResNet50</td><td>CIFAR100</td><td>78.97%</td><td>0.0326</td><td>0.879</td><td>0.003</td><td>50.12%</td><td>0.08</td><td>2.264</td><td>0.006</td></tr><tr><td>ResNet101</td><td>CIFAR100</td><td>79.21%</td><td>0.725</td><td>2.98</td><td>0.009</td><td>51.34%</td><td>0.470</td><td>3.62</td><td>0.009</td></tr><tr><td>GSD ResNet101</td><td>CIFAR100</td><td>79.82%</td><td>0.034</td><td>0.834</td><td>0.003</td><td>53.14%</td><td>0.082</td><td>2.11</td><td>0.006</td></tr><tr><td>ResNet152</td><td>CIFAR100</td><td>80.71%</td><td>0.0895</td><td>0.815</td><td>0.003</td><td>54.2%</td><td>0.233</td><td>2.45</td><td>0.007</td></tr><tr><td>GSD ResNet152</td><td>CIFAR100</td><td>79.85%</td><td>0.0364</td><td>0.827</td><td>0.003</td><td>53%</td><td>0.078</td><td>2.12</td><td>0.006</td></tr></table>

Table 4: Importance of Norm While norm is poorly calibrated, it is important for calibration.  

<table><tr><td></td><td>ECE</td><td>NLL</td><td>Brier</td><td>Entropy</td><td>Accuracy</td></tr><tr><td>Vanilla (||wy||||x|| cos φy)</td><td>0.023</td><td>0.18</td><td>0.007</td><td>0.068</td><td>95.61%</td></tr><tr><td>No Weight Norm (w/o ||wy||)</td><td>0.052</td><td>0.195</td><td>0.007</td><td>0.442</td><td>95.58%</td></tr><tr><td>No x Norm (w/o ||x||)</td><td>0.596</td><td>1.058</td><td>0.04</td><td>2.045</td><td>95.61%</td></tr><tr><td>Only Cosine (w/o ||wy||||x||)</td><td>0.747</td><td>1.584</td><td>0.07</td><td>2.247</td><td>95.58%</td></tr></table>

other deterministic methods (despite their significant complexity) and is comparable or better to multi-pass methods with fewer training hyperparameters.

Compared Methods We compare to several popular state-of-the-art models including stochastic Bayesian methods (multi-pass): Deep Ensemble [14] and MC dropout [7], and recent deterministic methods (single pass): SNGP [9] and DUQ [8].

Results In Tab. 1 and 2, we compare our model to the most recent state of art deterministic methods SNGP and DUQ using Wide ResNet 28-10 [21] as the model backbone and each model evaluated using the average of 10 seeds. We report accuracy, ECE and NLL on clean and corrupted CIFAR10/100 datasets [1]. Our method outperforms all single-pass methods on calibration when data is corrupted, and even surpass ensembles on error metrics for corrupted data. We had 2 versions of our model: Grid Searched: grid search  $\beta^{\prime}$  on the validation set to minimize ECE and Optimized: optimize  $\beta^{\prime}$  on the validation set via gradient descent to minimize NLL for 10 epochs, similar to temperature scaling. We report additional results with ResNet18 in Sec. A.3 and Sec. A.4 (Appendix) with image noise and rotation respectively.

Generalizability We explored how generalizable our method (Grid Searched) is by applying it to 12 different models and 4 different datasets in Tab. 3. We can see consistently that our model had stronger calibration across all models and metrics, including models known to be well calibrated like LeNet [22]. All models were tested on CIFAR10C and CIFAR100C datasets offered by [1] where the original CIFAR10 and CIFAR100 were pre-corrupted; these were used for consistent corruption benchmarking across all models. All non-CIFAR datasets were corrupted via rotation from angles [0,350] with 10 step angles in between and the average calibration and accuracy was taken across all degrees of rotation. Our models included: DenseNet [23], LeNet [22] and 6 varying sizes of ResNet, which are described in [24]. The datasets we experimented on CIFAR10 [25], CIFAR100 [26], MNIST [27] and SVHN [28], CIFAR10C [1], CIFAR100C [1]. We report Optimized results in Tab. 10 in A.7 (Appendix). Both tuning methods yield similar performance.

![](images/f119d0571d7b83a64339971b6744196886712f1be2f644da84f7ff2f403efaf7.jpg)  
(a) CIFAR100 Accuracy

![](images/239a19da8412e9582413064b1e656c91f41b4afb5a69327ef8bd23fffa3bf053.jpg)  
Figure 2: Accuracy, ECE, norm and cosine similarity on CIFAR100 validation set with clean and Gaussian noise trained on vanilla ResNet. In the shaded region, increase in norm is responsible for increase in ECE because cosine similarity is relatively flat. Throughout training, sensitivity of the cosine similarity improves while that of the norm remains insensitive.  
(b) CIFAR100ECE

![](images/4faec2055f096ed3016047569e9c7cd9814547b2fef164f26a2c1ca1ab66e92e.jpg)  
(c) CIFAR100 Norm

![](images/eb4f8fd90f8437c360bab74eae016abaecd4cf243dabfbd7b2e15baf3ffd5321.jpg)  
(d) CIFAR100 Cosine

Qualitative Comparison The current state-of-the-art single pass models for inference on OOD data, without training on OOD data, are SNGP [9] and DUQ [8]. The primary disadvantages of these models are: 1) Hyperparameter Combinatorics: Both DUQ and SNGP require many hyperparameters as shown in Tab. 9 in A.6 (Appendix). Our model only has one hyperparameter that is tuned post-training with 10 epochs on validation set. 2) Extended Training Time: DUQ requires a centroid embedding update every epoch, while SNGP requires sampling potentially high dimensional embeddings of training points, thus increasing training time while our model trains in the same amount of time as the model it is applied to. Bayesian MCDO [7] and Deep Ensemble [14] are considered the current state-of-the-art methods for multi-pass calibration. Bayesian MCDO requires multiple passes with dropout during inference in order to achieve stronger calibration. Deep Ensembles requires  $N$  times the number of parameters as the single model it is assembling where  $N$  is the number of models ensembled. The main disadvantage of multi-pass models is high inference complexity while our model adds no overhead computation at inference.

Importance of the Norm While we have shown and conjectured that the norm of  $x$  is uncalibrated to OOD data and not always well calibrated to IND data, one might suggest to simply remove the norm. We show in Tab. 4 though the norm is uncalibrated it is still important for inference. We trained ResNet18 on CIFAR10 and then ran inference with ResNet18 modified in the following: dividing out the norms of the weights for each class, dividing out the norm of the input and then dividing out both. As we can see the weight norm contributes minimally to inference as accuracy decreased by  $0.03\%$  without it and as previous work has shown the angle dominates classification. We can see with  $||\mathbf{x}||$  removed the entropy is at its highest while calibration is very poor, implying the distribution is much more uniform when it should be peaked, as a larger entropy implies a more uniform distribution. Thus the root of the issue does not lie in the existence of the norm, but it's lack of sensitivity.

# 4.2 Reasons for Bad Calibration under Distribution Shift

To identify the cause of bad calibration, we record the accuracy, ECE, norm and cosine similarity of a model during training of a vanilla ResNet model. Specifically, we record the evaluation statistics on clean data and also on data corrupted with Gaussian noise on CIFAR100. Fig. 2a and 2b show the accuracy and ECE respectively. We observe that evaluation on Gaussian noise corrupted data yields lower accuracy and higher ECE compared to evaluation on clean data. This demonstrates that the model's confidence fails to adapt to the decreasing accuracy. Fig. 2c and 2d show the change of average norm and average cosine similarity throughout training. The difference between Gaussian noised data and clean data is also reported. We observe that the norm of clean data and the norm of Gaussian noised data are close and the difference remains constantly low whereas the cosine similarity of the two diverges with training. This indicates that sensitivity of cosine similarity increases whereas sensitivity of the norm remains low with training. In the shaded region of Fig. 2b-2d where ECE increases the most, we observe that the norm also increases but the cosine similarity only increases slowly. Based on supporting literature [12], [11] and this correlation, the observation supports the conjecture that the insensitivity of the norm is responsible for bad calibration.

# 4.3 Empirical Support for the Disentangled Training

In the first set of experiments, we show that  $\alpha$  and  $\beta$  reflect the effects of the geometric decomposition as claimed in Sec. 3.2 with different  $\alpha - \beta$  configurations. From Fig. 5a - 5d (Appendix),

Table 5: OOD AUROC $\uparrow$  using Norm and Similarity We show OOD detection results using norm and cosine similarity. SVHN [29] is used as the OOD dataset. Our method ( $\alpha$ -regularized) significantly increases the sensitivity of feature norm.  

<table><tr><td>ResNet18</td><td>Criterion</td><td>CIFAR10</td><td>CIFAR10 (Incorrect)</td></tr><tr><td rowspan="2">Vanilla</td><td>Norm</td><td>90.48</td><td>67.23</td></tr><tr><td>Similarity</td><td>93.87</td><td>56.98</td></tr><tr><td rowspan="2">α- regularized</td><td>Norm</td><td>99.05</td><td>93.16</td></tr><tr><td>Similarity</td><td>97.09</td><td>74.82</td></tr><tr><td rowspan="2">α- unregularized</td><td>Norm</td><td>98.20</td><td>88.29</td></tr><tr><td>Similarity</td><td>94.72</td><td>60.63</td></tr></table>

![](images/2e50576782c785a863ea1662b1e4c30797eabe0d1322d775b80fce33adb5f754.jpg)  
(a) CIFAR10 vs. SVHN  
(a) CIFAR10 vs. SVHN AUROC  
Figure 3: Histogram of Norm Distribution Our model ( $\alpha$ -regularized) improves separation of norm between IND and OOD data.

<table><tr><td>ResNet18</td><td>Criterion</td><td>CIFAR100</td><td>CIFAR100 (Incorrect)</td></tr><tr><td rowspan="2">vanilla</td><td>Norm</td><td>79.38</td><td>62.66</td></tr><tr><td>Similarity</td><td>82.26</td><td>55.54</td></tr><tr><td rowspan="2">α- regularized</td><td>Norm</td><td>94.46</td><td>86.67</td></tr><tr><td>Similarity</td><td>85.68</td><td>63.24</td></tr><tr><td rowspan="2">α- unregularized</td><td>Norm</td><td>84.78</td><td>73.11</td></tr><tr><td>Similarity</td><td>72.61</td><td>42.90</td></tr></table>

![](images/fb42e3df0bc92b2948d1a794edf3d3600057348090b4bab222f0ebc6409d66a6.jpg)  
(b) CIFAR100 vs. SVHN AUROC  
(b) CIFAR100 vs. SVHN

we observe that the norm decreases linearly with  $\beta$  for fixed  $\alpha$ . From Fig. 5e - 5h (Appendix), we observe that the angle increases linearly with  $\arccos(1 / \alpha)$ . The observations are consistent with the original geometric motivation.  $\beta$  encodes an instance-independent portion,  $C_x$ , of the norm. As  $\beta$  increases,  $C_x$  increases and therefore the magnitude of the dependent component,  $\| \Delta x \|_2$  decreases linearly.  $\alpha$  encodes the inverse of the cosine of a relaxation angle,  $C_\phi$ . As  $\arccos(1 / \alpha)$  increases, the resulting angle,  $\Delta \phi$  increases linearly due to the increased relaxation angle encoded by  $\alpha$ .

In the second set of experiments, we show that the new model effectively increases the sensitivity of both the norm and the angle to input distribution shift as claimed in Sec. 3.3. Specifically, we measure OOD detection performance of the models using both the norm and the cosine similarity with the Area Under the Receiver Operating Characteristic (AUROC) curve metric. We use CIFAR10/100 as the IND data and SVHN [29] as the OOD data. In Tab. 5a and 5b we show two configurations of models in addition to vanilla ResNet18: ( $\alpha$ -regularized) we regularize  $\alpha$  such that it stays close to one as described in Sec. 3.3; ( $\alpha$ -unregularized) we optimize both  $\alpha$  and  $\beta$  freely without constraints. Compared to vanilla ResNet, the norms predicted by our models achieve significant improvement in separating IND data from OOD data. Additionally, we visualize the distribution of norms in Fig. 3a and 3b. The separation between IND and OOD data increases significantly compared to vanilla ResNet18. However, a large  $\alpha$  (see  $\alpha$ -unregularized in Tab. 5a and 5b) leads to marginal cosine similarity sensitivity improvement on CIFAR10 and CIFAR100. This indirectly confirms our observations in Sec. 4.2 and in prior works [11] that cosine similarity correlates well with distribution shift. Introducing further angle relaxation might not be always beneficial. While we mainly focus on calibration, our method also strengthens its base model's ability for OOD detection.

We also experiment on separating incorrectly classified IND data from OOD data. This relates to the idea of separating aleatoric uncertainty, which is uncertainty about the data and epistemic uncertainty, which is uncertainty due to lack of data. Incorrectly classified data should have high aleatoric uncertainty while OOD data should have high epistemic uncertainty [5]. As shown in Tab. 5a and 5b, our models using norm as the criterion can also separate incorrectly classified CIFAR10/CIFAR100 from SVHN data significantly better than vanilla ResNet.

# 5 Conclusion

In this paper, we studied the geometry of the last linear decision layer and identified the insensitivity of the norm as the culprit of bad calibration under distribution shift. To encourage sensitivity, we derived a general theory to decompose the norm and angular similarity. Inspired by the theory, we proposed a simple yet very effective training and inference scheme that encourages the norm to reflect distribution changes. The model outperforms other deterministic single pass-methods in calibration metrics with much fewer hyperparameters. We also demonstrated its superior generalizability on a variety of popular neural networks. Note that our problem and method have positive societal impact, as calibration under shift improves overall confidence and robustness of these models.

# References

[1] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. Proceedings of the International Conference on Learning Representations, 2019.  
[2] Sebastian Brechtel, Tobias Gindele, and Rüdiger Dillmann. Probabilistic decision-making under uncertainty for autonomous driving using continuous pomdpms. In 17th international IEEE conference on intelligent transportation systems (ITSC), pages 392-399. IEEE, 2014.  
[3] Yi Yang, Zhigang Ma, Feiping Nie, Xiaojun Chang, and Alexander G Hauptmann. Multi-class active learning by uncertainty sampling with diversity maximization. International Journal of Computer Vision, 113(2):113-127, 2015.  
[4] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning, pages 1321-1330. PMLR, 2017.  
[5] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua V Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. arXiv preprint arXiv:1906.02530, 2019.  
[6] Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? arXiv preprint arXiv:1703.04977, 2017.  
[7] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050-1059. PMLR, 2016.  
[8] Joost Van Amersfoort, Lewis Smith, Yee Whye Teh, and Yarin Gal. Uncertainty estimation using a single deep deterministic neural network. In International Conference on Machine Learning, pages 9690-9700. PMLR, 2020.  
[9] Jeremiah Zhe Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax-Weiss, and Balaji Lakshminarayanan. Simple and principled uncertainty estimation with deterministic deep learning via distance awareness. arXiv preprint arXiv:2006.10108, 2020.  
[10] Weiyang Liu, Zhen Liu, Zhiding Yu, Bo Dai, Rongmei Lin, Yisen Wang, James M Rehg, and Le Song. Decoupled networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2771-2779, 2018.  
[11] Beidi Chen, Weiyang Liu, Zhiding Yu, Jan Kautz, Anshumali Shrivastava, Animesh Garg, and Animashree Anandkumar. Angular visual hardness. In International Conference on Machine Learning, pages 1637-1648. PMLR, 2020.  
[12] Ioannis Kansizoglou, Loukas Bampis, and Antonios Gasteratos. Deep feature space: A geometrical perspective. arXiv preprint arXiv:2007.00062, 2020.  
[13] Weiyang Liu, Yandong Wen, Zhiding Yu, and Meng Yang. Large-margin softmax loss for convolutional neural networks. In ICML, volume 2, page 7, 2016.  
[14] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. arXiv preprint arXiv:1612.01474, 2016.  
[15] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. arXiv preprint arXiv:1802.05957, 2018.  
[16] Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of wasserstein gans. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[17] Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. In *The Journal of Machine Learning Research*, page 2822–2878, 2018.  
[18] Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning. Springer Series in Statistics. Springer New York Inc., New York, NY, USA, 2001.  
[19] Glenn W Brier. Verification of forecasts expressed in terms of probability. Monthly weather review, 78(1):1-3, 1950.  
[20] Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29, 2015.  
[21] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. CoRR, abs/1605.07146, 2016.  
[22] Yann Lecun, León Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. In Proceedings of the IEEE, pages 2278-2324, 1998.  
[23] Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. In CVPR, pages 2261-2269. IEEE Computer Society, 2017.  
[24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015.  
[25] Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10 (canadian institute for advanced research).  
[26] Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-100 (canadian institute for advanced research).  
[27] Yann LeCun and Corinna Cortes. MNIST handwritten digit database. 2010.  
[28] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. Reading digits in natural images with unsupervised feature learning. 2011.  
[29] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. 2011.  
[30] Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association, 102(477):359-378, 2007.  
[31] John Platt et al. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3):61-74, 1999.
