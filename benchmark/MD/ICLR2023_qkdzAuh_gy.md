# ID AND OOD PERFORMANCE ARE SOMETIMES INVERSELY CORRELATED ON REAL-WORLD DATASETS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Context. Several studies have empirically compared in-distribution (ID) and out-of-distribution (OOD) performance of various models. They report frequent positive correlations on benchmarks in computer vision and NLP. Surprisingly, they never observe inverse correlations suggesting necessary trade-offs. This matters to determine whether ID performance can serve as a proxy for OOD generalization.

Findings. This paper shows that inverse correlations between ID and OOD performance do happen in real-world benchmarks. They could be missed in past studies because of a biased selection of models. We show an example on the WILDS-Camelyon17 dataset, using models from multiple training epochs and random seeds. Our observations are particularly striking with models trained with a regularizer that diversifies the solutions to the ERM objective (Teney et al., 2022a).

Implications. We nuance recommendations and conclusions made in past studies.

- High OOD performance may sometimes require trading off ID performance.  
- Focusing on ID performance alone may not lead to optimal OOD performance: it can lead to diminishing and eventually negative returns in OOD performance.  
- Our example reminds that empirical studies only chart regimes achievable with existing methods: care is warranted in deriving prescriptive recommendations.

# 1 INTRODUCTION

Past observations. This paper complements existing studies that empirically compare in-distribution (ID) and out-of-distribution $^{1}$  (OOD) performance of deep learning models (Andreassen et al., 2021; Djolonga et al., 2021; Miller et al., 2021; Mania & Sra, 2020; Miller et al., 2020; Taori et al., 2020; Wenzel et al., 2022). It has long been known that models applied to OOD data suffer a drop in performance, e.g. in classification accuracy. The above studies show that, despite this gap, ID and OOD performance are often positively correlated $^{2}$  across models on benchmarks in computer vision (Miller et al., 2021) and NLP (Miller et al., 2020).

Past explanations. Frequent positive correlations are surprising because nothing forbids opposite, inverse ones. Indeed, ID and OOD data contain different associations between labels and features. One could imagine e.g. that an image background is associated with class  $\mathcal{C}_1$  ID and class  $\mathcal{C}_2$  OOD. The more a model relies on the presence of this background, the better its ID performance but the worse its OOD performance, resulting in an inverse correlation. Never observing inverse correlations has been explained with the possibility that real-world benchmarks might contain

![](images/3aadf1a259f9123c72b5c2a8eb1acaa2534431d1d7434f1042c0e828d70fb618.jpg)  
Figure 1: Past studies suggest that positive correlations between ID/OOD performance are ubiquitous. This paper shows, with a counterexample, that inverse correlations are possible and can be accidentally overlooked. The possible need for an ID/OOD trade-off is thus not merely theoretical and should be envisioned, e.g. preventing blind reliance on ID performance for model selection.

only mild distribution shifts (Mania & Sra, 2020). We will show that such observations can also be an artefact of study design.

A recent large-scale study. Wenzel et al. (2022) show that not all datasets display a clear positive correlation. The authors observe other patterns that sometimes reveal underspecification (D'Amour et al., 2020; Teney et al., 2022b; Lee et al., 2022), or severe shifts that prevent any training/test transfer. Surprisingly, they never observe inverse correlations:

"We did not observe any trade-off between accuracy and robustness, where more accurate models would overfit to spurious features that do not generalize." (Wenzel et al., 2022)

On the contrary, we do observe such cases and showcase it on a dataset from the above study.

Explaining inverse correlations. We name the underlying cause a misspecification, by extension of underspecification which was previously used to explain why models with similar ID performance can vary in OOD performance (D'Amour et al., 2020; Teney et al., 2022b; Lee et al., 2022). In cases of misspecification, the standard ERM objective (empirical risk minimization), which drives ID performance, conflicts with the goal of OOD performance. ID and OOD metrics can then vary independently and inversely to one another. In Section 5, we present a minimal theoretical example that illustrates how an inverse correlation pattern originates from the presence of both robust and spurious features in the data. In Section 6, we show that different patterns of ID/OOD performance occur with different magnitudes of distribution shifts.

# Summary of contributions.

- An empirical examination of ID vs. OOD performance on the WILD-Camelyon17 dataset (Koh et al., 2021) that shows an inverse correlation pattern conflicting with past evidence (Section 3).  
- An explanation and empirical verification that past studies could miss such patterns because of a biased sampling of models (Section 4).  
- A theoretical analysis showing when inverse correlations patterns can occur (Sections 5-6).  
- A revision of conclusions and recommendations made in past studies (Section 7).

# 2 PREVIOUSLY-OBSERVED PATTERNS OF ID VS. OOD PERFORMANCE

Past studies conclude that ID and OOD performance tend to vary jointly across models on many real-world datasets (Djolonga et al., 2021; Miller et al., 2021; Taori et al., 2020). Millet et al. report an almost-systematic linear correlation between probit-scaled ID and OOD accuracies. Mania & Sra (2020) explain this trend with the fact that real-world benchmarks contain only mild distribution shifts. Andreassen et al. (2021) find that pretrained models perform "above the linear trend" in the early stages of fine-tuning. Their OOD accuracy rises more quickly than their ID accuracy early on, even though the final accuracies agree with a linear trend.

Most recently, the large-scale study of Wenzel et al. (2022) is more nuanced: they observe a linear trend only on some datasets. Their setup consists in fine-tuning an ImageNet-pretrained model on a chosen dataset and evaluating it on matching ID and OOD test sets. They repeat the procedure with a variety of datasets, architectures, and implementation options such as data augmentations.

The scatter plots of ID/OOD accuracy in Wenzel et al. (2022) show four typical patterns (Figure 2).

![](images/a45aee875ffaa6a4226e2897aa97fd568445a8a6a789ed429b52369ac99d90ad.jpg)  
Figure 2: Typical patterns observed in Wenzel et al. (2022) (reproduced with permission).

1. Increasing line (positive correlation): mild distribution shift. ID and OOD accuracies are positively correlated. Focusing on classical generalization and ID performance brings concurrent OOD improvements.  
2. Vertical line: underspecification (D'Amour et al., 2020; Lee et al., 2022; Teney et al., 2022b). Different models obtain a similar high ID performance but different OOD performance. The objective of high ID performance does not sufficiently constrain the learning. Typically, multiple features in the data (a.k.a. biased or spurious features) can be used to obtain high ID performance, but not all of them are equally reliable on OOD data. To improve OOD performance, additional task-specific information is necessary, e.g. additional supervision or inductive biases (custom architectures, regularizers, etc.).  
3. Horizontal line, low OOD accuracy: severe distribution shift. No model performs well OOD. A severe shift prevents any transfer between training and OOD test data. The task needs to be significantly more constrained e.g. with task-specific inductive biases.  
4. No clear trend: underspecification. Models show a variety of ID and OOD accuracies. The difference with (2) is the wider variety along the ID axis, e.g. because a difficult learning task yields solutions of lower ID accuracy from local minima of the ERM objective.

The authors note the absence of decreasing patterns, which are however possible in theory.

5. Decreasing line (inverse correlation): misspecification. The highest accuracy ID and OOD are achieved by different models. Optima of the ERM objective, which are expected to be optima in ID performance, do not correspond to optima in OOD performance. This implies a trade-off: higher OOD performance is possible at the cost of lower ID performance.

# When does an inverse correlation occur between ID and OOD performance?

Intuitively, it can occur when there is a pattern in the data that is predictive in one distribution and misleading in the other. For example, object classes  $\mathcal{C}_1$  and  $\mathcal{C}_2$  are respectively associated with image backgrounds  $\mathcal{B}_1$  and  $\mathcal{B}_2$  in ID data, and respectively  $\mathcal{B}_2$  in  $\mathcal{B}_1$  ( swapped) in OOD data. Relying on the background can improve performance on either distribution but not both simultaneously. While such severe shifts might be rare, the next section presents an actual example.

# 3 NEW OBSERVATIONS: INVERSELY CORRELATED ID/OOD PERFORMANCE

We use the WILDS-Camelyon17 dataset described by Koh et al. (2021) in a manner similar to Wenzel et al. (2022). These authors evaluate different architectures, assuming different inductive biases will produce models covering a range of ID/OOD accuracies. For simplicity, we rely instead on different random seeds since D'Amour et al. (2020) showed this to be sufficient to cover a variety of ID/OOD accuracies on this dataset. To increase this variety even further without manually picking architectures, we also train models with the diversity-inducing method of Teney et al. (2022a).

Experimental details. We use DenseNet-121 models pretrained by the authors of the dataset using 10 different seeds. For each of these 10 models, we re-train the last linear layer from a random initialization for 1 to 10 epochs, while keeping the other layers frozen. These are referred to as "ERM models". We perform this re-training with 10 different seeds which gives  $10^{3}$  ERM models (10 pretraining seeds  $\times$  10 re-training seeds  $\times$  10 numbers of epochs). In addition, we repeat this re-training of the last layer with the diversity-inducing method of Teney et al. (2022a) (details in the box below). These are referred to as "diverse models". Each run of the methods produces 24 different models, giving a total of  $10^{3}$ . 24 such models ( $10^{3}$  as above  $\times$  24).

# Background: learning diverse solutions to a learning task.

A range of methods exist to identify multiple neural networks of similar high ID performance but that differ in other desirable properties such as OOD performance, interpretability, adversarial robustness, etc. These methods are relevant in cases of underspecification (D'Amour et al., 2020) i.e. when the standard ERM objective does not constrain the solution space to a unique one.

Recent methods consist of optimizing multiple models while encouraging diversity in their feature space (Heljakka et al., 2022; Yashima et al., 2022), prediction space (Pagliardini et al., 2022; Lee et al., 2022), or gradient space (Ross et al., 2018; 2020; Teney et al., 2022a;b).

This study uses the method of Teney et al. (2022a) which encourages gradient diversity. The method trains many copies of the same model in parallel – in our case, a linear classifier on top of a frozen DenseNet backbone (see Figure 3).

![](images/5426c953cb7a72d99e38775b78a6189112a2115b6c460fd5d71dee493238a5c7.jpg)  
Figure 3: Method used to train a diverse set of models. Each training image  $\pmb{x}$  goes through a frozen pretrained DenseNet to produce features  $h = f(\pmb{x})$ . We train a set of linear classifiers  $\{g_i\}_{i=1}^n$  on these features. A diversity loss minimizes the pairwise similarity between their input gradients.

The models are optimized by standard SGD to minimize the sum of a standard classification loss (cross-entropy) with a diversity loss that encourages diversity across models. Using  $\lambda$  a weight hyperparameter, the complete loss is  $\mathcal{L} = \mathcal{L}_{\mathrm{classification}} + \lambda \mathcal{L}_{\mathrm{diversity}}$ . The second term encourages each copy to rely on different features by minimizing the mutual alignment of input gradients:

$$
\mathcal {L} _ {\text {d i v e r s i t y}} = \Sigma_ {\boldsymbol {x} \in \text {T r a i n i n g}} \Sigma_ {i = 1} ^ {n} \Sigma_ {j = i + 1} ^ {n} \nabla_ {\boldsymbol {h}} g _ {i} (\boldsymbol {h}). \nabla_ {\boldsymbol {h}} g _ {j} (\boldsymbol {h}) \quad \text {w i t h} \quad \boldsymbol {h} = f (\boldsymbol {x}). \tag {1}
$$

These pairwise dot products quantify the mutual alignment of the gradients. Intuitively, minimizing this loss makes each model locally sensitive along different directions in its input space. Assuming that  $g$  produces a vector of logits (as many as there are classes),  $\nabla_h g(\cdot)$  refers to the gradient of the largest logit w.r.t. the classifier's input  $h$ . We use  $n = 24$  copies and a weight  $\lambda = 10$  that were selected for giving a wide range of ID accuracies. See Teney et al. (2022a) for details about the method. Variations were also described by Ross et al. (2020) and Teney et al. (2022b).

![](images/c0ea866fde6fc311ff896c076fa3695e991bbf2cc80d49e6dc18dd1b0254a2b3.jpg)  
Figure 4: Our new observations show that higher OOD accuracy can sometimes be traded for lower ID accuracy. Each panel corresponds to a different pretraining seed. Each dot represents a linear classifier on frozen features, re-trained with a different seed and/or number of epochs. They are re-trained with standard ERM (red dots  $\bullet$ ) or a diversity-inducing method (gray dots  $\bullet$ ). The latter set includes models with higher OOD/ lower ID accuracies. See Appendix A for additional plots.

![](images/65508fce7583039f4b6c0f3a697c8a0112c6cd28262cffd061b03f3ea137bff7.jpg)

![](images/ab1012f8195317ad141a63f9f1a795cc9f859001ab5421cc649c57cad30cd284.jpg)

![](images/db7491290461ecfee71c0d8941f392b2cc858a2bc4480becfb0be085beabaeee.jpg)

Results with ERM models. In Figure 4 we plot the ID vs. OOD accuracy of ERM models as red dots  $(\bullet)$ . Each panel corresponds to a different pretraining seed. The variation across panels (note the different Y-axis limits) shows that OOD performance varies across pre-training seeds even though the ID accuracy is similar, as noted by Koh et al. (2021). Our new observations are visible within each panel. The dots (models) in any panel differ in their re-training seed and/or number of epochs. The seeds induce little variation, but the number of epochs produce patterns of decreasing

trend (negative correlation). Despite the narrow ID variation (X axis), careful inspection confirms that the pattern appears in nearly all cases (see Appendix A for zoomed-in plots).

Results with diverse models. We plot models trained with the diversity-inducing method (Teney et al., 2022a) as gray dots  $(\bullet)$ . These models cover a wider range of accuracies and form patterns that extend those of ERM models. The decreasing trend is now obvious. This trend is also clearly juxtaposed with a rising trend where ID/OOD performance are positively correlated. This suggests a point of highest OOD performance after which the model overfits to ID data. Appendix A shows similar results with other pretraining seeds. The patterns are not always clearly discernible because large regions of the performance landscape are not covered, despite the diversity-inducing method. We further discuss this issue next.

# 4 WHY PAST STUDIES MISSED NEGATIVE CORRELATIONS:

# A BIASED SAMPLING OF MODELS

We identified several factors explaining the discrepancy between our observations and past studies.

- ERM models alone do not always form clear patterns (red dots  $\bullet$  in Figure 4). In our observations, the models trained with a diversity-inducing method (gray dots  $\bullet$ ) were key in making the suspected patterns more obvious, because they cover a wider range of accuracies.  
- The ID/OOD trade-off varies during training, as noted by Andreassen et al. (2021). This variation across training epochs is responsible for much of the newly observed patterns. However, models of different architectures or pretraining seeds are not always comparable with one another because of shifts in their overall performance (see e.g. different Y-axis limits across panels in Figure 4). Therefore the performance across epochs should be analyzed individually per model.  
- The "inverse correlation" patterns are not equally visible with all pretraining seeds. In some cases, a careful examination of zoomed-in plots is necessary, see Appendix A. This is a reminder that stochastic factors in deep learning can have large effects and that empirical studies should randomize them as much as possible.

To demonstrate these points, we plot our data (same as in Figure 4) while keeping only the ERM models trained for 10 epochs and including all pretraining seeds on the same panel. Figure 5 shows that these small changes reproduce the vertical line observed by Wenzel et al. (2022), which completely misses the inverse correlations patterns visible in Figure 4.

![](images/97591b452c264ca9bce22b2fe18bb63ab01b34c0b1fcc570fd0288920a5c1f39.jpg)  
Figure 5: We plot again the ERM models of Figure 4 (red dots  $\bullet$ ) but only include models trained for a fixed number of epochs and combine all pretraining seeds in the same plot. This reproduces the vertical line from (Wenzel et al., 2022), which completely misses the patterns of inverse correlation.

A general explanation is that past studies undersample regions of the ID/OOD performance space. They usually consider a variety of architectures in an attempt to sample this space. However, different architectures do not necessarily behave very differently from one another (see the box below). We lack methods to reliably identify models of high OOD performance, but the diversity-inducing method that we use yields models spanning a wide range of the performance spectrum.

# Why isn't it sufficient to evaluate a variety of architectures?

Different architectures do not necessarily induce radically different behaviour. Even CNNs and vision transformers were shown to have similar failure modes (Pinto et al., 2022). Distinct architectures can share similar inductive biases that are e.g. due to SGD, such as the simplicity bias (Scimeca et al., 2022; Shah et al., 2020) or neural anisotropies (Ortiz-Jimenez et al., 2021).

Therefore, independently trained models are not necessarily diverse despite the variety of architectures. ID/OOD performance may only vary along similar directions across models.

# 5 THEORETICAL ANALYSIS OF A LINEAR CASE

In this section, we present a minimal case that shows a trade-off between ID and OOD performance and aids understanding the cause of such a pattern. Let  $y \in \mathbb{R}$  be a target variable to be predicted by a model, and  $\pmb{x}$  the features used as input to the model. These features are a concatenation of invariant and spurious features (defined implicitly below):  $\pmb{x} = [x_{\mathrm{inv}}; x_{\mathrm{spu}}]$  with  $x_{\mathrm{inv}} \in \mathbb{R}^{d_{\mathrm{inv}}}$  and  $x_{\mathrm{spu}} \in \mathbb{R}^{d_{\mathrm{spu}}}$ . Following Arjovsky et al. (2019); Rosenfeld et al. (2020); Zhou et al. (2022), we consider the simple data-generating process defined by the following structural equations:

$$
y ^ {e} = \boldsymbol {\gamma} ^ {\top} \mathbf {x} _ {\mathrm {i n v}} ^ {e} + \epsilon_ {\mathrm {i n v}} \quad \mathbf {x} _ {\mathrm {s p u}} ^ {e} = y ^ {e} \mathbf {1} ^ {\mathbf {s}} + \boldsymbol {\alpha} ^ {e} \circ \epsilon_ {\mathrm {s p u}} \tag {2}
$$

where  $e \in \{e_{\mathrm{ID}}, e_{\mathrm{OOD}}\}$  is an environment index referring to ID or OOD data. The random variables  $\epsilon_{\mathrm{inv}}$  and  $\epsilon_{\mathrm{spu}}$  represent symmetric independent random noise with zero-mean, sub-Gaussian tail probabilities, and  $\operatorname{Var}\left(\epsilon_{\mathrm{inv}}\right) > 0$ ,  $\operatorname{Var}\left(\epsilon_{\mathrm{spu},i}\right) > 0$ ,  $\forall i \in [1, d_{\mathrm{spu}}]$ . The vector  $\gamma \in \mathbb{R}^{d_{\mathrm{inv}}}$  determines the relation between the target variable and the invariant features and is identical across environments. In contrast, the vector  $\alpha^e$  affects the spurious features and varies among environments. Therefore the invariant features are similarly predictive in ID and OOD data while the spurious ones are not.

To study the relationship between ID and OOD performance of a hypothetical predictive model, we assume that this model relies on a subset of the features  $\pmb{x}$ . This subset is identified by a binary mask  $\Phi \in \{0,1\}^{d_{\mathrm{inv}} + d_{\mathrm{spu}}}$ . Suppose we have already selected  $\hat{d}_{\mathrm{inv}}$  invariant features and  $\hat{d}_{\mathrm{spu}}$  spurious features, such that  $(\hat{d}_{\mathrm{inv}} + \hat{d}_{\mathrm{spu}}) = \hat{d} = ||\Phi_{\hat{d}}||_1$  where the subscript  $\hat{d}$  denotes the number of selected features. The features selected by  $\Phi_{\hat{d}}$  are  $[x_{\mathrm{inv},1}, \dots, x_{\mathrm{inv},\hat{d}_{\mathrm{inv}}}, x_{\mathrm{spu},1}, \dots, x_{\mathrm{spu},\hat{d}_{\mathrm{spu}}}]$ . Let  $\mathbb{E}$  denote either the in- and out-of-domain expectation as  $\mathbb{E}^{\mathrm{ID}}$  and  $\mathbb{E}^{\mathrm{OOD}}$ . We use  $\beta$  to denote the optimal parameter of the linear regression for a certain domain, i.e.,  $\beta_{\hat{d}} = \mathbb{E}[\Phi_{\hat{d}}(\pmb{x})^\top \Phi_{\hat{d}}(\pmb{x})] \mathbb{E}[\Phi_{\hat{d}}(\pmb{x})^\top y]$ . Then the MSE loss of the fitted linear regressor is  $\mathbb{E}[y - \Phi_{\hat{d}}(\pmb{x})^\top \beta_{\hat{d}}]^2$ . Further, let  $[\lambda_1^{\hat{d}}, \lambda_2^{\hat{d}}, \dots, \lambda_{\hat{d}}^{\hat{d}'}]$  denote the eigenvalues of  $\mathbb{E}[\Phi_{\hat{d}}(\pmb{x})^\top \Phi_{\hat{d}}(\pmb{x})]$  and  $[\pmb{v}_1^{\hat{d}}, \pmb{v}_2^{\hat{d}}, \dots, \pmb{v}_{\hat{d}}^{\hat{d}'}]$  the corresponding eigenvectors.

Given a feature mask  $\Phi_{\hat{d}}$ , we now examine how the ID and OOD losses of the model vary when including an additional spurious feature feature into  $\Phi_{\hat{d}}$  (see Appendix B for a proof).

Theorem 1. Including an additional spurious feature leads to the following change in loss  $\mathcal{L}$ :

$$
\mathcal {L} _ {\mathrm {I D}} (\Phi_ {\hat {d} + 1}) - \mathcal {L} _ {\mathrm {I D}} (\Phi_ {\hat {d}}) = \mathbb {E} ^ {\mathrm {I D}} [ y - \Phi_ {\hat {d}} (\pmb {x}) ^ {\top} \beta_ {\hat {d}} ^ {\mathrm {I D}} ] ^ {2} - \mathbb {E} [ y - \Phi_ {\hat {d} + 1} (\pmb {x}) ^ {\top} \beta_ {\hat {d} + 1} ^ {\mathrm {I D}} ] ^ {2} <   0
$$

$$
\mathcal {L} _ {\mathrm {O O D}} \left(\Phi_ {\hat {d} + 1}\right) - \mathcal {L} _ {\mathrm {O O D}} \left(\Phi_ {\hat {d}}\right) = Q _ {1} + Q _ {2} + Q _ {3}
$$

with  $Q_{1}, Q_{2}, Q_{3}$  defined as follows:

$$
{Q _ {1}} {= \mathbb {E} ^ {\mathrm {O O D}} [ y - \Phi_ {\hat {d}} (\pmb {x}) ^ {\top} \beta_ {\hat {d}} ^ {\mathrm {O O D}} ] ^ {2} - \mathbb {E} [ y - \Phi_ {\hat {d} + 1} (\pmb {x}) ^ {\top} \beta_ {\hat {d} + 1} ^ {\mathrm {O O D}} ] ^ {2}}
$$

$$
\begin{array}{l} Q _ {2} = \sum_ {i = 1} ^ {\hat {d}} \left[ \left(\mathbb {E} ^ {\mathrm {O O D}} [ \Phi_ {\hat {d}} (\pmb {x}) y ] ^ {\top} \pmb {v} _ {i} ^ {\mathrm {O O D}, \hat {d}}\right) ^ {2} \left(\lambda_ {i} ^ {\mathrm {O O D}, \hat {d}}\right) \left(\frac {1}{\lambda_ {i} ^ {\mathrm {I D} , \hat {d}}} - \frac {1}{\lambda_ {i} ^ {\mathrm {O O D} , \hat {d}}}\right) ^ {2} \right. \\ \left. - \left(\mathbb {E} ^ {\mathrm {O O D}} [ \Phi_ {\hat {d}} (\boldsymbol {x}) y ] ^ {\top} \boldsymbol {v} _ {i} ^ {\mathrm {O O D}, \hat {d} + 1}\right) ^ {2} \left(\lambda_ {i} ^ {\mathrm {O O D}, \hat {d} + 1}\right) \left(\frac {1}{\lambda_ {i} ^ {\mathrm {I D} , \hat {d} + 1}} - \frac {1}{\lambda_ {i} ^ {\mathrm {O O D} , \hat {d} + 1}}\right) ^ {2} \right] \\ \end{array}
$$

$$
Q _ {3} = \left(\mathbb {E} ^ {\mathrm {O O D}} [ \Phi_ {\hat {d} + 1} (\pmb {x}) y ] ^ {\top} \pmb {v} _ {\hat {d} + 1} ^ {\mathrm {O O D}, \hat {d} + 1}\right) ^ {2} \frac {((\alpha_ {\hat {d} + 1} ^ {\mathrm {I D}}) ^ {2} - (\alpha_ {\hat {d} + 1} ^ {\mathrm {O O D}}) ^ {2}}{(\lambda_ {\hat {d} + 1} ^ {\mathrm {I D} , \hat {d} + 1}) ^ {2} \lambda_ {\hat {d} + 1} ^ {\mathrm {O O D} , \hat {d} + 1}} > 0.
$$

Further, if the new feature is sufficiently unstable in the test domain,

i.e. if  $(\left(\alpha_{\hat{d} +1}^{\mathrm{ID}}\right)^2 -\left(\alpha_{\hat{d} +1}^{\mathrm{OOD}}\right)^2)^2$  is sufficiently large such that:

$$
\left| \left(\alpha_ {\hat {d} + 1} ^ {\mathrm {I D}}\right) ^ {2} - \left(\alpha_ {\hat {d} + 1} ^ {\mathrm {O O D}}\right) ^ {2} \right| > \sqrt {\frac {\left(\lambda_ {\hat {d} + 1} ^ {\mathrm {I D} , \hat {d} + 1}\right) ^ {2} \lambda_ {\hat {d} + 1} ^ {\mathrm {O O D} , \hat {d} + 1}}{\left(\mathbb {E} ^ {\mathrm {O O D}} [ \boldsymbol {\Phi} (\boldsymbol {x}) y ] ^ {\top} \boldsymbol {v} _ {\hat {d} + 1} ^ {\mathrm {O O D} , \hat {d} + 1}\right) ^ {2}}} \left| Q _ {1} + Q _ {2} \right|},
$$

then we have  $Q_{3} > |Q_{1} + Q_{2}|$  and therefore  $\mathcal{L}_{\mathrm{OOD}}(\Phi_{\hat{d} + 1}) - \mathcal{L}_{\mathrm{OOD}}(\Phi_{\hat{d}}) > 0$ .

Because  $\epsilon_{\mathrm{spu}}$  is a zero-mean symmetric random noise, the sign of  $\alpha_{\hat{d} +1}$  does not the matter in the results. Theorem 1 shows that adding a spurious feature to those used by the model can affect its ID and OOD losses in opposite directions, implying a trade-off between ID and OOD accuracy. In other words, this minimal case shows that a simple model without/with an extra (spurious) feature can exhibit an inverse correlation between its ID and OOD performance.

# 6 ORDERING ID/OOD PATTERNS ACCORDING TO SHIFT MAGNITUDE

The above analysis shows that inverse correlation patterns are essentially due to the presence of spurious features, i.e. features whose predictive relation with the target in ID data becomes misleading OOD. Occurrences of spurious features increase with the magnitude of the distribution shift. Therefore, the possible patterns in ID/OOD performance presented in Section 2 can be ordered according to the magnitude of the distribution shift they are likely to occur with (see Figure 6).

<table><tr><td></td><td>Positive transfer</td><td>Underspecification</td><td>Missspecification</td><td>No transfer</td></tr><tr><td>Distribution shift</td><td>Mild</td><td colspan="3">→ (Too?) Severe</td></tr><tr><td rowspan="3">Typical pattern (toy representation)</td><td>Positive correlation</td><td>Vertical line/no clear trend</td><td>Negative correlation</td><td>Low horizontal line</td></tr><tr><td>OOD Accuracy</td><td>OOD Accuracy</td><td>OOD Accuracy</td><td>OOD Accuracy</td></tr><tr><td>ID Accuracy</td><td>ID Accuracy</td><td>ID Accuracy</td><td>ID Accuracy</td></tr><tr><td>Valid approaches for improving OOD performance</td><td>Simply focus on improving ID performance</td><td colspan="2">Task-relevant inductive biases e.g. arch., regularizers. Data augmentation with task-relevant transformations. Non-i.i.d. training data e.g. multiple training domains.</td><td>Open question</td></tr><tr><td>Example datasets</td><td>ImageNet → ImageNet v2 (Recht et al., 2019)</td><td>PACS sketch → photograph (Li et al., 2017)</td><td>WILDS Camelyon17 (Koh et al., 2021)</td><td>DomainNet infographic → quickdraw (Koh et al., 2021)</td></tr></table>

Figure 6: Various patterns of ID vs. OOD performance occur at different levels of distribution shift.

With the smallest distribution shifts (leftmost case in Figure 6), for example training on ImageNet and testing on its replication ImageNet v2 (Recht et al., 2019), ID validation performance closely correlates with OOD test performance. This OOD setting is the easiest because one can focus on improving classical generalization and reap concurrent improvements OOD.

With a larger distribution shift, more features are likely to be spurious, which is likely to break the ID/OOD correlation. The task of improving OOD performance is likely to be under- or misspecified, i.e. there is not enough information to determine which features a model should rely on to perform well OOD. Valid approaches include modifying the objective function, injecting task-specific information with custom architectures (e.g. building-in invariance to rotations as in Teney & Hebert (2016)), well-chosen data augmentations, or inhomogeneous training data such as multiple training environments (Li et al., 2017) and counterfactual examples (Teney et al., 2020a).

With extreme distribution shifts, most predictive features are overwhelmingly spurious and it is very difficult to learn any one relevant in OOD data (rightmost case in Figure 6).

The proposed ordering of patterns is rather informal and could be further developed following the two axes of diversity shifts and correlation shifts proposed by Ye et al. (2022) (see also Wiles et al. (2021)). More recently, Wang & Veitch (2022) showed that the suitability of various methods for OOD generalization depends on particularities of the underlying causal structure of the task – which must therefore be known to select a suitable method. Identifying which ID/OOD patterns occur with particular causal structures might serve as a tool to understand the type of OOD situation one is facing and identify a suitable method.

# 7 REVISITING RECOMMENDATIONS MADE IN PAST STUDIES

We have established that observations in past studies were incomplete. We now bring nuance to some recommendations and conclusions made in these studies.

- Focusing on a single metric.

"We see the following potential prescriptive outcomes (...) correlation between OOD and ID performance can simplify model development since we can focus on a single metric." (Miller et al., 2021)

We demonstrated that inverse correlations are a possibility, hence there exist scenarios where an ID metric would be misleading. In general, relying on a single metric during model development is ill-advised (Teney et al., 2020b). Even more so here since it cannot capture trade-offs along multiple axes. A model with a suboptimal ID performance may have learned features that enable better OOD generalization. Our recommendation is to track multiple metrics e.g. performance on multiple distributions or qualitative interpretable predictions on representative test points.

- Improving ID performance for OOD robustness.

"If practitioners want to make the model more robust on OOD data, the main focus should be to improve the ID classification error. (...) We speculate that the risk of overfitting large pretrained models to the downstream test set is minimal, and it seems to be not a good strategy to, e.g., reduce the capacity of the model in the hope of better OOD generalization." (Wenzel et al., 2022)

This recommendation assumes the persistence of a positive correlation. On the opposite, we saw that a positive correlation can precede a regime of inverse correlation (Figure 4, left panels). If the goal is to improve OOD, focusing on ID performance is a blind alley since this goal requires to increase ID performance at times, and reduce it at other times.

Future achievable OOD performance.

As obvious as it is, it feels necessary to point out that empirical studies only chart regimes achievable with existing methods. Observations have limited predictive power, hence more care seems warranted when deriving prescriptive recommendations from empirical evidence.

Evidently, the best possible performance on Camelyon17 is not limited to the Pareto front observed in our experiments. For example, the current state of the art on this dataset (Robey et al., 2021; WILDS Leaderboard) injects additional task-relevant knowledge to bypass the under/misspecification of the ERM objective, and exceeds both our highest ID and OOD performance. The important message remains that a given hypothesis class (DenseNet architecture in our case) admits parametrizations whose ID and OOD performance do not necessarily correlate.

- Possible invalidation of existing studies.

The possibility of inverse correlations may invalidate studies that implicitly assume a positive one. For example, Angarano et al. (2022) evaluate the OOD robustness of backbone computer vision architectures. They find that modern architectures surpass domain generalization (DG) methods. However, they discard any model but those with the highest ID performance, a.k.a. "training domain validation" in Gulrajani & Lopez-Paz (2021). This means that any model with a high OOD performance but non-optimal ID is ignored. They also train every model for a fixed, large number of epochs (30). This may additionally prevent from finding models with high OOD performance since robustness is progressively lost during fine-tuning (Andreassen et al., 2021).

By design, this study is therefore incapable of finding OOD benefits of any architecture/hyperparameters/DG method that requires trading off some ID performance. Most importantly, once the implicit assumption of a positive correlation is enacted by throwing away models with non-maximal ID performance, there is no more opportunity to demonstrate its validity.

# 8 DISCUSSION

This paper showed that inverse correlations between ID/OOD performance are possible not only theoretically, but also happen in real-world data. We do not know how frequent this situation is. Although we examined a single counterexample, we also showed that past studies may have systematically overlooked such cases. This suffices to show that one cannot know a priori where a given task falls on the spectrum of Figure 6. It is thus ill-advised to blindly make the assumption of a positive correlation, which was suggested in the past.

Can we avoid inverse correlations with a larger training set? Scaling alone without data curation seems unlikely to prevent inverse correlations. Fang et al. (2022) examined a more general question and determined that the impressive robustness of the large vision-and-language model CLIP is determined by the distribution of its training data rather than its quantity. Similarly, inverse correlations stem from biases in the training distribution (e.g. a class  $\mathcal{C}_1$  appearing more frequently with image background  $\mathcal{B}_1$  than any other). And biases in a distribution do not vanish with more i.i.d. samples. Indeed, more data can cover more of the support of the distribution, but this coverage will remain uneven, i.e. biased. The problem can become one of "subpopulation shift" (Santurkar et al., 2020) rather than distribution shift, but it remains similarly challenging.

Training full networks with a diversity-inducing method. We showed inverse correlations with standard ERM models and - even more strikingly - with linear classifiers trained with a diversity-inducing method (Teney et al., 2022a). To the best of our knowledge, this method has not been applied to deep models because of its computational expense. It would be interesting to confirm our observations on networks trained entirely with this or other diversity-inducing methods (Section 3).

Qualitative differences along the Pareto frontier. We focused on quantitative performance. Interpretability methods could also be used to examine whether models of various ID/OOD trade-offs rely on different features and generalization strategies, as done in NLP by Juneja et al. (2022).

Recent work by Eastwood et al. (2022) also recognizes issues of model selection for OOD generalization. They get around selection based on either ID or OOD validation performance with a new domain generalization method (Quantile Risk Minimization) with a tunable trade-off. Other exciting advances by Wang & Veitch (2022) examine existing approaches to improve OOD performance with their suitability to different distribution shifts and underlying causal structures.

# REFERENCES

Anders Andreassen, Yasaman Bahri, Behnam Neyshabur, and Rebecca Roelofs. The evolution of out-of-distribution robustness throughout fine-tuning. arXiv preprint arXiv:2106.15831, 2021.  
Simone Angarano, Mauro Martini, Francesco Salvetti, Vittorio Mazzia, and Marcello Chiaberge. Back-to-bones: Rediscovering the role of backbones in domain generalization. arXiv preprint arXiv:2209.01121, 2022.  
Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.  
Alexander D'Amour, Katherine Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina Chen, Jonathan Deaton, Jacob Eisenstein, Matthew D Hoffman, et al. Under-specification presents challenges for credibility in modern machine learning. arXiv preprint arXiv:2011.03395, 2020.  
Josip Djolonga, Jessica Yung, Michael Tschannen, Rob Romijnders, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Matthias Minderer, Alexander D'Amour, Dan Moldovan, et al. On robustness and transferability of convolutional neural networks. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2021.  
Cian Eastwood, Alexander Robey, Shashank Singh, Julius von Kügelgen, Hamed Hassani, George J Pappas, and Bernhard Schölkopf. Probable domain generalization via quantile risk minimization. arXiv preprint arXiv:2207.09944, 2022.  
We only ran our experiments on Camelyon17. The dataset was not picked post hoc because of unusual results.

Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave, and Ludwig Schmidt. Data determines distributional robustness in contrastive language image pretraining (CLIP). arXiv preprint arXiv:2205.01397, 2022.  
Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In Proc. Int. Conf. Learn. Representations, 2021.  
Ari Heljakka, Martin Trapp, Juho Kannala, and Arno Solin. Representational multiplicity should be exposed, not eliminated. arXiv preprint arXiv:2206.08890, 2022.  
Jeevesh Juneja, Rachit Bansal, Kyunghyun Cho, Joao Sedoc, and Naomi Saphra. Linear connectivity reveals generalization strategies. arXiv preprint arXiv:2205.12411, 2022.  
Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, et al. WILDS: A benchmark of in-the-wild distribution shifts. In Proc. Int. Conf. Mach. Learn., 2021.  
Yoonho Lee, Huaxiu Yao, and Chelsea Finn. Diversify and disambiguate: Learning from underspecified data. arXiv preprint arXiv:2202.03418, 2022.  
Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M Hospedales. Deeper, broader and artier domain generalization. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pp. 5542-5550, 2017.  
Horia Mania and Suvrit Sra. Why do classifier accuracies show linear trends under distribution shift? arXiv preprint arXiv:2012.15483, 2020.  
John Miller, Karl Krauth, Benjamin Recht, and Ludwig Schmidt. The effect of natural distribution shift on question answering models. In Proc. Int. Conf. Mach. Learn., 2020.  
John P Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh, Vaishaal Shankar, Percy Liang, Yair Carmon, and Ludwig Schmidt. Accuracy on the line: on the strong correlation between out-of-distribution and in-distribution generalization. In Proc. Int. Conf. Mach. Learn., 2021.  
Guillermo Ortiz-Jimenez, Itamar Franco Salazar-Reque, Apostolos Modas, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard. A neural anisotropic view of underspecification in deep learning. In Proc. Int. Conf. Learn. Representations, 2021.  
Matteo Pagliardini, Martin Jaggi, François Fleuret, and Sai Praneeth Karimireddy. Agree to disagree: Diversity through disagreement for better transferability. arXiv preprint arXiv:2202.04414, 2022.  
Francesco Pinto, Philip HS Torr, and Puneet K Dokania. An impartial take to the CNN vs transformer robustness contest. arXiv preprint arXiv:2207.11347, 2022.  
Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. DoImagenet classifiers generalize toImagenet? In Proc. Int. Conf. Mach. Learn., 2019.  
Alexander Robey, George J Pappas, and Hamed Hassani. Model-based domain generalization. Proc. Advances in Neural Inf. Process. Syst., 2021.  
Elan Rosenfeld, Pradeep Ravikumar, and Andrej Risteski. The risks of invariant risk minimization. arXiv preprint arXiv:2010.05761, 2020.  
Andrew Ross, Weiwei Pan, Leo Celi, and Finale Doshi-Velez. Ensembles of locally independent prediction models. In Proc. Conf. AAAI, 2020.  
Andrew Slavin Ross, Weiwei Pan, and Finale Doshi-Velez. Learning qualitatively diverse and interpretable rules for classification. arXiv preprint arXiv:1806.08716, 2018.  
Shibani Santurkar, Dimitris Tsipras, and Aleksander Madry. BREEDS: Benchmarks for subpopulation shift. arXiv preprint arXiv:2008.04859, 2020.

Luca Scimeca, Seong Joon Oh, Sanghyuk Chun, Michael Poli, and Sangdoo Yun. Which shortcut cues will dnns choose? a study from the parameter-space perspective. In Proc. Int. Conf. Learn. Representations, 2022.  
Harshay Shah, Kaustav Tamuly, Aditi Raghunathan, Prateek Jain, and Praneeth Netrapalli. The pitfalls of simplicity bias in neural networks. arXiv preprint arXiv:2006.07710, 2020.  
Hidetoshi Shimodaira. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of statistical planning and inference, 2000.  
Rohan Taori, Achal Dave, Vaishaal Shankar, Nicholas Carlini, Benjamin Recht, and Ludwig Schmidt. Measuring robustness to natural distribution shifts in image classification. Proc. Advances in Neural Inf. Process. Syst., 2020.  
Damien Teney and Martial Hebert. Learning to extract motion from videos in convolutional neural networks. In *Asian Conference on Computer Vision*, pp. 412-428. Springer, 2016.  
Damien Teney, Ehsan Abbasnedjad, and Anton van den Hengel. Learning what makes a difference from counterfactual examples and gradient supervision. arXiv preprint arXiv:2004.09034, 2020a.  
Damien Teney, Kushal Kafle, Robik Shrestha, Ehsan Abbasnejad, Christopher Kanan, and Anton van den Hengel. On the value of out-of-distribution testing: An example of Goodhart's law. In Proc. Advances in Neural Inf. Process. Syst., 2020b.  
Damien Teney, Ehsan Abbasnejad, Simon Lucey, and Anton van den Hengel. Evading the simplicity bias: Training a diverse set of models discovers solutions with superior OOD generalization. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2022a.  
Damien Teney, Maxime Peyrard, and Ehsan Abbasnejad. Predicting is not understanding: Recognizing and addressing underspecification in machine learning. arXiv preprint arXiv:2207.02598, 2022b.  
Zihao Wang and Victor Veitch. A unified causal view of domain invariant representation learning. arXiv preprint arXiv:2208.06987, 2022.  
Florian Wenzel, Andrea Dittadi, Peter Vincent Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, et al. Assaying out-of-distribution generalization in transfer learning. arXiv preprint arXiv:2207.09239, 2022.  
WILDS Leaderboard. https://wilds.stanford.edu/leaderboard/.  
Olivia Wiles, Sven Gowal, Florian Stimberg, Sylvestre Alvise-Rebuffi, Ira Ktena, Taylan Cemgil, et al. A fine-grained analysis on distribution shift. arXiv preprint arXiv:2110.11328, 2021.  
Shingo Yashima, Teppei Suzuki, Kohta Ishikawa, Ikuro Sato, and Rei Kawakami. Feature space particle inference for neural network ensembles. arXiv preprint arXiv:2206.00944, 2022.  
Nanyang Ye, Kaican Li, Haoyue Bai, Runpeng Yu, Lanqing Hong, Fengwei Zhou, Zhenguo Li, and Jun Zhu. Ood-bench: Quantifying and understanding two dimensions of out-of-distribution generalization. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., 2022.  
Xiao Zhou, Yong Lin, Weizhong Zhang, and Tong Zhang. Sparse invariant risk minimization. In Proc. Int. Conf. Mach. Learn., 2022.

![](images/4a3a741eab4cd4cb5b5eeec03d8cbcdc25024d1aa291b6af9987b010fb749cc2.jpg)  
A ADDITIONAL RESULTS

![](images/2390d7d110e8aeef24909b6219b16aa08bf994f1af4c5360af0f8d9ba9168477.jpg)

![](images/26e02a25d06703d6005f8f98f7fae03ef0b1fac206aceb5a17969700a0e06e1f.jpg)

![](images/001015d3dc513510744e404fea16765d06aa1c84915d0c46d52ef998fcc5ac8d.jpg)

![](images/9e6d5dada2fc4368f8093b1d70479f0565c727ca98bd71d9a284fb650bc9c954.jpg)

![](images/f92bf5d15efc5384e62709d3c208cf111b045552bc1e7e1711372b538b5974b3.jpg)  
Figure 7: As in Figure 4, we show that higher OOD accuracy can be sometimes be traded off for a lower ID accuracy. Each panel shows results from a different pretrained model (i.e. pretrained with a different random seed). Each dot represents a linear classifier re-trained on features from this pretrained model with standard ERM (red dots  $\bullet$ ) or with a diversity-inducing method (Teney et al., 2022a) (gray dots  $\bullet$ ). The latter set includes models with higher OOD/ lower ID accuracies.

![](images/3d0fe1df37facae9bc35b6850de6e032f0fb95ca4910881b7a8eef3f60292796.jpg)

![](images/2fb8391d728e7d4eee81706f46ee2f6bda37ea3122cfce29a73a22285c04df05.jpg)

![](images/b9136fd125510578ee02e4f5ca914390fff4a6bfb181815fee08116082aab41b.jpg)

![](images/858e20122da372fd2c3bee0d4f7f77d0ac5063aa2f1e0534748ce161e9b940da.jpg)

![](images/5e91bd9643710fe5e0b2f306702754510d94ba7329e2dca2c5b7eaf35c098268.jpg)

![](images/de73b4f408748d87f15adb8f07ddfcaae3ddf5267e76ab806856aa0eef2c49e5.jpg)

![](images/058b58605bfbff0673288b693cb2400109eb39c9dbb058f64b32f210752c4ef6.jpg)

![](images/08858a5cdcb6927a3854081095a839c0327797e847a648ca53872719f74ad88a.jpg)

![](images/4f513b83e962af3e71a34e33e8c782d34b9a1efbbcdb2f3b4be434c704a8ba29.jpg)

![](images/6760e0c93ff89db318d29af2c5d9a2bdf791cfbeb6fceb7bdaeeb0370b437618.jpg)  
Figure 8: Same as in Figure 7, but using val-ood (instead of test-ood) as the OOD evaluation set.

![](images/ebec0aa16600190669b9787e0a079c2adaac49632ecf15f5f703399c91cab34f.jpg)

![](images/065296c6b9784f6d6fc5cdfd220796b299642edba6a2695c004c5bd0d1b2337c.jpg)

![](images/75c4fd07cc3b4cd92d6f40812391583d1e995e06bb5566449ac0cab3b3361d8f.jpg)

![](images/481670ec71d26c58eda6f487034a8d7e3e17ecb9a8be5e036cc8012ba7b4fa46.jpg)

![](images/d22214e41563699b2fe0d36eb7c295cd9ae6d028f049dc820074404b5c9d3782.jpg)

![](images/726c773904b3234911b7513650eb2819ee169b88cb1face7e38b03e9a13cf406.jpg)

![](images/812f309cfb5bfd07da4aa3f4a7c35d818829c9d983822359f86402394bf31db0.jpg)

![](images/727322f636740cfd131c248aae7814c0a390fbca0414e11d02d6d8ce293cbdd3.jpg)

![](images/59f607a883582eb652e2825f9814cf0152a4ad0e093b01ffbb1a2923cf7d2f07.jpg)

![](images/d705f5a721117bf80ced836a6e760589781a6680cb393263138909142f93957a.jpg)  
Figure 9: Same as in Figure 8, zoomed-in on ERM models (red dots  $\bullet$ ).

![](images/b4c2bf07e9cd10e4f094eb1b258cf28fc820e38050b05bf54795d8a4df8ca323.jpg)

![](images/a1aa8eab7090a14a098374eaeb2bc6fec4231fc7de01d35f9ae7392a13b420ff.jpg)

![](images/450928bed6aaabf1631ddb9ec73ae736e5d4174631cceee41477bfe0d11b0241.jpg)

![](images/9a251b340c17f3e048fdd9e75ab78de3bfff5256334e577edf1747e6bda9d4de.jpg)
