# UNDERSTANDING GENERALIZED LABEL SMOOTHING WHEN LEARNING WITH NOISY LABELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Label smoothing (LS) is an arising learning paradigm that uses the positively weighted average of both the hard training labels and uniformly distributed soft labels. It was shown that LS serves as a regularizer for training data with hard labels and therefore improves the generalization of the model. Later it was reported LS even helps with improving robustness when learning with noisy labels. However, we observe that the advantage of LS vanishes when we operate in a high label noise regime. Puzzled by the observation, we proceeded to discover that several proposed learning-with-noisy-labels solutions in the literature instead relate more closely to negative label smoothing (NLS), which defines as using a negative weight to combine the hard and soft labels! We show that NLS differs substantially from LS in their achieved model confidence. To differentiate the two cases, we will call LS the positive label smoothing (PLS), and this paper unifies PLS and NLS into generalized label smoothing (GLS). We provide understandings for the properties of GLS when learning with noisy labels. Among other established properties, we theoretically show NLS is considered more beneficial when the label noise rates are high. We provide extensive experimental results on multiple benchmarks to support our findings too.

# 1 INTRODUCTION

Label smoothing (LS) (Szegedy et al., 2016) is an arising learning paradigm that uses positively weighted average of both the hard training labels and uniformly distributed soft label:

$$
\mathbf {y} ^ {\mathrm {L S}, r} = (1 - r) \cdot \mathbf {y} + \frac {r}{K} \cdot \mathbf {1} \tag {1}
$$

where we denote the one-hot vector form of hard label and an all one vector as  $\mathbf{y}$ ,  $\mathbf{1}$  respectively.  $K$  is the number of label classes, and  $r$  is the smooth rate in the range of [0, 1]. It was shown that LS serves as a regularizer for the hard training data and therefore improves generalization of the model. The regularizer role of LS prevents the model from fitting overly on the target class. Empirical studies have demonstrated the effectiveness of LS in improving the model performance across various benchmarks (Pereyra et al., 2017) (such as image classification (Szegedy et al., 2016), machine translation (Vaswani et al., 2017), language modelling (Chorowski & Jaitly, 2017)) and model calibration (Müller et al., 2019).

Later it was reported LS even helps with improving robustness when learning with noisy labels (Lukasik et al., 2020). However, we observe that the advantage of LS vanishes when we operate in a high label noise regime. In Figure 1, we present a set of experiments on UCI datasets (Dua & Graff, 2017). We highlight best two smooth rates (possible to have tied smooth rates) under each label noise rate. Indeed, non-negative smooth rates (circles colored in red) outperform negative ones when the label noise rates are low. Nonetheless, with the increasing of noise rates, negative smooth rates  $r < 0$  (Eqn. (1), diamonds colored in green) appear to be more competitive when learning with noisy labels. Puzzled by the observation, we proceeded to discover

that several proposed learning-with-noisy-labels solutions in the literature, including Loss Correction (Patrini et al., 2017), NLNL (Kim et al., 2019) and Peer Loss (Liu & Guo, 2020), instead relate more closely to negative label smoothing (NLS), which defines as using a negative weight to combine the hard and soft labels!

![](images/608d002fe55f96d93df8b352cb17853f1282d17223997adb39cf9a47a06f912b.jpg)  
Figure 1: Optimal smooth rates on UCI datasets with different label noise rates.

This paper unifies label smoothing with either positive or negative smooth rate into a generalized label smoothing (GLS) framework. Our paper is motivated by the above inconsistent observations. And we aim to provide a more thorough understanding of GLS under the setting of learning with noisy labels, rather than proposing a new method to compete for state-of-the-art performances. We first show that negative label smoothing differs substantially from positive label smoothing in their achieved model confidence. With the presence of label noise, we then proceed to show that there exists a phase transition behavior when finding the optimal label smoothing rate for GLS. Particularly, when label noise is low, positive label smoothing is able to uncover the optimal model while negative label smoothing is considered more beneficial in a high label noise regime.

We provide extensive experimental evidences to support our findings. For instance, on multiple benchmark datasets, we present the clear transition of the optimal smoothing rate going from positive to negative when we keep increasing noise rates. On CIFAR-10 test data, we show a negative smoothing rate elicits higher model confidence on correct predictions and lower confidence on wrong predictions compared with the behavior of a positive one.

Our contributions summarize as follows:

- We provide understandings for a generalized notion of label smoothing (GLS) when learning with noisy labels, where the label smooth rate can go negative. (An understanding paper rather than a method paper)  
- We show that several robust loss functions in the noise learning literature correspond to learning with GLS, under certain noise rate models. (Section 3)  
- We theoretically show that negative label smoothing improves the expected model confidence over the data distribution. With the presence of label noise, we demonstrate learning with a negative smooth rate can be more robust to label noise compared with a positive rate when label noise rates are high. (Section 4, 5)  
- Empirical experiments on multiple benchmark datasets demonstrate that with the presence of label noise, NLS becomes competitively robust to label noise. We also empirically show how GLS results in trade-offs in model confidence, bias and variance of the generalization error. (Section 6, Appendix D and E)  
- Extensive experiment results validate our main theoretical conclusions. Besides, we also discuss practical considerations and multi-class extensions of GLS to mitigate the impact of label noise. (Appendix B, C)

We defer all proofs to Appendix F. Our work primarily contributes to the literature of learning with noisy labels (Scott et al., 2013; Natarajan et al., 2013; Liu & Tao, 2015; Patrini et al., 2017; Kim et al., 2019; Liu & Guo, 2020). Our core results are contingent on recent works of understanding the effect of label smoothing when training deep neural network models (Müller et al., 2019; Li et al., 2020; Yuan et al., 2020; Xu et al., 2020; Chen et al., 2021; Gao et al., 2020; Chen et al., 2020; Lee & Cheon, 2020; Liu, 2021), and in particular

when label noise presents (Lukasik et al., 2020). We generalize a concept called generalized label smoothing and provide new understandings for when, instead of setting a positive smoothing rate as the literature would normally do, a negative smoothing rate is considered a better option. Due to the space limit, we defer a more detailed discussion of related works to Appendix A.

# 2 PRELIMINARIES

# 2.1 LEARNING WITH SMOOTHED LABELS

For a  $K$ -class classification task, we denote by  $X \in \mathcal{X}$  a high-dimensional feature and  $Y \in \mathcal{Y} \coloneqq \{1, 2, \dots, K\}$  the corresponding label. Suppose  $(X,Y) \in \mathcal{X} \times \mathcal{Y}$  are drawn from a joint distribution  $\mathcal{D}$ . Let  $\mathbf{y_i}$  be the one-hot encoded vector form of  $y_i$  which generates according to  $Y$ . The random variable of smoothed label  $Y^{\mathrm{LS},r}$  with smooth rate  $r \in [0,1]$  generates  $\mathbf{y_i}^{\mathrm{LS},r}$  as (Szegedy et al., 2016):

$$
\mathbf {y} _ {\mathbf {i}} ^ {\mathrm {L S}, r} = (1 - r) \cdot \mathbf {y} _ {\mathbf {i}} + \frac {r}{K} \cdot \mathbf {1}.
$$

For example, when  $r = 0.3$ , the smoothed label of  $\mathbf{y_i} = [0,1,0]$  becomes  $\mathbf{y_i^{LS,r = 0.3}} = [0.1,0.8,0.1]$ .

We consider a broader setting where the smoothed label might be negatively related to the corresponding feature. As a supplementary to existed works on label smoothing, we explore the benefits of learning with generalized label smoothing (GLS), i.e.,  $r \in (-\infty, 1]$  instead of a non-negative  $r$ .

$$
\mathbf {y} _ {\mathbf {i}} ^ {\mathrm {G L S}, r} := (1 - r) \cdot \mathbf {y} _ {\mathbf {i}} + \frac {r}{K} \cdot \mathbf {1} \tag {2}
$$

where  $\mathbf{y}_{\mathbf{i}}^{\mathrm{GLS},r}$  is given by the random variable of generalized smooth label  $Y^{\mathrm{GLS},r}$ . We name the scenario  $r < 0$  as negative label smoothing (NLS). To clarify, we don't assume a strict lower bound for  $r$ . If  $r \to -\infty$ , normalizing  $\mathbf{y}_{\mathbf{i}}^{\mathrm{GLS},r}$  by  $1 - r$  returns  $\mathbf{y}_{\mathbf{i}}^{\mathrm{GLS},r} = \mathbf{y}_{\mathbf{i}} - 1 / K$ . We will show when imposing a negative smoothing parameter will be considered beneficial as compared to a positive one. In the main paper, we focus on the binary classification task where  $y_{i} \in \{0,1\}$  and  $K = 2$ . And we defer multi-class extensions to Appendix C. Denote  $f$  as a deep neural network,  $\mathbf{f}(\mathbf{x}_{\mathbf{i}})$  is the model prediction of  $x_{i} \in X$  with element  $\mathbf{f}(\mathbf{x}_{\mathbf{i}})_{y_i} := \mathbb{P}(Y = y_i | X = x_i, f)$ . Given the sample  $x \in \mathcal{X}$  and a hard label  $y \in \mathcal{V}$ , binary CE loss is defined as  $\ell_{\mathrm{CE}}(\mathbf{f}(\mathbf{x}), y) := -\log (\mathbf{f}(\mathbf{x})_y)$ . Throughout this paper, we shorthand  $\ell$  as  $\ell_{\mathrm{CE}}$  for a clean presentation.

# 2.2 LEARNING WITH NOISY LABELS

The noisy label literature considers the setting where we only have access to samples with noisy labels from  $(X, \tilde{Y})$ . Suppose random variables  $(X, \tilde{Y}) \in \mathcal{X} \times \tilde{\mathcal{V}}$  are drawn from a noisy joint distribution  $\tilde{\mathcal{D}}$ . Statistically, the random variable of noisy labels  $\tilde{Y}$  can be characterized by a noise transition matrix  $T$ , where each element  $T_{i,j}$  represents the probability of flipping the clean label  $Y = i$  to the noisy label  $\tilde{Y} = j$ , i.e.,  $T_{ij} = \mathbb{P}(\tilde{Y} = j | Y = i)$ . In this paper, we are interested in the widely studied class-dependent label noise. We assume the label noise is conditionally independent of features, i.e.,

$$
\mathbb {P} (\tilde {Y} = j | Y = i) = \mathbb {P} (\tilde {Y} = j | X, Y = i), \forall i, j \in [ K ].
$$

For the binary classification setting, define  $e_0 \coloneqq \mathbb{P}(\tilde{Y} = 1|Y = 0)$ ,  $e_1 \coloneqq \mathbb{P}(\tilde{Y} = 0|Y = 1)$ . Without loss of generality, we assume  $e_1 - e_0 = e_\Delta \geq 0$ . We denote the binary noise transition matrix in the noisy label setting as:  $T = \left( \begin{array}{cc} 1 - e_0 & e_0 \\ e_1 & 1 - e_1 \end{array} \right)$ .

# 2.3 MODEL CONFIDENCE

We define a key quantity, model confidence, that plays an important role in later sections.

Definition 1. Model confidence of model  $f$  for sample  $(x, y)$ . Given a model  $f$ , a sample  $x$  with its target label  $y \in \{0, 1\}$ , the model confidence of  $f$  w.r.t. sample  $x$  is defined as  $MC(f; x, y) = \mathbf{f}(\mathbf{x})_y - \mathbf{f}(\mathbf{x})_{1 - y}$ .

$\mathrm{MC}(f;x,y)$  in definition 1 characterizes the difference of the predicted probability between target class and the other class.  $\mathrm{MC}(f;x,y) = 0$  simply means  $f$  has no confident on its predictions since the model can not identify the target class of  $x$ .  $\mathrm{MC}(f;x,y)$  returns a negative value when  $f$  gives a wrong prediction and is not confident to predict the label of  $x$  as the target label  $y$ . To dig into how GLS influences the model confidence on correct and wrong predictions in following sections, we separate the distribution  $\mathcal{D}$  into

$$
\mathcal {D} _ {f} ^ {+} := \left\{\left(X, Y\right) \sim \mathcal {D}: \operatorname {M C} (f; X, Y) > 0 \right\}, \quad \mathcal {D} _ {f} ^ {-} := \left\{\left(X, Y\right) \sim \mathcal {D}: \operatorname {M C} (f; X, Y) \leq 0 \right\}.
$$

# 3 CONNECTION TO OTHER ROBUST METHODS

In this section, we aim to theoretically explore the connection between GLS and popular methods such as backward/forward loss correction (Natarajan et al., 2013; Patrini et al., 2017), NLNL (Kim et al., 2019) and peer loss (Liu & Guo, 2020). We defer the corresponding empirical validations to Appendix B.

For  $r \leq 1$ , let  $\tilde{\mathbf{y}}$  be the vector form of noisy label  $\tilde{y}$  obtained from  $\tilde{Y}$ , we define the  $r$  smoothed label of  $\tilde{y}$  as  $\tilde{\mathbf{y}}^{\mathrm{GLS},r}$ , where  $\tilde{\mathbf{y}}^{\mathrm{GLS},r} := (1 - r) \cdot \tilde{\mathbf{y}} + (r / K) \cdot \mathbf{1}$  and is generated by the random variable  $\tilde{Y}^{\mathrm{GLS},r}$ . Risk minimization of the Generalized Label Smoothing (GLS) w.r.t. noisy labels becomes:

$$
\text {R i s k M i n i m i z a t i o n U s i n g G L S :} \quad \min  _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} \left[ \ell (\mathbf {f} (\mathbf {X}), \tilde {Y} ^ {\mathrm {G L S}, r}) \right]. \tag {3}
$$

The GLS framework covers three special methods: PLS  $(r\in (0,1])$ , Vanilla (CE) Loss  $(r = 0)$  and NLS  $(r < 0)$ . Besides, we observe that NLS connects to a special case of label smoothing regularization. We highlight this in Theorem 1.

Theorem 1.  $\forall r\in [0,1]$  NLS with smooth rate  $-r$  is a special form of label smoothing regularization:

$$
\min  \mathbb {E} _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} \Big [ \ell (\mathbf {f} (\mathbf {X}), \tilde {Y} ^ {G L S, - r}) \Big ] = \min  \mathbb {E} _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} \Big [ 2 \cdot \ell (\mathbf {f} (\mathbf {X}), \tilde {Y}) - \ell (\mathbf {f} (\mathbf {X}), \tilde {Y} ^ {G L S, r}) \Big ].
$$

# 3.1 LOSS CORRECTION

Loss correction (Patrini et al., 2017) studies two robust loss designs which are based on the knowledge of nonsingular noise transition matrix  $T$ . The backward correction  $\ell^{\leftarrow}(\mathbf{f}(\mathbf{X}),\tilde{Y})$  re-weights the loss  $\ell (\mathbf{f}(\mathbf{X}),\tilde{Y})$  by  $T_{\hat{Y},\hat{Y}}^{-1}$  with  $\hat{Y}$  being the model predicted label, while the proposed forward correction  $\ell^{\rightarrow}(\mathbf{f}(\mathbf{X}),\tilde{Y})$  multiplies the model predictions by  $T$ .

Proposition 1. For  $r_{LC} \coloneqq \frac{2e_0}{2e_0 - 1} < 0$ ,  $\lambda_{LC} \coloneqq e_{\Delta} \cdot \frac{1}{1 - 2e_0}$ , risk minimization of both backward and forward correction (with the knowledge of noise rates) are equivalent to the combination of NLS and an extra bias term Bias-LC

$$
\begin{array}{l} \min  \mathbb {E} _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} \left[ \ell^ {\leftarrow} (\mathbf {f} (\mathbf {X}), \tilde {Y}) \right] = \min  \mathbb {E} _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} \left[ \ell^ {\rightarrow} (\mathbf {f} (\mathbf {X}), \tilde {Y}) \right] \\ = \min  \mathbb {E} _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} \left[ \ell (\mathbf {f} (\mathbf {X}), \tilde {Y} ^ {G L S, r _ {L C}}) \right] + \lambda_ {L C} \cdot \underbrace {\mathbb {E} _ {X , Y = 1} \left[ \ell (\mathbf {f} (\mathbf {X}) , 1) - \ell (\mathbf {f} (\mathbf {X}) , 0) \right]} _ {B i a s - L C}. \\ \end{array}
$$

The incurred Bias-LC controls the model confidence on  $(X,Y = 1)\sim \mathcal{D}_f$ . Note that when the noise rate is not substantially high, i.e.,  $e_0\in [0,1 / 2)$ ,  $\lambda_{\mathrm{LC}} > 0$ . Then, compared with loss correction, NLS with smooth rate  $r_{\mathrm{LC}}$  makes the model  $f$  to be less confident on  $(X,Y = 1)\sim \mathcal{D}_f^+$  and more confident on  $(X,Y = 1)\sim \mathcal{D}_f^-$  (wrong predictions). However, the impact of term Bias-LC is diminishing when either  $e_{\Delta}\rightarrow 0$  (symmetric noise rates) or  $e_0\rightarrow 0$  (low noise rates) as specified in Theorem 2.

Theorem 2. Assume the noise transition matrix is symmetric, i.e.,  $e_{\Delta} = 0$ , backward and forward loss correction are a special form of NLS with smooth rate  $r_{LC}$ .

# 3.2 LEARNING FROM COMPLEMENTARY LABELS

Complementary label (Ishida et al., 2017) was firstly introduced to mitigate the cost of collecting data. Rather than encouraging the model to fit directly on the target, learning from complementary labels trains the model to not fit on the complementary label which differs from the target. Later, an indirect training method "Negative Learning" (NL) (Kim et al., 2019) was proposed to reduce the risk of providing incorrect information with the presence of noisy labels and is robust to label noise in multi-class classification tasks. A more generic unbiased risk estimator of learning with complementary labels was proposed (Ishida et al., 2019) and is defined as:  $\ell_{\mathrm{CL}}(\mathbf{f}(\mathbf{X}),\tilde{Y})\coloneqq \ell (\mathbf{f}(\mathbf{X}),\tilde{Y}) - \ell (\mathbf{f}(\mathbf{X}),1 - \tilde{Y})$

Theorem 3. Learning from complementary labels with  $\ell_{CL}$  is equivalent to NLS with smooth rate  $r_{CL} \to -\infty$ :

$$
\min \mathbb {E} _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} \Big [ \ell_ {C L} (\mathbf {f} (\mathbf {X}), \tilde {Y}) \Big ] \Longleftrightarrow \min \mathbb {E} _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} [ \ell (\mathbf {f} (\mathbf {X}), \tilde {Y} ^ {G L S, r _ {C L} \to - \infty}) ].
$$

# 3.3 PEER LOSS FUNCTIONS

Peer loss functions (Liu & Guo, 2020) propose a family of robust loss measures which do not require the knowledge of noise rates. The mathematical representation of peer loss functions is  $\ell_{\mathrm{PL}}(\mathbf{f}(\mathbf{X}),\tilde{Y})\coloneqq$ $\ell (\mathbf{f}(\mathbf{X}),\tilde{Y}) - \ell (\mathbf{f}(\mathbf{X}_1),\tilde{Y}_2)$ , where  $(X_{i},\tilde{Y}_{i})\sim \tilde{\mathcal{D}}$ . The second term of peer loss evaluates on randomly paired data samples and labels to punish  $f$  from overly fitting on noisy labels.

Proposition 2. For  $r_{PL} \coloneqq 2 \cdot \mathbb{P}(\tilde{Y} = 1)$ ,  $\lambda_{PL} \coloneqq 1 - r_{PL}$ , risk minimization of peer loss is equivalent to negative label smoothing regularization with an extra term Bias-PL, i.e.,

$$
\begin{array}{l} \min  \mathbb {E} _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} \left[ \ell_ {P L} (\mathbf {f} (\mathbf {X}), \tilde {Y}) \right] = \min  \mathbb {E} _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} \left[ \ell (\mathbf {f} (\mathbf {X}), \tilde {Y}) - \ell (\mathbf {f} (\mathbf {X}), \tilde {Y} ^ {G L S, r _ {P L}}) \right] \\ + \lambda_ {P L} \cdot \underbrace {\mathbb {E} _ {X , \tilde {Y} = 1} \left[ \ell (\mathbf {f} (\mathbf {X}) , 1) - \ell (\mathbf {f} (\mathbf {X}) , 0) \right]} _ {B i a s - P L}. \\ \end{array}
$$

The incurred term Bias-PL controls the model confidence on  $(X, \hat{Y} = 1) \sim \tilde{\mathcal{D}}$  and has a diminishing effect as  $\mathbb{P}(\tilde{Y} = 1) \to 1/2$ . Generally, peer loss relates to GLS as the negatively weighted GLS term appears to be a regularizer. Note that we have access to the  $\mathbb{P}(\tilde{Y} = 1)$ , we can bridge the gap between GLS and peer loss by adding an estimable term Bias-PL. With some derivations, we further show in Theorem 4, when noisy priors are equal, peer loss has an exact GLS form.

Theorem 4. When the noisy labels have equal prior, i.e.,  $\mathbb{P}(\tilde{Y} = 0) = \mathbb{P}(\tilde{Y} = 1)$ , peer loss is a special form of NLS regularization with smooth rate  $r_{PL}$ . Besides,

$$
\min  \mathbb {E} _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} \left[ \ell_ {P L} (\mathbf {f} (\mathbf {X}), \tilde {Y}) \right] \Longleftrightarrow \min  \mathbb {E} _ {(X, \tilde {Y}) \sim \tilde {\mathcal {D}}} \left[ \ell (\mathbf {f} (\mathbf {X}), \tilde {Y} ^ {G L S, r \rightarrow - \infty}) \right].
$$

# 4 GLS AND MODEL CONFIDENCE

Now we show that NLS differs substantially from PLS in their achieved model confidence. The observation holds true for any dataset and we will present our findings when learning with clean data. This discussion sets the foundation for our discussion when learning with noisy labels in next section.

When the label is clean, i.e.,  $e_0 = e_1 = 0$ , Eqn. (3) reduces to:

$$
\min  \mathbb {E} _ {(X, Y) \sim \mathcal {D}} \left[ \ell (\mathbf {f} (\mathbf {X}), Y) \right] + \frac {r}{2} \cdot \mathbb {E} _ {(X, Y) \sim \mathcal {D}} \overbrace {\left[ \ell (\mathbf {f} (\mathbf {X}) , 1 - Y) - \ell (\mathbf {f} (\mathbf {X}) , Y) \right]} ^ {\text {T e r m M C} _ {\ell} (f; X, Y)}. \tag {4}
$$

It is clear that the difference between PLS and NLS lie in the weight of Term  $\mathbf{MC}_{\ell}(f;X,Y)$ . NLS encourages a high  $\mathbf{MC}_{\ell}(f;X,Y)$  in expectation while PLS has an opposite purpose. For any  $r_{\mathrm{N}} < 0$ ,  $r_{\mathrm{P}} \in (0,1)$ , denote by  $f_{r_{\mathrm{N}}}^{*}$ ,  $f_{r_{\mathrm{P}}}^{*}$  the optimal classifiers of Eqn. (4) learned with  $Y^{\mathrm{GLS},r_{\mathrm{N}}}$  and  $Y^{\mathrm{GLS},r_{\mathrm{P}}}$  respectively, we show that  $f_{r_{\mathrm{N}}}^{*}$  returns a higher expected model confidence compared with  $f_{r_{\mathrm{P}}}^{*}$  in Theorem 5.

Theorem 5. If  $f_{r_N}^* \neq f_{r_P}^*$ , we have:  $\mathbb{E}_{(X,Y) \sim \mathcal{D}}[MC(f_{r_N}^*; X, Y)] > \mathbb{E}_{(X,Y) \sim \mathcal{D}}[MC(f_{r_P}^*; X, Y)]$ .

We want to emphasize that the proof of Theorem 5 can be extended to any distribution  $\mathcal{D}$  even when learning with noisy labels, we state this observation in Corollary 1:

Corollary 1. For any data distribution  $\mathcal{D}'$ , the optimal classifier of NLS returns a higher model confidence than that of PLS in expectation.

Recent works (Liu, 2021; Cheng et al., 2020) have demonstrated that with the presence of label noise, learning with noisy labels directly will eventually result in unconfident model predictions. When  $\mathcal{D}' = \hat{\mathcal{D}}$ , will NLS remain confident in her predictions? We defer our answer to Section 5.

From above, we have shown that NLS and PLS have the opposite functionality on the model confidence when training on clean data. Given the unseen test data, learning with non-negative smooth rates may not always return the best outcome and whether NLS could outperform PLS in certain applications or not requires a more thorough study in the view of bias-variance trade-off. We defer our empirical analysis about the role of GLS in the bias-variance of the generalization error to Appendix E, where we observe that for PLS, the overall bias has an increasing tendency while the variance has the decreasing pattern with the increase of smooth rate.

# 5 GLS WITH NOISY LABELS

In this section, we target at the optimal candidates of  $r$  in GLS when the label noise presents. Empirical evidences have shown that learning with positive smoothed labels may result in the performance improvement. However, our observations (details in Appendix E) show that the a higher bias and lower variance incurred by PLS may not yield a better performance than NLS. Based on this observation, we delve into details to show when NLS is more favorable than PLS and Vanilla Loss. We start with stating Assumption 1:

Assumption 1. We assume learning with clean data distribution  $\mathcal{D}$  with smooth rate  $r^* \leq 1$  in GLS returns the best performance on the unseen clean test data distribution  $\mathcal{D}_{test}$ .

Assumption 1 simply offers an "anchor" point to initiate our analysis for the noisy label setting. To clarify, we don't rule out the possibility that other methods outperform GLS with optimal smooth rate  $r^*$ . Later in Section 6.1 and Appendix D, we will empirically test what  $r^*$  usually is on various benchmarks. We define optimal classifier on  $Y^*$  which follows  $r^*$  smooth label distribution as:  $f_{\mathcal{D}}^{*} \coloneqq \arg \min_{f} \mathbb{E}_{(X,Y) \sim \mathcal{D}}\left[\ell(\mathbf{f}(\mathbf{X}), Y^{*})\right]$ . With the introduction of  $r^*$  and  $f_{\mathcal{D}}^{*}$ , our goal is then to recover the classifier  $f$  using the noisy training labels. To bridge learning with noisy labels and clean labels for GLS, we define  $\lambda_1, \lambda_2$  and offer Theorem 6.

$$
\lambda_ {1} := \left[ \left(e _ {0} - \frac {r ^ {*}}{2}\right) + (1 - 2 e _ {0}) \cdot \frac {r}{2} \right], \quad \lambda_ {2} := e _ {\Delta} \cdot (1 - r).
$$

Theorem 6. The risk minimization of GLS (Eqn. (3)) in the noisy setting relates to the risk defined on the clean data with two additional bias terms:

$$
\begin{array}{l} \min  \underbrace {\mathbb {E} _ {(X , Y) \sim \mathcal {D}} \left[ \ell (\mathbf {f} (\mathbf {X}) , Y ^ {*}) \right]} _ {\text {T r u e R i s k}} + \lambda_ {1} \cdot \mathbb {E} _ {(X, Y) \sim \mathcal {D}} \left[ \ell (\mathbf {f} (\mathbf {X}), 1 - Y) - \ell (\mathbf {f} (\mathbf {X}), Y) \right] _ {M - I n c l} \\ \underbrace {+ \lambda_ {2} \cdot \mathbb {E} _ {X , Y = 1} \left[ \ell (\mathbf {f} (\mathbf {X}) , 0) - \ell (\mathbf {f} (\mathbf {X}) , 1) \right]} _ {M - I n c 2}. \tag {5} \\ \end{array}
$$

The True Risk is the risk minimization w.r.t. clean optimal label distribution  $Y^{*}$ . Training GLS on noisy labels results in two extra bias terms which affect the model confidence. We defer an empirical validation of Theorem 6 to Appendix B. Now we proceed to answer "what parameters are preferred in the noisy setting".

# 5.1 SYMMETRIC ERROR RATES WITH  $e_{\Delta} = 0$

Symmetric error rates  $e \coloneqq e_0 = e_1$  indicates the probability of flipping to the other class is equal for both classes. In this case,  $\lambda_{2} = 0$  and Term M-Inc2 is cancelled and Eqn. (5) reduces to

$$
\min  \underbrace {\mathbb {E} _ {(X , Y) \sim \mathcal {D}} \left[ \ell (\mathbf {f} (\mathbf {X}) , Y ^ {*}) \right]} _ {\text {T r u e R i s k}} + \lambda_ {1} \cdot \mathbb {E} _ {(X, Y) \sim \mathcal {D}} \left[ \ell (\mathbf {f} (\mathbf {X}), 1 - Y) - \ell (\mathbf {f} (\mathbf {X}), Y) \right]. \tag {6}
$$

Noisy labels impairs model confidence on Vanilla Loss In the GLS framework, define the optimal  $r$  that will cancel the impact of Term M-Inc1 as:

$$
\text {w h e n} r _ {\text {o p t}} := \frac {r ^ {*} - 2 e}{1 - 2 e}, \quad \mathrm {M} - \operatorname {I n c} 1 = 0. \tag {7}
$$

The threshold  $r_{\mathrm{opt}}$  in Eqn. 7 implies:

Theorem 7. With Assumption 1, GLS with smooth rate  $r = r_{opt}$  yields  $f_{\mathcal{D}}^{*}$ .

- When error rate  $e < r^{*} / 2$ ,  $r = r_{opt} > 0$  (PLS);  
- When error rate  $e = r^{*} / 2$ ,  $r = 0$  (Vanilla Loss);  
- When error rate  $e > r^{*} / 2$ ,  $r = r_{opt} < 0$  (NLS).

In Theorem 7, adopting NLS when noise rate  $e < r^{*} / 2$  induces  $\lambda_{1} < 0$ , Term M-Inc1 makes  $f$  overly-confident on its predictions compared with  $Y^{*}$ . In Figure 2, with the decreasing of  $r^{*}$ , PLS is less tolerant of labels with high noise. Similarly, if  $e \geq \frac{r^{*}}{2}$ , with the decreasing of  $r^{*}$ , NLS is more robust in the high noise regime while PLS makes the model  $f$  become less-confident on its predictions. Clearly, NLS outperforms PLS especially when noise rates are large and  $r^{*}$  is small.

![](images/be8c6755d5defa6b1be63db1568f3037e6d9d65f59844c440c766ae53eb29040.jpg)  
Figure 2: Decision between NLS, PLS given  $e$ ,  $r^*$ .

# 5.2 ASYMMETRIC ERROR RATES WITH  $e_{\Delta} \neq 0$

In this case, adopting  $r = \frac{r^* - 2e_0}{1 - 2e_0}$  removes the Term M-Inc1. However, when  $r < 1$ , Term M-Inc2 is not negligible due to assymmetric noise transition matrix. As a result, Term M-Inc2 becomes:

$$
e _ {\Delta} \cdot \frac {1 - r ^ {*}}{1 - 2 e _ {0}} \cdot \mathbb {E} _ {X, Y = 1} \left[ \ell (\mathbf {f} (\mathbf {X}), 0) - \ell (\mathbf {f} (\mathbf {X}), 1) \right], \quad \text {w i t h} e _ {\Delta} \cdot \frac {1 - r ^ {*}}{1 - 2 e _ {0}} \geq 0.
$$

According to Theorem 5, Term M-Inc2 in the minimization increases the model confidence on  $(X,Y = 0)\sim \mathcal{D}_f^+$ . The model will then become overly-confident with the class that has a low noise rate  $e_0$ . Meanwhile, Term M-Inc2 decreases the model confidence on  $(X,Y = 1)\sim \mathcal{D}_f^+$  (less-confident to the class with a high noise rate  $e_1$ ).

Practical considerations We also discuss practical considerations of GLS in Appendix C, including the estimation of optimal smoothing parameter  $r_{\mathrm{opt}}$ , existed solutions to estimate the noise transition matrix  $T$ , making GLS more robust to label noise by reducing the impact of Term M-Inc2, and how GLS extends to the multi-class setting.

# 6 EXPERIMENT RESULTS

We now present our empirical observations regarding the role of GLS under clean and noisy labels by using UCI datasets, CIFAR-10 and CIFAR-100.

# 6.1 WHAT IS THE PRACTICAL DISTRIBUTION OF  $r^*$  AND  $r_{\mathrm{OPT}}$ ?

$r^*$  and  $r_{\mathrm{opt}}$  on UCI datasets (Dua & Graff, 2017) As for UCI datasets, we pick Twonorm and Splice for illustration in the main paper. The noisy labels are generated by a symmetric noise transition matrix with noise rate  $e_i = [0.1, 0.2, 0.3, 0.4]$ . As highlighted in Table 1,  $r_{\mathrm{opt}}$  appears with positive values when the data is clean (same as  $r^*$ ) or of a low noise rate. With the increasing of noise rates, the performance of PLS results in a much larger degradation compared with NLS. We color-code different noise regimes where either PLS (red-ish) or NLS (green-ish) outperforms the other. Clearly there is a separation of the favored smoothing rate for different noise scenarios (upper left & low noise for PLS, bottom right & high noise for NLS).

Table 1: Test accuracies of GLS on clean and noisy UCI datasets with best two smooth rates (green: NLS; red: PLS). Results on more benchmark datasets are deferred to Appendix D.  

<table><tr><td rowspan="2">Smooth Rate</td><td colspan="5">Twonorm</td><td colspan="5">Splice</td></tr><tr><td>ei=0</td><td>ei=0.1</td><td>ei=0.2</td><td>ei=0.3</td><td>ei=0.4</td><td>ei=0</td><td>ei=0.1</td><td>ei=0.2</td><td>ei=0.3</td><td>ei=0.4</td></tr><tr><td>r=0.8</td><td>0.990</td><td>0.990</td><td>0.986</td><td>0.982</td><td>0.968</td><td>0.980</td><td>0.946</td><td>0.919</td><td>0.856</td><td>0.760</td></tr><tr><td>r=0.6</td><td>0.990</td><td>0.989</td><td>0.987</td><td>0.981</td><td>0.972</td><td>0.978</td><td>0.939</td><td>0.913</td><td>0.869</td><td>0.778</td></tr><tr><td>r=0.4</td><td>0.990</td><td>0.990</td><td>0.987</td><td>0.983</td><td>0.971</td><td>0.978</td><td>0.948</td><td>0.922</td><td>0.885</td><td>0.797</td></tr><tr><td>r=0.2</td><td>0.990</td><td>0.989</td><td>0.986</td><td>0.985</td><td>0.969</td><td>0.978</td><td>0.948</td><td>0.919</td><td>0.878</td><td>0.800</td></tr><tr><td>r=0.0</td><td>0.990</td><td>0.989</td><td>0.987</td><td>0.985</td><td>0.973</td><td>0.976</td><td>0.948</td><td>0.926</td><td>0.876</td><td>0.806</td></tr><tr><td>r=-0.4</td><td>0.986</td><td>0.988</td><td>0.988</td><td>0.986</td><td>0.972</td><td>0.961</td><td>0.956</td><td>0.928</td><td>0.880</td><td>0.817</td></tr><tr><td>r=-0.6</td><td>0.986</td><td>0.988</td><td>0.987</td><td>0.984</td><td>0.974</td><td>0.961</td><td>0.956</td><td>0.926</td><td>0.880</td><td>0.819</td></tr><tr><td>r=-1.0</td><td>0.986</td><td>0.986</td><td>0.988</td><td>0.985</td><td>0.977</td><td>0.956</td><td>0.954</td><td>0.932</td><td>0.889</td><td>0.819</td></tr><tr><td>r=-2.0</td><td>0.986</td><td>0.986</td><td>0.986</td><td>0.986</td><td>0.978</td><td>0.952</td><td>0.946</td><td>0.935</td><td>0.898</td><td>0.830</td></tr><tr><td>r=-4.0</td><td>0.986</td><td>0.986</td><td>0.986</td><td>0.986</td><td>0.983</td><td>0.946</td><td>0.943</td><td>0.939</td><td>0.911</td><td>0.830</td></tr><tr><td>r=-8.0</td><td>0.986</td><td>0.986</td><td>0.986</td><td>0.985</td><td>0.986</td><td>0.943</td><td>0.946</td><td>0.939</td><td>0.915</td><td>0.845</td></tr><tr><td>ropt=</td><td>[0.0, 0.8]</td><td>[0.4, 0.8]</td><td>[-1.0, -0.4]</td><td>[-4.0, -0.4]</td><td>-8.0</td><td>[0.0, 0.8]</td><td>[-0.6, -0.4]</td><td>[-8.0, -4.0]</td><td>-8.0</td><td>-8.0</td></tr></table>

$r^*$  and  $r_{\mathrm{opt}}$  on CIFAR datasets (Krizhevsky et al., 2009) When learning with a larger scale and more complex dataset, like CIFAR-10 and CIFAR-100, models are prone to converge on a local optimal solution rather than the global optimum. This phenomenon occurs frequently in NLS which ends up with performance degradation. Thus, in Table 2, when learning with noisy labels, we report the better performance of GLS between direct training and loading the same warm-up model. And we observe that the performance of NLS is more competitive than PLS when learning with clean data. Clearly, NLS outperforms PLS in CIFAR-10 and CIFAR-100 under various synthetic noise settings. The gap is larger when the noise rates are high.

Table 2: Test accuracies of GLS on synthetic noisy CIFAR datasets: we report the mean accuracy of each smooth rate setting and the standard deviation. Best two smooth rates for each synthetic noise setting are highlighted for each  $\epsilon$  (green: NLS; red: PLS).  

<table><tr><td rowspan="2">Smooth Rate</td><td colspan="4">Cifar-10 Symmetric</td><td colspan="2">Cifar-10 Asymmetric</td><td colspan="2">CIFAR-100 Symmetric</td></tr><tr><td>ε = 0.0</td><td>ε = 0.2</td><td>ε = 0.4</td><td>ε = 0.6</td><td>ε = 0.2</td><td>ε = 0.3</td><td>ε = 0.4</td><td>ε = 0.6</td></tr><tr><td>r = 0.8</td><td>92.91±0.06</td><td>88.88±1.61</td><td>81.48±2.91</td><td>73.16±0.16</td><td>90.45±0.06</td><td>87.83±0.13</td><td>54.04±0.93</td><td>39.50±0.18</td></tr><tr><td>r = 0.6</td><td>92.33±0.09</td><td>87.50±1.31</td><td>82.11±0.86</td><td>73.59±0.15</td><td>90.41±0.09</td><td>87.83±0.13</td><td>52.72±0.15</td><td>40.49±0.07</td></tr><tr><td>r = 0.4</td><td>93.05±0.04</td><td>87.13±0.07</td><td>81.50±1.42</td><td>74.21±0.19</td><td>90.49±0.10</td><td>87.90±0.13</td><td>54.26±0.07</td><td>41.57±0.05</td></tr><tr><td>r = 0.0</td><td>91.44±0.16</td><td>85.08±0.86</td><td>80.42±2.29</td><td>75.34±0.13</td><td>88.32±0.24</td><td>86.27±0.32</td><td>48.03±0.29</td><td>38.11±0.14</td></tr><tr><td>r = -0.4</td><td>93.55±0.06</td><td>87.55±0.08</td><td>81.58±0.19</td><td>75.95±0.13</td><td>87.27±1.83</td><td>88.33±0.06</td><td>56.87±0.08</td><td>43.70±0.16</td></tr><tr><td>r = -0.8</td><td>92.74±0.05</td><td>88.46±0.11</td><td>81.56±0.15</td><td>76.15±0.14</td><td>86.40±1.32</td><td>87.96±0.43</td><td>57.35±0.08</td><td>44.10±0.06</td></tr><tr><td>r = -1.0</td><td>92.58±0.08</td><td>88.58±0.08</td><td>81.95±0.10</td><td>76.20±0.10</td><td>88.47±0.15</td><td>87.50±0.73</td><td>57.44±0.09</td><td>43.85±0.19</td></tr><tr><td>r = -2.0</td><td>93.30±0.03</td><td>88.78±0.09</td><td>83.64±0.15</td><td>76.11±0.07</td><td>88.66±0.17</td><td>87.27±0.70</td><td>58.10±0.08</td><td>44.88±0.11</td></tr><tr><td>r = -4.0</td><td>93.13±0.04</td><td>88.90±0.07</td><td>84.34±0.13</td><td>77.22±0.09</td><td>89.56±0.17</td><td>87.29±0.59</td><td>58.35±0.09</td><td>46.38±0.05</td></tr><tr><td>r = -6.0</td><td>93.14±0.08</td><td>88.94±0.11</td><td>84.52±0.13</td><td>77.42±0.16</td><td>89.70±0.24</td><td>87.57±0.42</td><td>57.73±0.10</td><td>46.46±0.09</td></tr></table>

# 6.2 GLS AND MODEL CONFIDENCE

Model confidence on CIFAR-10 In Theorem 5, we have shown that NLS explicitly encourages a higher expected model confidence compared with PLS. Will NLS make the model be more confident on wrong predictions? We train GLS on CIFAR-10 with symmetric  $(\epsilon = 0.6)$  label noise. As shown in Figure 3, with the decreasing of smooth rates (from right to left), the model confidence on correct predictions gradually approach to its maximum, while for wrong predictions, the model confidence converges to its minimum value. In a nutshell, NLS makes the model prediction become over-confident in correct predictions and almost no confidence on wrong predictions.

![](images/28600f446be8214b4f42f940fe65217359ee6c84e48c59a3bea0f62268386d54.jpg)  
Figure 3: Model confidence distribution of correct and wrong predictions on CIFAR-10 test data. (From left to right: NLS ( $r = -4.0$ ,  $-0.8$ ), Vanilla Loss, PLS ( $r = 0.4$ )).

![](images/1da40b09b6b3d5663b3639042e9c56ad495b14635a6f75b9418c2b03476fa528.jpg)

![](images/c4cda3164de465b6e7df9b204391f086f0a6212480b584589becdf19e86b2002.jpg)

![](images/182661a5904eebd0c8e0d0facc2be2fa5acd499fa2aa1623ac81c1eb263f298e.jpg)

# 6.3 ADDITIONAL RESULTS

For interested readers, we defer more extensive empirical results to Appendix. Including:

Empirical validation of main theorems (Appendix B); Practical considerations of GLS (Appendix C);

Experiment comparisons with more methods on synthetic noisy CIFAR datasets (Appendix D);

Bias and variance of the generalization error on the clean data (Appendix E).

# 7 CONCLUSION

In this paper, we provide understandings for a generalized notion of label smoothing where the label smoothing rate can go negative. We demonstrate that learning with negatively smoothed labels explicitly improves the confidence of model prediction. This key property acts as a significant role when the confidence of model prediction drops. We make connections between negative label smoothing and existing learning with noisy label solutions. In contrast to existing works that promote the use of positive label smoothing, we show both theoretically and empirically the advantage of a negative smooth rate when the label noise rate increases. Our observations provide new understanding for the effects of label smoothing, especially when the training labels are imperfect.

# REFERENCES

Antonin Berthon, Bo Han, Gang Niu, Tongliang Liu, and Masashi Sugiyama. Confidence scores make instance-dependent label-noise learning possible. In International Conference on Machine Learning, pp. 825-836. PMLR, 2021.  
Blair Chen, Liu Ziyin, Zihao Wang, and Paul Pu Liang. An investigation of how label smoothing affects generalization. arXiv preprint arXiv:2010.12648, 2020.  
Tianlong Chen, Zhenyu Zhang, Sijia Liu, Shiyu Chang, and Zhangyang Wang. Robust overfitting may be mitigated by properly learned smoothening. In International Conference on Learning Representations, volume 1, 2021.  
Hao Cheng, Zhaowei Zhu, Xingyu Li, Yifei Gong, Xing Sun, and Yang Liu. Learning with instance-dependent label noise: A sample sieve approach. In International Conference on Learning Representations, 2020.  
Jan Chorowski and Navdeep Jaitly. Towards better decoding and language model integration in sequence to sequence models. Proc. Interspeech 2017, pp. 523-527, 2017.  
Dheeru Dua and Casey Graff. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ml.  
Yingbo Gao, Weiyue Wang, Christian Herold, Zijian Yang, and Hermann Ney. Towards a better understanding of label smoothing in neural machine translation. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pp. 212-223, 2020.  
Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In Advances in neural information processing systems, pp. 8527-8537, 2018.  
Ramaswamy Harish, Clayton Scott, and Ambuj Tewari. Mixture proportion estimation via kernel embeddings of distributions. In International conference on machine learning, pp. 2052-2060, 2016.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Geoffrey Hinton, Oriol Vinyls, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
Takashi Ishida, Gang Niu, Weihua Hu, and Masashi Sugiyama. Learning from complementary labels. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pp. 5644-5654, 2017.  
Takashi Ishida, Gang Niu, Aditya Menon, and Masashi Sugiyama. Complementary-label learning for arbitrary losses and models. In International Conference on Machine Learning, pp. 2971-2980. PMLR, 2019.  
Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In International Conference on Machine Learning, pp. 2304-2313. PMLR, 2018.  
Youngdong Kim, Junho Yim, Juseung Yun, and Junmo Kim. Nlnl: Negative learning for noisy labels. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 101-110, 2019.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.  
Doyup Lee and Yeongjae Cheon. Soft labeling affects out-of-distribution detection of deep neural networks. arXiv preprint arXiv:2007.03212, 2020.  
Weizhi Li, Gautam Dasarathy, and Visar Berisha. Regularization via structural label smoothing. In International Conference on Artificial Intelligence and Statistics, pp. 1453-1463. PMLR, 2020.  
Sheng Liu, Jonathan Niles-Weed, Narges Razavian, and Carlos Fernandez-Granda. Early-learning regularization prevents memorization of noisy labels. Advances in Neural Information Processing Systems, 33, 2020.  
Tongliang Liu and Dacheng Tao. Classification with noisy labels by importance reweighting. IEEE Transactions on pattern analysis and machine intelligence, 38(3):447-461, 2015.  
Yang Liu. The importance of understanding instance-level noisy labels. arXiv preprint arXiv:2102.05336, 2021.  
Yang Liu and Hongyi Guo. Peer loss functions: Learning from noisy labels without knowing noise rates. In International Conference on Machine Learning, pp. 6226-6236. PMLR, 2020.  
Michal Lukasik, Srinadh Bhojanapalli, Aditya Menon, and Sanjiv Kumar. Does label smoothing mitigate label noise? In International Conference on Machine Learning, pp. 6448-6458. PMLR, 2020.  
Xingjun Ma, Hanxun Huang, Yisen Wang, Simone Romano, Sarah Erfani, and James Bailey. Normalized loss functions for deep learning with noisy labels. In International Conference on Machine Learning, pp. 6543-6553. PMLR, 2020.  
Aditya Menon, Brendan Van Rooyen, Cheng Soon Ong, and Bob Williamson. Learning from corrupted binary labels via class-probability estimation. In International Conference on Machine Learning, pp. 125-134, 2015.  
Rafael Müller, Simon Kornblith, and Geoffrey Hinton. When does label smoothing help? In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 4696-4705, 2019.  
Nagarajan Natarajan, Inderjit S Dhillon, Pradeep K Ravikumar, and Ambuj Tewari. Learning with noisy labels. In Advances in neural information processing systems, pp. 1196-1204, 2013.  
Curtis G Northcutt, Lu Jiang, and Isaac L Chuang. Confident learning: Estimating uncertainty in dataset labels. Journal of Artificial Intelligence Research, 2021.  
Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: A loss correction approach. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1944-1952, 2017.  
Gabriel Pereyra, George Tucker, Jan Chorowski, Lukasz Kaiser, and Geoffrey Hinton. Regularizing neural networks by penalizing confident output distributions. arXiv preprint arXiv:1701.06548, 2017.  
Geoff Pleiss, Tianyi Zhang, Ethan R Elenberg, and Kilian Q Weinberger. Identifying mislabeled data using the area under the margin ranking. arXiv preprint arXiv:2001.10528, 2020.  
Scott Reed, Honglak Lee, Dragomir Anguelov, Christian Szegedy, Dumitru Erhan, and Andrew Rabinovich. Training deep neural networks on noisy labels with bootstrapping. arXiv preprint arXiv:1412.6596, 2014.

Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals of mathematical statistics, pp. 400-407, 1951.  
Clayton Scott, Gilles Blanchard, Gregory Handy, Sara Pozzi, and Marek Flaska. Classification with asymmetric label noise: Consistency and maximal denoising. In  $COLT$ , pp. 489-511, 2013.  
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2818-2826, 2016.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pp. 6000-6010, 2017.  
Hongxin Wei, Lei Feng, Xiangyu Chen, and Bo An. Combating noisy labels by agreement: A joint training method with co-regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13726-13735, 2020.  
Jiaheng Wei and Yang Liu. When optimizing  $f$ -divergence is robust with label noise. In International Conference on Learning Representations, 2020.  
Yi Xu, Yuanhong Xu, Qi Qian, Hao Li, and Rong Jin. Towards understanding label smoothing. arXiv preprint arXiv:2006.11653, 2020.  
Yilun Xu, Peng Cao, Yuqing Kong, and Yizhou Wang. L_dmi: An information-theoretic noise-robust loss function. NeurIPS, arXiv:1909.03388, 2019.  
Zitong Yang, Yaodong Yu, Chong You, Jacob Steinhardt, and Yi Ma. Rethinking bias-variance trade-off for generalization of neural networks. In International Conference on Machine Learning, pp. 10767-10777. PMLR, 2020.  
Quanming Yao, Hansi Yang, Bo Han, Gang Niu, and James Tin-Yau Kwok. Searching to exploit memorization effect in learning with noisy labels. In International Conference on Machine Learning, pp. 10789-10798. PMLR, 2020a.  
Yu Yao, Tongliang Liu, Bo Han, Mingming Gong, Jiankang Deng, Gang Niu, and Masashi Sugiyama. Dual t: Reducing estimation error for transition matrix in label-noise learning. arXiv preprint arXiv:2006.07805, 2020b.  
Xingrui Yu, Bo Han, Jiangchao Yao, Gang Niu, Ivor Tsang, and Masashi Sugiyama. How does disagreement help generalization against label corruption? In International Conference on Machine Learning, pp. 7164-7173. PMLR, 2019.  
Li Yuan, Francis EH Tay, Guilin Li, Tao Wang, and Jiashi Feng. Revisiting knowledge distillation via label smoothing regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3903-3911, 2020.  
Helong Zhou, Liangchen Song, Jiajie Chen, Ye Zhou, Guoli Wang, Junsong Yuan, and Qian Zhang. Rethinking soft labels for knowledge distillation: A bias-variance tradeoff perspective. arXiv preprint arXiv:2102.00650, 2021.  
Zhaowei Zhu, Yiwen Song, and Yang Liu. Clusterability as an alternative to anchor points when learning with noisy labels. arXiv preprint arXiv:2102.05291, 2021.
