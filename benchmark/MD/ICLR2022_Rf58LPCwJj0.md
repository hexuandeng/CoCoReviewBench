# OPTIMAL REPRESENTATIONS FOR COVARIATE SHIFTS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Machine learning systems often experience a distribution shift between training and testing. In this paper, we introduce a simple variational objective whose optima are exactly the set of representations on which risk minimizers are guaranteed to be robust to any distribution shift that preserves the Bayes predictor, e.g., covariate shifts. Our objective has two components. First, a representation must remain discriminative for the task, i.e., one predictor can simultaneously minimize the source and target risk. Second, the representation's marginal support needs to be the same across source and target. We make this observation practical by designing self-supervised learning methods that use unlabelled data and augmentations to train robust representations. Our objectives achieve state-of-the-art results on DomainBed, and give insights into the robustness of recent methods, such as CLIP.

# 1 INTRODUCTION

It is hard to build machine learning (ML) systems that are robust to distribution shifts between training (source) and testing (target). No method for domain generalization (DG) uniformly outperforms empirical source-risk minimizers (ERM) in practice (Gulrajani & Lopez-Paz, 2021).

Representation learning seems like a promising approach. For example, domain-invariant representation learning (e.g., DANN, Ganin et al., 2016) seeks representations that are invariant to distribution shifts, while retaining information about the source task. Unfortunately, this intuitive approach is not sound; there are examples of invariant representations that perform well on the source, but poorly on the target (Zhao et al., 2019; Johansson et al., 2019). While properties that would imply robust representations are known (Ben-David et al., 2010a), the minimal set of requirements is not known.

We introduce the first, simple, variational objective whose optima are exactly the set of representations on which source risk minimizers are guaranteed to generalize across distribution shifts that preserve the Bayes predictor. We work in an idealized DG (IDG) setting; we assume that a learner has access to the source population risk. Our variational characterization implies that it is both sufficient and necessary for optimal IDG that a representation, (a) remains discriminative for the learning task, i.e., there must exist predictors from the representation to the labels that can simultaneously minimize both source and target risk; and (b) keeps the support of its marginal distribution invariant to shifts.

This means that any optimal representation learning method must seek discriminative information about the target. Even worse, we prove that without access to some knowledge about the target, any representation learning algorithm cannot uniformly (over all discriminative tasks) outperform a constant representation, which may explain why DG methods struggle to outperform ERM.

We show how these challenges can be overcome with access to target inputs and particular types of data augmentations that carry discriminative information about downstream tasks, but minimal domain-specific information. Text descriptions of images are examples of such augmentations, because they are informative for many downstream classification tasks, but they remove a lot of domain-specific information. With such augmentations, we design practical self-supervised learning (SSL) objectives for learning robust representations. Our objectives give insights into the robustness of CLIP (Radford et al., 2021), and lead to improved CLIP-based representations that achieve state-of-the-art results on DomainBed (Gulrajani & Lopez-Paz, 2021). To summarize, we:

- provide all objectives whose optima achieve optimal domain generalization under covariate shift;  
- prove that it is impossible to learn useful representations without accessing target information;  
- provide practical objectives to learn robust representations using specific data augmentations;  
- get state-of-the-art results on typical domain generalization benchmarks.

# 2 BACKGROUND: DOMAIN GENERALIZATION AND REPRESENTATIONS

We are interested in predictions that are robust across distribution shifts. We formalize this using domain generalization (DG) language. Given a distribution  $p_{X,Y|d_s}$  over inputs  $x \in \mathcal{X}$  and labels  $y \in \mathcal{Y}$  from the source domain  $d_s \in \mathcal{D}$ , we select a predictor  $f: \mathcal{X} \to \Gamma$ . The predictions  $\gamma \in \Gamma$  could for example be labels or distributions over labels. Despite being selected on the source domain, we would like  $f$  to achieve a small expected risk with respect to a loss function  $\ell: \mathcal{Y} \times \Gamma \to \mathbb{R}_{\geq 0}$ ,

$$
\mathrm {R} _ {f} ^ {d} [ Y \mid X ] := \mathbb {E} _ {p _ {X, Y \mid d}} [ \ell (Y, f (X)) ], \tag {1}
$$

on a distribution  $p_{X,Y|d}$  from a target domain  $d = d_t \in \mathcal{D}$ , which is somehow related to  $d_s$ .

A common strategy for DG is to learn robust representations. Specifically, the idea is to split the problem of learning robust predictors into two: first, learn an encoder  $p_{Z|X}$ , which stochastically maps inputs  $X$  to representations  $Z$ . Then, learn a predictor  $h: \mathcal{Z} \to \Gamma$  from representations  $Z$  to labels  $Y$  using standard risk minimization. The goal is to design a robust representation, so that predictors  $h$  trained to minimize the source risk  $\mathrm{R}_h^{d_s}[Y|Z]$  also achieve low target risk  $\mathrm{R}_h^{d_t}[Y|Z]$ . Many methods have been proposed to try to learn such representations  $Z$  by enforcing invariance of the marginal  $p_{Z|d}$  (e.g., Ganin et al., 2016) to the domain  $d$ . Still, many of these proposals are not sound (Zhao et al., 2019; Johansson et al., 2019). Furthermore, these methods rarely outperform source empirical risk minimization (ERM) in practice (Gulrajani & Lopez-Paz, 2021).

# 3 OPTIMAL REPRESENTATIONS FOR DOMAIN GENERALIZATION

To separate domain generalization from finite sample generalization, we consider an idealized DG (IDG) setting, where the learner selects her predictor  $h$  according to the source population risk rather than empirical risk. In our conclusions, we discuss what is and is not lost in this idealization. We assume sample spaces  $\mathcal{X}, \mathcal{Z}, \mathcal{Y}, \mathcal{D}$  are discrete; formal statements and proofs are in Appxs. A and B.

# 3.1 DEFINING OPTIMAL REPRESENTATIONS FOR IDEALIZED DOMAIN GENERALIZATION

We are interested in evaluating the quality of a representation  $Z$  (i.e., encoder  $p_{Z|X}$ ) of  $X$  for IDG. In our model of IDG, the learner is given a random source  $D_s$ ; she selects any source risk minimizer; and she is scored according to her risk on a random target domain  $D_t$ . To give uniform guarantees while reflecting the uncertainty over which source-target pair she will be given, we measure the quality of  $Z$  as the expected value of the learner's worst-case choice.

Definition. The idealized domain generalization risk (IDG risk) of an encoder  $p_{Z|X}$  is the expected (over domains) worst-case (over source risk minimizers) target risk, i.e.,

$$
\mathrm {R} _ {\mathrm {I D G}} [ Y \mid Z ] := \mathbb {E} _ {p _ {D _ {s}, D _ {t}}} \left[ \sup  _ {h \in \mathcal {H} _ {D _ {s}} ^ {*}} \mathrm {R} _ {h} ^ {D _ {t}} [ Y \mid Z ] \right] \tag {2}
$$

where  $\mathcal{H}_{D_s}^* \coloneqq \arg \min_h \mathrm{R}_h^{D_s}[Y|Z]$  are the source risk minimizers. We call a representation  $Z^*$  (or its encoder) optimal for IDG if it minimizes the IDG risk:  $p_{Z^*|X} \in \arg \min_{p_{Z|X}} \mathrm{R}_{\mathrm{IDG}}[Y|Z]$ .

# 3.2 CHARACTERIZING OPTIMAL REPRESENTATIONS FOR IDG UNDER COVARIATE SHIFT

The IDG risk is useful to evaluate representations, but gives few insights into IDG and is impractical to optimize due to the supremum in Eq. (2). Under mild assumptions, we provide a simplified, equivalent objective, which is easier to optimize. For convenience, we assume that there is a unique Bayes predictor  $f^{*}$ , which minimizes the expected risk over domains, i.e.,  $f^{*} = \arg \min_{f}\mathbb{E}_{p_{D_{t}}}\left[\mathrm{R}_{f}^{D_{t}}\left[Y\mid X\right]\right]$ . This is satisfied by standard ML tasks  $p_{Y,X}$  and losses  $\ell$ . More importantly, we assume the following domain structure, which ensures the existence of optimal encoders and allows our simplification.

Assumptions. All domains  $d \in \mathcal{D}$  we consider are related by the following assumptions:

1. Generalized covariate shift. All domain-specific risk minimizers  $f \in \arg \min_{f}[\mathbb{R}_{f}^{d}[Y|X]]$  are equal to the Bayes predictor  $f^{*}$  on their support, i.e.,  $f(x) = f^{*}(x)$  for all  $x \in \operatorname{supp}(p_{X|d})$ .

![](images/9595a3bf1f05fe47c40bd8ec5bae65b1e937123414023d95847c4113414dbaa0.jpg)  
(a) discriminative & support match

![](images/c8107af4df6b850bb06c69c72025db966085db45bafae1737e49a74e99bae50d.jpg)  
Figure 1: (a) Optimal representations for IDG must be discriminative on all domains (simultaneously) and keep their support invariant: (b) without the discriminative requirement, a source-risk minimizer can mispredict the target, and (c) without support match, it can perform poorly in the worst case.  
(b) only support match

![](images/b86e7430f328e2acd6a9c2664a5d12d1c4d955ff30ee76dbfb1d5372d0f836ed.jpg)  
(c) only discriminative

2. Invariance of Bayes predictions. The set of Bayes predictions is the same for all domains, i.e.,  $\{f^{*}(x) \mid x \in \operatorname{supp}(p_{X|d})\} = \{f^{*}(x) \mid x \in \mathcal{X}\}$ .

Generalized covariate shift (GCS) ensures that  $f^{*}$  is simultaneously optimal on all domains. For log-loss  $\ell$  it recovers standard covariate shift, i.e.,  $p_{Y|x,d} = p_{Y|x}$ . For other losses, Asm. 1 is weaker, e.g., it only requires invariance of most likely labels for 0-1 loss, and of conditional expectations for MSE. Invariance of Bayes predictors is necessary to learn useful predictors using a single domain. For example, for 0-1 loss it ensures that each label is seen at least once in each domain.

The intuition behind our objective is that under GCS any source risk minimizer will make optimal predictions on all target samples  $x$  that are also in the source domain's support. Thus, optimal representations for IDG are exactly those that (a) ensure that all domains have the same support in  $Z$ , and (b) retain GCS from  $Z$  without sacrificing the ability to predict  $Y$  optimally, which can be ensured by minimizing the achievable risk of from  $Z$ . See Fig. 1.

Theorem 1. Under our assumptions, an encoder  $p_{Z^* \mid X}$  is optimal for IDG if and only if it minimizes the risk  $R[Y \mid Z] := \inf_h \mathbb{E}_{p_{D_t}}[\mathrm{R}_h^{D_t}[Y \mid Z]]$  while matching the support of  $Z$  across domains, i.e.,

$$
p _ {Z ^ {*} \mid X} \in \underset {p _ {Z \mid X}} {\arg \min } \quad \mathrm {R} [ Y \mid Z ] \tag {3}
$$

$$
\text {s . t .} \quad \forall d \in \mathcal {D}, \operatorname {s u p p} \left(p _ {Z \mid d}\right) = \operatorname {s u p p} \left(p _ {Z}\right) \tag {4}
$$

Moreover, such encoders exist and their IDG risk is the Bayes risk  $\mathrm{R_{IDG}}[Y|Z^*] = \mathrm{R}[Y|X]$ .

Theorem 2 provides an objective to learn representations on which performing risk minimization using a single domain is as good as performing risk minimization on all domains simultaneously, i.e.,  $\mathrm{R_{IDG}}[Y|Z^*] = \mathrm{R}[Y|X]$ . Other sufficient objectives had previously been proven or hinted towards (Ben-David et al., 2010a), e.g., minimizing the risk while matching the representation's marginal. To our knowledge, Thm. 2 is nevertheless the first that identifies the necessary and sufficient conditions for representations  $Z^*$  that are optimal for IDG. This has the advantage of giving better insights into IDG and provides a framework for deriving all objectives that describe optimal IDG.

The risk minimization (Eq. (3)) shows that one must have some knowledge about the target domains to learn optimal representations for IDG. Access to targets might seem unrealistic, but without such knowledge or additional assumptions it is provably impossible to beat even constant representations.

Proposition 1 (No free lunch for IDG). Let  $d_s$  be any source domain,  $Z_{d_s}$  be any representation chosen on source  $d_s$ , and  $C \in \mathcal{Z}$  be a constant representation. Under minor assumptions, for every "good" target domain outside the source's support on which  $Z_{d_s}$  outperforms  $C$  for IDG, there are many "bad" target domains on which  $Z_{d_s}$  is strictly worse than  $C$ . Formal statement in Appx. B.3.

Proposition 1 shows that target knowledge is necessary for learning useful representations in IDG. This may explain why previous methods have been unable to to outperform ERM in standard DG benchmarks (Gulrajani & Lopez-Paz, 2021): the knowledge they have access to is insufficient. Taken together, Prop. 1 and Thm. 2 say either you allow yourself to consider target domains  $d_{t}$ , in which case you can achieve an IDG risk that matches supervised learning, or you do not access  $d_{t}$ , in which case and any representation learning algorithm can achieve worst IDG risk than a constant.

![](images/b23ecbddb1e89806dc7984fd1460d9a6b2feba6872f7656797a01c3358cda125.jpg)  
(a) standard augmentations

![](images/01fffbea369f1ac4c2356980fcb4f9f42ed228dbf1b2cc5da9b2be7585e44554.jpg)  
Figure 2: Image-text augmentations are practical domain-covering augmentations. Arrows denote augmentations. Bubbles denote inputs that have the same representations, as induced by predicting the augmentations. (a) Standard augmentations are not domain-covering. (b) Supervised augmentations uniformly augment inputs inside their label class, irrespective of domains. (c) Image-text augmentations are (nearly) domain-covering as they map images across domains to similar descriptions.

![](images/cd9d292fe8961e073ffa17514a09797230d336a436c3fcdf4c997b211c3a2978.jpg)  
(b) supervised augmentations  
(c) image-text augmentations

# 4 LEARNING REPRESENTATIONS UNDER COVARIATE SHIFT

# 4.1 SELF-SUPERVISED LEARNING USING DOMAIN-COVERING AUGMENTATIONS

Our characterization of optimal representations for IDG (Thm. 2) requires labeled data from all domains, which is impractical. We show how this can be overcome with self-supervised learning (SSL), which is a technique for training representations without direct access to labels, and a particular class of data augmentations. E.g., in CLIP, images are augmented with alt-text collected on the internet and invariance is enforced between the representations of the image and its text pair (Radford et al., 2021). Representations learned like this preserve discriminative information about all downstream tasks  $Y$  whose label information is preserved by the augmentation (e.g., Dubois et al., 2021).

More precisely, an augmentation  $A$  is a random variable sampled conditionally from the input  $X$ . The key requirement that allows us to retain task information is, that, if any samples  $x, x' \in \mathcal{X}$  have the same augmentation conditional  $p_{A|x} = p_{A|x'}$ , then their Bayes predictions must be the same  $f^{*}(x) = f^{*}(x')$ . With augmentations that preserve label information in this way, one can learn an encoder that minimizes the risk  $R[Y|Z]$  by instead maximizing mutual information  $I[A;Z]$ . Intuitively, if  $Z$  has all information about the augmentation  $A$ , then it must have information about the conditional  $p_{A|X}$ , and thus the Bayes prediction  $f^{*}(X)$ .

This suggests the possibility of learning optimal representations for IDG by replacing Eq. (3) with a maximization of  $\operatorname{I}[A;Z]$ . Unfortunately, fully optimizing  $\operatorname{I}[A;Z]$  w.r.t.  $p_{Z|X}$  is not generally possible under the support constraint Eq. (4). This can be overcome under a domain-covering assumption, which requires that for every domain there is an example that gets mapped to every possible augmentation distribution, i.e.,  $\{p_{A|x} | x \in \mathrm{supp}(p_{X|d})\} = \{p_{A|x} | x \in \mathcal{X}\}$ .

Proposition 2. Let  $p_{A|X}$  be a domain-covering augmenter. Then any optimal solution  $p_{Z^{*}|X}$  of the following objective is optimal for IDG:

$$
p _ {Z ^ {*} \mid X} \in \underset {p _ {Z \mid X}} {\arg \max } \mathrm {I} [ A; Z ] \quad \text {s . t .} \quad \forall d \in \mathcal {D}, \operatorname {s u p p} \left(p _ {Z \mid d}\right) = \operatorname {s u p p} \left(p _ {Z}\right) \tag {5}
$$

Proposition 2 shows that we can still learn optimal representations for IDG without labels as long as we have access to the right augmentations. But how realistic are those augmentations? For 0-1 loss  $\ell$ , the most likely label should be preserved, which is satisfied by standard image augmentations like mild rotations and color jittering. Those augmentations are nevertheless not domain-covering for typical domains (e.g. sketches and photos), since outputs  $A$  are highly correlated with the domain  $D$  of the original input  $X$ , as seen in Fig. 2a.

A practical choice of augmentation that is nearly domain-covering, is a mapping from images to text descriptions, as with CLIP (Radford et al., 2021). Image-text augmentations have many advantages. First, text augmentations preserve label information for many downstream classification tasks. Second, they are close to being domain-covering, since images from different domains (e.g., sketches and photos) but similar semantics are often mapped to similar descriptions<sup>1</sup> (Fig. 2c). This may explain the incredible robustness of CLIP's representations when compared to other SSL methods (Chen et al., 2020; He et al., 2020; Grill et al., 2020). Finally, image-text pairs are easy to access in practice given their abundance on the internet.

There is still the question of enforcing the support constraint for augmented data  $(X,A)$  (Eq. (5)), which may not come with domain information. In this case, one can replace the support constraint with a stronger one, e.g., minimizing  $\operatorname{I}[Z;X]$  (see Sec. 4.2.2), that does not rely on the domain information. This highlights the potential of Prop. 2: if one can find a large source of inputs  $X$  and domain-covering augmentations  $A$  (e.g., the 400M image-text pairs of CLIP) then one can, in principle, learn optimal representations for IDG on any downstream task  $Y$  that  $A$  preserves.

# 4.2 PRACTICAL OBJECTIVES

We now design practical objectives for learning optimal representations without labels. Proposition 2 does provide an objective but it is impractical as it involves constrained optimization. We can nevertheless convert it to the following unconstrained objective by using a Lagrangian relaxation and introducing a domain bottleneck  $\mathrm{B}[Z, D]$  that enforces support match,

$$
\underset {p _ {Z \mid X}} {\arg \min } \quad \mathrm {H} [ A \mid Z ] + \lambda \mathrm {B} [ Z, D ], \tag {6}
$$

where  $\mathrm{H}[A|Z]$  replaces  $\mathrm{I}[A;Z] = \mathrm{H}[A] - \mathrm{H}[A|Z]$  as  $\mathrm{H}[A]$  is a constant with respect to the encoder. Eq. (6) is a valid reformulation of Prop. 2 as long as minimizing  $\mathrm{B}[Z,D]$  while maximizing  $\mathrm{I}[A;Z]$  enforces the support constraint Eq. (4). Below, we provide different choices of such  $\mathrm{B}[Z,D]$  each of which results in a different SSL objective. In practice, however, terms in Eq. (6) are hard to estimate from finite samples. We now discuss two variational bounds that can be efficiently estimated and optimized with stochastic gradient descent (Bottou, 2010). For simplicity, we use a deterministic encoder  $e_{\varphi}:\mathcal{X}\to \mathcal{Z}$  for the rest of the paper. Detailed derivations are in Appx. C.

For both practical objectives we use an upper bound on  $\mathrm{H}[A|Z] \leq \mathbb{E}_{p_{A,Z}}[-\log q(A|Z)]$ , where  $q$  is the contrastive variational distribution as in InfoNCE (Oord et al., 2018), which is standard in SSL. Specifically, for a sample  $X$ , we construct  $\mathbf{X} := \{X, X_1^-, \ldots, X_n^-\}$  where each  $X_i^-$  are i.i.d. sampled from  $p_X$ . Then we obtain a collection  $\mathbf{A} := \{A^+, A_1^-, \ldots, A_n^-\}$  of one positive augmentation  $A^+$  and  $n$  negatives  $A_i^-$  by independently sampling an augmentation from  $p_{A|X'}$  for each example  $X' \in \mathbf{X}$ . InfoNCE then uses a critic  $s_\psi$  to score how likely each  $A' \in \mathbf{A}$  is to be positive, i.e.,

$$
\mathrm {H} [ A \mid Z ] \leq \mathbb {E} _ {p _ {X, \mathbf {A}, Z}} \left[ - \log q _ {\psi , \mathbf {A}} \left(A ^ {+} \mid Z\right) \right] \quad \text {w h e r e} \quad q _ {\psi , \mathbf {A}} \left(A ^ {+} \mid Z\right) := \frac {\exp s _ {\psi} \left(A ^ {+} , Z\right)}{\sum_ {A ^ {\prime} \in \mathbf {A}} \exp s _ {\psi} \left(A ^ {\prime} , Z\right)}. \tag {7}
$$

When  $\mathcal{A} = \mathcal{X}$ , one can tie the critics' and encoders' parameters by passing augmentations through the encoder and taking an inner product, i.e.,  $s_{\psi}(A,Z) \coloneqq e_{\varphi}(A)^T Z$ .

Many previous DG regularizers (e.g., Ganin et al., 2016; Li et al., 2018b; a) could be valid the domain bottlenecks. In the following, we discuss two possible  $\mathrm{B}[Z,D]$ , the first of which is novel.

# 4.2.1 CONTRASTIVE ADVERSARIAL DOMAIN BOTTLENECK (CAD)

Our first domain bottleneck minimizes  $\mathrm{I}[Z;D]$  which enforces support match using a KL divergence. Dropping constants w.r.t.  $Z$  we thus aim to maximize  $\mathrm{H}[D|Z]$ . Domain-adversarial neural network (DANN, Ganin et al., 2016) does so by ensuring that a domain classifier  $q_{\phi}$  cannot predict domains from representations, i.e., it maximizes  $\mathbb{E}_{p_D,Z}[-\log q_\phi (D|Z)]\geq \mathrm{H}[D|Z]$  w.r.t. encoder parameter  $\varphi$  but minimizes it w.r.t.  $\phi$ . However, DANN suffers from two issues: (i) it maximizes an upper bound on the desired term; (ii) it requires adversarial training, which is challenging in practice.

# Algorithm 1 CAD objective

Require:  $e_{\varphi}, s_{\psi}, D, X, n$

1:  $Z\gets e_{\varphi}(X)$  
2:  $A^{+}\gets \mathrm{sample}(p_{A|X})$  
3:  $\{(D_i^-, X_i^-, A_i^-)\}_{i=1}^n \xleftarrow{\text{i.i.d.}}$  sample $(p_{D,X,A})$  
4:  $\mathbf{X},\mathbf{A}\gets \{X\} \cup \{X_{i}^{-}\}_{i = 1}^{n},\{A^{+}\} \cup \{A_{i}^{-}\}_{i = 1}^{n}$  
5:  $\mathbf{X}_D\gets \{X\} \cup \bigl \{X_i^- |D_i^- = D,i\in [n]\bigr \}$  
6:  $\mathcal{L}_{\mathrm{aug}}\gets -\log \frac{\exp s_{\psi}(A^{+},Z)}{\sum A^{\prime}\in\mathbf{A}\exp s_{\psi}(A^{\prime},Z)}$  
7:  $\mathcal{L}_{\mathrm{supp}}\gets \log \frac{\sum_{X^{\prime}\in\mathbf{X}_D}\exp e_{\varphi}(X^{\prime})^{T}Z}{\sum_{X^{\prime\prime}\in\mathbf{X}}\exp e_{\varphi}(X^{\prime\prime})^{T}Z}$  
8: return  $\mathcal{L}_{\mathrm{CAD}} = \mathcal{L}_{\mathrm{aug}} + \lambda \mathcal{L}_{\mathrm{supp}}$

To overcome these issues, we construct  $q(D\mid Z)$  without introducing additional parameters and with a bound that is tight with enough samples. In short, using the equality  $p_{D|Z} = \mathbb{E}_{p_{X|Z}}[p_{D|X}]$ , we set our variational distribution to  $q_{\varphi ,\mathbf{x},\mathbf{D}}(D\mid Z) = \mathbb{E}_{q_{\varphi ,\mathbf{x}}}[\hat{p} (D\mid X)]$ , where  $\hat{p}$  is a count estimate of  $p_{D|X}$ , and  $q_{\varphi ,\mathbf{x}}(X\mid Z)$  is a contrastive family similar to Eq. (7) but with critic  $e_{\varphi}(X)^T Z$ , Detailed

derivations and explanations are in Appx. C.3. The resulting contrastive adversarial domain (CAD) objective is detailed in Algorithm 1. First, sample domains  $\mathbf{D} := \{D, D_1^-, \ldots, D_n^-\}$  for each  $X' \in \mathbf{X}$ . Then collect inputs associated with the current domain  $D$ , i.e.,  $\mathbf{X}_D := \{X\} \cup \{X_i^- | D_i^- = D, i \in [n]\}$ . Finally, compute  $q_{\varphi, \mathbf{X}, \mathbf{D}}(D | Z) = \sum_{X' \in \mathbf{X}_D} q_{\varphi, \mathbf{x}}(X' | Z)$ . The resulting criterion is

$$
\mathcal {L} _ {\mathrm {C A D}} (\varphi , \psi) := \mathbb {E} _ {p _ {\mathbf {D}, \mathbf {X}, \mathbf {A}, Z}} \left[ - \log q _ {\psi , \mathbf {A}} \left(A ^ {+} \mid Z\right) + \lambda \log \left(\sum_ {X ^ {\prime} \in \mathbf {X} _ {D}} q _ {\varphi , \mathbf {X}} \left(X ^ {\prime} \mid Z\right)\right) \right]. \tag {8}
$$

In Appx. C.4, we also derive a conditional variation of CAD that minimizes  $\mathrm{I}[Z; D \mid Y]$ , which can be used when labels are available and supervised augmentations are used.

# 4.2.2 ENTROPY BOTTLENECK (ENT)

Our second domain bottleneck is the entropy bottleneck (Ent) that minimizes  $\mathrm{H}[Z] = \mathrm{I}[Z;X]\geq$ $\operatorname {I}[Z;D]$ , where the first equality uses the determinism of the encoder. Ent enforces support match by removing all information that is not needed to maximize I[Z; A]. In particular, minimizing  $\operatorname {I}[Z;X]$  is more stringent than  $\operatorname {I}[Z;D]$ , as it not only matches the representations across domains but also inside a domain. Although Ent removes more information than what is necessary compared to CAD, it has the advantage of not requiring access to domain samples  $\mathbf{D}$ , which are rarely accessible in SSL.

We consider the standard variational bound used in neural compression (Balle et al., 2016; Theis et al., 2017),  $\mathrm{H}[Z] \leq \mathbb{E}_{p_Z}[-\log q_\theta(Z)]$ , where an entropy model  $q_\theta(Z)$  is used. This leads to,

$$
\mathcal {L} _ {\text {E n t}} (\psi , \varphi , \theta) := \mathbb {E} _ {p _ {X, \mathbf {A}, Z}} \left[ - \log q _ {\psi , \mathbf {A}} \left(A ^ {+} \mid Z\right) - \lambda \log q _ {\theta} (Z) \right]. \tag {9}
$$

# 5 RELATED WORK

Provably robust representations  $\mathbf{Z}$  under covariate shift. Ben-David et al. (2010a) bounds the target risk using source risk and a divergence between source and target distributions. They do not consider representation learning, but in our setting, this implies that matching the marginal of  $Z$  while minimizing  $\mathrm{R}[Y|Z]$  is sufficient for optimality. Ben-David et al. (2010b) suggests that  $\mathrm{R}[Y|Z]$  is not sufficient, and Zhao et al. (2019) also prove that one should minimize the joint  $\mathrm{R}[Y|Z]$  instead of the source  $\mathrm{R}^{d_s}[Y|Z]$  risk. Similarly, des Combes et al. (2020) shows that matching the conditional  $p_{Y|Z,d} = p_{Y|Z}$  is sufficient. Johansson et al. (2019) take this further by proving that matching only the support of  $Z$  is also sufficient. Our work distinguishes itself from those and other related work on three key aspects: (i) We are the first to provide the set of necessary and sufficient conditions for robust representations; (ii) We prove that one can learn optimal  $Z^*$  with SSL using only large samples of inputs  $X$  and domain-covering augmentations  $A$ . (iii) We consider a general DG setting which deals with a less stringent generalized covariate shift and works for all standard losses and  $\mathcal{V}$  in ML. Still, our work is more specific than others, as we consider idealized DG and unrestricted predictors  $\mathcal{H}$ . Our theory could be combined with Dubois et al.'s (2020), who provide conditions for optimal generalization from finite samples and constrained  $\mathcal{H}$  in supervised learning.

Practical objectives for DG. The most popular DG methods aim to learn domain-invariant representation by minimizing various divergences between the conditionals  $p_{Z|d}$  and marginals  $p_{Z}$  (Long et al., 2015; Ganin et al., 2016; Sun & Saenko, 2016; Long et al., 2017; Li et al., 2018a; Shen et al., 2018; Nguyen et al., 2021). Others propose matching the conditional  $p_{Z|y,d}$  across domains instead (Gong et al., 2016; Li et al., 2018b; Tachet des Combes et al., 2020). These regularizers would all be valid domain bottlenecks  $\mathrm{B}[Z, D]$ . Another line of work aims at learning  $Z$  with invariant predictors  $p_{Y|z,d}$  across domains (e.g., Arjovsky et al., 2019; Krueger et al., 2021; Li et al., 2021). However, none of these methods outperform ERM with fair model selections (Gulrajani & Lopez-Paz, 2021).

# 6 EXPERIMENTS

In our experiments, we aimed to: (i) verify our theoretical results in practice; (ii) investigate our proposed representation learning objectives in practical DG; (iii) take advantage of pretrained SSL models (in particular, CLIP) to achieve powerful models for DG. Unless stated otherwise, we consider a two-stage training setup. First, the representation learner ("the representor") trains an encoder  $p_{Z|X}$  using a specified objective and freezes it. Then, the person performing predictions ("the learner")

![](images/bae9654b062b569b8e08d23b6e003f9cbd82c18cd173492cfcaf9c9408676505.jpg)  
(a) Effect of different objectives

![](images/7ffc87399bac500dcbbba1b6b6f69a212004bd43b525008a44ba37fade1c72fe.jpg)  
(b) Effect of  $\lambda$

![](images/83463e124978ba30ce58b6de800236fa58d7605be693834590700a7abd2467ca.jpg)  
Figure 3: (a) Adding bottlenecks significantly improves the worst-case DG performance and using domain-covering (DC) augmentations  $(\mathrm{H}[A|Z])$  performs as well as with labels  $(\mathrm{R}[Y|Z])$ . (b) Increasing the domain bottleneck weight  $\lambda$  will improve target performance until it decreases source performance. (c) DC augmentations are crucial but approx. DC aug. might be also be sufficient.  
(c) Effect of augmentations

trains her predictor  $h$  from  $Z$  by minimizing the risk on source data. Finally, the representation  $Z$  and predictor  $h$  are evaluated on target data. In all experiments, the learner uses a linear classifier for  $h$ . For the Ent bottleneck, we used Balle et al.'s (2018) entropy model. For the CAD bottleneck we used its conditional version whenever labels were available. When a model contains no domain bottleneck, we label it as "Base". For experimental details and additional results see Appxs. D and E.

# 6.1 SCIENTIFIC SETTING: OPTIMAL REPRESENTATIONS FOR WORST-CASE DG

To validate our theory, we studied optimal representations in a scientific setup that is as close to our IDG framework as possible with log-loss  $\ell$ . In particular, we used the PACS dataset (Li et al., 2017) and approximated the idealized DG by treating the dataset as the population distribution, i.e., we did not split datasets into train and test sets. To approximate the worst-case source predictor, we followed Dubois et al. (2020) by incorporating the wrongly labeled target data to the source domain. The experimental setup goes as follows: (i) the representative trains a ResNet-18 (He et al., 2016) to minimize the objective on labeled data from all domains; (ii) the learner trains a worst-case source classifier  $h$  on every possible pair of (source, target); (iii) the negative target risk (log likelihood) for each  $h$  is evaluated. We reported the log likelihood averaged over 5 seeds. For more realistic scenarios (i.e. non-idealized average-case DG) see Appx. E.2 which replicates the following results.

Do our domain bottlenecks improve worst-case DG? In Fig. 3a, we compare IDG performance of representations trained with (Ent, CAD) and without (Base) domain bottlenecks. We see that both bottlenecks significantly improve the worst-case DG, and nearly achieve the source-domain performance (0 log likelihood). This shows the importance of support match (Thm. 2) and the effectiveness of our bottlenecks to enforce it. In Appx. E.2, we show that bottlenecks also helps in practical scenarios, i.e., non-idealized average-case DG evaluated with accuracy  $(95.9\% \rightarrow 96.7\%)$ .

What is the effect of  $\lambda$ ? Fig. 3b shows the effect of the bottleneck weight  $\lambda$  on the worst-case target and source performance. We see that increasing  $\lambda$  will decrease the DG gap. As a result the target performance improves until  $\lambda \approx 10^{2}$ , where source performance starts to decrease.

What if the representative has access to domain-covering augmentations instead of labels? In Sec. 4.2, we provide a contrastive objective for using augmentations. To show the effectiveness of the objective, we compared minimizing  $\mathrm{H}[A|Z]$  using Eq. (7) to standard supervised risk minimization  $\mathrm{R}[Y|Z]$ . We ensured domain coverage by using supervised augmentations (Fig. 2b). The  $1^{\text{st}}$  and  $2^{\text{nd}}$  row of Fig. 3a show that our objective performs similarly to direct label prediction.

How important is the choice of augmentations? Prop. 2 shows that domain-covering (DC) augmentations are sufficient for achieving IDG, but it does not give necessary conditions. Here we investigate the effect of using our loss with different choices of augmentations. Specifically, we used  $\mathcal{L}_{\mathrm{CAD}}$  with five augmentations. The first two are DC. 'Supervised': augment inputs inside the label class across all domains as in Fig. 2b; 'SingleDom': augment inputs to same label samples from a fixed domain. The second two are not DC. 'Standard': standard SSL augmentations (Chen et al., 2020) as in Fig. 2a; 'IntraDom': augment inputs to same label and same domain samples. Finally, we consider 'ApproxDC', which is approximately DC by augmenting  $10\%$  of the time with 'Supervised' and  $90\%$  of the time with 'IntraDom'. Fig. 3c shows that the non-DC augmentations give terrible results compared to DC. Interestingly, 'ApproxDC' also performs very well, which suggests that approximately DC augmentations might be sufficient to learn optimal representations in practice.

What if the representative does not have access to target domains? Prop. 1 shows that DG without access to target domains is generally impossible. We empirically verified this by excluding a predefined target  $d_{t}$  domain from the representative's training set, i.e.,  $\mathcal{L}_{\mathrm{CAD}}$  is optimized on 3 of the 4 domains. The learner then trains a predictor  $h$  on each source. We finally evaluate each  $h$  on the target domain  $d_{t}$ , and average over choices of  $d_{t}$ . The resulting worst-case log likelihood was  $-4.2 \pm 0.2$ , which is significantly worse than when the representative had access to all domains  $(-0.8 \pm 0.2)$ .

# 6.2 APPROXIMATING OPTIMAL REPRESENTATIONS BY EXPLOITING PRETRAINED SSL

As discussed in Sec. 4.1, one can learn optimal representations for IDG by performing SSL with a domain bottleneck on a large sample of inputs  $X$  and domain covering augmentations  $A$ . This is nearly how CLIP was pretrained (SSL with 400M image-text pairs) except it did not include a domain bottleneck. In this section, we investigate how to take advantage of CLIP to approximate optimal representations for IDG. We did so in two simple steps. First, we froze the pretrained CLIP and added a multi-layer perceptron (MLP) that could effectively finetune CLIP's representations. Then, we trained the MLP by minimizing our CAD bottleneck and  $R[Y|Z]$  on the available data.

In all experiments, we used the standard DomainBed benchmark (with non-MNIST datasets) and protocol (Gulrajani & Lopez-Paz, 2021). In particular, we left out a target domain for evaluation and used the union of other domains for training both the encoder and the classifier. Contrary to our scientific setting, the representor does not get access to the target domain. All our representations were evaluated by fitting a linear classifier on source domains with source validation selection. As in DomainBed we selected the encoder based on 'oracle selection' over 10 hyperparameters, and reported the target accuracy averaged over all choices of targets and 5 random seeds. Due to space limit, we only included as baselines 'ERM' and 'DomainBed SOTA' which for each dataset is the best result over all baselines. The extended results and baselines are in Table 4. Details in Appx. D.3. We investigated two pretrained CLIP models with different number of parameters. The larger ViT-B/32 denoted 'CLIP L' and the smaller ResNet-50 denoted 'CLIP S'.

Table 1: Finetuning CLIP with our CAD bottleneck to achieve achieves SOTA performance on DomainBed. 'DomainBed SOTA' is the best (over models) result on each dataset.  

<table><tr><td>Algorithm</td><td>VLCS</td><td>PACS</td><td>OfficeHome</td><td>TerraIncognita</td><td>DomainNet</td></tr><tr><td>ERM</td><td>77.6 ± 0.3</td><td>86.7 ± 0.3</td><td>66.4 ± 0.5</td><td>53.0 ± 0.3</td><td>41.3 ± 0.1</td></tr><tr><td>DomainBed SOTA</td><td>79.9 ± 0.2</td><td>87.2 ± 0.1</td><td>68.4 ± 0.2</td><td>54.4 ± 0.3</td><td>41.8 ± 0.1</td></tr><tr><td>DINO + CAD</td><td>69.6 ± 0.6</td><td>76.1 ± 0.1</td><td>56.9 ± 0.5</td><td>25.9 ± 1.2</td><td>33.6 ± 0.1</td></tr><tr><td>CLIP S</td><td>81.1 ± 0.5</td><td>90.3 ± 0.2</td><td>70.6 ± 0.1</td><td>29.6 ± 0.8</td><td>47.7 ± 0.0</td></tr><tr><td>CLIP S + Base</td><td>81.6 ± 0.3</td><td>91.1 ± 0.3</td><td>70.6 ± 0.4</td><td>36.4 ± 0.7</td><td>46.7 ± 0.2</td></tr><tr><td>CLIP S + CAD</td><td>82.2 ± 0.3</td><td>92.4 ± 0.3</td><td>71.7 ± 0.6</td><td>36.1 ± 0.8</td><td>48.7 ± 0.1</td></tr><tr><td>CLIP L</td><td>80.7 ± 0.4</td><td>93.7 ± 0.8</td><td>79.9 ± 0.1</td><td>36.9 ± 0.6</td><td>52.8 ± 0.1</td></tr><tr><td>CLIP L + CAD</td><td>81.4 ± 0.8</td><td>94.7 ± 0.4</td><td>80.2 ± 0.2</td><td>39.7 ± 1.1</td><td>54.1 ± 0.1</td></tr></table>

Can we approximate optimal representations by exploiting pretrained CLIP? The last row in Table 1 shows that finetuning a large pretrained CLIP model with our CAD achieves SOTA on nearly all DomainBed benchmarks by a very large margin (see  $2^{\text{nd}}$  row). Note that the poor performance on TerraIncognita is likely because CLIP's dataset did not cover such images (camera traps monitoring animals). In Appx. E.2, we estimated the non-idealized DG performance of optimal representations on PACS (with access to all-domain labeled data) to be  $96.7\%$ , which is only  $2\%$  higher than CLIP L + CAD. This suggests that our simple SSL encoder might already be close to optimal.

Are gains due to the architectural differences? DomainBed's baselines finetuned an ImageNet (Deng et al., 2009) pretrained ResNet-50. In contrast, CLIP L pretrained a larger ViT. To decouple gains due to our objective from architectural gains, we evaluated ResNet-50 pretrained. Table 1 shows that CLIP S still outperforms DomainBed baselines. Our theory does not constrain the encoder and so we expect larger encoders to be better. Table 1 shows that CLIP L indeed outperforms CLIP S.

What is the effect of domain bottlenecks? In the last five rows of Table 1, we investigated the effect of finetuning with our CAD bottleneck. We see that for both CLIP L and CLIP S, it improves results by around  $1 \sim 2\%$ . These gains are due to the bottleneck, rather than due to the additional

MLP trained on source data as seen by 'CLIP S + Base'. Note that the raw CLIP S already significantly outperforms baselines. We hypothesize that this could be because SGD training of neural networks favors support match, e.g., by minimizing  $\mathrm{I}[X;Z]$  as suggested by Shwartz-Ziv & Tishby (2017).

Which pretrained SSL model to use? Our theory suggests that we can exploit pretrained SSL models as long as their augmentations are domain-covering and their training set covers desired domains. We investigated the effect of adapting SSL models that do not satisfy those properties by finetuning DINO (Caron et al., 2021), the current SOTA on SSL ImageNet. DINO only pretraind on ImageNet using standard augmentations. As a result, Table 1 shows that the finetuned DINO+CAD significantly underperforms compared to CLIP S and DomainBed baselines.

# 6.3 TOWARDS GENERIC ROBUST REPRESENTATIONS WITH SSL

In the previous section, we finetuned CLIP in a task specific fashion by optimizing  $\mathrm{R}[Y|Z]$  and our CAD bottleneck. To get generic (task agnostic) robust representations, one should instead directly use our objectives on a sufficiently large dataset with image-text augmentations. Unfortunately, we cannot fully train CLIP with our bottlenecks as we do not have access to CLIP's original dataset and sufficient compute. In this section, we aim to emulate such training of generic robust representations.

To do so we used LAION-400M (LAION, 2021) which is a public dataset that contains 400M web-crawled image-text pairs. Due to our computational budget, we again froze the pretrained CLIP L and only finetuned an additional MLP with our  $\mathcal{L}_{\mathrm{Ent}}$ . We used  $\mathcal{L}_{\mathrm{Ent}}$  as it only requires access to paired image  $X$  and text  $A$  but no prior information about domain  $D$ . As in CLIP's paper, we evaluated the learned representation  $Z$  in Taori et al.'s (2020) realistic setting, where a linear classifier  $h$  from  $Z$  is trained on ImageNet and tested on 7 natural distribution shift datasets. Details in Appx. D.4.

Would training CLIP with a bottleneck have improved its robustness? As shown in the last 2 rows of Table 2, finetuning CLIP L on LAION with  $\mathcal{L}_{\mathrm{Ent}}$  (LAION + Ent) outperforms finetuning without bottleneck (LAION + Base) on all 7 distribution shift datasets. This suggests that directly training CLIP with our Ent bottleneck would improve the robustness of learned representations. We hypothesize that the gains could be larger if SSL models trained  $\mathcal{L}_{\mathrm{Ent}}$  end-to-end. In Appx. E.4, we show similar results on DomainBed. Note that both models underperform the original CLIP L, likely due to non-end-to-end training and LAION data with (possibly) lower quality than CLIP's data.

Table 2: Finetuning CLIP L on LAION with an entropy bottleneck (LAION + Ent) improves its robustness compared to finetuning without (LAION + Base) on 7 distribution shift datasets. CLIP L is still better likely due to end-to-end training with higher quality data. IN denotes ImageNet.  

<table><tr><td></td><td>IN</td><td>IN-V2</td><td>IN-S</td><td>YT-BB</td><td>IN-Vid</td><td>ObjectNet</td><td>IN-A</td><td>IN-R</td><td>Avg.</td></tr><tr><td>CLIP L</td><td>75.2</td><td>64.2</td><td>41.0</td><td>58.4</td><td>71.6</td><td>42.8</td><td>27.5</td><td>62.9</td><td>52.6</td></tr><tr><td>LAION + Base</td><td>73.8</td><td>62.1</td><td>37.0</td><td>56.9</td><td>68.8</td><td>41.3</td><td>26.0</td><td>58.1</td><td>50.0</td></tr><tr><td>LAION + Ent</td><td>74.2</td><td>62.7</td><td>38.9</td><td>58.1</td><td>70.1</td><td>42.1</td><td>26.2</td><td>60.8</td><td>51.3</td></tr></table>

# 7 CONCLUSION

We gave a simple variational characterization of all representations on which source-risk minimizers are guaranteed to generalize to target domains that preserve the Bayes predictor. Similar to previous work, our theory strongly implies the need for target information when learning representations for domain generalization. Nevertheless, we identified a domain-covering property of data augmentations that make it possible to learn optimal representations from unlabelled data. Thus, we showed that it is possible to learn robust representations using only large sources of inputs  $X$  and augmentations  $A$ .

There are caveats that need to be addressed in future work. First, we studied an idealized DG, which assumes access to the population distributions. This gives insights into the challenges that are specific to DG, rather than finite sample challenges faced throughout ML. Second, we considered risk minimizers from an unconstrained hypothesis class. The support constraint can likely be weakened, if the hypothesis class is constrained. Finally, we focus only on optimal representations, but it would be interesting to characterize approximately optimal representations. Nevertheless, in this idealized setting, our characterization is a springboard from which all future objectives can be derived, and, in general, it brings us closer to the goal of robust machine learning systems.

Reproducibility Statement For our theoretical results, we include formal assumptions, statements, and proofs in Appxs. A and B. We include the detailed derivations of our algorithms in Appx. C. For our experiments, we include experimental details for reproducing our results in Appx. D and will release our code upon acceptance.

# REFERENCES

Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. Deep variational information bottleneck. arXiv preprint arXiv:1612.00410, 2016.  
Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.  
Johannes Balle, Valero Laparra, and Eero P Simoncelli. End-to-end optimized image compression. arXiv preprint arXiv:1611.01704, 2016.  
Johannes Balle, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. Variational image compression with a scale hyperprior. arXiv preprint arXiv:1802.01436, 2018.  
Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Danny Gutfreund, Joshua Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. 2019.  
Sara Beery, Grant Van Horn, and Pietro Perona. Recognition in terra incognita. In Proceedings of the European conference on computer vision (ECCV), pp. 456-473, 2018.  
Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from different domains. Machine Learning, 79(1):151-175, 2010a.  
Shai Ben-David, Tyler Lu, Teresa Luu, and David Pal. Impossibility theorems for domain adaptation. In Yee Whye Teh and Mike Titterington (eds.), Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pp. 129-136, Chia Laguna Resort, Sardinia, Italy, 13-15 May 2010b. PMLR. URL https://proceedings.mlr.press/v9/david10a.html.  
Léon Bottou. Large-scale machine learning with stochastic gradient descent. In Proceedings of COMPSTAT'2010, pp. 177-186. Springer, 2010.  
Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. arXiv preprint arXiv:2104.14294, 2021.  
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597-1607. PMLR, 2020.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248-255. IEEE, 2009.  
Remi Tachet des Combes, Han Zhao, Yu-Xiang Wang, and Geoffrey J. Gordon. Domain adaptation with conditional distribution matching and generalized label shift. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and HsuanTien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/bit/dfbfa7ddcfffeb581f50edcf9a0204bb-Abstract.html.  
Yann Dubois, Douwe Kiela, David J Schwab, and Ramakrishna Vedantam. Learning optimal representations with the decodable information bottleneck. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 18674-18690. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/d8ea5f53c1b1eb087ac2e356253395d8-Paper.pdf.

Yann Dubois, Benjamin Bloem-Reddy, Karen Ullrich, and Chris J. Maddison. Lossy compression for lossless prediction. arXiv preprint arXiv:2106.10800, 2021.  
Chen Fang, Ye Xu, and Daniel N Rockmore. Unbiased metric learning: On the utilization of multiple datasets and web images for softening bias. In Proceedings of the IEEE International Conference on Computer Vision, pp. 1657-1664, 2013.  
Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. The journal of machine learning research, 17(1):2096-2030, 2016.  
Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association, 102(477):359-378, 2007.  
Mingming Gong, Kun Zhang, Tongliang Liu, Dacheng Tao, Clark Glymour, and Bernhard Scholkopf. Domain adaptation with conditional transferable components. In International conference on machine learning, pp. 2839-2848. PMLR, 2016.  
Ian Goodfellow. Nips 2016 tutorial: Generative adversarial networks. arXiv preprint arXiv:1701.00160, 2016.  
Jean-Bastien Grill, Florian Strub, Florent Alché, Corentin Tallec, Pierre H Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent: A new approach to self-supervised learning. arXiv preprint arXiv:2006.07733, 2020.  
Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=1QdXeXDoWtI.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729-9738, 2020.  
Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. arXiv preprint arXiv:2006.16241, 2020.  
Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15262-15271, 2021.  
Fredrik D Johansson, David Sontag, and Rajesh Ranganath. Support and invertibility in domain-invariant representations. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 527-536. PMLR, 2019.  
Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. arXiv preprint arXiv:2004.11362, 2020.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Naveen Kodali, Jacob Abernethy, James Hays, and Zsolt Kira. On convergence and stability of gans. arXiv preprint arXiv:1705.07215, 2017.  
David Krueger, Ethan Caballero, Joern-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Remi Le Priol, and Aaron Courville. Out-of-distribution generalization via risk extrapolation (rex). In International Conference on Machine Learning, pp. 5815-5826. PMLR, 2021.

LAION. Laion-400m open dataset. https://laion.ai/laion-400-open-dataset, 2021. Accessed: 2021-09-14.  
Bo Li, Yifei Shen, Yezhen Wang, Wenzhen Zhu, Colorado J Reed, Jun Zhang, Dongsheng Li, Kurt Keutzer, and Han Zhao. Invariant information bottleneck for domain generalization. arXiv preprint arXiv:2106.06333, 2021.  
Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M Hospedales. Deeper, broader and artier domain generalization. In Proceedings of the IEEE international conference on computer vision, pp. 5542-5550, 2017.  
Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C Kot. Domain generalization with adversarial feature learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5400-5409, 2018a.  
Ya Li, Xinmei Tian, Mingming Gong, Yajing Liu, Tongliang Liu, Kun Zhang, and Dacheng Tao. Deep domain generalization via conditional invariant adversarial networks. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 624-639, 2018b.  
Mingsheng Long, Yue Cao, Jianmin Wang, and Michael Jordan. Learning transferable features with deep adaptation networks. In International conference on machine learning, pp. 97-105. PMLR, 2015.  
Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I Jordan. Deep transfer learning with joint adaptation networks. In International conference on machine learning, pp. 2208-2217. PMLR, 2017.  
A Tuan Nguyen, Toan Tran, Yarin Gal, Philip HS Torr, and Atulim Gunes Baydin. Kl guided domain adaptation. arXiv preprint arXiv:2106.07780, 2021.  
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1406-1415, 2019.  
Ben Poole, Sherjil Ozair, Aaron Van Den Oord, Alex Alemi, and George Tucker. On variational bounds of mutual information. In International Conference on Machine Learning, pp. 5171-5180. PMLR, 2019.  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 8748-8763. PMLR, 18-24 Jul 2021. URL https://proceedings.mlr.press/v139/radford21a.html.  
Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. DoImagenet classifiers generalize toImagenet? In International Conference on Machine Learning, pp. 5389-5400. PMLR, 2019.  
Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731, 2019.  
Vaishaal Shankar, Achal Dave, Rebecca Roelofs, Deva Ramanan, Benjamin Recht, and Ludwig Schmidt. Do image classifiers generalize across time? arXiv preprint arXiv:1906.02168, 2019.  
Jian Shen, Yanru Qu, Weinan Zhang, and Yong Yu. Wasserstein distance guided representation learning for domain adaptation. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.  
Ravid Shwartz-Ziv and Naftali Tishby. Opening the black box of deep neural networks via information. CoRR, abs/1703.00810, 2017. URL http://arxiv.org/abs/1703.00810.

Baochen Sun and Kate Saenko. Deep coral: Correlation alignment for deep domain adaptation. In European conference on computer vision, pp. 443-450. Springer, 2016.  
Remi Tachet des Combes, Han Zhao, Yu-Xiang Wang, and Geoffrey J Gordon. Domain adaptation with conditional distribution matching and generalized label shift. Advances in Neural Information Processing Systems, 33, 2020.  
Rohan Taori, Achal Dave, Vaishaal Shankar, Nicholas Carlini, Benjamin Recht, and Ludwig Schmidt. Measuring robustness to natural distribution shifts in image classification. arXiv preprint arXiv:2007.00644, 2020.  
Lucas Theis, Wenzhe Shi, Andrew Cunningham, and Ferenc Huszár. Lossy image compression with compressive autoencoders. arXiv preprint arXiv:1703.00395, 2017.  
Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000.  
Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5018-5027, 2017.  
Haohan Wang, Songwei Ge, Eric P Xing, and Zachary C Lipton. Learning robust global representations by penalizing local predictive power. arXiv preprint arXiv:1905.13549, 2019.  
Aolin Xu and Maxim Raginsky. Minimum excess risk in bayesian learning. arXiv preprint arXiv:2012.14868, 2020.  
Shen Yan, Huan Song, Nanxiang Li, Lincan Zou, and Liu Ren. Improve unsupervised domain adaptation with mixup training. arXiv preprint arXiv:2001.00677, 2020.  
Han Zhao, Remi Tachet Des Combes, Kun Zhang, and Geoffrey Gordon. On learning invariant representations for domain adaptation. In International Conference on Machine Learning, pp. 7523-7532. PMLR, 2019.
