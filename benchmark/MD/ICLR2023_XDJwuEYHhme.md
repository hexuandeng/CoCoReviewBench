# TOWARDS THE GENERALIZATION OF CONTRASTIVE SELF-SUPERVISED LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recently, self-supervised learning has attracted great attention, since it only requires unlabeled data for model training. Contrastive learning is one popular method for self-supervised learning and has achieved promising empirical performance. However, the theoretical understanding of its generalization ability is still limited. To this end, we define a kind of  $(\sigma, \delta)$ -measure to mathematically quantify the data augmentation, and then provide an upper bound of the downstream classification error rate based on the measure. It reveals that the generalization ability of contrastive self-supervised learning is related to three key factors: alignment of positive samples, divergence of class centers, and concentration of augmented data. The first two factors can be optimized by contrastive algorithms, while the third one is primarily determined by pre-defined data augmentation. With the above theoretical findings, we then study two canonical contrastive losses, InfoNCE and cross-correlation, to see how they satisfy the first two factors. Furthermore, we conduct various experiments to study the third factor, and observe that the downstream performance is highly correlated to the concentration of augmented data.

# 1 INTRODUCTION

Contrastive Self-Supervised Learning (SSL) has attracted great attention for its fantastic data efficiency and generalization ability in both computer vision (He et al., 2020; Chen et al., 2020a;b; Grill et al., 2020; Chen & He, 2021; Zbontar et al., 2021) and natural language processing (Fang et al., 2020; Wu et al., 2020; Giorgi et al., 2020; Gao et al., 2021; Yan et al., 2021). It learns the representation through a large number of unlabeled data and artificially defined self-supervision signals (i.e., regarding the augmented views of a data sample as positive samples). The model is updated by encouraging the features of positive samples close to each other. To overcome the feature collapse issue, different kinds of losses (e.g., InfoNCE (Chen et al., 2020a; He et al., 2020) and cross-correlation (Zbontar et al., 2021)) and training strategies (e.g., stop gradient (Grill et al., 2020; Chen & He, 2021)) are proposed.

In spite of the empirical success of contrastive SSL in terms of their generalization ability on downstream tasks, the theoretical understanding is still limited. Arora et al. (2019) propose a theoretical framework to show the provable downstream performance of contrastive SSL based on the InfoNCE loss. However, their results rely on the assumption that positive samples are drawn from the same latent class, instead of the augmented views of a data point as in practice. Wang & Isola (2020) propose alignment and uniformity to explain the downstream performance, but they are empirical indicators and lack of theoretical generalization guarantees. Both of the above works avoid characterizing the important role of data augmentation, which is the key to the success of contrastive SSL, since the only human knowledge is injected via data augmentation. Recently, HaoChen et al. (2021) propose to model the augmented data as a graph and study contrastive SSL from a matrix decomposition perspective, but it is only applicable to their own spectral contrastive loss.

Besides the limitations of existing contrastive SSL theories, there are also some interesting empirical observations that have not been unraveled theoretically yet. For example, why does the richer data augmentation lead to the more clustered structure in the embedding space (Figure 1) as well as the better downstream performance (also observed by Chen et al. (2020a))? Why is aligning positive samples (augmented from the "same data point") able to gather the samples from the "same latent class" into a cluster (Figure 1c)? More interestingly, decorrelating components of representation like

![](images/81e3c0895aeee792484a851ab0da8cc5dedfb42bedeb2e64c6f2c984407cc1b5.jpg)  
(a) Initial

![](images/6948b411b75487aebb5f482cd7254fb4e2adb3848fc468501db0b004f4dc239f.jpg)  
Figure 1: SimCLR's embedding space with different richnesses of data augmentations on CIFAR-10.  
(b) Only color distortion

![](images/3cfe31e3a0d96a1d718ed4e87a5077ba85302ccf1e0fb56f2d119f9a463c530a.jpg)  
(c) Multiple transformations

Barlow Twins (Zbontar et al., 2021) does not directly optimize the geometry of embedding space, but it still results in the clustered structure. Why is this?

In this paper, we focus on exploring the generalization ability of contrastive SSL provably, which can also provide insights for understanding the above interesting observations. We start with understanding the role of data augmentation in contrastive SSL. Intuitively, samples from the same latent class are likely to have similar augmented views, which are mapped to the close locations in the embedding space. Since the augmented views of each sample are encouraged to be clustered in the embedding space by contrastive learning, different samples from the same latent class tend to be pulled closer. As an example, let's consider two images of dogs with different backgrounds (Figure 2). If we augment them with transformation "crop", we may get two similar views (dog heads), whose representations (gray points in the embedding space) are close. As the

![](images/b08dc2395b91484e26004effe5cc0bd6e809c8785f384c00f60dd8f517b3f982.jpg)  
Figure 2: Mechanism of Clustering

augmented views of each dog image are enforced to be close in the embedding space due to the objective of contrastive learning, the representations of two dog images (green and blue points) will be pulled closer to their augmented views (gray points). In this way, aligning positive samples is able to gather samples from the same class, and thus results in the clustered embedding space. Following the above intuition, we define the augmented distance between two samples as the minimum distance between their augmented views, and further introduce the  $(\sigma, \delta)$ -augmentation to measure the concentration of augmented data, i.e., for each latent class, the proportion of samples located in a ball with diameter  $\delta$  (w.r.t. the augmented distance) is larger than  $\sigma$ .

With the mathematical description of data augmentation settled, we then prove an upper bound of downstream classification error rate in Section 3. It reveals that the generalization of contrastive SSL is related to three key factors. The first one is the alignment of positive samples, which is a common objective that contrastive algorithms aim to optimize. The second one is the divergence of class centers, which prevents the collapse of representation. The third factor is the concentration of augmented data, i.e., a sharper concentration of augmented data indicates a better generalization error bound. We remark that the first two factors, alignment and divergence, can be optimized by contrastive algorithms, while the third factor is primarily decided by the pre-determined data augmentation. Thus, data augmentation plays a role as important as contrastive algorithms in contrastive SSL.

We then study the above three factors in more depth. We rigorously prove that not only the InfoNCE loss but also the cross-correlation loss (which does not directly optimize the geometry of embedding space) can satisfy the first two factors in Section 4. For the third factor, we conduct various experiments on the real-world datasets and observe that the downstream performance of contrastive SSL is highly correlated to the concentration of augmented data in Section 5.

In summary, our contributions include: 1) proposing a novel  $(\sigma, \delta)$ -measure to quantify the data augmentation; 2) proposing a theoretical framework for contrastive SSL, which suggests that alignment, divergence, and concentration are key factors of generalization ability; 3) provably verifying that not only the InfoNCE loss but also the cross-correlation loss satisfy the alignment and divergence; 4) empirically showing that the concentration w.r.t. the proposed augmented distance is highly related to the downstream performance.

# RELATED WORK

Algorithms of Contrastive SSL. Early works such as MoCo (He et al., 2020) and SimCLR (Chen et al., 2020a), use the InfoNCE loss to pull the positive samples close while enforcing them away from the negative samples in the embedding space. These methods require large batch sizes (Chen et al., 2020a), memory banks (He et al., 2020), or carefully designed negative sampling strategies (Hu et al., 2021). To obviate these, some recent works get rid of negative samples and prevent representation collapse by cross-correlation loss (Zbontar et al., 2021; Bardes et al., 2021) or training strategies (Grill et al., 2020; Chen & He, 2021). In this paper, we mainly study the effectiveness of the InfoNCE loss and the cross-correlation loss, and do not enter the discussion of training strategies.

Theoretical Understandings of Contrastive SSL. Most theoretical analysis is based on the InfoNCE loss, and lack of understanding of recently proposed cross-correlation loss (Zbontar et al., 2021). Early works understand the InfoNCE loss based on maximizing the mutual information (MI) between positive samples (Oord et al., 2018; Bachman et al., 2019; Hjelm et al., 2018; Tian et al., 2019; 2020). However, Tschannen et al. (2019) find that optimizing tighter bounds of MI does not imply better representations. Thus, MI may not fully explain the success of InfoNCE. Besides, Arora et al. (2019) directly analyze the generalization of InfoNCE loss based on the assumption that positive samples are drawn from the same latent classes, which is different from practical contrastive algorithms. Furthermore, HaoChen et al. (2021) study contrastive SSL from a matrix decomposition perspective, but it is only applicable to their spectral contrastive loss. The behavior of InfoNCE is also studied from the perspective of alignment and uniformity (Wang & Isola, 2020), sparse coding model (Wen & Li, 2021), and the "expansion" assumption (Wei et al., 2020).

# 2 PROBLEM FORMULATION

Given a number of unlabeled training data i.i.d. drawn from an unknown distribution, each sample belongs to one of  $K$  latent classes  $C_1, C_2, \ldots, C_K$ . Based on an augmentation set  $A$ , the set of potential positive samples generated from a data point  $\pmb{x}$  is denoted as  $A(\pmb{x})$ . We assume that  $\pmb{x} \in A(\pmb{x})$  for any  $\pmb{x}$ , and samples from different classes never transfer to the same augmented sample, i.e.,  $\cap_{k=1}^{K} A(C_k) = \emptyset$ . Notation  $\|\cdot\|$  stands for  $\ell_2$ -norm or Frobenius norm for vectors and matrices.

Contrastive SSL aims to learn an encoder  $f$ , such that positive samples are closely aligned. In order to make the samples from different latent classes far away from each other, a class of methods (Chen et al., 2020a; He et al., 2020) use the InfoNCE loss to push away negative pairs, formulated as

$$
\mathcal{L}_{\text{InfoNCE}} = -\underset { \begin{array}{c}\boldsymbol {x},\boldsymbol{x}^{\prime}\\ \boldsymbol{x}^{-}\in A(\boldsymbol{x}^{\prime}) \end{array} }{\mathbb{E}}\underset { \begin{array}{c}\boldsymbol{x}_{1},\boldsymbol{x}_{2}\in A(\boldsymbol{x})\\ \boldsymbol{x}^{-}\in A(\boldsymbol{x}^{\prime}) \end{array} }{\mathbb{E}}\log \frac{e^{f(\boldsymbol{x}_{1})^{\top}f(\boldsymbol{x}_{2})}}{e^{f(\boldsymbol{x}_{1})^{\top}f(\boldsymbol{x}_{2})} + e^{f(\boldsymbol{x}_{1})^{\top}f(\boldsymbol{x}^{-})}},
$$

where  $\pmb{x}$ ,  $\pmb{x}'$  are two random data points. Some other methods (e.g., Barlow Twins (Zbontar et al., 2021)) use the cross-correlation loss to decorrelate the components of representation, formulated as

$$
\mathcal {L} _ {\mathrm {C r o s s - C o r r}} = \sum_ {i = 1} ^ {d} (1 - F _ {i i}) ^ {2} + \lambda \sum_ {i = 1} ^ {d} \sum_ {i \neq j} F _ {i j} ^ {2},
$$

where  $F_{ij} = \mathbb{E}_{\boldsymbol{x}}\mathbb{E}_{\boldsymbol{x}_1,\boldsymbol{x}_2\in A(\boldsymbol{x})}[f_i(\boldsymbol{x}_1)f_j(\boldsymbol{x}_2)], d$  is the dimension of encoder  $f$ , and encoder  $f$  is normalized as  $\mathbb{E}_{\boldsymbol{x}}\mathbb{E}_{\boldsymbol{x}'\in A(\boldsymbol{x})}[f_i(\boldsymbol{x}')^2] = 1$  for each dimension  $i$ .

The standard evaluation of contrastive SSL is to train a linear classifier over the learned representation using labeled data and regard its performance as the indicator. To simplify the analysis, we instead consider a non-parametric classifier – nearest neighbor (NN) classifier:

$$
G_{f}(\boldsymbol {x}) = \operatorname *{arg  min}_{k\in [K]}\| f(\boldsymbol {x}) - \mu_{k}\| ,
$$

where  $\mu_k \coloneqq \mathbb{E}_{\boldsymbol{x} \in C_k} \mathbb{E}_{\boldsymbol{x}' \in A(\boldsymbol{x})}[f(\boldsymbol{x}')]$  is the center of class  $C_k$ . In fact, the NN classifier is a special case of linear classifier, since it can be reformulated as  $G_f(\boldsymbol{x}) = \arg \max_{k \in [K]} (Wf(\boldsymbol{x}) + b)_k$ , where the  $k$ -th row of  $W$  is  $\mu_k$  and  $b_k = -\frac{1}{2}\|\mu_k\|^2$ . Therefore, the directly learned linear classifier used in practice should perform better than the NN classifier. In this paper, we use the classification

error rate to quantify the performance of  $G_{f}$ , formulated as

$$
\operatorname {E r r} \left(G _ {f}\right) = \sum_ {k = 1} ^ {K} \mathbb {P} \left[ G _ {f} (\boldsymbol {x}) \neq k, \forall \boldsymbol {x} \in C _ {k} \right].
$$

Our goal is to study why contrastive SSL is able to achieve a small  $\mathrm{Err}(G_f)$ .

# 3 GENERALIZATION GUARANTEE OF CONTRASTIVE SSL

Based on the NN classifier, if the samples are well clustered by latent classes in the embedding space, the error rate  $\mathrm{Err}(G_f)$  should be small. Thus, one expects to have a small intra-class distance  $\mathbb{E}_{\pmb{x}_1,\pmb{x}_2\in C_k}\| f(\pmb{x}_1) - f(\pmb{x}_2)\|^2$  for an encoder  $f$  learned by contrastive learning. However, contrastive algorithms only control the alignment of positive samples  $\mathbb{E}_{\pmb{x}_1,\pmb{x}_2\in A(\pmb{x})}\| f(\pmb{x}_1) - f(\pmb{x}_2)\|^2$ . To bridge the gap between these two distances, we need to take a close look at the role of data augmentation.

Motivated by Figure 2 introduced in Section 1, for a given augmentation set  $A$ , we define the augmented distance between two samples as the minimum distance between their augmented views:

$$
d _ {A} \left(\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2}\right) = \min  _ {\boldsymbol {x} _ {1} ^ {\prime} \in A \left(\boldsymbol {x} _ {1}\right), \boldsymbol {x} _ {2} ^ {\prime} \in A \left(\boldsymbol {x} _ {2}\right)} \| \boldsymbol {x} _ {1} ^ {\prime} - \boldsymbol {x} _ {2} ^ {\prime} \| . \tag {1}
$$

For the dog images in Figure 2, although they are quite different at the pixel level, they contain similar semantic meanings. Meanwhile, they have a small augmented distance. Thus, the semantic distance can be partially characterized by the proposed augmented distance. Based on the augmented distance, we now introduce the  $(\sigma, \delta)$ -augmentation to measure the concentration of augmented data.

Definition 1  $((\sigma, \delta)$ -Augmentation). The augmentation set  $A$  is called a  $(\sigma, \delta)$ -augmentation, if for each class  $C_k$ , there exists a subset  $C_k^0 \subseteq C_k$  (called a main part of  $C_k$ ), such that both  $\mathbb{P}[\pmb{x} \in C_k^0] \geq \sigma \mathbb{P}[\pmb{x} \in C_k]$  where  $\sigma \in (0,1]$  and  $\sup_{\pmb{x}_1, \pmb{x}_2 \in C_k^0} d_A(\pmb{x}_1, \pmb{x}_2) \leq \delta$  hold.

In other words, the main-part samples locate in a ball with diameter  $\delta$  (w.r.t. the augmented distance) and its proportion is larger than  $\sigma$ . Larger  $\sigma$  and smaller  $\delta$  indicate the sharper concentration of augmented data. One can verify that for any  $A' \supseteq A$  with richer augmentations, we have  $d_{A'}(\pmb{x}_1, \pmb{x}_2) \leq d_A(\pmb{x}_1, \pmb{x}_2)$  for any  $\pmb{x}_1, \pmb{x}_2$ . Therefore, richer data augmentations lead to sharper concentration as  $\delta$  gets smaller. With Definition 1, our analysis will focus on the samples in the main parts with good alignment, i.e.,  $(C_1^0 \cup \dots \cup C_K^0) \cap S_\varepsilon$ , where  $S_\varepsilon := \{\pmb{x} \in \cup_{k=1}^{K} C_k : \forall \pmb{x}_1, \pmb{x}_2 \in A(\pmb{x}), \|f(\pmb{x}_1) - f(\pmb{x}_2)\| \leq \varepsilon\}$  is the set of samples with  $\varepsilon$ -close representations among augmented data. Furthermore, we let  $R_\varepsilon := \mathbb{P}\left[S_\varepsilon\right]$ , which is provably small with good alignment (Theorem 2).

Lemma 3.1. For a  $(\sigma, \delta)$ -augmentation with main part  $C_k^0$  of each class  $C_k$ , if all samples belonging to  $(C_1^0 \cup \dots \cup C_K^0) \cap S_\varepsilon$  can be correctly classified by a classifier  $G$ , then its classification error rate  $\mathrm{Err}(G)$  is upper bounded by  $(1 - \sigma) + R_\varepsilon$ .

The proof is deferred to the appendix. The above lemma presents a simple sufficient condition to guarantee the generalization ability on downstream tasks. Based on it, we need to further explore when samples in  $(C_1^0\cup \dots \cup C_K^0)\cap S_\varepsilon$  can be all correctly classified by the NN classifier.

For simplicity, we assume that encoder  $f$  is normalized by  $\| f\| = r$ , and it is  $L$ -Lipschitz continuity, i.e., for any  $\pmb{x}_1,\pmb{x}_2$ ,  $\| f(\pmb{x}_1) - f(\pmb{x}_2)\| \leq L\| \pmb{x}_1 - \pmb{x}_2\|$ . We let  $p_k\coloneqq \mathbb{P}[\pmb {x}\in C_k]$  for any  $k\in [K]$ .

Lemma 3.2. Given a  $(\sigma, \delta)$ -augmentation used in contrastive SSL, for any  $\ell \in [K]$ , if  $\mu_{\ell}^{\top}\mu_{k} < r^{2}\left(1 - \rho_{\ell}(\sigma, \delta, \varepsilon) - \sqrt{2\rho_{\ell}(\sigma, \delta, \varepsilon)} - \frac{\Delta_{\mu}}{2}\right)$  holds for all  $k \neq \ell$ , then every sample  $\mathbf{x} \in C_{\ell}^{0} \cap S_{\varepsilon}$  can be correctly classified by the NN classifier  $G_{f}$ , where  $\rho_{\ell}(\sigma, \delta, \varepsilon) = 2(1 - \sigma) + \frac{R_{\varepsilon}}{p_{\ell}} + \sigma\left(\frac{L\delta}{r} + \frac{2\varepsilon}{r}\right)$  and  $\Delta_{\mu} = 1 - \min_{k \in [K]} \| \mu_{k} \|^{2} / r^{2}$ .

With Lemma 3.1 and 3.2, we can directly obtain the generalization guarantee of contrastive SSL:

Theorem 1. Given a  $(\sigma, \delta)$ -augmentation used in contrastive SSL, if

$$
\mu_ {\ell} ^ {\top} \mu_ {k} <   r ^ {2} \left(1 - \rho_ {m a x} (\sigma , \delta , \varepsilon) - \sqrt {2 \rho_ {m a x} (\sigma , \delta , \varepsilon)} - \frac {\Delta_ {\mu}}{2}\right) \tag {2}
$$

holds for any pair of  $(\ell, k)$  with  $\ell \neq k$ , then the downstream error rate of NN classifier  $G_{f}$

$$
\operatorname {E r r} \left(G _ {f}\right) \leq (1 - \sigma) + R _ {\varepsilon}, \tag {3}
$$

where  $\rho_{max}(\sigma, \delta, \varepsilon) = 2(1 - \sigma) + \frac{R_{\varepsilon}}{\min_{\ell} p_{\ell}} + \sigma \left(\frac{L \delta}{r} + \frac{2 \varepsilon}{r}\right)$  and  $\Delta_{\mu} = 1 - \min_{k \in [K]} \| \mu_k \|^2 / r^2$ .

The proof is deferred to the appendix. To better understand the above theorem, let us first consider a simple case that any two samples from the latent same class at least own a same augmented view  $(\sigma = 1,\delta = 0)$ , and the positive samples are perfectly aligned after contrastive learning  $(\varepsilon = 0,R_{\varepsilon} = 0)$ . In this case, the samples from the same latent class are embedded to a single point on the hypersphere, and thus arbitrarily small positive angle  $\frac{\langle\mu_{\ell},\mu_k\rangle}{\|\mu_{\ell}\|\cdot\|\mu_k\|} < 1$  is enough to distinguish them by the NN classifier. In fact, one can quickly verify that  $\rho_{max}(\sigma ,\delta ,\varepsilon) = \Delta_{\mu} = 0$  holds in the above case. According to Theorem 1, if  $\mu_{\ell}^{\top}\mu_{k} / r^{2} < 1 - \rho_{max}(\sigma ,\delta ,\varepsilon) - \sqrt{2\rho_{max}(\sigma,\delta,\varepsilon)} -\frac{\Delta_{\mu}}{2} = 1$ , then  $\mathrm{Err}(G_f) = 0$ , i.e., NN classifier can correctly recognize every sample when  $\mu_{\ell}^{\top}\mu_{k} / r^{2} < 1$ . Thus, the condition suggested by Theorem 1 is exactly the same as the intuition.

Theorem 1 implies three key factors to the success of contrastive SSL. The first one is the alignment of positive samples, which is a common objective that contrastive algorithms aim to optimize. Better alignment enables smaller  $R_{\varepsilon}$ , which directly decreases the generalization error bound (3). The second factor is the divergence of class centers, i.e., the distance between class centers should be large enough (small  $\mu_{\ell}^{\top}\mu_{k}$ ). The divergence condition (2) is related to the alignment  $(R_{\varepsilon})$  and data augmentation  $(\sigma ,\delta)$ . Better alignment and sharper concentration indicate smaller  $\rho_{max}(\sigma ,\delta ,\varepsilon)$ , and hence looser divergence condition. The third factor is the concentration of augmented data. When  $\delta$  is given, sharper concentration implies larger  $\sigma$ , which directly affects the generalization error bound (3). For example, richer data augmentations lead to sharper concentration (see the paragraph below Definition 1), and hence better generalization error bound. Only the first two factors can be optimized by contrastive algorithms, and we will provably show how it can be done via two concrete examples in Section 4. In contrast, the third factor is primarily decided by the pre-defined data augmentation and is unrelated to algorithms. We will empirically study how the concentration of augmented data affects the downstream performance in Section 5. In summary, Theorem 1 provides a framework for different algorithms to analyze their generalization abilities.

Compared with the alignment and uniformity proposed by Wang & Isola (2020), both of the works have the same meaning of "alignment" since it is the objective that contrastive algorithms aim to optimize, but our "divergence" is fundamentally different from their "uniformity". Uniformity requires "all data" uniformly distributed on the embedding hypersphere, while our divergence characterizes the cosine distance between "class centers". We do not require the divergence to be as large as better, instead, the divergence condition can be loosened by better alignment and concentration properties. As an example, consider the case below Theorem 1. Since all the samples from the same latent class are embedded to a single point on the hypersphere, in that case, an arbitrarily small positive angle (arbitrarily small divergence) is enough to distinguish them. More importantly, alignment and uniformity are empirical predictors for downstream performance, while our alignment and divergence have explicit theoretical guarantees (Theorem 1) for the generalization of contrastive SSL. Moreover, Wang & Isola (2020) does not consider the crucial effect of data augmentation. In fact, with bad concentration (e.g., using identity transformation as data augmentation), "perfect" alignment along with "perfect" uniformity still can not imply good downstream performance.

# 3.1 UPPER BOUND  $R_{\varepsilon}$  VIA ALIGNMENT

We now upper bound  $R_{\varepsilon}$  via the alignment

$$
\mathcal {L} _ {\text {a l i g n}} (f) := \underset {\boldsymbol {x}} {\mathbb {E}} \underset {\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2} \in A (\boldsymbol {x})} {\mathbb {E}} \| f (\boldsymbol {x} _ {1}) - f (\boldsymbol {x} _ {2}) \| ^ {2}, \tag {4}
$$

which is a common objective of contrastive losses. Recall that  $R_{\varepsilon}$  can be rewritten as

$$
R _ {\varepsilon} = \mathbb {P} \left[ \boldsymbol {x} \in \cup_ {k = 1} ^ {K} C _ {k} \colon \sup _ {\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2} \in A (\boldsymbol {x})} \| f (\boldsymbol {x} _ {1}) - f (\boldsymbol {x} _ {2}) \| > \varepsilon \right],
$$

and there is a gap between "sup operator" in  $R_{\varepsilon}$  and  $\mathbb{E}$  operator" in  $\mathcal{L}_{\mathrm{align}}(f)$ , which cannot be simply derived by concentration inequalities.

We separate the augmentation set  $A$  as discrete transformations  $\{A_{\gamma}(\cdot)\colon \gamma \in [m]\}$  and continuous transformations  $\{A_{\theta}(\cdot)\colon \theta \in [0,1]^n\}$ . For example, random cropping or flipping can be categorized into the discrete transformation, while the others like random color distortion or Gaussian blur can be regarded as the continuous transformation parameterized by the augmentation strength  $\theta$ . Without loss of generality, we assume that for any given  $\pmb{x}$ , its augmented data are uniformly random sampled, i.e.,  $\mathbb{P}[x' = A_{\gamma}(x)] = \frac{1}{2m}$  and  $\mathbb{P}[x'\in \{A_\theta (x)\colon \theta \in \Theta \} ] = \frac{\mathrm{vol}(\Theta)}{2}$  for any  $\Theta \subseteq [0,1]^n$ , where  $\mathrm{vol}(\Theta)$  denotes the volume of  $\Theta$ . For the continuous transformation, we further assume that the transformation is  $M$ -Lipschitz continuous w.r.t.  $\theta$ , i.e.,  $\| A_{\theta_1}(\pmb {x}) - A_{\theta_2}(\pmb {x})\| \leq M\| \theta_1 - \theta_2\|$  for any  $\pmb {x},\theta_1,\theta_2$ . With the above setting, we have the following theorem (proof is deferred to the appendix).

Theorem 2. If encoder  $f$  is  $L$ -Lipschitz continuous, then

$$
R _ {\varepsilon} ^ {2} \leq \eta (\varepsilon) ^ {2} \cdot \underset {\boldsymbol {x}} {\mathbb {E}} \underset {\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2} \in A (\boldsymbol {x})} {\mathbb {E}} \| f (\boldsymbol {x} _ {1}) - f (\boldsymbol {x} _ {2}) \| ^ {2} = \eta (\varepsilon) ^ {2} \cdot \mathcal {L} _ {\mathrm {a l i g n}} (f),
$$

where  $\eta (\varepsilon) = \inf_{h\in \left(0,\frac{\varepsilon}{2\sqrt{n}LM}\right)}\frac{4\max\{1,m^2h^{2n}\}}{h^{2n}(\varepsilon - 2\sqrt{n} LMh)} = \mathcal{O}\left(\frac{1}{\varepsilon}\right).$

The above theorem confirms that, with good alignment,  $R_{\varepsilon}$  is guaranteed to be small.

# 4 CONTRASTIVE LOSSES MEET ALIGNMENT AND DIVERGENCE

We now study two canonical contrastive losses, the InfoNCE loss and the cross-correlation loss, to see how they can achieve the good alignment (small  $\mathcal{L}_{\mathrm{align}}(f)$ ) and good divergence (small  $\mu_k^\top \mu_\ell$ ).

# 4.1 WARMUP: INFONCE LOSS

The population loss of InfoNCE (Chen et al., 2020a; He et al., 2020) is well known as:

$$
\mathcal{L}_{\text{InfoNCE}} = -\underset { \begin{array}{c}\boldsymbol {x},\boldsymbol{x}^{\prime}\\ \boldsymbol{x}_{1},\boldsymbol{x}_{2}\in A(\boldsymbol {x})\\ \boldsymbol{x}^{-}\in A(\boldsymbol{x}^{\prime}) \end{array} }{\mathbb{E}}\underset { \begin{array}{c}\boldsymbol{x}_{1},\boldsymbol{x}_{2}\in A(\boldsymbol {x})\\ \boldsymbol{x}^{-}\in A(\boldsymbol{x}^{\prime}) \end{array} }{\mathbb{E}}\log \frac{e^{f(\boldsymbol{x}_{1})^{\top}f(\boldsymbol{x}_{2})}}{e^{f(\boldsymbol{x}_{1})^{\top}f(\boldsymbol{x}_{2})} + e^{f(\boldsymbol{x}_{1})^{\top}f(\boldsymbol{x}^{-})}},
$$

where encoder  $f$  is normalized by  $\| f\| = 1$ . It can be divided into two parts:

$$
\begin{array}{l} \mathcal {L} _ {\text {I n f o N C E}} = \underset { \begin{array}{c} \boldsymbol {x}, \boldsymbol {x} ^ {\prime} \\ \boldsymbol {x} ^ {-} \in A \left(\boldsymbol {x} ^ {\prime}\right) \end{array} } {\mathbb {E}} \underset { \begin{array}{c} \boldsymbol {x} _ {1}, \boldsymbol {x} _ {2} \in A (\boldsymbol {x}) \\ \boldsymbol {x} ^ {-} \in A \left(\boldsymbol {x} ^ {\prime}\right) \end{array} } {\mathbb {E}} \left[ - f \left(\boldsymbol {x} _ {1}\right) ^ {\top} f \left(\boldsymbol {x} _ {2}\right) + \log \left(e ^ {f \left(\boldsymbol {x} _ {1}\right) ^ {\top} f \left(\boldsymbol {x} _ {2}\right)} + e ^ {f \left(\boldsymbol {x} _ {1}\right) ^ {\top} f \left(\boldsymbol {x} ^ {-}\right)}\right) \right] \tag {5} \\ = \underbrace {\frac {1}{2} \underset {\mathbf {x} _ {1} , \mathbf {x} _ {2} \in A (\mathbf {x})} {\mathbb {E}} [ \| f (\mathbf {x} _ {1}) - f (\mathbf {x} _ {2}) \| ^ {2} ] - 1} _ {=: \mathcal {L} _ {1} (f)} + \underbrace {\underset {\mathbf {x} , \mathbf {x} ^ {\prime}} {\mathbb {E}} \underset {\mathbf {x} _ {1} , \mathbf {x} _ {2} \in A (\mathbf {x})} {\mathbb {E}} \left[ \log \left(e ^ {f (\mathbf {x} _ {1}) ^ {\top} f (\mathbf {x} _ {2})} + e ^ {f (\mathbf {x} _ {1}) ^ {\top} f (\mathbf {x} ^ {-})}\right) \right]} _ {=: \mathcal {L} _ {2} (f)}. \\ \end{array}
$$

Regardless of the constant factors,  $\mathcal{L}_1(f)$  is exactly the alignment term in (4). Next, we take a close look at  $\mathcal{L}_2(f)$  to see how it links to the divergence condition required by Theorem 1.

Theorem 3. Assume that encoder  $f$  with norm 1 is  $L$ -Lipschitz continuous. If the augmented data is  $(\sigma, \delta)$ -augmented, then for any  $\varepsilon \geq 0$  and  $k \neq \ell$ , we have

$$
\mu_ {k} ^ {\top} \mu_ {\ell} \leq \log \left(\exp \left\{\frac {\mathcal {L} _ {2} (f) + \tau (\sigma , \delta , \varepsilon , R _ {\varepsilon})}{p _ {k} p _ {\ell}} \right\} - \exp (1 - \varepsilon)\right),
$$

where  $\tau (\sigma ,\delta ,\varepsilon ,R_{\varepsilon})$  is a non-negative term, decreasing with smaller  $\varepsilon$ ,  $R_{\varepsilon}$  or sharper concentration of augmented data, and  $\tau (\sigma ,\delta ,\varepsilon ,R_{\varepsilon}) = 0$  when  $\sigma = 1$ ,  $\delta = 0$ ,  $\varepsilon = 0$ ,  $R_{\varepsilon} = 0$ .

The specific formulation of  $\tau (\sigma ,\delta ,\varepsilon ,R_{\varepsilon})$  and the proof are deferred to the appendix. We remark that data augmentation  $(\sigma ,\delta)$ , parameter  $\varepsilon$ , and  $p_k,p_\ell$  are pre-determined before training procedure, and thus the upper bound of  $\mu_k^\top \mu_\ell$  in Theorem 3 varies only with  $\mathcal{L}_2(f)$  and  $R_{\varepsilon}$ , positively.

Therefore, minimizing  $\mathcal{L}_{\mathrm{InfoNCE}} = \mathcal{L}_1(f) + \mathcal{L}_2(f)$  leads to both small  $\mathcal{L}_1(f)$  and small  $\mathcal{L}_2(f)$ . Small  $\mathcal{L}_1(f)$  indicates good alignment  $\mathcal{L}_{\mathrm{align}}(f)$ , as well as small  $R_{\varepsilon}$  (Theorem 2). Small  $\mathcal{L}_2(f)$  along with small  $R_{\varepsilon}$  indicates good divergence (small  $\mu_k^\top \mu_\ell$ ) by Theorem 3. Hence, optimizing the InfoNCE loss can achieve both good alignment and good divergence. According to Theorem 1

and Theorem 2, the generalization ability of encoder  $f$  on the downstream task is implied, i.e.,  $\mathrm{Err}(G_f) \leq (1 - \sigma) + \eta(\varepsilon) \sqrt{2 + 2\mathcal{L}_1(f)}$ , when the upper bound of  $\mu_k^\top \mu_\ell$  in Theorem 3 is smaller than the threshold in Theorem 1.

It is worth mentioning that the form of InfoNCE is critical to meeting the requirement of divergence, which is found when we prove Theorem 3. For example, let us consider the contrastive loss (5) formulated in a linear form<sup>1</sup> instead of LogExp such that

$$
\mathcal{L}^{\prime}(f) = \underset { \begin{array}{c}\boldsymbol {x},\boldsymbol{x}^{\prime}\\ \boldsymbol{x}^{-}\in A(\boldsymbol{x}^{\prime}) \end{array} }{\mathbb{E}}\underset { \begin{array}{c}\boldsymbol{x}_{1},\boldsymbol{x}_{2}\in A(\boldsymbol{x})\\ \boldsymbol{x}^{-}\in A(\boldsymbol{x}^{\prime}) \end{array} }{\mathbb{E}}\left[-f(\boldsymbol{x}_{1})^{\top}f(\boldsymbol{x}_{2}) + \lambda f(\boldsymbol{x}_{1})^{\top}f(\boldsymbol{x}^{-})\right] = \mathcal{L}_{1}(f) + \lambda \mathcal{L}_{2}^{\prime}(f),
$$

where  $\mathcal{L}_2'(f)$  is the negative-pair term weighted by some  $\lambda > 0$ . Due to the independence between  $\pmb{x}$  and  $\pmb{x}'$ , we have  $\mathcal{L}_2'(f) = \| \mathbb{E}_x \mathbb{E}_{\pmb{x}_1 \in A(\pmb{x})} [f(\pmb{x}_1)] \|^2$ . Therefore, minimizing  $\mathcal{L}_2'(f)$  only leads to the representation with zero mean. Unfortunately, the objective of zero mean with  $\| f \| = 1$  can not obviate the dimensional collapse (Hua et al., 2021) of the model. For example, the encoder  $f$  can map the input data from multi classes into two points in the opposite directions on the hypersphere. This justifies the observation in (Wang & Liu, 2021): the uniformity of the encoder on the embedded hypersphere becomes worse when the temperature of the loss increases, where the loss degenerates to  $\mathcal{L}'(f)$  with infinite temperature.

# 4.2 CROSS-CORRELATION LOSS

Cross-correlation loss is first introduced by Barlow Twins (Zbontar et al., 2021). In contrast to InfoNCE loss, it trains the model via decorrelating the components of representation instead of directly optimizing the geometry of embedding space, but it is still observed to have clustered embedding space. To explore this, we study the cross-correlation loss in detail and show how it implicitly optimizes the alignment and divergence required by Theorem 1.

The population loss of cross-correlation can be formulated as

$$
\mathcal {L} _ {\text {C r o s s - C o r r}} = \sum_ {i = 1} ^ {d} \left(1 - \underset {\boldsymbol {x}} {\mathbb {E}} \underset {\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2} \in A (\boldsymbol {x})} {\mathbb {E}} [ f _ {i} (\boldsymbol {x} _ {1}) f _ {i} (\boldsymbol {x} _ {2}) ]\right) ^ {2} + \lambda \sum_ {i \neq j} \left(\underset {\boldsymbol {x}} {\mathbb {E}} \underset {\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2} \in A (\boldsymbol {x})} {\mathbb {E}} [ f _ {i} (\boldsymbol {x} _ {1}) f _ {j} (\boldsymbol {x} _ {2}) ]\right) ^ {2},
$$

with normalization condition of  $\mathbb{E}_{\boldsymbol{x}}\mathbb{E}_{\boldsymbol{x}_1\in A(\boldsymbol{x})}[f_i(\boldsymbol{x}_1)] = 0$  and  $\mathbb{E}_{\boldsymbol{x}}\mathbb{E}_{\boldsymbol{x}_1\in A(\boldsymbol{x})}[f_i(\boldsymbol{x}_1)^2] = 1$  for each  $i\in [d]$ , where  $d$  is the output dimension of encoder  $f$ . Positive coefficient  $\lambda$  balances the importance between diagonal and non-diagonal elements of cross-correlation matrix. When  $\lambda = 1$ , the above loss is exactly the difference between the cross-correlation matrix and identity matrix. Similar to Section 4.1, we first divide the loss into two parts, by defining

$$
\mathcal{L}_{1}(f):= \sum_{i = 1}^{d}\left(1 - \underset {\boldsymbol{x}}{\mathbb{E}}\underset {\boldsymbol{x}_{1},\boldsymbol{x}_{2}\in A(\boldsymbol {x})}{\mathbb{E}}[f_{i}(\boldsymbol{x}_{1})f_{i}(\boldsymbol{x}_{2})]\right)^{2}\text{and}\mathcal{L}_{2}(f):= \left\| \underset {\boldsymbol{x}}{\mathbb{E}}\underset {\boldsymbol{x}_{1},\boldsymbol{x}_{2}\in A(\boldsymbol {x})}{\mathbb{E}}[f(\boldsymbol{x}_{1})f(\boldsymbol{x}_{2})^{\top}] - I_{d}\right\|^{2}.
$$

In this way, the cross-correlation loss becomes  $\mathcal{L}_{\mathrm{Cross - Corr}} = (1 - \lambda)\mathcal{L}_1(f) + \lambda \mathcal{L}_2(f)$ . Then, we connect  $\mathcal{L}_1(f)$  and  $\mathcal{L}_2(f)$  with the alignment and divergence, respectively.

Lemma 4.1. For a given encoder  $f$ , the alignment  $\mathcal{L}_{\mathrm{align}}(f)$  in (4) is upper bounded via  $\mathcal{L}_1(f)$ , i.e.,

$$
\mathcal {L} _ {\text {a l i g n}} (f) = \underset {\boldsymbol {x}} {\mathbb {E}} \underset {\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2} \in A (\boldsymbol {x})} {\mathbb {E}} \| f (\boldsymbol {x} _ {1}) - f (\boldsymbol {x} _ {2}) \| ^ {2} \leq 2 \sqrt {d \mathcal {L} _ {1} (f)},
$$

where  $d$  is the output dimension of encoder  $f$ .

The above lemma connects  $\mathcal{L}_1(f)$  with  $\mathcal{L}_{\mathrm{align}}(f)$ , indicating that the diagonal elements of the cross-correlation matrix determine the alignment of positive samples. Next, we will link  $\mathcal{L}_2(f)$  to the divergence  $\mu_k^\top \mu_\ell$ . It is challenging because  $\mathcal{L}_2(f)$  is designed for reducing the redundancy between the encoder's output units, not for optimizing the geometry of embedding space.

Theorem 4. Assume that encoder  $f$  with norm  $\sqrt{d}$  is  $L$ -Lipschitz continuous. If the augmented data is  $(\sigma, \delta)$ -augmented, then for any  $\varepsilon \geq 0$  and  $k \neq \ell$ , we have

$$
\mu_ {k} ^ {\top} \mu_ {\ell} \leq \sqrt {\frac {2}{p _ {k} p _ {\ell}} \left(\mathcal {L} _ {2} (f) + \tau^ {\prime} (\sigma , \delta , \varepsilon , R _ {\varepsilon}) - \frac {d - K}{2}\right)},
$$

where  $\tau'(\sigma, \delta, \varepsilon, R_{\varepsilon})$  is an upper bound of  $\| \mathbb{E}_{\boldsymbol{x}} \mathbb{E}_{\boldsymbol{x}_1, \boldsymbol{x}_2 \in A(\boldsymbol{x})} [f(\boldsymbol{x}_1) f(\boldsymbol{x}_2)^\top] - \sum_{k=1}^{K} p_k \mu_k \mu_k^\top \|^2$ .

It is also called simple contrastive loss in some literature.

The specific formulation of  $\tau'(\sigma, \delta, \varepsilon, R_{\varepsilon})$  and proof are deferred to the appendix. Here we remark that  $\tau'(\sigma, \delta, \varepsilon, R_{\varepsilon})$  is a non-negative term, decreasing with smaller  $\varepsilon, R_{\varepsilon}$  or sharper concentration of augmented data, and  $\tau'(\sigma, \delta, \varepsilon, R_{\varepsilon}) = 0$  when  $\sigma = 1, \delta = 0, \varepsilon = 0, R_{\varepsilon} = 0$ . Since data augmentation  $(\sigma, \delta)$ , parameter  $\varepsilon$ , and  $p_k, p_\ell$  are pre-determined before training procedure, the upper bound of  $\mu_k^\top \mu_\ell$  in Theorem 4 varies only with  $\mathcal{L}_2(f)$  and  $R_{\varepsilon}$ , positively.

Therefore, minimizing  $\mathcal{L}_{\mathrm{Cross - Corr}}$  leads to small  $\mathcal{L}_1(f)$ , as well as small  $\mathcal{L}_2(f)$ . Small  $\mathcal{L}_1(f)$  indicates good alignment  $\mathcal{L}_{\mathrm{align}}(f)$  by Lemma 4.1 and small  $R_{\varepsilon}$  by Theorem 2. Small  $\mathcal{L}_2(f)$  along with small  $R_{\varepsilon}$  indicates good divergence (small  $\mu_k^\top \mu_\ell$ ) by Theorem 4. Hence, decorrelating the components of representation can achieve both good alignment and good divergence. According to Theorem 1 and Theorem 2, the generalization ability of encoder  $f$  on the downstream task is implied, i.e.,  $\mathrm{Err}(G_f) \leq (1 - \sigma) + \sqrt{2} \eta(\varepsilon) d^{\frac{1}{4}} \mathcal{L}_1(f)^{\frac{1}{4}}$ , when the upper bound of  $\mu_k^\top \mu_\ell$  in Theorem 4 is smaller than the threshold in Theorem 1.

Beyond the above two widely used contrastive learning losses, we further analyze a very recently proposed  $t$ -InfoNCE loss (Hu et al., 2022), which is a  $t$ -SNE style loss inspired by stochastic neighbor embedding. We show that it can also achieve good alignment and divergence in the appendix.

# 5 EMPIRICAL STUDY OF CONCENTRATION OF AUGMENTED DATA

Theorem 1 reveals that sharper concentration of augmented data w.r.t. the proposed augmented distance implies better generalization error bound regardless of algorithm. In this section, we empirically study the relationship between the concentration level and the real downstream performance.

Basic Setup. Our experiments are conducted on CIFAR-10 and CIFAR-100 (Krizhevsky, 2009). We consider 5 different kinds of transformations for performing data augmentations: (a) random cropping; (b) random Gaussian blur; (c) color dropping (i.e., randomly converting images to grayscale); (d) color distortion; (e) random horizontal flipping. We test different combinations of transformations via various SSL algorithms such as SimCLR (Chen et al., 2020a), Barlow Twins (Zbontar et al., 2021), MoCo (He et al., 2020), and SimSiam (Chen & He, 2021). We use ResNet-18 (He et al., 2016) as the encoder, and the other settings such as projection head remain the same as the original settings of algorithms. Each model is trained with a batch size of 512 and 800 epochs. To evaluate the quality of the encoder, we follow the KNN evaluation protocol (Wu et al., 2018).

Different Richness of Augmentations. We compose all 5 kinds of transformations together, and then successively drop one of the composed transformations from (e) to (b) to conduct 5 experiments for each dataset (Table 1). We observe that the downstream performance monotonously gets worse with the decrease of transformation number, under all four SSL algorithms, on both CIFAR-10 and CIFAR-100. Notice that richer augmentation implies sharper concentration (see the paragraph below Definition 1), and thus the concentration becomes less sharp from top to bottom for each dataset. Therefore, we observe that downstream performance becomes better with sharper concentration.

We also observe that (c) color dropping and (d) color distortion have a great impact on the performance of all algorithms. According to our theoretical framework, these two transformations enable the augmented data to vary in a very wide range, which makes the augmented distance (1) largely decrease. As an intuitive example, if the right dog image in Figure 2 is replaced by a Husky image, only with random cropping, one will get two dog heads with similar shapes but different colors, which still have a large augmented distance. Instead, if color distortion is further applied, one can get two similar dog heads both in shape and color. Therefore, these two dog images have similar augmented views, and thus their augmented distance (1) becomes very small. Notice that small augmented distance (1) indicates sharp concentration (small  $\delta$  in Definition 1). Therefore, we observe that dramatic change in concentration leads to wildly fluctuating downstream performance.

Different Strength of Augmentations. We fix (a) random cropping and (d) color distortion as data augmentation, and vary the strength of (d) in  $\{1, \frac{1}{2}, \frac{1}{4}, \frac{1}{8}\}$  to construct 4 groups of augmentations with different strength levels (Table 2). We observe that the downstream performance monotonously decreases with weaker color distortions, under all four SSL algorithms, on both CIFAR-10 and CIFAR-100. Recall that a stronger color distortion makes the augmented data vary in a wider range, leading to a smaller augmented distance (1) and thus sharper concentration. Therefore, we observe again that downstream performance becomes better with sharper concentration.

Table 1: Downstream performance under different richness of augmentations.  

<table><tr><td rowspan="2">Dataset</td><td colspan="5">Transformations</td><td colspan="4">Accuracy</td></tr><tr><td>(a)</td><td>(b)</td><td>(c)</td><td>(d)</td><td>(e)</td><td>SimCLR</td><td>Barlow Twins</td><td>MoCo</td><td>SimSiam</td></tr><tr><td rowspan="5">CIFAR-10</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>89.76 ± 0.12</td><td>86.91 ± 0.09</td><td>90.12 ± 0.12</td><td>90.59 ± 0.11</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>88.48 ± 0.22</td><td>85.38 ± 0.37</td><td>89.69 ± 0.11</td><td>89.34 ± 0.09</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td></td><td></td><td>83.50 ± 0.14</td><td>82.00 ± 0.59</td><td>86.78 ± 0.07</td><td>85.38 ± 0.09</td></tr><tr><td>✓</td><td>✓</td><td></td><td></td><td></td><td>63.23 ± 0.05</td><td>67.83 ± 0.94</td><td>75.12 ± 0.28</td><td>63.27 ± 0.30</td></tr><tr><td>✓</td><td></td><td></td><td></td><td></td><td>62.74 ± 0.18</td><td>67.77 ± 0.69</td><td>74.94 ± 0.22</td><td>61.47 ± 0.74</td></tr><tr><td rowspan="5">CIFAR-100</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>57.74 ± 0.12</td><td>57.99 ± 0.29</td><td>64.19 ± 0.14</td><td>63.48 ± 0.16</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>55.43 ± 0.10</td><td>55.22 ± 0.25</td><td>62.50 ± 0.28</td><td>60.31 ± 0.41</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td></td><td></td><td>45.10 ± 0.25</td><td>50.40 ± 0.64</td><td>57.04 ± 0.21</td><td>51.42 ± 0.14</td></tr><tr><td>✓</td><td>✓</td><td></td><td></td><td></td><td>28.01 ± 0.18</td><td>34.11 ± 0.59</td><td>40.18 ± 0.04</td><td>26.26 ± 0.30</td></tr><tr><td>✓</td><td></td><td></td><td></td><td></td><td>27.95 ± 0.09</td><td>34.05 ± 1.13</td><td>39.63 ± 0.31</td><td>25.90 ± 0.83</td></tr></table>

Table 2: Downstream performance under different strength of augmentations.  

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Color Distortion Strength</td><td colspan="4">Accuracy</td></tr><tr><td>SimCLR</td><td>Barlow Twins</td><td>MoCo</td><td>SimSiam</td></tr><tr><td rowspan="4">CIFAR-10</td><td>1</td><td>82.75 ± 0.24</td><td>82.58 ± 0.25</td><td>86.68 ± 0.05</td><td>82.50 ± 1.05</td></tr><tr><td>1/2</td><td>78.76 ± 0.18</td><td>81.88 ± 0.25</td><td>84.30 ± 0.14</td><td>81.80 ± 0.15</td></tr><tr><td>1/4</td><td>76.37 ± 0.11</td><td>79.64 ± 0.34</td><td>82.76 ± 0.09</td><td>78.80 ± 0.17</td></tr><tr><td>1/8</td><td>74.23 ± 0.16</td><td>77.96 ± 0.16</td><td>81.20 ± 0.12</td><td>76.09 ± 0.50</td></tr><tr><td rowspan="4">CIFAR-100</td><td>1</td><td>46.67 ± 0.42</td><td>50.39 ± 1.09</td><td>58.50 ± 0.51</td><td>49.94 ± 2.01</td></tr><tr><td>1/2</td><td>40.21 ± 0.05</td><td>48.76 ± 0.25</td><td>55.08 ± 0.09</td><td>46.27 ± 0.46</td></tr><tr><td>1/4</td><td>36.67 ± 0.08</td><td>46.22 ± 0.71</td><td>52.09 ± 0.18</td><td>42.02 ± 0.34</td></tr><tr><td>1/8</td><td>34.75 ± 0.20</td><td>44.72 ± 0.26</td><td>49.43 ± 0.16</td><td>36.26 ± 0.34</td></tr></table>

Different Composed Pairs of Transformations. To study the relationship between the concentration level and the corresponding downstream performance in a more fine-grained way, we compose transformations (a)-(e) in pairs to construct a total of  $\binom{5}{2}=10$  augmentations. Contrasted to the previous two groups of experiments, current composed augmentations do not have an apparent order of concentration levels. According to Definition 1, for a given  $\delta$ , a smaller  $(1-\sigma)$  corresponds to a sharper concentration. Thus, we mathematically compute  $(1-\sigma)$  (see appendix for details), and observe the correlation between classification error rate  $\mathrm{Err}(G_f)$  and  $(1-\sigma)$  under different  $\delta$  on CIFAR-10, based on the SimCLR model trained with 200 epochs.

Interestingly, downstream performance is surprisingly highly correlated to the concentration level (Figure 3). Specifically, if we fix one of composed transformations as (a), we find that both  $\mathrm{Err}(G_f)$  and  $(1 - \sigma)$  have the same order that  $(a,d) < (a,c) < (a,e) \approx (a,b)$ , under two values of  $\delta$ . Furthermore, among all 10 composed augmentations, augmentation  $(a,d)$  has the smallest value of  $(1 - \sigma)$ , while the corresponding performance is also the best one. In addition, we observe that the choice of  $\delta$  is not sensitive to the curve shape of  $(1 - \sigma)$ . These observations suggest that sharper concentration is most likely to have better downstream performance. This also provides an explanation for Figure 5 in Sim-CLR paper (Chen et al., 2020a) of why the composition of "crop & color" performs the best.

![](images/71e3f4024aa160ab8173e456430e28edf2028c1802dc641428b54f31312d32f5.jpg)  
Figure 3: The correlation between observed  $\operatorname{Err}(G_f)$  and computed value of  $(1 - \sigma)$ .

# 6 FUTURE WORK

One future work can be relaxing the assumption that  $\cap_{k=1}^{K} A(C_k) = \emptyset$ , corresponding to the situation that the data augmentation is too aggressive. Another possible direction is to refine the solution by taking the sample size into account, to justify the phenomenon that batch size is sensitive to the performance of SimCLR but not sensitive to Barlow Twins.

# REFERENCES

Sanjeev Arora, Hrishikesh Khandeparkar, Mikhail Khodak, Orestis Plevrakis, and Nikunj Saunshi. A theoretical analysis of contrastive unsupervised representation learning. arXiv preprint arXiv:1902.09229, 2019.  
Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In Advances in Neural Information Processing Systems, pp. 15535-15545, 2019.  
Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906, 2021.  
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709, 2020a.  
Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15750-15758, 2021.  
Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020b.  
Hongchao Fang, Sicheng Wang, Meng Zhou, Jiayuan Ding, and Pengtao Xie. Cert: Contrastive self-supervised learning for language understanding. arXiv preprint arXiv:2005.12766, 2020.  
Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. arXiv preprint arXiv:2104.08821, 2021.  
John M Giorgi, Osvald Nitski, Gary D Bader, and Bo Wang. Declutr: Deep contrastive learning for unsupervised textual representations. arXiv preprint arXiv:2006.03659, 2020.  
Jean-Bastien Grill, Florian Strub, Florent Alché, Corentin Tallec, Pierre H Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent: A new approach to self-supervised learning. arXiv preprint arXiv:2006.07733, 2020.  
Jeff Z HaoChen, Colin Wei, Adrien Gaidon, and Tengyu Ma. Provable guarantees for self-supervised deep learning with spectral contrastive loss. Advances in Neural Information Processing Systems, 34, 2021.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729-9738, 2020.  
R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. arXiv preprint arXiv:1808.06670, 2018.  
Qianjiang Hu, Xiao Wang, Wei Hu, and Guo-Jun Qi. Adco: Adversarial contrast for efficient learning of unsupervised representations from self-trained negative adversaries. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1074-1083, 2021.  
Tianyang Hu, Zhili Liu, Fengwei Zhou, Wenjia Wang, and Weiran Huang. Your contrastive learning is secretly doing stochastic neighbor embedding. arXiv preprint arXiv:2205.14814, 2022.  
Tianyu Hua, Wenxiao Wang, Zihui Xue, Sucheng Ren, Yue Wang, and Hang Zhao. On feature decorrelation in self-supervised learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9598-9608, 2021.  
Alex Krizhevsky. Learning multiple layers of features from tiny images. University of Toronto, 2009.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
Christos H Papadimitriou and Kenneth Steiglitz. Combinatorial optimization: algorithms and complexity. Courier Corporation, 1998.  
Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. arXiv preprint arXiv:1906.05849, 2019.  
Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning? arXiv preprint arXiv:2005.10243, 2020.  
Michael Tschannen, Josip Djolonga, Paul K Rubenstein, Sylvain Gelly, and Mario Lucic. On mutual information maximization for representation learning. arXiv preprint arXiv:1907.13625, 2019.  
Feng Wang and Huaping Liu. Understanding the behaviour of contrastive loss. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2495-2504, 2021.  
Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International Conference on Machine Learning, pp. 9929-9939. PMLR, 2020.  
Colin Wei, Kendrick Shen, Yining Chen, and Tengyu Ma. Theoretical analysis of self-training with deep networks on unlabeled data. arXiv preprint arXiv:2010.03622, 2020.  
Zixin Wen and Yuanzhi Li. Toward understanding the feature learning process of self-supervised contrastive learning. arXiv preprint arXiv:2105.15134, 2021.  
Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via nonparametric instance discrimination. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3733-3742, 2018.  
Zhuofeng Wu, Sinong Wang, Jiatao Gu, Madian Khabsa, Fei Sun, and Hao Ma. Clear: Contrastive learning for sentence representation. arXiv preprint arXiv:2012.15466, 2020.  
Yuanmeng Yan, Rumei Li, Sirui Wang, Fuzheng Zhang, Wei Wu, and Weiran Xu. Consert: A contrastive framework for self-supervised sentence representation transfer. arXiv preprint arXiv:2105.11741, 2021.  
Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised learning via redundancy reduction. arXiv preprint arXiv:2103.03230, 2021.
