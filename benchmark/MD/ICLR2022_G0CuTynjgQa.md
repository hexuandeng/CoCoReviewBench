# GENERALIZATION OF GANS AND OVERPARAMETERIZED MODELS UNDER LIPSCHITZ CONTINUITY

Anonymous authors

Paper under double-blind review

# ABSTRACT

Generative adversarial networks (GANs) are really complex, and little has been known about their generalization. The existing learning theories lack efficient tools to analyze generalization of GANs. To fill this gap, we introduce a novel tool to analyze generalization: Lipschitz continuity. We demonstrate its simplicity by showing generalization and consistency of overparameterized neural networks. We then use this tool to derive Lipschitz-based generalization bounds for GANs. In particular, our bounds show that penalizing the zero- and first-order informations of the GAN loss will improve generalization. Therefore, this work provides a unified theory for answering the long mystery of why imposing a Lipschitz constraint can help GANs to generalize well in practice.

# 1 INTRODUCTION

In Generative Adversarial Networks (GAN) (Goodfellow et al., 2014), we want to train a discriminator  $D$  and a generator  $G$  by solving the following problem:

$$
\min  _ {G} \max  _ {D} \mathbb {E} _ {x \sim P _ {d}} \log (D (x)) + \mathbb {E} _ {z \sim P _ {z}} \log (1 - D (G (z))) \tag {1}
$$

where  $P_{d}$  is a data distribution that generates real data, and  $P_{z}$  is some noise distribution.  $G$  is a mapping that maps a noise  $z$  to a point in the data space. After training,  $G$  can be used to generate novel but realistic data.

Since its introduction (Goodfellow et al., 2014), a significant progress has been made for developing GANs and for interesting applications (Hong et al., 2019). Some recent works (Brock et al., 2019; Zhang et al., 2019a; Karras et al., 2020b) can train a generator that produces synthetic images of extremely high quality. Nevertheless, little has been known about the generalization of the trained players. One main reason is that the training problem is unsupervised in nature and contains two players competing each other. Such a nature is entirely different from traditional learning problems (Mohri et al., 2018). The standard learning theories still lack an efficient tool to analyze GANs. Therefore, neural distance (Arora et al., 2017) was introduced for analyzing generalization of GANs. One major limitation of existing distance-based bounds (Arora et al., 2017; Zhang et al., 2018; Jiang et al., 2019; Husain et al., 2019) for generalization is the strong dependence on the capacity of the family, which defines the distance between two distributions, sometimes leading to trivial bounds. This limitation prevents us from fully understanding and identifying the key factors that contribute to the generalization of GANs.

This work has the following contributions:

$\triangleright$  We introduce Lipschitz continuity as a tool to analyze generalization of a learned function. This tool is surprisingly simple to analyze various complex models in general settings, including unsupervised and adversarial settings.  
> We show that Dropout or spectrally-normalized neural networks avoid the curse of dimensionality. The number of layers required to ensure good generalization is logarithmic in sample size. We further show consistency for overparameterized models. These results are significant, since existing theoretical works only apply to networks with no more than three layers and suffer from the curse of dimensionality. Hence this work provides a significant step toward answering the open theoretical issues of deep learning (Zhang et al., 2021).

$\triangleright$  Using Lipschitz continuity, we provide a comprehensive analysis on generalization of GANs which resolves the two open challenges in the GAN community: (i) Our bounds apply to any particular  $D$  or  $G$ , and hence overcome the major limitation of existing works; In particular, for the first time in the literature, we show that Dropout and spectral normalization can help GANs to avoid the curse of dimensionality. (ii) Lipschitz constraint is used popularly through various ways including gradient penalty (Gulrajani et al., 2017), spectral normalization (Miyato et al., 2018), dropout, and data augmentation. Our analysis provides an unified explanation for why imposing a Lipschitz constraint can help GANs to generalize well in practice.

Organization: We will review related work in the next section. Section 3 presents the bridge between Lipschitz continuity and generalization, and some analyses about deep neural networks. In Section 4, we analyze the generalization of GANs in various aspects. Section 5 concludes the paper.

# 2 RELATED WORK

Generalization in GANs: There are few efforts to analyze the generalization for GANs using the notion of neural distance,  $d_{\mathcal{D}}(P_d,P_g)$ , which is the distance between two distributions  $(P_d,P_g)$ , where  $\mathcal{D}$  is the discriminator family. Arora et al. (2017) analyze generalization by bounding the quantity  $|d_{\mathcal{D}}(P_d,P_g) - d_{\mathcal{D}}(\widehat{P}_d,\widehat{P}_g)|$ , where  $(\widehat{P}_d,\widehat{P}_g)$  are empirical versions of  $(P_d,P_g)$ . For a suitable choice of the loss  $V(P_d,P_z,D,G)$  in GANs, we can write  $|d_{\mathcal{D}}(P_d,P_g) - d_{\mathcal{D}}(\widehat{P}_d,\widehat{P}_g)| = |\max_{D\in \mathcal{D}}V(P_d,P_z,D,G) - \max_{D\in \mathcal{D}}V(\widehat{P}_d,\widehat{P}_z,D,G)|$ , where  $P_g$  is the induced distribution by putting samples from  $P_z$  through generator  $G$ . Both Arora et al. (2017) and Husain et al. (2019) analyze  $|\max_{D\in \mathcal{D}}V(P_d,P_z,D,G_o) - \max_{D\in \mathcal{D}}V(\widehat{P}_d,\widehat{P}_z,D,G_o)|$  to see generalization of a trained  $G_o$ , while (Zhang et al., 2018; Jiang et al., 2019) provide upper bounds for  $|\max_{D\in \mathcal{D}}V(P_d,\widehat{P}_z,D,G) - \min_{G\in \mathcal{G}}\max_{D\in \mathcal{D}}V(P_d,P_z,D,G)|$ . Note that those quantities of interest are non-standard in terms of learning theory.

A major limitation of those distance-based bounds (Arora et al., 2017; Zhang et al., 2018; Jiang et al., 2019; Husain et al., 2019) is the dependence on the notion of distance  $d_{\mathcal{D}}(\cdot, \cdot)$  which relies on the best  $D \in \mathcal{D}$  for measuring proximity between two distributions. The distance between two given distributions  $(\mu, \nu)$  may be small even when the two are far away (Arora et al., 2017). This is because there exists a perfect discriminator  $D$ , whenever  $\mu$  and  $\nu$  do not have overlapping supports (Arjovsky & Bottou, 2017). In those cases, a distance-based bound may be trivial. As a result, existing distance-based bounds are insufficient to understand generalization of GANs.

Qi (2020) shows a generalization bound for their proposed Loss-Sensitive GAN. Nonetheless, it is nontrivial to make their bound to work with other GAN losses. Wu et al. (2019) show that the discriminator will generalize if the learning algorithm is differentially private. Their concept of differential privacy basically requires that the learned function will change negligibly if the training set slightly changes. Such a requirement is known as algorithmic stability (Xu et al., 2010) and is nontrivial to assure in practice. Note that this assumption cannot be satisfied for GANs since their training is well-known to be unstable in practice.

Lipschitz continuity, stability, and generalization: Lipschitz continuity naturally appears in the formulation of Wasserstein GAN (WGAN) (Arjovsky et al., 2017). It was then quickly recognized as a key to improve various GANs (Fedus et al., 2018; Lucic et al., 2018; Mescheder et al., 2018; Kurach et al., 2019; Jenni & Favaro, 2019; Wu et al., 2019; Zhou et al., 2019; Qi, 2020; Chu et al., 2020). Gradient penalty (Gulrajani et al., 2017) and spectral normalization (Miyato et al., 2018) are two popular techniques to constraint the Lipschitz continuity of  $D$  or  $G$  w.r.t their inputs. Some other works (Mescheder et al., 2017; Nagarajan & Kolter, 2017; Sanjabi et al., 2018; Nie & Patel, 2019) suggest to control the Lipschitz continuity of  $D$  or  $G$  w.r.t their parameters. Data augmentation is another way to control the Lipschitz constant of the loss, and is really beneficial for training GANs (Zhao et al., 2020a;b; Zhang et al., 2020a; Tran et al., 2021). Those works empirically found that Lipschitz continuity can help improving stability and generalization of GANs. However, it has long been a mystery of why imposing a Lipschitz constraint can help GANs to generalize well. This work provides an unified explanation.

# 3 LIPSCHITZ CONTINUITY AND GENERALIZATION

In this section, we will point out the connection between Lipschitz continuity and generalization. We then discuss an application to two families of neural networks. Finally, we discuss about consistency of overparameterized models.

Notations: Consider a learning problem specified by a function/hypothesis class  $\mathcal{H}$ , an instance set  $\mathcal{Z}$  with diameter at most  $B$ , and a loss function  $f:\mathcal{H}\times \mathcal{Z}\to \mathbb{R}$  which is bounded by a constant  $C$ . Given a distribution  $P_{z}$  defined on  $\mathcal{Z}$ , the quality of a function  $h$  is measured by its expected loss  $F(P_{z},h) = \mathbb{E}_{z\sim P_{z}}[f(h,z)]$ . Since  $P_{z}$  is unknown, we need to rely on a finite training sample  $\pmb {S} = \{z_1,\dots,z_m\} \subset \mathcal{Z}$  and often work with the empirical loss  $F(\widehat{P}_z,h) = \mathbb{E}_{z\sim \widehat{P}_z}[f(h,z)] = \frac{1}{m}\sum_{z\in S}f(h,z)$ , where  $\widehat{P}_z$  is the empirical distribution defined on  $\pmb{S}$ . A learning algorithm  $\mathcal{A}$  will pick a function  $h_m\in \mathcal{H}$  based on input  $\pmb{S}$ , i.e.,  $h_m = \mathcal{A}(\mathcal{H},\pmb {S})$ .

We first establish the following result whose proof appears in Appendix A.

Theorem 1 (Lipschitz continuity  $\Rightarrow$  Generalization). If a loss  $f(h,z)$  is L-Lipschitz continuous w.r.t input  $z$  in a compact set  $\mathcal{Z}\subset \mathbb{R}^n$ , for any  $h\in \mathcal{H}$ , and  $\widehat{P}_z$  is the empirical distribution defined from m i.i.d. samples from distribution  $P_{z}$ , then  $\sup_{h\in \mathcal{H}}|F(P_z,h) - F(\widehat{P}_z,h)|$  is upper-bounded by

1.  $L\lambda + C\sqrt{(\lceil B^n\lambda^{-n}\rceil\log 4 - 2\log\delta) / m}$  with probability at least  $1 - \delta$ , for any constants  $\delta \in (0,1]$  and  $\lambda \in (0,B]$ .  
2.  $(LB + 2C)m^{-\alpha /n}$  with probability at least  $1 - 2\exp (-0.5m^{\alpha})$  , for any  $\alpha \leq n / (2 + n)$

This theorem tells that Lipschitz continuity is the key to ensure a function to generalize. Generalization can be better as the Lipschitz constant of the loss decreases. Note that there is a tradeoff between the Lipschitz constant and the expected loss  $F(P_z, h)$  of the learnt function. A smaller  $L$  means that both  $f$  and  $h$  are getting simpler and flatter, and hence may increase  $F(P_z, h)$ . In contrast, a decrease of  $F(P_z, h)$  may require  $h$  to be more complex and hence may increase the Lipschitz constants of  $f$ . Some recent works (Miyato et al., 2018; Gouk et al., 2021; Pauli et al., 2021) propose to put a penalty on the Lipschitz constant of  $h$  only. However, leaving open the Lipschitzness of  $f$  w.r.t  $h$  may not ensure the Lipschitz continuity of the loss.

Lipschitz continuity vs. Algorithmic robustness: Although the proof of Theorem 1 bases on algorithmic robustness (Xu & Mannor, 2012), Lipschitz-based bounds have two significant advantages. Firstly, the bounds in Theorem 1 are uniform which facilitates an analysis on consistency, whereas the bound by Xu & Mannor (2012) holds for a particular algorithm. Secondly, the assumption of Lipschitz continuity of the loss is more natural and practical than robustness of a learning algorithm. Indeed, it is sufficient to choose a family  $\mathcal{H}$  with Lipschitz continuous members to ensure Lipschitz continuity of various losses, including squared loss and hinge loss.

Theorem 1 presents generalization bounds in a general setting. Therefore those bounds are loose. Furthermore, the bound suffers from the curse of dimensionality. That is, in general, the number of samples should be exponential in the dimension  $n$  in order to achieve a small bound. Note that this limitation is common for any other approaches without further assumptions (Bach, 2017). For some special function classes, we can overcome this limitation as discussed next.

# 3.1 DEEP NEURAL NETWORKS THAT AVOID THE CURSE OF DIMENSIONALITY

We consider the two families of neural networks: one with bounded spectral norms for the weight matrices, and the other with Dropout. The following theorem whose proof appears in Appendix A.1 provides sharp bounds for the Lipschitz constant.

Theorem 2. Let fixed activation functions  $(\sigma_{1},\ldots ,\sigma_{K})$  , where  $\sigma_{i}$  is  $\rho_{i}$  -Lipschitz continuous. Let  $h_{\mathcal{W}}(x)\coloneqq \sigma_K(W_K\sigma_{K - 1}(W_{K - 1}\dots \sigma_1(W_1x)\dots))$  be the neural network associated with weight matrices  $(W_{1},\ldots ,W_{K})$  , and  $L_{h}$  be the Lipschitz constant of  $h$  . Let the bounds  $(s_1,\dots,s_K)$  and  $(b_{1},\ldots ,b_{K})$  be given.

Spectrally-normalized networks (SN-DNN): Let  $\mathcal{H}_{sn} = \{h_{\mathcal{W}}: \mathcal{W} = (W_1, \ldots, W_K), \| W_i \|_{\sigma} \leq s_i\}$ , where  $\| \cdot \|_{\sigma}$  is the spectral norm. Then  $\forall h \in \mathcal{H}_{sn}, L_h \leq \prod_{k=1}^{K} \rho_k s_k$ .

Dropout DNN: Let  $\mathcal{H}_{dr} = \{h_{\mathcal{W},q}: h_{\mathcal{W},q} = DrT(h_{\mathcal{W}},q), \| W_i\|_F \leq b_i\}$ , where  $DrT$  is the usual dropout training (Srivastava et al., 2014) with drop rate  $q$  for network  $h_{\mathcal{W}}$ , and  $\|\cdot\|_F$  is the Frobenius norm. Then  $\forall h \in \mathcal{H}_{dr}$ ,  $L_h \leq q^K \prod_{k=1}^K \rho_k b_k$ .

Some activation functions (e.g., SmoothReLU, Sigmoid, and Softmax) have small Lipschitz constants  $(\rho_{k} < 1)$ , while some others (e.g., ReLU, Leaky ReLU, Tanh) are 1-Lipschitz continuous over any compact set. This theorem suggests that the Lipschitz constant can be exponentially small as a neural network is deep (large  $K$ ) and uses Dropout at each layer, since  $q < 1$  is a popular choice in practice. On the other hand, the Lipschitz constant will be small if we control the spectral norms of weight matrices, e.g. by using spectral normalization (Miyato et al., 2018). The Lipschitz constant will be exponentially smaller as the neural network is deeper and the spectral norm at each layer is smaller than 1. This case often happens as observed by Miyato et al. (2018).

The generalization of spectrally-normalized (and Dropout) neural networks can be seen by combining Theorems 1 and 2. However such a trivial combination could not remove the curse of dimensionality. The following theorem shows stronger results whose proofs appear in Appendix A.1.

Theorem 3. Given the assumptions in Theorems 1 and 2, denote  $L_{f}$  as the Lipschitz constant of the loss  $f(h,z)$  w.r.t  $h$ . Assume that

1. SN-DNNs: For  $\mathcal{H}_{sn}$ , there exist  $q\in (0,1)$  and constant  $C_{sn}$  such that  $C_{sn}q^{K}\geq \prod_{k = 1}^{K}\rho_{k}s_{k}$ .  
2. Dropout DNNs: For  $\mathcal{H}_{dr}$  with drop rate  $q\in (0,1)$ , let  $C_{dr} = \prod_{k = 1}^{K}\rho_{k}b_{k}$ .

If the number of layers  $K \geq -\frac{1}{2} \log_q m$ , then for any  $\delta \in (0, 1]$  the followings hold with probability at least  $1 - \delta$ :

$$
\begin{array}{l} \sup  _ {h \in \mathcal {H} _ {s n}} | F (P _ {z}, h) - F (\hat {P} _ {z}, h) | \leq \left(C _ {s n} L _ {f} B + C \sqrt {\log 4 - 2 \log \delta}\right) m ^ {- 0. 5} \\ \sup _ {h \in \mathcal {H} _ {d r}} | F (P _ {z}, h) - F (\widehat {P} _ {z}, h) | \leq \left(C _ {d r} L _ {f} B + C \sqrt {\log 4 - 2 \log \delta}\right) m ^ {- 0. 5} \\ \end{array}
$$

The assumption of  $K \geq -\frac{1}{2} \log_q m$  is naturally met in practice. For example, when training from 1.2M images, constant  $q = 0.5$  requires  $K \geq 10$ , and  $q = 0.1$  requires  $K \geq 3$ . Note that Alexnet (Krizhevsky et al., 2012) has 8 layers and the generator of StyleGAN (Karras et al., 2021) has 18 layers. The assumption about SN-DNNs can be satisfied when choosing activations with Lipschitz constant  $\rho_k < 1$  or ensuring the spectral bound  $s_k < 1$  at any layer  $k$ . As mentioned before, such conditions are often satisfied in practice (Miyato et al., 2018) when using spectral normalization.

Comparison with state-of-the-art: Some recent studies (Bartlett et al., 2017; Neyshabur et al., 2018) provide generalization bounds for SN-DNNs for classification problems, using Radermacher complexity or PAC-Bayes. One major limitation of their works is that the sample size depends polynomially/exponentially on depth  $K$ . For SN-DNNs using ReLU, Golowich et al. (2020) improved the dependence to be linear in  $K$  if provided assumptions comparable with ours. In another view, when fixing  $m$ , their results require  $K = O(m)$  to get a meaningful generalization bound. This is impractical. In contrast, our result shows that it is sufficient to choose  $K$  which is logarithmic in  $m$ . Another limitation of the bounds in (Bartlett et al., 2017; Neyshabur et al., 2018; Golowich et al., 2020) is the dependence on  $1 / \gamma$ , where  $\gamma$  is the margin of the classification problem. Note that practical data may have a very small margin or may be inseparable. Hence those bounds are really limited and inapplicable to inseparable cases. On contrary, Theorem 3 holds in general settings, including inseparable classification and unsupervised problems.

Our result for Dropout DNNs holds in general settings including unsupervised learning. This is significant since state-of-the-art studies about Dropout (Arora et al., 2021; Mianjy & Arora, 2020; Mou et al., 2018) obtain efficient bounds only for networks with no more than 3 layers and for supervised learning. To the best of our knowledge, this work is the first showing that Dropout can help DNNs avoid the curse of dimensionality in general settings.

# 3.2 CONSISTENCY OF OVERPARAMERIZED MODELS

We have discussed generalization of a function by bounding the difference between the empirical and expected losses. In some situations, those bounds may not be enough, since both losses may be large inspite of their small difference. Next we consider consistency (Shalev-Shwartz et al., 2010) which helps us to see the goodness of a function compared with the best in their family.

Definition 1. A learning algorithm  $\mathcal{A}$  is said to be Consistent with rate  $\epsilon_{cons}(m)$  under distribution  $P_z$  if for all  $m$ ,  $\mathbb{E}_{\mathbf{S} \sim P_z^m} |F(P_z, \mathcal{A}(\mathcal{H}, \mathbf{S})) - F(P_z, h^*)| \leq \epsilon_{cons}(m)$ , where  $\epsilon_{cons}(m)$  must satisfy  $\epsilon_{cons}(m) \to 0$  as  $m \to \infty$ ,  $h^* = \arg \min_{h \in \mathcal{H}} F(P_z, h)$ .

Consistency says that, for any (but fixed)  $m$ , the learned function  $h_m = \mathcal{A}(\mathcal{H}, S)$  is required to be (in expectation) close to the optimal  $h^*$ . The closeness is measured by  $|F(P_z, h_m) - F(P_z, h^*)|$ . By considering this quantity, optimization error will naturally appear. We first show the following observation in Appendix A.2.

Lemma 4. Denote  $h^* = \arg \min_{h \in \mathcal{H}} F(P_z, h)$  and  $\widehat{P}_z$  is the empirical distribution defined from a sample  $S$  of size  $m$ . For any  $h_o \in \mathcal{H}$ , letting  $\epsilon_o = F(\widehat{P}_z, h_o) - \min_{h \in \mathcal{H}} F(\widehat{P}_z, h)$ , we have:

$$
\left| F \left(P _ {z}, h _ {o}\right) - F \left(P _ {z}, h ^ {*}\right) \right| \leq \epsilon_ {o} + 2 \sup  _ {h \in \mathcal {H}} \left| F \left(P _ {z}, h\right) - F \left(\widehat {P} _ {z}, h\right) \right|
$$

This lemma shows why the optimization error  $\epsilon_{o}$  and capacity of family  $\mathcal{H}$  control the goodness of a function. Combining Theorem 1 with Lemma 4 will lead to the following.

Theorem 5 (General family). Given the assumptions in Theorem 1, consider any function  $h_{o} \in \mathcal{H}$ . Let  $h^{*} = \arg \min_{h \in \mathcal{H}} F(P_{z}, h)$ , and  $\epsilon_{o} = F(\widehat{P}_{z}, h_{o}) - \min_{h \in \mathcal{H}} F(\widehat{P}_{z}, h)$  be the optimization error of  $h_{o}$  on a sample  $S$  of size  $m$ . For any  $\alpha \leq n / (2 + n)$ , with probability at least  $1 - 2\exp(-0.5m^{\alpha})$ :  $|F(P_{z}, h_{o}) - F(P_{z}, h^{*})| \leq \epsilon_{o} + 2(LB + 2C)m^{-\alpha / n}$ .

Corollary 1. Given the assumptions in Theorem 5, consider a learning algorithm  $\mathcal{A}$  and family  $\mathcal{H}$ .  $\mathcal{A}$  is consistent if, for any given sample  $S$  of size  $m$ , the learned function  $h_{o} = \mathcal{A}(\mathcal{H}, S)$  has optimization error at most  $\epsilon_{o}(m)$  which satisfies  $\epsilon_{o}(m) \to 0$  as  $m \to \infty$ .

This consistency result is quite general. Various traditional models learned by (stochastic) gradient decent satisfy this corollary when their training problems are convex. The reason is that convex problems can be solved efficiently (Allen-Zhu, 2017; Schmidt et al., 2017). However, one limitation of this consistency result is that it suffers from the curse of dimensionality. Further assumptions or exploitations about structural properties of  $\mathcal{H}$  are needed.

Combining Theorems 3 with Lemma 4 will lead to the following for Dropout DNNs. Similar results can be shown for SN-DNNs.

Theorem 6 (Dropout family). Given the assumptions in Theorem 3, consider any  $h_o \in \mathcal{H}_{dr}$ . Let  $h^* = \arg \min_{h \in \mathcal{H}_{dr}} F(P_z, h)$ , and  $\epsilon_o = F(\widehat{P}_z, h_o) - \min_{h \in \mathcal{H}_{dr}} F(\widehat{P}_z, h)$  be the optimization error of  $h_o$  on a sample  $S$  of size  $m$ . For any constant  $\delta \in (0, 1]$ , with probability at least  $1 - \delta$ :  $|F(P_z, h_o) - F(P_z, h^*)| \leq \epsilon_o + 2\left(C_{dr} L_f B + C \sqrt{\log 4 - 2\log \delta}\right) m^{-0.5}$

Corollary 2 (Consistency of Dropout DNNs). Given the assumptions in Theorem 6, consider a learning algorithm  $\mathcal{A}$  and family  $\mathcal{H}_{dr}$ . If, for any given sample  $\pmb{S}$  of size  $m$ , the learned function  $h_{o} = \mathcal{A}(\mathcal{H}_{dr}, \pmb{S})$  has optimization error at most  $\epsilon_{o}(m)$  which satisfies  $\epsilon_{o}(m) \to 0$  as  $m \to \infty$ , then  $\mathcal{A}$  is consistent with rate  $\epsilon_{o}(m) + 2\left(C_{dr}L_{f}B + C\sqrt{\log 4 - 2\log\delta}\right)m^{-0.5}$ .

Connection to overparameterization: Contrary to classical wisdoms about overfitting, modern machine learning exhibits a strange phenomenon: very rich models such as neural networks are trained to exactly fit (i.e., interpolate and  $\epsilon_{o} = 0$ ) the data, but often obtain high accuracy on test data (Belkin et al., 2019; Zhang et al., 2021). Those models often belong to overparameterization regime where the number of parameters in a model is far larger than  $m$ . Such a strikingly strange behavior could not be explained by traditional learning theories (Zhang et al., 2021). Some works try to understand overparameterization in linear regression (Bartlett et al., 2020) and kernel regression (Liang et al., 2020). Some recent results (Kuzborskij & Szepesvári, 2021; Ji et al., 2021; Hu et al., 2021; Jacot et al., 2018) on consistency hold only for shallow neural networks with no more than 3 layers. However, consistency of deep neural networks remains largely open.

Surprisingly, overparameterized models can lead to nonconvex but tractable training problems. Indeed, (Allen-Zhu et al., 2019; Nguyen & Mondelli, 2020) show that simple optimizers such as (stochastic) gradient descent can find global solutions of the training problems for popular DNN families, meaning  $\epsilon_{o} = 0$ . Combining those results with Corollary 2 will reveal consistency with rate  $O(m^{-0.5})$  for Dropout DNNs and SN-DNNs. To our knowledge, this is the first consistency result for overparameterized DNNs which are truly deep and avoid the curse of dimensionality.

# 4 GENERALIZATION OF GANS

This section presents a comprehensive analysis on generalization of GANs. We then discuss why Lipschitz constraint succeeds in practice.

Notations: Let  $S = \{x_{1},\ldots ,x_{m},z_{1},\ldots ,z_{m}\}$  consist of  $m$  i.i.d. samples from real distribution  $P_{d}$  defined on a compact set  $\mathcal{Z}_x\subset \mathbb{R}^{n_x}$  and  $m$  i.i.d. samples from noise distribution  $P_{z}$  defined on a compact set  $\mathcal{Z}_z\subset \mathbb{R}^n$ ,  $\widehat{P}_x$  and  $\widehat{P}_z$  be the empirical distributions defined from  $S$  respectively. Denote  $\mathcal{D}$  as the discriminator family and  $\mathcal{G}$  as the generator family. Let  $v(D,G,x,z) = \psi_1(D(x)) + \psi_2(1 - D(G(z)))$  be the loss defined from a real example  $x\sim P_d$ , a noise  $z\sim P_z$ , a discriminator  $D\in \mathcal{D}$ , and a generator  $G\in \mathcal{G}$ . Different choices of the measuring functions  $(\psi_{1},\psi_{2})$  will lead to different GANs. For example, saturating GAN (Goodfellow et al., 2014) uses  $\psi_{1}(x) = \psi_{2}(x) = \log (x)$ ; WGAN (Arjovsky et al., 2017) uses  $\psi_{1}(x) = - (x + a)^{2}$ ,  $\psi_{2}(x) = -(x + b)^{2}$  for some constants  $a,b$ ; EBGAN (Zhao et al., 2017) uses  $\psi_{1}(x) = x$ ,  $\psi_{2}(x) = \max (0,r - x)$  for some constant  $r$ . We will often work with:  $V(P_{d},P_{z},D,G) = \mathbb{E}_{x\sim P_{d}}\psi_{1}(D(x)) + \mathbb{E}_{z\sim P_{z}}\psi_{2}(1 - D(G(z)))$ ;  $V(P_{d},\widehat{P}_{z},D,G) = \mathbb{E}_{x\sim P_{d}}\psi_{1}(D(x)) + \mathbb{E}_{z\sim \widehat{P}_{z}}\psi_{2}(1 - D(G(z)))$ ;  $V(\widehat{P}_d,P_z,D,G) = \mathbb{E}_{x\sim \widehat{P}_d}\psi_1(D(x)) + \mathbb{E}_{z\sim \widehat{P}_z}\psi_2(1 - D(G(z)))$ ;  $V(\widehat{P}_d,\widehat{P}_z,D,G) = \mathbb{E}_{x\sim \widehat{P}_d}\psi_1(D(x)) + \mathbb{E}_{z\sim \widehat{P}_z}\psi_2(1 - D(G(z)))$ .

In practice, we only have a finite sample  $S$  and an optimizer will solve  $\min_{G \in \mathcal{G}} \max_{D \in \mathcal{D}} V(\widehat{P}_d, \widehat{P}_z, D, G)$  and return an approximate solution  $(D_o, G_o)$ , which can be different from the training optimum  $(D_o^*, G_o^*)$  and Nash solution  $(D^*, G^*)$ , where

$$
\left(D _ {o} ^ {*}, G _ {o} ^ {*}\right) = \arg \min  _ {G \in \mathcal {G}} \max  _ {D \in \mathcal {D}} V \left(\widehat {P} _ {d}, \widehat {P} _ {z}, D, G\right), \quad \left(D ^ {*}, G ^ {*}\right) = \arg \min  _ {G \in \mathcal {G}} \max  _ {D \in \mathcal {D}} V \left(P _ {d}, P _ {z}, D, G\right) \tag {2}
$$

In learning theory, we often estimate  $(V(P_{d},P_{z},D_{o},G_{o}) - V(\widehat{P}_{d},\widehat{P}_{z},D_{o},G_{o}))$  to see generalization. However a small bound on this quantity may not be enough, since  $V(P_{d},P_{z},D_{o},G_{o})$  can be far from the best  $V(P_{d},P_{z},D^{*},G^{*})$ . Another way (Bousquet et al., 2004) is to see How good is  $(D_o,G_o)$  compared to the Nash solution  $(D^{*},G^{*})$ ? In other words, we basically need to estimate the difference  $|V(P_d,P_z,D_o,G_o) - V(P_d,P_z,D^*,G^*)| = |V(P_d,P_z,D_o,G_o) - \min_{G\in \mathcal{G}}\max_{D\in \mathcal{D}}V(P_d,P_z,D,G)|$  where  $V(P_{d},P_{z},D_{o},G_{o})$  shows the quality of the fake distribution induced by generator  $G_{o}$ .

We first make the following error decomposition:

$$
\begin{array}{l} V (P _ {d}, P _ {z}, D _ {o}, G _ {o}) - V (P _ {d}, P _ {z}, D ^ {*}, G ^ {*}) = [ V (P _ {d}, P _ {z}, D _ {o}, G _ {o}) - V (\widehat {P} _ {d}, \widehat {P} _ {z}, D _ {o}, G _ {o}) ] + \\ \left[ V \big (\widehat {P} _ {d}, \widehat {P} _ {z}, D _ {o}, G _ {o} \big) - V \big (\widehat {P} _ {d}, \widehat {P} _ {z}, D _ {o} ^ {*}, G _ {o} ^ {*} \big) \right] + \left[ V \big (\widehat {P} _ {d}, \widehat {P} _ {z}, D _ {o} ^ {*}, G _ {o} ^ {*} \big) - V \big (P _ {d}, P _ {z}, D ^ {*}, G ^ {*} \big) \right] (3) \\ \end{array}
$$

The first term  $(V(P_{d},P_{z},D_{o},G_{o}) - V(\widehat{P}_{d},\widehat{P}_{z},D_{o},G_{o}))$  in the right-hand side of (3) shows the difference between the population and empirical losses of a specific solution  $(D_o,G_o)$ . The second term  $(V(\widehat{P}_d,\widehat{P}_z,D_o,G_o) - V(\widehat{P}_d,\widehat{P}_z,D_o^*,G_o^*))$  is in fact the Optimization error incurred by the optimizer. This error depends strongly on the capacity of the chosen optimizer. The third term  $(V(\widehat{P}_d,\widehat{P}_z,D_o^*,G_o^*) - V(P_d,P_z,D^*,G^*))$  is optimizer-independent and strongly depends on the capacity of both families  $(\mathcal{D},\mathcal{G})$ , since both  $V(\widehat{P}_d,\widehat{P}_z,D_o^*,G_o^*)$  and  $V(P_{d},P_{z},D^{*},G^{*})$  are optimizer-independent. We call this term Joint error of  $(\mathcal{D},\mathcal{G})$ . In the next subsections, we will provide upper bounds on both the error of  $(D_o,G_o)$  and joint error of  $(\mathcal{D},\mathcal{G})$ , and then generalization bounds that take the optimization error into account.

In the later discussions, we will often use the following assumptions and notation  $L = L_{\psi}L_{d}L_{g}$  which upper bounds the Lipschitz constant of the loss  $v(D,G,x,z)$ .

Assumption 1.  $\psi_{1}$  and  $\psi_{2}$  are  $L_{\psi}$ -Lipschitz continuous w.r.t. their inputs on a compact domain and upper-bounded by constant  $C \geq 0$ .

Assumption 2. Each generator  $G \in \mathcal{G}$  is  $L_{g}$ -Lipschitz continuous w.r.t its input  $z$  over a compact set  $\mathcal{Z}_{z} \subset \mathbb{R}^{n}$  with diameter at most  $B_{z}$ .

Assumption 3. Each discriminator  $D \in \mathcal{D}$  is  $L_{d}$ -Lipschitz continuous w.r.t its input  $x$  over a compact set  $\mathcal{Z}_x \subset \mathbb{R}^{n_x}$  with diameter at most  $B_{x}$ .

These assumptions are reasonable and satisfied by various GANs. For example, WGAN, LSGAN, EBGAN naturally satisfy Assumption 1, while saturating GAN will satisfy it if we constraint the

output of  $D$  to be in  $[\alpha, \beta] \subset (0,1)$  as suggested by Salimans et al. (2016). Spectral normalization and gradient penalty are popular techniques to regularize  $D$  and are crucial for large-scale generators (Zhang et al., 2019a; Karras et al., 2020b). Therefore Assumptions 3 and 2 are natural.

# 4.1 ERROR BOUNDS

We first concern on upper bounds of the generalization error of any generator when it is learned from a finite training noise data. The following results readily come from Theorem 1.

Corollary 3. Given the assumptions (1, 2, 3), for any  $\delta \in (0,1]$ ,  $\lambda \in (0,B_z]$ , with probability at least  $1 - \delta$ , we have

$$
\sup_{D\in \mathcal{D},G\in \mathcal{G}}|V(P_{d},P_{z},D,G) - V(P_{d},\widehat{P}_{z},D,G)|\leq L\lambda +\frac{C}{\sqrt{m}}\sqrt{\lceil B_{z}^{n}\lambda^{-n}\rceil\log 4 - 2\log\delta}
$$

$$
\sup _ {D \in \mathcal {D}, G \in \mathcal {G}} | V (\widehat {P} _ {d}, P _ {z}, D, G) - V (\widehat {P} _ {d}, \widehat {P} _ {z}, D, G) | \leq L \lambda + \frac {C}{\sqrt {m}} \sqrt {\lceil B _ {z} ^ {n} \lambda^ {- n} \rceil \log 4 - 2 \log \delta}
$$

This corollary tells the generalization of any generator  $G \in \mathcal{G}$  in both cases, where we have either a finite or infinite number of real samples. To see generalization of both players  $(D_o, G_o)$ , observe that  $|V(P_d, P_z, D_o, G_o) - V(\widehat{P}_d, \widehat{P}_z, D_o, G_o)| \leq \sup_{D \in \mathcal{D}, G \in \mathcal{G}} |V(\widehat{P}_d, \widehat{P}_z, D, G) - V(P_d, P_z, D, G)|$ . The following theorem provides an upper bound whose proof appears in Appendix B.

Theorem 7. Denote  $\epsilon(\mathcal{D},\mathcal{G}) = \sup_{D\in \mathcal{D},G\in \mathcal{G}}|V(\widehat{P}_d,\widehat{P}_z,D,G) - V(P_d,P_z,D,G)|$ . Given the assumptions (1, 2, 3), for any constants  $\delta, \delta_x \in (0,1]$ ,

(General family) for any  $\lambda \in (0,B_z],\lambda_x\in (0,B_x]$ , with probability at least  $1 - \delta -\delta_{x}$ :

$$
\epsilon (\mathcal {D}, \mathcal {G}) \leq L \lambda + \frac {C}{\sqrt {m}} \sqrt {\lceil B _ {z} ^ {n} \lambda^ {- n} \rceil \log 4 - 2 \log \delta} + L _ {\psi} L _ {d} \lambda_ {x} + \frac {C}{\sqrt {m}} \sqrt {\lceil B _ {x} ^ {n _ {x}} \lambda_ {x} ^ {- n _ {x}} \rceil \log 4 - 2 \log \delta_ {x}}
$$

$(D$  with spectral norm) given the assumptions in Theorem 3, with probability at least  $1 - 2\delta$ :

$$
\epsilon (\mathcal {H} _ {s n}, \mathcal {G}) \leq [ C _ {s n} L _ {\psi} L _ {g} B _ {z} + 2 C \sqrt {\log 4 - 2 \log \delta} + C _ {s n} L _ {\psi} B _ {x} ] m ^ {- 0. 5}
$$

$(D$  with Dropout) given the assumptions in Theorem 3, with probability at least  $1 - 2\delta$ :

$$
\epsilon (\mathcal {H} _ {d r}, \mathcal {G}) \leq [ C _ {d r} \bar {L} _ {\psi} L _ {g} B _ {z} + 2 C \sqrt {\log 4 - 2 \log \delta} + C _ {d r} L _ {\psi} B _ {x} ] m ^ {- 0. 5}
$$

For many models, such as WGAN, the measuring functions and  $D$  are Lipschitz continuous w.r.t their inputs. Note that the generator in WGAN, LSGAN, and EBGAN will be Lipschitz continuous w.r.t  $z$ , if we use some regularization methods such as gradient penalty or spectral normalization for both players. Theorem 7 also suggests that penalizing the zero-order  $(C)$  and first-order  $(L)$  informations of the loss can improve the generalization. This provides a significant evidence for the important role of gradient penalty or spectral normalization for the success of some large-scale generators (Zhang et al., 2019a; Brock et al., 2019; Karras et al., 2020b).

It is worth observing that a small Lipschitz constant of the loss not only requires that both discriminator and generator are Lipschitz continuous w.r.t their inputs, but also requires Lipschitz continuity of the loss w.r.t both players. Most existing efforts focus on ensuring the Lipschitz continuity of the players in GANs, and leave the loss open. Constraining on either discriminator or generator only may be insufficient to ensure Lipschitz continuity of the loss.

One advantage of the generalization bounds in Theorem 7 is that the upper bounds on  $|V(\widehat{P}_d, \widehat{P}_z, D, G) - V(P_d, P_z, D, G)|$  hold true for any particular  $(D, G)$  in their families. Meanwhile, the existing generalization bounds (Arora et al., 2017; Zhang et al., 2018; Jiang et al., 2019; Wu et al., 2019; Husain et al., 2019) hold true under the best discriminator. Hence the bounds in Theorem 7 are more practical than existing ones, since  $D$  is not trained to optimality before training  $G$  in practical implementations of GANs.

Next we consider the joint error  $V(\widehat{P}_d, \widehat{P}_z, D_o^*, G_o^*) - V(P_d, P_z, D^*, G^*)$  of both families  $(\mathcal{D}, \mathcal{G})$ . Such a quantity also shows the goodness of the training optimum  $(D_o^*, G_o^*)$  compared with the Nash solution  $(D^*, G^*)$ . It is worth observing that  $|V(\widehat{P}_d, \widehat{P}_z, D_o^*, G_o^*) - V(P_d, P_z, D^*, G^*)| = |\min_{G \in \mathcal{G}} \max_{D \in \mathcal{D}} V(\widehat{P}_d, \widehat{P}_z, D, G) - \min_{G \in \mathcal{G}} \max_{D \in \mathcal{D}} V(P_d, P_z, D, G)|$  measures the quality of the best players given a finite number of samples only, and such error does not depend on any optimizer. Hence it represents the Joint capacity of both generator and discriminator families. The following theorem provides an upper bound whose proof appears in Appendix B.

Theorem 8 (Joint error). Given the assumptions (1, 2, 3), for any constants  $\delta, \delta_x \in (0,1]$ ,  $\lambda \in (0,B_z]$ ,  $\lambda_x \in (0,B_x]$ , with probability at least  $1 - \delta - \delta_x$ :

$$
\begin{array}{l} | V (\widehat {P} _ {d}, \widehat {P} _ {z}, D _ {o} ^ {*}, G _ {o} ^ {*}) - V (P _ {d}, P _ {z}, D ^ {*}, G ^ {*}) | \leq L \lambda + \frac {C}{\sqrt {m}} \sqrt {\lceil B _ {z} ^ {n} \lambda^ {- n} \rceil \log 4 - 2 \log \delta} + L _ {\psi} L _ {d} \lambda_ {x} + \\ \frac {C}{\sqrt {m}} \sqrt {\lceil B _ {x} ^ {n _ {x}} \lambda_ {x} ^ {- n _ {x}} \rceil \log 4 - 2 \log \delta_ {x}}. \\ \end{array}
$$

This bound on joint capacity of  $(\mathcal{D},\mathcal{G})$  is loose, since few informations about those families are used. We can further tighten this bound when using spectral normalization or Dropout for discriminator, similar with Theorem 7.

# 4.2 FROM OPTIMIZATION ERROR TO GENERALIZATION

Finally we make a bidge between optimization error and generalization. The decomposition (3) contains three components, for which the first component is bounded in Theorem 7 while the third component is bounded in Theorem 8. Combining those observations will lead to the following result.

Theorem 9 (Generalization bounds for GANs). Assume the assumptions (1, 2, 3) and the optimization error  $|V(\widehat{P}_d,\widehat{P}_z,D_o,G_o) - \min_{G\in \mathcal{G}}\max_{D\in \mathcal{D}}V(\widehat{P}_d,\widehat{P}_z,D,G)|\leq \epsilon_o$ . Denote  $\epsilon_{cons}(\mathcal{D},\mathcal{G}) = |V(P_d,P_z,D_o,G_o) - V(P_d,P_z,D^*,G^*)|$ . For any constants  $\delta ,\delta_{x}\in (0,1]$ ,

1. (General family) for any  $\lambda \in (0,B_z],\lambda_x\in (0,B_x]$ , with probability at least  $1 - \delta -\delta_{x}$ :

$$
\begin{array}{l} \epsilon_ {c o n s} (\mathcal {D}, \mathcal {G}) \leq \epsilon_ {o} + 2 L \lambda + \frac {2 C}{\sqrt {m}} \sqrt {\lceil B _ {z} ^ {n} \lambda^ {- n} \rceil \log 4 - 2 \log \delta} + 2 L _ {\psi} L _ {d} \lambda_ {x} + \\ \frac {2 C}{\sqrt {m}} \sqrt {\lceil B _ {x} ^ {n _ {x}} \lambda_ {x} ^ {- n _ {x}} \rceil \log 4 - 2 \log \delta_ {x}}. \\ \end{array}
$$

2. (Spectral norm) given the assumptions in Theorem 3,  $\mathcal{D} \equiv \mathcal{H}_{sn}$ , with probability at least  $1 - 2\delta$ :  $\epsilon_{cons}(\mathcal{H}_{sn}, \mathcal{G}) \leq \epsilon_o + 2[C_{sn}L_\psi L_gB_z + 2C\sqrt{\log 4 - 2\log\delta} + C_{sn}L_\psi B_x]m^{-0.5}$  
3. (Dropout) given the assumptions in Theorem 3,  $\mathcal{D} \equiv \mathcal{H}_{sn}$ , with probability at least  $1 - 2\delta$ :

$$
\epsilon_ {c o n s} (\mathcal {H} _ {d r}, \mathcal {G}) \leq \epsilon_ {o} + 2 [ C _ {d r} L _ {\psi} L _ {g} B _ {z} + 2 C \sqrt {\log 4 - 2 \log \bar {\delta}} + C _ {d r} L _ {\psi} B _ {x} ] m ^ {- 0. 5}
$$

Theorems 7 and 9 provide us a comprehensive view about generalization of GANs. Note that their assumptions are naturally met in practice as pointed out before. For the first time in the GAN literature, our work reveals that GANs can avoid the curse of dimensionality when choosing appropriate  $(\mathcal{D},\mathcal{G})$ . Furthermore, a logarithmic (in  $m$ ) number of layers are sufficient for each player. Although this work shows this property for DNNs with spectral norm or Dropout. We believe that this property can hold for many other DNN families.

One important implication from Theorem 9 is that GANs can be consistent under suitable conditions. An example condition is overparameterization, for which the optimization error can be zero. Our experiments in Appendix F provide a good evidence for this conjecture as the well-trained discriminators often reach Nash equilibria. A recent investigation about optimization of overparameterized GANs appears in (Balaji et al., 2021). We leave this door open for the readers.

# 4.3 WHY A LIPSCHITZ CONSTRAINT IS CRUCIAL

Various works (Guo et al., 2019; Jenni & Favaro, 2019; Qi, 2020; Arjovsky et al., 2017; Gulrajani et al., 2017; Roth et al., 2017; Miyato et al., 2018; Zhou et al., 2019; Thanh-Tung et al., 2019; Jiang et al., 2019; Tanielian et al., 2020; Xu et al., 2020) try to ensure Lipschitz continuity of the discriminator or generator or both. The most popular techniques are gradient penalty (Gulrajani et al., 2017) and spectral normalization (Miyato et al., 2018). Those two techniques are really useful for different losses (Fedus et al., 2018) and high-capacity architectures (Kurach et al., 2019). From a large-scale evaluation, Kurach et al. (2019) found that gradient penalty can help the performance of GANs but does not stabilize the training, whereas using spectral normalization on  $G$  only is insufficient to ensure stability (Brock et al., 2019). Some recent large-scale generators (Brock et al., 2019; Zhang et al., 2019a; Karras et al., 2020b) use gradient penalty or spectral normalization to ensure their successes. Data augmentation (Zhao et al., 2020a;b; Tran et al., 2021; Karras et al., 2020a) also contributes to the excellent performance of GANs in practice, due to implicitly penalizing the Lipschitz constant of the loss (see Appendix D for explanation). Those empirical observations without a theory poses a long mystery of why can imposing a Lipschitz constraint help GANs to perform well? This work provides an answer:

- Theorems 7 and 9 show that a Lipschitz constraint on one player  $(D$  or  $G)$  only can help, but may be not enough. A penalty on the zero-order  $(C)$  and first-order  $(L_{\psi}, L_d, L_g)$  informations of the loss can lead to better generalization.  
- Dropout and spectral normalization are really efficient to control the complexity of the players and provide tight generalization bounds.  
- Spectral normalization (SN) (Miyato et al., 2018) is a popular technique to regularize GANs. Using SN, the spectral norms of the weight matrices are often small in practice as observed by Miyato et al. (2018), and hence the Lipschitz constant of  $D$  (or  $G$ ) can be exponentially small when using SN. In those cases, the assumptions of Theorem 9 are satisfied. Therefore the generalization bound in Theorem 9 is tight and supports well the success of spectrally-normalized GANs (Miyato et al., 2018; Zhang et al., 2019a).  
- WGAN (Arjovsky et al., 2017) naturally requires  $D$  to be 1-Lipschitz continuous. Weight clipping is used so that every element of network weights belongs to  $[-c, c]$  for some constant  $c$ . For some choices, e.g.  $c = 0.01$  in (Arjovsky et al., 2017), the spectral norm of the weight matrix at each layer can be smaller than  $1$ . In those cases the Lipschitz constant of  $D$  can be exponentially small, leading to tight bounds in Theorem 9 and better generalization.  
- SN, gradient penalty, and data augmentation are crucial parts of large-scale GANs (Brock et al., 2019; Zhang et al., 2019a; Karras et al., 2020b). As a result, Theorems 7 and 9 provide a strong support for their success in practice.  
- Our experiments with SN in Appendix F indeed show that SN can reduce the Lipschitz constants of the players and the loss. However, when SN is overused, the trained players can get underfitting and may hurt generalization. A reason is that an underfitted model can have a bad population loss and high optimization error.

# 4.4 TIGHTNESS OF THE BOUNDS FOR GANS AND AUTOENCODERS

Note that our bounds in Theorems 7 and 9 in general are not tight in terms of sample complexity and dimensionality. Taking  $\lambda = B_z m^{-1/(n+2)}$ ,  $\delta = 2 \exp(-0.5m^{n/(n+2)})$ ,  $\lambda_x = B_x m^{-1/(n_x+2)}$ ,  $\delta_x = 2 \exp(-0.5m^{n_x/(n_x+2)})$ , Theorem 7 provides  $\sup_{D \in \mathcal{D}, G \in \mathcal{G}} |V(\widehat{P}_d, \widehat{P}_z, D, G) - V(P_d, P_z, D, G)| \leq O(m^{-1/(n+2)} + m^{-1/(n_x+2)})$ . This bound  $O(m^{-1/(n+2)} + m^{-1/(n_x+2)})$  surpasses the previous best bound  $O(m^{-1/(1.5n)} + m^{-1/(1.5n_x)})$  in the GAN literature (Husain et al., 2019).

Sample-efficient bounds for Autoencoders: Husain et al. (2019) did a great job at connecting GANs and Autoencoder models. They showed that the generator objective in  $f$ -GAN (Nowozin et al., 2016) is upper bounded by the objective of Wasserstein Autoencoders (WAE) (Tolstikhin et al., 2018). Under some suitable conditions, the two objectives equal. They further showed the bound:  $\left| \max_{D \in \mathcal{D}} V(P_d, P_z, D, G) - \max_{D \in \mathcal{D}} V(\widehat{P}_d, \widehat{P}_z, D, G) \right| \leq O(m^{-1 / s_d} + m^{-1 / s_g})$ , where  $s_d > d^*(P_d)$  (the 1-upper Wasserstein dimension of  $P_d$ ) and  $s_g > d^*(P_g)$ . We show in Appendix C.1 that  $s_d > 1.5n_x$ ,  $s_g > 1.5n$  even for a simple distribution, where  $n_x$  is the dimensionality of real data, and  $n$  is the dimensionality of latent codes. Therefore their bound becomes  $O(m^{-\frac{1}{1.5n_x} + m^{-\frac{1}{1.5n}}})$ , which is significantly worse than our bound  $O(m^{-1 / (n + 2)} + m^{-1 / (n_x + 2)})$ . As a result, our work provides tighter generalization bounds for both GANs and Autoencoder models. More importantly, our results for DNNs with Dropout or spectral norm translate directly to Autoencoders, leading to significant tighter bounds.

# 5 CONCLUSION

We have presented a simple way to analyze generalization of various complex models that are hard for traditional learning theories. Some successful applications were done and made a significant step toward understanding DNNs, GANs, and Autoencoders. One limitation of our bounds is that the optimization aspect is left open.

# REFERENCES

Zeyuan Allen-Zhu. Katyusha: The first direct acceleration of stochastic gradient methods. The Journal of Machine Learning Research, 18(1):8194-8244, 2017.  
Zeyuan Allen-Zhu, Yanzhi Li, and Zhao Song. A convergence theory for deep learning via overparameterization. In International Conference on Machine Learning, pp. 242-252. PMLR, 2019.  
Martin Arjovsky and Leon Bottou. Towards principled methods for training generative adversarial networks. In International Conference on Learning Representations, 2017.  
Martin Arjovsky, Soumith Chintala, and Leon Bottou. Wasserstein generative adversarial networks. In Proceedings of the 34th International Conference on Machine Learning, 2017.  
Raman Arora, Peter Bartlett, Poorya Mianjy, and Nathan Srebro. Dropout: Explicit forms and capacity control. In International Conference on Machine Learning, pp. 351-361. PMLR, 2021.  
Sanjeev Arora, Rong Ge, Yingyu Liang, Tengyu Ma, and Yi Zhang. Generalization and equilibrium in generative adversarial nets (gans). In International Conference on Machine Learning, pp. 224-232, 2017.  
Haim Avron and Sivan Toledo. Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix. Journal of the ACM (JACM), 58(2):1-34, 2011.  
Francis Bach. Breaking the curse of dimensionality with convex neural networks. The Journal of Machine Learning Research, 18(1):629-681, 2017.  
Yogesh Balaji, Mohammadmahdi Sajedi, Neha Mukund Kalibhat, Mucong Ding, Dominik Stöger, Mahdi Soltanolkotabi, and Soheil Feizi. Understanding over-parameterization in generative adversarial networks. In International Conference on Learning Representations, 2021.  
Peter L Bartlett, Dylan J Foster, and Matus J Telgarsky. Spectrally-normalized margin bounds for neural networks. Advances in Neural Information Processing Systems, 30:6240-6249, 2017.  
Peter L Bartlett, Philip M Long, Gábor Lugosi, and Alexander Tsigler. Benign overfitting in linear regression. Proceedings of the National Academy of Sciences, 117(48):30063-30070, 2020.  
Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machine-learning practice and the classical bias-variance trade-off. Proceedings of the National Academy of Sciences, 116(32):15849-15854, 2019.  
Olivier Bousquet, Stephane Boucheron, and Gábor Lugosi. Introduction to statistical learning theory. In Machine Learning 2003, LNAI, volume 3176, pp. 169-207. Springer, 2004.  
Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In International Conference on Learning Representations, 2019.  
Casey Chu, Kentaro Minami, and Kenji Fukumizu. Smoothness and stability in gans. In International Conference on Learning Representations, 2020.  
William Fedus, Mihaela Rosca, Balaji Lakshminarayanan, Andrew M Dai, Shakir Mohamed, and Ian Goodfellow. Many paths to equilibrium: Gans do not need to decrease a divergence at every step. In International Conference on Learning Representations, 2018.  
Noah Golowich, Alexander Rakhlin, and Ohad Shamir. Size-independent sample complexity of neural networks. Information and Inference: A Journal of the IMA, 9(2):473-504, 2020.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pp. 2672-2680, 2014.  
Henry Gouk, Eibe Frank, Bernhard Pfahringer, and Michael J Cree. Regularisation of neural networks by enforcing lipschitz continuity. Machine Learning, 110(2):393-416, 2021.

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of wasserstein gans. In Advances in Neural Information Processing Systems, pp. 5767-5777, 2017.  
Tianyu Guo, Chang Xu, Boxin Shi, Chao Xu, and Dacheng Tao. Smooth deep image generator from noises. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 3731-3738, 2019.  
Yongjun Hong, Uiwon Hwang, Jaeyoon Yoo, and Sungroh Yoon. How generative adversarial networks and their variants work: An overview. ACM Computing Surveys (CSUR), 52(1):1-43, 2019.  
Tianyang Hu, Wenjia Wang, Cong Lin, and Guang Cheng. Regularization matters: A nonparametric perspective on overparametrized neural network. In International Conference on Artificial Intelligence and Statistics, pp. 829-837. PMLR, 2021.  
Hisham Husain, Richard Nock, and Robert C Williamson. A primal-dual link between gans and autoencoders. In Advances in Neural Information Processing Systems, volume 32, pp. 415-424, 2019.  
Michael F Hutchinson. A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines. Communications in Statistics - Simulation and Computation, 18(3):1059-1076, 1989.  
Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, pp. 8580-8589, 2018.  
Simon Jenni and Paolo Favaro. On stabilizing generative adversarial training with noise. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 12145-12153, 2019.  
Ziwei Ji, Justin D Li, and Matus Telgarsky. Early-stopped neural networks are consistent. arXiv preprint arXiv:2106.05932, 2021.  
Haoming Jiang, Zhehui Chen, Minshuo Chen, Feng Liu, Dingding Wang, and Tuo Zhao. On computation and generalization of generative adversarial networks under spectrum control. In International Conference on Learning Representations, 2019.  
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. In Advances in Neural Information Processing Systems, 2020a.  
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8110-8119, 2020b.  
Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021. doi: 10.1109/TPAMI.2020.2970919.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, volume 25, pp. 1097-1105, 2012.  
Karol Kurach, Mario Lučić, Xiaohua Zhai, Marcin Michalski, and Sylvain Gelly. A large-scale study on regularization and normalization in gans. In International Conference on Machine Learning, pp. 3581-3590, 2019.  
Ilja Kuzborskij and Csaba Szepesvári. Nonparametric regression with shallow overparameterized neural networks trained by gd with early stopping. In Conference on Learning Theory, pp. 2853-2890. PMLR, 2021.

Tengyuan Liang, Alexander Rakhlin, and Xiyu Zhai. On the multiple descent of minimum-norm interpolants and restricted lower isometry of kernels. In Conference on Learning Theory, pp. 2683-2711. PMLR, 2020.  
Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, and Olivier Bousquet. Are gans created equal? a large-scale study. In Advances in Neural Information Processing Systems, pp. 700-709, 2018.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.  
Xudong Mao, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial networks. In Proceedings of the IEEE International Conference on Computer Vision, pp. 2794-2802, 2017.  
Xudong Mao, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen Paul Smolley. On the effectiveness of least squares generative adversarial networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(12):2947-2960, 2019.  
Lars Mescheder, Sebastian Nowozin, and Andreas Geiger. The numerics of gans. In Advances in Neural Information Processing Systems, pp. 1825-1835, 2017.  
Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge? In International Conference on Machine Learning, pp. 3481-3490, 2018.  
Poorya Mianjy and Raman Arora. On convergence and generalization of dropout training. In Advances in Neural Information Processing Systems, volume 33, 2020.  
Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations, 2018.  
Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of Machine Learning. MIT Press, 2018.  
Wenlong Mou, Yuchen Zhou, Jun Gao, and Liwei Wang. Dropout training, data-dependent regularization, and generalization bounds. In International Conference on Machine Learning, pp. 3645-3653. PMLR, 2018.  
Vaishnavh Nagarajan and J Zico Kolter. Gradient descent gan optimization is locally stable. In Advances in Neural Information Processing Systems, pp. 5585-5595, 2017.  
Behnam Neyshabur, Srinadh Bhojanapalli, and Nathan Srebro. A pac-bayesian approach to spectrally-normalized margin bounds for neural networks. In International Conference on Learning Representations, 2018.  
Quynh Nguyen and Marco Mondelli. Global convergence of deep networks with one wide layer followed by pyramidal topology. In Advances in Neural Information Processing Systems, volume 33, 2020.  
Weili Nie and Ankit Patel. Towards a better understanding and regularization of gan training dynamics. In Conference on Uncertainty in Artificial Intelligence (UAI), 2019.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-gan: Training generative neural samplers using variational divergence minimization. In Advances in Neural Information Processing Systems, pp. 271-279, 2016.  
Patricia Pauli, Anne Koch, Julian Berberich, Paul Kohler, and Frank Allgower. Training robust neural networks using lipschitz bounds. IEEE Control Systems Letters, 2021.  
Guo-Jun Qi. Loss-sensitive generative adversarial networks on lipschitz densities. International Journal of Computer Vision, 128(5):1118-1140, 2020.

Kevin Roth, Aurelien Lucchi, Sebastian Nowozin, and Thomas Hofmann. Stabilizing training of generative adversarial networks through regularization. In Advances in Neural Information Processing Systems, pp. 2018-2028, 2017.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, pp. 2234-2242, 2016.  
Maziar Sanjabi, Jimmy Ba, Meisam Razaviyayn, and Jason D Lee. On the convergence and robustness of training gans with regularized optimal transport. In Advances in Neural Information Processing Systems, pp. 7091-7101, 2018.  
Mark Schmidt, Nicolas Le Roux, and Francis Bach. Minimizing finite sums with the stochastic average gradient. Mathematical Programming, 162(1-2):83-112, 2017.  
Shai Shalev-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan. Learnability, stability and uniform convergence. The Journal of Machine Learning Research, 11:2635-2670, 2010.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929-1958, 2014.  
Ugo Tanielian, Thibaut Issenhuth, Elvis Dohmatob, and Jeremie Mary. Learning disconnected manifolds: a no gan's land. In International Conference on Machine Learning, 2020.  
Hoang Thanh-Tung, Truyen Tran, and Svetha Venkatesh. Improving generalization and stability of generative adversarial networks. In International Conference on Learning Representations, 2019.  
Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schoelkopf. Wasserstein autoencoders. In International Conference on Learning Representations, 2018.  
Ngoc-Trung Tran, Viet-Hung Tran, Ngoc-Bao Nguyen, Trung-Kien Nguyen, and Ngai-Man Cheung. On data augmentation for gan training. IEEE Transactions on Image Processing, 30:1882-1897, 2021.  
Bingzhe Wu, Shiwan Zhao, Chaochao Chen, Haoyang Xu, Li Wang, Xiaolu Zhang, Guangyu Sun, and Jun Zhou. Generalization in generative adversarial networks: A novel perspective from privacy protection. In Advances in Neural Information Processing Systems, pp. 307-317, 2019.  
Huan Xu and Shie Mannor. Robustness and generalization. Machine learning, 86(3):391-423, 2012.  
Huan Xu, Constantine Caramanis, and Shie Mannor. Robust regression and lasso. IEEE Transactions on Information Theory, 56(7):3561-3574, 2010.  
Kun Xu, Chongxuan Li, Huanshu Wei, Jun Zhu, and Bo Zhang. Understanding and stabilizing gans' training dynamics with control theory. In Proceedings of the 37th International Conference on Machine Learning, 2020.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM, 64(3):107-115, 2021.  
Han Zhang, Ian Goodfellow, Dimitris Metaxas, and Augustus Odena. Self-attention generative adversarial networks. In International Conference on Machine Learning, pp. 7354–7363, 2019a.  
Han Zhang, Zizhao Zhang, Augustus Odena, and Honglak Lee. Consistency regularization for generative adversarial networks. In International Conference on Learning Representations, 2020a.  
Huan Zhang, Hongge Chen, Chaowei Xiao, Sven Gowal, Robert Stanforth, Bo Li, Duane Boning, and Cho-Jui Hsieh. Towards stable and efficient training of verifiably robust neural networks. In International Conference on Learning Representations, 2019b.

Jingfeng Zhang, Xilie Xu, Bo Han, Gang Niu, Lizhen Cui, Masashi Sugiyama, and Mohan Kankanhalli. Attacks which do not kill training make adversarial learning stronger. In International Conference on Machine Learning, pp. 11278-11287. PMLR, 2020b.  
Pengchuan Zhang, Qiang Liu, Dengyong Zhou, Tao Xu, and Xiaodong He. On the discrimination-generalization tradeoff in gans. In International Conference on Learning Representations, 2018.  
Junbo Zhao, Michael Mathieu, and Yann LeCun. Energy-based generative adversarial networks. In International Conference on Learning Representations, 2017.  
Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han. Differentiable augmentation for data-efficient gan training. In Advances in Neural Information Processing Systems, 2020a.  
Zhengli Zhao, Zizhao Zhang, Ting Chen, Sameer Singh, and Han Zhang. Image augmentations for gan training. arXiv preprint arXiv:2006.02595, 2020b.  
Zhiming Zhou, Jiadong Liang, Yuxuan Song, Lantao Yu, Hongwei Wang, Weinan Zhang, Yong Yu, and Zhihua Zhang. Lipschitz generative adversarial nets. In International Conference on Machine Learning, pp. 7584-7593, 2019.
