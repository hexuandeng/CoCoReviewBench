# ON INFORMATION MAXIMISATION IN MULTI-VIEW SELF-SUPERVISED LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

The strong performance of multi-view self-supervised learning (SSL) prompted the development of many different approaches (e.g. SimCLR, BYOL, and DINO). A unified understanding of how each of these methods achieves its performance has been limited by apparent differences across objectives and algorithmic details. Through the lens of information theory, we show that many of these approaches are maximising an approximate lower bound on the mutual information between the representations of multiple views of the same datum. Further, we show that this bound decomposes into a "reconstruction" term, treated identically by all SSL methods, and an "entropy" term, where existing SSL methods differ in their treatment. We prove that an exact optimisation of both terms of this lower bound encompasses and unifies current theoretical properties such as recovering the true latent variables of the underlying generative process (Zimmermann et al., 2021) or isolating content from style in such true latent variables (Von Kugelgen et al., 2021). This theoretical analysis motivates a naive but principled objective (EntRec), that exactly optimises both the reconstruction and entropy terms, thus benefiting from said theoretical properties unlike other SSL frameworks. Finally, we show EntRec achieves a downstream performance on-par with existing SSL methods on ImageNet (69.7% after 400 epochs) and on an array of transfer tasks when pre-trained on ImageNet. Furthermore, EntRec is more robust to modifying the batch size, a sensitive hyperparameter in other SSL methods.

# 1 INTRODUCTION

Representation learning tackles the problem of learning compressed representations of data which capture their semantic information. A necessary, but not sufficient, property of a good representation is thus that it is highly informative of said data. For this reason, many representation learning methods aim to maximise the mutual information between the input data and the representations, while including some biases in the model that steer that information to be semantic, e.g. (Agakov, 2004; Alemi et al., 2017; Hjelm et al., 2018; Oord et al., 2018; Velickovic et al., 2019). Moreover, mutual information has been the central object to understand the performance of many of these algorithms (Saxe et al., 2019; Rodríguez Gálvez et al., 2020; Goldfeld & Polyanskiy, 2020).

A subfield of representation learning is self-supervised learning (SSL), which consists of algorithms that learn representations by means of solving an artificial task with self-generated labels. A particularly successful approach to SSL is multi-view SSL, where different views of the input data are generated and the self-generated task is to make sure that representations of one view are predictive of the representations of the other views c.f. (Jing & Tian, 2020; Liu et al., 2022).

Some multi-view SSL algorithms like (Bachman et al., 2019; Federici et al., 2020; Tian et al., 2020a) focus on maximising the mutual information between the representations and the input data by maximising the mutual information between the representations of different views. Similarly, Shwartz-Ziv et al. showed that (Bardes et al., 2022, VICReg) also maximises this information, even though it was not designed for this purpose. Moreover, Tian et al. (2020b); Tsai et al. (2020) provide perspectives on why maximising this mutual information is attractive and discuss some of its properties. However, Tschannen et al. (2019); McAllester & Stratos (2020) warn about the caveats of this maximisation (e.g. that it is not sufficient for good representations). In this paper, we complement these efforts from multiple fronts and:

- Show that maximising the lower bound (1) on the mutual information between representations of different views has desirable properties in good representations (Section 2). More precisely, we show that this maximisation unifies current theories on learning the true explanatory factors of the input (Zimmermann et al., 2021) and separating semantic from irrelevant information (Von Kugelgen et al., 2021).  
- Show how many existing multi-view SSL algorithms, including (Chen et al., 2020a, SimCLR), (Tian et al., 2020a, CMC), (He et al., 2020, MoCo), (Chen & He, 2021, SimSiam), (Grill et al., 2020, BYOL), (Caron et al., 2018; 2020, DeepCluster and SwAV), and (Caron et al., 2021, DINO) also maximise this mutual information (Section 3), although not exactly maximising the lower bound (1). This provides us with a unifying framework to understand the different algorithms.  
- Demonstrate how a naive method that exactly maximises the aforementioned bound (1) on this mutual information (EntRec) has comparable performance to current state-of-the-art methods and is more robust to changes in training hyperparameters such as the batch size (Section 4 and Section 5).

Overall, this paper is a recognition of the importance of maximising the mutual information between the representations of different views of the input data, as doing so by maximising (1) has desirable properties (Section 2), and many methods that maximise it (Section 3), including naive ones (Section 4), have good empirical performance (Section 5). However, since maximising mutual information is not sufficient for good representations (Tschannen et al., 2019), this paper is also a call to include more biases in the model and the optimisation enforcing the representations to learn semantic information. Appendix A completes the positioning of the paper with respect to related work.

Notation Upper-case letters  $X$  represent random objects, lower-case letters  $x$  their realisations, calligraphic letters  $\mathcal{X}$  their outcome space, and  $\mathbb{P}_X$  their distribution. Random objects  $X$  are assumed to have a density  $\mathsf{p}_X$  with respect to some measure  $\mu$ ,<sup>1</sup> and the expectation of a function  $f$  of  $X$  is written as  $\mathbb{E}[f(X)] \coloneqq \mathbb{E}_{x \sim \mathsf{p}_X}[f(x)]$ . When two random objects  $X, Y$  are considered, the conditional density of  $X$  given  $Y$  is written as  $\mathsf{p}_{X|Y}$ , and for each realisation  $y$  of  $Y$  it describes the density  $\mathsf{p}_{X|Y} = y$ . Sometimes, the notation is abused to write a "variational" density  $\mathsf{q}_{X|Y}$  of  $X$  given  $Y$ . Formally, this amounts to considering a different random object  $\hat{X}$  such that  $\mathsf{p}_{\hat{X}|Y} = \mathsf{q}_{X|Y}$ . The mutual information between two random objects  $X$  and  $Y$  is written as  $\mathsf{l}(X; Y)$ , and their conditional mutual information given the random object  $Z$  as  $\mathsf{l}(X; Y|Z)$ . The Shannon entropy and differential entropy of a random object  $X$  are both written as  $\mathsf{H}(X)$ , and are clear from the context. The Jensen-Shannon divergence between two distributions  $\mathbb{P}$  and  $\mathbb{Q}$  is written as  $\mathsf{D_{JS}}(\mathbb{P}||\mathbb{Q})$ . A set of  $k$  elements  $x^{(1)}, \ldots, x^{(k)}$  is denoted as  $x^{(1:k)}$ , a (possibly unordered) subsequence  $x^{(a)}, \ldots, x^{(b)}$  of those elements is denoted as  $x^{(a:b)}$ , and all the elements in  $x^{(1:k)}$  except of  $x^{(i)}$  is denoted as  $x^{(-i)}$ .

# 2 MULTI-VIEW SSL AND MUTUAL INFORMATION

In multi-view SSL, two (or more) views (potentially generated using augmentations) of the same data sample  $X$  are generated (Bachman et al., 2019; Tian et al., 2020a;b; Chen et al., 2020a; Caron et al., 2020; Zbontar et al., 2021). Views  $V_{1}, V_{2}$  are engineered such that most of the semantic information  $S$  of the data is preserved (Tian et al., 2020b). This process generates two branches where the views are processed to generate representations  $R_{1}, R_{2}$  which are later projected into a lower dimensional space  $Z_{1}, Z_{2}$ . Finally, the model's parameters  $\theta$  are optimised so that the projected representations (projections) from one branch, say  $Z_{1}$ , are predictive of the representations of the other branch  $Z_{2}$  (see Figure 1). In particular, as shown in Section 3, many multi-view SSL methods aim to maximise the mutual information between the projections  $\mathsf{l}(Z_1;Z_2)$ .

Consider the following decomposition of the mutual information (Agakov, 2004; Rodríguez Gálvez et al., 2020)

$$
\mathrm {I} \left(Z _ {1}; Z _ {2}\right) = \mathrm {H} \left(Z _ {2}\right) - \mathrm {H} \left(Z _ {2} \mid Z _ {1}\right) \geq \overbrace {\mathrm {H} \left(Z _ {2}\right)} ^ {\text {E n t r o p y}} + \overbrace {\mathbb {E} \left[ \log \mathrm {q} _ {Z _ {2} \mid Z _ {1}} \left(Z _ {2}\right) \right]}. \tag {1}
$$

The role of both terms from (1) in SSL is distinct: the entropy term determines how much information from one projection can be learnt, while the reconstruction term determines how much of this available information is learnt. For instance, imagine the projections lay on the sphere: the more spread out (higher entropy) the projections of different images are, the more revealing (higher mutual information) it is if projections from different views of the same image are close (lower reconstruction error). On the other hand, if all images of one branch are projected to the same point (lowest entropy, also known as collapse), the projections from the other branch can't reveal any information about them, because their location is always the same.

Although large mutual information does not necessarily imply good downstream performance (Tschannen et al., 2019), maximising this lower bound is a sensitive objective since it promotes learning the semantic information and discarding irrelevant information.

To gain intuition, assume the data can be separated into some semantic  $S$  and some irrelevant variables  $U$  such that  $X = \varphi(S, U)$  and  $S \perp U$ . Further assume that the views can be written as  $V_{1} = \varphi(S, U_{1})$  and  $V_{2} = \varphi(S, U_{2})$ , where  $U_{1}$  and  $U_{2}$  are independent. Since the mutual information between the views contains only semantic information, maximising  $\mathsf{l}(Z_{1}, Z_{2})$  encourages the projections to learn only semantic information. Indeed, imagine the projections contain integrally the irrelevant variables, i.e.  $Z_{1} = (\psi_{1}(S), U_{1}), Z_{2} = (\psi_{2}(S), U_{2})$ , then their mutual information would be the same as if they did not contain them at all:  $\mathsf{l}(\psi_{1}(S), U_{1}; \psi_{2}(S), U_{2}) = \mathsf{l}(\psi_{1}(S); \psi_{2}(S))$ . Furthermore, assume the projections lay in a compact set  $\mathcal{Z} \subseteq \mathbb{R}^{d}$  and the reconstruction density is defined with a semi-metric  $\rho$  such that  $\mathsf{q}_{Z_{2}|Z_{1} = z_{1}}(z_{2}) \propto e^{-\rho(z_{1}, z_{2})}$ . Then, maximising the reconstruction term minimises  $\mathbb{E}[\rho(Z_{1}, Z_{2})]$ , thus pulling together the non-linear mappings  $\pi_{\theta} \circ f_{\theta} \circ \varphi(S, U_{1})$  and  $\pi_{\xi} \circ f_{\xi} \circ \varphi(S, U_{2})$  (see also Figure 1). Therefore, if the reconstruction is maximised, on average, the irrelevant variables  $U_{1}, U_{2}$  are not contributing to  $Z_{1}$  and  $Z_{2}$ , therefore promoting the discarding of irrelevant information.

Remark 1. SSL promotes learning semantic information and discarding irrelevant information. This highlights the importance of the selection of the views of the input data. This is not the object of this paper, and it has been previously studied by Tian et al. (2020b). A similar insight can be obtained with Tsai et al. (2020)'s framework.

Importantly, such benefits can be formalised when directly maximising the lower bound (1), as this maximisation unifies the theory from Zimmermann et al. (2021) and Von Kugelgen et al. (2021).

Theorem 1 (Informal. Details in Appendix B). Assume there is a true data generating process  $X = g(\tilde{Z})$ , where  $g$  is invertible and  $V_{1}$  and  $V_{2}$  can be understood as  $g(\tilde{Z}_1)$  and  $g(\tilde{Z}_2)$ . Then,

1. Under Zimmermann et al. (2021)'s conditions, maximising (1) ensures the projections  $Z$  equate the true latent variables  $\tilde{Z}$  up to affine transformations.  
2. Assume also the true latent variables are separated into semantic (or content)  $S$  and irrelevant (or style)  $U$  variables, such that  $\tilde{Z} = [S,U]$ , where  $[\cdot ]$  is the concatenation operator. Then, under Von Kugelgen et al. (2021)'s conditions, maximising (1) ensures that the projections isolate semantic information, in the sense that there is a bijection from  $Z$  to  $S$ .

# 3 MUTLI-VIEW SSL METHODS MAXIMISE MUTUAL INFORMATION

In this section, we demonstrate how many different multi-view SSL methods aim to maximise the mutual information  $\mathsf{l}(Z_1,Z_2)$ . In Figure 1, we schematically highlight the four prototypes that all main multi-view SSL methods that we are aware of can be partitioned into.

In the following, we exhibit how under the lens of the decomposition (1), the different methods employ different reconstruction densities,  $\mathfrak{q}_{Z_2|Z_1}$  or  $\mathfrak{q}_{W_2|Z_1}$ , and different ways to maximise or control the entropy,  $\mathsf{H}(Z_2)$  or  $\mathsf{H}(W_2)$ , which is empirically shown to be controlled in Appendix F. Importantly, none of them exactly maximises the lower bound (1) of  $\mathsf{l}(Z_1,Z_2)$ , preventing them from the theoretical benefits highlighted in Theorem 1.

First, in Section 3.1, we study contrastive methods (Tian et al., 2020a;b; Bachman et al., 2019; Chen et al., 2020a) and later, in Section 3.2, pure latent variables' (or projections') reconstruction methods (Grill et al., 2020; Chen & He, 2021; Caron et al., 2018; 2020; 2021).

![](images/b8db40daa71351aacc7fa3d3b9d3eac71727f849704480ac92453cdef6cf41cc.jpg)

![](images/6de42c0942bc8cea26d2e5e92a8841d149e3b029b80fbcafeb407cb9c502b572.jpg)

![](images/9ebb4b1f05ccb09e995ec9a98c1b0b856a3709c8541da6f609e48d920478f973.jpg)  
Figure 1: Graphical representation of main multi-view SSL prototypes. Solid lines describe the main variable flow: an image  $X$  is transformed with augmentations  $t_1, t_2$  to generate two views  $V_1, V_2$  that are encoded into representations  $R_1, R_2$  and projected into  $Z_1, Z_2$  (and potentially further processed into  $W_1, W_2$ ). Dashed lines describe method objectives, and dotted lines indicate optional relationships between variables. Top row: the parameters of the encoder  $f$  and projector  $\pi$  are shared for the processing of both views and the projections are manipulated so projections of one view are predictive of the other, and vice-versa. Bottom row: the parameters of the processing of  $V_2$  are distinct and the projections are manipulated so that projections of  $V_1$  are predictive of projections of  $V_2$ . Left column: the projections are not further processed. Right column: the projections are further processed into a surrogate variable  $W_1, W_2$  (potentially using another variable  $C$ ), and then are manipulated so that projections of one view are predictive of the surrogate variable of the other. For example, (a) is followed by SimCLR and EntRecCont, (b) is followed by SwAV and EntRecDisc, (c) is followed by BYOL, and (d) is followed by DINO.

![](images/c469482f051b9be41d7977198abd1d61f0c904f1efa9ae48688504484b51b4fa.jpg)

# 3.1 CONTRASTIVE LEARNING METHODS

Contrastive learning methods (Wu et al., 2018; Bachman et al., 2019; Tian et al., 2020a;b; Chen et al., 2020a;b; He et al., 2020; Ramapuram et al., 2021) have the InfoNCE loss (Oord et al., 2018) at their core and usually have a symmetric structure (Figure 1a).

Consider a batch of  $k$  data samples  $X^{(1:k)}$ . For each projection of each view of a sample  $X^{(i)}$ , say  $Z_1^{(i)}$ , these methods consider the projection of the other view of that image  $Z_2^{(i)}$  its positive pair and try to identify such a pair among a set of other projections (or negative pairs) by minimising a cross-entropy loss based on a similarity score. This similarity score is usually defined as the temperature normalised cosine similarity  $\sin(\cdot, \cdot)/\tau$ . Then, the different methods are essentially distinguished by the projections they consider negative pairs.

In what follows, we introduce two representatives of these methods and their relationship with maximising  $\mathsf{l}(Z_1,Z_2)$ . In Appendix C, we give the details and caveats of the analyses and discuss further contrastive methods, such as (He et al., 2020, MocO), that inherit the analyses below.

Contrastive Multiview Coding In contrastive multiview coding (Tian et al., 2020a;b, CMC), the negative pairs of a projection from one branch, say  $Z_1^{(i)}$ , are all the other projections from the opposite branch  $Z_2^{(-i)}$ . That is, for a batch of size  $k$  the optimised loss is

$$
\mathcal {L} _ {\mathrm {C M C}} (\theta) := \frac {1}{k} \sum_ {b = 1} ^ {2} \sum_ {i = 1} ^ {k} \log \left(\frac {\exp \left(\operatorname {s i m} \left(Z _ {b} ^ {(i)} , Z _ {\bar {b}} ^ {(i)}\right) / \tau\right)}{\sum_ {j = 1} ^ {k} \exp \left(\operatorname {s i m} \left(Z _ {b} ^ {(i)} , Z _ {\bar {b}} ^ {(j)}\right) / \tau\right)}\right), \tag {2}
$$

where  $\bar{b}$  is the opposite branch of  $b$ . This loss benefits from all the properties of the InfoNCE (Oord et al., 2018; Poole et al., 2019). For example, if both sets of projections  $Z_{1}^{(i)}$  as well as  $Z_{2}^{(j)}$  are i.i.d., then minimising  $\mathcal{L}_{\mathrm{CMC}}$  maximises a lower bound on the mutual information; more precisely  $\mathsf{l}(Z_1;Z_2)\geq \log k - \mathcal{L}_{\mathrm{CMC}}(\theta) / 2$ . However, minimising (2) does not directly maximise the lower bound (1). Looking at (10) in Appendix C, we can see how the numerator of the logarithm in (2) is

recovered by considering the reconstruction density  $\mathfrak{q}_{Z_2|Z_1 = z_1}$  to be von Mises-Fisher density with mean direction  $z_{1}$  and parameter  $1 / \tau$  and symmetrising with the reconstruction of the other branch. However, the denominator of the logarithm is not a Joe (1989)'s kernel density estimator (KDE) approximation of the entropy  $\mathsf{H}(Z_2)$  since the average of the logarithm of the kernel is taken over samples of  $\mathbb{P}_{Z_1}$  and not  $\mathbb{P}_{Z_2}$ . Hence, it only maximises an estimation of the entropy if  $Z_{1}$  and  $Z_{2}$  have the same (or approximately the same) marginals, i.e.  $\mathbb{P}_{Z_1}\cong \mathbb{P}_{Z_2}$ .

SimCLR In SimCLR (Chen et al., 2020a), the negative pairs of a projection from one branch, say  $Z_{1}^{(i)}$ , are all the other projections  $Z_{1}^{(-i)}, Z_{2}^{(1:k)}$ . That is, for a batch of size  $k$  the optimised loss is

$$
\mathcal {L} _ {\operatorname {S i m C L R}} (\theta) := \frac {1}{2 k} \sum_ {b = 1} ^ {2} \sum_ {i = 1} ^ {k} \log \left(\frac {\exp \left(\sin \left(Z _ {b} ^ {(i)} , Z _ {b} ^ {(i)}\right) / \tau\right)}{\sum_ {b ^ {\prime} = 1} ^ {2} \sum_ {j = 1} ^ {k} \mathbb {I} \left(\left(i , b\right) \neq (j , b ^ {\prime})\right) \exp \left(\sin \left(Z _ {b} ^ {(i)} , Z _ {b ^ {\prime}} ^ {(j)}\right) / \tau\right)}\right). \tag {3}
$$

This loss does not directly inherit the InfoNCE properties as the CMC loss (see Appendix C.2.1). However, (3) can be approximately rewritten according to the decomposition (1), as is demonstrated in (13) in Appendix C.2.2. The numerator of the logarithm in (3) is then recovered using a von Mises-Fisher reconstruction density, but the denominator is not an approximation of the entropy since the average of the logarithm of the kernel is taken, this time, over samples of both  $\mathbb{P}_{Z_1}$  and  $\mathbb{P}_{Z_2}$ . Nonetheless, this difference allows us to consider that the samples come from the mixture  $\mathbb{P}_Z = 0.5\mathbb{P}_{Z_1} + 0.5\mathbb{P}_{Z_2}$ , and thereby recover the KDE estimator from Joe (1989) of  $\mathsf{H}(Z)$  with a von Mises-Fisher density kernel. Therefore, taking into account the relationship between the Jensen-Shannon's divergence and the entropies of two random variables results in the approximate inequality

$$
\mathsf {l} \left(Z _ {1}; Z _ {2}\right) \gtrsim \log k - \mathcal {L} _ {\operatorname {S i m C L R}} (\theta) - \mathrm {D} _ {\mathrm {J S}} \left(\mathbb {P} _ {Z _ {1}} \| \mathbb {P} _ {Z _ {2}}\right),
$$

which reveals that minimising (3) approximately maximises  $\mathsf{l}(Z_1; Z_2)$  when  $\mathbb{P}_{Z_1}$  and  $\mathbb{P}_{Z_2}$  are equal.

# 3.2 PROJECTIONS' RECONSTRUCTION METHODS

The projections' reconstruction methods (Grill et al., 2020; Chen & He, 2021; Caron et al., 2020; 2021) focus on making sure that predictions from one branch are informative of those from the other branch. To achieve this goal, their loss functions consist of a term that can be understood as the reconstruction term in (1) with the appropriate density.

To avoid collapse in the absence of negative pairs, they have to employ different engineering techniques that as we show can help to maintain a high entropy term in (1) in different ways. Below we analyse the self-distillation methods (Grill et al., 2020, BYOL) and (Caron et al., 2021, DINO) from our information-theoretic viewpoint based on (1). In Appendix D, we further analyse other (non self-distillation) projections' reconstruction methods such as (Chen & He, 2021, SimSiam), (Caron et al., 2018, DeepCluster), and (Caron et al., 2020, SwAV).

BYOL In Bootstrap Your Own Latent (Grill et al., 2020, BYOL), they consider an asymmetric structure (Figure 1c) and try to predict the projections from the bottom branch  $Z_{2}$  using the predictions of the top branch  $Z_{1}$  and a small predictor network  $g_{\theta}$ . For this purpose, they try to minimise the  $\ell_{2}$  normalised mean squared error,

$$
\mathcal {L} _ {\mathrm {B Y O L}} (\theta) := \frac {1}{k} \sum_ {i = 1} ^ {k} \left\| \overline {{g _ {\theta} \left(Z _ {1} ^ {(i)}\right)}} - \overline {{Z _ {2} ^ {(i)}}} \right\| ^ {2} = 2 \left(1 - \frac {1}{k} \sum_ {i = 1} ^ {k} \operatorname {s i m} \left(g _ {\theta} \left(Z _ {1} ^ {(i)}\right), Z _ {2} ^ {(i)}\right)\right)
$$

using gradient descent, where  $\overline{a} \coloneqq a / \|a\|$ . Note that this is equivalent, up to constants that do not affect the optimisation, to maximising the reconstruction term in the decomposition (1) with a von Mises-Fisher reconstruction density with mean direction  $\overline{g_{\theta}(Z_1)}$  and concentration parameter 1, i.e.,  $\mathfrak{q}_{Z_2|Z_1 = z_1}(z_2) \propto \exp(\sin(g_{\theta}(z_1), z_2))$ . Note that the parameters of the branch that outputs the predicted projections  $Z_2$  are parameterised with different parameters  $\xi$ . Hence, if these parameters were fixed or modified so that  $\mathsf{H}(Z_2)$  is increasing or maintained constant, then minimising  $\mathcal{L}_{\mathrm{BYOL}}$  would indeed maximise the mutual information  $\mathsf{l}(Z_1; Z_2)$ . Finding a way to fix or modify them in

such a way is however challenging. For example, fixing  $\xi$  to random values ensures constant entropy  $\mathsf{H}(Z_2)$ , but also means that the then random projection  $Z_{2}$  contains very little information about  $X$ . Thus, in this case, while minimising  $\mathcal{L}_{\mathrm{BYOL}}$  would maximise  $\mathsf{l}(Z_1;Z_2)$ , the information learned is still little as by the data processing inequality  $\mathsf{l}(Z_1;Z_2)\leq \mathsf{l}(X;Z_2)$ . On the other hand, if  $\xi$  depends on  $\theta$ , there is the risk of collapse: for example, in the extreme case of  $\xi = \theta$ , minimising  $\mathcal{L}_{\mathrm{BYOL}}$  will maximise  $-\mathsf{H}(Z_2|Z_1)$ , and an optimal solution  $\theta^{\star}$  could be a highly concentrated or degenerate  $Z_{1}$  and  $Z_{2}$  around one point  $z$ , under which  $\mathsf{H}(Z_2)\to -\infty$ , which clearly does not maximise  $\mathsf{l}(Z_1;Z_2)$ .

In BYOL, they circumvent these issues by updating the parameters  $\xi$  during the optimisation with the moving average  $\xi \gets \lambda \xi + (1 - \lambda) \theta$  for some  $\lambda \in (0, 1)$  close to 1. The idea (hypothesis) is two-fold: on the one hand, while  $\xi$  does depend on  $\theta$ , the dependence is weak enough so that  $\mathsf{H}(Z_2)$  is not degrading to values yielding trivial bounds; and on the other hand, the dependence of  $\xi$  on  $\theta$ , while weak, still makes sure that the representations  $Z_2$  capture information about the data. This hypothesis is backed up by the results sweeping the parameter  $\lambda$  in (Grill et al., 2020). In fact, it has later been seen (Caron et al., 2021) that this dependence resembles a Polyak-Ruppert averaging with exponential decay (Polyak & Juditsky, 1992; Ruppert, 1988), which is standard practice to improve the performance of the model, e.g. (Jean et al., 2014).

DINO Caron et al. (2021, DINO) also consider an asymmetric structure (Figure 1d) and, similarly to DeepCluster and SwAV, generate a discrete surrogate variable  $W_{2} = \phi(Z_{2})$  and try to minimise a cross entropy term. More precisely, they minimise

$$
\mathcal {L} _ {\mathrm {D I N O}} (\theta) := \frac {1}{k} \sum_ {i = 1} ^ {k} \mathfrak {s} \left(\left(Z _ {2} ^ {(i)} - C\right) / \tau_ {2}\right) ^ {\intercal} \log \mathfrak {s} \left(Z _ {1} ^ {(i)} / \tau_ {1}\right),
$$

where  $C$  is some centring variable,  $\tau_{1},\tau_{2}$  are temperature hyperparameters, and s is the softmax operator. Letting  $\mathsf{p}_{W_2|Z_2 = z_2} = \mathsf{s}\big((z_2 - C) / \tau_2\big)$  and  $\mathsf{q}_{W_2|Z_1 = z_1} = \mathsf{s}(z_1 / \tau_1)$  shows that minimising  $\mathcal{L}_{\mathrm{DINO}}$  directly maximises the reconstruction term in the decomposition (1) of  $\mathsf{l}(Z_1,W_2)\leq \mathsf{l}(Z_1,Z_2)$ .

DINO does not directly maximise the entropy  $\mathsf{H}(W_2)$  to avoid collapse. However, they promote a high conditional entropy  $\mathsf{H}(W_2|Z_2)$  through the centring before the softmax operation defining  $\mathsf{p}_{W_2|Z_2}$ . To be precise, the centre  $C$  is updated with a moving average of the previous projections, that is,  $C\gets \mu C + \frac{(1 - \mu)}{k}\sum_{i = 1}^{k}Z_{2}$  for some  $\mu \in (0,1)$ . Then, the right balance between the moving average and temperature parameters  $\mu$  and  $\tau_{2}$  adjusts how uniform the conditional density  $\mathsf{p}_{W_2|Z_2}$  should be. Hence, since  $\mathsf{H}(W_2|Z_2)\leq \mathsf{H}(W_2)$ , controlling the conditional entropy controls  $\mathsf{H}(W_2)$ .

Finally, similarly to BYOL, DINO faces the potential problem of obtaining useless representations due to uninformative targets if the parameters  $\xi$  do not ensure that the projections  $Z_{2}$  capture enough information about the data  $X$ . They solve this issue as in BYOL updating them with a moving average  $\xi \gets \lambda \xi + (1 - \lambda)\theta$  for some  $\lambda \in (0,1)$ . As previously mentioned, with the appropriate selection of  $\lambda$ , this resembles a Polyak-Ruppert averaging with exponential decay (Polyak & Juditsky, 1992; Ruppert, 1988) and makes sure that  $Z_{2}$  captures information about the data  $X$  (Caron et al., 2021).

# 4 THE ENTREC METHOD

Previously, we established how many multi-view SSL methods aim to maximise the mutual information between the projections on both branches  $\mathsf{l}(Z_1;Z_2)$ , and that they can be understood by decomposing the mutual information into an entropy and a reconstruction term as in (1). However, none of these methods takes such a decomposition and tries to maximise these two terms directly.

In this section, we present the EntRec method which does exactly that, and naively maximises both entropy and reconstruction terms. The method comes in two variants: (i) EntRecCont, a direct maximisation of  $\mathsf{I}(Z_1,Z_2)$ , where the entropy is estimated with a KDE, that follows Figure 1a's structure; or (ii) EntRecDisc, a generation of a discrete surrogate random variable  $W_{b} = \phi (Z_{b})$  and posterior maximisation of  $\left(\mathsf{I}(Z_1;W_2) + \mathsf{I}(W_1;Z_2)\right) / 2\leq \mathsf{I}(Z_1;Z_2)$ , where the entropy of  $W_{b}$  can be easily estimated with a plug-in estimator, that follows Figure 1c's structure. EntRecCont, unlike all of the methods described previously, directly maximises the lower bound (1) on the mutual information  $\mathsf{I}(Z_1;Z_2)$ , and therefore enjoys the theoretical properties from Theorem 1, such as recovering true latent variables and separating semantic from irrelevant information. However, the KDE requires a large number of samples to properly estimate the entropy. EntRecDisc addresses this potential drawback by estimating the entropy of a discrete (surrogate) random variable instead,

although at the price of maximising a lower bound on  $\mathsf{l}(W_1;Z_2)$  only. Hence, it maximises a looser bound on  $\mathsf{l}(Z_1,Z_2)$  and does not benefit from the theoretical properties of Theorem 1.

# 4.1 ENTRECCONT

EntRecCont considers the mutual information decomposition from (1) and directly maximises an estimate of the lower bound by maximising the loss function

$$
\mathcal {L} _ {\text {E n t R e c C o n t}} (\theta) := - \frac {1}{2} \sum_ {b = 1} ^ {2} \left(\hat {\mathrm {H}} \left(Z _ {b}, Z _ {b} ^ {(1: k)}\right) + \frac {1}{k} \sum_ {i = 1} ^ {k} \log \left(\mathrm {q} _ {Z _ {b} \mid Z _ {\bar {b}} = Z _ {\bar {b}} ^ {(i)}} \left(Z _ {b} ^ {(i)}\right)\right)\right), \tag {4}
$$

where  $\hat{\mathsf{H}}(Z_b; Z_b^{(1:k)})$  is an estimate of the entropy and  $\mathfrak{q}_{Z_b|Z_{\bar{b}}}$  is a parameterised reconstruction density. Multiple potential estimates of the entropy  $\mathsf{H}(Z_b)$  exist, but this paper considers those generated with Joe (1989)'s KDE, which comes with the same caveats mentioned for the analysis of contrastive methods in Appendix C.3. More precisely, the estimator takes the form

$$
\hat {\mathsf {H}} (Z _ {b}, Z _ {b} ^ {(1: k)}) = \frac {1}{k} \sum_ {i = 1} ^ {k} \log \left(\frac {1}{k h ^ {d}} \sum_ {j = 1} ^ {k} \mathfrak {q} \left(\frac {Z _ {b} ^ {(i)} - Z _ {b} ^ {(j)}}{h}\right)\right),
$$

where  $\mathbf{q}$  is some kernel density and  $h \in \mathbb{R}_+$  is its bandwidth.

The loss (4) recovers Wang & Isola (2020)'s alignment-uniformity loss up to constants independent of the parameters  $\theta$  when the reconstruction density is  $\mathfrak{q}_{Z_b|Z_{\tilde{b}} = z_{\tilde{b}}}(\boldsymbol {z}_b)\propto \exp \left(-\| \boldsymbol {z}_b - \boldsymbol {z}_{\tilde{b}}\|^\alpha\right)$  and the kernel density is Gaussian. Applying the log-sum inequality to the entropy estimation term fully recovers that loss, revealing it is an estimation of a looser bound of the mutual information  $\mathsf{l}(Z_1,Z_2)$ .

Moreover, EntRecCont enjoys the following theoretical benefits on its asymptotic behaviour, whose proof follows readily from the law of large numbers and (Joe, 1989, Section 4).

Theorem 2 (The EntRec loss (4) tends to a proper bound on  $\mathsf{l}(Z_1;Z_2)$ ). If  $X_{i}$  are i.i.d. for all  $i\in [k]$  and  $f$  and  $\pi$  do not use batch normalisation, then, for a constant bandwidth  $h > 0$

$$
\lim  _ {k \rightarrow \infty} \mathcal {L} _ {\text {E n t R e c C o n t}} (\theta) - C _ {\mathrm {K D E}} (h, k) = - \frac {1}{2} \sum_ {b = 1} ^ {2} \left(\mathsf {H} \left(Z _ {b}\right) + \mathbb {E} \left[ \log \left(\mathsf {q} _ {Z _ {b} \mid Z _ {b}} \left(Z _ {b}\right)\right)\right]\right) \geq - \mathsf {I} \left(Z _ {1}; Z _ {2}\right), \tag {5}
$$

where  $C_{\mathrm{KDE}}(h,k) \in \mathcal{O}(h^8)$ , and where the convergence rate is  $\mathcal{O}(k^{-1}h^{4 - d}) + \mathcal{O}(k^{-2}h^{-2d})$  and  $d$  is the dimension of  $Z$ . Moreover, if  $h \in \mathcal{O}(k^{-\frac{1}{d}})$  and  $d > 4$ , then (8) still holds and  $C_{\mathrm{KDE}}(h,k) \to 0$ .

This theorem reveals that as the batch size increases, EntRecCont maximises exactly the lower bound (1) and therefore it enjoys the theoretical properties highlighted in Section 2 (details in Appendix B): (i) if there exists a true generative process that generates the data  $X$  from some true latent variables, maximising (4) may recover these latent variables; and (ii) if these latent variables are separated into some semantic (or content) variables and some irrelevant (or style) variables, maximising (4) may isolate and recover the content latent variables.

# 4.2 ENTRECDISC

The KDE of the entropy converges slowly for high dimensions (c.f. Theorem 2), which requires large batch sizes. It is also computationally costly (requires  $\mathcal{O}(k^2 d)$  operations) for large batch sizes. Therefore, it is suitable to have an alternative to (4) that does not involve KDEs, for example considering discrete random variables instead of continuous ones.

EntRecDisc considers a discrete surrogate random variable  $W_{b} = \phi (Z_{b})$  and maximises a lower bound of  $\left(\mathsf{l}(Z_1;W_2) + \mathsf{l}(Z_2;W_1)\right) / 2$ . This way, (i) considering the lower bound from (1) to each of the terms allows us to deal with the entropy of a discrete random variable and avoid KDEs, and (ii) by the data processing inequality, we are still maximising a lower bound on  $\mathsf{l}(Z_1,Z_2)$ . To be precise, this version of EntRec minimises the loss function

$$
\mathcal {L} _ {\text {E n t R e c D i s c}} (\theta) := - \frac {1}{2} \sum_ {b = 1} ^ {2} \left(\hat {\mathrm {H}} \left(W _ {b}; W _ {b} ^ {(1: k)}\right) + \frac {1}{k} \sum_ {i = 1} ^ {k} \mathbb {E} \left[ \log \left(\mathrm {q} _ {W _ {b} | Z _ {\tilde {b}} = Z _ {\tilde {b}} ^ {(i)}} \left(W _ {b} ^ {(i)}\right)\right) \right]\right). \tag {6}
$$

At first sight, both (4) and (6) seem indistinguishable except from the fact that now  $W_{b}$  is replacing  $Z_{b}$  and that the loss is a looser bound on  $\mathsf{l}(Z_1;Z_2)$ . However, this extra processing of the projections allows us to estimate the entropy better and to calculate the reconstruction term (cross-entropy) exactly. For instance, let  $\mathsf{p}_{W_b|Z_b = z_b} = \mathsf{s}(z_b)$ , where  $\mathsf{s}$  is the softmax operator. Then,  $W_{b}$  is a discrete random variable in  $[d]$  and  $\mathsf{H}(W_b)$  may be estimated with the plug-in estimator using the empirical estimate of the marginal, i.e.  $\hat{\mathsf{p}}_{W_b} = \frac{1}{k}\sum_{i = 1}^{k}\mathsf{p}_{W_b|Z_b = Z_b^{(i)}}$ . Finally, letting the reconstruction density be  $\mathsf{q}_{W_b|Z_b = z_b} = \mathsf{s}(z_b)$  results in the following loss function

$$
\mathcal {L} _ {\text {E n t R e c D i s c}} (\theta) = - \frac {1}{2 k} \sum_ {b = 1} ^ {2} \sum_ {i = 1} ^ {k} \left(- \mathsf {s} \left(Z _ {b} ^ {(i)}\right) ^ {\intercal} \log \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \mathsf {s} \left(Z _ {b} ^ {(j)}\right)\right) + \mathsf {s} \left(Z _ {b} ^ {(i)}\right) ^ {\intercal} \log \left(\mathsf {s} \left(Z _ {\bar {b}} ^ {(i)}\right)\right)\right). \tag {7}
$$

As opposed to the KDE of the entropy, the plug-in estimate only requires  $\mathcal{O}(kd)$  operations. Furthermore,  $\hat{\mathsf{p}}_{W_b}$  converges to  $\mathsf{p}_{W_b}$  at a  $\mathcal{O}(k^{-1/2})$  rate by the law of large numbers, meaning that the plug-in estimate of the entropy also converges faster than the KDE. Thus, (7) has a better asymptotic behaviour than (4). This is formalised in the following theorem.

Theorem 3 (The EntRec loss (7) tends to a proper bound on  $\mathsf{l}(Z_1;Z_2)$ ). If  $X_{i}$  are i.i.d. for all  $i\in [k]$  and  $f$  and  $\pi$  do not use batch normalisation, then

$$
\lim  _ {k \rightarrow \infty} \mathcal {L} _ {\text {E n t R e c D i s c}} (\theta) = - \frac {1}{2} \sum_ {b = 1} ^ {2} \left(\mathrm {H} \left(W _ {b}\right) + \mathbb {E} \left[ \log \left(\mathrm {q} _ {W _ {b} \mid Z _ {\bar {b}}} \left(W _ {b}\right)\right)\right]\right) \geq - \mathrm {I} \left(Z _ {1}; Z _ {2}\right), \tag {8}
$$

where the convergence rate is  $\mathcal{O}(k^{-1 / 2})$

# 5 EXPERIMENTS

While EntRec has a principled derivation that allows for direct maximisation of the lower bound in (1) and as a result is equipped with desirable theoretic properties, its practical use has yet to be investigated. We do this in a series of experiments, in all of which we use ImageNet (Deng et al., 2009) for pre-training. We compare both variants of EntRec with all methods analysed in Section 3. All experimental details can be found in Appendix E.

In the main part of the paper, we analyse how EntRec compares to the other methods in terms of top-1 classification accuracy on the ImageNet test set under the linear evaluation protocol, and further study how changes in individual training hyperparameters affect this metric for each method if no other training or model hyperparameter is changed. In Appendix F, we further include a comparison in terms of transfer learning performance (fine-tuned top-1 classification accuracy on 10 natural image data sets that differ from ImageNet) and a qualitative analysis of the behaviour of the entropy term during training for EntRecDisc and DINO (for which the entropy can be estimated accurately as they use discrete surrogate variables).

As can be seen from the first column of Table 1 and Table 2, EntRec's performance on the ImageNet test set is comparable to that of the other methods (with the exception of DINO, which outperforms all other methods). The same can be observed with respect to its transfer learning performance in the additional experiments that are included in Appendix F.

Table 1: Performance of the different methods across batch sizes after 400 epochs of training.  

<table><tr><td></td><td>Accuracy</td><td colspan="3">ΔAccuracy wrt. 4096</td></tr><tr><td></td><td>4096</td><td>2048</td><td>1024</td><td>512</td></tr><tr><td>SimCLR</td><td>69.5</td><td>-3.1</td><td>-2.5</td><td>-4.5</td></tr><tr><td>CMC</td><td>69.1</td><td>-1.1</td><td>-2.9</td><td>-4.8</td></tr><tr><td>BYOL</td><td>68.3</td><td>+0.3</td><td>-15.0</td><td>-34.2</td></tr><tr><td>DINO</td><td>71.6</td><td>+0.5</td><td>-15.0</td><td>-67.0</td></tr><tr><td>EntRecCont</td><td>69.7</td><td>-0.7</td><td>-2.3</td><td>-4.2</td></tr><tr><td>EntRecDisc</td><td>66.9</td><td>-1.0</td><td>-2.2</td><td>-3.9</td></tr></table>

Table 2: Performance of the different methods across epochs with batch size of 4096.  

<table><tr><td></td><td>Accuracy</td><td colspan="3">ΔAccuracy wrt. 400</td></tr><tr><td></td><td>400</td><td>300</td><td>200</td><td>100</td></tr><tr><td>SimCLR</td><td>69.5</td><td>-1.0</td><td>-2.2</td><td>-5.4</td></tr><tr><td>CMC</td><td>69.1</td><td>-0.6</td><td>-1.8</td><td>-5.6</td></tr><tr><td>BYOL</td><td>68.3</td><td>-2.8</td><td>-6.9</td><td>-15.7</td></tr><tr><td>DINO</td><td>71.6</td><td>+0.2</td><td>+0.3</td><td>-1.2</td></tr><tr><td>EntRecCont</td><td>69.7</td><td>-0.8</td><td>-2.1</td><td>-5.9</td></tr><tr><td>EntRecDisc</td><td>66.9</td><td>-0.2</td><td>-1.1</td><td>-4.1</td></tr></table>

Also, we see that EntRec compares overall favourably in terms of robustness to changes in training hyperparemeters:

- Batch size. It is known that lowering the batch size can adversely affect the performance of SSL methods (Chen et al., 2020a). In Table 1 we see this is also true for EntRec, however to a lesser extent than for all other methods. Importantly, the projections' reconstruction methods (DINO, BYOL) which rely on engineering techniques to maintain high entropy can be very severely affected by lower batch sizes, which potentially call for further hyperparameter adjustments to recover performance.  
- Epochs. Furthermore, SSL methods typically need a very high number of total training epochs to achieve their strongest performances (Grill et al., 2020). In Table 2 we see this is also true for EntRec. However, this time another method, DINO, seems to be the least affected by a lower number of total training epochs. Still, EntRec performs comparably to the remaining SSL methods and is notably more stable than BYOL again.

Altogether, these experimental results showcase that EntRec can indeed have practical use as a good objective for multi-view SSL beyond its theoretical benefits highlighted in the previous sections.

# 6 CONCLUSION

We provided a unifying information theoretic analysis of common SSL methods, showing how they partially or approximately maximise a lower bound to the mutual information between the learned representations of distinct views of the same datum. Based on this analysis, we introduced EntRec, a simple SSL method that directly maximises this lower bound. We showed that it possesses a range of desirable theoretical properties, such as recovering the true latent variables of the underlying generative process or isolating content from style in such true latent variables. Furthermore, we demonstrated empirically that its classification performance is comparable to the existing SSL methods and most notably robust to changes in individual training hyperparameters such as the batch size or the number of training epochs.

Limitations and future directions Maximising mutual information is not enough to learn good representations, and strong inductive biases are important. For instance, the usage of certain reconstruction densities or projection spaces when maximising the lower bound (1) grants the theoretical properties highlighted in Theorem 1 and deepened in Appendix B. Now that we know these methods maximise the mutual information between the representations of different views, the next step is to (i) understand which inductive biases they possess to justify their difference in performance (e.g. why DINO performs better when properly tuned), and (ii) designed methods from first principles that both maximise this mutual information and have these inductive biases.

Reproducibility statement Regarding our theoretical results, we made an effort to give clear explanations of any assumptions and complete proofs of all claims in the main part of the paper in Appendices B, C, and D. Regarding our experimental results, we included a section in the Appendix E that clearly states the experimental protocol used to obtain these results. Furthermore, we are working to release the code used in this paper as soon as possible.

# REFERENCES

David Barber Felix Agakov. The im algorithm: a variational approach to information maximization. Advances in neural information processing systems, 16(320):201, 2004.  
Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, and Kevin Murphy. Deep variational information bottleneck. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=HyxQzBceg.  
YM Asano, C Rupprecht, and A Vedaldi. Self-labelling via simultaneous clustering and representation learning. In International Conference on Learning Representations, 2019.  
Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/file/ddf354219aac374f1d40b7e760ee5bb7-Paper.pdf.  
Adrien Bardes, Jean Ponce, and Yann Lecun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. In ICLR 2022-10th International Conference on Learning Representations, 2022.  
Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In Proceedings of the European conference on computer vision (ECCV), pp. 132-149, 2018.  
Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. Advances in Neural Information Processing Systems, 33:9912-9924, 2020.  
Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9650-9660, 2021.  
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597-1607. PMLR, 2020a.  
Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15750-15758, 2021.  
Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020b.  
Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248-255. IEEE, 2009.  
Marco Federici, Anjan Dutta, Patrick Forre, Nate Kushman, and Zeynep Akata. Learning robust representations via multi-view information bottleneck. arXiv preprint arXiv:2002.07017, 2020.  
Ziv Goldfeld and Yury Polyanskiy. The information bottleneck problem and its applications in machine learning. IEEE Journal on Selected Areas in Information Theory, 1(1):19-38, 2020.

Priya Goyal, Piotr Dólar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
Robert M Gray. Entropy and information theory. Springer Science & Business Media, 2011.  
Jean-Bastien Grill, Florian Strub, Florent Alché, Coretin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271-21284, 2020.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729-9738, 2020.  
Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural networks for machine learning lecture 6a overview of mini-batch gradient descent. Cited on, 14(8):2, 2012.  
R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. arXiv preprint arXiv:1808.06670, 2018.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pp. 448-456. PMLR, 2015.  
Sébastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. On using very large target vocabulary for neural machine translation. arXiv preprint arXiv:1412.2007, 2014.  
Longlong Jing and Yingli Tian. Self-supervised visual feature learning with deep neural networks: A survey. IEEE transactions on pattern analysis and machine intelligence, 43(11):4037-4058, 2020.  
Harry Joe. Estimation of entropy and other functionals of a multivariate density. Annals of the Institute of Statistical Mathematics, 41(4):683-697, 1989.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Communications of the ACM, 60(6):84-90, 2017.  
Shuo Liu, Adria Malol-Ragolta, Emilia Parada-Cabeleiro, Kun Qian, Xin Jing, Alexander Kathan, Bin Hu, and Bjoern W Schuller. Audio self-supervised learning: A survey. arXiv preprint arXiv:2203.01205, 2022.  
Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.  
David McAllester and Karl Stratos. Formal limitations on the measurement of mutual information. In International Conference on Artificial Intelligence and Statistics, pp. 875-884. PMLR, 2020.  
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization, 30(4):838-855, 1992.  
Ben Poole, Sherjil Ozair, Aaron Van Den Oord, Alex Alemi, and George Tucker. On variational bounds of mutual information. In International Conference on Machine Learning, pp. 5171-5180. PMLR, 2019.  
Jason Ramapuram, Dan Busbridge, Xavier Suau, and Russ Webb. Stochastic contrastive learning. arXiv preprint arXiv:2110.00552, 2021.

Borja Rodríguez Galvez, Ragnar Thobaben, and Mikael Skoglund. The convex information bottle-neck lagrangian. Entropy, 22(1):98, 2020.  
David Ruppert. Efficient estimators from a slowly convergent robbins-monro procedure. School of Oper. Res. and Ind. Eng., Cornell Univ., Ithaca, NY, Tech. Rep, 781, 1988.  
Andrew M Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan D Tracey, and David D Cox. On the information bottleneck theory of deep learning. Journal of Statistical Mechanics: Theory and Experiment, 2019(12):124020, 2019.  
Ravid Shwartz-Ziv, Randall Balestriero, and Yann LeCun. What do we maximize in self-supervised learning? In First Workshop on Pre-training: Perspectives, Pitfalls, and Paths Forward at ICML 2022.  
Richard Sinkhorn. Diagonal equivalence to matrices with prescribed row and column sums. ii. Proceedings of the American Mathematical Society, 45(2):195-198, 1974.  
Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In European conference on computer vision, pp. 776-794. Springer, 2020a.  
Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning? Advances in Neural Information Processing Systems, 33:6827-6839, 2020b.  
Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000.  
Yao-Hung Hubert Tsai, Yue Wu, Ruslan Salakhutdinov, and Louis-Philippe Morency. Self-supervised learning from a multi-view perspective. In International Conference on Learning Representations, 2020.  
Michael Tschannen, Josip Djolonga, Paul K Rubenstein, Sylvain Gelly, and Mario Lucic. On mutual information maximization for representation learning. arXiv preprint arXiv:1907.13625, 2019.  
Petar Velickovic, William Fedus, William L Hamilton, Pietro Lio, Yoshua Bengio, and R Devon Hjelm. Deep graph infomax. *ICLR (Poster)*, 2(3):4, 2019.  
Julius Von Kugelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Scholkopf, Michel Besserve, and Francesco Locatello. Self-supervised learning with data augmentations provably isolates content from style. Advances in neural information processing systems, 34:16451-16467, 2021.  
Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International Conference on Machine Learning, pp. 9929-9939. PMLR, 2020.  
Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3733-3742, 2018.  
Yang You, Igor Gitman, and Boris Ginsburg. Large batch training of convolutional networks. arXiv preprint arXiv:1708.03888, 2017.  
Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised learning via redundancy reduction. In International Conference on Machine Learning, pp. 12310-12320. PMLR, 2021.  
Roland S Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, and Wieland Brendel. Contrastive learning inverts the data generating process. In International Conference on Machine Learning, pp. 12979-12990. PMLR, 2021.
