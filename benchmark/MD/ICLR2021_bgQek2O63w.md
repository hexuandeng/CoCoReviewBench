# SELF-SUPERVISED ADVERSARIAL ROBUSTNESS FOR THE LOW-LABEL, HIGH-DATA REGIME

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recent work discovered that training models to be invariant to adversarial perturbations requires substantially larger datasets than those required for standard classification. Perhaps more surprisingly, these larger datasets can be "mostly" unlabeled. Pseudo-labeling, a technique simultaneously pioneered by four separate and simultaneous work in 2019, has been proposed as a competitive alternative to labeled data for training adversariably robust models. However, when the amount of labeled data decreases, the performance of pseudo-labeling catastrophically drops, thus questioning the theoretical insights put forward by Uesato et al. (2019), which suggest that the sample complexity for learning an adversarially robust model from unlabeled data should match the fully supervised case. We introduce Bootstrap Your Own Robust Latents (BYORL), a self-supervised learning technique based on BYOL for training adversarially robust models. Our method enables us to train robust representations without any labels (reconciling practice with theory). This robust representation can be leveraged by a linear classifier to train adversarially robust models. We evaluate BYORL and pseudo-labeling on CIFAR-10 and demonstrate that BYORL achieves significantly higher robustness in the low-label regime (i.e., models resulting from BYORL are up to two times more accurate). Experiments on CIFAR-10 against  $\ell_2$  and  $\ell_{\infty}$  norm-bounded perturbations demonstrate that BYORL achieves near state-of-the-art robustness with as little as 500 labeled examples. We also note that against  $\ell_2$  norm-bounded perturbations of size  $\epsilon = 128 / 255$ , BYORL surpasses the known state-of-the-art with an accuracy under attack of  $77.61\%$  (against  $72.91\%$  for the prior art).

# 1 INTRODUCTION

As neural networks tackle challenges ranging from ranking content on the web (Covington et al., 2016) to autonomous driving (Bojarski et al., 2016) via medical diagnostics (De Fauw et al., 2018), it has becomes increasingly important to ensure that deployed models are robust and generalize to various input perturbations. Unfortunately, despite their success, neural networks are not intrinsically robust. In particular, the addition of small but carefully chosen deviations to the input, called adversarial perturbations, can cause the neural network to make incorrect predictions with high confidence (Carlini & Wagner, 2017a; Goodfellow et al., 2014; Kurakin et al., 2016; Szegedy et al., 2013). Starting with Szegedy et al. (2013), there has been a lot of work on understanding and generating adversarial perturbations (Carlini & Wagner, 2017b; Athalye & Sutskever, 2017), and on building models that are robust to such perturbations (Papernot et al., 2015; Madry et al., 2017; Kannan et al., 2018). Robust optimization techniques, like the one developed by Madry et al. (2017), learn robust models by trying to find the worst-case adversarial examples (by using gradient ascent on the training loss) at each training step and adding them to the training data.

Since Madry et al. (2017), various modifications to their original implementation have been proposed (Zhang et al., 2019; Pang et al., 2020; Huang et al., 2020; Qin et al., 2019). We highlight the simultaneous work from Carmon et al. (2019); Uesato et al. (2019); Zhai et al. (2019); Najafi et al. (2019) that pioneered the use of additional unlabeled data using pseudo-labeling. While, theoretically,

![](images/813d51b428273aa808f7f83ad6dd2e23dc56194392c8892347ab991208f7da4e.jpg)

![](images/d2a3b63194c19b54b437602bd79988b3545e8f0fb813395187eb3258eca3d345.jpg)

![](images/d3fbc0303561549a07cc9eab7112d98b38c64a2329c92ac39b2e0f659697eeac.jpg)

![](images/fceb2d980fc01e887da40e5276d8956981e9c26cfa0fb3fb61b7055fa021c2d0.jpg)  
Retrieved nearest neighbors

![](images/f149de1b679b4f34d955b0f645ac6dbbac12cad227a49fb768962a4675ea6dd5.jpg)

![](images/bfa4fed749648e4997dde865100c9e94c764f361e2449817f4e7d92be7e17046.jpg)

![](images/80eb88ac1163289d85a0d31d600c04411f06986c28a4e5c19e57e91a6175d32d.jpg)  
Clean query image Modified query image

![](images/8a7409281c3765634d929e004a81ee5ce8116debf145c9b1e5619638f6dc8021.jpg)

![](images/3a71c948ea66237815a4832925afef72593d6d2863830aca9d1cd26111742fed.jpg)

![](images/8f676668cc93d741f5aabbd6cf7c7af44cfbbd04d35e2981c629e9b93a9d481b.jpg)

![](images/36c38e4e4d0534cdc969a29ef54ec752a18514994dbf1a4479db89799d30a048.jpg)

![](images/3a8778f25a7a83be3002a07cdb41c07146e444200cea8b3429eb07133ecae094.jpg)  
Figure 1: Dangers of using non-robust representation learning. We use a non-robust self-supervised learning technique to learn image representations (i.e., BYOL; Grill et al., 2020). The right-hand side shows CIFAR-10 images closest (in representation space using cosine similarity) to the query image on the left. The top row demonstrates that, when given an unmodified image of an airplane, the nearest matches resemble that query image either visually or semantically. The bottom row demonstrates that a seemingly identical image can be used to retrieve images of animals which are both visually and semantically far from the query image.

robustness can be achieved with only limited amount of labeled data, in practice, it remains difficult to train models that are both robust and accurate in the low-label regime.

Finally, we note that there has been little work towards learning adversarially robust representations that allow for efficient training on multiple downstream tasks (with the exception of Cemgil et al., 2019; Kim et al., 2020). Learning good image representations is a key challenge in computer vision (Wiskott & Sejnowski, 2002; Hinton et al., 2006), and many different approaches have been proposed. Among them state-of-the-art methods include contrastive methods (Chen et al., 2020; Oord et al., 2018; He et al., 2020) and latent bootstrapping (Grill et al., 2020). However, none of these recent works consider the impact of adversarial manipulations, which can render the widespread use of general representations difficult. As an example, Fig. 1 demonstrates the effect that a non-robust representation has on a content retrieval task, where two seemingly identical query images are matched to widely different images (i.e., their nearest neighbors in representation space).

In this paper, we tackle the issue of learning robust representations that are adversarially robust on multiple downstream tasks in the low-label regime. Our contributions are as follows:

- We formulate Bootstrap Your Own Robust Latents (BYORL), a modification of Bootstrap Your Own Latents (BYOL) (Grill et al., 2020) that enables the training of robust representations without the need for any label information. These representations allow for efficient training on multiple downstream tasks with a fraction of the original labels.  
- Most notably, even with only  $1\%$  of the labels, BYORL comes close to or even exceeds previous state-of-the-art which uses all labels. For example, for  $\ell_2$  norm-bounded perturbations of size  $\epsilon = 128 / 255$  on CIFAR-10, BYORL achieves  $75.50\%$  robust accuracy compared to  $72.91\%$  for the previous state-of-the-art using all labels. BYORL reaches  $77.61\%$  robust accuracy when using all available labels (and additional unlabeled data extracted from 80M-TINYIMAGES; Torralba et al., 2008).  
- Finally, we show that the representations learned through BYORL transfer much better to downstream tasks (i.e., downscaled STL-10 (Coates et al., 2011) and CIFAR-100 (Krizhevsky et al., 2014)) than those obtained through pseudo-labeling and standard adversarial training. Importantly, we also highlight that classifiers trained on top of these robust representations do not need to be trained adversarily to be robust.

# 2 RELATED WORK

Adversarial robustness. Biggio et al. (2013) and Szegedy et al. (2013) observed that neural networks, while they achieve high accuracy on test data, are vulnerable to carefully crafted inputs perturbations, called adversarial examples. Since then, there has been several work on building stronger adversarial examples as well as defense against such adversarial examples (Carlini & Wagner, 2017b; Athalye & Sutskever, 2017; Goodfellow et al., 2014; Papernot et al., 2015; Madry

et al., 2017; Kannan et al., 2018). Arguably, the most successful approach for learning adversially robust models is adversarial training as proposed by Madry et al. (Athalye et al., 2018; Uesato et al., 2018). This classic version of adversarial training has been augmented in different ways – with changes in the attack procedure (e.g., by incorporating momentum; Dong et al., 2017), loss function (e.g., logit pairing; Mosbach et al., 2018) or model architecture (e.g., using attention; Zoran et al., 2020). We also highlight Zhang et al. (2019), who proposed TRADES which balances the trade-off between standard and robust accuracy. By construction, to the contrary of our proposed method, all aforementioned approaches use label information and are not capable of learning generic representations that might be useful to multiple downstream tasks.

Semi- and self-supervised learning. Since human annotations can be expensive, semi- and self-supervised learning approaches that leverage both labeled and unlabeled data have been proposed to improve model performance (Chapelle et al., 2009; Bachman et al., 2014; Berthelot et al., 2019; Laine & Aila, 2017; Miyato et al., 2018; Sajjadi et al., 2016; Xie et al., 2019). A common approach is to train networks to solve a manually-predefined pretext task (e.g., predicting the relative location of image patches) for representation learning, and later use the learned representation for a specific supervised learning task (Dosovitskiy et al., 2014; Doersch et al., 2015; Noroozi & Favaro, 2016). Recently, contrastive learning that uses different views of multiple augmented images has been an effective tool to learn rich representation from unsupervised data (Oord et al., 2018; Chen et al., 2020; He et al., 2020; Tian et al., 2020), as these methods achieve comparable performance to fully-supervised models. While these works focus on improving standard generalization, we leverage representation learning, as proposed by Grill et al. (2020), to improve adversarial generalization.

Semi- and self-supervised learning for adversarial robustness. Schmidt et al. (2018) showed that learning adversially robust models requires more data. As such, adversarial robustness with unlabeled data has recently drawn a lot of attention. We highlight the works by Uesato et al. (2019); Carmon et al. (2019); Zhai et al. (2019) which leveraged labeled data to train a standard classifier that is in turn used to pseudo-label the remaining unlabeled data. However, as shown by Uesato et al. (2019); Carmon et al. (2019); Zhai et al. (2019), when only  $10\%$  of the CIFAR-10 labels are available the robust accuracy drops significantly. In this paper, we focus on improving adversarial robustness in the low-label regime by leveraging unlabeled data (e.g., when  $1\% - 10\%$  of labels are available) to build robust representations. The result is a technique that significantly outperforms state-of-the-art pseudo-labeling techniques in the low-label regime and remains competitive with adversarial training when all labels are available.

# 3 METHOD

In this section, we explain BYORL which elegantly combines adversarial training with BYOL. Hence, we start by giving a brief description of adversarial training and BYOL.

# 3.1 ADVERSARIAL TRAINING

Madry et al. (2017) formulate a saddle point problem whose goal is to find model parameters  $\theta$  that minimize the adversarial risk:

$$
\mathbb {E} _ {(\boldsymbol {x}, y) \sim \mathcal {D}} \left[ \max  _ {\boldsymbol {\delta} \in \mathbb {S}} l (f (\boldsymbol {x} + \boldsymbol {\delta}; \boldsymbol {\theta}), y) \right] \tag {1}
$$

where  $\mathcal{D}$  is a data distribution over pairs of examples  $\pmb{x}$  and corresponding labels  $y$ ,  $f(\cdot; \pmb{\theta})$  is a model parametrized by  $\pmb{\theta}$ ,  $l$  is a suitable loss function (such as the  $0 - 1$  loss in the context of classification tasks), and  $\mathbb{S}$  defines the set of allowed perturbations (i.e., the adversarial input set or threat model).

Several methods (also known as "attacks") have been proposed to find adversarial examples (and effectively solve the inner maximization problem in Eq. 1). Classical adversarial training as proposed by Madry et al. (2017) uses Projected Gradient Descent (PGD),² which replaces the impractical  $0 - 1$  loss  $l$  with the cross-entropy loss  $\hat{l}$  and computes an adversarial perturbation  $\hat{\delta} = \delta^{(K)}$  in  $K$  gradient ascent steps of size  $\alpha$  as

$$
\boldsymbol {\delta} ^ {(k + 1)} \leftarrow \operatorname {p r o j} _ {\mathbb {S}} \left(\boldsymbol {\delta} ^ {(k)} + \alpha \nabla_ {\boldsymbol {\delta} ^ {(t)}} \hat {l} (f (\boldsymbol {x} + \boldsymbol {\delta} ^ {(k)}; \boldsymbol {\theta}), y)\right) \tag {2}
$$

![](images/f92ca9bf65667522e56d98d61a4fdc3a0954fa834066b5ffde75552c55557281.jpg)  
Figure 2: Flow diagram that highlights the difference between BYOL and BYORL. Whereas BYOL directly tries to maximize the cosine similarity between  $q(z; \theta)$  and  $z'$ , BYORL first executes an adversarial attack to retrieve an alternative image  $\hat{\pmb{v}}$ .

where  $\delta^{(0)}$  is chosen at random within  $\mathbb{S}$ , and where  $\mathrm{proj}_{\mathbb{A}}(\pmb{a})$  projects a point  $\pmb{a}$  back onto a set  $\mathbb{A}$ . Finally, for each example  $\pmb{x}$  with label  $y$ , adversarial training minimizes the loss given by

$$
\mathcal {L} _ {\boldsymbol {\theta}} ^ {\mathrm {A T}} = \hat {l} (f (\boldsymbol {x} + \hat {\boldsymbol {\delta}}; \boldsymbol {\theta}), y) \approx \max  _ {\boldsymbol {\delta} \in \mathbb {S}} \hat {l} (f (\boldsymbol {x} + \boldsymbol {\delta}; \boldsymbol {\theta}), y) \tag {3}
$$

where  $\hat{\delta}$  is given by Eq. 2 and  $\hat{l}$  is the softmax cross-entropy loss.

# 3.2 BOOTSTRAP YOUR OWN LATENTS

Many successful self-supervised learning approaches learn image representations by identifying whether different views belong to the same image (Dosovitskiy et al., 2014; Wu et al., 2018). Whereas contrastive methods formulate this prediction problem into one of discrimination (i.e., from the representation of an augmented view, they learn to discriminate between the representation of another augmented view of the same image, and the representations of augmented views of other images), BYOL relies on two neural networks: an online and a target network, that interact and learn from each other. The goal of the online network is to predict the target network representation of the same image under different augmented views, where the target network itself is defined by an exponential moving average of the online network parameters. We selected BYOL as the basis of our proposed method not only because it is currently the most successful representation learning technique, but also because it is more amenable to adversarial training, to contrary of contrastive methods which require large batch sizes (Chen et al., 2020; Oord et al., 2018) or memory banks He et al. (2020).

As shown in Fig. 2, the online network is composed of three stages: an encoder  $e(\cdot; \theta)$ , a projector  $g(\cdot; \theta)$  and a predictor  $q(\cdot; \theta)$ . Omitting the predictor, the target network has the same architecture as the online network, but uses a different set of weights  $\xi$ . As explained by Grill et al. (2020), in order to enhance representations while preventing their collapse, the target network's weights are allowed to change slowly throughout training. More precisely, given a decay rate  $\tau \in [0,1]$ , after each training step, the parameters  $\xi$  are updated as  $\xi \gets \tau \xi + (1 - \tau) \theta$ . Given an image  $\mathbf{x}$ , and two augmentations  $t, t' \sim \mathcal{T}$  sampled from a set of augmentations (e.g., random crops or recolorizations), BYOL produces two augmented views  $\mathbf{v} = t(\mathbf{x})$  and  $\mathbf{v}' = t'(x)$ . The first view passes through the online network, producing a representation  $h = e(\mathbf{x}; \theta)$  and a projection  $z = g(h; \theta)$ . The second

view similarly passes through the target network, producing a target projection  $z' = g \circ e(\pmb{v'}; \pmb{\xi})$ . Finally, given an online prediction  $q(\pmb{z}; \pmb{\theta})$  (which should be predictive of the target projection), BYOL minimizes the loss

$$
\mathcal {L} _ {\boldsymbol {\theta}} ^ {\mathrm {B Y O L}} = \left\| \frac {q (\boldsymbol {z} ; \boldsymbol {\theta})}{\| q (\boldsymbol {z} ; \boldsymbol {\theta}) \| _ {2}} - \frac {\boldsymbol {z} ^ {\prime}}{\| \boldsymbol {z} ^ {\prime} \| _ {2}} \right\| _ {2} ^ {2} = 2 - 2 \cdot \frac {\langle q (\boldsymbol {z} ; \boldsymbol {\theta}) , \boldsymbol {z} ^ {\prime} \rangle}{\| q (\boldsymbol {z} ; \boldsymbol {\theta}) \| _ {2} \cdot \| \boldsymbol {z} ^ {\prime} \| _ {2}}. \tag {4}
$$

At the end of training, everything but  $e$  and  $\pmb{\theta}$  is discarded and only the representation  $e(\pmb{x};\pmb{\theta})$  of an image  $\pmb{x}$  is used by downstream applications.

# 3.3 BOOTSTRAP YOUR OWN ROBUST LATENTS

We now introduce an effective and elegant approach to learn adversarially robust representations. Whereas previous (semi-)supervised techniques that are directly based on adversarial training require the presence of labels to produce adversarially robust neural networks (from which robust representations can be extracted at intermediate layers), our method can operate without any labels. As shown in the experimental section, the resulting representations can then be used to train linear classifiers (on top of these representation) that are intrinsically robust to adversaries. Our method, named Bootstrap Your Own Robust Latents or BYORL, consists of combining BYOL with adversarial training. A diagram summarizing BYORL is visible in Fig. 2.

For conciseness, we will denote by  $\gamma = g\circ e$  the composition of the encoder and projector and by  $\kappa = q\circ g\circ e$  the composition of the encoder, projector and predictor. Similarly to BYOL, BYORL starts by generating two views  $\pmb{v} = t(\pmb {x})$  and  $\pmb{v}^{\prime} = t^{\prime}(\pmb {x})$  of the same image  $\pmb{x}$ . While the second view goes through the target network unmodified to produce a target projection  $z^{\prime} = \gamma (v^{\prime};\xi)$ , the first view is further augmented via an adversarial attack. The goal of the adversarial attack is to maximize the disagreement between the online and target networks while respecting the threat model described by  $\mathbb{S}$  (see subsection 3.1). To this end, we would like to find an optimal perturbation  $\delta^{\star}\in \mathbb{S}$  that minimizes the resulting cosine similarity between the online prediction  $\kappa (\pmb {v} + \delta ;\pmb {\theta})$  and target projection  $z^{\prime}$ :

$$
\boldsymbol {\delta} ^ {\star} = \underset {\boldsymbol {\delta} \in \mathbb {S}} {\arg \min } \frac {\left\langle \kappa (\boldsymbol {v} + \boldsymbol {\delta} ; \boldsymbol {\theta}) , \boldsymbol {z} ^ {\prime} \right\rangle}{\| \kappa (\boldsymbol {v} + \boldsymbol {\delta} ; \boldsymbol {\theta}) \| _ {2} \cdot \| \boldsymbol {z} ^ {\prime} \| _ {2}}. \tag {5}
$$

Like adversarial training, we can leverage PGD to approximate  $\delta^{\star}$  by  $\hat{\delta}$ . Taking  $K$  steps of size  $\alpha$ , resulting in  $\hat{\delta} = \delta^{(K)}$ , we have

$$
\boldsymbol {\delta} ^ {(k + 1)} \leftarrow \operatorname {p r o j} _ {\mathbb {S}} \left(\boldsymbol {\delta} ^ {(k)} + \alpha \nabla_ {\boldsymbol {\delta} ^ {(t)}} \frac {\langle \kappa (\boldsymbol {v} + \boldsymbol {\delta} ^ {(t)} ; \boldsymbol {\theta}) , \boldsymbol {z} ^ {\prime} \rangle}{\| \kappa (\boldsymbol {v} + \boldsymbol {\delta} ^ {(t)} ; \boldsymbol {\theta}) \| _ {2} \cdot \| \boldsymbol {z} ^ {\prime} \| _ {2}}\right) \tag {6}
$$

where  $\delta^{(0)}$  is chosen at random within  $\mathbb{S}$ . Finally, we seek to maximize the agreement between the adversarially modified online prediction  $\kappa (\pmb {v} + \hat{\pmb{\delta}};\pmb {\theta})$  and the target projection  $z^{\prime}$  by updating the online weights  $\pmb{\theta}$  as to minimize the following loss:

$$
\begin{array}{l} \mathcal {L} _ {\boldsymbol {\theta}} ^ {\mathrm {B Y O R L}} = 2 - 2 \cdot \frac {\left\langle \kappa (\boldsymbol {v} + \hat {\delta} ; \boldsymbol {\theta}) , \boldsymbol {z} ^ {\prime} \right\rangle}{\| \kappa (\boldsymbol {v} + \hat {\delta} ; \boldsymbol {\theta}) \| _ {2} \cdot \| \boldsymbol {z} ^ {\prime} \| _ {2}} (7) \\ \approx 2 - 2 \cdot \min  _ {\boldsymbol {\delta} \in \mathbb {S}} \frac {\left\langle \kappa (\boldsymbol {v} + \boldsymbol {\delta} ; \boldsymbol {\theta}) , \boldsymbol {z} ^ {\prime} \right\rangle}{\| \kappa (\boldsymbol {v} + \boldsymbol {\delta} ; \boldsymbol {\theta}) \| _ {2} \cdot \| \boldsymbol {z} ^ {\prime} \| _ {2}}. (8) \\ \end{array}
$$

Here are a few additional considerations. First, we symmetrize the loss  $\mathcal{L}_{\theta}^{\mathrm{BYORL}}$  in Eq. 8 by feeding  $\pmb{v}'$  to the online network and  $\pmb{v}$  to the target network. The adversarial attack is executed on  $\pmb{v}'$  instead of  $\pmb{v}$  and tries to minimize the cosine similarity between the online prediction  $\kappa(\pmb{v}' + \delta; \pmb{\theta})$  and target projection  $\gamma(\pmb{v}; \pmb{\xi})$ . Second, one can observe that the adversarial attack is always performed through the online network. We could similarly perform the attack through the target network, but we found that training was less stable as batch statistics (needed by batch normalization) were not representative of statistics induced by adversarial examples (as the online network would receive clean rather than adversarial images). Third, instead of the proposed method, we could imagine making two passes through the online network (for both the clean and adversarial images) and maximizing the agreement between both online predictions (in addition to maximizing the agreement with the target projection). We found that this increased the risk of representation collapse as this adds an incentive for the online network to output constant predictions (i.e., collapsed representations are the perfect defense against adversarial attacks).

# 4 EXPERIMENTS

We assess the performance of BYORL across multiple axes. We evaluate the robustness of the resulting representations by training robust linear classifiers on top of these representations. First, we compare to the performance of various classifiers (comparing BYORL with pseudo-labeling). Second, we study how these robust representations transfer to unseen new tasks. Finally, we also evaluate whether robustness transfers to downstream tasks - even when the final task is not treated as being adversarial.

# 4.1 SETUP AND IMPLEMENTATION DETAILS

We highlight here the most important components and defer some of the details to Appendix A.

Architecture. We use a convolutional residual network (He et al., 2015) with 34 layers (Pre-Activation ResNet-34) as our encoder  $e$ . We also use wider (from  $\times 1$  to  $\times 4$ ) ResNets. The projector  $g$  and predictor  $q$  networks are MLPs with hidden dimension 4096 and output dimension 256.

Outer optimization. We use the LARS optimizer (You et al., 2017) with a cosine learning rate schedule (Loshchilov & Hutter, 2017) over 1000 epochs. We set the learning rate to 2 and use a global weight decay parameter of  $5 \cdot 10^{-4}$ . For the target network, the exponential moving average parameter  $\tau$  starts from 0.996 and is increased to one during training. We use a batch size of 512.

Inner optimization. The inner minimization in Eq. 8 is implemented using  $K$  PGD steps (constrained by an  $\ell_2$  or  $\ell_{\infty}$  norm-bounded ball). Unless specified otherwise, we set  $K$  to 40 and use an adaptive step size  $\alpha$  (see Algorithm 1 in Croce & Hein, 2020). For  $\ell_{\infty}$  and  $\ell_2$  norm-bounded perturbations, the gradients in Eq. 6 are first normalized to their sign or by their  $\ell_2$  norm, respectively.

Evaluation protocol. We evaluate the performance of BYORL on CIFAR-10 against adversarial  $\ell_{2}$  and  $\ell_{\infty}$  norm-bounded perturbations (CIFAR-100 andImagenet results are in the appendix). For that purpose, we train a linear classifier parametrized by coefficients  $W$  and offsets  $b$  on top of frozen BYORL representations, following the procedure described in Kolesnikov et al. (2019); Chen et al. (2020). The linear model is either trained in a non-robust manner (i.e.,  $\min_{W,b} \mathbb{E}_{(x,y) \in \mathcal{D}} \hat{l}(We(x;\theta) + b,y)$ ) or adversarially (i.e.,  $\min_{W,b} \mathbb{E}_{(x,y) \in \mathcal{D}} \max_{\delta \in \mathbb{S}} \hat{l}(We(x + \delta; \theta) + b,y)$ ). We then compute the accuracy of the combined model  $We(\cdot; \theta) + b$  against adversarial attacks (i.e., we count a successful attack as a misclassification). In order to get faithful results, all models are evaluated using a strong attack which combines elements of the AutoAttack procedure (Croce & Hein, 2020) with the MultiTargeted attack (Gowal et al., 2019). Namely, we use a sequence of AutoPGD on the cross-entropy loss with 5 restarts and 100 steps, AutoPGD on the difference of logits ratio loss with 5 restarts and 100 steps, MultiTargeted on the margin loss with 20 restarts and 200 steps and Square (Andriushchenko et al., 2019), an efficient black-box attack, with 5000 queries.

Baseline. Throughout the experimental section, we compare BYORL with adversarial training (combined with pseudo-labeling to handle missing labels). Pseudo-labeling is currently, to the best of our knowledge, the most successful semi-supervised method for learning adversariably robust models (Carmon et al., 2019; Uesato et al., 2019; Zhai et al., 2019; Najafi et al., 2019). More specifically, we use Unsupervised Adversarial Training with Fixed Targets (UAT-FT) (Uesato et al., 2019). When  $100\%$  of the labels are available UAT-FT is equivalent to classical adversarial training, as proposed by Madry et al. (2017). In settings where less than  $100\%$  of the labels are available, we train a separate non-robust model (with an architecture identical to the robust model being trained) on the available labeled data and use it to pseudo-label the rest of the unlabeled images. UAT-FT uses the same network architectures than those used by BYORL.

# 4.2 RESULTS

Robustness on CIFAR-10. We evaluate BYORL and UAT-FT on a wide range of tasks across different threats for various amounts of available labels. As is typical in the literature (Rice et al., 2020; Augustin et al., 2020), we evaluate our models on CIFAR-10 against  $\ell_2$  and  $\ell_{\infty}$  norm-bounded perturbations of size  $\epsilon = 128 / 255$  and  $\epsilon = 8 / 255$  (CIFAR-100 and IMAGENET are evaluated in the appendix). CIFAR-10 contains 60K images (i.e., 50K in the train set and 10K in the test set). As such,

![](images/d759f8af0bc23d02ab4a22938d380a9cf358f10799fc27bf4d2befa116f7943a.jpg)  
(a) CIFAR-10 only

![](images/d4a6dc0c275322e801860b6c3b1ffd6ea7587dafa8ae05ee37021977189b3466.jpg)  
(b) CIFAR-10 and 80M-TINYIMAGES

![](images/b566c46518d414f6f5162f35bd421bc7df5240b016a97807644e18c554273040.jpg)  
Figure 3: Accuracy under  $\ell_2$  attack of size  $\epsilon = 128 / 255$  for different CIFAR-10 models as a function of the ratio of available labels. Panel a restricts the available data to CIFAR-10 only (labeled and unlabeled), while panel b uses  $500\mathrm{K}$  additional unlabeled images extracted from 80M-TINYIMAGES.  
(a) CIFAR-10 only

![](images/d71e37bf3478cb4a09f951a7aab8cbf90c04c8eb5976171c19434dce5595ecec.jpg)  
Figure 4: Accuracy under  $\ell_{\infty}$  attack of size  $\epsilon = 8 / 255$  for different CIFAR-10 models as a function of the ratio of available labels. Panel a restricts the available data to CIFAR-10 only (labeled and unlabeled), while panel b uses  $500\mathrm{K}$  additional unlabeled images extracted from 80M-TINYIMAGES.  
(b) CIFAR-10 and 80M-TINYIMAGES

when we evaluate on  $1\%$  of labeled data, we only use 500 random labeled images from CIFAR-10 (we do not artificially balance the number of labels per class). As done in Carmon et al. (2019) and Uesato et al. (2019), we also explore the limits of BYORL in the setting where additional unlabeled data is available. This additional data is extracted from 80M-TINYIMAGES and consists of 500K unlabeled  $32 \times 32$  images<sup>3</sup>. In settings without this additional data, we use a ResNet-34  $\times 2$ , whereas in settings with this additional data, we use a ResNet-34  $\times 4$ .

Fig. 3 shows the robust accuracy of BYORL and UAT-FT on the full CIFAR-10 test set against  $\ell_2$  norm-bounded perturbations. We observe that linear classifiers trained on top of robust BYORL representations are more robust than those trained with UAT-FT. In particular, we highlight that when only 500 labeled images are available, BYORL remains competitive with state-of-the-art methods that use all labels: without additional data from 80M-TINYIMAGES, BYORL reaches  $65.43\%$  compared to  $69.24\%$  (Engstrom et al., 2019); with additional data from 80M-TINYIMAGES, BYORL reaches  $75.50\%$  compared to  $72.91\%$  (Augustin et al., 2020).

Fig. 4 shows the robust accuracy of BYORL and UAT-FT on the full CIFAR-10 test set against  $\ell_{\infty}$  norm-bounded perturbations. Again, we can observe that BYORL remains competitive: in the low-label regime, BYORL surpasses UAT-FT by a significant margin (up to  $2\times$  more accurate); in the high-label regime BYORL loses a few percentage points. Perhaps surprisingly, under both threat models (i.e.,  $\ell_{\infty}$  and  $\ell_2$ ), BYORL already reaches its optimal performance when  $5\%$  of the labels are available.

Transfer to unseen tasks. We evaluate our robust representations on other classification datasets to assess whether the features learned on CIFAR-10 are generic and thus useful across image domains, or if they are CIFAR-10-specific. As a comparison, we test whether pre-logits activations resulting

Table 1: Robust accuracy (under adversarial attack) obtained by finetuning a linear head on top of robust representations trained on CIFAR-10.  

<table><tr><td rowspan="2">METHOD</td><td rowspan="2">NORM</td><td rowspan="2">RADIUS</td><td colspan="3">STL-10</td><td colspan="3">CIFAR-100</td></tr><tr><td>1%</td><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td></tr><tr><td>BYORL</td><td rowspan="2">\( \ell_2 \)</td><td rowspan="2">\( \epsilon = 128/255 \)</td><td>33.85%</td><td>53.23%</td><td>57.88%</td><td>10.09%</td><td>22.51%</td><td>28.24%</td></tr><tr><td>UAT-FT</td><td>37.71%</td><td>42.16%</td><td>53.23%</td><td>3.68%</td><td>10.80%</td><td>14.43%</td></tr><tr><td>BYORL</td><td rowspan="2">\( \ell_\infty \)</td><td rowspan="2">\( \epsilon = 8/255 \)</td><td>24.18%</td><td>36.30%</td><td>37.28%</td><td>5.28%</td><td>10.12%</td><td>14.82%</td></tr><tr><td>UAT-FT</td><td>23.47%</td><td>36.52%</td><td>37.79%</td><td>2.18%</td><td>4.88%</td><td>7.21%</td></tr></table>

Table 2: Clean (no perturbations) and robust (under adversarial attack) accuracy obtained when training robust and non-robust representations on CIFAR-10 against  $\ell_2$  norm-bounded perturbations of size  $\epsilon = 128 / 255$ . We evaluate the representations by training/finetuning a robust and non-robust linear head on CIFAR-10, STL-10 and CIFAR-100.  

<table><tr><td rowspan="2">TRAINING OF 
REPRESENTATION</td><td rowspan="2">LINEAR HEAD</td><td rowspan="2">NORM</td><td rowspan="2">RADIUS</td><td colspan="2">CIFAR-10</td></tr><tr><td>Clean</td><td>Robust</td></tr><tr><td>Robust (BYORL)</td><td>Robust (AT) on CIFAR-10</td><td></td><td></td><td>93.01%</td><td>77.61%</td></tr><tr><td>Robust (BYORL)</td><td>Non-robust on CIFAR-10</td><td>l2</td><td>ε = 128/255</td><td>93.19%</td><td>77.09%</td></tr><tr><td>Non-robust (BYOL)</td><td>Robust (AT) on CIFAR-10</td><td></td><td></td><td>91.23%</td><td>0.05%</td></tr><tr><td>Non-robust (BYOL)</td><td>Non-robust on CIFAR-10</td><td></td><td></td><td>94.76%</td><td>0.00%</td></tr><tr><td colspan="2">FINETUNING OF LINEAR HEAD</td><td></td><td></td><td colspan="2">STL-10</td></tr><tr><td>Robust (BYORL)</td><td>Robust (AT) on STL-10</td><td>l2</td><td>ε = 128/255</td><td>76.49%</td><td>57.88%</td></tr><tr><td>Robust (BYORL)</td><td>Non-robust on STL-10</td><td></td><td></td><td>77.54%</td><td>57.66%</td></tr><tr><td colspan="2">FINETUNING OF LINEAR HEAD</td><td></td><td></td><td colspan="2">CIFAR-100</td></tr><tr><td>Robust (BYORL)</td><td>Robust (AT) on CIFAR-100</td><td>l2</td><td>ε = 128/255</td><td>48.34%</td><td>28.24%</td></tr><tr><td>Robust (BYORL)</td><td>Non-robust on CIFAR-100</td><td></td><td></td><td>49.20%</td><td>27.17%</td></tr></table>

from training a model using UAT-FT can also be used for transfer learning. For both representations, we train a robust linear model using adversarial training (see subsection 3.1) with different label availability on STL-10 and CIFAR-100 against  $\ell_{\infty}$  and  $\ell_{2}$  norm-bounded perturbations. Table 1 shows that BYORL representations result in equivalent or more robust models than UAT-FT representations. We note, however, that – at least on CIFAR-100 – the robust accuracy remains significantly lower than models trained directly on CIFAR-100.

Transfer without adversarial training. So far, the linear classifiers trained on top of BYORL representations were trained robustly using adversarial training. We now evaluate whether adversarial training is needed for downstream tasks. Conversely, we also verify that learning robust representations is needed to obtain robust linear classifiers. Table 2 shows the robust accuracy of four models: (i) an adversarially trained linear model on top of robust BYORL representations, (ii) a classically trained (not necessarily robust) linear model on top of robust BYORL representations, (iii) an adversarially trained linear model on top of non-robust BYOL representations, and (iv) a classically trained linear model on top of non-robust BYOL representations. Although not a guarantee in theory (Allen-Zhu & Li, 2020), we observe that the adversarial training of the linear classifier is, in practice, not necessary and that it is enough to train robust representations to obtain robust classifiers. Indeed, for all three considered downstream tasks (CIFAR-10, STL-10 and CIFAR-100), the resulting non-robustly trained linear classifiers are within a few percentage points of the robustly trained ones.

# 5 CONCLUSION

In this work, we present BYORL, a modification of BYOL that enables us to train robust image representations. To the contrary of previous methods, BYORL does not require the presence of label information. In fact, it is even possible to use these robust representations to train adversarially robust classifiers on multiple downstream tasks (without the need to use adversarial training). Interestingly, classifiers using BYORL representations can be trained with as little as 500 labeled examples. Across all experiments, BYORL with  $1\%$  of labels (i.e., 500 labeled examples) matches or surpasses the performance of pseudo-labeling (implemented through UAT-FT) with  $10 - 20\%$  of labels (i.e., between 5K and 10K labeled examples).

# REFERENCES

Zeyuan Allen-Zhu and Yuanzhi Li. Feature purification: How adversarial training performs robust deep learning. arXiv preprint arXiv:2005.10190, 2020. URL https://arxiv.org/pdf/2005.10190.  
Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, and Matthias Hein. Square Attack: a query-efficient black-box adversarial attack via random search. arXiv preprint arXiv:1912.00049, 2019. URL https://arxiv.org/pdf/1912.00049.  
Anish Athalye and Ilya Sutskever. Synthesizing robust adversarial examples. arXiv preprint arXiv:1707.07397, 2017.  
Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. arXiv preprint arXiv:1802.00420, 2018. URL https://arxiv.org/pdf/1802.00420.  
Maximilian Augustin, Alexander Meinke, and Matthias Hein. Adversarial Robustness on In-and Out-Distribution Improves Explainability. arXiv preprint arXiv:2003.09461, 2020. URL https://arxiv.org/pdf/2003.09461.  
Philip Bachman, Ouais Alsharif, and Doina Precup. Learning with pseudo-ensembles. In NeurIPS, 2014.  
David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin Raffel. MixMatch: A Holistic Approach to Semi-Supervised Learning. arXiv:1905.02249, 2019.  
Battista Biggio, Igino Corona, Davide Maiorca, Blaine Nelson, Nedim Šrndić, Pavel Laskov, Giorgio Giacinto, and Fabio Roli. Evasion attacks against machine learning at test time. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 387-402. Springer, 2013.  
Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, and Karol Zieba. End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316, 2016. URL https://arxiv.org/pdf/1604.07316.  
Nicholas Carlini and David Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods. In Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security, pp. 3-14. ACM, 2017a.  
Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy, pp. 39-57. IEEE, 2017b.  
Yair Carmon, Aditi Raghunathan, Ludwig Schmidt, John C Duchi, and Percy S Liang. Unlabeled data improves adversarial robustness. In Advances in Neural Information Processing Systems, pp. 11190-11201, 2019. URL https://papers.nips.cc/paper/9298-unlabeled-data-improves-adversarial-robustness.pdf.  
Taylan Cemgil, Sumedh Ghaisas, Krishnamurthy (Dj) Dvijotham, and Pushmeet Kohli. Adversarial Robust Representations with Smooth Encoders. In International Conference on Learning Representations, 2019. URL https://openreview.net/pdf?id=H1gfFaEYDS.  
Olivier Chapelle, Bernhard Scholkopf, and Alexander Zien. Semi-Supervised Learning. MITPress, 2009.  
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709, 2020. URL https://arxiv.org/pdf/2002.05709.  
Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. Proceedings of Machine Learning Research, pp. 215-223. JMLR Workshop and Conference Proceedings, 2011. URL http://proceedings.mlr.press/v15/coates11a.html.  
Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for YouTube recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems, 2016.

Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. arXiv preprint arXiv:2003.01690, 2020. URL https://arxiv.org/pdf/2003.01690.  
Jeffrey De Fauw, Joseph R Ledsam, Bernardino Romera-Paredes, Stanislav Nikolov, Nenad Tomasev, Sam Blackwell, Harry Askham, Xavier Glorot, Brendan O'Donoghue, Daniel Visentin, George van den Driessche, Balaji Lakshminarayanan, Clemens Meyer, Faith Mackinder, Simon Bouton, Kareem Ayoub, Reena Chopra, Dominic King, Alan Karthikesalingam, Cfan O Hughes, Rosalind Raine, Julian Hughes, Dawn A Sim, Catherine Egan, Adnan Tufail, Hugh Montgomery, Demis Hassabis, Geraint Rees, Trevor Back, Peng T Khaw, Mustafa Suleyman, Julien Cornebise, Pearse A Keane, and Olaf Ronneberger. Clinically applicable deep learning for diagnosis and referral in retinal disease. In Nature Medicine, 2018. URL https://www.nature.com/articles/s41591-018-0107-6.pdf.  
Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In Computer Vision and Pattern Recognition, 2015.  
Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, and Jianguo Li. Boosting Adversarial Attacks with Momentum. arXiv preprint arXiv:1710.06081, 2017. URL https://arxiv.org/pdf/1710.06081.  
Alexey Dosovitskiy, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with convolutional neural networks. In Neural Information Processing Systems, 2014.  
Logan Engstrom, Andrew Ilyas, Hadi Salman, Shibani Santurkar, and Dimitris Tsipras. Robustness (python library), 2019. URL https://github.com/MadryLab/robustness.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Sven Gowal, Jonathan Uesato, Chongli Qin, Po-Sen Huang, Timothy Mann, and Pushmeet Kohli. An Alternative Surrogate Loss for PGD-based Adversarial Testing. arXiv preprint arXiv:1910.09338, 2019. URL https://arxiv.org/pdf/1910.09338.  
Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. arXiv preprint arXiv:2006.07733, 2020. URL https://arxiv.org/pdf/2006.07733.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015. URL https://arxiv.org/pdf/1512.03385.  
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729-9738, 2020.  
Geoffrey E Hinton, Simon Osindero, and Yee-Whye Teh. A fast learning algorithm for deep belief nets. Neural computation, 18(7):1527-1554, 2006.  
Lang Huang, Chao Zhang, and Hongyang Zhang. Self-Adaptive Training: beyond Empirical Risk Minimization. arXiv preprint arXiv:2002.10319, 2020. URL https://arxiv.org/pdf/2002.10319.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, 2015.  
Harini Kannan, Alexey Kurakin, and Ian Goodfellow. Adversarial Logit Pairing. arXiv preprint arXiv:1803.06373, 2018.  
Minseon Kim, Jihoon Tack, and Sung Ju Hwang. Adversarial Self-Supervised Contrastive Learning. arXiv preprint arXiv:2006.07589, 2020. URL https://arxiv.org/pdf/2006.07589.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Alexander Kolesnikov, Xiaohua Zhai, and Lucas Beyer. Revisiting self-supervised visual representation learning. In Computer Vision and Pattern Recognition, 2019.  
Simon Kornblith, Jonathon Shlens, and Quoc V Le. Do better ImageNet models transfer better? In Computer Vision and Pattern Recognition, 2019.  
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The CIFAR-10 dataset. 2014. URL http://www.cs.toronto.edu/kriz/cifar.html.  
Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial examples in the physical world. arXiv preprint arXiv:1607.02533, 2016. URL https://arxiv.org/pdf/1607.02533.  
Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learnings. In ICLR, 2017.  
Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts. In International Conference on Learning Representations, 2017.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.  
Takeru Miyato, Shin ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning. TPAMI, 2018.  
Marius Mosbach, Maksym Andriushchenko, Thomas Trost, Matthias Hein, and Dietrich Klakow. Logit Pairing Methods Can Fool Gradient-Based Attacks. arXiv preprint arXiv:1810.12042, 2018. URL https://arxiv.org/pdf/1810.12042.  
Vinod Nair and Geoffrey E. Hinton. Rectified linear units improve restricted boltzmann machines. In International Conference on Machine Learning, 2010.  
Amir Najafi, Shin-ichi Maeda, Masanori Koyama, and Takeru Miyato. Robustness to adversarial perturbations in learning from incomplete data. arXiv preprint arXiv:1905.13021, 2019. URL https://arxiv.org/pdf/1905.13021.  
Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In European Conference on Computer Vision, 2016.  
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
Tianyu Pang, Xiao Yang, Yinpeng Dong, Kun Xu, Hang Su, and Jun Zhu. Boosting Adversarial Training with Hypersphere Embedding. arXiv preprint arXiv:2002.08619, 2020. URL https://arxiv.org/pdf/2002.08619.  
Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, and Ananthram Swami. Distillation as a defense to adversarial perturbations against deep neural networks. arXiv preprint arXiv:1511.04508, 2015.  
Chongli Qin, James Martens, Sven Gowal, Dilip Krishnan, Krishnamurthy Dvijotham, Alhussein Fawzi, Soham De, Robert Stanforth, and Pushmeet Kohli. Adversarial Robustness through Local Linearization. arXiv preprint arXiv:1907.02610, 2019. URL https://arxiv.org/pdf/1907.02610.  
Leslie Rice, Eric Wong, and J. Zico Kolter. Overfitting in adversarially robust deep learning. arXiv preprint arXiv:2002.11569, 2020. URL https://arxiv.org/pdf/2002.11569.  
Mehdi Sajjadi, Mehran Javanmardi, and Tolga Tasdizen. Regularization with stochastic transformations and perturbations for deep semi-supervised learning. In NeurIPS, 2016.  
Ludwig Schmidt, Shibani Santurkar, Dimitris Tsipras, Kunal Talwar, and Aleksander Madry. Adversarially Robust Generalization Requires More Data. In Advances in Neural Information Processing Systems. 2018. URL http://papers.nips.cc/paper/7749-adversarially-robust-generalization-requires-more-data.pdf.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.

Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning. arXiv preprint arXiv:2005.10243, 2020.  
Antonio Torralba, Rob Fergus, and William T. Freeman. 80 million tiny images: a large dataset for non-parametric object and scene recognition. TPAMI, 2008.  
Jonathan Uesato, Brendan O'Donoghue, Aaron van den Oord, and Pushmeet Kohli. Adversarial Risk and the Dangers of Evaluating Against Weak Attacks. arXiv preprint arXiv:1802.05666, 2018.  
Jonathan Uesato, Jean-Baptiste Alayrac, Po-Sen Huang, Robert Stanforth, Alhussein Fawzi, and Pushmeet Kohli. Are labels required for improving adversarial robustness? arXiv preprint arXiv:1905.13725, 2019. URL https://arxiv.org/pdf/1905.13725.  
Laurenz Wiskott and Terrence J Sejnowski. Slow feature analysis: Unsupervised learning of invariances. Neural Computation, 14(4):715-770, 2002.  
Eric Wong, Leslie Rice, and J. Zico Kolter. Fast is better than free: Revisiting adversarial training. arXiv preprint arXiv:2001.03994, 2020. URL https://arxiv.org/pdf/2001.03994.  
Zhirong Wu, Yuanjun Xiong, Stella Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance-level discrimination. arXiv preprint arXiv:1805.01978, 2018. URL https://arxiv.org/pdf/1805.01978.  
Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V. Le. Unsupervised Data Augmentation. arXiv:1904.12848, 2019.  
Yang You, Igor Gitman, and Boris Ginsburg. Scaling SGD batch size to 32k for ImageNet training. arXiv preprint arXiv:1708.03888, 2017.  
Runtian Zhai, Tianle Cai, Di He, Chen Dan, Kun He, John Hopcroft, and Liwei Wang. Adversarily Robust Generalization Just Requires More Unlabeled Data. arXiv preprint arXiv:1906.00555, 2019. URL https://arxiv.org/pdf/1906.00555.  
Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, and Michael I. Jordan. Theoretically Principled Trade-off between Robustness and Accuracy. arXiv preprint arXiv:1901.08573, 2019. URL https://arxiv.org/pdf/1901.08573.  
Richard Zhang, Phillip Isola, and Alexei A. Efros. Colorful image colorization. In European Conference on Computer Vision, 2016.  
Daniel Zoran, Mike Chrzanowski, Po-Sen Huang, Sven Gowal, Alex Mott, and Pushmeet Kohli. Towards robust image classification using sequential attention models. In CVPR, 2020.
