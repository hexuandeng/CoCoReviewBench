# SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

One of the challenges in the study of generative adversarial networks is the instability of its training. In this paper, we propose a novel weight normalization technique called spectral normalization to stabilize the training of the discriminator. Our new normalization technique is computationally light and easy to incorporate into existing implementations. We tested the efficacy of spectral normalization on CIFAR10, STL-10, and ILSVRC2012 dataset, and we experimentally confirmed that spectrally normalized GANs (SN-GANs) is capable of generating images of better or equal quality relative to the previous training stabilization techniques.

# 1 INTRODUCTION

Generative adversarial networks (GANs) (Goodfellow et al., 2014) have been enjoying considerable success as a framework of generative models in recent years, and it has been applied to numerous types of tasks and datasets (Radford et al., 2016; Salimans et al., 2016; Ho & Ermon, 2016; Li et al., 2017). In a nutshell, GANs are a framework to produce a model distribution that mimics a given target distribution, and it consists of a generator that produces the model distribution and a discriminator that distinguishes the model distribution from the target. The concept is to consecutively train the model distribution and the discriminator in turn, with the goal of reducing the difference between the model distribution and the target distribution measured by the best discriminator possible at each step of the training. GANs have been drawing attention in the machine learning community not only for its ability to learn highly structured probability distribution but also for its theoretically interesting aspects. For example, (Nowozin et al., 2016; Uehara et al., 2016; Mohamed & Lakshminarayanan, 2017) revealed that the training of the discriminator amounts to the training of a good estimator for the density ratio between the model distribution and the target. This is a perspective that opens the door to the methods of implicit models (Mohamed & Lakshminarayanan, 2017; Tran et al., 2017) that can be used to carry out variational optimization without the direct knowledge of the density function.

A persisting challenge in the training of GANs is the performance control of the discriminator. In high dimensional spaces, the density ratio estimation by the discriminator is often inaccurate and unstable during the training, and generator networks fail to learn the multimodal structure of the target distribution. Even worse, when the support of the model distribution and the support of the target distribution are disjoint, there exists a discriminator that can perfectly distinguish the model distribution from the target (Arjovsky & Bottou, 2017). Once such discriminator is produced in this situation, the training of the generator comes to complete stop, because the derivative of the so-produced discriminator with respect to the input turns out to be 0. This motivates us to introduce some form of restriction to the choice of the discriminator.

In this paper, we propose a novel weight normalization method called spectral normalization that can stabilize the training of discriminator networks. Our normalization enjoys following favorable properties.

- Lipschitz constant is the only hyper-parameter to be tuned, and the algorithm does not require intensive tuning of the only hyper-parameter for satisfactory performance.  
- Implementation is simple and the additional computational cost is small.

In fact, our normalization method also functioned well even without tuning Lipschitz constant, which is the only hyper parameter. In this study, we provide explanations of the effectiveness of

spectral normalization for GANs against other regularization techniques, such as weight normalization (Salimans & Kingma, 2016), weight clipping (Arjovsky et al., 2017), and gradient penalty (Gulrajani et al., 2017). We also show that, in the absence of complimentary regularization techniques (e.g., batch normalization, weight decay and feature matching on the discriminator), spectral normalization can improve the sheer quality of the generated images better than weight normalization and gradient penalty.

# 2 METHOD

In this section, we will lay the theoretical groundwork for our proposed method. Let us consider a simple discriminator made of a neural network of the following form:

$$
f (\boldsymbol {x}, \theta) = W ^ {L + 1} a _ {L} \left(W ^ {L} \left(a _ {L - 1} \left(W ^ {L - 1} \left(\dots a _ {1} \left(W ^ {1} \boldsymbol {x}\right) \dots\right)\right)\right)\right), \tag {1}
$$

where  $\theta := \{W^1, \ldots, W^L, W^{L+1}\}$  is the learning parameters set,  $W^l \in \mathbb{R}^{d_l \times d_{l-1}}$ ,  $W^{L+1} \in \mathbb{R}^{1 \times d_L}$ , and  $a_l$  is an element-wise non-linear activation function. We omit the bias term of each layer for simplicity. The final output of the discriminator is given by

$$
D (\boldsymbol {x}, \theta) = \mathcal {A} (f (\boldsymbol {x}, \theta)), \tag {2}
$$

where  $\mathcal{A}$  is an activation function corresponding to the divergence of distance measure of the user's choice. The standard formulation of GANs is given by

$$
\min _ {G} \max _ {D} V (G, D)
$$

where min and max of  $G$  and  $D$  are taken over the set of generator and discriminator functions, respectively. The conventional form of  $V(G,D)$  (Goodfellow et al., 2014) is given by  $\operatorname{E}_{\boldsymbol{x} \sim q_{\mathrm{data}}}[\log D(\boldsymbol{x})] + \operatorname{E}_{\boldsymbol{x}' \sim p_G}[\log (1 - D(\boldsymbol{x}'))]$ , where  $q_{\mathrm{data}}$  is the data distribution and  $p_G$  is the (model) generator distribution to be learned through the adversarial min-max optimization. The activation function  $\mathcal{A}$  that is used in the  $D$  of this expression is some continuous function with range [0, 1] (e.g., sigmoid function). It is known that, for a fixed generator  $G$ , the optimal discriminator for this form of  $V(G,D)$  is given by  $D_G^*(\boldsymbol{x}) \coloneqq q_{\mathrm{data}}(\boldsymbol{x}) / (q_{\mathrm{data}}(\boldsymbol{x}) + p_G(\boldsymbol{x}))$ .

The machine learning community has been pointing out recently that the function space from which the discriminators are selected crucially affects the performance of GANs. A number of works (Uehara et al., 2016; Qi, 2017; Gulrajani et al., 2017) advocate the importance of Lipschitz continuity in assuring the boundedness of statistics. For example, the optimal discriminator of GANs on the above standard formulation takes the form

$$
D _ {G} ^ {*} (\boldsymbol {x}) = \frac {q _ {\mathrm {d a t a}} (\boldsymbol {x})}{q _ {\mathrm {d a t a}} (\boldsymbol {x}) + p _ {G} (\boldsymbol {x})} = \operatorname {s i g m o i d} \left(f ^ {*} (\boldsymbol {x})\right), \text {w h e r e} f ^ {*} (\boldsymbol {x}) = \log q _ {\mathrm {d a t a}} (\boldsymbol {x}) - \log p _ {G} (\boldsymbol {x}), \tag {3}
$$

and its derivative

$$
\nabla_ {\boldsymbol {x}} f ^ {*} (\boldsymbol {x}) = \frac {1}{q _ {\mathrm {d a t a}} (\boldsymbol {x})} \nabla_ {\boldsymbol {x}} q _ {\mathrm {d a t a}} (\boldsymbol {x}) - \frac {1}{p _ {G} (\boldsymbol {x})} \nabla_ {\boldsymbol {x}} p _ {G} (\boldsymbol {x}) \tag {4}
$$

can be unbounded or even incomputable. This prompts us to introduce some regularity condition to the derivative of  $f(x)$ .

A particularly successful works in this array are (Qi, 2017; Arjovsky et al., 2017; Gulrajani et al., 2017), which proposed methods to control the Lipschitz constant of the discriminator by adding regularization terms defined on input examples  $\pmb{x}$ . We would follow their footsteps and search for the discriminator  $D$  from the set of  $K$ -Lipschitz continuous functions, that is,

$$
\underset {\| f \| _ {\mathrm {L i p}} \leq K} {\arg \max } V (G, D), \tag {5}
$$

where we mean by  $\| f\|_{\mathrm{Lip}}$  the smallest value  $M$  such that  $\| f(\pmb {x}) - f(\pmb{x}^{\prime})\| /\| \pmb {x} - \pmb{x}^{\prime}\| \leq M$  for any  $\pmb {x},\pmb{x}^{\prime}$ , with the norm being the  $\ell_2$  norm.

While input based regularizations allow for relatively easy formulations based on samples, they also suffer from the fact that, they cannot impose regularization on the space outside of the supports of the generator and data distributions without introducing somewhat heuristic means. A method we would introduce in this paper, called spectral normalization, is a method that aims to skirt this issue by normalizing the weight matrices using the technique devised by Yoshida & Miyato (2017).

# 2.1 SPECTRAL NORMALIZATION

Our spectral normalization controls the Lipschitz constant of the discriminator function  $f$  by literally constraining the spectral norm of each layer  $g: h_{in} \mapsto h_{out}$ . By definition, Lipschitz norm  $\| g \|_{\mathrm{Lip}}$  is equal to  $\sup_h \sigma(\nabla g(h))$ , where  $\sigma(A)$  is the spectral norm of the matrix  $A$  ( $L_2$  matrix norm of  $A$ )

$$
\sigma (A) := \max  _ {\boldsymbol {h}: \boldsymbol {h} \neq \boldsymbol {0}} \frac {\| A \boldsymbol {h} \| _ {2}}{\| \boldsymbol {h} \| _ {2}} = \max  _ {\| \boldsymbol {h} \| _ {2} \leq 1} \| A \boldsymbol {h} \| _ {2}, \tag {6}
$$

which is equivalent to the largest singular value of  $A$ . Therefore, for a linear layer  $g(h) = Wh$ , the norm is given by  $\| g\|_{\mathrm{Lip}} = \sup_h\sigma (\nabla g(h)) = \sup_h\sigma (W) = \sigma (W)$ . If the Lipschitz norm of the activation function  $\| a_l\|_{\mathrm{Lip}}$  is equal to  $1^1$ , we can use the inequality  $\| g_1\circ g_2\|_{\mathrm{Lip}}\leq \| g_1\|_{\mathrm{Lip}}\cdot \| g_2\|_{\mathrm{Lip}}$  to observe the following bound on  $\| f\|_{\mathrm{Lip}}$ :

$$
\begin{array}{l} \| f \| _ {\operatorname {L i p}} \leq \| \left(\boldsymbol {h} _ {L} \mapsto W ^ {L + 1} \boldsymbol {h} _ {L}\right) \| _ {\operatorname {L i p}} \cdot \| a _ {L} \| _ {\operatorname {L i p}} \cdot \| \left(\boldsymbol {h} _ {L - 1} \mapsto W ^ {L} \boldsymbol {h} _ {L - 1}\right) \| _ {\operatorname {L i p}} \\ \dots \| a _ {1} \| _ {\mathrm {L i p}} \cdot \| \left(\boldsymbol {h} _ {0} \mapsto W ^ {1} \boldsymbol {h} _ {0}\right) \| _ {\mathrm {L i p}} = \prod_ {l = 1} ^ {L + 1} \| \left(\boldsymbol {h} _ {l - 1} \mapsto W ^ {l} \boldsymbol {h} _ {l - 1}\right) \| _ {\mathrm {L i p}} = \prod_ {l = 1} ^ {L + 1} \sigma (W ^ {l}). \tag {7} \\ \end{array}
$$

Our spectral normalization normalizes the spectral norm of the weight matrix  $W$  so that it satisfies the Lipschitz constraint  $\sigma(W) = 1$ :

$$
\bar {W} _ {\mathrm {S N}} (W) := W / \sigma (W). \tag {8}
$$

If we normalize each  $W^l$  using (8), we can appeal to the inequality (7) and the fact that  $\sigma(\bar{W}_{\mathrm{SN}}(W)) = 1$  to see that  $\| f \|_{\mathrm{Lip}}$  is bounded from above by 1.

# 2.2 FAST APPROXIMATION OF THE SPECTRAL NORM  $\sigma (W)$

As we mentioned above, the spectral norm  $\sigma(W)$  that we use to regularize each layer of the discriminator is the largest singular value of  $W$ . If we naively apply singular value decomposition to compute the  $\sigma(W)$  at each round of the algorithm, the algorithm can become computationally heavy. Instead, we can use the power iteration method to estimate  $\sigma(W)$  (Yoshida & Miyato, 2017). With power iteration method, we can estimate the spectral norm with very small additional computational time relative to the full computational cost of the vanilla GANs. Please see Appendix A for the detail method and Algorithm 1 for the summary of the actual spectral normalization algorithm.

# 2.3 GRADIENT ANALYSIS OF THE SPECTRALLY NORMALIZED WEIGHTS

The gradient of  $\bar{W}_{\mathrm{SN}}(W)$  with respect to  $W_{ij}$  is:

$$
\begin{array}{l} \frac {\partial \bar {W} _ {\mathrm {S N}} (W)}{\partial W _ {i j}} = \frac {1}{\sigma (W)} E _ {i j} - \frac {1}{\sigma (W) ^ {2}} \frac {\partial \sigma (W)}{\partial W _ {i j}} W = \frac {1}{\sigma (W)} E _ {i j} - \frac {\left[ \boldsymbol {u} _ {1} \boldsymbol {v} _ {1} ^ {\mathrm {T}} \right] _ {i j}}{\sigma (W) ^ {2}} W (9) \\ = \frac {1}{\sigma (W)} \left(E _ {i j} - \left[ \boldsymbol {u} _ {1} \boldsymbol {v} _ {1} ^ {\mathrm {T}} \right] _ {i j} \bar {W} _ {\mathrm {S N}}\right), (10) \\ \end{array}
$$

where  $E_{ij}$  is the matrix whose  $(i,j)$ -th entry is 1 and zero everywhere else, and  $\pmb{u}_1$  and  $\pmb{v}_1$  are respectively the first left and right singular vectors of  $W$ . If  $\pmb{h}$  is the hidden layer in the network to be transformed by  $\tilde{W}_{SN}$ , the derivative of the  $V(G,D)$  calculated over the mini-batch with respect to  $W$  of the discriminator  $D$  is given by:

$$
\begin{array}{l} \frac {\partial V (G , D)}{\partial W} = \frac {1}{\sigma (W)} \left(\hat {\mathrm {E}} \left[ \boldsymbol {\delta} \boldsymbol {h} ^ {\mathrm {T}} \right] - \left(\hat {\mathrm {E}} \left[ \boldsymbol {\delta} ^ {\mathrm {T}} \bar {W} _ {\mathrm {S N}} \boldsymbol {h} \right]\right) \boldsymbol {u} _ {1} \boldsymbol {v} _ {1} ^ {\mathrm {T}}\right) (11) \\ = \frac {1}{\sigma (W)} \left(\hat {\mathrm {E}} \left[ \boldsymbol {\delta} \boldsymbol {h} ^ {\mathrm {T}} \right] - \lambda \boldsymbol {u} _ {1} \boldsymbol {v} _ {1} ^ {\mathrm {T}}\right) (12) \\ \end{array}
$$

where  $\delta \coloneqq \left(\frac{\partial V(G, D)}{\partial \left(\bar{W}_{\mathrm{SN}} h\right)}\right)^{\mathrm{T}}, \lambda \coloneqq \hat{\mathrm{E}}\left[\delta^{\mathrm{T}}\left(\bar{W}_{\mathrm{SN}} h\right)\right]$ , and  $\hat{\mathrm{E}}[\cdot]$  represents empirical expectation over the mini-batch.  $\frac{\partial V}{\partial W} = 0$  when  $\hat{\mathrm{E}}[\delta h^{\mathrm{T}}] = k u_1 v_1^T$  for some  $k \in \mathbb{R}$ .

We would like to comment on the implication of (12). The first term  $\hat{\mathrm{E}}\left[\delta h^{\mathrm{T}}\right]$  is the same as the derivative of the weights without normalization. In this light, the second term in the expression can be seen as the regularization term penalizing the first singular components with the adaptive regularization coefficient  $\lambda$ .  $\lambda$  is positive when  $\delta$  and  $\bar{W}_{\mathrm{SN}}h$  are pointing in similar direction, and this prevents the column space of  $W$  from concentrating into one particular direction in the course of the training. In other words, spectral normalization prevents the transformation of each layer from becoming to sensitive in one direction. We can also use spectral normalization to devise a new parametrization for the model. Namely, we can split the layer map into two separate trainable components: spectrally normalized map and the spectral norm constant. As it turns out, this parametrization has its merit on its own and promotes the performance of GANs (See Appendix E).

# 3 SPECTRAL NORMALIZATION VS OTHER REGULARIZATION TECHNIQUES

The weight normalization introduced by Salimans & Kingma (2016) is a method that normalizes the  $\ell_2$  norm of each row vector in the weight matrix. Mathematically, this is equivalent to requiring the weight by the weight normalization  $\bar{W}_{\mathrm{WN}}$ :

$$
\sigma_ {1} \left(\bar {W} _ {\mathrm {W N}}\right) ^ {2} + \sigma_ {2} \left(\bar {W} _ {\mathrm {W N}}\right) ^ {2} + \dots + \sigma_ {T} \left(\bar {W} _ {\mathrm {W N}}\right) ^ {2} = d _ {o}, \text {w h e r e} T = \min  \left(d _ {i}, d _ {o}\right), \tag {13}
$$

where  $\sigma_t(A)$  is a  $t$ -th singular value of matrix  $A$ . Therefore, up to a scalar, this is same as the Frobenius normalization, which requires the sum of the squared singular values to be 1. These normalizations, however, inadvertently impose much stronger constraints on the matrix than intended. If  $\bar{W}_{\mathrm{WN}}$  is the weight normalized matrix of dimension  $d_i \times d_o$ , the norm  $\| \bar{W}_{\mathrm{WN}} h \|_2$  for a fixed unit vector  $h$  is maximized at  $\| \bar{W}_{\mathrm{WN}} h \|_2 = \sqrt{d_o}$  when  $\sigma_1(\bar{W}_{\mathrm{WN}}) = \sqrt{d_o}$  and  $\sigma_t(\bar{W}_{\mathrm{WN}}) = 0$  for  $t = 2, \ldots, T$ , which means that  $\bar{W}_{\mathrm{WN}}$  is of rank one. Similar things can be said to the Frobenius normalization (See the appendix for more details). Using such  $\bar{W}_{\mathrm{WN}}$  corresponds to using only one feature to discriminate the model probability distribution from the target. In order to retain as much norm of the input as possible and hence to make the discriminator more sensitive, one would hope to make the norm of  $\bar{W}_{\mathrm{WN}} h$  large. For weight normalization, however, this comes at the cost of reducing the rank and hence the number of features to be used for the discriminator. Thus, there is a conflict of interests between weight normalization and our desire to use as many features as possible to distinguish the generator distribution from the target distribution. The former interest often reigns over the other in many cases, inadvertently diminishing the number of features to be used by the discriminators. Consequently, the algorithm would produce a rather arbitrary model distribution that matches the target distribution only at select few features. Weight clipping (Arjovsky et al., 2017) also suffers from same pitfall.

Our spectral normalization, on the other hand, do not suffer from such a conflict in interest. Note that the Lipschitz constant of a linear operator is determined only by the maximum singular value. In other words, the spectral norm is independent of rank. Thus, unlike the weight normalization, our spectral normalization allows the parameter matrix to use as many features as possible while satisfying local 1-Lipschitz constraint. Our spectral normalization leaves more freedom in choosing the number of singular components (features) to feed to the next layer of the discriminator.

Gulrajani et al. (2017) used Gradient penalty method in combination with WGAN. In their work, they placed  $K$ -Lipschitz constant on the discriminator by augmenting the objective function with the regularizer that rewards the function for having local 1-Lipschitz constant (i.e.  $\| \nabla_{\hat{x}}f\| _2 = 1$ ) at discrete sets of points of the form  $\hat{\pmb{x}}\coloneqq \epsilon \tilde{\pmb{x}} +(1 - \epsilon)\pmb {x}$  generated by interpolating a sample  $\tilde{\pmb{x}}$  from generative distribution and a sample  $\pmb{x}$  from the data distribution. While this rather straightforward approach does not suffer from the problems we mentioned above regarding the effective dimension of the feature space, the approach has an obvious weakness of being heavily dependent on the support of the current generative distribution. As a matter of course, the generative distribution and its support gradually changes in the course of the training, and this can destabilize the effect of such regularization. In fact, we empirically observed that a high learning rate can destabilize the performance of WGAN-GP. On the contrary, our spectral normalization regularizes the function the operator space, and the effect of the regularization is more stable with respect to the choice of the batch. Training with our spectral normalization does not easily destabilize with aggressive learning rate. Moreover, WGAN-GP requires more computational cost than our spectral normalization with single-step power iteration, because the computation of  $\| \nabla_{\hat{x}}f\| _2$  requires one whole round of forward and backward propagation. In the appendix section, we compare the computational cost of the two methods for the same number of updates.

# 4 EXPERIMENTS

In order to evaluate the efficacy of our experiment and investigate the reason behind its efficacy, we conducted a set of extensive experiments of unsupervised image generation on CIFAR-10 (Torralba et al., 2008) and STL-10 (Coates et al., 2011), and compared our method against other normalization techniques. To see how our method fares against large dataset, we also applied our method on ILSVRC2012 dataset (ImageNet) (Russakovsky et al., 2015) as well. This section is structured as follows. First, we will discuss the objective functions we used to train the architecture, and then we will describe the optimization settings we used in the experiments. We will then explain two performance measures on the images to evaluate the images produced by the trained generators. Finally, we will summarize our results on the CIFAR-10, STL-10, and on ImageNet.

As for the architecture of the discriminator and generator, we used convolutional neural networks. Also, for the evaluation of the spectral norm for the convolutional weight  $W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}} \times h \times w}$ , we treated the operator as a square matrix of dimension  $d_{\mathrm{out}} \times (d_{\mathrm{in}}hw)^2$ . We trained the parameters of the generator with batch normalization (Ioffe & Szegedy, 2015). We refer the readers to Table 3 in the appendix section for more details of the architectures.

For all methods other than WGAN-GP, we used the following standard objective function for the adversarial loss:

$$
V (G, D) := \underset {x \sim q _ {\text {d a t a}}} {\operatorname {E}} [ \log D (\boldsymbol {x}) ] + \underset {\boldsymbol {z} \sim p (\boldsymbol {z})} {\operatorname {E}} [ \log (1 - D (G (\boldsymbol {z}))) ], \tag {14}
$$

where  $z \in \mathbb{R}^{d_z}$  is a latent variable,  $p(z)$  is the standard normal distribution  $\mathcal{N}(0,I)$ , and  $G: \mathbb{R}^{d_z} \to \mathbb{R}^{d_0}$  is a deterministic generator function. We set  $d_z$  to 128 for all of our experiments. For the updates of  $G$ , we used the alternate cost proposed by Goodfellow et al. (2014)  $-\mathrm{E}_{z \sim p(z)}[\log(D(G(z)))]$  as used in Goodfellow et al. (2014) and Warde-Farley & Bengio (2017). For the updates of  $D$ , we used the original cost defined in (14). We also tested the performance of the algorithm with the so-called hinge loss, which is given by

$$
V _ {D} (\hat {G}, D) = \underset {\boldsymbol {x} \sim q _ {\mathrm {d a t a}}} {\operatorname {E}} [ \min  (0, - 1 + D (\boldsymbol {x})) ] + \underset {\boldsymbol {z} \sim p (\boldsymbol {z})} {\operatorname {E}} [ \min  (0, - 1 - D (\hat {G} (\boldsymbol {z})) ] \tag {15}
$$

$$
V _ {G} (G, \hat {D}) = - \underset {\boldsymbol {z} \sim p (\boldsymbol {z})} {\operatorname {E}} \left[ \hat {D} (G (\boldsymbol {z})) \right] \tag {16}
$$

respectively for the discriminator and for the generator. Optimizing these objectives is equivalent to minimizing the so-called reverse KL divergence  $\mathrm{KL}[p_g||q_{\mathrm{data}}]$ . This types of loss has been already proposed and used in (Lim & Ye, 2017; Tran et al., 2017). The algorithm based on the hinge loss also showed good performance when evaluated with inception score and FID. For Wasserstein GANs with gradient penalty (WGAN-GP) (Gulrajani et al., 2017), we used the following objective function:  $V(G,D)\coloneqq \operatorname{E}_{\boldsymbol{x}\sim q_{\mathrm{data}}}[D(\boldsymbol {x})] - \operatorname{E}_{\boldsymbol{z}\sim p(\boldsymbol {z})}[D(G(\boldsymbol {z}))] - \lambda \operatorname{E}_{\hat{\boldsymbol{x}}\sim p_{\hat{\boldsymbol{x}}}}[(\| \nabla_{\hat{\boldsymbol{x}}}D(\hat{\boldsymbol{x}})\|_2 - 1)^2 ]$ , where the regularization term is the one we introduced in the appendix section D.4.

For quantitative assessment of generated examples, we used inception score (Salimans et al., 2016) and Fréchet inception distance (FID) Heusel et al. (2017). Please see Appendix B.1 for the details of each score.

# 4.1 RESULTS ON CIFAR10 AND STL-10

In this section, we report the accuracy of the spectral normalization (we use the abbreviation: SNGAN for the spectrally normalized GANs) during the training, and the dependence of the algorithm's performance on the hyperparameters of the optimizer. We also compare the performance quality of the algorithm against those of other regularization/normalization techniques for the discriminator networks, including: WGAN-GP (Gulrajani et al., 2017), batch-normalization (BN) (Ioffe & Szegedy, 2015), layer normalization (LN) (Ba et al., 2016) and weight normalization (WN) (Salimans & Kingma, 2016). In order to evaluate the stand-alone efficacy of the gradient penalty, we also applied the penalty term (28) to the standard adversarial loss of GANs (14). We would refer to this

Table 1: Hyper-parameter settings we tested in our experiments.  $\dagger$ ,  $\ddagger$  and  $\star$  are the hyperparameter settings following Gulrajani et al. (2017), Warde-Farley & Bengio (2017) and Radford et al. (2016), respectively.  

<table><tr><td>Setting</td><td>α</td><td>β1</td><td>β2</td><td>ndis</td></tr><tr><td>A†</td><td>0.0001</td><td>0.5</td><td>0.9</td><td>5</td></tr><tr><td>B‡</td><td>0.0001</td><td>0.5</td><td>0.999</td><td>1</td></tr><tr><td>C*</td><td>0.0002</td><td>0.5</td><td>0.999</td><td>1</td></tr><tr><td>D</td><td>0.001</td><td>0.5</td><td>0.9</td><td>5</td></tr><tr><td>E</td><td>0.001</td><td>0.5</td><td>0.999</td><td>5</td></tr><tr><td>F</td><td>0.001</td><td>0.9</td><td>0.999</td><td>5</td></tr></table>

![](images/5016fb577b206682cf6465e8339a5161d62ee10889d2bc8619d961a2d77967d9.jpg)  
(a) CIFAR-10

![](images/2b0e5dfd8fb83d12181b1249266adc937f35486a01813e82b96fc6be304f069c.jpg)  
(b) STL-10  
Figure 1: Inception scores on CIFAR-10 and STL-10 with different methods and hyperparameters (higher is better).

method as 'GAN-GP'. For each method with gradient penalty, we set  $\lambda$  to 10, as suggested in Gulrajani et al. (2017). For all comparative studies throughout, we excluded the multiplier parameter  $\gamma$  in the weight normalization method, as well as in batch normalization and layer normalization method. This was done in order to prevent the methods from overtly violating the Lipschitz condition. When we experimented with the multiplier parameter, we were in fact not able to achieve any improvement.

For optimization, we used the Adam optimizer Kingma & Ba (2015) in all of our experiments. We tested with 6 settings for (1)  $n_{\mathrm{dis}}$ , the number of updates of the discriminator per one update of the generator and (2)  $\beta_1, \beta_2$ , the first and second order momentum of the hyper-parameters on Adam (the learning rate  $\alpha$ ). We list the details of these settings in Table 1 in the appendix section. Out of these 6 settings, A, B, and C are the settings used in previous representative works. The purpose of the settings D, E, and F is to evaluate the performance of the algorithms implemented with more aggressive learning rates. For the details of the architectures of convolutional networks deployed in the generator and the discriminator, we refer the readers to Table 3 in the appendix section. Number of updates for GAN generator were 100K for all experiments, unless otherwise noted.

Firstly, we inspected the spectral norms of each layer during the training to make sure that our spectral normalization procedure is indeed serving its purpose. As we can see in the Figure 8 in the C.1, the spectral norms of these layers floats around 1-1.05 region throughout the training. Please see Appendix C.1 for more details.

In Figures 1 and 2 we show the inception scores of each method with the settings A-F. We can see that spectral normalization is relatively robust to aggressive learning rates and momentum parameters. WGAN-GP fails to train good GANs at high learning rates and high momentum parameters on both CIFAR-10 and STL-10. Weight normalization is more robust than WGAN-GP on CIFAR-10 in this aspect. The optimal performance of weight normalization was inferior to both WGAN-GP and spectral normalization on STL-10, which consists of more diverse examples than CIFAR-10. Best scores of spectral normalization are better than all other methods on both CIFAR-10 and STL-10.

![](images/2b538e420027e4d20e73ef25a84a5e39d9a7b9207f3bf5521e9c99e6e3eda2f7.jpg)  
(a) CIFAR-10

![](images/3f7ddf8c00ca20515709e5d0c96e7699b2e190b8f9065be75b445cfe1e81a33d.jpg)  
(b) STL-10  
Figure 2: FIDs on CIFAR-10 and STL-10 with different methods and hyperparameters (lower is better).

Table 2: Inception scores and FIDs with unsupervised image generation on CIFAR-10. † (Radford et al., 2016) (experimented by Yang et al. (2017)), ‡ (Yang et al., 2017), * (Warde-Farley & Bengio, 2017), †† (Gulrajani et al., 2017)  

<table><tr><td rowspan="2">Method</td><td colspan="2">Inception score</td><td colspan="2">FID</td></tr><tr><td>CIFAR-10</td><td>STL-10</td><td>CIFAR-10</td><td>STL-10</td></tr><tr><td>(Real data)</td><td>11.24±.12</td><td>26.08±.26</td><td>7.8</td><td>7.9</td></tr><tr><td>(Standard CNN)</td><td></td><td></td><td></td><td></td></tr><tr><td>GAN-GP</td><td>6.93±.08</td><td></td><td>37.7</td><td></td></tr><tr><td>WGAN-GP</td><td>6.68±.06</td><td>8.42±.13</td><td>40.2</td><td>55.1</td></tr><tr><td>Batch Norm.</td><td>6.27±.10</td><td></td><td>56.3</td><td></td></tr><tr><td>Layer Norm.</td><td>7.19±.12</td><td>7.61±.12</td><td>33.9</td><td>75.6</td></tr><tr><td>Weight Norm.</td><td>6.84±.07</td><td>7.16±.10</td><td>34.7</td><td>73.4</td></tr><tr><td>(ours) SN-GANs</td><td>7.42±.08</td><td>8.28±.09</td><td>29.3</td><td>53.1</td></tr><tr><td>(ours) SN-GANs (2x updates)</td><td></td><td>8.69±.09</td><td></td><td>47.5</td></tr><tr><td>(ours) SN-GANs, Eq.(16)</td><td>7.58±.12</td><td></td><td>25.5</td><td></td></tr><tr><td>(ours) SN-GANs, Eq.(16) (2x updates)</td><td></td><td>8.79±.14</td><td></td><td>43.2</td></tr><tr><td>(ours) SN-GANs, Eq.(16) (ResNet)</td><td>8.24±.08</td><td>9.04±.12</td><td>17.5</td><td>38.3</td></tr><tr><td>DCGAN†</td><td>6.64±.14</td><td>7.84±.07</td><td></td><td></td></tr><tr><td>LR-GANs‡</td><td>7.17±.07</td><td></td><td></td><td></td></tr><tr><td>Warde-Farley et al.*</td><td>7.72±.13</td><td>8.51±.13</td><td></td><td></td></tr><tr><td>WGAN-GP (ResNet)††</td><td>7.86±.08</td><td></td><td></td><td></td></tr></table>

In Tables 2, we show the inception scores of the different methods with optimal settings on CIFAR-10 and STL-10 dataset. We see that SN-GANs performed better than all contemporaries on the optimal settings<sup>3</sup>. SN-GANs performed even better with hinge loss (16).

In Figure 5 we show the images produced by the generators trained with WGAN-GP, weight normalization, and spectral normalization. SN-GANs were consistently better than GANs with weight normalization in terms of the quality of generated images. To be more precise, as we mentioned in Section 3, the set of images generated by spectral normalization was clearer and more diverse than the images produced by the weight normalization. We can also see that WGAN-GP failed to train good GANs with high learning rates and high momentum (D,E and F). The generated images with GAN-GP, batch normalization and layer normalization is shown in Figure 11 in the appendix section.

We compared our algorithm against multiple benchmark methods in Table 2. We also tested the performance of our method on ResNet based GANs used in Gulrajani et al. (2017). Please see Table 4 and 5 in the appendix section for the detail network architectures. Please note that all methods listed thereof are all different in both optimization methods and the architecture of the model. Our implementation of our algorithm was able to superior to all the predecessors in the performance.

# 4.1.1 ANALYSIS OF SN-GANS

Singular values analysis on the weights of the discriminator  $D$  In Figure 4, we show the squared singular values of the weight matrices in the final discriminator  $D$  produced by each method using the parameter that yielded the best inception score. As we prophesied in Section 3, the singular values of the first to fifth layers trained with the weight normalization concentrate on a few components. That is, the weight matrices of these layers tend to be rank deficit. On the other hand, the singular values of the weight matrices in those layers trained with spectral normalization is more broadly distributed. When the goal is to distinguish a pair of probability distributions on the low-dimensional nonlinear data manifold embedded in a high dimensional space, rank deficiencies in lower layers can be especially fatal. Outputs of lower layers have gone through only few sets of rectified linear transformations, which means that they tend to lie on the space that is linear in most parts. Marginalizing out many features of the input distribution in such space can result in oversimplified discriminator. We can actually confirm the effect of this phenomenon on the generated images especially in Figure 5b. The images generated with spectral normalization is more diverse and complex than those generated with weight normalization.

Training time On CIFAR-10, SN-GANs is a tad slower than weight normalization (about  $110 \sim 120\%$  computational time), but significantly faster than WGAN-GP. As we mentioned in Section 3, WGAN-GP is slower than other methods because WGAN-GP needs to calculate the gradient of gradient norm  $\| \nabla_{\pmb{x}}D\| _2$ . For STL-10, the computational time of SN-GANs is almost the same as vanilla GANs, because the relative computational cost of the power iteration (17) is negligible when compared to the cost of forward and backward propagation on CIFAR-10 (images size of STL-10 is larger  $(48 \times 48)$ ). Please see Figure 9 in the appendix section for the actual computational time.

# 4.2 CLASS CONDITIONAL IMAGE GENERATION ON IMAGENET

![](images/025e2009e27d5d3fdf0afba1fe453680d7c889552eaff6a54ab4e306dbca981d.jpg)  
Figure 3: Learning curves of Inception score with different methods.

To show that our method remains effective on large high dimensional dataset, we also applied our method to the training of class conditional GANs on ILRSVRC2012 dataset with 1000 classes, each consisting of approximately 1300 images, which we compressed to  $128 \times 128$  pixels. Regarding the adversarial loss for conditional GANs, we used practically the same formulation used in Mirza & Osindero (2014), except that we replaced the standard GANs loss with hinge loss (16). Please see Appendix B.3 for the details of experimental settings.

As we can see in the learning curves in Figure 3, our SN-GANs are the only methods with successful training sequence among all other methods. To our knowledge, our method is the first of its kind in succeeding to produce decent images from ImageNet dataset with a single pair of a discriminator and a generator (Figure 6). To measure the degree of mode-collapse, we followed the footstep of Odena et al. (2017) and computed the intra MS-SSIM Odena et al. (2017) for pairs of independently generated GANs images of each class. We see that our SN-GAN ((intra MS-SSIM)=0.101) is

suffering less from the mode-collapse than AC-GANs ((intra MS-SSIM)~0.25).

# 5 CONCLUSION

This paper proposes spectral normalization as a stabilizer of training of GANs. When we apply spectral normalization to the GANs on image generation tasks, the generated examples are more diverse than the conventional weight normalization and achieve better or comparative inception scores relative to previous studies. The method imposes global regularization on the discriminator as opposed to local regularization introduced by WGAN-GP, and can possibly used in combinations. In the future work, we would like to further investigate where our methods stand amongst other methods on more theoretical basis, and experiment our algorithm on larger and more complex datasets.

# REFERENCES

Martin Arjovsky and Léon Bottou. Towards principled methods for training generative adversarial networks. In ICLR, 2017.  
Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In ICML, pp. 214-223, 2017.  
Devansh Arpit, Yingbo Zhou, Bhargava U Kota, and Venu Govindaraju. Normalization propagation: A parametric technique for removing internal covariate shift in deep networks. In ICML, pp. 1168-1176, 2016.  
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.  
Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In AISTATS, pp. 215-223, 2011.  
DC Dowson and BV Landau. The frechet distance between multivariate normal distributions. Journal of Multivariate Analysis, 12(3):450-455, 1982.  
Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Alex Lamb, Martin Arjovsky, Olivier Mastropietro, and Aaron Courville. Adversarily learned inference. In ICLR, 2017.  
Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In AISTATS, pp. 315-323, 2011.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NIPS, pp. 2672-2680, 2014.  
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein GANs. arXiv preprint arXiv:1704.00028, 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pp. 770-778, 2016.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Günter Klambauer, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a nash equilibrium. arXiv preprint arXiv:1706.08500, 2017.  
Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. In NIPS, pp. 4565-4573, 2016.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, pp. 448-456, 2015.  
Kevin Jarrett, Koray Kavukcuoglu, Marc'Aurelio Ranzato, and Yann LeCun. What is the best multi-stage architecture for object recognition? In ICCV, pp. 2146-2153, 2009.  
Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu. Improving training of deep neural networks via singular value bounding. In CVPR, 2017.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.  
Jiwei Li, Will Monroe, Tianlin Shi, Alan Ritter, and Dan Jurafsky. Adversarial learning for neural dialogue generation. In EMNLP, pp. 2147-2159, 2017.  
Jae Hyun Lim and Jong Chul Ye. Geometric GAN. arXiv preprint arXiv:1705.02894, 2017.  
Andrew L Maas, Awni Y Hannun, and Andrew Y Ng. Rectifier nonlinearities improve neural network acoustic models. In ICML Workshop on Deep Learning for Audio, Speech and Language Processing, 2013.  
Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014.  
Shakir Mohamed and Balaji Lakshminarayanan. Learning in implicit generative models. NIPS Workshop on Adversarial Training, 2017.  
Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In ICML, pp. 807-814, 2010.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-GAN: Training generative neural samplers using variational divergence minimization. In NIPS, pp. 271-279, 2016.  
Augustus Odena, Christopher Olah, and Jonathon Shlens. Conditional image synthesis with auxiliary classifier GANs. In ICML, pp. 2642-2651, 2017.  
Guo-Jun Qi. Loss-sensitive generative adversarial networks on lipschitz densities. arXiv preprint arXiv:1701.06264, 2017.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In ICLR, 2016.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211-252, 2015.  
Masaki Saito, Eiichi Matsumoto, and Shunta Saito. Temporal generative adversarial nets with singular value clipping. In ICCV, 2017.  
Tim Salimans and Diederik P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In NIPS, pp. 901-909, 2016.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training GANs. In NIPS, pp. 2226-2234, 2016.

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In CVPR, pp. 1-9, 2015.  
Antonio Torralba, Rob Fergus, and William T Freeman. 80 million tiny images: A large data set for nonparametric object and scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 30 (11):1958-1970, 2008.  
Dustin Tran, Rajesh Ranganath, and David M Blei. Deep and hierarchical implicit models. arXiv preprint arXiv:1702.08896, 2017.  
Masatoshi Uehara, Issei Sato, Masahiro Suzuki, Kotaro Nakayama, and Yutaka Matsuo. Generative adversarial nets from a density ratio estimation perspective. NIPS Workshop on Adversarial Training, 2016.  
David Warde-Farley and Yoshua Bengio. Improving generative adversarial networks with denoising feature matching. In ICLR, 2017.  
Sitao Xiang and Hao Li. On the effect of batch normalization and weight normalization in generative adversarial networks. arXiv preprint arXiv:1704.03971, 2017.  
Jianwei Yang, Anitha Kannan, Dhruv Batra, and Devi Parikh. LR-GAN: Layered recursive generative adversarial networks for image generation. ICLR, 2017.  
Yuichi Yoshida and Takeru Miyato. Spectral norm regularization for improving the generalizability of deep learning. arXiv preprint arXiv:1705.10941, 2017.

![](images/cc582ccccc1f0a1a6cdd46c12b23da8de89fcd86c214b1f02fa88accd29f5a66.jpg)

![](images/308973bb64922000b9dd4bd2847f34ee275fef846108229070af22ecace45cb6.jpg)  
(a) CIFAR-10  
(b) STL-10  
Figure 4: Squared singular values of weight matrices trained with different methods: Weight Normalization (WN) and Spectral Normalization (SN). We scaled the singular values so that the largest singular values is equal to 1. For WN and SN, we calculated singular values of the normalized weight matrices.

![](images/b7c8fd54991b8b4d0a6e6e68697a9f1042f4803a415c55c545f864250aee7f8a.jpg)  
Figure 5: Generated images on different methods: WGAN-GP, weight normalization, and spectral normalization on CIFAR-10 and STL-10.

![](images/17404665106ba755518c5c1a33225671ca11d21fe034440edd25be0ad4e305f5.jpg)  
Gray whale

![](images/df6d78b99aca4ca260372e239e37040d11cb5ebc6bdc2df337eeb6a42bdfc4c6.jpg)  
Tiger

![](images/e855a2a9bd0af06a1e701c117b0b851eeb80a99024083cde5f7471b1b5df1b85.jpg)  
Welsh springer spaniel

![](images/9bbcb605e2a17ed57b41b9f507a98d5e3a08012292f50521d79d56c83cdbb429.jpg)  
Chiffonier

![](images/24d148dc467d20b3769fab000c7b02d461c478bc21966b834485627fdc441c05.jpg)  
Mosque

![](images/783fa5f15791c9ec50b56d5ebffb56f0fe7501a926f990614e5dba8f3cb31074.jpg)  
Daisy  
Figure 6: 128x128 pixel images generated by SN-GANs trained on ILSVRC2012 dataset. The inception score is 21.9.

![](images/0bf218a7fe6b8061b3101e90c7fe4415cd5215e2f18679c86751fc0aca6b96fd.jpg)  
Palace

![](images/10c78ba040acc48d21514822221235202498667b581435b0bfea47fd202c91d4.jpg)  
Sandbar

![](images/b686c2daee04feef89aa039f38e390b02278d5033815c83d65ae77dae56f8b45.jpg)  
Presian cat

![](images/107734290dc072261b53fa60921ec1bc6d1bda035ed0b03a7d47a5c6f4976300.jpg)  
Fire truck

![](images/f829da69448afc48b0dffa705796db4a79c6f5493f4e15409d03c38eedd3fb3f.jpg)  
Schooner

![](images/6da9dd342dd9218c6428c009883c7a5cb264514bda9d62b87432539a7dbda20d.jpg)  
Pizza
